"""
Inference engine for vacuum-sealed TurboQuant checkpoints.

Expected layer files (from quantize_real_llama.py):
  - packed_ternary : uint8  [out_features, in_features // 4]
  - packed_qjl     : uint8  [out_features, in_features // 8]
  - alpha_groups   : float16[out_features, in_features // group_size]
  - mae_groups     : float16[out_features, in_features // group_size]
  - shape          : int32  [2] = [out_features, in_features]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    safe_open = None
    load_safetensors = None


HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".torch_extensions"))

NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_86,code=sm_86",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--ptxas-options=-O3",
]

CQ = None


def build_extension():
    return load(
        name="cracked_quant",
        sources=[
            str(HERE / "fused_fwht.cu"),
            str(HERE / "ternary_packing.cu"),
            str(HERE / "cracked_quant.cpp"),
        ],
        extra_cuda_cflags=NVCC_FLAGS,
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def _has_tail_ngram_loop(token_ids: list[int], ngram_size: int, min_repeats: int) -> bool:
    if ngram_size <= 0 or min_repeats <= 1:
        return False
    needed = ngram_size * min_repeats
    if len(token_ids) < needed:
        return False
    tail = token_ids[-needed:]
    first = tail[:ngram_size]
    for i in range(1, min_repeats):
        start = i * ngram_size
        if tail[start:start + ngram_size] != first:
            return False
    return True


class TurboLinear(nn.Module):
    """
    nn.Linear replacement backed by packed TurboQuant tensors.
    """

    def __init__(
        self,
        packed_ternary: torch.Tensor,
        packed_qjl: torch.Tensor,
        alpha_groups: torch.Tensor,
        mae_groups: torch.Tensor,
        in_features: int,
        out_features: int,
        group_size: int,
        block_size: int,
        bias: Optional[torch.Tensor] = None,
        cache_weights: bool = False,
        inverse_mode: str = "normalized",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.block_size = block_size
        self.cache_weights = cache_weights
        self.inverse_mode = inverse_mode
        self._cached_weight: Optional[torch.Tensor] = None

        self.register_buffer("packed_ternary", packed_ternary.contiguous())
        self.register_buffer("packed_qjl", packed_qjl.contiguous())
        self.register_buffer("alpha_groups", alpha_groups.contiguous())
        self.register_buffer("mae_groups", mae_groups.contiguous())

        if bias is not None:
            self.bias = nn.Parameter(bias.contiguous(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if CQ is None:
            raise RuntimeError("cracked_quant extension was not initialized")

        if not x.is_cuda:
            raise RuntimeError("TurboLinear expects CUDA input tensors")

        if self.cache_weights and self._cached_weight is not None:
            w = self._cached_weight
        else:
            # Scale-first reconstruction:
            # W_fp16 = (W_ternary * alpha_group) + (sgn(R) * MAE_group)
            w = CQ.turbo_unpack(
                self.packed_ternary,
                self.packed_qjl,
                self.alpha_groups,
                self.mae_groups,
                self.group_size,
            )
            # Quantized weights are stored in rotated space. Recover original
            # structure via inverse block-diagonal Hadamard transform.
            # For normalized FWHT, inverse is normalized FWHT.
            # For unnormalized FWHT, inverse is FWHT / block_size.
            n_blocks = self.in_features // self.block_size
            w_blocks = w.view(self.out_features, n_blocks, self.block_size).contiguous()
            if self.inverse_mode == "divide_n":
                w = (CQ.fwht_forward(w_blocks, False) / float(self.block_size)).view(
                    self.out_features, self.in_features
                )
            else:
                w = CQ.fwht_inverse_block(w_blocks, self.block_size).view(
                    self.out_features, self.in_features
                )
            if self.cache_weights:
                self._cached_weight = w
        y = torch.matmul(x, w.transpose(0, 1))
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def materialize_cached_weight(self) -> None:
        if not self.cache_weights or self._cached_weight is not None:
            return
        dummy = torch.zeros((1, 1, self.in_features), dtype=torch.float16, device=self.packed_ternary.device)
        _ = self.forward(dummy)


def _split_parent(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _read_layer_file(layer_path: Path):
    if layer_path.suffix == ".safetensors":
        if load_safetensors is None:
            raise RuntimeError("safetensors is required to read .safetensors layer files")
        return load_safetensors(str(layer_path), device="cpu"), {}
    obj = torch.load(layer_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected tensor container type in {layer_path}")
    return obj, {}


def _read_group_size_from_safetensor(path: Path) -> Optional[int]:
    if safe_open is None or path.suffix != ".safetensors":
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata()
        if meta is None:
            return None
        gs = meta.get("group_size")
        return int(gs) if gs is not None else None
    except Exception:
        return None


def _read_block_size_from_safetensor(path: Path) -> Optional[int]:
    if safe_open is None or path.suffix != ".safetensors":
        return None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata()
        if meta is None:
            return None
        b = meta.get("block_size")
        return int(b) if b is not None else None
    except Exception:
        return None


def load_vacuum_model(
    model_path: str,
    model_id: Optional[str] = None,
    device: str = "cuda",
    cache_weights: bool = False,
    block_size: Optional[int] = None,
    inverse_mode: str = "normalized",
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load original architecture and replace quantized nn.Linear layers
    with TurboLinear modules from vacuum-sealed layer files.
    """
    vacuum_dir = Path(model_path)
    if not vacuum_dir.exists():
        raise FileNotFoundError(f"Vacuum model directory not found: {vacuum_dir}")

    report_path = vacuum_dir / "report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    if model_id is None:
        model_id = report.get("model_id", "microsoft/phi-2")
    default_group_size = int(report.get("group_size", 128))
    default_block_size = int(block_size if block_size is not None else report.get("block_size", 128))

    print(f"[load] architecture: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    replaced = 0
    missing = 0

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        base = name.replace(".", "__")
        st_path = vacuum_dir / f"{base}.safetensors"
        pt_path = vacuum_dir / f"{base}.pt"

        layer_file = st_path if st_path.exists() else (pt_path if pt_path.exists() else None)
        if layer_file is None:
            missing += 1
            continue

        tensors, _ = _read_layer_file(layer_file)
        required = {"packed_ternary", "packed_qjl", "alpha_groups", "mae_groups", "shape"}
        if not required.issubset(tensors.keys()):
            raise RuntimeError(f"{layer_file} is missing required tensors: {required - set(tensors.keys())}")

        shape = tensors["shape"].to(torch.int64).flatten()
        out_features = int(shape[0].item())
        in_features = int(shape[1].item())
        if module.weight.shape != (out_features, in_features):
            raise RuntimeError(
                f"{name}: architecture shape {tuple(module.weight.shape)} does not match packed shape "
                f"{(out_features, in_features)}"
            )

        layer_group_size = _read_group_size_from_safetensor(layer_file) or default_group_size
        layer_block_size = _read_block_size_from_safetensor(layer_file) or default_block_size
        if in_features % layer_block_size != 0:
            raise RuntimeError(
                f"{name}: in_features={in_features} is not divisible by block_size={layer_block_size}"
            )
        bias = module.bias.detach().to(torch.float16).cpu() if module.bias is not None else None

        turbo = TurboLinear(
            packed_ternary=tensors["packed_ternary"].to(torch.uint8),
            packed_qjl=tensors["packed_qjl"].to(torch.uint8),
            alpha_groups=tensors["alpha_groups"].to(torch.float16),
            mae_groups=tensors["mae_groups"].to(torch.float16),
            in_features=in_features,
            out_features=out_features,
            group_size=layer_group_size,
            block_size=layer_block_size,
            bias=bias,
            cache_weights=cache_weights,
            inverse_mode=inverse_mode,
        )

        parent, child_name = _split_parent(model, name)
        setattr(parent, child_name, turbo)
        replaced += 1

    print(f"[load] replaced TurboLinear layers: {replaced}")
    print(f"[load] missing packed layers (kept fp16): {missing}")

    model.to(device)
    model.eval()

    if cache_weights:
        print("[cache] materializing TurboLinear weights once at startup...")
        cached = 0
        for mod in model.modules():
            if isinstance(mod, TurboLinear):
                mod.materialize_cached_weight()
                cached += 1
        print(f"[cache] done. materialized layers: {cached}")

    return model, tokenizer


@torch.no_grad()
def generate_with_tps(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    ngram_stop_size: int = 3,
    ngram_stop_repeats: int = 3,
) -> Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    else:
        attention_mask = torch.ones_like(input_ids, device="cuda")

    # Warm-up sync for fair TPS timing.
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generated_tokens = []
    generated_ids = torch.empty((1, 0), dtype=torch.long, device="cuda")
    cur_input_ids = input_ids
    cur_attention_mask = attention_mask
    past_key_values = None

    # Step-by-step sampling gives us explicit control of repetition penalties
    # over only the generated response region (not the original prompt).
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=cur_input_ids,
            attention_mask=cur_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()

        if generated_ids.numel() > 0 and repetition_penalty > 1.0:
            unique_ids = torch.unique(generated_ids[0])
            logits[:, unique_ids] = logits[:, unique_ids] / repetition_penalty

        if temperature <= 0.0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[..., 1:] = remove_mask[..., :-1].clone()
            remove_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = int(next_token.item())
        generated_tokens.append(token_id)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        if _has_tail_ngram_loop(generated_tokens, ngram_stop_size, ngram_stop_repeats):
            break

        cur_input_ids = next_token
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones((1, 1), dtype=cur_attention_mask.dtype, device="cuda")],
            dim=1,
        )

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new_tokens = len(generated_tokens)
    tps = float(new_tokens) / max(dt, 1e-9)

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    prompt_clean = prompt.strip()
    if prompt_clean:
        # Trim prompt echo at the beginning of generated text.
        while text.lower().startswith(prompt_clean.lower()):
            text = text[len(prompt_clean):].lstrip()
    return text, tps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on vacuum-sealed TurboQuant model.")
    parser.add_argument("--model-path", default="vacuum_sealed_model")
    parser.add_argument("--model-id", default=None,
                        help="Override HF architecture ID. If omitted, uses report.json model_id.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--ngram-stop-size", type=int, default=3,
                        help="Tail n-gram size for hard-stop loop detector (<=0 disables).")
    parser.add_argument("--ngram-stop-repeats", type=int, default=3,
                        help="Consecutive repeats needed to trigger n-gram hard-stop.")
    parser.add_argument("--cache-weights", action="store_true",
                        help="Cache unpacked FP16 TurboLinear weights in VRAM.")
    parser.add_argument("--block-size", type=int, default=None,
                        help="Override inverse-Hadamard block size (default: metadata/report, else 128).")
    parser.add_argument(
        "--inverse-mode",
        choices=["normalized", "divide_n"],
        default="normalized",
        help="Inverse-Hadamard mode. Use 'normalized' for FWHT(..., normalize=True) quantization.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TurboLinear inference.")

    global CQ
    print("[build] loading cracked_quant extension...")
    CQ = build_extension()
    print("[build] ready.")

    model, tokenizer = load_vacuum_model(
        args.model_path,
        model_id=args.model_id,
        device="cuda",
        cache_weights=args.cache_weights,
        block_size=args.block_size,
        inverse_mode=args.inverse_mode,
    )

    print("\nTurbo chat ready. Type 'exit' to quit.\n")
    while True:
        prompt = input("You> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        response, tps = generate_with_tps(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            ngram_stop_size=args.ngram_stop_size,
            ngram_stop_repeats=args.ngram_stop_repeats,
        )
        print(f"\nModel> {response}\n")
        print(f"[perf] TPS: {tps:.2f} tokens/s\n")


if __name__ == "__main__":
    main()
