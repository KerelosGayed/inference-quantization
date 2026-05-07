"""
Streamed real-model quantization for Llama/Phi-style CausalLM checkpoints.

Pipeline per Linear layer:
  1) Move weight to GPU (one layer at a time)
  2) Block-diagonal Hadamard rotation (fused FWHT)
  3) Group-wise ternary + 1-bit QJL residual quantization (group_size=32)
  4) Pack ternary + QJL bits with cracked_quant kernels
  5) Move packed tensors back to CPU and write to disk immediately

This keeps VRAM usage low enough for RTX 3080 Ti (12 GB) workflows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    safe_open = None
    save_safetensors = None

from transformers import AutoModelForCausalLM


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


@dataclass
class LayerStats:
    name: str
    layer_type: str
    shape: Tuple[int, int]
    snr_db: float
    mse: float
    var: float
    original_bytes: int
    quantized_bytes: int


def fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024.0:
            return f"{f:8.2f} {unit}"
        f /= 1024.0
    return f"{f:8.2f} TiB"


def build_extension():
    print("[build] compiling cracked_quant extension...", flush=True)
    cq = load(
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
    print("[build] done.", flush=True)
    return cq


def hadamard_rotate(cq, w: torch.Tensor, block_size: int) -> torch.Tensor:
    out_features, in_features = w.shape
    if in_features % block_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by block_size={block_size}"
        )
    if block_size <= 0 or (block_size & (block_size - 1)) != 0 or block_size > 1024:
        raise ValueError("block_size must be a power of 2 in [1, 1024]")
    n_blocks = in_features // block_size
    w_blocks = w.view(out_features, n_blocks, block_size).contiguous()
    return cq.fwht_forward(w_blocks, True).view(out_features, in_features)


def ternary_quantize_groupwise(w_rot: torch.Tensor, group_size: int):
    out_features, in_features = w_rot.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )
    n_groups = in_features // group_size
    grouped = w_rot.view(out_features, n_groups, group_size)
    alpha = grouped.abs().mean(dim=-1).clamp_min(1e-8)
    w_q = (grouped / alpha.unsqueeze(-1)).round().clamp_(-1.0, 1.0).to(torch.float16)
    return w_q.view(out_features, in_features).contiguous(), alpha.to(torch.float16).contiguous()


def quantize_linear_layer(
    cq,
    name: str,
    w_cpu: torch.Tensor,
    out_dir: Path,
    group_size: int,
    block_size: int,
) -> LayerStats:
    if w_cpu.dim() != 2:
        raise ValueError(f"{name}: expected 2D linear weight, got shape {tuple(w_cpu.shape)}")

    out_features, in_features = w_cpu.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"{name}: in_features={in_features} must be divisible by group_size={group_size}"
        )
    if in_features % block_size != 0:
        raise ValueError(
            f"{name}: in_features={in_features} must be divisible by block_size={block_size}"
        )

    w = w_cpu.to(device="cuda", dtype=torch.float16, non_blocking=True).contiguous()
    w_rot = hadamard_rotate(cq, w, block_size=block_size)

    w_q, alpha = ternary_quantize_groupwise(w_rot, group_size=group_size)
    n_groups = in_features // group_size
    alpha_expand = alpha.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features)

    w_quantized = w_q * alpha_expand
    residual = (w_rot - w_quantized).contiguous()
    mae = residual.view(out_features, n_groups, group_size).abs().mean(dim=-1)
    mae = mae.clamp_min(1e-8).to(torch.float16).contiguous()

    packed_ternary = cq.ternary_pack(w_q, group_size)
    packed_qjl = cq.qjl_pack(residual, group_size)

    w_turbo = cq.turbo_unpack(
        packed_ternary,
        packed_qjl,
        alpha.contiguous(),
        mae.contiguous(),
        group_size,
    )

    mse = (w_rot.float() - w_turbo.float()).pow(2).mean().item()
    var = w_rot.float().var(unbiased=False).item()
    snr_db = 10.0 * math.log10(var / mse) if mse > 0 else float("inf")

    original_bytes = w.numel() * 2
    quantized_bytes = (
        packed_ternary.numel() * packed_ternary.element_size()
        + packed_qjl.numel() * packed_qjl.element_size()
        + alpha.numel() * alpha.element_size()
        + mae.numel() * mae.element_size()
    )

    layer_file = out_dir / f"{name.replace('.', '__')}.safetensors"
    tensors = {
        "packed_ternary": packed_ternary.cpu().contiguous(),
        "packed_qjl": packed_qjl.cpu().contiguous(),
        "alpha_groups": alpha.cpu().contiguous(),
        "mae_groups": mae.cpu().contiguous(),
        "shape": torch.tensor([out_features, in_features], dtype=torch.int32),
    }

    if save_safetensors is not None:
        save_safetensors(
            tensors,
            str(layer_file),
            metadata={
                "layer_name": name,
                "layout": "packed_ternary:uint8, packed_qjl:uint8, alpha_groups:fp16, mae_groups:fp16",
                "group_size": str(group_size),
                "block_size": str(block_size),
            },
        )
    else:
        torch.save(tensors, layer_file.with_suffix(".pt"))

    del w, w_rot, w_q, alpha, alpha_expand, w_quantized, residual, mae
    del packed_ternary, packed_qjl, w_turbo
    torch.cuda.empty_cache()

    return LayerStats(
        name=name,
        layer_type=name.split(".")[-1],
        shape=(out_features, in_features),
        snr_db=snr_db,
        mse=mse,
        var=var,
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
    )


def is_sensitive_layer(name: str) -> bool:
    lname = name.lower()
    return ("lm_head" in lname) or ("embed" in lname)


def _extract_block_index(name: str) -> int | None:
    # Supports common HF naming: model.layers.{i}.*
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is not None:
        return int(m.group(1))
    # Fallback for some GPT-like stacks using .h.{i}.*
    m = re.search(r"\.h\.(\d+)\.", name)
    if m is not None:
        return int(m.group(1))
    return None


def is_protected_block_layer(name: str, num_hidden_layers: int, protect_blocks: int) -> bool:
    if protect_blocks <= 0:
        return False
    idx = _extract_block_index(name)
    if idx is None:
        return False
    return idx < protect_blocks or idx >= max(0, num_hidden_layers - protect_blocks)


def has_valid_quant_file(layer_file: Path, expected_shape: Tuple[int, int]) -> bool:
    if not layer_file.exists():
        return False
    try:
        if layer_file.suffix == ".safetensors":
            if safe_open is None:
                return False
            with safe_open(str(layer_file), framework="pt", device="cpu") as f:
                keys = set(f.keys())
                required = {"packed_ternary", "packed_qjl", "alpha_groups", "mae_groups", "shape"}
                if not required.issubset(keys):
                    return False
                shape = f.get_tensor("shape").to(torch.int64).flatten()
                if shape.numel() != 2:
                    return False
                return (int(shape[0].item()), int(shape[1].item())) == expected_shape
        obj = torch.load(layer_file, map_location="cpu")
        if not isinstance(obj, dict):
            return False
        required = {"packed_ternary", "packed_qjl", "alpha_groups", "mae_groups", "shape"}
        if not required.issubset(obj.keys()):
            return False
        shape = obj["shape"].to(torch.int64).flatten()
        if shape.numel() != 2:
            return False
        return (int(shape[0].item()), int(shape[1].item())) == expected_shape
    except Exception:
        return False


def load_model_with_fallback(primary_id: str, fallback_id: str | None, token: str | None):
    model_ids: Iterable[str] = [primary_id] + ([fallback_id] if fallback_id else [])
    last_error: Exception | None = None
    for model_id in model_ids:
        if model_id is None:
            continue
        print(f"[model] loading {model_id} on CPU...", flush=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="cpu",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                token=token,
                trust_remote_code=True,
            )
            print(f"[model] loaded: {model_id}", flush=True)
            return model, model_id
        except Exception as exc:  # pragma: no cover - runtime fallback
            print(f"[model] failed to load {model_id}: {exc}", flush=True)
            last_error = exc
    raise RuntimeError("Unable to load any requested model IDs") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamed real-model TurboQuant pass.")
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--fallback-model-id", default="microsoft/phi-2")
    parser.add_argument("--output-dir", default="vacuum_sealed_model")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=0,
                        help="Optional cap for quick smoke tests (0 = all layers)")
    parser.add_argument("--protect-blocks", type=int, default=0,
                        help="Keep first/last K transformer blocks in FP16 (0 = disabled).")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--resume", action="store_true",
                        help="Skip layers that already have valid packed files on disk.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    cq = build_extension()
    model, loaded_id = load_model_with_fallback(args.model_id, args.fallback_model_id, args.hf_token)
    model.eval()
    num_hidden_layers = int(getattr(model.config, "num_hidden_layers", 0))

    layer_stats: list[LayerStats] = []
    processed = 0
    skipped = 0
    protected = 0
    protected_block_layers = 0
    resumed = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if args.max_layers > 0 and processed >= args.max_layers:
            break

        if is_sensitive_layer(name):
            print(f"[protect] keeping {name} in FP16/BF16 (sensitive layer)", flush=True)
            protected += 1
            continue
        if num_hidden_layers > 0 and is_protected_block_layer(name, num_hidden_layers, args.protect_blocks):
            print(f"[protect] keeping {name} in FP16 (protected transformer block window)", flush=True)
            protected += 1
            protected_block_layers += 1
            skipped += 1
            continue

        w = module.weight.detach()
        if w.dim() != 2:
            skipped += 1
            continue
        out_features, in_features = w.shape
        if in_features % args.group_size != 0 or in_features % args.block_size != 0:
            print(
                f"[skip] {name}: shape={tuple(w.shape)} not divisible by "
                f"group_size={args.group_size} and block_size={args.block_size}",
                flush=True,
            )
            skipped += 1
            continue

        layer_file = out_dir / f"{name.replace('.', '__')}.safetensors"
        if args.resume and has_valid_quant_file(layer_file, (out_features, in_features)):
            print(f"[resume] using existing packed layer for {name}", flush=True)
            resumed += 1
            continue

        print(f"[layer] quantizing {name} shape={tuple(w.shape)}", flush=True)
        stats = quantize_linear_layer(
            cq=cq,
            name=name,
            w_cpu=w.cpu(),
            out_dir=out_dir,
            group_size=args.group_size,
            block_size=args.block_size,
        )
        processed += 1
        layer_stats.append(stats)
        ratio = stats.original_bytes / max(stats.quantized_bytes, 1)
        print(
            f"        SNR={stats.snr_db:6.2f} dB  "
            f"ratio={ratio:5.2f}x  "
            f"orig={fmt_bytes(stats.original_bytes)} -> quant={fmt_bytes(stats.quantized_bytes)}",
            flush=True,
        )

    if not layer_stats:
        raise RuntimeError("No linear layers were quantized.")

    # Aggregate by major layer type: q_proj, gate_proj, o_proj, ...
    by_type: Dict[str, Dict[str, float]] = {}
    for s in layer_stats:
        bucket = by_type.setdefault(
            s.layer_type,
            {
                "count": 0.0,
                "signal_energy": 0.0,
                "noise_energy": 0.0,
                "original_bytes": 0.0,
                "quantized_bytes": 0.0,
            },
        )
        n = float(s.shape[0] * s.shape[1])
        bucket["count"] += 1.0
        bucket["signal_energy"] += s.var * n
        bucket["noise_energy"] += s.mse * n
        bucket["original_bytes"] += float(s.original_bytes)
        bucket["quantized_bytes"] += float(s.quantized_bytes)

    print("\n=== Per-type summary ===")
    summary_types = {}
    for layer_type, agg in sorted(by_type.items()):
        signal = agg["signal_energy"]
        noise = max(agg["noise_energy"], 1e-30)
        snr_db = 10.0 * math.log10(signal / noise)
        ratio = agg["original_bytes"] / max(agg["quantized_bytes"], 1.0)
        print(
            f"{layer_type:>16s} | layers={int(agg['count']):3d} | "
            f"SNR={snr_db:6.2f} dB | ratio={ratio:5.2f}x"
        )
        summary_types[layer_type] = {
            "layers": int(agg["count"]),
            "snr_db": snr_db,
            "compression_ratio": ratio,
            "original_bytes": int(agg["original_bytes"]),
            "quantized_bytes": int(agg["quantized_bytes"]),
        }

    total_original = int(sum(s.original_bytes for s in layer_stats))
    total_quantized = int(sum(s.quantized_bytes for s in layer_stats))
    total_ratio = total_original / max(total_quantized, 1)

    total_disk_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    elapsed = time.time() - start

    print("\n=== Final check ===")
    print(f"Model ID                    : {loaded_id}")
    print(f"Quantized linear layers     : {processed}")
    print(f"Skipped linear layers       : {skipped}")
    print(f"Protected sensitive layers  : {protected}")
    print(f"Protected block layers      : {protected_block_layers}")
    print(f"Resumed packed layers       : {resumed}")
    print(f"Original linear bytes       : {fmt_bytes(total_original)}")
    print(f"Quantized payload bytes     : {fmt_bytes(total_quantized)}")
    print(f"Compression ratio           : {total_ratio:.2f}x")
    print(f"Vacuum-sealed on-disk size  : {fmt_bytes(total_disk_bytes)}")
    print(f"Elapsed                     : {elapsed/60.0:.2f} min")

    report = {
        "model_id": loaded_id,
        "group_size": args.group_size,
        "block_size": args.block_size,
        "processed_layers": processed,
        "skipped_layers": skipped,
        "protected_layers": protected,
        "protected_block_layers": protected_block_layers,
        "resumed_layers": resumed,
        "original_linear_bytes": total_original,
        "quantized_payload_bytes": total_quantized,
        "compression_ratio": total_ratio,
        "on_disk_bytes": total_disk_bytes,
        "per_type": summary_types,
        "layers": [
            {
                "name": s.name,
                "type": s.layer_type,
                "shape": list(s.shape),
                "snr_db": s.snr_db,
                "mse": s.mse,
                "var": s.var,
                "original_bytes": s.original_bytes,
                "quantized_bytes": s.quantized_bytes,
            }
            for s in layer_stats
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
