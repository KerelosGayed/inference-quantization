"""
Numerical integrity debug harness for TurboLinear.

What it checks:
  1) Output parity between original nn.Linear and TurboLinear on same input X
  2) MSE and cosine similarity of outputs
  3) First 10 output values side-by-side
  4) Broadcast/group mapping audit for alpha/MAE scaling
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import inference as inf


HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(HERE / ".torch_extensions"))


def find_linear(model: nn.Module, target_name: Optional[str]) -> tuple[str, nn.Linear]:
    if target_name:
        mod = dict(model.named_modules()).get(target_name)
        if mod is None:
            raise ValueError(f"Layer '{target_name}' not found in model.")
        if not isinstance(mod, nn.Linear):
            raise ValueError(f"Layer '{target_name}' exists but is not nn.Linear.")
        return target_name, mod

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            return name, mod
    raise RuntimeError("No nn.Linear layer found in model.")


def quantize_groupwise(w_fp16: torch.Tensor, group_size: int):
    out_features, in_features = w_fp16.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size}"
        )
    n_groups = in_features // group_size
    grouped = w_fp16.view(out_features, n_groups, group_size)
    alpha = grouped.abs().mean(dim=-1).clamp_min(1e-8)
    w_codes = (grouped / alpha.unsqueeze(-1)).round().clamp_(-1.0, 1.0).to(torch.float16)
    w_q = w_codes.view(out_features, in_features).contiguous()
    alpha = alpha.to(torch.float16).contiguous()
    alpha_expand = alpha.unsqueeze(-1).expand(-1, -1, group_size).reshape(out_features, in_features)

    residual = (w_fp16 - (w_q * alpha_expand)).contiguous()
    mae = residual.view(out_features, n_groups, group_size).abs().mean(dim=-1)
    mae = mae.clamp_min(1e-8).to(torch.float16).contiguous()
    return w_q, alpha, mae, residual


def hadamard_rotate(cq, w: torch.Tensor, block_size: int) -> torch.Tensor:
    out_features, in_features = w.shape
    if in_features % block_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by block_size={block_size}"
        )
    n_blocks = in_features // block_size
    w_blocks = w.view(out_features, n_blocks, block_size).contiguous()
    return cq.fwht_forward(w_blocks, True).view(out_features, in_features)


def unpack_qjl_signs(packed_qjl: torch.Tensor, in_features: int) -> torch.Tensor:
    # packed_qjl: [out_features, in_features // 8], uint8
    out_features = packed_qjl.size(0)
    cols = torch.arange(in_features, device=packed_qjl.device, dtype=torch.int64)
    byte_idx = cols // 8
    bit_idx = cols % 8
    bytes_per_col = packed_qjl[:, byte_idx]  # [out_features, in_features]
    bits = ((bytes_per_col >> bit_idx.unsqueeze(0)) & 0x1).to(torch.float16)
    signs = bits * 2.0 - 1.0
    return signs.view(out_features, in_features)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Debug TurboLinear numerical integrity.")
    parser.add_argument("--model-id", default="microsoft/phi-2")
    parser.add_argument("--layer-name", default=None,
                        help="Exact nn.Linear module name. Defaults to first linear layer.")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--inverse-mode",
        choices=["normalized", "divide_n"],
        default="normalized",
        help="Must match inference inverse mode for fair parity checks.",
    )
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for debug_layers.py")

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    layer_name, layer = find_linear(model, args.layer_name)
    print(f"[layer] Using: {layer_name} shape={tuple(layer.weight.shape)}")

    in_features = layer.in_features
    out_features = layer.out_features
    if in_features % args.group_size != 0:
        raise ValueError(
            f"Selected layer in_features={in_features} is not divisible by group_size={args.group_size}"
        )
    if in_features % args.block_size != 0:
        raise ValueError(
            f"Selected layer in_features={in_features} is not divisible by block_size={args.block_size}"
        )

    # Build cracked_quant and wire the same TurboLinear used by inference.py.
    inf.CQ = inf.build_extension()

    weight = layer.weight.detach().to("cuda", dtype=torch.float16).contiguous()
    bias = layer.bias.detach().to("cuda", dtype=torch.float16).contiguous() if layer.bias is not None else None

    weight_rot = hadamard_rotate(inf.CQ, weight, args.block_size)
    w_q, alpha, mae, residual = quantize_groupwise(weight_rot, args.group_size)
    packed_ternary = inf.CQ.ternary_pack(w_q, args.group_size)
    packed_qjl = inf.CQ.qjl_pack(residual, args.group_size)

    turbo = inf.TurboLinear(
        packed_ternary=packed_ternary,
        packed_qjl=packed_qjl,
        alpha_groups=alpha,
        mae_groups=mae,
        in_features=in_features,
        out_features=out_features,
        group_size=args.group_size,
        block_size=args.block_size,
        bias=bias,
        cache_weights=False,
        inverse_mode=args.inverse_mode,
    ).to("cuda")
    ref_layer = nn.Linear(in_features, out_features, bias=(bias is not None), dtype=torch.float16).to("cuda")
    ref_layer.weight.data.copy_(weight)
    if bias is not None:
        ref_layer.bias.data.copy_(bias)
    print(f"[bias] comparison includes bias: {bias is not None}")

    x = torch.randn(args.batch, args.seq, in_features, device="cuda", dtype=torch.float16)
    y_ref = ref_layer(x)
    y_turbo = turbo(x)

    mse = (y_ref.float() - y_turbo.float()).pow(2).mean().item()
    cos = torch.nn.functional.cosine_similarity(
        y_ref.float().flatten(), y_turbo.float().flatten(), dim=0
    ).item()
    print(f"[metrics] Output MSE           : {mse:.6e}")
    print(f"[metrics] Output Cosine Sim    : {cos:.8f}")

    ref_flat = y_ref.flatten().float()
    turbo_flat = y_turbo.flatten().float()
    print("\n[first 10 output values]")
    print(" idx | ref_output        | turbo_output")
    print("-----+-------------------+-------------------")
    for i in range(min(10, ref_flat.numel())):
        print(f"{i:4d} | {ref_flat[i].item(): .8f} | {turbo_flat[i].item(): .8f}")

    # Broadcast / group audit in rotated space:
    # manual_w = (ternary_code * alpha_group) + (sign * mae_group)
    w_codes = inf.CQ.ternary_unpack(packed_ternary, 1.0)  # {-1,0,+1}
    w_unpack = inf.CQ.turbo_unpack(packed_ternary, packed_qjl, alpha, mae, args.group_size)
    n_groups = in_features // args.group_size
    alpha_expand = alpha.unsqueeze(-1).expand(-1, -1, args.group_size).reshape(out_features, in_features)
    mae_expand = mae.unsqueeze(-1).expand(-1, -1, args.group_size).reshape(out_features, in_features)
    qjl_sign = unpack_qjl_signs(packed_qjl, in_features)
    w_manual = w_codes * alpha_expand + qjl_sign * mae_expand

    max_abs_recon_err = (w_unpack.float() - w_manual.float()).abs().max().item()
    mse_recon = (w_unpack.float() - w_manual.float()).pow(2).mean().item()
    print("\n[broadcast audit]")
    print(f"groups: {n_groups} (group_size={args.group_size})")
    print(f"max|turbo_unpack - manual| : {max_abs_recon_err:.6e}")
    print(f"mse(turbo_unpack, manual)  : {mse_recon:.6e}")

    # Boundary check around first few group transitions for row 0.
    row = 0
    boundary_cols = []
    for g in range(1, min(n_groups, 4)):
        c0 = g * args.group_size - 1
        c1 = g * args.group_size
        boundary_cols.extend([c0, c1])
    if not boundary_cols:
        boundary_cols = list(range(min(8, in_features)))

    print("\n[group boundary samples @ row 0]")
    print(" col | group | code | sign | alpha_g     | mae_g       | manual_w")
    print("-----+-------+------+------|-------------+-------------+-------------")
    for c in boundary_cols:
        g = c // args.group_size
        code = int(w_codes[row, c].item())
        sign = int(qjl_sign[row, c].item())
        a = float(alpha[row, g].item())
        e = float(mae[row, g].item())
        wm = float(w_manual[row, c].item())
        print(f"{c:4d} | {g:5d} | {code:4d} | {sign:4d} | {a: .8f} | {e: .8f} | {wm: .8f}")

    # Inverse transform audit: recovered original-space weights.
    n_blocks = in_features // args.block_size
    w_blocks = w_unpack.view(out_features, n_blocks, args.block_size).contiguous()
    if args.inverse_mode == "divide_n":
        w_unrot = (inf.CQ.fwht_forward(w_blocks, False) / float(args.block_size)).view(
            out_features, in_features
        )
    else:
        w_unrot = inf.CQ.fwht_inverse_block(w_blocks, args.block_size).view(
            out_features, in_features
        )
    inv_mse = (w_unrot.float() - weight.float()).pow(2).mean().item()
    inv_cos = torch.nn.functional.cosine_similarity(
        w_unrot.float().flatten(), weight.float().flatten(), dim=0
    ).item()
    print("\n[inverse hadamard audit]")
    print(f"MSE(recovered_w, original_w) : {inv_mse:.6e}")
    print(f"Cos(recovered_w, original_w) : {inv_cos:.8f}")


if __name__ == "__main__":
    main()
