"""
End-to-end 1.58-bit ternary quantization pipeline for a Llama-3-8B-style
linear layer (4096 x 4096 - matches `q_proj` / `o_proj`).

Pipeline:

    [1] Rotation phase      : block-diagonal Fast Walsh-Hadamard Transform
                              along the input axis to "smooth out" outliers.
    [2] Scaling phase       : group-wise alpha_g = mean(|W_rot_group|).
    [3] Quantization phase  : W_q = clamp(round(W_rot / alpha_g), -1, 1)
                              with values exactly in { -1, 0, +1 }.
    [4] Packing phase       : pack ternary + 1-bit QJL residual signs.
    [5] Validation          : VRAM accounting, compression ratio, MSE/SNR
                              between W_rot and reconstructed recovery.

Run:   python quantize_model.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


# ===========================================================================
# 0.  Build the cracked_quant extension (JIT)
# ===========================================================================
HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

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

print("[build] JIT-compiling cracked_quant (FWHT + ternary) for sm_86 ...",
      flush=True)
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
print("[build] done.\n", flush=True)


# ===========================================================================
# 1.  Helpers
# ===========================================================================
def fmt_bytes(n: int) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024.0:
            return f"{n:8.2f} {unit}"
        n /= 1024.0
    return f"{n:8.2f} TiB"


def banner(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


# ===========================================================================
# 2.  Rotation phase  -  block-diagonal Hadamard transform
# ===========================================================================
def hadamard_rotate(W: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Apply a block-diagonal orthonormal Hadamard rotation along the *input*
    (last) axis of a weight matrix W of shape (out_features, in_features).

    Why rotate?
    -----------
    LLM activations and weights are heavy-tailed: a small number of
    coordinates carry massive magnitude (the "outlier" channels first
    documented in LLM.int8). Outliers are catastrophic for ternary
    quantization, because alpha = mean(|W|) is dragged upwards and
    nearly every non-outlier weight collapses to zero after rounding.

    The Hadamard transform H is an orthonormal matrix whose rows are
    orthogonal +/-1 patterns. Multiplying a vector v by H produces a new
    vector whose every coordinate is a normalized sum (with random-ish
    signs) of *all* original coordinates. By the Central Limit Theorem
    that mixed distribution is much closer to a Gaussian - the heavy
    tail is spread evenly across all dims and outliers vanish.

    Crucially, H is its own inverse (H @ H = I when normalized by 1/sqrt(N)),
    so the transformation is *free* at inference time:

        y = W @ x  =  (W @ H^T) @ (H @ x)  =  W_rot @ x_rot

    We pre-multiply the weights once, offline (this script), and rotate
    activations on the fly with the same FWHT kernel.

    For Llama-3 we use a block-diagonal Hadamard with block_size = head_dim
    (= 128). This matches the QuaRot / SpinQuant recipe and lets us reuse
    the FWHT kernel even though the full hidden_size (4096) exceeds the
    kernel's 1024 ceiling.
    """
    assert W.dim() == 2, "expected (out_features, in_features)"
    out_features, in_features = W.shape
    assert in_features % block_size == 0, \
        f"in_features ({in_features}) must be a multiple of block_size ({block_size})"
    assert (block_size & (block_size - 1)) == 0 and block_size <= 1024, \
        "block_size must be a power of 2 in [1, 1024]"

    n_blocks = in_features // block_size
    W_blocks = W.view(out_features, n_blocks, block_size).contiguous()
    W_rot = cq.fwht_forward(W_blocks, True)        # FWHT along last dim
    return W_rot.view(out_features, in_features)


# ===========================================================================
# 3.  Group-wise scaling + ternary quantization (BitNet b1.58 style)
# ===========================================================================
def ternary_quantize_groupwise(W_rot: torch.Tensor, group_size: int):
    """
    Group-wise ternary quantization along the input axis.

      alpha_g = mean(|W_rot_group|)       shape: (out_features, n_groups)
      W_q   = clamp(round(W_rot / alpha), -1, +1)

    Returns (W_q in {-1, 0, +1} as fp16, alpha_g as fp16).
    """
    assert W_rot.dim() == 2, "expected (out_features, in_features)"
    out_features, in_features = W_rot.shape
    assert group_size > 0 and in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )
    n_groups = in_features // group_size
    grouped = W_rot.view(out_features, n_groups, group_size)
    alpha = grouped.abs().mean(dim=-1).clamp_min(1e-8)
    W_scaled = grouped / alpha.unsqueeze(-1)
    W_q = W_scaled.round().clamp_(-1.0, 1.0).to(torch.float16)
    return W_q.view(out_features, in_features).contiguous(), alpha.to(torch.float16).contiguous()


# ===========================================================================
# 4.  Main pipeline
# ===========================================================================
def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available - aborting.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(0xCAFE)

    # ---- Llama-3-8B q_proj / o_proj shape -----------------------------------
    OUT_DIM, IN_DIM = 4096, 4096
    BLOCK_SIZE = 128                # = Llama-3 head_dim, QuaRot convention
    GROUP_SIZE = 128                # group-wise alpha / MAE granularity

    props = torch.cuda.get_device_properties(device)
    print(f"GPU              : {props.name}")
    print(f"Compute cap      : {props.major}.{props.minor}")
    print(f"Total VRAM       : {fmt_bytes(props.total_memory)}")
    print(f"Layer shape      : ({OUT_DIM}, {IN_DIM})  "
          f"[Llama-3-8B q_proj/o_proj]")
    print(f"Hadamard block   : {BLOCK_SIZE}  "
          f"[block-diagonal, {IN_DIM // BLOCK_SIZE} blocks per row]")
    print(f"Quant group size : {GROUP_SIZE}  "
          f"[{IN_DIM // GROUP_SIZE} groups per row]")

    # ---- Synthesize a layer with a heavy-tailed outlier distribution -------
    # Real Llama weights are ~ N(0, 0.02) but with rare outliers up to ~1.0.
    # We inject those manually so that the rotation phase has something
    # meaningful to smooth out.
    W = (torch.randn(OUT_DIM, IN_DIM, device=device, dtype=dtype) * 0.02)
    n_outliers = (OUT_DIM * IN_DIM) // 1000     # 0.1% of weights
    flat = W.view(-1)
    idx = torch.randint(0, flat.numel(), (n_outliers,), device=device)
    flat[idx] += torch.randn(n_outliers, device=device, dtype=dtype) * 0.5

    banner("[1] ROTATION PHASE - block-diagonal Hadamard transform")
    kurt_before = _kurtosis(W)
    W_rot = hadamard_rotate(W, block_size=BLOCK_SIZE)
    kurt_after = _kurtosis(W_rot)
    max_before = W.abs().max().item()
    max_after  = W_rot.abs().max().item()
    print(f"  max|W|       before -> after :  {max_before:.4f}  ->  {max_after:.4f}")
    print(f"  kurtosis     before -> after :  {kurt_before:7.2f}  ->  {kurt_after:7.2f}"
          f"   (lower = closer to Gaussian)")

    banner("[2-3] GROUP-WISE SCALING + TERNARY QUANTIZATION")
    W_q, alpha = ternary_quantize_groupwise(W_rot, GROUP_SIZE)
    n_neg = (W_q == -1).sum().item()
    n_zero = (W_q ==  0).sum().item()
    n_pos = (W_q == +1).sum().item()
    total = W_q.numel()
    print(f"  alpha (group-wise) range : [{alpha.min().item():.5f},"
          f" {alpha.max().item():.5f}]")
    print(f"  ternary distribution  :  -1: {n_neg/total:6.2%}   "
          f"0: {n_zero/total:6.2%}   +1: {n_pos/total:6.2%}")

    banner("[4] PACKING PHASE - ternary + 1-bit QJL residual")
    alpha_expand = alpha.unsqueeze(-1).expand(-1, -1, GROUP_SIZE).reshape(OUT_DIM, IN_DIM)
    W_packed = cq.ternary_pack(W_q, GROUP_SIZE)
    W_quantized = W_q * alpha_expand
    residual = (W_rot - W_quantized).contiguous()
    scale_error = residual.view(OUT_DIM, IN_DIM // GROUP_SIZE, GROUP_SIZE).abs().mean(dim=-1)
    scale_error = scale_error.clamp_min(1e-8).to(torch.float16).contiguous()
    qjl_packed = cq.qjl_pack(residual, GROUP_SIZE)
    assert W_packed.dtype == torch.uint8
    assert qjl_packed.dtype == torch.uint8
    assert W_packed.shape == (OUT_DIM, IN_DIM // 4)
    assert qjl_packed.shape == (OUT_DIM, IN_DIM // 8)
    print(f"  packed shape : {tuple(W_packed.shape)}  dtype={W_packed.dtype}")
    print(f"  qjl shape    : {tuple(qjl_packed.shape)}  dtype={qjl_packed.dtype}")
    print(f"  scale tensors shape : alpha={tuple(alpha.shape)}  mae={tuple(scale_error.shape)}")
    print(f"  MAE (group-wise) range : [{scale_error.min().item():.5f}, {scale_error.max().item():.5f}]")

    # ---- VRAM accounting (theoretical numel * dtype_size) ------------------
    fp16_orig_bytes = W.numel() * W.element_size()
    packed_bytes    = W_packed.numel() * W_packed.element_size()
    qjl_bytes       = qjl_packed.numel() * qjl_packed.element_size()
    alpha_bytes     = alpha.numel() * alpha.element_size()
    error_bytes     = scale_error.numel() * scale_error.element_size()
    quantized_total = packed_bytes + qjl_bytes + alpha_bytes + error_bytes
    ratio           = fp16_orig_bytes / quantized_total

    banner("[5] VALIDATION - VRAM, compression, MSE")
    print(f"  Original FP16            : {fmt_bytes(fp16_orig_bytes)}")
    print(f"  Packed 2-bit weights     : {fmt_bytes(packed_bytes)}")
    print(f"  Packed 1-bit QJL signs   : {fmt_bytes(qjl_bytes)}")
    print(f"  Group alpha (FP16)       : {fmt_bytes(alpha_bytes)}")
    print(f"  Group MAE (FP16)         : {fmt_bytes(error_bytes)}")
    print(f"  Total quantized          : {fmt_bytes(quantized_total)}")
    print(f"  Compression ratio        : {ratio:.2f} x"
          f"   (raw (2+1)-bit/16-bit = 5.33 x)")

    # ---- Baseline ternary reconstruction ----------------------------------
    W_rec_codes = cq.ternary_unpack(W_packed, 1.0)        # in {-1, 0, +1}
    W_rec = W_rec_codes * alpha_expand                    # broadcast per group

    mse = (W_rot.float() - W_rec.float()).pow(2).mean().item()
    var = W_rot.float().var().item()
    rel = mse / max(var, 1e-30)
    snr_db = 10.0 * math.log10(var / mse) if mse > 0 else float("inf")

    # ---- TurboQuant reconstruction with QJL residual ----------------------
    W_turbo = cq.turbo_unpack(
        W_packed,
        qjl_packed,
        alpha.contiguous(),
        scale_error.contiguous(),
        GROUP_SIZE,
    )
    mse_turbo = (W_rot.float() - W_turbo.float()).pow(2).mean().item()
    rel_turbo = mse_turbo / max(var, 1e-30)
    snr_turbo_db = 10.0 * math.log10(var / mse_turbo) if mse_turbo > 0 else float("inf")

    print(f"  MSE ternary              : {mse:.6e}")
    print(f"  MSE turbo (QJL)          : {mse_turbo:.6e}")
    print(f"  Var(W_rot)               : {var:.6e}")
    print(f"  Relative MSE ternary     : {rel:.4f}"
          f"   (= noise / signal power)")
    print(f"  Relative MSE turbo       : {rel_turbo:.4f}"
          f"   (= noise / signal power)")
    print(f"  SNR ternary              : {snr_db:6.2f} dB")
    print(f"  SNR turbo (group-wise QJL): {snr_turbo_db:6.2f} dB"
          f"   (target >= 10 dB for stable inference)")

    # ---- Extrapolate to the full Llama-3-8B model -------------------------
    # ~8.03B parameters total. In practice the embedding + lm_head are kept
    # at FP16 (~525 MiB), and only the ~7.5B Linear weights are ternarized.
    banner("EXTRAPOLATION - full Llama-3-8B in 12 GiB VRAM?")
    LLAMA3_LINEAR_PARAMS = 7_504_924_672          # ~ 7.5 B linear params
    LLAMA3_EMBED_PARAMS  =   525_336_576          # tied embeddings @ FP16
    fp16_full     = (LLAMA3_LINEAR_PARAMS + LLAMA3_EMBED_PARAMS) * 2
    groups_per_row = IN_DIM // GROUP_SIZE
    scale_params = (LLAMA3_LINEAR_PARAMS // IN_DIM) * groups_per_row
    quantized_full = (LLAMA3_LINEAR_PARAMS // 4)         \
                   + (LLAMA3_LINEAR_PARAMS // 8)         \
                   + (scale_params * 2)                  \
                   + (scale_params * 2)                  \
                   + (LLAMA3_EMBED_PARAMS * 2)
    headroom = 12 * (1024 ** 3) - quantized_full
    print(f"  FP16 weights             : {fmt_bytes(fp16_full)}")
    print(f"  Quantized weights        : {fmt_bytes(quantized_full)}")
    print(f"  Compression ratio        : {fp16_full / quantized_full:.2f} x")
    print(f"  12 GiB budget headroom   : {fmt_bytes(headroom)}"
          f"   ({'OK -> KV cache + activations fit' if headroom > 0 else 'overflow!'})")
    print()


def _kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of a tensor (in float32). 0.0 = Gaussian."""
    f = x.float().flatten()
    m = f.mean()
    s = f.std(unbiased=False)
    return ((f - m).pow(4).mean() / (s.pow(4) + 1e-30) - 3.0).item()


if __name__ == "__main__":
    main()
