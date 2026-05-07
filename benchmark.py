"""
Benchmark: fused FWHT kernel vs. torch.matmul against a Hadamard matrix.

* Both paths apply an orthonormal Hadamard transform (entries +/- 1/sqrt(N))
  to the last dimension of an FP16 tensor.
* Timing uses CUDA events and is reported in microseconds.
* The kernel is JIT-compiled on first run via torch.utils.cpp_extension.load
  with sm_86-specific flags; subsequent runs hit the build cache.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


# ---------------------------------------------------------------------------
# 1.  JIT build the extension (locked to Compute Capability 8.6 = Ampere)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

# Force nvcc to emit only sm_86 SASS so the kernel is exactly tuned for the
# RTX 3080 Ti. TORCH_CUDA_ARCH_LIST also feeds PyTorch's own arch detection.
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

print("[build] JIT-compiling cracked_quant (FWHT + ternary) for sm_86 ...", flush=True)
fwht_ext = load(
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


# ---------------------------------------------------------------------------
# 2.  Reference: Sylvester-Hadamard matrix, normalised to be orthonormal
# ---------------------------------------------------------------------------
def hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return an n x n orthonormal Hadamard matrix (n must be a power of 2)."""
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while H.shape[0] < n:
        top = torch.cat([H,  H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    H = H / (n ** 0.5)
    return H.to(dtype)


# ---------------------------------------------------------------------------
# 3.  Timing helpers
# ---------------------------------------------------------------------------
def time_us(fn, iters: int = 1000, warmup: int = 50) -> float:
    """Return mean runtime of `fn` in microseconds, measured on a CUDA stream."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds -> convert to microseconds
    return start.elapsed_time(end) * 1000.0 / iters


# ---------------------------------------------------------------------------
# 4.  Benchmark driver
# ---------------------------------------------------------------------------
def run(sizes_n, sizes_b, iters: int) -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available - aborting.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda")
    dtype = torch.float16
    props = torch.cuda.get_device_properties(device)
    print(f"GPU         : {props.name}")
    print(f"Compute cap : {props.major}.{props.minor}")
    print(f"SMs         : {props.multi_processor_count}")
    print(f"Memory      : {props.total_memory / (1024**3):.1f} GiB")
    print(f"Iterations  : {iters} (per measurement)\n")

    header = (
        f"{'N':>5} {'B':>7} | {'matmul (us)':>12} {'fwht (us)':>10} "
        f"{'speedup':>8} | {'max |err|':>10}"
    )
    print(header)
    print("-" * len(header))

    for N in sizes_n:
        H = hadamard_matrix(N, device, dtype)        # (N, N) FP16
        # H is symmetric for power-of-2 Sylvester construction, so x @ H == x @ H.T
        for B in sizes_b:
            x = torch.randn(B, N, device=device, dtype=dtype)

            # ---- Correctness check (FP32-promoted reference) ----
            y_matmul = x @ H
            y_kernel = fwht_ext.fwht_forward(x, True)
            max_err = (y_matmul.float() - y_kernel.float()).abs().max().item()

            # ---- Timing ----
            def run_matmul():
                # Re-bind locals so closures capture current x, H
                _ = x @ H
            def run_kernel():
                _ = fwht_ext.fwht_forward(x, True)

            t_mm  = time_us(run_matmul, iters=iters)
            t_fw  = time_us(run_kernel, iters=iters)
            speed = t_mm / t_fw if t_fw > 0 else float("inf")

            print(
                f"{N:>5} {B:>7} | {t_mm:>12.2f} {t_fw:>10.2f} "
                f"{speed:>7.2f}x | {max_err:>10.2e}"
            )
        print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fused FWHT vs torch.matmul benchmark")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Transform sizes N (must be powers of 2, <= 1024)",
    )
    p.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384, 65536],
        help="Batch sizes B to sweep",
    )
    p.add_argument(
        "--iters",
        type=int,
        default=2000,
        help="Iterations per timed measurement",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.sizes, args.batches, args.iters)
