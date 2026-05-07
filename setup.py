"""
Build script for the `cracked_quant` PyTorch C++/CUDA extension.

Compiles three translation units into ONE Python module:

    fused_fwht.cu       - fused Fast Walsh-Hadamard Transform (FP16)
    ternary_packing.cu  - ternary + 1-bit QJL residual pack/unpack kernels
    cracked_quant.cpp   - unified pybind11 module registration

Hard-tuned for Compute Capability 8.6 (Ampere consumer GPUs - RTX 30xx
including the 3080 Ti). Targets CUDA 13.1 / Ubuntu 24.04 / gcc 13 / C++17.

Usage:

    # Ahead-of-time install
    python setup.py install               # or: pip install -e .

    # Then in Python:
    import cracked_quant as cq
    y       = cq.fwht_forward(x_fp16, normalize=True)
    packed  = cq.ternary_pack(W_ternary, group_size=128)
    qjl     = cq.qjl_pack(W_residual, group_size=128)
    W_hat   = cq.turbo_unpack(packed, qjl, alpha_groups, mae_groups, group_size=128)

    # Or JIT (no install) - same flag profile, three sources:
    from torch.utils.cpp_extension import load
    cq = load(name="cracked_quant",
              sources=["fused_fwht.cu", "ternary_packing.cu", "cracked_quant.cpp"],
              extra_cuda_cflags=NVCC_FLAGS, verbose=True)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

# Ampere consumer (sm_86) - RTX 3080 / 3080 Ti / 3090 / A40 etc.
NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_86,code=sm_86",
    # Make sure all FP16 helpers are visible
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # Helpful for kernel tuning - dump register / smem usage at compile time
    "-Xptxas=-v",
    "--expt-relaxed-constexpr",
    "--ptxas-options=-O3",
]

CXX_FLAGS = ["-O3", "-std=c++17"]


setup(
    name="cracked_quant",
    version="0.1.0",
    description=(
        "Cracked LLM inference kernels for RTX 3080 Ti (sm_86): "
        "fused FWHT + ternary + QJL residual packing."
    ),
    ext_modules=[
        CUDAExtension(
            name="cracked_quant",
            sources=[
                "fused_fwht.cu",
                "ternary_packing.cu",
                "cracked_quant.cpp",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
