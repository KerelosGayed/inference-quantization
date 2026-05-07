// Unified Python bindings for the cracked_quant extension.
//
// This file links together the two CUDA translation units
//   * fused_fwht.cu         (Fast Walsh-Hadamard Transform, FP16)
//   * ternary_packing.cu    (1.58-bit ternary weight pack / unpack)
// into a single Python module named `cracked_quant`.
//
// The host wrappers themselves are defined in those .cu files; we only
// forward-declare them here and register them with pybind11.

#include <torch/extension.h>

// ---------- forward declarations (definitions live in the .cu files) ------
torch::Tensor fwht_forward(torch::Tensor x, bool normalize);
torch::Tensor fwht_inverse_block(torch::Tensor x, int64_t block_size) {
    TORCH_CHECK(x.is_cuda(), "fwht_inverse_block: input must be on CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "fwht_inverse_block: input must be float16");
    TORCH_CHECK(x.is_contiguous(), "fwht_inverse_block: input must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "fwht_inverse_block: input must have >= 1 dimension");
    TORCH_CHECK(block_size >= 1 && block_size <= 1024,
                "fwht_inverse_block: block_size must satisfy 1 <= block_size <= 1024");
    TORCH_CHECK((block_size & (block_size - 1)) == 0,
                "fwht_inverse_block: block_size must be a power of 2");
    TORCH_CHECK(x.size(-1) == block_size,
                "fwht_inverse_block: last dimension must equal block_size");

    // Quantization pipeline stores rotated weights using normalized FWHT
    // (scale = 1/sqrt(N)). The inverse is therefore the same normalized FWHT.
    // Using unnormalized FWHT / N here would introduce an extra 1/sqrt(N)
    // shrinkage and a large magnitude mismatch.
    return fwht_forward(x, true);
}
torch::Tensor ternary_pack(torch::Tensor x, int64_t group_size);
torch::Tensor ternary_unpack(torch::Tensor packed, double scale);
torch::Tensor qjl_pack(torch::Tensor residual, int64_t group_size);
torch::Tensor turbo_unpack(torch::Tensor packed_ternary,
                           torch::Tensor packed_qjl,
                           torch::Tensor alpha,
                           torch::Tensor scale_error,
                           int64_t group_size);

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() =
        "cracked_quant - fused FWHT + 1.58-bit ternary packing kernels, "
        "tuned for NVIDIA Ampere consumer GPUs (Compute Capability 8.6).";

    // ------------------------- FWHT --------------------------------------
    m.def(
        "fwht_forward",
        &fwht_forward,
        "Fused orthonormal Fast Walsh-Hadamard Transform along the last "
        "dimension. Last dim must be a power of 2 in [1, 1024]. FP16 only.",
        pybind11::arg("x"),
        pybind11::arg("normalize") = true);

    m.def(
        "fwht_inverse_block",
        &fwht_inverse_block,
        "Inverse block Hadamard for normalized FWHT pipelines. "
        "Input must be float16 on CUDA and have last dim equal to block_size.",
        pybind11::arg("x"),
        pybind11::arg("block_size"));

    // -------------------- Ternary pack / unpack --------------------------
    m.def(
        "ternary_pack",
        &ternary_pack,
        "Pack a tensor of ternary values { -1, 0, +1 } into 2-bit fields "
        "(4 weights per uint8). Last dim must be a multiple of 4 and "
        "divisible by group_size. Accepts float16 or float32 input.",
        pybind11::arg("x"),
        pybind11::arg("group_size") = 128);

    m.def(
        "ternary_unpack",
        &ternary_unpack,
        "Unpack a uint8 tensor of 2-bit ternary codes back to float16, "
        "expanding the last dim by 4x. Each output element is multiplied by "
        "`scale` (set to 1.0 for raw {-1, 0, +1}).",
        pybind11::arg("packed"),
        pybind11::arg("scale") = 1.0);

    m.def(
        "qjl_pack",
        &qjl_pack,
        "Pack residual sign bits (8 residual signs per uint8). "
        "Each bit is 1 for non-negative residual and 0 for negative. "
        "Last dim must be divisible by group_size.",
        pybind11::arg("residual"),
        pybind11::arg("group_size") = 128);

    m.def(
        "turbo_unpack",
        &turbo_unpack,
        "TurboQuant reconstruction. Combines packed ternary codes and packed "
        "QJL sign bits with per-group alpha and scale_error: "
        "W_rec = W_ternary + sign * scale_error.",
        pybind11::arg("packed_ternary"),
        pybind11::arg("packed_qjl"),
        pybind11::arg("alpha"),
        pybind11::arg("scale_error"),
        pybind11::arg("group_size") = 128);
}
