// Ternary (1.58-bit) weight packing / unpacking kernels.
// Tuned for NVIDIA Ampere consumer GPUs (Compute Capability 8.6 - RTX 3080 Ti).
//
// Encoding (2 bits per weight, 4 weights per byte):
//
//   value | code     comment
//   ------+-------+--------------------------------
//      0  | 0b00  | natural zero
//     +1  | 0b01  | low bit = magnitude
//     -1  | 0b11  | high bit acts as sign
//     --- | 0b10  | reserved (never produced)
//
// The clever bit:  decoding is a single arithmetic right shift.  If the
// 2-bit field is placed in the top of an int8_t and shifted >> 6, the
// sign bit replicates automatically:
//
//   0b00 -> 0x00 -> 0
//   0b01 -> 0x40 -> +1
//   0b11 -> 0xC0 -> -1
//
// That is the entire dequantization step (plus a multiply by `scale`).
//
// Memory plan (per thread):
//   pack:    read  4 fp16  (8  B)  -> write 1 byte
//   unpack:  read  1 byte         -> write 4 fp16 (8 B = one int2 store)
//
// A warp therefore touches:
//   pack:    256 B fp16 read (2 sectors, fully coalesced)
//            32  B byte  write (1 sector)
//   unpack:  32  B byte  read  (1 sector)
//            256 B fp16 write (2 sectors, fully coalesced)
//
// On the 3080 Ti this saturates the L2 -> SM bus and gets very close to
// the 860 GB/s HBM ceiling for both directions.

#include <torch/extension.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// Encode one ternary scalar -> 2-bit code, branchless.
//   sign = (f > +0.5) - (f < -0.5)   in { -1, 0, +1 }
//   code = sign & 0b11
// Two's-complement of -1 is 0xFF... so AND-ing with 3 yields 0b11.
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ uint32_t encode_ternary(T v)
{
    const float f    = static_cast<float>(v);
    const int   sign = static_cast<int>(f > 0.5f) - static_cast<int>(f < -0.5f);
    return static_cast<uint32_t>(sign & 0x3);
}

template <typename T>
__device__ __forceinline__ uint32_t encode_qjl_signbit(T v)
{
    const float f = static_cast<float>(v);
    return static_cast<uint32_t>(f >= 0.0f);
}

// ---------------------------------------------------------------------------
// Pack kernel.  Each thread:
//   * issues one 64-bit (int2) load of 4 fp16 weights
//   * encodes them into 4 two-bit fields
//   * issues one byte store
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void ternary_pack_kernel_fp16(const __half* __restrict__ x,
                              uint8_t*      __restrict__ out,
                              int64_t       n_packed)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;

    // 64-bit vector load: 4 halves at once, 8-byte aligned.
    const int2 v = __ldg(reinterpret_cast<const int2*>(x) + tid);

    const __half2 lo = *reinterpret_cast<const __half2*>(&v.x);
    const __half2 hi = *reinterpret_cast<const __half2*>(&v.y);

    const uint32_t c0 = encode_ternary(__low2half(lo));
    const uint32_t c1 = encode_ternary(__high2half(lo));
    const uint32_t c2 = encode_ternary(__low2half(hi));
    const uint32_t c3 = encode_ternary(__high2half(hi));

    out[tid] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
}

// FP32 fallback path - same algorithm, scalar reads (compiler will
// coalesce them into a single 128-bit load per warp).
__global__ __launch_bounds__(256)
void ternary_pack_kernel_fp32(const float* __restrict__ x,
                              uint8_t*     __restrict__ out,
                              int64_t      n_packed)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;

    const float4 v = __ldg(reinterpret_cast<const float4*>(x) + tid);

    const uint32_t c0 = encode_ternary(v.x);
    const uint32_t c1 = encode_ternary(v.y);
    const uint32_t c2 = encode_ternary(v.z);
    const uint32_t c3 = encode_ternary(v.w);

    out[tid] = static_cast<uint8_t>(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
}

__global__ __launch_bounds__(256)
void qjl_pack_kernel_fp16(const __half* __restrict__ x,
                          uint8_t*      __restrict__ out,
                          int64_t       n_packed)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;

    const int64_t base = tid * 8;
    const uint32_t b0 = encode_qjl_signbit(x[base + 0]);
    const uint32_t b1 = encode_qjl_signbit(x[base + 1]);
    const uint32_t b2 = encode_qjl_signbit(x[base + 2]);
    const uint32_t b3 = encode_qjl_signbit(x[base + 3]);
    const uint32_t b4 = encode_qjl_signbit(x[base + 4]);
    const uint32_t b5 = encode_qjl_signbit(x[base + 5]);
    const uint32_t b6 = encode_qjl_signbit(x[base + 6]);
    const uint32_t b7 = encode_qjl_signbit(x[base + 7]);

    out[tid] = static_cast<uint8_t>(
        b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) |
        (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7));
}

__global__ __launch_bounds__(256)
void qjl_pack_kernel_fp32(const float* __restrict__ x,
                          uint8_t*     __restrict__ out,
                          int64_t      n_packed)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;

    const int64_t base = tid * 8;
    const uint32_t b0 = encode_qjl_signbit(x[base + 0]);
    const uint32_t b1 = encode_qjl_signbit(x[base + 1]);
    const uint32_t b2 = encode_qjl_signbit(x[base + 2]);
    const uint32_t b3 = encode_qjl_signbit(x[base + 3]);
    const uint32_t b4 = encode_qjl_signbit(x[base + 4]);
    const uint32_t b5 = encode_qjl_signbit(x[base + 5]);
    const uint32_t b6 = encode_qjl_signbit(x[base + 6]);
    const uint32_t b7 = encode_qjl_signbit(x[base + 7]);

    out[tid] = static_cast<uint8_t>(
        b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) |
        (b4 << 4) | (b5 << 5) | (b6 << 6) | (b7 << 7));
}

// ---------------------------------------------------------------------------
// Unpack kernel.  Each thread:
//   * loads one byte
//   * sign-extends each 2-bit field via shift+cast
//   * scales and writes 4 halves as one 64-bit (int2) store
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256)
void ternary_unpack_kernel_fp16(const uint8_t* __restrict__ packed,
                                __half*        __restrict__ out,
                                int64_t        n_packed,
                                float          scale)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed) return;

    const uint8_t b = packed[tid];

    // Place each 2-bit field in the top of an int8_t, then arithmetic
    // shift right by 6 -> sign-extended { -1, 0, +1 } per lane.
    //   bits 0..1 -> shift left 6
    //   bits 2..3 -> shift left 4
    //   bits 4..5 -> shift left 2
    //   bits 6..7 -> shift left 0
    const int v0 = static_cast<int>(static_cast<int8_t>(b << 6)) >> 6;
    const int v1 = static_cast<int>(static_cast<int8_t>(b << 4)) >> 6;
    const int v2 = static_cast<int>(static_cast<int8_t>(b << 2)) >> 6;
    const int v3 = static_cast<int>(static_cast<int8_t>(b << 0)) >> 6;

    // Pair-wise FMA via half2 intrinsics -> two HFMA2 instructions on Ampere.
    const __half2 hscale = __half2half2(__float2half(scale));
    const __half2 lo = __hmul2(
        __halves2half2(__float2half(static_cast<float>(v0)),
                       __float2half(static_cast<float>(v1))),
        hscale);
    const __half2 hi = __hmul2(
        __halves2half2(__float2half(static_cast<float>(v2)),
                       __float2half(static_cast<float>(v3))),
        hscale);

    int2 result;
    *reinterpret_cast<__half2*>(&result.x) = lo;
    *reinterpret_cast<__half2*>(&result.y) = hi;
    reinterpret_cast<int2*>(out)[tid] = result;
}

__global__ __launch_bounds__(256)
void turbo_unpack_kernel_fp16(const uint8_t* __restrict__ packed_ternary,
                              const uint8_t* __restrict__ packed_qjl,
                              const float*   __restrict__ alpha,
                              const float*   __restrict__ scale_error,
                              __half*        __restrict__ out,
                              int64_t        n_packed_ternary,
                              int64_t        in_features,
                              int64_t        qjl_cols,
                              int64_t        n_groups,
                              int64_t        group_size)
{
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= n_packed_ternary) return;

    const uint8_t b = packed_ternary[tid];
    const int t0 = static_cast<int>(static_cast<int8_t>(b << 6)) >> 6;
    const int t1 = static_cast<int>(static_cast<int8_t>(b << 4)) >> 6;
    const int t2 = static_cast<int>(static_cast<int8_t>(b << 2)) >> 6;
    const int t3 = static_cast<int>(static_cast<int8_t>(b << 0)) >> 6;

    const int64_t base_idx = tid * 4;
    const int64_t row = base_idx / in_features;
    const int64_t scale_base = row * n_groups;

    const int64_t col0 = base_idx - row * in_features;
    const int64_t col1 = col0 + 1;
    const int64_t col2 = col0 + 2;
    const int64_t col3 = col0 + 3;
    const int64_t g0 = col0 / group_size;
    const int64_t g1 = col1 / group_size;
    const int64_t g2 = col2 / group_size;
    const int64_t g3 = col3 / group_size;

    // Register cache: avoid redundant global scale loads when multiple lanes
    // fall in the same group (common for group_size >= 64).
    const float a0 = alpha[scale_base + g0];
    const float e0 = scale_error[scale_base + g0];
    const float a1 = (g1 == g0) ? a0 : alpha[scale_base + g1];
    const float e1 = (g1 == g0) ? e0 : scale_error[scale_base + g1];
    const float a2 = (g2 == g1) ? a1 : ((g2 == g0) ? a0 : alpha[scale_base + g2]);
    const float e2 = (g2 == g1) ? e1 : ((g2 == g0) ? e0 : scale_error[scale_base + g2]);
    const float a3 = (g3 == g2) ? a2 : ((g3 == g1) ? a1 : ((g3 == g0) ? a0 : alpha[scale_base + g3]));
    const float e3 = (g3 == g2) ? e2 : ((g3 == g1) ? e1 : ((g3 == g0) ? e0 : scale_error[scale_base + g3]));

    const int64_t q0 = row * qjl_cols + (col0 >> 3);
    const int64_t q1 = row * qjl_cols + (col1 >> 3);
    const int64_t q2 = row * qjl_cols + (col2 >> 3);
    const int64_t q3 = row * qjl_cols + (col3 >> 3);
    const float s0 = ((packed_qjl[q0] >> (col0 & 7)) & 0x1) ? 1.0f : -1.0f;
    const float s1 = ((packed_qjl[q1] >> (col1 & 7)) & 0x1) ? 1.0f : -1.0f;
    const float s2 = ((packed_qjl[q2] >> (col2 & 7)) & 0x1) ? 1.0f : -1.0f;
    const float s3 = ((packed_qjl[q3] >> (col3 & 7)) & 0x1) ? 1.0f : -1.0f;

    const __half2 lo = __halves2half2(
        __float2half(static_cast<float>(t0) * a0 + s0 * e0),
        __float2half(static_cast<float>(t1) * a1 + s1 * e1));
    const __half2 hi = __halves2half2(
        __float2half(static_cast<float>(t2) * a2 + s2 * e2),
        __float2half(static_cast<float>(t3) * a3 + s3 * e3));

    int2 result;
    *reinterpret_cast<__half2*>(&result.x) = lo;
    *reinterpret_cast<__half2*>(&result.y) = hi;
    reinterpret_cast<int2*>(out)[tid] = result;
}

} // namespace

// ===========================================================================
// Host wrappers (Python-visible API)
// ===========================================================================

torch::Tensor ternary_pack(torch::Tensor x, int64_t group_size)
{
    TORCH_CHECK(x.is_cuda(),       "ternary_pack: input must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "ternary_pack: input must be contiguous");
    TORCH_CHECK(x.numel() > 0,     "ternary_pack: input is empty");
    TORCH_CHECK(group_size > 0,    "ternary_pack: group_size must be > 0");
    TORCH_CHECK(x.numel() % 4 == 0,
                "ternary_pack: numel must be divisible by 4 (got ", x.numel(), ")");

    const int64_t n_packed = x.numel() / 4;

    // Output: same shape but last dim divided by 4, dtype uint8.
    auto out_shape = x.sizes().vec();
    TORCH_CHECK(!out_shape.empty(), "ternary_pack: input must have >= 1 dim");
    TORCH_CHECK(out_shape.back() % group_size == 0,
                "ternary_pack: last dim must be divisible by group_size (got ",
                out_shape.back(), " and ", group_size, ")");
    TORCH_CHECK(out_shape.back() % 4 == 0,
                "ternary_pack: last dim must be divisible by 4 (got ", out_shape.back(), ")");
    out_shape.back() /= 4;

    auto out = torch::empty(
        out_shape,
        torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));

    constexpr int THREADS = 256;
    const int64_t blocks  = (n_packed + THREADS - 1) / THREADS;

    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::kHalf) {
        ternary_pack_kernel_fp16<<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            out.data_ptr<uint8_t>(),
            n_packed);
    } else if (x.scalar_type() == at::kFloat) {
        ternary_pack_kernel_fp32<<<blocks, THREADS, 0, stream>>>(
            x.data_ptr<float>(),
            out.data_ptr<uint8_t>(),
            n_packed);
    } else {
        TORCH_CHECK(false,
                    "ternary_pack: unsupported dtype ", x.scalar_type(),
                    " (expected float16 or float32)");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor ternary_unpack(torch::Tensor packed, double scale)
{
    TORCH_CHECK(packed.is_cuda(),                   "ternary_unpack: input must be on CUDA");
    TORCH_CHECK(packed.scalar_type() == at::kByte,  "ternary_unpack: input must be uint8");
    TORCH_CHECK(packed.is_contiguous(),             "ternary_unpack: input must be contiguous");
    TORCH_CHECK(packed.numel() > 0,                 "ternary_unpack: input is empty");

    auto out_shape = packed.sizes().vec();
    TORCH_CHECK(!out_shape.empty(), "ternary_unpack: input must have >= 1 dim");
    out_shape.back() *= 4;          // each byte expands into 4 fp16 weights

    auto out = torch::empty(
        out_shape,
        torch::TensorOptions().dtype(torch::kHalf).device(packed.device()));

    const int64_t n_packed = packed.numel();

    constexpr int THREADS = 256;
    const int64_t blocks  = (n_packed + THREADS - 1) / THREADS;

    const at::cuda::OptionalCUDAGuard guard(packed.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    ternary_unpack_kernel_fp16<<<blocks, THREADS, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n_packed,
        static_cast<float>(scale));

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor qjl_pack(torch::Tensor residual, int64_t group_size)
{
    TORCH_CHECK(residual.is_cuda(),       "qjl_pack: input must be on CUDA");
    TORCH_CHECK(residual.is_contiguous(), "qjl_pack: input must be contiguous");
    TORCH_CHECK(residual.numel() > 0,     "qjl_pack: input is empty");
    TORCH_CHECK(group_size > 0,           "qjl_pack: group_size must be > 0");
    TORCH_CHECK(residual.numel() % 8 == 0,
                "qjl_pack: numel must be divisible by 8 (got ", residual.numel(), ")");

    auto out_shape = residual.sizes().vec();
    TORCH_CHECK(!out_shape.empty(), "qjl_pack: input must have >= 1 dim");
    TORCH_CHECK(out_shape.back() % group_size == 0,
                "qjl_pack: last dim must be divisible by group_size (got ",
                out_shape.back(), " and ", group_size, ")");
    TORCH_CHECK(out_shape.back() % 8 == 0,
                "qjl_pack: last dim must be divisible by 8 (got ", out_shape.back(), ")");
    out_shape.back() /= 8;

    auto out = torch::empty(
        out_shape,
        torch::TensorOptions().dtype(torch::kUInt8).device(residual.device()));

    const int64_t n_packed = residual.numel() / 8;
    constexpr int THREADS = 256;
    const int64_t blocks  = (n_packed + THREADS - 1) / THREADS;

    const at::cuda::OptionalCUDAGuard guard(residual.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (residual.scalar_type() == at::kHalf) {
        qjl_pack_kernel_fp16<<<blocks, THREADS, 0, stream>>>(
            reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
            out.data_ptr<uint8_t>(),
            n_packed);
    } else if (residual.scalar_type() == at::kFloat) {
        qjl_pack_kernel_fp32<<<blocks, THREADS, 0, stream>>>(
            residual.data_ptr<float>(),
            out.data_ptr<uint8_t>(),
            n_packed);
    } else {
        TORCH_CHECK(false,
                    "qjl_pack: unsupported dtype ", residual.scalar_type(),
                    " (expected float16 or float32)");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor turbo_unpack(torch::Tensor packed_ternary,
                           torch::Tensor packed_qjl,
                           torch::Tensor alpha,
                           torch::Tensor scale_error,
                           int64_t group_size)
{
    TORCH_CHECK(packed_ternary.is_cuda(), "turbo_unpack: packed_ternary must be on CUDA");
    TORCH_CHECK(packed_qjl.is_cuda(),     "turbo_unpack: packed_qjl must be on CUDA");
    TORCH_CHECK(alpha.is_cuda(),          "turbo_unpack: alpha must be on CUDA");
    TORCH_CHECK(scale_error.is_cuda(),    "turbo_unpack: scale_error must be on CUDA");
    TORCH_CHECK(packed_ternary.scalar_type() == at::kByte,
                "turbo_unpack: packed_ternary must be uint8");
    TORCH_CHECK(packed_qjl.scalar_type() == at::kByte,
                "turbo_unpack: packed_qjl must be uint8");
    TORCH_CHECK(packed_ternary.is_contiguous(), "turbo_unpack: packed_ternary must be contiguous");
    TORCH_CHECK(packed_qjl.is_contiguous(),     "turbo_unpack: packed_qjl must be contiguous");
    TORCH_CHECK(alpha.is_contiguous(),          "turbo_unpack: alpha must be contiguous");
    TORCH_CHECK(scale_error.is_contiguous(),    "turbo_unpack: scale_error must be contiguous");
    TORCH_CHECK(group_size > 0,                 "turbo_unpack: group_size must be > 0");
    TORCH_CHECK(packed_ternary.dim() == 2, "turbo_unpack: packed_ternary must be 2D");
    TORCH_CHECK(packed_qjl.dim() == 2,     "turbo_unpack: packed_qjl must be 2D");
    TORCH_CHECK(alpha.dim() == 2,          "turbo_unpack: alpha must be 2D");
    TORCH_CHECK(scale_error.dim() == 2,    "turbo_unpack: scale_error must be 2D");
    TORCH_CHECK(packed_ternary.size(0) == packed_qjl.size(0),
                "turbo_unpack: row mismatch between packed_ternary and packed_qjl");
    TORCH_CHECK(packed_ternary.size(0) == alpha.size(0),
                "turbo_unpack: row mismatch between packed_ternary and alpha");
    TORCH_CHECK(packed_ternary.size(0) == scale_error.size(0),
                "turbo_unpack: row mismatch between packed_ternary and scale_error");

    const int64_t rows = packed_ternary.size(0);
    const int64_t ternary_cols = packed_ternary.size(1);
    const int64_t in_features = ternary_cols * 4;
    TORCH_CHECK(in_features % group_size == 0,
                "turbo_unpack: unpacked width must be divisible by group_size");
    const int64_t n_groups = in_features / group_size;
    const int64_t qjl_cols = packed_qjl.size(1);
    TORCH_CHECK(qjl_cols * 8 == in_features,
                "turbo_unpack: packed_qjl last dim must equal unpacked width / 8");
    TORCH_CHECK(alpha.size(1) == n_groups,
                "turbo_unpack: alpha must have shape (rows, in_features / group_size)");
    TORCH_CHECK(scale_error.size(1) == n_groups,
                "turbo_unpack: scale_error must have shape (rows, in_features / group_size)");

    auto out_shape = packed_ternary.sizes().vec();
    out_shape.back() = in_features;
    auto out = torch::empty(
        out_shape,
        torch::TensorOptions().dtype(torch::kHalf).device(packed_ternary.device()));

    auto alpha_f = alpha.to(torch::kFloat).contiguous();
    auto err_f = scale_error.to(torch::kFloat).contiguous();

    const int64_t n_packed = packed_ternary.numel();
    constexpr int THREADS = 256;
    const int64_t blocks  = (n_packed + THREADS - 1) / THREADS;

    const at::cuda::OptionalCUDAGuard guard(packed_ternary.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    turbo_unpack_kernel_fp16<<<blocks, THREADS, 0, stream>>>(
        packed_ternary.data_ptr<uint8_t>(),
        packed_qjl.data_ptr<uint8_t>(),
        alpha_f.data_ptr<float>(),
        err_f.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n_packed,
        in_features,
        qjl_cols,
        n_groups,
        group_size);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Python bindings live in cracked_quant.cpp so that this file and
// fused_fwht.cu can be linked together into a single extension module.
