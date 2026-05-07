// Fused Fast Walsh-Hadamard Transform (FWHT) for FP16
// Optimised for NVIDIA Ampere (Compute Capability 8.6, e.g. RTX 3080 Ti)
//
// Strategy
// --------
//   * One warp transforms one row of length N (power of 2, N <= 1024).
//   * Thread t in a warp holds elements { t, t+32, t+64, ..., t+(K-1)*32 }
//     where K = max(1, N/32).  All values live in registers as FP32.
//   * Butterfly stages with stride s < 32 are executed with
//     __shfl_xor_sync, so the data never leaves the register file.
//   * Butterfly stages with stride s >= 32 are register-level swaps
//     between the K elements held by a single thread.
//   * The kernel is templated on LOG_N so every butterfly stage is
//     fully unrolled by nvcc -> tight, branch-free SASS.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>

namespace {

constexpr int      WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Templated FWHT kernel.
//   LOG_N          : log2(transform length), 0..10
//   ROWS_PER_BLOCK : number of warps (= rows) processed per CUDA block
// ---------------------------------------------------------------------------
template <int LOG_N, int ROWS_PER_BLOCK>
__global__ __launch_bounds__(WARP_SIZE * ROWS_PER_BLOCK)
void fwht_kernel_fp16(const __half* __restrict__ x,
                      __half*       __restrict__ y,
                      int   B,
                      float scale)
{
    constexpr int N            = 1 << LOG_N;
    constexpr int K            = (N < WARP_SIZE) ? 1 : (N / WARP_SIZE);
    constexpr int WARP_STAGES  = (LOG_N < 5) ? LOG_N : 5;

    const int warp_in_block = threadIdx.y;
    const int row           = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;
    if (row >= B) return;

    const int lane = threadIdx.x;                 // 0..31

    const __half* row_in  = x + row * N;
    __half*       row_out = y + row * N;

    // -----------------------------------------------------------------------
    // 1.  Coalesced load: thread `lane` reads halves at strided positions.
    //     For N >= 32 every lane participates and the 32 reads form a
    //     64-byte transaction (half a cache line) per K-iteration.
    // -----------------------------------------------------------------------
    float regs[K];

    if constexpr (N >= WARP_SIZE) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            regs[k] = __half2float(row_in[k * WARP_SIZE + lane]);
        }
    } else {
        // N < 32: only the first N lanes carry real data; the rest are
        // padded with zero so that shuffle butterflies remain harmless
        // (lanes >= N never partner with lanes < N for any stride < N).
        regs[0] = (lane < N) ? __half2float(row_in[lane]) : 0.0f;
    }

    // -----------------------------------------------------------------------
    // 2.  Intra-warp butterfly stages (strides 1, 2, 4, 8, 16).
    //     Both partners exchange their value through the shuffle, so each
    //     lane computes  out = (lane has higher index in pair) ? other - self
    //                                                          : self + other
    //     This realises the Sylvester-Hadamard recursion in registers.
    // -----------------------------------------------------------------------
    #pragma unroll
    for (int s = 0; s < WARP_STAGES; ++s) {
        const int bit = 1 << s;
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            const float other = __shfl_xor_sync(FULL_MASK, regs[k], bit, WARP_SIZE);
            regs[k] = (lane & bit) ? (other - regs[k]) : (regs[k] + other);
        }
    }

    // -----------------------------------------------------------------------
    // 3.  Inter-warp / register-level butterfly stages (strides 32, 64, ...).
    //     Both partners live in the same thread but at different K indices,
    //     so the butterfly becomes a pure register operation - no shuffle,
    //     no shared memory, no global traffic.
    // -----------------------------------------------------------------------
    if constexpr (LOG_N > 5) {
        #pragma unroll
        for (int s = 5; s < LOG_N; ++s) {
            const int reg_bit = 1 << (s - 5);
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                if ((k & reg_bit) == 0) {
                    const int   j = k | reg_bit;
                    const float a = regs[k];
                    const float b = regs[j];
                    regs[k] = a + b;
                    regs[j] = a - b;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 4.  Apply normalisation (1/sqrt(N) for the orthogonal Hadamard) and
    //     store back in FP16.  Same coalesced pattern as the load.
    // -----------------------------------------------------------------------
    if constexpr (N >= WARP_SIZE) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            row_out[k * WARP_SIZE + lane] = __float2half(regs[k] * scale);
        }
    } else {
        if (lane < N) {
            row_out[lane] = __float2half(regs[0] * scale);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Host-side launcher.  Accepts any tensor whose last dim is a power of 2
// up to 1024; all leading dims are flattened into the batch axis.
// ---------------------------------------------------------------------------
torch::Tensor fwht_forward(torch::Tensor x, bool normalize)
{
    TORCH_CHECK(x.is_cuda(),                    "fwht: input must live on CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf,   "fwht: input must be float16 (at::Half)");
    TORCH_CHECK(x.is_contiguous(),              "fwht: input must be contiguous");
    TORCH_CHECK(x.dim() >= 1,                   "fwht: input must have at least 1 dim");

    const int64_t N = x.size(-1);
    TORCH_CHECK(N >= 1 && N <= 1024,            "fwht: last dim must satisfy 1 <= N <= 1024");
    TORCH_CHECK((N & (N - 1)) == 0,             "fwht: last dim must be a power of 2");

    int log_n = 0;
    while ((1 << log_n) < N) ++log_n;

    auto y = torch::empty_like(x);

    const int64_t B     = x.numel() / N;
    const float   scale = normalize ? (1.0f / std::sqrt(static_cast<float>(N))) : 1.0f;

    constexpr int ROWS_PER_BLOCK = 4;            // 4 warps = 128 threads / block

    const dim3 block(WARP_SIZE, ROWS_PER_BLOCK);
    const dim3 grid(static_cast<unsigned>((B + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK));

    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const __half* xp = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
    __half*       yp = reinterpret_cast<__half*>(y.data_ptr<at::Half>());

#define FWHT_DISPATCH(LN)                                                      \
    case LN:                                                                   \
        fwht_kernel_fp16<LN, ROWS_PER_BLOCK>                                   \
            <<<grid, block, 0, stream>>>(xp, yp,                               \
                                         static_cast<int>(B), scale);          \
        break

    switch (log_n) {
        FWHT_DISPATCH(0);
        FWHT_DISPATCH(1);
        FWHT_DISPATCH(2);
        FWHT_DISPATCH(3);
        FWHT_DISPATCH(4);
        FWHT_DISPATCH(5);
        FWHT_DISPATCH(6);
        FWHT_DISPATCH(7);
        FWHT_DISPATCH(8);
        FWHT_DISPATCH(9);
        FWHT_DISPATCH(10);
        default:
            TORCH_CHECK(false, "fwht: unsupported LOG_N=", log_n);
    }
#undef FWHT_DISPATCH

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Python bindings live in cracked_quant.cpp so that this file and
// ternary_packing.cu can be linked together into a single extension module.
