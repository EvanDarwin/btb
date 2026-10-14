// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The GGUF types a linear multiplies as stored (btb.kinds.stored_quants()), on the card: each type's blocks are
// read as the file holds them and dequantised in registers, never expanded to bf16 in memory.
//
//   btb_gemv_<key>_bf16_m{1,..,32}  y[m][r] = sum_c W[r][c] * x[m][c] over the raw blocks (bf16 x and y, f32
//                                   accumulation): btb_gemv_bf16's warp-a-row shape and fixed order, so row r of an
//                                   M-row pass is bit-for-bit the one-row step's; y's rows `ldy` apart, so the
//                                   pieces of a merged projection land side by side in one buffer
//   btb_dequant_<key>_bf16          the whole [R, C] weight as bf16, gguf's numbers (a long prompt's GEMM)
//
// <key> is kinds.quant_key (Q4_K -> q4k). A type's blocks are read through its `decode` (btb_quant_decode.cuh);
// every kernel takes the lattice types' codebook and sign table, null for the rest.

#include "btb_quant_decode.cuh"

// y[m][r] over the raw blocks of an [R, C] weight: one warp a row, lane l the weights at 32 * g + l of each
// 32-weight group g, decoded once and reused across the M rows; acc[m] runs the groups in order, independent of
// M. C a multiple of the type's block (a 32-block type's tail of fewer than 8 blocks taken a group at a time)
template <class Q, int M>
__device__ __forceinline__ void gemv_quant_rows(const uint8_t* __restrict__ w, QTab t, const bf16* __restrict__ x,
                                                bf16* __restrict__ y, int R, int C, int ldy) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int groups = C >> 5;
    const uint8_t* row = w + (size_t)r * (C / Q::BW) * Q::BB;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int g0 = 0; g0 < groups; g0 += 8) {
        const uint8_t* unit = row + (size_t)(g0 >> 3) * Q::UB;
        const int n = min(8, groups - g0);
        float wv[8];
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) wv[ib] = ib < n ? Q::decode(unit, ib, lane, t) : 0.f;
        const bf16* xc = x + (size_t)g0 * 32 + lane;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const bf16* xr = xc + (size_t)m * C;
            float a = acc[m];
#pragma unroll
            for (int ib = 0; ib < 8; ++ib)
                if (ib < n) a = fmaf(wv[ib], bf2f(xr[32 * ib]), a);
            acc[m] = a;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * ldy + r] = f2bf(acc[m]);
    }
}

// the [R, C] weight as bf16, a 256-thread block a 256-weight unit (a 32-block type's last unit may hold fewer)
template <class Q>
__device__ __forceinline__ void dequant_quant(const uint8_t* __restrict__ w, QTab t, bf16* __restrict__ out,
                                              long long n) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int ib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    out[i] = f2bf(Q::decode(w + (size_t)blockIdx.x * Q::UB, ib, lane, t));
}

// the entry points' parameters, each a role (btb_roles.cuh), and their bodies: a type's struct is named by its key,
// so a type's kernels name their decode
#define GEMV_QUANT_ARGS                                                                                       \
    const QuantBlocks* __restrict__ w, const QuantGrid* lattice, const QuantSigns* signs,                     \
        const MatIn* __restrict__ x, MatOut* __restrict__ y, OutDim R, InDim C, OutStride ldy
#define GEMV_QUANT_BODY(Q, M) \
    { gemv_quant_rows<Q, M>(w, QTab{lattice, signs}, x, y, R, C, ldy); }
#define DEQUANT_ARGS \
    const QuantBlocks* __restrict__ w, const QuantGrid* lattice, const QuantSigns* signs, Dequant* __restrict__ out, \
        QuantValues n
#define DEQUANT_BODY(Q) \
    { dequant_quant<Q>(w, QTab{lattice, signs}, out, n); }

// a type's gemv at every width: a macro a type, its kernels' names ending in the width (the bindings' one variant)
#define GEMV_Q40(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q40_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q40, M)
#define GEMV_Q41(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q41_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q41, M)
#define GEMV_Q80(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q80_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q80, M)
#define GEMV_IQ4NL(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq4nl_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq4nl, M)
#define GEMV_Q2K(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q2k_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q2k, M)
#define GEMV_Q3K(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q3k_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q3k, M)
#define GEMV_Q4K(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q4k_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q4k, M)
#define GEMV_Q5K(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q5k_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q5k, M)
#define GEMV_Q6K(M) \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q6k_bf16_m##M(GEMV_QUANT_ARGS) GEMV_QUANT_BODY(q6k, M)
#define GEMV_IQ4XS(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq4xs_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq4xs, M)
#define GEMV_IQ3XXS(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq3xxs_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq3xxs, M)
#define GEMV_IQ3S(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq3s_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq3s, M)
#define GEMV_IQ2XXS(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq2xxs_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq2xxs, M)
#define GEMV_IQ2XS(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq2xs_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq2xs, M)
#define GEMV_IQ2S(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq2s_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq2s, M)
#define GEMV_IQ1S(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq1s_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq1s, M)
#define GEMV_IQ1M(M)                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_iq1m_bf16_m##M(GEMV_QUANT_ARGS) \
        GEMV_QUANT_BODY(iq1m, M)

GEMV_Q40(1)
GEMV_Q40(2)
GEMV_Q40(4)
GEMV_Q40(8)
GEMV_Q40(16)
GEMV_Q40(32)
GEMV_Q41(1)
GEMV_Q41(2)
GEMV_Q41(4)
GEMV_Q41(8)
GEMV_Q41(16)
GEMV_Q41(32)
GEMV_Q80(1)
GEMV_Q80(2)
GEMV_Q80(4)
GEMV_Q80(8)
GEMV_Q80(16)
GEMV_Q80(32)
GEMV_IQ4NL(1)
GEMV_IQ4NL(2)
GEMV_IQ4NL(4)
GEMV_IQ4NL(8)
GEMV_IQ4NL(16)
GEMV_IQ4NL(32)
GEMV_Q2K(1)
GEMV_Q2K(2)
GEMV_Q2K(4)
GEMV_Q2K(8)
GEMV_Q2K(16)
GEMV_Q2K(32)
GEMV_Q3K(1)
GEMV_Q3K(2)
GEMV_Q3K(4)
GEMV_Q3K(8)
GEMV_Q3K(16)
GEMV_Q3K(32)
GEMV_Q4K(1)
GEMV_Q4K(2)
GEMV_Q4K(4)
GEMV_Q4K(8)
GEMV_Q4K(16)
GEMV_Q4K(32)
GEMV_Q5K(1)
GEMV_Q5K(2)
GEMV_Q5K(4)
GEMV_Q5K(8)
GEMV_Q5K(16)
GEMV_Q5K(32)
GEMV_Q6K(1)
GEMV_Q6K(2)
GEMV_Q6K(4)
GEMV_Q6K(8)
GEMV_Q6K(16)
GEMV_Q6K(32)
GEMV_IQ4XS(1)
GEMV_IQ4XS(2)
GEMV_IQ4XS(4)
GEMV_IQ4XS(8)
GEMV_IQ4XS(16)
GEMV_IQ4XS(32)
GEMV_IQ3XXS(1)
GEMV_IQ3XXS(2)
GEMV_IQ3XXS(4)
GEMV_IQ3XXS(8)
GEMV_IQ3XXS(16)
GEMV_IQ3XXS(32)
GEMV_IQ3S(1)
GEMV_IQ3S(2)
GEMV_IQ3S(4)
GEMV_IQ3S(8)
GEMV_IQ3S(16)
GEMV_IQ3S(32)
GEMV_IQ2XXS(1)
GEMV_IQ2XXS(2)
GEMV_IQ2XXS(4)
GEMV_IQ2XXS(8)
GEMV_IQ2XXS(16)
GEMV_IQ2XXS(32)
GEMV_IQ2XS(1)
GEMV_IQ2XS(2)
GEMV_IQ2XS(4)
GEMV_IQ2XS(8)
GEMV_IQ2XS(16)
GEMV_IQ2XS(32)
GEMV_IQ2S(1)
GEMV_IQ2S(2)
GEMV_IQ2S(4)
GEMV_IQ2S(8)
GEMV_IQ2S(16)
GEMV_IQ2S(32)
GEMV_IQ1S(1)
GEMV_IQ1S(2)
GEMV_IQ1S(4)
GEMV_IQ1S(8)
GEMV_IQ1S(16)
GEMV_IQ1S(32)
GEMV_IQ1M(1)
GEMV_IQ1M(2)
GEMV_IQ1M(4)
GEMV_IQ1M(8)
GEMV_IQ1M(16)
GEMV_IQ1M(32)

// a type's whole weight as bf16
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q40_bf16(DEQUANT_ARGS) DEQUANT_BODY(q40)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q41_bf16(DEQUANT_ARGS) DEQUANT_BODY(q41)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q80_bf16(DEQUANT_ARGS) DEQUANT_BODY(q80)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq4nl_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq4nl)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q2k_bf16(DEQUANT_ARGS) DEQUANT_BODY(q2k)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q3k_bf16(DEQUANT_ARGS) DEQUANT_BODY(q3k)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q4k_bf16(DEQUANT_ARGS) DEQUANT_BODY(q4k)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q5k_bf16(DEQUANT_ARGS) DEQUANT_BODY(q5k)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_q6k_bf16(DEQUANT_ARGS) DEQUANT_BODY(q6k)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq4xs_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq4xs)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq3xxs_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq3xxs)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq3s_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq3s)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq2xxs_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq2xxs)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq2xs_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq2xs)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq2s_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq2s)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq1s_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq1s)
extern "C" __global__ void __launch_bounds__(256) btb_dequant_iq1m_bf16(DEQUANT_ARGS) DEQUANT_BODY(iq1m)
