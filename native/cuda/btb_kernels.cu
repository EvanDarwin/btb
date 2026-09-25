// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The card's decode kernels. Every reduction runs in an order fixed by the element's index alone - never by
// the number of rows a pass carries, the cache's length or the launch's timing - so a verify pass of T rows
// computes each row bit-for-bit as the one-row step does, and one captured graph serves every length.
//
//   btb_gemv_bf16_m{1,..,32}      y[m][r] = sum_c w[r][c] * x[m][c]      (bf16 in, f32 accumulate, bf16 out)
//   btb_gemv_q{2,3,4,5,6}k_bf16_m{1,..,32}  the same over raw k-quant weights (ggml's superblocks as stored),
//                                 dequantised in registers; the same warp-a-row shape and fixed order
//   btb_gemv_{silu,gelu}_bf16_m{1,..,32}  the down projection with act(g) * u folded into its x load
//   btb_attn_split_d{64,128,256}  one pass of T queries over the cache (a sliding layer's window of it), a tree of
//                                 T rows at its end, split over the sequence
//   btb_norm_rope_kv_d{64,128,256} q/k RMSNorm, rope, the pass's rows written into the cache
//   btb_add_rmsnorm               h += y; x = rmsnorm(h) * w   (or * (1 + w), the zero-centred norm)
//   btb_sandwich_add              h += rmsnorm(y) * w          (the sandwich block: the delta normed, then added)
//   btb_{silu,gelu}_mul           m = act(g) * u  over a [T, 2I] gate/up block
//
// The weights stream with evict-first loads (read once a step); the cache and the activations take the
// default policy, so a persisting-L2 window over the cache's front keeps a short context's attention in L2.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace cg = cooperative_groups;

typedef __nv_bfloat16 bf16;

#define NEG_INF (__int_as_float(0xff800000))

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16_rn(v); }
__device__ __forceinline__ float bfround(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// a lane's E consecutive bf16 of a row as one vector load (E = 2, 4 or 8: 4, 8 or 16 bytes)
template <int E>
struct Vec;
template <>
struct Vec<2> {
    unsigned v;
    __device__ __forceinline__ void load(const bf16* p) { v = *reinterpret_cast<const unsigned*>(p); }
    __device__ __forceinline__ float at(int e) const { return bf2f(reinterpret_cast<const bf16*>(&v)[e]); }
};
template <>
struct Vec<4> {
    uint2 v;
    __device__ __forceinline__ void load(const bf16* p) { v = *reinterpret_cast<const uint2*>(p); }
    __device__ __forceinline__ float at(int e) const { return bf2f(reinterpret_cast<const bf16*>(&v)[e]); }
};
template <>
struct Vec<8> {
    uint4 v;
    __device__ __forceinline__ void load(const bf16* p) { v = *reinterpret_cast<const uint4*>(p); }
    __device__ __forceinline__ float at(int e) const { return bf2f(reinterpret_cast<const bf16*>(&v)[e]); }
};

__device__ __forceinline__ float warp_sum(float a) {
    a += __shfl_xor_sync(0xffffffffu, a, 16);
    a += __shfl_xor_sync(0xffffffffu, a, 8);
    a += __shfl_xor_sync(0xffffffffu, a, 4);
    a += __shfl_xor_sync(0xffffffffu, a, 2);
    a += __shfl_xor_sync(0xffffffffu, a, 1);
    return a;
}

// ---------------------------------------------------------------------------------------------------------
// gemv: w [R, C] bf16 row-major (C a multiple of 8), x [M, C] bf16, y [M, R] bf16. One warp per row; lane l
// takes chunks l, l + 32, ... of 8 elements, four chunks in flight; acc[m] runs the chunks in that order and a
// chunk's 8 elements in index order, then the butterfly over the lanes. The sequence for (m, r) does not
// depend on M, so row r of a 16-row pass equals row r of the 1-row step.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_rows(const bf16* __restrict__ w, const bf16* __restrict__ x,
                                          bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nchunk = C >> 3;
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + (size_t)r * nchunk;
    const uint4* xrow = reinterpret_cast<const uint4*>(x);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    int c = lane;
    // four chunks in flight per lane
    for (; c + 96 < nchunk; c += 128) {
        uint4 wv0 = __ldcs(wrow + c);
        uint4 wv1 = __ldcs(wrow + c + 32);
        uint4 wv2 = __ldcs(wrow + c + 64);
        uint4 wv3 = __ldcs(wrow + c + 96);
        const uint4 wv[4] = {wv0, wv1, wv2, wv3};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const bf16* wp = reinterpret_cast<const bf16*>(&wv[k]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                uint4 xv = __ldg(xrow + (size_t)m * nchunk + c + 32 * k);
                const bf16* xp = reinterpret_cast<const bf16*>(&xv);
                float a = acc[m];
#pragma unroll
                for (int e = 0; e < 8; ++e) a = fmaf(bf2f(wp[e]), bf2f(xp[e]), a);
                acc[m] = a;
            }
        }
    }
    for (; c < nchunk; c += 32) {
        uint4 wv = __ldcs(wrow + c);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 xv = __ldg(xrow + (size_t)m * nchunk + c);
            const bf16* xp = reinterpret_cast<const bf16*>(&xv);
            float a = acc[m];
#pragma unroll
            for (int e = 0; e < 8; ++e) a = fmaf(bf2f(wp[e]), bf2f(xp[e]), a);
            acc[m] = a;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV(M)                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_bf16_m##M(                                    \
        const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {         \
        gemv_rows<M>(w, x, y, R, C);                                                                         \
    }
GEMV(1)
GEMV(2)
GEMV(4)
GEMV(8)
GEMV(16)
GEMV(32)

// the MLP's gate activation in fp32 as torch computes it: silu x / (1 + e^-x), or gelu in its tanh form,
// 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) (Gemma's gelu_pytorch_tanh)
__device__ __forceinline__ float act_silu(float g) { return g / (1.f + expf(-g)); }
__device__ __forceinline__ float act_gelu(float g) {
    return 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
}
template <int ACT>
__device__ __forceinline__ float act_of(float g) {
    return ACT == 0 ? act_silu(g) : act_gelu(g);
}

// the down projection with act(g) * up folded into its x load: gu [M, 2C] (gate then up), x[m][c] =
// bf16(bf16(act(g)) * u) - the same values, bit for bit, as btb_{silu,gelu}_mul would have written, so the
// kernel and the buffer between them go away. The elementwise math is per lane, per element, in index order.
template <int M, int ACT>
__device__ __forceinline__ void gemv_act_rows(const bf16* __restrict__ w, const bf16* __restrict__ gu,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nchunk = C >> 3;
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + (size_t)r * nchunk;
    const uint4* grow = reinterpret_cast<const uint4*>(gu);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        uint4 wv = __ldcs(wrow + c);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 gv = __ldg(grow + (size_t)m * 2 * nchunk + c);
            uint4 uv = __ldg(grow + (size_t)m * 2 * nchunk + nchunk + c);
            const bf16* gp = reinterpret_cast<const bf16*>(&gv);
            const bf16* up = reinterpret_cast<const bf16*>(&uv);
            float a = acc[m];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float g = bf2f(gp[e]);
                const float s = bfround(act_of<ACT>(g));
                const float xe = bfround(s * bf2f(up[e]));
                a = fmaf(bf2f(wp[e]), xe, a);
            }
            acc[m] = a;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_ACT(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_silu_bf16_m##M(                               \
        const bf16* __restrict__ w, const bf16* __restrict__ gu, bf16* __restrict__ y, int R, int C) {        \
        gemv_act_rows<M, 0>(w, gu, y, R, C);                                                                 \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_gelu_bf16_m##M(                               \
        const bf16* __restrict__ w, const bf16* __restrict__ gu, bf16* __restrict__ y, int R, int C) {        \
        gemv_act_rows<M, 1>(w, gu, y, R, C);                                                                 \
    }
GEMV_ACT(1)
GEMV_ACT(2)
GEMV_ACT(4)
GEMV_ACT(8)
GEMV_ACT(16)
GEMV_ACT(32)

// ---------------------------------------------------------------------------------------------------------
// gemv over Q4_K weights: w the raw ggml Q4_K bytes [R, C/256] row-major superblocks of 144 B, x [M, C] bf16,
// y [M, R] bf16. One warp a row; lane l is the inner index 0..31, so the 32 lanes cooperate on a superblock
// (the byte read qs[32k + l] and the two x reads at ...+ l are each 32 contiguous values, coalesced) and each
// lane owns eight of the 256 weights. The decode is llama.cpp's (144 B: d f16, dmin f16, 12 B of 6-bit
// scale/min pairs read by get_scale_min_k4, then 128 B of nibbles), dequantised in registers - the 4-bit
// weights never expand to bf16. value = d*sc[j]*q - dmin*mn[j], j = weight >> 5. acc[m] runs the blocks and
// the (k, lo/hi) pairs in a fixed order, independent of M, so row r of a T-row pass equals row r of the
// one-row step, as the bf16 gemv above.
// ---------------------------------------------------------------------------------------------------------
// the 6-bit scale and min of sub-block j (0..7) from the 12 packed bytes, llama.cpp's get_scale_min_k4
__device__ __forceinline__ void q4k_scale_min(const uint8_t* s, int j, float& sc, float& mn) {
    if (j < 4) {
        sc = (float)(s[j] & 63);
        mn = (float)(s[j + 4] & 63);
    } else {
        sc = (float)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}

template <int M>
__device__ __forceinline__ void gemv_q4k_rows(const uint8_t* __restrict__ w, const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nsb = C >> 8;  // superblocks of 256 weights
    const uint8_t* base = w + (size_t)r * nsb * 144;
    // NVIDIA layout: lane l owns one uint (4 nibble bytes, 8 weights) of every superblock - bytes [4l, 4l+3],
    // all in sub-block pair k = l >> 3 at inner offset (l & 7) * 4 - so a superblock's 128 nibble bytes are one
    // coalesced 128 B __ldcs a warp (evict-first, the weight stream never evicts the cache's persisting-L2
    // window) and each lane's scale pair unpacks once. The eight weights are decoded once and reused across the
    // M rows; x rides the default policy as a uint2 (4 bf16). acc runs the blocks and the (lo/hi, i) in a fixed
    // order independent of M, so row r of a T-row pass equals row r of the one-row step.
    const int k = lane >> 3;
    const int inner = (lane & 7) << 2;
    const int colL = (2 * k) * 32 + inner;
    const int colH = (2 * k + 1) * 32 + inner;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int c = 0; c < nsb; ++c) {
        const uint8_t* blk = base + (size_t)c * 144;
        const unsigned hdr = __ldcs(reinterpret_cast<const unsigned*>(blk));  // d in the low half, dmin in the high
        const float d = __half2float(__ushort_as_half((unsigned short)(hdr & 0xFFFF)));
        const float dm = __half2float(__ushort_as_half((unsigned short)(hdr >> 16)));
        const uint8_t* s = blk + 4;
        float sclo, mnlo, schi, mnhi;
        q4k_scale_min(s, 2 * k, sclo, mnlo);
        q4k_scale_min(s, 2 * k + 1, schi, mnhi);
        const unsigned packed = __ldcs(reinterpret_cast<const unsigned*>(blk + 16 + (lane << 2)));
        float wl[4], wh[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte = (packed >> (8 * i)) & 0xFF;
            wl[i] = d * sclo * (float)(byte & 0xF) - dm * mnlo;
            wh[i] = d * schi * (float)(byte >> 4) - dm * mnhi;
        }
        const bf16* xc = x + (size_t)c * 256;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const bf16* xr = xc + (size_t)m * C;
            Vec<4> xl, xh;
            xl.load(xr + colL);
            xh.load(xr + colH);
            float a = acc[m];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a = fmaf(wl[i], xl.at(i), a);
                a = fmaf(wh[i], xh.at(i), a);
            }
            acc[m] = a;
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_Q4K(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q4k_bf16_m##M(                                \
        const uint8_t* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {      \
        gemv_q4k_rows<M>(w, x, y, R, C);                                                                     \
    }
GEMV_Q4K(1)
GEMV_Q4K(2)
GEMV_Q4K(4)
GEMV_Q4K(8)
GEMV_Q4K(16)
GEMV_Q4K(32)

// ---------------------------------------------------------------------------------------------------------
// gemv over Q5_K weights: 176 B/superblock - d f16, dmin f16, 12 scale bytes (get_scale_min_k4, as Q4_K), a 32
// B qh plane (the 5th bit of each weight), then 128 B of low nibbles. q = nibble | (qh_bit << 4); value =
// d*sc[j]*q - dmin*mn[j]. lane = inner index 0..31 (the qh plane and the nibble sub-blocks index by it), so the
// qs/qh and x reads are coalesced across the warp; qh[lane] is read once and each (k) weight pair is decoded
// once and reused across the M rows, __ldcs evict-first. Fixed accumulation order, batch-invariant.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_q5k_rows(const uint8_t* __restrict__ w, const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nsb = C >> 8;
    const uint8_t* baseW = w + (size_t)r * nsb * 176;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int c = 0; c < nsb; ++c) {
        const uint8_t* blk = baseW + (size_t)c * 176;
        const unsigned hdr = __ldcs(reinterpret_cast<const unsigned*>(blk));
        const float d = __half2float(__ushort_as_half((unsigned short)(hdr & 0xFFFF)));
        const float dm = __half2float(__ushort_as_half((unsigned short)(hdr >> 16)));
        const uint8_t* s = blk + 4;
        const uint8_t hbit = __ldcs(blk + 16 + lane);
        const uint8_t* qs = blk + 48;
        const bf16* xc = x + (size_t)c * 256;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float sclo, mnlo, schi, mnhi;
            q4k_scale_min(s, 2 * k, sclo, mnlo);
            q4k_scale_min(s, 2 * k + 1, schi, mnhi);
            const uint8_t byte = __ldcs(qs + 32 * k + lane);
            const int qlo = (byte & 0xF) | (((hbit >> (2 * k)) & 1) << 4);
            const int qhi = (byte >> 4) | (((hbit >> (2 * k + 1)) & 1) << 4);
            const float wlo = d * sclo * (float)qlo - dm * mnlo;
            const float whi = d * schi * (float)qhi - dm * mnhi;
            const int clo = (2 * k) * 32 + lane;
            const int chi = (2 * k + 1) * 32 + lane;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const bf16* xr = xc + (size_t)m * C;
                acc[m] = fmaf(whi, bf2f(xr[chi]), fmaf(wlo, bf2f(xr[clo]), acc[m]));
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_Q5K(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q5k_bf16_m##M(                                \
        const uint8_t* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {      \
        gemv_q5k_rows<M>(w, x, y, R, C);                                                                     \
    }
GEMV_Q5K(1)
GEMV_Q5K(2)
GEMV_Q5K(4)
GEMV_Q5K(8)
GEMV_Q5K(16)
GEMV_Q5K(32)

// ---------------------------------------------------------------------------------------------------------
// gemv over Q6_K weights: 210 B/superblock - ql[128] low nibbles, qh[64] the high 2 bits, scales[16] int8,
// d f16. value = d * sc[is] * (q - 32), q the 6-bit weight. lane = inner index 0..31; each lane owns eight
// weights (four per half h), ql/qh read __ldcs evict-first and coalesced across the warp, the eight weights
// decoded once and reused across the M rows. Fixed accumulation order, batch-invariant.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_q6k_rows(const uint8_t* __restrict__ w, const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nsb = C >> 8;
    const uint8_t* baseW = w + (size_t)r * nsb * 210;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int c = 0; c < nsb; ++c) {
        const uint8_t* blk = baseW + (size_t)c * 210;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
        const float df = __half2float(__ushort_as_half((unsigned short)__ldcs(reinterpret_cast<const unsigned short*>(blk + 208))));
        const bf16* xc = x + (size_t)c * 256;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int o = h * 128, qlo = h * 64, qho = h * 32, sco = h * 8;
            const int isc = sco + (lane >> 4);
            const uint8_t l0 = __ldcs(ql + qlo + lane);
            const uint8_t l1 = __ldcs(ql + qlo + lane + 32);
            const uint8_t hb = __ldcs(qh + qho + lane);
            const float w1 = df * (float)sc[isc + 0] * (float)((int)((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32);
            const float w2 = df * (float)sc[isc + 2] * (float)((int)((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32);
            const float w3 = df * (float)sc[isc + 4] * (float)((int)((l0 >> 4) | (((hb >> 4) & 3) << 4)) - 32);
            const float w4 = df * (float)sc[isc + 6] * (float)((int)((l1 >> 4) | (((hb >> 6) & 3) << 4)) - 32);
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const bf16* xr = xc + (size_t)m * C + o;
                float a = acc[m];
                a = fmaf(w1, bf2f(xr[lane]), a);
                a = fmaf(w2, bf2f(xr[lane + 32]), a);
                a = fmaf(w3, bf2f(xr[lane + 64]), a);
                a = fmaf(w4, bf2f(xr[lane + 96]), a);
                acc[m] = a;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_Q6K(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q6k_bf16_m##M(                                \
        const uint8_t* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {      \
        gemv_q6k_rows<M>(w, x, y, R, C);                                                                     \
    }
GEMV_Q6K(1)
GEMV_Q6K(2)
GEMV_Q6K(4)
GEMV_Q6K(8)
GEMV_Q6K(16)
GEMV_Q6K(32)

// ---------------------------------------------------------------------------------------------------------
// gemv over Q2_K weights: 84 B/superblock - scales[16] (a 4-bit scale and 4-bit min a 16-weight sub-block) @0,
// qs[64] (2-bit) @16, d f16 @80, dmin f16 @82. value = d*(s&0xF)*q - dmin*(s>>4). lane = inner index; sub =
// lane>>4 picks the 16-weight sub-block. qs coalesced across the warp, __ldcs; the four weights of a half are
// decoded once and reused across the M rows. Fixed accumulation order, batch-invariant.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_q2k_rows(const uint8_t* __restrict__ w, const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nsb = C >> 8;
    const uint8_t* baseW = w + (size_t)r * nsb * 84;
    const int sub = lane >> 4;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int c = 0; c < nsb; ++c) {
        const uint8_t* blk = baseW + (size_t)c * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        const unsigned dd = __ldcs(reinterpret_cast<const unsigned*>(blk + 80));  // d low half, dmin high half
        const float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFF)));
        const float dm = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
        const bf16* xc = x + (size_t)c * 256;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t byte = __ldcs(qs + h * 32 + lane);
            float wj[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t s = scales[h * 8 + 2 * j + sub];
                wj[j] = d * (float)(s & 0xF) * (float)((byte >> (2 * j)) & 3) - dm * (float)(s >> 4);
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const bf16* xr = xc + (size_t)m * C + h * 128 + lane;
                float a = acc[m];
#pragma unroll
                for (int j = 0; j < 4; ++j) a = fmaf(wj[j], bf2f(xr[j * 32]), a);
                acc[m] = a;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_Q2K(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q2k_bf16_m##M(                                \
        const uint8_t* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {      \
        gemv_q2k_rows<M>(w, x, y, R, C);                                                                     \
    }
GEMV_Q2K(1)
GEMV_Q2K(2)
GEMV_Q2K(4)
GEMV_Q2K(8)
GEMV_Q2K(16)
GEMV_Q2K(32)

// the 16 signed 6-bit scales of a Q3_K superblock from its 12 packed bytes (llama.cpp's kmask dance), as four
// uints whose bytes are the scales 0..63 (read (scl >> 32) then - 32 for the value). The 12 bytes are read one
// at a time - blk + 96 is not 4-aligned for every superblock (110 B blocks) - so no unaligned word load.
__device__ __forceinline__ void q3k_scales(const uint8_t* sc, unsigned aux[4]) {
    const unsigned a0 = (unsigned)sc[0] | ((unsigned)sc[1] << 8) | ((unsigned)sc[2] << 16) | ((unsigned)sc[3] << 24);
    const unsigned a1 = (unsigned)sc[4] | ((unsigned)sc[5] << 8) | ((unsigned)sc[6] << 16) | ((unsigned)sc[7] << 24);
    const unsigned a2 = (unsigned)sc[8] | ((unsigned)sc[9] << 8) | ((unsigned)sc[10] << 16) | ((unsigned)sc[11] << 24);
    const unsigned k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    aux[2] = ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4);
    aux[3] = ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4);
    aux[0] = (a0 & k2) | (((a2 >> 0) & k1) << 4);
    aux[1] = (a1 & k2) | (((a2 >> 2) & k1) << 4);
}

// ---------------------------------------------------------------------------------------------------------
// gemv over Q3_K weights: 110 B/superblock - hmask[32] (the 3rd bit) @0, qs[64] (the low 2 bits) @32, 12 scale
// bytes @96, d f16 @108. q = ((qs[h*32+l] >> 2j) & 3) - (hmask bit set ? 0 : 4); value = d*(scale-32)*q, the 16
// signed 6-bit scales unpacked once per superblock. lane = inner; sub = lane>>4. hmask/qs coalesced, __ldcs;
// decode reused across the M rows. Fixed accumulation order, batch-invariant.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_q3k_rows(const uint8_t* __restrict__ w, const bf16* __restrict__ x,
                                              bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nsb = C >> 8;
    const uint8_t* baseW = w + (size_t)r * nsb * 110;
    const int sub = lane >> 4;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll 4
    for (int c = 0; c < nsb; ++c) {
        const uint8_t* blk = baseW + (size_t)c * 110;
        unsigned aux[4];
        q3k_scales(blk + 96, aux);
        const float d = __half2float(__ushort_as_half((unsigned short)__ldcs(reinterpret_cast<const unsigned short*>(blk + 108))));
        const uint8_t hm = __ldcs(blk + lane);
        const uint8_t* qs = blk + 32;
        const bf16* xc = x + (size_t)c * 256;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t byte = __ldcs(qs + h * 32 + lane);
            float wj[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int q = (int)((byte >> (2 * j)) & 3);
                if (!(hm & (1 << (h * 4 + j)))) q -= 4;
                const int idx = h * 8 + 2 * j + sub;
                const int sval = (int)((aux[idx >> 2] >> (8 * (idx & 3))) & 0xFF);
                wj[j] = d * (float)(sval - 32) * (float)q;
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const bf16* xr = xc + (size_t)m * C + h * 128 + lane;
                float a = acc[m];
#pragma unroll
                for (int j = 0; j < 4; ++j) a = fmaf(wj[j], bf2f(xr[j * 32]), a);
                acc[m] = a;
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = warp_sum(acc[m]);
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = f2bf(acc[m]);
    }
}

#define GEMV_Q3K(M)                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_q3k_bf16_m##M(                                \
        const uint8_t* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C) {      \
        gemv_q3k_rows<M>(w, x, y, R, C);                                                                     \
    }
GEMV_Q3K(1)
GEMV_Q3K(2)
GEMV_Q3K(4)
GEMV_Q3K(8)
GEMV_Q3K(16)
GEMV_Q3K(32)

// ---------------------------------------------------------------------------------------------------------
// attention: q [T, Hq, D] bf16 (normed, roped), K/V [Hk, cap, D] bf16 (the cache layer's own buffer: row j
// of head g at (g * cap + j) * D), out [T, Hq, D] bf16. *n0 rows stand before the pass; the pass's own T rows
// sit at slots n0 .. n0+T-1, row u parented to par[u] (-1 at the root), and query t sees the prefix, its
// ancestors and itself. Block (h, t) of 8 warps walks the keys in LOGICAL order - position j: the prefix row
// j for j < n0, then the ancestor of t at depth j - n0 (whatever slot holds it) - key j to warp (j >> 5) & 7,
// each warp in increasing j with an online softmax, warp 0 folding the eight partial states in a fixed order.
// That is exactly the sequence a one-row step at the same position walks, so a verify pass over any tree
// gives the bits of the one-row steps of its accepted path. The kernel below is that walk split over the
// sequence; the one-block-per-head form it grew from is gone (equal at short contexts, 2-4x slower past 4K).
// ---------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------
// the attention above, split over the sequence: block (h, t, s) walks the logical keys [s*ATTN_SPLIT, (s+1)*
// ATTN_SPLIT) of query t exactly as above (key -> warp by logical index, warps folded in order), stores its
// state, and the last block to arrive for (h, t) folds the S states in split order and writes the row. The
// blocks past the sequence's end store an empty state and take part in the count, so S is fixed per graph
// (the arena's capacity over the split length) and the result depends on the key set alone - one graph,
// every length, the bits of the one-row steps at long contexts too, where one block per head left the card
// latency-bound. `part_*` hold S x T x Hq states, `cnt` T x Hq arrival counts (zero between launches: the
// last block resets its own).
// ---------------------------------------------------------------------------------------------------------
#define ATTN_SPLIT 1024

template <int D>
__device__ __forceinline__ void attn_decode_split(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                                  const bf16* __restrict__ V, bf16* __restrict__ out,
                                                  const int* __restrict__ n0p, const int* __restrict__ par,
                                                  int T, int Hq, int Hk, int cap, float scale,
                                                  float* __restrict__ part_m, float* __restrict__ part_l,
                                                  float* __restrict__ part_acc, int* __restrict__ cnt, int S,
                                                  int win) {
    constexpr int E = D / 32;
    const int h = blockIdx.x, t = blockIdx.y, s = blockIdx.z;
    const int g = h / (Hq / Hk);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int n0 = *n0p;
    // no query of this pass reaches past n0 + T rows: a block whose range starts beyond that leaves before
    // the walk and the barrier, so at a short context the launch costs what the unsplit kernel costs
    if (s * ATTN_SPLIT >= n0 + T) return;
    __shared__ int anc[32];
    __shared__ int s_d;
    if (threadIdx.x == 0) {
        int tmp[32];
        int len = 0;
        for (int p = t; p >= 0 && p < T && len < 32; p = par[p]) tmp[len++] = p;
        for (int e = 0; e < len; ++e) anc[e] = tmp[len - 1 - e];
        s_d = len - 1;
    }
    __syncthreads();
    const int d = s_d;
    float qf[E];
    const bf16* qp = q + ((size_t)t * Hq + h) * D + lane * E;
#pragma unroll
    for (int e = 0; e < E; ++e) qf[e] = bf2f(qp[e]);
    constexpr size_t rowstride = D;
    const bf16* Kg = K + (size_t)g * cap * D + lane * E;
    const bf16* Vg = V + (size_t)g * cap * D + lane * E;
    float m = NEG_INF, l = 0.f, acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
    const int n = n0 + d + 1;
    // a sliding layer (win > 0) sees the last win logical keys, [first, n): the walk below keeps its key -> warp
    // map and passes over the keys before first, so a windowed row folds as the full row's tail would
    const int first = win > 0 ? max(n - win, 0) : 0;
    // the splits this query needs: a single one writes its row directly, without the partials
    const int S_active = (n + ATTN_SPLIT - 1) / ATTN_SPLIT;
    if (s >= S_active) return;
    const int lo = s * ATTN_SPLIT, hi = min(lo + ATTN_SPLIT, n);
    for (int j0 = lo + w * 32; j0 < hi; j0 += 256) {
        const int j1 = min(j0 + 32, hi);
        for (int j = j0; j < j1; j += 4) {
            const int c4 = min(4, j1 - j);
            Vec<E> kv[4], vv[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < c4 && j + u >= first) {
                    const int jj = j + u;
                    const int slot = jj < n0 ? jj : n0 + anc[jj - n0];
                    kv[u].load(Kg + (size_t)slot * rowstride);
                    vv[u].load(Vg + (size_t)slot * rowstride);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < c4 && j + u >= first) {
                    float sc = 0.f;
#pragma unroll
                    for (int e = 0; e < E; ++e) sc = fmaf(qf[e], kv[u].at(e), sc);
                    sc = warp_sum(sc) * scale;
                    const float mn = fmaxf(m, sc);
                    const float corr = expf(m - mn);
                    const float p = expf(sc - mn);
                    l = l * corr + p;
#pragma unroll
                    for (int e = 0; e < E; ++e) acc[e] = fmaf(p, vv[u].at(e), acc[e] * corr);
                    m = mn;
                }
            }
        }
    }
    __shared__ float sm_m[8], sm_l[8];
    __shared__ float sm_acc[8][D];
    if (lane == 0) {
        sm_m[w] = m;
        sm_l[w] = l;
    }
#pragma unroll
    for (int e = 0; e < E; ++e) sm_acc[w][lane * E + e] = acc[e];
    __syncthreads();
    if (w != 0) return;
    // the block's state: the eight warps folded in order
    float M = sm_m[0], L = sm_l[0], o[E];
#pragma unroll
    for (int e = 0; e < E; ++e) o[e] = sm_acc[0][lane * E + e];
    for (int u = 1; u < 8; ++u) {
        const float mu = sm_m[u];
        if (mu == NEG_INF) continue;
        if (M == NEG_INF) {
            M = mu;
            L = sm_l[u];
#pragma unroll
            for (int e = 0; e < E; ++e) o[e] = sm_acc[u][lane * E + e];
            continue;
        }
        const float Mn = fmaxf(M, mu);
        const float c0 = expf(M - Mn), c1 = expf(mu - Mn);
        L = L * c0 + sm_l[u] * c1;
#pragma unroll
        for (int e = 0; e < E; ++e) o[e] = o[e] * c0 + sm_acc[u][lane * E + e] * c1;
        M = Mn;
    }
    const size_t row = ((size_t)t * Hq + h);
    if (S_active == 1) {
        bf16* op = out + row * D + lane * E;
        const float inv = 1.f / L;
#pragma unroll
        for (int e = 0; e < E; ++e) op[e] = f2bf(o[e] * inv);
        return;
    }
    const size_t idx = (size_t)s * T * Hq + row;
    if (lane == 0) {
        part_m[idx] = M;
        part_l[idx] = L;
    }
#pragma unroll
    for (int e = 0; e < E; ++e) part_acc[idx * D + lane * E + e] = o[e];
    __threadfence();
    __syncwarp();  // every lane's partial stores precede lane 0's arrival: the last arriver folds a whole state
    int last = 0;
    if (lane == 0) last = (atomicAdd(cnt + row, 1) == S_active - 1) ? 1 : 0;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();
    // the last block for (h, t): the active states in split order, read past L1
    M = NEG_INF;
    L = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) o[e] = 0.f;
    for (int u = 0; u < S_active; ++u) {
        const size_t iu = (size_t)u * T * Hq + row;
        const float mu = __ldcg(part_m + iu);
        if (mu == NEG_INF) continue;
        const float lu = __ldcg(part_l + iu);
        if (M == NEG_INF) {
            M = mu;
            L = lu;
#pragma unroll
            for (int e = 0; e < E; ++e) o[e] = __ldcg(part_acc + iu * D + lane * E + e);
            continue;
        }
        const float Mn = fmaxf(M, mu);
        const float c0 = expf(M - Mn), c1 = expf(mu - Mn);
        L = L * c0 + lu * c1;
#pragma unroll
        for (int e = 0; e < E; ++e) o[e] = o[e] * c0 + __ldcg(part_acc + iu * D + lane * E + e) * c1;
        M = Mn;
    }
    bf16* op = out + row * D + lane * E;
    const float inv = 1.f / L;
#pragma unroll
    for (int e = 0; e < E; ++e) op[e] = f2bf(o[e] * inv);
    if (lane == 0) cnt[row] = 0;
}

#define ATTN_SPLIT_K(D)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256) btb_attn_split_d##D(                                   \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ n0p, const int* __restrict__ par, int T, int Hq,     \
        int Hk, int cap, float scale, float* __restrict__ part_m, float* __restrict__ part_l,                \
        float* __restrict__ part_acc, int* __restrict__ cnt, int S, int win) {                               \
        attn_decode_split<D>(q, K, V, out, n0p, par, T, Hq, Hk, cap, scale, part_m, part_l, part_acc, cnt, S,   \
                             win);                                                                           \
    }
ATTN_SPLIT_K(64)
ATTN_SPLIT_K(128)
ATTN_SPLIT_K(256)

// ---------------------------------------------------------------------------------------------------------
// norm + rope + cache write: qkv [T, (Hq + 2 Hk) * D] bf16 (the merged projection's output), wq/wk [D] the
// q/k norm weights (null: no norm), cos/sin [positions, D] bf16 (the model's own rotary tables), the rows'
// positions n0 + depth[t], the cache slots n0 + t of K/V [Hk, cap, D]. One warp per (head, t): lanes 0-15 hold the first half of
// the head's dims and 16-31 the second, so rotate_half is a shuffle with lane ^ 16. The norm is the fused
// F.rms_norm (fp32 throughout, one rounding), the rope the engine's fused form: q*cos rounded to bf16, then
// one fused multiply-add with the rotated half and sin, rounded once. Warps past Hq + Hk copy the v rows.
// ---------------------------------------------------------------------------------------------------------
template <int D>
__device__ __forceinline__ void norm_rope_kv(const bf16* __restrict__ qkv, const bf16* __restrict__ wq,
                                             const bf16* __restrict__ wk, float eps, const bf16* __restrict__ cos_t,
                                             const bf16* __restrict__ sin_t, const int* __restrict__ n0p,
                                             const int* __restrict__ depth, bf16* __restrict__ K,
                                             bf16* __restrict__ V, bf16* __restrict__ qo, int T, int Hq, int Hk,
                                             int cap, int centered) {
    constexpr int E = D / 32;
    const int head = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int n0 = *n0p;
    const int width = (Hq + 2 * Hk) * D;
    const bf16* src = qkv + (size_t)t * width + (size_t)head * D + lane * E;
    const int slot = n0 + t;
    if (head >= Hq + Hk) {
        const int g = head - Hq - Hk;
        bf16* dst = V + ((size_t)g * cap + slot) * D + lane * E;
#pragma unroll
        for (int e = 0; e < E; ++e) dst[e] = src[e];
        return;
    }
    const bool is_q = head < Hq;
    const bf16* wn = is_q ? wq : wk;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) v[e] = bf2f(src[e]);
    if (wn != nullptr) {
        float ss = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) ss = fmaf(v[e], v[e], ss);
        ss = warp_sum(ss);
        const float rstd = rsqrtf(ss / (float)D + eps);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float wv = bf2f(wn[lane * E + e]);
            v[e] = bfround(v[e] * rstd * (centered ? 1.f + wv : wv));
        }
    }
    // rope: out = bf16(bf16(v * cos) + rot * sin), rot = -v[i + D/2] for the first half, v[i - D/2] after
    const int pos = n0 + depth[t];
    const bf16* cp = cos_t + (size_t)pos * D + lane * E;
    const bf16* sp = sin_t + (size_t)pos * D + lane * E;
    float rot[E];
#pragma unroll
    for (int e = 0; e < E; ++e) rot[e] = __shfl_xor_sync(0xffffffffu, v[e], 16);
    const float sgn = lane < 16 ? -1.f : 1.f;
    float o[E];
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const float qc = bfround(v[e] * bf2f(cp[e]));
        o[e] = fmaf(sgn * rot[e], bf2f(sp[e]), qc);
    }
    bf16* dst = is_q ? (qo + ((size_t)t * Hq + head) * D + lane * E)
                     : (K + ((size_t)(head - Hq) * cap + slot) * D + lane * E);
#pragma unroll
    for (int e = 0; e < E; ++e) dst[e] = f2bf(o[e]);
}

#define NORMROPE(D)                                                                                          \
    extern "C" __global__ void __launch_bounds__(32) btb_norm_rope_kv_d##D(                                  \
        const bf16* __restrict__ qkv, const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,   \
        const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t, const int* __restrict__ n0p,         \
        const int* __restrict__ depth, bf16* __restrict__ K, bf16* __restrict__ V, bf16* __restrict__ qo,    \
        int T, int Hq, int Hk, int cap, int centered) {                                                      \
        norm_rope_kv<D>(qkv, wq, wk, eps, cos_t, sin_t, n0p, depth, K, V, qo, T, Hq, Hk, cap, centered);     \
    }
NORMROPE(64)
NORMROPE(128)
NORMROPE(256)

// ---------------------------------------------------------------------------------------------------------
// residual + norm: h [T, H] bf16 in place (h += y when y is given, rounded as torch's bf16 add), then
// x = bf16(h * rstd * w) (or * (1 + w)). One block of 256 threads per row; thread i sums the squares of
// elements i, i + 256, ... in order, then a fixed tree over the block.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_add_rmsnorm(bf16* __restrict__ h, const bf16* __restrict__ y,
                                                                  const bf16* __restrict__ w, float eps,
                                                                  bf16* __restrict__ x, int H, int centered) {
    const int t = blockIdx.x, tid = threadIdx.x;
    bf16* hr = h + (size_t)t * H;
    const bf16* yr = y == nullptr ? nullptr : y + (size_t)t * H;
    bf16* xr = x + (size_t)t * H;
    float ss = 0.f;
    for (int i = tid; i < H; i += 256) {
        float hv = bf2f(hr[i]);
        if (yr != nullptr) {
            hv = bfround(hv + bf2f(yr[i]));
            hr[i] = f2bf(hv);
        }
        ss = fmaf(hv, hv, ss);
    }
    __shared__ float red[256];
    red[tid] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(red[0] / (float)H + eps);
    for (int i = tid; i < H; i += 256) {
        const float wv = bf2f(w[i]);
        xr[i] = f2bf(bf2f(hr[i]) * rstd * (centered ? 1.f + wv : wv));
    }
}

// ---------------------------------------------------------------------------------------------------------
// the sandwich block's residual (Gemma): h [T, H] bf16 += bf16(y * rstd * w) (or * (1 + w)), the delta normed
// before it is added rather than after; the add rounded as torch's bf16 add. The row's reduction as above.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_sandwich_add(bf16* __restrict__ h, const bf16* __restrict__ y,
                                                                   const bf16* __restrict__ w, float eps, int H,
                                                                   int centered) {
    const int t = blockIdx.x, tid = threadIdx.x;
    bf16* hr = h + (size_t)t * H;
    const bf16* yr = y + (size_t)t * H;
    float ss = 0.f;
    for (int i = tid; i < H; i += 256) {
        const float v = bf2f(yr[i]);
        ss = fmaf(v, v, ss);
    }
    __shared__ float red[256];
    red[tid] = ss;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(red[0] / (float)H + eps);
    for (int i = tid; i < H; i += 256) {
        const float wv = bf2f(w[i]);
        const float normed = bfround(bf2f(yr[i]) * rstd * (centered ? 1.f + wv : wv));
        hr[i] = f2bf(bf2f(hr[i]) + normed);
    }
}

// the tensor-core matvec for wide passes (one kernel for every row count, so a one-row step and a 32-row
// verify pass share their bits): btb_gemv_mma.cuh
#include "btb_gemv_mma.cuh"

// ---------------------------------------------------------------------------------------------------------
// the step's handoff to the host: the token and the cache's length written straight into pinned host memory
// (device-addressable under unified addressing) by one thread of one block - a kernel node at the graph's
// end, where a memcpy node was its own submission. The token lands first and a system fence orders it before
// the length, so a host that has seen the length also sees the token.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void btb_publish(const int* __restrict__ n0, const long long* __restrict__ ids,
                                       long long* out_tok, int* out_n0) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_tok = *ids;
        __threadfence_system();
        *out_n0 = *n0;
    }
}

// ---------------------------------------------------------------------------------------------------------
// act(g) * up over a [T, 2I] block: m[t][i] = bf16(bf16(act(g)) * u), the activation in fp32 as torch's.
// ---------------------------------------------------------------------------------------------------------
template <int ACT>
__device__ __forceinline__ void act_mul(const bf16* __restrict__ gu, bf16* __restrict__ m, int T, int I) {
    const size_t n = (size_t)T * I;
    for (size_t k = (size_t)blockIdx.x * 256 + threadIdx.x; k < n; k += (size_t)gridDim.x * 256) {
        const size_t t = k / I, i = k - t * I;
        const float g = bf2f(gu[t * (2 * (size_t)I) + i]);
        const float u = bf2f(gu[t * (2 * (size_t)I) + I + i]);
        const float s = bfround(act_of<ACT>(g));
        m[k] = f2bf(s * u);
    }
}
extern "C" __global__ void __launch_bounds__(256) btb_silu_mul(const bf16* __restrict__ gu, bf16* __restrict__ m,
                                                               int T, int I) {
    act_mul<0>(gu, m, T, I);
}
extern "C" __global__ void __launch_bounds__(256) btb_gelu_mul(const bf16* __restrict__ gu, bf16* __restrict__ m,
                                                               int T, int I) {
    act_mul<1>(gu, m, T, I);
}

// ---------------------------------------------------------------------------------------------------------
// the pick: a multi-block pipeline over [R, V] float32 logits (a row spread across every SM, ordinary launches).
// btb_sample_max reduces the row's max; btb_sample_hist/btb_sample_bin find the top-k and top-p thresholds by
// radix select over integer histograms (three levels of a float's 32 ordered bits, 11 + 11 + 10, no sort);
// btb_sample_draw takes the argmax of s + hashed Gumbel over the kept tokens, the lowest index among equals
// (temperature 0 is the plain argmax, handled upstream). The masses are exp(s - max) in 2^30 fixed point (an
// exact u64 sum, so the threshold is the same however the rows fall across threads), the exp and the hash the
// CPU kernel's (native/src/sample.rs) to the bit, so this pick's kept set and the CPU pick's are one set.
// btb_sample_keys derives the row's key from (seed, position) on the card so the captured step graph draws the
// same tokens the sequential loop keys by row.
// ---------------------------------------------------------------------------------------------------------
#define SMP_BINS 2048
#define SMP_THREADS 1024
#define SMP_WARPS 32
#define SMP_MASS 1073741824.0f  // 2^30 a unit of mass
#define SMP_FLOOR (-21.0f)      // a token this many nats under the top carries below 2^-30: out of the mass and the race

__device__ __forceinline__ unsigned smp_ordered(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// exp(t) for t <= 0, native/src/sample.rs `fast_exp` to the bit: t = n ln2 + r, e^r a degree-7 series, 0 below -87
__device__ __forceinline__ float smp_exp(float t) {
    if (t < -87.0f) return 0.0f;
    float n = roundf(t * 1.4426950408889634f);
    float r = t - n * 0.69314575f - n * 1.4286068e-6f;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333334f
              + r * (0.001388889f + r * 0.0001984127f))))));
    int e = (int)n + 127;
    e = e < 1 ? 1 : (e > 254 ? 254 : e);
    return __int_as_float((unsigned)e << 23) * p;
}

// the row's uniform of a token and its Gumbel, btb/mlx/sample.py `uniform_of`/`gumbel_of` and the CPU kernel
__device__ __forceinline__ float smp_uniform(unsigned long long key, unsigned i) {
    unsigned long long h = key ^ ((unsigned long long)i * 0x9E3779B97F4A7C15ull);
    h ^= h >> 32;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 29;
    h *= 0x94D049BB133111EBull;
    h ^= h >> 32;
    // 23 bits: the top of a 24-bit range rounds to 1.0 in float32, an infinite Gumbel that wins the row
    return ((float)(unsigned)(h >> 41) + 0.5f) * (1.0f / 8388608.0f);
}
__device__ __forceinline__ float smp_gumbel(unsigned long long key, unsigned i) {
    return -logf(-logf(smp_uniform(key, i)));
}

// the row's noise key from (seed, position[, salt]): btb/sampling.py `_mix` / `key_for`, a splitmix over the parts
__device__ __forceinline__ unsigned long long smp_mix_step(unsigned long long x, unsigned long long p) {
    x = (x ^ p) + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}
__device__ __forceinline__ unsigned long long smp_key(unsigned long long seed, unsigned long long pos, unsigned salt) {
    unsigned long long x = smp_mix_step(smp_mix_step(0ull, seed), pos);
    if (salt) x = smp_mix_step(x, (unsigned long long)salt);
    return x & 0x7FFFFFFFFFFFFFFFull;
}

// the ordered uint back to its float (the inverse of smp_ordered), for reading a max stored by atomicMax
__device__ __forceinline__ float smp_unordered(unsigned o) {
    unsigned u = (o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o;
    return __uint_as_float(u);
}

// a candidate (value, index) packed so one atomicMax over a row keeps the max value, the lowest index on a tie:
// the value's ordered bits high, ~index low (a larger ~index is a smaller index)
__device__ __forceinline__ unsigned long long smp_pack(float v, unsigned i) {
    return ((unsigned long long)smp_ordered(v) << 32) | (unsigned long long)(~i);
}

// this block's best (max value, lowest index) reduced and merged into the row's global winner by atomicMax
__device__ __forceinline__ void smp_block_argmax(float best, unsigned bi, unsigned long long* g) {
    __shared__ float rf[SMP_WARPS];
    __shared__ unsigned ri[SMP_WARPS];
    const unsigned tid = threadIdx.x, lane = tid & 31u, wid = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) { rf[wid] = best; ri[wid] = bi; }
    __syncthreads();
    if (tid == 0) {
        float b = rf[0];
        unsigned bb = ri[0];
        for (unsigned s = 1; s < SMP_WARPS; ++s)
            if (rf[s] > b || (rf[s] == b && ri[s] < bb)) { b = rf[s]; bb = ri[s]; }
        if (b > NEG_INF) atomicMax(g, smp_pack(b, bb));
    }
}

// ---- the multi-block pipeline: a row spread over every SM as ordinary (graph-capturable, contention-safe)
// launches. Three radix levels (11 + 11 + 10 = the exact 32-bit threshold, so the kept set is the CPU pick's to
// the token), the histograms an exact u64 sum in global memory. Scratch a row: gM [R] (ordered max), ghist
// [R, 2048] u64, gpre [R] (the running threshold, ordered bits), gcarry [R] u64 (the radix remainder), gbest [R]
// u64 (the packed draw). The pieces run: max -> (k levels 0,1,2) -> (p levels 0,1,2) -> draw, each over grid
// (blocks-a-row, R, 1). top-k keeps its threshold in gpreK, top-p in gpreP; the draw keeps tokens at or above
// either. Greedy is the plain argmax elsewhere, so these run only for a temperature draw.

// M = max(x / T): grid over the row, atomicMax(ordered) into gM[r] (pre-zeroed). T from fp[0] on the device, so
// the captured step graph reads a temperature filled before the replay rather than one baked at capture.
extern "C" __global__ void __launch_bounds__(SMP_THREADS) btb_sample_max(const float* __restrict__ x, int V,
                                                                         const float* __restrict__ fp,
                                                                         unsigned* gM) {
    const float invT = fp[0];
    const unsigned r = blockIdx.y;
    const float* row = x + (size_t)r * (unsigned)V;
    const unsigned g0 = blockIdx.x * blockDim.x + threadIdx.x, gs = gridDim.x * blockDim.x;
    float m = NEG_INF;
    for (unsigned i = g0; i < (unsigned)V; i += gs) m = fmaxf(m, row[i] * invT);
    __shared__ float rf[SMP_WARPS];
    const unsigned lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) rf[wid] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float mm = rf[0];
        for (unsigned s = 1; s < SMP_WARPS; ++s) mm = fmaxf(mm, rf[s]);
        atomicMax(&gM[r], smp_ordered(mm));
    }
}

// one radix level's histogram: mode 0 counts (top-k), 1 masses (top-p). `gpre[r]` the prefix resolved so far,
// `gfloor[r]` the ordered floor a token must clear (0 for top-k; the top-k threshold for top-p). ghist pre-zeroed
extern "C" __global__ void __launch_bounds__(SMP_THREADS) btb_sample_hist(const float* __restrict__ x, int V,
                                                                          const float* __restrict__ fp, int level,
                                                                          int mode, const unsigned* gM,
                                                                          const unsigned* gpre, const unsigned* gfloor,
                                                                          unsigned long long* ghist) {
    const float invT = fp[0];
    __shared__ unsigned long long sh[SMP_BINS];  // this block's private histogram: shared atomics, then one flush
    const unsigned r = blockIdx.y;
    const float* row = x + (size_t)r * (unsigned)V;
    unsigned long long* hist = ghist + (size_t)r * SMP_BINS;
    const unsigned g0 = blockIdx.x * blockDim.x + threadIdx.x, gs = gridDim.x * blockDim.x;
    // three levels resolve a float's 32 ordered bits, 11 + 11 + 10 (an exact threshold, the CPU kernel's)
    const unsigned bits = level < 2 ? 11u : 10u;
    const unsigned shift = level == 0 ? 21u : (level == 1 ? 10u : 0u), hi = shift + bits;
    const unsigned prefix = gpre[r], floor = gfloor[r];
    const float M = mode ? smp_unordered(gM[r]) : 0.0f;
    for (unsigned b = threadIdx.x; b < SMP_BINS; b += blockDim.x) sh[b] = 0ull;
    __syncthreads();
    for (unsigned i = g0; i < (unsigned)V; i += gs) {
        float s = row[i] * invT;
        unsigned u = smp_ordered(s);
        if (u < floor) continue;
        if (hi < 32u && (u >> hi) != (prefix >> hi)) continue;
        if (mode == 0) {
            atomicAdd(&sh[(u >> shift) & (SMP_BINS - 1)], 1ull);
        } else if (s - M >= SMP_FLOOR) {
            unsigned long long w = (unsigned long long)(smp_exp(s - M) * SMP_MASS);
            if (w) atomicAdd(&sh[(u >> shift) & (SMP_BINS - 1)], w);
        }
    }
    __syncthreads();
    for (unsigned b = threadIdx.x; b < SMP_BINS; b += blockDim.x)
        if (sh[b]) atomicAdd(&hist[b], sh[b]);
}

// one radix level's pick: one block a row scans the histogram from the top, extends `gpre[r]` by the bin the
// count/mass first reaches the target, carries the remainder in `gcarry[r]`. top-k's target is `top_k` at level
// 0 then the carry; top-p's is `top_p` of the level-0 total then the carry.
extern "C" __global__ void __launch_bounds__(SMP_THREADS) btb_sample_bin(int level, int mode, int top_k,
                                                                         const float* __restrict__ fp,
                                                                         const unsigned long long* ghist,
                                                                         unsigned* gpre, unsigned long long* gcarry) {
    // the whole block loads the row's histogram into shared (latency hidden across threads), then the scan runs
    // off shared - a serial scan of global memory here was 2048 dependent ~600-cycle loads on one thread, the
    // pipeline's whole cost. The level-0 mass total is a block reduction over the same shared copy.
    const unsigned CH = 64, PER = SMP_BINS / CH;  // 64 chunks of 32 bins: a two-level scan, no 2048-long chain
    __shared__ unsigned long long sh[SMP_BINS];
    __shared__ unsigned long long cs[CH];  // each chunk's total (for the coarse scan) and the mass total
    const unsigned r = blockIdx.x;
    const unsigned long long* hist = ghist + (size_t)r * SMP_BINS;
    for (unsigned b = threadIdx.x; b < SMP_BINS; b += blockDim.x) sh[b] = hist[b];
    __syncthreads();
    if (threadIdx.x < CH) {
        unsigned long long c = 0ull;
        for (unsigned j = 0; j < PER; ++j) c += sh[threadIdx.x * PER + j];
        cs[threadIdx.x] = c;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const unsigned shift = level == 0 ? 21u : (level == 1 ? 10u : 0u);
    unsigned long long target;
    if (level != 0) {
        target = gcarry[r];
    } else if (mode == 0) {
        target = (unsigned long long)top_k;
    } else {
        unsigned long long z = 0ull;
        for (unsigned c = 0; c < CH; ++c) z += cs[c];  // the level-0 mass total
        target = (unsigned long long)((float)z * fp[1]);
        if (!target) target = 1ull;  // at least the top token
    }
    // coarse: the chunk the running total first reaches; fine: the bin inside it
    unsigned long long acc = 0ull;
    int chunk = 0;
    for (int c = CH - 1; c >= 0; --c) {
        acc += cs[c];
        if (acc >= target) { chunk = c; break; }
    }
    unsigned long long rem = target - (acc - cs[chunk]);  // still needed from the top of this chunk
    unsigned long long a = 0ull;
    unsigned pick = (unsigned)chunk * PER;
    for (int b = (int)((chunk + 1) * PER) - 1; b >= (int)(chunk * PER); --b) {
        unsigned long long c = sh[b];
        a += c;
        if (a >= rem) { pick = (unsigned)b; rem -= a - c; break; }
    }
    gpre[r] = (level == 0 ? 0u : gpre[r]) | (pick << shift);
    gcarry[r] = rem;
}

// the draw: grid over the row, argmax of s + gumbel(key, i) over the tokens at or above the floor (the larger of
// the top-k and top-p thresholds), the lowest index among equals; atomicMax-packed into gbest[r] (pre-zeroed).
// `keys` [R, 2] uint32 (lo, hi). out[r] the picked token (uint32).
extern "C" __global__ void __launch_bounds__(SMP_THREADS) btb_sample_draw(
    const float* __restrict__ x, const unsigned* __restrict__ keys, int V, const float* __restrict__ fp,
    const unsigned* gM, const unsigned* gpreK, const unsigned* gpreP, unsigned long long* gbest) {
    const float invT = fp[0];
    const unsigned r = blockIdx.y;
    const float* row = x + (size_t)r * (unsigned)V;
    const unsigned g0 = blockIdx.x * blockDim.x + threadIdx.x, gs = gridDim.x * blockDim.x;
    const unsigned floor = max(gpreK[r], gpreP[r]);
    const float M = smp_unordered(gM[r]);
    const unsigned long long key = ((unsigned long long)keys[2 * r + 1] << 32) | (unsigned long long)keys[2 * r];
    float best = NEG_INF;
    unsigned bi = 0xFFFFFFFFu;
    for (unsigned i = g0; i < (unsigned)V; i += gs) {
        float s = row[i] * invT;
        if (smp_ordered(s) < floor || s - M < SMP_FLOOR) continue;
        float v = s + smp_gumbel(key, i);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    smp_block_argmax(best, bi, &gbest[r]);
}

// write out[r] = the picked token from the packed gbest (a tiny finalize after the draw's grid has settled). The
// launch rounds R up to whole blocks, so the threads past R leave: unguarded, a one-row pick's 255 spare threads
// wrote ~1 KB past `out` into whatever the allocator had placed after it
extern "C" __global__ void btb_sample_out(const unsigned long long* gbest, unsigned* out, int R) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (unsigned)R) return;
    out[r] = ~(unsigned)(gbest[r] & 0xFFFFFFFFull);
}

// derive a row's key from (seed, position) on the card, into keys[0..1] - the step graph's draw reads it, so a
// replay draws the token the sequential loop's key_for(pos) keys
extern "C" __global__ void btb_sample_keys(const unsigned* seed, const int* pos, int salt, unsigned* keys) {
    if (blockIdx.x || threadIdx.x) return;
    const unsigned long long sd = ((unsigned long long)seed[1] << 32) | (unsigned long long)seed[0];
    const unsigned long long k = smp_key(sd, (unsigned long long)(unsigned)pos[0], (unsigned)salt);
    keys[0] = (unsigned)(k & 0xFFFFFFFFull);
    keys[1] = (unsigned)(k >> 32);
}

// ---- verify: the drawn-tree rejection-sampling pass, one block a row (a tree node), mirroring the Metal verify
// (btb/mlx/sample.py). A node is not on the single-token hot path (a speculative pass carries T<=16 rows), so a
// block a row is enough. The threshold is the pick's (single-block here, three radix levels); the draws use the
// same hashed Gumbel, so the emitted token is a draw from the target p at every node (exact speculation).
#define SMP_MAXC 32  // children a node can carry: a pass has at most CARD_T_MAX = 32 rows, the root one of them

// this block's M (max s) and kept threshold (ordered bits) for the row, the pick's radix in one block
__device__ void smp_thresh_block(const float* row, unsigned V, float invT, unsigned top_k, float top_p, float* Mout,
                                 unsigned* keptout) {
    __shared__ unsigned long long histo[SMP_BINS];
    __shared__ float redf[SMP_WARPS];
    __shared__ unsigned chosen;
    __shared__ unsigned long long needl;
    __shared__ float Msh;
    const unsigned tid = threadIdx.x, lane = tid & 31u, wid = tid >> 5;
    float m = NEG_INF;
    for (unsigned i = tid; i < V; i += SMP_THREADS) m = fmaxf(m, row[i] * invT);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) redf[wid] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = redf[0];
        for (unsigned s = 1; s < SMP_WARPS; ++s) mm = fmaxf(mm, redf[s]);
        Msh = mm;
    }
    __syncthreads();
    const float M = Msh;
    unsigned kept = 0u;
    for (int pass = 0; pass < 2; ++pass) {  // pass 0 top-k (counts), pass 1 top-p (masses)
        const bool on = pass == 0 ? (top_k > 0u && top_k < V) : (top_p < 1.0f);
        if (!on) continue;
        unsigned prefix = 0u;
        unsigned long long need = pass == 0 ? (unsigned long long)top_k : 0ull;
        for (unsigned level = 0; level < 3; ++level) {
            const unsigned bits = level < 2 ? 11u : 10u;
            const unsigned shift = level == 0 ? 21u : (level == 1 ? 10u : 0u), hi = shift + bits;
            for (unsigned b = tid; b < SMP_BINS; b += SMP_THREADS) histo[b] = 0ull;
            __syncthreads();
            for (unsigned i = tid; i < V; i += SMP_THREADS) {
                float s = row[i] * invT;
                unsigned u = smp_ordered(s);
                if (u < kept) continue;  // top-p keeps the top-k's floor; top-k has kept == 0
                if (hi < 32u && (u >> hi) != (prefix >> hi)) continue;
                if (pass == 0) {
                    atomicAdd(&histo[(u >> shift) & (SMP_BINS - 1)], 1ull);
                } else if (s - M >= SMP_FLOOR) {
                    unsigned long long w = (unsigned long long)(smp_exp(s - M) * SMP_MASS);
                    if (w) atomicAdd(&histo[(u >> shift) & (SMP_BINS - 1)], w);
                }
            }
            __syncthreads();
            if (tid == 0) {
                unsigned long long target = need;
                if (pass == 1 && level == 0) {
                    unsigned long long z = 0ull;
                    for (unsigned b = 0; b < SMP_BINS; ++b) z += histo[b];
                    target = (unsigned long long)((float)z * top_p);
                    if (!target) target = 1ull;  // at least the top token
                }
                unsigned long long acc = 0ull;
                unsigned pick = 0u;
                for (int b = SMP_BINS - 1; b >= 0; --b) {
                    unsigned long long c = histo[b];
                    acc += c;
                    if (acc >= target) { pick = (unsigned)b; target -= acc - c; break; }
                }
                chosen = prefix | (pick << shift);
                needl = target;
            }
            __syncthreads();
            prefix = chosen;
            need = needl;
        }
        if (prefix > kept) kept = prefix;
    }
    *Mout = M;
    *keptout = kept;
}

// token x's residual mass after `k` rejected trials (btb/mlx/sample.py `residual_of`): w0 = exp(s - M) over the
// kept tokens, w_{j+1} = max(0, w_j / zp[j] - q'_j(x) / zq[j]), q' the drafter's mass with the tried children out
__device__ float smp_residual(const float* row, const float* qrow, const int* kid, unsigned x, unsigned k,
                              float invT, float M, unsigned tmin, unsigned Vd, const float* zp, const float* zq) {
    float s = row[x] * invT;
    float w = (smp_ordered(s) >= tmin) ? smp_exp(s - M) : 0.0f;
    float qx = (x < Vd) ? qrow[x] : 0.0f;
    for (unsigned j = 0; j < k; ++j) {
        w = fmaxf(0.0f, w / zp[j] - qx / zq[j]);
        if (kid[j] >= 0 && (unsigned)kid[j] == x) qx = 0.0f;
    }
    return w;
}

// x [T, V] node logits, q [T, Vd] the drafter's distribution a node, kids [T, C] int32 (draw order, -1 pad),
// hasq [T] (0: point-mass drafts), keys [T, 2], fp [1/T, top_p]; out[r] packed (slot + 1) << 24 | token
extern "C" __global__ void __launch_bounds__(SMP_THREADS) btb_sample_verify(
    const float* __restrict__ x, const float* __restrict__ q, const int* __restrict__ kids,
    const unsigned* __restrict__ hasq, const unsigned* __restrict__ keys, int V, int Vd, int C, int top_k,
    const float* __restrict__ fp, unsigned* __restrict__ out) {
    const float invT = fp[0], top_p = fp[1];
    const unsigned r = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, wid = tid >> 5;
    const float* row = x + (size_t)r * (unsigned)V;
    const float* qrow = q + (size_t)r * (unsigned)Vd;
    const unsigned long long key = ((unsigned long long)keys[2 * r + 1] << 32) | (unsigned long long)keys[2 * r];
    __shared__ float redf[SMP_WARPS];
    __shared__ unsigned redu[SMP_WARPS];
    __shared__ int kid[SMP_MAXC];
    __shared__ float zp[SMP_MAXC + 1];
    __shared__ float zq[SMP_MAXC + 1];
    __shared__ unsigned outcome0, outcome1;
    float M;
    unsigned tmin;
    smp_thresh_block(row, (unsigned)V, invT, (unsigned)top_k, top_p, &M, &tmin);
    for (unsigned c = tid; c < (unsigned)C; c += SMP_THREADS) kid[c] = kids[(size_t)r * (unsigned)C + c];
    __syncthreads();

    if (hasq[r] == 0u) {  // point-mass drafts: the draw from p, the child that equals it accepted
        float best = NEG_INF;
        unsigned bi = 0xFFFFFFFFu;
        for (unsigned i = tid; i < (unsigned)V; i += SMP_THREADS) {
            float s = row[i] * invT;
            if (smp_ordered(s) < tmin || s - M < SMP_FLOOR) continue;
            float v = s + smp_gumbel(key, i);
            if (v > best || (v == best && i < bi)) { best = v; bi = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
        }
        if (lane == 0) { redf[wid] = best; redu[wid] = bi; }
        __syncthreads();
        if (tid == 0) {
            float b = redf[0];
            unsigned bb = redu[0];
            for (unsigned s = 1; s < SMP_WARPS; ++s)
                if (redf[s] > b || (redf[s] == b && redu[s] < bb)) { b = redf[s]; bb = redu[s]; }
            unsigned slot = 0u;
            for (unsigned c = 0; c < (unsigned)C; ++c)
                if (kid[c] >= 0 && (unsigned)kid[c] == bb) { slot = c + 1; break; }
            out[r] = (slot << 24) | bb;
        }
        return;
    }

    // zp[0] the target's mass over the kept tokens, zq[0] the drafter's over its vocabulary
    float part = 0.0f;
    for (unsigned i = tid; i < (unsigned)V; i += SMP_THREADS) {
        float s = row[i] * invT;
        if (smp_ordered(s) >= tmin) part += smp_exp(s - M);
    }
    for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
    if (lane == 0) redf[wid] = part;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (unsigned s = 0; s < SMP_WARPS; ++s) z += redf[s];
        zp[0] = z;
    }
    __syncthreads();
    part = 0.0f;
    for (unsigned i = tid; i < (unsigned)Vd; i += SMP_THREADS) part += qrow[i];
    for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
    if (lane == 0) redf[wid] = part;
    __syncthreads();
    if (tid == 0) {
        float z = 0.0f;
        for (unsigned s = 0; s < SMP_WARPS; ++s) z += redf[s];
        zq[0] = z;
        outcome0 = 0u;
        outcome1 = 0u;
    }
    __syncthreads();

    // the trials: child i accepted when u_i * Q_i(c) < P_i(c); a rejection leaves the residual for the next
    unsigned tried = 0u;
    for (unsigned i = 0; i < (unsigned)C; ++i) {
        int c = kid[i];
        if (c < 0) break;
        float qc = ((unsigned)c < (unsigned)Vd && zq[i] > 0.0f) ? fminf(qrow[c] / zq[i], 1.0f) : 0.0f;
        if (!(qc > 0.0f)) break;  // a child the drafter gives no mass is not a draw of its: the trials end
        float wc = smp_residual(row, qrow, kid, (unsigned)c, i, invT, M, tmin, (unsigned)Vd, zp, zq);
        float pc = wc / zp[i];
        float u = smp_uniform(key, 0xF00000u + i);
        if (u * qc < pc) {
            if (tid == 0) { outcome0 = i + 1; outcome1 = (unsigned)c; }
            __syncthreads();
            break;
        }
        part = 0.0f;
        for (unsigned xx = tid; xx < (unsigned)V; xx += SMP_THREADS)
            part += smp_residual(row, qrow, kid, xx, i + 1, invT, M, tmin, (unsigned)Vd, zp, zq);
        for (int o = 16; o > 0; o >>= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
        if (lane == 0) redf[wid] = part;
        __syncthreads();
        if (tid == 0) {
            float z = 0.0f;
            for (unsigned s = 0; s < SMP_WARPS; ++s) z += redf[s];
            zp[i + 1] = z;
            zq[i + 1] = zq[i] - qrow[c];
            if (!(z > 0.0f)) { outcome0 = i + 1; outcome1 = (unsigned)c; }  // an empty residual: P = Q, accept
        }
        __syncthreads();
        if (outcome0 != 0u) break;
        tried = i + 1;
    }
    if (outcome0 != 0u) {
        if (tid == 0) out[r] = (outcome0 << 24) | outcome1;
        return;
    }
    // none accepted: a draw from the residual after every trial
    float best = NEG_INF;
    unsigned bi = 0xFFFFFFFFu;
    for (unsigned xx = tid; xx < (unsigned)V; xx += SMP_THREADS) {
        float w = smp_residual(row, qrow, kid, xx, tried, invT, M, tmin, (unsigned)Vd, zp, zq);
        if (w <= 0.0f) continue;
        float v = logf(w) + smp_gumbel(key, xx);
        if (v > best || (v == best && xx < bi)) { best = v; bi = xx; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        unsigned oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    if (lane == 0) { redf[wid] = best; redu[wid] = bi; }
    __syncthreads();
    if (tid == 0) {
        float b = redf[0];
        unsigned bb = redu[0];
        for (unsigned s = 1; s < SMP_WARPS; ++s)
            if (redf[s] > b || (redf[s] == b && redu[s] < bb)) { b = redf[s]; bb = redu[s]; }
        out[r] = bb;
    }
}
