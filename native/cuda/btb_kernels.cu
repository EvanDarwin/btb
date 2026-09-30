// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The card's decode kernels. Every reduction runs in an order fixed by the element's index alone - never by
// the number of rows a pass carries, the cache's length or the launch's timing - so a verify pass of T rows
// computes each row bit-for-bit as the one-row step does, and one captured graph serves every length.
//
//   btb_gemv_bf16_m{1,..,32}      y[m][r] = sum_c w[r][c] * x[m][c]      (bf16 in, f32 accumulate, bf16 out)
//   btb_gemv_{silu,gelu}_bf16_m{1,..,32}  the down projection with act(g) * u folded into its x load
//   btb_attn_split_d{64,128,256}  one pass of T queries over the cache (a sliding layer's window of it), a tree of
//                                 T rows at its end, split over the sequence
//   btb_attn_rows_d{64,128,256}   the same over T sequences a token each (a fork's or a batch's rows)
//   btb_norm_rope_kv_d{64,128,256} q/k RMSNorm, rope, the pass's rows written into the cache
//   btb_norm_rope_kv_rows_d{64,128,256}  the same for T sequences, each at its own position
//   btb_add_rmsnorm               h += y; x = rmsnorm(h) * w   (or * (1 + w), the zero-centred norm)
//   btb_sandwich_add              h += rmsnorm(y) * w          (the sandwich block: the delta normed, then added)
//   btb_{silu,gelu}_mul           m = act(g) * u  over a [T, 2I] gate/up block
//   btb_mx4_widen                 MXFP4 experts at given seats of a depot's stacks to bf16, one pass
//   btb_delta_nodes{,_bf16}       the gated DeltaNet over a pass's nodes (a chain, a tree, the commit's path)
//   btb_conv_window               the DeltaNet's conv state after a chain of steps
//   Qwen4:
//   btb_hc_rmsnorm                h[g] += y * inj[g] per stream; x = each stream's rmsnorm * (1 + w)
//   btb_hc_act, btb_hc_mix        the stream mixer: silu(down / G); mean_g sigmoid(up) * x, the inject weights
//   btb_moe_route                 softmax -> top-k -> renormalise, the picks published to the host
//   btb_moe_combine               y = routed + sigmoid(gate) * shared
//   btb_sigmoid_mul, btb_gemv_sgate_bf16_m{1,..,32}  the attention's output gate, alone or folded into o_proj
//   btb_norm_rope_part_d{128,256} q/k centred norm, partial rope, the pass's rows into the cache (or the
//                                 indexer's q and raw key)
//   btb_qsa_pool_d{128,256}       the indexer's pooled keys of the committed blocks, caught up on the card
//   btb_qsa_select_d{128,256}     a tree node's picked key blocks (the indexer's top-k over its visible blocks)
//   btb_qsa_attn_split_d{128,256} the attention over a node's picks and its tail: btb_attn_split's walk over a list
//   btb_gemv_lane16_f32_m{1,..,32}  the host's bf16 x f32 gemv (native/src/gemv.rs) to the bit: an expert seated on
//                                 the card computes what it computes on the host
//   btb_ple_gate, btb_ple_conv    the per-layer n-gram embedding over a pass's nodes: the key's gate on the streams
//                                 and the gated rows, then the dilated conv along each node's path into the streams
//
// The weights stream with evict-first loads (read once a step); the cache and the activations take the
// default policy, so a persisting-L2 window over the cache's front keeps a short context's attention in L2.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
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
//
// ROWS: the T queries are T sequences instead of one tree - a fork's rows, or a batch's - each one token at its
// own position. Row t's keys are its prefix, lens[t] rows from slot offs[t] (a fork's rows share one prefix, a
// batch's lie end to end), then its own steps, step i at slot base + i * W + col[t]: every row's step i in one
// stretch of W slots. The walk is row t's logical keys in order, the sequence a one-row step of that sequence
// alone walks over a contiguous cache, so a row's bits are its own decode's whatever rows step beside it.
// ---------------------------------------------------------------------------------------------------------
#define ATTN_SPLIT 1024
// the keys a warp loads, then scores, then folds in order: its K and V rows standing together, and their scores'
// butterflies side by side instead of each behind the last key's softmax. A short sequence - a block a head, a
// warp 128 keys at 1k - is nothing but that chain (Qwen3-0.6B at 1k: 32 us a layer for 4 MB). A key's score is
// its own dot product and butterfly and the fold is key by key in order whatever the batch, so the batch moves
// no bit; it is bounded by the registers the rows take (E a lane a row), which the long-context tree needs for
// the blocks it keeps resident
#define ATTN_BATCH(E) ((E) >= 8 ? 4 : 8)

// the batch's B scores q . k_u * scale, each over the lane's E elements then the butterfly: independent chains
template <int E, int B>
__device__ __forceinline__ void attn_scores(float* sc, const float* qf, const Vec<E>* kv, float scale) {
#pragma unroll
    for (int u = 0; u < B; ++u) {
        float d = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) d = fmaf(qf[e], kv[u].at(e), d);
        sc[u] = d;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int u = 0; u < B; ++u) sc[u] += __shfl_xor_sync(0xffffffffu, sc[u], o);
    }
#pragma unroll
    for (int u = 0; u < B; ++u) sc[u] *= scale;
}

// the rows' layout, one int32 array on the card: [base, steps, W, then per row (col, off, len)]; a row with
// len < 0 is padding (the pass is launched for a width its graph was captured at) and computes nothing
struct RowsLayout {
    int base = 0, step = 0, W = 0, col = 0, off = 0, len = 0;
    RowsLayout() = default;
    __device__ __forceinline__ RowsLayout(const int* __restrict__ rw, int t)
        : base(rw[0]), step(rw[1]), W(rw[2]), col(rw[3 + 3 * t]), off(rw[4 + 3 * t]), len(rw[5 + 3 * t]) {}
    // the cache slot of the row's logical key j
    __device__ __forceinline__ int slot(int j) const { return j < len ? off + j : base + (j - len) * W + col; }
};

// the walk's forms: WALK_TREE a tree of T rows at the cache's end, WALK_ROWS T sequences a token each, WALK_QSA the
// tree's rows each over the QSA indexer's picks alone (btb_qsa_attn_split below)
#define WALK_TREE 0
#define WALK_ROWS 1
#define WALK_QSA 2

template <int D, int MODE>
__device__ __forceinline__ void attn_decode_split(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                                  const bf16* __restrict__ V, bf16* __restrict__ out,
                                                  const int* __restrict__ n0p, const int* __restrict__ par,
                                                  const int* __restrict__ rw, const int* __restrict__ sel,
                                                  const int* __restrict__ nsel, int ratio, int ktop, int T, int Hq,
                                                  int Hk, int cap, float scale, float* __restrict__ part_m,
                                                  float* __restrict__ part_l, float* __restrict__ part_acc,
                                                  int* __restrict__ cnt, int S, int win) {
    constexpr int E = D / 32;
    const int h = blockIdx.x, t = blockIdx.y, s = blockIdx.z;
    const int g = h / (Hq / Hk);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    // the tree's walk (its prefix length and each row's ancestors) or the rows' layout: one of the two is used
    [[maybe_unused]] __shared__ int anc[32];
    [[maybe_unused]] __shared__ int s_d;
    [[maybe_unused]] int n0 = 0;
    [[maybe_unused]] RowsLayout r;
    // QSA: the row's complete blocks, the keys of its picks (nsel * ratio) and its picks' block indices
    [[maybe_unused]] int nb = 0, npk = 0;
    [[maybe_unused]] const int* selt = nullptr;
    int n;  // the length of the walk: the row's logical keys (QSA: its list)
    if constexpr (MODE == WALK_ROWS) {
        r = RowsLayout(rw, t);
        if (r.len < 0) return;
        n = r.len + r.step + 1;
        if (s * ATTN_SPLIT >= n) return;
    } else {
        n0 = *n0p;
        if constexpr (MODE == WALK_TREE) {
            // no query of this pass reaches past n0 + T rows: a block whose range starts beyond that leaves before
            // the walk and the barrier, so at a short context the launch costs what the unsplit kernel costs
            if (s * ATTN_SPLIT >= n0 + T) return;
        } else {
            if (par[t] < -1) return;  // a padding row: nothing to attend
        }
        if (threadIdx.x == 0) {
            int tmp[32];
            int len = 0;
            for (int p = t; p >= 0 && p < T && len < 32; p = par[p]) tmp[len++] = p;
            for (int e = 0; e < len; ++e) anc[e] = tmp[len - 1 - e];
            s_d = len - 1;
        }
        __syncthreads();
        n = n0 + s_d + 1;
        if constexpr (MODE == WALK_QSA) {
            // the list: the picked blocks' keys in block order, then the partial tail - ascending positions
            nb = n / ratio;
            npk = nsel[t] * ratio;
            selt = sel + (size_t)t * ktop;
            n = npk + (n - nb * ratio);
        }
    }
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
    // a sliding layer (win > 0) sees the last win logical keys, [first, n): the walk below keeps its key -> warp
    // map and passes over the keys before first, so a windowed row folds as the full row's tail would
    const int first = win > 0 ? max(n - win, 0) : 0;
    // the splits this query needs: a single one writes its row directly, without the partials
    const int S_active = (n + ATTN_SPLIT - 1) / ATTN_SPLIT;
    if (s >= S_active) return;
    const int lo = s * ATTN_SPLIT, hi = min(lo + ATTN_SPLIT, n);
    for (int j0 = lo + w * 32; j0 < hi; j0 += 256) {
        const int j1 = min(j0 + 32, hi);
        for (int j = j0; j < j1; j += ATTN_BATCH(E)) {
            const int c4 = min(ATTN_BATCH(E), j1 - j);
            Vec<E> kv[ATTN_BATCH(E)] = {}, vv[ATTN_BATCH(E)];
#pragma unroll
            for (int u = 0; u < ATTN_BATCH(E); ++u) {
                if (u < c4 && j + u >= first) {
                    const int jj = j + u;
                    int slot;
                    if constexpr (MODE == WALK_ROWS) {
                        slot = r.slot(jj);
                    } else {
                        int pos = jj;  // list index -> logical position (the identity but for QSA)
                        if constexpr (MODE == WALK_QSA)
                            pos = jj < npk ? selt[jj / ratio] * ratio + jj % ratio : nb * ratio + (jj - npk);
                        slot = pos < n0 ? pos : n0 + anc[pos - n0];
                    }
                    kv[u].load(Kg + (size_t)slot * rowstride);
                    vv[u].load(Vg + (size_t)slot * rowstride);
                }
            }
            float sc[ATTN_BATCH(E)];
            attn_scores<E, ATTN_BATCH(E)>(sc, qf, kv, scale);
#pragma unroll
            for (int u = 0; u < ATTN_BATCH(E); ++u) {
                if (u < c4 && j + u >= first) {
                    const float mn = fmaxf(m, sc[u]);
                    const float corr = expf(m - mn);
                    const float p = expf(sc[u] - mn);
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
        attn_decode_split<D, WALK_TREE>(q, K, V, out, n0p, par, nullptr, nullptr, nullptr, 0, 0, T, Hq, Hk,  \
                                        cap, scale, part_m, part_l, part_acc, cnt, S, win);                  \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256) btb_attn_rows_d##D(                                    \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ rw, int T, int Hq, int Hk, int cap, float scale,     \
        float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,                \
        int* __restrict__ cnt, int S, int win) {                                                             \
        attn_decode_split<D, WALK_ROWS>(q, K, V, out, nullptr, nullptr, rw, nullptr, nullptr, 0, 0, T, Hq,   \
                                        Hk, cap, scale, part_m, part_l, part_acc, cnt, S, win);              \
    }
ATTN_SPLIT_K(64)
ATTN_SPLIT_K(128)
ATTN_SPLIT_K(256)

// ---------------------------------------------------------------------------------------------------------
// the tree walk above for the G query heads of one KV group at once: one block loads each of its keys' K and V
// rows once and runs every head's walk over them - where a block a query head reads the group's rows G times
// over, and a tree pass's T rows T times more. The L2 catches most of that for a one-row step; a long tree's
// verify it does not (Qwen3-0.6B, a 5-row tree: 11% off the attention at 8k, 27% at 40k). Each head's walk is the
// one above operation for operation - its keys in the same warps and order, its online softmax, its warps
// folded in order (warp h folds head h), its splits folded in order by the last block - so a head's bits are
// the one-head kernel's, and the partials and counts are the one-head kernel's, a head's by its own row.
// ---------------------------------------------------------------------------------------------------------
template <int D, int G>
__device__ __forceinline__ void attn_split_gqa(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                               const bf16* __restrict__ V, bf16* __restrict__ out, int n0,
                                               const int* __restrict__ par, int g, int T, int Hq, int cap,
                                               float scale, float* __restrict__ part_m, float* __restrict__ part_l,
                                               float* __restrict__ part_acc, int* __restrict__ cnt, int win) {
    static_assert(G <= 8, "a head's fold is one warp's");
    constexpr int E = D / 32;
    const int t = blockIdx.y, s = blockIdx.z;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
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
    const int n = n0 + s_d + 1;
    const int S_active = (n + ATTN_SPLIT - 1) / ATTN_SPLIT;
    if (s >= S_active) return;
    float qf[G][E];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        const bf16* qp = q + ((size_t)t * Hq + g * G + hh) * D + lane * E;
#pragma unroll
        for (int e = 0; e < E; ++e) qf[hh][e] = bf2f(qp[e]);
    }
    const bf16* Kg = K + (size_t)g * cap * D + lane * E;
    const bf16* Vg = V + (size_t)g * cap * D + lane * E;
    float m[G], l[G], acc[G][E];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        m[hh] = NEG_INF;
        l[hh] = 0.f;
#pragma unroll
        for (int e = 0; e < E; ++e) acc[hh][e] = 0.f;
    }
    const int first = win > 0 ? max(n - win, 0) : 0;
    const int lo = s * ATTN_SPLIT, hi = min(lo + ATTN_SPLIT, n);
    // four keys a batch: the G heads' scores are the independent chains here, and the block keeps the registers
    // the long-context tree needs resident (at eight, 97 of them on d128 x 2 heads, 16% slower at 40k)
    constexpr int B = 4;
    for (int j0 = lo + w * 32; j0 < hi; j0 += 256) {
        const int j1 = min(j0 + 32, hi);
        for (int j = j0; j < j1; j += B) {
            const int c4 = min(B, j1 - j);
            Vec<E> kv[B] = {}, vv[B];
#pragma unroll
            for (int u = 0; u < B; ++u) {
                if (u < c4 && j + u >= first) {
                    const int jj = j + u;
                    const int slot = jj < n0 ? jj : n0 + anc[jj - n0];
                    kv[u].load(Kg + (size_t)slot * D);
                    vv[u].load(Vg + (size_t)slot * D);
                }
            }
            // head by head: the batch's scores, then the head's fold over them in key order
#pragma unroll
            for (int hh = 0; hh < G; ++hh) {
                float sc[B];
                attn_scores<E, B>(sc, qf[hh], kv, scale);
#pragma unroll
                for (int u = 0; u < B; ++u) {
                    if (u < c4 && j + u >= first) {
                        const float mn = fmaxf(m[hh], sc[u]);
                        const float corr = expf(m[hh] - mn);
                        const float p = expf(sc[u] - mn);
                        l[hh] = l[hh] * corr + p;
#pragma unroll
                        for (int e = 0; e < E; ++e) acc[hh][e] = fmaf(p, vv[u].at(e), acc[hh][e] * corr);
                        m[hh] = mn;
                    }
                }
            }
        }
    }
    // every warp's state for every head, then warp h folds head h's eight in order and writes its row, or its
    // partial and - the last split to arrive - the row from every split
    __shared__ float sm_m[G][8], sm_l[G][8];
    __shared__ float sm_acc[G][8][D];
#pragma unroll
    for (int hh = 0; hh < G; ++hh) {
        if (lane == 0) {
            sm_m[hh][w] = m[hh];
            sm_l[hh][w] = l[hh];
        }
#pragma unroll
        for (int e = 0; e < E; ++e) sm_acc[hh][w][lane * E + e] = acc[hh][e];
    }
    __syncthreads();
    if (w >= G) return;
    const int hh = w;
    float M = sm_m[hh][0], L = sm_l[hh][0], o[E];
#pragma unroll
    for (int e = 0; e < E; ++e) o[e] = sm_acc[hh][0][lane * E + e];
    for (int u = 1; u < 8; ++u) {
        const float mu = sm_m[hh][u];
        if (mu == NEG_INF) continue;
        if (M == NEG_INF) {
            M = mu;
            L = sm_l[hh][u];
#pragma unroll
            for (int e = 0; e < E; ++e) o[e] = sm_acc[hh][u][lane * E + e];
            continue;
        }
        const float Mn = fmaxf(M, mu);
        const float c0 = expf(M - Mn), c1 = expf(mu - Mn);
        L = L * c0 + sm_l[hh][u] * c1;
#pragma unroll
        for (int e = 0; e < E; ++e) o[e] = o[e] * c0 + sm_acc[hh][u][lane * E + e] * c1;
        M = Mn;
    }
    const size_t row = ((size_t)t * Hq + g * G + hh);
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
    __syncwarp();
    int last = 0;
    if (lane == 0) last = (atomicAdd(cnt + row, 1) == S_active - 1) ? 1 : 0;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) return;
    __threadfence();
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

// btb_attn_split_d{D}'s launch - grid (Hq, T, S) - plus `sharedp`: from a prefix of *sharedp keys on (n0 and
// the threshold both read on the card, so one captured graph serves every context and the host can move the
// threshold between replays), block (g G, t, s) runs its group's G heads and the group's other blocks leave;
// below it every block runs its own head. Both give every head the same bits.
#define ATTN_SPLIT_GQA_K(G, D)                                                                               \
    extern "C" __global__ void __launch_bounds__(256) btb_attn_split_gqa##G##_d##D(                          \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ n0p, const int* __restrict__ par, int T, int Hq,     \
        int Hk, int cap, float scale, float* __restrict__ part_m, float* __restrict__ part_l,                \
        float* __restrict__ part_acc, int* __restrict__ cnt, int S, int win, const int* __restrict__ sharedp) { \
        const int n0 = *n0p;                                                                                 \
        if (blockIdx.z * ATTN_SPLIT >= n0 + T) return;                                                       \
        if (n0 >= *sharedp) {                                                                                \
            if (blockIdx.x % G) return;                                                                      \
            attn_split_gqa<D, G>(q, K, V, out, n0, par, blockIdx.x / G, T, Hq, cap, scale, part_m, part_l,   \
                                 part_acc, cnt, win);                                                        \
        } else {                                                                                             \
            attn_decode_split<D, WALK_TREE>(q, K, V, out, n0p, par, nullptr, nullptr, nullptr, 0, 0, T, Hq,  \
                                            Hk, cap, scale, part_m, part_l, part_acc, cnt, S, win);          \
        }                                                                                                    \
    }
ATTN_SPLIT_GQA_K(2, 64)
ATTN_SPLIT_GQA_K(2, 128)
ATTN_SPLIT_GQA_K(2, 256)
ATTN_SPLIT_GQA_K(4, 64)
ATTN_SPLIT_GQA_K(4, 128)
ATTN_SPLIT_GQA_K(4, 256)
ATTN_SPLIT_GQA_K(8, 64)
ATTN_SPLIT_GQA_K(8, 128)

// ---------------------------------------------------------------------------------------------------------
// norm + rope + cache write: qkv [T, (Hq + 2 Hk) * D] bf16 (the merged projection's output), wq/wk [D] the
// q/k norm weights (null: no norm), cos/sin [positions, D] bf16 (the model's own rotary tables), the rows'
// positions n0 + depth[t], the cache slots n0 + t of K/V [Hk, cap, D]. One warp per (head, t): lanes 0-15 hold the first half of
// the head's dims and 16-31 the second, so rotate_half is a shuffle with lane ^ 16. The norm is the fused
// F.rms_norm (fp32 throughout, one rounding), the rope the engine's fused form: q*cos rounded to bf16, then
// one fused multiply-add with the rotated half and sin, rounded once. Warps past Hq + Hk copy the v rows.
// ROWS: row t is a sequence of its own (the attention's RowsLayout), at position len + steps, its row written
// to its step's slot.
// ---------------------------------------------------------------------------------------------------------
template <int D, bool ROWS>
__device__ __forceinline__ void norm_rope_kv(const bf16* __restrict__ qkv, const bf16* __restrict__ wq,
                                             const bf16* __restrict__ wk, float eps, const bf16* __restrict__ cos_t,
                                             const bf16* __restrict__ sin_t, const int* __restrict__ n0p,
                                             const int* __restrict__ depth, const int* __restrict__ rw,
                                             bf16* __restrict__ K, bf16* __restrict__ V, bf16* __restrict__ qo,
                                             int T, int Hq, int Hk, int cap, int centered) {
    constexpr int E = D / 32;
    const int head = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    int slot, pos;
    if constexpr (ROWS) {
        const RowsLayout r(rw, t);
        if (r.len < 0) return;
        pos = r.len + r.step;
        slot = r.slot(pos);
    } else {
        const int n0 = *n0p;
        slot = n0 + t;
        pos = n0 + depth[t];
    }
    const int width = (Hq + 2 * Hk) * D;
    const bf16* src = qkv + (size_t)t * width + (size_t)head * D + lane * E;
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
        norm_rope_kv<D, false>(qkv, wq, wk, eps, cos_t, sin_t, n0p, depth, nullptr, K, V, qo, T, Hq, Hk, cap,  \
                               centered);                                                                    \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(32) btb_norm_rope_kv_rows_d##D(                             \
        const bf16* __restrict__ qkv, const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,   \
        const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t, const int* __restrict__ rw,          \
        bf16* __restrict__ K, bf16* __restrict__ V, bf16* __restrict__ qo, int T, int Hq, int Hk, int cap,   \
        int centered) {                                                                                      \
        norm_rope_kv<D, true>(qkv, wq, wk, eps, cos_t, sin_t, nullptr, nullptr, rw, K, V, qo, T, Hq, Hk, cap,  \
                              centered);                                                                     \
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
// end, where a memcpy node was its own submission. The token lands first, with the card's clock at the step's end
// (ns, `out_ts`: when the steps ran, whatever the host was doing when it looked), and a system fence orders them
// before the length, so a host that has seen the length also sees the token and the time.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void btb_publish(const int* __restrict__ n0, const long long* __restrict__ ids,
                                       long long* out_tok, int* out_n0, long long* out_ts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out_tok = *ids;
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        *out_ts = (long long)t;
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
// The gated DeltaNet over T nodes (the host's btb_delta_step, node by node), one block per value head h. Node j's
// input is row j of mixed [T, C] (C = 2 Hk dk + Hv dv, the q | k | v projection before its conv), z [T, Hv dv],
// a and b [T, Hv]; its conv window is its own row and its ancestors' back up the tree (`parents`, -1 at the
// root; a chain when null), then the pass's starting window conv0 [C, K]; its recurrent state steps from its
// parent's. With `scratch` [T, Hv, dk, dv] each node's state is written there and `state` [Hv, dk, dv] is only
// read: a verify pass over a tree, the cache untouched until the accepted path is known. Without it the nodes
// are a chain stepped in place on `state`: the one-token step, and the accepted path's commit. The conv state
// is never written here - the caller keeps the window's last K rows (btb_conv_window). Every sum runs in an order
// fixed by the element's index, never by T: a node computes bit for bit as the same node of a one-token step.
//
// btb_delta_nodes_bf16 is the same walk over the card program's own buffers. A node's inputs are read in place as
// bf16 columns of the merged projection `proj` (row stride ps; q|k|v from column oq, z from oz, b from ob, a from
// oa) - widening a bf16 is exact, so a node's float math is btb_delta_nodes's on the widened rows, bit for bit -
// and its output is rounded to bf16 as the module's `.to(x.dtype)` rounds it. Node j reads proj row rows[j] (j
// when null): the commit re-steps the accepted path's rows of the verify pass's projection as a chain. A negative
// row is padding and ends the pass, so a graph captured at one width serves every shorter path. `slots` [T] names
// the scratch slot a node's state goes to (families/qwen4/verify.py `_slots`: a node steps in its parent's slot
// when it is the parent's last child, every earlier child having read the parent's state already), so a chain of
// any length needs one slot and a tree one per branch; null is a slot a node. In place is safe: a thread reads
// each state element of its column before it writes that element, and no other thread touches the column.
// ---------------------------------------------------------------------------------------------------------
#define DELTA_MAX_K 8
#define DELTA_THREADS 128

__device__ __forceinline__ float delta_block_sum(float v, float* red) {
    // a fixed-order tree over the block's threads (DELTA_THREADS a power of two)
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = DELTA_THREADS / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float delta_silu(float x) { return x < -88.f ? 0.f : x / (1.f + expf(-x)); }
__device__ __forceinline__ float delta_sigmoid(float x) { return x < -88.f ? 0.f : 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float ld_f(float v) { return v; }
__device__ __forceinline__ float ld_f(bf16 v) { return bf2f(v); }
__device__ __forceinline__ void st_f(float* p, float v) { *p = v; }
__device__ __forceinline__ void st_f(bf16* p, float v) { *p = f2bf(v); }

// where the nodes' inputs live: row r's conv input column c at x[r * xs + c], its z, a and b likewise
template <typename In>
struct DeltaIn {
    const In* x;
    const In* z;
    const In* a;
    const In* b;
    size_t xs, zs, as, bs;
};

template <typename In, typename Out>
__device__ __forceinline__ void delta_nodes_run(const DeltaIn<In> in, const int* __restrict__ rowp,
                                                const float* __restrict__ conv_w, const float* __restrict__ conv_b,
                                                const float* __restrict__ conv0, float* state, float* scratch,
                                                const int* __restrict__ slots, const int* __restrict__ parents,
                                                const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                                const float* __restrict__ norm_w, float eps, int gate, int T, int K,
                                                int Hk, int Hv, int dk, int dv, Out* __restrict__ out, size_t os) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int rep = Hv / Hk;
    const int hk = h / rep;
    const int key_dim = Hk * dk;
    extern __shared__ float sh[];
    float* qs = sh;          // [dk] this head's conv'd query, then normalised
    float* ks = qs + dk;     // [dk] its key
    float* vs = ks + dk;     // [dv] its value
    float* core = vs + dv;   // [dv] S^T q
    float* red = core + dv;  // [DELTA_THREADS] the reductions
    const float scale = (float)(1.0 / sqrt((double)dk));
    const size_t hs = (size_t)dk * dv;  // one head's state
    auto row_of = [&](int n) { return rowp != nullptr ? rowp[n] : n; };
    auto slot_of = [&](int n) { return slots != nullptr ? slots[n] : n; };
    for (int j = 0; j < T; ++j) {
        const int rj = row_of(j);
        if (rj < 0) break;  // padding: the pass ends here
        // the window's rows, oldest first: node j and its ancestors, then the starting window's newest columns
        int win[DELTA_MAX_K];
        int n = j, have = 0;
        for (int t = K - 1; t >= 0; --t) {
            win[t] = n >= 0 ? row_of(n) : -1;  // -1 past the pass's first node
            if (n >= 0) {
                n = parents != nullptr ? parents[n] : n - 1;
                ++have;
            }
        }
        const int pre = K - have;  // taps read from conv0: its last `pre` columns, oldest first
        auto conv_at = [&](int c) {
            float acc = conv_b != nullptr ? conv_b[c] : 0.f;
            for (int t = 0; t < K; ++t) {
                const float x =
                    t < pre ? conv0[(size_t)c * K + (K - pre + t)] : ld_f(in.x[(size_t)win[t] * in.xs + c]);
                acc += conv_w[(size_t)c * K + t] * x;
            }
            return delta_silu(acc);
        };
        for (int i = tid; i < dk; i += DELTA_THREADS) {
            qs[i] = conv_at(hk * dk + i);
            ks[i] = conv_at(key_dim + hk * dk + i);
        }
        for (int i = tid; i < dv; i += DELTA_THREADS) vs[i] = conv_at(2 * key_dim + h * dv + i);
        __syncthreads();
        float q2 = 0.f, k2 = 0.f;
        for (int i = tid; i < dk; i += DELTA_THREADS) {
            q2 += qs[i] * qs[i];
            k2 += ks[i] * ks[i];
        }
        const float qi = 1.f / sqrtf(delta_block_sum(q2, red) + 1e-6f);
        const float ki = 1.f / sqrtf(delta_block_sum(k2, red) + 1e-6f);
        for (int i = tid; i < dk; i += DELTA_THREADS) {
            qs[i] = (qs[i] * qi) * scale;
            ks[i] = ks[i] * ki;
        }
        __syncthreads();
        const float bj = ld_f(in.b[(size_t)rj * in.bs + h]);
        const float beta = 1.f / (1.f + expf(-bj));
        const float x = ld_f(in.a[(size_t)rj * in.as + h]) + dt_bias[h];
        const float g = -expf(a_log[h]) * (x > 20.f ? x : log1pf(expf(x)));
        const float decay = expf(g);
        // the state this node steps from, and where it steps to
        const int par = scratch == nullptr ? -1 : (parents != nullptr ? parents[j] : j - 1);
        const float* src =
            (scratch != nullptr && par >= 0) ? scratch + ((size_t)slot_of(par) * Hv + h) * hs : state + h * hs;
        float* dst = scratch != nullptr ? scratch + ((size_t)slot_of(j) * Hv + h) * hs : state + h * hs;
        // a column of the state a thread: kv = S^T k, delta = (v - kv) beta, S = decay S + k delta^T, core = S^T q
        for (int col = tid; col < dv; col += DELTA_THREADS) {
            float kv = 0.f;
            for (int d = 0; d < dk; ++d) kv += (src[(size_t)d * dv + col] * decay) * ks[d];
            const float dl = (vs[col] - kv) * beta;
            float c = 0.f;
            for (int d = 0; d < dk; ++d) {
                const float s = src[(size_t)d * dv + col] * decay + ks[d] * dl;
                dst[(size_t)d * dv + col] = s;
                c += s * qs[d];
            }
            core[col] = c;
        }
        __syncthreads();
        if (out != nullptr) {  // uniform over the block, so the reduction's barriers are too
            float c2 = 0.f;
            for (int i = tid; i < dv; i += DELTA_THREADS) c2 += core[i] * core[i];
            const float rms = 1.f / sqrtf(delta_block_sum(c2, red) / (float)dv + eps);
            const In* zr = in.z + (size_t)rj * in.zs + (size_t)h * dv;
            Out* o = out + (size_t)j * os + (size_t)h * dv;
            for (int col = tid; col < dv; col += DELTA_THREADS) {
                const float zv = ld_f(zr[col]);
                const float gz = gate == 1 ? delta_sigmoid(zv) : delta_silu(zv);
                st_f(o + col, (norm_w[col] * (core[col] * rms)) * gz);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(DELTA_THREADS) btb_delta_nodes(
    const float* __restrict__ mixed, const float* __restrict__ z, const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ conv_w, const float* __restrict__ conv_b,
    const float* __restrict__ conv0, float* state, float* scratch, const int* __restrict__ parents,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* __restrict__ norm_w, float eps,
    int gate, int T, int C, int K, int Hk, int Hv, int dk, int dv, float* __restrict__ out) {
    const DeltaIn<float> in{mixed, z, a, b, (size_t)C, (size_t)Hv * dv, (size_t)Hv, (size_t)Hv};
    delta_nodes_run<float, float>(in, nullptr, conv_w, conv_b, conv0, state, scratch, nullptr, parents, a_log,
                                  dt_bias, norm_w, eps, gate, T, K, Hk, Hv, dk, dv, out, (size_t)Hv * dv);
}

extern "C" __global__ void __launch_bounds__(DELTA_THREADS) btb_delta_nodes_bf16(
    const bf16* __restrict__ proj, int ps, int oq, int oz, int ob, int oa, const int* __restrict__ rows,
    const float* __restrict__ conv_w, const float* __restrict__ conv_b, const float* __restrict__ conv0, float* state,
    float* scratch, const int* __restrict__ slots, const int* __restrict__ parents, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, const float* __restrict__ norm_w, float eps, int gate, int T, int K, int Hk,
    int Hv, int dk, int dv, bf16* __restrict__ out) {
    const size_t s = (size_t)ps;
    const DeltaIn<bf16> in{proj + oq, proj + oz, proj + oa, proj + ob, s, s, s, s};
    delta_nodes_run<bf16, bf16>(in, rows, conv_w, conv_b, conv0, state, scratch, slots, parents, a_log, dt_bias,
                                norm_w, eps, gate, T, K, Hk, Hv, dk, dv, out, (size_t)Hv * dv);
}

// ---------------------------------------------------------------------------------------------------------
// the DeltaNet's conv state after a chain of steps: conv [C, K] float32 in place becomes the last K of (its K
// columns, then the qkv inputs of proj rows rows[0..n) - j when null - read from column oq of a ps-strided row).
// The step's and the commit's partner to btb_delta_nodes_bf16, launched after it (the nodes read the old window).
// A negative row is padding and ends the rows, as there. A thread a channel, its old window held in registers:
// in place, and nothing summed, so a row count changes nothing but which columns come from where.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_conv_window(float* __restrict__ conv,
                                                                  const bf16* __restrict__ proj, int ps, int oq,
                                                                  const int* __restrict__ rows, int n, int C, int K) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    int cnt = n;
    if (rows != nullptr) {
        cnt = 0;
        while (cnt < n && rows[cnt] >= 0) ++cnt;
    }
    float old[DELTA_MAX_K];
    for (int t = 0; t < K; ++t) old[t] = conv[(size_t)c * K + t];
    for (int t = 0; t < K; ++t) {
        const int s = cnt + t;  // its place in (the old window, then the rows' inputs)
        float v;
        if (s < K) {
            v = old[s];
        } else {
            const int r = rows != nullptr ? rows[s - K] : s - K;
            v = bf2f(proj[(size_t)r * ps + oq + c]);
        }
        conv[(size_t)c * K + t] = v;
    }
}

// MXFP4 experts widened to bf16 for a grouped call, in one pass: a thread a 32-value block, its 16 bytes and its
// scale read once and its 32 bf16 written once, for the experts at `seats` of a depot's stacks (`blocks` [seats,
// groups, 16], `scales` [seats, groups], out [len(seats), groups, 32]). A value is its fp4 code (the low nibble
// first) times 2^(scale - 127), made as ldexpf makes it and rounded to bf16 - exact wherever it is finite, as no fp4
// code needs more than two of bf16's bits - the torch widening's values (`dequant_blocks`) bit for bit.
__constant__ float BTB_FP4[16] = {0.f,  .5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                  -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

extern "C" __global__ void __launch_bounds__(256) btb_mx4_widen(const unsigned char* __restrict__ blocks,
                                                                const unsigned char* __restrict__ scales,
                                                                const int* __restrict__ seats, long long groups,
                                                                bf16* __restrict__ out) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const long long at = (long long)seats[blockIdx.y] * groups + g;
    const uint4 q = __ldcs(reinterpret_cast<const uint4*>(blocks) + at);
    const int e = (int)scales[at] - 127;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&q);
    __align__(16) bf16 v[32];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        v[2 * j] = f2bf(ldexpf(BTB_FP4[b[j] & 15], e));
        v[2 * j + 1] = f2bf(ldexpf(BTB_FP4[b[j] >> 4], e));
    }
    uint4* o = reinterpret_cast<uint4*>(out) + ((long long)blockIdx.y * groups + g) * 4;
    const uint4* s = reinterpret_cast<const uint4*>(v);
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = s[j];
}

// =========================================================================================================
// Qwen4 (qwen4_exp): the hyper-connection streams, the gated attention's partial rope and output gate, and the
// MoE's router and shared-expert combine. Each kernel reproduces the module's own bf16 rounding sequence (one
// rounding wherever torch's bf16 op rounds), and each row's arithmetic runs in an order fixed by the element's
// index: a row's reductions stay inside its own block or warp, and no loop bound or split depends on T. So row t
// of a T-row launch is row t of the one-row launch, bit for bit, whatever rows ride beside it or pad the pass.
// =========================================================================================================

// torch's sigmoid in fp32 (at::native sigmoid_kernel_cuda: 1 / (1 + e^-x)), as its bf16 op computes before rounding
__device__ __forceinline__ float act_sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

// btb_add_rmsnorm's reduction: thread i's own sum, then the fixed tree over a 256-thread block
__device__ __forceinline__ float block_sum256(float v, float* red) {
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

// ---------------------------------------------------------------------------------------------------------
// the hyper-connection write and read (Qwen4ExpTextDecoderLayer's `hyper_input + y * inject` per stream, then the
// next Qwen4ExpTextGatedResidual's hc_norm): h [T, G, H] bf16 in place, G streams (hc_count) of H. With y [T, H]
// given, h[t, g] = bf16(h + bf16(y * inj[t, g])) - the decoder layer's two bf16 ops - with inj [T, G] the block's
// inject weights (btb_hc_mix). Then x [T, G, H] = bf16(h * rstd * (1 + w)) with rstd over stream g's H elements
// alone (the norm's group_size H) and w [G H] its zero-centred weight (or * w, `centered` 0). Block (t, g), 256
// threads: btb_add_rmsnorm's fixed-order sum within the one stream of the one row, so T enters nowhere.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_hc_rmsnorm(bf16* __restrict__ h, const bf16* __restrict__ y,
                                                                 const bf16* __restrict__ inj,
                                                                 const bf16* __restrict__ w, float eps,
                                                                 bf16* __restrict__ x, int H, int G, int centered) {
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const size_t base = ((size_t)t * G + g) * H;
    bf16* hr = h + base;
    bf16* xr = x + base;
    const bf16* yr = y == nullptr ? nullptr : y + (size_t)t * H;
    const float iv = y == nullptr ? 0.f : bf2f(inj[(size_t)t * G + g]);
    float ss = 0.f;
    for (int i = tid; i < H; i += 256) {
        float hv = bf2f(hr[i]);
        if (yr != nullptr) {
            hv = bfround(hv + bfround(bf2f(yr[i]) * iv));
            hr[i] = f2bf(hv);
        }
        ss = fmaf(hv, hv, ss);
    }
    __shared__ float red[256];
    const float rstd = rsqrtf(block_sum256(ss, red) / (float)H + eps);
    const bf16* wg = w + (size_t)g * H;
    for (int i = tid; i < H; i += 256) {
        const float wv = bf2f(wg[i]);
        xr[i] = f2bf(bf2f(hr[i]) * rstd * (centered ? 1.f + wv : wv));
    }
}

// ---------------------------------------------------------------------------------------------------------
// the mixer's low-rank activation: act [T, R] = bf16(silu(bf16(dn / G))) over the first R columns of the down
// projection's rows (dn [T, ds], ds = R + G when the block-inject logits share its gemv). The division as torch's
// bf16 div by a Python scalar computes it - a multiply by the fp32 reciprocal - then silu in fp32 as btb_silu_mul.
// Elementwise: nothing depends on another row.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_hc_act(const bf16* __restrict__ dn, int ds,
                                                             bf16* __restrict__ act, int R, int G) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= R) return;
    const float v = bfround(bf2f(dn[(size_t)t * ds + i]) * (1.f / (float)G));
    act[(size_t)t * R + i] = f2bf(act_silu(v));
}

// ---------------------------------------------------------------------------------------------------------
// the mix: xn [T, G, H] the normed streams (btb_hc_rmsnorm's x), up [T, G, H] the up projection's logits, mixed
// [T, H] = bf16(sum_g bf16(bf16(sigmoid(up)) * xn) * (1 / G)) - torch's bf16 sigmoid, bf16 product, then its mean
// over the stream dim (an fp32 sum, g = 0 .. G-1 in order, times the fp32 factor 1/G, rounded once). With inj
// given, the block's inject weights too: inj [T, G] = 2 * bf16(sigmoid(bf16(dn[t, R + g] / G))), the last G
// columns of the merged [down | block_inject] gemv's row (dn [T, ds]). Null inj: the model's final mixer, which
// has no inject. Elementwise over (t, i), the G-term sum inside the one thread.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_hc_mix(const bf16* __restrict__ xn, const bf16* __restrict__ up,
                                                             const bf16* __restrict__ dn, int ds,
                                                             bf16* __restrict__ mixed, bf16* __restrict__ inj, int H,
                                                             int G, int R) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    const float rg = 1.f / (float)G;
    if (i < H) {
        float s = 0.f;
        for (int g = 0; g < G; ++g) {
            const size_t k = ((size_t)t * G + g) * H + i;
            const float m = bfround(act_sigmoid(bf2f(up[k])));
            s += bfround(m * bf2f(xn[k]));
        }
        mixed[(size_t)t * H + i] = f2bf(s * rg);
    }
    if (inj != nullptr && blockIdx.x == 0 && threadIdx.x < G) {
        const int g = threadIdx.x;
        const float v = bfround(bf2f(dn[(size_t)t * ds + R + g]) * rg);
        inj[(size_t)t * G + g] = f2bf(2.f * bfround(act_sigmoid(v)));
    }
}

// ---------------------------------------------------------------------------------------------------------
// the router (Qwen4ExpTextTopKRouter): logits [T, ls] bf16, the E expert logits of a row then (in the merged
// [E + 1, H] gemv) the shared expert's gate logit at column E, which btb_moe_combine reads where it is. Per row: the
// fp32 softmax over the E logits (max, e^(l - max), their sum, p = e / sum), the top k in the total order (p desc,
// index asc - a unique pick however the row is walked, and the logits' order, so torch's topk on distinct
// values), their weights p / sum(top k) summed in rank order and rounded to bf16, into idx [T, k] int32 and w [T,
// k] bf16. One block a row: the sums a thread's own strided elements then the fixed tree; nothing crosses rows.
//
// With hidx/hw (pinned host memory, device-addressable under unified addressing) the rows go to the host as well
// - the experts the host serves need them without a copy node - and, btb_publish's handoff over a grid, each block
// fences its row out to the system before it counts itself in `cnt` (zero between launches); the last block in
// fences again and writes *seq to *hseq and resets the count. A host that has seen the sequence has seen every row.
// ---------------------------------------------------------------------------------------------------------
#define ROUTE_THREADS 256
#define ROUTE_MAX_K 32

extern "C" __global__ void __launch_bounds__(ROUTE_THREADS) btb_moe_route(
    const bf16* __restrict__ logits, int ls, int E, int k, int* __restrict__ idx, bf16* __restrict__ w, int* hidx,
    bf16* hw, unsigned* cnt, const int* __restrict__ seq, int* hseq) {
    extern __shared__ float pr[];  // [E] the row's probabilities; a picked one set below every live one
    __shared__ float red[ROUTE_THREADS];
    __shared__ int redi[ROUTE_THREADS];
    __shared__ float topv[ROUTE_MAX_K];
    __shared__ int topi[ROUTE_MAX_K];
    const int t = blockIdx.x, tid = threadIdx.x;
    const bf16* lr = logits + (size_t)t * ls;
    float m = NEG_INF;
    for (int e = tid; e < E; e += ROUTE_THREADS) m = fmaxf(m, bf2f(lr[e]));
    red[tid] = m;
    __syncthreads();
    for (int s = ROUTE_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float s = 0.f;
    for (int e = tid; e < E; e += ROUTE_THREADS) {
        const float p = expf(bf2f(lr[e]) - m);
        pr[e] = p;
        s += p;
    }
    s = block_sum256(s, red);
    for (int e = tid; e < E; e += ROUTE_THREADS) pr[e] = pr[e] / s;
    __syncthreads();
    // k rounds of the total order's maximum; probabilities are >= 0, so -1 is below every live one and a pick is -2.
    // A row whose probabilities are NaN (a NaN or infinite logit) compares below nothing: it still takes live experts,
    // the first it meets, their weights NaN as torch's topk leaves them - never the "none" index, written through
    constexpr int NONE = 0x7fffffff;
    for (int j = 0; j < k; ++j) {
        float bv = -1.f;
        int bi = NONE;
        for (int e = tid; e < E; e += ROUTE_THREADS) {
            const float v = pr[e];
            if (v > bv || (bi == NONE && v != -2.f)) {  // e ascends within the thread, so a tie keeps the lower index
                bv = v;
                bi = e;
            }
        }
        red[tid] = bv;
        redi[tid] = bi;
        __syncthreads();
        for (int st = ROUTE_THREADS / 2; st > 0; st >>= 1) {
            if (tid < st) {
                const float ov = red[tid + st];
                const int oi = redi[tid + st];
                if (oi != NONE && (redi[tid] == NONE || ov > red[tid] || (ov == red[tid] && oi < redi[tid]))) {
                    red[tid] = ov;
                    redi[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            topv[j] = red[0];
            topi[j] = redi[0];
            if (redi[0] != NONE) pr[redi[0]] = -2.f;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.f;
        for (int j = 0; j < k; ++j) sum += topv[j];
        for (int j = 0; j < k; ++j) {
            const bf16 wv = f2bf(topv[j] / sum);
            const size_t o = (size_t)t * k + j;
            idx[o] = topi[j];
            w[o] = wv;
            if (hidx != nullptr) {
                hidx[o] = topi[j];
                hw[o] = wv;
            }
        }
        if (hseq != nullptr) {
            __threadfence_system();
            if (atomicAdd(cnt, 1u) == gridDim.x - 1) {
                *cnt = 0;
                __threadfence_system();
                *hseq = *seq;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// the MoE's output (Qwen4ExpTextSparseMoeBlock): y [T, H] = bf16(yr + bf16(bf16(sigmoid(sg)) * ys)), yr the routed
// experts' sum, ys the shared expert's output, sg the shared-expert gate logit of row t at sg[t * sgs] (the
// router's merged logits row, column E). torch's three bf16 ops. y may be yr (in place): a thread reads its
// element before it writes it. Elementwise.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_moe_combine(const bf16* yr, const bf16* __restrict__ ys,
                                                                  const bf16* __restrict__ sg, int sgs, bf16* y,
                                                                  int H) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= H) return;
    const float gate = bfround(act_sigmoid(bf2f(sg[(size_t)t * sgs])));
    const size_t k = (size_t)t * H + i;
    y[k] = f2bf(bf2f(yr[k]) + bfround(gate * bf2f(ys[k])));
}

// ---------------------------------------------------------------------------------------------------------
// the attention's output gate (Qwen4ExpTextAttention: attn * sigmoid(gate) before o_proj). att [T, C] (C = Hq D)
// and its gate at its own place in the q projection: gate element (t, c) at gate[t * gs + (c / D) * hs + c % D]
// (q_proj's rows interleave each head's D query dims with its D gate dims, so the gate is the q projection's
// buffer offset by D with head stride hs = 2 D). x = bf16(att * bf16(sigmoid(gate))), torch's two bf16 ops.
// btb_sigmoid_mul writes x out [T, C] for the tensor-core matvec; btb_gemv_sgate_bf16_m{M} is o_proj with x made
// in its load, as btb_gemv_silu folds its gate: the same x values and the same per-(m, r) accumulation order as
// btb_gemv_bf16_m{M} (chunks c, c + 32, .. of a lane, a chunk's 8 in index order, then the butterfly), so it is
// btb_gemv_bf16 over btb_sigmoid_mul's x bit for bit, and row m's sequence does not depend on M.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_sigmoid_mul(const bf16* __restrict__ att,
                                                                  const bf16* __restrict__ gate, int gs, int D, int hs,
                                                                  bf16* __restrict__ out, int C) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= C) return;
    const float g = bf2f(gate[(size_t)t * gs + (size_t)(i / D) * hs + i % D]);
    const size_t k = (size_t)t * C + i;
    out[k] = f2bf(bf2f(att[k]) * bfround(act_sigmoid(g)));
}

template <int M>
__device__ __forceinline__ void gemv_sgate_rows(const bf16* __restrict__ w, const bf16* __restrict__ att,
                                                const bf16* __restrict__ gate, int gs, int D, int hs,
                                                bf16* __restrict__ y, int R, int C) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= R) return;
    const int nchunk = C >> 3;
    const uint4* wrow = reinterpret_cast<const uint4*>(w) + (size_t)r * nchunk;
    const uint4* arow = reinterpret_cast<const uint4*>(att);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        uint4 wv = __ldcs(wrow + c);
        const bf16* wp = reinterpret_cast<const bf16*>(&wv);
        const int e0 = c << 3;  // a chunk never crosses a head: D is a multiple of 8
        const size_t goff = (size_t)(e0 / D) * hs + e0 % D;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 av = __ldg(arow + (size_t)m * nchunk + c);
            uint4 gv = __ldg(reinterpret_cast<const uint4*>(gate + (size_t)m * gs + goff));
            const bf16* ap = reinterpret_cast<const bf16*>(&av);
            const bf16* gp = reinterpret_cast<const bf16*>(&gv);
            float a = acc[m];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float xe = bfround(bf2f(ap[e]) * bfround(act_sigmoid(bf2f(gp[e]))));
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

#define GEMV_SGATE(M)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_sgate_bf16_m##M(                              \
        const bf16* __restrict__ w, const bf16* __restrict__ att, const bf16* __restrict__ gate, int gs, int D, \
        int hs, bf16* __restrict__ y, int R, int C) {                                                        \
        gemv_sgate_rows<M>(w, att, gate, gs, D, hs, y, R, C);                                                \
    }
GEMV_SGATE(1)
GEMV_SGATE(2)
GEMV_SGATE(4)
GEMV_SGATE(8)
GEMV_SGATE(16)
GEMV_SGATE(32)

// ---------------------------------------------------------------------------------------------------------
// Qwen4's q/k norm + partial rope + cache write: btb_norm_rope_kv's walk over Qwen4's layout. qkv [T, width] bf16
// the merged projection's rows: q head h at column h * qhs (qhs = 2 D where q_proj interleaves each head's gate,
// D for the indexer's q), k head g at koff + g D, v head g at voff + g D. The norm is Qwen4ExpTextRMSNorm (fp32,
// rsqrt(mean + eps), times (1 + w), one rounding; `centered` 0 for * w; null weights skip it). The rope is the
// module's apply_rotary_pos_emb on the first ROT dims only (partial_rotary_factor): cos/sin [positions, ts] bf16
// hold ROT columns a row, and torch's bf16 ops round three times - q*cos, rotate_half(q)*sin, their sum. Lanes
// hold E = D/32 consecutive dims, so the rotary half's partner is lane ^ (ROT / 2E) (ROT a multiple of 2E, ROT/2E a
// power of two) and the lanes past ROT pass their normed dims through. Positions and slots as btb_norm_rope_kv:
// row t at position n0 + depth[t], its K/V rows at cache slot n0 + t of K/V [Hk, cap, D]; q to qo [T, Hq, D].
// `rawk` 1 is the QSA indexer: its q normed and roped the same way, its key head copied to the raw-key arena K as
// it came out of the projection (the indexer pools and norms raw keys later), and no V (grid Hq + Hk heads).
// One warp per (head, t): the norm's sum is the warp's butterfly over a head's own dims, whatever T is.
// ---------------------------------------------------------------------------------------------------------
// a head's norm and partial rope, one warp over the head's D = 32 E dims (lane l holds dims l E .. l E + E - 1): the
// Qwen4ExpTextRMSNorm of the head, fp32 throughout (the sum of squares the lane's own E in order, then the warp's
// butterfly), times (1 + w) (or w), rounded once
template <int E>
__device__ __forceinline__ void head_norm(float (&v)[E], const bf16* __restrict__ wn, float eps, int centered) {
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < E; ++e) ss = fmaf(v[e], v[e], ss);
    ss = warp_sum(ss);
    const float rstd = rsqrtf(ss / (float)(32 * E) + eps);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const float wv = bf2f(wn[lane * E + e]);
        v[e] = bfround(v[e] * rstd * (centered ? 1.f + wv : wv));
    }
}

// apply_rotary_pos_emb over the first `rot` dims (cos/sin the position's row of rot columns), torch's three bf16
// roundings: o = bf16(v * cos) + bf16(rotate_half(v) * sin), rotate_half's partner the lane ^ (rot / 2E); the dims
// past rot pass through. o is the sum before its last rounding (the caller's store rounds it).
template <int E>
__device__ __forceinline__ void rope_part(const float (&v)[E], float (&o)[E], const bf16* __restrict__ cos_row,
                                          const bf16* __restrict__ sin_row, int rot) {
    const int lane = threadIdx.x & 31;
    const int half = rot / (2 * E);  // lanes in a rotary half
    float pv[E];
#pragma unroll
    for (int e = 0; e < E; ++e) pv[e] = __shfl_xor_sync(0xffffffffu, v[e], half & 31);
    if (lane * E < rot) {
        // out = bf16(bf16(v * cos) + bf16(rot * sin)), rot = -v[i + ROT/2] in the first half, v[i - ROT/2] after
        const bf16* cp = cos_row + lane * E;
        const bf16* sp = sin_row + lane * E;
        const float sgn = (lane & half) ? 1.f : -1.f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const float qc = bfround(v[e] * bf2f(cp[e]));
            const float rs = bfround(sgn * pv[e] * bf2f(sp[e]));
            o[e] = qc + rs;
        }
    } else {
#pragma unroll
        for (int e = 0; e < E; ++e) o[e] = v[e];
    }
}

template <int D>
__device__ __forceinline__ void norm_rope_part(const bf16* __restrict__ qkv, int width, int qhs, int koff, int voff,
                                               const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,
                                               const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t,
                                               int ts, int rot, const int* __restrict__ n0p,
                                               const int* __restrict__ depth, bf16* __restrict__ K,
                                               bf16* __restrict__ V, bf16* __restrict__ qo, int Hq, int Hk, int cap,
                                               int centered, int rawk) {
    constexpr int E = D / 32;
    const int head = blockIdx.x, t = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int n0 = *n0p;
    const int slot = n0 + t;
    const int pos = n0 + depth[t];
    const bf16* row = qkv + (size_t)t * width;
    if (head >= Hq + Hk) {
        const int g = head - Hq - Hk;
        const bf16* src = row + voff + (size_t)g * D + lane * E;
        bf16* dst = V + ((size_t)g * cap + slot) * D + lane * E;
#pragma unroll
        for (int e = 0; e < E; ++e) dst[e] = src[e];
        return;
    }
    const bool is_q = head < Hq;
    const bf16* src = (is_q ? row + (size_t)head * qhs : row + koff + (size_t)(head - Hq) * D) + lane * E;
    bf16* dst = is_q ? (qo + ((size_t)t * Hq + head) * D + lane * E)
                     : (K + ((size_t)(head - Hq) * cap + slot) * D + lane * E);
    if (!is_q && rawk) {
#pragma unroll
        for (int e = 0; e < E; ++e) dst[e] = src[e];
        return;
    }
    const bf16* wn = is_q ? wq : wk;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) v[e] = bf2f(src[e]);
    if (wn != nullptr) head_norm<E>(v, wn, eps, centered);
    float o[E];
    rope_part<E>(v, o, cos_t + (size_t)pos * ts, sin_t + (size_t)pos * ts, rot);
#pragma unroll
    for (int e = 0; e < E; ++e) dst[e] = f2bf(o[e]);
}

#define NORMROPE_PART(D)                                                                                     \
    extern "C" __global__ void __launch_bounds__(32) btb_norm_rope_part_d##D(                                \
        const bf16* __restrict__ qkv, int width, int qhs, int koff, int voff, const bf16* __restrict__ wq,   \
        const bf16* __restrict__ wk, float eps, const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t, \
        int ts, int rot, const int* __restrict__ n0p, const int* __restrict__ depth, bf16* __restrict__ K,  \
        bf16* __restrict__ V, bf16* __restrict__ qo, int Hq, int Hk, int cap, int centered, int rawk) {      \
        norm_rope_part<D>(qkv, width, qhs, koff, voff, wq, wk, eps, cos_t, sin_t, ts, rot, n0p, depth, K, V, qo, \
                          Hq, Hk, cap, centered, rawk);                                                      \
    }
NORMROPE_PART(128)
NORMROPE_PART(256)

// =========================================================================================================
// Qwen4's sparse attention (QSA) on the card: the indexer's pick of key blocks, and the attention over the picks.
// The indexer (Qwen4ExpTextQSAIndexer) cuts a row's VISIBLE sequence - the prefix, then (a tree's node) its
// ancestors, then itself - into blocks of r = compress_ratio consecutive positions; a complete block's key is
// k_layernorm(mean of its r raw keys) roped at the block's first position; the row scores block b as
// sum_h relu(q_h . key_b) / sqrt(di) over its index heads and keeps its top block_topk blocks and its partial tail
// block - or everything it sees when its complete blocks number block_topk or fewer.
//
// raw [cap, DI] bf16 is the layer's raw-key arena (btb_norm_rope_part's `rawk` writes a pass row's key at its slot
// n0 + t), pk [NBcap, DI] bf16 the pooled keys of the COMMITTED complete blocks (b < n0 / r: no tree node changes
// them), grown by btb_qsa_pool at the start of a pass. A node's blocks past them - the prefix's partial tail, its
// ancestors, itself: at most 32 / r + 1 blocks - are pooled per node in btb_qsa_select by the same qsa_pool_block,
// from the same raw bits at the same positions, so a block pooled in flight is the block btb_qsa_pool writes once
// the path commits. That is the row invariance of the pick: node t's scores, and so its picks, are the one-token
// step's over its committed path, bit for bit, whatever else rides in the pass.
// =========================================================================================================
#define QSA_THREADS 512
#define QSA_WARPS 16
#define QSA_MAX_FLIGHT 33  // blocks past the committed ones a node completes: (n0 + 32) / r - n0 / r <= 32 / r + 1

// the cache slot of a node's logical position j: the prefix's own row, then the node's ancestor at depth j - n0
struct TreeSlot {
    const int* anc;
    int n0;
    __device__ __forceinline__ int operator()(int j) const { return j < n0 ? j : n0 + anc[j - n0]; }
};
struct PlainSlot {
    __device__ __forceinline__ int operator()(int j) const { return j; }
};

// block b's key, one warp (lane l holds dims l E ..): the r raw keys summed in order in fp32 and scaled by the fp32
// 1/r (torch's mean on the card: its sum times the factor; for r a power of two the same as its CPU sum / r), to
// bf16 (`.to(raw.dtype)`), k_layernorm (head_norm: fp32, times 1 + w, rounded once), then the partial rope at the
// block's first position b r. o is the key before its store's rounding.
template <int E, typename Slot>
__device__ __forceinline__ void qsa_pool_block(const bf16* __restrict__ raw, Slot slot_of, int b, int r,
                                               const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t,
                                               int ts, int rot, const bf16* __restrict__ kw, float eps,
                                               float (&o)[E]) {
    constexpr int DI = 32 * E;
    const int lane = threadIdx.x & 31;
    float acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) acc[e] = 0.f;
    for (int i = 0; i < r; ++i) {
        const bf16* p = raw + (size_t)slot_of(b * r + i) * DI + lane * E;
#pragma unroll
        for (int e = 0; e < E; ++e) acc[e] += bf2f(p[e]);
    }
    const float inv = 1.f / (float)r;
    float v[E];
#pragma unroll
    for (int e = 0; e < E; ++e) v[e] = bfround(acc[e] * inv);
    head_norm<E>(v, kw, eps, 1);
    const size_t pos = (size_t)b * r;
    rope_part<E>(v, o, cos_t + pos * ts, sin_t + pos * ts, rot);
}

// ---------------------------------------------------------------------------------------------------------
// btb_qsa_pool_d{128,256}: pk grown from pk_len[0] to the committed complete blocks n0 / r, both read on the card,
// so one captured graph serves every pass: grid ceil(max_new / 8) x 256 threads, warp w of block x pooling block
// pk_len + 8 x + w, the rest idle. max_new is the most the committed blocks grow between two pools (a pass's
// accepted tokens / r + 1, a prefill's blocks). Every block counts itself in pk_len[1] (zero between launches) once
// its keys are out, and the last one in moves pk_len[0] on and zeroes the count - so no block reads the length after
// it moved. An n0 shorter than the blocks pooled (a rewind) pulls pk_len[0] back to n0 / r; a caller that rewrites
// committed raw keys without passing that n0 through here lowers pk_len[0] itself. A key is a function of its r raw
// keys and its position alone.
// ---------------------------------------------------------------------------------------------------------
template <int DI>
__device__ __forceinline__ void qsa_pool(const bf16* __restrict__ raw, bf16* __restrict__ pk, int* pk_len,
                                         const int* __restrict__ n0p, const bf16* __restrict__ cos_t,
                                         const bf16* __restrict__ sin_t, int ts, int rot, const bf16* __restrict__ kw,
                                         float eps, int r, int max_new) {
    constexpr int E = DI / 32;
    __shared__ int s_lo, s_hi;
    if (threadIdx.x == 0) {
        const int hi = *n0p / r;
        const int lo = min(*(volatile int*)pk_len, hi);
        s_lo = lo;
        s_hi = min(hi, lo + max_new);
    }
    __syncthreads();
    const int b = s_lo + blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (b < s_hi) {
        float o[E];
        qsa_pool_block<E>(raw, PlainSlot{}, b, r, cos_t, sin_t, ts, rot, kw, eps, o);
        bf16* dst = pk + (size_t)b * DI + (threadIdx.x & 31) * E;
#pragma unroll
        for (int e = 0; e < E; ++e) dst[e] = f2bf(o[e]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(pk_len + 1, 1) == (int)gridDim.x - 1) {
            pk_len[0] = s_hi;
            pk_len[1] = 0;
            __threadfence();
        }
    }
}

// the scores' total order as unsigned keys: a larger float, a larger key (the sign bit flipped over, a negative's
// bits inverted), so the top k is a radix select over integers
__device__ __forceinline__ unsigned qsa_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------------------------------------------------
// btb_qsa_select_d{128,256}: node t's picks, a block of QSA_THREADS a node. qi [T, Hi, DI] bf16 its index queries
// (normed and roped: btb_norm_rope_part), n0 / par the tree as btb_attn_split walks it (par -2: a padding row, nothing
// picked). The node sees n = n0 + depth + 1 positions, nb = n / r complete blocks: nb <= k_top keeps them all (the
// reference's top-k over all of them); otherwise its blocks past the committed ones are pooled into shared memory and
// every block scored - a warp a block, its key in registers wherever it came from, for each head in ascending order
// the lane's E products in order then the warp's butterfly, relu, added to the block's sum, times 1/sqrt(DI) - into
// scores[t, :nb]. The top k in the total order (score desc, block asc) is a radix select over the keys' bits: integer
// counts, so the threshold and the ties it keeps (the lowest blocks) depend on the scores alone. sel[t, :k] takes the
// picks in ascending block order (a block-wide compaction in index order), nsel[t] their count (k, or nb when it
// keeps all). Nothing crosses nodes, and T bounds no loop but the ancestor walk: node t's picks are the T = 1
// launch's over its committed path.
// ---------------------------------------------------------------------------------------------------------
template <int DI>
__device__ __forceinline__ void qsa_select(const bf16* __restrict__ qi, const bf16* __restrict__ pk,
                                           const bf16* __restrict__ raw, const int* __restrict__ n0p,
                                           const int* __restrict__ par, const bf16* __restrict__ cos_t,
                                           const bf16* __restrict__ sin_t, int ts, int rot,
                                           const bf16* __restrict__ kw, float eps, int T, int Hi, int r, int ktop,
                                           int NBcap, float* scores, int* __restrict__ sel, int* __restrict__ nsel) {
    constexpr int E = DI / 32;
    extern __shared__ __align__(16) unsigned char qsa_smem[];  // [Hi, DI] bf16: the node's index queries
    __shared__ bf16 fl[QSA_MAX_FLIGHT][DI];                     // the node's blocks past the committed ones
    __shared__ int anc[32];
    __shared__ int s_d, s_need;
    __shared__ unsigned s_prefix;
    __shared__ unsigned hist[256];
    __shared__ int wc_eq[QSA_WARPS], wc_take[QSA_WARPS];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    if (par[t] < -1) {
        if (tid == 0) nsel[t] = 0;
        return;
    }
    const int n0 = *n0p;
    if (tid == 0) {
        int tmp[32];
        int len = 0;
        for (int p = t; p >= 0 && p < T && len < 32; p = par[p]) tmp[len++] = p;
        for (int e = 0; e < len; ++e) anc[e] = tmp[len - 1 - e];
        s_d = len - 1;
    }
    __syncthreads();
    const int n = n0 + s_d + 1, nb = n / r, nc = n0 / r;
    int* selt = sel + (size_t)t * ktop;
    if (nb <= ktop) {
        for (int b = tid; b < nb; b += QSA_THREADS) selt[b] = b;
        if (tid == 0) nsel[t] = nb;
        return;
    }
    const TreeSlot slot_of{anc, n0};
    for (int k = w; nc + k < nb; k += QSA_WARPS) {
        float o[E];
        qsa_pool_block<E>(raw, slot_of, nc + k, r, cos_t, sin_t, ts, rot, kw, eps, o);
#pragma unroll
        for (int e = 0; e < E; ++e) fl[k][lane * E + e] = f2bf(o[e]);
    }
    bf16* qs = reinterpret_cast<bf16*>(qsa_smem);
    const bf16* qrow = qi + (size_t)t * Hi * DI;
    for (int i = tid; i < Hi * DI; i += QSA_THREADS) qs[i] = qrow[i];
    __syncthreads();
    float* sc = scores + (size_t)t * NBcap;
    const float inv = 1.f / sqrtf((float)DI);
    for (int b = w; b < nb; b += QSA_WARPS) {
        const bf16* kp = (b < nc ? pk + (size_t)b * DI : &fl[b - nc][0]) + lane * E;
        float kf[E];
#pragma unroll
        for (int e = 0; e < E; ++e) kf[e] = bf2f(kp[e]);
        float s = 0.f;
        for (int h = 0; h < Hi; ++h) {
            const bf16* qh = qs + h * DI + lane * E;
            float d = 0.f;
#pragma unroll
            for (int e = 0; e < E; ++e) d = fmaf(bf2f(qh[e]), kf[e], d);
            d = warp_sum(d);
            s += fmaxf(d, 0.f);
        }
        if (lane == 0) sc[b] = s * inv;
    }
    __syncthreads();
    // the threshold key, 8 bits a round from the top: the bin where the count from above reaches the picks still
    // wanted. After four rounds `prefix` is the k-th largest key and `need` the ties at it to take.
    unsigned prefix = 0u, mask = 0u;
    int need = ktop;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += QSA_THREADS) hist[i] = 0u;
        __syncthreads();
        for (int b = tid; b < nb; b += QSA_THREADS) {
            const unsigned u = qsa_key(sc[b]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int cum = 0, bin = 255;
            for (; bin > 0; --bin) {
                if (cum + (int)hist[bin] >= need) break;
                cum += (int)hist[bin];
            }
            s_prefix = prefix | ((unsigned)bin << shift);
            s_need = need - cum;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask |= 255u << shift;
    }
    // the picks in block order: above the threshold, or at it among its first `need`
    const unsigned below = (1u << lane) - 1u;
    int base = 0, eqbase = 0;
    for (int c0 = 0; c0 < nb; c0 += QSA_THREADS) {
        const int b = c0 + tid;
        const unsigned u = b < nb ? qsa_key(sc[b]) : 0u;
        const bool eq = b < nb && u == prefix;
        const unsigned be = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) wc_eq[w] = __popc(be);
        __syncthreads();
        int eqoff = eqbase, eqtot = 0;
        for (int i = 0; i < QSA_WARPS; ++i) {
            if (i < w) eqoff += wc_eq[i];
            eqtot += wc_eq[i];
        }
        const bool take = (b < nb && u > prefix) || (eq && eqoff + __popc(be & below) < need);
        const unsigned bt = __ballot_sync(0xffffffffu, take);
        if (lane == 0) wc_take[w] = __popc(bt);
        __syncthreads();
        int off = base, tot = 0;
        for (int i = 0; i < QSA_WARPS; ++i) {
            if (i < w) off += wc_take[i];
            tot += wc_take[i];
        }
        if (take) selt[off + __popc(bt & below)] = b;
        base += tot;
        eqbase += eqtot;
        __syncthreads();
    }
    if (tid == 0) nsel[t] = ktop;
}

// ---------------------------------------------------------------------------------------------------------
// btb_qsa_attn_split_d{128,256}: btb_attn_split's walk over a node's KEY LIST instead of its whole sequence - its
// picked blocks' positions in block order, then its partial tail, all ascending - list index j at position p(j),
// cache slot p < n0 ? p : n0 + anc[p - n0]. Key j to warp (j >> 5) & 7, split j / ATTN_SPLIT, the folds as
// btb_attn_split's: all by LIST index, so a node's bits are a function of its list and its keys' values alone - the
// T = 1 launch's over its committed path. The grid's splits are fixed by the budget, S = ceil((k_top + 1) r /
// ATTN_SPLIT) (a list holds at most k_top r + r - 1 keys), so the cost is the budget's, not the context's; a node
// under the budget lists every position it sees in order and walks exactly as btb_attn_split walks it. Padding rows
// (par -2) write nothing. Qwen4: 24 q heads, 2 kv heads, head_dim 256, scale 1/16, the output then through the gate
// (btb_sigmoid_mul / btb_gemv_sgate).
// ---------------------------------------------------------------------------------------------------------
#define QSA_K(D)                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) btb_qsa_pool_d##D(                                     \
        const bf16* __restrict__ raw, bf16* __restrict__ pk, int* pk_len, const int* __restrict__ n0p,       \
        const bf16* __restrict__ cos_t, const bf16* __restrict__ sin_t, int ts, int rot,                     \
        const bf16* __restrict__ kw, float eps, int r, int max_new) {                                        \
        qsa_pool<D>(raw, pk, pk_len, n0p, cos_t, sin_t, ts, rot, kw, eps, r, max_new);                       \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(QSA_THREADS) btb_qsa_select_d##D(                           \
        const bf16* __restrict__ qi, const bf16* __restrict__ pk, const bf16* __restrict__ raw,              \
        const int* __restrict__ n0p, const int* __restrict__ par, const bf16* __restrict__ cos_t,            \
        const bf16* __restrict__ sin_t, int ts, int rot, const bf16* __restrict__ kw, float eps, int T,      \
        int Hi, int r, int ktop, int NBcap, float* scores, int* __restrict__ sel, int* __restrict__ nsel) {  \
        qsa_select<D>(qi, pk, raw, n0p, par, cos_t, sin_t, ts, rot, kw, eps, T, Hi, r, ktop, NBcap, scores,  \
                      sel, nsel);                                                                            \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256) btb_qsa_attn_split_d##D(                               \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ n0p, const int* __restrict__ par,                    \
        const int* __restrict__ sel, const int* __restrict__ nsel, int r, int ktop, int T, int Hq, int Hk,   \
        int cap, float scale, float* __restrict__ part_m, float* __restrict__ part_l,                        \
        float* __restrict__ part_acc, int* __restrict__ cnt, int S) {                                        \
        attn_decode_split<D, WALK_QSA>(q, K, V, out, n0p, par, nullptr, sel, nsel, r, ktop, T, Hq, Hk, cap,  \
                                       scale, part_m, part_l, part_acc, cnt, S, 0);                          \
    }
QSA_K(128)
QSA_K(256)

// ---------------------------------------------------------------------------------------------------------
// btb_gemv_lane16_f32_m{1,..,32}: the host's gemv (native/src/gemv.rs, btb_gemv_bf16_rows: the host experts' matvec)
// to the bit, so an expert's output does not depend on which side of the bus it is seated. w [R, C] bf16 row-major,
// x [M, C] f32, y [M, R] f32. The host keeps 16 lane sums a (vector, row): lane j takes the columns c = j mod 16 in
// increasing order, one fused multiply-add each (its scalar mul_add, AVX2's and AVX-512's fmadd and NEON's fmla are
// one IEEE fma a lane; its 512-column tiles are whole groups of 16, so a column's lane is c mod 16 in every tier, the
// tail's too), then reduce16 folds them pairwise: (0+1)(2+3).., pairs of those, pairs of those, the two halves. Here
// a half-warp is a row: its lane j runs lane j's fma chain, and the butterfly over xor 1, 2, 4, 8 is that fold (each
// level adds the two partial sums reduce16 adds, and an IEEE add does not care which is on the left). (m, r)'s
// sequence knows nothing of M, R or the launch: row-invariant, and the host's bits.
// ---------------------------------------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void gemv_lane16_rows(const bf16* __restrict__ w, const float* __restrict__ x,
                                                 float* __restrict__ y, int R, int C) {
    const int j = threadIdx.x & 15;
    const int r = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    if (r < R) {  // a half-warp past R runs no columns but still takes part in its warp's shuffles
        const unsigned short* wr = reinterpret_cast<const unsigned short*>(w) + (size_t)r * C;
        int c = j;
        for (; c + 48 < C; c += 64) {  // four of the lane's columns in flight, taken in order
            float wv[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) wv[k] = __uint_as_float((unsigned)__ldcs(wr + c + 16 * k) << 16);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int m = 0; m < M; ++m) acc[m] = fmaf(wv[k], __ldg(x + (size_t)m * C + c + 16 * k), acc[m]);
            }
        }
        for (; c < C; c += 16) {
            const float wv = __uint_as_float((unsigned)__ldcs(wr + c) << 16);
#pragma unroll
            for (int m = 0; m < M; ++m) acc[m] = fmaf(wv, __ldg(x + (size_t)m * C + c), acc[m]);
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 4);
        a += __shfl_xor_sync(0xffffffffu, a, 8);
        acc[m] = a;
    }
    if (r < R && j == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = acc[m];
    }
}

#define GEMV_LANE16(M)                                                                                       \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_lane16_f32_m##M(                              \
        const bf16* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int R, int C) {      \
        gemv_lane16_rows<M>(w, x, y, R, C);                                                                  \
    }
GEMV_LANE16(1)
GEMV_LANE16(2)
GEMV_LANE16(4)
GEMV_LANE16(8)
GEMV_LANE16(16)
GEMV_LANE16(32)

// btb_gemv_lane16_mx4_f32_m{1,..,32}: the host's MXFP4 gemv (native/src/gemv.rs, the mx4 tasks) to the bit, so an
// MXFP4 expert seated on the card computes what it computes on the host. w as the checkpoint stores it: `blocks`
// [R, C/32, 16] (byte k of a block packs weights 2k, low nibble, and 2k+1, high), `scales` [R, C/32] e8m0; x [M, C]
// f32, y [M, R] f32. The host widens each weight to fp4 * 2^(s-127) - the code's exact value times mx_scale's exact
// power of two (2^-126 * 0.5 for s == 0), one IEEE multiply - then runs the bf16 gemv's accumulation over the widened
// tile: lane j the columns c = j mod 16 in order, one fma each, reduce16's pairwise fold. So here: lane j of a row's
// half-warp takes weight j then weight j + 16 of every block (bytes j/2 and 8 + j/2, nibble j & 1), widened as the
// host widens it, into gemv_lane16_rows's fma chain and butterfly.
__device__ __forceinline__ float mx4_scale(unsigned s) {
    const float f = __uint_as_float((s == 0u ? 1u : s) << 23);
    return s == 0u ? __fmul_rn(f, 0.5f) : f;
}

template <int M>
__device__ __forceinline__ void gemv_lane16_mx4_rows(const unsigned char* __restrict__ blocks,
                                                     const unsigned char* __restrict__ scales,
                                                     const float* __restrict__ x, float* __restrict__ y, int R, int C) {
    const int j = threadIdx.x & 15;
    const int r = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
    const int G = C >> 5;
    const int sh = (j & 1) << 2;
    float acc[M];
#pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.f;
    if (r < R) {  // a half-warp past R runs no columns but still takes part in its warp's shuffles
        const unsigned char* br = blocks + (size_t)r * G * 16;
        const unsigned char* sr = scales + (size_t)r * G;
        for (int g = 0; g < G; ++g) {
            const float sc = mx4_scale((unsigned)__ldcs(sr + g));
            const float w0 = __fmul_rn(BTB_FP4[(__ldcs(br + g * 16 + (j >> 1)) >> sh) & 15], sc);
            const float w1 = __fmul_rn(BTB_FP4[(__ldcs(br + g * 16 + 8 + (j >> 1)) >> sh) & 15], sc);
            const int c = g * 32 + j;
#pragma unroll
            for (int m = 0; m < M; ++m) {
                acc[m] = fmaf(w0, __ldg(x + (size_t)m * C + c), acc[m]);
                acc[m] = fmaf(w1, __ldg(x + (size_t)m * C + c + 16), acc[m]);
            }
        }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        a += __shfl_xor_sync(0xffffffffu, a, 1);
        a += __shfl_xor_sync(0xffffffffu, a, 2);
        a += __shfl_xor_sync(0xffffffffu, a, 4);
        a += __shfl_xor_sync(0xffffffffu, a, 8);
        acc[m] = a;
    }
    if (r < R && j == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) y[(size_t)m * R + r] = acc[m];
    }
}

#define GEMV_LANE16_MX4(M)                                                                                   \
    extern "C" __global__ void __launch_bounds__(128) btb_gemv_lane16_mx4_f32_m##M(                          \
        const unsigned char* __restrict__ blocks, const unsigned char* __restrict__ scales,                  \
        const float* __restrict__ x, float* __restrict__ y, int R, int C) {                                  \
        gemv_lane16_mx4_rows<M>(blocks, scales, x, y, R, C);                                                 \
    }
GEMV_LANE16_MX4(1)
GEMV_LANE16_MX4(2)
GEMV_LANE16_MX4(4)
GEMV_LANE16_MX4(8)
GEMV_LANE16_MX4(16)
GEMV_LANE16_MX4(32)

// ---------------------------------------------------------------------------------------------------------
// Qwen4's per-layer n-gram embedding (Qwen4ExpTextPLELayer) over a pass's nodes, as families/qwen4/verify.py
// `_ple_forward` steps it: two launches, the second reading rows the first wrote for the node's ancestors.
//
// btb_ple_gate, block (t, g): kv [T, ks] bf16 the merged [key_proj | value_proj] rows of the nodes' n-gram
// embeddings (stream g's key at g H, the shared value at G H), xq [T, G H] the node's streams through norm_query.
// The key is norm_key'd (Qwen4ExpTextRMSNorm over stream g, fp32, times 1 + w, rounded once), the gate is the sum of
// bf16(key * query) in fp32 rounded to bf16, times the fp32 1 / sqrt(H) (torch's bf16 division by a Python scalar),
// its signed square root through torch's bf16 ops (abs, clamp_min(1e-6), sqrt, times the sign), then
// gated = bf16(bf16(sigmoid(gate)) * value) and normed = norm_conv(gated) over the stream. The sums are the block's
// fixed-order tree over the one stream of the one row.
//
// btb_ple_conv, a thread a channel c of node t: the dilated depthwise conv along the node's path - tap k reads the
// normed row dil (K - 1 - k) steps back, the pass's own nodes first (`par`, -1 at a root; null a chain) and the kept
// window `pre` [C, Lp] (Lp = (K - 1) dil, oldest first) past them - each tap bf16(w * x) and the taps summed in
// index order in bf16, as the host's; then h[t, c] = bf16(h + bf16(gated + bf16(silu(acc)))) - the layer's
// `hidden_states + ple(...)`. `update` (a one-row step) shifts the node's normed row into `pre` in place: a thread
// reads its channel's window before it writes it, and no other thread touches the channel. A padding row (par < -1)
// is skipped by both. Nothing crosses nodes but the ancestors' normed rows, which are the rows their own steps make:
// a node computes bit for bit as the one-token step of its path.
// ---------------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) btb_ple_gate(const bf16* __restrict__ kv, int ks,
                                                               const bf16* __restrict__ xq,
                                                               const bf16* __restrict__ wk,
                                                               const bf16* __restrict__ wc, float eps, float inv_h,
                                                               const int* __restrict__ par, bf16* __restrict__ gated,
                                                               bf16* __restrict__ normed, int H, int G) {
    const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    if (par != nullptr && par[t] < -1) return;
    __shared__ float red[256];
    const bf16* kr = kv + (size_t)t * ks + (size_t)g * H;
    const bf16* vr = kv + (size_t)t * ks + (size_t)G * H;
    const size_t base = ((size_t)t * G + g) * H;
    const bf16* qr = xq + base;
    const bf16* wkg = wk + (size_t)g * H;
    const bf16* wcg = wc + (size_t)g * H;
    float ss = 0.f;
    for (int i = tid; i < H; i += 256) {
        const float v = bf2f(kr[i]);
        ss = fmaf(v, v, ss);
    }
    const float rk = rsqrtf(block_sum256(ss, red) / (float)H + eps);
    float d = 0.f;
    for (int i = tid; i < H; i += 256) {
        const float kn = bfround(bf2f(kr[i]) * rk * (1.f + bf2f(wkg[i])));
        d += bfround(kn * bf2f(qr[i]));
    }
    float gate = bfround(block_sum256(d, red));
    gate = bfround(gate * inv_h);
    const float mag = bfround(sqrtf(bfround(fmaxf(fabsf(gate), 1e-6f))));
    gate = gate > 0.f ? mag : (gate < 0.f ? -mag : 0.f * mag);
    const float sig = bfround(act_sigmoid(gate));
    float s2 = 0.f;
    for (int i = tid; i < H; i += 256) {
        const float gv = bfround(sig * bf2f(vr[i]));
        gated[base + i] = f2bf(gv);
        s2 = fmaf(gv, gv, s2);
    }
    const float rc = rsqrtf(block_sum256(s2, red) / (float)H + eps);
    for (int i = tid; i < H; i += 256) {
        const float gv = bf2f(gated[base + i]);  // this thread's own write above
        normed[base + i] = f2bf(gv * rc * (1.f + bf2f(wcg[i])));
    }
}

#define PLE_MAX_K 8
extern "C" __global__ void __launch_bounds__(256) btb_ple_conv(const bf16* __restrict__ gated,
                                                               const bf16* __restrict__ normed,
                                                               const bf16* __restrict__ w, bf16* pre,
                                                               const int* __restrict__ par, bf16* __restrict__ h, int C,
                                                               int K, int dil, int update) {
    const int t = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x;
    if (par != nullptr && par[t] < -1) return;
    __shared__ int path[32];
    __shared__ int plen;
    if (threadIdx.x == 0) {
        int len = 0;
        for (int p = t; p >= 0 && len < 32; p = par != nullptr ? par[p] : p - 1) path[len++] = p;
        plen = len;
    }
    __syncthreads();
    if (c >= C) return;
    const int Lp = (K - 1) * dil;
    bf16* pr = pre + (size_t)c * Lp;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int s = dil * (K - 1 - k);  // steps back along the node's path
        const float x = s < plen ? bf2f(normed[(size_t)path[s] * C + c]) : bf2f(pr[Lp - (s - plen) - 1]);
        const float term = bfround(bf2f(w[(size_t)c * K + k]) * x);
        acc = k == 0 ? term : bfround(acc + term);
    }
    const size_t o = (size_t)t * C + c;
    const float out = bfround(bf2f(gated[o]) + bfround(act_silu(acc)));
    h[o] = f2bf(bf2f(h[o]) + out);
    if (update) {
        for (int j = 0; j + 1 < Lp; ++j) pr[j] = pr[j + 1];
        if (Lp > 0) pr[Lp - 1] = normed[o];
    }
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
