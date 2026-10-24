# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Aluminum, btb's attention on MLX at head sizes 64, 96, 128 and 256: one arithmetic for every row however it is
reached (a prefill chunk or its rows in other chunks, a decode step, a tree's node, a batch's row, a paged or int8
cache's), so a row's bits are a function of its query and keys alone. A row's keys fall in segments of SEGK and
blocks of BK from key 0; a segment runs MLX's steel block loop from scratch (S and P.V on the matrix unit) and the
segments fold in order. The prefill tiles rows over the matrix unit; decode runs a row's segments side by side and
folds them."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from .core import metal_kernel, mx
from .gemv import _gemv_lock

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

HEADS = (64, 96, 128, 256)
MAXG = 8
SEGK = 512


def block(D: int) -> int:
    """keys a block at head size D: 32 below 128, 16 from 128 (MLX's steel tiling)"""
    return 32 if D < 128 else 16


def eager(D: int) -> bool:
    """the max moved every block from a finite floor (MLX's) below head 128, lazily by TAU from -inf from 128: each
    measured the faster at its head size, the arithmetic per head size fixed by it"""
    return D < 128


def bf16p(D: int) -> bool:
    """p rounded to bf16 for P.V at head 256 (the precision its prefill kernel had), float below it (MLX's)"""
    return D == 256


_HEADER = r"""
#define SEGK 512
#define TAU 8.0f
// KQ: an int8 row's element as the bf16 its dequantized row holds (the row's scale `sc`), else the bf16 element
#if KQ
#define BTB_KV(p, e, sc) ((bfloat)((float)(p)[e] * (sc)))
#else
#define BTB_KV(p, e, sc) ((p)[e])
#endif
// 8 bf16 values as a uint4 of their bits (by value: a thread array read through a uint4 pointer is not defined)
inline uint4 btb_pack8(thread const bfloat* x) {
    return uint4(as_type<uint>(bfloat2(x[0], x[1])), as_type<uint>(bfloat2(x[2], x[3])),
                 as_type<uint>(bfloat2(x[4], x[5])), as_type<uint>(bfloat2(x[6], x[7])));
}
// PB: p rounded to bf16 (its sum and its product with V the rounded values')
#if PB
#define BTB_P(x) ((float)(bfloat)(x))
#else
#define BTB_P(x) (x)
#endif
// EAGER: the max moved every block from a finite floor (a first block's factor 1, a masked key's p 0, no row dead);
// else lazily by TAU from -inf
#define MINIT (EAGER ? -FLT_MAX : -INFINITY)
#define LIVE(m) (EAGER || (m) != -INFINITY)
inline uint btb_fm(uint lane) { return ((lane >> 1) & 3) + 4 * (lane >> 4); }
inline uint btb_fn(uint lane) { return 2 * (lane & 1) + 4 * ((lane >> 3) & 1); }

// a segment's (m, l, o) folded into the running (M, L, O); `first` the row's first element (a and b its row's)
inline void btb_fold(thread float& M, thread float& L, thread float& O, float m, float l, float o, thread float& a,
                     thread float& b, bool first) {
    if (first) {
        float mn = max(M, m);
        a = fast::exp2(M - mn);
        b = fast::exp2(m - mn);
        L = fma(L, a, l * b);
        M = mn;
    }
    O = fma(O, a, o * b);
}
"""

# heads below 256 on MLX's steel tiling: a threadgroup (32 rows, query head, chunk z) of 4 simdgroups of 8 rows; K
# (transposed) then V staged a block at a time in one buffer. Chunk z's meta[z] (T, past, window, kL, pbase, qoff,
# kbatch): its row t at past + t over keys [lo, past + t], kL its last key + 1, key j at row pbase + j of slice kbatch;
# its rows q [qoff + t] and out [qoff + t] of [*, Hq, D], q in its own dtype, k / v [B, Hk, cap, D] bf16 (KQ: int8, a
# scale a row in ks / vs [B, Hk, cap]; MAP: the rows before past through rmap, the chunk's own in k2 / v2)
_PREFILL = r"""
    constexpr int BQ = 32, BK = BK_V, BD = D_V, TK = BK / 8, TD = BD / 8, NT = 128;
    constexpr int LDQ = BD + 8, LDK = BK + 8, LDV = BD + 8;
    constexpr int KVSZ = BD * LDK > BK * LDV ? BD * LDK : BK * LDV;
    threadgroup bfloat Qs[BQ * LDQ];
    threadgroup bfloat KVs[KVSZ];
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = sg * 32 + lane;
    uint qb = threadgroup_position_in_grid.x, h = threadgroup_position_in_grid.y;
    auto mz = meta + threadgroup_position_in_grid.z * 7;
    int T = mz[0], past = mz[1], window = mz[2], kL = mz[3], pbase = mz[4], qoff = mz[5], kbatch = mz[6];
    if ((int)qb * BQ >= T) return;
    int Hq = q_shape[1], Hk = k_shape[1], cap = k_shape[2];
    int hk = h / (Hq / Hk);
    size_t base = ((size_t)kbatch * Hk + hk) * cap + pbase;
    auto kh = k + base * BD;
    auto vh = v + base * BD;
#if MAP
    auto k2h = k2 + (size_t)hk * k2_shape[2] * BD;
    auto v2h = v2 + (size_t)hk * k2_shape[2] * BD;
#define KROW(j) ((j) < past ? kh + (size_t)rmap[j] * BD : k2h + (size_t)((j) - past) * BD)
#define VROW(j) ((j) < past ? vh + (size_t)rmap[j] * BD : v2h + (size_t)((j) - past) * BD)
#else
#define KROW(j) (kh + (size_t)(j) * BD)
#define VROW(j) (vh + (size_t)(j) * BD)
#endif
#if KQ
#define KSC(j) ks[base + (j)]
#define VSC(j) vs[base + (j)]
#else
#define KSC(j) 1.0f
#define VSC(j) 1.0f
#endif
    float scale = sc[0] * M_LOG2E_F;
    {
        constexpr int QR = BQ * BD / NT, QC = BD / QR;
        int bi = tid / QC, bj = QR * (tid % QC), t = qb * BQ + bi;
        auto src = q + ((size_t)(qoff + min(t, T - 1)) * Hq + h) * BD + bj;
        #pragma clang loop unroll(full)
        for (int j = 0; j < QR; ++j) Qs[bi * LDQ + bj + j] = t < T ? (bfloat)src[j] : (bfloat)0.0f;
    }
    uint fm = btb_fm(lane), fn = btb_fn(lane);
    const int tm = 8 * sg;
    int rpos = past + qb * BQ + tm + fm;
    int hi = min(rpos + 1, kL), lo = window ? max(0, rpos + 1 - window) : 0;
    float M = MINIT, sum = 0.0f;
#if SINKS
    float Mt = sinks[h] * M_LOG2E_F, Lt = 1.0f;
#else
    float Mt = -INFINITY, Lt = 0.0f;
#endif
    // the folded state, read only once it holds a segment
    simdgroup_float8x8 O[TD], Ot[TD];
    bool have = false;
#define OTR(c, e) (have ? Ot[c].thread_elements()[e] : 0.0f)
    #pragma clang loop unroll(full)
    for (int c = 0; c < TD; ++c) O[c] = simdgroup_float8x8(0.0f);
    int p_first = past + qb * BQ, p_last = min(past + (int)(qb + 1) * BQ, past + T) - 1;
    int lo_max = window ? max(0, p_last + 1 - window) : 0;
    int kb0 = window ? max(0, p_first + 1 - window) / BK : 0, kb1 = (p_last + BK) / BK;
    int kb_causal = p_first / BK, nk_full = kL / BK;
    for (int kb = kb0; kb < kb1; ++kb) {
        if (kb > kb0 && kb * BK % SEGK == 0) {
            // the segment before folded into the running state, a fresh one begun
            float a = 0.0f, b = 0.0f;
            #pragma clang loop unroll(full)
            for (int c = 0; c < TD; ++c) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    if (LIVE(M)) {
                        float Ox = OTR(c, e);
                        btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], a, b, c == 0 && e == 0);
                        Ot[c].thread_elements()[e] = Ox;
                    }
                }
                O[c] = simdgroup_float8x8(0.0f);
            }
            have = have || LIVE(M);
            M = MINIT;
            sum = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            constexpr int KR = BK * BD / NT, KC = BD / KR;
            int bi = tid / KC, bj = KR * (tid % KC), j = kb * BK + bi;
            // the row a branch, not a select an element (selected element by element: 16% of the prefill, measured)
            if (j < kL) {
                auto src = KROW(j) + bj;
                float ksc = KSC(j);
                (void)ksc;
                #pragma clang loop unroll(full)
                for (int e = 0; e < KR; ++e) KVs[(bj + e) * LDK + bi] = BTB_KV(src, e, ksc);
            } else {
                #pragma clang loop unroll(full)
                for (int e = 0; e < KR; ++e) KVs[(bj + e) * LDK + bi] = (bfloat)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 S[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) S[f] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (int dd = 0; dd < TD; ++dd) {
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_bfloat8x8 Qf, Kf[TK];
            simdgroup_load(Qf, Qs + tm * LDQ + dd * 8, LDQ);
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) simdgroup_load(Kf[f], KVs + dd * 8 * LDK + f * 8, LDK);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) simdgroup_multiply_accumulate(S[f], Qf, Kf[f], S[f]);
        }
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            S[f].thread_elements()[0] *= scale;
            S[f].thread_elements()[1] *= scale;
        }
        if (kb >= kb_causal || kb * BK < lo_max || kb >= nk_full) {
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    int j = kb * BK + f * 8 + fn + e;
                    if (j < lo || j >= hi) S[f].thread_elements()[e] = -INFINITY;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            constexpr int VR = BK * BD / NT, VC = BD / VR;
            int bi = tid / VC, bj = VR * (tid % VC), j = kb * BK + bi;
            if (j < kL) {
                auto src = VROW(j) + bj;
                float vsc = VSC(j);
                (void)vsc;
                #pragma clang loop unroll(full)
                for (int e = 0; e < VR; ++e) KVs[bi * LDV + bj + e] = BTB_KV(src, e, vsc);
            } else {
                #pragma clang loop unroll(full)
                for (int e = 0; e < VR; ++e) KVs[bi * LDV + bj + e] = (bfloat)0.0f;
            }
        }
        // the block's max for the lane's row: its own values, then the row's four lanes (a max in any order)
        float bm = -INFINITY;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) bm = max(bm, max(S[f].thread_elements()[0], S[f].thread_elements()[1]));
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        bool up = EAGER ? true : bm > M + TAU;
        float nm = EAGER ? max(M, bm) : up ? bm : M;
        bool dead = EAGER ? false : nm == -INFINITY;
        float factor = up ? fast::exp2(M - nm) : 1.0f;
        M = nm;
        // the block's sum for the lane's row: its own pairs in slice order, then the row's four lanes (each pair of
        // lanes commutes, so all four hold the same bits)
        float rs = 0.0f;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            float p0 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[0] - nm));
            float p1 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[1] - nm));
            S[f].thread_elements()[0] = p0;
            S[f].thread_elements()[1] = p1;
            rs = f == 0 ? p0 + p1 : rs + (p0 + p1);
        }
        rs = rs + simd_shuffle_xor(rs, 1);
        rs = rs + simd_shuffle_xor(rs, 8);
        sum = fma(sum, factor, rs);
        if (EAGER || !simd_all(factor == 1.0f)) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TD; ++c) {
                O[c].thread_elements()[0] *= factor;
                O[c].thread_elements()[1] *= factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // P.V in float (MLX's): V's fragments widened lane by lane
        #pragma clang loop unroll(full)
        for (int c = 0; c < TD; ++c) {
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                if (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                simdgroup_float8x8 Vf;
                Vf.thread_elements()[0] = (float)KVs[(f * 8 + fm) * LDV + c * 8 + fn];
                Vf.thread_elements()[1] = (float)KVs[(f * 8 + fm) * LDV + c * 8 + fn + 1];
                if (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(O[c], S[f], Vf, O[c]);
            }
        }
    }
    int t = qb * BQ + tm + fm;
    if (t >= T) return;
    device OT* orow = out + ((size_t)(qoff + t) * Hq + h) * BD;
    float a = 0.0f, b = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < TD; ++c) {
        #pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            float Ox = OTR(c, e);
            if (LIVE(M)) btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], a, b, c == 0 && e == 0);
            orow[c * 8 + fn + e] = (OT)precise::divide(Ox, Lt);
        }
    }
"""

# head 256: a threadgroup (kv head, 32 flat rows, chunk z; row R position R / G, query head hk * G + R % G), K and V
# staged row-major together, simdgroup pairs (pair p rows 8p.., half hv dims 128 hv..), S's halves traded through xs and
# added low first, the folded state in registers (measured 1-2% faster than a device scratch). As _PREFILL otherwise
_PREFILL256 = r"""
    constexpr int BQ = 32, BK = 16, BD = 256, HD = 128, TK = 2, TH = HD / 8, G = G_V;
    constexpr int LD = BD + 8;
    threadgroup bfloat Ks[BK * LD];
    threadgroup bfloat Vs[BK * LD];
    threadgroup float xs[8][TK * 64];
    uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup, tid = sg * 32 + lane;
    uint pr = sg >> 1, hv = sg & 1;
    uint qb = threadgroup_position_in_grid.x, hk = threadgroup_position_in_grid.y, z = threadgroup_position_in_grid.z;
    auto mz = meta + z * 7;
    int T = mz[0], past = mz[1], window = mz[2], kL = mz[3], pbase = mz[4], qoff = mz[5], kbatch = mz[6];
    if ((int)qb * BQ >= T * G) return;
    int Hq = q_shape[1], Hk = k_shape[1], cap = k_shape[2];
    size_t base = ((size_t)kbatch * Hk + hk) * cap + pbase;
    auto kh = k + base * BD;
    auto vh = v + base * BD;
#if MAP
    auto k2h = k2 + (size_t)hk * k2_shape[2] * BD;
    auto v2h = v2 + (size_t)hk * k2_shape[2] * BD;
#define KROW(j) ((j) < past ? kh + (size_t)rmap[j] * BD : k2h + (size_t)((j) - past) * BD)
#define VROW(j) ((j) < past ? vh + (size_t)rmap[j] * BD : v2h + (size_t)((j) - past) * BD)
#else
#define KROW(j) (kh + (size_t)(j) * BD)
#define VROW(j) (vh + (size_t)(j) * BD)
#endif
    float scale = sc[0] * M_LOG2E_F;
    uint fm = btb_fm(lane), fn = btb_fn(lane);
    int R0 = qb * BQ, R = R0 + 8 * pr + fm;
    int t = R / G, hq = hk * G + R % G;
    bool real = t < T;
    // q's fragments read a block at a time, a row past the chunk as its last row (computed, never stored): held in
    // registers, or selected, they cost the loop 5-10% (measured)
    auto qr = q + ((size_t)(qoff + min(t, T - 1)) * Hq + hq) * BD + hv * HD + fn;
    int rpos = past + t;
    int hi = min(rpos + 1, kL), lo = window ? max(0, rpos + 1 - window) : 0;
    float M = MINIT, sum = 0.0f;
#if SINKS
    float Mt = sinks[hq] * M_LOG2E_F, Lt = 1.0f;
#else
    float Mt = -INFINITY, Lt = 0.0f;
#endif
    simdgroup_float8x8 O[TH];
    bool have = false;
    simdgroup_float8x8 Ot[TH];
#define OTR(c, e) (have ? Ot[c].thread_elements()[e] : 0.0f)
    #pragma clang loop unroll(full)
    for (int c = 0; c < TH; ++c) O[c] = simdgroup_float8x8(0.0f);
    int t_first = R0 / G, t_last = min((R0 + BQ - 1) / G, T - 1);
    int p_first = past + t_first, p_last = past + t_last;
    int lo_max = window ? max(0, p_last + 1 - window) : 0;
    int kb0 = window ? max(0, p_first + 1 - window) / BK : 0, kb1 = (p_last + BK) / BK;
    int kb_causal = p_first / BK, nk_full = kL / BK;
    for (int kb = kb0; kb < kb1; ++kb) {
        if (kb > kb0 && kb * BK % SEGK == 0) {
            float a = 0.0f, b = 0.0f;
            #pragma clang loop unroll(full)
            for (int c = 0; c < TH; ++c) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    if (LIVE(M)) {
                        float Ox = OTR(c, e);
                        btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], a, b, c == 0 && e == 0);
                        Ot[c].thread_elements()[e] = Ox;
                    }
                }
                O[c] = simdgroup_float8x8(0.0f);
            }
            have = have || LIVE(M);
            M = MINIT;
            sum = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            int vi = tid / 16, vj = 16 * (tid % 16), j = kb * BK + vi;
            uint4 k0 = uint4(0u), k1 = uint4(0u), v0 = uint4(0u), v1 = uint4(0u);
            // the row a branch, not a select each (as below 256)
            if (j < kL) {
#if KQ
                // an int8 row's 16 elements dequantized to bf16 as they are staged
                auto kr = KROW(j) + vj;
                auto vr = VROW(j) + vj;
                float ksc = ks[base + j], vsc = vs[base + j];
                bfloat xk[16], xv[16];
                #pragma clang loop unroll(full)
                for (int e = 0; e < 16; ++e) {
                    xk[e] = BTB_KV(kr, e, ksc);
                    xv[e] = BTB_KV(vr, e, vsc);
                }
                k0 = btb_pack8(xk);
                k1 = btb_pack8(xk + 8);
                v0 = btb_pack8(xv);
                v1 = btb_pack8(xv + 8);
#else
                auto kr = (const device uint4*)(KROW(j) + vj);
                auto vr = (const device uint4*)(VROW(j) + vj);
                k0 = kr[0];
                k1 = kr[1];
                v0 = vr[0];
                v1 = vr[1];
#endif
            }
            *(threadgroup uint4*)(Ks + vi * LD + vj) = k0;
            *(threadgroup uint4*)(Ks + vi * LD + vj + 8) = k1;
            *(threadgroup uint4*)(Vs + vi * LD + vj) = v0;
            *(threadgroup uint4*)(Vs + vi * LD + vj + 8) = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 S[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) S[f] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (int dd = 0; dd < TH; ++dd) {
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                simdgroup_bfloat8x8 Kb;
                simdgroup_load(Kb, Ks + f * 8 * LD + hv * HD + dd * 8, LD, ulong2(0, 0), true);
                simdgroup_bfloat8x8 Qf;
                Qf.thread_elements()[0] = (bfloat)qr[dd * 8];
                Qf.thread_elements()[1] = (bfloat)qr[dd * 8 + 1];
                simdgroup_multiply_accumulate(S[f], Qf, Kb, S[f]);
            }
        }
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) simdgroup_store(S[f], &xs[sg][f * 64], 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bool need = kb >= kb_causal || kb * BK < lo_max || kb >= nk_full;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                float other = xs[sg ^ 1][f * 64 + fm * 8 + fn + e];
                float mine = S[f].thread_elements()[e];
                float s = (hv == 0 ? mine + other : other + mine) * scale;
                int j = kb * BK + f * 8 + fn + e;
                S[f].thread_elements()[e] = (need && (j < lo || j >= hi)) ? -INFINITY : s;
            }
        }
        float bm = -INFINITY;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) bm = max(bm, max(S[f].thread_elements()[0], S[f].thread_elements()[1]));
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        bool up = EAGER ? true : bm > M + TAU;
        float nm = EAGER ? max(M, bm) : up ? bm : M;
        bool dead = EAGER ? false : nm == -INFINITY;
        float factor = up ? fast::exp2(M - nm) : 1.0f;
        M = nm;
        float rs = 0.0f;
        simdgroup_bfloat8x8 P[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            float p0 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[0] - nm));
            float p1 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[1] - nm));
            P[f].thread_elements()[0] = (bfloat)p0;
            P[f].thread_elements()[1] = (bfloat)p1;
            rs = f == 0 ? p0 + p1 : rs + (p0 + p1);
        }
        rs = rs + simd_shuffle_xor(rs, 1);
        rs = rs + simd_shuffle_xor(rs, 8);
        sum = fma(sum, factor, rs);
        if (EAGER || !simd_all(factor == 1.0f)) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TH; ++c) {
                O[c].thread_elements()[0] *= factor;
                O[c].thread_elements()[1] *= factor;
            }
        }
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TH; ++c) {
                simdgroup_bfloat8x8 Vb;
                simdgroup_load(Vb, Vs + f * 8 * LD + hv * HD + c * 8, LD);
                simdgroup_multiply_accumulate(O[c], P[f], Vb, O[c]);
            }
        }
    }
    if (!real) return;
    device OT* orow = out + ((size_t)(qoff + t) * Hq + hq) * BD + hv * HD;
    float a = 0.0f, b = 0.0f;
    #pragma clang loop unroll(full)
    for (int c = 0; c < TH; ++c) {
        #pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            float Ox = OTR(c, e);
            if (LIVE(M)) btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], a, b, c == 0 && e == 0);
            orow[c * 8 + fn + e] = (OT)precise::divide(Ox, Lt);
        }
    }
"""

# a key's score for each of the threadgroup's GH query heads: an fma chain over d from 0 (head 256: the halves' chains,
# low first, added), scaled. `kq` an int8 row and its scale (the value the dequantized bf16 row holds), else bf16
_SCORE = r"""
template <int BD, int GH, typename KP>
inline void btb_scores(KP kr, float ksc, threadgroup const float (*qs)[BD], float scale, thread float* s) {
    float a[GH], b[GH];
    for (int g = 0; g < GH; ++g) { a[g] = 0.0f; b[g] = 0.0f; }
    constexpr int HALF = BD == 256 ? 128 : BD;
    for (int d0 = 0; d0 < BD; d0 += 8) {
        float kv[8];
        for (int e = 0; e < 8; ++e) kv[e] = (float)BTB_KV(kr, d0 + e, ksc);
        for (int g = 0; g < GH; ++g)
            for (int e = 0; e < 8; ++e) {
                if (d0 < HALF) a[g] = fma(qs[g][d0 + e], kv[e], a[g]);
                else b[g] = fma(qs[g][d0 + e], kv[e], b[g]);
            }
    }
    for (int g = 0; g < GH; ++g) s[g] = (BD == 256 ? a[g] + b[g] : a[g]) * scale;
}
"""

# decode, as device functions its own launch and the megakernel's attention op both call: a threadgroup a (kv head and
# group of GH query heads, segment of the node's, node t) of NTH threads. A node's logical keys [start, n) (meta[t] =
# (n, segments, past, start, batch, seg, pbase, kbatch)): key i < past cache row i, else past + path[t][i - past]; a row
# at or past `seg` from k2 / v2's slice `batch`, else through rmap (MAP) or from row pbase of slice kbatch. The
# segment's scores (a thread a key), its block chain spread over every thread in the prefill's order (the lazy walk and
# the sums' fma chain alone serial), its P.V the prefill's own on the matrix unit (the heads a fragment's rows) over V
# staged CH keys at a time; a node of one segment writes out (its fold of one), else its partial (m, l, o) for
# btb_alu_fold. The inputs reach it through `BtbAluIn`, the pointers MLX may bind in the constant address space (a
# small array) typed by template parameters; its threadgroup memory is the caller's, `tgm_layout`'s
_SEGMENT_FN = r"""
template <typename QT, typename KT, typename MT, typename PT, typename ST, typename ST2, typename RT, typename SKT>
struct BtbAluIn {
    QT q;
    KT k, v, k2, v2;
    MT meta;
    PT path;
    ST ks, vs;
    ST2 ks2, vs2;
    RT rmap;
    SKT sinks;
    float sc;
    int Hq, Hk, cap, PD, cap2, NSX;
};

template <int BD, int G, int GH, int NTH, int CH, int VT, typename A, typename OT>
inline void btb_alu_segment(thread const A& a, uint gx, uint s, uint t, uint tid, threadgroup uchar* tgm,
                            device OT* out, device float* pm, device float* pl, device float* po) {
    constexpr int BK = BTB_BK, NB = SEGK / BK, NHG = G / GH, LD = BD + 8, CW = BD / 8;
    constexpr int NSG = NTH / 32, TDS = (CW + NSG - 1) / NSG;
    uint lane = tid % 32, sg = tid / 32;
    uint hk = gx / NHG, hg = gx % NHG;
    int Hq = a.Hq, Hk = a.Hk, cap = a.cap, PD = a.PD, NSX = a.NSX;
    auto mt = a.meta + (size_t)t * 8;
    int n = mt[0], ns = mt[1], past = mt[2], start = mt[3], batch = mt[4], segr = mt[5], pbase = mt[6], kbatch = mt[7];
    (void)batch;
    (void)segr;
    if ((int)s >= ns) return;
    int seg = start / SEGK + (int)s;
    // the segment's blocks holding a live key of the node: [jb, je) keys
    int jb = max(seg * SEGK, start / BK * BK), je = min((seg + 1) * SEGK, (n + BK - 1) / BK * BK);
    int nk = je - jb, nb = nk / BK;
    int h0 = hk * G + hg * GH;
    threadgroup float (*qs)[BD] = (threadgroup float (*)[BD])tgm;
    threadgroup float (*pt)[SEGK] = (threadgroup float (*)[SEGK])(tgm + 4 * GH * BD);
    threadgroup float (*bm)[NB] = (threadgroup float (*)[NB])(tgm + 4 * GH * (BD + SEGK));
    threadgroup float (*mb)[NB] = bm + GH;
    threadgroup float (*ft)[NB] = mb + GH;
    threadgroup float (*rsum)[NB] = ft + GH;
    threadgroup float* ms = (threadgroup float*)(rsum + GH);
    threadgroup float* ls = ms + GH;
    threadgroup bfloat* vt = (threadgroup bfloat*)(tgm + VT);
    for (int e = tid; e < GH * BD; e += NTH) qs[e / BD][e % BD] = (float)(bfloat)a.q[((size_t)t * Hq + h0) * BD + e];
    // logical key i's row offset (in rows of D) and whether it lies in k2 / v2
    size_t b1 = ((size_t)kbatch * Hk + hk) * cap + pbase;
#if SEG
    size_t b2 = ((size_t)batch * Hk + hk) * a.cap2;
#endif
#define LROW(i) ((i) < past ? (i) : past + (int)a.path[(size_t)t * PD + ((i) - past)])
#if SEG && MAP
#define ROFF(r) ((r) >= segr ? b2 + (size_t)((r) - segr) : (size_t)hk * cap + a.rmap[r])
#elif SEG
#define ROFF(r) ((r) >= segr ? b2 + (size_t)((r) - segr) : b1 + (size_t)(r))
#elif MAP
#define ROFF(r) ((size_t)hk * cap + a.rmap[r])
#else
#define ROFF(r) (b1 + (size_t)(r))
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = a.sc * M_LOG2E_F;
    for (int i = tid; i < nk; i += NTH) {
        int j = jb + i;
        float sv[GH];
        bool live = j >= start && j < n;
        if (live) {
            int r = LROW(j);
            size_t off = ROFF(r);
#if SEG
            bool own = r >= segr;
            auto kr = (own ? a.k2 : a.k) + off * BD;
#if KQ
            float ksc = own ? a.ks2[off] : a.ks[off];
#endif
#else
            auto kr = a.k + off * BD;
#if KQ
            float ksc = a.ks[off];
#endif
#endif
#if !KQ
            float ksc = 1.0f;
#endif
            btb_scores<BD, GH>(kr, ksc, qs, scale, sv);
        }
        for (int g = 0; g < GH; ++g) pt[g][i] = live ? sv[g] : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the blocks' maxima: a lane a (head, key), a block's lanes reduced together (a max in any order)
    for (int e0 = tid - lane; e0 < GH * nk; e0 += NTH) {
        int e = e0 + lane, g = min(e / nk, GH - 1), i = e - g * nk;
        bool ok = e < GH * nk;
        float x = ok ? pt[g][i] : -INFINITY;
        for (int o = 1; o < BK; o <<= 1) x = max(x, simd_shuffle_xor(x, o));
        if (ok && i % BK == 0) bm[g][i / BK] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the row's max block by block as the prefill moves it: a simdgroup a head, lane b block b's
    for (int g = sg; g < GH; g += NTH / 32) {
        float carry = MINIT, prev = MINIT, nmb = MINIT;
        bool upb = false;
        for (int i = 0; i < nb; ++i) {
            float x = bm[g][i];
            bool up = EAGER ? true : x > carry + TAU;
            float nm = EAGER ? max(carry, x) : up ? x : carry;
            if ((int)lane == i) {
                prev = carry;
                upb = up;
                nmb = nm;
            }
            carry = nm;
        }
        if ((int)lane < nb) {
            mb[g][lane] = nmb;
            ft[g][lane] = upb ? fast::exp2(prev - nmb) : 1.0f;
        }
        if (lane == 0) ms[g] = carry;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // p, and each block's sum in the prefill's order: the lane of keys (e, e + 1) of each slice of 8 adds the pair, the
    // slices in order, then lanes e and e + 2, then the two of those
    for (int e0 = tid - lane; e0 < GH * nk; e0 += NTH) {
        int e = e0 + lane, g = min(e / nk, GH - 1), i = e - g * nk, b = i / BK;
        bool ok = e < GH * nk;
        float m_ = ok ? mb[g][b] : -INFINITY;
        float p = ok && LIVE(m_) ? BTB_P(fast::exp2(pt[g][i] - m_)) : 0.0f;
        if (ok) pt[g][i] = p;
        float x = p + simd_shuffle_xor(p, 1);
        float s_ = x + simd_shuffle_xor(x, 8);
        if (BK == 32) {
            s_ = s_ + simd_shuffle_xor(x, 16);
            s_ = s_ + simd_shuffle_xor(x, 24);
        }
        float r = s_ + simd_shuffle_xor(s_, 2);
        r = r + simd_shuffle_xor(r, 4);
        if (ok && i % BK == 0) rsum[g][b] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((int)tid < GH) {
        float sum = 0.0f;
        for (int b = 0; b < nb; ++b) sum = fma(sum, ft[tid][b], rsum[tid][b]);
        ls[tid] = sum;
    }
    // O on the matrix unit, the prefill's P.V: the heads a fragment's rows (those past GH zero), 8 columns a
    // fragment, column fragment c on simdgroup c % NSG, V staged CH keys at a time (a key past the node's zero)
    uint fm = btb_fm(lane), fn = btb_fn(lane);
    bool rowok = (int)fm < GH;
    int gr = rowok ? (int)fm : 0;
    simdgroup_float8x8 acc[TDS];
    #pragma clang loop unroll(full)
    for (int u = 0; u < TDS; ++u) acc[u] = simdgroup_float8x8(0.0f);
    for (int k0 = 0; k0 < nk; k0 += CH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = tid; e < CH * CW; e += NTH) {
            int r_ = e / CW, cc = (e % CW) * 8, i = jb + k0 + r_;
            uint4 w = uint4(0u);
            if (k0 + r_ < nk && i < n) {
                int r = LROW(i);
                size_t off = ROFF(r);
#if SEG
                bool own = r >= segr;
                auto vr = (own ? a.v2 : a.v) + off * BD + cc;
#else
                auto vr = a.v + off * BD + cc;
#endif
#if KQ
#if SEG
                float vsc = own ? a.vs2[off] : a.vs[off];
#else
                float vsc = a.vs[off];
#endif
                bfloat x[8];
                for (int u = 0; u < 8; ++u) x[u] = BTB_KV(vr, u, vsc);
                w = btb_pack8(x);
#else
                w = *(const device uint4*)vr;
#endif
            }
            *(threadgroup uint4*)(vt + r_ * LD + cc) = w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int r = 0; r < CH && k0 + r < nk; r += BK) {
            float fa = rowok ? ft[gr][(k0 + r) / BK] : 1.0f;
            #pragma clang loop unroll(full)
            for (int u = 0; u < TDS; ++u) {
                acc[u].thread_elements()[0] *= fa;
                acc[u].thread_elements()[1] *= fa;
            }
            #pragma clang loop unroll(full)
            for (int f = 0; f < BK / 8; ++f) {
                int kk = k0 + r + f * 8 + (int)fn;
#if PB
                simdgroup_bfloat8x8 P;
                P.thread_elements()[0] = rowok ? (bfloat)pt[gr][kk] : (bfloat)0.0f;
                P.thread_elements()[1] = rowok ? (bfloat)pt[gr][kk + 1] : (bfloat)0.0f;
#else
                simdgroup_float8x8 P;
                P.thread_elements()[0] = rowok ? pt[gr][kk] : 0.0f;
                P.thread_elements()[1] = rowok ? pt[gr][kk + 1] : 0.0f;
#endif
                #pragma clang loop unroll(full)
                for (int u = 0; u < TDS; ++u) {
                    int c = (int)sg + u * NSG;
                    if (TDS * NSG == CW || c < CW) {
#if PB
                        simdgroup_bfloat8x8 Vx;
                        simdgroup_load(Vx, vt + (r + f * 8) * LD + c * 8, LD);
#else
                        simdgroup_float8x8 Vx;
                        Vx.thread_elements()[0] = (float)vt[(r + f * 8 + fm) * LD + c * 8 + fn];
                        Vx.thread_elements()[1] = (float)vt[(r + f * 8 + fm) * LD + c * 8 + fn + 1];
#endif
                        simdgroup_multiply_accumulate(acc[u], P, Vx, acc[u]);
                    }
                }
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int u = 0; u < TDS; ++u) {
        int c8 = ((int)sg + u * NSG) * 8;
        if (!rowok || c8 >= BD) continue;
        size_t row = (size_t)t * Hq + h0 + gr;
        #pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            int c = c8 + (int)fn + e;
            float ov = acc[u].thread_elements()[e];
            if (ns == 1) {
                // a node of one segment: its fold of one, out
#if SINKS
                float Mt = a.sinks[h0 + gr] * M_LOG2E_F, Lt = 1.0f;
#else
                float Mt = -INFINITY, Lt = 0.0f;
#endif
                float Ot = 0.0f, fa = 0.0f, fb = 0.0f;
                if (LIVE(ms[gr])) btb_fold(Mt, Lt, Ot, ms[gr], ls[gr], ov, fa, fb, true);
                out[row * BD + c] = (OT)precise::divide(Ot, Lt);
            } else {
                if (c == 0) {
                    pm[row * NSX + s] = ms[gr];
                    pl[row * NSX + s] = ls[gr];
                }
                po[(row * NSX + s) * BD + c] = ov;
            }
        }
    }
#undef LROW
#undef ROFF
}

// the fold: output e (node, query head, column) of T * Hq * BD, the node's segments in order; a node of one segment is
// its segment's own (copied from `so` where that is another buffer). The inputs' pointer types the launch's (MLX binds
// a small input, a few heads' partials, in the constant address space)
template <int BD, typename MT, typename SKT, typename PMT, typename PLT, typename POT, typename SOT, typename OT>
inline void btb_alu_fold(MT meta, SKT sinks, PMT pm, PLT pl, POT po, SOT so, int T, int Hq, int NSX, uint e,
                         device OT* out) {
    if ((int)e >= T * Hq * BD) return;
    int row = e / BD, c = e % BD, t = row / Hq;
    int ns = meta[(size_t)t * 8 + 1];
    if (ns == 1) {
        if (so) out[e] = so[e];
        return;
    }
#if SINKS
    float Mt = sinks[row % Hq] * M_LOG2E_F, Lt = 1.0f;
#else
    float Mt = -INFINITY, Lt = 0.0f;
    (void)sinks;
#endif
    float Ot = 0.0f, a = 0.0f, b = 0.0f;
    for (int s = 0; s < ns; ++s) {
        float m = pm[row * NSX + s];
        if (LIVE(m)) btb_fold(Mt, Lt, Ot, m, pl[row * NSX + s], po[((size_t)row * NSX + s) * BD + c], a, b, true);
    }
    out[e] = (OT)precise::divide(Ot, Lt);
}
"""

# a verify pass's nodes on the prefill's tile (or head 256's phased kernel, the decode's phases over the rows), as
# device functions their own launches and the megakernel's attention ops call: a threadgroup (BQ flat rows, kv head
# hk, segment z of those the nodes span), row R node R / G and query head hk * G + R % G, a node's keys and rows as the
# segment function's. The nodes share the rows before `past` (past, batch, seg, pbase and kbatch the same in every
# node's meta): a block wholly before it is read once for the rows and is the prefill's on the matrix unit; a block
# reaching past it is each row's own, its scores a lane's chain over d, its P.V continued a key at a time. A node's
# (m, l, o) of the segment is its partial, its out where it has the one
_TREE_FN = r"""
#define TLROW(t, i) ((i) < past ? (i) : past + (int)a.path[(size_t)(t) * PD + ((i) - past)])
#if SEG && MAP
#define TROFF(r) ((r) >= segr ? b2 + (size_t)((r) - segr) : (size_t)hk * cap + a.rmap[r])
#elif SEG
#define TROFF(r) ((r) >= segr ? b2 + (size_t)((r) - segr) : b1 + (size_t)(r))
#elif MAP
#define TROFF(r) ((size_t)hk * cap + a.rmap[r])
#else
#define TROFF(r) (b1 + (size_t)(r))
#endif
#if SEG
#define TKP(r, off) (((r) >= segr ? a.k2 : a.k) + (off) * BD)
#define TVP(r, off) (((r) >= segr ? a.v2 : a.v) + (off) * BD)
#else
#define TKP(r, off) (a.k + (off) * BD)
#define TVP(r, off) (a.v + (off) * BD)
#endif
#if KQ && SEG
#define TKS(r, off) ((r) >= segr ? a.ks2[off] : a.ks[off])
#define TVS(r, off) ((r) >= segr ? a.vs2[off] : a.vs[off])
#elif KQ
#define TKS(r, off) (a.ks[off])
#define TVS(r, off) (a.vs[off])
#else
#define TKS(r, off) 1.0f
#define TVS(r, off) 1.0f
#endif

// segment z of those the nodes span: the earliest start's plus z
template <typename MT>
inline int btb_tree_seg(MT meta, int T, uint z) {
    int s0 = 0x7fffffff;
    for (int u = 0; u < T; ++u) s0 = min(s0, (int)meta[u * 8 + 3] / SEGK);
    return s0 + (int)z;
}

// the keys of segment s rows R0 .. R0 + BQ - 1 read, [klo, khi) (a row's past its own masked), and the latest a row's
// begin, lmax (a block wholly before it needs no mask)
template <typename MT>
inline void btb_tree_span(MT meta, int T, int G, int R0, int BQ, int s, thread int& klo, thread int& khi,
                          thread int& lmax) {
    klo = 0x7fffffff;
    khi = 0;
    lmax = 0;
    for (int u = R0 / G; u <= min((R0 + BQ - 1) / G, T - 1); ++u) {
        int nu = meta[u * 8], su = meta[u * 8 + 3], s0 = su / SEGK;
        if (s < s0 || s >= s0 + (int)meta[u * 8 + 1]) continue;
        int lu = max(su, s * SEGK);
        klo = min(klo, lu);
        khi = max(khi, min(nu, (s + 1) * SEGK));
        lmax = max(lmax, lu);
    }
}

// heads below 256: the prefill's tile, NSG simdgroups of 8 rows, K (transposed) then V staged in one buffer after Q
template <int BD, int G, int NSG, typename A, typename OT>
inline void btb_alu_tree(thread const A& a, int T, uint qb, uint hk, uint z, uint tid, threadgroup uchar* tgm,
                         device OT* out, device float* pm, device float* pl, device float* po) {
    constexpr int BQ = 8 * NSG, NT = 32 * NSG, BK = BTB_BK, TK = BK / 8, TD = BD / 8;
    // K and V loaded a block ahead where a block is 32 keys: measured (M3 Pro, cold caches) ~5% faster at head 64 over
    // short prefixes, 2-6% slower at 128 (16 keys a block)
    constexpr bool PF = BK == 32;
    constexpr int LDQ = BD + 8, LDK = BK + 8, LDV = BD + 8;
    int R0 = qb * BQ;
    if (R0 >= T * G) return;
    int Hq = a.Hq, Hk = a.Hk, cap = a.cap, PD = a.PD, NSX = a.NSX;
    int past = a.meta[2], batch = a.meta[4], segr = a.meta[5], pbase = a.meta[6], kbatch = a.meta[7];
    (void)batch;
    (void)segr;
    int s = btb_tree_seg(a.meta, T, z);
    int klo, khi, lmax;
    btb_tree_span(a.meta, T, G, R0, BQ, s, klo, khi, lmax);
    if (khi <= klo) return;
    size_t b1 = ((size_t)kbatch * Hk + hk) * cap + pbase;
#if SEG
    size_t b2 = ((size_t)batch * Hk + hk) * a.cap2;
#endif
    threadgroup bfloat* Qs = (threadgroup bfloat*)tgm;
    threadgroup bfloat* KVs = Qs + BQ * LDQ;
    uint lane = tid % 32, sg = tid / 32;
    float scale = a.sc * M_LOG2E_F;
    {
        constexpr int QR = BD / 4;
        int bi = tid / 4, bj = QR * (tid % 4), R = R0 + bi, t = R / G;
        auto src = a.q + ((size_t)min(t, T - 1) * Hq + hk * G + R % G) * BD + bj;
        #pragma clang loop unroll(full)
        for (int j = 0; j < QR; ++j) Qs[bi * LDQ + bj + j] = t < T ? (bfloat)src[j] : (bfloat)0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint fm = btb_fm(lane), fn = btb_fn(lane);
    const int tm = 8 * sg;
    // the lane's row: node t, query head hq, its keys in the segment [lo, hi), its partial's slot; a simdgroup wholly
    // past the pass's rows idles through the barriers
    int R = R0 + tm + fm, t = R / G, hq = hk * G + R % G, tc = min(t, T - 1);
    bool idle = R0 + tm >= T * G;
    int n_t = a.meta[tc * 8], ns_t = a.meta[tc * 8 + 1], st_t = a.meta[tc * 8 + 3];
    int slot = s - st_t / SEGK;
    bool act = t < T && slot >= 0 && slot < ns_t;
    int lo = max(st_t, s * SEGK), hi = min(n_t, (s + 1) * SEGK);
    int kb0 = klo / BK, kb1 = (khi + BK - 1) / BK;
    float M = MINIT, sum = 0.0f;
    simdgroup_float8x8 O[TD];
    #pragma clang loop unroll(full)
    for (int c = 0; c < TD; ++c) O[c] = simdgroup_float8x8(0.0f);
    // a row's four lanes differ in bits 0 and 3
    uint lbase = lane & ~9u;
    // a shared block's K and V: a thread's KR elements of row bi from column bj of each (under PF their loads behind
    // the block before's work), staged as stored (an int8 row's dequantized as it is staged)
    constexpr int KR = BK * BD / NT, KC = BD / KR;
    int bi = tid / KC, bj = KR * (tid % KC);
#if KQ
    char kx[KR], vx[KR];
#else
    bfloat kx[KR], vx[KR];
#endif
    float kxs = 1.0f, vxs = 1.0f;
#define TFETCHK(kb_)                                                                                                   \
    {                                                                                                                  \
        int j_ = (kb_) * BK + bi;                                                                                      \
        size_t o_ = TROFF(j_);                                                                                         \
        auto p_ = TKP(j_, o_) + bj;                                                                                    \
        kxs = TKS(j_, o_);                                                                                             \
        _Pragma("clang loop unroll(full)") for (int e = 0; e < KR; ++e) kx[e] = p_[e];                            \
    }
#define TFETCHV(kb_)                                                                                                 \
    {                                                                                                                  \
        int j_ = (kb_) * BK + bi;                                                                                      \
        size_t o_ = TROFF(j_);                                                                                         \
        auto p_ = TVP(j_, o_) + bj;                                                                                    \
        vxs = TVS(j_, o_);                                                                                             \
        _Pragma("clang loop unroll(full)") for (int e = 0; e < KR; ++e) vx[e] = p_[e];                            \
    }
    if (PF && (kb0 + 1) * BK <= past) {
        TFETCHK(kb0)
        TFETCHV(kb0)
    }
    for (int kb = kb0; kb < kb1; ++kb) {
        bool shared = (kb + 1) * BK <= past, ahead = PF && (kb + 2) * BK <= past;
        simdgroup_float8x8 S[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) S[f] = simdgroup_float8x8(0.0f);
        if (shared) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!PF) TFETCHK(kb)
            #pragma clang loop unroll(full)
            for (int e = 0; e < KR; ++e) KVs[(bj + e) * LDK + bi] = BTB_KV(kx, e, kxs);
            if (ahead) TFETCHK(kb + 1)
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!idle) {
                #pragma clang loop unroll(full)
                for (int dd = 0; dd < TD; ++dd) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_bfloat8x8 Qf, Kf[TK];
                    simdgroup_load(Qf, Qs + tm * LDQ + dd * 8, LDQ);
                    #pragma clang loop unroll(full)
                    for (int f = 0; f < TK; ++f) simdgroup_load(Kf[f], KVs + dd * 8 * LDK + f * 8, LDK);
                    simdgroup_barrier(mem_flags::mem_none);
                    #pragma clang loop unroll(full)
                    for (int f = 0; f < TK; ++f) simdgroup_multiply_accumulate(S[f], Qf, Kf[f], S[f]);
                }
                #pragma clang loop unroll(full)
                for (int f = 0; f < TK; ++f) {
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < 2; ++e) {
                        int j = kb * BK + f * 8 + fn + e;
                        float x = S[f].thread_elements()[e] * scale;
                        S[f].thread_elements()[e] = kb * BK < lmax && (j < lo || j >= hi) ? -INFINITY : x;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!PF) TFETCHV(kb)
            #pragma clang loop unroll(full)
            for (int e = 0; e < KR; ++e) KVs[bi * LDV + bj + e] = BTB_KV(vx, e, vxs);
            if (ahead) TFETCHV(kb + 1)
        } else if (!idle) {
            // the row's own keys: each score the chain over d, q from the stage, the key's row through the path
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    int j = kb * BK + f * 8 + (int)fn + e;
                    float x = -INFINITY;
                    if (j >= lo && j < hi) {
                        int r = TLROW(tc, j);
                        size_t off = TROFF(r);
                        auto kp = TKP(r, off);
                        float ksc = TKS(r, off);
                        (void)ksc;
                        float acc = 0.0f;
                        for (int d = 0; d < BD; ++d)
                            acc = fma((float)Qs[(tm + fm) * LDQ + d], (float)BTB_KV(kp, d, ksc), acc);
                        x = acc * scale;
                    }
                    S[f].thread_elements()[e] = x;
                }
            }
        }
        if (!idle) {
            // the block's max, p and sum for the lane's row: the prefill's own
            float bm = -INFINITY;
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) bm = max(bm, max(S[f].thread_elements()[0], S[f].thread_elements()[1]));
            bm = max(bm, simd_shuffle_xor(bm, 1));
            bm = max(bm, simd_shuffle_xor(bm, 8));
            bool up = EAGER ? true : bm > M + TAU;
            float nm = EAGER ? max(M, bm) : up ? bm : M;
            bool dead = EAGER ? false : nm == -INFINITY;
            float factor = up ? fast::exp2(M - nm) : 1.0f;
            M = nm;
            float rs = 0.0f;
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                float p0 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[0] - nm));
                float p1 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[1] - nm));
                S[f].thread_elements()[0] = p0;
                S[f].thread_elements()[1] = p1;
                rs = f == 0 ? p0 + p1 : rs + (p0 + p1);
            }
            rs = rs + simd_shuffle_xor(rs, 1);
            rs = rs + simd_shuffle_xor(rs, 8);
            sum = fma(sum, factor, rs);
            if (EAGER || !simd_all(factor == 1.0f)) {
                #pragma clang loop unroll(full)
                for (int c = 0; c < TD; ++c) {
                    O[c].thread_elements()[0] *= factor;
                    O[c].thread_elements()[1] *= factor;
                }
            }
        }
        if (shared) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!idle) {
                #pragma clang loop unroll(full)
                for (int c = 0; c < TD; ++c) {
                    #pragma clang loop unroll(full)
                    for (int f = 0; f < TK; ++f) {
                        if (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_float8x8 Vf;
                        Vf.thread_elements()[0] = (float)KVs[(f * 8 + fm) * LDV + c * 8 + fn];
                        Vf.thread_elements()[1] = (float)KVs[(f * 8 + fm) * LDV + c * 8 + fn + 1];
                        if (BD >= 128) simdgroup_barrier(mem_flags::mem_none);
                        simdgroup_multiply_accumulate(O[c], S[f], Vf, O[c]);
                    }
                }
            }
        } else if (!idle) {
            // the row's P.V a key at a time in the block's order: key kk's p from the row's lane holding it, V from the
            // key's row (zero past the node's keys)
            for (int kk = 0; kk < BK; ++kk) {
                int f = kk / 8, w = kk % 8;
                uint src = lbase | ((uint)(w >> 1) & 1u) | (((uint)(w >> 2) & 1u) << 3);
                float p = simd_shuffle(S[f].thread_elements()[w & 1], (ushort)src);
                int j = kb * BK + kk;
                bool in = j < n_t;
                int r = TLROW(tc, min(j, n_t - 1));
                size_t off = TROFF(r);
                auto vp = TVP(r, off) + fn;
                float vsc = TVS(r, off);
                (void)vsc;
                #pragma clang loop unroll(full)
                for (int c = 0; c < TD; ++c) {
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < 2; ++e) {
                        float vv = in ? (float)BTB_KV(vp, c * 8 + e, vsc) : 0.0f;
                        O[c].thread_elements()[e] = fma(p, vv, O[c].thread_elements()[e]);
                    }
                }
            }
        }
    }
    if (!act) return;
    size_t row = (size_t)t * Hq + hq;
    if (ns_t == 1) {
#if SINKS
        float Mt = a.sinks[hq] * M_LOG2E_F, Lt = 1.0f;
#else
        float Mt = -INFINITY, Lt = 0.0f;
#endif
        float fa = 0.0f, fb = 0.0f;
        #pragma clang loop unroll(full)
        for (int c = 0; c < TD; ++c) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                float Ox = 0.0f;
                if (LIVE(M)) btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], fa, fb, c == 0 && e == 0);
                out[row * BD + c * 8 + fn + e] = (OT)precise::divide(Ox, Lt);
            }
        }
    } else {
        if (fn == 0) {
            pm[row * NSX + slot] = M;
            pl[row * NSX + slot] = sum;
        }
        #pragma clang loop unroll(full)
        for (int c = 0; c < TD; ++c) {
            po[(row * NSX + slot) * BD + c * 8 + fn] = O[c].thread_elements()[0];
            po[(row * NSX + slot) * BD + c * 8 + fn + 1] = O[c].thread_elements()[1];
        }
    }
}

// head 256: the head-256 prefill's tile, 32 rows in groups of 8 (row group p rows 8p..), each group's pair of
// simdgroups the score's halves and the columns' (hv dims 128 hv..) for P.V, K and V staged row-major together, S's
// halves traded through xs and added low first; a row's own keys each lane's chain over its half, traded the same way
template <int BD, int G, int NSG, typename A, typename OT>
inline void btb_alu_tree256(thread const A& a, int T, uint qb, uint hk, uint z, uint tid, threadgroup uchar* tgm,
                            device OT* out, device float* pm, device float* pl, device float* po) {
    static_assert(BD == 256 && NSG == 8, "head 256's tile: 8 simdgroups");
    constexpr int BQ = 32, BK = 16, HD = 128, TK = 2, TH = HD / 8, LD = BD + 8;
    int R0 = qb * BQ;
    if (R0 >= T * G) return;
    int Hq = a.Hq, Hk = a.Hk, cap = a.cap, PD = a.PD, NSX = a.NSX;
    int past = a.meta[2], batch = a.meta[4], segr = a.meta[5], pbase = a.meta[6], kbatch = a.meta[7];
    (void)batch;
    (void)segr;
    int s = btb_tree_seg(a.meta, T, z);
    int klo, khi, lmax;
    btb_tree_span(a.meta, T, G, R0, BQ, s, klo, khi, lmax);
    if (khi <= klo) return;
    size_t b1 = ((size_t)kbatch * Hk + hk) * cap + pbase;
#if SEG
    size_t b2 = ((size_t)batch * Hk + hk) * a.cap2;
#endif
    threadgroup bfloat* Ks = (threadgroup bfloat*)tgm;
    threadgroup bfloat* Vs = Ks + BK * LD;
    threadgroup float (*xs)[TK * 64] = (threadgroup float (*)[TK * 64])(Vs + BK * LD);
    uint lane = tid % 32, sg = tid / 32, pr = sg >> 1, hv = sg & 1;
    float scale = a.sc * M_LOG2E_F;
    uint fm = btb_fm(lane), fn = btb_fn(lane);
    int R = R0 + 8 * pr + fm, t = R / G, hq = hk * G + R % G, tc = min(t, T - 1);
    bool idle = R0 + 8 * (int)pr >= T * G;
    auto qrow = a.q + ((size_t)tc * Hq + hq) * BD + hv * HD;
    auto qr = qrow + fn;
    int n_t = a.meta[tc * 8], ns_t = a.meta[tc * 8 + 1], st_t = a.meta[tc * 8 + 3];
    int slot = s - st_t / SEGK;
    bool act = t < T && slot >= 0 && slot < ns_t;
    int lo = max(st_t, s * SEGK), hi = min(n_t, (s + 1) * SEGK);
    int kb0 = klo / BK, kb1 = (khi + BK - 1) / BK;
    float M = MINIT, sum = 0.0f;
    simdgroup_float8x8 O[TH];
    #pragma clang loop unroll(full)
    for (int c = 0; c < TH; ++c) O[c] = simdgroup_float8x8(0.0f);
    uint lbase = lane & ~9u;
    // a shared block's K and V: a thread's 16 elements of row vi from column vj of each (an int8 row's bytes and
    // scale, dequantized as it is staged; loaded a block ahead, measured 2-14% slower)
    int vi = tid / 16, vj = 16 * (tid % 16);
#if KQ
    char kx[16], vx[16];
    float kxs = 1.0f, vxs = 1.0f;
#define TFETCH(kb_)                                                                                                    \
    {                                                                                                                  \
        int j_ = (kb_) * BK + vi;                                                                                      \
        size_t o_ = TROFF(j_);                                                                                         \
        auto kp_ = TKP(j_, o_) + vj;                                                                                   \
        auto vp_ = TVP(j_, o_) + vj;                                                                                   \
        kxs = TKS(j_, o_);                                                                                             \
        vxs = TVS(j_, o_);                                                                                             \
        _Pragma("clang loop unroll(full)") for (int e = 0; e < 16; ++e) {                                              \
            kx[e] = kp_[e];                                                                                            \
            vx[e] = vp_[e];                                                                                            \
        }                                                                                                              \
    }
#else
    uint4 kx[2], vx[2];
#define TFETCH(kb_)                                                                                                    \
    {                                                                                                                  \
        int j_ = (kb_) * BK + vi;                                                                                      \
        size_t o_ = TROFF(j_);                                                                                         \
        auto kp_ = (const device uint4*)(TKP(j_, o_) + vj);                                                            \
        auto vp_ = (const device uint4*)(TVP(j_, o_) + vj);                                                            \
        kx[0] = kp_[0];                                                                                                \
        kx[1] = kp_[1];                                                                                                \
        vx[0] = vp_[0];                                                                                                \
        vx[1] = vp_[1];                                                                                                \
    }
#endif
    for (int kb = kb0; kb < kb1; ++kb) {
        bool shared = (kb + 1) * BK <= past;
        simdgroup_float8x8 S[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) S[f] = simdgroup_float8x8(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared) {
            TFETCH(kb)
#if KQ
            {
                bfloat xk[16], xv[16];
                #pragma clang loop unroll(full)
                for (int e = 0; e < 16; ++e) {
                    xk[e] = BTB_KV(kx, e, kxs);
                    xv[e] = BTB_KV(vx, e, vxs);
                }
                *(threadgroup uint4*)(Ks + vi * LD + vj) = btb_pack8(xk);
                *(threadgroup uint4*)(Ks + vi * LD + vj + 8) = btb_pack8(xk + 8);
                *(threadgroup uint4*)(Vs + vi * LD + vj) = btb_pack8(xv);
                *(threadgroup uint4*)(Vs + vi * LD + vj + 8) = btb_pack8(xv + 8);
            }
#else
            *(threadgroup uint4*)(Ks + vi * LD + vj) = kx[0];
            *(threadgroup uint4*)(Ks + vi * LD + vj + 8) = kx[1];
            *(threadgroup uint4*)(Vs + vi * LD + vj) = vx[0];
            *(threadgroup uint4*)(Vs + vi * LD + vj + 8) = vx[1];
#endif
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!idle) {
                #pragma clang loop unroll(full)
                for (int dd = 0; dd < TH; ++dd) {
                    #pragma clang loop unroll(full)
                    for (int f = 0; f < TK; ++f) {
                        simdgroup_bfloat8x8 Kb;
                        simdgroup_load(Kb, Ks + f * 8 * LD + hv * HD + dd * 8, LD, ulong2(0, 0), true);
                        simdgroup_bfloat8x8 Qf;
                        Qf.thread_elements()[0] = (bfloat)qr[dd * 8];
                        Qf.thread_elements()[1] = (bfloat)qr[dd * 8 + 1];
                        simdgroup_multiply_accumulate(S[f], Qf, Kb, S[f]);
                    }
                }
            }
        } else if (!idle) {
            // the row's own keys: this half's chain over its 128 dims, the key's row through the path
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    int j = kb * BK + f * 8 + (int)fn + e;
                    float acc = 0.0f;
                    if (j >= lo && j < hi) {
                        int r = TLROW(tc, j);
                        size_t off = TROFF(r);
                        auto kp = TKP(r, off) + hv * HD;
                        float ksc = TKS(r, off);
                        (void)ksc;
                        for (int d = 0; d < HD; ++d) acc = fma((float)(bfloat)qrow[d], (float)BTB_KV(kp, d, ksc), acc);
                    }
                    S[f].thread_elements()[e] = acc;
                }
            }
        }
        if (!idle) {
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) simdgroup_store(S[f], &xs[sg][f * 64], 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (idle) continue;
        bool need = !shared || kb * BK < lmax;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                float other = xs[sg ^ 1][f * 64 + fm * 8 + fn + e];
                float mine = S[f].thread_elements()[e];
                float x = (hv == 0 ? mine + other : other + mine) * scale;
                int j = kb * BK + f * 8 + fn + e;
                S[f].thread_elements()[e] = need && (j < lo || j >= hi) ? -INFINITY : x;
            }
        }
        float bm = -INFINITY;
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) bm = max(bm, max(S[f].thread_elements()[0], S[f].thread_elements()[1]));
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        bool up = EAGER ? true : bm > M + TAU;
        float nm = EAGER ? max(M, bm) : up ? bm : M;
        bool dead = EAGER ? false : nm == -INFINITY;
        float factor = up ? fast::exp2(M - nm) : 1.0f;
        M = nm;
        float rs = 0.0f;
        simdgroup_bfloat8x8 P[TK];
        #pragma clang loop unroll(full)
        for (int f = 0; f < TK; ++f) {
            float p0 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[0] - nm));
            float p1 = dead ? 0.0f : BTB_P(fast::exp2(S[f].thread_elements()[1] - nm));
            P[f].thread_elements()[0] = (bfloat)p0;
            P[f].thread_elements()[1] = (bfloat)p1;
            S[f].thread_elements()[0] = p0;
            S[f].thread_elements()[1] = p1;
            rs = f == 0 ? p0 + p1 : rs + (p0 + p1);
        }
        rs = rs + simd_shuffle_xor(rs, 1);
        rs = rs + simd_shuffle_xor(rs, 8);
        sum = fma(sum, factor, rs);
        if (EAGER || !simd_all(factor == 1.0f)) {
            #pragma clang loop unroll(full)
            for (int c = 0; c < TH; ++c) {
                O[c].thread_elements()[0] *= factor;
                O[c].thread_elements()[1] *= factor;
            }
        }
        if (shared) {
            #pragma clang loop unroll(full)
            for (int f = 0; f < TK; ++f) {
                #pragma clang loop unroll(full)
                for (int c = 0; c < TH; ++c) {
                    simdgroup_bfloat8x8 Vb;
                    simdgroup_load(Vb, Vs + f * 8 * LD + hv * HD + c * 8, LD);
                    simdgroup_multiply_accumulate(O[c], P[f], Vb, O[c]);
                }
            }
        } else {
            for (int kk = 0; kk < BK; ++kk) {
                int f = kk / 8, w = kk % 8;
                uint src = lbase | ((uint)(w >> 1) & 1u) | (((uint)(w >> 2) & 1u) << 3);
                float p = simd_shuffle(S[f].thread_elements()[w & 1], (ushort)src);
                int j = kb * BK + kk;
                bool in = j < n_t;
                int r = TLROW(tc, min(j, n_t - 1));
                size_t off = TROFF(r);
                auto vp = TVP(r, off) + hv * HD + fn;
                float vsc = TVS(r, off);
                (void)vsc;
                #pragma clang loop unroll(full)
                for (int c = 0; c < TH; ++c) {
                    #pragma clang loop unroll(full)
                    for (int e = 0; e < 2; ++e) {
                        float vv = in ? (float)BTB_KV(vp, c * 8 + e, vsc) : 0.0f;
                        O[c].thread_elements()[e] = fma(p, vv, O[c].thread_elements()[e]);
                    }
                }
            }
        }
    }
    if (!act) return;
    size_t row = (size_t)t * Hq + hq;
    if (ns_t == 1) {
#if SINKS
        float Mt = a.sinks[hq] * M_LOG2E_F, Lt = 1.0f;
#else
        float Mt = -INFINITY, Lt = 0.0f;
#endif
        float fa = 0.0f, fb = 0.0f;
        #pragma clang loop unroll(full)
        for (int c = 0; c < TH; ++c) {
            #pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                float Ox = 0.0f;
                if (LIVE(M)) btb_fold(Mt, Lt, Ox, M, sum, O[c].thread_elements()[e], fa, fb, c == 0 && e == 0);
                out[row * BD + hv * HD + c * 8 + fn + e] = (OT)precise::divide(Ox, Lt);
            }
        }
    } else {
        if (fn == 0 && hv == 0) {
            pm[row * NSX + slot] = M;
            pl[row * NSX + slot] = sum;
        }
        #pragma clang loop unroll(full)
        for (int c = 0; c < TH; ++c) {
            po[(row * NSX + slot) * BD + hv * HD + c * 8 + fn] = O[c].thread_elements()[0];
            po[(row * NSX + slot) * BD + hv * HD + c * 8 + fn + 1] = O[c].thread_elements()[1];
        }
    }
}

// head 256 phased: the decode's phases over R of the tile's rows (R / 8 row fragments), a segment in R / 8 passes
// whose scores fit threadgroup memory (the walk's max, the sums and P.V carried across): a pass's scores at once
// (shared blocks S^T = K . Q^T on the matrix unit, K's rows straight from device as A's; a row's own keys scalar
// chains), block b on simdgroup b % NSG; P.V a column fragment a chain over the blocks, V's rows straight from device
template <int BD, int G, int NSG, int R, typename A, typename OT>
inline void btb_alu_phased256(thread const A& a, int T, uint qb, uint hk, uint z, uint tid, threadgroup uchar* tgm,
                              device OT* out, device float* pm, device float* pl, device float* po) {
    static_assert(BD == 256 && (R == 8 || R == 16) && BD / 8 % NSG == 0, "head 256 phased: 8 or 16 rows");
    constexpr int RF = R / 8, SUB = SEGK / RF, BK = 16, NBS = SUB / BK, NT = 32 * NSG, HD = BD / 2, TH = HD / 8;
    constexpr int LDQ = BD + 8, TDS = BD / 8 / NSG;
    int R0 = qb * R;
    if (R0 >= T * G) return;
    int Hq = a.Hq, Hk = a.Hk, cap = a.cap, PD = a.PD, NSX = a.NSX;
    int past = a.meta[2], batch = a.meta[4], segr = a.meta[5], pbase = a.meta[6], kbatch = a.meta[7];
    (void)batch;
    (void)segr;
    int s = btb_tree_seg(a.meta, T, z);
    int klo, khi, lmax;
    btb_tree_span(a.meta, T, G, R0, R, s, klo, khi, lmax);
    if (khi <= klo) return;
    size_t b1 = ((size_t)kbatch * Hk + hk) * cap + pbase;
#if SEG
    size_t b2 = ((size_t)batch * Hk + hk) * a.cap2;
#endif
    int jb = klo / BK * BK, nb = ((khi + BK - 1) / BK * BK - jb) / BK;
    threadgroup float (*pt)[SUB] = (threadgroup float (*)[SUB])tgm;
    threadgroup float (*bm)[NBS] = (threadgroup float (*)[NBS])(tgm + 4 * R * SUB);
    threadgroup float (*mb)[NBS] = bm + R;
    threadgroup float (*ft)[NBS] = mb + R;
    threadgroup float (*rsum)[NBS] = ft + R;
    threadgroup float* ms = (threadgroup float*)(rsum + R);
    threadgroup float* ls = ms + R;
    threadgroup int* rlo = (threadgroup int*)(ls + R);
    threadgroup int* rhi = rlo + R;
    threadgroup bfloat* qs = (threadgroup bfloat*)(rhi + R);
    uint lane = tid % 32, sg = tid / 32, fm = btb_fm(lane), fn = btb_fn(lane);
    float scale = a.sc * M_LOG2E_F;
    for (int e = tid; e < R * BD; e += NT) {
        int r_ = e / BD, Rr = R0 + r_;
        qs[r_ * LDQ + e % BD] = (bfloat)a.q[((size_t)min(Rr / G, T - 1) * Hq + hk * G + Rr % G) * BD + e % BD];
    }
    // a row's keys in this segment [rlo, rhi) (none for a row past the nodes), its running max and sum
    if ((int)tid < R) {
        int t_ = (R0 + (int)tid) / G;
        bool has = t_ < T;
        rlo[tid] = has ? max((int)a.meta[t_ * 8 + 3], s * SEGK) : 0;
        rhi[tid] = has ? min((int)a.meta[t_ * 8], (s + 1) * SEGK) : 0;
        ms[tid] = MINIT;
        ls[tid] = 0.0f;
    }
    int tcm[RF], n_m[RF];
    simdgroup_float8x8 acc[RF][TDS];
    #pragma clang loop unroll(full)
    for (int rf = 0; rf < RF; ++rf) {
        tcm[rf] = min((R0 + rf * 8 + (int)fm) / G, T - 1);
        n_m[rf] = a.meta[tcm[rf] * 8];
        #pragma clang loop unroll(full)
        for (int u = 0; u < TDS; ++u) acc[rf][u] = simdgroup_float8x8(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b0 = 0; b0 < nb; b0 += NBS) {
        int nbs = min(NBS, nb - b0), nk = nbs * BK, kb0 = jb + b0 * BK;
        for (int bl = sg; bl < nbs; bl += NSG) {
            int k0 = kb0 + bl * BK;
            if (k0 + BK <= past) {
                #pragma clang loop unroll(full)
                for (int f = 0; f < 2; ++f) {
                    int j = k0 + f * 8 + (int)fm;
                    size_t off = TROFF(j);
                    auto kp = TKP(j, off) + fn;
                    float ksc = TKS(j, off);
                    (void)ksc;
                    simdgroup_float8x8 lo8[RF], hi8[RF];
                    #pragma clang loop unroll(full)
                    for (int rf = 0; rf < RF; ++rf) {
                        lo8[rf] = simdgroup_float8x8(0.0f);
                        hi8[rf] = simdgroup_float8x8(0.0f);
                    }
                    #pragma clang loop unroll(full)
                    for (int dd = 0; dd < TH; ++dd) {
                        simdgroup_bfloat8x8 K0, K1;
                        K0.thread_elements()[0] = BTB_KV(kp, dd * 8, ksc);
                        K0.thread_elements()[1] = BTB_KV(kp, dd * 8 + 1, ksc);
                        K1.thread_elements()[0] = BTB_KV(kp, HD + dd * 8, ksc);
                        K1.thread_elements()[1] = BTB_KV(kp, HD + dd * 8 + 1, ksc);
                        #pragma clang loop unroll(full)
                        for (int rf = 0; rf < RF; ++rf) {
                            simdgroup_bfloat8x8 Q0, Q1;
                            simdgroup_load(Q0, qs + rf * 8 * LDQ + dd * 8, LDQ, ulong2(0, 0), true);
                            simdgroup_load(Q1, qs + rf * 8 * LDQ + HD + dd * 8, LDQ, ulong2(0, 0), true);
                            simdgroup_multiply_accumulate(lo8[rf], K0, Q0, lo8[rf]);
                            simdgroup_multiply_accumulate(hi8[rf], K1, Q1, hi8[rf]);
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (int rf = 0; rf < RF; ++rf) {
                        #pragma clang loop unroll(full)
                        for (int e = 0; e < 2; ++e) {
                            int r_ = rf * 8 + (int)fn + e;
                            float x = (lo8[rf].thread_elements()[e] + hi8[rf].thread_elements()[e]) * scale;
                            pt[r_][j - kb0] = j < rlo[r_] || j >= rhi[r_] ? -INFINITY : x;
                        }
                    }
                }
            } else {
                for (int e = lane; e < R * BK; e += 32) {
                    int r_ = e / BK, j = k0 + e % BK;
                    float x = -INFINITY;
                    if (j >= rlo[r_] && j < rhi[r_]) {
                        int r = TLROW(min((R0 + r_) / G, T - 1), j);
                        size_t off = TROFF(r);
                        auto kp = TKP(r, off);
                        float ksc = TKS(r, off);
                        (void)ksc;
                        float c0 = 0.0f, c1 = 0.0f;
                        for (int d = 0; d < HD; ++d) c0 = fma((float)qs[r_ * LDQ + d], (float)BTB_KV(kp, d, ksc), c0);
                        for (int d = HD; d < BD; ++d) c1 = fma((float)qs[r_ * LDQ + d], (float)BTB_KV(kp, d, ksc), c1);
                        x = (c0 + c1) * scale;
                    }
                    pt[r_][j - kb0] = x;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the pass's block maxima, the walk from the last pass's max, p and the sums: btb_alu_segment's
        for (int e0 = tid - lane; e0 < R * nk; e0 += NT) {
            int e = e0 + lane, g = min(e / nk, R - 1), i = e - g * nk;
            bool ok = e < R * nk;
            float x = ok ? pt[g][i] : -INFINITY;
            for (int o = 1; o < BK; o <<= 1) x = max(x, simd_shuffle_xor(x, o));
            if (ok && i % BK == 0) bm[g][i / BK] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int g = sg; g < R; g += NSG) {
            float carry = ms[g], prev = MINIT, nmb = MINIT;
            bool upb = false;
            for (int i = 0; i < nbs; ++i) {
                float x = bm[g][i];
                bool up = EAGER ? true : x > carry + TAU;
                float nm = EAGER ? max(carry, x) : up ? x : carry;
                if ((int)lane == i) {
                    prev = carry;
                    upb = up;
                    nmb = nm;
                }
                carry = nm;
            }
            if ((int)lane < nbs) {
                mb[g][lane] = nmb;
                ft[g][lane] = upb ? fast::exp2(prev - nmb) : 1.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) ms[g] = carry;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e0 = tid - lane; e0 < R * nk; e0 += NT) {
            int e = e0 + lane, g = min(e / nk, R - 1), i = e - g * nk, b = i / BK;
            bool ok = e < R * nk;
            float m_ = ok ? mb[g][b] : -INFINITY;
            float p = ok && LIVE(m_) ? BTB_P(fast::exp2(pt[g][i] - m_)) : 0.0f;
            if (ok) pt[g][i] = p;
            float x = p + simd_shuffle_xor(p, 1);
            float s_ = x + simd_shuffle_xor(x, 8);
            float r = s_ + simd_shuffle_xor(s_, 2);
            r = r + simd_shuffle_xor(r, 4);
            if (ok && i % BK == 0) rsum[g][b] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)tid < R) {
            float sum = ls[tid];
            for (int b = 0; b < nbs; ++b) sum = fma(sum, ft[tid][b], rsum[tid][b]);
            ls[tid] = sum;
        }
        for (int bl = 0; bl < nbs; ++bl) {
            #pragma clang loop unroll(full)
            for (int rf = 0; rf < RF; ++rf) {
                float fa = ft[rf * 8 + fm][bl];
                #pragma clang loop unroll(full)
                for (int u = 0; u < TDS; ++u) {
                    acc[rf][u].thread_elements()[0] *= fa;
                    acc[rf][u].thread_elements()[1] *= fa;
                }
            }
            int k0 = kb0 + bl * BK;
            if (k0 + BK <= past) {
                #pragma clang loop unroll(full)
                for (int f = 0; f < 2; ++f) {
                    simdgroup_bfloat8x8 P[RF];
                    #pragma clang loop unroll(full)
                    for (int rf = 0; rf < RF; ++rf) {
                        P[rf].thread_elements()[0] = (bfloat)pt[rf * 8 + fm][bl * BK + f * 8 + fn];
                        P[rf].thread_elements()[1] = (bfloat)pt[rf * 8 + fm][bl * BK + f * 8 + fn + 1];
                    }
                    int j = k0 + f * 8 + (int)fm;
                    size_t off = TROFF(j);
                    auto vp = TVP(j, off) + fn;
                    float vsc = TVS(j, off);
                    (void)vsc;
                    #pragma clang loop unroll(full)
                    for (int u = 0; u < TDS; ++u) {
                        int c = (int)sg + u * NSG;
                        simdgroup_bfloat8x8 Vx;
                        Vx.thread_elements()[0] = BTB_KV(vp, c * 8, vsc);
                        Vx.thread_elements()[1] = BTB_KV(vp, c * 8 + 1, vsc);
                        #pragma clang loop unroll(full)
                        for (int rf = 0; rf < RF; ++rf)
                            simdgroup_multiply_accumulate(acc[rf][u], P[rf], Vx, acc[rf][u]);
                    }
                }
            } else {
                // the rows' own keys: per lane, the tile's
                #pragma clang loop unroll(full)
                for (int rf = 0; rf < RF; ++rf) {
                    for (int kk = 0; kk < BK; ++kk) {
                        float p = pt[rf * 8 + fm][bl * BK + kk];
                        int j = k0 + kk;
                        bool in = j < n_m[rf];
                        int r = TLROW(tcm[rf], min(j, n_m[rf] - 1));
                        size_t off = TROFF(r);
                        auto vp = TVP(r, off) + fn;
                        float vsc = TVS(r, off);
                        (void)vsc;
                        #pragma clang loop unroll(full)
                        for (int u = 0; u < TDS; ++u) {
                            #pragma clang loop unroll(full)
                            for (int e = 0; e < 2; ++e) {
                                float vv = in ? (float)BTB_KV(vp, ((int)sg + u * NSG) * 8 + e, vsc) : 0.0f;
                                acc[rf][u].thread_elements()[e] = fma(p, vv, acc[rf][u].thread_elements()[e]);
                            }
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma clang loop unroll(full)
    for (int rf = 0; rf < RF; ++rf) {
        int Rm = R0 + rf * 8 + (int)fm, tm = Rm / G;
        int ns_m = a.meta[tcm[rf] * 8 + 1], slot = s - (int)a.meta[tcm[rf] * 8 + 3] / SEGK;
        if (tm >= T || slot < 0 || slot >= ns_m) continue;
        int hq = hk * G + Rm % G;
        size_t row = (size_t)tm * Hq + hq;
        float M = ms[rf * 8 + fm], sum = ls[rf * 8 + fm];
        if (ns_m == 1) {
#if SINKS
            float Mt = a.sinks[hq] * M_LOG2E_F, Lt = 1.0f;
#else
            float Mt = -INFINITY, Lt = 0.0f;
#endif
            float fa = 0.0f, fb = 0.0f;
            #pragma clang loop unroll(full)
            for (int u = 0; u < TDS; ++u) {
                #pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    float Ox = 0.0f;
                    if (LIVE(M))
                        btb_fold(Mt, Lt, Ox, M, sum, acc[rf][u].thread_elements()[e], fa, fb, u == 0 && e == 0);
                    out[row * BD + ((int)sg + u * NSG) * 8 + fn + e] = (OT)precise::divide(Ox, Lt);
                }
            }
        } else {
            if (fn == 0 && sg == 0) {
                pm[row * NSX + slot] = M;
                pl[row * NSX + slot] = sum;
            }
            #pragma clang loop unroll(full)
            for (int u = 0; u < TDS; ++u) {
                size_t o_ = (row * NSX + slot) * BD + ((int)sg + u * NSG) * 8 + fn;
                po[o_] = acc[rf][u].thread_elements()[0];
                po[o_ + 1] = acc[rf][u].thread_elements()[1];
            }
        }
    }
}
#undef TLROW
#undef TROFF
#undef TKP
#undef TVP
#undef TKS
#undef TVS
#undef TFETCHK
#undef TFETCHV
#undef TFETCH
"""

_kernels: dict[tuple[object, ...], MetalKernel] = {}
_arrays: dict[tuple[object, ...], mx_.array] = {}


def _const(key: tuple[object, ...], vals: Sequence[float | int], dt: mx_.Dtype) -> mx_.array:
    """a small array of the launch's, made once and kept for the layers after (a chunk's meta, a scale); the kept
    ones dropped together past 4096 (a session's chunks each a key of its own)"""
    a = _arrays.get(key)
    if a is None:
        if len(_arrays) >= 4096:
            _arrays.clear()
        a = _arrays[key] = mx.array(list(vals), dtype=dt)
        mx.eval(a)
    return a


def _defs(D: int, **flags: int | bool) -> str:
    defs = {"EAGER": eager(D), "PB": bf16p(D), **flags}
    return "".join(f"#define {k} {int(v)}\n" for k, v in defs.items()) + _HEADER


def _prefill_kernel(D: int, g: int, sinks: bool, mapped: bool, kq: bool) -> MetalKernel:
    key = ("prefill", D, g if D == 256 else 0, sinks, mapped, kq)
    with _gemv_lock:
        k = _kernels.get(key)
        if k is None:
            src = _PREFILL256.replace("G_V", str(g)) if D == 256 else _PREFILL
            rows, grp = "i8" if kq else "bf16", f"_g{g}" if D == 256 else ""
            k = _kernels[key] = metal_kernel(
                name=f"btb_aluminum_prefill_{rows}_d{D}{grp}{'_sinks' if sinks else ''}{'_map' if mapped else ''}",
                input_names=[
                    "q",
                    "k",
                    "v",
                    "meta",
                    "sc",
                    *(["ks", "vs"] if kq else []),
                    *(["sinks"] if sinks else []),
                    *(["rmap", "k2", "v2"] if mapped else []),
                ],
                output_names=["out"],
                header=_defs(D, KQ=kq, SINKS=sinks, MAP=mapped),
                source=src.replace("BK_V", str(block(D))).replace("D_V", str(D)),
            )
    return k


def prefill(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    past: int,
    scale: float,
    window: int | None = None,
    sinks: mx_.array | None = None,
    odt: mx_.Dtype | None = None,
    rmap: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    kbatch: int = 0,
    pbase: int = 0,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
) -> mx_.array:
    """Causal attention for a chunk: `q` [T, Hq, D] (any float dtype, taken as bf16), `k`/`v` the cache [B, Hk, cap,
    D] (bf16; int8 with `ks`/`vs` [B, Hk, cap] a scale a row, read as the dequantized bf16) holding the chunk's rows at
    [past, past + T) of slice `kbatch` from row `pbase`; row t attends over keys (past + t - window, past + t]. A paged
    cache's rows before `past` through `rmap` (uint32), the chunk's own in `k2`/`v2` [1, Hk, >= T, D]; `sinks` [Hq]
    float32 a softmax column with no value. Returns [T, Hq, D] in `odt` (q's dtype by default), lazily"""
    T = int(q.shape[0])
    chunk = [(T, int(past), int(pbase), 0, int(kbatch))]
    return _prefill(q, k, v, chunk, scale, window, sinks, odt, rmap, k2, v2, ks, vs)


def prefill_rows(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    lens: Sequence[int],
    pbases: Sequence[int],
    scale: float,
    window: int | None = None,
    odt: mx_.Dtype | None = None,
    kbatch: int = 0,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
) -> mx_.array:
    """A batched prefill in one launch, each prompt as its own prefill computes it: prompt b's lens[b] rows end to end
    in `q` [sum(lens), Hq, D], its keys from row pbases[b] of slice `kbatch` of `k`/`v` [B, Hk, cap, D] (bf16, or int8
    with `ks`/`vs` as `prefill` takes them). Returns [sum(lens), Hq, D] in `odt` (q's dtype by default), lazily"""
    chunks, qoff = [], 0
    for L, pb in zip(lens, pbases, strict=True):
        chunks.append((int(L), 0, int(pb), qoff, int(kbatch)))
        qoff += int(L)
    return _prefill(q, k, v, chunks, scale, window, None, odt, None, None, None, ks, vs)


def _prefill(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    chunks: Sequence[tuple[int, int, int, int, int]],
    scale: float,
    window: int | None,
    sinks: mx_.array | None,
    odt: mx_.Dtype | None,
    rmap: mx_.array | None,
    k2: mx_.array | None,
    v2: mx_.array | None,
    ks: mx_.array | None,
    vs: mx_.array | None,
) -> mx_.array:
    """`prefill` over `chunks` (T, past, pbase, qoff, kbatch), a grid row each"""
    m = mx
    Tq, Hq, D = (int(x) for x in q.shape)
    Hk = int(k.shape[1])
    g = Hq // Hk
    scales: list[mx_.array] = []
    if ks is not None and vs is not None:
        scales = [ks, vs]
    if D not in HEADS or g * Hk != Hq or g > MAXG or k.dtype != (m.int8 if scales else m.bfloat16):
        raise ValueError(
            f"[aluminum] prefill: head {HEADS}, Hq/Hk <= {MAXG}, a bf16 cache or an int8 one with its scales; got "
            f"{q.shape} over {k.shape} {k.dtype}"
        )
    paged: list[mx_.array] = []
    if rmap is not None:
        if k2 is None or v2 is None:
            raise ValueError("[aluminum] prefill: a row map takes the chunk's own rows in k2/v2")
        if any(pb or kb for _, _, pb, _, kb in chunks):
            raise ValueError("[aluminum] prefill: a row map reads the region's rows, at no slice or offset of its own")
        if scales:
            raise ValueError("[aluminum] prefill: a paged cache's rows are bf16")
        paged = [rmap, k2, v2]
    win = int(window or 0)
    flat = [x for T, past, pb, qo, kb in chunks for x in (T, past, win, past + T, pb, qo, kb)]
    meta = _const(("pf", *flat), flat, m.int32)
    sc = _const(("sc", float(scale)), [float(scale)], m.float32)
    odt = odt if odt is not None else q.dtype
    heads = Hk if D == 256 else Hq
    sgs = 8 if D == 256 else 4
    nqb = max(-(-(T * g if D == 256 else T) // 32) for T, *_ in chunks)
    return _prefill_kernel(D, g, sinks is not None, bool(paged), bool(scales))(
        inputs=[q, k, v, meta, sc, *scales, *([sinks] if sinks is not None else []), *paged],
        template=[("OT", odt)],
        grid=(nqb * 32, heads * sgs, len(chunks)),
        threadgroup=(32, sgs, 1),
        output_shapes=[(Tq, Hq, D)],
        output_dtypes=[odt],
    )[0]


TG_MEMORY = 32768


def tgm_layout(D: int, gh: int, ch: int) -> tuple[int, int]:
    """The segment function's threadgroup memory for gh query heads and a V stage of ch keys: (the stage's offset, the
    bytes) - the queries, the scores, the blocks' maxima, theirs moved, factors and sums, the max and sum a head, then
    V's stage 16-byte aligned"""
    nb = SEGK // block(D)
    vt = -(-4 * gh * (D + SEGK + 4 * nb + 2) // 16) * 16
    return vt, vt + ch * (D + 8) * 2


def decode_shape(D: int, g: int) -> tuple[int, int]:
    """(query heads a threadgroup, V's keys a stage) at head size D and g query heads a kv head: the largest group
    whose state and a stage of a block fit a threadgroup's memory, then the longest stage that fits"""
    for gh in (g, g // 2, g // 4, 1):
        if gh < 1 or g % gh:
            continue
        stages = [c for c in (128, 64, 32, 16) if c >= block(D) and tgm_layout(D, gh, c)[1] <= TG_MEMORY]
        if stages:
            return gh, stages[0]
    raise ValueError(f"[aluminum] no decode shape for head {D}, {g} query heads a kv head")


# a decode grid of this many threadgroups or more fills the GPU (more where a threadgroup's heads span past 512
# columns: `busy`); its threadgroups then the lighter `busy_shape`
BUSY = 64


def busy(D: int, g: int, Hk: int, T: int, splits: int) -> bool:
    """whether a decode grid of T nodes over `splits` segments takes `busy_shape`: BUSY threadgroups for heads of up
    to 512 columns, proportionally more past them (the busy shape's 256 threads each scoring more). Measured (M3 Pro,
    cold caches): head 256 with 4 query heads a threadgroup x1.02-1.06 at 64 threadgroups, x0.90-0.96 at 128"""
    gh = decode_shape(D, g)[0]
    return T * Hk * (g // gh) * splits * 512 >= BUSY * max(512, gh * D)


def busy_shape(D: int, g: int) -> tuple[int, int, int]:
    """(query heads a threadgroup, V's keys a stage, threads) for a `busy` decode grid: 256 threads and the longest
    stage within half a threadgroup's memory, else a block. Measured (M3 Pro, cold caches) x0.65-0.95 of
    `decode_shape`'s on busy grids at heads 64 / 96 / 128 / 256, x1.0-1.2 on grids of 32 threadgroups or fewer"""
    gh, _ = decode_shape(D, g)
    fits = [c for c in (128, 64, 32, 16) if c >= block(D) and tgm_layout(D, gh, c)[1] <= TG_MEMORY // 2]
    return gh, fits[0] if fits else block(D), 256


def tree_rows(D: int, nsg: int) -> int:
    """the tree tile's rows at head size D on nsg simdgroups: 8 a simdgroup (below 256), 8 a pair of them (256)"""
    return 4 * nsg if D == 256 else 8 * nsg


def tree_tgm(D: int, nsg: int) -> int:
    """the tree tile's threadgroup memory at head size D on nsg simdgroups: Q's rows and the K / V stage (below 256),
    K's and V's stages and the halves' exchange (256)"""
    bk = block(D)
    if D == 256:
        return 2 * bk * (D + 8) * 2 + nsg * bk * 8 * 4
    return (8 * nsg * (D + 8) + max(D * (bk + 8), bk * (D + 8))) * 2


def phased_tgm(rows: int) -> int:
    """head 256's phased kernel's threadgroup memory at `rows` rows: a pass's scores (a segment's keys over rows / 8
    passes), its blocks' maxima, theirs moved, factors and sums, a row's max, sum and keys, then Q's rows (bf16)"""
    sub = SEGK * 8 // rows
    return 4 * rows * (sub + 4 * (sub // 16) + 4) + 2 * rows * (256 + 8)


# the tree tile's simdgroups in its own launch (the megakernel's threadgroups are 8 at every head size)
TREE_NSG = {D: 8 if D == 256 else 4 for D in HEADS}


class TreeKind(StrEnum):
    """a tree pass's attention kernel: node by node (the decode's), the tile, or head 256's phased kernel"""

    NODES = "nodes"
    TILE = "tile"
    PHASED = "phased"


class TreePick(NamedTuple):
    """a tree kernel and its rows and simdgroups a threadgroup (node by node: neither)"""

    kind: TreeKind
    rows: int = 0
    nsg: int = 0


NODE_BY_NODE = TreePick(TreeKind.NODES)


def tree_kernels(D: int, nsg: int = 0) -> tuple[TreePick, ...]:
    """every tree kernel at head size D on threadgroups of nsg simdgroups (0: each launch's own), the tile first: the
    tile, and at head 256 the phased kernel at 8 and 16 rows (its own launch's on 8 or 16 simdgroups)"""
    n = nsg or TREE_NSG[D]
    tile = TreePick(TreeKind.TILE, tree_rows(D, n), n)
    if D != 256:
        return (tile,)
    return (tile, *(TreePick(TreeKind.PHASED, r, s) for r in (8, 16) for s in ((nsg,) if nsg else (8, 16))))


# head 256's dispatch (`tree_pick`): a phased grid below PHASED_SMALL threadgroups takes 16 simdgroups (its own
# launch's); from PHASED_MORE threadgroups more rows where they halve the grid, so the K / V re-reads (16 rows only
# where the K / V the grid reads reaches PHASED_KV segments of a kv head: below it the re-reads stay in cache); the
# megakernel's phased grid below MEGA_NODES threadgroups goes node by node (on its 8 simdgroups 2.9% from the fastest)
PHASED_SMALL = 24
PHASED_MORE = 48
PHASED_KV = 16
MEGA_NODES = 12


def tree_pick(D: int, g: int, Hk: int, T: int, splits: int, nsg: int = 0) -> TreePick:
    """the kernel a tree of T nodes over `splits` segments takes on threadgroups of nsg simdgroups (0: its own
    launch's choice): below head 256 the tile once its (node, kv head, segment) items times sqrt(g) reach 256; at 256
    the phased kernel, more rows or the tile past PHASED_MORE. Fitted (M3 Pro, cold caches) on trees of 2 .. 64 nodes
    over 300 .. 8000 rows: from the fastest kernel 0.4% on geometric mean below 256 (24% worst), 0.55% at 256 (11%)"""
    if T < 2:
        return NODE_BY_NODE
    if D != 256:
        n = nsg or TREE_NSG[D]
        return TreePick(TreeKind.TILE, tree_rows(D, n), n) if T * splits * Hk * math.sqrt(g) >= 256 else NODE_BY_NODE
    g8, g16, g32 = (-(-(T * g) // r) * Hk * splits for r in (8, 16, 32))
    if nsg and g8 < MEGA_NODES:
        return NODE_BY_NODE
    rows, grid = 8, g8
    if g8 >= PHASED_MORE and g16 < g8 and Hk * splits >= PHASED_KV:
        if g16 >= PHASED_MORE and g32 < g16:
            return TreePick(TreeKind.TILE, 32, 8)
        rows, grid = 16, g16
    return TreePick(TreeKind.PHASED, rows, nsg or (16 if grid < PHASED_SMALL else 8))


def device_header(D: int, kq: bool = False, seg: bool = False, mapped: bool = False, sinks: bool = False) -> str:
    """The segment, tree and fold device functions at head size D, for a launch's header (the megakernel's among
    them)"""
    return _defs(D, KQ=kq, SEG=seg, MAP=mapped, SINKS=sinks, BTB_BK=block(D)) + _SCORE + _SEGMENT_FN + _TREE_FN


def _alu_names(kq: bool, seg: bool, mapped: bool, sinks: bool, own: Sequence[str]) -> list[str]:
    """a decode or tree launch's inputs: q, the cache, the nodes' meta and path, its `own`, then its variant's"""
    return (
        ["q", "k", "v", "meta", "path", *own]
        + (["ks", "vs"] if kq else [])
        + (["k2", "v2"] if seg else [])
        + (["ks2", "vs2"] if kq and seg else [])
        + (["rmap"] if mapped else [])
        + (["sinks"] if sinks else [])
    )


def _alu_in(kq: bool, seg: bool, mapped: bool, sinks: bool, nsx: str) -> str:
    """the launch's `BtbAluIn` a over its inputs (`_alu_names`), the partials' segments a row `nsx`"""
    kt = "const device char*" if kq else "const device bfloat*"
    types = [
        "decltype(&q[0])",
        kt,
        "decltype(&meta[0])",
        "decltype(&path[0])",
        "decltype(&ks[0])" if kq else "const device float*",
        "decltype(&ks2[0])" if kq and seg else "const device float*",
        "decltype(&rmap[0])" if mapped else "const device uint*",
        "decltype(&sinks[0])" if sinks else "const device float*",
    ]
    fields = [
        "q",
        f"({kt})k",
        f"({kt})v",
        f"({kt})k2" if seg else "nullptr",
        f"({kt})v2" if seg else "nullptr",
        "meta",
        "path",
        "ks" if kq else "nullptr",
        "vs" if kq else "nullptr",
        "ks2" if kq and seg else "nullptr",
        "vs2" if kq and seg else "nullptr",
        "rmap" if mapped else "nullptr",
        "sinks" if sinks else "nullptr",
        "sc[0]",
        "(int)q_shape[1]",
        "(int)k_shape[1]",
        "(int)k_shape[2]",
        "(int)path_shape[1]",
        "(int)k2_shape[2]" if seg else "0",
        nsx,
    ]
    return f"    BtbAluIn<{', '.join(types)}> a = {{{', '.join(fields)}}};\n"


def _variant(kq: bool, seg: bool, mapped: bool, sinks: bool) -> str:
    """a decode or tree kernel's name past its kind: its rows' type and its inputs"""
    return f"{'i8' if kq else 'bf16'}{'_seg' if seg else ''}{'_map' if mapped else ''}{'_sinks' if sinks else ''}"


def _decode_kernel(
    D: int, g: int, kq: bool, seg: bool, mapped: bool, sinks: bool, busy: bool
) -> tuple[MetalKernel, int, int]:
    key = ("decode", D, g, kq, seg, mapped, sinks, busy)
    if busy:
        gh, ch, nth = busy_shape(D, g)
    else:
        gh, ch = decode_shape(D, g)
        # measured (M3 Pro): 256 threads, 512 where the heads' columns pass 256 (the scores' keys a thread halved)
        nth = 256 if gh * D <= 256 else 512
    with _gemv_lock:
        ks = _kernels.get(key)
        if ks is None:
            vt, tgm = tgm_layout(D, gh, ch)
            src = f"""
    threadgroup uchar tgm[{tgm}];
{_alu_in(kq, seg, mapped, sinks, "(int)threadgroups_per_grid.y")}
    btb_alu_segment<{D}, {g}, {gh}, {nth}, {ch}, {vt}>(a, threadgroup_position_in_grid.x,
        threadgroup_position_in_grid.y, threadgroup_position_in_grid.z, thread_position_in_threadgroup.x, tgm, out, pm,
        pl, po);
"""
            ks = _kernels[key] = metal_kernel(
                name=f"btb_aluminum_decode_{_variant(kq, seg, mapped, sinks)}{'_busy' if busy else ''}_d{D}_g{g}",
                input_names=_alu_names(kq, seg, mapped, sinks, ["sc"]),
                output_names=["out", "pm", "pl", "po"],
                header=device_header(D, kq, seg, mapped, sinks),
                source=src,
            )
    return ks, gh, nth


def _tree_kernel(D: int, g: int, kq: bool, seg: bool, mapped: bool, sinks: bool, rows: int) -> MetalKernel:
    key = ("tree", D, g, kq, seg, mapped, sinks, rows)
    nsg = TREE_NSG[D]
    with _gemv_lock:
        kern = _kernels.get(key)
        if kern is None:
            fn = f"{'256' if D == 256 else ''}<{D}, {g}, {nsg}>"
            src = f"""
    threadgroup uchar tgm[{tree_tgm(D, nsg)}];
{_alu_in(kq, seg, mapped, sinks, "ns[0]")}
    btb_alu_tree{fn}(a, (int)q_shape[0], threadgroup_position_in_grid.x, threadgroup_position_in_grid.y,
        threadgroup_position_in_grid.z, thread_position_in_threadgroup.x, tgm, out, pm, pl, po);
"""
            kern = _kernels[key] = metal_kernel(
                name=f"btb_aluminum_tree_{_variant(kq, seg, mapped, sinks)}_r{rows}_d{D}_g{g}",
                input_names=_alu_names(kq, seg, mapped, sinks, ["sc", "ns"]),
                output_names=["out", "pm", "pl", "po"],
                header=device_header(D, kq, seg, mapped, sinks),
                source=src,
            )
    return kern


def _phased_kernel(g: int, kq: bool, seg: bool, mapped: bool, sinks: bool, rows: int, nsg: int) -> MetalKernel:
    key = ("phased", g, kq, seg, mapped, sinks, rows, nsg)
    with _gemv_lock:
        kern = _kernels.get(key)
        if kern is None:
            src = f"""
    threadgroup uchar tgm[{phased_tgm(rows)}];
{_alu_in(kq, seg, mapped, sinks, "ns[0]")}
    btb_alu_phased256<256, {g}, {nsg}, {rows}>(a, (int)q_shape[0], threadgroup_position_in_grid.x,
        threadgroup_position_in_grid.y, threadgroup_position_in_grid.z, thread_position_in_threadgroup.x, tgm, out, pm,
        pl, po);
"""
            kern = _kernels[key] = metal_kernel(
                name=f"btb_aluminum_phased_{_variant(kq, seg, mapped, sinks)}_r{rows}_n{nsg}_d256_g{g}",
                input_names=_alu_names(kq, seg, mapped, sinks, ["sc", "ns"]),
                output_names=["out", "pm", "pl", "po"],
                header=device_header(256, kq, seg, mapped, sinks),
                source=src,
            )
    return kern


def _fold_kernel(D: int, sinks: bool) -> MetalKernel:
    key = ("fold", D, sinks)
    with _gemv_lock:
        fold = _kernels.get(key)
        if fold is None:
            sk = "sinks" if sinks else "(const device float*)nullptr"
            fold = _kernels[key] = metal_kernel(
                name=f"btb_aluminum_fold_d{D}{'_sinks' if sinks else ''}",
                input_names=["pm", "pl", "po", "meta", "so", *(["sinks"] if sinks else [])],
                output_names=["out"],
                header=device_header(D, sinks=sinks),
                source=f"""
    btb_alu_fold<{D}>(meta, {sk}, pm, pl, po, so, (int)so_shape[0], (int)so_shape[1], (int)pm_shape[1],
                     thread_position_in_grid.x, out);
""",
            )
    return fold


def nodes(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    meta: mx_.array,
    path: mx_.array,
    scale: float,
    splits: int,
    sinks: mx_.array | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    ks2: mx_.array | None = None,
    vs2: mx_.array | None = None,
    odt: mx_.Dtype | None = None,
    rmap: mx_.array | None = None,
    shared: bool = False,
) -> mx_.array:
    """Attention for T query nodes, each over its own keys as the prefill computes that row: `q` [T, Hq, D] (taken as
    bf16), `k`/`v` the cache [B, Hk, cap, D] (bf16; int8 with `ks`/`vs` [B, Hk, cap] a scale a row, read as the
    dequantized bf16), `meta` uint32 [T, 8] and `path` uint32 [T, PD] as attn.py's `tree_meta`/`rows_meta` make them,
    `splits` the segments the nodes span. `k2`/`v2` the rows at or past a node's `seg`, `rmap` a paged cache's row
    map, `sinks` [Hq] float32; `shared` nodes over one sequence's rows (`tree_meta`'s) go to the tree kernel
    `tree_pick` picks. Returns [T, Hq, D] in `odt` (float32 by default), lazily"""
    T, Hq, D = (int(x) for x in q.shape)
    Hk = int(k.shape[1])
    pick = tree_pick(D, Hq // max(Hk, 1), Hk, T, int(splits)) if shared else NODE_BY_NODE
    return _nodes(q, k, v, meta, path, scale, splits, sinks, ks, vs, k2, v2, ks2, vs2, odt, rmap, pick)


def tree(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    meta: mx_.array,
    path: mx_.array,
    scale: float,
    splits: int,
    sinks: mx_.array | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    ks2: mx_.array | None = None,
    vs2: mx_.array | None = None,
    odt: mx_.Dtype | None = None,
    rmap: mx_.array | None = None,
    pick: TreePick | None = None,
) -> mx_.array:
    """`nodes` for one sequence's nodes (`tree_meta`'s) on the tree kernel `pick` (one of `tree_kernels`, by default
    the tile), however many: the same bits"""
    pick = pick if pick is not None else tree_kernels(int(q.shape[-1]))[0]
    return _nodes(q, k, v, meta, path, scale, splits, sinks, ks, vs, k2, v2, ks2, vs2, odt, rmap, pick)


def _nodes(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    meta: mx_.array,
    path: mx_.array,
    scale: float,
    splits: int,
    sinks: mx_.array | None,
    ks: mx_.array | None,
    vs: mx_.array | None,
    k2: mx_.array | None,
    v2: mx_.array | None,
    ks2: mx_.array | None,
    vs2: mx_.array | None,
    odt: mx_.Dtype | None,
    rmap: mx_.array | None,
    pick: TreePick,
) -> mx_.array:
    m = mx
    T, Hq, D = (int(x) for x in q.shape)
    Hk = int(k.shape[1])
    g = Hq // Hk
    if D not in HEADS or g * Hk != Hq or g > MAXG:
        raise ValueError(f"[aluminum] nodes: head {HEADS} and Hq/Hk <= {MAXG}; got {q.shape} over {k.shape}")
    # the launch's inputs past its own, as the kernel's variant names them
    extra: list[mx_.array] = []
    kq = seg = False
    if ks is not None and vs is not None:
        kq = True
        extra += [ks, vs]
    if k2 is not None and v2 is not None:
        seg = True
        extra += [k2, v2]
    if kq and seg:
        if ks2 is None or vs2 is None:
            raise ValueError("[aluminum] an int8 cache's rows in k2/v2 take their scales in ks2/vs2")
        extra += [ks2, vs2]
    if rmap is not None:
        if kq:
            raise ValueError("[aluminum] a paged cache's rows are bf16")
        extra.append(rmap)
    if sinks is not None:
        extra.append(sinks)
    sc = _const(("sc", float(scale)), [float(scale)], m.float32)
    odt = odt if odt is not None else m.float32
    splits = max(1, int(splits))
    shapes = [(T, Hq, D), (T * Hq, splits), (T * Hq, splits), (T * Hq, splits, D)]
    dts = [odt, m.float32, m.float32, m.float32]
    mapped, sk = rmap is not None, sinks is not None
    if pick.kind is not TreeKind.NODES:
        if pick not in tree_kernels(D):
            raise ValueError(f"[aluminum] no tree kernel {pick} at head {D}: {tree_kernels(D)}")
        kern = (
            _phased_kernel(g, kq, seg, mapped, sk, pick.rows, pick.nsg)
            if pick.kind is TreeKind.PHASED
            else _tree_kernel(D, g, kq, seg, mapped, sk, pick.rows)
        )
        ns = _const(("ns", splits), [splits], m.int32)
        out, pm, pl, po = kern(
            inputs=[q, k, v, meta, path, sc, ns, *extra],
            template=[("OT", odt)],
            grid=(-(-(T * g) // pick.rows) * 32 * pick.nsg, Hk, splits),
            threadgroup=(32 * pick.nsg, 1, 1),
            output_shapes=shapes,
            output_dtypes=dts,
        )
    else:
        many = busy(D, g, Hk, T, splits)
        kern, gh, nth = _decode_kernel(D, g, kq, seg, mapped, sk, many)
        out, pm, pl, po = kern(
            inputs=[q, k, v, meta, path, sc, *extra],
            template=[("OT", odt)],
            grid=(Hk * (g // gh) * nth, splits, T),
            threadgroup=(nth, 1, 1),
            output_shapes=shapes,
            output_dtypes=dts,
        )
    if splits == 1:
        return out
    return _fold_kernel(D, sinks is not None)(
        inputs=[pm, pl, po, meta, out, *([sinks] if sinks is not None else [])],
        template=[("OT", odt)],
        grid=(-(-(T * Hq * D) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, Hq, D)],
        output_dtypes=[odt],
    )[0]
