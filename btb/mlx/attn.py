# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The node attention kernel (T query nodes over a cache, each along its own path: a tree's verify pass, a batch
of rows, a forest of prompts; bit-exact to the one-row step), rope for rows at their positions, the int8 cache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..kinds import Parents
from .core import mx
from .gemv import _gemv_lock

if TYPE_CHECKING:
    import mlx.core as mx_
ATTN_MAXG = 8


# nodes of a pass per threadgroup, sharing each prefix row's read: measured slower above 1 on an M3 Pro (the
# kernel is latency-bound per threadgroup, not read-bound; the per-node registers cost more than the reads save)
ATTN_NODES = 1


# rows per block of the cache: 1024 reads 4000 rows in 0.11 ms a layer against 0.18 at 256 (fewer partials to fold)
ATTN_BLOCK = 1024


_ATTN_SRC = """
    // one threadgroup per (kv head, block of P rows, group of NT query nodes). The cache's rows are blocks of P
    // from row 0: row j belongs to block j / P, simdgroup (j / 32) % 8 of it, lane j % 32 - a function of j
    // alone, so a node folds its rows in one order whatever its length, its window or its company, and a tree
    // node is computed bit for bit as the one-row step at that position (the fold kernel then folds a node's
    // blocks x 8 partials in one fixed order). A lane reads its row of K once for every node of the group that
    // shares it (the prefix; a tree row is a node's own) and takes the G dot products per node in registers, q
    // from threadgroup memory; a block of 32 rows folds its softmax per node with one simd_max and one simd_sum
    // per query head, and the P.V product spreads the D dims over the lanes (D / 32 each), each row's p handed
    // round by one shuffle. Node t attends over the logical rows [start, rows) of `meta[t] = (rows, blocks,
    // past, start, batch, seg, pbase, kbatch)` in cache slice `batch` (a batched decode: one node per row, each
    // its own slice and row count), where logical row j < past is cache row j and logical row past + d is cache
    // row past + path[t][d] (the d-th node on its root-to-self path, the tree's rows sitting after the prefix).
    // `start` is the sliding window's first row, max(0, rows - W) (0 for a full-attention layer); the node's
    // blocks are start / P .. (rows - 1) / P, `blocks` of them, block s of the grid the s-th.
    const uint G = G_VALUE;
    const uint D = D_VALUE;
    const uint NT = NT_VALUE;
    const uint P = P_VALUE;
    const uint DL = D / 32;
    const uint KC = D / 8;
    uint h = threadgroup_position_in_grid.x;
    uint s = threadgroup_position_in_grid.y;
    uint grp = threadgroup_position_in_grid.z;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    // k, v: the cache's own buffers [1, Hk, cap, D] bf16, read as uint16 bit patterns
    uint Hk = k_shape[1];
    uint cap = k_shape[2];
    uint Hq = q_shape[1];
    uint T = q_shape[0];
    uint ns_max = threadgroups_per_grid.y;
    uint PD = path_shape[1];
    const device uint16_t* kb16 = (const device uint16_t*)k;
    const device uint16_t* vb16 = (const device uint16_t*)v;
    // the group's nodes: each its rows in this block, [lo, hi), and its K/V rows' bases. `kbidx` and `pbase`
    // place a node's prefill rows in k: slice kbidx, from row pbase (a flat buffer holding every row's prompt
    // end to end has kbidx 0 and the row's offset as pbase; per-row slices have kbidx = the row, pbase 0)
    uint node[NT], past_[NT], lo[NT], hi[NT];
    bool has[NT];
    const device uint4* kh[NT];
    const device uint16_t* vhb[NT];
#if KQ
    // an int8 cache: k, v are [B, Hk, cap, D] int8 and ks, vs [B, Hk, cap] float32, one scale per row; the
    // scales are read by index through their own names (a small input may arrive in the constant address
    // space, from which no device pointer can be taken)
    const device char* kq[NT];
    const device char* vq[NT];
    size_t ksb[NT];
#endif
#if SEG
    // a two-segment cache (the batched decode): logical rows [0, seg) are the row's prefill in k/v, rows
    // [seg, rows) its decode steps in k2/v2 at (row - seg). Every row's step t lands at slot t of k2, so the
    // batch appends with one in-place write; the logical order, so the lanes and the fold, is the single
    // decode's, and so are the bits
    uint seg[NT];
    uint cap2 = k2_shape[2];
    const device uint16_t* kb2[NT];
    const device uint16_t* vb2[NT];
#if KQ
    const device char* kq2[NT];
    const device char* vq2[NT];
    size_t ksb2[NT];
#endif
#endif
    uint lo_min = 0xffffffffu, hi_max = 0;
    bool any = false;
    for (uint t = 0; t < NT; ++t) {
        node[t] = grp * NT + t;
        has[t] = false;
        lo[t] = 0;
        hi[t] = 0;
        past_[t] = 0;
        kh[t] = (const device uint4*)kb16;
        vhb[t] = vb16;
#if KQ
        kq[t] = (const device char*)k;
        vq[t] = (const device char*)v;
        ksb[t] = 0;
#endif
#if SEG
        seg[t] = 0;
        kb2[t] = (const device uint16_t*)k2;
        vb2[t] = (const device uint16_t*)v2;
#if KQ
        kq2[t] = (const device char*)k2;
        vq2[t] = (const device char*)v2;
        ksb2[t] = 0;
#endif
#endif
        if (node[t] < T) {
            const device uint* mt = meta + node[t] * 8;
            uint nrows = mt[0];
            uint nb = mt[1];
            past_[t] = mt[2];
            uint start = mt[3];
            uint bidx = mt[4];
            uint pbase = mt[6];
            uint kbidx = mt[7];
            if (s < nb) {
                uint origin = (start / P + s) * P;
                lo[t] = max(start, origin);
                hi[t] = min(nrows, origin + P);
                has[t] = lo[t] < hi[t];
            }
            if (has[t]) {
                any = true;
                lo_min = min(lo_min, lo[t]);
                hi_max = max(hi_max, hi[t]);
                size_t kb = (((size_t)kbidx * Hk + h) * cap + pbase) * D;
                kh[t] = (const device uint4*)(kb16 + kb);
                vhb[t] = vb16 + kb;
#if KQ
                kq[t] = (const device char*)k + kb;
                vq[t] = (const device char*)v + kb;
                ksb[t] = ((size_t)kbidx * Hk + h) * cap + pbase;
#endif
#if SEG
                seg[t] = mt[5];
                size_t b2 = ((size_t)bidx * Hk + h) * cap2 * D;
                kb2[t] = (const device uint16_t*)k2 + b2;
                vb2[t] = (const device uint16_t*)v2 + b2;
#if KQ
                kq2[t] = (const device char*)k2 + b2;
                vq2[t] = (const device char*)v2 + b2;
                ksb2[t] = ((size_t)bidx * Hk + h) * cap2;
#endif
#endif
            }
        }
    }
    if (!any) return;
    threadgroup float4 qs[NT][G][D / 4];
    uint tid = sg * 32 + lane;
    for (uint e = tid; e < NT * G * (D / 4); e += 256) {
        uint t = e / (G * (D / 4));
        uint rem = e - t * G * (D / 4);
        uint i = rem / (D / 4);
        uint c = rem - i * (D / 4);
        float4 qv = float4(0.0f);
        if (has[t]) {
            // q in its own dtype, scaled here (the float32 multiply the caller used to do as an op of its own)
            const device QT* qi = q + ((size_t)node[t] * Hq + h * G + i) * D + c * 4;
            qv = float4((float)qi[0], (float)qi[1], (float)qi[2], (float)qi[3]) * sc[0];
        }
        qs[t][i][c] = qv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m[NT][G];
    float l[NT][G];
    float acc[NT][G][DL];
    for (uint t = 0; t < NT; ++t) {
        for (uint i = 0; i < G; ++i) {
            m[t][i] = -INFINITY;
            l[t][i] = 0.0f;
            for (uint w = 0; w < DL; ++w) acc[t][i][w] = 0.0f;
        }
    }
    for (uint b0 = (lo_min & ~31u) + sg * 32; b0 < hi_max; b0 += 256) {
        uint j = b0 + lane;
        float sd[NT][G];
        bool live[NT];
        // this lane's K row per node: a prefix row of one slice is one address for every node, read once
#if KQ
        const device char* kqr[NT];
        float ksc[NT];
#else
        const device uint4* kr[NT];
#endif
        for (uint t = 0; t < NT; ++t) {
            live[t] = has[t] && j >= lo[t] && j < hi[t];
            for (uint i = 0; i < G; ++i) sd[t][i] = live[t] ? 0.0f : -INFINITY;
#if KQ
            kqr[t] = kq[0];
            ksc[t] = 0.0f;
#else
            kr[t] = kh[0];
#endif
            if (live[t]) {
                uint pj = j < past_[t] ? j : past_[t] + path[(size_t)node[t] * PD + (j - past_[t])];
#if KQ
                // the row's scale multiplies the dot product once: sum(q . (s * kq)) = s * sum(q . kq), and
                // the int8 values are exact in float32, so the product is the dequantized row's to rounding
                const device char* kq_ = kq[t];
                float sc_;
#if SEG
                if (pj >= seg[t]) { kq_ = kq2[t]; pj -= seg[t]; sc_ = ks2[ksb2[t] + pj]; } else { sc_ = ks[ksb[t] + pj]; }
#else
                sc_ = ks[ksb[t] + pj];
#endif
                kqr[t] = kq_ + (size_t)pj * D;
                ksc[t] = sc_;
#else
                const device uint4* kh_ = kh[t];
#if SEG
                if (pj >= seg[t]) { kh_ = (const device uint4*)kb2[t]; pj -= seg[t]; }
#endif
                kr[t] = kh_ + (size_t)pj * KC;
#endif
            }
        }
#if KQ
        for (uint c = 0; c < KC; ++c) {
            char4 a0 = char4(0);
            char4 b0_ = char4(0);
            for (uint t = 0; t < NT; ++t) {
                if (!live[t]) continue;
                char4 a, b;
                if (t > 0 && live[0] && kqr[t] == kqr[0]) {
                    a = a0;
                    b = b0_;
                } else {
                    const device char4* kr4 = (const device char4*)kqr[t];
                    a = kr4[2 * c];
                    b = kr4[2 * c + 1];
                }
                if (t == 0) { a0 = a; b0_ = b; }
                for (uint i = 0; i < G; ++i) {
                    float4 qa = qs[t][i][2 * c];
                    float4 qb = qs[t][i][2 * c + 1];
                    sd[t][i] += qa.x * (float)a.x + qa.y * (float)a.y + qa.z * (float)a.z + qa.w * (float)a.w
                              + qb.x * (float)b.x + qb.y * (float)b.y + qb.z * (float)b.z + qb.w * (float)b.w;
                }
            }
        }
        for (uint t = 0; t < NT; ++t) {
            if (live[t]) for (uint i = 0; i < G; ++i) sd[t][i] *= ksc[t];
        }
#else
        for (uint c = 0; c < KC; ++c) {
            uint4 pk0 = uint4(0u);
            for (uint t = 0; t < NT; ++t) {
                if (!live[t]) continue;
                uint4 pk = (t > 0 && live[0] && kr[t] == kr[0]) ? pk0 : kr[t][c];
                if (t == 0) pk0 = pk;
                float k0 = as_type<float>(pk.x << 16), k1 = as_type<float>(pk.x & 0xffff0000u);
                float k2_ = as_type<float>(pk.y << 16), k3 = as_type<float>(pk.y & 0xffff0000u);
                float k4 = as_type<float>(pk.z << 16), k5 = as_type<float>(pk.z & 0xffff0000u);
                float k6 = as_type<float>(pk.w << 16), k7 = as_type<float>(pk.w & 0xffff0000u);
                for (uint i = 0; i < G; ++i) {
                    float4 qa = qs[t][i][2 * c];
                    float4 qb = qs[t][i][2 * c + 1];
                    sd[t][i] += qa.x * k0 + qa.y * k1 + qa.z * k2_ + qa.w * k3
                              + qb.x * k4 + qb.y * k5 + qb.z * k6 + qb.w * k7;
                }
            }
        }
#endif
        for (uint t = 0; t < NT; ++t) {
            if (!has[t]) continue;
            for (uint i = 0; i < G; ++i) {
                float bm = simd_max(sd[t][i]);
                if (bm == -INFINITY) {
                    // none of these 32 rows is the node's: nothing to fold
                    sd[t][i] = 0.0f;
                    continue;
                }
                float mn = max(m[t][i], bm);
                float corr = exp(m[t][i] - mn);
                float p = live[t] ? exp(sd[t][i] - mn) : 0.0f;
                l[t][i] = l[t][i] * corr + simd_sum(p);
                for (uint w = 0; w < DL; ++w) acc[t][i][w] *= corr;
                m[t][i] = mn;
                sd[t][i] = p;
            }
        }
        uint nrow = min(32u, hi_max - b0);
        for (uint r = 0; r < nrow; ++r) {
            uint rr = b0 + r;
            float vv0[DL];
            bool got0 = false;
            const device char* vp0 = (const device char*)vb16;
            for (uint t = 0; t < NT; ++t) {
                if (!has[t] || rr < lo[t] || rr >= hi[t]) continue;
                uint pr = rr < past_[t] ? rr : past_[t] + path[(size_t)node[t] * PD + (rr - past_[t])];
                float vv[DL];
#if KQ
                const device char* vq_ = vq[t];
                float vsc;
#if SEG
                if (pr >= seg[t]) { vq_ = vq2[t]; pr -= seg[t]; vsc = vs2[ksb2[t] + pr]; } else { vsc = vs[ksb[t] + pr]; }
#else
                vsc = vs[ksb[t] + pr];
#endif
                const device char* vp = vq_ + (size_t)pr * D;
                if (t > 0 && got0 && vp == vp0) {
                    for (uint w = 0; w < DL; ++w) vv[w] = vv0[w];
                } else {
                    const device char* vr = vp + lane * DL;
                    for (uint w = 0; w < DL; ++w) vv[w] = (float)vr[w] * vsc;
                }
#else
                const device uint16_t* vhb_ = vhb[t];
#if SEG
                if (pr >= seg[t]) { vhb_ = vb2[t]; pr -= seg[t]; }
#endif
                const device char* vp = (const device char*)(vhb_ + (size_t)pr * D);
                if (t > 0 && got0 && vp == vp0) {
                    for (uint w = 0; w < DL; ++w) vv[w] = vv0[w];
                } else {
                    const device uint* vw = (const device uint*)(vhb_ + (size_t)pr * D + lane * DL);
                    for (uint w = 0; w < DL / 2; ++w) {
                        uint x = vw[w];
                        vv[2 * w] = as_type<float>(x << 16);
                        vv[2 * w + 1] = as_type<float>(x & 0xffff0000u);
                    }
                }
#endif
                if (t == 0) {
                    got0 = true;
                    vp0 = vp;
                    for (uint w = 0; w < DL; ++w) vv0[w] = vv[w];
                }
                for (uint i = 0; i < G; ++i) {
                    float p = simd_shuffle(sd[t][i], r);
                    for (uint w = 0; w < DL; ++w) acc[t][i][w] += p * vv[w];
                }
            }
        }
    }
    // each node's partial from this simdgroup: (node, kv head, block * 8 + simdgroup, query head)
    for (uint t = 0; t < NT; ++t) {
        if (!has[t]) continue;
        size_t pb = ((((size_t)node[t] * Hk + h) * ns_max + s) * 8 + sg) * G;
        for (uint i = 0; i < G; ++i) {
            size_t p = pb + i;
            if (lane == 0) {
                pm[p] = m[t][i];
                pl[p] = l[t][i];
            }
            device float* po = pacc + p * D + lane * DL;
            for (uint w = 0; w < DL; ++w) po[w] = acc[t][i][w];
        }
    }
"""


_ATTN_FOLD_SRC = """
    // one simdgroup per (node, query head): fold that node's splits x 8 simdgroup partials (m, l, acc) in
    // one fixed order; each lane keeps D / 32 dims. With SINKS, the query head's sink logit (gpt-oss: one
    // learned scalar per query head, an extra column of the softmax with no value, taken as it is -- the
    // scores are scaled, the sink is not, as transformers' eager_attention_forward concatenates it) joins
    // the maximum and opens the denominator. It is per query head, so it is the same term whatever the node
    // count or the row count is, and it drops out of the arithmetic entirely when SINKS is 0.
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint j = threadgroup_position_in_grid.x * 8 + sg;
    uint T = pm_shape[0];
    uint Hk = pm_shape[1];
    uint NP = pm_shape[2];
    uint G = pm_shape[3];
    uint D = pacc_shape[4];
    uint DL = D / 32;
    if (j >= T * Hk * G) return;
    uint node = j / (Hk * G);
    uint hq = j - node * Hk * G;
    uint h = hq / G;
    uint i = hq - h * G;
    uint np_ = meta[node * 8 + 1] * 8;
    size_t base = ((size_t)node * Hk + h) * NP;
    float M = -INFINITY;
    float sk = 0.0f;
    if (SINKS) {
        sk = sinks[h * G + i];
        M = sk;
    }
    for (uint s = 0; s < np_; ++s) M = max(M, pm[(base + s) * G + i]);
    float L = SINKS ? exp(sk - M) : 0.0f;
    float o[8];
    for (uint t = 0; t < 8; ++t) o[t] = 0.0f;
    for (uint s = 0; s < np_; ++s) {
        size_t p = (base + s) * G + i;
        float w = exp(pm[p] - M);
        L += w * pl[p];
        const device float* pa = pacc + p * D + lane * DL;
        for (uint t = 0; t < DL; ++t) o[t] += w * pa[t];
    }
    float inv = 1.0f / L;
    size_t ob = (size_t)j * D + lane * DL;
    for (uint t = 0; t < DL; ++t) out[ob + t] = (OT)(o[t] * inv);
"""


_attn_kernels: dict[Any, Any] = {}


_attn_fold = None


_NO_SINKS = None


def attn_splits(n: int, start: int = 0) -> int:
    """how many blocks of ATTN_BLOCK rows a query of logical length `n` over the rows from `start` touches: the
    blocks are fixed stretches of the cache, so a tree node and the one-row step at the same position fold the
    same partials, and the nodes of one pass share a block's rows"""
    return (int(n) + ATTN_BLOCK - 1) // ATTN_BLOCK - int(start) // ATTN_BLOCK


def attn_window_start(n: int, window: int | None = None) -> int:
    """The first row a query of logical length `n` sees under a sliding `window`: max(0, n - window); 0 for full
    attention."""
    n = int(n)
    if not window:
        return 0
    return max(0, n - int(window))


def attn_params(
    n: int,
    window: int | None = None,
    batch: int = 0,
    pbase: int = 0,
    kbatch: int | None = None,
) -> tuple[mx_.array, int]:
    """`attn_decode`'s (meta, splits) for one row over `n` rows, built once per forward (per (n, window) when the
    windows alternate); `batch` the cache slice, `pbase` / `kbatch` the row's place in the k buffer."""
    n = int(n)
    start = attn_window_start(n, window)
    splits = attn_splits(n, start)
    kb = int(batch) if kbatch is None else int(kbatch)
    meta = mx().array([[n, splits, n, start, int(batch), n, int(pbase), kb]], dtype=mx().uint32)
    return meta, splits


_ROPE_ROWS_SRC = """
    // x [B, H, D] (T), freqs [half] float32 as rope_fast passes them (1 / inv_freq), pos [B] uint32: row b
    // rotated at position pos[b]. The arithmetic of MLX's own rope kernel line for line - the reciprocal of
    // the frequency, theta = position * inv_freq in float32, fast cos and sin, the half-split pairs
    // (i, i + half), x1 cos - x2 sin and x1 sin + x2 cos in float32, cast back - so a row here is the bits
    // rope_fast gives the one-row step at that position, and B rows at B positions are one launch.
    uint i = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint hd2 = freqs_shape[0];          // the rotated half (`half` is a Metal type)
    uint H = x_shape[1];
    uint D = x_shape[2];
    if (i >= hd2 || h >= H || b >= (uint)x_shape[0]) return;
    float inv_freq = 1.0f / freqs[i];
    float L = 1.0f * static_cast<float>(pos[b]);
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    size_t i1 = ((size_t)b * H + h) * D + i;
    size_t i2 = i1 + hd2;
    float x1 = static_cast<float>(x[i1]);
    float x2 = static_cast<float>(x[i2]);
    out[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
    out[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
"""


_rope_rows_kernel = None


def rope_rows(x: mx_.array, rd: int, freqs: mx_.array, scaling: float, pos: Sequence[int] | mx_.array) -> mx_.array:
    """`rope_fast` for B rows at B positions in one launch: x [B, heads, d] rotated on its first `rd` dims, row b at
    pos[b]; bit-identical to rope_fast at each position."""
    global _rope_rows_kernel
    m = mx()
    if _rope_rows_kernel is None:
        _rope_rows_kernel = m.fast.metal_kernel(
            name="btb_rope_rows", input_names=["x", "freqs", "pos"], output_names=["out"], source=_ROPE_ROWS_SRC
        )
    B, H, D = (int(s) for s in x.shape)
    rd = int(rd)
    xr = x if rd == D else x[..., :rd]
    p = pos if isinstance(pos, m.array) else m.array([int(v) for v in pos], dtype=m.uint32)
    y = _rope_rows_kernel(
        inputs=[xr, freqs, p],
        template=[("T", x.dtype)],
        grid=(rd // 2, H, B),
        threadgroup=(min(256, rd // 2), 1, 1),
        output_shapes=[(B, H, rd)],
        output_dtypes=[x.dtype],
    )[0]
    if scaling != 1.0:
        y = y * scaling
    return y if rd == D else m.concatenate([y, x[..., rd:]], axis=-1)


def rows_meta(
    ns: Sequence[int], window: int | None = None, segs: Sequence[int] | None = None, pbases: Sequence[int] | None = None
) -> tuple[mx_.array, mx_.array, int]:
    """A batched decode step's node metadata (meta [B, 8], path [B, 1], splits): row b has ns[b] live rows, its
    rows past segs[b] in the step buffer, its prefill in slice b or at pbases[b] of one flat buffer."""
    m = mx()
    B = len(ns)
    meta = np.zeros((B, 8), dtype=np.uint32)
    for b, n in enumerate(ns):
        n = int(n)
        start = attn_window_start(n, window)
        seg = n if segs is None else int(segs[b])
        pb, kb = (0, b) if pbases is None else (int(pbases[b]), 0)
        meta[b] = (n, attn_splits(n, start), n, start, b, seg, pb, kb)
    path = np.zeros((B, 1), dtype=np.uint32)
    return m.array(meta), m.array(path), int(meta[:, 1].max())


def forest_meta(
    lens: Sequence[int], pbases: Sequence[int], window: int | None = None
) -> tuple[mx_.array, mx_.array, int]:
    """A batched prefill's node metadata: token p of a row is a node of p + 1 rows, all in place, row b's from
    pbases[b] of one flat buffer - causal attention as T decode-shaped nodes in one launch."""
    m = mx()
    lens = [int(L) for L in lens]
    T = sum(lens)
    # (rows, splits, past, start, batch, seg, pbase, kbatch) per node, as attn_params would set them
    ns = np.concatenate([np.arange(1, L + 1) for L in lens]) if T else np.zeros(0, dtype=np.int64)
    start = np.where(ns > int(window), ns - int(window), 0) if window else np.zeros_like(ns)
    splits = (ns + ATTN_BLOCK - 1) // ATTN_BLOCK - start // ATTN_BLOCK
    meta = np.zeros((T, 8), dtype=np.uint32)
    meta[:, 0] = meta[:, 2] = meta[:, 5] = ns
    meta[:, 1] = splits
    meta[:, 3] = start
    meta[:, 6] = np.repeat(np.asarray([int(p) for p in pbases], dtype=np.int64), lens)
    path = np.zeros((T, 1), dtype=np.uint32)
    return m.array(meta), m.array(path), int(splits.max()) if T else 1


_ROPE_ROWS2_SRC = """
    // rope_rows for q (x [B, Hq, D]) and k (y [B, Hk, D]) in one launch: heads below Hq are q's, the rest k's;
    // the arithmetic is rope_rows' to the bit
    uint i = thread_position_in_grid.x;
    uint hh = thread_position_in_grid.y;
    uint b = thread_position_in_grid.z;
    uint hd2 = freqs_shape[0];
    uint Hq = x_shape[1];
    uint Hk = y_shape[1];
    uint D = x_shape[2];
    if (i >= hd2 || hh >= Hq + Hk || b >= (uint)x_shape[0]) return;
    float inv_freq = 1.0f / freqs[i];
    float L = 1.0f * static_cast<float>(pos[b]);
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    if (hh < Hq) {
        size_t i1 = ((size_t)b * Hq + hh) * D + i;
        size_t i2 = i1 + hd2;
        float x1 = static_cast<float>(x[i1]);
        float x2 = static_cast<float>(x[i2]);
        ox[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
        ox[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
    } else {
        size_t i1 = ((size_t)b * Hk + (hh - Hq)) * D + i;
        size_t i2 = i1 + hd2;
        float x1 = static_cast<float>(y[i1]);
        float x2 = static_cast<float>(y[i2]);
        oy[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
        oy[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
    }
"""


_rope_rows2_kernel = None


def rope_rows2(
    q: mx_.array, k: mx_.array, rd: int, freqs: mx_.array, scaling: float, pos: Sequence[int] | mx_.array
) -> tuple[mx_.array, mx_.array]:
    """`rope_rows` for q [B, Hq, d] and k [B, Hk, d] in one launch (full rotary dims only; partial dims take
    two `rope_rows` calls). Returns (q rotated, k rotated), the bits of rope_rows."""
    global _rope_rows2_kernel
    m = mx()
    B, Hq, D = (int(s) for s in q.shape)
    if int(rd) != D or q.dtype != k.dtype:
        return rope_rows(q, rd, freqs, scaling, pos), rope_rows(k, rd, freqs, scaling, pos)
    if _rope_rows2_kernel is None:
        _rope_rows2_kernel = m.fast.metal_kernel(
            name="btb_rope_rows2",
            input_names=["x", "y", "freqs", "pos"],
            output_names=["ox", "oy"],
            source=_ROPE_ROWS2_SRC,
        )
    Hk = int(k.shape[1])
    p = pos if isinstance(pos, m.array) else m.array([int(v) for v in pos], dtype=m.uint32)
    ox, oy = _rope_rows2_kernel(
        inputs=[q, k, freqs, p],
        template=[("T", q.dtype)],
        grid=(D // 2, Hq + Hk, B),
        threadgroup=(min(256, D // 2), 1, 1),
        output_shapes=[(B, Hq, D), (B, Hk, D)],
        output_dtypes=[q.dtype, q.dtype],
    )
    if scaling != 1.0:
        ox, oy = ox * scaling, oy * scaling
    return ox, oy


def attn_rows(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    ns: Sequence[int],
    scale: float,
    window: int | None = None,
    sinks: mx_.array | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
    segs: Sequence[int] | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    ks2: mx_.array | None = None,
    vs2: mx_.array | None = None,
    prepared: tuple[mx_.array, mx_.array, int] | None = None,
    odt: Any = None,
    pbases: Sequence[int] | None = None,
) -> mx_.array:
    """A batched decode step: `q[b]` ([B, Hq, D] float32) over cache slice b of ns[b] live rows, each row as the
    one-row step at its length computes it. `segs`: rows past segs[b] come from `k2`/`v2`; `pbases`: the prefill
    rows in one flat buffer; `prepared` = `rows_meta(...)` shared across layers. Returns [B, Hq, D], lazily."""
    meta, path, splits = prepared if prepared is not None else rows_meta(ns, window, segs, pbases)
    return attn_nodes(
        q, k, v, meta, path, scale, splits, sinks=sinks, ks=ks, vs=vs, k2=k2, v2=v2, ks2=ks2, vs2=vs2, odt=odt
    )


def _attn_kernel(D: int, g: int, kq: bool = False, seg: bool = False, nt: int = 1) -> tuple[Any, Any]:
    """the node kernel for head_dim D and g query heads per kv head over groups of `nt` nodes, compiled once per
    shape; `kq` reads an int8 cache with a scale per row, `seg` a two-segment cache (rows past `seg` from k2/v2)"""
    global _attn_fold
    m = mx()
    with _gemv_lock:
        k = _attn_kernels.get((D, g, kq, seg, nt))
        if k is None:
            names = (
                ["q", "k", "v", "meta", "path", "sc"]
                + (["ks", "vs"] if kq else [])
                + (["k2", "v2"] if seg else [])
                + (["ks2", "vs2"] if (kq and seg) else [])
            )
            k = m.fast.metal_kernel(
                name=f"btb_attn_nodes_{'i8' if kq else 'bf16'}{'_seg' if seg else ''}_d{D}_g{g}_n{nt}",
                input_names=names,
                output_names=["pm", "pl", "pacc"],
                header=f"#define KQ {1 if kq else 0}\n#define SEG {1 if seg else 0}\n",
                source=_ATTN_SRC.replace("G_VALUE", str(g))
                .replace("D_VALUE", str(D))
                .replace("NT_VALUE", str(nt))
                .replace("P_VALUE", str(ATTN_BLOCK)),
            )
            _attn_kernels[(D, g, kq, seg, nt)] = k
        if _attn_fold is None:
            _attn_fold = m.fast.metal_kernel(
                name="btb_attn_fold",
                input_names=["pm", "pl", "pacc", "meta", "sinks"],
                output_names=["out"],
                source=_ATTN_FOLD_SRC,
            )
    return k, _attn_fold


_SCALES: dict[Any, Any] = {}


def attn_nodes(
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
    odt: Any = None,
) -> mx_.array:
    """Attention for T query nodes over a cache: `q` [T, Hq, D] float32, `k`/`v` the cache buffers [1, Hk, cap, D]
    as they are, `meta` uint32 [T, 8] per node, `path` uint32 [T, PD] the node's rows as offsets after `past`,
    `sinks` [Hq] float32 an extra softmax column with no value (gpt-oss), `ks`/`vs` an int8 cache's scales.
    float32 throughout; returns [T, Hq, D], lazily. Needs head_dim 64/128/256 and Hq / Hk <= ATTN_MAXG."""
    global _NO_SINKS
    m = mx()
    T, Hq, D = (int(x) for x in q.shape)
    if D not in (64, 128, 256):
        raise ValueError(f"[mlx] attn_nodes takes head_dim 64, 128 or 256, not {D}")
    Hk = int(k.shape[1])
    g = Hq // Hk
    kq = ks is not None
    seg = k2 is not None
    nt = max(1, min(T, ATTN_NODES if g <= 4 else max(1, ATTN_NODES // 2)))
    kern, fold = _attn_kernel(D, g, kq, seg, nt)
    if sinks is None:
        if _NO_SINKS is None:
            _NO_SINKS = m.zeros((1,), dtype=m.float32)
            m.eval(_NO_SINKS)
        sk = _NO_SINKS
    else:
        sk = sinks
    # the scale multiplies q in float32 inside the kernel; q stays in its own dtype and the fold writes `odt`, so the
    # caller casts nothing
    sc = _SCALES.get(float(scale))
    if sc is None:
        sc = _SCALES[float(scale)] = m.array([float(scale)], dtype=m.float32)
        m.eval(sc)
    odt = odt if odt is not None else m.float32
    pm, pl, pacc = kern(
        inputs=(
            [q, k, v, meta, path, sc]
            + ([ks, vs] if kq else [])
            + ([k2, v2] if seg else [])
            + ([ks2, vs2] if (kq and seg) else [])
        ),
        template=[("QT", q.dtype)],
        grid=(Hk * 256, splits, (T + nt - 1) // nt),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, Hk, splits * 8, g), (T, Hk, splits * 8, g), (T, Hk, splits * 8, g, D)],
        output_dtypes=[m.float32, m.float32, m.float32],
    )
    return fold(
        inputs=[pm, pl, pacc, meta, sk],
        template=[("SINKS", 0 if sinks is None else 1), ("OT", odt)],
        grid=(((T * Hq + 7) // 8) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, Hq, D)],
        output_dtypes=[odt],
    )[0]


_NO_PATH = None


# the prefill kernel's tiling: keys a block, simdgroups a threadgroup (four pairs)
ATTN_PF_BLOCK = 16
ATTN_PF_SIMD = 8


_ATTN_PREFILL_SRC = """
    // one threadgroup per (kv head, tile of NS / 2 pairs x RP query positions): the simdgroups in pairs, a pair
    // owning RP = 8 / G positions x G heads = 8 query rows, each half of the pair one half of the head's D (the
    // partial scores meet in threadgroup memory); the cache's K/V rows staged BK at a time for the threadgroup;
    // Q.K^T and P.V as 8x8 matrix multiplies (bf16 in, float32 accumulate); the online softmax on the
    // fragments, the running max moved only when a row's max outgrows it by TAU (p stays below e^TAU)
    const uint G = G_VALUE;
    const uint D = 256;
    const uint HD = 128;
    const uint BK = BK_VALUE;
    const uint NS = NS_VALUE;
    const uint KF = BK / 8;
    const uint HF = HD / 8;
    const uint RP = 8 / G;
    const uint NP = NS / 2;
    const uint LD = D + 8;
    const uint NT = NS * 32;
    const float TAU = 8.0f;
    uint h = threadgroup_position_in_grid.x;
    uint tile = threadgroup_position_in_grid.y;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint pair = sg >> 1;
    uint hv = sg & 1;
    uint tid = sg * 32 + lane;
    uint Hk = k_shape[1];
    uint cap = k_shape[2];
    uint Hq = q_shape[1];
    uint T = q_shape[0];
    uint past = par[0];
    uint p0 = (tile * NP + pair) * RP;
    // the 8x8 fragment's layout: lane holds row fm, columns fn and fn + 1
    uint fm = ((lane >> 1) & 3) + 4 * (lane >> 4);
    uint fn = 2 * (lane & 1) + 4 * ((lane >> 3) & 1);
    // this lane's row: position p0 + fm / G (clamped past the chunk's end, not stored), query head h * G + fm % G
    uint pr = min(p0 + fm / G, T - 1);
    bool store = p0 + fm / G < T;
    size_t qoff = ((size_t)pr * Hq + h * G + fm % G) * D + hv * HD + fn;
    const device QT* qp = q + qoff;
    float scale = sc[0];
    const device bfloat* kh = (const device bfloat*)k + (size_t)h * cap * D;
    const device bfloat* vh = (const device bfloat*)v + (size_t)h * cap * D;
    threadgroup bfloat Ks[BK * LD];
    threadgroup bfloat Vs[BK * LD];
    threadgroup float xs[NS][8 * BK];
    uint jmax = past + min((tile + 1) * NP * RP, T);
    uint jall = past + p0 + 1;
    uint jrow = past + pr + 1;
    float m_ = -INFINITY, l_ = 0.0f;
    simdgroup_float8x8 O[HF];
    #pragma clang loop unroll(full)
    for (uint c = 0; c < HF; ++c) O[c] = simdgroup_float8x8(0.0f);
    for (uint j0 = 0; j0 < jmax; j0 += BK) {
        // the last block is read from cap - BK when the cache ends short of a full block: the rows outside
        // [j0, jrow) are masked either way
        uint jb = min(j0, cap - BK);
        bool masked = (j0 + BK > jall) || (jb != j0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < BK * (D / 8); e += NT) {
            uint r = e / (D / 8), c = (e % (D / 8)) * 8;
            *(threadgroup uint4*)(Ks + r * LD + c) = *(const device uint4*)(kh + (size_t)(jb + r) * D + c);
            *(threadgroup uint4*)(Vs + r * LD + c) = *(const device uint4*)(vh + (size_t)(jb + r) * D + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 S[KF];
        #pragma clang loop unroll(full)
        for (uint f = 0; f < KF; ++f) S[f] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (uint d = 0; d < HF; ++d) {
            simdgroup_bfloat8x8 Qf;
            Qf.thread_elements()[0] = (bfloat)((float)qp[d * 8] * scale);
            Qf.thread_elements()[1] = (bfloat)((float)qp[d * 8 + 1] * scale);
            #pragma clang loop unroll(full)
            for (uint f = 0; f < KF; ++f) {
                simdgroup_bfloat8x8 Kf;
                simdgroup_load(Kf, Ks + f * 8 * LD + hv * HD + d * 8, LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[f], Qf, Kf, S[f]);
            }
        }
        #pragma clang loop unroll(full)
        for (uint f = 0; f < KF; ++f) simdgroup_store(S[f], &xs[sg][f * 8], BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup float* xp = &xs[sg ^ 1][fm * BK + fn];
        float bm = -INFINITY;
        #pragma clang loop unroll(full)
        for (uint f = 0; f < KF; ++f) {
            float s0 = S[f].thread_elements()[0] + xp[f * 8];
            float s1 = S[f].thread_elements()[1] + xp[f * 8 + 1];
            if (masked) {
                uint j = jb + f * 8 + fn;
                if (j < j0 || j >= jrow) s0 = -INFINITY;
                if (j + 1 < j0 || j + 1 >= jrow) s1 = -INFINITY;
            }
            S[f].thread_elements()[0] = s0;
            S[f].thread_elements()[1] = s1;
            bm = max(bm, max(s0, s1));
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        if (simd_any(bm > m_ + TAU)) {
            float mn = max(m_, bm);
            float corr = exp(m_ - mn);
            l_ *= corr;
            m_ = mn;
            #pragma clang loop unroll(full)
            for (uint c = 0; c < HF; ++c) {
                O[c].thread_elements()[0] *= corr;
                O[c].thread_elements()[1] *= corr;
            }
        }
        float ps = 0.0f;
        simdgroup_bfloat8x8 P[KF];
        #pragma clang loop unroll(full)
        for (uint f = 0; f < KF; ++f) {
            bfloat pa = (bfloat)exp(S[f].thread_elements()[0] - m_);
            bfloat pb = (bfloat)exp(S[f].thread_elements()[1] - m_);
            ps += (float)pa + (float)pb;
            P[f].thread_elements()[0] = pa;
            P[f].thread_elements()[1] = pb;
        }
        ps += simd_shuffle_xor(ps, 1);
        ps += simd_shuffle_xor(ps, 8);
        l_ += ps;
        #pragma clang loop unroll(full)
        for (uint f = 0; f < KF; ++f) {
            #pragma clang loop unroll(full)
            for (uint c = 0; c < HF; ++c) {
                simdgroup_bfloat8x8 Vf;
                simdgroup_load(Vf, Vs + f * 8 * LD + hv * HD + c * 8, LD);
                simdgroup_multiply_accumulate(O[c], P[f], Vf, O[c]);
            }
        }
    }
    if (!store) return;
    float inv = 1.0f / l_;
    device OT* op = out + qoff;
    #pragma clang loop unroll(full)
    for (uint c = 0; c < HF; ++c) {
        op[c * 8] = (OT)(O[c].thread_elements()[0] * inv);
        op[c * 8 + 1] = (OT)(O[c].thread_elements()[1] * inv);
    }
"""


_prefill_kernels: dict[int, Any] = {}


def _attn_prefill_kernel(g: int) -> Any:
    """the prefill kernel for g query heads per kv head at head_dim 256, compiled once per g"""
    m = mx()
    with _gemv_lock:
        k = _prefill_kernels.get(g)
        if k is None:
            k = _prefill_kernels[g] = m.fast.metal_kernel(
                name=f"btb_attn_prefill_d256_g{g}",
                input_names=["q", "k", "v", "par", "sc"],
                output_names=["out"],
                source=_ATTN_PREFILL_SRC.replace("G_VALUE", str(g))
                .replace("BK_VALUE", str(ATTN_PF_BLOCK))
                .replace("NS_VALUE", str(ATTN_PF_SIMD)),
            )
    return k


def attn_prefill(q: mx_.array, k: mx_.array, v: mx_.array, past: int, scale: float, odt: Any = None) -> mx_.array:
    """Causal attention for a prefill chunk at head size 256: `q` [T, Hq, 256] (bf16 or float32), `k`/`v` the cache
    buffers [1, Hk, cap, 256] bf16 holding the chunk's rows at [past, past + T); row t attends over cache rows
    [0, past + t]. float32 accumulation (P rounded to bf16 for the value product); returns [T, Hq, 256] in `odt`
    (q's dtype by default), lazily. Needs Hq / Hk in (1, 2, 4, 8) and cap >= ATTN_PF_BLOCK."""
    m = mx()
    T, Hq, D = (int(x) for x in q.shape)
    Hk, cap = int(k.shape[1]), int(k.shape[2])
    g = Hq // Hk
    past = int(past)
    if D != 256 or g * Hk != Hq or 8 % g or k.dtype != m.bfloat16 or cap < ATTN_PF_BLOCK or past + T > cap:
        raise ValueError(
            f"[attn] attn_prefill: head 256, Hq/Hk in (1, 2, 4, 8), a bf16 cache; got {q.shape} over {k.shape}"
        )
    sc = _SCALES.get(float(scale))
    if sc is None:
        sc = _SCALES[float(scale)] = m.array([float(scale)], dtype=m.float32)
        m.eval(sc)
    odt = odt if odt is not None else q.dtype
    rows = ATTN_PF_SIMD // 2 * (8 // g)
    return _attn_prefill_kernel(g)(
        inputs=[q, k, v, m.array([past, T], dtype=m.uint32), sc],
        template=[("QT", q.dtype), ("OT", odt)],
        grid=(Hk * ATTN_PF_SIMD * 32, (T + rows - 1) // rows, 1),
        threadgroup=(ATTN_PF_SIMD * 32, 1, 1),
        output_shapes=[(T, Hq, D)],
        output_dtypes=[odt],
    )[0]


def attn_decode(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    n: int,
    scale: float,
    params: tuple[mx_.array, int] | None = None,
    sinks: mx_.array | None = None,
    window: int | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
    batch: int = 0,
    pbase: int = 0,
    kbatch: int | None = None,
) -> mx_.array:
    """One query row per head over the first `n` rows: `q` [Hq, D] float32 -> [Hq, D] float32, lazily
    (`attn_nodes` over one node); `params` = `attn_params(n, window=...)` when shared across layers."""
    global _NO_PATH
    m = mx()
    if params is None:
        params = attn_params(n, window, batch, pbase, kbatch)
    meta, splits = params
    if _NO_PATH is None:
        _NO_PATH = m.zeros((1, 1), dtype=m.uint32)
        m.eval(_NO_PATH)
    return attn_nodes(q[None], k, v, meta, _NO_PATH, scale, splits, sinks=sinks, ks=ks, vs=vs)[0]


def attn_tree(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    past: int,
    parents: Parents,
    scale: float,
    sinks: mx_.array | None = None,
    window: int | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
    batch: int = 0,
    pbase: int = 0,
    kbatch: int | None = None,
    prepared: tuple[mx_.array, mx_.array, int] | None = None,
) -> mx_.array:
    """`attn_nodes` for a speculative pass: `parents` (-1 for the prefix) gives each node its path, the rows sit
    after `past` in node order; `sinks` and `window` are gpt-oss's; `prepared` = `tree_meta(...)`."""
    meta, path, splits = prepared if prepared is not None else tree_meta(past, parents, window, batch, pbase, kbatch)
    return attn_nodes(q, k, v, meta, path, scale, splits, sinks=sinks, ks=ks, vs=vs)


def tree_meta(
    past: int,
    parents: Parents,
    window: int | None = None,
    batch: int = 0,
    pbase: int = 0,
    kbatch: int | None = None,
) -> tuple[mx_.array, mx_.array, int]:
    """`attn_tree`'s (meta [T, 8], path [T, D], splits), which depend on the tree and the cache's geometry, not on
    the layer: built once per pass."""
    m = mx()
    T = len(parents)
    paths = []
    for t in range(T):
        p, cur = [], t
        while cur >= 0:
            p.append(cur)
            cur = parents[cur]
        paths.append(p[::-1])
    D = max(len(p) for p in paths)
    path = np.zeros((T, D), dtype=np.uint32)
    meta = np.zeros((T, 8), dtype=np.uint32)
    for t, p in enumerate(paths):
        path[t, : len(p)] = p
        n = int(past) + len(p)
        start = attn_window_start(n, window)
        meta[t] = (
            n,
            attn_splits(n, start),
            int(past),
            start,
            int(batch),
            n,
            int(pbase),
            int(batch) if kbatch is None else int(kbatch),
        )
    return m.array(meta), m.array(path), int(meta[:, 1].max())


def kv_quantize(x: mx_.array) -> tuple[mx_.array, mx_.array]:
    """`x` [..., T, D] to int8 with one float32 scale per row (s = max|row| / 127, 1 for a zero row): the
    dequantized row is within s / 2 of x."""
    m = mx()
    xf = x.astype(m.float32)
    s = m.maximum(m.max(m.abs(xf), axis=-1), 1e-30) / 127.0
    q = m.round(xf / s[..., None]).astype(m.int8)
    return q, s


def kv_dequantize(q: mx_.array, s: mx_.array, dtype: Any = None) -> mx_.array:
    """the int8 rows back to `dtype` (bf16 by default): q * s, one scale per row"""
    m = mx()
    return (q.astype(m.float32) * s[..., None]).astype(dtype or m.bfloat16)
