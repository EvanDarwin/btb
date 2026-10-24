# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Attention over aluminum's kernels (`aluminum`): T query nodes over a cache, each along its own path (a tree's verify
pass, a batch of rows, a forest of prompts) and its metadata, a prefill chunk, every row's bits the one-row step's;
rope for rows at their positions, the int8 cache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from ..kinds import Parents
from . import aluminum
from .core import metal_kernel, mx

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

    # what `rows_meta` and `tree_meta` make once for a pass's layers: (meta, path, splits)
    NodeMeta: TypeAlias = tuple[mx_.array, mx_.array, int]
ATTN_MAXG = aluminum.MAXG
ATTN_HEADS = aluminum.HEADS


# a node's keys in segments of this many rows from row 0, aluminum's: a node's meta counts its segments
ATTN_BLOCK = aluminum.SEGK


def attn_splits(n: int, start: int = 0) -> int:
    """how many segments of ATTN_BLOCK rows a query of logical length `n` over the rows from `start` touches: the
    segments are fixed stretches of a row's keys, so a tree node and the one-row step at the same position fold the
    same partials"""
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
    seg: int | None = None,
) -> tuple[mx_.array, int]:
    """`attn_decode`'s (meta, splits) for one row over `n` rows, built once per forward (per (n, window) when the
    windows alternate); `batch` the cache slice, `pbase` / `kbatch` the row's place in the k buffer, `seg` the rows
    before the step's own (in k2/v2 past it: a paged cache's, `n - 1`)."""
    n = int(n)
    start = attn_window_start(n, window)
    splits = attn_splits(n, start)
    kb = int(batch) if kbatch is None else int(kbatch)
    sg = n if seg is None else int(seg)
    meta = mx.array([[n, splits, n, start, int(batch), sg, int(pbase), kb]], dtype=mx.uint32)
    return meta, splits


_ROPE_ROWS_SRC = """
    // x [B, H, D] (T), freqs [half] float32 as rope_fast passes them (1 / inv_freq), pos [B] uint32: row b
    // rotated at position pos[b]. The arithmetic of MLX's own rope kernel (ml-explore/mlx, MIT, Copyright (c)
    // 2023 Apple Inc.) line for line - the reciprocal of the frequency, theta = position * inv_freq in
    // float32, fast cos and sin, the half-split pairs (i, i + half), x1 cos - x2 sin and x1 sin + x2 cos in
    // float32, cast back - so a row here is the bits rope_fast gives the one-row step at that position, and
    // B rows at B positions are one launch.
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


_rope_rows_kernel: MetalKernel | None = None


def rope_rows(x: mx_.array, rd: int, freqs: mx_.array, scaling: float, pos: Sequence[int] | mx_.array) -> mx_.array:
    """`rope_fast` for B rows at B positions in one launch: x [B, heads, d] rotated on its first `rd` dims, row b at
    pos[b]; bit-identical to rope_fast at each position."""
    global _rope_rows_kernel
    m = mx
    if _rope_rows_kernel is None:
        _rope_rows_kernel = metal_kernel(
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
) -> NodeMeta:
    """A batched decode step's node metadata (meta [B, 8], path [B, 1], splits): row b has ns[b] live rows, its
    rows past segs[b] in the step buffer, its prefill in slice b or at pbases[b] of one flat buffer."""
    m = mx
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


def forest_meta(lens: Sequence[int], pbases: Sequence[int], window: int | None = None) -> NodeMeta:
    """A batched prefill's node metadata: token p of a row is a node of p + 1 rows, all in place, row b's from
    pbases[b] of one flat buffer - causal attention as T decode-shaped nodes in one launch."""
    m = mx
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


_rope_rows2_kernel: MetalKernel | None = None


def rope_rows2(
    q: mx_.array, k: mx_.array, rd: int, freqs: mx_.array, scaling: float, pos: Sequence[int] | mx_.array
) -> tuple[mx_.array, mx_.array]:
    """`rope_rows` for q [B, Hq, d] and k [B, Hk, d] in one launch (full rotary dims only; partial dims take
    two `rope_rows` calls). Returns (q rotated, k rotated), the bits of rope_rows."""
    global _rope_rows2_kernel
    m = mx
    B, Hq, D = (int(s) for s in q.shape)
    if int(rd) != D or q.dtype != k.dtype:
        return rope_rows(q, rd, freqs, scaling, pos), rope_rows(k, rd, freqs, scaling, pos)
    if _rope_rows2_kernel is None:
        _rope_rows2_kernel = metal_kernel(
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
    prepared: NodeMeta | None = None,
    odt: mx_.Dtype | None = None,
    pbases: Sequence[int] | None = None,
    rmap: mx_.array | None = None,
) -> mx_.array:
    """A batched decode step: `q[b]` ([B, Hq, D] float32) over cache slice b of ns[b] live rows, each row as the
    one-row step at its length computes it. `segs`: rows past segs[b] come from `k2`/`v2`; `pbases`: the prefill
    rows in one flat buffer; `rmap`: a fork's rows over a paged session, the prefill rows (every row's the same)
    through its row map; `prepared` = `rows_meta(...)` shared across layers. Returns [B, Hq, D], lazily."""
    meta, path, splits = prepared if prepared is not None else rows_meta(ns, window, segs, pbases)
    return attn_nodes(
        q,
        k,
        v,
        meta,
        path,
        scale,
        splits,
        sinks=sinks,
        ks=ks,
        vs=vs,
        k2=k2,
        v2=v2,
        ks2=ks2,
        vs2=vs2,
        odt=odt,
        rmap=rmap,
    )


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
    odt: mx_.Dtype | None = None,
    rmap: mx_.array | None = None,
    shared: bool = False,
) -> mx_.array:
    """Attention for T query nodes over a cache, each row as aluminum's prefill computes it: `q` [T, Hq, D] (taken as
    bf16), `k`/`v` the cache buffers [B, Hk, cap, D] as they are, `meta` uint32 [T, 8] per node, `path` uint32 [T, PD]
    the node's rows as offsets after `past`, `sinks` [Hq] float32 an extra softmax column with no value (gpt-oss),
    `ks`/`vs` an int8 cache's scales, `rmap` uint32 a paged cache's row map: its rows before a node's `seg` through it
    (row j at buffer row rmap[j]), the pass's own from `k2`/`v2` as the step buffer's are (bf16 rows), or without k2
    every row through it (the pass's own written to their pages already). `shared`: the nodes are one sequence's
    (`tree_meta`'s), which lets them share their reads of its rows. Returns [T, Hq, D] in `odt` (float32 by default),
    lazily. Needs a head_dim of ATTN_HEADS and Hq / Hk <= ATTN_MAXG."""
    return aluminum.nodes(
        q,
        k,
        v,
        meta,
        path,
        scale,
        splits,
        sinks=sinks,
        ks=ks,
        vs=vs,
        k2=k2,
        v2=v2,
        ks2=ks2,
        vs2=vs2,
        odt=odt,
        rmap=rmap,
        shared=shared,
    )


_NO_PATH: mx_.array | None = None


def attn_prefill(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    past: int,
    scale: float,
    odt: mx_.Dtype | None = None,
    rmap: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    window: int | None = None,
    sinks: mx_.array | None = None,
    kbatch: int = 0,
    pbase: int = 0,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
) -> mx_.array:
    """Causal attention for a prefill chunk, each row as `attn_nodes` computes it: `q` [T, Hq, D] (taken as bf16),
    `k`/`v` the cache buffers [B, Hk, cap, D] (bf16, or int8 with their scales `ks`/`vs`) holding the chunk's rows at
    [past, past + T) of slice `kbatch` from row `pbase`; row t attends over cache rows (past + t - window, past + t]. A
    paged cache's rows before `past` through its row map `rmap` (uint32, row j at buffer row rmap[j], at least one
    entry), the chunk's own in `k2`/`v2` [1, Hk, >= T, D]; `sinks` [Hq] float32 gpt-oss's. Returns [T, Hq, D] in `odt`
    (q's dtype by default), lazily. Needs a head_dim of ATTN_HEADS and Hq / Hk <= ATTN_MAXG."""
    m = mx
    T = int(q.shape[0])
    past = int(past)
    if rmap is not None:
        if k2 is None or v2 is None:
            raise ValueError("[attn] attn_prefill: a row map takes the chunk's own rows in k2/v2")
        fits = int(rmap.shape[0]) >= max(past, 1) and int(k2.shape[2]) >= T and k2.dtype == m.bfloat16
    else:
        fits = int(pbase) + past + T <= int(k.shape[2]) and int(kbatch) < int(k.shape[0])
    if not fits:
        raise ValueError(f"[attn] attn_prefill: {T} rows at {past} past the cache {k.shape}")
    return aluminum.prefill(
        q,
        k,
        v,
        past,
        scale,
        window=window,
        sinks=sinks,
        odt=odt,
        rmap=rmap,
        k2=k2,
        v2=v2,
        kbatch=kbatch,
        pbase=pbase,
        ks=ks,
        vs=vs,
    )


def attn_prefill_rows(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    lens: Sequence[int],
    pbases: Sequence[int],
    scale: float,
    window: int | None = None,
    odt: mx_.Dtype | None = None,
    ks: mx_.array | None = None,
    vs: mx_.array | None = None,
) -> mx_.array:
    """A batched prefill (a forest's prompts) in one launch, each prompt's rows as its own `attn_prefill` computes
    them: prompt b's lens[b] rows end to end in `q` [sum(lens), Hq, D], its keys from row pbases[b] of one flat buffer
    `k`/`v` [1, Hk, cap, D] (bf16, or int8 with their scales `ks`/`vs`). Returns [sum(lens), Hq, D] in `odt` (q's
    dtype by default), lazily."""
    if any(int(p) + int(L) > int(k.shape[2]) for L, p in zip(lens, pbases, strict=True)):
        raise ValueError(f"[attn] attn_prefill_rows: a prompt past the buffer {k.shape}")
    return aluminum.prefill_rows(q, k, v, lens, pbases, scale, window=window, odt=odt, ks=ks, vs=vs)


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
    rmap: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    odt: mx_.Dtype | None = None,
) -> mx_.array:
    """One query row per head over the first `n` rows: `q` [Hq, D] (taken as bf16) -> [Hq, D] in `odt` (float32 by
    default), lazily (`attn_nodes` over one node); `params` = `attn_params(n, window=...)` when shared across layers.
    The step's own row in `k2`/`v2` [1, Hk, 1, D] (a paged cache's or an arena's), the rows before it through `rmap`,
    a paged cache's row map, or in place without one"""
    global _NO_PATH
    m = mx
    if params is None:
        params = attn_params(n, window, batch, pbase, kbatch, seg=int(n) - 1 if k2 is not None else None)
    meta, splits = params
    if _NO_PATH is None:
        _NO_PATH = m.zeros((1, 1), dtype=m.uint32)
        m.eval(_NO_PATH)
    out = attn_nodes(
        q[None], k, v, meta, _NO_PATH, scale, splits, sinks=sinks, ks=ks, vs=vs, rmap=rmap, k2=k2, v2=v2, odt=odt
    )
    return out[0]


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
    prepared: NodeMeta | None = None,
    rmap: mx_.array | None = None,
    k2: mx_.array | None = None,
    v2: mx_.array | None = None,
    odt: mx_.Dtype | None = None,
) -> mx_.array:
    """`attn_nodes` for a speculative pass: `parents` (-1 for the prefix) gives each node its path, the rows sit
    after `past` in node order; `sinks` and `window` are gpt-oss's; `prepared` = `tree_meta(...)`. The tree's own rows
    in `k2`/`v2` [1, Hk, T, D] in node order (a paged cache's or an arena's), the prefix through `rmap`, a paged
    cache's row map, or in place without one; the output in `odt` (float32 by default)"""
    meta, path, splits = (
        prepared
        if prepared is not None
        else tree_meta(past, parents, window, batch, pbase, kbatch, seg=int(past) if k2 is not None else None)
    )
    return attn_nodes(
        q, k, v, meta, path, scale, splits, sinks=sinks, ks=ks, vs=vs, rmap=rmap, k2=k2, v2=v2, odt=odt, shared=True
    )


def tree_meta(
    past: int,
    parents: Parents,
    window: int | None = None,
    batch: int = 0,
    pbase: int = 0,
    kbatch: int | None = None,
    seg: int | None = None,
) -> NodeMeta:
    """`attn_tree`'s (meta [T, 8], path [T, D], splits), which depend on the tree and the cache's geometry, not on
    the layer: built once per pass. `seg` the rows before the pass's own (a paged cache's prefix, `past`); `splits`
    the segments the nodes span (under a window, one past the most a node has where a shallow node's first segment is
    before the deepest's)."""
    m = mx
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
            n if seg is None else int(seg),
            int(pbase),
            int(batch) if kbatch is None else int(kbatch),
        )
    span = attn_splits(int(meta[:, 0].max())) - int(meta[:, 3].min()) // ATTN_BLOCK
    return m.array(meta), m.array(path), span


def kv_quantize(x: mx_.array) -> tuple[mx_.array, mx_.array]:
    """`x` [..., T, D] to int8 with one float32 scale per row (s = max|row| / 127, 1 for a zero row): the
    dequantized row is within s / 2 of x."""
    m = mx
    xf = x.astype(m.float32)
    s = m.maximum(m.max(m.abs(xf), axis=-1), 1e-30) / 127.0
    q = m.round(xf / s[..., None]).astype(m.int8)
    return q, s


def kv_dequantize(q: mx_.array, s: mx_.array, dtype: mx_.Dtype | None = None) -> mx_.array:
    """the int8 rows back to `dtype` (bf16 by default): q * s, one scale per row"""
    m = mx
    return (q.astype(m.float32) * s[..., None]).astype(dtype or m.bfloat16)
