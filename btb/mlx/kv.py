# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""In-place writes into a K/V arena: a layer's buffers are views of one shared array, and a view's slice
assignment never reaches the parent (copy-on-write), so the rows are written through the buffer itself."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .core import metal_kernel, mx

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

_STORE_SRC = """
    // rows [Hk, T, D] (16-bit) written into buf [1, Hk, cap, D] from row n0[0], in place
    uint i = thread_position_in_grid.x;
    uint Hk = rows_shape[0], T = rows_shape[1], D = rows_shape[2];
    uint cap = buf_shape[2];
    if (i >= Hk * T * D) return;
    uint h = i / (T * D), r = (i / D) % T, c = i % D;
    ((device uint16_t*)buf)[((size_t)h * cap + n0[0] + r) * D + c] = rows[i];
    if (i == 0) done[0] = 1u;
"""

_SCATTER_SRC = """
    // a pass's K and V rows [1, Hk, T, D] written into kb / vb [1, Hk, cap, D] (all of the buffers' dtype E) at rows
    // at[0..T), in place: a paged cache's rows where its table puts them, K's threads first, then V's
    uint i = thread_position_in_grid.x;
    uint Hk = k_shape[1], T = k_shape[2], D = k_shape[3];
    uint cap = kb_shape[2];
    uint n = Hk * T * D;
    if (i >= 2 * n) return;
    uint e = i < n ? i : i - n;
    uint h = e / (T * D), r = (e / D) % T, c = e % D;
    size_t o = ((size_t)h * cap + at[r]) * D + c;
    if (i < n) ((device E*)kb)[o] = k[e]; else ((device E*)vb)[o] = v[e];
    if (i == 0) done[0] = 1u;
"""

_lock = threading.Lock()
_store_kernel: MetalKernel | None = None
_scatter_kernel: MetalKernel | None = None


def kv_store(buf: mx_.array, rows: mx_.array, n0: int) -> mx_.array:
    """`rows` [Hk, T, D] bf16 into `buf` [1, Hk, cap, D] at row `n0`, through the buffer's own bytes. Returns
    the kernel's flag: evaluate it before anything reads the rows (the write has no graph dependency)."""
    global _store_kernel
    m = mx()
    with _lock:
        if _store_kernel is None:
            _store_kernel = metal_kernel(
                name="btb_kv_store", input_names=["buf", "rows", "n0"], output_names=["done"], source=_STORE_SRC
            )
    at = m.array(np.full((8,), int(n0), dtype=np.uint32))
    return _store_kernel(
        inputs=[buf.view(m.uint16), rows.astype(buf.dtype).view(m.uint16), at],
        grid=(int(rows.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[m.uint32],
    )[0]


def kv_scatter(kb: mx_.array, vb: mx_.array, k: mx_.array, v: mx_.array, at: mx_.array) -> mx_.array:
    """`k`/`v` [1, Hk, T, D] into `kb`/`vb` [1, Hk, cap, D] at rows `at` (uint32 [T]), in the buffers' dtype, through
    their own bytes, in one launch. Returns the kernel's flag to evaluate: the write has no graph edge, and MLX orders
    no read of the buffers in the same evaluation after it (`mx.depends` neither), so the rows are read by a later
    one"""
    global _scatter_kernel
    m = mx()
    with _lock:
        if _scatter_kernel is None:
            _scatter_kernel = metal_kernel(
                name="btb_kv_scatter",
                input_names=["kb", "vb", "k", "v", "at"],
                output_names=["done"],
                source=_SCATTER_SRC,
            )
    # a cast of rows already in the buffers' dtype is a copy of its own (MLX's astype makes one)
    k = k if k.dtype == kb.dtype else k.astype(kb.dtype)
    v = v if v.dtype == vb.dtype else v.astype(vb.dtype)
    return _scatter_kernel(
        inputs=[kb, vb, k, v, at],
        template=[("E", kb.dtype)],
        grid=(2 * int(k.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[m.uint32],
    )[0]


def kv_gather(buf: mx_.array, keep: Sequence[int], base: int) -> mx_.array:
    """rows `keep` of `buf` [1, Hk, cap, D] become rows base.. (a gathered copy written back in place); returns
    the flag to evaluate"""
    m = mx()
    idx = m.array(np.asarray(list(keep), dtype=np.int32))
    rows = m.take(buf[0], idx, axis=1)
    return kv_store(buf, rows, base)
