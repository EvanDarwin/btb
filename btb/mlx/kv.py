# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""In-place writes into a K/V arena: a layer's buffers are views of one shared array, and a view's slice
assignment never reaches the parent (copy-on-write), so the rows are written through the buffer itself."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_

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

_lock = threading.Lock()
_store_kernel = None


def kv_store(buf: mx_.array, rows: mx_.array, n0: int) -> mx_.array:
    """`rows` [Hk, T, D] bf16 into `buf` [1, Hk, cap, D] at row `n0`, through the buffer's own bytes. Returns
    the kernel's flag: evaluate it before anything reads the rows (the write has no graph dependency)."""
    global _store_kernel
    m = mx()
    with _lock:
        if _store_kernel is None:
            _store_kernel = m.fast.metal_kernel(
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


def kv_gather(buf: mx_.array, keep: Sequence[int], base: int) -> mx_.array:
    """rows `keep` of `buf` [1, Hk, cap, D] become rows base.. (a gathered copy written back in place); returns
    the flag to evaluate"""
    m = mx()
    idx = m.array(np.asarray(list(keep), dtype=np.int32))
    rows = m.take(buf[0], idx, axis=1)
    return kv_store(buf, rows, base)
