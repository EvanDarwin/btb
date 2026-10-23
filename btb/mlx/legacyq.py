# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""ggml's legacy block quants on Metal - Q4_0, Q4_1, Q8_0 - read as stored, 32 weights a block, each weight widened
in registers and accumulated in fp32 so row i is bit-identical for a 1-row and a 16-row call (which MLX's
`quantized_matmul` is not: its multi-row kernel is not its one-row one, so a speculative verify pass would drift off
the plain step). `matvec_legacy` is the linear, `dequant_legacy` the whole matrix for a path wanting a bf16 copy."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from ..kinds import Quant
from .core import metal_kernel, mx
from .matvec import Frags, Shapes, StoredKernel, blocks, launch

if TYPE_CHECKING:
    import mlx.core as mx_

# (block bytes, kernel id) by kind. Q4_0: d f16, 16 nibble bytes, value d*(q-8). Q4_1: d f16, m f16, 16 nibble
# bytes, value d*q + m. Byte i holds weight i in its low nibble and weight i+16 in its high. Q8_0: d f16, 32 int8,
# value d*q. Every block offset is even, so the f16 fields are aligned.
KINDS: dict[str, tuple[int, int]] = {"q4_0": (18, 0), "q4_1": (20, 1), "q8_0": (34, 2)}

_WEIGHT = r"""
inline float legacy_weight(const device uchar* blk, uint i) {
    float d = (float)(*(const device half*)(blk));
#if KIND == 0
    uchar b = blk[2 + (i & 15)];
    return d * (float)((int)(i < 16 ? (b & 0xF) : (b >> 4)) - 8);
#elif KIND == 1
    uchar b = blk[4 + (i & 15)];
    return d * (float)(i < 16 ? (b & 0xF) : (b >> 4)) + (float)(*(const device half*)(blk + 2));
#else
    return d * (float)((const device char*)(blk + 2))[i];
#endif
}
"""

# one simdgroup RP rows; four lanes share a block and eight blocks go a pass. Lane `q` of a block takes its bytes
# 4q..4q+3: for Q4_0/Q4_1 the weights 4q..4q+3 (low nibbles) and 4q+16..4q+19 (high), for Q8_0 the weights
# 8q..8q+7, and reads its eight activations of each pass row as two (Q8_0: one) aligned vector loads, applied to
# its RP rows' weights; fp32 MAC, then simd_sum. A row's chain - its block's widening, the fixed eight-term sum,
# the accumulate - is the same at any TR and whichever rows share its group, so RP is free to follow TR.
MV = Frags(
    "    uint q = lane & 3;\n",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * BLKB;
            float d = (float)(*(const device half*)(blk));
#if KIND == 2
            const device char* qs = (const device char*)(blk + 2) + 8 * q;
            #pragma unroll
            for (int t = 0; t < 8; t++) w[r][t] = d * (float)qs[t];
#else
#if KIND == 1
            float mn = (float)(*(const device half*)(blk + 2));
            const device uchar* qs = blk + 4 + 4 * q;
#else
            const device uchar* qs = blk + 2 + 4 * q;
#endif
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                uchar b = qs[t];
#if KIND == 1
                w[r][t] = d * (float)(b & 0xF) + mn;
                w[r][t + 4] = d * (float)(b >> 4) + mn;
#else
                w[r][t] = d * (float)((int)(b & 0xF) - 8);
                w[r][t + 4] = d * (float)((int)(b >> 4) - 8);
#endif
            }
#endif
""",
    r"""
#if KIND == 2
            vec<T, 4> a = *(const device vec<T, 4>*)(xc + 8 * q);
            vec<T, 4> b = *(const device vec<T, 4>*)(xc + 8 * q + 4);
#else
            vec<T, 4> a = *(const device vec<T, 4>*)(xc + 4 * q);
            vec<T, 4> b = *(const device vec<T, 4>*)(xc + 4 * q + 16);
#endif
            float xv[8] = {(float)a[0], (float)a[1], (float)a[2], (float)a[3],
                           (float)b[0], (float)b[1], (float)b[2], (float)b[3]};
            #pragma unroll
            for (int r = 0; r < RP; r++) {
                float s = 0.0f;
                #pragma unroll
                for (int t = 0; t < 8; t++) s += w[r][t] * xv[t];
                acc[r][tr] += s;
            }
""",
    step=8,
    lane_block="(lane >> 2)",
)
_MATVEC = blocks(MV)

# one thread a block
_DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= NBLK) return;
    const device uchar* blk = W + (size_t)t * BLKB;
    for (uint i = 0; i < 32; i++) out[(size_t)t * 32 + i] = (T)legacy_weight(blk, i);
"""

_deq_kernels: dict[str, Any] = {}
_lock = threading.Lock()
# this kernel's own launch shapes (`matvec.Shapes`), its lanes splitting a row's blocks rather than one superblock:
# measured as the default's, over Q4_0, Q4_1 and Q8_0
SHAPES: Shapes = {
    **{t: (1, t) for t in range(1, 17)},
    1: (2, 1),
    3: (4, 3),
    4: (2, 4),
    11: (4, 12),
    12: (2, 12),
    13: (4, 14),
    14: (4, 14),
    15: (4, 15),
    16: (1, 8),
}
# a kind's own where it measured apart: Q8_0's 16-wide pass on four rows a simdgroup (0.94 and 0.75 of (1, 8))
KIND_SHAPES: dict[str, Shapes] = {"q8_0": {**SHAPES, 16: (4, 8)}}


def _kind(kind: str) -> str:
    blkb, kid = KINDS[kind]
    return f"#define BLKB {blkb}\n#define KIND {kid}\n"


def _defs(kind: str) -> str:
    return _kind(kind) + _WEIGHT


# Q8_0 for kernels built on its chain (the megakernel): a row in bf16, a weight a thread as `_DEQUANT` widens it
KERNELS: dict[Quant, StoredKernel] = {
    Quant.Q8_0: StoredKernel(
        MV,
        _WEIGHT,
        _kind("q8_0"),
        32,
        r"""
    for (uint i = tid; i < NSB * 32; i += 256)
        dst[i] = (bfloat)legacy_weight(W + ((size_t)row * NSB + i / 32) * BLKB, i % 32);
""",
        KIND_SHAPES.get("q8_0", SHAPES),
    )
}


def matvec_legacy(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw `kind` bytes (rows * cols / 32 row-major blocks), x
    bf16 or float32; fp32 accumulation over the blocks as stored, y in x's dtype. 1..16 rows a launch, a wider pass
    chunked."""
    nb = cols // 32
    shapes = KIND_SHAPES.get(kind, SHAPES)
    return launch(f"btb_{kind}_mv_{rows}_{nb}", _MATVEC, _defs(kind), [("W", w_bytes)], x, rows, cols, 32, shapes)


def dequant_legacy(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight, the numbers the gguf package dequantizes to"""
    m = mx()
    nblk = rows * cols // 32
    with _lock:
        k = _deq_kernels.get(kind)
        if k is None:
            k = _deq_kernels[kind] = metal_kernel(
                name=f"btb_{kind}_dequant",
                input_names=["W"],
                output_names=["out"],
                header=_defs(kind),
                source=_DEQUANT,
            )
    return k(
        inputs=[w_bytes],
        grid=(((nblk + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]
