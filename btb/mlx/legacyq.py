# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""ggml's legacy block quants on Metal - Q4_0, Q4_1, Q8_0 - read as stored, 32 weights a block, each weight widened
in registers and accumulated in fp32 so row i is bit-identical for a 1-row and a 16-row call (which MLX's
`quantized_matmul` is not: its multi-row kernel is not its one-row one, so a speculative verify pass would drift off
the plain step). `matvec_legacy` is the linear, `dequant_legacy` the whole matrix for a path wanting a bf16 copy."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from .core import mx

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
_MATVEC = r"""
    uint gid = thread_position_in_grid.x;
    uint grp = gid / 32, lane = gid % 32;
    uint tr0 = (grp % (TR / TRS)) * TRS;  // the pass rows this simdgroup takes: TRS of the TR, split for registers
    uint row0 = (grp / (TR / TRS)) * RP;
    if (row0 >= ROWS) return;
    uint q = lane & 3;
    float acc[RP][TRS];
    #pragma unroll
    for (int r = 0; r < RP; r++)
        #pragma unroll
        for (int u = 0; u < TRS; u++) acc[r][u] = 0.0f;
    for (uint c = lane >> 2; c < NB; c += 8) {
        float w[RP][8];
        #pragma unroll
        for (int r = 0; r < RP; r++) {
            uint row = min(row0 + r, (uint)ROWS - 1);  // a tail group repeats its last row, never written
            const device uchar* blk = W + ((size_t)row * NB + c) * BLKB;
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
        }
        #pragma unroll
        for (int tr = 0; tr < TRS; tr++) {
            const device T* xb = x + (size_t)(tr0 + tr) * (NB * 32) + (size_t)c * 32;
#if KIND == 2
            vec<T, 4> a = *(const device vec<T, 4>*)(xb + 8 * q);
            vec<T, 4> b = *(const device vec<T, 4>*)(xb + 8 * q + 4);
#else
            vec<T, 4> a = *(const device vec<T, 4>*)(xb + 4 * q);
            vec<T, 4> b = *(const device vec<T, 4>*)(xb + 4 * q + 16);
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
        }
    }
    #pragma unroll
    for (int r = 0; r < RP; r++) {
        if (row0 + r >= ROWS) break;
        #pragma unroll
        for (int tr = 0; tr < TRS; tr++) {
            float a = simd_sum(acc[r][tr]);
            if (lane == 0) out[(size_t)(tr0 + tr) * ROWS + row0 + r] = (T)a;
        }
    }
"""

# one thread a block
_DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= NBLK) return;
    const device uchar* blk = W + (size_t)t * BLKB;
    for (uint i = 0; i < 32; i++) out[(size_t)t * 32 + i] = (T)legacy_weight(blk, i);
"""

_matvec_kernels: dict[tuple[str, int, int, int, int, int], Any] = {}
_deq_kernels: dict[str, Any] = {}
_lock = threading.Lock()
ROWS_MAX = 16  # a call takes up to 16 pass rows (acc[RP][TRS] in registers); a wider pass is chunked
# (output rows, pass rows) a simdgroup, by pass rows, as measured against `quantized_matmul` at 4096 x 4096: the
# rows share each activation load, the pass rows split across simdgroups to keep the accumulators in registers. A
# row's bits depend on neither; a width not listed takes one row and the widest of 8/4/2/1 pass rows dividing it.
SHAPE_BY_TR: dict[int, tuple[int, int]] = {1: (2, 1), 2: (2, 2), 4: (1, 4), 8: (1, 8), 12: (2, 4), 16: (1, 8)}


def _shape(tr: int) -> tuple[int, int]:
    return SHAPE_BY_TR.get(tr) or (1, next(s for s in (8, 4, 2, 1) if tr % s == 0))


def _defs(kind: str) -> str:
    blkb, kid = KINDS[kind]
    return f"#define BLKB {blkb}\n#define KIND {kid}\n{_WEIGHT}"


def _matvec_kernel(kind: str, rows: int, nb: int, tr: int, rp: int, trs: int) -> Any:
    key = (kind, rows, nb, tr, rp, trs)
    with _lock:
        k = _matvec_kernels.get(key)
        if k is None:
            k = _matvec_kernels[key] = mx().fast.metal_kernel(
                name=f"btb_{kind}_mv_{rows}_{nb}_{tr}_{rp}_{trs}",
                input_names=["W", "x"],
                output_names=["out"],
                header=(
                    f"{_defs(kind)}#define ROWS {rows}\n#define NB {nb}\n#define TR {tr}\n#define RP {rp}\n"
                    f"#define TRS {trs}\n"
                ),
                source=_MATVEC,
            )
    return k


def matvec_legacy(
    kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int, shape: tuple[int, int] | None = None
) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw `kind` bytes (rows * cols / 32 row-major blocks), x
    bf16 or float32; fp32 accumulation over the blocks as stored, y in x's dtype. 1..16 rows a launch, a wider pass
    chunked; `shape` overrides (output rows, pass rows) a simdgroup (the bits are the same at any)."""
    m = mx()
    nb = cols // 32
    outs = []
    for s in range(0, int(x.shape[0]), ROWS_MAX):
        xr = x[s : s + ROWS_MAX]
        tr = int(xr.shape[0])
        g, trs = shape or _shape(tr)
        trs = min(trs, tr) if tr % min(trs, tr) == 0 else tr
        groups = -(-rows // g) * (tr // trs)
        grid = ((groups * 32 + 255) // 256) * 256  # a simdgroup a (row group, pass-row group), 256-thread groups
        outs.append(
            _matvec_kernel(kind, rows, nb, tr, g, trs)(
                inputs=[w_bytes, xr],
                grid=(grid, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(tr, rows)],
                output_dtypes=[x.dtype],
                template=[("T", x.dtype)],
            )[0]
        )
    return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


def dequant_legacy(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight, the numbers the gguf package dequantizes to"""
    m = mx()
    nblk = rows * cols // 32
    with _lock:
        k = _deq_kernels.get(kind)
        if k is None:
            k = _deq_kernels[kind] = m.fast.metal_kernel(
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
