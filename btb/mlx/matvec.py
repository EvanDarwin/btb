# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The stored GGUF types' matvec launch on Metal: up to 16 pass rows a launch, each simdgroup RP output rows by
TRS pass rows of it. The output rows share each activation load; the pass rows split across simdgroups where the
accumulators would not fit in registers. A row's chain is the same whichever shape runs it, so row i is
bit-identical for a 1-row and a 16-row call."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

ROWS_MAX = 16  # pass rows a launch; a wider pass is chunked

# a launch's pass rows -> (RP, TRS). A TRS past the pass rows pads them to TRS: the padding rows read the last real
# row's activations and are never written (11 and 13 rows compile to slower code than 12 and 14).
Shapes = Mapping[int, tuple[int, int]]

# the k-quant, Q6_K and IQ4_XS kernels' (the default): each width's least mean time on an M3 Pro over those types
# at [3072, 1024] and [9728, 2560], of 1, 2 or 4 rows by all the pass rows, the 4- and 8-row splits, and the
# padded width
SHAPES: Shapes = {
    1: (4, 1),
    2: (4, 2),
    3: (4, 3),
    4: (2, 4),
    5: (4, 6),
    6: (4, 6),
    7: (4, 8),
    8: (4, 8),
    9: (4, 9),
    10: (4, 10),
    11: (4, 12),
    12: (4, 12),
    13: (4, 14),
    14: (4, 14),
    15: (4, 15),
    16: (4, 16),
}

# the launch's grid as (row group, pass-row group) simdgroups, and XROW(t) pass row t's activations (a padding row
# reads the last real one's)
PROLOGUE = r"""
    uint gid = thread_position_in_grid.x;
    uint grp = gid / 32, lane = gid % 32;
    uint tr0 = (grp % (TR / TRS)) * TRS;
    uint row0 = (grp / (TR / TRS)) * RP;
    if (row0 >= ROWS) return;
"""

EPILOGUE = r"""
    #pragma unroll
    for (int r = 0; r < RP; r++) {
        if (row0 + r >= ROWS) break;
        #pragma unroll
        for (int tr = 0; tr < TRS; tr++) {
            float a = simd_sum(acc[r][tr]);
            if (lane == 0 && tr0 + tr < TRN) out[(size_t)(tr0 + tr) * ROWS + row0 + r] = (T)a;
        }
    }
"""


def blocks(lanes: str, decl: str, dec: str, apply: str) -> str:
    """a kernel walking a row's NSB blocks of BW weights: `lanes` the lane's constants, `decl` a block's registers,
    `dec` decoding row r's block c into them (`row` clamped: a tail group repeats its last row, never written),
    `apply` adding pass row tr's activations at `xc` into acc[r][tr] for each r - the row's own chain"""
    return (
        PROLOGUE
        + lanes
        + r"""
    float acc[RP][TRS];
    #pragma unroll
    for (int r = 0; r < RP; r++)
        #pragma unroll
        for (int u = 0; u < TRS; u++) acc[r][u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
"""
        + decl
        + r"""
        #pragma unroll
        for (int r = 0; r < RP; r++) {
            uint row = min(row0 + r, (uint)ROWS - 1);
"""
        + dec
        + r"""
        }
        #pragma unroll
        for (int tr = 0; tr < TRS; tr++) {
            const device T* xc = XROW(tr0 + tr) + (size_t)c * BW;
"""
        + apply
        + r"""
        }
    }
"""
        + EPILOGUE
    )


def shape(shapes: Shapes, tr: int) -> tuple[int, int, int]:
    """(RP, TRS, launch width) for `tr` pass rows: the table's, else 4 rows by all of them"""
    rp, trs = shapes.get(tr) or (4, tr)
    return rp, trs, tr if tr % trs == 0 else trs


_kernels: dict[tuple[str, int, int, int], MetalKernel] = {}
_lock = threading.Lock()


def launch(
    name: str,
    source: str,
    defs: str,
    inputs: Sequence[tuple[str, mx_.array]],
    x: mx_.array,
    rows: int,
    cols: int,
    bw: int,
    shapes: Shapes | None = None,
) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T through `source`, a kernel reading `inputs` by name beside x over NSB blocks of
    `bw` weights (the last one partial where `bw` does not divide `cols`); fp32 accumulation, y in x's dtype. `name`
    is the caller's kernel name for the matrix, the launch shape appended; `shapes` a kernel's own, else `SHAPES`."""
    m = mx()
    nsb = -(-cols // bw)
    n = int(x.shape[0])
    outs = []
    for s in range(0, n, ROWS_MAX):
        xr = x if n <= ROWS_MAX else x[s : s + ROWS_MAX]
        tr = int(xr.shape[0])
        rp, trs, trl = shape(SHAPES if shapes is None else shapes, tr)
        key = (name, tr, rp, trs)
        k = _kernels.get(key)
        if k is None:
            k = _compile(key, source, defs, inputs, rows, nsb, bw, cols, trl)
        groups = -(-rows // rp) * (trl // trs)
        outs.append(
            k(
                inputs=[a for _, a in inputs] + [xr],
                grid=(((groups * 32 + 255) // 256) * 256, 1, 1),  # a simdgroup a (row group, pass-row group)
                threadgroup=(256, 1, 1),
                output_shapes=[(tr, rows)],
                output_dtypes=[x.dtype],
                template=[("T", x.dtype)],
            )[0]
        )
    return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


def _compile(
    key: tuple[str, int, int, int],
    source: str,
    defs: str,
    inputs: Sequence[tuple[str, mx_.array]],
    rows: int,
    nsb: int,
    bw: int,
    cols: int,
    trl: int,
) -> MetalKernel:
    name, tr, rp, trs = key
    shaped = f"{name}_{tr}_{rp}_{trs}"  # the caller's name (the kernel the matrix runs), its launch shape appended
    with _lock:
        k = _kernels.get(key)
        if k is None:
            k = _kernels[key] = mx().fast.metal_kernel(
                name=shaped,
                input_names=[n for n, _ in inputs] + ["x"],
                output_names=["out"],
                header=(
                    f"{defs}#define ROWS {rows}\n#define NSB {nsb}\n#define BW {bw}\n#define COLS {cols}\n"
                    f"#define TR {trl}\n#define TRN {tr}\n#define RP {rp}\n#define TRS {trs}\n"
                    + (
                        "#define XROW(t) (x + (size_t)(t) * COLS)\n"
                        if trl == tr
                        else "#define XROW(t) (x + (size_t)min((uint)(t), (uint)TRN - 1) * COLS)\n"
                    )
                ),
                source=source,
            )
    return k
