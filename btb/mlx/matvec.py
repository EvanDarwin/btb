# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The stored GGUF types' matmul on Metal, one chain at every width. A type's `w8` gives 8 consecutive weights of a
row as float32 from its streams, and every kernel over the type is built on it. The chain runs K a 64-group at a time
in an order of its own (`chain_order`): logical column 8*ks + f of a group is physical (f/2)*16 + 2*ks + f%2, so a
lane's 16 consecutive weights land straight in its 8x8 matrix fragments, each rounded to bf16 for a bf16 x. The
matvec multiplies them there in ascending logical K; a prefill's gemm (the same chain, steel.py) runs over the dequant
written in that order and x taken in it (`chain_x`). So a row has the same bits in a step, a verify pass, a batch, a
prefill cut anywhere, the megakernel and a mixture's gather."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from .core import metal_kernel, mx
from .gemv import TILE_ROWS

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel


@dataclass(frozen=True)
class Table:
    """a codebook a stored type's kernels read beside its streams: the name its source reads it by, its element
    type, and its array"""

    name: str
    ctype: str
    make: Callable[[], mx_.array]


@dataclass(frozen=True)
class StoredKernel:
    """a stored type as every kernel over it is built: `w8`, the body of a device function writing the float32
    weights of row `row`, columns k..k+7 (k a multiple of 8, under `cols`, the row's width) into `w` from the type's
    `streams` and `tables` (each read by its name); `header` the device functions it calls, `defs` the defines it
    reads, `bw` the weights a block (a row is whole blocks). `streams` in `kinds.stored_stream_bytes`' order"""

    w8: str
    bw: int
    header: str = ""
    defs: str = ""
    streams: tuple[tuple[str, str], ...] = (("W", "uint8_t"),)
    tables: tuple[Table, ...] = ()


def inputs(k: StoredKernel) -> list[tuple[str, str]]:
    """what a type's kernels read, each by its name and element type: its streams, then its tables"""
    return [*k.streams, *((t.name, t.ctype) for t in k.tables)]


def device_fns(k: StoredKernel) -> str:
    """the type's device functions and `w8` over its inputs; a kernel calls it as W8(row, k, w), its COLS the width"""
    params = "".join(f"const device {t}* {n}, " for n, t in inputs(k))
    args = "".join(f"{n}, " for n, _ in inputs(k))
    return (
        f"{k.header}inline void w8({params}uint cols, uint row, uint k, thread float* w) {{\n{k.w8}}}\n"
        f"#define W8(row, k, w) w8({args}COLS, row, k, w)\n"
    )


# y[b, n] = sum_k x[b, k] w[n, k] for NB rows of x: a simdgroup owns 8 rows of W and runs K a 64-group at a time in
# the chain's order, 8 matrix multiply-accumulates a group into one float accumulator an 8-row block of x. Lane (fm,
# fn) holds A's [fm][fn, fn+1] (row fm's logical columns fn, fn+1 of each of the 8: its physical 16*(fn/2)..+15, two
# w8s) and B's [fm][fn, fn+1] (x rows fn, fn+1 at logical column fm: physical 16*(fm/2) + 2*ks + fm%2, from two or four
# 16-byte loads). XB: bf16 x and A (a weight rounded to bf16), y bf16 (round to nearest even); else float32. XROW(t)
# and YROW(t) are row t of x and of y; BIND declares a gather's item's streams. A row past ROWS is decoded from the
# last real one and never written; a column past COLS (a 32-wide type's half group) is zero.
TILE = r"""
    constexpr uint NA = (NB + 7) / 8;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * SGN + sg) * 8;
    if (n0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
    typedef simdgroup_bfloat8x8 SM;
    constexpr uint XV = 2;
#else
    typedef float ST;
    typedef simdgroup_float8x8 SM;
    constexpr uint XV = 4;
#endif
    uint qid = lane / 4;
    uint fm = (qid & 4) + (lane / 2) % 4, fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint row = min(n0 + fm, (uint)ROWS - 1), wq = 16 * (fn / 2), xq = 16 * (fm / 2), xe = fm % 2;
    simdgroup_float8x8 acc[NA];
    #pragma unroll
    for (uint j = 0; j < NA; ++j) acc[j] = simdgroup_float8x8(0.0f);
    for (uint g = 0; g < COLS; g += 64) {
        float w[16];
        if (g + wq < COLS) {
            W8(row, g + wq, w);
            W8(row, g + wq + 8, w + 8);
        } else {
            for (uint i = 0; i < 16; ++i) w[i] = 0.0f;
        }
        ST a[16];
        #pragma unroll
        for (uint i = 0; i < 16; ++i) a[i] = (ST)w[i];
        bool xin = g + xq < COLS;
        #pragma unroll
        for (uint j = 0; j < NA; ++j) {
            uint t0 = 8 * j + fn, t1 = t0 + 1;
            const device uint4* p0 = (const device uint4*)(XROW(min(t0, (uint)NB - 1)) + g + xq);
            const device uint4* p1 = (const device uint4*)(XROW(min(t1, (uint)NB - 1)) + g + xq);
            uint4 v0[XV], v1[XV];
            #pragma unroll
            for (uint v = 0; v < XV; ++v) {
                v0[v] = xin && t0 < NB ? p0[v] : uint4(0u);
                v1[v] = xin && t1 < NB ? p1[v] : uint4(0u);
            }
            #pragma unroll
            for (uint ks = 0; ks < 8; ++ks) {
                SM A, B;
                A.thread_elements()[0] = a[2 * ks];
                A.thread_elements()[1] = a[2 * ks + 1];
#if XB
                uint u0 = v0[ks / 4][ks % 4], u1 = v1[ks / 4][ks % 4];
                B.thread_elements()[0] = as_type<bfloat>((ushort)(xe ? u0 >> 16 : u0 & 0xFFFF));
                B.thread_elements()[1] = as_type<bfloat>((ushort)(xe ? u1 >> 16 : u1 & 0xFFFF));
#else
                uint4 q0 = v0[ks / 2], q1 = v1[ks / 2];
                B.thread_elements()[0] = as_type<float>(ks % 2 ? (xe ? q0.w : q0.z) : (xe ? q0.y : q0.x));
                B.thread_elements()[1] = as_type<float>(ks % 2 ? (xe ? q1.w : q1.z) : (xe ? q1.y : q1.x));
#endif
                simdgroup_multiply_accumulate(acc[j], A, B, acc[j]);
            }
        }
    }
    if (n0 + fm < ROWS) {
        #pragma unroll
        for (uint j = 0; j < NA; ++j) {
            #pragma unroll
            for (uint e = 0; e < 2; ++e) {
                uint t = 8 * j + fn + e;
                if (t < NB) {
#if XB
                    uint u = as_type<uint>(acc[j].thread_elements()[e]);
                    u += 0x7FFFu + ((u >> 16) & 1u);
                    ((device uint16_t*)YROW(t))[n0 + fm] = (uint16_t)(u >> 16);
#else
                    ((device float*)YROW(t))[n0 + fm] = acc[j].thread_elements()[e];
#endif
                }
            }
        }
    }
"""
# a tile launch's simdgroups a threadgroup: it holds no threadgroup memory, and 2 and 4 tie, 8 and 16 run slower
SGN = 4


def tile_defines(rows: int, cols: int, nb: int, xb: bool) -> str:
    """a tile launch's defines but its rows' addresses: the matrix, the pass rows, its simdgroups a threadgroup"""
    return f"#define ROWS {rows}\n#define COLS {cols}\n#define NB {nb}\n#define SGN {SGN}\n#define XB {int(xb)}\n"


def tile_groups(rows: int) -> int:
    """the threadgroups a tile launch over `rows` outputs takes: a simdgroup each 8 rows, `SGN` a threadgroup"""
    return -(-rows // (8 * SGN))


def chain_cols(cols: int) -> int:
    """the chain's columns over a row of `cols`: whole 64-groups, the last one's tail zero"""
    return -(-cols // 64) * 64


@functools.cache
def chain_order(cols: int) -> mx_.array:
    """int32 [chain_cols(cols)]: the physical column of each of the chain's logical ones"""
    lg = np.arange(chain_cols(cols))
    g, ks, f = lg - lg % 64, lg % 64 // 8, lg % 8
    order = mx.array((g + f // 2 * 16 + 2 * ks + f % 2).astype(np.int32))
    mx.eval(order)
    return order


def chain_x(x: mx_.array, cols: int) -> mx_.array:
    """x [b, cols] in the chain's column order (`chain_order`), zero past `cols`: a prefill's gemm over the dequant
    `chain` writes is the matvec's chain"""
    m = mx
    pc = chain_cols(cols)
    xp = x if pc == cols else m.pad(x, [(0, 0), (0, pc - cols)])
    return m.take(xp, chain_order(cols), axis=1)


# one thread 8 consecutive weights of a row, as the tile rounds them: OT bf16 or float32, OC columns a row out. CHAIN:
# into the chain's order (physical k + i is logical (k & ~63) + 8 * ((k % 16 + i) / 2) + 2 * (k % 64 / 16) + i % 2),
# a 32-wide type's half group zero
DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= ROWS * (OC / 8)) return;
    uint row = t / (OC / 8), k = (t % (OC / 8)) * 8;
    float w[8];
    if (k < COLS) {
        W8(row, k, w);
    } else {
        for (uint i = 0; i < 8; ++i) w[i] = 0.0f;
    }
#if CHAIN
    device OT* o = out + (size_t)row * OC + (k & ~63u) + 4 * (k % 16) + 2 * (k % 64 / 16);
    #pragma unroll
    for (uint i = 0; i < 8; ++i) o[8 * (i / 2) + i % 2] = (OT)w[i];
#else
    #pragma unroll
    for (uint i = 0; i < 8; ++i) out[(size_t)row * OC + k + i] = (OT)w[i];
#endif
"""

# row `row` of the matrix in bf16 into dst, a thread 8 weights of each 2048 (a megakernel's embedding: the dequant's)
DECODE_ROW = r"""
    for (uint k = tid * 8; k < COLS; k += 2048) {
        float w[8];
        W8(row, k, w);
        for (uint i = 0; i < 8; ++i) dst[k + i] = (bfloat)w[i];
    }
"""

_kernels: dict[tuple[object, ...], MetalKernel] = {}
_lock = threading.Lock()


def _whole(k: StoredKernel, cols: int) -> None:
    if cols % k.bw:
        raise ValueError(f"[mlx] a stored row of {cols} columns is not whole blocks of {k.bw}")


class Source(NamedTuple):
    """a kernel as `metal_kernel` builds it: its name, its inputs' names, its header and its body (one output, out)"""

    name: str
    inputs: list[str]
    header: str
    body: str


def compiled(key: tuple[object, ...], make: Callable[[], Source]) -> MetalKernel:
    """the kernel cached under `key`, built from `make()` on its first call: a launch is a lookup, its source (several
    KB, ~5 us to assemble) made once"""
    k = _kernels.get(key)
    if k is None:
        with _lock:
            k = _kernels.get(key)
            if k is None:
                s = make()
                k = _kernels[key] = metal_kernel(
                    name=s.name, input_names=s.inputs, output_names=["out"], header=s.header, source=s.body
                )
    return k


def _tile(name: str, k: StoredKernel, rows: int, cols: int, nb: int, xb: bool) -> Source:
    _whole(k, cols)
    return Source(
        f"{name}_{nb}_x{int(xb)}",
        [*(nm for nm, _ in inputs(k)), "x"],
        "#include <metal_simdgroup_matrix>\n" + k.defs + device_fns(k),
        tile_defines(rows, cols, nb, xb)
        + "#define XROW(t) (x + (size_t)(t) * COLS)\n#define YROW(t) (out + (size_t)(t) * ROWS)\n"
        + TILE.replace("BIND", ""),
    )


def launch(name: str, k: StoredKernel, ins: Sequence[mx_.array], x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T over a stored type's `ins` (its streams then its tables, `inputs`' order), x bf16
    or float32, y in its dtype: launches of about equal rows, `gemv.TILE_ROWS` at most. `name` is the caller's kernel
    name for the matrix (its rows and columns), the pass rows appended"""
    m = mx
    xb = x.dtype == m.bfloat16
    xin = x if xb else x.astype(m.float32)
    b = int(xin.shape[0])
    n = -(-b // TILE_ROWS[xb])
    size = -(-b // n)
    grid, tg = (tile_groups(rows) * 32 * SGN, 1, 1), (32 * SGN, 1, 1)
    outs = []
    for s in range(0, b, size):
        xr = xin if n == 1 else xin[s : s + size]
        nb = int(xr.shape[0])
        kern = compiled((name, nb, xb), functools.partial(_tile, name, k, rows, cols, nb, xb))
        outs.append(
            kern(inputs=[*ins, xr], grid=grid, threadgroup=tg, output_shapes=[(nb, rows)], output_dtypes=[xr.dtype])[0]
        )
    y = outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)
    return y if y.dtype == x.dtype else y.astype(x.dtype)


def _dequant(name: str, k: StoredKernel, rows: int, cols: int, ot: str, chain: bool) -> Source:
    _whole(k, cols)
    return Source(
        f"{name}_{rows}_{cols}_{ot}" + ("_chain" if chain else ""),
        [nm for nm, _ in inputs(k)],
        k.defs + device_fns(k),
        f"#define ROWS {rows}\n#define COLS {cols}\n#define OC {chain_cols(cols) if chain else cols}\n"
        f"#define OT {ot}\n#define CHAIN {int(chain)}\n" + DEQUANT,
    )


def dequant(
    name: str, k: StoredKernel, ins: Sequence[mx_.array], rows: int, cols: int, dtype: mx_.Dtype, chain: bool = False
) -> mx_.array:
    """the whole weight in `dtype` (bf16, or float32 for a float32 x's gemm), the tile's weights: [rows, cols], or with
    `chain` [rows, chain_cols(cols)] in the chain's column order (a prefill's gemm over `chain_x`)"""
    m = mx
    ot = "bfloat" if dtype == m.bfloat16 else "float"
    oc = chain_cols(cols) if chain else cols
    kern = compiled((name, rows, cols, ot, chain), lambda: _dequant(name, k, rows, cols, ot, chain))
    n = rows * oc // 8
    return kern(
        inputs=list(ins), grid=(-(-n // 256) * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, oc)],
        output_dtypes=[dtype],
    )[0]  # fmt: skip
