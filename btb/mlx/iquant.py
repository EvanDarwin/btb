# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The IQ quants on Metal: llama.cpp's non-linear 4-bit types (IQ4_NL, IQ4_XS) and the lattice types (grid-codebook
quants) read as stored, 8 consecutive weights of a row decoded in registers (`w8`) through the fixed IQ4 codebook or
the type's grid, so the weights never expand to bf16 in memory on the decode path and the matvec, the dequant and the
megakernel have the same bits (`matvec.StoredKernel`). IQ4_NL: 18 B/32 (d, 16 nibble bytes) on disk, split at bind by
`repack_iq4nl` into an f16 stream and a nibble stream. IQ4_XS: 136 B/256 (d, a 6-bit scale a 32-block, 128 nibble
bytes)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np

from ..kinds import Quant, QuantClass, quant_key, quants_of
from .core import mx
from .matvec import StoredKernel, Table, dequant, launch

if TYPE_CHECKING:
    from collections.abc import Callable

    import mlx.core as mx_

# ggml's IQ lattice codebooks (the grid a quant maps its indices through, and the 128-entry sign table shared by
# the *_XXS/XS/S quants) built from the gguf package's deterministic constants, cached as device buffers the
# kernels read; passing them as inputs keeps the kernel source free of the thousands of literal grid values.
_tables: dict[str, mx_.array] = {}
_tbl_lock = threading.Lock()


def _grid(name: str) -> mx_.array:
    """the int8 grid buffer for a gguf IQ quant class name (e.g. 'IQ3_XXS'), flattened (entries * values)."""
    with _tbl_lock:
        g = _tables.get(name)
        if g is None:
            from gguf import quants

            cls = getattr(quants, name)
            cls.init_grid()
            g = _tables[name] = mx.array(np.ascontiguousarray(cls.grid).reshape(-1).astype(np.int8))
            mx.eval(g)
    return g


def _device(a: mx_.array) -> mx_.array:
    """`a` as an input the sources read through a `device` pointer: MLX hands a kernel an array of fewer than 8
    elements in `constant` memory, so a tiny matrix's one-a-block stream is padded to 8"""
    n = int(a.size)
    return a if n >= 8 else mx.concatenate([a.reshape(-1), mx.zeros((8 - n,), dtype=a.dtype)])


def _ksigns() -> mx_.array:
    """the 128-entry sign table (ggml's ksigns_iq2xs): index -> the 8 sign bits of a group of 8 weights."""
    with _tbl_lock:
        k = _tables.get("ksigns")
        if k is None:
            from gguf import quants

            k = _tables["ksigns"] = mx.array(np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8).copy())
            mx.eval(k)
    return k


# the fixed IQ4 non-linear codebook (ggml's kvalues_iq4nl): a 4-bit index maps to one of these 16 levels.
_KV = "constant int KV[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};\n"

# IQ4_NL: a block of 32 as Dd (an f16 a block) and Qq (16 nibble bytes a block): column j (0..15) the low nibble of
# byte j, column j+16 the high one; value = d * KV[nibble].
_W8_IQ4NL = r"""
    size_t b = (size_t)row * (cols / 32) + k / 32;
    uint i0 = k % 32, sh = i0 < 16 ? 0 : 4;
    float d = (float)Dd[b];
    const device uchar* q = Qq + b * 16 + (i0 & 15);
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = d * (float)KV[(q[t] >> sh) & 0xF];
"""
# IQ4_XS: 136 B a superblock of 256. d f16, scales_h u16, scales_l[4], then 128 nibble bytes. Eight 32-blocks a
# superblock; block ib has a 6-bit scale ls = (scales_l nibble) | (scales_h 2 bits << 4), value = d*(ls-32)*KV[q],
# q the nibble of qs[16*ib + (col&15)] (low half of the block's cols the low nibble, high half the high).
_W8_IQ4XS = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * 136;
    uint c = k % 256, ib = c / 32, bx = c % 32, sh = bx < 16 ? 0 : 4;
    uint shi = (uint)blk[2] | ((uint)blk[3] << 8);
    int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((shi >> (2 * ib)) & 3) << 4);
    float dl = (float)(*(const device half*)blk) * (float)(ls - 32);
    const device uchar* qs = blk + 8 + 16 * ib + (bx & 15);
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = dl * (float)KV[(qs[t] >> sh) & 0xF];
"""

# The lattice types: a 32-block of a superblock of 256 is four groups of 8 weights, 8 columns at k one group
# (ib = c / 32, l = c % 32 / 8), each a grid entry of 8 (or two of 4) signed and scaled by the block's db.
_LATTICE_AT = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * BLKB;
    uint c = k % 256, ib = c / 32, l = c % 32 / 8;
    float d = (float)(*(const device half*)blk);
"""
# IQ3_XXS: 98 B. d f16, 64 grid-index bytes (two 4-value entries a group), then 8 uint32 (one a 32-block: the top 4
# bits a scale, the low 28 four 7-bit sign-table indices); db = d*(0.5 + scale)*0.5.
_W8_IQ3XXS = r"""
    const device uchar* sca = blk + 66 + 4 * ib;
    uint aux = (uint)sca[0] | ((uint)sca[1] << 8) | ((uint)sca[2] << 16) | ((uint)sca[3] << 24);
    float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    uchar signs = KS[(aux >> (7 * l)) & 127];
    uint i1 = blk[2 + 8 * ib + 2 * l], i2 = blk[3 + 8 * ib + 2 * l];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        float v = (float)GRID[(t < 4 ? i1 : i2) * 4 + t % 4];
        w[t] = db * (((signs >> t) & 1) ? -v : v);
    }
"""
# IQ2_XXS: 66 B. d f16, then a 32-block's 8 bytes: word0 4 grid-index bytes (a group each), word1 a 4-bit scale in
# the top nibble and four 7-bit sign-table indices; the (256,8) grid, db = d*(0.5 + scale)*0.25.
_W8_IQ2XXS = r"""
    const device uchar* grp = blk + 2 + 8 * ib;
    uint w1 = (uint)grp[4] | ((uint)grp[5] << 8) | ((uint)grp[6] << 16) | ((uint)grp[7] << 24);
    float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
    uchar signs = KS[(w1 >> (7 * l)) & 127];
    uint gi = grp[l];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) { float v = (float)GRID[gi * 8 + t]; w[t] = db * (((signs >> t) & 1) ? -v : v); }
"""
# IQ2_XS: 74 B. d f16, 32 uint16 qs (a group: low 9 bits a (512,8) grid index, top 7 a sign-table index), 8 scale
# bytes (16 4-bit sub-scales, one a 16-weight half-block); db = d*(0.5 + sc4)*0.25.
_W8_IQ2XS = r"""
    uint g = 4 * ib + l, s = g / 2;
    uint q16 = (uint)blk[2 + 2 * g] | ((uint)blk[3 + 2 * g] << 8);
    float db = d * (0.5f + (float)((blk[66 + s / 2] >> (4 * (s & 1))) & 0xF)) * 0.25f;
    uchar signs = KS[q16 >> 9];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) { float v = (float)GRID[(q16 & 511) * 8 + t]; w[t] = db * (((signs >> t) & 1) ? -v : v); }
"""
# IQ2_S: 82 B. d f16, 32 qs bytes (low 8 index bits a group), 32 sign bytes (a bit a weight), 8 qh bytes (2 high
# index bits a group, into the (1024,8) grid), 8 scale bytes (16 4-bit sub-scales); db = d*(0.5 + sc4)*0.25.
_W8_IQ2S = r"""
    uint g = 4 * ib + l, s = g / 2;
    uint idx = (uint)blk[2 + g] | ((((uint)blk[66 + g / 4] >> (2 * (g & 3))) & 3) << 8);
    float db = d * (0.5f + (float)((blk[74 + s / 2] >> (4 * (s & 1))) & 0xF)) * 0.25f;
    uchar signs = blk[34 + g];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) { float v = (float)GRID[idx * 8 + t]; w[t] = db * (((signs >> t) & 1) ? -v : v); }
"""
# IQ1_S: 50 B. d f16, 32 qs bytes (low 8 index bits), 8 uint16 qh (a 32-block: 3 high index bits a group, a 3-bit
# scale in bits 12-14, a delta sign in bit 15). The (2048,8) ternary grid; a weight is dl*(grid + delta),
# dl = d*(2*scale+1), delta = +-0.125.
_W8_IQ1S = r"""
    uint qhv = (uint)blk[34 + 2 * ib] | ((uint)blk[35 + 2 * ib] << 8);
    uint idx = (uint)blk[2 + 4 * ib + l] | (((qhv >> (3 * l)) & 7) << 8);
    float dl = d * (float)(2 * ((qhv >> 12) & 7) + 1);
    float delta = (qhv & 0x8000) ? -0.125f : 0.125f;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = dl * ((float)GRID[idx * 8 + t] + delta);
"""
# IQ3_S: 110 B. d f16, 64 qs bytes (low 8 index bits, one a group of 4), 8 qh bytes (the 9th index bit as a plane),
# 32 sign bytes (a bit a weight), 4 scale bytes (8 4-bit scales a 32-block). The (512,4) grid; db = d*(1 + 2*sc4).
_W8_IQ3S = r"""
    uint q1 = 8 * ib + 2 * l, q2 = q1 + 1;
    uint i1 = (uint)blk[2 + q1] | ((((uint)blk[66 + q1 / 8] >> (q1 & 7)) & 1) << 8);
    uint i2 = (uint)blk[2 + q2] | ((((uint)blk[66 + q2 / 8] >> (q2 & 7)) & 1) << 8);
    float db = d * (float)(1 + 2 * ((blk[106 + ib / 2] >> (4 * (ib & 1))) & 0xF));
    uchar signs = blk[74 + 4 * ib + l];
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        float v = (float)GRID[(t < 4 ? i1 : i2) * 4 + t % 4];
        w[t] = db * (((signs >> t) & 1) ? -v : v);
    }
"""
# IQ1_M: 56 B, no leading f16: 32 qs bytes (low 8 index bits, a group), 16 qh bytes (a nibble a group: 3 high index
# bits and a delta-sign bit), 8 scale bytes whose top nibbles reassemble d and whose low 12 bits hold sixteen 3-bit
# sub-scales. The kernels read the side streams `_side_iq1m` makes once at bind (Dd: d an f16 a superblock; SB: the
# sub-scales as nibble pairs, 8 B a superblock). The (2048,8) ternary grid; dl = d*(2*sub+1), weight = dl*(grid + delta).
_W8_IQ1M = r"""
    size_t sb = (size_t)row * (cols / 256) + k / 256;
    const device uchar* blk = W + sb * 56;
    uint g = k % 256 / 8, si = g / 2;
    uint sub = (SB[sb * 8 + si / 2] >> (4 * (si & 1))) & 7;
    uchar nib = (blk[32 + g / 2] >> (4 * (g & 1))) & 0xF;
    uint idx = (uint)blk[g] | ((uint)(nib & 7) << 8);
    float delta = (nib & 8) ? -0.125f : 0.125f;
    float dl = (float)Dd[sb] * (float)(2 * sub + 1);
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = dl * ((float)GRID[idx * 8 + t] + delta);
"""


def _side_iq1m(raw: mx_.array) -> tuple[mx_.array, mx_.array]:
    """IQ1_M's side streams from the raw 56-byte superblocks, in the graph (once at bind, or each pass over a slot's
    view): the block f16 d reassembled here rather than in the kernel, and the sixteen 3-bit sub-scales packed as
    nibble pairs (8 B)."""
    m = mx
    scb = raw.reshape(-1, 56)[:, 48:56].reshape(-1).view(m.uint16).reshape(-1, 4).astype(m.uint32)
    dbits = (scb[:, 0] >> 12) | ((scb[:, 1] >> 12) << 4) | ((scb[:, 2] >> 12) << 8) | ((scb[:, 3] >> 12) << 12)
    sub = (scb.reshape(-1, 4, 1) >> m.array([0, 3, 6, 9], dtype=m.uint32).reshape(1, 1, 4)) & 7
    sub = sub.reshape(-1, 16)
    sb = (sub[:, 0::2] | (sub[:, 1::2] << 4)).astype(m.uint8).reshape(-1)
    return dbits.astype(m.uint16).view(m.float16), sb


class Lattice(TypedDict):
    """an IQ lattice kind: its superblock's bytes, its grid's gguf class name, whether it reads the shared sign
    table, its `w8` body (after `_LATTICE_AT`, but IQ1_M's own), and optionally a `side` repack (raw -> the streams
    named `side_names` its kernels read beside the raw blocks)"""

    bytes: int
    grid: str
    ksigns: bool
    w8: str
    side: NotRequired[Callable[[mx_.array], tuple[mx_.array, ...]]]
    side_names: NotRequired[list[str]]


# Adding a quant is one entry here plus a GGUF-name -> kind row in the engine binder
_LATT: dict[str, Lattice] = {
    "iq3xxs": {"bytes": 98, "grid": "IQ3_XXS", "ksigns": True, "w8": _LATTICE_AT + _W8_IQ3XXS},
    "iq2xxs": {"bytes": 66, "grid": "IQ2_XXS", "ksigns": True, "w8": _LATTICE_AT + _W8_IQ2XXS},
    "iq2xs": {"bytes": 74, "grid": "IQ2_XS", "ksigns": True, "w8": _LATTICE_AT + _W8_IQ2XS},
    "iq2s": {"bytes": 82, "grid": "IQ2_S", "ksigns": False, "w8": _LATTICE_AT + _W8_IQ2S},
    "iq1s": {"bytes": 50, "grid": "IQ1_S", "ksigns": False, "w8": _LATTICE_AT + _W8_IQ1S},
    "iq3s": {"bytes": 110, "grid": "IQ3_S", "ksigns": False, "w8": _LATTICE_AT + _W8_IQ3S},
    "iq1m": {
        "bytes": 56,
        "grid": "IQ1_M",
        "ksigns": False,
        "w8": _W8_IQ1M,
        "side": _side_iq1m,
        "side_names": ["Dd", "SB"],
    },
}


def repack_lattice(kind: str, raw: mx_.array) -> tuple[mx_.array, ...]:
    """the side streams `kind`'s kernels read beside the raw bytes (empty for most), built once."""
    side = _LATT[kind].get("side")
    return tuple(side(raw)) if side is not None else ()


def lattice_side(kind: str) -> bool:
    """whether this kind's kernels read side streams `repack_lattice` builds beside its bytes"""
    return _LATT[kind].get("side") is not None


def _codebooks(kind: str) -> tuple[Table, ...]:
    """the codebooks `kind`'s kernels read: its grid, and the sign table where it reads one"""
    spec = _LATT[kind]
    grid = Table("GRID", "int8_t", lambda: _grid(spec["grid"]))
    return (grid, Table("KS", "uint8_t", _ksigns)) if spec["ksigns"] else (grid,)


def _lattice_kernel(kind: str) -> StoredKernel:
    spec = _LATT[kind]
    streams = (("W", "uint8_t"), *((n, "half" if n == "Dd" else "uint8_t") for n in spec.get("side_names", [])))
    return StoredKernel(spec["w8"], 256, defs=f"#define BLKB {spec['bytes']}\n", streams=streams,
                        tables=_codebooks(kind))  # fmt: skip


def repack_iq4nl(raw: mx_.array) -> tuple[mx_.array, mx_.array]:
    """ggml's interleaved 18-byte IQ4_NL blocks split into the two streams the kernels read: the f16 scales (one a
    block) and the nibble bytes (16 a block). In the graph: once at bind, or each pass over a slot's view"""
    b = raw.reshape(-1, 18)
    return b[:, 0:2].reshape(-1).view(mx.float16), b[:, 2:18].reshape(-1)


# every IQ4 and lattice type as every kernel builds on it (`mlx.stored.STORED`): its w8 over its streams (IQ4_NL's
# split scales and nibbles, IQ1_M's side streams beside its blocks) and tables
KERNELS: dict[Quant, StoredKernel] = {
    Quant.IQ4_NL: StoredKernel(_W8_IQ4NL, 32, _KV, streams=(("Dd", "half"), ("Qq", "uint8_t"))),
    Quant.IQ4_XS: StoredKernel(_W8_IQ4XS, 256, _KV),
    **{q: _lattice_kernel(quant_key(q)) for q in quants_of(QuantClass.LATTICE)},
}


def matvec_iq4nl(d: mx_.array, q: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for an IQ4_NL weight as `repack_iq4nl` splits it; x bf16 or float32, y in its
    dtype, any rows (`matvec.launch`)."""
    return launch(f"btb_iq4nl_mv_{rows}_{cols // 32}", KERNELS[Quant.IQ4_NL], [_device(d), q], x, rows, cols)


def dequant_iq4nl(
    d: mx_.array, q: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None, chain: bool = False
) -> mx_.array:
    """the full weight in `dtype` (bf16 by default) for an IQ4_NL tensor from its repacked streams, `chain` in its
    chain's column order (`matvec.dequant`)."""
    k = KERNELS[Quant.IQ4_NL]
    return dequant("btb_iq4nl_dequant", k, [_device(d), q], rows, cols, dtype or mx.bfloat16, chain)


def matvec_iq4xs(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_iq4nl` for IQ4_XS (136 B/256 superblocks, a 6-bit scale a 32-block)."""
    return launch(f"btb_iq4xs_mv_{rows}_{cols // 256}", KERNELS[Quant.IQ4_XS], [w_bytes], x, rows, cols)


def dequant_iq4xs(
    w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None, chain: bool = False
) -> mx_.array:
    """`dequant_iq4nl` for an IQ4_XS tensor."""
    return dequant("btb_iq4xs_dequant", KERNELS[Quant.IQ4_XS], [w_bytes], rows, cols, dtype or mx.bfloat16, chain)


def matvec_lattice(
    kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int, side: tuple[mx_.array, ...] = ()
) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for a raw IQ lattice weight of `kind`, `side` its `repack_lattice` streams; x
    bf16 or float32, y in its dtype, any rows (`matvec.launch`)."""
    k = KERNELS[Quant[_LATT[kind]["grid"]]]
    ins = [w_bytes, *(_device(s) for s in side), *(t.make() for t in k.tables)]
    return launch(f"btb_{kind}_mv_{rows}_{cols // 256}", k, ins, x, rows, cols)


def dequant_lattice(
    kind: str, w_bytes: mx_.array, rows: int, cols: int, side: tuple[mx_.array, ...] = (),
    dtype: mx_.Dtype | None = None, chain: bool = False,
) -> mx_.array:  # fmt: skip
    """the full weight in `dtype` (bf16 by default) for an IQ lattice tensor of `kind`, `side` its `repack_lattice`
    streams, `chain` in its chain's column order (`matvec.dequant`)"""
    k = KERNELS[Quant[_LATT[kind]["grid"]]]
    ins = [w_bytes, *(_device(s) for s in side), *(t.make() for t in k.tables)]
    return dequant(f"btb_{kind}_dequant", k, ins, rows, cols, dtype or mx.bfloat16, chain)
