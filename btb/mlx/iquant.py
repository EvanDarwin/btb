# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""IQ4 on Metal: llama.cpp's non-linear 4-bit quants read as stored and mapped through the fixed IQ4 codebook
in registers, so the weights never expand to bf16 in memory and each output row accumulates in fp32
independently of the batch (row i is bit-identical for a 1-row and a 16-row call). `matvec_iq4nl`/`matvec_iq4xs`
are the linear (`matvec.launch`, the 32 lanes of a simdgroup cooperating on a block for coalesced reads, up to 16
rows a launch reusing the one weight read); `dequant_*` the whole matrix for the rare bf16-copy path. IQ4_NL: 18 B/32
(d, 16 nibble bytes) on disk, split at bind by `repack_iq4nl` into an f16 stream and a nibble stream so a block
is one aligned uint4. IQ4_XS: 136 B/256 (d, a 6-bit scale a 32-block from scales_h/scales_l, 128 nibble bytes)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np

from ..kinds import Quant, QuantClass, quant_key, quants_of
from .core import metal_kernel, mx
from .matvec import SHAPES, Frags, Shapes, StoredKernel, Table, blocks, launch

if TYPE_CHECKING:
    from collections.abc import Callable

    import mlx.core as mx_

    from .core import MetalKernel

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

# IQ4_NL: 18 B a block of 32 on disk (d f16, 16 nibble bytes; col j (0..15) is the low nibble of byte j, col j+16
# the high nibble; value = d * KV[nibble]). Repacked at bind into two streams, Dd (an f16 a block) and Qq (16 bytes
# a block, so a block's nibbles are one aligned uint4): ggml's 18-byte stride straddles cache lines and cannot be
# vector-loaded. lane = block (one lane owns 32 weights: one uint4 + one f16 a block; 32 lanes read 512 contiguous
# bytes). Register-only, batch-invariant. cols = NB*32, walked 32 blocks (BW = 1024 weights) a step.
IQ4NL_FRAGS = Frags(
    "",
    "        uint b = c * 32 + lane;\n        bool ok = b < NB;\n        float dd[RP];\n        uint4 qq[RP];\n",
    r"""
            dd[r] = ok ? (float)Dd[(size_t)row * NB + b] : 0.0f;
            qq[r] = ok ? ((const device uint4*)Qq)[(size_t)row * NB + b] : uint4(0);
""",
    r"""
            if (ok) {
                const device T* xb = xc + lane * 32;  // read in place a row: 32 held in registers spill
                #pragma unroll
                for (int r = 0; r < RP; r++) {
                    uint4 q = qq[r];
                    float p = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) { uint by = (q.x >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[k]      + (float)KV[by >> 4] * (float)xb[16 + k]; }
                    #pragma unroll
                    for (int k = 0; k < 4; k++) { uint by = (q.y >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[4 + k]  + (float)KV[by >> 4] * (float)xb[20 + k]; }
                    #pragma unroll
                    for (int k = 0; k < 4; k++) { uint by = (q.z >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[8 + k]  + (float)KV[by >> 4] * (float)xb[24 + k]; }
                    #pragma unroll
                    for (int k = 0; k < 4; k++) { uint by = (q.w >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[12 + k] + (float)KV[by >> 4] * (float)xb[28 + k]; }
                    acc[r][tr] += dd[r] * p;
                }
            }
""",
)
_MATVEC_IQ4NL = blocks(IQ4NL_FRAGS)
# a 32-block of IQ4_NL as bf16 (or T): the dequant's and a megakernel's embedding row's
_IQ4NL_BLOCK = r"""
template <typename T>
inline void iq4nl_block(float d, uint4 q, device T* o) {
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.x >> (8*k)) & 0xFF; o[k]      = (T)(d * (float)KV[by & 0xF]); o[16 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.y >> (8*k)) & 0xFF; o[4 + k]  = (T)(d * (float)KV[by & 0xF]); o[20 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.z >> (8*k)) & 0xFF; o[8 + k]  = (T)(d * (float)KV[by & 0xF]); o[24 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.w >> (8*k)) & 0xFF; o[12 + k] = (T)(d * (float)KV[by & 0xF]); o[28 + k] = (T)(d * (float)KV[by >> 4]); }
}
"""
_DEQUANT_IQ4NL = r"""
    uint b = thread_position_in_grid.x;
    if (b >= NBLK) return;
    iq4nl_block((float)Dd[b], ((const device uint4*)Qq)[b], out + (size_t)b * 32);
"""

# IQ4_XS: 136 B a superblock of 256. d f16, scales_h u16, scales_l[4], then 128 nibble bytes. Eight 32-blocks a
# superblock; block ib has a 6-bit scale ls = (scales_l nibble) | (scales_h 2 bits << 4), value = d*(ls-32)*KV[q],
# q the nibble of qs[16*ib + (col&15)] (low half of the block's cols the low nibble, high half the high). lane =
# col within a 32-block (0..31). Register-only, batch-invariant. cols = NSB*256.
IQ4XS_FRAGS = Frags(
    "    uint hi = lane >> 4, bx = lane & 15;\n",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 136;
            float d = (float)(*(const device half*)blk);
            uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
            const device uchar* sl = blk + 4;
            const device uchar* qs = blk + 8;
            #pragma unroll
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                float dl = d * (float)(ls - 32);
                uchar byte = qs[16 * ib + bx];
                int q = hi ? (byte >> 4) : (byte & 0xF);
                w[r][ib] = dl * (float)KV[q];
            }
""",
    r"""
            float xv[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[k] = (float)xc[32 * k + lane];
            #pragma unroll
            for (int r = 0; r < RP; r++)
                #pragma unroll
                for (int k = 0; k < 8; k++) acc[r][tr] += w[r][k] * xv[k];
""",
)
_MATVEC_IQ4XS = blocks(IQ4XS_FRAGS)
# a lane (one column of each of the eight 32-blocks) of an IQ4_XS superblock `blk` as T: the dequant's and a
# megakernel's embedding row's
_IQ4XS_LANE = r"""
template <typename T>
inline void iq4xs_lane(const device uchar* blk, uint lane, device T* o) {
    uint hi = lane >> 4, bx = lane & 15;
    float d = (float)(*(const device half*)blk);
    uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
    const device uchar* sl = blk + 4;
    const device uchar* qs = blk + 8;
    for (int ib = 0; ib < 8; ib++) {
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uchar byte = qs[16 * ib + bx];
        int q = hi ? (byte >> 4) : (byte & 0xF);
        o[32 * ib + lane] = (T)(dl * (float)KV[q]);
    }
}
"""
_DEQUANT_IQ4XS = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    iq4xs_lane(W + (size_t)blk_i * 136, lane, out + (size_t)blk_i * 256);
"""
_lock = threading.Lock()
_deq: dict[str, MetalKernel] = {}
# IQ4_NL's launch shapes and the lattice kernels' (`matvec.Shapes`), measured as the default's over their own types
IQ4NL_SHAPES: Shapes = {
    **{t: (4, t) for t in range(1, 17)},
    6: (2, 6),
    12: (4, 4),
    15: (4, 16),
    16: (4, 4),
}
LATTICE_SHAPES: Shapes = {**SHAPES, 1: (2, 1)}
# (block bytes, block weights, matvec source, dequant source, the device functions it calls) a kind
_IQ = {
    "iq4xs": (136, 256, _MATVEC_IQ4XS, _DEQUANT_IQ4XS, _IQ4XS_LANE),
}


def _matvec(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    _, blkw, src, _, _ = _IQ[kind]
    nb = cols // blkw
    return launch(f"btb_{kind}_mv_{rows}_{nb}", src, _KV, [("W", w_bytes)], x, rows, cols, blkw)


def _dequant(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    m = mx
    _, blkw, _, src, fns = _IQ[kind]
    nblk = int(rows * cols // blkw)
    with _lock:
        k = _deq.get(kind)
        if k is None:
            k = _deq[kind] = metal_kernel(
                name=f"btb_{kind}_dequant", input_names=["W"], output_names=["out"], header=_KV + fns, source=src
            )
    grid = ((nblk * 32 + 255) // 256) * 256
    return k(
        inputs=[w_bytes],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


_iq4nl_deq = None


def repack_iq4nl(raw: mx_.array) -> tuple[mx_.array, mx_.array]:
    """ggml's interleaved 18-byte IQ4_NL blocks split into the two streams the kernels read: the f16 scales (one a
    block) and the nibble bytes (16 a block, so each block is one aligned uint4). In the graph: once at bind, or
    each pass over a slot's view"""
    b = raw.reshape(-1, 18)
    return b[:, 0:2].reshape(-1).view(mx.float16), b[:, 2:18].reshape(-1)


def matvec_iq4nl(d: mx_.array, q: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for an IQ4_NL weight as `repack_iq4nl` splits it; fp32 accumulation, y in
    x's dtype, 1..16 rows a launch."""
    nb = cols // 32
    return launch(
        f"btb_iq4nl_mv_{rows}_{nb}",
        _MATVEC_IQ4NL,
        f"{_KV}#define NB {nb}\n",
        [("Dd", _device(d)), ("Qq", q)],
        x,
        rows,
        cols,
        1024,
        IQ4NL_SHAPES,
    )


def dequant_iq4nl(d: mx_.array, q: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for an IQ4_NL tensor from its repacked streams."""
    global _iq4nl_deq
    m = mx
    nblk = int(rows * cols // 32)
    with _lock:
        if _iq4nl_deq is None:
            _iq4nl_deq = metal_kernel(
                name="btb_iq4nl_dequant",
                input_names=["Dd", "Qq"],
                output_names=["out"],
                header=_KV + _IQ4NL_BLOCK,
                source=_DEQUANT_IQ4NL,
            )
    grid = ((nblk + 255) // 256) * 256
    return _iq4nl_deq(
        inputs=[_device(d), q],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


def matvec_iq4xs(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_iq4nl` for IQ4_XS (136 B/256 superblocks, a 6-bit scale a 32-block)."""
    return _matvec("iq4xs", w_bytes, x, rows, cols)


def dequant_iq4xs(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for an IQ4_XS tensor."""
    return _dequant("iq4xs", w_bytes, rows, cols)


# IQ3_XXS: 98 B a superblock of 256. d f16, 64 grid-index bytes, then 8 uint32 (one a 32-block: the top 4 bits a
# scale, the low 28 four 7-bit sign-table indices). Eight 32-blocks a superblock, four groups of 8 a block; a
# group's two grid indices give 4 values each through the (256,4) grid, signed by the group's sign byte, scaled
# by db = d*(0.5 + scale)*0.5. lane = col within a 32-block (0..31), l = lane>>3 the group, m = lane&7 the value.
# Register-only, batch-invariant. cols = NSB*256.
# Each lattice linear (`matvec.blocks`): a lane owns a group of 8 weights at cb = 8*lane, decoded to g[r][8] and a
# scale db[r]; a pass row's eight activations go through them as p = sum g*x, then acc += db*p.
_DECL_G8 = "        float g[RP][8], db[RP];\n"
_APPLY_G8 = r"""
            float xv[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[k] = (float)xc[cb + k];
            #pragma unroll
            for (int r = 0; r < RP; r++) {
                float p = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) p += g[r][k] * xv[k];
                acc[r][tr] += db[r] * p;
            }
"""
IQ3XXS_FRAGS = Frags(
    "    uint ib = lane >> 2, l = lane & 3, cb = 8 * lane;   // two 4-value grid entries a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 98;
            float d = (float)(*(const device half*)blk);
            const device uchar* qs = blk + 2;
            const device uchar* sca = blk + 66;
            uint aux = (uint)sca[4*ib] | ((uint)sca[4*ib+1]<<8) | ((uint)sca[4*ib+2]<<16) | ((uint)sca[4*ib+3]<<24);
            db[r] = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            uchar signs = KS[(aux >> (7 * l)) & 127];
            uint i1 = qs[8 * ib + 2 * l], i2 = qs[8 * ib + 2 * l + 1];
            #pragma unroll
            for (int j = 0; j < 4; j++) { float v = (float)GRID[i1 * 4 + j]; g[r][j] = (signs & (1 << j)) ? -v : v; }
            #pragma unroll
            for (int j = 0; j < 4; j++) { float v = (float)GRID[i2 * 4 + j]; g[r][4 + j] = (signs & (1 << (j + 4))) ? -v : v; }
""",
    _APPLY_G8,
)
# each lattice kind's lane of a superblock `blk` as T (one column of each of its eight 32-blocks): the dequant's and a
# megakernel's embedding row's, over the kind's grid (and the sign table where it reads one)
_IQ3XXS_LANE = r"""
template <typename T>
inline void iq3xxs_lane(const device uchar* blk, uint lane, const device int8_t* GRID, const device uint8_t* KS,
                        device T* o) {
    uint l = lane >> 3, m = lane & 7, qsoff = 2 * l + (m >= 4 ? 1 : 0), gj = m & 3;
    uchar smask = 1 << m;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2;
    const device uchar* sca = blk + 66;
    for (int ib = 0; ib < 8; ib++) {
        uint aux = (uint)sca[4*ib] | ((uint)sca[4*ib+1]<<8) | ((uint)sca[4*ib+2]<<16) | ((uint)sca[4*ib+3]<<24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        uchar idx = qs[8 * ib + qsoff];
        float gval = (float)GRID[(uint)idx * 4 + gj];
        uchar signs = KS[(aux >> (7 * l)) & 127];
        o[32 * ib + lane] = (T)(db * gval * ((signs & smask) ? -1.0f : 1.0f));
    }
}
"""
# IQ2_XXS: 66 B a superblock of 256. d f16, then 8 interleaved groups of 8 bytes (word0 = 4 grid-index bytes,
# word1 = a 4-bit scale in the top nibble and four 7-bit sign-table indices below). A group is a 32-block, four
# sub-groups of 8 a block; the (256,8) grid gives 8 values a sub-group, signed by the group's sign byte, scaled by
# db = d*(0.5 + scale)*0.25. lane = col within a 32-block, l = lane>>3 the sub-group, j = lane&7 the value.
IQ2XXS_FRAGS = Frags(
    "    uint ig = lane >> 2, sub = lane & 3, cb = 8 * lane;   // one grid entry a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 66;
            float d = (float)(*(const device half*)blk);
            const device uchar* grp = blk + 2 + 8 * ig;
            uint w1 = (uint)grp[4] | ((uint)grp[5]<<8) | ((uint)grp[6]<<16) | ((uint)grp[7]<<24);
            db[r] = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
            uchar signs = KS[(w1 >> (7 * sub)) & 127];
            uint gi = grp[sub];
            #pragma unroll
            for (int k = 0; k < 8; k++) { float v = (float)GRID[gi * 8 + k]; g[r][k] = (signs & (1 << k)) ? -v : v; }
""",
    _APPLY_G8,
)
_IQ2XXS_LANE = r"""
template <typename T>
inline void iq2xxs_lane(const device uchar* blk, uint lane, const device int8_t* GRID, const device uint8_t* KS,
                        device T* o) {
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2;
    for (int ib = 0; ib < 8; ib++) {
        const device uchar* grp = qs + 8 * ib;
        uint w1 = (uint)grp[4] | ((uint)grp[5]<<8) | ((uint)grp[6]<<16) | ((uint)grp[7]<<24);
        float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
        float gval = (float)GRID[(uint)grp[l] * 8 + j];
        uchar signs = KS[(w1 >> (7 * l)) & 127];
        o[32 * ib + lane] = (T)(db * gval * ((signs & jmask) ? -1.0f : 1.0f));
    }
}
"""
# IQ2_XS: 74 B a superblock. d f16, 32 uint16 qs (a group of 8: low 9 bits a (512,8) grid index, top 7 bits a
# sign-table index), 8 scale bytes (16 4-bit sub-scales, one a 16-weight half-block). db = d*(0.5 + sc4)*0.25.
IQ2XS_FRAGS = Frags(
    "    uint s = lane >> 1, cb = 8 * lane;   // one uint16 grid entry a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 74;
            float d = (float)(*(const device half*)blk);
            const device uchar* p = blk + 2 + 2 * lane;
            const device uchar* sc = blk + 66;
            uint q16 = (uint)p[0] | ((uint)p[1] << 8);
            uint sc4 = (sc[s >> 1] >> (4 * (s & 1))) & 0xF;
            db[r] = d * (0.5f + (float)sc4) * 0.25f;
            uchar signs = KS[q16 >> 9];
            uint gi = q16 & 511;
            #pragma unroll
            for (int k = 0; k < 8; k++) { float v = (float)GRID[gi * 8 + k]; g[r][k] = (signs & (1 << k)) ? -v : v; }
""",
    _APPLY_G8,
)
_IQ2XS_LANE = r"""
template <typename T>
inline void iq2xs_lane(const device uchar* blk, uint lane, const device int8_t* GRID, const device uint8_t* KS,
                       device T* o) {
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* sc = blk + 66;
    for (int ib = 0; ib < 8; ib++) {
        uint grp = 4 * ib + l;
        const device uchar* p = qs + 2 * grp;
        uint q16 = (uint)p[0] | ((uint)p[1] << 8);
        uint s = grp >> 1; uint sc4 = (sc[s >> 1] >> (4 * (s & 1))) & 0xF;
        float db = d * (0.5f + (float)sc4) * 0.25f;
        float gval = (float)GRID[(q16 & 511) * 8 + j];
        uchar signs = KS[q16 >> 9];
        o[32 * ib + lane] = (T)(db * gval * ((signs & jmask) ? -1.0f : 1.0f));
    }
}
"""
# IQ2_S: 82 B a superblock. d f16, 32 qs bytes (low 8 index bits), 32 explicit sign bytes (a bit a weight), 8 qh
# bytes (2 high index bits a group, into the (1024,8) grid), 8 scale bytes (16 4-bit sub-scales). db = d*(0.5+sc4)*0.25.
IQ2S_FRAGS = Frags(
    "    uint b = lane >> 1, cb = 8 * lane;\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 82;
            float d = (float)(*(const device half*)blk);
            const device uchar* qs = blk + 2;
            const device uchar* sgn = blk + 34;
            const device uchar* qh = blk + 66;
            const device uchar* sc = blk + 74;
            uint idx = (uint)qs[lane] | ((((uint)qh[lane >> 2] >> (2 * (lane & 3))) & 3) << 8);
            uint sc4 = (sc[b >> 1] >> (4 * (b & 1))) & 0xF;
            db[r] = d * (0.5f + (float)sc4) * 0.25f;
            uchar signs = sgn[lane];
            #pragma unroll
            for (int k = 0; k < 8; k++) { float v = (float)GRID[idx * 8 + k]; g[r][k] = (signs & (1 << k)) ? -v : v; }
""",
    _APPLY_G8,
)
_IQ2S_LANE = r"""
template <typename T>
inline void iq2s_lane(const device uchar* blk, uint lane, const device int8_t* GRID, device T* o) {
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* sgn = blk + 34;
    const device uchar* qh = blk + 66; const device uchar* sc = blk + 74;
    for (int ib = 0; ib < 8; ib++) {
        uint grp = 4 * ib + l;
        uint idx = (uint)qs[grp] | ((((uint)qh[grp >> 2] >> (2 * (grp & 3))) & 3) << 8);
        uint b = grp >> 1; uint sc4 = (sc[b >> 1] >> (4 * (b & 1))) & 0xF;
        float db = d * (0.5f + (float)sc4) * 0.25f;
        float gval = (float)GRID[idx * 8 + j];
        o[32 * ib + lane] = (T)(db * gval * ((sgn[grp] & jmask) ? -1.0f : 1.0f));
    }
}
"""
# IQ1_S: 50 B a superblock. d f16, 32 qs bytes (low 8 index bits), 8 uint16 qh (a 32-block: 3 high index bits a
# sub-block of 8, a 3-bit scale in bits 12-14, a delta sign in bit 15). The (2048,8) grid is ternary; a weight is
# dl*(grid + delta), dl = d*(2*scale+1), delta = +-0.125. No sign table.
IQ1S_FRAGS = Frags(
    "    uint ib = lane >> 2, l = lane & 3, cb = 8 * lane;   // one ternary grid entry a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 50;
            float d = (float)(*(const device half*)blk);
            const device uchar* qs = blk + 2;
            const device uchar* p = blk + 34 + 2 * ib;
            uint qhv = (uint)p[0] | ((uint)p[1] << 8);
            uint idx = (uint)qs[lane] | (((qhv >> (3 * l)) & 7) << 8);
            db[r] = d * (float)(2 * ((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) ? -0.125f : 0.125f;
            #pragma unroll
            for (int k = 0; k < 8; k++) g[r][k] = (float)GRID[idx * 8 + k] + delta;
""",
    _APPLY_G8,
)
_IQ1S_LANE = r"""
template <typename T>
inline void iq1s_lane(const device uchar* blk, uint lane, const device int8_t* GRID, device T* o) {
    uint l = lane >> 3, j = lane & 7;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* qhb = blk + 34;
    for (int ib = 0; ib < 8; ib++) {
        const device uchar* p = qhb + 2 * ib;
        uint qhv = (uint)p[0] | ((uint)p[1] << 8);
        uint idx = (uint)qs[ib * 4 + l] | ((((qhv >> (3 * l)) & 7)) << 8);
        float dl = d * (float)(2 * ((qhv >> 12) & 7) + 1);
        float delta = (qhv & 0x8000) ? -0.125f : 0.125f;
        o[32 * ib + lane] = (T)(dl * ((float)GRID[idx * 8 + j] + delta));
    }
}
"""

# IQ3_S: 110 B a superblock. d f16, 64 qs bytes (low 8 index bits, one a group of 4 weights), 8 qh bytes (the 9th
# index bit as a plane), 32 explicit sign bytes (a bit a weight), 4 scale bytes (8 4-bit scales a 32-block). The
# (512,4) grid gives 4 values a group; db = d*(1 + 2*sc4). lane = col within a 32-block, qb = lane>>2 the group.
IQ3S_FRAGS = Frags(
    "    uint ib = lane >> 2, cb = 8 * lane, qb1 = 2 * lane, qb2 = 2 * lane + 1;   // two 4-value entries a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 110;
            float d = (float)(*(const device half*)blk);
            const device uchar* qs = blk + 2;
            const device uchar* qh = blk + 66;
            const device uchar* sgn = blk + 74;
            const device uchar* sc = blk + 106;
            uint i1 = (uint)qs[qb1] | ((((uint)qh[qb1 >> 3] >> (qb1 & 7)) & 1) << 8);
            uint i2 = (uint)qs[qb2] | ((((uint)qh[qb2 >> 3] >> (qb2 & 7)) & 1) << 8);
            uint sc4 = (sc[ib >> 1] >> (4 * (ib & 1))) & 0xF;
            db[r] = d * (float)(1 + 2 * sc4);
            uchar signs = sgn[lane];
            #pragma unroll
            for (int j = 0; j < 4; j++) { float v = (float)GRID[i1 * 4 + j]; g[r][j] = (signs & (1 << j)) ? -v : v; }
            #pragma unroll
            for (int j = 0; j < 4; j++) { float v = (float)GRID[i2 * 4 + j]; g[r][4 + j] = (signs & (1 << (j + 4))) ? -v : v; }
""",
    _APPLY_G8,
)
_IQ3S_LANE = r"""
template <typename T>
inline void iq3s_lane(const device uchar* blk, uint lane, const device int8_t* GRID, device T* o) {
    uint ql = lane >> 2, j = lane & 3, sl = lane >> 3; uchar sbit = 1 << (lane & 7);
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* qh = blk + 66;
    const device uchar* sgn = blk + 74; const device uchar* sc = blk + 106;
    for (int ib = 0; ib < 8; ib++) {
        uint qb = 8 * ib + ql;
        uint idx = (uint)qs[qb] | ((((uint)qh[qb >> 3] >> (qb & 7)) & 1) << 8);
        uint sc4 = (sc[ib >> 1] >> (4 * (ib & 1))) & 0xF;
        float db = d * (float)(1 + 2 * sc4);
        float gval = (float)GRID[idx * 4 + j];
        o[32 * ib + lane] = (T)(db * gval * ((sgn[4 * ib + sl] & sbit) ? -1.0f : 1.0f));
    }
}
"""

# IQ1_M: 56 B a superblock, no leading f16. 32 qs bytes (low 8 index bits, a group of 8), 16 qh bytes (a nibble a
# group: 3 high index bits and a delta-sign bit), 8 scale bytes as four uint16 whose top nibbles reassemble the
# block's f16 d and whose low 12 bits hold sixteen 3-bit sub-scales. The (2048,8) ternary grid (shared with IQ1_S);
# dl = d*(2*sub+1), delta = +-0.125, weight = dl*(grid + delta). The matvec reads a side stream built once at bind
# (`_side_iq1m`: the f16 d precomputed, the sub-scales as nibble pairs) instead of reassembling d and selecting a
# sub-scale through four uint16 every superblock; +10 bytes a superblock, 1.3x faster. lane = group of 8.
IQ1M_FRAGS = Frags(
    "    uint si = lane >> 1, cb = 8 * lane;   // one ternary grid entry a lane\n",
    _DECL_G8,
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 56;
            const device uchar* qs = blk;
            const device uchar* qh = blk + 32;
            float d = (float)Dd[(size_t)row * NSB + c];
            uint sub = (SB[((size_t)row * NSB + c) * 8 + (si >> 1)] >> (4 * (si & 1))) & 7;
            uchar nib = (qh[lane >> 1] >> (4 * (lane & 1))) & 0xF;
            uint idx = (uint)qs[lane] | (((uint)(nib & 7)) << 8);
            float delta = (nib & 8) ? -0.125f : 0.125f;
            db[r] = d * (float)(2 * sub + 1);
            #pragma unroll
            for (int k = 0; k < 8; k++) g[r][k] = (float)GRID[idx * 8 + k] + delta;
""",
    _APPLY_G8,
)
_IQ1M_LANE = r"""
template <typename T>
inline void iq1m_lane(const device uchar* blk, uint lane, const device int8_t* GRID, device T* o) {
    uint j8 = lane >> 3, e = lane & 7, sl = lane >> 4;
    const device uchar* qs = blk; const device uchar* qh = blk + 32; const device uchar* scb = blk + 48;
    uint s0 = (uint)scb[0]|((uint)scb[1]<<8), s1 = (uint)scb[2]|((uint)scb[3]<<8);
    uint s2 = (uint)scb[4]|((uint)scb[5]<<8), s3 = (uint)scb[6]|((uint)scb[7]<<8);
    ushort dbits = (ushort)((s0>>12) | ((s1>>12)<<4) | ((s2>>12)<<8) | ((s3>>12)<<12));
    float d = (float)as_type<half>(dbits);
    for (int ib = 0; ib < 8; ib++) {
        uint j = 4 * ib + j8;
        uchar nib = (qh[j >> 1] >> (4 * (j & 1))) & 0xF;
        uint idx = (uint)qs[j] | (((uint)(nib & 7)) << 8);
        float delta = (nib & 8) ? -0.125f : 0.125f;
        uint si = 2 * ib + sl;
        uint s16 = si < 4 ? s0 : (si < 8 ? s1 : (si < 12 ? s2 : s3));
        uint sub = (s16 >> (3 * (si & 3))) & 7;
        float dl = d * (float)(2 * sub + 1);
        o[32 * ib + lane] = (T)(dl * ((float)GRID[idx * 8 + e] + delta));
    }
}
"""


def _side_iq1m(raw: mx_.array) -> tuple[mx_.array, mx_.array]:
    """IQ1_M's matvec side stream from the raw 56-byte superblocks, in the graph (once at bind, or each pass over a
    slot's view): the block f16 d reassembled here rather than in the kernel, and the sixteen 3-bit sub-scales
    packed as nibble pairs (8 B)."""
    m = mx
    scb = raw.reshape(-1, 56)[:, 48:56].reshape(-1).view(m.uint16).reshape(-1, 4).astype(m.uint32)
    dbits = (scb[:, 0] >> 12) | ((scb[:, 1] >> 12) << 4) | ((scb[:, 2] >> 12) << 8) | ((scb[:, 3] >> 12) << 12)
    sub = (scb.reshape(-1, 4, 1) >> m.array([0, 3, 6, 9], dtype=m.uint32).reshape(1, 1, 4)) & 7
    sub = sub.reshape(-1, 16)
    sb = (sub[:, 0::2] | (sub[:, 1::2] << 4)).astype(m.uint8).reshape(-1)
    return dbits.astype(m.uint16).view(m.float16), sb


# the IQ lattice family (grid-codebook quants) on one launcher: each entry is the superblock byte size, the grid
# class name, whether the shared sign table is read, the matvec's pieces, its lane's dequant (`<kind>_lane`), and
# optionally a `side` repack (raw -> extra device streams named `side_names`) the matvec reads beside the raw blocks.
# A kernel always takes W, x and GRID; KS (the sign table) only when the entry reads it. Adding a quant is one entry
# here plus a GGUF-name -> kind row in the engine binder. All are register-only and batch-invariant (row i identical
# at any T).
class Lattice(TypedDict):
    """an IQ lattice kind's entry of `_LATT`"""

    bytes: int
    grid: str
    ksigns: bool
    frags: Frags
    lane: str
    side: NotRequired[Callable[[mx_.array], tuple[mx_.array, ...]]]
    side_names: NotRequired[list[str]]


_LATT: dict[str, Lattice] = {
    "iq3xxs": {"bytes": 98, "grid": "IQ3_XXS", "ksigns": True, "frags": IQ3XXS_FRAGS, "lane": _IQ3XXS_LANE},
    "iq2xxs": {"bytes": 66, "grid": "IQ2_XXS", "ksigns": True, "frags": IQ2XXS_FRAGS, "lane": _IQ2XXS_LANE},
    "iq2xs": {"bytes": 74, "grid": "IQ2_XS", "ksigns": True, "frags": IQ2XS_FRAGS, "lane": _IQ2XS_LANE},
    "iq2s": {"bytes": 82, "grid": "IQ2_S", "ksigns": False, "frags": IQ2S_FRAGS, "lane": _IQ2S_LANE},
    "iq1s": {"bytes": 50, "grid": "IQ1_S", "ksigns": False, "frags": IQ1S_FRAGS, "lane": _IQ1S_LANE},
    "iq3s": {"bytes": 110, "grid": "IQ3_S", "ksigns": False, "frags": IQ3S_FRAGS, "lane": _IQ3S_LANE},
    "iq1m": {
        "bytes": 56,
        "grid": "IQ1_M",
        "ksigns": False,
        "frags": IQ1M_FRAGS,
        "lane": _IQ1M_LANE,
        "side": _side_iq1m,
        "side_names": ["Dd", "SB"],
    },
}


def _lane_call(kind: str, blk: str, lane: str, out: str) -> str:
    """the call of `kind`'s lane dequant on block pointer `blk`, lane `lane`, into `out`"""
    tables = "GRID, KS" if _LATT[kind]["ksigns"] else "GRID"
    return f"{kind}_lane({blk}, {lane}, {tables}, {out})"


def _lattice_dequant(kind: str) -> str:
    """`kind`'s dequant kernel: a (superblock, lane) a thread through its lane function"""
    b = _LATT[kind]["bytes"]
    return (
        "    uint gid = thread_position_in_grid.x;\n    uint blk_i = gid / 32, lane = gid % 32;\n"
        "    if (blk_i >= NBLK) return;\n"
        f"    {_lane_call(kind, f'W + (size_t)blk_i * {b}', 'lane', 'out + (size_t)blk_i * 256')};\n"
    )


def _lattice_decode(kind: str) -> str:
    """a row of a `kind` matrix in bf16 for a megakernel's embedding, a (superblock, lane) a thread as the dequant
    decodes it"""
    b = _LATT[kind]["bytes"]
    blk = f"W + ((size_t)row * NSB + i / 32) * {b}"
    return f"    for (uint i = tid; i < NSB * 32; i += 256)\n        {_lane_call(kind, blk, 'i % 32', 'dst + (size_t)(i / 32) * 256')};\n"


# each lattice kind's matvec source, built once from its pieces
_LATT_MV: dict[str, str] = {k: blocks(v["frags"]) for k, v in _LATT.items()}


def repack_lattice(kind: str, raw: mx_.array) -> tuple[mx_.array, ...]:
    """the side streams `matvec_lattice` wants beside the raw bytes for this kind (empty for most), built once."""
    side = _LATT[kind].get("side")
    return tuple(side(raw)) if side is not None else ()


def lattice_side(kind: str) -> bool:
    """whether this kind's matvec reads side streams `repack_lattice` builds beside its bytes"""
    return _LATT[kind].get("side") is not None


_latt_deq: dict[str, MetalKernel] = {}


def matvec_lattice(
    kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int, side: tuple[mx_.array, ...] = ()
) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for a raw IQ lattice weight; fp32 accumulation, y in x's dtype, 1..16 rows a
    launch. `kind` selects the layout from `_LATT`; the grid (and sign table, when used) ride along as buffers,
    and `side` is this kind's `repack_lattice` streams when it has any."""
    spec = _LATT[kind]
    tables = [("GRID", _grid(spec["grid"]))] + ([("KS", _ksigns())] if spec["ksigns"] else [])
    sides = [(n, _device(s)) for n, s in zip(spec.get("side_names", []), side, strict=True)]
    nsb = cols // 256
    return launch(
        f"btb_{kind}_mv_{rows}_{nsb}",
        _LATT_MV[kind],
        "",
        [("W", w_bytes), *tables, *sides],
        x,
        rows,
        cols,
        256,
        LATTICE_SHAPES,
    )


def dequant_lattice(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for an IQ lattice tensor of the given `kind`."""
    m = mx
    spec = _LATT[kind]
    inames = ["W", "GRID"] + (["KS"] if spec["ksigns"] else [])
    tbls = [_grid(spec["grid"])] + ([_ksigns()] if spec["ksigns"] else [])
    nblk = int(rows * cols // 256)
    with _lock:
        k = _latt_deq.get(kind)
        if k is None:
            k = _latt_deq[kind] = metal_kernel(
                name=f"btb_{kind}_dequant",
                input_names=inames,
                output_names=["out"],
                header=spec["lane"],
                source=_lattice_dequant(kind),
            )
    g = ((nblk * 32 + 255) // 256) * 256
    return k(
        inputs=[w_bytes, *tbls],
        grid=(g, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


def _codebooks(kind: str) -> tuple[Table, ...]:
    """the codebooks `kind`'s kernels read: its grid, and the sign table where it reads one"""
    spec = _LATT[kind]
    grid = Table("GRID", "int8_t", lambda: _grid(spec["grid"]))
    return (grid, Table("KS", "uint8_t", _ksigns)) if spec["ksigns"] else (grid,)


def _lattice_kernel(kind: str) -> StoredKernel:
    spec = _LATT[kind]
    streams = (("W", "uint8_t"), *((n, "half" if n == "Dd" else "uint8_t") for n in spec.get("side_names", [])))
    decode = _lattice_decode(kind)
    return StoredKernel(spec["frags"], spec["lane"], "", 256, decode, LATTICE_SHAPES, streams, _codebooks(kind))


_IQ4NL_DECODE = (
    "    for (uint b = tid; b < COLS / 32; b += 256)\n"
    "        iq4nl_block((float)Dd[(size_t)row * (COLS / 32) + b], ((const device uint4*)Qq)[(size_t)row * (COLS / 32) + b],"
    " dst + (size_t)b * 32);\n"
)
_IQ4XS_DECODE = (
    "    for (uint i = tid; i < NSB * 32; i += 256)\n"
    "        iq4xs_lane(W + ((size_t)row * NSB + i / 32) * 136, i % 32, dst + (size_t)(i / 32) * 256);\n"
)
# every IQ4 and lattice type as the megakernel builds on it (`mlx.mega.kernel.STORED`): its matvec's pieces over its
# streams (IQ4_NL's split scales and nibbles, IQ1_M's side streams beside its blocks) and tables, its dequant a row of
# an embedding
KERNELS: dict[Quant, StoredKernel] = {
    Quant.IQ4_NL: StoredKernel(
        IQ4NL_FRAGS,
        _KV + _IQ4NL_BLOCK,
        "#define NB (COLS / 32)\n",
        1024,
        _IQ4NL_DECODE,
        IQ4NL_SHAPES,
        (("Dd", "half"), ("Qq", "uint8_t")),
    ),
    Quant.IQ4_XS: StoredKernel(IQ4XS_FRAGS, _KV + _IQ4XS_LANE, "", 256, _IQ4XS_DECODE),
    **{q: _lattice_kernel(quant_key(q)) for q in quants_of(QuantClass.LATTICE)},
}
