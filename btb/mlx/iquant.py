# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""IQ4 on Metal: llama.cpp's non-linear 4-bit quants read as stored and mapped through the fixed IQ4 codebook
in registers, so the weights never expand to bf16 in memory and each output row accumulates in fp32
independently of the batch (row i is bit-identical for a 1-row and a 16-row call). `matvec_iq4nl`/`matvec_iq4xs`
are the linear (one simdgroup a row, the 32 lanes cooperating on a block for coalesced reads, up to 16 rows a
launch reusing the one weight read); `dequant_*` the whole matrix for the rare bf16-copy path. IQ4_NL: 18 B/32
(d, 16 nibble bytes) on disk, split at bind by `repack_iq4nl` into an f16 stream and a nibble stream so a block
is one aligned uint4. IQ4_XS: 136 B/256 (d, a 6-bit scale a 32-block from scales_h/scales_l, 128 nibble bytes)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_

# ggml's IQ lattice codebooks (the grid a quant maps its indices through, and the 128-entry sign table shared by
# the *_XXS/XS/S quants) built from the gguf package's deterministic constants, cached as device buffers the
# kernels read; passing them as inputs keeps the kernel source free of the thousands of literal grid values.
_tables: dict[str, Any] = {}
_tbl_lock = threading.Lock()


def _grid(name: str) -> Any:
    """the int8 grid buffer for a gguf IQ quant class name (e.g. 'IQ3_XXS'), flattened (entries * values)."""
    with _tbl_lock:
        g = _tables.get(name)
        if g is None:
            from gguf import quants

            cls = getattr(quants, name)
            cls.init_grid()
            g = _tables[name] = mx().array(np.ascontiguousarray(cls.grid).reshape(-1).astype(np.int8))
            mx().eval(g)
    return g


def _ksigns() -> Any:
    """the 128-entry sign table (ggml's ksigns_iq2xs): index -> the 8 sign bits of a group of 8 weights."""
    with _tbl_lock:
        k = _tables.get("ksigns")
        if k is None:
            from gguf import quants

            k = _tables["ksigns"] = mx().array(np.frombuffer(quants.IQ2_XXS.ksigns, dtype=np.uint8).copy())
            mx().eval(k)
    return k

# the fixed IQ4 non-linear codebook (ggml's kvalues_iq4nl): a 4-bit index maps to one of these 16 levels.
_KV = "constant int KV[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};\n"

# IQ4_NL: 18 B a block of 32 on disk (d f16, 16 nibble bytes; col j (0..15) is the low nibble of byte j, col j+16
# the high nibble; value = d * KV[nibble]). Repacked at bind into two streams, Dd (an f16 a block) and Qq (16 bytes
# a block, so a block's nibbles are one aligned uint4): ggml's 18-byte stride straddles cache lines and cannot be
# vector-loaded. lane = block (one lane owns 32 weights: one uint4 + one f16 a block; 32 lanes read 512 contiguous
# bytes). Register-only, batch-invariant. cols = NB*32.
_MATVEC_IQ4NL = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, lane = gid % 32;
    if (row >= ROWS) return;
    const device half* Drow = Dd + (size_t)row * NB;
    const device uint4* Qrow = (const device uint4*)Qq + (size_t)row * NB;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c0 = 0; c0 < NB; c0 += 32) {
        uint b = c0 + lane;
        if (b < NB) {
            float d = (float)Drow[b];
            uint4 q = Qrow[b];
            for (int tr = 0; tr < TR; tr++) {
                const device T* xb = x + (size_t)tr * (NB * 32) + (size_t)b * 32;
                float p = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; k++) { uint by = (q.x >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[k]      + (float)KV[by >> 4] * (float)xb[16 + k]; }
                #pragma unroll
                for (int k = 0; k < 4; k++) { uint by = (q.y >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[4 + k]  + (float)KV[by >> 4] * (float)xb[20 + k]; }
                #pragma unroll
                for (int k = 0; k < 4; k++) { uint by = (q.z >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[8 + k]  + (float)KV[by >> 4] * (float)xb[24 + k]; }
                #pragma unroll
                for (int k = 0; k < 4; k++) { uint by = (q.w >> (8*k)) & 0xFF; p += (float)KV[by & 0xF] * (float)xb[12 + k] + (float)KV[by >> 4] * (float)xb[28 + k]; }
                acc[tr] += d * p;
            }
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (lane == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ4NL = r"""
    uint b = thread_position_in_grid.x;
    if (b >= NBLK) return;
    float d = (float)Dd[b];
    uint4 q = ((const device uint4*)Qq)[b];
    device T* o = out + (size_t)b * 32;
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.x >> (8*k)) & 0xFF; o[k]      = (T)(d * (float)KV[by & 0xF]); o[16 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.y >> (8*k)) & 0xFF; o[4 + k]  = (T)(d * (float)KV[by & 0xF]); o[20 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.z >> (8*k)) & 0xFF; o[8 + k]  = (T)(d * (float)KV[by & 0xF]); o[24 + k] = (T)(d * (float)KV[by >> 4]); }
    #pragma unroll
    for (int k = 0; k < 4; k++) { uint by = (q.w >> (8*k)) & 0xFF; o[12 + k] = (T)(d * (float)KV[by & 0xF]); o[28 + k] = (T)(d * (float)KV[by >> 4]); }
"""

# IQ4_XS: 136 B a superblock of 256. d f16, scales_h u16, scales_l[4], then 128 nibble bytes. Eight 32-blocks a
# superblock; block ib has a 6-bit scale ls = (scales_l nibble) | (scales_h 2 bits << 4), value = d*(ls-32)*KV[q],
# q the nibble of qs[16*ib + (col&15)] (low half of the block's cols the low nibble, high half the high). lane =
# col within a 32-block (0..31). Register-only, batch-invariant. cols = NSB*256.
_MATVEC_IQ4XS = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, lane = gid % 32;
    if (row >= ROWS) return;
    uint hi = lane >> 4, bx = lane & 15;
    const device uchar* base = W + (size_t)row * NSB * 136;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 136;
        float d = (float)(*(const device half*)blk);
        uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
        const device uchar* sl = blk + 4;
        const device uchar* qs = blk + 8;
        const device T* xc = x + (size_t)c * 256;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            uchar byte = qs[16 * ib + bx];
            int q = hi ? (byte >> 4) : (byte & 0xF);
            float w = dl * (float)KV[q];
            uint col = 32 * ib + lane;
            for (int tr = 0; tr < TR; tr++) acc[tr] += w * (float)xc[(size_t)tr * (NSB * 256) + col];
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) {
        float a = simd_sum(acc[tr]);
        if (lane == 0) out[(size_t)tr * ROWS + row] = (T)a;
    }
"""
_DEQUANT_IQ4XS = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint hi = lane >> 4, bx = lane & 15;
    const device uchar* blk = W + (size_t)blk_i * 136;
    float d = (float)(*(const device half*)blk);
    uint sh = (uint)blk[2] | ((uint)blk[3] << 8);
    const device uchar* sl = blk + 4;
    const device uchar* qs = blk + 8;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        int ls = ((sl[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uchar byte = qs[16 * ib + bx];
        int q = hi ? (byte >> 4) : (byte & 0xF);
        o[32 * ib + lane] = (T)(dl * (float)KV[q]);
    }
"""
_lock = threading.Lock()
ROWS_MAX = 16
_mv: dict[Any, Any] = {}
_deq: dict[str, Any] = {}
# (block bytes, block weights, matvec source, dequant source) a kind
_IQ = {
    "iq4xs": (136, 256, _MATVEC_IQ4XS, _DEQUANT_IQ4XS),
}


def _matvec(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    m = mx()
    _, blkw, src, _ = _IQ[kind]
    nb = cols // blkw
    grid = ((rows * 32 + 255) // 256) * 256
    outs = []
    for s in range(0, int(x.shape[0]), ROWS_MAX):
        xr = x[s : s + ROWS_MAX]
        tr = int(xr.shape[0])
        key = (kind, rows, nb, tr)
        with _lock:
            k = _mv.get(key)
            if k is None:
                k = _mv[key] = m.fast.metal_kernel(
                    name=f"btb_{kind}_mv_{rows}_{nb}_{tr}",
                    input_names=["W", "x"],
                    output_names=["out"],
                    header=f"{_KV}#define ROWS {rows}\n#define NSB {nb}\n#define TR {tr}\n",
                    source=src,
                )
        outs.append(
            k(inputs=[w_bytes, xr], grid=(grid, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(tr, rows)], output_dtypes=[x.dtype], template=[("T", x.dtype)])[0]
        )
    return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


def _dequant(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    m = mx()
    _, blkw, _, src = _IQ[kind]
    nblk = int(rows * cols // blkw)
    with _lock:
        k = _deq.get(kind)
        if k is None:
            k = _deq[kind] = m.fast.metal_kernel(
                name=f"btb_{kind}_dequant", input_names=["W"], output_names=["out"], header=_KV, source=src
            )
    grid = ((nblk * 32 + 255) // 256) * 256
    return k(
        inputs=[w_bytes], grid=(grid, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)], output_dtypes=[m.bfloat16], template=[("T", m.bfloat16), ("NBLK", nblk)]
    )[0]


_iq4nl_mv: dict[Any, Any] = {}
_iq4nl_deq = None


def repack_iq4nl(raw: Any) -> tuple[mx_.array, mx_.array]:
    """ggml's interleaved 18-byte IQ4_NL blocks split into the two streams the kernels read: the f16 scales (one a
    block) and the nibble bytes (16 a block, so each block is one aligned uint4). Done once at bind."""
    b = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)).reshape(-1, 18)
    m = mx()
    return (
        m.array(np.ascontiguousarray(b[:, 0:2]).reshape(-1).view(np.float16)),
        m.array(np.ascontiguousarray(b[:, 2:18]).reshape(-1)),
    )


def matvec_iq4nl(d: mx_.array, q: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for an IQ4_NL weight as `repack_iq4nl` splits it; fp32 accumulation, y in
    x's dtype, 1..16 rows a launch."""
    m = mx()
    nb = cols // 32
    grid = ((rows * 32 + 255) // 256) * 256
    outs = []
    for s in range(0, int(x.shape[0]), ROWS_MAX):
        xr = x[s : s + ROWS_MAX]
        tr = int(xr.shape[0])
        key = (rows, nb, tr)
        with _lock:
            k = _iq4nl_mv.get(key)
            if k is None:
                k = _iq4nl_mv[key] = m.fast.metal_kernel(
                    name=f"btb_iq4nl_mv_{rows}_{nb}_{tr}",
                    input_names=["Dd", "Qq", "x"],
                    output_names=["out"],
                    header=f"{_KV}#define ROWS {rows}\n#define NB {nb}\n#define TR {tr}\n",
                    source=_MATVEC_IQ4NL,
                )
        outs.append(
            k(inputs=[d, q, xr], grid=(grid, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(tr, rows)], output_dtypes=[x.dtype], template=[("T", x.dtype)])[0]
        )
    return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


def dequant_iq4nl(d: mx_.array, q: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for an IQ4_NL tensor from its repacked streams."""
    global _iq4nl_deq
    m = mx()
    nblk = int(rows * cols // 32)
    with _lock:
        if _iq4nl_deq is None:
            _iq4nl_deq = m.fast.metal_kernel(
                name="btb_iq4nl_dequant", input_names=["Dd", "Qq"], output_names=["out"], header=_KV, source=_DEQUANT_IQ4NL
            )
    grid = ((nblk + 255) // 256) * 256
    return _iq4nl_deq(
        inputs=[d, q], grid=(grid, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)], output_dtypes=[m.bfloat16], template=[("T", m.bfloat16), ("NBLK", nblk)]
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
_MATVEC_IQ3XXS = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint ib = gpr >> 2, l = gpr & 3, cb = 8 * gpr;   // one lane owns a group of 8 (two 4-value grid entries)
    const device uchar* base = W + (size_t)row * NSB * 98;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 98;
        float d = (float)(*(const device half*)blk);
        const device uchar* qs = blk + 2;
        const device uchar* sca = blk + 66;
        uint aux = (uint)sca[4*ib] | ((uint)sca[4*ib+1]<<8) | ((uint)sca[4*ib+2]<<16) | ((uint)sca[4*ib+3]<<24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        uchar signs = KS[(aux >> (7 * l)) & 127];
        uint i1 = qs[8 * ib + 2 * l], i2 = qs[8 * ib + 2 * l + 1];
        float g[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) { float v = (float)GRID[i1 * 4 + j]; g[j] = (signs & (1 << j)) ? -v : v; }
        #pragma unroll
        for (int j = 0; j < 4; j++) { float v = (float)GRID[i2 * 4 + j]; g[4 + j] = (signs & (1 << (j + 4))) ? -v : v; }
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float p = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) p += g[k] * (float)xc[k];
            acc[tr] += db * p;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) {
        float a = simd_sum(acc[tr]);
        if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a;
    }
"""
_DEQUANT_IQ3XXS = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint l = lane >> 3, m = lane & 7, qsoff = 2 * l + (m >= 4 ? 1 : 0), gj = m & 3;
    uchar smask = 1 << m;
    const device uchar* blk = W + (size_t)blk_i * 98;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2;
    const device uchar* sca = blk + 66;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        uint aux = (uint)sca[4*ib] | ((uint)sca[4*ib+1]<<8) | ((uint)sca[4*ib+2]<<16) | ((uint)sca[4*ib+3]<<24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        uchar idx = qs[8 * ib + qsoff];
        float gval = (float)GRID[(uint)idx * 4 + gj];
        uchar signs = KS[(aux >> (7 * l)) & 127];
        o[32 * ib + lane] = (T)(db * gval * ((signs & smask) ? -1.0f : 1.0f));
    }
"""
# IQ2_XXS: 66 B a superblock of 256. d f16, then 8 interleaved groups of 8 bytes (word0 = 4 grid-index bytes,
# word1 = a 4-bit scale in the top nibble and four 7-bit sign-table indices below). A group is a 32-block, four
# sub-groups of 8 a block; the (256,8) grid gives 8 values a sub-group, signed by the group's sign byte, scaled by
# db = d*(0.5 + scale)*0.25. lane = col within a 32-block, l = lane>>3 the sub-group, j = lane&7 the value.
_MATVEC_IQ2XXS = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint ig = gpr >> 2, sub = gpr & 3, cb = 8 * gpr;   // one lane owns a group of 8 = one grid entry
    const device uchar* base = W + (size_t)row * NSB * 66;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 66;
        float d = (float)(*(const device half*)blk);
        const device uchar* grp = blk + 2 + 8 * ig;
        uint w1 = (uint)grp[4] | ((uint)grp[5]<<8) | ((uint)grp[6]<<16) | ((uint)grp[7]<<24);
        float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
        uchar signs = KS[(w1 >> (7 * sub)) & 127];
        uint gi = grp[sub];
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) { float v = (float)GRID[gi * 8 + k]; g[k] = (signs & (1 << k)) ? -v : v; }
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float p = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) p += g[k] * (float)xc[k];
            acc[tr] += db * p;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ2XXS = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    const device uchar* blk = W + (size_t)blk_i * 66;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        const device uchar* grp = qs + 8 * ib;
        uint w1 = (uint)grp[4] | ((uint)grp[5]<<8) | ((uint)grp[6]<<16) | ((uint)grp[7]<<24);
        float db = d * (0.5f + (float)(w1 >> 28)) * 0.25f;
        float gval = (float)GRID[(uint)grp[l] * 8 + j];
        uchar signs = KS[(w1 >> (7 * l)) & 127];
        o[32 * ib + lane] = (T)(db * gval * ((signs & jmask) ? -1.0f : 1.0f));
    }
"""
# IQ2_XS: 74 B a superblock. d f16, 32 uint16 qs (a group of 8: low 9 bits a (512,8) grid index, top 7 bits a
# sign-table index), 8 scale bytes (16 4-bit sub-scales, one a 16-weight half-block). db = d*(0.5 + sc4)*0.25.
_MATVEC_IQ2XS = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint s = gpr >> 1, cb = 8 * gpr;   // one lane owns a group of 8 = one uint16 grid entry
    const device uchar* base = W + (size_t)row * NSB * 74;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 74;
        float d = (float)(*(const device half*)blk);
        const device uchar* p = blk + 2 + 2 * gpr;
        const device uchar* sc = blk + 66;
        uint q16 = (uint)p[0] | ((uint)p[1] << 8);
        uint sc4 = (sc[s >> 1] >> (4 * (s & 1))) & 0xF;
        float db = d * (0.5f + (float)sc4) * 0.25f;
        uchar signs = KS[q16 >> 9];
        uint gi = q16 & 511;
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) { float v = (float)GRID[gi * 8 + k]; g[k] = (signs & (1 << k)) ? -v : v; }
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float pp = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) pp += g[k] * (float)xc[k];
            acc[tr] += db * pp;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ2XS = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    const device uchar* blk = W + (size_t)blk_i * 74;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* sc = blk + 66;
    device T* o = out + (size_t)blk_i * 256;
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
"""
# IQ2_S: 82 B a superblock. d f16, 32 qs bytes (low 8 index bits), 32 explicit sign bytes (a bit a weight), 8 qh
# bytes (2 high index bits a group, into the (1024,8) grid), 8 scale bytes (16 4-bit sub-scales). db = d*(0.5+sc4)*0.25.
_MATVEC_IQ2S = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint b = gpr >> 1, cb = 8 * gpr;   // one lane owns a group of 8
    const device uchar* base = W + (size_t)row * NSB * 82;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 82;
        float d = (float)(*(const device half*)blk);
        const device uchar* qs = blk + 2;
        const device uchar* sgn = blk + 34;
        const device uchar* qh = blk + 66;
        const device uchar* sc = blk + 74;
        uint idx = (uint)qs[gpr] | ((((uint)qh[gpr >> 2] >> (2 * (gpr & 3))) & 3) << 8);
        uint sc4 = (sc[b >> 1] >> (4 * (b & 1))) & 0xF;
        float db = d * (0.5f + (float)sc4) * 0.25f;
        uchar signs = sgn[gpr];
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) { float v = (float)GRID[idx * 8 + k]; g[k] = (signs & (1 << k)) ? -v : v; }
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float p = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) p += g[k] * (float)xc[k];
            acc[tr] += db * p;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ2S = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint l = lane >> 3, j = lane & 7; uchar jmask = 1 << j;
    const device uchar* blk = W + (size_t)blk_i * 82;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* sgn = blk + 34;
    const device uchar* qh = blk + 66; const device uchar* sc = blk + 74;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        uint grp = 4 * ib + l;
        uint idx = (uint)qs[grp] | ((((uint)qh[grp >> 2] >> (2 * (grp & 3))) & 3) << 8);
        uint b = grp >> 1; uint sc4 = (sc[b >> 1] >> (4 * (b & 1))) & 0xF;
        float db = d * (0.5f + (float)sc4) * 0.25f;
        float gval = (float)GRID[idx * 8 + j];
        o[32 * ib + lane] = (T)(db * gval * ((sgn[grp] & jmask) ? -1.0f : 1.0f));
    }
"""
# IQ1_S: 50 B a superblock. d f16, 32 qs bytes (low 8 index bits), 8 uint16 qh (a 32-block: 3 high index bits a
# sub-block of 8, a 3-bit scale in bits 12-14, a delta sign in bit 15). The (2048,8) grid is ternary; a weight is
# dl*(grid + delta), dl = d*(2*scale+1), delta = +-0.125. No sign table.
_MATVEC_IQ1S = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint ib = gpr >> 2, l = gpr & 3, cb = 8 * gpr;   // one lane owns a group of 8 = one ternary grid entry
    const device uchar* base = W + (size_t)row * NSB * 50;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 50;
        float d = (float)(*(const device half*)blk);
        const device uchar* qs = blk + 2;
        const device uchar* p = blk + 34 + 2 * ib;
        uint qhv = (uint)p[0] | ((uint)p[1] << 8);
        uint idx = (uint)qs[gpr] | (((qhv >> (3 * l)) & 7) << 8);
        float dl = d * (float)(2 * ((qhv >> 12) & 7) + 1);
        float delta = (qhv & 0x8000) ? -0.125f : 0.125f;
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) g[k] = (float)GRID[idx * 8 + k] + delta;
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float pp = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) pp += g[k] * (float)xc[k];
            acc[tr] += dl * pp;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ1S = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint l = lane >> 3, j = lane & 7;
    const device uchar* blk = W + (size_t)blk_i * 50;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* qhb = blk + 34;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        const device uchar* p = qhb + 2 * ib;
        uint qhv = (uint)p[0] | ((uint)p[1] << 8);
        uint idx = (uint)qs[ib * 4 + l] | ((((qhv >> (3 * l)) & 7)) << 8);
        float dl = d * (float)(2 * ((qhv >> 12) & 7) + 1);
        float delta = (qhv & 0x8000) ? -0.125f : 0.125f;
        o[32 * ib + lane] = (T)(dl * ((float)GRID[idx * 8 + j] + delta));
    }
"""

# IQ3_S: 110 B a superblock. d f16, 64 qs bytes (low 8 index bits, one a group of 4 weights), 8 qh bytes (the 9th
# index bit as a plane), 32 explicit sign bytes (a bit a weight), 4 scale bytes (8 4-bit scales a 32-block). The
# (512,4) grid gives 4 values a group; db = d*(1 + 2*sc4). lane = col within a 32-block, qb = lane>>2 the group.
_MATVEC_IQ3S = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint ib = gpr >> 2, cb = 8 * gpr, qb1 = 2 * gpr, qb2 = 2 * gpr + 1;   // group of 8 = two 4-value grid entries
    const device uchar* base = W + (size_t)row * NSB * 110;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 110;
        float d = (float)(*(const device half*)blk);
        const device uchar* qs = blk + 2;
        const device uchar* qh = blk + 66;
        const device uchar* sgn = blk + 74;
        const device uchar* sc = blk + 106;
        uint i1 = (uint)qs[qb1] | ((((uint)qh[qb1 >> 3] >> (qb1 & 7)) & 1) << 8);
        uint i2 = (uint)qs[qb2] | ((((uint)qh[qb2 >> 3] >> (qb2 & 7)) & 1) << 8);
        uint sc4 = (sc[ib >> 1] >> (4 * (ib & 1))) & 0xF;
        float db = d * (float)(1 + 2 * sc4);
        uchar signs = sgn[gpr];
        float g[8];
        #pragma unroll
        for (int j = 0; j < 4; j++) { float v = (float)GRID[i1 * 4 + j]; g[j] = (signs & (1 << j)) ? -v : v; }
        #pragma unroll
        for (int j = 0; j < 4; j++) { float v = (float)GRID[i2 * 4 + j]; g[4 + j] = (signs & (1 << (j + 4))) ? -v : v; }
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float p = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) p += g[k] * (float)xc[k];
            acc[tr] += db * p;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ3S = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint ql = lane >> 2, j = lane & 3, sl = lane >> 3; uchar sbit = 1 << (lane & 7);
    const device uchar* blk = W + (size_t)blk_i * 110;
    float d = (float)(*(const device half*)blk);
    const device uchar* qs = blk + 2; const device uchar* qh = blk + 66;
    const device uchar* sgn = blk + 74; const device uchar* sc = blk + 106;
    device T* o = out + (size_t)blk_i * 256;
    for (int ib = 0; ib < 8; ib++) {
        uint qb = 8 * ib + ql;
        uint idx = (uint)qs[qb] | ((((uint)qh[qb >> 3] >> (qb & 7)) & 1) << 8);
        uint sc4 = (sc[ib >> 1] >> (4 * (ib & 1))) & 0xF;
        float db = d * (float)(1 + 2 * sc4);
        float gval = (float)GRID[idx * 4 + j];
        o[32 * ib + lane] = (T)(db * gval * ((sgn[4 * ib + sl] & sbit) ? -1.0f : 1.0f));
    }
"""

# IQ1_M: 56 B a superblock, no leading f16. 32 qs bytes (low 8 index bits, a group of 8), 16 qh bytes (a nibble a
# group: 3 high index bits and a delta-sign bit), 8 scale bytes as four uint16 whose top nibbles reassemble the
# block's f16 d and whose low 12 bits hold sixteen 3-bit sub-scales. The (2048,8) ternary grid (shared with IQ1_S);
# dl = d*(2*sub+1), delta = +-0.125, weight = dl*(grid + delta). The matvec reads a side stream built once at bind
# (`_side_iq1m`: the f16 d precomputed, the sub-scales as nibble pairs) instead of reassembling d and selecting a
# sub-scale through four uint16 every superblock; +10 bytes a superblock, 1.3x faster. lane = group of 8.
_MATVEC_IQ1M = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, gpr = gid % 32;
    if (row >= ROWS) return;
    uint si = gpr >> 1, cb = 8 * gpr;   // one lane owns a group of 8 = one ternary grid entry
    const device uchar* base = W + (size_t)row * NSB * 56;
    const device half* Drow = Dd + (size_t)row * NSB;
    const device uchar* Srow = SB + (size_t)row * NSB * 8;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 56;
        const device uchar* qs = blk;
        const device uchar* qh = blk + 32;
        float d = (float)Drow[c];
        uint sub = (Srow[c * 8 + (si >> 1)] >> (4 * (si & 1))) & 7;
        uchar nib = (qh[gpr >> 1] >> (4 * (gpr & 1))) & 0xF;
        uint idx = (uint)qs[gpr] | (((uint)(nib & 7)) << 8);
        float delta = (nib & 8) ? -0.125f : 0.125f;
        float dl = d * (float)(2 * sub + 1);
        float g[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) g[k] = (float)GRID[idx * 8 + k] + delta;
        for (int tr = 0; tr < TR; tr++) {
            const device T* xc = x + (size_t)tr * (NSB * 256) + (size_t)c * 256 + cb;
            float p = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) p += g[k] * (float)xc[k];
            acc[tr] += dl * p;
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) { float a = simd_sum(acc[tr]); if (gpr == 0) out[(size_t)tr * ROWS + row] = (T)a; }
"""
_DEQUANT_IQ1M = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    uint j8 = lane >> 3, e = lane & 7, sl = lane >> 4;
    const device uchar* blk = W + (size_t)blk_i * 56;
    const device uchar* qs = blk; const device uchar* qh = blk + 32; const device uchar* scb = blk + 48;
    uint s0 = (uint)scb[0]|((uint)scb[1]<<8), s1 = (uint)scb[2]|((uint)scb[3]<<8);
    uint s2 = (uint)scb[4]|((uint)scb[5]<<8), s3 = (uint)scb[6]|((uint)scb[7]<<8);
    ushort dbits = (ushort)((s0>>12) | ((s1>>12)<<4) | ((s2>>12)<<8) | ((s3>>12)<<12));
    float d = (float)as_type<half>(dbits);
    device T* o = out + (size_t)blk_i * 256;
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
"""

def _side_iq1m(raw: Any) -> tuple[Any, Any]:
    """IQ1_M's matvec side stream, built once at bind from the raw 56-byte superblocks: the block f16 d
    reassembled here rather than in the kernel, and the sixteen 3-bit sub-scales packed as nibble pairs (8 B)."""
    b = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)).reshape(-1, 56)
    scb = np.ascontiguousarray(b[:, 48:56]).view(np.uint16).reshape(-1, 4).astype(np.uint32)
    dbits = ((scb[:, 0] >> 12) | ((scb[:, 1] >> 12) << 4) | ((scb[:, 2] >> 12) << 8) | ((scb[:, 3] >> 12) << 12)).astype(np.uint16)
    sub = ((scb.reshape(-1, 4, 1) >> np.array([0, 3, 6, 9], np.uint32).reshape(1, 1, 4)) & 7).reshape(-1, 16).astype(np.uint8)
    sb = (sub[:, 0::2] | (sub[:, 1::2] << 4)).astype(np.uint8).reshape(-1)
    m = mx()
    return m.array(dbits.view(np.float16)), m.array(sb)


# the IQ lattice family (grid-codebook quants) on one launcher: each entry is the superblock byte size, the grid
# class name, whether the shared sign table is read, the matvec/dequant Metal sources, and optionally a `side`
# repack (raw -> extra device streams named `side_names`) the matvec reads beside the raw blocks. A kernel always
# takes W, x and GRID; KS (the sign table) only when the entry reads it. Adding a quant is one entry here plus a
# GGUF-name -> kind row in the engine binder. All are register-only and batch-invariant (row i identical at any T).
_LATT: dict[str, dict[str, Any]] = {
    "iq3xxs": {"bytes": 98, "grid": "IQ3_XXS", "ksigns": True, "mv": _MATVEC_IQ3XXS, "deq": _DEQUANT_IQ3XXS},
    "iq2xxs": {"bytes": 66, "grid": "IQ2_XXS", "ksigns": True, "mv": _MATVEC_IQ2XXS, "deq": _DEQUANT_IQ2XXS},
    "iq2xs": {"bytes": 74, "grid": "IQ2_XS", "ksigns": True, "mv": _MATVEC_IQ2XS, "deq": _DEQUANT_IQ2XS},
    "iq2s": {"bytes": 82, "grid": "IQ2_S", "ksigns": False, "mv": _MATVEC_IQ2S, "deq": _DEQUANT_IQ2S},
    "iq1s": {"bytes": 50, "grid": "IQ1_S", "ksigns": False, "mv": _MATVEC_IQ1S, "deq": _DEQUANT_IQ1S},
    "iq3s": {"bytes": 110, "grid": "IQ3_S", "ksigns": False, "mv": _MATVEC_IQ3S, "deq": _DEQUANT_IQ3S},
    "iq1m": {"bytes": 56, "grid": "IQ1_M", "ksigns": False, "mv": _MATVEC_IQ1M, "deq": _DEQUANT_IQ1M,
             "side": _side_iq1m, "side_names": ["Dd", "SB"]},
}


def repack_lattice(kind: str, raw: Any) -> tuple[Any, ...]:
    """the side streams `matvec_lattice` wants beside the raw bytes for this kind (empty for most), built once."""
    side = _LATT[kind].get("side")
    return tuple(side(raw)) if side is not None else ()
_latt_mv: dict[Any, Any] = {}
_latt_deq: dict[str, Any] = {}


def matvec_lattice(
    kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int, side: tuple[Any, ...] = ()
) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T for a raw IQ lattice weight; fp32 accumulation, y in x's dtype, 1..16 rows a
    launch. `kind` selects the layout from `_LATT`; the grid (and sign table, when used) ride along as buffers,
    and `side` is this kind's `repack_lattice` streams when it has any."""
    m = mx()
    spec = _LATT[kind]
    inames = ["W", "x", "GRID"] + (["KS"] if spec["ksigns"] else []) + list(spec.get("side_names", []))
    tbls = [_grid(spec["grid"])] + ([_ksigns()] if spec["ksigns"] else []) + list(side)
    nsb = cols // 256
    g = ((rows * 32 + 255) // 256) * 256
    outs = []
    for s in range(0, int(x.shape[0]), ROWS_MAX):
        xr = x[s : s + ROWS_MAX]
        tr = int(xr.shape[0])
        key = (kind, rows, nsb, tr)
        with _lock:
            k = _latt_mv.get(key)
            if k is None:
                k = _latt_mv[key] = m.fast.metal_kernel(
                    name=f"btb_{kind}_mv_{rows}_{nsb}_{tr}",
                    input_names=inames,
                    output_names=["out"],
                    header=f"#define ROWS {rows}\n#define NSB {nsb}\n#define TR {tr}\n",
                    source=spec["mv"],
                )
        outs.append(
            k(inputs=[w_bytes, xr, *tbls], grid=(g, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(tr, rows)], output_dtypes=[x.dtype], template=[("T", x.dtype)])[0]
        )
    return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


def dequant_lattice(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for an IQ lattice tensor of the given `kind`."""
    m = mx()
    spec = _LATT[kind]
    inames = ["W", "GRID"] + (["KS"] if spec["ksigns"] else [])
    tbls = [_grid(spec["grid"])] + ([_ksigns()] if spec["ksigns"] else [])
    nblk = int(rows * cols // 256)
    with _lock:
        k = _latt_deq.get(kind)
        if k is None:
            k = _latt_deq[kind] = m.fast.metal_kernel(
                name=f"btb_{kind}_dequant", input_names=inames, output_names=["out"], header="", source=spec["deq"]
            )
    g = ((nblk * 32 + 255) // 256) * 256
    return k(
        inputs=[w_bytes, *tbls], grid=(g, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)], output_dtypes=[m.bfloat16], template=[("T", m.bfloat16), ("NBLK", nblk)]
    )[0]
