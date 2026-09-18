# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Q6_K on Metal: llama.cpp's Q6_K superblocks read as stored and dequantized in registers (free under the
memory-bound read, by the roofline), so the 6-bit weights never expand to bf16 in memory. `matvec_q6k` is the
linear (one simdgroup a row, the 32 lanes cooperating on a superblock for coalesced reads, up to 16 rows an
invocation reusing the one weight read); `gather_q6k` reads only an embedding's own rows; `dequant_q6k` the
whole matrix, for the rare path that wants a bf16 copy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import mx
from .launch import BlockKernel, RowKernel

if TYPE_CHECKING:
    import mlx.core as mx_

# 210 bytes a superblock of 256 weights: ql[128] (low nibbles), qh[64] (high 2 bits), scales[16] int8, d f16;
# value = d * scales[is] * (q - 32), is = h*8 + l/16 over the sub-block indices {is, is+2, is+4, is+6}.
_DECODE = r"""
template <typename U>
inline void q6k_decode_block(const device uchar* blk, device U* o) {
    const device uchar* ql = blk;
    const device uchar* qh = blk + 128;
    const device char* sc = (const device char*)(blk + 192);
    float df = (float)(*(const device half*)(blk + 208));
    for (int h = 0; h < 2; h++) {
        uint base = h * 128, qlo = h * 64, qho = h * 32, sco = h * 8;
        for (int l = 0; l < 32; l++) {
            int isc = sco + l / 16;
            uchar l0 = ql[qlo + l], l1 = ql[qlo + l + 32], hb = qh[qho + l];
            o[base + l]      = (U)(df * sc[isc + 0] * (((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32));
            o[base + l + 32] = (U)(df * sc[isc + 2] * (((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32));
            o[base + l + 64] = (U)(df * sc[isc + 4] * (((l0 >> 4)  | (((hb >> 4) & 3) << 4)) - 32));
            o[base + l + 96] = (U)(df * sc[isc + 6] * (((l1 >> 4)  | (((hb >> 6) & 3) << 4)) - 32));
        }
    }
}
"""

# the linear: one simdgroup a row, lane = inner index l so all 32 lanes cooperate on a superblock (coalesced
# ql/qh/x), each lane its four weights, then simd_sum. Cooperative, so it does not share q6k_decode_block.
_MATVEC = r"""
    uint gid = thread_position_in_grid.x;
    uint row = gid / 32, lane = gid % 32;
    if (row >= ROWS) return;
    const device uchar* base = W + (size_t)row * NSB * 210;
    float acc[TR];
    #pragma unroll
    for (int u = 0; u < TR; u++) acc[u] = 0.0f;
    for (uint c = 0; c < NSB; c++) {
        const device uchar* blk = base + (size_t)c * 210;
        const device uchar* ql = blk;
        const device uchar* qh = blk + 128;
        const device char* sc = (const device char*)(blk + 192);
        float df = (float)(*(const device half*)(blk + 208));
        for (int h = 0; h < 2; h++) {
            uint o = h * 128, qlo = h * 64, qho = h * 32, sco = h * 8;
            int isc = sco + lane / 16;
            uchar l0 = ql[qlo + lane], l1 = ql[qlo + lane + 32], hb = qh[qho + lane];
            float w1 = df * sc[isc + 0] * (((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32);
            float w2 = df * sc[isc + 2] * (((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32);
            float w3 = df * sc[isc + 4] * (((l0 >> 4)  | (((hb >> 4) & 3) << 4)) - 32);
            float w4 = df * sc[isc + 6] * (((l1 >> 4)  | (((hb >> 6) & 3) << 4)) - 32);
            for (int tr = 0; tr < TR; tr++) {
                const device T* xb = x + (size_t)tr * (NSB * 256) + (size_t)c * 256;
                acc[tr] += w1 * (float)xb[o + lane] + w2 * (float)xb[o + lane + 32]
                         + w3 * (float)xb[o + lane + 64] + w4 * (float)xb[o + lane + 96];
            }
        }
    }
    #pragma unroll
    for (int tr = 0; tr < TR; tr++) {
        float a = simd_sum(acc[tr]);
        if (lane == 0) out[(size_t)tr * ROWS + row] = (T)a;
    }
"""

# one thread a superblock; `dequant` walks the matrix, `gather` an embedding's own rows by the ids in `tok`.
_DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= NBLK) return;
    q6k_decode_block<T>(W + (size_t)t * 210, out + (size_t)t * 256);
"""
_GATHER = r"""
    uint i = thread_position_in_grid.x;
    uint nrows = tok_shape[0];
    if (i >= nrows * NSB) return;
    uint t = i / NSB, c = i % NSB;
    uint row = (uint)tok[t];
    q6k_decode_block<T>(W + ((size_t)row * NSB + (size_t)c) * 210, out + ((size_t)t * NSB + (size_t)c) * 256);
"""

_MV = RowKernel("q6k_mv", ["W", "x"], "", _MATVEC)
_DEQ = BlockKernel("q6k_dequant", ["W"], _DECODE, _DEQUANT)
_gathers: dict[int, BlockKernel] = {}  # by superblocks a row: the gather's source reads NSB as a define


def matvec_q6k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw Q6_K bytes (rows * cols / 256 row-major superblocks).
    float32 accumulation over the blocks as stored, y in x's dtype; 1..16 rows a launch, a wider pass chunked."""
    return _MV.matvec([w_bytes], x, [], rows, cols // 256)


def gather_q6k(w_bytes: mx_.array, tok: mx_.array, cols: int) -> mx_.array:
    """the embedding rows for token ids `tok` [T] (int32): [T, cols] bf16, each row's superblocks dequantized
    straight from the packed bytes - no bf16 copy of the whole table."""
    nsb = cols // 256
    k = _gathers.get(nsb)
    if k is None:
        k = _gathers[nsb] = BlockKernel(f"q6k_gather_{nsb}", ["W", "tok"], _DECODE + f"#define NSB {nsb}\n", _GATHER)
    n = int(tok.shape[0])
    return k.run([w_bytes, tok.astype(mx().int32)], n * nsb, (n, cols), nblk=False)


def dequant_q6k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight, the numbers llama.cpp dequantizes (for a path that wants a bf16 copy)."""
    return _DEQ.run([w_bytes], rows * cols // 256, (rows, cols))
