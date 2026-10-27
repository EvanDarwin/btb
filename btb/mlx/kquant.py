# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The k-quants on Metal - Q2_K, Q3_K, Q4_K, Q5_K - read as stored: each type's `w8` decodes 8 consecutive weights of
a row from its superblocks in registers, and the matvec, the dequant and the megakernel are built on it
(`matvec.StoredKernel`), so the 2-5 bit weights never expand to bf16 in memory on the decode path and every kernel
over a type has the same bits. Native as stored: 84, 110, 144 and 176 B a superblock of 256."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kinds import Quant
from .core import mx
from .matvec import StoredKernel, dequant, launch

if TYPE_CHECKING:
    import mlx.core as mx_

# Q4_K: 144 bytes a superblock of 256 weights: d f16, dmin f16, 12 bytes packing a 6-bit scale and a 6-bit min for
# each of 8 sub-blocks of 32, then 128 bytes of nibbles (byte b = qs[32k + i] holds the low nibble for sub-block 2k
# and the high nibble for sub-block 2k+1, both at inner index i); value = d*sc[j]*q - dmin*mn[j], j = weight // 32.
# Q5_K: 176 bytes, the 5th bit of weight (j, i) bit j of qh[i] (32 bytes at 16), the nibbles at 48. The 6-bit
# scale/min unpack is llama.cpp's get_scale_min_k4.
_SM = r"""
inline void q4k_scale_min(const device uchar* s, int j, thread float& sc, thread float& mn) {
    if (j < 4) { sc = (float)(s[j] & 63); mn = (float)(s[j + 4] & 63); }
    else {
        sc = (float)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}
"""
_W8_Q45 = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * BLKB;
    uint j = (k % 256) / 32, i0 = k % 32, sh = (j & 1) * 4;
    float sc, mn;
    q4k_scale_min(blk + 4, j, sc, mn);
    float a = (float)(*(const device half*)blk) * sc, b = (float)(*(const device half*)(blk + 2)) * mn;
    const device uchar* qs = blk + QSOFF + 32 * (j / 2) + i0;
    const device uchar* qh = blk + 16 + i0;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        uint q = (qs[t] >> sh) & 0xF;
        if (Q5) q |= ((qh[t] >> j) & 1) << 4;
        w[t] = fma(a, (float)q, -b);
    }
"""
# Q4_K: no qh, nibbles at 16. Q5_K: the 5th bit in qh at 16, nibbles at 48.
_KQ = {"q4k": (144, 16, 0), "q5k": (176, 48, 1)}


def _defs(kind: str) -> str:
    blkb, qsoff, q5 = _KQ[kind]
    return f"#define BLKB {blkb}\n#define QSOFF {qsoff}\n#define Q5 {q5}\n"


# Q2_K: 84 B a superblock. scales[16] (a 4-bit scale and 4-bit min a 16-weight sub-block), qs[64] (2-bit), d f16,
# dmin f16. Column c: h = c / 128, j = c % 128 / 32, m = c % 32; q = (qs[32h + m] >> 2j) & 3, the scale byte
# scales[8h + 2j + m / 16]; value = d*(scale & 0xF)*q - dmin*(scale >> 4).
_W8_Q2K = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * 84;
    uint c = k % 256, h = c / 128, j = c % 128 / 32, m = c % 32;
    uchar s = blk[h * 8 + 2 * j + m / 16];
    float a = (float)(*(const device half*)(blk + 80)) * (float)(s & 0xF);
    float b = (float)(*(const device half*)(blk + 82)) * (float)(s >> 4);
    const device uchar* qs = blk + 16 + h * 32 + m;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = fma(a, (float)((qs[t] >> (2 * j)) & 3), -b);
"""

# Q3_K: 110 B a superblock. hmask[32] (the 3rd bit of each weight), qs[64] (2 low bits), scales[12] (16 signed 6-bit
# scales packed with the kmask dance), d f16. Column c as Q2_K's; q = ((qs[32h + m] >> 2j) & 3) - (hmask[m] bit
# 4h + j set ? 0 : 4), value = d*(scale[8h + 2j + m / 16] - 32)*q.
_Q3K_AUX = r"""
inline void q3k_aux(const device uchar* sc, thread uint aux[4]) {
    uint a0 = (uint)sc[0] | ((uint)sc[1]<<8) | ((uint)sc[2]<<16) | ((uint)sc[3]<<24);
    uint a1 = (uint)sc[4] | ((uint)sc[5]<<8) | ((uint)sc[6]<<16) | ((uint)sc[7]<<24);
    uint a2 = (uint)sc[8] | ((uint)sc[9]<<8) | ((uint)sc[10]<<16) | ((uint)sc[11]<<24);
    const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
    aux[2] = ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4);
    aux[3] = ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4);
    aux[0] = (a0 & k2) | (((a2 >> 0) & k1) << 4);
    aux[1] = (a1 & k2) | (((a2 >> 2) & k1) << 4);
}
"""
_W8_Q3K = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * 110;
    uint c = k % 256, h = c / 128, j = c % 128 / 32, m = c % 32;
    thread uint aux[4];
    q3k_aux(blk + 96, aux);
    int sc = (int)((const thread char*)aux)[h * 8 + 2 * j + m / 16] - 32;
    float a = (float)(*(const device half*)(blk + 108)) * (float)sc;
    const device uchar* qs = blk + 32 + h * 32 + m;
    const device uchar* hm = blk + m;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) {
        int q = (qs[t] >> (2 * j)) & 3;
        if (!(hm[t] & (1 << (h * 4 + j)))) q -= 4;
        w[t] = a * (float)q;
    }
"""

KERNELS: dict[Quant, StoredKernel] = {
    Quant.Q2_K: StoredKernel(_W8_Q2K, 256),
    # TILE at 4 float32 rows: a head ran 8 points slower on TILE4
    Quant.Q3_K: StoredKernel(_W8_Q3K, 256, _Q3K_AUX, tile4=False),
    Quant.Q4_K: StoredKernel(_W8_Q45, 256, _SM, _defs("q4k")),
    Quant.Q5_K: StoredKernel(_W8_Q45, 256, _SM, _defs("q5k")),
}
_BY_KIND = {"q2k": Quant.Q2_K, "q3k": Quant.Q3_K, "q4k": Quant.Q4_K, "q5k": Quant.Q5_K}


def _matvec(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    return launch(f"btb_{kind}_mv_{rows}_{cols // 256}", KERNELS[_BY_KIND[kind]], [w_bytes], x, rows, cols)


def _dequant(kind: str, w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None) -> mx_.array:
    k = KERNELS[_BY_KIND[kind]]
    return dequant(f"btb_{kind}_dequant", k, [w_bytes], rows, cols, dtype or mx.bfloat16)


def matvec_q4k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw Q4_K bytes (rows * cols / 256 row-major superblocks),
    x bf16 or float32, y in its dtype; any rows (`matvec.launch`)."""
    return _matvec("q4k", w_bytes, x, rows, cols)


def dequant_q4k(w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """the full weight in `dtype` (bf16 by default): the matvec's weights, a prefill's gemm's (`matvec.dequant`)."""
    return _dequant("q4k", w_bytes, rows, cols, dtype)


def matvec_q5k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q5_K (176 B superblocks, the 5th bit in the qh plane)."""
    return _matvec("q5k", w_bytes, x, rows, cols)


def dequant_q5k(w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """`dequant_q4k` for Q5_K."""
    return _dequant("q5k", w_bytes, rows, cols, dtype)


def matvec_q2k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q2_K (84 B superblocks, 2-bit, 16-weight sub-blocks with 4-bit scale/min)."""
    return _matvec("q2k", w_bytes, x, rows, cols)


def dequant_q2k(w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """`dequant_q4k` for Q2_K."""
    return _dequant("q2k", w_bytes, rows, cols, dtype)


def matvec_q3k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q3_K (110 B superblocks, 3-bit split qs+hmask, 16 signed 6-bit scales)."""
    return _matvec("q3k", w_bytes, x, rows, cols)


def dequant_q3k(w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """`dequant_q4k` for Q3_K."""
    return _dequant("q3k", w_bytes, rows, cols, dtype)
