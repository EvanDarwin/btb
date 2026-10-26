# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Q6_K on Metal: llama.cpp's Q6_K superblocks read as stored, 8 consecutive weights of a row decoded in registers
(`w8`), so the 6-bit weights never expand to bf16 in memory on the decode path and the matvec, the dequant and the
megakernel have the same bits (`matvec.StoredKernel`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kinds import Quant
from .core import mx
from .matvec import StoredKernel, dequant, launch

if TYPE_CHECKING:
    import mlx.core as mx_

# 210 bytes a superblock of 256 weights: ql[128] (low nibbles), qh[64] (high 2 bits), scales[16] int8, d f16. Column
# c: h = c / 128, quarter u = c % 128 / 32, l = c % 32; the nibble is ql[64h + l + 32 (u & 1)]'s low (u < 2) or high
# one, the high bits (qh[32h + l] >> 2u) & 3, the scale scales[8h + l / 16 + 2u]; value = d*scale*(q - 32).
_W8 = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 256) + k / 256) * 210;
    uint c = k % 256, h = c / 128, u = c % 128 / 32, l = c % 32, sh = (u >> 1) * 4;
    float a = (float)(*(const device half*)(blk + 208)) * (float)((const device char*)(blk + 192))[h * 8 + l / 16 + 2 * u];
    const device uchar* ql = blk + h * 64 + l + (u & 1) * 32;
    const device uchar* qh = blk + 128 + h * 32 + l;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = a * (float)((int)(((ql[t] >> sh) & 0xF) | (((qh[t] >> (2 * u)) & 3) << 4)) - 32);
"""

KERNELS: dict[Quant, StoredKernel] = {Quant.Q6_K: StoredKernel(_W8, 256)}


def matvec_q6k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw Q6_K bytes (rows * cols / 256 row-major superblocks),
    x bf16 or float32, y in its dtype; any rows (`matvec.launch`)."""
    return launch(f"btb_q6k_mv_{rows}_{cols // 256}", KERNELS[Quant.Q6_K], [w_bytes], x, rows, cols)


def dequant_q6k(
    w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None, chain: bool = False
) -> mx_.array:
    """the full weight in `dtype` (bf16 by default): the matvec's weights, `chain` in its chain's column order for a
    prefill's gemm (`matvec.dequant`)."""
    return dequant("btb_q6k_dequant", KERNELS[Quant.Q6_K], [w_bytes], rows, cols, dtype or mx.bfloat16, chain)
