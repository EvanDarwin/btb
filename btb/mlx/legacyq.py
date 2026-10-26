# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""ggml's legacy block quants on Metal - Q4_0, Q4_1, Q8_0 - read as stored, 32 weights a block, 8 consecutive weights
of a row decoded in registers (`w8`): the matvec, the dequant and the megakernel are built on it
(`matvec.StoredKernel`), so each has the same bits (which MLX's `quantized_matmul` does not across widths: its
multi-row kernel is not its one-row one, so a speculative verify pass would drift off the plain step)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..kinds import Quant
from .core import mx
from .matvec import StoredKernel, dequant, launch

if TYPE_CHECKING:
    import mlx.core as mx_

# (block bytes, kernel id) by kind. Q4_0: d f16, 16 nibble bytes, value d*(q-8). Q4_1: d f16, m f16, 16 nibble
# bytes, value d*q + m. Byte i holds weight i in its low nibble and weight i+16 in its high. Q8_0: d f16, 32 int8,
# value d*q. Every block offset is even, so the f16 fields are aligned.
KINDS: dict[str, tuple[int, int]] = {"q4_0": (18, 0), "q4_1": (20, 1), "q8_0": (34, 2)}

_W8 = r"""
    const device uchar* blk = W + ((size_t)row * (cols / 32) + k / 32) * BLKB;
    uint i0 = k % 32, sh = i0 < 16 ? 0 : 4;
    float d = (float)(*(const device half*)blk);
#if KIND == 2
    const device char* qs = (const device char*)(blk + 2) + i0;
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = d * (float)qs[t];
#elif KIND == 1
    float mn = (float)(*(const device half*)(blk + 2));
    const device uchar* qs = blk + 4 + (i0 & 15);
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = fma(d, (float)((qs[t] >> sh) & 0xF), mn);
#else
    const device uchar* qs = blk + 2 + (i0 & 15);
    #pragma unroll
    for (uint t = 0; t < 8; ++t) w[t] = d * (float)((int)((qs[t] >> sh) & 0xF) - 8);
#endif
"""


def _kind(kind: str) -> str:
    blkb, kid = KINDS[kind]
    return f"#define BLKB {blkb}\n#define KIND {kid}\n"


# every kind for kernels built on its chain (the megakernel)
KERNELS: dict[Quant, StoredKernel] = {Quant[k.upper()]: StoredKernel(_W8, 32, defs=_kind(k)) for k in KINDS}


def matvec_legacy(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw `kind` bytes (rows * cols / 32 row-major blocks), x
    bf16 or float32, y in its dtype; any rows (`matvec.launch`)."""
    return launch(f"btb_{kind}_mv_{rows}_{cols // 32}", KERNELS[Quant[kind.upper()]], [w_bytes], x, rows, cols)


def dequant_legacy(
    kind: str, w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None, chain: bool = False
) -> mx_.array:
    """the full weight in `dtype` (bf16 by default): the matvec's weights, `chain` in its chain's column order for a
    prefill's gemm (`matvec.dequant`)"""
    k = KERNELS[Quant[kind.upper()]]
    return dequant(f"btb_{kind}_dequant", k, [w_bytes], rows, cols, dtype or mx.bfloat16, chain)
