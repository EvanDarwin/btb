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

# a block's weights c and c + 1 (c even), one 16-bit load (a block is 2-byte aligned): Q8_0's two int8 times d, Q4_0's
# and Q4_1's two nibbles of consecutive bytes (low below column 16, else high) as 0x1p23 + q (exact for a small integer,
# no convert), Q4_0's q - 8 likewise. Both `w8` and `w2` are built on it, so both have its bits
_PAIR = r"""
inline float2 legacy_pair(const device uchar* blk, uint c) {
    float d = (float)(*(const device half*)blk);
#if KIND == 2
    char2 q = as_type<char2>(*(const device ushort*)(blk + 2 + c));
    return float2(d * (float)q.x, d * (float)q.y);
#elif KIND == 1
    float mn = (float)(*(const device half*)(blk + 2));
    uint v = (uint)(*(const device ushort*)(blk + 4 + (c & 15))) >> (c < 16 ? 0 : 4);
    return float2(fma(d, as_type<float>(0x4B000000u | (v & 0xF)) - 8388608.0f, mn),
                  fma(d, as_type<float>(0x4B000000u | ((v >> 8) & 0xF)) - 8388608.0f, mn));
#else
    uint v = (uint)(*(const device ushort*)(blk + 2 + (c & 15))) >> (c < 16 ? 0 : 4);
    return float2(d * (as_type<float>(0x4B000000u | (v & 0xF)) - 8388616.0f),
                  d * (as_type<float>(0x4B000000u | ((v >> 8) & 0xF)) - 8388616.0f));
#endif
}
"""
_BLOCK = "    const device uchar* blk = W + ((size_t)row * (cols / 32) + k / 32) * BLKB;\n"
_W8 = (
    _BLOCK
    + r"""
    #pragma unroll
    for (uint t = 0; t < 4; ++t) {
        float2 v = legacy_pair(blk, k % 32 + 2 * t);
        w[2 * t] = v.x;
        w[2 * t + 1] = v.y;
    }
"""
)
_W2 = (
    _BLOCK
    + r"""
    float2 v = legacy_pair(blk, k % 32);
    w[0] = v.x;
    w[1] = v.y;
"""
)


def _kind(kind: str) -> str:
    blkb, kid = KINDS[kind]
    return f"#define BLKB {blkb}\n#define KIND {kid}\n"


def _kernel(kind: str) -> StoredKernel:
    # Q8_0 takes PAIR2 to 5 pass rows: its head's float32 x at 2-5 rows ran +4-29% on LANE2 or TILE, within 2% of the
    # lane-strided kernel on PAIR2. Q4_0 and Q4_1 ran 8-20 points slower on PAIR2 than on LANE2 for a bf16 x
    if kind == "q8_0":
        return StoredKernel(_W8, 32, header=_PAIR, defs=_kind(kind), w2=_W2, lane_rows=5)
    return StoredKernel(_W8, 32, header=_PAIR, defs=_kind(kind))


# every kind for kernels built on its chain (the megakernel)
KERNELS: dict[Quant, StoredKernel] = {Quant[k.upper()]: _kernel(k) for k in KINDS}


def matvec_legacy(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw `kind` bytes (rows * cols / 32 row-major blocks), x
    bf16 or float32, y in its dtype; any rows (`matvec.launch`)."""
    return launch(f"btb_{kind}_mv_{rows}_{cols // 32}", KERNELS[Quant[kind.upper()]], [w_bytes], x, rows, cols)


def dequant_legacy(kind: str, w_bytes: mx_.array, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """the full weight in `dtype` (bf16 by default): the matvec's weights, a prefill's gemm's (`matvec.dequant`)"""
    k = KERNELS[Quant[kind.upper()]]
    return dequant(f"btb_{kind}_dequant", k, [w_bytes], rows, cols, dtype or mx.bfloat16)
