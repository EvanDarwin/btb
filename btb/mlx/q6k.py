# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Q6_K on Metal: llama.cpp's Q6_K superblocks read as stored and dequantized in registers (free under the
memory-bound read, by the roofline), so the 6-bit weights never expand to bf16 in memory. `matvec_q6k` is the
linear (`matvec.launch`, the 32 lanes of a simdgroup cooperating on a superblock for coalesced reads, up to 16
rows an invocation reusing the one weight read); `gather_q6k` reads only an embedding's own rows; `dequant_q6k`
the whole matrix, for the rare path that wants a bf16 copy."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from ..kinds import Quant
from .core import metal_kernel, mx
from .matvec import Frags, StoredKernel, blocks, launch

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

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

# the linear (`matvec.blocks`): lane = inner index l so all 32 lanes cooperate on a superblock (coalesced ql/qh/x),
# each lane its eight weights (four a half, at col h*128 + i*32 + lane), then simd_sum. Cooperative, so it does
# not share q6k_decode_block.
MV = Frags(
    "",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 210;
            const device uchar* ql = blk;
            const device uchar* qh = blk + 128;
            const device char* sc = (const device char*)(blk + 192);
            float df = (float)(*(const device half*)(blk + 208));
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                uint qlo = h * 64, qho = h * 32, sco = h * 8;
                int isc = sco + lane / 16;
                uchar l0 = ql[qlo + lane], l1 = ql[qlo + lane + 32], hb = qh[qho + lane];
                w[r][4 * h + 0] = df * sc[isc + 0] * (((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32);
                w[r][4 * h + 1] = df * sc[isc + 2] * (((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32);
                w[r][4 * h + 2] = df * sc[isc + 4] * (((l0 >> 4)  | (((hb >> 4) & 3) << 4)) - 32);
                w[r][4 * h + 3] = df * sc[isc + 6] * (((l1 >> 4)  | (((hb >> 6) & 3) << 4)) - 32);
            }
""",
    r"""
            float xv[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[k] = (float)xc[(k >> 2) * 128 + (k & 3) * 32 + lane];
            #pragma unroll
            for (int r = 0; r < RP; r++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    acc[r][tr] += w[r][4 * h] * xv[4 * h] + w[r][4 * h + 1] * xv[4 * h + 1]
                                + w[r][4 * h + 2] * xv[4 * h + 2] + w[r][4 * h + 3] * xv[4 * h + 3];
""",
)
_MATVEC = blocks(MV)

# `dequant` walks the matrix a simdgroup a superblock, lane = inner index l as in the matvec, so each store is 32
# consecutive weights (a thread a superblock measured 12x slower); `gather` an embedding's own rows by `tok`, a
# thread a superblock. Both are `q6k_decode_block`'s numbers.
_DEQUANT = r"""
    uint gid = thread_position_in_grid.x;
    uint t = gid / 32, l = gid % 32;
    if (t >= NBLK) return;
    const device uchar* blk = W + (size_t)t * 210;
    const device uchar* ql = blk;
    const device uchar* qh = blk + 128;
    const device char* sc = (const device char*)(blk + 192);
    float df = (float)(*(const device half*)(blk + 208));
    device T* o = out + (size_t)t * 256;
    for (int h = 0; h < 2; h++) {
        uint base = h * 128, qlo = h * 64, qho = h * 32, sco = h * 8;
        int isc = sco + l / 16;
        uchar l0 = ql[qlo + l], l1 = ql[qlo + l + 32], hb = qh[qho + l];
        o[base + l]      = (T)(df * sc[isc + 0] * (((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32));
        o[base + l + 32] = (T)(df * sc[isc + 2] * (((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32));
        o[base + l + 64] = (T)(df * sc[isc + 4] * (((l0 >> 4)  | (((hb >> 4) & 3) << 4)) - 32));
        o[base + l + 96] = (T)(df * sc[isc + 6] * (((l1 >> 4)  | (((hb >> 6) & 3) << 4)) - 32));
    }
"""
_GATHER = r"""
    uint i = thread_position_in_grid.x;
    uint nrows = tok_shape[0];
    if (i >= nrows * NSB) return;
    uint t = i / NSB, c = i % NSB;
    uint row = (uint)tok[t];
    q6k_decode_block<T>(W + ((size_t)row * NSB + (size_t)c) * 210, out + ((size_t)t * NSB + (size_t)c) * 256);
"""

# Q6_K for kernels built on its chain (the megakernel): a row in bf16, a superblock a thread as `_GATHER` decodes it
KERNELS: dict[Quant, StoredKernel] = {
    Quant.Q6_K: StoredKernel(
        MV,
        _DECODE,
        "",
        256,
        r"""
    for (uint c = tid; c < NSB; c += 256)
        q6k_decode_block<bfloat>(W + ((size_t)row * NSB + c) * 210, dst + (size_t)c * 256);
""",
    )
}

_deq_kernel: MetalKernel | None = None
_gather_kernels: dict[int, MetalKernel] = {}
_lock = threading.Lock()


def matvec_q6k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw Q6_K bytes (rows * cols / 256 row-major superblocks).
    float32 accumulation over the blocks as stored, y in x's dtype; 1..16 rows a launch, a wider pass chunked."""
    nsb = cols // 256
    return launch(f"btb_q6k_mv_{rows}_{nsb}", _MATVEC, "", [("W", w_bytes)], x, rows, cols, 256)


def gather_q6k(w_bytes: mx_.array, tok: mx_.array, cols: int) -> mx_.array:
    """the embedding rows for token ids `tok` [T] (int32): [T, cols] bf16, each row's superblocks dequantized
    straight from the packed bytes - no bf16 copy of the whole table."""
    m = mx
    nsb = cols // 256
    with _lock:
        k = _gather_kernels.get(nsb)
        if k is None:
            k = _gather_kernels[nsb] = metal_kernel(
                name=f"btb_q6k_gather_{nsb}",
                input_names=["W", "tok"],
                output_names=["out"],
                header=_DECODE + f"#define NSB {nsb}\n",
                source=_GATHER,
            )
    n = int(tok.shape[0]) * nsb
    grid = ((n + 255) // 256) * 256
    return k(
        inputs=[w_bytes, tok.astype(m.int32)],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(int(tok.shape[0]), cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16)],
    )[0]


def dequant_q6k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight, the numbers llama.cpp dequantizes (for a path that wants a bf16 copy)."""
    global _deq_kernel
    m = mx
    nblk = rows * cols // 256
    with _lock:
        if _deq_kernel is None:
            _deq_kernel = metal_kernel(
                name="btb_q6k_dequant", input_names=["W"], output_names=["out"], header="", source=_DEQUANT
            )
    grid = ((nblk * 32 + 255) // 256) * 256
    return _deq_kernel(
        inputs=[w_bytes],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]
