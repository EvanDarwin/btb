# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Q4_K on Metal: llama.cpp's Q4_K superblocks read as stored and dequantized in registers, so the 4-bit
weights never expand to bf16 in memory and each output row accumulates in fp32 independently of the batch
(row i is bit-identical for a 1-row and a 16-row call, which MLX's `quantized_matmul` is not). `matvec_q4k` is
the linear (`matvec.launch`: the 32 lanes of a simdgroup cooperating on a superblock for coalesced reads, up to 16
pass rows an invocation reusing the one weight read); `dequant_q4k` the whole matrix, for the rare path wanting a
bf16 copy. Native 144 B/256 as stored, vs MLX's 160 B repack."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from ..kinds import Quant
from .core import metal_kernel, mx
from .matvec import Frags, StoredKernel, blocks, launch

if TYPE_CHECKING:
    import mlx.core as mx_

# 144 bytes a superblock of 256 weights: d f16, dmin f16, 12 bytes packing a 6-bit scale and a 6-bit min for
# each of 8 sub-blocks of 32, then 128 bytes of nibbles (byte b = qs[32k + i] holds the low nibble for
# sub-block 2k and the high nibble for sub-block 2k+1, both at inner index i); value = d*sc[j]*q - dmin*mn[j],
# j = weight // 32. The 6-bit scale/min unpack is llama.cpp's get_scale_min_k4.
_SM = r"""
inline void q4k_scale_min(const device uchar* s, int j, thread float& sc, thread float& mn) {
    if (j < 4) { sc = (float)(s[j] & 63); mn = (float)(s[j + 4] & 63); }
    else {
        sc = (float)((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4));
        mn = (float)((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
    }
}
template <typename U>
inline void q4k_decode_block(const device uchar* blk, device U* o) {
    float d  = (float)(*(const device half*)(blk));
    float dm = (float)(*(const device half*)(blk + 2));
    const device uchar* s = blk + 4;
    const device uchar* qs = blk + 16;
    for (int k = 0; k < 4; k++) {
        int jlo = 2 * k, jhi = 2 * k + 1;
        float sclo, mnlo, schi, mnhi;
        q4k_scale_min(s, jlo, sclo, mnlo);
        q4k_scale_min(s, jhi, schi, mnhi);
        for (int i = 0; i < 32; i++) {
            uchar byte = qs[32 * k + i];
            o[jlo * 32 + i] = (U)(d * sclo * (byte & 0xF) - dm * mnlo);
            o[jhi * 32 + i] = (U)(d * schi * (byte >> 4) - dm * mnhi);
        }
    }
}
// Q5_K: 176 bytes a superblock. d f16, dmin f16, 12 scale bytes (get_scale_min_k4, as Q4_K), 32 bytes qh (the
// 5th bit of each weight), 128 bytes qs (low nibbles). q = nibble | (qh_bit << 4), bit 2k for sub-block 2k, 2k+1
// for 2k+1; value = d*sc[j]*q - dmin*mn[j].
template <typename U>
inline void q5k_decode_block(const device uchar* blk, device U* o) {
    float d  = (float)(*(const device half*)(blk));
    float dm = (float)(*(const device half*)(blk + 2));
    const device uchar* s = blk + 4;
    const device uchar* qh = blk + 16;
    const device uchar* qs = blk + 48;
    for (int k = 0; k < 4; k++) {
        int jlo = 2 * k, jhi = 2 * k + 1;
        float sclo, mnlo, schi, mnhi;
        q4k_scale_min(s, jlo, sclo, mnlo);
        q4k_scale_min(s, jhi, schi, mnhi);
        for (int i = 0; i < 32; i++) {
            uchar byte = qs[32 * k + i], h = qh[i];
            int qlo = (byte & 0xF) | (((h >> (2 * k)) & 1) << 4);
            int qhi = (byte >> 4)  | (((h >> (2 * k + 1)) & 1) << 4);
            o[jlo * 32 + i] = (U)(d * sclo * qlo - dm * mnlo);
            o[jhi * 32 + i] = (U)(d * schi * qhi - dm * mnhi);
        }
    }
}
"""

# The linears share one form (`matvec.blocks`): lane = inner index so the 32 lanes cooperate on a superblock
# (coalesced reads), eight weights a lane decoded into registers, then each pass row's eight activations applied
# to them in the column order the row's chain has always taken, and simd_sum. cols = NSB*256.
# Q4_K and Q5_K share a source (BLKB block bytes, QSOFF the nibble offset, Q5 adds the 5th bit from the qh plane at
# blk+16); a lane's weights pair as (sub-block 2k, 2k+1) at inner index lane.
MV_Q45 = Frags(
    "",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * BLKB;
            float d  = (float)(*(const device half*)(blk));
            float dm = (float)(*(const device half*)(blk + 2));
            const device uchar* s = blk + 4;
            const device uchar* qs = blk + QSOFF;
#if Q5
            uchar h = (blk + 16)[lane];
#endif
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                float sclo, mnlo, schi, mnhi;
                q4k_scale_min(s, 2 * k, sclo, mnlo);
                q4k_scale_min(s, 2 * k + 1, schi, mnhi);
                uchar byte = qs[32 * k + lane];
#if Q5
                int qlo = (byte & 0xF) | (((h >> (2 * k)) & 1) << 4);
                int qhi = (byte >> 4)  | (((h >> (2 * k + 1)) & 1) << 4);
#else
                int qlo = byte & 0xF, qhi = byte >> 4;
#endif
                w[r][2 * k] = d * sclo * qlo - dm * mnlo;
                w[r][2 * k + 1] = d * schi * qhi - dm * mnhi;
            }
""",
    r"""
            float xv[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[k] = (float)xc[k * 32 + lane];
            #pragma unroll
            for (int r = 0; r < RP; r++)
                #pragma unroll
                for (int k = 0; k < 4; k++) acc[r][tr] += w[r][2 * k] * xv[2 * k] + w[r][2 * k + 1] * xv[2 * k + 1];
""",
)
_MATVEC = blocks(MV_Q45)
# Q2_K and Q3_K: a lane's eight weights at col h*128 + j*32 + lane (h 0..1, j 0..3), one a term
_APPLY_HJ = r"""
            float xv[8];
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[k] = (float)xc[(k >> 2) * 128 + (k & 3) * 32 + lane];
            #pragma unroll
            for (int r = 0; r < RP; r++)
                #pragma unroll
                for (int k = 0; k < 8; k++) acc[r][tr] += w[r][k] * xv[k];
"""

# one thread a superblock; `dequant` walks the matrix.
_DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= NBLK) return;
#if Q5
    q5k_decode_block<T>(W + (size_t)t * BLKB, out + (size_t)t * 256);
#else
    q4k_decode_block<T>(W + (size_t)t * BLKB, out + (size_t)t * 256);
#endif
"""

# Q4_K: 144 B/superblock, nibbles at offset 16, no qh. Q5_K: 176 B, nibbles at 48, the 5th bit in qh at 16.
_KQ = {"q4k": (144, 16, 0), "q5k": (176, 48, 1)}
_deq_kernels: dict[str, Any] = {}
_lock = threading.Lock()


def _defs(kind: str) -> str:
    blkb, qsoff, q5 = _KQ[kind]
    return f"#define BLKB {blkb}\n#define QSOFF {qsoff}\n#define Q5 {q5}\n"


# a row of the matrix in bf16, a superblock a thread as `_DEQUANT` decodes it (the step's tied embedding)
_DECODE_ROW = r"""
    for (uint c = tid; c < NSB; c += 256) {
#if Q5
        q5k_decode_block<bfloat>(W + ((size_t)row * NSB + c) * BLKB, dst + (size_t)c * 256);
#else
        q4k_decode_block<bfloat>(W + ((size_t)row * NSB + c) * BLKB, dst + (size_t)c * 256);
#endif
    }
"""


def _matvec(kind: str, w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    nsb = cols // 256
    return launch(f"btb_{kind}_mv_{rows}_{nsb}", _MATVEC, f"{_SM}{_defs(kind)}", [("W", w_bytes)], x, rows, cols, 256)


def _dequant(kind: str, w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    m = mx()
    nblk = rows * cols // 256
    with _lock:
        k = _deq_kernels.get(kind)
        if k is None:
            k = _deq_kernels[kind] = metal_kernel(
                name=f"btb_{kind}_dequant",
                input_names=["W"],
                output_names=["out"],
                header=f"{_SM}{_defs(kind)}",
                source=_DEQUANT,
            )
    grid = ((nblk + 255) // 256) * 256
    return k(
        inputs=[w_bytes],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


def matvec_q4k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W[rows, cols]^T, W the raw Q4_K bytes (rows * cols / 256 row-major superblocks).
    float32 accumulation over the blocks as stored, y in x's dtype; 1..16 rows a launch, a wider pass chunked."""
    return _matvec("q4k", w_bytes, x, rows, cols)


def dequant_q4k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight, the numbers llama.cpp dequantizes (for a path that wants a bf16 copy)."""
    return _dequant("q4k", w_bytes, rows, cols)


def matvec_q5k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q5_K (176 B superblocks, the 5th bit in the qh plane)."""
    return _matvec("q5k", w_bytes, x, rows, cols)


def dequant_q5k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for a Q5_K tensor."""
    return _dequant("q5k", w_bytes, rows, cols)


# Q2_K: 84 B/superblock. scales[16] (a 4-bit scale and 4-bit min a 16-weight sub-block), qs[64] (2-bit), d f16,
# dmin f16. ggml order: two halves (h), qs advances 32 a half; four groups (j, shift 2j) a half; two sub-blocks
# a group (l on q[l], q[l+16]). value = d*(scales[si]&0xF)*q - dmin*(scales[si]>>4), col = h*128 + j*32 + m.
# lane = m (0..31): reads qs[h*32+m], the four j groups share the byte at shifts 0/2/4/6. Register-only,
# batch-invariant.
MV_Q2K = Frags(
    "    uint sub = lane >> 4;\n",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 84;
            const device uchar* scales = blk;
            const device uchar* qs = blk + 16;
            float d  = (float)(*(const device half*)(blk + 80));
            float dm = (float)(*(const device half*)(blk + 82));
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                uchar byte = qs[h * 32 + lane];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    uchar s = scales[h * 8 + 2 * j + sub];
                    w[r][4 * h + j] = d * (s & 0xF) * ((byte >> (2 * j)) & 3) - dm * (s >> 4);
                }
            }
""",
    _APPLY_HJ,
)
_MATVEC_Q2K = blocks(MV_Q2K)
# a superblock's eight weights of lane `lane` (cols h*128 + j*32 + lane) into o: the dequant's, and the embedding's
_Q2K_LANE = r"""
template <typename U>
inline void q2k_decode_lane(const device uchar* blk, uint lane, device U* o) {
    uint sub = lane >> 4;
    const device uchar* scales = blk;
    const device uchar* qs = blk + 16;
    float d  = (float)(*(const device half*)(blk + 80));
    float dm = (float)(*(const device half*)(blk + 82));
    for (int h = 0; h < 2; h++) {
        uchar byte = qs[h * 32 + lane];
        for (int j = 0; j < 4; j++) {
            uchar s = scales[h * 8 + 2 * j + sub];
            o[h * 128 + j * 32 + lane] = (U)(d * (s & 0xF) * ((byte >> (2 * j)) & 3) - dm * (s >> 4));
        }
    }
}
"""
_DEQUANT_Q2K = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    q2k_decode_lane<T>(W + (size_t)blk_i * 84, lane, out + (size_t)blk_i * 256);
"""
_q2k_deq = None


def matvec_q2k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q2_K (84 B superblocks, 2-bit, 16-weight sub-blocks with 4-bit scale/min)."""
    nsb = cols // 256
    return launch(f"btb_q2k_mv_{rows}_{nsb}", _MATVEC_Q2K, "", [("W", w_bytes)], x, rows, cols, 256)


def dequant_q2k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for a Q2_K tensor."""
    global _q2k_deq
    m = mx()
    nblk = rows * cols // 256
    with _lock:
        if _q2k_deq is None:
            _q2k_deq = metal_kernel(
                name="btb_q2k_dequant", input_names=["W"], output_names=["out"], header=_Q2K_LANE, source=_DEQUANT_Q2K
            )
    grid = ((nblk * 32 + 255) // 256) * 256
    return _q2k_deq(
        inputs=[w_bytes],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


# Q3_K: 110 B/superblock. hmask[32] (the 3rd bit of each weight), qs[64] (2 low bits), scales[12] (16 signed
# 6-bit scales packed with the kmask dance), d f16. col = h*128 + j*32 + sub*16 + l (h 0..1, j 0..3, sub 0..1,
# l 0..15); q = ((qs[h*32+sub*16+l] >> 2j) & 3) - (hmask bit h*4+j set ? 0 : 4); value = d*scale[h*8+2j+sub]*q.
# lane = sub*16 + l (0..31): reads hmask[lane] once and qs[h*32+lane] a half; the 16 scales unpack once a
# superblock into registers. Register-only, batch-invariant.
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
// a superblock's eight weights of lane `lane` (cols h*128 + j*32 + lane) into o: the dequant's, and the embedding's
template <typename U>
inline void q3k_decode_lane(const device uchar* blk, uint lane, device U* o) {
    uint sub = lane >> 4;
    const device uchar* qs = blk + 32;
    thread uint aux[4];
    q3k_aux(blk + 96, aux);
    const thread char* scl = (const thread char*)aux;
    float d = (float)(*(const device half*)(blk + 108));
    uchar hm = blk[lane];
    for (int h = 0; h < 2; h++) {
        uchar byte = qs[h * 32 + lane];
        for (int j = 0; j < 4; j++) {
            int q = (byte >> (2 * j)) & 3;
            if (!(hm & (1 << (h * 4 + j)))) q -= 4;
            o[h * 128 + j * 32 + lane] = (U)(d * (float)((int)scl[h * 8 + 2 * j + sub] - 32) * (float)q);
        }
    }
}
"""
MV_Q3K = Frags(
    "    uint sub = lane >> 4;\n",
    "        float w[RP][8];\n",
    r"""
            const device uchar* blk = W + ((size_t)row * NSB + c) * 110;
            const device uchar* qs = blk + 32;
            thread uint aux[4];
            q3k_aux(blk + 96, aux);
            const thread char* scl = (const thread char*)aux;
            float d = (float)(*(const device half*)(blk + 108));
            uchar hm = blk[lane];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                uchar byte = qs[h * 32 + lane];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int q = (byte >> (2 * j)) & 3;
                    if (!(hm & (1 << (h * 4 + j)))) q -= 4;
                    w[r][4 * h + j] = d * (float)((int)scl[h * 8 + 2 * j + sub] - 32) * (float)q;
                }
            }
""",
    _APPLY_HJ,
)
_MATVEC_Q3K = blocks(MV_Q3K)
_DEQUANT_Q3K = r"""
    uint gid = thread_position_in_grid.x;
    uint blk_i = gid / 32, lane = gid % 32;
    if (blk_i >= NBLK) return;
    q3k_decode_lane<T>(W + (size_t)blk_i * 110, lane, out + (size_t)blk_i * 256);
"""
_q3k_deq = None


def matvec_q3k(w_bytes: mx_.array, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """`matvec_q4k` for Q3_K (110 B superblocks, 3-bit split qs+hmask, 16 signed 6-bit scales)."""
    nsb = cols // 256
    return launch(f"btb_q3k_mv_{rows}_{nsb}", _MATVEC_Q3K, _Q3K_AUX, [("W", w_bytes)], x, rows, cols, 256)


def dequant_q3k(w_bytes: mx_.array, rows: int, cols: int) -> mx_.array:
    """the full [rows, cols] bf16 weight for a Q3_K tensor."""
    global _q3k_deq
    m = mx()
    nblk = int(rows * cols // 256)
    with _lock:
        if _q3k_deq is None:
            _q3k_deq = metal_kernel(
                name="btb_q3k_dequant", input_names=["W"], output_names=["out"], header=_Q3K_AUX, source=_DEQUANT_Q3K
            )
    grid = ((nblk * 32 + 255) // 256) * 256
    return _q3k_deq(
        inputs=[w_bytes],
        grid=(grid, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[m.bfloat16],
        template=[("T", m.bfloat16), ("NBLK", nblk)],
    )[0]


def _decode_lanes(fn: str, blkb: int) -> str:
    """a row of the matrix in bf16, a (superblock, lane) a thread as the dequant decodes it through `fn` (the step's
    tied embedding)"""
    return (
        "    for (uint i = tid; i < NSB * 32; i += 256)\n"
        f"        {fn}<bfloat>(W + ((size_t)row * NSB + i / 32) * {blkb}, i % 32, dst + (size_t)(i / 32) * 256);\n"
    )


KERNELS: dict[Quant, StoredKernel] = {
    Quant.Q2_K: StoredKernel(MV_Q2K, _Q2K_LANE, "", 256, _decode_lanes("q2k_decode_lane", 84)),
    Quant.Q3_K: StoredKernel(MV_Q3K, _Q3K_AUX, "", 256, _decode_lanes("q3k_decode_lane", 110)),
    Quant.Q4_K: StoredKernel(MV_Q45, _SM, _defs("q4k"), 256, _DECODE_ROW),
    Quant.Q5_K: StoredKernel(MV_Q45, _SM, _defs("q5k"), 256, _DECODE_ROW),
}
