# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's own matmul kernels on Metal: the 12-bit unpack, the batch-invariant matvec for a tile of up to 16
rows (8x8 matrix multiplies, each row its own chain), and the MXFP4 kernels for gpt-oss's experts."""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_
_UNPACK_SRC = """
    // one thread per pair of weights (one byte of codes): out[2i] and out[2i + 1]
    uint i = thread_position_in_grid.x;
    uint n = lo_shape[0];
    uint j = 2 * i;
    if (j >= n) return;
    uint8_t c = hi4[i];
    out[j] = (uint16_t)(((uint16_t)table[c & 15] << 8) | (uint16_t)lo[j]);
    if (j + 1 < n) out[j + 1] = (uint16_t)(((uint16_t)table[c >> 4] << 8) | (uint16_t)lo[j + 1]);
"""

_unpack_kernel = None


def unpack_bf16(
    lo: mx_.array,
    hi4: mx_.array,
    table: mx_.array,
    n: int,
    shape: Sequence[int],
    esc_idx: Any = None,
    esc_val: Any = None,
) -> mx_.array:
    """The 12-bit store's inverse on the GPU, bit-identical to `engine.unpack_bf16`: `lo` u8[n], `hi4` u8[(n+1)/2],
    `table` u8[16], `esc_idx` i32[k] / `esc_val` u8[k] (or None); the escapes scattered in after one pass."""
    global _unpack_kernel
    m = mx()
    with _gemv_lock:
        if _unpack_kernel is None:
            _unpack_kernel = m.fast.metal_kernel(
                name="btb_unpack_p12", input_names=["lo", "hi4", "table"], output_names=["out"], source=_UNPACK_SRC
            )
    pairs = (n + 1) // 2
    out = _unpack_kernel(
        inputs=[lo, hi4, table],
        grid=(pairs, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[m.uint16],
    )[0]
    if esc_idx is not None and esc_idx.size:
        out[esc_idx] = (esc_val.astype(m.uint16) << 8) | m.take(lo, esc_idx).astype(m.uint16)
    return out.view(m.bfloat16).reshape(*shape)


GEMV_ROWS = 16

_GEMM16_HEADER = "#include <metal_simdgroup_matrix>\n"

_GEMM16_SRC = """
    // y[b, n] = sum_k x[b, k] w[n, k] for NB <= 16 rows of x. A simdgroup owns 8 rows of w and streams K in steps
    // of 64: its lanes stage the 8 x 64 weight tile and the 16 x 64 activation tile (rows past NB zero) in the
    // simdgroup's own threadgroup memory, then eight 8x8 matrix multiply-accumulates into float accumulators;
    // no threadgroup barrier, the next step's pieces loading while this one computes. A row's chain is the
    // hardware's fixed order over K whatever else the tile holds, so row b is the same bits alone or in a
    // 16-row tile, and the tile's rows cost no memory traffic beyond the one weight read.
    // XB: bf16 x and bf16 tiles, y bf16 (round to nearest even); else float32 x and float tiles (the weight
    // widened at the stage), y float32.
    constexpr uint KT = 64;
    constexpr uint SGN = XB ? 8 : 4;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * SGN + sg) * 8;
    uint N = w_shape[0];
    uint K = w_shape[1];
    if (n0 >= N) return;
#if XB
    typedef bfloat ST;
    typedef simdgroup_bfloat8x8 SM;
#else
    typedef float ST;
    typedef simdgroup_float8x8 SM;
#endif
    threadgroup ST ws_all[SGN][8 * KT];
    threadgroup ST xs_all[SGN][16 * KT];
    threadgroup ST* ws = ws_all[sg];
    threadgroup ST* xs = xs_all[sg];
    // the weight tile: lane l stages uint4 l and l + 32 of its 8 x 8 uint4s (row i / 8, k (i % 8) * 8)
    uint wr0 = lane / 8, wr1 = 4 + lane / 8, wc = (lane % 8) * 8;
    const device uint4* wp0 = (const device uint4*)(w + (size_t)min(n0 + wr0, N - 1) * K + wc);
    const device uint4* wp1 = (const device uint4*)(w + (size_t)min(n0 + wr1, N - 1) * K + wc);
    bool ok0 = n0 + wr0 < N, ok1 = n0 + wr1 < N;
    // the activation tile: rows lane / 8 + 4i; the second 8-row half only past 8 rows
    constexpr uint XR = NB > 8 ? 4 : 2;
    uint xr[4];
    bool xok[4];
    const device uint4* xp[4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) {
        xr[i] = lane / 8 + 4 * i;
        xok[i] = xr[i] < NB;
        xp[i] = (const device uint4*)(x + (size_t)min(xr[i], (uint)NB - 1) * K + wc);
    }
    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
    uint nstep = (K + KT - 1) / KT;
    uint4 w0 = uint4(0u), w1 = uint4(0u), xv[4], xv2[4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) { xv[i] = uint4(0u); xv2[i] = uint4(0u); }
    if (ok0 && wc < K) w0 = wp0[0];
    if (ok1 && wc < K) w1 = wp1[0];
    #pragma unroll
    for (uint i = 0; i < XR; ++i) if (xok[i] && wc < K) { xv[i] = xp[i][0]; if (!XB) xv2[i] = xp[i][1]; }
    for (uint st = 0; st < nstep; ++st) {
#if XB
        *((threadgroup uint4*)(ws + wr0 * KT + wc)) = w0;
        *((threadgroup uint4*)(ws + wr1 * KT + wc)) = w1;
        #pragma unroll
        for (uint i = 0; i < XR; ++i) *((threadgroup uint4*)(xs + xr[i] * KT + wc)) = xv[i];
#else
        {
            threadgroup float4* d0 = (threadgroup float4*)(ws + wr0 * KT + wc);
            threadgroup float4* d1 = (threadgroup float4*)(ws + wr1 * KT + wc);
            d0[0] = float4(as_type<float>(w0.x << 16), as_type<float>(w0.x & 0xffff0000u),
                           as_type<float>(w0.y << 16), as_type<float>(w0.y & 0xffff0000u));
            d0[1] = float4(as_type<float>(w0.z << 16), as_type<float>(w0.z & 0xffff0000u),
                           as_type<float>(w0.w << 16), as_type<float>(w0.w & 0xffff0000u));
            d1[0] = float4(as_type<float>(w1.x << 16), as_type<float>(w1.x & 0xffff0000u),
                           as_type<float>(w1.y << 16), as_type<float>(w1.y & 0xffff0000u));
            d1[1] = float4(as_type<float>(w1.z << 16), as_type<float>(w1.z & 0xffff0000u),
                           as_type<float>(w1.w << 16), as_type<float>(w1.w & 0xffff0000u));
            #pragma unroll
            for (uint i = 0; i < XR; ++i) {
                threadgroup uint4* dx = (threadgroup uint4*)(xs + xr[i] * KT + wc);
                dx[0] = xv[i];
                dx[1] = xv2[i];
            }
        }
#endif
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint kn = (st + 1) * KT;
        bool more = st + 1 < nstep;
        w0 = (more && ok0 && kn + wc < K) ? wp0[(st + 1) * (KT / 8)] : uint4(0u);
        w1 = (more && ok1 && kn + wc < K) ? wp1[(st + 1) * (KT / 8)] : uint4(0u);
        #pragma unroll
        for (uint i = 0; i < XR; ++i) {
            bool g = more && xok[i] && kn + wc < K;
            if (XB) {
                xv[i] = g ? xp[i][(st + 1) * (KT / 8)] : uint4(0u);
            } else {
                xv[i] = g ? xp[i][(st + 1) * (KT / 4)] : uint4(0u);
                xv2[i] = g ? xp[i][(st + 1) * (KT / 4) + 1] : uint4(0u);
            }
        }
        #pragma unroll
        for (uint ks = 0; ks < KT / 8; ++ks) {
            SM A0, A1, B;
            simdgroup_load(B, ws + ks * 8, KT, ulong2(0, 0), true);
            simdgroup_load(A0, xs + ks * 8, KT);
            simdgroup_multiply_accumulate(acc0, A0, B, acc0);
            if (NB > 8) {
                simdgroup_load(A1, xs + 8 * KT + ks * 8, KT);
                simdgroup_multiply_accumulate(acc1, A1, B, acc1);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* out = (threadgroup float*)xs;
    simdgroup_store(acc0, out, 8);
    if (NB > 8) simdgroup_store(acc1, out + 64, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint o = lane; o < NB * 8; o += 32) {
        uint b = o / 8, n = o % 8;
        if (n0 + n < N) {
#if XB
            uint u = as_type<uint>(out[o]);
            u += 0x7FFFu + ((u >> 16) & 1u);
            y[(size_t)b * N + n0 + n] = (uint16_t)(u >> 16);
#else
            y[(size_t)b * N + n0 + n] = out[o];
#endif
        }
    }
"""


_gemm16_kernels: dict[int, Any] = {}
_gemv_lock = threading.Lock()


def _gemm16_kernel(xb: int) -> Any:
    m = mx()
    with _gemv_lock:
        k = _gemm16_kernels.get(xb)
        if k is None:
            k = _gemm16_kernels[xb] = m.fast.metal_kernel(
                name=f"btb_gemm16_x{xb}",
                input_names=["w", "x"],
                output_names=["y"],
                header=_GEMM16_HEADER + f"#define XB {xb}\n",
                source=_GEMM16_SRC,
            )
    return k


def gemv(w_bf16: mx_.array, x: mx_.array) -> mx_.array:
    """y[b, rows] = x[b, cols] w[rows, cols]^T for 1..16 rows of x (bf16 or float32), float32 accumulation over
    the bf16 weight as it is, y in x's dtype. Batch-invariant: row i of y is the same bits for 1 row or 16.
    Needs cols % 8 == 0."""
    m = mx()
    rows = int(w_bf16.shape[0])
    b = int(x.shape[0])
    if not 1 <= b <= GEMV_ROWS:
        raise ValueError(f"[mlx] the matvec takes 1..{GEMV_ROWS} rows, got {b}")
    xb = x.dtype == m.bfloat16
    sgn = 8 if xb else 4
    tg = (rows + 8 * sgn - 1) // (8 * sgn)
    y = _gemm16_kernel(1 if xb else 0)(
        inputs=[w_bf16.view(m.uint16), x.view(m.uint16) if xb else x],
        template=[("NB", b)],
        grid=(tg * 32 * sgn, 1, 1),
        threadgroup=(32 * sgn, 1, 1),
        output_shapes=[(b, rows)],
        output_dtypes=[m.uint16 if xb else m.float32],
    )[0]
    return y.view(m.bfloat16) if xb else y


@functools.cache
def read_bps() -> float:
    """The rate the GPU reads weights at, bytes a second: this matvec (what an MLX pass is made of, and bound by
    its reads) for one row over a 256 MiB bf16 weight, eight launches a reading, the best of five. Measured once
    for the process; a plan prices an MLX pass from it."""
    m = mx()
    rows, cols = 16384, 8192
    w = m.random.normal((rows, cols)).astype(m.bfloat16)
    x = m.ones((1, cols), dtype=m.bfloat16)
    m.eval(w, x, gemv(w, x))
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        m.eval([gemv(w, x) for _ in range(8)])
        best = min(best, (time.perf_counter() - t0) / 8)
    del w, x
    m.clear_cache()
    return rows * cols * 2 / best


def gemv_f32(w_bf16: mx_.array, x_f32: mx_.array) -> mx_.array:
    """`gemv` for float32 x (the name the tests and the older callers use)."""
    return gemv(w_bf16, x_f32)


MXFP4_BLOCK = 32
MXFP4_BLOCK_BYTES = 16
MXFP4_GGML_BYTES = 17  # ggml's block: the scale byte, then the 16 nibble bytes

_MXFP4_HEADER = """
constant float BTB_FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// 2**(e - 127) exactly: the exponent field of a float32, stepped down once for e == 0 (2**-127 is below the
// smallest normal, so it cannot be written as an exponent field on its own). e == 255 is a NaN scale in the
// MX spec and is not produced by any checkpoint; it would come out here as an infinity.
inline float btb_mxfp4_scale(uint8_t e) {
    uint b = (uint)(e < 1 ? 1 : e) << 23;
    float f = as_type<float>(b);
    return e == 0 ? f * 0.5f : f;
}
"""

_GEMV_MXFP4_SRC = """
    // one simdgroup per TM output rows: the 32 lanes stride the row's K / 32 blocks and reduce. A block's
    // 16 bytes are read as one uint4 and dequantized eight weights at a time (one 32-bit word), the block's
    // scale multiplied into each weight, so the chain per output is: blocks in increasing order, and inside
    // a block the weights in increasing K. That chain is written out with explicit fma and does not depend
    // on how many rows of x the call carries, so row i of y is the same bits alone as in a 16-row tile --
    // the verify pass computes each row exactly as the one-row step does. NB is the row count (a
    // compile-time constant, 1..16), TM the output rows per simdgroup, ROWS/KDIM the matrix shape.
    // XB: x is bf16 bits; YB: y is written as bf16 bits (round to nearest even). GGML: the matrix is in
    // ggml's layout (17-byte blocks, the scale first, low nibbles weights 0..15, high nibbles 16..31), read
    // as stored; the eight weights of a chunk are the same weights in the same order either way. PAIR (with
    // GGML): the matrix is gate over up, two buffers of ROWS / 2 rows each (w, s), and row r's output goes
    // to the checkpoint's interleaved position (2r for gate, 2(r - ROWS/2) + 1 for up).
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint mi = threadgroup_position_in_grid.z;
    WSEL_
    const uint rows = ROWS;
    const uint NBLK = KDIM / 32;
    uint r0 = (threadgroup_position_in_grid.x * 8 + sg) * TM;
    if (r0 >= rows) return;
    uint nm = min((uint)TM, rows - r0);
    const device uint4* wq = (const device uint4*)wb_;
    const device uint8_t* sc8 = sb_;
    float acc[TM][NB];
    #pragma unroll
    for (uint mI = 0; mI < TM; ++mI) for (uint j = 0; j < NB; ++j) acc[mI][j] = 0.0f;
    for (uint blk = lane; blk < NBLK; blk += 32) {
        uint4 pk[TM];
        float sf[TM];
        #pragma unroll
        for (uint mI = 0; mI < TM; ++mI) {
            pk[mI] = uint4(0u);
            sf[mI] = 0.0f;
            if (mI < nm) {
                size_t at = (size_t)(r0 + mI) * NBLK + blk;
                if (GGML) {
                    // the block's 16 data bytes start at 17 * at + 1, any alignment: five 4-byte-aligned word
                    // loads and a fixed shift give the four words the checkpoint layout reads as one uint4;
                    // the buffer's last word is assembled byte by byte so no load runs past it
                    const uint HALF = PAIR ? ROWS / 2 : ROWS;
                    const size_t WBYTES = (size_t)HALF * NBLK * 17;
                    uint r = r0 + mI;
                    const device uint8_t* wsrc = (PAIR && r >= HALF) ? sb_ : wb_;
                    if (PAIR && r >= HALF) at = (size_t)(r - HALF) * NBLK + blk;
                    size_t o = at * 17 + 1;
                    uint sh = 8 * (uint)(o & 3);
                    size_t a4 = o - (o & 3);
                    const device uint* p = (const device uint*)(wsrc + a4);
                    uint w0 = p[0], w1 = p[1], w2 = p[2], w3 = p[3], w4 = 0u;
                    if (a4 + 20 <= WBYTES) {
                        w4 = p[4];
                    } else {
                        for (uint j = 0; j < 4; ++j) {
                            size_t idx = a4 + 16 + j;
                            if (idx < WBYTES) w4 |= (uint)wsrc[idx] << (8 * j);
                        }
                    }
                    // branchless funnel: the high part shifted by 31 - sh then 1 more, so sh == 0 contributes 0
                    uint rs = 31 - sh;
                    pk[mI] = uint4((w0 >> sh) | ((w1 << rs) << 1), (w1 >> sh) | ((w2 << rs) << 1),
                                   (w2 >> sh) | ((w3 << rs) << 1), (w3 >> sh) | ((w4 << rs) << 1));
                    sf[mI] = btb_mxfp4_scale(wsrc[o - 1]);
                } else {
                    pk[mI] = wq[at];
                    sf[mI] = btb_mxfp4_scale(sc8[at]);
                }
            }
        }
        #pragma unroll
        for (uint c = 0; c < 4; ++c) {
            float wv[TM][8];
            #pragma unroll
            for (uint mI = 0; mI < TM; ++mI) {
                float s_ = sf[mI];
                if (GGML) {
                    // chunk c: weights 16 * (c / 2) + 8 * (c % 2) .. + 7, the (c / 2) nibble of bytes 8 * (c % 2) ..
                    uint lo_w = (c & 1) == 0 ? pk[mI].x : pk[mI].z;
                    uint hi_w = (c & 1) == 0 ? pk[mI].y : pk[mI].w;
                    uint sh = (c >> 1) * 4;
                    #pragma unroll
                    for (uint t = 0; t < 4; ++t) wv[mI][t] = BTB_FP4[(lo_w >> (8 * t + sh)) & 15u] * s_;
                    #pragma unroll
                    for (uint t = 0; t < 4; ++t) wv[mI][4 + t] = BTB_FP4[(hi_w >> (8 * t + sh)) & 15u] * s_;
                } else {
                    uint u = c == 0 ? pk[mI].x : (c == 1 ? pk[mI].y : (c == 2 ? pk[mI].z : pk[mI].w));
                    #pragma unroll
                    for (uint t = 0; t < 8; ++t) wv[mI][t] = BTB_FP4[(u >> (4 * t)) & 15u] * s_;
                }
            }
            uint kb = blk * 32 + c * 8;
            #pragma unroll
            for (uint j = 0; j < NB; ++j) {
                float x0, x1, x2, x3, x4, x5, x6, x7;
                size_t xr = (size_t)(mi * NB + j) * KDIM + kb;
                if (XB) {
                    uint4 xp = *((const device uint4*)(x + xr));
                    x0 = as_type<float>(xp.x << 16); x1 = as_type<float>(xp.x & 0xffff0000u);
                    x2 = as_type<float>(xp.y << 16); x3 = as_type<float>(xp.y & 0xffff0000u);
                    x4 = as_type<float>(xp.z << 16); x5 = as_type<float>(xp.z & 0xffff0000u);
                    x6 = as_type<float>(xp.w << 16); x7 = as_type<float>(xp.w & 0xffff0000u);
                } else {
                    const device float4* xi = (const device float4*)(x + xr);
                    float4 xa = xi[0];
                    float4 xbv = xi[1];
                    x0 = xa.x; x1 = xa.y; x2 = xa.z; x3 = xa.w;
                    x4 = xbv.x; x5 = xbv.y; x6 = xbv.z; x7 = xbv.w;
                }
                #pragma unroll
                for (uint mI = 0; mI < TM; ++mI) {
                    float a_ = acc[mI][j];
                    a_ = fma(wv[mI][0], x0, a_); a_ = fma(wv[mI][1], x1, a_);
                    a_ = fma(wv[mI][2], x2, a_); a_ = fma(wv[mI][3], x3, a_);
                    a_ = fma(wv[mI][4], x4, a_); a_ = fma(wv[mI][5], x5, a_);
                    a_ = fma(wv[mI][6], x6, a_); a_ = fma(wv[mI][7], x7, a_);
                    acc[mI][j] = a_;
                }
            }
        }
    }
    #pragma unroll
    for (uint mI = 0; mI < TM; ++mI) {
        #pragma unroll
        for (uint j = 0; j < NB; ++j) {
            float s = simd_sum(acc[mI][j]);
            if (lane == 0 && mI < nm) {
                uint r = r0 + mI;
                if (PAIR) r = r < ROWS / 2 ? 2 * r : 2 * (r - ROWS / 2) + 1;
                size_t at = (size_t)(mi * NB + j) * rows + r;
                if (YB) {
                    uint u = as_type<uint>(s);
                    u += 0x7FFFu + ((u >> 16) & 1u);
                    y[at] = (uint16_t)(u >> 16);
                } else {
                    y[at] = s;
                }
            }
        }
    }
"""

_MXFP4_DEQUANT_SRC = """
    // one thread per block: 32 weights out of 16 bytes and one exponent, in K order (GGML: ggml's layout)
    uint i = thread_position_in_grid.x;
    const uint NBLK = KDIM / 32;
    uint total = ROWS * NBLK;
    if (i >= total) return;
    device OT* o = out + (size_t)i * 32;
    if (GGML) {
        const device uint8_t* bp = w + (size_t)i * 17;
        float sf = btb_mxfp4_scale(bp[0]);
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            uint8_t byte = bp[1 + j];
            o[j] = (OT)(BTB_FP4[byte & 15u] * sf);
            o[16 + j] = (OT)(BTB_FP4[byte >> 4] * sf);
        }
        return;
    }
    uint4 pk = ((const device uint4*)w)[i];
    float sf = btb_mxfp4_scale(s[i]);
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        uint u = c == 0 ? pk.x : (c == 1 ? pk.y : (c == 2 ? pk.z : pk.w));
        #pragma unroll
        for (uint t = 0; t < 8; ++t) o[c * 8 + t] = (OT)(BTB_FP4[(u >> (4 * t)) & 15u] * sf);
    }
"""

# output rows per simdgroup, at one row and at several: four; the tile never changes a row's arithmetic
# (`check_mxfp4_gemv`). Eight at one row measured a tenth slower at gpt-oss's expert shapes, either layout.
MXFP4_TM = 4
MXFP4_TM1 = 4

_mxfp4_kernels: dict[Any, Any] = {}
_mxfp4_dequant_kernel = None


def mxfp4_bytes(rows: int, k: int) -> int:
    """The bytes one `[rows, k]` MXFP4 matrix takes in a slot: its blocks then its scales."""
    rows, k = int(rows), int(k)
    if k % MXFP4_BLOCK:
        raise ValueError(f"[mlx] K = {k} is not a multiple of {MXFP4_BLOCK}")
    g = k // MXFP4_BLOCK
    return rows * g * MXFP4_BLOCK_BYTES + rows * g


def mxfp4_split(slot: mx_.array, rows: int, k: int, ggml: bool = False) -> tuple[mx_.array, mx_.array]:
    """A matrix's (blocks, scales) uint8 views out of one slot view (MLX slices share the bytes); a pair passes
    through. In ggml's layout the blocks carry the scales: (raw, raw)."""
    if isinstance(slot, (tuple, list)):
        b, s = slot
        return b.reshape(-1), _mxfp4_pad(s.reshape(-1))
    a = slot.reshape(-1)
    g = int(k) // MXFP4_BLOCK
    if ggml:
        need = int(rows) * g * MXFP4_GGML_BYTES
        if int(a.size) < need:
            raise ValueError(f"[mlx] a ggml MXFP4 [{rows}, {k}] matrix needs {need} bytes, the view has {int(a.size)}")
        raw = a[:need]
        return raw, raw
    nb = int(rows) * g * MXFP4_BLOCK_BYTES
    need = nb + int(rows) * g
    if int(a.size) < need:
        raise ValueError(f"[mlx] an MXFP4 [{rows}, {k}] matrix needs {need} bytes, the slot view has {int(a.size)}")
    return a[:nb], _mxfp4_pad(a[nb:need])


def _mxfp4_pad(a: mx_.array) -> mx_.array:
    """MLX passes an input of <= 4 elements in the constant address space, where a block pointer cannot live:
    pad a fixture-sized input into a real buffer."""
    m = mx()
    if int(a.size) > 4:
        return a
    return m.concatenate([a, m.zeros((8 - int(a.size),), dtype=a.dtype)])


def _mxfp4_kernel(n: int, ggml: bool = False, pair: bool = False) -> Any:
    """the matvec over n matrices in one dispatch (n = 1 is the single call), compiled once per (n, layout)"""
    m = mx()
    with _gemv_lock:
        k = _mxfp4_kernels.get((n, ggml, pair))
        if k is None:
            names = []
            for i in range(n):
                names += [f"w{i}", f"s{i}"]
            sel = "const device uint8_t* wb_ = w0;\n    const device uint8_t* sb_ = s0;"
            for i in range(1, n):
                sel += f"\n    if (mi == {i}) {{ wb_ = w{i}; sb_ = s{i}; }}"
            k = m.fast.metal_kernel(
                name=f"btb_gemv_mxfp4{'_ggml' if ggml else ''}{'_pair' if pair else ''}_x{n}",
                input_names=[*names, "x"],
                output_names=["y"],
                header=_MXFP4_HEADER,
                source=_GEMV_MXFP4_SRC.replace("WSEL_", sel),
            )
            _mxfp4_kernels[(n, ggml, pair)] = k
    return k


def _mxfp4_call(
    mats: Sequence[Any],
    rows: int,
    k: int,
    x: mx_.array,
    tm: int | None = None,
    ggml: bool = False,
    pair: bool = False,
) -> mx_.array:
    """the dispatch behind `gemv_mxfp4` / `gemv_mxfp4_group`: `mats` the (blocks, scales) views, `x` [n, b, k]"""
    m = mx()
    n = len(mats)
    rows, k = int(rows), int(k)
    b = int(x.shape[1])
    if b > GEMV_ROWS:
        raise ValueError(f"[mlx] the MXFP4 matvec takes 1..{GEMV_ROWS} rows, got {b}")
    if int(x.shape[-1]) != k:
        raise ValueError(f"[mlx] x has {int(x.shape[-1])} columns, the matrix has K = {k}")
    xb = x.dtype == m.bfloat16
    tm = int(tm or (MXFP4_TM1 if b == 1 else MXFP4_TM))
    tg = (rows + 8 * tm - 1) // (8 * tm)
    ins = []
    for w, s in mats:
        ins += [w, s]
    y = _mxfp4_kernel(n, ggml, pair)(
        inputs=[*ins, x.view(m.uint16) if xb else x],
        template=[
            ("XB", 1 if xb else 0),
            ("YB", 1 if xb else 0),
            ("TM", tm),
            ("NB", b),
            ("ROWS", rows),
            ("KDIM", k),
            ("GGML", 1 if ggml else 0),
            ("PAIR", 1 if pair else 0),
        ],
        grid=(tg * 256, 1, n),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, b, rows)],
        output_dtypes=[m.uint16 if xb else m.float32],
    )[0]
    return y.view(m.bfloat16) if xb else y


def gemv_mxfp4(
    slot: mx_.array, rows: int, k: int, x: mx_.array, tm: int | None = None, ggml: bool = False
) -> mx_.array:
    """y[b, rows] = W x for one MXFP4 matrix: `slot` its bytes (blocks then scales; ggml's 17-byte blocks when
    `ggml`) as a uint8 view or a pair, `rows`/`k` its shape, `x` [b, k] with b in 1..16. float32 accumulation,
    y in x's dtype, batch-invariant, the same bits in either layout; k % 32 == 0, the view 16-byte aligned."""
    return _mxfp4_call([mxfp4_split(slot, rows, k, ggml)], rows, k, x[None], tm, ggml)[0]


def gemv_mxfp4_group(
    slots: Sequence[mx_.array], rows: int, k: int, x: mx_.array, tm: int | None = None, ggml: bool = False
) -> mx_.array:
    """N MXFP4 matvecs of one shape in one dispatch, each bit-identical to its own `gemv_mxfp4`: `slots` N views,
    `x` [N, b, k] (broadcast one tile when shared). Returns [N, b, rows] in x's dtype, lazily."""
    return _mxfp4_call([mxfp4_split(s, rows, k, ggml) for s in slots], rows, k, x, tm, ggml)


def gemv_mxfp4_pair(
    gate: mx_.array, up: mx_.array, inter: int, k: int, x: mx_.array, tm: int | None = None
) -> mx_.array:
    """y[b, 2 * inter] = gate_up x for a GGUF's gate and up (two [inter, k] matrices in ggml's layout) in one
    dispatch, y interleaved as the checkpoint's gate_up (gate at the even positions): the bits of `gemv_mxfp4`
    over each half."""
    return _mxfp4_call([(gate.reshape(-1), up.reshape(-1))], 2 * int(inter), k, x[None], tm, True, True)[0]


def matmul_mxfp4_pair(gate: mx_.array, up: mx_.array, inter: int, k: int, x: mx_.array) -> mx_.array:
    """`matmul_mxfp4` for a GGUF's gate and up: the pair matvec up to 16 rows, a gemm over the interleaved
    dequantized rows past that"""
    m = mx()
    if int(x.shape[0]) <= GEMV_ROWS:
        return gemv_mxfp4_pair(gate, up, inter, k, x)
    dt = m.bfloat16 if x.dtype == m.bfloat16 else m.float32
    w = m.stack([mxfp4_dequant(gate, inter, k, dt, ggml=True), mxfp4_dequant(up, inter, k, dt, ggml=True)], axis=1)
    return m.matmul(x, w.reshape(2 * int(inter), int(k)).T)


def mxfp4_dequant(slot: mx_.array, rows: int, k: int, dtype: Any = None, ggml: bool = False) -> mx_.array:
    """One MXFP4 matrix as a dense `[rows, k]` array (bf16 by default): what prefill-sized calls multiply
    through MLX's gemm, and the reference the matvec is checked against."""
    global _mxfp4_dequant_kernel
    m = mx()
    dtype = dtype or m.bfloat16
    rows, k = int(rows), int(k)
    with _gemv_lock:
        if _mxfp4_dequant_kernel is None:
            _mxfp4_dequant_kernel = m.fast.metal_kernel(
                name="btb_mxfp4_dequant",
                input_names=["w", "s"],
                output_names=["out"],
                header=_MXFP4_HEADER,
                source=_MXFP4_DEQUANT_SRC,
            )
    n = rows * (k // MXFP4_BLOCK)
    return _mxfp4_dequant_kernel(
        inputs=list(mxfp4_split(slot, rows, k, ggml)),
        template=[("ROWS", rows), ("KDIM", k), ("OT", dtype), ("GGML", 1 if ggml else 0)],
        grid=(((n + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, k)],
        output_dtypes=[dtype],
    )[0]


def matmul_mxfp4(slot: mx_.array, rows: int, k: int, x: mx_.array, ggml: bool = False) -> mx_.array:
    """y = W x for any number of rows: the batch-invariant matvec up to 16 rows, MLX's gemm over a
    dequantized copy past that (prefill, where the engine does not verify a tree)."""
    m = mx()
    if int(x.shape[0]) <= GEMV_ROWS:
        return gemv_mxfp4(slot, rows, k, x, ggml=ggml)
    w = mxfp4_dequant(slot, rows, k, m.bfloat16 if x.dtype == m.bfloat16 else m.float32, ggml=ggml)
    return m.matmul(x, w.T)


def experts_mxfp4_step(
    x: mx_.array,
    gate_up: Sequence[mx_.array],
    down: Sequence[mx_.array],
    weights: mx_.array,
    hidden: int,
    inter: int,
    gate_up_bias: Sequence[mx_.array] | None = None,
    down_bias: Sequence[mx_.array] | None = None,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> mx_.array:
    """One token's k experts as gpt-oss computes them: `x` [b, hidden] (b in 1..16), `gate_up`/`down` lists of k
    MXFP4 views in expert order, `weights` the router weights (k, or [k, b]), the optional bias lists. Returns
    [b, hidden] in x's dtype: down_e(glu(gate_up_e x + b_e)) + b_e with `GptOssExperts._apply_gate`'s gate,
    weighted and summed in the given order. Two grouped dispatches and elementwise glue: batch-invariant."""
    m = mx()
    k = len(gate_up)
    if len(down) != k:
        raise ValueError(f"[mlx] {k} gate_up matrices and {len(down)} down matrices")
    b = int(x.shape[0])
    hidden, inter = int(hidden), int(inter)
    gu = gemv_mxfp4_group(gate_up, 2 * inter, hidden, m.broadcast_to(x[None], (k, b, hidden)))
    if gate_up_bias is not None:
        gu = gu + m.stack([a.astype(gu.dtype) for a in gate_up_bias])[:, None, :]
    gate = m.minimum(gu[..., 0::2], limit)
    up = m.clip(gu[..., 1::2], -limit, limit)
    h = ((up + 1) * (gate * m.sigmoid(alpha * gate))).astype(x.dtype)
    out = gemv_mxfp4_group(down, hidden, inter, h)
    if down_bias is not None:
        out = out + m.stack([a.astype(out.dtype) for a in down_bias])[:, None, :]
    w = weights if isinstance(weights, m.array) else m.array(np.asarray(weights, dtype=np.float32))
    w = w.astype(out.dtype).reshape(k, -1, 1)
    total = out[0] * w[0]
    for e in range(1, k):
        total = total + out[e] * w[e]
    return total
