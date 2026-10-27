# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The stored GGUF types' matmul on Metal, one chain at every width. A type's `w8` gives 8 consecutive weights of a
row as float32 from its streams (rounded to bf16 for a bf16 x), and every kernel over the type is built on it. The
chain: a row's 32-column blocks in 8 segments (block b in segment b % 8), each block in fragment order (logical 8*ks +
f is physical 8*(f/2) + 2*ks + f%2), each segment an fma chain from +0 over its blocks ascending, the segments' sums
folded in order, ((0 + s0) + s1) + ... + s7, every add an fma by 1 so no compile reassociates it. One pass row runs
it with a segment a lane (`LANE`), a few with a segment a lane over two rows (`LANE2`, or `PAIR2` for a type that
decodes a pair of weights), more on the matrix multiply-accumulates with a segment an accumulator (`TILE`, or two
for 4 float32 rows: `TILE4`), a prefill on steel over the dequant (`steel.gemm_chain`). So a row has the same bits in a step, a verify pass, a batch, a
prefill cut anywhere, the megakernel and a mixture's gather."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from .core import metal_kernel, mx

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel


@dataclass(frozen=True)
class Table:
    """a codebook a stored type's kernels read beside its streams: the name its source reads it by, its element
    type, and its array"""

    name: str
    ctype: str
    make: Callable[[], mx_.array]


@dataclass(frozen=True)
class StoredKernel:
    """a stored type as every kernel over it is built: `w8`, the body of a device function writing the float32
    weights of row `row`, columns k..k+7 (k a multiple of 8, under `cols`, the row's width) into `w` from the type's
    `streams` and `tables` (each read by its name); `header` the device functions it calls, `defs` the defines it
    reads, `bw` the weights a block (a row is whole blocks), `lane_rows` the float32 x's pass rows to which a launch
    takes `LANE2` (past them `TILE4` to 4 rows where `tile4`, then `TILE`) and `bf16_rows` a bf16 x's (past them
    `TILE`), `w2` (where a type has one) the body writing weights k and k + 1 (k even) of row `row` with `w8`'s bits,
    which takes `PAIR2` for a float32 x's `LANE2`. `streams` in `kinds.stored_stream_bytes`' order"""

    w8: str
    bw: int
    header: str = ""
    defs: str = ""
    streams: tuple[tuple[str, str], ...] = (("W", "uint8_t"),)
    tables: tuple[Table, ...] = ()
    lane_rows: int = 3
    bf16_rows: int = 3
    w2: str = ""
    tile4: bool = True


def inputs(k: StoredKernel) -> list[tuple[str, str]]:
    """what a type's kernels read, each by its name and element type: its streams, then its tables"""
    return [*k.streams, *((t.name, t.ctype) for t in k.tables)]


def device_fns(k: StoredKernel) -> str:
    """the type's device functions and `w8` (and `w2`) over its inputs; a kernel calls it as W8(row, k, w) (W2), its
    COLS the width"""
    params = "".join(f"const device {t}* {n}, " for n, t in inputs(k))
    args = "".join(f"{n}, " for n, _ in inputs(k))
    fns = f"{k.header}inline void w8({params}uint cols, uint row, uint k, thread float* w) {{\n{k.w8}}}\n"
    fns += f"#define W8(row, k, w) w8({args}COLS, row, k, w)\n"
    if k.w2:
        fns += f"inline void w2({params}uint cols, uint row, uint k, thread float* w) {{\n{k.w2}}}\n"
        fns += f"#define W2(row, k, w) w2({args}COLS, row, k, w)\n"
    return fns


# y[b, n] = sum_k x[b, k] w[n, k] for NB rows of x, a row of W a group of 8 lanes (4 rows a simdgroup): lane p runs
# segment p (blocks p, p + 8, ..., each in the chain's order: logical 8 * ks + f is physical 8 * (f / 2) + 2 * ks +
# f % 2), its block's 32 x values loaded before the decode; then the 8 lanes' sums folded in order through shuffles.
# The one-row launches' and a gather's (an item one row)
LANE = r"""
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint r0 = (threadgroup_position_in_grid.x * SGN + sg) * 4;
    if (r0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
#else
    typedef float ST;
#endif
    uint p = lane % 8, rl = lane / 8;
    uint row = min(r0 + rl, (uint)ROWS - 1);
    float acc[NB];
    #pragma unroll
    for (uint t = 0; t < NB; ++t) acc[t] = 0.0f;
    for (uint b = p; b < COLS / 32; b += 8) {
        // the block's x widened ahead of the decode (packed and widened at each use measured 1.6-2.6x)
        float xf[NB][32];
        #pragma unroll
        for (uint t = 0; t < NB; ++t) {
#if XB
            const device uint4* xr = (const device uint4*)(XROW(t) + b * 32);
            #pragma unroll
            for (uint q = 0; q < 4; ++q) {
                uint4 v = xr[q];
                uint u[4] = {v.x, v.y, v.z, v.w};
                #pragma unroll
                for (uint i = 0; i < 8; ++i) xf[t][8 * q + i] = as_type<float>(i % 2 ? u[i / 2] & 0xFFFF0000u : u[i / 2] << 16);
            }
#else
            const device float4* xr = (const device float4*)(XROW(t) + b * 32);
            #pragma unroll
            for (uint q = 0; q < 8; ++q) {
                float4 v = xr[q];
                xf[t][4 * q] = v.x;
                xf[t][4 * q + 1] = v.y;
                xf[t][4 * q + 2] = v.z;
                xf[t][4 * q + 3] = v.w;
            }
#endif
        }
        float w[32];
        #pragma unroll
        for (uint q = 0; q < 4; ++q) {
            W8(row, b * 32 + 8 * q, w + 8 * q);
            #pragma unroll
            for (uint i = 0; i < 8; ++i) w[8 * q + i] = (float)(ST)w[8 * q + i];
        }
        #pragma unroll
        for (uint ks = 0; ks < 4; ++ks)
            #pragma unroll
            for (uint f = 0; f < 8; ++f) {
                uint c = 8 * (f / 2) + 2 * ks + f % 2;
                #pragma unroll
                for (uint t = 0; t < NB; ++t) acc[t] = fma(w[c], xf[t][c], acc[t]);
            }
    }
    #pragma unroll
    for (uint t = 0; t < NB; ++t) {
        float v = 0.0f;
        for (uint i = 0; i < 8; ++i) v = fma(simd_shuffle(acc[t], (lane & ~7u) + i), 1.0f, v);
        if (p == 0 && r0 + rl < ROWS) ((device ST*)YROW(t))[r0 + rl] = (ST)v;
    }
"""

# LANE for a few pass rows: a lane two rows of W (8 a simdgroup), both rows' blocks decoded, then each pass row's x
# pairs loaded once for both rows. The x a lane reads is what a pass row costs (LANE's whole block of x a pass row
# measured +20-40% at 4 rows); two rows a lane halve it
LANE2 = r"""
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint r0 = (threadgroup_position_in_grid.x * SGN + sg) * 8;
    if (r0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
#else
    typedef float ST;
#endif
    uint p = lane % 8, rl = lane / 8;
    uint ra = min(r0 + 2 * rl, (uint)ROWS - 1), rb = min(r0 + 2 * rl + 1, (uint)ROWS - 1);
    float acc[2][NB];
    #pragma unroll
    for (uint t = 0; t < NB; ++t) acc[0][t] = acc[1][t] = 0.0f;
    for (uint b = p; b < COLS / 32; b += 8) {
        float w[2][32];
        #pragma unroll
        for (uint q = 0; q < 4; ++q) {
            W8(ra, b * 32 + 8 * q, w[0] + 8 * q);
            W8(rb, b * 32 + 8 * q, w[1] + 8 * q);
            #pragma unroll
            for (uint i = 0; i < 8; ++i) {
                w[0][8 * q + i] = (float)(ST)w[0][8 * q + i];
                w[1][8 * q + i] = (float)(ST)w[1][8 * q + i];
            }
        }
        #pragma unroll
        for (uint t = 0; t < NB; ++t) {
#if XB
            const device uint* xr = (const device uint*)(XROW(t) + b * 32);
#else
            const device float2* xr = (const device float2*)(XROW(t) + b * 32);
#endif
            #pragma unroll
            for (uint ks = 0; ks < 4; ++ks)
                #pragma unroll
                for (uint g = 0; g < 4; ++g) {
                    // the pair at physical 8 * g + 2 * ks: logical 8 * ks + 2 * g and its successor
#if XB
                    uint u = xr[4 * g + ks];
                    float x0 = as_type<float>(u << 16), x1 = as_type<float>(u & 0xFFFF0000u);
#else
                    float2 v = xr[4 * g + ks];
                    float x0 = v.x, x1 = v.y;
#endif
                    #pragma unroll
                    for (uint r = 0; r < 2; ++r) {
                        acc[r][t] = fma(w[r][8 * g + 2 * ks], x0, acc[r][t]);
                        acc[r][t] = fma(w[r][8 * g + 2 * ks + 1], x1, acc[r][t]);
                    }
                }
        }
    }
    #pragma unroll
    for (uint r = 0; r < 2; ++r)
        #pragma unroll
        for (uint t = 0; t < NB; ++t) {
            float v = 0.0f;
            for (uint i = 0; i < 8; ++i) v = fma(simd_shuffle(acc[r][t], (lane & ~7u) + i), 1.0f, v);
            if (p == 0 && r0 + 2 * rl + r < ROWS) ((device ST*)YROW(t))[r0 + 2 * rl + r] = (ST)v;
        }
"""

# LANE2 for a type that decodes a pair (`StoredKernel.w2`): each pair of both rows decoded where it is multiplied,
# the pass rows innermost. Holding no block of decoded weights, a lane keeps few registers (a Q8_0 head's float32 x at
# 2-5 pass rows within 2% of the lane-strided kernel, against +4-29% on LANE2 or TILE)
PAIR2 = r"""
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint r0 = (threadgroup_position_in_grid.x * SGN + sg) * 8;
    if (r0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
#else
    typedef float ST;
#endif
    uint p = lane % 8, rl = lane / 8;
    uint ra = min(r0 + 2 * rl, (uint)ROWS - 1), rb = min(r0 + 2 * rl + 1, (uint)ROWS - 1);
    float acc[2][NB];
    #pragma unroll
    for (uint t = 0; t < NB; ++t) acc[0][t] = acc[1][t] = 0.0f;
    for (uint b = p; b < COLS / 32; b += 8) {
        #pragma unroll
        for (uint ks = 0; ks < 4; ++ks)
            #pragma unroll
            for (uint g = 0; g < 4; ++g) {
                // the pair at physical 8 * g + 2 * ks: logical 8 * ks + 2 * g and its successor
                float w[2][2];
                W2(ra, b * 32 + 8 * g + 2 * ks, w[0]);
                W2(rb, b * 32 + 8 * g + 2 * ks, w[1]);
                #pragma unroll
                for (uint r = 0; r < 2; ++r) {
                    w[r][0] = (float)(ST)w[r][0];
                    w[r][1] = (float)(ST)w[r][1];
                }
                #pragma unroll
                for (uint t = 0; t < NB; ++t) {
#if XB
                    uint u = ((const device uint*)(XROW(t) + b * 32))[4 * g + ks];
                    float x0 = as_type<float>(u << 16), x1 = as_type<float>(u & 0xFFFF0000u);
#else
                    float2 v = ((const device float2*)(XROW(t) + b * 32))[4 * g + ks];
                    float x0 = v.x, x1 = v.y;
#endif
                    #pragma unroll
                    for (uint r = 0; r < 2; ++r) {
                        acc[r][t] = fma(w[r][0], x0, acc[r][t]);
                        acc[r][t] = fma(w[r][1], x1, acc[r][t]);
                    }
                }
            }
    }
    #pragma unroll
    for (uint r = 0; r < 2; ++r)
        #pragma unroll
        for (uint t = 0; t < NB; ++t) {
            float v = 0.0f;
            for (uint i = 0; i < 8; ++i) v = fma(simd_shuffle(acc[r][t], (lane & ~7u) + i), 1.0f, v);
            if (p == 0 && r0 + 2 * rl + r < ROWS) ((device ST*)YROW(t))[r0 + 2 * rl + r] = (ST)v;
        }
"""

# the same for more pass rows on the matrix multiply-accumulates: a simdgroup owns 8 rows of W and walks the
# blocks in order, block b's 4 multiplies into segment b % 8's accumulator, an 8-row block of x each; lane (fm, fn)
# holds A's [fm][fn, fn+1] (row fm's physical 8 * (fn / 2)..+7 of the block, one w8) and B's [fm][fn, fn+1] (x rows fn,
# fn + 1 at physical 8 * (fm / 2) + 2 * ks + fm % 2, one or two 16-byte loads). The segments fold in order at the end
TILE = r"""
    constexpr uint NA = (NB + 7) / 8;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * SGN + sg) * 8;
    if (n0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
    typedef simdgroup_bfloat8x8 SM;
    constexpr uint XV = 1;
#else
    typedef float ST;
    typedef simdgroup_float8x8 SM;
    constexpr uint XV = 2;
#endif
    uint qid = lane / 4;
    uint fm = (qid & 4) + (lane / 2) % 4, fn = (qid & 2) * 2 + (lane % 2) * 2;
    uint row = min(n0 + fm, (uint)ROWS - 1), wq = 8 * (fn / 2), xe = fm % 2;
    // x rows fn, fn + 1 of each 8-row block at physical 8 * (fm / 2) of a block, XS 16-byte loads a block apart; a row
    // past NB reads the last one's (its products land in columns never stored). A float32 x skips those loads
    // (measured faster at 3-6 rows); bf16 keeps them (skipping measured +10-20%)
    constexpr uint XS = 32 / (16 / sizeof(ST));
    bool ok0[NA], ok1[NA];
    const device uint4* x0[NA];
    const device uint4* x1[NA];
    #pragma unroll
    for (uint j = 0; j < NA; ++j) {
        uint t0 = 8 * j + fn;
        ok0[j] = XB || t0 < NB;
        ok1[j] = XB || t0 + 1 < NB;
        x0[j] = (const device uint4*)(XROW(min(t0, (uint)NB - 1)) + 8 * (fm / 2));
        x1[j] = (const device uint4*)(XROW(min(t0 + 1, (uint)NB - 1)) + 8 * (fm / 2));
    }
    simdgroup_float8x8 acc[8 * NA];
    #pragma unroll
    for (uint i = 0; i < 8 * NA; ++i) acc[i] = simdgroup_float8x8(0.0f);
    for (uint b0 = 0; b0 < COLS / 32; b0 += 8) {
        #pragma unroll
        for (uint s = 0; s < 8; ++s) {
            uint b = b0 + s;
            if (b < COLS / 32) {
                float w[8];
                W8(row, b * 32 + wq, w);
                #pragma unroll
                for (uint j = 0; j < NA; ++j) {
                    uint4 v0[XV], v1[XV];
                    #pragma unroll
                    for (uint v = 0; v < XV; ++v) {
                        v0[v] = ok0[j] ? x0[j][b * XS + v] : uint4(0u);
                        v1[v] = ok1[j] ? x1[j][b * XS + v] : uint4(0u);
                    }
                    #pragma unroll
                    for (uint ks = 0; ks < 4; ++ks) {
                        SM A, B;
                        A.thread_elements()[0] = (ST)w[2 * ks];
                        A.thread_elements()[1] = (ST)w[2 * ks + 1];
#if XB
                        uint u0 = v0[0][ks], u1 = v1[0][ks];
                        B.thread_elements()[0] = as_type<bfloat>((ushort)(xe ? u0 >> 16 : u0 & 0xFFFF));
                        B.thread_elements()[1] = as_type<bfloat>((ushort)(xe ? u1 >> 16 : u1 & 0xFFFF));
#else
                        uint4 q0 = v0[ks / 2], q1 = v1[ks / 2];
                        B.thread_elements()[0] = as_type<float>(ks % 2 ? (xe ? q0.w : q0.z) : (xe ? q0.y : q0.x));
                        B.thread_elements()[1] = as_type<float>(ks % 2 ? (xe ? q1.w : q1.z) : (xe ? q1.y : q1.x));
#endif
                        simdgroup_multiply_accumulate(acc[s * NA + j], A, B, acc[s * NA + j]);
                    }
                }
            }
        }
    }
    // every element folded before any branch: read under a row's guard, the accumulators go to the stack (2-3x)
    float2 v[NA];
    #pragma unroll
    for (uint j = 0; j < NA; ++j) {
        v[j] = float2(0.0f);
        #pragma unroll
        for (uint s = 0; s < 8; ++s) {
            v[j].x = fma(acc[s * NA + j].thread_elements()[0], 1.0f, v[j].x);
            v[j].y = fma(acc[s * NA + j].thread_elements()[1], 1.0f, v[j].y);
        }
    }
    if (n0 + fm < ROWS) {
        #pragma unroll
        for (uint j = 0; j < NA; ++j) {
            #pragma unroll
            for (uint e = 0; e < 2; ++e) {
                uint t = 8 * j + fn + e;
                if (t < NB) ((device ST*)YROW(t))[n0 + fm] = (ST)v[j][e];
            }
        }
    }
"""

# TILE for 4 pass rows at most, 4 rows of W a simdgroup: A's rows 4..7 and B's columns 4..7 are block b + 1's, so a
# multiply puts block b's products in C's top-left quadrant and block b + 1's in its bottom-right, each as TILE has
# them, and one accumulator holds segments s and s + 1 (s even). Twice TILE's simdgroups, no x row loaded twice (a
# float32 head at 4 rows ran 8-10 points faster than on TILE)
TILE4 = r"""
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint n0 = (threadgroup_position_in_grid.x * SGN + sg) * 4;
    if (n0 >= ROWS) return;
BIND
#if XB
    typedef bfloat ST;
    typedef simdgroup_bfloat8x8 SM;
    constexpr uint XV = 1;
#else
    typedef float ST;
    typedef simdgroup_float8x8 SM;
    constexpr uint XV = 2;
#endif
    constexpr uint NBK = COLS / 32;
    constexpr bool ODD = NBK % 2;
    uint qid = lane / 4;
    uint fm = (qid & 4) + (lane / 2) % 4, fn = (qid & 2) * 2 + (lane % 2) * 2;
    // A: row fm % 4 of W in block b + fm / 4, physical 8 * (fn / 2)..+7. B: x rows fn % 4, + 1 of block b + fn / 4
    uint row = min(n0 + fm % 4, (uint)ROWS - 1), wq = 8 * (fn / 2), xe = fm % 2, ba = fm / 4, bb = fn / 4;
    constexpr uint XS = 32 / (16 / sizeof(ST));
    uint t0 = fn % 4;
    bool ok0 = XB || t0 < NB, ok1 = XB || t0 + 1 < NB;
    const device uint4* x0 = (const device uint4*)(XROW(min(t0, (uint)NB - 1)) + 8 * (fm / 2));
    const device uint4* x1 = (const device uint4*)(XROW(min(t0 + 1, (uint)NB - 1)) + 8 * (fm / 2));
    simdgroup_float8x8 acc[4];
    #pragma unroll
    for (uint i = 0; i < 4; ++i) acc[i] = simdgroup_float8x8(0.0f);
    for (uint b0 = 0; b0 < NBK; b0 += 8) {
        #pragma unroll
        for (uint s = 0; s < 8; s += 2) {
            uint b = b0 + s;
            if (b < NBK) {
                // an odd count's last block has no b + 1: zeros there add +0 to segment s + 1
                bool wok = !ODD || b + ba < NBK, xok = !ODD || b + bb < NBK;
                float w[8];
                if (wok) {
                    W8(row, (b + ba) * 32 + wq, w);
                } else {
                    #pragma unroll
                    for (uint i = 0; i < 8; ++i) w[i] = 0.0f;
                }
                uint4 v0[XV], v1[XV];
                #pragma unroll
                for (uint v = 0; v < XV; ++v) {
                    v0[v] = ok0 && xok ? x0[(b + bb) * XS + v] : uint4(0u);
                    v1[v] = ok1 && xok ? x1[(b + bb) * XS + v] : uint4(0u);
                }
                #pragma unroll
                for (uint ks = 0; ks < 4; ++ks) {
                    SM A, B;
                    A.thread_elements()[0] = (ST)w[2 * ks];
                    A.thread_elements()[1] = (ST)w[2 * ks + 1];
#if XB
                    uint u0 = v0[0][ks], u1 = v1[0][ks];
                    B.thread_elements()[0] = as_type<bfloat>((ushort)(xe ? u0 >> 16 : u0 & 0xFFFF));
                    B.thread_elements()[1] = as_type<bfloat>((ushort)(xe ? u1 >> 16 : u1 & 0xFFFF));
#else
                    uint4 q0 = v0[ks / 2], q1 = v1[ks / 2];
                    B.thread_elements()[0] = as_type<float>(ks % 2 ? (xe ? q0.w : q0.z) : (xe ? q0.y : q0.x));
                    B.thread_elements()[1] = as_type<float>(ks % 2 ? (xe ? q1.w : q1.z) : (xe ? q1.y : q1.x));
#endif
                    simdgroup_multiply_accumulate(acc[s / 2], A, B, acc[s / 2]);
                }
            }
        }
    }
    // lane L < 8 holds C[L / 2][(L % 2) * 2 + e] (segment 2j), lane L + 24 the same row and pass row's bottom-right
    // element (segment 2j + 1): every lane shuffles, lanes 0..7 fold and store
    float2 v = float2(0.0f);
    #pragma unroll
    for (uint j = 0; j < 4; ++j) {
        float e0 = acc[j].thread_elements()[0], e1 = acc[j].thread_elements()[1];
        float o0 = simd_shuffle_down(e0, 24u), o1 = simd_shuffle_down(e1, 24u);
        v.x = fma(e0, 1.0f, v.x);
        v.x = fma(o0, 1.0f, v.x);
        v.y = fma(e1, 1.0f, v.y);
        v.y = fma(o1, 1.0f, v.y);
    }
    if (lane < 8 && n0 + fm < ROWS) {
        #pragma unroll
        for (uint e = 0; e < 2; ++e) {
            uint t = fn + e;
            if (t < NB) ((device ST*)YROW(t))[n0 + fm] = (ST)v[e];
        }
    }
"""
# a launch's simdgroups a threadgroup: no kernel holds threadgroup memory; 2 and 4 tie, 8 and 16 run slower
SGN = 4
# a launch's pass rows at most: TILE's 8 segment accumulators an 8-row block of x
LAUNCH_ROWS = 16


def body(nb: int, k: StoredKernel, xb: bool) -> str:
    """the kernel body a launch of `nb` pass rows of x (bf16 if `xb`, else float32) over type `k` takes"""
    if nb == 1:
        return LANE
    # a bf16 x is a layer's, a float32 x the head's (150k rows): their crossovers measured apart. TILE4 at 4 float32
    # rows ran a tie to 25 points faster than TILE over the types reaching it, 5-40 slower for bf16 at 2-4 rows
    if xb:
        return LANE2 if nb <= k.bf16_rows else TILE
    if nb <= k.lane_rows:
        return PAIR2 if k.w2 else LANE2
    return TILE4 if nb <= 4 and k.tile4 else TILE


def tile_defines(rows: int, cols: int, nb: int, xb: bool) -> str:
    """a launch's defines but its rows' addresses: the matrix, the pass rows, its simdgroups a threadgroup"""
    return f"#define ROWS {rows}\n#define COLS {cols}\n#define NB {nb}\n#define SGN {SGN}\n#define XB {int(xb)}\n"


def groups(rows: int, nb: int, k: StoredKernel, xb: bool) -> int:
    """the threadgroups the launch `body(nb, k, xb)` over `rows` outputs takes, `SGN` simdgroups each: 4 rows a
    simdgroup on LANE and TILE4, 8 on LANE2, PAIR2 and TILE"""
    return -(-rows // ((4 if body(nb, k, xb) in (LANE, TILE4) else 8) * SGN))


# the matrix's weights as the kernels round them, OT bf16 or float32, a thread 8 consecutive weights
DEQUANT = r"""
    uint t = thread_position_in_grid.x;
    if (t >= ROWS * (COLS / 8)) return;
    uint row = t / (COLS / 8), k = (t % (COLS / 8)) * 8;
    float w[8];
    W8(row, k, w);
    #pragma unroll
    for (uint i = 0; i < 8; ++i) out[(size_t)row * COLS + k + i] = (OT)w[i];
"""

# row `row` of the matrix in bf16 into dst, a thread 8 weights of each 2048 (a megakernel's embedding: the dequant's)
DECODE_ROW = r"""
    for (uint k = tid * 8; k < COLS; k += 2048) {
        float w[8];
        W8(row, k, w);
        for (uint i = 0; i < 8; ++i) dst[k + i] = (bfloat)w[i];
    }
"""

_kernels: dict[tuple[object, ...], MetalKernel] = {}
_lock = threading.Lock()


def _whole(k: StoredKernel, cols: int) -> None:
    if cols % k.bw:
        raise ValueError(f"[mlx] a stored row of {cols} columns is not whole blocks of {k.bw}")


class Source(NamedTuple):
    """a kernel as `metal_kernel` builds it: its name, its inputs' names, its header and its body (one output, out)"""

    name: str
    inputs: list[str]
    header: str
    body: str


def compiled(key: tuple[object, ...], make: Callable[[], Source]) -> MetalKernel:
    """the kernel cached under `key`, built from `make()` on its first call: a launch is a lookup, its source (several
    KB, ~5 us to assemble) made once"""
    k = _kernels.get(key)
    if k is None:
        with _lock:
            k = _kernels.get(key)
            if k is None:
                s = make()
                k = _kernels[key] = metal_kernel(
                    name=s.name, input_names=s.inputs, output_names=["out"], header=s.header, source=s.body
                )
    return k


def _launched(name: str, k: StoredKernel, rows: int, cols: int, nb: int, xb: bool) -> Source:
    _whole(k, cols)
    return Source(
        f"{name}_{nb}_x{int(xb)}",
        [*(nm for nm, _ in inputs(k)), "x"],
        "#include <metal_simdgroup_matrix>\n" + k.defs + device_fns(k),
        tile_defines(rows, cols, nb, xb)
        + "#define XROW(t) (x + (size_t)(t) * COLS)\n#define YROW(t) (out + (size_t)(t) * ROWS)\n"
        + body(nb, k, xb).replace("BIND", ""),
    )


def launch(name: str, k: StoredKernel, ins: Sequence[mx_.array], x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T over a stored type's `ins` (its streams then its tables, `inputs`' order), x bf16
    or float32, y in its dtype: launches of about equal rows, `LAUNCH_ROWS` at most, each on the kernel its rows take
    (`body`). `name` is the caller's kernel name for the matrix (its rows and columns), the pass rows appended"""
    m = mx
    xb = x.dtype == m.bfloat16
    xin = x if xb else x.astype(m.float32)
    b = int(xin.shape[0])
    n = -(-b // LAUNCH_ROWS)
    size = -(-b // n)
    outs = []
    for s in range(0, b, size):
        xr = xin if n == 1 else xin[s : s + size]
        nb = int(xr.shape[0])
        kern = compiled((name, nb, xb), functools.partial(_launched, name, k, rows, cols, nb, xb))
        got = kern(inputs=[*ins, xr], grid=(groups(rows, nb, k, xb) * 32 * SGN, 1, 1), threadgroup=(32 * SGN, 1, 1),
                   output_shapes=[(nb, rows)], output_dtypes=[xr.dtype])  # fmt: skip
        outs.append(got[0])
    y = outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)
    return y if y.dtype == x.dtype else y.astype(x.dtype)


def _dequant(name: str, k: StoredKernel, rows: int, cols: int, ot: str) -> Source:
    _whole(k, cols)
    return Source(
        f"{name}_{rows}_{cols}_{ot}",
        [nm for nm, _ in inputs(k)],
        k.defs + device_fns(k),
        f"#define ROWS {rows}\n#define COLS {cols}\n#define OT {ot}\n" + DEQUANT,
    )


def dequant(name: str, k: StoredKernel, ins: Sequence[mx_.array], rows: int, cols: int, dtype: mx_.Dtype) -> mx_.array:
    """the whole [rows, cols] weight in `dtype` (bf16, or float32 for a float32 x's gemm), as the kernels round it: a
    prefill's (`steel.gemm_chain`)"""
    m = mx
    ot = "bfloat" if dtype == m.bfloat16 else "float"
    kern = compiled((name, rows, cols, ot), lambda: _dequant(name, k, rows, cols, ot))
    n = rows * cols // 8
    return kern(
        inputs=list(ins), grid=(-(-n // 256) * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(rows, cols)],
        output_dtypes=[dtype],
    )[0]  # fmt: skip
