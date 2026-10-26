# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The pass's Metal: its ops as device functions, each wrapped as a kernel of its own that reads its instruction's row
of the table, so a program's instructions are dispatches the GPU schedules itself (`replay`), compiled behind MLX's own
preamble with MLX's math, the bits the step's. A program compiles in the ops it uses."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import NamedTuple

from ...kinds import Quant, quant_key
from .. import aluminum, fused, matvec
from ..core import mx
from ..gemv import GEMM_SRC, gemm_sgn
from ..matvec import StoredKernel
from ..stored import STORED

# a table row: op, units (the op's work items, a threadgroup each), REFS (buffer, offset) references from column
# REF0, n0, n1, n2, f0's bits, and from column DEP0 the instructions it reads the output of (index + 1, 0 for none)
REFS = 6
REF0 = 2
NUM = REF0 + 2 * REFS  # n0's column
DEP0 = NUM + 4
DEPS = 4  # a DeltaNet's conv reads its four input projections' pieces where they are not one weight
W = DEP0 + DEPS
# dims: rows of the pass, the path's row stride, the K/V regions' rows a head, past, then unused to 8
DIMS = 8

TYPES = r"""
#include <metal_simdgroup_matrix>
struct Ins {
    const device uint8_t* A;
    const device uint8_t* B;
    device uint8_t* C;
    device uint8_t* D;
    const device uint8_t* E;
    const device uint8_t* F;
    uint items, n0, n1, n2;
    float f0;
};
struct Ctx {
    uint tid, sg, lane;
    threadgroup uint8_t* tgm;
    const device uint* dims;
    const device float* freqs;
    const device uint* ids;
    const device uint* pos;
    const device uint* meta;
    const device uint* path;
    const device uint* rmap;
    device uint* out;
};
"""

# the step's own MLX ops where a family runs off the fused kernels, rounded where MLX's kernels round
MLX_OPS = r"""
// a row's inverse RMS as MLX's rms_norm takes it (rms_norm.metal, RMS_N_READS 4, RMS_LOOPED_LIMIT 4096): its
// threadgroup of 32 * ceil(H / 128) threads of 4 consecutive elements each (1024 threads past the limit, each then
// striding the row by 4096), a thread's squares summed in its order, simd_sum within each of its simdgroups, simd_sum
// across them, precise rsqrt. Ours walk its simdgroups in turn, each lane standing for the thread at its own lane, so
// every simd_sum takes MLX's values in MLX's lanes
template <typename X>
inline float mlx_rms_inv(const device X* xr, uint H, float eps, thread const Ctx& c) {
    threadgroup float* part = (threadgroup float*)c.tgm;
    if (c.tid < 32) part[c.tid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool looped = H > 4096;
    uint vt = looped ? 1024 : 32 * ((H + 127) / 128);
    for (uint v = c.sg; v < vt / 32; v += 8) {
        uint lid = v * 32 + c.lane;
        float acc = 0.0f;
        if (!looped) {
            float xv[4];
            for (uint i = 0; i < 4; ++i) xv[i] = lid * 4 + i < H ? (float)xr[lid * 4 + i] : 0.0f;
            for (uint i = 0; i < 4; ++i) acc += xv[i] * xv[i];
        } else {
            for (uint r = 0; r < H; r += vt * 4) {
                for (uint i = 0; i < 4; ++i) {
                    uint j = r + lid * 4 + i;
                    if (j < H) { float xv = (float)xr[j]; acc += xv * xv; }
                }
            }
        }
        acc = simd_sum(acc);
        if (c.lane == 0) part[v] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c.sg == 0) {
        float acc = simd_sum(part[c.lane]);
        if (c.lane == 0) part[32] = metal::precise::rsqrt(acc / H + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = part[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return inv;
}

// a bf16 row through MLX's rms_norm as `_mlx_norm` runs it: x * inv cast to the weight's type, times the weight (a
// bf16 weight's product rounded in bfloat; a float32 one's, the norm taken on the row cast to float32, rounded once
// back to bf16)
template <typename W>
inline void mlx_rms_norm_row(const device bfloat* xr, device bfloat* yr, const device W* w, uint H, float eps,
                             thread const Ctx& c) {
    float inv = mlx_rms_inv(xr, H, eps, c);
    for (uint i = c.tid; i < H; i += 256) yr[i] = (bfloat)(w[i] * static_cast<W>((float)xr[i] * inv));
}

// MLX's sigmoid (unary_ops.h `Sigmoid`: 1 / (1 + exp|x|), mirrored for x >= 0) as its library computes it on bf16:
// each step rounded to bf16, exp the precise one (MLX's kernels are built without fast math; a custom kernel's
// metal::exp is the fast one, a bf16 ulp off at x = -6.84375)
template <typename T>
inline T mlx_sigmoid(T x) {
    T e = static_cast<T>(metal::precise::exp(static_cast<float>(metal::abs(x))));
    T y = static_cast<T>(1.0f / static_cast<float>(static_cast<T>(1.0f + static_cast<float>(e))));
    return (x < 0) ? y : static_cast<T>(1.0f - static_cast<float>(y));
}

// gelu_pytorch_tanh as backend.act's MLX ops make it on bf16, `0.5 * a * (1.0 + tanh(0.7978845608028654 * (a +
// 0.044715 * a * a * a)))` left to right: each op a bf16 one, a Python float rounded to bf16 first, tanh the precise one
inline bfloat mlx_gelu_tanh(bfloat a) {
    bfloat cube = (((bfloat)0.044715f * a) * a) * a;
    bfloat t = (bfloat)metal::precise::tanh((float)((bfloat)0.7978845608028654f * (a + cube)));
    return ((bfloat)0.5f * a) * ((bfloat)1.0f + t);
}

// MLX's log1p (utils.h) and logaddexp (binary_ops.h `LogAddExp`) on float32, their log and exp the precise ones
inline float mlx_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) return metal::numeric_limits<float>::max();
    if (xp1 == 1.0f) return x;
    return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
}

inline float mlx_logaddexp(float x, float y) {
    if (metal::isnan(x) || metal::isnan(y)) return metal::numeric_limits<float>::quiet_NaN();
    constexpr float inf = metal::numeric_limits<float>::infinity();
    float maxval = metal::max(x, y), minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval : (maxval + mlx_log1p(metal::precise::exp(minval - maxval)));
}

// a float32 row of H <= 128 elements' inverse RMS as MLX's rms_norm takes it, by one simdgroup: MLX's threadgroup of
// 32 threads of 4 elements each, its second simd_sum over a single partial and zeros (`mlx_rms_inv`'s, one simdgroup)
inline float mlx_rms_inv_simd(const device float* xr, uint H, float eps, uint lane) {
    float acc = 0.0f;
    float xv[4];
    for (uint i = 0; i < 4; ++i) xv[i] = lane * 4 + i < H ? xr[lane * 4 + i] : 0.0f;
    for (uint i = 0; i < 4; ++i) acc += xv[i] * xv[i];
    acc = simd_sum(acc);
    return metal::precise::rsqrt(acc / H + eps);
}
"""


class OpName(StrEnum):
    """the pass's ops, each a device function `op_<name>(I, item, c)` (a `GRID3` op's `item` a uint3: its grid's
    position); its code in the table is its place here, from 1"""

    RMSNORM = "rmsnorm"
    MATVEC = "matvec"
    GATED = "gated"
    ADD_RMSNORM = "add_rmsnorm"
    EMBED = "embed"
    QKROPE = "qkrope"
    ATTN = "attn"
    ATTN_TREE = "attn_tree"
    ATTN_PHASED = "attn_phased"
    FOLD = "fold"
    ARGMAX = "argmax"
    FNORM = "fnorm"
    HEAD = "head"
    ARGMAX_PART = "argmax_part"
    SANDWICH_ADD = "sandwich_add"
    GATE = "gate"
    DELTA_CONV = "delta_conv"
    DELTA_RULE = "delta_rule"
    DELTA_NORM = "delta_norm"
    # a stored type's matvec and embedding row (`STORED`), named by its `kinds.quant_key`
    MV_Q2K = "mv_q2k"
    MV_Q3K = "mv_q3k"
    MV_Q4K = "mv_q4k"
    MV_Q5K = "mv_q5k"
    MV_Q6K = "mv_q6k"
    MV_Q40 = "mv_q40"
    MV_Q41 = "mv_q41"
    MV_Q80 = "mv_q80"
    MV_IQ4NL = "mv_iq4nl"
    MV_IQ4XS = "mv_iq4xs"
    MV_IQ3XXS = "mv_iq3xxs"
    MV_IQ3S = "mv_iq3s"
    MV_IQ2XXS = "mv_iq2xxs"
    MV_IQ2XS = "mv_iq2xs"
    MV_IQ2S = "mv_iq2s"
    MV_IQ1S = "mv_iq1s"
    MV_IQ1M = "mv_iq1m"
    EMBED_Q2K = "embed_q2k"
    EMBED_Q3K = "embed_q3k"
    EMBED_Q4K = "embed_q4k"
    EMBED_Q5K = "embed_q5k"
    EMBED_Q6K = "embed_q6k"
    EMBED_Q40 = "embed_q40"
    EMBED_Q41 = "embed_q41"
    EMBED_Q80 = "embed_q80"
    EMBED_IQ4NL = "embed_iq4nl"
    EMBED_IQ4XS = "embed_iq4xs"
    EMBED_IQ3XXS = "embed_iq3xxs"
    EMBED_IQ3S = "embed_iq3s"
    EMBED_IQ2XXS = "embed_iq2xxs"
    EMBED_IQ2XS = "embed_iq2xs"
    EMBED_IQ2S = "embed_iq2s"
    EMBED_IQ1S = "embed_iq1s"
    EMBED_IQ1M = "embed_iq1m"


CODE: dict[OpName, int] = {o: i + 1 for i, o in enumerate(OpName)}
SOURCE: dict[OpName, str] = {}
# the ops dispatched on the attention's own grids, (kv head and head group, segment, node) and the tree kernels' (the
# rows, kv head, segment): the threadgroups reading one head's segment sit together, measured 10% faster at 16 rows
# over 2000 than the same items on a line
GRID3: frozenset[OpName] = frozenset({OpName.ATTN, OpName.ATTN_TREE, OpName.ATTN_PHASED})

# each stored type's matvec and embedding row, through its launch's own chain (`stored.STORED`)
MV_OP: dict[Quant, OpName] = {q: OpName(f"mv_{quant_key(q)}") for q in STORED}
EMBED_OP: dict[Quant, OpName] = {q: OpName(f"embed_{quant_key(q)}") for q in STORED}
# what an op needs ahead of it in the kernel's header, once however many ops need it: a stored type's namespace
_PRELUDE: dict[OpName, tuple[str, str]] = {}


def _op(name: OpName, fn: str) -> None:
    SOURCE[name] = fn


_op(
    OpName.RMSNORM,
    r"""
inline void op_rmsnorm(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item` of x [T, H] (A) with weight w [H] (B, bf16, or float32 where n1 & 1: a centered norm's 1 + w) -> y
    // (C); H = n0, eps = f0
    uint H = I.n0;
    const device bfloat* xr = (const device bfloat*)I.A + (size_t)item * H;
    device bfloat* yr = (device bfloat*)I.C + (size_t)item * H;
    if (I.n1 & 1u) mlx_rms_norm_row(xr, yr, (const device float*)I.B, H, I.f0, c);
    else mlx_rms_norm_row(xr, yr, (const device bfloat*)I.B, H, I.f0, c);
}
""",
)

_op(
    OpName.MATVEC,
    r"""
inline void op_matvec(thread const Ins& I, uint item, thread const Ctx& c) {
    // y [T, rows] (C) = x [T, cols] (B) . W [rows, cols]^T (A); rows = n0, cols = n1, n2 & 1 a float32 y, n2 >> 1
    // y's row stride where y is a piece of a wider row (0: rows); the item's share of the 8-row tiles, a simdgroup's
    // each (the gemv tile kernel's body, NB a runtime value)
    const uint KT = MV_KT;
    const uint NB = c.dims[0];
    uint Nr = I.n0, K = I.n1, LD = (I.n2 >> 1) ? (I.n2 >> 1) : I.n0;
    uint ntile = (Nr + 7) / 8;
    uint t0 = (uint)(((ulong)item * ntile) / I.items), t1 = (uint)(((ulong)(item + 1) * ntile) / I.items);
    for (uint tile = t0 + c.sg; tile < t1; tile += 8) {
        uint nb0 = tile * 8;
        const device uint16_t* w = (const device uint16_t*)I.A;
        const device uint16_t* x = (const device uint16_t*)I.B;
        device uint16_t* y = (device uint16_t*)I.C;
        threadgroup bfloat* ws = (threadgroup bfloat*)c.tgm + c.sg * (8 * KT);
        threadgroup bfloat* xs = (threadgroup bfloat*)c.tgm + 8 * (8 * KT) + c.sg * (XROWS * KT);
        const uint PL = KT / 8;
        uint lane = c.lane;
        uint wr0 = lane / PL, wr1 = (32 / PL) + lane / PL, wc = (lane % PL) * 8;
        const device uint4* wp0 = (const device uint4*)(w + (size_t)min(nb0 + wr0, Nr - 1) * K + wc);
        const device uint4* wp1 = (const device uint4*)(w + (size_t)min(nb0 + wr1, Nr - 1) * K + wc);
        bool ok0 = nb0 + wr0 < Nr, ok1 = (32 / PL) < 8 && nb0 + wr1 < Nr;
        const uint XR = XROWS * PL / 32;   // x pieces a lane: rows x PL columns over 32 lanes
        uint xr[XR]; bool xok[XR]; const device uint4* xp[XR];
        for (uint i = 0; i < XR; ++i) {
            xr[i] = lane / PL + (32 / PL) * i;
            xok[i] = xr[i] < NB;
            xp[i] = (const device uint4*)(x + (size_t)min(xr[i], NB - 1) * K + wc);
        }
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
        uint nstep = (K + KT - 1) / KT;
        // a step's loads are issued one step ahead, into registers (deeper measured slower, here too: 1-4 steps ahead
        // took 15.7 / 17.0 / 18.7 / 20.1 ms a T=1 pass of Qwen3-0.6B)
        uint4 w0v = (0 < nstep && ok0 && wc < K) ? wp0[0] : uint4(0u);
        uint4 w1v = (0 < nstep && ok1 && wc < K) ? wp1[0] : uint4(0u);
        uint4 xv[XR];
        for (uint i = 0; i < XR; ++i) xv[i] = (0 < nstep && xok[i] && wc < K) ? xp[i][0] : uint4(0u);
        for (uint st = 0; st < nstep; ++st) {
            *((threadgroup uint4*)(ws + wr0 * KT + wc)) = w0v;
            if ((32 / PL) < 8) *((threadgroup uint4*)(ws + wr1 * KT + wc)) = w1v;
            for (uint i = 0; i < XR; ++i) *((threadgroup uint4*)(xs + xr[i] * KT + wc)) = xv[i];
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint sn = st + 1;
            uint kn = sn * KT;
            bool more = sn < nstep;
            w0v = (more && ok0 && kn + wc < K) ? wp0[sn * (KT / 8)] : uint4(0u);
            w1v = (more && ok1 && kn + wc < K) ? wp1[sn * (KT / 8)] : uint4(0u);
            for (uint i = 0; i < XR; ++i) xv[i] = (more && xok[i] && kn + wc < K) ? xp[i][sn * (KT / 8)] : uint4(0u);
            for (uint ks = 0; ks < KT / 8; ++ks) {
                simdgroup_bfloat8x8 A0, A1, Bt;
                simdgroup_load(Bt, ws + ks * 8, KT, ulong2(0, 0), true);
                simdgroup_load(A0, xs + ks * 8, KT);
                simdgroup_multiply_accumulate(acc0, A0, Bt, acc0);
                if (XROWS > 8) {
                    simdgroup_load(A1, xs + 8 * KT + ks * 8, KT);
                    simdgroup_multiply_accumulate(acc1, A1, Bt, acc1);
                }
            }
            // no barrier after the fragments are read: the next step's stores follow them in the simdgroup's own
            // program order (a second barrier measured 1-2% slower; none at all the same)
        }
        threadgroup float* out = (threadgroup float*)xs;
        simdgroup_store(acc0, out, 8);
        if (XROWS > 8) simdgroup_store(acc1, out + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = lane; o < NB * 8; o += 32) {
            uint b = o / 8, n = o % 8;
            if (nb0 + n < Nr) {
                if (I.n2 & 1u) {
                    ((device float*)I.C)[(size_t)b * LD + nb0 + n] = out[o];
                } else {
                    uint u = as_type<uint>(out[o]);
                    u += 0x7FFFu + ((u >> 16) & 1u);
                    y[(size_t)b * LD + nb0 + n] = (uint16_t)(u >> 16);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
""",
)

_op(
    OpName.GATED,
    r"""
inline void op_gated(thread const Ins& I, uint item, thread const Ctx& c) {
    // mid [T, I] (C) = act(gu[:, :I]) * gu[:, I:] (A); I = n0, T = n1; item -> 256 elements. n2: 0 the fused kernel's
    // silu, 1 the step's own MLX ops for it (`a * sigmoid(a)` times up, each a bf16 op), 2 MLX's ops for gelu-tanh
    uint In = I.n0;
    uint e = item * 256 + c.tid;
    if (e < I.n1 * In) {
        uint b = e / In, col = e - b * In;
        const device bfloat* r = (const device bfloat*)I.A + (size_t)b * 2 * In;
        bfloat g = r[col], u = r[In + col];
        device bfloat* out = (device bfloat*)I.C + e;
        if (I.n2 == 2) *out = mlx_gelu_tanh(g) * u;
        else if (I.n2 == 1) *out = (g * mlx_sigmoid(g)) * u;
        else *out = btb_silu_mul<bfloat>((float)g, (float)u);
    }
}
""",
)

_op(
    OpName.SANDWICH_ADD,
    r"""
inline void op_sandwich_add(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item`: out (C) = h (A) + rmsnorm(y (B)) * w (E: float32 where n1 & 1, a centered norm's 1 + w, else bf16),
    // Gemma's sandwich residual (fused.py `btb_sandwich_add_row`); H = n0, eps = f0
    uint H = I.n0;
    size_t o = (size_t)item * H;
    const device bfloat* hr = (const device bfloat*)I.A + o;
    const device bfloat* yr = (const device bfloat*)I.B + o;
    device bfloat* outr = (device bfloat*)I.C + o;
    threadgroup float* part = (threadgroup float*)c.tgm;
    if (I.n1 & 1u) btb_sandwich_add_row(hr, yr, outr, (const device float*)I.E, H, I.f0, c.tid, c.lane, c.sg, part);
    else btb_sandwich_add_row(hr, yr, outr, (const device bfloat*)I.E, H, I.f0, c.tid, c.lane, c.sg, part);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.ADD_RMSNORM,
    r"""
inline void op_add_rmsnorm(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item`: h2 (C) = h (A) + y (B); x (D) = rmsnorm(h2) * w (E, float32 where n1 & 2: a centered norm's 1 + w);
    // H = n0, eps = f0. n1 & 1: as the step's own MLX ops make them (a family off the fused kernels): the add rounded
    // to bf16, then MLX's rms_norm over it
    uint H = I.n0;
    size_t o = (size_t)item * H;
    const device bfloat* hr = (const device bfloat*)I.A + o;
    const device bfloat* yr = (const device bfloat*)I.B + o;
    device bfloat* h2r = (device bfloat*)I.C + o;
    device bfloat* xr = (device bfloat*)I.D + o;
    bool wf32 = I.n1 & 2u;
    if (I.n1 & 1u) {
        for (uint i = c.tid; i < H; i += 256) h2r[i] = (bfloat)((float)hr[i] + (float)yr[i]);
        if (wf32) mlx_rms_norm_row(h2r, xr, (const device float*)I.E, H, I.f0, c);
        else mlx_rms_norm_row(h2r, xr, (const device bfloat*)I.E, H, I.f0, c);
        return;
    }
    threadgroup float* part = (threadgroup float*)c.tgm;
    if (wf32) btb_add_rmsnorm_row(hr, yr, h2r, xr, (const device float*)I.E, H, I.f0, c.tid, c.lane, c.sg, part);
    else btb_add_rmsnorm_row(hr, yr, h2r, xr, (const device bfloat*)I.E, H, I.f0, c.tid, c.lane, c.sg, part);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.EMBED,
    r"""
inline void op_embed(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item` of h (C) = the table (A, [V, H]) row ids[item]; H = n0. n1 & 1: times f0 as the step scales it
    // (`rows * embed_scale`: MLX rounds the Python float to bf16, then the product)
    uint H = I.n0;
    const device bfloat* src = (const device bfloat*)I.A + (size_t)c.ids[item] * H;
    device bfloat* dst = (device bfloat*)I.C + (size_t)item * H;
    if (I.n1 & 1u) {
        bfloat s = (bfloat)I.f0;
        for (uint i = c.tid; i < H; i += 256) dst[i] = src[i] * s;
    } else {
        for (uint i = c.tid; i < H; i += 256) dst[i] = src[i];
    }
}
""",
)

_op(
    OpName.QKROPE,
    r"""
inline void op_qkrope(thread const Ins& I, uint item, thread const Ctx& c) {
    // the q/k norms (where n0's top bit is set), the rope on the first 2 * hd2 dims (hd2 = n0's low bits) and the
    // append: item -> (row t, head hh) over Hq q, Hk k and Hk v heads of qkv (A, [T, (Hq + 2Hk) * D]; where n1's bit 30
    // is set [T, (2Hq + 2Hk) * D], each q head followed by its gate's, Qwen3.5's); q -> C [T, Hq, D]; k, v -> row
    // past + t (D: this layer's K, E: its V, each [Hk, cap, D]; under MAP the row the conversation's row past + t lies
    // at); the norm weights qn, kn [D] end to end at B (bf16, or float32 where n2's top bit is set: a centered norm's
    // 1 + w), the norms the fused kernel's or, where n1's top bit is set, MLX's rms_norm rounded to bf16 before the
    // rope as the step's own ops make it; Hq = n1 & 0xffff with the rope's row of `freqs` above it, Hk = n2 & 0xffff,
    // D = n2's bits 16..30; f0 the norms' eps, or without them the scale on the rotated dims (rope_fast's
    // `y * scaling`: MLX rounds the Python float to the array's bf16 and rounds the product)
    bool norm = I.n0 >> 31, wf32 = I.n2 >> 31, mlxnorm = I.n1 >> 31, gated = (I.n1 >> 30) & 1u;
    uint hd2 = I.n0 & 0x7fffffffu;
    uint Hq = I.n1 & 0xffffu, Hk = I.n2 & 0xffffu, D = (I.n2 >> 16) & 0x7fffu;
    const device float* freqs = c.freqs + ((I.n1 >> 16) & 0x3fffu) * hd2;
    uint CAPR = c.dims[2];
    uint per = Hq + 2 * Hk;
    uint t = item / per, hh = item - t * per;
#if MAP
    uint row = c.rmap[c.dims[3] + t];  // a paged cache: the region row the conversation's row past + t lies at
#else
    uint row = c.dims[3] + t;
#endif
    uint at = gated ? (hh < Hq ? 2 * hh : hh + Hq) : hh;
    const device bfloat* src = (const device bfloat*)I.A + ((size_t)t * (gated ? per + Hq : per) + at) * D;
    if (hh >= Hq + Hk) {
        device bfloat* dst = (device bfloat*)I.E + ((size_t)(hh - Hq - Hk) * CAPR + row) * D;
        for (uint i = c.tid; i < D; i += 256) dst[i] = src[i];
        return;
    }
    bool isq = hh < Hq;
    device bfloat* dst = isq ? ((device bfloat*)I.C + ((size_t)t * Hq + hh) * D)
                             : ((device bfloat*)I.D + ((size_t)(hh - Hq) * CAPR + row) * D);
    const device bfloat* wn = (const device bfloat*)I.B + (isq ? 0 : D);
    threadgroup float* part = (threadgroup float*)c.tgm;
    if (mlxnorm) {
        // the head normed into its destination, then turned there (every element read back by the thread that wrote
        // it); the norm's weights and eps as the fused norm's, read by MLX's arithmetic
        if (wf32) mlx_rms_norm_row(src, dst, (const device float*)I.B + (isq ? 0 : D), D, I.f0, c);
        else mlx_rms_norm_row(src, dst, wn, D, I.f0, c);
        threadgroup_barrier(mem_flags::mem_device);
        btb_qk_norm_rope_head<false>(dst, dst, wn, D, hd2, 0.0f, freqs, c.pos[t], c.tid, c.lane, c.sg, part, part + 8);
    } else if (norm && wf32) {
        const device float* wf = (const device float*)I.B + (isq ? 0 : D);
        btb_qk_norm_rope_head<true>(src, dst, wf, D, hd2, I.f0, freqs, c.pos[t], c.tid, c.lane, c.sg, part, part + 8);
    } else if (norm) {
        btb_qk_norm_rope_head<true>(src, dst, wn, D, hd2, I.f0, freqs, c.pos[t], c.tid, c.lane, c.sg, part, part + 8);
    } else {
        btb_qk_norm_rope_head<false>(src, dst, wn, D, hd2, 0.0f, freqs, c.pos[t], c.tid, c.lane, c.sg, part, part + 8);
        float rs = (float)(bfloat)I.f0;
        if (rs != 1.0f && c.tid < hd2) {
            dst[c.tid] = (bfloat)((float)dst[c.tid] * rs);
            dst[c.tid + hd2] = (bfloat)((float)dst[c.tid + hd2] * rs);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.ATTN,
    r"""
inline void op_attn(thread const Ins& I, uint3 at, thread const Ctx& c) {
    // aluminum's decode (`btb_alu_segment`) at grid position (kv head and head group, segment, node): q (A, [T, Hq, D]
    // bf16), this layer's K (B) and V (E) [Hk, cap, D] (under MAP every row through the row map, the pass's own written
    // to their pages already); a node of one segment's row out at C + A_OFF, the others' partials m (C), l (D) and o
    // (C + PACC_OFF). n0 = Hk * ATT_NHG, segments = n1, T = n2's low half with the layer type's row of `meta` above
    // it, scale = f0
    uint Hk = I.n0 / ATT_NHG, Tn = I.n2 & 0xffffu;
    BtbAluIn<const device bfloat*, const device bfloat*, const device uint*, const device uint*, const device float*,
             const device float*, const device uint*, const device float*> a = {
        (const device bfloat*)I.A, (const device bfloat*)I.B, (const device bfloat*)I.E, nullptr, nullptr,
        c.meta + (I.n2 >> 16) * Tn * 8, c.path, nullptr, nullptr, nullptr, nullptr, c.rmap, nullptr, I.f0,
        (int)(Hk * ATT_G), (int)Hk, (int)c.dims[2], (int)c.dims[1], 0, (int)I.n1};
    btb_alu_segment<ATT_D, ATT_G, ATT_GH, 256, ATT_CH, ATT_VT>(a, at.x, at.y, at.z, c.tid, c.tgm,
        (device bfloat*)(I.C + A_OFF), (device float*)I.C, (device float*)I.D, (device float*)(I.C + PACC_OFF));
}
""",
)

_op(
    OpName.ATTN_TREE,
    r"""
inline void op_attn_tree(thread const Ins& I, uint3 at, thread const Ctx& c) {
    // aluminum's tree tile (`btb_alu_tree`, 8 simdgroups) at grid position (the tile's rows, kv head, segment of those
    // the nodes span): op_attn's buffers and outputs; Hk = n1, segments = n2's low half with the layer type's row of
    // `meta` above it, the pass's rows dims[0], scale = f0
    uint Hk = I.n1, ns = I.n2 & 0xffffu, Tn = c.dims[0];
    BtbAluIn<const device bfloat*, const device bfloat*, const device uint*, const device uint*, const device float*,
             const device float*, const device uint*, const device float*> a = {
        (const device bfloat*)I.A, (const device bfloat*)I.B, (const device bfloat*)I.E, nullptr, nullptr,
        c.meta + (I.n2 >> 16) * Tn * 8, c.path, nullptr, nullptr, nullptr, nullptr, c.rmap, nullptr, I.f0,
        (int)(Hk * ATT_G), (int)Hk, (int)c.dims[2], (int)c.dims[1], 0, (int)ns};
#if ATT_D == 256
    btb_alu_tree256<ATT_D, ATT_G, 8>(a, (int)Tn, at.x, at.y, at.z, c.tid, c.tgm,
#else
    btb_alu_tree<ATT_D, ATT_G, 8>(a, (int)Tn, at.x, at.y, at.z, c.tid, c.tgm,
#endif
        (device bfloat*)(I.C + A_OFF), (device float*)I.C, (device float*)I.D, (device float*)(I.C + PACC_OFF));
}
""",
)

_op(
    OpName.ATTN_PHASED,
    r"""
inline void op_attn_phased(thread const Ins& I, uint3 at, thread const Ctx& c) {
    // aluminum's phased tree attention (`btb_alu_phased256`, head 256, 8 simdgroups) at grid position (the rows, kv
    // head, segment of those the nodes span): op_attn_tree's buffers, outputs and n, 16 rows a threadgroup where n2's
    // top bit is set, else 8
    uint Hk = I.n1, ns = I.n2 & 0xffffu, Tn = c.dims[0];
    BtbAluIn<const device bfloat*, const device bfloat*, const device uint*, const device uint*, const device float*,
             const device float*, const device uint*, const device float*> a = {
        (const device bfloat*)I.A, (const device bfloat*)I.B, (const device bfloat*)I.E, nullptr, nullptr,
        c.meta + ((I.n2 >> 16) & 0x7fffu) * Tn * 8, c.path, nullptr, nullptr, nullptr, nullptr, c.rmap, nullptr, I.f0,
        (int)(Hk * ATT_G), (int)Hk, (int)c.dims[2], (int)c.dims[1], 0, (int)ns};
    device bfloat* one = (device bfloat*)(I.C + A_OFF);
    device float* pacc = (device float*)(I.C + PACC_OFF);
    if (I.n2 >> 31)
        btb_alu_phased256<ATT_D, ATT_G, 8, 16>(a, (int)Tn, at.x, at.y, at.z, c.tid, c.tgm, one, (device float*)I.C,
                                               (device float*)I.D, pacc);
    else
        btb_alu_phased256<ATT_D, ATT_G, 8, 8>(a, (int)Tn, at.x, at.y, at.z, c.tid, c.tgm, one, (device float*)I.C,
                                              (device float*)I.D, pacc);
}
""",
)

_op(
    OpName.FOLD,
    r"""
inline void op_fold(thread const Ins& I, uint item, thread const Ctx& c) {
    // aluminum's fold (`btb_alu_fold`), an output (node, query head, column) a thread, item * 256 + tid: the partials
    // m (A), l (B), o (A + PACC_OFF) of the nodes of more than one segment folded in order, a node of one segment's
    // row the attention wrote (A + A_OFF) copied -> out (C, [T, Hq * D] bf16); Hk = n0, segments = n1, T and the
    // layer type's `meta` row as the attention's (n2)
    uint Tn = I.n2 & 0xffffu;
    btb_alu_fold<ATT_D>(c.meta + (I.n2 >> 16) * Tn * 8, (const device float*)nullptr, (const device float*)I.A,
                        (const device float*)I.B, (const device float*)(I.A + PACC_OFF),
                        (const device bfloat*)(I.A + A_OFF), (int)Tn, (int)(I.n0 * ATT_G), (int)I.n1,
                        item * 256 + c.tid, (device bfloat*)I.C);
}
""",
)

_op(
    OpName.ARGMAX,
    r"""
inline void op_argmax(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item`'s id: the partials (A: [T, P] float, [T, P] uint at n2) folded in part order, the first of equal
    // values winning (the lowest index); P = n1
    uint P = I.n1;
    if (c.tid == 0) {
        const device float* pv = (const device float*)I.A + (size_t)item * P;
        const device uint* pi = (const device uint*)I.A + I.n2 + (size_t)item * P;
        float b = pv[0]; uint bi0 = pi[0];
        for (uint p = 1; p < P; ++p) if (pv[p] > b) { b = pv[p]; bi0 = pi[p]; }
        c.out[item] = bi0;
    }
}
""",
)

_op(
    OpName.FNORM,
    r"""
inline void op_fnorm(thread const Ins& I, uint item, thread const Ctx& c) {
    // the final norm in float32: x (C, [T, H] float32) = rmsnorm(h (A, bf16)) * w (B, float32); where n1 & 1, h is
    // the residual add h + y (E) rounded to bf16, as the block's own add would store it, and that residual (the last
    // layer's output, a layer hook's) into D [T, H] bf16; H = n0, eps = f0
    uint H = I.n0;
    const device bfloat* hr = (const device bfloat*)I.A + (size_t)item * H;
    const device bfloat* yr = (const device bfloat*)I.E + (size_t)item * H;
    bool add = I.n1 & 1u;
    const device float* w = (const device float*)I.B;
    device float* xr = (device float*)I.C + (size_t)item * H;
    threadgroup float* part = (threadgroup float*)c.tgm;
    float ss = 0.0f;
    for (uint i = c.tid; i < H; i += 256) {
        float v = add ? (float)(bfloat)((float)hr[i] + (float)yr[i]) : (float)hr[i];
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (c.lane == 0) part[c.sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint q = 0; q < 8; ++q) tot += part[q];
    float scale = metal::rsqrt(tot / (float)H + I.f0);
    device bfloat* res = (device bfloat*)I.D + (size_t)item * H;
    for (uint i = c.tid; i < H; i += 256) {
        float v = add ? (float)(bfloat)((float)hr[i] + (float)yr[i]) : (float)hr[i];
        xr[i] = v * scale * w[i];
        res[i] = (bfloat)v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.ARGMAX_PART,
    r"""
inline void op_argmax_part(thread const Ins& I, uint item, thread const Ctx& c) {
    // item -> (row t = item / P, part p): the best of the logits (A, [T, V] float32) in part p of row t, its value
    // and index into C ([T, P] float, [T, P] uint behind it at n2); V = n0, P = n1
    uint V = I.n0, P = I.n1;
    uint t = item / P, p = item - t * P;
    uint lo = (uint)(((ulong)p * V) / P), hi = (uint)(((ulong)(p + 1) * V) / P);
    const device float* row = (const device float*)I.A + (size_t)t * V;
    threadgroup float* bv = (threadgroup float*)c.tgm;
    threadgroup uint* bi = (threadgroup uint*)c.tgm + 256;
    float best = -INFINITY; uint bidx = lo;
    for (uint i = lo + c.tid; i < hi; i += 256) { float v = row[i]; if (v > best) { best = v; bidx = i; } }
    bv[c.tid] = best; bi[c.tid] = bidx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c.tid == 0) {
        float b = bv[0]; uint bi0 = bi[0];
        for (uint k = 1; k < 256; ++k) if (bv[k] > b || (bv[k] == b && bi[k] < bi0)) { b = bv[k]; bi0 = bi[k]; }
        ((device float*)I.C)[item] = b;
        ((device uint*)I.C)[I.n2 + item] = bi0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.GATE,
    r"""
inline void op_gate(thread const Ins& I, uint item, thread const Ctx& c) {
    // y (C, [T, n0] bf16) = a (A) * sigmoid(gate), Qwen3.5's output gate as the step's MLX ops make it on bf16; the
    // gate the second half of each q head's 2 * n1 columns in qkv (B, rows of n2); item -> 256 elements
    uint e = item * 256 + c.tid, W = I.n0, D = I.n1;
    if (e >= c.dims[0] * W) return;
    uint t = e / W, j = e - t * W;
    bfloat g = ((const device bfloat*)I.B)[(size_t)t * I.n2 + (j / D) * 2 * D + D + j % D];
    ((device bfloat*)I.C)[e] = ((const device bfloat*)I.A)[e] * mlx_sigmoid(g);
}
""",
)

_op(
    OpName.DELTA_CONV,
    r"""
inline void op_delta_conv(thread const Ins& I, uint item, thread const Ctx& c) {
    // the DeltaNet's front, as the step's float32 MLX ops make it (delta.py `delta_tree_step`): a thread a channel of
    // the C = n0 the conv reads (`mixed`, the first C columns of each row of the projections p, A: [T, n1] bf16), each
    // node's window its K = DELTA_K newest inputs along its path, the stored inputs (the conv state E, [C, K] float32,
    // its last K - 1) past the root; xc (C, [T, C]) = silu(sum_j window_j * w_j + bias), w and the bias the layer's
    // constants (B: conv_w [C, K], conv_b [C], a_log, dt_bias [Hv]). Thread e < T * Hv besides: g = -exp(a_log) *
    // logaddexp(a + dt_bias, 0) and beta = sigmoid(b) (D, [T, 2Hv]: g then beta), b and a p's last 2Hv columns. n2: Hv
    // (low 16 bits), a bias (bit 16), the windows kept for a verify's commit (bit 17: F, [T, C, K]) else the last
    // node's written back over the state (a chain's: one row, or a continuation's)
    uint C = I.n0, P = I.n1, Hv = I.n2 & 0xffffu, T = c.dims[0];
    bool bias = (I.n2 >> 16) & 1u, keep = (I.n2 >> 17) & 1u;
    uint e = item * 256 + c.tid;
    const device bfloat* p = (const device bfloat*)I.A;
    const device float* w = (const device float*)I.B;
    if (e < T * Hv) {
        uint t = e / Hv, h = e - t * Hv;
        const device float* a_log = w + C * DELTA_K + (bias ? C : 0);
        const device float* dt_bias = a_log + Hv;
        float b = (float)p[(size_t)t * P + P - 2 * Hv + h], a = (float)p[(size_t)t * P + P - Hv + h];
        float lae = mlx_logaddexp(a + dt_bias[h], 0.0f);
        ((device float*)I.D)[(size_t)t * 2 * Hv + h] = -metal::precise::exp(a_log[h]) * lae;
        ((device float*)I.D)[(size_t)t * 2 * Hv + Hv + h] = mlx_sigmoid(b);
    }
    if (e >= C) return;
    device float* st = (device float*)I.E + (size_t)e * DELTA_K;
    float prev[DELTA_K - 1];
    for (uint j = 0; j + 1 < DELTA_K; ++j) prev[j] = st[j + 1];
    float in[XROWS];
    for (uint t = 0; t < T; ++t) in[t] = (float)p[(size_t)t * P + e];
    const device float* we = w + (size_t)e * DELTA_K;
    float win[DELTA_K];
    for (uint t = 0; t < T; ++t) {
        uint d = c.pos[t] - c.dims[3];
        const device uint* path = c.path + (size_t)t * c.dims[1];
        for (uint j = 0; j < DELTA_K; ++j) {
            // j steps back from the node: its path's nodes, then the stored inputs newest first
            win[DELTA_K - 1 - j] = j <= d ? in[path[d - j]] : prev[DELTA_K - 1 - (j - d)];
        }
        float xc = win[0] * we[0];
        for (uint j = 1; j < DELTA_K; ++j) {
            float pr = win[j] * we[j];
            xc = xc + pr;
        }
        if (bias) xc = xc + w[C * DELTA_K + e];
        ((device float*)I.C)[(size_t)t * C + e] = xc * mlx_sigmoid(xc);
        if (keep) {
            device float* ck = (device float*)I.F + ((size_t)t * C + e) * DELTA_K;
            for (uint j = 0; j < DELTA_K; ++j) ck[j] = win[j];
        }
    }
    if (!keep)
        for (uint j = 0; j < DELTA_K; ++j) st[j] = win[j];
}
""",
)

_op(
    OpName.DELTA_RULE,
    r"""
inline void op_delta_rule(thread const Ins& I, uint item, thread const Ctx& c) {
    // the gated delta rule over the pass's nodes, delta.py's tree kernel node by node (each from its parent's state,
    // the root from the layer's, E: [Hv, DK, DV] float32): lane = c.lane over DELTA_NPT key elements each, column j of
    // head h a simdgroup, 8 a threadgroup. q, k raw (the first 2 * key_dim columns of xc, A: [T, C]), v behind them;
    // g and beta (B, [T, 2Hv]); out (C, [T, Hv, DV]). Hv = n0, Hk = n1; n2 set: a verify's, the states left for its
    // commit, else the last node's state written back over the layer's (a chain's)
    const uint DK = DELTA_DK, DV = DELTA_DV, NPT = DELTA_NPT;
    uint Hv = I.n0, Hk = I.n1, T = c.dims[0];
    uint h = item / (DV / 8), j = (item % (DV / 8)) * 8 + c.sg, lane = c.lane;
    uint hk = h / (Hv / Hk), key_dim = Hk * DK, C = 2 * key_dim + Hv * DV;
    const device float* xc = (const device float*)I.A;
    const device float* gb = (const device float*)I.B;
    device float* state = (device float*)I.E;
    const float qscale = 1.0f / metal::precise::sqrt((float)DK);
    float S[XROWS][NPT];
    for (uint t = 0; t < T; ++t) {
        uint d = c.pos[t] - c.dims[3];
        int par = d ? (int)c.path[(size_t)t * c.dims[1] + d - 1] : -1;
        float cur[NPT];
        if (par < 0) {
            for (uint i = 0; i < NPT; ++i) cur[i] = state[((size_t)h * DK + lane * NPT + i) * DV + j];
        } else {
            for (uint i = 0; i < NPT; ++i) cur[i] = S[par][i];
        }
        float gt = exp(gb[(size_t)t * 2 * Hv + h]);
        float bt = gb[(size_t)t * 2 * Hv + Hv + h];
        float vt = xc[(size_t)t * C + 2 * key_dim + h * DV + j];
        const device float* qt = xc + (size_t)t * C + hk * DK + lane * NPT;
        const device float* kt = qt + key_dim;
        float kk[NPT];
        float qq[NPT];
        for (uint i = 0; i < NPT; ++i) { kk[i] = kt[i]; qq[i] = qt[i]; }
        float ksq = 0.0f, qsq = 0.0f;
        for (uint i = 0; i < NPT; ++i) { ksq = fma(kk[i], kk[i], ksq); qsq = fma(qq[i], qq[i], qsq); }
        float kn = metal::precise::rsqrt(simd_sum(ksq) + 1e-6f);
        float qn = metal::precise::rsqrt(simd_sum(qsq) + 1e-6f) * qscale;
        for (uint i = 0; i < NPT; ++i) { kk[i] *= kn; qq[i] *= qn; }
        float kv = 0.0f;
        for (uint i = 0; i < NPT; ++i) {
            cur[i] *= gt;
            kv = fma(cur[i], kk[i], kv);
        }
        kv = simd_sum(kv);
        float delta = (vt - kv) * bt;
        float o = 0.0f;
        for (uint i = 0; i < NPT; ++i) {
            cur[i] = fma(kk[i], delta, cur[i]);
            o = fma(cur[i], qq[i], o);
        }
        o = simd_sum(o);
        if (lane == 0) ((device float*)I.C)[((size_t)t * Hv + h) * DV + j] = o;
        for (uint i = 0; i < NPT; ++i) S[t][i] = cur[i];
    }
    if (!I.n2)
        for (uint i = 0; i < NPT; ++i) state[((size_t)h * DK + lane * NPT + i) * DV + j] = S[T - 1][i];
}
""",
)

_op(
    OpName.DELTA_NORM,
    r"""
inline void op_delta_norm(thread const Ins& I, uint item, thread const Ctx& c) {
    // the DeltaNet's gated norm as the step's float32 MLX ops make it (delta.py `_delta_post`): a simdgroup a row of
    // the rule's out (A, [T * Hv, DV]) through MLX's rms_norm with the layer's weight (E, [DV] float32), times
    // z * sigmoid(z) (z: Hv * DV columns of each row of the projections, B: rows of n1 bf16, from column n2), rounded
    // to bf16 (C, [T, Hv * DV]) for the out projection. Hv = n0, eps = f0
    const uint DV = DELTA_DV;
    uint r = item * 8 + c.sg, Hv = I.n0;
    if (r >= c.dims[0] * Hv) return;
    uint t = r / Hv, h = r - t * Hv;
    const device float* x = (const device float*)I.A + (size_t)r * DV;
    float inv = mlx_rms_inv_simd(x, DV, I.f0, c.lane);
    const device float* w = (const device float*)I.E;
    const device bfloat* z = (const device bfloat*)I.B + (size_t)t * I.n1 + I.n2 + h * DV;
    device bfloat* y = (device bfloat*)I.C + (size_t)r * DV;
    for (uint i = c.lane; i < DV; i += 32) {
        float n = w[i] * (x[i] * inv);
        float zf = (float)z[i];
        float s = zf * mlx_sigmoid(zf);
        y[i] = (bfloat)(n * s);
    }
}
""",
)

# a stored type's `embed` in its namespace: a row decoded to bf16
_STORED_FNS = r"""
inline void embed(PARAMS, uint row, uint COLS, device bfloat* dst, uint tid) {
    const uint NSB = COLS / BW;
DECODE
}
"""

# the instruction fields a stored type's inputs (`stored_inputs`) arrive in beside x (B) and y (C), in order, and
# each field's reference (`Ins`' order)
STORED_FIELDS = ("A", "D", "E", "F")
REF_OF = {f: i for i, f in enumerate("ABCDEF")}


def stored_inputs(k: StoredKernel) -> list[tuple[str, str]]:
    """what a stored type's kernels read, each by its name and element type: its streams, then its tables"""
    ins = [*k.streams, *((t.name, t.ctype) for t in k.tables)]
    assert len(ins) <= len(STORED_FIELDS), f"a stored type reads {len(ins)} inputs, past the instruction's"
    return ins


_EMBED = r"""
inline void op_embed_KEY(thread const Ins& I, uint item, thread const Ctx& c) {
    // row `item` of h (C) = the table's row ids[item] (A, D, E, F: TYPE's streams and tables) as the step's gather
    // decodes it; H = n0; n1 & 1: then times f0 as the step scales it (`rows * embed_scale`: the Python float rounded
    // to bf16 first)
    device bfloat* dst = (device bfloat*)I.C + (size_t)item * I.n0;
    btb_KEY::embed(ARGS, c.ids[item], I.n0, dst, c.tid);
    if (I.n1 & 1u) {
        threadgroup_barrier(mem_flags::mem_device);
        bfloat s = (bfloat)I.f0;
        for (uint i = c.tid; i < I.n0; i += 256) dst[i] = dst[i] * s;
    }
}
"""


def _undo(defs: str) -> str:
    """`#undef`s of what `defs` defines"""
    names = (ln.split()[1].split("(")[0] for ln in defs.splitlines() if ln.startswith("#define "))
    return "".join(f"#undef {n}\n" for n in names)


def _stored_prelude(q: Quant, k: StoredKernel) -> str:
    """type `q`'s namespace for the header: its defines, its device functions and `embed`, the defines undone"""
    defs = f"#define BW {k.bw}\n{k.defs}"
    params = ", ".join(f"const device {t}* {n}" for n, t in stored_inputs(k))
    fns = _STORED_FNS.replace("PARAMS", params).replace("DECODE", k.decode)
    return f"{defs}namespace btb_{quant_key(q)} {{\n{k.header}{fns}}}\n{_undo(defs)}"


for _q, _k in STORED.items():
    _PRELUDE[EMBED_OP[_q]] = (quant_key(_q), _stored_prelude(_q, _k))
    _args = ", ".join(f"(const device {t}*)I.{a}" for (_, t), a in zip(stored_inputs(_k), STORED_FIELDS, strict=False))
    _op(EMBED_OP[_q], _EMBED.replace("KEY", quant_key(_q)).replace("TYPE", _q.value).replace("ARGS", _args))

# a stored type's matvec, a kernel of its own a matrix and pass width (`stored_source`); the bf16 head's, the step's
# float32 matvec a kernel of its own a matrix and width (`head_source`)
MV_Q: dict[OpName, Quant] = {o: q for q, o in MV_OP.items()}
PER_SHAPE: frozenset[OpName] = frozenset({*MV_Q, OpName.HEAD})
assert set(SOURCE) == set(OpName) - PER_SHAPE, "an op without its device function"

# a command's buffers after the table of addresses its references go through (`Buf`): these, then the instruction's
# index
INPUTS = ("freqs", "ids", "pos", "meta", "path", "rmap", "tab", "dims", "out")
# a buffer's type in a kernel's signature where it is not a uint array read
_DECL = {"freqs": "const device float*", "out": "device uint*"}
# an entry of the table of addresses: a buffer's GPU address (the weights', the scratch, the constants, each layer's K
# and V, numbered as `Program.table` numbers them); a kernel argument cannot be a pointer to pointers itself
_BUF = "struct Buf { device uint8_t* p; };\n"


def _mlx_header(rel: str, seen: set[str]) -> str:
    """MLX's kernel header `rel` from the installed package with its own includes inlined once each (a library built
    from source has no include path): the preamble MLX compiles its custom kernels behind"""
    if rel in seen:
        return ""
    seen.add(rel)
    pkg = mx.__file__
    if pkg is None:
        raise RuntimeError("[mega] MLX is installed without a path to its headers")
    root = os.path.join(os.path.dirname(pkg), "include")
    out = []
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        for line in fh.read().splitlines():
            s = line.strip()
            if s.startswith('#include "') and s.endswith('"'):
                out.append(_mlx_header(s[len('#include "') : -1], seen))
            elif s != "#pragma once":
                out.append(line)
    return "\n".join(out) + "\n"


def kernel_name(op: OpName) -> str:
    return f"btb_mega_{op.value}"


class StoredMv(NamedTuple):
    """a stored matvec instruction's kernel: its type, its matrix's rows and columns, the pass's rows, y's row stride
    and whether x and y are float32 (the head) rather than bf16"""

    q: Quant
    rows: int
    cols: int
    T: int
    ld: int
    f32: bool


def stored_name(s: StoredMv) -> str:
    return f"btb_mega_mv_{quant_key(s.q)}_{s.rows}_{s.cols}_{s.T}_{s.ld}_{int(s.f32)}"


def stored_groups(q: Quant, rows: int, T: int) -> int:
    """the threadgroups of 256 a stored matvec of `rows` at `T` pass rows takes, its launch's grid"""
    return -(-matvec.threads(rows, T, STORED[q].shapes) // 256)


def _signature(name: str, attrs: str) -> str:
    """kernel `name` over a command's buffers (the table of addresses `B` its references index, then the pass's), its
    row of the table `f`, `attrs` its position arguments"""
    ins = ", ".join(
        f"{_DECL.get(n, 'const device uint*')} {n} [[buffer({j + 1})]]" for j, n in enumerate((*INPUTS, "mi"))
    )
    return f"""
kernel void {name}(const device Buf* B [[buffer(0)]], {ins}, {attrs}) {{
    const device uint* f = tab + mi[0] * ROW;
"""


def _wrap(op: OpName, tgm: int) -> str:
    """op `op`'s kernel: the instruction's row (its index bound at the command's own offset) read into the op's view,
    the threadgroup's memory `tgm` bytes, a `GRID3` op given its grid position"""
    pos = "uint3" if op in GRID3 else "uint"
    tid = "tid3.x" if op in GRID3 else "tid3"
    attrs = (
        f"{pos} tid3 [[thread_position_in_threadgroup]], {pos} g [[threadgroup_position_in_grid]],\n"
        "    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]"
    )
    return (
        _signature(kernel_name(op), attrs)
        + f"""    threadgroup uint8_t tgm[{max(16, tgm)}];
    Ins I = {{REF(0), REF(1), REF(2), REF(3), REF(4), REF(5), f[1], f[NUM], f[NUM + 1], f[NUM + 2],
             as_type<float>(f[NUM + 3])}};
    Ctx c = {{{tid}, sg, lane, tgm, dims, freqs, ids, pos, meta, path, rmap, out}};
    op_{op.value}(I, g, c);
}}
"""
    )


def _preamble() -> str:
    """MLX's preamble, then what every kernel reads its instruction through"""
    return (
        _mlx_header("mlx/backend/metal/kernels/utils.h", set())
        + "#include <metal_stdlib>\nusing namespace metal;\n"
        + _BUF
        + f"#define ROW {W}\n#define NUM {NUM}\n"
        + f"#define REF(k) (B[f[{REF0} + 2 * (k)]].p + f[{REF0} + 2 * (k) + 1])\n"
    )


def stored_source(mvs: Iterable[StoredMv]) -> str:
    """the library of stored matvec kernels `mvs`: each its type's launch (`matvec.blocks`, the step's kernel) with its
    matrix, pass rows and y's row stride as constants, y [T, rows] (C) = x [T, cols] (B) . W^T, W the type's streams
    and tables (`stored_inputs`, in A, D, E, F); a thread of the launch's grid a thread of the command's"""
    by_q: dict[Quant, list[StoredMv]] = {}
    for s in mvs:
        by_q.setdefault(s.q, []).append(s)
    out = [_preamble()]
    refs = {f: REF_OF[f] for f in STORED_FIELDS}
    for q, ss in by_q.items():
        k, key = STORED[q], quant_key(q)
        ins = stored_inputs(k)
        params = "".join(f"const device {t}* {n}, " for n, t in ins)
        args = "".join(f"(const device {t}*)REF({refs[f]}), " for (_, t), f in zip(ins, STORED_FIELDS, strict=False))
        out.append(f"{k.defs}namespace btb_{key} {{\n{k.header}")
        for s in ss:
            d = matvec.defines(s.rows, s.cols, k.bw, s.T, k.shapes, s.ld)
            out.append(
                f"{d}template <typename T>\ninline void {stored_name(s)}({params}const device T* x, "
                f"device T* out, uint3 thread_position_in_grid) {{\n{matvec.blocks(k.frags)}}}\n{_undo(d)}"
            )
        out.append("}\n")
        for s in ss:
            t = "float" if s.f32 else "bfloat"
            out.append(
                _signature(stored_name(s), "uint3 gp [[thread_position_in_grid]]")
                + f"    btb_{key}::{stored_name(s)}<{t}>({args}(const device {t}*)REF(1), (device {t}*)REF(2), gp);\n}}\n"
            )
        out.append(_undo(k.defs))
    return "".join(out)


class HeadMv(NamedTuple):
    """the bf16 head's matvec instruction's kernel: its matrix's rows and columns and the pass's rows"""

    rows: int
    cols: int
    T: int


def head_name(h: HeadMv) -> str:
    return f"btb_mega_head_{h.rows}_{h.cols}_{h.T}"


def head_threads(T: int) -> int:
    """the threads a head kernel's threadgroup takes at T pass rows: the step's float32 matvec's simdgroups"""
    return 32 * gemm_sgn(T, False)


def head_groups(rows: int, T: int) -> int:
    """the threadgroups a head of `rows` takes at T pass rows: a simdgroup each 8 of its rows, as the step's launch"""
    return -(-rows // (8 * gemm_sgn(T, False)))


def head_source(heads: Iterable[HeadMv]) -> str:
    """the library of head kernels `heads`: each the step's float32 matvec (gemv.py `GEMM_SRC`, the launch
    `Backend.matmul` makes for the step's head) with its matrix and pass rows as constants, y [T, rows] float32 (C) = x
    [T, cols] float32 (B) . W^T, W (A) bf16; a threadgroup of the launch's a threadgroup of the command's"""
    # the command's buffers it reads, at their places in every command's binding (the source names its own `out`)
    tab, mi = INPUTS.index("tab") + 1, len(INPUTS) + 1
    out = [_preamble(), "#include <metal_simdgroup_matrix>\n"]
    for h in heads:
        d = f"#define NB {h.T}\n#define SGN {gemm_sgn(h.T, False)}\n#define XB 0\n#define WF16 0\n#define WF32 0\n"
        sig = (
            f"kernel void {head_name(h)}(const device Buf* B [[buffer(0)]], const device uint* tab [[buffer({tab})]], "
            f"const device uint* mi [[buffer({mi})]], "
            "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], "
            "uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], "
            "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n"
            "    const device uint* f = tab + mi[0] * ROW;\n"
            "    const device uint16_t* w = (const device uint16_t*)REF(0);\n"
            "    const device float* x = (const device float*)REF(1);\n"
            "    device float* y = (device float*)REF(2);\n"
            f"    const int w_shape[2] = {{{h.rows}, {h.cols}}};\n"
        )
        out.append(d + sig + GEMM_SRC + "}\n" + _undo(d))
    return "".join(out)


def source(ops: Iterable[OpName], defs: Mapping[str, int], tgm: Mapping[OpName, int], mapped: bool) -> str:
    """the library of a program's `ops`: MLX's preamble, the ops' device functions under `defs`, their compile-time
    constants (XROWS, MV_KT and the attention's shape), and a kernel `btb_mega_<op>` each, its threadgroup memory
    `tgm[op]` bytes; `mapped` a paged cache's, its rows read and written through the conversation's row map"""
    used = sorted(set(ops), key=CODE.__getitem__)
    return (
        _preamble()
        + "".join(f"#define {n} {v}\n" for n, v in sorted(defs.items()))
        + TYPES
        + MLX_OPS
        + fused.FUNCS
        + aluminum.device_header(defs["ATT_D"], mapped=mapped)
        + "".join(dict(_PRELUDE[o] for o in used if o in _PRELUDE).values())
        + "".join(SOURCE[o] for o in used)
        + "".join(_wrap(o, tgm[o]) for o in used)
    )
