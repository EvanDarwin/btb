# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel's Metal: a grid of threadgroups running a pass's instructions in order, each cut into tickets, ticket
k threadgroup k's own, run once the instructions it reads are done. A threadgroup kept waiting takes the lowest ticket
nobody has claimed below its own, so the pass completes at any residency, a single threadgroup's included. The ops
are device functions; a program compiles in the ones it uses."""

from __future__ import annotations

import threading
import zlib
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import TypeAlias

from .. import attn, fused
from ..core import MetalKernel, mx

# a table row: op, tickets, units (the op's own work items, a ticket's share of them run in turn), REFS (buffer,
# offset) references from column REF0, n0, n1, n2, f0's bits, and from column DEP0 the instructions it waits on
# (index + 1, 0 for none)
REFS = 6
REF0 = 3
NUM = REF0 + 2 * REFS  # n0's column
DEP0 = NUM + 4
DEPS = 3
W = DEP0 + DEPS
# dims: rows of the pass, the path's row stride, the arena's rows a head, past, the probe's spins (0 for a pass),
# instructions. Every per-pass input holds at least 8 elements: MLX binds a smaller one as `constant`, which the ops'
# device pointers cannot take
DIMS = 8
# polls of a threadgroup's wait on its own ticket's inputs before it looks for an unclaimed ticket below it to run:
# long enough that a resident threadgroup merely behind keeps its own (at 2^12 a Qwen3-0.6B T=1 pass took ~60 tickets
# that way, a thief's own ticket then waiting on the stolen one, 4-9% slower; past residency 2^16 recovers as fast)
PATIENCE = 1 << 16
# polls before the pass gives up on an instruction's inputs and flags it: a scheduling bug, never a load
STUCK = 1 << 24
CS = 32  # the scheduler's counters this many words apart, a 128-byte line each
TICKETS_MAX = 32 * (CS - 1)  # the claim bits a counter line holds after its count: an instruction's most tickets

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
inline float mlx_rms_inv(const device bfloat* xr, uint H, float eps, thread const Ctx& c) {
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
"""


class OpName(StrEnum):
    """the megakernel's ops, each a device function `op_<name>(I, item, c)`; its code in the table is its place
    here, from 1"""

    RMSNORM = "rmsnorm"
    MATVEC = "matvec"
    GATED = "gated"
    ADD_RMSNORM = "add_rmsnorm"
    EMBED = "embed"
    QKROPE = "qkrope"
    ATTN = "attn"
    FOLD = "fold"
    ARGMAX = "argmax"
    FNORM = "fnorm"
    HEAD = "head"
    ARGMAX_PART = "argmax_part"
    SANDWICH_ADD = "sandwich_add"


CODE: dict[OpName, int] = {o: i + 1 for i, o in enumerate(OpName)}
SOURCE: dict[OpName, str] = {}


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
    // y [T, rows] (C) = x [T, cols] (B) . W [rows, cols]^T (A); rows = n0, cols = n1, n2 & 1 a float32 y; the
    // item's share of the 8-row tiles, a simdgroup's each (the gemv tile kernel's body, NB a runtime value)
    const uint KT = MV_KT;
    const uint NB = c.dims[0];
    uint Nr = I.n0, K = I.n1;
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
                    ((device float*)I.C)[(size_t)b * Nr + nb0 + n] = out[o];
                } else {
                    uint u = as_type<uint>(out[o]);
                    u += 0x7FFFu + ((u >> 16) & 1u);
                    y[(size_t)b * Nr + nb0 + n] = (uint16_t)(u >> 16);
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
    // append: item -> (row t, head hh) over Hq q, Hk k and Hk v heads of qkv (A, [T, (Hq + 2Hk) * D]); q -> C
    // [T, Hq, D]; k, v -> the arena at row past + t (D: this layer's K, E: its V, each [Hk, cap, D]); the norm
    // weights qn, kn [D] end to end at B (bf16, or float32 where n2's top bit is set: a centered norm's 1 + w); Hq =
    // n1 & 0xffff with the rope's row of `freqs` above it, Hk = n2 & 0xffff, D = n2's bits 16..30; f0 the norms' eps, or
    // without them the scale on the rotated dims (rope_fast's `y * scaling`: MLX rounds the Python float to the
    // array's bf16 and rounds the product)
    bool norm = I.n0 >> 31, wf32 = I.n2 >> 31;
    uint hd2 = I.n0 & 0x7fffffffu;
    uint Hq = I.n1 & 0xffffu, Hk = I.n2 & 0xffffu, D = (I.n2 >> 16) & 0x7fffu;
    const device float* freqs = c.freqs + (I.n1 >> 16) * hd2;
    uint row0 = c.dims[3];
    uint CAPR = c.dims[2];
    uint per = Hq + 2 * Hk;
    uint t = item / per, hh = item - t * per;
    const device bfloat* src = (const device bfloat*)I.A + ((size_t)t * per + hh) * D;
    if (hh >= Hq + Hk) {
        device bfloat* dst = (device bfloat*)I.E + ((size_t)(hh - Hq - Hk) * CAPR + row0 + t) * D;
        for (uint i = c.tid; i < D; i += 256) dst[i] = src[i];
        return;
    }
    bool isq = hh < Hq;
    device bfloat* dst = isq ? ((device bfloat*)I.C + ((size_t)t * Hq + hh) * D)
                             : ((device bfloat*)I.D + ((size_t)(hh - Hq) * CAPR + row0 + t) * D);
    const device bfloat* wn = (const device bfloat*)I.B + (isq ? 0 : D);
    threadgroup float* part = (threadgroup float*)c.tgm;
    if (norm && wf32) {
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
inline void op_attn(thread const Ins& I, uint item, thread const Ctx& c) {
    // the node attention (attn.py `btb_attn_nodes`, a node a threadgroup): item -> (kv head, block, node); q (A,
    // [T, Hq, D] bf16), this layer's K (B) and V (E) in the arena [Hk, cap, D]; the partials m (C), l (D) and acc
    // (C + PACC_OFF); Hk = n0, splits = n1's low bits, T = n2's low half with the layer type's row of `meta` above it,
    // scale = f0. With n1's top bit (one block a node) the threadgroup folds its own partials (`btb_attn_fold`, a
    // query head a simdgroup) into out at C + A_OFF
    uint Hk = I.n0, ns = I.n1 & 0x7fffffffu, Tn = I.n2 & 0xffffu;
    const device uint* meta = c.meta + (I.n2 >> 16) * Tn * 8;
    uint h = item % Hk, node = item / (Hk * ns);
    BtbAttnIn<bfloat, const device uint*, const device uint*, const device float*, const device float*,
              const device uint*> a = {
        (const device bfloat*)I.A, (const device uint16_t*)I.B, (const device uint16_t*)I.E, meta, c.path,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, I.f0, Hk, c.dims[2], Hk * ATT_G, Tn, ns,
        c.dims[1], 0u};
    device float* pm = (device float*)I.C;
    device float* pl = (device float*)I.D;
    device float* pacc = (device float*)(I.C + PACC_OFF);
    btb_attn_nodes<ATT_G, ATT_D, 1, ATT_P>(a, h, (item / Hk) % ns, node, c.sg, c.lane, (threadgroup float4*)c.tgm,
                                           pm, pl, pacc);
    if (I.n1 >> 31) {
        // the partials this threadgroup wrote, read back by it after the barrier orders its device writes
        threadgroup_barrier(mem_flags::mem_device);
        if (c.sg < ATT_G) {
            btb_attn_fold<false>((const device float*)pm, (const device float*)pl, (const device float*)pacc, meta,
                                 (const device float*)nullptr, (node * Hk + h) * ATT_G + c.sg, c.lane, Tn, Hk, 8,
                                 ATT_G, ATT_D, (device bfloat*)(I.C + A_OFF));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.FOLD,
    r"""
inline void op_fold(thread const Ins& I, uint item, thread const Ctx& c) {
    // the fold (attn.py `btb_attn_fold`), a simdgroup per (node, query head), item * 8 + sg: the partials m (A),
    // l (B), acc (A + PACC_OFF) folded in one fixed order -> out (C, [T, Hq * D] bf16); Hk = n0, splits = n1, T and
    // the layer type's `meta` row as the attention's (n2)
    uint Tn = I.n2 & 0xffffu;
    btb_attn_fold<false>((const device float*)I.A, (const device float*)I.B, (const device float*)(I.A + PACC_OFF),
                         c.meta + (I.n2 >> 16) * Tn * 8, (const device float*)nullptr, item * 8 + c.sg, c.lane, Tn,
                         I.n0, I.n1 * 8, ATT_G, ATT_D, (device bfloat*)I.C);
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
    // the residual add h + y (E) rounded to bf16, as the block's own add would store it; H = n0, eps = f0
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
    for (uint i = c.tid; i < H; i += 256) {
        float v = add ? (float)(bfloat)((float)hr[i] + (float)yr[i]) : (float)hr[i];
        xr[i] = v * scale * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
""",
)

_op(
    OpName.HEAD,
    r"""
inline void op_head(thread const Ins& I, uint item, thread const Ctx& c) {
    // y [T, rows] float32 (C) = x [T, cols] float32 (B) . W [rows, cols]^T (A) as float 8x8 tiles (the weight widened
    // at the stage, K steps of 32: the tile kernel's float path, the same chain); rows = n0, cols = n1
    const uint KT = 32;
    const uint NB = c.dims[0];
    uint Nr = I.n0, K = I.n1;
    uint ntile = (Nr + 7) / 8;
    uint t0 = (uint)(((ulong)item * ntile) / I.items), t1 = (uint)(((ulong)(item + 1) * ntile) / I.items);
    uint lane = c.lane;
    for (uint tile = t0 + c.sg; tile < t1; tile += 8) {
        uint nb0 = tile * 8;
        const device uint16_t* w = (const device uint16_t*)I.A;
        const device float* x = (const device float*)I.B;
        threadgroup float* ws = (threadgroup float*)c.tgm + c.sg * (8 * KT);
        threadgroup float* xs = (threadgroup float*)c.tgm + 8 * (8 * KT) + c.sg * (XROWS * KT);
        uint wr = lane / 4, wc = (lane % 4) * 8;
        const device uint4* wp = (const device uint4*)(w + (size_t)min(nb0 + wr, Nr - 1) * K + wc);
        bool okw = nb0 + wr < Nr;
        const uint XR = XROWS / 4;
        uint xr[4]; bool xok[4]; const device uint4* xp[4];
        for (uint i = 0; i < XR; ++i) {
            uint piece = lane + 32 * i;
            xr[i] = piece / 8;
            uint xc = (piece % 8) * 4;
            xok[i] = xr[i] < NB;
            xp[i] = (const device uint4*)(x + (size_t)min(xr[i], NB - 1) * K + xc);
        }
        simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
        uint nstep = (K + KT - 1) / KT;
        for (uint st = 0; st < nstep; ++st) {
            uint k0 = st * KT;
            uint4 pk = (okw && k0 + wc < K) ? wp[st * (KT / 8)] : uint4(0u);
            threadgroup float4* d0 = (threadgroup float4*)(ws + wr * KT + wc);
            d0[0] = float4(as_type<float>(pk.x << 16), as_type<float>(pk.x & 0xffff0000u), as_type<float>(pk.y << 16), as_type<float>(pk.y & 0xffff0000u));
            d0[1] = float4(as_type<float>(pk.z << 16), as_type<float>(pk.z & 0xffff0000u), as_type<float>(pk.w << 16), as_type<float>(pk.w & 0xffff0000u));
            for (uint i = 0; i < XR; ++i) {
                uint piece = lane + 32 * i;
                uint xc = (piece % 8) * 4;
                uint4 xv = (xok[i] && k0 + xc < K) ? xp[i][st * (KT / 4)] : uint4(0u);
                *((threadgroup uint4*)(xs + xr[i] * KT + xc)) = xv;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ks = 0; ks < KT / 8; ++ks) {
                simdgroup_float8x8 A0, A1, Bt;
                simdgroup_load(Bt, ws + ks * 8, KT, ulong2(0, 0), true);
                simdgroup_load(A0, xs + ks * 8, KT);
                simdgroup_multiply_accumulate(acc0, A0, Bt, acc0);
                if (XROWS > 8) {
                    simdgroup_load(A1, xs + 8 * KT + ks * 8, KT);
                    simdgroup_multiply_accumulate(acc1, A1, Bt, acc1);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup float* outp = xs;
        simdgroup_store(acc0, outp, 8);
        if (XROWS > 8) simdgroup_store(acc1, outp + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint o = lane; o < NB * 8; o += 32) {
            uint b = o / 8, n = o % 8;
            if (nb0 + n < Nr) ((device float*)I.C)[(size_t)b * Nr + nb0 + n] = outp[o];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
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


_SCHED = r"""
    BUFFER_TABLE
    uint tid = thread_position_in_threadgroup.x;
    device atomic_uint* cnt = (device atomic_uint*)counters;
    threadgroup uint8_t tgm[TGM_BYTES];
    threadgroup uint sel[2];  // the ticket thread 0 picked: its instruction (N: none left) and its index
    if (dims[4]) {
        // the residency probe: how many of these threadgroups run at once (cnt: live, the most live, arrivals, done).
        // Each stays until all have arrived or its spins run out; the first to run out ends the wait for the rest,
        // so the threadgroups that could not start until others left pass straight through
        if (tid == 0 && !atomic_load_explicit(cnt + 3, memory_order_relaxed)) {
            uint live = atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed) + 1;
            atomic_fetch_max_explicit(cnt + 1, live, memory_order_relaxed);
            atomic_fetch_add_explicit(cnt + 2, 1u, memory_order_relaxed);
            uint G = threadgroups_per_grid.x;
            uint s = 0;
            while (atomic_load_explicit(cnt + 2, memory_order_relaxed) < G && !atomic_load_explicit(cnt + 3, memory_order_relaxed)) {
                if (++s >= dims[4]) atomic_store_explicit(cnt + 3, 1u, memory_order_relaxed);
            }
            atomic_fetch_sub_explicit(cnt, 1u, memory_order_relaxed);
        }
        return;
    }
    Ctx c = {tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tgm, dims, freqs, ids, pos, meta, path, out};
    const uint N = dims[5];
    const uint g = threadgroup_position_in_grid.x;
    const uint gbit = 1u << (g & 31);
    // the counters a line each (CS words apart): instruction n's done count, then its claim bits from word 1 (ticket
    // k a bit); after them the lowest instruction that may still hold an unclaimed ticket, and the stuck flag last
    device atomic_uint* lo = cnt + N * CS;
    device atomic_uint* stuck = cnt + (N + 2) * CS - 1;
    // thread 0's: the next instruction to look in for this threadgroup's own ticket, the one it holds while it waits
    // on that ticket's inputs (N for none), and the claim bits as they stood before it set its own
    uint n = 0, home = N, claim = 0;
    for (;;) {
        if (tid == 0) {
            uint wn = N, wk = 0, waited = 0;
            for (;;) {
                if (home == N) {
                    while (n < N && g >= tab[n * ROW + 1]) ++n;
                    if (n == N) break;
                    home = n++;
                    // the claim's round trip runs under the reads of its inputs' counts below
                    claim = atomic_fetch_or_explicit(cnt + home * CS + 1 + g / 32, gbit, memory_order_relaxed);
                }
                const device uint* f = tab + home * ROW;
                bool ready = true;
                uint spins = 0;
                for (uint q = DEP0; q < ROW && ready; ++q) {
                    uint d = f[q];
                    if (d == 0) continue;
                    uint need = tab[(d - 1) * ROW + 1];
                    while (atomic_load_explicit(cnt + (d - 1) * CS, memory_order_relaxed) < need) {
                        if (++spins >= PATIENCE) { ready = false; break; }
                    }
                }
                waited += spins;
                if (waited > STUCK) { atomic_store_explicit(stuck, home + 1, memory_order_relaxed); ready = true; }
                if (ready) {
                    uint h = home;
                    home = N;
                    if (!(claim & gbit)) { wn = h; wk = g; break; }
                    continue;  // another threadgroup took it
                }
                // a long wait: the lowest ticket nobody has claimed below its own. Every instruction under that one is
                // wholly claimed, by threadgroups that are running, so its inputs finish; a ticket at or above its own
                // could wait on the one it holds
                bool got = false;
                for (uint s = atomic_load_explicit(lo, memory_order_relaxed); s < home && !got; ++s) {
                    uint tk = tab[s * ROW + 1];
                    for (uint w = 0; w * 32 < tk && !got; ++w) {
                        uint want = tk - w * 32 >= 32 ? 0xffffffffu : (1u << (tk - w * 32)) - 1u;
                        device atomic_uint* mw = cnt + s * CS + 1 + w;
                        uint cur = atomic_load_explicit(mw, memory_order_relaxed);
                        while ((cur & want) != want) {
                            uint b = ctz(~cur & want);
                            uint prev = atomic_fetch_or_explicit(mw, 1u << b, memory_order_relaxed);
                            if (!(prev & (1u << b))) { wn = s; wk = w * 32 + b; got = true; break; }
                            cur = prev | (1u << b);
                        }
                    }
                    if (!got) atomic_fetch_max_explicit(lo, s + 1, memory_order_relaxed);
                }
                if (got) {
                    const device uint* fs = tab + wn * ROW;
                    for (uint q = DEP0; q < ROW; ++q) {
                        uint d = fs[q];
                        if (d == 0) continue;
                        uint need = tab[(d - 1) * ROW + 1];
                        uint sp = 0;
                        while (atomic_load_explicit(cnt + (d - 1) * CS, memory_order_relaxed) < need) {
                            if (++sp > STUCK) { atomic_store_explicit(stuck, wn + 1, memory_order_relaxed); break; }
                        }
                    }
                    break;
                }
            }
            sel[0] = wn;
            sel[1] = wk;
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        uint m = sel[0], k = sel[1];
        if (m >= N) break;
        const device uint* f = tab + m * ROW;
        Ins I = {REF(0), REF(1), (device uint8_t*)REF(2), (device uint8_t*)REF(3), REF(4), REF(5), f[2], f[NUM],
                 f[NUM + 1], f[NUM + 2], as_type<float>(f[NUM + 3])};
        // this ticket's share of the instruction's units, run in turn (32-bit bounds: units x tickets stays far
        // below 2^32, and a 64-bit division is emulated)
        uint tickets = f[1];
        uint u0 = k, u1 = k + 1;
        if (tickets != I.items) {
            u0 = (k * I.items) / tickets;
            u1 = ((k + 1) * I.items) / tickets;
        }
        for (uint item = u0; item < u1; ++item) {
            if (item != u0) threadgroup_barrier(mem_flags::mem_threadgroup);
            switch (f[0]) {
DISPATCH
            }
        }
        // the ticket's writes made visible before it counts as done
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (tid == 0) atomic_fetch_add_explicit(cnt + m * CS, 1u, memory_order_relaxed);
    }
"""

assert set(SOURCE) == set(OpName), "an op without its device function"

# a kernel by its weight buffers, its ops and its compile-time constants
KernelKey: TypeAlias = tuple[int, tuple[OpName, ...], tuple[tuple[str, int], ...]]

_lock = threading.Lock()
_kernels: dict[KernelKey, MetalKernel] = {}

# the per-pass inputs after the weight buffers, the scratch, the constants and the arena
INPUTS = ("freqs", "ids", "pos", "meta", "path", "tab", "dims")


def kernel(nbufs: int, ops: Iterable[OpName], defs: Mapping[str, int]) -> MetalKernel:
    """the megakernel over `nbufs` weight buffers with `ops` compiled in and `defs` its compile-time constants
    (XROWS, MV_KT, TGM_BYTES and the attention's shape), compiled once a process"""
    used = tuple(sorted(set(ops), key=CODE.__getitem__))
    key: KernelKey = (nbufs, used, tuple(sorted(defs.items())))
    with _lock:
        k = _kernels.get(key)
        if k is None:
            names = [f"w{i}" for i in range(nbufs)]
            table = f"const device uint8_t* B[{nbufs + 3}] = {{{', '.join([*names, 'scr', 'consts', 'kv'])}}};"
            consts = "".join(f"#define {n} {v}\n" for n, v in sorted(defs.items()))
            dispatch = "\n".join(f"                case {CODE[o]}: op_{o.value}(I, item, c); break;" for o in used)
            src = _SCHED.replace("BUFFER_TABLE", table).replace("DISPATCH", dispatch)
            k = _kernels[key] = mx().fast.metal_kernel(
                name=f"btb_mega_{nbufs}_{zlib.crc32(repr(key).encode()):08x}",
                input_names=[*names, "scr", "consts", "kv", *INPUTS],
                output_names=["out", "counters"],
                header=consts
                + f"#define ROW {W}\n#define NUM {NUM}\n#define DEP0 {DEP0}\n#define CS {CS}\n"
                + f"#define PATIENCE {PATIENCE}u\n#define STUCK {STUCK}u\n"
                + f"#define REF(k) (B[f[{REF0} + 2 * (k)]] + f[{REF0} + 2 * (k) + 1])\n"
                + TYPES
                + MLX_OPS
                + fused.FUNCS
                + "#define KQ 0\n#define SEG 0\n#define MAP 0\n"
                + attn.ATTN_FUNCS
                + "".join(SOURCE[o] for o in used),
                source=src,
            )
    return k
