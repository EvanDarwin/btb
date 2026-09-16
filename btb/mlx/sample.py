# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The pick as one Metal dispatch: a threadgroup per row of logits finds the top-k and top-p thresholds by radix
select over integer histograms (deterministic), draws hashed Gumbel noise per token and takes the argmax of the
kept tokens; temperature 0 is the plain argmax. No sort, no gather: a few passes over the row."""

from __future__ import annotations

import threading
from typing import Any

from .core import mx

BINS = 2048  # a radix level: 11 bits, three levels resolve a float's 32 ordered bits (11 + 11 + 10)
THREADS = 1024  # a row's threadgroup: the passes over the row are latency bound, so the widest group
MASS_SCALE = 1073741824.0  # a token's mass exp(s - max) in fixed point, 2^30 a unit (the sums in 64 bits)

_lock = threading.Lock()
_kernel: Any = None
_verify_kernel: Any = None
_consts: dict[tuple[int, int, float, float], tuple[Any, Any]] = {}
_nodep: Any = None


def _dep(m: Any) -> Any:
    global _nodep
    if _nodep is None:
        _nodep = m.zeros((1,), dtype=m.uint32)
        m.eval(_nodep)
    return _nodep


_SETUP = r"""
    // one threadgroup a row: `x` [R, V] float32 logits, `keys` [R, 2] uint32 (the row's noise key), `cfg` uint32
    // [top_k, mode] (mode bit 0 sampled, bit 1 top-k on, bit 2 top-p on), `fp` float32 [1 / temperature, top_p];
    // out[r] the picked token. Ties keep every token at a threshold; the argmax takes the lowest index
    const uint V = x_shape[1];
    const uint r = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const device float* row = x + (size_t)r * V;
    const float invT = fp[0];
    const float top_p = fp[1];
    const uint top_k = cfg[0];
    const uint mode = cfg[1];
    threadgroup float redf[SG_VALUE];
    threadgroup uint redu[SG_VALUE];
    threadgroup atomic_uint hcnt[BINS_VALUE];
    threadgroup atomic_uint hlo[BINS_VALUE];
    threadgroup atomic_uint hhi[BINS_VALUE];
    threadgroup uint chosen[2];
    threadgroup ulong needl[1];
    threadgroup uint ccnt[64];    // the 2048 bins as 64 chunks of 32: the scan two levels, not one 2048-long chain
    threadgroup ulong cmass[64];

    if ((mode & 1u) == 0u) {
        // greedy: the argmax, the lowest index among equals
        float best = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint i = tid; i < V; i += THREADS_VALUE) { float v = row[i]; if (v > best) { best = v; bi = i; } }
        float sbest = simd_max(best);
        uint cand = (best == sbest) ? bi : 0xFFFFFFFFu;
        uint sbi = simd_min(cand);
        if (lane == 0) { redf[sg] = sbest; redu[sg] = sbi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = redf[0]; uint b_i = redu[0];
            for (uint s = 1; s < SG_VALUE; ++s) if (redf[s] > b || (redf[s] == b && redu[s] < b_i)) { b = redf[s]; b_i = redu[s]; }
            out[r] = b_i;
        }
        return;
    }

    // s_i = x_i / T; M = max s (its pass shared with the top-k's first level when there is one)
    float M = -INFINITY;
    float m = -INFINITY;
    const bool topk_on = (mode & 2u) != 0u && top_k < V;
    if (!topk_on) {
        for (uint i = tid; i < V; i += THREADS_VALUE) m = max(m, row[i] * invT);
        m = simd_max(m);
        if (lane == 0) redf[sg] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        M = redf[0];
        for (uint s = 1; s < SG_VALUE; ++s) M = max(M, redf[s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // the ordered-uint image of s (monotone), and the two thresholds as ordered uints: a token is kept when
    // u >= tk (top-k) and u >= tp (top-p)
    uint tk = 0u, tp = 0u;
    if (topk_on) {
        uint prefix = 0u;
        uint need = top_k;
        uint shift = 32;
        for (uint level = 0; level < 3; ++level) {
            uint bits = level < 2 ? 11u : 10u;
            shift -= bits;
            for (uint b = tid; b < BINS_VALUE; b += THREADS_VALUE) atomic_store_explicit(&hcnt[b], 0u, memory_order_relaxed);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint hi_shift = shift + bits;
            for (uint i = tid; i < V; i += THREADS_VALUE) {
                float s = row[i] * invT;
                if (level == 0) m = max(m, s);
                uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                if (hi_shift >= 32 || (u >> hi_shift) == (prefix >> hi_shift)) {
                    uint b = (u >> shift) & (BINS_VALUE - 1);
                    atomic_fetch_add_explicit(&hcnt[b], 1u, memory_order_relaxed);
                }
            }
            if (level == 0) {
                m = simd_max(m);
                if (lane == 0) redf[sg] = m;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (level == 0) {
                M = redf[0];
                for (uint s = 1; s < SG_VALUE; ++s) M = max(M, redf[s]);
            }
            if (tid < 64u) {
                uint c = 0u;
                for (uint j = 0; j < 32u; ++j) c += atomic_load_explicit(&hcnt[tid * 32u + j], memory_order_relaxed);
                ccnt[tid] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                // coarse: the chunk the running count first reaches; fine: the bin inside it
                uint acc = 0u; uint ch = 0u;
                for (int c = 63; c >= 0; --c) { acc += ccnt[c]; if (acc >= need) { ch = (uint)c; break; } }
                uint rem = need - (acc - ccnt[ch]);
                uint a = 0u; uint pick = ch * 32u;
                for (int b = (int)(ch * 32u) + 31; b >= (int)(ch * 32u); --b) {
                    uint c = atomic_load_explicit(&hcnt[b], memory_order_relaxed);
                    a += c;
                    if (a >= rem) { pick = (uint)b; rem -= (a - c); break; }
                }
                chosen[0] = pick; chosen[1] = rem;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= chosen[0] << shift;
            need = chosen[1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tk = prefix;
    }
    if ((mode & 4u) != 0u) {
        // the target mass top_p * Z, Z the sum of the first level's bins (the masses of every token top-k kept,
        // in fixed point)
        ulong need = 0ul;
        uint prefix = 0u;
        uint shift = 32;
        for (uint level = 0; level < 3; ++level) {
            uint bits = level < 2 ? 11u : 10u;
            shift -= bits;
            for (uint b = tid; b < BINS_VALUE; b += THREADS_VALUE) {
                atomic_store_explicit(&hlo[b], 0u, memory_order_relaxed);
                atomic_store_explicit(&hhi[b], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint hi_shift = shift + bits;
            for (uint i = tid; i < V; i += THREADS_VALUE) {
                float s = row[i] * invT;
                uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
                if (u >= tk && (hi_shift >= 32 || (u >> hi_shift) == (prefix >> hi_shift))) {
                    uint b = (u >> shift) & (BINS_VALUE - 1);
                    ulong w = (ulong)(exp(s - M) * MASS_SCALE_VALUE);
                    if (w) {  // the tail below 2^-30 of the top carries nothing: no atomic for it
                        uint lo = (uint)(w & 0xFFFFFFFFul), hi = (uint)(w >> 32);
                        uint old = atomic_fetch_add_explicit(&hlo[b], lo, memory_order_relaxed);
                        if (old + lo < old) hi += 1u;
                        if (hi) atomic_fetch_add_explicit(&hhi[b], hi, memory_order_relaxed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 64u) {
                ulong c = 0ul;
                for (uint j = 0; j < 32u; ++j) {
                    uint b = tid * 32u + j;
                    c += ((ulong)atomic_load_explicit(&hhi[b], memory_order_relaxed) << 32)
                       | (ulong)atomic_load_explicit(&hlo[b], memory_order_relaxed);
                }
                cmass[tid] = c;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                if (level == 0) {
                    ulong zt = 0ul;
                    for (uint c = 0; c < 64u; ++c) zt += cmass[c];
                    need = max((ulong)((float)zt * top_p), 1ul);  // at least the top token
                }
                ulong acc = 0ul; uint ch = 0u;
                for (int c = 63; c >= 0; --c) { acc += cmass[c]; if (acc >= need) { ch = (uint)c; break; } }
                ulong rem = need - (acc - cmass[ch]);
                ulong a = 0ul; uint pick = ch * 32u;
                for (int b = (int)(ch * 32u) + 31; b >= (int)(ch * 32u); --b) {
                    ulong c = ((ulong)atomic_load_explicit(&hhi[b], memory_order_relaxed) << 32)
                            | (ulong)atomic_load_explicit(&hlo[b], memory_order_relaxed);
                    a += c;
                    if (a >= rem) { pick = (uint)b; rem -= (a - c); break; }
                }
                chosen[0] = pick; needl[0] = rem;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix |= chosen[0] << shift;
            need = needl[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tp = prefix;
    }
"""

_PICK_TAIL = r"""
    uint tmin = max(tk, tp);

    // the draw: argmax over the kept tokens of s + gumbel(hash(key, i)); the lowest index among equals
    ulong key = ((ulong)keys[r * 2 + 1] << 32) | (ulong)keys[r * 2];
    float best = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < V; i += THREADS_VALUE) {
        float s = row[i] * invT;
        uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        if (u < tmin) continue;
        float v = s + gumbel_of(key, i);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    float sbest = simd_max(best);
    uint cand = (best == sbest) ? bi : 0xFFFFFFFFu;
    uint sbi = simd_min(cand);
    if (lane == 0) { redf[sg] = sbest; redu[sg] = sbi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = redf[0]; uint b_i = redu[0];
        for (uint s = 1; s < SG_VALUE; ++s) if (redf[s] > b || (redf[s] == b && redu[s] < b_i)) { b = redf[s]; b_i = redu[s]; }
        out[r] = b_i;
    }
"""


_VERIFY_TAIL = r"""
    uint tmin = max(tk, tp);
    const uint C = cfg[2];
    const uint Vd = cfg[3];
    const device float* qrow = q + (size_t)r * Vd;
    ulong key = ((ulong)keys[r * 2 + 1] << 32) | (ulong)keys[r * 2];
    threadgroup float zp[MAXC_VALUE + 1];
    threadgroup float zq[MAXC_VALUE + 1];
    threadgroup uint outcome[2];
    threadgroup int kid[MAXC_VALUE];
    for (uint c = tid; c < C; c += THREADS_VALUE) kid[c] = kids[(size_t)r * C + c];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (hasq[r] == 0u) {
        // point-mass drafts: the draw from p, the child that equals it accepted
        float best = -INFINITY; uint bi = 0xFFFFFFFFu;
        for (uint i = tid; i < V; i += THREADS_VALUE) {
            float s = row[i] * invT;
            uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
            if (u < tmin) continue;
            float v = s + gumbel_of(key, i);
            if (v > best || (v == best && i < bi)) { best = v; bi = i; }
        }
        float sbest = simd_max(best);
        uint cand = (best == sbest) ? bi : 0xFFFFFFFFu;
        uint sbi = simd_min(cand);
        if (lane == 0) { redf[sg] = sbest; redu[sg] = sbi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = redf[0]; uint b_i = redu[0];
            for (uint s = 1; s < SG_VALUE; ++s) if (redf[s] > b || (redf[s] == b && redu[s] < b_i)) { b = redf[s]; b_i = redu[s]; }
            uint slot = 0u;
            for (uint c = 0; c < C; ++c) if (kid[c] >= 0 && (uint)kid[c] == b_i) { slot = c + 1; break; }
            out[r] = (slot << 24) | b_i;
        }
        return;
    }

    // the target's masses w0 = exp(s - M) over the kept tokens, and the drafter's mass over its vocabulary
    float part = 0.0f;
    for (uint i = tid; i < V; i += THREADS_VALUE) {
        float s = row[i] * invT;
        uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        if (u >= tmin) part += exp(s - M);
    }
    part = simd_sum(part);
    if (lane == 0) redf[sg] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float z = 0.0f; for (uint s = 0; s < SG_VALUE; ++s) z += redf[s]; zp[0] = z; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    part = 0.0f;
    for (uint i = tid; i < Vd; i += THREADS_VALUE) part += qrow[i];
    part = simd_sum(part);
    if (lane == 0) redf[sg] = part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) { float z = 0.0f; for (uint s = 0; s < SG_VALUE; ++s) z += redf[s]; zq[0] = z; outcome[0] = 0u; outcome[1] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the trials: child i accepted when u_i * Q_i(c) < P_i(c); a rejection leaves the residual, whose mass is
    // summed for the next trial (every token's residual recomputed from the scalars, nothing stored). A child
    // the drafter gives no mass is not a draw of its: the trials end there. A residual with no mass means
    // P_i = Q_i (the rejection was rounding), where the trial accepts.
    uint tried = 0u;
    for (uint i = 0; i < C; ++i) {
        int c = kid[i];
        if (c < 0) break;
        float qc = ((uint)c < Vd && zq[i] > 0.0f) ? min(qrow[c] / zq[i], 1.0f) : 0.0f;
        if (!(qc > 0.0f)) break;
        float wc = residual_of(row, qrow, kid, (uint)c, i, invT, M, tmin, Vd, zp, zq);
        float pc = wc / zp[i];
        float u = uniform_of(key, 0xF00000u + i);
        if (u * qc < pc) {
            if (tid == 0) { outcome[0] = i + 1; outcome[1] = (uint)c; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            break;
        }
        // rejected: the next residual's mass
        part = 0.0f;
        for (uint x = tid; x < V; x += THREADS_VALUE) part += residual_of(row, qrow, kid, x, i + 1, invT, M, tmin, Vd, zp, zq);
        part = simd_sum(part);
        if (lane == 0) redf[sg] = part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float z = 0.0f; for (uint s = 0; s < SG_VALUE; ++s) z += redf[s];
            zp[i + 1] = z;
            zq[i + 1] = zq[i] - qrow[c];
            if (!(z > 0.0f)) { outcome[0] = i + 1; outcome[1] = (uint)c; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (outcome[0] != 0u) break;
        tried = i + 1;
    }
    if (outcome[0] != 0u) {
        if (tid == 0) out[r] = (outcome[0] << 24) | outcome[1];
        return;
    }
    // none accepted: a draw from the residual after every trial
    float best = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint x = tid; x < V; x += THREADS_VALUE) {
        float w = residual_of(row, qrow, kid, x, tried, invT, M, tmin, Vd, zp, zq);
        if (w <= 0.0f) continue;
        float v = log(w) + gumbel_of(key, x);
        if (v > best || (v == best && x < bi)) { best = v; bi = x; }
    }
    float sbest = simd_max(best);
    uint cand = (best == sbest) ? bi : 0xFFFFFFFFu;
    uint sbi = simd_min(cand);
    if (lane == 0) { redf[sg] = sbest; redu[sg] = sbi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = redf[0]; uint b_i = redu[0];
        for (uint s = 1; s < SG_VALUE; ++s) if (redf[s] > b || (redf[s] == b && redu[s] < b_i)) { b = redf[s]; b_i = redu[s]; }
        out[r] = b_i;
    }
"""

_HEADER = r"""
    static inline float uniform_of(ulong key, uint i) {
        ulong h = key ^ ((ulong)i * 0x9E3779B97F4A7C15ul);
        h ^= h >> 32; h *= 0xBF58476D1CE4E5B9ul; h ^= h >> 29; h *= 0x94D049BB133111EBul; h ^= h >> 32;
        // 23 bits: the top of a 24-bit range rounds to 1.0 in float32, an infinite Gumbel that wins the row
        return ((float)(uint)(h >> 41) + 0.5f) * (1.0f / 8388608.0f);
    }
    static inline float gumbel_of(ulong key, uint i) {
        float uf = uniform_of(key, i);
        return -log(-log(uf));
    }
    // token x's residual mass after `k` rejected trials at the row: w0 = exp(s - M) over the kept tokens,
    // w_{j+1} = max(0, w_j / zp[j] - q'_j(x) / zq[j]), q'_j the drafter's mass with the tried children removed
    static inline float residual_of(const device float* row, const device float* qrow, threadgroup int* kid,
                                    uint x, uint k, float invT, float M, uint tmin, uint Vd,
                                    threadgroup float* zp, threadgroup float* zq) {
        float s = row[x] * invT;
        uint u = as_type<uint>(s); u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        float w = (u >= tmin) ? exp(s - M) : 0.0f;
        float qx = (x < Vd) ? qrow[x] : 0.0f;
        for (uint j = 0; j < k; ++j) {
            w = max(0.0f, w / zp[j] - qx / zq[j]);
            if (kid[j] >= 0 && (uint)kid[j] == x) qx = 0.0f;  // removed from the drafter's mass from here on
        }
        return w;
    }
"""
MAXC = 16  # children a row the verify kernel takes


def _fill(src: str) -> str:
    return (
        src.replace("BINS_VALUE", str(BINS))
        .replace("THREADS_VALUE", str(THREADS))
        .replace("SG_VALUE", str(THREADS // 32))
        .replace("MASS_SCALE_VALUE", f"{MASS_SCALE:.1f}f")
        .replace("MAXC_VALUE", str(MAXC))
    )


def _get() -> Any:
    global _kernel
    m = mx()
    with _lock:
        if _kernel is None:
            _kernel = m.fast.metal_kernel(
                name="btb_sample_pick",
                input_names=["x", "keys", "cfg", "fp", "dep"],
                output_names=["out"],
                header=_fill(_HEADER),
                source=_fill(_SETUP + _PICK_TAIL),
            )
    return _kernel


def _get_verify() -> Any:
    global _verify_kernel
    m = mx()
    with _lock:
        if _verify_kernel is None:
            _verify_kernel = m.fast.metal_kernel(
                name="btb_sample_verify",
                input_names=["x", "q", "kids", "hasq", "keys", "cfg", "fp", "dep"],
                output_names=["out"],
                header=_fill(_HEADER),
                source=_fill(_SETUP + _VERIFY_TAIL),
            )
    return _verify_kernel


def pick(x: Any, keys: Any, temperature: float, top_k: int, top_p: float, after: Any = None) -> Any:
    """The picked token of every row of `x` [R, V] (float32, lazy) under one noise key a row (`keys` [R] ints):
    the argmax at temperature 0, else the draw; [R] uint32, lazily. `after`: an array the pick must follow (the
    output of a kernel that wrote `x` in place), else nothing."""
    m = mx()
    dep = after if after is not None else _dep(m)
    R = int(x.shape[0])
    sampled = temperature > 0.0
    mode = (1 if sampled else 0) | (2 if sampled and top_k > 0 else 0) | (4 if sampled and top_p < 1.0 else 0)
    ck = (int(top_k), mode, float(temperature), float(top_p))
    consts = _consts.get(ck)
    if consts is None:
        consts = _consts[ck] = (
            m.array([int(top_k), mode], dtype=m.uint32),
            m.array([1.0 / temperature if sampled else 1.0, float(top_p)], dtype=m.float32),
        )
        m.eval(*consts)
    kk = [v for k in keys for v in (int(k) & 0xFFFFFFFF, (int(k) >> 32) & 0xFFFFFFFF)]
    return _get()(
        inputs=[x, m.array(kk, dtype=m.uint32), *consts, dep],
        grid=(R * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(R,)],
        output_dtypes=[m.uint32],
    )[0]


def verify(
    x: Any, q: Any, kids: Any, hasq: Any, keys: Any, temperature: float, top_k: int, top_p: float, after: Any = None
) -> Any:
    """The verify pass's outcomes: `x` [T, V] the nodes' logits, `q` [T, Vd] the drafter's distribution at each
    node (its children drawn from it), `kids` [T, C] int32 the children in draw order (-1 padding), `hasq` [T]
    (0: the row's drafts are point masses), one key a row; [T] uint32 packed (slot + 1) << 24 | token, lazily."""
    m = mx()
    T, C = int(kids.shape[0]), int(kids.shape[1])
    Vd = int(q.shape[1])
    if C > MAXC:
        raise ValueError(f"[sample] verify: {C} children a row, at most {MAXC}")
    mode = 1 | (2 if top_k > 0 else 0) | (4 if top_p < 1.0 else 0)
    kk = [v for k in keys for v in (int(k) & 0xFFFFFFFF, (int(k) >> 32) & 0xFFFFFFFF)]
    return _get_verify()(
        inputs=[
            x,
            q,
            kids,
            hasq,
            m.array(kk, dtype=m.uint32),
            m.array([int(top_k), mode, C, Vd], dtype=m.uint32),
            m.array([1.0 / temperature, float(top_p)], dtype=m.float32),
            after if after is not None else _dep(m),
        ],
        grid=(T * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(T,)],
        output_dtypes=[m.uint32],
    )[0]
