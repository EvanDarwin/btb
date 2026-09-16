# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The gated DeltaNet on Metal: the compiled one-position step, the recurrent kernel over windows of positions,
the tree variant (every node's conv reads its own ancestry), and the prefill over a whole prompt."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..kinds import Parents
from .core import mx
from .gemv import _gemv_lock

if TYPE_CHECKING:
    import mlx.core as mx_


def delta_step_fn(hk: int, hv: int, dk: int, dv: int, key_dim: int, eps: float, has_bias: bool) -> Callable[..., Any]:
    """A compiled function for one DeltaNet position (conv update, l2-normed q/k, the gated delta rule on the
    [Hv, dk, dv] state, the gated RMSNorm; float32). Returns (core [Hv*dv], conv_new [C, K], state_new)."""
    m = mx()
    scale = float(dk) ** -0.5

    def step(
        mixed: Any,
        z: Any,
        a: Any,
        b: Any,
        conv: Any,
        state: Any,
        conv_w: Any,
        conv_b: Any,
        a_log: Any,
        dt_bias: Any,
        norm_w: Any,
    ) -> Any:
        conv_new = m.concatenate([conv[:, 1:], mixed[:, None]], axis=1)
        xc = (conv_new * conv_w).sum(axis=-1)
        if has_bias:
            xc = xc + conv_b
        xc = xc * m.sigmoid(xc)
        q = xc[:key_dim].reshape(hk, dk)
        k = xc[key_dim : 2 * key_dim].reshape(hk, dk)
        v = xc[2 * key_dim :].reshape(hv, dv)
        q = q * m.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6) * scale
        k = k * m.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
        if hv > hk:
            q = m.repeat(q, hv // hk, axis=0)
            k = m.repeat(k, hv // hk, axis=0)
        beta = m.sigmoid(b)
        g = -m.exp(a_log) * m.logaddexp(a + dt_bias, 0.0)
        S = state * m.exp(g)[:, None, None]
        kv = (S * k[:, :, None]).sum(axis=1)
        delta = (v - kv) * beta[:, None]
        S = S + k[:, :, None] * delta[:, None, :]
        o = (S * q[:, :, None]).sum(axis=1)
        core = norm_w * (o * m.rsqrt((o * o).mean(axis=-1, keepdims=True) + eps))
        zz = z.reshape(hv, dv)
        core = core * (zz * m.sigmoid(zz))
        return core.reshape(hv * dv), conv_new, S

    return m.compile(step)


_chunk_body_fn = None


def _chunk_body() -> Callable[..., Any]:
    """The per-chunk step of the chunked rule, compiled once (its shapes repeat across chunks and layers)."""
    global _chunk_body_fn
    if _chunk_body_fn is None:
        m = mx()

        def body(q_i: Any, k_i: Any, v_i: Any, g_i: Any, decay_i: Any, kcd_i: Any, state: Any) -> Any:
            attn_i = (q_i @ k_i.transpose(0, 2, 1)) * decay_i
            v_new = v_i - kcd_i @ state
            out_i = (q_i * m.exp(g_i)[..., None]) @ state + attn_i @ v_new
            state = (
                state * m.exp(g_i[:, -1])[:, None, None]
                + (k_i * m.exp(g_i[:, -1, None] - g_i)[..., None]).transpose(0, 2, 1) @ v_new
            )
            return out_i, state

        _chunk_body_fn = m.compile(body)
    return _chunk_body_fn


_DELTA_SRC = """
    // the gated delta rule over T positions of one head, one dispatch: thread (lane, j, h) keeps the state
    // column S[k, j] for k = lane + 32 i in registers and walks the positions; the two contractions over k
    // (S^T k, S^T q) are simd sums across the 32 lanes. q, k [T, H, DK] (normed, q scaled), v [T, H, DV],
    // g [T, H] (log decay), beta [T, H], state [H, DK, DV] -> out [T, H, DV], state_out [H, DK, DV].
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    uint h = thread_position_in_grid.z;
    uint T = v_shape[0];
    uint H = v_shape[1];
    uint HK = q_shape[1];
    uint hk = h / (H / HK);
    const float qscale = 1.0f / metal::precise::sqrt((float)DK);
    // lane owns the contiguous slice [lane * NPT, lane * NPT + NPT) of the key dim: its q and k load as one
    // vector each (a threadgroup-shared load with barriers measured slower). q and k arrive raw, per kv head:
    // the l2 norms are simd sums here, the same arithmetic for every position however many there are
    // `order` picks the positions to run (a root-to-node path of a tree pass, or 0..T-1; -1 ends it); INPLACE
    // writes the new state over `state` itself (each thread its own slice, read at the start): the cache's
    // buffer, no copy
    uint NO = order_shape[0];
    float S[NPT];
    for (uint i = 0; i < NPT; ++i) S[i] = state[((size_t)h * DK + lane * NPT + i) * DV + j];
    for (uint tt = 0; tt < NO; ++tt) {
        int t_ = order[tt];
        if (t_ < 0) break;
        uint t = (uint)t_;
        size_t th = (size_t)t * H + h;
        size_t tk = (size_t)t * HK + hk;
        float gt = exp(g[th]);
        float bt = beta[th];
        float vt = v[th * DV + j];
        const device float* qt = q + tk * DK + lane * NPT;
        const device float* kt = k + tk * DK + lane * NPT;
        float kk[NPT];
        float qq[NPT];
        if (NPT == 4) {
            float4 k4 = *((const device float4*)kt);
            float4 q4 = *((const device float4*)qt);
            kk[0] = k4.x; kk[1] = k4.y; kk[2] = k4.z; kk[3] = k4.w;
            qq[0] = q4.x; qq[1] = q4.y; qq[2] = q4.z; qq[3] = q4.w;
        } else {
            for (uint i = 0; i < NPT; ++i) { kk[i] = kt[i]; qq[i] = qt[i]; }
        }
        float ksq = 0.0f, qsq = 0.0f;
        for (uint i = 0; i < NPT; ++i) { ksq = fma(kk[i], kk[i], ksq); qsq = fma(qq[i], qq[i], qsq); }
        float kn = metal::precise::rsqrt(simd_sum(ksq) + 1e-6f);
        float qn = metal::precise::rsqrt(simd_sum(qsq) + 1e-6f) * qscale;
        for (uint i = 0; i < NPT; ++i) { kk[i] *= kn; qq[i] *= qn; }
        float kv = 0.0f;
        for (uint i = 0; i < NPT; ++i) {
            S[i] *= gt;
            kv = fma(S[i], kk[i], kv);
        }
        kv = simd_sum(kv);
        float delta = (vt - kv) * bt;
        float o = 0.0f;
        for (uint i = 0; i < NPT; ++i) {
            S[i] = fma(kk[i], delta, S[i]);
            o = fma(S[i], qq[i], o);
        }
        o = simd_sum(o);
        if (lane == 0) out[(size_t)tt * H * DV + h * DV + j] = o;
    }
#if INPLACE
    device float* dst = (device float*)state;
    for (uint i = 0; i < NPT; ++i) dst[((size_t)h * DK + lane * NPT + i) * DV + j] = S[i];
#else
    for (uint i = 0; i < NPT; ++i) state_out[((size_t)h * DK + lane * NPT + i) * DV + j] = S[i];
#endif
"""


_delta_kernels: dict[int, Any] = {}


def _delta_cols(dv: int) -> int:
    """threads per threadgroup along dv: 32 columns make a lane's state elements one 128-byte line"""
    return 32 if dv % 32 == 0 else (8 if dv % 8 == 0 else 1)


def delta_recurrent(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    g: mx_.array,
    beta: mx_.array,
    state: mx_.array,
    order: Sequence[int] | None = None,
    inplace: bool = False,
) -> tuple[mx_.array, mx_.array | None]:
    """The gated delta rule over T positions as one Metal dispatch (float32): q, k [T, Hk, dk] raw, v [T, H, dv],
    g [T, H] log decay, beta [T, H], state [H, dk, dv] (a leading 1 allowed); `order` the positions to run, in
    order (all when None). Returns (out [len(order), H, dv], state_new), lazily; `inplace` writes the new state
    into `state`'s own buffer and returns None for it. dk % 32 == 0."""
    m = mx()
    key = 1 if inplace else 0
    with _gemv_lock:
        kern = _delta_kernels.get(key)
        if kern is None:
            kern = _delta_kernels[key] = m.fast.metal_kernel(
                name=f"btb_delta_recurrent{'_inplace' if inplace else ''}",
                input_names=["q", "k", "v", "g", "beta", "state", "order"],
                output_names=["out"] if inplace else ["out", "state_out"],
                header=f"#define INPLACE {key}\n",
                source=_DELTA_SRC,
            )
    T, dk = int(q.shape[0]), int(q.shape[2])
    H, dv = int(v.shape[1]), int(v.shape[-1])
    idx = list(range(T)) if order is None else [int(i) for i in order]
    outs = kern(
        inputs=[q, k, v, g, beta, state, _pad8(m.array(np.asarray(idx, dtype=np.int32)), fill=-1)],
        template=[("DK", dk), ("DV", dv), ("NPT", dk // 32)],
        grid=(32, dv, H),
        threadgroup=(32, _delta_cols(dv), 1),
        output_shapes=[(len(idx), H, dv)] + ([] if inplace else [(H, dk, dv)]),
        output_dtypes=[m.float32] + ([] if inplace else [m.float32]),
    )
    return outs[0], (None if inplace else outs[1])


_DELTA_TREE_SRC = """
    // the gated delta rule over T nodes of a tree, visited in `order` (depth-first: a node's ancestors precede it,
    // and stack slot depth - 1 holds its parent's state): the sequence kernel's arithmetic node by node, each
    // depth's state in registers, so a node costs no state traffic and the cache is not touched. Only the
    // outputs are written; the accepted path's state is recomputed at the commit (`delta_chain_state`).
    uint lane = thread_position_in_grid.x;
    uint j = thread_position_in_grid.y;
    uint h = thread_position_in_grid.z;
    uint T = v_shape[0];
    uint H = v_shape[1];
    uint HK = q_shape[1];
    uint hk = h / (H / HK);
    const float qscale = 1.0f / metal::precise::sqrt((float)DK);
    float S[MD][NPT];
    for (uint tt = 0; tt < T; ++tt) {
        uint t = order[tt];
        uint d = depth[t];
        float cur[NPT];
        if (d == 0) {
            for (uint i = 0; i < NPT; ++i) cur[i] = state[((size_t)h * DK + lane * NPT + i) * DV + j];
        } else {
            for (uint i = 0; i < NPT; ++i) cur[i] = S[d - 1][i];
        }
        size_t th = (size_t)t * H + h;
        size_t tk = (size_t)t * HK + hk;
        float gt = exp(g[th]);
        float bt = beta[th];
        float vt = v[th * DV + j];
        const device float* qt = q + tk * DK + lane * NPT;
        const device float* kt = k + tk * DK + lane * NPT;
        float kk[NPT];
        float qq[NPT];
        if (NPT == 4) {
            float4 k4 = *((const device float4*)kt);
            float4 q4 = *((const device float4*)qt);
            kk[0] = k4.x; kk[1] = k4.y; kk[2] = k4.z; kk[3] = k4.w;
            qq[0] = q4.x; qq[1] = q4.y; qq[2] = q4.z; qq[3] = q4.w;
        } else {
            for (uint i = 0; i < NPT; ++i) { kk[i] = kt[i]; qq[i] = qt[i]; }
        }
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
        if (lane == 0) out[th * DV + j] = o;
        for (uint i = 0; i < NPT; ++i) S[d][i] = cur[i];
    }
"""


_delta_tree_kernel = None


def _dfs_order(parents: Parents) -> tuple[list[int], list[int]]:
    """(a depth-first visiting order, each node's depth) for `parents` (-1 for a root, parent[t] < t)."""
    T = len(parents)
    kids: list[list[int]] = [[] for _ in range(T)]
    roots = []
    depth = [0] * T
    for t, p in enumerate(parents):
        if p < 0:
            roots.append(t)
        else:
            kids[p].append(t)
            depth[t] = depth[p] + 1
    order: list[int] = []
    stack = roots[::-1]
    while stack:
        t = stack.pop()
        order.append(t)
        stack.extend(kids[t][::-1])
    return order, depth


def delta_tree(
    q: mx_.array,
    k: mx_.array,
    v: mx_.array,
    g: mx_.array,
    beta: mx_.array,
    state: mx_.array,
    parents: Parents,
) -> mx_.array:
    """`delta_recurrent` over a tree of T nodes (`parents`: -1 for the cache's state, parent[t] < t): every node
    from its parent's state, no checkpoint written. Returns out [T, H, dv], lazily."""
    global _delta_tree_kernel
    m = mx()
    with _gemv_lock:
        if _delta_tree_kernel is None:
            _delta_tree_kernel = m.fast.metal_kernel(
                name="btb_delta_tree",
                input_names=["q", "k", "v", "g", "beta", "state", "order", "depth"],
                output_names=["out"],
                source=_DELTA_TREE_SRC,
            )
    T, dk = int(q.shape[0]), int(q.shape[2])
    H, dv = int(v.shape[1]), int(v.shape[-1])
    order, depth = _dfs_order([int(p) for p in parents])
    if T > 16:
        raise ValueError(f"[mlx] the tree recurrence takes at most 16 nodes, got {T}")
    return _delta_tree_kernel(
        inputs=[
            q,
            k,
            v,
            g,
            beta,
            state,
            _pad8(m.array(np.asarray(order, dtype=np.int32))),
            _pad8(m.array(np.asarray(depth, dtype=np.int32))),
        ],
        template=[("DK", dk), ("DV", dv), ("NPT", dk // 32), ("MD", 16)],
        grid=(32, dv, H),
        threadgroup=(32, _delta_cols(dv), 1),
        output_shapes=[(T, H, dv)],
        output_dtypes=[m.float32],
    )[0]


def _pad8(a: mx_.array, fill: int = 0) -> mx_.array:
    """an input of <= 4 elements would arrive in the constant address space; pad it into a real buffer"""
    m = mx()
    if int(a.size) > 4:
        return a
    return m.concatenate([a, m.full((8 - int(a.size),), fill, dtype=a.dtype)])


def delta_chain_state(
    heads: Sequence[mx_.array], state0: mx_.array, idx: Sequence[int] | None = None, inplace: bool = False
) -> mx_.array | None:
    """The state after the nodes `idx` (a root-to-node path, in order; every node when None) of a tree pass's
    `heads` = (q, k, v, g, beta), from `state0`: the sequence kernel over that chain, so the bits are the
    tree's at that node. Returns [Hv, dk, dv], lazily; `inplace` writes it into `state0`'s buffer and returns
    the kernel's output to evaluate instead."""
    q, k, v, g, beta = heads
    out, new = delta_recurrent(q, k, v, g, beta, state0, order=idx, inplace=inplace)
    return out if inplace else new


def _tree_conv_index(parents: Parents, T: int, K: int) -> Any:
    """for every node, the rows of concat(stored inputs [K-1], nodes [T]) its conv window reads, oldest
    first: its K-1 ancestors (the cache's newest inputs past the root) and itself"""
    idx = np.empty((T, K), dtype=np.int32)
    for p in range(T):
        cur, beyond = p, 0
        for t in range(K):
            if cur >= 0:
                idx[p, K - 1 - t] = K - 1 + cur
                cur = parents[cur]
            else:
                idx[p, K - 1 - t] = K - 2 - beyond
                beyond += 1
    return idx


def delta_tree_step(
    mixed: mx_.array,
    z: mx_.array,
    a: mx_.array,
    b: mx_.array,
    conv_prev: mx_.array,
    state0: mx_.array,
    parents: Parents,
    conv_w: mx_.array,
    conv_b: mx_.array | None,
    a_log: mx_.array,
    dt_bias: mx_.array,
    norm_w: mx_.array,
    eps: float,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    key_dim: int,
) -> Any:
    """A DeltaNet layer over T tree nodes from `conv_prev` [C, K-1] and `state0`: each node's conv reads its
    ancestry, the rule runs from its parent's state. Returns (core [T, Hv*dv], conv checkpoints [T, C, K], the
    heads (q, k, v, g, beta) for `delta_chain_state`), lazily; the cache is not touched."""
    m = mx()
    T = int(mixed.shape[0])
    K = int(conv_w.shape[1])
    par = [int(p) for p in parents] if parents is not None else list(range(-1, T - 1))
    idx = m.array(_tree_conv_index(par, T, K))
    rows = m.concatenate([conv_prev.T, mixed], axis=0)[idx]  # [T, K, C], oldest first
    xc = rows[:, 0] * conv_w[:, 0][None]
    for j in range(1, K):
        xc = xc + rows[:, j] * conv_w[:, j][None]
    if conv_b is not None:
        xc = xc + conv_b[None]
    xc = xc * m.sigmoid(xc)
    q, k, v, g, beta = _delta_heads(xc, a, b, a_log, dt_bias, hk, hv, dk, dv, key_dim, raw=True)
    state = state0 if state0 is not None else m.zeros((hv, dk, dv), dtype=m.float32)
    core = delta_tree(q, k, v, g, beta, state, par)
    return _delta_post(core, z, norm_w, eps, hv, dv), rows.transpose(0, 2, 1), (q, k, v, g, beta)


def _delta_heads(
    xc: mx_.array,
    a: mx_.array,
    b: mx_.array,
    a_log: mx_.array,
    dt_bias: mx_.array,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    key_dim: int,
    raw: bool = False,
) -> Any:
    """The conv's activated output xc [T, C] split into q, k, v with g and beta [T, Hv]; `raw` leaves q and k
    un-normed [T, Hk, dk] for the recurrence kernels, else normed, scaled and repeated to [T, Hv, dk]."""
    m = mx()
    T = int(xc.shape[0])
    q = xc[:, :key_dim].reshape(T, hk, dk)
    k = xc[:, key_dim : 2 * key_dim].reshape(T, hk, dk)
    v = xc[:, 2 * key_dim :].reshape(T, hv, dv)
    if not raw:
        q = q * m.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6) * (float(dk) ** -0.5)
        k = k * m.rsqrt((k * k).sum(axis=-1, keepdims=True) + 1e-6)
        if hv > hk:
            q = m.repeat(q, hv // hk, axis=1)
            k = m.repeat(k, hv // hk, axis=1)
    beta = m.sigmoid(b)
    g = -m.exp(a_log) * m.logaddexp(a + dt_bias, 0.0)
    return q, k, v, g, beta


def _delta_prep(
    mixed: mx_.array,
    a: mx_.array,
    b: mx_.array,
    conv_prev: mx_.array,
    conv_w: mx_.array,
    conv_b: mx_.array | None,
    a_log: mx_.array,
    dt_bias: mx_.array,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    key_dim: int,
    raw: bool = False,
) -> Any:
    """The DeltaNet's front: the causal conv over the stored and new inputs, the split into q, k, v, g, beta, and
    the new conv state."""
    m = mx()
    T = int(mixed.shape[0])
    K = int(conv_w.shape[1])
    full = m.concatenate([conv_prev, mixed.T], axis=1)
    L = int(full.shape[1])
    xc = full[:, : L - K + 1] * conv_w[:, 0:1]
    for j in range(1, K):
        xc = xc + full[:, j : L - K + 1 + j] * conv_w[:, j : j + 1]
    xc = xc[:, -T:]
    if conv_b is not None:
        xc = xc + conv_b[:, None]
    xc = xc * m.sigmoid(xc)
    conv_new = full[:, -K:]
    q, k, v, g, beta = _delta_heads(xc.T, a, b, a_log, dt_bias, hk, hv, dk, dv, key_dim, raw=raw)
    return q, k, v, g, beta, conv_new


def _delta_post(core: mx_.array, z: mx_.array, norm_w: mx_.array, eps: float, hv: int, dv: int) -> mx_.array:
    """The gated RMSNorm: core [T, Hv, dv] normed per head (the fused rms_norm: one kernel, the same per row
    at any row count), scaled, gated by silu(z)."""
    m = mx()
    T = int(core.shape[0])
    core = m.fast.rms_norm(core.reshape(T * hv, dv), norm_w, float(eps)).reshape(T, hv, dv)
    zz = z.reshape(T, hv, dv)
    return (core * (zz * m.sigmoid(zz))).reshape(T, hv * dv)


def delta_prefill(
    mixed: mx_.array,
    z: mx_.array,
    a: mx_.array,
    b: mx_.array,
    conv_prev: mx_.array,
    state0: mx_.array,
    conv_w: mx_.array,
    conv_b: mx_.array | None,
    a_log: mx_.array,
    dt_bias: mx_.array,
    norm_w: mx_.array,
    eps: float,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    key_dim: int,
    mode: str = "recurrent",
    chunk: int = 64,
    window: int = 512,
    inplace: bool = False,
) -> Any:
    """A DeltaNet layer over T positions in MLX (float32) from `conv_prev` [C, K] and `state0`: `recurrent`
    the exact rule as one dispatch, `chunk` the module's chunked rule. Returns (core [T, Hv*dv], conv_new,
    state_new), lazily; `inplace` (recurrent, T <= window) writes the state into `state0` and returns None."""
    m = mx()
    recurrent = mode == "recurrent" and dk % 32 == 0
    q, k, v, g, beta, conv_new = _delta_prep(
        mixed, a, b, conv_prev, conv_w, conv_b, a_log, dt_bias, hk, hv, dk, dv, key_dim, raw=recurrent
    )
    T = int(mixed.shape[0])
    state = state0 if state0 is not None else m.zeros((hv, dk, dv), dtype=m.float32)
    if recurrent:
        # one dispatch per window of positions, the state carried: a single dispatch over a long sequence
        # goes superlinear once its q/k/v stream outgrows the cache (8K: 98 ms a layer, windowed: 36)
        W = int(window)
        if T <= W:
            core, state_out = delta_recurrent(q, k, v, g, beta, state, inplace=inplace)
            return _delta_post(core, z, norm_w, eps, hv, dv), conv_new, state_out
        outs = []
        for s0 in range(0, T, W):
            o, snew = delta_recurrent(
                q[s0 : s0 + W], k[s0 : s0 + W], v[s0 : s0 + W], g[s0 : s0 + W], beta[s0 : s0 + W], state
            )
            assert snew is not None  # a non-inplace recurrent call always returns the new state
            state = snew
            outs.append(o)
        core = m.concatenate(outs, axis=0)
        return _delta_post(core, z, norm_w, eps, hv, dv), conv_new, state
    # heads first: [H, T, d]
    q = q.transpose(1, 0, 2)
    k = k.transpose(1, 0, 2)
    v = v.transpose(1, 0, 2)
    beta = beta.T
    g = g.T
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = m.pad(q, [(0, 0), (0, pad), (0, 0)])
        k = m.pad(k, [(0, 0), (0, pad), (0, 0)])
        v = m.pad(v, [(0, 0), (0, pad), (0, 0)])
        beta = m.pad(beta, [(0, 0), (0, pad)])
        g = m.pad(g, [(0, 0), (0, pad)])
    n = (T + pad) // chunk
    q = q.reshape(hv, n, chunk, dk)
    k = k.reshape(hv, n, chunk, dk)
    v = v.reshape(hv, n, chunk, dv)
    beta = beta.reshape(hv, n, chunk)
    g = m.cumsum(g.reshape(hv, n, chunk), axis=-1)
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    tril = m.tril(m.ones((chunk, chunk), dtype=m.bool_))
    strict = m.tril(m.ones((chunk, chunk), dtype=m.bool_), k=-1)
    diff = g[..., :, None] - g[..., None, :]
    decay = m.where(tril, m.exp(m.where(tril, diff, 0.0)), 0.0)
    A = m.where(strict, -((k_beta @ k.transpose(0, 1, 3, 2)) * decay), 0.0)
    eye = m.eye(chunk, dtype=m.float32)
    # (I - A)^-1 by forward substitution on the CPU stream (a few ms per thousand chunks): the reference's
    # row loop; a product of powers of A is unstable when a chunk's keys are nearly alike
    attn = m.linalg.solve_triangular(eye - A, m.broadcast_to(eye, A.shape), upper=False, stream=m.cpu)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * m.exp(g)[..., None])
    body = _chunk_body()
    outs = []
    for i in range(n):
        out_i, state = body(q[:, i], k[:, i], value[:, i], g[:, i], decay[:, i], k_cumdecay[:, i], state)
        outs.append(out_i)
    core = m.concatenate(outs, axis=1)[:, :T].transpose(1, 0, 2)
    return _delta_post(core, z, norm_w, eps, hv, dv), conv_new, state


delta_chunk_prefill = delta_prefill
