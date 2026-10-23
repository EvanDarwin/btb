# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Fused Metal kernels for the dense decode layer: residual add + RMSNorm, q/k RMSNorm + rope, and silu(gate) * up.
Each replaces several dispatches with one; every path runs the same kernels, so a verify pass and the one-row
step compute alike. Their math is the row functions of `FUNCS`, which the megakernel's ops call too."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .core import MetalKernel, mx

if TYPE_CHECKING:
    import mlx.core as mx_

_lock = threading.Lock()
_kernels: dict[str, MetalKernel] = {}

# The row functions: a launch below reads its row or head from its grid position and calls one, and so does the
# megakernel's matching op (btb/mlx/mega), so the two compute the same bits from one source. A function's pointer
# types are template parameters where MLX may bind a small input in the constant address space. A launch's header is
# its own function alone: MLX copies a kernel's source on every call, and the four together measured ~0.5 us a launch.
ADD_RMSNORM_FN = r"""
// h2 = h + y (the residual, in T), x = rmsnorm(h2) * w in float32 over the row, written in T. 256 threads a row; the
// sum of squares reduced within each simdgroup, then across the eight in a fixed order, so a row's result never
// depends on its neighbours
template <typename T, typename W>
inline void btb_add_rmsnorm_row(const device T* hr, const device T* yr, device T* h2r, device T* xr, W w, uint H,
                                float eps, uint tid, uint lane, uint sg, threadgroup float* part) {
    float ss = 0.0f;
    for (uint i = tid; i < H; i += 256) {
        float v = static_cast<float>(hr[i]) + static_cast<float>(yr[i]);
        T vt = static_cast<T>(v);
        h2r[i] = vt;
        float vf = static_cast<float>(vt);
        ss = fma(vf, vf, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8; ++g) tot += part[g];
    float scale = metal::rsqrt(tot / (float)H + eps);
    for (uint i = tid; i < H; i += 256) {
        float vf = static_cast<float>(h2r[i]);
        xr[i] = static_cast<T>(vf * scale * static_cast<float>(w[i]));
    }
}
"""

SANDWICH_ADD_FN = r"""
// out = h + rmsnorm(y) * w (Gemma's sandwich residual, the mirror of the above, which norms the sum): y RMS-normed over
// the row in float32, scaled by w, cast to T and added to the residual h. Same reduction shape
template <typename T, typename W>
inline void btb_sandwich_add_row(const device T* hr, const device T* yr, device T* outr, W w, uint H, float eps,
                                 uint tid, uint lane, uint sg, threadgroup float* part) {
    float ss = 0.0f;
    for (uint i = tid; i < H; i += 256) {
        float v = static_cast<float>(yr[i]);
        ss = fma(v, v, ss);
    }
    ss = simd_sum(ss);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint g = 0; g < 8; ++g) tot += part[g];
    float scale = metal::rsqrt(tot / (float)H + eps);
    for (uint i = tid; i < H; i += 256) {
        float normed = static_cast<float>(yr[i]) * scale * static_cast<float>(w[i]);
        outr[i] = static_cast<T>(static_cast<float>(hr[i]) + static_cast<float>(static_cast<T>(normed)));
    }
}
"""

QK_NORM_ROPE_FN = r"""
// one head's D values (thread i takes element i) RMS-normed over D in float32 (the eight-simdgroup reduction above),
// scaled by wn, then rotated on the first 2 * hd2 dims at position `pos` with rope_rows' arithmetic to the bit: the
// reciprocal frequency, theta = pos * inv_freq, fast cos and sin, the half-split pairs (i, i + hd2); the dims past the
// rotary pass through normed. NORM false rotates the head as it is (rope_rows alone: a family with no q/k norms)
template <bool NORM, typename T, typename W, typename F>
inline void btb_qk_norm_rope_head(const device T* src, device T* dst, W wn, uint D, uint hd2, float eps, F freqs,
                                  uint pos, uint i, uint lane, uint sg, threadgroup float* part,
                                  threadgroup float* normed) {
    float v = i < D ? static_cast<float>(src[i]) : 0.0f;
    if (NORM) {
        float wv = i < D ? static_cast<float>(wn[i]) : 0.0f;
        float ss = simd_sum(v * v);
        if (lane == 0) part[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        uint nsg = (D + 31) / 32;
        for (uint g = 0; g < nsg; ++g) tot += part[g];
        float scale = metal::rsqrt(tot / (float)D + eps);
        if (i < D) normed[i] = static_cast<float>(static_cast<T>(v * scale * wv));
    } else if (i < D) {
        normed[i] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= D) return;
    if (i < hd2) {
        float inv_freq = 1.0f / freqs[i];
        float L = 1.0f * static_cast<float>(pos);
        float theta = L * inv_freq;
        float costheta = metal::fast::cos(theta);
        float sintheta = metal::fast::sin(theta);
        float x1 = normed[i];
        float x2 = normed[i + hd2];
        dst[i] = static_cast<T>(x1 * costheta - x2 * sintheta);
        dst[i + hd2] = static_cast<T>(x1 * sintheta + x2 * costheta);
    } else if (i >= 2 * hd2) {
        dst[i] = static_cast<T>(normed[i]);
    }
}
"""

SILU_MUL_FN = r"""
// silu(g) * u, each product rounded to T as the separate ops round it
template <typename T>
inline T btb_silu_mul(float g, float u) {
    float s = static_cast<float>(static_cast<T>(1.0f / (1.0f + metal::exp(-g))));
    float si = static_cast<float>(static_cast<T>(g * s));
    return static_cast<T>(si * u);
}
"""

FUNCS = ADD_RMSNORM_FN + SANDWICH_ADD_FN + QK_NORM_ROPE_FN + SILU_MUL_FN


def _kernel(name: str, inputs: Sequence[str], outputs: Sequence[str], header: str, source: str) -> MetalKernel:
    k = _kernels.get(name)
    if k is None:
        with _lock:
            k = _kernels.get(name)
            if k is None:
                k = _kernels[name] = mx().fast.metal_kernel(
                    name=name, input_names=list(inputs), output_names=list(outputs), header=header, source=source
                )
    return k


# --- residual add + RMSNorm: a threadgroup of 256 a row ---------------------------------------------------------
_ADD_RMSNORM_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint H = h_shape[1];
    threadgroup float part[8];
    size_t o = (size_t)row * H;
    btb_add_rmsnorm_row(h + o, y + o, h2 + o, x + o, w, H, eps[0], thread_position_in_threadgroup.x,
                        thread_index_in_simdgroup, simdgroup_index_in_threadgroup, part);
"""


def add_rmsnorm(h: mx_.array, y: mx_.array, w: mx_.array, eps: float) -> tuple[mx_.array, mx_.array]:
    """(h + y, rmsnorm(h + y) * w) for h, y [T, H] in one launch; both in h's dtype, lazily."""
    m = mx()
    k = _kernel("btb_add_rmsnorm", ["h", "y", "w", "eps"], ["h2", "x"], ADD_RMSNORM_FN, _ADD_RMSNORM_SRC)
    T, H = int(h.shape[0]), int(h.shape[1])
    h2, x = k(
        inputs=[h, y, w, m.array([float(eps)], dtype=m.float32)],
        template=[("T", h.dtype)],
        grid=(256 * T, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, H), (T, H)],
        output_dtypes=[h.dtype, h.dtype],
    )
    return h2, x


# --- RMSNorm the sub-block output, then residual add (Gemma's sandwich norm): a threadgroup of 256 a row ----------
_SANDWICH_ADD_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint H = h_shape[1];
    threadgroup float part[8];
    size_t o = (size_t)row * H;
    btb_sandwich_add_row(h + o, y + o, out + o, w, H, eps[0], thread_position_in_threadgroup.x,
                         thread_index_in_simdgroup, simdgroup_index_in_threadgroup, part);
"""


def sandwich_add(h: mx_.array, y: mx_.array, w: mx_.array, eps: float) -> mx_.array:
    """h + rmsnorm(y) * w for h, y [T, H] in one launch, in h's dtype — Gemma's sandwich residual, the delta
    normed before it is added rather than after. `w` is read as stored (float32 for a 1 + weight family, the
    product the module takes)."""
    m = mx()
    k = _kernel("btb_sandwich_add", ["h", "y", "w", "eps"], ["out"], SANDWICH_ADD_FN, _SANDWICH_ADD_SRC)
    T, H = int(h.shape[0]), int(h.shape[1])
    return k(
        inputs=[h, y, w, m.array([float(eps)], dtype=m.float32)],
        template=[("T", h.dtype)],
        grid=(256 * T, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, H)],
        output_dtypes=[h.dtype],
    )[0]


# --- q/k RMSNorm + rope: a threadgroup of D threads a (row, head), q's heads then k's ----------------------------
_QK_NORM_ROPE_SRC = """
    uint hh = threadgroup_position_in_grid.x;
    uint t = threadgroup_position_in_grid.y;
    uint Hq = q_shape[1];
    uint Hk = k_shape[1];
    uint D = q_shape[2];
    threadgroup float part[8];
    threadgroup float normed[256];
    bool isq = hh < Hq;
    const device T* src = isq ? (q + ((size_t)t * Hq + hh) * D) : (k + ((size_t)t * Hk + (hh - Hq)) * D);
    device T* dst = isq ? (oq + ((size_t)t * Hq + hh) * D) : (ok + ((size_t)t * Hk + (hh - Hq)) * D);
    btb_qk_norm_rope_head<true>(src, dst, isq ? qn : kn, D, (uint)freqs_shape[0], eps[0], freqs, pos[t],
                          thread_position_in_threadgroup.x, thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
                          part, normed);
"""


def qk_norm_rope(
    q: mx_.array,
    k: mx_.array,
    qn: mx_.array,
    kn: mx_.array,
    eps: float,
    rd: int,
    freqs: mx_.array,
    scaling: float,
    pos: Sequence[int] | mx_.array,
) -> tuple[mx_.array, mx_.array]:
    """RMSNorm over each head of q [T, Hq, D] and k [T, Hk, D] (weights qn, kn [D], read as stored) then rope on
    the first `rd` dims at pos[t], in one launch; the rotation is rope_rows' to the bit. Returns (q, k) in q's
    dtype, lazily."""
    m = mx()
    kern = _kernel(
        "btb_qk_norm_rope",
        ["q", "k", "qn", "kn", "eps", "freqs", "pos"],
        ["oq", "ok"],
        QK_NORM_ROPE_FN,
        _QK_NORM_ROPE_SRC,
    )
    T, Hq, D = (int(s) for s in q.shape)
    Hk = int(k.shape[1])
    assert D <= 256 and rd <= D
    p = pos if isinstance(pos, m.array) else m.array([int(v) for v in pos], dtype=m.uint32)
    dt = q.dtype
    oq, ok = kern(
        inputs=[
            q,
            k,
            qn,
            kn,
            m.array([float(eps)], dtype=m.float32),
            freqs,
            p,
        ],
        template=[("T", dt)],
        grid=(D * (Hq + Hk), T, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(T, Hq, D), (T, Hk, D)],
        output_dtypes=[dt, dt],
    )
    if scaling != 1.0:
        oq, ok = oq * scaling, ok * scaling
    return oq, ok


# --- silu(gate) * up: a thread an output -------------------------------------------------------------------------
_SILU_MUL_SRC = """
    // mid[b, c] = silu(gu[b, c]) * gu[b, cols + c]
    uint i = thread_position_in_grid.x;
    uint cols = gu_shape[1] / 2;
    uint n = gu_shape[0] * cols;
    if (i >= n) return;
    uint b = i / cols, c = i - b * cols;
    const device T* r = gu + (size_t)b * 2 * cols;
    out[i] = btb_silu_mul<T>(static_cast<float>(r[c]), static_cast<float>(r[cols + c]));
"""


def silu_mul(gu: mx_.array) -> mx_.array:
    """silu(gu[:, :cols]) * gu[:, cols:] in gu's dtype (bf16 or float32), one launch, lazily."""
    b, two = int(gu.shape[0]), int(gu.shape[1])
    cols = two // 2
    k = _kernel("btb_silu_mul", ["gu"], ["out"], SILU_MUL_FN, _SILU_MUL_SRC)
    return k(
        inputs=[gu],
        template=[("T", gu.dtype)],
        grid=(((b * cols + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, cols)],
        output_dtypes=[gu.dtype],
    )[0]
