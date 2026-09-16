# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Fused Metal kernels for the dense decode layer: residual add + RMSNorm, q/k RMSNorm + rope, and silu(gate) * up.
Each replaces several dispatches with one; every path runs the same kernels, so a verify pass and the one-row
step compute alike."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_

_lock = threading.Lock()
_kernels: dict[str, Any] = {}


def _kernel(name: str, inputs: Sequence[str], outputs: Sequence[str], source: str) -> Any:
    k = _kernels.get(name)
    if k is None:
        with _lock:
            k = _kernels.get(name)
            if k is None:
                k = _kernels[name] = mx().fast.metal_kernel(
                    name=name, input_names=list(inputs), output_names=list(outputs), source=source
                )
    return k


# --- residual add + RMSNorm ------------------------------------------------------------------------------------
# h2 = h + y (the residual, in h's dtype), x = rmsnorm(h2) * w in float32 over the row, written in T. One
# threadgroup of 256 threads per row; the sum of squares reduced within each simdgroup then across the eight in
# a fixed order, so a row's result never depends on its neighbours.
_ADD_RMSNORM_SRC = """
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint H = h_shape[1];
    threadgroup float part[8];
    const device T* hr = h + (size_t)row * H;
    const device T* yr = y + (size_t)row * H;
    device T* h2r = h2 + (size_t)row * H;
    device T* xr = x + (size_t)row * H;
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
    float scale = metal::rsqrt(tot / (float)H + eps[0]);
    for (uint i = tid; i < H; i += 256) {
        float vf = static_cast<float>(h2r[i]);
        xr[i] = static_cast<T>(vf * scale * static_cast<float>(w[i]));
    }
"""


def add_rmsnorm(h: mx_.array, y: mx_.array, w: mx_.array, eps: float) -> tuple[mx_.array, mx_.array]:
    """(h + y, rmsnorm(h + y) * w) for h, y [T, H] in one launch; both in h's dtype, lazily."""
    m = mx()
    k = _kernel("btb_add_rmsnorm", ["h", "y", "w", "eps"], ["h2", "x"], _ADD_RMSNORM_SRC)
    T, H = int(h.shape[0]), int(h.shape[1])
    return k(
        inputs=[h, y, w.astype(h.dtype) if w.dtype != h.dtype else w, m.array([float(eps)], dtype=m.float32)],
        template=[("T", h.dtype)],
        grid=(256 * T, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, H), (T, H)],
        output_dtypes=[h.dtype, h.dtype],
    )


# --- q/k RMSNorm + rope --------------------------------------------------------------------------------------
# q [T, Hq, D] and k [T, Hk, D]: each head's D values RMS-normed over D (float32, the eight-simdgroup reduction as
# above) and scaled by qn / kn, then rotated on the first rd dims at pos[t] with rope_rows' arithmetic to the bit
# (the reciprocal frequency, theta = pos * inv_freq, fast cos and sin, the half-split pairs). One threadgroup of
# D threads per (row, head): thread i normalizes element i and, for i < rd/2, rotates the pair (i, i + rd/2).
_QK_NORM_ROPE_SRC = """
    uint hh = threadgroup_position_in_grid.x;   // head over q's then k's
    uint t = threadgroup_position_in_grid.y;
    uint i = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint Hq = q_shape[1];
    uint Hk = k_shape[1];
    uint D = q_shape[2];
    uint hd2 = freqs_shape[0];
    threadgroup float part[8];
    threadgroup float normed[256];
    bool isq = hh < Hq;
    const device T* src = isq ? (q + ((size_t)t * Hq + hh) * D) : (k + ((size_t)t * Hk + (hh - Hq)) * D);
    const device T* wv = isq ? qn : kn;
    device T* dst = isq ? (oq + ((size_t)t * Hq + hh) * D) : (ok + ((size_t)t * Hk + (hh - Hq)) * D);
    float v = i < D ? static_cast<float>(src[i]) : 0.0f;
    float ss = simd_sum(v * v);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    uint nsg = (D + 31) / 32;
    for (uint g = 0; g < nsg; ++g) tot += part[g];
    float scale = metal::rsqrt(tot / (float)D + eps[0]);
    if (i < D) normed[i] = static_cast<float>(static_cast<T>(v * scale * static_cast<float>(wv[i])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= D) return;
    if (i < hd2) {
        float inv_freq = 1.0f / freqs[i];
        float L = 1.0f * static_cast<float>(pos[t]);
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
    """RMSNorm over each head of q [T, Hq, D] and k [T, Hk, D] (weights qn, kn [D]) then rope on the first `rd`
    dims at pos[t], in one launch; the rotation is rope_rows' to the bit. Returns (q, k) in q's dtype, lazily."""
    m = mx()
    kern = _kernel("btb_qk_norm_rope", ["q", "k", "qn", "kn", "eps", "freqs", "pos"], ["oq", "ok"], _QK_NORM_ROPE_SRC)
    T, Hq, D = (int(s) for s in q.shape)
    Hk = int(k.shape[1])
    assert D <= 256 and rd <= D
    p = pos if isinstance(pos, m.array) else m.array([int(v) for v in pos], dtype=m.uint32)
    dt = q.dtype
    oq, ok = kern(
        inputs=[
            q,
            k,
            qn.astype(dt) if qn.dtype != dt else qn,
            kn.astype(dt) if kn.dtype != dt else kn,
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


# --- silu(gate) * up -------------------------------------------------------------------------------------------
_SILU_MUL_SRC = """
    // mid[b, c] = silu(gu[b, c]) * gu[b, cols + c], each product rounded to T as the separate ops round it
    uint i = thread_position_in_grid.x;
    uint cols = gu_shape[1] / 2;
    uint n = gu_shape[0] * cols;
    if (i >= n) return;
    uint b = i / cols, c = i - b * cols;
    const device T* r = gu + (size_t)b * 2 * cols;
    float g = static_cast<float>(r[c]);
    float u = static_cast<float>(r[cols + c]);
    float s = static_cast<float>(static_cast<T>(1.0f / (1.0f + metal::exp(-g))));
    float si = static_cast<float>(static_cast<T>(g * s));
    out[i] = static_cast<T>(si * u);
"""


def silu_mul(gu: mx_.array) -> mx_.array:
    """silu(gu[:, :cols]) * gu[:, cols:] in gu's dtype (bf16 or float32), one launch, lazily."""
    b, two = int(gu.shape[0]), int(gu.shape[1])
    cols = two // 2
    k = _kernel("btb_silu_mul", ["gu"], ["out"], _SILU_MUL_SRC)
    return k(
        inputs=[gu],
        template=[("T", gu.dtype)],
        grid=(((b * cols + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(b, cols)],
        output_dtypes=[gu.dtype],
    )[0]
