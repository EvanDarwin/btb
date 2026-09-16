# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX backend an engine on the MLX device holds: its weights as views over shared buffers, the matmul
route (the kernel up to 16 rows, the GEMM past), rope, the fused views, and the accounting of what MLX holds."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..kinds import Log
from .core import bf16_weight, from_mx, info, mx, to_mx
from .gemv import GEMV_ROWS, gemv, matmul_mxfp4, matmul_mxfp4_pair, unpack_bf16

if TYPE_CHECKING:
    import mlx.core as mx_
    import torch

    from .core import Shared


class Weight:
    """A linear's weight for the GPU: `get()` is the bf16 [rows, cols] array. Three kinds: a resident copy, a
    view of a shared byte slot, or a packed record over a slot that unpacks on the GPU once per pass."""

    packed: Any
    shape: Any

    # a slot weight's place in its shared buffer (set by `weight_slot`; `weight_fused` reads them)
    sh: Any
    off: int
    nb: int
    a: Any

    quant: tuple[Any, Any, Any, int, int] | None  # (wq, scales, biases, bits, group): affine, multiplied as stored

    def __init__(
        self,
        a: Any = None,
        packed: tuple[Any, ...] | None = None,
        shape: Sequence[int] | None = None,
        quant: tuple[Any, Any, Any, int, int] | None = None,
    ) -> None:
        # every array made here is evaluated at once: a lazy node made on one thread cannot be evaluated by
        # another (MLX's default stream is per thread), and the server runs each request on its own thread
        m = mx()
        if a is not None:
            m.eval(a)
        if packed is not None:
            m.eval(*[x for x in packed[:5] if x is not None])
        if quant is not None:
            m.eval(*quant[:3])
        self.a = a
        self.packed = packed
        self.quant = quant
        self.shape = tuple(shape) if shape is not None else tuple(int(s) for s in a.shape)

    def get(self) -> mx_.array:
        if self.a is None:
            if self.quant is not None:
                wq, sc, bi, bits, group = self.quant
                self.a = mx().dequantize(wq, sc, bi, group_size=group, bits=bits).astype(mx().bfloat16)
            else:
                lo, hi4, tbl, esc_idx, esc_val, n = self.packed
                self.a = unpack_bf16(lo, hi4, tbl, n, self.shape, esc_idx, esc_val)
            mx().eval(self.a)  # made on whichever thread asked first; a lazy node cannot cross threads
        return self.a

    def invalidate(self) -> None:
        """The slot under a streamed weight was rewritten: drop the unpacked copy of a packed one."""
        if self.packed is not None or self.quant is not None:
            self.a = None

    def drop(self) -> None:
        self.invalidate()


class Backend:
    gemm_rows: Any
    info: Any
    mxfp4: bool
    sm: Any
    stat: Any

    def __init__(
        self,
        sm: Any,
        gemm_rows: int = 64,
        cache_limit_gb: float = 2.0,
        wire: bool = True,
        log: Log | None = None,
    ) -> None:
        m = mx()
        self.sm = sm
        self.gemm_rows = int(gemm_rows)
        self.info = info()
        self.stat = {"linears": 0, "evals": 0, "s": 0.0, "layers": 0, "layer_s": 0.0}
        # the MXFP4 expert path (`experts_mx`) is on: the engine's expert bank sends gpt-oss's experts here
        # from the store's shared slots
        self.mxfp4 = True
        with contextlib.suppress(Exception):
            m.set_cache_limit(int(cache_limit_gb * 2**30))
        if wire:
            with contextlib.suppress(Exception):
                m.set_wired_limit(int(self.info.get("max_recommended_working_set_size", 0)))
        if log:
            log(
                f"[mlx] {self.info.get('device_name')}: unified memory {self.info.get('memory_size', 0) / 2**30:.0f} GB, "
                f"recommended working set {self.info.get('max_recommended_working_set_size', 0) / 2**30:.1f} GB"
            )

    # -- memory --
    @staticmethod
    def active_bytes() -> int:
        return int(mx().get_active_memory())

    @staticmethod
    def held_bytes() -> int:
        """everything MLX holds: the arrays alive and the freed buffers it keeps cached - an exact count,
        whether or not the pages behind them have been touched yet"""
        m = mx()
        return int(m.get_active_memory()) + int(m.get_cache_memory())

    @staticmethod
    def peak_bytes() -> int:
        return int(mx().get_peak_memory())

    @staticmethod
    def clear_cache() -> None:
        mx().clear_cache()

    # -- weights --
    def weight(self, t: torch.Tensor) -> Weight:
        return Weight(bf16_weight(t))

    def weight_slot(self, shared: Shared, off: int, nb: int, shape: Sequence[int]) -> Weight:
        w = Weight(shared.view_mx(off, nb, mx().bfloat16, shape))
        w.sh, w.off, w.nb = shared, int(off), int(nb)
        return w

    def weight_fused(self, ws: Sequence[Any]) -> Weight | None:
        """One weight over adjacent slot weights with the same columns (one matmul instead of several), None when they
        are not adjacent. A view, no copy."""
        if not ws or any(getattr(w, "sh", None) is None for w in ws):
            return None
        sh = ws[0].sh
        cols = ws[0].shape[1]
        off = ws[0].off
        for w in ws:
            if w.sh is not sh or w.shape[1] != cols or w.off != off:
                return None
            off += w.nb
        nb = off - ws[0].off
        f = Weight(sh.view_mx(ws[0].off, nb, mx().bfloat16, (sum(w.shape[0] for w in ws), cols)))
        f.sh, f.off, f.nb = sh, ws[0].off, nb
        return f

    def weight_slot_packed(self, shared: Shared, off: int, e: dict[str, Any], shape: Sequence[int]) -> Weight:
        m = mx()
        a1 = e["lo"]
        a2 = a1 + e["hi4"] + e["pad"]
        a3 = a2 + 4 * e["esc"]
        lo = shared.mx[off : off + a1]
        hi4 = shared.mx[off + a1 : off + a1 + e["hi4"]]
        esc_idx = shared.mx[off + a2 : off + a3].view(m.int32) if e["esc"] else None
        esc_val = shared.mx[off + a3 : off + a3 + e["esc"]] if e["esc"] else None
        table = m.array(np.asarray(e["table"], dtype=np.uint8))
        return Weight(packed=(lo, hi4, table, esc_idx, esc_val, int(e["n"])), shape=shape)

    # -- the linear --
    def weight_affine(self, wq: Any, scales: Any, biases: Any, bits: int, group: int, shape: Sequence[int]) -> Weight:
        """a weight the kernels multiply as stored: the affine form (`mx.quantized_matmul`), never a bf16 copy"""
        m = mx()
        q = (m.array(wq), m.array(scales), m.array(biases), int(bits), int(group))
        return Weight(quant=q, shape=shape)

    def matmul(self, x: mx_.array, w: Weight) -> mx_.array:
        """x [b, cols] MLX (bf16 or float32), w a `Weight`: y [b, rows] in x's dtype, lazily."""
        m = mx()
        if w.quant is not None:
            wq, sc, bi, bits, group = w.quant
            y = m.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=group, bits=bits)
            return y if y.dtype == x.dtype else y.astype(x.dtype)
        W = w.get()
        # up to 16 rows: the engine's kernel, one weight read for the tile and batch-invariant rows (a verify
        # pass computes each row as the one-row step does); past that (prefill) MLX's gemm
        if int(x.shape[0]) <= GEMV_ROWS and int(W.shape[1]) % 8 == 0 and x.dtype in (m.float32, m.bfloat16):
            return gemv(W, x)
        if x.dtype == m.float32:
            return m.matmul(x, W.astype(m.float32).T)
        return m.matmul(x, W.T)

    def linear(self, x: torch.Tensor, w: Weight) -> torch.Tensor:
        """torch in, torch out: the module boundary of `_HostLinear`."""
        m = mx()
        t0 = time.perf_counter()
        shp = x.shape
        cols = int(shp[-1])
        x2 = x.reshape(-1, cols)
        y = self.matmul(to_mx(x2), w)
        m.eval(y)
        out = from_mx(y).view(*shp[:-1], int(y.shape[-1]))
        st = self.stat
        st["linears"] += 1
        st["evals"] += 1
        st["s"] += time.perf_counter() - t0
        return out

    def act(self, name: str) -> Callable[[mx_.array], mx_.array] | None:
        m = mx()
        if name in ("silu", "swish"):
            return lambda a: a * m.sigmoid(a)
        if name == "gelu":
            return lambda a: 0.5 * a * (1.0 + m.erf(a / 1.4142135623730951))
        if name in ("gelu_pytorch_tanh", "gelu_new"):
            return lambda a: 0.5 * a * (1.0 + m.tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)))
        if name == "relu":
            return lambda a: m.maximum(a, 0)
        return None

    def experts(self, x: torch.Tensor, hits: Any, act: Any) -> Any:
        """The mixture-of-experts step for one call: `hits` a list of (token_idx, top_k_weights, (gu, dn)) per active
        expert in expert order; returns the summed [T, H] as a torch tensor. One graph, one eval."""
        m = mx()
        t0 = time.perf_counter()
        xm = to_mx(x)
        T = int(xm.shape[0])
        H = int(xm.shape[1])
        final = m.zeros((T, H), dtype=xm.dtype)
        for token_idx, wts, (gu, dn) in hits:
            whole = T == 1 and len(token_idx) == 1
            idx = None if whole else m.array(np.asarray(token_idx, dtype=np.int32))
            cur = xm if whole else m.take(xm, idx, axis=0)
            gate, up = m.split(self.matmul(cur, gu), 2, axis=-1)
            h = act(gate) * up
            y = self.matmul(h, dn) * m.array(np.asarray(wts, dtype=np.float32)).astype(xm.dtype)[:, None]
            final = final + y if whole else final.at[idx].add(y)
        m.eval(final)
        st = self.stat
        st["evals"] += 1
        st["s"] += time.perf_counter() - t0
        return from_mx(final)

    def expert_mx(
        self,
        xm: mx_.array,
        token_idx: Any,
        wts: Any,
        pair: Any,
        e: int,
        gu_shape: Sequence[int],
        dn_shape: Sequence[int],
        gu_bias: Any,
        dn_bias: Any,
        alpha: float,
        limit: float,
        ggml: bool = False,
    ) -> Any:
        """One MXFP4 expert over its rows as its own graph (`async_eval`): `xm` [T, H], `token_idx`/`wts` its rows and
        router weights, `pair` = (gu, dn) uint8 slot views with `gu_shape`/`dn_shape` (`ggml`: a GGUF's, gu the
        (gate, up) views in ggml's layout), the bias tables, `alpha`/`limit` the gate's. Up to 16 rows through
        the batch-invariant matvec, more through a gemm. Returns (idx, y): the rows' indices (None for the whole
        input) and the weighted output in x's dtype."""
        m = mx()
        gu, dn = pair
        T = int(xm.shape[0])
        whole = T == 1 and len(token_idx) == 1
        idx = None if whole else m.array(np.asarray(token_idx, dtype=np.int32))
        cur = xm if whole else m.take(xm, idx, axis=0)
        if ggml:
            y = matmul_mxfp4_pair(gu[0], gu[1], gu_shape[0] // 2, gu_shape[1], cur) + gu_bias[e].astype(xm.dtype)
        else:
            y = matmul_mxfp4(gu, gu_shape[0], gu_shape[1], cur) + gu_bias[e].astype(xm.dtype)
        gate = m.minimum(y[..., 0::2], limit)
        up = m.clip(y[..., 1::2], -limit, limit)
        h = ((up + 1) * (gate * m.sigmoid(alpha * gate))).astype(xm.dtype)
        o = matmul_mxfp4(dn, dn_shape[0], dn_shape[1], h, ggml=ggml) + dn_bias[e].astype(xm.dtype)
        y = o * m.array(np.asarray(wts, dtype=np.float32)).astype(xm.dtype)[:, None]
        m.async_eval(y)
        return idx, y

    def experts_mx_sum(self, xm: mx_.array, parts: Any) -> Any:
        """The experts' outputs (in expert order) summed into [T, H] one add at a time, the same order alone or in a
        tile. Returns a torch tensor; one eval."""
        m = mx()
        t0 = time.perf_counter()
        final = m.zeros(xm.shape, dtype=xm.dtype)
        for idx, y in parts:
            final = final + y if idx is None else final.at[idx].add(y)
        m.eval(final)
        st = self.stat
        st["evals"] += 1
        st["s"] += time.perf_counter() - t0
        return from_mx(final)

    @staticmethod
    def rope_fast(x: mx_.array, rd: int, freqs: mx_.array, scaling: float, offset: int) -> mx_.array:
        """transformers' RoPE for positions offset, offset+1, ... as one kernel: x [..., heads, T, d] rotated on its
        first rd dims with the scaling on the rotated part; the table path to float32 rounding."""
        m = mx()
        y = m.fast.rope(x, rd, traditional=False, base=None, scale=1.0, offset=int(offset), freqs=freqs)
        if scaling != 1.0:
            if rd == int(x.shape[-1]):
                y = y * scaling
            else:
                y = m.concatenate([y[..., :rd] * scaling, y[..., rd:]], axis=-1)
        return y

    @staticmethod
    def rope(x: mx_.array, cos: mx_.array, sin: mx_.array) -> mx_.array:
        """transformers' rotate-half RoPE on x [T, heads, d] with cos/sin [T, rd] (rd <= d: the tail passes)."""
        m = mx()
        d = int(x.shape[-1])
        rd = int(cos.shape[-1])
        xr = x if rd == d else x[..., :rd]
        half = rd // 2
        rot = m.concatenate([-xr[..., half:], xr[..., :half]], axis=-1)
        out = xr * m.expand_dims(cos, -2) + rot * m.expand_dims(sin, -2)
        return out if rd == d else m.concatenate([out, x[..., rd:]], axis=-1)

    def close(self) -> None:
        # every queued kernel finishes before the buffers it reads are released
        with contextlib.suppress(Exception):
            mx().synchronize()
        with contextlib.suppress(Exception):
            mx().clear_cache()
