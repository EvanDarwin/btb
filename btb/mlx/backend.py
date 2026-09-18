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
from .iquant import (
    dequant_iq4nl,
    dequant_iq4xs,
    dequant_lattice,
    matvec_iq4nl,
    matvec_iq4xs,
    matvec_lattice,
    repack_iq4nl,
    repack_lattice,
)
from .kquant import dequant_q2k, dequant_q3k, dequant_q4k, dequant_q5k, matvec_q2k, matvec_q3k, matvec_q4k, matvec_q5k
from .q6k import dequant_q6k, matvec_q6k

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
    q6k: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF Q6_K weight, multiplied by its own kernel
    q4k: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF Q4_K weight, multiplied by its own kernel
    q5k: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF Q5_K weight, multiplied by its own kernel
    q2k: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF Q2_K weight, multiplied by its own kernel
    q3k: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF Q3_K weight, multiplied by its own kernel
    iq4nl: tuple[Any, Any, int, int] | None  # (d f16, nibble bytes, rows, cols): IQ4_NL repacked struct-of-arrays
    iq4xs: tuple[Any, int, int] | None  # (raw bytes, rows, cols): a GGUF IQ4_XS weight, multiplied by its own kernel
    latt: tuple[str, Any, int, int, tuple[Any, ...]] | None  # (kind, raw bytes, rows, cols, side streams): IQ lattice

    def __init__(
        self,
        a: Any = None,
        packed: tuple[Any, ...] | None = None,
        shape: Sequence[int] | None = None,
        quant: tuple[Any, Any, Any, int, int] | None = None,
        q6k: tuple[Any, int, int] | None = None,
        q4k: tuple[Any, int, int] | None = None,
        q5k: tuple[Any, int, int] | None = None,
        q2k: tuple[Any, int, int] | None = None,
        q3k: tuple[Any, int, int] | None = None,
        iq4nl: tuple[Any, Any, int, int] | None = None,
        iq4xs: tuple[Any, int, int] | None = None,
        latt: tuple[str, Any, int, int, tuple[Any, ...]] | None = None,
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
        if q6k is not None:
            m.eval(q6k[0])
        if q4k is not None:
            m.eval(q4k[0])
        if q5k is not None:
            m.eval(q5k[0])
        if q2k is not None:
            m.eval(q2k[0])
        if q3k is not None:
            m.eval(q3k[0])
        if iq4nl is not None:
            m.eval(iq4nl[0], iq4nl[1])
        if iq4xs is not None:
            m.eval(iq4xs[0])
        if latt is not None:
            m.eval(latt[1], *latt[4])
        self.a = a
        self.packed = packed
        self.quant = quant
        self.q6k = q6k
        self.q4k = q4k
        self.q5k = q5k
        self.q2k = q2k
        self.q3k = q3k
        self.iq4nl = iq4nl
        self.iq4xs = iq4xs
        self.latt = latt
        self.shape = tuple(shape) if shape is not None else tuple(int(s) for s in a.shape)

    def get(self) -> mx_.array:
        if self.a is None:
            if self.q6k is not None:
                raw, rows, cols = self.q6k
                self.a = dequant_q6k(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.q4k is not None:
                raw, rows, cols = self.q4k
                self.a = dequant_q4k(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.q5k is not None:
                raw, rows, cols = self.q5k
                self.a = dequant_q5k(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.q2k is not None:
                raw, rows, cols = self.q2k
                self.a = dequant_q2k(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.q3k is not None:
                raw, rows, cols = self.q3k
                self.a = dequant_q3k(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.iq4nl is not None:
                d, q, rows, cols = self.iq4nl
                self.a = dequant_iq4nl(d, q, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.iq4xs is not None:
                raw, rows, cols = self.iq4xs
                self.a = dequant_iq4xs(raw, rows, cols)
                mx().eval(self.a)
                return self.a
            if self.latt is not None:
                kind, raw, rows, cols, _side = self.latt
                self.a = dequant_lattice(kind, raw, rows, cols)
                mx().eval(self.a)
                return self.a
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
    def weight_affine(
        self, wq: Any, scales: Any, biases: Any, bits: int, group: int, shape: Sequence[int], dtype: Any = None
    ) -> Weight:
        """a weight the kernels multiply as stored: the affine form (`mx.quantized_matmul`), never a bf16 copy.
        The scales and biases are held at `dtype` (the compute dtype, bf16): matching the activations is the path
        MLX's kernel runs fastest, and on an already-quantized weight their rounding is inside the file's own loss.
        `dtype` None keeps them float32 (an `--fp32` run, where the repack stays exact)."""
        m = mx()
        sd = dtype or m.float32
        q = (m.array(wq), m.array(scales).astype(sd), m.array(biases).astype(sd), int(bits), int(group))
        return Weight(quant=q, shape=shape)

    def weight_q6k(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF Q6_K weight kept as its own bytes (`matvec_q6k` reads the superblocks as stored), never a bf16
        copy. `raw` the file's uint8 bytes for `shape` [rows, cols]."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(q6k=(mx().array(raw), rows, cols), shape=shape)

    def weight_q5k(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF Q5_K weight kept as its own bytes (`matvec_q5k` reads the superblocks as stored)."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(q5k=(mx().array(raw), rows, cols), shape=shape)

    def weight_q2k(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF Q2_K weight kept as its own bytes (`matvec_q2k` reads the superblocks as stored)."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(q2k=(mx().array(raw), rows, cols), shape=shape)

    def weight_q3k(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF Q3_K weight kept as its own bytes (`matvec_q3k` reads the superblocks as stored)."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(q3k=(mx().array(raw), rows, cols), shape=shape)

    def weight_iq4nl(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF IQ4_NL weight repacked once into struct-of-arrays (the f16 scales and the nibble bytes as two
        contiguous streams): ggml's 18-byte blocks straddle cache lines, the split streams read at DRAM speed."""
        rows, cols = int(shape[0]), int(shape[1])
        d, q = repack_iq4nl(raw)
        return Weight(iq4nl=(d, q, rows, cols), shape=shape)

    def weight_iq4xs(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF IQ4_XS weight kept as its own bytes (`matvec_iq4xs` reads the superblocks as stored)."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(iq4xs=(mx().array(raw), rows, cols), shape=shape)

    def weight_lattice(self, kind: str, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF IQ lattice weight (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M) kept as its own bytes, multiplied by
        `matvec_lattice` reading the grid-codebook superblocks as stored."""
        rows, cols = int(shape[0]), int(shape[1])
        side = repack_lattice(kind, raw)
        return Weight(latt=(kind, mx().array(raw), rows, cols, side), shape=shape)

    def weight_q4k(self, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF Q4_K weight kept as its own bytes (`matvec_q4k` reads the superblocks as stored), never a bf16
        or affine-repacked copy. `raw` the file's uint8 bytes for `shape` [rows, cols]."""
        rows, cols = int(shape[0]), int(shape[1])
        return Weight(q4k=(mx().array(raw), rows, cols), shape=shape)

    def matmul(self, x: mx_.array, w: Weight) -> mx_.array:
        """x [b, cols] MLX (bf16 or float32), w a `Weight`: y [b, rows] in x's dtype, lazily."""
        m = mx()
        if w.q6k is not None:
            raw, rows, cols = w.q6k
            return matvec_q6k(raw, x, rows, cols)
        if w.q4k is not None:
            raw, rows, cols = w.q4k
            return matvec_q4k(raw, x, rows, cols)
        if w.q5k is not None:
            raw, rows, cols = w.q5k
            return matvec_q5k(raw, x, rows, cols)
        if w.q2k is not None:
            raw, rows, cols = w.q2k
            return matvec_q2k(raw, x, rows, cols)
        if w.q3k is not None:
            raw, rows, cols = w.q3k
            return matvec_q3k(raw, x, rows, cols)
        if w.iq4nl is not None:
            d, q, rows, cols = w.iq4nl
            return matvec_iq4nl(d, q, x, rows, cols)
        if w.iq4xs is not None:
            raw, rows, cols = w.iq4xs
            return matvec_iq4xs(raw, x, rows, cols)
        if w.latt is not None:
            kind, raw, rows, cols, side = w.latt
            return matvec_lattice(kind, raw, x, rows, cols, side)
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
