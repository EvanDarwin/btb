# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX backend an engine on the MLX device holds: its weights as views over shared buffers, the matmul
route (the kernel up to 16 rows, the GEMM past), rope, the fused views, and the accounting of what MLX holds."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..kinds import QUANT_KIND, Log, Quant, QuantClass, quant_key
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
from .legacyq import dequant_legacy, matvec_legacy
from .q6k import dequant_q6k, gather_q6k, matvec_q6k

if TYPE_CHECKING:
    import mlx.core as mx_
    import torch

    from .core import Shared


# the k-quant and IQ4 types' own matvec and dequant, by `kinds.quant_key`; the affine and lattice classes each
# share one family keyed by the type (`stored_matvec`). Held to `kinds.stored_quants()` by test_runtime_guards
_MATVEC: dict[str, Callable[..., Any]] = {
    "q2k": matvec_q2k,
    "q3k": matvec_q3k,
    "q4k": matvec_q4k,
    "q5k": matvec_q5k,
    "q6k": matvec_q6k,
    "iq4nl": matvec_iq4nl,
    "iq4xs": matvec_iq4xs,
}
_DEQUANT: dict[str, Callable[..., Any]] = {
    "q2k": dequant_q2k,
    "q3k": dequant_q3k,
    "q4k": dequant_q4k,
    "q5k": dequant_q5k,
    "q6k": dequant_q6k,
    "iq4nl": dequant_iq4nl,
    "iq4xs": dequant_iq4xs,
}


def stored_streams(q: Quant, raw: Any) -> tuple[Any, ...]:
    """the arrays type `q`'s kernels read for the file's bytes `raw`: the bytes, IQ4_NL's split streams, a lattice
    type's bytes and side streams. Host bytes are copied in and repacked once, here; a slot's MLX view is read in
    place, its repack in the graph of each pass that reads it"""
    m = mx()
    a = raw if isinstance(raw, m.array) else m.array(np.asarray(raw, dtype=np.uint8).reshape(-1))
    out = (
        repack_iq4nl(a)
        if q is Quant.IQ4_NL
        else (a, *repack_lattice(quant_key(q), a))
        if QUANT_KIND[q] is QuantClass.LATTICE
        else (a,)
    )
    if a is not raw:
        m.eval(*out)
    return out


def stored_matvec(q: Quant, streams: tuple[Any, ...], x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T on type `q`'s own kernel over its `stored_streams`: f32 accumulation, y in x's
    dtype, batch-invariant (16 rows a launch)"""
    c, key = QUANT_KIND[q], quant_key(q)
    if c is QuantClass.AFFINE:
        return matvec_legacy(q.value.lower(), streams[0], x, rows, cols)
    if c is QuantClass.LATTICE:
        return matvec_lattice(key, streams[0], x, rows, cols, streams[1:])
    return _MATVEC[key](*streams, x, rows, cols)


def stored_dequant(q: Quant, streams: tuple[Any, ...], rows: int, cols: int) -> mx_.array:
    """the whole [rows, cols] weight of type `q` in bf16 from its `stored_streams`, gguf's numbers"""
    c, key = QUANT_KIND[q], quant_key(q)
    if c is QuantClass.AFFINE:
        return dequant_legacy(q.value.lower(), streams[0], rows, cols)
    if c is QuantClass.LATTICE:
        return dequant_lattice(key, streams[0], rows, cols)
    return _DEQUANT[key](*streams, rows, cols)


def _join_stored(ws: Sequence[Weight]) -> Weight | None:
    """resident stored weights of one type and width as one launch's: their streams joined (a stored matrix is its
    rows' blocks in order, so the join is a matrix whose rows run their own chains), each member then a view of its
    rows of the join, the bytes held once. None for mixed types, or a slot's (rewritten in place: a join goes stale)"""
    w0 = ws[0]
    if any(w.slot or w.q is not w0.q or w.shape[1] != w0.shape[1] or len(w.streams) != len(w0.streams) for w in ws):
        return None
    m = mx()
    parts = [[s.reshape(-1) for s in w.streams] for w in ws]
    joined = tuple(m.concatenate([p[i] for p in parts]) for i in range(len(w0.streams)))
    m.eval(*joined)
    at = [0] * len(joined)
    for w, p in zip(ws, parts, strict=True):
        views = []
        for i, s in enumerate(p):
            views.append(joined[i][at[i] : at[i] + int(s.size)])
            at[i] += int(s.size)
        w.streams = tuple(views)
        m.eval(*w.streams)
    return Weight(q=w0.q, streams=joined, shape=(sum(int(w.shape[0]) for w in ws), int(w0.shape[1])))


class Weight:
    """A linear's weight for the GPU: `get()` is the bf16 [rows, cols] array. Four kinds: a resident copy, a
    view of a shared byte slot, a packed record over a slot that unpacks on the GPU once per pass, or a GGUF
    type's blocks (`q`, `streams`) multiplied by the type's own kernel."""

    packed: Any
    shape: Any

    # a slot weight's place in its shared buffer (set by `weight_slot`; `weight_fused` reads them)
    sh: Any
    off: int
    nb: int
    a: Any
    host: Any  # a slot weight's bytes as a bf16 torch view, for the host's gemv over a few rows (`_HostLinear`)

    q: Quant | None  # a GGUF type multiplied as stored, its kernels' arrays in `streams` (`stored_streams`)
    streams: tuple[Any, ...]
    slot: bool  # `streams` read a slot rewritten in place (`weight_stored_slot`), never copied out of it

    def __init__(
        self,
        a: Any = None,
        packed: tuple[Any, ...] | None = None,
        shape: Sequence[int] | None = None,
        q: Quant | None = None,
        streams: tuple[Any, ...] = (),
        slot: bool = False,
    ) -> None:
        # every array made here is evaluated at once: a lazy node made on one thread cannot be evaluated by
        # another (MLX's default stream is per thread), and the server runs each request on its own thread
        m = mx()
        if a is not None:
            m.eval(a)
        if packed is not None:
            m.eval(*[x for x in packed[:5] if x is not None])
        if streams:
            m.eval(*streams)
        self.a = a
        self.packed = packed
        self.q = q
        self.streams = streams
        self.slot = slot
        self.shape = tuple(shape) if shape is not None else tuple(int(s) for s in a.shape)

    def get(self) -> mx_.array:
        if self.a is None:
            if self.q is not None:
                rows, cols = (int(s) for s in self.shape)
                self.a = stored_dequant(self.q, self.streams, rows, cols)
            else:
                lo, hi4, tbl, esc_idx, esc_val, n = self.packed
                self.a = unpack_bf16(lo, hi4, tbl, n, self.shape, esc_idx, esc_val)
            mx().eval(self.a)  # made on whichever thread asked first; a lazy node cannot cross threads
        return self.a

    def invalidate(self) -> None:
        """The slot under a streamed weight was rewritten: drop the unpacked copy of a packed one."""
        if self.packed is not None or self.q is not None:
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
                f"[mlx] {self.info.get('device_name')}: "
                f"unified memory {self.info.get('memory_size', 0) / 2**30:.0f} GB, "
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
        """One weight over several with the same columns (one matmul instead of several), None where they cannot be
        one: adjacent slot weights as a view, no copy; resident stored weights of one type joined (`_join_stored`)."""
        if ws and all(getattr(w, "q", None) is not None for w in ws):
            return _join_stored(ws)
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
    def weight_stored(self, q: Quant, raw: Any, shape: Sequence[int]) -> Weight:
        """a GGUF weight of a stored type `q` kept as its own blocks (`stored_streams`), never a bf16 copy: its
        type's kernel reads them as stored, batch-invariant. `raw` the file's uint8 bytes for `shape`."""
        return Weight(q=q, streams=stored_streams(q, raw), shape=shape)

    def weight_stored_slot(self, shared: Shared, off: int, nb: int, q: Quant, shape: Sequence[int]) -> Weight:
        """a stored type's blocks as the `nb` bytes of a shared slot at `off` (a cold layer's or an expert's,
        rewritten as the slot is): read in place, a type that repacks them repacking in each pass's graph"""
        return Weight(
            q=q, streams=stored_streams(q, shared.view_mx(off, nb, mx().uint8, (nb,))), shape=shape, slot=True
        )

    @staticmethod
    def gather_rows(w: Weight, tok: mx_.array) -> mx_.array:
        """rows `tok` (int32 [T]) of a stored weight as bf16 [T, cols], decoded from their own blocks - a tied
        head's table read without a bf16 copy of the vocabulary: Q6_K's gather kernel, else each row's blocks taken
        and dequantized"""
        assert w.q is not None  # the gather reads a stored weight's blocks
        m, (rows, cols) = mx(), (int(s) for s in w.shape)
        if w.q is Quant.Q6_K:
            return gather_q6k(w.streams[0], tok, cols)
        # the dequant reads the bytes alone, but IQ4_NL's split scale and nibble streams; a row's are contiguous
        streams = w.streams if w.q is Quant.IQ4_NL else w.streams[:1]
        picked = tuple(m.take(s.reshape(rows, -1), tok, axis=0).reshape(-1) for s in streams)
        return stored_dequant(w.q, picked, int(tok.shape[0]), cols)

    def matmul(self, x: mx_.array, w: Weight) -> mx_.array:
        """x [b, cols] MLX (bf16 or float32), w a `Weight`: y [b, rows] in x's dtype, lazily."""
        m = mx()
        if w.q is not None:
            rows, cols = (int(s) for s in w.shape)
            # the type's kernel reads the weight once a 16 rows, so a prefill of `gemm_rows` rows or more widens it
            # once to bf16 for MLX's gemm (4x faster at 512 rows), as a bf16 weight takes the gemm past 16
            if int(x.shape[0]) < self.gemm_rows:
                return stored_matvec(w.q, w.streams, x, rows, cols)
            W = stored_dequant(w.q, w.streams, rows, cols)
            return m.matmul(x, (W.astype(m.float32) if x.dtype == m.float32 else W).T)
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
        expert in expert order, `gu` one weight or a GGUF's (gate, up) pair; returns the summed [T, H] as a torch
        tensor. One graph, one eval."""
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
            if isinstance(gu, tuple):
                gate, up = self.matmul(cur, gu[0]), self.matmul(cur, gu[1])
            else:
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
        act: Callable[[mx_.array], mx_.array] | None = None,
    ) -> Any:
        """One MXFP4 expert over its rows as its own graph (`async_eval`): `xm` [T, H], `token_idx`/`wts` its rows and
        router weights, `pair` = (gu, dn) uint8 slot views with `gu_shape`/`dn_shape` (`ggml`: a GGUF's, gu the
        (gate, up) views in ggml's layout), the bias tables (None for a family without them), and the gate:
        gpt-oss's clamped GLU with `alpha`/`limit`, or `act(gate) * up` where `act` is given. Up to 16 rows through
        the batch-invariant matvec, more through a gemm. Returns (idx, y): the rows' indices (None for the whole
        input) and the weighted output in x's dtype."""
        m = mx()
        gu, dn = pair
        T = int(xm.shape[0])
        whole = T == 1 and len(token_idx) == 1
        idx = None if whole else m.array(np.asarray(token_idx, dtype=np.int32))
        cur = xm if whole else m.take(xm, idx, axis=0)
        if ggml:
            y = matmul_mxfp4_pair(gu[0], gu[1], gu_shape[0] // 2, gu_shape[1], cur)
        else:
            y = matmul_mxfp4(gu, gu_shape[0], gu_shape[1], cur)
        if gu_bias is not None:
            y = y + gu_bias[e].astype(xm.dtype)
        if act is not None:  # the pair's rows interleave gate and up
            h = (act(y[..., 0::2]) * y[..., 1::2]).astype(xm.dtype)
        else:
            gate = m.minimum(y[..., 0::2], limit)
            up = m.clip(y[..., 1::2], -limit, limit)
            h = ((up + 1) * (gate * m.sigmoid(alpha * gate))).astype(xm.dtype)
        o = matmul_mxfp4(dn, dn_shape[0], dn_shape[1], h, ggml=ggml)
        if dn_bias is not None:
            o = o + dn_bias[e].astype(xm.dtype)
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
