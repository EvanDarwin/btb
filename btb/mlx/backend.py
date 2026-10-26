# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX backend an engine on the MLX device holds: its weights as views over shared buffers, the matmul
route (the matvec for few rows, steel's gemm past), rope, the fused views, and the accounting of what MLX holds."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import numpy as np

from ..kinds import QUANT_KIND, Json, Log, Quant, QuantClass, quant_key, stored_in_place
from .core import bf16_weight, from_mx, info, mx, to_mx
from .gemv import gemv, matmul_mxfp4, matmul_mxfp4_pair, unpack_bf16
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
from .q6k import dequant_q6k, matvec_q6k
from .steel import FROM_ROWS, gemm, gemm_chain

if TYPE_CHECKING:
    import mlx.core as mx_
    import torch

    from .core import Shared

# a stored type's arrays its kernels read (`stored_streams`): its blocks' bytes, then any split or side streams
Streams: TypeAlias = "tuple[mx_.array, ...]"
# the file's bytes of a stored weight: an MLX view of a slot or a pool block, else host bytes copied in
RawBytes: TypeAlias = "mx_.array | np.ndarray"
# a 12-bit record's arrays (`unpack_bf16`): low bytes, high nibbles, the table, the escapes' indices and values, n
Packed: TypeAlias = "tuple[mx_.array, mx_.array, mx_.array, mx_.array | None, mx_.array | None, int]"
# an expert's weights as the MLX step multiplies them: gate and up as one weight or a GGUF's pair, then down
ExpertWeights: TypeAlias = "tuple[Weight | tuple[Weight, Weight], Weight]"
# an MXFP4 expert's bytes in its slot: gate_up (ggml's layout: the gate and up views), then down
Mxfp4Pair: TypeAlias = "tuple[mx_.array | tuple[mx_.array, mx_.array], mx_.array]"
# an expert's rows of the input, their router weights, and its weights (`Backend.experts`)
ExpertHit: TypeAlias = "tuple[Sequence[int], Sequence[float], ExpertWeights]"
# an expert's output (`Backend.expert_mx`): its rows' indices (None: the whole input) and its weighted rows
ExpertOut: TypeAlias = "tuple[mx_.array | None, mx_.array]"


class PackedEntry(TypedDict):
    """a 12-bit record's layout in its slot (the cold tier's manifest entry): the low bytes', high nibbles', padding's
    and escapes' sizes, the value table and the element count"""

    lo: int
    hi4: int
    pad: int
    esc: int
    table: Sequence[int]
    n: int


class Stat(TypedDict):
    """what the backend has run: linears, evaluations and their seconds, layers stepped and their seconds"""

    linears: int
    evals: int
    s: float
    layers: int
    layer_s: float


# the k-quant and IQ4 types' own matvec and dequant, by `kinds.quant_key` (each its own streams first, so `...`); the
# affine and lattice classes each share one family keyed by the type (`stored_matvec`). Held to
# `kinds.stored_quants()` by test_runtime_guards
_MATVEC: dict[str, Callable[..., mx_.array]] = {
    "q2k": matvec_q2k,
    "q3k": matvec_q3k,
    "q4k": matvec_q4k,
    "q5k": matvec_q5k,
    "q6k": matvec_q6k,
    "iq4nl": matvec_iq4nl,
    "iq4xs": matvec_iq4xs,
}
_DEQUANT: dict[str, Callable[..., mx_.array]] = {
    "q2k": dequant_q2k,
    "q3k": dequant_q3k,
    "q4k": dequant_q4k,
    "q5k": dequant_q5k,
    "q6k": dequant_q6k,
    "iq4nl": dequant_iq4nl,
    "iq4xs": dequant_iq4xs,
}


def stored_streams(q: Quant, raw: RawBytes) -> Streams:
    """the arrays type `q`'s kernels read for the file's bytes `raw`: the bytes, IQ4_NL's split streams, a lattice
    type's bytes and side streams. Host bytes are copied in and repacked once, here; a slot's MLX view is read in
    place, repacked after each fill (`Weight.invalidate`)"""
    m = mx
    a = raw if isinstance(raw, m.array) else m.array(np.asarray(raw, dtype=np.uint8).reshape(-1))
    out = (
        (a,) if stored_in_place(q) else repack_iq4nl(a) if q is Quant.IQ4_NL else (a, *repack_lattice(quant_key(q), a))
    )
    if a is not raw:
        m.eval(*out)
    return out


def place_streams(q: Quant, raw: np.ndarray, dst: np.ndarray, offs: Sequence[int]) -> list[int]:
    """type `q`'s streams for the file's bytes `raw` written into `dst` (bytes) at `offs` (`kinds.pooled_layout`'s),
    once: the blocks as they are, a split type's streams made from them first (`raw` may lie under `dst`). Returns
    each stream's bytes"""
    if stored_in_place(q):
        parts = [raw.reshape(-1).view(np.uint8)]
    else:
        parts = [np.asarray(s).reshape(-1).view(np.uint8).copy() for s in stored_streams(q, raw)]
    for o, p in zip(offs, parts, strict=True):
        dst[o : o + p.size] = p
    return [int(p.size) for p in parts]


def stored_matvec(q: Quant, streams: Streams, x: mx_.array, rows: int, cols: int) -> mx_.array:
    """y[T, rows] = x[T, cols] . W^T on type `q`'s own kernel over its `stored_streams`: the gemm's chain over its
    weights (`matvec`), y in x's dtype, each row's bits its own whatever the width"""
    c, key = QUANT_KIND[q], quant_key(q)
    if c is QuantClass.FLOAT:
        return gemv(_floats(q, streams[0], rows, cols), x)
    if c is QuantClass.AFFINE:
        return matvec_legacy(q.value.lower(), streams[0], x, rows, cols)
    if c is QuantClass.LATTICE:
        return matvec_lattice(key, streams[0], x, rows, cols, streams[1:])
    return _MATVEC[key](*streams, x, rows, cols)


def _floats(q: Quant, raw: mx_.array, rows: int, cols: int) -> mx_.array:
    """a float type's bytes (`EXPERT_FLOATS`: a GGUF's expert held as the file's) as its [rows, cols] matrix"""
    m = mx
    return raw.view(m.float16 if q is Quant.F16 else m.bfloat16).reshape(rows, cols)


def stored_dequant(q: Quant, streams: Streams, rows: int, cols: int, dtype: mx_.Dtype | None = None) -> mx_.array:
    """the whole [rows, cols] weight of type `q` from its `stored_streams` in `dtype` (bf16 by default; float32 for
    a float32 x's gemm): the weights its matvec multiplies"""
    c, key = QUANT_KIND[q], quant_key(q)
    if c is QuantClass.FLOAT:
        return _floats(q, streams[0], rows, cols).astype(dtype or mx.bfloat16)
    if c is QuantClass.AFFINE:
        return dequant_legacy(q.value.lower(), streams[0], rows, cols, dtype)
    if c is QuantClass.LATTICE:
        return dequant_lattice(key, streams[0], rows, cols, streams[1:], dtype)
    return _DEQUANT[key](*streams, rows, cols, dtype)


def _join_stored(ws: Sequence[Weight]) -> Weight | None:
    """resident stored weights of one type and width as one launch's: their streams joined (a stored matrix is its
    rows' blocks in order, so the join is a matrix whose rows run their own chains), each member then a view of its
    rows of the join, the bytes held once. None for mixed types, or a slot's (rewritten in place: a join goes stale)"""
    w0 = ws[0]
    if any(w.slot or w.q is not w0.q or w.shape[1] != w0.shape[1] or len(w.streams) != len(w0.streams) for w in ws):
        return None
    m = mx
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


def stream_dtypes(q: Quant) -> tuple[mx_.Dtype, ...]:
    """the element type each of type `q`'s streams is read as (`kinds.stored_stream_bytes`' order): bytes, but the
    f16 scale streams IQ4_NL and IQ1_M split out"""
    m = mx
    if q is Quant.IQ4_NL:
        return m.float16, m.uint8
    if q is Quant.IQ1_M:
        return m.uint8, m.float16, m.uint8
    return (m.uint8,)


def _pooled(sh: Shared, q: Quant, offs: Sequence[int], nbs: Sequence[int], shape: Sequence[int]) -> Weight:
    """a stored weight over its streams in pool block `sh`, stream i the `nbs[i]` bytes at `offs[i]`"""
    streams = tuple(sh.view_mx(o, n, dt, (n // dt.size,)) for o, n, dt in zip(offs, nbs, stream_dtypes(q), strict=True))
    w = Weight(q=q, streams=streams, shape=shape)
    w.sh, w.offs, w.nbs = sh, tuple(int(o) for o in offs), tuple(int(n) for n in nbs)
    w.off, w.nb = w.offs[0], w.nbs[0]
    return w


def _join_pooled(ws: Sequence[Weight]) -> Weight | None:
    """pool-resident stored weights (`Backend.weight_stored_pool`) of one type and width, each stream end to end in
    one block, as one launch's: a view over each stream's bytes. None where they are not: a copy would hold the bytes
    twice"""
    w0 = ws[0]
    sh = w0.sh
    if sh is None or w0.q is None or any(w.sh is not sh for w in ws):
        return None
    offs = list(w0.offs)
    for w in ws:
        if w.q is not w0.q or w.shape[1] != w0.shape[1] or list(w.offs) != offs:
            return None
        offs = [o + n for o, n in zip(offs, w.nbs, strict=True)]
    nbs = [o - o0 for o, o0 in zip(offs, w0.offs, strict=True)]
    return _pooled(sh, w0.q, w0.offs, nbs, (sum(int(w.shape[0]) for w in ws), int(w0.shape[1])))


class Weight:
    """A linear's weight for the GPU: `get()` is the bf16 [rows, cols] array. Four kinds: a resident copy, a
    view of a shared byte slot, a packed record over a slot that unpacks on the GPU once per pass, or a GGUF
    type's blocks (`q`, `streams`) multiplied by the type's own kernel."""

    packed: Packed | None
    shape: tuple[int, ...]

    # a slot weight's place in its shared buffer (set by `weight_slot`, none elsewhere; `weight_fused` reads them), a
    # pooled stored weight's its first stream's; every stream's in `offs`/`nbs`
    sh: Shared | None
    off: int
    nb: int
    offs: tuple[int, ...]
    nbs: tuple[int, ...]
    a: mx_.array | None

    q: Quant | None  # a GGUF type multiplied as stored, its kernels' arrays in `streams` (`stored_streams`)
    streams: Streams
    slot: bool  # `streams` read a slot rewritten in place (`weight_stored_slot`), never copied out of it
    raw: mx_.array | None  # a slot weight's bytes as the slot holds them, which `invalidate` repacks

    def __init__(
        self,
        a: mx_.array | None = None,
        packed: Packed | None = None,
        shape: Sequence[int] | None = None,
        q: Quant | None = None,
        streams: Streams = (),
        slot: bool = False,
    ) -> None:
        # every array made here is evaluated at once: a lazy node made on one thread cannot be evaluated by
        # another (MLX's default stream is per thread), and the server runs each request on its own thread
        m = mx
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
        self.raw = None
        self.sh, self.off, self.nb = None, 0, 0
        self.offs, self.nbs = (), ()
        if shape is None:
            if a is None:
                raise ValueError("[mlx] a weight without its array takes its shape")
            shape = a.shape
        self.shape = tuple(int(s) for s in shape)

    def get(self) -> mx_.array:
        if self.a is None:
            if self.q is not None:
                rows, cols = (int(s) for s in self.shape)
                self.a = stored_dequant(self.q, self.streams, rows, cols)
            elif self.packed is not None:
                lo, hi4, tbl, esc_idx, esc_val, n = self.packed
                self.a = unpack_bf16(lo, hi4, tbl, n, self.shape, esc_idx, esc_val)
            else:
                raise ValueError("[mlx] a weight with neither its array, a stored type's streams nor a 12-bit record")
            mx.eval(self.a)  # made on whichever thread asked first; a lazy node cannot cross threads
        return self.a

    def invalidate(self) -> None:
        """The slot under a streamed weight was rewritten: drop the unpacked copy of a packed one, and repack a stored
        type's streams from the slot's new bytes (IQ4_NL's split, a lattice type's side streams are copies)."""
        if self.packed is not None or self.q is not None:
            self.a = None
        if self.slot and self.q is not None:
            assert self.raw is not None  # a slot's weight keeps the slot's bytes (`weight_stored_slot`)
            self.streams = stored_streams(self.q, self.raw)
            mx.eval(*self.streams)

    def drop(self) -> None:
        """The slot goes back to be refilled: drop the unpacked copy (the streams repack once the fill lands)."""
        if self.packed is not None or self.q is not None:
            self.a = None

    def first_rows(self, n: int) -> Weight:
        """the weight's first `n` rows as a weight of their own over the same bytes, nothing decoded: a stored type's
        streams each hold its blocks row by row, so the rows are a prefix of each; a 12-bit record has no row prefix"""
        rows, cols = (int(s) for s in self.shape)
        if self.q is not None:
            return Weight(q=self.q, streams=tuple(s[: int(s.size) * n // rows] for s in self.streams), shape=(n, cols))
        if self.packed is not None:
            raise ValueError("[mlx] a 12-bit record's rows are not a prefix of its streams")
        return Weight(self.get()[:n])


def _expert_rows(xm: mx_.array, token_idx: Sequence[int]) -> ExpertOut:
    """an expert's rows of xm [T, H]: (their indices, None where it takes the one row whole; the rows)"""
    if int(xm.shape[0]) == 1 and len(token_idx) == 1:
        return None, xm
    idx = mx.array(np.asarray(token_idx, dtype=np.int32))
    return idx, mx.take(xm, idx, axis=0)


def act(name: str) -> Callable[[mx_.array], mx_.array] | None:
    """an MLP activation by its config name as MLX ops (the step's, and what the megakernel's copies are held to), None
    for one btb has none for"""
    m = mx
    if name in ("silu", "swish"):
        return lambda a: a * m.sigmoid(a)
    if name == "gelu":
        return lambda a: 0.5 * a * (1.0 + m.erf(a / 1.4142135623730951))
    if name in ("gelu_pytorch_tanh", "gelu_new"):
        return lambda a: 0.5 * a * (1.0 + m.tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)))
    if name == "relu":
        return lambda a: m.maximum(a, 0)
    return None


class Backend:
    gemm_rows: int
    info: Json
    mxfp4: bool
    stat: Stat

    def __init__(
        self,
        gemm_rows: int = 64,
        cache_limit_gb: float = 2.0,
        wire: bool = True,
        log: Log | None = None,
    ) -> None:
        m = mx
        self.gemm_rows = int(gemm_rows)
        self.info = info()
        self.stat = Stat(linears=0, evals=0, s=0.0, layers=0, layer_s=0.0)
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
        return int(mx.get_active_memory())

    @staticmethod
    def held_bytes() -> int:
        """everything MLX holds: the arrays alive and the freed buffers it keeps cached - an exact count,
        whether or not the pages behind them have been touched yet"""
        m = mx
        return int(m.get_active_memory()) + int(m.get_cache_memory())

    @staticmethod
    def peak_bytes() -> int:
        return int(mx.get_peak_memory())

    @staticmethod
    def clear_cache() -> None:
        mx.clear_cache()

    # -- weights --
    def weight(self, t: torch.Tensor) -> Weight:
        return Weight(bf16_weight(t))

    def weight_slot(self, shared: Shared, off: int, nb: int, shape: Sequence[int]) -> Weight:
        w = Weight(shared.view_mx(off, nb, mx.bfloat16, shape))
        w.sh, w.off, w.nb = shared, int(off), int(nb)
        return w

    def weight_fused(self, ws: Sequence[Weight]) -> Weight | None:
        """One weight over several with the same columns (one matmul instead of several), None where they cannot be
        one: adjacent slot weights as a view, no copy; resident stored weights of one type joined (`_join_stored`)."""
        if ws and all(w.q is not None for w in ws):
            return _join_pooled(ws) if any(w.sh is not None for w in ws) else _join_stored(ws)
        sh = ws[0].sh if ws else None
        if sh is None or any(w.sh is None for w in ws):
            return None
        cols = ws[0].shape[1]
        off = ws[0].off
        for w in ws:
            if w.sh is not sh or w.shape[1] != cols or w.off != off:
                return None
            off += w.nb
        nb = off - ws[0].off
        f = Weight(sh.view_mx(ws[0].off, nb, mx.bfloat16, (sum(w.shape[0] for w in ws), cols)))
        f.sh, f.off, f.nb = sh, ws[0].off, nb
        return f

    @staticmethod
    def weight_slot_packed(shared: Shared, off: int, e: PackedEntry, shape: Sequence[int]) -> Weight:
        """a 12-bit record (`e`'s layout) at `off` of a shared slot as a weight that unpacks on the GPU"""
        m = mx
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
    def weight_stored(self, q: Quant, raw: RawBytes, shape: Sequence[int]) -> Weight:
        """a GGUF weight of a stored type `q` kept as its own blocks (`stored_streams`), never a bf16 copy: its
        type's kernel reads them as stored, batch-invariant. `raw` the file's uint8 bytes for `shape`."""
        return Weight(q=q, streams=stored_streams(q, raw), shape=shape)

    def weight_stored_pool(
        self, shared: Shared, offs: Sequence[int], nbs: Sequence[int], q: Quant, shape: Sequence[int]
    ) -> Weight:
        """a resident GGUF weight of a stored type as its streams (`kinds.stored_stream_bytes`) in a pool block, stream
        i the `nbs[i]` bytes at `offs[i]`, never rewritten: neighbours join as a view a stream (`weight_fused`), the
        megakernel reads them where they are"""
        return _pooled(shared, q, offs, nbs, shape)

    def weight_stored_slot(self, shared: Shared, off: int, nb: int, q: Quant, shape: Sequence[int]) -> Weight:
        """a stored type's blocks as the `nb` bytes of a shared slot at `off` (a cold layer's or an expert's,
        rewritten as the slot is): read in place, a type that repacks them repacking after each fill (`invalidate`)"""
        raw = shared.view_mx(off, nb, mx.uint8, (nb,))
        w = Weight(q=q, streams=stored_streams(q, raw), shape=shape, slot=True)
        w.raw = raw
        return w

    @staticmethod
    def gather_rows(w: Weight, tok: mx_.array) -> mx_.array:
        """rows `tok` (int32 [T]) of a stored weight as bf16 [T, cols], decoded from their own blocks - a tied
        head's table read without a bf16 copy of the vocabulary: each row's part of each stream taken (a stream holds
        its rows in order), then dequantized as the megakernel's embedding decodes a row"""
        assert w.q is not None  # the gather reads a stored weight's blocks
        m, (rows, cols) = mx, (int(s) for s in w.shape)
        picked = tuple(m.take(s.reshape(rows, -1), tok, axis=0).reshape(-1) for s in w.streams)
        return stored_dequant(w.q, picked, int(tok.shape[0]), cols)

    def matmul(self, x: mx_.array, w: Weight) -> mx_.array:
        """x [b, cols] MLX (bf16 or float32), w a `Weight`: y [b, rows] in x's dtype, lazily."""
        m = mx
        floats = w.q is not None and QUANT_KIND[w.q] is QuantClass.FLOAT
        if w.q is not None and not floats and x.dtype in (m.float32, m.bfloat16):
            rows, cols = (int(s) for s in w.shape)
            # the type's kernels decode its weights a launch; a prefill of `gemm_rows` rows or more decodes them once
            # (bf16, or float32 for a float32 x: the kernels' numbers) for steel's gemm on the same chain
            if int(x.shape[0]) < self.gemm_rows:
                return stored_matvec(w.q, w.streams, x, rows, cols)
            return gemm_chain(stored_dequant(w.q, w.streams, rows, cols, x.dtype), x)
        # a GGUF's float experts as the file holds them, the dense route's chain over them as they are
        W = _floats(w.q, w.streams[0], *(int(s) for s in w.shape)) if floats and w.q is not None else w.get()
        # each row's bits its own whatever the width (a verify pass, a batch or any cut of a prefill computes each row
        # as the one-row step does): the engine's matvec, then steel's gemm (the same chain) from `FROM_ROWS` rows of
        # bf16, or `gemm_rows` of float32 (the gemm widens the weight a whole copy, the matvec as it stages it)
        if int(W.shape[1]) % 8 == 0 and x.dtype in (m.float32, m.bfloat16):
            narrow = FROM_ROWS if W.dtype == m.bfloat16 and x.dtype == m.bfloat16 else self.gemm_rows
            return gemv(W, x) if int(x.shape[0]) < narrow else gemm(W, x)
        if x.dtype == m.float32:
            return m.matmul(x, W.astype(m.float32).T)
        return m.matmul(x, W.T)

    def linear(self, x: torch.Tensor, w: Weight) -> torch.Tensor:
        """torch in, torch out: the module boundary of `_HostLinear`."""
        m = mx
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
        return act(name)

    def experts(
        self, x: torch.Tensor, hits: Sequence[ExpertHit], act: Callable[[mx_.array], mx_.array]
    ) -> torch.Tensor:
        """The mixture-of-experts step for one call: `hits` a list of (token_idx, top_k_weights, (gu, dn)) per active
        expert in expert order, `gu` one weight or a GGUF's (gate, up) pair; returns the summed [T, H] as a torch
        tensor. One graph, one eval."""
        m = mx
        t0 = time.perf_counter()
        xm = to_mx(x)
        T = int(xm.shape[0])
        H = int(xm.shape[1])
        final = m.zeros((T, H), dtype=xm.dtype)
        for token_idx, wts, (gu, dn) in hits:
            idx, cur = _expert_rows(xm, token_idx)
            if isinstance(gu, tuple):
                gate, up = self.matmul(cur, gu[0]), self.matmul(cur, gu[1])
            else:
                gate, up = m.split(self.matmul(cur, gu), 2, axis=-1)
            h = act(gate) * up
            y = self.matmul(h, dn) * m.array(np.asarray(wts, dtype=np.float32)).astype(xm.dtype)[:, None]
            final = final + y if idx is None else final.at[idx].add(y)
        m.eval(final)
        st = self.stat
        st["evals"] += 1
        st["s"] += time.perf_counter() - t0
        return from_mx(final)

    def expert_mx(
        self,
        xm: mx_.array,
        token_idx: Sequence[int],
        wts: Sequence[float],
        pair: Mxfp4Pair,
        e: int,
        gu_shape: Sequence[int],
        dn_shape: Sequence[int],
        gu_bias: mx_.array | None,
        dn_bias: mx_.array | None,
        alpha: float,
        limit: float,
        ggml: bool = False,
        act: Callable[[mx_.array], mx_.array] | None = None,
    ) -> ExpertOut:
        """One MXFP4 expert over its rows as its own graph (`async_eval`): `xm` [T, H], `token_idx`/`wts` its rows and
        router weights, `pair` = (gu, dn) uint8 slot views with `gu_shape`/`dn_shape` (`ggml`: a GGUF's, gu the
        (gate, up) views in ggml's layout), the bias tables (None for a family without them), and the gate:
        gpt-oss's clamped GLU with `alpha`/`limit`, or `act(gate) * up` where `act` is given. Under `gemm_rows` rows
        through the batch-invariant matvec, from there a gemm. Returns (idx, y): the rows' indices (None for the whole
        input) and the weighted output in x's dtype."""
        m = mx
        gu, dn = pair
        idx, cur = _expert_rows(xm, token_idx)
        g = self.gemm_rows
        if ggml:
            assert isinstance(gu, tuple)  # ggml's layout: the gate's and up's views
            y = matmul_mxfp4_pair(gu[0], gu[1], gu_shape[0] // 2, gu_shape[1], cur, gemm_rows=g)
        else:
            assert not isinstance(gu, tuple)
            y = matmul_mxfp4(gu, gu_shape[0], gu_shape[1], cur, gemm_rows=g)
        if gu_bias is not None:
            y = y + gu_bias[e].astype(xm.dtype)
        if act is not None:  # the pair's rows interleave gate and up
            h = (act(y[..., 0::2]) * y[..., 1::2]).astype(xm.dtype)
        else:
            gate = m.minimum(y[..., 0::2], limit)
            up = m.clip(y[..., 1::2], -limit, limit)
            h = ((up + 1) * (gate * m.sigmoid(alpha * gate))).astype(xm.dtype)
        o = matmul_mxfp4(dn, dn_shape[0], dn_shape[1], h, ggml=ggml, gemm_rows=g)
        if dn_bias is not None:
            o = o + dn_bias[e].astype(xm.dtype)
        y = o * m.array(np.asarray(wts, dtype=np.float32)).astype(xm.dtype)[:, None]
        m.async_eval(y)
        return idx, y

    def experts_mx_sum(self, xm: mx_.array, parts: Sequence[ExpertOut]) -> torch.Tensor:
        """The experts' outputs (in expert order) summed into [T, H] one add at a time, the same order alone or in a
        tile. Returns a torch tensor; one eval."""
        m = mx
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
        m = mx
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
        m = mx
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
            mx.synchronize()
        with contextlib.suppress(Exception):
            mx.clear_cache()
