# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Rows stepped together over one cache: a session forked into rows that go on from its end (parallel samples,
best-of-n, beam search over one prefill; its cache read by every row and copied by none), or several sessions
batched (continuous batching: each row its own session, sessions joining and leaving between steps)."""

from __future__ import annotations

import copy
import time
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self, overload

import torch

from .. import mlx as mlxdev
from ..api import api, in_hook
from ..kinds import LayerKind, PassReport, PassTag, Tokens
from ..sampling import GREEDY, Sampling
from .cache import (
    CardRowsLayer,
    ForkIndexedLayer,
    ForkLayer,
    GraphStates,
    GrowLayer,
    _DynamicLayer,
    attention_rows,
    indexer_keys,
    linear_layer,
    mark_forked,
)
from .generate import lin_layer
from .hooks import Hooks, LogitsProcessor, OnPass, OnRowToken, Taps

if TYPE_CHECKING:
    import mlx.core as mx_
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    from ..session import Session
    from .cache import CacheLayer, KvCache
    from .generate import LinLayer, LinSnap
    from .model import StreamedTextModel
    from .text import BatchGeneration

    # a row's own attention rows taken out of the batch: a fork's torch (k, v[, indexer keys]), or the MLX step
    # buffer's slices (k, v[, their int8 scales])
    RowKV = tuple[torch.Tensor, ...] | list[mx_.array]


@dataclass
class _Row:
    """a row taken out of the batch: its own rows of each attention layer, its recurrent states, and its drawn
    token not fed or its next-token logits"""

    kv: dict[int, RowKV] = field(default_factory=dict)
    lin: dict[int, LinSnap] = field(default_factory=dict)
    pending: int | None = None
    logits: torch.Tensor | None = None


@api("rows")
class _Rows:
    """The rows and their cache: `sess[r]` the session row r writes back into, `base[r]` the tokens its cache
    prefix holds, `rows[r]` its tokens since the batch formed; `_slot[r]` its place in the batch (None once out),
    `_pend` the live rows' drawn tokens not fed, `_logits` their next-token logits, `_am` a ragged batch's mask."""

    salted = False  # each row draws under `sampling.row(r)` (a fork's rows, alike but for the draw)

    def __init__(self, eng: StreamedTextModel) -> None:
        self.eng = eng
        self.sess: list[Session] = []
        self.base: list[list[int]] = []
        self.rows: list[list[int]] = []
        self._slot: list[int | None] = []
        self._pend: list[int] | None = None
        self._logits: torch.Tensor | None = None
        self._am: torch.Tensor | None = None
        self.cache: KvCache | None = None
        # "rows": the MLX batched step over a flat buffer; "card": the card graph's rows pass over its arena;
        # "fork": the torch pass over ForkLayers
        self.mode = ""
        self._lens: list[int] = []  # on the card, the live rows' prefix lengths (a batch's mask, should they leave it)

    # -- the cache --
    def _card_ok(self, B: int) -> bool:
        """the card graph's rows pass takes B rows: the whole model resident on the card as one run"""
        return bool(self.eng._card_rows_ok(B))

    def _off_card(self) -> None:
        """the rows leave the card's arena for the torch pass (a layer left the card, or the rows outgrew the
        graphs): a fork's layers, a batch's prefixes left-padded under a mask"""
        cache = self._check()
        for i, cl in enumerate(cache.layers):
            if isinstance(cl, CardRowsLayer):
                cache.layers[i] = cl.to_fork()
        self.eng._card_rows_release(cache)
        lens = self._lens
        if len(set(lens)) > 1:
            P = max(lens)
            t = cache.get_seq_length() - P
            self._am = torch.zeros((len(lens), P + t), dtype=torch.long)
            for b, n in enumerate(lens):
                self._am[b, P - n :] = 1
        self.mode = "fork"

    def _rows_ok(self, sessions: Sequence[Session], B: int) -> bool:
        """the MLX batched decode takes these rows: a dense family, every attention layer an MLX buffer"""
        eng = self.eng
        return bool(
            getattr(eng, "mlx", None) is not None
            and eng._mlx_batch_ok(max(2, B), None, False)
            and all(
                isinstance(cl, GrowLayer) and cl.shared and cl._mx is not None
                for s in sessions
                for i, cl in enumerate(_cache_of(s).layers)
                if eng.layer_types[i] != LayerKind.LINEAR
            )
        )

    def _called(self, tag: PassTag) -> None:
        self.eng._called(tag)

    def _shell(self, parent: KvCache) -> KvCache:
        cache = copy.copy(parent)
        cache.layers = list(parent.layers)
        return self.eng._track(mark_forked(cache))

    @staticmethod
    def _lin_layer(pls: Sequence[CacheLayer], B: int) -> LinearAttentionCacheLayerMixin:
        """a recurrent layer whose rows are the given layers' states: one layer's repeated B times, or several
        layers' one a row"""
        lins = [linear_layer(p) for p in pls]
        cl = copy.copy(lins[0])

        def rows(ts: list[torch.Tensor | None]) -> torch.Tensor | None:
            t0 = ts[0]
            if t0 is None:
                return None  # a state slot not yet filled
            if len(ts) == 1:
                return t0.repeat(B, *([1] * (t0.dim() - 1)))
            return torch.cat([t for t in ts if t is not None], dim=0)

        for name, val in vars(lins[0]).items():
            if isinstance(val, dict):
                setattr(cl, name, dict(val))
        cl.conv_states = {k: rows([p.conv_states[k] for p in lins]) for k in lins[0].conv_states}
        cl.recurrent_states = {k: rows([p.recurrent_states[k] for p in lins]) for k in lins[0].recurrent_states}
        if isinstance(cl, GraphStates):
            cl._mx_pending = cl._mx_prev = None
        return cl

    # -- the rows --
    @property
    def live(self) -> list[int]:
        """the rows in the batch, in its order"""
        live = [(b, r) for r, b in enumerate(self._slot) if b is not None]
        return [r for _, r in sorted(live)]

    @property
    def logits(self) -> torch.Tensor | None:
        """the live rows' next-token logits [live, V] float32; None while `generate`'s last tokens wait to be fed"""
        return self._logits

    @property
    def pending(self) -> list[int] | None:
        """the tokens `generate` drew last for the live rows and has not fed; `step()` feeds them"""
        return None if self._pend is None else list(self._pend)

    def tokens(self, r: int) -> list[int]:
        """row r's whole sequence"""
        return [*self.base[r], *self.rows[r]]

    def _check(self) -> KvCache:
        """the batch's cache; a ValueError once it is closed"""
        if self.cache is None:
            raise ValueError(f"this {type(self).__name__} is closed")
        return self.cache

    @overload
    def step(self, tokens: Tokens | None = None) -> torch.Tensor: ...

    @overload
    def step(self, tokens: Tokens | None = None, *, taps: Sequence[int]) -> tuple[torch.Tensor, Taps]: ...

    def step(
        self, tokens: Tokens | None = None, *, taps: Sequence[int] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, Taps]:
        """Feed a token to each live row, in `live` order (None: the ones `generate` drew last), and return the
        next token's logits there, [live, V] float32; with `taps`, also those layers' states at the fed
        tokens, {layer: [live, H]} float32 (as `Session.feed` taps them)."""
        self._check()
        live = self.live
        if not live:
            raise ValueError("no row is live")
        if tokens is None:
            if self._pend is None:
                raise ValueError("step needs a token a live row")
            toks = list(self._pend)
        else:
            if self._pend is not None:
                raise ValueError("the live rows hold drawn tokens not fed yet: step() feeds them first")
            toks = [int(t) for t in tokens]
        if len(toks) != len(live):
            raise ValueError(f"{len(toks)} tokens for {len(live)} live rows")
        want = tuple(int(i) % self.eng.L for i in taps) if taps is not None else ()
        out, hidden = self.eng._serial(self._advance, toks, want)
        if tokens is not None:
            for r, t in zip(live, toks):
                self.rows[r].append(t)
        self._pend, self._logits = None, out
        return out if taps is None else (out, hidden)

    def leave(self, r: int) -> None:
        """Row r out of the batch before its stop token (a candidate done early): the others step on, every row
        keeping its number. A fork keeps the row for `keep`; a batch writes it back into its session."""
        self._check()
        slot = self._slot[int(r)] if 0 <= int(r) < len(self._slot) else None
        if slot is None:
            raise ValueError(f"row {r} is not in the batch")
        self.eng._serial(self._leave, [slot])

    def _advance(self, toks: list[int], taps: Sequence[int] = (), host: bool = True) -> tuple[torch.Tensor, Taps]:
        """one token into each live row: their logits [live, V] float32 - on the host, or with `host=False` where
        the pass left them (`generate` picks there, and brings back a token a row instead of the vocabulary) - and
        the `taps` layers' states there"""
        eng, cache = self.eng, self._check()
        B = len(toks)
        if self.mode == "card" and not eng._card_rows_ok(B, cache):
            self._off_card()
        # one step is one transaction over the rows: a pass failing part way leaves every row where the step began
        undo = self._point(cache)
        try:
            return self._step_rows(cache, toks, taps, host)
        except BaseException:
            for u in undo:
                u()
            raise

    def _point(self, cache: KvCache) -> list[Callable[[], None]]:
        """what puts the rows back where they stand now: each layer's own step count (the rows past it are the step's
        to write), a batch's padding mask, and a hybrid's recurrent states (copies: the pass writes them in place)"""
        eng = self.eng
        undo: list[Callable[[], None]] = []
        for i, cl in enumerate(cache.layers):
            if eng.layer_types[i] == LayerKind.LINEAR:
                lin = lin_layer(cl)
                snap = eng._lin_snap(lin)

                def states(lin: LinLayer = lin, snap: LinSnap = snap) -> None:
                    eng._lin_restore(lin, snap)

                undo.append(states)
            elif isinstance(cl, ForkLayer):
                t0, ti0 = cl._t, getattr(cl, "_ti", None)

                def cut(cl: ForkLayer = cl, t0: int = t0, ti0: torch.Tensor | None = ti0) -> None:
                    cl._t, cl._cat = t0, None
                    if isinstance(cl, ForkIndexedLayer):
                        cl._ti = ti0

                undo.append(cut)
            elif isinstance(cl, CardRowsLayer):
                # the card's rows pass counts a step only once its replay is through; cut back all the same
                def steps(cl: CardRowsLayer = cl, t0: int = cl._t) -> None:
                    cl._t = t0

                undo.append(steps)
            elif isinstance(cl, GrowLayer) and cl._ns is not None:
                t0, ns0, n0 = cl._t, list(cl._ns), cl._n

                def back(cl: GrowLayer = cl, t0: int = t0, ns0: list[int] = ns0, n0: int = n0) -> None:
                    cl._t, cl._ns, cl._n = t0, list(ns0), n0

                undo.append(back)
        am = self._am
        undo.append(lambda: setattr(self, "_am", am))
        return undo

    def _step_rows(self, cache: KvCache, toks: list[int], taps: Sequence[int], host: bool) -> tuple[torch.Tensor, Taps]:
        """`_advance`'s pass, on the path the rows take"""
        eng, B = self.eng, len(toks)
        eng._tag({"rows": PassTag.ROWS_FLAT, "card": PassTag.ROWS_CARD}.get(self.mode, PassTag.ROWS_JOINED))
        if self.mode == "card":
            lg, tapped = eng._card_rows_step(cache, toks, tuple(taps))
            return (lg.cpu() if host else lg), tapped
        seen: Taps = {}

        def keep(i: int, h: torch.Tensor) -> None:
            if i in taps:
                seen[i] = h.reshape(B, -1, h.shape[-1])[:, -1].float().cpu()

        on_layer = keep if taps else None
        if self.mode == "rows":
            m = mlxdev.mx()
            hm = eng._mlx_embed_rows(m.array(toks, dtype=m.int32))
            if eng.compute_dtype is not None and eng.compute_dtype != torch.bfloat16:
                hm = hm.astype(m.float32)
            ns = [int(p) for p in self._rows_layers()[0]._ns]
            # a tapped pass is evaluated in place, its layers' outputs read with it; else the logits come lazily
            lg = eng._forward_mlx(None, None, cache, on_layer, False, True, eng.L, hm=hm, lazy=not taps, rows=ns)
            if isinstance(lg, torch.Tensor):
                return lg.reshape(B, -1).float().cpu(), seen
            lg = lg.astype(m.float32)
            m.eval(lg)
            return mlxdev.from_mx(lg).clone(), seen
        ids = torch.tensor(toks, dtype=torch.long).view(-1, 1)
        am = None
        if self._am is not None:
            am = self._am = torch.cat([self._am, torch.ones((B, 1), dtype=torch.long)], dim=1)
        out = eng.forward(ids, cache=cache, attention_mask=am, on_layer=on_layer)
        assert out is not None
        lg = out[:, -1].float()
        return (lg.cpu() if host else lg), seen

    def _rows_layers(self) -> list[GrowLayer]:
        return [cl for cl in self._check().layers if isinstance(cl, GrowLayer)]

    def generate(
        self,
        max_new: int,
        eos: Tokens | None = None,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        on_pass: OnPass | None = None,
        on_token: OnRowToken | None = None,
        until: Callable[[int], object] | None = None,
    ) -> BatchGeneration:
        """Decode every live row up to `max_new` tokens, a row leaving the batch at a stop token (`eos`, the
        model's by default). Returns a `Generation` whose tokens (and `logprobs`) are a list a row - empty for a
        row not live. The live rows' last tokens are drawn and not fed (`pending`); hooks as `generate` takes them,
        and `on_token(row, token)` called with each token drawn. `until(row)`, asked of each live row once its
        token is drawn (and `on_token` has seen it): true, the row leaves the batch there as its stop token would
        make it - a stop string, a client gone - and the rest go on."""
        self._check()
        from .text import GenerateStats, Generation

        eng = self.eng
        eng._pass_reset()  # one report a generate, as the model's own
        stop = {int(e) for e in (eng.stop_ids if eos is None else eos)}
        smp = (sampling if sampling is not None else getattr(eng, "sampling", None) or GREEDY).seeded()
        eng._tag(PassTag.SPEC_OFF, PassTag.SAMPLE_GREEDY if smp.greedy else PassTag.SAMPLE_STOCHASTIC)
        if processors or logprobs is not None:
            eng._tag(PassTag.PICK_HOOKED)
        # the caller's callbacks run between two steps: a call from them into the engine is refused (btb.api)
        hk = Hooks(
            tuple(in_hook(p) for p in processors),
            None if logprobs is None else int(logprobs),
            (),
            None if on_pass is None else in_hook(on_pass),
        )
        on_token = None if on_token is None else in_hook(on_token)
        until = None if until is None else in_hook(until)
        hk.rows(len(self.sess))
        new: list[list[int]] = [[] for _ in self.sess]
        t0 = time.perf_counter()
        steps = 0

        def run() -> None:
            nonlocal steps
            for k in range(int(max_new)):
                live = self.live
                if eng._stop_asked() or not live:
                    break
                ts = time.perf_counter()
                if self._pend is not None:
                    self._logits = self._advance(self._pend, host=False)[0]
                    self._pend = None
                    steps += 1
                assert self._logits is not None
                lg = self._logits
                if hk.processors:
                    lg = hk.process([self.tokens(r) for r in live], lg)
                keys = [
                    (smp.row(r) if self.salted else smp).key_for(len(self.base[r]) + len(self.rows[r]) - 1)
                    for r in live
                ]
                picks = [int(t) for t in smp.pick_torch(lg, keys).tolist()]
                # the step's picks are the rows' before a callback sees one: a callback raising leaves every row
                # holding its drawn token, pending
                for r, t in zip(live, picks):
                    self.rows[r].append(t)
                    new[r].append(t)
                self._pend, self._logits = picks, None
                done = [j for j, t in enumerate(picks) if t in stop]
                if done:
                    self._leave(done)
                for j, (r, t) in enumerate(zip(live, picks)):
                    hk.record(r, lg[j], t)
                    if on_token is not None:
                        on_token(r, t)
                if until is not None:
                    # the caller's say, once the rows' tokens are theirs: a row it is done with leaves here
                    now = self.live
                    quit = [b for b, r in enumerate(now) if until(r)]
                    if quit:
                        self._leave(quit)
                if hk.on_pass is not None:
                    hk.on_pass(
                        {
                            "index": k,
                            "drafted": 0,
                            "accepted": 0,
                            "tokens": len(live),
                            "seconds": time.perf_counter() - ts,
                        }
                    )

        def reported() -> PassReport:
            run()
            return eng.last_pass_report()  # under the lock: this call's report

        report = eng._serial(reported)
        stats: GenerateStats = {"cap": int(max_new), "proposer": "greedy", "forwards": steps}
        if not smp.greedy and smp.seed is not None:
            stats["seed"] = smp.seed
        stats["seconds"] = time.perf_counter() - t0
        gen: BatchGeneration = Generation(new, stats, hk.lp if hk.logprobs is not None else None, None)
        gen.report = report
        return gen

    # -- re-forming the batch --
    def _select(self, slots: list[int]) -> None:
        """the batch's rows at `slots` become the batch, in that order"""
        cache = self._check()
        if self.mode == "card" and slots and not self.eng._card_rows_ok(len(slots), cache):
            self._off_card()
        if self.mode == "card":
            self.eng._card_rows_select(cache, slots)
            self._lens = [self._lens[b] for b in slots]
        elif self.mode == "rows":
            m = mlxdev.mx()
            idx = m.array(slots, dtype=m.int32)
            for gl in self._rows_layers():
                if gl._mx2 is not None:
                    gl._mx2 = [m.take(x, idx, axis=0) for x in gl._mx2]
                    m.eval(*gl._mx2)
                gl._ns = [gl._ns[b] for b in slots]
                gl._seg = [gl._seg[b] for b in slots] if gl._seg is not None else None
                gl._offs = [gl._offs[b] for b in slots]
                gl._rows_total = len(slots)
                gl._n = max(gl._ns) if gl._ns else 0
        else:
            ti = torch.tensor(slots, dtype=torch.long)
            for cl in cache.layers:
                if isinstance(cl, ForkLayer):
                    cl.select(ti)
        for i in range(self.eng.L):
            if self.eng.layer_types[i] == LayerKind.LINEAR:
                lin = linear_layer(cache.layers[i])
                for states in (lin.conv_states, lin.recurrent_states):
                    for k, v in states.items():
                        if v is not None:
                            states[k] = v.index_select(0, _index(slots, v))
        pick = torch.tensor(slots, dtype=torch.long)
        if self._logits is not None:
            self._logits = self._logits[pick]
        if self._am is not None:
            self._am = self._am[pick]
        if self._pend is not None:
            self._pend = [self._pend[b] for b in slots]

    def _take(self, b: int) -> _Row:
        """the batch's row b, copied out: its own rows and its recurrent states"""
        out = _Row()
        for i, cl in enumerate(self._check().layers):
            if isinstance(cl, (ForkLayer, CardRowsLayer)):
                kv = cl.row(b)
                if kv is not None:
                    out.kv[i] = kv
            elif isinstance(cl, GrowLayer) and self.mode == "rows":
                if cl._mx2 is not None and cl._t:
                    m = mlxdev.mx()
                    parts = [m.contiguous(x[b : b + 1, :, : cl._t]) for x in cl._mx2]
                    m.eval(*parts)
                    out.kv[i] = parts
            elif self.eng.layer_types[i] == LayerKind.LINEAR:
                lin = linear_layer(cl)
                out.lin[i] = (
                    {k: v[b : b + 1].clone() for k, v in lin.conv_states.items() if v is not None},
                    {k: v[b : b + 1].clone() for k, v in lin.recurrent_states.items() if v is not None},
                )
        if self._pend is not None:
            out.pending = self._pend[b]
        elif self._logits is not None:
            out.logits = self._logits[b].clone()
        return out

    def _write(self, r: int, row: _Row) -> None:
        """row r committed into its session, one transaction (`Session._merge_row`): its own rows appended to the
        session's cache, its states put in, its tokens and its pending token or logits the session's"""
        from transformers.cache_utils import CacheLayerMixin, DynamicIndexedLayer

        eng = self.eng

        def write(cache: KvCache) -> None:
            for i, kv in row.kv.items():
                pl = cache.layers[i]
                if isinstance(kv, list):
                    # the MLX step buffer's slices, into the session's MLX buffer
                    assert isinstance(pl, GrowLayer)
                    if pl.bits:
                        k = mlxdev.kv_dequantize(kv[0], kv[2])
                        v = mlxdev.kv_dequantize(kv[1], kv[3])
                    else:
                        k, v = kv[0], kv[1]
                    pl.mx_update(k, v)
                    mlxdev.mx().eval(*[x for x in pl._mx if x is not None])
                else:
                    assert isinstance(pl, CacheLayerMixin)
                    pl.update(kv[0], kv[1])
                    if len(kv) > 2 and isinstance(pl, DynamicIndexedLayer):
                        pl.update_indexer(kv[2])
            for i, snap in row.lin.items():
                eng._lin_restore(lin_layer(cache.layers[i]), snap)

        toks = self.rows[r][:-1] if row.pending is not None else self.rows[r]
        self.sess[r]._merge_row(eng, self, toks, write, row.pending, row.logits)

    def _leave(self, slots: list[int]) -> None:
        """the batch's rows at `slots` leave it (their stop token drawn), the rest re-formed. A row's write-back
        failing leaves it in the batch, the rows written before it gone: none is written twice"""
        live = self.live
        gone: list[int] = []
        try:
            for b in slots:
                self._left(live[b], self._take(b))
                gone.append(b)
        finally:
            if gone:
                stay = [b for b in range(len(live)) if b not in gone]
                self._select(stay)
                for r in live:
                    self._slot[r] = None
                for j, b in enumerate(stay):
                    self._slot[live[b]] = j

    def _left(self, r: int, row: _Row) -> None:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.cache is not None:
            self.close()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


@api("branches")
class Branches(_Rows):
    """`n` rows forked from a session's end, stepped together over its cache (`session.fork(n)`): `logits` holds
    the live rows' next-token logits, `step(tokens)` feeds one token a live row, `generate` decodes them (row r
    drawing under `sampling.row(r)`: row 0 the draw the session's own decode makes, the rest seeds of their own),
    a row leaving the batch at its stop token; `reorder` re-forms the batch from its rows (a beam's survivors);
    `keep(r)` writes row r's tokens into the session and closes the fork. Rows are numbered from 0 as forked."""

    salted = True

    def __init__(self, session: Session, n: int) -> None:
        if int(n) < 1:
            raise ValueError(f"a fork needs at least one row, not {n}")
        eng = session._bound()
        if not len(session):
            raise ValueError("an empty session has nothing to fork: feed it a prompt first")
        super().__init__(eng)
        self.session = session
        self._out: dict[int, _Row] = {}

        def run() -> None:
            # taken under the decode lock, where a feed or another fork checks it: two threads cannot both have it
            session._unforked()
            self._open(int(n))
            session.forked = self

        eng._serial(run)

    @property
    def n(self) -> int:
        return len(self.rows)

    def _open(self, B: int) -> None:
        s, eng = self.session, self.eng
        s._settle(eng)
        s._flush(eng)
        parent = _cache_of(s)
        assert s.logits is not None
        rows_ok = self._rows_ok([s], B)
        card_ok = not rows_ok and self._card_ok(B)
        cache = self._shell(parent)
        for i, pl in enumerate(parent.layers):
            if eng.layer_types[i] == LayerKind.LINEAR:
                cache.layers[i] = self._lin_layer([pl], B)
            elif rows_ok:
                assert isinstance(pl, GrowLayer)
                cache.layers[i] = self._alias(pl, B)
            elif card_ok:
                continue  # every layer formed below, together, over the session's rows in the card's arena
            else:
                if isinstance(pl, GrowLayer) and pl._buf is not None:
                    # rows in the card's arena move out, or the next cache to take it would write over them
                    pl.detach()
                k, v = attention_rows(pl)
                cache.layers[i] = _fork_layer(pl, k, v, indexer_keys(pl), B, i)
        if card_ok:
            self._lens = eng._card_rows_form(cache, [parent], [0] * B)
        self.cache, self.mode = cache, ("rows" if rows_ok else "card" if card_ok else "fork")
        self.sess = [s] * B
        self.base = [list(s.ids)] * B
        self.rows = [[] for _ in range(B)]
        self._slot = list(range(B))
        self._logits = s.logits[None].expand(B, -1).clone()

    @staticmethod
    def _alias(pl: GrowLayer, B: int) -> GrowLayer:
        """the MLX batched layer over the parent's own buffer: a flat cache whose every row starts at offset 0"""
        pl._apply_overrides()
        P = int(pl.get_seq_length())
        cl = GrowLayer(shared=True, bits=pl.bits)
        empty = torch.empty(0, dtype=pl.dtype)
        cl.lazy_initialization(empty, empty)
        cl.batch_rows(B, dec_cap=256, lens=[P] * B)
        cl._offs = [0] * B
        cl._mx = list(pl._mx)
        cl._shape, cl._b, cl._ptr = pl._shape, 1, pl._ptr
        cl._ns = [P] * B
        cl.select_row(None)
        return cl

    def _left(self, r: int, row: _Row) -> None:
        self._out[r] = row

    def reorder(self, rows: Sequence[int]) -> None:
        """Re-form the batch as copies of `rows` (a beam's survivors, with repeats): row j becomes a copy of row
        rows[j] - its tokens, its cache, its logits - and rows are numbered 0.. in the new order."""
        self._check()
        if self._out:
            raise ValueError("a row has left the batch: reorder the rows before any stops")
        picks = [int(r) for r in rows]
        if not picks or any(not 0 <= r < self.n for r in picks):
            raise ValueError(f"reorder takes rows of 0..{self.n - 1}")
        slots = [self._slot[r] for r in picks]
        live = [b for b in slots if b is not None]
        self.eng._serial(self._select, live)
        self.rows = [list(self.rows[r]) for r in picks]
        self.base = [self.base[0]] * len(picks)
        self.sess = [self.session] * len(picks)
        self._slot = list(range(len(picks)))

    def keep(self, r: int) -> Session:
        """Row r's tokens written into the session (its own rows appended to the session's cache) and the fork
        closed; returns the session, which goes on from row r's end."""
        self._check()
        if not 0 <= int(r) < self.n:
            raise ValueError(f"no row {r}: rows 0..{self.n - 1}")
        r = int(r)

        def run() -> None:
            slot = self._slot[r]
            self._write(r, self._out[r] if slot is None else self._take(slot))

        self.eng._serial(run)
        self.close()
        return self.session

    def close(self) -> None:
        """the fork dropped: the session goes on from where it was forked"""
        if self.session.forked is self:
            self.session.forked = None
        if self.cache is not None and self.mode == "card":
            self.eng._card_rows_release(self.cache)
        self.cache, self._logits, self._pend, self._out = None, None, None, {}
        self._slot = [None] * len(self._slot)


@api("batch")
class Batch(_Rows):
    """Sessions decoded together, a row each (`model.batch(sessions)`): `logits`, `step` and `generate` as a
    fork's, each row drawing as its session's own decode would. A row leaving at its stop token is written back
    into its session at once, which is free again; `join(session)` adds a row between steps (the rows' caches
    copied into the batch's); `close()` writes every live row back. Rows are numbered in joining order."""

    def __init__(self, eng: StreamedTextModel, sessions: Sequence[Session]) -> None:
        super().__init__(eng)
        sessions = list(sessions)
        if not sessions:
            raise ValueError("a batch needs at least one session")

        def run() -> None:
            self._admit(sessions)
            self._form(sessions, list(range(len(sessions))))
            for s in sessions:
                s.forked = self

        self.eng._serial(run)

    def _admit(self, sessions: Sequence[Session]) -> None:
        """the sessions free to join, under the decode lock (where a feed or a fork checks them too); one that
        left the batch joins again as any other"""
        for s in sessions:
            if s.engine is not self.eng:
                raise ValueError("a batch's sessions are the model's own: make them with this model's session()")
            s._unforked()
            if not len(s):
                raise ValueError("an empty session has nothing to decode: feed it a prompt first")
        if len({id(s) for s in sessions}) != len(sessions):
            raise ValueError("a session is given twice")

    def _form(self, sessions: list[Session], nums: list[int]) -> None:
        """the batch formed anew over `sessions`, row nums[j] the j-th: their caches copied into the batch's"""
        eng = self.eng
        for s in sessions:
            s._settle(eng, owner=self)
            s._flush(eng)
        B = len(sessions)
        lens = [len(s.ids) for s in sessions]
        caches = [_cache_of(s) for s in sessions]
        rows_ok = self._rows_ok(sessions, B)
        card_ok = not rows_ok and self._card_ok(B)
        if self.cache is not None and self.mode == "card":
            eng._card_rows_release(self.cache)  # the batch formed anew: its rows are written back already
        cache = self._shell(caches[0])
        for i in range(eng.L):
            pls = [c.layers[i] for c in caches]
            if eng.layer_types[i] == LayerKind.LINEAR:
                cache.layers[i] = self._lin_layer(pls, B)
            elif rows_ok:
                grows = [pl for pl in pls if isinstance(pl, GrowLayer)]
                assert len(grows) == len(pls)
                cache.layers[i] = self._flat(grows, lens)
            elif card_ok:
                continue  # every layer formed below, together: the sessions' rows end to end in the card's arena
            else:
                # left-padded: every row ends at the longest and steps together, the mask hiding the padding
                kvs = [attention_rows(pl) for pl in pls]
                k = _pad([kv[0][0] for kv in kvs], lens, 1)
                v = _pad([kv[1][0] for kv in kvs], lens, 1)
                iks = [indexer_keys(pl) for pl in pls]
                ik = _pad([x[0] for x in iks if x is not None], lens, 0) if iks[0] is not None else None
                cache.layers[i] = _fork_layer(pls[0], k, v, ik, B, i)
        if card_ok:
            self._lens = eng._card_rows_form(cache, caches, list(range(B)))
        self.cache, self.mode = cache, ("rows" if rows_ok else "card" if card_ok else "fork")
        self._am = None
        if self.mode == "fork" and len(set(lens)) > 1:
            P = max(lens)
            self._am = torch.zeros((B, P), dtype=torch.long)
            for b, n in enumerate(lens):
                self._am[b, P - n :] = 1
        grow = max(nums) + 1 - len(self.sess)
        self.sess += [sessions[0]] * grow
        self.base += [[] for _ in range(grow)]
        self.rows += [[] for _ in range(grow)]
        self._slot += [None] * grow
        for j, (r, s) in enumerate(zip(nums, sessions)):
            self.sess[r], self.base[r], self.rows[r], self._slot[r] = s, list(s.ids), [], j
        logits = [s.logits for s in sessions]
        self._logits = torch.stack([lg for lg in logits if lg is not None])
        self._pend = None

    @staticmethod
    def _flat(pls: list[GrowLayer], lens: list[int]) -> GrowLayer:
        """the MLX batched layer: the sessions' rows end to end in one flat buffer, a copy"""
        m = mlxdev.mx()
        for pl in pls:
            pl._apply_overrides()
        p0 = pls[0]
        cl = GrowLayer(shared=True, bits=p0.bits)
        empty = torch.empty(0, dtype=p0.dtype)
        cl.lazy_initialization(empty, empty)
        cl.batch_rows(len(pls), dec_cap=256, lens=lens)
        cl._ensure_flat(int(p0._shape[1]), int(p0._shape[-1]), p0.dtype)
        for pl, off, n in zip(pls, cl._offs, lens):
            for dst, src in zip(cl._mx, pl._mx):
                dst[0:1, :, off : off + n] = src[0:1, :, :n]
        m.eval(*cl._mx)
        cl._ns = list(lens)
        cl.select_row(None)
        return cl

    def _left(self, r: int, row: _Row) -> None:
        self._write(r, row)
        self.sess[r].forked = None

    def join(self, session: Session) -> int:
        """Add `session` as a row (its tokens fed and logits in hand first); the live rows' drawn tokens are fed
        and every live row's cache copied into the batch formed anew. Returns the new row's number."""
        self._check()

        def run() -> None:
            self._admit([session])
            # the joiner ready before the live rows are torn down for the batch formed anew: a join it would refuse
            # leaves the batch as it was
            session._settle(self.eng)
            _cache_of(session)
            if self._pend is not None:
                self._logits, self._pend = self._advance(self._pend)[0], None
            live = self.live
            self._write_back()
            try:
                self._form([self.sess[r] for r in live] + [session], [*live, len(self.sess)])
            except BaseException:
                # the rows are in their sessions already: they go free, and the batch holds none
                for r in live:
                    if self.sess[r].forked is self:
                        self.sess[r].forked = None
                raise
            session.forked = self

        self.eng._serial(run)
        return len(self.sess) - 1

    def _write_back(self) -> None:
        """every live row written into its session, each out of the batch as soon as it is in: a write failing
        part way leaves the rows before it written once, the rest still live"""
        for r, b in sorted(((r, b) for r, b in enumerate(self._slot) if b is not None), key=lambda rb: rb[1]):
            self._write(r, self._take(b))
            self._slot[r] = None

    def close(self) -> None:
        """every live row written back into its session, and the batch dropped"""
        if self.cache is None:
            return
        self.eng._serial(self._write_back)
        for s in self.sess:
            if s.forked is self:
                s.forked = None
        if self.mode == "card":
            self.eng._card_rows_release(self.cache)
        self.cache, self._logits, self._pend = None, None, None
        self._slot = [None] * len(self._slot)


def _cache_of(s: Session) -> KvCache:
    """a session's cache, which a fork or a batch reads from; a ValueError for a session without one"""
    if s.cache is None:
        raise ValueError("an empty session has nothing to decode: feed it a prompt first")
    return s.cache


def _index(slots: list[int], like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(slots, dtype=torch.long, device=like.device)


def _pad(rows: list[torch.Tensor], lens: list[int], dim: int) -> torch.Tensor:
    """each row's first lens[b] positions (along `dim`) right-aligned in one zero batch"""
    P = max(lens)
    shape = list(rows[0].shape)
    shape[dim] = P
    out = rows[0].new_zeros(len(rows), *shape)
    for b, (x, n) in enumerate(zip(rows, lens)):
        out[b].narrow(dim, P - n, n).copy_(x.narrow(dim, 0, n))
    return out


def _fork_layer(pl: CacheLayer, k: torch.Tensor, v: torch.Tensor, ik: torch.Tensor | None, B: int, i: int) -> ForkLayer:
    """the fork's layer over prefix rows `k`, `v` (and a sparse layer's indexer keys `ik`) of parent layer `pl`"""
    if not isinstance(pl, _DynamicLayer) or getattr(pl, "is_sliding", False):
        # a layer that evicts rows (a sliding window's) cannot be joined to rows grown after it
        raise NotImplementedError(f"a fork of a {type(pl).__name__} cache layer (layer {i})")
    if ik is not None:
        return ForkIndexedLayer(k, v, ik, B)
    return ForkLayer(k, v, B)
