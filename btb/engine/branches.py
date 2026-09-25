# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Rows stepped together over one cache: a session forked into rows that go on from its end (parallel samples,
best-of-n, beam search over one prefill; its cache read by every row and copied by none), or several sessions
batched (continuous batching: each row its own session, sessions joining and leaving between steps)."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch

from .. import mlx as mlxdev
from ..kinds import LayerKind, PassTag, Tokens
from ..sampling import GREEDY, Sampling
from .cache import ForkIndexedLayer, ForkLayer, GrowLayer, _DynamicLayer
from .hooks import Hooks, LogitsProcessor, PassStats

if TYPE_CHECKING:
    from ..session import Session
    from .text import Generation


@dataclass
class _Row:
    """a row taken out of the batch: its own rows of each attention layer - torch (k, v) or MLX step-buffer
    slices - its recurrent states, and its drawn token not fed or its next-token logits"""

    kv: dict[int, Any] = field(default_factory=dict)
    lin: dict[int, tuple[dict[Any, torch.Tensor], dict[Any, torch.Tensor]]] = field(default_factory=dict)
    pending: int | None = None
    logits: torch.Tensor | None = None


class _Rows:
    """The rows and their cache: `sess[r]` the session row r writes back into, `base[r]` the tokens its cache
    prefix holds, `rows[r]` its tokens since the batch formed; `_slot[r]` its place in the batch (None once out),
    `_pend` the live rows' drawn tokens not fed, `_logits` their next-token logits, `_am` a ragged batch's mask."""

    salted = False  # each row draws under `sampling.row(r)` (a fork's rows, alike but for the draw)

    def __init__(self, eng: Any) -> None:
        self.eng = eng
        self.sess: list[Session] = []
        self.base: list[list[int]] = []
        self.rows: list[list[int]] = []
        self._slot: list[int | None] = []
        self._pend: list[int] | None = None
        self._logits: torch.Tensor | None = None
        self._am: torch.Tensor | None = None
        self.cache: Any = None
        self.mode = ""  # "rows": the MLX batched step over a flat buffer; "fork": the torch pass over ForkLayers

    # -- the cache --
    def _rows_ok(self, sessions: Sequence[Session], B: int) -> bool:
        """the MLX batched decode takes these rows: a dense family, every attention layer an MLX buffer"""
        eng = self.eng
        return bool(
            getattr(eng, "mlx", None) is not None
            and eng._mlx_batch_ok(max(2, B), None, False)
            and all(
                isinstance(cl, GrowLayer) and cl.shared and cl._mx is not None
                for s in sessions
                for i, cl in enumerate(s.cache.layers)
                if eng.layer_types[i] != LayerKind.LINEAR
            )
        )

    def _shell(self, parent: Any) -> Any:
        cache = copy.copy(parent)
        cache.layers = list(parent.layers)
        cache.btb_fork = True
        return self.eng._track(cache)

    @staticmethod
    def _lin_layer(pls: Sequence[Any], B: int) -> Any:
        """a recurrent layer whose rows are the given layers' states: one layer's repeated B times, or several
        layers' one a row"""
        cl = copy.copy(pls[0])

        def rows(ts: list[Any]) -> Any:
            if not isinstance(ts[0], torch.Tensor):
                return ts[0]  # a state slot not yet filled
            return ts[0].repeat(B, *([1] * (ts[0].dim() - 1))) if len(ts) == 1 else torch.cat(ts, dim=0)

        for name in ("conv_states", "recurrent_states"):
            st = getattr(pls[0], name)
            if isinstance(st, dict):
                setattr(cl, name, {k: rows([getattr(p, name)[k] for p in pls]) for k in st})
            else:
                setattr(cl, name, rows([getattr(p, name) for p in pls]))
        for name, val in vars(pls[0]).items():
            if isinstance(val, dict) and name not in ("conv_states", "recurrent_states"):
                setattr(cl, name, dict(val))
        if hasattr(cl, "_mx_pending"):
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

    def _check(self) -> None:
        if self.cache is None:
            raise ValueError(f"this {type(self).__name__} is closed")

    def step(self, tokens: Tokens | None = None) -> torch.Tensor:
        """Feed a token to each live row, in `live` order (None: the ones `generate` drew last), and return the
        next token's logits there, [live, V] float32."""
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
        out = self.eng._serial(self._advance, toks)
        if tokens is not None:
            for r, t in zip(live, toks):
                self.rows[r].append(t)
        self._pend, self._logits = None, out
        return cast(torch.Tensor, out)

    def _advance(self, toks: list[int]) -> torch.Tensor:
        eng = self.eng
        eng._tag(PassTag.ROWS_FLAT if self.mode == "rows" else PassTag.ROWS_JOINED)
        if self.mode == "rows":
            m = mlxdev.mx()
            hm = eng._mlx_embed_rows(m.array(toks, dtype=m.int32))
            if eng.compute_dtype is not None and eng.compute_dtype != torch.bfloat16:
                hm = hm.astype(m.float32)
            ns = [int(p) for p in self._rows_layers()[0]._ns]
            lg = eng._forward_mlx(None, None, self.cache, None, False, True, eng.L, hm=hm, lazy=True, rows=ns)
            lg = lg.astype(m.float32)
            m.eval(lg)
            return mlxdev.from_mx(lg).clone()
        ids = torch.tensor(toks, dtype=torch.long).view(-1, 1)
        am = None
        if self._am is not None:
            am = self._am = torch.cat([self._am, torch.ones((len(toks), 1), dtype=torch.long)], dim=1)
        return cast(torch.Tensor, eng.forward(ids, cache=self.cache, attention_mask=am))[:, -1].float().cpu()

    def _rows_layers(self) -> list[GrowLayer]:
        return [cl for cl in self.cache.layers if isinstance(cl, GrowLayer)]

    def generate(
        self,
        max_new: int,
        eos: Tokens | None = None,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        on_pass: Callable[[PassStats], Any] | None = None,
        on_token: Callable[[int, int], Any] | None = None,
    ) -> Generation:
        """Decode every live row up to `max_new` tokens, a row leaving the batch at a stop token (`eos`, the
        model's by default). Returns a `Generation` whose tokens (and `logprobs`) are a list a row - empty for a
        row not live. The live rows' last tokens are drawn and not fed (`pending`); hooks as `generate` takes them,
        and `on_token(row, token)` called with each token drawn."""
        self._check()
        from .text import GenerateStats, Generation

        eng = self.eng
        eng._pass_reset()  # one report a generate, as the model's own
        stop = {int(e) for e in (eng.stop_ids if eos is None else eos)}
        smp = (sampling if sampling is not None else getattr(eng, "sampling", None) or GREEDY).seeded()
        eng._tag(PassTag.SPEC_OFF, PassTag.SAMPLE_GREEDY if smp.greedy else PassTag.SAMPLE_STOCHASTIC)
        if processors or logprobs is not None:
            eng._tag(PassTag.PICK_HOOKED)
        hk = Hooks(tuple(processors), None if logprobs is None else int(logprobs), (), on_pass)
        hk.rows(len(self.sess))
        new: list[list[int]] = [[] for _ in self.sess]
        t0 = time.perf_counter()
        steps = 0

        def run() -> None:
            nonlocal steps
            for k in range(int(max_new)):
                live = self.live
                if eng.abort.is_set() or not live:
                    break
                ts = time.perf_counter()
                if self._pend is not None:
                    self._logits = self._advance(self._pend)
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
                done = []
                for j, (r, t) in enumerate(zip(live, picks)):
                    hk.record(r, lg[j], t)
                    self.rows[r].append(t)
                    new[r].append(t)
                    if on_token is not None:
                        on_token(r, t)
                    if t in stop:
                        done.append(j)
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
                self._pend, self._logits = picks, None
                if done:
                    self._leave(done)

        eng._serial(run)
        stats: dict[str, Any] = {"cap": int(max_new), "proposer": "greedy", "forwards": steps}
        if not smp.greedy:
            stats["seed"] = smp.seed
        stats["seconds"] = time.perf_counter() - t0
        return Generation(new, cast(GenerateStats, stats), hk.lp if hk.logprobs is not None else None, None)

    # -- re-forming the batch --
    def _select(self, slots: list[int]) -> None:
        """the batch's rows at `slots` become the batch, in that order"""
        if self.mode == "rows":
            m = mlxdev.mx()
            idx = m.array(slots, dtype=m.int32)
            for cl in self._rows_layers():
                if cl._mx2 is not None:
                    cl._mx2 = [m.take(x, idx, axis=0) for x in cl._mx2]
                    m.eval(*cl._mx2)
                cl._ns = [cl._ns[b] for b in slots]
                cl._seg = [cl._seg[b] for b in slots] if cl._seg is not None else None
                cl._offs = [cl._offs[b] for b in slots]
                cl._rows_total = len(slots)
                cl._n = max(cl._ns) if cl._ns else 0
        else:
            ti = torch.tensor(slots, dtype=torch.long)
            for cl in self.cache.layers:
                if isinstance(cl, ForkLayer):
                    cl.select(ti)
        for i in range(self.eng.L):
            if self.eng.layer_types[i] == LayerKind.LINEAR:
                cl = self.cache.layers[i]
                for name in ("conv_states", "recurrent_states"):
                    st = getattr(cl, name)
                    if isinstance(st, dict):
                        setattr(
                            cl,
                            name,
                            {
                                k: v.index_select(0, _index(slots, v)) if isinstance(v, torch.Tensor) else v
                                for k, v in st.items()
                            },
                        )
                    else:
                        setattr(cl, name, st.index_select(0, _index(slots, st)))
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
        for i, cl in enumerate(self.cache.layers):
            if isinstance(cl, ForkLayer):
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
                c, r = cl.conv_states, cl.recurrent_states
                out.lin[i] = (
                    {
                        k: v[b : b + 1].clone()
                        for k, v in (c.items() if isinstance(c, dict) else [(0, c)])
                        if isinstance(v, torch.Tensor)
                    },
                    {
                        k: v[b : b + 1].clone()
                        for k, v in (r.items() if isinstance(r, dict) else [(0, r)])
                        if isinstance(v, torch.Tensor)
                    },
                )
        if self._pend is not None:
            out.pending = self._pend[b]
        elif self._logits is not None:
            out.logits = self._logits[b].clone()
        return out

    def _write(self, r: int, row: _Row) -> None:
        """row r written into its session: its own rows appended to the session's cache, its states restored"""
        s, eng = self.sess[r], self.eng
        for i, kv in row.kv.items():
            pl = s.cache.layers[i]
            if self.mode == "rows":
                if pl.bits:
                    k = mlxdev.kv_dequantize(kv[0], kv[2])
                    v = mlxdev.kv_dequantize(kv[1], kv[3])
                else:
                    k, v = kv[0], kv[1]
                pl.mx_update(k, v)
                mlxdev.mx().eval(*[x for x in pl._mx if x is not None])
            else:
                pl.update(kv[0], kv[1])
                if len(kv) > 2:
                    pl.update_indexer(kv[2])
        for i, snap in row.lin.items():
            eng._lin_restore(s.cache.layers[i], snap)
        s.ids.extend(self.rows[r][:-1] if row.pending is not None else self.rows[r])
        s.n_prompt = len(s.ids)
        s.pending, s.logits = row.pending, row.logits
        s.dr, s.dr_len, s.pend_h = None, 0, None

    def _leave(self, slots: list[int]) -> None:
        """the batch's rows at `slots` leave it (their stop token drawn), the rest re-formed"""
        live = self.live
        for b in slots:
            self._left(live[b], self._take(b))
        gone = set(slots)
        stay = [b for b in range(len(live)) if b not in gone]
        self._select(stay)
        for r in live:
            self._slot[r] = None
        for j, b in enumerate(stay):
            self._slot[live[b]] = j

    def _left(self, r: int, row: _Row) -> None:
        raise NotImplementedError

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.cache is not None:
            self.close()

    def close(self) -> None:
        raise NotImplementedError


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
        eng._serial(self._open, int(n))
        session.forked = self

    @property
    def n(self) -> int:
        return len(self.rows)

    def _open(self, B: int) -> None:
        s, eng = self.session, self.eng
        s._settle(eng)
        s._flush(eng)
        assert s.logits is not None and s.cache is not None
        rows_ok = self._rows_ok([s], B)
        cache = self._shell(s.cache)
        for i, pl in enumerate(s.cache.layers):
            if eng.layer_types[i] == LayerKind.LINEAR:
                cache.layers[i] = self._lin_layer([pl], B)
            elif rows_ok:
                cache.layers[i] = self._alias(pl, B)
            else:
                if isinstance(pl, GrowLayer) and pl._buf is not None:
                    # rows in the card's arena move out, or the next cache to take it would write over them
                    pl.detach()
                cache.layers[i] = _fork_layer(pl, pl.keys, pl.values, getattr(pl, "indexer_keys", None), B, i)
        self.cache, self.mode = cache, ("rows" if rows_ok else "fork")
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
        slots = [cast(int, self._slot[r]) for r in picks]
        self.eng._serial(self._select, slots)
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
        self.cache, self._logits, self._pend, self._out = None, None, None, {}


class Batch(_Rows):
    """Sessions decoded together, a row each (`model.batch(sessions)`): `logits`, `step` and `generate` as a
    fork's, each row drawing as its session's own decode would. A row leaving at its stop token is written back
    into its session at once, which is free again; `join(session)` adds a row between steps (the rows' caches
    copied into the batch's); `close()` writes every live row back. Rows are numbered in joining order."""

    def __init__(self, eng: Any, sessions: Sequence[Session]) -> None:
        super().__init__(eng)
        sessions = list(sessions)
        if not sessions:
            raise ValueError("a batch needs at least one session")
        self._admit(sessions)
        self.eng._serial(self._form, sessions, list(range(len(sessions))))
        for s in sessions:
            s.forked = self

    def _admit(self, sessions: Sequence[Session]) -> None:
        for s in sessions:
            if s._bound() is not self.eng:
                raise ValueError("a batch's sessions are the model's own: make them with this model's session()")
            if not len(s):
                raise ValueError("an empty session has nothing to decode: feed it a prompt first")
            if s in self.sess:
                raise ValueError("a session is in the batch already")
        if len({id(s) for s in sessions}) != len(sessions):
            raise ValueError("a session is given twice")

    def _form(self, sessions: list[Session], nums: list[int]) -> None:
        """the batch formed anew over `sessions`, row nums[j] the j-th: their caches copied into the batch's"""
        eng = self.eng
        for s in sessions:
            s._settle(eng)
            s._flush(eng)
        B = len(sessions)
        lens = [len(s.ids) for s in sessions]
        rows_ok = self._rows_ok(sessions, B)
        cache = self._shell(sessions[0].cache)
        for i in range(eng.L):
            pls = [s.cache.layers[i] for s in sessions]
            if eng.layer_types[i] == LayerKind.LINEAR:
                cache.layers[i] = self._lin_layer(pls, B)
            elif rows_ok:
                cache.layers[i] = self._flat(pls, lens)
            else:
                # left-padded: every row ends at the longest and steps together, the mask hiding the padding
                k = _pad([pl.keys[0] for pl in pls], lens, 1)
                v = _pad([pl.values[0] for pl in pls], lens, 1)
                ik = getattr(pls[0], "indexer_keys", None)
                if ik is not None:
                    ik = _pad([pl.indexer_keys[0] for pl in pls], lens, 0)
                cache.layers[i] = _fork_layer(pls[0], k, v, ik, B, i)
        self.cache, self.mode = cache, ("rows" if rows_ok else "fork")
        self._am = None
        if not rows_ok and len(set(lens)) > 1:
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
        self._logits = torch.stack([cast(torch.Tensor, s.logits) for s in sessions])
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
        self._admit([session])

        def run() -> None:
            if self._pend is not None:
                self._logits, self._pend = self._advance(self._pend), None
            live = self.live
            for b, r in enumerate(live):
                self._write(r, self._take(b))
            for r in live:
                self._slot[r] = None
            self._form([self.sess[r] for r in live] + [session], [*live, len(self.sess)])

        self.eng._serial(run)
        session.forked = self
        return len(self.sess) - 1

    def close(self) -> None:
        """every live row written back into its session, and the batch dropped"""
        if self.cache is None:
            return

        def run() -> None:
            for b, r in enumerate(self.live):
                self._write(r, self._take(b))

        self.eng._serial(run)
        for s in self.sess:
            if s.forked is self:
                s.forked = None
        self.cache, self._logits, self._pend = None, None, None


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


def _fork_layer(pl: Any, k: torch.Tensor, v: torch.Tensor, ik: Any, B: int, i: int) -> ForkLayer:
    """the fork's layer over prefix rows `k`, `v` (and a sparse layer's indexer keys `ik`) of parent layer `pl`"""
    if not isinstance(pl, _DynamicLayer) or getattr(pl, "is_sliding", False):
        # a layer that evicts rows (a sliding window's) cannot be joined to rows grown after it
        raise NotImplementedError(f"a fork of a {type(pl).__name__} cache layer (layer {i})")
    if ik is not None:
        return ForkIndexedLayer(k, v, ik, B)
    return ForkLayer(k, v, B)
