# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A sequence's state between calls - its tokens and its cache - what a new prompt reuses of it, and the moves a
caller makes on it: feed tokens, mark a point, rewind to one, fork into rows.

A session is always in one of four states (`State`), and every change goes through one transaction (`_Txn`): a
rollback point is noted, the passes append to the cache past it, and `commit` is the one write of the tokens and the
state; left uncommitted - a hook raising, memory refused, a pass failing part way - the cache is cut back to the
point and the recurrent states, the drafter and the state are as they were. docs/sessions.md has the design."""

from __future__ import annotations

import contextlib
import enum
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, Unpack

from .api import api
from .kinds import LayerKind, PassTag, Tokens

if TYPE_CHECKING:
    import torch

    from .engine.branches import Branches, _Rows
    from .engine.cache import KvCache
    from .engine.drafter import MTPDrafter
    from .engine.generate import LinSnap
    from .engine.hooks import Taps
    from .engine.model import StreamedTextModel
    from .engine.state import _State
    from .engine.text import GenerateArgs, RowGeneration

# a chat template's generation tail is a handful of tokens (Qwen3's empty think block is four); a prompt that
# parts from the previous one further back than this is another conversation, and teaches no tail
TAIL_MAX = 64


class State(enum.Enum):
    """what a session holds: nothing; every token and the next token's logits; every token but the last, drawn
    and not fed yet; or lent to a fork or a batch, which holds it still"""

    EMPTY = "empty"
    READY = "ready"
    PENDING = "pending"
    LENT = "lent"


@dataclass(frozen=True, eq=False)
class Step:
    """What a feed or a rows' step gives back (docs/api.md), one shape whatever was asked: `logits` after each fed
    token, float32 - [T, V] for a session's feed (the last one's, [1, V], with `last_only`), [live, V] for rows -
    and `hidden`, the `taps` layers' states at the fed tokens ({layer: [T, H]} or {layer: [live, H]}; empty when no
    taps were asked for)."""

    logits: torch.Tensor
    hidden: Taps = field(default_factory=dict)


class Anchor(TypedDict):
    """a hybrid's point to resume a later prompt from: the rows before it, its linear layers' states there, the next
    token's logits there (so the point is a state of its own, `Ready`), and the last layer's state at its last row
    (the drafter's start)"""

    n: int
    states: dict[int, LinSnap]
    logits: torch.Tensor | None
    h_last: torch.Tensor | None


@dataclass(frozen=True)
class Mark:
    """a point in a session to rewind to: the rows its cache held, a hybrid's recurrent states there (they cannot
    be cropped back, so they are kept), the token drawn but not yet fed or the next token's logits, and the tokens
    it was taken on"""

    n: int
    states: dict[int, LinSnap] = field(default_factory=dict)
    pending: int | None = None
    logits: torch.Tensor | None = None
    # the tokens the mark was taken on: a rewind to it from a session that has since gone another way is refused,
    # not a cache whose rows are other tokens than the mark's logits follow
    path: tuple[int, ...] | None = None


def crop(cache: KvCache, engine: _State, n: int) -> None:
    """the attention layers of `cache` cut back to their first n rows"""
    from transformers.cache_utils import CacheLayerMixin, DynamicIndexedLayer, DynamicSlidingWindowLayer

    for i in range(engine.L):
        cl = cache.layers[i]
        if not isinstance(cl, CacheLayerMixin) or cl.keys is None or cl.values is None or cl.keys.shape[-2] <= n:
            continue
        cl.keys, cl.values = cl.keys[..., :n, :], cl.values[..., :n, :]
        if isinstance(cl, DynamicSlidingWindowLayer):
            cl.cumulative_length = int(cl.keys.shape[-2])
        if isinstance(cl, DynamicIndexedLayer) and cl.indexer_keys is not None and cl.indexer_keys.numel():
            cl.indexer_keys = cl.indexer_keys[:, :n]


def _cat(hs: list[torch.Tensor]) -> torch.Tensor:
    """a layer's states over a pass that reached it in pieces, joined along the positions"""
    import torch

    return torch.cat(hs, dim=0)


@dataclass
class _Point:
    """a session as it stood, which a transaction left uncommitted goes back to: the rows the cache held (the tokens
    `ids[:n]`), the token pending or the logits in hand, a hybrid's recurrent states there (None: untouched by the
    transaction, or a dense model), and the prompt length the tail is learned from"""

    n: int
    pending: int | None
    logits: torch.Tensor | None
    states: dict[int, LinSnap] | None
    n_prompt: int


class _Txn:
    """One change to a session: `commit` is its one write; left without it, `rollback` puts the session back at the
    point. The point is where the session stood when the transaction began, or - a prompt replacing the tokens past
    what it shares with them - where the call keeps them (`keep`)."""

    def __init__(self, s: Session, eng: _State, states: bool) -> None:
        self.s, self.eng, self.done = s, eng, False
        s._flush(eng)
        self.point = _Point(len(s.ids), s._pending, s.logits, s._snap(eng) if states else None, s.n_prompt)

    def keep(self, point: _Point) -> None:
        """the point moved to where the call keeps the session: the tokens past it are the call's to replace"""
        self.point = point

    def commit(
        self,
        ids: Sequence[int],
        *,
        pending: int | None = None,
        logits: torch.Tensor | None = None,
        cache: KvCache | None = None,
        n_prompt: int | None = None,
        anchors: Sequence[Anchor] | None = None,
        drafter: tuple[MTPDrafter | None, int, torch.Tensor | None] = (None, 0, None),
    ) -> None:
        """the session made `ids` (every one the cache holds), with the drawn token not yet fed or the next token's
        logits - exactly one of them, so no state without either is ever made"""
        if (pending is None) == (logits is None):
            raise AssertionError("a commit holds a pending token or the next token's logits, not both nor neither")
        s = self.s
        if cache is not None:
            s.cache = cache
        s.ids = [int(t) for t in ids]
        s._pending, s.logits = (int(pending) if pending is not None else None), logits
        s.n_prompt = len(s.ids) if n_prompt is None else int(n_prompt)
        if anchors is not None:
            s.anchor = list(anchors)
        dr, dr_len, pend_h = drafter
        s.dr, s.dr_len = dr, int(dr_len)
        s.pend_h = pend_h.detach().clone() if pend_h is not None else None
        self.done = True

    def rollback(self) -> None:
        """the session as it stood at the point: the cache cut back to its rows, the recurrent states restored, the
        drafter's rows let go (they no longer follow the cache)"""
        s, eng, p = self.s, self.eng, self.point
        dr = s.dr
        s.dr, s.dr_len, s.pend_h = None, 0, None
        if dr is not None and hasattr(dr, "reset"):
            dr.reset()
        self.done = True
        if p.n == 0 and p.pending is None and p.logits is None:
            s.cache, s.anchor, s.ids, s.n_prompt, s._pending, s.logits = None, [], [], 0, None, None
            return
        if p.n == 0 or s.cache is None:
            # a point over no rows: a fresh cache. No rows is no recurrent state either, and a snapshot taken there
            # holds none to restore - a hybrid's states the transaction's passes made would be left standing
            s.cache = eng.new_cache()
        else:
            import torch

            from .engine.generate import lin_layer

            # in inference mode whoever leaves the transaction: the states are inference tensors, written in place
            with torch.inference_mode():
                crop(s.cache, eng, p.n)
                for i, snap in (p.states or {}).items():
                    eng._lin_restore(lin_layer(s.cache.layers[i]), snap)
        del s.ids[p.n :]
        s.anchor = [a for a in s.anchor if a["n"] <= p.n]
        s.n_prompt = min(p.n_prompt, p.n)
        s._pending, s.logits = p.pending, p.logits


@api("session")
class Session:
    """The tokens the cache holds and the cache, a hybrid's DeltaNet snapshots (`anchor`), the drafter's tail
    (`dr`, `dr_len`, `pend_h`) and the template's generation tail (`tail`, learned or given). Made by
    `model.session()`, it is bound to that engine and takes `feed`, `mark`, `rewind`, `crop`, `fork` and
    `generate`; any session can be passed as `generate(session=...)`. Cut and read the rows through the session
    (`crop`, `rows`), not `cache.layers`: a layer's tensors may live in memory btb moves or hands to another cache.
    Its `state` is always one of `State`'s; a call failing part way leaves it as it was (docs/sessions.md)."""

    def __init__(self, tail: int = 0, engine: StreamedTextModel | None = None) -> None:
        self.ids: list[int] = []
        self.n_prompt = 0
        self.cache: KvCache | None = None
        self.anchor: list[Anchor] = []
        self.dr: MTPDrafter | None = None
        self.dr_len = 0
        self.pend_h: torch.Tensor | None = None
        self.tail = int(tail)
        self.engine = engine
        # written only by a transaction's commit or rollback: the sequence's last token when a decode drew it and
        # the cache does not hold it yet; else the next token's logits [V]
        self._pending: int | None = None
        self.logits: torch.Tensor | None = None
        self._forked: weakref.ref[_Rows] | None = None
        self._active: _Txn | None = None  # the transaction under way, one at a time
        # the next token's logits a decode reusing every row takes in place of a prefill (`_begin_decode`'s whole)
        self._held: torch.Tensor | None = None

    @property
    def state(self) -> State:
        """what the session holds now: `Empty`, `Ready` (the next token's logits in hand), `Pending` (a drawn token
        not fed yet), or `Lent` to a fork or a batch"""
        if self.forked is not None:
            return State.LENT
        if self.cache is None:
            return State.EMPTY
        return State.PENDING if self._pending is not None else State.READY

    @property
    def forked(self) -> _Rows | None:
        """the live `Branches` or `Batch` over this session, which holds it still. Held weakly: one dropped unclosed
        is one closed - the session goes on from where it was forked, or where it joined a batch (a row the batch
        wrote back as it left stays; `close` writes the rest)"""
        ref = self._forked
        return ref() if ref is not None else None

    @forked.setter
    def forked(self, rows: _Rows | None) -> None:
        self._forked = weakref.ref(rows) if rows is not None else None

    @property
    def fresh(self) -> bool:
        return self.cache is None

    @property
    def tokens(self) -> list[int]:
        """the sequence's ids, in order"""
        return [*self.ids, self._pending] if self._pending is not None else list(self.ids)

    def __len__(self) -> int:
        return len(self.ids) + (self._pending is not None)

    def _called(self, tag: PassTag) -> None:
        if self.engine is not None:
            self.engine._called(tag)

    def _bound(self) -> StreamedTextModel:
        if self.engine is None:
            raise ValueError("this session is not bound to an engine: make it with model.session()")
        self._unforked()
        return self.engine

    def _unforked(self, owner: _Rows | None = None) -> None:
        """a ValueError for a session a fork or a batch holds (but `owner`, writing its own rows back). Checked again
        under the decode lock, where a fork takes the session: a check before it could pass as another thread
        forks it"""
        held = self.forked
        if held is not None and held is not owner:
            raise ValueError("this session is forked: `keep` a row or `close` the branches first")

    def _flush(self, eng: _State) -> None:
        """the recurrent states an MLX decode kept in its graph, written into the cache's tensors"""
        if self.cache is not None and getattr(eng, "mlx", None) is not None:
            eng._mlx_flush_states(self.cache)

    def _snap(self, eng: _State) -> dict[int, LinSnap] | None:
        """a hybrid's recurrent states as they stand, copies; None for a dense model or an empty session"""
        cache = self.cache
        if cache is None or LayerKind.LINEAR not in eng.layer_types:
            return None
        from .engine.generate import lin_layer

        return {
            i: eng._lin_snap(lin_layer(cache.layers[i])) for i in range(eng.L) if eng.layer_types[i] == LayerKind.LINEAR
        }

    @contextlib.contextmanager
    def _txn(self, eng: _State, states: bool = True, owner: _Rows | None = None) -> Iterator[_Txn]:
        """One change, on the decode's thread under its lock: committed, or rolled back however the block is left.
        `states`: the recurrent states noted at the point, for a change that moves them. `owner`: the fork or batch
        holding the session, writing its own rows back."""
        self._unforked(owner)
        if self._active is not None:
            raise RuntimeError("a session change inside another: the session is mid-change")
        t = self._active = _Txn(self, eng, states)
        try:
            yield t
        finally:
            self._active = None
            if not t.done:
                t.rollback()

    def feed(self, ids: Tokens, last_only: bool = False, taps: Sequence[int] = ()) -> Step:
        """Append `ids` to the sequence: a `Step` of the logits after each of them, [T, V] in float32 - the
        building block of a loop of your own: feed a draft, read its logits, `rewind` what you reject.
        `last_only`: the last one's alone, [1, V], and the head run on that row alone (a long prelude's [T, V]
        is gigabytes). `taps`: also those layers' states at each fed position, {layer: [T, H]} float32 - layer i's
        the residual stream leaving block i, before the final norm (transformers' `hidden_states[i + 1]`)."""
        eng = self._bound()
        new = [int(t) for t in ids]
        if not new:
            raise ValueError("feed needs at least one token")
        lead = 0 if self._pending is None else 1
        want = tuple(int(i) % eng.L for i in taps)

        def run() -> Step:
            with self._txn(eng) as t:
                out, hidden = self._append(t, new, last_only, want)
            return Step(out if last_only else out[lead:], {i: h[lead:] for i, h in hidden.items()})

        return eng._serial(run)

    def next_logits(self) -> torch.Tensor:
        """The next token's logits [V] float32: `logits`, the token a decode drew last fed first when one waits
        (a pass, so a call rather than a read)"""
        eng = self._bound()
        if self.cache is None and self._pending is None:
            raise ValueError("an empty session has nothing to go on from: feed it a prompt first")
        if self._pending is not None:
            eng._serial(self._settle, eng)
        assert self.logits is not None
        return self.logits

    def _append(
        self, t: _Txn, new: list[int], last_only: bool = False, taps: Sequence[int] = ()
    ) -> tuple[torch.Tensor, Taps]:
        """the pending token and `new` into the cache and committed: the logits after each, [T, V] float32 (the
        last one's, [1, V], with `last_only`), and the `taps` layers' states at each, {layer: [T, H]}"""
        eng = t.eng
        new = [self._pending, *new] if self._pending is not None else new
        cache = self.cache if self.cache is not None else eng.new_cache()
        seen: dict[int, list[torch.Tensor]] = {i: [] for i in taps}

        def keep(i: int, h: torch.Tensor) -> None:
            if i in seen:
                seen[i].append(h[0].float().cpu())

        import torch

        from .engine.forward import PREFILL_MIN_ROWS

        hook = keep if taps else None
        if len(new) > min(PREFILL_MIN_ROWS, int(eng.prefill_chunk or PREFILL_MIN_ROWS)):
            # a long feed goes in the chunks the free memory prices, as a prompt's prefill does
            logits = eng._prefill(torch.tensor([new]), cache, on_layer=hook, last_only=last_only)
        else:
            logits = eng.forward([new], cache=cache, last_only=last_only, on_layer=hook)
        assert logits is not None
        out = logits[0].float().cpu()
        # the drafter's rows no longer follow the cache: the commit lets them go
        t.commit([*self.ids, *new], logits=out[-1].clone(), cache=cache)
        return out, {i: hs[0] if len(hs) == 1 else _cat(hs) for i, hs in seen.items()}

    def _settle(self, eng: StreamedTextModel, owner: _Rows | None = None) -> None:
        """every token in the cache and the next token's logits in hand, `Ready` (what a fork starts from): a
        pending token fed"""
        if self.cache is None and self._pending is None:
            raise ValueError("an empty session has nothing to go on from: feed it a prompt first")
        if self._pending is not None:
            with self._txn(eng, owner=owner) as t:
                self._append(t, [], last_only=True)

    def mark(self) -> Mark:
        """this point of the sequence, to `rewind` to later"""
        eng = self._bound()
        if self.cache is None or LayerKind.LINEAR not in eng.layer_types:
            return Mark(len(self.ids), {}, self._pending, self.logits, tuple(self.tokens))

        def run() -> Mark:
            self._unforked()
            self._flush(eng)
            return Mark(len(self.ids), self._snap(eng) or {}, self._pending, self.logits, tuple(self.tokens))

        return eng._serial(run)

    def rewind(self, mark: Mark) -> None:
        """Back to `mark`: everything after it is dropped from the tokens and the cache, a hybrid's recurrent
        states restored as they were there. A decode from here is the one a fresh session at that point gives."""
        eng = self._bound()
        if mark.n + (mark.pending is not None) > len(self):
            raise ValueError(f"a mark at {mark.n} is past the session's {len(self)} tokens")
        if mark.path is not None and tuple(self.tokens[: len(mark.path)]) != mark.path:
            raise ValueError(
                "this mark is not on the session's path: the tokens before it have changed since it was taken"
            )
        hybrid = LayerKind.LINEAR in eng.layer_types
        if hybrid and self.cache is not None and mark.n and not mark.states:
            raise ValueError("this mark was taken before the session had a cache: rewind to it from a fresh session")
        self._back(eng, mark)

    def _back(self, eng: StreamedTextModel, mark: Mark) -> None:
        """the session moved back to `mark`: the rollback of a transaction whose point is the mark"""

        def run() -> None:
            with self._txn(eng, states=False) as t:
                t.keep(_Point(mark.n, mark.pending, mark.logits, dict(mark.states), self.n_prompt))
            # left uncommitted on purpose: the rollback is the move

        eng._serial(run)

    def crop(self, n: int) -> None:
        """Keep the sequence's first `n` tokens, the rest dropped from the tokens and the cache; the n-th is left
        drawn and not fed (`next_logits()` feeds it), as a decode leaves its last. A hybrid's recurrent states
        cannot be cut back: `mark` the point beforehand and `rewind` to it."""
        eng = self._bound()
        n = int(n)
        if not 0 <= n <= len(self):
            raise ValueError(f"a crop to {n} tokens of a session of {len(self)}")
        if n == len(self):
            return
        if n and LayerKind.LINEAR in eng.layer_types:
            raise ValueError("a hybrid's recurrent states cannot be cut back: mark the point, then rewind to it")
        toks = self.tokens
        self._back(eng, Mark(n - 1, {}, toks[n - 1], None) if n else Mark(0))

    def rows(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention layer `layer`'s keys and values [1, kv heads, positions, head dim], copies of your own: the
        cache's rows as they stand (a drawn token not yet fed is not among them)"""
        eng = self._bound()
        i = int(layer)
        if not 0 <= i < eng.L:
            raise IndexError(f"layer {layer} of a model of {eng.L}")
        if eng.layer_types[i] == LayerKind.LINEAR:
            raise ValueError(f"layer {i} is a linear-attention layer: its state is recurrent, it has no rows")
        cache = self.cache
        if cache is None:
            raise ValueError("the session has no cache yet: feed it a prompt first")

        def run() -> tuple[torch.Tensor, torch.Tensor]:
            import torch
            from transformers.cache_utils import CacheLayerMixin

            self._unforked()
            self._flush(eng)
            cl = cache.layers[i]
            assert isinstance(cl, CacheLayerMixin) and cl.keys is not None and cl.values is not None
            # made outside inference mode, so the caller can write to them
            with torch.inference_mode(False):
                return cl.keys.clone(), cl.values.clone()

        return eng._serial(run)

    def sync(self, ids: Tokens) -> torch.Tensor:
        """Make the sequence `ids`, keeping what of the cache a new prompt would reuse and feeding the rest; returns
        the next token's logits [V]"""
        eng = self._bound()
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("sync needs at least one token")

        def run() -> torch.Tensor:
            with self._txn(eng, states=False) as t:
                _cache, reuse, _anchored = self._reuse(t, ids, whole=False)
                return self._append(t, ids[reuse:], last_only=True)[0][-1]

        return eng._serial(run)

    def fork(self, n: int) -> Branches:
        """`n` rows going on from here together (`btb.Branches`), sharing this session's cache, which holds
        still until one of them is kept"""
        from .engine.branches import Branches

        return Branches(self, n)

    def generate(self, max_new: int | None = None, **kw: Unpack[GenerateArgs]) -> RowGeneration:
        """Decode on from where the sequence stands (`model.generate` with this session; its keyword arguments)"""
        if not len(self):
            raise ValueError("an empty session has nothing to go on from: feed it a prompt first")
        return self._bound().generate(self.tokens, max_new, session=self, **kw)

    def _match(self, prompt: Tokens) -> int:
        """The prefix the new prompt shares with the cache's tokens (at most n - 1); learns `tail`, the distance
        before the previous prompt's end at which its re-rendering diverged."""
        n = len(prompt)
        lim = min(len(self.ids), n - 1)
        m = 0
        while m < lim and prompt[m] == self.ids[m]:
            m += 1
        if self.n_prompt and 0 < m < self.n_prompt <= len(self.ids) and self.n_prompt - m <= TAIL_MAX:
            self.tail = int(self.n_prompt - m)
        return m

    def _anchored(self, m: int) -> Anchor | None:
        """the latest DeltaNet snapshot at or before the matched prefix, None when there is none"""
        best = None
        for a in self.anchor:
            if 0 < a["n"] <= m and (best is None or a["n"] > best["n"]):
                best = a
        return best

    def _reuse(self, t: _Txn, prompt: Tokens, whole: bool) -> tuple[KvCache | None, int, Anchor | None]:
        """What `prompt` keeps of the session, inside transaction `t`: (cache, rows kept, the anchor restored), or
        (None, 0, None). The session is cut to what it keeps and `t`'s point moved there when that is less than
        all of it: an extension of the tokens keeps every row (the point where the session stands, its recurrent
        states noted); a dense cache parting from the prompt at m keeps m rows (the point `Pending` on the m-th
        token); a hybrid's states come back from the latest anchor inside the shared prefix (the point that anchor,
        `Ready` on its logits); nothing kept is the point `Empty`. With `whole`, a prompt that is the session's
        tokens, fed and their logits in hand, keeps every row and the logits wait in `_held` for the prefill."""
        eng = t.eng
        self._held = None
        prompt = [int(x) for x in prompt]
        if self.cache is None:
            return None, 0, None
        if (
            whole
            and self._pending is None
            and self.logits is not None
            and getattr(eng, "mlx", None) is None
            and prompt == self.ids
        ):
            t.point.states = self._snap(eng)
            self._held, self.logits = self.logits, None
            return self.cache, len(self.ids), None
        m = self._match(prompt)
        if 0 < len(self.ids) and m == len(self.ids) and len(prompt) > len(self.ids):
            # the prompt extends the tokens (a pending token among them): every row kept, the point where it stands
            t.point.states = self._snap(eng)
            self._pending, self.logits = None, None
            return self.cache, len(self.ids), None
        if LayerKind.LINEAR in eng.layer_types:
            anchored = self._anchored(m)
            if anchored is not None and anchored["logits"] is not None:
                from .engine.generate import lin_layer

                a = int(anchored["n"])
                for i, snap in anchored["states"].items():
                    eng._lin_restore(lin_layer(self.cache.layers[i]), snap)
                crop(self.cache, eng, a)
                del self.ids[a:]
                self.anchor = [x for x in self.anchor if x["n"] <= a]
                self._pending, self.logits = None, None
                t.keep(_Point(a, None, anchored["logits"], dict(anchored["states"]), min(self.n_prompt, a)))
                return self.cache, a, anchored
        elif m > 0:
            # m rows kept for the prefill. Should the call fail: where the prompt goes on as the session did past
            # them (it re-runs the session's own tokens, replacing nothing), the session is left `Pending` on its
            # next token - no token lost; where the prompt parts from it at m, `Pending` on the m-th (the state a
            # crop to m leaves) - the rows past m were the call's to replace
            toks = self.tokens
            if m < len(toks) and m < len(prompt) and prompt[m] == toks[m]:
                t.keep(_Point(m, toks[m], None, None, min(self.n_prompt, m)))
            else:
                t.keep(_Point(m - 1, self.ids[m - 1], None, None, min(self.n_prompt, m - 1)))
            crop(self.cache, eng, m)
            del self.ids[m:]
            self.anchor = [x for x in self.anchor if x["n"] <= m]
            self._pending, self.logits = None, None
            return self.cache, m, None
        # nothing of the last conversation serves this one: its cache goes now, not when the new turn's commit
        # replaces it, or the new prefill runs beside a full cache of the old (7 GB twice at 40k); the drafter's own
        # cache, which the engine holds, goes with it (a gigabyte at 120k rows)
        # Should the call fail, the session keeps what the prompt still shares with it: its first token, pending over
        # no rows, where the prompt starts as the session does (a one-token session re-run, a prompt too short for a
        # row to be kept); else nothing
        toks = self.tokens
        first = toks[0] if toks and prompt and prompt[0] == toks[0] else None
        dr = self.dr
        self.cache, self.anchor, self.ids, self.n_prompt = None, [], [], 0
        self._pending, self.logits = None, None
        self.dr, self.dr_len, self.pend_h = None, 0, None
        if dr is not None and hasattr(dr, "reset"):
            dr.reset()
        t.keep(_Point(0, first, None, None, 0))
        return None, 0, None

    # -- a decode over the session (`generate(session=...)`): the transaction spans the engine's decode loop --------

    @contextlib.contextmanager
    def _decoding(self, eng: _State) -> Iterator[None]:
        """a decode's transaction: begun by `_begin_decode` inside the loop, committed by `_commit_decode`, rolled
        back however the block is left otherwise"""
        self._unforked()
        if self._active is not None:
            raise RuntimeError("a session change inside another: the session is mid-change")
        try:
            yield
        finally:
            t, self._active = self._active, None
            self._held = None
            if t is not None and not t.done:
                t.rollback()

    def _begin_decode(
        self, engine: _State, prompt: Tokens, whole: bool = False
    ) -> tuple[KvCache | None, int, Anchor | None]:
        """What a decode of `prompt` reuses: (cache, rows reused, the anchor restored), the transaction begun"""
        self._unforked()
        if self._active is not None:
            raise RuntimeError("a decode over a session begins inside `_decoding`, once")
        t = self._active = _Txn(self, engine, states=False)
        return self._reuse(t, prompt, whole)

    def _commit_decode(
        self,
        prompt: Tokens,
        out: Tokens,
        cache: KvCache,
        anchors: Sequence[Anchor] | None,
        dr: MTPDrafter | None = None,
        dr_len: int = 0,
        pend_h: torch.Tensor | None = None,
    ) -> None:
        """what the next call finds: the cache holds the prompt and the answer but its last token, `Pending`"""
        t = self._active
        if t is None:
            raise RuntimeError("a decode commits the transaction `_begin_decode` began")
        t.commit(
            [*prompt, *out[:-1]],
            pending=int(out[-1]),
            cache=cache,
            n_prompt=len(prompt),
            anchors=anchors or (),
            drafter=(dr, dr_len, pend_h),
        )

    # -- a fork's or a batch's row written back ------------------------------------------------------------------

    def _merge_row(
        self,
        eng: StreamedTextModel,
        owner: _Rows,
        toks: Sequence[int],
        write: Callable[[KvCache], None],
        pending: int | None,
        logits: torch.Tensor | None,
    ) -> None:
        """a row of `owner`'s committed: `write(cache)` appends its rows to the session's cache and puts its recurrent
        states in, the tokens `toks` follow, and the row's pending token or logits are the session's. A write failing
        part way is rolled back: the cache cut to where the session was lent, its recurrent states as they were"""
        if self.cache is None:
            raise ValueError("an empty session has nothing to decode: feed it a prompt first")
        with self._txn(eng, owner=owner) as t:
            write(self.cache)
            t.commit([*self.ids, *toks], pending=pending, logits=logits)
