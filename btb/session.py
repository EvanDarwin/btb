# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A sequence's state between calls - its tokens and its cache - what a new prompt reuses of it, and the moves a
caller makes on it: feed tokens, mark a point, rewind to one, fork into rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, Unpack, overload

from .kinds import LayerKind, Tokens

if TYPE_CHECKING:
    import torch

    from .engine.branches import Branches, _Rows
    from .engine.cache import KvCache, LinearStates
    from .engine.drafter import MTPDrafter
    from .engine.hooks import Taps
    from .engine.model import StreamedTextModel
    from .engine.state import _State
    from .engine.text import GenerateArgs, RowGeneration

# a chat template's generation tail is a handful of tokens (Qwen3's empty think block is four); a prompt that
# parts from the previous one further back than this is another conversation, and teaches no tail
TAIL_MAX = 64


class Anchor(TypedDict):
    """a hybrid's point to resume a later prompt from: the rows before it, its linear layers' states there, and the
    last layer's state at its last row (the drafter's start)"""

    n: int
    states: dict[int, LinearStates]
    h_last: torch.Tensor | None


@dataclass(frozen=True)
class Mark:
    """a point in a session to rewind to: the rows its cache held, a hybrid's recurrent states there (they cannot
    be cropped back, so they are kept), and the token drawn but not yet fed or the next token's logits"""

    n: int
    states: dict[int, LinearStates] = field(default_factory=dict)
    pending: int | None = None
    logits: torch.Tensor | None = None


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


class Session:
    """The tokens the cache holds and the cache, a hybrid's DeltaNet snapshots (`anchor`), the drafter's tail
    (`dr`, `dr_len`, `pend_h`) and the template's generation tail (`tail`, learned or given). Made by
    `model.session()`, it is bound to that engine and takes `feed`, `mark`, `rewind`, `crop`, `fork` and
    `generate`; any session can be passed as `generate(session=...)`. Cut and read the rows through the session
    (`crop`, `rows`), not `cache.layers`: a layer's tensors may live in memory btb moves or hands to another cache."""

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
        # the sequence's last token when a decode drew it and the cache does not hold it yet; else, after a
        # `feed`, the next token's logits [V]
        self.pending: int | None = None
        self.logits: torch.Tensor | None = None
        self.forked: _Rows | None = None  # the live `Branches` or `Batch` over this session, which holds it still

    @property
    def fresh(self) -> bool:
        return self.cache is None

    @property
    def tokens(self) -> list[int]:
        """the sequence's ids, in order"""
        return [*self.ids, self.pending] if self.pending is not None else list(self.ids)

    def __len__(self) -> int:
        return len(self.ids) + (self.pending is not None)

    def _bound(self) -> StreamedTextModel:
        if self.engine is None:
            raise ValueError("this session is not bound to an engine: make it with model.session()")
        if self.forked is not None:
            raise ValueError("this session is forked: `keep` a row or `close` the branches first")
        return self.engine

    def _flush(self, eng: StreamedTextModel) -> None:
        """the recurrent states an MLX decode kept in its graph, written into the cache's tensors"""
        if self.cache is not None and getattr(eng, "mlx", None) is not None:
            eng._mlx_flush_states(self.cache)

    @overload
    def feed(self, ids: Tokens, last_only: bool = False) -> torch.Tensor: ...

    @overload
    def feed(self, ids: Tokens, last_only: bool = False, *, taps: Sequence[int]) -> tuple[torch.Tensor, Taps]: ...

    def feed(
        self, ids: Tokens, last_only: bool = False, *, taps: Sequence[int] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, Taps]:
        """Append `ids` to the sequence and return the logits after each of them, [T, V] in float32 - the
        building block of a loop of your own: feed a draft, read its logits, `rewind` what you reject.
        `last_only`: the last one's alone, [1, V], and the head run on that row alone (a long prelude's [T, V]
        is gigabytes). `taps`: also those layers' states at each fed position, {layer: [T, H]} float32 - layer i's
        the residual stream leaving block i, before the final norm (transformers' `hidden_states[i + 1]`)."""
        eng = self._bound()
        new = [int(t) for t in ids]
        if not new:
            raise ValueError("feed needs at least one token")
        lead = 0 if self.pending is None else 1
        want = tuple(int(i) % eng.L for i in taps) if taps is not None else ()

        def run() -> tuple[torch.Tensor, Taps]:
            out, hidden = self._append(eng, new, last_only, want)
            return (out if last_only else out[lead:]), {i: h[lead:] for i, h in hidden.items()}

        logits, hidden = eng._serial(run)
        return logits if taps is None else (logits, hidden)

    def _append(
        self, eng: StreamedTextModel, new: list[int], last_only: bool = False, taps: Sequence[int] = ()
    ) -> tuple[torch.Tensor, Taps]:
        """the pending token and `new` into the cache: the logits after each, [T, V] float32 (the last one's,
        [1, V], with `last_only`), and the `taps` layers' states at each, {layer: [T, H]} (on the decode's thread,
        under its lock)"""
        new = [self.pending, *new] if self.pending is not None else new
        if self.cache is None:
            self.cache = eng.new_cache()
        seen: dict[int, list[torch.Tensor]] = {i: [] for i in taps}

        def keep(i: int, h: torch.Tensor) -> None:
            if i in seen:
                seen[i].append(h[0].float().cpu())

        import torch

        from .engine.forward import PREFILL_MIN_ROWS

        hook = keep if taps else None
        if len(new) > min(PREFILL_MIN_ROWS, int(eng.prefill_chunk or PREFILL_MIN_ROWS)):
            # a long feed goes in the chunks the free memory prices, as a prompt's prefill does
            logits = eng._prefill(torch.tensor([new]), self.cache, on_layer=hook, last_only=last_only)
        else:
            logits = eng.forward([new], cache=self.cache, last_only=last_only, on_layer=hook)
        assert logits is not None
        out = logits[0].float().cpu()
        self.ids.extend(new)
        self.n_prompt = len(self.ids)
        self.pending, self.logits = None, out[-1].clone()
        # the drafter's rows no longer follow the cache
        self.dr, self.dr_len, self.pend_h = None, 0, None
        return out, {i: hs[0] if len(hs) == 1 else _cat(hs) for i, hs in seen.items()}

    def _settle(self, eng: StreamedTextModel) -> None:
        """every token in the cache and the next token's logits in hand (what a fork starts from)"""
        if self.pending is None and self.logits is None:
            if LayerKind.LINEAR in eng.layer_types:
                raise ValueError("the next token's logits are not kept here: feed the next token, then fork")
            if self.cache is None:
                raise ValueError("an empty session has nothing to go on from: feed it a prompt first")
            # the last token again: its row cut and re-run
            crop(self.cache, eng, len(self.ids) - 1)
            self.pending = self.ids.pop()
        if self.pending is not None:
            self._append(eng, [], last_only=True)

    def mark(self) -> Mark:
        """this point of the sequence, to `rewind` to later"""
        eng = self._bound()
        cache = self.cache
        if cache is None or LayerKind.LINEAR not in eng.layer_types:
            return Mark(len(self.ids), {}, self.pending, self.logits)

        def run() -> Mark:
            self._flush(eng)
            states = {i: eng._lin_snap(cache.layers[i]) for i in range(eng.L) if eng.layer_types[i] == LayerKind.LINEAR}
            return Mark(len(self.ids), states, self.pending, self.logits)

        return eng._serial(run)

    def rewind(self, mark: Mark) -> None:
        """Back to `mark`: everything after it is dropped from the tokens and the cache, a hybrid's recurrent
        states restored as they were there. A decode from here is the one a fresh session at that point gives."""
        eng = self._bound()
        if mark.n + (mark.pending is not None) > len(self):
            raise ValueError(f"a mark at {mark.n} is past the session's {len(self)} tokens")
        hybrid = LayerKind.LINEAR in eng.layer_types
        if hybrid and self.cache is not None and mark.n and not mark.states:
            raise ValueError("this mark was taken before the session had a cache: rewind to it from a fresh session")

        def run() -> None:
            if mark.n == 0 or self.cache is None:
                self.cache, self.anchor = None, []
            else:
                self._flush(eng)
                crop(self.cache, eng, mark.n)
                for i, snap in mark.states.items():
                    eng._lin_restore(self.cache.layers[i], snap)
                self.anchor = [a for a in self.anchor if a["n"] <= mark.n]
            del self.ids[mark.n :]
            self.n_prompt = min(self.n_prompt, mark.n)
            self.pending, self.logits = mark.pending, mark.logits
            self.dr, self.dr_len, self.pend_h = None, 0, None

        eng._serial(run)

    def crop(self, n: int) -> None:
        """Keep the sequence's first `n` tokens, the rest dropped from the tokens and the cache. A hybrid's recurrent
        states cannot be cut back: `mark` the point beforehand and `rewind` to it."""
        eng = self._bound()
        n = int(n)
        if not 0 <= n <= len(self):
            raise ValueError(f"a crop to {n} tokens of a session of {len(self)}")
        if n == len(self):
            return
        if n and LayerKind.LINEAR in eng.layer_types:
            raise ValueError("a hybrid's recurrent states cannot be cut back: mark the point, then rewind to it")
        self.rewind(Mark(n))

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
            self._flush(eng)
            cache, reuse, _ = self.open(eng, ids)
            if cache is not None:
                del self.ids[reuse:]
                self.anchor = [a for a in self.anchor if a["n"] <= reuse]
            return self._append(eng, ids[reuse:], last_only=True)[0][-1]

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

    def match(self, prompt: Tokens) -> int:
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

    def anchored(self, m: int) -> Anchor | None:
        """the latest DeltaNet snapshot at or before the matched prefix, None when there is none"""
        best = None
        for a in self.anchor:
            if 0 < a["n"] <= m and (best is None or a["n"] > best["n"]):
                best = a
        return best

    def open(self, engine: _State, prompt: Tokens) -> tuple[KvCache | None, int, Anchor | None]:
        """What a new prompt reuses: (cache, rows reused, the snapshot restored), or (None, 0, None). A dense
        cache is cropped to the shared prefix; a hybrid's DeltaNet states cannot be cropped, so they come
        back from the latest snapshot inside it."""
        if self.forked is not None:
            raise ValueError("this session is forked: `keep` a row or `close` the branches first")
        self.pending, self.logits = None, None
        if self.cache is None:
            return None, 0, None
        n = len(prompt)
        m = self.match(prompt)
        cache, reuse, anchored = None, 0, None
        if 0 < len(self.ids) < n and m == len(self.ids):
            cache, reuse = self.cache, len(self.ids)
        elif LayerKind.LINEAR in engine.layer_types:
            anchored = self.anchored(m)
            if anchored is not None:
                cache, reuse = self.cache, int(anchored["n"])
                for i, snap in anchored["states"].items():
                    engine._lin_restore(cache.layers[i], snap)
        elif m > 0:
            cache, reuse = self.cache, m
        if cache is None:
            # nothing of the last conversation serves this one: its cache goes now, not when the new turn's
            # `keep` replaces it, or the new prefill runs beside a full cache of the old (7 GB twice at 40k); the
            # drafter's own cache, which the engine holds, goes with it (a gigabyte at 120k rows)
            dr = self.dr
            self.cache, self.anchor, self.ids, self.n_prompt = None, [], [], 0
            self.dr, self.dr_len, self.pend_h = None, 0, None
            if dr is not None and hasattr(dr, "reset"):
                dr.reset()
        if cache is not None:
            crop(cache, engine, reuse)
        return cache, reuse, anchored

    def keep(
        self,
        prompt: Tokens,
        out: Tokens,
        cache: KvCache,
        anchors: Sequence[Anchor] | None,
        dr: MTPDrafter | None = None,
        dr_len: int = 0,
        pend_h: torch.Tensor | None = None,
    ) -> None:
        """what the next call finds: the cache holds the prompt and the answer but its last token"""
        self.ids = [*prompt, *out[:-1]]
        self.pending, self.logits = (int(out[-1]) if len(out) else None), None
        self.n_prompt = len(prompt)
        self.cache = cache
        self.anchor = list(anchors or ())
        self.dr, self.dr_len = dr, int(dr_len)
        self.pend_h = pend_h.detach().clone() if pend_h is not None else None
