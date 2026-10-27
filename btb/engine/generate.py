# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Decoding: greedy and speculative (the tree), sessions that keep a cache between turns, the DeltaNet state
snapshots, and `serve` over ragged prompts in scheduler-sized epochs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, TypeAlias, overload

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin, LinearAttentionLayer

from .. import mlx as mlxdev
from .. import trace
from ..draft import NGramProposer, SpanBank, Spans
from ..kinds import PROPOSER_TAG, LayerKind, PassTag, Proposer, TokenRows, Tokens
from ..options import Device
from ..sampling import GREEDY, Pick, Sampling, Verify
from ..session import TAIL_MAX, Session
from .cache import linear_layer
from .drafter import MTPDrafter, SampledTree
from .fused import DELTA_BLOCK
from .hooks import Hooks, LayerTap, OnToken
from .paged import PagedCache
from .state import _State

if TYPE_CHECKING:
    from typing import TypeIs

    import mlx.core as mx_

    from ..session import Anchor
    from .cache import CacheLayer, KvCache
    from .mlx_forward import OnLayer
    from .model import StreamedTextModel
    from .propose import ChainProposer, ChainSource
    from .text import GenerateStats

# a DeltaNet layer's cache layer, as transformers keeps it: its conv and recurrent states each by index, None until set
LinLayer = LinearAttentionCacheLayerMixin
# a DeltaNet layer's (conv, recurrent) states copied, the ones set, for a later `_lin_restore`
LinSnap = tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
# the committed rows a run of plain steps leaves the MTP drafter at most: past them it takes them without drafting,
# so a long run neither regathers its hidden rows every step nor hands the next draft one outsized step
PEND_ROWS = 16
# what the plain loop returns: one row's tokens, a list per row for several, or under a session the row's tokens
# and its counts
GreedyOut: TypeAlias = "list[int] | list[list[int]] | tuple[list[int], GenerateStats]"


class VerifyTree(NamedTuple):
    """drafts as a verify pass's tree: node 0 the root (the pass's first row), node j > 0 drafting guesses[j - 1]
    under parents[j] at depth[j]; each node's children in node order, and each node's source"""

    guesses: list[int]
    parents: list[int]
    depth: list[int]
    children: dict[int, list[int]]
    tags: list[str]


@dataclass(frozen=True)
class Picked:
    """a verify pass that picked in its graph (the fused MLX path): each row's pick [T], a token or, under a
    `Verify`, a packed outcome"""

    ids: torch.Tensor


@dataclass(frozen=True)
class Scored:
    """a verify pass that left the pick to its caller: each row's logits [T, V]"""

    logits: torch.Tensor


def _verified(out: torch.Tensor | None) -> Picked | Scored:
    """a one-sequence verify pass's result told apart once: `forward` hands back integer ids where it picked in
    its graph, the logits otherwise"""
    if out is None:
        raise RuntimeError("a verify pass through the head returned nothing")
    rows = out[0]
    return Picked(rows) if rows.dtype in (torch.int32, torch.int64) else Scored(rows)


@dataclass(frozen=True)
class DraftTail:
    """the committed rows the MTP drafter has yet to take: their tokens, and the target's last-layer rows [1, T, H]
    under them and under the token after them, one row more. The drafter holds every row before them"""

    toks: list[int]
    h: torch.Tensor

    def __post_init__(self) -> None:
        if int(self.h.shape[1]) != len(self.toks) + 1:
            raise ValueError(f"[mtp] {len(self.toks)} pending tokens over {int(self.h.shape[1])} hidden rows")


def lin_layer(cl: CacheLayer) -> LinLayer:
    """`cl` as the helpers below take it; a TypeError for a layer that is not a DeltaNet's"""
    return linear_layer(cl)


def _block(n: int) -> int:
    """the last block's end (a multiple of `DELTA_BLOCK`) at or before position n"""
    return max(0, int(n)) // DELTA_BLOCK * DELTA_BLOCK


def _states_bytes(cl: LinLayer) -> int:
    """what a snapshot of a DeltaNet layer's states copies (`_lin_snap`)"""
    total = 0
    for s in (cl.conv_states, cl.recurrent_states):
        for v in s.values():
            if v is not None:
                total += v.numel() * v.element_size()
    return total


def _lin(cl: LinLayer) -> tuple[torch.Tensor, torch.Tensor]:
    """a DeltaNet layer's (conv, recurrent) states; an error for a layer whose prefill has not set them"""
    c, r = cl.conv_states[0], cl.recurrent_states[0]
    if c is None or r is None:
        raise RuntimeError(f"[delta] a {type(cl).__name__}'s states were read before a pass set them")
    return c, r


def _verify_of(smp: Sampling, T: int, tree: SampledTree) -> Pick:
    """the verify pick of a sampled tree of `T` rows: row j's draws in draw order and its drafter distribution
    (the root's under -1, node j's under j - 1), rows without one flagged as point masses; the plain sampling
    when no row has one. The rows are all torch's or all MLX's, as the drafter made them"""
    kids = [[int(t) for t in tree.draws.get(j - 1, [])] for j in range(T)]
    rows = [tree.probs.get(j - 1) for j in range(T)]
    hasq = [r is not None for r in rows]
    some = next((r for r in rows if r is not None), None)
    if some is None:
        return smp
    q: torch.Tensor | mx_.array
    if isinstance(some, torch.Tensor):
        q = torch.stack([r if isinstance(r, torch.Tensor) else torch.zeros_like(some) for r in rows])
    else:
        m = mlxdev.mx
        q = m.stack([r if isinstance(r, m.array) else m.zeros_like(some) for r in rows])
    return Verify(smp, q, kids, hasq)


def _chains_tree(chains: Sequence[tuple[Tokens, str]], budget: int) -> VerifyTree:
    """Chains merged by shared prefix into a tree: node 0 the root, node j > 0 carrying guesses[j - 1]; at most
    `budget` nodes beyond the root."""
    guesses: list[int] = []
    parents, depth, tags = [-1], [0], ["root"]
    children: dict[int, list[int]] = {}
    for toks, tag in chains:
        node = 0
        for d, t in enumerate(toks):
            t = int(t)
            nxt = next((c for c in children.get(node, []) if guesses[c - 1] == t), None)
            if nxt is None:
                if len(guesses) >= budget:
                    break
                nxt = len(guesses) + 1
                guesses.append(t)
                parents.append(node)
                depth.append(d + 1)
                tags.append(tag)
                children.setdefault(node, []).append(nxt)
            node = nxt
    return VerifyTree(guesses, parents, depth, children, tags)


class PassClock:
    """A speculative pass's seconds, as the pricer takes them. On a card, the time the card spends between two events
    on the stream the pass is launched on (the card graph's side streams join it): the pass's own work and the gaps
    while the host launches it, not the work queued before it. Timed from the host, a pass's figure took whatever the
    card still had in hand - the last commit's writes, the drafter's - so a one-row step read slow and a wide pass
    fast or slow by what preceded it: 0.17 to 3.3 one-row steps between answers on one card, and where no width came
    within a quarter over a step the answers took plain steps throughout. Elsewhere the wall clock: the host computes
    as it goes. Read after the pass's results reached the host, when both events have completed."""

    def __init__(self, dev: torch.device) -> None:
        self.dev = dev
        self.card = dev.type == "cuda"
        self.ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if self.card else None
        self.t0 = self.t1 = 0.0

    def start(self) -> None:
        if self.ev is not None:
            self.ev[0].record(torch.cuda.current_stream(self.dev))
        self.t0 = time.perf_counter()

    def stop(self) -> None:
        if self.ev is not None:
            self.ev[1].record(torch.cuda.current_stream(self.dev))
        self.t1 = time.perf_counter()

    def seconds(self) -> float:
        if self.ev is not None:
            self.ev[1].synchronize()  # done already where the results were read; a wait otherwise
            return self.ev[0].elapsed_time(self.ev[1]) / 1e3
        return self.t1 - self.t0


def _children(parents: Sequence[int]) -> dict[int, list[int]]:
    """each tree node's children in node order, from the nodes' parents (node 0 the root)"""
    children: dict[int, list[int]] = {}
    for j in range(1, len(parents)):
        children.setdefault(parents[j], []).append(j)
    return children


def _one_row(out: list[int] | list[list[int]]) -> TypeIs[list[int]]:
    """whether the plain loop handed back one row's tokens, a flat list, rather than a list per row"""
    return not out or isinstance(out[0], int)


def as_rows(out: GreedyOut) -> list[list[int]]:
    """a call's tokens from the plain loop without a session, a list per row (one row's flat list wrapped)"""
    if isinstance(out, tuple):
        raise TypeError("the plain loop returned a session's counts to a call without a session")
    return [out] if _one_row(out) else out


def _source_tag(where: ChainSource) -> str:
    """a proposed chain's source as the census and the pricing tag it: a banked span's tag, a draft model's, `self`
    for the n-gram corpus's own continuation"""
    if isinstance(where, tuple):
        return where[0]
    return where if isinstance(where, str) else "self"


class _GenerateMixin(_State):
    _lin = staticmethod(_lin)

    @staticmethod
    def _lin_snap(cl: LinLayer, dev: torch.device | None = None) -> LinSnap:
        """a DeltaNet layer's states copied - where they lie, or straight to `dev` (a snapshot the prefix tree keeps
        in RAM, never a copy on the card first)"""

        def one(v: torch.Tensor) -> torch.Tensor:
            return v.clone() if dev is None else v.to(dev, copy=True)

        def copy(s: dict[int, torch.Tensor | None]) -> dict[int, torch.Tensor]:
            return {k: one(v) for k, v in s.items() if v is not None}

        return copy(cl.conv_states), copy(cl.recurrent_states)

    def _lin_home(self, i: int) -> torch.device:
        """where linear layer i's recurrent states live: the host's for a layer the host runs, else the engine's
        device - where a snapshot restored into a fresh cache puts them"""
        return torch.device("cpu") if i in self.host or self.dev.type == Device.CPU else torch.device(self.dev)

    @staticmethod
    def _lin_restore(cl: LinLayer, snap: LinSnap, dev: torch.device | None = None) -> None:
        """`snap` into the layer's states: copied into the slots it holds, a slot it has not filled made on `dev`
        (else where the snapshot lies)"""
        for conv_slot, have, states in ((True, cl.conv_states, snap[0]), (False, cl.recurrent_states, snap[1])):
            for k, v in states.items():
                cur = have.get(k)
                if cur is not None and cur.dtype == v.dtype and cur.shape == v.shape:
                    cur.copy_(v)
                    continue
                # a slot the layer has not filled yet (a host prefill leaves some to the first step; a fresh cache
                # opening on the tree's snapshot, all), or one a step has widened since (Qwen4's holds its states
                # float32): a copy of the snapshot's own, in its dtype, which the next pass computes in, on `dev` -
                # never a view of the snapshot's (a later pass writes the layer's states in place). One not filled
                # yet through transformers' own setters, so the next pass goes on from the state, not from none
                own = v.to(dev or v.device, copy=True)
                if cur is not None:
                    have[k] = own
                elif not conv_slot:
                    cl.update_recurrent_state(own, k)
                elif isinstance(cl, LinearAttentionLayer):
                    cl.update_conv_state(own, k, conv_kernel_size=int(v.shape[-1]))
                else:
                    cl.update_conv_state(own, k)

    def _session_prefill(
        self, ids: torch.Tensor, cache: KvCache, reuse: int, session: Session | None, on_layer: OnLayer | None = None
    ) -> tuple[torch.Tensor, list[Anchor] | None]:
        """Prefill ids[:, reuse:]: the logits [1, 1, V] of its last row; for a hybrid in a session, the DeltaNet state
        snapshots the next turn can resume from (the prompt's end, and the point the re-rendering will diverge at once
        the tail is known)."""
        n = int(ids.shape[1])
        hybrid = LayerKind.LINEAR in self.layer_types
        # every row reused (`Session._begin_decode`'s whole): the session's logits for its last token, nothing to run
        held: torch.Tensor | None = None
        if session is not None and reuse == n:
            held, session._held = session._held, None
            assert held is not None, "a prompt the cache holds whole comes with its logits"
            held = held.view(1, 1, -1)
        if session is None or not hybrid:
            if held is not None:
                return held, None
            logits: torch.Tensor = self._prefill(ids[:, reuse:], cache, on_layer=on_layer)
            return logits, None
        hs: list[torch.Tensor] = []

        def snap(at: int, logits: torch.Tensor) -> Anchor:
            return self._anchor(cache, at, logits, hs[-1][:, -1:].detach().clone() if hs else None)

        def gather(i: int, h: torch.Tensor) -> None:
            # the last layer's rows are gathered across the chunks for one call below; the others go straight on
            if i == self.L - 1:
                hs.append(h)
            elif on_layer is not None:
                on_layer(i, h)

        hook = gather if on_layer is not None else None
        paged = cache if isinstance(cache, PagedCache) else None
        if held is not None:
            # a paged cache keeps a hybrid's states at a block's end alone (`_prefill_at_blocks`): the prompt's is none
            return held, ([] if paged is not None else [snap(n, held)])
        anchors: list[Anchor]
        if paged is not None:
            logits, anchors = self._prefill_at_blocks(ids, paged, reuse, int(session.tail), hook, snap)
        else:
            anchors = []
            d = int(session.tail)
            cut = n - d
            if d > 0 and reuse < cut:
                at_cut = self._prefill(ids[:, reuse:cut], cache, on_layer=hook)
                anchors.append(snap(cut, at_cut))
                logits = self._prefill(ids[:, cut:], cache, on_layer=hook)
            else:
                logits = self._prefill(ids[:, reuse:], cache, on_layer=hook)
            anchors.append(snap(n, logits))
        if on_layer is not None:
            on_layer(self.L - 1, hs[0] if len(hs) == 1 else torch.cat(hs, dim=1))
        return logits, anchors

    def _anchor(self, cache: KvCache, at: int, logits: torch.Tensor, h_last: torch.Tensor | None) -> Anchor:
        """a hybrid's point to resume from (`Anchor`): the DeltaNet layers' states at position `at`, copies, and the
        next token's logits there - the last row of `logits` - so the point is a state of its own (`Ready`), which a
        failed call goes back to. A paged cache's go to the prefix tree, kept in RAM: copied there, a card layer's too"""
        host = torch.device("cpu") if isinstance(cache, PagedCache) else None
        return {
            "n": at,
            "logits": logits[0, -1].float().cpu().clone(),
            "states": {
                i: self._lin_snap(lin_layer(cache.layers[i]), host)
                for i in range(self.L)
                if self.layer_types[i] == LayerKind.LINEAR
            },
            "h_last": h_last if h_last is None or host is None else h_last.to(host),
        }

    def _prefill_at_blocks(
        self,
        ids: torch.Tensor,
        cache: PagedCache,
        reuse: int,
        tail: int,
        hook: OnLayer | None,
        snap: Callable[[int, torch.Tensor], Anchor],
        last_only: bool = True,
    ) -> tuple[torch.Tensor, list[Anchor]]:
        """A hybrid's prompt into a paged cache from `reuse` - 0, or a block's end (`DELTA_BLOCK`) its states were kept
        at - cut at the block ends a later prompt resumes from, its states kept at each for the tree (`snap`): the last
        before the prompt's last token (the prompt again), and the last before where the next turn's re-rendering
        parts from it (`tail` back; TAIL_MAX, the furthest a tail is learned, until it is known). Every call past the
        prompt's first runs as a later chunk of the prompt (`_batched_cont`: its DeltaNet layers through their chunked
        rule, not a step's), and each starts at a block's end, so its rows and states are the prompt's prefilled whole
        and cold: what a snapshot handed to another conversation must be. A snapshot the RAM has no room for, even
        with the tree's least used let go, is not kept (`PrefixCache.snap_room`): a later prompt resumes earlier.
        The last row's logits, or every row's from `reuse` on (the calls' joined) without `last_only`"""
        n = int(ids.shape[1])
        back = int(tail) or TAIL_MAX
        cuts = sorted({c for c in (_block(n - 1), _block(n - back)) if reuse < c < n})
        linear = [i for i in range(self.L) if self.layer_types[i] == LayerKind.LINEAR]
        anchors: list[Anchor] = []
        outs: list[torch.Tensor] = []
        a = int(reuse)
        was = self._batched_cont
        try:
            for b in (*cuts, n):
                self._batched_cont = a > 0
                out: torch.Tensor = self._prefill(ids[:, a:b], cache, on_layer=hook, last_only=last_only)
                outs.append(out)
                cache.prefilled(a, b)
                if b < n and cache.table.cold == b:
                    size = sum(_states_bytes(lin_layer(cache.layers[i])) for i in linear)
                    if cache.prefix.snap_room(size + 4 * int(out.shape[-1])):
                        anchors.append(snap(b, out))
                a = b
        finally:
            self._batched_cont = was
        return (outs[-1] if last_only or len(outs) == 1 else torch.cat(outs, dim=1)), anchors

    @staticmethod
    def _path_tokens(j: int, guesses: Sequence[int], parents: Sequence[int] | None) -> list[int]:
        """the drafted tokens between the pass's root and node j: its ancestors' in a tree, the chain's first j
        otherwise"""
        if parents is None:
            return list(guesses[:j])
        out: list[int] = []
        while j > 0:
            out.append(int(guesses[j - 1]))
            j = parents[j]
        return out[::-1]

    @staticmethod
    def _hook_commit(
        hk: Hooks, logits: torch.Tensor | None, taps: dict[int, torch.Tensor], node: int, token: int
    ) -> None:
        """a committed token's log-probability off its node's row, and the tapped layers' state at that row"""
        if hk.needs_logits:
            assert logits is not None  # a hooked pick runs no pick in the graph: the pass hands back its logits
            hk.record(0, logits[node], token)
        for i, h in taps.items():
            hk.tap(0, i, h[0, node])

    @staticmethod
    def _lin_set(cl: LinLayer, conv: torch.Tensor, rec: torch.Tensor) -> None:
        cl.conv_states[0] = conv
        cl.recurrent_states[0] = rec

    def mtp_drafter(self) -> MTPDrafter:
        """the family's drafter over the checkpoint's drafting head, built on first use and kept"""
        if getattr(self, "aj", None) is None:
            cls = self.fam.drafter_cls()
            if cls is None:
                raise NotImplementedError(
                    f"btb has no MTP drafter for {self.fam.name}: speculate with the n-gram proposer"
                )
            from .model import StreamedTextModel

            if not isinstance(self, StreamedTextModel):
                raise TypeError(f"an MTP drafter drafts for an engine, not a {type(self).__name__}")
            t0 = time.time()
            self.aj = cls(self, weights=getattr(self, "drafter_weights", None), dev=getattr(self, "drafter_dev", None))
            self.aj.build_s = time.time() - t0
        assert self.aj is not None  # set above whenever it was unset or None
        return self.aj

    @torch.inference_mode()
    def generate_speculative(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        v_max: int = 8,
        n_min: int = 2,
        n_max: int = 4,
        on_token: OnToken | None = None,
        # the str half is the public boundary (a CLI/serve value, a test); Proposer.of() normalizes it below and
        # rejects an unknown name. Internally sm.proposer is already a Proposer.
        proposer: Proposer | str = Proposer.NGRAM,
        spans: Spans = (),
        session: Session | None = None,
        sampling: Sampling | None = None,
        hooks: Hooks | None = None,
    ) -> tuple[list[int], GenerateStats]:
        ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
        prompt: list[int] = ids[0].tolist()
        n = len(prompt)
        eos = {int(e) for e in eos_ids}
        smp = (sampling or GREEDY).seeded()
        self._tag(PassTag.SAMPLE_GREEDY if smp.greedy else PassTag.SAMPLE_STOCHASTIC)
        hk = hooks if hooks is not None and hooks.any else None
        if hk is not None:
            hk.rows(1)
            if hk.needs_logits:
                self._tag(PassTag.PICK_HOOKED)
        t0 = time.time()
        # a string from a caller's config reads into the enum here; an unknown one raises rather than decoding
        # as n-gram, which no report would have shown
        prop_kind = Proposer.of(proposer)
        use_dyn = prop_kind is Proposer.MTP_DYN
        use_tree = prop_kind in (Proposer.MTP_TREE, Proposer.MTP_DYN)
        use_mtp = prop_kind.mtp
        # the last layer's rows [1, T, H] of the pass that ran last, the drafter's input
        last: dict[str, torch.Tensor] = {}
        taps: dict[int, torch.Tensor] = {}
        tap_ids = set(hk.taps) if hk is not None else set()
        tapped = bool(tap_ids)

        def hold(i: int, h: torch.Tensor) -> None:
            if i == self.L - 1:
                last["h"] = h
            if i in tap_ids:
                taps[i] = h

        aw = LayerTap(hold, frozenset({self.L - 1, *tap_ids}))

        # a decode needing the last token's layers (the drafting head, taps) re-runs it; else a fed session goes on
        # from the logits it holds
        whole = not (use_mtp or tapped)
        opened, reuse, anchored = session._begin_decode(self, prompt, whole) if session is not None else (None, 0, None)
        # as long as the call reaches - the prompt, the answer and the widest verify pass past its last token - as
        # the greedy loop's is, so the two hold the same rows: where no reach is named a host layer takes the whole
        # context window's
        cache: KvCache = opened if opened is not None else self._decode_cache(n + int(max_new) + self._spec_full(v_max))
        logits, anchors = self._session_prefill(
            ids, cache, reuse, session, on_layer=aw if (use_mtp or tapped) else None
        )
        head = logits[0, -1:]
        if hk is not None and hk.needs_logits:
            head = hk.process([prompt], head.float())
        first = int(smp.pick_torch(head, [smp.key_for(n - 1)])[0])
        if hk is not None:
            hk.record(0, head[0], first)
            for i, h in taps.items():
                hk.tap(0, i, h[0, -1])
            if hk.on_pass is not None:
                hk.on_pass({"index": 0, "drafted": 0, "accepted": 0, "tokens": 1, "seconds": time.time() - t0})
        t_prefill = time.time() - t0
        # without a drafting head the n-gram continuations at every order and follower verify as one tree where the
        # tree's rows cost next to nothing (the fused MLX path; a card holding every layer, whose memory-bound decode
        # reads the weights once for every row), down to order 1
        ngram_tree = (
            not use_mtp
            and bool(getattr(self, "ngram_tree", True))
            and (
                (self.mlx is not None and self._mlx_tree_able(cache))
                or (self.dev.type == Device.CUDA and not self.host and int(getattr(self, "tree_budget", 0) or 0) > 0)
                # the host tier verifies a tree through its causal-mask pass (the receipts run one); the budget
                # weighs its rows against the host's own pass-cost curve
                or (self.dev.type == Device.CPU and self.mlx is None and bool(getattr(self, "_host_cost", None)))
            )
        )
        # spans: (tag, ids) pairs added to this proposer, or a SpanBank whose standing index it looks up
        bank = spans if isinstance(spans, SpanBank) else None
        ngram = NGramProposer(prompt, n_max=n_max, n_min=1 if ngram_tree else n_min, bank=bank)
        if not isinstance(spans, SpanBank):
            for tag, sp in spans:
                ngram.add_sequence(sp, tag)
        prop: ChainProposer = ngram
        draft: StreamedTextModel | None = getattr(self, "draft_engine", None)
        if draft is not None:
            # a small model of the family drafts: a tree where this tier verifies trees, a chain elsewhere; the
            # n-gram chains ride behind its own
            from .propose import ModelProposer, UnionProposer

            ks: Sequence[int] = getattr(self, "draft_ks", None) or (3, 2, 1)
            prop = UnionProposer(ModelProposer(draft, prompt, ks=ks), ngram)
        self._tag(PassTag.SPEC_DRAFT if draft is not None and not use_mtp else PROPOSER_TAG[prop_kind])
        by_src: dict[str, dict[str, int]] = {"drafted": {}, "accepted": {}}
        if use_mtp:
            dr = self.mtp_drafter()
            h_new = last["h"]
            h_prev = (anchored.get("h_last") if anchored else session.pend_h) if (reuse and session) else None
            # the drafter's rows are this session's only while it follows it: another session's decode since took
            # it over. Let go before they change, so a call failing part way leaves them no session's
            ours = session is not None and dr.follows is not None and dr.follows() is session
            dr.follows = None
            if (
                session is not None
                and session.dr is dr
                and ours
                and isinstance(cache, PagedCache)
                and LayerKind.LINEAR in self.layer_types
                and reuse <= int(session.dr_len)
            ):
                # a paged hybrid resumed at or before where its own drafter's rows reach (`_open_snapshot` cut them
                # to what the prompt shares): kept to there, the rest from the states the prompt's prefill made
                dr_len = int(session.dr_len)
                dr.crop(dr_len)
                if n - 1 > dr_len:
                    dr.extend(prompt[dr_len + 1 : n], h_new[:, dr_len - reuse : n - 1 - reuse], dr_len)
            elif (
                reuse
                and session is not None
                and session.dr is dr
                and ours
                and h_prev is not None
                and (anchored or reuse > int(session.dr_len))
            ):
                dr_len = reuse - 1 if anchored else int(session.dr_len)
                dr.crop(dr_len)
                # the held rows from the drafter's end up to the rows kept (a prompt parting from the session inside
                # the rows the drafter had yet to take keeps fewer than it held)
                h_prev = h_prev[:, : reuse - dr_len]
                h_cat = torch.cat([h_prev.to(h_new.device, h_new.dtype), h_new], dim=1)
                if n - 1 > dr_len:
                    dr.extend(prompt[dr_len + 1 : n], h_cat[:, : n - 1 - dr_len], dr_len)
            elif reuse:
                dr.reset()
                if n - 1 > reuse:
                    dr.extend(prompt[reuse + 1 : n], h_new[:, : n - 1 - reuse], reuse)
            else:
                dr.prefill(prompt, h_new)
            # the committed rows the drafter has yet to take, `cur`'s under the last of them. The drafter holds every
            # row before them - the target's rows less 1 + len(pend.toks) - and a pass that drafts takes them first;
            # a plain step leaves them to the next
            pend = DraftTail([], h_new[:, -1:])
        self.vram_trim("prefill")
        committed = [first]

        def ax() -> None:
            if session is None:
                return
            if use_mtp:
                # the drafter's rows: the target's (the prompt and the answer but its last token) less the pending
                session._commit_decode(
                    prompt, committed, cache, anchors, dr, n + len(committed) - 2 - len(pend.toks), pend.h
                )
            else:
                session._commit_decode(prompt, committed, cache, anchors)

        prop.extend(first)
        if on_token:
            on_token(first)
        census: GenerateStats = {
            "forwards": 1,
            "proposed": 0,
            "accepted": 0,
            "proposer": prop_kind.value,
            "drafted_by_pos": [0] * v_max,
            "accepted_by_pos": [0] * v_max,
            "reused": reuse,
            "prefill_s": round(t_prefill, 3),
            "anchored": anchored is not None,
        }
        if not smp.greedy and smp.seed is not None:  # `seeded` gave a sampled call its seed
            census["seed"] = smp.seed
        if first in eos or max_new <= 1:
            census["seconds"] = time.time() - t0
            census["tokens_per_pass"] = 1.0
            ax()
            return committed, census
        cur = first
        phase = {"propose": 0.0, "forward": 0.0, "commit": 0.0}
        # the drafter's steps and time of this call's passes (its prefill excluded)
        dr_steps0 = int(getattr(getattr(self, "aj", None), "steps", 0) or 0)
        dr_step_s0 = float(getattr(getattr(self, "aj", None), "step_s", 0.0) or 0.0)
        # the passes priced by what their rows cost (`SpecCost`): the base curve's width narrowed where the rows'
        # expert reads outweigh what their drafts return, a plain step where no width pays
        pricer = self._spec_pricer(v_max)
        pricer.begin()
        clock = PassClock(self.dev)
        store = getattr(self, "expert_store", None)
        store_reads: Callable[[], int] | None = getattr(store, "reads", None)
        store_waited: Callable[[], float] | None = getattr(store, "waited", None)
        if not (callable(store_reads) and callable(store_waited)):
            store_reads = store_waited = None
        while len(committed) < max_new and not self._stop_asked():
            tp = time.perf_counter()
            v = min(v_max, max_new - len(committed) - 1)
            budget = self._spec_budget(v_max, cache.get_seq_length())
            rows, probe = pricer.plan(budget, census["forwards"])
            if trace.ON:  # every pass: nothing built for the trace when it is off
                trace.changed(
                    (trace.token(self), "spec rows"),
                    rows,
                    "speculation: passes %d rows wide as priced (%s)",
                    rows,
                    "a plain step - no width pays" if rows <= 1 else f"the budget {budget}",
                )
            v = max(0, min(v, rows - 1))
            if v > 0:
                exact = self.fam.verify_exact(self, cache)
                if trace.ON:
                    trace.changed(
                        (trace.token(self), "spec exact"),
                        exact,
                        "speculation: %s",
                        "the paths verify exactly - drafting"
                        if exact
                        else "the paths would not verify exactly as they stand - plain one-row passes",
                    )
                if not exact:
                    # a pass the family's paths would not verify exactly as they stand now (Qwen4's off its kernels)
                    # is a plain one-row pass: nothing drafted
                    v = 0
            # each draft's path probability (1 for a chain's), and the width the drafter was asked for
            node_p: list[float] = []
            asked = v
            base_len = cache.get_seq_length()
            src = "mtp"
            # the pass's drafts: a chain's (node j + 1 its j-th guess), or a tree's as `VerifyTree` reads them
            guesses: list[int]
            parents: list[int]
            depth: list[int]
            children: dict[int, list[int]]
            node_tags: list[str] | None = None
            tree_now = use_tree
            sampled: SampledTree | None = None
            if use_tree and v <= 0:
                guesses, parents, depth = [], [-1], [0]
                children = {}
            elif use_dyn:
                # the drafter back to the rows before the pending ones: its last draft's steps past them were over
                # its own guesses
                dr.crop(base_len - 1 - len(pend.toks))
                extra: list[tuple[list[int], float, str]] = []
                ng_p = float(getattr(self, "ngram_p", 0.0))
                if ng_p > 0:
                    # the proposer's chain, tagged by where it came from: a draft model's, ahead of the n-gram's
                    ng_toks, where = prop.propose_with_source(v_max)
                    ng = [int(t) for t in ng_toks]
                    if ng:
                        extra.append((ng, ng_p, _source_tag(where)))
                # the tree's nodes, the root's row beside them within the widest pass (`_spec_full`: a family's exact
                # verify bounds it)
                n_draft = min(int(getattr(self, "tree_budget", 0) or v), self._spec_full(v_max) - 1)
                if pricer.active() and rows < pricer.full:
                    # the pricing sized the pass short of the whole tree: one node past its plan, so the prune
                    # below has a choice
                    n_draft = min(n_draft, rows)
                drawn = dr.au(
                    [*pend.toks, cur],
                    pend.h,
                    base_len - 1 - len(pend.toks),
                    n_draft,
                    min_prob=float(getattr(self, "tree_min_prob", 0.0)),
                    extra_chains=extra,
                    with_tags=True,
                    sampling=smp,
                )
                # under a temperature the drafter drew its children from its own distribution, returned with the
                # nodes' distributions: the verify pass accepts against them
                sampled = drawn if isinstance(drawn, SampledTree) else None
                node_p = [float(x) for x in dr.last_p]
                # the drafter's nodes are the tree's past its root: node j + 1 the drafter's j-th
                guesses = [int(t) for t in drawn.tokens]
                parents = [-1] + [0 if p < 0 else p + 1 for p in drawn.parents]
                depth = [0] + [int(d) for d in drawn.depths]
                node_tags = ["root", *drawn.tags]
                children = _children(parents)
                src = "union" if extra else "mtp"
                asked = n_draft
            elif use_tree:
                dr.crop(base_len - 1 - len(pend.toks))
                g1, ca, cb = dr.at([*pend.toks, cur], pend.h, base_len - 1 - len(pend.toks), v)
                assert g1 is not None  # `at` drafts a first guess wherever it is asked for one, as v > 0 asks
                guesses = [g1, *ca, *cb]
                # each chain hangs off the first guess (row 1), its rows one after another; a pass with room for one
                # guess (v 1, near the end of a decode) has no chains, so no row hangs off it
                parents = [
                    -1,
                    0,
                    *([1, *range(2, 1 + len(ca))] if ca else []),
                    *([1, *range(2 + len(ca), 1 + len(ca) + len(cb))] if cb else []),
                ]
                depth = [0, 1, *list(range(2, 2 + len(ca))), *list(range(2, 2 + len(cb)))]
                children = _children(parents)
            elif use_mtp:
                dr.crop(base_len - 1 - len(pend.toks))
                guesses = dr.ar([*pend.toks, cur], pend.h, base_len - 1 - len(pend.toks), v)
            else:
                chains = prop.propose_chains(v) if (ngram_tree and v > 0) else []
                if len(chains) > 1:
                    # the chains merged by shared prefix into one tree: on MLX at most 15 rows with the root (the
                    # matvec kernel's tile: up to 15 drafted rows verify at the cost of one); on the card the
                    # tree's budget
                    cap = 14 if self.mlx is not None else int(getattr(self, "tree_budget", 0) or 14)
                    cap = max(1, min(cap, rows - 1))
                    guesses, parents, depth, children, node_tags = _chains_tree(chains, cap)
                    src, tree_now = "ngram_tree", True
                elif chains:
                    # the one chain as drawn, tagged as a tree's nodes are: proposing again finds a draft model's
                    # pass spent on it
                    guesses, src = [int(g) for g in chains[0][0]], chains[0][1]
                else:
                    chain, where = prop.propose_with_source(v)
                    guesses = [int(g) for g in chain]
                    src = _source_tag(where)
            # the drafts made before the pricing kept a prefix
            drafted = len(guesses)
            if len(node_p) != drafted:
                node_p = [1.0] * drafted  # a chain or an unweighted tree: each node's gain its source's acceptance
            # the drafter's tree is priced always - an empty one too, saying nothing wider gained - the other drafts
            # where the pricing sizes the pass
            if use_dyn or (drafted and pricer.active()):
                # the drafts priced node by node in the drafter's order, and verified as far as they pay (`prune`):
                # a probe drafts the whole width and verifies one draft at least - unpriced, a chain verified every
                # one of them, 15 rows of a mixture's experts for drafts accepted one time in ten (gpt-oss-120b,
                # n-gram chains). A prefix keeps every node's parent (a node follows its parent in the chain, the
                # merged tree and the drafter's); a row's draws under a temperature stay whole, so a pruned child
                # ends the walk as one the tree never held. Size only, never a token
                tags_ = node_tags[1:] if node_tags else [src] * drafted
                deps_ = depth[1:] if tree_now else list(range(1, drafted + 1))
                gains_ = [pricer.gain(str(t_), int(d_), node_p[j]) for j, (t_, d_) in enumerate(zip(tags_, deps_))]
                pricer.record_tree(gains_, capped=drafted >= asked)
                keep = pricer.prune(gains_, probe)
                if keep < drafted:
                    guesses, node_p = guesses[:keep], node_p[:keep]
                    if tree_now:
                        parents, depth = parents[: keep + 1], depth[: keep + 1]
                        node_tags = node_tags[: keep + 1] if node_tags else node_tags
                        children = _children(parents)
            self.aa(parents if tree_now else None)
            pick: Pick = smp
            if tree_now and sampled is not None:
                pick = _verify_of(smp, len(guesses) + 1, sampled)
            tf = time.perf_counter()
            phase["propose"] += tf - tp
            taps.clear()
            reads0 = store_reads() if store_reads is not None else 0
            wait0 = store_waited() if store_waited is not None else 0.0
            clock.start()
            try:
                res = _verified(
                    self.forward(
                        [[cur, *guesses]],
                        cache=cache,
                        last_only=False,
                        on_layer=aw if (use_mtp or use_tree or tapped) else None,
                        positions=([[base_len + d for d in depth]] if tree_now else None),
                        # a hooked pick needs the logits here: no pick in the graph
                        pick=None if (hk is not None and hk.needs_logits) else pick,
                    )
                )
            finally:
                self.ab()
            census["forwards"] += 1
            census["proposed"] += len(guesses)
            a = 0
            stop = False
            new: list[int] = []
            path = [0]
            # the fused MLX path hands back the picks itself; the other paths the logits, picked here under the
            # same keys (a node's cache position)
            scores: torch.Tensor | None = None
            if isinstance(res, Picked):
                picked = res.ids
            else:
                scores = res.logits
                if hk is not None and hk.needs_logits:
                    # each node's row read with the ids that reach it: the committed ones and its drafts
                    ctx = [
                        prompt + committed + self._path_tokens(j, guesses, parents if tree_now else None)
                        for j in range(len(guesses) + 1)
                    ]
                    scores = hk.process(ctx, scores.float())
                keys = [smp.key_for(base_len + (depth[j] if tree_now else j)) for j in range(len(guesses) + 1)]
                picked = pick.pick_torch(scores, keys)
            clock.stop()
            # each row's token: under a Verify its packed outcome's - the accepted draw, or the one from the residual
            outcomes: list[int] = picked.tolist()
            row_tok = [Verify.unpack(o)[1] for o in outcomes] if isinstance(pick, Verify) else outcomes
            tc = time.perf_counter()
            phase["forward"] += tc - tf
            if tree_now:
                node = 0
                while True:
                    # the outcome at the node: the accepted draw or the token drawn from the residual under a
                    # Verify, the target's own draw otherwise; the walk goes on where the tree holds it
                    t = row_tok[node]
                    nxt = [c for c in children.get(node, []) if guesses[c - 1] == t]
                    if hk is not None:
                        self._hook_commit(hk, scores, taps, node, t)
                    committed.append(t)
                    new.append(t)
                    prop.extend(t)
                    if on_token:
                        on_token(t)
                    if t in eos or len(committed) >= max_new:
                        stop = True
                        break
                    if not nxt:
                        break
                    node = nxt[0]
                    path.append(node)
                    a += 1
                    census["accepted_by_pos"][min(depth[node] - 1, v_max - 1)] += 1
                    if node_tags:
                        k_ = "tag:" + node_tags[node]
                        by_src["accepted"][k_] = by_src["accepted"].get(k_, 0) + 1
                for j in range(1, len(guesses) + 1):
                    census["drafted_by_pos"][min(depth[j] - 1, v_max - 1)] += 1
                    if node_tags:
                        k_ = "tag:" + node_tags[j]
                        by_src["drafted"][k_] = by_src["drafted"].get(k_, 0) + 1
            else:
                for j in range(len(guesses) + 1):
                    t = row_tok[j]
                    if hk is not None:
                        self._hook_commit(hk, scores, taps, j, t)
                    committed.append(t)
                    new.append(t)
                    prop.extend(t)
                    if on_token:
                        on_token(t)
                    if t in eos or len(committed) >= max_new:
                        stop = True
                        break
                    if j < len(guesses) and t == guesses[j]:
                        a += 1
                        path.append(j + 1)
                        continue
                    break
                for j in range(len(guesses)):
                    census["drafted_by_pos"][j] += 1
                    if j < a:
                        census["accepted_by_pos"][j] += 1
            census["accepted"] += a
            self._tag_spec(len(guesses), a)
            n_rows = len(guesses) + 1
            tags_all = node_tags if node_tags else ["root", *([src] * len(guesses))]
            deps_all = depth if tree_now else list(range(n_rows))
            pricer.record_pass(
                n_rows,
                clock.seconds(),
                (store_reads() - reads0) if store_reads is not None else 0,
                (tf - tp) if drafted else None,
                [(str(tags_all[j]), int(deps_all[j]), node_p[j - 1]) for j in range(1, n_rows)],
                [str(tags_all[path[d]]) for d in range(1, len(path))],
                (store_waited() - wait0) if store_waited is not None else 0.0,
            )
            if guesses:
                by_src["drafted"][src] = by_src["drafted"].get(src, 0) + len(guesses)
                by_src["accepted"][src] = by_src["accepted"].get(src, 0) + a
            self.ad(cache, base_len, path)
            if (use_mtp or use_tree) and v > 0:
                # the drafter took the pending rows and `cur`'s at the root: the pass's accepted rows wait next
                pend = DraftTail(new[:a], last["h"][:, path])
            elif use_mtp or use_tree:
                # a plain step drafted nothing, so the drafter took nothing: `cur` and its row join the pending, which
                # a run too long for one step goes in without drafting (its cache is every committed row either way).
                # Where the step's rows lie: a card prefill's host layers left the prompt's on the card
                row = last["h"][:, path]
                pend = DraftTail([*pend.toks, cur], torch.cat([pend.h.to(row.device), row], dim=1))
                if len(pend.toks) >= PEND_ROWS:
                    dr_rows = cache.get_seq_length() - 1 - len(pend.toks)
                    dr.crop(dr_rows)
                    dr.extend(pend.toks, pend.h[:, :-1], dr_rows)
                    pend = DraftTail([], pend.h[:, -1:])
            phase["commit"] += time.perf_counter() - tc
            if hk is not None and hk.on_pass is not None:
                hk.on_pass(
                    {
                        "index": census["forwards"] - 1,
                        "drafted": len(guesses),
                        "accepted": a,
                        "tokens": len(new),
                        "seconds": time.perf_counter() - tp,
                    }
                )
            if stop:
                break
            cur = committed[-1]
        ax()
        census["seconds"] = time.time() - t0
        census["tokens_per_pass"] = round(len(committed) / max(1, census["forwards"]), 3)
        census["by_source"] = by_src
        census["phase_s"] = {k: round(x, 3) for k, x in phase.items()}
        census["priced"] = pricer.report()
        if use_mtp:
            census["mtp_build_s"] = round(dr.build_s, 2)
            census["mtp_steps"] = dr.steps - dr_steps0
            census["mtp_step_s"] = round(dr.step_s - dr_step_s0, 3)
        self.log(
            f"[stream-spec] {len(committed)} tokens in {census['forwards']} forwards "
            f"({len(committed) / max(1, census['forwards']):.2f} tokens/pass; accepted "
            f"{census['accepted']} of {census['proposed']} drafted) in {census['seconds']:.1f}s: a pass "
            f"{sum(phase.values()) / max(1, census['forwards'] - 1) * 1e3:.1f} ms = propose "
            f"{phase['propose'] / max(1, census['forwards'] - 1) * 1e3:.1f} + forward "
            f"{phase['forward'] / max(1, census['forwards'] - 1) * 1e3:.1f} + commit "
            f"{phase['commit'] / max(1, census['forwards'] - 1) * 1e3:.1f}"
            + (
                # this call's steps (`mtp_steps`): the drafter's own totals ran from its first call, and read as
                # 409 ms of a 128 ms propose on Qwen3.8-flash-next's second prompt
                f" (the drafter {census['mtp_step_s'] / max(1, census['forwards'] - 1) * 1e3:.1f} of the propose, "
                f"{census['mtp_steps'] / max(1, census['forwards'] - 1):.1f} steps a pass)"
                if use_mtp
                else ""
            )
            + "; accepted by depth "
            + " ".join(f"{a_}/{d_}" for a_, d_ in zip(census["accepted_by_pos"], census["drafted_by_pos"]) if d_)
            + pricer.line()
        )
        return committed, census

    # a session's decode: its row's tokens and the counts
    @overload
    def generate_greedy(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = ...,
        on_token: OnToken | None = ...,
        attention_mask: torch.Tensor | None = ...,
        on_layer: OnLayer | None = ...,
        prefill_only: bool = ...,
        *,
        session: Session,
        sampling: Sampling | None = ...,
        hooks: Hooks | None = ...,
    ) -> tuple[list[int], GenerateStats]: ...

    # one prompt's ids: its tokens
    @overload
    def generate_greedy(
        self,
        ids: Tokens,
        max_new: int,
        eos_ids: Tokens = ...,
        on_token: OnToken | None = ...,
        attention_mask: torch.Tensor | None = ...,
        on_layer: OnLayer | None = ...,
        prefill_only: bool = ...,
        session: None = ...,
        sampling: Sampling | None = ...,
        hooks: Hooks | None = ...,
    ) -> list[int]: ...

    # rows of ids: a list per row, one row's tokens alone
    @overload
    def generate_greedy(
        self,
        ids: torch.Tensor | TokenRows,
        max_new: int,
        eos_ids: Tokens = ...,
        on_token: OnToken | None = ...,
        attention_mask: torch.Tensor | None = ...,
        on_layer: OnLayer | None = ...,
        prefill_only: bool = ...,
        session: None = ...,
        sampling: Sampling | None = ...,
        hooks: Hooks | None = ...,
    ) -> list[int] | list[list[int]]: ...

    # ids or a session the caller holds as a union: any of the three
    @overload
    def generate_greedy(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = ...,
        on_token: OnToken | None = ...,
        attention_mask: torch.Tensor | None = ...,
        on_layer: OnLayer | None = ...,
        prefill_only: bool = ...,
        session: Session | None = ...,
        sampling: Sampling | None = ...,
        hooks: Hooks | None = ...,
    ) -> GreedyOut: ...

    @torch.inference_mode()
    def generate_greedy(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        on_token: OnToken | None = None,
        attention_mask: torch.Tensor | None = None,
        on_layer: OnLayer | None = None,
        prefill_only: bool = False,
        session: Session | None = None,
        sampling: Sampling | None = None,
        hooks: Hooks | None = None,
    ) -> GreedyOut:
        ids = torch.as_tensor(ids, dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.view(1, -1)
        B = ids.shape[0]
        eos = {int(e) for e in eos_ids}
        smp = (sampling or GREEDY).seeded()
        self._tag(PassTag.SPEC_OFF, PassTag.SAMPLE_GREEDY if smp.greedy else PassTag.SAMPLE_STOCHASTIC)
        # hooks see every pick and every tapped layer: the paths that pick inside their graph stand aside; a pass
        # callback alone leaves them be, the one-token loops reporting each token as a pass
        hk = hooks if hooks is not None and hooks.any else None
        fused = hk is None or not hk.active
        one = hk.per_token(on_token) if hk is not None and fused else on_token
        if hk is not None and hk.needs_logits:
            self._tag(PassTag.PICK_HOOKED)
        t0 = time.time()
        if fused and self._mlx_greedy_ok(B, attention_mask, on_layer, prefill_only):
            out_s, census = self._generate_greedy_mlx(ids, max_new, eos, one, t0, session, smp)
            return (out_s, census) if session is not None else out_s
        if session is not None:
            raise ValueError("[stream] a session needs the MLX pipelined decode or the speculative loop")
        if hk is None and self._mlx_batch_ok(B, on_layer, prefill_only):
            return self._generate_greedy_mlx_batch(ids, max_new, eos, attention_mask, t0, smp)
        if (
            fused
            and self.dev.type == Device.CUDA
            and self._card_greedy_ok(None, B, attention_mask, on_layer, prefill_only, session)
        ):
            # every layer on the card in one run: the step is a self-advancing graph and the host trails it
            try:
                return self._card_generate_greedy(ids, max_new, eos, one, t0, smp)
            except RuntimeError as e:
                if smp.greedy:
                    raise
                # the sampled step could not be captured: the step loop below, its draws keyed by the row
                self.log(f"[card] the sampled step graph failed ({e}); the step loop")
        target_len = int(ids.shape[1]) + int(max_new)
        if B > 1 and self.dev.type == Device.CUDA and not getattr(self, "_in_epoch", False):
            mb = self.scheduler.max_batch(target_len)
            if mb is not None and mb < B:
                # more rows than the free VRAM holds at this length: fixed-size epochs of `mb`, computed once
                # (`_in_epoch` stops a sub-call re-sizing off the shifted free memory)
                am_t = None if attention_mask is None else torch.as_tensor(attention_mask, dtype=torch.long)
                rows: list[list[int]] = []
                self._in_epoch = True
                try:
                    for s in range(0, B, mb):
                        csz = min(mb, B - s)
                        part = hk.child() if hk is not None else None
                        sub = self.generate_greedy(
                            ids[s : s + csz],
                            max_new,
                            eos_ids=eos_ids,
                            on_token=(on_token if s == 0 else None),
                            attention_mask=(None if am_t is None else am_t[s : s + csz]),
                            on_layer=on_layer,
                            prefill_only=prefill_only,
                            sampling=smp,
                            hooks=part,
                        )
                        rows.extend(as_rows(sub))
                        if hk is not None and part is not None:
                            hk.extend(part)
                finally:
                    self._in_epoch = False
                return rows
        # one sequence over the prefix cache's pages where the engine has one; rows side by side keep a cache of their
        # own (a paged cache holds one sequence)
        single = B == 1 and attention_mask is None
        cache: KvCache = self._decode_cache(target_len) if single else self.new_cache(max_len=target_len)
        am = None if attention_mask is None else torch.as_tensor(attention_mask, dtype=torch.long)
        out: list[list[int]] = [[] for _ in range(B)]
        done = [False] * B
        seen: dict[int, torch.Tensor] = {}
        layer_hook = on_layer
        ctx: list[list[int]] = []
        if hk is not None:
            hk.rows(B)
            # each row's own ids for the processors: a left-padded row without its pad
            ctx = [
                [
                    int(t)
                    for t, keep in zip(ids[b].tolist(), (am[b].tolist() if am is not None else [1] * ids.shape[1]))
                    if keep
                ]
                for b in range(B)
            ]
            taps = set(hk.taps)
            if taps:

                def tap_hook(i: int, h: torch.Tensor) -> None:
                    if on_layer is not None:
                        on_layer(i, h)
                    if i in taps:
                        seen[i] = h

                layer_hook = tap_hook

        logits: torch.Tensor = self._prefill(ids, cache, on_layer=layer_hook, attention_mask=am)
        self.vram_trim("prefill")
        pos0 = int(ids.shape[1]) - 1  # the row the prompt's last token holds: step k picks at pos0 + k
        for step in range(max_new):
            if self._stop_asked():
                break
            ts = time.perf_counter()
            last = logits[:, -1]
            if hk is not None and hk.needs_logits:
                last = hk.process([ctx[b] + out[b] for b in range(B)], last.float())
            nx = smp.pick_torch(last, [smp.key_for(pos0 + step)] * B)
            live = B - sum(done)
            for b in range(B):
                if not done[b]:
                    if hk is not None:
                        hk.record(b, last[b], int(nx[b]))
                        for i, h in seen.items():
                            hk.tap(b, i, h[b, -1])
                    out[b].append(int(nx[b]))
                    if int(nx[b]) in eos:
                        done[b] = True
            if on_token:
                on_token(int(nx[0]))
            if hk is not None and hk.on_pass is not None:
                hk.on_pass(
                    {
                        "index": step,
                        "drafted": 0,
                        "accepted": 0,
                        "tokens": live,
                        "seconds": time.perf_counter() - ts,
                    }
                )
            if all(done) or step == max_new - 1:
                break
            if am is not None:
                am = torch.cat([am, torch.ones((B, 1), dtype=torch.long)], dim=1)
            seen.clear()
            logits = self.forward(
                nx.view(B, 1), cache=cache, attention_mask=am, on_layer=(None if prefill_only else layer_hook)
            )
        n_tok = sum(len(o) for o in out)
        self.log(
            f"[stream] generated {n_tok} tokens over {B} rows in {time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, len(out[0])):.1f} s/step incl. prefill)"
        )
        return out[0] if B == 1 else out

    def serve(
        self,
        prompts: TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        pad_id: int | None = None,
        sampling: Sampling | None = None,
        hooks: Hooks | None = None,
    ) -> list[list[int]]:
        """Decode ragged prompts as fixed-size epochs sized by the scheduler for the longest waiting prompt, each
        run to completion before the next forms. Returns a token list per prompt in input order; `hooks` collects
        a row per prompt, in the same order."""
        prompts = [list(p) for p in prompts]
        if not prompts:
            return []
        if pad_id is None:
            pid = getattr(self.cfg, "pad_token_id", None)
            pad_id = int(pid if pid is not None else (next(iter(eos_ids)) if eos_ids else 0))
        results: list[list[int]] = [[] for _ in prompts]
        i = 0
        while i < len(prompts):
            longest = max(len(prompts[j]) for j in range(i, len(prompts)))
            B, _ = self.scheduler.plan(len(prompts) - i, longest + int(max_new))
            # the epoch's KV reservation is let go however the epoch ends, a refusal or an abort included: left
            # standing, every reading of the device's free memory would count it until the next epoch's plan
            try:
                rows = prompts[i : i + B]
                part = hooks.child() if hooks is not None else None
                if len(rows) == 1:
                    outs = as_rows(
                        self.generate_greedy([rows[0]], max_new, eos_ids=eos_ids, sampling=sampling, hooks=part)
                    )
                else:
                    ids, mask = self.pad_left(rows, pad_id)
                    outs = as_rows(
                        self.generate_greedy(
                            ids, max_new, eos_ids=eos_ids, attention_mask=mask, sampling=sampling, hooks=part
                        )
                    )
                if hooks is not None and part is not None:
                    part.rows(len(rows))
                    hooks.extend(part)
                for k in range(len(rows)):
                    results[i + k] = outs[k]
                i += len(rows)
            finally:
                self.scheduler.release()
        return results
