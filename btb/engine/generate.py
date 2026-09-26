# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Decoding: greedy and speculative (the tree), sessions that keep a cache between turns, the DeltaNet state
snapshots, and `serve` over ragged prompts in scheduler-sized epochs."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch

from .. import mlx as mlxdev
from ..draft import NGramProposer, SpanBank, Spans
from ..kinds import PROPOSER_TAG, Json, LayerKind, PassTag, Proposer, TokenRows, Tokens
from ..options import Device
from ..sampling import GREEDY, Sampling, Verify
from ..session import Session
from .cache import linear_layer
from .drafter import MTPDrafter
from .hooks import Hooks
from .state import _State

if TYPE_CHECKING:
    from .cache import CacheLayer

# one of a DeltaNet layer's two states, as the layer holds it: a tensor, or a dict of them by index
LinState = torch.Tensor | dict[int, torch.Tensor]
# a DeltaNet layer's (conv, recurrent) states copied, for a later `_lin_restore`
LinSnap = tuple[LinState, LinState]


class LinLayer(Protocol):
    """a cache layer of a DeltaNet (linear attention) layer, as the helpers below read and write it"""

    conv_states: LinState
    recurrent_states: LinState


def lin_layer(cl: CacheLayer) -> LinLayer:
    """`cl` as the helpers below take it; a TypeError for a layer that is not a DeltaNet's. transformers types its
    states dict[int, Tensor | None], and `_lin_snap` copies only the ones set."""
    return cast(LinLayer, linear_layer(cl))


def _lin(cl: LinLayer) -> tuple[torch.Tensor, torch.Tensor]:
    c, r = cl.conv_states, cl.recurrent_states
    if isinstance(c, dict):
        c = c[0]
    if isinstance(r, dict):
        r = r[0]
    return c, r


def _verify_of(smp: Sampling, T: int, qrows: dict[int, Any], draws: dict[int, list[int]]) -> Any:
    """the verify pick of a sampled tree of `T` rows: row j's draws in draw order and its drafter distribution
    (the root's under -1, node j's under j - 1), rows without one flagged as point masses; the plain sampling
    when no row has one"""
    kids = [[int(t) for t in draws.get(j - 1, [])] for j in range(T)]
    rows = [qrows.get(j - 1) for j in range(T)]
    hasq = [r is not None for r in rows]
    some = next((r for r in rows if r is not None), None)
    if some is None:
        return smp
    if isinstance(some, torch.Tensor):
        q: Any = torch.stack([r if r is not None else torch.zeros_like(some) for r in rows])
    else:
        m = mlxdev.mx()
        q = m.stack([r if r is not None else m.zeros_like(some) for r in rows])
    return Verify(smp, q, kids, hasq)


def _chains_tree(
    chains: Sequence[tuple[Tokens, str]], budget: int
) -> tuple[list[int], list[int], list[int], dict[int, list[int]], list[str]]:
    """Chains merged by shared prefix into a tree: node 0 the root, node j > 0 carrying guesses[j - 1]; returns
    (guesses, parents, depth, children, tags), at most `budget` nodes beyond the root."""
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
    return guesses, parents, depth, children, tags


class _GenerateMixin(_State):
    _lin = staticmethod(_lin)

    @staticmethod
    def _lin_snap(cl: LinLayer) -> LinSnap:
        def copy(s: LinState) -> LinState:
            if isinstance(s, dict):
                return {k: v.clone() for k, v in s.items() if isinstance(v, torch.Tensor)}
            return s.clone()

        return copy(cl.conv_states), copy(cl.recurrent_states)

    @staticmethod
    def _lin_restore(cl: LinLayer, snap: LinSnap) -> None:
        conv, rec = snap
        if isinstance(conv, dict) and isinstance(rec, dict):
            for k, v in conv.items():
                cl.conv_states[k].copy_(v)
            for k, v in rec.items():
                cl.recurrent_states[k].copy_(v)
            return
        assert isinstance(conv, torch.Tensor) and isinstance(rec, torch.Tensor)  # _lin_snap copies both alike
        c, r = _lin(cl)
        c.copy_(conv)
        r.copy_(rec)

    def _session_prefill(
        self, ids: torch.Tensor, cache: Any, reuse: int, session: Session | None, on_layer: Any = None
    ) -> tuple[Any, Any]:
        """Prefill ids[:, reuse:]; for a hybrid in a session returns the DeltaNet state snapshots the next turn can
        resume from (the prompt's end, and the point the re-rendering will diverge at once the tail is known)."""
        n = int(ids.shape[1])
        hybrid = LayerKind.LINEAR in self.layer_types
        # every row reused (`Session._open`'s whole): the session's logits for its last token, nothing to run
        held = None
        if session is not None and reuse == n:
            held, session._held = session._held, None
            assert held is not None, "a prompt the cache holds whole comes with its logits"
            held = held.view(1, 1, -1)
        if session is None or not hybrid:
            return (held if held is not None else self._prefill(ids[:, reuse:], cache, on_layer=on_layer)), None
        hs: list[torch.Tensor] = []

        def snap(at: int) -> Any:
            return {
                "n": at,
                "states": {
                    i: self._lin_snap(cache.layers[i]) for i in range(self.L) if self.layer_types[i] == LayerKind.LINEAR
                },
                "h_last": hs[-1][:, -1:].detach().clone() if hs else None,
            }

        def gather(i: int, h: torch.Tensor) -> None:
            # the last layer's rows are gathered across the chunks for one call below; the others go straight on
            if i == self.L - 1:
                hs.append(h)
            elif on_layer is not None:
                on_layer(i, h)

        hook = gather if on_layer is not None else None
        if held is not None:
            return held, [snap(n)]
        anchors = []
        d = int(session.tail)
        cut = n - d
        if d > 0 and reuse < cut:
            self._prefill(ids[:, reuse:cut], cache, on_layer=hook)
            anchors.append(snap(cut))
            logits = self._prefill(ids[:, cut:], cache, on_layer=hook)
        else:
            logits = self._prefill(ids[:, reuse:], cache, on_layer=hook)
        anchors.append(snap(n))
        if on_layer is not None:
            on_layer(self.L - 1, hs[0] if len(hs) == 1 else torch.cat(hs, dim=1))
        return logits, anchors

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
    def _hook_commit(hk: Hooks, logits: torch.Tensor, taps: dict[int, torch.Tensor], node: int, token: int) -> None:
        """a committed token's log-probability off its node's row, and the tapped layers' state at that row"""
        if hk.needs_logits:
            hk.record(0, logits[node], token)
        for i, h in taps.items():
            hk.tap(0, i, h[0, node])

    @staticmethod
    def _lin_set(cl: LinLayer, conv: torch.Tensor, rec: torch.Tensor) -> None:
        if isinstance(cl.conv_states, dict):
            cl.conv_states[0] = conv
        else:
            cl.conv_states = conv
        if isinstance(cl.recurrent_states, dict):
            cl.recurrent_states[0] = rec
        else:
            cl.recurrent_states = rec

    def mtp_drafter(self) -> MTPDrafter:
        if getattr(self, "aj", None) is None:
            t0 = time.time()
            self.aj = MTPDrafter(
                self, weights=getattr(self, "drafter_weights", None), dev=getattr(self, "drafter_dev", None)
            )
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
        on_token: Callable[[int], Any] | None = None,
        # the str half is the public boundary (a CLI/serve value, a test); Proposer.of() normalizes it below and
        # rejects an unknown name. Internally sm.proposer is already a Proposer.
        proposer: Proposer | str = Proposer.NGRAM,
        spans: Spans = (),
        session: Session | None = None,
        sampling: Sampling | None = None,
        hooks: Hooks | None = None,
    ) -> tuple[list[int], Json]:
        ids = torch.as_tensor(ids, dtype=torch.long).view(1, -1)
        prompt = ids[0].tolist()
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
        last = {}
        taps: dict[int, torch.Tensor] = {}
        tap_ids = set(hk.taps) if hk is not None else set()
        tapped = bool(tap_ids)

        def aw(i: int, h: torch.Tensor) -> None:
            if i == self.L - 1:
                last["h"] = h
            if i in tap_ids:
                taps[i] = h

        # a decode needing the last token's layers (the drafting head, taps) re-runs it; else a fed session goes on
        # from the logits it holds
        whole = not (use_mtp or tapped)
        cache, reuse, anchored = session._open(self, prompt, whole) if session is not None else (None, 0, None)
        if cache is None:
            cache = self.new_cache()
        logits: Any
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
                (getattr(self, "mlx", None) is not None and self._mlx_tree_able(cache))
                or (self.dev.type == Device.CUDA and not self.host and int(getattr(self, "tree_budget", 0) or 0) > 0)
                # the host tier verifies a tree through its causal-mask pass (the receipts run one); the budget
                # weighs its rows against the host's own pass-cost curve
                or (
                    self.dev.type == Device.CPU
                    and getattr(self, "mlx", None) is None
                    and bool(getattr(self, "_host_cost", None))
                )
            )
        )
        # spans: (tag, ids) pairs added to this proposer, or a SpanBank whose standing index it looks up
        bank = spans if isinstance(spans, SpanBank) else None
        prop: Any = NGramProposer(prompt, n_max=n_max, n_min=1 if ngram_tree else n_min, bank=bank)
        if not isinstance(spans, SpanBank):
            for tag, sp in spans:
                prop.add_sequence(sp, tag)
        draft = getattr(self, "draft_engine", None)
        if draft is not None:
            # a small model of the family drafts: a tree where this tier verifies trees, a chain elsewhere; the
            # n-gram chains ride behind its own
            from .propose import ModelProposer, UnionProposer

            ks = getattr(self, "draft_ks", None) or (3, 2, 1)
            prop = UnionProposer(ModelProposer(draft, prompt, ks=ks), prop)
        self._tag(PassTag.SPEC_DRAFT if draft is not None and not use_mtp else PROPOSER_TAG[prop_kind])
        by_src: dict[str, dict[str, int]] = {"drafted": {}, "accepted": {}}
        last_base = n
        if use_mtp:
            dr = self.mtp_drafter()
            h_new = last["h"]
            h_prev = (anchored.get("h_last") if anchored else session.pend_h) if (reuse and session) else None
            if reuse and session is not None and session.dr is dr and h_prev is not None:
                dr_len = reuse - 1 if anchored else int(session.dr_len)
                dr.crop(dr_len)
                h_cat = torch.cat([h_prev.to(h_new.device, h_new.dtype), h_new], dim=1)
                if n - 1 > dr_len:
                    dr.extend(prompt[dr_len + 1 : n], h_cat[:, : n - 1 - dr_len], dr_len)
            elif reuse:
                dr.reset()
                if n - 1 > reuse:
                    dr.extend(prompt[reuse + 1 : n], h_new[:, : n - 1 - reuse], reuse)
            else:
                dr.prefill(prompt, h_new)
            pend_toks: list[Any] = []
            pend_h = h_new[:, -1:]
        self.vram_trim("prefill")
        committed = [first]

        def ax() -> None:
            if session is None:
                return
            if use_mtp:
                session._keep(prompt, committed, cache, anchors, dr, last_base if len(committed) > 1 else n - 1, pend_h)
            else:
                session._keep(prompt, committed, cache, anchors)

        prop.extend(first)
        if on_token:
            on_token(first)
        census: Json = {
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
        if not smp.greedy:
            census["seed"] = smp.seed
        if first in eos or max_new <= 1:
            census["seconds"] = time.time() - t0
            census["tokens_per_pass"] = 1.0
            ax()
            return committed, census
        cur = first
        phase = {"propose": 0.0, "forward": 0.0, "commit": 0.0}
        # tokens a pass has been yielding (the root and its accepted drafts), against which the card weighs
        # the width of the next pass
        ema_tokens = 1.0
        # the drafter's steps and time of this call's passes (its prefill excluded)
        dr_steps0 = int(getattr(getattr(self, "aj", None), "steps", 0) or 0)
        dr_step_s0 = float(getattr(getattr(self, "aj", None), "step_s", 0.0) or 0.0)
        while len(committed) < max_new and not self._stop_asked():
            tp = time.perf_counter()
            v = min(v_max, max_new - len(committed) - 1)
            budget = self._spec_budget(ema_tokens, census["forwards"], v_max, cache.get_seq_length())
            v = max(0, min(v, budget - 1))
            base_len = cache.get_seq_length()
            last_base = base_len
            src = "mtp"
            node_tags = None
            tree_now = use_tree
            tree_fn = getattr(self, "_tree_fn", None)
            qrows: Any = None
            draws: Any = None
            if use_tree and v <= 0:
                guesses, parents, depth = [], [-1], [0]
                children: dict[Any, Any] = {}
            elif use_dyn:
                dr.crop(base_len)
                extra = []
                ng_p = float(getattr(self, "ngram_p", 0.0))
                if ng_p > 0:
                    ng, _where = prop.propose_with_source(v_max)
                    ng = [int(t) for t in ng]
                    if ng:
                        extra.append((ng, ng_p, "ngram"))
                for sp_toks, sp_p, sp_tag in getattr(self, "extra_chains", ()):
                    extra.append((sp_toks, sp_p, sp_tag))
                drawn = dr.au(
                    [*pend_toks, cur],
                    pend_h,
                    base_len - 1 - len(pend_toks),
                    int(getattr(self, "tree_budget", 0) or v),
                    min_prob=float(getattr(self, "tree_min_prob", 0.0)),
                    extra_chains=extra,
                    with_tags=True,
                    sampling=smp,
                )
                # under a temperature the drafter drew its children from its own distribution, returned with the
                # nodes' distributions: the verify pass accepts against them
                tk, par, dep, tg = drawn[:4]
                qrows, draws = (drawn[4], drawn[5]) if len(drawn) > 4 else (None, None)
                guesses = [int(t) for t in tk]
                parents = [-1] + [0 if p < 0 else p + 1 for p in par]
                depth = [0] + [int(d) for d in dep]
                node_tags = ["root", *list(tg)]
                children = {}
                for j in range(1, len(guesses) + 1):
                    children.setdefault(parents[j], []).append(j)
                src = "union" if extra else "mtp"
            elif use_tree:
                if tree_fn is not None:
                    g1, ca, cb = tree_fn(committed)
                else:
                    dr.crop(base_len)
                    g1, ca, cb = dr.at([*pend_toks, cur], pend_h, base_len - 1 - len(pend_toks), v)
                guesses = [g1, *ca, *cb]
                parents = [-1, 0, 1, *list(range(2, 1 + len(ca))), 1, *list(range(2 + len(ca), 1 + len(ca) + len(cb)))]
                depth = [0, 1, *list(range(2, 2 + len(ca))), *list(range(2, 2 + len(cb)))]
                children = {}
                for j in range(1, len(guesses) + 1):
                    children.setdefault(parents[j], []).append(j)
            elif use_mtp:
                dr.crop(base_len)
                guesses = dr.ar([*pend_toks, cur], pend_h, base_len - 1 - len(pend_toks), v)
            else:
                chains = prop.propose_chains(v) if (ngram_tree and v > 0) else []
                if len(chains) > 1:
                    # the chains merged by shared prefix into one tree: on MLX at most 15 rows with the root (the
                    # matvec kernel's tile: up to 15 drafted rows verify at the cost of one); on the card the
                    # tree's budget
                    cap = 14 if getattr(self, "mlx", None) is not None else int(getattr(self, "tree_budget", 0) or 14)
                    cap = max(1, min(cap, budget - 1))
                    guesses, parents, depth, children, node_tags = _chains_tree(chains, cap)
                    src, tree_now = "ngram_tree", True
                else:
                    guesses, where = prop.propose_with_source(v)
                    guesses = [int(g) for g in guesses]
                    src = where[0] if isinstance(where, tuple) else "self"
            self.aa(parents if tree_now else None)
            pick: Any = smp
            if tree_now and qrows is not None:
                pick = _verify_of(smp, len(guesses) + 1, qrows, draws)
            tf = time.perf_counter()
            phase["propose"] += tf - tp
            taps.clear()
            try:
                out = self.forward(
                    [[cur, *guesses]],
                    cache=cache,
                    last_only=False,
                    on_layer=aw if (use_mtp or use_tree or tapped) else None,
                    positions=([[base_len + d for d in depth]] if tree_now else None),
                    # a hooked pick needs the logits here: no pick in the graph
                    pick=None if (hk is not None and hk.needs_logits) else pick,
                )[0]
            finally:
                self.ab()
            census["forwards"] += 1
            census["proposed"] += len(guesses)
            a = 0
            stop = False
            new = []
            path = [0]
            # the fused MLX path hands back the picks itself; the other paths the logits, picked here under the
            # same keys (a node's cache position)
            if out.dtype in (torch.int32, torch.int64):
                am_all = out.tolist()
            else:
                if hk is not None and hk.needs_logits:
                    # each node's row read with the ids that reach it: the committed ones and its drafts
                    ctx = [
                        prompt + committed + self._path_tokens(j, guesses, parents if tree_now else None)
                        for j in range(len(guesses) + 1)
                    ]
                    out = hk.process(ctx, out.float())
                keys = [smp.key_for(base_len + (depth[j] if tree_now else j)) for j in range(len(guesses) + 1)]
                am_all = pick.pick_torch(out, keys).tolist()
            tc = time.perf_counter()
            phase["forward"] += tc - tf
            if tree_now:
                node = 0
                while True:
                    # the outcome at the node: the accepted draw or the token drawn from the residual under a
                    # Verify, the target's own draw otherwise; the walk goes on where the tree holds it
                    t = Verify.unpack(am_all[node])[1] if isinstance(pick, Verify) else am_all[node]
                    nxt = [c for c in children.get(node, []) if guesses[c - 1] == t]
                    if hk is not None:
                        self._hook_commit(hk, out, taps, node, t)
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
                    t = am_all[j]
                    if hk is not None:
                        self._hook_commit(hk, out, taps, j, t)
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
            ema_tokens = 0.85 * ema_tokens + 0.15 * (1 + a)
            if guesses:
                by_src["drafted"][src] = by_src["drafted"].get(src, 0) + len(guesses)
                by_src["accepted"][src] = by_src["accepted"].get(src, 0) + a
            self.ad(cache, base_len, path)
            if use_mtp or use_tree:
                pend_toks, pend_h = new[:a], last["h"][:, path]
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
                f" (the drafter {dr.step_s / max(1, census['forwards'] - 1) * 1e3:.1f} of the propose, "
                f"{dr.steps / max(1, census['forwards'] - 1):.1f} steps a pass)"
                if use_mtp
                else ""
            )
            + "; accepted by depth "
            + " ".join(f"{a_}/{d_}" for a_, d_ in zip(census["accepted_by_pos"], census["drafted_by_pos"]) if d_)
        )
        return committed, census

    @torch.inference_mode()
    def generate_greedy(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        on_token: Callable[[int], Any] | None = None,
        attention_mask: torch.Tensor | None = None,
        on_layer: Any = None,
        prefill_only: bool = False,
        session: Any = None,
        sampling: Sampling | None = None,
        hooks: Hooks | None = None,
    ) -> Any:
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
                        rows.extend([sub] if csz == 1 else sub)
                        if hk is not None and part is not None:
                            hk.extend(part)
                finally:
                    self._in_epoch = False
                return rows
        cache = self.new_cache(max_len=target_len)
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
        results: list[Any] = [None] * len(prompts)
        i = 0
        while i < len(prompts):
            longest = max(len(prompts[j]) for j in range(i, len(prompts)))
            B, _ = self.scheduler.plan(len(prompts) - i, longest + int(max_new))
            rows = prompts[i : i + B]
            part = hooks.child() if hooks is not None else None
            if len(rows) == 1:
                outs = [self.generate_greedy([rows[0]], max_new, eos_ids=eos_ids, sampling=sampling, hooks=part)]
            else:
                ids, mask = self.pad_left(rows, pad_id)
                o = self.generate_greedy(
                    ids, max_new, eos_ids=eos_ids, attention_mask=mask, sampling=sampling, hooks=part
                )
                outs = o if isinstance(o[0], (list, tuple)) else [o]
            if hooks is not None and part is not None:
                part.rows(len(rows))
                hooks.extend(part)
            for k in range(len(rows)):
                results[i + k] = outs[k]
            i += len(rows)
            self.scheduler.release()
        return results
