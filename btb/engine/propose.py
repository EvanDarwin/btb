# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Drafters verified by the big model's tree pass (every output token the big model's own), in the n-gram
proposer's interface: a small model of the same family, and the big model's own last layers."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import torch

INJECT = ("zero", "noise", "stale", "embed", "delta", "unembed", "memory")
SAMPLE = ("topk", "temp", "cloud")


class TailDraft(NamedTuple):
    """The tail drafter's knobs (`--tail-draft key=value,...`): `layers` end layers run as the drafter; `inject`
    how the boundary residual becomes the next position's input; `alpha` the injected embedding's rms as a
    multiple of the residual's; `sample` how candidates come off the logits; `k` how many; `minp` the probability
    a top-k candidate must clear (0 keeps all k: right where the verify's rows are free); `depth` the chain under
    each candidate, the tail's top pick per level (one more tail pass a level); `temp` a draw's temperature;
    `noise` the cloud's spread as a fraction of the residual's rms; `race` 0 drafts every pass whatever it costs
    (measuring acceptance where the drafter cannot pay for itself), 1 races the forward."""

    layers: int = 8
    inject: str = "embed"
    alpha: float = 1.0
    sample: str = "topk"
    k: int = 4
    minp: float = 0.0
    depth: int = 1
    temp: float = 1.0
    noise: float = 0.1
    race: int = 1

    @classmethod
    def parse(cls, spec: str) -> TailDraft:
        """'layers=8,inject=embed,k=4' - any subset; 1, true or an empty spec takes the defaults"""
        kw: dict[str, Any] = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part or part.lower() in ("1", "true", "on"):
                continue
            if "=" not in part:
                raise ValueError(f"tail_draft: expected key=value, got {part!r}")
            key, val = (s.strip() for s in part.split("=", 1))
            if key not in cls._fields:
                raise ValueError(f"tail_draft: unknown key {key!r} (one of {', '.join(cls._fields)})")
            kw[key] = type(cls._field_defaults[key])(val)
        td = cls(**kw)
        if td.inject not in INJECT:
            raise ValueError(f"tail_draft: inject is one of {', '.join(INJECT)}, not {td.inject!r}")
        if td.sample not in SAMPLE:
            raise ValueError(f"tail_draft: sample is one of {', '.join(SAMPLE)}, not {td.sample!r}")
        if td.layers < 1 or td.k < 1 or td.depth < 1 or td.temp <= 0 or td.noise < 0 or not 0 <= td.minp < 1:
            raise ValueError(
                "tail_draft: layers, k and depth are 1 or more, temp above 0, noise 0 or more, minp in [0, 1)"
            )
        return td


class ModelProposer:
    """A tree of drafts from a small engine, one pass of it a depth: pass 1 catches the drafter's cache up over
    the tokens committed since the last proposal and reads the root's top-ks[0]; each later pass re-runs the
    tree so far and reads its leaves' top-ks[d]; the chains are the root-to-leaf paths, at most `nodes` of
    them in the tree (the verify tile). The drafter's cache keeps the committed rows only; the tree's rows are
    cropped after each pass. Every pass costs the small model's step (the 0.6B: 11.5 ms in its kernel)."""

    def __init__(
        self, engine: Any, prompt: Iterable[int], ks: Sequence[int] = (3, 2, 1), nodes: int = 14, tag: str = "draft"
    ) -> None:
        self.sm = engine
        self.ks, self.nodes, self.tag = [int(k) for k in ks], int(nodes), tag
        ids = [int(t) for t in prompt]
        self.cache = engine.new_cache()
        with torch.inference_mode():
            engine._prefill(torch.tensor([ids]), self.cache)
        self.n = len(ids)
        self.pending: list[int] = []
        self.passes = 0

    def add_sequence(self, ids: Iterable[int], tag: Any) -> int:
        return 0

    def extend(self, tok_id: int) -> None:
        self.pending.append(int(tok_id))

    def _topk(self, toks: list[int], parents: list[int], k: int) -> list[list[int]]:
        """one pass of the drafter over `toks` with the tree `parents` (-1 a root): the top-k token ids [T, k] a
        row, sorted by logit descending. On the MLX megakernel the top-k reduction runs on the graph over the
        kernel's scratch logits - k ids a row cross to the host, never the whole vocab; the torch tiers top-k the
        pass's logits directly."""
        sm = self.sm
        T = len(toks)
        mg = getattr(sm, "_mega", None)
        fast = mg is not None and sm._mega_ok(self.cache, T, None, True, True, None)
        with torch.inference_mode():
            sm.aa(parents)
            try:
                if fast:
                    sm.forward([toks], cache=self.cache, last_only=False, pick=True)
                else:
                    lg = sm.forward([toks], cache=self.cache, last_only=False)
            finally:
                sm.ab()
        self.passes += 1
        if not fast:
            return torch.topk(lg[0].float(), min(k, lg.shape[-1]), dim=-1).indices.tolist()
        assert mg is not None  # fast is set only when the megakernel is present
        import mlx.core as mx
        import numpy as np

        from .drafter import mlx_topk_ids

        V = mg.V
        raw = mg.scr[mg.toff["logits"] : mg.toff["logits"] + T * V * 4].view(mx.float32).reshape(T, V)
        ids = mlx_topk_ids(raw, k)
        mx.eval(ids)
        return np.array(ids).tolist()

    def _crop(self) -> None:
        for cl in self.cache.layers:
            if cl.get_seq_length() > self.n:
                cl.crop(self.n)

    def propose_chains(self, v: int, tree: bool = True) -> list[tuple[list[int], str]]:
        """the drafts as chains: the tree of `ks` where the target verifies trees, one greedy chain else"""
        if v <= 0 or not self.pending:
            return []
        ks = list(self.ks[:v]) if tree else [1] * min(v, len(self.ks))
        toks = self.pending
        self.pending = []
        # pass 1: the committed tokens as a chain; the last row's top-k are the depth-1 nodes
        k = min(ks[0], self.nodes)
        row = self._topk(toks, list(range(-1, len(toks) - 1)), k)[-1]
        self.n += len(toks)
        nodes_t = [int(t) for t in row]  # the tree's tokens, node order
        nodes_p = [-1] * k  # each node's parent in the tree (-1 the root, which sits in the cache)
        leaves = list(range(k))
        for d in range(1, len(ks)):
            if len(nodes_t) >= self.nodes:
                break
            # the tree so far as one pass; the leaves' top-k are the next depth
            kd = ks[d]
            rows = self._topk(nodes_t, nodes_p, kd)
            self._crop()
            top = [rows[leaf] for leaf in leaves]
            new_leaves = []
            for leaf, row in zip(leaves, top):
                for t in row:
                    if len(nodes_t) >= self.nodes:
                        break
                    nodes_t.append(int(t))
                    nodes_p.append(leaf)
                    new_leaves.append(len(nodes_t) - 1)
            leaves = new_leaves
        chains = []
        for leaf in leaves:
            path = []
            j = leaf
            while j >= 0:
                path.append(nodes_t[j])
                j = nodes_p[j]
            chains.append((path[::-1], self.tag))
        return chains

    def propose_with_source(self, v: int) -> tuple[list[int], Any]:
        """one greedy chain, for a target that verifies chains (the host tier, a card without the tree)"""
        chains = self.propose_chains(v, tree=False)
        return (chains[0][0], self.tag) if chains else ([], None)


class UnionProposer:
    """The model's chains first, the n-gram proposer's after: both fed every committed token."""

    def __init__(self, model: ModelProposer | TailProposer, ngram: Any) -> None:
        self.model, self.ngram = model, ngram

    def add_sequence(self, ids: Iterable[int], tag: Any) -> int:
        return self.ngram.add_sequence(ids, tag)

    def extend(self, tok_id: int) -> None:
        self.model.extend(tok_id)
        self.ngram.extend(tok_id)

    def anchor(self, path: Sequence[int], toks: Sequence[int]) -> None:
        for p in (self.model, self.ngram):
            a = getattr(p, "anchor", None)
            if a is not None:
                a(path, toks)

    def propose_chains(self, v: int) -> list[tuple[list[int], str]]:
        return self.model.propose_chains(v) + self.ngram.propose_chains(v)

    def propose_with_source(self, v: int) -> tuple[list[int], Any]:
        out = self.model.propose_with_source(v)
        return out if out[0] else self.ngram.propose_with_source(v)


class TailProposer:
    """The model's own last `layers` layers draft, with no drafting head and no second model: the residual at the
    tail's boundary for the last verified row (the verify pass keeps it), the newest token's embedding added, runs
    the tail alone at the next position over the live cache; the logits give the depth-1 candidates, and the tail's
    rows are cropped after. Those layers are resident, so a pass that waits on the drive pays nothing for it; the
    verify decides, so a wrong draft only misses."""

    def __init__(self, engine: Any, spec: TailDraft, cache: Any, tag: str = "tail") -> None:
        if engine.fam.hybrid:
            raise ValueError("tail_draft serves the dense families")
        import mlx.core as mx

        # the tail is the resident run at the model's end, at most `layers` deep: a streamed layer in it would
        # wait on a ring this pass never starts
        L = int(engine.L)
        res = 0
        while res < L - 1 and (L - 1 - res) in engine.mlx_layers and (L - 1 - res) not in engine.cold:
            res += 1
        layers = min(spec.layers, res)
        if layers < 1:
            raise ValueError("tail_draft: no resident MLX layer at the model's end to draft from")
        if layers < spec.layers:
            engine.log(f"[tail] {spec.layers} layers asked, the last {layers} are resident: drafting with those")
        self.sm, self.spec, self.cache, self.tag = engine, spec, cache, tag
        self.start = L - layers  # the tail is layers start..L-1, fed the residual after start-1
        self.row: int | None = None  # the verify row whose boundary residual anchors the next draft
        self.prev: int | None = None  # the token at that row
        self.cur: int | None = None  # the newest committed token, not yet through the model
        self.passes = 0
        self.mem: dict[int, Any] = {}  # token -> its boundary residual at its latest verified position
        # the race: a draft runs only while its cost, as a share of a verify pass, is under the tokens it has
        # been adding to one; the pass's time starts from the warm-up's one-row cost and follows the run
        cost = getattr(engine, "_mlx_cost", None) or {}
        self.fwd_s: float | None = float(cost[1]) if 1 in cost else None
        self.step_s: float | None = None
        self.gain = 0.5
        self.seen = 0  # verify passes observed
        self.armed = True  # the next draft is worth its cost: the pass before it keeps the residual
        self._drafted = False
        self._bail_fwd: float | None = None  # the pass's time when the drafter last stood down
        self._key = mx.random.key(0)
        self._gen = torch.Generator().manual_seed(0)
        engine._tail_h = None

    @property
    def tap(self) -> int:
        """the layer after which the verify pass keeps its rows' residual: the tail's input"""
        return self.start - 1

    def add_sequence(self, ids: Iterable[int], tag: Any) -> int:
        return 0

    def extend(self, tok_id: int) -> None:
        self.prev, self.cur = self.cur, int(tok_id)

    def seed(self) -> None:
        """the prompt's rows into the memory (the last forward tapped, `_tail_ids` its tokens), a token's latest
        position winning"""
        if self.spec.inject != "memory":
            return
        h_all = getattr(self.sm, "_tail_h", None)
        ids = getattr(self.sm, "_tail_ids", None)
        if h_all is None or not ids or int(h_all.shape[0]) != len(ids):
            return
        import mlx.core as mx

        last = {int(t): i for i, t in enumerate(ids)}
        rows = mx.take(h_all, mx.array(list(last.values()), dtype=mx.int32), axis=0)
        mx.eval(rows)
        for i, t in enumerate(last):
            self.mem[t] = rows[i]
        self.sm.log(f"[tail] memory seeded with {len(last)} distinct tokens of the prompt's {len(ids)}")

    def observe(self, fwd_s: float, accepted: int) -> None:
        """after a verify pass: its time, and how many of this drafter's rows it took (its yield, counted on the
        passes it drafted for)"""
        self.seen += 1
        self.fwd_s = fwd_s if self.fwd_s is None else 0.8 * self.fwd_s + 0.2 * fwd_s
        if self._drafted:
            self.gain = 0.85 * self.gain + 0.15 * accepted
            self._drafted = False

    def arm(self) -> bool:
        """Before a verify pass: whether the draft after it beats the forward - its time, as a share of a pass,
        under the tokens it has been adding to one - so the pass keeps the residual for it. Standing down, it
        tries again every 64th pass, or as soon as the forward has slowed by a quarter; an unarmed pass keeps
        no anchor, and the megakernel is free to run it."""
        if not self.spec.race or self.fwd_s is None or self.step_s is None or self.step_s < self.fwd_s * self.gain:
            self.armed = True
        else:
            slowed = self._bail_fwd is not None and self.fwd_s >= 1.25 * self._bail_fwd
            self.armed = (self.seen + 1) % 64 == 0 or slowed
            if self._bail_fwd is None:
                self._bail_fwd = self.fwd_s
        if self.armed:
            self._bail_fwd = None
        else:
            self.row = None
        return self.armed

    def anchor(self, path: Sequence[int], toks: Sequence[int]) -> None:
        """the pass's accepted rows (`toks` the rows' tokens): the last anchors the next draft, and for `memory`
        each one's residual is kept by its token - the model's own representation of it"""
        self.row = int(path[-1])
        if self.spec.inject != "memory":
            return
        h_all = getattr(self.sm, "_tail_h", None)
        if h_all is None:
            return
        import mlx.core as mx

        rows = mx.take(h_all, mx.array([int(j) for j in path], dtype=mx.int32), axis=0)
        mx.eval(rows)
        for i, j in enumerate(path):
            self.mem[int(toks[j])] = rows[i]

    def _draw(self, shape: tuple[int, ...]) -> Any:
        import mlx.core as mx

        self._key, sub = mx.random.split(self._key)
        return mx.random.normal(shape, key=sub)

    def _embed(self, tok: int) -> Any:
        import mlx.core as mx

        return self.sm._mlx_embed_rows(mx.array([tok], dtype=mx.int32))

    def _input(self, h: Any, tok: int | None = None) -> Any:
        """the tail's input at the position holding `tok` (the newest committed token unless given) from the
        boundary residual `h` [1, H], by `inject`"""
        import mlx.core as mx

        sp = self.spec
        if sp.inject == "stale":
            return h
        if sp.inject == "zero":
            return mx.zeros_like(h)
        if sp.inject == "noise":
            rms = mx.sqrt(mx.mean(h.astype(mx.float32) ** 2))
            return (self._draw(tuple(h.shape)) * rms).astype(h.dtype)
        if tok is None:
            tok = self.cur
        assert tok is not None
        if sp.inject == "memory":
            # the model's own residual for this token at its latest earlier position, blended in by alpha (1: that
            # residual alone); a token not seen yet leaves the stale one
            got = self.mem.get(tok)
            return h if got is None else h + sp.alpha * (got[None] - h)
        if sp.inject == "unembed":
            get = getattr(getattr(self.sm.head_host, "mx", None), "get", None)
            if get is None:
                raise RuntimeError("tail_draft inject=unembed needs the head's table on MLX")
            e = get()[tok][None]
        else:
            e = self._embed(tok)
            if sp.inject == "delta" and self.prev is not None:
                e = e - self._embed(self.prev)
        # the embedding brought to the residual's rms times alpha: raw, it is small beside a late residual
        hf, ef = h.astype(mx.float32), e.astype(mx.float32)
        scale = sp.alpha * mx.sqrt(mx.mean(hf**2)) / (mx.sqrt(mx.mean(ef**2)) + 1e-6)
        return (hf + scale * ef).astype(h.dtype)

    def propose_chains(self, v: int, tree: bool = True) -> list[tuple[list[int], str]]:
        if v <= 0 or self.row is None or self.cur is None or not self.armed:
            return []
        t0 = time.perf_counter()
        out = self._draft(v)
        dt = time.perf_counter() - t0
        self.step_s = dt if self.step_s is None else 0.8 * self.step_s + 0.2 * dt
        self._drafted = bool(out)
        return out

    def _draft(self, v: int) -> list[tuple[list[int], str]]:
        h_all = getattr(self.sm, "_tail_h", None)
        if h_all is None:
            return []
        import mlx.core as mx

        sm, sp, cache = self.sm, self.spec, self.cache
        n = int(cache.layers[self.start].get_seq_length())
        x = self._input(h_all[self.row][None])
        rows = sp.k if sp.sample == "cloud" else 1
        if rows > 1:
            # the cloud: k copies spread by noise, every one a root at the next position attending the prefix alone
            rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2))
            x = x + (self._draw((rows, int(x.shape[1]))) * (sp.noise * rms)).astype(x.dtype)
            sm.aa([-1] * rows)
        with torch.inference_mode():
            try:
                lg = sm._forward_mlx(None, None, cache, None, rows == 1, True, sm.L, hm=x, start=self.start)
            finally:
                if rows > 1:
                    sm.ab()
        self._crop(n)
        self.passes += 1
        lg = lg[0].float()
        if sp.sample == "cloud":
            ids = lg.argmax(dim=-1).tolist()
        elif sp.sample == "temp":
            p = torch.softmax(lg[-1] / sp.temp, dim=-1)
            ids = torch.multinomial(p, 4 * sp.k, replacement=True, generator=self._gen).tolist()
        else:
            top = torch.topk(lg[-1], sp.k)
            ids = top.indices.tolist()
            if sp.minp > 0:
                # only the candidates the tail believes in: fewer rows, so a small tile keeps room for other chains
                probs = torch.softmax(lg[-1], dim=-1)[top.indices].tolist()
                ids = [t for t, q in zip(ids, probs) if q >= sp.minp]
        out: list[int] = []
        for t in ids:
            if int(t) not in out:
                out.append(int(t))
            if len(out) >= sp.k:
                break
        if sp.depth <= 1 or sp.sample == "cloud" or not out:
            return [([t], self.tag) for t in out]
        # deeper: under each candidate the tail's top pick, a level a pass. The tree is the verify pass's shape -
        # node 0 the newest token at the next position, the candidates its children, each node at past + its
        # depth - re-run whole as one tail pass a level (cropped after), every node's input the anchor residual
        # with its own token's memory; the chains are the paths below node 0
        anchor = h_all[self.row][None]
        cur = self.cur
        assert cur is not None  # propose_chains checked
        nodes_t, nodes_p = [cur, *out], [-1] + [0] * len(out)
        leaves = list(range(1, len(nodes_t)))
        for _ in range(1, sp.depth):
            x = mx.concatenate([self._input(anchor, t) for t in nodes_t], axis=0)
            sm.aa(nodes_p)
            with torch.inference_mode():
                try:
                    lg = sm._forward_mlx(None, None, cache, None, False, True, sm.L, hm=x, start=self.start)
                finally:
                    sm.ab()
            self._crop(n)
            self.passes += 1
            lg = lg[0].float()
            grown = []
            for leaf in leaves:
                nodes_t.append(int(lg[leaf].argmax()))
                nodes_p.append(leaf)
                grown.append(len(nodes_t) - 1)
            leaves = grown
        chains = []
        for leaf in leaves:
            path = []
            j = leaf
            while j > 0:
                path.append(nodes_t[j])
                j = nodes_p[j]
            chains.append((path[::-1], self.tag))
        return chains

    def _crop(self, n: int) -> None:
        """the tail layers back to the committed length: their rows past it were a draft's"""
        for i in range(self.start, self.sm.L):
            cl = self.cache.layers[i]
            if cl.get_seq_length() > n:
                cl.crop(n)

    def propose_with_source(self, v: int) -> tuple[list[int], Any]:
        chains = self.propose_chains(v)
        return (chains[0][0], self.tag) if chains else ([], None)
