# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A small model of the same family as the drafter: its greedy tree over the committed tokens, verified by the
big model's tree pass (every output token the big model's own). The proposer's interface is the n-gram one."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch


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

    def __init__(self, model: ModelProposer, ngram: Any) -> None:
        self.model, self.ngram = model, ngram

    def add_sequence(self, ids: Iterable[int], tag: Any) -> int:
        return self.ngram.add_sequence(ids, tag)

    def extend(self, tok_id: int) -> None:
        self.model.extend(tok_id)
        self.ngram.extend(tok_id)

    def propose_chains(self, v: int) -> list[tuple[list[int], str]]:
        return self.model.propose_chains(v) + self.ngram.propose_chains(v)

    def propose_with_source(self, v: int) -> tuple[list[int], Any]:
        out = self.model.propose_with_source(v)
        return out if out[0] else self.ngram.propose_with_source(v)
