# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A small model of the same family as the drafter: its greedy tree over the committed tokens, verified by the
big model's tree pass (every output token the big model's own). The proposer's interface is the n-gram one."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

import torch

from ..kinds import LayerKind
from ..sampling import GREEDY
from .generate import lin_layer

if TYPE_CHECKING:
    import threading

    from ..draft import NGramProposer
    from ..mlx.mega import MegaPass
    from .cache import KvCache
    from .generate import LinSnap
    from .model import StreamedTextModel

# drafts as chains, each with its source's tag (`ngram3`, `seq4`, `draft`)
Chains = list[tuple[list[int], str]]
# where `propose_with_source`'s chain came from: a banked span's (tag, id, position after the match), the n-gram
# corpus position it continues from, or a draft model's tag; None with no chain
ChainSource = tuple[str, int, int] | int | str | None


class ChainProposer(Protocol):
    """what the speculative loop drafts from without a drafting head: the n-gram proposer (btb/draft.py), or a draft
    model's chains ahead of it (`UnionProposer`); fed every committed token (`extend`)"""

    def add_sequence(self, ids: Iterable[int], tag: str) -> int: ...

    def extend(self, tok_id: int) -> None: ...

    def propose_chains(self, v: int) -> Chains: ...

    def propose_with_source(self, v: int) -> tuple[list[int], ChainSource]: ...


class ModelProposer:
    """A tree of drafts from a small engine, one pass of it a depth: pass 1 catches the drafter's cache up over
    the tokens committed since the last proposal and reads the root's top-ks[0]; each later pass re-runs the
    tree so far and reads its leaves' top-ks[d]; the chains are the root-to-leaf paths, at most `nodes` of
    them in the tree (the verify tile). The drafter's cache keeps the committed rows only; the tree's rows are
    cropped after each pass, and a hybrid's DeltaNet state put back to the committed rows'. Every pass costs the
    small model's step (the 0.6B: 11.5 ms in its kernel)."""

    def __init__(
        self,
        engine: StreamedTextModel,
        prompt: Iterable[int],
        ks: Sequence[int] = (3, 2, 1),
        nodes: int = 14,
        tag: str = "draft",
    ) -> None:
        self.sm = engine
        self.ks, self.nodes, self.tag = [int(k) for k in ks], int(nodes), tag
        ids = [int(t) for t in prompt]
        self.cache: KvCache = engine.new_cache()
        with torch.inference_mode(), self._held():
            engine._prefill(torch.tensor([ids]), self.cache)
        self.n = len(ids)
        self.pending: list[int] = []
        self.passes = 0
        self.marks: dict[int, LinSnap] = {}  # a hybrid drafter's DeltaNet state at the committed rows, by layer

    def _held(self) -> contextlib.AbstractContextManager[object]:
        """the drafting engine's decode lock, held for its passes as a decode holds it: its adapt watchers
        (`watch_vram_budget`, `watch_ram`) take a free lock for an idle engine, and moved its layers mid-pass"""
        lock: threading.RLock | None = getattr(self.sm, "_decode_lock", None)
        return lock if lock is not None else contextlib.nullcontext()

    def add_sequence(self, ids: Iterable[int], tag: str) -> int:
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
        mg: MegaPass | None = getattr(sm, "_mega", None)
        fast = mg is not None and sm._mega_ok(self.cache, T, None, True, GREEDY, None)
        # a node at the cache's next position plus its depth, as the target's verify pass places its tree: the torch
        # and card paths rotate a pass's rows at the positions it is handed (the card reads its depths off them), and
        # without them at past + row - every node past a depth's first one off; MLX takes the depths from `parents`
        depth: list[int] = []
        for p in parents:
            depth.append(0 if p < 0 else depth[p] + 1)
        pos = [[self.n + d for d in depth]]
        with torch.inference_mode(), self._held():
            sm.aa(parents)
            try:
                if fast:
                    sm.forward([toks], cache=self.cache, last_only=False, positions=pos, pick=True)
                else:
                    lg = sm.forward([toks], cache=self.cache, last_only=False, positions=pos)
            finally:
                sm.ab()
        self.passes += 1
        if not fast:
            assert lg is not None  # a whole pass with the head returns its logits (None only stops short of the head)
            return torch.topk(lg[0].float(), min(k, lg.shape[-1]), dim=-1).indices.tolist()
        assert mg is not None  # fast is set only when the megakernel is present
        import mlx.core as mx
        import numpy as np

        from .drafter import mlx_topk_ids

        ids = mlx_topk_ids(mg.logits(T), k)
        mx.eval(ids)
        return np.array(ids).tolist()

    def _mark(self) -> None:
        """the DeltaNet state at the committed rows: on MLX a drafter's tree pass, which keeps no checkpoints when
        the engine does not speculate, writes the state along the nodes it ran; `_crop` puts this back"""
        sm = self.sm
        linear = [i for i, kind in enumerate(sm.layer_types) if kind == LayerKind.LINEAR]
        self.marks = {i: sm._lin_snap(lin_layer(self.cache.layers[i])) for i in linear}

    def _crop(self) -> None:
        from transformers.cache_utils import DynamicLayer

        for i, cl in enumerate(self.cache.layers):
            snap = self.marks.get(i)
            if snap is not None:
                self.sm._lin_restore(lin_layer(cl), snap)
            elif isinstance(cl, DynamicLayer) and cl.get_seq_length() > self.n:
                cl.crop(self.n)

    def propose_chains(self, v: int, tree: bool = True) -> Chains:
        """the drafts as chains: the tree of `ks` where the target verifies trees, one greedy chain else"""
        if v <= 0 or not self.pending:
            return []
        ks = list(self.ks[:v]) if tree else [1] * min(v, len(self.ks))
        toks = self.pending
        self.pending = []
        # pass 1: the committed tokens as a chain; the last row's top-k are the depth-1 nodes
        k = min(ks[0], self.nodes)
        base = self.n
        row = self._topk(toks, list(range(-1, len(toks) - 1)), k)[-1]
        # the pass's rows committed as the target's verify pass commits its accepted path: on the host a tree pass
        # leaves a hybrid's DeltaNet state to the commit, and the whole chain is committed here
        self.sm.ad(self.cache, base, list(range(len(toks))))
        self.n += len(toks)
        self._mark()
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
        chains: Chains = []
        for leaf in leaves:
            path = []
            j = leaf
            while j >= 0:
                path.append(nodes_t[j])
                j = nodes_p[j]
            chains.append((path[::-1], self.tag))
        return chains

    def propose_with_source(self, v: int) -> tuple[list[int], ChainSource]:
        """one greedy chain, for a target that verifies chains (the host tier, a card without the tree)"""
        chains = self.propose_chains(v, tree=False)
        return (chains[0][0], self.tag) if chains else ([], None)


class UnionProposer:
    """The model's chains first, the n-gram proposer's after: both fed every committed token."""

    def __init__(self, model: ModelProposer, ngram: NGramProposer) -> None:
        self.model, self.ngram = model, ngram

    def add_sequence(self, ids: Iterable[int], tag: str) -> int:
        return self.ngram.add_sequence(ids, tag)

    def extend(self, tok_id: int) -> None:
        self.model.extend(tok_id)
        self.ngram.extend(tok_id)

    def propose_chains(self, v: int) -> Chains:
        return self.model.propose_chains(v) + self.ngram.propose_chains(v)

    def propose_with_source(self, v: int) -> tuple[list[int], ChainSource]:
        out = self.model.propose_with_source(v)
        return out if out[0] else self.ngram.propose_with_source(v)
