# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's sparse attention indexer through btb (btb/engine/families/qwen4/qsa.py) against the reference's own
forward: by default the selected-token mask is the reference's bit for bit - on the host and the card, float32 and
bf16, a first chunk and one continuing a cache, a bool mask and an additive one, a speculative pass's tree - and any
other mask takes the reference's forward. `sparse` (`--sparse`) is held to the reference indexer's choices: the share of the rows'
selected tokens it agrees on."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

from btb.engine.families.qwen4.qsa import install
from tests.helpers import fixture, layer_count

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


class _Keys:
    """a cache's indexer keys alone, as `update_indexer` keeps them: the layer's keys so far, the new ones appended"""

    def __init__(self, past: torch.Tensor | None) -> None:
        self.keys = past

    def update_indexer(self, new: torch.Tensor, layer_idx: int) -> torch.Tensor:
        self.keys = new if self.keys is None else torch.cat([self.keys, new], dim=1)
        return self.keys


def _parts(budget: int, dtype: torch.dtype, dev: str) -> tuple[Any, Any, Any]:
    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cfg = AutoConfig.from_pretrained(fixture("tiny_q4"))
    cfg.indexer_budget = budget
    g = torch.Generator().manual_seed(budget)
    ix = m.Qwen4ExpTextQSAIndexer(cfg, layer_idx=3)
    for p in ix.parameters():
        p.data = (torch.randn(p.shape, generator=g) * 0.5 + (1.0 if p.dim() == 1 else 0.0)).to(dtype)
    rot = m.Qwen4ExpTextRotaryEmbedding(config=cfg)
    return cfg, ix.to(dev), rot.to(dev)


def _call(
    ix: Any, cfg: Any, rot: Any, S: int, past: int, dtype: torch.dtype, dev: str, bool_mask: bool, seed: int
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, S, cfg.hidden_size, generator=g).to(dev, dtype)
    prev = torch.randn(1, past, cfg.indexer_head_dim, generator=g).to(dev, dtype) if past else None
    kv = past + S
    pos = torch.arange(kv, device=dev).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(h, pos)
    allow = torch.arange(kv, device=dev)[None, :] <= (torch.arange(S, device=dev) + past)[:, None]
    mask: torch.Tensor = allow.view(1, 1, S, kv)
    if not bool_mask:
        mask = torch.where(mask, torch.zeros((), dtype=dtype, device=dev), torch.finfo(dtype).min)
    with torch.no_grad():
        return ix(h, (cos.to(dtype), sin.to(dtype)), mask, _Keys(prev))


def _pair(budget: int, dtype: torch.dtype, dev: str, sparse: bool = False) -> tuple[Any, Any, Any, Any]:
    cfg, ref, rot = _parts(budget, dtype, dev)
    ours = copy.deepcopy(ref)
    install(ours, sparse=sparse)
    return cfg, ref, ours, rot


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", [8, 64])
def test_the_selection_is_the_references_bit_for_bit(dev: str, dtype: torch.dtype, budget: int) -> None:
    cfg, ref, ours, rot = _pair(budget, dtype, dev)
    for S, past, bool_mask in ((40, 0, True), (150, 0, True), (33, 71, True), (150, 0, False), (9, 300, False)):
        a = _call(ref, cfg, rot, S, past, dtype, dev, bool_mask, seed=S + past)
        b = _call(ours, cfg, rot, S, past, dtype, dev, bool_mask, seed=S + past)
        assert a.dtype == b.dtype and a.shape == b.shape, (a.shape, b.shape)
        assert torch.equal(a, b), f"S={S} past={past} bool={bool_mask}: {int((a != b).sum())} entries part"


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_trees_selection_is_the_references_bit_for_bit(dev: str, dtype: torch.dtype) -> None:
    """a verify pass's rows - the prefix, then a tree whose rows see themselves and their ancestors, roped at their
    depth - select what the reference selects row for row: a prefix a whole number of blocks and one that is not
    (its tail shares a block with the rows' own), short of the budget and past it"""
    from btb.engine.forward import tree_mask

    parents = [-1, 0, 1, 2, 1, 4, 0, 6, 7, 3, 9, 10]
    depth = [0]
    for p in parents[1:]:
        depth.append(depth[p] + 1)
    cfg, ref, ours, rot = _pair(8, dtype, dev)
    S = len(parents)
    for past, bool_mask in ((16, True), (37, True), (64, False), (130, True), (211, False)):
        g = torch.Generator().manual_seed(past)
        h = torch.randn(1, S, cfg.hidden_size, generator=g).to(dev, dtype)
        prev = torch.randn(1, past, cfg.indexer_head_dim, generator=g).to(dev, dtype)
        pos = torch.tensor(list(range(past)) + [past + d for d in depth], device=dev).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rot(h, pos)
        kv = past + S
        causal = (torch.arange(kv, device=dev)[None, :] <= (torch.arange(S, device=dev) + past)[:, None]).view(
            1, 1, S, kv
        )
        mask = tree_mask(causal, past, parents)
        if not bool_mask:
            mask = torch.where(mask, torch.zeros((), dtype=dtype, device=dev), torch.finfo(dtype).min)
        with torch.no_grad():
            a = ref(h, (cos.to(dtype), sin.to(dtype)), mask, _Keys(prev))
            b = ours(h, (cos.to(dtype), sin.to(dtype)), mask, _Keys(prev))
        assert torch.equal(a, b), f"past={past} bool={bool_mask}: {int((a != b).sum())} entries part"


def test_a_padded_mask_takes_the_references_forward() -> None:
    cfg, ref, ours, rot = _pair(8, torch.float32, "cpu")
    g = torch.Generator().manual_seed(1)
    S = 24
    h = torch.randn(1, S, cfg.hidden_size, generator=g)
    pos = torch.arange(S).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(h, pos)
    allow = torch.arange(S)[None, :] <= torch.arange(S)[:, None]
    allow[:, :5] = False  # five left-padded positions: no row sees them
    mask = allow.view(1, 1, S, S)
    with torch.no_grad():
        a = ref(h, (cos, sin), mask, _Keys(None))
        b = ours(h, (cos, sin), mask, _Keys(None))
    assert torch.equal(a, b)


@pytest.mark.parametrize("dev", DEVICES)
def test_sparse_agrees_with_the_reference_indexer(dev: str) -> None:
    """`--sparse` against the reference indexer's choices: the rows' selected tokens it keeps too, reported and
    held above 99% (the one-pass sum parts from the reference's only at a near-tie on the budget's edge)"""
    dtype = torch.bfloat16
    cfg, ref, ours, rot = _pair(64, dtype, dev, sparse=True)
    same = total = 0
    for S, past in ((150, 0), (64, 200), (257, 31)):
        a = _call(ref, cfg, rot, S, past, dtype, dev, True, seed=S + past)
        b = _call(ours, cfg, rot, S, past, dtype, dev, True, seed=S + past)
        same += int((a & b).sum())
        total += int(a.sum())
    agree = same / total
    print(f"[sparse] {agree:.4%} of the reference's selected tokens kept")
    assert agree > 0.99, agree


def test_the_option_reaches_every_indexer() -> None:
    from btb.engine import StreamedTextModel
    from btb.options import check

    assert check({"sparse": 1})["sparse"] in (1, True)
    for sparse in (False, True):
        sm = StreamedTextModel(
            fixture("tiny_q4"),
            resident_head=True,
            device="cpu",
            cpu_layers=range(layer_count(fixture("tiny_q4"))),
            sparse=sparse,
            log=lambda *a: None,
        )
        try:
            flags = {m.btb_sparse for layer in sm.host.values() for m in layer.modules() if hasattr(m, "btb_sparse")}
            assert sm.host and flags == {sparse}, flags
        finally:
            sm.close()
