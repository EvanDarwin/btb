# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's MTP drafter (btb/engine/families/qwen4/drafter.py) over tiny_q4's drafting head, on the CPU: its layer's
experts are the expert store's - read as the store's layer L, one past the trunk's, never placed with the drafter -
its dense tensors are exactly what the planner prices, its tree's branches carry the indexer's keys with K and V,
and a speculative decode with it as the proposer (a chain, a fixed tree, its own tree) is the plain greedy decode's,
short of the indexer's budget and past it. One load of the model; every case runs inside it."""

from __future__ import annotations

import os

import torch

from btb.engine import StreamedTextModel
from btb.engine.cache import indexer_keys
from btb.engine.families.qwen4.drafter import Qwen4Drafter
from btb.engine.host import _Experts
from tests.helpers import fixture, host_model, speculation

# a prompt short of the indexer's budget (8 blocks of 4), and one past it, where both the trunk's sparse attention
# and the drafter's select
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61],
    "long": ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70],
}
PROPOSERS = ("mtp_dyn", "mtp_tree", "mtp")


def test_the_drafter_reads_its_experts_through_the_store_and_its_speculation_is_the_plain_decode() -> None:
    StreamedTextModel.register_attention()
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q4"))
        try:
            store = sm.expert_store
            assert store is not None, "tiny_q4 on the host runs its mixture through the expert store"
            dr = sm.mtp_drafter()
            assert isinstance(dr, Qwen4Drafter)
            L = int(sm.L)
            ex = dr.layer.mlp.experts
            assert isinstance(ex, _Experts) and ex.layer == L and ex.base == "mtp.layers.0.mlp.experts."
            # no expert tensor placed: none among the drafter's tensors, none read whole by the layer
            placed = dr.named_tensors()
            assert not [k for k in placed if ".experts." in k] and ex.gate_up is None and ex.down is None
            held = [n for m in (dr.layer, dr.head) for n, _t in m.named_parameters()]
            assert not [n for n in held if ".experts." in n], held
            # the dense tensors are the head's every non-expert tensor, at the bytes the planner prices
            dense = {k for k in sm.weight_map if k.startswith("mtp.") and sm.fam.dense_key(k)}
            assert set(placed) == dense and set(dr.dense_keys) == dense
            assert 2 * dr.dense_numel == sm._drafter_bytes() > 0

            sm.expert_trace = []
            for name, prompt in PROMPTS.items():
                plain = sm.generate_greedy(prompt, 16)
                for proposer in PROPOSERS:
                    speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
                    toks, census = sm.generate_speculative(prompt, 16, proposer=proposer, v_max=4)
                    assert toks == plain, f"{name}, {proposer}: {toks} != {plain}"
                    assert census["proposed"] > 0, f"{name}, {proposer}: nothing was drafted, so nothing verified"
                    assert census["mtp_steps"] > 0, f"{name}, {proposer}: the drafter never stepped"
                    # the tree's branches leave the drafter's cache as the root left it: the indexer's keys with K/V
                    cl = dr.cache.layers[0]
                    ik = indexer_keys(cl)
                    assert ik is not None and ik.shape[0] == cl.keys.shape[0] == 1
                    assert ik.shape[1] == cl.keys.shape[-2], f"{name}, {proposer}: indexer keys out of step"
            # the drafter's calls were the store's: its layer's picks recorded under L, its recipe read from mtp.*
            assert any(layer == L for layer, _picks in sm.expert_trace), "the drafter's layer never called the store"
            shard = sm.weight_map["mtp.layers.0.mlp.experts.gate_up_proj"]
            assert L in store.recipes and os.path.basename(store.recipes[L][0][0]) == os.path.basename(shard)
            assert any(key[0] == L for key in store.rides), "no expert of the drafter's layer rode the store"
        finally:
            sm.expert_trace = None
            sm.close()
