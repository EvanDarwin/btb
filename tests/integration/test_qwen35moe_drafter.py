# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3.5's MTP drafter (btb/engine/families/qwen35/drafter.py) over tiny_q35moe's drafting head, whose layer is a
mixture kept as Qwen3.5-MoE's checkpoints keep it, one tensor an expert: the experts are the expert store's, read
as the store's layer L from those pieces into the stacked layout, never placed with the drafter; the block computes
as transformers' over the checkpoint's own tensors; and a speculative decode with it as the proposer is the plain
greedy decode, on the host and through the drafter's MLX graph. The verify pass keeps a decode exact whatever the
drafter proposes, so the block is held to the reference itself."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.experts import Pieces
from btb.engine.families.qwen35.drafter import LAYER, Qwen35Drafter
from btb.engine.host import _Experts
from tests.helpers import fixture, host_model, loaded_model, need_mlx, speculation

PROMPT = [3, 17, 42, 5, 99, 120, 7, 7]
PROPOSERS = ("mtp", "mtp_tree", "mtp_dyn")


def _reference_block(sm: StreamedTextModel) -> torch.nn.Module:
    """transformers' mixture block in float32 over the head's tensors, its experts stacked from the per-expert ones"""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

    block = Qwen3_5MoeSparseMoeBlock(sm.cfg).float().eval()
    E, pre = int(sm.cfg.num_experts), LAYER + "mlp."
    get = lambda k: sm._get(pre + k).float()
    state = {
        "gate.weight": get("gate.weight"),
        "experts.gate_up_proj": torch.stack(
            [torch.cat([get(f"experts.{e}.gate_proj.weight"), get(f"experts.{e}.up_proj.weight")]) for e in range(E)]
        ),
        "experts.down_proj": torch.stack([get(f"experts.{e}.down_proj.weight") for e in range(E)]),
        "shared_expert_gate.weight": get("shared_expert_gate.weight"),
    }
    state |= {
        f"shared_expert.{k}.weight": get(f"shared_expert.{k}.weight") for k in ("gate_proj", "up_proj", "down_proj")
    }
    block.load_state_dict(state)
    return block


def _held_to_the_reference(sm: StreamedTextModel, dr: Qwen35Drafter) -> None:
    """the drafter's block over rows of the hidden width: transformers' to float32 rounding, its experts the store's
    layer L, each part read from the checkpoint's per-expert pieces"""
    store = sm.expert_store
    assert store is not None, "tiny_q35moe runs its mixture through the expert store"
    L = int(sm.L)
    ex = dr.layer.mlp.experts
    assert isinstance(ex, _Experts) and ex.layer == L and ex.base == LAYER + "mlp.experts."
    assert not [k for k in dr.named_tensors() if ".experts." in k], "an expert placed with the drafter"
    x = torch.randn(1, 5, int(sm.cfg.hidden_size), generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        got = dr.layer.mlp(x).float()
        want = _reference_block(sm)(x)
    assert all(isinstance(p[0], Pieces) for p in store.recipes[L]), store.recipes[L]
    d = float((got - want).abs().max())
    assert d <= 1e-4 * float(want.abs().max()), f"the drafter's mixture vs transformers': {d}"


def _speculates_as_plain(sm: StreamedTextModel) -> None:
    plain = sm.generate_greedy(PROMPT, 16)
    for proposer in PROPOSERS:
        speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
        toks, census = sm.generate_speculative(PROMPT, 16, proposer=proposer, v_max=4)
        assert toks == plain, f"{proposer}: {toks} != {plain}"
        assert census["proposed"] > 0 and census["mtp_steps"] > 0, f"{proposer}: the drafter never drafted"


def test_the_mixture_drafter_on_the_host() -> None:
    StreamedTextModel.register_attention()
    sm = host_model(fixture("tiny_q35moe"))
    try:
        dr = sm.mtp_drafter()
        assert isinstance(dr, Qwen35Drafter)
        _held_to_the_reference(sm, dr)
        _speculates_as_plain(sm)
    finally:
        sm.close()


def test_the_mixture_drafter_on_mlx() -> None:
    need_mlx()
    with loaded_model(fixture("tiny_q35moe"), device="mlx") as sm:
        dr = sm.mtp_drafter()
        assert isinstance(dr, Qwen35Drafter)
        assert dr._mlx_ready(), "the drafter's step is not its MLX graph"
        _held_to_the_reference(sm, dr)
        _speculates_as_plain(sm)


def test_the_mixture_drafter_drafts_through_a_load() -> None:
    """through btb.load on the CPU, whose warm-up prices the mixture's 2-row pass at about twice a step (each row
    reads its own experts), a load's first passes still draft at the whole width, so what the drafter returns sizes
    the passes; each proposer from a fresh pricing, and the decode the plain one"""
    with loaded_model(fixture("tiny_q35moe"), device="cpu") as sm:
        plain = sm.generate_greedy(PROMPT, 16)
        for proposer in PROPOSERS:
            speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
            sm._spec_cost = None  # nothing drafted on this load yet
            toks, census = sm.generate_speculative(PROMPT, 16, proposer=proposer, v_max=4)
            assert toks == plain, f"{proposer}: {toks} != {plain}"
            assert census["proposed"] > 0 and census["mtp_steps"] > 0, f"{proposer}: the drafter never drafted"


def test_the_mixture_drafter_reads_experts_wider_than_the_trunks(tmp_path: Path) -> None:
    """a GGUF whose drafting block keeps its experts Q8_0 beside the trunk's Q4_0: the store's slot parts are sized
    for the widest expert of any block, the NextN one's included, so the drafter reads its experts whole and the
    decode with it drafting is the plain one"""
    from btb.kinds import Quant
    from tests.helpers import GGUF_FIXTURES
    from tests.make_fixtures import retyped_gguf

    L = int(json.load(open(os.path.join(fixture("tiny_q35moe"), "config.json"), encoding="utf-8"))["num_hidden_layers"])
    out = str(tmp_path / "tiny_q35moe-q4_0-nextn-q8_0.gguf")
    nextn = {f"blk.{L}.ffn_{k}_exps.weight" for k in ("gate", "up", "down")}
    retyped_gguf(os.path.join(GGUF_FIXTURES, "tiny_q35moe-q4_0.gguf"), out, nextn, Quant.Q8_0)
    StreamedTextModel.register_attention()
    sm = host_model(out, cpu_layers=range(L))
    try:
        _speculates_as_plain(sm)
        store = sm.expert_store
        assert store is not None and store.types, "the experts are not read as stored"
        trunk = store.types[min(store.types)]
        assert set(store.types[L]) == {Quant.Q8_0} and set(trunk) == {Quant.Q4_0}, (store.types[L], trunk)
    finally:
        sm.close()


@pytest.mark.parametrize("stem", ["tiny_q35moe-f8_e4m3"])
def test_the_mixture_drafter_reads_fp8_pieces(stem: str) -> None:
    """the FP8 twin's head keeps each expert's matrices e4m3 with their own scale grids: the store joins the pieces
    and their grids into the stacked layout, and the decode with it drafting is the plain one"""
    StreamedTextModel.register_attention()
    sm = host_model(fixture(stem))
    try:
        dr = sm.mtp_drafter()
        store = sm.expert_store
        assert store is not None and store.f8
        dr.layer.mlp(torch.zeros(1, 1, int(sm.cfg.hidden_size)))
        assert all(isinstance(p[0], Pieces) for p in store.recipes[int(sm.L)])
        _speculates_as_plain(sm)
    finally:
        sm.close()
