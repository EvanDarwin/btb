# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's speculative verify (btb/engine/families/qwen4/verify.py, qsa.py's tree rows): every node of a tree pass
computes as the one-token steps of its own path, the commit leaves the cache where those steps would, and a
speculative decode is the plain decode's tokens - past the indexer's budget, where the sparse attention selects, and
at its edge, where a tree's rows fall on both sides of it. On the CPU over tiny_q4, whose indexer keeps 2 blocks of 4
(`block_topk`, a budget of 8 positions): a row that sees past 8 positions selects. One load of the model; every
case runs inside it."""

from __future__ import annotations

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.forward import path_of
from btb.engine.native import Native
from btb.kinds import LayerKind
from tests.helpers import CHUNK, DEPTH, PARENTS, PATH, fixture, forward_logits, host_model, speculation

# a prompt of 16 and one of 70 (repeating, so the n-gram proposer drafts), both past the budget, and one at its
# edge: its 8 positions fill the 2 blocks, so a tree's shallow rows keep every block they see and its deepest select
_LONG = ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70]
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61],
    "long": _LONG,
    "edge": _LONG[:8],
}
NEXT = 77  # the token fed after the commit


def _steps(sm: StreamedTextModel, prompt: list[int], toks: list[int]) -> list[torch.Tensor]:
    """the logits after each of `toks` fed one at a time over the prefilled prompt: the plain decode's steps"""
    cache = sm.new_cache()
    forward_logits(sm, [prompt], cache)
    return [forward_logits(sm, [[t]], cache)[0, -1] for t in toks]


def test_a_tree_verifies_as_the_steps_of_its_paths_and_a_speculative_decode_is_the_plain_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    StreamedTextModel.register_attention()
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q4"))
        try:
            for name, prompt in PROMPTS.items():
                cache = sm.new_cache()
                forward_logits(sm, [prompt], cache)
                base = cache.get_seq_length()
                sm.aa(PARENTS)
                try:
                    lg = forward_logits(sm, [CHUNK], cache, last_only=False, positions=[[base + d for d in DEPTH]])[0]
                finally:
                    sm.ab()
                for j in range(len(CHUNK)):
                    path = path_of(PARENTS, j)[::-1]
                    want = _steps(sm, prompt, [CHUNK[k] for k in path])[-1]
                    assert torch.equal(lg[j], want), (
                        f"{name}: node {j} parts from its path's steps by {float((lg[j] - want).abs().max()):.3e}"
                    )
                sm.ad(cache, base, list(PATH))
                assert cache.get_seq_length() == base + len(PATH)
                nxt = forward_logits(sm, [[NEXT]], cache)[0, -1]
                want = _steps(sm, prompt, [CHUNK[k] for k in PATH] + [NEXT])[-1]
                assert torch.equal(nxt, want), (
                    f"{name}: the step after the commit parts by {float((nxt - want).abs().max()):.3e}"
                )

                plain = sm.generate_greedy(prompt, 16)
                # a random tiny model's greedy tokens seldom repeat the prompt's n-grams, so the proposer is handed
                # spans to draft from: the plain decode's own tokens, and a twin that leaves them after two, whose
                # drafts branch off the right ones and are refused
                right = [prompt[-1], *plain]
                astray = [*right[:3], *(t + 1 for t in right[3:])]
                spans = [("plain", right), ("astray", astray)]
                for tree_budget in (8, 0):  # the n-gram continuations as one tree, then as a chain
                    speculation(sm, tree_budget=tree_budget, tree_min_prob=0.0, ngram_p=0.9, v_max=4, price=False)
                    toks, census = sm.generate_speculative(prompt, 16, proposer="ngram", v_max=4, spans=spans)
                    assert toks == plain, f"{name}, tree budget {tree_budget}: {toks} != {plain}"
                    assert census["proposed"] > 0, f"{name}: nothing was drafted, so nothing was verified"
            assert sm.fam.verify_exact(sm, sm.new_cache())
            # without the host's DeltaNet step a host layer runs the reference module, which steps a tree's rows as
            # one chain: every pass is taken as a plain one-row pass (`verify_exact`), the plain decode's tokens
            prompt = PROMPTS["long"]
            with monkeypatch.context() as mp:
                mp.setattr(Native, "delta_step", None)
                assert not sm.fam.verify_exact(sm, sm.new_cache())
                plain = sm.generate_greedy(prompt, 16)
                speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.9, v_max=4, price=False)
                toks, census = sm.generate_speculative(
                    prompt, 16, proposer="ngram", v_max=4, spans=[("plain", [prompt[-1], *plain])]
                )
            assert toks == plain, f"no delta kernel: {toks} != {plain}"
            assert census["proposed"] == 0, "a pass drafted rows the host could not verify exactly"
        finally:
            sm.close()


def test_a_template_steps_and_commits_each_streamed_deltanet_layer_with_its_own_weights() -> None:
    """a template streams several DeltaNet layers through one module, refilled in place: the one-token step reads
    each layer's own float32 operands (`_consts` rebuilt at each layer switch, not the first layer's kept), and a
    speculative pass's commit steps each layer with the operands its pass read (`_kept`), not those of the last
    layer loaded into the module by then - a one-row pass committed is the plain step's, and so is the step after.
    tiny_q4 on the CPU in bf16, its PLE layer (1) on the host, which the DeltaNet template's structure lacks, and
    every other layer streamed. One load"""
    StreamedTextModel.register_attention()
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q4"), dtype=torch.bfloat16, cpu_layers=[1])
        try:
            linear = [i for i, lt in enumerate(sm.layer_types) if lt == LayerKind.LINEAR]
            streamed = [i for i in linear if i not in sm.host and i not in sm.resident]
            assert len(streamed) > 1, streamed
            prompt = PROMPTS["short"]
            steps = _steps(sm, prompt, [CHUNK[0], NEXT])
            la = sm.templates[sm.layer_types[streamed[0]]][0].linear_attn
            assert la.layer_idx == streamed[-1]
            c = la._btb_consts
            for k, t in (("conv_w", la.conv1d.weight.squeeze(1)), ("a_log", la.A_log), ("norm_w", la.norm.weight)):
                assert torch.equal(c[k], t.float()), f"the step's {k} is not layer {streamed[-1]}'s"
            cache = sm.new_cache()
            forward_logits(sm, [prompt], cache)
            base = cache.get_seq_length()
            sm.aa(None)
            try:
                lg = forward_logits(sm, [[CHUNK[0]]], cache, last_only=False)[0, -1]
            finally:
                sm.ab()
            assert torch.equal(lg, steps[0]), f"the pass parts from the step by {float((lg - steps[0]).abs().max())}"
            sm.ad(cache, base, [0])
            nxt = forward_logits(sm, [[NEXT]], cache)[0, -1]
            assert torch.equal(nxt, steps[1]), (
                f"the step after the commit parts by {float((nxt - steps[1]).abs().max()):.3e}"
            )
        finally:
            sm.close()
