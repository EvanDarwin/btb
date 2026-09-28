# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's speculative verify (btb/engine/families/qwen4/verify.py, qsa.py's tree rows): every node of a tree pass
computes as the one-token steps of its own path, the commit leaves the cache where those steps would, and a
speculative decode is the plain decode's tokens - short of the indexer's budget and past it, where the sparse
attention selects. On the CPU over tiny_q4, whose budget is 8 blocks of 4: a row past 32 positions selects. One load
of the model; every case runs inside it."""

from __future__ import annotations

import torch

from btb.engine import StreamedTextModel
from btb.engine.forward import path_of
from tests.helpers import CHUNK, DEPTH, PARENTS, PATH, fixture, forward_logits, host_model, speculation

# a prompt short of the budget, and one past it (repeating, so the n-gram proposer drafts)
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61],
    "long": ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70],
}
NEXT = 77  # the token fed after the commit


def _steps(sm: StreamedTextModel, prompt: list[int], toks: list[int]) -> list[torch.Tensor]:
    """the logits after each of `toks` fed one at a time over the prefilled prompt: the plain decode's steps"""
    cache = sm.new_cache()
    forward_logits(sm, [prompt], cache)
    return [forward_logits(sm, [[t]], cache)[0, -1] for t in toks]


def test_a_tree_verifies_as_the_steps_of_its_paths_and_a_speculative_decode_is_the_plain_one() -> None:
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
        finally:
            sm.close()
