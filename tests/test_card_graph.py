# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card graph at the engine level, on Qwen3-0.6B from the local cache: a verify pass of any shape
reproduces the greedy steps bit for bit, the in-place tree commit leaves the cache exactly as the greedy
steps would, and the speculative loop's answer is the greedy answer. Skipped without a card, the fatbin, or
the cached model (never downloads)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import pytest
import torch
from pytest import FixtureRequest

from btb.engine.families import Family
from btb.kinds import FamilyKind, Tokens
from tests.helpers import cached, checkout, forward_logits

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from transformers.cache_utils import DynamicCache

    from btb.engine.model import StreamedTextModel

# the (sm, tok) an engine fixture yields, and what greedy_reference returns: prompt ids, greedy tokens, logits
EngineTok = tuple["StreamedTextModel", "PreTrainedTokenizerBase"]
GreedyRef = tuple[list[int], list[int], list[torch.Tensor]]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module", params=["fp32-chain", "tensor-cores"])
def engine(request: FixtureRequest) -> Iterator[EngineTok]:
    """the engine with one GEMV kernel pinned for every pass width, each kernel in turn: the bit-equalities
    below are held on both (the default engine takes one kernel for every width itself, tested last)"""
    import btb

    if btb.kernels_path() is None:
        pytest.skip("btb_kernels.fatbin not built")
    path = cached(MODEL)
    if path is None:
        pytest.skip(f"{MODEL} is not in the local cache")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    sm = btb.load(path, device="cuda", cpu_layers=0, log=None)
    if not sm._card_family_ok() or len(sm.resident) != sm.L:
        pytest.skip("the model does not take the card graph on this machine")
    if request.param == "tensor-cores" and not sm._card_mma_avail():
        pytest.skip("the tensor-core GEMV is not in the fatbin")
    setattr(sm, "card_mma", request.param == "tensor-cores")  # noqa: B010  a getattr config knob
    yield sm, tok
    sm.close()


@pytest.fixture(scope="module")
def prompt_ids(engine: EngineTok) -> list[list[int]]:
    from btb import template

    sm, tok = engine
    with open(checkout("bench", "questions.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [
        tok(template(tok, [{"role": "user", "content": r["prompt"]}]), add_special_tokens=False)["input_ids"]
        for r in rows
    ]


@pytest.fixture(scope="module")
def greedy_reference(engine: EngineTok, prompt_ids: list[list[int]]) -> GreedyRef:
    """the greedy tokens of prompt 0 and the logits of every step, built one token at a time"""
    sm, _ = engine
    ids = prompt_ids[0]
    with torch.inference_mode():
        cache = sm.new_cache(max_len=len(ids) + 64)
        lg = sm._prefill(torch.tensor([ids]), cache)
        toks = [int(lg[0, -1].argmax())]
        logits = []
        for _ in range(30):
            lg = forward_logits(sm, torch.tensor([[toks[-1]]]), cache=cache)[0, -1].float()
            logits.append(lg.clone())
            toks.append(int(lg.argmax()))
    return ids, toks, logits


def _fresh(sm: StreamedTextModel, ids: Tokens, toks: Tokens, k: int) -> DynamicCache:
    cache = sm.new_cache(max_len=len(ids) + 64)
    sm._prefill(torch.tensor([ids]), cache)
    for j in range(k):
        forward_logits(sm, torch.tensor([[toks[j]]]), cache=cache)
    return cache


@pytest.mark.parametrize("k,T", [(0, 2), (0, 9), (5, 4), (5, 17), (5, 32)])
def test_a_chain_pass_reproduces_the_greedy_steps_bit_for_bit(
    engine: EngineTok, greedy_reference: GreedyRef, k: int, T: int
) -> None:
    sm, _ = engine
    ids, toks, logits = greedy_reference
    T = min(T, len(logits) - k)
    with torch.inference_mode():
        cache = _fresh(sm, ids, toks, k)
        sm.aa(None)
        try:
            out = forward_logits(sm, [toks[k : k + T]], cache=cache, last_only=False)[0].float()
        finally:
            sm.ab()
    for t in range(T):
        assert torch.equal(out[t], logits[k + t]), f"row {t} of a {T}-row pass at step {k + t}"


def test_a_tree_pass_reproduces_the_greedy_steps_on_every_branch(
    engine: EngineTok, greedy_reference: GreedyRef
) -> None:
    sm, _ = engine
    ids, toks, logits = greedy_reference
    P, k = len(ids), 5
    wrong = (toks[k + 1] + 7) % 150000
    # root; a wrong branch 1 -> 2; the greedy continuation as the second branch 3 -> 4
    parents, depth = [-1, 0, 1, 0, 3], [0, 1, 2, 1, 2]
    with torch.inference_mode():
        cache = _fresh(sm, ids, toks, k)
        sm.aa(parents)
        try:
            out = forward_logits(
                sm,
                [[toks[k], wrong, wrong, toks[k + 1], toks[k + 2]]],
                cache=cache,
                last_only=False,
                positions=[[P + k + d for d in depth]],
            )[0].float()
        finally:
            sm.ab()
        assert torch.equal(out[0], logits[k])
        assert torch.equal(out[3], logits[k + 1])
        assert torch.equal(out[4], logits[k + 2])
        # the in-place commit of the second branch, then the next steps are the greedy steps
        sm.ad(cache, P + k, [0, 3, 4])
        assert cache.get_seq_length() == P + k + 3
        lg = forward_logits(sm, torch.tensor([[toks[k + 3]]]), cache=cache)[0, -1].float()
        assert torch.equal(lg, logits[k + 3])
        # and a second tree straight after the gather
        k2 = k + 4
        sm.aa(parents)
        try:
            out = forward_logits(
                sm,
                [[toks[k2], toks[k2 + 1], toks[k2 + 2], wrong, wrong]],
                cache=cache,
                last_only=False,
                positions=[[P + k2 + d for d in depth]],
            )[0].float()
        finally:
            sm.ab()
        for t in range(3):
            assert torch.equal(out[t], logits[k2 + t])


@pytest.mark.parametrize("path", [[0, 1], [0, 1, 2], [0]])
def test_a_chain_commit_crops_in_place(engine: EngineTok, greedy_reference: GreedyRef, path: Sequence[int]) -> None:
    sm, _ = engine
    ids, toks, logits = greedy_reference
    P, k = len(ids), 5
    wrong = (toks[k + 1] + 7) % 150000
    with torch.inference_mode():
        cache = _fresh(sm, ids, toks, k)
        sm.aa(None)
        try:
            forward_logits(sm, [[*toks[k : k + 3], wrong, wrong, wrong]], cache=cache, last_only=False)
        finally:
            sm.ab()
        sm.ad(cache, P + k, path)
        n = P + k + len(path)
        assert cache.get_seq_length() == n
        nxt = k + len(path)
        lg = forward_logits(sm, torch.tensor([[toks[nxt]]]), cache=cache)[0, -1].float()
        assert torch.equal(lg, logits[nxt])


def test_the_step_graph_loop_gives_the_step_by_step_tokens(
    engine: EngineTok, greedy_reference: GreedyRef, prompt_ids: list[list[int]]
) -> None:
    """the self-advancing graph (embedding, layers, head, argmax and the next token inside one replay, the
    host a token behind) returns exactly the tokens of the one-step-at-a-time loop, on every prompt"""

    sm, _ = engine
    ids, toks, _ = greedy_reference
    assert sm._card_greedy_ok(None, 1, None, None, False, None)
    out, _ = sm.generate(ids, len(toks), eos=(), speculate=False)
    assert out == toks
    # against the step loop itself, with the pipeline switched off, and with an eos that stops it early
    for pid in prompt_ids:
        sm.card_pipeline = False
        try:
            ref, _ = sm.generate(pid, 40, eos=(), speculate=False)
        finally:
            sm.card_pipeline = True
        got, _ = sm.generate(pid, 40, eos=(), speculate=False)
        assert got == ref
        cut, _ = sm.generate(pid, 40, eos=(ref[20],), speculate=False)
        assert cut == ref[: ref.index(ref[20]) + 1]


def test_the_card_gate_declines_another_family_without_reading_its_shapes() -> None:
    """the gate names the family before it reads the config: a mixture of experts (qwen4, gpt-oss) carries
    no `intermediate_size`, and reading it first crashed the 180B on load"""
    from types import SimpleNamespace

    from btb.engine.cuda import _CudaMixin

    class _Gate(_CudaMixin):
        def __init__(self) -> None:
            self.fam = Family(kind=FamilyKind.QWEN4, own=True)
            self.mlx = None
            self.resident_fp32 = False
            self.compute_dtype = None
            self.cfg = SimpleNamespace(
                hidden_size=2048,
                num_attention_heads=16,
                num_key_value_heads=4,
                head_dim=128,
                moe_intermediate_size=1024,
            )

    assert _Gate()._card_family_ok() is False
    # and a dims read on such a config takes the experts' width rather than raising
    H, Hq, Hk, D, I = _Gate()._card_dims()
    assert (H, Hq, Hk, D, I) == (2048, 16, 4, 128, 1024)


@pytest.fixture(scope="module")
def default_engine() -> Iterator[StreamedTextModel]:
    """the engine as `btb.load` leaves it: nothing forced, the warm-up's own kernel choice"""
    import btb

    if btb.kernels_path() is None:
        pytest.skip("btb_kernels.fatbin not built")
    path = cached(MODEL)
    if path is None:
        pytest.skip(f"{MODEL} is not in the local cache")
    forced = os.environ.pop("BTB_CARD_MMA", None)
    try:
        sm = btb.load(path, device="cuda", cpu_layers=0, log=None)
    finally:
        if forced is not None:
            os.environ["BTB_CARD_MMA"] = forced
    if not sm._card_family_ok() or len(sm.resident) != sm.L or not sm._card_mma_avail():
        sm.close()
        pytest.skip("the model does not take both GEMV kernels on this machine")
    yield sm
    sm.close()


def test_the_default_engine_takes_one_gemv_for_every_width_and_is_exact(default_engine: StreamedTextModel) -> None:
    """the warm-up's choice is one kernel for the whole engine: the two sum a row in different orders, and a
    step on one with a pass on the other parted at bf16 near-ties (0/8 identical at 256 tokens on this model);
    on one kernel a step and a 15-row pass agree bit for bit, and the speculative answer is the greedy answer"""
    sm = default_engine
    choice = sm._cg["mma_for"]
    assert choice and len(set(choice.values())) == 1, f"a kernel per width: {choice}"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "bench", "questions.jsonl"), encoding="utf-8") as fh:
        prompts = [sm.prompt_ids(json.loads(line)["prompt"]) for line in fh if line.strip()][:3]
    for ids in prompts:
        g = sm.generate(ids, 128, eos=(), speculate=False).tokens
        with torch.inference_mode():
            cache = sm.new_cache()
            forward_logits(sm, [ids], cache=cache)
            steps = [forward_logits(sm, [[t]], cache=cache)[0, -1].clone() for t in g[:15]]
            cache = sm.new_cache()
            forward_logits(sm, [ids], cache=cache)
            wide = forward_logits(sm, [g[:15]], cache=cache, last_only=False)[0]
        for i, step in enumerate(steps):
            assert torch.equal(step, wide[i]), f"position {i}: a one-row step and a 15-row pass differ"
        s, census = sm.generate(ids, 128, eos=())
        assert s == g, f"diverged at token {next(i for i, (a, b) in enumerate(zip(g, s)) if a != b)} of {len(g)}"
        assert census["forwards"] < len(s)


def test_the_step_graph_samples_as_the_step_loop_and_repeats_under_a_seed(
    engine: EngineTok, prompt_ids: list[list[int]]
) -> None:
    """under a temperature the self-advancing graph draws inside the replay from the card's generator: a seed
    repeats its answer, another seed changes it, the answer is not the greedy one; the step loop with the
    pipeline off and the speculative loop draw by the cache row and agree with each other"""
    from btb.sampling import Sampling

    sm, _ = engine
    s1, s2 = Sampling(temperature=0.8, top_p=0.9, seed=1), Sampling(temperature=0.8, top_p=0.9, seed=2)
    ids = prompt_ids[0]
    a, _ = sm.generate(ids, 40, eos=(), speculate=False, sampling=s1)
    b, _ = sm.generate(ids, 40, eos=(), speculate=False, sampling=s1)
    c, _ = sm.generate(ids, 40, eos=(), speculate=False, sampling=s2)
    g, _ = sm.generate(ids, 40, eos=(), speculate=False)
    assert a == b and a != c and a != g
    sm.card_pipeline = False
    try:
        ref, _ = sm.generate(ids, 40, eos=(), speculate=False, sampling=s1)
    finally:
        sm.card_pipeline = True
    spec, census = sm.generate(ids, 40, eos=(), sampling=s1)
    assert spec == ref, f"diverged at token {next(i for i, (x, y) in enumerate(zip(ref, spec)) if x != y)}"
    assert census["seed"] == 1


def test_the_speculative_answer_is_the_greedy_answer(engine: EngineTok, prompt_ids: list[list[int]]) -> None:

    sm, _ = engine
    for ids in prompt_ids:
        g, _ = sm.generate(ids, 128, eos=(), speculate=False)
        o, census = sm.generate(ids, 128, eos=())
        assert o == g, f"diverged at token {next(i for i, (a, b) in enumerate(zip(g, o)) if a != b)} of {len(g)}"
        assert census["forwards"] < len(o)
