# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Speculation's exactness for every family on every placement: each row of a verify pass - a chain's, and a tree's
path nodes beside a wrong sibling - is the one-row greedy step's at its position, bit for bit, and a speculative
decode is the greedy one. The placements: the card's own graph or program where the family has one, the card's torch
path (btb's kernels refused, as where they are absent - and the path Phi-3, Qwen3.5 and gpt-oss always take), the
attention cache in host RAM (kv_host), a layer on the host, and the CPU alone.

At a tiny fixture's widths cuBLAS runs one row and five through one kernel, so a step and a verify agreed there while
Qwen3-4B's parted (a kv_host verify's projections at five rows, its steps' at one) and gpt-oss-120b's did (its verify's
rows in one call). So each family's model here is built at real widths - two layers of Qwen3-4B's, Phi-4-mini's,
gpt-oss's attention, Qwen3.5's gated DeltaNet and attention, Gemma 3's with a small window, Qwen4's five layers at
its card program's head shapes - into the test's folder, loaded once a placement, and gone when its test ends (one
wide model on the disk at a time). The rows are compared as bits, never tokens: a random model's argmax shrugs off the ulp
a real model's finds dozens of tokens in. Every placement is run and every parting reported together."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from typing import Any

import pytest
import torch

from tests.helpers import loaded_model, need_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")

# a prompt past the small windows below, repeating so a decode repeats it (the n-gram drafter's food)
PROMPT = [3, 17, 42, 5, 99, 120, 7, 7, 200, 12, 45, 8] * 3
NEW = 5
SPEC_NEW = 32


def _wide(fam: str) -> tuple[Callable[..., None], dict[str, Any]]:
    """a family's builder and the widths it is built at here"""
    from tests.make_fixtures import build_gemma3, build_gpt_oss, build_phi3, build_q4_card, build_q35, build_qwen3

    specs: dict[str, tuple[Callable[..., None], dict[str, Any]]] = {
        "qwen3": (
            build_qwen3,
            {"hidden_size": 2560, "intermediate_size": 5120, "num_attention_heads": 32, "num_key_value_heads": 8,
             "head_dim": 128, "num_hidden_layers": 2, "vocab_size": 4096},
        ),
        "phi3": (
            build_phi3,
            {"hidden_size": 3072, "intermediate_size": 4096, "num_attention_heads": 24, "num_key_value_heads": 8,
             "num_hidden_layers": 2, "vocab_size": 4096},
        ),
        "gemma3": (
            build_gemma3,
            {"hidden_size": 1152, "intermediate_size": 2304, "num_attention_heads": 4, "num_key_value_heads": 1,
             "head_dim": 256, "num_hidden_layers": 6, "vocab_size": 4096, "sliding_window": 8},
        ),
        "gpt_oss": (
            build_gpt_oss,
            {"H": 2880, "HEADS": 64, "KV_HEADS": 8, "HEAD_DIM": 64, "INTER": 2880, "EXPERTS": 4, "TOP_K": 2,
             "LAYERS": 2, "VOCAB": 4096, "WINDOW": 4},
        ),
        "q35": (
            build_q35,
            {"hidden_size": 2048, "intermediate_size": 4096, "num_attention_heads": 16, "num_key_value_heads": 4,
             "head_dim": 256, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
             "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16,
             "linear_num_value_heads": 32, "vocab_size": 4096},
        ),
        # Qwen4 at the card program's shapes (`build_q4_card`: its kernels' head and indexer widths), as wide as a
        # real model's projections and experts
        "q4": (
            build_q4_card,
            {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
             "moe_intermediate_size": 768, "shared_expert_intermediate_size": 768, "num_experts": 16,
             "num_experts_per_tok": 4, "hc_lowrank": 64, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
             "linear_num_key_heads": 16, "linear_num_value_heads": 32, "indexer_n_heads": 16, "ple_embed_dim": 512,
             "vocab_size": 4096},
        ),
    }  # fmt: skip
    return specs[fam]


def _model(fam: str, root: str) -> str:
    """the family's model: built at real widths under `root`"""
    from tests.make_fixtures import write_tokenizer

    out = os.path.join(root, fam)
    build, over = _wide(fam)
    build(out, over=over)
    write_tokenizer(out)
    return out


def _steps(sm: Any, toks: list[int]) -> list[torch.Tensor]:
    """the logits of each one-row greedy step over `toks` after the prompt"""
    cache = sm.new_cache()
    sm.forward([list(PROMPT)], cache=cache)
    out = []
    for t in toks:
        o = sm.forward([[t]], cache=cache)
        assert o is not None
        out.append(o[0, -1].float().cpu())
    return out


def _verify(sm: Any, ids: list[int], parents: list[int] | None, positions: list[int] | None) -> torch.Tensor:
    """a verify pass's logits, every row: `parents` a tree's (None a chain), `positions` its rows' depths"""
    cache = sm.new_cache()
    sm.forward([list(PROMPT)], cache=cache)
    sm.aa(parents)
    try:
        out = sm.forward([ids], cache=cache, last_only=False, positions=[positions] if positions else None)
    finally:
        sm.ab()
    assert out is not None
    return out[0].float().cpu()


def _differs(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"{int((a != b).sum())} of {a.numel()} logits differ (max {float((a - b).abs().max()):.3g})"


def _partings(sm: Any, where: str) -> list[str]:
    """every way a verify's rows, or a speculative decode, part from the greedy steps on this placement"""
    out: list[str] = []
    toks = [int(t) for t in sm.generate(list(PROMPT), NEW, eos=(), speculate=False).tokens][:NEW]
    # asked as the speculative loop asks it: over the prompt's cache (an empty one is a prefill, which a family's
    # card program declines - Qwen4's read as speculating by one-row passes on every placement)
    probe = sm.new_cache()
    sm.forward([list(PROMPT)], cache=probe)
    exact = bool(sm.fam.verify_exact(sm, probe))
    del probe
    if exact:
        steps = _steps(sm, toks)
        chain = _verify(sm, toks, None, None)
        out += [f"{where}: chain row {r}: {_differs(chain[r], steps[r])}"
                for r in range(len(toks)) if not torch.equal(chain[r], steps[r])]  # fmt: skip
        P = len(PROMPT)
        wrong = (toks[1] + 7) % 256
        tree = _verify(sm, [toks[0], wrong, toks[1], toks[2]], [-1, 0, 0, 2], [P, P + 1, P + 1, P + 2])
        out += [f"{where}: tree node {r}: {_differs(tree[r], steps[w])}"
                for r, w in ((0, 0), (2, 1), (3, 2)) if not torch.equal(tree[r], steps[w])]  # fmt: skip
    greedy = [int(t) for t in sm.generate(list(PROMPT), SPEC_NEW, eos=(), speculate=False).tokens]
    spec = [int(t) for t in sm.generate(list(PROMPT), SPEC_NEW, eos=()).tokens]
    if spec != greedy:
        at = next(i for i, (a, b) in enumerate(zip(spec, greedy, strict=False)) if a != b)
        out.append(f"{where}: the speculative decode parts from the greedy one at token {at}")
    if not exact:
        out.append(f"{where}: no exact verify here - the engine speculates by one-row passes (verify_exact False)")
    return out


# (name, load options, after the load) for each placement a family is run on
PLACEMENTS: list[tuple[str, dict[str, Any], Callable[[Any], None] | None]] = [
    ("the card", {"device": "cuda"}, None),
    # the card graph and a family's card program (Qwen4's) both off: the torch layers on the card, as when they decline
    (
        "the card's torch path",
        {"device": "cuda"},
        lambda sm: sm.__dict__.update(card_graphs=False, card_programs=False),
    ),
    ("kv in host RAM", {"device": "cuda", "kv_host": True}, None),
    ("a layer on the host", {"device": "cuda", "cpu_layers": 1}, None),
    ("the CPU", {"device": "cpu"}, None),
]


@pytest.mark.parametrize("fam", ["qwen3", "phi3", "gemma3", "gpt_oss", "q35", "q4"])
def test_every_verify_row_is_the_greedy_steps_on_every_placement(fam: str, tmp_path: Any) -> None:
    need_cuda()
    path = _model(fam, str(tmp_path))
    found: list[str] = []
    try:
        for name, kw, after in PLACEMENTS:
            with loaded_model(path, context=512, **kw) as sm, torch.inference_mode():
                if after is not None:
                    after(sm)
                found += _partings(sm, f"{fam} on {name}")
    finally:
        shutil.rmtree(os.path.join(str(tmp_path), fam), ignore_errors=True)
    # a placement that cannot verify exactly speculates by one-row passes: speculation off there, a gap, not a pass
    assert not found, "\n".join(found)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
