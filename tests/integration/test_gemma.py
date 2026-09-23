# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Gemma 3 against transformers' own float32 forward. The engine runs the module's decoder layers (its block is a
sandwich norm with q/k norms, its embedding scaled by sqrt(hidden), and its rope split local/global by layer type),
so this pins the parts the engine still owns: the scaled embedding, the per-layer-type rope the pass hands each
layer, and the sliding/full mask split. A tiny random text model exercises both layer types and a real window."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from btb import mlx_available
from tests.helpers import (
    CHUNK,
    PARENTS,
    PATH,
    forward_logits,
    host_model,
    loaded_model,
    max_abs,
    speculation,
    tree_next,
    tree_pass,
)
from tests.make_fixtures import write_tokenizer

PROMPT = [[3, 17, 42, 5, 99, 120, 7, 7, 200, 12, 45, 8, 3, 17, 60, 61]]
CONT = [[31, 4, 77]]
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
REPO_270M = "google/gemma-3-270m"
PARA = (
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting each of the one hundred and "
    "twelve steps as he had for thirty years. From the lamp room he could see the fishing boats turning for "
    "harbour, their lanterns swinging, and the long grey line of the reef where the water broke white. "
)


def _build(model_dir: str, dtype: torch.dtype = torch.float32) -> None:
    """A tiny Gemma 3 text model: four layers alternating sliding/full (a window of 4, shorter than the prompt),
    distinct local/global rope, a non-head-dim query scale, and norm weights perturbed off zero so the 1 + weight
    centering matters. float32 for the transformers-parity check; bf16 for the MLX fused path, which binds its
    resident weights only when they are bf16."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    cfg = Gemma3TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=4,
        layer_types=LAYER_TYPES,
        query_pre_attn_scalar=32,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    torch.manual_seed(20260917)
    model = Gemma3ForCausalLM(cfg).float().eval()
    # every norm weight inits to zero (1 + 0 == 1, an identity that would hide a centering bug): push it off zero
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and name.endswith(".weight"):
                p.add_(torch.randn_like(p) * 0.2)
    assert cfg.layer_types == LAYER_TYPES
    model.to(dtype).save_pretrained(model_dir, safe_serialization=True)
    write_tokenizer(model_dir)


def _reference(model_dir: str) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """transformers' own float32 forward: the prompt's last logits, the continuation's over the kept cache, and
    ten greedy tokens - the answers the engine must reproduce."""
    from transformers import Gemma3ForCausalLM

    model = Gemma3ForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    with torch.inference_mode():
        out = model(input_ids=torch.tensor(PROMPT), use_cache=True)
        prompt_logits = out.logits[0, -1].clone()
        cache = out.past_key_values
        cont_logits = model(input_ids=torch.tensor(CONT), past_key_values=cache, use_cache=True).logits[0, -1].clone()
        g = model(input_ids=torch.tensor(PROMPT), use_cache=True)
        cache, tok, greedy = g.past_key_values, int(g.logits[0, -1].argmax()), []
        greedy.append(tok)
        while len(greedy) < 10:
            o = model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
            tok = int(o.logits[0, -1].argmax())
            greedy.append(tok)
    return prompt_logits, cont_logits, greedy


def test_gemma3_matches_transformers(tmp_path: Path) -> None:
    model_dir = str(tmp_path / "tiny_gemma3")
    _build(model_dir)
    ref_prompt, ref_cont, ref_greedy = _reference(model_dir)
    sm = host_model(model_dir)
    try:
        with torch.inference_mode():
            cache = sm.new_cache()
            prompt_logits = sm._prefill(torch.tensor(PROMPT), cache)[0, -1]
            cont_logits = forward_logits(sm, CONT, cache)[0, -1]
            greedy = sm.generate_greedy(PROMPT, 10)
    finally:
        sm.close()
    dp = max_abs(prompt_logits, ref_prompt)
    dc = max_abs(cont_logits, ref_cont)
    assert dp < 1e-4, f"prompt logits diverged: {dp:.2e}"
    assert dc < 1e-4, f"continuation logits diverged: {dc:.2e}"
    assert greedy == ref_greedy, f"greedy diverged: {greedy} vs {ref_greedy}"


def test_gemma3_tree_matches_transformers(tmp_path: Path) -> None:
    """A verify pass over the receipts' tree on the host path (`ai`), where the window of 4 falls inside the
    tree: every node's logits equal transformers' over the prompt and the node's own path, and the step after
    the committed path equals a plain continuation."""
    from transformers import Gemma3ForCausalLM

    model_dir = str(tmp_path / "tiny_gemma3")
    _build(model_dir)
    ref = Gemma3ForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()

    def path_to(j: int) -> list[int]:
        out = []
        while j >= 0:
            out.append(CHUNK[j])
            j = PARENTS[j]
        return out[::-1]

    sm = host_model(model_dir)
    try:
        with torch.inference_mode():
            cache = sm.new_cache()
            sm._prefill(torch.tensor(PROMPT), cache)
            lg, _base = tree_pass(sm, cache)
            nxt = tree_next(sm, cache)
            for j in range(len(CHUNK)):
                r = ref(input_ids=torch.tensor([PROMPT[0] + path_to(j)])).logits[0, -1]
                d = max_abs(lg[j], r)
                assert d < 1e-4, f"node {j} diverged: {d:.2e}"
            ids = PROMPT[0] + [CHUNK[q] for q in PATH] + [CHUNK[PATH[-1]]]
            r = ref(input_ids=torch.tensor([ids])).logits[0, -1]
            d = max_abs(nxt, r)
            assert d < 1e-4, f"the step after the path diverged: {d:.2e}"
    finally:
        sm.close()


@pytest.mark.skipif(not mlx_available(), reason="MLX not available")
def test_gemma3_fused_matches_eager(tmp_path: Path) -> None:
    """The fused MLX block against the eager per-layer path on the same bf16 weights (Milestone 2): both run on
    MLX, so this isolates the sandwich-norm ordering, the dual local/global rope, and the sliding-window attention
    the fused path adds from the numerics. The eager path is already pinned to transformers above."""
    model_dir = str(tmp_path / "tiny_gemma3")
    _build(model_dir, torch.bfloat16)  # the fused MLX path binds resident weights only when they are bf16
    with loaded_model(model_dir, device="mlx") as sm:
        speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
        assert sm.mlx is not None, "the tiny model did not land on the MLX device"
        # the fused path only engages when every layer is MLX-resident; otherwise this would compare eager to eager
        assert all(i in sm.mlx_layers and i in sm.host for i in range(sm.L)), "layers not all MLX-resident"
        with torch.inference_mode():
            cache = sm.new_cache()
            fused_prompt = sm._prefill(torch.tensor(PROMPT), cache)[0, -1].float()
            fused_cont = forward_logits(sm, CONT, cache)[0, -1].float()
            fused_greedy = sm.generate_greedy(PROMPT, 10)
            sm.mlx_fused = False  # type: ignore[attr-defined]  # the eager per-layer path, as the reference
            cache = sm.new_cache()
            eager_prompt = sm._prefill(torch.tensor(PROMPT), cache)[0, -1].float()
            eager_cont = forward_logits(sm, CONT, cache)[0, -1].float()
            eager_greedy = sm.generate_greedy(PROMPT, 10)
    dp = max_abs(fused_prompt, eager_prompt)
    dc = max_abs(fused_cont, eager_cont)
    # both paths norm in float32 over the same bf16 weights; what is left is the fused kernels' own arithmetic
    # (matmul order, Metal's rsqrt and fast cos/sin) on a random model with no margins: 9e-3 measured, where a
    # wrong norm order, rope or window lands past 1e0; the greedy gates are on 270m below
    assert dp < 3e-2, f"fused vs eager prompt logits diverged: {dp:.2e}"
    assert dc < 3e-2, f"fused vs eager continuation logits diverged: {dc:.2e}"
    assert len(fused_greedy) == len(eager_greedy) == 10  # both paths produced a full run


def _cached(repo: str) -> bool:
    # check config.json is in the local cache, not a full snapshot: btb.resolve fetches only the model files, so
    # a whole-snapshot check trips on the missing .gitattributes/README.md and skips a model that is really there
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


@pytest.mark.skipif(not mlx_available() or not _cached(REPO_270M), reason="needs MLX + cached 270m")
def test_gemma3_270m_fused_matches_eager() -> None:
    """The decisive fused-path gate: on real bf16 weights (head_dim 256, so the node kernels fire) the fused
    block's greedy output must equal the eager path's. Real logits have clear margins, so a matching greedy here
    is unambiguous where the tiny random model's cannot be."""
    import btb

    prompt = [[2, 818, 5279, 529, 7001, 563]]  # "The capital of France is", the earlier 270m parity prompt
    sm = btb.load(REPO_270M, device="mlx")

    def greedy() -> list[int]:
        return list(sm.generate_greedy(prompt, 12))

    try:
        speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
        with torch.inference_mode():
            fused = greedy()  # the full fused MLX path (kernels + node attention)
            sm.mlx_fused = False  # type: ignore[attr-defined]  # eager per-layer module path (the reference)
            eager = greedy()
    finally:
        sm.close()
    assert fused == eager, f"270m fused {fused} != eager {eager}"


def _long_prompt(repo: str, min_tokens: int) -> list[int]:
    """a natural prompt of at least `min_tokens`: one BOS, then a paragraph repeated"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    bos, *body = tok(PARA)["input_ids"]
    return [bos, *(body * -(-min_tokens // len(body)))]


@pytest.mark.skipif(not mlx_available() or not _cached(REPO_270M), reason="needs MLX + cached 270m")
def test_gemma3_270m_past_window() -> None:
    """The sliding layers past their window (512 on 270m), where the fused path's window first bites: the SDPA
    mask over the prefill, the node kernel's window on every decode row, and on the chains n-gram speculation
    verifies. Fused greedy must equal eager greedy, and the speculative stream must equal greedy."""
    with loaded_model(REPO_270M, device="mlx") as sm:
        win = int(sm.cfg.sliding_window)
        prompt = _long_prompt(REPO_270M, win + 128)
        assert len(prompt) > win
        with torch.inference_mode():
            spec, census = sm.generate_speculative([prompt], 16, proposer="ngram")
            fused = list(sm.generate_greedy([prompt], 16))
            sm.mlx_fused = False  # type: ignore[attr-defined]  # eager per-layer module path (the reference)
            eager = list(sm.generate_greedy([prompt], 16))
    assert fused == eager, f"270m past the window: fused {fused} != eager {eager}"
    assert spec == fused, f"270m past the window: speculative {spec} != greedy {fused} ({census})"
