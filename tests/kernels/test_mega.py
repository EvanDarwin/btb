# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel on Qwen3-0.6B from the local cache (two weight buffers, an untied embedding): a pass of any
width and shape gives the fused path's argmax ids and cache rows bit for bit, and the greedy and speculative
loops give the fused loops' tokens. Skipped without MLX or the cached model (never downloads)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
import torch

from btb.engine.cache import GrowLayer
from btb.kinds import Parents, Tokens
from tests.helpers import NO_LOG, need_cached, need_mlx

if TYPE_CHECKING:
    import mlx.core as mx

    from btb.engine.model import StreamedTextModel
    from btb.mlx.mega import MegaPass

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def engine() -> Iterator[StreamedTextModel]:
    import btb

    need_mlx()
    path = need_cached(MODEL, f"{MODEL} is not in the local cache")
    sm = btb.load(path, device="mlx", log=NO_LOG, mlx_mega=1)
    assert sm._mega is not None, "the megakernel did not build on the 0.6B"
    yield sm
    sm.close()


def _pass(
    sm: StreamedTextModel, mega: MegaPass | None, prompt: Tokens, toks: Tokens, parents: Parents
) -> tuple[list[int], list[mx.array]]:
    """one pass over a fresh cache: (argmax ids, layer 0's and the last layer's K rows of the pass)"""
    import mlx.core as mx

    sm._mega = mega
    cache = sm.new_cache(max_len=4096)
    past = len(prompt)
    with torch.inference_mode():
        sm._prefill(torch.tensor([prompt]), cache)
        sm.aa(parents)
        try:
            out = sm.forward([toks], cache=cache, last_only=False, pick=True)
            assert out is not None
            ids = [int(v) for v in out[0].tolist()]
        finally:
            sm.ab()
    T = len(toks)
    rows = [cache.layers[i]._mx[0][0, :, past : past + T].astype(mx.float32) for i in (0, len(cache.layers) - 1)]
    return ids, rows


@pytest.mark.parametrize("past", [51, 129, 300])
@pytest.mark.parametrize(
    "parents",
    [[-1], list(range(-1, 4)), [-1, 0, 1, 2, 3, 0, 5, 6, 1, 8, 0, 10, 11, 12, 13], list(range(-1, 15))],
    ids=["one", "chain5", "tree15", "chain16"],
)
def test_pass_matches_fused(engine: StreamedTextModel, past: int, parents: Parents) -> None:
    import mlx.core as mx

    sm = engine
    mega = sm._mega
    prompt = [151644, *range(1000, 1000 + past - 1)]
    toks = [2000 + 7 * t for t in range(len(parents))]
    try:
        f_ids, f_rows = _pass(sm, None, prompt, toks, parents)
        m_ids, m_rows = _pass(sm, mega, prompt, toks, parents)
    finally:
        sm._mega = mega
    assert m_ids == f_ids, f"the megakernel's ids {m_ids} are not the fused path's {f_ids}"
    for a, b in zip(m_rows, f_rows):
        assert mx.array_equal(a, b), "the megakernel's K rows differ from the fused path's"


def test_tree_positions_take_the_kernel(engine: StreamedTextModel) -> None:
    """the speculative loop hands every tree pass its nodes' positions (past + depth): the kernel takes those and
    refuses any other"""
    sm = engine
    parents = [-1, 0, 1, 0, 3]
    cache = sm.new_cache(max_len=4096)
    with torch.inference_mode():
        sm._prefill(torch.tensor([[151644, *range(1000, 1020)]]), cache)
    past = cache.get_seq_length()
    sm.aa(parents)
    try:
        assert sm._mega_ok(cache, 5, None, True, True, [[past, past + 1, past + 2, past + 1, past + 2]])
        assert not sm._mega_ok(cache, 5, None, True, True, [[past, past + 1, past + 2, past + 3, past + 4]])
        assert not sm._mega_ok(cache, 5, None, True, True, [[past + 1, past + 2, past + 3, past + 2, past + 3]])
    finally:
        sm.ab()
    assert sm._mega_ok(cache, 3, None, True, True, [[past, past + 1, past + 2]]), "a chain's positions"


def test_loops_match_fused(engine: StreamedTextModel) -> None:
    sm = engine
    mega = sm._mega
    ids = torch.tensor([[151644, *range(1000, 1040)]])
    out = {}
    try:
        with torch.inference_mode():
            for use in (False, True):
                sm._mega = mega if use else None
                out[("greedy", use)] = list(sm.generate_greedy(ids, 48))
                out[("spec", use)] = list(sm.generate_speculative(ids, 48)[0])
    finally:
        sm._mega = mega
    g = out[("greedy", False)]
    assert len(g) == 48
    for k, v in out.items():
        assert v == g, f"{k} gave {v[:12]}... against the fused greedy loop's {g[:12]}..."


def test_loops_match_sampled(engine: StreamedTextModel) -> None:
    """under a temperature the megakernel loops (the pick over the logits the kernel leaves in its scratch) give the
    fused loops' tokens under one seed, speculative and sequential alike; the sample is not the greedy answer"""
    from btb.sampling import Sampling

    sm = engine
    mega = sm._mega
    ids = torch.tensor([[151644, *range(1000, 1040)]])
    s3 = Sampling(temperature=0.7, top_p=0.9, seed=3)
    out = {}
    try:
        with torch.inference_mode():
            for use in (False, True):
                sm._mega = mega if use else None
                out[("plain", use)] = list(sm.generate_greedy(ids, 48, sampling=s3))
                out[("spec", use)] = list(sm.generate_speculative(ids, 48, sampling=s3)[0])
            sm._mega = mega
            greedy = list(sm.generate_greedy(ids, 48))
    finally:
        sm._mega = mega
    g = out[("plain", False)]
    assert len(g) == 48 and g != greedy
    for k, v in out.items():
        assert v == g, f"{k} gave {v[:12]}... against the fused sampled loop's {g[:12]}..."


def test_arena_layers(engine: StreamedTextModel) -> None:
    """every layer of a cache built with the megakernel lives in the arena"""
    cache = engine.new_cache(max_len=1024)
    assert all(isinstance(cl, GrowLayer) and cl.arena is not None for cl in cache.layers)
