# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel on Qwen3-0.6B from the local cache (two weight buffers, an untied embedding): a pass of any
width and shape gives the fused path's argmax ids and cache rows bit for bit, and the greedy and speculative
loops give the fused loops' tokens. Skipped without MLX or the cached model (never downloads). Then every family the
megakernel composes, on its tiny fixture, against its own step path."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import pytest
import torch

from btb.engine.cache import GrowLayer
from btb.kinds import Parents, Tokens
from tests.cert import spec
from tests.helpers import FIXTURES, NO_LOG, need_cached, need_mlx

if TYPE_CHECKING:
    import mlx.core as mx

    from btb.engine.model import StreamedTextModel
    from btb.mlx.core import MetalKernel
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


# the fixtures of every family whose layers the megakernel composes, by the cert's own gate
STEMS = [spec.FIXTURE_STEM[k] for k in spec.FIXTURE_STEM if spec.mega_refusal(k, spec.Storage.SAFE_BF16) is None]


@pytest.fixture(scope="module", params=STEMS)
def tiny(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    import btb

    need_mlx()
    stem = str(request.param)
    sm = btb.load(os.path.join(FIXTURES, stem), device="mlx", log=NO_LOG, mlx_mega=1)
    assert sm._mega is not None, f"the megakernel did not build on {stem}"
    yield sm
    sm.close()


def _vocab(sm: StreamedTextModel) -> int:
    return int(sm.cfg.vocab_size)


def _along(sm: StreamedTextModel, prompt: Tokens, toks: Tokens, parents: Parents, j: int) -> tuple[int, list[mx.array]]:
    """node j's id and K rows (layer 0's, the last layer's) as the step makes them, one row at a time along j's path"""
    path, cur = [], j
    while cur >= 0:
        path.append(cur)
        cur = parents[cur]
    cache = sm.new_cache(max_len=4096)
    with torch.inference_mode():
        sm._prefill(torch.tensor([prompt]), cache)
        for n in reversed(path):
            out = sm.forward([[toks[n]]], cache=cache, last_only=False, pick=True)
    assert out is not None
    row = len(prompt) + len(path) - 1
    return int(out[0, 0]), [cache.layers[i]._mx[0][0, :, row : row + 1] for i in (0, len(cache.layers) - 1)]


@pytest.mark.parametrize("past", [9, 70])
@pytest.mark.parametrize("parents", [[-1], list(range(-1, 4)), [-1, 0, 1, 0, 3, 3, 1]], ids=["one", "chain4", "tree7"])
def test_family_pass_matches_step(tiny: StreamedTextModel, past: int, parents: Parents) -> None:
    """each node of a pass on a family's fixture is its step path's one-row step along the node's path, its id and its
    cache rows bit for bit: what makes a speculative pass exact, whether or not the family's step verifies trees on
    MLX itself (the sandwich's does not)"""
    import mlx.core as mx

    sm = tiny
    mega = sm._mega
    V = _vocab(sm)
    prompt = [1 + (7 * t) % (V - 1) for t in range(past)]
    toks = [1 + (13 * t + 5) % (V - 1) for t in range(len(parents))]
    try:
        m_ids, m_rows = _pass(sm, mega, prompt, toks, parents)
        sm._mega = None
        for j in range(len(parents)):
            f_id, f_rows = _along(sm, prompt, toks, parents, j)
            assert m_ids[j] == f_id, f"node {j}: the megakernel's id {m_ids[j]} is not the step's {f_id}"
            for a, b in zip(m_rows, f_rows):
                assert mx.array_equal(a[:, j : j + 1], b.astype(mx.float32)), f"node {j}: K rows differ from the step's"
    finally:
        sm._mega = mega


def test_family_loops_match_step(tiny: StreamedTextModel) -> None:
    sm = tiny
    mega = sm._mega
    V = _vocab(sm)
    ids = torch.tensor([[1 + (11 * t) % (V - 1) for t in range(24)]])
    out = {}
    try:
        with torch.inference_mode():
            for use in (False, True):
                sm._mega = mega if use else None
                out[("greedy", use)] = list(sm.generate_greedy(ids, 24))
                out[("spec", use)] = list(sm.generate_speculative(ids, 24)[0])
    finally:
        sm._mega = mega
    g = out[("greedy", False)]
    for k, v in out.items():
        assert v == g, f"{k} gave {v[:12]}... against the step's greedy loop's {g[:12]}..."


def test_mlx_ops_emulated_bit_for_bit() -> None:
    """the megakernel's copies of MLX's own ops (kernel.py `MLX_OPS`), run as the megakernel runs them (256 threads a
    row), give MLX's bits: rms_norm with a bf16 weight and with a float32 one (the step's `_mlx_norm` on a centered
    norm), at widths each side of MLX's looped limit; and over every finite bf16, the gated activations (`act`) times
    up and the embedding's scale"""
    import mlx.core as mx
    import numpy as np

    from btb.mlx.backend import act
    from btb.mlx.mega.kernel import MLX_OPS, TYPES

    need_mlx()
    ctx = "Ctx c = {tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tgm, nullptr, nullptr, nullptr, "
    ctx += "nullptr, nullptr, nullptr, nullptr};"
    head = "    threadgroup uint8_t tgm[256];\n    uint tid = thread_position_in_threadgroup.x;\n    " + ctx + "\n"
    norm = cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="test_mega_mlx_rms",
            input_names=["x", "w", "eps"],
            output_names=["y"],
            header=TYPES + MLX_OPS,
            source=head
            + "    uint r = threadgroup_position_in_grid.x, H = x_shape[1];\n"
            + "    mlx_rms_norm_row((const device bfloat*)x + (size_t)r * H, (device bfloat*)y + (size_t)r * H, w, H, "
            + "eps[0], c);\n",
        ),
    )
    rng = np.random.default_rng(0)
    for wdt in (mx.bfloat16, mx.float32):
        for H in (64, 1536, 2560, 4096, 5120):
            x = mx.array((rng.standard_normal((64, H)) * 3).astype(np.float32)).astype(mx.bfloat16)
            w = mx.array((1 + 0.1 * rng.standard_normal(H)).astype(np.float32)).astype(wdt)
            got = norm(
                inputs=[x, w, mx.array([1e-6], dtype=mx.float32)],
                grid=(64 * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(64, H)],
                output_dtypes=[mx.bfloat16],
            )[0]
            xs = x if wdt == mx.bfloat16 else x.astype(mx.float32)
            want = mx.fast.rms_norm(xs, w, 1e-6).astype(mx.bfloat16)
            assert mx.array_equal(got.view(mx.uint16), want.view(mx.uint16)), f"rms_norm {wdt} H {H}"
    # the gated MLP's activations times up, and the embedding scale, against the step's own MLX expressions
    elem = cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="test_mega_mlx_elementwise",
            input_names=["g", "u", "s"],
            output_names=["silu", "gelu", "scaled"],
            header=TYPES + MLX_OPS,
            source="    uint i = thread_position_in_grid.x;\n"
            + "    bfloat a = ((const device bfloat*)g)[i], b = ((const device bfloat*)u)[i];\n"
            + "    ((device bfloat*)silu)[i] = (a * mlx_sigmoid(a)) * b;\n"
            + "    ((device bfloat*)gelu)[i] = mlx_gelu_tanh(a) * b;\n"
            + "    ((device bfloat*)scaled)[i] = a * (bfloat)s[0];\n",
        ),
    )
    # every finite bf16 gate (the all-ones exponents, infinities and NaNs, read as zero)
    bits = np.arange(1 << 16, dtype=np.uint16)
    bits[(bits & 0x7F80) == 0x7F80] = 0
    g = mx.array(bits).view(mx.bfloat16)
    u = mx.array(rng.standard_normal(1 << 16).astype(np.float32)).astype(mx.bfloat16)
    scale = 2560**0.5  # Gemma 3 4B's embedding scale, sqrt(hidden)
    silu, gelu, scaled = elem(
        inputs=[g, u, mx.array([scale], dtype=mx.float32)],
        grid=(1 << 16, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1 << 16,)] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    )
    silu_f, gelu_f = act("silu"), act("gelu_pytorch_tanh")
    assert silu_f is not None and gelu_f is not None
    for name, got_, want_ in (
        ("silu", silu, silu_f(g) * u),
        ("gelu_pytorch_tanh", gelu, gelu_f(g) * u),
        ("embed_scale", scaled, g * scale),
    ):
        assert mx.array_equal(got_.view(mx.uint16), want_.view(mx.uint16)), name
