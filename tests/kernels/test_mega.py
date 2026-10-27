# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel on Qwen3-0.6B from the local cache (two weight buffers, an untied embedding), its Q4_K_M file and a
Qwen3.5 2-bit file of lattice types: a pass of any width and shape gives the step's argmax ids and cache rows bit for
bit, and the greedy and speculative loops give the step's tokens. Skipped without MLX or the cached model (never
downloads). Then every family the megakernel composes, on its tiny fixture, against its own step path."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
import torch

from btb.engine.cache import GrowLayer
from btb.engine.generate import lin_layer
from btb.kinds import MEGA_STORED, Cap, LayerKind, Parents, Tokens
from btb.sampling import GREEDY
from tests.cert import core, spec
from tests.helpers import FIXTURES, NO_LOG, mlx_keys, need_cached, need_mlx

if TYPE_CHECKING:
    import mlx.core as mx

    from btb.engine.model import StreamedTextModel
    from btb.engine.prefix import PrefixCache
    from btb.mlx.mega import MegaPass

MODEL = "Qwen/Qwen3-0.6B"
# the release's Q4_K_M file: Q4_K blocks with Q6_K ones in the same layers (a v and a down of their own type)
MODEL_Q4KM = "unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q4_K_M.gguf"
# a hybrid's dynamic 2-bit file: IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS and IQ4_XS blocks beside k-quant and Q8_0 ones
MODEL_UDQ2 = "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-UD-Q2_K_XL.gguf"


@pytest.fixture(scope="module", params=[MODEL, MODEL_Q4KM, MODEL_UDQ2])
def engine(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    import btb

    need_mlx()
    spec_ = str(request.param)
    path = need_cached(spec_, f"{spec_} is not in the local cache")
    sm = btb.load(path, device="mlx", log=NO_LOG, mlx_mega=1)
    assert sm._mega is not None, f"the megakernel did not build on {spec_}"
    yield sm
    sm.close()


def _kv_layers(sm: StreamedTextModel) -> tuple[int, int]:
    """the first and the last layer that keep K rows (a hybrid's attention layers, not its DeltaNet ones)"""
    att = [i for i in range(sm.L) if sm.layer_types[i] != LayerKind.LINEAR]
    return att[0], att[-1]


def _pass(
    sm: StreamedTextModel,
    mega: MegaPass | None,
    prompt: Tokens,
    toks: Tokens,
    parents: Parents,
    arena: MegaPass | None = None,
) -> tuple[list[int], list[mx.array]]:
    """one pass by `mega` (None: the step) over a fresh cache, the arena of `arena` (else of `mega`; a plain cache
    without either): (argmax ids, the first and the last attention layer's K rows after it, the prompt's and the
    pass's)"""
    import mlx.core as mx

    sm._mega = arena or mega
    cache = sm.new_cache(max_len=4096)
    past = len(prompt)
    with torch.inference_mode():
        sm._prefill(torch.tensor([prompt]), cache)
        sm._mega = mega
        sm.aa(parents)
        try:
            out = sm.forward([toks], cache=cache, last_only=False, pick=True)
            assert out is not None
            # a path that picks hands back ids, one that does not (the sandwich's step under speculation) logits
            ids = [int(v) for v in (out[0].argmax(-1) if out.dim() == 3 else out[0]).tolist()]
        finally:
            sm.ab()
    att = [cl for i, cl in enumerate(cache.layers) if sm.layer_types[i] != LayerKind.LINEAR]
    if arena or mega:
        assert all(isinstance(cl, GrowLayer) and cl.arena is not None for cl in att), "the pass left the arena"
    T = len(toks)
    rows = [mlx_keys(cache, i)[0, :, : past + T].astype(mx.float32) for i in _kv_layers(sm)]
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


def _tree_kernels_match_step(
    sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch, prompt: Tokens, toks: Tokens, parents: Parents
) -> None:
    """for each tree kernel the megakernel has at the model's head size (`aluminum.tree_kernels` on its threadgroups,
    `aluminum.tree_pick` made to take it for every tree, in a pass built for it): each node of a pass is the step's
    one-row step along its path, its id and its cache rows bit for bit"""
    import mlx.core as mx

    from btb.mlx import aluminum
    from btb.mlx.mega import MegaPass
    from btb.mlx.mega.replay import THREADS

    mega = sm._mega
    assert mega is not None
    past = len(prompt)
    try:
        sm._mega = None
        steps = [_along(sm, prompt, toks, parents, j) for j in range(len(parents))]
        for pick in aluminum.tree_kernels(mega.hd, THREADS // 32):
            monkeypatch.setattr(
                aluminum, "tree_pick", lambda D, g, Hk, T, splits, nsg=0, p=pick: p if T > 1 else aluminum.NODE_BY_NODE
            )
            m_ids, m_rows = _pass(sm, MegaPass(sm), prompt, toks, parents)
            sm._mega = None
            for j, (f_id, f_rows) in enumerate(steps):
                on = f"node {j} on the {pick.kind} at {pick.rows} rows"
                assert m_ids[j] == f_id, f"{on}: the pass's id {m_ids[j]} is not the step's {f_id}"
                for a, b in zip(m_rows, f_rows):
                    row = a[:, past + j : past + j + 1]
                    assert mx.array_equal(row, b.astype(mx.float32)), f"{on}: K rows differ from the step's"
    finally:
        sm._mega = mega


def test_tree_kernels_match_step(engine: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """the megakernel's tree kernels over a prefix of three segments: each node the step's one-row step along its
    path, bit for bit (its segments' partials folded)"""
    past = 1100
    parents = [-1, 0, 1, 0, 3, 3, 1]
    prompt = [151644, *range(1000, 1000 + past - 1)]
    _tree_kernels_match_step(engine, monkeypatch, prompt, [2000 + 7 * t for t in range(len(parents))], parents)


@pytest.mark.parametrize("past", [51, 300])
@pytest.mark.parametrize("parents", [[-1], [-1, 0, 1, 2, 3, 0, 5, 6, 1, 8, 0, 10, 11, 12, 13]], ids=["one", "tree15"])
def test_step_over_the_arena_matches_a_plain_cache(engine: StreamedTextModel, past: int, parents: Parents) -> None:
    """a pass the step takes over the megakernel's arena (the prompt's rows and its own written with the pass, read as
    a paged layer's are) gives a plain cache's ids and rows"""
    prompt = [151644, *range(1000, 1000 + past - 1)]
    _arena_matches_a_plain_cache(engine, prompt, [2000 + 7 * t for t in range(len(parents))], parents)


def _arena_matches_a_plain_cache(sm: StreamedTextModel, prompt: Tokens, toks: Tokens, parents: Parents) -> None:
    """the step's pass over the megakernel's arena gives its pass over a plain cache's ids and rows"""
    import mlx.core as mx

    mega = sm._mega
    try:
        p_ids, p_rows = _pass(sm, None, prompt, toks, parents)
        a_ids, a_rows = _pass(sm, None, prompt, toks, parents, arena=mega)
    finally:
        sm._mega = mega
    assert a_ids == p_ids, f"the step's ids over the arena {a_ids} are not a plain cache's {p_ids}"
    for a, b in zip(a_rows, p_rows):
        assert mx.array_equal(a, b), "the step's K rows in the arena differ from a plain cache's"


def _paged_pass(
    sm: StreamedTextModel, pc: PrefixCache, mega: MegaPass | None, prompt: Tokens, toks: Tokens, parents: Parents
) -> tuple[list[int], list[mx.array]]:
    """`_pass` over a fresh paged cache of `pc`: the prompt prefilled by the step, the pass by `mega` (None: the
    step); the ids, and the first and the last attention layer's K rows read back through the conversation's row map"""
    import mlx.core as mx

    sm._mega = None
    cache = pc.new()
    past = len(prompt)
    with torch.inference_mode():
        sm._prefill(torch.tensor([prompt]), cache)
        sm._mega = mega
        sm.aa(parents)
        try:
            out = sm.forward([toks], cache=cache, last_only=False, pick=True)
            assert out is not None
            ids = [int(v) for v in out[0].tolist()]
        finally:
            sm.ab()
    rows = mx.array(cache.table.rows()[: past + len(toks)].numpy().astype("uint32"))
    host = pc.pool.host
    got = [mx.take(host.mx_view(i)[0][0], rows, axis=1).astype(mx.float32) for i in _kv_layers(sm)]
    cache.crop_to(0)
    return ids, got


@pytest.mark.parametrize("past", [51, 300])
@pytest.mark.parametrize(
    "parents",
    [[-1], list(range(-1, 4)), [-1, 0, 1, 2, 3, 0, 5, 6, 1, 8, 0, 10, 11, 12, 13]],
    ids=["one", "chain5", "tree15"],
)
def test_pass_over_pages_matches_the_step(engine: StreamedTextModel, past: int, parents: Parents) -> None:
    """a pass over a paged cache (the prefix cache's pool): the megakernel writes its rows where the table puts them
    and reads every row through the row map, and gives the step's ids and rows over the same kind of cache"""
    import mlx.core as mx

    from btb.engine.prefix import PrefixCache

    sm = engine
    mega = sm._mega
    assert mega is not None
    pc = PrefixCache(sm)
    prompt = [151644, *range(1000, 1000 + past - 1)]
    toks = [2000 + 7 * t for t in range(len(parents))]
    try:
        s_ids, s_rows = _paged_pass(sm, pc, None, prompt, toks, parents)
        assert sm._mega_paged_ok(pc.new()), "the megakernel refuses the prefix cache's pages"
        m_ids, m_rows = _paged_pass(sm, pc, mega, prompt, toks, parents)
    finally:
        sm._mega = mega
        pc.close()
    assert m_ids == s_ids, f"the megakernel's ids over pages {m_ids} are not the step's {s_ids}"
    for a, b in zip(m_rows, s_rows):
        assert mx.array_equal(a, b), "the megakernel's K rows in the pages differ from the step's"


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
        assert sm._mega_ok(cache, 5, None, True, GREEDY, [[past, past + 1, past + 2, past + 1, past + 2]])
        assert not sm._mega_ok(cache, 5, None, True, GREEDY, [[past, past + 1, past + 2, past + 3, past + 4]])
        assert not sm._mega_ok(cache, 5, None, True, GREEDY, [[past + 1, past + 2, past + 3, past + 2, past + 3]])
    finally:
        sm.ab()
    assert sm._mega_ok(cache, 3, None, True, GREEDY, [[past, past + 1, past + 2]]), "a chain's positions"


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
    """every attention layer of a cache built with the megakernel lives in the arena"""
    cache = engine.new_cache(max_len=1024)
    att = [cl for i, cl in enumerate(cache.layers) if engine.layer_types[i] != LayerKind.LINEAR]
    assert att and all(isinstance(cl, GrowLayer) and cl.arena is not None for cl in att)


# the fixtures of every family whose layers the megakernel composes, by the cert's own gate, then each as a GGUF of
# every stored type the kernel multiplies as stored
STEMS = [spec.FIXTURE_STEM[k] for k in spec.FIXTURE_STEM if spec.mega_refusal(k, spec.Storage.SAFE_BF16) is None]
GGUF_STEMS = [os.path.join("gguf", f"{s}-{q.value.lower()}.gguf") for s in STEMS for q in sorted(MEGA_STORED)]


@pytest.fixture(scope="module", params=STEMS + GGUF_STEMS)
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
    return int(out[0, 0]), [mlx_keys(cache, i)[0, :, row : row + 1] for i in _kv_layers(sm)]


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
                row = a[:, past + j : past + j + 1]
                assert mx.array_equal(row, b.astype(mx.float32)), f"node {j}: K rows differ from the step's"
    finally:
        sm._mega = mega


@pytest.mark.parametrize("parents", [[-1], [-1, 0, 1, 0, 3, 3, 1]], ids=["one", "tree7"])
def test_family_step_over_the_arena_matches_a_plain_cache(tiny: StreamedTextModel, parents: Parents) -> None:
    """every family's step over the megakernel's arena, a tree's metadata read off its first attention layer (a
    hybrid's first layer keeps no rows), gives its pass over a plain cache's ids and rows"""
    V = _vocab(tiny)
    prompt = [1 + (7 * t) % (V - 1) for t in range(70)]
    _arena_matches_a_plain_cache(tiny, prompt, [1 + (13 * t + 5) % (V - 1) for t in range(len(parents))], parents)


@pytest.mark.parametrize("past", [70, 600])
def test_family_tree_kernels_match_step(tiny: StreamedTextModel, past: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """every family's megakernel tree kernels over a prefix of one segment and of two: each node its step path's
    one-row step along the node's path, bit for bit"""
    parents = [-1, 0, 1, 0, 3, 3, 1]
    V = _vocab(tiny)
    prompt = [1 + (7 * t) % (V - 1) for t in range(past)]
    toks = [1 + (13 * t + 5) % (V - 1) for t in range(len(parents))]
    _tree_kernels_match_step(tiny, monkeypatch, prompt, toks, parents)


def test_split_projections_match_step(tiny: StreamedTextModel) -> None:
    """with q/k/v, gate/up and a DeltaNet's four inputs not one weight (a GGUF's v of another type than q and k),
    each written as its own piece of one output row: every node of a tree pass is still the step's one-row step
    along its path"""
    import mlx.core as mx

    from btb.mlx.mega import MegaPass

    sm = tiny
    mega = sm._mega
    held: list[tuple[object, str, object]] = []
    for i in range(sm.L):
        for part, name in (("self_attn", "_mx_qkv"), ("mlp", "_mx_gu"), ("linear_attn", "_mx_in")):
            mod = getattr(sm.host[i], part, None)
            if mod is not None and getattr(mod, name, None) is not None:
                held.append((mod, name, getattr(mod, name)))
                setattr(mod, name, None)
    parents = [-1, 0, 1, 0, 3, 3, 1]
    V = _vocab(sm)
    prompt = [1 + (7 * t) % (V - 1) for t in range(70)]
    toks = [1 + (13 * t + 5) % (V - 1) for t in range(len(parents))]
    try:
        split = MegaPass(sm)
        m_ids, m_rows = _pass(sm, split, prompt, toks, parents)
        sm._mega = None
        for j in range(len(parents)):
            f_id, f_rows = _along(sm, prompt, toks, parents, j)
            assert m_ids[j] == f_id, f"node {j}: the split pass's id {m_ids[j]} is not the step's {f_id}"
            for a, b in zip(m_rows, f_rows):
                row = a[:, len(prompt) + j : len(prompt) + j + 1]
                assert mx.array_equal(row, b.astype(mx.float32)), f"node {j}: K rows differ from the step's"
    finally:
        for mod, name, w in held:
            setattr(mod, name, w)
        sm._mega = mega


# the fixtures of every hybrid family the megakernel composes: their DeltaNet layers' states
HYBRID_STEMS = [s for s in STEMS if Cap.HYBRID in core.flags(next(k for k, v in spec.FIXTURE_STEM.items() if v == s))]


@pytest.fixture(scope="module", params=HYBRID_STEMS)
def hybrid(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    import btb

    need_mlx()
    sm = btb.load(os.path.join(FIXTURES, str(request.param)), device="mlx", log=NO_LOG, mlx_mega=1)
    assert sm._mega is not None, f"the megakernel did not build on {request.param}"
    yield sm
    sm.close()


def test_delta_states_match_step(hybrid: StreamedTextModel) -> None:
    """the DeltaNet states a pass writes back - after a row, another, and a chain of four - are the step's bit for
    bit, each layer's conv window and recurrent state, and the passes ran on the megakernel"""
    from btb.kinds import PassTag

    sm = hybrid
    mega = sm._mega
    V = _vocab(sm)
    prompt = [1 + (7 * t) % (V - 1) for t in range(40)]
    passes = [[5], [9], [11, 13, 17, 19]]
    linear = [i for i in range(sm.L) if sm.layer_types[i] == LayerKind.LINEAR]
    states: dict[bool, list[tuple[torch.Tensor, ...]]] = {}
    try:
        for use in (False, True):
            sm._mega = mega if use else None
            cache = sm.new_cache(max_len=4096)
            with torch.inference_mode():
                sm._prefill(torch.tensor([prompt]), cache)
                sm._pass_reset()
                for toks in passes:
                    sm.forward([toks], cache=cache, last_only=False, pick=True)
            assert (PassTag.MLX_MEGA in sm.last_pass_report()) is use, "the passes did not take the path asked"
            states[use] = [tuple(t.clone() for t in sm._lin(lin_layer(cache.layers[i]))) for i in linear]
    finally:
        sm._mega = mega
    for i, step, ours in zip(linear, states[False], states[True], strict=True):
        assert torch.equal(step[0], ours[0]), f"layer {i}'s conv window differs from the step's"
        assert torch.equal(step[1], ours[1]), f"layer {i}'s recurrent state differs from the step's"


def test_every_shape_places_the_logits_alike(tiny: StreamedTextModel) -> None:
    """a sampled pass reads its logits at one offset (`MegaPass.logits`), so every pass shape lays its scratch out as
    the widest does: a one-segment program once left out the fold's output, and a shape replayed after a program of
    the other kind had been built read another region"""
    mg = tiny._mega
    assert mg is not None
    want = mg.logits_off
    for T in (1, 7, 16):
        for splits in (1, 2, mg.SPL):
            for argmax in (True, False):
                mg.table(T, splits, argmax)
                assert mg.logits_off == want, f"{T} rows over {splits} segments place the logits elsewhere"


def test_a_pass_let_go_is_freed_at_once(tiny: StreamedTextModel) -> None:
    """a pass dropped (a shed) is freed with its last reference, its replay's context and buffers with it, not at the
    cycle collector's next round: the replay holds its pass weakly"""
    import gc
    import weakref

    from btb.mlx.mega import MegaPass

    gc.disable()
    try:
        mp = MegaPass(tiny)
        alive = weakref.ref(mp), weakref.ref(mp.replay)
        del mp
        assert alive[0]() is None and alive[1]() is None, "the pass outlived its last reference"
    finally:
        gc.enable()


def test_family_loops_match_step(tiny: StreamedTextModel) -> None:
    """the greedy and speculative loops on the megakernel give the step's tokens, and the speculative one's passes
    take it: an MTP drafter's verify passes hook the last layer's output, which the megakernel hands over"""
    from btb.kinds import PassTag

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
                sm._pass_reset()
                out[("spec", use)] = list(sm.generate_speculative(ids, 24)[0])
                assert (PassTag.MLX_MEGA in sm.last_pass_report()) is use, "the speculative loop's passes"
    finally:
        sm._mega = mega
    g = out[("greedy", False)]
    for k, v in out.items():
        assert v == g, f"{k} gave {v[:12]}... against the step's greedy loop's {g[:12]}..."


def test_mlx_float_ops_emulated_bit_for_bit() -> None:
    """the megakernel's copies of the float32 MLX ops a DeltaNet's front and norm take (kernel.py `MLX_OPS`): sigmoid
    and logaddexp against zero (the decay's softplus) over values across the range and its edges, and rms_norm over a
    row of up to 128 by one simdgroup (`mlx_rms_inv_simd`), against MLX's own"""
    need_mlx()
    import mlx.core as mx
    import numpy as np

    from btb.mlx.core import metal_kernel
    from btb.mlx.mega.kernel import MLX_OPS, TYPES

    elem = metal_kernel(
        name="test_mega_mlx_float",
        input_names=["x"],
        output_names=["sig", "lae"],
        header=TYPES + MLX_OPS,
        source="    uint i = thread_position_in_grid.x;\n    sig[i] = mlx_sigmoid(x[i]);\n"
        + "    lae[i] = mlx_logaddexp(x[i], 0.0f);\n",
    )
    rng = np.random.default_rng(0)
    edges = [0.0, -0.0, 1e-30, -1e-30, 1e-7, -1e-7, 88.7, -88.7, 104.0, -104.0, 1e30, -1e30, np.inf, -np.inf]
    vals = np.concatenate([*(rng.standard_normal(1 << 18) * s for s in (0.01, 1, 10, 100)), np.array(edges)])
    x = mx.array(vals.astype(np.float32))
    sig, lae = elem(
        inputs=[x],
        grid=(int(x.size), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[mx.float32, mx.float32],
    )
    assert mx.array_equal(sig.view(mx.uint32), mx.sigmoid(x).view(mx.uint32)), "sigmoid"
    assert mx.array_equal(lae.view(mx.uint32), mx.logaddexp(x, 0.0).view(mx.uint32)), "logaddexp"
    rms = metal_kernel(
        name="test_mega_mlx_rms_simd",
        input_names=["x", "w", "eps"],
        output_names=["y"],
        header=TYPES + MLX_OPS,
        source="    uint r = threadgroup_position_in_grid.x, lane = thread_position_in_threadgroup.x, H = x_shape[1];\n"
        + "    float inv = mlx_rms_inv_simd(x + (size_t)r * H, H, eps[0], lane);\n"
        + "    for (uint i = lane; i < H; i += 32) y[(size_t)r * H + i] = w[i] * (x[(size_t)r * H + i] * inv);\n",
    )
    for H in (32, 64, 128):
        x = mx.array((rng.standard_normal((256, H)) * 3).astype(np.float32))
        w = mx.array((1 + 0.1 * rng.standard_normal(H)).astype(np.float32))
        got = rms(
            inputs=[x, w, mx.array([1e-6], dtype=mx.float32)],
            grid=(256 * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(256, H)],
            output_dtypes=[mx.float32],
        )[0]
        want = mx.fast.rms_norm(x, w, 1e-6)
        assert mx.array_equal(got.view(mx.uint32), want.view(mx.uint32)), f"rms_norm float32 H {H}"


def test_mlx_ops_emulated_bit_for_bit() -> None:
    """the megakernel's copies of MLX's own ops (kernel.py `MLX_OPS`), run as the megakernel runs them (256 threads a
    row), give MLX's bits: rms_norm with a bf16 weight and with a float32 one (the step's `_mlx_norm` on a centered
    norm), at widths each side of MLX's looped limit; and over every finite bf16, the gated activations (`act`) times
    up and the embedding's scale"""
    need_mlx()
    import mlx.core as mx
    import numpy as np

    from btb.mlx.backend import act
    from btb.mlx.core import metal_kernel
    from btb.mlx.mega.kernel import MLX_OPS, TYPES

    ctx = "Ctx c = {tid, simdgroup_index_in_threadgroup, thread_index_in_simdgroup, tgm, nullptr, nullptr, nullptr, "
    ctx += "nullptr, nullptr, nullptr, nullptr};"
    head = "    threadgroup uint8_t tgm[256];\n    uint tid = thread_position_in_threadgroup.x;\n    " + ctx + "\n"
    norm = metal_kernel(
        name="test_mega_mlx_rms",
        input_names=["x", "w", "eps"],
        output_names=["y"],
        header=TYPES + MLX_OPS,
        source=head
        + "    uint r = threadgroup_position_in_grid.x, H = x_shape[1];\n"
        + "    mlx_rms_norm_row((const device bfloat*)x + (size_t)r * H, (device bfloat*)y + (size_t)r * H, w, H, "
        + "eps[0], c);\n",
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
    elem = metal_kernel(
        name="test_mega_mlx_elementwise",
        input_names=["g", "u", "s"],
        output_names=["silu", "gelu", "scaled"],
        header=TYPES + MLX_OPS,
        source="    uint i = thread_position_in_grid.x;\n"
        + "    bfloat a = ((const device bfloat*)g)[i], b = ((const device bfloat*)u)[i];\n"
        + "    ((device bfloat*)silu)[i] = (a * mlx_sigmoid(a)) * b;\n"
        + "    ((device bfloat*)gelu)[i] = mlx_gelu_tanh(a) * b;\n"
        + "    ((device bfloat*)scaled)[i] = a * (bfloat)s[0];\n",
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
