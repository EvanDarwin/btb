# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A mixture's pass on the GPU (btb/mlx/moe.py) and Qwen3.5-MoE's hybrid step over it: the route's picks and weights
as it defines them, each row its own whatever rows travel with it; a gather's rows each its expert's own matmul bit
for bit, every stored type and the float ones, over slots in two blocks; the combine as separate bf16 operations; the
float32 router's matvec row-invariant; and the hybrid step on the mixture fixture and its GGUF twins, a verify pass's
nodes their one-row steps bit for bit, a pass over the store's resident gather the bits of one whose picks the host
seated."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from btb.kinds import Cap, LayerKind, Parents, PassTag, Proposer, Quant, RouteMode, Tokens, stored_quants
from tests.cert import core, spec
from tests.helpers import FIXTURES, NO_LOG, mlx_keys, need_mlx, stored_blocks

if TYPE_CHECKING:
    import mlx.core as mx

    from btb.engine.model import StreamedTextModel
    from btb.mlx.backend import Weight
    from btb.mlx.moe import Gather


def _bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to bf16 (to nearest, ties to even), as float32"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


@pytest.mark.parametrize("mode", list(RouteMode), ids=lambda m: m.value)
@pytest.mark.parametrize(("E", "k"), [(4, 2), (128, 8), (256, 8)])
def test_the_route_picks_and_weighs_as_it_defines(mode: RouteMode, E: int, k: int) -> None:
    """each row's k experts are its k largest logits (equal ones to the lower index), listed ascending; their weights
    the definition's to bf16 (a softmax over every expert, renormalized over the picks or not, or over the picks'
    logits alone); and a row routed alone is the row routed among sixteen, bit for bit"""
    need_mlx()
    import mlx.core as mx

    from btb.mlx import moe

    T = 16
    lg = np.random.default_rng(E).standard_normal((T, E)).astype(np.float32)
    lg[3] = lg[3, 0]  # every logit equal: the k lowest experts
    lg[5, [E // 2, E - 1]] = lg[5].max() + 1.0  # a tie at the top
    ids, w = moe.route(mx.array(lg), k, mode, mx.bfloat16)
    ids_np, w_np = np.array(ids), np.array(w.astype(mx.float32))
    for t in range(T):
        picks = np.array(sorted(sorted(range(E), key=lambda e: (-lg[t, e], e))[:k]))
        assert list(ids_np[t]) == list(picks), (t, ids_np[t], picks)
        x = lg[t].astype(np.float64)
        if mode is RouteMode.SOFTMAX_PICKS:
            p = np.exp(x[picks] - x[picks].max())
            p /= p.sum()
        else:
            p = (np.exp(x - x.max()) / np.exp(x - x.max()).sum())[picks]
            if mode is RouteMode.SOFTMAX_ALL_NORM:
                p /= p.sum()
        assert np.allclose(w_np[t], p, rtol=2**-8, atol=0), (t, w_np[t], p)
        one_ids, one_w = moe.route(mx.array(lg[t : t + 1]), k, mode, mx.bfloat16)
        assert np.array_equal(np.array(one_ids)[0], ids_np[t]) and mx.array_equal(one_w[0], w[t]), t


def _slots(q: Quant | None, shape: tuple[int, int], n: int) -> tuple[Gather, list[Weight]]:
    """`n` random experts' part of type `q` (None: bf16) of `shape`, in two blocks as a store's slots hold them (a slot
    an expert from 16 bytes past the block's start, its streams laid out as `kinds.pooled_layout` lays one weight's),
    and each as the weight the per-expert path multiplies (`Backend.weight_stored_pool`, or a bf16 slot)"""
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.kinds import pooled_layout
    from btb.mlx.backend import place_streams
    from btb.mlx.moe import STREAMS_MAX, TAB_W, Gather

    be = mlxdev.Backend.__new__(mlxdev.Backend)
    rows, cols = shape
    rng = np.random.default_rng(7)
    mats = [(rng.standard_normal((rows, cols)) * 0.05).astype(np.float32) for _ in range(n)]
    if q is Quant.F16:
        raws = [m.astype(np.float16).view(np.uint8).reshape(-1) for m in mats]
    elif q in (None, Quant.BF16):
        raws = [(_bf16(m).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1) for m in mats]
    else:
        raws = [stored_blocks(q, rows, cols, int(rng.integers(1 << 30))) for _ in range(n)]
    halves = [list(range(n // 2)), list(range(n // 2, n))]
    tab = np.zeros((n, TAB_W), dtype=np.uint32)
    blocks, ws = [], {}
    # a slot an expert, its streams where the part's layout puts them (`_ExpertStore._part_streams`)
    (single,), part = pooled_layout([(q or Quant.BF16, int(raws[0].size))])
    stride = -(-part // 16) * 16
    for b, members in enumerate(halves):
        sh = mlxdev.Shared(16 + stride * len(members))
        for j, e in enumerate(members):
            offs = [16 + stride * j + o for o in single]
            nbs = place_streams(q or Quant.BF16, raws[e], sh.np, offs)
            tab[e, 0] = b
            tab[e, 1 : 1 + len(offs)] = offs
            assert len(offs) <= STREAMS_MAX
            if q is None:
                ws[e] = be.weight_slot(sh, offs[0], nbs[0], shape)
            else:
                ws[e] = be.weight_stored_pool(sh, offs, nbs, q, shape)
        blocks.append(sh.whole)
    return Gather(tuple(blocks), mx.array(tab)), [ws[e] for e in range(n)]


@pytest.mark.parametrize(
    "q", [None, *stored_quants(), Quant.BF16, Quant.F16], ids=lambda q: getattr(q, "value", "bf16")
)
@pytest.mark.parametrize("kp", [1, 4])
@pytest.mark.parametrize("f32", [False, True], ids=["bf16", "f32"])
def test_a_gathers_rows_are_each_experts_own_matmul(q: Quant | None, kp: int, f32: bool) -> None:
    """every row of a gather - an item's expert's part over its row of x, the slots in two blocks - is that expert's
    own matmul over the row (what the per-expert path computes) bit for bit: a step's and a verify pass's rows, each
    pick its type's one-row chain, over bf16 rows and an fp32 run's"""
    need_mlx()
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.mlx import moe

    be = mlxdev.Backend.__new__(mlxdev.Backend)
    be.gemm_rows = 64
    n, rows, cols = 6, 48, 512
    g, ws = _slots(q, (rows, cols), n)
    T = 16 // kp  # sixteen items: a row's kp picks of one x row each, or each its own row
    rng = np.random.default_rng(11)
    ids = np.stack([rng.choice(n, size=kp, replace=False) for _ in range(T)]).astype(np.uint32)
    x = mx.array(rng.standard_normal((T, cols)).astype(np.float32)).astype(mx.float32 if f32 else mx.bfloat16)
    got = moe.gather(q, g, mx.array(ids.reshape(-1)), x, rows, cols, 0, kp)
    assert got.dtype == x.dtype
    for i, e in enumerate(ids.reshape(-1)):
        want = be.matmul(x[i // kp : i // kp + 1], ws[int(e)])
        assert mx.array_equal(got[i : i + 1], want), (getattr(q, "value", "bf16"), kp, f32, i, int(e))


def test_the_combine_is_its_separate_bf16_operations() -> None:
    """a row's output is its picks' rows each times its weight, summed from zero in their order, each operation
    rounded to bf16; with a shared expert, plus sigmoid(gate) times its row, the sigmoid to float32's rounding"""
    need_mlx()
    import mlx.core as mx

    from btb.mlx import moe

    T, K, H = 5, 4, 96
    rng = np.random.default_rng(3)
    y = _bf16(rng.standard_normal((T, K, H)).astype(np.float32))
    w = _bf16(rng.random((T, K)).astype(np.float32))
    sh = _bf16(rng.standard_normal((T, H)).astype(np.float32))
    g = _bf16(rng.standard_normal(T).astype(np.float32))
    acc = np.zeros((T, H), np.float32)
    for k in range(K):
        acc = _bf16(acc + _bf16(y[:, k] * w[:, k : k + 1]))
    b = lambda a: mx.array(a).astype(mx.bfloat16)
    got = np.array(moe.combine(b(y), b(w)).astype(mx.float32))
    assert np.array_equal(got, acc)
    s = _bf16(1.0 / (1.0 + np.exp(-g.astype(np.float64))).astype(np.float32))
    want = _bf16(acc + _bf16(s[:, None] * sh))
    got = np.array(moe.combine(b(y), b(w), b(g), b(sh)).astype(mx.float32))
    assert np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-3)) <= 2**-7, "beyond a bf16 rounding of a sigmoid"


def test_a_float32_weights_matvec_is_row_invariant() -> None:
    """the step's float32 matvec over a float32 weight (a router kept as stored): to float32's rounding of the
    product, and each row the same bits alone as among sixteen"""
    need_mlx()
    import mlx.core as mx

    from btb.mlx.gemv import gemv

    rng = np.random.default_rng(5)
    W = rng.standard_normal((40, 512)).astype(np.float32)
    x = rng.standard_normal((16, 512)).astype(np.float32)
    y = gemv(mx.array(W), mx.array(x))
    want = x.astype(np.float64) @ W.T.astype(np.float64)
    assert np.max(np.abs(np.array(y) - want) / (np.abs(x) @ np.abs(W.T))) < 1e-6
    for t in (0, 7, 15):
        assert mx.array_equal(gemv(mx.array(W), mx.array(x[t : t + 1]))[0], y[t]), t
    yb = gemv(mx.array(W), mx.array(x).astype(mx.bfloat16))
    assert yb.dtype == mx.float32 and mx.array_equal(
        yb, gemv(mx.array(W), mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    )


# the mixture families the hybrid step takes, and each as a GGUF of every stored type its experts gather
MOE_STEMS = [
    spec.FIXTURE_STEM[k] for k in spec.FIXTURE_STEM if Cap.MOE in core.flags(k) and Cap.HYBRID in core.flags(k)
]
MOE_GGUF = [os.path.join("gguf", f"{s}-{q.value.lower()}.gguf") for s in MOE_STEMS for q in stored_quants()]


@pytest.fixture(scope="module", params=MOE_STEMS + MOE_GGUF)
def mixture(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    import btb

    need_mlx()
    sm = btb.load(os.path.join(FIXTURES, str(request.param)), device="mlx", log=NO_LOG)
    yield sm
    sm.close()


def _attn_layer(sm: StreamedTextModel) -> int:
    return max(i for i in range(sm.L) if sm.layer_types[i] != LayerKind.LINEAR)


def _verify(sm: StreamedTextModel, prompt: Tokens, toks: Tokens, parents: Parents) -> tuple[list[int], mx.array]:
    """a verify pass over a fresh cache after the prompt: its ids and the last attention layer's K rows"""
    cache = sm.new_cache(max_len=4096)
    with torch.inference_mode():
        sm._prefill(torch.tensor([prompt]), cache)
        sm.aa(parents)
        try:
            sm._pass_reset()
            out = sm.forward([toks], cache=cache, last_only=False, pick=True)
            report = sm.last_pass_report()
        finally:
            sm.ab()
    assert out is not None and PassTag.MLX_HYBRID in report, f"the pass took {report}"
    L = _attn_layer(sm)
    return [int(v) for v in out[0].tolist()], mlx_keys(cache, L)[0, :, : len(prompt) + len(toks)]


def _along(sm: StreamedTextModel, prompt: Tokens, toks: Tokens, parents: Parents, j: int) -> tuple[int, mx.array]:
    """node j's id and K row as the one-row step makes them, one row at a time along j's path"""
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
    return int(out[0, 0]), mlx_keys(cache, _attn_layer(sm))[0, :, row : row + 1]


def _landed(sm: StreamedTextModel) -> None:
    """every read the store has queued (a layer's fill) landed"""
    store = sm.expert_store
    assert store is not None
    for sl in list(store.slots.values()):
        if sl.parts is not None:
            sl.parts.result()


@pytest.mark.parametrize("parents", [[-1, 0, 1, 2], [-1, 0, 1, 0, 3, 3, 1]], ids=["chain4", "tree7"])
def test_a_mixtures_verify_pass_is_its_one_row_steps(mixture: StreamedTextModel, parents: Parents) -> None:
    """each node of a verify pass over the mixture fixture (the hybrid step: the picks on the GPU) is the one-row
    step along its path, its id and its K row bit for bit"""
    import mlx.core as mx

    sm = mixture
    V = int(sm.cfg.vocab_size)
    prompt = [1 + (7 * t) % (V - 1) for t in range(40)]
    toks = [1 + (13 * t + 5) % (V - 1) for t in range(len(parents))]
    ids, rows = _verify(sm, prompt, toks, parents)
    for j in range(len(parents)):
        f_id, f_row = _along(sm, prompt, toks, parents, j)
        assert ids[j] == f_id, f"node {j}: the pass's id {ids[j]} is not the step's {f_id}"
        assert mx.array_equal(rows[:, len(prompt) + j : len(prompt) + j + 1], f_row), f"node {j}: K rows differ"


def test_the_resident_gather_is_the_seated_picks(mixture: StreamedTextModel) -> None:
    """a pass before the store's layers have filled (the host seats each layer's picks) and the same pass once every
    expert has landed (each layer's resident gather, no host) give the same ids and rows, and the second took no seat"""
    import mlx.core as mx

    sm = mixture
    store = sm.expert_store
    assert store is not None and store.holds_all, "a fixture's store holds every expert"
    V = int(sm.cfg.vocab_size)
    prompt = [1 + (11 * t) % (V - 1) for t in range(30)]
    toks = [1 + (3 * t + 2) % (V - 1) for t in range(5)]
    parents = [-1, 0, 1, 0, 3]
    store.release(want=1 << 62)  # every block given back: the first pass's layers seat their picks
    seated = _verify(sm, prompt, toks, parents)
    _landed(sm)
    calls = store.stat["calls"]
    resident = _verify(sm, prompt, toks, parents)
    # the prompt's prefill asks the store a call a layer (its rows' own matmuls); the verify pass none
    layers = sum(1 for i in sm.host if sm.fam.mixture(sm.host[i].mlp) is not None)
    assert store.stat["calls"] - calls == layers, "a resident pass asked the store for its picks"
    assert seated[0] == resident[0] and mx.array_equal(seated[1], resident[1])


@pytest.mark.parametrize("proposer", [Proposer.MTP, Proposer.MTP_DYN], ids=lambda p: p.value)
def test_a_drafted_loop_decodes_as_the_plain_loop(mixture: StreamedTextModel, proposer: Proposer) -> None:
    """the mixture's speculative loop, its MTP head drafting through the store's layer past the trunk's, decodes the
    plain loop's tokens: the head's seats come and go between the trunk's passes, never under a graph that still
    reads a slot, and the store counts the head's experts among the ones it holds"""
    sm = mixture
    before, floor = sm.proposer, sm.tree_min_prob
    sm.proposer = proposer
    sm.tree_min_prob = 0.0  # a tiny random model's flat distribution clears no path-probability floor
    sm._host_cost, sm._mlx_cost, sm._card_cost = {}, {}, {}  # the configured width, not a tiny model's learnt one
    try:
        plain = [int(t) for t in sm.generate([1, 2, 3, 4], 12, speculate=False).tokens]
        drafted = [int(t) for t in sm.generate([1, 2, 3, 4], 12, speculate=True).tokens]
    finally:
        sm.proposer, sm.tree_min_prob = before, floor
    assert sm.last_pass_report().spec_proposed > 0, "nothing was drafted"
    assert drafted == plain, f"drafted {drafted} against the plain loop's {plain}"
    store = sm.expert_store
    assert store is not None and store.holds_all
    assert store.n_slots == store._expert_layers() * int(sm.n_experts) > int(sm.L) * int(sm.n_experts), "the head's"
