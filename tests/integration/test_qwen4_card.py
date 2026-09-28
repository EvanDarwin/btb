# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's card program (btb/engine/families/qwen4/card.py) over a Qwen4 at the card kernels' shapes
(tests/make_fixtures.py `build_q4_card`, built where the test runs): every node of a verify pass is the one-token
steps of its path on the card, bit for bit - short of the indexer's budget and past it, where the sparse attention
selects; the commit leaves the cache where those steps would; a captured replay is the eager run of the same kernels;
a speculative decode (the n-gram proposer over given spans, a chain and a tree, and the drafting head's proposers) is
the plain decode's tokens; the program's buffers are what it asked the scheduler for; and it is within tolerance of
the torch path it stands in for, which takes the passes where the program declines (a model at other shapes, or the
program turned off). With part of the model on the host - the first layers and the last, or two in the middle - the
program runs its resident layers in segments and the host layers between them through the host path, and all of the
above holds as it does with every layer on the card. One load a placement; every case runs inside it."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
import torch

from btb.engine.cache import ArenaIndexedLayer, indexer_keys
from btb.engine.forward import path_of
from tests.helpers import (
    CHUNK,
    DEPTH,
    PARENTS,
    PATH,
    fixture,
    forward_logits,
    host_model,
    layer_count,
    need_card_kernels,
    rel_err,
    speculation,
)

if TYPE_CHECKING:
    from btb.engine.model import StreamedTextModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")

# a prompt short of the indexer's budget (4 blocks of 4), and one past it (repeating, so the n-gram proposer drafts)
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8],
    "long": ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70],
}
NEXT = 77  # the token fed after the commit


# the placements with part of the model on the host (the fixture's layers: DeltaNet, the n-gram embedding's DeltaNet,
# sparse attention, DeltaNet, sparse attention): the first two and the last on the host - one segment between them,
# the tail after a host layer - or the middle two: two segments, the first closed into the streams for the host, the
# second a lone sparse layer
MIXED = {"ends_on_host": ([0, 1, 4], [2, 3]), "middle_on_host": ([2, 3], [0, 1, 4])}


@pytest.fixture(scope="module")
def card_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    from btb.engine import StreamedTextModel
    from tests.make_fixtures import build_q4_card

    need_card_kernels()
    path = str(tmp_path_factory.mktemp("q4card") / "tiny_q4_card")
    build_q4_card(path)
    StreamedTextModel.register_attention()
    return path


@pytest.fixture(scope="module")
def card(card_path: str) -> Iterator[StreamedTextModel]:
    path = card_path
    sm = host_model(path, device="cuda", dtype=torch.bfloat16, cpu_layers=[], resident_layers=range(layer_count(path)))
    try:
        yield sm
    finally:
        sm.close()


@pytest.fixture(scope="module", params=list(MIXED))
def mixed(request: pytest.FixtureRequest, card_path: str) -> Iterator[StreamedTextModel]:
    host, resident = MIXED[request.param]
    sm = host_model(card_path, device="cuda", dtype=torch.bfloat16, cpu_layers=host, resident_layers=resident)
    try:
        yield sm
    finally:
        sm.close()


def _prog(sm: StreamedTextModel) -> Any:
    prog = getattr(sm, "_cp", None)
    assert prog is not None, "the card program never ran"
    return prog


def _prefilled(sm: StreamedTextModel, prompt: list[int]) -> tuple[Any, torch.Tensor]:
    cache = sm.new_cache()
    return cache, forward_logits(sm, [prompt], cache)[0, -1]


def _steps(sm: StreamedTextModel, prompt: list[int], toks: list[int]) -> tuple[Any, list[torch.Tensor]]:
    """the cache and the logits after each of `toks` fed one at a time over the prefilled prompt (the program's
    one-row steps)"""
    cache, _ = _prefilled(sm, prompt)
    return cache, [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in toks]


def _tree(sm: StreamedTextModel, cache: Any) -> tuple[torch.Tensor, int]:
    base = cache.get_seq_length()
    sm.aa(PARENTS)
    try:
        lg = forward_logits(sm, [CHUNK], cache, last_only=False, positions=[[base + d for d in DEPTH]])[0].clone()
    finally:
        sm.ab()
    return lg, base


def _state(sm: StreamedTextModel, cache: Any) -> dict[str, torch.Tensor]:
    """every row and state the cache holds, copied: the sparse layers' keys, values and raw indexer keys, the
    DeltaNet's conv window and recurrent state, the n-gram embedding's kept inputs and ids"""
    out: dict[str, torch.Tensor] = {}
    for i, cl in enumerate(cache.layers):
        if hasattr(cl, "conv_states") and isinstance(cl.conv_states, dict):
            for k, v in cl.conv_states.items():
                if isinstance(v, torch.Tensor):
                    out[f"{i}.conv{k}"] = v.detach().clone().to(v.dtype if k == 2 else torch.float32)
            rec = cl.recurrent_states[0]
            out[f"{i}.rec"] = rec.detach().float().clone()
        else:
            ik = indexer_keys(cl)
            assert ik is not None
            out[f"{i}.k"], out[f"{i}.v"], out[f"{i}.ik"] = cl.keys.clone(), cl.values.clone(), ik.clone()
    return out


def _verify_is_steps(sm: StreamedTextModel) -> None:
    """each node of a tree's verify pass is the one-token steps of its path; the commit leaves the cache as the
    accepted path's steps do, and the step after it is theirs"""
    with torch.inference_mode():
        for name, prompt in PROMPTS.items():
            cache, _ = _prefilled(sm, prompt)
            lg, base = _tree(sm, cache)
            prog = _prog(sm)
            assert any(k[2] == "tree" for k in prog.graphs if len(k) == 4), f"{name}: the tree was not the program's"
            assert all(isinstance(cache.layers[i], ArenaIndexedLayer) for i in prog.sparse)
            sm.ad(cache, base, list(PATH))
            assert cache.get_seq_length() == base + len(PATH)
            got = _state(sm, cache)
            nxt = forward_logits(sm, [[NEXT]], cache)[0, -1].clone()
            del cache
            # each node against the one-token steps of its path, a fresh cache each
            for j in range(len(CHUNK)):
                path = path_of(PARENTS, j)[::-1]
                _c, steps = _steps(sm, prompt, [CHUNK[k] for k in path])
                assert torch.equal(lg[j], steps[-1]), (
                    f"{name}: node {j} parts from its path's steps by {float((lg[j] - steps[-1]).abs().max()):.3e}"
                )
            # the commit: the cache as the path's steps leave it, and the step after it
            ref, _ = _steps(sm, prompt, [CHUNK[k] for k in PATH])
            want = _state(sm, ref)
            assert set(got) == set(want)
            for key, t in want.items():
                assert torch.equal(got[key], t), f"{name}: after the commit {key} is not the steps'"
            assert torch.equal(nxt, forward_logits(sm, [[NEXT]], ref)[0, -1]), f"{name}: the step after the commit"


def test_a_verify_pass_is_its_paths_steps_and_commits_as_they_would(card: StreamedTextModel) -> None:
    _verify_is_steps(card)


def _captured_is_eager(sm: StreamedTextModel) -> None:
    """a captured step, verify pass, commit and step after it are the eager run of the same kernels"""
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        runs = []
        for eager in (False, True):
            setattr(sm, "card_program_eager", eager)  # noqa: B010  the runner's test knob
            try:
                cache, first = _prefilled(sm, prompt)
                step = forward_logits(sm, [[NEXT]], cache)[0, -1].clone()
                lg, base = _tree(sm, cache)
                sm.ad(cache, base, list(PATH))
                after = forward_logits(sm, [[NEXT]], cache)[0, -1].clone()
                runs.append((first.clone(), step, lg, after, _state(sm, cache)))
            finally:
                setattr(sm, "card_program_eager", False)  # noqa: B010
        (f0, s0, l0, a0, st0), (f1, s1, l1, a1, st1) = runs
        assert torch.equal(f0, f1), "the torch prefill is not repeatable: nothing below can be compared"
        assert torch.equal(s0, s1), "a captured step parts from the eager run"
        assert torch.equal(l0, l1), "a captured verify pass parts from the eager run"
        assert torch.equal(a0, a1), "the step after a captured commit parts from the eager run"
        for key, t in st0.items():
            assert torch.equal(st1[key], t), f"{key} after the captured run parts from the eager one"


def test_a_captured_replay_is_the_eager_run(card: StreamedTextModel) -> None:
    _captured_is_eager(card)


def _speculative_is_plain(sm: StreamedTextModel) -> None:
    """a speculative decode - the n-gram proposer over given spans as a tree and as a chain, the drafting head's
    proposers - is the plain decode's tokens, through the program's steps, verify passes and layer taps"""
    with torch.inference_mode():
        for name, prompt in PROMPTS.items():
            plain = sm.generate_greedy(prompt, 16)
            right = [prompt[-1], *plain]
            astray = [*right[:3], *(t + 1 for t in right[3:])]
            spans = [("plain", right), ("astray", astray)]
            for tree_budget in (8, 0):  # the n-gram continuations as one tree, then as a chain
                speculation(sm, tree_budget=tree_budget, tree_min_prob=0.0, ngram_p=0.9, v_max=4, price=False)
                toks, census = sm.generate_speculative(prompt, 16, proposer="ngram", v_max=4, spans=spans)
                assert toks == plain, f"{name}, tree budget {tree_budget}: {toks} != {plain}"
                assert census["proposed"] > 0, f"{name}: nothing was drafted, so nothing was verified"
            for proposer in ("mtp", "mtp_tree", "mtp_dyn"):
                speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
                toks, census = sm.generate_speculative(prompt, 16, proposer=proposer, v_max=4)
                assert toks == plain, f"{name}, {proposer}: {toks} != {plain}"
                assert census["mtp_steps"] > 0, f"{name}, {proposer}: the drafter never stepped"
        prog = _prog(sm)
        modes = {k[2] for k in prog.graphs if len(k) == 4}
        assert modes == {"step", "tree"}, f"the decodes ran the program's {modes} graphs"
        assert any(k[3] for k in prog.graphs if len(k) == 4), "the drafter's layer tap never ran the program"


def test_a_speculative_decode_is_the_plain_decode(card: StreamedTextModel) -> None:
    _speculative_is_plain(card)


def _stats(sm: StreamedTextModel) -> dict[str, Any]:
    """what the expert store and the engine's expert calls have counted so far"""
    store = sm.expert_store
    return {"store": dict(store.stat) if store is not None else {}, "calls": dict(sm.expert_stat)}


def _warm(sm: StreamedTextModel, t_max: int, per_width: int) -> dict[int, float]:
    """the warm-up from no graphs: it reads no expert, captures the verify graphs of each width once
    (`per_width` a width: the resident layers', the closes' and the tail's) and nothing else, and its curve is a
    width's cost - the passes padded to one width cost the same"""
    prog = _prog(sm)
    prog._drop_graphs()
    before = _stats(sm)
    widths = {sm._card_m(T) for T in range(1, t_max + 1)}
    assert sm.card_warm(PROMPTS["short"], t_max=t_max) == per_width * len(widths)
    assert _stats(sm) == before, "the warm-up read experts through the store"
    assert {k[-2] for k in prog.graphs if k[0] != "commit"} == {"tree"}, "the warm-up captured more than it timed"
    cost: dict[int, float] = getattr(sm, "_card_cost", {})
    assert set(cost) == set(range(1, t_max + 1)) and all(c > 0 for c in cost.values())
    for T in cost:
        assert cost[T] == cost[sm._card_m(T)], f"a {T}-row pass is not priced as its graphs' width"
    return cost


def test_the_program_holds_what_it_was_granted(card: StreamedTextModel) -> None:
    sm = card
    # the warm-up times each verify width's graphs on the card alone - no expert read, the step's graphs left to
    # their first use - for the budget, which prices the rows' expert reads apart
    cost = _warm(sm, 4, int(_prog(sm).L) + 1)
    assert cost[2] < 2 * cost[1], f"a 2-row pass costs {cost[2] / cost[1]:.2f} one-row passes on the card's kernels"
    with torch.inference_mode():
        cache, _ = _prefilled(sm, PROMPTS["short"])
        forward_logits(sm, [[NEXT]], cache)
        prog = _prog(sm)
        L = int(prog.L)
        # one graph a layer and the tail for each (rows, mode, taps) a pass ran, and the commit's
        shapes = {k[1:] for k in prog.graphs if len(k) == 4}
        assert len([k for k in prog.graphs if len(k) == 4]) == (L + 1) * len(shapes)
        assert prog.held_bytes() == sum(prog.held.values()), (
            f"the program holds {prog.held_bytes()} bytes, was granted {sum(prog.held.values())}: {prog.held}"
        )
        granted = sm.scheduler.granted
        assert granted.get("scratch@cuda", 0) >= prog.held["the pass buffers"]
        assert granted.get("kv@cuda", 0) >= prog.held["the linear layers' states"]
        assert granted.get("scratch@cpu", 0) >= prog.held["the host's pinned rows"]


def test_the_torch_path_takes_what_the_program_declines(card: StreamedTextModel) -> None:
    """off (the engine's knob), the program leaves the pass to the torch path, which the program's step and
    verify pass stay within tolerance of"""
    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        out = {}
        for on in (True, False):
            setattr(sm, "card_programs", on)  # noqa: B010  the engine's knob
            try:
                n_graphs = len(_prog(sm).graphs) if getattr(sm, "_cp", None) is not None else 0
                cache, _ = _prefilled(sm, prompt)
                step = forward_logits(sm, [[NEXT]], cache)[0, -1].float()
                lg, _base = _tree(sm, cache)
                out[on] = (step, lg.float())
                if not on:
                    assert len(_prog(sm).graphs) == n_graphs, "the program ran while turned off"
            finally:
                setattr(sm, "card_programs", True)  # noqa: B010
        (s1, t1), (s0, t0) = out[True], out[False]
        # bf16 through five layers whose sums run in the kernels' order, not cuBLAS's and sdpa's (the DeltaNet layers
        # agree to the bit; the sparse attention's reductions part in the last bits)
        assert rel_err(s1, s0) < 0.02, f"the program's step is {rel_err(s1, s0):.4f} from the torch path's"
        assert rel_err(t1, t0) < 0.02, f"the program's verify pass is {rel_err(t1, t0):.4f} from the torch path's"


def test_the_program_declines_a_model_at_other_shapes() -> None:
    """tiny_q4's heads (32 wide) are not the kernels': the program declines and the card runs the torch path"""
    from btb.engine import StreamedTextModel

    need_card_kernels()
    path = fixture("tiny_q4")
    StreamedTextModel.register_attention()
    sm = host_model(path, device="cuda", dtype=torch.bfloat16, cpu_layers=[], resident_layers=range(layer_count(path)))
    try:
        with torch.inference_mode():
            cache, _ = _prefilled(sm, PROMPTS["short"])
            assert sm._card_program(cache, 1, 1, cache.get_seq_length(), None, None, None) is None
            prog = getattr(sm, "_cp", None)
            assert prog is not None and prog.why_not(sm) == "shapes the kernels are not written for"
            toks = sm.generate_greedy(PROMPTS["short"], 4)
            assert len(toks) == 4 and not prog.graphs
    finally:
        sm.close()


def test_experts_seated_on_the_card_change_no_bit(card: StreamedTextModel) -> None:
    """experts seated on the card (the store's VRAM seats) multiply as the host does (`_Experts._seated`): the
    program's steps and verify pass read the same bits with them as without"""
    from btb.engine.experts import VramSeats

    sm = card
    store = sm.expert_store
    assert store is not None and store.per is not None
    prompt, toks = PROMPTS["long"], [NEXT, 5, 9]

    def run() -> list[torch.Tensor]:
        cache, _ = _prefilled(sm, prompt)
        out = [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in toks]
        lg, base = _tree(sm, cache)
        sm.ad(cache, base, list(PATH))
        return [*out, lg, forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]

    with torch.inference_mode():
        base = run()
        gu_n, gu_shape, dn_shape = store.shapes
        store.vram = VramSeats(
            64, store._held(store.per), (store._held(gu_n), gu_shape, dn_shape), sm.dev, min_rides=1, per_pass=64
        )
        try:
            run()  # the first rides seat the experts
            seated = run()
            assert store.vram.copies > 0 and store.vram.seat_of, "no expert was seated on the card"
        finally:
            store.vram = None
    for j, (a, b) in enumerate(zip(base, seated, strict=True)):
        assert torch.equal(a, b), f"pass {j} parts with experts seated on the card"


def test_an_arena_grown_under_a_cache_drops_the_graphs_and_keeps_the_rows(card: StreamedTextModel) -> None:
    """the arena regrown mid-decode (a context past its rows): the rows copied over, every bound layer attached to
    the new one, the graphs dropped and captured again - the decode goes on as if nothing moved"""
    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        ref, steps = _steps(sm, prompt, [NEXT, 5, 9])
        del ref
        cache, _ = _prefilled(sm, prompt)
        got = [forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]
        prog = _prog(sm)
        cap = int(prog.A["cap"])
        prog.arena(cap + 1)
        assert int(prog.A["cap"]) > cap and not prog.graphs, "the arena did not grow, or kept its graphs"
        assert prog.held_bytes() == sum(prog.held.values())
        got += [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in (5, 9)]
    for j, (a, b) in enumerate(zip(steps, got, strict=True)):
        assert torch.equal(a, b), f"step {j} parts once the arena grew"


# -- part of the model on the host --------------------------------------------------------------------------------


def test_a_mixed_placement_runs_the_program_in_segments_exact_to_its_steps(mixed: StreamedTextModel) -> None:
    """with layers on the host the program runs the resident ones in segments and the host ones between them: a
    verify pass's every node is its path's one-token steps and the commit theirs, across the card's layers and the
    host's alike; a captured pass is the eager one; a speculative decode is the plain one; the graphs are the resident
    layers', a close where a segment hands its streams to the host, and the tail; and the program holds what it was
    granted, the edge's rows among it"""
    sm = mixed
    host = sorted(sm.host)
    L = int(sm.L)
    res = [i for i in range(L) if i not in host]
    ends = [i for i in res if i + 1 < L and i + 1 in host]  # the segments' closes
    sm.card_warm(PROMPTS["short"], t_max=1)  # the program made, its layout read
    _warm(sm, 2, len(res) + 1 + len(ends))
    with torch.inference_mode():
        cache, _ = _prefilled(sm, PROMPTS["short"])
        forward_logits(sm, [[NEXT]], cache)
        prog = _prog(sm)
        why = prog.why_not(sm)
        assert why is None and prog.host_layers == host and host, f"the program declined the placement: {why}"
        assert all(isinstance(cache.layers[i], ArenaIndexedLayer) for i in prog.sparse)
        assert not any(isinstance(cache.layers[i], ArenaIndexedLayer) for i in host), "a host layer's rows in the arena"
    _verify_is_steps(sm)
    _captured_is_eager(sm)
    _speculative_is_plain(sm)
    closes = sorted({k[1] for k in prog.graphs if k[0] == "close"})
    assert closes == [b - 1 for _a, b in prog.segs if b < L], f"the segments {prog.segs} closed after {closes}"
    layers = {k[0] for k in prog.graphs if len(k) == 4}
    assert layers == {*res, L}, f"graphs for layers {sorted(layers)}, resident {res}"
    assert prog.held_bytes() == sum(prog.held.values()), (
        f"the program holds {prog.held_bytes()} bytes, was granted {sum(prog.held.values())}: {prog.held}"
    )
    edge = prog.held["the edge's rows"]
    assert edge > 0 and sm.scheduler.granted.get("scratch@cpu", 0) >= edge + prog.held["the host's pinned rows"]


def test_a_placement_whose_edge_the_ledger_refuses_takes_the_torch_path(
    mixed: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The edge's rows are asked for before a pass, with the expert store allowed to give room back for them (it grows
    into the RAM the ledger shows free, so a program first built after it has finds none). Refused even so, the
    program declines the placement: its passes take the torch path, none fails part way, the refusal is its reason."""
    from btb.engine.scheduler import MemoryGrantError

    sm = mixed
    prompt = PROMPTS["short"]
    with torch.inference_mode():
        _prog(sm).close()
        sm._cp = None  # a program built afresh, as one first built after the store has grown
        real = sm.scheduler.grant
        asked: list[Any] = []

        def grant(nbytes: int, kind: str, **kw: Any) -> None:
            if "the edge's rows" in str(kw.get("requester", "")):
                asked.append(kw.get("reclaim"))
                raise MemoryGrantError("[grant] REFUSED the test's edge: no room")
            real(nbytes, kind, **kw)

        monkeypatch.setattr(sm.scheduler, "grant", grant)
        toks = sm.generate_greedy(prompt, 8)
        prog = _prog(sm)
        assert asked == [True], f"the edge asked for {len(asked)} times; reclaim {asked}"
        assert not prog.ok() and "the edge's rows refused" in str(prog.why_not(sm) or prog._why)
        assert not prog.graphs, "a pass ran the program after it declined"
        assert len(toks) == 8, "the decode ran to its end on the torch path"
        prog.close()
        sm._cp = None
