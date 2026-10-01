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

import dataclasses
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
# sparse attention, DeltaNet, sparse attention), as (host, resident, read from the drive each pass): the first two and
# the last on the host - one segment between them, the tail after a host layer - or the middle two: two segments, the
# first closed into the streams for the host, the second a lone sparse layer; or the sparse layers alone on the card,
# a lone sparse layer a segment and the DeltaNet between them read from the drive - no DeltaNet state and no n-gram
# embedding in the program at all
MIXED = {
    "ends_on_host": ([0, 1, 4], [2, 3], []),
    "middle_on_host": ([2, 3], [0, 1, 4], []),
    "sparse_only_drive_between": ([0, 1, 3], [2, 4], [3]),
}


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
    host, resident, cold = MIXED[request.param]
    sm = host_model(
        card_path, device="cuda", dtype=torch.bfloat16, cpu_layers=host, resident_layers=resident, cold_layers=cold
    )
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
    proposers - is the plain decode's tokens, through the program's steps, verify passes and layer taps. The passes'
    widths are the drafts asked for, not what a warm-up's timing allows (a 2-row pass timed over a quarter past a
    step on a busy card sized every pass to one row, and nothing was verified)"""
    cost = sm.__dict__.pop("_card_cost", None)
    try:
        _speculative_passes(sm)
    finally:
        if cost is not None:
            sm._card_cost = cost


def _speculative_passes(sm: StreamedTextModel) -> None:
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
    cost: dict[int, float] = getattr(sm, "_card_prog_cost", {})
    assert set(cost) == set(range(1, t_max + 1)) and all(c > 0 for c in cost.values())
    # the engine's pass cost only where the program runs the whole model: the host layers are not in its curve
    assert (getattr(sm, "_card_cost", None) is cost) == (not sm.host), "a partial curve priced the whole pass"
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


def test_a_let_go_drops_the_programs_blocks_and_the_next_pass_reads_them_again(card: StreamedTextModel) -> None:
    """a shed lets the card's graphs go first (`_card_let_go`), the program's with them: its merged weight blocks,
    their float32 operands and its graphs - held, a layer shed freed nothing on the card, and a yield gave up every
    layer and the head for one cut of the budget. The next pass reads them again, and its logits are the ones
    before, bit for bit"""
    sm = card
    with torch.inference_mode():
        cache, _ = _prefilled(sm, PROMPTS["short"])
        before = forward_logits(sm, [[NEXT]], cache)[0, -1].clone()
        prog = _prog(sm)
        assert prog.W and prog.graphs, "no pass ran the program"
        sm._card_let_go()
        assert not prog.W and not prog.graphs, "the program kept its blocks or its graphs through a let-go"
        assert not [w for w in prog.held if w.endswith("operands in float32")], f"operands still held: {prog.held}"
        assert prog.held_bytes() == sum(prog.held.values()), prog.held
        del cache
        cache, _ = _prefilled(sm, PROMPTS["short"])
        after = forward_logits(sm, [[NEXT]], cache)[0, -1]
        assert prog.W and prog.graphs, "the pass after the let-go did not run the program"
    assert torch.equal(before, after), f"the pass after a let-go parts by {float((before - after).abs().max()):.3e}"


def test_a_state_left_in_bf16_is_widened_in_place_on_the_torch_path(card: StreamedTextModel) -> None:
    """the torch path's DeltaNet step on the card (the program off) over a recurrent state a pass left in bf16: the
    state is widened to float32 in place in the cache, once, and the step computes as it does over the same values
    handed to it in float32; a verify pass's kept path whose engine is gone steps nothing"""
    from btb.engine.families.qwen4.verify import _PathStep

    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        setattr(sm, "card_programs", False)  # noqa: B010  the engine's knob
        try:
            out = {}
            for how in ("bf16", "widened"):
                cache, _ = _prefilled(sm, prompt)
                for i in [i for i, lt in enumerate(sm.layer_types) if lt == "linear_attention"]:
                    conv, rec = sm._lin(cache.layers[i])
                    low = rec.bfloat16()
                    sm._lin_set(cache.layers[i], conv, low if how == "bf16" else low.float())
                lg = forward_logits(sm, [[NEXT]], cache)[0, -1].clone()
                recs = [
                    sm._lin(cache.layers[i])[1] for i in range(int(sm.L)) if sm.layer_types[i] == "linear_attention"
                ]
                assert all(r.dtype == torch.float32 and r.is_contiguous() for r in recs), how
                out[how] = lg
        finally:
            setattr(sm, "card_programs", True)  # noqa: B010
    assert torch.equal(out["bf16"], out["widened"]), "the widened state steps otherwise than the same values in float32"

    class Gone:
        pass

    ghost = Gone()
    step = _PathStep(ghost, None, None, None, None, None, None, {})
    del ghost
    step.restore([0, 1])  # with no rows kept (None), a step it took would raise


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
            assert sm.card_warm(PROMPTS["short"], t_max=2) == 0 and not prog.graphs, "a declined program was timed"
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


# -- a cache the program holds, changed under it --------------------------------------------------------------------


def test_a_placement_moved_under_a_bound_cache_lets_it_go_and_binds_it_again(card: StreamedTextModel) -> None:
    """a placement that moves a layer on or off the card lets go of every buffer the resident layers sized: the cache
    bound to them keeps copies of its rows and states (its sparse layers detached from the arena, one that let go of
    it already left as it is). The next pass reads
    the placement afresh - the weights read again, a merged block taken where it lies - binds the cache again, its
    rows copied back into the arena, and the decode goes on as its steps"""
    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        _ref, steps = _steps(sm, prompt, [NEXT, 5, 9])
        cache, _ = _prefilled(sm, prompt)
        got = [forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]
        prog = _prog(sm)
        L, sparse, linear = int(prog.L), list(prog.sparse), list(prog.linear)
        assert prog.S is not None
        mine = {id(t) for ts in prog.S.values() for t in ts}
        merged = prog.weights(linear[0])["proj"]
        cache.layers[sparse[0]].detach()  # one layer let go of the arena already: its rows a copy of their own
        prog._layout(tuple(range(L - 1)))  # as a placement moving the last layer off the card lays the program out
        assert prog.owner is None and prog.B is None and prog.A is None and prog.S is None and not prog.graphs
        assert all(cache.layers[i]._arena is None for i in sparse), "a sparse layer still views the arena"
        held = [t for i in linear for t in sm._lin(cache.layers[i])]
        assert not any(id(t) in mine for t in held), "the cache still holds the program's states"
        prog._layout(tuple(range(L - 1)))  # the same layout again: nothing let go or made
        assert prog.res == tuple(range(L - 1)) and prog.B is None
        prog.version = None  # the next pass reads the placement afresh
        got += [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in (5, 9)]
        assert prog.res == tuple(range(L)) and prog.owner is not None and prog.owner() is cache
        assert prog.A is not None
        for j, i in enumerate(sparse):
            assert cache.layers[i].attached_to(prog.A["kv"][j, 0]), f"layer {i} not bound again"
        again = prog.weights(linear[0])["proj"]
        assert again is not merged and again.data_ptr() == merged.data_ptr(), "the merged block was made anew"
        assert prog.held_bytes() == sum(prog.held.values())
    for j, (a, b) in enumerate(zip(steps, got, strict=True)):
        assert torch.equal(a, b), f"step {j} parts once the placement moved"


def test_a_placement_whose_copies_the_ledger_refuses_is_declined_and_laid_out_again(
    card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a placement moved under a bound cache whose rows' copies out of the arena the ledger refuses part way: the
    program declines it - no error out of the pass's check - with its old buffers kept, so every sparse layer reads
    rows that stand (one let go of the arena, its rows a copy of their own; the other still the arena's); the next look
    lays the placement out again, binds the cache, and the decode goes on as its steps"""
    from btb.engine.scheduler import MemoryGrantError

    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        _ref, steps = _steps(sm, prompt, [NEXT, 5, 9])
        cache, _ = _prefilled(sm, prompt)
        got = [forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]
        prog = _prog(sm)
        assert prog.ok() and prog.A is not None
        A, sparse, L = prog.A, list(prog.sparse), int(prog.L)
        assert len(sparse) >= 2
        before = [t.clone() for t in (cache.layers[sparse[0]].keys, cache.layers[sparse[0]].indexer_keys)]
        copies = []

        def counted(real: Any) -> Any:
            # a bound layer asks through the grant it was bound with, not the scheduler's attribute read afresh
            def grant(nbytes: int, kind: str, **kw: Any) -> None:
                if "arena rows, copied out" in str(kw.get("requester", "")):
                    copies.append(nbytes)
                    if len(copies) == 2:
                        raise MemoryGrantError("[grant] REFUSED the test's copy: no room")
                real(nbytes, kind, **kw)

            return grant

        # a placement changed under the program (a layer moved elsewhere and back): its layout is another placement's
        # than the engine's now, so the next look lays it out again - and keeps doing so until one lands
        assert prog.res == tuple(range(L))
        stale = tuple(reversed(prog.res))
        prog.res = stale
        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", counted(sm.scheduler.grant))
            for i in sparse:
                mp.setattr(cache.layers[i], "grant", counted(cache.layers[i].grant))
            moved = dataclasses.replace(sm.device.snapshot(), version=-1)
            mp.setattr(sm.device, "snapshot", lambda: moved)
            assert not prog.ok(), "the program took a placement whose copies were refused"
            assert "copied out" in str(prog._why) and len(copies) == 2
        assert prog.A is A and prog.res == stale, "the old layout let go with a layer still attached to it"
        first, second = cache.layers[sparse[0]], cache.layers[sparse[1]]
        assert not first.attached and second.attached_to(A["kv"][1, 0])
        assert torch.equal(first.keys, before[0]) and torch.equal(first.indexer_keys, before[1])
        got += [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in (5, 9)]
        assert prog.ok() and prog.owner is not None and prog.owner() is cache
        assert prog.A is not None and all(cache.layers[i].attached_to(prog.A["kv"][j, 0]) for j, i in enumerate(sparse))
    for j, (a, b) in enumerate(zip(steps, got, strict=True)):
        assert torch.equal(a, b), f"step {j} parts once a refused layout was laid out again"


def test_a_bound_layer_given_up_to_the_host_moves_its_rows_straight_there(
    card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a sparse layer the program holds, its rows moved to the host as a shed moves a layer's (`_cache_to`): every row
    goes straight to the host, asked for there - nothing copied, or asked for, on the card, the room a shed is short
    of - the layer leaves the arena, and an arena grown after it does not attach it again"""
    sm = card
    with torch.inference_mode():
        cache, _ = _prefilled(sm, PROMPTS["long"])
        forward_logits(sm, [[NEXT]], cache)
        prog = _prog(sm)
        i = prog.sparse[0]
        cl = cache.layers[i]
        assert isinstance(cl, ArenaIndexedLayer) and cl.attached
        assert cl.keys is not None and cl.values is not None and cl.indexer_keys is not None
        rows = [t.clone() for t in (cl.keys, cl.values, cl.indexer_keys)]
        real = cl.grant  # the grant the layer was bound with, which its move asks through
        assert real is not None, "a bound layer with no grant to ask its move through"
        asked: list[str] = []

        def grant(nbytes: int, kind: str, **kw: Any) -> None:
            asked.append(torch.device(kw.get("device") or sm.dev).type)
            real(nbytes, kind, **kw)

        with monkeypatch.context() as mp:
            mp.setattr(cl, "grant", grant)
            sm._cache_to(cache, i, "cpu")
        assert not cl.attached and asked == ["cpu"], f"the move asked for room on {asked}"
        for a, b in zip(rows, (cl.keys, cl.values, cl.indexer_keys), strict=True):
            assert b is not None and b.device.type == "cpu" and torch.equal(a.cpu(), b)
        assert prog.A is not None
        prog.arena(int(prog.A["cap"]) + 1)
        assert not cl.attached, "the arena's regrowth attached a layer given up to the host"
        prog.close()


def test_a_fork_of_a_bound_session_keeps_its_rows_while_another_cache_takes_the_arena(card: StreamedTextModel) -> None:
    """a fork of a session the program holds takes rows of its own - the session's sparse layers let the arena go,
    their rows a copy asked of the ledger - so another cache bound to the program before the fork's first step, its
    rows written over the arena's front, leaves the fork's steps as they are with nothing between"""
    sm = card

    def forked(between: bool) -> torch.Tensor:
        s = sm.session(PROMPTS["long"])
        s.feed([NEXT])
        prog = _prog(sm)
        assert prog.owner is not None and prog.owner() is s.cache, "the session is not the program's"
        with s.fork(2) as br:
            assert s.cache is not None
            assert not any(s.cache.layers[i].attached for i in prog.sparse), "the session's rows still the arena's"
            if between:
                other = sm.session(PROMPTS["short"][::-1])
                other.feed([7])
                assert prog.owner() is other.cache, "the other session did not take the program"
            lg = br.step([5, 9]).logits
            assert lg is not None
            return lg.clone()

    with torch.inference_mode():
        want = forked(False)
        got = forked(True)
    assert torch.equal(want, got), "the fork's steps read another session's rows"


def test_a_state_a_torch_path_replaced_is_taken_back(card: StreamedTextModel) -> None:
    """a torch path over the bound cache may leave a state tensor of its own where the program's was - a DeltaNet
    layer's conv window, or its recurrent state: the next pass copies it into the program's and binds that again,
    and the decode goes on as its steps"""
    sm = card
    prompt = PROMPTS["long"]
    with torch.inference_mode():
        _ref, steps = _steps(sm, prompt, [NEXT, 5, 9])
        cache, _ = _prefilled(sm, prompt)
        got = [forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]
        prog = _prog(sm)
        assert prog.S is not None
        j, i = 0, prog.linear[0]
        cl = cache.layers[i]
        for t, which in ((5, "conv"), (9, "rec")):
            conv, rec = sm._lin(cl)
            sm._lin_set(cl, conv.clone() if which == "conv" else conv, rec.clone() if which == "rec" else rec)
            got.append(forward_logits(sm, [[t]], cache)[0, -1].clone())
            conv, rec = sm._lin(cl)
            assert conv is prog.S["conv"][j] and rec is prog.S["rec"][j], f"the {which} state was not bound again"
    for k, (a, b) in enumerate(zip(steps, got, strict=True)):
        assert torch.equal(a, b), f"step {k} parts once a torch path replaced a state"


def test_a_chunk_the_torch_path_writes_past_the_arena_grows_it(
    card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a chunk of rows the torch path writes into the bound cache (a pass of many rows outside speculation) past the
    arena's rows: the layer asks the program for room, which regrows the arena - the rows copied, every bound layer
    attached again, the graphs dropped - and, the pass being off the program, reads the cache's mirrors again. The
    steps after it are the same decode's over an arena that never had to grow"""
    sm = card
    prompt = PROMPTS["short"]
    chunk = [(7 * i + 3) % 500 + 2 for i in range(1100)]

    def run() -> list[torch.Tensor]:
        cache, _ = _prefilled(sm, prompt)
        out = [forward_logits(sm, [[NEXT]], cache)[0, -1].clone()]
        forward_logits(sm, [chunk], cache)
        out += [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in (5, 9)]
        assert cache.get_seq_length() == len(prompt) + 1 + len(chunk) + 2
        return out

    with torch.inference_mode():
        want = run()
        prog = _prog(sm)
        assert prog.A is not None and int(prog.A["cap"]) >= len(prompt) + len(chunk) + 3 + prog.ROWS
        prog.close()  # the arena made afresh, at the smallest it starts at
        monkeypatch.setattr(prog, "ARENA_MIN", 1024)
        logs: list[str] = []
        monkeypatch.setattr(sm, "log", lambda *a, **_k: logs.append(" ".join(str(x) for x in a)))
        got = run()
        assert prog.A is not None and int(prog.A["cap"]) == 2048
        assert any("arena grown to 2048 rows" in line for line in logs), logs
        assert prog.held_bytes() == sum(prog.held.values())
    for j, (a, b) in enumerate(zip(want, got, strict=True)):
        assert torch.equal(a, b), f"step {j} parts once the chunk grew the arena"


def test_a_cache_rewound_to_an_earlier_row_goes_on_as_its_steps(card: StreamedTextModel) -> None:
    """a cache rewound as a session resuming at a shorter prefix rewinds it - the sparse layers' rows cropped, the
    DeltaNet and n-gram states restored from a snapshot: the pooled keys past the rewind are pooled again, the n-gram
    context read again, and the steps after are those of a decode that never went past it"""
    sm = card
    prompt, head, astray, tail = PROMPTS["long"], [NEXT, 5, 9], [1, 2, 3, 4, 6, 8, 10, 11], [12, 40]
    with torch.inference_mode():
        _ref, want = _steps(sm, prompt, head + tail)
        cache, _ = _prefilled(sm, prompt)
        for t in head:
            forward_logits(sm, [[t]], cache)
        prog = _prog(sm)
        snap = {i: sm._lin_snap(cache.layers[i]) for i in prog.linear}
        for t in astray:
            forward_logits(sm, [[t]], cache)
        r = max(1, int(prog.r))
        assert (len(prompt) + len(head) + len(astray)) // r > (len(prompt) + len(head)) // r, "no block to pool again"
        for i in prog.sparse:
            cache.layers[i].crop(-len(astray))
        for i, s in snap.items():
            sm._lin_restore(cache.layers[i], s)
        assert cache.get_seq_length() == len(prompt) + len(head)
        got = [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in tail]
    for j, (a, b) in enumerate(zip(want[len(head) :], got, strict=True)):
        assert torch.equal(a, b), f"step {j} after the rewind parts from the steps that never went past it"


def test_a_pass_that_fails_part_way_leaves_the_next_exact(
    card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a router that never publishes is a named error, not a hang; a pass that raises part way (the host's part
    failing at a layer) leaves graphs that may still publish, which the next pass - a warm-up's or a decode's - lets
    finish first: the decode after it is its steps"""
    sm = card
    prompt = PROMPTS["short"]
    with torch.inference_mode():
        _ref, steps = _steps(sm, prompt, [NEXT, 5, 9])
        prog = _prog(sm)
        with monkeypatch.context() as mp:
            mp.setattr(prog, "SPIN_S", 0.0)
            with pytest.raises(RuntimeError, match=r"\[card\] layer 0's router did not publish within 0 s"):
                prog.between(0, dry=True)
        real = prog.between

        def fails(i: int, dry: bool = False) -> None:
            if i == 2 and not dry:
                raise RuntimeError("the test's host fault at layer 2")
            real(i, dry)

        for then in ("warm-up", "decode"):
            cache, _ = _prefilled(sm, prompt)
            with monkeypatch.context() as mp:
                mp.setattr(prog, "between", fails)
                with pytest.raises(RuntimeError, match="the test's host fault at layer 2"):
                    forward_logits(sm, [[NEXT]], cache)
            assert not prog.clean, "a pass that raised part way counts as ended"
            if then == "warm-up":
                assert sm.card_warm(prompt, t_max=1) >= 0 and prog.clean
            _c, got = _steps(sm, prompt, [NEXT, 5, 9])
            for j, (a, b) in enumerate(zip(steps, got, strict=True)):
                assert torch.equal(a, b), f"after a failed pass and a {then}, step {j} parts"


def test_a_commit_is_its_own_passes_and_a_headless_pass_the_final_rows(card: StreamedTextModel) -> None:
    """a verify pass's commit steps its path into the states only for its own pass: an empty path commits nothing,
    and asked after another pass ran on the program it is refused (the pass buffers hold that pass's projections by
    then). A pass asked without the head returns the final mixer's rows the headed pass computed, bit for bit. The
    n-gram rows are staged once a pass, whichever of the host's parts comes to them first"""
    sm = card
    prompt = PROMPTS["short"]
    with torch.inference_mode():
        cache, _ = _prefilled(sm, prompt)
        _lg, _base = _tree(sm, cache)
        prog = _prog(sm)
        before = _state(sm, cache)
        prog.commit([])
        after = _state(sm, cache)
        assert all(torch.equal(after[k], t) for k, t in before.items()), "an empty path stepped a state"
        other, _ = _prefilled(sm, prompt)
        forward_logits(sm, [[NEXT]], other)
        with pytest.raises(RuntimeError, match="a verify pass's commit asked after another pass ran on the program"):
            prog.commit(list(PATH))
        h1 = prog.hidden(1).clone()
        again, _ = _prefilled(sm, prompt)
        h2 = sm.forward([[NEXT]], cache=again, head=False)
        assert h2 is not None and tuple(h2.shape) == (1, 1, int(sm.cfg.hidden_size)) and torch.equal(h2, h1)
        assert prog.B is not None
        staged = prog.B["emb_h"].clone()
        prog.ids = [t + 1 for t in prog.ids]  # other ids: a second staging would write other rows
        prog.staged = True
        prog._stage()
        assert torch.equal(prog.B["emb_h"], staged), "a pass's n-gram rows were staged twice"


def test_without_the_cards_kernels_no_program_is_made(card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """the card's kernels bound afresh, the engine names them and the card's cache sizes (read once and held); with
    none to bind no program is made, and the warm-up measures nothing - nor where the placement moved between the
    program's check and the pass's hold on it"""
    from btb.engine.native import Native

    sm = card
    real = Native.card_kernels()
    assert real is not None
    prompt = PROMPTS["short"]
    ids = torch.tensor([prompt])
    cs = sm.scheduler.caches()
    assert sm.scheduler.caches() is cs and cs["gpu_l2"] > 0, "the card's L2 not read, or read again"
    with torch.inference_mode(), monkeypatch.context() as mp:
        logs: list[str] = []
        mp.setattr(sm, "log", lambda *a, **_k: logs.append(" ".join(str(x) for x in a)))
        mp.setattr(Native, "cuda", None)
        mp.setattr(Native, "card_kernels", staticmethod(lambda: real))
        assert sm._card_kernels() is real
        line = next((ln for ln in logs if ln.startswith("[card] kernels")), "")
        assert f"gpu_l2 {cs['gpu_l2'] / 2**20:.0f} MB" in line, logs
        mp.setattr(Native, "card_kernels", staticmethod(lambda: None))
        mp.setattr(sm, "_cp", None)
        cache, _ = _prefilled(sm, prompt)
        assert sm._card_program(cache, 1, 1, cache.get_seq_length(), None, None, None) is None
        assert sm._cp is None, "a program was made with no kernels to run it"
        assert sm._card_program_warm(ids, 2) == 0
    prog = _prog(sm)
    with torch.inference_mode(), monkeypatch.context() as mp:
        mp.setattr(prog, "ok", lambda: True)
        mp.setattr(prog, "version", object())  # the placement moved after the check
        n = len(prog.graphs)
        assert sm._card_program_warm(ids, 2) == 0 and len(prog.graphs) == n


@pytest.fixture(scope="module")
def storeless(card_path: str) -> Iterator[StreamedTextModel]:
    """the card fixture with no expert store: the experts multiplied from the checkpoint's tables"""
    L = layer_count(card_path)
    sm = host_model(
        card_path, device="cuda", dtype=torch.bfloat16, cpu_layers=[], resident_layers=range(L), expert_cache_gb=0
    )
    try:
        yield sm
    finally:
        sm.close()


def test_a_model_with_no_expert_store_runs_the_program_exact_to_its_steps(storeless: StreamedTextModel) -> None:
    """with no store (the experts from the checkpoint's tables, no lookahead to read a router off its module) the
    program runs the model all the same: each node of a verify pass is its path's steps, the commit theirs"""
    assert storeless.expert_store is None
    _verify_is_steps(storeless)
    assert _prog(storeless).why_not(storeless) is None


# -- what keeps the program off -----------------------------------------------------------------------------------


def test_the_program_names_what_keeps_it_off(card: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """`why_not` over the loaded model with one thing changed at a time - the card's kernels, the compute, the tier,
    the layers as placed, the head and the final mixer, the routing, the experts' serving, a PLE layer the config
    does not name - names that thing, and with nothing changed declines nothing. A program built afresh reads each
    (the live one is left as it is); one whose engine is gone runs nothing, a matrix it cannot take is refused, and
    with no scheduler to ask it counts its bytes alone"""
    import types

    from btb.engine.families.qwen4 import card as mod
    from btb.engine.native import Native
    from btb.kinds import LayerKind

    sm = card
    L = int(sm.L)
    kern = Native.card_kernels()
    assert kern is not None
    lacking = lambda keep: types.SimpleNamespace(fn={n: kern.fn[n] for n in keep})
    cases: list[tuple[str, list[tuple[Any, str, Any]], Any, str | None]] = [
        ("as loaded", [], None, None),
        ("no kernels", [(Native, "card_kernels", staticmethod(lambda: None))], None, "no card kernels"),
        ("no hc kernel", [(Native, "card_kernels", staticmethod(lambda: lacking(())))], None, "the kernels lack btb_hc_rmsnorm"),
        (
            "no sparse kernels",
            [(Native, "card_kernels", staticmethod(lambda: lacking(mod._KERNELS)))],
            None,
            "the kernels lack btb_norm_rope_part_d128",
        ),
        ("float32", [(sm, "compute_dtype", torch.float32)], None, "a float32 compute"),
        ("resident fp32", [(sm, "resident_fp32", True)], None, "a float32 compute"),
        ("kv on the host", [(sm, "kv_host", True)], None, "not the card's tier"),
        ("no layer", [], (), "no layer on the card"),
        ("streamed", [], tuple(range(L - 1)), f"layer {L - 1} streamed through the card, neither resident nor on the host"),
        ("other kind", [], "full", "a layer type the program has no body for"),
        ("no head", [(sm, "head", None)], None, "the head not on the card in bf16"),
        ("no mixer", [(sm, "mixer", None)], None, "the final mixer not on the card in bf16"),
        ("routing", [(sm.cfg, "norm_topk_prob", False)], None, "an activation or a routing the kernels are not written for"),
        ("experts", [(mod, "_Experts", type("NotTheStore", (), {}))], None, "a mixture not served through the store"),
        ("ple unnamed", [], "no ple", "a PLE layer the config does not name"),
    ]  # fmt: skip
    for name, patch, layout, want in cases:
        with monkeypatch.context() as mp:
            for obj, attr, v in patch:
                mp.setattr(obj, attr, v)
            prog = mod.Qwen4Card(sm)
            if layout == "full":
                prog.types[0] = LayerKind.FULL
            elif layout == "no ple":
                prog.ple_all = []
            prog._layout(tuple(range(L)) if layout is None or isinstance(layout, str) else layout)
            got = prog.why_not(sm)
        assert got == want, f"{name}: {got!r}, not {want!r}"
    prog = mod.Qwen4Card(sm)
    prog._sm = lambda: None  # type: ignore[assignment]  # the engine let go
    assert not prog.ok(), "a program whose engine is gone ran"
    lin = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match=r"the program takes bf16 matrices on the card, got torch\.float32 cpu"):
        mod.Qwen4Card(sm)._merge("a float32 host matrix", [(lin, "weight")])
    t = torch.zeros(4, 8, dtype=torch.bfloat16, device=sm.dev).t()
    with pytest.raises(RuntimeError, match="a transposed view: the program takes a contiguous bf16 tensor on the card"):
        mod.Qwen4Card._bf(t, "a transposed view")
    # an engine with no scheduler (a stub's): the bytes are the program's own count alone
    with monkeypatch.context() as mp:
        mp.setattr(sm, "scheduler", None)
        prog = mod.Qwen4Card(sm)
        prog._grant("the test's rows", 64, "scratch", sm.dev)
        assert prog.held == {"the test's rows": 64}


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
