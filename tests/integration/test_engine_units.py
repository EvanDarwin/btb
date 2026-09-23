# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's own contracts, each one a bug that reached a run: the weight map an unsharded checkpoint has
to synthesise, the n-gram chains merged into one verification tree, the attention cache's buffer grown in
place, the placement `load()` plans from --device and --cpu-layers, and the exactness a speculative decode
owes the greedy one. test_unit.py holds the model-free checks; these need a fixture model, and the ones
marked so need a card. Everything here runs in seconds: the fixtures are a few hundred kilobytes."""

import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch

import btb
from btb.draft import NGramProposer, Spans
from btb.engine import StreamedTextModel
from btb.engine.cache import GrowLayer
from btb.engine.generate import _chains_tree
from btb.kinds import Json, TokenRows
from tests.helpers import (
    ROOT,
    fixture,
    forward_logits,
    host_model,
    layer_count,
    loaded_model,
    max_abs,
    need_cuda,
    need_mlx,
    rel_err,
    safetensors_state,
    speculation,
)

# a prompt whose own n-grams repeat, so the proposer holds several followers per key and the chains at the
# different orders really differ (a straight-line prompt proposes one chain and never builds a tree)
REPEATING = [[7, 11, 5, 9, 7, 11, 5, 3, 7, 11, 2, 9, 7, 11, 5, 9]]
# a prompt tiny_q35 answers with 38 distinct tokens over 48: a constant answer would accept every draft and
# the exactness check would pass without ever rejecting one
VARIED = [[167, 487, 79, 204, 335, 26, 39, 422, 276, 50, 189, 300, 31, 467, 261, 111]]


# --- the unsharded checkpoint's weight map (btb/engine/model.py) ---------------------------------------------


def test_an_unsharded_checkpoint_reads_its_weight_map_off_the_headers(tmp_path: Path) -> None:
    """A checkpoint of one `model.safetensors` carries no index.json, so the map of tensor name -> file is read
    off the safetensors headers. The header's `__metadata__` entry is not a tensor and must not enter the map;
    the map must name every tensor the sharded form has, and the model built from it must answer the same."""
    from safetensors.torch import save_file

    fx = fixture("tiny_qwen3")
    one = tmp_path / "unsharded"
    one.mkdir()
    merged = safetensors_state(fx)
    # metadata=... is what puts `__metadata__` in the header: without it the exclusion is never exercised
    save_file(merged, str(one / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(os.path.join(fx, "config.json"), one)
    assert not os.path.exists(one / "model.safetensors.index.json")
    with open(one / "model.safetensors", "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    assert "__metadata__" in header, "the fixture must carry the metadata entry the map has to skip"

    with torch.inference_mode():
        sharded = host_model(fx)
        try:
            single = host_model(str(one))
            try:
                assert set(single.weight_map.values()) == {"model.safetensors"}
                assert set(single.weight_map) == set(sharded.weight_map)
                assert "__metadata__" not in single.weight_map
                assert single.prefix == sharded.prefix and single.head_key == sharded.head_key
                # the map is not decoration: every read goes through it
                assert single.generate_greedy(REPEATING, 8) == sharded.generate_greedy(REPEATING, 8)
            finally:
                single.close()
        finally:
            sharded.close()


# --- the chains merged into one tree (btb/engine/generate.py, btb/draft.py) ---------------------------------


def _walk(guesses: Sequence[int], children: dict[int, list[int]], toks: Sequence[int]) -> int | None:
    """the node the chain `toks` reaches from the root, or None where the tree does not hold all of it"""
    node = 0
    for t in toks:
        nxt = [c for c in children.get(node, []) if guesses[c - 1] == t]
        if not nxt:
            return None
        node = nxt[0]
    return node


def _check_tree(
    guesses: Sequence[int],
    parents: Sequence[int],
    depth: Sequence[int],
    children: dict[int, list[int]],
    tags: Sequence[str],
) -> None:
    """the invariants the speculative loop reads the tree by: one parent above every node, depth counted from
    the root, `children` the exact inverse of `parents`, and no two siblings carrying the same token (they
    would be the same node)"""
    assert len(parents) == len(depth) == len(tags) == len(guesses) + 1
    assert parents[0] == -1 and depth[0] == 0 and tags[0] == "root"
    rebuilt: dict[int, list[int]] = {}
    for j in range(1, len(guesses) + 1):
        assert 0 <= parents[j] < j, "a node's parent is an earlier node"
        assert depth[j] == depth[parents[j]] + 1
        rebuilt.setdefault(parents[j], []).append(j)
    assert rebuilt == children
    for kids in children.values():
        toks = [guesses[c - 1] for c in kids]
        assert len(set(toks)) == len(toks), "siblings with the same token are one node"


def test_chains_tree_merges_shared_prefixes_and_keeps_every_chain() -> None:
    chains = [([5, 6, 7, 8], "a"), ([5, 6, 9], "b"), ([5, 6, 7, 8], "dup"), ([4], "c"), ([5, 3], "d")]
    guesses, parents, depth, children, tags = _chains_tree(chains, 14)
    _check_tree(guesses, parents, depth, children, tags)
    assert guesses == [5, 6, 7, 8, 9, 4, 3]
    assert parents == [-1, 0, 1, 2, 3, 2, 0, 1]
    assert depth == [0, 1, 2, 3, 4, 3, 1, 2]
    assert children == {0: [1, 6], 1: [2, 7], 2: [3, 5], 3: [4]}
    # a node's tag is the chain that first reached it; a repeat of an earlier chain adds nothing
    assert tags == ["root", "a", "a", "a", "a", "b", "c", "d"]
    for toks, _tag in chains:
        node = _walk(guesses, children, toks)
        assert node is not None and depth[node] == len(toks), f"{toks} does not replay"


def test_chains_tree_stops_at_the_budget() -> None:
    chains = [([5, 6, 7, 8], "a"), ([5, 6, 9], "b"), ([4], "c")]
    for budget in range(0, 8):
        guesses, parents, depth, children, tags = _chains_tree(chains, budget)
        _check_tree(guesses, parents, depth, children, tags)
        assert len(guesses) <= budget, "the budget caps the nodes beyond the root"
    guesses, parents, depth, children, tags = _chains_tree(chains, 0)
    assert guesses == [] and parents == [-1] and depth == [0] and children == {} and tags == ["root"]
    # the cap cuts a chain where it runs out, it does not drop the chain: the first three of chain "a" stand
    guesses, _p, _d, children, tags = _chains_tree(chains, 3)
    assert guesses == [5, 6, 7] and tags == ["root", "a", "a", "a"]
    assert _walk(guesses, children, [5, 6, 7]) == 3 and _walk(guesses, children, [4]) is None


def test_ngram_followers_rank_by_count_then_recency_and_cap() -> None:
    """The map for an n-gram holds each follower token with its count and the position after its latest
    occurrence; `_followers` ranks the most frequent first, the most recent among equals, at most `followers`."""
    p = NGramProposer([1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6], n_max=2, n_min=2, followers=3)
    assert p.maps[2][(1, 2)] == {3: [1, 2], 4: [1, 5], 5: [1, 8], 6: [1, 11]}
    fl = p._followers(2, (1, 2))
    assert fl == [(1, 11), (1, 8), (1, 5)], "the most recent first among equal counts, the oldest dropped at the cap"
    assert [p.corpus[pos] for _c, pos in fl] == [6, 5, 4]
    q = NGramProposer([1, 2, 3, 1, 2, 3], n_max=2, n_min=2, followers=4)
    assert q.maps[2][(1, 2)] == {3: [2, 5]}, "the same follower counts up and moves; it does not take a second slot"
    r = NGramProposer([1, 2, 3, 1, 2, 3, 1, 2, 4], n_max=2, n_min=2, followers=4)
    assert r._followers(2, (1, 2)) == [(2, 5), (1, 8)], "a follower seen twice outranks a newer one seen once"


def test_propose_chains_drops_a_chain_that_repeats_or_prefixes_another() -> None:
    """Every order proposes a continuation from the same tail; the orders that agree would verify the same
    tokens twice, so only the first (the longest match) is kept."""
    p = NGramProposer([1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 4, 5], n_max=4, n_min=2)
    chains = p.propose_chains(4)
    assert [c for c, _ in chains] == [[9, 9, 1, 2]], chains
    assert [t for _c, t in chains] == ["ngram4"], "the longest order names the chain that survives"
    # a sequence banked by `add_sequence` is offered before the corpus's own at the same order
    p.add_sequence([2, 3, 4, 5, 77, 78], "banked")
    chains = p.propose_chains(4)
    assert chains[0] == ([77, 78], "seq4"), chains
    assert ([9, 9, 1, 2], "ngram4") in chains, "a chain that differs is kept beside it"


# --- the attention cache grown in place (btb/engine/cache.py) -----------------------------------------------


def _kv(
    B: int, Hk: int, T: int, d: int, dtype: torch.dtype = torch.bfloat16, fill: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.full((B, Hk, T, d), float(fill), dtype=dtype)
    return k, k.clone()


def test_grow_layer_appends_rows_in_order() -> None:
    layer = GrowLayer()
    keys, _values = layer.update(*_kv(1, 2, 3, 8, fill=1.0))
    assert keys.shape == (1, 2, 3, 8) and layer.get_seq_length() == 3
    keys, values = layer.update(*_kv(1, 2, 2, 8, fill=2.0))
    assert keys.shape == (1, 2, 5, 8) and layer.get_seq_length() == 5
    assert keys[0, 0, :, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    assert values[0, 0, :, 0].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    # the rows the caller holds are the buffer's, not a copy of it
    assert keys.untyped_storage().data_ptr() == layer._buf[0].untyped_storage().data_ptr()


def test_grow_layer_reserves_the_cap_hint_and_not_the_floor() -> None:
    """`cap_hint` is prompt + max_new: the caller knows how far the sequence runs, so the buffer is that long
    and no longer. Without it every layer takes the 4096-position floor, which a batch of short decodes
    multiplies into an OOM."""
    layer = GrowLayer(cap_hint=40)
    layer.update(*_kv(1, 2, 3, 8, fill=1.0))
    assert layer._buf[0].shape[-2] == 40, "the hint is the capacity, not a floor of 4096"
    layer.update(*_kv(1, 2, 4, 8, fill=2.0))
    assert layer._buf[0].shape[-2] == 40, "a second append inside the hint reserves nothing more"
    assert layer.get_seq_length() == 7
    assert GrowLayer()._buf is None
    plain = GrowLayer()
    plain.update(*_kv(1, 2, 3, 8))
    assert plain._buf[0].shape[-2] == 4096, "no hint: the floor"


def test_grow_layer_does_not_compound_its_capacity_when_only_the_placement_changes() -> None:
    """The rows come back in another dtype (or from another device, or as another batch) with room to spare:
    the buffer is re-cut where they now live, and re-cutting it must not also grow it. It did - `have +
    max(4096, have // 8)` on every move - and a 0.6B model's cache compounded into gigabytes over one answer,
    4096 -> 8192 -> ... on a cache that never held more than a few hundred rows."""
    layer = GrowLayer()
    layer.update(*_kv(1, 2, 4, 8, torch.bfloat16, fill=1.0))
    had = layer._buf[0].shape[-2]
    layer.update(*_kv(1, 2, 2, 8, torch.float32, fill=2.0))
    assert layer.keys.dtype == torch.float32 and layer.get_seq_length() == 6
    assert layer.keys[0, 0, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0], "the rows survive the change"
    assert layer._buf[0].shape[-2] == had, f"capacity {had} -> {layer._buf[0].shape[-2]} on a dtype change"
    layer.update(*_kv(2, 2, 2, 8, torch.float32, fill=3.0))
    assert layer._buf[0].shape[-2] == had, f"capacity {had} -> {layer._buf[0].shape[-2]} on a batch change"
    # the capacity is kept only while the rows fit: past the buffer it grows as it always did
    layer.update(*_kv(2, 2, had, 8, torch.float32, fill=4.0))
    assert layer._buf[0].shape[-2] > had and layer.get_seq_length() == 8 + had


# --- the placement load() plans (btb/__init__.py) -----------------------------------------------------------


def test_load_on_the_cpu_takes_the_budget_from_the_checkpoints_drafting_head() -> None:
    """No card and no `mtp.*` weights: nothing draws a tree, so the budget is 0 and the n-gram proposer runs.
    The same checkpoint with a drafting head takes the budget and the head's proposer."""
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        assert sm.dev.type == "cpu"
        assert not any(k.startswith("mtp.") for k in sm.weight_map)
        assert sm.tree_budget == 0 and sm.proposer == "ngram"
        assert set(sm.host) == set(range(sm.L)) and not sm.resident and not sm.cold
    with loaded_model(fixture("tiny_q35"), device="cpu") as sm:
        assert any(k.startswith("mtp.") for k in sm.weight_map)
        assert sm.tree_budget == 16 and sm.proposer == "mtp_dyn"


def test_load_keeps_the_schedulers_floor_free() -> None:
    """the RAM the engine keeps free of its grants is the plan's host budget floor - a tenth of the RAM available
    at load, or the OS's own figure plus the growth the checkpoint's shape prices where that is more - and
    --ram-reserve names another, which the plan and the engine then share; the report carries the budget and
    reads the run's growth against it"""
    from btb.engine.scheduler import BatchScheduler
    from btb.sysinfo import os_memory_floor

    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        assert sm.plan is not None and sm.plan.budget is not None
        b = sm.plan.budget
        assert (
            sm.ram_reserve == b.floor == max(b.os_floor + b.growth, int(BatchScheduler.RAM_FLOOR_SHARE * b.available))
        )
        assert b.os_floor == os_memory_floor() and b.growth == BatchScheduler.growth_estimate(sm)
        assert b.footprint > 0 and b.available > 0
        rep = sm.report()
        assert rep["plan"]["budget"]["floor"] == b.floor
        assert rep["growth"]["estimate_gb"] == b.growth / 2**30 and rep["growth"]["working_set_gb"] > 0
    with loaded_model(fixture("tiny_qwen3"), device="cpu", ram_reserve_gb=1.5) as sm:
        assert sm.plan is not None and sm.plan.budget is not None
        assert sm.ram_reserve == int(1.5 * 2**30) == sm.plan.budget.floor


def test_the_ram_policy_sheds_a_warm_layer_to_the_ring_and_takes_it_back() -> None:
    """the host tier's counterpart of the VRAM policy, on the packed fixture: a layer shed to the ring streams
    from the store each pass and the tokens are the greedy loop's; regrown, it is on the store's mapped bytes
    again and the tokens still are"""
    from btb.engine.host import _HostLinear

    with loaded_model(fixture("tiny_qwen3-pack12"), device="cpu", v_max=0) as sm:
        assert sm.host and not sm.cold and sm.ram_watch
        before, _ = sm.generate(REPEATING, 12, speculate=False)
        i = sm.ram_shed("the test")
        assert i == max(sm.host) and sm.cold == {i} and sm.ram_state.shed == [i]
        assert sm.cold_ring.slots and i in sm.cold_ring.slot_of
        lins = [m for m in sm.host[i].modules() if isinstance(m, _HostLinear) and m.key]
        assert lins and all(m.packed is not None for m in lins)
        during, _ = sm.generate(REPEATING, 12, speculate=False)
        assert during == before, "a layer read from the drive each pass answers as it did from RAM"
        assert sm.ram_regrow() == i and not sm.cold and not sm.ram_state.shed and not sm.cold_ring.slots
        after, _ = sm.generate(REPEATING, 12, speculate=False)
        assert after == before


def test_the_12_bit_model_widens_bit_exact_with_torch_alone() -> None:
    """`btb.pack12.load_state_dict` on the pack12 fixture, no engine: every tensor the parent has comes back -
    the packed layers widened from the shards, the rest read through from the parent - bit for bit."""
    from btb import pack12

    pack, src = fixture("tiny_qwen3-pack12"), fixture("tiny_qwen3")
    ents = pack12.entries(pack)
    assert ents and all(e["lo"] == e["n"] and e["esc"] >= 0 for e in ents.values())
    got = pack12.load_state_dict(pack)
    ref = safetensors_state(src)
    assert set(got) == set(ref) and set(ents) <= set(ref)
    for k, t in ref.items():
        assert got[k].dtype == t.dtype and got[k].shape == t.shape, k
        assert torch.equal(got[k].contiguous().view(torch.uint8), t.contiguous().view(torch.uint8)), f"{k} differs"


def test_the_12_bit_model_answers_as_its_parent_with_and_without_the_native_kernels() -> None:
    """the exactness law across the format: the pack12 fixture and its parent emit the same greedy tokens on the
    CPU with the native kernels here, and on torch alone in a fresh interpreter (a loaded kernel library stays
    loaded for a process; without one the engine reads the bf16 through the parent the pack names)."""
    import subprocess

    src, pack = fixture("tiny_qwen3"), fixture("tiny_qwen3-pack12")
    with loaded_model(src, device="cpu", v_max=0) as a, loaded_model(pack, device="cpu", v_max=0) as b:
        assert b.pack is not None and b._packed
        assert a.generate(REPEATING, 12, speculate=False)[0] == b.generate(REPEATING, 12, speculate=False)[0]
    code = (
        "import json, sys, btb\n"
        "src, pack, ids = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])\n"
        "out = []\n"
        "for p in (src, pack):\n"
        "    with btb.load(p, device='cpu', native='', v_max=0) as m:\n"
        "        out.append(m.generate(ids, 12, speculate=False)[0])\n"
        "from btb.engine.native import Native\n"
        "assert Native.gemv is None and Native.gemv_p12 is None, 'the torch-alone arm loaded a kernel library'\n"
        "print(json.dumps(out))\n"
    )
    r = subprocess.run(
        # relative paths, as the command line passes them: the parent is found through the pack, not the cwd
        [sys.executable, "-c", code, os.path.relpath(src, ROOT), os.path.relpath(pack, ROOT), json.dumps(REPEATING)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    plain, packed = json.loads(r.stdout.strip().splitlines()[-1])
    assert plain == packed and len(packed) == 12


def test_load_places_the_layers_the_cpu_share_names() -> None:
    """--cpu-layers N with no --model and no --resident-last: the first N layers on the CPU kernels, the card
    keeping what the plan gives it of the rest. N = 0 is the whole model on the card when it fits."""
    dev = need_cuda()
    path = fixture("tiny_q35")
    with loaded_model(path, device=dev, cpu_layers=0) as sm:
        assert sm.dev.type == "cuda"
        assert len(sm.resident) == sm.L and not sm.host and not sm.cold
    with loaded_model(path, device=dev, cpu_layers=2) as sm:
        assert set(sm.host) == {0, 1}, "the first N layers are the CPU's"
        assert set(sm.resident) | set(sm.cold) == set(range(2, sm.L))
        assert not (set(sm.resident) & set(sm.host)), "a layer is in one place"


def test_load_gives_a_card_that_holds_every_layer_a_tree_without_a_drafting_head() -> None:
    """No `mtp.*` weights, but every layer on the card: a verify pass reads the weights once, so the n-gram
    proposer's chains are worth merging into a tree. The proposer stays the n-gram one - there is no head."""
    with loaded_model(fixture("tiny_qwen3"), device=need_cuda(), cpu_layers=0) as sm:
        assert not sm.host and len(sm.resident) == sm.L
        # 15 drafted rows and the root: the widest pass the card's GEMV serves at its 16-row cost
        assert sm.tree_budget == 15 and sm.proposer == "ngram"


# --- exactness: a speculative decode is the greedy one -------------------------------------------------------


def _exactly_greedy(sm: StreamedTextModel, prompt: TokenRows, n_new: int, spans: Spans = ()) -> tuple[list[int], Json]:
    """(the greedy answer, the speculative loop's census), failing where the two answers part"""
    speculation(sm, tree_min_prob=0.0, ngram_p=0.9, tree_read="step")
    greedy = sm.generate_greedy(prompt, n_new)
    spec, census = sm.generate_speculative(prompt, n_new, proposer="ngram", v_max=4, spans=spans)
    if spec != greedy:
        j = next(i for i in range(min(len(spec), len(greedy))) if spec[i] != greedy[i])
        pytest.fail(
            f"speculative left the greedy path at token {j}: greedy {greedy[max(0, j - 3) : j + 3]} "
            f"speculative {spec[max(0, j - 3) : j + 3]} (census {census})"
        )
    assert len(spec) == len(greedy)
    return greedy, census


def test_speculative_is_the_greedy_answer_on_the_host() -> None:
    """`bench` reports `identical N/M`: a draft that is not verified to the greedy token is a wrong answer,
    not a slow one. 48 tokens of an answer with real variety, so drafts are really rejected."""
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q35"))
        try:
            sm.tree_budget = 16
            greedy, census = _exactly_greedy(sm, VARIED, 48)
            assert len(set(greedy)) > 10, "a constant answer would accept every draft: the check would be empty"
            assert census["accepted"] > 0, "nothing was drafted: the loop under test never ran"
            assert census["proposed"] > census["accepted"], "no draft was rejected: the crop was never taken"
        finally:
            sm.close()


def test_speculative_is_the_sampled_answer_on_the_host() -> None:
    """Under a temperature the verify pass draws every node's token from the target's own distribution and
    follows the draft that matches: the answer is the sequential sampled loop's, token for token, under one
    seed (the noise is keyed by the cache row a token is decided at); another seed is another answer."""
    from btb.sampling import Sampling

    with torch.inference_mode():
        sm = host_model(fixture("tiny_q35"))
        try:
            sm.tree_budget = 16
            sm.tree_min_prob = 0.0
            sm.ngram_p = 0.9
            sm.tree_read = "step"
            sm.drafter_weights = None
            s5 = Sampling(temperature=0.9, top_p=0.95, seed=5)
            plain = sm.generate_greedy(VARIED, 48, sampling=s5)
            assert sm.generate_greedy(VARIED, 48) != plain, "the sample is not the greedy answer"
            assert sm.generate_greedy(VARIED, 48, sampling=Sampling(temperature=0.9, top_p=0.95, seed=6)) != plain
            assert sm.generate_greedy(VARIED, 48, sampling=s5) == plain, "a seed repeats its answer"
            # the fixture's logits are flat, so a draft the prompt suggests rarely matches a draw: the answer itself
            # banked as a span is a draft that matches at every node, its tail reversed one that is rejected
            for tag, span, rejected in (("answer", plain, False), ("wrong", plain[:8] + plain[8:][::-1], True)):
                spec, census = sm.generate_speculative(
                    VARIED, 48, proposer="ngram", v_max=4, spans=[(tag, span)], sampling=s5
                )
                assert spec == plain, f"{tag}: the sampled speculative answer left the sequential one: {spec[:12]}"
                assert census["seed"] == 5 and census["accepted"] > 0, census
                assert (census["proposed"] > census["accepted"]) == rejected, census
        finally:
            sm.close()


def test_the_sampled_drafting_tree_draws_the_sequential_distribution_on_the_host() -> None:
    """Under a temperature the drafting head draws its children from its own distribution and the verify pass
    accepts each against the target with the residual rule: exact, so over many seeds the speculative loop's
    first token is distributed as the sequential sampled loop's; and the drafts do get accepted."""
    from collections import Counter

    from btb.sampling import Sampling

    with torch.inference_mode():
        sm = host_model(fixture("tiny_q35"))
        try:
            sm.tree_budget = 16
            sm.tree_min_prob = 0.0
            sm.ngram_p = 0.0
            sm.tree_read = "step"
            n = 400
            seq: Counter[int] = Counter()
            spec: Counter[int] = Counter()
            accepted = proposed = 0
            for seed in range(n):
                s = Sampling(temperature=0.1, seed=seed)
                seq[int(sm.generate_greedy(VARIED, 4, sampling=s)[0])] += 1
                out, census = sm.generate_speculative(VARIED, 4, proposer="mtp_dyn", v_max=4, sampling=s)
                spec[int(out[0])] += 1
                accepted += census["accepted"]
                proposed += census["proposed"]
            assert proposed > 0 and accepted > 0, (accepted, proposed)
            for key in {k for k, c in seq.items() if c / n > 0.05} | {k for k, c in spec.items() if c / n > 0.05}:
                a, b = seq[key] / n, spec[key] / n
                assert abs(a - b) < 0.06, f"{key}: sequential {a:.3f} vs speculative {b:.3f}"
            assert sum(c for c in spec.values() if c / n > 0.05) > 0, (
                "the fixture's distribution has no mass to compare"
            )
        finally:
            sm.close()


def test_speculative_is_the_greedy_answer_on_the_card_with_the_ngram_tree() -> None:
    """A card holding every layer verifies the n-gram chains as one tree (several rows for the cost of one):
    the merge, the tree's positions and the crop of the rejected path all run here and nowhere on the host."""
    dev = need_cuda()
    with loaded_model(fixture("tiny_qwen3"), device=dev, cpu_layers=0) as sm:
        assert sm.tree_budget > 0 and not sm.host, "the tree only draws when the card holds every layer"
        with torch.inference_mode():
            greedy, census = _exactly_greedy(sm, REPEATING, 48)
            assert "ngram_tree" in census["by_source"]["drafted"], f"no tree was built: {census['by_source']}"
            assert census["proposed"] > census["accepted"], "every node was accepted: the crop never ran"
            # a banked sequence that disagrees with the answer: the drafter trusts it, the verify must not
            bad = list(greedy)
            for i in (5, 17, 31):
                bad[i] = (bad[i] + 1) % int(sm.cfg.vocab_size)
            spec, _c = _exactly_greedy(sm, REPEATING, 48, spans=[("misleading", bad)])
            assert spec == greedy, "a misleading draft changed the answer"


def test_the_drafter_answers_the_greedy_loop_over_a_head_slice_on_the_card_and_on_the_host() -> None:
    """With the head cut to `draft_vocab` rows, the speculative tokens must equal the greedy loop's on the card at
    8 and 16 bits and on the host. The head must be the slice's size, and at 8 bits the bf16 fc must be released."""
    from btb.engine.drafter import _Int8Linear
    from btb.engine.host import _HostLinear
    from btb.engine.native import Native

    path = fixture("tiny_q35")
    arms = [("cuda", 8), ("cuda", 16)] if torch.cuda.is_available() else []
    if Native.gemv is not None:
        arms.append(("cpu", 16))
    if not arms:
        pytest.skip("neither a CUDA device nor the native kernels")
    for dev, bits in arms:
        if dev == "cuda":
            btb.CUDA = True  # a cpu `load()` earlier clears the package's flag; torch still holds the device
        with loaded_model(path, device=dev, draft_vocab=256, draft_bits=bits) as sm:
            assert sm.proposer == "mtp_dyn" and sm.draft_bits == bits and sm.draft_vocab == 256
            speculation(sm, tree_min_prob=0.0, tree_read="step")
            with torch.inference_mode():
                greedy = sm.generate_greedy(VARIED, 48)
                spec, census = sm.generate_speculative(VARIED, 48, proposer="mtp_dyn", v_max=4)
            assert spec == greedy, f"{dev} at {bits} bits: the speculative tokens left the greedy path ({census})"
            assert census["forwards"] > 0
            dr = sm.aj
            assert dr is not None
            H = int(sm.cfg.hidden_size)
            if dev == "cpu":
                head = dr._host_head()
                assert isinstance(head, _HostLinear) and tuple(head.weight.shape) == (256, H)
                assert getattr(dr, "_head_t", None) is None, "the host drafter built a torch head"
                continue
            head = dr._head_t
            assert head is not None, "the drafter never made its head"
            if bits < 16:
                assert isinstance(head, _Int8Linear) and tuple(head.w8.shape) == (256, H)
                assert dr.fc is None and dr.fc8 is not None, "the bf16 fc is still held beside its int8 copy"
                assert any(isinstance(m, _Int8Linear) for m in dr.layer.modules()), "the layer was not packed"
            else:
                assert not isinstance(head, _Int8Linear) and tuple(head.shape) == (256, H)
                assert dr.fc is not None and getattr(dr, "fc8", None) is None


# --- the two tiers answer alike ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "tiny_qwen3",
        "tiny_phi3",
        # tiny_q35 once failed here from position 0 at every placement (max |dlogit| 0.79): the fused RMSNorm
        # ignored the class's zero-centred (1 + weight) scale. It is the regression net for that fix.
        "tiny_q35",
    ],
)
def test_the_card_and_the_host_answer_the_same_prompt_alike(name: str) -> None:
    """Both tiers compute in float32 over the same bf16 weights: the sums reassociate, nothing else. The
    receipts suite certifies the host and the MLX tier; this is the same claim for the card."""
    dev = need_cuda()
    path = fixture(name)
    with torch.inference_mode():
        host = host_model(path)
        try:
            ref = forward_logits(host, REPEATING, host.new_cache())[0, -1].float()
        finally:
            host.close()
        card = host_model(path, device=dev, cpu_layers=(), resident_layers=range(layer_count(path)))
        try:
            got = forward_logits(card, REPEATING, cache=card.new_cache())[0, -1].float().cpu()
        finally:
            card.close()
    assert max_abs(got, ref) < 1e-4, f"{name}: the card and the host disagree"


def test_spec_budget_without_a_cost_curve_is_the_wider_of_the_tree_and_the_chain() -> None:
    """The speculative loop caps a pass's rows at the budget; without the card's cost curve (the CPU and MLX
    tiers) the budget must be the configured width, the chain's included: a tree budget of 0 with v_max 4 once
    returned 1 and switched speculation off on every model without a drafting head."""
    from btb.engine.cuda import _CudaMixin

    class _Engine(_CudaMixin):
        def __init__(self, tree_budget: int) -> None:
            self.tree_budget = tree_budget

    assert _Engine(0)._spec_budget(1.0, 0, v_max=4) == 5, "a chain of four drafts and the root"
    assert _Engine(16)._spec_budget(1.0, 0, v_max=4) == 17, "the tree's rows when wider"
    assert _Engine(0)._spec_budget(1.0, 0) == 1, "nothing configured: one row"
    e = _Engine(15)
    e._card_cost = {1: 1.0, 2: 1.1, 4: 1.2, 8: 1.6, 16: 2.0}
    assert e._spec_budget(1.0, 1, v_max=4) == 4, "with a curve: the widest pass within a quarter over a step"
    assert e._spec_budget(2.0, 1, v_max=4) == 16, "and wider while the passes yield the tokens to pay for it"


def test_int8_linear_packs_in_memory_and_multiplies() -> None:
    """the drafter's int8 linear on a torch tier: one scale a row, the product within the rounding of a
    row-scaled int8 weight, through torch's packed matmul where the device has it"""
    from btb.engine.drafter import _Int8Linear

    torch.manual_seed(0)
    w = torch.randn(48, 64, dtype=torch.bfloat16)
    lin = _Int8Linear(w)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    y = lin(x)
    ref = x.float() @ w.float().T
    assert y.shape == (3, 48)
    err = rel_err(y.float(), ref)
    assert err < 0.05, f"the int8 product is off by {err:.3f} of the range"
    assert lin.w8.dtype == torch.int8 and lin.scale.shape == (48,)
    assert lin.packed_mm == torch._C._dispatch_has_kernel_for_dispatch_key("aten::_weight_int8pack_mm", "CPU")


def test_hybrid_session_continues_past_16_new_rows_on_mlx() -> None:
    """a session's next turn on the hybrid brings more rows than the node step takes (16): the continuation runs
    as a chunked one from the stored states and answers as a fresh run of the whole prompt does. A tree of
    thousands of nodes was once built here - one state checkpoint per row per layer - until the kernel refused it."""
    need_mlx()
    from btb.session import Session

    first = REPEATING[0]
    second = first + [3, 5, 2, 9, 11, 7, 5, 3, 2, 9, 11, 7, 5, 3, 7, 11, 5, 9, 2, 3, 9, 7, 11, 5, 2, 3, 9, 7]
    assert len(second) - len(first) > 16
    with loaded_model(fixture("tiny_q35"), device="mlx", v_max=0) as sm:
        s = Session()
        sm.generate(first, 4, session=s)
        with_session = sm.generate(second, 8, session=s)[0]
        fresh = sm.generate(second, 8)[0]
    assert with_session == fresh


def test_hybrid_prefill_chunks_with_the_drafter_hook_on_mlx() -> None:
    """a prompt longer than the prefill chunk on the hybrid, whose drafter hooks the last layer: the chunks' rows
    reach the hook joined, and the speculative loop answers as the unchunked prefill did"""
    need_mlx()
    prompt = REPEATING[0] * 4
    with loaded_model(fixture("tiny_q35"), device="mlx") as sm:
        whole = sm.generate(prompt, 12)[0]
        sm.prefill_chunk = 24
        chunked = sm.generate(prompt, 12)[0]
    assert chunked == whole


def test_mlx_decodes_run_on_one_worker_thread_whichever_thread_asks() -> None:
    """MLX keeps a lazy array's stream per thread: a model's decodes, asked from any thread (a server's request
    threads, a stream's own), run on the model's one worker thread, and close() from the main thread retires it"""
    need_mlx()
    import threading

    seen: list[threading.Thread] = []
    sm = btb.load(fixture("tiny_qwen3"), device="mlx", v_max=0)
    try:
        outs = []
        for _ in range(2):
            t = threading.Thread(
                target=lambda: outs.append(
                    sm.generate(REPEATING[0], 4, on_token=lambda _t: seen.append(threading.current_thread()))[0]
                )
            )
            t.start()
            t.join()
        outs.append(sm.generate(REPEATING[0], 4, on_token=lambda _t: seen.append(threading.current_thread()))[0])
        assert outs[0] == outs[1] == outs[2]
        workers = set(seen)
        assert len(workers) == 1, workers
        assert next(iter(workers)) is not threading.main_thread() and next(iter(workers)).name.startswith("btb-mlx")
    finally:
        sm.close()
    assert sm._worker is None


def test_native_isa_is_a_tier_the_cert_matrix_knows() -> None:
    """the tier the library reports (what bench.yml banks a baseline under) is one of the `Isa` variants
    tests/cert/native_ops reads from the crate, so the two never disagree on a name"""
    from btb.engine.native import isa, native_path
    from tests.cert.native_ops import ISA_TIERS

    if native_path() is None:
        pytest.skip("no native library")
    assert isa() in ISA_TIERS, (isa(), ISA_TIERS)


@pytest.mark.timing
def test_native_open_names_the_os_error_when_the_file_limit_is_hit(tmp_path: Path) -> None:
    """a reader's handle that fails to open says why: the crash was a bare 'btb_open returned -8' at the 256th
    descriptor of a 15-shard model"""
    if sys.platform == "win32":
        pytest.skip("the file-limit test uses the POSIX resource module")
    import errno
    import resource

    from btb.engine.native import Native, NativeError, native_path

    dll = native_path()
    if dll is None:
        pytest.skip("no native library")
    Native.load_gemv(dll)
    if Native.open is None:
        pytest.skip("the native library has no kept-handle reader")
    f = tmp_path / "shard.bin"
    f.write_bytes(b"\0" * 8192)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    hs: list[int] = []
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (96, hard))
        with pytest.raises(NativeError) as e:
            for _ in range(120):
                hs.append(Native.open(f))
        assert (e.value.call, e.value.rc, e.value.errno) == ("btb_open", -8, errno.EMFILE)
    finally:
        for h in hs:
            Native.close(h)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


# --- a checkpoint's float precision against the kernels' (tiers._held, native._refuse) -----------------------


def test_an_fp32_checkpoint_is_held_as_bf16_and_read_whole_where_widened() -> None:
    """an fp32 checkpoint's weights reach the bf16 readers as bf16 (the direct readers take them from memory,
    cast), while a caller widening to float32 itself reads them at their own precision; a bf16 checkpoint is
    read straight off the drive as before"""
    f32, bf16 = host_model(fixture("tiny_qwen3-f32")), host_model(fixture("tiny_qwen3"))
    try:
        k = next(k for k in f32.weight_map if k.endswith("q_proj.weight"))
        assert f32.held_cast and not bf16.held_cast
        assert f32._get(k).dtype == torch.bfloat16 and f32._get(k, stored=True).dtype == torch.float32
        assert torch.equal(f32._get(k), bf16._get(k))
        assert f32._span(k) == (None, 0, f32._get(k).numel() * 2)
        path, _off, nb = bf16._span(k)
        assert path is not None and nb == bf16._get(k).numel() * 2
    finally:
        f32.close()
        bf16.close()


def test_every_native_kernel_refuses_a_dtype_it_was_not_built_for() -> None:
    """a kernel reads its tensors through raw pointers: an fp32 weight handed to the bf16 gemv was read as bf16
    pairs and decoded garbage without a word. Every fixed-type binding now refuses before the call, naming the
    argument, and writes nothing"""
    from btb.engine.native import Native, NativeDtypeError, native_path
    from btb.mxfp4 import MxWeight, ggml_bytes, matrix_bytes

    dll = native_path()
    if dll is None:
        pytest.skip("no native library")
    Native.load_gemv(dll)
    r, c = 4, 32
    bf, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    w, x = torch.zeros(r, c, dtype=bf), torch.zeros(1, c, dtype=f32)
    nb, ns = matrix_bytes(r, c)
    mx = MxWeight(torch.zeros(nb, dtype=u8), torch.zeros(ns, dtype=u8), r, c)
    mx_bad = MxWeight(torch.zeros(nb, dtype=torch.int8), torch.zeros(ns, dtype=u8), r, c)
    gg = MxWeight.from_ggml(torch.zeros(ggml_bytes(r, c), dtype=u8), r, c)
    gg_bad = MxWeight.from_ggml(torch.zeros(ggml_bytes(r, c), dtype=torch.int8), r, c)
    p12 = (torch.zeros(r * c, dtype=u8), torch.zeros(r * c // 2, dtype=u8), torch.zeros(16, dtype=u8))
    no_esc = (torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=u8), 0)
    q, kv = torch.zeros(2, 16, dtype=f32), torch.zeros(1, 3, 16, dtype=bf)
    hk, hv, dk, dv, cd, ks = 1, 1, 4, 4, 12, 4

    def delta(**bad: torch.Tensor) -> Callable[[torch.Tensor], None]:
        t = {
            "mixed": torch.zeros(cd, dtype=f32),
            "conv_state": torch.zeros(cd, ks, dtype=f32),
            "conv_w": torch.zeros(cd, ks, dtype=f32),
            "conv_b": torch.zeros(cd, dtype=f32),
            "z": torch.zeros(hv * dv, dtype=f32),
            "a": torch.zeros(hv, dtype=f32),
            "b": torch.zeros(hv, dtype=f32),
            "a_log": torch.zeros(hv, dtype=f32),
            "dt_bias": torch.zeros(hv, dtype=f32),
            "state": torch.zeros(hv, dk, dv, dtype=f32),
            "norm_w": torch.zeros(dv, dtype=f32),
            **bad,
        }
        return lambda y: Native.delta_step(
            t["mixed"], t["conv_state"], t["conv_w"], t["conv_b"], t["z"], t["a"], t["b"], t["a_log"],
            t["dt_bias"], t["state"], hk, hv, dk, dv, t["norm_w"], 1e-6, y,
        )  # fmt: skip

    # (binding, call, argument, the call with that one argument wrong, the dtype its `y` must be written in)
    cases: list[tuple[str, str, str, Callable[[torch.Tensor], None], torch.dtype]] = [
        ("gemv", "btb_gemv_bf16_rows", "w", lambda y: Native.gemv(w.float(), x, y), f32),
        ("gemv", "btb_gemv_bf16_rows", "w", lambda y: Native.gemv(w.half(), x, y), f32),
        ("gemv", "btb_gemv_bf16_rows", "x", lambda y: Native.gemv(w, x.bfloat16(), y), f32),
        ("gemv", "btb_gemv_bf16_rows", "y", lambda y: Native.gemv(w, x, y), torch.float64),
        ("gemv_group", "btb_gemv_bf16_group", "w", lambda y: Native.gemv_group([w, w.float()], [x, x], [y, y]), f32),
        ("gemv_p12", "btb_gemv_p12_rows", "x", lambda y: Native.gemv_p12(*p12, *no_esc, r, c, x.half(), y), f32),
        (
            "gemv_p12",
            "btb_gemv_p12_rows",
            "lo",
            lambda y: Native.gemv_p12(p12[0].to(torch.int8), *p12[1:], *no_esc, r, c, x, y),
            f32,
        ),
        ("gemv_mx4", "btb_gemv_mxfp4_rows", "blocks", lambda y: Native.gemv_mx4(mx_bad, x, y), f32),
        (
            "gemv_mx4_group",
            "btb_gemv_mxfp4_group",
            "blocks",
            lambda y: Native.gemv_mx4_group([mx, mx_bad], [x, x], [y, y]),
            f32,
        ),
        ("gemv_mx4_ggml", "btb_gemv_mxfp4_ggml_rows", "blocks", lambda y: Native.gemv_mx4_ggml(gg_bad, x, y), f32),
        ("gemv_mx4_ggml", "btb_gemv_mxfp4_ggml_rows", "x", lambda y: Native.gemv_mx4_ggml(gg, x.half(), y), f32),
        (
            "gemv_mx4_ggml_group",
            "btb_gemv_mxfp4_ggml_group",
            "blocks",
            lambda y: Native.gemv_mx4_ggml_group([gg, gg_bad], [x, x], [y, y]),
            f32,
        ),
        ("attn_decode", "btb_attn_decode", "k", lambda y: Native.attn_decode(q, kv.half(), kv, 1.0, y), f32),
        ("attn_decode", "btb_attn_decode", "v", lambda y: Native.attn_decode(q, kv, kv.float(), 1.0, y), f32),
        ("attn_decode", "btb_attn_decode", "q", lambda y: Native.attn_decode(q.bfloat16(), kv, kv, 1.0, y), f32),
        ("delta_step", "btb_delta_step", "norm_w", delta(norm_w=torch.zeros(dv, dtype=bf)), f32),
        ("delta_step", "btb_delta_step", "conv_b", delta(conv_b=torch.zeros(cd, dtype=bf)), f32),
    ]
    missing = sorted({b for b, *_ in cases if getattr(Native, b) is None})
    assert not missing, f"the library built in this tree binds every kernel; unbound: {missing}"
    for binding, call, arg, run, ydt in cases:
        y = torch.full((2, 16) if binding == "attn_decode" else (1, r), 7.0, dtype=ydt)
        if binding == "delta_step":
            y = torch.full((hv * dv,), 7.0, dtype=f32)
        with pytest.raises(NativeDtypeError) as e:
            run(y)
        assert (e.value.call, e.value.arg) == (call, arg), (binding, arg, str(e.value))
        assert bool((y == 7.0).all()), f"{call} wrote its output before refusing {arg}"


@pytest.mark.parametrize("device", ["cpu", "mlx"])
@pytest.mark.parametrize("name", ["tiny_qwen3-f16", "tiny_qwen3-f32", "gguf/tiny_qwen3-q8_0.gguf"])
def test_a_layer_shed_to_the_drive_off_a_weight_not_stored_as_bf16_answers_as_from_ram(name: str, device: str) -> None:
    """the cold ring over weights it cannot read off the drive as bf16 - a cast twin's floats, a GGUF's Q8_0
    blocks - takes them from `_get` each pass ("mem"). `_bind_cold` read such a tensor's key as a 12-bit record
    and raised when a busy machine shed a layer mid-decode; shed under the decode's inference mode, the layer
    now answers as it did from RAM, and regrown it still does"""
    path = fixture(name)
    if device == "mlx":
        need_mlx()
    with loaded_model(path, device=device, v_max=0) as sm:
        before, _ = sm.generate(REPEATING, 12, speculate=False)
        with torch.inference_mode():
            i = sm.ram_shed("the test")
        assert i is not None and sm.cold == {i}
        assert {it[5] for it in sm.cold_ring.recipe[i]} == {"mem"}, "every linear of the shed layer is read cast"
        during, _ = sm.generate(REPEATING, 12, speculate=False)
        assert during == before, "a layer read through the ring each pass answers as it did from RAM"
        assert sm.ram_regrow() == i and not sm.cold
        after, _ = sm.generate(REPEATING, 12, speculate=False)
        assert after == before
