# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The mixture-of-experts paths on the host that the plain decode over a bf16 checkpoint leaves alone, over tiny_q4
on the CPU: its precision twins (float32, float16, FP8, the 12-bit store) and llama.cpp's MXFP4 GGUF decoding through
the expert store - plainly, speculatively (the n-gram proposer, the MTP drafter) and from the checkpoint's tables -
as their parent, or as themselves where their numbers are their own; the drafting head read from a file in the
checkpoint's place and run on a torch tier, its trees sampled; the verify pass priced by a drive's reads; ragged rows
in one batch; and the refusals and odd layouts of the layers' host modules. tiny_q4 is loaded once for the file with
a store below its experts (every pass reads some of them again), each twin once."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.host import _HostLinear
from btb.engine.native import Native
from btb.engine.spec_cost import SpecCost
from tests.helpers import (
    GGUF_FIXTURES,
    NO_LOG,
    fixture,
    forward_logits,
    host_model,
    loaded_model,
    speculation,
)

# two prompts past the indexer's budget (2 blocks of 4), the second far past it
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61],
    "long": ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70],
}
NEW = 12
# a store of some twenty slots: a one-row pass's eighteen experts fit, the model's seventy-two do not
SMALL_GB = 0.0006
# the probe of a drive a missed expert costs ~10 ms on, and one it costs seconds on
DRIVE = {"big_mb": 6.25, "single_ms": 40.0, "fixed_ms": 5.0, "ahead": 1, "measured": True}
SLOW = {"big_mb": 6.25, "single_ms": 40.0, "fixed_ms": 2000.0, "ahead": 1, "measured": True}


@pytest.fixture(scope="module")
def q4() -> Iterator[StreamedTextModel]:
    """tiny_q4 on the host over a store that seats a pass's experts and not the model's"""
    StreamedTextModel.register_attention()
    sm = host_model(fixture("tiny_q4"), expert_cache_gb=SMALL_GB)
    try:
        yield sm
    finally:
        sm.close()


def _plain(sm: StreamedTextModel) -> dict[str, list[int]]:
    return {name: [int(t) for t in sm.generate_greedy(p, NEW)] for name, p in PROMPTS.items()}


def _spans(prompt: list[int], plain: list[int]) -> list[tuple[str, list[int]]]:
    """spans for the n-gram proposer to draft from (a random tiny model's tokens seldom repeat the prompt's): the
    plain decode's own tokens, and a twin that leaves them after two, whose drafts are refused"""
    right = [prompt[-1], *plain]
    return [("plain", right), ("astray", [*right[:3], *(t + 1 for t in right[3:])])]


def _speculates_as_plain(sm: StreamedTextModel, plain: dict[str, list[int]], what: str) -> None:
    """every prompt's speculative decode, the MTP drafter's tree beside the n-gram chains and the chains alone, is
    the plain one"""
    for name, prompt in PROMPTS.items():
        for proposer in ("mtp_dyn", "ngram"):
            speculation(sm, tree_budget=16, tree_min_prob=0.0, ngram_p=0.9, v_max=4, price=False)
            toks, census = sm.generate_speculative(
                prompt, NEW, proposer=proposer, v_max=4, spans=_spans(prompt, plain[name])
            )
            assert toks == plain[name], f"{what}, {name}, {proposer}: {toks} != {plain[name]}"
            assert census["proposed"] > 0, f"{what}, {name}, {proposer}: nothing was drafted"


def _from_tables(sm: StreamedTextModel, plain: dict[str, list[int]], what: str) -> None:
    """the store set aside: every call reads the checkpoint's expert tables whole, and decodes as the store did"""
    store, sm.expert_store = sm.expert_store, None
    try:
        got = _plain(sm)
    finally:
        sm.expert_store = store
    assert got == plain, f"{what}: the checkpoint's tables decode otherwise than the store"


def test_a_store_below_the_models_experts_decodes_every_twin_as_its_parent(q4: StreamedTextModel) -> None:
    """tiny_q4's float32, float16 and 12-bit twins decode as tiny_q4 through the expert store (a float expert read
    as stored and rewritten as the bf16 it holds), its FP8 twin as itself (its experts multiplied as stored, its
    drafting head's matrices too); every twin's speculative decode is its plain one, and so is its decode from the
    checkpoint's tables. llama.cpp's MXFP4 GGUF of tiny_q4 decodes from its tables as through its store."""
    with torch.inference_mode():
        store = q4.expert_store
        assert store is not None
        want = _plain(q4)
        assert store.n_slots < q4.L * q4.n_experts, "the store holds every expert: nothing is read again"
        _from_tables(q4, want, "tiny_q4")
        for twin, dt in (
            ("tiny_q4-f32", torch.float32),
            ("tiny_q4-f16", torch.float16),
            ("tiny_q4-pack12", torch.bfloat16),
            ("tiny_q4-f8_e4m3", None),
        ):
            sm = host_model(fixture(twin), packed=twin.endswith("-pack12"))
            try:
                st = sm.expert_store
                assert st is not None
                plain = _plain(sm)
                if dt is None:
                    assert st.f8, f"{twin}: the experts are not read as FP8"
                else:
                    assert st.dt == dt, f"{twin}: the store holds {st.dt}"
                    assert plain == want, f"{twin} decodes otherwise than tiny_q4: {plain} != {want}"
                _speculates_as_plain(sm, plain, twin)
                _from_tables(sm, plain, twin)
                if twin.endswith("-pack12"):
                    # a host layer made again on the packed engine (a shed layer regrown) multiplies the 12-bit store
                    def packed(i: int, sm: StreamedTextModel = sm) -> int:
                        return sum(1 for m in sm.host[i].modules() if isinstance(m, _HostLinear) and m.packed)

                    n = packed(1)
                    sm.host[1] = sm._make_host_layer(1)
                    assert packed(1) == n > 0, f"{twin}: the regrown layer holds {packed(1)} packed linears, not {n}"
                    assert _plain(sm) == plain, f"{twin}: a regrown layer decodes otherwise"
            finally:
                sm.close()
        path = os.path.join(GGUF_FIXTURES, "tiny_q4-mxfp4.gguf")
        if not os.path.exists(path):
            pytest.skip("tiny_q4-mxfp4.gguf is not here (tests/make_fixtures.py)")
        with loaded_model(path, device="cpu", v_max=0) as sm:
            st = sm.expert_store
            assert st is not None and st.ggml
            _from_tables(sm, _plain(sm), "tiny_q4-mxfp4.gguf")


def test_the_drafting_head_reads_a_file_in_its_place_and_runs_on_a_torch_tier(
    q4: StreamedTextModel, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """a drafting head read from a weights file (`--draft-model` a file): its tensors are the file's, the rest the
    checkpoint's, and the speculative decode is the plain one whatever it drafts. Built where the host's kernels are
    not (a torch tier), the head is torch's modules in bf16 - at 8 bits its linears and its head's first rows packed
    to int8 - and still verifies to the plain tokens."""
    from safetensors.torch import load_file

    from btb.engine.drafter import _Int8Linear
    from btb.engine.families.qwen4.drafter import Qwen4Drafter

    prompt = PROMPTS["long"]
    with pytest.raises(ValueError, match="btb does not train Qwen4's drafting head"):
        Qwen4Drafter(q4, train=True)
    with torch.inference_mode():
        plain = [int(t) for t in q4.generate_greedy(prompt, NEW)]
        head = load_file(os.path.join(fixture("tiny_q4"), "model-mtp.safetensors"))
        src = {k: v for k, v in head.items() if q4.fam.dense_key(k)}
        src["mtp.fc_embedding.weight"] = torch.zeros_like(src["mtp.fc_embedding.weight"])
        path = tmp_path / "head.pt"
        torch.save(src, path)
        lines: list[str] = []
        speculation(q4, tree_budget=16, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
        q4.aj, q4.drafter_weights, q4.log = None, str(path), lambda *a, **k: lines.append(" ".join(map(str, a)))
        try:
            dr = q4.mtp_drafter()
            assert isinstance(dr, Qwen4Drafter) and dr.host
            placed = dr.named_tensors()
            assert not bool(placed["mtp.fc_embedding.weight"].any()), "the file's zeroed matrix is not the drafter's"
            assert torch.equal(placed["mtp.fc_hidden.weight"].float(), src["mtp.fc_hidden.weight"].float())
            assert any(f"drafter weights <- {path} ({len(src)} tensors)" in x for x in lines), lines
            toks, census = q4.generate_speculative(prompt, NEW, proposer="mtp_dyn", v_max=4)
            assert toks == plain and census["proposed"] > 0, (toks, plain, census)
        finally:
            q4.aj, q4.drafter_weights, q4.log = None, None, NO_LOG
        # a torch tier: the host's gemv absent, the drafter's modules torch's own
        monkeypatch.setattr(Native, "gemv", None)
        plain = [int(t) for t in q4.generate_greedy(prompt, NEW)]
        try:
            for bits in (16, 8):
                speculation(q4, tree_budget=16, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
                q4.aj, q4.draft_bits, q4.draft_vocab = None, bits, 256
                toks, census = q4.generate_speculative(prompt, NEW, proposer="mtp_dyn", v_max=4)
                assert toks == plain, f"{bits} bits: {toks} != {plain}"
                assert census["mtp_steps"] > 0 and census["proposed"] > 0, census
                tdr = q4.aj
                assert isinstance(tdr, Qwen4Drafter) and not tdr.host and tdr.cd == torch.bfloat16
                packed = [m for m in tdr.layer.modules() if isinstance(m, _Int8Linear)]
                if bits == 8:
                    assert packed, "no linear of the drafter was packed to 8 bits"
                    assert isinstance(tdr._head_t, _Int8Linear) and tdr._head_t.w8.shape[0] == 256
                else:
                    assert not packed and isinstance(tdr._head_t, torch.Tensor) and q4.head is not None
                    assert tdr._head_t.shape[0] == 256 and tdr._head_t.data_ptr() == q4.head.weight.data_ptr()
        finally:
            q4.aj, q4.draft_bits, q4.draft_vocab = None, 16, 0


def test_a_sampled_tree_of_the_drafting_head_verifies_against_the_targets_draws(q4: StreamedTextModel) -> None:
    """Under a temperature the drafting head draws each node's children from its own distribution, the tree grown
    past its root's children, and the verify pass accepts each against the target's draw. Where the sampling leaves
    one token a position (top_k 1, the draw a point mass) the speculative answer is the sequential one, which is the
    greedy one; under a plain temperature a seed repeats its answer, its first token the sequential loop's (drawn
    off the prefill under the same key), and drafts are accepted. (A random tiny model's logits are too flat for its
    answers to be compared as distributions: tests/integration/test_engine_units.py does that on tiny_q35, through
    the same loop.) A tree whose next depth carries too little probability is not stepped; a tree of no nodes, or a
    fan that keeps only the children next to the best, drafts accordingly."""
    from btb.sampling import Sampling

    prompt = PROMPTS["short"]
    with torch.inference_mode():
        speculation(q4, tree_budget=16, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
        greedy = [int(t) for t in q4.generate_greedy(prompt, NEW)]
        for seed in range(4):
            point = Sampling(temperature=0.7, top_k=1, seed=seed)
            seq = [int(t) for t in q4.generate_greedy(prompt, NEW, sampling=point)]
            out, census = q4.generate_speculative(prompt, NEW, proposer="mtp_dyn", v_max=4, sampling=point)
            assert seq == greedy and out == seq, f"seed {seed}: {out} / {seq} / {greedy}"
            assert census["proposed"] > 0
        accepted = proposed = 0
        for seed in range(12):
            s = Sampling(temperature=0.3, seed=seed)
            first = int(q4.generate_greedy(prompt, 4, sampling=s)[0])
            out, census = q4.generate_speculative(prompt, 4, proposer="mtp_dyn", v_max=4, sampling=s)
            again, _ = q4.generate_speculative(prompt, 4, proposer="mtp_dyn", v_max=4, sampling=s)
            assert out == again and int(out[0]) == first, f"seed {seed}: {out} / {again} / {first}"
            assert census["seed"] == seed
            accepted += census["accepted"]
            proposed += census["proposed"]
        assert proposed > 0 and accepted > 0, (accepted, proposed)
        # a step that must carry half the probability mass: the tree stops at its root's children
        q4.tree_step_mass = 0.5
        try:
            plain = [int(t) for t in q4.generate_greedy(prompt, NEW)]
            toks, census = q4.generate_speculative(prompt, NEW, proposer="mtp_dyn", v_max=4)
            assert toks == plain and census["proposed"] > 0
        finally:
            q4.tree_step_mass = 0.0
        # the drafter's tree over a fresh cache: a child far below its best sibling is not drafted (the fan's ratio)
        dr = q4.mtp_drafter()
        h = torch.randn(1, 1, int(q4.cfg.hidden_size) * int(q4.fam.streams), generator=torch.Generator().manual_seed(3))
        assert dr.au([prompt[-1]], h, 0, 0, with_tags=True) == ([], [], [], []), "a tree of no nodes"
        roots = {}
        for fan in (0.05, 0.999):
            dr.reset()
            _toks, parents, _depth = dr.au([prompt[-1]], h, 0, 8, fan_ratio=fan, expand_k=1)
            roots[fan] = parents.count(-1)
        assert 1 <= roots[0.999] < roots[0.05], f"the fan's ratio kept {roots} children of the root"
        # a tree held to depth one is the root's children alone; an n-gram chain through the root's best child names
        # it as the chain's where the chain is likelier, and leaves it the drafter's where it is not
        dr.reset()
        toks, _parents, depth, tags = dr.au([prompt[-1]], h, 0, 12, max_depth=1, with_tags=True)
        assert set(depth) == {1} and set(tags) == {"mtp"}, (depth, tags)
        for p, tag in ((1.0, "ngram"), (1e-4, "mtp")):
            dr.reset()
            got = dr.au([prompt[-1]], h, 0, 12, max_depth=1, extra_chains=[([toks[0]], p, "ngram")], with_tags=True)
            assert got[3][got[0].index(toks[0])] == tag, f"a chain at {p}: {got}"


def test_the_pricer_sizes_real_passes_by_the_drives_reads(q4: StreamedTextModel) -> None:
    """The verify pass priced by what its rows cost over a drive the store's probe prices (a missed expert ~10 ms):
    the first passes are plain steps measuring the step, then the drafting head's tree is probed whole, pruned to
    the prefix the model says pays and its calibration fed - the passes of more than one row verified; on a drive
    where a miss costs seconds no draft pays, and after a run of plain steps a pass verifies one draft all the same.
    Whatever the widths, the speculative tokens are the plain ones."""
    store = q4.expert_store
    assert store is not None
    with torch.inference_mode():
        plain = {name: [int(t) for t in q4.generate_greedy(p, 40)] for name, p in PROMPTS.items()}
        try:
            for drive in (DRIVE, SLOW):
                store.drive = dict(drive)
                q4._spec_cost = None  # the pricer learns afresh over each drive
                miss = store.miss_s()
                assert miss == pytest.approx(store.expert_s(drive, int(store.per or 0))) and miss > 0
                speculation(q4, tree_budget=16, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=True)
                for i, (name, prompt) in enumerate(PROMPTS.items()):
                    toks, census = q4.generate_speculative(prompt, 40, proposer="mtp_dyn", v_max=4)
                    assert toks == plain[name], f"{name}: {toks} != {plain[name]}"
                    rep = census["priced"]
                    assert rep["active"] and rep["priced_passes"] > 0, rep
                    assert rep["miss_ms"] == round(miss * 1e3, 3) and rep["reads_step"] > 0, rep
                    assert rep["calibration"].get("mtp"), f"no draft was verified: {rep}"
                    rows = {int(k): int(v) for k, v in rep["rows"].items()}
                    if i == 0:
                        assert rows.get(1, 0) >= SpecCost.WARM, f"the step was not measured first: {rows}"
                    assert any(k > 1 for k in rows), f"no pass verified a draft: {rows}"
                    if drive is SLOW:
                        assert rows.get(1, 0) > sum(v for k, v in rows.items() if k > 1), f"drafts paid: {rows}"
                        assert rows.get(2, 0) >= 1, f"no run of plain steps verified one draft: {rows}"
        finally:
            store.drive = None
            q4._spec_cost = None


def test_ragged_rows_in_one_batch_decode_as_each_row_alone(q4: StreamedTextModel) -> None:
    """prompts of three lengths served as one batch (left-padded: the n-gram embedding reads the end-of-text id at
    a padded position, the DeltaNet's convolution skips it): every row's tokens are the row decoded alone"""
    rows = [PROMPTS["short"][:9], PROMPTS["short"], PROMPTS["long"][:23]]
    with torch.inference_mode():
        alone = [[int(t) for t in q4.generate_greedy(r, 8)] for r in rows]
        batch = q4.serve(rows, 8)
    assert [[int(t) for t in b] for b in batch] == alone


def _attn_call(sm: StreamedTextModel, i: int, prompt: list[int]) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """layer `i`'s attention module and the arguments the one-token step after `prompt` hands it"""
    attn = sm.host[i].self_attn
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    hook = attn.register_forward_pre_hook(lambda _m, a, k: seen.append((a, dict(k))), with_kwargs=True)
    try:
        cache = sm.new_cache()
        forward_logits(sm, [prompt], cache)
        forward_logits(sm, [[77]], cache)
    finally:
        hook.remove()
    args, kw = seen[-1]
    return attn, args, kw


def _prefilled(sm: StreamedTextModel, prompt: list[int]) -> Any:
    cache = sm.new_cache()
    forward_logits(sm, [prompt], cache)
    return cache


def test_the_host_layers_refuse_what_they_cannot_compute_and_take_any_cache_layout(
    q4: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qwen4's host modules: the sparse attention refuses a row that sees no key and a mask that does not cover the
    cache, and reads a cache that hands back its rows strided or in float16 as those rows' float32 values, bit for
    bit; a hyper-connection refuses the wrong width; the DeltaNet's step takes a recurrent state a prefill left
    strided or in bf16 as its float32 values, rewritten in the cache; a pass with no cache at all is a prefill's"""
    prompt = PROMPTS["short"]
    sparse = next(i for i, lt in enumerate(q4.layer_types) if lt == "qwen_sparse_attention")
    delta = next(i for i, lt in enumerate(q4.layer_types) if lt == "linear_attention")
    with torch.inference_mode():
        # a pass with no cache: the indexer pools the pass's own keys, and the logits are a prefill's
        assert torch.equal(forward_logits(q4, [prompt], None), forward_logits(q4, [prompt], q4.new_cache()))
        attn, args, kw = _attn_call(q4, sparse, prompt)
        mask = kw["attention_mask"]
        n = int(mask.shape[-1])
        assert n == len(prompt) + 1

        def replay(wrap: Any = None, **over: Any) -> torch.Tensor:
            cache = _prefilled(q4, prompt)
            if wrap is not None:
                update = cache.update
                cache.update = lambda k, v, *a, **k2: wrap(*update(k, v, *a, **k2))
            out, _ = attn(*args, **{**kw, "past_key_values": cache, **over})
            return out

        base = replay()

        def restride(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(-1, -2).contiguous().transpose(-1, -2)  # the same values, each row's at a stride

        strided = replay(lambda k, v: (restride(k), restride(v)))
        assert torch.equal(strided, base), "a strided cache's rows attend otherwise"
        half = replay(lambda k, v: (k.half(), v.half()))
        rounded = replay(lambda k, v: (k.half().float(), v.half().float()))
        assert torch.equal(half, rounded), "a float16 cache is not read as its float32 values"
        # the indexer's selection held to everything, so the mask alone decides what each row sees
        monkeypatch.setattr(attn.indexer, "forward", lambda h, pe, m, c: torch.ones_like(m, dtype=torch.bool))
        with pytest.raises(RuntimeError, match="a query row sees no key"):
            replay(attention_mask=torch.zeros_like(mask, dtype=torch.bool))
        wide = torch.ones(*mask.shape[:-1], n + 2, dtype=torch.bool)
        with pytest.raises(RuntimeError, match=rf"the mask covers {n + 2} keys, the cache holds {n}"):
            replay(attention_mask=wide)
        monkeypatch.undo()

        hc = q4.host[0].attn_hyper_connection
        width = int(hc.hc_count) * int(hc.hidden_size)
        q4.aa([-1, 0])
        try:
            with pytest.raises(ValueError, match=f"Expected {width} hyper-connection features, got {width + 1}"):
                hc(torch.zeros(1, 2, width + 1))
        finally:
            q4.ab()

        def step(edit: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
            cache = _prefilled(q4, prompt)
            cl = cache.layers[delta]
            if edit is not None:
                conv, rec = q4._lin(cl)
                q4._lin_set(cl, conv, edit(rec))
            out = forward_logits(q4, [[77]], cache)[0, -1]
            return out, q4._lin(cl)[1]

        want, rec0 = step()
        assert rec0.dtype == torch.float32 and rec0.is_contiguous()
        got, rec = step(lambda r: r.transpose(-1, -2).contiguous().transpose(-1, -2))
        assert torch.equal(got, want) and rec.is_contiguous(), "a strided recurrent state steps otherwise"
        got, rec = step(lambda r: r.bfloat16())
        ref, _ = step(lambda r: r.bfloat16().float())
        assert torch.equal(got, ref) and rec.dtype == torch.float32 and rec.is_contiguous()
