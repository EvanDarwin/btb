"""The cert runner: walk the manifest's runnable cells and, for each device sub-path a family's fixture can take
on this machine, assert the engine loads it, that the sub-path's OWN `PassTag` engaged, and that greedy decode is
deterministic (two independent loads of the same fixture produce identical tokens). A greedy cell is also held to
the banked oracle (`oracle.assert_matches`), so a deterministically-wrong path fails here rather than certifying
itself; a sampled cell is held to the oracle's banked seeded draw where it computes in fp32, and elsewhere holds a
greedy decode of the same load to the greedy reference (`oracle.holds_sampled` says why). A speculative cell
decodes one load both plainly and down its drafting path, and holds the two to the same tokens.

The tag assertion is what makes a receipt worth something: it is the sub-path's own `expect`, carried on
`spec.DeviceSubpath`, so a renamed key cannot quietly turn it off and a fallback (torch on a card without the
fatbin, the step path where the megakernel would not build) can never bank a cell it did not run.

Cells whose hardware is absent here skip (recorded in the skip ledger); cells the manifest calls a GAP skip with
that gap's reason. Run the MLX cells on an Apple-silicon box, ideally under the GPU lock.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
import torch

import btb
from btb.kinds import FamilyKind, LayerKind, PassTag
from tests.helpers import FIXTURES, GGUF_FIXTURES, assert_same_tokens, loaded_model

if TYPE_CHECKING:
    from _pytest.mark.structures import ParameterSet

    from btb.engine.model import StreamedTextModel

from . import core, manifest, oracle, receipt, spec
from .oracle import PROMPT, N  # the decodes the oracle banks; the cells must not drift from them


@functools.cache
def _hardware_here(hw: spec.Hardware) -> bool:
    if hw is spec.Hardware.CPU:
        return True
    if hw is spec.Hardware.MLX:
        return btb.mlx_available()
    if hw in spec.NO_BACKEND:
        return False
    import torch

    # torch's ROCm build answers to torch.cuda too; only an NVIDIA card is this hardware
    return torch.cuda.is_available() and torch.version.hip is None


def _subpaths(surface: spec.Surface) -> tuple[spec.DeviceSubpath, ...]:
    """the sub-paths this surface runs, taken from spec by key - the knobs are never re-spelled here."""
    return spec.subpaths(*spec.SURFACE_SUBPATHS[surface])


def _why_not(
    kind: FamilyKind | None,
    storage: spec.Storage | None,
    dev: spec.DeviceSubpath,
    path: str,
    decode: spec.DecodePath | None = spec.DecodePath.GREEDY,
) -> str | None:
    """why a cell does not run on this machine, None when it does. A cell that is not this cell: one whose knob
    has nothing to act on (the manifest's dnr - a GGUF knob off GGUF, an expert-store knob on a dense family),
    one the engine cannot engage at all, which would decode down another path and bank a receipt for one it
    never took (the manifest's subpath_gap), a decode path the engine does not take for the family (its
    gap_reason), then hardware this machine lacks and a fixture not built. `decode` None for a shape axis, which
    has no storage cell to be distinct from."""
    name = os.path.basename(path)
    if decode is not None:
        if kind is None or storage is None:
            return f"{name}: no served family or supported quant claims it (a manifest orphan fixture)"
        same = manifest.dnr(kind, storage, dev, spec.DecodePath.GREEDY)
        if same is not None:
            return same
        refused = manifest.subpath_gap(kind, storage, dev)
        if refused is not None:
            return f"GAP (manifest --check fails on this): {refused}"
        gap = manifest.gap_reason(kind, storage, dev, decode) if decode is not spec.DecodePath.GREEDY else None
        if gap is not None:
            return f"GAP (manifest --check fails on this): {gap.value}"
    if not _hardware_here(dev.hardware):
        return f"{dev.hardware.value} not available on this machine"
    if not os.path.exists(path):
        return f"fixture {name} not built"
    return None


def _cell(*values: object, cid: str, why: str | None) -> ParameterSet:
    """a runner cell, marked skipped when it cannot run here: decided at collection, so a skipped cell never
    sets up the fixtures whose teardown keeps each test's memory its own"""
    return pytest.param(*values, id=cid, marks=[pytest.mark.skip(reason=why)] if why else [])


def _run_twice(
    path: str, knobs: dict[str, object], check: Callable[[StreamedTextModel], None] | None = None
) -> list[list[int]]:
    """two independent loads of one fixture, each decoding PROMPT greedily; `check` runs against the first
    model, where the pass report still describes this cell's own decode."""
    runs: list[list[int]] = []
    for i in range(2):
        with loaded_model(path, **knobs) as sm:
            runs.append([int(t) for t in sm.generate(list(PROMPT), N, speculate=False).tokens])
            if i == 0 and check is not None:
                check(sm)
    return runs


def _cells() -> list[ParameterSet]:
    # every served family x each safetensors precision it has a fixture for (the BF16 fixture and the twins
    # spec.fixture_paths binds) x the sub-paths the safetensors surface runs (spec.SURFACE_SUBPATHS - the single
    # table the manifest counts, never a second copy here) x each decode (greedy, sampled), so a new device
    # sub-path, decode or twin is exercised without editing this file.
    out: list[ParameterSet] = []
    for kind in core.served_kinds():
        if kind not in spec.FIXTURE_STEM:
            continue  # no fixture stem - a manifest GAP, not a runner cell
        for storage, info in spec.STORAGE.items():
            if info.container is not spec.Container.SAFETENSORS:
                continue
            paths = spec.fixture_paths(kind, storage)
            if not paths:
                continue  # no twin at this precision - the manifest's precision gap, not a runner cell
            subject = manifest.safetensors_subject(kind, storage)
            for dev in _subpaths(spec.Surface.SAFETENSORS):
                why = _why_not(kind, storage, dev, paths[0])
                for decode in spec.Decode:
                    cid = f"{subject}-{dev.key}-{decode.value}"
                    out.append(_cell(kind, storage, paths[0], dev, decode, cid=cid, why=why))
    return out


@pytest.mark.parametrize("kind,storage,path,dev,decode", _cells())
def test_cell_loads_and_is_deterministic(
    kind: FamilyKind, storage: spec.Storage, path: str, dev: spec.DeviceSubpath, decode: spec.Decode
) -> None:
    # a device sub-path the engine cannot engage for this family or this fixture never gets here (`_why_not`):
    # running mlx-mega on a head_dim-16 fixture just uses the step path and would pass without exercising
    # anything, so the runner never banks a receipt a fallback earned.
    stem = os.path.basename(path)
    sampled = decode is spec.Decode.SAMPLED
    runs: list[list[int]] = []
    held: list[int] | None = None  # the greedy decode a bf16 sampled cell holds to the oracle instead of its draw
    for i in range(2):  # two independent loads: catches load nondeterminism too, not just decode
        with loaded_model(path, **dev.knobs) as sm:
            runs.append(oracle.decode(sm, PROMPT, oracle.sampling("sampled" if sampled else "tokens")))
            if i == 0:
                _assert_path_engaged(sm, kind, storage, dev, decode, stem)
                if sampled and not oracle.holds_sampled(sm):
                    held = oracle.decode(sm, PROMPT)
    assert_same_tokens(runs[0], runs[1], f"{stem} on {dev.key}/{decode.value} decoded differently across two loads")
    # correctness, not just determinism: a deterministically-WRONG path fails against the banked reference
    if held is not None:
        oracle.assert_matches(kind, held, dev.hardware.value, storage=storage)
    else:
        oracle.assert_matches(kind, runs[0], dev.hardware.value, sampled=sampled, storage=storage)
    receipt.record(manifest.safetensors_id(kind, dev.key, decode, storage))  # ran+passed here (cross-machine union)


def _assert_path_engaged(
    sm: StreamedTextModel,
    kind: FamilyKind,
    storage: spec.Storage,
    dev: spec.DeviceSubpath,
    decode: spec.Decode | None,
    subject: str,
    path: spec.DecodePath = spec.DecodePath.GREEDY,
) -> None:
    """the intended path must actually engage - else a fallback (torch on cuda without the fatbin, the per-op
    path, the step path standing in for the megakernel) would decode fine and certify the wrong thing. The tags
    come from the sub-path itself (spec.DeviceSubpath.expects) and the decode path (spec.decode_tag), never from
    a table keyed by name here."""
    report = sm.last_pass_report()
    want_decode = spec.decode_tag(path)
    assert want_decode in report, (
        f"{subject} on {dev.key}/{path.value}: {want_decode} never engaged (got {sorted(report.tags)})"
    )
    if decode is not None:
        # the sampler: a sampled run must show the stochastic draw (proof the sample kernels ran), greedy the argmax
        sampled = decode is spec.Decode.SAMPLED
        assert (PassTag.SAMPLE_STOCHASTIC if sampled else PassTag.SAMPLE_GREEDY) in report, (
            f"{subject} on {dev.key}/{decode.value}: sampler tag missing (got {sorted(report.tags)})"
        )
    for want in sorted(dev.expects(kind, storage)):
        assert want in report, (
            f"{subject} on {dev.key} ({storage.value}): {want} never engaged (got {sorted(report.tags)}); "
            f"a fallback ran"
        )


# The GGUF storage axis: each tiny GGUF fixture crossed with the sub-paths the GGUF surface runs (dequant to
# bf16 on any device; gguf_packed as-stored on MLX). Certifies that reading a file's blocks - dequantized or as
# stored - loads and decodes reproducibly, and that the binding it claims is the one the pass reports. The FILES
# are discovered from the fixture directory, so a new twin is picked up; the sub-paths come from spec.


def _gguf_cells() -> list[ParameterSet]:
    out: list[ParameterSet] = []
    if not os.path.isdir(GGUF_FIXTURES):
        return out
    for fname in sorted(os.listdir(GGUF_FIXTURES)):
        if not fname.endswith(".gguf"):
            continue
        kind = spec.kind_of_stem(fname[: -len(".gguf")].rsplit("-", 1)[0])
        storage = spec.storage_of_gguf(fname)
        path = os.path.join(GGUF_FIXTURES, fname)
        for dev in _subpaths(spec.Surface.GGUF):
            why = _why_not(kind, storage, dev, path)
            out.append(_cell(kind, storage, fname, dev, cid=f"{fname[: -len('.gguf')]}-{dev.key}", why=why))
    return out


@pytest.mark.parametrize("kind,storage,fname,dev", _gguf_cells())
def test_gguf_cell_loads_and_is_deterministic(
    kind: FamilyKind, storage: spec.Storage, fname: str, dev: spec.DeviceSubpath
) -> None:
    path = os.path.join(GGUF_FIXTURES, fname)
    runs = _run_twice(path, dict(dev.knobs), lambda sm: _assert_path_engaged(sm, kind, storage, dev, None, fname))
    assert_same_tokens(runs[0], runs[1], f"{fname} on {dev.key} decoded differently across two loads")
    receipt.record(manifest.gguf_id(fname, dev.key))


# Speculative decoding: every fixture a greedy cell loads, on every sub-path its surface runs, down each decode
# path that drafts. The verify pass keeps only the tokens the model itself would pick, so a speculative decode
# must equal the plain loop's on the same load; each cell decodes both ways and holds them equal, with the
# proposer it meant to run (and the sub-path's own tags) engaged, and requires that drafts were proposed at all.


def _spec_cells() -> list[ParameterSet]:
    out: list[ParameterSet] = []
    for kind in core.served_kinds():
        for storage, info in spec.STORAGE.items():
            surface = spec.CONTAINER_SURFACE[info.container]
            for path in spec.fixture_paths(kind, storage):
                for dev in _subpaths(surface):
                    for decode in spec.SPEC_SETUP:
                        cid = manifest.spec_id(kind, storage, path, dev.key, decode).replace("/", "-")
                        why = _why_not(kind, storage, dev, path, decode)
                        out.append(_cell(kind, storage, path, dev, decode, cid=cid, why=why))
    return out


@pytest.mark.parametrize("kind,storage,path,dev,decode", _spec_cells())
def test_speculation_decodes_as_the_plain_loop(
    kind: FamilyKind, storage: spec.Storage, path: str, dev: spec.DeviceSubpath, decode: spec.DecodePath
) -> None:
    name = os.path.basename(path)
    setup = spec.SPEC_SETUP[decode]
    knobs = {**dev.knobs, **setup.knobs, **({"draft_model": path} if setup.draft else {})}
    with loaded_model(path, **knobs) as sm:
        sm.proposer = setup.proposer
        # the pass-cost curves the load times size a pass to what drafts have been yielding, which on a tiny random
        # fixture is one row: no draft at all. The cell certifies the drafting path, so it takes the configured budget
        sm._host_cost, sm._mlx_cost, sm._card_cost = {}, {}, {}
        plain = [int(t) for t in sm.generate(list(PROMPT), N, speculate=False).tokens]
        spans = [("plain", [*PROMPT, *plain])] if setup.echo else []
        drafted = [int(t) for t in sm.generate(list(PROMPT), N, speculate=True, spans=spans).tokens]
        _assert_path_engaged(sm, kind, storage, dev, None, name, decode)
        proposed = sm.last_pass_report().spec_proposed
    assert proposed > 0, f"{name} on {dev.key}/{decode.value} proposed no draft: nothing was verified"
    assert_same_tokens(drafted, plain, f"{name} on {dev.key}/{decode.value} drafted other tokens than the plain loop")
    receipt.record(manifest.spec_id(kind, storage, path, dev.key, decode))


# The 12-bit store (btb pack) is architecture-agnostic - it requantizes weights, so every family has a
# -pack12 twin. `v_max=0` turns speculation off (required for the MoE families, harmless for the dense ones);
# a sub-path whose own knobs name `v_max` keeps its own.
PACK12_KNOBS: dict[str, object] = {"v_max": 0}


def _pack12_cells() -> list[ParameterSet]:
    out: list[ParameterSet] = []
    for kind in core.served_kinds():
        stem = spec.FIXTURE_STEM.get(kind)
        if stem is None:
            continue
        path = os.path.join(FIXTURES, stem + "-pack12")
        for dev in _subpaths(spec.Surface.PACK12):
            why = _why_not(kind, spec.Storage.PACK12, dev, path)
            out.append(_cell(kind, stem, dev, cid=f"{kind.value}-pack12-{dev.key}", why=why))
    return out


@pytest.mark.parametrize("kind,stem,dev", _pack12_cells())
def test_pack12_cell_loads_and_is_deterministic(kind: FamilyKind, stem: str, dev: spec.DeviceSubpath) -> None:
    path = os.path.join(FIXTURES, stem + "-pack12")
    storage = spec.Storage.PACK12
    runs = _run_twice(
        path,
        {**PACK12_KNOBS, **dev.knobs},
        lambda sm: _assert_path_engaged(sm, kind, storage, dev, None, f"{stem}-pack12"),
    )
    assert_same_tokens(runs[0], runs[1], f"{stem}-pack12 on {dev.key} decoded differently across two loads")
    receipt.record(manifest.stem_id(spec.Surface.PACK12, stem, dev.key))


# input-shape axes beyond the single short stream: a batch of ragged rows (the batched decode loop) and a longer
# prompt (cache growth, and gemma3's sliding_window=32 eviction). One per backend here (cpu, mlx-step) - the loop
# is device-specific, family-orthogonal enough that a representative device per backend exercises it. The 1024+
# attention split (ATTN_SPLIT) still needs a real model: tiny fixtures cap at max_position_embeddings=512.
BATCH_ROWS = [[1, 2, 3, 4], [5, 6, 7, 8]]  # two rows through the batched loop (rectangular: one tensor)
LONG_PROMPT = list(range(1, 65))  # 64 tokens: past gemma3's sliding_window (32), and real cache growth


def _shape_cells(surface: spec.Surface) -> list[ParameterSet]:
    out: list[ParameterSet] = []
    for kind in core.served_kinds():
        stem = spec.FIXTURE_STEM.get(kind)
        if stem is None:
            continue
        for dev in _subpaths(surface):
            why = _why_not(kind, None, dev, os.path.join(FIXTURES, stem), None)
            out.append(_cell(stem, dev, cid=f"{kind.value}-{surface.value}-{dev.key}", why=why))
    return out


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.BATCH))
def test_batch_is_deterministic(stem: str, dev: spec.DeviceSubpath) -> None:
    """the batched decode loop (several rows at once) is reproducible per row across two independent loads."""
    path = os.path.join(FIXTURES, stem)
    runs = []
    for _ in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            runs.append(
                [list(r) for r in sm.generate([list(r) for r in BATCH_ROWS], N, speculate=False).tokens]
            )  # per-row
    assert runs[0] == runs[1], f"{stem} on {dev.key}: batched decode differed across two loads: {runs[0]} != {runs[1]}"
    receipt.record(manifest.stem_id(spec.Surface.BATCH, stem, dev.key))


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.CONTEXT))
def test_context_growth_is_deterministic(stem: str, dev: spec.DeviceSubpath) -> None:
    """a longer prompt (past gemma3's sliding window, and real cache growth) decodes reproducibly across two loads.
    Does NOT reach the 1024 attention split - tiny fixtures cap at 512; that path needs a cached real model."""
    path = os.path.join(FIXTURES, stem)
    runs = []
    for _ in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            runs.append(list(sm.generate(list(LONG_PROMPT), N, speculate=False).tokens))
    assert_same_tokens(runs[0], runs[1], f"{stem} on {dev.key}: long-context decode differed across two loads")
    receipt.record(manifest.stem_id(spec.Surface.CONTEXT, stem, dev.key))


def _axis_tags(
    sm: StreamedTextModel, surface: spec.Surface, stem: str, dev: spec.DeviceSubpath, calls: bool = True
) -> None:
    """the surface's tags in the report: the last pass's forks, and (`calls`) every API call made on the model"""
    kind = next(k for k, s in spec.FIXTURE_STEM.items() if s == stem)
    report = sm.last_pass_report()
    for want in sorted(spec.SURFACE_TAGS[surface](kind, dev)):
        if calls or "." not in want.value:
            assert want in report, (
                f"{stem} on {dev.key}/{surface.value}: {want} never engaged (got {sorted(report.tags)})"
            )


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.HOOKED))
def test_hooked_decode_is_the_plain_one(stem: str, dev: spec.DeviceSubpath) -> None:
    """a decode with every hook on (a processor, logprobs, a tapped layer) picks over the logits in hand, the
    in-graph picks standing aside, and draws the plain decode's tokens - reproducibly across two loads"""
    if not _hardware_here(dev.hardware):
        pytest.skip(f"{dev.hardware.value} not available on this machine")
    path = os.path.join(FIXTURES, stem)
    if not os.path.isdir(path):
        pytest.skip(f"fixture {stem} not built")
    runs = []
    for i in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            plain = oracle.decode(sm, PROMPT)
            g = sm.generate(list(PROMPT), N, speculate=False, processors=[lambda ids, lg: lg], logprobs=2, taps=[-1])
            if i == 0:
                _axis_tags(sm, spec.Surface.HOOKED, stem, dev)
            toks = list(g.tokens)
            assert_same_tokens(plain, toks, f"{stem} on {dev.key}: the hooked decode left the plain one")
            assert g.logprobs is not None and g.hidden is not None
            assert [t.token for t in g.logprobs] == toks and all(len(t.top) == 2 for t in g.logprobs)
            assert int(next(iter(g.hidden.values())).shape[0]) == len(toks)
            runs.append(toks)
    assert_same_tokens(runs[0], runs[1], f"{stem} on {dev.key}: hooked decode differed across two loads")
    receipt.record(manifest.stem_id(spec.Surface.HOOKED, stem, dev.key))


RAGGED = [list(PROMPT), [9, 8, 7, 6, 5], list(range(20, 33))]  # sessions of three lengths batched together


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.FORK))
def test_fork_and_batch_are_deterministic(stem: str, dev: spec.DeviceSubpath) -> None:
    """a session forked into sampled rows and sessions of ragged lengths batched decode, step, re-form and let
    rows go reproducibly across two loads, and the fork's rows keep going as a session. Every call the rows,
    a fork and a batch declare is made (spec.SURFACE_TAGS)."""
    if not _hardware_here(dev.hardware):
        pytest.skip(f"{dev.hardware.value} not available on this machine")
    path = os.path.join(FIXTURES, stem)
    if not os.path.isdir(path):
        pytest.skip(f"fixture {stem} not built")
    runs = []
    for i in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            br = sm.session(list(PROMPT)).fork(3)
            rows = br.generate(N, eos=(), sampling=oracle.sampling("sampled")).tokens
            if i == 0:
                _axis_tags(sm, spec.Surface.FORK, stem, dev, calls=False)
            br.step()
            br.reorder([0, 0, 2])
            assert br.logits is not None
            _, tapped = br.step([int(t) for t in br.logits.argmax(-1)], taps=[-1])
            br.leave(2)
            forked = [br.tokens(r) for r in range(br.n)]
            more = list(br.keep(1).generate(4, eos=(), speculate=False).tokens)
            # a fork let go: the session stands where it was forked (`keep` closes its fork too, but a call made
            # inside another is not the caller's, so the close is made here as a caller makes it)
            held = sm.session(list(PROMPT))
            held.fork(2).close()
            assert held.tokens == list(PROMPT) and held.forked is None
            with sm.batch([sm.session(r) for r in RAGGED[:2]]) as bt:
                batched = bt.generate(N, eos=()).tokens
                joined = bt.join(sm.session(RAGGED[2]))
                assert bt.logits is not None
                bt.step([int(t) for t in bt.logits.argmax(-1)])
                bt.leave(0)
                rejoined = [bt.tokens(r) for r in range(joined + 1)]
            if i == 0:
                _axis_tags(sm, spec.Surface.FORK, stem, dev)
            runs.append((rows, forked, more, batched, rejoined, sorted(tapped)))
    assert runs[0] == runs[1], f"{stem} on {dev.key}: a fork or a batch differed across two loads"
    receipt.record(manifest.stem_id(spec.Surface.FORK, stem, dev.key))


OTHER = [44, 2, 90, 13, 7, 21]  # a session's second feed
MiB = 2**20


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.MODEL))
def test_the_models_calls_are_deterministic(stem: str, dev: spec.DeviceSubpath) -> None:
    """every call the model's API declares - its own and a room's (spec.SURFACE_TAGS) - made on one load, their
    answers the same across two"""
    if not _hardware_here(dev.hardware):
        pytest.skip(f"{dev.hardware.value} not available on this machine")
    path = os.path.join(FIXTURES, stem)
    if not os.path.isdir(path):
        pytest.skip(f"fixture {stem} not built")
    runs = []
    for i in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            ids = sm.prompt_ids("hello")
            toks = list(sm.generate(list(PROMPT), N, eos=(), speculate=False).tokens)
            pick = int(sm.project(sm.hidden(list(PROMPT), layers=(-1,))[sm.L - 1][-1]).argmax())
            vec = sm.encode(["hello", "there"])
            assert torch.allclose(vec.norm(dim=-1), torch.ones(2), atol=1e-3)
            said = sm.ask("hello", max_new=4)
            streamed = "".join(sm.stream("hello", 4))
            chatted = sm.chat(max_new=4).ask("hello")
            many = sm.ask_many(["hello", "there"], max_new=4)
            with sm.batch([sm.session(list(PROMPT))]) as bt:
                batched = bt.generate(4, eos=()).tokens
            with sm.reserve("cert", MiB):
                pass
            lent = [sm.empty(64).numel(), float(sm.zeros(64).sum()), float(sm.full(64, 2.0).sum())]
            assert "cpu" in sm.memory()
            with sm.room(MiB, name="cert") as room:
                held = [room.empty(16).numel(), float(room.zeros(16).sum()), float(room.full(16, 3.0).sum())]
            sm.peak_memory()
            if i == 0:
                _axis_tags(sm, spec.Surface.MODEL, stem, dev)
            runs.append((ids, toks, pick, said, streamed, chatted, many, batched, lent, held))
    assert runs[0] == runs[1], f"{stem} on {dev.key}: the model's calls answered differently across two loads"
    receipt.record(manifest.stem_id(spec.Surface.MODEL, stem, dev.key))


@pytest.mark.parametrize("stem,dev", _shape_cells(spec.Surface.SESSION))
def test_a_sessions_calls_are_deterministic(stem: str, dev: spec.DeviceSubpath) -> None:
    """every call a session declares (spec.SURFACE_TAGS) made on one load: a rewind gives back the mark's logits,
    a hybrid refuses a crop, and the answers are the same across two loads"""
    if not _hardware_here(dev.hardware):
        pytest.skip(f"{dev.hardware.value} not available on this machine")
    path = os.path.join(FIXTURES, stem)
    if not os.path.isdir(path):
        pytest.skip(f"fixture {stem} not built")
    runs = []
    for i in range(2):
        with loaded_model(path, **dev.knobs) as sm:
            hybrid = LayerKind.LINEAR in sm.layer_types
            s = sm.session(list(PROMPT))
            m = s.mark()
            fed = s.feed(OTHER)
            last, tapped = s.feed([4], last_only=True, taps=[-1])
            s.rewind(m)
            assert torch.equal(s.feed(OTHER), fed), f"{stem} on {dev.key}: a rewind left the mark's logits"
            k, _v = s.rows(next(j for j, kind in enumerate(sm.layer_types) if kind != LayerKind.LINEAR))
            if hybrid:
                with pytest.raises(ValueError, match="mark the point"):
                    s.crop(len(PROMPT))
            else:
                s.crop(len(PROMPT))
            synced = s.sync(list(PROMPT) + OTHER)
            s.fork(2).close()
            toks = list(s.generate(N, eos=(), speculate=False).tokens)
            if i == 0:
                _axis_tags(sm, spec.Surface.SESSION, stem, dev)
            runs.append(
                (fed.argmax(-1).tolist(), int(last.argmax()), sorted(tapped), list(k.shape), int(synced.argmax()), toks)
            )
    assert runs[0] == runs[1], f"{stem} on {dev.key}: a session's calls answered differently across two loads"
    receipt.record(manifest.stem_id(spec.Surface.SESSION, stem, dev.key))


def test_cross_process_determinism() -> None:
    """determinism across PROCESSES, not just two in-process loads: a fresh interpreter that loads and greedily
    decodes the same fixture on cpu yields the same tokens. Catches process-global nondeterminism (RNG, thread
    state) the in-process check cannot. cpu only - deterministic and no GPU needed."""
    stem = spec.FIXTURE_STEM[FamilyKind.QWEN3]
    path = os.path.join(FIXTURES, stem)
    if not os.path.isdir(path):
        pytest.skip(f"fixture {stem} not built")
    script = (
        f"import btb; sm=btb.load({path!r}, device='cpu'); "
        f"print(' '.join(map(str, sm.generate({list(PROMPT)!r}, {N}, speculate=False).tokens))); sm.close()"
    )
    outs = [
        subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip()
        for _ in range(2)
    ]
    assert outs[0] == outs[1], f"cross-process nondeterminism on {stem}: {outs[0]!r} != {outs[1]!r}"


@pytest.mark.cert_gap
def test_cross_machine_coverage() -> None:
    """the union gate: every runnable cell must appear in some receipt, so a cell no machine has ever run+passed
    fails here. No one machine has every device - CI has no GPU, the Mac no CUDA - so this is the union of the
    dev's boxes and CI, read from `$BTB_CERT_RECEIPTS` (else tests/cert/receipts). Receipts are not committed:
    a machine emits one with `BTB_CERT_RECEIPT=<path> pytest tests/cert/test_cert_runner.py` and it travels in
    the pull request body as a ```json btb-receipt block, which CI drops into that directory beside its own
    artifacts before running this."""
    uncovered = sorted(manifest.runnable_ids() - receipt.merged())
    assert not uncovered, (
        f"{len(uncovered)} runnable cells no receipt covers (never run+passed on any machine). Run the cert on a "
        f"machine with the needed device and put its receipt in the PR body:\n  " + "\n  ".join(uncovered)
    )
