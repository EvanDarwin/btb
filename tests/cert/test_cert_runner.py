"""The cert runner: walk the manifest's runnable cells and, for each device sub-path a family's fixture can take
on this machine, assert the engine loads it, that the sub-path's OWN `PassTag` engaged, and that greedy decode is
deterministic (two independent loads of the same fixture produce identical tokens). A greedy cell is also held to
the banked oracle (`oracle.assert_matches`), so a deterministically-wrong path fails here rather than certifying
itself; a sampled cell is held to the oracle's banked seeded draw where it computes in fp32, and elsewhere holds a
greedy decode of the same load to the greedy reference (`oracle.holds_sampled` says why).

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

import btb
from btb.kinds import FamilyKind, PassTag
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
    import torch

    return torch.cuda.is_available()


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
    never took (the manifest's subpath_gap), then hardware this machine lacks and a fixture not built. `decode`
    None for a shape axis, which has no storage cell to be distinct from."""
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
        oracle.assert_matches(kind, held, dev.hardware.value)
    else:
        oracle.assert_matches(kind, runs[0], dev.hardware.value, sampled=sampled)
    receipt.record(manifest.safetensors_id(kind, dev.key, decode, storage))  # ran+passed here (cross-machine union)


def _assert_path_engaged(
    sm: StreamedTextModel,
    kind: FamilyKind,
    storage: spec.Storage,
    dev: spec.DeviceSubpath,
    decode: spec.Decode | None,
    subject: str,
) -> None:
    """the intended path must actually engage - else a fallback (torch on cuda without the fatbin, the per-op
    path, the step path standing in for the megakernel) would decode fine and certify the wrong thing. The tags
    come from the sub-path itself (spec.DeviceSubpath.expects), never from a table keyed by name here."""
    report = sm.last_pass_report()
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


# The 12-bit store (btb pack) is architecture-agnostic - it requantizes weights, so every family has a
# -pack12 twin. `v_max=0` turns speculation off (required for the MoE families, harmless for the dense ones).
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
        {**dev.knobs, **PACK12_KNOBS},
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
