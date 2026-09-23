"""The certification coverage manifest, both halves in one place because they are one question - what is
covered:

  A. the committed tiny-fixture cells: core's served families (read via `core.py`, never a hand list) crossed
     with the cert axes (`spec.py`), each cell DNR (not a distinct run, with a reason), COVERED, BOUND or GAP.
     COVERED is proof, not presence: every receipt id the cell needs is in `receipt.merged()`, which the runner
     writes only after the sub-path's own `PassTag` engaged. A fixture on disk with no such proof is BOUND. A
     GAP carries a `Missing` kind whose plain-language what/how lives in the `MISSING` table. A family added to
     core is iterated on its own; uncovered, it is a GAP that fails `--check`.
  B. the real-model prerequisites: the cached-only HF models and GGML quant depths a tiny fixture cannot stand
     in for (scale, real weights, the i-quant lattices gguf-py cannot write). Never committed, never
     downloaded; missing ones and the tests they mute are shown, not silently skipped.

    python -m tests.cert.manifest --report    # the grids, DNR breakdown, plain-language gaps, and prereqs
    python -m tests.cert.manifest --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.manifest --check      # nonzero on any GAP, unproven cell, reasonless DNR, orphan or fork
    python -m tests.cert.manifest --strict     # --check, and also nonzero if a real-model target is not cached

Receipts are not committed: `$BTB_CERT_RECEIPTS` (else tests/cert/receipts) is the directory CI fills from the
PR body's ```json btb-receipt block and its own artifacts, so the same command reads one machine's proof locally
and the union of every machine's in CI.

The `--missing` output is a to-do an agent can act on (a future skill wraps it): each line names a gap kind, the
count and families, what is absent, and how to close it. The taxonomy is `Missing` + `MISSING`, one place."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from gguf import GGMLQuantizationType as GQ

from btb.kinds import Cap, FamilyKind, PassTag, Quant, QuantClass

from . import core, receipt, spec

FIXTURES = spec.FIXTURES
GGUF_DIR = spec.GGUF_DIR

# the shapes the card's step-graph kernels are written for (btb/engine/cuda.py:295: a lane holds D/32 dims of a
# head), and the megakernel's head multiple (btb/mlx/mega.py:599). Neither is reachable from a torch-free
# module, so test_manifest holds both against the source they are quoted from.
CARD_HEAD_DIMS: tuple[int, ...] = (64, 128, 256)
MEGA_HEAD_MULTIPLE = 64


class Verdict(StrEnum):
    DNR = "DNR"  # not a distinct run (identical to another cell); carries a reason
    COVERED = "COVERED"  # every receipt id the cell needs was run+passed with its own PassTag engaged
    BOUND = "BOUND"  # the fixture is on disk and the path is implemented; no receipt proves it ever ran
    GAP = "GAP"  # a real path nothing covers; carries a Missing kind


class Missing(StrEnum):
    """what a GAP needs, as a stable key. `MISSING` maps each to (what is absent, how to close it) in plain
    language, so the report and the `--missing` list read as a to-do an agent (or person) can act on directly.
    A new gap kind is one member here and one row in MISSING - the natural-language taxonomy lives in one place."""

    GGUF_LOAD = "gguf-load"
    MXFP4_NON_GPTOSS = "mxfp4-non-gptoss"
    GPTOSS_GGUF_TWIN = "gptoss-gguf-twin"
    KIQUANT_FIXTURE = "kiquant-fixture"
    MEGAKERNEL = "megakernel"
    MEGA_STORAGE = "mega-storage"
    MEGA_SHAPE = "mega-shape"
    CARD_GRAPH_FAMILY = "card-graph-family"
    CARD_GRAPH_SHAPE = "card-graph-shape"
    SPEC_MTP_HEAD = "spec-mtp-head"
    SPEC_DECODE = "spec-decode"
    FP16_FIXTURE = "fp16-fixture"
    FP32_FIXTURE = "fp32-fixture"
    FP8_UNIMPLEMENTED = "fp8-unimplemented"
    QUANT_FIXTURE = "quant-fixture"
    NO_RUNNER_CELL = "no-runner-cell"
    NO_FIXTURE = "no-fixture"


# kind -> (what is missing, how to close it), both a full sentence. This is the single natural-language source
# for what the cert is short of; keep each entry actionable - name the file/knob a fix touches.
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.GGUF_LOAD: (
        "the engine cannot load this family from a GGUF file - its llama.cpp architecture is not in "
        "hf.ARCH_MODEL_TYPES, so no GGUF path exists for it",
        "add the family's architecture to hf.ARCH_MODEL_TYPES with its GGUF tensor-name mapping, then commit a "
        "tiny GGUF fixture (tests/make_fixtures.py) so the storage cells can bind",
    ),
    Missing.MXFP4_NON_GPTOSS: (
        "an mxfp4 GGUF is expected for a family that does not use mxfp4 (mxfp4 is gpt-oss's expert format)",
        "either wire mxfp4 loading for this family and add a fixture, or, if mxfp4 is genuinely gpt-oss-only, "
        "record that as a DNR in dnr() with the proof - do not leave it an implied gap",
    ),
    Missing.GPTOSS_GGUF_TWIN: (
        "gpt-oss ships only an mxfp4 GGUF; no GGUF twin of it in any other stored type is produced",
        "produce a bf16/affine GGUF of gpt-oss and add the fixture, or record it as a DNR with the reason it "
        "cannot exist",
    ),
    Missing.KIQUANT_FIXTURE: (
        "no committed k-quant/i-quant tiny fixture exists (gguf-py cannot write these block types), so the "
        "as-stored k/i-quant kernels are exercised by nothing in CI",
        "cache the real Qwen3-0.6B GGUFs the manifest lists (--strict tracks them) so test_gguf runs the k/i-quant "
        "kernels, or add a fixture-generation path that can emit these block types",
    ),
    Missing.MEGAKERNEL: (
        "the fused MLX megakernel is not written for this family: it takes the dense kernel layout alone "
        "(mlx_forward.py:1320 refuses a sandwich family), so a MoE family runs the per-op host path "
        "(MLX_PEROP), a hybrid family the DeltaNet forward (MLX_HYBRID), and the rest the step path "
        "(MLX_STEP_FUSED for a sandwich layout, MLX_STEP_UNFUSED without one)",
        "implement the megakernel for this family's layout, or record it as a DNR if the architecture genuinely "
        "precludes a dense fused kernel",
    ),
    Missing.MEGA_STORAGE: (
        "the megakernel refuses this storage: it reads weights out of slot buffers, so as-stored quant blocks "
        "(mlx_forward.py:1320, `self.mlx_state.affine`) and the 12-bit store's packed weights (mega.py:609, "
        "'a weight outside a slot buffer') both fall to the step path",
        "teach the megakernel to read packed weights, or record the refusal as a DNR once it is provably "
        "permanent - until then the cell is a real path the cert cannot certify",
    ),
    Missing.MEGA_SHAPE: (
        "the megakernel takes a head of 64k dims (mega.py:599, `self.hd % 64`); every committed tiny fixture is "
        "head_dim 16, so it never builds here and any cell asserting it would be certified by the step path",
        "run this cell against a cached real model (Qwen3-0.6B is head_dim 128 - see the prereqs below), which "
        "is the only way the megakernel is exercised at all",
    ),
    Missing.CARD_GRAPH_FAMILY: (
        "the captured card step graph is not written for this family (cuda.py:284 takes the kernel layout or "
        "the sandwich layout, and never a family that brings its own layer), so a card run of it decodes "
        "through the torch modules - PassTag.CUDA_TORCH_FALLBACK, the `cuda-torch` sub-path",
        "write the card kernels for this family's layout, or leave the family certified under `cuda-torch`, "
        "which is the path it really runs",
    ),
    Missing.CARD_GRAPH_SHAPE: (
        "the card kernels take head_dim 64, 128 or 256 with 8-aligned widths (cuda.py:295); the tiny fixtures "
        "are head_dim 16, so the graph never captures and a fallback would certify the cell",
        "run this cell against a cached real model on the CUDA box (Qwen3-0.6B is head_dim 128)",
    ),
    Missing.SPEC_MTP_HEAD: (
        "speculation via an MTP head cannot run: this family's tiny fixture has no MTP drafter head (no `mtp.*` "
        "entry in its weight map, the engine's own probe at scheduler.py:573)",
        "add an MTP head to the fixture in tests/make_fixtures.py (see _mtp_head), then add a spec-mtp runner "
        "cell - spec.has_mtp_head reads the fixture, so no list needs editing",
    ),
    Missing.SPEC_DECODE: (
        "speculation decode is not exercised at all - the cert runner decodes greedily (speculate=False)",
        "add spec-decode runner cells (MTP, n-gram, and a draft-model pair) that assert the accepted tokens match "
        "a banked greedy oracle; speculation is otherwise entirely uncertified",
    ),
    Missing.FP16_FIXTURE: (
        "no fp16 safetensors fixture exists - the engine reads fp16 checkpoints (hf._DTYPE_LABEL) but the cert "
        "has no fp16 twin, so the fp16 storage path is unexercised",
        "add an fp16 twin of the tiny fixtures in tests/make_fixtures.py; spec.fixture_paths binds it as soon as "
        "the precision has a fixture of its own (it must never bind the bf16 directory)",
    ),
    Missing.FP32_FIXTURE: (
        "no fp32 safetensors fixture exists - every committed fixture's header is BF16, so the fp32 LOAD path "
        "(a float32 checkpoint read and cast) is exercised by nothing; the `fp32` load option is a compute "
        "dtype, not this",
        "add an fp32 twin of the tiny fixtures in tests/make_fixtures.py; until then this cell must not bind the "
        "bf16 fixture, which is what made it read as covered",
    ),
    Missing.FP8_UNIMPLEMENTED: (
        "fp8 checkpoint loading is unimplemented (no float8/e4m3/e5m2 path in the engine), so the fp8 storage "
        "axis cannot be covered",
        "implement fp8 dequant/load in the engine, then add an fp8 fixture; or record fp8 as out of scope",
    ),
    Missing.QUANT_FIXTURE: (
        "this storage stands for several stored types (one kernel each) and only some have a tiny GGUF twin, so "
        "certifying the cell off the ones that exist would claim kernels nothing reads",
        "write the missing twins in tests/make_fixtures.py - the cell binds one file per kinds.Quant member of "
        "its class, so the report names exactly which are absent",
    ),
    Missing.NO_RUNNER_CELL: (
        "the fixture is on disk and the path is implemented, but no cert run loads this storage on this device "
        "sub-path - there is no receipt id it could ever produce",
        "add the combination to spec.SURFACE_SUBPATHS and give test_cert_runner a cell for it, or record why the "
        "combination is not a distinct run in dnr()",
    ),
    Missing.NO_FIXTURE: (
        "this family is served but has no tiny fixture for this storage at all",
        "add the family's fixture stem to spec.FIXTURE_STEM and generate the twin in tests/make_fixtures.py",
    ),
}

# the safetensors precisions with no committed fixture, each to the gap kind that says why. A precision added to
# spec.Storage with no row here falls through to NO_FIXTURE; test_manifest holds the table total.
SAFE_PRECISION_GAP: dict[spec.Storage, Missing] = {
    spec.Storage.SAFE_FP16: Missing.FP16_FIXTURE,
    spec.Storage.SAFE_FP32: Missing.FP32_FIXTURE,
    spec.Storage.SAFE_FP8: Missing.FP8_UNIMPLEMENTED,
}


# --- engine forks the cell grid does not reach -------------------------------------------------------------


@dataclass(frozen=True)
class Fork:
    """a branch the engine really takes that no cell can certify because nothing marks it: `where` is the fork,
    `selects` what chooses it, `why` what the cert would need first. An unmarked fork is the hole a cell cannot
    see - a run down the other branch would pass - so each one is reported and fails --check."""

    name: str
    where: str
    selects: str
    why: str


UNTAGGED_FORKS: tuple[Fork, ...] = ()

# PassTags no device sub-path expects, each with why the grid does not reach it. Held total against PassTag, so
# a new fork in the engine is a member here or a sub-path that asserts it - never a tag nothing looks at.
FORK_NOTES: dict[PassTag, str] = {
    PassTag.MLX_STEP: "the umbrella tag for the MLX step path; the cells assert the fused/unfused fork below it",
    PassTag.MLX_SDPA: "MLX's own fused attention, taken where the node kernel does not apply; no cell forces it",
    PassTag.TIER_RESIDENT: "every tiny fixture is wholly resident, so every cell's pass carries it, none asserts it",
    PassTag.TIER_COLD: "the cold tier needs a model the machine cannot hold; no tiny fixture reaches it",
    PassTag.TIER_STREAMED: "a layer on no tier at all (device.py:178) needs a model larger than RAM + VRAM",
    PassTag.HEAD_RESIDENT: (
        "the head held where the model runs, which is what every cell but cpu-headstream takes (or not - the "
        "planner's `head_on_card` decides where `resident_head` is unset); none asserts it"
    ),
    PassTag.PREFILL_CARD: (
        "a prefill's host layers on the card (forward.py:324): the planner decides `prefill_card` and the "
        "`prefill_card_min` option only sets the rows it takes, so a cell would need a card, a cpu/card split "
        "and a prompt that long - none of which the tiny fixtures reach here"
    ),
    PassTag.EXPERT_TABLES: (
        "the MoE experts with no store (host.py:434), which needs expert_cache_gb=0 - and that option takes a "
        "figure above 0, so nothing a caller may pass selects this arm; it is the no-native-read_direct path"
    ),
    PassTag.EXPERT_STORE: (
        "the store served the layer's experts (host.py:424), which every MoE cell does, so each carries it "
        "and none asserts it"
    ),
    PassTag.EXPERT_VRAM_SEAT: (
        "an expert seated in VRAM (`vram_experts_gb`, experts.py:1093) needs a card, which this grid's MoE "
        "fixtures have no cell for"
    ),
    PassTag.EXPERT_MXFP4_DEQUANT: (
        "MXFP4 experts widened instead of multiplied as stored (host.py:376) happens only where the native "
        "library was built without the mx4 matvec, which the cert's own machines are not"
    ),
    PassTag.SPEC_OFF: "every runner cell decodes with speculate=False; the tag is carried but nothing asserts it",
    PassTag.SPEC_MTP: "no runner cell speculates (Missing.SPEC_DECODE); the decode axis reports it per family",
    PassTag.SPEC_DRAFT: "no runner cell speculates; a sibling draft model also needs a second cached model",
    PassTag.SPEC_NGRAM: "no runner cell speculates (Missing.SPEC_DECODE)",
    PassTag.SPEC_ACCEPT: "an accepted draft; unreachable while no cell speculates",
    PassTag.SPEC_REJECT: "a rejected draft; unreachable while no cell speculates",
}


def expected_tags() -> frozenset[PassTag]:
    """every PassTag some device sub-path's cell would assert, over the served families and storages - the tags
    the grid is able to certify. Derived from the sub-path table, so adding a sub-path closes a fork note."""
    out = {PassTag.SAMPLE_GREEDY, PassTag.SAMPLE_STOCHASTIC}  # asserted by every runner cell, per decode
    for dev in spec.DEVICE_SUBPATHS:
        for kind in core.served_kinds():
            for st in spec.Storage:
                out |= dev.expects(kind, st)
    return frozenset(out)


def unnoted_tags() -> list[PassTag]:
    """PassTags that neither a sub-path asserts nor FORK_NOTES explains - a fork that would pass unexamined."""
    return [t for t in PassTag if t not in expected_tags() and t not in FORK_NOTES]


def noted_but_asserted() -> list[PassTag]:
    """tags a cell now expects that FORK_NOTES still calls uncertified - the note outlived the fix, and while it
    stands --check reports a fork as uncovered that a sub-path is in fact asserting."""
    return [t for t in FORK_NOTES if t in expected_tags()]


# --- Part A: the committed tiny-fixture cell matrix --------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    kind: FamilyKind
    storage: spec.Storage
    device: str
    decode: spec.DecodePath
    verdict: Verdict
    reason: str = ""
    ids: tuple[str, ...] = ()  # the receipt ids that, together, prove this cell ran with its PassTag engaged


# --- the cell id: the one spelling the runner records and the manifest reads --------------------------------


def cell_id(surface: spec.Surface, *parts: str) -> str:
    """a cert run's canonical id: the surface it loaded from, then what it loaded and which sub-path ran it.
    The runner's record() and the manifest's coverage both build ids here, so the two cannot drift apart."""
    return "/".join([surface.value, *parts])


def safetensors_id(kind: FamilyKind, key: str, decode: spec.Decode) -> str:
    return cell_id(spec.Surface.SAFETENSORS, kind.value, key, decode.value)


def gguf_id(fname: str, key: str) -> str:
    """the id for one tiny GGUF file on one sub-path; `fname` is the file, so the quant is part of the id and
    coverage is per stored type rather than per storage class."""
    return cell_id(spec.Surface.GGUF, os.path.basename(fname).removesuffix(".gguf"), key)


def stem_id(surface: spec.Surface, stem: str, key: str) -> str:
    """the id for a whole-checkpoint surface (the 12-bit store, the input-shape axes), keyed by fixture stem."""
    return cell_id(surface, stem, key)


def cell_ids(
    kind: FamilyKind, storage: spec.Storage, dev: spec.DeviceSubpath, decode: spec.DecodePath
) -> tuple[str, ...]:
    """every receipt id a cell needs before it is COVERED, empty when no cert run exercises the combination at
    all (which is Missing.NO_RUNNER_CELL, not silence). The runner decodes without speculation, so only the
    GREEDY decode path has runs; its two samplers are two ids of the same cell."""
    stem = spec.FIXTURE_STEM.get(kind)
    if stem is None or decode is not spec.DecodePath.GREEDY:
        return ()
    container = spec.STORAGE[storage].container
    surface = spec.CONTAINER_SURFACE[container]
    if dev.key not in spec.SURFACE_SUBPATHS[surface]:
        return ()
    if container is spec.Container.SAFETENSORS:
        if storage is not spec.Storage.SAFE_BF16:
            return ()  # the committed fixtures are bf16 headers; no other precision has a run
        return tuple(safetensors_id(kind, dev.key, d) for d in spec.Decode)
    if container is spec.Container.PACK12:
        return (stem_id(surface, stem, dev.key),)
    return tuple(gguf_id(p, dev.key) for p in spec.fixture_paths(kind, storage))


def shape_ids() -> frozenset[str]:
    """the input-shape axes' ids (a ragged batch, a long prompt), which sit outside the storage/device grid:
    one per served family with a fixture, per sub-path the shape runs take."""
    out: set[str] = set()
    for kind in core.served_kinds():
        stem = spec.FIXTURE_STEM.get(kind)
        if stem is None or not os.path.isdir(os.path.join(FIXTURES, stem)):
            continue
        for surface in (spec.Surface.BATCH, spec.Surface.CONTEXT):
            out |= {stem_id(surface, stem, key) for key in spec.SURFACE_SUBPATHS[surface]}
    return frozenset(out)


def runnable_ids() -> frozenset[str]:
    """every id some machine could run+pass: the grid's runnable cells plus the input-shape axes. The union
    gate (test_cross_machine_coverage) measures the merged receipts against exactly this."""
    ids = {i for c in compute_cells() if c.verdict in (Verdict.BOUND, Verdict.COVERED) for i in c.ids}
    return frozenset(ids | shape_ids())


# --- the verdicts -------------------------------------------------------------------------------------------


def dnr(kind: FamilyKind, storage: spec.Storage, dev: spec.DeviceSubpath, decode: spec.DecodePath) -> str | None:
    """cells that are not a distinct run at all: an axis combination that produces the SAME run as another cell,
    provably and with no judgment about worth. This is the ONLY thing that is legitimately "did not run" - it is
    never used to file away an unimplemented path (that is a GAP, see gap_reason). Both cases are a knob with
    nothing to act on: a GGUF-storage knob on a non-GGUF fixture, and an expert-store knob on a family that
    builds no store. `DeviceSubpath.only` and `.needs` name what a knob acts on, so the rule reads the axis."""
    if dev.only is not None and spec.STORAGE[storage].container is not dev.only:
        return (
            f"not a distinct cell: {dev.key}'s knob is a no-op off {dev.only.value} storage "
            f"(identical to the plain {dev.hardware.value} run)"
        )
    if dev.needs is not None and dev.needs not in core.flags(kind):
        return (
            f"not a distinct cell: {dev.key}'s knob is a no-op on a family without {dev.needs.value} "
            f"(identical to the plain {dev.hardware.value} run)"
        )
    return None


def _mega_gap(kind: FamilyKind, storage: spec.Storage) -> Missing | None:
    """why the megakernel cannot engage, by the engine's own gates: the family's layout (mlx_forward.py:1320
    takes kernel_layout and refuses a sandwich), the weight binding it reads through, then the head width."""
    fl = core.flags(kind)
    if Cap.KERNEL_LAYOUT not in fl or Cap.SANDWICH in fl:
        return Missing.MEGAKERNEL
    qc = spec.quant_class(storage)
    if spec.STORAGE[storage].container is spec.Container.PACK12 or qc not in (None, QuantClass.FLOAT):
        return Missing.MEGA_STORAGE
    hd = spec.head_dim(kind)
    if hd is None or hd % MEGA_HEAD_MULTIPLE:
        return Missing.MEGA_SHAPE
    return None


def _card_graph_gap(kind: FamilyKind) -> Missing | None:
    """why the captured card graph cannot engage: the family core declares, then the fixture's head width."""
    if not core.card_family_ok(kind):
        return Missing.CARD_GRAPH_FAMILY
    return None if spec.head_dim(kind) in CARD_HEAD_DIMS else Missing.CARD_GRAPH_SHAPE


def subpath_gap(kind: FamilyKind, storage: spec.Storage, dev: spec.DeviceSubpath) -> Missing | None:
    """why this device sub-path cannot engage on this family and storage, by the engine's own gates - the one
    rule the grid and the runner share. A run of a cell this refuses would decode fine down another path and
    bank a receipt for one it never took, so the runner skips exactly these."""
    if dev.key == "mlx-mega":
        return _mega_gap(kind, storage)
    if dev.key == "cuda-graph":
        return _card_graph_gap(kind)
    return None


def gap_reason(
    kind: FamilyKind, storage: spec.Storage, dev: spec.DeviceSubpath, decode: spec.DecodePath
) -> Missing | None:
    """the Missing kind for a real cell nothing covers - the feature is unimplemented for this family, or the cert
    does not exercise the path. None means the cell is coverable. MISSING[kind] holds the plain-language what/how.
    Never "impossible"; a GAP that fails the gate.

    The order is the order the engine hits them: a storage it cannot load at all, then a sub-path that cannot
    engage on this family or this fixture, then an artifact the cert does not have, then a decode path nothing
    drives. The deepest true reason wins, so a cell that would still not run with every fixture in place says so."""
    fl = core.flags(kind)
    info = spec.STORAGE[storage]
    qc = spec.quant_class(storage)
    # 1. storage the engine cannot load for this family
    if info.container is spec.Container.GGUF:
        if kind not in core.gguf_kinds():
            return Missing.GGUF_LOAD
        if qc is QuantClass.MXFP4 and Cap.MXFP4 not in fl:
            return Missing.MXFP4_NON_GPTOSS
        if Cap.MXFP4 in fl and qc is not QuantClass.MXFP4:
            return Missing.GPTOSS_GGUF_TWIN
    # 2. the device sub-path the engine does not engage for this family or this fixture
    refused = subpath_gap(kind, storage, dev)
    if refused is not None:
        return refused
    # 3. an artifact the cert does not have
    if info.container is spec.Container.SAFETENSORS and storage in SAFE_PRECISION_GAP:
        return SAFE_PRECISION_GAP[storage]
    if qc in (QuantClass.KQUANT, QuantClass.IQ4, QuantClass.LATTICE):
        return Missing.KIQUANT_FIXTURE
    paths = spec.fixture_paths(kind, storage)
    absent = [p for p in paths if not os.path.exists(p)]
    if not paths or len(absent) == len(paths):
        return Missing.NO_FIXTURE
    if absent:
        return Missing.QUANT_FIXTURE
    # 4. a decode path the cert runner does not drive
    proposer = spec.DECODE_PROPOSER[decode]
    if proposer is not None and proposer.mtp and not spec.has_mtp_head(kind):
        return Missing.SPEC_MTP_HEAD
    if spec.DECODE_KIND[decode] is not spec.DecodeKind.PLAIN:
        return Missing.SPEC_DECODE
    return None


def bound_paths() -> set[str]:
    """the exact set of on-disk artifacts the cells bind, over every served family and storage."""
    return {p for kind in core.served_kinds() for st in spec.Storage for p in spec.fixture_paths(kind, st)}


def fixture_gaps() -> list[str]:
    """fixtures on disk that no cell binds - an orphan (a stale packing scheme, a GGUF twin of a type no storage
    member stands for). The fixture-side of native_ops.bench_gaps: a fixture nothing exercises is a reported gap,
    so it cannot linger uncertified. Computed against `bound_paths()`, not a name prefix, so a twin that no cell
    reads is visible even when its stem is a served family's."""
    expected = bound_paths()
    out: list[str] = []
    for name in sorted(os.listdir(FIXTURES)):
        path = os.path.join(FIXTURES, name)
        if name == os.path.basename(GGUF_DIR) or not os.path.isdir(path):
            continue
        if path not in expected:
            out.append(name)
    if os.path.isdir(GGUF_DIR):
        for name in sorted(os.listdir(GGUF_DIR)):
            if name.endswith(".gguf") and os.path.join(GGUF_DIR, name) not in expected:
                out.append(f"gguf/{name}")
    return out


def compute_cells(receipts: frozenset[str] | set[str] | None = None) -> list[Cell]:
    """the grid. A cell is COVERED only when every id it needs is in the merged receipts - the runner records an
    id after asserting the sub-path's own PassTag engaged, so a fallback cannot produce one. A runnable cell
    whose fixture is on disk but whose ids are not all banked is BOUND: implemented, unproven."""
    banked = receipt.merged() if receipts is None else set(receipts)
    cells: list[Cell] = []
    for kind in core.served_kinds():
        for storage in spec.Storage:
            for dev in spec.DEVICE_SUBPATHS:
                for decode in spec.DecodePath:
                    non_cell = dnr(kind, storage, dev, decode)
                    if non_cell is not None:
                        cells.append(Cell(kind, storage, dev.key, decode, Verdict.DNR, non_cell))
                        continue
                    why = gap_reason(kind, storage, dev, decode)
                    if why is not None:
                        cells.append(Cell(kind, storage, dev.key, decode, Verdict.GAP, why.value))
                        continue
                    ids = cell_ids(kind, storage, dev, decode)
                    if not ids:
                        cells.append(Cell(kind, storage, dev.key, decode, Verdict.GAP, Missing.NO_RUNNER_CELL.value))
                        continue
                    proven = set(ids) <= banked
                    cells.append(
                        Cell(
                            kind,
                            storage,
                            dev.key,
                            decode,
                            Verdict.COVERED if proven else Verdict.BOUND,
                            ""
                            if proven
                            else f"unproven: {len([i for i in ids if i not in banked])} of "
                            f"{len(ids)} receipt id(s) never banked",
                            ids,
                        )
                    )
    return cells


def _symbol(cells: list[Cell]) -> str:
    v = {c.verdict for c in cells}
    if not v or v == {Verdict.DNR}:
        return "-"
    if Verdict.GAP in v:
        return "!" if v & {Verdict.COVERED, Verdict.BOUND} else "x"
    return "b" if Verdict.BOUND in v else "ok"


def _grid(cells: list[Cell], col_of: Callable[[Cell], str], cols: list[str], title: str) -> list[str]:
    """the coverage grid as space-aligned monospace columns (a terminal does not render markdown tables)."""
    names = [spec.FIXTURE_STEM.get(k) or k.value for k in core.served_kinds()]
    fam_w = max([len("family"), *(len(n) for n in names)])
    col_w = [len(c) for c in cols]  # symbols (ok/!/x/-) are never wider than a column header
    header = "  ".join([f"{'family':<{fam_w}}", *(f"{c:>{w}}" for c, w in zip(cols, col_w))])
    lines = [
        f"{title} (ok=proven by receipt, b=bound but unproven, !=partial gap, x=all gap, -=none run):",
        "",
        header,
        "  ".join(["-" * fam_w, *("-" * w for w in col_w)]),
    ]
    for kind, name in zip(core.served_kinds(), names):
        syms = [_symbol([c for c in cells if c.kind is kind and col_of(c) == col]) for col in cols]
        lines.append("  ".join([f"{name:<{fam_w}}", *(f"{s:>{w}}" for s, w in zip(syms, col_w))]))
    return lines


# --- Part B: the real-model prerequisites (cached-only, never committed) -----------------------------------

# GGML quant types the engine has kernels for, from the one declaration in kinds.Quant (mapped to gguf's enum
# by name) - never a hand list. A type exercised by no fixture or cached model is an uncovered quant; a new
# kernel is a new Quant member, and it appears here on its own.
SUPPORTED_QUANTS: frozenset[GQ] = frozenset(GQ[q.name] for q in Quant)
# what the committed tiny GGUF fixtures already exercise (make_fixtures writes only these); always credited
TINY_FIXTURE_QUANTS: frozenset[GQ] = frozenset({GQ.BF16, GQ.F16, GQ.Q4_0, GQ.Q8_0, GQ.MXFP4})
# the Unsloth Qwen3-0.6B GGUF file names -> the GGML type each exercises (the mixes to their dominant block type)
QWEN3_06B_GGUF: dict[str, GQ] = {
    "BF16": GQ.BF16, "Q8_0": GQ.Q8_0, "Q6_K": GQ.Q6_K, "Q5_K_M": GQ.Q5_K, "Q5_K_S": GQ.Q5_K,
    "Q4_K_M": GQ.Q4_K, "Q4_K_S": GQ.Q4_K, "Q4_0": GQ.Q4_0, "Q4_1": GQ.Q4_1, "Q3_K_M": GQ.Q3_K,
    "Q3_K_S": GQ.Q3_K, "Q2_K": GQ.Q2_K, "IQ4_NL": GQ.IQ4_NL, "IQ4_XS": GQ.IQ4_XS,
    "UD-IQ3_XXS": GQ.IQ3_XXS, "UD-IQ2_M": GQ.IQ2_S, "UD-IQ1_M": GQ.IQ1_M,
}  # fmt: skip


@dataclass(frozen=True)
class Target:
    """a real HF model the full cert wants cached: `spec` is what `cached()` resolves, `quant` the GGML type a
    GGUF target exercises (None for safetensors), `tests` the tests it feeds (empty = an uncovered path)."""

    spec: str
    kind: FamilyKind
    params: str
    moe: bool
    quant: GQ | None
    tests: str = ""


def targets() -> list[Target]:
    K = FamilyKind
    out = [
        Target("Qwen/Qwen3-0.6B", K.QWEN3, "0.6B", False, None, "test_mega, test_card_graph, test_gguf ref"),
        Target("Qwen/Qwen3-1.7B", K.QWEN3, "1.7B", False, None),
        Target("Qwen/Qwen3-4B", K.QWEN3, "4B", False, None),
        Target("Qwen/Qwen3-8B", K.QWEN3, "8B", False, None),
        Target("Qwen/Qwen3-14B", K.QWEN3, "14B", False, None),
        Target("Qwen/Qwen3-32B", K.QWEN3, "32B", False, None),
        Target(
            "unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf", K.QWEN3, "4B", False, GQ.Q4_K, "test_gguf draft speculation"
        ),
        Target("microsoft/Phi-3-mini-4k-instruct", K.PHI3, "3.8B", False, None),
        Target("microsoft/Phi-3-medium-4k-instruct", K.PHI3, "14B", False, None),
        Target("openai/gpt-oss-20b", K.GPT_OSS, "20B", True, GQ.MXFP4),
        Target("openai/gpt-oss-120b", K.GPT_OSS, "120B", True, GQ.MXFP4),  # the flagship the expert store exists for
        Target("google/gemma-3-270m", K.GEMMA3, "270M", False, None, "test_gemma (MLX fused, past-window)"),
        Target("google/gemma-3-1b-it", K.GEMMA3, "1B", False, None),
        Target("google/gemma-3-4b-it", K.GEMMA3, "4B", False, None),
        Target("google/gemma-3-12b-it", K.GEMMA3, "12B", False, None),
        Target("google/gemma-3-27b-it", K.GEMMA3, "27B", False, None),
    ]
    out += [
        Target(f"unsloth/Qwen3-0.6B-GGUF:Qwen3-0.6B-{n}.gguf", K.QWEN3, "0.6B", False, q, "test_gguf real-quant")
        for n, q in QWEN3_06B_GGUF.items()
    ]
    return out


def _present(spec_str: str) -> bool:
    """present in the local cache, by the exact rule the tests use (btb.resolve local-only, never a download)."""
    try:
        from btb import resolve

        resolve(spec_str, local=True)
        return True
    except Exception:
        return False


def target_status() -> list[tuple[Target, bool]]:
    return [(t, _present(t.spec)) for t in targets()]


def covered_quants() -> set[GQ]:
    """GGML quants exercised here: the committed tiny fixtures (always) plus what cached real targets add."""
    return set(TINY_FIXTURE_QUANTS) | {t.quant for t, ok in target_status() if ok and t.quant is not None}


# --- one report / summary / gate over both halves ----------------------------------------------------------


def missing_items() -> list[tuple[Missing, int, list[str]]]:
    """the structured 'what is missing': each Missing kind the manifest currently has, how many cells it covers,
    and the families it affects. The natural-language backing for `--missing`, iterable by a future skill."""
    gaps = [c for c in compute_cells() if c.verdict is Verdict.GAP]
    out: list[tuple[Missing, int, list[str]]] = []
    for kind in Missing:
        cs = [c for c in gaps if c.reason == kind.value]
        if cs:
            fams = sorted({spec.FIXTURE_STEM.get(c.kind) or c.kind.value for c in cs})
            out.append((kind, len(cs), fams))
    return out


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close it, per kind. This is the agent-facing
    to-do; reads the same whether printed by --missing or embedded in --report."""
    items = missing_items()
    total = sum(n for _k, n, _f in items)
    lines = [f"{total} uncovered cells across {len(items)} kinds of missing work:", ""]
    for kind, n, fams in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {n} cells - families: {', '.join(fams)}")
        lines.append(f"    missing:  {what}")
        lines.append(f"    to close: {how}")
        lines.append("")
    return lines


def summary_line() -> str:
    """the one line conftest prints at the top of a run: fixture-cell gaps, cells bound but never proven by a
    receipt, uncached real models and quant gaps."""
    cells = compute_cells()
    gaps = len({(c.kind, c.storage) for c in cells if c.verdict is Verdict.GAP})
    unproven = sum(1 for c in cells if c.verdict is Verdict.BOUND)
    miss = sum(1 for _t, ok in target_status() if not ok)
    qgap = len(SUPPORTED_QUANTS - covered_quants())
    if not gaps and not unproven and not miss and not qgap:
        return "cert coverage: no fixture gaps, every cell proven, every target cached, every quant exercised"
    return (
        f"cert coverage: {gaps} fixture gap(s), {unproven} cell(s) bound but unproven, {miss} real model(s) not "
        f"cached, {qgap} supported quant(s) unexercised (python -m tests.cert.manifest --report)"
    )


def _claimed_note() -> str:
    """the ids whose only evidence is a receipt pasted into the PR body - a claim CI cannot verify - named in the
    PROOF line so a reviewer sees what a cert run verified and what is merely asserted."""
    src = receipt.by_source()
    claimed = src.get("pr-body", set()) - src.get("run", set())
    if not claimed:
        return ""
    return f"; {len(claimed)} of them CLAIMED in the PR body only (pasted, unverified by CI - the reviewer judges)"


def report() -> int:
    cells = compute_cells()
    problems = core.consistency_problems()
    if problems:
        print("CORE FAMILY DRIFT (the redundancy has diverged):")
        for p in problems:
            print(f"  ! {p}")
        print()
    n = {v: sum(1 for c in cells if c.verdict is v) for v in Verdict}
    print(
        f"A. FIXTURE CELLS: {len(core.served_kinds())} families x {len(spec.Storage)} storage x "
        f"{len(spec.DEVICE_SUBPATHS)} device x {len(spec.DecodePath)} decode = {len(cells)} cells\n"
    )
    print("\n".join(_grid(cells, lambda c: c.device, [d.key for d in spec.DEVICE_SUBPATHS], "family x device")))
    print()
    print("\n".join(_grid(cells, lambda c: c.storage.value, [s.value for s in spec.Storage], "family x storage")))
    reasons: dict[str, int] = {}
    for c in cells:
        if c.verdict is Verdict.DNR:
            reasons[c.reason] = reasons.get(c.reason, 0) + 1
    print(f"\ndid-not-run (not a distinct cell): {n[Verdict.DNR]} cells, by reason:")
    for reason, cnt in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {cnt:>4}  {reason}")
    print(
        f"\nPROOF: {n[Verdict.COVERED]} cells COVERED (a receipt banks every id, with the sub-path's PassTag "
        f"asserted), {n[Verdict.BOUND]} BOUND (fixture and path present, no receipt) from "
        f"{len(receipt.merged())} banked id(s) in {receipt.receipts_dir()}{_claimed_note()}"
    )
    print(f"\nGAPS (uncovered, --check fails): {n[Verdict.GAP]} cells. What is missing:\n")
    print("\n".join("  " + line if line else "" for line in render_missing()))
    orphans = fixture_gaps()
    if orphans:
        print(f"\norphan fixtures ({len(orphans)}): on disk, no cell binds them:")
        for name in orphans:
            print(f"  {name}")
    print(f"\nENGINE FORKS NO CELL CERTIFIES ({len(UNTAGGED_FORKS)} untagged, {len(FORK_NOTES)} tagged):")
    for fork in UNTAGGED_FORKS:
        print(f"  [{fork.name}] {fork.where} - selected by {fork.selects}")
        print(f"      {fork.why}")
    for tag, note in FORK_NOTES.items():
        print(f"  [{tag.value}] {note}")
    for tag in unnoted_tags():
        print(f"  ! {tag.value}: a PassTag no sub-path asserts and FORK_NOTES does not explain")
    for tag in noted_but_asserted():
        print(f"  ! {tag.value}: a sub-path asserts this now; its FORK_NOTES entry is stale")

    # Part B
    st = target_status()
    present = [t for t, ok in st if ok]
    missing = [t for t, ok in st if not ok]
    print(f"\nB. REAL-MODEL PREREQS (cached-only, never committed): {len(present)}/{len(st)} cached")
    for t in missing:
        print(
            f"  [ ] {t.spec:<52} {t.kind.value} {t.params}{' MoE' if t.moe else ''}  "
            f"{t.tests or 'no test yet - uncovered path'}"
        )
    qgap = sorted(SUPPORTED_QUANTS - covered_quants(), key=lambda q: q.name)
    if qgap:
        print(f"\nquants supported but exercised by nothing ({len(qgap)}): " + ", ".join(q.name for q in qgap))
        print("  (i-quant lattices Unsloth ships only inside UD-* mixes)")
    return 0


def check(strict: bool = False) -> int:
    """the blocking gate: nonzero while any real path is uncovered. It is MEANT to fail - a gap, a cell no
    receipt proves, an unexplained fork and a stray fixture are all things to close, never to soften."""
    cells = compute_cells()
    problems = core.consistency_problems() + spec.totality_problems()
    reasonless = [c for c in cells if c.verdict is Verdict.DNR and not c.reason]
    gaps = sorted({(c.kind.value, c.storage.value) for c in cells if c.verdict is Verdict.GAP})
    unproven = sorted({i for c in cells if c.verdict is Verdict.BOUND for i in c.ids} - receipt.merged())
    for p in problems:
        print(f"FAIL core drift: {p}", file=sys.stderr)
    if reasonless:
        print(f"FAIL: {len(reasonless)} DNR cells without a reason", file=sys.stderr)
    for kind, storage in gaps:
        print(f"FAIL gap: {kind} :: {storage}", file=sys.stderr)
    for cid in unproven:
        print(f"FAIL unproven: {cid} is runnable but no receipt banks it", file=sys.stderr)
    orphans = fixture_gaps()
    for name in orphans:
        print(f"FAIL orphan fixture: {name} is on disk but no cell binds it", file=sys.stderr)
    for fork in UNTAGGED_FORKS:
        print(f"FAIL untagged fork: {fork.name} ({fork.where}) - {fork.why}", file=sys.stderr)
    for tag, note in FORK_NOTES.items():
        print(f"FAIL uncertified fork: {tag.value} - {note}", file=sys.stderr)
    for tag in unnoted_tags():
        print(f"FAIL unexplained fork: {tag.value} is asserted by no sub-path and noted nowhere", file=sys.stderr)
    for tag in noted_but_asserted():
        print(f"FAIL stale fork note: {tag.value} is asserted by a sub-path but still in FORK_NOTES", file=sys.stderr)
    missing = [t for t, ok in target_status() if not ok] if strict else []
    for t in missing:
        print(f"FAIL prereq (strict): {t.spec} not cached", file=sys.stderr)
    return (
        1 if (problems or reasonless or gaps or unproven or orphans or UNTAGGED_FORKS or FORK_NOTES or missing) else 0
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--report", action="store_true", help="print the grids, DNR breakdown, and real-model prereqs (default)"
    )
    ap.add_argument(
        "--missing",
        action="store_true",
        help="print only the plain-language gaps: what is missing and how to close each",
    )
    ap.add_argument("--check", action="store_true", help="exit nonzero on any gap, reasonless DNR, or core drift")
    ap.add_argument("--strict", action="store_true", help="--check, and also fail if a real-model target is uncached")
    a = ap.parse_args(argv)
    if a.check or a.strict:
        return check(strict=a.strict)
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
