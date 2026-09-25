"""The manifest as a suite gate. It derives families from core (`core.served_kinds`), so these hold for any
family core adds - a new one is iterated, and uncovered it appears in the pinned gap set below rather than
passing quietly. The blocking gate is `python -m tests.cert.manifest --check`, which stays red while any gap or
unproven cell is open; these tests hold the manifest's SHAPE: that COVERED means a receipt, that the axes are
still total against btb.kinds, and that the gap set is exactly the one the repo knows about."""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

from btb.kinds import QUANT_KIND, Cap, FamilyKind, PassTag, Proposer, Quant, QuantClass

from . import core, manifest, spec

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The gaps the repo knows it has, as `Missing` kind -> the families it affects. This is the intentionally-red
# half made explicit: `--check` fails on every one of them, and this test fails when the set MOVES - a new gap
# nobody wrote down, or a closed gap still claimed here. Update it in the same change that opens or closes one.
EXPECTED_GAPS: dict[manifest.Missing, list[str]] = {
    manifest.Missing.GGUF_LOAD: ["tiny_gemma3", "tiny_q35", "tiny_q4"],
    manifest.Missing.MXFP4_NON_GPTOSS: ["tiny_phi3", "tiny_qwen3"],
    manifest.Missing.GPTOSS_GGUF_TWIN: ["tiny_gpt_oss"],
    manifest.Missing.KIQUANT_FIXTURE: ["tiny_phi3", "tiny_qwen3"],
    manifest.Missing.MEGAKERNEL: ["tiny_gemma3", "tiny_gpt_oss", "tiny_phi3", "tiny_q35", "tiny_q4"],
    manifest.Missing.MEGA_STORAGE: ["tiny_qwen3"],
    manifest.Missing.MEGA_SHAPE: ["tiny_qwen3"],
    manifest.Missing.CARD_GRAPH_FAMILY: ["tiny_gpt_oss", "tiny_phi3", "tiny_q35", "tiny_q4"],
    manifest.Missing.CARD_GRAPH_SHAPE: ["tiny_gemma3", "tiny_qwen3"],
    manifest.Missing.SPEC_MTP_HEAD: ["tiny_gemma3", "tiny_gpt_oss", "tiny_phi3", "tiny_q4", "tiny_qwen3"],
    manifest.Missing.SPEC_DECODE: ["tiny_gemma3", "tiny_gpt_oss", "tiny_phi3", "tiny_q35", "tiny_q4", "tiny_qwen3"],
    manifest.Missing.FP8_UNIMPLEMENTED: [
        "tiny_gemma3", "tiny_gpt_oss", "tiny_phi3", "tiny_q35", "tiny_q4", "tiny_qwen3",
    ],
    manifest.Missing.QUANT_FIXTURE: ["tiny_phi3", "tiny_qwen3"],
    manifest.Missing.NO_RUNNER_CELL: [
        "tiny_gemma3", "tiny_gpt_oss", "tiny_phi3", "tiny_q35", "tiny_q4", "tiny_qwen3",
    ],
    manifest.Missing.NO_FIXTURE: ["tiny_phi3"],
}  # fmt: skip


def _source(*rel: str) -> str:
    """a btb file's text, read rather than imported: these guards must hold on a machine with no torch."""
    with open(os.path.join(core.BTB_SRC, *rel), encoding="utf-8") as f:
        return f.read()


def _receipts(tmp_path: Path, monkeypatch: MonkeyPatch, ids: list[str]) -> None:
    """point the manifest at a receipts directory holding exactly `ids` - the union CI assembles, in miniature."""
    (tmp_path / "machine.json").write_text(json.dumps({"machine": "test", "covered": ids}), encoding="utf-8")
    monkeypatch.setenv("BTB_CERT_RECEIPTS", str(tmp_path))


def test_manifest_computes() -> None:
    cells = manifest.compute_cells()
    assert cells, "the manifest produced no cells"
    assert any(c.verdict is manifest.Verdict.BOUND or c.verdict is manifest.Verdict.COVERED for c in cells), (
        "no cell is even bound - fixture binding is broken"
    )


def test_no_reasonless_dnr() -> None:
    """every impossible cell says why - a DNR is never a silent drop."""
    bad = [c for c in manifest.compute_cells() if c.verdict is manifest.Verdict.DNR and not c.reason]
    assert not bad, f"{len(bad)} DNR cells carry no reason"


def test_core_family_lists_agree() -> None:
    """the redundancy guard: SERVE_TYPES/SUPPORTED_MODEL_TYPES derive from kinds.KIND_OF, every served family has
    CAPS, and every GGUF arch target is served. A family half-declared trips this."""
    problems = core.consistency_problems()
    assert problems == [], "core family set is inconsistent:\n  " + "\n  ".join(problems)


def test_axes_are_total_against_core() -> None:
    """the axes still cover everything btb.kinds declares: every Quant has a storage member, every Proposer a
    decode path, every Hardware a sub-path. A new kernel or proposer lands here before it can be miscounted."""
    problems = spec.totality_problems()
    assert problems == [], "the cert axes have fallen behind btb.kinds:\n  " + "\n  ".join(problems)
    unbanked = {
        s
        for s, info in spec.STORAGE.items()
        if info.container is spec.Container.SAFETENSORS and s is not spec.Storage.SAFE_BF16
    }
    assert set(manifest.SAFE_PRECISION_GAP) == unbanked, "a safetensors precision with no gap kind saying why"


def test_every_quant_is_classified_and_exercised() -> None:
    """the quant drift guard (torch-free half): every kinds.Quant has a QUANT_KIND class (so a new member is
    forced into a bind path), and SUPPORTED_QUANTS covers exactly the enum - the manifest cannot fall behind a
    new quant kernel, because both derive from the one Quant declaration."""
    from gguf import GGMLQuantizationType as GQ

    unclassified = [q.value for q in Quant if q not in QUANT_KIND]
    assert not unclassified, f"Quant members with no QUANT_KIND class: {unclassified}"
    assert {q.name for q in manifest.SUPPORTED_QUANTS} == {q.name for q in Quant}
    assert manifest.SUPPORTED_QUANTS == frozenset(GQ[q.name] for q in Quant)


# --- COVERED means a receipt, never a file on disk ----------------------------------------------------------


def _cell(cells: list[manifest.Cell], storage: spec.Storage, device: str) -> manifest.Cell:
    got = [c for c in cells if c.kind is FamilyKind.QWEN3 and c.storage is storage and c.device == device]
    greedy = [c for c in got if c.decode is spec.DecodePath.GREEDY]
    assert len(greedy) == 1, f"expected one greedy cell for {storage.value}/{device}, got {len(greedy)}"
    return greedy[0]


def test_a_fixture_on_disk_is_bound_not_covered(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the keystone: with no receipt, a cell whose fixture is right there on disk is BOUND, not COVERED. This is
    the overclaim the manifest used to make - os.path.exists is presence, not proof."""
    _receipts(tmp_path, monkeypatch, [])
    cell = _cell(manifest.compute_cells(), spec.Storage.SAFE_BF16, "cpu")
    assert os.path.isdir(os.path.join(spec.FIXTURES, spec.FIXTURE_STEM[FamilyKind.QWEN3]))
    assert cell.verdict is manifest.Verdict.BOUND, cell
    assert cell.ids, "a bound cell names the ids that would prove it"


def test_a_cell_flips_to_covered_on_its_receipt(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """...and flips to COVERED exactly when every id it needs is in the receipts directory - a partial receipt
    (one sampler of two) is still unproven, because the cell is both runs."""
    ids = list(_cell(manifest.compute_cells(), spec.Storage.SAFE_BF16, "cpu").ids)
    assert len(ids) == 2, ids
    _receipts(tmp_path, monkeypatch, ids[:1])
    assert _cell(manifest.compute_cells(), spec.Storage.SAFE_BF16, "cpu").verdict is manifest.Verdict.BOUND
    _receipts(tmp_path, monkeypatch, ids)
    assert _cell(manifest.compute_cells(), spec.Storage.SAFE_BF16, "cpu").verdict is manifest.Verdict.COVERED


def test_a_receipt_cannot_buy_a_gap(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """a banked id for a cell the engine cannot engage (qwen3's megakernel on a head_dim-16 fixture) leaves the
    cell a GAP. Coverage is a verdict about the path, so a stale or fallback-earned id buys nothing."""
    stale = [manifest.safetensors_id(FamilyKind.QWEN3, "mlx-mega", d) for d in spec.Decode]
    _receipts(tmp_path, monkeypatch, stale)
    cell = _cell(manifest.compute_cells(), spec.Storage.SAFE_BF16, "mlx-mega")
    assert cell.verdict is manifest.Verdict.GAP and cell.reason == manifest.Missing.MEGA_SHAPE.value, cell


def test_ids_are_built_in_one_place() -> None:
    """the manifest's ids and the runner's recorded ids are the same strings because both call cell_id()."""
    assert manifest.safetensors_id(FamilyKind.QWEN3, "cpu", spec.Decode.GREEDY) == "safetensors/qwen3/cpu/greedy"
    assert manifest.gguf_id("tiny_qwen3-q4_0.gguf", "mlx-packed") == "gguf/tiny_qwen3-q4_0/mlx-packed"
    assert manifest.stem_id(spec.Surface.PACK12, "tiny_qwen3", "cpu") == "pack12/tiny_qwen3/cpu"
    assert manifest.runnable_ids() >= manifest.shape_ids()


def test_runnable_ids_exclude_gaps() -> None:
    """the union gate asks only for runs a machine could make: a gap cell's would-be id is not demanded, and a
    plain cpu cell's is."""
    ids = manifest.runnable_ids()
    assert manifest.safetensors_id(FamilyKind.QWEN3, "mlx-mega", spec.Decode.GREEDY) not in ids
    assert manifest.safetensors_id(FamilyKind.QWEN3, "cpu", spec.Decode.GREEDY) in ids


# --- the axes derive from btb.kinds, and name only real knobs -----------------------------------------------


def test_storage_derives_from_quant_kind() -> None:
    """every supported stored type belongs to exactly one storage member, the bind paths are not merged (IQ4 and
    the lattice i-quants are different kernel families) and the two float precisions are not either (BF16 is read
    straight off the file, F16 goes through dequant)."""
    claimed = [q for info in spec.STORAGE.values() for q in info.quants]
    assert sorted(claimed, key=lambda q: q.name) == sorted(Quant, key=lambda q: q.name)
    assert spec.quant_class(spec.Storage.GGUF_IQ4) is QuantClass.IQ4
    assert spec.quant_class(spec.Storage.GGUF_LATTICE) is QuantClass.LATTICE
    assert spec.STORAGE[spec.Storage.GGUF_BF16].quants == (Quant.BF16,)
    assert spec.STORAGE[spec.Storage.GGUF_F16].quants == (Quant.F16,)


def test_the_f16_gguf_twin_is_bound() -> None:
    """tests/fixtures/gguf/tiny_qwen3-f16.gguf: the runner ran it while no cell bound it. It binds GGUF_F16 now."""
    paths = spec.fixture_paths(FamilyKind.QWEN3, spec.Storage.GGUF_F16)
    assert [os.path.basename(p) for p in paths] == ["tiny_qwen3-f16.gguf"]
    assert all(os.path.exists(p) for p in paths)
    assert spec.storage_of_gguf("tiny_qwen3-f16.gguf") is spec.Storage.GGUF_F16


def test_a_precision_cell_binds_only_its_own_twin(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """every family's fp16/fp32 cell binds its twin, whose headers carry that precision and no other float - never
    the bf16 checkpoint - and a twin whose headers say otherwise binds nothing, whatever its name: the cell is then
    the precision's own gap kind."""
    for kind in spec.FIXTURE_STEM:
        for storage in (spec.Storage.SAFE_FP16, spec.Storage.SAFE_FP32):
            (path,) = spec.fixture_paths(kind, storage)
            assert path != spec.twin_path(spec.FIXTURE_STEM[kind], spec.Storage.SAFE_BF16)
            assert spec.header_dtypes(path) & spec.FLOAT_HEADERS == {spec.STORAGE[storage].fp}, path
    stem = spec.FIXTURE_STEM[FamilyKind.QWEN3]
    bf16 = spec.twin_path(stem, spec.Storage.SAFE_BF16)
    monkeypatch.setattr(spec, "FIXTURES", str(tmp_path))
    # a copy, not a symlink: Windows refuses symlinks without Developer Mode or admin (WinError 1314); the
    # fixture is a few hundred KB and only its headers are read
    shutil.copytree(bf16, spec.twin_path(stem, spec.Storage.SAFE_FP16))
    assert spec.fixture_paths(FamilyKind.QWEN3, spec.Storage.SAFE_FP16) == ()
    why = manifest.gap_reason(FamilyKind.QWEN3, spec.Storage.SAFE_FP16, spec.SUBPATH["cpu"], spec.DecodePath.GREEDY)
    assert why is manifest.Missing.FP16_FIXTURE


def test_decode_paths_derive_from_proposer() -> None:
    """one decode path per kinds.Proposer, plus greedy and the sibling draft model (which is not a proposer)."""
    assert {p for p in spec.DECODE_PROPOSER.values() if p is not None} == set(Proposer)
    assert len(spec.DecodePath) == len(Proposer) + 2
    assert spec.DECODE_KIND[spec.DecodePath.SPEC_DRAFT] is spec.DecodeKind.DRAFT_MODEL


def test_every_subpath_knob_is_a_real_load_option() -> None:
    """the cuda-cold lesson: `cold_layers` was a planner kwarg, not an option, so that sub-path raised
    UnknownOption on the one machine that could run it. Every knob is checked against options.KNOWN here."""
    from btb import options

    assert "cuda-cold" not in spec.SUBPATH, "cuda-cold is removed: cold_layers is not a load option"
    for dev in spec.DEVICE_SUBPATHS:
        unknown = sorted(set(dev.knobs) - set(options.KNOWN) - {"device"})
        assert not unknown, f"{dev.key} passes {unknown}, which btb.load does not take"
        assert "cold_layers" not in dev.knobs


def test_every_subpath_declares_its_tag() -> None:
    """a sub-path's expected PassTag travels with it, for every served family and storage - so a cell cannot be
    certified by whatever tag happened to be there."""
    for dev in spec.DEVICE_SUBPATHS:
        for kind in core.served_kinds():
            for storage in spec.Storage:
                assert isinstance(dev.expect(kind, storage), PassTag), f"{dev.key} has no tag for {kind.value}"
                assert dev.expect(kind, storage) in dev.expects(kind, storage)


def test_the_mlx_step_fork_is_tagged_per_family() -> None:
    """the fused/unfused fork: the families with the kernel or sandwich layout take the fused kernels, phi3's
    partial rotary and the MoE and hybrid families do not - and mlx-fp32 never does, its state is not bf16."""
    step, fp32 = spec.SUBPATH["mlx-step"], spec.SUBPATH["mlx-fp32"]
    bf16 = spec.Storage.SAFE_BF16
    assert step.expect(FamilyKind.QWEN3, bf16) is PassTag.MLX_STEP_FUSED
    assert step.expect(FamilyKind.GEMMA3, bf16) is PassTag.MLX_STEP_FUSED
    assert step.expect(FamilyKind.PHI3, bf16) is PassTag.MLX_STEP_UNFUSED
    assert step.expect(FamilyKind.QWEN3_5, bf16) is PassTag.MLX_HYBRID
    assert step.expect(FamilyKind.GPT_OSS, bf16) is PassTag.MLX_PEROP
    assert fp32.expect(FamilyKind.QWEN3, bf16) is PassTag.MLX_STEP_UNFUSED


def test_the_residency_fork_is_a_cell_per_hardware() -> None:
    """the expert store's `bus_pass`: the DEFAULT sub-paths expect the Bus Pass on a MoE family (the option's
    documented default, which now reaches the store), the riders variants the plain line, and on a family with
    no store the knob selects nothing at all - so those cells are a DNR, not a claim of coverage."""
    bf16 = spec.Storage.SAFE_BF16
    for hw, default, riders in (("cpu", "cpu", "cpu-riders"), ("mlx", "mlx-step", "mlx-riders")):
        assert spec.SUBPATH[riders].knobs == {"device": hw, "bus_pass": 0}
        assert PassTag.EXPERT_BUS_PASS in spec.SUBPATH[default].expects(FamilyKind.GPT_OSS, bf16)
        assert PassTag.EXPERT_LINE in spec.SUBPATH[riders].expects(FamilyKind.GPT_OSS, bf16)
        assert spec.SUBPATH[default].expects(FamilyKind.QWEN3, bf16) == {
            spec.SUBPATH[default].expect(FamilyKind.QWEN3, bf16)
        }
    assert {d.hardware for d in spec.DEVICE_SUBPATHS if d.needs is Cap.MOE} == set(spec.Hardware)
    dense = manifest.dnr(FamilyKind.QWEN3, bf16, spec.SUBPATH["cpu-riders"], spec.DecodePath.GREEDY)
    assert dense is not None and "moe" in dense
    assert manifest.dnr(FamilyKind.GPT_OSS, bf16, spec.SUBPATH["cpu-riders"], spec.DecodePath.GREEDY) is None


def test_no_pass_tag_goes_unexamined() -> None:
    """every PassTag is either asserted by a device sub-path or explained in FORK_NOTES - a new fork in the
    engine cannot be added without the manifest saying whether a cell reaches it - and never both, so a note
    that outlived its fix (the residency policy's was one) cannot go on reporting a certified fork as open."""
    assert manifest.unnoted_tags() == [], (
        f"PassTags no sub-path asserts and FORK_NOTES does not explain: {[t.value for t in manifest.unnoted_tags()]}"
    )
    assert manifest.noted_but_asserted() == [], (
        f"FORK_NOTES entries a sub-path now asserts: {[t.value for t in manifest.noted_but_asserted()]}"
    )


def test_uncovered_forks_are_named() -> None:
    """an untagged fork is one no PassTag marks, so a cell asserting it would be satisfied by the other branch.
    The three this list carried (the streamed head, the card prefill, the expert store's residency) all emit a
    tag now and are FORK_NOTES entries or cells; whatever is added back says where it is and what closes it."""
    for fork in manifest.UNTAGGED_FORKS:
        assert ".py:" in fork.where and fork.selects and fork.why, fork


def test_the_option_selected_forks_have_cells() -> None:
    """the forks a real load option selects are asserted by a sub-path, not merely noted: `resident_head` off
    streams the head, and an MXFP4 GGUF's experts bind as stored. A tag here that slid back into FORK_NOTES
    would mean a knob nothing certifies."""
    asserted = manifest.expected_tags()
    for tag in (PassTag.HEAD_STREAMED, PassTag.EXPERT_MXFP4_ASSTORED, PassTag.EXPERT_BUS_PASS, PassTag.EXPERT_LINE):
        assert tag in asserted and tag not in manifest.FORK_NOTES, tag
    assert spec.SUBPATH["cpu-headstream"].knobs["resident_head"] == 0
    packed = spec.SUBPATH["mlx-packed"].expect(FamilyKind.GPT_OSS, spec.Storage.GGUF_MXFP4)
    assert packed is PassTag.EXPERT_MXFP4_ASSTORED


# --- the replicated engine predicates still match their source ----------------------------------------------


def test_card_family_predicate_matches_cuda_source() -> None:
    """core.card_family_ok replicates cuda.py's family clause (the manifest is torch-free, so it cannot call
    it). This holds the two together: the capabilities named in `_card_family_ok` are exactly the ones here."""
    src = _source("engine", "cuda.py")
    body = src.split("def _card_family_ok", 1)[1].split("\n    def ", 1)[0]
    named = {c for c in Cap if f"self.fam.{c.value}" in body}
    assert named == core.CARD_FAMILY_CAPS, f"cuda.py's family clause names {sorted(c.value for c in named)}"
    assert f"D in {manifest.CARD_HEAD_DIMS}" in body, "the card's head widths moved; CARD_HEAD_DIMS is stale"
    rejected = sorted(k.value for k in core.served_kinds() if not core.card_family_ok(k))
    assert rejected == ["gpt_oss", "phi3", "qwen3_5", "qwen4"]


def test_mega_head_multiple_matches_its_source() -> None:
    """the megakernel's head gate, quoted from mega.py so MEGA_SHAPE cannot outlive the constraint."""
    assert f"self.hd % {manifest.MEGA_HEAD_MULTIPLE}" in _source("mlx", "mega.py")
    assert spec.head_dim(FamilyKind.QWEN3) == 16, "the tiny fixture changed; MEGA_SHAPE may no longer hold"


def test_mtp_head_is_read_from_the_fixture() -> None:
    """which fixture carries a drafting head is read from its weight map, not from a stem list."""
    assert spec.has_mtp_head(FamilyKind.QWEN3_5)
    assert not spec.has_mtp_head(FamilyKind.QWEN3)


def test_core_flags_are_capability_members() -> None:
    """core.flags hands back Cap members, so a consumer compares enum to enum instead of matching strings."""
    fl = core.flags(FamilyKind.QWEN3)
    assert Cap.KERNEL_LAYOUT in fl and all(isinstance(c, Cap) for c in fl)


# --- fixtures and gaps ---------------------------------------------------------------------------------------


def test_no_orphan_fixtures() -> None:
    """every fixture on disk is bound by a cell - a stale/unreferenced fixture (the fixture-side of a new kernel
    with no matrix row) fails here rather than lingering uncertified."""
    orphans = manifest.fixture_gaps()
    assert orphans == [], f"{len(orphans)} orphan fixtures: {orphans}"


def test_fixture_gaps_sees_an_unbound_gguf_twin(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the orphan check is the exact set of paths the cells bind, not a stem prefix: a twin whose stored type no
    storage member stands for is an orphan even though its stem is a served family's."""
    gguf = tmp_path / "gguf"
    gguf.mkdir()
    (gguf / spec.gguf_name("tiny_qwen3", Quant.BF16)).write_bytes(b"")
    (gguf / "tiny_qwen3-q9_9.gguf").write_bytes(b"")
    for mod in (spec, manifest):
        monkeypatch.setattr(mod, "FIXTURES", str(tmp_path))
        monkeypatch.setattr(mod, "GGUF_DIR", str(gguf))
    assert manifest.fixture_gaps() == ["gguf/tiny_qwen3-q9_9.gguf"]


def test_the_known_gaps_are_exactly_these() -> None:
    """the intentionally-red set, pinned: `--check` fails on every kind below, and this fails if the set moves -
    a gap nobody recorded, or one closed without being struck from the list."""
    got = {kind: fams for kind, _n, fams in manifest.missing_items()}
    assert got == EXPECTED_GAPS
    # a precision's gap kind is idle while every family has that twin (test_a_precision_cell_binds_only_its_own_twin
    # holds it live for a family that loses one)
    idle = {
        gap
        for storage, gap in manifest.SAFE_PRECISION_GAP.items()
        if all(spec.fixture_paths(kind, storage) for kind in spec.FIXTURE_STEM)
    }
    assert set(got) | idle == set(manifest.Missing), "a Missing kind that no cell uses, or a cell kind not pinned here"


def test_the_gate_stays_red_while_gaps_are_open() -> None:
    """--check is the blocking gate and is MEANT to fail while the set above is non-empty; it must never be
    softened into a pass. (It writes its FAIL lines to stderr, which pytest captures.)"""
    assert manifest.check() == 1
