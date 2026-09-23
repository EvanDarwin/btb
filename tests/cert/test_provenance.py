"""The pass-provenance system (btb.kinds.PassTag / PassReport, StreamedTextModel.last_pass_report): a lint that
every forward entry point records a tag, and that the real forks emit the tag of the path they took on the tiny
fixtures. The cert asserts a decode pass ran the intended code path from these tags, so a silent fallback to
torch or the per-op path is caught instead of certified."""

from __future__ import annotations

import ast
import os
import re

import pytest

import btb
from btb.engine.native import Native
from btb.kinds import PassReport, PassTag
from tests.helpers import FIXTURES, loaded_model

BTB = os.path.dirname(os.path.abspath(btb.__file__))  # the package under test: the checkout's, or the wheel's
_ENGINE = os.path.join(BTB, "engine")
QWEN3 = os.path.join(FIXTURES, "tiny_qwen3")

_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+_forward", re.M)  # the count the AST walk must reproduce
_TAG_RE = re.compile(r"\bPassTag\.([A-Z][A-Z_0-9]*)")  # a tag the engine records (its declaration names no prefix)

Forward = tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]  # (engine module, a _forward* entry point)


def _engine_sources() -> list[tuple[str, str]]:
    """(module, source) for every btb/engine/*.py, so a forward path in a module nobody listed is still linted"""
    out = []
    for name in sorted(os.listdir(_ENGINE)):
        if name.endswith(".py"):
            with open(os.path.join(_ENGINE, name), encoding="utf-8") as f:
                out.append((name, f.read()))
    return out


def forward_entries() -> list[Forward]:
    """every `def _forward*` under btb/engine - the compute entry points, whichever module holds them"""
    out: list[Forward] = []
    for name, src in _engine_sources():
        for node in ast.walk(ast.parse(src, filename=name)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("_forward"):
                out.append((name, node))
    return out


def _is_declaration(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """whether `fn` only declares the entry point for a mixin to implement (`raise NotImplementedError`, `...`):
    it takes no compute path, so it records no tag."""
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if not body:
        return True
    exc = body[0].exc if len(body) == 1 and isinstance(body[0], ast.Raise) else None
    name = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(name, ast.Name) and name.id == "NotImplementedError"


def _tags_self_tag(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """whether `fn`'s body calls self._tag (the compute-path record); a nested helper's call counts too"""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_tag"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            return True
    return False


def test_every_forward_entry_records_a_tag() -> None:
    """AST lint: every implemented `def _forward*` anywhere under btb/engine records a `self._tag(...)`, so a NEW
    forward path - in forward.py, mlx_forward.py, cuda.py or a module that does not exist yet - fails here until
    it is tagged. AST over a runtime check because it needs no hardware and fails on an untagged path even when
    this machine cannot exercise it (the CUDA forwards never run on the Mac)."""
    entries = forward_entries()
    spelled = sum(len(_DEF_RE.findall(src)) for _name, src in _engine_sources())
    assert entries and len(entries) == spelled, f"walked {len(entries)} forward entries, the source spells {spelled}"
    untagged = [f"{m}:{fn.lineno} {fn.name}" for m, fn in entries if not _is_declaration(fn) and not _tags_self_tag(fn)]
    assert not untagged, "these forward entry points record no PassTag (add self._tag at the fork):\n  " + "\n  ".join(
        untagged
    )


def test_every_pass_tag_is_emitted() -> None:
    """the reverse lint: every PassTag member is recorded somewhere under btb/, so a tag nothing produces - a fork
    that was removed, or one declared and never wired - is a reported gap rather than a member no report can carry."""
    used: set[str] = set()
    for root, _dirs, files in os.walk(BTB):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    used |= set(_TAG_RE.findall(f.read()))
    orphans = sorted(t.name for t in PassTag if t.name not in used)
    assert not orphans, f"these PassTag members are never recorded in btb/ (wire the fork, or drop them): {orphans}"


def test_pass_report_is_immutable_and_empty_by_default() -> None:
    r = PassReport()
    assert r.tags == frozenset() and r.spec_proposed == 0 and r.spec_accepted == 0
    assert PassTag.MLX_MEGA not in r
    with pytest.raises(AttributeError):
        r.tags = frozenset({PassTag.MLX_MEGA})  # type: ignore[misc]


def test_cpu_native_and_sampling_tags() -> None:
    """the CPU tier decodes through the native gemv kernel (CPU_NATIVE) on the host tier (TIER_HOST); greedy vs a
    temperature picks SAMPLE_GREEDY vs SAMPLE_STOCHASTIC; the plain loop is SPEC_OFF"""
    if not os.path.isdir(QWEN3):
        pytest.skip("tiny_qwen3 not built")
    with loaded_model(QWEN3, device="cpu") as sm:
        if Native.gemv is None:
            pytest.skip("the native CPU gemv kernel is not built here")  # the load binds it if it exists
        sm.generate([1, 2, 3, 4], 6, speculate=False)
        greedy = sm.last_pass_report()
        assert {PassTag.CPU_NATIVE, PassTag.TIER_HOST, PassTag.SPEC_OFF, PassTag.SAMPLE_GREEDY} <= greedy.tags
        assert PassTag.SAMPLE_STOCHASTIC not in greedy

        sm.generate([1, 2, 3, 4], 6, speculate=False, sampling=btb.Sampling(temperature=0.8, seed=1))
        sampled = sm.last_pass_report()
        assert PassTag.SAMPLE_STOCHASTIC in sampled and PassTag.SAMPLE_GREEDY not in sampled
        assert PassTag.CPU_NATIVE in sampled


@pytest.mark.skipif(not btb.mlx_available(), reason="MLX is not available on this machine")
def test_mlx_mega_and_step_tags() -> None:
    """qwen3 on MLX: greedy decode runs the megakernel (MLX_MEGA) when the load built it, else the fused step
    (MLX_STEP); with mlx_mega=0 it is always the step and never the megakernel. A bf16 checkpoint is the
    dequantized path, QUANT_DEQUANT. The tiny fixtures carry head_dim 16, below the megakernel's 64-multiple, so
    MLX_MEGA is only reachable on a real model here - the step path is what the fixtures exercise."""
    if not os.path.isdir(QWEN3):
        pytest.skip("tiny_qwen3 not built")
    with loaded_model(QWEN3, device="mlx") as sm:
        built = sm._mega is not None
        sm.generate([1, 2, 3, 4], 6, speculate=False)
        default = sm.last_pass_report()
    want = PassTag.MLX_MEGA if built else PassTag.MLX_STEP
    assert want in default, f"expected {want.value}, got {sorted(t.value for t in default.tags)}"
    assert PassTag.QUANT_DEQUANT in default and PassTag.SAMPLE_GREEDY in default

    with loaded_model(QWEN3, device="mlx", mlx_mega=0) as sm:
        assert sm._mega is None  # mlx_mega=0 forbids the megakernel outright
        sm.generate([1, 2, 3, 4], 6, speculate=False)
        step = sm.last_pass_report()
    assert PassTag.MLX_STEP in step, f"expected MLX_STEP, got {sorted(t.value for t in step.tags)}"
    assert PassTag.MLX_MEGA not in step


def test_streamed_head_tag() -> None:
    """`resident_head=0` streams the head from the checkpoint (HEAD_STREAMED); held, the same decode records
    HEAD_RESIDENT. The tokens are the same either way - this is which multiply produced them."""
    if not os.path.isdir(QWEN3):
        pytest.skip("tiny_qwen3 not built")
    tags = []
    for flag in (0, 1):
        with loaded_model(QWEN3, device="cpu", resident_head=flag) as sm:
            sm.generate([1, 2, 3, 4], 4, speculate=False)
            tags.append(sm.last_pass_report().tags)
    assert PassTag.HEAD_STREAMED in tags[0] and PassTag.HEAD_RESIDENT not in tags[0]
    assert PassTag.HEAD_RESIDENT in tags[1] and PassTag.HEAD_STREAMED not in tags[1]


def test_expert_store_and_mxfp4_tags() -> None:
    """gpt-oss's MXFP4 experts: the store serves them (EXPERT_STORE) and their blocks are multiplied as stored
    (EXPERT_MXFP4_ASSTORED) - `mlx_state.affine` never sees them, which is why the pass reports QUANT_DEQUANT.
    The residency policy follows `bus_pass`: the documented default of 1 is the Bus Pass, 0 the plain line."""
    path = os.path.join(FIXTURES, "gguf", "tiny_gpt_oss-mxfp4.gguf")
    if not os.path.isfile(path):
        pytest.skip("tiny_gpt_oss-mxfp4.gguf not built")
    for flag, policy in ((0, PassTag.EXPERT_LINE), (1, PassTag.EXPERT_BUS_PASS)):
        with loaded_model(path, device="cpu", bus_pass=flag) as sm:
            if sm.expert_store is None:
                pytest.skip("this build has no direct reader, so the experts run off the checkpoint's tables")
            sm.generate([1, 2, 3, 4], 4, speculate=False)
            tags = sm.last_pass_report().tags
        want = {PassTag.EXPERT_STORE, PassTag.EXPERT_MXFP4_ASSTORED, policy}
        assert want <= tags, sorted(t.value for t in tags)
        assert PassTag.EXPERT_TABLES not in tags and PassTag.EXPERT_MXFP4_DEQUANT not in tags


@pytest.mark.skipif(not btb.mlx_available(), reason="MLX is not available on this machine")
def test_mlx_mxfp4_experts_are_as_stored() -> None:
    """the same on MLX, where the experts go through the GPU's MXFP4 matvec over the store's slots: the
    `gguf-mxfp4 / mlx-packed` cell asserts EXPERT_MXFP4_ASSTORED because that is the binding the run takes."""
    path = os.path.join(FIXTURES, "gguf", "tiny_gpt_oss-mxfp4.gguf")
    if not os.path.isfile(path):
        pytest.skip("tiny_gpt_oss-mxfp4.gguf not built")
    with loaded_model(path, device="mlx", gguf_packed=1) as sm:
        sm.generate([1, 2, 3, 4], 4, speculate=False)
        tags = sm.last_pass_report().tags
    assert PassTag.EXPERT_MXFP4_ASSTORED in tags, sorted(t.value for t in tags)


@pytest.mark.skipif(not btb.mlx_available(), reason="MLX is not available on this machine")
def test_mlx_gguf_packed_vs_dequant_quant_tags() -> None:
    """an affine-quantized GGUF on MLX: read as stored it records QUANT_ASSTORED (the matvec kernels bind the
    packed bytes, mlx_state.affine), dequantized to bf16 it records QUANT_DEQUANT. A dense family (qwen3), so
    the affine linears actually reach the fused path rather than the MoE per-op path."""
    gdir = os.path.join(FIXTURES, "gguf")
    files = sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []
    affine = [f for f in files if f.startswith("tiny_qwen3-") and (f.endswith("q4_0.gguf") or f.endswith("q8_0.gguf"))]
    if not affine:
        pytest.skip("no affine-quant dense GGUF fixture built")
    path = os.path.join(gdir, affine[0])
    with loaded_model(path, device="mlx", gguf_packed=1) as sm:
        sm.generate([1, 2, 3, 4], 6, speculate=False)
        packed = sm.last_pass_report()
    assert PassTag.QUANT_ASSTORED in packed, f"expected QUANT_ASSTORED, got {sorted(t.value for t in packed.tags)}"

    with loaded_model(path, device="mlx", gguf_packed=0) as sm:
        sm.generate([1, 2, 3, 4], 6, speculate=False)
        dequant = sm.last_pass_report()
    assert PassTag.QUANT_DEQUANT in dequant and PassTag.QUANT_ASSTORED not in dequant
