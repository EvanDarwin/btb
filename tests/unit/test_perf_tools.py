# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The two perf tools that turn benchmark JSON into a verdict: bench/e2e_delta.py (two pytest-benchmark runs ->
{id, delta, lo, hi}) and .github/scripts/bench_compare.py (those deltas -> the PR comment and the gate's exit
code). Synthetic stats only - no timing runs here, so the band math and the gate are pinned exactly. Torch-free."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from types import ModuleType

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(name: str, *rel: str) -> ModuleType:
    """a script that is not part of a package (bench/, .github/scripts/), loaded the way `python <path>` does."""
    path = os.path.join(ROOT, *rel)
    if not os.path.exists(path):
        pytest.skip(f"{os.path.join(*rel)} is not in this checkout")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)  # bench_compare imports tests.cert from the repo root
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def ed() -> ModuleType:
    return _load("e2e_delta", "bench", "e2e_delta.py")


@pytest.fixture(scope="module")
def bc() -> ModuleType:
    return _load("bench_compare", ".github", "scripts", "bench_compare.py")


def _bench(name: str, median: float, stddev: float, rounds: int, **rest: object) -> dict[str, object]:
    """one benchmark the way pytest-benchmark writes it into --benchmark-json"""
    return {"fullname": name, "name": name, "stats": {"median": median, "stddev": stddev, "rounds": rounds}, **rest}


def _doc(*benches: dict[str, object], isa: str = "") -> dict[str, object]:
    return {"benchmarks": list(benches), "machine_info": {"isa": isa}}


# --- e2e_delta: the band, the skips, the ids ------------------------------------------------------------------


def test_the_band_is_the_standard_error_of_the_median(ed: ModuleType) -> None:
    """the point estimate is a ratio of MEDIANS, so the 95% half-width is 1.96 * sqrt(pi/2) * SEM, not 1.96 * SEM:
    under the normal approximation the median's standard error is sqrt(pi/2) times the mean's."""
    base = _doc(_bench("b", median=100.0, stddev=10.0, rounds=100))
    pr = _doc(_bench("b", median=100.0, stddev=0.0, rounds=100))
    (e,) = ed.deltas(base, pr)
    sem = (10.0 / math.sqrt(100)) / 100.0  # relative standard error of the mean: 1%
    assert e["hi"] == pytest.approx(1.96 * math.sqrt(math.pi / 2) * sem)
    assert e["lo"] == pytest.approx(-e["hi"])
    assert e["hi"] / (1.96 * sem) == pytest.approx(math.sqrt(math.pi / 2))  # the widening, stated


def test_the_two_runs_combine_in_quadrature(ed: ModuleType) -> None:
    base = _doc(_bench("b", median=100.0, stddev=10.0, rounds=100))
    pr = _doc(_bench("b", median=200.0, stddev=40.0, rounds=100))
    (e,) = ed.deltas(base, pr)
    assert e["delta"] == pytest.approx(1.0)  # 100 -> 200 is +100%
    half = 1.96 * math.sqrt(math.pi / 2) * math.hypot(0.01, 0.02)
    assert (e["lo"], e["hi"]) == pytest.approx((1.0 - half, 1.0 + half))


def test_a_zero_or_negative_baseline_median_is_skipped(ed: ModuleType) -> None:
    """nothing to divide by - the ratio is undefined, so the benchmark drops rather than reading as a change"""
    for bad in (0.0, -1.0):
        base = _doc(_bench("b", median=bad, stddev=1.0, rounds=10))
        pr = _doc(_bench("b", median=5.0, stddev=1.0, rounds=10))
        assert ed.deltas(base, pr) == []


def test_two_isa_tiers_carry_through_and_void_the_delta(ed: ModuleType) -> None:
    """the tier comes from each run's machine_info (bench/conftest.py); two tiers are two kernels, not a delta"""
    base = _doc(_bench("b", median=100.0, stddev=1.0, rounds=100), isa="avx2")
    (e,) = ed.deltas(base, _doc(_bench("b", median=200.0, stddev=1.0, rounds=100), isa="avx2"))
    assert e["delta"] == pytest.approx(1.0) and (e["isa"], e["baseline_isa"]) == ("avx2", "avx2")
    (e,) = ed.deltas(base, _doc(_bench("b", median=200.0, stddev=1.0, rounds=100), isa="avx512"))
    assert e["delta"] is None and (e["isa"], e["baseline_isa"]) == ("avx512", "avx2")
    (e,) = ed.deltas(base, _doc(_bench("b", median=200.0, stddev=1.0, rounds=100)))  # unrecorded: compared
    assert e["delta"] == pytest.approx(1.0)


def test_a_benchmark_in_only_one_run_is_dropped(ed: ModuleType) -> None:
    base = _doc(_bench("both", 1.0, 0.0, 10), _bench("gone", 1.0, 0.0, 10))
    pr = _doc(_bench("both", 1.0, 0.0, 10), _bench("added", 1.0, 0.0, 10))
    assert [e["id"] for e in ed.deltas(base, pr)] == ["both"]


@pytest.mark.parametrize(
    "extra,want_id,want_section",
    [
        ({"params": {"device": "cpu", "model": "tiny_qwen3"}, "group": "cpu"}, "cpu/tiny_qwen3", "cpu"),
        ({"params": {"kind": "q4k", "t": "1"}, "group": "mlx-quant"}, "mlx-quant/q4k-1", "mlx-quant"),
        ({"params": {"kind": "q4k"}}, "q4k", ""),
        ({}, "test_thing", ""),
    ],
)
def test_the_id_is_the_group_and_the_param_label(
    ed: ModuleType, extra: dict[str, object], want_id: str, want_section: str
) -> None:
    """a label that already names its device carries the section (`cpu/tiny_qwen3`); anything else is prefixed by
    its bench group, so bench_compare drops it into the right collapsible section."""
    base = _doc(_bench("test_thing", 1.0, 0.0, 10, **extra))
    pr = _doc(_bench("test_thing", 1.0, 0.0, 10, **extra))
    (e,) = ed.deltas(base, pr)
    assert (e["id"], e["section"]) == (want_id, want_section)


# --- bench_compare: the gate and the comment ------------------------------------------------------------------


def test_the_gate_trips_only_when_the_whole_ci_clears_the_threshold(bc: ModuleType) -> None:
    """strictly above: a CI lower bound sitting exactly on the line is not a regression, and one straddling zero
    is runner noise, not a slowdown"""
    entries = [
        {"id": "over", "delta": 0.20, "lo": 0.11, "hi": 0.30},
        {"id": "on-the-line", "delta": 0.20, "lo": 0.10, "hi": 0.30},
        {"id": "straddles", "delta": 0.20, "lo": -0.01, "hi": 0.41},
        {"id": "faster", "delta": -0.20, "lo": -0.30, "hi": -0.11},
    ]
    assert [e["id"] for e in bc.regressions(entries, 0.10)] == ["over"]


def test_a_new_benchmark_never_trips_the_gate(bc: ModuleType) -> None:
    """delta=None is a bench the baseline has never seen; its lo/hi are placeholders, not a measured slowdown"""
    entries = [{"id": "brand-new", "delta": None, "lo": 99.0, "hi": 99.0}]
    assert bc.regressions(entries, 0.10) == []


def test_section_reads_the_device_then_the_op_family(bc: ModuleType) -> None:
    """both vocabularies are derived (spec.Hardware, native_ops.op_families), so this checks the matching rule,
    not the lists: an exact head, a `<family>_<variant>` head, then the head itself as its own section."""
    device, family = bc.DEVICES[0], sorted(bc.OP_FAMILIES)[0]
    assert bc._section(f"{device}/tiny_qwen3") == device
    assert bc._section(f"{family}/b16") == family
    assert bc._section(f"{family}_bf16/b16") == family
    assert bc._section("unheard-of/x") == "unheard-of"


def test_render_marks_the_comment_and_groups_into_sections(bc: ModuleType) -> None:
    """the sticky comment's marker (how the workflow finds the comment to update) and one <details> per section,
    open with a 🔴 where a regression cleared the noise floor"""
    entries = [
        {"id": "cpu/tiny_qwen3", "section": "cpu", "delta": 0.30, "lo": 0.25, "hi": 0.35},
        {"id": "cpu/tiny_phi3", "section": "cpu", "delta": 0.001, "lo": -0.02, "hi": 0.02},
        {"id": "gemv/b16", "delta": -0.30, "lo": -0.35, "hi": -0.25},
    ]
    body, regressed = bc.render(entries, 0.05)
    assert regressed is True
    assert body.startswith(bc.MARKER)
    assert "3 benchmarks compared" in body
    assert "<details open><summary>cpu · 2 benches" in body  # the regressed section is open
    assert "<details><summary>gemv · 1 benches" in body  # the clean one stays collapsed
    assert "🔴 slower" in body and "🟢 faster" in body and "≈ noise" in body


def test_render_says_so_when_there_is_nothing_to_compare(bc: ModuleType) -> None:
    body, regressed = bc.render([], 0.05)
    assert regressed is False
    assert bc.MARKER in body and "_No benchmark results found._" in body


def test_a_tier_mismatch_voids_every_row_and_the_gate(bc: ModuleType) -> None:
    """the e2e entries carry the two tiers; a mismatch voids the criterion rows too (same CPU) so nothing is
    compared and nothing can trip the gate; a matching tier is just stated"""
    criterion = {"id": "gemv/b16", "delta": 0.30, "lo": 0.25, "hi": 0.35}
    e2e = {"id": "cpu/tiny_qwen3", "section": "cpu", "delta": 0.30, "lo": 0.25, "hi": 0.35}
    same = bc.comparable([criterion, {**e2e, "isa": "avx2", "baseline_isa": "avx2"}])
    assert [e["delta"] for e in same] == [0.30, 0.30]
    body, regressed = bc.render(same, 0.05)
    assert regressed and "at ISA tier `avx2`" in body
    voided = bc.comparable([criterion, {**e2e, "delta": None, "isa": "avx512", "baseline_isa": "avx2"}])
    assert [e["delta"] for e in voided] == [None, None]
    assert bc.regressions(voided, 0.10) == []
    body, regressed = bc.render(voided, 0.05)
    assert not regressed and "ran at ISA tier `avx512`, the main baseline's at `avx2`" in body
    assert "not comparable" in body and "🔴" not in body
    body, _ = bc.render([criterion], 0.05)
    assert "ISA tier" not in body


def test_a_new_benchmark_renders_as_new(bc: ModuleType) -> None:
    body, regressed = bc.render([{"id": "gemv/b16", "delta": None, "lo": 0.0, "hi": 0.0}], 0.05)
    assert regressed is False
    assert "_new_" in body and "0 benchmarks compared" in body
