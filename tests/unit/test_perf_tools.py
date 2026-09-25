# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The two perf tools that turn benchmark JSON into a verdict: bench/e2e_delta.py (two pytest-benchmark runs ->
{id, delta, lo, hi}) and bench/report.py (those deltas -> the PR comment and the gate's exit code). Synthetic
stats only - no timing runs here, so the band math and the gate are pinned exactly. Torch-free."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from pytest import CaptureFixture

from bench import e2e_delta, report
from bench.e2e_delta import Bench, Doc
from bench.report import Entry


def _bench(
    name: str, median: float, stddev: float, rounds: int, params: dict[str, str] | None = None, group: str = ""
) -> Bench:
    """one benchmark the way pytest-benchmark writes it into --benchmark-json"""
    b: Bench = {"fullname": name, "name": name, "stats": {"median": median, "stddev": stddev, "rounds": rounds}}
    if params is not None:
        b["params"] = params
    if group:
        b["group"] = group
    return b


def _doc(*benches: Bench, isa: str = "") -> Doc:
    return {"benchmarks": list(benches), "machine_info": {"isa": isa}}


# --- e2e_delta: the band, the skips, the ids ------------------------------------------------------------------


def test_the_band_is_the_standard_error_of_the_median() -> None:
    """the point estimate is a ratio of MEDIANS, so the 95% half-width is 1.96 * sqrt(pi/2) * SEM, not 1.96 * SEM:
    under the normal approximation the median's standard error is sqrt(pi/2) times the mean's."""
    base = _doc(_bench("b", median=100.0, stddev=10.0, rounds=100))
    pr = _doc(_bench("b", median=100.0, stddev=0.0, rounds=100))
    (e,) = e2e_delta.deltas(base, pr)
    sem = (10.0 / math.sqrt(100)) / 100.0  # relative standard error of the mean: 1%
    assert e["hi"] == pytest.approx(1.96 * math.sqrt(math.pi / 2) * sem)
    assert e["lo"] == pytest.approx(-e["hi"])
    assert e["hi"] / (1.96 * sem) == pytest.approx(math.sqrt(math.pi / 2))  # the widening, stated


def test_the_two_runs_combine_in_quadrature() -> None:
    base = _doc(_bench("b", median=100.0, stddev=10.0, rounds=100))
    pr = _doc(_bench("b", median=200.0, stddev=40.0, rounds=100))
    (e,) = e2e_delta.deltas(base, pr)
    assert e["delta"] == pytest.approx(1.0)  # 100 -> 200 is +100%
    half = 1.96 * math.sqrt(math.pi / 2) * math.hypot(0.01, 0.02)
    assert (e["lo"], e["hi"]) == pytest.approx((1.0 - half, 1.0 + half))


def test_a_zero_or_negative_baseline_median_has_no_delta() -> None:
    """nothing to divide by - the ratio is undefined, so the benchmark lists as new rather than as a change"""
    for bad in (0.0, -1.0):
        base = _doc(_bench("b", median=bad, stddev=1.0, rounds=10))
        pr = _doc(_bench("b", median=5.0, stddev=1.0, rounds=10))
        (e,) = e2e_delta.deltas(base, pr)
        assert e["delta"] is None


def test_the_pr_runs_isa_tier_carries_through() -> None:
    """the tier comes from the PR run's machine_info (bench/conftest.py), for the report to state"""
    base = _doc(_bench("b", median=100.0, stddev=1.0, rounds=100), isa="avx2")
    (e,) = e2e_delta.deltas(base, _doc(_bench("b", median=200.0, stddev=1.0, rounds=100), isa="avx512"))
    assert e["delta"] == pytest.approx(1.0) and e["isa"] == "avx512"


def test_a_new_benchmark_lists_and_a_gone_one_drops() -> None:
    base = _doc(_bench("both", 1.0, 0.0, 10), _bench("gone", 1.0, 0.0, 10))
    pr = _doc(_bench("both", 1.0, 0.0, 10), _bench("added", 1.0, 0.0, 10))
    assert [(e["id"], e["delta"]) for e in e2e_delta.deltas(base, pr)] == [("added", None), ("both", 0.0)]


@pytest.mark.parametrize(
    "params,group,want_id,want_section",
    [
        ({"device": "cpu", "model": "tiny_qwen3"}, "cpu", "cpu/tiny_qwen3", "cpu"),
        ({"kind": "q4k", "t": "1"}, "mlx-quant", "mlx-quant/q4k-1", "mlx-quant"),
        ({"kind": "q4k"}, "", "q4k", ""),
        (None, "", "test_thing", ""),
    ],
)
def test_the_id_is_the_group_and_the_param_label(
    params: dict[str, str] | None, group: str, want_id: str, want_section: str
) -> None:
    """a label that already names its device carries the section (`cpu/tiny_qwen3`); anything else is prefixed by
    its bench group, so the report drops it into the right collapsible section."""
    base = _doc(_bench("test_thing", 1.0, 0.0, 10, params, group))
    pr = _doc(_bench("test_thing", 1.0, 0.0, 10, params, group))
    (e,) = e2e_delta.deltas(base, pr)
    assert (e["id"], e["section"]) == (want_id, want_section)


# --- report: the gate and the comment -------------------------------------------------------------------------


def _criterion(root: str, bench: str, median: float, half: float, run: str = "new") -> None:
    """one criterion bench's <run>/estimates.json under a result root, the way `cargo bench` lays it out"""
    d = os.path.join(root, "native", "target", "criterion", bench, run)
    os.makedirs(d, exist_ok=True)
    est = {
        "median": {
            "point_estimate": median,
            "confidence_interval": {"lower_bound": median - half, "upper_bound": median + half},
        }
    }
    with open(os.path.join(d, "estimates.json"), "w", encoding="utf-8") as f:
        json.dump(est, f)


def test_criterion_deltas_come_from_the_two_result_roots(tmp_path: Path) -> None:
    """the ratio of the two medians, the two 95% half-widths combined in quadrature (relative), the id is the
    bench's path under criterion/, a bench the baseline lacks is new, and report/ is skipped"""
    base, pr = str(tmp_path / "base"), str(tmp_path / "pr")
    _criterion(base, "gemv_bf16_group/neon/4", 100.0, 1.0)
    _criterion(pr, "gemv_bf16_group/neon/4", 110.0, 2.2)
    _criterion(pr, "attn/neon/4", 5.0, 0.1)
    _criterion(pr, "report", 1.0, 0.0)
    entries = {e["id"]: e for e in report._from_criterion(pr, base)}
    assert set(entries) == {"gemv_bf16_group/neon/4", "attn/neon/4"}
    e = entries["gemv_bf16_group/neon/4"]
    half = math.hypot(0.01, 0.02)
    assert e["delta"] == pytest.approx(0.10) and (e["lo"], e["hi"]) == pytest.approx((0.10 - half, 0.10 + half))
    assert entries["attn/neon/4"]["delta"] is None
    assert all(e["delta"] is None for e in report._from_criterion(pr, None))


def test_interleaved_pairs_fold_so_a_flag_needs_every_pair(tmp_path: Path) -> None:
    """runs a and b compare pair by pair: the delta is their mean and the band spans both, so a slowdown in one
    pair only (the runner drifting under it) reads as noise, while one in both is flagged and gated"""
    base, pr = str(tmp_path / "base"), str(tmp_path / "pr")
    for run, drifted in (("a", 130.0), ("b", 100.0)):
        _criterion(base, "delta_step/one_pair", 100.0, 0.1, run)
        _criterion(pr, "delta_step/one_pair", drifted, 0.1, run)
        _criterion(base, "delta_step/both", 100.0, 0.1, run)
        _criterion(pr, "delta_step/both", 130.0, 0.1, run)
    assert report.runs(pr) == ("a", "b")
    entries = {e["id"]: e for e in report.compare(pr, base)}
    one, both = entries["delta_step/one_pair"], entries["delta_step/both"]
    assert one["delta"] == pytest.approx(0.15) and one["lo"] < 0.05 < one["hi"]
    assert both["delta"] == pytest.approx(0.30) and both["lo"] > 0.10
    assert [e["id"] for e in report.regressions(list(entries.values()), 0.10)] == ["delta_step/both"]
    body, regressed = report.render(list(entries.values()), 0.05, len(report.runs(pr)))
    assert regressed and "2 pairs" in body and body.count("🔴 slower") == 1


def test_main_compares_two_roots_and_gates(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """the CLI CI runs: a result root each, one render for the comment and the exit code"""
    base, pr = str(tmp_path / "base"), str(tmp_path / "pr")
    _criterion(base, "gemv/b16", 100.0, 1.0)
    _criterion(pr, "gemv/b16", 130.0, 1.0)
    for root, median in ((base, 1.0), (pr, 1.0)):
        with open(os.path.join(root, "e2e.json"), "w", encoding="utf-8") as f:
            json.dump(_doc(_bench("cpu", median, 0.0, 10, {"device": "cpu", "model": "m"}, "cpu"), isa="neon"), f)
    out = str(tmp_path / "c.md")
    base_sha = "b" * 40
    args = [pr, "--baseline", base, "--base-ref", "main", "--base-sha", base_sha, "--gate", "--out", out]
    assert report.main(args) == 1
    body = open(out, encoding="utf-8").read()
    assert "`gemv/b16`" in body and "🔴 slower" in body and "`cpu/m`" in body and "at ISA tier `neon`" in body
    assert f"Compared against `main` at {base_sha}." in body
    assert report._against("main", "<img src=x>") == "", "only a plain hex SHA is named"
    assert "FAIL: 1 benchmark(s) regressed" in capsys.readouterr().err
    assert report.main([pr, "--gate"]) == 0  # no baseline: every row is new, nothing to gate


def test_the_gate_trips_only_when_the_whole_ci_clears_the_threshold() -> None:
    """strictly above: a CI lower bound sitting exactly on the line is not a regression, and one straddling zero
    is runner noise, not a slowdown"""
    entries: list[Entry] = [
        {"id": "over", "delta": 0.20, "lo": 0.11, "hi": 0.30},
        {"id": "on-the-line", "delta": 0.20, "lo": 0.10, "hi": 0.30},
        {"id": "straddles", "delta": 0.20, "lo": -0.01, "hi": 0.41},
        {"id": "faster", "delta": -0.20, "lo": -0.30, "hi": -0.11},
    ]
    assert [e["id"] for e in report.regressions(entries, 0.10)] == ["over"]


def test_a_new_benchmark_never_trips_the_gate() -> None:
    """delta=None is a bench the baseline has never seen; its lo/hi are placeholders, not a measured slowdown"""
    entries: list[Entry] = [{"id": "brand-new", "delta": None, "lo": 99.0, "hi": 99.0}]
    assert report.regressions(entries, 0.10) == []


def test_section_reads_the_device_then_the_op_family() -> None:
    """both vocabularies are derived (spec.Hardware, native_ops.op_families), so this checks the matching rule,
    not the lists: an exact head, a `<family>_<variant>` head, then the head itself as its own section."""
    device, family = report.DEVICES[0], sorted(report.OP_FAMILIES)[0]
    assert report._section(f"{device}/tiny_qwen3") == device
    assert report._section(f"{family}/b16") == family
    assert report._section(f"{family}_bf16/b16") == family
    assert report._section("unheard-of/x") == "unheard-of"


def test_render_marks_the_comment_and_groups_into_sections() -> None:
    """the sticky comment's marker (how the workflow finds the comment to update) and one <details> per section,
    open with a 🔴 where a regression cleared the noise floor"""
    entries: list[Entry] = [
        {"id": "cpu/tiny_qwen3", "section": "cpu", "delta": 0.30, "lo": 0.25, "hi": 0.35},
        {"id": "cpu/tiny_phi3", "section": "cpu", "delta": 0.001, "lo": -0.02, "hi": 0.02},
        {"id": "gemv/b16", "delta": -0.30, "lo": -0.35, "hi": -0.25},
    ]
    body, regressed = report.render(entries, 0.05)
    assert regressed is True
    assert body.startswith(report.MARKER)
    assert "3 benchmarks compared" in body
    assert "<details open><summary>cpu · 2 benches" in body  # the regressed section is open
    assert "<details><summary>gemv · 1 benches" in body  # the clean one stays collapsed
    assert "🔴 slower" in body and "🟢 faster" in body and "≈ noise" in body


def test_render_says_so_when_there_is_nothing_to_compare() -> None:
    body, regressed = report.render([], 0.05)
    assert regressed is False
    assert report.MARKER in body and "_No benchmark results found._" in body


def test_render_states_the_runs_isa_tier() -> None:
    criterion: Entry = {"id": "gemv/b16", "delta": 0.01, "lo": -0.01, "hi": 0.03}
    e2e: Entry = {"id": "cpu/tiny_qwen3", "section": "cpu", "delta": 0.0, "lo": -0.02, "hi": 0.02, "isa": "avx2"}
    body, _ = report.render([criterion, e2e], 0.05)
    assert "at ISA tier `avx2`" in body
    body, _ = report.render([criterion], 0.05)
    assert "ISA tier" not in body


def test_a_new_benchmark_renders_as_new() -> None:
    new: Entry = {"id": "gemv/b16", "delta": None, "lo": 0.0, "hi": 0.0}
    body, regressed = report.render([new], 0.05)
    assert regressed is False
    assert "_new_" in body and "0 benchmarks compared" in body
