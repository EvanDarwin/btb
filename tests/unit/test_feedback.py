# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What btb tells the user when a run is worth reporting: out-of-memory recognition across allocator flavours and
the crash report a failed run prints for an issue, and the `--profile` folder an end-to-end run writes
(report.json, events.npz). Model-free but for one tiny-fixture run on the CPU."""

import json
import os
from pathlib import Path

import pytest

from btb.kinds import Json
from tests.helpers import fixture


def test_oom_is_recognized_and_reported() -> None:
    """the last resort: an out-of-memory failure of any flavour is named and turned into a paste-ready report"""
    from btb.feedback import ISSUES, crash_notice, is_oom, issue_report

    class OutOfMemoryError(RuntimeError):  # torch.cuda's, by name
        pass

    assert is_oom(MemoryError())
    assert is_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB"))
    assert is_oom(
        RuntimeError("[metal::malloc] Attempting to allocate 8589934592 bytes which is greater than the maximum")
    )
    assert is_oom(RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 1000 bytes"))
    chained = ValueError("the load failed")
    chained.__cause__ = MemoryError()
    assert is_oom(chained)
    assert not is_oom(ValueError("bad shape"))
    assert not is_oom(KeyError("x"))
    e = RuntimeError("MPS backend out of memory (MPS allocated: 9.0 GB)")
    assert ISSUES in crash_notice(e) and "--ram-reserve" in crash_notice(e)
    assert "--ram-reserve" not in crash_notice(ValueError("bad shape"))
    r = issue_report(e)  # before an engine is up: no reproduction, placement or memory sections
    assert r.startswith("## btb crash report") and "MPS backend out of memory" in r and "```sh" in r
    assert "| ID | Engine(s) | Details |" in r and "<summary>Traceback</summary>" in r and "--profile DIR" in r
    assert "### Reproduction" not in r and "### Placement" not in r


class _Engine:
    """an engine as the report reads it: a ledger and the knobs the reproduction command asks for"""

    cfg = None
    context = 4096
    ram_reserve = 2**30
    vram_watch = False

    def report(self) -> Json:
        return {
            "device": "cpu",
            "placement": {
                "resident": [],
                "host": [0, 1],
                "cold": [],
                "head": "host",
                "drafter": "none",
                "kv": "host",
                "compute_dtype": "fp32",
                "packed": False,
            },
            "plan": {"free": {"vram_gb": 0.0, "ram_gb": 20.0}, "caps": {}},
            "peak": {"ram_gb": 3.0, "vram_reserved_gb": 0.0, "mlx_gb": 0.0},
            "speculation": {"proposer": "ngram", "tree_budget": 0, "v_max": 4, "draft_vocab": 0},
            "counters": {"load_s": 1.0, "compute_s": 2.0, "cold_wait_s": 0.0},
        }


def test_the_crash_report_carries_the_engines_ledger_when_one_is_up() -> None:
    """with an engine: the reproduction command at the run's resolved values, the placement, the memory and the
    inference lines, the expert-store summary where one is given, and the profile folder to attach"""
    import argparse

    from btb.feedback import issue_report

    a = argparse.Namespace(cmd="run", path="m", device="cpu", prompt="hi", new=8, file=None)
    r = issue_report(RuntimeError("boom"), a, _Engine(), ["run", "m"], "/tmp/diag", "hits 10 misses 2")
    for part in (
        "### Reproduction",
        "btb run m",
        "--device cpu",
        "--kv-host 1",
        "### Placement",
        "2 host",
        "### Memory",
        "RAM 20.0 GB",
        "### Inference",
        "proposer ngram",
        "### Expert store",
        "hits 10 misses 2",
        "Attach the profile folder `/tmp/diag`",
    ):
        assert part in r, part


def test_profile_writes_the_ledger_and_the_trace_and_nothing_else(tmp_path: Path) -> None:
    """`--profile DIR` leaves two files: report.json, the engine's ledger, and events.npz, the expert trace (a
    dense model leaves the watchdog's events in it); no page beside them"""
    from btb.cli import main

    fx = fixture("tiny_qwen3")
    if not os.path.exists(os.path.join(fx, "tokenizer.json")):
        pytest.skip("the tiny_qwen3 fixture has no tokenizer: these tests run from a checkout")
    d = tmp_path / "diag"
    rc = main(["run", fx, "-p", "hello", "-d", "cpu", "--new", "2", "--profile", str(d)])
    assert rc == 0
    assert sorted(p.name for p in d.iterdir()) == ["events.npz", "report.json"]
    report = json.loads((d / "report.json").read_text(encoding="utf-8"))
    assert report["device"] == "cpu" and "placement" in report and "counters" in report
