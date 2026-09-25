"""The Route, its probe and the expert store on a drive that seeks. The machine has none, so the models of
docs/route-hdd-plan.md stand in (tests/drive_sim.py) under the real code: the probe measures the simulated
drive through the reader's own names and the rule answers it; the queue's order and merging are read off
the drive's served commands and its simulated clock; the store's verdict, warning and withheld predictions
follow from the profile the probe took."""

from __future__ import annotations

import random
from typing import Any

import pytest
import torch
from pytest import MonkeyPatch

from btb.engine.scheduler import BatchScheduler
from tests.drive_sim import SeekingDrive
from tests.helpers import GB, expert_store, slot_size, stub_engine

SHARD = 3 * GB
EXPERT = 6553600


def _probe(monkeypatch: MonkeyPatch, model: str, reorder: bool) -> tuple[dict[str, Any], SeekingDrive]:
    drive = SeekingDrive(model, reorder=reorder, files={"F:/shard-00.st": SHARD})
    s = BatchScheduler(stub_engine())
    drive.install(monkeypatch, s)
    try:
        return s.disk("F:/shard-00.st"), drive
    finally:
        drive.stop()


def test_the_probe_reads_each_drive_and_the_rule_answers_it(monkeypatch: MonkeyPatch) -> None:
    p, _ = _probe(monkeypatch, "hdd_7200", True)
    assert 40 < p["single_ms"] < 80, "a seek, half a turn and 6 MB at 150 MB/s"
    assert p["fixed_ms"] > 5, "a 4 KB read is a seek and half a turn"
    # a 48 ms read fits the 50 ms window once: one prediction may hold (the study's table says the same)
    assert p["depth"] == 16 and p["merge"] and p["ahead"] == 1, "deep on a drive that reorders, merged"
    p, _ = _probe(monkeypatch, "hdd_7200", False)
    assert p["depth"] == 1 and p["merge"] and p["ahead"] == 1, "one at a time on a queue that interleaves"
    p, _ = _probe(monkeypatch, "hdd_5400", True)
    assert 60 < p["single_ms"] < 100
    assert p["depth"] == 16 and p["merge"] and p["ahead"] == 0, "a 74 ms read outlasts the window: no predictions"
    p, _ = _probe(monkeypatch, "sata_ssd", True)
    assert p["depth"] == 16 and not p["merge"] and p["ahead"] == 3
    p, _ = _probe(monkeypatch, "nvme", True)
    assert p["depth"] == 16 and not p["merge"] and p["ahead"] == 8


def _route_on(
    monkeypatch: MonkeyPatch, model: str, reorder: bool, depth: int, merge: bool = False
) -> tuple[SeekingDrive, BatchScheduler, dict[str, Any]]:
    drive = SeekingDrive(model, reorder=reorder, files={"F:/a.st": SHARD}, fill=(merge and depth == 1))
    s = BatchScheduler(stub_engine())
    drive.install(monkeypatch, s)
    st = s._disk_state()
    st["depth"], st["merge"] = depth, merge
    st["workers"] = [None]
    return drive, s, st


def _issue_and_drain(s: BatchScheduler, st: dict[str, Any], reads: list[tuple[str, int, int, torch.Tensor]]) -> None:
    futs = [s.disk_read(path, off, n, dst, s.DISK_DEMAND) for path, off, n, dst in reads]
    st["workers"] = []
    with st["cv"]:
        s._disk_start(st)
        st["cv"].notify_all()
    for f in futs:
        f.result(timeout=30)
    s.disk_close()


def test_the_queue_issues_a_burst_in_offset_order_and_the_disk_seeks_less_for_it(monkeypatch: MonkeyPatch) -> None:
    rng = random.Random(7)
    offs = [rng.randrange(0, (SHARD - EXPERT) // 4096) * 4096 for _ in range(20)]
    dst = torch.empty(EXPERT, dtype=torch.uint8)
    drive, s, st = _route_on(monkeypatch, "hdd_7200", False, 1)
    _issue_and_drain(s, st, [("F:/a.st", o, EXPERT, dst) for o in offs])
    served = [o for _, o, _ in drive.order]
    assert served == sorted(offs), "the heap hands the drive the lowest offset first"
    assert drive.seeks == 20, "one seek an expert, its commands contiguous"
    drive.stop()
    # the same reads in call order, straight at the drive
    other = SeekingDrive("hdd_7200", reorder=False, files={"F:/a.st": SHARD})
    for o in offs:
        other.read("F:/a.st", o, EXPERT)
    other.stop()
    assert drive.seek_s < other.seek_s, "twenty sorted seeks walk the platter once; random ones cross it"
    assert drive.xfer_s / drive.now > 0.8, "a disk token is transfer-bound: 6 MB is 44 ms, a seek 10"


def test_a_deep_queue_costs_nothing_where_the_drive_reorders_and_thrashes_where_it_does_not(
    monkeypatch: MonkeyPatch,
) -> None:
    rng = random.Random(11)
    offs = [rng.randrange(0, (SHARD - EXPERT) // 4096) * 4096 for _ in range(16)]
    dst = torch.empty(EXPERT, dtype=torch.uint8)
    times = {}
    for reorder in (True, False):
        for depth in (1, 16):
            drive, s, st = _route_on(monkeypatch, "hdd_7200", reorder, depth)
            _issue_and_drain(s, st, [("F:/a.st", o, EXPERT, dst) for o in offs])
            drive.stop()
            times[(reorder, depth)] = drive.now
    assert times[(True, 16)] < times[(True, 1)] * 1.05, "nearest-first: a deep queue is served no slower"
    assert times[(False, 16)] > times[(False, 1)] * 1.3, "a fair interleave: every command switch is a seek"


def test_adjacent_experts_go_as_one_read_and_one_seek_on_a_disk(monkeypatch: MonkeyPatch) -> None:
    drive, s, st = _route_on(monkeypatch, "hdd_7200", True, 1, merge=True)
    n = 4096
    base = GB
    dsts = [torch.zeros(n, dtype=torch.uint8) for _ in range(5)]
    _issue_and_drain(s, st, [("F:/a.st", base + i * n, n, d) for i, d in enumerate(dsts)])
    assert drive.order == [("F:/a.st", base, 5 * n)], "the run of five as one read"
    assert drive.seeks == 1
    for i, d in enumerate(dsts):
        want = torch.arange(base + i * n, base + (i + 1) * n, dtype=torch.int64).to(torch.uint8)
        assert torch.equal(d, want), "each destination its own bytes out of the run"
    drive.stop()


def test_the_store_on_a_disk_withholds_predictions_warns_of_a_long_prompt_and_calls_it_saturated(
    monkeypatch: MonkeyPatch,
) -> None:
    files = {"F:/gu0.st": GB, "F:/dn0.st": GB, "F:/gu1.st": GB, "F:/dn1.st": GB}
    drive = SeekingDrive("hdd_5400", reorder=True, files=files)
    s = BatchScheduler(stub_engine())
    drive.install(monkeypatch, s)
    st, sm = expert_store(monkeypatch, s, n_layers=48, n_experts=512, lookahead_rows=(10, 6), files="F:/")
    lines: list[str] = []
    sm.log = lines.append
    st.drive = s.disk("F:/gu0.st")
    assert st.drive["ahead"] == 0 and st.drive["merge"]
    report = st.drive_report(st.drive, st.n_slots, 48 * 512, slot_size(st))
    assert "ms" in report and "4096 of 24576" in report
    assert st._grow(256) > 0
    assert st.lookahead(0, torch.ones(1, 4)) == 0, "no prediction on a drive whose read outlasts the window"
    # a prompt of 51 rows asking 130 experts a layer: minutes on this drive, said before the wait
    _ready, pending = st.get(0, "layers.0.mlp.experts.", list(range(130)), keep=False, rows=51)
    assert sum("minutes to its first token" in x for x in lines) == 1
    st.wait(pending)
    assert drive.commands >= 260, "every part of every expert came off the disk"
    # seventeen one-row passes: the sixteenth closes and the medians call the drive saturated
    for i in range(17):
        _ready, pend = st.get(0, "layers.0.mlp.experts.", [200 + i, 300 + i], rows=1)
        st.wait(pend)
        _ready, pend = st.get(1, "layers.1.mlp.experts.", [200 + i], rows=1)
        st.wait(pend)
    assert st._decided and st.saturated and any("saturated, no predictions" in x for x in lines)
    s.disk_close()
    drive.stop()


def test_a_token_of_ninety_misses_on_a_disk_is_seconds_of_transfer(monkeypatch: MonkeyPatch) -> None:
    rng = random.Random(3)
    offs = [rng.randrange(0, (SHARD - EXPERT) // 4096) * 4096 for _ in range(90)]
    dst = torch.empty(EXPERT, dtype=torch.uint8)
    drive, s, st = _route_on(monkeypatch, "hdd_7200", True, 16)
    _issue_and_drain(s, st, [("F:/a.st", o, EXPERT, dst) for o in offs])
    drive.stop()
    assert 4.0 < drive.now < 8.0, "ninety experts at 150 MB/s: about four seconds of transfer plus the seeks"
    assert drive.xfer_s / drive.now > 0.8


@pytest.mark.timing
def test_the_route_notices_a_drive_that_slows_and_says_so_until_it_recovers(monkeypatch: MonkeyPatch) -> None:
    """the live rate: the last reads' bytes over the drive's busy time against the probe's rate;
    a drive at a fifth of it is slow (predictions withheld) and one back above four fifths has recovered"""
    rng = random.Random(5)
    lines: list[str] = []
    # the drive's cost realized as real time, since the Route's live rate reads the wall's clock
    drive = SeekingDrive("hdd_7200", reorder=True, files={"F:/a.st": SHARD}, time_scale=0.1)
    s = BatchScheduler(stub_engine(log=lines.append))
    drive.install(monkeypatch, s)
    st = s._disk_state()
    st["depth"], st["merge"] = 16, False
    dst = torch.empty(EXPERT, dtype=torch.uint8)

    def burst(n: int) -> None:
        offs = [rng.randrange(0, (SHARD - EXPERT) // 4096) * 4096 for _ in range(n)]
        futs = [s.disk_read("F:/a.st", o, EXPERT, dst, s.DISK_DEMAND) for o in offs]
        for f in futs:
            f.result(timeout=60)

    burst(64)
    assert not s.disk_slow()
    full, _ = s.disk_live()
    assert full > 0, "the live rate is kept whether or not the probe ran"
    st["expect_gbs"] = full
    drive.rate /= 5
    burst(64)
    assert s.disk_slow(), "a fifth of the probe's rate is slow"
    assert any("the drive slowed" in x and "predictions withheld" in x for x in lines)
    drive.rate *= 5
    burst(96)
    assert not s.disk_slow(), "and back above four fifths it has recovered"
    assert any("the drive recovered" in x for x in lines)
    assert sum("the drive slowed" in x for x in lines) == 1, "said at the turn, not every pulse"
    s.disk_close()
    drive.stop()
