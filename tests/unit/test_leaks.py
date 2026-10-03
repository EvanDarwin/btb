# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a closed engine still holds (btb/engine/leaks.py), on stand-in engines: the census names what a closed
engine, a btb module's globals or a btb thread still reach, and what only a cycle holds; the caller's own tensors
are not the engine's; an address handed back out by the allocator does not hide a new storage; and the engine's
own check at close logs what its registry and ledger still hold, never raises. CPU tensors a few MB large, the
size floor lowered to match."""

from __future__ import annotations

import gc
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from pytest import MonkeyPatch

from btb.engine import leaks
from btb.engine.holdings import Holdings, Stage

MB = 1 << 20


class Engine:
    """a stand-in: what an engine is to the census - an object built, then closed - with the registry and the log
    the engine's own check reads"""

    def __init__(self) -> None:
        self.kept: Any = None
        self.store: Any = None
        self.holdings = Holdings()
        self.said: list[str] = []

    def log(self, msg: str) -> None:
        self.said.append(msg)


@pytest.fixture(autouse=True)
def _small(monkeypatch: MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(leaks, "MIN_BYTES", MB)
    monkeypatch.setenv("BTB_LEAK_HARNESS", "1")
    leaks._LIVE.clear()
    leaks._BASE.clear()
    leaks._FOUND.clear()
    yield
    leaks._LIVE.clear()
    leaks._BASE.clear()
    leaks._FOUND.clear()


def _buf() -> torch.Tensor:
    return torch.ones(2 * MB, dtype=torch.uint8)


def _life(e: Engine) -> list[str]:
    """`e` closed, and the harness's check"""
    leaks.closed(e)
    return leaks.verify()


def test_a_clean_close_passes() -> None:
    e = Engine()
    leaks.track(e)
    e.kept = _buf()
    e.kept = None
    assert _life(e) == []


def test_what_the_closed_engine_still_reaches_is_kept_by_its_path() -> None:
    e = Engine()
    leaks.track(e)
    e.store = {"blocks": [(_buf(), [0, 1])]}
    found = "\n".join(_life(e))
    assert "still reached as engine 0.store['blocks'][0][0]" in found, found


def test_what_only_a_cycle_holds_is_kept_and_named() -> None:
    """a buffer a reference cycle keeps is freed at the collector's next full pass, which a long-lived process may
    never run: kept memory, and the report names what the cycle is made of"""

    class Store:
        def __init__(self) -> None:
            self.blocks = [_buf()]
            self.size = lambda: len(self.blocks)  # the closure holds the store: a cycle

    e = Engine()
    leaks.track(e)
    was = gc.isenabled()
    gc.disable()
    try:
        e.store = Store()
        e.store = None  # dropped by the engine, alive in its cycle
        found = "\n".join(_life(e))
    finally:
        if was:
            gc.enable()
    assert "held only by a reference cycle with" in found and "Store.__init__.<locals>.<lambda>" in found, found


def test_what_a_btb_modules_globals_keep_is_kept(monkeypatch: MonkeyPatch) -> None:
    from btb.engine.families.gpt_oss import sinks

    e = Engine()
    leaks.track(e)
    monkeypatch.setattr(sinks, "_SCORES", {"cuda:0": _buf()})
    found = "\n".join(_life(e))
    assert "btb.engine.families.gpt_oss.sinks._SCORES['cuda:0']" in found, found


def test_what_a_btb_thread_still_holds_is_kept() -> None:
    """a reader that outlived its join holds its buffer in its frame: btb's, not the caller's"""
    e = Engine()
    leaks.track(e)
    go, done = threading.Event(), threading.Event()

    def reader(buf: torch.Tensor) -> None:
        go.set()
        done.wait(10)
        del buf

    # the thread's function made to read as btb's: its code's file is the census's test of a btb frame
    reader.__code__ = reader.__code__.replace(co_filename=leaks.__file__)
    t = threading.Thread(target=reader, args=(_buf(),), daemon=True)
    t.start()
    go.wait(10)
    try:
        found = "\n".join(_life(e))
    finally:
        done.set()
        t.join(10)
    assert "thread" in found and "in reader" in found, found


def test_the_callers_own_tensors_are_not_the_engines() -> None:
    e = Engine()
    leaks.track(e)
    logits = _buf()  # the caller kept what the engine returned
    assert _life(e) == []
    assert logits.numel() == 2 * MB


def test_the_census_waits_for_the_last_live_engine() -> None:
    a, b = Engine(), Engine()
    leaks.track(a)
    leaks.track(b)
    a.kept = _buf()
    assert _life(a) == []  # b is alive: what is left cannot be told from what b holds
    b.kept, a.kept = a.kept, None
    assert any("still reached as engine" in x and ".kept" in x for x in _life(b))


def test_an_address_handed_back_out_does_not_hide_a_new_storage() -> None:
    """a storage the caller had at the baseline, freed, and its address given to the engine's: new all the same"""
    old = _buf()
    e = Engine()
    leaks.track(e)
    addr = old.untyped_storage().data_ptr()
    del old
    e.kept = next((t for t in (_buf() for _ in range(64)) if t.untyped_storage().data_ptr() == addr), None)
    if e.kept is None:
        pytest.skip("the allocator did not hand the address back out")
    assert any("engine 0.kept" in x for x in _life(e))


def test_the_engines_own_check_logs_what_it_still_holds_and_never_raises() -> None:
    e = Engine()
    leaks.track(e)
    e.holdings.own(Stage.MEMORY, "a buffer nobody released", lambda: None)
    leaks.closed(e)
    assert any("still held: a buffer nobody released" in s for s in e.said), e.said


def test_holdings_let_go_stage_by_stage_last_taken_first_and_keep_what_raised() -> None:
    h, order = Holdings(), []

    def fail() -> None:
        order.append("store")
        raise OSError("disk gone")

    h.own(Stage.FILES, "maps", lambda: order.append("maps"))
    h.own(Stage.MEMORY, "weights", lambda: order.append("weights"))
    h.own(Stage.MEMORY, "store", fail)
    h.own(Stage.STOP, "readers", lambda: order.append("readers"))
    errors = h.release_all()
    assert order == ["readers", "store", "weights", "maps"], "threads first, then the last taken first, files last"
    assert len(errors) == 1 and h.names() == ["store"], "what raised stays held for another try"
    assert any("releasing store" in n for n in getattr(errors[0], "__notes__", []))


def test_naming_a_big_cycle_on_a_big_heap_takes_moments() -> None:
    """what a cycle is made of, named on a heap of a late suite's size: the census once walked back from the leaked
    tensor with `gc.get_referrers`, a scan of the whole heap for each of up to 10,000 objects of the cycle - 17 ms a
    step on a heap of 1.2M objects, minutes a leaked storage, inside an engine's close while the serve registry held
    its lock (a full-suite run stalled five hours on it). The walk goes through an index of the garbage built once"""
    import time

    heap = [[i] for i in range(300_000)]  # a late suite's heap, in part
    e = Engine()
    leaks.track(e)
    was = gc.isenabled()
    gc.disable()
    try:
        ring: list[dict[str, Any]] = [{} for _ in range(20_000)]
        for a, b in zip(ring, [*ring[1:], ring[0]], strict=True):
            a["next"] = b
        ring[0]["t"] = _buf()
        del ring, a, b  # a cycle of 20,000 dicts holding a tensor, let go
        t0 = time.perf_counter()
        found = "\n".join(_life(e))
        took = time.perf_counter() - t0
    finally:
        if was:
            gc.enable()
    assert "held only by a reference cycle" in found, found
    assert took < 10, f"the census took {took:.1f} s"
    assert len(heap) == 300_000


def test_an_engines_close_scans_no_heap_unless_asked(monkeypatch: MonkeyPatch) -> None:
    """the check an engine's `close()` runs itself reads its registry and its ledger and nothing more: no scan of the
    heap unless the census is asked for (the test harness, or `BTB_LEAK_CHECK=1`). The census once ran inside every
    close - the serve registry closes its engines holding its lock, and a full-suite run stalled five hours in there"""
    monkeypatch.delenv("BTB_LEAK_HARNESS", raising=False)
    monkeypatch.delenv("BTB_LEAK_CHECK", raising=False)
    scans: list[int] = []
    objects, referrers = gc.get_objects, gc.get_referrers

    def counted_objects(*a: Any) -> list[Any]:
        scans.append(1)
        return objects(*a)

    def counted_referrers(*a: Any) -> list[Any]:
        scans.append(1)
        return referrers(*a)

    monkeypatch.setattr(gc, "get_objects", counted_objects)
    monkeypatch.setattr(gc, "get_referrers", counted_referrers)
    e = Engine()
    leaks.track(e)
    e.kept = _buf()  # kept, as a leak would be: still no scan
    leaks.closed(e)
    assert scans == [], "a close scanned the heap"
    monkeypatch.setenv("BTB_LEAK_CHECK", "1")
    f = Engine()
    leaks.track(f)
    f.kept = _buf()
    leaks.closed(f)
    assert scans and any("still reached as engine" in s for s in f.said), "asked for, the census runs and logs"
