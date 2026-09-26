# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The lending ledger alone (docs/lending.md), over a stub engine on the CPU: what is lent is read from what lives,
never tallied - so threads lending, filling rooms, releasing and dropping them while others read find every reading
within what is lent and nothing left once all is gone, each loan a return once; a loan held by a reference cycle
counts until the collection that frees it; and lending registers no finalizer, so no collection runs ledger code."""

from __future__ import annotations

import gc
import queue
import random
import threading
import weakref
from collections.abc import Iterator

import pytest
import torch

from btb.engine.device import Device
from btb.engine.memory import Room
from btb.engine.scheduler import MemoryGrantError

CPU = torch.device("cpu")


class _Engine:
    dev = CPU

    def _called(self, tag: object) -> None:
        pass


@pytest.fixture
def led() -> Iterator[Device]:
    yield Device(_Engine())


def _lent(led: Device, nbytes: int, counted: bool = True) -> torch.Tensor:
    t = led.lend(lambda: torch.empty(nbytes, dtype=torch.uint8), nbytes, CPU, counted)
    assert t is not None
    return t


def test_a_loan_counts_while_it_lives_and_comes_back_once(led: Device) -> None:
    eng = _Engine()
    t, seen = _lent(led, 1024), _lent(led, 2048, counted=False)
    r = Room(led, "r#1", 4096, CPU, False, eng)  # type: ignore[arg-type]
    inside = r.empty(1000, dtype=torch.uint8)
    # the loose tensor, the room's tensor (unseen: its own), the room less it; the seen tensor the reading's
    assert led.reserved(CPU) == 1024 + 1000 + (4096 - 1000) and led.returns == 0
    r.release()
    r.release()
    assert led.reserved(CPU) == 1024 + 1000 and led.returns == 1, "a room let go twice comes back once"
    del t, seen, inside, r
    assert led.reserved(CPU) == 0 and led.returns == 4


def test_a_room_dropped_unreleased_comes_back(led: Device) -> None:
    r = Room(led, "r#1", 4096, CPU, True, _Engine())  # type: ignore[arg-type]
    assert led.reserved(CPU) == 4096
    del r
    gc.collect()
    assert led.reserved(CPU) == 0 and led.returns == 1


def test_a_loan_held_by_a_cycle_counts_until_the_collection_that_frees_it(led: Device) -> None:
    class Holder:
        pass

    gc.disable()  # the collection is the test's, not an allocation's
    try:
        h = Holder()
        h.me = h  # type: ignore[attr-defined]
        h.t = _lent(led, 1024)  # type: ignore[attr-defined]
        r = Room(led, "cyclic#1", 4096, CPU, True, _Engine())  # type: ignore[arg-type]
        r.me = r  # type: ignore[attr-defined]
        del h, r
        assert led.reserved(CPU) == 1024 + 4096 and led.returns == 0, "gone before anything freed it"
        gc.collect()
        assert led.reserved(CPU) == 0 and led.returns == 2
    finally:
        gc.enable()


def test_lending_registers_no_finalizer(led: Device) -> None:
    """a collection runs no ledger code: lending leaves nothing for one to call"""
    before = len(weakref.finalize._registry)  # type: ignore[attr-defined]
    r = Room(led, "r#1", 4096, CPU, False, _Engine())  # type: ignore[arg-type]
    kept = [r.zeros(64, dtype=torch.uint8), _lent(led, 64), _lent(led, 64, counted=False)]
    assert len(weakref.finalize._registry) == before  # type: ignore[attr-defined]
    del kept, r


def test_a_full_room_refuses_and_a_released_one_makes_nothing(led: Device) -> None:
    r = Room(led, "r#1", 1024, CPU, True, _Engine())  # type: ignore[arg-type]
    t = r.empty(1000, dtype=torch.uint8)
    with pytest.raises(MemoryGrantError, match="MiB left"):
        r.empty(100, dtype=torch.uint8)
    del t
    assert r.empty(1024, dtype=torch.uint8).numel() == 1024, "a tensor gone gives its bytes back to the room"
    r.release()
    with pytest.raises(ValueError, match="released"):
        r.empty(1)


THREADS, ROUNDS = 8, 150
ROOM, EACH, LOOSE = 4096, 1000, 512  # a room, each of its three tensors, a loose loan


def test_lending_on_many_threads_while_others_read_adds_up() -> None:
    """lenders on eight threads make rooms, fill them, lend beside them, release some rooms and drop others, and
    hand their tensors to a thread that drops them there, collecting now and then; readers read throughout. Every
    reading is within what the lenders can hold at once; once all are joined and collected nothing is held, and
    each loan came back exactly once"""
    led = Device(_Engine())
    eng = _Engine()
    errors: list[BaseException] = []
    lending = threading.Event()
    lending.set()
    drop: queue.SimpleQueue[tuple[list[torch.Tensor], threading.Event] | None] = queue.SimpleQueue()
    # at most per lender at once: a room (its tensors inside it, or outliving it released) and a loose loan
    bound = THREADS * (ROOM + LOOSE)

    def lender(k: int) -> None:
        rng = random.Random(k)
        try:
            for i in range(ROUNDS):
                r = Room(led, f"r{k}#{i}", ROOM, CPU, bool(i % 3), eng)  # type: ignore[arg-type]
                ts = [r.empty(EACH, dtype=torch.uint8) for _ in range(3)]
                loose = _lent(led, LOOSE, counted=rng.random() < 0.5)
                if rng.random() < 0.5:
                    r.release()
                dropped = threading.Event()
                drop.put((ts + [loose], dropped))
                del ts, loose, r
                dropped.wait()
                if rng.random() < 0.05:
                    gc.collect()
        except BaseException as e:
            errors.append(e)

    def dropper() -> None:
        while (got := drop.get()) is not None:
            got[0].clear()  # the last references, dropped on this thread
            got[1].set()

    def reader() -> None:
        try:
            while lending.is_set():
                n = led.reserved(CPU)
                assert 0 <= n <= bound, n
                assert led.returns >= 0
        except BaseException as e:
            errors.append(e)

    readers = [threading.Thread(target=reader) for _ in range(2)]
    drops = threading.Thread(target=dropper)
    lenders = [threading.Thread(target=lender, args=(k,)) for k in range(THREADS)]
    for th in [*readers, drops, *lenders]:
        th.start()
    for th in lenders:
        th.join()
    drop.put(None)
    drops.join()
    lending.clear()
    for th in readers:
        th.join()
    gc.collect()
    assert not errors, errors[0]
    assert led.reserved(CPU) == 0
    assert led.returns == THREADS * ROUNDS * 5, "a room, its three tensors and a loose loan: each back once"
