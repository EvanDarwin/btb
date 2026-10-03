# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The `-vv` trace (btb/trace.py): off, a call is one flag check and nothing is queued; on, a call queues and
returns - the writer thread formats and writes, so a stream that blocks never holds the caller up; `changed` says a
fact once per change; a malformed event is said as such and never takes the writer down. Device-free, model-free."""

from __future__ import annotations

import io
import sys
import threading
import time
from collections.abc import Iterator

import pytest

from btb import trace


@pytest.fixture(autouse=True)
def _as_found() -> Iterator[None]:
    """the trace as the test found it, put back after: a suite run with BTB_TRACE=1 keeps tracing past this module
    (switched off here, every later test ran with the trace off, its output left on a dead buffer)"""
    on, out = trace.ON, trace._state["out"]
    try:
        yield
    finally:
        trace.disable()
        trace._q.clear()
        trace._last.clear()
        if on:
            trace.enable(out)


@pytest.fixture
def traced() -> Iterator[io.StringIO]:
    """the trace on into a buffer, emptied first"""
    out = io.StringIO()
    trace._q.clear()
    trace._last.clear()
    trace.enable(out)
    yield out


def test_off_queues_nothing() -> None:
    trace.disable()
    trace._q.clear()
    trace._last.clear()  # what a suite before this one said with the trace on (BTB_TRACE=1)
    trace.event("layer %d", 3)
    trace.changed("k", 1, "k is %d", 1)
    assert not trace._q
    assert not trace._last


def test_on_writes_the_formatted_line(traced: io.StringIO) -> None:
    trace.event("vram: layer %d off the card: -%.2f GB VRAM", 35, 0.19)
    trace.event("no arguments, %s kept as written")
    trace.flush()
    lines = traced.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].lstrip().startswith("btb trace")
    assert lines[0].endswith("vram: layer 35 off the card: -0.19 GB VRAM")
    assert lines[1].endswith("no arguments, %s kept as written")


def test_changed_says_a_fact_once_per_change(traced: io.StringIO) -> None:
    for v in (1, 1, 1, 2, 2, 1):
        trace.changed("lane", v, "lane %d", v)
    trace.flush()
    said = [ln.rsplit("  ", 1)[-1] for ln in traced.getvalue().splitlines()]
    assert said == ["lane 1", "lane 2", "lane 1"]


class _BadRepr:
    """an argument whose every rendering raises (a torn-down engine object, a tensor on a lost device)"""

    def __repr__(self) -> str:
        raise RuntimeError("no repr")

    __str__ = __repr__


def test_a_malformed_event_is_said_and_the_writer_lives(traced: io.StringIO) -> None:
    """a format that does not fit its arguments, `%d` of an infinity (OverflowError, once the writer's end) and an
    argument no rendering survives: each said as malformed, and the writer thread still writing after them"""
    trace.event("%d rows", "not a number")
    trace.event("%d rows", float("inf"))
    trace.event("the object %s", _BadRepr())
    time.sleep(0.2)  # the writer thread's own drain, not a flush from here
    trace.event("after them")
    trace.flush()
    thread = trace._state["thread"]
    assert thread is not None and thread.is_alive(), "a malformed event took the writer down"
    text = traced.getvalue()
    assert text.count("'%d rows' with 1 argument(s)") == 2, text
    assert "RuntimeError: no repr" in text, text
    assert text.rstrip().endswith("after them")


def test_with_nowhere_to_write_the_trace_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """a windowed Python has no stderr: enabled there, the trace queued every event for good and wrote none"""
    trace.disable()
    monkeypatch.setattr(sys, "stderr", None)
    trace.enable()
    trace.event("layer %d", 3)
    assert not trace.ON and not trace._q


class _Slow(io.StringIO):
    """a stream whose writes take `delay` (a console the user scrolled back, a pipe nobody reads)"""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.writing = threading.Event()

    def write(self, s: str) -> int:
        self.writing.set()
        time.sleep(self.delay)
        return super().write(s)


def test_a_blocked_stream_never_holds_the_caller() -> None:
    out = _Slow(0.5)
    trace._q.clear()
    trace.enable(out)
    trace.event("first")
    assert out.writing.wait(2.0)  # the writer thread is inside the slow write now
    t0 = time.perf_counter()
    for i in range(10_000):
        trace.event("pass %d", i)
    spent = time.perf_counter() - t0
    # 10k events while the writer is blocked: the caller only appends (a few ms), never waits the 0.5 s write
    assert spent < 0.25, spent
    trace.flush()
    assert "pass 9999" in out.getvalue()
