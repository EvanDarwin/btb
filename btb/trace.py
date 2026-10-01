# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The trace: btb's automatic decisions as they are taken, for `-vv` (or `BTB_TRACE=1`).

btb decides a lot on its own - where a layer sits and when it moves, what it gives back to another program, which
path a pass takes, which kernel or lane the tuner picks, when speculation is priced out - and none of it showed. The
trace says each as it happens, from wherever it happens, the inference passes included, without holding them up:

- off (the default), a trace call is one module-flag check and returns;
- on, the caller appends `(time, format, args)` to a deque (thread-safe, no lock, no formatting, no I/O) and
  returns; a daemon thread formats and writes the lines to stderr in batches, flushed at exit and by `flush()`.

`changed(key, value, ...)` says a fact only when it moves - the path the passes take, the tuner's lane, whether a
pass verifies exactly - so a pass's state shows once per change, never once per token."""

from __future__ import annotations

import atexit
import contextlib
import itertools
import os
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from typing import Any, TextIO

# read on every call: the one check a disabled trace costs
ON = False

_T0 = time.perf_counter()
_q: deque[tuple[float, str, tuple[Any, ...]]] = deque()
_last: dict[Any, Any] = {}
_state: dict[str, Any] = {"thread": None, "out": None}
_lock = threading.Lock()  # enable and the drains (the writer's, `flush`'s); never taken by `event`


def enable(out: TextIO | None = None) -> None:
    """start the trace (idempotent): events from here on go to `out` (stderr by default) through the writer thread.
    With nowhere to write (no `out`, and no stderr - a windowed Python), it stays off: the events would only queue"""
    global ON
    with _lock:
        _state["out"] = out if out is not None else sys.stderr
        if _state["out"] is None:
            ON = False
            return
        if _state["thread"] is None:
            t = threading.Thread(target=_run, name="btb-trace", daemon=True)
            _state["thread"] = t
            t.start()
            atexit.register(flush)
        ON = True


def disable() -> None:
    """stop taking events; what is queued is still written"""
    global ON
    ON = False
    flush()


def event(fmt: str, *args: Any) -> None:
    """one decision or event: `fmt % args`, formatted and written by the writer thread (`fmt` alone where no args)"""
    if ON:
        _q.append((time.perf_counter(), fmt, args))


def changed(key: Any, value: Any, fmt: str, *args: Any) -> None:
    """`event(fmt, *args)` only when `key`'s value is not the one last said (the first time too): a pass's state, said
    once per change. The comparison runs in the caller's thread - a dict read"""
    if ON and _last.get(key, _MISSING) != value:
        _last[key] = value
        _q.append((time.perf_counter(), fmt, args))


_MISSING = object()


def say(msg: object, *_args: Any) -> None:
    """a log line as a trace event: the engine's log under -vv (`btb.load(log=trace.say)`), so each line is said once,
    in the trace's order and time, by the writer thread"""
    event("%s", msg)


def logged(log: Callable[..., object] | None) -> Callable[..., object]:
    """`log` (None: nothing), every line of it a trace event too while the trace runs - what the engine logs is in
    the trace whatever log a library passes, or none; `say` itself as it is"""
    if log is say:
        return say

    def both(msg: object, *args: Any) -> object:
        if ON:
            event("%s", msg)
        return log(msg, *args) if log is not None else None

    return both


# a number per object for `changed` keys: never another's after it is gone (an `id` is reused by the next object at
# that address - a new store inherited a dead one's last word, and its first went unsaid)
_TOKENS: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_NEXT = itertools.count(1)


def token(obj: Any) -> int:
    """a number naming `obj` for as long as it lives, and no other object after: a `changed` key that is one engine's
    or one store's, so two engines in a process neither hide nor flip each other's facts"""
    t = _TOKENS.get(obj)
    if t is None:
        t = _TOKENS[obj] = next(_NEXT)
    return t


def _line(t: float, fmt: str, args: tuple[Any, ...]) -> str:
    try:
        msg = fmt % args if args else fmt
    except Exception as e:  # a malformed event says so, never takes the writer down: `%d` of inf, a raising repr
        msg = f"{fmt!r} with {len(args)} argument(s) ({type(e).__name__}: {e})"
    return f"  btb trace {t - _T0:9.3f}s  {msg}\n"


def _drain() -> None:
    out = _state["out"]
    if not _q:
        return
    if out is None:  # nowhere to write: what was queued goes, not kept for good
        _q.clear()
        return
    lines = []
    while _q:
        try:
            lines.append(_line(*_q.popleft()))
        except IndexError:
            break
    try:
        out.write("".join(lines))
        out.flush()
    except (OSError, ValueError):  # a closed stream at interpreter exit
        pass


def _run() -> None:
    while True:
        time.sleep(0.05)
        # one drain at a time (a `flush` beside it): the lines stay in the order they were taken. The writer outlives
        # any one batch: dead, every later event was only queued
        with _lock, contextlib.suppress(Exception):
            _drain()


def flush() -> None:
    """write everything queued now (at exit, at an engine's close, before a report)"""
    with _lock:
        _drain()


if os.environ.get("BTB_TRACE", "") not in ("", "0"):
    enable()
