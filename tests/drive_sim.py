"""A drive that seeks, standing in for the native reader under the Route and its probe: one actuator, a seek
curve, rotation, a media rate, commands of a fixed size served nearest-first (a drive that reorders) or by a
fair interleave of every outstanding read (a queue that does not), and a simulated clock the probe reads
through `BatchScheduler._disk_clock`. Time is simulated, so a disk's seconds cost the tests nothing; a short
real pause before each choice lets the readers' requests queue up so the drive has a set to choose from, as a
real one would. The models are docs/route-hdd-plan.md's, section 1."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import TYPE_CHECKING, TypedDict

import torch

from tests.helpers import MB

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from btb.engine.scheduler import BatchScheduler

MODELS: dict[str, dict[str, float]] = {
    "hdd_7200": {"rate": 150e6, "rot": 0.00417, "track": 0.0008, "full": 0.017, "fixed": 0.0},
    "hdd_5400": {"rate": 100e6, "rot": 0.00556, "track": 0.0015, "full": 0.022, "fixed": 0.0},
    "sata_ssd": {"rate": 500e6, "rot": 0.0, "track": 0.0, "full": 0.0, "fixed": 0.0001},
    "nvme": {"rate": 5.2e9, "rot": 0.0, "track": 0.0, "full": 0.0, "fixed": 0.00013},
}


class _Read(TypedDict):
    path: str
    off: int
    n: int
    cmds: list[tuple[int, int]]
    at: int
    cost: float
    event: threading.Event


class SeekingDrive:
    def __init__(
        self,
        model: str = "hdd_7200",
        reorder: bool = True,
        files: dict[str, int] | None = None,
        span: float = 326e9,
        cmd: int = MB,
        pause: float = 0.0002,
        fill: bool = False,
        time_scale: float = 0.0,
    ) -> None:
        p = MODELS[model]
        self.rate, self.rot, self.track, self.full, self.fixed = p["rate"], p["rot"], p["track"], p["full"], p["fixed"]
        self.reorder = bool(reorder)
        self.span = float(span)
        self.cmd = int(cmd)
        self.pause = float(pause)
        self.fill = bool(fill)
        # a command's simulated cost realized as real time at this scale (0: the pause alone), for what reads
        # the wall's clock rather than the drive's: the Route's live rate, which must see a drive slow down
        self.time_scale = float(time_scale)
        self.files: dict[str, int] = {str(k): int(v) for k, v in (files or {}).items()}
        # the files laid out on the platter in name order, each whole
        self.base: dict[str, int] = {}
        at = 0
        for path in sorted(self.files):
            self.base[path] = at
            at += self.files[path]
        self.now = 0.0
        self.head = 0
        self.commands = 0
        self.seeks = 0
        self.bytes = 0
        self.seek_s = 0.0
        self.rot_s = 0.0
        self.xfer_s = 0.0
        self.order: list[tuple[str, int, int]] = []
        self._cv = threading.Condition()
        self._reads: list[_Read] = []
        self._rr = 0
        self._handles: dict[int, str] = {}
        self._next = 1
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True, name="drive")
        self._thread.start()

    # -- the clock the probe reads
    def clock(self) -> float:
        return self.now

    def address(self, path: str, off: int) -> int:
        return self.base[str(path)] + int(off)

    def _seek(self, d: int) -> float:
        if self.full <= 0:
            return 0.0
        return self.track + (self.full - self.track) * math.sqrt(min(1.0, d / self.span))

    # -- a read: queued as commands, done when its last command is served
    def read(self, path: str, off: int, n: int, dst: torch.Tensor | None = None) -> float:
        path, off, n = str(path), int(off), int(n)
        if path not in self.base:
            raise FileNotFoundError(path)
        addr = self.address(path, off)
        cmds = [(addr + i, min(self.cmd, n - i)) for i in range(0, n, self.cmd)]
        req: _Read = {"path": path, "off": off, "n": n, "cmds": cmds, "at": 0, "cost": 0.0, "event": threading.Event()}
        with self._cv:
            self._reads.append(req)
            self._cv.notify()
        req["event"].wait()
        if self.fill and dst is not None and dst.numel():
            dst.copy_(torch.arange(off, off + n, dtype=torch.int64).to(torch.uint8))
        return float(req["cost"])

    def _serve(self) -> None:
        while True:
            with self._cv:
                while not self._reads and not self._stop:
                    self._cv.wait(timeout=0.05)
                if self._stop:
                    return
            with self._cv:
                if not self._reads:
                    continue
                if self.reorder:
                    req = min(self._reads, key=lambda r: abs(r["cmds"][r["at"]][0] - self.head))
                else:
                    self._rr %= len(self._reads)
                    req = self._reads[self._rr]
                    self._rr += 1
                addr, n = req["cmds"][req["at"]]
                d = abs(addr - self.head)
                seek = 0.0 if d == 0 and self.commands else self._seek(d)
                rot = 0.0 if d == 0 and self.commands else self.rot
                xfer = n / self.rate
                cost = self.fixed + seek + rot + xfer
                self.now += cost
                self.head = addr + n
                self.commands += 1
                self.seeks += 1 if (seek + rot) > 0 else 0
                self.bytes += n
                self.seek_s += seek
                self.rot_s += rot
                self.xfer_s += xfer
                req["cost"] += cost
                req["at"] += 1
                if req["at"] == len(req["cmds"]):
                    self._reads.remove(req)
                    self.order.append((req["path"], req["off"], req["n"]))
                    req["event"].set()
            # real time a command: the pause that lets the next requests queue up, or the command's own cost
            # at the scale asked for
            wait = max(self.pause, cost * self.time_scale)
            if wait > 0:
                time.sleep(wait)

    # -- the native reader's names, bound to this drive
    def install(self, monkeypatch: MonkeyPatch, sched: BatchScheduler | None = None) -> None:
        from btb.engine import native as native_mod

        drive = self

        def open_(path: str | os.PathLike[str]) -> int:
            h = drive._next
            drive._next += 1
            drive._handles[h] = str(path)
            return h

        def read_at(h: int, off: int, n: int, dst: torch.Tensor, chunk: int = 0, depth: int = 0) -> None:
            drive.read(drive._handles[int(h)], off, n, dst)

        def close_(h: int) -> None:
            drive._handles.pop(int(h), None)

        def read_direct(path: str | os.PathLike[str], off: int, n: int, dst: torch.Tensor, chunk: int = 0) -> None:
            drive.read(str(path), off, n, dst)

        monkeypatch.setattr(native_mod.Native, "open", staticmethod(open_), raising=False)
        monkeypatch.setattr(native_mod.Native, "read_at", staticmethod(read_at), raising=False)
        monkeypatch.setattr(native_mod.Native, "close", staticmethod(close_), raising=False)
        monkeypatch.setattr(native_mod.Native, "read_direct", staticmethod(read_direct), raising=False)
        real_getsize = os.path.getsize
        monkeypatch.setattr(os.path, "getsize", lambda p: drive.files.get(str(p)) or real_getsize(p))
        # a simulated file's volume is its path's drive prefix on every OS (the real lookup stats the path)
        from btb.engine.scheduler import BatchScheduler

        real_volume = BatchScheduler._volume

        def volume(path: str) -> str:
            if str(path) in drive.files:
                return os.path.splitdrive(str(path))[0].upper() or os.path.dirname(str(path))
            return real_volume(path)

        monkeypatch.setattr(BatchScheduler, "_volume", staticmethod(volume))
        if sched is not None:
            sched._disk_clock = drive.clock

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=5)
