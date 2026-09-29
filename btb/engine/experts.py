# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The expert store: a MoE model's experts kept in RAM in their stored form and read from the drive on a miss."""

from __future__ import annotations

import enum
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Container, Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait as wait_for
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .. import mlx as mlxdev
from ..fp8 import F8Weight
from ..kinds import PassTag
from ..mxfp4 import BLOCK, MxGateUp, MxWeight, stored_mxfp4
from ..options import Device
from .device import where
from .host import bf16_in_place, stored_parts
from .native import Native
from .scheduler import MemoryGrantError

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor
    from types import ModuleType


class ExpertProfile:
    """Everything the store did, one row per event in an append-only int64 array: no allocation on the path
    (the array doubles when full), one lock shared with the reader threads, saved by `save` as an .npz with
    `events` [n, 11], `kinds`, `shards` (a path per shard id) and `picks` [m, k] int16, every call's picks row by
    row - which experts each of its rows asked, in the router's order - kept the same way in an array of their own.
    Columns: t_ns from the first event, step (layer 0's call starts a pass), kind, layer, expert, bytes, shard,
    offset, dur_ns, slot, aux. By kind:

        hit      the expert served from its slot
        miss     one read of a missing expert's part: bytes, shard, offset, dur_ns of that read, slot, aux the
                 part (0 gate_up, 1 down). Why it was read is in the history: a (layer, expert)'s first miss
                 is its fill, a later one a reload after an eviction
        reread   a hit whose slot a release took under the call, read again
        evict    the expert a slot was taken from: aux 0 for the call's need, 1 for the machine's memory
        grow     a block added: expert the slots, bytes, aux the host's free bytes after
        release  a block given back for the machine: expert the slots, bytes, aux the host's free bytes after
        call     one layer's call done: expert the rows, bytes the hits, shard the misses, dur_ns the wait, offset
                 the call's first row in `picks` (its rows the next `expert` of them)
        ahead    the lookahead predicted the expert and queued its reads into a ring slot: aux the depth (1 the
                 next layer, 2 the one after); a hit with aux 1 later is that prediction used
        gil      the watchdog's wake (`watch`): dur_ns how late it woke; aux 1 when the threads' frames were kept
        drive    the Route's live rate turned: dur_ns the rate in bytes a second; aux 1 slowed, 0 recovered
    """

    KINDS = ("hit", "miss", "reread", "evict", "grow", "release", "call", "ahead", "gil", "drive")
    HIT, MISS, REREAD, EVICT, GROW, RELEASE, CALL, AHEAD, GIL, DRIVE = range(10)

    def __init__(self, path: str, cap: int = 1 << 17) -> None:
        self.path = str(path)
        self.a = np.zeros((int(cap), 11), dtype=np.int64)
        self.n = 0
        # the calls' picks: sized by the first call (its k), each row an int16 per pick, -1 past a narrower call's k
        self.p = np.full((0, 0), -1, dtype=np.int16)
        self._pt = torch.from_numpy(self.p)  # the same memory, for a copy straight from the router's tensor
        self.m = 0
        self.step = 0
        self.t0 = time.perf_counter_ns()
        self.shards: dict[str, int] = {}
        self._lock = threading.Lock()
        self._watching = False
        self.snapshots: list[str] = []  # the threads' frames at each late wake, saved with the trace

    def watch(self, every_s: float = 0.001, late_ms: float = 3.0) -> None:
        """A thread that sleeps `every_s` and records how late it woke (kind `gil`: dur_ns the lateness): a wake
        held past the interpreter's switch interval means another thread held the GIL through it. On a wake
        later than `late_ms`, at most ten times a second, every thread's current frames are kept (aux 1 on the
        row) and saved with the trace as `snapshots`, which names the holder."""
        if self._watching:
            return
        self._watching = True
        interval = float(every_s)

        def run() -> None:
            last_snap = 0.0
            while self._watching:
                t0 = time.perf_counter_ns()
                time.sleep(interval)
                late = time.perf_counter_ns() - t0 - int(interval * 1e9)
                snap = late > late_ms * 1e6 and (time.perf_counter() - last_snap) > 0.1
                self.add(self.GIL, dur_ns=max(0, late), aux=1 if snap else 0)
                if snap:
                    last_snap = time.perf_counter()
                    self._snapshot(late)

        threading.Thread(target=run, daemon=True, name="gil-watch").start()

    def _snapshot(self, late_ns: int) -> None:
        names = {th.ident: th.name for th in threading.enumerate()}
        me = threading.get_ident()
        lines = [f"t={(time.perf_counter_ns() - self.t0) / 1e9:.3f}s step={self.step} late={late_ns / 1e6:.1f}ms"]
        for tid, frame in sys._current_frames().items():
            if tid == me:
                continue
            chain: list[str] = []
            f: Any = frame
            while f is not None and len(chain) < 4:
                chain.append(f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} {f.f_code.co_name}")
                f = f.f_back
            lines.append(f"  {names.get(tid, tid)}: " + " <- ".join(chain))
        with self._lock:
            self.snapshots.append("\n".join(lines))

    def shard(self, path: str) -> int:
        i = self.shards.get(path)
        if i is None:
            i = self.shards[path] = len(self.shards)
        return i

    def add(
        self,
        kind: int,
        layer: int = -1,
        expert: int = -1,
        nbytes: int = 0,
        shard: int = -1,
        offset: int = -1,
        dur_ns: int = 0,
        slot: int = -1,
        aux: int = 0,
    ) -> None:
        t = time.perf_counter_ns() - self.t0
        with self._lock:
            if self.n == self.a.shape[0]:
                self.a = np.concatenate([self.a, np.zeros_like(self.a)])
            self.a[self.n] = (t, self.step, kind, layer, expert, nbytes, shard, offset, dur_ns, slot, aux)
            self.n += 1

    def keep_picks(self, top: torch.Tensor) -> int:
        """A call's picks [rows, k] (on the host) copied into `picks`, returned as the offset of its first row: the
        `call` event's offset. The array doubles when full, as the events' does, and widens once for a call of more
        picks than the first; the copy is torch's into the array's own memory, the int64 indices narrowed in place,
        with nothing allocated for it."""
        rows, k = int(top.shape[0]), int(top.shape[-1])
        with self._lock:
            cap, width = self.p.shape
            if self.m + rows > cap or k > width:
                grown = np.full((max(2 * cap, self.m + rows, 1 << 14), max(width, k)), -1, dtype=np.int16)
                grown[: self.m, :width] = self.p[: self.m]
                self.p, self._pt = grown, torch.from_numpy(grown)
            off = self.m
            self._pt[off : off + rows, :k].copy_(top.reshape(rows, k))
            self.m += rows
        return off

    def save(self) -> str:
        self._watching = False
        with self._lock:
            ev = self.a[: self.n].copy()
            picks = self.p[: self.m].copy()
            snaps = list(self.snapshots)
        shards = [p for p, _ in sorted(self.shards.items(), key=lambda kv: kv[1])]
        np.savez(
            self.path,
            events=ev,
            kinds=np.array(self.KINDS),
            shards=np.array(shards),
            snapshots=np.array(snaps),
            picks=picks,
        )
        text = self.summary(ev) + f"\n[profile] {ev.shape[0]} events -> {self.path}"
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
        return text

    @classmethod
    def summary(cls, ev: Any) -> str:
        k = ev[:, 2]
        hits = int((k == cls.HIT).sum())
        miss = ev[(k == cls.MISS) & (ev[:, 10] == 0)]
        parts = ev[k == cls.MISS]
        calls = ev[k == cls.CALL]
        keys = miss[:, 3] * 65536 + miss[:, 4]
        reloads = int(miss.shape[0] - np.unique(keys).shape[0]) if miss.shape[0] else 0
        steps = int(ev[:, 1].max()) if ev.shape[0] else 0
        wait = float(calls[:, 8].sum()) / 1e9 if calls.shape[0] else 0.0
        rd = parts[:, 8] / 1e6 if parts.shape[0] else np.zeros(1)
        return (
            f"[profile] {steps} passes, {hits + miss.shape[0]} experts asked: {hits} hits, {miss.shape[0]} misses "
            f"({reloads} reloads of an expert read before), {parts[:, 5].sum() / 2**30:.2f} GiB read, "
            f"{int((k == cls.EVICT).sum())} evictions, {int((k == cls.GROW).sum())} grows, "
            f"{int((k == cls.RELEASE).sum())} releases; waited {wait:.1f} s; a part read "
            f"{rd.mean():.1f} ms mean, {np.percentile(rd, 99):.1f} p99, {rd.max():.1f} max"
        )


class Parts:
    """an expert's reads: `result` waits for every part, `done` says whether they have all landed"""

    __slots__ = ("futs",)

    def __init__(self, futs: list[Any]) -> None:
        self.futs = futs

    def result(self) -> None:
        for f in self.futs:
            f.result()

    def settle(self) -> None:
        """every part finished, landed or withdrawn: a part in flight is waited for even after another was
        cancelled (`result` stops at the first cancelled one, while the rest still write into the slot). A
        withdrawn part never runs (only queued reads withdraw), and `concurrent.futures.wait` never counts a future
        cancelled by hand as done: each part not withdrawn is waited on itself, its failure left to its reader"""
        for f in self.futs:
            if not f.cancelled():
                f.exception()

    def done(self) -> bool:
        return all(f.done() for f in self.futs)


class Riders:
    """Who has a seat in the store and who gives it up: one line (`t1`, key to slot) by last ride, the oldest
    bumped first. `capacity` gives the seats, for the shapes that size themselves by it."""

    __slots__ = ("capacity", "t1")

    def __init__(self, capacity: Any) -> None:
        self.capacity = capacity
        self.t1: OrderedDict[Any, Any] = OrderedDict()

    def __contains__(self, key: Any) -> bool:
        return key in self.t1

    def __len__(self) -> int:
        return len(self.t1)

    def get(self, key: Any) -> Any:
        """the rider's slot if seated, its ride noted; None otherwise"""
        s = self.t1.get(key)
        if s is not None:
            self.t1.move_to_end(key)
        return s

    def admit(self, key: Any, slot: Any) -> None:
        self.t1[key] = slot

    def peek(self, key: Any) -> Any:
        """the rider's slot if seated, no ride noted; None otherwise"""
        return self.t1.get(key)

    def victim(self, skip: Any = ()) -> tuple[Any, Any] | None:
        """the rider who gives up a seat, removed: the oldest ride; a rider whose slot is in `skip` (the call's
        own) is passed over. None when nobody can be bumped."""
        # the oldest, looked at in place: a copy of the line for every seat a prefill takes was a second of
        # Python a layer with the readers' completions queued behind it
        for _ in range(len(self.t1)):
            key = next(iter(self.t1))
            s = self.t1[key]
            if s in skip:
                # the call's own: in use this very moment, so the most recent of all
                self.t1.move_to_end(key)
                continue
            del self.t1[key]
            self.bumped(key)
            return key, s
        return None

    def bumped(self, key: Any) -> None:
        """a rider lost its seat: nothing to remember here"""

    def pop(self, key: Any) -> Any:
        """unseat a rider without remembering it (the slot went back to the machine, or into the ring)"""
        return self.t1.pop(key, None)

    def demote(self, key: Any) -> None:
        """to the front of the line: a prefill's expert gives up its seat first"""
        if key in self.t1:
            self.t1.move_to_end(key, last=False)

    def oldest_slot(self) -> Any:
        return next(iter(self.t1.values()), None)

    def items(self) -> Any:
        yield from self.t1.items()


class BusPass(Riders):
    """The Bus Pass, ARC's shape: day riders (seen once, `t1`) and regulars (ridden twice or more, `t2`) in two
    lines by last ride, and a ghost line of the recently bumped from each (`b1`, `b2`). A bumped day rider who
    comes back grows the day riders' share of the seats (`p`), a bumped regular the regulars', so the split
    between recency and frequency follows the traffic."""

    __slots__ = ("b1", "b2", "p", "t2")

    def __init__(self, capacity: Any) -> None:
        super().__init__(capacity)
        self.t2: OrderedDict[Any, Any] = OrderedDict()
        self.b1: OrderedDict[Any, None] = OrderedDict()
        self.b2: OrderedDict[Any, None] = OrderedDict()
        self.p = 0

    def __contains__(self, key: Any) -> bool:
        return key in self.t1 or key in self.t2

    def __len__(self) -> int:
        return len(self.t1) + len(self.t2)

    def get(self, key: Any) -> Any:
        """a day rider's second ride makes it a regular; a regular's ride renews its seat"""
        s = self.t1.pop(key, None)
        if s is not None:
            self.t2[key] = s
            return s
        s = self.t2.get(key)
        if s is not None:
            self.t2.move_to_end(key)
        return s

    def peek(self, key: Any) -> Any:
        s = self.t1.get(key)
        return s if s is not None else self.t2.get(key)

    def admit(self, key: Any, slot: Any) -> None:
        """a returning ghost is seated as a regular and moves the split its way; a stranger is a day rider"""
        if key in self.b1:
            self.p = min(int(self.capacity()), self.p + max(1, len(self.b2) // max(1, len(self.b1))))
            del self.b1[key]
            self.t2[key] = slot
        elif key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) // max(1, len(self.b2))))
            del self.b2[key]
            self.t2[key] = slot
        else:
            self.t1[key] = slot

    def _lines(self) -> list[tuple[Any, Any]]:
        day_first = len(self.t1) > self.p or not self.t2
        return [(self.t1, self.b1), (self.t2, self.b2)] if day_first else [(self.t2, self.b2), (self.t1, self.b1)]

    def victim(self, skip: Any = ()) -> tuple[Any, Any] | None:
        """the oldest day rider while the day riders hold more than their share, else the oldest regular; the
        bumped rider is remembered as a ghost of its line, the ghost lines as long as the seats"""
        c = int(self.capacity())
        for line, ghost in self._lines():
            for _ in range(len(line)):
                key = next(iter(line))
                s = line[key]
                if s in skip:
                    line.move_to_end(key)
                    continue
                del line[key]
                ghost[key] = None
                while len(ghost) > c:
                    ghost.popitem(last=False)
                return key, s
        return None

    def pop(self, key: Any) -> Any:
        s = self.t1.pop(key, None)
        return s if s is not None else self.t2.pop(key, None)

    def demote(self, key: Any) -> None:
        if key in self.t1:
            self.t1.move_to_end(key, last=False)
        elif key in self.t2:
            self.t2.move_to_end(key, last=False)

    def oldest_slot(self) -> Any:
        for line, _ in self._lines():
            for s in line.values():
                return s
        return None

    def items(self) -> Any:
        yield from self.t1.items()
        yield from self.t2.items()


class LayerDepot:
    """A layer's experts held on the card while a prefill sweeps its chunks through it (`_prefill_by_layer`): each
    expert crosses the bus once for the whole prompt, not once a chunk. An expert is seated the first time the
    layer asks for it, its copy queued on a stream of its own so the next expert's bytes move under this one's
    matmuls; the compute waits on that expert's copy alone. A new layer takes the seats over once the card is
    done with the last one's. The weights are the store's bytes as they are - bf16, or MXFP4 or FP8 as stored
    (`stored_parts`) - a slot holding one expert's parts; a product over a seated bf16 expert is the one the call's
    own upload (`w.to(card).to(x.dtype)`) gives.

    The depot grows as the layers ask of it and no further. It opens with the scratch slots alone, and a layer
    asking for an expert it has no seat for grows a block of `BLOCK` seats, out of what the device's ledger has
    free that nothing has spoken for - the pass's own working set and the cache's growth are - counted there as it
    is made and given back with the depot (`close`). Where the ledger has no block to give, an expert rides one of
    the scratch slots, on the same copy stream: uploaded again by every chunk that asks, but under the matmuls
    rather than as a pageable copy that waits for the card to drain; where it has not even the scratch, the call
    takes the loop. The blocks come from a pool of their own in torch's allocator, so they never split, or are
    split out of, the blocks the pass's own buffers are cut from.

    A call's experts go to the card in waves (`place`), each wave multiplied as one grouped matmul a block it
    touched: a slot's rows its expert's, an idle slot none.

    The store's pages are pageable (and on Windows a pinned store is refused), so a copy from them holds the host
    until it lands, at the driver's pageable rate. Each upload goes through a small ring of pinned buffers instead:
    the host copies the expert into one (a memcpy torch spreads over the cores), and the bus takes it from there at
    the pinned rate while the host moves on; a buffer is filled again once its last copy out has landed. The
    buffers are lent through the ledger too. `BTB_PREFILL_STAGE=0`, or a machine that will not pin the ring, uploads
    from the store's pages."""

    SCRATCH = 32
    BLOCK = 16
    STAGE = 4
    # a ceiling on the seats, below the ledger's (the tests' small depots)
    MAX_SEATS: int | None = None

    def __init__(self, dev: torch.device, ledger: Any, sched: Any = None) -> None:
        # the card by its index: a pool of torch's allocator is a device's own
        self.dev = where(dev)
        self.ledger = ledger
        self.sched = sched
        self.pool: torch.cuda.MemPool | None = torch.cuda.MemPool()
        self.copy = torch.cuda.Stream(device=dev)
        self.layer = -1
        self.seat: dict[int, int] = {}
        # the slots' blocks, each one stack a stored part: block 0 the scratch slots, the rest the seats as grown
        self.blocks: list[list[torch.Tensor]] = []
        self.where: list[tuple[int, int]] = []  # a slot's (block, row)
        # the stored form the first expert showed, every expert after the same: each part's shape and dtype
        self.form: tuple[tuple[torch.Size, torch.dtype], ...] | None = None
        self.per = 0
        # scratch j is free once the matmuls handed it are done: the event recorded at the next `get`, after
        # the caller queued them
        self.turn = 0
        self.lent: int | None = None
        self.free: list[torch.cuda.Event | None] = [None] * self.SCRATCH
        self.stat: dict[str, float] = {
            "seated": 0,
            "reused": 0,
            "scratch": 0,
            "passed": 0,
            "bytes": 0,
            "upload_s": 0.0,
            "blocks": 0,
            "held": 0,
            # calls the depot could not open for (no room for its scratch slots): the loop took them
            "refused": 0,
        }
        # a wave took the scratch slots over: the next `get` to take one waits for the card to finish what came before
        self.fence_scratch = False
        self.stage: list[tuple[torch.Tensor, ...]] | None = None
        self.stage_done: list[torch.cuda.Event | None] = [None] * self.STAGE
        self.stage_turn = 0
        self.staged = os.environ.get("BTB_PREFILL_STAGE", "1") != "0"

    @property
    def n_seats(self) -> int:
        """the seats grown so far"""
        return max(0, len(self.where) - self.SCRATCH)

    def _block(self, n: int) -> list[torch.Tensor] | None:
        """`n` slots of the depot's form, out of what the ledger has free that nothing has spoken for, asked of the
        scheduler as a large allocation is; None where there is not that much"""
        assert self.form is not None
        nbytes = n * self.per
        if self.pool is None:  # closed: nothing asked of the scheduler for a block never made
            return None
        # from the depot's own pool, which reuses none of torch's cached blocks: the card's free memory alone
        if nbytes > int(self.ledger.free(self.dev, unreserved=True, pooled=True) or 0):
            return None
        if self.sched is not None:
            try:
                self.sched.grant(nbytes, "depot", requester="LayerDepot", device=self.dev)
            except MemoryGrantError:
                return None
        try:
            with torch.cuda.use_mem_pool(self.pool, device=self.dev):
                block = [torch.empty((n, *shape), dtype=dt, device=self.dev) for shape, dt in self.form]
        except torch.OutOfMemoryError:
            return None
        for t in block:
            self.ledger.lend(lambda t=t: t, t.numel() * t.element_size(), self.dev, counted=False)
        self.stat["blocks"] += 1
        self.stat["held"] += nbytes
        return block

    def _fits(self, parts: tuple[torch.Tensor, ...]) -> bool:
        """whether `parts` has the depot's form, opened by the first expert: every expert after the same"""
        form = tuple((p.shape, p.dtype) for p in parts)
        if self.form is None:
            return self._open(form)
        return bool(self.blocks) and form == self.form

    def open_at(self, form: tuple[tuple[torch.Size, torch.dtype], ...], seats: int) -> int:
        """The depot opened at the store's `form` before the sweep's first call and grown to `seats` seats (a
        layer's experts), as far as the ledger has room: its blocks taken before the pass's own buffers are cut, so
        the card's free memory is theirs to take - a depot opened mid-sweep finds it inside the one block the pass's
        buffers split, which its own pool cannot use. Returns the seats it holds; 0 and closed where not even its
        scratch fits (the first call then asks again)."""
        if self.form is not None or not self._open(form):
            return self.n_seats
        while self.n_seats < int(seats) and (self.MAX_SEATS is None or self.n_seats < self.MAX_SEATS):
            block = self._block(self.BLOCK)
            if block is None:
                break
            b = len(self.blocks)
            self.blocks.append(block)
            self.where += [(b, j) for j in range(self.BLOCK)]
        return self.n_seats

    def _open(self, form: tuple[tuple[torch.Size, torch.dtype], ...]) -> bool:
        """the depot opened at `form`, each stored part's shape and dtype: its scratch slots, where the ledger has
        them"""
        self.form = form
        self.per = sum(int(torch.Size(shape).numel()) * torch.empty(0, dtype=dt).element_size() for shape, dt in form)
        scratch = self._block(self.SCRATCH)
        if scratch is None:
            # not open: the next call asks again, and may find the room this one did not
            self.form, self.per = None, 0
            self.stat["refused"] += 1
            return False
        self.blocks = [scratch]
        self.where = [(0, j) for j in range(self.SCRATCH)]
        if self.staged:
            cpu = torch.device("cpu")
            try:
                self.stage = [
                    tuple(
                        self.ledger.lend(
                            lambda shape=shape, dt=dt: torch.empty(shape, dtype=dt).pin_memory(),
                            int(torch.Size(shape).numel()) * torch.empty(0, dtype=dt).element_size(),
                            cpu,
                            counted=False,
                        )
                        for shape, dt in form
                    )
                    for _ in range(self.STAGE)
                ]
            except RuntimeError:  # the machine would not pin even these: the store's pages it is
                self.stage = None
        return True

    def takes(self, parts: tuple[torch.Tensor, ...] | None) -> bool:
        """whether the depot takes experts stored as `parts`: open at that form (the first call opens it), with
        its scratch slots at least"""
        return parts is not None and self._fits(parts)

    def _seat(self, e: int) -> int | None:
        """a seat for expert `e` of this layer: the next of the seats grown, or one of a block grown for it now;
        None where the ledger has no block to give"""
        k = len(self.seat)
        if self.MAX_SEATS is not None and k >= self.MAX_SEATS:
            return None
        if k >= self.n_seats:
            block = self._block(self.BLOCK)
            if block is None:
                return None
            b = len(self.blocks)
            self.blocks.append(block)
            self.where += [(b, j) for j in range(self.BLOCK)]
        s = self.seat[e] = self.SCRATCH + k
        return s

    def at(self, s: int) -> tuple[list[torch.Tensor], int]:
        """slot `s`'s block (one stack a part) and its row there"""
        b, j = self.where[s]
        return self.blocks[b], j

    def _upload(self, s: int, parts: tuple[torch.Tensor, ...], main: torch.cuda.Stream) -> None:
        stacks, row = self.at(s)
        t0 = time.perf_counter()
        ready = torch.cuda.Event()
        src: tuple[torch.Tensor, ...] = parts
        j = -1
        if self.stage is not None:
            # into the next pinned buffer once its last copy out has landed, then from there over the bus
            j = self.stage_turn
            self.stage_turn = (j + 1) % self.STAGE
            done = self.stage_done[j]
            if done is not None:
                done.synchronize()
            src = self.stage[j]
            for buf, p in zip(src, parts, strict=True):
                buf.copy_(p)
        with torch.cuda.stream(self.copy):
            for st, p in zip(stacks, src, strict=True):
                st[row].copy_(p, non_blocking=True)
            ready.record(self.copy)
        if j >= 0:
            self.stage_done[j] = ready
        main.wait_event(ready)
        self.stat["bytes"] += sum(p.numel() * p.element_size() for p in parts)
        # the host's share of the upload: the pageable copy until it lands, or the copy into a pinned buffer
        self.stat["upload_s"] += time.perf_counter() - t0

    def _slot(self, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        stacks, row = self.at(s)
        return stacks[0][row], stacks[1][row]

    def get(self, layer: int, e: int, gu: Any, dn: Any) -> tuple[Any, Any]:
        """bf16 expert `e` of `layer` on the card, for the per-expert loop: its seat, seated now, or a scratch slot;
        the host's views back for any other form (the loop takes those on the host's kernels)"""
        parts = stored_parts(gu, dn)
        if parts is None or len(parts) != 2:
            self.stat["passed"] += 1
            return gu, dn
        main = torch.cuda.current_stream(self.dev)
        if self.lent is not None:
            done = torch.cuda.Event()
            done.record(main)
            self.free[self.lent], self.lent = done, None
        if layer != self.layer:
            # the last layer's slots are the card's until its queued matmuls have read them
            done = torch.cuda.Event()
            done.record(main)
            self.copy.wait_event(done)
            self.layer, self.seat = layer, {}
        s = self.seat.get(e)
        if s is not None:
            self.stat["reused"] += 1
            return self._slot(s)
        if not self._fits(parts):
            self.stat["passed"] += 1
            return gu, dn
        s = self._seat(e)
        if s is not None:
            self._upload(s, parts, main)
            self.stat["seated"] += 1
            return self._slot(s)
        j = self.turn
        self.turn = (j + 1) % self.SCRATCH
        ev = self.free[j]
        if ev is not None:
            self.copy.wait_event(ev)
        if self.fence_scratch:
            done = torch.cuda.Event()
            done.record(main)
            self.copy.wait_event(done)
            self.fence_scratch = False
        self._upload(j, parts, main)
        self.lent = j
        self.stat["scratch"] += 1
        return self._slot(j)

    def place(self, layer: int, items: Sequence[tuple[int, Any, Any]]) -> list[int | None]:
        """A wave of a call's experts on the card together, for grouped matmuls over the slots: each expert's slot
        - its seat, seated now, or a scratch slot - in the order given, or None where it has no place this wave (the
        scratch is taken, or the stored form is not one the card path takes). The uploads are queued, the card's
        compute waits on each; the scratch is this wave's once the card is done with everything queued before it."""
        main = torch.cuda.current_stream(self.dev)
        before = torch.cuda.Event()
        before.record(main)
        if layer != self.layer:
            # the last layer's slots are the card's until its queued matmuls have read them
            self.copy.wait_event(before)
            self.layer, self.seat = layer, {}
        out: list[int | None] = []
        scratch = 0
        fenced = False
        for e, gu, dn in items:
            # a seat of this layer's first: an expert seated by an earlier chunk comes without its bytes in RAM
            s = self.seat.get(e)
            if s is not None:
                self.stat["reused"] += 1
                out.append(s)
                continue
            parts = stored_parts(gu, dn)
            if parts is None:
                self.stat["passed"] += 1
                out.append(None)
                continue
            if not self._fits(parts):
                self.stat["passed"] += 1
                out.append(None)
                continue
            s = self._seat(e)
            if s is not None:
                self._upload(s, parts, main)
                self.stat["seated"] += 1
                out.append(s)
                continue
            if scratch < self.SCRATCH:
                if not fenced:
                    self.copy.wait_event(before)
                    fenced = True
                s = scratch
                scratch += 1
                self._upload(s, parts, main)
                self.stat["scratch"] += 1
                out.append(s)
                continue
            out.append(None)
        if scratch:
            self.fence_scratch = True
        return out

    def seated(self, layer: int) -> set[int]:
        """the experts of `layer` with a seat on the card, copied by an earlier chunk of the layer: a chunk after
        it multiplies them from their seats, and needs nothing of them in RAM"""
        return set(self.seat) if layer == self.layer else set()

    def settle(self) -> None:
        """every queued copy landed: the store may hand the host bytes behind them to another expert after this"""
        self.copy.synchronize()

    def close(self) -> None:
        """the depot given back: its copies landed, its blocks and their pool let go"""
        self.settle()
        self.blocks, self.where, self.seat, self.stage = [], [], {}, None
        self.pool = None


class VramSeats:
    """The first-class seats: the regulars with the most rides copied onto the card, where an expert costs no
    host memory traffic and no read. A rider earns a seat with `min_rides` rides; the seats are given up by
    last ride; at most `per_pass` promotions a pass keep the copies off the token's time. The RAM copy stays
    (a seat given up falls back to it), so a seat costs the store nothing but the copy. bf16 experts only (an
    fp16/fp32 one is seated as the bf16 its slot holds): the card multiplies the stored tensors as they are."""

    def __init__(self, n_seats: int, per: int, shapes: Any, device: Any, min_rides: int = 8, per_pass: int = 4) -> None:
        self.n = int(n_seats)
        self.per = int(per)
        self.shapes = shapes
        self.device = device
        self.min_rides = int(min_rides)
        self.per_pass = int(per_pass)
        self.buf = torch.empty(self.n * self.per, dtype=torch.uint8, device=device) if self.n > 0 else None
        self.seat_of: OrderedDict[Any, int] = OrderedDict()
        self.free: list[int] = list(range(self.n))
        self.left = self.per_pass
        self.copies = 0

    def __contains__(self, key: Any) -> bool:
        return key in self.seat_of

    def views(self, key: Any) -> Any:
        j = self.seat_of[key]
        self.seat_of.move_to_end(key)
        assert self.buf is not None  # a seated key implies the depot buffer was allocated
        region = self.buf[j * self.per : (j + 1) * self.per]
        gu_n, gu_shape, dn_shape = self.shapes
        return (
            region[:gu_n].view(torch.bfloat16).view(*gu_shape),
            region[gu_n:].view(torch.bfloat16).view(*dn_shape),
        )

    def new_pass(self) -> None:
        self.left = self.per_pass

    def offer(self, key: Any, rides: int, region: torch.Tensor) -> bool:
        """a rider with `rides` rides and its bytes in `region` (the RAM slot): seated if it has earned it and
        the pass has a promotion left; the seat of the least recently ridden is taken when none is free"""
        if self.buf is None or key in self.seat_of or rides < self.min_rides or self.left <= 0:
            return False
        if self.free:
            j = self.free.pop()
        else:
            _old, j = self.seat_of.popitem(last=False)
        self.buf[j * self.per : (j + 1) * self.per].copy_(region[: self.per], non_blocking=False)
        self.seat_of[key] = j
        self.left -= 1
        self.copies += 1
        return True


class SlotState(enum.Enum):
    """what a slot of the store holds, and who it is for"""

    FREE = "free"  # nothing: the next seat taken
    RING = "ring"  # the lookahead ring's, holding nothing (its prediction was used, lapsed, or taken back)
    PREDICTED = "predicted"  # the ring's, a prediction for `key` read into it, landed or landing
    RESIDENT = "resident"  # the line's (`res`): `key`'s expert, landed or landing for a call that waits on it


@dataclass(slots=True, eq=False)
class Slot:
    """one slot of the store: where its bytes sit, what it holds, and the reads still writing them"""

    sid: int
    block: int
    row: int
    state: SlotState = SlotState.FREE
    key: tuple[int, int] | None = None
    parts: Parts | None = None  # the expert's reads into the slot, until they are settled or the slot is reused
    delta: tuple[int, ...] = ()  # where a padded direct read left each part in its region (the file's misalignment)
    bf16: bool = False  # an fp16/fp32 expert rewritten as bf16 in place since it was read
    depth: int = 0  # a prediction's lookahead depth

    def landing(self) -> bool:
        return self.parts is not None and not self.parts.done()


class _Either:
    """two slot sets asked as one - a wave's own slots and those of the call's experts still to come - without the
    union a seat at a time would copy"""

    __slots__ = ("a", "b")

    def __init__(self, a: Any, b: Any) -> None:
        self.a, self.b = a, b

    def __contains__(self, s: object) -> bool:
        return s in self.a or s in self.b


class StoreCall:
    """One MoE layer call's experts as the store serves them. `wave()` seats the longest prefix of what is left
    that the store can hold now - ascending, as a call lists them, so a row's experts are multiplied and summed in
    ascending order across the waves - and the caller multiplies it before asking for the next: a store held below
    a call's experts (the machine's commit can hold it near one layer's, and a long prompt's call asks for all of
    one) serves the call in turn instead of refusing it. `whole()` seats every one at once or raises: the paths
    that multiply a call's experts together."""

    __slots__ = ("base", "keep", "layer", "read", "rest", "rows", "store")

    def __init__(self, store: _ExpertStore, layer: int, base: str, ids: list[int], keep: bool, rows: int) -> None:
        self.store, self.layer, self.base, self.keep, self.rows = store, layer, base, keep, rows
        self.rest = ids
        self.read = 0  # the call's experts not in hand when their wave was seated: read, or still landing

    @property
    def done(self) -> bool:
        return not self.rest

    def wave(self) -> tuple[dict[int, Any], list[Any]]:
        """the next wave: (ready, pending), and `rest` what is left after it; nothing once the call is done"""
        if not self.rest:
            return {}, []
        ready, pending, self.rest = self.store._wave(self.layer, self.base, self.rest, self.keep, self.rows, False)
        self.read += len(pending)
        return ready, pending

    def whole(self) -> tuple[dict[int, Any], list[Any]]:
        ready, pending, _ = self.store._wave(self.layer, self.base, self.rest, self.keep, self.rows, True)
        self.rest = []
        self.read += len(pending)
        return ready, pending


def _line_size(store: _ExpertStore | None) -> int:
    """the store's line as its residency policy sizes it: the live slots less the lookahead's ring (none once the
    store is gone)"""
    return 0 if store is None else store.live() - len(store.ring)


class _ExpertStore:
    # a block of slots is at most this many bytes, so a shortfall under the reserve gives back a block's worth
    # and not the gigabytes of a first growth; and the store grows only `margin` above the reserve (a quarter
    # of it, at least a block), so a release under the reserve is never followed by a regrowth into the same
    # bytes - the two thresholds stand a margin apart
    BLOCK_MAX = 1 << 30

    _os: ModuleType
    blocks: dict[int, tuple[torch.Tensor, list[int]]]
    budget: int
    dt: torch.dtype  # a bf16-layout checkpoint's expert element type as stored (an fp16/fp32 one is cast once read)
    free: list[int]  # the slots in FREE, the next seat taken from the end
    last_slots: dict[int, int]  # the current wave's experts and their slots: the call is multiplying them
    lru: Any
    max_call: int
    mx: bool
    f8: bool
    f8_sdt: torch.dtype
    n_slots: int
    next_slot: int
    parked: list[int]
    per: int | None
    pool: ThreadPoolExecutor
    recipes: dict[int, Any]
    reserve: int
    scratch_n: int
    shapes: Any
    shared: Any
    sizes: Any
    slots: dict[int, Slot]  # every live slot's record: its state changes only through the transitions below
    sm: Any
    stat: Any

    def __init__(self, sm: Any, budget_bytes: int, reserve_bytes: int, readers: int = 16, scratch: int = 16) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.sm = sm
        self.recipes = {}
        self.per = None
        self.shapes = None
        self.sizes = None
        self.dt = torch.bfloat16
        self.mx = stored_mxfp4(sm.fam.mxfp4, getattr(sm, "gguf", None))
        # an FP8 checkpoint's experts: each projection's e4m3 bytes and its scale grid, multiplied as stored
        self.f8 = bool(getattr(sm, "fp8_experts", False))
        self.f8_sdt = torch.float32  # the scale grids' stored dtype, from the first recipe
        # a GGUF's experts: gate, up and down in ggml's block layout, multiplied as stored (btb/mxfp4.py)
        self.ggml = self.mx and getattr(sm, "gguf", None) is not None
        # a GGUF's other experts, which no kernel multiplies as stored: a read dequantizes the expert (its layout
        # inverted, `GGUFModel.get`) into the slot as the bf16 a checkpoint's read would land there
        self.dequant = not self.mx and getattr(sm, "gguf", None) is not None
        self.keys: dict[int, tuple[str, ...]] = {}  # the dequantized experts' tensor names by layer
        self.n_slots = 0
        self.budget = int(budget_bytes)
        self.reserve = int(reserve_bytes)
        self.block_max = int(self.BLOCK_MAX)
        self.margin = max(self.block_max, self.reserve // 4)
        self.scratch_n = int(scratch)
        # the residency policy: the store's line, or the Bus Pass (the configuration's `bus_pass`, BTB_BUS_PASS
        # over it); `lru` is the day riders' line, which in the store's own shape is the whole line. Read off
        # the model with no default of its own: a model built without a policy must fail, not run the wrong one
        bp = os.environ.get("BTB_BUS_PASS")
        bus_pass = bool(int(bp)) if bp not in (None, "") else bool(sm.bus_pass)
        # the line's size read through a weak reference: a closure holding the store would make it a cycle, left
        # with its gigabytes to the collector's next full pass once the engine lets it go
        me = weakref.ref(self)
        self.res = (BusPass if bus_pass else Riders)(lambda: _line_size(me()))
        self.res_tag = PassTag.EXPERT_BUS_PASS if bus_pass else PassTag.EXPERT_LINE
        self.lru = self.res.t1
        # the depot's pages held in RAM (`store_pin`, BTB_STORE_PIN over it: 0 pageable, 1 pinned, "auto" pinned
        # beside a card): a drive writing straight into a page the machine has trimmed pays the fault on the read
        sp: str | int | None = os.environ.get("BTB_STORE_PIN")
        sp = sp if sp not in (None, "") else sm.store_pin
        dev = getattr(sm, "dev", None)
        on_card = dev is not None and getattr(dev, "type", "") == Device.CUDA
        self.pin = on_card if str(sp).strip().lower() == "auto" else bool(int(sp or 0)) and on_card
        self.cached_reads = self._read_mode()
        self.free = []
        self.parked = []
        self.blocks = {}
        self.shared = {}
        self.last_slots = {}
        self.slots = {}
        self.next_slot = 0
        self.max_call = 0
        self.pool = ThreadPoolExecutor(max_workers=int(readers))
        self.readers = int(readers)
        # hit/miss per expert asked; bytes the misses' bytes; s the store's own time in get/wait; wait_s the
        # time a layer's forward blocked on its reads; read_s/read_n/read_max_s each expert read as the reader
        # saw it; calls and miss_calls per layer call; adjacent the misses next to another miss of the same
        # call in the tensor (one read could serve both)
        self.stat = {
            "hit": 0,
            "miss": 0,
            "bytes": 0,
            "s": 0.0,
            "blocks": 0,
            "released": 0,
            "wait_s": 0.0,
            "read_s": 0.0,
            "read_n": 0,
            "read_max_s": 0.0,
            "calls": 0,
            "miss_calls": 0,
            "adjacent": 0,
            "ahead": 0,
            "ahead_used": 0,
            "ahead_dropped": 0,
            "ahead_recycled": 0,
        }
        self._lock = threading.Lock()
        self._os = os
        # the Timetable's ring: slots that hold the lookahead's predictions until a layer asks for one (it is
        # promoted into the store) or the ring wraps (the slot is reused); never the store's own slots. Oldest
        # first, a set in order: a slot leaves it from anywhere in one step
        self.ring_n = 64
        self.ring: OrderedDict[int, None] = OrderedDict()
        # the layer a layer-by-layer prefill last read ahead from: one lookahead a layer, at its first chunk
        self.sweep_layer = -1
        # the predictions: (layer, expert) to the ring slot being read for it (a PREDICTED slot's key)
        self.ahead: dict[tuple[int, int], int] = {}
        # the model's first layer with experts, where a pass's first call is (`_first_layer`)
        self._first_moe: int | None = None
        self._routers: dict[int, Any] = {}
        # the slot's layout: `stride` bytes a slot, each part's region starting at `part_at[p]`. Padded (the
        # torch path with the positional reader), every region starts on a sector and holds two sectors of
        # slack, so a read of the aligned span around the expert's bytes lands straight in the slot with no
        # bounce; the expert then sits `slots[slot].delta[p]` bytes into its region, its file offset's own
        # misalignment. Unpadded (MLX's shared blocks, the reader without handles): the parts back to back
        self.stride: Any = None
        self.part_at: tuple[int, ...] = ()
        self.padded = False
        self._sizes_of: dict[str, int] = {}
        # the first-class seats on the card (`vram_experts_gb`: 0 none, "auto" what the card has to spare, a
        # figure in GB), made on the first recipe; `rides` counts every ask per (layer, expert) for the seating
        self.vram: VramSeats | None = None
        # the drive under the shards (the Route's profile, taken on the first recipe) and what the first one-row
        # pass said about it: saturated when its misses' reads outlast the pass's own compute
        self.drive: dict[str, Any] | None = None
        self.saturated = False
        self._pass: dict[str, Any] | None = None
        self._pass_samples: list[tuple[float, float, int, float, float]] = []
        self._decided = False
        self._warned_prefill = False
        self.rides: dict[tuple[int, int], int] = {}

    def live(self) -> int:
        return sum(len(ids) for _, ids in self.blocks.values())

    @staticmethod
    def _read_mode() -> bool:
        """Whether a miss is read through the system's file cache. Where the host's commit, not its RAM, is what
        the store can grow into (Windows charges commit for every allocation and the page file bounds it, while
        the file cache's pages are RAM that no commit is charged for), the cache is a second RAM tier the store
        cannot otherwise have: a miss read through it fills it, and the same expert missed again while it holds
        the bytes is a copy out of RAM - 2 ms against the drive's 25 for a 120B's expert on an SSD. Elsewhere the
        cache and the store draw on the same RAM (on unified memory, the RAM the model runs in), so a miss reads
        around it. `BTB_EXPERT_READS` (cached, direct) decides where it is set."""
        env = (os.environ.get("BTB_EXPERT_READS") or "").strip().lower()
        if env in ("cached", "direct"):
            return env == "cached"
        from ..sysinfo import host_commit_bytes, host_free_bytes

        # free-read: which of the host's limits binds chooses how a miss is read, never what fits
        return int(host_commit_bytes()) < int(host_free_bytes())

    def close(self) -> None:
        """The store given back whole: its readers stopped, every read still writing into a slot settled, then its
        blocks, its card seats and every record of what sat where let go. The engine's `close` calls it after the
        drive's readers are stopped; the store is unusable afterwards."""
        self.pool.shutdown(wait=True)
        for sl in self.slots.values():
            if sl.parts is not None:
                sl.parts.settle()  # a read in flight still writes into its slot
        self.blocks.clear()
        self.shared.clear()
        self.vram = None
        self.res = (type(self.res))(lambda: 0)
        self.lru = self.res.t1
        self.ring.clear()
        self.ahead.clear()
        self.free, self.parked = [], []
        self.slots.clear()
        self.last_slots.clear()
        self.n_slots = 0

    # -- the slot table's transitions: the one way a slot changes hands --------------------------------------------

    def check(self) -> None:
        """The slot table's invariants, raised as an AssertionError where one fails (the store's tests run it after
        every wave and lookahead): every live slot recorded once; the free list exactly the FREE slots; the ring
        only its own (RING or PREDICTED); every prediction's slot PREDICTED under its key, in the ring; every seat
        on the line RESIDENT under its key."""
        live = {s for _b, ids in self.blocks.values() for s in ids}
        if set(self.slots) != live:
            raise AssertionError(f"slots recorded {sorted(set(self.slots) ^ live)} differ from the blocks' live ones")
        if len(set(self.free)) != len(self.free):
            raise AssertionError(f"a slot twice in the free list: {self.free}")
        free = set(self.free)
        for sl in self.slots.values():
            if (sl.state is SlotState.FREE) != (sl.sid in free):
                raise AssertionError(f"slot {sl.sid} is {sl.state.value} but {'' if sl.sid in free else 'not '}free")
            if (sl.state in (SlotState.RING, SlotState.PREDICTED)) != (sl.sid in self.ring):
                raise AssertionError(
                    f"slot {sl.sid} is {sl.state.value} but {'' if sl.sid in self.ring else 'not '}in the ring"
                )
        for key, s in self.ahead.items():
            sl = self.slots[s]
            if sl.state is not SlotState.PREDICTED or sl.key != key:
                raise AssertionError(f"prediction {key} names slot {s}, which is {sl.state.value} for {sl.key}")
        for key, s in self.res.items():
            sl = self.slots[s]
            if sl.state is not SlotState.RESIDENT or sl.key != key:
                raise AssertionError(f"line seat {key} names slot {s}, which is {sl.state.value} for {sl.key}")
        counts = dict.fromkeys(SlotState, 0)
        for sl in self.slots.values():
            counts[sl.state] += 1
        if counts[SlotState.PREDICTED] != len(self.ahead) or counts[SlotState.RESIDENT] != len(self.res):
            raise AssertionError(
                f"{counts[SlotState.PREDICTED]} predicted slots for {len(self.ahead)} predictions, "
                f"{counts[SlotState.RESIDENT]} resident for {len(self.res)} seats on the line"
            )

    def _taken(self, s: int) -> Slot:
        """slot `s` out of whatever held it, empty and nobody's, for its taker to fill: a prediction or a line's
        seat given up, its reads settled first - a part still in flight writes into the slot, and the slot is
        never handed on while it does"""
        sl = self.slots[s]
        if sl.parts is not None:
            sl.parts.settle()
        if sl.state is SlotState.PREDICTED and sl.key is not None:
            self.ahead.pop(sl.key, None)
        self.ring.pop(s, None)
        sl.state, sl.key, sl.parts, sl.depth, sl.bf16, sl.delta = SlotState.FREE, None, None, 0, False, ()
        return sl

    def _resident(self, s: int, key: tuple[int, int], parts: Parts | None = None) -> None:
        """slot `s` the line's seat for `key`: a miss about to be read into it (`parts` set once queued), or a
        prediction promoted with the reads it already has"""
        sl = self.slots[s]
        self.ring.pop(s, None)
        if sl.state is SlotState.PREDICTED:
            self.ahead.pop(key, None)
        else:
            sl.parts = parts
        sl.state, sl.key = SlotState.RESIDENT, key
        self.res.admit(key, s)

    def _predicted(self, s: int, key: tuple[int, int], parts: Parts, depth: int) -> None:
        """ring slot `s` read for the prediction `key`"""
        sl = self.slots[s]
        sl.state, sl.key, sl.parts, sl.depth = SlotState.PREDICTED, key, parts, depth
        self.ahead[key] = s

    def _to_ring(self, s: int, first: bool = False) -> None:
        """slot `s` the ring's, empty (settled by `_taken`): newest, or with `first` the next reused"""
        self._taken(s).state = SlotState.RING
        self.ring[s] = None
        if first:
            self.ring.move_to_end(s, last=False)

    def _to_free(self, s: int) -> None:
        """slot `s` back among the free, its reads settled"""
        self._taken(s)
        self.free.append(s)

    def _seat_for(
        self, skip: set[int], protect: Container[int] = frozenset(), ring: bool = False, last: bool = False
    ) -> int | None:
        """A slot for an expert about to be read, by the one order every taker keeps: a free one; one the store
        grows (the host's ledger allowing - a whole block while the room holds one); the line's oldest seat whose
        slot is not in `skip` (a wave's own, being multiplied) nor `protect` (the call's experts still to come);
        with `ring`, a prediction taken back - the call before a guess - whose reads have landed or can be
        withdrawn whole; and with `last`, one of `protect` after all (a wave's first expert: a wave must seat one
        to go on, and any later one gains nothing by unseating an expert the call still wants). None when none of
        these has one."""
        if not self.free and self.live() < self.n_slots:
            self._grow(1)
        if self.free:
            s = self.free.pop()
            self.slots[s].state = SlotState.FREE
            return s
        v = self.res.victim(_Either(skip, protect))
        if v is None and ring:
            got = self._ring_take(skip)
            if got is not None:
                return got
        if v is None and last:
            v = self.res.victim(skip)
        if v is None:
            return None
        key, s = v
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None:
            prof.add(prof.EVICT, key[0], key[1], self.per, slot=s, aux=0)
        self._taken(s)
        return s

    def _grow(self, need: int) -> Any:
        assert self.per is not None  # the store is sized before this runs
        room = self.n_slots - self.live()
        if room <= 0:
            return 0
        # the device's ledger on the host: above the reserve, less what is spoken for (on MLX its own count of what
        # the load started with less what MLX holds, exact whether or not the pages are touched)
        usable = self._host_free() - self.margin
        # a whole block while the room above the margin holds one, then what is left of it (the fill is a few
        # blocks, never a trickle of ever smaller ones), at least what the call needs
        k = max(int(need), int(min(usable - 2 * self.per, self.block_max) // self.per))
        k = min(k, room)
        if self.sm.mlx is not None:
            # an MLX array's dimension is an int32: a shared block stays under 2 GB (gpt-oss's 13 MB
            # slots: 162 a block) and the store grows by several blocks instead
            k = min(k, max(int(need), (2**31 - 1) // self.per))
        if k < int(need) or usable < (k + 2) * self.per:
            return 0
        ids: list[int] = []
        while len(ids) < k:
            ids.append(self.parked.pop() if self.parked else self.next_slot)
            if ids[-1] == self.next_slot:
                self.next_slot += 1
        b = self.stat["blocks"]
        self.stat["blocks"] += 1
        stride = int(self.stride or self.per)
        if self.sm.mlx is not None:
            sh = mlxdev.Shared(k * stride)
            self.shared[b] = sh
            buf = sh.torch
        else:
            # a sector over, and the block's start moved up to the sector: every slot and every part region then
            # sits on a 4 KB boundary, which is what an unbuffered read straight into it needs
            raw = None
            if self.pin:
                try:
                    raw = torch.empty(k * stride + 4096, dtype=torch.uint8, pin_memory=True)
                except RuntimeError as e:
                    self.pin = False
                    self.sm.log(
                        f"[experts] store: the machine would not pin a {k * stride / 2**30:.2f} GB block ({e}); "
                        "pageable from here"
                    )
            if raw is None:
                raw = torch.empty(k * stride + 4096, dtype=torch.uint8)
                # a byte a page written now: the OS counts a page only once it is touched, and a block it cannot see
                # would read as free to the ledger until the reads fill it, spent again by the next growth
                raw[::4096].zero_()
            skew = (-raw.data_ptr()) % 4096
            buf = raw[skew : skew + k * stride]
        self.blocks[b] = (buf, ids)
        for j, s in enumerate(ids):
            self.slots[s] = Slot(s, b, j)
        self.free.extend(ids)
        free_now = self._host_free()
        self.sm.log(
            f"[experts] store +{k} slots ({k * self.per / 2**30:.2f} GB), {self.live()} of {self.n_slots} live, "
            f"{self._host_line()}"
        )
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None:
            prof.add(prof.GROW, expert=k, nbytes=k * self.per, aux=int(free_now))
        return k

    def _release_block(self, b: Any) -> None:
        """block `b` back to the machine, and every slot in it off its line, out of the ring and out of the free: a
        prediction's queued reads withdrawn (drive time for nothing), one still in flight left to its reader, whose
        view of the block keeps its bytes alive until it lands"""
        assert self.per is not None  # the store is sized before this runs
        buf, ids = self.blocks.pop(b)
        self.shared.pop(b, None)
        gone = set(ids)
        prof = getattr(self.sm, "expert_profile", None)
        sched = getattr(self.sm, "scheduler", None)
        for s in ids:
            sl = self.slots.pop(s)
            if sl.state is SlotState.RESIDENT and sl.key is not None:
                if prof is not None:
                    prof.add(prof.EVICT, sl.key[0], sl.key[1], self.per, slot=s, aux=1)
                self.res.pop(sl.key)
            elif sl.state is SlotState.PREDICTED and sl.key is not None:
                if sched is not None and hasattr(sched, "disk_drop"):
                    sched.disk_drop(sl.key)
                self.ahead.pop(sl.key, None)
            self.ring.pop(s, None)
        self.free = [s for s in self.free if s not in gone]
        self.parked.extend(ids)
        self.stat["released"] += len(ids)
        del buf
        if prof is not None:
            prof.add(prof.RELEASE, expert=len(ids), nbytes=len(ids) * self.per, aux=int(self._host_free()))

    def _host_free(self) -> int:
        """what the device's ledger has free on the host: above the reserve, less what is spoken for"""
        return int(self.sm.device.free(torch.device("cpu"), unreserved=True) or 0)

    def _host_line(self) -> str:
        """the host's figure and what it is made of: the RAM the OS has available, the commit it has left (the
        tighter of the two is the ledger's), the reserve, and what the ledger holds spoken for there"""
        from ..sysinfo import host_commit_bytes, host_free_bytes

        cpu = torch.device("cpu")
        held = self.sm.device.spoken_for(cpu)
        G = 2**30
        # free-read: the log's breakdown of the ledger's host figure, never a decision
        ram = host_free_bytes()
        # free-read: the same breakdown's commit
        commit = host_commit_bytes()
        return (
            f"{self._host_free() / G:.1f} GB free to btb: {ram / G:.1f} GB of RAM available, "
            f"{commit / G:.1f} GB of commit left, less the {self.reserve / G:.1f} GB reserve"
            + ("".join(f", {k} {n / G:.2f} GB" for k, n in held.items()) if held else "")
        )

    def releasable(self) -> int:
        """the host bytes `release` could give the machine: the blocks above the slots the largest call served
        needs, at what a slot holds on average (blocks go whole) - a bound for sizing, not a promise"""
        live = self.live()
        if live <= self.max_call or not self.blocks:
            return 0
        held = sum(int(getattr(buf, "nbytes", 0)) for buf, _ids in self.blocks.values())
        return held * (live - self.max_call) // live

    def release(self, want: int = 1) -> Any:
        """blocks given back, oldest residents' first, until the ledger has `want` bytes free on the host above the
        reserve (by default: while it has none), never below the largest call the store has served"""
        freed = 0
        while self.blocks and self._host_free() < want:
            victim = None
            s = self.res.oldest_slot()
            if s is not None:
                victim = self.slots[s].block
            if victim is None:
                victim = next(iter(self.blocks))
            if self.live() - len(self.blocks[victim][1]) < self.max_call:
                # below the largest call served the next call cannot be served at all, so the store holds here and
                # leaves the reserve to the machine's paging
                if not self.stat.get("floor"):
                    self.stat["floor"] = 1
                    self.sm.log(
                        f"[experts] store holds {self.live()} slots for calls of {self.max_call}: "
                        f"{self._host_free() / 2**30:.1f} GB free above the reserve, "
                        f"{want / 2**30:.1f} GB asked"
                    )
                break
            freed += len(self.blocks[victim][1])
            self._release_block(victim)
        if freed:
            self.sm.log(
                f"[experts] store -{freed} slots for the machine, {self.live()} live, "
                f"{self._host_free() / 2**30:.1f} GB free above the reserve"
            )
        return freed

    def _recipe(self, layer: int, base: str) -> Any:
        r = self.recipes.get(layer)
        if r is None:
            # bf16 experts are two tensors per expert, MXFP4 four (blocks then scales), each as the checkpoint has
            # them (btb/mxfp4.py); a GGUF's three (gate, up, down), the file's own tensors
            if self.ggml:
                names = tuple(f"blk.{layer}.ffn_{k}_exps.weight" for k in ("gate", "up", "down"))
            elif self.mx:
                names = ("gate_up_proj_blocks", "gate_up_proj_scales", "down_proj_blocks", "down_proj_scales")
            elif self.f8:
                names = ("gate_up_proj", "gate_up_proj_scale_inv", "down_proj", "down_proj_scale_inv")
            else:
                names = ("gate_up_proj", "down_proj")
            parts = []
            for name in names:
                key = name if self.ggml else base + name
                shard = self.sm.weight_map[key]
                _mm, hdr, hoff = self.sm._shard(shard)
                info = hdr[key]
                a, b = info["data_offsets"]
                shape = tuple(int(x) for x in info["shape"])
                if (b - a) % shape[0]:
                    raise RuntimeError(f"[experts] {key}: {b - a} bytes do not divide by {shape[0]} experts")
                parts.append((self._os.path.join(self.sm.dir, shard), hoff + a, (b - a) // shape[0], shape[1:]))
                if self.f8 and name.endswith("_scale_inv"):
                    self.f8_sdt = self.sm.ST_DTYPES[info["dtype"]]
                elif not (self.ggml or self.mx or self.f8):
                    self.dt = self.sm.ST_DTYPES[info["dtype"]]
            if self.dequant:
                self.keys[layer] = tuple(base + name for name in names)
            r = self.recipes[layer] = parts
            sched = getattr(self.sm, "scheduler", None)
            if sched is not None and hasattr(sched, "disk"):
                # the drive under the shard, probed on the first recipe that touches it
                for p in parts:
                    d = sched.disk(p[0])
                    if self.drive is None and d.get("measured"):
                        self.drive = d
            per = sum(p[2] for p in parts)
            shapes = tuple(p[3] for p in parts)
            if self.shapes is None:
                self.per = per
                # bf16: (gate_up bytes, gate_up shape, down shape), what `_views` splits the slot on;
                # MXFP4: the four tensors' per-expert shapes, and `sizes` their byte counts
                self.shapes = shapes if self.mx or self.f8 else (parts[0][2], *shapes)
                self.sizes = tuple(p[2] for p in parts)
                # reads straight into the slot (padded regions) unless BTB_STORE_PADDED=0 asks for the bounce
                self.padded = (
                    self.sm.mlx is None
                    and not self.dequant
                    and Native.read_at is not None
                    and os.environ.get("BTB_STORE_PADDED", "1") != "0"
                )
                self.stride, self.part_at = self._layout(self.sizes, self.padded)
                # every layer's experts: the trunk's, and a drafting layer's (Qwen4's MTP layer is the store's layer L)
                drafting = {
                    k.split(".mlp.experts.")[0]
                    for k in self.sm.weight_map
                    if k.startswith("mtp.") and ".mlp.experts." in k
                }
                total = (int(self.sm.L) + len(drafting)) * int(self.sm.n_experts)
                self.n_slots = min(total, max(self.scratch_n + 1, int(self.budget // self.stride)))
                # the ring is carved out of the store: an eighth of it at most, none of a store too small to spare
                self.ring_n = min(self.ring_n, self.n_slots // 8)
                self._open_vram()
                self.sm.log(
                    f"[experts] store: up to {self.n_slots} slots x {per / 2**20:.1f} MB = "
                    f"{self.n_slots * per / 2**30:.1f} GB in RAM, allocated as needed in blocks of at most "
                    f"{self.block_max / 2**30:.1f} GB, {self.margin / 2**30:.1f} GB above a "
                    f"{self.reserve / 2**30:.1f} GB reserve, {'pinned' if self.pin else 'pageable'}, "
                    f"{self.pool._max_workers} readers, "
                    + (
                        "misses read through the file cache (RAM past the commit limit holds them for a re-read)"
                        if self.cached_reads
                        else "misses read around the file cache"
                    )
                )
                if self.drive is not None:
                    self.sm.log(self.drive_report(self.drive, self.n_slots, total, per))
            elif per != self.per or tuple(p[2] for p in parts) != self.sizes:
                raise RuntimeError(f"[experts] layer {layer} expert shape differs from layer 0")
        return r

    def _open_vram(self) -> None:
        """the card's seats for the bf16 experts, sized by `sm.vram_experts_gb`: a figure, or "auto" for the
        card's free memory less its margin and a gigabyte kept for the cache to grow into"""
        assert self.per is not None  # the store is sized before this runs
        want = getattr(self.sm, "vram_experts_gb", 0)
        dev = getattr(self.sm, "dev", None)
        if self.mx or self.f8 or dev is None or dev.type != Device.CUDA or not want or self.stride is None:
            return
        if want == "auto":
            sched = getattr(self.sm, "scheduler", None)
            # free as the grant below reads it: less every reservation, an epoch's KV too (its room is the cache's,
            # not the seats'), so the seats sized here are seats the grant gives
            free = sched.free_for(dev, draws="") if sched is not None else None
            room = (free or 0) - (1 << 30)
        else:
            room = int(float(want) * 2**30)
        per = self._held(self.per)
        n = max(0, int(room // per))
        if n <= 0:
            return
        # the seats asked of the scheduler before they are made, and lent through the ledger while they live
        sched = getattr(self.sm, "scheduler", None)
        if sched is not None:
            try:
                sched.grant(n * per, "experts", requester="the expert store's seats on the card", device=dev)
            except MemoryGrantError as e:
                self.sm.log(f"[experts] no seats on the card: {e}")
                return
        gu_n, gu_shape, dn_shape = self.shapes
        self.vram = VramSeats(n, per, (self._held(gu_n), gu_shape, dn_shape), dev)
        buf, ledger = self.vram.buf, getattr(self.sm, "device", None)
        if ledger is not None and buf is not None:
            ledger.lend(lambda: buf, n * per, dev, counted=False)
        self.sm.log(f"[experts] {n} seats on the card ({n * per / 2**30:.2f} GB) for the most ridden experts")

    @staticmethod
    def expert_s(drive: dict[str, Any], per: int) -> float:
        """what a missed expert costs on `drive` from the Route's probe: two fixed costs (its two parts) and its
        bytes at the single-stream rate; 0 without a probe"""
        big = float(drive.get("big_mb", 0.0)) * 2**20
        single = float(drive.get("single_ms", 0.0)) / 1e3
        if big <= 0 or single <= 0:
            return 0.0
        return 2 * float(drive.get("fixed_ms", 0.0)) / 1e3 + per * single / big

    @classmethod
    def drive_report(cls, drive: dict[str, Any], n_slots: int, total: int, per: int) -> str:
        """what a user with this drive is buying, said at load from the probe and the store's plan: a missed
        expert's cost and the store's share of the experts (a token's misses times the one give the other)"""
        cost = cls.expert_s(drive, per)
        if cost <= 0:
            return "[experts] this drive: not measured"
        return (
            f"[experts] this drive: a missed expert costs {cost * 1e3:.1f} ms; the store seats {n_slots} of {total} "
            f"experts ({100.0 * n_slots / max(1, total):.0f}%), and a token waits about its misses times that"
        )

    def reads(self) -> int:
        """the expert reads the store has put to the drive so far: its misses and the lookahead's predictions not
        withdrawn - a pass's count, taken before and after it, is what the speculative pricing charges its rows"""
        st = self.stat
        return int(st["miss"] + st["ahead"] - st["ahead_dropped"])

    def waited(self) -> float:
        """the seconds the forwards have waited on the store's reads so far (`wait_s`): a pass's, taken before and
        after it, is what the speculative pricing takes out of its seconds to leave the rows' compute"""
        return float(self.stat["wait_s"])

    def miss_s(self) -> float:
        """a missed expert's seconds: the drive's probe (`expert_s`); without one, the reads timed so far spread
        over the readers that ran them side by side; 0 before either"""
        cost = self.expert_s(self.drive or {}, int(self.per or 0))
        if cost > 0:
            return cost
        n = int(self.stat["read_n"])
        return float(self.stat["read_s"]) / n / max(1, self.readers) if n else 0.0

    PASS_SAMPLES = 16

    def _pass_closed(self, wall: float, wait: float, misses: int) -> str:
        """A one-row pass closed: its misses' drive time against its compute, over the last PASS_SAMPLES passes.
        Their medians (one pass after a prefill is a reload, not a rate) say whether the drive is saturated,
        and then the Timetable withholds predictions: each would take a read from the layer waiting now. Taken
        again every pass; returns the log line when the verdict is first taken or turns, else ''."""
        cost = self.expert_s(self.drive or {}, int(self.per or 0))
        if cost <= 0:
            return ""
        self._pass_samples.append((misses * cost, max(0.0, wall - wait), misses, wait, wall))
        if len(self._pass_samples) > self.PASS_SAMPLES:
            del self._pass_samples[: len(self._pass_samples) - self.PASS_SAMPLES]
        if len(self._pass_samples) < self.PASS_SAMPLES:
            return ""
        drive_s = float(np.median([s[0] for s in self._pass_samples]))
        compute = float(np.median([s[1] for s in self._pass_samples]))
        misses_md = float(np.median([s[2] for s in self._pass_samples]))
        saturated = drive_s > compute
        turned = saturated != self.saturated
        first = not self._decided
        self._decided = True
        self.saturated = saturated
        if not (first or turned):
            return ""
        return (
            f"[experts] the last {len(self._pass_samples)} one-row passes: a median {misses_md:.0f} misses = "
            f"{drive_s:.2f} s of the drive against {compute:.2f} s of compute: "
            + ("the drive is saturated, no predictions" if saturated else "the drive has room")
        )

    def _seat(self, layer: int, e: int, slot: Any) -> None:
        """after a ride: the expert offered a seat on the card once it has ridden enough"""
        assert self.per is not None  # the store is sized before this runs
        v = self.vram
        if v is None or v.buf is None:
            return
        key = (layer, e)
        if key in v.seat_of or self.rides.get(key, 0) < v.min_rides or v.left <= 0:
            return
        self._to_bf16(slot)
        region = self._region(slot)
        d = self.slots[slot].delta or (0,) * len(self.sizes)
        at0 = (self.part_at[0] if self.part_at else 0) + d[0]
        at1 = (self.part_at[1] if self.part_at else self.sizes[0]) + d[1]
        if self.padded or self.dt != torch.bfloat16:
            # the seat holds the two parts' bf16 back to back: gathered from their padded regions, or from the
            # head of each part a float32 expert was rewritten into
            h0, h1 = self._held(self.sizes[0]), self._held(self.sizes[1])
            packed = torch.cat([region[at0 : at0 + h0], region[at1 : at1 + h1]])
        else:
            packed = region[at0 : at0 + self.per]
        v.offer(key, self.rides.get(key, 0), packed)

    @staticmethod
    def _layout(sizes: Sequence[int], padded: bool) -> tuple[int, tuple[int, ...]]:
        """(stride, the parts' region starts) for slots of parts of `sizes` bytes: back to back, or padded so
        each region starts on a sector with two sectors of slack for the expert's misalignment"""
        at, pos = [], 0
        for n in sizes:
            at.append(pos)
            pos += (-(-(int(n) + 8192) // 4096) * 4096) if padded else int(n)
        return pos, tuple(at)

    def _size_of(self, path: str) -> int:
        n = self._sizes_of.get(path)
        if n is None:
            n = self._sizes_of[path] = self._os.path.getsize(path)
        return n

    def _region(self, slot: Any) -> Any:
        assert self.per is not None  # the store is sized before this runs
        sl = self.slots[slot]
        stride = int(self.stride or self.per)
        return self.blocks[sl.block][0][sl.row * stride : (sl.row + 1) * stride]

    def _part(self, slot: Any, region: Any, p: int) -> Any:
        """part `p` of the expert in `slot`, as the bytes sit in the region"""
        sl = self.slots.get(slot) if slot is not None else None
        at = (self.part_at[p] if self.part_at else sum(self.sizes[:p])) + (
            (sl.delta if sl is not None else ()) or (0,) * 8
        )[p]
        return region[at : at + self.sizes[p]]

    def f8_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """the FP8 experts' shapes, (gate_up [2I, H], down [H, I])"""
        return (int(self.shapes[0][0]), int(self.shapes[0][1])), (int(self.shapes[2][0]), int(self.shapes[2][1]))

    def mx_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """the MXFP4 experts' logical shapes, (gate_up [2I, H], down [H, I]), in either layout"""
        if self.ggml:
            (rows, k), _, (drows, dk) = self.shapes
            return (2 * int(rows), int(k)), (int(drows), int(dk))
        gu, dn = self.shapes[0], self.shapes[2]
        return (int(gu[0]), int(gu[1]) * BLOCK), (int(dn[0]), int(dn[1]) * BLOCK)

    def form(self) -> tuple[tuple[torch.Size, torch.dtype], ...] | None:
        """an expert's parts as the card takes them (`stored_parts`), each part's shape and dtype, off the store's
        layout alone - no expert read: what a prefill's depot opens at before its first call. The layout is read off
        the first layer with experts where no call has read it yet (headers only); None where the card takes no
        expert of this form"""
        if self.shapes is None:
            for i in range(int(self.sm.L)):
                try:
                    self._recipe(i, f"{self.sm.prefix}layers.{i}.mlp.experts.")
                    break
                except KeyError:  # a dense layer: no experts to read the layout off
                    continue
            else:
                return None
        region = torch.zeros(int(self.stride or self.per or 0), dtype=torch.uint8)
        parts = stored_parts(*self._views_in(region))
        return None if parts is None else tuple((p.shape, p.dtype) for p in parts)

    def _views(self, slot: Any) -> Any:
        return self._views_in(self._region(slot), slot)

    def _views_in(self, region: Any, slot: Any = None) -> Any:
        """an expert's views over `region`, its slot's bytes (`slot` for where a direct read left them)"""
        if self.ggml:
            (rows, k), _, (drows, dk) = self.shapes
            gate = MxWeight.from_ggml(self._part(slot, region, 0), int(rows), int(k))
            up = MxWeight.from_ggml(self._part(slot, region, 1), int(rows), int(k))
            return MxGateUp(gate, up), MxWeight.from_ggml(self._part(slot, region, 2), int(drows), int(dk))
        if self.mx:
            out = []
            for i in (0, 2):
                rows, g = int(self.shapes[i][0]), int(self.shapes[i][1])
                out.append(MxWeight(self._part(slot, region, i), self._part(slot, region, i + 1), rows, g * BLOCK))
            return out[0], out[1]
        if self.f8:
            pair = []
            for i in (0, 2):
                (rows, cols), grid = self.shapes[i], self.shapes[i + 1]
                raw = self._part(slot, region, i + 1)
                # a scale part a direct read left off its element alignment is copied to one
                if raw.data_ptr() % self.f8_sdt.itemsize:
                    raw = raw.clone()
                s = raw.view(self.f8_sdt).reshape(*grid)
                pair.append(F8Weight(self._part(slot, region, i), s, int(rows), int(cols)))
            return pair[0], pair[1]
        _gu_n, gu_shape, dn_shape = self.shapes
        if self.dt == torch.bfloat16:
            return (
                self._part(slot, region, 0).view(torch.bfloat16).view(*gu_shape),
                self._part(slot, region, 1).view(torch.bfloat16).view(*dn_shape),
            )
        # an fp16/fp32 expert as `_to_bf16` left it (a view taken before it lands has the shapes only)
        gu = self._part(slot, region, 0)[: self._held(self.sizes[0])].view(torch.bfloat16).view(*gu_shape)
        dn = self._part(slot, region, 1)[: self._held(self.sizes[1])].view(torch.bfloat16).view(*dn_shape)
        return gu, dn

    def _held(self, n: int) -> int:
        """the bytes of `n` stored bytes of expert once held as bf16"""
        return n // self.dt.itemsize * 2

    def _to_bf16(self, slot: Any) -> None:
        """a landed fp16/fp32 expert rewritten as bf16 in its slot, once per read, so every use after reads it as
        a bf16 expert's bytes"""
        if self.dt == torch.bfloat16:
            return
        sl = self.slots[slot]
        if sl.bf16:
            return
        region = self._region(slot)
        for p in (0, 1):
            bf16_in_place(self._part(slot, region, p), self.dt)
        sl.bf16 = True

    def _views_mx(self, slot: Any) -> Any:
        assert self.per is not None  # the store is sized before this runs
        sl = self.slots[slot]
        sh = self.shared[sl.block]
        off = sl.row * int(self.stride or self.per)
        be = self.sm.mlx
        if self.mx:
            # the slot as the GPU's MXFP4 matvec reads it: gate_up's blocks then scales, then down's (a GGUF's:
            # gate, up, down in ggml's blocks), as uint8 views of the shared block (the store's readers write,
            # the kernel reads, the same bytes)
            region = sh.view_mx(off, self.per, mlxdev.mx().uint8, (self.per,))
            if self.ggml:
                at = [self.part_at[p] if self.part_at else sum(self.sizes[:p]) for p in range(3)]
                parts = [region[at[p] : at[p] + self.sizes[p]] for p in range(3)]
                return (parts[0], parts[1]), parts[2]
            n0 = self.sizes[0] + self.sizes[1]
            return region[:n0], region[n0:]
        gu_n, gu_shape, dn_shape = self.shapes
        if self.dt == torch.bfloat16:
            return be.weight_slot(sh, off, gu_n, gu_shape), be.weight_slot(sh, off + gu_n, self.per - gu_n, dn_shape)
        return (
            be.weight_slot(sh, off, self._held(gu_n), gu_shape),
            be.weight_slot(sh, off + gu_n, self._held(self.per - gu_n), dn_shape),
        )

    def _note_read(
        self, prof: Any, layer: int, e: int, part: int, path: str, off: int, n_e: int, slot: Any, dur_ns: int
    ) -> None:
        """one part of an expert landed: the profile's row and the store's read counters"""
        if prof is not None:
            prof.add(prof.MISS, layer, e, n_e, prof.shard(path), off, dur_ns, slot, aux=part)
        dt = dur_ns / 1e9
        with self._lock:
            st = self.stat
            st["read_s"] += dt
            st["read_n"] += 1
            if dt > st["read_max_s"]:
                st["read_max_s"] = dt

    def _read(self, parts: Any, e: Any, slot: Any, layer: int = -1) -> None:
        """the expert's parts read on this thread (the path without a scheduler's Route)"""
        sl = self.slots[slot]
        sl.bf16, sl.delta = False, (0,) * len(parts)
        region = self._region(slot)
        prof = getattr(self.sm, "expert_profile", None)
        for j, (path, off, n_e, _) in enumerate(parts):
            at = self.part_at[j] if self.part_at else sum(self.sizes[:j])
            tp = time.perf_counter_ns()
            if self.dequant:
                w = self.sm.gguf.get(self.sm._gguf_names[self.keys[layer][j]], expert=int(e))
                region[at : at + n_e].view(torch.bfloat16).copy_(w.reshape(-1))
            else:
                Native.read_direct(path, off + e * n_e, n_e, region[at : at + n_e], self.sm.cold_chunk)
            self._note_read(prof, layer, e, j, path, off + e * n_e, n_e, slot, time.perf_counter_ns() - tp)

    def _submit(self, sched: Any, parts: Any, e: Any, slot: Any, layer: int, priority: int) -> Any:
        """the expert's parts queued on the Route, one read each, with the profile's row and the counters
        taken as each lands; returns what `wait` and the layer's forward call `.result()` on. In a padded slot
        the read is the sector-aligned span around the part, straight into its region. A dequantized expert is
        no drive read: the store's own readers fill it."""
        self.slots[slot].bf16 = False
        if self.dequant:
            return Parts([self.pool.submit(self._read, parts, e, slot, layer)])
        region = self._region(slot)
        prof = getattr(self.sm, "expert_profile", None)
        futs = []
        deltas = []
        for j, (path, off, n_e, _) in enumerate(parts):
            o = off + e * n_e
            at = self.part_at[j] if self.part_at else sum(self.sizes[:j])
            if self.padded:
                delta = o % 4096
                o0 = o - delta
                n0 = -(-(delta + n_e) // 4096) * 4096
                if o0 + n0 > self._size_of(path):
                    # the file's tail: the span is cut at the end, and the reader bounces this one read
                    n0 = self._size_of(path) - o0
            else:
                delta, o0, n0 = 0, o, n_e
            deltas.append(delta)

            def landed(dur_ns: int, j: int = j, path: str = path, o: int = o, n_e: int = n_e) -> None:
                self._note_read(prof, layer, e, j, path, o, n_e, slot, dur_ns)

            futs.append(
                sched.disk_read(
                    path,
                    o0,
                    n0,
                    region[at : at + n0],
                    priority,
                    key=(layer, e),
                    on_done=landed,
                    chunk=self.sm.cold_chunk,
                    cached=self.cached_reads,
                )
            )
        self.slots[slot].delta = tuple(deltas)
        return Parts(futs)

    # -- the Timetable: the next layers' routers run on this layer's input, their picks read ahead into the ring --

    def _router(self, layer: int) -> Any:
        """the router of `layer` where the layer's module lives (the card, or the host), as (weight, bias or None);
        None without one. Qwen's is `mlp.gate`, gpt-oss's `mlp.router` (with a bias its logits add)"""
        mod = None
        for tier in ("resident", "host"):
            mod = (getattr(self.sm, tier, None) or {}).get(layer)
            if mod is not None:
                break
        hit = self._routers.get(layer)
        if hit is not None and hit[0] is mod:  # the layer's module as of the last call (a shed layer moves)
            return hit[1]
        mlp = getattr(mod, "mlp", None) if mod is not None else None
        gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
        # the router's own linear: the module itself (Qwen's), or the one btb wraps it around (gpt-oss's `_Router`)
        lin = gate if isinstance(getattr(gate, "weight", None), torch.Tensor) else None
        if lin is None and isinstance(gate, torch.nn.Module):
            lin = next((m for m in gate.modules() if isinstance(getattr(m, "weight", None), torch.Tensor)), None)
        w = getattr(lin, "weight", None)
        b = getattr(lin, "bias", None)
        wb = (w, b if isinstance(b, torch.Tensor) and b.numel() else None) if isinstance(w, torch.Tensor) else None
        self._routers[layer] = (mod, wb)
        return wb

    def _ring_slot(self, cap: int | None = None, skip: Any = ()) -> int | None:
        """a slot for a new prediction: the ring grows to `ring_n` slots (a sweep's `cap`) out of the store's own
        (`_seat_for`, never another prediction's), then reuses its oldest - unless that one is still being read, in
        which case the ring is full and the prediction is not made"""
        if len(self.ring) < (self.ring_n if cap is None else cap):
            s = self._seat_for(set(skip))
            if s is None:
                return None
            self._to_ring(s)
            return s
        s = next(iter(self.ring))
        sl = self.slots[s]
        if sl.state is SlotState.PREDICTED:
            if sl.landing():
                return None
            self.stat["ahead_recycled"] += 1
        self._to_ring(s)
        return s

    def _ring_take(self, skip: Any = ()) -> int | None:
        """a slot of the ring's given back to a call that has no other: the oldest one empty, or whose prediction
        has landed or can still be withdrawn whole, is the call's; None when every one is still being written"""
        sched = getattr(self.sm, "scheduler", None)
        for s in list(self.ring):
            if s in skip:
                continue
            sl = self.slots[s]
            if sl.state is SlotState.PREDICTED:
                if sl.landing() and sl.key is not None and sched is not None and hasattr(sched, "disk_drop"):
                    sched.disk_drop(sl.key, whole=True)  # withdrawn whole, or left to land whole (`_lapsed`)
                if sl.landing():
                    continue
                self.stat["ahead_dropped"] += 1
            self._taken(s)
            return s
        return None

    def lookahead(self, layer: int, h: torch.Tensor, sweep: int = 0) -> int:
        """Run the routers of the layers after `layer` on its MoE input `h` [T, hidden] and queue the reads of
        their top picks that are neither resident nor already predicted, the next layer's first: `sm.lookahead`
        gives the picks per depth ((10, 6): the next layer's top-10, the one after's top-6; measured on the
        180B, the next layer's top-10 holds 57% of its misses, top-20 78%). Over several rows (a verify
        pass) the picks are the union of each row's top-k, at most 2k of them by their best logit across the
        rows. Returns the reads queued.

        `sweep` (a layer-by-layer prefill, at a layer's first chunk; the experts that chunk asked): the next layer
        only, every expert some row's routing picks - a prefill layer asks for most of them - read into a ring as
        large as the store's room past this layer's own experts, while this layer's chunks compute. The decode's
        verdicts (a saturated drive, a slow one) do not hold it back: its reads queue behind every demand read
        and issue in the gaps between them; a drive that allows no prediction in flight still gets none."""
        # one row: few picks - measured on the 180B, a prediction past the third pick is right one time in four
        # and costs a read the layer waiting now then queues behind; a pass of several rows (a prefill, a tree):
        # more, the union of the rows' picks is right nine times in ten there
        rows = int(h.shape[0]) if h.dim() > 1 else 1
        ks = getattr(self.sm, "lookahead", ()) if rows == 1 else getattr(self.sm, "lookahead_rows", (10, 6))
        if sweep:
            ks = (int(getattr(self.sm.cfg, "num_experts_per_tok", 10) or 10),)
        sched = getattr(self.sm, "scheduler", None)
        if not ks or sched is None or not hasattr(sched, "disk_read") or self.per is None:
            return 0
        if self.drive is not None and int(self.drive.get("ahead", 1)) == 0:
            # one read outlasts the window the rule allows: a prediction is a read taken from the layer waiting now
            return 0
        if not sweep and self.saturated:
            # a drive with no idle time (its misses outlast the compute): the same, right or wrong
            return 0
        slow = getattr(sched, "disk_slow", None)
        if not sweep and slow is not None and slow():
            # the drive is delivering under half the probe's rate right now: the same arithmetic, live
            return 0
        cap = None
        if sweep:
            # the seats the store can hold now - its live ones and what the host's ledger would still let it grow
            # (commit, not only RAM, can stop it well short of its ceiling) - less the layer's own and a margin
            can = min(int(self.n_slots), self.live() + max(0, self._host_free() - self.margin) // int(self.per))
            cap = max(self.ring_n, can - int(sweep) - 64)
        # a sweep reads ahead after the layer's own call was handed its slots and before the call has read them: the
        # ring must not take one, or a prediction lands in the bytes the call is about to multiply
        skip = set(self.last_slots.values()) if sweep else ()
        n = 0
        prof = getattr(self.sm, "expert_profile", None)
        for d, k in enumerate(ks, start=1):
            target = layer + d
            if k <= 0 or target >= int(self.sm.L):
                break
            wb = self._router(target)
            if wb is None:
                continue
            base = self.recipes.get(target)
            if base is None:
                mod = (getattr(self.sm, "resident", None) or {}).get(target) or (
                    getattr(self.sm, "host", None) or {}
                ).get(target)
                ex = getattr(getattr(mod, "mlp", None), "experts", None)
                if ex is None or not hasattr(ex, "base"):
                    continue
                parts = self._recipe(target, ex.base)
            else:
                parts = base
            with torch.no_grad():
                w, bias = wb
                logits = torch.nn.functional.linear(
                    h.reshape(-1, h.shape[-1]).to(w.device, w.dtype), w, None if bias is None else bias.to(w.dtype)
                )
                kk = min(int(k), logits.shape[-1])
                if logits.shape[0] == 1:
                    picks = torch.topk(logits[0], kk).indices.tolist()
                else:
                    best = logits.max(dim=0).values
                    cand = torch.unique(torch.topk(logits, kk, dim=-1).indices)
                    if cap is not None:
                        # every expert some row picks, the strongest first, as many as the ring can take
                        picks = cand[torch.argsort(best[cand], descending=True)].tolist()[:cap]
                    else:
                        keep = torch.topk(best[cand], min(2 * kk, cand.shape[0])).indices
                        picks = cand[keep].tolist()
            for e in picks:
                key = (target, int(e))
                if key in self.res or key in self.ahead or (self.vram is not None and key in self.vram):
                    continue
                s = self._ring_slot(cap, skip)
                if s is None:
                    return n
                self._predicted(s, key, self._submit(sched, parts, int(e), s, target, sched.DISK_AHEAD + d - 1), d)
                self.stat["ahead"] += 1
                n += 1
                if prof is not None:
                    prof.add(prof.AHEAD, target, int(e), self.per, slot=s, aux=d)
        return n

    def sweep_end(self) -> None:
        """a layer-by-layer prefill done: the ring back to `ring_n` slots, the rest the store's again - a
        prediction still reading is withdrawn or let land (its slot is being written), then forgotten"""
        self.sweep_layer = -1
        sched = getattr(self.sm, "scheduler", None)
        while len(self.ring) > self.ring_n:
            s = next(reversed(self.ring))
            sl = self.slots[s]
            if sl.state is SlotState.PREDICTED and sl.key is not None:
                if sched is not None and hasattr(sched, "disk_drop"):
                    sched.disk_drop(sl.key)
                self.stat["ahead_dropped"] += 1
            self._to_free(s)

    def _lapsed(self, layer: int, asked: Any, sched: Any) -> None:
        """the predictions for `layer` it did not ask for: reads not yet issued are withdrawn and their slots put
        first in line for reuse; bytes that already landed stay in the ring until it wraps (the same expert at
        the same layer a token later is a third of the traffic)"""
        for key in [k for k in self.ahead if k[0] == layer and k[1] not in asked]:
            s = self.ahead[key]
            sl = self.slots[s]
            if not sl.landing():
                continue
            if sched is not None and hasattr(sched, "disk_drop"):
                # withdrawn whole or not at all: an expert with a part in flight is left to land entire - it stays
                # predicted, promoted whole if a later call asks for it, and `_ring_slot` passes its slot over until
                # then; a part withdrawn beside one that lands would leave the slot a mix of two experts that reads
                # as landed
                sched.disk_drop(key, whole=True)
            if sl.landing():
                continue
            self.stat["ahead_dropped"] += 1
            self._to_ring(s, first=True)

    def call(self, layer: int, base: str, ids: Sequence[int], keep: bool = True, rows: int = 1) -> StoreCall:
        """A MoE layer call's experts `ids` of `layer` (ascending, as a call lists them), to be seated in waves
        (`StoreCall.wave`) or whole (`StoreCall.whole`). The call at the model's first MoE layer starts a pass: the
        profile's step, the card seats' promotions, and the one-row pass's verdict on the drive are taken here,
        once a call, never once a wave. `keep` off, a prefill's experts give their seats up first; `rows` the
        call's rows."""
        t0 = time.perf_counter()
        self._recipe(layer, base)
        assert self.per is not None  # _recipe sizes the store on the first call
        if layer == self._first_layer():
            prof = getattr(self.sm, "expert_profile", None)
            if prof is not None:
                prof.step += 1
            if self.vram is not None:
                self.vram.new_pass()
            # the one-row passes, as each closes, say whether this drive has room for predictions, live
            ps = self._pass
            if ps is not None and ps["rows"] == 1:
                line = self._pass_closed(t0 - ps["t0"], self.stat["wait_s"] - ps["w0"], self.stat["miss"] - ps["m0"])
                if line:
                    self.sm.log(line)
            self._pass = {"t0": t0, "w0": self.stat["wait_s"], "m0": self.stat["miss"], "rows": int(rows)}
        self.stat["calls"] += 1
        return StoreCall(self, layer, base, [int(e) for e in ids], keep, rows)

    def get(self, layer: int, base: str, ids: Sequence[int], keep: bool = True, rows: int = 1) -> Any:
        """a call's experts, every one seated at once (`StoreCall.whole`): (ready, pending)"""
        return self.call(layer, base, ids, keep, rows).whole()

    def _first_layer(self) -> int:
        """the model's first layer with experts: where a pass's first call is"""
        if self._first_moe is None:
            self._first_moe = 0
            for i in range(int(self.sm.L)):
                if i in self.recipes:
                    self._first_moe = i
                    break
                try:
                    self._recipe(i, f"{getattr(self.sm, 'prefix', '')}layers.{i}.mlp.experts.")
                except KeyError:  # a dense layer: no experts
                    continue
                self._first_moe = i
                break
        return self._first_moe

    def _wave(self, layer: int, base: str, ids: list[int], keep: bool, rows: int, whole: bool) -> Any:
        """The longest prefix of `ids` the store can seat now, in one ascending pass: each expert a card seat, a
        resident, a prediction promoted, or a miss seated by `_seat_for` and read; the pass stops before the first
        expert with no seat (with `whole`, it raises), and nothing past it is touched - no seat taken, no ride
        counted, no read queued. Returns (ready, pending, rest): the views of the experts in hand, the (expert,
        reads, slot) of those still landing, and the experts left for the next wave."""
        t0 = time.perf_counter()
        parts = self._recipe(layer, base)
        assert self.per is not None  # sized by the recipe before any call
        prof = getattr(self.sm, "expert_profile", None)
        sched = getattr(self.sm, "scheduler", None)
        # the blocks the machine asks back given back first, so the seats counted are the seats there are - and never
        # during the pass, where a block given back could take a hit already handed out. The call's misses count first
        # toward the seats release keeps, so it never gives back the ones this call is about to read into
        misses = sum(
            1
            for e in ids
            if self.res.peek((layer, e)) is None
            and (layer, e) not in self.ahead
            and not (self.vram is not None and (layer, e) in self.vram)
        )
        self.max_call = max(self.max_call, misses)
        self.release()
        if self.ahead:
            # every expert the call still wants, a later wave's too: a prediction one of them will want is kept
            self._lapsed(layer, set(ids), sched)
        out: dict[int, int] = {}
        todo: list[int] = []
        waiting: list[tuple[int, Parts, int]] = []
        on_card: dict[int, Any] = {}
        taken: set[int] = set()
        # the seats of the call's residents still to come, which a miss before them does not take while anything
        # else gives way: found once, and each let go of as the pass reaches its expert
        protect = {p for e in ids if (p := self.res.peek((layer, e))) is not None}
        stop = len(ids)
        for n, e in enumerate(ids):
            key = (layer, e)
            if self.vram is not None and key in self.vram:
                # a first-class seat: the card multiplies it, no bytes move
                self.sm._tag(PassTag.EXPERT_VRAM_SEAT)
                self.rides[key] = self.rides.get(key, 0) + 1
                on_card[e] = self.vram.views(key)
                self.stat["hit"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=-2, aux=2)
                continue
            s = self.res.get(key)
            if s is not None:
                self.rides[key] = self.rides.get(key, 0) + 1
                protect.discard(s)
                out[e] = s
                taken.add(s)
                self.stat["hit"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=s)
                self._seat(layer, e, s)
                continue
            s = self.ahead.get(key)
            if s is not None:
                # the lookahead read it (or is reading it): promoted out of the ring into the store
                self.rides[key] = self.rides.get(key, 0) + 1
                self._resident(s, key)
                out[e] = s
                taken.add(s)
                self.stat["hit"] += 1
                self.stat["ahead_used"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=s, aux=1)
                sl = self.slots[s]
                if sl.landing() and sl.parts is not None:
                    waiting.append((e, sl.parts, s))
                continue
            s = self._seat_for(taken, protect, ring=True, last=n == 0)
            if s is None:
                if whole or n == 0:
                    # refused whole: the misses this pass seated have no reads queued, so they leave the line and
                    # their slots go back free - left seated, a later call would take their unread bytes as a hit
                    for m in todo:
                        self.res.pop((layer, m))
                        self._to_free(out[m])
                    raise RuntimeError(
                        f"[experts] one call needs {len(ids)} experts and the store seats {n} of them: "
                        f"{len(taken)} of {self.live()} live seats the call's, {len(self.ring)} the lookahead's "
                        f"({self._host_line()})"
                    )
                stop = n  # no seat for this one: the wave ends before it
                break
            protect.discard(s)  # a later resident's seat the first expert had to take
            self.rides[key] = self.rides.get(key, 0) + 1
            self._resident(s, key)
            out[e] = s
            taken.add(s)
            todo.append(e)
            self.stat["miss"] += 1
        if todo:
            self.stat["miss_calls"] += 1
            self.stat["adjacent"] += sum(1 for a, b in pairwise(todo) if b == a + 1)
        self.max_call = max(self.max_call, len(todo))
        if not keep:
            for e in todo:
                self.res.demote((layer, e))
        self.last_slots = dict(out)
        if rows > 1 and layer == self._first_layer() and todo and not self._warned_prefill and self.drive is not None:
            # a prompt on a drive that seeks is minutes to its first token: said once, before the wait, so the
            # user knows what was bought rather than that the engine hung
            est = len(todo) * int(self.sm.L) * self.expert_s(self.drive, self.per)
            if est > 60.0:
                self._warned_prefill = True
                self.sm.log(
                    f"[experts] this prompt reads about {len(todo)} experts a layer: on this drive that is about "
                    f"{est / 60:.0f} minutes to its first token"
                )
        if sched is not None and hasattr(sched, "disk_read"):
            # a decode row's misses are a layer waiting now; a prefill's experts (`keep` off) are the sweep
            pri = sched.DISK_DEMAND if keep else sched.DISK_SWEEP
            futs = {e: self._submit(sched, parts, e, out[e], layer, pri) for e in todo}
        else:
            futs = {e: Parts([self.pool.submit(self._read, parts, e, out[e], layer)]) for e in todo}
        for e, f in futs.items():
            self.slots[out[e]].parts = f
        self.stat["bytes"] += len(todo) * self.per
        self.stat["s"] += time.perf_counter() - t0
        still = {e for e, _, _ in waiting}
        if self.dt != torch.bfloat16:
            for e, s in out.items():
                if e not in futs and e not in still:
                    self._to_bf16(s)
        ready = {e: self._views(s) for e, s in out.items() if e not in futs and e not in still}
        ready.update(on_card)
        pending = [(e, futs[e], out[e]) for e in todo] + waiting
        return ready, pending, ids[stop:]

    def wait(self, pending: Any) -> Any:
        """every pending read landed, as `landed` would yield them, in one call: e -> views"""
        t0 = time.perf_counter()
        done = {}
        for e, f, s in pending:
            f.result()
            self._to_bf16(s)
            done[e] = self._views(s)
        self.stat["wait_s"] += time.perf_counter() - t0
        return done

    def landed(self, pending: Any) -> Any:
        """The pending experts as their bytes land, in batches: each yield is the list of (expert, parts, slot)
        whose reads have completed since the last, so a layer computes what has arrived while the rest is still
        on its way instead of waiting in submission order. The time blocked between batches is `wait_s`."""
        left = list(pending)
        while left:
            ready = [p for p in left if p[1].done()]
            if not ready:
                t0 = time.perf_counter()
                wait_for([f for p in left for f in p[1].futs if not f.done()], return_when=FIRST_COMPLETED)
                self.stat["wait_s"] += time.perf_counter() - t0
                continue
            for p in ready:
                left.remove(p)
                p[1].result()
                self._to_bf16(p[2])
            yield ready
