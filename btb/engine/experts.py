# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The expert store: a MoE model's experts kept in RAM in their stored form and read from the drive on a miss."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import wait as wait_for
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .. import mlx as mlxdev
from ..kinds import PassTag
from ..mxfp4 import BLOCK, MxGateUp, MxWeight
from ..options import Device
from ..sysinfo import host_free_bytes
from .host import bf16_in_place
from .native import Native

if TYPE_CHECKING:
    from concurrent.futures import ThreadPoolExecutor
    from types import ModuleType


class ExpertProfile:
    """Everything the store did, one row per event in an append-only int64 array: no allocation on the path
    (the array doubles when full), one lock shared with the reader threads, saved by `save` as an .npz with
    `events` [n, 11], `kinds` and `shards` (a path per shard id). Columns: t_ns from the first event, step
    (layer 0's call starts a pass), kind, layer, expert, bytes, shard, offset, dur_ns, slot, aux. By kind:

        hit      the expert served from its slot
        miss     one read of a missing expert's part: bytes, shard, offset, dur_ns of that read, slot, aux the
                 part (0 gate_up, 1 down). Why it was read is in the history: a (layer, expert)'s first miss
                 is its fill, a later one a reload after an eviction
        reread   a hit whose slot a release took under the call, read again
        evict    the expert a slot was taken from: aux 0 for the call's need, 1 for the machine's memory
        grow     a block added: expert the slots, bytes, aux the host's free bytes after
        release  a block given back for the machine: expert the slots, bytes, aux the host's free bytes after
        call     one layer's call done: expert the rows, bytes the hits, shard the misses, dur_ns the wait
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

    def save(self) -> str:
        self._watching = False
        with self._lock:
            ev = self.a[: self.n].copy()
            snaps = list(self.snapshots)
        shards = [p for p, _ in sorted(self.shards.items(), key=lambda kv: kv[1])]
        np.savez(self.path, events=ev, kinds=np.array(self.KINDS), shards=np.array(shards), snapshots=np.array(snaps))
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

    def drop(self, key: Any) -> None:
        j = self.seat_of.pop(key, None)
        if j is not None:
            self.free.append(j)


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
    as_bf16: set[Any]  # the slots whose fp16/fp32 expert has been rewritten as bf16 since it was read
    free: Any
    last_slots: Any
    lru: Any
    max_call: int
    mx: bool
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
    slot_of: dict[int, tuple[int, int]]
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
        self.as_bf16 = set()
        self.mx = bool(sm.fam.mxfp4)
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
        self.res = (BusPass if bus_pass else Riders)(lambda: self.live() - len(self.ring))
        self.res_tag = PassTag.EXPERT_BUS_PASS if bus_pass else PassTag.EXPERT_LINE
        self.lru = self.res.t1
        # the depot's pages held in RAM (`store_pin`, BTB_STORE_PIN over it: 0 pageable, 1 pinned, "auto" pinned
        # beside a card): a drive writing straight into a page the machine has trimmed pays the fault on the read
        sp: str | int | None = os.environ.get("BTB_STORE_PIN")
        sp = sp if sp not in (None, "") else sm.store_pin
        dev = getattr(sm, "dev", None)
        on_card = dev is not None and getattr(dev, "type", "") == Device.CUDA
        self.pin = on_card if str(sp).strip().lower() == "auto" else bool(int(sp or 0)) and on_card
        self.free = []
        self.parked = []
        self.blocks = {}
        self.shared = {}
        self.last_slots = {}
        self.slot_of = {}
        self.next_slot = 0
        self.max_call = 0
        self.pool = ThreadPoolExecutor(max_workers=int(readers))
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
        # promoted into the store) or the ring wraps (the slot is reused); never the store's own slots
        self.ring_n = 64
        self.ring: list[int] = []
        self.ahead: dict[tuple[int, int], dict[str, Any]] = {}
        self._routers: dict[int, Any] = {}
        # the slot's layout: `stride` bytes a slot, each part's region starting at `part_at[p]`. Padded (the
        # torch path with the positional reader), every region starts on a sector and holds two sectors of
        # slack, so a read of the aligned span around the expert's bytes lands straight in the slot with no
        # bounce; the expert then sits `slot_delta[slot][p]` bytes into its region, its file offset's own
        # misalignment. Unpadded (MLX's shared blocks, the reader without handles): the parts back to back
        self.stride: Any = None
        self.part_at: tuple[int, ...] = ()
        self.padded = False
        self.slot_delta: dict[Any, tuple[int, ...]] = {}
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

    def _grow(self, need: int) -> Any:
        assert self.per is not None  # the store is sized before this runs
        room = self.n_slots - self.live()
        if room <= 0:
            return 0
        if self.sm.mlx is not None:
            # the engine's ledger: what the load started with, less the reserve, less everything MLX holds (exact
            # whether or not the pages are touched)
            headroom = self.sm.mem_start - self.reserve - self.sm.mlx.held_bytes()
        else:
            headroom = host_free_bytes() - self.reserve
        usable = headroom - self.margin
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
                        f"[experts] store: the machine would not pin a {k * stride / 2**30:.2f} GB block ({e}); pageable from here"
                    )
            if raw is None:
                raw = torch.empty(k * stride + 4096, dtype=torch.uint8)
            skew = (-raw.data_ptr()) % 4096
            buf = raw[skew : skew + k * stride]
        self.blocks[b] = (buf, ids)
        for j, s in enumerate(ids):
            self.slot_of[s] = (b, j)
        self.free.extend(ids)
        free_now = host_free_bytes()
        self.sm.log(
            f"[experts] store +{k} slots ({k * self.per / 2**30:.2f} GB), {self.live()} of {self.n_slots} live, "
            f"{free_now / 2**30:.1f} GB free"
        )
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None:
            prof.add(prof.GROW, expert=k, nbytes=k * self.per, aux=int(free_now))
        return k

    def _release_block(self, b: Any) -> None:
        assert self.per is not None  # the store is sized before this runs
        buf, ids = self.blocks.pop(b)
        self.shared.pop(b, None)
        gone = set(ids)
        prof = getattr(self.sm, "expert_profile", None)
        for key, s in [(k, s) for k, s in self.res.items() if s in gone]:
            if prof is not None:
                prof.add(prof.EVICT, key[0], key[1], self.per, slot=s, aux=1)
            self.res.pop(key)
        sched = getattr(self.sm, "scheduler", None)
        for key in [k for k, a in self.ahead.items() if a["slot"] in gone]:
            if sched is not None and hasattr(sched, "disk_drop"):
                sched.disk_drop(key)  # a read not yet issued into a block given back is drive time for nothing
            del self.ahead[key]
        self.ring = [s for s in self.ring if s not in gone]
        self.free = [s for s in self.free if s not in gone]
        for s in ids:
            del self.slot_of[s]
        self.parked.extend(ids)
        self.stat["released"] += len(ids)
        del buf
        if prof is not None:
            prof.add(prof.RELEASE, expert=len(ids), nbytes=len(ids) * self.per, aux=int(host_free_bytes()))

    def release(self) -> Any:
        freed = 0
        while self.blocks and host_free_bytes() < self.reserve:
            victim = None
            s = self.res.oldest_slot()
            if s is not None:
                victim = self.slot_of[s][0]
            if victim is None:
                victim = next(iter(self.blocks))
            if self.live() - len(self.blocks[victim][1]) < self.max_call:
                # below the largest call served the next call cannot be served at all, so the store holds here and
                # leaves the reserve to the machine's paging
                if not self.stat.get("floor"):
                    self.stat["floor"] = 1
                    self.sm.log(
                        f"[experts] store holds {self.live()} slots for calls of {self.max_call}: "
                        f"{host_free_bytes() / 2**30:.1f} GB free is under the "
                        f"{self.reserve / 2**30:.1f} GB reserve"
                    )
                break
            freed += len(self.blocks[victim][1])
            self._release_block(victim)
        if freed:
            self.sm.log(
                f"[experts] store -{freed} slots for the machine, {self.live()} live, "
                f"{host_free_bytes() / 2**30:.1f} GB free"
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
                if not (self.ggml or self.mx):
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
                self.shapes = shapes if self.mx else (parts[0][2], *shapes)
                self.sizes = tuple(p[2] for p in parts)
                # reads straight into the slot (padded regions) unless BTB_STORE_PADDED=0 asks for the bounce
                self.padded = (
                    self.sm.mlx is None
                    and not self.dequant
                    and Native.read_at is not None
                    and os.environ.get("BTB_STORE_PADDED", "1") != "0"
                )
                self.stride, self.part_at = self._layout(self.sizes, self.padded)
                total = int(self.sm.L) * int(self.sm.n_experts)
                self.n_slots = min(total, max(self.scratch_n + 1, int(self.budget // self.stride)))
                # the ring is carved out of the store: an eighth of it at most, none of a store too small to spare
                self.ring_n = min(self.ring_n, self.n_slots // 8)
                self._open_vram()
                self.sm.log(
                    f"[experts] store: up to {self.n_slots} slots x {per / 2**20:.1f} MB = "
                    f"{self.n_slots * per / 2**30:.1f} GB in RAM, allocated as needed in blocks of at most "
                    f"{self.block_max / 2**30:.1f} GB, {self.margin / 2**30:.1f} GB above a "
                    f"{self.reserve / 2**30:.1f} GB reserve, {'pinned' if self.pin else 'pageable'}, "
                    f"{self.pool._max_workers} readers"
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
        if self.mx or dev is None or dev.type != Device.CUDA or not want or self.stride is None:
            return
        if want == "auto":
            sched = getattr(self.sm, "scheduler", None)
            free = sched.free_vram() if sched is not None else None
            room = (free or 0) - (1 << 30)
        else:
            room = int(float(want) * 2**30)
        per = self._held(self.per)
        n = max(0, int(room // per))
        if n <= 0:
            return
        gu_n, gu_shape, dn_shape = self.shapes
        self.vram = VramSeats(n, per, (self._held(gu_n), gu_shape, dn_shape), dev)
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
        d = self.slot_delta.get(slot) or (0,) * len(self.sizes)
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
        b, j = self.slot_of[slot]
        stride = int(self.stride or self.per)
        return self.blocks[b][0][j * stride : (j + 1) * stride]

    def _part(self, slot: Any, region: Any, p: int) -> Any:
        """part `p` of the expert in `slot`, as the bytes sit in the region"""
        at = (self.part_at[p] if self.part_at else sum(self.sizes[:p])) + (self.slot_delta.get(slot) or (0,) * 8)[p]
        return region[at : at + self.sizes[p]]

    def mx_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """the MXFP4 experts' logical shapes, (gate_up [2I, H], down [H, I]), in either layout"""
        if self.ggml:
            (rows, k), _, (drows, dk) = self.shapes
            return (2 * int(rows), int(k)), (int(drows), int(dk))
        gu, dn = self.shapes[0], self.shapes[2]
        return (int(gu[0]), int(gu[1]) * BLOCK), (int(dn[0]), int(dn[1]) * BLOCK)

    def _views(self, slot: Any) -> Any:
        region = self._region(slot)
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
        if self.dt == torch.bfloat16 or slot in self.as_bf16:
            return
        region = self._region(slot)
        for p in (0, 1):
            bf16_in_place(self._part(slot, region, p), self.dt)
        self.as_bf16.add(slot)

    def _views_mx(self, slot: Any) -> Any:
        assert self.per is not None  # the store is sized before this runs
        b, j = self.slot_of[slot]
        sh = self.shared[b]
        off = j * int(self.stride or self.per)
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
        self.as_bf16.discard(slot)
        region = self._region(slot)
        prof = getattr(self.sm, "expert_profile", None)
        self.slot_delta[slot] = (0,) * len(parts)
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
        self.as_bf16.discard(slot)
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
                )
            )
        self.slot_delta[slot] = tuple(deltas)
        return Parts(futs)

    # -- the Timetable: the next layers' routers run on this layer's input, their picks read ahead into the ring --

    def _router(self, layer: int) -> Any:
        """the router weight of `layer` where the layer's module lives (the card, or the host), None without one"""
        mod = None
        for tier in ("resident", "host"):
            mod = (getattr(self.sm, tier, None) or {}).get(layer)
            if mod is not None:
                break
        hit = self._routers.get(layer)
        if hit is not None and hit[0] is mod:  # the layer's module as of the last call (a shed layer moves)
            return hit[1]
        gate = getattr(getattr(mod, "mlp", None), "gate", None) if mod is not None else None
        w = getattr(gate, "weight", None)
        self._routers[layer] = (mod, w)
        return w

    def _ring_slot(self, sched: Any) -> Any:
        """a slot for a new prediction: the ring grows to `ring_n` slots out of the store's free ones (a grown
        block, or the oldest resident's slot), then reuses its oldest entry - unless that one is still in flight,
        in which case the ring is full and the prediction is not made"""
        if len(self.ring) < self.ring_n:
            if not self.free and self.live() < self.n_slots:
                self._grow(1)
            if self.free:
                s = self.free.pop()
            else:
                v = self.res.victim()
                if v is None:
                    return None
                key, s = v
                prof = getattr(self.sm, "expert_profile", None)
                if prof is not None:
                    prof.add(prof.EVICT, key[0], key[1], self.per, slot=s, aux=0)
            self.ring.append(s)
            return s
        s = self.ring[0]
        old = next((k for k, a in self.ahead.items() if a["slot"] == s), None)
        if old is not None:
            a = self.ahead[old]
            if not a["parts"].done():
                return None
            del self.ahead[old]
            self.stat["ahead_recycled"] += 1
        self.ring.pop(0)
        self.ring.append(s)
        return s

    def lookahead(self, layer: int, h: torch.Tensor) -> int:
        """Run the routers of the layers after `layer` on its MoE input `h` [T, hidden] and queue the reads of
        their top picks that are neither resident nor already predicted, the next layer's first: `sm.lookahead`
        gives the picks per depth ((10, 6): the next layer's top-10, the one after's top-6; measured on the
        180B, the next layer's top-10 holds 57% of its misses, top-20 78%). Over several rows (a verify
        pass) the picks are the union of each row's top-k, at most 2k of them by their best logit across the
        rows. Returns the reads queued."""
        # one row: few picks - measured on the 180B, a prediction past the third pick is right one time in four
        # and costs a read the layer waiting now then queues behind; a pass of several rows (a prefill, a tree):
        # more, the union of the rows' picks is right nine times in ten there
        rows = int(h.shape[0]) if h.dim() > 1 else 1
        ks = getattr(self.sm, "lookahead", ()) if rows == 1 else getattr(self.sm, "lookahead_rows", (10, 6))
        sched = getattr(self.sm, "scheduler", None)
        if not ks or sched is None or not hasattr(sched, "disk_read") or self.per is None:
            return 0
        if self.saturated or (self.drive is not None and int(self.drive.get("ahead", 1)) == 0):
            # a drive with no idle time (its misses outlast the compute, or one read outlasts the window the
            # rule allows): a prediction is a read taken from the layer waiting now, right or wrong
            return 0
        slow = getattr(sched, "disk_slow", None)
        if slow is not None and slow():
            # the drive is delivering under half the probe's rate right now: the same arithmetic, live
            return 0
        n = 0
        prof = getattr(self.sm, "expert_profile", None)
        for d, k in enumerate(ks, start=1):
            target = layer + d
            if k <= 0 or target >= int(self.sm.L):
                break
            w = self._router(target)
            if w is None:
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
                logits = torch.matmul(h.reshape(-1, h.shape[-1]).to(w.device, w.dtype), w.T)
                kk = min(int(k), logits.shape[-1])
                if logits.shape[0] == 1:
                    picks = torch.topk(logits[0], kk).indices.tolist()
                else:
                    best = logits.max(dim=0).values
                    cand = torch.unique(torch.topk(logits, kk, dim=-1).indices)
                    keep = torch.topk(best[cand], min(2 * kk, cand.shape[0])).indices
                    picks = cand[keep].tolist()
            for e in picks:
                key = (target, int(e))
                if key in self.res or key in self.ahead or (self.vram is not None and key in self.vram):
                    continue
                s = self._ring_slot(sched)
                if s is None:
                    return n
                parts_f = self._submit(sched, parts, int(e), s, target, sched.DISK_AHEAD + d - 1)
                self.ahead[key] = {"slot": s, "parts": parts_f, "d": d}
                self.stat["ahead"] += 1
                n += 1
                if prof is not None:
                    prof.add(prof.AHEAD, target, int(e), self.per, slot=s, aux=d)
        return n

    def _lapsed(self, layer: int, asked: Any, sched: Any) -> None:
        """the predictions for `layer` it did not ask for: reads not yet issued are withdrawn and their slots put
        first in line for reuse; bytes that already landed stay in the ring until it wraps (the same expert at
        the same layer a token later is a third of the traffic)"""
        for key in [k for k in self.ahead if k[0] == layer and k[1] not in asked]:
            a = self.ahead[key]
            if a["parts"].done():
                continue
            if sched is not None and hasattr(sched, "disk_drop"):
                sched.disk_drop(key)
            if not a["parts"].done():
                # a part is in flight (only queued reads withdraw): the record stays, so `_ring_slot` sees the
                # slot is still being written and passes it over until the read lands
                continue
            del self.ahead[key]
            self.stat["ahead_dropped"] += 1
            s = a["slot"]
            if s in self.ring:
                self.ring.remove(s)
                self.ring.insert(0, s)

    def get(self, layer: int, base: str, ids: Sequence[int], keep: bool = True, rows: int = 1) -> Any:
        t0 = time.perf_counter()
        parts = self._recipe(layer, base)
        assert self.per is not None  # _recipe sizes the store on the first call
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None and layer == 0:
            prof.step += 1
        sched = getattr(self.sm, "scheduler", None)
        if layer == 0 and self.vram is not None:
            self.vram.new_pass()
        if layer == 0:
            # the one-row passes, as each closes, say whether this drive has room for predictions, live
            ps = self._pass
            if ps is not None and ps["rows"] == 1:
                line = self._pass_closed(t0 - ps["t0"], self.stat["wait_s"] - ps["w0"], self.stat["miss"] - ps["m0"])
                if line:
                    self.sm.log(line)
            self._pass = {"t0": t0, "w0": self.stat["wait_s"], "m0": self.stat["miss"], "rows": int(rows)}
        out = {}
        todo = []
        waiting = []
        on_card = {}
        for e in ids:
            key = (layer, e)
            self.rides[key] = self.rides.get(key, 0) + 1
            if self.vram is not None and key in self.vram:
                # a first-class seat: the card multiplies it, no bytes move
                self.sm._tag(PassTag.EXPERT_VRAM_SEAT)
                on_card[e] = self.vram.views(key)
                self.stat["hit"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=-2, aux=2)
                continue
            s = self.res.get(key)
            if s is not None:
                out[e] = s
                self.stat["hit"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=s)
                self._seat(layer, e, s)
                continue
            a = self.ahead.pop(key, None)
            if a is not None:
                # the lookahead read it (or is reading it): promoted out of the ring into the store
                s = a["slot"]
                if s in self.ring:
                    self.ring.remove(s)
                self.res.admit(key, s)
                out[e] = s
                self.stat["hit"] += 1
                self.stat["ahead_used"] += 1
                if prof is not None:
                    prof.add(prof.HIT, layer, e, self.per, slot=s, aux=1)
                if not a["parts"].done():
                    waiting.append((e, a["parts"], s))
                continue
            self.stat["miss"] += 1
            todo.append(e)
        if self.ahead:
            self._lapsed(layer, {int(x) for x in ids}, sched)
        if len(todo) > self.n_slots:
            raise RuntimeError(f"[experts] one call needs {len(todo)} experts, the store holds {self.n_slots} slots")
        self.stat["calls"] += 1
        if todo:
            self.stat["miss_calls"] += 1
            srt = sorted(todo)
            self.stat["adjacent"] += sum(1 for a, b in pairwise(srt) if b == a + 1)
        self.max_call = max(self.max_call, len(todo))
        self.release()
        # release() frees whole blocks by LRU age; a hit collected above can share a block with the
        # evicted slot and be freed as collateral. Re-read any hit whose slot release() took.
        for e in [e for e, s in out.items() if s not in self.slot_of]:
            self.res.pop((layer, e))
            self.stat["hit"] -= 1
            self.stat["miss"] += 1
            del out[e]
            todo.append(e)
            if prof is not None:
                prof.add(prof.REREAD, layer, e, self.per)
        waiting = [w for w in waiting if w[0] in out]  # a promoted read whose slot went is re-read like a hit
        if len(self.free) < len(todo):
            self._grow(len(todo) - len(self.free))
        while self.live() < len(todo):
            if not self._grow(len(todo) - self.live()):
                raise RuntimeError(
                    f"[experts] one call needs {len(todo)} experts and the machine has no room "
                    f"above the {self.reserve / 2**30:.1f} GB reserve"
                )
        taken = set(out.values())
        for e in todo:
            if self.free:
                s = self.free.pop()
            else:
                v = self.res.victim(taken)
                if v is None:
                    raise RuntimeError(f"[experts] one call needs {len(todo)} experts and every seat is the call's own")
                key, s = v
                if prof is not None:
                    prof.add(prof.EVICT, key[0], key[1], self.per, slot=s, aux=0)
            taken.add(s)
            self.res.admit((layer, e), s)
            out[e] = s
        if not keep:
            for e in todo:
                self.res.demote((layer, e))
        self.last_slots = dict(out)
        if rows > 1 and layer == 0 and todo and not self._warned_prefill and self.drive is not None:
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
        return ready, pending

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
