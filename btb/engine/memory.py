# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The VRAM policy: what the card and the host hold, and when a layer or the head is shed to the host, regrown
onto the card, reallocated after a bleed, or the card's cache trimmed. The raw machine sensors it reads (host
RAM, the GPU pressure counter) live in `btb.sysinfo`; the free-memory ledger it consults is `Device.free`."""

from __future__ import annotations

import itertools
import math
import time
import weakref
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from ..api import api
from ..kinds import Log, PassTag
from ..options import Device
from ..sysinfo import (
    _darwin_available_bytes,
    hard_page_faults,
    host_commit_bytes,
    host_free_bytes,
    host_total_bytes,
    memory_pressure,
    vram_pressure,
    vram_pressure_line,
)
from .cache import GrowLayer
from .device import DeviceSpec, torch_device
from .scheduler import EPOCH, MemoryGrantError
from .state import _State
from .tiers import ColdRing

if TYPE_CHECKING:
    from .cache import KvCache
    from .device import Device as DeviceLedger


@dataclass
class VramPolicyState:
    """The VRAM policy's own bookkeeping between steps: how much slower than its best a card step must be to
    count as contended, how many free checks precede a regrow, how often pressure is read, the shared-memory
    floor learned on the first read, and the counters and the held reason it carries from one read to the next."""

    contention: float = 4.0
    regrow_after: int = 3
    period: float = 1.0
    contended: int = 0
    free_checks: int = 0
    shared_floor: int | None = None
    realloc_tried: bool = False
    last_t: float = 0.0
    total: int | None = None
    held: str | None = None


@dataclass
class RamPolicyState:
    """Counters for `ram_policy`: consecutive short and clean readings, the last fault count, the layers shed."""

    period: float = 1.0
    regrow_after: int = 5
    last_t: float = 0.0
    faults: int | None = None
    paging: int = 0
    clean: int = 0
    shed: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class DeviceMemory:
    """One device's memory as btb sees it, in bytes: `free` above btb's margin and what is spoken for, `reserved`
    what rooms, lent tensors btb cannot see and a batch's coming KV hold there, and `sheddable` what btb itself
    gives up to make room. `lendable` is the most `empty` or `room` can get there."""

    device: str
    free: int
    reserved: int
    sheddable: int

    @property
    def lendable(self) -> int:
        return self.free + self.sheddable


@api("room")
class Room:
    """Room made for memory btb does not allocate itself - a second model, a library's workspace - and kept from
    btb until `release()`, the end of a `with` block, or the room being dropped. Rooms add up, each its own.
    `empty`/`zeros`/`full` hand out tensors inside it: their bytes come off what the room holds while they live, so
    the room and the tensor are not counted twice, and a tensor outliving its room is still counted until it goes
    (in btb's free reading, or on MLX, whose reading cannot see torch's, under a reservation of its own)."""

    def __init__(
        self, ledger: DeviceLedger, tag: str, nbytes: int, device: torch.device, seen: bool, engine: _State
    ) -> None:
        self.nbytes, self.device = int(nbytes), device
        self.used = 0  # what the room's own tensors take, while they live
        # a room outliving its model keeps no model alive: the engine and its ledger are held weakly
        self._ledger, self._loan, self._seen = weakref.ref(ledger), tag, seen
        self._lock = ledger.lock  # `used` and the hold, against the finalizers of the room's tensors
        self._engine = weakref.ref(engine)
        self._give_back = weakref.finalize(self, _give_back, weakref.ref(ledger), tag)

    def _called(self, tag: PassTag) -> None:
        eng = self._engine()
        if eng is not None:
            eng._called(tag)

    @property
    def held(self) -> bool:
        return self._give_back.alive

    def release(self) -> None:
        self._give_back()

    def empty(self, shape: int | Sequence[int], dtype: torch.dtype | None = None) -> torch.Tensor:
        """a tensor of the room's, as `torch.empty` makes it on the room's device; a MemoryGrantError past what is
        left of the room"""
        return self._alloc(shape, dtype, None)

    def zeros(self, shape: int | Sequence[int], dtype: torch.dtype | None = None) -> torch.Tensor:
        """`empty`, filled with zeros"""
        return self._alloc(shape, dtype, 0)

    def full(self, shape: int | Sequence[int], value: float, dtype: torch.dtype | None = None) -> torch.Tensor:
        """`empty`, filled with `value`"""
        return self._alloc(shape, dtype, value)

    def _alloc(self, shape: int | Sequence[int], dtype: torch.dtype | None, fill: float | None) -> torch.Tensor:
        dt = dtype if dtype is not None else torch.get_default_dtype()
        size = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        nbytes = math.prod(size) * torch.empty(0, dtype=dt).element_size()
        with self._lock:  # two threads filling the room see each other's tensors
            if not self.held:
                raise ValueError("this room was released: make another")
            if self.used + nbytes > self.nbytes:
                raise MemoryGrantError(
                    f"a {dt} tensor of shape {list(size)} ({nbytes / 2**20:.1f} MiB) in a room of "
                    f"{self.nbytes / 2**20:.1f} MiB with {(self.nbytes - self.used) / 2**20:.1f} MiB left"
                )
            t = torch.empty(size, dtype=dt, device=self.device)
            self.used += nbytes
            ledger = self._ledger()
            if not self._seen and ledger is not None:
                # a ledger that cannot see torch's allocations counts the tensor itself, until its last view is
                # gone: the room released or dropped meanwhile, the tensor is still there
                tag = f"{self._loan}/tensor#{next(_LOANS)}"
                ledger.reserve(tag, nbytes, self.device)
                weakref.finalize(t.untyped_storage(), _give_back, weakref.ref(ledger), tag)
            self._hold()
        weakref.finalize(t.untyped_storage(), _emptied, weakref.ref(self), nbytes)
        if fill is not None:
            t.fill_(fill)
        return t

    def _hold(self) -> None:
        """what the room keeps from btb now: what its tensors have not taken (they count themselves - in btb's free
        reading, or where it cannot see them under tags of their own that outlive the room). Under the ledger's
        lock, which the release's takes too: a room let go meanwhile on another thread is not held again after"""
        with self._lock:
            ledger = self._ledger()
            if self.held and ledger is not None:
                ledger.reserve(self._loan, self.nbytes - self.used, self.device)

    def __enter__(self) -> Room:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _give_back(ledger: weakref.ref[DeviceLedger], tag: str | None) -> None:
    """lent memory back to btb's ledger (from a finalizer: any thread, any time) - a room's or a loan's the ledger
    holds under `tag`, or (None) one it saw itself; the lending policy regrows what making room shed"""
    led = ledger()
    if led is None:
        return
    if tag is not None:
        led.release(tag)
    led.returned = True


def _emptied(room: weakref.ref[Room], nbytes: int) -> None:
    """a room's tensor gone (from a finalizer): its bytes are the room's to hold again"""
    r = room()
    if r is not None:
        with r._lock:
            r.used -= nbytes
            r._hold()


class _MemoryMixin(_State):
    vram_pressure = staticmethod(vram_pressure)
    vram_pressure_line = staticmethod(vram_pressure_line)
    host_free_bytes = staticmethod(host_free_bytes)
    _darwin_available_bytes = staticmethod(_darwin_available_bytes)
    host_total_bytes = staticmethod(host_total_bytes)
    host_commit_bytes = staticmethod(host_commit_bytes)

    def vram_realloc(self, log: Log | None = None) -> list[str]:
        log = log or self.log
        done = []
        if self.resident_head and self.head is not None and self.dev.type == Device.CUDA:
            self.head = None
            torch.cuda.empty_cache()
            with self._meta:
                self.head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
            self._adopt(self.head, "weight", self._get(self.head_key))
            done.append("head")
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            self.aj = None
            torch.cuda.empty_cache()
            done.append("drafter")
        log(f"[vram] REALLOC {done or 'nothing'} (bled with room on the card); " + vram_pressure_line())
        return done

    def vram_shed(self, cache: Any = None, log: Log | None = None) -> str | None:
        log = log or self.log
        moved = None
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            self.aj = None
            self.drafter_dev = torch.device("cpu")
            moved = "drafter"
        elif self.resident:
            i = max(self.resident)
            tmpl = self.resident.pop(i)
            del tmpl
            self.host[i] = self._make_host_layer(i)
            self._caches_to(i, "cpu", cache)
            moved = f"layer {i}"
        elif self.resident_head and self.dev.type == Device.CUDA:
            self._head_host()
            self.head = None
            self.resident_head = False
            moved = "head"
        if moved is None:
            return None
        self._shed.append(moved)
        if self.dev.type == Device.CUDA:
            torch.cuda.empty_cache()
        log(f"[vram] SHED {moved} -> host (shed so far: {self._shed}); " + vram_pressure_line())
        return moved

    def vram_regrow(self, cache: Any = None, log: Log | None = None) -> Any:
        log = log or self.log
        if not self._shed:
            return None
        what = self._shed.pop()
        if what == "head":
            with self._meta:
                self.head = torch.nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
            self._adopt(self.head, "weight", self._get(self.head_key))
            self.resident_head = True
        elif what == "drafter":
            self.drafter_dev = None
            self.aj = None
        else:
            i = int(what.split()[1])
            tmpl = self._new_layer(i)
            self._load_layer(i, tmpl, first=True)
            if self.resident_fp32 and self.compute_dtype is not None and self.compute_dtype != torch.bfloat16:
                for p in tmpl.parameters():
                    p.data = p.data.to(self.compute_dtype)
            self.resident[i] = tmpl
            self.host.pop(i, None)
            self._caches_to(i, self.dev, cache)
        log(f"[vram] REGROW {what} -> card (still shed: {self._shed}); " + vram_pressure_line())
        return what

    def vram_trim(self, tag: str = "") -> Any:
        if self.dev.type != Device.CUDA:
            return 0.0, 0.0
        before = torch.cuda.memory_reserved() / 2**30
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        alloc, res = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
        if before - res > 0.05 or getattr(self, "vram_watch", False):
            self.log(
                f"[vram] trim{(' ' + tag) if tag else ''}: reserved {before:.2f} -> {res:.2f} GB (allocated {alloc:.2f})"
            )
        return alloc, res

    def vram_policy(self, cache: Any = None, log: Log | None = None) -> None:
        if not self.vram_watch or self.dev.type != Device.CUDA:
            return
        # a batched decode is sized by the scheduler (the one OOM guard), so the per-step streaming policy stands
        # aside: shedding mid-batch only slows it, and the triggers fire on a large batch's legitimate KV growth
        if cache is not None and getattr(cache, "layers", None):
            k0 = getattr(cache.layers[0], "keys", None)
            if k0 is not None and k0.numel() and int(k0.shape[0]) > 1:
                return
        # the pressure read is a PDH query (4.4 ms on this machine) - longer than a small model's whole step on
        # the card graph. A model holding every layer on the card with a quarter of the card still free has
        # nothing to shed, so the policy stands aside for it; elsewhere it reads pressure once a second, which
        # is how fast pressure moves, not once a token
        now = time.time()
        if now - self.vram_state.last_t < self.vram_state.period:
            return
        self.vram_state.last_t = now
        resident, L = getattr(self, "resident", None), getattr(self, "L", None)
        if (
            resident is not None
            and L is not None
            and len(resident) == L
            and not getattr(self, "host", None)
            and not getattr(self, "cold", None)
        ):
            total = self.vram_state.total
            if total is None:
                try:
                    total = int(torch.cuda.get_device_properties(self.dev).total_memory)
                except Exception:
                    total = 0
                self.vram_state.total = total
            room = self.device.free(unreserved=True)
            if total and room is not None and room > total // 4:
                return
        p = vram_pressure()
        if not p:
            return
        shared = p["process"]["shared"]
        floor = self.vram_state.shared_floor
        if floor is None:
            floor = self.vram_state.shared_floor = shared + (128 << 20)
        bled = shared > floor
        last, best = getattr(self, "_last_card_ms", 0.0), getattr(self, "_card_ms_min", None)
        contended = best is not None and last > 200.0 and last > self.vram_state.contention * best
        if contended:
            self.vram_state.contended += 1
        else:
            self.vram_state.contended = 0
        if not bled:
            self.vram_state.realloc_tried = False
        if bled and not self.vram_state.realloc_tried:
            room = self.device.free(unreserved=True)
            if room is not None and room > self._realloc_bytes():
                self.vram_state.realloc_tried = True
                self.device.request("realloc", lambda: self.vram_realloc(log))
                return
        if bled or self.vram_state.contended >= 2:
            self.vram_state.free_checks = 0
            self.vram_state.contended = 0
            why = (
                f"bled ({(shared - floor) / 2**20:.0f} MB above the floor)"
                if bled
                else f"card contended ({last:.0f} ms vs best {best:.0f})"
            )
            # shedding is a memory decision and the signals above are proxies for one (the PDH counter is noisy
            # under WDDM; a tree's verify pass looks contended next to a graphed one-token step): a resident
            # layer leaves the card only when the card is short of its margin; else the signal is logged once
            room = self.device.free(unreserved=True)
            if room is not None and room > 0:
                kind = why.split(" ")[0]
                if self.vram_state.held != kind:
                    (log or self.log)(
                        f"[vram] {why}, {room / 2**30:.1f} GB free above the margin: holding the card's layers"
                    )
                self.vram_state.held = kind
                return
            self.vram_state.held = None
            if self.device.request("shed", lambda: self.vram_shed(cache, log)) is not None:
                (log or self.log)(f"[vram] reason: {why}")
            self._card_ms_min = None
        elif self._shed:
            room = self.device.free(unreserved=True)
            if room is not None and room > self._regrow_bytes():
                self.vram_state.free_checks += 1
            else:
                self.vram_state.free_checks = 0
            if self.vram_state.free_checks >= self.vram_state.regrow_after:
                self.vram_state.free_checks = 0
                self.device.request("regrow", lambda: self.vram_regrow(cache, log))

    def ram_policy(self, log: Log | None = None) -> None:
        """Once a second: shed one warm layer to the ring after two consecutive readings with no free host memory
        above the reserve (or an OS low-memory signal). Regrow the last shed layer after `regrow_after`
        consecutive clean readings with room for it."""
        if not getattr(self, "ram_watch", False) or self.mlx is not None:
            return
        st = self.ram_state
        now = time.time()
        if now - st.last_t < st.period:
            return
        st.last_t = now
        faults = hard_page_faults()
        delta = faults - st.faults if st.faults is not None else 0
        st.faults = faults
        low = bool(memory_pressure().get("low"))
        room = self.device.free(torch.device("cpu"), unreserved=True)
        if low or (room is not None and room <= 0):
            st.paging += 1
            st.clean = 0
        else:
            st.clean += 1
            st.paging = 0
        warm = [i for i in sorted(self.host) if i not in self.cold]
        if st.paging >= 2 and warm:
            st.paging = 0
            why = f"{0 if room is None else room / 2**30:.2f} GB free above the reserve, hard faults +{delta}" + (
                ", the OS short of memory" if low else ""
            )
            self.device.request("ram-shed", lambda: self.ram_shed(why, log))
        elif st.shed and st.clean >= st.regrow_after:
            room = self.device.free(torch.device("cpu"), unreserved=True)
            need = self._layer_bytes_stored(st.shed[-1], bool(getattr(self, "_packed", None)))
            if room is not None and room > need:
                st.clean = 0
                self.device.request("ram-regrow", lambda: self.ram_regrow(log))

    def ram_shed(self, why: str = "", log: Log | None = None) -> int | None:
        """the last warm layer to the ring: its weights read each pass from the drive, its pages in RAM given back
        to the OS, the ring rebuilt over the new order (the next pass restarts its reader)"""
        log = log or self.log
        warm = [i for i in sorted(self.host) if i not in self.cold]
        if not warm:
            return None
        i = warm[-1]
        self._cold_stop()
        self.cold.add(i)
        self.ram_state.shed.append(i)
        self._bind_cold()
        log(f"[ram] SHED layer {i} -> drive ({why or 'asked'}); {len(self.cold)} layers from the drive each pass")
        return i

    def ram_regrow(self, log: Log | None = None) -> int | None:
        """the last layer shed back into RAM: its linears on the store's mapped bytes again, the ring rebuilt
        without it (emptied when it was the last cold layer)"""
        log = log or self.log
        if not self.ram_state.shed:
            return None
        i = self.ram_state.shed.pop()
        self._cold_stop()
        self.cold.discard(i)
        self._rebind_warm(i)
        if self.cold:
            self._bind_cold()
        else:
            self.cold_ring = ColdRing()
        log(f"[ram] REGROW layer {i} -> RAM (still shed: {self.ram_state.shed}); {len(self.cold)} from the drive")
        return i

    def layer_bytes(self) -> dict[int, int]:
        sizes: dict[int, int] = {}
        by_shard: dict[str, Any] = {}
        for k, sh in self.weight_map.items():
            by_shard.setdefault(sh, []).append(k)
        for sh, keys in by_shard.items():
            _, hdr, _ = self._shard(sh)
            for k in keys:
                if not k.startswith(self.prefix + "layers.") or not self._dense_key(k):
                    continue
                i = int(k[len(self.prefix) + len("layers.") :].split(".")[0])
                info = hdr[k]
                n = 1
                for d in info["shape"]:
                    n *= int(d)
                cast = self._cast_on_read(info)  # held as bf16 (tiers._held)
                b = 2 if cast else torch.empty(0, dtype=self.ST_DTYPES[info["dtype"]]).element_size()
                sizes[i] = sizes.get(i, 0) + n * b
        return sizes

    # -- lending: the caller's calls are `_LendMixin`'s, over these helpers `cache_room` shares ------------------

    def _lend(
        self, shape: int | Sequence[int], dtype: torch.dtype | None, device: DeviceSpec | None, fill: float | None
    ) -> torch.Tensor:
        dev = self._lend_device(device)
        dt = dtype if dtype is not None else torch.get_default_dtype()
        size = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        nbytes = math.prod(size) * torch.empty(0, dtype=dt).element_size()
        what = f"a {dt} tensor of shape {list(size)}"

        def run() -> torch.Tensor:
            self._lend_check(what)
            tried = self._make_room(dev, nbytes, what)
            while True:
                try:
                    t = torch.empty(size, dtype=dt, device=dev)
                    break
                except torch.OutOfMemoryError:
                    # room enough in all but one piece (the allocator's pool is split): one more step, then again
                    if not self._give_up_one(dev, nbytes, tried):
                        raise MemoryGrantError(self._short(dev, nbytes, what)) from None
            if fill is not None:
                t.fill_(fill)
            tag = None
            if not self._sees(dev):
                # a ledger that cannot see torch's allocations (MLX's) holds this one until its last view is gone
                tag = f"tensor#{next(_LOANS)}"
                self.device.reserve(tag, nbytes, dev)
            # either way its going is the lending policy's cue to regrow what making room for it shed
            weakref.finalize(t.untyped_storage(), _give_back, weakref.ref(self.device), tag)
            return t

        return self._serial(run)

    def _lend_device(self, device: DeviceSpec | None) -> torch.device:
        dev = torch_device(device) if device is not None else (self.dev if self.dev.type == Device.CUDA else None)
        dev = dev if dev is not None else torch.device("cpu")
        if dev.type not in (Device.CUDA, Device.CPU):
            raise ValueError(f"btb lends memory on a card or the host, not {dev}")
        return dev

    def _lend_check(self, what: str) -> None:
        if self.device.held():
            raise RuntimeError(
                f"{what} asked for inside a pass (a hook): btb moves nothing mid-pass; ask before the decode starts"
            )

    def _sees(self, dev: torch.device) -> bool:
        """whether btb's ledger on `dev` counts a torch allocation once it is made (MLX's counts only its own)"""
        return dev.type == Device.CUDA or getattr(self, "mlx", None) is None

    def _short(self, dev: torch.device, nbytes: int, what: str) -> str:
        room = int(self.device.free(dev, unreserved=True) or 0)
        return (
            f"{what} ({nbytes / 2**20:.1f} MiB) on {dev}: {room / 2**20:.1f} MiB free once btb gave up all it can "
            f"(model.memory() shows the room)"
        )

    def _make_room(self, dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
        """room for `nbytes` on `dev` above the margin and what is spoken for (all but the caller's `own`
        reservation), cheapest first; MemoryGrantError when everything btb can give leaves too little. Nothing to
        make where btb holds nothing (a card it does not run on). Returns the one-off steps taken, for a retry to
        skip"""
        tried: set[str] = set()
        while True:
            room = self.device.free(dev, unreserved=True, own=own)
            if room is None or room >= nbytes:
                return tried
            if not self._give_up_one(dev, nbytes - room, tried):
                self.device.returned = True  # refused: what was shed on the way grows back once there is room
                raise MemoryGrantError(self._short(dev, nbytes, what))

    def cache_room(self, cache: KvCache | None, B: int, T: int) -> None:
        """Room made, before a pass, for the buffers its cache appends will allocate. The scheduler's grant is
        asked for them inside the pass, where nothing may move, so a refusal there could only fail; here btb gives
        up what it holds, cheapest first, as `empty` does. Priced as the grant prices them. With `adapt` off the
        placement is pinned: nothing is given up, and the grant refuses what does not fit."""
        if cache is None or not getattr(self, "adapt", True):
            return
        first = next(((i, cl) for i, cl in enumerate(cache.layers) if isinstance(cl, GrowLayer)), None)
        if first is None:
            return
        c = self.cfg
        hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        dev0, dt0 = self._kv_home(*first)
        if not first[1].growth(B, T, Hk, d, dt0, dev0.type == Device.CPU):
            return  # the layers grow together: the first one fitting is the pass needing nothing
        need: dict[torch.device, int] = {}
        for i, cl in enumerate(cache.layers):
            if not isinstance(cl, GrowLayer):
                continue
            dev, dt = self._kv_home(i, cl)
            need[dev] = need.get(dev, 0) + cl.growth(B, T, Hk, d, dt, dev.type == Device.CPU)
        for dev, nbytes in need.items():
            if nbytes:
                self._make_room(dev, nbytes, f"the cache's growth for {B}x{T} rows", own=EPOCH)

    def _kv_home(self, i: int, cl: GrowLayer) -> tuple[torch.device, torch.dtype]:
        """where layer i's cache rows live and in what dtype: its buffer's, or where the layer runs"""
        if cl._buf is not None:
            return cl._buf[0].device, cl._buf[0].dtype
        cdt = self.compute_dtype if self.compute_dtype is not None else torch.bfloat16
        if cl.shared or getattr(self, "mlx", None) is not None:
            return torch.device("cpu"), cdt
        if i in self.resident:
            return (torch.device("cpu") if getattr(self, "kv_host", False) else self.dev), cdt
        return torch.device("cpu"), torch.float32  # a host layer runs in float32 on the CPU

    def _give_up_one(self, dev: torch.device, short: int, tried: set[str]) -> bool:
        """the cheapest thing btb holds on `dev`, given up toward `short` bytes: on a card the drafter, then layers
        from the top, then the head (each live cache following its layer); on the host MLX's cached buffers, the
        expert store's blocks, then a warm layer to the drive. False when nothing is left"""
        log = self.log
        lent = self.__dict__.setdefault("_lent_shed", [])
        if dev.type == Device.CUDA:
            if self.dev.type != Device.CUDA or self.device.request("lend", lambda: self.vram_shed(None, log)) is None:
                return False
            if not self.vram_watch:  # a running policy regrows what it sheds; none runs to regrow this
                lent.append("card")
            return True
        mlx = getattr(self, "mlx", None)
        if mlx is not None and mlx.held_bytes() > mlx.active_bytes():
            mlx.clear_cache()
            return True
        store = getattr(self, "expert_store", None)
        if store is not None and store.blocks and "store" not in tried:
            # the store gives blocks back while the host is short of its reserve: short by what is missing, now
            tried.add("store")
            store.reserve += short
            try:
                store.release()
            finally:
                store.reserve -= short
            return True
        if self.device.request("lend", lambda: self.ram_shed("room asked for", log)) is None:
            return False
        if mlx is not None:
            mlx.clear_cache()
        if mlx is not None or not self.ram_watch:
            lent.append("host")
        return True

    def lend_policy(self) -> None:
        """Once a second: what making room shed grows back once lent memory has come back and there is room for it
        again - where no memory policy runs to regrow it (a card with `vram_watch` off, the host on MLX)"""
        lent = self.__dict__.get("_lent_shed")
        if not lent or not self.device.returned:
            return
        now = time.time()
        if now - self.__dict__.get("_lent_t", 0.0) < 1.0:
            return
        self._lent_t = now
        if lent[-1] == "card":
            room = self.device.free(self.dev, unreserved=True)
            if room is not None and room > self._regrow_bytes():
                lent.pop()
                self.device.request("regrow", lambda: self.vram_regrow(None, self.log))
        elif self.ram_state.shed:
            room = self.device.free(torch.device("cpu"), unreserved=True)
            need = self._layer_bytes_stored(self.ram_state.shed[-1], bool(getattr(self, "_packed", None)))
            if room is not None and room > need:
                lent.pop()
                self.device.request("ram-regrow", lambda: self.ram_regrow(self.log))
        else:
            lent.pop()
        if not lent:
            self.device.returned = False

    def _sheddable(self, dev: torch.device) -> int:
        """what `_give_up_one` can free on `dev`, in bytes. `memory()` asks from any thread, as a decode's shed or
        regrow moves layers: it counts over copies of what it reads"""
        if dev.type == Device.CUDA:
            if self.dev.type != Device.CUDA:
                return 0
            fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
            resident = list(self.resident)
            n = sum(self._layer_bytes(i) for i in resident) * (2 if (fp32 and self.resident_fp32) else 1)
            head = self.head
            if self.resident_head and head is not None:
                n += head.weight.numel() * head.weight.element_size()
            aj = getattr(self, "aj", None)
            if aj is not None and aj.dev.type == Device.CUDA:
                n += sum(self._get(k).numel() * 2 for k in self.weight_map if k.startswith("mtp."))
            return n + sum(_bytes_on(c, dev, resident) for c in list(self.__dict__.get("_live_caches", ())))
        packed = bool(getattr(self, "_packed", None))
        n = sum(self._layer_bytes_stored(i, packed) for i in list(self.host) if i not in self.cold)
        store = getattr(self, "expert_store", None)
        if store is not None and store.per:
            n += store.live() * int(store.per)
        mlx = getattr(self, "mlx", None)
        if mlx is not None:
            n += max(0, mlx.held_bytes() - mlx.active_bytes())
        return n


_LOANS: Iterator[int] = itertools.count(1)


def _bytes_on(cache: KvCache, dev: torch.device, layers: Iterable[int]) -> int:
    """the bytes `cache` holds on `dev` for `layers`"""
    n = 0
    for i in layers:
        if i >= len(cache.layers):
            continue
        cl = cache.layers[i]
        for attr in ("keys", "values"):
            t = getattr(cl, attr, None)
            if isinstance(t, torch.Tensor) and t.device.type == dev.type:
                n += t.numel() * t.element_size()
    return n


@api("model")
class _LendMixin(_MemoryMixin):
    """lending: memory for the caller's own tensors, the model's API over `_MemoryMixin`'s policies"""

    def empty(
        self, shape: int | Sequence[int], dtype: torch.dtype | None = None, device: DeviceSpec | None = None
    ) -> torch.Tensor:
        """
        A tensor of your own beside the model, as `torch.empty` makes it, with room made for it first: btb gives up
        what it holds there, cheapest first, instead of your allocation running out of memory. It is a plain tensor
        and lives as long as you keep it; btb takes the room back once it is gone. `device` defaults to the one the
        model computes on (the host on Apple silicon). A `MemoryGrantError` when even that is not enough.
        """
        return self._lend(shape, dtype, device, None)

    def zeros(
        self, shape: int | Sequence[int], dtype: torch.dtype | None = None, device: DeviceSpec | None = None
    ) -> torch.Tensor:
        """`empty`, filled with zeros"""
        return self._lend(shape, dtype, device, 0)

    def full(
        self,
        shape: int | Sequence[int],
        value: float,
        dtype: torch.dtype | None = None,
        device: DeviceSpec | None = None,
    ) -> torch.Tensor:
        """`empty`, filled with `value`"""
        return self._lend(shape, dtype, device, value)

    def room(self, nbytes: int, device: DeviceSpec | None = None, name: str = "room") -> Room:
        """
        Room for memory btb does not allocate itself - a second model, a library's workspace - made now and kept
        from btb until the `Room` is released (`release()`, or as a `with` block). Keep it while that memory is in
        use; tensors taken through it (`Room.empty`) count against it, not beside it. A `MemoryGrantError` when
        btb cannot make that much.
        """
        dev = self._lend_device(device)
        tag = f"{name}#{next(_LOANS)}"

        def run() -> Room:
            self._lend_check(name)
            self._make_room(dev, int(nbytes), f"room {name!r}")
            self.device.reserve(tag, int(nbytes), dev)
            return Room(self.device, tag, int(nbytes), dev, self._sees(dev), self)

        return self._serial(run)

    def memory(self) -> dict[str, DeviceMemory]:
        """what btb sees of each device it runs on, by name ("cuda:0", "cpu") - what `empty` and `room` can get"""
        out = {}
        devs = [self.dev] if self.dev.type == Device.CUDA else []
        for dev in [*devs, torch.device("cpu")]:
            out[str(dev)] = DeviceMemory(
                str(dev),
                int(self.device.free(dev, unreserved=True) or 0),
                self.device.reserved(dev),
                self._sheddable(dev),
            )
        return out
