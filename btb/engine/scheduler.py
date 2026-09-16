# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The batch scheduler: how many sequences decode together, how much KV each reserves, the one gate every
large allocation asks before it is made, the machine's cache sizes, and the Route: the drive under the model's
files, probed through our own reader, and the queue every read of it goes through."""

from __future__ import annotations

import contextlib
import heapq
import os
import random
import sys
import threading
import time
import types as types_
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch

from .. import pool
from ..kinds import LayerKind, Log, Tier
from ..options import Device, DeviceName
from ..sysinfo import (
    host_cache_sizes,
    host_commit_bytes,
    host_free_bytes,
    host_total_bytes,
    os_memory_floor,
    process_working_set_bytes,
)

if TYPE_CHECKING:
    pass


def _size(n: int) -> str:
    """bytes at the unit that shows them: GiB to two places, else MiB, else KiB"""
    n = int(n)
    if n >= 2**30 // 100:
        return f"{n / 2**30:.2f} GiB"
    if n >= 2**20 // 100:
        return f"{n / 2**20:.2f} MiB"
    return f"{n / 2**10:.1f} KiB"


class MemoryGrantError(MemoryError):
    """A memory request the scheduler refused. Raised before the allocation, so the message names what asked
    and for how much - torch's own OOM is raised from inside the allocator and names neither."""


class PlanError(MemoryGrantError):
    """A placement the host cannot carry: the memory above the floor does not hold the smallest working set a
    plan needs - the ring's slots, the head and the drafter where they run on the host, the cache's first rows,
    the staging a streamed layer crosses through. Raised before anything is loaded, naming the shortfall."""


@dataclass(frozen=True)
class PlanBytes:
    """The sizes a plan priced, in bytes."""

    vram_layers: int
    head: int
    drafter: int
    warm: int
    cold: int
    slots: int
    shadow: int
    templates: int
    kv_card: int = 0
    kv_host: int = 0
    staging: int = 0


@dataclass(frozen=True)
class PlanCaps:
    """The memory a plan was drawn against, in GB."""

    ram_gb: float
    vram_gb: float
    os_reserve_gb: float
    vram_reserve_gb: float


@dataclass(frozen=True)
class PlanFree:
    """What was free when a plan was taken, in GB, and how long it waited for memory to come back."""

    vram_gb: float
    ram_gb: float
    ram_gb_first: float
    settle_s: float


@dataclass(frozen=True)
class HostBudget:
    """The host's RAM as the plan and the engine spend it, read once when a plan is drawn. Nothing in it is
    a share of the machine, and every reader of a reserve takes it from here."""

    total: int
    available: int  # what the OS would hand out now (AvailPhys / MemAvailable / free + inactive) plus the pool's blocks
    commit: int  # the commit charge still open to private allocations
    footprint: int  # this process's working set at the reading; already out of `available`, kept to read a run's growth
    os_floor: int  # what the OS itself keeps free where it publishes it (Linux's zone watermarks); 0 elsewhere
    growth: int  # the run's own allocations past the buffers the plan prices, from the model's shape
    floor: int  # the RAM left to other programs: --ram-reserve's figure, else the larger of RAM_FLOOR_SHARE of
    # the available RAM and os_floor + growth

    @property
    def reserve(self) -> int:
        return int(self.floor)

    @property
    def spendable(self) -> int:
        """what a plan may place in RAM: the available memory above the floor"""
        return max(0, int(self.available) - int(self.floor))


@dataclass(frozen=True)
class DriveBenchmark:
    """The drive under a model's files, probed once per volume and kept for the process: the plan prices the
    cold tier from `bps`, and the engine's Route (`BatchScheduler.disk`) binds the same measurement instead of
    measuring again. `profile` is the Route's record: the fixed cost of a read, an expert-sized read alone,
    the burst rates by readers in flight, the sequential rate, and the rule's depth, merge and ahead."""

    volume: Any
    profile: dict[str, Any]
    _kept: ClassVar[dict[Any, DriveBenchmark]] = {}

    @property
    def measured(self) -> bool:
        return bool(self.profile.get("measured"))

    @property
    def bps(self) -> float:
        """bytes a second the cold tier streams at: the burst rate at the rule's depth, else the sequential
        rate; 0 without a probe"""
        rates = self.profile.get("rates") or {}
        gbs = float(rates.get(int(self.profile.get("depth", 0) or 0), (0.0, 0.0))[0]) if rates else 0.0
        return (gbs or float(self.profile.get("seq_gbs", 0.0) or 0.0)) * 1e9

    @classmethod
    def kept(cls, volume: Any) -> DriveBenchmark | None:
        return cls._kept.get(volume)

    @classmethod
    def take(cls, volume: Any, path: str, clock: Any = None) -> DriveBenchmark:
        """the volume's benchmark: the one kept, else the drive probed now on `path` and kept when the probe ran. A
        simulated drive (`clock` given, tests/drive_sim.py) is probed every time and never kept."""
        b = cls._kept.get(volume) if clock is None else None
        if b is None:
            p = BatchScheduler.measure_drive(path, clock=clock)
            p.update(BatchScheduler._disk_rule(p))
            b = cls(volume, p)
            if clock is None and b.measured:
                cls._kept[volume] = b
        return b


@dataclass(frozen=True)
class Plan:
    """A placement: the layers the card holds, the ones the host runs from RAM, the ones streamed from the drive
    each pass, where the head and the drafter sit, and the sizes and free memory it was priced from."""

    device: str
    resident: tuple[int, ...]
    host: tuple[int, ...]
    cold: tuple[int, ...]
    warm: tuple[int, ...]
    head_on_card: bool
    drafter_on_card: bool
    prefill_card: bool
    kv_host: bool
    has_mtp: bool
    moe: bool
    predicted_ms_per_token: float
    bytes: PlanBytes
    caps: PlanCaps
    free: PlanFree
    budget: HostBudget | None = None
    drive: DriveBenchmark | None = None  # the drive's measurement where layers stream from it

    def __str__(self) -> str:
        b = self.bytes
        head = Tier.CARD if self.head_on_card else Tier.HOST
        drafter = Tier.CARD if self.drafter_on_card else (Tier.HOST if self.has_mtp else Tier.NONE)
        s = (
            f"{self.device}: {len(self.resident)} resident ({b.vram_layers / 2**30:.2f} GB), {len(self.host)} host "
            f"({len(self.cold)} streamed from the drive, {b.warm / 2**30:.2f} GB in RAM, {b.cold / 2**30:.2f} GB a pass), "
            f"head {head}, drafter {drafter}{', cache in RAM' if self.kv_host else ''}; "
            f"{self.predicted_ms_per_token:.1f} ms a token predicted; "
            f"free RAM {self.free.ram_gb:.1f} GB, VRAM {self.free.vram_gb:.1f} GB"
        )
        if self.budget is not None:
            s += f"; {self.budget.floor / 2**30:.2f} GB kept free"
        if self.drive is not None:
            s += f"; the drive {self.drive.bps / 1e9:.2f} GB/s"
        return s

    def cold_slots(self, warm_bytes: int | None = None, n_cold: int | None = None) -> int:
        """Slots for the cold reader's ring: the two the plan reserved plus what keeps the drive busy through the
        warm layers' compute, within the memory the plan left and the cold layers themselves. Under a named CPU
        share (`--cpu-layers N`: `warm_bytes` what the host keeps, `n_cold` the rest) the ring is the RAM tier of
        the layers no host kernel runs - a slot still holding its layer is not read again - so every cold layer
        the spare memory holds keeps a slot."""
        slot_b = self.bytes.slots // 2
        named = warm_bytes is not None
        warm = self.bytes.warm if warm_bytes is None else int(warm_bytes)
        cold = len(self.cold) if n_cold is None else int(n_cold)
        if cold <= 2 or slot_b <= 0:
            return 2
        # the plan's working room and reserve: the budget's one floor where the plan carried one, else the two
        # figures a plan priced without it
        kept = self.budget.floor if self.budget is not None else 2 * 2**30 + int(self.caps.os_reserve_gb * 2**30)
        spare = (
            int(self.caps.ram_gb * 2**30)
            - kept
            - warm
            - (0 if self.head_on_card else self.bytes.head)
            - (0 if self.drafter_on_card else self.bytes.drafter)
            - self.bytes.slots
            - self.bytes.kv_host
            - self.bytes.staging
        )
        if named:
            wanted = cold
        else:
            from .tiers import DRIVE_BPS, RAM_BPS

            warm_ms = warm / RAM_BPS * 1e3  # the plan's own rates
            read_ms = slot_b / (self.drive.bps if self.drive is not None and self.drive.bps > 0 else DRIVE_BPS) * 1e3
            wanted = 2 + int(-(-warm_ms // read_ms)) if read_ms > 0 else 2
        extra = max(0, min(wanted - 2, spare // slot_b, cold - 2))
        return 2 + int(extra)


class SchedulerHost(Protocol):
    """What the scheduler reads of the engine it serves; anything with these attributes may stand in for one
    (`BatchScheduler.for_model` builds one over a transformers model), which is how a program with a model
    that is not btb's gets the grant gate, the epoch sizing and the KV pricing over it."""

    cfg: Any
    layer_types: Sequence[LayerKind]
    compute_dtype: torch.dtype | None
    dev: torch.device
    ram_reserve: int
    vram_margin: int

    def log(self, *a: Any, **k: Any) -> Any: ...


@dataclass
class _Host:
    """A `SchedulerHost` over a model that is not the engine's."""

    cfg: Any
    layer_types: list[LayerKind]
    compute_dtype: torch.dtype | None
    dev: torch.device
    ram_reserve: int
    vram_margin: int
    log: Log
    mlx: Any = None
    kv_bits: int | None = None
    mem_start: int = 0


class BatchScheduler:
    """Decides how many sequences decode together and the KV each reserves, from the memory free at plan time;
    a batch past it runs as epochs. The size is chosen when an epoch forms and held for its life. Also where a
    large allocation asks first (`grant`): the engine's other buffers reached torch unasked and OOM'd from inside
    the allocator, naming nothing."""

    # a grant at or above this fraction of the free memory is logged as it is given: it fits, so it is not
    # refused, but it is the size at which a growth bug first shows itself and the line dates the request
    warn_fraction: float = 0.25
    # the drive probe's clock; a simulated drive's in the tests, the wall's when None
    _disk_clock: Callable[[], float] | None = None
    # GPU backpressure: a transient Metal recovery (a discarded command buffer) is the card's own signal to ease
    # off for a few cycles, the compute-side analogue of a saturated drive. The streak escalates the pause and a
    # clean stretch forgets it.
    _gpu_streak: int = 0
    _gpu_cool_until: float = 0.0

    @staticmethod
    def growth_estimate(probe: Any, rows: int = 16) -> int:
        """The RAM a run allocates past the buffers a plan prices, from the model's own shape: the widest pass's
        float32 activations - `rows` rows through the residual and the attention's projections (3 x hidden)
        and the MLP's two intermediates, two layers in flight - and its logits over the vocabulary. The figure
        a run then records (`report()["growth"]`: the working set past the plan) is what this is checked
        against."""
        c = probe.cfg
        H = int(c.hidden_size)
        inter = int(getattr(c, "intermediate_size", None) or getattr(c, "moe_intermediate_size", None) or 4 * H)
        V = int(c.vocab_size)
        return 2 * int(rows) * (3 * H + 2 * inter) * 4 + int(rows) * V * 4

    VRAM_MARGIN_GB = 0.5  # the card's margin: this much, or 8% of a card smaller than that makes
    VRAM_MARGIN_SHARE = 0.08
    RAM_FLOOR_SHARE = 0.10  # of the RAM available at load: the least the host keeps for other programs

    @staticmethod
    def vram_margin_gb(total_bytes: int) -> float:
        """the VRAM kept free by default on a card of `total_bytes`: the smaller of 0.5 GB and 8% of the card,
        so the card is used as fully as its size allows; `--vram-reserve` names another"""
        return min(BatchScheduler.VRAM_MARGIN_GB, total_bytes / 2**30 * BatchScheduler.VRAM_MARGIN_SHARE)

    @staticmethod
    def measure_host(floor_gb: float | None = None, growth: int = 0) -> HostBudget:
        """The host's RAM as it stands: the OS's figures, the pool's blocks counted as available (they are this
        model's memory already), this process's working set, what the OS itself keeps free where it says so,
        and the floor, the RAM left to other programs: `floor_gb` where the caller names one (--ram-reserve),
        else the larger of `RAM_FLOOR_SHARE` of the available RAM and the OS's figure plus the run's `growth`.
        The floor moves the plan's bottom line only; the run still gives memory back under pressure."""
        os_floor = int(os_memory_floor())
        available = int(host_free_bytes()) + int(pool.POOL.free_bytes())
        floor = (
            int(float(floor_gb) * 2**30)
            if floor_gb is not None
            else max(os_floor + int(growth), int(BatchScheduler.RAM_FLOOR_SHARE * available))
        )
        return HostBudget(
            total=int(host_total_bytes()),
            available=available,
            commit=int(host_commit_bytes()),
            footprint=int(process_working_set_bytes()),
            os_floor=os_floor,
            growth=int(growth),
            floor=floor,
        )

    @classmethod
    def for_model(
        cls,
        model: Any,
        device: str | torch.device = "cuda",
        ram_reserve_gb: float | None = None,
        vram_reserve_gb: float | None = None,
        log: Log | None = None,
    ) -> BatchScheduler:
        """The scheduler over a transformers model (or its config) that is not the engine's: the KV priced from
        the config's heads, the host floor as a plan takes it (`ram_reserve_gb` names one), the
        card's margin as a load takes it (`vram_margin_gb`; `vram_reserve_gb` names one)."""
        cfg = getattr(model, "config", model)
        cfg = getattr(cfg, "text_config", cfg)
        L = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        types = [LayerKind.of(t) for t in (getattr(cfg, "layer_types", None) or [LayerKind.FULL] * L)]
        dtype = None
        params = getattr(model, "parameters", None)
        if callable(params):
            dtype = next((p.dtype for p in params()), None)
        dev = torch.device(device)
        if vram_reserve_gb is None:
            vram_reserve_gb = (
                BatchScheduler.vram_margin_gb(torch.cuda.get_device_properties(dev).total_memory)
                if dev.type == Device.CUDA and torch.cuda.is_available()
                else 0.0
            )
        growth = 0
        if all(hasattr(cfg, k) for k in ("hidden_size", "vocab_size")):
            growth = cls.growth_estimate(types_.SimpleNamespace(cfg=cfg))
        hb = cls.measure_host(ram_reserve_gb, growth)
        host = _Host(
            cfg=cfg,
            layer_types=types,
            compute_dtype=dtype,
            dev=dev,
            ram_reserve=int(hb.floor),
            vram_margin=int(float(vram_reserve_gb) * 2**30),
            log=log or (lambda *_a, **_k: None),
        )
        return cls(host)

    def __init__(self, sm: Any) -> None:
        self.sm = sm
        self.batch: int | None = None  # the size held for the current epoch (sticky)
        self.reserve: int | None = None  # the KV length reserved for it
        self._caches: dict[str, int] | None = None
        self.granted: dict[str, int] = {}  # bytes granted so far by kind and device ("kv@cuda"): the report's ledger

    def gpu_recovered(self) -> float:
        """Register a transient GPU recovery and return the seconds to pause before the retry: escalating with a
        recent streak (0.1, 0.2, 0.4, ... capped at 2s), so a one-off blip costs about one pause and sustained
        pressure backs off harder. A stretch of clean passes past the last cooldown clears the streak."""
        now = time.monotonic()
        if now > self._gpu_cool_until + 5.0:
            self._gpu_streak = 0
        self._gpu_streak += 1
        wait = min(2.0, 0.1 * float(2 ** (self._gpu_streak - 1)))
        self._gpu_cool_until = now + wait
        return wait

    def caches(self) -> dict[str, int]:
        """The memory hierarchy's sizes, read once and held here: the card's L2, the most of it the driver
        sets aside as persisting, the widest persisting window one stream may name, and the host's L2 and L3
        (bytes; 0 where the level does not exist or cannot be read). The engine's placement and pinning
        decisions read these; nothing else queries the driver or the OS for them."""
        if self._caches is not None:
            return self._caches
        cs = {"gpu_l2": 0, "gpu_l2_persist": 0, "gpu_l2_window": 0, "host_l2": 0, "host_l3": 0}
        from .native import Native

        k = Native.cuda
        if k is not None and self.sm.dev.type == Device.CUDA:
            lim, win = k.persist_limit()
            cs.update(gpu_l2=k.l2_bytes(), gpu_l2_persist=lim, gpu_l2_window=win)
        cs.update(host_cache_sizes())
        self._caches = cs
        return cs

    def pin_bytes(self, wanted: int) -> int:
        """How many bytes of a region the card should hold as persisting L2: the request clipped to what the
        driver sets aside and to the widest window, never more than asked."""
        cs = self.caches()
        return max(0, min(int(wanted), cs["gpu_l2_persist"], cs["gpu_l2_window"]))

    def _kv_bytes_per_row_token(self) -> int:
        # one sequence's KV growth per decoded position: k and v over the full-attention layers only (a
        # linear-attention layer carries a fixed-size recurrent state, not a per-position cache)
        c = self.sm.cfg
        hq = int(c.num_attention_heads)
        hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        n_attn = sum(1 for lt in self.sm.layer_types if lt != LayerKind.LINEAR)
        dt = self.sm.compute_dtype if self.sm.compute_dtype is not None else torch.bfloat16
        el = torch.empty(0, dtype=dt).element_size()
        if getattr(self.sm, "kv_bits", None) == 8:
            el = 1  # an int8 cache: one byte a value (its float32 scale per row is 4 / d of that)
        return 2 * n_attn * hk * d * el

    def kv_row_bytes(self, target_len: int) -> int:
        """The KV one sequence holds when it has run to `target_len` positions."""
        return self._kv_bytes_per_row_token() * max(1, int(target_len))

    def free_vram(self) -> int | None:
        """VRAM free right now above the engine's reserve, or None off the card (the host tier is bound by
        CPU throughput, not KV memory, so it does not size a batch this way)."""
        if self.sm.dev.type != Device.CUDA and getattr(self.sm, "mlx", None) is None:
            return None
        return self.free_for(None)

    def free_for(self, device: Any = None) -> int | None:
        """What an allocation on `device` (the engine's own when None) may take: the card's free VRAM above
        the engine's margin, MLX's ledger on the unified device (the host and the GPU spend one pool), or the
        host's free RAM above the engine's RAM reserve. None when nothing here can price the device. The
        arithmetic is the device's (`Device.free`), the one ledger the plan and the memory policy read too;
        an engine built without one (a stub under test) is priced the same way directly."""
        dv = getattr(self.sm, "device", None)
        if dv is not None:
            return dv.free(device)
        from .device import free_bytes

        dev = self.sm.dev if device is None else torch.device(device)
        if dev.type == Device.CUDA:
            if self.sm.dev.type != Device.CUDA:
                return None
            return free_bytes(dev, int(getattr(self.sm, "vram_margin", 0) or 0))
        if getattr(self.sm, "mlx", None) is not None:
            # unified memory: the engine's own ledger - the RAM the load started with, less the reserve, less
            # everything MLX holds (exact whether or not the pages are touched; see the expert store)
            return max(0, int(self.sm.mem_start) - int(self.sm.ram_reserve) - int(self.sm.mlx.held_bytes()))
        return max(0, host_free_bytes() - int(getattr(self.sm, "ram_reserve", 0) or 0))

    def grant(
        self,
        nbytes: int,
        kind: str,
        *,
        requester: str = "",
        B: int = 1,
        cap: int | None = None,
        bound: int | None = None,
        device: Any = None,
    ) -> None:
        """Ask before allocating: returns on a request the engine can afford, raises `MemoryGrantError` on one
        it cannot. One free-memory read: for an allocation path (a growth, a tier load), never a per-token one.
        A `kv` request whose per-row capacity `cap` is past `bound` (the length the sequence itself can reach)
        is refused whatever its size: a growth bug, not a need. A large but affordable request is logged."""
        nbytes = int(nbytes)
        who = requester or kind
        if kind == "kv" and cap and bound and int(cap) > int(bound):
            raise MemoryGrantError(
                f"[grant] REFUSED {who}: kv cap {int(cap)} rows past the sequence ceiling {int(bound)} "
                f"(B={int(B)}, {_size(nbytes)}) - the buffer is growing past anything these rows reach"
            )
        free = self.free_for(device)
        if free is None:
            return
        dev = self.sm.dev if device is None else torch.device(device)
        # the card's margin is its OOM guard and the host's floor is the OS's own (or the one --ram-reserve
        # names): a request past either is refused
        if nbytes > free:
            raise MemoryGrantError(
                f"[grant] REFUSED {who}: {kind} {_size(nbytes)} on {dev}, only {_size(free)} free{self._ledger_line()}"
            )
        tag = f"{kind}@{dev.type}"
        self.granted[tag] = self.granted.get(tag, 0) + nbytes
        if free and nbytes >= self.warn_fraction * free:
            self.sm.log(f"[grant] LARGE {who}: {kind} {_size(nbytes)} of {_size(free)} free")

    def _ledger_line(self) -> str:
        """on MLX, the arithmetic behind the free figure, which the OS's own count does not show: the RAM at
        load, less the reserve, less what MLX holds"""
        mlx = getattr(self.sm, "mlx", None)
        if mlx is None:
            return ""
        with contextlib.suppress(Exception):
            return (
                f" (MLX's ledger: {_size(self.sm.mem_start)} at load, less {_size(self.sm.ram_reserve)} reserved, "
                f"less {_size(mlx.held_bytes())} held by MLX)"
            )
        return ""

    def max_batch(self, target_len: int) -> int | None:
        """Sequences to decode together: what the free VRAM holds for `target_len` positions of KV. None off the
        card (the host tier scales by its own kernels)."""
        free = self.free_vram()
        if free is None:
            return None
        return max(1, int(free // max(1, self.kv_row_bytes(target_len))))

    def plan(self, n_pending: int, target_len: int) -> tuple[int, int]:
        """Size the next epoch: at most `n_pending` sequences, at most what the free VRAM holds for
        `target_len` positions of KV. Returns (batch, reserve_len). Call it only when forming an epoch."""
        target_len = int(target_len)
        mb = self.max_batch(target_len)
        batch = int(n_pending) if mb is None else max(1, min(int(n_pending), mb))
        self.batch, self.reserve = batch, target_len
        dv = getattr(self.sm, "device", None)
        if dv is not None and mb is not None:
            # the epoch's KV is spoken for from here to `release()`: the memory policy must not read it as
            # free and shed a layer to make room the batch is about to take anyway
            dv.reserve("epoch", self.kv_row_bytes(target_len) * batch)
        return batch, target_len

    def release(self) -> None:
        """The epoch is over: its KV reservation is let go (the buffers themselves free with the cache)."""
        dv = getattr(self.sm, "device", None)
        if dv is not None:
            dv.release("epoch")

    @staticmethod
    def plan_placement(
        probe: Any,
        device: Any,
        vram_gb: float,
        *,
        packed: bool,
        fp32: bool,
        vram_reserve_gb: float,
        os_reserve_gb: float | None = None,
        budget: HostBudget | None = None,
        context: int = 0,
        kv_host: bool | None = None,
        settle_s: float = 30.0,
        drive: DriveBenchmark | None = None,
    ) -> Plan:
        """The placement for `probe` (an engine opened on the CPU to price its layers) against the host budget
        (`budget`, else `measure_host` now, with `os_reserve_gb` as the floor where one is named) and
        the `vram_gb` the caller read. The budget's floor is the plan's whole reserve: its working room is 0,
        because the footprint a working room stood for is already out of the available figure. A process that
        just exited gives its memory back over several seconds, and a plan taken in that window would put layers
        on the drive for the whole run: while the plan is short of RAM and free RAM is still rising, it waits
        (up to `settle_s`) and prices again. `kv_host` None prices the attention cache on the card and in RAM
        and keeps RAM only where the layers the cache would evict stream at more a token than the host's
        attention reads at the full context; True or False is the placement asked for. `drive` is the drive's
        measurement to price the cold tier from; None measures it on the model's largest file when the
        placement streams layers, once per volume for the process (the engine's Route reuses it)."""
        name = DeviceName.parse(device)
        card = name is not None and name.kind.card
        has_mtp = any(k.startswith("mtp.") for k in probe.weight_map)
        if budget is None:
            budget = BatchScheduler.measure_host(os_reserve_gb, BatchScheduler.growth_estimate(probe))
        hb = budget
        ram_gb = hb.available / 2**30
        price = lambda ram, kv, bps: probe.plan_budget(
            ram,
            vram_gb,
            packed=packed,
            fp32=fp32,
            working_ram_gb=0.0,
            drafter=has_mtp,
            prefill_card=card,
            os_reserve_gb=hb.floor / 2**30,
            vram_reserve_gb=vram_reserve_gb,
            context=int(context or 0),
            kv_host=kv,
            drive_bps=bps,
        )

        def choose(ram: float, bps: float | None = None) -> tuple[dict[str, Any], bool]:
            if kv_host is not None or not card:
                kv = bool(kv_host)
                return price(ram, kv, bps), kv
            on_card = price(ram, False, bps)
            if not on_card["host"]:
                return on_card, False
            in_ram = price(ram, True, bps)
            cost = lambda o: float(o["predicted_ms_per_token"]) + float(o.get("kv_read_ms", 0.0))
            return (in_ram, True) if cost(in_ram) < cost(on_card) else (on_card, False)

        out, kv_chosen = choose(ram_gb)
        ram0, waited = ram_gb, 0.0
        while out["cold"] and waited < settle_s:
            time.sleep(2)
            waited += 2
            again = BatchScheduler.measure_host(hb.floor / 2**30)
            now_gb = again.available / 2**30
            if now_gb - ram_gb < 0.25:
                break
            hb, ram_gb = again, now_gb
            out, kv_chosen = choose(ram_gb)
        if drive is None and out["cold"]:
            f = BatchScheduler._drive_file(probe)
            if f is not None:
                drive = DriveBenchmark.take(BatchScheduler._volume(f), f)
        if drive is not None and drive.measured:
            out, kv_chosen = choose(ram_gb, drive.bps)  # the cold tier at the drive's own rate
        else:
            drive = None
        b, cp = out["bytes"], out["caps"]
        # the smallest working set the placement needs in RAM whatever streams: refused by name when the host
        # cannot carry it, instead of a plan with every layer on the drive and nowhere to land them
        least = (
            int(b["slots"])
            + (0 if out["head_on_card"] else int(b["head"]))
            + (0 if out["drafter_on_card"] else int(b["drafter"]))
            + int(b.get("kv_host", 0))
            + int(b.get("staging", 0))
        )
        room = min(hb.available, hb.commit) - hb.floor
        if room < least:
            raise PlanError(
                f"[plan] REFUSED: the host has {hb.available / 2**30:.2f} GB available ({hb.commit / 2**30:.2f} GB of "
                f"commit) above the {hb.floor / 2**30:.2f} GB it keeps, and the smallest placement needs "
                f"{least / 2**30:.2f} GB in RAM: the ring's slots, the head and the drafter where they run on the "
                f"host, the cache's first rows, the staging"
            )
        return Plan(
            device=str(device),  # the record's device as a name ('cuda:1', 'mlx', 'cpu'): it is written to the report
            resident=tuple(out["resident"]),
            host=tuple(out["host"]),
            cold=tuple(out["cold"]),
            warm=tuple(out["warm"]),
            head_on_card=bool(out["head_on_card"]),
            drafter_on_card=bool(out["drafter_on_card"]),
            prefill_card=bool(out["prefill_card"]),
            kv_host=kv_chosen,
            has_mtp=has_mtp,
            moe=bool(probe.fam.moe),
            predicted_ms_per_token=float(out["predicted_ms_per_token"]),
            bytes=PlanBytes(
                vram_layers=int(b["vram_layers"]),
                head=int(b["head"]),
                drafter=int(b["drafter"]),
                warm=int(b["warm"]),
                cold=int(b["cold"]),
                slots=int(b["slots"]),
                shadow=int(b["shadow"]),
                templates=int(b["templates"]),
                kv_card=int(b.get("kv_card", 0)),
                kv_host=int(b.get("kv_host", 0)),
                staging=int(b.get("staging", 0)),
            ),
            caps=PlanCaps(
                ram_gb=float(cp["ram_gb"]),
                vram_gb=float(cp["vram_gb"]),
                os_reserve_gb=float(cp["os_reserve_gb"]),
                vram_reserve_gb=float(cp["vram_reserve_gb"]),
            ),
            free=PlanFree(vram_gb=vram_gb, ram_gb=ram_gb, ram_gb_first=ram0, settle_s=waited),
            budget=hb,
            drive=drive,
        )

    @staticmethod
    def _drive_file(probe: Any) -> str | None:
        """the largest of the model's weight files, the one the drive probe reads; None for a probe without
        files on disk"""
        d = str(getattr(probe, "dir", "") or "")
        best, size = None, 0
        for f in {str(v) for v in (getattr(probe, "weight_map", None) or {}).values()}:
            path = f if os.path.isabs(f) else os.path.join(d, f)
            try:
                n = os.path.getsize(path)
            except OSError:
                continue
            if n > size:
                best, size = path, n
        return best

    # -- the Route: the drive under the model's files, probed once, and the queue every read goes through --
    # priorities: a layer waiting now first; the lookahead's reads by distance (depth d at DISK_AHEAD + d - 1);
    # a prefill's experts last, served in offset order, which makes them a sweep
    DISK_DEMAND = 0
    DISK_AHEAD = 1
    DISK_SWEEP = 3

    def _disk_state(self) -> dict[str, Any]:
        st = getattr(self, "_disk", None)
        if st is None:
            st = self._disk = {
                "profiles": {},
                "queue": [],
                "by_key": {},
                "reqs": {},
                "seq": 0,
                "workers": [],
                "depth": 0,
                "merge": False,
                "cv": threading.Condition(),
                "stop": False,
                "inflight": 0,
                "inflight_ahead": 0,
                "inflight_demand": 0,
                "ahead_cap": 2,
                # the drive as it is now: the last reads' spans, the rate they make over the drive's busy
                # time, the probe's rate at this depth, and whether the one has fallen under the other
                "live": [],
                "live_gbs": 0.0,
                "expect_gbs": 0.0,
                "slow": False,
                "pulse": 0,
            }
        return st

    @staticmethod
    def _volume(path: str) -> Any:
        if sys.platform == "win32":
            return os.path.splitdrive(os.path.abspath(path))[0].upper() or os.path.abspath(path)
        return os.stat(path).st_dev

    DISK_DEPTHS = (1, 4, 16)

    @staticmethod
    def _disk_rule(p: dict[str, Any]) -> dict[str, Any]:
        """What the profile decides. Readers in flight: the deepest of 1, 4, 16 whose burst rate is not below the
        best shallower one's by more than the measurements' spread; a tie goes deeper (a reader that loses its
        core leaves a short queue idle). Adjacent reads merged when a read's fixed cost is four times an expert
        span's bounce copy. Predictions capped at half the readers and 50 ms of drive time."""
        rates = p.get("rates") or {}
        if 1 not in rates:
            return {"depth": 4, "merge": False, "ahead": 2}
        floor = 0.05 if int(p.get("reps", 1)) >= 3 else 0.10
        depth = 1
        best, spread = rates[1]
        for d in BatchScheduler.DISK_DEPTHS[1:]:
            if d not in rates:
                break
            m1, s1 = rates[d]
            if best - m1 > max(spread, s1, floor * best):
                break
            depth = d
            if m1 > best:
                best, spread = m1, s1
        copy = float(p.get("copy_ms", 0.0))
        merge = copy > 0 and float(p.get("fixed_ms", 0.0)) > 4 * copy
        single_s = float(p.get("single_ms", 0.0)) / 1e3
        ahead = max(1, depth // 2)
        if single_s > 0:
            # none at all on a drive whose one read exceeds the window: a prediction there is not a gap
            # filled but a whole read taken from the layer waiting now, and it cannot be recalled in flight
            ahead = min(ahead, int(0.05 / single_s))
        return {"depth": depth, "merge": merge, "ahead": ahead}

    def disk(self, path: str) -> dict[str, Any]:
        """The profile of the drive under `path`, taken once per volume through the Route's own reader on
        that file and nothing else: the fixed cost of a read (4 KB), an expert-sized read alone, bursts of
        expert-sized reads at random offsets over one, four and sixteen readers, a copy of an expert span in
        RAM, the sequential rate; and what they decide (`_disk_rule`). Without the reader, or on a file too
        small to measure on, the defaults: four in flight, no merging."""
        st = self._disk_state()
        vol = self._volume(path)
        p = st["profiles"].get(vol)
        if p is None:
            reused = DriveBenchmark.kept(vol) is not None
            p = dict(DriveBenchmark.take(vol, path, clock=self._disk_clock).profile)
            forced = int(os.environ.get("BTB_ROUTE_DEPTH", "0") or 0)
            if forced > 0:
                p["depth"] = forced
                p["ahead"] = min(int(p["ahead"]), max(1, forced // 2))
            st["profiles"][vol] = p
            # the queue keeps one reader for the prediction class even where the rule allows no predictions: the
            # cold ring reads in that class, and the store itself withholds predictions. The reader count is
            # fixed once the readers run (the cold ring may have started them); the rest of the rule applies
            if not st["workers"]:
                st["depth"] = int(p["depth"])
            elif int(st["depth"]) != int(p["depth"]):
                self.sm.log(f"[disk] {vol}: {st['depth']} readers already running; the probe asked for {p['depth']}")
            st["merge"] = bool(p["merge"])
            st["ahead_cap"] = max(1, int(p["ahead"]))
            rates = p.get("rates") or {}
            st["expect_gbs"] = float(rates.get(int(st["depth"]), (0.0, 0.0))[0]) if rates else 0.0
            how = (" (forced)" if forced > 0 else "") + (" (the plan's measurement)" if reused else "")
            if p["measured"]:
                bursts = ", ".join(f"{m:.2f}+-{s:.2f} over {k}" for k, (m, s) in sorted(p["rates"].items()))
                self.sm.log(
                    f"[disk] {vol}: a read costs {p['fixed_ms']:.2f} ms; {p['big_mb']:.1f} MB alone {p['single_ms']:.2f} ms "
                    f"({p['single_gbs']:.2f} GB/s), bursts {bursts} readers (GB/s, {p['reps']} each), "
                    f"{p['seq_gbs']:.2f} sequential, a span copies in {p['copy_ms']:.2f} ms -> {p['depth']} in flight{how}, "
                    f"{p['ahead']} ahead, adjacent reads {'merged' if p['merge'] else 'separate'} "
                    f"({p['cost_s']:.1f} s to measure)"
                )
            else:
                self.sm.log(f"[disk] {vol}: not measured; {p['depth']} in flight{how}")
        return p

    @staticmethod
    def measure_drive(path: str, clock: Any = None) -> dict[str, Any]:
        """The drive under `path` probed the way the Route reads it: through kept handles, one a reader,
        into 4 KB-aligned buffers the drive writes itself. Each depth's burst is timed from the moment its
        readers are released (they are started first, and each takes the next offset as the Route's do); a
        fast drive gets three bursts of 32 reads a depth, a slow one (an expert-sized read over 20 ms) two of
        16, and the median and spread of each are kept for the rule. `clock` is the drive's: the wall's, or a
        simulated drive's (tests/drive_sim.py), which is how a disk is probed on a machine without one."""
        from .native import Native

        out: dict[str, Any] = {
            "measured": False,
            "fixed_ms": 0.0,
            "single_ms": 0.0,
            "single_gbs": 0.0,
            "copy_ms": 0.0,
            "rates": {},
            "reps": 0,
            "seq_gbs": 0.0,
            "big_mb": 0.0,
            "cost_s": 0.0,
        }
        try:
            size = os.path.getsize(path)
        except OSError:
            return out
        if Native.read_direct is None or size < (64 << 20):
            return out
        t_start = time.perf_counter()
        rng = random.Random(1)
        big = int(min(6553600, size // 16) // 4096 * 4096)
        kept = Native.open is not None and Native.read_at is not None and Native.close is not None

        def at(n: int) -> int:
            return rng.randrange(0, (size - n) // 4096) * 4096

        def aligned(n: int) -> torch.Tensor:
            raw = torch.empty(n + 4096, dtype=torch.uint8)
            a = (-raw.data_ptr()) % 4096
            return raw[a : a + n]

        class Reader:
            def __init__(self) -> None:
                self.h = Native.open(path) if kept else None

            def read(self, off: int, n: int, dst: torch.Tensor) -> None:
                if self.h is None:
                    Native.read_direct(path, off, n, dst, 0)
                else:
                    Native.read_at(self.h, off, n, dst, 0, 0)

            def close(self) -> None:
                if self.h is not None:
                    Native.close(self.h)

        clock = clock or time.perf_counter

        def timed(f: Any) -> float:
            t0 = clock()
            f()
            return clock() - t0

        rd0 = Reader()
        small, buf = aligned(4096), aligned(big)
        fixed = sorted(timed(lambda: rd0.read(at(4096), 4096, small)) for _ in range(8))[4]
        single = sorted(timed(lambda: rd0.read(at(big), big, buf)) for _ in range(6))[3]
        rd0.close()
        slow = single > 0.02
        reps, burst = (2, 16) if slow else (3, 32)

        def rate(k: int) -> float:
            readers = [Reader() for _ in range(k)]
            bufs = [aligned(big) for _ in range(k)]
            offs = iter([at(big) for _ in range(burst)])
            lock, go = threading.Lock(), threading.Event()

            def work(i: int) -> None:
                go.wait()
                while True:
                    with lock:
                        o = next(offs, None)
                    if o is None:
                        return
                    readers[i].read(o, big, bufs[i])

            th = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(k)]
            for x in th:
                x.start()
            t0 = clock()
            go.set()
            for x in th:
                x.join()
            dt = clock() - t0
            for r in readers:
                r.close()
            return burst * big / dt / 1e9

        rates: dict[int, tuple[float, float]] = {}
        for k in BatchScheduler.DISK_DEPTHS:
            rs = sorted(rate(k) for _ in range(reps))
            rates[k] = (rs[len(rs) // 2], rs[-1] - rs[0])
        n_seq = int(min((32 if slow else 128) << 20, size // 4) // 4096 * 4096)
        sb = aligned(n_seq)
        rd1 = Reader()
        o = at(n_seq)
        seq = n_seq / timed(lambda: rd1.read(o, n_seq, sb)) / 1e9
        rd1.close()
        # the copy a merged read costs, priced on the sequential buffer and scaled to a span. RAM's time is the
        # wall's whatever clock the drive keeps. warm first, then take the fastest of a few: a fresh buffer's
        # first copy pays one-time page faults, not the steady copy rate, and on a slow or loaded box that
        # artifact can dwarf the copy and flip the merge decision
        other = torch.empty_like(sb)
        other.copy_(sb)
        best = float("inf")
        for _ in range(3):
            t_copy = time.perf_counter()
            other.copy_(sb)
            best = min(best, time.perf_counter() - t_copy)
        copy = best * big / n_seq
        out.update(
            measured=True,
            fixed_ms=fixed * 1e3,
            single_ms=single * 1e3,
            single_gbs=big / single / 1e9,
            copy_ms=copy * 1e3,
            rates=rates,
            reps=reps,
            seq_gbs=seq,
            big_mb=big / 2**20,
            cost_s=time.perf_counter() - t_start,
        )
        return out

    def disk_read(
        self,
        path: str,
        off: int,
        n: int,
        dst: torch.Tensor,
        priority: int = 0,
        key: Any = None,
        on_done: Any = None,
        chunk: int = 0,
        depth: int = 0,
    ) -> Future[float]:
        """Queue one read of `n` bytes at `off` of `path` into `dst` and return its future (the read's seconds
        as the result). A read already queued or in flight for the same bytes is not queued twice: its
        future is returned, its priority raised if the new request's is higher, its `on_done` added.
        `key` names the requester (a layer and expert) so `disk_drop` can withdraw its reads before they are
        issued; `on_done(dur_ns)` runs on the reader thread as the bytes land. `depth` is the reader's own
        parallelism inside the one read (1 for a request that is one of many, the default for a lone big
        one)."""
        st = self._disk_state()
        # the same bytes into the same place: two experts whose padded spans share a sector are two reads
        k = (path, int(off), int(n), int(dst.data_ptr()))
        with st["cv"]:
            r = st["by_key"].get(k)
            if r is not None and r["state"] in ("queued", "inflight"):
                if on_done is not None:
                    r["on_done"].append(on_done)
                if priority < r["priority"] and r["state"] == "queued":
                    r["priority"] = int(priority)
                    r["ver"] += 1
                    heapq.heappush(st["queue"], (r["priority"], path, int(off), r["seq"], r["ver"]))
                return r["future"]
            st["seq"] += 1
            r = {
                "path": path,
                "off": int(off),
                "n": int(n),
                "dst": dst,
                "priority": int(priority),
                "seq": st["seq"],
                "ver": 0,
                "key": key,
                "chunk": int(chunk),
                "depth": int(depth),
                "on_done": [on_done] if on_done is not None else [],
                "future": Future(),
                "state": "queued",
            }
            st["reqs"][r["seq"]] = r
            st["by_key"][k] = r
            heapq.heappush(st["queue"], (r["priority"], path, int(off), r["seq"], 0))
            self._disk_start(st)
            st["cv"].notify()
        return r["future"]

    def disk_drop(self, key: Any) -> int:
        """Withdraw every queued read of `key` (a prediction that lapsed): their futures are cancelled; a read
        already in flight completes. Returns how many were withdrawn."""
        st = self._disk_state()
        n = 0
        with st["cv"]:
            for r in list(st["reqs"].values()):
                if r["key"] == key and r["state"] == "queued":
                    r["state"] = "dropped"
                    del st["reqs"][r["seq"]]
                    kk = (r["path"], r["off"], r["n"], int(r["dst"].data_ptr()))
                    if st["by_key"].get(kk) is r:
                        del st["by_key"][kk]
                    r["future"].cancel()
                    n += 1
        return n

    def _disk_start(self, st: dict[str, Any]) -> None:
        st["workers"] = [t for t in st["workers"] if not hasattr(t, "is_alive") or t.is_alive()]
        if not st["workers"]:
            st["stop"] = False  # the last reader of a closed queue is gone
        if st["workers"] or st["stop"]:
            return
        depth = int(st["depth"]) or 4
        for i in range(depth):
            t = threading.Thread(target=self._disk_worker, args=(st,), daemon=True, name=f"route-{i}")
            t.start()
            st["workers"].append(t)

    @staticmethod
    def _is_ahead(priority: int) -> bool:
        return BatchScheduler.DISK_AHEAD <= int(priority) < BatchScheduler.DISK_SWEEP

    def _disk_take(self, st: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        """under the lock: the next read to issue, highest priority then by path and offset, with the queued
        reads adjacent to it in the same file and class when the drive merges; None when nothing can be issued.
        Within a class the order is the file's, so on a drive that seeks a deep queue is safe: the drive's own
        reordering does the rest, and a queue issued in call order would cost it half again (the model in
        docs/route-hdd-plan.md, 7.2 against 11.7 s a token on a queue that does not reorder). The lookahead's
        reads hold at most half the readers, so a layer waiting now always finds one free"""
        deferred = []
        try:
            while st["queue"]:
                pri, path, _off, seq, ver = heapq.heappop(st["queue"])
                r = st["reqs"].get(seq)
                if r is None or r["state"] != "queued" or r["ver"] != ver:
                    continue
                if self._is_ahead(pri) and (st["inflight_ahead"] >= st["ahead_cap"] or st["inflight_demand"] > 0):
                    # the lookahead takes the gaps between the layers' bursts, never a share of a burst: a
                    # prediction issued beside a demand read slows the layer waiting now by the bytes it takes
                    deferred.append((pri, path, _off, seq, ver))
                    continue
                r["state"] = "inflight"
                st["inflight"] += 1
                if self._is_ahead(pri):
                    st["inflight_ahead"] += 1
                elif pri < self.DISK_AHEAD:
                    st["inflight_demand"] += 1
                return self._disk_partner(st, r, pri, path)
            return None
        finally:
            for entry in deferred:
                heapq.heappush(st["queue"], entry)

    DISK_MERGE_MAX = 64 << 20

    def _disk_partner(
        self, st: dict[str, Any], r: dict[str, Any], pri: int, path: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """the read taken, with the queued reads of the same file and class that each start inside or at the
        end of the run so far and reach past it, taken in turn while the merged span stays under
        DISK_MERGE_MAX, when the drive merges: two padded spans share their boundary sector, and a run of
        adjacent experts (a prefill's, a quarter of its misses on the 180B) is one seek instead of one each"""
        partners: list[dict[str, Any]] = []
        if st["merge"]:
            end = r["off"] + r["n"]
            while st["queue"]:
                p2, path2, off2, seq2, ver2 = st["queue"][0]
                r2 = st["reqs"].get(seq2)
                if (
                    r2 is None
                    or r2["state"] != "queued"
                    or r2["ver"] != ver2
                    or path2 != path
                    or p2 != pri
                    or not (r["off"] < off2 <= end < off2 + r2["n"])
                    or off2 + r2["n"] - r["off"] > self.DISK_MERGE_MAX
                ):
                    break
                heapq.heappop(st["queue"])
                r2["state"] = "inflight"
                st["inflight"] += 1
                if self._is_ahead(p2):
                    st["inflight_ahead"] += 1
                elif p2 < self.DISK_AHEAD:
                    st["inflight_demand"] += 1
                partners.append(r2)
                end = off2 + r2["n"]
        return r, partners

    def _disk_worker(self, st: dict[str, Any]) -> None:
        from .native import Native

        # this reader's own handle per file, share-read, held for the run: an open handle halves a read's cost,
        # and it is one handle a thread because a synchronous handle serializes the reads that share it
        handles: dict[str, int] = {}

        def read(path: str, off: int, n: int, dst: torch.Tensor, chunk: int, depth: int) -> None:
            if Native.read_at is None or Native.open is None:
                Native.read_direct(path, off, n, dst, chunk)
                return
            h = handles.get(path)
            if h is None:
                h = handles[path] = Native.open(path)
            Native.read_at(h, off, n, dst, chunk, depth)

        try:
            while True:
                with st["cv"]:
                    taken = None
                    while not st["stop"]:
                        taken = self._disk_take(st) if st["queue"] else None
                        if taken is not None:
                            break
                        # nothing issuable: the queue is empty, or holds only lookahead reads past their cap
                        st["cv"].wait(timeout=0.05)
                    if st["stop"]:
                        return
                self._disk_serve(st, taken, read)
        finally:
            if Native.close is not None:
                for h in handles.values():
                    Native.close(h)

    def _disk_serve(self, st: dict[str, Any], taken: Any, read: Any) -> None:
        """one read (or a merged run) issued and finished: the destinations written, the callbacks run, the
        futures settled"""
        r, partners = taken
        reqs = [r, *partners]
        err: BaseException | None = None
        t0 = time.perf_counter_ns()
        try:
            if not partners:
                read(r["path"], r["off"], r["n"], r["dst"], r["chunk"], r["depth"])
            else:
                span = partners[-1]["off"] + partners[-1]["n"] - r["off"]
                run = torch.empty(span, dtype=torch.uint8)
                read(r["path"], r["off"], span, run, r["chunk"], r["depth"])
                for q in reqs:
                    at = q["off"] - r["off"]
                    q["dst"].copy_(run[at : at + q["n"]])
        except BaseException as e:
            err = e
        t1 = time.perf_counter_ns()
        dur = t1 - t0
        turned = None
        with st["cv"]:
            for q in reqs:
                q["state"] = "done"
                st["inflight"] -= 1
                if self._is_ahead(q["priority"]):
                    st["inflight_ahead"] -= 1
                elif q["priority"] < self.DISK_AHEAD:
                    st["inflight_demand"] -= 1
                st["reqs"].pop(q["seq"], None)
                kk = (q["path"], q["off"], q["n"], int(q["dst"].data_ptr()))
                if st["by_key"].get(kk) is q:
                    del st["by_key"][kk]
            if err is None:
                st["live"].append((t0, t1, sum(q["n"] for q in reqs)))
                if len(st["live"]) > self.DISK_LIVE:
                    del st["live"][: len(st["live"]) - self.DISK_LIVE]
                st["pulse"] += 1
                if st["pulse"] % 16 == 0:
                    turned = self._disk_pulse(st)
            # a reader held back by the lookahead's cap may go now
            st["cv"].notify_all()
        if turned is not None:
            self._disk_turned(st, turned)
        for q in reqs:
            for cb in q["on_done"]:
                try:
                    cb(dur)
                except Exception as e:  # a callback's failure must not end the reader or strand the futures
                    self.sm.log(f"[disk] a read's on_done raised: {e!r}")
            if err is not None:
                q["future"].set_exception(err)
            else:
                q["future"].set_result(dur / 1e9)

    DISK_LIVE = 64

    def _disk_pulse(self, st: dict[str, Any]) -> tuple[str, float] | None:
        """under the lock, every sixteenth read: the last reads' bytes over the drive's busy time (the union
        of their spans, so the idle between a decode's bursts is not counted against it) against the rate the
        probe's rate at this depth. Slow under half of it, recovered above four fifths; the turn returned"""
        live = st["live"]
        if len(live) < 32:
            return None
        spans = sorted(live)
        busy = 0
        a, b = spans[0][0], spans[0][1]
        for s0, s1, _n in spans[1:]:
            if s0 > b:
                busy += b - a
                a, b = s0, s1
            else:
                b = max(b, s1)
        busy += b - a
        if busy <= 0:
            return None
        gbs = sum(n for _s0, _s1, n in spans) / (busy / 1e9) / 1e9
        st["live_gbs"] = gbs
        if st["expect_gbs"] <= 0:
            return None
        if not st["slow"] and gbs < 0.5 * st["expect_gbs"]:
            st["slow"] = True
            return ("slowed", gbs)
        if st["slow"] and gbs > 0.8 * st["expect_gbs"]:
            st["slow"] = False
            return ("recovered", gbs)
        return None

    def _disk_turned(self, st: dict[str, Any], turn: tuple[str, float]) -> None:
        what, gbs = turn
        self.sm.log(
            f"[disk] the drive {what}: {gbs:.2f} GB/s over the last {len(st['live'])} reads against "
            f"the probe's {st['expect_gbs']:.2f} at {st['depth']} in flight"
            + ("; predictions withheld" if what == "slowed" else "")
        )
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None:
            prof.add(prof.DRIVE, dur_ns=int(gbs * 1e9), aux=1 if what == "slowed" else 0)

    def disk_slow(self) -> bool:
        """whether the drive is delivering under half the probe's rate, over its last reads"""
        st = getattr(self, "_disk", None)
        return bool(st and st["slow"])

    def disk_live(self) -> tuple[float, float]:
        """(the drive's rate over its last reads, the probe's rate at this depth), in GB/s"""
        st = getattr(self, "_disk", None)
        return (float(st["live_gbs"]), float(st["expect_gbs"])) if st else (0.0, 0.0)

    def disk_close(self) -> None:
        """Stop the readers: the queue is emptied (its futures cancelled), the threads joined."""
        st = getattr(self, "_disk", None)
        if st is None:
            return
        with st["cv"]:
            st["stop"] = True
            for r in st["reqs"].values():
                if r["state"] == "queued":
                    r["state"] = "dropped"
                    r["future"].cancel()
            st["queue"].clear()
            st["cv"].notify_all()
        for t in st["workers"]:
            t.join(timeout=30)
        alive = [t for t in st["workers"] if t.is_alive()]
        if alive:  # a reader deep in a read outlives the join: it keeps `stop` and ends on its own
            self.sm.log(f"[disk] {len(alive)} reader(s) still in a read at close")
        st["workers"] = alive
        st["stop"] = bool(alive)
