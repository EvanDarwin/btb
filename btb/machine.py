# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb bench`'s machine report: what this machine is and what it does. First the facts as the OS and the drivers
give them (`sysinfo`): the CPU, its cores and the features its kernels can use, the memory and its modules, each
card and its link, the drives. Then each op btb runs, swept across the sizes it runs at, where it runs: the host's
memory from L1 to DRAM and its native kernels by weight size, rows and context; the card's memory, kernels and bus
by the same; Apple's GPU; and the drive under the models read the way btb reads it, by read size and depth. A sweep
shows where a cache ends, where a call's fixed cost stops mattering, and where an op turns from bandwidth to compute.

Each point is sampled until its median's 95% confidence interval is within `ci` of the median, or for `budget_s`,
whichever comes first; one whose time ran out is reported as it stands, marked. The interval is distribution-free
(the order statistics around the median), so a noisy machine's stalls widen it rather than drag the median, and a
quiet machine settles a point in well under a second. Streamed operands rotate through pools sized past the
machine's own L3 and L2, so a call reads memory as a decode step's layers do; held ones are the same each call."""

from __future__ import annotations

import bisect
import contextlib
import ctypes
import functools
import gc
import math
import os
import platform
import random
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import torch
import torch.nn.functional as F

from . import sysinfo
from .feedback import btb_version
from .kinds import Json

Z95 = 1.959964  # the normal quantile of a two-sided 95% interval
KB, MB, GB = 1 << 10, 1 << 20, 1 << 30
SECTIONS = ("memory", "cpu", "card", "bus", "mlx", "drive")

# the shapes the ops are timed at: a 4B dense model's (Qwen3-4B), a bf16 mixture's experts, gpt-oss's MXFP4 experts
LINEAR = (9728, 2560)  # an MLP projection, [out, in]
SHORT = (1024, 3072)  # a short weight: Qwen3-0.6B's down projection, too few rows to fill a card at 16 a block
MLP = (2560, 9728)  # hidden, intermediate: gate and up as one [2I, H] weight
EXPERT = (1280, 2560)  # a bf16 mixture's expert gate_up (640 wide)
MX4 = (5760, 2880)  # gpt-oss's expert gate_up as stored
HEADS = (32, 8, 128)  # the attention's query heads, KV heads and head dim
DELTA = (16, 32, 128, 128, 4)  # Qwen3.5's linear attention: key heads, value heads, key dim, value dim, conv taps
# the sweeps: working sets from an L1 to past any cache, square weights from L2-sized to past an L3, the contexts an
# attention reads over, the vocabularies a pick runs over, the transfers a bus carries and the reads a drive serves
HOST_SETS = (32 * KB, 256 * KB, 1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, 1 * GB)
CARD_SETS = (1 * MB, 4 * MB, 16 * MB, 64 * MB, 256 * MB, 1 * GB)
SQUARES = (512, 1024, 2048, 4096, 8192)
CONTEXTS = (512, 2048, 8192, 32768, 131072)
VOCABS = (32000, 151936, 262144)
TRANSFERS = (4 * KB, 64 * KB, 1 * MB, 16 * MB, 256 * MB)
READS = (4 * KB, 64 * KB, 1 * MB, 32 * MB, 128 * MB)
# PCIe's rate a lane, each direction, GB/s: 8b/10b coding to gen 2, 128b/130b from gen 3
PCIE_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}
CHUNK = 32 * MB  # a drive write: flushed one at a time
DRIVE_KEEP = 4 * GB  # what the drive keeps free of the bench's file


def median_ci(xs: Sequence[float]) -> tuple[float, float, float]:
    """(median, low, high): the sample's median and its 95% confidence interval, distribution-free - the order
    statistics at ranks n/2 -+ 1.96 sqrt(n)/2 (the binomial's normal approximation), clamped to the sample. A
    sorted sample sorts again in one pass"""
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    d = Z95 * math.sqrt(n) / 2
    return med, s[max(0, math.floor(n / 2 - d) - 1)], s[min(n - 1, math.ceil(n / 2 + d))]


@dataclass(frozen=True)
class Timing:
    """one op's time: the median seconds a call and its 95% interval, from `n` samples of `calls` calls each;
    `settled` when the interval closed within the target before the time ran out"""

    median: float
    lo: float
    hi: float
    n: int
    calls: int
    settled: bool

    @property
    def half(self) -> float:
        """the interval's half-width relative to the median"""
        return (self.hi - self.lo) / 2 / self.median if self.median > 0 else math.inf

    def per(self, k: int) -> Timing:
        """the time of one of the `k` launches a call made"""
        return replace(self, median=self.median / k, lo=self.lo / k, hi=self.hi / k)


class Clock:
    """the host's clock over a stretch of work"""

    t0 = 0.0

    def start(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        return time.perf_counter() - self.t0


class CardClock(Clock):
    """the card's clock: events on the current stream around the work, so a kernel's time leaves the host's
    launches and its wait out"""

    def __init__(self) -> None:
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.a.record()

    def stop(self) -> float:
        self.b.record()
        self.b.synchronize()
        return float(self.a.elapsed_time(self.b)) / 1e3


@dataclass(frozen=True)
class Sampler:
    """How an op is timed: samples of `calls` calls each - as many as make a sample `sample_s` long, past the
    clock's resolution and a call's own overhead - taken until the median's 95% interval is within `ci` of it (at
    least `min_n` samples), or `budget_s` has passed, or `max_n` samples, whichever comes first"""

    ci: float = 0.02
    budget_s: float = 15.0
    min_n: int = 10
    max_n: int = 1_000_000
    sample_s: float = 0.002

    def time(self, call: Callable[[], object], clock: Clock | None = None) -> Timing:
        clock = clock or Clock()
        end = time.perf_counter() + self.budget_s
        clock.start()
        call()  # the first call's one-off costs (a kernel's load, pages first touched) out of the samples
        clock.stop()
        clock.start()
        call()
        one = max(clock.stop(), 1e-9)
        calls = max(1, min(100_000, math.ceil(self.sample_s / one)))
        xs: list[float] = []  # kept sorted as they land: a fast op takes thousands in its budget
        while True:
            clock.start()
            for _ in range(calls):
                call()
            bisect.insort(xs, clock.stop() / calls)
            if len(xs) < self.min_n:
                continue
            med, lo, hi = median_ci(xs)
            settled = hi - lo <= 2 * self.ci * med
            if settled or len(xs) >= self.max_n or time.perf_counter() >= end:
                return Timing(med, lo, hi, len(xs), calls, settled)


def _dur(s: float) -> str:
    return f"{s * 1e6:.1f} us" if s < 1e-3 else f"{s * 1e3:.2f} ms" if s < 1 else f"{s:.2f} s"


def _size(b: float) -> str:
    return f"{b / GB:.1f} GB" if b >= GB else f"{b / MB:.0f} MB" if b >= MB else f"{b / KB:.0f} KB"


def _rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


@dataclass
class Row:
    """one line of the report: an op at a shape and its timing (None where it did not run: `note` says why), with
    what a call moves and computes - its rates - the peak its bytes are held to, and where it sits in its sweep
    (`at`: the swept quantities, for a reader to plot)"""

    section: str
    op: str
    shape: str
    t: Timing | None
    nbytes: float = 0.0
    flops: float = 0.0
    peak: float = 0.0  # bytes a second
    note: str = ""
    at: Json = field(default_factory=dict)

    def json(self) -> Json:
        out: Json = {"section": self.section, "op": self.op, "shape": self.shape, "at": self.at}
        t = self.t
        if t is not None:
            out.update(median_s=t.median, lo_s=t.lo, hi_s=t.hi, n=t.n, calls=t.calls, settled=t.settled)
            if self.nbytes:
                out["gb_s"] = self.nbytes / t.median / 1e9
                if self.peak:
                    out["of_peak"] = self.nbytes / t.median / self.peak
            if self.flops:
                out["gflop_s"] = self.flops / t.median / 1e9
        if self.note:
            out["note"] = self.note
        return out

    def line(self, nested: bool = False) -> str:
        """the row as the report prints it; `nested` under its sweep's heading, by its shape alone"""
        label = f"  {self.shape}" if nested else f"{self.op}, {self.shape}" if self.shape else self.op
        t = self.t
        if t is None:
            return f"  {label:<66} {'-':>9}          {self.note}"
        rates = []
        if self.nbytes:
            bps = self.nbytes / t.median
            rate = f"{bps / 1e9:7.1f} GB/s" if bps >= 1e9 else f"{bps / 1e6:7.1f} MB/s"
            rates.append(rate + (f" ({bps / self.peak:4.0%} of peak)" if self.peak else ""))
        if self.flops:
            f = self.flops / t.median
            rates.append(f"{f / 1e12:6.2f} TFLOP/s" if f >= 1e12 else f"{f / 1e9:6.1f} GFLOP/s")
        if self.note:
            rates.append(self.note)
        mark = " " if t.settled else "~"
        return f"  {label:<66} {_dur(t.median):>9} ±{t.half * 100:4.1f}%{mark}  " + "  ".join(rates)


class Report:
    """the timings as they are taken, each said as it lands - a sweep's under its heading - and what was found
    from them (`found`)"""

    def __init__(self, sampler: Sampler, say: Callable[[str], object]) -> None:
        self.sampler, self.say = sampler, say
        self.rows: list[Row] = []
        self.found: Json = {}
        self.where = ""
        self.head = ""

    def section(self, name: str, detail: str = "") -> None:
        self.where, self.head = name, ""
        self.say("")
        self.say(name + (f": {detail}" if detail else ""))

    def sweep(self, op: str, by: str) -> str:
        """a sweep's heading: the rows timed as `op` from here on print by their shape beneath it"""
        self.head = op
        self.say(f"  {op}, by {by}")
        return op

    def add(self, row: Row) -> None:
        self.rows.append(row)
        self.say(row.line(nested=row.op == self.head))

    def skip(self, op: str, shape: str, why: str) -> None:
        self.add(Row(self.where, op, shape, None, note=why))

    def time(
        self,
        op: str,
        shape: str,
        call: Callable[[], object],
        *,
        clock: Clock | None = None,
        per: int = 1,
        nbytes: float = 0.0,
        flops: float = 0.0,
        peak: float = 0.0,
        note: str = "",
        at: Json | None = None,
        **how: Any,
    ) -> Timing | None:
        """`call` timed by the sampler (`how` overriding its settings), as one of the `per` launches it makes; a
        failure is a row saying so, never the report's end"""
        try:
            t = (replace(self.sampler, **how) if how else self.sampler).time(call, clock).per(per)
        except Exception as e:
            self.skip(op, shape, f"failed: {type(e).__name__}: {e}")
            return None
        self.add(Row(self.where, op, shape, t, nbytes, flops, peak, note, at or {}))
        return t

    def card(self, op: str, shape: str, body: Callable[[], object], per: int, **kw: Any) -> Timing | None:
        """`body` - `per` launches on the card - captured once as a graph and timed replaying it on the card's
        clock, as a step replays its layers: the host's launches stay out of a kernel's time"""
        try:
            body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
        except Exception as e:
            self.skip(op, shape, f"failed: {type(e).__name__}: {e}")
            return None
        return self.time(op, shape, g.replay, clock=CardClock(), per=per, **kw)


class Pool:
    """copies of an operand, enough to pass `nbytes` (`_pool`), handed out in turn: a call reads its operand from
    memory, never from a cache the last call filled. The one pool of the machine report and of bench/e2e_bench.py"""

    def __init__(self, make: Callable[[], Any], each: int, nbytes: int, cap: int = 64) -> None:
        self.items, self.i = _pool(make, each, nbytes, cap), 0

    @classmethod
    def of(cls, items: list[Any]) -> Pool:
        """a pool of `items` as they are"""
        p = cls.__new__(cls)
        p.items, p.i = items, 0
        return p

    @classmethod
    def one(cls, item: Any) -> Pool:
        """one operand handed out every call: the small weights a cache holds whatever the bench does"""
        return cls.of([item])

    def next(self) -> Any:
        self.i = (self.i + 1) % len(self.items)
        return self.items[self.i]

    __call__ = next

    @property
    def first(self) -> Any:
        return self.items[0]


def _pool(make: Callable[[], Any], each: int, past: int, cap: int = 64) -> list[Any]:
    """copies of an operand, enough to pass `past` bytes (two at least): a call takes the next, so it reads memory
    rather than what the last call left in a cache"""
    return [make() for _ in range(max(2, min(cap, -(-past // max(1, each)))))]


def _bf16(*shape: int, device: Any = None) -> torch.Tensor:
    """bf16 values of a weight's scale; never subnormal, which some CPUs take a slow path for"""
    return torch.empty(*shape, dtype=torch.bfloat16, device=device).uniform_(-1, 1)


def host_free() -> int:
    """the RAM available now, as the OS counts it: what the bench sizes its own operands by"""
    # free-read: the machine report runs with no model and no ledger; it sizes its own operands, nothing it places
    return int(sysinfo.host_free_bytes())


def card_free(dev: torch.device) -> int:
    """the card's free memory now, as the driver counts it: what the bench sizes its own operands by"""
    # free-read: the machine report runs with no model and no ledger; it sizes its own operands, nothing it places
    return int(torch.cuda.mem_get_info(dev)[0])


def host_past() -> int:
    """the bytes a host pool passes: four of the L3 (the L2 where there is none), 64 MB at least, held to 1/16 of
    the RAM available so the bench never presses on the machine"""
    c = sysinfo.host_cache_sizes()
    want = max(4 * (c["host_l3"] or c["host_l2"]), 64 * MB)
    return int(max(16 * MB, min(want, host_free() // 16)))


@contextlib.contextmanager
def _torch_threads(n: int) -> Iterator[None]:
    was = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(was)


def _let_go() -> None:
    """one op's operands given back before the next: the host's, the card's cached blocks, the pinned pool's"""
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        host_empty = getattr(torch._C, "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()


def _reread(x: torch.Tensor, n: int, at_least: int) -> torch.Tensor:
    """the first `n` bytes of float32 `x` as a view that reads them over and over, at least `at_least` bytes a
    pass: a stride-0 row of them, so a set smaller than a cache stays in it and is read from it"""
    k = n // 4
    return x[:k].unsqueeze(0).expand(max(1, at_least // n), k)


# -- the facts ----------------------------------------------------------------------------------------------------


def default_drive() -> str:
    """the folder models are read from: the Hugging Face cache (HF_HUB_CACHE), or the nearest folder above it that
    exists before a first download makes it"""
    from huggingface_hub import constants

    d = os.path.abspath(constants.HF_HUB_CACHE)
    while not os.path.isdir(d) and os.path.dirname(d) != d:
        d = os.path.dirname(d)
    return d


def _memory_peak(mods: list[Json] | None, channels: int | None) -> float | None:
    """the modules' peak rate in GB/s: each channel's transfer rate times its data width; None where the channels
    are not named"""
    if not mods or not channels:
        return None
    mts = min(m["mts"] or m["rated_mts"] for m in mods)
    width = max(m["width"] or 64 for m in mods)
    return channels * mts * 1e6 * width / 8 / 1e9 if mts else None


def _card_facts(i: int) -> Json:
    from .engine.device import pci_bus_id

    p: Any = torch.cuda.get_device_properties(i)  # its PCI ids and L2 are there, not in torch's stubs
    bus: str | None = None
    with contextlib.suppress(Exception):  # a build whose properties carry no PCI ids (ROCm, an old torch)
        bus = pci_bus_id(i)
    nv = (sysinfo.nvml_card(bus) if bus else None) or {}
    card: Json = {
        "index": i,
        "name": p.name,
        "total": int(p.total_memory),
        "free": sysinfo.nvml_free_bytes(bus) if bus else None,
        "compute": f"{p.major}.{p.minor}",
        "sms": int(p.multi_processor_count),
        "l2": int(getattr(p, "L2_cache_size", 0)),
        "bus_id": bus,
        **nv,
    }
    if nv.get("mem_mhz") and nv.get("bus_bits"):
        # double data rate: the clock NVML gives moves two words a pin (21 Gbps GDDR6X reads 10501 MHz)
        card["peak_gb_s"] = 2 * nv["mem_mhz"] * 1e6 * nv["bus_bits"] / 8 / 1e9
    if nv.get("pcie_max_gen") in PCIE_LANE and nv.get("pcie_max_width"):
        card["pcie_gb_s"] = PCIE_LANE[nv["pcie_max_gen"]] * nv["pcie_max_width"]
    return card


def facts(drives: Sequence[str]) -> Json:
    """the machine as the OS and the drivers describe it, read in a moment: nothing is timed here"""
    from .engine.device import mlx_available
    from .engine.native import Native, cpu_features, isa

    # the library the ops will run on - the install's, or the one --native named - and none where the run is torch's
    lib = Native._gemv_lib if Native.gemv is not None else None
    feats, tier = None, ""
    if lib is not None:
        with contextlib.suppress(Exception):
            feats = cpu_features(lib)
        with contextlib.suppress(Exception):
            tier = isa(lib)
    mods = sysinfo.memory_modules()
    channels = sysinfo.memory_channels(mods) if mods else None
    groups = sysinfo.host_cache_groups()
    out: Json = {
        "system": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "btb": btb_version(),
        },
        "cpu": {
            "name": sysinfo.host_cpu_name(),
            **sysinfo.host_cpu_topology(),
            "caches": sysinfo.cache_sizes(groups),
            "cache_groups": groups,
            "features": feats if feats is not None else [f.lower() for f in sysinfo.host_simd()],
            "features_from": "the native library" if feats is not None else "the OS",
            "native_isa": tier,
            "torch_isa": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(),
        },
        "memory": {
            "total": sysinfo.host_total_bytes(),
            "available": host_free(),
            "modules": mods,
            "channels": channels,
            "peak_gb_s": _memory_peak(mods, channels),
        },
        "cards": [_card_facts(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "drives": [],
    }
    if mlx_available():
        out["mlx"] = {"name": sysinfo.host_cpu_name(), "memory": sysinfo.host_total_bytes()}
    for d in drives:
        try:
            u = shutil.disk_usage(d)
            out["drives"].append({"path": d, "free": u.free, "total": u.total})
        except OSError as e:
            out["drives"].append({"path": d, "error": str(e)})
    return out


def _cpu_lines(c: Json) -> list[str]:
    classes = c.get("classes") or []
    kinds = " (" + ", ".join(f"{k['cores']} {k['name']}" for k in classes) + ")" if len(classes) > 1 else ""
    clock = []
    if c.get("base_mhz"):
        clock.append(f"rated {c['base_mhz'] / 1000:.2f} GHz")
    if c.get("max_mhz"):
        clock.append(f"up to {c['max_mhz'] / 1000:.2f} GHz")
    return [
        f"  cpu      {c['name'] or 'unknown'}: {c.get('physical', '?')} cores{kinds}, {c.get('logical', '?')} threads"
        + (f"; {', '.join(clock)}" if clock else ""),
        f"           features {' '.join(c['features']) or 'none detected'} (from {c['features_from']}); native "
        f"kernels {c['native_isa'] or 'none (no library)'}; torch {c['torch_isa']}, {c['torch_threads']} threads",
        *_cache_lines(c["cache_groups"]),
    ]


def _exact(b: int) -> str:
    """a cache's size as it is: a 1.25 MB L2 not rounded to 1 MB"""
    for unit, name in ((GB, "GB"), (MB, "MB"), (KB, "KB")):
        if b >= unit:
            return f"{b / unit:.2f}".rstrip("0").rstrip(".") + f" {name}"
    return f"{b} B"


def _cache_lines(groups: list[Json]) -> list[str]:
    """each cache level's instances - how many, their size, the threads sharing one - and the level's total"""
    if not groups:
        return ["           caches   not readable here"]
    lines = []
    for lvl, name in ((1, "L1d"), (2, "L2"), (3, "L3")):
        gs = [g for g in groups if g["level"] == lvl]
        parts = [
            (f"{g['count']} x " if g["count"] > 1 else "")
            + f"{_exact(g['size'])} ({g['threads']} thread{'s' if g['threads'] > 1 else ''}"
            + (" each)" if g["count"] > 1 else ")")
            for g in gs
        ]
        total = sum(g["size"] * g["count"] for g in gs)
        tail = f" = {_exact(total)}" if lvl > 1 and sum(g["count"] for g in gs) > 1 else ""
        lines.append(f"           {name:<4}{' + '.join(parts) + tail if gs else 'none listed'}")
    return lines


def _memory_lines(m: Json) -> list[str]:
    head = f"  memory   {_size(m['total'])}, {_size(m['available'])} available"
    mods = m["modules"]
    if not mods:
        return [head + "; the modules are not readable here (macOS lists none; Linux needs root or systemd's udev)"]
    groups: dict[tuple[Any, ...], int] = {}
    for x in mods:
        key = (x["size_gb"], x["kind"], x["form"], x["mts"], x["rated_mts"])
        groups[key] = groups.get(key, 0) + 1
    for (gb, kind, form, mts, rated), n in groups.items():
        head += f"; {n} x {gb:g} GB {kind} {form}" + (f" at {mts} MT/s" if mts else "")
        head += f" (rated {rated})" if rated and rated != mts else ""
    if m["channels"]:
        head += f", {m['channels']} channels"
    if m["peak_gb_s"]:
        head += f": peak {m['peak_gb_s']:.1f} GB/s"
    lines = [head]
    parts = sorted({f"{x['maker']} {x['part']}".strip() for x in mods} - {""})
    slots = ", ".join(x["slot"] for x in mods if x["slot"])
    if parts or slots:
        lines.append(f"           {'; '.join(parts)}" + (f" in {slots}" if slots else ""))
    return lines


def _card_lines(card: Json) -> list[str]:
    clk = f" up to {card['sm_mhz']} MHz" if card.get("sm_mhz") else ""
    free = f", {_size(card['free'])} free" if card.get("free") is not None else ""
    lines = [
        f"  cuda:{card['index']}   {card['name']}: {_size(card['total'])}{free}; compute {card['compute']}, "
        f"{card['sms']} SMs{clk}, L2 {_size(card['l2'])}"
    ]
    more = []
    if card.get("peak_gb_s"):
        more.append(f"memory {card['bus_bits']}-bit at {card['mem_mhz']} MHz: peak {card['peak_gb_s']:.0f} GB/s")
    if card.get("pcie_gen"):
        more.append(
            f"PCIe {card['pcie_gen']}.0 x{card['pcie_width']} now, {card.get('pcie_max_gen', '?')}.0 "
            f"x{card.get('pcie_max_width', '?')} at most"
            + (f": {card['pcie_gb_s']:.1f} GB/s a direction" if card.get("pcie_gb_s") else "")
        )
    if card.get("driver"):
        more.append(f"driver {card['driver']}" + (f", {card['power_w']} W" if card.get("power_w") else ""))
    if more:
        lines.append("           " + "; ".join(more))
    return lines


def facts_lines(f: Json) -> list[str]:
    """the facts as the report prints them"""
    s = f["system"]
    lines = [
        "machine",
        f"  system   {s['os']} ({s['machine']}); Python {s['python']}, torch {s['torch']}"
        + (f" (CUDA {s['cuda']})" if s["cuda"] else "")
        + f", btb {s['btb']}",
        *_cpu_lines(f["cpu"]),
        *_memory_lines(f["memory"]),
    ]
    for card in f["cards"]:
        lines += _card_lines(card)
    if f.get("mlx"):
        lines.append(f"  mlx      {f['mlx']['name']}: {_size(f['mlx']['memory'])} unified memory")
    for d in f["drives"]:
        if "error" in d:
            lines.append(f"  drive    {d['path']}: {d['error']}")
        else:
            lines.append(f"  drive    {d['path']}: {_size(d['free'])} free of {_size(d['total'])}")
    return lines


# -- the host's memory ----------------------------------------------------------------------------------------------


def _thread_counts(topo: Json) -> list[int]:
    """the thread counts the native gemv is streamed at: the powers of two, the performance cores, every core and
    every thread"""
    logical = int(topo.get("logical") or os.cpu_count() or 1)
    ts = {1, logical, int(topo.get("physical") or logical)}
    classes = topo.get("classes") or []
    if len(classes) > 1:
        ts.add(int(classes[0]["cores"]))
    t = 2
    while t < logical:
        ts.add(t)
        t *= 2
    return sorted(x for x in ts if 0 < x <= logical)


def _host_set_sizes(groups: list[Json]) -> list[int]:
    """the read sweep's working sets: HOST_SETS, and half and twice each cache size the OS lists - each cache's
    edge between two points on every machine, a 1.25 MB L2 as well as a 2 MB one"""
    edges = {int(g["size"]) for g in groups}
    return sorted({*HOST_SETS, *(s // 2 for s in edges), *(2 * s for s in edges)})


def _host_sets(rep: Report, peak: float) -> None:
    """torch's read over each working set, every thread and one, and its copy past the caches: each set re-read in
    place, so one that fits a cache is read from it - the rate steps down at each cache's edge. Held to the
    modules' peak only past the L3"""
    groups = sysinfo.host_cache_groups()
    # each thread reads a re-read set whole: it is past the caches once past one L3 (a chiplet's), not their sum
    l3 = max((g["size"] for g in groups if g["level"] == 3), default=0) or 32 * MB
    sets = [n for n in _host_set_sizes(groups) if n <= max(64 * MB, host_free() // 8)]
    x = torch.empty(sets[-1] // 4, dtype=torch.float32).uniform_()
    threads = torch.get_num_threads()
    for k, label in ((threads, f"{threads} threads"), (1, "1 thread")):
        with _torch_threads(k):
            op = rep.sweep(f"read ({label}, torch)", "working set")
            for n in sets:
                v = _reread(x, n, 64 * MB)
                past = peak if n >= 4 * l3 else 0.0
                rep.time(op, _exact(n), v.sum, nbytes=v.numel() * 4, peak=past, at={"bytes": n, "threads": k})
    n = sets[-1] // 2
    src, dst = x[: n // 4], torch.empty(n // 4)

    def copy() -> None:
        dst.copy_(src)

    shape = f"{_size(n)}, read and write ({threads} threads, torch)"
    rep.time("copy", shape, copy, nbytes=2 * n, peak=peak, at={"bytes": n})


def _gemv_threads(rep: Report, past: int, peak: float, topo: Json) -> None:
    """the native gemv's one-row stream by its thread count: where the memory saturates"""
    from .engine.native import Native, NativeError

    R, C = LINEAR
    if Native.gemv is None:
        return rep.skip("gemv bf16 stream", "", "no native library")
    ws = Pool(functools.partial(_bf16, R, C), R * C * 2, past)
    x, y = torch.randn(1, C), torch.empty(1, R)
    fn = Native._gemv_lib.btb_gemv_bf16_rows  # the kernel itself, its thread count named a call

    def at(threads: int) -> Callable[[], None]:
        def call() -> None:
            w = ws()
            rc = fn(w.data_ptr(), R, C, x.data_ptr(), 1, y.data_ptr(), threads)
            if rc != 0:
                raise NativeError("btb_gemv_bf16_rows", rc)

        return call

    op = rep.sweep(f"gemv bf16 stream (native), {R}x{C}, 1 row", "threads")
    for t in _thread_counts(topo):
        rep.time(op, f"{t} thread{'s' if t > 1 else ''}", at(t), nbytes=R * C * 2, peak=peak, at={"threads": t})
    return None


def memory(rep: Report, f: Json) -> None:
    past = host_past()
    peak = (f["memory"].get("peak_gb_s") or 0.0) * 1e9
    rep.section("memory", "the host's, from its caches to DRAM")
    _host_sets(rep, peak)
    _let_go()
    _gemv_threads(rep, past, peak, f["cpu"])


# -- the host's ops -------------------------------------------------------------------------------------------------


def _native_gemv(ws: Pool, x: torch.Tensor, y: torch.Tensor) -> Callable[[], None]:
    from .engine.native import Native

    return lambda: Native.gemv(ws(), x, y)


def _cpu_gemv_size(rep: Report, op: str, side: int, held: bool, past: int) -> None:
    """one point of `_cpu_gemv_sizes`, its weights let go when it returns"""
    each = side * side * 2
    make = functools.partial(_bf16, side, side)
    ws = Pool.one(make()) if held else Pool(make, each, past, cap=4096)
    call = _native_gemv(ws, torch.randn(1, side), torch.empty(1, side))
    shape = f"{side}x{side} ({_size(each)})"
    rep.time(op, shape, call, nbytes=each, flops=2 * side * side, at={"bytes": each, "held": held})


def _cpu_gemv_sizes(rep: Report, past: int) -> None:
    """the native one-row gemv by weight size, the same weight each call (held in a cache while it fits one) and
    a pool of them past the L3 (streamed from DRAM, as a decode's layers are)"""
    from .engine.native import Native

    if Native.gemv is None:
        return rep.skip("gemv bf16", "", "no native library")
    for held in (True, False):
        how = "the same weight each call" if held else "weights streamed past the caches"
        op = rep.sweep(f"gemv bf16 (native), 1 row, {how}", "weight size")
        for side in SQUARES:
            _cpu_gemv_size(rep, op, side, held, past)
            _let_go()
    return None


def _cpu_rows(rep: Report, past: int) -> None:
    """the host's matmul by rows over a streamed weight: the native gemv, torch's float32 GEMM over the weight
    widened (the engine's path from `Native.gemm_rows` rows), torch's bf16 matmul - and the rows from which the GEMM
    is the faster, against the rows the engine switches at"""
    from .engine.native import Native

    R, C = LINEAR
    ws = Pool(functools.partial(_bf16, R, C), R * C * 2, past)
    native: dict[int, Timing] = {}
    gemm: dict[int, Timing] = {}
    if Native.gemv is None:
        rep.skip("gemv bf16 (native)", f"{R}x{C}", "no native library")
    else:
        op = rep.sweep(f"gemv bf16 (native), {R}x{C} streamed", "rows")
        for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            call = _native_gemv(ws, torch.randn(rows, C), torch.empty(rows, R))
            t = rep.time(op, _rows(rows), call, nbytes=R * C * 2, flops=2 * R * C * rows, at={"rows": rows})
            if t is not None:
                native[rows] = t

    def widened(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        return lambda: F.linear(x, ws().float())

    def plain(x: torch.Tensor) -> Callable[[], torch.Tensor]:
        return lambda: F.linear(x, ws())

    op = rep.sweep(f"gemm f32 (torch, the bf16 weight widened), {R}x{C} streamed", "rows")
    for rows in (16, 32, 64, 128, 256, 512, 1024, 2048):
        t = rep.time(op, _rows(rows), widened(torch.randn(rows, C)), flops=2 * R * C * rows, at={"rows": rows})
        if t is not None:
            gemm[rows] = t
    op = rep.sweep(f"matmul bf16 (torch), {R}x{C} streamed", "rows")
    for rows in (64, 512):
        rep.time(op, _rows(rows), plain(torch.randn(rows, C).bfloat16()), flops=2 * R * C * rows, at={"rows": rows})
    shared = sorted(set(native) & set(gemm))
    if shared:
        at = gemm_from(native, gemm)
        rep.found["host matmul"] = {"gemm_from_rows": at, "rows_compared": shared, "engine_gemm_rows": Native.gemm_rows}
        rep.say(
            f"  the GEMM overtakes the native gemv from {at} rows here"
            if at is not None
            else f"  the native gemv holds through {shared[-1]} rows here"
        )
        rep.say(f"  (the engine switches at {Native.gemm_rows})")


def gemm_from(gemv: dict[int, Timing], gemm: dict[int, Timing]) -> int | None:
    """the fewest rows, of those both were timed at, from which the GEMM is the faster at every count: its whole
    interval below the gemv's there and at each count past it - one row count that happens to clear, with the gemv
    back ahead past it, is not a crossover. None where the gemv holds"""
    rows = sorted(set(gemv) & set(gemm))
    wins = [gemm[r].hi < gemv[r].lo for r in rows]
    for i, r in enumerate(rows):
        if all(wins[i:]):
            return r
    return None


def _cpu_group(rep: Report, past: int) -> None:
    from .engine.native import Native

    R, C = EXPERT
    op = f"gemv bf16 group (native), experts of {R}x{C}, 1 row each"
    if Native.gemv_group is None:
        return rep.skip(op, "", "not in this library")

    def point(n: int) -> None:
        gs = Pool(lambda: [_bf16(R, C) for _ in range(n)], n * R * C * 2, past)
        xs, ys = [torch.randn(1, C) for _ in range(n)], [torch.empty(1, R) for _ in range(n)]
        rep.time(op, f"{n} experts", lambda: Native.gemv_group(gs(), xs, ys), nbytes=n * R * C * 2, at={"experts": n})

    rep.sweep(op, "experts in the call")
    for n in (4, 16, 64):
        point(n)
        _let_go()
    return None


def _mx4(device: Any = None) -> Any:
    from .mxfp4 import MxWeight

    R, C = MX4
    g = C // 32
    blocks = torch.randint(0, 256, (R * g * 16,), dtype=torch.uint8, device=device)
    return MxWeight(blocks, torch.randint(118, 124, (R * g,), dtype=torch.uint8, device=device), R, C)


def _mx4_ggml() -> Any:
    """an MXFP4 matrix in a GGUF's 17-byte blocks: each block's scale byte, then its 16 nibble bytes"""
    from .mxfp4 import MxWeight

    R, C = MX4
    raw = torch.randint(0, 256, (R * C // 32, 17), dtype=torch.uint8)
    raw[:, 0] = 120
    return MxWeight.from_ggml(raw.view(-1), R, C)


def _by_rows(rep: Report, op: str, fn: Any, ws: Pool, R: int, C: int, rows: Sequence[int], each: int) -> None:
    """a stored-format gemv `fn` (an MXFP4, FP8 or pack-12 weight) by rows, over the weights `ws` in turn"""

    def call(x: torch.Tensor, y: torch.Tensor) -> Callable[[], None]:
        return lambda: fn(ws(), x, y)

    rep.sweep(op, "rows")
    for r in rows:
        x, y = torch.randn(r, C), torch.empty(r, R)
        rep.time(op, _rows(r), call(x, y), nbytes=each, flops=2 * R * C * r, at={"rows": r})


def _cpu_mx4(rep: Report, past: int) -> None:
    from .engine.native import Native

    R, C = MX4
    each = R * C // 32 * 17
    op = f"gemv mxfp4 (native), {R}x{C}"
    if Native.gemv_mx4 is None:
        rep.skip(op, "", "not in this library")
    else:
        _by_rows(rep, op, Native.gemv_mx4, Pool(_mx4, each, past), R, C, (1, 4, 16), each)
        _let_go()
    if Native.gemv_mx4_group is not None:
        _cpu_mx4_group(rep, past, 8)
        _let_go()
    if Native.gemv_mx4_ggml is not None:
        ws = Pool(_mx4_ggml, each, past)
        x, y = torch.randn(1, C), torch.empty(1, R)
        op = "gemv mxfp4, a GGUF's blocks (native)"
        rep.time(op, f"{R}x{C}, 1 row", lambda: Native.gemv_mx4_ggml(ws(), x, y), nbytes=each, at={"rows": 1})


def _cpu_mx4_group(rep: Report, past: int, n: int) -> None:
    from .engine.native import Native

    R, C = MX4
    each = R * C // 32 * 17
    gs = Pool(lambda: [_mx4() for _ in range(n)], n * each, past)
    xs, ys = [torch.randn(1, C)] * n, [torch.empty(1, R) for _ in range(n)]
    shape = f"{n} experts of {R}x{C}, 1 row each"
    call = lambda: Native.gemv_mx4_group(gs(), xs, ys)
    rep.time("gemv mxfp4 group (native)", shape, call, nbytes=n * each, at={"experts": n})


def _cpu_fp8(rep: Report, past: int) -> None:
    from .engine.native import Native
    from .fp8 import F8Weight

    R, C = LINEAR
    op = f"gemv fp8 e4m3 (native), {R}x{C}, 128x128 scales"
    if Native.gemv_fp8 is None:
        return rep.skip(op, "", "not in this library")
    grid = (R // 128, C // 128)  # a 128x128 block's scale, as the FP8 checkpoints carry them

    def make() -> F8Weight:
        return F8Weight(torch.randint(0, 0x7F, (R * C,), dtype=torch.uint8), torch.rand(*grid) + 0.5, R, C)

    each = R * C + grid[0] * grid[1] * 4
    _by_rows(rep, op, Native.gemv_fp8, Pool(make, each, past), R, C, (1, 4, 16), each)
    return None


def _cpu_p12(rep: Report, past: int) -> None:
    from .engine.native import Native

    R, C = LINEAR
    n = R * C
    op = f"gemv pack12 (native), {R}x{C}"
    if Native.gemv_p12 is None:
        return rep.skip(op, "", "not in this library")
    # 15 high bytes of bf16 values near one, as a weight's table holds them; no escapes
    high = [0x3F, 0xBF, 0x3E, 0xBE, 0x3D, 0xBD, 0x3C, 0xBC, 0x3B, 0xBB, 0x40, 0xC0, 0x3A, 0xBA, 0x41, 0]
    tbl = torch.tensor(high, dtype=torch.uint8)
    none_i, none_v = torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.uint8)

    def make() -> tuple[torch.Tensor, torch.Tensor]:
        codes = torch.randint(0, 15, (2, n // 2), dtype=torch.uint8)
        return torch.randint(0, 256, (n,), dtype=torch.uint8), codes[0] | (codes[1] << 4)

    def p12(packed: tuple[torch.Tensor, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> None:
        Native.gemv_p12(packed[0], packed[1], tbl, none_i, none_v, 0, R, C, x, y)

    _by_rows(rep, op, p12, Pool(make, n * 3 // 2, past), R, C, (1, 4, 16), n * 3 // 2)
    return None


def _cpu_attn(rep: Report, past: int) -> None:
    """one row's decode attention by the context it reads over, and a tree's nodes over 4096 rows by their count"""
    from .engine.native import Native

    Hq, Hk, D = HEADS
    op = f"attention decode bf16 (native), {Hq}/{Hk} heads d{D}, 1 row"
    if Native.attn_decode is None:
        return rep.skip(op, "", "not in this library")

    def point(n: int) -> None:
        each = 2 * Hk * n * D * 2
        if 2 * each > host_free() // 8:
            return rep.skip(op, f"{n} rows", f"needs {_size(2 * each)}, more than an eighth of the RAM available")
        kv = Pool(lambda: (_bf16(Hk, n, D), _bf16(Hk, n, D)), each, past, cap=16)
        q, out = torch.randn(Hq, D), torch.empty(Hq, D)

        def call() -> None:
            k, v = kv()
            Native.attn_decode(q, k, v, D**-0.5, out)

        rep.time(op, f"{n} rows", call, nbytes=each, at={"context": n})
        return None

    rep.sweep(op, "context")
    for n in CONTEXTS:
        point(n)
        _let_go()
    if Native.attn_nodes is None:
        return None
    n = 4096
    each = 2 * Hk * n * D * 2
    kv = Pool(lambda: (_bf16(Hk, n, D), _bf16(Hk, n, D)), each, past, cap=16)

    def nodes(T: int) -> Callable[[], None]:
        offs = torch.arange(T + 1, dtype=torch.int32) * n
        idx = torch.arange(n, dtype=torch.int32).repeat(T)
        qs, outs = torch.randn(T, Hq, D), torch.empty(T, Hq, D)

        def call() -> None:
            k, v = kv()
            Native.attn_nodes(qs, k, v, offs, idx, D**-0.5, outs)

        return call

    op = rep.sweep(f"attention over a tree's nodes (native), {n} rows each", "nodes")
    for T in (2, 8, 32):
        rep.time(op, f"{T} nodes", nodes(T), nbytes=each, at={"nodes": T, "context": n})
    return None


def _cpu_delta(rep: Report, past: int) -> None:
    from .engine.native import Native

    hk, hv, dk, dv, K = DELTA
    C = 2 * hk * dk + hv * dv
    shape = f"{hk}/{hv} heads, a {dk}x{dv} state each"
    if Native.delta_step is None:
        return rep.skip("deltanet step (native)", shape, "not in this library")
    g = torch.Generator().manual_seed(1)

    def t(*s: int) -> torch.Tensor:
        return torch.randn(*s, generator=g) * 0.5

    mixed0, conv0 = t(C), t(C, K)
    mixed, conv_state, conv_w = mixed0.clone(), conv0.clone(), t(C, K)
    z, a, b, a_log, dt_bias = t(hv * dv), t(hv), t(hv), t(hv), t(hv)
    state, norm_w, out = torch.zeros(hv, dk, dv), torch.ones(dv), torch.empty(hv * dv)

    def call() -> None:
        # the inputs the step updates in place put back, so a step never iterates its own output down to zero
        mixed.copy_(mixed0)
        conv_state.copy_(conv0)
        Native.delta_step(
            mixed, conv_state, conv_w, None, z, a, b, a_log, dt_bias, state, hk, hv, dk, dv, norm_w, 1e-6, out
        )

    rep.time("deltanet step (native)", shape, call, nbytes=2 * hv * dk * dv * 4)
    return None


def _cpu_pick(rep: Report, past: int) -> None:
    from .engine.native import Native

    op = "sample (native), t 0.8, top-p 0.9, 1 row"
    if Native.sample_pick is None:
        return rep.skip(op, "", "not in this library")

    def pick(lg: torch.Tensor) -> Callable[[], torch.Tensor]:
        return lambda: Native.sample_pick(lg, [1], 0.8, 0, 0.9)

    rep.sweep(op, "vocabulary")
    for V in VOCABS:
        rep.time(op, f"{V} logits", pick(torch.randn(1, V)), at={"vocab": V})
    return None


def _cpu_sdpa(rep: Report, past: int) -> None:
    Hq, Hk, D = HEADS

    def prefill(S: int) -> Callable[[], torch.Tensor]:
        q, k, v = torch.randn(1, Hq, S, D), torch.randn(1, Hk, S, D), torch.randn(1, Hk, S, D)
        return lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    op = rep.sweep(f"attention prefill f32 (torch), {Hq}/{Hk} heads d{D}, causal", "rows")
    for S in (128, 512, 2048):
        rep.time(op, _rows(S), prefill(S), flops=2 * Hq * S * S * D, at={"rows": S})


def cpu_ops(rep: Report) -> None:
    from .engine.native import Native, isa

    past = host_past()
    if Native.gemv is not None:
        tier = isa(Native._gemv_lib)
        rep.section("cpu", f"btb's native kernels ({tier}, every core); streamed weights {_size(past)} in turn")
    else:
        rep.section("cpu", "no native library: the host's matmuls run on torch")
    ops = (_cpu_gemv_sizes, _cpu_rows, _cpu_group, _cpu_mx4, _cpu_fp8, _cpu_p12, _cpu_attn, _cpu_delta, _cpu_pick)
    for op in (*ops, _cpu_sdpa):
        try:
            op(rep, past)
        except Exception as e:  # an op's setup (its operands' allocation): its row says so, the report goes on
            rep.skip(op.__name__.removeprefix("_cpu_"), "", f"failed: {type(e).__name__}: {e}")
        _let_go()


# -- the card -------------------------------------------------------------------------------------------------------


def _launches(
    k: Any, name: str, grid: tuple[int, int, int], block: tuple[int, int, int], argss: list[list[Any]]
) -> Callable[[], None]:
    """a graph's body: kernel `name` launched once with each of `argss`"""

    def body() -> None:
        for args in argss:
            k.launch(name, grid, block, args)

    return body


def _card_memory(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """torch's read on the card over each working set, re-read in place (a set the L2 holds is read from it), and
    its copy past the L2"""
    sets = [n for n in CARD_SETS if n <= max(16 * MB, card_free(dev) // 4)]
    x = torch.empty(sets[-1] // 4, dtype=torch.float32, device=dev).uniform_()
    op = rep.sweep("read (torch)", "working set")
    for n in sets:
        v = _reread(x, n, 256 * MB)
        past_l2 = peak if n >= 4 * l2 else 0.0
        rep.time(op, _size(n), v.sum, clock=CardClock(), nbytes=v.numel() * 4, peak=past_l2, at={"bytes": n})
    n = sets[-1] // 2
    src, dst = x[: n // 4], torch.empty(n // 4, device=dev)

    def copy() -> None:
        dst.copy_(src)

    shape = f"{_size(n)}, read and write"
    rep.time("copy (torch)", shape, copy, clock=CardClock(), nbytes=2 * n, peak=peak, at={"bytes": n})


def _card_gemv_rows(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """btb's matvecs over a streamed projection by rows: the plain one at each width it is built for, the tensor-core
    one, and the host's bit-exact one"""
    from .engine.cuda import _CudaMixin as Card

    R, C = LINEAR
    ws = _pool(functools.partial(_bf16, R, C, device=dev), R * C * 2, past)
    P, ci = k.ptr, ctypes.c_int
    op = rep.sweep(f"gemv bf16, {R}x{C} streamed", "rows")
    for M in (1, 2, 4, 8, 16, 32):
        x, y = _bf16(M, C, device=dev), torch.empty(M, R, dtype=torch.bfloat16, device=dev)
        argss = [[P(w), P(x), P(y), ci(R), ci(C)] for w in ws]
        plain = _launches(k, f"btb_gemv_bf16_m{M}", ((R + 3) // 4, 1, 1), (128, 1, 1), argss)
        rep.card(op, _rows(M), plain, len(ws), nbytes=R * C * 2, flops=2 * R * C * M, peak=peak, at={"rows": M})
    name = "btb_gemv_mma_bf16"
    op = rep.sweep(f"gemv bf16 tensor cores, {R}x{C} streamed", "rows")
    if name in k.fn:
        # x padded to the 32 rows the kernel always carries
        x, y = _bf16(32, C, device=dev), torch.empty(32, R, dtype=torch.bfloat16, device=dev)
        grid, block = (Card._card_mma_grid(R, C), 1, 1), (32 * Card._card_mma_warps(R, C), 1, 1)
        for T in (1, 4, 16, 32):
            mma = _launches(k, name, grid, block, [[P(w), P(x), P(y), ci(R), ci(C), ci(T)] for w in ws])
            rep.card(op, _rows(T), mma, len(ws), nbytes=R * C * 2, flops=2 * R * C * T, peak=peak, at={"rows": T})
    else:
        rep.skip(op, "", "not in this build")
    xf, yf = torch.randn(1, C, device=dev), torch.empty(1, R, device=dev)

    def lane16() -> None:
        for w in ws:
            k.gemv_lane16(w, xf, yf)

    shape = f"{R}x{C}, 1 row"
    rep.card("gemv bf16, the host's bits", shape, lane16, len(ws), nbytes=R * C * 2, peak=peak, at={"rows": 1})


def _card_gemv_size(
    rep: Report, k: Any, dev: torch.device, op: str, name: str, side: int, held: bool, past: int, peak: float
) -> None:
    """one point of `_card_gemv_sizes`: kernel `name`'s one-row matvec over a [side, side] weight, the same each
    launch (`held`) or a pool of them past the L2; its weights let go when it returns"""
    from .engine.cuda import _CudaMixin as Card

    P, ci = k.ptr, ctypes.c_int
    each = side * side * 2
    make = functools.partial(_bf16, side, side, device=dev)
    ws = [make()] * 16 if held else _pool(make, each, past, cap=4096)
    mma = "mma" in name
    rows = 32 if mma else 1  # the tensor-core kernel's x is padded to the 32 rows it always carries
    x, y = _bf16(rows, side, device=dev), torch.empty(rows, side, dtype=torch.bfloat16, device=dev)
    if mma:
        grid, block = (Card._card_mma_grid(side, side), 1, 1), (32 * Card._card_mma_warps(side, side), 1, 1)
        argss = [[P(w), P(x), P(y), ci(side), ci(side), ci(1)] for w in ws]
    else:
        grid, block = ((side + 3) // 4, 1, 1), (128, 1, 1)
        argss = [[P(w), P(x), P(y), ci(side), ci(side)] for w in ws]
    body = _launches(k, name, grid, block, argss)
    shape = f"{side}x{side} ({_size(each)})"
    rep.card(op, shape, body, len(ws), nbytes=each, peak=0.0 if held else peak, at={"bytes": each, "held": held})


def _card_gemv_sizes(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """btb's one-row matvecs by weight size: the plain and the tensor-core one over weights streamed past the L2,
    and the tensor-core one over the same weight each launch (held in the L2 while it fits); then a short weight at
    16 and at 8 rows a block, where the engine takes 8 once 16 would leave SMs idle (`_card_mma_narrow`)"""
    from .engine.cuda import _CudaMixin as Card

    for name, kind, held in (
        ("btb_gemv_bf16_m1", "gemv bf16, 1 row, weights streamed", False),
        ("btb_gemv_mma_bf16", "gemv bf16 tensor cores, 1 row, weights streamed", False),
        ("btb_gemv_mma_bf16", "gemv bf16 tensor cores, 1 row, the same weight each launch", True),
    ):
        if name not in k.fn:
            continue
        op = rep.sweep(kind, "weight size")
        for side in SQUARES:
            _card_gemv_size(rep, k, dev, op, name, side, held, past, peak)
            _let_go()
    if "btb_gemv_mma_bf16" not in k.fn:
        return
    P, ci = k.ptr, ctypes.c_int
    R, C = SHORT
    ws = _pool(functools.partial(_bf16, R, C, device=dev), R * C * 2, past, cap=4096)
    x, y = _bf16(32, C, device=dev), torch.empty(32, R, dtype=torch.bfloat16, device=dev)
    block = (32 * Card._card_mma_warps(R, C), 1, 1)
    op = rep.sweep(f"gemv bf16 tensor cores, a short {R}x{C}, 1 row", "rows a block")
    for name, per_block in (("btb_gemv_mma_bf16", 16), ("btb_gemv_mma8_bf16", 8)):
        if name not in k.fn:
            rep.skip(op, f"{per_block} rows a block", "not in this build")
            continue
        grid = ((R + per_block - 1) // per_block, 1, 1)
        body = _launches(k, name, grid, block, [[P(w), P(x), P(y), ci(R), ci(C), ci(1)] for w in ws])
        shape, at = f"{per_block} rows a block", {"rows_a_block": per_block}
        rep.card(op, shape, body, len(ws), nbytes=R * C * 2, peak=peak, at=at)


def _card_glu(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    from .engine.cuda import _CudaMixin as Card

    H, I = MLP
    name, label, shape = "btb_gemv_mma_glu_silu", "gate and up with silu, tensor cores", f"{2 * I}x{H}, 1 row"
    if name not in k.fn:
        return rep.skip(label, shape, "not in this build")
    gus = _pool(functools.partial(_bf16, 2 * I, H, device=dev), 2 * I * H * 2, past)
    x, m = _bf16(32, H, device=dev), torch.empty(32, I, dtype=torch.bfloat16, device=dev)
    P, ci = k.ptr, ctypes.c_int
    block = (32 * Card._card_mma_warps(2 * I, H), 1, 1)
    glu = _launches(k, name, ((I + 15) // 16, 1, 1), block, [[P(gu), P(x), P(m), ci(I), ci(H), ci(1)] for gu in gus])
    rep.card(label, shape, glu, len(gus), nbytes=2 * I * H * 2, peak=peak, at={"rows": 1})
    return None


def _card_mx4(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """the host's MXFP4 gemv on the card, bit for bit, and the depot's widening of an expert to bf16"""
    R, C = MX4
    g = C // 32
    each = R * g * 17
    ws = _pool(lambda: _mx4(dev), each, past)
    xf, yf = torch.randn(1, C, device=dev), torch.empty(1, R, device=dev)

    def lane16() -> None:
        for w in ws:
            k.gemv_lane16_mx4(w, xf, yf)

    rep.card("gemv mxfp4, the host's bits", f"{R}x{C}, 1 row", lane16, len(ws), nbytes=each, peak=peak)
    ws.clear()
    _let_go()
    seats = max(2, min(64, -(-past // each)))
    blocks = torch.randint(0, 256, (seats, R, g, 16), dtype=torch.uint8, device=dev)
    scales = torch.randint(118, 124, (seats, R, g), dtype=torch.uint8, device=dev)
    picks = [torch.tensor([s], dtype=torch.int32, device=dev) for s in range(seats)]
    out = torch.empty(1, R * C, dtype=torch.bfloat16, device=dev)

    def widen() -> None:
        for s in picks:
            k.mx4_widen(blocks, scales, s, out)

    rep.card("mxfp4 widened to bf16", f"an expert's {R}x{C}", widen, seats, nbytes=each + R * C * 2, peak=peak)


def _card_attn_at(
    rep: Report, k: Any, dev: torch.device, past: int, peak: float, op: str, name: str, n: int, T: int
) -> None:
    """`T` rows' decode attention over a cache of `n` rows (a step's one, or a tree's verify) by kernel `name`: a
    block a query head, or a block a KV group reading each key once for its group's heads"""
    Hq, Hk, D = HEADS
    cap = (n + T + 1023) // 1024 * 1024  # btb_attn_split's split is 1024 keys
    S = cap // 1024
    live = 2 * Hk * (n + T) * D * 2
    if 4 * Hk * cap * D * 2 > card_free(dev) // 4:
        return rep.skip(op, f"{_rows(T)} over {n}", f"needs two caches of {_size(2 * Hk * cap * D * 2)}")
    kvs = _pool(lambda: (_bf16(Hk, cap, D, device=dev), _bf16(Hk, cap, D, device=dev)), live, past, 28)
    q, out = _bf16(T, Hq, D, device=dev), torch.empty(T, Hq, D, dtype=torch.bfloat16, device=dev)
    pm, pl = torch.zeros(S * T * Hq, device=dev), torch.zeros(S * T * Hq, device=dev)
    pa, cnt = torch.zeros(S * T * Hq * D, device=dev), torch.zeros(T * Hq, dtype=torch.int32, device=dev)
    n0 = torch.tensor([n], dtype=torch.int32, device=dev)
    par = torch.tensor([-1, 0, 1, 0, 3, 4, 2, 6][:T], dtype=torch.int32, device=dev)  # a tree's parents
    grouped_from = torch.tensor([0], dtype=torch.int32, device=dev)  # each key read once for its group's heads
    shared = grouped_from if "gqa" in name else None
    parts = (pm, pl, pa, cnt)
    argss = [k.attn_args(q, K, V, out, [n0, par], T, Hq, Hk, D**-0.5, *parts, S, 0, shared) for K, V in kvs]
    body = _launches(k, name, (Hq, T, S), (256, 1, 1), argss)
    rep.card(op, f"{_rows(T)} over {n}", body, len(kvs), nbytes=live, peak=peak, at={"context": n, "rows": T})
    return None


def _card_attn(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """decode attention by context, a step's one row a head a block; and a tree's five rows over long contexts in
    both forms, where the engine weighs them"""
    Hq, Hk, D = HEADS
    split, grouped = f"btb_attn_split_d{D}", f"btb_attn_split_gqa{Hq // Hk}_d{D}"
    if split not in k.fn:
        return rep.skip("attention", "", "not in this build")
    op = rep.sweep(f"attention, a head a block, {Hq}/{Hk} heads d{D}", "context")
    for n in CONTEXTS:
        _card_attn_at(rep, k, dev, past, peak, op, split, n, 1)
        _let_go()
    for name, form in ((split, "a head a block"), (grouped, "a KV group a block")):
        if name not in k.fn:
            continue
        op = rep.sweep(f"attention over a tree's 5 rows, {form}", "context")
        for n in (8192, 32768):
            _card_attn_at(rep, k, dev, past, peak, op, name, n, 5)
            _let_go()
    return None


def _card_rows(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """a fork's decode attention: every row over the shared prefix and its own steps, in one launch, by its rows"""
    Hq, Hk, D = HEADS
    n, steps = 4096, 16
    name = f"btb_attn_rows_d{D}"
    op = f"attention, a fork's rows over a {n}-row prefix, {Hq}/{Hk} heads d{D}"
    if name not in k.fn:
        return rep.skip(op, "", "not in this build")

    def point(rows: int) -> None:
        cap = (n + (steps + 1) * rows + 1023) // 1024 * 1024
        S = cap // 1024
        live = 2 * Hk * (n + steps * rows) * D * 2
        kvs = _pool(lambda: (_bf16(Hk, cap, D, device=dev), _bf16(Hk, cap, D, device=dev)), live, past, 28)
        q, out = _bf16(rows, Hq, D, device=dev), torch.empty(rows, Hq, D, dtype=torch.bfloat16, device=dev)
        pm, pl = torch.zeros(S * rows * Hq, device=dev), torch.zeros(S * rows * Hq, device=dev)
        pa, cnt = torch.zeros(S * rows * Hq * D, device=dev), torch.zeros(rows * Hq, dtype=torch.int32, device=dev)
        # the rows' layout: their steps from slot n, interleaved `rows` wide; each row's prefix the shared n rows
        lay = [n, steps, rows, *[x for c in range(rows) for x in (c, 0, n)]]
        rw = torch.tensor(lay, dtype=torch.int32, device=dev)
        argss = [k.attn_args(q, K, V, out, [rw], rows, Hq, Hk, D**-0.5, pm, pl, pa, cnt, S) for K, V in kvs]
        body = _launches(k, name, (Hq, rows, S), (256, 1, 1), argss)
        rep.card(op, _rows(rows), body, len(kvs), nbytes=live, peak=peak, at={"rows": rows, "context": n})

    rep.sweep(op, "rows")
    for rows in (2, 8, 32):
        point(rows)
        _let_go()
    return None


def _card_small(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """the kernels between the matmuls, a step's worth of each in a graph: a norm, an activation, the router, the
    DeltaNet step - and the pick, the token back on the host"""
    H, I = MLP
    P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
    reps = 64
    # the residual add's y zero: a graph replayed for seconds leaves the residual as it was
    h, x = _bf16(1, H, device=dev), torch.empty(1, H, dtype=torch.bfloat16, device=dev)
    y, w = torch.zeros(1, H, dtype=torch.bfloat16, device=dev), torch.ones(H, dtype=torch.bfloat16, device=dev)
    norm = [P(h), P(y), P(w), cf(1e-6), P(x), ci(H), ci(0)]
    body = _launches(k, "btb_add_rmsnorm", (1, 1, 1), (256, 1, 1), [norm] * reps)
    rep.card("residual add and rmsnorm", f"{H} wide, 1 row", body, reps)
    gu, m = _bf16(1, 2 * I, device=dev), torch.empty(1, I, dtype=torch.bfloat16, device=dev)
    act = [P(gu), P(m), ci(1), ci(I)]
    body = _launches(k, "btb_silu_mul", (min(4096, (I + 255) // 256), 1, 1), (256, 1, 1), [act] * reps)
    rep.card("silu(gate) x up", f"{I} wide, 1 row", body, reps)
    E, top = 128, 8
    logits = _bf16(1, E, device=dev)
    idx, wt = torch.empty(1, top, dtype=torch.int32, device=dev), torch.empty(1, top, dtype=torch.bfloat16, device=dev)

    def route() -> None:
        for _ in range(reps):
            k.moe_route(logits, E, top, idx, wt)

    rep.card("expert router", f"{E} experts, top {top}, 1 row", route, reps)
    hk, hv, dk, dv, K = DELTA
    C = 2 * hk * dk + hv * dv

    def t(*s: int) -> torch.Tensor:
        return torch.randn(*s, device=dev) * 0.5

    mixed, z, a, b, conv_w, conv0 = t(1, C), t(1, hv * dv), t(1, hv), t(1, hv), t(C, K), t(C, K)
    a_log, dt_bias, norm_w = t(hv), t(hv), torch.ones(dv, device=dev)
    state, out = torch.zeros(hv, dk, dv, device=dev), torch.empty(1, hv * dv, device=dev)
    steps = 16

    def delta() -> None:
        for _ in range(steps):
            k.delta_nodes(
                mixed,
                z,
                a,
                b,
                conv_w,
                None,
                conv0,
                state,
                None,
                None,
                a_log,
                dt_bias,
                norm_w,
                1e-6,
                0,
                hk,
                hv,
                dk,
                dv,
                out,
            )

    rep.card("deltanet step", f"{hk}/{hv} heads, a {dk}x{dv} state each", delta, steps, nbytes=2 * hv * dk * dv * 4)

    def pick(lg: torch.Tensor) -> Callable[[], torch.Tensor]:
        return lambda: k.pick(lg, [1], 0.8, 0, 0.9).cpu()

    op = rep.sweep("sample, the token to the host, t 0.8, top-p 0.9", "vocabulary")
    for V in VOCABS:
        rep.time(op, f"{V} logits", pick(torch.randn(1, V, device=dev)), at={"vocab": V})


def _card_layer(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """the rest of a layer's kernels, a step's worth of each in a graph: the q/k norm, rope and cache write every
    attention layer runs; Gemma's GELU and sandwich norm; a mixture's shared-expert combine; Qwen4's attention
    output gate, alone and folded into o_proj; the down projection with the activation folded into its read"""
    H, I = MLP
    Hq, Hk, D = HEADS
    P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
    reps = 64
    cap = 4096
    qkv = _bf16(1, (Hq + 2 * Hk) * D, device=dev)
    wq, wk = torch.ones(D, dtype=torch.bfloat16, device=dev), torch.ones(D, dtype=torch.bfloat16, device=dev)
    cos, sin = _bf16(cap, D, device=dev), _bf16(cap, D, device=dev)
    n0, depth = torch.tensor([cap // 2], dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
    K, V = _bf16(Hk, cap, D, device=dev), _bf16(Hk, cap, D, device=dev)
    qo = torch.empty(1, Hq, D, dtype=torch.bfloat16, device=dev)
    rope = [P(qkv), P(wq), P(wk), cf(1e-6), P(cos), P(sin), P(n0), P(depth), P(K), P(V), P(qo)]
    rope += [ci(1), ci(Hq), ci(Hk), ci(cap * D), ci(D), ci(0)]
    body = _launches(k, f"btb_norm_rope_kv_d{D}", (Hq + 2 * Hk, 1, 1), (32, 1, 1), [rope] * reps)
    rep.card("q/k norm, rope and the cache write", f"{Hq}/{Hk} heads d{D}, 1 row", body, reps)
    h, y = _bf16(1, H, device=dev), torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
    w = torch.ones(H, dtype=torch.bfloat16, device=dev)
    sandwich = [P(h), P(y), P(w), cf(1e-6), ci(H), ci(1)]
    body = _launches(k, "btb_sandwich_add", (1, 1, 1), (256, 1, 1), [sandwich] * reps)
    rep.card("sandwich norm and add (Gemma)", f"{H} wide, 1 row", body, reps)
    gu, m = _bf16(1, 2 * I, device=dev), torch.empty(1, I, dtype=torch.bfloat16, device=dev)
    gelu = [P(gu), P(m), ci(1), ci(I)]
    body = _launches(k, "btb_gelu_mul", (min(4096, (I + 255) // 256), 1, 1), (256, 1, 1), [gelu] * reps)
    rep.card("gelu(gate) x up (Gemma)", f"{I} wide, 1 row", body, reps)
    yr, ys, out = _bf16(1, H, device=dev), _bf16(1, H, device=dev), torch.empty(1, H, dtype=torch.bfloat16, device=dev)
    logits = _bf16(1, 129, device=dev)

    def combine() -> None:
        for _ in range(reps):
            k.moe_combine(yr, ys, logits, 128, out)

    rep.card("shared expert combined", f"{H} wide, 1 row", combine, reps)
    att, gated = _bf16(1, Hq * D, device=dev), torch.empty(1, Hq * D, dtype=torch.bfloat16, device=dev)
    qg = _bf16(1, 2 * Hq * D, device=dev)  # Qwen4's q rows: each head's q then its gate

    def gate() -> None:
        for _ in range(reps):
            k.sigmoid_mul(att, qg, gated, D, D, 2 * D)

    rep.card("attention output gate (Qwen4)", f"{Hq} heads d{D}, 1 row", gate, reps)
    os_ = _pool(functools.partial(_bf16, H, Hq * D, device=dev), H * Hq * D * 2, past)
    yo = torch.empty(1, H, dtype=torch.bfloat16, device=dev)

    def sgate() -> None:
        for wo in os_:
            k.gemv_sgate(wo, att, qg, yo, D, D, 2 * D)

    o_bytes = H * Hq * D * 2
    rep.card("o_proj with the gate folded (Qwen4)", f"{H}x{Hq * D}, 1 row", sgate, len(os_), nbytes=o_bytes, peak=peak)
    downs = _pool(functools.partial(_bf16, H, I, device=dev), H * I * 2, past)
    gu1 = _bf16(1, 2 * I, device=dev)
    argss = [[P(dn), P(gu1), P(yo), ci(H), ci(I)] for dn in downs]
    body = _launches(k, "btb_gemv_silu_bf16_m1", ((H + 3) // 4, 1, 1), (128, 1, 1), argss)
    rep.card("down with silu(gate) x up folded", f"{H}x{I}, 1 row", body, len(downs), nbytes=H * I * 2, peak=peak)


def _mm(x: torch.Tensor, weights: list[torch.Tensor]) -> Callable[[], None]:
    """a graph's body: torch's matmul of `x` by each of `weights`"""

    def body() -> None:
        for w in weights:
            F.linear(x, w)

    return body


def _card_cublas(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """torch's cuBLAS matmul as the engine's card layers call it: over a streamed projection by rows, from a step's
    one to a prefill's thousands, and square by size - where it turns from the memory's rate to the tensor cores'"""
    R, C = LINEAR
    ws = _pool(functools.partial(_bf16, R, C, device=dev), R * C * 2, past)
    op = rep.sweep(f"matmul bf16 (cuBLAS), {R}x{C} streamed", "rows")
    for rows in (1, 4, 16, 64, 256, 1024, 4096):
        body, flops = _mm(_bf16(rows, C, device=dev), ws), 2 * R * C * rows
        rep.card(op, _rows(rows), body, len(ws), nbytes=R * C * 2, flops=flops, peak=peak, at={"rows": rows})
    ws.clear()
    _let_go()
    op = rep.sweep("matmul bf16 (cuBLAS), square", "size")
    for side in (1024, 2048, 4096, 8192):
        body = _mm(_bf16(side, side, device=dev), [_bf16(side, side, device=dev)])
        rep.card(op, f"{side}x{side}x{side}", body, 1, flops=2 * side**3, at={"side": side})
        _let_go()


def _card_prefill(rep: Report, k: Any, dev: torch.device, past: int, peak: float, l2: int) -> None:
    """a prefill's attention (torch's) by rows, and a mixture prefill's experts as one grouped matmul by the rows
    each expert takes"""
    Hq, Hk, D = HEADS

    def attn(S: int) -> Callable[[], torch.Tensor]:
        q, kk, v = _bf16(1, Hq, S, D, device=dev), _bf16(1, Hk, S, D, device=dev), _bf16(1, Hk, S, D, device=dev)
        return lambda: F.scaled_dot_product_attention(q, kk, v, is_causal=True, enable_gqa=True)

    op = rep.sweep(f"attention prefill bf16 (torch), {Hq}/{Hk} heads d{D}, causal", "rows")
    for S in (512, 2048, 8192):
        rep.time(op, _rows(S), attn(S), clock=CardClock(), flops=2 * Hq * S * S * D, at={"rows": S})
    _let_go()
    grouped = getattr(torch, "_grouped_mm", None)
    n, H = 8, MX4[1]
    op = f"experts grouped bf16 (torch), {n} experts of {H}x{H}"
    if grouped is None:
        return rep.skip(op, "", "this torch has no _grouped_mm")
    gu, dn = _bf16(n, 2 * H, H, device=dev), _bf16(n, H, H, device=dev)

    def experts(rows: int) -> Callable[[], torch.Tensor]:
        x = _bf16(n * rows, H, device=dev)
        ends = torch.arange(1, n + 1, dtype=torch.int32, device=dev) * rows

        def call() -> torch.Tensor:
            g, u = grouped(x, gu.transpose(1, 2), offs=ends).chunk(2, dim=-1)
            return grouped((F.silu(g) * u).contiguous(), dn.transpose(1, 2), offs=ends)

        return call

    rep.sweep(op, "rows each")
    for rows in (8, 32, 128):
        flops, w_bytes = 2 * n * rows * 3 * H * H, n * 3 * H * H * 2
        rep.time(
            op,
            f"{rows} rows each",
            experts(rows),
            clock=CardClock(),
            nbytes=w_bytes,
            flops=flops,
            peak=peak,
            at={"rows": rows},
        )
    return None


def card_ops(rep: Report, f: Json, index: int) -> None:
    from .engine.native import Native

    dev = torch.device("cuda", index)
    card = f["cards"][index]
    k = Native.card_kernels()
    free = card_free(dev)
    l2 = k.l2_bytes() if k is not None else int(card.get("l2") or 0)
    past = int(max(16 * MB, min(max(4 * l2, 64 * MB), free // 8)))
    peak = (card.get("peak_gb_s") or 0.0) * 1e9
    kernels = "btb's kernels" if k is not None else f"btb's kernels not loaded ({Native.cuda_reason})"
    rep.section(f"cuda:{index}", f"{card['name']}, {kernels}; streamed weights {_size(past)} in turn")
    ops: list[Callable[[Report, Any, torch.device, int, float, int], None]] = [_card_memory]
    if k is not None:
        ops += [_card_gemv_rows, _card_gemv_sizes, _card_glu, _card_mx4, _card_attn, _card_rows, _card_small]
        ops += [_card_layer]
    ops += [_card_cublas, _card_prefill]
    for op in ops:
        try:
            op(rep, k, dev, past, peak, l2)
        except Exception as e:  # an op's setup (its operands' allocation): its row says so, the report goes on
            rep.skip(op.__name__.removeprefix("_card_"), "", f"failed: {type(e).__name__}: {e}")
        _let_go()


# -- the bus --------------------------------------------------------------------------------------------------------


def _bus_link(f: Json, index: int) -> tuple[Any, Any, float]:
    """the link's generation and width as it runs under a transfer (a card trains it down when idle), and its
    rate each direction"""
    dev = torch.device("cuda", index)
    h = torch.empty(64 * MB, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(64 * MB, dtype=torch.uint8, device=dev)
    for _ in range(8):
        d.copy_(h, non_blocking=True)
    bus_id = f["cards"][index]["bus_id"]
    link = (sysinfo.nvml_card(bus_id) if bus_id else None) or {}
    torch.cuda.synchronize()
    gen, width = link.get("pcie_gen"), link.get("pcie_width")
    return gen, width, PCIE_LANE.get(gen or 0, 0.0) * (width or 0) * 1e9


def _bus_copies(rep: Report, index: int, peak: float) -> None:
    """copies each way by size, pinned and pageable, as the engine pays them: the copy and the wait for it"""
    dev = torch.device("cuda", index)
    n = max(TRANSFERS)
    room = host_free() // 8
    if n > room:
        n = max(t for t in TRANSFERS if t <= max(16 * MB, room))
    sizes = [t for t in TRANSFERS if t <= n]
    pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    pageable = torch.ones(n, dtype=torch.uint8)  # its pages touched
    d = torch.empty(n, dtype=torch.uint8, device=dev)

    def copy(dst: torch.Tensor, src: torch.Tensor) -> Callable[[], None]:
        def call() -> None:
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize()

        return call

    for host, kind in ((pinned, "pinned"), (pageable, "pageable")):
        for up in (True, False):
            op = rep.sweep(f"{'to' if up else 'from'} the card, {kind}", "size")
            for s in sizes:
                call = copy(d[:s], host[:s]) if up else copy(host[:s], d[:s])
                at = {"bytes": s, "pinned": kind == "pinned", "to_card": up}
                rep.time(op, _size(s), call, nbytes=s, peak=peak if s >= 16 * MB else 0.0, at=at)


def _bus_trip(rep: Report, index: int) -> None:
    """a hidden row to the card, one kernel on it, and back: the handoff a split pass pays at each crossing"""
    dev = torch.device("cuda", index)
    H = MLP[0]
    x = torch.ones(H, dtype=torch.bfloat16, pin_memory=True)
    back = torch.empty(H, dtype=torch.bfloat16, pin_memory=True)
    xd = torch.empty(H, dtype=torch.bfloat16, device=dev)

    def trip() -> None:
        xd.copy_(x, non_blocking=True)
        xd.mul_(1)
        back.copy_(xd, non_blocking=True)
        torch.cuda.synchronize()

    rep.time("a row's round trip", f"{H} bf16 to the card, one kernel, back", trip)


def bus(rep: Report, f: Json, index: int) -> None:
    gen, width, peak = _bus_link(f, index)
    rep.found[f"cuda:{index} link"] = {"pcie_gen": gen, "pcie_width": width, "peak_gb_s": peak / 1e9}
    rep.section(
        f"bus to cuda:{index}",
        f"PCIe {gen}.0 x{width} under a transfer, {peak / 1e9:.1f} GB/s a direction" if peak else "the link unread",
    )
    _let_go()
    _bus_copies(rep, index, peak)
    _let_go()
    _bus_trip(rep, index)


# -- Apple's GPU ----------------------------------------------------------------------------------------------------


def _mlx_gemv(rep: Report) -> None:
    from .mlx.core import mx
    from .mlx.gemv import gemv

    m = mx()

    def point(op: str, side: int) -> None:
        w = m.random.normal((side, side)).astype(m.bfloat16)
        x = m.ones((1, side), dtype=m.bfloat16)
        m.eval(w, x)
        each = side * side * 2
        rep.time(op, f"{side}x{side} ({_size(each)})", lambda: m.eval(gemv(w, x)), nbytes=each, at={"bytes": each})

    op = rep.sweep("gemv bf16, 1 row", "weight size")
    for side in (*SQUARES, 16384):
        point(op, side)
        m.clear_cache()


def _mlx_attn(rep: Report) -> None:
    from .mlx.attn import attn_decode
    from .mlx.core import mx

    m = mx()
    Hq, Hk, D = HEADS

    def point(op: str, n: int) -> None:
        q = m.random.normal((Hq, D)).astype(m.float32)
        kb = m.random.normal((1, Hk, n, D)).astype(m.bfloat16)
        vb = m.random.normal((1, Hk, n, D)).astype(m.bfloat16)
        m.eval(q, kb, vb)
        call = lambda: m.eval(attn_decode(q, kb, vb, n, D**-0.5))
        rep.time(op, f"{n} rows", call, nbytes=2 * Hk * n * D * 2, at={"context": n})

    op = rep.sweep(f"attention decode bf16, {Hq}/{Hk} heads d{D}, 1 row", "context")
    for n in CONTEXTS:
        point(op, n)
        m.clear_cache()


def mlx_ops(rep: Report) -> None:
    rep.section("mlx", "Apple's GPU, through btb's Metal kernels")
    for op in (_mlx_gemv, _mlx_attn):
        try:
            op(rep)
        except Exception as e:
            rep.skip("mlx", "", f"failed: {type(e).__name__}: {e}")


# -- the drives -----------------------------------------------------------------------------------------------------


def _aligned(n: int) -> torch.Tensor:
    """`n` bytes at a 4096-byte boundary: an unbuffered read's destination"""
    raw = torch.empty(n + 4096, dtype=torch.uint8)
    a = (-raw.data_ptr()) % 4096
    return raw[a : a + n]


def _write(rep: Report, fd: int, budget: int) -> None:
    """the drive's sequential write into the open file `fd`, each write flushed, the file `budget` bytes at most
    whatever the sampler does: 32 MB at a time (half the budget, or the least the sampler takes) - the file brought to
    8 of them at least for the reads - then 1 MB at a time in what the budget has left, where the flush's own cost
    shows"""
    chunk = os.urandom(CHUNK)  # incompressible: a filesystem that compresses or dedupes writes what it is given
    warm = 2  # the sampler's warm-up calls write too

    def write(n: int) -> Callable[[], None]:
        def call() -> None:
            view = memoryview(chunk)[:n]
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)

        return call

    op = rep.sweep("write, each flushed, sequential", "size")
    big = max(rep.sampler.min_n, budget // 2 // CHUNK - warm)
    rep.time(op, _size(CHUNK), write(CHUNK), nbytes=CHUNK, sample_s=0.0, max_n=big, at={"bytes": CHUNK})
    while os.fstat(fd).st_size < 8 * CHUNK:
        write(CHUNK)()
    small = min(MB, CHUNK)
    left = (budget - os.fstat(fd).st_size) // small - warm
    if left < rep.sampler.min_n:
        return rep.skip(op, _size(small), "no room left in the test file's budget")
    rep.time(op, _size(small), write(small), nbytes=small, sample_s=0.0, max_n=left, at={"bytes": small})
    return None


def _rate(t: Timing, nbytes: int) -> tuple[float, float]:
    """(the median rate in GB/s, its interval's width): a read's figures as the Route's rule takes them"""
    return nbytes / t.median / 1e9, nbytes / t.lo / 1e9 - nbytes / t.hi / 1e9


def _reads(rep: Report, path: str, handles: list[int], where: str) -> None:
    """btb's own reader on the file at `path` (unbuffered: the drive is timed, not the OS's cache): reads one at a
    time by size, an expert's span among them; the span and a 4 KB read with 4 and 16 in flight; and the reading
    rule btb's Route draws from them (`BatchScheduler._disk_rule`)"""
    from .engine.native import Native
    from .engine.scheduler import BatchScheduler

    size = os.path.getsize(path)
    rng = random.Random(1)

    def at(nb: int) -> int:
        return rng.randrange(0, (size - nb) // 4096) * 4096

    h = Native.open(path)
    handles.append(h)
    span = int(min(6553600, size // 16) // 4096 * 4096)  # the drive probe's expert-sized read
    sizes = sorted({n for n in READS if n <= size // 2} | {span})
    bufs = {n: _aligned(n) for n in sizes}

    def one(nb: int) -> Callable[[], None]:
        return lambda: Native.read_at(h, at(nb), nb, bufs[nb])

    alone: dict[int, Timing] = {}
    op = rep.sweep("read, one in flight, at random offsets", "size")
    for nb in sizes:
        label = _size(nb) + (" (an expert's span)" if nb == span else "")
        t = rep.time(op, label, one(nb), nbytes=nb, at={"bytes": nb, "depth": 1})
        if t is not None:
            alone[nb] = t
    rates: dict[int, tuple[float, float]] = {}
    if span in alone:
        rates[1] = _rate(alone[span], span)
    for nb, each in ((span, 2), (4096, 32)):
        op = rep.sweep(f"read {_size(nb)}, at random offsets", "reads in flight")
        for depth in (4, 16):
            hs = [Native.open(path) for _ in range(depth)]
            handles += hs
            dst = [_aligned(nb) for _ in range(depth)]

            def reader(
                i: int, nb: int = nb, each: int = each, hs: list[int] = hs, dst: list[torch.Tensor] = dst
            ) -> None:
                for _ in range(each):
                    Native.read_at(hs[i], at(nb), nb, dst[i])

            with ThreadPoolExecutor(depth) as ex:

                def burst(ex: ThreadPoolExecutor = ex, depth: int = depth) -> None:
                    list(ex.map(reader, range(depth)))

                t = rep.time(
                    op, f"{depth} in flight", burst, nbytes=each * depth * nb, at={"bytes": nb, "depth": depth}
                )
            if t is not None and nb == span:
                rates[depth] = _rate(t, each * depth * span)
    if 4096 not in alone or span not in alone:
        return
    big = bufs[span]
    other = torch.empty_like(big)
    copy = rep.sampler.time(lambda: other.copy_(big)).median  # what a merged read's bounce copy costs
    profile = {"rates": rates, "reps": 3, "fixed_ms": alone[4096].median * 1e3, "single_ms": alone[span].median * 1e3}
    rule = BatchScheduler._disk_rule({**profile, "copy_ms": copy * 1e3})
    rep.found[f"drive {where}"] = rule
    rep.say(
        f"  btb's reader here: {rule['depth']} reads in flight, {rule['ahead']} ahead, adjacent reads "
        f"{'merged' if rule['merge'] else 'apart'}"
    )


def drive(rep: Report, where: str) -> None:
    """The drive under `where`, through a file the bench writes there and removes: its sequential write, then btb's
    own reader on it (`_reads`)"""
    from .engine.native import Native

    rep.section("drive", where)
    try:
        free = shutil.disk_usage(where).free
    except OSError as e:
        return rep.skip("drive", "", f"not readable: {e}")
    # every write together, the file's most: the least the sampler writes for each sweep (its warm-up and its least
    # samples) and the 8 chunks the reads need must fit, with DRIVE_KEEP left over
    budget = int(min(64 * CHUNK, free - DRIVE_KEEP))
    need = max(8 * CHUNK, (rep.sampler.min_n + 2) * (CHUNK + min(MB, CHUNK)))
    if budget < need:
        return rep.skip("write", "", f"{_size(free)} free: no room for a test file beside the {_size(DRIVE_KEEP)} kept")
    path = os.path.join(where, f".btb-bench-{os.getpid()}.tmp")
    handles: list[int] = []
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except OSError as e:
        return rep.skip("write", "", f"cannot write here: {e}")
    try:
        try:
            _write(rep, fd, budget)
        except OSError as e:  # the drive filled or refused a write: its row says so, the report goes on
            return rep.skip("write", "", f"failed: {e}")
        finally:
            os.close(fd)
        if Native.open is None or Native.read_at is None:
            return rep.skip("read", "", "no native library: btb's unbuffered reader is not here")
        try:
            _reads(rep, path, handles, where)
        except Exception as e:  # an unbuffered open the filesystem refuses (tmpfs, some network mounts)
            rep.skip("read", "", f"failed: {type(e).__name__}: {e}")
        return None
    finally:
        for h in handles:
            with contextlib.suppress(Exception):
                Native.close(h)
        with contextlib.suppress(OSError):
            os.remove(path)


# -- the report -----------------------------------------------------------------------------------------------------


def run(
    sampler: Sampler,
    say: Callable[[str], object],
    card: int | None = None,
    drives: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> Json:
    """The machine report: the facts, then each section's timings as they land (`say` takes each line), the card
    `card` the one timed (None: no card), the drives under `drives` written and read; the sections in `skip` left
    out. Returns the report as data."""
    from .engine.device import mlx_available

    t0 = time.perf_counter()
    f = facts(drives)
    for line in facts_lines(f):
        say(line)
    say("")
    say(
        f"each point: its median, sampled until the 95% interval is within ±{sampler.ci:.0%} of it or for "
        f"{sampler.budget_s:g} s, whichever comes first ('~': the time ran out first, the interval as it stood)"
    )
    rep = Report(sampler, say)
    if "memory" not in skip:
        memory(rep, f)
        _let_go()
    if "cpu" not in skip:
        cpu_ops(rep)
    if card is not None:
        torch.cuda.set_device(card)
        if "card" not in skip:
            card_ops(rep, f, card)
        if "bus" not in skip:
            bus(rep, f, card)
            _let_go()
    if "mlx" not in skip and mlx_available():
        mlx_ops(rep)
    if "drive" not in skip:
        for d in drives:
            drive(rep, d)
            _let_go()
    took = time.perf_counter() - t0
    timed = [r for r in rep.rows if r.t is not None]
    wide = sum(1 for r in timed if r.t is not None and not r.t.settled)
    say("")
    say(
        f"done in {took:.0f} s: {len(timed)} timings"
        + (f", {wide} still wider than ±{sampler.ci:.0%} when their {sampler.budget_s:g} s ran out (~)" if wide else "")
    )
    return {
        "machine": f,
        "sampler": asdict(sampler),
        "timings": [r.json() for r in rep.rows],
        "found": rep.found,
        "took_s": took,
    }
