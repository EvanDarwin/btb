# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The machine report (btb/machine.py, `btb bench` without a model) and the hardware readers under it
(btb/sysinfo.py): the median's interval and the sampler's stopping rule, the SMBIOS memory devices (and udev's copy
of them) and the channels their locators name, the CPU's cores, caches and features - Linux's sysfs and macOS's
sysctl read from fakes on any OS, then this machine's own - the host, card and MLX sections at small sizes, a drive
written and read through btb's reader, the report at full size on a small machine's free RAM, and the command's
routing between the report and a model's bench. Model-free; the drive tests write a file into a temporary folder (a few
MB, or 256 MB at full size) and remove it."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from pytest import CaptureFixture, MonkeyPatch

from btb import machine, sysinfo
from btb.kinds import HostCache, HostCacheTotal, Json
from tests.helpers import KB, MB, fixture


def test_the_median_interval_is_the_order_statistics_around_it() -> None:
    """ranks n/2 -+ 1.96 sqrt(n)/2 of the sorted sample - 9 and 22 of 30 - whatever order it came in, and for a
    skewed distribution the interval holds the true median about 95% of the time"""
    xs = [float(x) for x in range(1, 31)]
    random.Random(0).shuffle(xs)
    assert machine.median_ci(xs) == (15.5, 9.0, 22.0)
    rng = random.Random(1)
    held = 0
    for _ in range(2000):
        lo, hi = machine.median_ci([rng.expovariate(1.0) for _ in range(30)])[1:]
        held += lo <= math.log(2) <= hi
    assert 0.93 <= held / 2000 <= 0.99, held / 2000


class _Script(machine.Clock):
    """a clock whose stretches last as scripted, in turn"""

    def __init__(self, *durations: float) -> None:
        self.durations, self.i = durations, 0

    def stop(self) -> float:
        d = self.durations[self.i % len(self.durations)]
        self.i += 1
        return d


def test_the_sampler_stops_at_the_interval_or_the_time_whichever_first() -> None:
    """a steady op settles at the least samples; a noisy one is sampled until its samples or its time run out,
    and comes back marked unsettled with the interval as it stood"""
    s = machine.Sampler(ci=0.02, budget_s=60.0, min_n=10, max_n=200, sample_s=0.0)
    t = s.time(lambda: None, _Script(1e-3))
    assert (t.n, t.calls, t.settled) == (10, 1, True)
    assert t.median == t.lo == t.hi == 1e-3
    t = s.time(lambda: None, _Script(1e-3, 2e-3))
    assert (t.n, t.settled) == (200, False), "a bimodal op never closes within 2%: max_n ends it"
    assert (t.lo, t.hi) == (1e-3, 2e-3) and t.half > 0.02
    t = machine.Sampler(budget_s=0.0, min_n=10, sample_s=0.0).time(lambda: None, _Script(1e-3, 2e-3))
    assert (t.n, t.settled) == (10, False), "with the time spent, the least samples are still taken"
    assert t.per(4).median == t.median / 4


def test_a_working_set_is_read_over_and_over_in_place() -> None:
    """a set smaller than the pass is a stride-0 view of it, never a copy: the same bytes summed as often as the
    pass needs, so a set a cache holds is read from that cache"""
    import torch

    x = torch.rand(1 << 16)
    v = machine._reread(x, 32 << 10, 1 << 20)
    assert v.shape == (32, 8192) and v.stride() == (0, 1) and v.data_ptr() == x.data_ptr()
    assert torch.allclose(v.sum(), 32 * x[:8192].sum(), rtol=1e-4)
    assert machine._reread(x, 256 << 10, 64 << 10).shape == (1, 65536), "a set past the pass is read once"


def _t(median: float, half: float = 0.01) -> machine.Timing:
    return machine.Timing(median, median * (1 - half), median * (1 + half), 10, 1, True)


def test_the_gemm_takes_over_from_the_rows_it_stays_ahead() -> None:
    """the crossover is the fewest rows from which the GEMM is clearly faster at every count after: a lone count
    that clears, or one inside the noise, is not it; a gemv that holds gives none"""
    gemv = {16: _t(1.0), 32: _t(1.0), 64: _t(1.0), 128: _t(1.0), 256: _t(1.0)}
    assert machine.gemm_from(gemv, {16: _t(2.0), 32: _t(0.5), 64: _t(1.5), 128: _t(0.5), 256: _t(0.4)}) == 128
    assert machine.gemm_from(gemv, {16: _t(2.0), 32: _t(1.5), 64: _t(0.99), 128: _t(0.5), 256: _t(0.4)}) == 128
    assert machine.gemm_from(gemv, {16: _t(2.0), 256: _t(1.5)}) is None
    assert machine.gemm_from(gemv, {}) is None


def _type17(strings: list[str], size: int, ext_mb: int = 0, rated: int = 5600, runs: int = 4800) -> bytes:
    """one SMBIOS 3.3 memory device: 0x5C bytes formatted, then its strings"""
    s = bytearray(0x5C)
    s[0], s[1] = 17, 0x5C
    struct.pack_into("<HH", s, 0x08, 72, 64)  # total and data width
    struct.pack_into("<H", s, 0x0C, size)
    s[0x0E] = 0x09  # DIMM
    s[0x10], s[0x11] = 1, 2  # device and bank locators
    s[0x12] = 0x22  # DDR5
    struct.pack_into("<H", s, 0x15, rated)
    s[0x17], s[0x18], s[0x1A] = 3, 4, 5  # maker, serial, part
    s[0x1B] = 0x02  # rank 2
    struct.pack_into("<I", s, 0x1C, ext_mb)
    struct.pack_into("<H", s, 0x20, runs)
    return bytes(s) + b"".join(x.encode() + b"\0" for x in strings) + b"\0"


def test_smbios_memory_devices_read_as_modules() -> None:
    """a table's type-17 structures among others: a 32 GB module through the extended size, a KB-sized one, an
    empty slot left out; the serial number never read into the module"""
    other = bytes([4, 6, 0, 0, 0, 0]) + b"cpu\0\0"  # a processor record, its one string
    strings = ["Controller0-ChannelA-DIMM0", "BANK 0", "Maker", "SECRET123", "PART-1"]
    table = (
        other
        + _type17(strings, 0x7FFF, ext_mb=32768)
        + _type17(strings, 0x8000 | 512)
        + _type17(strings, 0)
        + bytes([127, 4, 0, 0])
        + b"\0\0"
    )
    found = sysinfo._smbios_structs(table, 17)
    assert len(found) == 3
    mods = [m for s, st in found if (m := sysinfo._module(s, st)) is not None]
    assert len(mods) == 2, "the empty slot is not a module"
    m = mods[0]
    assert (m["size_gb"], m["kind"], m["form"], m["rated_mts"], m["mts"]) == (32.0, "DDR5", "DIMM", 5600, 4800)
    assert (m["width"], m["rank"], m["slot"], m["bank"], m["maker"], m["part"]) == (
        64, 2, "Controller0-ChannelA-DIMM0", "BANK 0", "Maker", "PART-1"
    )  # fmt: skip
    assert mods[1]["size_gb"] < 0.01, "512 KB, not 512 MB"
    assert "SECRET123" not in json.dumps(mods)


def test_the_channels_are_the_ones_the_locators_name() -> None:
    """a channel per Intel desktop controller, two modules on one named channel sharing it, AMD's bank locators,
    an ASUS-style slot name; nothing named, no count - and so no peak rate guessed"""

    def mods(*slots: tuple[str, str]) -> list[Json]:
        return [{"slot": slot, "bank": bank, "mts": 4800, "rated_mts": 4800, "width": 64} for slot, bank in slots]

    ch = sysinfo.memory_channels
    assert ch(mods(("Controller0-DIMM1", "BANK 0"), ("Controller1-DIMM1", "BANK 0"))) == 2
    assert ch(mods(("Controller0-ChannelA-DIMM0", ""), ("Controller0-ChannelA-DIMM1", ""))) == 1
    assert ch(mods(("DIMM 1", "P0 CHANNEL A"), ("DIMM 1", "P0 CHANNEL B"))) == 2
    assert ch(mods(("DIMM_A2", "BANK 0"), ("DIMM_B2", "BANK 1"))) == 2
    assert ch(mods(("DIMM 1", "BANK 0"), ("DIMM 2", "BANK 1"))) is None
    two = mods(("Controller0-DIMM1", ""), ("Controller1-DIMM1", ""))
    assert machine._memory_peak(two, 2) == pytest.approx(76.8)
    assert machine._memory_peak(two, None) is None


_UDEV = """I:5129876
E:MEMORY_ARRAY_LOCATION=System Board Or Motherboard
E:MEMORY_ARRAY_MAX_CAPACITY=137438953472
E:MEMORY_DEVICE_0_TOTAL_WIDTH=64
E:MEMORY_DEVICE_0_DATA_WIDTH=64
E:MEMORY_DEVICE_0_SIZE=34359738368
E:MEMORY_DEVICE_0_FORM_FACTOR=DIMM
E:MEMORY_DEVICE_0_LOCATOR=Controller0-ChannelA-DIMM0
E:MEMORY_DEVICE_0_BANK_LOCATOR=BANK 0
E:MEMORY_DEVICE_0_TYPE=DDR5
E:MEMORY_DEVICE_0_SPEED_MTS=5600
E:MEMORY_DEVICE_0_MANUFACTURER=Maker
E:MEMORY_DEVICE_0_SERIAL_NUMBER=SECRET123
E:MEMORY_DEVICE_0_ASSET_TAG=TAG456
E:MEMORY_DEVICE_0_PART_NUMBER=PART-1
E:MEMORY_DEVICE_0_RANK=2
E:MEMORY_DEVICE_0_CONFIGURED_SPEED_MTS=4800
E:MEMORY_DEVICE_1_PRESENT=0
E:MEMORY_DEVICE_1_LOCATOR=Controller0-ChannelA-DIMM1
E:MEMORY_DEVICE_2_DATA_WIDTH=64
E:MEMORY_DEVICE_2_SIZE=17179869184
E:MEMORY_DEVICE_2_FORM_FACTOR=Row Of Chips
E:MEMORY_DEVICE_2_LOCATOR=Controller1-ChannelA-DIMM0
E:MEMORY_DEVICE_2_TYPE=<OUT OF SPEC>
E:MEMORY_DEVICE_2_MANUFACTURER=Not Specified
E:MEMORY_DEVICE_2_SPEED_MTS=5600
G:systemd
V:1
"""


def test_udev_memory_devices_read_as_modules(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """without root, Linux's modules come from udev's database: the populated slots as the SMBIOS table gives them,
    an empty slot left out, no serial number or asset tag; a database without them (no systemd's helper) or none at
    all reads as no modules"""
    db = tmp_path / "dmi-id"  # udev's own name, "+dmi:id", is not a name Windows allows
    db.write_text(_UDEV, encoding="utf-8")
    monkeypatch.setattr(sysinfo, "_UDEV_DMI", str(db))
    mods = sysinfo._udev_modules()
    assert mods is not None and len(mods) == 2, mods
    assert mods[0] == {
        "size_gb": 32.0, "kind": "DDR5", "form": "DIMM", "rated_mts": 5600, "mts": 4800, "width": 64, "rank": 2,
        "slot": "Controller0-ChannelA-DIMM0", "bank": "BANK 0", "maker": "Maker", "part": "PART-1",
    }  # fmt: skip
    # the helper's names for the codes read as the table's own read gives them: "row of chips", "unknown"
    two = mods[1]
    got = (two["size_gb"], two["form"], two["kind"], two["mts"], two["maker"])
    assert got == (16.0, "row of chips", "unknown", 0, "")
    assert "SECRET123" not in json.dumps(mods) and "TAG456" not in json.dumps(mods)
    assert sysinfo.memory_channels(mods) == 2
    with monkeypatch.context() as m:  # Linux without root: the table unreadable, udev's copy read in its place
        m.setattr(sysinfo, "_smbios_table", lambda: None)
        m.setattr(sysinfo.sys, "platform", "linux")
        assert sysinfo.memory_modules() == mods
    db.write_text("I:5129876\nE:ID_VENDOR=x\nV:1\n", encoding="utf-8")
    assert sysinfo._udev_modules() is None
    monkeypatch.setattr(sysinfo, "_UDEV_DMI", str(tmp_path / "none"))
    assert sysinfo._udev_modules() is None


Level = tuple[int, int, int]  # a cache level's (instance, share, total)


def _figures(groups: list[Json]) -> tuple[dict[str, int], dict[str, int]]:
    """`cache_sizes` and `cache_totals` of `groups`"""
    return sysinfo.cache_sizes(groups), sysinfo.cache_totals(groups)


def _want(l1d: Level, l2: Level, l3: Level = (0, 0, 0)) -> tuple[dict[str, int], dict[str, int]]:
    """`_figures` as each level's (smallest instance, smallest share with every thread busy, total)"""
    sizes: dict[str, int] = {
        HostCache.L1D_INSTANCE: l1d[0], HostCache.L1D_SHARE: l1d[1],
        HostCache.L2_INSTANCE: l2[0], HostCache.L2_SHARE: l2[1],
        HostCache.L3_INSTANCE: l3[0], HostCache.L3_SHARE: l3[1],
    }  # fmt: skip
    return sizes, {HostCacheTotal.L1D: l1d[2], HostCacheTotal.L2: l2[2], HostCacheTotal.L3: l3[2]}


def _sysfs(root: Path, files: dict[str, str]) -> str:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    return str(root)


def test_linux_sysfs_reads_as_cores_and_caches(tmp_path: Path) -> None:
    """three sysfs trees: an Intel hybrid (two hyperthreaded performance cores, two efficiency cores sharing an
    L2), an Arm big.LITTLE whose core ids restart in each cluster, and an old kernel with no sibling lists"""
    cpu = "devices/system/cpu"
    files = {f"{cpu}/online": "0-5", "devices/cpu_core/cpus": "0-3"}
    for c in range(6):
        p = c < 4
        sib = f"{c - c % 2}-{c - c % 2 + 1}" if p else str(c)
        d = f"{cpu}/cpu{c}"
        files |= {
            f"{d}/topology/core_cpus_list": sib,
            f"{d}/cpufreq/base_frequency": "3400000" if p else "2500000",
            f"{d}/cpufreq/cpuinfo_max_freq": "5400000" if p else "4200000",
        }
        caches = [(1, "Data", "48K" if p else "32K", sib), (1, "Instruction", "64K", sib),
                  (2, "Unified", "2048K" if p else "4096K", sib if p else "4-5"), (3, "Unified", "30M", "0-5")]  # fmt: skip
        for i, (lvl, kind, size, shared) in enumerate(caches):
            ix = f"{d}/cache/index{i}"
            files |= {f"{ix}/level": str(lvl), f"{ix}/type": kind, f"{ix}/size": size, f"{ix}/shared_cpu_list": shared}
    root = _sysfs(tmp_path / "intel", files)
    groups = sysinfo._cache_groups(sysinfo._linux_caches(root))
    assert groups == [
        {"level": 1, "size": 48 * KB, "threads": 2, "count": 2}, {"level": 1, "size": 32 * KB, "threads": 1, "count": 2},
        {"level": 2, "size": 4 * MB, "threads": 2, "count": 1}, {"level": 2, "size": 2 * MB, "threads": 2, "count": 2},
        {"level": 3, "size": 30 * MB, "threads": 6, "count": 1},
    ]  # fmt: skip
    assert _figures(groups) == _want(
        l1d=(32 * KB, 24 * KB, 160 * KB), l2=(2 * MB, MB, 8 * MB), l3=(30 * MB, 5 * MB, 30 * MB)
    )
    assert sysinfo._linux_topology(root) == {
        "logical": 6, "physical": 4, "base_mhz": 3400, "max_mhz": 5400,
        "classes": [
            {"name": "performance", "cores": 2, "threads": 4, "base_mhz": 3400, "max_mhz": 5400},
            {"name": "efficiency", "cores": 2, "threads": 2, "base_mhz": 2500, "max_mhz": 4200},
        ],
    }  # fmt: skip
    # Arm: four little cores, then two clusters of two big ones whose core ids both count from 0
    files = {f"{cpu}/online": "0-7"}
    for c in range(8):
        d = f"{cpu}/cpu{c}"
        files |= {
            f"{d}/cpu_capacity": "1024" if c >= 4 else "446",
            f"{d}/topology/physical_package_id": "0",
            f"{d}/topology/core_id": str(c if c < 4 else (c - 4) % 2),
            f"{d}/topology/core_cpus_list": str(c),
        }
    topo = sysinfo._linux_topology(_sysfs(tmp_path / "arm", files))
    assert (topo["logical"], topo["physical"]) == (8, 8)
    assert [(k["name"], k["cores"], k["threads"]) for k in topo["classes"]] == [
        ("performance", 4, 4), ("efficiency", 4, 4)
    ]  # fmt: skip
    assert machine._thread_counts(topo) == [1, 2, 4, 8]
    # no sibling lists, no cpufreq, no caches: the package and core ids, one class, nothing made up
    files = {f"{cpu}/online": "0-3"}
    for c in range(4):
        files |= {f"{cpu}/cpu{c}/topology/physical_package_id": "0", f"{cpu}/cpu{c}/topology/core_id": str(c // 2)}
    root = _sysfs(tmp_path / "old", files)
    assert sysinfo._linux_topology(root) == {
        "logical": 4, "physical": 2, "base_mhz": 0, "max_mhz": 0,
        "classes": [{"name": "cores", "cores": 2, "threads": 4, "base_mhz": 0, "max_mhz": 0}],
    }  # fmt: skip
    assert sysinfo._linux_caches(root) == []


def _sysctl(monkeypatch: MonkeyPatch, values: dict[str, int | bytes]) -> None:
    """macOS's sysctl as `values` holds it: the integers and the raw strings and arrays"""
    monkeypatch.setattr(sysinfo, "_sysctl_int", lambda n: v if isinstance(v := values.get(n), int) else None)
    monkeypatch.setattr(sysinfo, "_sysctl_raw", lambda n: v if isinstance(v := values.get(n), bytes) else None)


def _cacheconfig(*share: int) -> bytes:
    """hw.cacheconfig: the logical CPUs sharing memory, then each level's cache, as ten uint64s"""
    return struct.pack("<10Q", *share, *[0] * (10 - len(share)))


def test_macos_sysctl_reads_as_cores_and_caches(monkeypatch: MonkeyPatch) -> None:
    """Apple silicon's perflevels as its performance and efficiency cores, each cluster's L2 counted (hw.l2cachesize
    alone is one cluster's), no L3 and no clock; an Intel Mac as one class with its boost clock and three caches"""
    _sysctl(monkeypatch, {
        "hw.nperflevels": 2, "hw.physicalcpu": 12, "hw.logicalcpu": 12, "hw.l1dcachesize": 64 * KB,
        "hw.l2cachesize": 4 * MB, "hw.cacheconfig": _cacheconfig(12, 1, 4),
        "hw.perflevel0.name": b"Performance\0", "hw.perflevel0.physicalcpu": 8, "hw.perflevel0.logicalcpu": 8,
        "hw.perflevel0.l1dcachesize": 128 * KB, "hw.perflevel0.l2cachesize": 16 * MB, "hw.perflevel0.cpusperl2": 4,
        "hw.perflevel1.name": b"Efficiency\0", "hw.perflevel1.physicalcpu": 4, "hw.perflevel1.logicalcpu": 4,
        "hw.perflevel1.l1dcachesize": 64 * KB, "hw.perflevel1.l2cachesize": 4 * MB, "hw.perflevel1.cpusperl2": 4,
    })  # fmt: skip
    topo = sysinfo._darwin_topology()
    assert (topo["logical"], topo["physical"], topo["max_mhz"]) == (12, 12, 0)
    assert [(k["name"], k["cores"], k["threads"]) for k in topo["classes"]] == [
        ("performance", 8, 8), ("efficiency", 4, 4)
    ]  # fmt: skip
    assert 8 in machine._thread_counts(topo), "the performance cores are a thread count of their own"
    groups = sysinfo._cache_groups(sysinfo._darwin_caches())
    assert groups == [
        {"level": 1, "size": 128 * KB, "threads": 1, "count": 8}, {"level": 1, "size": 64 * KB, "threads": 1, "count": 4},
        {"level": 2, "size": 16 * MB, "threads": 4, "count": 2}, {"level": 2, "size": 4 * MB, "threads": 4, "count": 1},
    ]  # fmt: skip
    assert _figures(groups) == _want(l1d=(64 * KB, 64 * KB, 1280 * KB), l2=(4 * MB, MB, 36 * MB))
    # perflevels that give the L1d and L2 but no L3: the L3 still read, from hw.l3cachesize
    _sysctl(monkeypatch, {
        "hw.nperflevels": 1, "hw.logicalcpu": 4, "hw.l3cachesize": 8 * MB, "hw.cacheconfig": _cacheconfig(4, 1, 1, 4),
        "hw.perflevel0.physicalcpu": 4, "hw.perflevel0.logicalcpu": 4,
        "hw.perflevel0.l1dcachesize": 32 * KB, "hw.perflevel0.l2cachesize": MB, "hw.perflevel0.cpusperl2": 1,
    })  # fmt: skip
    assert sysinfo._cache_groups(sysinfo._darwin_caches()) == [
        {"level": 1, "size": 32 * KB, "threads": 1, "count": 4}, {"level": 2, "size": MB, "threads": 1, "count": 4},
        {"level": 3, "size": 8 * MB, "threads": 4, "count": 1},
    ]  # fmt: skip
    _sysctl(monkeypatch, {
        "hw.physicalcpu": 6, "hw.logicalcpu": 12, "hw.cpufrequency_max": 2_600_000_000,
        "hw.l1dcachesize": 32 * KB, "hw.l2cachesize": 256 * KB, "hw.l3cachesize": 12 * MB,
        "hw.cacheconfig": _cacheconfig(12, 2, 2, 12),
    })  # fmt: skip
    assert sysinfo._darwin_topology() == {
        "logical": 12, "physical": 6, "base_mhz": 0, "max_mhz": 2600,
        "classes": [{"name": "cores", "cores": 6, "threads": 12, "base_mhz": 0, "max_mhz": 2600}],
    }  # fmt: skip
    groups = sysinfo._cache_groups(sysinfo._darwin_caches())
    assert _figures(groups) == _want(
        l1d=(32 * KB, 16 * KB, 192 * KB), l2=(256 * KB, 128 * KB, 1536 * KB), l3=(12 * MB, MB, 12 * MB)
    )


# (level, size, threads sharing it) for every cache instance, as the OS readers return them
_I7_13700K = (
    [(1, 48 * KB, 2)] * 8 + [(1, 32 * KB, 1)] * 8 + [(2, 2 * MB, 2)] * 8 + [(2, 4 * MB, 4)] * 2 + [(3, 30 * MB, 24)]
)
_R9_7950X = [(1, 32 * KB, 2)] * 16 + [(2, MB, 2)] * 16 + [(3, 32 * MB, 16)] * 2
_R9_7950X3D = [(1, 32 * KB, 2)] * 16 + [(2, MB, 2)] * 16 + [(3, 96 * MB, 16), (3, 32 * MB, 16)]
_M2 = [(1, 128 * KB, 1)] * 4 + [(1, 64 * KB, 1)] * 4 + [(2, 16 * MB, 4), (2, 4 * MB, 4)]


def test_each_level_reports_its_instance_share_and_total(monkeypatch: MonkeyPatch) -> None:
    """A thread can run on any core or chiplet, so each level reports its smallest instance (a thread alone) and its
    smallest per-thread share (every thread busy: an instance over the threads sharing it, hyperthreads counted). The
    totals size the bench's pools, which must outgrow every cache."""
    got = {name: _figures(sysinfo._cache_groups(c)) for name, c in (
        ("13700K", _I7_13700K), ("7950X", _R9_7950X), ("7950X3D", _R9_7950X3D), ("M2", _M2))}  # fmt: skip
    # L2 1 MB a thread either way: 2 MB over a P-core's 2 hyperthreads, 4 MB over an E-cluster's 4 cores
    assert got["13700K"] == _want(
        l1d=(32 * KB, 24 * KB, 640 * KB), l2=(2 * MB, MB, 24 * MB), l3=(30 * MB, 1280 * KB, 30 * MB)
    )
    assert got["7950X"] == _want(
        l1d=(32 * KB, 16 * KB, 512 * KB), l2=(MB, 512 * KB, 16 * MB), l3=(32 * MB, 2 * MB, 64 * MB)
    )
    # the smaller chiplet's L3, not the 96 MB V-cache
    assert got["7950X3D"] == _want(
        l1d=(32 * KB, 16 * KB, 512 * KB), l2=(MB, 512 * KB, 16 * MB), l3=(32 * MB, 2 * MB, 128 * MB)
    )
    # the efficiency cluster's L2, 4 MB over its 4 cores; no L3 listed
    assert got["M2"] == _want(l1d=(64 * KB, 64 * KB, 768 * KB), l2=(4 * MB, MB, 20 * MB))
    assert _figures([]) == _want(l1d=(0, 0, 0), l2=(0, 0, 0))
    # the pools: 4x the total L3, while the RAM allows
    monkeypatch.setattr(sysinfo, "host_cache_groups", lambda: sysinfo._cache_groups(_R9_7950X3D))
    monkeypatch.setattr(machine, "host_free", lambda: 64 * 1024 * MB)
    assert machine.host_past() == 512 * MB


def _win_cache(level: int, kind: int, size: int, lps: list[int], groups: int = 1, wide: bool = True) -> bytes:
    """one RelationCache record of GetLogicalProcessorInformationEx (kind 0 unified, 1 instruction, 2 data), its
    processors in group 0; `groups` 0 as Windows before 11 writes it; `wide` False a 32-bit process's (a 4-byte
    mask, 12-byte affinities)"""
    n = 56 if wide else 52
    rec = bytearray(n)
    struct.pack_into("<II", rec, 0, 2, n)
    rec[8] = level
    struct.pack_into("<Ii", rec, 12, size, kind)
    struct.pack_into("<H", rec, 38, groups)
    struct.pack_into("<QH" if wide else "<IH", rec, 40, sum(1 << p for p in lps), 0)
    return bytes(rec)


def test_a_cpu_reads_the_same_on_every_os(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """one CPU - two chiplets of two hyperthreaded cores, a 32 KB L1d and a 1 MB L2 a core, a 32 MB L3 a chiplet -
    as Linux's sysfs, Windows' processor records (a 64-bit process's and a 32-bit one's) and macOS's sysctl (with
    perflevels and without) describe it, read to the same caches: each L3 counted, 64 MB in all"""
    want = [
        {"level": 1, "size": 32 * KB, "threads": 2, "count": 4},
        {"level": 2, "size": MB, "threads": 2, "count": 4},
        {"level": 3, "size": 32 * MB, "threads": 4, "count": 2},
    ]
    cpu = "devices/system/cpu"
    files: dict[str, str] = {}
    for c in range(8):
        core, ccd = f"{c - c % 2}-{c - c % 2 + 1}", "0-3" if c < 4 else "4-7"
        caches = [(1, "Data", "32K", core), (1, "Instruction", "32K", core), (2, "Unified", "1024K", core),
                  (3, "Unified", "32768K", ccd)]  # fmt: skip
        for i, (lvl, kind, size, shared) in enumerate(caches):
            ix = f"{cpu}/cpu{c}/cache/index{i}"
            files |= {f"{ix}/level": str(lvl), f"{ix}/type": kind, f"{ix}/size": size, f"{ix}/shared_cpu_list": shared}
    linux = sysinfo._cache_groups(sysinfo._linux_caches(_sysfs(tmp_path, files)))
    per_core = [(1, 2, 32 * KB), (1, 1, 32 * KB), (2, 0, MB)]
    raw = b"".join(_win_cache(lvl, kind, size, [2 * k, 2 * k + 1]) for k in range(4) for lvl, kind, size in per_core)
    raw += _win_cache(3, 0, 32 * MB, [0, 1, 2, 3]) + _win_cache(3, 0, 32 * MB, [4, 5, 6, 7], groups=0)
    raw += _win_cache(3, 0, 32 * MB, [4, 5, 6, 7])  # the same cache listed twice is one cache
    windows = sysinfo._cache_groups(sysinfo._win_caches(raw))
    raw32 = b"".join(
        _win_cache(lvl, kind, size, [2 * k, 2 * k + 1], wide=False) for k in range(4) for lvl, kind, size in per_core
    )
    raw32 += _win_cache(3, 0, 32 * MB, [0, 1, 2, 3], wide=False) + _win_cache(3, 0, 32 * MB, [4, 5, 6, 7], wide=False)
    windows32 = sysinfo._cache_groups(sysinfo._win_caches(raw32, wide=False))
    # macOS: hw.cacheconfig counts logical CPUs; a hyperthreaded Mac's perflevels, whose cpusperl2 here counts
    # cores, are not read for the sharing
    mac: dict[str, int | bytes] = {
        "hw.logicalcpu": 8, "hw.cacheconfig": _cacheconfig(8, 2, 2, 4),
        "hw.l1dcachesize": 32 * KB, "hw.l2cachesize": MB, "hw.l3cachesize": 32 * MB,
    }  # fmt: skip
    _sysctl(monkeypatch, mac)
    old_mac = sysinfo._cache_groups(sysinfo._darwin_caches())
    _sysctl(monkeypatch, {
        **mac, "hw.nperflevels": 1, "hw.perflevel0.physicalcpu": 4, "hw.perflevel0.logicalcpu": 8,
        "hw.perflevel0.l1dcachesize": 32 * KB, "hw.perflevel0.l2cachesize": MB, "hw.perflevel0.cpusperl2": 1,
        "hw.perflevel0.l3cachesize": 32 * MB, "hw.perflevel0.cpusperl3": 2,
    })  # fmt: skip
    new_mac = sysinfo._cache_groups(sysinfo._darwin_caches())
    assert linux == windows == windows32 == old_mac == new_mac == want, (linux, windows, windows32, old_mac, new_mac)
    assert _figures(want) == _want(
        l1d=(32 * KB, 16 * KB, 128 * KB), l2=(MB, 512 * KB, 4 * MB), l3=(32 * MB, 8 * MB, 64 * MB)
    )
    assert machine._cache_lines(want) == [
        "           L1d 4 x 32 KB (2 threads each); 16 KB a thread with all busy",
        "           L2  4 x 1 MB (2 threads each) = 4 MB; 512 KB a thread with all busy",
        "           L3  2 x 32 MB (4 threads each) = 64 MB; 8 MB a thread with all busy",
    ]
    assert machine._exact(1280 * KB) == "1.25 MB", "an Alder Lake P-core's L2, not rounded to 1 MB"


def test_the_cores_add_up() -> None:
    """the classes' cores and threads are the CPU's, the fastest class first; a cpulist reads as its CPUs"""
    assert sysinfo._cpulist("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    topo = sysinfo.host_cpu_topology()
    if not topo:
        pytest.skip("the OS gave no core topology")
    assert 0 < topo["physical"] <= topo["logical"]
    assert sum(c["cores"] for c in topo["classes"]) == topo["physical"]
    assert sum(c["threads"] for c in topo["classes"]) == topo["logical"]
    counts = machine._thread_counts(topo)
    assert counts[0] == 1 and counts[-1] == topo["logical"] and topo["physical"] in counts
    # this machine's caches: no thread under two instances of a level (a VM may list fewer threads than it has)
    groups = sysinfo.host_cache_groups()
    for lvl in (1, 2, 3):
        assert sum(g["threads"] * g["count"] for g in groups if g["level"] == lvl) <= topo["logical"], groups
    if groups:
        c = sysinfo.host_cache_sizes()
        assert 0 < c[HostCache.L1D_SHARE] <= c[HostCache.L1D_INSTANCE], groups


def test_the_features_hold_the_tier_the_library_picked() -> None:
    """the native library's feature list carries what its kernel tier needs"""
    from btb.engine.native import cpu_features, isa

    feats = cpu_features()
    if feats is None:
        pytest.skip("no native library here")
    need = {"avx512": {"avx512f", "avx512bw", "avx2", "fma"}, "avx2": {"avx2", "fma"}, "neon": {"neon"}}
    assert need.get(isa(), set()) <= set(feats), (isa(), feats)


def test_a_drive_is_written_read_and_left_as_it_was(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the drive section in small: MB writes, each flushed, then btb's reader on the file and the rule its Route
    draws from the reads - and the folder empty afterwards. A filesystem that refuses unbuffered reads (tmpfs)
    gives failed rows, never a broken report"""
    import shutil

    from btb.engine.native import Native

    monkeypatch.setattr(machine, "CHUNK", MB)
    monkeypatch.setattr(machine, "DRIVE_KEEP", 4 * MB)
    # the drive as 24 MB free: the file may take 20, every write together, whatever the sampler does
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(machine.shutil, "disk_usage", lambda _p: usage._replace(free=24 * MB))
    written = [0]
    write = os.write

    def counted(fd: int, data: bytes | memoryview) -> int:
        n = write(fd, data)
        written[0] += n
        return n

    monkeypatch.setattr(machine.os, "write", counted)
    lines: list[str] = []
    rep = machine.Report(machine.Sampler(ci=0.0, budget_s=0.2, min_n=3), lines.append)
    machine.drive(rep, str(tmp_path))
    assert os.listdir(tmp_path) == [], "the test file is removed"
    assert 0 < written[0] <= 20 * MB, f"{written[0] / MB:.0f} MB written of the 20 the drive could spare"
    writes = [r for r in rep.rows if r.op.startswith("write")]
    assert len(writes) == 2 and all(r.t is not None for r in writes), lines
    if Native.read_at is None:
        assert any("reader is not here" in r.note for r in rep.rows)
        return
    alone = {r.at["bytes"] for r in rep.rows if r.op.startswith("read, one in flight")}
    assert {4096, MB} <= alone, lines
    deep = [(r.at["bytes"], r.at["depth"]) for r in rep.rows if r.at.get("depth", 1) > 1]
    assert sorted(d for _, d in deep) == [4, 4, 16, 16] and 4096 in {b for b, _ in deep}, lines
    reads = [r for r in rep.rows if r.op.startswith("read")]
    if all(r.t is not None for r in reads):
        rule = rep.found[f"drive {tmp_path}"]
        assert rule["depth"] in (1, 4, 16) and rule["ahead"] >= 0, rule


def _small(monkeypatch: MonkeyPatch) -> tuple[machine.Report, list[str]]:
    """the report's sweeps cut to sizes a unit test affords, and a report whose every point is a few samples"""
    small = {
        "HOST_SETS": (32 * KB, MB, 4 * MB), "SQUARES": (512, 1024), "MLX_LARGER": (), "LINEAR": (256, 512),
        "EXPERT": (256, 512), "MX4": (256, 512), "HEADS": (4, 2, 128), "CONTEXTS": (512, 2048), "VOCABS": (32000,),
    }  # fmt: skip
    for name, v in small.items():
        monkeypatch.setattr(machine, name, v)
    monkeypatch.setattr(machine, "host_past", lambda: 8 * MB)
    # a 1.25 MB L2 a thread: the read sweep adds the sets either side of it
    l2 = [{"level": 2, "size": 1280 * KB, "threads": 1, "count": 1}]
    monkeypatch.setattr(machine.sysinfo, "host_cache_groups", lambda: l2)
    lines: list[str] = []
    return machine.Report(machine.Sampler(ci=0.5, budget_s=0.05, min_n=3, sample_s=0.0005), lines.append), lines


def test_the_host_sections_run_every_op(monkeypatch: MonkeyPatch) -> None:
    """the memory and cpu sections at small sizes on this machine's OS and native library - each platform CI
    tests a wheel on, Linux and macOS among them: every op is timed, or its row says the library lacks it; none
    fails"""
    from btb.engine.native import Native

    rep, lines = _small(monkeypatch)
    machine.memory(rep, {"memory": {"peak_gb_s": None}, "cpu": sysinfo.host_cpu_topology()})
    machine.cpu_ops(rep)
    assert not [r for r in rep.rows if r.note.startswith("failed")], "\n".join(lines)
    timed = {r.op.split(",")[0] for r in rep.rows if r.t is not None}
    assert {"read (1 thread", "copy", "attention prefill f32 (torch)"} <= timed, timed
    sets = [r.at["bytes"] for r in rep.rows if r.op.startswith("read (1 thread")]
    assert sets == [32 * KB, 640 * KB, MB, 2560 * KB, 4 * MB], "the L2's edge between two points"
    if Native.gemv is not None:
        assert {"gemv bf16 (native)", "gemv bf16 stream (native)"} <= timed, timed
        assert "host matmul" in rep.found, "\n".join(lines)


def test_a_pool_past_the_room_is_refused_before_any_copy_is_made(monkeypatch: MonkeyPatch) -> None:
    """a pool whose copies would take more than 1/8 of the free RAM raises NoRoom and makes nothing, counting what a
    copy holds where that is more than a call reads"""
    monkeypatch.setattr(machine, "host_free", lambda: 8 * MB)  # room: 1 MB
    made: list[int] = []

    def make() -> int:
        made.append(1)
        return len(made)

    assert len(machine._pool(make, 256 * KB, 512 * KB)) == 2
    assert len(machine._pool(make, 768 * KB, 512 * KB)) == 1, "one copy past `past` is enough on its own"
    made.clear()
    with pytest.raises(machine.NoRoom):
        machine._pool(make, 1200 * KB, 0)
    with pytest.raises(machine.NoRoom):
        machine._pool(make, 4 * KB, 0, size=1200 * KB)  # a cache longer than the context it is read over
    assert made == []


def test_a_point_past_the_room_is_skipped_and_its_sweep_goes_on(monkeypatch: MonkeyPatch) -> None:
    """the experts' group sweep with RAM for 4 and 16 experts' pools but not 64's: the two are timed and 64's row says
    it was skipped, where its allocation once ran the machine into swap or ended the report"""
    from btb.engine.native import Native

    if Native.gemv_group is None:
        pytest.skip("needs the native library's grouped gemv")
    rep, lines = _small(monkeypatch)
    monkeypatch.setattr(machine, "host_free", lambda: 96 * MB)  # room: 12 MB; 64 experts' one copy needs 16 MB
    machine._cpu_group(rep, 8 * MB)
    timed = [r.shape for r in rep.rows if r.t is not None]
    skipped = {r.shape: r.note for r in rep.rows if r.t is None}
    assert timed == ["4 experts", "16 experts"], "\n".join(lines)
    assert list(skipped) == ["64 experts"] and skipped["64 experts"].startswith("skipped:"), "\n".join(lines)


# the host sections and a drive at full size in a process of their own, its free RAM pinned (argv[1] bytes): each
# row and how far the process's peak grew past where it stood with torch and the native library loaded
SMALL_MACHINE = """
import json, sys
import torch
from btb import machine, sysinfo
from btb.engine.native import Native

free = int(sys.argv[1])
machine.host_free = lambda: free
f = machine.facts([])
_ = Native.gemv
torch.randn(256, 256) @ torch.randn(256, 256)
base = sysinfo.peak_rss_bytes()
rep = machine.Report(machine.Sampler(ci=0.5, budget_s=0.05, min_n=3, sample_s=0.0005), lambda line: None)
machine.memory(rep, f)
machine.cpu_ops(rep)
machine.drive(rep, sys.argv[2])
rows = [{"op": r.op, "shape": r.shape, "timed": r.t is not None, "note": r.note} for r in rep.rows]
print(json.dumps({"grown": sysinfo.peak_rss_bytes() - base, "rows": rows}))
"""


@pytest.mark.parametrize("free_mb", [1024, 512])
def test_the_report_runs_on_a_4gb_machine_without_a_card(tmp_path: Path, free_mb: int) -> None:
    """the host sections and the drive at full size with the RAM a 4 GB machine has free once its OS and torch are in
    (1 GB, and 512 MB when it is busy): every row timed, or skipped for want of room; none fails, and the report's own
    peak stays under half of what is free"""
    free = free_mb * MB
    root = Path(__file__).resolve().parents[2]  # `import btb`: the checkout's, or the installed wheel's beside it
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    cmd = [sys.executable, "-c", SMALL_MACHINE, str(free), str(tmp_path)]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=root, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert not [r for r in got["rows"] if r["note"].startswith("failed")], got["rows"]
    assert sum(r["timed"] for r in got["rows"]) > 50, got["rows"]
    assert got["grown"] < free // 2, f"the report's peak grew {got['grown'] / MB:.0f} MB with {free_mb} MB free"


def _card_kernels() -> Any:
    from btb.engine.native import Native

    k = Native.card_kernels() if torch.cuda.is_available() else None
    if k is None:
        pytest.skip("needs a CUDA card and btb's kernels")
    return k


def test_a_graph_body_holds_its_tensors() -> None:
    """a graph body a card helper hands back keeps its tensors alive: a pointer is a bare address, and a body that kept
    only addresses read freed memory once its helper returned (the prefill attention's did)"""
    k = _card_kernels()
    dev = torch.device("cuda", torch.cuda.current_device())
    R, C = 256, 512

    def bf(*s: int) -> torch.Tensor:
        return torch.zeros(*s, dtype=torch.bfloat16, device=dev)

    makers = {
        "gemv": lambda: machine._gemv_body(k, [bf(R, C)], bf(1, C), bf(1, R), R, C, 1),
        "mma": lambda: machine._mma_body(k, [bf(R, C)], bf(32, C), bf(32, R), R, C, 1),
        "mma form": lambda: machine._mma_form_body(k, [bf(R, C)], bf(32, C), bf(32, R), R, C, True),
        "prefill": lambda: machine._prefill_body(k, dev, 512),
    }
    for name, make in makers.items():
        gc.collect()
        before = torch.cuda.memory_allocated(dev)
        body = make()
        gc.collect()
        assert torch.cuda.memory_allocated(dev) > before, f"{name}: its tensors went with the helper"
        body()
        torch.cuda.synchronize()
        del body


def test_the_card_section_runs_every_op(monkeypatch: MonkeyPatch) -> None:
    """the card section at small sizes through btb's kernels: every op timed, or its row says the build lacks it;
    none fails"""
    _card_kernels()
    rep, lines = _small(monkeypatch)
    small = {"MLP": (512, 1024), "SHORT": (256, 512), "DELTA": (2, 4, 64, 64, 4), "CARD_SETS": (MB, 4 * MB)}
    for name, v in small.items():
        monkeypatch.setattr(machine, name, v)
    card = {"name": "the card", "l2": 0, "peak_gb_s": None}
    machine.card_ops(rep, {"cards": [card]}, torch.cuda.current_device())
    assert not [r for r in rep.rows if r.note.startswith("failed")], "\n".join(lines)
    timed = {r.op.split(",")[0] for r in rep.rows if r.t is not None}
    assert {"attention prefill bf16", "residual add and rmsnorm", "q/k norm"} <= timed, timed


def test_the_bus_section_runs_every_op_and_a_card_without_room_skips_the_copies(monkeypatch: MonkeyPatch) -> None:
    """the bus section at small sizes: every copy and the round trip timed. Then with the card's room too small for
    any copy: the link read and the copies each leave a skipped row, and the round trip still runs"""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA card")
    rep, lines = _small(monkeypatch)
    monkeypatch.setattr(machine, "TRANSFERS", (4 * KB, MB))
    i = torch.cuda.current_device()
    f: Json = {"cards": [{"bus_id": None}] * (i + 1)}
    machine.bus(rep, f, i)
    assert all(r.t is not None for r in rep.rows), "\n".join(lines)
    ops = {r.op for r in rep.rows}
    assert {"to the card, pinned", "from the card, pageable", "a row's round trip"} <= ops, ops
    rep, lines = _small(monkeypatch)
    monkeypatch.setattr(machine, "card_free", lambda dev: 2 * KB)  # room: 512 bytes
    machine.bus(rep, f, i)
    skipped = {r.op: r.note for r in rep.rows if r.t is None}
    assert list(skipped) == ["link", "copies"], "\n".join(lines)
    assert all(n.startswith("skipped:") for n in skipped.values()), skipped
    assert [r.op for r in rep.rows if r.t is not None] == ["a row's round trip"], "\n".join(lines)


def test_the_mlx_section_runs_every_op(monkeypatch: MonkeyPatch) -> None:
    """Apple's GPU section at small sizes, through btb's Metal kernels: every op timed"""
    from btb.engine.device import mlx_available

    if not mlx_available():
        pytest.skip("MLX is not available here")
    rep, lines = _small(monkeypatch)
    machine.mlx_ops(rep)
    assert rep.rows and all(r.t is not None for r in rep.rows), "\n".join(lines)
    assert len({r.op for r in rep.rows}) == 2, "the gemv and the attention"


def test_the_mlx_section_skips_what_the_free_ram_cannot_hold(monkeypatch: MonkeyPatch) -> None:
    """Apple's GPU section with 8 GB free (a 1 GB room), through a stand-in for MLX on any OS: the 16384-square
    weight and the 128k-row caches (each 1.5 GB with the float32 values they are made from) are skipped before
    anything is made, and every other point is timed"""
    import importlib
    import types

    # the modules themselves: btb.mlx re-exports `gemv` the function under the module's name
    core, gemv, attn = (importlib.import_module(f"btb.mlx.{n}") for n in ("core", "gemv", "attn"))
    made: list[tuple[int, ...]] = []

    class Array:
        def astype(self, dtype: str) -> Array:
            return self

    def normal(shape: tuple[int, ...]) -> Array:
        made.append(tuple(shape))
        return Array()

    m = types.SimpleNamespace(
        bfloat16="bf16", float32="f32", random=types.SimpleNamespace(normal=normal),
        ones=lambda shape, dtype=None: Array(), eval=lambda *a: None, clear_cache=lambda: None,
    )  # fmt: skip
    monkeypatch.setattr(core, "mx", lambda: m)
    monkeypatch.setattr(gemv, "gemv", lambda w, x: None)
    monkeypatch.setattr(attn, "attn_decode", lambda q, k, v, n, scale: None)
    monkeypatch.setattr(machine, "host_free", lambda: 8 * 1024 * MB)
    lines: list[str] = []
    rep = machine.Report(machine.Sampler(ci=0.5, budget_s=0.01, min_n=3, sample_s=0.0001), lines.append)
    machine.mlx_ops(rep)
    skipped = {r.shape: r.note for r in rep.rows if r.t is None}
    assert set(skipped) == {"16384x16384 (512 MB)", "131072 rows"}, "\n".join(lines)
    assert all(n.startswith("skipped:") for n in skipped.values()), skipped
    assert (16384, 16384) not in made and (1, 8, 131072, 128) not in made, "a skipped point made nothing"
    assert sum(r.t is not None for r in rep.rows) == len(machine.SQUARES) + len(machine.CONTEXTS) - 1


def test_bench_is_the_machine_report_without_a_model(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """no model: the report, its JSON alone on stdout (every section skipped here, the facts still read); a model
    without --prompts is refused by its flag; an unknown section is refused"""
    from btb.cli import main

    skip = ",".join(machine.SECTIONS)
    assert main(["bench", "--json", "--skip", skip]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["timings"] == [] and out["sampler"]["budget_s"] == 15.0 and out["sampler"]["ci"] == 0.02
    assert out["machine"]["cpu"]["logical"] >= 1 and out["machine"]["memory"]["total"] > 0
    assert main(["bench", "no/such-model"]) == 2
    assert "--prompts" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["bench", "--skip", "gpu"])
    # each form refuses the other's flags rather than leave them unread, and the sampler's settings their nonsense
    assert main(["bench", "--prompts", __file__]) == 2
    assert "--prompts: times a model's prompt set" in capsys.readouterr().err
    assert main(["bench", "no/such-model", "--prompts", __file__, "--skip", "drive"]) == 2
    assert "--skip: is the machine report's" in capsys.readouterr().err
    # a model's load options too, but the device the report times, which it reads
    for flag in (["--cpu-layers", "2"], ["--temperature", "0.7"], ["--verbose"]):
        assert main(["bench", *flag]) == 2
        assert f"{flag[0]}: sets how a model loads" in capsys.readouterr().err
    assert main(["bench", "--json", "--skip", skip, "-d", "cpu"]) == 0
    capsys.readouterr()
    # an empty prompt set is refused before the model loads
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert main(["bench", "no/such-model", "--prompts", str(empty)]) == 2
    assert "no prompts in it" in capsys.readouterr().err
    for bad in (["--ci", "0"], ["--ci", "1.5"], ["--max-seconds", "0"], ["--max-seconds", "-3"]):
        with pytest.raises(SystemExit):
            main(["bench", *bad])


def test_a_models_bench_prints_its_record_as_json(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """A model bench with --json prints its record to stdout, the same record --out writes, with progress on
    stderr."""
    from btb.cli import main

    fx = fixture("tiny_q35")
    prompts, out = tmp_path / "prompts.jsonl", tmp_path / "out.jsonl"
    prompts.write_text('{"prompt": "hi"}\n', encoding="utf-8")
    rc = main(["bench", fx, "-d", "cpu", "--prompts", str(prompts), "--new", "4", "--json", "--out", str(out)])
    got = capsys.readouterr()
    assert rc == 0, got.err
    record = json.loads(got.out)
    assert record["label"] == "tiny_q35" and [c["new"] for c in record["cells"]] == [4], record
    assert json.loads(out.read_text(encoding="utf-8")) == record
    assert "[bench] tiny_q35 new=4" in got.err
