# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""OS-level measurement: what the machine has and what is free right now. Stateless queries (Windows through
ctypes, Linux from /proc and sysfs, macOS from sysctlbyname), no torch. The card's free VRAM as torch's
allocator sees it is `engine/device.py`'s, not here."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import re
import struct
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import pool
from .kinds import HostCache, Json

# macOS: sysctlbyname through libc, bound once - a read is microseconds, where spawning `sysctl` from a process
# with the model mapped was milliseconds a call inside the decode loop's once-a-second policy check
_SYSCTL: list[Any] = []


def _sysctl_int(name: str) -> int | None:
    """The integer value of a macOS sysctl by name, or None where the name is unknown or this is not macOS."""
    if sys.platform != "darwin":
        return None
    if not _SYSCTL:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libSystem.B.dylib")
        fn = libc.sysctlbyname
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        fn.restype = ctypes.c_int
        _SYSCTL.append(fn)
    val = ctypes.c_int64(0)
    size = ctypes.c_size_t(ctypes.sizeof(val))
    if _SYSCTL[0](name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0) != 0:
        return None
    return int(val.value) if size.value == 8 else int(ctypes.c_int32.from_buffer_copy(val).value)


def _sysctl_raw(name: str) -> bytes | None:
    """The raw bytes of a macOS sysctl by name (a string or a struct), or None."""
    if sys.platform != "darwin":
        return None
    _sysctl_int("hw.ncpu")  # binds the function
    size = ctypes.c_size_t(0)
    if _SYSCTL[0](name.encode(), None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if _SYSCTL[0](name.encode(), buf, ctypes.byref(size), None, 0) != 0:
        return None
    return buf.raw[: size.value]


@lru_cache(maxsize=1)
def host_cpu_name() -> str:
    """The CPU's model name, as the OS reports it (read once)."""
    if sys.platform == "darwin":
        raw = _sysctl_raw("machdep.cpu.brand_string")
        return raw.split(b"\0", 1)[0].decode(errors="replace").strip() if raw else ""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return ""
    if sys.platform == "win32":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
            if name:
                return name
        except OSError:
            pass
    import platform

    return platform.processor()


def host_cores() -> int:
    """The CPU cores: physical on macOS, the OS's count elsewhere."""
    if sys.platform == "darwin":
        return int(_sysctl_int("hw.physicalcpu") or 0)
    return int(os.cpu_count() or 0)


@lru_cache(maxsize=1)
def _host_simd() -> tuple[str, ...]:
    import platform

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64") or machine.startswith("arm"):
        return ("NEON",)
    if sys.platform == "win32":
        f = ctypes.windll.kernel32.IsProcessorFeaturePresent
        f.argtypes = [ctypes.c_ulong]
        f.restype = ctypes.c_int
        for name, pf in (("AVX-512", 41), ("AVX2", 40), ("AVX", 39)):
            if f(pf):
                return (name,)
        return ()
    if sys.platform == "linux":
        try:
            flags: set[str] = set()
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith(("flags", "Features")):
                        flags = set(line.split(":", 1)[1].split())
                        break
        except OSError:
            return ()
        if "neon" in flags or "asimd" in flags:
            return ("NEON",)
        for name, flag in (("AVX-512", "avx512f"), ("AVX2", "avx2"), ("AVX", "avx")):
            if flag in flags:
                return (name,)
        return ()
    if sys.platform == "darwin":
        for name, key in (("AVX-512", "hw.optional.avx512f"), ("AVX2", "hw.optional.avx2_0")):
            if _sysctl_int(key):
                return (name,)
    return ()


def host_simd() -> list[str]:
    """The SIMD instruction set the native CPU kernels use here: the top of AVX-512 / AVX2 / AVX on x86, NEON on
    ARM (detected once). Empty where none is detected."""
    return list(_host_simd())


def raise_file_limit(want: int = 65536) -> int:
    """Lift this process's open-file soft limit to `want` (or the hard limit, whichever is lower) and return the
    limit in force. macOS starts a process at 256, which the drive readers (a handle per thread per shard) pass on
    a 15-shard model; Windows has no such limit and returns 0."""
    if sys.platform == "win32":
        return 0
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    top = int(want) if hard == resource.RLIM_INFINITY else min(int(want), int(hard))
    if soft < top:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (top, hard))
            soft = top
        except (ValueError, OSError):
            pass
    return int(soft)


def swap_used_bytes() -> int:
    """Swap in use: macOS's vm.swapusage, Linux's SwapTotal - SwapFree; 0 on Windows (its page file has no
    such figure) and where nothing can be read."""
    if sys.platform == "darwin":
        # struct xsw_usage: total, avail, used (u64 each), pagesize, encrypted
        raw = _sysctl_raw("vm.swapusage")
        return int.from_bytes(raw[16:24], sys.byteorder) if raw and len(raw) >= 24 else 0
    if sys.platform == "linux":
        try:
            total = free = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("SwapFree:"):
                        free = int(line.split()[1]) * 1024
            return max(0, total - free)
        except (OSError, ValueError):
            return 0
    return 0


def memory_level() -> float:
    """macOS's kern.memorystatus_level: the free-memory percentage its own killer reads; 100 elsewhere."""
    if sys.platform == "darwin":
        lvl = _sysctl_int("kern.memorystatus_level")
        return float(lvl) if lvl is not None else 100.0
    return 100.0


class _MemoryStatusEx(ctypes.Structure):
    """Windows GlobalMemoryStatusEx's MEMORYSTATUSEX: physical, page-file and virtual totals and what is free."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _memory_status() -> _MemoryStatusEx:
    if sys.platform != "win32":
        raise RuntimeError("[sysinfo] GlobalMemoryStatusEx is a Windows call")
    ms = _MemoryStatusEx()
    ms.dwLength = ctypes.sizeof(_MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms


def host_free_bytes() -> int:
    if sys.platform == "win32":
        return int(_memory_status().ullAvailPhys)
    if sys.platform == "darwin":
        return _darwin_available_bytes()
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _darwin_available_bytes() -> int:
    # in pool.py, which needs it before torch is imported
    return pool.darwin_available_bytes()


def host_total_bytes() -> int:
    if sys.platform == "win32":
        return int(_memory_status().ullTotalPhys)
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def host_commit_bytes() -> int:
    if sys.platform != "win32":
        return host_free_bytes()
    return int(_memory_status().ullAvailPageFile)


def _process_counters() -> Any:
    """Windows' PROCESS_MEMORY_COUNTERS for this process through GetProcessMemoryInfo, or None off Windows and on
    a failed read: the one struct behind the peak working set and the page-fault count."""
    if sys.platform == "win32":

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = k32.K32GetProcessMemoryInfo
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        fn.restype = ctypes.c_int
        return pmc if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) else None
    return None


def peak_rss_bytes() -> int:
    """the process's peak resident set (peak working set) in bytes: Windows through GetProcessMemoryInfo, else
    getrusage's ru_maxrss (bytes on macOS, KiB on Linux). The one reader the engine's report and the bench
    comparison both call, so the struct lives in one place."""
    if sys.platform == "win32":
        pmc = _process_counters()
        return int(pmc.PeakWorkingSetSize) if pmc is not None else 0
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) * (1 if sys.platform == "darwin" else 1024)


def page_faults() -> int:
    """the page faults this process has taken so far: every fault, the soft and demand-zero ones included
    (Windows through GetProcessMemoryInfo; elsewhere getrusage's minor and major together). The disk's share
    of them is `hard_page_faults`."""
    if sys.platform == "win32":
        pmc = _process_counters()
        return int(pmc.PageFaultCount) if pmc is not None else 0
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF)
    return int(r.ru_minflt) + int(r.ru_majflt)


def process_working_set_bytes() -> int:
    """the process's resident set right now: Windows through GetProcessMemoryInfo, Linux from /proc/self/statm,
    macOS the peak (getrusage keeps no current figure). The room this process has already taken - the
    interpreter, torch, a CUDA context - which a free-memory reading has already left out."""
    if sys.platform == "win32":
        pmc = _process_counters()
        return int(pmc.WorkingSetSize) if pmc is not None else 0
    if sys.platform == "linux":
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return 0
    return peak_rss_bytes()


def process_read_bytes() -> int:
    """bytes this process has read through the file APIs so far: Windows through GetProcessIoCounters
    (ReadTransferCount), Linux from /proc/self/io (read_bytes, what reached the storage layer), else 0. The ring's
    and the store's reads as the OS counted them; the memory manager's paging of a mapped file is not among them,
    so the drive's total less this figure is the paging."""
    if sys.platform == "win32":

        class IOC(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        ioc = IOC()
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        fn = k32.GetProcessIoCounters
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(IOC)]
        fn.restype = ctypes.c_int
        return int(ioc.ReadTransferCount) if fn(k32.GetCurrentProcess(), ctypes.byref(ioc)) else 0
    if sys.platform == "linux":
        try:
            with open("/proc/self/io") as f:
                for line in f:
                    if line.startswith("read_bytes:"):
                        return int(line.split()[1])
        except OSError:
            pass
    return 0


def hard_page_faults() -> int:
    """the page faults this process resolved from the disk - mapped weights the OS trimmed and read back - against
    `page_faults()`, which counts the soft and demand-zero ones too. Windows keeps the figure in the system's
    process table (SYSTEM_PROCESS_INFORMATION.HardFaultCount): the table is read whole, this process's entry is
    taken by its id and nothing else in it is looked at, and the entry is trusted only where its own total fault
    count agrees with GetProcessMemoryInfo's - a layout this reader does not expect gives 0, never a number.
    Elsewhere getrusage's major faults."""
    if sys.platform != "win32":
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_majflt)
    try:
        ntdll = ctypes.windll.ntdll
    except OSError:
        return 0
    total = page_faults()
    size = 1 << 18
    for _ in range(8):
        buf = ctypes.create_string_buffer(size)
        need = ctypes.c_ulong(0)
        status = int(ntdll.NtQuerySystemInformation(5, buf, size, ctypes.byref(need))) & 0xFFFFFFFF
        if status == 0xC0000004:  # STATUS_INFO_LENGTH_MISMATCH: the table has outgrown the buffer
            size = max(size * 2, int(need.value) + (1 << 16))
            continue
        if status != 0:
            return 0
        break
    else:
        return 0
    pid = os.getpid()
    raw = buf.raw
    off = 0
    while off + 136 <= len(raw):
        nxt = int.from_bytes(raw[off : off + 4], "little")
        if int.from_bytes(raw[off + 80 : off + 88], "little") == pid:
            hard = int.from_bytes(raw[off + 16 : off + 20], "little")
            faults = int.from_bytes(raw[off + 128 : off + 132], "little")
            # the same total GetProcessMemoryInfo gave moments ago: a disagreement past what a few steps fault
            # is a layout this reader has wrong
            if hard <= faults and 0 <= faults - total < (1 << 20):
                return hard
            return 0
        if nxt == 0:
            break
        off += nxt
    return 0


def os_memory_floor() -> int:
    """The memory the OS itself keeps free, where it publishes the figure: Linux's zone high watermarks
    (/proc/zoneinfo - what the kernel reclaims towards before allocations start to stall). Windows and macOS
    publish no threshold; their word is the pressure signal (`memory_pressure`), read live, so 0 here: nothing
    is presumed on their behalf."""
    if sys.platform != "linux":
        return 0
    try:
        pages = 0
        with open("/proc/zoneinfo") as f:
            for line in f:
                s = line.split()
                if len(s) == 2 and s[0] == "high":
                    pages += int(s[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 0


_MEM_NOTIFY: dict[str, Any] = {}


def memory_pressure() -> Json:
    """The OS's own word on memory pressure right now: `low` when it says memory is short, `level` its figure.
    Windows: the low-memory resource notification (QueryMemoryResourceNotification; level 1.0 when signaled).
    Linux: /proc/pressure/memory - `level` the share of the last ten seconds some task stalled on memory,
    `low` when every task did (`full`). macOS: kern.memorystatus_vm_pressure_level (1 normal, 2 warning,
    4 critical; low from warning). {"low": False, "level": 0.0} where nothing can be read."""
    out: Json = {"low": False, "level": 0.0}
    try:
        if sys.platform == "win32":
            k32 = ctypes.windll.kernel32
            h = _MEM_NOTIFY.get("low")
            if h is None:
                k32.CreateMemoryResourceNotification.restype = ctypes.c_void_p
                h = k32.CreateMemoryResourceNotification(0)  # LowMemoryResourceNotification
                _MEM_NOTIFY["low"] = h
            if h:
                state = ctypes.c_int(0)
                fn = k32.QueryMemoryResourceNotification
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
                fn.restype = ctypes.c_int
                if fn(h, ctypes.byref(state)):
                    out["low"] = bool(state.value)
                    out["level"] = 1.0 if state.value else 0.0
        elif sys.platform == "linux":
            with open("/proc/pressure/memory") as f:
                for line in f:
                    kind, *rest = line.split()
                    avg10 = 0.0
                    for tok in rest:
                        k, _, v = tok.partition("=")
                        if k == "avg10":
                            avg10 = float(v) / 100.0
                    if kind == "some":
                        out["level"] = avg10
                    elif kind == "full" and avg10 > 0.0:
                        out["low"] = True
        elif sys.platform == "darwin":
            lvl = _sysctl_int("kern.memorystatus_vm_pressure_level")
            if lvl is not None:
                out["level"] = float(lvl)
                out["low"] = lvl >= 2
    except Exception:
        pass
    return out


# -- the GPU memory the OS reports for this process and the adapters it shares, through Windows PDH ----------
# opening a query and adding its four counters cost ~4 ms a call - longer than a small model's whole decode
# step - where a collect on a kept query is a fraction of a millisecond; the query is kept while an engine is on a
# card and closed with the last (`vram_pressure_close`): its counter providers hold tens of MB of the process
_PDH_FMT = 0x00000400 | 0x00001000
_PDH: dict[int, Any] = {}
_PDH_LOCK = threading.Lock()  # a read on a watcher's thread and the close on the engine's never cross


class _PdhV(ctypes.Union):
    _fields_ = [("l", ctypes.c_long), ("d", ctypes.c_double), ("ll", ctypes.c_longlong), ("s", ctypes.c_void_p)]


class _PdhCV(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("v", _PdhV)]


class _PdhItem(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhCV)]


def _pdh_open(pid: int) -> Any:
    if sys.platform != "win32":
        return None
    try:
        pdh = ctypes.windll.pdh
    except OSError:
        return None
    q = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(q)) != 0:
        return None
    paths = {
        ("adapters", "dedicated"): r"\GPU Adapter Memory(*)\Dedicated Usage",
        ("adapters", "shared"): r"\GPU Adapter Memory(*)\Shared Usage",
        ("process", "dedicated"): rf"\GPU Process Memory(pid_{pid}_*)\Dedicated Usage",
        ("process", "shared"): rf"\GPU Process Memory(pid_{pid}_*)\Shared Usage",
    }
    hs = {}
    for k, p in paths.items():
        h = ctypes.c_void_p()
        if pdh.PdhAddEnglishCounterW(q, p, 0, ctypes.byref(h)) == 0:
            hs[k] = h
    return (pdh, q, hs)


def vram_pressure(pid: int | None = None) -> Json | None:
    if sys.platform != "win32":
        return None
    pid = os.getpid() if pid is None else int(pid)
    with _PDH_LOCK:
        return _vram_pressure(pid)


def vram_pressure_close() -> None:
    """every kept query closed (`vram_pressure` opens one again on its next read): the last engine on a card
    closing lets go of what its counter providers hold"""
    with _PDH_LOCK:
        for pdh, q, _hs in _PDH.values():
            pdh.PdhCloseQuery(q)
        _PDH.clear()


def _vram_pressure(pid: int) -> Json | None:
    st = _PDH.get(pid)
    if st is None:
        st = _pdh_open(pid)
        if st is None:
            return None
        _PDH[pid] = st
    pdh, q, hs = st
    out: Json = {"adapters": {}, "process": {"dedicated": 0, "shared": 0}}
    if pdh.PdhCollectQueryData(q) != 0:
        return None
    seen_process = False
    for (grp, what), h in hs.items():
        n, cnt = ctypes.c_ulong(0), ctypes.c_ulong(0)
        pdh.PdhGetFormattedCounterArrayW(h, _PDH_FMT, ctypes.byref(n), ctypes.byref(cnt), None)
        if n.value == 0:
            continue
        buf = ctypes.create_string_buffer(n.value)
        if pdh.PdhGetFormattedCounterArrayW(h, _PDH_FMT, ctypes.byref(n), ctypes.byref(cnt), buf) != 0:
            continue
        items = ctypes.cast(buf, ctypes.POINTER(_PdhItem))
        for j in range(cnt.value):
            name, val = items[j].szName, int(items[j].FmtValue.v.ll)
            if grp == "adapters":
                out["adapters"].setdefault(name, {"dedicated": 0, "shared": 0})[what] = val
            else:
                out["process"][what] += val
                seen_process = True
    if not seen_process:
        # the process had no GPU instance when the wildcard was expanded (queried before its first
        # allocation): the query is reopened on the next read so the instance is picked up
        pdh.PdhCloseQuery(q)
        _PDH.pop(pid, None)
    return out


def vram_pressure_line(pid: int | None = None) -> str:
    p = vram_pressure(pid)
    if not p:
        return "[vram] sensor unavailable"
    ad = max(p["adapters"].values(), key=lambda a: a["dedicated"], default={"dedicated": 0, "shared": 0})
    pr = p["process"]
    return (
        f"[vram] adapter dedicated {ad['dedicated'] / 2**30:.2f} GB, shared (paged to RAM, all tenants) "
        f"{ad['shared'] / 2**30:.2f} GB | this process dedicated {pr['dedicated'] / 2**30:.2f} GB, shared "
        f"{pr['shared'] / 2**30:.2f} GB{'  <- BLED' if pr['shared'] > 0 else ''}"
    )


# -- this process's WDDM video-memory budget, through DXGI (IDXGIAdapter3::QueryVideoMemoryInfo) ------------------
# Windows keeps a process's card memory resident up to its budget and pages past it - its own allocations or, as the
# card fills, another program's. The budget moves with the card's tenants: 10.96 GB of a 12 GB card for a process
# alone on it, 5.8 GB against a game in the foreground. The usage it is compared to is every allocation of the
# process on the adapter, the CUDA allocator's reserved pool included. The adapter is opened once and kept while it
# answers (`_wddm_for`): (the IDXGIAdapter3 or None, when it was looked for)
_WDDM: dict[tuple[str, int, int | None], tuple[Any, float]] = {}
# a card DXGI could not name is looked for again this often, not every pass: a driver reset (a TDR) lists it twice
# for a while, then once again
WDDM_RETRY_S = 10.0


class _Guid(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]


def _guid(s: str) -> _Guid:
    h = s.replace("-", "")
    g = _Guid(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16))
    for i in range(8):
        g.d[i] = int(h[16 + 2 * i : 18 + 2 * i], 16)
    return g


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("desc", ctypes.c_wchar * 128),
        ("vendor", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("subsys", ctypes.c_uint),
        ("rev", ctypes.c_uint),
        ("dedicated_video", ctypes.c_size_t),
        ("dedicated_system", ctypes.c_size_t),
        ("shared_system", ctypes.c_size_t),
        ("luid_low", ctypes.c_uint32),
        ("luid_high", ctypes.c_int32),
        ("flags", ctypes.c_uint),
    ]


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("budget", ctypes.c_uint64),
        ("usage", ctypes.c_uint64),
        ("available_for_reservation", ctypes.c_uint64),
        ("reservation", ctypes.c_uint64),
    ]


def _com(obj: ctypes.c_void_p, slot: int, *argtypes: Any) -> Any:
    """the COM method in vtable `slot` of `obj`; a failing HRESULT raises OSError"""
    if sys.platform != "win32":
        raise OSError("[sysinfo] COM is a Windows interface")
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtbl[slot])


def _com_ref(obj: Any, slot: int) -> None:
    """IUnknown's AddRef (slot 1) or Release (slot 2) on `obj`: each returns the new count, not an HRESULT"""
    if sys.platform != "win32":
        return
    if obj:
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[slot])(obj)


def _com_release(obj: Any) -> None:
    """one reference to the COM object `obj` let go: every factory and adapter DXGI hands out is counted, and one
    never released keeps the kernel's adapter handle open for the life of the process"""
    _com_ref(obj, 2)


_CUDA_LUID: dict[int, int] = {}  # each card's LUID once read; one that could not be is asked again


def cuda_luid(ordinal: int) -> int | None:
    """The LUID Windows knows the card by that CUDA numbers `ordinal` (the driver's cuDeviceGetLuid): what names the
    one DXGI adapter that is this card where DXGI lists it more than once. None off Windows or where it cannot be
    read"""
    if sys.platform != "win32":
        return None
    if ordinal not in _CUDA_LUID:
        try:
            cu = ctypes.WinDLL("nvcuda.dll")
            d = ctypes.c_int()
            buf = (ctypes.c_char * 8)()
            mask = ctypes.c_uint()
            if (
                cu.cuInit(0) == 0
                and cu.cuDeviceGet(ctypes.byref(d), int(ordinal)) == 0
                and cu.cuDeviceGetLuid(buf, ctypes.byref(mask), d) == 0
            ):
                _CUDA_LUID[ordinal] = int.from_bytes(bytes(buf), "little")  # LowPart, then HighPart
        except (OSError, AttributeError):
            pass
    return _CUDA_LUID.get(ordinal)


def _wddm_matches(name: str, total: int) -> list[tuple[Any, int]]:
    """every NVIDIA adapter DXGI lists as `name` with about `total` bytes of dedicated memory, as (IDXGIAdapter1, its
    LUID as one unsigned 64-bit figure - CUDA's form, `cuda_luid`); none off Windows or where DXGI cannot be asked.
    The caller owns each adapter's reference (`_com_release`); the factory and every adapter not returned are
    released here"""
    if sys.platform != "win32":
        return []
    dxgi = ctypes.windll.dxgi
    fac = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(fac)):
        return []
    hits: list[tuple[Any, int]] = []
    try:
        for i in range(16):
            ad = ctypes.c_void_p()
            try:
                _com(fac, 12, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(fac, i, ctypes.byref(ad))  # EnumAdapters1
            except OSError:  # DXGI_ERROR_NOT_FOUND: past the last adapter
                break
            kept = False
            try:
                d = _AdapterDesc1()
                _com(ad, 10, ctypes.POINTER(_AdapterDesc1))(ad, ctypes.byref(d))  # GetDesc1
                near = abs(int(d.dedicated_video) - total) < max(512 << 20, total // 16)
                if d.vendor == 0x10DE and d.desc.strip() == name.strip() and near:
                    # HighPart is a signed LONG: its bits taken as they are, so the figure is CUDA's unsigned one
                    hits.append((ad, ((int(d.luid_high) & 0xFFFFFFFF) << 32) | (int(d.luid_low) & 0xFFFFFFFF)))
                    kept = True
            finally:
                if not kept:
                    _com_release(ad)
    except BaseException:
        for ad, _luid in hits:
            _com_release(ad)
        raise
    finally:
        _com_release(fac)
    return hits


def _wddm_adapter(name: str, total: int, ordinal: int | None = None) -> Any:
    """the NVIDIA adapter DXGI lists as `name` with about `total` bytes of dedicated memory (DXGI counts less of the
    card than CUDA does: 11994 MiB of a 4070 Ti's 12282), as an IDXGIAdapter3: the one whose LUID is CUDA's for card
    `ordinal` where DXGI lists several - after a driver reset (a TDR) it listed this 4070 Ti twice, and matched by name
    alone the budget went unread, so adapt gave nothing back to a game - else the one match by name; None where none
    or several match"""
    hits = _wddm_matches(name, total)
    try:
        luid = cuda_luid(ordinal) if ordinal is not None and len(hits) > 1 else None
        found = [h for h in hits if h[1] == luid] if luid is not None else hits
        if len(found) != 1:
            return None
        ad = found[0][0]
        a3 = ctypes.c_void_p()
        _com(ad, 0, ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))(  # QueryInterface: IDXGIAdapter3
            ad, ctypes.byref(_guid("645967a4-1392-4310-a798-8053ce3e93fd")), ctypes.byref(a3)
        )
        return a3  # its own reference: the IDXGIAdapter1 it was asked of is released with the rest
    finally:
        for ad1, _luid in hits:
            _com_release(ad1)


def _wddm_for(name: str, total: int, ordinal: int | None) -> Any:
    """the card's IDXGIAdapter3 (`_wddm_adapter`): found once and kept while it answers (`wddm_info` drops one that
    fails a read - a driver reset leaves the old adapter answering nothing); a card not found, or a lookup DXGI
    failed, is looked for again after WDDM_RETRY_S, not on every read"""
    key = (name, int(total), ordinal)
    hit = _WDDM.get(key)
    now = time.monotonic()
    if hit is None or (hit[0] is None and now - hit[1] >= WDDM_RETRY_S):
        try:
            a3 = _wddm_adapter(name, int(total), ordinal)
        except OSError:
            a3 = None
        hit = _WDDM[key] = (a3, now)
    return hit[0]


def wddm_info(name: str, total: int, ordinal: int | None = None) -> tuple[int, int] | None:
    """(budget, usage): this process's WDDM budget on the card named `name` with `total` bytes (CUDA's card `ordinal`,
    which names it where DXGI lists it more than once) - what Windows keeps resident for it before paging, which
    falls when another program takes the card - and what it uses there. None off Windows, where DXGI cannot name the
    card uniquely, or where it cannot be read. Microseconds a read."""
    if sys.platform != "win32":
        return None
    try:
        a3 = _wddm_for(name, total, ordinal)
    except OSError:
        return None
    if a3 is None:
        return None
    m = _VideoMemoryInfo()
    try:
        # QueryVideoMemoryInfo: node 0, the local (dedicated) segment group
        _com(a3, 14, ctypes.c_uint, ctypes.c_int, ctypes.POINTER(_VideoMemoryInfo))(a3, 0, 0, ctypes.byref(m))
    except OSError:
        # the adapter gone (a driver reset): released, and found afresh next read (a budget event registered on it
        # holds a reference of its own until it is unregistered)
        gone = _WDDM.pop((name, int(total), ordinal), None)
        if gone is not None:
            _com_release(gone[0])
        return None
    return int(m.budget), int(m.usage)


@dataclass(frozen=True)
class BudgetEvent:
    """a budget event's registration (`wddm_budget_event`): the Windows event, the cookie, and the adapter it was
    registered on - unregistered on that one, whatever has been found for the card since"""

    event: int
    cookie: int
    adapter: Any


def wddm_budget_event(name: str, total: int, ordinal: int | None = None) -> BudgetEvent | None:
    """the event the OS signals whenever this process's WDDM budget on the card named `name` changes - another
    program asking for the card (a game started or brought to the front) shrinks it at once, before any of this
    process's memory is paged out (IDXGIAdapter3::RegisterVideoMemoryBudgetChangeNotificationEvent). An auto-reset
    event, until `wddm_budget_unregister`. None off Windows or where the card cannot be named."""
    if sys.platform != "win32":
        return None
    try:
        a3 = _wddm_for(name, total, ordinal)
        if a3 is None:
            return None
        k32 = ctypes.windll.kernel32
        k32.CreateEventW.restype = ctypes.c_void_p
        ev = k32.CreateEventW(None, False, False, None)
        if not ev:
            return None
        cookie = ctypes.c_uint32(0)
        try:
            _com(a3, 16, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))(a3, ev, ctypes.byref(cookie))
        except OSError:
            k32.CloseHandle(ctypes.c_void_p(ev))
            return None
        # a reference of the registration's own: the cache's is released when a read finds the adapter gone, and the
        # unregister still needs the adapter it registered on
        _com_ref(a3, 1)
        return BudgetEvent(int(ev), int(cookie.value), a3)
    except OSError:
        return None


def wddm_budget_stale(reg: BudgetEvent, name: str, total: int, ordinal: int | None = None) -> bool:
    """whether `reg` was registered on an adapter the card is no longer found as (a driver reset: `wddm_info` let
    the old one go and found it afresh), so the OS signals another registration's event now, not this one's"""
    if sys.platform != "win32":
        return False
    hit = _WDDM.get((name, int(total), ordinal))
    a3 = hit[0] if hit is not None else None
    return a3 is not None and getattr(a3, "value", a3) != getattr(reg.adapter, "value", reg.adapter)


def wddm_budget_unregister(reg: BudgetEvent) -> None:
    """the budget event `wddm_budget_event` registered, unregistered, its handle closed and its adapter released"""
    if sys.platform != "win32":
        return
    # UnregisterVideoMemoryBudgetChangeNotification returns void, not an HRESULT: called through its own prototype
    vtbl = ctypes.cast(reg.adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32)(vtbl[17])(reg.adapter, ctypes.c_uint32(reg.cookie))
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(reg.event))
    _com_release(reg.adapter)


def release_pages(ptr: int, nbytes: int) -> None:
    """The pages of [ptr, ptr + nbytes) of a read-only file mapping out of this process's resident memory, counted
    available by the OS at once: on Windows to the standby list (VirtualUnlock on pages never locked takes them out
    of the working set), elsewhere unmapped from the process (madvise MADV_DONTNEED; the file's pages stay in the page
    cache). A mapping no longer read otherwise stays resident until the OS gets round to trimming it, and a program
    asking for memory meanwhile sees none of it. Only for a read-only file mapping: its bytes are the file's, read
    again on the next touch"""
    if nbytes <= 0:
        return
    import mmap

    page = int(mmap.PAGESIZE)
    lo = int(ptr) // page * page
    hi = -(-(int(ptr) + int(nbytes)) // page) * page
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        k32.VirtualUnlock.restype = ctypes.c_int
        k32.VirtualUnlock(lo, hi - lo)  # FALSE with ERROR_NOT_LOCKED: the pages left the working set, as asked
        return
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    libc.madvise(lo, hi - lo, int(getattr(mmap, "MADV_DONTNEED", 4)))


def wait_event(handle: int, timeout_s: float) -> bool:
    """whether the Windows event `handle` was signalled within `timeout_s` seconds"""
    if sys.platform != "win32":
        raise RuntimeError("[sysinfo] WaitForSingleObject is a Windows call")
    k32 = ctypes.windll.kernel32
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.WaitForSingleObject.restype = ctypes.c_uint32
    return int(k32.WaitForSingleObject(handle, int(timeout_s * 1000))) == 0  # WAIT_OBJECT_0


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


# NVML, loaded once: the library (None where it is absent or would not initialize) and each card's handle by its PCI
# bus id - dropped when a read fails, as after a driver reset
_NVML: dict[str, Any] = {"lib": None, "tried": False, "handles": {}}


def _nvml() -> Any:
    if not _NVML["tried"]:
        _NVML["tried"] = True
        # the driver's own copy in System32 on current drivers; the NVSMI folder on older ones
        nvsmi = os.path.join(os.environ.get("PROGRAMFILES", r"C:\Program Files"), "NVIDIA Corporation", "NVSMI")
        names = (
            ["nvml.dll", os.path.join(nvsmi, "nvml.dll")]
            if sys.platform == "win32"
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        )
        for n in names:
            try:
                lib = ctypes.CDLL(n)
            except OSError:
                continue
            if lib.nvmlInit_v2() == 0:
                _NVML["lib"] = lib
                break
    return _NVML["lib"]


def _nvml_handle(bus_id: str) -> Any:
    """NVML's handle on the card at PCI `bus_id`, looked up once; None without NVML or where it names no card"""
    lib = _nvml()
    find = getattr(lib, "nvmlDeviceGetHandleByPciBusId_v2", None) if lib is not None else None
    if find is None:
        return None
    h = _NVML["handles"].get(bus_id)
    if h is None:
        h = ctypes.c_void_p()
        if find(bus_id.encode(), ctypes.byref(h)) != 0:
            return None
        _NVML["handles"][bus_id] = h
    return h


def nvml_free_bytes(bus_id: str) -> int | None:
    """The card's free memory across every process as NVML counts it - the reading `nvidia-smi` prints, without
    starting it (a process a read, 36 ms each on this machine) - for the card at PCI `bus_id` ("00000000:01:00.0"):
    NVML numbers the cards in its own order, which CUDA's (and CUDA_VISIBLE_DEVICES) does not follow. None where NVML
    is absent (no NVIDIA driver) or the card cannot be read. Microseconds a read."""
    h = _nvml_handle(bus_id)
    if h is None:
        return None
    lib = _NVML["lib"]
    m = _NvmlMemory()
    if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) != 0:
        _NVML["handles"].pop(bus_id, None)  # a stale handle (a driver reset): looked up afresh next read
        return None
    return int(m.free)


def host_cache_groups() -> list[Json]:
    """The host CPU's data and unified caches, L1 to L3, as the OS lists them: each kind of instance once with its
    `level`, `size` in bytes, the `threads` (logical processors) sharing one, and the `count` of them - a P-core's
    own L2 shared by its two hyperthreads, an efficiency cluster's by four cores, a chiplet's L3 by its cores.
    Windows through GetLogicalProcessorInformationEx, Linux from sysfs, macOS from sysctl; every OS read to this
    one form, so a CPU reads the same on each. Level by level, the largest first; [] where none can be read. Read
    once; each call a fresh list."""
    return _cache_groups(list(_host_caches()))


@lru_cache(maxsize=1)
def _host_caches() -> tuple[tuple[int, int, int], ...]:
    """the OS's caches as (level, size, threads) instances, read once: they cannot change while btb runs"""
    try:
        if sys.platform == "win32":
            raw = _win_records(2)  # RelationCache
            return tuple(_win_caches(raw) if raw else [])
        if sys.platform == "linux":
            return tuple(_linux_caches())
        if sys.platform == "darwin":
            return tuple(_darwin_caches())
    except Exception:
        pass
    return ()


def host_cache_sizes() -> dict[str, int]:
    """The host CPU's cache sizes in bytes, from `host_cache_groups`, keyed by `HostCache`.

    Each level has the smallest single instance (a thread running alone, on any core: a 7950X3D's L3 is 32 MB, not
    96) and the smallest per-thread share with every thread busy (a 13700K's L2 is 1 MB a thread: 2 MB per P-core
    over its 2 hyperthreads, 4 MB per E-core cluster over 4 cores). A consumer picks by how many threads it runs.
    The L2 and L3 totals add up every instance: what a working set must pass to miss every cache. 0 where a level is
    missing or can't be read."""
    return cache_sizes(host_cache_groups())


def cache_sizes(groups: list[Json]) -> dict[str, int]:
    """`host_cache_sizes`' figures from `host_cache_groups`' form."""
    out: dict[str, int] = dict.fromkeys(HostCache, 0)
    levels = (
        (1, HostCache.L1D, HostCache.L1D_PER_THREAD, None),
        (2, HostCache.L2, HostCache.L2_PER_THREAD, HostCache.L2_TOTAL),
        (3, HostCache.L3, HostCache.L3_PER_THREAD, HostCache.L3_TOTAL),
    )
    for level, one, per_thread, total in levels:
        gs = [g for g in groups if g["level"] == level]
        if gs:
            out[one] = min(g["size"] for g in gs)
            out[per_thread] = min(g["size"] // g["threads"] for g in gs)
            if total is not None:
                out[total] = sum(g["size"] * g["count"] for g in gs)
    return out


def _cache_groups(caches: list[tuple[int, int, int]]) -> list[Json]:
    """each (level, size, threads) instance counted into `host_cache_groups`' form"""
    n: dict[tuple[int, int, int], int] = {}
    for c in caches:
        n[c] = n.get(c, 0) + 1
    return [
        {"level": lvl, "size": size, "threads": threads, "count": k}
        for (lvl, size, threads), k in sorted(n.items(), key=lambda e: (e[0][0], -e[0][1], -e[0][2]))
    ]


def _win_records(rel: int) -> bytes | None:
    """GetLogicalProcessorInformationEx's records of relationship `rel` (0 the cores, 2 the caches), every
    processor group's; None where it gives none (or this is not Windows)"""
    if sys.platform != "win32":
        return None
    k32 = ctypes.windll.kernel32
    n = ctypes.c_uint32(0)
    k32.GetLogicalProcessorInformationEx(rel, None, ctypes.byref(n))  # the size it needs
    buf = ctypes.create_string_buffer(max(1, n.value))
    if not n.value or not k32.GetLogicalProcessorInformationEx(rel, buf, ctypes.byref(n)):
        return None
    return buf.raw[: n.value]


def _win_each(raw: bytes, at: int, wide: bool | None = None) -> Iterator[tuple[int, int, set[int]]]:
    """each record of `_win_records` as (its relationship, its offset, the logical processors it names): a record
    is Relationship and Size, then the relationship's fields; GroupCount (0 before Windows 11: one group) sits
    just before the GROUP_AFFINITYs, `at` bytes in - each a mask the pointer's width (`wide`: 64-bit, this
    process's by default), the group and reserved bytes. A record cut short ends the walk."""
    wide = ctypes.sizeof(ctypes.c_void_p) == 8 if wide is None else wide
    bits, fmt, stride = (64, "<QH", 16) if wide else (32, "<IH", 12)
    off = 0
    while off + 8 <= len(raw):
        rel, size = struct.unpack_from("<II", raw, off)
        if size < at + stride or off + size > len(raw):
            break
        (groups,) = struct.unpack_from("<H", raw, off + at - 2)
        lps: set[int] = set()
        for g in range(min(max(1, groups), (size - at) // stride)):
            mask, grp = struct.unpack_from(fmt, raw, off + at + stride * g)
            lps.update(grp * bits + b for b in range(bits) if mask >> b & 1)
        yield rel, off, lps
        off += size


def _win_caches(raw: bytes, wide: bool | None = None) -> list[tuple[int, int, int]]:
    """each data or unified cache in RelationCache records, once by the processors sharing it: after Relationship
    and Size, CACHE_RELATIONSHIP - Level, Associativity, LineSize, CacheSize, Type (0 unified, 1 instruction,
    2 data, 3 trace), 18 reserved bytes, GroupCount - and its GROUP_AFFINITYs from offset 40"""
    out = []
    seen: set[tuple[int, int, frozenset[int]]] = set()
    for rel, off, lps in _win_each(raw, 40, wide):
        level = raw[off + 8]
        nbytes, kind = struct.unpack_from("<Ii", raw, off + 12)
        key = (level, kind, frozenset(lps))
        if rel == 2 and kind in (0, 2) and 1 <= level <= 3 and lps and key not in seen:
            seen.add(key)
            out.append((level, int(nbytes), len(lps)))
    return out


def _linux_caches(root: str = "/sys") -> list[tuple[int, int, int]]:
    """each data or unified cache in sysfs under `root` once, by the CPUs sharing it (shared_cpu_list); an entry
    whose files cannot be read is passed over"""
    out = []
    seen: set[tuple[str, str]] = set()
    for p in sorted(glob.glob(f"{glob.escape(root)}/devices/system/cpu/cpu*/cache/index*/level")):
        d = p[: -len("level")]
        lvl, shared, s = _read(p), _read(d + "shared_cpu_list"), _read(d + "size")
        if lvl not in ("1", "2", "3") or _read(d + "type") == "Instruction" or not shared or not s:
            continue
        if (lvl, shared) in seen:
            continue
        seen.add((lvl, shared))
        unit = {"K": 1024, "M": 2**20}.get(s[-1], 1)
        out.append((int(lvl), int(s[:-1] if unit > 1 else s) * unit, len(_cpulist(shared))))
    return out


def _darwin_caches() -> list[tuple[int, int, int]]:
    """sysctl's caches: each perflevel's (macOS 12 on) - its cores' own L1d, an L2 per `cpusperl2` of its CPUs, an
    L3 per `cpusperl3` - where its cores run a thread each (Apple silicon: CPUs are cores, whichever `cpusperl`
    counts); else, and for a level the perflevels leave out, hw.l1d/l2/l3cachesize, each shared by the logical
    CPUs hw.cacheconfig gives for its level (Intel Macs, hyperthreaded; Apple silicon publishes no L3)"""
    out: list[tuple[int, int, int]] = []
    levels = int(_sysctl_int("hw.nperflevels") or 0)
    for i in range(levels):
        p = f"hw.perflevel{i}."
        cores = int(_sysctl_int(p + "physicalcpu") or 0)
        if not cores or int(_sysctl_int(p + "logicalcpu") or cores) != cores:
            # hyperthreaded: whether `cpusperl` counts its cores or its threads is not said; hw.cacheconfig's are threads
            out = []
            break
        l1 = _sysctl_int(p + "l1dcachesize")
        if l1:
            out += [(1, l1, 1)] * cores
        for lvl in (2, 3):
            size, per = _sysctl_int(f"{p}l{lvl}cachesize"), _sysctl_int(f"{p}cpusperl{lvl}")
            if size and per:
                out += [(lvl, size, per)] * max(1, cores // per)
    have = {c[0] for c in out}
    raw = _sysctl_raw("hw.cacheconfig")  # uint64s: the logical CPUs sharing memory, then each level's cache
    share = struct.unpack_from(f"<{len(raw) // 8}Q", raw) if raw else ()
    logical = int(_sysctl_int("hw.logicalcpu") or 0)
    for lvl, name in ((1, "hw.l1dcachesize"), (2, "hw.l2cachesize"), (3, "hw.l3cachesize")):
        size = _sysctl_int(name)
        per = int(share[lvl]) if len(share) > lvl and share[lvl] else 1
        if size and logical and lvl not in have:
            out += [(lvl, size, per)] * max(1, logical // per)
    return out


# -- the hardware as the OS and the drivers describe it: what the machine report (`btb bench`) lists -------------


def _read(path: str) -> str | None:
    """a small sysfs file's text, stripped; None where it cannot be read"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except (OSError, ValueError):
        return None


def _cpulist(s: str | None) -> set[int]:
    """the CPUs a Linux cpulist names ("0-7,16")"""
    out: set[int] = set()
    for part in (s or "").split(","):
        lo, _, hi = part.strip().partition("-")
        if lo.isdigit():
            out.update(range(int(lo), int(hi if hi.isdigit() else lo) + 1))
    return out


def _class_names(n: int) -> list[str]:
    return ["cores"] if n == 1 else ["performance", "efficiency"] if n == 2 else [f"class {i}" for i in range(n)]


def _win_topology() -> Json:
    """physical cores by efficiency class (GetLogicalProcessorInformationEx's RelationProcessorCore records) and
    the package's rated clock (CallNtPowerInformation's ProcessorInformation: the one figure for every core, a
    hybrid's efficiency cores included, so it is not given per class)"""
    raw = _win_records(0)  # RelationProcessorCore
    if raw is None:
        return {}
    # after Relationship and Size, PROCESSOR_RELATIONSHIP - Flags, EfficiencyClass, 20 reserved bytes, GroupCount -
    # and its GROUP_AFFINITYs from offset 32
    cores = [(raw[off + 9], lps) for rel, off, lps in _win_each(raw, 32) if rel == 0]
    logical = sum(len(lps) for _, lps in cores)
    rated = 0
    try:
        # PROCESSOR_POWER_INFORMATION: Number, MaxMhz, CurrentMhz, MhzLimit, MaxIdleState, CurrentIdleState
        out = ctypes.create_string_buffer(24 * logical)
        if ctypes.windll.powrprof.CallNtPowerInformation(11, None, 0, out, 24 * logical) == 0:
            rated = max(struct.unpack_from("<I", out.raw, 24 * i + 4)[0] for i in range(logical))
    except (OSError, AttributeError, ValueError):
        pass
    by: dict[int, Json] = {}
    for eff, lps in cores:
        c = by.setdefault(eff, {"cores": 0, "threads": 0, "base_mhz": 0, "max_mhz": 0})
        c["cores"] += 1
        c["threads"] += len(lps)
    order = sorted(by, reverse=True)  # the higher the class, the faster the core
    names = _class_names(len(order))
    return {
        "logical": logical,
        "physical": len(cores),
        "base_mhz": int(rated),
        "max_mhz": 0,
        "classes": [{"name": names[i], **by[e]} for i, e in enumerate(order)],
    }


def _linux_topology(root: str = "/sys") -> Json:
    """sysfs under `root`: physical cores from each CPU's thread siblings (its package and core id where the kernel
    lists none: Arm's core ids restart in each cluster), classes from cpu_capacity (Arm) or the hybrid PMUs' cpu
    lists (Intel's cpu_core), clocks from cpufreq (kHz)"""
    online = _cpulist(_read(f"{root}/devices/system/cpu/online"))
    if not online:
        online = {int(p.rsplit("cpu", 1)[1]) for p in glob.glob(f"{glob.escape(root)}/devices/system/cpu/cpu[0-9]*")}
    big = _cpulist(_read(f"{root}/devices/cpu_core/cpus"))
    by: dict[int, Json] = {}
    for c in sorted(online):
        d = f"{root}/devices/system/cpu/cpu{c}/"
        cap = _read(d + "cpu_capacity")
        key = int(cap) if cap and cap.isdigit() else (1 if c in big else 0)
        k = by.setdefault(key, {"core_ids": set(), "threads": 0, "base_mhz": 0, "max_mhz": 0})
        sib = _read(d + "topology/core_cpus_list") or _read(d + "topology/thread_siblings_list")
        k["core_ids"].add(sib or f"{_read(d + 'topology/physical_package_id')}/{_read(d + 'topology/core_id') or c}")
        k["threads"] += 1
        for field, name in (("base_mhz", "base_frequency"), ("max_mhz", "cpuinfo_max_freq")):
            v = _read(d + "cpufreq/" + name)
            if v and v.isdigit():
                k[field] = max(k[field], int(v) // 1000)
    order = sorted(by, reverse=True)
    names = _class_names(len(order))
    classes = [
        {"name": names[i], "cores": len(by[e]["core_ids"]), "threads": by[e]["threads"],
         "base_mhz": by[e]["base_mhz"], "max_mhz": by[e]["max_mhz"]}
        for i, e in enumerate(order)
    ]  # fmt: skip
    return {
        "logical": len(online),
        "physical": sum(c["cores"] for c in classes),
        "base_mhz": max((c["base_mhz"] for c in classes), default=0),
        "max_mhz": max((c["max_mhz"] for c in classes), default=0),
        "classes": classes,
    }


def _darwin_topology() -> Json:
    """sysctl's perflevels (Apple silicon's performance and efficiency cores; one level on an Intel Mac)"""
    levels = int(_sysctl_int("hw.nperflevels") or 0)
    top = int(_sysctl_int("hw.cpufrequency_max") or 0) // 1_000_000  # Intel Macs; Apple silicon publishes none
    classes = []
    for i in range(levels):
        raw = _sysctl_raw(f"hw.perflevel{i}.name")
        name = raw.split(b"\0", 1)[0].decode(errors="replace").lower() if raw else _class_names(levels)[i]
        classes.append(
            {
                "name": name,
                "cores": int(_sysctl_int(f"hw.perflevel{i}.physicalcpu") or 0),
                "threads": int(_sysctl_int(f"hw.perflevel{i}.logicalcpu") or 0),
                "base_mhz": 0,
                "max_mhz": top,
            }
        )
    physical, logical = int(_sysctl_int("hw.physicalcpu") or 0), int(_sysctl_int("hw.logicalcpu") or 0)
    if not classes:
        classes = [{"name": "cores", "cores": physical, "threads": logical, "base_mhz": 0, "max_mhz": top}]
    return {"logical": logical, "physical": physical, "base_mhz": 0, "max_mhz": top, "classes": classes}


@lru_cache(maxsize=1)
def host_cpu_topology() -> Json:
    """The CPU's cores as the OS groups them: `logical` processors, `physical` cores, the clocks the OS reports in
    MHz - `base_mhz` the rated clock (Windows, Linux's intel_pstate), `max_mhz` the highest a core may boost to
    (Linux, Intel Macs), 0 where it gives none - and `classes`: each kind of core, the fastest first (a hybrid's
    performance and efficiency cores, a single class elsewhere) with its `cores`, `threads` and clocks where the OS
    gives them per core (Linux; 0 elsewhere). Windows through GetLogicalProcessorInformationEx and
    CallNtPowerInformation, Linux from sysfs, macOS from sysctl. Read once; {} where nothing could be read."""
    try:
        if sys.platform == "win32":
            return _win_topology()
        if sys.platform == "linux":
            return _linux_topology()
        if sys.platform == "darwin":
            return _darwin_topology()
    except Exception:
        pass
    return {}


# SMBIOS type 17's memory type and form factor bytes (the DMTF's SMBIOS reference, 7.18.1-2), the ones a machine
# btb runs on carries
_SMBIOS_KIND = {
    0x12: "DDR", 0x13: "DDR2", 0x18: "DDR3", 0x1A: "DDR4", 0x1B: "LPDDR", 0x1C: "LPDDR2", 0x1D: "LPDDR3",
    0x1E: "LPDDR4", 0x20: "HBM", 0x21: "HBM2", 0x22: "DDR5", 0x23: "LPDDR5", 0x24: "HBM3",
}  # fmt: skip
_SMBIOS_FORM = {0x09: "DIMM", 0x0B: "row of chips", 0x0D: "SODIMM", 0x0F: "FB-DIMM", 0x10: "die", 0x11: "CAMM"}


def _smbios_table() -> bytes | None:
    """the firmware's SMBIOS structure table: Windows through GetSystemFirmwareTable('RSMB') (its 8-byte header
    dropped), Linux from /sys/firmware/dmi/tables/DMI (readable by root alone on most distributions); None elsewhere
    or where it cannot be read"""
    if sys.platform == "win32":
        fn = ctypes.windll.kernel32.GetSystemFirmwareTable
        fn.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        sig = int.from_bytes(b"RSMB", "big")
        size = int(fn(sig, 0, None, 0))
        if size <= 8:
            return None
        buf = ctypes.create_string_buffer(size)
        if int(fn(sig, 0, buf, size)) != size:
            return None
        (n,) = struct.unpack_from("<I", buf.raw, 4)
        return buf.raw[8 : 8 + n]
    if sys.platform == "linux":
        try:
            with open("/sys/firmware/dmi/tables/DMI", "rb") as f:
                return f.read()
        except OSError:
            return None
    return None


def _smbios_structs(t: bytes, kind: int) -> list[tuple[bytes, list[str]]]:
    """every structure of `kind` in the table: its formatted area and its strings (string n is [n - 1])"""
    out = []
    off = 0
    while off + 4 <= len(t):
        typ, ln = t[off], t[off + 1]
        if ln < 4:
            break
        end = t.find(b"\0\0", off + ln)
        if end < 0:
            break
        if typ == kind:
            strings = [s.decode("ascii", errors="replace").strip() for s in t[off + ln : end].split(b"\0")]
            out.append((t[off : off + ln], strings))
        if typ == 127:  # the end of the table
            break
        off = end + 2
    return out


def _module(s: bytes, strings: list[str]) -> Json | None:
    """one SMBIOS memory device (type 17) as `memory_modules` lists it, None for an empty slot"""

    def u16(o: int) -> int:
        return int.from_bytes(s[o : o + 2], "little") if len(s) >= o + 2 else 0

    def u32(o: int) -> int:
        return int.from_bytes(s[o : o + 4], "little") if len(s) >= o + 4 else 0

    def text(o: int) -> str:
        i = s[o] if len(s) > o else 0
        return strings[i - 1] if 0 < i <= len(strings) else ""

    size = u16(0x0C)
    if size in (0, 0xFFFF):
        return None
    mb = (u32(0x1C) & 0x7FFFFFFF) if size == 0x7FFF else (size & 0x7FFF) / 1024 if size & 0x8000 else size
    rated = u32(0x54) if u16(0x15) == 0xFFFF else u16(0x15)
    runs = u32(0x58) if u16(0x20) == 0xFFFF else u16(0x20)
    width = u16(0x0A)
    return {
        "size_gb": round(mb / 1024, 2),
        "kind": _SMBIOS_KIND.get(s[0x12] if len(s) > 0x12 else 0, "unknown"),
        "form": _SMBIOS_FORM.get(s[0x0E] if len(s) > 0x0E else 0, "other"),
        "rated_mts": int(rated),
        "mts": int(runs),
        "width": 0 if width == 0xFFFF else int(width),
        "rank": (s[0x1B] & 0x0F) if len(s) > 0x1B else 0,
        "slot": text(0x10),
        "bank": text(0x11),
        "maker": text(0x17),
        "part": text(0x1A),
    }


_UDEV_DMI = "/run/udev/data/+dmi:id"


def _udev_modules() -> list[Json] | None:
    """the SMBIOS memory devices as systemd's udev keeps them: its dmi_memory_id helper (systemd 248 on) reads the
    table as root at boot, and udev's database holds what it found for the dmi device, readable by every user, as
    `E:MEMORY_DEVICE_<n>_<FIELD>=<value>` lines. The populated slots as `memory_modules` lists them; None where
    the database holds none (no systemd, an older one, a container)"""
    try:
        with open(_UDEV_DMI, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return None
    slots: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        m = re.fullmatch(r"E:MEMORY_DEVICE_(\d+)_([A-Z_]+)=(.*)", line.strip())
        if m:
            slots.setdefault(int(m[1]), {})[m[2]] = m[3].strip()
    if not slots:
        return None

    def num(d: dict[str, str], k: str) -> int:
        v = d.get(k, "")
        return int(v) if v.isdigit() else 0

    def text_of(d: dict[str, str], k: str) -> str:
        v = d.get(k, "")
        return "" if v == "Not Specified" else v  # the helper's word for a string the table leaves empty

    # the helper's names for the type and form codes ("Row Of Chips", "Unknown", "<OUT OF SPEC>") mapped to the
    # names `_module` gives the same codes: a module reads the same from udev as from the table
    kinds = {v.lower(): v for v in _SMBIOS_KIND.values()}
    forms = {v.lower(): v for v in _SMBIOS_FORM.values()}

    out = []
    for _, d in sorted(slots.items()):
        size = num(d, "SIZE")  # bytes; an empty slot carries PRESENT=0 and no size
        if d.get("PRESENT") == "0" or not size:
            continue
        out.append(
            {
                "size_gb": round(size / 2**30, 2),
                "kind": kinds.get(d.get("TYPE", "").lower(), "unknown"),
                "form": forms.get(d.get("FORM_FACTOR", "").lower(), "other"),
                "rated_mts": num(d, "SPEED_MTS"),
                "mts": num(d, "CONFIGURED_SPEED_MTS"),
                "width": num(d, "DATA_WIDTH"),
                "rank": num(d, "RANK"),
                "slot": text_of(d, "LOCATOR"),
                "bank": text_of(d, "BANK_LOCATOR"),
                "maker": text_of(d, "MANUFACTURER"),
                "part": text_of(d, "PART_NUMBER"),
            }
        )
    return out


def memory_modules() -> list[Json] | None:
    """The memory modules the firmware lists (SMBIOS type 17), the populated slots: each one's `size_gb`, `kind`
    (DDR4, DDR5, LPDDR5...), `form` (DIMM, SODIMM...), `rated_mts` (its rated transfer rate, MT/s) and `mts` (the
    rate the firmware runs it at; 0 where it does not say), `width` (data bits), `rank`, `slot` and `bank` (the
    board's locators), `maker` and `part`. Serial numbers and asset tags are left out. Linux reads the table where
    it may (root) and udev's copy of it otherwise. None where neither can be read (macOS; Linux without systemd's
    copy)."""
    try:
        t = _smbios_table()
        if t is not None:
            return [m for s, strings in _smbios_structs(t, 17) if (m := _module(s, strings)) is not None]
        return _udev_modules() if sys.platform == "linux" else None
    except Exception:
        return None


def memory_channels(mods: list[Json]) -> int | None:
    """The channels the modules sit on, read off the locators where the board names them: `ChannelA`,
    `P0 CHANNEL B` and `Controller1-ChannelA-DIMM0` (a channel and its controller), `Controller0-DIMM1` (a controller
    of one channel: Intel's desktop DDR5), or `DIMM_A2` (channel A's second slot). None where any module's locators
    name none of these: a peak rate is not guessed from a module count, as two modules on one channel share its
    rate."""
    keys: set[tuple[str, str]] = set()
    for m in mods:
        text = f"{m.get('bank', '')} {m.get('slot', '')}".lower()
        ch = re.search(r"channel\s*[-_]?\s*([a-z0-9]+)", text) or re.search(r"\bdimm[\s_-]?([a-h])\d", text)
        ctl = re.search(r"(?:controller|socket|node|cpu|\bp)\s*[-_]?\s*(\d+)", text)
        if ch is None and ctl is None:
            return None
        keys.add((ctl.group(1) if ctl else "", ch.group(1) if ch else ""))
    return len(keys) or None


def _nvml_uint(fn: str, h: Any, *args: int) -> int | None:
    """an NVML query that writes one unsigned int, None where the driver lacks it or refuses it"""
    f = getattr(_NVML["lib"], fn, None)
    if f is None:
        return None
    v = ctypes.c_uint(0)
    return int(v.value) if f(h, *[ctypes.c_int(a) for a in args], ctypes.byref(v)) == 0 else None


def nvml_card(bus_id: str) -> Json | None:
    """The card at PCI `bus_id` as NVML describes it: its PCIe link now (`pcie_gen`, `pcie_width`) and at most
    (`pcie_max_gen`, `pcie_max_width`: the slot's and the card's lesser), the peak clocks in MHz (`sm_mhz`,
    `mem_mhz`), the memory bus in bits (`bus_bits`), the power limit in watts (`power_w`) and the `driver`'s
    version. A field NVML cannot give is left out; None without NVML. The link trains down when the card idles:
    read under a transfer, it is the speed the bus runs at."""
    h = _nvml_handle(bus_id)
    if h is None:
        return None
    out: Json = {}
    for key, fn, args in (
        ("pcie_gen", "nvmlDeviceGetCurrPcieLinkGeneration", ()),
        ("pcie_width", "nvmlDeviceGetCurrPcieLinkWidth", ()),
        ("pcie_max_gen", "nvmlDeviceGetMaxPcieLinkGeneration", ()),
        ("pcie_max_width", "nvmlDeviceGetMaxPcieLinkWidth", ()),
        ("sm_mhz", "nvmlDeviceGetMaxClockInfo", (1,)),  # NVML_CLOCK_SM
        ("mem_mhz", "nvmlDeviceGetMaxClockInfo", (2,)),  # NVML_CLOCK_MEM
        ("bus_bits", "nvmlDeviceGetMemoryBusWidth", ()),
        ("power_w", "nvmlDeviceGetPowerManagementLimit", ()),
    ):
        v = _nvml_uint(fn, h, *args)
        if v is not None:
            out[key] = v // 1000 if key == "power_w" else v  # the limit is in milliwatts
    ver = ctypes.create_string_buffer(96)
    f = getattr(_NVML["lib"], "nvmlSystemGetDriverVersion", None)
    if f is not None and f(ver, ctypes.c_uint(96)) == 0:
        out["driver"] = ver.value.decode(errors="replace")
    return out
