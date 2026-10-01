# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""OS-level measurement: what the machine has and what is free right now. Stateless queries (Windows through
ctypes, Linux from /proc and sysfs, macOS from sysctlbyname), no torch. The card's free VRAM as torch's
allocator sees it is `engine/device.py`'s, not here."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import pool
from .kinds import Json

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
# step - where a collect on a kept query is a fraction of a millisecond; the query lives for the process
_PDH_FMT = 0x00000400 | 0x00001000
_PDH: dict[int, Any] = {}


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
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtbl[slot])


def _com_ref(obj: Any, slot: int) -> None:
    """IUnknown's AddRef (slot 1) or Release (slot 2) on `obj`: each returns the new count, not an HRESULT"""
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


def nvml_free_bytes(bus_id: str) -> int | None:
    """The card's free memory across every process as NVML counts it - the reading `nvidia-smi` prints, without
    starting it (a process a read, 36 ms each on this machine) - for the card at PCI `bus_id` ("00000000:01:00.0"):
    NVML numbers the cards in its own order, which CUDA's (and CUDA_VISIBLE_DEVICES) does not follow. None where NVML
    is absent (no NVIDIA driver) or the card cannot be read. Microseconds a read."""
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
    m = _NvmlMemory()
    if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m)) != 0:
        _NVML["handles"].pop(bus_id, None)  # a stale handle (a driver reset): looked up afresh next read
        return None
    return int(m.free)


def host_cache_sizes() -> dict[str, int]:
    """The host CPU's L2 (all cores' together) and L3 in bytes, read from the OS: Windows through the
    processor-information table, Linux from sysfs, macOS from sysctl. Zeros where a level cannot be read."""
    out = {"host_l2": 0, "host_l3": 0}
    try:
        if sys.platform == "win32":

            class _Cache(ctypes.Structure):
                _fields_ = [
                    ("Level", ctypes.c_ubyte),
                    ("Associativity", ctypes.c_ubyte),
                    ("LineSize", ctypes.c_ushort),
                    ("Size", ctypes.c_uint32),
                    ("Type", ctypes.c_int),
                ]

            class _Union(ctypes.Union):
                _fields_ = [("Cache", _Cache), ("Reserved", ctypes.c_ulonglong * 2)]

            class _Info(ctypes.Structure):
                _fields_ = [("ProcessorMask", ctypes.c_size_t), ("Relationship", ctypes.c_int), ("u", _Union)]

            k32 = ctypes.windll.kernel32
            n = ctypes.c_uint32(0)
            k32.GetLogicalProcessorInformation(None, ctypes.byref(n))
            cnt = n.value // ctypes.sizeof(_Info)
            buf = (_Info * cnt)()
            if k32.GetLogicalProcessorInformation(buf, ctypes.byref(n)):
                for e in buf:
                    if e.Relationship == 2:  # RelationCache
                        c = e.u.Cache
                        if c.Level == 2:
                            out["host_l2"] += int(c.Size)
                        elif c.Level == 3:
                            # one entry per sharing group: the same L3 appears under each mask it serves
                            out["host_l3"] = max(out["host_l3"], int(c.Size))
        elif sys.platform == "linux":
            import glob

            for lvl in (2, 3):
                seen: set[str] = set()
                tot = 0
                for p in glob.glob("/sys/devices/system/cpu/cpu*/cache/index*/level"):
                    d = p[: -len("level")]
                    with open(p) as fh:
                        if fh.read().strip() != str(lvl):
                            continue
                    with open(d + "shared_cpu_list") as fh:
                        key = fh.read().strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    with open(d + "size") as fh:
                        s = fh.read().strip()
                    tot += int(s[:-1]) * (1024 if s.endswith("K") else 2**20 if s.endswith("M") else 1)
                out[f"host_l{lvl}"] = tot
        elif sys.platform == "darwin":
            for lvl in (2, 3):
                v = _sysctl_int(f"hw.l{lvl}cachesize")
                if v:
                    out[f"host_l{lvl}"] = v
    except Exception:
        pass
    return out
