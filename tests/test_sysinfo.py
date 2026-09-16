# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The machine sensors (btb/sysinfo.py): the page-fault and process counters hold together and only climb, and
the OS's own word on its memory floor and pressure. Reads this process and the OS - no model, torch-free."""

import os
import sys

import pytest

from tests.helpers import MB


def test_page_faults_count_the_pages_a_process_touches() -> None:
    """a Linux kernel with transparent huge pages on backs an anonymous region with 2 MB pages and faults 64 MB
    in 33 times, so the region is asked for base pages"""
    import mmap

    from btb.sysinfo import page_faults

    m = mmap.mmap(-1, 64 * MB)
    if hasattr(m, "madvise") and hasattr(mmap, "MADV_NOHUGEPAGE"):
        m.madvise(mmap.MADV_NOHUGEPAGE)
    f0 = page_faults()
    for k in range(0, len(m), 4096):
        m[k] = 1
    assert page_faults() - f0 >= 64, "touching 64 MB of fresh pages faults them in"
    m.close()


def test_the_process_counters_hold_together() -> None:
    """the hard faults are among the faults, the working set is at most the peak, every counter only climbs,
    and a file read shows up in the bytes the process read (Windows counts every ReadFile; Linux's figure is
    the storage layer's, which a cached file never reaches)"""
    import tempfile

    from btb.sysinfo import hard_page_faults, page_faults, peak_rss_bytes, process_read_bytes, process_working_set_bytes

    h0, f0 = hard_page_faults(), page_faults()
    assert 0 <= h0 <= f0
    # Linux keeps the resident count in per-thread counters synced every 64 events, so /proc's figure and
    # getrusage's peak can stand a few hundred KB apart at one instant
    ws, peak = process_working_set_bytes(), peak_rss_bytes()
    assert 0 < ws <= peak + (4 * MB if sys.platform == "linux" else 0), (ws, peak)
    r0 = process_read_bytes()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(os.urandom(8 * MB))
        name = f.name
    try:
        with open(name, "rb") as f:
            while f.read(MB):
                pass
    finally:
        os.remove(name)
    r1 = process_read_bytes()
    assert r1 >= r0
    if sys.platform == "win32":
        assert r1 - r0 >= 8 * MB
    assert hard_page_faults() >= h0 and page_faults() >= f0


def test_the_os_speaks_for_its_own_memory() -> None:
    """the floor the OS keeps is its published figure or nothing (never a share of the box), and its word on
    pressure is a plain yes or no with the figure it was read from"""
    from btb.sysinfo import host_total_bytes, memory_pressure, os_memory_floor

    floor = os_memory_floor()
    assert 0 <= floor < host_total_bytes() // 10
    if sys.platform != "linux":
        assert floor == 0
    p = memory_pressure()
    assert set(p) == {"low", "level"} and isinstance(p["low"], bool) and isinstance(p["level"], float)
    assert p["level"] >= 0.0


def test_raise_file_limit_lifts_a_low_soft_limit() -> None:
    """macOS starts a process at 256 open files, fewer than the drive readers' handle per thread per shard"""
    if sys.platform == "win32":
        pytest.skip("the file limit is a POSIX rlimit")
    import resource

    from btb.sysinfo import raise_file_limit

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        assert raise_file_limit(4096) == want
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == want
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
