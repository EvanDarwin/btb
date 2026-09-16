# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The machine as the engine reads it, with nothing loaded. `btb.sysinfo` is the torch-free layer under the
memory system: free and total RAM, the commit charge, the OS's own floor and its pressure signal, the
card's memory and who holds it, this process's working set, page faults and bytes read.
`btb.host_budget()` is the same reading as a plan takes it.
"""

import sys

import btb
import btb.sysinfo as si


def main(argv=None):
    gb = lambda b: f"{b / 2**30:.2f} GB"
    print(f"RAM {gb(si.host_free_bytes())} free of {gb(si.host_total_bytes())}; commit {gb(si.host_commit_bytes())}")
    print(f"the OS keeps {gb(si.os_memory_floor())} for itself; pressure {si.memory_pressure()}")
    print(f"card: {si.vram_pressure_line() or 'none'}")
    print(
        f"this process: {si.process_working_set_bytes() / 2**20:.0f} MB working set, peak {si.peak_rss_bytes() / 2**20:.0f} MB, "
        f"{si.page_faults()} page faults ({si.hard_page_faults()} hard), {si.process_read_bytes() / 2**20:.0f} MB read"
    )
    print(f"caches: {si.host_cache_sizes()}; torch imported so far: {'torch' in sys.modules}")
    b = btb.host_budget()
    print(f"host budget: {gb(b.available)} available, floor {gb(b.floor)} (the OS's {gb(b.os_floor)})")
    return {"free": si.host_free_bytes(), "pressure": si.memory_pressure(), "budget": b}


if __name__ == "__main__":
    main()
