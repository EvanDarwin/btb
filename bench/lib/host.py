# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The machine: what the numbers were measured on, its memory as the kernel sees it, the card's used
memory, the process's own peak
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys

from bench.lib import ROOT, capture
from bench.lib.records import BenchMemoryState, BenchSpecs

if sys.platform != "win32":
    import resource

    def _rusage_peak() -> int:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r) * (1 if sys.platform == "darwin" else 1024)

else:

    def _rusage_peak() -> int:
        return 0


def host_specs() -> BenchSpecs:
    """
    What the numbers were measured on: CPU, RAM, card, OS, and the versions of what ran
    """
    import torch

    from btb import mlx_available
    from btb.sysinfo import host_cores, host_cpu_name, host_total_bytes

    s: BenchSpecs = {
        "host": socket.gethostname(),
        "platform": sys.platform,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu": host_cpu_name(),
        "cores": host_cores(),
        "ram_gb": round(host_total_bytes() / 2**30),
    }
    try:
        import transformers

        s["transformers"] = transformers.__version__
    except Exception:
        pass
    if sys.platform == "darwin":
        s["os"] = "macOS " + platform.mac_ver()[0]
    elif sys.platform == "win32":
        s["os"] = f"Windows {platform.release()} ({platform.version()})"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        s["gpu"] = p.name
        s["vram_gb"] = round(p.total_memory / 2**30, 1)
    if mlx_available():
        try:
            import mlx.core as mx

            from btb import mlx as mlxdev

            info = mlxdev.info()
            s["gpu"] = info.get("device_name", "Apple GPU")
            s["unified_memory_gb"] = round(info.get("memory_size", 0) / 2**30)
            s["mlx"] = getattr(mx, "__version__", "")
        except Exception:
            pass
    return s


def memory_state() -> BenchMemoryState:
    """
    The machine's memory as the kernel sees it: `level` its free-memory percentage (macOS's
    kern.memorystatus_level, the figure its own killer reads; 100 where unknown), `swap_gb` the swap in use,
    `disk_gb` the free space on the results' volume (swap files fill it)
    """
    from btb.sysinfo import memory_level, swap_used_bytes

    return {
        "level": memory_level(),
        "swap_gb": swap_used_bytes() / 2**30,
        "disk_gb": shutil.disk_usage(ROOT).free / 2**30,
    }


# WARN: this must only be called _after_ the external tool has run,
#   to prevent btb's loading from affecting their run
def peak_rss_bytes() -> int:
    """
    Retrieve btb's own peak-working-set reader
    """
    try:
        from btb.sysinfo import peak_rss_bytes as reader

        return reader()
    except Exception:
        return _rusage_peak()


def gpu_used_gb() -> float:
    """
    The card's used memory in GB (the whole board). Windows WDDM does not report per-process card memory,
    so a run's own peak is taken as the rise above the baseline measured before the model loads
    """
    out = capture(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=10).splitlines()
    if out and out[0].strip().isdigit():
        return int(out[0].strip()) / 1024.0
    return 0.0


def add_cuda_dll_dirs() -> None:
    """
    llama.cpp's CUDA build links the CUDA runtime by soname; on Windows those DLLs are the toolkit's
    (under bin\\x64 on CUDA 13, bin on 12) and a CUDA torch build's (torch/lib). Put whatever is present on
    the search path so ggml-cuda.dll resolves - the wheel bundles ggml-cuda but not the runtime it needs
    """
    if sys.platform != "win32":
        return
    dirs = []
    cp = os.environ.get("CUDA_PATH")
    if cp:
        dirs += [os.path.join(cp, "bin", "x64"), os.path.join(cp, "bin")]
    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(root):
        for v in sorted(os.listdir(root), reverse=True):
            dirs += [os.path.join(root, v, "bin", "x64"), os.path.join(root, v, "bin")]
    try:
        import torch

        dirs.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    seen = set()
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d)
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
