# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What the megakernel knows of the GPU under it: MLX's name and architecture for the device, its core count from the
IO registry where it has one, and Metal's threadgroup limits. How many of the kernel's threadgroups run at once is
measured on the compiled kernel (`kernel.Kernel.resident`), never assumed from a chip table."""

from __future__ import annotations

import functools
import plistlib
import subprocess
from dataclasses import dataclass

from ..core import mx

# Metal's limits on every Apple GPU family: a threadgroup's memory (MTLDevice.maxThreadgroupMemoryLength) and the
# simdgroup's width
TG_MEMORY = 32768
SIMD = 32


@dataclass(frozen=True)
class Gpu:
    name: str  # "Apple M3 Pro"
    arch: str  # MLX's architecture string: "applegpu_g15s" is generation 15 (M3), variant s (Pro)
    cores: int | None  # the IO registry's gpu-core-count; None without an AGX accelerator (a virtual GPU)
    memory: int  # the working set Metal recommends, bytes


@functools.cache
def gpu() -> Gpu:
    """the device this process's MLX runs on, read once"""
    m = mx()
    info = (getattr(m, "device_info", None) or m.metal.device_info)()
    return Gpu(
        str(info.get("device_name", "")),
        str(info.get("architecture", "")),
        _cores(),
        int(info.get("max_recommended_working_set_size", 0)),
    )


def _cores() -> int | None:
    try:
        out = subprocess.run(
            ["ioreg", "-rc", "AGXAccelerator", "-d1", "-a"], capture_output=True, timeout=5, check=False
        ).stdout
        for d in plistlib.loads(out) if out else []:
            n = d.get("gpu-core-count") if isinstance(d, dict) else None
            if isinstance(n, int) and n > 0:
                return n
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError):
        return None
    return None
