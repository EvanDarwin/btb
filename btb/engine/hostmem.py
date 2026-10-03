# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Pinned host memory the card reads and writes in place, at its exact size: `cudaHostAlloc` through the CUDA runtime
torch loaded, wrapped as a CPU tensor.

torch's own pinned memory (`pin_memory=True`) comes from its caching host allocator, which rounds a block up to a
power of two and keeps a freed one for reuse: 1.5 GB took 2 GB here, and kept it after it was let go. A card program's
attention rows kept in RAM (`kv_host`) are tens of GB at a million positions - 20 GB taken as 32, and a regrowth
holding both - so they are allocated here, and handed back to the driver when the last view of them is gone.

The kernels read the memory at its host address: under unified addressing a `cudaHostAlloc` block is the card's at
the host's pointer (the card's kernels already publish to torch's pinned memory so). Memory pinned by
`cudaHostRegister` is not: this card cannot use a registered block's host pointer (CAN_USE_HOST_POINTER_FOR_REGISTERED_
MEM 0 under WDDM), its device address another."""

from __future__ import annotations

import ctypes
import glob
import math
import os
import threading
import weakref
from typing import Any

import torch

_CUDART: dict[str, Any] = {}
_LOCK = threading.Lock()

# cudaHostAllocPortable: the block pinned for every context, not only the one current when it was made
_PORTABLE = 1


def _runtime_path() -> str | None:
    """the CUDA runtime library torch loaded: beside torch (Windows' wheels, a bundled Linux one - `libcudart-<hash>`
    there), else the one the process has mapped (a Linux wheel's from the nvidia-cuda-runtime package, which torch
    loads by its own path), else that package's own. None where none is found"""
    here = os.path.join(os.path.dirname(torch.__file__), "lib")
    for pat in ("cudart64_*.dll", "libcudart.so*", "libcudart-*.so*"):
        names = sorted(glob.glob(os.path.join(here, pat)))
        if names:
            return names[-1]
    if os.path.exists("/proc/self/maps"):
        torch.cuda.init()  # the runtime mapped before the process is asked for it
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 5)  # address, perms, offset, dev, inode, then the path where one is mapped
                path = parts[5].strip() if len(parts) == 6 else ""
                if "libcudart" in os.path.basename(path) and os.path.exists(path):
                    return path
    try:
        import nvidia.cuda_runtime as rt  # type: ignore[import-not-found]

        names = sorted(glob.glob(os.path.join(os.path.dirname(rt.__file__), "lib", "libcudart.so*")))
        if names:
            return names[-1]
    except ImportError:
        pass
    return None


def can_pin() -> bool:
    """whether `pinned` has a CUDA runtime to pin memory with (`_runtime_path`)"""
    try:
        _cudart()
    except RuntimeError:
        return False
    return True


def _cudart() -> Any:
    """the CUDA runtime torch loaded (the one library instance, so the same context; `_runtime_path`)"""
    with _LOCK:
        lib = _CUDART.get("lib")
        if lib is None:
            path = _runtime_path()
            if path is None:
                raise RuntimeError("[hostmem] torch's CUDA runtime library could not be found")
            lib = ctypes.CDLL(path)
            lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
            lib.cudaHostAlloc.restype = ctypes.c_int
            lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
            lib.cudaFreeHost.restype = ctypes.c_int
            _CUDART["lib"] = lib
        return lib


def _free(ptr: int) -> None:
    lib = _CUDART.get("lib")
    if lib is not None:
        lib.cudaFreeHost(ctypes.c_void_p(ptr))


def pinned(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """an uninitialized CPU tensor of `shape` and `dtype` in pinned memory of exactly its bytes, which the card's
    kernels read and write at its host pointer; the memory freed once no view of it is left. A MemoryError where the
    driver will not pin that much"""
    torch.cuda.init()  # the runtime's context made before the first block is pinned for it
    n = math.prod(int(s) for s in shape) * torch.empty(0, dtype=dtype).element_size()
    ptr = ctypes.c_void_p()
    err = _cudart().cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(max(1, n)), _PORTABLE)
    if err != 0 or not ptr.value:
        raise MemoryError(f"[hostmem] cudaHostAlloc of {n / 2**30:.2f} GiB failed (cudaError {err})")
    buf = (ctypes.c_uint8 * max(1, n)).from_address(ptr.value)
    # the tensor holds the buffer, the buffer the block: freed with the last view of it
    weakref.finalize(buf, _free, ptr.value)
    t = torch.frombuffer(buf, dtype=torch.uint8, count=n) if n else torch.empty(0, dtype=torch.uint8)
    return t.view(dtype).view(*shape)
