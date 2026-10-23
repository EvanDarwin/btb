# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb_native`'s Metal entries (macOS) through ctypes: a context's buffers of its own or aliases of host memory, a
library, its pipelines and its indirect command buffers, each by index. Torch- and MLX-free."""

from __future__ import annotations

import ctypes
import resource
import weakref
from collections.abc import Sequence

import numpy as np

from .native_files import native_path

PAGE = resource.getpagesize()


def paged(n: int) -> int:
    """`n` bytes rounded up to whole pages (one at least): memory a context can alias"""
    return max(PAGE, -(-int(n) // PAGE) * PAGE)


class MetalError(RuntimeError):
    """a native Metal entry's failure: its code and Metal's own message"""


class Metal:
    """one context of the native Metal entries, closed with everything in it once the object is let go; every handle
    it returns indexes into it"""

    def __init__(self) -> None:
        p = native_path()
        if p is None:
            raise MetalError("[metal] no native library in this install")
        lib = ctypes.CDLL(p)
        if not hasattr(lib, "btb_metal_open"):
            raise MetalError("[metal] the native library has no Metal entries (built off macOS, or before them)")
        i64, i32, u32, u64, ptr, txt = (
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        size = ctypes.c_size_t
        sigs: dict[str, tuple[list[type], type]] = {
            "open": ([], i64),
            "close": ([i64], i32),
            "buffer": ([i64, u64], i64),
            "alias": ([i64, ptr, u64], i64),
            "free_buffer": ([i64, i64], i32),
            "free_icb": ([i64, i64], i32),
            "contents": ([i64, i64, ctypes.POINTER(ptr), ctypes.POINTER(u64)], i32),
            "compile": ([i64, txt, i32, txt, size], i32),
            "pipeline": ([i64, txt, txt, size], i64),
            "icb": ([i64, u32, u32], i64),
            "cmd": ([i64, i64, u32, i64, u32, ctypes.POINTER(i64), ctypes.POINTER(u64), u32, u32, u32, u32, i32], i32),
            "run": ([i64, i64, u32, ctypes.POINTER(ctypes.c_double), txt, size], i32),
        }
        for name, (args, res) in sigs.items():
            f = getattr(lib, f"btb_metal_{name}")
            f.argtypes, f.restype = args, res
        self.lib = lib
        self.err = ctypes.create_string_buffer(1 << 14)
        self.h = self._ok(lib.btb_metal_open(), "open")
        weakref.finalize(self, lib.btb_metal_close, self.h)

    def _ok(self, rc: int, what: str) -> int:
        if rc < 0:
            raise MetalError(f"[metal] {what}: code {rc} {self.err.value.decode(errors='replace')}")
        return int(rc)

    def buffer(self, nbytes: int) -> int:
        """a zeroed buffer of the context's own"""
        return self._ok(self.lib.btb_metal_buffer(self.h, nbytes), "buffer")

    def alias(self, view: np.ndarray) -> int:
        """a buffer over `view`'s own bytes, which must be whole pages at a page and outlive the buffer"""
        at, n = int(view.ctypes.data), int(view.nbytes)
        if at % PAGE or n % PAGE:
            raise ValueError(f"[metal] {n} bytes at {at:#x} are not whole pages at a page: they cannot be aliased")
        return self._ok(self.lib.btb_metal_alias(self.h, at, n), "alias")

    def _contents(self, b: int) -> tuple[int, int]:
        p, n = ctypes.c_void_p(), ctypes.c_uint64()
        self._ok(self.lib.btb_metal_contents(self.h, b, ctypes.byref(p), ctypes.byref(n)), "contents")
        return int(p.value or 0), int(n.value)

    def write(self, b: int, data: np.ndarray) -> None:
        raw = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        at, n = self._contents(b)
        if raw.size > n:
            raise ValueError(f"[metal] {raw.size} bytes into a buffer of {n}")
        ctypes.memmove(at, raw.ctypes.data, raw.size)

    def read(self, b: int, count: int, dtype: type[np.generic]) -> np.ndarray:
        out = np.empty(count, dtype=dtype)
        at, n = self._contents(b)
        if out.nbytes > n:
            raise ValueError(f"[metal] {out.nbytes} bytes out of a buffer of {n}")
        ctypes.memmove(out.ctypes.data, at, out.nbytes)
        return out

    def compile(self, src: str) -> None:
        """`src` as the context's library, under safe math (fast math changes results)"""
        self._ok(self.lib.btb_metal_compile(self.h, src.encode(), 0, self.err, len(self.err)), "compile")

    def pipeline(self, name: str) -> int:
        return self._ok(self.lib.btb_metal_pipeline(self.h, name.encode(), self.err, len(self.err)), f"pipeline {name}")

    def icb(self, cmds: int, binds: int) -> int:
        return self._ok(self.lib.btb_metal_icb(self.h, cmds, binds), "indirect command buffer")

    def cmd(
        self,
        icb: int,
        k: int,
        pso: int,
        binds: ctypes.Array[ctypes.c_int64],
        offs: Sequence[int],
        grid: Sequence[int],
        threads: int,
        barrier: bool,
    ) -> None:
        """command `k` of `icb`: pipeline `pso` over buffers `binds` at byte offsets `offs`, `grid` threadgroups of
        `threads`, after the commands before it when `barrier`"""
        o = (ctypes.c_uint64 * len(offs))(*offs)
        rc = self.lib.btb_metal_cmd(self.h, icb, k, pso, len(binds), binds, o, *grid, threads, int(barrier))
        self._ok(rc, f"command {k}")

    def run(self, icb: int, n: int) -> float:
        """the first `n` commands of `icb` replayed and waited for: the GPU's seconds"""
        g = ctypes.c_double()
        self._ok(self.lib.btb_metal_run(self.h, icb, n, ctypes.byref(g), self.err, len(self.err)), "run")
        return float(g.value)

    def free_buffer(self, b: int) -> None:
        self._ok(self.lib.btb_metal_free_buffer(self.h, b), "free buffer")

    def free_icb(self, i: int) -> None:
        self._ok(self.lib.btb_metal_free_icb(self.h, i), "free indirect command buffer")
