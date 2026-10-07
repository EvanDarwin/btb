# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The native kernels' handles (the C-ABI library loaded through ctypes) and the MLX backend, shared by every
module of the engine: `Native.load` binds them once per process."""

from __future__ import annotations

import contextlib
import ctypes
import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .. import mlx as mlxdev
from ..native_files import kernels_path, native_path
from ..sysinfo import raise_file_limit


def isa() -> str:
    """the kernel tier the native library selected for this process (avx512/avx2/neon/scalar; `BTB_NATIVE_ISA`
    honored) - what a benchmark's numbers belong to"""
    p = native_path()
    if p is None:
        raise NativeError("btb_isa", -1, ": no native library in this install")
    f = ctypes.CDLL(p).btb_isa
    f.restype = ctypes.c_char_p
    return f().decode()


class NativeError(RuntimeError):
    """A call into the native library failed: `call` its name, `rc` what it returned, `detail` what it was
    asked (a path, an offset), `errno` the OS error it left when the call reads the OS (else 0)."""

    def __init__(self, call: str, rc: int, detail: str = "", os_error: bool = False) -> None:
        self.call, self.rc, self.detail = call, int(rc), detail
        self.errno = ctypes.get_errno() if os_error else 0
        super().__init__(f"{call} returned {rc}{detail}{f': {os.strerror(self.errno)}' if self.errno else ''}")


class NativeDtypeError(TypeError):
    """A tensor handed to a native kernel is not the element type the kernel reads through its pointer: `call`
    the kernel, `arg` the argument, `got` its dtype and `want` what the kernel was built for. Raised before the
    call by a binding loaded with `checked`, so a wrong tensor is refused rather than read as another type's."""

    def __init__(self, call: str, arg: str, got: torch.dtype, want: tuple[torch.dtype, ...]) -> None:
        self.call, self.arg, self.got, self.want = call, arg, got, want
        super().__init__(f"{call}: {arg} is {got}, the kernel reads {' or '.join(str(w) for w in want)}")


_BF16, _F32, _U8, _I32 = torch.bfloat16, torch.float32, torch.uint8, torch.int32
# what one argument must be: a tensor (or each of a group's tensors) and the dtypes the kernel reads it as
Want = tuple[torch.Tensor | Sequence[torch.Tensor], tuple[torch.dtype, ...]]

# each fixed-type binding: its kernel, and its arguments' wants from the call's own arguments, in the order a
# refusal names the first wrong one
_WANTS: dict[str, tuple[str, Callable[..., dict[str, Want]]]] = {
    "gemv": ("btb_gemv_bf16_rows", lambda w, x, y: {"w": (w, (_BF16,)), "x": (x, (_F32,)), "y": (y, (_F32,))}),
    "gemv_p12": (
        "btb_gemv_p12_rows",
        lambda lo, hi4, tbl, esc_idx, esc_val, n_esc, rows, cols, x, y: {
            "lo": (lo, (_U8,)),
            "hi4": (hi4, (_U8,)),
            "table": (tbl, (_U8,)),
            **({"esc_idx": (esc_idx, (_I32,)), "esc_val": (esc_val, (_U8,))} if n_esc else {}),
            "x": (x, (_F32,)),
            "y": (y, (_F32,)),
        },
    ),
    "gemv_group": (
        "btb_gemv_bf16_group",
        lambda ws, xs, ys: {"w": (ws, (_BF16,)), "x": (xs, (_F32,)), "y": (ys, (_F32,))},
    ),
    "gemv_mx4": (
        "btb_gemv_mxfp4_rows",
        lambda w, x, y: {
            "blocks": (w.blocks, (_U8,)),
            "scales": (w.scales, (_U8,)),
            "x": (x, (_F32,)),
            "y": (y, (_F32,)),
        },
    ),
    "gemv_mx4_group": (
        "btb_gemv_mxfp4_group",
        lambda ws, xs, ys: {
            "blocks": ([w.blocks for w in ws], (_U8,)),
            "scales": ([w.scales for w in ws], (_U8,)),
            "x": (xs, (_F32,)),
            "y": (ys, (_F32,)),
        },
    ),
    "gemv_mx4_ggml": (
        "btb_gemv_mxfp4_ggml_rows",
        lambda w, x, y: {"blocks": (w.blocks, (_U8,)), "x": (x, (_F32,)), "y": (y, (_F32,))},
    ),
    "gemv_mx4_ggml_group": (
        "btb_gemv_mxfp4_ggml_group",
        lambda ws, xs, ys: {"blocks": ([w.blocks for w in ws], (_U8,)), "x": (xs, (_F32,)), "y": (ys, (_F32,))},
    ),
    "gemv_fp8": (
        "btb_gemv_fp8_rows",
        lambda w, x, y: {"w": (w.w, (_U8,)), "scales": (w.scales, (_F32,)), "x": (x, (_F32,)), "y": (y, (_F32,))},
    ),
    "gemv_fp8_group": (
        "btb_gemv_fp8_group",
        lambda ws, xs, ys: {
            "w": ([w.w for w in ws], (_U8,)),
            "scales": ([w.scales for w in ws], (_F32,)),
            "x": (xs, (_F32,)),
            "y": (ys, (_F32,)),
        },
    ),
    "delta_step": (
        "btb_delta_step",
        lambda mixed, conv_state, conv_w, conv_b, z, a, b, a_log, dt_bias, state, hk, hv, dk, dv, norm_w, eps, out, gate=0: {
            "conv_b": ([] if conv_b is None else conv_b, (_F32,)),
            **{
                k: (t, (_F32,))
                for k, t in (
                    ("mixed", mixed),
                    ("conv_state", conv_state),
                    ("conv_w", conv_w),
                    ("z", z),
                    ("a", a),
                    ("b", b),
                    ("a_log", a_log),
                    ("dt_bias", dt_bias),
                    ("state", state),
                    ("norm_w", norm_w),
                    ("out", out),
                )
            },
        },
    ),
    "attn_decode": (
        "btb_attn_decode",
        lambda q, k, v, scale, out: {
            "q": (q, (_F32,)),
            "k": (k, (_BF16, _F32)),
            "v": (v, (k.dtype,)),
            "out": (out, (_F32,)),
        },
    ),
    "attn_nodes": (
        "btb_attn_nodes",
        lambda q, k, v, offs, idx, scale, out: {
            "q": (q, (_F32,)),
            "k": (k, (_BF16, _F32)),
            "v": (v, (k.dtype,)),
            "offs": (offs, (_I32,)),
            "idx": (idx, (_I32,)),
            "out": (out, (_F32,)),
        },
    ),
}


def _checked(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """the binding `name`, refusing the first argument whose dtype its kernel does not read before calling it"""
    call, wants = _WANTS[name]

    def run(*args: Any, **kw: Any) -> Any:
        for arg, (ts, want) in wants(*args, **kw).items():
            for t in [ts] if isinstance(ts, torch.Tensor) else ts:
                if t.dtype not in want:
                    raise NativeDtypeError(call, arg, t.dtype, want)
        return fn(*args, **kw)

    return run


def quiet_omp() -> None:
    """Load libiomp, silently fail if not found"""
    p = os.path.join(os.path.dirname(torch.__file__), "lib", "libiomp5md.dll")
    if os.path.exists(p):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(p).kmp_set_blocktime(0)


class _Binding(type):
    """A kernel handle read before any load binds the install's library first (`Native.bind_install`), as
    `btb.load` binds it: a handle is never None only because no model has loaded yet in this process - None means
    the library lacks that kernel, or a load chose none (`native=''`, `Native.unbind`)."""

    def __getattr__(cls, name: str) -> Any:
        # reached only for a handle not bound yet: a bound one (None included) is a plain class attribute
        if name in type.__getattribute__(cls, "HANDLES"):
            type.__getattribute__(cls, "bind_install")()
            return type.__getattribute__(cls, name)
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")


class Native(metaclass=_Binding):
    """Process-wide handles: the native kernels `load_gemv` bound (bound on the first read when nothing bound them
    yet; None where the library lacks one, or with no library), the MLX backend once an engine made one, and the
    row thresholds routing a matmul between kernels and GEMMs."""

    _gemv_lib: Any = None
    HANDLES = (
        "gemv",
        "gemv_p12",
        "gemv_group",
        "gemv_mx4",
        "gemv_mx4_group",
        "gemv_mx4_ggml",
        "gemv_mx4_ggml_group",
        "gemv_fp8",
        "gemv_fp8_group",
        "attn_decode",
        "attn_nodes",
        "delta_step",
        "sample_pick",
        "read_direct",
        "open",
        "open_cached",
        "read_at",
        "close",
    )

    # the handles: unset until bound (the metaclass binds them on the first read), each then a kernel or None
    gemv: Any
    gemv_p12: Any
    gemv_group: Any
    gemv_mx4: Any
    gemv_mx4_group: Any
    gemv_mx4_ggml: Any
    gemv_mx4_ggml_group: Any
    gemv_fp8: Any
    gemv_fp8_group: Any
    attn_decode: Any
    attn_nodes: Any
    delta_step: Any
    sample_pick: Any
    read_direct: Any
    open: Any
    open_cached: Any
    read_at: Any
    close: Any
    # the MLX backend (`mlxdev.Backend`) of the engine on the MLX device
    mlx: Any = None
    # a matmul of this many rows or more goes to the GEMM; fewer rows take the one-row kernel per row
    gemm_rows = 64

    # a prefill of this many rows or more goes to the CPU-stream GEMM; fewer (one-row steps, verify passes of <= 16
    # rows) stay on the native kernel, so a verify row computes as the one-row step does
    cpu_gemm_rows = 17

    @classmethod
    def bind_install(cls) -> None:
        """the install's library bound as `btb.load` binds it (the kernels' own pool over every core), or every
        handle None where this install has none"""
        from ..native_files import native_path

        p = native_path()
        if p:
            cls.load_gemv(p)
        else:
            cls.unbind()

    @classmethod
    def unbind(cls) -> None:
        """every handle None: the torch-alone path (`load(native='')`), whatever an earlier load bound"""
        for name in cls.HANDLES:
            setattr(cls, name, None)
        cls._gemv_lib = None

    @classmethod
    def load_gemv(
        cls, dll_path: str | os.PathLike[str], threads: int = 0, checked: bool = False
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
        """Bind the library's kernels onto the class; `checked` wraps each fixed-type one to refuse a tensor of
        another dtype (`NativeDtypeError`) - a debugging aid, off by default, so a decode step pays nothing for
        what the engine fixes at load. Returns the bf16 gemv."""
        lib = ctypes.CDLL(dll_path, use_errno=True)
        raise_file_limit()
        f = lib.btb_gemv_bf16_rows
        f.restype = ctypes.c_int32
        f.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]

        def gemv(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
            rc = f(w.data_ptr(), w.shape[0], w.shape[1], x.data_ptr(), x.shape[0], y.data_ptr(), threads)
            if rc != 0:
                raise NativeError("btb_gemv_bf16_rows", rc)

        cls.gemv = gemv
        cls._gemv_lib = lib
        cls.gemv_p12 = None
        if hasattr(lib, "btb_gemv_p12_rows"):
            g = lib.btb_gemv_p12_rows
            g.restype = ctypes.c_int32
            g.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]

            def gemv_p12(
                lo: torch.Tensor,
                hi4: torch.Tensor,
                tbl: torch.Tensor,
                esc_idx: torch.Tensor,
                esc_val: torch.Tensor,
                n_esc: int,
                rows: int,
                cols: int,
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> None:
                rc = g(
                    lo.data_ptr(),
                    hi4.data_ptr(),
                    tbl.data_ptr(),
                    esc_idx.data_ptr() if n_esc else 0,
                    esc_val.data_ptr() if n_esc else 0,
                    n_esc,
                    rows,
                    cols,
                    x.data_ptr(),
                    x.shape[0],
                    y.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_p12_rows", rc)

            cls.gemv_p12 = gemv_p12
        cls.gemv_group = None
        if hasattr(lib, "btb_gemv_bf16_group"):
            gg = lib.btb_gemv_bf16_group
            gg.restype = ctypes.c_int32
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            gg.argtypes = [
                S,
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                S,
            ]

            def gemv_group(ws: Sequence[torch.Tensor], xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> None:
                n = len(ws)
                rc = gg(
                    n,
                    (P * n)(*[w.data_ptr() for w in ws]),
                    (S * n)(*[w.shape[0] for w in ws]),
                    (S * n)(*[w.shape[1] for w in ws]),
                    (P * n)(*[x.data_ptr() for x in xs]),
                    (S * n)(*[x.shape[0] for x in xs]),
                    (P * n)(*[y.data_ptr() for y in ys]),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_bf16_group", rc)

            cls.gemv_group = gemv_group
        cls.gemv_mx4 = None
        if hasattr(lib, "btb_gemv_mxfp4_rows"):
            mx = lib.btb_gemv_mxfp4_rows
            mx.restype = ctypes.c_int32
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            mx.argtypes = [P, P, S, S, P, S, P, S]

            def gemv_mx4(w: Any, x: torch.Tensor, y: torch.Tensor) -> None:
                rows, cols = w.shape
                rc = mx(
                    w.blocks.data_ptr(),
                    w.scales.data_ptr(),
                    rows,
                    cols,
                    x.data_ptr(),
                    x.shape[0],
                    y.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_mxfp4_rows", rc)

            cls.gemv_mx4 = gemv_mx4
        cls.gemv_mx4_group = None
        if hasattr(lib, "btb_gemv_mxfp4_group"):
            mg = lib.btb_gemv_mxfp4_group
            mg.restype = ctypes.c_int32
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            mg.argtypes = [
                S,
                ctypes.POINTER(P),
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                S,
            ]

            def gemv_mx4_group(ws: Sequence[Any], xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> None:
                n = len(ws)
                rc = mg(
                    n,
                    (P * n)(*[w.blocks.data_ptr() for w in ws]),
                    (P * n)(*[w.scales.data_ptr() for w in ws]),
                    (S * n)(*[w.shape[0] for w in ws]),
                    (S * n)(*[w.shape[1] for w in ws]),
                    (P * n)(*[x.data_ptr() for x in xs]),
                    (S * n)(*[x.shape[0] for x in xs]),
                    (P * n)(*[y.data_ptr() for y in ys]),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_mxfp4_group", rc)

            cls.gemv_mx4_group = gemv_mx4_group
        cls.gemv_mx4_ggml = cls.gemv_mx4_ggml_group = None
        if hasattr(lib, "btb_gemv_mxfp4_ggml_rows") and hasattr(lib, "btb_gemv_mxfp4_ggml_group"):
            # the same matvec over ggml's block layout (a GGUF's experts): `w.blocks` the 17-byte blocks
            mgr, mgg = lib.btb_gemv_mxfp4_ggml_rows, lib.btb_gemv_mxfp4_ggml_group
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            mgr.restype = mgg.restype = ctypes.c_int32
            mgr.argtypes = [P, S, S, P, S, P, S]
            mgg.argtypes = [
                S,
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                ctypes.POINTER(S),
                ctypes.POINTER(P),
                S,
            ]

            def gemv_mx4_ggml(w: Any, x: torch.Tensor, y: torch.Tensor) -> None:
                rows, cols = w.shape
                rc = mgr(w.blocks.data_ptr(), rows, cols, x.data_ptr(), x.shape[0], y.data_ptr(), threads)
                if rc != 0:
                    raise NativeError("btb_gemv_mxfp4_ggml_rows", rc)

            def gemv_mx4_ggml_group(ws: Sequence[Any], xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> None:
                n = len(ws)
                rc = mgg(
                    n,
                    (P * n)(*[w.blocks.data_ptr() for w in ws]),
                    (S * n)(*[w.shape[0] for w in ws]),
                    (S * n)(*[w.shape[1] for w in ws]),
                    (P * n)(*[x.data_ptr() for x in xs]),
                    (S * n)(*[x.shape[0] for x in xs]),
                    (P * n)(*[y.data_ptr() for y in ys]),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_mxfp4_ggml_group", rc)

            cls.gemv_mx4_ggml, cls.gemv_mx4_ggml_group = gemv_mx4_ggml, gemv_mx4_ggml_group
        cls.gemv_fp8 = cls.gemv_fp8_group = None
        if hasattr(lib, "btb_gemv_fp8_rows") and hasattr(lib, "btb_gemv_fp8_group"):
            # an FP8 matrix as stored (btb/fp8.py's F8Weight): its e4m3 bytes and its f32 scale grid
            f8r, f8g = lib.btb_gemv_fp8_rows, lib.btb_gemv_fp8_group
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            f8r.restype = f8g.restype = ctypes.c_int32
            PP, PS = ctypes.POINTER(P), ctypes.POINTER(S)
            f8r.argtypes = [P, P, S, S, S, S, P, S, P, S]
            f8g.argtypes = [S, PP, PP, PS, PS, PS, PS, PP, PS, PP, S]

            def gemv_fp8(w: Any, x: torch.Tensor, y: torch.Tensor) -> None:
                (rows, cols), (sr, sc) = w.shape, w.grid
                rc = f8r(
                    w.w.data_ptr(),
                    w.scales.data_ptr(),
                    rows,
                    cols,
                    sr,
                    sc,
                    x.data_ptr(),
                    x.shape[0],
                    y.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_fp8_rows", rc)

            def gemv_fp8_group(ws: Sequence[Any], xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> None:
                n = len(ws)
                rc = f8g(
                    n,
                    (P * n)(*[w.w.data_ptr() for w in ws]),
                    (P * n)(*[w.scales.data_ptr() for w in ws]),
                    (S * n)(*[w.shape[0] for w in ws]),
                    (S * n)(*[w.shape[1] for w in ws]),
                    (S * n)(*[w.grid[0] for w in ws]),
                    (S * n)(*[w.grid[1] for w in ws]),
                    (P * n)(*[x.data_ptr() for x in xs]),
                    (S * n)(*[x.shape[0] for x in xs]),
                    (P * n)(*[y.data_ptr() for y in ys]),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_gemv_fp8_group", rc)

            cls.gemv_fp8, cls.gemv_fp8_group = gemv_fp8, gemv_fp8_group
        cls.read_direct = None
        if hasattr(lib, "btb_read_direct"):
            rd = lib.btb_read_direct
            rd.restype = ctypes.c_int32
            rd.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]

            def read_direct(path: str | os.PathLike[str], off: int, nb: int, dst: torch.Tensor, chunk: int = 0) -> None:
                # the library takes a NUL-terminated UTF-16 path on every platform (wchar_t is 32-bit off Windows)
                rc = rd(str(path).encode("utf-16-le") + b"\0\0", int(off), int(nb), dst.data_ptr(), int(chunk))
                if rc != 0:
                    raise NativeError("btb_read_direct", rc, f" for {path} @ {off}+{nb}", os_error=True)

            cls.read_direct = read_direct
        cls.open = cls.open_cached = cls.read_at = cls.close = None
        if hasattr(lib, "btb_open") and hasattr(lib, "btb_read_at") and hasattr(lib, "btb_close"):
            op, ra, cl = lib.btb_open, lib.btb_read_at, lib.btb_close
            op.restype = ctypes.c_int64
            op.argtypes = [ctypes.c_char_p]
            ra.restype = ctypes.c_int32
            ra.argtypes = [
                ctypes.c_int64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_size_t,
            ]
            cl.restype = ctypes.c_int32
            cl.argtypes = [ctypes.c_int64]

            def open_file(path: str | os.PathLike[str]) -> int:
                """the file held open for unbuffered reads (share-read only), by handle; `close` releases it"""
                h = op(str(path).encode("utf-16-le") + b"\0\0")
                if h < 0:
                    raise NativeError("btb_open", h, f" for {path}", os_error=True)
                return int(h)

            def read_at(handle: int, off: int, nb: int, dst: torch.Tensor, chunk: int = 0, depth: int = 0) -> None:
                """`read_direct` on an open handle, from any number of threads at once; with `off`, `nb` and
                `dst` all 4096-aligned the drive writes `dst` itself"""
                rc = ra(int(handle), int(off), int(nb), dst.data_ptr(), int(chunk), int(depth))
                if rc != 0:
                    raise NativeError("btb_read_at", rc, f" for handle {handle} @ {off}+{nb}", os_error=True)

            def close_file(handle: int) -> None:
                rc = cl(int(handle))
                if rc != 0:
                    raise NativeError("btb_close", rc, f" for handle {handle}")

            cls.open, cls.read_at, cls.close = open_file, read_at, close_file
            if hasattr(lib, "btb_open_cached"):
                oc = lib.btb_open_cached
                oc.restype = ctypes.c_int64
                oc.argtypes = [ctypes.c_char_p]

                def open_cached(path: str | os.PathLike[str]) -> int:
                    """`open`, the file read through the system's file cache: a read fills it, and a read of
                    bytes it still holds is a copy out of RAM that no commit is charged for"""
                    h = oc(str(path).encode("utf-16-le") + b"\0\0")
                    if h < 0:
                        raise NativeError("btb_open_cached", h, f" for {path}", os_error=True)
                    return int(h)

                cls.open_cached = open_cached
        cls.delta_step = None
        if hasattr(lib, "btb_delta_step"):
            ds = lib.btb_delta_step
            ds.restype = ctypes.c_int32
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            ds.argtypes = [P, P, P, P, S, S, P, P, P, P, P, P, S, S, S, S, P, ctypes.c_float, ctypes.c_uint32, P, S]

            def delta_step(
                mixed: torch.Tensor,
                conv_state: torch.Tensor,
                conv_w: torch.Tensor,
                conv_b: torch.Tensor | None,
                z: torch.Tensor,
                a: torch.Tensor,
                b: torch.Tensor,
                a_log: torch.Tensor,
                dt_bias: torch.Tensor,
                state: torch.Tensor,
                hk: int,
                hv: int,
                dk: int,
                dv: int,
                norm_w: torch.Tensor,
                eps: float,
                out: torch.Tensor,
                gate: int = 0,
            ) -> None:
                """`gate`: the gated norm's activation, 0 silu (Qwen3.5), 1 sigmoid (Qwen4)"""
                rc = ds(
                    mixed.data_ptr(),
                    conv_state.data_ptr(),
                    conv_w.data_ptr(),
                    conv_b.data_ptr() if conv_b is not None else 0,
                    conv_w.shape[0],
                    conv_w.shape[1],
                    z.data_ptr(),
                    a.data_ptr(),
                    b.data_ptr(),
                    a_log.data_ptr(),
                    dt_bias.data_ptr(),
                    state.data_ptr(),
                    hk,
                    hv,
                    dk,
                    dv,
                    norm_w.data_ptr(),
                    float(eps),
                    int(gate),
                    out.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_delta_step", rc)

            cls.delta_step = delta_step
        cls.sample_pick = None
        if hasattr(lib, "btb_sample_pick"):
            sp = lib.btb_sample_pick
            sp.restype = ctypes.c_int32
            sp.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_float,
                ctypes.c_uint32,
                ctypes.c_float,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]

            def sample_pick(
                x: torch.Tensor, keys: Sequence[int], temperature: float, top_k: int, top_p: float
            ) -> torch.Tensor:
                """the token of every row of `x` [R, V] float32 on the CPU under one key a row; [R] int64"""
                x = x.contiguous().float()
                R, V = int(x.shape[0]), int(x.shape[1])
                if len(keys) != R:
                    raise ValueError(f"btb_sample_pick: {len(keys)} keys for {R} rows")
                ks = (ctypes.c_uint64 * R)(*[int(k) & 0xFFFFFFFFFFFFFFFF for k in keys])
                out = torch.empty(R, dtype=torch.int32)
                rc = sp(
                    x.data_ptr(),
                    R,
                    V,
                    ctypes.addressof(ks),
                    float(temperature),
                    int(top_k),
                    float(top_p),
                    out.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise RuntimeError(f"btb_sample_pick returned {rc}")
                return out.to(torch.int64)

            cls.sample_pick = sample_pick
        cls.attn_decode = None
        if hasattr(lib, "btb_attn_decode_bf16") and hasattr(lib, "btb_attn_decode_f32"):
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            fns = {}
            for name, dt in (("btb_attn_decode_bf16", torch.bfloat16), ("btb_attn_decode_f32", torch.float32)):
                fn = getattr(lib, name)
                fn.restype = ctypes.c_int32
                fn.argtypes = [P, P, P, S, S, S, S, S, S, ctypes.c_float, P, S]
                fns[dt] = fn

            def attn_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, out: torch.Tensor) -> None:
                fn = fns[k.dtype]
                hk, n, d = k.shape
                rc = fn(
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    n,
                    q.shape[0],
                    hk,
                    d,
                    k.stride(0),
                    v.stride(0),
                    float(scale),
                    out.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise RuntimeError(f"btb_attn_decode returned {rc}")

            cls.attn_decode = attn_decode
        cls.attn_nodes = None
        if hasattr(lib, "btb_attn_nodes_bf16") and hasattr(lib, "btb_attn_nodes_f32"):
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            nfns = {}
            for name, dt in (("btb_attn_nodes_bf16", torch.bfloat16), ("btb_attn_nodes_f32", torch.float32)):
                fn = getattr(lib, name)
                fn.restype = ctypes.c_int32
                fn.argtypes = [P, P, P, P, P, S, S, S, S, S, S, S, ctypes.c_float, P, S]
                nfns[dt] = fn

            def attn_nodes(
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                offs: torch.Tensor,
                idx: torch.Tensor,
                scale: float,
                out: torch.Tensor,
            ) -> None:
                """`attn_decode` for T queries at once, each over its own list of cache rows: `q` and `out`
                [T, hq, d] float32, `k`/`v` [hk, n_rows, d] as `attn_decode` takes them, `offs` [T + 1] and `idx`
                int32, query t attending the rows `idx[offs[t]:offs[t + 1]]` in list order. Row t is the one-row
                step over those rows bit for bit (a list of `range(n)` is `attn_decode` over n rows)"""
                fn = nfns[k.dtype]
                hk, n_rows, d = k.shape
                rc = fn(
                    q.data_ptr(),
                    k.data_ptr(),
                    v.data_ptr(),
                    offs.data_ptr(),
                    idx.data_ptr(),
                    offs.shape[0] - 1,
                    n_rows,
                    q.shape[1],
                    hk,
                    d,
                    k.stride(0),
                    v.stride(0),
                    float(scale),
                    out.data_ptr(),
                    threads,
                )
                if rc != 0:
                    raise NativeError("btb_attn_nodes", rc)

            cls.attn_nodes = attn_nodes
        if checked:
            for name in _WANTS:
                bound = getattr(cls, name)
                if bound is not None:
                    setattr(cls, name, _checked(name, bound))
        return cls.gemv

    # the card's own kernels (a `_Cuda`), bound by `load_cuda` while an engine is on a card; None until then and
    # again once the last one closes (`unload_cuda`)
    cuda: Any = None
    cuda_reason: str | None = None  # why they are not bound, once `card_kernels` tried and failed

    @classmethod
    def load_cuda(cls, fatbin_path: str | os.PathLike[str]) -> _Cuda:
        if cls.cuda is None:
            cls.cuda = _Cuda(fatbin_path)
        return cls.cuda

    @classmethod
    def unload_cuda(cls) -> None:
        """the card's kernels let go - the module and its image out of the driver - by the last engine on a card
        to close; the next `card_kernels` loads them again"""
        k, cls.cuda = cls.cuda, None
        if k is not None:
            k.close()

    @classmethod
    def card_kernels(cls) -> _Cuda | None:
        """the card's kernels from the install's fatbin, bound on the first call; None with `cuda_reason` set
        when the fatbin is not here or the driver refuses it (no libcuda: an AMD card through ROCm), tried
        once a process"""
        if cls.cuda is not None:
            return cls.cuda
        if cls.cuda_reason is not None:
            return None
        p = kernels_path()
        if p is None:
            cls.cuda_reason = "btb_kernels.fatbin is not in this install (run `python build.py`)"
            return None
        try:
            return cls.load_cuda(p)
        except Exception as e:  # OSError: no driver library; RuntimeError: the driver refused the image
            cls.cuda_reason = f"{type(e).__name__}: {e}"
            return None

    @staticmethod
    def _cpu_gemm(x2: torch.Tensor, w: Any) -> torch.Tensor:
        """`x2` [b, cols] float32 times a bf16 weight [rows, cols] on MLX's CPU stream, one bf16 pass with f32
        accumulation: [b, rows] float32 at half the f32 path's time, no f32 copy of the weight."""
        m = mlxdev.mx()
        y = m.matmul(mlxdev.to_mx(x2.bfloat16()), w.T, stream=m.cpu).astype(m.float32)
        m.eval(y)
        return mlxdev.from_mx(y).clone()


class _AccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _StreamAttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", _AccessPolicyWindow), ("pad", ctypes.c_ubyte * 64)]


class _Cuda:
    """The card's kernels through the driver API: the fatbin loaded into torch's context, one function
    handle per kernel, and `launch` on a torch stream - inside a graph capture too, where the launches are
    recorded like any other node. `persist(base, nbytes)` sets the stream's persisting-L2 window (the part
    of the cache the weights' streaming loads never evict) and `persist_limit()` the card's ceiling for it."""

    _NAMES = {"windows": "nvcuda.dll", "linux": "libcuda.so.1"}
    KERNELS = (
        "btb_gemv_bf16_m1",
        "btb_gemv_bf16_m2",
        "btb_gemv_bf16_m4",
        "btb_gemv_bf16_m8",
        "btb_gemv_bf16_m16",
        "btb_gemv_bf16_m32",
        "btb_gemv_silu_bf16_m1",
        "btb_gemv_silu_bf16_m2",
        "btb_gemv_silu_bf16_m4",
        "btb_gemv_silu_bf16_m8",
        "btb_gemv_silu_bf16_m16",
        "btb_gemv_silu_bf16_m32",
        "btb_gemv_gelu_bf16_m1",
        "btb_gemv_gelu_bf16_m2",
        "btb_gemv_gelu_bf16_m4",
        "btb_gemv_gelu_bf16_m8",
        "btb_gemv_gelu_bf16_m16",
        "btb_gemv_gelu_bf16_m32",
        "btb_gemv_mma_bf16",
        "btb_gemv_mma8_bf16",
        "btb_gemv_mma_glu_silu",
        "btb_gemv_mma_glu_gelu",
        "btb_l2_warm",
        "btb_attn_split_d64",
        "btb_attn_split_d128",
        "btb_attn_split_d256",
        "btb_attn_split_gqa2_d64",
        "btb_attn_split_gqa2_d128",
        "btb_attn_split_gqa2_d256",
        "btb_attn_split_gqa4_d64",
        "btb_attn_split_gqa4_d128",
        "btb_attn_split_gqa4_d256",
        "btb_attn_split_gqa8_d64",
        "btb_attn_split_gqa8_d128",
        "btb_norm_rope_kv_d64",
        "btb_norm_rope_kv_d128",
        "btb_norm_rope_kv_d256",
        "btb_attn_rows_d64",
        "btb_attn_rows_d128",
        "btb_attn_rows_d256",
        "btb_norm_rope_kv_rows_d64",
        "btb_norm_rope_kv_rows_d128",
        "btb_norm_rope_kv_rows_d256",
        # the same walks and write through a cache's row map: a paged cache's rows wherever their pages lie
        "btb_attn_split_tbl_d64",
        "btb_attn_split_tbl_d128",
        "btb_attn_split_tbl_d256",
        "btb_attn_split_gqa2_tbl_d64",
        "btb_attn_split_gqa2_tbl_d128",
        "btb_attn_split_gqa2_tbl_d256",
        "btb_attn_split_gqa4_tbl_d64",
        "btb_attn_split_gqa4_tbl_d128",
        "btb_attn_split_gqa4_tbl_d256",
        "btb_attn_split_gqa8_tbl_d64",
        "btb_attn_split_gqa8_tbl_d128",
        "btb_norm_rope_kv_tbl_d64",
        "btb_norm_rope_kv_tbl_d128",
        "btb_norm_rope_kv_tbl_d256",
        "btb_add_rmsnorm",
        "btb_sandwich_add",
        "btb_silu_mul",
        "btb_gelu_mul",
        "btb_mx4_widen",
        "btb_delta_nodes",
        "btb_delta_nodes_bf16",
        "btb_conv_window",
        "btb_hc_rmsnorm",
        "btb_hc_act",
        "btb_hc_mix",
        "btb_moe_route",
        "btb_moe_combine",
        "btb_sigmoid_mul",
        "btb_gemv_sgate_bf16_m1",
        "btb_gemv_sgate_bf16_m2",
        "btb_gemv_sgate_bf16_m4",
        "btb_gemv_sgate_bf16_m8",
        "btb_gemv_sgate_bf16_m16",
        "btb_gemv_sgate_bf16_m32",
        "btb_norm_rope_part_d128",
        "btb_norm_rope_part_d256",
        "btb_qsa_pool_d128",
        "btb_qsa_pool_d256",
        "btb_qsa_select_d128",
        "btb_qsa_select_d256",
        "btb_qsa_attn_split_d128",
        "btb_qsa_attn_split_d256",
        "btb_gemv_lane16_f32_m1",
        "btb_gemv_lane16_f32_m2",
        "btb_gemv_lane16_f32_m4",
        "btb_gemv_lane16_f32_m8",
        "btb_gemv_lane16_f32_m16",
        "btb_gemv_lane16_f32_m32",
        "btb_gemv_lane16_mx4_f32_m1",
        "btb_gemv_lane16_mx4_f32_m2",
        "btb_gemv_lane16_mx4_f32_m4",
        "btb_gemv_lane16_mx4_f32_m8",
        "btb_gemv_lane16_mx4_f32_m16",
        "btb_gemv_lane16_mx4_f32_m32",
        "btb_ple_gate",
        "btb_ple_conv",
        "btb_publish",
        "btb_sample_max",
        "btb_sample_hist",
        "btb_sample_bin",
        "btb_sample_draw",
        "btb_sample_out",
        "btb_sample_keys",
        "btb_sample_verify",
    )
    # cuda.h
    _LIMIT_PERSISTING_L2 = 0x06
    _ATTR_L2_SIZE = 38
    _ATTR_MAX_PERSISTING_L2 = 108
    _ATTR_MAX_WINDOW = 109
    _STREAM_ATTR_WINDOW = 1
    _PROP_NORMAL, _PROP_STREAMING, _PROP_PERSISTING = 0, 1, 2

    def __init__(self, fatbin_path: str | os.PathLike[str]) -> None:

        name = self._NAMES.get({"win32": "windows", "linux": "linux"}.get(sys.platform, sys.platform))
        if name is None:
            raise RuntimeError(f"[cuda] no driver library for {sys.platform}")
        if not os.path.exists(fatbin_path):
            raise RuntimeError(f"[cuda] the card's kernels are missing: {fatbin_path} (run `python build.py`)")
        self.lib = ctypes.CDLL(name)
        self._check = self.lib.cuGetErrorString
        self._check.restype = ctypes.c_int
        self._check.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        # torch's primary context, made current on this thread: the module loads into the context the
        # engine's tensors and streams live in (torch creates it on the first allocation, not at init)
        torch.cuda.init()
        torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        self._call("cuInit", ctypes.c_uint(0))
        dev0 = ctypes.c_int(torch.cuda.current_device())
        ctx = ctypes.c_void_p()
        self._call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev0)
        self._retained = dev0
        self._call("cuCtxSetCurrent", ctx)
        # the image is the driver's once loaded (it copies what it keeps): read for the call, not held after it
        with open(fatbin_path, "rb") as fh:
            image = fh.read()
        self.module: ctypes.c_void_p | None = ctypes.c_void_p()
        self._call("cuModuleLoadData", ctypes.byref(self.module), ctypes.c_char_p(image))
        del image
        self.fn: dict[str, ctypes.c_void_p] = {}
        for k in self.KERNELS:
            f = ctypes.c_void_p()
            self._call("cuModuleGetFunction", ctypes.byref(f), self.module, ctypes.c_char_p(k.encode()))
            self.fn[k] = f
        dev = ctypes.c_int()
        self._call("cuCtxGetDevice", ctypes.byref(dev))
        self.device = int(dev.value)
        self.sms = self.attr(16)  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT

    def _call(self, name: str, *args: Any) -> None:
        f = getattr(self.lib, name)
        f.restype = ctypes.c_int
        rc = f(*args)
        if rc != 0:
            s = ctypes.c_char_p()
            self._check(rc, ctypes.byref(s))
            raise RuntimeError(f"[cuda] {name} failed: {rc} {(s.value or b'').decode()}")

    def attr(self, which: int) -> int:
        v = ctypes.c_int()
        self._call("cuDeviceGetAttribute", ctypes.byref(v), ctypes.c_int(which), ctypes.c_int(self.device))
        return int(v.value)

    def l2_bytes(self) -> int:
        """the card's L2, as the driver reports it"""
        return self.attr(self._ATTR_L2_SIZE)

    def persist_limit(self) -> tuple[int, int]:
        """(the most bytes the card sets aside as persisting L2, the widest window one stream may name)"""
        return self.attr(self._ATTR_MAX_PERSISTING_L2), self.attr(self._ATTR_MAX_WINDOW)

    def persist(self, base: int, nbytes: int, stream: torch.cuda.Stream | None = None, hit_ratio: float = 1.0) -> int:
        """Pin the range [base, base + nbytes) as persisting L2 for kernels launched on `stream` (the current
        one by default): the card sets aside that much of its L2, and lines of the range stay through the
        weights' streaming loads. Returns the bytes actually set aside (the range clipped to the card's
        limits). nbytes 0 clears the window."""
        lim, win = self.persist_limit()
        nb = min(int(nbytes), lim, win)
        self._call("cuCtxSetLimit", ctypes.c_int(self._LIMIT_PERSISTING_L2), ctypes.c_size_t(nb))
        v = _StreamAttrValue()
        v.accessPolicyWindow.base_ptr = ctypes.c_void_p(int(base) if nb else 0)
        v.accessPolicyWindow.num_bytes = nb
        v.accessPolicyWindow.hitRatio = float(hit_ratio) if nb else 0.0
        v.accessPolicyWindow.hitProp = self._PROP_PERSISTING if nb else self._PROP_NORMAL
        v.accessPolicyWindow.missProp = self._PROP_NORMAL
        st = (stream or torch.cuda.current_stream()).cuda_stream
        self._call("cuStreamSetAttribute", ctypes.c_void_p(st), ctypes.c_int(self._STREAM_ATTR_WINDOW), ctypes.byref(v))
        return nb

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        stream: torch.cuda.Stream | None = None,
        shared: int = 0,
    ) -> None:
        """Launch kernel `name` with `args` - each a ctypes value (c_void_p for a pointer, c_int, c_float) - on
        `stream` (the current torch stream by default)."""
        if self.module is None:
            raise RuntimeError(
                f"[cuda] {name}: the card's kernels were let go with the last engine on the card "
                "(`Native.unload_cuda`); `Native.card_kernels()` loads them again"
            )
        n = len(args)
        params = (ctypes.c_void_p * n)(*[ctypes.addressof(a) for a in args])
        st = (stream or torch.cuda.current_stream()).cuda_stream
        self._call(
            "cuLaunchKernel",
            self.fn[name],
            ctypes.c_uint(grid[0]),
            ctypes.c_uint(grid[1]),
            ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]),
            ctypes.c_uint(block[1]),
            ctypes.c_uint(block[2]),
            ctypes.c_uint(shared),
            ctypes.c_void_p(st),
            params,
            None,
        )

    def close(self) -> None:
        """The module unloaded - its code and the driver's copy of the image - and the primary context's retain
        given back; the function handles go with it. Safe to call twice"""
        if self.module is None:
            return
        module, self.module = self.module, None
        self.fn = {}
        self._call("cuModuleUnload", module)
        self._call("cuDevicePrimaryCtxRelease", self._retained)

    @staticmethod
    def ptr(t: torch.Tensor | None) -> ctypes.c_void_p:
        return ctypes.c_void_p(0 if t is None else t.data_ptr())

    def mx4_widen(
        self, blocks: torch.Tensor, scales: torch.Tensor, seats: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """MXFP4 experts at `seats` of the stacks `blocks` [S, .., 16] and `scales` [S, ..] (a depot's: uint8,
        contiguous, on the card) widened to bf16 [len(seats), groups * 32] in one pass (`btb_mx4_widen`), a
        stack's row of 32-value blocks at a time: `dequant_blocks`'s values, bit for bit, with no gathered copy of
        the bytes and no index into a table"""
        groups = int(scales[0].numel())
        if (
            blocks.dtype != torch.uint8
            or scales.dtype != torch.uint8
            or not (blocks.is_contiguous() and scales.is_contiguous())
            or int(blocks[0].numel()) != 16 * groups
            or blocks.device != scales.device
        ):
            raise ValueError(
                f"[cuda] mx4_widen: blocks {tuple(blocks.shape)} {blocks.dtype} and scales {tuple(scales.shape)} "
                f"{scales.dtype} are not a contiguous uint8 stack of 16-byte blocks and their scales"
            )
        seats = seats.to(blocks.device, torch.int32)
        n = int(seats.numel())
        if out is None:
            out = torch.empty(n, groups * 32, dtype=torch.bfloat16, device=blocks.device)
        elif out.dtype != torch.bfloat16 or not out.is_contiguous() or out.numel() != n * groups * 32:
            raise ValueError(f"[cuda] mx4_widen: out must be {n * groups * 32} contiguous bf16")
        if n:
            P = self.ptr
            self.launch(
                "btb_mx4_widen",
                ((groups + 255) // 256, n, 1),
                (256, 1, 1),
                [P(blocks), P(scales), P(seats), ctypes.c_longlong(groups), P(out)],
            )
        return out

    DELTA_THREADS = 128  # btb_delta_nodes's block (DELTA_THREADS in btb_kernels.cu)
    DELTA_MAX_K = 8

    def delta_nodes(
        self,
        mixed: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor | None,
        conv0: torch.Tensor,
        state: torch.Tensor,
        scratch: torch.Tensor | None,
        parents: torch.Tensor | None,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        gate: int,
        hk: int,
        hv: int,
        dk: int,
        dv: int,
        out: torch.Tensor,
    ) -> None:
        """The gated DeltaNet over T nodes on the card (`btb_delta_nodes`), every tensor float32, contiguous and
        on the card: node j's rows of `mixed` [T, C], `z` [T, hv * dv], `a` and `b` [T, hv]; the conv `conv_w`
        [C, K] (`conv_b` [C] or None) over its window from `conv0` [C, K]; its state stepped from its parent's
        (`parents` [T] int32, -1 at the root; None a chain). With `scratch` [T, hv, dk, dv] each node's state is
        written there and `state` [hv, dk, dv] only read (a verify pass); without it the chain steps `state` in
        place (a step, a commit). `out` [T, hv * dv] takes the gated-normed output (`gate` 0 silu, 1 sigmoid).
        The conv state is the caller's to keep."""
        T, C = (int(s) for s in mixed.shape)
        K = int(conv_w.shape[1])
        named = {"mixed": mixed, "z": z, "a": a, "b": b, "conv_w": conv_w, "conv0": conv0, "state": state}
        named |= {"a_log": a_log, "dt_bias": dt_bias, "norm_w": norm_w, "out": out}
        if conv_b is not None:
            named["conv_b"] = conv_b
        if scratch is not None:
            named["scratch"] = scratch
        bad = [k for k, t in named.items() if t.dtype != torch.float32 or not t.is_contiguous() or not t.is_cuda]
        if bad:
            raise ValueError(f"[cuda] delta_nodes: {', '.join(bad)} must be contiguous float32 on the card")
        if (
            not 0 < K <= self.DELTA_MAX_K
            or hv % hk
            or C != 2 * hk * dk + hv * dv
            or tuple(conv0.shape) != (C, K)
            or state.numel() != hv * dk * dv
            or tuple(z.shape) != (T, hv * dv)
            or tuple(a.shape) != (T, hv)
            or tuple(b.shape) != (T, hv)
            or tuple(out.shape) != (T, hv * dv)
            or (scratch is not None and scratch.numel() < T * hv * dk * dv)
            or gate not in (0, 1)
        ):
            raise ValueError(
                f"[cuda] delta_nodes: shapes do not agree (T {T}, C {C}, K {K}, heads {hk}/{hv}, dims {dk}/{dv}, "
                f"gate {gate})"
            )
        if parents is not None and (parents.dtype != torch.int32 or parents.numel() != T or not parents.is_cuda):
            raise ValueError(f"[cuda] delta_nodes: parents must be {T} int32 on the card")
        P = self.ptr
        shared = (2 * dk + 2 * dv + self.DELTA_THREADS) * 4
        self.launch(
            "btb_delta_nodes",
            (hv, 1, 1),
            (self.DELTA_THREADS, 1, 1),
            [
                P(mixed),
                P(z),
                P(a),
                P(b),
                P(conv_w),
                P(conv_b),
                P(conv0),
                P(state),
                P(scratch),
                P(parents),
                P(a_log),
                P(dt_bias),
                P(norm_w),
                ctypes.c_float(float(eps)),
                ctypes.c_int(int(gate)),
                ctypes.c_int(T),
                ctypes.c_int(C),
                ctypes.c_int(K),
                ctypes.c_int(int(hk)),
                ctypes.c_int(int(hv)),
                ctypes.c_int(int(dk)),
                ctypes.c_int(int(dv)),
                P(out),
            ],
            shared=shared,
        )

    # -- Qwen4's kernels: each launcher takes the caller's buffers (it allocates nothing, so a graph can capture it)
    # and refuses a call the kernel would misread with a ValueError before it launches

    @staticmethod
    def _want(call: str, dtype: torch.dtype, **named: torch.Tensor | None) -> None:
        """refuse the named tensors (None skipped) that are not contiguous `dtype` on the card"""
        bad = [
            k
            for k, t in named.items()
            if t is not None and (t.dtype != dtype or not t.is_contiguous() or not t.is_cuda)
        ]
        if bad:
            name = str(dtype).removeprefix("torch.")
            raise ValueError(f"[cuda] {call}: {', '.join(bad)} must be contiguous {name} on the card")

    @staticmethod
    def _want_rows(call: str, dtype: torch.dtype, **named: torch.Tensor | None) -> None:
        """`_want` for a card program's arena rows, which may be kept in RAM (`kv_host`): contiguous `dtype` on the
        card, or in pinned host memory the kernels read in place (btb/engine/hostmem.py)"""
        bad = [
            k
            for k, t in named.items()
            if t is not None
            and (
                t.dtype != dtype
                or not t.is_contiguous()
                or not (t.is_cuda or (t.device.type == "cpu" and t.is_pinned()))
            )
        ]
        if bad:
            name = str(dtype).removeprefix("torch.")
            raise ValueError(f"[cuda] {call}: {', '.join(bad)} must be contiguous {name} on the card or pinned in RAM")

    @staticmethod
    def _want_kv_rows(call: str, K: torch.Tensor, V: torch.Tensor | None) -> None:
        """a cache's K/V rows as [Hk, cap, D] views the kernels index by their strides (head g's row j at g * hs +
        j * rs): bf16 on the card or pinned in RAM, a row's D elements side by side, V laid out as K - head-major or
        position-major (the arenas'), contiguous or not"""
        bad = [
            name
            for name, t in (("K", K), ("V", V))
            if t is not None
            and (
                t.dtype != torch.bfloat16
                or t.dim() != 3
                or t.stride(-1) != 1
                or not (t.is_cuda or (t.device.type == "cpu" and t.is_pinned()))
                or (t is V and (V.shape != K.shape or V.stride() != K.stride()))
            )
        ]
        if bad:
            raise ValueError(
                f"[cuda] {call}: {', '.join(bad)} must be [Hk, cap, D] bf16 rows (a row's D side by side, V laid out "
                f"as K) on the card or pinned in RAM"
            )

    @staticmethod
    def _at(t: torch.Tensor, off: int) -> ctypes.c_void_p:
        """the address of element `off` of contiguous `t`"""
        return ctypes.c_void_p(t.data_ptr() + int(off) * t.element_size())

    def hc_rmsnorm(
        self,
        h: torch.Tensor,
        y: torch.Tensor | None,
        inj: torch.Tensor | None,
        w: torch.Tensor,
        eps: float,
        x: torch.Tensor,
        streams: int,
        centered: bool = True,
    ) -> None:
        """Qwen4's hyper-connection write and read (`btb_hc_rmsnorm`), every tensor bf16: the streams `h` [T, G * H]
        in place, h[t, g] += y[t] * inj[t, g] when `y` [T, H] and `inj` [T, G] (the block's output and its inject
        weights) are given, then `x` [T, G * H] = each stream's own rmsnorm times (1 + w) (`w` [G * H], the
        hc_norm's zero-centred weight; `centered` False: times w)."""
        self._want("hc_rmsnorm", torch.bfloat16, h=h, y=y, inj=inj, w=w, x=x)
        G = int(streams)
        if h.dim() != 2 or G < 1 or h.shape[1] % G:
            raise ValueError(f"[cuda] hc_rmsnorm: shapes do not agree (h {tuple(h.shape)}, {G} streams)")
        T, H = int(h.shape[0]), int(h.shape[1]) // G
        if (
            x.shape != h.shape
            or w.numel() != G * H
            or (y is None) != (inj is None)
            or (y is not None and (tuple(y.shape) != (T, H) or inj is None or tuple(inj.shape) != (T, G)))
        ):
            raise ValueError(f"[cuda] hc_rmsnorm: shapes do not agree (T {T}, {G} streams of {H})")
        if T:
            P, ci = self.ptr, ctypes.c_int
            self.launch(
                "btb_hc_rmsnorm",
                (T, G, 1),
                (256, 1, 1),
                [P(h), P(y), P(inj), P(w), ctypes.c_float(float(eps)), P(x), ci(H), ci(G), ci(int(bool(centered)))],
            )

    def hc_act(self, dn: torch.Tensor, act: torch.Tensor, streams: int) -> None:
        """the mixer's low-rank activation (`btb_hc_act`): `act` [T, R] = silu(dn[:, :R] / G), `dn` [T, ds] bf16 the
        down projection's rows (ds = R + G when the inject logits share its gemv)"""
        self._want("hc_act", torch.bfloat16, dn=dn, act=act)
        G = int(streams)
        if dn.dim() != 2 or act.dim() != 2 or dn.shape[0] != act.shape[0] or act.shape[1] > dn.shape[1] or G < 1:
            raise ValueError(f"[cuda] hc_act: shapes do not agree (dn {tuple(dn.shape)}, act {tuple(act.shape)})")
        T, R = (int(s) for s in act.shape)
        if T and R:
            ci = ctypes.c_int
            self.launch(
                "btb_hc_act",
                ((R + 255) // 256, T, 1),
                (256, 1, 1),
                [self.ptr(dn), ci(int(dn.shape[1])), self.ptr(act), ci(R), ci(G)],
            )

    def hc_mix(
        self,
        xn: torch.Tensor,
        up: torch.Tensor,
        dn: torch.Tensor | None,
        mixed: torch.Tensor,
        inj: torch.Tensor | None,
        streams: int,
        lowrank: int,
    ) -> None:
        """the stream mix (`btb_hc_mix`), every tensor bf16: `mixed` [T, H] = mean over the G streams of
        sigmoid(up) * xn (`xn`, `up` [T, G * H]); with `inj` [T, G], the block's inject weights 2 sigmoid(dn[:, R + g]
        / G) from `dn` [T, ds] (ds >= R + G, R = `lowrank`). `inj` and `dn` None: the final mixer."""
        self._want("hc_mix", torch.bfloat16, xn=xn, up=up, dn=dn, mixed=mixed, inj=inj)
        G, R = int(streams), int(lowrank)
        if mixed.dim() != 2 or G < 1:
            raise ValueError(f"[cuda] hc_mix: shapes do not agree (mixed {tuple(mixed.shape)}, {G} streams)")
        T, H = (int(s) for s in mixed.shape)
        if (
            tuple(xn.shape) != (T, G * H)
            or up.shape != xn.shape
            or (inj is not None and (dn is None or tuple(inj.shape) != (T, G)))
            or (dn is not None and (dn.dim() != 2 or dn.shape[0] != T or dn.shape[1] < R + G))
        ):
            raise ValueError(f"[cuda] hc_mix: shapes do not agree (T {T}, {G} streams of {H}, lowrank {R})")
        if T:
            P, ci = self.ptr, ctypes.c_int
            ds = int(dn.shape[1]) if dn is not None else 0
            self.launch(
                "btb_hc_mix",
                ((H + 255) // 256, T, 1),
                (256, 1, 1),
                [P(xn), P(up), P(dn), ci(ds), P(mixed), P(inj), ci(H), ci(G), ci(R)],
            )

    ROUTE_MAX_K = 32  # ROUTE_MAX_K in btb_kernels.cu

    def moe_route(
        self,
        logits: torch.Tensor,
        experts: int,
        k: int,
        idx: torch.Tensor,
        w: torch.Tensor,
        hidx: torch.Tensor | None = None,
        hw: torch.Tensor | None = None,
        cnt: torch.Tensor | None = None,
        seq: torch.Tensor | None = None,
        hseq: torch.Tensor | None = None,
    ) -> None:
        """The router (`btb_moe_route`): `logits` [T, ls] bf16 (the first `experts` columns the experts', ls = E + 1
        when the shared expert's gate logit rides at column E) to `idx` [T, k] int32 and `w` [T, k] bf16 on the card:
        fp32 softmax, top k (probability desc, index asc), renormalised. With the host outputs - `hidx` [T, k] int32
        and `hw` [T, k] bf16 in pinned host memory, `cnt` [1] int32 on the card (zero; the kernel leaves it zero),
        `seq` [1] int32 on the card and `hseq` [1] int32 pinned - the rows land on the host too, then *seq in
        *hseq once every row is there (all five or none)."""
        self._want("moe_route", torch.bfloat16, logits=logits, w=w)
        self._want("moe_route", torch.int32, idx=idx, cnt=cnt, seq=seq)
        E, k = int(experts), int(k)
        if logits.dim() != 2:
            raise ValueError(f"[cuda] moe_route: shapes do not agree (logits {tuple(logits.shape)})")
        T = int(logits.shape[0])
        if (
            not 0 < k <= min(E, self.ROUTE_MAX_K)
            or E > int(logits.shape[1])
            or E * 4 > 48 * 1024
            or tuple(idx.shape) != (T, k)
            or tuple(w.shape) != (T, k)
        ):
            raise ValueError(f"[cuda] moe_route: shapes do not agree (T {T}, {E} experts, top {k})")
        host = (hidx, hw, cnt, seq, hseq)
        if any(t is not None for t in host):
            if any(t is None for t in host):
                raise ValueError("[cuda] moe_route: the host outputs are hidx, hw, cnt, seq and hseq together")
            assert hidx is not None and hw is not None and cnt is not None and seq is not None and hseq is not None
            pinned = {
                "hidx": (hidx, torch.int32, (T, k)),
                "hw": (hw, torch.bfloat16, (T, k)),
                "hseq": (hseq, torch.int32, (1,)),
            }
            bad = [
                n
                for n, (t, dt, shape) in pinned.items()
                if t.dtype != dt or tuple(t.shape) != shape or t.is_cuda or not t.is_pinned() or not t.is_contiguous()
            ]
            if bad or cnt.numel() != 1 or seq.numel() != 1:
                raise ValueError(
                    f"[cuda] moe_route: {', '.join(bad) or 'cnt, seq'} must be pinned host rows of the pass"
                )
        if T:
            P, ci = self.ptr, ctypes.c_int
            self.launch(
                "btb_moe_route",
                (T, 1, 1),
                (256, 1, 1),
                [
                    P(logits),
                    ci(int(logits.shape[1])),
                    ci(E),
                    ci(k),
                    P(idx),
                    P(w),
                    P(hidx),
                    P(hw),
                    P(cnt),
                    P(seq),
                    P(hseq),
                ],
                shared=E * 4,
            )

    def moe_combine(
        self, yr: torch.Tensor, ys: torch.Tensor, logits: torch.Tensor, gate_col: int, y: torch.Tensor
    ) -> None:
        """the MoE's output (`btb_moe_combine`), every tensor bf16: `y` [T, H] = yr + sigmoid(gate) * ys, the gate
        logit of row t at logits[t, gate_col] (the router's merged row). `y` may be `yr`."""
        self._want("moe_combine", torch.bfloat16, yr=yr, ys=ys, logits=logits, y=y)
        if yr.dim() != 2 or ys.shape != yr.shape or y.shape != yr.shape or logits.dim() != 2:
            raise ValueError(f"[cuda] moe_combine: shapes do not agree (yr {tuple(yr.shape)})")
        T, H = (int(s) for s in yr.shape)
        if logits.shape[0] != T or not 0 <= int(gate_col) < int(logits.shape[1]):
            raise ValueError(f"[cuda] moe_combine: shapes do not agree (logits {tuple(logits.shape)}, col {gate_col})")
        if T and H:
            P, ci = self.ptr, ctypes.c_int
            self.launch(
                "btb_moe_combine",
                ((H + 255) // 256, T, 1),
                (256, 1, 1),
                [P(yr), P(ys), self._at(logits, int(gate_col)), ci(int(logits.shape[1])), P(y), ci(H)],
            )

    def _gate_args(
        self, call: str, att: torch.Tensor, qkv: torch.Tensor, head_dim: int, gate_off: int, gate_stride: int
    ) -> None:
        D, off, hs = int(head_dim), int(gate_off), int(gate_stride)
        if att.dim() != 2 or qkv.dim() != 2 or qkv.shape[0] < att.shape[0] or D < 8:
            raise ValueError(f"[cuda] {call}: shapes do not agree (att {tuple(att.shape)}, qkv {tuple(qkv.shape)})")
        C, width = int(att.shape[1]), int(qkv.shape[1])
        if C % D or D % 8 or off % 8 or hs % 8 or width % 8 or off < 0 or off + (C // D - 1) * hs + D > width:
            raise ValueError(
                f"[cuda] {call}: shapes do not agree (C {C}, head_dim {D}, the gate at {off} + head * {hs} of {width})"
            )

    def sigmoid_mul(
        self, att: torch.Tensor, qkv: torch.Tensor, out: torch.Tensor, head_dim: int, gate_off: int, gate_stride: int
    ) -> None:
        """the attention's output gate alone (`btb_sigmoid_mul`), every tensor bf16: `out` [T, C] = att *
        sigmoid(gate), `att` [T, C] (C = heads * head_dim), gate element (t, c) at qkv[t, gate_off + (c // head_dim)
        * gate_stride + c % head_dim] (Qwen4: gate_off head_dim, gate_stride 2 head_dim)"""
        self._want("sigmoid_mul", torch.bfloat16, att=att, qkv=qkv, out=out)
        self._gate_args("sigmoid_mul", att, qkv, head_dim, gate_off, gate_stride)
        if out.shape != att.shape:
            raise ValueError(f"[cuda] sigmoid_mul: shapes do not agree (out {tuple(out.shape)})")
        T, C = (int(s) for s in att.shape)
        if T:
            ci = ctypes.c_int
            self.launch(
                "btb_sigmoid_mul",
                ((C + 255) // 256, T, 1),
                (256, 1, 1),
                [
                    self.ptr(att),
                    self._at(qkv, gate_off),
                    ci(int(qkv.shape[1])),
                    ci(int(head_dim)),
                    ci(int(gate_stride)),
                    self.ptr(out),
                    ci(C),
                ],
            )

    GEMV_ROWS = (1, 2, 4, 8, 16, 32)  # the M of btb_gemv_*_bf16_m{M}

    def gemv_sgate(
        self,
        w: torch.Tensor,
        att: torch.Tensor,
        qkv: torch.Tensor,
        y: torch.Tensor,
        head_dim: int,
        gate_off: int,
        gate_stride: int,
    ) -> None:
        """o_proj over the gated attention (`btb_gemv_sgate_bf16_m{M}`), every tensor bf16: `y` [M, R] = `w` [R, C]
        times att * sigmoid(gate) (`att` [M, C], the gate as `sigmoid_mul` reads it), M one of GEMV_ROWS - the pass's
        rows padded to it, as the gemvs are"""
        self._want("gemv_sgate", torch.bfloat16, w=w, att=att, qkv=qkv, y=y)
        self._gate_args("gemv_sgate", att, qkv, head_dim, gate_off, gate_stride)
        M, C = (int(s) for s in att.shape)
        if (
            M not in self.GEMV_ROWS
            or w.dim() != 2
            or int(w.shape[1]) != C
            or C % 8
            or tuple(y.shape) != (M, int(w.shape[0]))
        ):
            raise ValueError(
                f"[cuda] gemv_sgate: shapes do not agree (w {tuple(w.shape)}, att {tuple(att.shape)}, M {M})"
            )
        R = int(w.shape[0])
        ci = ctypes.c_int
        self.launch(
            f"btb_gemv_sgate_bf16_m{M}",
            ((R + 3) // 4, 1, 1),
            (128, 1, 1),
            [
                self.ptr(w),
                self.ptr(att),
                self._at(qkv, gate_off),
                ci(int(qkv.shape[1])),
                ci(int(head_dim)),
                ci(int(gate_stride)),
                self.ptr(y),
                ci(R),
                ci(C),
            ],
        )

    def norm_rope_part(
        self,
        qkv: torch.Tensor,
        qo: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        n0: torch.Tensor,
        depth: torch.Tensor,
        heads: int,
        kv_heads: int,
        q_stride: int,
        k_off: int,
        v_off: int,
        wq: torch.Tensor | None,
        wk: torch.Tensor | None,
        eps: float,
        centered: bool = True,
        raw_key: bool = False,
        q_col: int = 0,
    ) -> None:
        """Qwen4's q/k norm, partial rope and cache write (`btb_norm_rope_part_d{128,256}`), bf16 but for `n0` [1]
        and `depth` [T] int32 on the card. `qkv` [T, width] the merged projection's rows: q head h at column q_col +
        h * q_stride, k head g at k_off + g * D, v head g at v_off + g * D (k_off and v_off at or past q_col: the
        kernel reads a row from q_col on). The rotary tables `cos`/`sin` [positions, ROT] rope the first ROT dims (ROT
        their width); row t sits at position n0 + depth[t] and cache slot n0 + t of `K`/`V` [kv_heads, cap, D]; the q
        heads go to `qo` [T, heads, D]. `wq`/`wk` [D] the zero-centred norms (None: none). `raw_key`: the QSA indexer
        - its key heads copied raw into `K` (the raw-key arena), `V` None."""
        self._want("norm_rope_part", torch.bfloat16, qkv=qkv, qo=qo, wq=wq, wk=wk)
        self._want_rows("norm_rope_part", torch.bfloat16, cos=cos, sin=sin)
        self._want_kv_rows("norm_rope_part", K, V)
        self._want("norm_rope_part", torch.int32, n0=n0, depth=depth)
        Hq, Hk, qs, ko, vo, q0 = int(heads), int(kv_heads), int(q_stride), int(k_off), int(v_off), int(q_col)
        if K.dim() != 3 or qkv.dim() != 2 or cos.dim() != 2:
            raise ValueError(f"[cuda] norm_rope_part: shapes do not agree (K {tuple(K.shape)}, qkv {tuple(qkv.shape)})")
        D = int(K.shape[2])
        T, width = (int(s) for s in qkv.shape)
        rot = int(cos.shape[1])
        E = D // 32
        half = rot // (2 * E) if E else 0
        if (
            D not in (128, 256)
            or int(K.shape[0]) != Hk
            or (V is not None and (raw_key or V.shape != K.shape))
            or qo.numel() != T * Hq * D
            or sin.shape != cos.shape
            or not 0 < rot <= D
            or rot % (2 * E)
            or half & (half - 1)
            or n0.numel() != 1
            or depth.numel() != T
            or (wq is not None and wq.numel() != D)
            or (wk is not None and wk.numel() != D)
            or Hq < 1
            or Hk < 1
            or q0 < 0
            or ko < q0
            or (V is not None and vo < q0)
            or q0 + (Hq - 1) * qs + D > width
            or ko + Hk * D > width
            or (V is not None and vo + Hk * D > width)
        ):
            raise ValueError(
                f"[cuda] norm_rope_part: shapes do not agree (T {T}, width {width}, heads {Hq}/{Hk}, head_dim {D}, "
                f"rot {rot})"
            )
        if T:
            P, ci = self.ptr, ctypes.c_int
            self.launch(
                f"btb_norm_rope_part_d{D}",
                (Hq + Hk + (Hk if V is not None else 0), T, 1),
                (32, 1, 1),
                [
                    self._at(qkv, q0),
                    ci(width),
                    ci(qs),
                    ci(ko - q0),
                    ci(max(0, vo - q0)),
                    P(wq),
                    P(wk),
                    ctypes.c_float(float(eps)),
                    P(cos),
                    P(sin),
                    ci(rot),
                    ci(rot),
                    P(n0),
                    P(depth),
                    P(K),
                    P(V),
                    P(qo),
                    ci(Hq),
                    ci(Hk),
                    ci(int(K.stride(0))),  # head g's slot j at g * hs + j * rs: either layout of [Hk, cap, D]
                    ci(int(K.stride(1))),
                    ci(int(bool(centered))),
                    ci(int(bool(raw_key))),
                ],
            )

    # -- Qwen4's sparse attention (QSA): the indexer's picks and the attention over them
    QSA_THREADS = 512  # btb_qsa_select's block (QSA_THREADS in btb_kernels.cu)
    QSA_MAX_FLIGHT = 33  # blocks a node pools in flight (QSA_MAX_FLIGHT): 32 // ratio + 1 at T <= 32
    WALK_MAX_T = 32  # the rows an ancestor walk follows (the kernels' anc[32])
    ATTN_SPLIT = 1024  # keys a split of the attention walks (ATTN_SPLIT in btb_kernels.cu)
    _SMEM = 48 * 1024  # the shared memory a block gets without opting in

    @classmethod
    def qsa_splits(cls, ratio: int, k_top: int) -> int:
        """the splits of `btb_qsa_attn_split`'s grid: a node's list holds at most k_top * ratio + ratio - 1 keys"""
        return ((int(k_top) + 1) * int(ratio) + cls.ATTN_SPLIT - 1) // cls.ATTN_SPLIT

    @staticmethod
    def _qsa_keys(
        call: str,
        raw: torch.Tensor,
        pk: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kw: torch.Tensor,
        ratio: int,
    ) -> tuple[int, int, int]:
        """the indexer's key arrays checked against each other: (di, the raw arena's capacity, the rotary dims)"""
        if raw.dim() == 3 and raw.shape[0] == 1:
            raw = raw[0]  # the arena as btb_norm_rope_part writes it, [1, cap, di]
        if raw.dim() != 2 or pk.dim() != 2 or cos.dim() != 2:
            raise ValueError(f"[cuda] {call}: shapes do not agree (raw {tuple(raw.shape)}, pk {tuple(pk.shape)})")
        di, cap, rot = int(pk.shape[1]), int(raw.shape[0]), int(cos.shape[1])
        E = di // 32
        half = rot // (2 * E) if E else 0
        if (
            di not in (128, 256)
            or int(raw.shape[1]) != di
            or sin.shape != cos.shape
            or kw.numel() != di
            or not 0 < rot <= di
            or rot % (2 * E)
            or half & (half - 1)
            or int(ratio) < 1
            or int(pk.shape[0]) < cap // int(ratio)
        ):
            raise ValueError(
                f"[cuda] {call}: shapes do not agree (raw {tuple(raw.shape)}, pk {tuple(pk.shape)}, rot {rot}, "
                f"ratio {ratio})"
            )
        return di, cap, rot

    def qsa_pool(
        self,
        raw: torch.Tensor,
        pk: torch.Tensor,
        pk_len: torch.Tensor,
        n0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kw: torch.Tensor,
        eps: float,
        ratio: int,
        max_new: int,
    ) -> None:
        """The indexer's pooled keys caught up (`btb_qsa_pool_d{128,256}`), bf16 but for `pk_len` [2] and `n0` [1]
        int32 on the card: blocks pk_len[0] .. n0 // ratio of the raw-key arena `raw` [cap, di] (or [1, cap, di]) -
        each the mean of its `ratio` raw keys, k_layernorm'd by `kw` [di] (zero-centred), roped by `cos`/`sin`
        [positions, rot] at its first position - into `pk` [NBcap, di] (NBcap >= cap // ratio), at most `max_new` of
        them a launch; pk_len[0] then n0 // ratio (pk_len[1] the launch's arrival count, zero between launches)."""
        self._want("qsa_pool", torch.bfloat16, pk=pk, kw=kw)
        self._want_rows("qsa_pool", torch.bfloat16, raw=raw, cos=cos, sin=sin)
        self._want("qsa_pool", torch.int32, pk_len=pk_len, n0=n0)
        di, _cap, rot = self._qsa_keys("qsa_pool", raw, pk, cos, sin, kw, ratio)
        if pk_len.numel() != 2 or n0.numel() != 1 or int(max_new) < 1:
            raise ValueError(f"[cuda] qsa_pool: shapes do not agree (pk_len {pk_len.numel()}, max_new {max_new})")
        P, ci = self.ptr, ctypes.c_int
        self.launch(
            f"btb_qsa_pool_d{di}",
            ((int(max_new) + 7) // 8, 1, 1),
            (256, 1, 1),
            [
                P(raw),
                P(pk),
                P(pk_len),
                P(n0),
                P(cos),
                P(sin),
                ci(rot),
                ci(rot),
                P(kw),
                ctypes.c_float(float(eps)),
                ci(int(ratio)),
                ci(int(max_new)),
            ],
        )

    def qsa_select(
        self,
        qi: torch.Tensor,
        pk: torch.Tensor,
        raw: torch.Tensor,
        n0: torch.Tensor,
        par: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kw: torch.Tensor,
        eps: float,
        ratio: int,
        k_top: int,
        scores: torch.Tensor,
        sel: torch.Tensor,
        nsel: torch.Tensor,
    ) -> None:
        """The indexer's picks for a pass's T nodes (`btb_qsa_select_d{128,256}`): `qi` [T, Hi, di] bf16 the index
        queries (normed, roped), `pk`/`raw`/`cos`/`sin`/`kw` as `qsa_pool` (pk caught up to n0 // ratio), `n0` [1] and
        `par` [T] int32 the tree (par -2: a padding row). `sel` [T, k_top] int32 takes each node's picked blocks in
        ascending order and `nsel` [T] int32 their count (k_top, or every complete block it sees when they number
        k_top or fewer; 0 for padding); `scores` [T, NBcap] float32 is the kernel's scratch."""
        self._want("qsa_select", torch.bfloat16, qi=qi, pk=pk, kw=kw)
        self._want_rows("qsa_select", torch.bfloat16, raw=raw, cos=cos, sin=sin)
        self._want("qsa_select", torch.int32, n0=n0, par=par, sel=sel, nsel=nsel)
        self._want("qsa_select", torch.float32, scores=scores)
        di, _cap, rot = self._qsa_keys("qsa_select", raw, pk, cos, sin, kw, ratio)
        if qi.dim() != 3:
            raise ValueError(f"[cuda] qsa_select: shapes do not agree (qi {tuple(qi.shape)})")
        T, Hi = int(qi.shape[0]), int(qi.shape[1])
        NB, k = int(pk.shape[0]), int(k_top)
        if (
            int(qi.shape[2]) != di
            or not 0 < T <= self.WALK_MAX_T
            or Hi < 1
            or k < 1
            or n0.numel() != 1
            or par.numel() != T
            or tuple(sel.shape) != (T, k)
            or nsel.numel() != T
            or tuple(scores.shape) != (T, NB)
            or (Hi + self.QSA_MAX_FLIGHT) * di * 2 + 2048 > self._SMEM
        ):
            raise ValueError(
                f"[cuda] qsa_select: shapes do not agree (T {T}, {Hi} index heads of {di}, k_top {k}, NBcap {NB})"
            )
        P, ci = self.ptr, ctypes.c_int
        self.launch(
            f"btb_qsa_select_d{di}",
            (T, 1, 1),
            (self.QSA_THREADS, 1, 1),
            [
                P(qi),
                P(pk),
                P(raw),
                P(n0),
                P(par),
                P(cos),
                P(sin),
                ci(rot),
                ci(rot),
                P(kw),
                ctypes.c_float(float(eps)),
                ci(T),
                ci(Hi),
                ci(int(ratio)),
                ci(k),
                ci(NB),
                P(scores),
                P(sel),
                P(nsel),
            ],
            shared=Hi * di * 2,
        )

    def qsa_attn_split(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        out: torch.Tensor,
        n0: torch.Tensor,
        par: torch.Tensor,
        sel: torch.Tensor,
        nsel: torch.Tensor,
        ratio: int,
        k_top: int,
        scale: float,
        part_m: torch.Tensor,
        part_l: torch.Tensor,
        part_acc: torch.Tensor,
        cnt: torch.Tensor,
    ) -> None:
        """The attention of a pass's T nodes over their QSA picks (`btb_qsa_attn_split_d{128,256}`): `q` [T, Hq, D]
        bf16 over `K`/`V` [Hk, cap, D] bf16, node t's keys its picked blocks (`sel`/`nsel` from `qsa_select`, blocks
        of `ratio` positions) then its partial tail, into `out` [T, Hq, D] (padding rows, par -2, left as they are).
        `n0` [1] and `par` [T] int32 the tree as `btb_attn_split` walks it. The splits' states: `part_m`/`part_l`
        [S * T * Hq] and `part_acc` [S * T * Hq * D] float32, `cnt` [T * Hq] int32 (zero; the kernel leaves it zero),
        S = `qsa_splits(ratio, k_top)`."""
        self._want("qsa_attn_split", torch.bfloat16, q=q, out=out)
        self._want_kv_rows("qsa_attn_split", K, V)
        self._want("qsa_attn_split", torch.int32, n0=n0, par=par, sel=sel, nsel=nsel, cnt=cnt)
        self._want("qsa_attn_split", torch.float32, part_m=part_m, part_l=part_l, part_acc=part_acc)
        if q.dim() != 3 or K.dim() != 3:
            raise ValueError(f"[cuda] qsa_attn_split: shapes do not agree (q {tuple(q.shape)}, K {tuple(K.shape)})")
        T, Hq, D = (int(s) for s in q.shape)
        Hk = int(K.shape[0])
        k = int(k_top)
        S = self.qsa_splits(ratio, k)
        if (
            D not in (128, 256)
            or int(K.shape[2]) != D
            or V.shape != K.shape
            or out.shape != q.shape
            or not 0 < T <= self.WALK_MAX_T
            or Hk < 1
            or Hq % Hk
            or int(ratio) < 1
            or k < 1
            or n0.numel() != 1
            or par.numel() != T
            or tuple(sel.shape) != (T, k)
            or nsel.numel() != T
            or part_m.numel() < S * T * Hq
            or part_l.numel() < S * T * Hq
            or part_acc.numel() < S * T * Hq * D
            or cnt.numel() < T * Hq
        ):
            raise ValueError(
                f"[cuda] qsa_attn_split: shapes do not agree (T {T}, heads {Hq}/{Hk}, head_dim {D}, k_top {k}, "
                f"ratio {ratio}, {S} splits)"
            )
        P, ci = self.ptr, ctypes.c_int
        self.launch(
            f"btb_qsa_attn_split_d{D}",
            (Hq, T, S),
            (256, 1, 1),
            [
                P(q),
                P(K),
                P(V),
                P(out),
                P(n0),
                P(par),
                P(sel),
                P(nsel),
                ci(int(ratio)),
                ci(k),
                ci(T),
                ci(Hq),
                ci(Hk),
                ci(int(K.stride(0))),  # head g's row j at g * hs + j * rs: either layout of [Hk, cap, D]
                ci(int(K.stride(1))),
                ctypes.c_float(float(scale)),
                P(part_m),
                P(part_l),
                P(part_acc),
                P(cnt),
                ci(S),
            ],
        )

    def gemv_lane16(self, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
        """The host's gemv on the card (`btb_gemv_lane16_f32_m{M}`): `y` [M, R] float32 = `w` [R, C] bf16 times `x`
        [M, C] float32, bit for bit `Native.gemv`'s (its 16 lane sums and their pairwise fold), M one of GEMV_ROWS"""
        self._want("gemv_lane16", torch.bfloat16, w=w)
        self._want("gemv_lane16", torch.float32, x=x, y=y)
        if w.dim() != 2 or x.dim() != 2:
            raise ValueError(f"[cuda] gemv_lane16: shapes do not agree (w {tuple(w.shape)}, x {tuple(x.shape)})")
        R, C = (int(s) for s in w.shape)
        M = int(x.shape[0])
        if M not in self.GEMV_ROWS or int(x.shape[1]) != C or tuple(y.shape) != (M, R):
            raise ValueError(
                f"[cuda] gemv_lane16: shapes do not agree (w {tuple(w.shape)}, x {tuple(x.shape)}, y {tuple(y.shape)})"
            )
        if R:
            ci = ctypes.c_int
            self.launch(
                f"btb_gemv_lane16_f32_m{M}",
                ((R + 7) // 8, 1, 1),
                (128, 1, 1),
                [self.ptr(w), self.ptr(x), self.ptr(y), ci(R), ci(C)],
            )

    def gemv_lane16_mx4(self, w: Any, x: torch.Tensor, y: torch.Tensor) -> None:
        """The host's MXFP4 gemv on the card (`btb_gemv_lane16_mx4_f32_m{M}`): `y` [M, R] float32 = `w` (an MxWeight
        in the checkpoint's layout, [R, C], its blocks and scales on the card) times `x` [M, C] float32, bit for bit
        `Native.gemv_mx4`'s (each weight widened as the host widens it, then gemv_lane16's sums), M one of GEMV_ROWS"""
        if getattr(w, "ggml", True) or w.scales is None:
            raise ValueError("[cuda] gemv_lane16_mx4: the checkpoint's MXFP4 layout only (blocks and scales apart)")
        self._want("gemv_lane16_mx4", torch.uint8, blocks=w.blocks, scales=w.scales)
        self._want("gemv_lane16_mx4", torch.float32, x=x, y=y)
        R, C = (int(s) for s in w.shape)
        M = int(x.shape[0])
        if C % 32 or x.dim() != 2 or M not in self.GEMV_ROWS or int(x.shape[1]) != C or tuple(y.shape) != (M, R):
            raise ValueError(
                f"[cuda] gemv_lane16_mx4: shapes do not agree (w {w.shape}, x {tuple(x.shape)}, y {tuple(y.shape)})"
            )
        if w.blocks.numel() != R * C // 2 or w.scales.numel() != R * C // 32:
            raise ValueError(f"[cuda] gemv_lane16_mx4: w {w.shape} needs {R * C // 2} + {R * C // 32} bytes")
        if R:
            ci = ctypes.c_int
            self.launch(
                f"btb_gemv_lane16_mx4_f32_m{M}",
                ((R + 7) // 8, 1, 1),
                (128, 1, 1),
                [self.ptr(w.blocks), self.ptr(w.scales), self.ptr(x), self.ptr(y), ci(R), ci(C)],
            )

    PLE_WALK = 32  # the ancestors btb_ple_conv's walk follows (its path[32])

    def ple_nodes(
        self,
        kv: torch.Tensor,
        xq: torch.Tensor,
        wk: torch.Tensor,
        wc: torch.Tensor,
        conv_w: torch.Tensor,
        pre: torch.Tensor,
        par: torch.Tensor | None,
        h: torch.Tensor,
        gated: torch.Tensor,
        normed: torch.Tensor,
        streams: int,
        dilation: int,
        eps: float,
        update: bool = False,
    ) -> None:
        """Qwen4's per-layer n-gram embedding over a pass's T nodes (`btb_ple_gate`, then `btb_ple_conv`), bf16 but
        for `par` [T] int32 on the card (-1 a root, -2 padding; None a chain): `kv` [T, (G + 1) H] the merged
        [key_proj | value_proj] rows of the nodes' embeddings, `xq` [T, G H] the streams through norm_query, `wk`/`wc`
        [G H] the zero-centred norm_key and norm_conv weights, `conv_w` [G H, K] the dilated depthwise conv's taps,
        `pre` [G H, (K - 1) dilation] its kept normed inputs, oldest first. `gated`/`normed` [T, G H] take the gated
        rows and their norm (a commit keeps its path's normed rows); `h` [T, G H], the streams, takes += the
        embedding's output. `update` (a one-row step) shifts the row's normed input into `pre`."""
        self._want("ple_nodes", torch.bfloat16, kv=kv, xq=xq, wk=wk, wc=wc, conv_w=conv_w, pre=pre, h=h)
        self._want("ple_nodes", torch.bfloat16, gated=gated, normed=normed)
        self._want("ple_nodes", torch.int32, par=par)
        G, dil = int(streams), int(dilation)
        if h.dim() != 2 or kv.dim() != 2 or conv_w.dim() != 2 or G < 1 or h.shape[1] % G:
            raise ValueError(f"[cuda] ple_nodes: shapes do not agree (h {tuple(h.shape)}, {G} streams)")
        T, C = (int(s) for s in h.shape)
        H, K = C // G, int(conv_w.shape[1])
        Lp = (K - 1) * dil
        if (
            H < 1
            or dil < 1
            or not 0 < K
            or Lp >= self.PLE_WALK
            or tuple(kv.shape) != (T, C + H)
            or xq.shape != h.shape
            or gated.shape != h.shape
            or normed.shape != h.shape
            or wk.numel() != C
            or wc.numel() != C
            or int(conv_w.shape[0]) != C
            or pre.numel() != C * Lp
            or (par is not None and par.numel() != T)
            or not 0 < T <= self.WALK_MAX_T
            or (update and T != 1)
        ):
            raise ValueError(
                f"[cuda] ple_nodes: shapes do not agree (T {T}, {G} streams of {H}, K {K}, dilation {dil}, "
                f"kv {tuple(kv.shape)}, pre {tuple(pre.shape)}{', update over several rows' if update and T != 1 else ''})"
            )
        inv_h = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(math.sqrt(H), dtype=torch.float32))
        P, ci, cf = self.ptr, ctypes.c_int, ctypes.c_float
        self.launch(
            "btb_ple_gate",
            (T, G, 1),
            (256, 1, 1),
            [
                P(kv),
                ci(C + H),
                P(xq),
                P(wk),
                P(wc),
                cf(float(eps)),
                cf(inv_h),
                P(par),
                P(gated),
                P(normed),
                ci(H),
                ci(G),
            ],
        )
        self.launch(
            "btb_ple_conv",
            ((C + 255) // 256, T, 1),
            (256, 1, 1),
            [P(gated), P(normed), P(conv_w), P(pre), P(par), P(h), ci(C), ci(K), ci(dil), ci(int(bool(update)))],
        )

    def delta_nodes_bf16(
        self,
        proj: torch.Tensor,
        offsets: tuple[int, int, int, int],
        rows: torch.Tensor | None,
        conv_w: torch.Tensor,
        conv_b: torch.Tensor | None,
        conv0: torch.Tensor,
        state: torch.Tensor,
        scratch: torch.Tensor | None,
        slots: torch.Tensor | None,
        parents: torch.Tensor | None,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_w: torch.Tensor,
        eps: float,
        gate: int,
        hk: int,
        hv: int,
        dk: int,
        dv: int,
        out: torch.Tensor | None,
    ) -> None:
        """`delta_nodes` over the card program's bf16 buffers (`btb_delta_nodes_bf16`): each node's q|k|v, z, b and
        a read in place from `proj` [rows, ps] bf16 at column `offsets` (q, z, b, a), node j at proj row rows[j]
        (`rows` [T] int32, -1 padding at the end; None: row j of every row), its output rounded into `out` [T, hv *
        dv] bf16 (None: states only, a commit). The float32 operands as `delta_nodes`; with `scratch` [slots, hv,
        dk, dv] a node's state goes to scratch[slots[j]] (`slots` [T] int32, a node in its parent's slot when it is
        the parent's last child; None: slot j), without it the chain steps `state` in place."""
        self._want("delta_nodes_bf16", torch.bfloat16, proj=proj, out=out)
        self._want(
            "delta_nodes_bf16",
            torch.float32,
            conv_w=conv_w,
            conv_b=conv_b,
            conv0=conv0,
            state=state,
            scratch=scratch,
            a_log=a_log,
            dt_bias=dt_bias,
            norm_w=norm_w,
        )
        self._want("delta_nodes_bf16", torch.int32, rows=rows, slots=slots, parents=parents)
        if proj.dim() != 2 or conv_w.dim() != 2:
            raise ValueError(f"[cuda] delta_nodes_bf16: shapes do not agree (proj {tuple(proj.shape)})")
        T = int(rows.numel()) if rows is not None else int(proj.shape[0])
        ps = int(proj.shape[1])
        oq, oz, ob, oa = (int(o) for o in offsets)
        C, K = (int(s) for s in conv_w.shape)
        if (
            not 0 < K <= self.DELTA_MAX_K
            or hk < 1
            or hv % hk
            or C != 2 * hk * dk + hv * dv
            or tuple(conv0.shape) != (C, K)
            or (conv_b is not None and conv_b.numel() != C)
            or state.numel() != hv * dk * dv
            or min(oq, oz, ob, oa) < 0
            or oq + C > ps
            or oz + hv * dv > ps
            or ob + hv > ps
            or oa + hv > ps
            or (out is not None and tuple(out.shape) != (T, hv * dv))
            or (scratch is not None and (scratch.dim() != 4 or tuple(scratch.shape[1:]) != (hv, dk, dv)))
            or a_log.numel() != hv
            or dt_bias.numel() != hv
            or norm_w.numel() != dv
            or gate not in (0, 1)
        ):
            raise ValueError(
                f"[cuda] delta_nodes_bf16: shapes do not agree (T {T}, C {C}, K {K}, heads {hk}/{hv}, dims {dk}/{dv}, "
                f"gate {gate}, proj {tuple(proj.shape)} at {offsets})"
            )
        for n, t in (("slots", slots), ("parents", parents)):
            if t is not None and (scratch is None or t.numel() != T):
                raise ValueError(f"[cuda] delta_nodes_bf16: {n} must be {T} int32 on the card, with a scratch")
        if T:
            P, ci = self.ptr, ctypes.c_int
            self.launch(
                "btb_delta_nodes_bf16",
                (hv, 1, 1),
                (self.DELTA_THREADS, 1, 1),
                [
                    P(proj),
                    ci(ps),
                    ci(oq),
                    ci(oz),
                    ci(ob),
                    ci(oa),
                    P(rows),
                    P(conv_w),
                    P(conv_b),
                    P(conv0),
                    P(state),
                    P(scratch),
                    P(slots),
                    P(parents),
                    P(a_log),
                    P(dt_bias),
                    P(norm_w),
                    ctypes.c_float(float(eps)),
                    ci(int(gate)),
                    ci(T),
                    ci(K),
                    ci(int(hk)),
                    ci(int(hv)),
                    ci(int(dk)),
                    ci(int(dv)),
                    P(out),
                ],
                shared=(2 * dk + 2 * dv + self.DELTA_THREADS) * 4,
            )

    def conv_window(
        self, conv: torch.Tensor, proj: torch.Tensor, q_off: int, rows: torch.Tensor | None, n: int
    ) -> None:
        """the DeltaNet's conv state after a chain of steps (`btb_conv_window`): `conv` [C, K] float32 in place
        becomes the last K of its columns then the q|k|v inputs of proj rows rows[:n] (`rows` int32, -1 padding at the
        end; None: rows 0 .. n-1), read from column `q_off` of `proj` [rows, ps] bf16. Launch it after the nodes."""
        self._want("conv_window", torch.float32, conv=conv)
        self._want("conv_window", torch.bfloat16, proj=proj)
        self._want("conv_window", torch.int32, rows=rows)
        n = int(n)
        if conv.dim() != 2 or proj.dim() != 2:
            raise ValueError(f"[cuda] conv_window: shapes do not agree (conv {tuple(conv.shape)})")
        C, K = (int(s) for s in conv.shape)
        if (
            not 0 < K <= self.DELTA_MAX_K
            or n < 0
            or int(q_off) < 0
            or int(q_off) + C > int(proj.shape[1])
            or (rows is None and n > int(proj.shape[0]))
            or (rows is not None and n > rows.numel())
        ):
            raise ValueError(f"[cuda] conv_window: shapes do not agree (C {C}, K {K}, n {n}, proj {tuple(proj.shape)})")
        if C:
            ci = ctypes.c_int
            self.launch(
                "btb_conv_window",
                ((C + 255) // 256, 1, 1),
                (256, 1, 1),
                [
                    self.ptr(conv),
                    self.ptr(proj),
                    ci(int(proj.shape[1])),
                    ci(int(q_off)),
                    self.ptr(rows),
                    ci(n),
                    ci(C),
                    ci(K),
                ],
            )

    def pick(self, x: torch.Tensor, keys: Sequence[int], temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        """The fused card pick: `x` [R, V] cuda float32, one key a row; [R] int64 cuda. The multi-block pipeline -
        a max, the top-k/top-p thresholds by three radix levels of global histograms (the exact 32-bit threshold),
        then the Gumbel-max draw over the kept tokens - each an ordinary launch over the whole card (a row across
        every SM). Bit-for-bit with the CPU port (native/src/sample.rs); a greedy pick is the argmax upstream."""
        x = x.contiguous().float()
        R, V = int(x.shape[0]), int(x.shape[1])
        if len(keys) != R:
            raise ValueError(f"[cuda] pick: {len(keys)} keys for {R} rows")
        u32 = lambda v: (
            v - (1 << 32) if v >= (1 << 31) else v
        )  # the low/high halves as int32 bits (torch has no uint32)
        kk = torch.tensor(
            [[u32(int(k) & 0xFFFFFFFF), u32((int(k) >> 32) & 0xFFFFFFFF)] for k in keys],
            dtype=torch.int32,
            device=x.device,
        )
        fp = torch.tensor([1.0 / temperature, float(top_p)], dtype=torch.float32, device=x.device)
        gM = torch.zeros(R, dtype=torch.int32, device=x.device)
        ghist = torch.empty(R * 2048, dtype=torch.int64, device=x.device)
        gpreK = torch.zeros(R, dtype=torch.int32, device=x.device)
        gpreP = torch.zeros(R, dtype=torch.int32, device=x.device)
        gcarry = torch.empty(R, dtype=torch.int64, device=x.device)
        gbest = torch.zeros(R, dtype=torch.int64, device=x.device)
        out = torch.empty(R, dtype=torch.int32, device=x.device)
        P, I = self.ptr, ctypes.c_int
        nb = min(max(1, (V + 1023) // 1024), 4 * self.sms)  # blocks a row (grid-stride); an ordinary launch
        self.launch("btb_sample_max", (nb, R, 1), (1024, 1, 1), [P(x), I(V), P(fp), P(gM)])

        def levels(mode: int, prefix: torch.Tensor, floor: torch.Tensor) -> None:
            for lvl in (0, 1, 2):  # three radix levels = the exact 32-bit threshold, bit-for-bit with the CPU kernel
                ghist.zero_()
                self.launch(
                    "btb_sample_hist",
                    (nb, R, 1),
                    (1024, 1, 1),
                    [P(x), I(V), P(fp), I(lvl), I(mode), P(gM), P(prefix), P(floor), P(ghist)],
                )
                self.launch(
                    "btb_sample_bin",
                    (R, 1, 1),
                    (1024, 1, 1),
                    [I(lvl), I(mode), I(int(top_k)), P(fp), P(ghist), P(prefix), P(gcarry)],
                )

        if 0 < top_k < V:
            levels(0, gpreK, gpreP)  # gpreP is 0 here, so the top-k floor is 0
        if top_p < 1.0:
            levels(1, gpreP, gpreK)  # the top-p mass over the tokens the top-k kept
        self.launch(
            "btb_sample_draw",
            (nb, R, 1),
            (1024, 1, 1),
            [P(x), P(kk), I(V), P(fp), P(gM), P(gpreK), P(gpreP), P(gbest)],
        )
        self.launch("btb_sample_out", ((R + 255) // 256, 1, 1), (256, 1, 1), [P(gbest), P(out), I(R)])
        return out.to(torch.int64)

    def verify(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        kids: torch.Tensor,
        hasq: torch.Tensor,
        keys: Sequence[int],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """The drawn-tree verify on the card: `x` [T, V] node logits, `q` [T, Vd] the drafter's distribution a
        node, `kids` [T, C] int32 (draw order, -1 pad), `hasq` [T] uint32 (0 point-mass), one key a row; [T] int64
        packed (slot + 1) << 24 | token. One block a row - a speculative pass carries few rows. Mirrors the Metal
        verify: the emitted token is a draw from the target at every node."""
        x = x.contiguous().float()
        q = q.contiguous().float()
        kids = kids.contiguous().to(torch.int32)
        T, V = int(x.shape[0]), int(x.shape[1])
        Vd, C = int(q.shape[1]), int(kids.shape[1])
        if C > 32:  # SMP_MAXC in btb_kernels.cu: a pass is at most 32 rows, so a node has at most 31 children
            raise ValueError(f"[cuda] verify: {C} children a node; the kernel's table holds 32")
        u32 = lambda v: v - (1 << 32) if v >= (1 << 31) else v
        kk = torch.tensor(
            [[u32(int(k) & 0xFFFFFFFF), u32((int(k) >> 32) & 0xFFFFFFFF)] for k in keys],
            dtype=torch.int32,
            device=x.device,
        )
        hasq = hasq.contiguous().to(torch.int32)
        fp = torch.tensor([1.0 / temperature, float(top_p)], dtype=torch.float32, device=x.device)
        out = torch.empty(T, dtype=torch.int32, device=x.device)
        P, I = self.ptr, ctypes.c_int
        self.launch(
            "btb_sample_verify",
            (T, 1, 1),
            (1024, 1, 1),
            [P(x), P(q), P(kids), P(hasq), P(kk), I(V), I(Vd), I(C), I(int(top_k)), P(fp), P(out)],
        )
        return out.to(torch.int64) & 0xFFFFFFFF  # the packing is unsigned: a slot past 126 sets the sign bit
