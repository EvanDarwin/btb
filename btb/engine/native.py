# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The native kernels' handles (the C-ABI library loaded through ctypes) and the MLX backend, shared by every
module of the engine: `Native.load` binds them once per process."""

from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import sys
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .. import mlx as mlxdev
from ..sysinfo import raise_file_limit

# -- where the built library and the CUDA kernels ship: btb/native/<platform-tag>/ ---------------------------
_LIB_NAMES = {"windows": "btb_native.dll", "linux": "libbtb_native.so", "macos": "libbtb_native.dylib"}
# the package directory (btb/); this module sits one level down, in btb/engine/
_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def native_tag() -> str:
    """Calculate the native platform tag"""
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)
    m = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(m, m)
    return f"{system}-{arch}"


def native_path() -> str | None:
    tag = native_tag()
    own = _LIB_NAMES.get(tag.split("-")[0])
    names = [own] if own else list(_LIB_NAMES.values())
    cands = [os.path.join(_PKG, "native", tag, n) for n in names]
    cands += [os.path.join(_PKG, "native", n) for n in _LIB_NAMES.values()]
    for p in cands:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def kernels_path() -> str | None:
    """the CUDA decode kernels (btb_kernels.fatbin)"""
    tag = native_tag()
    for p in (
        os.path.join(_PKG, "native", tag, "btb_kernels.fatbin"),
        os.path.join(_PKG, "native", "btb_kernels.fatbin"),
    ):
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


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
    "delta_step": (
        "btb_delta_step",
        lambda mixed, conv_state, conv_w, conv_b, z, a, b, a_log, dt_bias, state, hk, hv, dk, dv, norm_w, eps, out: {
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


class Native:
    """Process-wide handles: the native kernels once `load_gemv` bound them (None without the library), the MLX
    backend once an engine made one, and the row thresholds routing a matmul between kernels and GEMMs."""

    _gemv_lib: Any = None
    HANDLES = (
        "gemv",
        "gemv_p12",
        "gemv_group",
        "gemv_mx4",
        "gemv_mx4_group",
        "gemv_mx4_ggml",
        "gemv_mx4_ggml_group",
        "attn_decode",
        "delta_step",
        "read_direct",
        "open",
        "read_at",
        "close",
    )

    gemv: Any = None
    gemv_p12: Any = None
    gemv_group: Any = None
    gemv_mx4: Any = None
    gemv_mx4_group: Any = None
    gemv_mx4_ggml: Any = None
    gemv_mx4_ggml_group: Any = None
    attn_decode: Any = None
    delta_step: Any = None
    sample_pick: Any = None
    read_direct: Any = None
    open: Any = None
    read_at: Any = None
    close: Any = None
    # the MLX backend (`mlxdev.Backend`) of the engine on the MLX device
    mlx: Any = None
    # a matmul of this many rows or more goes to the GEMM; fewer rows take the one-row kernel per row
    gemm_rows = 64

    # a prefill of this many rows or more goes to the CPU-stream GEMM; fewer (one-row steps, verify passes of <= 16
    # rows) stay on the native kernel, so a verify row computes as the one-row step does
    cpu_gemm_rows = 17

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
        cls.open = cls.read_at = cls.close = None
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
        cls.delta_step = None
        if hasattr(lib, "btb_delta_step"):
            ds = lib.btb_delta_step
            ds.restype = ctypes.c_int32
            P = ctypes.c_void_p
            S = ctypes.c_size_t
            ds.argtypes = [P, P, P, P, S, S, P, P, P, P, P, P, S, S, S, S, P, ctypes.c_float, P, S]

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
            ) -> None:
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
        if checked:
            for name in _WANTS:
                bound = getattr(cls, name)
                if bound is not None:
                    setattr(cls, name, _checked(name, bound))
        return cls.gemv

    # the card's own kernels (a `_Cuda`), bound by `load_cuda` once per process; None until then
    cuda: Any = None
    cuda_reason: str | None = None  # why they are not bound, once `card_kernels` tried and failed

    @classmethod
    def load_cuda(cls, fatbin_path: str | os.PathLike[str]) -> _Cuda:
        if cls.cuda is None:
            cls.cuda = _Cuda(fatbin_path)
        return cls.cuda

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
        "btb_attn_split_d64",
        "btb_attn_split_d128",
        "btb_attn_split_d256",
        "btb_norm_rope_kv_d64",
        "btb_norm_rope_kv_d128",
        "btb_norm_rope_kv_d256",
        "btb_add_rmsnorm",
        "btb_sandwich_add",
        "btb_silu_mul",
        "btb_gelu_mul",
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
        import sys

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
        self._call("cuCtxSetCurrent", ctx)
        with open(fatbin_path, "rb") as fh:
            self._image = fh.read()
        self.module = ctypes.c_void_p()
        self._call("cuModuleLoadData", ctypes.byref(self.module), ctypes.c_char_p(self._image))
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

    @staticmethod
    def ptr(t: torch.Tensor | None) -> ctypes.c_void_p:
        return ctypes.c_void_p(0 if t is None else t.data_ptr())

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
