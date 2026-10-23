# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""MLX on Apple silicon: availability, dtypes, the torch <-> MLX conversions, and shared buffers (bytes MLX
owns that torch views, so the two sides never copy)."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, TypeAlias, TypeVar

import numpy as np
import torch

from .. import trace
from ..kinds import Json
from ..metal import paged

if TYPE_CHECKING:
    import mlx.core as mx_

Template: TypeAlias = "Sequence[tuple[str, mx_.Dtype | int | bool]]"  # a kernel's compile-time values, by name


class MetalKernel(Protocol):
    """what `mx.fast.metal_kernel` builds (its stub says `object`): a launch, its outputs lazily"""

    def __call__(
        self,
        *,
        inputs: Sequence[mx_.array],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[mx_.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Template | None = ...,
    ) -> list[mx_.array]: ...


class _Kernel:
    """a kernel `metal_kernel` builds: MLX's, the first launch of each of its instances timed under the trace"""

    def __init__(self, name: str, k: MetalKernel) -> None:
        self.name, self.k = name, k
        self.seen: set[tuple[str, ...]] = set()

    def __call__(
        self,
        *,
        inputs: Sequence[mx_.array],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[mx_.Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Template | None = None,
    ) -> list[mx_.array]:
        def launch() -> list[mx_.array]:
            return self.k(
                inputs=inputs,
                output_shapes=output_shapes,
                output_dtypes=output_dtypes,
                grid=grid,
                threadgroup=threadgroup,
                template=template,
            )

        if not trace.ON:
            return launch()
        key = (repr(template), *(str(a.dtype) for a in inputs), *(str(d) for d in output_dtypes))
        if key in self.seen:
            return launch()
        self.seen.add(key)
        m = mx()
        m.eval(*inputs)
        t0 = time.perf_counter()
        out = launch()
        m.eval(*out)
        values = ", ".join(f"{n}={str(v).removeprefix('mlx.core.')}" for n, v in template or ())
        trace.event("mlx: kernel %s(%s) compiled and run once in %.1f ms", self.name, values, _ms(t0))
        return out


def metal_kernel(
    name: str, input_names: Sequence[str], output_names: Sequence[str], source: str, header: str = ""
) -> MetalKernel:
    """`mx.fast.metal_kernel`. MLX compiles each instance of a kernel (its template's values, its inputs' and outputs'
    types) at its first evaluation, inside whatever graph launched it; under the trace (-vv) an instance's first launch
    is evaluated alone, its inputs first, and said with its time"""
    k = mx().fast.metal_kernel(
        name=name, input_names=list(input_names), output_names=list(output_names), source=source, header=header
    )
    return _Kernel(name, k)


P = ParamSpec("P")
R = TypeVar("R")


def compiled(name: str, fn: Callable[P, R]) -> Callable[P, R]:
    """`mx.compile(fn)`. MLX traces and compiles it again for each new set of argument shapes and constants, at that
    call; under the trace (-vv) such a call is evaluated alone, its arguments first, and said with its time"""
    m = mx()
    c = m.compile(fn)
    seen: set[tuple[str, ...]] = set()

    def run(*args: P.args, **kw: P.kwargs) -> R:
        if not trace.ON:
            return c(*args, **kw)
        arrays = [a for a in (*args, *kw.values()) if isinstance(a, m.array)]
        key = tuple(f"{a.shape}{a.dtype}" if isinstance(a, m.array) else repr(a) for a in (*args, *kw.values()))
        if key in seen:
            return c(*args, **kw)
        seen.add(key)
        m.eval(*arrays)
        t0 = time.perf_counter()
        out = c(*args, **kw)
        m.eval(out)
        trace.event("mlx: %s compiled and run once in %.1f ms", name, _ms(t0))
        return out

    return run


def _ms(t0: float) -> float:
    return 1e3 * (time.perf_counter() - t0)


_mx: Any = None
_TorchTensor = torch.Tensor
_TorchDtype = torch.dtype


_np_dtype: dict[Any, Any] = {}


def available() -> bool:
    try:
        import mlx.core as mx
    except ImportError:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def mx() -> ModuleType:
    global _mx
    if _mx is None:
        import mlx.core as m

        _mx = m
    return _mx


def info() -> Json:
    m = mx()
    return dict(m.device_info())


_TORCH_TO_MX = None


def _dtypes() -> dict[torch.dtype, Any]:
    global _TORCH_TO_MX
    if _TORCH_TO_MX is None:
        m = mx()
        _TORCH_TO_MX = {
            torch.float32: m.float32,
            torch.bfloat16: m.bfloat16,
            torch.float16: m.float16,
            torch.int32: m.int32,
            torch.int64: m.int64,
            torch.uint8: m.uint8,
            torch.int16: m.int16,
            torch.bool: m.bool_,
        }
    return _TORCH_TO_MX


def torch_dtype(mx_dtype: Any) -> torch.dtype:
    """The torch dtype of an MLX dtype, without touching the array (a view would evaluate it)."""
    for t, m in _dtypes().items():
        if m == mx_dtype:
            return t
    raise TypeError(f"[mlx] no torch dtype for {mx_dtype}")


def to_mx(t: torch.Tensor) -> mx_.array:
    """A torch CPU tensor as an MLX array (a copy; the activations that cross are small)."""
    m = mx()
    t = t.detach()
    if not t.is_contiguous():
        t = t.contiguous()
    if t.dtype == torch.bfloat16:
        return m.array(t.view(torch.int16).numpy()).view(m.bfloat16)
    if t.dtype == torch.float16:
        return m.array(t.view(torch.int16).numpy()).view(m.float16)
    return m.array(t.numpy())


def from_mx(a: mx_.array) -> torch.Tensor:
    """An MLX array as a torch CPU tensor over the same memory (evaluates the array). bf16 crosses as int16 bits."""
    m = mx()
    if a.dtype == m.bfloat16:
        return torch.from_numpy(np.array(a.view(m.int16), copy=False)).view(torch.bfloat16)
    if a.dtype == m.float16:
        return torch.from_numpy(np.array(a.view(m.int16), copy=False)).view(torch.float16)
    return torch.from_numpy(np.array(a, copy=False))


class Shared:
    """A byte buffer allocated by MLX and viewed by torch: what the cold tier and the expert store stream into,
    and what the attention cache grows in, so the GPU reads what the CPU wrote with no copy."""

    def __init__(self, nbytes: int, stream: Any = None, defer: bool = False) -> None:
        m = mx()
        self.nbytes = int(nbytes)
        n = max(1, self.nbytes)
        # `stream=m.cpu` zero-fills on the CPU, so the pages are touched once, by the side that writes the
        # weights next; `defer` queues the fill and returns, and the views wait for it on first use
        with m.stream(stream) if stream is not None else contextlib.nullcontext():
            # whole pages underneath (`whole`, what btb.metal aliases), the views over the bytes asked for
            self.whole = m.zeros((paged(n),), dtype=m.uint8)
            self.mx = self.whole if self.whole.size == n else self.whole[:n]
        self._np: Any = None
        self._torch: Any = None
        if defer:
            m.async_eval(self.mx)
        else:
            self._ready()

    @classmethod
    def wrap(cls, arr: mx_.array, npview: np.ndarray | None = None) -> Shared:
        """a Shared over an evaluated MLX byte array made elsewhere (the pool's blocks)"""
        self = cls.__new__(cls)
        self.nbytes = int(arr.size)
        self.whole = self.mx = arr
        self._np = npview if npview is not None else np.array(arr, copy=False)
        self._torch = torch.from_numpy(self._np)
        return self

    def _ready(self) -> None:
        if self._np is None:
            mx().eval(self.mx)
            self._np = np.array(self.mx, copy=False)
            self._torch = torch.from_numpy(self._np)

    @property
    def np(self) -> np.ndarray:
        self._ready()
        return self._np

    @property
    def torch(self) -> _TorchTensor:
        self._ready()
        return self._torch

    def view_mx(self, off: int, nbytes: int, dtype: Any, shape: Sequence[int]) -> mx_.array:
        return self.mx[off : off + nbytes].view(dtype).reshape(*shape)

    def view_torch(self, off: int, nbytes: int, dtype: _TorchDtype, shape: Sequence[int]) -> _TorchTensor:
        return self.torch[off : off + nbytes].view(dtype).view(*shape)


def shared_tensor(shape: Sequence[int], dtype: torch.dtype) -> Any:
    """A torch tensor of `shape`/`dtype` over MLX memory, with its MLX twin (`t._mx_twin`) of the same shape."""
    n = 1
    for d in shape:
        n *= int(d)
    nb = n * torch.empty(0, dtype=dtype).element_size()
    sh = Shared(nb)
    t: Any = sh.view_torch(0, nb, dtype, shape) if n else torch.empty(shape, dtype=dtype)
    twin = sh.view_mx(0, nb, _dtypes()[dtype], shape) if n else mx().zeros(shape, dtype=_dtypes()[dtype])
    t._mx_shared = sh
    t._mx_twin = twin
    return t


def bf16_weight(t: torch.Tensor) -> mx_.array:
    """A bf16 [rows, cols] torch tensor (an mmap view of the checkpoint) copied into unified memory as bf16."""
    m = mx()
    a = m.array(t.detach().contiguous().view(torch.int16).numpy()).view(m.bfloat16)
    m.eval(a)
    return a
