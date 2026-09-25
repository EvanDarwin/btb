# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""How a packed weight's Metal kernels are built and launched, once for every storage type (kquant.py, q6k.py
and iquant.py hold the sources). A `RowKernel` is the one-simdgroup-a-row matvec family over one source, built
per (rows, blocks a row, rows of x) with the #defines the source reads and launched over x in chunks of ROWS_MAX
rows; a `BlockKernel` is a dequant or a gather over blocks, built once. `mx.fast.metal_kernel` compiles on first
use, so every kernel is cached for the process under one lock."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from .core import mx

if TYPE_CHECKING:
    import mlx.core as mx_

    # a compiled `mx.fast.metal_kernel`: called with its inputs, grid, threadgroup, output shapes/dtypes and
    # template values, it returns its outputs
    MetalKernel = Callable[..., list[mx_.array]]
    # the device arrays a launch reads (a weight's streams, the tables a kernel indexes)
    Streams = Sequence[mx_.array]
    # a kernel's template values: a dtype (T) or an integer constant (NBLK)
    Template = list[tuple[str, mx_.Dtype | int]]

# rows a matvec launch takes (acc[TR] in registers); a wider pass is chunked
ROWS_MAX = 16
# a compiled kernel by its stem and the #defines it was built with
KernelKey = tuple[str | int, ...]
_lock = threading.Lock()
_kernels: dict[KernelKey, MetalKernel] = {}


def _build(key: KernelKey, name: str, inputs: Sequence[str], header: str, source: str) -> MetalKernel:
    with _lock:
        k = _kernels.get(key)
        if k is None:
            k = _kernels[key] = mx().fast.metal_kernel(
                name=name, input_names=list(inputs), output_names=["out"], header=header, source=source
            )
    return k


class RowKernel:
    """A matvec family over one Metal source: `stem` names it, `inputs` its buffers in order (the streams before
    x, x, then the tables and side streams after), `header` the source's static prelude, `nb_define` the name the
    source reads the blocks-a-row count under. `matvec` is y[T, rows] = x[T, cols] . W^T in x's dtype, fp32
    accumulation in the kernel, 1..ROWS_MAX rows a launch, batch-invariant (row i the same at any T)."""

    def __init__(self, stem: str, inputs: Sequence[str], header: str, source: str, nb_define: str = "NSB") -> None:
        self.stem, self.inputs, self.header, self.source, self.nb_define = (
            stem,
            tuple(inputs),
            header,
            source,
            nb_define,
        )

    def matvec(self, before: Streams, x: mx_.array, after: Streams, rows: int, nb: int) -> mx_.array:
        m = mx()
        grid = ((rows * 32 + 255) // 256) * 256  # a simdgroup an output row, padded to whole 256-thread groups
        outs = []
        for s in range(0, int(x.shape[0]), ROWS_MAX):
            xr = x[s : s + ROWS_MAX]
            tr = int(xr.shape[0])
            k = _build(
                (self.stem, rows, nb, tr),
                f"btb_{self.stem}_{rows}_{nb}_{tr}",
                self.inputs,
                f"{self.header}#define ROWS {rows}\n#define {self.nb_define} {nb}\n#define TR {tr}\n",
                self.source,
            )
            outs.append(
                k(
                    inputs=[*before, xr, *after],
                    grid=(grid, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(tr, rows)],
                    output_dtypes=[x.dtype],
                    template=[("T", x.dtype)],
                )[0]
            )
        return outs[0] if len(outs) == 1 else m.concatenate(outs, axis=0)


class BlockKernel:
    """A kernel over blocks - a dequant, a gather - built once per `stem`, `per` threads a block (1, or 32 for a
    simdgroup a block). `run` launches it over `n` blocks with `streams` as its inputs, writing bf16 of `shape`;
    the source reads NBLK where it takes the count (`nblk`), and T is always the output dtype."""

    def __init__(self, stem: str, inputs: Sequence[str], header: str, source: str, per: int = 1) -> None:
        self.stem, self.inputs, self.header, self.source, self.per = stem, tuple(inputs), header, source, per

    def run(self, streams: Streams, n: int, shape: tuple[int, int], nblk: bool = True) -> mx_.array:
        m = mx()
        k = _build((self.stem,), f"btb_{self.stem}", self.inputs, self.header, self.source)
        template: Template = [("T", m.bfloat16)]
        if nblk:
            template.append(("NBLK", n))
        return k(
            inputs=list(streams),
            grid=(((n * self.per + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[shape],
            output_dtypes=[m.bfloat16],
            template=template,
        )[0]
