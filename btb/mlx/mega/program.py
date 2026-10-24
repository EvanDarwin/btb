# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A pass's program: its instructions over tensors in a write-once scratch, appended op by op. Each instruction reads
the output of the ones that wrote the tensors it reads, recorded as it is appended, so a block is written as the ops
it runs and the table's dependencies follow from it."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .kernel import CODE, DEPS, NUM, REF0, REFS, OpName, W

# the buffers past the weights, numbered after them once the weights are counted: the scratch, the constants, then
# each layer's K and V (`kv`)
SCR, CONSTS = -1, -2

Ref: TypeAlias = tuple[int, int]  # (buffer, byte offset), as the table carries a reference


def kv(i: int, which: int) -> int:
    """layer i's K (0) or V (1) rows [Hk, cap, D], a buffer of their own a pass binds to the cache it runs over"""
    return -3 - (2 * i + which)


@dataclass
class Tensor:
    """a region an op reads or writes: `ref` its start, `cols` a row's elements of `esize` bytes; `src` the instruction
    that wrote it this pass (-1 before, or for a weight or a constant, which nothing writes); `scratch` a region the
    pass writes before any op reads it; `parts` the pieces of it that several ops write, a reader waiting on each"""

    ref: Ref
    cols: int = 0
    esize: int = 2
    src: int = -1
    scratch: bool = False
    parts: tuple[Tensor, ...] = ()

    def piece(self, col: int, cols: int) -> Tensor:
        """`cols` of each row from column `col`, a region of its own an op writes (its rows `self.cols` apart)"""
        return Tensor((self.ref[0], self.ref[1] + col * self.esize), cols, self.esize, scratch=self.scratch)


# an instruction's references in order, A, B (read), C, D (written), E, F (read, or written by the op that says so);
# fewer than REFS leave the rest unset
Refs: TypeAlias = tuple[Tensor | None, ...]


def fbits(x: float) -> int:
    """a float's bits as the table carries them"""
    return int(struct.unpack("<I", struct.pack("<f", float(x)))[0])


def align(n: int) -> int:
    return (n + 255) // 256 * 256


class Program:
    """the instructions of a pass of `T` rows, its scratch laid out for `rows` rows (the widest pass, so every width's
    program places its tensors alike)"""

    def __init__(self, T: int, rows: int) -> None:
        self.T, self.R = T, rows
        self.size = 0  # the scratch's bytes
        self.rows: list[list[int]] = []
        self.ops: set[OpName] = set()
        self.before: list[int] = []  # each instruction's transitive producers, a bit each

    def scratch(self, cols: int, esize: int = 2, rows: int | None = None) -> Tensor:
        """a fresh region of `rows` (the widest pass's, by default) rows of `cols` elements"""
        t = Tensor((SCR, self.size), cols, esize, scratch=True)
        self.size += align((self.R if rows is None else rows) * cols * esize)
        return t

    def emit(
        self,
        op: OpName,
        items: int,
        refs: Refs,
        n: tuple[int, int, int] = (0, 0, 0),
        f0: float = 0.0,
        reads: Iterable[Tensor] = (),
        writes: Iterable[Tensor] = (),
    ) -> int:
        """append `op` over `items` work items, a threadgroup each, with its references and numbers; it reads the output
        of the writers of `reads` and becomes the writer of `writes`, each written once a pass"""
        idx = len(self.rows)
        read: set[int] = set()
        for t in (u for r in reads for u in r.parts or (r,)):
            if t.scratch and t.src < 0:
                raise ValueError(f"[mega] {op} reads a region no op has written this pass")
            if t.src >= 0:
                read.add(t.src)
        # a writer another read-from writer already read from is left out: a dispatch ordered after that one is ordered
        # after it too (the replay's barriers place a command after every one before it)
        deps = {d for d in read if not any(self.before[e] >> d & 1 for e in read)}
        before = 0
        for d in deps:
            before |= self.before[d] | 1 << d
        self.before.append(before)
        if len(deps) > DEPS:
            raise ValueError(f"[mega] {op} reads the output of {len(deps)} instructions, the table holds {DEPS}")
        for t in writes:
            if t.src >= 0:
                raise ValueError(f"[mega] {op} writes a region instruction {t.src} wrote this pass")
            t.src = idx
        if len(refs) > REFS:
            raise ValueError(f"[mega] {op} takes {len(refs)} references, the table holds {REFS}")
        flat = [v for r in (*refs, *[None] * (REFS - len(refs))) for v in (r.ref if r is not None else (0, 0))]
        waits = [d + 1 for d in sorted(deps)]
        self.rows.append([CODE[op], int(items), *flat, *n, fbits(f0), *waits, *[0] * (DEPS - len(waits))])
        self.ops.add(op)
        return idx

    def table(self, nbufs: int) -> np.ndarray:
        """the table, uint32 [N, W], the buffers past the weights numbered from `nbufs` (the scratch, the constants,
        layer i's K at nbufs + 2 + 2i and its V beside it)"""
        tab = np.asarray(self.rows, dtype=np.int64)
        for col in range(REF0, NUM, 2):
            tab[:, col] = np.where(tab[:, col] < 0, nbufs - 1 - tab[:, col], tab[:, col])
        assert tab.shape[1] == W
        return tab.astype(np.uint32)
