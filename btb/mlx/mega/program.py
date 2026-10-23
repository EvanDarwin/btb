# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A pass's program: the megakernel's instructions over tensors in a write-once scratch, appended op by op. Each
instruction waits on the ones that wrote the tensors it reads, recorded as it is appended, so a block is written as
the ops it runs and the table's tickets and dependencies follow from it."""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .kernel import CODE, DEPS, NUM, REF0, REFS, TICKETS_MAX, OpName, W

# the buffers past the weights, placed after them once the weights are counted: the scratch, the constants, the arena
SCR, CONSTS, KV = -1, -2, -3

Ref: TypeAlias = tuple[int, int]  # (buffer, byte offset), as the table carries a reference


@dataclass
class Tensor:
    """a region an op reads or writes: `ref` its start, `cols` a row's elements of `esize` bytes; `src` the instruction
    that wrote it this pass (-1 before, or for a weight or a constant, which nothing writes); `scratch` a region the
    pass writes before any op reads it"""

    ref: Ref
    cols: int = 0
    esize: int = 2
    src: int = -1
    scratch: bool = False


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
    program places its tensors alike) and its work cut for `grid` threadgroups"""

    def __init__(self, T: int, rows: int, grid: int) -> None:
        if grid > TICKETS_MAX:
            raise ValueError(f"[mega] a grid of {grid} threadgroups, an instruction's claim bits hold {TICKETS_MAX}")
        self.T, self.R, self.grid = T, rows, grid
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
        """append `op` over `items` work items (cut into at most a grid's tickets, ticket k threadgroup k's, a ticket's
        share of them run in turn) with its references and numbers; it waits on the writers of `reads` and becomes the
        writer of `writes`, each written once a pass"""
        idx = len(self.rows)
        read: set[int] = set()
        for t in reads:
            if t.scratch and t.src < 0:
                raise ValueError(f"[mega] {op} reads a region no op has written this pass")
            if t.src >= 0:
                read.add(t.src)
        # a writer another waited-on writer already waited on is left out: its writes are seen through that one's
        # count (each count is published after a device fence and read before one), and every wait is a serial
        # round trip to memory a ticket makes before its work
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
        tickets = min(int(items), self.grid)
        self.rows.append([CODE[op], tickets, int(items), *flat, *n, fbits(f0), *waits, *[0] * (DEPS - len(waits))])
        self.ops.add(op)
        return idx

    def table(self, nbufs: int) -> np.ndarray:
        """the table, uint32 [N, W], the buffers past the weights numbered from `nbufs`"""
        tab = np.asarray(self.rows, dtype=np.int64)
        for col in range(REF0, NUM, 2):
            tab[:, col] = np.where(tab[:, col] < 0, nbufs - 1 - tab[:, col], tab[:, col])
        assert tab.shape[1] == W
        return tab.astype(np.uint32)
