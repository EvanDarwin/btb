# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A pass replayed from a Metal indirect command buffer (`btb.metal`): a pass shape's program encoded once, each
instruction a dispatch of its own threadgroups the GPU schedules itself, replayed in one command buffer. It reads and
writes MLX's own buffers - the weights, the constants, the scratch, a cache's arena - aliased page for page, never
copied; a pass's inputs are buffers of the replay's own. A pass starts once MLX's work before it is done and is done
before the host reads its ids, so MLX's next work sees what it wrote."""

from __future__ import annotations

import ctypes
import threading
import time
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ... import trace
from ...metal import Metal
from ..core import mx
from .kernel import (
    CODE,
    DEP0,
    DEPS,
    DIMS,
    GRID3,
    MV_Q,
    NUM,
    OpName,
    StoredMv,
    kernel_name,
    source,
    stored_name,
    stored_source,
)

if TYPE_CHECKING:
    import mlx.core as mx_

    from .model import KvOff, MegaPass

THREADS = 256  # a threadgroup's threads, every op's
OP_OF: dict[int, OpName] = {c: o for o, c in CODE.items()}


def _alias(m: Metal, arr: mx_.array) -> int:
    """MLX's evaluated array `arr` as a buffer over its own bytes"""
    return m.alias(np.array(arr, copy=False))


class Shape(NamedTuple):
    """a pass shape's program encoded: its indirect command buffer, its commands, its table's own buffer"""

    icb: int
    n: int
    tab: int


class Arena(NamedTuple):
    """a cache's arena as the replay binds it: its alias and the shapes encoded over it"""

    buf: int
    shapes: dict[tuple[int, int, int, int, int, bool], Shape]


class Replay:
    """a MegaPass's passes on the GPU: its buffers aliased, its ops compiled a kernel each (per row cap, XROWS 8 and
    16; a stored matvec per matrix and pass width, as the step compiles it), and each pass shape's program over each
    arena encoded once"""

    def __init__(self, mp: MegaPass) -> None:
        self.mp = mp
        self.metal = m = Metal()
        # reentrant: an arena's finalizer can run from a collection inside a pass (never on that pass's own arena)
        self.lock = threading.RLock()
        R = mp.ROWS
        self.bufs = [_alias(m, b) for b in mp.bufs]
        self.consts, self.scr = _alias(m, mp.consts), _alias(m, mp.scr)
        self.freqs = m.buffer(int(mp.freqs.nbytes))
        m.write(self.freqs, np.asarray(mp.freqs))
        self.ids, self.pos, self.out = m.buffer(4 * R), m.buffer(4 * R), m.buffer(4 * R)
        self.dims = m.buffer(4 * DIMS)
        self.meta = m.buffer(4 * 8 * R * len(mp.wins))
        self.path = m.buffer(4 * R * max(R, 8))
        self.mi = m.buffer(4 * mp.instructions)
        m.write(self.mi, np.arange(mp.instructions, dtype=np.uint32))
        self.psos = {xr: self._pipelines(xr) for xr in (8, 16)}
        self.stored: dict[StoredMv, int] = {}
        self.arenas: dict[int, Arena] = {}

    def _pipelines(self, xr: int) -> dict[int, int]:
        """the ops' kernels for passes of up to xr rows, by op code (the stored matvecs apart)"""
        mp, m = self.mp, self.metal
        ops = mp.ops - set(MV_Q)
        t0 = time.perf_counter()
        m.compile(source(mp.nb, ops, mp.defs(xr), mp.tgm(xr)))
        got = {CODE[o]: m.pipeline(kernel_name(o)) for o in ops}
        ms = 1e3 * (time.perf_counter() - t0)
        trace.event("megakernel: %d ops compiled for passes of up to %d rows in %.0f ms", len(ops), xr, ms)
        return got

    def _stored(self, mvs: Sequence[StoredMv]) -> None:
        """the kernels of the stored matvecs `mvs` not yet compiled, in one library (a pass width's, on its first
        pass: the step's launches compile on first use too)"""
        new = list(dict.fromkeys(s for s in mvs if s not in self.stored))
        if new:
            t0 = time.perf_counter()
            self.metal.compile(stored_source(self.mp.nb, new))
            self.stored.update({s: self.metal.pipeline(stored_name(s)) for s in new})
            ms = 1e3 * (time.perf_counter() - t0)
            widths = ", ".join(str(t) for t in sorted({s.T for s in new}))
            trace.event("megakernel: %d stored matvecs compiled for %s-row passes in %.0f ms", len(new), widths, ms)

    @staticmethod
    def _mv(f: np.ndarray, T: int) -> StoredMv | None:
        """instruction row `f`'s stored matvec, or None where it is another op"""
        q = MV_Q.get(OP_OF[int(f[0])])
        if q is None:
            return None
        rows, cols, n2 = int(f[NUM]), int(f[NUM + 1]), int(f[NUM + 2])
        return StoredMv(q, rows, cols, T, (n2 >> 1) or rows, bool(n2 & 1))

    def _arena(self, kv: mx_.array) -> Arena:
        """`kv`'s alias and shapes, both let go when it is (the replay held weakly: a cache outliving it holds nothing
        of it)"""
        got = self.arenas.get(id(kv))
        if got is None:
            got = self.arenas[id(kv)] = Arena(_alias(self.metal, kv), {})
            weakref.finalize(kv, Replay._drop_from, weakref.ref(self), id(kv))
        return got

    @staticmethod
    def _drop_from(ref: weakref.ref[Replay], key: int) -> None:
        r = ref()
        if r is not None:
            r._drop(key)

    def _drop(self, key: int) -> None:
        """an arena gone: its shapes' command buffers, then its alias (a command buffer does not hold what it binds)"""
        with self.lock:
            a = self.arenas.pop(key, None)
            if a is None:
                return
            for s in a.shapes.values():
                self.metal.free_icb(s.icb)
                self.metal.free_buffer(s.tab)
            self.metal.free_buffer(a.buf)

    def _encode(self, a: Arena, T: int, splits: int, kv_off: KvOff, argmax: bool) -> Shape:
        """the program of a pass shape over arena `a` as commands: a dispatch an instruction, its threadgroups its
        units (a GRID3 op's (kv heads, blocks, rows)), after a barrier where it reads an instruction encoded since the
        last one"""
        mp, m = self.mp, self.metal
        tab = mp.table(T, splits, kv_off, argmax)
        n = int(tab.shape[0])
        tb = m.buffer(int(tab.nbytes))
        m.write(tb, tab)
        psos = self.psos[8 if T <= 8 else 16]
        mvs = [self._mv(f, T) for f in tab]
        self._stored([s for s in mvs if s is not None])
        t0 = time.perf_counter()
        order = [*self.bufs, self.scr, self.consts, a.buf, self.freqs, self.ids, self.pos, self.meta, self.path, tb]
        order += [self.dims, self.out, self.mi]
        binds = (ctypes.c_int64 * len(order))(*order)
        icb = m.icb(n, len(order))
        last, barriers = 0, 0  # the commands from `last` on may run beside each other
        for k in range(n):
            f, mv = tab[k], mvs[k]
            op = OP_OF[int(f[0])]
            deps = [int(d) - 1 for d in f[DEP0 : DEP0 + DEPS] if int(d)]
            barrier = any(d >= last for d in deps)
            if barrier:
                last, barriers = k, barriers + 1
            grid = (int(f[NUM]), int(f[NUM + 1]), int(f[NUM + 2]) & 0xFFFF) if op in GRID3 else (int(f[1]), 1, 1)
            pso = psos[int(f[0])] if mv is None else self.stored[mv]
            m.cmd(icb, k, pso, binds, [0] * (len(order) - 1) + [4 * k], grid, THREADS, barrier)
        ms = 1e3 * (time.perf_counter() - t0)
        trace.event("megakernel: a %d-row pass encoded, %d commands after %d barriers, in %.1f ms", T, n, barriers, ms)
        return Shape(icb, n, tb)

    def run(
        self,
        kv: mx_.array,
        cap: int,
        kv_off: KvOff,
        ids: Sequence[int],
        pos: Sequence[int],
        meta: mx_.array,
        path: mx_.array,
        splits: int,
        past: int,
        argmax: bool,
    ) -> np.ndarray:
        """the pass over arena `kv`, run and waited for: the argmax ids uint32 [T] (zeros without `argmax`)"""
        mp, m = self.mp, self.metal
        T = len(ids)
        pad = [0] * (mp.ROWS - T)
        meta_np = np.asarray(meta).astype(np.uint32)
        path_np = np.asarray(path).astype(np.uint32)
        if path_np.shape[1] < 8:
            path_np = np.pad(path_np, ((0, 0), (0, 8 - path_np.shape[1])))
        dims = np.zeros(DIMS, dtype=np.uint32)
        dims[:4] = (T, path_np.shape[1], cap, past)
        # the arena's rows and anything else MLX was writing, done before the replay reads them
        mx().synchronize()
        with self.lock:
            a = self._arena(kv)
            key = (T, splits, kv_off(0, 0), kv_off(0, 1), kv_off(mp.L - 1, 1), argmax)
            s = a.shapes.get(key)
            if s is None:
                s = a.shapes[key] = self._encode(a, T, splits, kv_off, argmax)
            m.write(self.ids, np.asarray([*ids, *pad], dtype=np.uint32))
            m.write(self.pos, np.asarray([*pos, *pad], dtype=np.uint32))
            m.write(self.meta, meta_np)
            m.write(self.path, path_np)
            m.write(self.dims, dims)
            m.run(s.icb, s.n)
            return m.read(self.out, T, np.uint32) if argmax else np.zeros(T, dtype=np.uint32)
