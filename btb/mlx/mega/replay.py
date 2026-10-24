# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A pass replayed from a Metal indirect command buffer (`btb.metal`): a pass shape's program encoded once, each
instruction a dispatch of its own threadgroups the GPU schedules itself, replayed in one command buffer. It reads and
writes MLX's own buffers - the weights, the constants, the scratch, a cache's K and V - aliased page for page, never
copied, each through a table of their GPU addresses the pass's references index, so a pass binds whichever cache it
runs over by rewriting the table. A pass starts once MLX's work before it is done and is done before the host reads
its ids, so MLX's next work sees what it wrote."""

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
from ..attn import ATTN_BLOCK
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

    from .model import MegaPass

THREADS = 256  # a threadgroup's threads, every op's
OP_OF: dict[int, OpName] = {c: o for o, c in CODE.items()}


class KvSlot(NamedTuple):
    """a layer's K or V rows [Hk, cap, D] bf16 as a pass binds them: the MLX array they lie in (an allocation of whole
    pages, `Shared.whole`) and their byte offset in it"""

    arr: mx_.array
    off: int


class Shape(NamedTuple):
    """a pass shape's program encoded: its indirect command buffer, its commands, its table's own buffer"""

    icb: int
    n: int
    tab: int


def _alias(m: Metal, arr: mx_.array) -> int:
    """MLX's evaluated array `arr` as a buffer over its own bytes"""
    return m.alias(np.array(arr, copy=False))


class Replay:
    """a MegaPass's passes on the GPU: its buffers aliased and their addresses tabled, its ops compiled a kernel each
    (per row cap, XROWS 8 and 16, and per kind of cache, rows where they lie or through a row map; a stored matvec per
    matrix and pass width, as the step compiles it), and each pass shape's program encoded once"""

    def __init__(self, mp: MegaPass) -> None:
        self.mp = mp
        self.metal = m = Metal()
        # reentrant: a region's finalizer can run from a collection inside a pass (never on a region that pass binds)
        self.lock = threading.RLock()
        R = mp.ROWS
        self.addr = np.zeros(mp.nb + 2 + 2 * mp.L, dtype=np.uint64)  # the table of addresses, `Program.table`'s order
        for j, b in enumerate([*(_alias(m, a) for a in mp.bufs), _alias(m, mp.scr), _alias(m, mp.consts)]):
            self.addr[j] = m.gpu_address(b)
        self.ptrs = m.buffer(8 * self.addr.size)
        self.freqs = m.buffer(int(mp.freqs.nbytes))
        m.write(self.freqs, np.asarray(mp.freqs))
        self.ids, self.pos, self.out = m.buffer(4 * R), m.buffer(4 * R), m.buffer(4 * R)
        self.dims = m.buffer(4 * DIMS)
        self.meta = m.buffer(4 * 8 * R * len(mp.wins))
        self.path = m.buffer(4 * R * max(R, 8))
        self.rows = mp.SPL * ATTN_BLOCK  # the rows a pass reads at most: a row map's length
        self.rmap = m.buffer(4 * self.rows)
        self.mi = m.buffer(4 * mp.instructions)
        m.write(self.mi, np.arange(mp.instructions, dtype=np.uint32))
        self.psos: dict[tuple[int, bool], dict[int, int]] = {}
        self.stored: dict[StoredMv, int] = {}
        self.shapes: dict[tuple[int, int, bool, bool], Shape] = {}
        self.kv: dict[int, tuple[int, int]] = {}  # an aliased K/V array's id -> (its buffer, its address)

    def _pipelines(self, xr: int, mapped: bool) -> dict[int, int]:
        """the ops' kernels for passes of up to xr rows over a cache's rows where they lie or through a row map, by op
        code (the stored matvecs apart), compiled the first time a pass needs them"""
        got = self.psos.get((xr, mapped))
        if got is None:
            mp, m = self.mp, self.metal
            ops = mp.ops - set(MV_Q)
            t0 = time.perf_counter()
            m.compile(source(ops, mp.defs(xr), mp.tgm(xr), mapped))
            got = self.psos[(xr, mapped)] = {CODE[o]: m.pipeline(kernel_name(o)) for o in ops}
            ms = 1e3 * (time.perf_counter() - t0)
            how = "through a row map" if mapped else "where they lie"
            trace.event("megakernel: %d ops compiled for passes of up to %d rows %s in %.0f ms", len(ops), xr, how, ms)
        return got

    def _stored(self, mvs: Sequence[StoredMv]) -> None:
        """the kernels of the stored matvecs `mvs` not yet compiled, in one library (a pass width's, on its first
        pass: the step's launches compile on first use too)"""
        new = list(dict.fromkeys(s for s in mvs if s not in self.stored))
        if new:
            t0 = time.perf_counter()
            self.metal.compile(stored_source(new))
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

    def _kv_address(self, arr: mx_.array) -> int:
        """the address of a K/V array the pass binds, aliased the first time and let go when the array is"""
        got = self.kv.get(id(arr))
        if got is None:
            b = _alias(self.metal, arr)
            got = self.kv[id(arr)] = (b, self.metal.gpu_address(b))
            weakref.finalize(arr, Replay._drop_from, weakref.ref(self), id(arr))
        return got[1]

    @staticmethod
    def _drop_from(ref: weakref.ref[Replay], key: int) -> None:
        r = ref()
        if r is not None:
            r._drop(key)

    def _drop(self, key: int) -> None:
        """a K/V array gone: its alias, which no pass binds after (each rewrites the table first)"""
        with self.lock:
            got = self.kv.pop(key, None)
            if got is not None:
                self.metal.free_buffer(got[0])

    def _encode(self, T: int, splits: int, argmax: bool, mapped: bool) -> Shape:
        """the program of a pass shape as commands: a dispatch an instruction, its threadgroups its units (a GRID3 op's
        (kv heads, blocks, rows)), after a barrier where it reads an instruction encoded since the last one"""
        mp, m = self.mp, self.metal
        tab = mp.table(T, splits, argmax)
        n = int(tab.shape[0])
        tb = m.buffer(int(tab.nbytes))
        m.write(tb, tab)
        psos = self._pipelines(8 if T <= 8 else 16, mapped)
        mvs = [self._mv(f, T) for f in tab]
        self._stored([s for s in mvs if s is not None])
        t0 = time.perf_counter()
        order = [self.ptrs, self.freqs, self.ids, self.pos, self.meta, self.path, self.rmap, tb, self.dims, self.out]
        order.append(self.mi)
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
        kvs: Sequence[KvSlot],
        cap: int,
        rmap: np.ndarray | None,
        ids: Sequence[int],
        pos: Sequence[int],
        meta: mx_.array,
        path: mx_.array,
        splits: int,
        past: int,
        argmax: bool,
    ) -> np.ndarray:
        """the pass over the K and V rows `kvs` (each layer's K then V, `cap` rows a head), read and written through
        `rmap` where it is given, run and waited for: the argmax ids uint32 [T] (zeros without `argmax`)"""
        mp, m = self.mp, self.metal
        T = len(ids)
        mapped = rmap is not None
        if rmap is not None and len(rmap) > self.rows:
            raise ValueError(f"[mega] a row map of {len(rmap)} rows, a pass reads {self.rows}")
        pad = [0] * (mp.ROWS - T)
        meta_np = np.asarray(meta).astype(np.uint32)
        path_np = np.asarray(path).astype(np.uint32)
        if path_np.shape[1] < 8:
            path_np = np.pad(path_np, ((0, 0), (0, 8 - path_np.shape[1])))
        dims = np.zeros(DIMS, dtype=np.uint32)
        dims[:4] = (T, path_np.shape[1], cap, past)
        # the cache's rows and anything else MLX was writing, done before the replay reads them
        mx.synchronize()
        with self.lock:
            key = (T, splits, argmax, mapped)
            s = self.shapes.get(key)
            if s is None:
                s = self.shapes[key] = self._encode(T, splits, argmax, mapped)
            base = mp.nb + 2
            for j, slot in enumerate(kvs):
                self.addr[base + j] = self._kv_address(slot.arr) + int(slot.off)
            m.write(self.ptrs, self.addr)
            if rmap is not None:
                m.write(self.rmap, np.ascontiguousarray(rmap, dtype=np.uint32))
            m.write(self.ids, np.asarray([*ids, *pad], dtype=np.uint32))
            m.write(self.pos, np.asarray([*pos, *pad], dtype=np.uint32))
            m.write(self.meta, meta_np)
            m.write(self.path, path_np)
            m.write(self.dims, dims)
            m.run(s.icb, s.n)
            return m.read(self.out, T, np.uint32) if argmax else np.zeros(T, dtype=np.uint32)
