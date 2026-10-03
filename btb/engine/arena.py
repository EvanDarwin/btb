# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The arenas the card's kernels read a cache's rows from - the card graph's K and V, Qwen4's program's rows, raw and
pooled keys and rope tables - as row-major regions that grow at their ends and nowhere else (K and V position-major,
[rows, Hk, D]: a row's heads side by side).

Grown in place where the card's driver maps memory onto reserved addresses (CUDA's virtual memory management): each
region reserves addresses past the rows a sequence reaches and has physical memory mapped onto its end as the rows
need it - on the card, or pinned in RAM where the card reads it in place (`host`) - so nothing moves, nothing is
copied, nothing is held for rows not reached, and the kernels' pointers stand through every growth. Where the driver
cannot, each region is a buffer of its own, regrown one layer at a time - its rows copied over, its holders moved,
the old buffer given back before the next layer's is made: the arena and one layer's growth at once, never a second
whole arena.

Every view torch has of a region keeps it alive (as torch's own memory is, for a cache a caller keeps past the
engine's close); the arena's own hold ends with `close`."""

from __future__ import annotations

import contextlib
import ctypes
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from . import hostmem

_DRIVER = {"win32": "nvcuda.dll", "linux": "libcuda.so.1"}
_VMM_SUPPORTED = 102  # CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_DEVICE, _HOST = 1, 2  # CU_MEM_LOCATION_TYPE_DEVICE, _HOST


class _Loc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _Prop(ctypes.Structure):
    """CUmemAllocationProp: pinned memory on one card or in RAM, no shareable handle"""

    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _Loc),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _AllocFlags),
    ]


class _Access(ctypes.Structure):
    """CUmemAccessDesc"""

    _fields_ = [("location", _Loc), ("flags", ctypes.c_int)]


class _Driver:
    """the driver's virtual memory calls for one card, its memory made on the card or (`host`) pinned in RAM; `gran`
    the mapping granularity"""

    def __init__(self, device: int, host: bool = False) -> None:
        self.lib = ctypes.CDLL(_DRIVER[sys.platform])
        self.device, self.host = int(device), bool(host)
        v = ctypes.c_int()
        f = self.lib.cuDeviceGetAttribute
        f.restype = ctypes.c_int
        rc = f(ctypes.byref(v), ctypes.c_int(_VMM_SUPPORTED), ctypes.c_int(self.device))
        self.supported = rc == 0 and v.value == 1
        self.prop = _Prop()
        self.prop.type = 1  # CU_MEM_ALLOCATION_TYPE_PINNED
        self.prop.location.type = _HOST if self.host else _DEVICE
        self.prop.location.id = 0 if self.host else self.device
        # the card reads and writes every region; RAM's the host too
        n = 2 if self.host else 1
        self.access = (_Access * n)()
        self.access[0].location.type, self.access[0].location.id, self.access[0].flags = _DEVICE, self.device, 3
        if self.host:
            self.access[1].location.type, self.access[1].location.id, self.access[1].flags = _HOST, 0, 3
        self.gran = 0
        if self.supported:
            g = ctypes.c_size_t()
            f = self.lib.cuMemGetAllocationGranularity
            f.restype = ctypes.c_int
            if f(ctypes.byref(g), ctypes.byref(self.prop), ctypes.c_int(0)) == 0 and g.value:
                self.gran = int(g.value)
            else:
                self.supported = False  # no memory of this kind behind reserved addresses

    def call(self, name: str, *args: Any) -> None:
        f = getattr(self.lib, name)
        f.restype = ctypes.c_int
        rc = f(*args)
        if rc != 0:
            raise RuntimeError(f"[arena] {name} failed: CUresult {rc}")


def _driver(dev: torch.device, host: bool) -> _Driver | None:
    """the card's driver where it maps memory of this kind onto reserved addresses, else None (the fallback's
    layer-by-layer regrowth): a card without it, a build without CUDA's driver (an AMD card through ROCm), the CPU"""
    if dev.type != "cuda" or sys.platform not in _DRIVER or torch.version.cuda is None:
        return None
    try:
        d = _Driver(dev.index if dev.index is not None else torch.cuda.current_device(), host)
    except (OSError, AttributeError, RuntimeError):
        return None
    return d if d.supported else None


class _Chunk:
    """physical memory the driver made; given back once released and no region maps it"""

    def __init__(self, drv: _Driver, nbytes: int) -> None:
        h = ctypes.c_ulonglong()
        drv.call("cuMemCreate", ctypes.byref(h), ctypes.c_size_t(nbytes), ctypes.byref(drv.prop), ctypes.c_ulonglong(0))
        self.handle, self.nbytes = int(h.value), int(nbytes)
        weakref.finalize(self, drv.call, "cuMemRelease", ctypes.c_ulonglong(self.handle))


class _Region:
    """reserved addresses with physical chunks mapped onto their front, in order; unmapped and freed once nothing
    holds it - the arena, or a tensor over it (the array interface's, or a CPU view's ctypes array)"""

    def __init__(self, drv: _Driver, reserve: int) -> None:
        self.drv = drv
        p = ctypes.c_void_p()
        drv.call("cuMemAddressReserve", ctypes.byref(p), ctypes.c_size_t(reserve), ctypes.c_size_t(drv.gran),
                 ctypes.c_void_p(0), ctypes.c_ulonglong(0))  # fmt: skip
        self.ptr, self.reserve = int(p.value or 0), int(reserve)
        self.chunks: list[_Chunk] = []
        self.mapped = 0
        weakref.finalize(self, _Region._free, drv, self.ptr, self.reserve, self.chunks)

    def map(self, chunk: _Chunk) -> None:
        at = ctypes.c_void_p(self.ptr + self.mapped)
        self.drv.call("cuMemMap", at, ctypes.c_size_t(chunk.nbytes), ctypes.c_size_t(0),
                      ctypes.c_ulonglong(chunk.handle), ctypes.c_ulonglong(0))  # fmt: skip
        self.drv.call("cuMemSetAccess", at, ctypes.c_size_t(chunk.nbytes), self.drv.access,
                      ctypes.c_size_t(len(self.drv.access)))  # fmt: skip
        self.chunks.append(chunk)
        self.mapped += chunk.nbytes

    def tensor(self, dev: torch.device) -> torch.Tensor:
        """every reserved byte as one uint8 tensor (only bytes below the mapped front are ever touched): the card's
        through the array interface, or RAM's as a CPU view, which torch takes for pinned memory. Made once memory is
        mapped at the region's start (torch reads the pointer as it makes the card's)"""
        if self.drv.host:
            buf = (ctypes.c_uint8 * self.reserve).from_address(self.ptr)
            buf._region = self  # type: ignore[attr-defined]  # torch's view holds the array, the array the region
            return torch.frombuffer(buf, dtype=torch.uint8)
        return torch.as_tensor(_Cai(self), device=dev)

    @staticmethod
    def _free(drv: _Driver, ptr: int, reserve: int, chunks: list[_Chunk]) -> None:
        # the card's queued work done first: unmapping waits for nothing, and a copy out of these rows queued just
        # before the last view went (a cache detaching from a closed arena) read unmapped addresses (torch's own
        # cudaFree waits the same way). A card that already failed gives its addresses back all the same
        with contextlib.suppress(RuntimeError):
            torch.cuda.synchronize(drv.device)
        at = ptr
        for c in chunks:
            drv.call("cuMemUnmap", ctypes.c_void_p(at), ctypes.c_size_t(c.nbytes))
            at += c.nbytes
        chunks.clear()  # a chunk no other region maps is released with its last reference
        drv.call("cuMemAddressFree", ctypes.c_void_p(ptr), ctypes.c_size_t(reserve))


class _Cai:
    """a region's addresses as `__cuda_array_interface__`; the tensor torch makes of it holds it, and it the region"""

    def __init__(self, region: _Region) -> None:
        self.region = region
        self.__cuda_array_interface__ = {
            "shape": (region.reserve,),
            "typestr": "|u1",
            "data": (region.ptr, False),
            "version": 3,
        }


@dataclass(frozen=True)
class Rows:
    """a named run of regions: `count` of them (a layer each, or 1 shared), rows of `row` elements of `dtype`, and
    `cap // per + extra` rows for a cap (a pooled region holds a row per `per` positions)"""

    name: str
    count: int
    row: tuple[int, ...]
    dtype: torch.dtype = torch.bfloat16
    per: int = 1
    extra: int = 0

    @property
    def row_bytes(self) -> int:
        n = 1
        for d in self.row:
            n *= int(d)
        return n * torch.empty(0, dtype=self.dtype).element_size()

    def rows(self, cap: int) -> int:
        return int(cap) // self.per + self.extra


class RowArena:
    """Named row-major regions (`add`), `cap` positions of them usable, each grown at its end: `view(name, i)` is
    region i of `name` as [rows, *row] - on the card, or (`host`) pinned in RAM the card reads in place. `ceiling`:
    the positions a sequence of this model reaches, the addresses reserved up front where the driver maps in place
    (no memory behind them until the rows come)."""

    def __init__(self, dev: torch.device, host: bool = False, ceiling: int = 1) -> None:
        self.dev, self.host = dev, bool(host)
        self.cap = 0
        self.ceiling = max(int(ceiling), 1)
        self.drv = _driver(dev, self.host)
        self.specs: dict[str, Rows] = {}
        # in place: a region and a uint8 tensor over its reservation per region; the fallback: a buffer per region
        self.regions: dict[str, list[_Region]] = {}
        self.wholes: dict[str, list[torch.Tensor]] = {}
        self.bufs: dict[str, list[torch.Tensor]] = {}

    def add(self, name: str, count: int, row: tuple[int, ...], dtype: torch.dtype = torch.bfloat16, per: int = 1,
            extra: int = 0) -> None:  # fmt: skip
        """a run of `count` regions named `name`, made at the first growth"""
        assert not self.cap, "regions are added before the arena first grows"
        self.specs[name] = Rows(name, int(count), tuple(int(d) for d in row), dtype, max(1, int(per)), int(extra))

    @property
    def in_place(self) -> bool:
        """whether growth maps memory onto the rows' end (nothing moves) rather than regrowing layer by layer"""
        return self.drv is not None

    def nbytes(self, cap: int) -> int:
        """the bytes `cap` positions take in every region: whole chunks where the driver maps them"""
        return sum(s.count * self._whole(s.rows(cap) * s.row_bytes) for s in self.specs.values())

    def _whole(self, nbytes: int) -> int:
        g = self.drv.gran if self.drv is not None else 1
        return -(-int(nbytes) // g) * g

    def rows_for(self, cap: int) -> int:
        """the positions a growth to `cap` maps anyway: every region's whole chunks where it maps in place (a
        position past `cap` its mapping already holds is free to use - grow to this to have them), else `cap`"""
        if self.drv is None:
            return int(cap)
        usable = [((self._whole(s.rows(cap) * s.row_bytes) // s.row_bytes) - s.extra) * s.per for s in
                  self.specs.values()]  # fmt: skip
        return max(int(cap), min(usable))

    def grow(self, cap: int, moved: Callable[[int], None] | None = None) -> None:
        """`cap` positions usable in every region. In place each region's front is mapped on to them and
        nothing moves. Else layer by layer - every region of index j regrown, its rows copied into a buffer long
        enough, `moved(j)` told so its holders take the new views, and its old buffer given back - before index j + 1
        is regrown."""
        cap = int(cap)
        if cap <= self.cap:
            return
        if self.drv is not None:
            self._grow_in_place(cap)
            return
        old = self.cap
        for j in range(max(s.count for s in self.specs.values())):
            for s in self.specs.values():
                if j >= s.count:
                    continue
                bufs = self.bufs.setdefault(s.name, [])
                if self.host:
                    new = hostmem.pinned((s.rows(cap), *s.row), s.dtype)
                else:
                    new = torch.empty(s.rows(cap), *s.row, dtype=s.dtype, device=self.dev)
                if j < len(bufs):
                    n = s.rows(old)
                    new[:n].copy_(bufs[j][:n])
                    bufs[j] = new
                else:
                    bufs.append(new)
                del new
            self.cap = cap  # the views `moved` takes are the new buffers' whole length
            if moved is not None:
                moved(j)
            if not self.host and self.dev.type == "cuda":
                # the old buffers back to the card, not into torch's cache where they are kept beside the new ones
                torch.cuda.empty_cache()

    def _grow_in_place(self, cap: int) -> None:
        assert self.drv is not None
        need = {s.name: self._whole(s.rows(cap) * s.row_bytes) for s in self.specs.values()}
        if not self.regions or any(need[n] > rs[0].reserve for n, rs in self.regions.items()):
            # the first time, or past a reservation the rows outran (a batch's prompts end to end): every region's
            # addresses reserved anew, the chunks already made mapped onto them in order - the rows untouched, the
            # pointers moved (whoever holds the old ones keeps the old addresses alive till done)
            top = max(cap, self.ceiling)
            old = self.regions
            self.regions, self.wholes = {}, {}
            for s in self.specs.values():
                size = max(need[s.name], self._whole(s.rows(top) * s.row_bytes))
                rs = []
                for i in range(s.count):
                    r = _Region(self.drv, size)
                    for c in old[s.name][i].chunks if s.name in old else ():
                        r.map(c)
                    rs.append(r)
                self.regions[s.name] = rs
        for name, rs in self.regions.items():
            for r in rs:
                if need[name] > r.mapped:
                    r.map(_Chunk(self.drv, need[name] - r.mapped))
        if not self.wholes:
            self.wholes = {n: [r.tensor(self.dev) for r in rs] for n, rs in self.regions.items()}
        # the positions asked for, whatever else the chunks hold: arenas grown together agree on their views' lengths
        self.cap = cap

    def view(self, name: str, i: int = 0) -> torch.Tensor:
        """region i of `name`: its rows for the usable positions, [rows, *row]"""
        s = self.specs[name]
        n = s.rows(self.cap)
        if self.drv is None:
            return self.bufs[name][i][:n]
        rows = self.wholes[name][i][: n * s.row_bytes]
        return rows.view(s.dtype).view(n, *s.row)

    def data_ptr(self, name: str) -> int:
        """the first region of `name`'s rows: the arena's front, which a persisting-L2 window covers"""
        return int(self.view(name, 0).data_ptr()) if self.cap else 0

    def close(self) -> None:
        """the arena's own hold let go: what no cache still views goes back at once"""
        self.regions, self.wholes, self.bufs, self.cap = {}, {}, {}, 0


class Run:
    """the regions of one name in an arena as an indexable run: `run[i]` is region i's rows ([rows, *row])"""

    def __init__(self, arena: RowArena, name: str) -> None:
        self.arena, self.name = arena, name

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.arena.view(self.name, i)


class KvPair:
    """an arena's K and V runs as the [layers, 2, Hk, cap, D] index the caches take: `[j, w]` layer j's K (w 0) or V
    (w 1) as a [Hk, cap, D] view of its position-major rows, whose strides the kernels take"""

    def __init__(self, arena: RowArena, k: str = "k", v: str = "v") -> None:
        self.arena, self.names = arena, (k, v)

    def __getitem__(self, key: tuple[int, int]) -> torch.Tensor:
        j, w = key
        return self.arena.view(self.names[w], j).permute(1, 0, 2)


class KvArena(RowArena):
    """The card graph's arena: `n` layers' K and V, each position-major [rows, Hk, D] bf16. `[j, w]` is layer j's K
    (w 0) or V (w 1) as a [Hk, cap, D] view - the layout torch's caches index - whose strides (`stride(0)` = D,
    `stride(1)` = Hk * D) the kernels take."""

    def __init__(self, n: int, Hk: int, D: int, dev: torch.device, ceiling: int) -> None:
        super().__init__(dev, host=False, ceiling=ceiling)
        self.n = int(n)
        self.add("k", n, (Hk, D))
        self.add("v", n, (Hk, D))

    def __getitem__(self, key: tuple[int, int]) -> torch.Tensor:
        return KvPair(self)[key]

    def front(self) -> int:
        """the first layer's K rows: the arena's front, which the persisting-L2 window covers"""
        return self.data_ptr("k")
