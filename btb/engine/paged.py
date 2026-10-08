# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's KV rows in the pages of a pool every conversation on the engine shares (btb/engine/kvpool.py).

`KvPool` is the pages and where their rows lie. The layers the host runs keep theirs in the `HostRegion`: per layer a
K and a V region [Hk, rows, D] indexed by page id, head-major as the native kernels read a cache (head g's row r at
g * rows * D + r * D). On a card the layers the card runs keep theirs in the `CardRegion`: per layer an arena in VRAM,
position-major [slots * PAGE, Hk, D] as the card's kernels read it, a page at a slot of its own - the pages of the
conversation the card decodes (`bind`) - and every other page parked in pinned RAM until a conversation reading it is
bound again. A conversation's `Table` is its row map - position j of the sequence at row `rows()[j]` (page * PAGE +
offset) - and the pages it holds a reference on; a prefix two conversations share is the same rows, read in place by
both. `PagedCache` is the cache a session decodes over: its attention layers `PagedLayer`s over one table. Every reader
takes a layer's rows through the map - the host's native attention by row, the card's kernels through the card's row
map (`CardRegion.tbl`) - so a row is never copied to be read; a path that would read the layer as one contiguous buffer
meets a `PagedError` rather than a silent gather.
"""

from __future__ import annotations

import contextlib
import heapq
import threading
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from ..kinds import LayerKind
from .arena import KvArena, RowArena
from .kvpool import PAGE, Page, PagePool
from .scheduler import MemoryGrantError

if TYPE_CHECKING:
    from .prefix import PrefixCache
    from .radix import RadixTree

# the pages a move between the regions copies at once (a layer's rows going to the other region, pages parked or
# brought back in runs too long): what it stages beside the rows is bounded by it
MOVE_PAGES = 64

# the pool's own tensors - its regions, its maps - made as plain ones whatever mode the pass growing them runs in: an
# inference tensor refuses the writes made outside one (a tree's eviction freeing a page, a layer moving between the
# regions), and every method making one runs under this
_kept = torch.inference_mode(False)


class PagedError(RuntimeError):
    """a paged cache read as one contiguous buffer: a path the paged reader does not reach (yet)"""


def _runs(pairs: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """(from, to) slot pairs as runs (from, to, count) where both go up by one together, at most MOVE_PAGES long:
    one copy a run instead of one a page"""
    out: list[tuple[int, int, int]] = []
    for s, d in pairs:
        if out:
            s0, d0, n = out[-1]
            if s == s0 + n and d == d0 + n and n < MOVE_PAGES:
                out[-1] = (s0, d0, n + 1)
                continue
        out.append((s, d, 1))
    return out


class HostRegion:
    """the rows of the layers the host runs: K and V [Hk, rows, D] a layer (`hk` heads of `d`) by page id, in the rows'
    own dtype unless one is given, grown by GROW as the ledger grants it - a buffer at a time, as a layer cache grows
    its own, each let go once its rows are copied over"""

    GROW = 1.5
    MIN_PAGES = 16

    def __init__(
        self,
        layers: Sequence[int],
        hk: int,
        d: int,
        grant: Callable[..., None] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.layers = [int(i) for i in layers]
        self.hk, self.d = int(hk), int(d)
        self.grant = grant
        self.cap = 0  # the pages the regions hold rows for
        self.kv_dtype = dtype  # the rows' dtype where it is not their own (a bf16 host's)
        self.dtype: torch.dtype | None = None  # the regions' dtype, from the first rows written
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}

    def _el(self) -> int:
        return torch.empty(0, dtype=self.dtype or self.kv_dtype or torch.float32).element_size()

    def _target(self, pages: int) -> int:
        """the pages a growth to hold `pages` makes room for: GROW past what the regions hold"""
        return max(int(pages), int(self.cap * self.GROW), self.MIN_PAGES)

    def nbytes(self) -> int:
        """what the regions hold"""
        return 2 * len(self.k) * self.hk * self.cap * PAGE * self.d * self._el()

    def growth(self, pages: int) -> int:
        """the free memory a growth to hold `pages` pages takes at its peak: what the regions add, and the buffer a
        new one replaces, held until its rows are copied over (the last one's, every other let go by then); 0 where
        they hold them already"""
        if pages <= self.cap or not self.layers:
            return 0
        new = self._target(pages)
        row = self.hk * PAGE * self.d * self._el()
        return 2 * len(self.layers) * (new - self.cap) * row + self.cap * row

    def shape(self, k: torch.Tensor) -> None:
        """the rows' dtype from the first ones written, [B, Hk, T, D] - their heads and width the pool's"""
        if int(k.shape[1]) != self.hk or int(k.shape[-1]) != self.d:
            raise PagedError(
                f"rows of {int(k.shape[1])} heads of {int(k.shape[-1])} into a pool of {self.hk} of {self.d}"
            )
        if self.dtype is None:
            self.dtype = self.kv_dtype or k.dtype
            if self.cap:
                # pages made before the host's first rows (the card's layers wrote first): their regions made now
                for i in self.layers:
                    self._regrow(i, self.cap)

    def grow(self, pages: int) -> None:
        """the regions hold rows for `pages` pages: each layer's K and V regrown in turn as the ledger grants it -
        once the rows' dtype is known (`shape`), the pages counted till then"""
        if pages <= self.cap:
            return
        new = self._target(pages)
        if self.dtype is not None:
            for i in self.layers:
                self._regrow(i, new)
        self.cap = new

    @_kept
    def _regrow(self, i: int, new: int) -> None:
        rows, el = new * PAGE, self._el()
        for store in (self.k, self.v):
            old = store.get(i)
            if old is not None and old.shape[1] >= rows:
                continue  # grown by an attempt the ledger refused part way
            if self.grant is not None:
                # free RAM alone: the expert store is a cache too, and the conversations least recently used go
                # before its blocks do (`KvPool.alloc`, `PrefixCache.room`)
                self.grant(
                    self.hk * rows * self.d * el,
                    "kv",
                    requester=f"the prefix cache's layer {i}, {new} pages of {PAGE} rows",
                    device="cpu",
                    held=old.numel() * el if old is not None else 0,
                )
            assert self.dtype is not None
            buf = torch.empty(self.hk, rows, self.d, dtype=self.dtype)
            if old is not None:
                buf[:, : old.shape[1]] = old
            store[i] = buf

    def add(self, i: int) -> None:
        """layer i's rows held here from now on (it left the card): its regions made at the pages held, granted"""
        if i in self.layers:
            return
        if self.dtype is None:
            # the card's rows widened as the host's layers keep theirs (`host_kv_dtype`): bf16's, or float32
            self.dtype = self.kv_dtype or torch.float32
        if self.cap:
            self._regrow(i, self.cap)
        self.layers.append(int(i))

    def drop(self, i: int) -> None:
        """layer i's rows held elsewhere from now on (it came onto the card): its regions let go"""
        if i in self.layers:
            self.layers.remove(i)
        self.k.pop(i, None)
        self.v.pop(i, None)

    def write(self, i: int, rows: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """layer i's `k`, `v` [Hk, T, D] at `rows` (T of them, int64 on the host)"""
        self.k[i].index_copy_(1, rows, k.to("cpu", self.dtype))
        self.v[i].index_copy_(1, rows, v.to("cpu", self.dtype))

    def view(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """layer i's K and V regions [1, Hk, rows, D], read through a table's rows"""
        return self.k[i][None], self.v[i][None]

    def gather(self, i: int, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """layer i's `rows` in order as [1, Hk, n, D] tensors of their own: a copy"""
        return self.k[i][:, rows][None], self.v[i][:, rows][None]

    def move(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """every layer's rows at `src` copied to `dst` (the source read whole first, so the two may overlap)"""
        for store in (self.k, self.v):
            for buf in store.values():
                buf[:, dst] = buf[:, src]

    def close(self) -> None:
        """the regions let go with the engine: nothing reads them after"""
        self.k.clear()
        self.v.clear()
        self.cap = 0


class CardRegion:
    """The rows of the layers the card runs. On the card: per layer an arena (`KvArena` of one layer: K and V
    position-major [rows, Hk, D], grown in place where the driver maps memory, so the kernels' pointers stand), a
    page's rows at `slot * PAGE ..` of every layer's - the pages of the conversation the card decodes (`bind`), the
    rest parked as another is bound. In pinned RAM: the park, the same layout, a page at a park slot of its own until
    a conversation reading it is bound again. `tbl` is the bound conversation's row map on the card: position j at card
    row `tbl[j]`, as the card's kernels read it. `version` moves with every page that changes slots (the map is made
    again); `layout` with every move of the arenas' addresses, their length or the map's buffer (what a graph captured
    over them is made again)"""

    GROW = 1.5
    MIN_SLOTS = 16

    @_kept
    def __init__(
        self,
        layers: Sequence[int],
        hk: int,
        d: int,
        dev: torch.device,
        grant: Callable[..., None] | None = None,
        ceiling: int = 1,
        lock: Any = None,
    ) -> None:
        self.hk, self.d, self.dev = int(hk), int(d), torch.device(dev)
        self.grant = grant
        # the page pool's: a conversation's table let go on another thread frees its slots under it (`freed`)
        self.lock = lock if lock is not None else threading.RLock()
        self.ceiling = max(int(ceiling), PAGE)
        self.dtype = torch.bfloat16
        self.arenas: dict[int, KvArena] = {int(i): self._arena() for i in layers}
        self.parks: dict[int, RowArena] = {}
        self.slots: list[Page | None] = []  # card slot -> the page there
        self.free: list[int] = []  # the free card slots, a heap: the lowest first, so a conversation's pages run
        self.parked: list[Page | None] = []  # park slot -> the page there
        self.pfree: list[int] = []
        self.slot_of = torch.full((0,), -1, dtype=torch.int64)  # page id -> its card slot, -1 off the card
        self.version = 0
        self.layout = 0
        self.loaded = 0  # the pages brought back from the park with rows to bring, all told
        self.closed = False
        self.tbl: torch.Tensor | None = None
        self._bound: weakref.ref[Table] | None = None
        self._mapped = (-1, 0)  # the region's version and the positions the map holds for the bound table
        # the pages let go while the slots were moving (`_moving`), their slots freed once the move is done
        self._moving = 0
        self._let_go: list[Page] = []

    @contextlib.contextmanager
    def _move(self) -> Iterator[None]:
        """the slots changing - pages parked, brought back, placed - under the lock: a page let go meanwhile has its
        slots freed once the change is done. The lock is reentrant, and a conversation's table collected mid-change
        (the collector run by an allocation of the copies) lets its pages go on this very thread: a page among those
        moving, freed then, read its slot as gone - the move then wrote slot -1, the region's last"""
        with self.lock:
            self._moving += 1
            try:
                yield
            finally:
                self._moving -= 1
                while not self._moving and self._let_go:
                    self._free_slots(self._let_go.pop())

    def _arena(self) -> KvArena:
        return KvArena(1, self.hk, self.d, self.dev, self.ceiling)

    def _park_arena(self) -> RowArena:
        """a layer's park: K and V position-major, pinned in RAM where the card is one (its copies there run on the
        card's stream), grown at its end"""
        a = RowArena(self.dev, host=self.dev.type == "cuda", ceiling=self.ceiling)
        a.add("k", 1, (self.hk, self.d))
        a.add("v", 1, (self.hk, self.d))
        return a

    @property
    def layers(self) -> list[int]:
        return list(self.arenas)

    @property
    def cap(self) -> int:
        """the card slots the arenas hold rows for"""
        return len(self.slots)

    def _page_bytes(self) -> int:
        """one page's K and V of one layer"""
        return 2 * PAGE * self.hk * self.d * torch.empty(0, dtype=self.dtype).element_size()

    def nbytes(self) -> int:
        """what the arenas hold on the card"""
        return len(self.arenas) * self.layer_bytes()

    def layer_bytes(self) -> int:
        """what one layer's arena holds on the card: a layer leaving it frees that, one coming takes it"""
        return self.cap * self._page_bytes()

    def park_nbytes(self) -> int:
        """what the park holds in RAM"""
        return len(self.arenas) * len(self.parked) * self._page_bytes()

    def _target(self, have: int, slots: int) -> int:
        return max(int(slots), int(have * self.GROW), self.MIN_SLOTS)

    def growth(self, slots: int) -> int:
        """the card memory a growth to `slots` slots takes: what the arenas add (in place nothing moves; regrown, a
        layer's buffer at a time beside its predecessor); 0 where they hold them already"""
        if slots <= self.cap or not self.arenas:
            return 0
        new = self._target(self.cap, slots)
        a = next(iter(self.arenas.values()))
        extra = 0 if a.in_place else self.cap * self._page_bytes()
        return len(self.arenas) * (new - self.cap) * self._page_bytes() + extra

    def park_growth(self, slots: int) -> int:
        """the RAM a growth of the park to `slots` slots takes; 0 where it holds them already"""
        if slots <= len(self.parked) or not self.arenas:
            return 0
        return len(self.arenas) * (self._target(len(self.parked), slots) - len(self.parked)) * self._page_bytes()

    def need(self, table: Table, new: int) -> tuple[int, int]:
        """what binding `table` and `new` more pages of it take past what the region holds: (card slots, park
        slots) - its pages off the card brought in and its new ones placed, less the slots its binding parks (the
        pages there it does not read), which take park slots of their own"""
        held = table.held
        bound = self._bound() if self._bound is not None else None
        off = sum(1 for p in held.values() if p.slot < 0)
        others = 0 if bound is table else sum(1 for p in self.slots if p is not None and p.id not in held)
        back = sum(1 for p in held.values() if p.park >= 0)
        slots = max(0, off + int(new) - others - len(self.free))
        parks = max(0, others - back - len(self.pfree))
        return slots, parks

    @_kept
    def _grow(self, slots: int) -> None:
        if slots <= self.cap:
            return
        new = self._target(self.cap, slots)
        if self.grant is not None and self.arenas:
            self.grant(
                len(self.arenas) * new * self._page_bytes(),
                "kv",
                requester=f"the prefix cache's card arenas, {len(self.arenas)} layers x {new} pages of {PAGE} rows",
                device=self.dev,
                held=self.nbytes(),
            )
        for a in self.arenas.values():
            a.grow(new * PAGE)
        for s in range(len(self.slots), new):
            heapq.heappush(self.free, s)
        self.slots.extend([None] * (new - len(self.slots)))
        self.layout += 1  # the arenas longer, perhaps at new addresses

    @_kept
    def _grow_park(self, slots: int) -> None:
        if slots <= len(self.parked):
            return
        new = self._target(len(self.parked), slots)
        if self.grant is not None and self.arenas:
            self.grant(
                len(self.arenas) * new * self._page_bytes(),
                "kv",
                requester=f"the prefix cache's park, {len(self.arenas)} layers x {new} pages of {PAGE} rows",
                device="cpu",
                held=self.park_nbytes(),
            )
        for i in self.arenas:
            p = self.parks.get(i)
            if p is None:
                p = self.parks[i] = self._park_arena()
            p.grow(new * PAGE)
        for s in range(len(self.parked), new):
            heapq.heappush(self.pfree, s)
        self.parked.extend([None] * (new - len(self.parked)))

    @_kept
    def _set_slot(self, p: Page, s: int) -> None:
        if p.id >= len(self.slot_of):
            grown = torch.full((max(p.id + 1, 2 * len(self.slot_of), 256),), -1, dtype=torch.int64)
            grown[: len(self.slot_of)] = self.slot_of
            self.slot_of = grown
        self.slot_of[p.id] = s
        p.slot = s

    def _take(self) -> int:
        if not self.free:
            self._grow(self.cap + 1)
        return heapq.heappop(self.free)

    def _take_park(self) -> int:
        if not self.pfree:
            self._grow_park(len(self.parked) + 1)
        return heapq.heappop(self.pfree)

    def place(self, p: Page) -> None:
        """a new page `p` on the card, at the lowest free slot or one the arenas grow for"""
        with self._move():
            s = self._take()
            self.slots[s] = p
            self._set_slot(p, s)
            p.where = "card"

    def freed(self, p: Page) -> None:
        """page `p` has no holder left: its slot on the card or in the park free again - once the slots moving now
        are moved (`_move`) - and nothing to free once the region is closed, a table outliving the engine letting its
        pages go"""
        with self.lock:
            if self._moving and not self.closed:
                self._let_go.append(p)
                return
            self._free_slots(p)

    def _free_slots(self, p: Page) -> None:
        with self.lock:
            if self.closed:
                p.slot = p.park = -1
                return
            if p.slot >= 0:
                self.slots[p.slot] = None
                heapq.heappush(self.free, p.slot)
                self._set_slot(p, -1)
            if p.park >= 0:
                self.parked[p.park] = None
                heapq.heappush(self.pfree, p.park)
                p.park = -1
            p.where = "host"

    def _copy(self, src: dict[int, Any], dst: dict[int, Any], runs: list[tuple[int, int, int]]) -> None:
        """every layer's K and V rows of each run (from slot, to slot, pages) copied from `src`'s arenas to `dst`'s:
        the card's and the park's, either way, on the card's stream"""
        for i in self.arenas:
            for w in ("k", "v"):
                a, b = src[i].view(w, 0), dst[i].view(w, 0)
                for s, d, n in runs:
                    b[d * PAGE : (d + n) * PAGE].copy_(a[s * PAGE : (s + n) * PAGE], non_blocking=True)

    def park(self, pages: Sequence[Page]) -> None:
        """`pages` off the card into the park, their card slots free again"""
        with self._move():
            pages = sorted(pages, key=lambda p: p.slot)
            if not pages:
                return
            # the park grown for them all before a slot is taken: a growth refused leaves every page where it was
            self._grow_park(len(self.parked) + max(0, len(pages) - len(self.pfree)))
            pairs = []
            for p in pages:
                ps = self._take_park()
                pairs.append((p.slot, ps))
                self.parked[ps] = p
            self._copy(self.arenas, self.parks, _runs(pairs))
            for p, (s, ps) in zip(pages, pairs, strict=True):
                self.slots[s] = None
                heapq.heappush(self.free, s)
                self._set_slot(p, -1)
                p.park, p.where = ps, "park"
            self.version += 1

    def load(self, pages: Sequence[Page]) -> None:
        """`pages` out of the park onto the card, their park slots free again"""
        with self._move():
            pages = sorted(pages, key=lambda p: p.park)
            if not pages:
                return
            # the arenas grown for them all before a slot is taken: a growth refused leaves every page where it was
            self._grow(self.cap + max(0, len(pages) - len(self.free)))
            pairs = []
            for p in pages:
                s = self._take()
                pairs.append((p.park, s))
                self.slots[s] = p
            self._copy(self.parks, self.arenas, _runs(pairs))
            for p, (ps, s) in zip(pages, pairs, strict=True):
                self.parked[ps] = None
                heapq.heappush(self.pfree, ps)
                self._set_slot(p, s)
                p.park, p.where = -1, "card"
            self.version += 1
            if self.arenas:
                self.loaded += len(pages)

    def bind(self, table: Table) -> torch.Tensor:
        """`table`'s pages on the card - those in the park brought back, the pages on the card it does not read
        parked first - and its row map on the card (`tbl`, int32: position j at card row tbl[j]) brought up to date,
        only its new positions uploaded while it stays the one bound"""
        with self._move():
            bound = self._bound() if self._bound is not None else None
            tbl = self.tbl
            if bound is not table:
                held = table.held
                # none bound until this one is: a load refused after the park leaves the table bound before with its
                # pages parked, and its next bind must bring them back, not read its map as current
                self._bound = None
                self.park([p for p in self.slots if p is not None and p.id not in held])
                self.load([p for p in held.values() if p.park >= 0])
                for p in held.values():
                    if p.slot < 0:
                        self.place(p)
            elif (
                table.low >= len(table)
                and self._mapped[0] == self.version
                and tbl is not None
                and len(tbl) >= self.cap * PAGE
            ):
                return tbl
            if tbl is None or len(tbl) < self.cap * PAGE:
                with torch.inference_mode(False):  # the pool's own, as `_kept`'s are
                    tbl = self.tbl = torch.zeros(max(self.cap * PAGE, PAGE), dtype=torch.int32, device=self.dev)
                self.layout += 1
                self._mapped = (-1, 0)
            n = len(table)
            lo = min(table.low, self._mapped[1]) if bound is table and self._mapped[0] == self.version else 0
            if n > lo:
                tbl[lo:n].copy_(self.rows(table.rows()[lo:n]).to(torch.int32), non_blocking=True)
            table.low = n
            self._bound = weakref.ref(table)
            self._mapped = (self.version, n)
            return tbl

    def rows(self, rows: torch.Tensor) -> torch.Tensor:
        """page rows (page * PAGE + offset, int64 on the host) as card rows (slot * PAGE + offset): their pages must
        be on the card"""
        s = self.slot_of[rows // PAGE] if len(self.slot_of) else torch.full_like(rows, -1)
        if bool((s < 0).any()):
            raise PagedError("a page read on the card is not on it: its conversation was not bound")
        return s * PAGE + rows % PAGE

    def view(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """layer i's K and V arenas as [1, Hk, rows, D] views (position-major: a head's row r at r * Hk * D), read
        through the card's row map"""
        a = self.arenas[i]
        return a[0, 0][None], a[0, 1][None]

    def write(self, i: int, at: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """layer i's `k`, `v` [Hk, T, D] at card rows `at` (T of them, the bound map's, on the card)"""
        a, idx = self.arenas[i], at.to(torch.int64)
        a.view("k", 0).index_copy_(0, idx, k.transpose(0, 1).to(self.dev, self.dtype))
        a.view("v", 0).index_copy_(0, idx, v.transpose(0, 1).to(self.dev, self.dtype))

    def gather(self, i: int, rows: torch.Tensor, pages: Sequence[Page]) -> tuple[torch.Tensor, torch.Tensor]:
        """layer i's page rows `rows` in order as [1, Hk, n, D] tensors of their own on the card, wherever their
        pages (`pages`, the pool's by id) lie - the card, or the park: a copy"""
        out = []
        for w in ("k", "v"):
            got = torch.empty(len(rows), self.hk, self.d, dtype=self.dtype, device=self.dev)
            for a, b in self._spans(rows):
                p, off = pages[int(rows[a]) // PAGE], int(rows[a]) % PAGE
                if p.slot >= 0:
                    got[a:b] = self.arenas[i].view(w, 0)[p.slot * PAGE + off : p.slot * PAGE + off + b - a]
                elif p.park >= 0:
                    got[a:b] = self.parks[i].view(w, 0)[p.park * PAGE + off : p.park * PAGE + off + b - a]
                else:
                    raise PagedError(f"page {p.id} has no rows on the card or in the park")
            out.append(got.transpose(0, 1)[None])
        return out[0], out[1]

    @staticmethod
    def _spans(rows: torch.Tensor) -> list[tuple[int, int]]:
        """`rows` cut where they leave a page or skip a row: each span one page's consecutive rows"""
        r = rows.tolist()
        out: list[tuple[int, int]] = []
        a = 0
        for j in range(1, len(r) + 1):
            if j == len(r) or r[j] != r[j - 1] + 1 or r[j] // PAGE != r[a] // PAGE:
                out.append((a, j))
                a = j
        return out

    def move(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """every layer's page rows at `src` copied to `dst` on the card (the source read whole first)"""
        cs, cd = self.rows(src).to(self.dev), self.rows(dst).to(self.dev)
        for a in self.arenas.values():
            for w in ("k", "v"):
                buf = a.view(w, 0)
                buf[cd] = buf[cs]

    @_kept
    def add(self, i: int, host: HostRegion, live: Sequence[Page]) -> None:
        """layer i's rows held here from now on (it came onto the card): its arena and park made at the slots the
        region holds, granted, and every live page's rows of it copied in from the host's region"""
        if i in self.arenas:
            return
        pb = self._page_bytes()
        if self.grant is not None:
            self.grant(self.cap * pb, "kv", requester=f"the prefix cache's card arena for layer {i}", device=self.dev)
            if self.parked:
                who = f"the prefix cache's park for layer {i}"
                self.grant(len(self.parked) * pb, "kv", requester=who, device="cpu")
        a = self.arenas[i] = self._arena()
        if self.cap:
            a.grow(self.cap * PAGE)
        if self.parked:
            pk = self.parks[i] = self._park_arena()
            pk.grow(len(self.parked) * PAGE)
        self.layout += 1
        if i not in host.k:
            return
        for w, src in (("k", host.k[i]), ("v", host.v[i])):
            for p in live:
                rows = src[:, p.id * PAGE : (p.id + 1) * PAGE].transpose(0, 1)
                if p.slot >= 0:
                    a.view(w, 0)[p.slot * PAGE : (p.slot + 1) * PAGE].copy_(rows)
                elif p.park >= 0:
                    self.parks[i].view(w, 0)[p.park * PAGE : (p.park + 1) * PAGE].copy_(rows)

    def drop(self, i: int, host: HostRegion, live: Sequence[Page]) -> None:
        """layer i's rows held on the host from now on (it left the card): every live page's rows of it copied to
        the host's region (made for it there first), its arena and park let go"""
        if i not in self.arenas:
            return
        host.add(i)
        if i in host.k:
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)  # the park's copies in flight landed before the host reads it
            a, pk = self.arenas[i], self.parks.get(i)
            for w, dst in (("k", host.k[i]), ("v", host.v[i])):
                for p in live:
                    if p.slot >= 0:
                        rows = a.view(w, 0)[p.slot * PAGE : (p.slot + 1) * PAGE]
                    elif p.park >= 0 and pk is not None:
                        rows = pk.view(w, 0)[p.park * PAGE : (p.park + 1) * PAGE]
                    else:
                        continue
                    dst[:, p.id * PAGE : (p.id + 1) * PAGE].copy_(rows.transpose(0, 1))
        self.arenas.pop(i).close()
        pk = self.parks.pop(i, None)
        if pk is not None:
            pk.close()
        self.layout += 1

    @_kept
    def close(self) -> None:
        """the arenas and the park let go with the engine"""
        for a in (*self.arenas.values(), *self.parks.values()):
            a.close()
        self.arenas.clear()
        self.parks.clear()
        self.slots, self.free, self.parked, self.pfree = [], [], [], []
        self.slot_of = torch.full((0,), -1, dtype=torch.int64)
        self.tbl = None
        self._bound = None
        self.closed = True


class KvPool:
    """The pages every conversation's rows lie in, and where: the host's layers' rows in the `HostRegion` by page id,
    the card's layers' in the `CardRegion` (on the card, or parked) by slot. A page is allocated for a writer, the host
    region grown for it as the ledger grants; refused, the conversations the tree holds let go, least recently used
    first, their freed pages taking the rows"""

    def __init__(
        self,
        host_layers: Sequence[int],
        hk: int,
        d: int,
        grant: Callable[..., None] | None = None,
        dtype: torch.dtype | None = None,
        card_layers: Sequence[int] = (),
        dev: torch.device | None = None,
        ceiling: int = 1,
    ) -> None:
        self.hk, self.d = int(hk), int(d)
        self.pages = PagePool(grow=self._grow, freed=self._freed)
        self.host = HostRegion(host_layers, hk, d, grant, dtype)
        # a card's region whether or not it runs a layer now: the layers coming onto it later find their pages placed
        self.card: CardRegion | None = None
        if dev is not None and torch.device(dev).type == "cuda":
            self.card = CardRegion(card_layers, hk, d, torch.device(dev), grant, ceiling, self.pages.lock)
        self.tree: RadixTree | None = None

    def _grow(self, pages: int) -> None:
        self.host.grow(pages)

    def _freed(self, p: Page) -> None:
        if self.card is not None:
            self.card.freed(p)

    def on_card(self, i: int) -> bool:
        return self.card is not None and i in self.card.arenas

    def alloc(self, writer: object) -> Page:
        """a page for `writer` to append to - placed on the card where there is one - a free one, else the host's
        region grown; refused, conversations the tree holds let go, least recently used first, until a page frees.
        The card's room is the pass's to have made (`cache_room`): a slot it cannot grow for is refused as it is"""
        while True:
            try:
                p = self.pages.alloc(writer)
                break
            except MemoryGrantError:
                if self.tree is None or not self.tree.evict(enough=lambda freed: freed >= 1):
                    raise
        if self.card is not None:
            try:
                self.card.place(p)
            except BaseException:
                self.pages.unref(p)
                raise
        return p

    def shape(self, i: int, k: torch.Tensor) -> None:
        if self.on_card(i):
            if int(k.shape[1]) != self.hk or int(k.shape[-1]) != self.d:
                raise PagedError(f"rows of {int(k.shape[1])} heads of {int(k.shape[-1])} into a pool of {self.hk}")
            return
        self.host.shape(k)

    def move(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """every layer's rows at page rows `src` copied to `dst`, on the host and on the card"""
        self.host.move(src, dst)
        if self.card is not None and self.card.arenas:
            self.card.move(src, dst)

    def rehome(self, i: int, card: bool) -> None:
        """layer i's rows moved to the region where it now runs - the card's, or the host's - every conversation's
        at once: the pages are the pool's, so a layer's placement moves them once for all"""
        if self.card is None or card == self.on_card(i):
            return
        live = [p for p in self.pages.pages if p.refs > 0]
        if card:
            self.card.add(i, self.host, live)
            self.host.drop(i)
        else:
            self.card.drop(i, self.host, live)

    def nbytes(self) -> dict[str, int]:
        """what the regions hold, by where: the host's and the park's in RAM, the card's arenas on the card"""
        out = {"cpu": self.host.nbytes()}
        if self.card is not None:
            out["cpu"] += self.card.park_nbytes()
            out[str(self.card.dev)] = self.card.nbytes()
        return out

    def close(self) -> None:
        self.host.close()
        if self.card is not None:
            self.card.close()


class Table:
    """A conversation's rows: `rows()[j]` is the pool row holding its position j (page * PAGE + offset), in every
    layer. It holds a reference on each page it reads (`count` its rows there), and appends only to a page it is the
    writer of, past the rows another holder froze (kvpool's rules); a prefix taken from the tree is read in place, the
    conversation's own rows going on in pages of its own. Let go (`release`, or collected), its references go with
    it. `low` is the first position whose row changed since the card last mapped it (`CardRegion.bind`)."""

    def __init__(self, pool: KvPool, rows: Sequence[int] = ()) -> None:
        self.pool = pool
        self.token = object()  # the pages' writer: not the table, which no page may keep alive
        self.n = 0
        self.low = 0
        self.version = 0  # moved by every change to the map: what is built from it (a pass's row lists) is keyed by it
        self._buf = torch.empty(0, dtype=torch.int64)
        self.held: dict[int, Page] = {}
        self.count: dict[int, int] = {}
        self.tail: Page | None = None
        self._fin = weakref.finalize(self, Table._let_go, pool.pages, self.held, self.count)
        self._take(rows)

    @staticmethod
    def _let_go(pages: PagePool, held: dict[int, Page], count: dict[int, int]) -> None:
        for p in list(held.values()):
            pages.unref(p)
        held.clear()
        count.clear()

    def release(self) -> None:
        """every page let go: the table reads nothing after (and what it holds again it lets go when collected)"""
        self._fin()
        self._fin = weakref.finalize(self, Table._let_go, self.pool.pages, self.held, self.count)
        self.n, self.tail, self.low = 0, None, 0
        self.version += 1

    def __len__(self) -> int:
        return self.n

    def rows(self) -> torch.Tensor:
        """the row map: position j's pool row, int64, a view the next append may move"""
        return self._buf[: self.n]

    @_kept
    def _reserve(self, n: int) -> None:
        if n > len(self._buf):
            buf = torch.empty(max(n, 2 * len(self._buf), 256), dtype=torch.int64)
            buf[: self.n] = self._buf[: self.n]
            self._buf = buf

    def _take(self, rows: Sequence[int]) -> None:
        """`rows` read from another holder (the tree) appended: a hold on each page they lie in"""
        rows = [int(r) for r in rows]
        if not rows:
            return
        for r in rows:
            pid = r // PAGE
            if pid not in self.held:
                p = self.pool.pages.pages[pid]
                self.pool.pages.ref(p)
                self.held[pid] = p
            self.count[pid] = self.count.get(pid, 0) + 1
        self._reserve(self.n + len(rows))
        self._buf[self.n : self.n + len(rows)] = torch.tensor(rows, dtype=torch.int64)
        self.low = min(self.low, self.n)
        self.n += len(rows)
        self.version += 1

    def _own_tail(self) -> Page | None:
        """the page the table appends to: its tail, while it is the writer there and the page has room"""
        p = self.tail
        if p is None or p.writer is not self.token or p.fill >= PAGE or p.id not in self.held:
            return None
        return p

    def pages_for(self, T: int) -> int:
        """the new pages T more rows take: what the tail's room does not hold"""
        p = self._own_tail()
        past = int(T) - (PAGE - p.fill if p is not None else 0)
        return max(0, -(-past // PAGE))

    def extend(self, n: int) -> None:
        """rows for positions up to `n`: on the table's own tail page while it has room, else on new pages"""
        if n <= self.n:
            return
        self.version += 1
        self.low = min(self.low, self.n)
        self._reserve(n)
        while self.n < n:
            p = self._own_tail()
            if p is None:
                p = self.pool.alloc(self.token)
                self.held[p.id] = p
                self.tail = p
            take = min(PAGE - p.fill, n - self.n)
            first = p.id * PAGE + p.fill
            self._buf[self.n : self.n + take] = torch.arange(first, first + take, dtype=torch.int64)
            self.count[p.id] = self.count.get(p.id, 0) + take
            p.fill += take
            self.n += take

    def crop(self, n: int) -> None:
        """positions past `n` let go: a page the table no longer reads is unheld, and its own tail page takes its
        rows back where no other holder froze them (else it writes there no more)"""
        n = max(0, int(n))
        if n >= self.n:
            return
        self.version += 1
        self.low = min(self.low, n)
        gone = self._buf[n : self.n].tolist()
        self.n = n
        for r in gone:
            pid = r // PAGE
            c = self.count[pid] - 1
            if c:
                self.count[pid] = c
                continue
            del self.count[pid]
            self.pool.pages.unref(self.held.pop(pid))
        t = self.tail
        if t is None:
            return
        if t.id not in self.held:
            self.tail = None
            return
        last = int(self._buf[n - 1]) if n else -1
        end = last % PAGE + 1 if last >= 0 and last // PAGE == t.id else 0
        if t.writer is self.token and t.frozen <= end:
            t.fill = end
        else:
            self.tail = None

    def path_rows(self, base: int, path: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """a verify pass's accepted path (`path` its nodes, the pass's rows from `base`): the rows to copy where the
        path's positions are, (from, to), None where they are there already"""
        if list(path) == list(range(len(path))):
            return None
        src = self._buf[torch.tensor([base + j for j in path], dtype=torch.int64)].clone()
        dst = self._buf[base : base + len(path)].clone()
        return src, dst


class PagedKV:
    """a card layer's rows as its module hands them to its attention (`btb_sdpa`,
    families/attention.py): the layer - its arenas read through the card's row map - and the pass's place, its `T`
    rows after `n0`. Not a tensor: a reader that would take it for one fails, rather than read the arena's slots as
    the sequence's positions"""

    __slots__ = ("T", "layer", "n0")

    def __init__(self, layer: PagedLayer, n0: int, T: int) -> None:
        self.layer, self.n0, self.T = layer, int(n0), int(T)


class PagedLayer(DynamicLayer):
    """one attention layer of a `PagedCache`: its rows the table's, written into its region of the pool (the host's,
    or the card's where the card runs the layer). `append` is the paged readers' (the region and the table's map
    back); `update` serves a card layer's rows as a `PagedKV` the engine's attention reads through the map, and a
    host layer's first rows of a fresh prompt to its module, whose attention then reads them as they came - a host
    layer's module past its first rows is a `PagedError`, as `keys`/`values` are: `gather()` is the explicit copy (a
    fork's prefix, `Session.rows`). `hop` holds a host layer's rows on the card for a prefill's chunks there, `land`
    puts the chunks' rows back"""

    paged = True

    def __init__(self, table: Table, pool: KvPool, i: int) -> None:
        super().__init__()
        self.table, self.pool, self.i = table, pool, int(i)
        self.n = 0
        self.is_initialized = True
        self._hop: tuple[torch.Tensor, torch.Tensor, int] | None = None

    @property
    def keys(self) -> torch.Tensor:
        raise PagedError(f"layer {self.i}'s rows are paged: read them through the row map, or gather() a copy")

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise PagedError(f"layer {self.i}'s rows are paged: written by append, not assigned")

    @property
    def values(self) -> torch.Tensor:
        raise PagedError(f"layer {self.i}'s rows are paged: read them through the row map, or gather() a copy")

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise PagedError(f"layer {self.i}'s rows are paged: written by append, not assigned")

    @property
    def on_card(self) -> bool:
        """whether the card runs the layer: its rows in the card's arenas, read through the card's row map"""
        return self.pool.on_card(self.i)

    def get_seq_length(self) -> int:
        return self.n

    def set_front(self, n: int) -> None:
        """the card wrote the layer's rows up to `n` (a graph's replay, through the card's row map)"""
        self.n = int(n)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`k`, `v` [1, Hk, T, D] after the layer's rows: their rows on the table (made by whichever layer reaches
        them first), the region's K and V [1, Hk, rows, D] back, read through the table's rows (the card's map of
        them where the card runs the layer)"""
        if int(k.shape[0]) != 1:
            raise PagedError(f"a paged cache holds one sequence, not {int(k.shape[0])}")
        T = int(k.shape[-2])
        self.pool.shape(self.i, k)
        self.table.extend(self.n + T)
        if self._hop is not None:
            kb, vb, _ = self._hop
            kb[0, :, self.n : self.n + T].copy_(k[0])
            vb[0, :, self.n : self.n + T].copy_(v[0])
            self.n += T
            return kb[..., : self.n, :], vb[..., : self.n, :]
        if self.on_card:
            card = self.pool.card
            assert card is not None
            tbl = card.bind(self.table)
            card.write(self.i, tbl[self.n : self.n + T], k[0], v[0])
            self.n += T
            return card.view(self.i)
        host = self.pool.host
        host.write(self.i, self.table.rows()[self.n : self.n + T], k[0], v[0])
        self.n += T
        return host.view(self.i)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        first, T = self.n == 0, int(key_states.shape[-2])
        got = self.append(key_states, value_states)
        if self._hop is not None:
            return got
        if self.on_card:
            # the first rows too: read through the map by the kernels a contiguous cache's rows take, never the
            # module's own rows (a fused projection's slices, strided apart, sent them to sdpa and its bits)
            kv = PagedKV(self, self.n - T, T)
            return kv, kv
        if first:
            return key_states.to(self.pool.host.dtype), value_states.to(self.pool.host.dtype)
        raise PagedError(
            f"layer {self.i}: a module read a paged cache past its first rows, as one buffer - a path the paged "
            "reader does not take"
        )

    def hop(self, kb: torch.Tensor, vb: torch.Tensor) -> None:
        """a host layer's rows into `(kb, vb)` - [1, Hk, cap, D] buffers the caller owns on the card, room for every
        row its passes bring - for a prefill's chunks there: they append in place and read the rows as one buffer,
        as a contiguous layer's hop does (`GrowLayer.hop`), until `land`"""
        n = self.n
        if n:
            k, v = self.pool.host.gather(self.i, self.table.rows()[:n])
            kb[..., :n, :].copy_(k)
            vb[..., :n, :].copy_(v)
        self._hop = (kb, vb, n)

    def land(self, *_: Any) -> None:
        """the rows the hop's chunks made, back into the host's region through the table; the buffers the caller's
        again"""
        if self._hop is None:
            return
        kb, vb, n0 = self._hop
        self._hop = None
        if self.n > n0:
            self.pool.host.write(self.i, self.table.rows()[n0 : self.n], kb[0, :, n0 : self.n], vb[0, :, n0 : self.n])

    def gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        """the layer's rows in order as [1, Hk, n, D] tensors of their own: a copy"""
        rows = self.table.rows()[: self.n]
        if self.on_card:
            card = self.pool.card
            assert card is not None
            return card.gather(self.i, rows, self.pool.pages.pages)
        return self.pool.host.gather(self.i, rows)

    def crop(self, n: int) -> None:
        self.n = min(self.n, max(0, int(n)))


class PagedCache(DynamicCache):
    """a session's cache over the engine's pool: its attention layers `PagedLayer`s over one table, opened on the
    rows `rows` of a prefix the tree holds (read in place). `commit` gives the tree what the session made"""

    paged = True

    def __init__(self, prefix: PrefixCache, rows: Sequence[int] = ()) -> None:
        super().__init__(config=prefix.cfg)
        self.prefix = prefix
        self.table = Table(prefix.pool, rows)
        for i, lt in enumerate(prefix.layer_types):
            if lt != LayerKind.LINEAR:
                pl = PagedLayer(self.table, prefix.pool, i)
                pl.n = len(self.table)
                self.layers[i] = pl

    def growth(self, T: int) -> dict[str, int]:
        """the memory the pool's growth for the next T rows takes, by where: the host's region past its free pages
        (`HostRegion.growth`) and the park's for the pages binding the table parks, in RAM; the card's slots for the
        table's pages and its new ones past those free, on the card. Empty where the pool holds them already"""
        pool = self.prefix.pool
        new = self.table.pages_for(T)
        out: dict[str, int] = {}
        short = new - len(pool.pages.free)
        ram = pool.host.growth(len(pool.pages.pages) + short) if short > 0 else 0
        card = pool.card
        if card is not None:
            slots, parks = card.need(self.table, new)
            ram += card.park_growth(len(card.parked) + parks) if parks else 0
            c = card.growth(card.cap + slots) if slots else 0
            if c:
                out[str(card.dev)] = c
        if ram:
            out["cpu"] = out.get("cpu", 0) + ram
        return out

    def bind(self, T: int = 0) -> torch.Tensor | None:
        """the table on the card for a pass of `T` more rows - its rows reserved, its pages there, its map uploaded
        (`CardRegion.bind`): the card's map, None where the pool has no card. The rows counted from the layers the pass
        has yet to run: the host layers before a card run have appended the pass's rows already, and counted from
        them a prompt's chunk reserved its rows twice"""
        card = self.prefix.pool.card
        if card is None:
            return None
        n = min((cl.n for cl in self.layers if isinstance(cl, PagedLayer)), default=0)
        self.table.extend(n + int(T))
        return card.bind(self.table)

    def crop(self, max_length: int) -> None:
        """transformers' crop (a negative length counting from the end), the table cut with the layers: a layer cut
        alone would leave the table's rows past it to be written again, rows the tree may read"""
        n = len(self.table)
        self.crop_to(n + max_length if max_length < 0 else max_length)

    def crop_to(self, n: int) -> None:
        """the sequence's first `n` positions kept, every layer's and the table's"""
        for cl in self.layers:
            if isinstance(cl, PagedLayer):
                cl.crop(n)
        self.table.crop(n)

    def keep_path(self, base: int, path: Sequence[int]) -> None:
        """a verify pass's accepted path into place (`path` its nodes, the pass's rows from `base`): each layer's
        accepted rows copied down to the path's positions in its own pages, the rest let go"""
        moved = self.table.path_rows(base, path)
        if moved is not None:
            self.prefix.pool.move(*moved)
        self.crop_to(base + len(path))

    def commit(self, ids: Sequence[int]) -> None:
        """the session's tokens and the rows holding them into the tree: what a later prompt opens on"""
        n = min(len(ids), len(self.table))
        if n:
            self.prefix.insert(ids[:n], self.table.rows()[:n].tolist())

    def release(self) -> None:
        self.table.release()
        for cl in self.layers:
            if isinstance(cl, PagedLayer):
                cl.n = 0
