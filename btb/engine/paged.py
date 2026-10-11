# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's KV rows in the pages of a pool every conversation on the engine shares (btb/engine/kvpool.py).

`KvPool` is the pages and where their rows lie. The layers the host runs keep theirs in the `HostRegion`: per layer a
K and a V region [Hk, rows, D] indexed by page id, head-major as the native kernels read a cache (head g's row r at
g * rows * D + r * D). On a card the layers the card runs keep theirs in the `CardRegion`: per layer an arena in VRAM,
position-major [slots * PAGE, Hk, D] as the card's kernels read it, a page at a slot of its own - the pages of the
conversation the card decodes (`bind`), and other conversations' until their slots are wanted, then parked in pinned
RAM, least recently used first, until a conversation reading them is bound again. A conversation's `Table` is its row map - position j of the sequence at row `rows()[j]` (page * PAGE +
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
from .fused import DELTA_BLOCK
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


def _page_rows(starts: Iterable[int]) -> torch.Tensor:
    """the rows of the pages at `starts` (page ids, or slots), PAGE each, in order: an int64 index on the host"""
    s = torch.tensor(list(starts), dtype=torch.int64)
    return (s[:, None] * PAGE + torch.arange(PAGE)[None]).flatten()


def _consecutive(pages: Iterable[Page]) -> list[list[Page]]:
    """`pages` (on the card) in runs whose slots go up by one, at most MOVE_PAGES long: a run's rows one slice of an
    arena"""
    out: list[list[Page]] = []
    for p in sorted(pages, key=lambda p: p.slot):
        if out and out[-1][-1].slot + 1 == p.slot and len(out[-1]) < MOVE_PAGES:
            out[-1].append(p)
        else:
            out.append([p])
    return out


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
        # a layer's regions short of `cap` pages: a fill the ledger refused part way (`_fill`), finished at the next
        # ask - a write's `shape`, a growth - before any row lands in them
        self._short = False

    def _el(self) -> int:
        return torch.empty(0, dtype=self.dtype or self.kv_dtype or torch.float32).element_size()

    def _target(self, pages: int) -> int:
        """the pages a growth to hold `pages` makes room for: GROW past what the regions hold"""
        return max(int(pages), int(self.cap * self.GROW), self.MIN_PAGES)

    def nbytes(self) -> int:
        """what the regions hold"""
        return sum(t.numel() * t.element_size() for store in (self.k, self.v) for t in store.values())

    def growth(self, pages: int) -> int:
        """the free memory a growth to hold `pages` pages takes at its peak: what the regions add, and the buffer a
        new one replaces, held until its rows are copied over (the last one's, every other let go by then); 0 where
        they hold them already"""
        if (pages <= self.cap and not self._short) or not self.layers:
            return 0
        new = self._target(pages) if pages > self.cap else self.cap
        row = self.hk * PAGE * self.d * self._el()
        if self.dtype is None:
            return 2 * len(self.layers) * (new - self.cap) * row + self.cap * row
        have = [store[i].shape[1] // PAGE if i in store else 0 for store in (self.k, self.v) for i in self.layers]
        return sum(max(0, new - n) for n in have) * row + max(have, default=0) * row

    def shape(self, k: torch.Tensor) -> None:
        """the rows' dtype from the first ones written, [B, Hk, T, D] - their heads and width the pool's - and every
        layer's regions made for the pages counted (`_fill`) before a row lands in them"""
        if int(k.shape[1]) != self.hk or int(k.shape[-1]) != self.d:
            raise PagedError(
                f"rows of {int(k.shape[1])} heads of {int(k.shape[-1])} into a pool of {self.hk} of {self.d}"
            )
        if self.dtype is None:
            self.dtype = self.kv_dtype or k.dtype
            # pages made before the host's first rows (the card's layers wrote first): their regions made now
            self._short = bool(self.cap)
        if self._short:
            self._fill(self.cap)

    def grow(self, pages: int) -> None:
        """the regions hold rows for `pages` pages: each layer's K and V regrown in turn as the ledger grants it -
        once the rows' dtype is known (`shape`), the pages counted till then"""
        if pages <= self.cap and not self._short:
            return
        new = self._target(pages) if pages > self.cap else self.cap
        if self.dtype is not None:
            self._fill(new)
        self.cap = new

    def _fill(self, pages: int) -> None:
        """every layer's K and V made or regrown to `pages` pages as the ledger grants each. Refused part way, the
        layers grown keep theirs and the region stays short (`_short`): the next write or growth goes on from the first
        layer short - with the dtype set and the pages counted, nothing asked again, and a write to a layer whose
        regions were never made raised KeyError however much RAM had come back"""
        self._short = True
        for i in self.layers:
            self._regrow(i, pages)
        self._short = False

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

    @_kept
    def shrink(self, pages: int) -> int:
        """the regions cut to `pages` pages - the pool let every page past them go (`PagePool.trim`) - a buffer at a
        time, its rows kept copied into one of the new length as the ledger grants it, or let go whole where no page
        is left. Refused part way, the buffers not yet cut keep their length, their rows past `pages` unread, and the
        next ask cuts them: what is cut is read off the buffers' lengths, not the page count - cut already, a retry
        found nothing to do and they stayed held. Returns the bytes given back: grown for the longest conversation, the
        regions held them until the engine closed"""
        pages = max(0, int(pages))
        before = self.nbytes()
        rows, el = pages * PAGE, self._el()
        try:
            for store in (self.k, self.v):
                for i in list(store):
                    old = store[i]
                    if old.shape[1] <= rows:
                        continue
                    if rows:
                        if self.grant is not None:
                            # a copy of rows the ledger counted already, made smaller: it spends no reservation
                            self.grant(
                                self.hk * rows * self.d * el,
                                "kv",
                                requester=f"the prefix cache's layer {i} cut to {pages} pages",
                                device="cpu",
                                draws="",
                            )
                        store[i] = old[:, :rows].clone()
                    else:
                        del store[i]
                    del old
        except MemoryGrantError:
            pass
        self.cap = min(self.cap, pages)
        return max(0, before - self.nbytes())

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
    page's rows at `slot * PAGE ..` of every layer's - the pages of the conversation the card decodes (`bind`), and
    other conversations' until their slots are wanted, then parked, least recently used first. In pinned RAM: the
    park, the same layout, a page at a park slot of its own until a conversation reading it is bound again. `tbl` is the bound conversation's row map on the card: position j at card
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
        # copies into or out of the park queued on the card's stream and not yet waited for (`_settle`)
        self._queued = False

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

    def _el(self) -> int:
        return torch.empty(0, dtype=self.dtype).element_size()

    def _page_bytes(self) -> int:
        """one page's K and V of one layer"""
        return 2 * PAGE * self.hk * self.d * self._el()

    def nbytes(self) -> int:
        """what the arenas hold on the card"""
        return len(self.arenas) * self.layer_bytes()

    def layer_bytes(self) -> int:
        """what one layer's arena holds on the card: a layer leaving it frees that, one coming takes it"""
        return self._arena_bytes(self.cap)

    def park_nbytes(self) -> int:
        """what the park holds in RAM"""
        return len(self.arenas) * self._park_bytes(len(self.parked))

    def _arena_bytes(self, slots: int) -> int:
        """one layer's arena of `slots` slots as it takes the card (`RowArena.nbytes`): whole chunks of the driver's
        where it maps in place - a page's K or V rounded up to the chunk - its rows' bytes where it is regrown. Priced
        as pages, a growth the grant passed could take more than it asked for"""
        if not slots:
            return 0
        a = next(iter(self.arenas.values()), None)
        if a is None:  # no layer on the card yet: an arena not grown, only to size one
            a = self._sizer = getattr(self, "_sizer", None) or self._arena()
        return int(a.nbytes(slots * PAGE))

    def _park_bytes(self, slots: int) -> int:
        """one layer's park of `slots` slots as it takes RAM, as the arenas' (`_arena_bytes`)"""
        if not slots:
            return 0
        p = next(iter(self.parks.values()), None)
        if p is None:
            p = self._park_sizer = getattr(self, "_park_sizer", None) or self._park_arena()
        return int(p.nbytes(slots * PAGE))

    def _target(self, have: int, slots: int) -> int:
        return max(int(slots), int(have * self.GROW), self.MIN_SLOTS)

    def growth(self, slots: int) -> int:
        """the card memory a growth to `slots` slots takes: what the arenas add (in place nothing moves; regrown, a
        layer's buffer at a time beside its predecessor); 0 where they hold them already"""
        if slots <= self.cap or not self.arenas:
            return 0
        new = self._target(self.cap, slots)
        return len(self.arenas) * (self._arena_bytes(new) - self._arena_bytes(self.cap)) + self._beside()

    def park_growth(self, slots: int) -> int:
        """the RAM a growth of the park to `slots` slots takes at its peak: what the parks add, and regrown rather
        than mapped in place, a layer's park at a time, its old one beside its new; 0 where it holds them already"""
        have = len(self.parked)
        if slots <= have or not self.arenas:
            return 0
        new = self._target(have, slots)
        return len(self.arenas) * (self._park_bytes(new) - self._park_bytes(have)) + self._park_beside()

    def _park_beside(self) -> int:
        """the old park a park's growth holds beside its new one: one layer's, where the parks are regrown rather than
        mapped in place"""
        p = next(iter(self.parks.values()), None)
        return 0 if p is None or p.in_place else self._park_bytes(len(self.parked))

    def _beside(self) -> int:
        """the old arena an arenas' growth holds beside its new one: one layer's, where they are regrown rather than
        mapped in place"""
        a = next(iter(self.arenas.values()), None)
        return 0 if a is None or a.in_place else self._arena_bytes(self.cap)

    def _counts(self, table: Table) -> tuple[int, int]:
        """(the table's pages in the park, the pages on the card it does not read): what binding it moves - a page is
        placed on the card as it is made (`KvPool.alloc`), so each one a table holds is on the card or in the park. In
        constant time while it stays bound with nothing moved since its map was made - each page it holds is on the
        card then - else over its pages (a switch's)"""
        held = table.held
        bound = self._bound() if self._bound is not None else None
        used = len(self.slots) - len(self.free)
        if bound is table and self._mapped[0] == self.version:
            return 0, used - len(held)
        back = sum(1 for p in held.values() if p.park >= 0)
        return back, used - (len(held) - back)

    def need(self, table: Table, new: int) -> tuple[int, int]:
        """what binding `table` and placing `new` more pages of it take past what the region holds: (card slots, park
        slots), moved as `bind` and `reserve` move them - its parked pages into the free slots first, then traded with
        pages on the card it does not read (their park slots theirs), the rest into slots the arenas grow for; its new
        pages into the free slots left, then into the slots of more pages it does not read, parked, then grown"""
        back, others = self._counts(table)
        free, pfree = len(self.free), len(self.pfree)
        loaded = min(back, free)  # into free slots: their park slots free after
        traded = min(back - loaded, others)
        grow = back - loaded - traded
        free, others, pfree = free - loaded, others - traded, pfree + loaded
        want = int(new)
        into_free = min(want, free)
        parked = min(want - into_free, others)
        grow += want - into_free - parked
        return grow, max(0, parked - pfree)

    def _others(self, held: dict[int, Page]) -> list[Page]:
        """the pages on the card a table holding `held` does not read, least recently used first: the ones parked for
        its room"""
        return sorted((p for p in self.slots if p is not None and p.id not in held), key=lambda p: p.tick)

    def reserve(self, k: int, table: Table) -> None:
        """`k` free slots on the card for `table`'s pages to come: the slots of pages it does not read parked for them,
        least recently used first, the rest grown - one move for the pass's new pages, not one a page"""
        with self._move():
            short = int(k) - len(self.free)
            if short <= 0:
                return
            out = self._others(table.held)[:short]
            self.park(out)
            if short > len(out):
                self._grow(self.cap + short - len(out))

    @_kept
    def _grow(self, slots: int) -> None:
        if slots <= self.cap:
            return
        new = self._target(self.cap, slots)
        if self.grant is not None and self.arenas:
            # the growth's peak (`growth`), which the free memory must hold - not the arenas' whole new size: grown in
            # place, what they hold already is no part of it (asked for whole, a growth `growth` priced as fitting was
            # refused). The old arena held beside its new one is let go once filled
            self.grant(
                self.growth(slots),
                "kv",
                requester=f"the prefix cache's card arenas, {len(self.arenas)} layers x {new} pages of {PAGE} rows",
                device=self.dev,
                held=self._beside(),
            )
        try:
            for a in self.arenas.values():
                a.grow(new * PAGE)
        finally:
            # the arenas longer, perhaps at new addresses - those grown before a later one's growth failed among them:
            # every graph captured over the old addresses made again (moved on success alone, a request after the
            # failure took a graph that read and wrote the arena where it no longer was)
            self.layout += 1
        for s in range(len(self.slots), new):
            heapq.heappush(self.free, s)
        self.slots.extend([None] * (new - len(self.slots)))

    @_kept
    def _grow_park(self, slots: int) -> None:
        if slots <= len(self.parked):
            return
        new = self._target(len(self.parked), slots)
        if self.grant is not None and self.arenas:
            # the growth's peak (`park_growth`), as the card arenas' (`_grow`)
            self.grant(
                self.park_growth(slots),
                "kv",
                requester=f"the prefix cache's park, {len(self.arenas)} layers x {new} pages of {PAGE} rows",
                device="cpu",
                held=self._park_beside(),
            )
        # a park regrown where the driver maps no RAM in place is copied by the host: the copies queued into it first
        self._settle()
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

    def _copy(self, src: dict[int, Any], dst: dict[int, Any], runs: list[tuple[int, int, int]]) -> None:
        """every layer's K and V rows of each run (from slot, to slot, pages) copied from `src`'s arenas to `dst`'s:
        the card's and the park's, either way, on the card's stream"""
        for i in self.arenas:
            for w in ("k", "v"):
                a, b = src[i].view(w, 0), dst[i].view(w, 0)
                for s, d, n in runs:
                    b[d * PAGE : (d + n) * PAGE].copy_(a[s * PAGE : (s + n) * PAGE], non_blocking=True)
        self._queued = self._queued or bool(runs and self.arenas)

    def _settle(self) -> None:
        """the copies queued into and out of the park landed, before the host reads or copies the park itself: they
        run on the card's stream, the host's reads do not wait for it (a gather read rows a trade was still writing,
        another conversation's)"""
        if self._queued:
            if self.dev.type == "cuda":
                torch.cuda.synchronize(self.dev)
            self._queued = False

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
                p.park = ps
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
                p.park = -1
            self.version += 1
            if self.arenas:
                self.loaded += len(pages)

    def bind_for(self, table: Table, n: int) -> torch.Tensor:
        """`table` bound (`bind`) with rows for its positions up to `n`: its own pages onto the card first, then room
        for its new ones made in one move (`reserve`), then they are placed and the map brought up to date - placed
        first, the new pages took slots the arenas grew for while the slots of the pages the binding then parked stood
        empty. Each step a move of its own, under the lock: a page let go during one is freed before the table takes
        new ones (`KvPool.alloc`), which could otherwise be that very page"""
        with self.lock:
            self.bind(table)
            self.reserve(table.pages_for(max(0, int(n) - len(table))), table)
            table.extend(int(n))
            return self.bind(table)

    @_kept
    def swap(self, out: Sequence[Page], back: Sequence[Page]) -> None:
        """each page of `out` (on the card) and the page of `back` beside it (in the park) trading places, the parked
        one into the card slot and the card's into the park slot, through a staging buffer of up to MOVE_PAGES pages
        on the card: a switch between conversations that fill the card and the park grows neither, where parking the
        one before loading the other grew the park by every page it took off the card"""
        pairs = list(zip(out, back, strict=False))
        if not pairs:
            return
        with self._move():
            if self.arenas:
                n = min(len(pairs), MOVE_PAGES)
                if self.grant is not None:
                    self.grant(
                        n * self._page_bytes() // 2,
                        "kv",
                        requester="the prefix cache's trade of pages between the card and the park",
                        device=self.dev,
                        draws="",
                    )
                stage = torch.empty(n * PAGE, self.hk, self.d, dtype=self.dtype, device=self.dev)
                for i in self.arenas:
                    for w in ("k", "v"):
                        card, park = self.arenas[i].view(w, 0), self.parks[i].view(w, 0)
                        for c0 in range(0, len(pairs), n):
                            chunk = pairs[c0 : c0 + n]
                            for j, (o, _) in enumerate(chunk):
                                stage[j * PAGE : (j + 1) * PAGE].copy_(card[o.slot * PAGE : (o.slot + 1) * PAGE])
                            for o, b in chunk:
                                src = park[b.park * PAGE : (b.park + 1) * PAGE]
                                card[o.slot * PAGE : (o.slot + 1) * PAGE].copy_(src, non_blocking=True)
                            for j, (_, b) in enumerate(chunk):
                                src = stage[j * PAGE : (j + 1) * PAGE]
                                park[b.park * PAGE : (b.park + 1) * PAGE].copy_(src, non_blocking=True)
                self._queued = True
            for o, b in pairs:
                s, ps = o.slot, b.park
                self.slots[s], self.parked[ps] = b, o
                self._set_slot(b, s)
                b.park = -1
                self._set_slot(o, -1)
                o.park = ps
            self.version += 1
            if self.arenas:
                self.loaded += len(pairs)

    def bind(self, table: Table) -> torch.Tensor:
        """`table`'s pages on the card and its row map there (`tbl`, int32: position j at card row tbl[j]) brought up
        to date, only its new positions uploaded while it stays the one bound. Its pages in the park come back into
        the free slots, then trade places with pages on the card it does not read (least recently used first), and
        the rest into slots the arenas grow for. Another conversation's pages stay where they are until their slots
        are wanted: a switch back and forth with room for both moves nothing"""
        with self._move():
            bound = self._bound() if self._bound is not None else None
            tbl = self.tbl
            if bound is not table:
                held = table.held
                # none bound until this one is: a load refused part way leaves the table bound before with its pages
                # elsewhere, and its next bind must bring them back, not read its map as current
                self._bound = None
                # its pages used now: the conversation bound last is the last parked (`_others`), a prefix it shares
                # with an idle one among them - stamped only when made, a hot system prompt went first
                table.pool.pages.tick(held.values())
                back = sorted((p for p in held.values() if p.park >= 0), key=lambda p: p.park)
                into_free = min(len(back), len(self.free))
                self.load(back[:into_free])
                rest = back[into_free:]
                if rest:
                    out = self._others(held)[: len(rest)]
                    self.swap(out, rest[: len(out)])
                    self.load(rest[len(out) :])
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
        pages (`pages`, the pool's by id) lie - the card, or the park: a copy, one gather from each"""
        r = rows.to(torch.int64).cpu()
        pid, off = r // PAGE, r % PAGE
        slot = torch.full_like(pid, -1)
        known = pid < len(self.slot_of)
        slot[known] = self.slot_of[pid[known]]
        on = slot >= 0
        at_on, at_off = torch.nonzero(on).flatten(), torch.nonzero(~on).flatten()
        park = torch.empty(0, dtype=torch.int64)
        if len(at_off):
            ids, inv = torch.unique(pid[at_off], return_inverse=True)
            where = torch.tensor([pages[int(x)].park for x in ids.tolist()], dtype=torch.int64)
            if bool((where < 0).any()):
                raise PagedError(f"pages {ids[where < 0].tolist()} have no rows on the card or in the park")
            park = where[inv] * PAGE + off[at_off]
            self._settle()  # read on the host: what a park or a trade is still copying into it landed first
        out = []
        for w in ("k", "v"):
            got = torch.empty(len(r), self.hk, self.d, dtype=self.dtype, device=self.dev)
            if len(at_on):
                src = (slot[at_on] * PAGE + off[at_on]).to(self.dev)
                got[at_on.to(self.dev)] = self.arenas[i].view(w, 0).index_select(0, src)
            if len(at_off):
                got[at_off.to(self.dev)] = self.parks[i].view(w, 0).index_select(0, park).to(self.dev)
            out.append(got.transpose(0, 1)[None])
        return out[0], out[1]

    def _stage(self, n: int, what: str, host: HostRegion) -> None:
        """the copies of a layer's move asked of the scheduler: `n` pages' K or V staged in RAM - gathered off the
        host's region, made contiguous, cast - a run at a time. Nothing on the card: a run of slots is sent straight
        into the arena or read straight out of it, so a shed, which is what makes the card's room, never waits for
        room on the card (staged there, a game launching found the shed refused and every request failing)"""
        if self.grant is not None and n:
            el = torch.empty(0, dtype=host.dtype or host.kv_dtype or torch.float32).element_size()
            nbytes = n * PAGE * self.hk * self.d * (2 * el + self._el())
            who = f"the prefix cache's {what}, a run at a time"
            self.grant(nbytes, "kv", requester=who, device="cpu", draws="")

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
        region holds and every live page's rows of it copied in from the host's region, a run of slots a copy straight
        into the arena. Under the move (`_move`): a conversation let go meanwhile frees its pages once the copies that
        read their slots are done. Every grant asked before anything is made, the arena and park registered once their
        rows are in: refused or failed part way, the layer's rows are the host's as they were, and the move asked
        again makes it (registered first, a refused stage left an arena of no rows the next move took as made)"""
        if i in self.arenas:
            return
        with self._move():
            rows = i in host.k
            on = [p for p in live if p.slot >= 0] if rows else []
            parked = [p for p in live if p.slot < 0 and p.park >= 0] if rows else []
            if self.grant is not None:
                who = f"the prefix cache's card arena for layer {i}"
                self.grant(self._arena_bytes(self.cap), "kv", requester=who, device=self.dev)
                if self.parked:
                    who = f"the prefix cache's park for layer {i}"
                    self.grant(self._park_bytes(len(self.parked)), "kv", requester=who, device="cpu")
            self._stage(min(MOVE_PAGES, max(len(on), len(parked))), f"layer {i}'s rows onto the card", host)
            a = self._arena()
            pk = self._park_arena() if self.parked else None
            try:
                if self.cap:
                    a.grow(self.cap * PAGE)
                if pk is not None:
                    pk.grow(len(self.parked) * PAGE)
                for w, src in (("k", host.k[i]), ("v", host.v[i])) if rows else ():
                    for run in _consecutive(on):
                        got = src.index_select(1, _page_rows(p.id for p in run)).transpose(0, 1).contiguous()
                        s = run[0].slot
                        a.view(w, 0)[s * PAGE : (s + len(run)) * PAGE].copy_(got.to(self.dtype))
                    for c0 in range(0, len(parked), MOVE_PAGES):
                        assert pk is not None
                        run = parked[c0 : c0 + MOVE_PAGES]
                        got = src.index_select(1, _page_rows(p.id for p in run)).transpose(0, 1)
                        pk.view(w, 0).index_copy_(0, _page_rows(p.park for p in run), got.to(self.dtype))
            except BaseException:
                a.close()
                if pk is not None:
                    pk.close()
                raise
            self.arenas[i] = a
            if pk is not None:
                self.parks[i] = pk
            self.layout += 1

    def drop(self, i: int, host: HostRegion, live: Sequence[Page]) -> None:
        """layer i's rows held on the host from now on (it left the card): every live page's rows of it copied to
        the host's region (made for it there first), a run of slots read straight off the arena in one copy - nothing
        made on the card, whose room the shed is making - and its arena and park let go. Under the move, as `add`;
        refused or failed part way, the host's region made for it goes again and the layer's rows stay the card's"""
        if i not in self.arenas:
            return
        with self._move():
            a, pk = self.arenas[i], self.parks.get(i)
            on = [p for p in live if p.slot >= 0]
            parked = [p for p in live if p.slot < 0 and p.park >= 0] if pk is not None else []
            self._stage(min(MOVE_PAGES, max(len(on), len(parked))), f"layer {i}'s rows off the card", host)
            made = i not in host.layers
            try:
                host.add(i)
                if i in host.k:
                    self._settle()  # the park's copies in flight landed before the host reads it
                    for w, dst in (("k", host.k[i]), ("v", host.v[i])):
                        for run in _consecutive(on):
                            s = run[0].slot
                            got = a.view(w, 0)[s * PAGE : (s + len(run)) * PAGE].cpu()
                            dst.index_copy_(1, _page_rows(p.id for p in run), got.transpose(0, 1).to(dst.dtype))
                        for c0 in range(0, len(parked), MOVE_PAGES):
                            assert pk is not None
                            run = parked[c0 : c0 + MOVE_PAGES]
                            got = pk.view(w, 0).index_select(0, _page_rows(p.park for p in run))
                            dst.index_copy_(1, _page_rows(p.id for p in run), got.transpose(0, 1).to(dst.dtype))
            except BaseException:
                if made:
                    host.drop(i)
                raise
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
        self.grant = grant  # the scheduler's, which a fork of a paged layer asks as its own rows grow
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
        The card's room is the pass's to have made (`cache_room`): a slot it cannot grow for is refused as it is. Never
        while the card's slots move: a page let go there waits to be freed (`CardRegion._move`), and handed out again
        meanwhile its new slot would be the one freed"""
        if self.card is not None and self.card._moving:
            raise PagedError("a page asked for while the card's slots move: the move's held-back frees would take it")
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
        # the pages found and moved under the card's move: one a conversation lets go meanwhile (its table collected
        # by an allocation of the copies, or on another thread) is freed once the copies are done, never mid-copy
        with self.card._move():
            live = [p for p in self.pages.pages if p.refs > 0]
            if card:
                self.card.add(i, self.host, live)
                self.host.drop(i)
            else:
                self.card.drop(i, self.host, live)

    def trim(self) -> int:
        """the free pages past the last one held let go, and the host's region cut to the pages left - never below its
        first growth (`HostRegion.MIN_PAGES`), the room any request starts in: the RAM given back. Its buffers grow for
        the longest conversation and kept that length once its pages were free - a long decode with no session held
        gigabytes no conversation read. Cut to nothing, a model that had given up all it could for another program
        could not grow them again to answer its next request. Nothing while the card's slots move (a page let go there
        is freed once the move is done)"""
        if self.card is not None and self.card._moving:
            return 0
        with self.pages.lock:
            return self.host.shrink(max(self.pages.trim(), HostRegion.MIN_PAGES))

    def trimmable(self) -> int:
        """the RAM `trim` would give back now: every buffer past the pages left, read off its length (a cut refused
        part way leaves some longer than the page count says)"""
        with self.pages.lock:
            pages = self.pages.pages
            n = len(pages)
            while n and pages[n - 1].refs <= 0:
                n -= 1
            n = max(n, HostRegion.MIN_PAGES)
            row = self.host.hk * PAGE * self.host.d * self.host._el()
            return sum(max(0, t.shape[1] // PAGE - n) * row for s in (self.host.k, self.host.v) for t in s.values())

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
        # the conversation's commit held back for the tree (`PagedCache.commit`), and the position from which its rows
        # were made off the route its steps take, never given to the tree (`PagedCache.off_route`)
        self.held_ids: list[int] | None = None
        self.exact: int | None = None
        # a hybrid's: the positions its rows and recurrent states are the prompt's prefilled cold up to - a decode's
        # step, a prompt's call starting inside a block, made others - and the snapshots of its states held back with
        # the commit, by position (`PagedCache.prefilled`, `commit`). None for a model with no recurrent layer
        self.cold: int | None = None
        self.held_snaps: dict[int, Any] | None = None
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
        self.held_ids = self.exact = self.held_snaps = None
        if self.cold is not None:
            self.cold = 0
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
        if self.held_ids is not None and len(self.held_ids) > n:
            del self.held_ids[n:]  # the commit held back cut with the rows: what it cuts never reached the tree
        if self.held_snaps:
            self.held_snaps = {k: s for k, s in self.held_snaps.items() if k <= n} or None
        if self.cold is not None:
            self.cold = min(self.cold, n)
        if self.exact is not None and n <= self.exact:
            # every row made off the route cut: the rows made from here are the route's again (kept, the tree was
            # given nothing past them for the table's life)
            self.exact = None
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

    @property
    def grant(self) -> Callable[..., None] | None:
        """the scheduler's gate, the pool's: a fork of the layer (`branches._fork_layer`) asks it as its rows grow"""
        return self.pool.grant

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
        if self._hop is None and self.on_card:
            card = self.pool.card
            assert card is not None
            tbl = card.bind_for(self.table, self.n + T)
            card.write(self.i, tbl[self.n : self.n + T], k[0], v[0])
            self.n += T
            return card.view(self.i)
        self.table.extend(self.n + T)
        if self._hop is not None:
            kb, vb, _ = self._hop
            kb[0, :, self.n : self.n + T].copy_(k[0])
            vb[0, :, self.n : self.n + T].copy_(v[0])
            self.n += T
            return kb[..., : self.n, :], vb[..., : self.n, :]
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

    def unhop(self) -> None:
        """a hop given up (the prefill failed before `land`): the layer's rows read and written in the host's region
        again, the buffers the caller's - the chunks' rows past the hop's start never made, as a failed pass's rows
        are not (the session's rollback cuts them)"""
        if self._hop is not None:
            self.n = min(self.n, self._hop[2])
            self._hop = None

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
        """the layer's rows in order as [1, Hk, n, D] tensors of their own: a copy, asked of the scheduler first (a
        fork's prefix, a batch's rows: the whole conversation's K and V again)"""
        rows = self.table.rows()[: self.n]
        card = self.pool.card if self.on_card else None
        dtype = card.dtype if card is not None else (self.pool.host.dtype or torch.float32)
        dev = card.dev if card is not None else torch.device("cpu")
        if self.grant is not None and self.n:
            el = torch.empty(0, dtype=dtype).element_size()
            self.grant(
                2 * self.n * self.pool.hk * self.pool.d * el,
                "kv",
                requester=f"a paged layer's rows gathered, {self.n} rows of layer {self.i}",
                device=dev,
                draws="",
            )
        if card is not None:
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
        if LayerKind.LINEAR in prefix.layer_types:
            # the tree's rows are a cold prefill's, up to the snapshot a hybrid opens on (`PrefixCache.put`)
            self.table.cold = len(self.table)
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

    def bind(self, T: int = 0) -> None:
        """the table on the card for a pass of `T` more rows - its rows reserved, its pages there, its map uploaded
        (`CardRegion.bind`), which the card's kernels read from the region. The rows counted from the layers the pass
        has yet to run: the host layers before a card run have appended the pass's rows already, and counted from
        them a prompt's chunk reserved its rows twice (`CardRegion.bind_for`). Asked only of a pool with a card"""
        card = self.prefix.pool.card
        assert card is not None
        n = min((cl.n for cl in self.layers if isinstance(cl, PagedLayer)), default=0)
        card.bind_for(self.table, n + int(T))

    def crop(self, max_length: int) -> None:
        """transformers' crop (a negative length counting from the end), the table cut with the layers: a layer cut
        alone would leave the table's rows past it to be written again, rows the tree may read"""
        n = len(self.table)
        self.crop_to(n + max_length if max_length < 0 else max_length)

    def crop_to(self, n: int) -> None:
        """the sequence's first `n` positions kept, every layer's and the table's - and of the commit it holds back for
        the tree (`commit`): the rows past `n` never frozen, the table writes on in its own page"""
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

    def prefilled(self, a: int, b: int) -> None:
        """a hybrid's prompt prefilled from position `a` to `b` in one call: its rows and states the prompt's cold where
        they went on from a cold prefill's at a block's end (`DELTA_BLOCK`, where the DeltaNet's chunked rule cuts its
        blocks), as every call of the prompt taken whole does"""
        t = self.table
        if t.cold is not None and t.cold == a and a % DELTA_BLOCK == 0:
            t.cold = int(b)

    def commit(self, ids: Sequence[int], snaps: Sequence[Any] = ()) -> None:
        """the session's tokens and the rows holding them for the tree, what a later prompt opens on: held back on the
        table until the tree is next read (`PrefixCache.hold`) - the table, which holds the pages, not the cache, so a
        session let go is collected - a later commit replacing it, a crop cutting it. A hybrid's `snaps` (anchors, by
        their `n`) are held back with it, beside those an earlier commit held that it still holds the rows of"""
        n = min(len(ids), len(self.table))
        t = self.table
        t.held_ids = [int(x) for x in ids[:n]] if n else None
        if t.cold is not None:
            kept = {k: s for k, s in (t.held_snaps or {}).items() if k <= n}
            kept.update({int(s["n"]): s for s in snaps if 0 < int(s["n"]) <= n})
            t.held_snaps = kept or None
        if n:
            self.prefix.hold(t)
        else:
            self.prefix.let_go(t)

    def off_route(self, n: int) -> None:
        """the rows from position `n` on made off the route a step takes (a card layer the pass ran through torch,
        out of room on the card): the conversation reads them, the tree is never given them - a hit there would
        decode other bits than its prompt cold"""
        t = self.table
        t.exact = int(n) if t.exact is None else min(t.exact, int(n))

    def flush(self) -> None:
        """the commit held back into the tree now (`PrefixCache.put`)"""
        self.prefix.put(self.table)

    def release(self) -> None:
        """the table let go, what the session committed in the tree first: the next conversation finds it there"""
        self.flush()
        self.table.release()
        for cl in self.layers:
            if isinstance(cl, PagedLayer):
                cl.n = 0
