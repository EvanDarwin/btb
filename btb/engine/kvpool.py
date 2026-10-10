# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The KV pages every conversation's rows live in, and who holds them.

A page is `PAGE` rows of every attention layer. A conversation's cache is a list of rows - `page * PAGE + offset` each,
its logical positions in order - so a prefix two conversations share is the same pages, referenced by both and held
once. A page counts its holders (`refs`: the conversations' tables and the prefix tree's nodes); with none it goes back
to the free list. Its rows are written once, in order: `fill` is how many hold rows, `frozen` how many another holder
references (never rewritten), and only its `writer` appends past `fill`. Where a page's rows lie (the card's slot,
its parking in pinned RAM, the host's region) is the regions' to move (btb/engine/paged.py); this module is the
bookkeeping.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# rows a page: the unit allocated, moved and let go. The attention reads key j through the row map whatever its page,
# so the size is memory's choice, never the answer's: a conversation wastes at most PAGE - 1 rows at its end
PAGE = 64


class PoolError(RuntimeError):
    """a page asked of a pool that has none and may not grow, or a holder's count gone wrong"""


@dataclass(eq=False)
class Page:
    """one page's bookkeeping: its holders, the rows written and frozen, its writer, when it was last used (the pool's
    clock), and on a card where the rows of the layers the card runs sit - the card's slot for it, or the park's in
    pinned RAM (-1: none)"""

    id: int
    refs: int = 0
    fill: int = 0
    frozen: int = 0
    writer: object | None = None
    tick: int = 0
    slot: int = -1
    park: int = -1


class PagePool:
    """Pages handed out and taken back. `alloc()` gives a free page (one holder: the caller) or, with none free, asks
    `grow(pages)` - the ledger's grant for the storage of `pages` pages in all - before it adds one; a grow that
    raises leaves the pool as it was. `ref`/`unref` count holders; the last `unref` frees the page, and tells `freed`
    (the regions holding a page somewhere of their own let that go)."""

    def __init__(self, grow: Callable[[int], None] | None = None, freed: Callable[[Page], None] | None = None) -> None:
        self.pages: list[Page] = []
        # the free page ids, a heap: the lowest handed out first, so the pages held gather at the front and the free
        # ones past the last held can be let go (`trim`)
        self.free: list[int] = []
        self._grow = grow
        self._freed = freed
        self.clock = 0
        # a conversation's table can be let go from another thread (its session collected there): the counts and
        # the free list change under this, whoever changes them
        self.lock = threading.RLock()

    def __len__(self) -> int:
        """pages held: the pool's size less its free list"""
        return len(self.pages) - len(self.free)

    def tick(self, pages: Iterable[Page]) -> None:
        """`pages` used now: the least recently used are the first parked or let go"""
        self.clock += 1
        for p in pages:
            p.tick = self.clock

    def alloc(self, writer: object | None = None) -> Page:
        with self.lock:
            if self.free:
                p = self.pages[heapq.heappop(self.free)]
            else:
                if self._grow is not None:
                    self._grow(len(self.pages) + 1)
                p = Page(len(self.pages))
                self.pages.append(p)
            p.refs, p.fill, p.frozen, p.writer = 1, 0, 0, writer
            self.clock += 1
            p.tick = self.clock
            return p

    def ref(self, p: Page) -> None:
        with self.lock:
            if p.refs <= 0:
                raise PoolError(f"page {p.id} referenced after it was freed")
            p.refs += 1

    def unref(self, p: Page) -> bool:
        """one holder less; True when that was the last and the page is free again"""
        with self.lock:
            if p.refs <= 0:
                raise PoolError(f"page {p.id} let go more often than it was held")
            p.refs -= 1
            if p.refs:
                return False
            p.fill = p.frozen = 0
            p.writer = None
            if self._freed is not None:
                self._freed(p)
            heapq.heappush(self.free, p.id)
            return True

    def trim(self) -> int:
        """the free pages past the last one held let go: the pool's size after, the rows past it no page's (the
        regions' to give back, `KvPool.trim`)"""
        with self.lock:
            n = len(self.pages)
            while n and self.pages[n - 1].refs <= 0:
                n -= 1
            if n < len(self.pages):
                del self.pages[n:]
                self.free = [i for i in self.free if i < n]
                heapq.heapify(self.free)
            return n

    def of(self, rows: Iterable[int]) -> list[Page]:
        """the distinct pages `rows` lie in, in the order first met"""
        seen: dict[int, Page] = {}
        for r in rows:
            i = int(r) // PAGE
            if i not in seen:
                seen[i] = self.pages[i]
        return list(seen.values())

    def freeze(self, rows: Iterable[int]) -> None:
        """`rows` referenced by another holder from now on: never written again"""
        for r in rows:
            p = self.pages[int(r) // PAGE]
            p.frozen = max(p.frozen, int(r) % PAGE + 1)

    def check(self) -> list[str]:
        """the bookkeeping's own consistency: what is wrong, or nothing"""
        free = set(self.free)
        bad: list[str] = []
        if len(free) != len(self.free):
            bad.append("a page is on the free list twice")
        for p in self.pages:
            if p.id in free and p.refs:
                bad.append(f"page {p.id} is free with {p.refs} holders")
            if p.id not in free and p.refs <= 0:
                bad.append(f"page {p.id} is held by no one and not free")
            if not 0 <= p.frozen <= p.fill <= PAGE and p.refs:
                bad.append(f"page {p.id}: frozen {p.frozen}, fill {p.fill} of {PAGE}")
        return bad
