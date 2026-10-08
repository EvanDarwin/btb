# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The KV pages every conversation's rows live in, and who holds them.

A page is `PAGE` rows of every attention layer. A conversation's cache is a list of rows - `page * PAGE + offset` each,
its logical positions in order - so a prefix two conversations share is the same pages, referenced by both and held
once. A page counts its holders (`refs`: the conversations' tables and the prefix tree's nodes); with none it goes back
to the free list. Its rows are written once, in order: `fill` is how many hold rows, `frozen` how many another holder
references (never rewritten), and only its `writer` appends past `fill`. Where a page lives (`where`: the card, its
parking in pinned RAM, or the host) is the pool's to move; this module is the bookkeeping, the regions holding the
tensors are the engine's.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# rows a page: the unit allocated, moved and let go. The attention reads key j through the row map whatever its page,
# so the size is memory's choice, never the answer's: a conversation wastes at most PAGE - 1 rows at its end
PAGE = 64


class PoolError(RuntimeError):
    """a page asked of a pool that has none and may not grow, or a holder's count gone wrong"""


@dataclass(eq=False)
class Page:
    """one page's bookkeeping: its holders, the rows written and frozen, its writer, where it lives, when it was last
    used (the pool's clock)"""

    id: int
    refs: int = 0
    fill: int = 0
    frozen: int = 0
    writer: object | None = None
    where: str = "host"
    tick: int = 0


class PagePool:
    """Pages handed out and taken back. `alloc()` gives a free page (one holder: the caller) or, with none free, asks
    `grow(pages)` - the ledger's grant for the storage of `pages` pages in all - before it adds one; a grow that
    raises leaves the pool as it was. `ref`/`unref` count holders; the last `unref` frees the page."""

    def __init__(self, grow: Callable[[int], None] | None = None, rows: int = PAGE) -> None:
        self.rows = int(rows)
        self.pages: list[Page] = []
        self.free: list[int] = []
        self._grow = grow
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
                p = self.pages[self.free.pop()]
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
            self.free.append(p.id)
            return True

    def of(self, rows: Iterable[int]) -> list[Page]:
        """the distinct pages `rows` lie in, in the order first met"""
        seen: dict[int, Page] = {}
        for r in rows:
            i = int(r) // self.rows
            if i not in seen:
                seen[i] = self.pages[i]
        return list(seen.values())

    def freeze(self, rows: Iterable[int]) -> None:
        """`rows` referenced by another holder from now on: never written again"""
        for r in rows:
            p = self.pages[int(r) // self.rows]
            p.frozen = max(p.frozen, int(r) % self.rows + 1)

    def lru(self, where: str | None = None, keep: Callable[[Page], bool] | None = None) -> list[Page]:
        """the held pages (on `where`, any when None), least recently used first, but those `keep` holds back"""
        free = set(self.free)
        out = [
            p
            for p in self.pages
            if p.id not in free and (where is None or p.where == where) and (keep is None or not keep(p))
        ]
        out.sort(key=lambda p: p.tick)
        return out

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
            if not 0 <= p.frozen <= p.fill <= self.rows and p.refs:
                bad.append(f"page {p.id}: frozen {p.frozen}, fill {p.fill} of {self.rows}")
        return bad


def describe(pool: PagePool) -> dict[str, Any]:
    """the pool's figures for a log line or a report: pages held, free, by where they live"""
    where: dict[str, int] = {}
    free = set(pool.free)
    for p in pool.pages:
        if p.id not in free:
            where[p.where] = where.get(p.where, 0) + 1
    return {"held": len(pool), "free": len(pool.free), "rows": pool.rows, "where": where}
