# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The KV pages' bookkeeping (btb/engine/kvpool.py): holders counted, a page freed with its last, growth asked of the
ledger before a page is made, rows frozen once another holder reads them, the least recently used first."""

from __future__ import annotations

import pytest

from btb.engine.kvpool import PAGE, PagePool, PoolError, describe


def test_a_page_lives_while_anyone_holds_it() -> None:
    """two holders of a page: the first to let go leaves it standing, the last frees it for the next alloc, which
    hands it out fresh (no rows, no writer) - and nothing is let go twice"""
    pool = PagePool()
    a = pool.alloc(writer="a")
    assert a.refs == 1 and a.writer == "a" and len(pool) == 1
    a.fill = 10
    pool.ref(a)
    assert not pool.unref(a) and len(pool) == 1
    assert pool.unref(a) and len(pool) == 0 and pool.free == [a.id]
    with pytest.raises(PoolError):
        pool.unref(a)
    with pytest.raises(PoolError):
        pool.ref(a)
    b = pool.alloc()
    assert b is a and b.fill == 0 and b.writer is None and b.refs == 1
    assert not pool.check()


def test_growth_is_asked_before_a_page_is_made() -> None:
    """a pool with no free page asks the ledger for the storage of one more first; refused, it holds what it held"""
    asked: list[int] = []
    refuse = [False]

    def grow(pages: int) -> None:
        asked.append(pages)
        if refuse[0]:
            raise MemoryError("refused")

    pool = PagePool(grow)
    first = pool.alloc()
    second = pool.alloc()
    assert asked == [1, 2]
    pool.unref(first)
    assert pool.alloc() is first and asked == [1, 2], "a free page was asked of the ledger again"
    refuse[0] = True
    with pytest.raises(MemoryError):
        pool.alloc()
    assert len(pool.pages) == 2 and len(pool) == 2 and not pool.check()
    del second


def test_rows_frozen_once_another_holder_reads_them() -> None:
    """`freeze` marks the rows another holder references, per page, to the furthest; `of` names the pages rows lie in
    in the order first met"""
    pool = PagePool()
    p, q = pool.alloc(), pool.alloc()
    p.fill, q.fill = PAGE, 5
    rows = [p.id * PAGE + 3, p.id * PAGE + 9, q.id * PAGE + 4]
    pool.freeze(rows)
    assert (p.frozen, q.frozen) == (10, 5)
    pool.freeze([p.id * PAGE + 2])
    assert p.frozen == 10, "freezing fewer rows thawed some"
    assert [x.id for x in pool.of([q.id * PAGE, p.id * PAGE + 1, q.id * PAGE + 2])] == [q.id, p.id]
    assert not pool.check()


def test_the_least_recently_used_come_first() -> None:
    """`lru` orders the held pages by their last use, on a tier or any, without those held back"""
    pool = PagePool()
    a, b, c = pool.alloc(), pool.alloc(), pool.alloc()
    b.where = "card"
    pool.tick([a])
    assert [p.id for p in pool.lru()] == [b.id, c.id, a.id]
    assert [p.id for p in pool.lru("card")] == [b.id]
    assert [p.id for p in pool.lru(keep=lambda p: p is c)] == [b.id, a.id]
    pool.unref(b)
    assert [p.id for p in pool.lru()] == [c.id, a.id], "a free page is listed as held"
    assert describe(pool) == {"held": 2, "free": 1, "rows": PAGE, "where": {"host": 2}}
