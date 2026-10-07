# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The prefix tree (btb/engine/radix.py): token paths to the rows holding them, shared where conversations start alike,
split where they part and at every snapshot, pages held by the nodes reading them, the least recently used leaves
first out."""

from __future__ import annotations

import random

from btb.engine.kvpool import PAGE, PagePool
from btb.engine.radix import RadixTree


def table(pool: PagePool, n: int) -> list[int]:
    """`n` rows in fresh pages, as a conversation's table writes them: page by page, each filled in order"""
    rows: list[int] = []
    while len(rows) < n:
        p = pool.alloc()
        take = min(PAGE, n - len(rows))
        rows.extend(p.id * PAGE + i for i in range(take))
        p.fill = take
    return rows


def let_go(pool: PagePool, rows: list[int]) -> None:
    """a table dropping its rows: one hold less on each page it read"""
    for p in pool.of(rows):
        pool.unref(p)


def test_a_shared_prefix_is_the_same_rows() -> None:
    """two conversations through one system prompt: the second's match is the first's rows, and its own tokens go in
    beside them, the node split where they part"""
    pool = PagePool()
    tree = RadixTree(pool)
    sysp = list(range(100, 200))
    a = [*sysp, 1, 2, 3]
    rows_a = table(pool, len(a))
    tree.insert(a, rows_a)
    m = tree.match([*sysp, 7, 8])
    assert m.length == len(sysp) and m.rows == rows_a[: len(sysp)]
    rows_b = m.rows + table(pool, 2)
    tree.insert([*sysp, 7, 8], rows_b)
    assert [n.end for n in tree.root.kids.values()] == [len(sysp)], "the shared prefix is not one node"
    (top,) = tree.root.kids.values()
    assert sorted(len(k.key) for k in top.kids.values()) == [2, 3]
    assert tree.match(a).rows == rows_a and tree.match([*sysp, 7, 8]).rows == rows_b
    assert not tree.check() and not pool.check()


def test_the_tree_keeps_its_rows_and_frees_with_its_last_holder() -> None:
    """what the tree holds keeps the rows it was given first; the pages live while a node or a table reads them,
    and evicting every leaf after the tables let go frees them all"""
    pool = PagePool()
    tree = RadixTree(pool)
    toks = list(range(150))
    first = table(pool, 150)
    tree.insert(toks, first)
    again = table(pool, 150)  # the same tokens prefilled once more elsewhere
    tree.insert(toks, again)
    assert tree.match(toks).rows == first
    let_go(pool, again)
    assert len(pool) == len(pool.of(first)), "the duplicate rows outlived their table"
    let_go(pool, first)
    assert len(pool) == len(pool.of(first)), "the tree's pages went with the table"
    assert tree.evict() == len(pool.of(first)) and len(pool) == 0
    assert not tree.root.kids and not pool.check()


def test_a_page_both_halves_of_a_split_read_is_held_by_each() -> None:
    """a split inside a page: both nodes hold it, so evicting one half leaves the other's rows standing"""
    pool = PagePool()
    tree = RadixTree(pool)
    toks = list(range(40))
    rows = table(pool, 40)  # one page
    tree.insert(toks, rows)
    let_go(pool, rows)
    (page,) = pool.of(rows)
    assert page.refs == 1
    tree.insert([*toks[:20], 99], [*rows[:20], *table(pool, 1)])
    assert page.refs == 2, "the split did not hold the page for both halves"
    tree.evict(enough=lambda freed: page.refs == 1 or freed > 0)
    assert page.refs >= 1 and tree.match(toks[:20]).rows == rows[:20]


def test_rows_the_tree_reads_are_frozen() -> None:
    """a page's rows the tree references are frozen: their writer may go on past them, never write them again"""
    pool = PagePool()
    tree = RadixTree(pool)
    rows = table(pool, 30)
    tree.insert(list(range(20)), rows[:20])
    (page,) = pool.of(rows)
    assert (page.fill, page.frozen) == (30, 20)


def test_snapshots_land_at_node_ends_and_the_deepest_within_a_match_is_found() -> None:
    """a snapshot kept after the system prompt and after the first turn: each splits the path there; a prompt
    parting inside the first turn resumes from the system prompt's, one parting after it from the turn's"""
    pool = PagePool()
    tree = RadixTree(pool)
    toks = list(range(300))
    tree.insert(toks, table(pool, 300), {100: "after-system", 220: "after-turn"})
    assert sorted(n.end for n in tree.nodes()) == [100, 220, 300]
    m = tree.match([*toks[:150], 999])
    assert (m.length, m.snap_at, m.snap) == (150, 100, "after-system")
    m = tree.match([*toks[:250], 999])
    assert (m.length, m.snap_at, m.snap) == (250, 220, "after-turn")
    assert tree.match([5, 6]).length == 0
    tree.insert(toks, table(pool, 300), {100: "another"})
    assert tree.match(toks).snap_at == 220 and tree.match(toks[:120]).snap == "after-system"
    assert tree.drop_snaps(lambda dropped: dropped >= 1) == 1
    assert sum(n.snap is not None for n in tree.nodes()) == 1


def test_the_least_recently_used_leaf_goes_first() -> None:
    """eviction takes the leaf used longest ago, and a parent left childless goes after, as a leaf of its own"""
    pool = PagePool()
    tree = RadixTree(pool)
    base = list(range(64))
    a, b = [*base, *range(1000, 1064)], [*base, *range(2000, 2064)]
    ra = table(pool, 128)
    rb = ra[:64] + table(pool, 64)
    tree.insert(a, ra)
    tree.insert(b, rb)
    let_go(pool, ra)
    for p in pool.of(rb[64:]):
        pool.unref(p)
    tree.match(a)  # a used last
    assert tree.evict(enough=lambda freed: freed >= 1) == 1
    assert tree.match(a).length == 128 and tree.match(b).length == 64
    assert tree.evict() == 2 and len(pool) == 0


def test_random_conversations_keep_the_books_straight() -> None:
    """many conversations sharing prefixes at random, tables let go as they end and leaves evicted at random: the
    tree and the pool stay consistent, every match's rows are the rows inserted for those tokens, and once all is
    let go no page is held"""
    rng = random.Random(7)
    pool = PagePool()
    tree = RadixTree(pool)
    truth: dict[tuple[int, ...], int] = {}
    for _ in range(200):
        n = rng.randint(1, 260)
        toks = [rng.choice((1, 2, 3)) for _ in range(rng.randint(0, 40))] + [rng.randint(0, 9) for _ in range(n)]
        m = tree.match(toks)
        rows = m.rows + table(pool, len(toks) - m.length)
        held = pool.of(m.rows)
        for p in held:
            pool.ref(p)  # the conversation's table reads the matched rows too
        tree.insert(toks, rows)
        for i, r in enumerate(tree.match(toks).rows):
            truth.setdefault(tuple(toks[: i + 1]), r)
            assert truth[tuple(toks[: i + 1])] == r
        let_go(pool, rows)
        if rng.random() < 0.3:
            tree.evict(enough=lambda freed: freed >= rng.randint(1, 4))
            truth = {k: v for k, v in truth.items() if tree.match(list(k)).length == len(k)}
        assert not tree.check() and not pool.check()
    tree.evict()
    assert len(pool) == 0 and not tree.root.kids
