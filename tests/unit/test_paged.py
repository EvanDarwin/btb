# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's rows in the engine's pool of pages (btb/engine/paged.py): written through a table and read back in
order through its map whatever pages they lie in; a prefix two caches share is the same rows, held once and never
rewritten; a crop takes back only what no one else reads; a verify's accepted path moved into place; the pool grown a
buffer at a time as the ledger grants, the tree's conversations let go first where it refuses; a table let go or
collected gives its pages back. Model-free: a config and random rows."""

from __future__ import annotations

import gc
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Qwen3Config

from btb.engine.kvpool import PAGE
from btb.engine.paged import HostPool, PagedCache, PagedError, PagedLayer, Table
from btb.engine.prefix import PrefixCache
from btb.engine.scheduler import MemoryGrantError
from btb.kinds import LayerKind

HK, D, L = 2, 8, 3
ROW = HK * PAGE * D * 4  # one page of one float32 buffer (a layer's K, or its V)


def prefix(grant: Callable[..., None] | None = None) -> PrefixCache:
    """a prefix cache over L full-attention layers of HK heads of D, its growth asked of `grant`"""
    cfg = Qwen3Config(
        num_hidden_layers=L,
        num_attention_heads=2 * HK,
        num_key_value_heads=HK,
        head_dim=D,
        hidden_size=32,
        intermediate_size=32,
        vocab_size=64,
    )
    sm = SimpleNamespace(
        cfg=cfg,
        layer_types=[LayerKind.FULL] * L,
        host_kv_dtype=lambda: torch.float32,
        scheduler=SimpleNamespace(grant=grant) if grant is not None else None,
    )
    return PrefixCache(cast(Any, sm))


def rows(T: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, HK, T, D, generator=g), torch.randn(1, HK, T, D, generator=g)


def write(c: PagedCache, k: torch.Tensor, v: torch.Tensor) -> None:
    """`k`, `v` after every layer's rows, layer i's offset by i so no two layers hold the same"""
    for i, cl in enumerate(c.layers):
        cast(PagedLayer, cl).append(k + i, v - i)


def read(c: PagedCache, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(PagedLayer, c.layers[i]).gather()


def test_rows_go_in_through_the_table_and_come_back_in_order() -> None:
    """rows past a page's end go on in the next page; each layer's come back in order through the map, as one
    contiguous buffer holds them; the layer is never read as one buffer, nor written past its first rows as one"""
    pc = prefix()
    c = pc.new()
    k1, v1 = rows(PAGE + 10, 1)
    k2, v2 = rows(PAGE - 3, 2)
    write(c, k1, v1)
    write(c, k2, v2)
    n = 2 * PAGE + 7
    assert c.get_seq_length() == n == len(c.table) and c.table.rows().tolist() == list(range(n))
    for i in range(L):
        gk, gv = read(c, i)
        assert torch.equal(gk, torch.cat([k1, k2], -2) + i) and torch.equal(gv, torch.cat([v1, v2], -2) - i)
    assert len(pc.pool.pages) == 3 and not pc.pool.pages.check()
    cl = cast(PagedLayer, c.layers[0])
    with pytest.raises(PagedError):
        _ = cl.keys
    with pytest.raises(PagedError):
        cl.update(k1, v1)


def test_a_fresh_layers_first_rows_serve_a_module_as_they_came() -> None:
    """a module's first write to a fresh layer gets its rows back in the pool's dtype (a contiguous layer's own),
    kept there; rows of another shape are refused"""
    pool = HostPool([0], HK, D, dtype=torch.bfloat16)
    layer = PagedLayer(Table(pool), pool, 0)
    k, v = rows(3, 3)
    gk, gv = layer.update(k, v)
    assert gk.dtype == gv.dtype == torch.bfloat16 and torch.equal(gk, k.to(torch.bfloat16))
    assert torch.equal(layer.gather()[1], v.to(torch.bfloat16))
    wide = torch.zeros(1, HK + 1, 3, D)
    with pytest.raises(PagedError):
        PagedLayer(Table(pool), pool, 0).append(wide, wide)


def test_a_shared_prefix_is_the_same_rows_held_once() -> None:
    """a cache opened on the tree's rows reads the very rows another conversation wrote - a page counted once
    however many hold it - and each goes on in pages of its own, never over the rows the other reads"""
    pc = prefix()
    a = pc.new()
    k, v = rows(100, 5)
    write(a, k, v)
    ids = list(range(1, 101))
    a.commit(ids)
    m = pc.tree.match(ids[:80])
    assert m.length == 80 and m.rows == a.table.rows()[:80].tolist()
    b = pc.new(m.rows)
    assert b.get_seq_length() == 80 and len(pc.pool.pages) == 2
    assert [p.refs for p in pc.pool.pages.pages] == [3, 3], "a's table, the tree's node and b's table each hold one"
    k2, v2 = rows(5, 6)
    write(b, k2, v2)
    assert len(pc.pool.pages) == 3, "b wrote into a page a's rows are in"
    k3, v3 = rows(3, 7)
    write(a, k3, v3)
    assert len(pc.pool.pages) == 3, "a took a new page where its own had room past the rows the tree froze"
    for i in range(L):
        assert torch.equal(read(b, i)[0], torch.cat([k[..., :80, :], k2], -2) + i)
        assert torch.equal(read(a, i)[1], torch.cat([v, v3], -2) - i)
    assert not pc.pool.pages.check() and not pc.tree.check()


def test_a_crop_takes_back_only_rows_no_one_else_reads() -> None:
    """cut back, a table writes again over its own rows no other holder reads; over rows the tree froze it never
    does - it goes on in a new page, the tree's rows as they were"""
    pc = prefix()
    a = pc.new()
    k, v = rows(100, 8)
    write(a, k, v)
    a.crop_to(90)
    k2, v2 = rows(5, 9)
    write(a, k2, v2)
    assert len(pc.pool.pages) == 2 and a.table.rows()[90:].tolist() == list(range(PAGE + 26, PAGE + 31))
    a.commit(list(range(95)))
    a.crop_to(90)
    write(a, *rows(5, 10))
    assert len(pc.pool.pages) == 3, "a crop let rows the tree reads be written again"
    m = pc.tree.match(list(range(95)))
    assert m.length == 95
    assert torch.equal(pc.pool.k[0][:, torch.tensor(m.rows)][None], torch.cat([k[..., :90, :], k2], -2))
    a.crop(-3)  # transformers' crop, from the end: the table cut with the layers
    assert a.get_seq_length() == len(a.table) == 92
    assert not pc.pool.pages.check()


def test_a_verify_keeps_its_accepted_path_in_place() -> None:
    """a verify pass's nodes after the prefix: the accepted path's rows moved down to its positions in every layer,
    the rest let go; a chain's path is a crop"""
    pc = prefix()
    a = pc.new()
    k, v = rows(10, 11)
    write(a, k, v)
    kt, vt = rows(4, 12)
    write(a, kt, vt)
    a.keep_path(10, [0, 2])
    assert a.get_seq_length() == 12 == len(a.table)
    for i in range(L):
        gk, gv = read(a, i)
        assert torch.equal(gk, torch.cat([k, kt[..., [0, 2], :]], -2) + i)
        assert torch.equal(gv, torch.cat([v, vt[..., [0, 2], :]], -2) - i)
    write(a, *rows(2, 13))
    a.keep_path(12, [0])
    assert a.get_seq_length() == 13 and torch.equal(
        read(a, 1)[0][..., :12, :], torch.cat([k, kt[..., [0, 2], :]], -2) + 1
    )


def test_a_table_let_go_or_collected_gives_its_pages_back() -> None:
    """`release` gives every page back at once; a table held again after one, or never released, gives them back
    when it is collected"""
    pc = prefix()
    a = pc.new()
    write(a, *rows(70, 14))
    assert len(pc.pool.pages) == 2
    a.release()
    assert len(pc.pool.pages) == 0 and a.get_seq_length() == 0
    write(a, *rows(3, 15))
    b = pc.new()
    write(b, *rows(3, 16))
    assert len(pc.pool.pages) == 2
    del a, b
    gc.collect()
    assert len(pc.pool.pages) == 0 and not pc.pool.pages.check()


def test_the_pool_grows_a_buffer_at_a_time_and_lets_the_tree_go_when_refused() -> None:
    """the pool's growth is asked of the ledger a layer's buffer at a time, each with the one it replaces; its
    figure (`growth`) is 0 while the tail and the free pages hold the rows. Refused, the tree's least recently used
    conversation goes and its pages take the rows"""
    asked: list[tuple[int, int]] = []
    refuse = [False]

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        assert kind == "kv" and kw["device"] == "cpu"
        asked.append((nbytes, int(kw["held"])))
        if refuse[0]:
            raise MemoryGrantError("refused")

    pc = prefix(grant)
    a = pc.new()
    first = HostPool.MIN_PAGES
    assert a.growth(1) == 2 * L * first * ROW, "a first growth takes what the regions then hold"
    write(a, *rows(1, 17))
    assert asked == [(first * ROW, 0)] * (2 * L) and pc.pool.cap == first
    assert pc.pool.nbytes() == 2 * L * first * ROW
    room = (PAGE - 1) + (first - 1) * PAGE
    new = int(first * HostPool.GROW)
    assert a.table.pages_for(room) == first - 1 and a.growth(room) == 0
    # what the regions add, and one buffer held while its replacement fills
    assert a.growth(room + 1) == 2 * L * (new - first) * ROW + first * ROW
    write(a, *rows(room, 18))
    assert len(asked) == 2 * L, "a growth was asked where the pool held the rows"
    a.commit(list(range(a.get_seq_length())))
    a.release()
    assert len(pc.pool.pages) == first, "the tree holds the conversation's pages"
    refuse[0] = True
    b = pc.new()
    write(b, *rows(PAGE + 1, 19))
    assert len(asked) == 2 * L + 1 and pc.pool.cap == first
    assert not list(pc.tree.nodes()) and len(pc.pool.pages) == 2 and not pc.pool.pages.check()
    with pytest.raises(MemoryGrantError):
        write(b, *rows(first * PAGE, 20))


def test_room_before_a_pass_is_made_of_the_trees_conversations_only_where_growth_is_refused() -> None:
    """`room` leaves the tree alone where the growth fits the free memory, and lets its least recently used
    conversations go, one at a time, where it does not - until the free pages hold the rows"""
    pc = prefix(lambda *_a, **_k: None)
    olds = []
    for j in range(3):
        c = pc.new()
        write(c, *rows(PAGE * 5, 21 + j))
        c.commit([100 * j + t for t in range(PAGE * 5)])
        olds.append(c)
    for c in olds:
        c.release()
    assert len(pc.pool.pages) == 15 and pc.pool.cap == HostPool.MIN_PAGES
    pc.tree.match([100])  # the second conversation used last
    b = pc.new()
    need = PAGE * 6  # one free page past the 15 the tree holds: five more wanted
    pc.room(b, need, lambda: 1 << 40)
    assert len(list(pc.tree.nodes())) == 3, "a conversation went where the growth fit"
    pc.room(b, need, lambda: 0)
    left = {n.key[0] for n in pc.tree.nodes()}
    assert left == {100, 200}, f"not the least recently used went first: {left}"
    assert b.growth(need) == 0


def test_a_closed_prefix_cache_holds_nothing() -> None:
    pc = prefix()
    c = pc.new()
    write(c, *rows(10, 30))
    c.commit(list(range(10)))
    c.release()
    pc.close()
    assert not list(pc.tree.nodes()) and len(pc.pool.pages) == 0 and not pc.pool.k and pc.pool.cap == 0
