# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's rows in the engine's pool of pages (btb/engine/paged.py): written through a table and read back in
order through its map whatever pages they lie in; a prefix two caches share is the same rows, held once and never
rewritten; a crop takes back only what no one else reads; a verify's accepted path moved into place; the pool grown a
buffer at a time as the ledger grants, the tree's conversations let go first where it refuses; a table let go or
collected gives its pages back. The card's region: the bound conversation's pages in its slots and every other parked,
its map uploaded only where it moved, a layer's rows following it between the regions - run here over arenas on the
host, the card's own layout. Model-free: a config and random rows."""

from __future__ import annotations

import gc
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import Qwen3Config

from btb.engine.kvpool import PAGE
from btb.engine.paged import CardRegion, HostRegion, KvPool, PagedCache, PagedError, PagedKV, PagedLayer, Table
from btb.engine.prefix import PrefixCache
from btb.engine.scheduler import MemoryGrantError
from btb.kinds import LayerKind

HK, D, L = 2, 8, 3
ROW = HK * PAGE * D * 4  # one page of one float32 buffer (a layer's K, or its V)


def prefix(grant: Callable[..., None] | None = None, card: tuple[int, ...] = ()) -> PrefixCache:
    """a prefix cache over L full-attention layers of HK heads of D, its growth asked of `grant`; the layers `card`
    in a card region whose arenas lie on the host (the card's layout and bookkeeping, no card needed)"""
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
        dev=torch.device("cpu"),
        resident={},
        layer_types=[LayerKind.FULL] * L,
        host_kv_dtype=lambda: torch.float32,
        scheduler=SimpleNamespace(grant=grant) if grant is not None else None,
    )
    pc = PrefixCache(cast(Any, sm))
    if card:
        pool = pc.pool
        for i in card:
            pool.host.drop(i)
        pool.card = CardRegion(card, HK, D, torch.device("cpu"), grant, PAGE, pool.pages.lock)
    return pc


def bf(t: torch.Tensor) -> torch.Tensor:
    """rows as the card's arenas keep them: bf16"""
    return t.to(torch.bfloat16)


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
    pool = KvPool([0], HK, D, dtype=torch.bfloat16)
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
    assert torch.equal(pc.pool.host.k[0][:, torch.tensor(m.rows)][None], torch.cat([k[..., :90, :], k2], -2))
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
    first = HostRegion.MIN_PAGES
    assert a.growth(1) == {"cpu": 2 * L * first * ROW}, "a first growth takes what the regions then hold"
    write(a, *rows(1, 17))
    assert asked == [(first * ROW, 0)] * (2 * L) and pc.pool.host.cap == first
    assert pc.pool.nbytes() == {"cpu": 2 * L * first * ROW}
    room = (PAGE - 1) + (first - 1) * PAGE
    new = int(first * HostRegion.GROW)
    assert a.table.pages_for(room) == first - 1 and a.growth(room) == {}
    # what the regions add, and one buffer held while its replacement fills
    assert a.growth(room + 1) == {"cpu": 2 * L * (new - first) * ROW + first * ROW}
    write(a, *rows(room, 18))
    assert len(asked) == 2 * L, "a growth was asked where the pool held the rows"
    a.commit(list(range(a.get_seq_length())))
    a.release()
    assert len(pc.pool.pages) == first, "the tree holds the conversation's pages"
    refuse[0] = True
    b = pc.new()
    write(b, *rows(PAGE + 1, 19))
    assert len(asked) == 2 * L + 1 and pc.pool.host.cap == first
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
    assert len(pc.pool.pages) == 15 and pc.pool.host.cap == HostRegion.MIN_PAGES
    pc.tree.match([100])  # the second conversation used last
    b = pc.new()
    need = PAGE * 6  # one free page past the 15 the tree holds: five more wanted
    pc.room(b, need, lambda: 1 << 40)
    assert len(list(pc.tree.nodes())) == 3, "a conversation went where the growth fit"
    pc.room(b, need, lambda: 0)
    left = {n.key[0] for n in pc.tree.nodes()}
    assert left == {100, 200}, f"not the least recently used went first: {left}"
    assert b.growth(need) == {}


def test_a_closed_prefix_cache_holds_nothing() -> None:
    """closed, the cache holds nothing; a conversation outliving it lets its pages go after with nothing left to free
    on the card (a session collected after its engine closed)"""
    pc = prefix(card=(0,))
    c, late = pc.new(), pc.new()
    write(c, *rows(10, 30))
    write(late, *rows(70, 31))
    c.commit(list(range(10)))
    c.release()
    pc.close()
    assert not list(pc.tree.nodes()) and len(pc.pool.pages) == 2 and not pc.pool.host.k and pc.pool.host.cap == 0
    late.release()
    assert len(pc.pool.pages) == 0 and all(p.slot < 0 and p.park < 0 for p in pc.pool.pages.pages)


def card_rows(c: PagedCache, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    """layer i's rows of `c` read as the card's kernels read them: its arena's rows at the card's map of the table
    (the conversation bound first), [1, Hk, n, D]"""
    card = c.prefix.pool.card
    assert card is not None
    tbl = card.bind(c.table)[: len(c.table)].long()
    k, v = card.view(i)
    return k[:, :, tbl], v[:, :, tbl]


def test_the_card_holds_the_bound_conversations_pages_and_parks_the_rest() -> None:
    """the card runs layers 0 and 1, the host layer 2: a conversation's rows of the card's layers go into its slots,
    read back through the card's map as written; another conversation bound parks the first one's own pages - a
    prefix the two share stays - and the first bound again brings them back, every row as it was, the map made again"""
    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None and card.layers == [0, 1] and pc.pool.host.layers == [2]
    a = pc.new()
    k, v = rows(2 * PAGE + 5, 40)
    write(a, k, v)
    a.commit(list(range(2 * PAGE + 5)))
    assert len(card.slots) >= 3 and not card.parked, "a's pages on the card, none parked"
    for i in (0, 1):
        assert torch.equal(read(a, i)[0], bf(k + i)) and torch.equal(read(a, i)[1], bf(v - i))
        ck, cv = card_rows(a, i)
        assert torch.equal(ck, bf(k + i)) and torch.equal(cv, bf(v - i))
    assert torch.equal(read(a, 2)[0], k + 2), "the host's layer in the host's region, as written"
    m = pc.tree.match(list(range(PAGE + 3)))
    b = pc.new(m.rows)  # b shares a's first page whole and three rows of its second
    k2, v2 = rows(10, 41)
    ver = card.version
    write(b, k2, v2)  # b's first card layer's rows bind b
    parked = {p.id for p in pc.pool.pages.pages if p.park >= 0}
    assert parked == {2}, f"a's own third page parked, the two it shares with b kept: {parked}"
    assert card.version > ver
    for i in (0, 1):
        ck, _ = card_rows(b, i)
        assert torch.equal(ck, bf(torch.cat([k[..., : PAGE + 3, :], k2], -2) + i))
        assert torch.equal(read(a, i)[0], bf(k + i)), "a's rows read back from the park"
    for i in (0, 1):
        ck, cv = card_rows(a, i)
        assert torch.equal(ck, bf(k + i)) and torch.equal(cv, bf(v - i)), "a's rows brought back as they were"
    assert all(p.park < 0 for p in a.table.held.values())
    assert not pc.pool.pages.check()


def test_a_conversation_let_go_while_the_card_parks_frees_its_slots_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """a conversation collected while the card moves its pages - the collector run by an allocation of the copies, on
    the same thread under the region's reentrant lock - lets them go once the move is done: each page keeps the slots
    the move reads and writes, and then every slot it held is free again, once"""
    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None
    a, c = pc.new(), pc.new()
    write(a, *rows(PAGE + 7, 49))  # a bound: its two pages on the card
    held = list(a.table.held.values())
    assert all(p.slot >= 0 for p in held)
    copy = card._copy

    def collected_mid_copy(src: dict[int, Any], dst: dict[int, Any], runs: list[tuple[int, int, int]]) -> None:
        if a.table.held:
            a.release()  # as a's finalizer would, run by the collector inside the move
        copy(src, dst, runs)

    monkeypatch.setattr(card, "_copy", collected_mid_copy)
    kc, vc = rows(9, 50)
    write(c, kc, vc)  # c bound: a's pages parked, a let go meanwhile
    assert all(p.refs == 0 and p.slot < 0 and p.park < 0 for p in held), "a's pages left a slot behind"
    assert not any(p in held for p in [*card.slots, *card.parked] if p is not None)
    assert len(set(card.free)) == len(card.free) and len(set(card.pfree)) == len(card.pfree), "a slot freed twice"
    assert min([*card.free, *card.pfree], default=0) >= 0
    assert not pc.pool.pages.check()
    for i in (0, 1):
        ck, cv = card_rows(c, i)
        assert torch.equal(ck, bf(kc + i)) and torch.equal(cv, bf(vc - i))


def test_the_cards_map_uploads_only_the_positions_that_moved() -> None:
    """while a table stays bound, an append uploads its new positions alone, a crop then the positions written again;
    a card layer's module past its first rows is handed the rows through the map (`PagedKV`), never a buffer"""
    pc = prefix(card=(0, 1, 2))
    card = pc.pool.card
    assert card is not None
    a = pc.new()
    write(a, *rows(5, 42))
    tbl = card.bind(a.table)
    assert a.table.low == len(a.table) == 5
    tbl[:5] = -7  # stale entries the next upload must leave alone
    write(a, *rows(3, 43))
    assert tbl[:5].tolist() == [-7] * 5 and tbl[5:8].tolist() == card.rows(a.table.rows()[5:8]).tolist()
    a.crop_to(6)
    write(a, *rows(2, 44))
    assert tbl[:5].tolist() == [-7] * 5 and tbl[6:8].tolist() == card.rows(a.table.rows()[6:8]).tolist()
    layer = cast(PagedLayer, a.layers[1])
    kv, kv2 = layer.update(*rows(1, 45))
    assert isinstance(kv, PagedKV) and kv is kv2 and (kv.n0, kv.T) == (8, 1)


def test_a_layer_moving_between_the_regions_takes_every_conversations_rows() -> None:
    """a layer shed to the host takes every conversation's rows of it - the bound one's from the card, a parked one's
    from the park - into the host's region, and back onto the card when it regrows: the same rows, read the same"""
    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None
    a, b = pc.new(), pc.new()
    ka, va = rows(PAGE + 7, 46)
    kb, vb = rows(20, 47)
    write(a, ka, va)
    write(b, kb, vb)
    card.bind(b.table)  # a's pages parked
    assert any(p.park >= 0 for p in a.table.held.values())
    pc.pool.rehome(1, card=False)
    assert card.layers == [0] and 1 in pc.pool.host.layers and not pc.pool.on_card(1)
    for c, k, v in ((a, ka, va), (b, kb, vb)):
        got = read(c, 1)
        assert torch.equal(got[0], bf(k + 1).float()) and torch.equal(got[1], bf(v - 1).float())
    write(b, *rows(2, 48))  # the shed layer's new rows into the host's region
    pc.pool.rehome(1, card=True)
    assert card.layers == [0, 1] and 1 not in pc.pool.host.layers
    for c, k in ((a, ka), (b, kb)):
        assert torch.equal(read(c, 1)[0][..., : k.shape[-2], :], bf(k + 1))
    assert torch.equal(card_rows(a, 1)[0], bf(ka + 1))
