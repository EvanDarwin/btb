# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's rows in the engine's pool of pages (btb/engine/paged.py): written through a table and read back in
order through its map whatever pages they lie in; a prefix two caches share is the same rows, held once and never
rewritten; a crop takes back only what no one else reads; a verify's accepted path moved into place; the pool grown a
buffer at a time as the ledger grants, the tree's conversations let go first where it refuses; a table let go or
collected gives its pages back. The card's region: the bound conversation's pages in its slots, another's parked only
when their slots are wanted, its map uploaded only where it moved, a layer's rows following it between the regions -
run here over arenas on the host, the card's own layout. Model-free: a config and random rows."""

from __future__ import annotations

import gc
import weakref
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


def prefix(
    grant: Callable[..., None] | None = None, card: tuple[int, ...] = (), types: list[LayerKind] | None = None
) -> PrefixCache:
    """a prefix cache over L full-attention layers of HK heads of D, its growth asked of `grant`; the layers `card`
    in a card region whose arenas lie on the host (the card's layout and bookkeeping, no card needed). `types`: the
    layers' kinds instead (a hybrid's: its linear layers have no rows in the pool)"""
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
        layer_types=types or [LayerKind.FULL] * L,
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


def test_a_hop_given_up_leaves_the_layer_reading_its_region() -> None:
    """a host layer hopped onto a prefill's buffers whose sweep failed before landing it gives the hop up: its rows
    read and written in the host's region again, the chunks' rows in the buffers never made (the failed pass's, cut by
    its rollback) - left hopped, every later pass appended into the dead buffers and was refused"""
    pool = KvPool([0], HK, D, dtype=torch.float32)
    table = Table(pool)
    layer = PagedLayer(table, pool, 0)
    k, v = rows(5, 60)
    layer.update(k, v)
    kb, vb = torch.zeros(1, HK, 16, D), torch.zeros(1, HK, 16, D)
    layer.hop(kb, vb)
    layer.append(*rows(3, 61))  # a chunk into the hop's buffers
    layer.unhop()
    table.crop(5)  # the rollback's cut
    assert layer._hop is None and layer.get_seq_length() == 5
    k2, v2 = rows(2, 62)
    gk, _ = layer.append(k2, v2)
    assert gk.untyped_storage().data_ptr() != kb.untyped_storage().data_ptr(), "still appending into the hop's buffers"
    got = layer.gather()
    assert torch.equal(got[0], torch.cat([k, k2], -2)) and torch.equal(got[1], torch.cat([v, v2], -2))


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
    """cut back, a table writes again over its own rows no other holder reads - its commit too, while the tree has
    not read it yet (held back, cut with the table: a mark/feed/rewind loop takes no page a turn); over rows the tree
    took it never does - it goes on in a new page, the tree's rows as they were"""
    pc = prefix()
    a = pc.new()
    k, v = rows(100, 8)
    write(a, k, v)
    a.crop_to(90)
    k2, v2 = rows(5, 9)
    write(a, k2, v2)
    assert len(pc.pool.pages) == 2 and a.table.rows()[90:].tolist() == list(range(PAGE + 26, PAGE + 31))
    for turn in range(3):  # committed, then rewound before anything read the tree: the same page, written again
        a.commit(list(range(95)))
        a.crop_to(90)
        write(a, *rows(5, 20 + turn))
        assert len(pc.pool.pages) == 2, f"turn {turn}: a rewound commit left its page to the tree"
    a.crop_to(90)
    write(a, k2, v2)
    a.commit(list(range(95)))
    m = pc.tree.match(list(range(95)))  # the tree read: the commit in it, its rows frozen
    assert m.length == 95
    a.crop_to(90)
    write(a, *rows(5, 10))
    assert len(pc.pool.pages) == 3, "a crop let rows the tree reads be written again"
    assert torch.equal(pc.pool.host.k[0][:, torch.tensor(m.rows)][None], torch.cat([k[..., :90, :], k2], -2))
    a.crop(-3)  # transformers' crop, from the end: the table cut with the layers
    assert a.get_seq_length() == len(a.table) == 92
    assert not pc.pool.pages.check()


def test_a_commit_reaches_the_tree_when_it_is_read() -> None:
    """a commit held back goes in when the tree is next read - another prompt's match, an eviction - or the session
    lets its cache go; the session asking keeps its own out of its match (what it parts from it keeps or cuts itself);
    a later commit replaces the one held, a crop cuts it"""
    pc = prefix()
    a, b = pc.new(), pc.new()
    write(a, *rows(30, 30))
    a.commit(list(range(30)))
    assert not list(pc.tree.nodes()), "held back until the tree is read"
    assert pc.match(list(range(30)), asking=a).length == 0, "the asking session's own commit left out of its match"
    assert pc.match(list(range(30))).length == 30, "in the tree once another reads it"
    write(b, *rows(40, 31))
    b.commit([100 + t for t in range(40)])
    b.crop_to(25)
    b.release()  # let go: what it committed - cut to 25 - in the tree first
    m = pc.tree.match([100 + t for t in range(40)])
    assert m.length == 25
    c = pc.new()
    write(c, *rows(20, 32))
    c.commit([200 + t for t in range(20)])
    gone = weakref.ref(c)
    del c
    gc.collect()
    assert gone() is None, "a session's cache let go is collected, its commit held on the table alone"
    assert pc.match([200 + t for t in range(20)]).length == 20, "and the commit reaches the tree all the same"
    assert pc.tree.evict() > 0 and not list(pc.tree.nodes())
    assert not pc.pool.pages.check()


def test_rows_made_off_the_route_never_reach_the_tree() -> None:
    """a conversation whose rows from some position were made off the route its steps take (a card layer run through
    torch, out of room on the card) reads them itself, and the tree is given the rows before them alone: a hit on
    them would decode other bits than its prompt cold"""
    pc = prefix()
    a = pc.new()
    write(a, *rows(50, 70))
    a.off_route(40)
    a.off_route(30)  # the earliest wins
    a.commit(list(range(50)))
    assert pc.match(list(range(50))).length == 30
    assert a.get_seq_length() == 50
    # cut back past them (a rewind, a regenerate) and made again on the route: the tree is given them, as it would
    # have been had the fallback never run - cut short of them, the rows past the cut stay out
    b = pc.new()
    write(b, *rows(50, 71))
    b.off_route(40)
    b.crop_to(45)
    b.commit([500 + t for t in range(45)])
    assert pc.match([500 + t for t in range(45)]).length == 40, "a crop short of the fallback's rows lost the mark"
    b.crop_to(20)
    write(b, *rows(30, 72))
    again = [*range(500, 520), *range(600, 630)]
    b.commit(again)
    assert pc.match(again).length == 50, "rows remade on the route kept out of the tree"
    assert not pc.pool.pages.check() and not pc.tree.check()


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


def test_the_card_parks_another_conversations_pages_only_when_their_slots_are_wanted() -> None:
    """the card runs layers 0 and 1, the host layer 2: a conversation's rows of the card's layers go into its slots,
    read back through the card's map as written. Another conversation bound leaves the first one's pages where they
    are while the card has room for both - nothing moves; a third filling the card parks the least recently used page
    it does not read for its room, one move for all its pages; the first bound again trades its parked page for one
    it does not read - every row as it was, the map made again, neither the card nor the park grown for the trade"""
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
    assert not any(p.park >= 0 for p in pc.pool.pages.pages) and card.version == ver, "room for both: nothing moved"
    for i in (0, 1):
        ck, _ = card_rows(b, i)
        assert torch.equal(ck, bf(torch.cat([k[..., : PAGE + 3, :], k2], -2) + i))
    free = len(card.free)
    c = pc.new()
    k3, v3 = rows(free * PAGE + 1, 42)  # a page more than the free slots hold
    write(c, k3, v3)
    parked = [p.id for p in pc.pool.pages.pages if p.park >= 0]
    # a's own last page, unread since a was bound: the page a and b share b's binding used since
    assert parked == [2], f"the least recently used page c does not read parked for its room: {parked}"
    for i in (0, 1):
        assert torch.equal(read(a, i)[0], bf(k + i)), "a's rows read back from the park"
        assert torch.equal(card_rows(c, i)[0], bf(k3 + i))
    cap, park = card.cap, len(card.parked)
    for i in (0, 1):
        ck, cv = card_rows(a, i)
        assert torch.equal(ck, bf(k + i)) and torch.equal(cv, bf(v - i)), "a's rows brought back as they were"
    assert card.cap == cap and len(card.parked) == park, "the trade grew neither the card nor the park"
    assert all(p.park < 0 for p in a.table.held.values())
    for i in (0, 1):
        assert torch.equal(read(b, i)[0], bf(torch.cat([k[..., : PAGE + 3, :], k2], -2) + i)), "b's page traded"
        assert torch.equal(read(c, i)[0], bf(k3 + i))
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
    assert all(p.slot >= 0 for p in a.table.held.values())
    copy = card._copy

    def collected_mid_copy(src: dict[int, Any], dst: dict[int, Any], runs: list[tuple[int, int, int]]) -> None:
        if a.table.held:
            a.release()  # as a's finalizer would, run by the collector inside the move
        copy(src, dst, runs)

    monkeypatch.setattr(card, "_copy", collected_mid_copy)
    kc, vc = rows(len(card.free) * PAGE + 1, 50)  # a page more than the free slots hold
    write(c, kc, vc)  # c bound: a page of a's parked for its room, a let go meanwhile
    assert not a.table.held and not region_check(card)
    assert not pc.pool.pages.check()
    for i in (0, 1):
        ck, cv = card_rows(c, i)
        assert torch.equal(ck, bf(kc + i)) and torch.equal(cv, bf(vc - i))


def test_a_binding_grows_the_card_or_the_park_only_where_its_price_said() -> None:
    """what a pass's binding is priced at before the pass (`CardRegion.need`, the ledger's ask) is what it then grows:
    a switch back to a conversation whose pages fill the park, while another's fill the card, trades them - priced at
    nothing and growing nothing (it grew the park by every page it parked) - and new pages past every slot the card
    can free are priced and grown"""
    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None
    a, b = pc.new(), pc.new()
    ka, va = rows(8 * PAGE, 60)
    write(a, ka, va)
    kb, vb = rows(len(card.free) * PAGE + 6 * PAGE, 61)  # the card full: six of a's pages parked for b's
    write(b, kb, vb)
    assert sum(p.park >= 0 for p in a.table.held.values()) == 6 and not card.free

    def bind(c: PagedCache, rows_more: int) -> tuple[tuple[int, int], tuple[bool, bool]]:
        price = card.need(c.table, c.table.pages_for(rows_more))
        cap, park = card.cap, len(card.parked)
        card.bind_for(c.table, len(c.table) + rows_more)
        assert not region_check(card)
        return price, (card.cap > cap, len(card.parked) > park)

    assert bind(a, 0) == ((0, 0), (False, False)), "a switch back trades a's parked pages for b's"
    assert all(p.slot >= 0 for p in a.table.held.values())
    (grow, _), grown = bind(a, (card.cap + 1) * PAGE)  # more than every slot the card can free
    assert grow > 0 and grown[0], "new pages past the card's room priced and grown"
    for c, k in ((a, ka), (b, kb)):
        assert torch.equal(read(c, 1)[0][..., : k.shape[-2], :], bf(k + 1)), "every row as it was written"
    assert not pc.pool.pages.check()


def region_check(card: CardRegion) -> list[str]:
    """the card region's slots against its pages: every slot holding a live page that names it back, the free lists
    the empty slots once each"""
    bad: list[str] = []
    for where, slots, free, at in (("card", card.slots, card.free, "slot"), ("park", card.parked, card.pfree, "park")):
        empty = {s for s, p in enumerate(slots) if p is None}
        if sorted(free) != sorted(empty):
            bad.append(f"the {where}'s free slots {sorted(free)} are not its empty ones {sorted(empty)}")
        for s, p in enumerate(slots):
            if p is not None and (p.refs <= 0 or getattr(p, at) != s):
                bad.append(f"{where} slot {s}: page {p.id} ({p.refs} holders) names slot {getattr(p, at)}")
    return bad


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
    write(a, ka, va)
    kb, vb = rows((len(card.free) + 1) * PAGE, 47)  # more pages than the free slots: one of a's parked for them
    write(b, kb, vb)
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


def test_a_conversation_let_go_while_a_layer_moves_frees_its_pages_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """a conversation collected while a layer's rows move between the regions (the collector run by an allocation of
    the copies, on the same thread): its pages keep the slots the copies read until the move is done, then free once
    - the layer moves whole, the other conversation's rows as they were, every way"""
    from btb.engine import paged

    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None
    b = pc.new()
    kb, vb = rows(PAGE + 9, 85)
    write(b, kb, vb)
    real = paged._page_rows
    for to_card in (False, True):
        victim = pc.new()
        write(victim, *rows(2 * PAGE + 3, 86))  # bound last: its pages before b's in no run of slots
        assert all(p.slot >= 0 for p in victim.table.held.values())

        def collected(starts: Any, victim: PagedCache = victim) -> torch.Tensor:
            if victim.table.held:
                victim.release()  # as its finalizer would, run by the collector inside the move
            return real(starts)

        monkeypatch.setattr(paged, "_page_rows", collected)
        pc.pool.rehome(1, card=to_card)
        monkeypatch.setattr(paged, "_page_rows", real)
        assert pc.pool.on_card(1) == to_card and (1 in pc.pool.host.layers) != to_card
        assert not victim.table.held and not region_check(card) and not pc.pool.pages.check()
        got = read(b, 1)
        assert torch.equal(got[0].to(torch.bfloat16), bf(kb + 1)) and torch.equal(got[1].to(torch.bfloat16), bf(vb - 1))


def test_a_layer_move_refused_part_way_leaves_its_rows_where_they_were() -> None:
    """a layer's move asks every grant before it makes anything, and registers what it made only once the rows are
    in: refused, the layer's rows are where they were - the host's region whole, no arena of no rows on the card for
    the next move to take as made - and asked again it moves them. A layer leaving the card stages its rows in RAM
    alone: nothing asked of the card, whose room is what the shed makes"""
    refuse: list[str] = []  # the requests refused, by what their requester says
    asked: list[tuple[str, Any]] = []

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        asked.append((str(kw.get("requester")), kw.get("device")))
        if any(r in str(kw.get("requester")) for r in refuse):
            raise MemoryGrantError("refused", device=kw.get("device"))

    pc = prefix(grant, card=(0, 1))
    card = pc.pool.card
    assert card is not None
    a = pc.new()
    k, v = rows(2 * PAGE + 5, 87)
    write(a, k, v)
    refuse[:] = ["a run at a time"]
    with pytest.raises(MemoryGrantError):
        pc.pool.rehome(1, card=False)  # its staging refused: nothing made on the host
    assert pc.pool.on_card(1) and 1 not in pc.pool.host.layers and 1 not in pc.pool.host.k, "a host region left"
    refuse.clear()
    asked.clear()
    pc.pool.rehome(1, card=False)
    assert not pc.pool.on_card(1) and 1 in pc.pool.host.layers
    staged = [dev for who, dev in asked if "a run at a time" in who]
    assert staged == ["cpu"], f"a shed staged its rows on the card: {asked}"
    refuse[:] = ["card arena for layer 1"]
    with pytest.raises(MemoryGrantError):
        pc.pool.rehome(1, card=True)  # the arena refused: nothing made
    assert 1 not in card.arenas and 1 not in card.parks and 1 in pc.pool.host.layers
    refuse[:] = ["a run at a time"]
    with pytest.raises(MemoryGrantError):
        pc.pool.rehome(1, card=True)  # the staging refused after the arena's grant: still nothing made
    assert 1 not in card.arenas and 1 not in card.parks and 1 in pc.pool.host.k
    refuse.clear()
    pc.pool.rehome(1, card=True)
    assert pc.pool.on_card(1) and 1 not in pc.pool.host.layers
    assert torch.equal(card_rows(a, 1)[0], bf(k + 1)) and torch.equal(card_rows(a, 1)[1], bf(v - 1))


def test_the_host_reads_the_park_only_once_the_copies_into_it_landed(monkeypatch: pytest.MonkeyPatch) -> None:
    """the park's copies run on the card's stream, which the host's own reads of the park do not wait for: a gather
    of parked rows waits for the copies queued into the park first (a batch re-formed after its write-back read a
    trade's rows before they landed: another conversation's keys), and nothing is waited for once they have"""
    pc = prefix(card=(0, 1))
    card = pc.pool.card
    assert card is not None
    waits: list[bool] = []
    settle = card._settle

    def watch() -> None:
        waits.append(card._queued)
        settle()

    monkeypatch.setattr(card, "_settle", watch)
    a, b = pc.new(), pc.new()
    ka, va = rows(PAGE + 7, 88)
    write(a, ka, va)
    write(b, *rows((len(card.free) + 2) * PAGE, 89))  # both of a's pages parked for b's room
    assert all(p.park >= 0 for p in a.table.held.values()) and card._queued
    for i in (0, 1):
        got = read(a, i)
        assert torch.equal(got[0], bf(ka + i)) and torch.equal(got[1], bf(va - i))
    assert waits and waits[-1] is False and True in waits and not card._queued, waits


def test_a_hybrids_commit_reaches_the_tree_as_far_as_a_snapshot_made_cold() -> None:
    """a hybrid's rows go to the tree only as far as its deepest snapshot of states made cold reaches: a prefill call
    starting at a block's end on the cold rows moves the frontier, one starting inside a block (a decode's step, a
    feed after one) does not; the commit's snapshots past its rows are dropped, those within kept at their nodes'
    ends; a cut takes back the frontier and the snapshots past it; a conversation opened on the tree's rows to a
    snapshot is cold to there, and one let go is cold nowhere"""
    pc = prefix(types=[LayerKind.LINEAR, LayerKind.FULL, LayerKind.FULL])
    c = pc.new()
    assert c.table.cold == 0 and not isinstance(c.layers[0], PagedLayer)
    n = 3 * PAGE + 10
    k, v = rows(n, 90)
    for i in (1, 2):
        cast(PagedLayer, c.layers[i]).append(k, v)
    for a, b in ((0, PAGE), (PAGE, 2 * PAGE), (2 * PAGE, n)):
        c.prefilled(a, b)
    assert c.table.cold == n
    c.prefilled(n, n + 5)  # on from the prompt's end, inside a block
    c.prefilled(PAGE, n)  # behind the frontier
    assert c.table.cold == n
    snaps: list[Any] = [{"n": PAGE}, {"n": 2 * PAGE}, {"n": n + 1}]
    c.commit(list(range(n)), snaps)
    assert sorted(c.table.held_snaps or {}) == [PAGE, 2 * PAGE]
    c.crop_to(2 * PAGE - 3)  # a cut below the second snapshot
    assert sorted(c.table.held_snaps or {}) == [PAGE] and c.table.cold == 2 * PAGE - 3
    c.flush()
    m = pc.match(list(range(n)))
    assert (m.length, m.snap_at, m.snap) == (PAGE, PAGE, snaps[0]), "the tree holds rows past the deepest snapshot"
    d = pc.new(m.rows[: m.snap_at])
    assert d.table.cold == PAGE
    c.release()
    assert c.table.cold == 0 and c.table.held_snaps is None
    assert pc.new().table.cold == 0 and prefix().new().table.cold is None


def test_a_snapshot_the_ram_has_no_room_for_lets_the_least_used_go_first() -> None:
    """a hybrid's snapshot is granted its RAM before it is taken: refused, the tree's least recently used snapshot goes
    (its node stays, read to the one before) and the grant is asked again; with none left to let go, it is not taken"""
    refusals = [0]

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        if "block's end" in str(kw.get("requester", "")) and refusals[0]:
            refusals[0] -= 1
            raise MemoryGrantError("no room for a snapshot")

    pc = prefix(grant, types=[LayerKind.LINEAR, LayerKind.FULL, LayerKind.FULL])
    paths = [list(range(PAGE)), list(range(1000, 1000 + PAGE))]  # two conversations sharing nothing
    held = []
    for j, ids in enumerate(paths):
        c = pc.new()
        k, v = rows(PAGE, 91 + j)
        for i in (1, 2):
            cast(PagedLayer, c.layers[i]).append(k, v)
        c.prefilled(0, PAGE)
        c.commit(ids, [{"n": PAGE, "path": j}])
        c.flush()
        held.append(c)
    assert len([n for n in pc.tree.nodes() if n.snap is not None]) == 2
    pc.match(paths[0])
    pc.match(paths[1])  # the second used last
    refusals[0] = 1
    assert pc.snap_room(1 << 20)
    kept = [n.snap["path"] for n in pc.tree.nodes() if n.snap is not None]
    assert kept == [1], f"the snapshot let go was not the least recently used: {kept} kept"
    refusals[0] = 1 << 30
    assert not pc.snap_room(1 << 20)
    assert not [n for n in pc.tree.nodes() if n.snap is not None] and len(list(pc.tree.nodes())) == 2


def test_free_pages_past_the_last_held_one_give_their_ram_back() -> None:
    """a long decode with no session grew the host's regions for its rows, and let go its pages were free while the
    regions kept their length until the engine closed - gigabytes no conversation read. `trim` lets the free pages past
    the last held one go and cuts the regions to the pages left, never below their first growth (`MIN_PAGES`, the room
    any request starts in - cut to nothing, a model that gave up all it could had none to answer in); the rows of the
    pages held read as they were, and the next growth goes on from there. A region cut to nothing is made again by
    the next growth"""
    asked: list[str] = []

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        asked.append(str(kw.get("requester", "")))

    first = HostRegion.MIN_PAGES
    pc = prefix(grant)
    keep = pc.new()
    k0, v0 = rows(PAGE, 21)
    write(keep, k0, v0)  # page 0, held
    long = pc.new()
    write(long, *rows(40 * PAGE, 22))  # pages 1..40
    grown = pc.pool.nbytes()["cpu"]
    long.release()
    assert pc.pool.trimmable() == grown - 2 * L * first * ROW > 0
    assert pc.pool.trim() == grown - 2 * L * first * ROW and pc.pool.nbytes()["cpu"] == 2 * L * first * ROW
    assert len(pc.pool.pages.pages) == 1 and pc.pool.host.cap == first and not pc.pool.pages.check()
    assert any(f"cut to {first} pages" in a for a in asked), "the cut copies were not asked of the ledger"
    for i in range(L):
        got = read(keep, i)
        assert torch.equal(got[0], k0 + i) and torch.equal(got[1], v0 - i)
    k1, v1 = rows(3 * PAGE, 23)
    write(keep, k1, v1)  # on in the room left
    for i in range(L):
        got = read(keep, i)
        assert torch.equal(got[0], torch.cat([k0, k1], -2) + i) and torch.equal(got[1], torch.cat([v0, v1], -2) - i)
    keep.release()
    assert pc.pool.trim() == 0 and pc.pool.trimmable() == 0, "the first growth's room given back"
    assert pc.pool.pages.pages == [] and pc.pool.nbytes()["cpu"] == 2 * L * first * ROW
    assert pc.pool.host.shrink(0) == 2 * L * first * ROW and pc.pool.host.cap == 0 and not pc.pool.host.k
    again = pc.new()
    write(again, *rows(PAGE + 3, 24))
    assert len(pc.pool.pages.pages) == 2 and pc.pool.host.cap == first
    for i in range(L):
        assert read(again, i)[0].shape[-2] == PAGE + 3


def test_a_trim_refused_part_way_gives_the_rest_back_at_the_next_ask() -> None:
    """a trim's cut copies are asked of the ledger a buffer at a time. Refused part way, the buffers not yet cut keep
    their length, still counted as what a trim gives back (`trimmable`), and the next trim cuts them: with the page
    count cut already, the retry found nothing to do and they stayed held"""
    cuts = [0]

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        if "cut to" in str(kw.get("requester", "")):
            cuts[0] += 1
            if cuts[0] == 2:  # the second buffer's cut, once
                raise MemoryGrantError("refused")

    first = HostRegion.MIN_PAGES
    floor = 2 * L * first * ROW
    pc = prefix(grant)
    long = pc.new()
    write(long, *rows(40 * PAGE, 26))
    grown = pc.pool.nbytes()["cpu"]
    long.release()
    part = pc.pool.trim()
    assert 0 < part < grown - floor, "the refusal stopped nothing, or everything"
    assert pc.pool.trimmable() == pc.pool.nbytes()["cpu"] - floor > 0, "the buffers left long are not counted"
    assert pc.pool.trim() == grown - floor - part and pc.pool.nbytes()["cpu"] == floor
    assert pc.pool.trimmable() == 0 and pc.pool.host.cap == first


def test_a_host_region_refused_part_way_is_made_whole_at_the_next_write() -> None:
    """pages counted before the host's first rows (a card engine's card layers wrote first) have their regions made
    at the host's first write, a layer at a time as the ledger grants each. Refused part way, the next write makes the
    rest: with the dtype set and the pages counted, nothing was asked again and a write to a layer whose regions were
    never made raised KeyError, however much RAM had come back"""
    calls = [0]

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        calls[0] += 1
        if calls[0] == 3:  # layer 1's K: layer 0 whole, layers 1 and 2 none
            raise MemoryGrantError("refused")

    host = HostRegion([0, 1, 2], HK, D, grant)
    host.grow(4)
    k, v = rows(PAGE, 25)
    with pytest.raises(MemoryGrantError):
        host.shape(k)
    assert host.dtype is not None and sorted(host.k) == [0] and host.growth(4) > 0
    host.shape(k)
    at = torch.arange(PAGE)
    for i in (0, 1, 2):
        host.write(i, at, k[0], v[0])
        got = host.gather(i, at)
        assert torch.equal(got[0][0], k[0]) and torch.equal(got[1][0], v[0])
    assert host.growth(4) == 0


def test_the_card_region_asks_its_growths_peak_not_its_whole_new_size() -> None:
    """the card's arenas and the park ask the ledger for what a growth takes at its peak - what they add, and where
    regrown rather than mapped in place one layer's old buffer beside its new - the figure `growth` and `park_growth`
    price. Asked for their whole new size, a growth priced as fitting was refused (1 GiB to 1.5 GiB with 600 MiB
    free): here the ledger refuses anything past the price"""
    asked: list[tuple[str, int, int]] = []
    budget: list[int | None] = [None]

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        asked.append((str(kw.get("requester", "")), int(nbytes), int(kw.get("held", 0))))
        if budget[0] is not None and nbytes > budget[0]:
            raise MemoryGrantError("refused")

    pc = prefix(grant, card=(0, 1))
    card = pc.pool.card
    assert card is not None
    card._grow(HostRegion.MIN_PAGES)
    old = card.cap
    budget[0] = price = card.growth(old + 1)
    card._grow(old + 1)
    added = len(card.arenas) * (card.cap - old) * card._page_bytes()
    assert asked[-1][1:] == (price, price - added) and "card arenas" in asked[-1][0], asked[-1]
    card._grow_park(4)
    have = len(card.parked)
    budget[0] = price = card.park_growth(have + 1)
    card._grow_park(have + 1)
    added = len(card.arenas) * (len(card.parked) - have) * card._page_bytes()
    assert asked[-1][1:] == (price, price - added) and "park" in asked[-1][0], asked[-1]
