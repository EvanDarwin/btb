# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card graph's attention arena (`btb.engine.arena.KvArena`): its rows grow without ever being held twice. In
place where the card's driver maps memory onto reserved addresses - the pointers the kernels and the captured graphs
hold stand, the rows stay where they are, and the card gives exactly the growth; past a reservation the same chunks
are mapped onto new addresses, the rows untouched. Elsewhere layer by layer, the cache and one layer's growth at
most. A view of the arena keeps its memory as torch's own would, and its going gives the memory back."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from btb.engine.arena import KvArena, RowArena
from tests.helpers import need_cuda

N, HK, D = 4, 8, 128
ROW = HK * D * 2  # a row's bytes, every head


def _in_place() -> KvArena:
    need_cuda()
    a = KvArena(N, HK, D, torch.device("cuda"), ceiling=8192)
    if not a.in_place:
        pytest.skip("this card's driver does not map memory onto reserved addresses (the fallback's test covers it)")
    return a


def _used() -> int:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return total - free


def test_growing_in_place_keeps_every_pointer_and_row_and_takes_only_the_growth() -> None:
    a = _in_place()
    a.grow(a.rows_for(1000))  # every row the first chunks map
    assert a.cap * ROW % a.drv.gran == 0 and a.cap >= 1000  # type: ignore[union-attr]
    for j in range(N):
        for w in (0, 1):
            a[j, w].fill_(j * 2 + w)
    ptrs = [a[j, w].data_ptr() for j in range(N) for w in (0, 1)]
    k = a[0, 0]
    assert (k.stride(0), k.stride(1)) == (D, HK * D), "position-major: a head D apart, a row every head's D"
    before, cap0 = _used(), a.cap
    a.grow(3000)
    grown = _used() - before
    assert [a[j, w].data_ptr() for j in range(N) for w in (0, 1)] == ptrs, "a pointer moved"
    assert all(bool((a[j, w][:, :cap0] == j * 2 + w).all()) for j in range(N) for w in (0, 1)), "a row moved"
    # the card gives the growth and no more: no second arena, not even for a moment (mem_get_info sees the driver's
    # mappings, which torch's allocator does not)
    assert grown == a.nbytes(a.cap) - a.nbytes(cap0), f"{grown} bytes for a growth of {a.nbytes(a.cap - cap0)}"
    # past the reservation (a batch's prompts end to end): the chunks mapped onto new addresses, the rows kept
    a.grow(20000)
    assert a.cap >= 20000
    assert all(bool((a[j, w][:, :cap0] == j * 2 + w).all()) for j in range(N) for w in (0, 1))
    a.close()


def test_a_view_keeps_its_region_and_the_last_one_gives_it_back() -> None:
    a = _in_place()
    a.grow(1000)
    k = a[1, 1]
    k.fill_(2.5)
    region = weakref.ref(a.regions["v"][1])
    a.close()
    del a
    gc.collect()
    assert region() is not None and float(k[0, 5, 0]) == 2.5, "a cache's view of a closed arena still reads its rows"
    before = _used()
    del k
    gc.collect()
    assert region() is None and before - _used() > 0, "the last view's going gave the memory back"


def test_a_copy_queued_out_of_the_last_view_lands_before_the_rows_are_unmapped() -> None:
    """a cache detaching from a closed arena: its rows' copy queued on the card, the last view gone at once. The
    region waits for the card before unmapping (as torch's own free does), so the copy reads rows, not unmapped
    addresses (once an illegal access that poisoned the process)"""
    a = _in_place()
    a.grow(a.rows_for(4096))
    a[2, 1].fill_(3.0)
    v = a[2, 1]
    a.close()
    torch.cuda._sleep(200_000_000)  # the stream held busy, so the copy behind it is still queued as the view goes
    out = v.clone()
    del v
    gc.collect()
    torch.cuda.synchronize()
    assert bool((out == 3.0).all()), "the copy read the rows"


def test_the_fallback_regrows_a_layer_at_a_time_and_moves_its_holders() -> None:
    need_cuda()
    a = KvArena(N, HK, D, torch.device("cuda"), ceiling=8192)
    a.drv = None  # the path a card without the driver's mapping takes
    a.grow(1024)
    for j in range(N):
        a[j, 0].fill_(j)
    held = [a[j, 0] for j in range(N)]  # the caches' views
    moves: list[int] = []

    def moved(j: int) -> None:
        moves.append(j)
        held[j] = a[j, 0]  # the holder takes the new view, letting the old buffer go

    layer_old, layer_new = 2 * 1024 * ROW, 2 * 4096 * ROW
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    other = torch.cuda.memory_allocated() - N * layer_old  # everything but the arena
    a.grow(4096, moved=moved)
    peak = torch.cuda.max_memory_allocated() - other
    assert moves == list(range(N)), "every layer's holder is told, in turn"
    assert all(bool((held[j][:, :1024] == j).all()) for j in range(N)), "the rows came over"
    # the grown cache and the one old layer being copied out: never the old cache and the new one together (torch's
    # own small blocks come and go meanwhile: a MiB of slack, a layer here is 4 MiB)
    slack = 1 << 20
    assert peak <= N * layer_new + layer_old + slack, f"peak {peak} past the grown cache and one old layer"
    assert abs(torch.cuda.memory_allocated() - other - N * layer_new) <= slack, "the old buffers are gone"
    a.close()


def test_rows_pinned_in_ram_grow_in_place_and_the_card_reads_them() -> None:
    """an arena kept in RAM (`kv_host`) the card reads in place: CPU views torch takes for pinned memory, grown with
    no pointer moving and no row copied, the card reading what the host wrote"""
    need_cuda()
    a = RowArena(torch.device("cuda"), host=True, ceiling=8192)
    a.add("k", 2, (HK, D))
    if not a.in_place:
        pytest.skip("this card's driver does not map RAM onto reserved addresses (the fallback regrows by layer)")
    a.grow(1000)
    k = a.view("k", 1)
    assert k.device.type == "cpu" and k.is_pinned() and tuple(k.shape) == (a.cap, HK, D)
    k[:10].fill_(4.0)
    p0, cap0 = k.data_ptr(), a.cap
    a.grow(5000)
    assert a.view("k", 1).data_ptr() == p0 and bool((a.view("k", 1)[:10] == 4.0).all())
    assert float(a.view("k", 1)[:10].to("cuda")[3, 0, 0]) == 4.0, "the card reads the host's rows"
    assert a.cap > cap0
    a.close()


def test_a_pooled_region_holds_a_row_per_block() -> None:
    """a region of a row per `per` positions (Qwen4's pooled keys) grows with the rest: cap // per + extra rows"""
    need_cuda()
    a = RowArena(torch.device("cuda"), ceiling=8192)
    a.add("raw", 2, (64,))
    a.add("pk", 2, (64,), per=4, extra=1)
    a.grow(4096)
    assert a.view("raw", 0).shape[0] == a.cap and a.view("pk", 0).shape[0] == a.cap // 4 + 1
    a.view("pk", 1)[:3].fill_(1.5)
    a.grow(9000)
    assert a.view("pk", 1).shape[0] == a.cap // 4 + 1 and bool((a.view("pk", 1)[:3] == 1.5).all())
    a.close()
