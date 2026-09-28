# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The expert store's residency policy and its pages, from the load option through to the store. Both are read
at store construction, inside `StreamedTextModel.__init__`, so they have to be on the model before it is built -
which is the bug this pins: assigned after the load they never applied, and every model ran the plain line while
`bus_pass` defaulted to the Bus Pass. A MoE fixture on the CPU, so the whole file is seconds."""

from __future__ import annotations

import gc
import types
import weakref

import pytest
import torch
from pytest import MonkeyPatch

from btb.engine.experts import BusPass, Riders, _ExpertStore
from btb.engine.families import Family
from btb.kinds import FamilyKind, PassTag
from tests.helpers import GB, KB, NO_LOG, fixture, loaded_model

MOE = "tiny_gpt_oss"  # the served MoE family whose tiny fixture builds a store


def _policy(path: str, **load_kw: object) -> tuple[str, frozenset[PassTag]]:
    """(the residency class the store chose, the tags one greedy pass recorded)"""
    with loaded_model(path, device="cpu", **load_kw) as sm:
        assert sm.expert_store is not None, "the expert store did not open"
        sm.generate([1, 2, 3, 4], 4, speculate=False)
        return type(sm.expert_store.res).__name__, sm.last_pass_report().tags


def test_the_documented_default_is_the_bus_pass() -> None:
    """no option and no environment: `bus_pass` defaults to 1 (btb/options.py), and the pass shows it ran."""
    res, tags = _policy(fixture(MOE))
    assert res == BusPass.__name__
    assert PassTag.EXPERT_BUS_PASS in tags and PassTag.EXPERT_LINE not in tags


def test_bus_pass_off_takes_the_plain_line() -> None:
    res, tags = _policy(fixture(MOE), bus_pass=0)
    assert res == Riders.__name__
    assert PassTag.EXPERT_LINE in tags and PassTag.EXPERT_BUS_PASS not in tags


def test_the_environment_knob_wins_over_the_option(monkeypatch: MonkeyPatch) -> None:
    """BTB_BUS_PASS is documented as overriding the option, in both directions."""
    monkeypatch.setenv("BTB_BUS_PASS", "0")
    assert _policy(fixture(MOE), bus_pass=1)[0] == Riders.__name__
    monkeypatch.setenv("BTB_BUS_PASS", "1")
    assert _policy(fixture(MOE), bus_pass=0)[0] == BusPass.__name__


def _stub(**attrs: object) -> types.SimpleNamespace:
    """a model beside a card, carrying only what the store reads while it is being built"""
    return types.SimpleNamespace(log=NO_LOG, dev=torch.device("cuda"), fam=Family(kind=FamilyKind.QWEN3), **attrs)


def _store(**attrs: object) -> _ExpertStore:
    return _ExpertStore(_stub(**attrs), budget_bytes=64 * KB, reserve_bytes=GB)


def test_store_pin_reaches_the_store() -> None:
    """the pages the store's blocks are allocated in: pinned where `store_pin` asks and the model sits beside a
    card, 'auto' the card's own answer. Pageable off a card whatever the option says - there is nothing to pin
    for."""
    assert _store(bus_pass=True, store_pin=1).pin
    assert not _store(bus_pass=True, store_pin=0).pin
    with loaded_model(fixture(MOE), device="cpu", store_pin=1) as sm:
        assert sm.store_pin == 1  # on the model before the store was built
        assert sm.expert_store is None or not sm.expert_store.pin


def test_the_environment_knob_wins_for_the_pages(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("BTB_STORE_PIN", "auto")
    assert _store(bus_pass=True, store_pin=0).pin
    monkeypatch.setenv("BTB_STORE_PIN", "0")
    assert not _store(bus_pass=True, store_pin=1).pin


def test_a_model_with_no_policy_is_refused() -> None:
    """the fallback that hid the bug: a model built without a policy must fail loudly rather than be handed one
    the caller never chose."""
    for attrs in ({"store_pin": 0}, {"bus_pass": True}):
        with pytest.raises(AttributeError):
            _store(**attrs)


@pytest.mark.parametrize("bus_pass", [True, False])
def test_a_store_nothing_holds_is_freed_at_once(bus_pass: bool) -> None:
    """a store is freed when its last reference goes, not at the collector's next full pass: its residency policy
    reads the store's size through a weak reference. Held by a closure the store was a cycle, and an engine closed
    but still bound (a caller's `with`) left its gigabytes to the collector"""
    st = _store(bus_pass=bus_pass, store_pin=0)
    gone = weakref.ref(st)
    was = gc.isenabled()
    gc.disable()
    try:
        del st
        assert gone() is None, "the store outlived its last reference: something in it holds it in a cycle"
    finally:
        if was:
            gc.enable()


def test_closing_the_engine_lets_the_store_go() -> None:
    """`close` gives the expert store's blocks back itself: after a `with` the engine is still bound in the
    caller's code, and the store's gigabytes must not stay with it"""
    with loaded_model(fixture(MOE), device="cpu") as sm:
        sm.generate([1, 2, 3, 4], 4, speculate=False)
        store = sm.expert_store
        assert store is not None and store.blocks, "the store never grew a block"
        bufs = [weakref.ref(buf) for buf, _ids in store.blocks.values()]
        del store
    assert sm.expert_store is None
    assert all(b() is None for b in bufs), "the closed engine's store blocks are still held"


def test_a_miss_is_read_through_the_file_cache_where_commit_binds_the_host(monkeypatch: MonkeyPatch) -> None:
    """where the host's commit is what the store can grow into and not its RAM (Windows), the file cache is RAM the
    store cannot otherwise hold, and a miss is read through it; where the two are one figure, around it.
    `BTB_EXPERT_READS` decides where it is set"""
    from btb import sysinfo

    monkeypatch.delenv("BTB_EXPERT_READS", raising=False)
    monkeypatch.setattr(sysinfo, "host_free_bytes", lambda: 30 * GB)
    monkeypatch.setattr(sysinfo, "host_commit_bytes", lambda: 5 * GB)
    assert _store(bus_pass=True, store_pin=0).cached_reads, "commit below the RAM: through the cache"
    monkeypatch.setattr(sysinfo, "host_commit_bytes", lambda: 30 * GB)
    assert not _store(bus_pass=True, store_pin=0).cached_reads, "one figure: around it"
    monkeypatch.setenv("BTB_EXPERT_READS", "cached")
    assert _store(bus_pass=True, store_pin=0).cached_reads
    monkeypatch.setenv("BTB_EXPERT_READS", "direct")
    monkeypatch.setattr(sysinfo, "host_commit_bytes", lambda: 5 * GB)
    assert not _store(bus_pass=True, store_pin=0).cached_reads
