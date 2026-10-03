# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a pass speaks for in the device's ledger it gives back however the pass ends. A layer-by-layer prefill
reserves its working set and the prompt's KV and grows a depot of experts on the card; an epoch of rows reserves
its KV. A fault in the middle of either - a layer that raises, the depot's own cleanup raising, a decode that
raises - must leave the ledger's reservations where they were before the call and no depot attached: left standing,
every later reading of the card's free memory counts them, and the next call is refused room it has."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from tests.helpers import fixture, loaded_model, need_cuda

if TYPE_CHECKING:
    from btb.engine.model import StreamedTextModel

PROMPT = [(7 * i + 3) % 200 + 2 for i in range(40)]


class Fault(RuntimeError):
    """the fault the tests put in a pass"""


@pytest.fixture(scope="module")
def sm() -> Iterator[StreamedTextModel]:
    """one mixture on the card for the whole file: a host layer prefilled on the card in chunks of 8, layer by
    layer, so a prefill reserves its working set and grows a depot"""
    dev = need_cuda()
    kw: dict[str, Any] = {"device": dev, "cpu_layers": 1, "prefill_chunk": 8, "prefill_card_min": 4}
    with loaded_model(fixture("tiny_q4"), **kw) as m:
        yield m


def _held(sm: StreamedTextModel) -> tuple[int, int]:
    return int(sm.device.reserved(sm.dev)), int(sm.device.reserved("cpu"))


def _fault_after(n: int) -> Any:
    """a stand-in that raises on its `n`-th call and passes the rest to the method it replaces"""
    calls = [0]

    def wrap(inner: Any) -> Any:
        def call(*a: Any, **k: Any) -> Any:
            calls[0] += 1
            if calls[0] == n:
                raise Fault(f"the test's fault, call {n}")
            return inner(*a, **k)

        return call

    return wrap


@pytest.mark.parametrize("where", ["a layer", "the depot's cleanup"])
def test_a_prefill_that_fails_gives_back_what_it_reserved(
    sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    from btb.engine.experts import LayerDepot

    sm.generate(list(PROMPT), 1, eos=(), speculate=False)  # once cleanly: what a sweep leaves when it ends well
    before = _held(sm)
    if where == "a layer":
        monkeypatch.setattr(sm, "_run_card_layer", _fault_after(2)(sm._run_card_layer))
    else:
        monkeypatch.setattr(LayerDepot, "close", _fault_after(1)(LayerDepot.close))
    with pytest.raises(Fault):
        sm.generate(list(PROMPT), 1, eos=(), speculate=False)
    assert _held(sm) == before, f"a prefill that failed at {where} left {_held(sm)} reserved (was {before})"
    assert getattr(sm, "_depot", None) is None, f"a prefill that failed at {where} left its depot attached"
    monkeypatch.undo()
    sm.generate(list(PROMPT), 1, eos=(), speculate=False)  # and the next one is not refused room it has
    assert _held(sm) == before


def test_rows_coming_onto_a_full_card_are_refused_before_any_moves(sm: StreamedTextModel) -> None:
    """a layer coming onto the card takes its rows in every live cache along, granted together first: with no room
    for them the move is refused whole - every row still where it was, the allocator never run out - and with room
    it goes through"""
    from btb.engine.scheduler import MemoryGrantError
    from btb.engine.tiers import _rows_bytes
    from tests.integration.test_never_oom import _allocator, squeezed

    i = min(sm.host)
    kept = [sm.session(PROMPT), sm.session(PROMPT[:20])]
    away = sum(_rows_bytes(s.cache, i, sm.dev) for s in kept)
    assert away, "the host layer's rows start on the host"
    with squeezed(sm, 0):
        before = _allocator(sm.dev)
        with pytest.raises(MemoryGrantError, match="onto the card"):
            sm._caches_to(i, sm.dev)
        assert _allocator(sm.dev) == before
    assert sum(_rows_bytes(s.cache, i, sm.dev) for s in kept) == away, "a refused move moved rows"
    sm._caches_to(i, sm.dev)
    assert sum(_rows_bytes(s.cache, i, sm.dev) for s in kept) == 0
    sm._caches_to(i, "cpu")


def test_an_epoch_that_fails_gives_back_its_kv(sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """rows of their own lengths go in epochs the scheduler sizes, each reserving its KV: a decode that raises
    mid-epoch still lets that reservation go"""
    rows = [PROMPT[:9], PROMPT[:13]]
    sm.generate(rows, 2, eos=(), speculate=False)
    before = _held(sm)
    monkeypatch.setattr(sm, "generate_greedy", _fault_after(1)(sm.generate_greedy))
    with pytest.raises(Fault):
        sm.generate(rows, 2, eos=(), speculate=False)
    assert _held(sm) == before, f"an epoch that failed left {_held(sm)} reserved (was {before})"
