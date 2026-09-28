# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""An engine's life from its load to its close (btb/engine/holdings.py): a close lets go of everything the engine
registered, one release raising neither stops the rest nor is forgotten, and a load that fails after the engine is
built closes it. On the CPU with a tiny mixture, so the store, its readers and the weights are all registered; the
harness's census (tests/conftest.py) checks each test for what a closed engine kept."""

from __future__ import annotations

from typing import Any

import pytest
from pytest import MonkeyPatch

import btb
from btb.engine.holdings import Stage
from btb.engine.model import StreamedTextModel
from tests.helpers import NO_LOG, fixture

MOE = "tiny_gpt_oss"


def test_a_close_lets_go_of_every_holding() -> None:
    sm = btb.load(fixture(MOE), device="cpu", log=NO_LOG)
    sm.generate([1, 2, 3, 4], 4, speculate=False)
    held = sm.holdings.names()
    assert "the expert store" in held and "the weights" in held and "the drive's readers" in held, held
    sm.close()
    assert len(sm.holdings) == 0 and sm.expert_store is None
    sm.close()  # a second close finds nothing to do


def test_a_release_that_raises_stops_nothing_else_and_is_tried_again() -> None:
    sm = btb.load(fixture(MOE), device="cpu", log=NO_LOG)
    tries: list[int] = []

    def flaky() -> None:
        tries.append(1)
        if len(tries) == 1:
            raise OSError("the drive went away")

    sm.holdings.own(Stage.RECORD, "a flaky record", flaky)
    with pytest.raises(ExceptionGroup) as got:
        sm.close()
    assert any("releasing a flaky record" in n for e in got.value.exceptions for n in getattr(e, "__notes__", []))
    assert sm.holdings.names() == ["a flaky record"], "every other holding let go; the one that raised kept"
    assert sm.expert_store is None
    sm.close()  # tried again, and let go
    assert len(sm.holdings) == 0 and len(tries) == 2


def test_a_load_that_fails_after_the_engine_is_built_closes_it(monkeypatch: MonkeyPatch) -> None:
    """a warm-up raising (or a draft that does not match, a declined download) closes the engine load built, before
    the error goes on - not left to the collector with its store and weights"""
    built: list[Any] = []
    real = StreamedTextModel.close

    def close(self: StreamedTextModel) -> None:
        built.append(self)
        real(self)

    def warm(self: StreamedTextModel) -> int:
        raise RuntimeError("the warm-up failed")

    monkeypatch.setattr(StreamedTextModel, "close", close)
    monkeypatch.setattr(StreamedTextModel, "warm", warm)
    with pytest.raises(RuntimeError, match="the warm-up failed"):
        btb.load(fixture(MOE), device="cpu", log=NO_LOG)
    # the engine load built, not its placement probe (closed however the load ends): the one given a draft slot
    engines = [e for e in built if "draft_engine" in vars(e)]
    assert len(engines) == 1 and len(engines[0].holdings) == 0, "the built engine was closed"
