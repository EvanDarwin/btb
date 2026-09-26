# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The API's calls as calls: a stream's close stops its own decode and no other, a callback cannot call back into
the engine mid-step, each generate carries its own report, a call is counted once however many calls it makes
inside, and a `Generation` survives pickling. On the CPU with the tiny fixtures."""

from __future__ import annotations

import copy
import functools
import pickle
import threading
from collections.abc import Iterator

import pytest
import torch

from btb.api import api
from btb.engine import StreamedTextModel
from btb.kinds import PassTag
from btb.sampling import Sampling
from btb.session import Session
from tests.helpers import fixture, loaded_model

PROMPT = [5, 17, 99, 3, 42, 8, 61, 7, 12, 30]


@pytest.fixture
def sm() -> Iterator[StreamedTextModel]:
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as m:
        yield m


def test_closing_a_stream_stops_its_own_decode_and_no_other(sm: StreamedTextModel) -> None:
    """a stream queued behind another caller's decode and closed: that decode runs to its end"""
    going = threading.Event()
    got: list[int] = []

    def other() -> None:
        g = sm.generate(PROMPT, 200, eos=(), speculate=False, on_token=lambda _t: going.set())
        got.extend(g.tokens)

    a = threading.Thread(target=other)
    a.start()
    assert going.wait(30)
    st = sm.stream(PROMPT, 50)
    st.close()
    a.join()
    assert len(got) == 200
    assert not sm.abort.is_set()


def test_a_callback_calling_back_into_the_engine_is_refused(sm: StreamedTextModel) -> None:
    """mid-decode the engine is between steps: a callback feeding the session being decoded, or starting a decode
    of its own, is refused (the decode stops with the refusal); reading the model's memory is not a call into it"""
    s = sm.session(PROMPT)
    with pytest.raises(RuntimeError, match="callback"):
        s.generate(4, eos=(), speculate=False, on_token=lambda _t: s.feed([1, 2, 3]))
    toks = s.tokens
    ref = sm.session(toks)
    assert torch.allclose(s.feed([7])[-1], ref.feed([7])[-1], atol=1e-4)
    with pytest.raises(RuntimeError, match="callback"):
        sm.generate(PROMPT, 4, eos=(), speculate=False, on_token=lambda _t: sm.generate(PROMPT, 2))
    seen: list[int] = []
    sm.generate(PROMPT, 3, eos=(), speculate=False, on_pass=lambda _p: seen.append(sm.memory()["cpu"].free))
    assert seen
    br = sm.session(PROMPT).fork(2)
    with pytest.raises(RuntimeError, match="callback"):
        br.generate(3, eos=(), on_token=lambda r, _t: br.leave(r))


def test_each_generate_carries_its_own_report(sm: StreamedTextModel) -> None:
    hot = sm.generate(PROMPT, 3, eos=(), speculate=False, sampling=Sampling(temperature=0.9, seed=1))
    cold = sm.generate(PROMPT, 3, eos=(), speculate=False)
    assert PassTag.SAMPLE_STOCHASTIC in hot.report.tags and PassTag.SAMPLE_GREEDY not in hot.report.tags
    assert PassTag.SAMPLE_GREEDY in cold.report.tags


def test_a_call_is_counted_once_however_many_calls_it_makes(sm: StreamedTextModel) -> None:
    """`crop` rewinds and `session(ids)` feeds, inside: the report counts the calls the caller made"""
    sm.session(PROMPT).crop(4)
    calls = {t for t in sm.last_pass_report().tags if t.value.startswith(("model.", "session."))}
    assert calls == {PassTag.API_MODEL_SESSION, PassTag.API_SESSION_CROP}


def test_an_override_of_an_api_method_is_recorded_under_its_tag(sm: StreamedTextModel) -> None:
    class Mine(Session):
        def mark(self):  # type: ignore[no-untyped-def]
            return super().mark()

    s = Mine(engine=sm)
    s.feed(PROMPT)
    before = sm.last_pass_report().tags
    s.mark()
    assert PassTag.API_SESSION_MARK in sm.last_pass_report().tags - before or PassTag.API_SESSION_MARK in before
    assert getattr(Mine.mark, "__btb_tag__", None) is PassTag.API_SESSION_MARK


def test_a_public_callable_that_is_not_a_function_is_refused() -> None:
    with pytest.raises(TypeError, match="not a plain function"):

        @api("room")
        class R:
            def _called(self, tag: PassTag) -> None:
                pass

            empty = functools.partial(print)


def test_a_generation_survives_pickling_and_copying(sm: StreamedTextModel) -> None:
    g = sm.generate(PROMPT, 3, eos=(), speculate=False, logprobs=2)
    for h in (pickle.loads(pickle.dumps(g)), copy.copy(g), copy.deepcopy(g)):
        assert list(h.tokens) == list(g.tokens) and h.stats == g.stats
        assert h.logprobs == g.logprobs and h.hidden == g.hidden


@pytest.mark.parametrize("stem", ["tiny_qwen3", "tiny_q35"])
def test_a_fed_session_decodes_on_from_its_whole_cache(stem: str) -> None:
    """a session fed its prompt and decoded from there reuses every row it holds - a hybrid's recurrent states
    included, which no prefix can be cut back to - and draws what a fresh decode draws"""
    with loaded_model(fixture(stem), device="cpu") as m:
        s = m.session(PROMPT)
        g = s.generate(6, eos=(), speculate=False)
        assert g.stats["reused"] == len(PROMPT)
        assert list(g.tokens) == list(m.generate(PROMPT, 6, eos=(), speculate=False).tokens)


def test_a_turn_that_fails_leaves_no_unanswered_message(sm: StreamedTextModel) -> None:
    chat = sm.chat(max_new=4)

    def boom(_ids: object, _lg: torch.Tensor) -> torch.Tensor:
        raise ValueError("the caller's")

    with pytest.raises(ValueError, match="the caller's"):
        chat.ask("hi", processors=[boom])
    assert chat.history == []
    chat.ask("hi")
    assert [m["role"] for m in chat.history] == ["user", "assistant"]
