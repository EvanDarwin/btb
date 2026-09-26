# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The API's calls as calls: a stream's close stops its own decode and no other, a callback cannot call back into
the engine mid-step, each generate carries its own report, a call is counted once however many calls it makes
inside, and a `Generation` survives pickling. On the CPU with the tiny fixtures."""

from __future__ import annotations

import copy
import dataclasses
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
from btb.session import Session, State, Step
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
        s.generate(4, eos=(), speculate=False, on_token=lambda _t: s.feed([1, 2, 3]).logits)
    toks = s.tokens
    ref = sm.session(toks)
    assert torch.allclose(s.feed([7]).logits[-1], ref.feed([7]).logits[-1], atol=1e-4)
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
        assert h.logprobs == g.logprobs and h.hidden == g.hidden and h.report == g.report


def test_a_generation_is_a_record_not_a_pair(sm: StreamedTextModel) -> None:
    """docs/api.md: read by name; it neither unpacks nor indexes as (tokens, stats), and it is not written to"""
    g = sm.generate(PROMPT, 3, eos=(), speculate=False)
    with pytest.raises(TypeError):
        _tokens, _stats = g  # type: ignore[misc]
    with pytest.raises(TypeError):
        g[0]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.tokens = []  # type: ignore[misc]


def test_a_feed_and_a_step_give_one_shape_whatever_is_asked(sm: StreamedTextModel) -> None:
    """a `Step` either way: its states empty without taps, a layer's with them"""
    s = sm.session(PROMPT)
    plain, tapped = s.feed([1, 2]), s.feed([3], taps=[-1])
    assert isinstance(plain, Step) and plain.logits.shape[0] == 2 and plain.hidden == {}
    assert isinstance(tapped, Step) and list(tapped.hidden) == [sm.L - 1]
    with s.fork(2) as br:
        got = br.step([4, 5], taps=[0])
        assert isinstance(got, Step) and got.logits.shape[0] == 2 and list(got.hidden) == [0]


def test_the_drawn_token_is_fed_when_asked_for_what_follows_it(sm: StreamedTextModel) -> None:
    """a decode leaves its last token drawn: `logits` is None until it is fed, `next_logits()` feeds it and is what
    a session fed the same tokens gives; a session's and rows' `pending` are gone"""
    s = sm.session(PROMPT)
    s.generate(3, eos=(), speculate=False)
    before, waiting = s.state, s.logits
    assert before is State.PENDING and waiting is None and not hasattr(s, "pending")
    got = s.next_logits()
    assert s.state is State.READY and s.logits is got
    assert torch.allclose(got, sm.session(s.tokens).next_logits(), atol=1e-4)
    assert s.next_logits() is got, "a second ask feeds nothing"
    with sm.session(PROMPT).fork(2) as br:
        br.generate(3, eos=())
        drawn = br.logits
        assert drawn is None and not hasattr(br, "pending")
        with pytest.raises(ValueError, match="advance"):
            br.step([1, 2])
        lg = br.next_logits()
        assert lg.shape[0] == 2 and br.logits is lg
        with pytest.raises(ValueError, match=r"step\(tokens\)"):
            br.advance()
        with pytest.raises(TypeError):
            br.step()  # type: ignore[call-arg]


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
