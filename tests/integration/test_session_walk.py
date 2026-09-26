# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A model-based random walk over a session's moves (docs/sessions.md): seeded sequences of feed, mark and rewind,
crop, sync, generate, fork and batch - failures injected into the passes along the way - and after every move,
whatever it raised, the session checked against the model it must follow:

* its state is one of four, and the state's invariant holds (a cache holding exactly the tokens it has fed);
* its next token's logits are a fresh session's fed the same tokens (the reference);
* a failed move left it where the move's rollback point says; a lent session refuses every move.

On the CPU, every family's fixture; `BTB_WALK_SEEDS` runs more seeds than the default."""

from __future__ import annotations

import contextlib
import gc
import os
import random
from collections.abc import Iterator

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.branches import Batch, Branches
from btb.kinds import LayerKind
from btb.session import Mark, Session, State
from tests.cert import spec
from tests.helpers import fixture, loaded_model

STEMS = sorted(set(spec.FIXTURE_STEM.values()))
SEEDS = range(int(os.environ.get("BTB_WALK_SEEDS", "3")))
STEPS = 40
VOCAB = 200  # the tokens a walk feeds: inside every fixture's vocabulary

_models: dict[str, StreamedTextModel] = {}
_open = contextlib.ExitStack()


@pytest.fixture(scope="module", autouse=True)
def _close_models() -> Iterator[None]:
    yield
    _open.close()
    _models.clear()


def model(stem: str) -> StreamedTextModel:
    if stem not in _models:
        _models[stem] = _open.enter_context(loaded_model(fixture(stem), device="cpu"))
    return _models[stem]


class Boom(RuntimeError):
    """a failure injected into a move"""


@contextlib.contextmanager
def failing(sm: StreamedTextModel, at: int | None) -> Iterator[None]:
    """the `at`-th layer pass from here raises (None: none does)"""
    if at is None:
        yield
        return
    run = sm.device.run_layer
    calls = [0]

    def once(i: int, h: torch.Tensor, pas: object) -> torch.Tensor:
        calls[0] += 1
        if calls[0] == at:
            raise Boom(f"layer pass {at}")
        return run(i, h, pas)

    sm.device.run_layer = once  # type: ignore[method-assign]
    try:
        yield
    finally:
        sm.device.run_layer = run  # type: ignore[method-assign]


class Walk:
    """one seeded walk over one session of a model"""

    def __init__(self, sm: StreamedTextModel, seed: int) -> None:
        self.sm, self.rng = sm, random.Random(seed)
        self.s = sm.session()
        self.marks: list[Mark] = []
        self.hybrid = LayerKind.LINEAR in sm.layer_types
        self.log: list[str] = []

    def toks(self, lo: int = 1, hi: int = 4) -> list[int]:
        return [self.rng.randrange(1, VOCAB) for _ in range(self.rng.randint(lo, hi))]

    def fault(self) -> int | None:
        """a failure a third of the time, at a layer pass the move is likely to reach"""
        return self.rng.randint(1, 3 * self.sm.L) if self.rng.random() < 1 / 3 else None

    # -- the model the session must follow --

    def check(self) -> None:
        s, sm = self.s, self.sm
        where = " -> ".join(self.log[-6:])
        st = s.state
        assert st in (State.EMPTY, State.READY, State.PENDING), f"{st} after {where}"
        if st is State.EMPTY:
            assert s.cache is None and not s.ids and s.pending is None and s.logits is None, where
            return
        assert s.cache is not None
        assert s.cache.get_seq_length() == len(s.ids), f"{s.cache.get_seq_length()} rows for {len(s.ids)}: {where}"
        if st is State.READY:
            assert s.logits is not None and s.pending is None, where
        else:
            assert s.pending is not None and s.logits is None, where
        # the reference: a fresh session fed the same tokens gives the same next logits (a probe, rewound after)
        here = s.mark()
        got = s.feed([7])[-1]
        s.rewind(here)
        want = sm.session(s.tokens).feed([7])[-1]
        d = float((got - want).abs().max())
        assert d <= 1e-4 * max(1.0, float(want.abs().max())), f"logits {d} apart after {where}"
        assert s.cache.get_seq_length() == len(s.ids) and s.tokens == list(here.path or ()), where

    # -- the moves --

    def feed(self) -> None:
        before, new, at = self.s.tokens, self.toks(), self.fault()
        self.log.append(f"feed{new}{'!' + str(at) if at else ''}")
        try:
            with failing(self.sm, at):
                self.s.feed(new)
        except Boom:
            assert self.s.tokens == before, "a failed feed changed the tokens"
            return
        assert self.s.tokens == before + new

    def mark(self) -> None:
        if self.s.state is not State.EMPTY:
            self.log.append("mark")
            self.marks.append(self.s.mark())

    def rewind(self) -> None:
        if not self.marks:
            return
        m = self.rng.choice(self.marks)
        on_path = m.path is not None and self.s.tokens[: len(m.path)] == list(m.path)
        self.log.append(f"rewind{m.n}{'' if on_path else '(off path)'}")
        if not on_path:
            with pytest.raises(ValueError):
                self.s.rewind(m)
            return
        self.s.rewind(m)
        assert self.s.tokens == list(m.path or ())

    def crop(self) -> None:
        if self.hybrid or not len(self.s):
            return
        n = self.rng.randint(0, len(self.s))
        before = self.s.tokens
        self.log.append(f"crop{n}")
        self.s.crop(n)
        assert self.s.tokens == before[:n]

    def sync(self) -> None:
        before = self.s.tokens
        keep = self.rng.randint(0, len(before))
        target = before[:keep] + self.toks()
        at = self.fault()
        self.log.append(f"sync{keep}+{len(target) - keep}{'!' + str(at) if at else ''}")
        try:
            with failing(self.sm, at):
                self.s.sync(target)
        except Boom:
            got = self.s.tokens
            assert before[: len(got)] == got, "a failed sync left tokens the session never held"
            shared = next(
                (i for i, (x, y) in enumerate(zip(target, before, strict=False)) if x != y),
                min(len(target), len(before)),
            )
            # what the target shares with the session is kept, all but a hybrid's (back to its anchor)
            assert self.hybrid or len(got) >= min(shared, len(target) - 1), f"a failed sync kept {len(got)} of {shared}"
            return
        assert self.s.tokens == target

    def generate(self) -> None:
        if not len(self.s):
            return
        before, n, at = self.s.tokens, self.rng.randint(1, 5), self.fault()
        spec_on = self.rng.random() < 0.5
        self.log.append(f"generate{n}{'s' if spec_on else ''}{'!' + str(at) if at else ''}")
        try:
            with failing(self.sm, at):
                g = self.s.generate(n, eos=(), speculate=spec_on)
        except Boom:
            got = self.s.tokens
            assert before[: len(got)] == got, "a failed decode left tokens the session never held"
            # a decode of the session's own tokens replaces nothing: a dense one loses none of them (a hybrid that
            # must re-run its last tokens, for the drafting head, goes back to its anchor)
            assert self.hybrid or got == before, f"a failed decode lost {len(before) - len(got)} tokens"
            return
        assert self.s.tokens == before + list(g.tokens)

    def rows_step(self, rs: Branches | Batch, toks: list[int]) -> None:
        """a step of a fork's or a batch's rows, failing part way a third of the time: then the rows are as before
        (the tokens they hold, pending or not) and the same step is taken again"""
        at = self.fault()
        if at is not None:
            rows, pend = [list(r) for r in rs.rows], rs.pending
            self.log.append(f"step!{at}")
            try:
                with failing(self.sm, at):
                    rs.step(toks)
            except Boom:
                assert [list(r) for r in rs.rows] == rows and rs.pending == pend, "a failed step moved the rows"
            else:
                return
        rs.step(toks)

    def fork(self) -> None:
        if not len(self.s):
            return
        before, n = self.s.tokens, self.rng.randint(1, 3)
        self.log.append(f"fork{n}")
        br = self.s.fork(n)
        assert self.s.state is State.LENT
        with pytest.raises(ValueError, match="forked"):
            self.s.feed([1])
        self.rows_step(br, self.toks(n, n))
        if self.rng.random() < 0.5:
            at = self.fault()
            self.log.append(f"rows-generate{'!' + str(at) if at else ''}")
            # a failed step inside leaves every row its draws so far, the last pending: the rows go on from there
            with contextlib.suppress(Boom), failing(self.sm, at):
                br.generate(self.rng.randint(1, 3), eos=())
        if n > 1 and self.rng.random() < 0.3:
            br.leave(self.rng.randrange(n))
        how = self.rng.choice(["keep", "close", "drop"])
        self.log.append(how)
        if how == "keep":
            r = self.rng.randrange(br.n)
            row = list(br.rows[r])
            br.keep(r)
            assert self.s.tokens == before + row
        elif how == "close":
            br.close()
            assert self.s.tokens == before
        else:
            del br
            gc.collect()
            assert self.s.tokens == before

    def batch(self) -> None:
        if not len(self.s):
            return
        other = self.sm.session(self.toks(2, 6))
        before, picks = self.s.tokens, self.toks(2, 2)
        how = self.rng.choice(["close", "raise", "drop"])
        self.log.append(f"batch-{how}")
        if how == "drop":
            bt = self.sm.batch([self.s, other])
            assert self.s.state is State.LENT
            self.rows_step(bt, picks)
            del bt
            gc.collect()
            assert self.s.tokens == before, "a batch dropped unclosed wrote a row back"
            return
        with contextlib.suppress(Boom), self.sm.batch([self.s, other]) as bt:
            assert self.s.state is State.LENT
            self.rows_step(bt, picks)
            if how == "raise":
                raise Boom("inside the batch")
        assert self.s.tokens == before + picks[:1]

    MOVES = ("feed", "feed", "mark", "rewind", "crop", "sync", "generate", "generate", "fork", "batch")

    def run(self, steps: int) -> None:
        self.s.feed(self.toks(3, 8))
        self.check()
        for _ in range(steps):
            getattr(self, self.rng.choice(self.MOVES))()
            self.check()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("stem", STEMS)
def test_a_session_follows_its_model_through_any_walk(stem: str, seed: int) -> None:
    Walk(model(stem), seed).run(STEPS)


def test_a_mark_off_the_sessions_path_is_refused() -> None:
    """a mark from before the session went another way names rows that now hold other tokens: refused"""
    sm = model(STEMS[0])
    s = sm.session([5, 17, 99, 3])
    m = s.mark()
    s.feed([1, 2])
    s.crop(2)
    s.feed([9, 9])
    with pytest.raises(ValueError, match="path"):
        s.rewind(m)
    assert isinstance(s, Session) and s.tokens == [5, 17, 9, 9]
