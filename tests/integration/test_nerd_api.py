# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The programmatic API over a decode, on every family's tiny fixture: hooks (logits processors, logprobs, taps,
pass stats), hidden states and the head, a session's mark/rewind, a continued assistant turn, a fork's rows and a
batch of sessions. On the CPU every comparison is exact; on MLX a batched pass and a single-row one sum in bf16
in different orders, so there logits are held to a tolerance and tokens only where one path computes both."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
import torch

from btb.kinds import LayerKind
from btb.sampling import Sampling
from btb.text import template
from tests.cert import spec
from tests.helpers import fixture, loaded_model, need_mlx

STEMS = sorted(set(spec.FIXTURE_STEM.values()))
DEVICES = ["cpu", "mlx"]
PROMPT = [5, 17, 99, 3, 42, 8, 61, 7, 12, 30]
OTHER = [44, 2, 90, 13, 7, 21]
LONG = [9, 8, 7, 6, 5, 4, 3, 2, 1, 11, 12, 13, 14]
N = 8
SMP = Sampling(temperature=0.9, seed=11)
# the gap between two ways of computing the same values, relative to their size: on the CPU a prefill's gemm
# and a step's gemv sum in fp32 in different orders; on MLX a batched pass and a single-row one do in bf16
TOL = {"cpu": 1e-4, "mlx": 0.05}

_models: dict[tuple[str, str], Any] = {}
_open = contextlib.ExitStack()


@pytest.fixture(scope="module", autouse=True)
def _close_models() -> Iterator[None]:
    yield
    _open.close()
    _models.clear()


def model(stem: str, device: str) -> Any:
    """the fixture loaded once for the module on `device`"""
    if device == "mlx":
        need_mlx()
    key = (stem, device)
    if key not in _models:
        _models[key] = _open.enter_context(loaded_model(fixture(stem), device=device))
    return _models[key]


def solo(sm: Any, ids: Sequence[int], n: int, sampling: Sampling | None = None, **kw: Any) -> list[int]:
    return [int(t) for t in sm.generate(list(ids), n, eos=(), speculate=False, sampling=sampling, **kw).tokens]


def close(a: torch.Tensor, b: torch.Tensor, device: str) -> None:
    d = (a.float() - b.float()).abs().max().item()
    assert d <= TOL[device] * max(1.0, b.float().abs().max().item()), d


cells = pytest.mark.parametrize("device", DEVICES)
families = pytest.mark.parametrize("stem", STEMS)


# -- hooks ---------------------------------------------------------------------------------------------------------


@cells
@families
@pytest.mark.parametrize("speculate", [False, True])
def test_hooks_leave_the_answer_as_it_was(stem: str, device: str, speculate: bool) -> None:
    """a processor that changes nothing, logprobs and taps: the paths that pick in their graph stand aside and
    the tokens are the unhooked decode's"""
    sm = model(stem, device)
    plain = list(sm.generate(PROMPT, N, eos=(), speculate=speculate).tokens)
    g = sm.generate(PROMPT, N, eos=(), speculate=speculate, processors=[lambda ids, lg: lg], logprobs=2, taps=[0, -1])
    assert list(g.tokens) == plain
    assert [t.token for t in g.logprobs] == plain and all(len(t.top) == 2 for t in g.logprobs)
    assert sorted(g.hidden) == [0, sm.L - 1] and all(h.shape[0] == len(plain) for h in g.hidden.values())


@cells
@families
@pytest.mark.parametrize("speculate", [False, True])
def test_a_processor_decides_every_pick(stem: str, device: str, speculate: bool) -> None:
    """a processor that allows one token a position steers the answer to exactly those tokens - a speculative
    pass's drafts judged under the same rule, each with its own ids"""
    sm = model(stem, device)
    want = [3, 1, 4, 1, 5, 9, 2, 6]

    def force(ids: Sequence[int], lg: torch.Tensor) -> torch.Tensor:
        # every candidate row, a rejected draft's included, carries the prompt and then its own path
        k = len(ids) - len(PROMPT)
        assert list(ids[: len(PROMPT)]) == PROMPT and 0 <= k < len(want)
        out = torch.full_like(lg, float("-inf"))
        out[want[k]] = 0.0
        return out

    assert list(sm.generate(PROMPT, len(want), eos=(), speculate=speculate, processors=[force]).tokens) == want


@cells
@families
def test_logprobs_are_the_distribution_the_token_was_drawn_from(stem: str, device: str) -> None:
    """each token's logprob and its top-k are log_softmax of the logits at its position, the logits a session
    fed the same tokens returns"""
    sm = model(stem, device)
    g = sm.generate(PROMPT, N, eos=(), speculate=False, logprobs=3)
    toks = list(g.tokens)
    s = sm.session(PROMPT)
    ref = torch.log_softmax(torch.cat([s.logits[None], s.feed(toks[:-1])]).float(), -1)
    for k, t in enumerate(g.logprobs):
        close(torch.tensor(t.logprob), ref[k, t.token], device)
        if device == "cpu":
            assert [i for i, _ in t.top] == torch.topk(ref[k], 3).indices.tolist()


@cells
@families
def test_speculative_logprobs_are_each_positions_own(stem: str, device: str) -> None:
    """a sampled speculative decode records, for each token it commits, the log-probability at that token's own
    position - an accepted draft's under its verify row, not the pass's first"""
    sm = model(stem, device)
    g = sm.generate(PROMPT, N, eos=(), speculate=True, sampling=SMP, logprobs=0)
    toks = list(g.tokens)
    s = sm.session(PROMPT)
    ref = torch.log_softmax(torch.cat([s.logits[None], s.feed(toks[:-1])]).float(), -1)
    got = torch.tensor([t.logprob for t in g.logprobs])
    want = ref[torch.arange(len(toks)), torch.tensor(toks)]
    # a verify pass and a sequential feed differ like a prefill and a step do (a hybrid's chunked DeltaNet most)
    assert (got - want).abs().max().item() <= (1e-2 if device == "cpu" else 0.1)


@cells
@families
@pytest.mark.parametrize("speculate", [False, True])
def test_on_pass_counts_every_token(stem: str, device: str, speculate: bool) -> None:
    """the passes' stats add up to the answer: their tokens, in order, accepted drafts never above drafted"""
    sm = model(stem, device)
    seen: list[Any] = []
    g = sm.generate(PROMPT, N, eos=(), speculate=speculate, on_pass=seen.append)
    assert [p["index"] for p in seen] == list(range(len(seen)))
    assert sum(p["tokens"] for p in seen) == len(g.tokens)
    assert all(0 <= p["accepted"] <= p["drafted"] and p["seconds"] >= 0 for p in seen)
    assert g.stats["tokens_per_pass"] == pytest.approx(len(g.tokens) / len(seen), abs=1e-3)


# -- hidden states ---------------------------------------------------------------------------------------------------


@cells
@families
def test_taps_are_the_hidden_states_at_each_pick(stem: str, device: str) -> None:
    """a tapped layer's state at each new token is its output at the position that token was picked from: the
    nearest of a prefill's rows, and as near as the residual stream's bf16 lets two computations of it be"""
    sm = model(stem, device)
    g = sm.generate(PROMPT, N, eos=(), speculate=False, taps=[1])
    toks = list(g.tokens)
    h = sm.hidden(PROMPT + toks[:-1], layers=(1,))[1][len(PROMPT) - 1 :]
    dist = torch.cdist(g.hidden[1].float(), h.float())
    assert dist.argmin(dim=1).tolist() == list(range(len(toks)))
    assert (g.hidden[1] - h).abs().max().item() <= 3e-2 * h.abs().max().item()


@cells
@families
def test_the_last_layer_through_the_head_is_the_logits(stem: str, device: str) -> None:
    sm = model(stem, device)
    h = sm.hidden(PROMPT, layers=(-1,))[sm.L - 1]
    close(sm.project(h[-1]), sm.session(PROMPT).logits, device)


@cells
@families
def test_encode_gives_a_unit_vector_a_text(stem: str, device: str) -> None:
    sm = model(stem, device)
    v = sm.encode(["hello there", "a second text"])
    assert v.shape == (2, sm.hidden([1, 2], layers=(-1,))[sm.L - 1].shape[-1])
    assert torch.allclose(v.norm(dim=-1), torch.ones(2), atol=1e-4)
    m = sm.encode("hello there", pool="mean", normalize=False)
    assert m.shape[0] == 1


# -- a session by hand -------------------------------------------------------------------------------------------------


@cells
@families
def test_rewind_returns_the_session_to_the_mark(stem: str, device: str) -> None:
    """after a rewind the session is what it was at the mark: the same logits for the same tokens, a hybrid's
    recurrent states restored with the attention rows cut"""
    sm = model(stem, device)
    s = sm.session(PROMPT)
    m = s.mark()
    first = s.feed(OTHER)
    s.feed([1, 2, 3])
    s.rewind(m)
    assert s.tokens == PROMPT
    again = s.feed(OTHER)
    assert torch.equal(first, again)
    s.rewind(m)
    assert list(s.generate(N, eos=(), speculate=False).tokens) == solo(sm, PROMPT, N)


@cells
@families
def test_a_session_goes_on_after_generate(stem: str, device: str) -> None:
    """generate leaves its last token drawn and not fed; the next call feeds it first"""
    sm = model(stem, device)
    s = sm.session(PROMPT)
    a = list(s.generate(4, eos=(), speculate=False).tokens)
    assert s.tokens == PROMPT + a and s.pending == a[-1]
    b = list(s.generate(4, eos=(), speculate=False).tokens)
    assert a + b == solo(sm, PROMPT, 8)


@cells
@families
def test_crop_is_a_rewind_to_that_length(stem: str, device: str) -> None:
    """a dense session crops to any length and decodes on as a fresh one there would; a hybrid is refused and told
    to mark the point"""
    sm = model(stem, device)
    s = sm.session(PROMPT)
    s.feed(OTHER)
    if LayerKind.LINEAR in sm.layer_types:
        with pytest.raises(ValueError, match="mark the point"):
            s.crop(len(PROMPT))
        return
    s.crop(len(PROMPT))
    assert s.tokens == PROMPT
    assert list(s.generate(N, eos=(), speculate=False).tokens) == solo(sm, PROMPT, N)
    with pytest.raises(ValueError, match="crop to"):
        s.crop(len(s) + 1)


@cells
@families
def test_rows_are_copies_of_the_caches_rows(stem: str, device: str) -> None:
    """`rows` hands back the cache's keys and values as tensors of the caller's own: writing to them leaves the
    session's next logits as they were"""
    sm = model(stem, device)
    s = sm.session(PROMPT)
    m = s.mark()
    i = next(j for j, kind in enumerate(sm.layer_types) if kind != LayerKind.LINEAR)
    k, v = s.rows(i)
    assert k.shape[0] == 1 and k.shape[-2] <= len(PROMPT) and v.shape == k.shape
    before = s.feed(OTHER)
    s.rewind(m)
    k.fill_(0.0)
    v.fill_(0.0)
    assert torch.equal(s.feed(OTHER), before)
    if LayerKind.LINEAR in sm.layer_types:
        with pytest.raises(ValueError, match="linear-attention"):
            s.rows(sm.layer_types.index(LayerKind.LINEAR))


def test_a_mark_past_the_end_is_refused() -> None:
    sm = model(STEMS[0], "cpu")
    m = sm.session(PROMPT).mark()
    with pytest.raises(ValueError, match="past"):
        sm.session(PROMPT[:3]).rewind(m)


# -- a turn continued ------------------------------------------------------------------------------------------------


@families
def test_continue_final_leaves_the_assistant_turn_open(stem: str) -> None:
    sm = model(stem, "cpu")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Sure, x"}]
    assert template(sm.tokenizer, msgs, continue_final=True).endswith("Sure, x")
    ids = sm._reply_ids("hi", False, "Sure, x")
    assert ids == sm.prompt_ids(msgs, continue_final=True)
    assert sm.ask("hi", 4, prefill="Sure, x").startswith("Sure, x")


# -- a fork's rows ---------------------------------------------------------------------------------------------------


@cells
@families
def test_a_forks_rows_are_the_sessions_own_draws(stem: str, device: str) -> None:
    """row r of a fork draws what the session's own decode draws under `sampling.row(r)`; row 0 is the session's
    own draw. Kept, a row's session goes on as one fed the same tokens would."""
    sm = model(stem, device)
    br = sm.session(PROMPT).fork(3)
    g = br.generate(N, eos=(), sampling=SMP, logprobs=0)
    if device == "cpu" or br.mode == "rows":
        for r in range(3):
            assert g.tokens[r] == solo(sm, PROMPT, N, SMP.row(r)), r
    assert len({tuple(t) for t in g.tokens}) > 1, "the rows drew alike"
    s = br.keep(1)
    assert s.forked is None and s.tokens == PROMPT + g.tokens[1]
    by_hand = sm.session(PROMPT)
    by_hand.feed(g.tokens[1])
    close(s.feed([7])[-1], by_hand.feed([7])[-1], device)


@cells
@families
def test_a_forks_step_is_each_rows_own_pass(stem: str, device: str) -> None:
    """the rows' logits after a step are those of a session fed the same tokens; a reorder copies rows"""
    sm = model(stem, device)
    s = sm.session(PROMPT)
    with s.fork(2) as br:
        close(br.logits[0], s.logits, device)
        br.step([1, 2])
        br.reorder([1, 1, 0])
        lg = br.step([4, 5, 6])
        close(lg[0], sm.session(PROMPT + [2]).feed([4])[-1], device)
        close(lg[1], sm.session(PROMPT + [2]).feed([5])[-1], device)
        close(lg[2], sm.session(PROMPT + [1]).feed([6])[-1], device)
        assert br.rows == [[2, 4], [2, 5], [1, 6]]
    assert s.forked is None and s.tokens == PROMPT


@cells
@families
def test_a_row_leaves_at_its_stop_token(stem: str, device: str) -> None:
    """a row that draws a stop token leaves the batch; the others go on and still draw their own tokens"""
    sm = model(stem, device)
    ref = [solo(sm, PROMPT, N, SMP.row(r)) for r in range(3)]
    stop = ref[1][2]
    s = sm.session(PROMPT)
    br = s.fork(3)
    g = br.generate(N, eos=(stop,), sampling=SMP)
    for r in range(3):
        cut = ref[r][: ref[r].index(stop) + 1] if stop in ref[r] else ref[r]
        if device == "cpu" or br.mode == "rows":
            assert g.tokens[r] == cut, r
        assert (r in br.live) == (stop not in g.tokens[r])
    kept = br.keep(1)
    assert kept.pending == stop and kept.tokens == PROMPT + g.tokens[1]


def test_a_forked_session_holds_still() -> None:
    sm = model(STEMS[0], "cpu")
    s = sm.session(PROMPT)
    br = s.fork(2)
    with pytest.raises(ValueError, match="forked"):
        s.feed([1])
    with pytest.raises(ValueError, match="forked"):
        s.fork(2)
    br.close()
    s.feed([1])
    assert s.tokens == PROMPT + [1]


# -- a batch of sessions -----------------------------------------------------------------------------------------------


@cells
@families
def test_a_batch_decodes_each_session_as_its_own(stem: str, device: str) -> None:
    """sessions of their own lengths decoded together, one joining between calls: each row draws what its
    session would alone, and each session goes on from its row's end once the batch closes"""
    sm = model(stem, device)
    a, b = sm.session(PROMPT), sm.session(OTHER)
    bt = sm.batch([a, b])
    g1 = bt.generate(4, eos=(), sampling=SMP)
    c = sm.session(LONG)
    rc = bt.join(c)
    g2 = bt.generate(4, eos=(), sampling=SMP)
    bt.close()
    assert a.forked is b.forked is c.forked is None
    assert a.tokens == PROMPT + g1.tokens[0] + g2.tokens[0]
    if device == "cpu" or bt.mode == "rows":
        assert g1.tokens[0] + g2.tokens[0] == solo(sm, PROMPT, 8, SMP)
        assert g1.tokens[1] + g2.tokens[1] == solo(sm, OTHER, 8, SMP)
        assert g2.tokens[rc] == solo(sm, LONG, 4, SMP)
    if device == "cpu":
        assert list(a.generate(3, eos=(), sampling=SMP, speculate=False).tokens) == solo(sm, PROMPT, 11, SMP)[8:]


@cells
@families
def test_a_batch_row_is_written_back_when_it_stops(stem: str, device: str) -> None:
    sm = model(stem, device)
    ref = solo(sm, OTHER, N)
    stop = ref[2]
    a, b = sm.session(PROMPT), sm.session(OTHER)
    with sm.batch([a, b]) as bt:
        g = bt.generate(N, eos=(stop,))
        assert b.forked is None and 1 not in bt.live
        assert b.tokens == OTHER + g.tokens[1] and b.pending == stop
        if device == "cpu" or bt.mode == "rows":
            assert g.tokens[1] == ref[: ref.index(stop) + 1]
    assert a.forked is None


@cells
@families
def test_generate_takes_rows_of_their_own_lengths(stem: str, device: str) -> None:
    sm = model(stem, device)
    g = sm.generate([PROMPT, OTHER], 4, eos=(), speculate=False, logprobs=0)
    assert len(g.tokens) == 2 and all(len(t) == 4 for t in g.tokens)
    assert [len(lp) for lp in g.logprobs] == [4, 4]
    if device == "cpu":
        assert g.tokens == [solo(sm, PROMPT, 4), solo(sm, OTHER, 4)]
