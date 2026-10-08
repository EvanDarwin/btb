# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's prefix cache (btb/engine/prefix.py): every session's rows in one pool of pages under one tree, a
prompt opening on the longest prefix any conversation left. On each dense family's fixture, one load each - on the
CPU, and on the card in each placement its rows take (the card's region under the card graph, host layers beside it,
every row in RAM under `kv_host`):

* paged rows give the contiguous cache's bits - a prompt, a continuation, a one-row step, greedy and speculative
  decodes, a reuse, on the card through the graphs and through the torch layers alike - every logit `torch.equal`
  on the same load with the prefix cache off;
* a side request between two turns leaves the conversation where it was: the next turn opens on its rows - on the
  card brought back from the RAM they were parked in meanwhile - and answers as the uninterrupted conversation does;
* two conversations opening alike read the same rows of one set of pages, held once;
* a layer the card gives up takes every conversation's rows to the host and back, the answers as a contiguous
  cache's;
* an engine the prefix cache does not serve yet says why, and its sessions keep contiguous caches."""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.paged import PagedCache
from btb.engine.prefix import PrefixCache
from btb.kinds import PassTag
from tests.helpers import FIXTURES, fixture, shared_key, shared_model

DENSE = ["tiny_gemma3", "tiny_phi3", "tiny_qwen3"]
OTHERS = ["tiny_gpt_oss", "tiny_q35", "tiny_q4"]
VOCAB = 200  # inside every fixture's vocabulary
# where a conversation's rows lie: the host's region on the CPU; on the card, the card's region under the card graph
# (`card`), beside a layer the host runs (`split`), or every row in RAM (`kvhost`), each placement pinned (`adapt` off);
# and the card as `btb serve` loads it (`served`: the options' defaults, its memory policies reading every pass's cache)
PLACES: dict[str, dict[str, Any]] = {
    "cpu": {"device": "cpu"},
    "card": {"device": "cuda", "adapt": 0},
    "split": {"device": "cuda", "adapt": 0, "cpu_layers": 1},
    "kvhost": {"device": "cuda", "adapt": 0, "kv_host": 1},
    "served": {"device": "cuda"},
}
ON_CARD = ["card", "split", "kvhost", "served"]
CARD = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")


def model(stem: str, place: str = "cpu") -> StreamedTextModel:
    """the fixture in a placement, loaded once for every test on it (tests.helpers.shared_model)"""
    return shared_model(fixture(stem), **PLACES[place])


def shared_model_keys(item: pytest.Item) -> list[tuple[object, ...]]:
    p = getattr(getattr(item, "callspec", None), "params", {})
    return [shared_key(os.path.join(FIXTURES, p.get("stem", DENSE[0])), PLACES[p.get("place", "cpu")])]


def prefix(sm: StreamedTextModel) -> PrefixCache:
    """the engine's prefix cache, the conversations earlier tests left let go"""
    pc = sm._prefix_cache()
    assert pc is not None, PrefixCache.why_not(sm)
    pc.tree.evict()
    return pc


@contextlib.contextmanager
def contiguous(sm: StreamedTextModel) -> Iterator[None]:
    """the engine's sessions on contiguous caches of their own, as an engine without a prefix cache keeps them"""
    held = sm.__dict__.get("_kv")
    sm.__dict__["_kv"] = None
    try:
        yield
    finally:
        sm.__dict__["_kv"] = held


def toks(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(3, VOCAB) for _ in range(n)]


def chatty(rng: random.Random, n: int) -> list[int]:
    """`n` tokens repeating a few phrases: a prompt the speculation's lookup drafts from"""
    phrases = [toks(rng, rng.randint(3, 6)) for _ in range(4)]
    out: list[int] = []
    while len(out) < n:
        out += rng.choice(phrases)
    return out[:n]


def walk(sm: StreamedTextModel, seed: int) -> list[torch.Tensor | list[int]]:
    """one session's run, everything it saw: a prompt past one prefill chunk, a continuation, a one-row step,
    greedy and speculative decodes, a reuse of most of it with a new ending, the logits after each"""
    rng = random.Random(seed)
    s = sm.session()
    seen: list[torch.Tensor | list[int]] = []
    prompt = chatty(rng, 150)
    seen.append(s.feed(prompt).logits)
    seen.append(s.feed(toks(rng, 5)).logits)
    seen.append(s.feed(toks(rng, 1)).logits)
    keep = sm.prefill_chunk
    sm.prefill_chunk = 16  # prompts past a chunk: their later chunks over the rows before them
    try:
        seen.append(s.feed(chatty(rng, 70)).logits)  # the chunk loop (every row's logits)
        seen.append(s.feed(chatty(rng, 50), last_only=True).logits)  # the layer-by-layer sweep
    finally:
        sm.prefill_chunk = keep
    for spec in (False, True):
        seen.append(list(s.generate(12, eos=(), speculate=spec).tokens))
        seen.append(s.next_logits())
    tail = chatty(rng, 30)
    tail[0] = s.tokens[120] % (VOCAB - 4) + 4  # parting from the session at 120
    again = [*s.tokens[:120], *tail]
    seen.append(list(sm.generate(again, 10, eos=(), session=s, speculate=True).tokens))
    assert s.last_reuse == 120
    seen.append(s.next_logits())
    # a decode no session lends a cache to: on the card the self-advancing step graph over every row it reaches
    seen.append(list(sm.generate(chatty(rng, 40), 12, eos=(), speculate=False).tokens))
    return seen


PLACED = ["cpu", *(pytest.param(p, marks=CARD) for p in ON_CARD)]


def same(paged: list[torch.Tensor | list[int]], flat: list[torch.Tensor | list[int]], what: str) -> None:
    for j, (got, want) in enumerate(zip(paged, flat, strict=True)):
        if isinstance(want, torch.Tensor):
            assert isinstance(got, torch.Tensor) and torch.equal(got, want), f"{what}, step {j}: logits apart"
        else:
            assert got == want, f"{what}, step {j}: {got} != {want}"


@pytest.mark.parametrize("place", PLACED)
@pytest.mark.parametrize("stem", DENSE)
def test_paged_rows_give_the_contiguous_caches_bits(stem: str, place: str) -> None:
    """on the card through its graphs and, with them off, through the torch layers: the paged and the contiguous
    cache's every logit the same, their attention one set of kernels"""
    sm = model(stem, place)
    graphs = [True, False] if place == "card" else [True]
    keep = getattr(sm, "card_graphs", True)
    try:
        for on in graphs:
            sm.__dict__.update(card_graphs=on)
            prefix(sm)  # each walk from nothing kept: the last one's conversations would open this one's prompts
            paged = walk(sm, 1)
            assert PassTag.KV_PAGED in sm.last_pass_report()
            with contiguous(sm):
                flat = walk(sm, 1)
                assert PassTag.KV_CONTIGUOUS in sm.last_pass_report()
            same(paged, flat, f"{place}{'' if on else ', card graphs off'}")
    finally:
        sm.__dict__.update(card_graphs=keep)


@pytest.mark.parametrize("place", PLACED)
@pytest.mark.parametrize("stem", DENSE)
def test_a_side_request_leaves_the_conversation_where_it_was(stem: str, place: str) -> None:
    """a turn of conversation A, an unrelated prompt on the same session (a title call, a sub-agent), A's next turn:
    the next turn opens on every row A left - its prompt and its answer but the answer's last token, still pending
    when the side request came - and answers as A uninterrupted on a contiguous cache does, logits and all (the same
    rows prefilled from the same point). On the card the side request parks A's pages in RAM, and A's next turn
    brings them back"""
    sm = model(stem, place)
    rng = random.Random(2)
    a1, more, side = toks(rng, 100), toks(rng, 20), toks(rng, 50)
    side[0] = (a1[0] + 1) % VOCAB or 3  # nothing in common with A
    prefix(sm)
    s = sm.session()
    out1 = list(sm.generate(a1, 6, eos=(), session=s, speculate=False).tokens)
    sm.generate(side, 4, eos=(), session=s, speculate=False)
    assert s.last_reuse == 0
    got = list(sm.generate([*a1, *out1, *more], 6, eos=(), session=s, speculate=False).tokens)
    assert s.last_reuse == len(a1) + len(out1) - 1, "the next turn did not find the conversation's rows"
    rep = sm.last_pass_report()
    assert PassTag.PREFIX_SHARED in rep and PassTag.PREFIX_HIT in rep and PassTag.KV_PAGED in rep
    if place in ("card", "split", "served"):
        assert PassTag.KV_PARK in rep, "A's pages were not brought back from RAM"
    after = s.next_logits()
    with contiguous(sm):
        alone = sm.session()
        assert list(sm.generate(a1, 6, eos=(), session=alone, speculate=False).tokens) == out1
        assert list(sm.generate([*a1, *out1, *more], 6, eos=(), session=alone, speculate=False).tokens) == got
        assert torch.equal(after, alone.next_logits())


@pytest.mark.parametrize("place", PLACED)
@pytest.mark.parametrize("stem", DENSE)
def test_two_conversations_opening_alike_read_the_same_rows(stem: str, place: str) -> None:
    """a second session whose prompt opens as the first's did reads the first's rows in place: the same rows of the
    same pages, held once, and answers as the first session would going there itself on a contiguous cache"""
    sm = model(stem, place)
    rng = random.Random(3)
    system, u1, u2 = toks(rng, 300), toks(rng, 20), toks(rng, 20)
    pc = prefix(sm)
    s1 = sm.session()
    s1.feed([*system, *u1])
    held = len(pc.pool.pages)
    s2 = sm.session()
    got = list(sm.generate([*system, *u2], 6, eos=(), session=s2, speculate=False).tokens)
    assert s2.last_reuse == len(system) and PassTag.PREFIX_SHARED in sm.last_pass_report()
    c1, c2 = s1.cache, s2.cache
    assert isinstance(c1, PagedCache) and isinstance(c2, PagedCache)
    assert torch.equal(c1.table.rows()[: len(system)], c2.table.rows()[: len(system)])
    shared = pc.pool.pages.of(c2.table.rows()[: len(system)].tolist())
    assert all(p.refs >= 3 for p in shared), "the rows read in place are not held by both and the tree"
    own = len(pc.pool.pages) - held
    assert own <= -(-(len(u2) + 6) // 64) + 1, f"the second conversation took {own} pages for its {len(u2)} rows"
    k1, _ = s1.rows(0)
    k2, _ = s2.rows(0)
    assert torch.equal(k1[..., : len(system), :], k2[..., : len(system), :])
    after = s2.next_logits()
    with contiguous(sm):
        alone = sm.session()
        alone.feed([*system, *u1])
        assert list(sm.generate([*system, *u2], 6, eos=(), session=alone, speculate=False).tokens) == got
        assert alone.last_reuse == len(system) and torch.equal(after, alone.next_logits())


@CARD
@pytest.mark.parametrize("stem", DENSE)
def test_a_layer_the_card_gives_up_takes_every_conversations_rows_and_back(stem: str) -> None:
    """a layer shed to the host and regrown on the card takes every conversation's rows of it each way - the one the
    card holds and one parked in RAM - once for all of them; the conversations go on as contiguous caches moved the
    same way do, logits and all. Last on its load: the placement it leaves is another load's"""
    sm = model(stem, "card")
    rng = random.Random(4)
    p1, p2, more = toks(rng, 90), toks(rng, 70), toks(rng, 10)
    p2[0] = (p1[0] + 1) % VOCAB or 3

    def run() -> list[torch.Tensor | list[int]]:
        seen: list[torch.Tensor | list[int]] = []
        a, b = sm.session(), sm.session()
        seen.append(a.feed(p1).logits)
        seen.append(b.feed(p2).logits)  # a's pages parked
        i = max(sm.resident)
        sm.device.request("shed", sm.vram_shed)  # as the memory policy moves a layer: the placement's version moves
        assert i in sm.host
        seen.append(list(a.generate(4, eos=(), speculate=False).tokens))
        seen.append(b.feed(more).logits)
        sm.device.request("regrow", sm.vram_regrow)
        assert i in sm.resident
        seen.append(list(a.generate(4, eos=(), speculate=True).tokens))
        seen.append(list(b.generate(4, eos=(), speculate=False).tokens))
        return seen

    pc = prefix(sm)
    paged = run()
    card = pc.pool.card
    assert card is not None and sorted(card.layers) == sorted(i for i in sm.resident), "the layers' rows not home"
    with contiguous(sm):
        flat = run()
    same(paged, flat, "a layer shed and regrown")


@pytest.mark.parametrize("stem", OTHERS)
def test_an_engine_the_prefix_cache_does_not_serve_says_why(stem: str) -> None:
    """a hybrid, a family with its own layer, a family whose attention is its own module's: the reason, and its
    sessions' caches contiguous ones of their own"""
    sm = model(stem)
    assert PrefixCache.why_not(sm) and sm._prefix_cache() is None
    s = sm.session([5, 6, 7, 8])
    assert not getattr(s.cache, "paged", False)
    assert PassTag.KV_PAGED not in sm.last_pass_report()
