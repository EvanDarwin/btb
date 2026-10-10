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

from btb.engine import MemoryGrantError, StreamedTextModel
from btb.engine.cache import ForkLayer
from btb.engine.cuda import CardPassFailed
from btb.engine.kvpool import PAGE
from btb.engine.paged import PagedCache
from btb.engine.prefix import PrefixCache
from btb.kinds import PassTag
from tests.cert.spec import card_rows, kind_of_stem
from tests.helpers import FIXTURES, fixture, shared_key, shared_model

DENSE = ["tiny_gemma3", "tiny_phi3", "tiny_qwen3"]
# the dense families the card graph serves, whose prompts take its kernels (`_forward_card_prefill`), by the cert's
# own rule (`card_rows`): Phi-3's fused projections run its torch modules (`Missing.PREFIX_INVARIANCE`)
GRAPHED = [s for s in DENSE if (k := kind_of_stem(s)) is not None and card_rows(k)]
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
    """the engine's sessions on contiguous caches of their own, as an engine without a prefix cache keeps them - the
    engine left as it was found after: a prefix cache not made yet is made when next asked for, never left off for
    the tests after on the same load"""
    made = "_kv" in sm.__dict__
    held = sm.__dict__.get("_kv")
    sm.__dict__["_kv"] = None
    try:
        yield
    finally:
        if made:
            sm.__dict__["_kv"] = held
        else:
            sm.__dict__.pop("_kv", None)


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
    rows prefilled from the same point). A side request sharing nothing with A, and one sharing its opening tokens
    (the chat template's header), whose prompt the session's own rows serve: A's rows go to the tree before the
    session parts from them. On the card the side request, longer than the card's region holds, parks A's pages in RAM
    for its room, and A's next turn brings them back"""
    sm = model(stem, place)
    card = prefix(sm).pool.card
    for shared in (0, 3):
        rng = random.Random(2)
        a1, more, side = toks(rng, 100), toks(rng, 20), toks(rng, ((card.cap if card is not None else 0) + 1) * PAGE)
        side[:shared] = a1[:shared]
        side[shared] = (a1[shared] + 1) % VOCAB or 3  # nothing more in common with A
        prefix(sm)
        s = sm.session()
        out1 = list(sm.generate(a1, 6, eos=(), session=s, speculate=False).tokens)
        sm.generate(side, 4, eos=(), session=s, speculate=False)
        assert s.last_reuse == shared
        got = list(sm.generate([*a1, *out1, *more], 6, eos=(), session=s, speculate=False).tokens)
        assert s.last_reuse == len(a1) + len(out1) - 1, f"the next turn did not find the conversation's rows ({shared})"
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
@pytest.mark.parametrize("place", ["card"])  # named, so the test declares the load it takes (`shared_model_keys`)
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_hit_decodes_as_its_prompt_cold(stem: str, place: str) -> None:
    """on the card, a conversation's next turn read from the cache - its rows made by the turn before's prefill and
    by its decode's steps and verify passes - answers as the same prompt decoded cold, every logit and token: each of
    a prompt's rows is the row its step makes (`_forward_card_prefill`), whole or in chunks, greedy or speculative"""
    sm = model(stem, place)
    rng = random.Random(4)
    a1, more = chatty(rng, 150), chatty(rng, 40)
    keep = sm.prefill_chunk
    try:
        for chunk in (None, 16):  # the prompt whole, or in chunks: the chunk loop and the layer-by-layer sweep
            for spec in (False, True):
                sm.prefill_chunk = chunk
                prefix(sm)
                s = sm.session()
                out1 = list(sm.generate(a1, 12, eos=(), session=s, speculate=spec).tokens)
                got = list(sm.generate([*a1, *out1, *more], 12, eos=(), session=s, speculate=spec).tokens)
                assert s.last_reuse >= len(a1) + len(out1) - 1, "the next turn did not find the conversation's rows"
                assert PassTag.PREFIX_HIT in sm.last_pass_report()
                after = s.next_logits()
                prefix(sm)  # nothing kept: the same prompt cold
                cold = sm.session()
                want = list(sm.generate([*a1, *out1, *more], 12, eos=(), session=cold, speculate=spec).tokens)
                assert cold.last_reuse == 0
                assert PassTag.CARD_PREFILL in sm.last_pass_report(), "the cold prompt did not take the card's kernels"
                assert got == want, f"chunks of {chunk}, speculative {spec}: the hit {got} against cold {want}"
                assert torch.equal(after, cold.next_logits()), f"chunks of {chunk}, speculative {spec}: logits apart"
    finally:
        sm.prefill_chunk = keep


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_prompts_card_rows_outlive_another_taking_the_arena(stem: str, place: str) -> None:
    """a contiguous cache's fresh prompt, its rows written by the card's kernels straight into the arena, keeps every
    one of them when another session's prompt takes the arena - copied out as an appended cache's are (left
    uninitialized, the eviction took the layer for empty and let them go) - and goes on as it would have alone"""
    sm = model(stem, place)
    rng = random.Random(5)
    p1, p2, more = toks(rng, 90), toks(rng, 70), toks(rng, 10)
    p2[0] = (p1[0] + 1) % VOCAB or 3
    with contiguous(sm):
        a = sm.session()
        a.feed(p1)
        assert PassTag.CARD_PREFILL in sm.last_pass_report()
        held = [a.rows(i) for i in range(sm.L)]
        b = sm.session()
        b.feed(p2)  # the arena the second prompt's
        assert sm._card_state()["arena"]["owner"]() is b.cache, "the second prompt did not take the arena"
        for i, (k, v) in enumerate(held):
            k2, v2 = a.rows(i)
            assert torch.equal(k, k2) and torch.equal(v, v2), f"layer {i}: the first prompt's rows were let go"
        got = a.feed(more).logits
        alone = sm.session()
        alone.feed(p1)
        assert torch.equal(got, alone.feed(more).logits)


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_sweep_leaves_the_arena_with_the_cache_holding_it(stem: str, place: str) -> None:
    """contiguous caches: a prompt swept layer by layer while another session's rows hold the card graphs' arena
    takes the torch layers over rows of its own - the arena, its holder's rows and its holder's next turn as they
    were (bound by the sweep's layers, the arena was taken from its holder at the length the new cache had, its
    chunks' rows written past it)"""
    sm = model(stem, place)
    rng = random.Random(9)
    p1, p2, more = toks(rng, 90), toks(rng, 150), toks(rng, 10)
    p2[0] = (p1[0] + 1) % VOCAB or 3
    keep = sm.prefill_chunk
    with contiguous(sm):
        a = sm.session()
        a.feed(p1)
        assert sm._card_state()["arena"]["owner"]() is a.cache
        held = [a.rows(i) for i in range(sm.L)]
        sm.prefill_chunk = 16
        try:
            b = sm.session()
            b.feed(p2, last_only=True)  # the layer-by-layer sweep
        finally:
            sm.prefill_chunk = keep
        assert sm._card_state()["arena"]["owner"]() is a.cache, "the sweep took the arena from the session holding it"
        assert b.cache is not None and b.cache.get_seq_length() == len(p2)
        for i, (k, v) in enumerate(held):
            k2, v2 = a.rows(i)
            assert torch.equal(k, k2) and torch.equal(v, v2), f"layer {i}: the holder's rows moved under the sweep"
        got = a.feed(more).logits
        alone = sm.session()
        alone.feed(p1)
        assert torch.equal(got, alone.feed(more).logits)


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_prompts_logits_are_its_own_whatever_its_chunks(stem: str, place: str) -> None:
    """every row's logits of a prompt fed in chunks are the whole prompt's, paged or contiguous: each chunk's its
    own, never the buffer the next chunk's tail writes again"""
    sm = model(stem, place)
    p = chatty(random.Random(6), 150)
    keep = sm.prefill_chunk
    got: dict[tuple[int | None, bool], torch.Tensor] = {}
    try:
        for chunk in (None, 16):
            sm.prefill_chunk = chunk
            for paged in (True, False):
                prefix(sm)
                with contextlib.nullcontext() if paged else contiguous(sm):
                    got[(chunk, paged)] = sm.session().feed(p).logits
    finally:
        sm.prefill_chunk = keep
    want = got[(None, True)]
    assert want.shape[0] == len(p)
    for (chunk, paged), lg in got.items():
        assert torch.equal(lg, want), f"chunks of {chunk}, {'paged' if paged else 'contiguous'}: logits apart"


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_hooked_decode_speculates_as_it_steps(stem: str, place: str) -> None:
    """taps on: the verify passes - trees of the n-gram drafts - and the steps both take the card's kernels, so the
    speculative decode's tokens and every tapped layer's states are the plain decode's, bit for bit (a hooked tree
    took the torch layers, its steps the card's)"""
    sm = model(stem, place)
    p = chatty(random.Random(7), 120)
    g: dict[bool, Any] = {}
    for spec in (False, True):
        prefix(sm)
        g[spec] = sm.generate(p, 24, eos=(), speculate=spec, taps=[0, -1])
    assert list(g[True].tokens) == list(g[False].tokens)
    hs, hg = g[True].hidden, g[False].hidden
    assert hs is not None and hg is not None and sorted(hs) == sorted(hg) == [0, sm.L - 1]
    for i in hg:
        assert torch.equal(hs[i], hg[i]), f"layer {i}: the speculative decode's taps apart from the steps'"


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_card_prefill_short_of_room_runs_the_torch_path(
    stem: str, place: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the card prefill's buffers refused (the ledger's MemoryGrantError), or the card out of memory at its tail: the
    pass runs the torch layers over the cache as it found it - every row written once - and answers as those layers
    do with the card graph off; a hooked pass out of room past its hook's first layer raises (`CardPassFailed`), its
    hook never fed the layers twice"""
    sm = model(stem, place)
    rng = random.Random(8)
    p = toks(rng, 90)
    take = sm.scratch.take

    def refuse(name: str, *a: Any, **kw: Any) -> torch.Tensor:
        if name.startswith("card prefill"):
            raise MemoryGrantError(f"{name}: refused for the test")
        return take(name, *a, **kw)

    def full(*a: Any, **kw: Any) -> torch.Tensor:
        raise torch.OutOfMemoryError("CUDA out of memory (the test's)")

    keep = getattr(sm, "card_graphs", True)
    try:
        for paged in (True, False):
            with contextlib.nullcontext() if paged else contiguous(sm):
                sm.__dict__.update(card_graphs=False)
                if paged:
                    prefix(sm)
                want = sm.session().feed(p).logits
                sm.__dict__.update(card_graphs=keep)
                for short in ("buffers", "tail"):
                    what = f"{'paged' if paged else 'contiguous'}, short of {short}"
                    if paged:
                        prefix(sm)
                    if short == "buffers":
                        monkeypatch.setattr(sm.scratch, "take", refuse)
                    else:
                        monkeypatch.setattr(sm, "_card_tail", full)
                    try:
                        s = sm.session()
                        got = s.feed(p).logits
                        assert PassTag.CARD_PREFILL in sm.last_pass_report(), f"{what}: the card's kernels not tried"
                        sm.__dict__.pop("_card_off", None)  # the card graph back, off since the refusal
                        if paged:
                            pc = sm._prefix_cache()
                            assert pc is not None and pc.match(p).length == 0, f"{what}: torch's rows given to the tree"
                        if short == "tail":
                            with pytest.raises(CardPassFailed):
                                sm.session().feed(p, taps=[0])
                    finally:
                        monkeypatch.undo()
                        sm.__dict__.pop("_card_off", None)
                    assert s.cache is not None and s.cache.get_seq_length() == len(p), f"{what}: rows written twice"
                    assert torch.equal(got, want), f"{what}: the torch path's logits apart"
    finally:
        sm.__dict__.update(card_graphs=keep)
        sm.__dict__.pop("_card_off", None)


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", ["tiny_qwen3"])
def test_a_prompt_past_the_cards_grid_is_its_chunks_bits(stem: str, place: str) -> None:
    """a prompt of more rows than a launch takes on the card's grid.y (65535) in one pass - its rows' writes and its
    matmuls launched in slices - gives the last row's logits its chunks give"""
    sm = model(stem, place)
    rng = random.Random(10)
    p = [rng.randrange(3, VOCAB) for _ in range(70_000)]
    keep = sm.prefill_chunk
    got: dict[int, torch.Tensor] = {}
    try:
        for chunk in (1 << 17, 4096):
            sm.prefill_chunk = chunk
            prefix(sm)
            s = sm.session()
            got[chunk] = s.feed(p, last_only=True).logits
            assert PassTag.CARD_PREFILL in sm.last_pass_report()
            del s
    finally:
        sm.prefill_chunk = keep
        prefix(sm)
    assert torch.equal(got[1 << 17], got[4096])


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", DENSE)
def test_a_layer_the_card_gives_up_takes_every_conversations_rows_and_back(stem: str, place: str) -> None:
    """a layer shed to the host and regrown on the card takes every conversation's rows of it each way - the one the
    card holds and one parked in RAM - once for all of them; the conversations go on as contiguous caches moved the
    same way do, logits and all. Last on its load: the placement it leaves is another load's"""
    sm = model(stem, place)
    rng = random.Random(4)
    p1, p2, more = toks(rng, 90), toks(rng, 70), toks(rng, 10)
    p2[0] = (p1[0] + 1) % VOCAB or 3

    def run() -> list[torch.Tensor | list[int]]:
        seen: list[torch.Tensor | list[int]] = []
        a, b = sm.session(), sm.session()
        seen.append(a.feed(p1).logits)
        seen.append(b.feed(p2).logits)
        pc = sm._prefix_cache()
        card = pc.pool.card if pc is not None else None
        if card is not None:
            # every page b does not read parked, as a request wanting the card's slots would park them: a's among them
            assert isinstance(b.cache, PagedCache)
            others = sum(1 for p in card.slots if p is not None and p.id not in b.cache.table.held)
            card.reserve(len(card.free) + others, b.cache.table)
            assert isinstance(a.cache, PagedCache) and all(p.park >= 0 for p in a.cache.table.held.values())
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


@pytest.mark.parametrize("place", ["cpu", pytest.param("kvhost", marks=CARD)])
@pytest.mark.parametrize("stem", ["tiny_qwen3"])
def test_rows_off_the_conversation_are_their_own_and_asked_for(stem: str, place: str) -> None:
    """a fork and a batch of conversations in the pages are caches of their own - never a session's table read as
    paged (priced as its growth, bound to it every step) - whose rows are asked of the ledger as they grow (a paged
    layer hands its fork the pool's gate) and step as a contiguous session's fork and batch do; and an idle
    conversation holds none of its last pass's row lists"""
    sm = model(stem, place)
    rng = random.Random(11)
    p, q = toks(rng, 90), toks(rng, 40)

    def run() -> list[torch.Tensor]:
        seen: list[torch.Tensor] = []
        a, b = sm.session(), sm.session()
        a.feed(p)
        b.feed(q)
        if sm._prefix_cache() is not None:
            assert isinstance(a.cache, PagedCache) and a.cache.__dict__.get("_lists") is None, "an idle cache's lists"
        with a.fork(3) as br:
            assert br.cache is not None and not getattr(br.cache, "paged", False)
            assert all(cl.grant is not None for cl in br.cache.layers if isinstance(cl, ForkLayer))
            seen += [br.step([5, 6, 7]).logits, br.step([8, 9, 10]).logits]
        with sm.batch([a, b]) as bt:
            assert bt.cache is not None and not getattr(bt.cache, "paged", False)
            seen += [bt.step([11, 12]).logits, bt.step([13, 14]).logits]
        return seen

    prefix(sm)
    paged = run()
    with contiguous(sm):
        flat = run()
    same(list(paged), list(flat), f"{place}: a fork and a batch")


@pytest.mark.parametrize("stem", ["tiny_qwen3"])
def test_rows_a_fork_made_off_the_steps_route_never_reach_the_tree(stem: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """a fork's or a batch's rows made through the torch pass where a step makes its rows on the card's kernels (a
    fork wider than the card's rows pass, a batch beside a host layer, one the card ran out of room for) are their
    session's to read, never the tree's: a prompt opening on them decoded other bits than the same prompt cold. The
    route here is the host engine's read as a card engine's would be (`_card_off_route`): its forks' pass is torch's"""
    sm = model(stem, "cpu")
    pc = prefix(sm)
    rng = random.Random(14)
    p = toks(rng, 90)
    for off in (False, True):
        pc.tree.evict()
        monkeypatch.setattr(sm, "_card_off_route", lambda i, off=off: off)
        a, b = sm.session(), sm.session()
        a.feed(p)
        with a.fork(2) as br:
            br.step([5, 6])
            br.step([7, 8])
            br.keep(0)
        assert a.tokens == [*p, 5, 7]
        b.feed(toks(rng, 40))  # another prompt: the tree read, a's commit put in
        assert pc.match(a.tokens).length == (len(p) if off else len(a.tokens)), off
        a.generate(3, eos=(), speculate=False)  # the session reads its rows still
    monkeypatch.undo()


@CARD
@pytest.mark.parametrize("place", ["card"])
@pytest.mark.parametrize("stem", GRAPHED)
def test_a_long_prompt_leaves_no_scratch_and_graphs_share_their_states(stem: str, place: str) -> None:
    """a prompt past a step's rows lets the card's scratch of its width go with it (kept, it held the prompt's width
    for every step after); the card graph's attention states are one set for every graph of a width, asked of the
    ledger, not one for each run, tail and kernel"""
    sm = model(stem, place)
    prefix(sm)
    s = sm.session()
    s.feed(chatty(random.Random(12), 300))
    assert not [k for k in sm.scratch._bufs if k[0].startswith("card")], "a prompt's scratch kept past it"
    s.generate(8, eos=(), speculate=True)
    st = sm._card_state()
    parts = st.get("parts", {})
    for g in st["graphs"].values():
        got = parts.get((int(g["key"][2]), int(g["S"])))
        assert got is not None and g["part_m"] is got["part_m"] and g["cnt"] is got["cnt"], g["key"]


@pytest.mark.parametrize("stem", OTHERS)
def test_an_engine_the_prefix_cache_does_not_serve_says_why(stem: str) -> None:
    """a hybrid, a family with its own layer, a family whose attention is its own module's: the reason, and its
    sessions' caches contiguous ones of their own"""
    sm = model(stem)
    assert PrefixCache.why_not(sm) and sm._prefix_cache() is None
    s = sm.session([5, 6, 7, 8])
    assert not getattr(s.cache, "paged", False)
    assert PassTag.KV_PAGED not in sm.last_pass_report()
