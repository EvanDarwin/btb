# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's MTP drafter (btb/engine/families/qwen4/drafter.py) over tiny_q4's drafting head, on the CPU: its layer's
experts are the expert store's - read as the store's layer L, one past the trunk's, never placed with the drafter -
its dense tensors are exactly what the planner prices, its tree's branches carry the indexer's keys with K and V,
and a speculative decode with it as the proposer (a chain, a fixed tree, its own tree) is the plain greedy decode's,
short of the indexer's budget and past it, plain steps between its drafts too (its cache still every committed row);
and the expert profile (`--profile`) keeps every call's picks, row by
row, as the engine's `expert_trace` has them. One load of the model; every case runs inside it."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest
import torch

from btb.engine import StreamedTextModel, generate
from btb.engine.cache import indexer_keys
from btb.engine.experts import ExpertProfile
from btb.engine.families.qwen4.drafter import Qwen4Drafter
from btb.engine.host import _Experts
from btb.kinds import Proposer
from tests.helpers import fixture, host_model, speculation

# a prompt at the indexer's budget (2 blocks of 4: every row still attends all it sees), and one past it, where
# both the trunk's sparse attention and the drafter's select
PROMPTS = {
    "short": [3, 17, 42, 5, 99, 120, 7, 7],
    "long": ([3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61, 88, 21, 4] * 4)[:70],
}
PROPOSERS = ("mtp_dyn", "mtp_tree", "mtp")


class _Plain:
    """a pricer planning plain steps (one row) between drafting passes, on `pattern`, the real one's plan otherwise"""

    def __init__(self, pricer: Any, pattern: str) -> None:
        self.pricer, self.pattern, self.k = pricer, pattern, 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pricer, name)

    def plan(self, cap: int, passes: int) -> tuple[int, bool]:
        planned = self.pricer.plan(cap, passes)
        self.k += 1
        return (1, False) if self.pattern[(self.k - 1) % len(self.pattern)] == "1" else planned


def _held_rows(mp: pytest.MonkeyPatch, dr: Qwen4Drafter, n: int, on: list[int]) -> None:
    """every feed of the drafter checked to go on from its cache's end and to bring it to every row committed:
    `n` the prompt's tokens, `on` the tokens committed so far. The target holds the prompt and all of them but the
    last, T rows: a draft's root feeds the drafter through row T - 1 (the last token over the target's last row),
    the prompt's prefill and a plain run taken without drafting through row T - 2 (every row the target has)"""

    def wrap(name: str, orig: Callable[..., Any]) -> Callable[..., Any]:
        def fed(toks: Any, h: torch.Tensor, pos0: int, *args: Any, **kw: Any) -> Any:
            # a draft of no width feeds nothing
            if name == "extend" or int(args[0]) > 0:
                held = dr.cache.get_seq_length()
                assert held == pos0, f"{name}: the drafter holds {held} rows, fed at {pos0}"
                assert int(h.shape[1]) == len(toks), (name, int(h.shape[1]), len(toks))
                T = n + max(0, len(on) - 1)
                assert pos0 + len(toks) == (T if name != "extend" else T - 1), (name, pos0, len(toks), len(on))
            return orig(toks, h, pos0, *args, **kw)

        return fed

    for name in ("ar", "at", "au", "extend"):
        mp.setattr(dr, name, wrap(name, getattr(dr, name)))


def test_the_drafter_reads_its_experts_through_the_store_and_its_speculation_is_the_plain_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    StreamedTextModel.register_attention()
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q4"))
        try:
            store = sm.expert_store
            assert store is not None, "tiny_q4 on the host runs its mixture through the expert store"
            dr = sm.mtp_drafter()
            assert isinstance(dr, Qwen4Drafter)
            L = int(sm.L)
            ex = dr.layer.mlp.experts
            assert isinstance(ex, _Experts) and ex.layer == L and ex.base == "mtp.layers.0.mlp.experts."
            # no expert tensor placed: none among the drafter's tensors, none read whole by the layer
            placed = dr.named_tensors()
            assert not [k for k in placed if ".experts." in k] and ex.gate_up is None and ex.down is None
            held = [n for m in (dr.layer, dr.head) for n, _t in m.named_parameters()]
            assert not [n for n in held if ".experts." in n], held
            # the dense tensors are the head's every non-expert tensor, at the bytes the planner prices
            dense = {k for k in sm.weight_map if k.startswith("mtp.") and sm.fam.dense_key(k)}
            assert set(placed) == dense and set(dr.dense_keys) == dense
            assert 2 * dr.dense_numel == sm._drafter_bytes() > 0

            sm.expert_trace = []
            # the profile beside the trace (never saved): it keeps every call's picks, as bench/e2e_bench.py's routing
            # replay reads them back
            prof = sm.expert_profile = ExpertProfile("unsaved.npz")
            for name, prompt in PROMPTS.items():
                plain = sm.generate_greedy(prompt, 16)
                for proposer in PROPOSERS:
                    speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
                    toks, census = sm.generate_speculative(prompt, 16, proposer=proposer, v_max=4)
                    assert toks == plain, f"{name}, {proposer}: {toks} != {plain}"
                    assert census["proposed"] > 0, f"{name}, {proposer}: nothing was drafted, so nothing verified"
                    assert census["mtp_steps"] > 0, f"{name}, {proposer}: the drafter never stepped"
                    # the tree's branches leave the drafter's cache as the root left it: the indexer's keys with K/V
                    cl = dr.cache.layers[0]
                    ik = indexer_keys(cl)
                    assert ik is not None and ik.shape[0] == cl.keys.shape[0] == 1
                    assert ik.shape[1] == cl.keys.shape[-2], f"{name}, {proposer}: indexer keys out of step"
                    # plain steps the pricer plans between drafting passes (its warm steps, a width that does not
                    # pay): the drafter still takes every committed row, at the next draft's root or, past
                    # PEND_ROWS of them, without drafting - its cache the one every pass drafting leaves
                    for pattern, bound in (("1d", 16), ("11d", 16), ("111d", 2)):
                        on: list[int] = []
                        with monkeypatch.context() as mp:
                            _held_rows(mp, dr, len(prompt), on)
                            real = sm._spec_pricer

                            def plain_steps(
                                v_max: int | None = None, real: Callable[..., Any] = real, pattern: str = pattern
                            ) -> _Plain:
                                return _Plain(real(v_max), pattern)

                            mp.setattr(sm, "_spec_pricer", plain_steps)
                            mp.setattr(generate, "PEND_ROWS", bound)
                            toks, census = sm.generate_speculative(
                                prompt, 16, proposer=proposer, v_max=4, on_token=on.append
                            )
                        case = f"{name}, {proposer}, {pattern}"
                        assert toks == plain == on, f"{case}: {toks} != {plain}"
                        assert census["proposed"] > 0, f"{case}: nothing was drafted"
                    # a session going on from such a decode: the drafter resumes from the rows it had taken, the
                    # ones it had yet to take fed first
                    s = sm.session()
                    for turn in (prompt, None):
                        with monkeypatch.context() as mp:
                            real = sm._spec_pricer

                            def plain_run(v_max: int | None = None, real: Callable[..., Any] = real) -> _Plain:
                                return _Plain(real(v_max), "11d")

                            mp.setattr(sm, "_spec_pricer", plain_run)
                            mp.setattr(sm, "proposer", Proposer.of(proposer), raising=False)
                            ids = turn if turn is not None else [*s.tokens, 5]
                            on = []
                            _held_rows(mp, dr, len(ids), on)
                            got = sm.generate(ids, 7, eos=(), speculate=True, session=s, on_token=on.append).tokens
                        assert list(got) == sm.generate_greedy(ids, 7), f"{name}, {proposer}: a session's decode"
            # the drafter's calls were the store's: its layer's picks recorded under L, its recipe read from mtp.*
            assert any(layer == L for layer, _picks in sm.expert_trace), "the drafter's layer never called the store"
            shard = sm.weight_map["mtp.layers.0.mlp.experts.gate_up_proj"]
            assert L in store.recipes and os.path.basename(store.recipes[L][0][0]) == os.path.basename(shard)
            assert any(key[0] == L for key in store.rides), "no expert of the drafter's layer rode the store"
            # the profile's calls are the trace's, one for one: the layer, the rows, and every row's picks
            calls = prof.a[: prof.n][prof.a[: prof.n, 2] == prof.CALL]
            assert len(calls) == len(sm.expert_trace) and any(int(c[4]) > 1 for c in calls)
            for c, (layer, picks) in zip(calls, sm.expert_trace, strict=True):
                rows, off = int(c[4]), int(c[7])
                assert (int(c[3]), rows) == (layer, picks.shape[0])
                assert torch.equal(torch.from_numpy(prof.p[off : off + rows, : picks.shape[1]]).long(), picks.long())
        finally:
            sm.expert_trace = None
            sm.expert_profile = None
            sm.close()
