# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The speculative pass's pricing (`SpecCost`) against a scripted world, no model: a drive-bound mixture of experts
with the 180B's measured figures verifies a short chain - on the card program too, whose warm-up curve timed the
store's reads into its rows - a resident one keeps the wide tree, the width chosen is the fastest the model predicts,
the probes feed the calibration whatever the base curve caps the pass at, and the calibration by depth converges on
the acceptance it is shown."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

import pytest

from btb.engine.spec_cost import SpecCost

MISS_S = 0.0024
READS_16_OVER_1 = 1660.0 / 419.0  # the 180B's 16-row pass against its step
EXP = math.log(READS_16_OVER_1) / math.log(16)  # ~0.497: the rows' experts overlap
TREE = 16  # the MTP tree's budget
# the drafter's tree in its order: (depth, path probability, whether it lies on the main line)
NODES = [
    (1, 0.80, True),
    (2, 0.60, True),
    (3, 0.45, True),
    (1, 0.08, False),
    (4, 0.33, True),
    (2, 0.06, False),
    (1, 0.04, False),
    (5, 0.24, True),
    (3, 0.05, False),
    (2, 0.04, False),
    (6, 0.17, True),
    (1, 0.02, False),
    (4, 0.03, False),
    (3, 0.02, False),
    (7, 0.12, True),
    (2, 0.02, False),
]
# the card program's warm-up on the 180B (35 layers resident, 13 on the host): whole passes, the store's reads for
# the rows' experts in them
CARD_CURVE = {1: 0.50364, 2: 0.50364 * 2.32, 4: 0.50364 * 3.85, 8: 0.50364 * 5.11, 16: 0.50364 * 6.50}


@dataclass
class World:
    """a pass of k rows: its compute `compute * k ** h`, and its wait on the store: a step's `wait` and the
    extra rows' reads, `reads * k ** EXP` of them, each a missed expert's seconds"""

    compute: float = 0.9
    wait: float = 0.4
    reads: float = 419.0
    h: float = 0.1
    miss_s: float = MISS_S
    # P(the accepted path reaches depth d): the 180B's first run (16/18, 11/18, 3/18 of the passes)
    reach: list[float] = field(default_factory=lambda: [16 / 18, 11 / 18, 3 / 18, 0.0])
    cap: int = TREE + 1  # the base curve's budget (`_spec_budget`)
    curve: dict[int, float] | None = None

    def n_reads(self, k: int) -> float:
        return self.reads * k**EXP

    def waited(self, k: int) -> float:
        return self.wait + self.miss_s * (self.n_reads(k) - self.reads)

    def seconds(self, k: int) -> float:
        return self.compute * k**self.h + self.waited(k)


WORLD_180B = World()
# the 180B on the card program: a step 0.916 s, its compute the warm-up's one-row 0.504 s, 321 reads a step, 2.247 ms
# a read; the base curve's budget one row (its 2-row pass 2.32x); the drafter taking 67% at depth 1, 46% at depth 2
WORLD_CARD = World(
    compute=0.504,
    wait=0.9162 - 0.504,
    reads=321.3,
    h=0.15,
    miss_s=0.002247,
    reach=[0.67, 0.46, 0.12, 0.0],
    cap=1,
    curve=CARD_CURVE,
)


def run(passes: int, w: World = WORLD_180B, seed: int = 0, call: int | None = None) -> tuple[SpecCost, list[int]]:
    """`passes` of the loop against the scripted world: plan, draft, prune, verify, record; the rows each carried.
    How deep each verified pass is accepted follows the golden-ratio sequence from `seed`: the world's acceptance
    by depth, evenly spread, never a lucky or unlucky streak. `call`: the passes of one generate call - the loop's
    pass count starts again each call (generate.py's `census["forwards"]`), where one pricer serves them all"""
    pc = SpecCost(SpecCost.gamma_of(512, 10))
    u = (seed * 0.7548776662) % 1.0
    rows_seen = []
    for i in range(1, passes + 1):
        pc.price(TREE + 1, w.curve, w.miss_s)
        rows, probe = pc.plan(w.cap, (i - 1) % call + 1 if call else i)
        if rows <= 1:
            pc.record_pass(1, w.seconds(1), w.n_reads(1), None, wait=w.waited(1))
            rows_seen.append(1)
            continue
        n_draft = min(TREE, rows) if pc.active() and rows < TREE + 1 else TREE
        tree = NODES[:n_draft]
        gains = [pc.gain("mtp", d, p) for d, p, _ in tree]
        pc.record_tree(gains, capped=True)
        keep = pc.prune(gains, probe)
        kept = tree[:keep]
        # the accepted path: the main line as deep as the pass reached, while the tree holds it
        u = (u + 0.6180339887) % 1.0
        reach = sum(1 for r in w.reach if u < r)
        main = [d for d, _p, on in kept if on]
        acc = [d for d in range(1, reach + 1) if d in main]
        acc = acc[: next((j for j, d in enumerate(acc, 1) if d != j), len(acc))]
        pc.record_pass(
            keep + 1,
            w.seconds(keep + 1),
            w.n_reads(keep + 1),
            0.05,
            [("mtp", d, p) for d, p, _ in kept],
            ["mtp"] * len(acc),
            wait=w.waited(keep + 1),
        )
        rows_seen.append(keep + 1)
    return pc, rows_seen


def steady(rows: list[int], skip: int = 20) -> list[int]:
    """the rows past the first `skip` passes, the probes left out"""
    return [r for i, r in enumerate(rows[skip:], skip + 1) if i % SpecCost.PROBE]


def test_drive_bound_moe_verifies_a_short_chain() -> None:
    pc, rows = run(80)
    st = steady(rows)
    # the 180B's passes carry the root and a chain of one or two drafts (a third where its calibration, drafted
    # less often, still leans on the prior), never the tree
    assert st and all(2 <= r <= 4 for r in st), rows
    assert max(set(st), key=st.count) in (2, 3), rows
    assert pc.active()
    # the exponents are measured, not the priors (the reads' independent routing ~0.97, the compute's 0.2)
    assert abs(pc.gamma - EXP) < 0.05, pc.gamma
    assert abs(pc.h - WORLD_180B.h) < 0.05, pc.h
    k = max(set(st), key=st.count)
    plain = 1.0 / pc.cost(1)
    chosen = pc.rate(k, pc.expected(k - 1), drafted=False)
    wide = pc.rate(TREE + 1, pc.expected(TREE), drafted=False)
    assert chosen > plain * 1.2, (chosen, plain)
    assert chosen > wide * 1.3, (chosen, wide)
    rep = pc.report()
    assert rep["active"] and rep["priced_passes"] > 0 and rep["predicted_s"] > 0


def test_card_program_whose_curve_timed_the_reads_still_drafts() -> None:
    # the run that never drafted: the card program's warm-up curve (2 rows 2.32x a step, the store's reads in it)
    # capped `_spec_budget` at one row; priced by its compute and its reads apart, the pass drafts a chain of 1-2
    pc, rows = run(64, WORLD_CARD)
    rep = pc.report()
    assert rep["calibration"], "no draft was ever verified"
    st = steady(rows)
    assert st and max(set(st), key=st.count) in (2, 3), rows
    assert all(r <= 4 for r in st), rows
    # the curve's 2.32x is not the model's: the rows' compute is fitted near the world's, the reads priced apart
    assert pc.ratio(2) < 1.3 and abs(pc.h - WORLD_CARD.h) < 0.05, (pc.ratio(2), pc.h)
    assert pc.cost(1) == pytest.approx(0.9162, rel=0.02)
    k = max(set(st), key=st.count)
    assert pc.rate(k, pc.expected(k - 1), drafted=False) > 1.2 / pc.cost(1)


def test_probe_fires_with_empty_calibration_under_a_cap_of_one() -> None:
    # the 180B run's state: an active pricing with a step measured 61 times over, nothing ever drafted (the
    # calibration empty), and the base curve's budget one row every pass
    pc = SpecCost(SpecCost.gamma_of(512, 10))
    pc.price(TREE + 1, CARD_CURVE, 0.002247)
    for _ in range(61):
        pc.record_pass(1, 0.9162, 321.3, None, wait=0.412)
    assert pc.active() and pc.ready() and not pc.report()["calibration"]
    # the next pass probes the full width, the base curve's cap of one set aside
    rows, probe = pc.plan(1, 62)
    assert (rows, probe) == (TREE + 1, True)
    gains = [pc.gain("mtp", d, p) for d, p, _ in NODES]
    keep = pc.prune(gains, probe)
    assert keep >= 1
    pc.record_pass(keep + 1, 1.3, 321.3 * (keep + 1) ** EXP, 0.05, [("mtp", 1, 0.8)][:keep], ["mtp"], wait=0.6)
    assert pc.report()["calibration"], "the probe fed nothing"
    # and every PROBE-th pass after probes again, whatever the width the model would pick
    probes = [i for i in range(63, 63 + SpecCost.PROBE) if pc.plan(1, i)[1]]
    assert probes == [i for i in range(63, 63 + SpecCost.PROBE) if i % SpecCost.PROBE == 0]


def test_resident_moe_keeps_the_wide_tree() -> None:
    # every expert in VRAM or RAM: a step reads nothing, so the pricing measures its plain steps and stands aside
    pc, rows = run(40, World(reads=0.0, wait=0.0))
    assert rows[:2] == [1, 1]
    assert not pc.active()
    assert pc.plan(TREE + 1, 5) == (TREE + 1, False)
    assert pc.plan(3, 16) == (3, False), "unpriced, the base curve's budget stands"
    assert pc.prune([0.1] * TREE, False) == TREE
    # no drive measured at all (a dense model, a store without a probe): the base sizing from the first pass
    pc2 = SpecCost()
    pc2.price(TREE + 1, {1: 1.0, 8: 1.05, 17: 1.2}, 0.0)
    assert pc2.plan(TREE + 1, 1) == (TREE + 1, False)
    assert pc2.plan(4, 16) == (4, False)
    assert pc2.prune([0.01] * TREE, False) == TREE


def test_first_priced_pass_measures_a_step_then_probes() -> None:
    pc = SpecCost()
    pc.price(TREE + 1, None, MISS_S)
    assert pc.plan(TREE + 1, 1) == (1, False)
    # the first plain step after a prefill reloads: a second one is the step taken
    pc.record_pass(1, 3.9, 838.0, None, wait=3.0)
    assert not pc.ready()
    assert pc.plan(TREE + 1, 2) == (1, False)
    pc.record_pass(1, 1.3, 419.0, None, wait=0.4)
    assert pc.ready() and pc.c1 == pytest.approx(1.3) and pc.u1 == pytest.approx(0.9) and pc.m1 == 419.0
    assert pc.plan(TREE + 1, 3) == (TREE + 1, True)
    # a probe verifies at least one draft, whatever it costs
    assert pc.prune([0.0] * TREE, True) == 1
    assert pc.prune([0.0] * TREE, False) == 0


def test_chosen_width_maximizes_predicted_rate() -> None:
    pc, _rows = run(60)
    k, probe = pc.plan(TREE + 1, 61)
    assert not probe
    rates = {1: 1.0 / pc.cost(1), **{j: pc.rate(j, pc.expected(j - 1), False) for j in range(2, TREE + 2)}}
    assert rates[k] == pytest.approx(max(rates.values()))
    # after drafting: the prefix kept is the fastest of the prefixes
    gains = [pc.gain("mtp", d, p) for d, p, _ in NODES]
    keep = pc.prune(gains, False)
    cum = [0.0]
    for g in gains:
        cum.append(cum[-1] + g)
    by_prefix = [pc.rate(j + 1, cum[j], True) for j in range(len(gains) + 1)]
    assert by_prefix[keep] == pytest.approx(max(by_prefix))


def test_calibration_by_depth_converges() -> None:
    # the drafter says 0.9 at depth 1 and 0.6 at depth 2; the target takes depth 1 every other pass, depth 2 never
    pc = SpecCost()
    for i in range(60):
        acc = ["mtp"] if i % 2 == 0 else []
        pc.record_pass(3, 1.0, 0.0, 0.01, [("mtp", 1, 0.9), ("mtp", 2, 0.6)], acc)
    assert pc.gain("mtp", 1, 0.9) == pytest.approx(0.5, abs=0.06)
    assert pc.gain("mtp", 2, 0.6) < 0.1
    # another source keeps its own calibration
    assert pc.gain("ngram", 1, 0.5) == pytest.approx(0.5)


def test_same_history_same_widths() -> None:
    assert run(50, seed=3)[1] == run(50, seed=3)[1]
    assert run(50, WORLD_CARD, seed=3)[1] == run(50, WORLD_CARD, seed=3)[1]


def test_exponents_fit_the_passes() -> None:
    w = WORLD_180B
    pc = SpecCost(SpecCost.gamma_of(512, 10))
    assert pc.gamma > 0.9  # the independent routing's prior
    pc.price(TREE + 1, CARD_CURVE, MISS_S)
    for _ in range(2):
        pc.record_pass(1, w.seconds(1), w.n_reads(1), None, wait=w.waited(1))
    for k in (16, 4) * 15:
        pc.record_pass(k, w.seconds(k), w.n_reads(k), 0.05, wait=w.waited(k))
    assert abs(pc.gamma - EXP) < 0.02 and abs(pc.h - w.h) < 0.02
    assert pc.c1 == pytest.approx(w.seconds(1), rel=0.01)
    for k in (2, 4, 8, 16):
        assert pc.cost(k) == pytest.approx(w.seconds(k), rel=0.03), k


def test_base_curve_ratio() -> None:
    pc = SpecCost()
    pc.price(9, {1: 2.0, 2: 2.2, 4: 3.0}, 0.0)
    assert pc.ratio(1) == 1.0
    assert pc.ratio(3) == pytest.approx(1.3)
    assert pc.ratio(6) == pytest.approx(1.9)  # past the widest along its last step
    pc.price(9, None, 0.0)
    assert pc.ratio(7) == 1.0
    # a priced store: the curve (which timed the reads with the rows) set aside for the fitted compute
    pc.price(9, {1: 2.0, 2: 4.6}, MISS_S)
    assert pc.ratio(2) == pytest.approx(2**SpecCost.H0)


def test_a_warm_up_curve_a_burst_spoiled_is_measured_back_by_the_passes() -> None:
    """the warm-up timed beside a game, its 2-row samples catching the card's bursts and its 1-row ones not: the
    2-row pass priced at 6 one-row ones (the card program's measured 6, 21 and 53). The passes measure the curve back:
    each width a pass ran at its measured seconds, the others the warm-up's scaled to the step's measured seconds"""
    pc = SpecCost()
    pc.price(9, {1: 1.0, 2: 6.0, 4: 7.0}, 0.0)
    assert pc.curve_now() == {1: 1.0, 2: 6.0, 4: 7.0}, "nothing measured: the warm-up's"
    for _ in range(3):
        pc.record_pass(1, 2.0, 0.0, None)  # the step, measured, twice the warm-up's: the curve scaled with it
    assert pc.curve_now()[4] == pytest.approx(14.0) and pc.ratio(4) == pytest.approx(7.0)
    for _ in range(12):
        pc.record_pass(2, 2.2, 0.0, 0.01)
    assert pc.curve_now()[2] == pytest.approx(2.2, rel=0.01) and pc.ratio(2) == pytest.approx(1.1, rel=0.01)
    # the next call's warm-up curve (the engine prices every call) keeps what the passes measured
    pc.price(9, {1: 1.0, 2: 6.0, 4: 7.0}, 0.0)
    assert pc.ratio(2) == pytest.approx(1.1, rel=0.01)


def test_a_burst_leaves_a_measured_width_as_it_was_and_a_faster_pass_moves_it_at_once() -> None:
    pc = SpecCost()
    pc.price(9, {1: 1.0, 2: 1.2}, 0.0)
    pc.record_pass(1, 1.0, 0.0, None)
    pc.record_pass(1, 200.0, 0.0, None)  # another program's slice of the card: a pass 200 times its width's
    assert pc.live[1] == pytest.approx(1.0), "the burst moved the width's seconds"
    pc.record_pass(1, 0.5, 0.0, None)  # the card freed: the pass faster than the estimate, taken whole
    assert pc.live[1] == pytest.approx(0.5)


def test_a_card_programs_pass_is_measured_at_its_graphs_width() -> None:
    """a card program pads a pass's rows to its graphs' widths: a 3-row pass is measured as the 4-row one it ran"""
    pc = SpecCost()
    pc.price(9, {1: 1.0, 2: 1.5, 3: 2.0, 4: 2.0}, 0.0, lambda k: 4 if k > 2 else k)
    pc.record_pass(3, 1.25, 0.0, 0.01)
    now = pc.curve_now()
    assert now[3] == now[4] == pytest.approx(1.25)
    assert set(pc.live) == {4}


def test_a_base_curve_that_priced_every_pass_plain_is_measured_again_after_a_run_of_steps() -> None:
    """the base curve, as measured, sized every pass a plain step (`_spec_budget` 1): a run of PROBE_FAR plain steps
    across calls verifies one draft - the wider widths measured again, so a curve the warm-up took beside a game does
    not keep speculation off for the load; never where no speculation is configured, nor without a curve"""
    pc = SpecCost()
    pc.price(9, {1: 1.0, 2: 6.0}, 0.0)
    for i in range(SpecCost.PROBE_FAR):
        assert pc.plan(1, i % 7 + 1) == (1, False), i  # the calls' own pass counts start again each call
        pc.record_pass(1, 1.0, 0.0, None)
    assert pc.plan(1, 3) == (2, True)
    pc.record_pass(2, 1.1, 0.0, 0.01)
    assert pc.plan(1, 4) == (1, False), "the draft verified: the run starts again"
    assert pc.ratio(2) < 2.0, "the 2-row pass measured back from the warm-up's 6 steps"
    for pc2, full, curve in ((SpecCost(), 1, {1: 1.0, 2: 6.0}), (SpecCost(), 9, None)):
        pc2.price(full, curve, 0.0)
        for _ in range(SpecCost.PROBE_FAR + 1):
            pc2.record_pass(1, 1.0, 0.0, None)
        assert pc2.plan(1, 5) == (1, False)


def test_the_expected_gain_of_a_trees_first_nodes() -> None:
    """before any tree, the prior's PRIOR_Q + PRIOR_Q**2 + ...; after, the decayed gain of the recent trees' first
    j nodes, and past the widest tree drafted what that one gained; nothing for no node. A priced store's widest
    pass is always wider than the step: its reads cost what a step's do not"""
    pc = SpecCost()
    q = SpecCost.PRIOR_Q
    assert pc.expected(0) == 0.0 and pc.expected(3) == pytest.approx(q + q**2 + q**3)
    pc.record_tree([0.5, 0.25], capped=True)
    assert pc.expected(1) == pytest.approx(0.5) and pc.expected(2) == pytest.approx(0.75)
    assert pc.expected(6) == pytest.approx(0.75), "past the widest tree: what it gained"
    pc.record_tree([0.5], capped=False)  # a tree that stopped short: nothing wider would have gained more
    assert pc.expected(2) == pytest.approx(0.8 * 0.75 + 0.2 * 0.5)
    # a wide pass before any step is measured teaches the fits nothing: they are ratios to the step's figures
    pc.price(TREE + 1, None, MISS_S)
    pc.record_pass(4, 2.0, 900.0, 0.05, wait=1.0)
    assert pc.u1 is None and pc.m1 is None and (pc.h, pc.gamma) == (SpecCost.H0, pc.gamma0)
    pc.price(1, None, MISS_S)
    for _ in range(SpecCost.WARM):
        pc.record_pass(1, 1.0, 400.0, None, wait=0.5)
    assert pc.ready() and not pc.active() and pc.plan(1, 5) == (1, False)


def test_a_run_of_plain_steps_verifies_one_draft() -> None:
    # drafts that could pay were they accepted, accepted one time in ten: the passes are plain steps, but the pass
    # after a run of PROBE_PLAIN asks for one draft (its prune may keep a second, never the tree), so the verdict
    # rests on a fed calibration
    w = World(miss_s=0.002, reach=[0.1, 0.0])
    pc, rows = run(48, w)
    assert pc.could_pay(), "the world was meant to be one where an accepted draft pays"
    st = steady(rows, skip=8)
    assert st.count(1) >= len(st) * 0.8, rows
    assert all(r <= 3 for r in st), rows
    # once the rows' cost is fitted (the first probes' passes; before it the verdict can lean either way)
    runs = "".join("p" if r == 1 else "d" for r in rows[SpecCost.PROBE :]).split("d")
    assert max(len(r) for r in runs) <= SpecCost.PROBE_PLAIN, rows
    assert pc.report()["calibration"]


def test_where_no_draft_could_pay_the_probes_only_measure_the_cost() -> None:
    # a drive so slow no width beats the step even were every draft accepted: the acceptance decides nothing, so
    # the calibration probes stop - the first probe, then one every PROBE_FAR passes measuring the rows' cost again -
    # where they had cost Qwen3.8-flash-next a sixth of its call (a probe every 16 passes and after every 8 plain)
    w = World(miss_s=0.05, reach=[1.0, 1.0, 1.0, 0.0])
    # calls of 40 passes, the pricer across them: a count of the call's passes never reached PROBE_FAR, and a verdict
    # fitted once kept the drafts off for the engine's life
    pc, rows = run(3 * SpecCost.PROBE_FAR, w, call=40)
    assert not pc.could_pay()
    wide = [i for i, r in enumerate(rows, 1) if r > 1]
    first = SpecCost.WARM + 1
    assert wide[0] == first and len(wide) >= 3, f"passes wider than the step: {wide}"
    assert all(b - a == SpecCost.PROBE_FAR + 1 for a, b in pairwise(wide)), f"not every PROBE_FAR plain steps: {wide}"
    assert all(rows[i - 1] == 2 for i in wide), rows
    # a drive that speeds up reopens them: the plain steps' own figures move the verdict
    fast = World(miss_s=0.0005, reach=[1.0, 1.0, 1.0, 0.0])
    for _ in range(40):
        pc.record_pass(1, fast.seconds(1), fast.n_reads(1), None, wait=fast.waited(1))
    pc.price(TREE + 1, None, fast.miss_s)
    assert pc.could_pay()


def test_another_programs_slices_of_the_card_do_not_price_the_widths() -> None:
    """beside a program that has the card, a pass takes whatever slices of it land on that pass - 4.4 ms for a
    Qwen3-0.6B step on a free card, 37 to 218 beside one - whatever its width. The curve takes each width's fastest
    recent pass, so the widths keep their own costs' ratios: a decayed mean priced a 9-row pass at a tenth of a step
    one answer and every width out the next"""
    import random

    curve = {1: 0.0044, 2: 0.0046, 5: 0.0050, 9: 0.0053}
    pc = SpecCost()
    pc.price(9, curve, 0.0)
    rng = random.Random(7)
    for _ in range(200):
        k = rng.choice(list(curve))
        # three passes in four meet a slice of 30 to 210 ms; the rest run on a free card
        extra = rng.uniform(0.030, 0.210) if rng.random() < 0.75 else 0.0
        pc.record_pass(k, curve[k] + extra, 0.0, None)
    now = pc.curve_now()
    for k in curve:
        assert now[k] / now[1] == pytest.approx(curve[k] / curve[1], rel=1e-9), now


def test_a_width_that_costs_more_now_is_priced_at_it_within_its_window() -> None:
    """the fastest of a width's last LIVE_N passes: a pass grown dearer (a longer context's attention) is priced at
    its new cost once its window holds no faster one"""
    pc = SpecCost()
    pc.price(9, {1: 0.004, 4: 0.005}, 0.0)
    for _ in range(pc.LIVE_N):
        pc.record_pass(4, 0.005, 0.0, None)
    for _ in range(pc.LIVE_N):
        pc.record_pass(4, 0.008, 0.0, None)
    assert pc.live[4] == pytest.approx(0.008)


def test_a_pass_on_the_card_is_timed_by_its_own_work_not_what_was_queued_before_it() -> None:
    """the card's clock (`PassClock`): work the card still had in hand when the pass began - the last commit's
    writes, the drafter's - is not the pass's. Timed from the host, a short pass behind a long queue read as the
    queue, and the widths' measured costs swung from 0.17 to 3.3 one-row steps between answers"""
    import torch

    from btb.engine.generate import PassClock
    from tests.helpers import need_cuda

    dev = torch.device(need_cuda())
    x = torch.ones(1 << 20, device=dev)
    clock = PassClock(dev)
    for _ in range(3):  # the first launch's setup out of the way
        clock.start()
        (x * 2).sum()
        clock.stop()
        clock.seconds()
    torch.cuda.synchronize()
    alone = []
    for _ in range(5):
        clock.start()
        y = (x * 2).sum()
        clock.stop()
        y.item()
        alone.append(clock.seconds())
    torch.cuda._sleep(300_000_000)  # the card's queue busy for a while (~0.1 s) before the pass begins
    t0 = time.perf_counter()
    clock.start()
    y = (x * 2).sum()
    clock.stop()
    y.item()
    host = time.perf_counter() - t0
    busy = clock.seconds()
    assert host > 10 * max(alone), "the queue was not busy: the test shows nothing"
    assert busy < 5 * max(alone) + 1e-3, f"{busy * 1e3:.3f} ms behind the queue, {max(alone) * 1e3:.3f} ms alone"


def test_a_pass_off_the_card_is_timed_by_the_wall_clock() -> None:
    import torch

    from btb.engine.generate import PassClock

    clock = PassClock(torch.device("cpu"))
    clock.start()
    time.sleep(0.02)
    clock.stop()
    assert 0.015 <= clock.seconds() < 1.0
