# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The speculative pass's pricing (`SpecCost`) against a scripted world, no model: a drive-bound mixture of experts
with the 180B's measured figures verifies a short chain - on the card program too, whose warm-up curve timed the
store's reads into its rows - a resident one keeps the wide tree, the width chosen is the fastest the model predicts,
the probes feed the calibration whatever the base curve caps the pass at, and the calibration by depth converges on
the acceptance it is shown."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def run(passes: int, w: World = WORLD_180B, seed: int = 0) -> tuple[SpecCost, list[int]]:
    """`passes` of the loop against the scripted world: plan, draft, prune, verify, record; the rows each carried.
    How deep each verified pass is accepted follows the golden-ratio sequence from `seed`: the world's acceptance
    by depth, evenly spread, never a lucky or unlucky streak"""
    pc = SpecCost(SpecCost.gamma_of(512, 10))
    u = (seed * 0.7548776662) % 1.0
    rows_seen = []
    for i in range(1, passes + 1):
        pc.price(TREE + 1, w.curve, w.miss_s)
        rows, probe = pc.plan(w.cap, i)
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
    # a drive so slow no draft pays: the passes are plain steps, but every PROBE_PLAIN-th of a run verifies one
    # draft (a row wider, not the whole tree), so the verdict rests on a fed calibration
    w = World(miss_s=0.05, reach=[0.1, 0.0])
    pc, rows = run(48, w)
    st = steady(rows, skip=4)
    assert st.count(1) >= len(st) * 0.8, rows
    assert all(r <= 2 for r in st), rows
    runs = "".join("p" if r == 1 else "d" for r in rows[2:]).split("d")
    assert max(len(r) for r in runs) <= SpecCost.PROBE_PLAIN, rows
    assert pc.report()["calibration"]
