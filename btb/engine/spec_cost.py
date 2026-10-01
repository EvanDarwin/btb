# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The speculative verify pass priced by what its rows cost, so a pass is never wider than its drafts pay for.

A pass of k rows (the root and k - 1 drafts) returns the root's token and every draft on the accepted path, and costs
its rows' compute and, for a mixture of experts whose experts stream from the store, the expert reads its extra rows
add over a one-row step. The price sets the pass's width: the rows whose expected tokens a second are the most, and a
plain step where no width beats the step itself.

Gain. A draft node's gain is the chance its path is the accepted path: the drafter's path probability, calibrated by
what that source's nodes at that depth have been accepted at - a decayed ratio of the nodes accepted to the
probability drafted there. A chain's nodes carry no probability of their own (1), so their calibration is the chain's
running acceptance by depth. A pass's expected tokens are 1 + its nodes' gains.

Cost. A pass of k rows costs

    cost(k) = u1 * r(k) + w1 + s * m1 * (k ** g - 1)

with u1 a plain step's compute - its seconds less the time it waited on the store's reads - and w1 that wait; m1 a
plain step's expert reads (the store's misses and the lookahead's reads not withdrawn); g the exponent the reads grow
by with the rows, fitted online from the wide passes' reads - a pass's rows route to overlapping experts, and how much
they overlap is measured, the independent-routing figure only its prior; s a missed expert's seconds (the drive's
probe). r(k) is the rows' compute over one row's. With a priced store it is k ** h, h fitted online from the passes'
compute (their seconds less their waits) from a prior of rows near free: the warm-up's cost curve times whole passes,
the store's reads for the rows' experts and the host layers' rows among them, so it would count the reads twice.
Without one it is the curve's (1 without a curve: the rows free). The plain steps measured are the first WARM a priced
store decodes (the first after a prefill a reload, only the last taken) and any a pass chooses after. A pass that
drafted also pays the drafter's measured seconds, d.

Width. A pass carries the k maximizing (1 + G(k - 1)) / (d + cost(k)) against the plain step's 1 / cost(1), with G
the expected gain of a tree's first nodes in the drafter's order: before drafting, the decayed gains of the recent
passes' trees; after, the drafted tree's own (a prefix of that order keeps every node's parent). Every PROBE-th pass
(and the first priced one) drafts the whole tree and verifies one draft at least, whatever the model says, and the
pass after a run of PROBE_PLAIN plain steps verifies one draft: the calibration is fed while a draft could pay at all.
Where no width would beat the step even were every draft accepted (`could_pay`: the rows cost near a step each, the
drafter a large share of one), the acceptance decides nothing and only a run of PROBE_FAR plain steps - the pricer's,
across calls - verifies one draft, the rows' cost and the drafter's measured again: a store that fills, a drive that
speeds up, a card no longer contended, reopen the drafts. The pricing sizes a
pass only where the rows' reads matter - the widest pass's modelled reads a twentieth of a step or more - and then
over the engine's full width, the base curve's budget set aside (its curve priced the reads into the rows); a model
whose experts sit in VRAM or RAM, and a dense model, keep the width the base curve gives them.

Every figure is a Python float updated by a few operations a pass: nothing synchronizes, and the same history gives
the same widths.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _mix(old: float | None, new: float, keep: float) -> float:
    """a decayed mean: `new` taken whole when there is no `old`"""
    return new if old is None else keep * old + (1.0 - keep) * new


def _rows(k: int) -> int:
    return k


class SpecCost:
    CAL_DECAY = 0.95  # a verified pass's weight in the calibration by depth: about the last twenty
    CAL_PRIOR = 1.0  # the calibration's prior weight, at 1: the drafter's probability taken at its word
    GAIN_DECAY = 0.8  # in the expected gain of a tree's first nodes
    TIME_DECAY = 0.7  # in the step's seconds, the drafter's and a one-row pass's reads
    FIT_DECAY = 0.9  # in the fit of the reads' exponent
    FIT_PRIOR = 0.5  # the exponents' prior weight, in the fit's units (ln k squared: a 2-row pass weighs 0.48)
    H0 = 0.2  # the rows' compute exponent before a wide pass is measured: rows near free
    PRIOR_Q = 0.5  # before any tree is drafted, a tree's j-th node gains PRIOR_Q ** j
    PROBE = 16  # every PROBE-th pass drafts to the full width and verifies at least one draft
    # a run of plain steps this long verifies one draft on the next pass: the calibration a plain verdict rests on
    # is kept fed, at most one pass in PROBE_PLAIN a row wider than the step
    PROBE_PLAIN = 8
    # where no width would beat the step were every draft it carries accepted (`could_pay`), the acceptance has
    # nothing to decide: a run of plain steps this long, across calls, verifies one draft - the rows' cost and the
    # drafter's measured again (Qwen3.8-flash-next on the host: a 2-row pass 2.2 steps and the drafter 0.58 s, its
    # probes a sixth of the call)
    PROBE_FAR = 64
    DRIVE_SHARE = 0.05  # the widest pass's modelled reads under this share of a step: the base sizing stands
    DEPTHS = 8  # depths reported
    # plain steps measured before a priced store's passes are sized: the first after a prefill reloads what the
    # prompt's sweep gave up, a reload and not a step, so the last of them is the one taken
    WARM = 2

    def __init__(self, gamma0: float = 0.9) -> None:
        self.gamma0 = min(1.5, max(0.0, float(gamma0)))
        self.gamma = self.gamma0
        self.m1: float | None = None  # a one-row pass's expert reads
        self.steps_seen = 0  # the one-row passes measured
        self.u1: float | None = None  # a one-row pass's compute: its seconds less its wait on the store
        self.w1 = 0.0  # a one-row pass's wait on the store's reads
        self.h = self.H0
        self.draft_s: float | None = None  # the drafter's seconds a pass that drafted
        self.miss_s = 0.0
        self.curve: dict[int, float] = {}
        self.width: Callable[[int], int] = _rows  # the curve's width a pass of k rows runs at
        self.live: dict[int, float] = {}  # each width's seconds as the passes measure them, decayed
        self.full = 1
        self.cal_a: dict[tuple[str, int], float] = {}
        self.cal_p: dict[tuple[str, int], float] = {}
        self.gains: dict[int, float] = {}  # j nodes -> the decayed gain of a tree's first j nodes
        # the fit's decayed sums, the prior among them as a pass that fades like any other
        self._sxx = self.FIT_PRIOR
        self._sxy = self.FIT_PRIOR * self.gamma0
        self._hxx = self.FIT_PRIOR
        self._hxy = self.FIT_PRIOR * self.H0
        # the first pass priced once a step is measured probes: the tree drafted whole, one draft verified at least
        self.probe_next = True
        self.plain_run = 0  # the plain steps since a pass last verified a draft
        self.begin()

    @staticmethod
    def gamma_of(n_experts: int, top_k: int, k_ref: int = 8) -> float:
        """the reads' exponent were a pass's rows routed independently: the distinct experts `k_ref` rows ask of
        `n_experts` at `top_k` a row, against one row's, as a power of the rows; 0.9 when the routing is unknown"""
        n, k = int(n_experts), int(top_k)
        if n <= 0 or k <= 0 or k >= n:
            return 0.9
        distinct = n * (1.0 - (1.0 - k / n) ** k_ref)
        return math.log(distinct / k) / math.log(k_ref)

    def begin(self) -> None:
        """a call's report started afresh; what was learned stays"""
        self.rows: dict[int, int] = {}
        self.pred_s = 0.0
        self.meas_s = 0.0
        self.priced = 0

    def price(
        self,
        full: int,
        curve: Mapping[int, float] | None,
        miss_s: float,
        width: Callable[[int], int] | None = None,
    ) -> None:
        """the pass's inputs: the widest pass (root included), the base cost curve {rows: seconds} the load's
        warm-up timed, a missed expert's seconds (0: no store, or its drive unmeasured), and the width a pass of k
        rows runs at (a card program pads its rows to its graphs' widths; the rows themselves by default)"""
        self.full = max(1, int(full))
        self.curve = {int(t): float(c) for t, c in curve.items()} if curve and curve.get(1, 0.0) > 0 else {}
        self.miss_s = max(0.0, float(miss_s))
        self.width = width or _rows

    def curve_now(self) -> dict[int, float]:
        """the base curve as the passes measure it now: each width a pass ran at its measured seconds, the others
        the warm-up's scaled by the step's measured seconds over the warm-up's. The warm-up times the load's moment:
        beside another program its samples caught the card's bursts unevenly (a 2-row pass priced at 6, 21 and 53
        one-row ones), and taken as it was the curve kept speculation priced out for the load; {} without one"""
        cv = self.curve
        if not cv:
            return {}
        step = self.live.get(self.width(1))
        scale = step / cv[1] if step else 1.0
        return {t: self.live.get(self.width(t), c * scale) for t, c in cv.items()}

    # -- the model ---------------------------------------------------------------------------------------------

    @property
    def c1(self) -> float | None:
        """a one-row pass's seconds: its compute and its wait"""
        return None if self.u1 is None else self.u1 + self.w1

    def ratio(self, k: int) -> float:
        """the rows' compute over one row's: with a priced store k ** h, fitted; else the base curve's k-row pass
        over its one-row pass as the passes measure it (`curve_now`: timed widths read, others interpolated, past the
        widest extended along its last step); 1 without a curve"""
        if self.miss_s > 0:
            return float(k) ** self.h
        cv = self.curve_now()
        if not cv:
            return 1.0
        c1 = cv[1]
        if k in cv:
            return cv[k] / c1
        ks = sorted(cv)
        lo = max((t for t in ks if t < k), default=ks[0])
        hi = min((t for t in ks if t > k), default=None)
        if hi is not None:
            return (cv[lo] + (cv[hi] - cv[lo]) * (k - lo) / (hi - lo)) / c1
        prev = max((t for t in ks if t < lo), default=None)
        slope = max(0.0, (cv[lo] - cv[prev]) / (lo - prev)) if prev is not None else 0.0
        return (cv[lo] + slope * (k - lo)) / c1

    def extra(self, k: int) -> float:
        """the seconds of the expert reads a k-row pass adds over a one-row pass"""
        if self.m1 is None or self.miss_s <= 0 or k <= 1:
            return 0.0
        return self.miss_s * self.m1 * (k**self.gamma - 1.0)

    def cost(self, k: int) -> float:
        """a k-row pass's seconds, the drafter's excluded"""
        return (self.u1 or 0.0) * self.ratio(k) + self.w1 + self.extra(k)

    def ready(self) -> bool:
        """whether the step is measured: WARM one-row passes"""
        return self.steps_seen >= self.WARM and self.u1 is not None and self.m1 is not None

    def active(self) -> bool:
        """whether the rows' reads size the pass: a store with a measured drive, and (once a step is measured)
        the widest pass's reads a DRIVE_SHARE of a step or more"""
        if self.miss_s <= 0:
            return False
        if not self.ready():
            return True
        return self.extra(self.full) >= self.DRIVE_SHARE * (self.c1 or 0.0)

    def cal(self, tag: str, depth: int) -> float:
        """`tag`'s calibration at `depth`: the nodes accepted there over the probability drafted there, decayed"""
        key = (tag, int(depth))
        return (self.cal_a.get(key, 0.0) + self.CAL_PRIOR) / (self.cal_p.get(key, 0.0) + self.CAL_PRIOR)

    def gain(self, tag: str, depth: int, p: float) -> float:
        """a node's expected gain: its path probability `p` under its source's calibration at its depth"""
        return min(1.0, max(0.0, self.cal(tag, depth) * float(p)))

    def expected(self, j: int) -> float:
        """the expected gain of a tree's first j nodes, from the recent trees: past the widest drafted, what that
        one gained; before any, PRIOR_Q ** 1 + ... + PRIOR_Q ** j"""
        if j <= 0:
            return 0.0
        if j in self.gains:
            return self.gains[j]
        below = [t for t in self.gains if t < j]
        if below:
            return self.gains[max(below)]
        return sum(self.PRIOR_Q**i for i in range(1, j + 1))

    def rate(self, k: int, g: float, drafted: bool) -> float:
        """expected tokens a second of a k-row pass expected to gain `g`"""
        d = (self.draft_s or 0.0) if (drafted or k > 1) else 0.0
        return (1.0 + g) / max(1e-9, d + self.cost(k))

    def could_pay(self) -> bool:
        """whether some width would beat the plain step were every draft it carries accepted: only then has the
        drafts' acceptance a verdict to give, and the calibration probes something to feed"""
        plain = self.rate(1, 0.0, False)
        return any(self.rate(k, k - 1.0, True) > plain for k in range(2, self.full + 1))

    def best(self, gains: Sequence[float], drafted: bool, at_least: int = 1) -> int:
        """the rows (root included) of the fastest pass, `gains[j]` the gain of its first j nodes; the narrowest
        of equals"""
        top = len(gains)
        k0 = max(1, min(at_least, top))
        best_k, best_r = k0, self.rate(k0, gains[k0 - 1], drafted)
        for k in range(k0 + 1, top + 1):
            r = self.rate(k, gains[k - 1], drafted)
            if r > best_r * (1.0 + 1e-9):
                best_k, best_r = k, r
        return best_k

    # -- the pass ----------------------------------------------------------------------------------------------

    def plan(self, cap: int, passes: int) -> tuple[int, bool]:
        """(the rows to draft for, root included; whether this pass probes). Unpriced, the base curve's `cap`.
        Priced, the engine's full width bounds it - `cap` set aside, its curve having priced the store's reads into
        the rows - the first WARM passes plain steps measuring the step, a probe the full width"""
        cap = max(1, int(cap))
        if not self.active():
            if cap == 1 and self.full > 1 and self.curve and self.plain_run >= self.PROBE_FAR:
                # the base curve, as measured, sized every pass a plain step: a run of PROBE_FAR of them (the
                # pricer's, across calls) verifies one draft, so the wider widths are measured again - a curve that
                # priced the drafts out beside a game reopens them once the game lets the card go
                return 2, True
            return cap, False
        if not self.ready():
            return 1, False
        # active and measured: the widest pass's reads cost something, so it is wider than the step
        cap = self.full
        # the probes feed the calibration while a draft could pay at all, and measure the cost now and then when none
        # could (the first probe always: it measures the rows and the drafter the verdict rests on). The run of plain
        # steps is the pricer's own, across calls: `passes` starts again each call, and a count of it never reached
        # PROBE_FAR in calls of fewer passes - a verdict fitted once under contention kept the drafts off for good
        pays = self.could_pay()
        if self.probe_next or (passes % self.PROBE == 0 and pays):
            self.probe_next = False
            return cap, True
        k = self.best([self.expected(j) for j in range(cap)], drafted=False)
        if k == 1 and self.plain_run >= (self.PROBE_PLAIN if pays else self.PROBE_FAR):
            return 2, True
        return k, False

    def prune(self, gains: Sequence[float], probe: bool) -> int:
        """how many of a drafted tree's nodes, in the drafter's order with these gains, the pass verifies"""
        n = len(gains)
        if n == 0 or not self.active() or not self.ready():
            return n
        cum = [0.0]
        for g in gains:
            cum.append(cum[-1] + g)
        return self.best(cum, drafted=True, at_least=2 if probe else 1) - 1

    def record_tree(self, gains: Sequence[float], capped: bool) -> None:
        """a drafted tree's node gains in the drafter's order, into the expected gain of a tree's first nodes;
        a tree that stopped short of its budget (`capped` off) says nothing wider would have gained more"""
        g = 0.0
        for j, x in enumerate(gains, 1):
            g += x
            self.gains[j] = _mix(self.gains.get(j), g, self.GAIN_DECAY)
        if not capped:
            n = len(gains)
            for j in [t for t in self.gains if t > n]:
                self.gains[j] = _mix(self.gains[j], g, self.GAIN_DECAY)

    def record_pass(
        self,
        k: int,
        seconds: float,
        reads: float,
        draft_s: float | None,
        nodes: Sequence[tuple[str, int, float]] = (),
        accepted: Sequence[str] = (),
        wait: float = 0.0,
    ) -> None:
        """a pass of `k` rows done in `seconds` (the verify, the drafter's `draft_s` apart, None when it did not
        draft), putting `reads` expert reads to the drive and waiting `wait` of its seconds on the store's reads;
        `nodes` its drafts' (source, depth, path probability), `accepted` the sources of its accepted drafts by
        depth"""
        k = max(1, int(k))
        if self.ready():
            self.pred_s += self.cost(k)
            self.meas_s += float(seconds)
            self.priced += 1
        self.rows[k] = self.rows.get(k, 0) + 1
        if nodes:
            for key in self.cal_a:
                self.cal_a[key] *= self.CAL_DECAY
            for key in self.cal_p:
                self.cal_p[key] *= self.CAL_DECAY
            for tag, depth, p in nodes:
                key = (tag, int(depth))
                self.cal_p[key] = self.cal_p.get(key, 0.0) + float(p)
            for depth, tag in enumerate(accepted, 1):
                key = (tag, depth)
                self.cal_a[key] = self.cal_a.get(key, 0.0) + 1.0
        reads = max(0.0, float(reads))
        wait = min(max(0.0, float(wait)), float(seconds))
        compute = max(1e-6, float(seconds) - wait)
        warm = k == 1 and self.steps_seen < self.WARM
        self.plain_run = self.plain_run + 1 if k == 1 else 0
        if k == 1:
            self.steps_seen += 1
            self.m1 = reads if warm else _mix(self.m1, reads, self.TIME_DECAY)
            self.u1 = compute if warm else _mix(self.u1, compute, self.TIME_DECAY)
            self.w1 = wait if warm else _mix(self.w1, wait, self.TIME_DECAY)
        elif self.m1 is not None and self.u1 is not None and self.miss_s > 0:
            x = math.log(k)
            # the rows' compute over a step's against ln k, through the origin, from rows near free
            self._hxx = self.FIT_DECAY * self._hxx + x * x
            self._hxy = self.FIT_DECAY * self._hxy + x * math.log(compute / self.u1)
            self.h = min(1.5, max(0.0, self._hxy / self._hxx))
            # ln of the reads' growth over a step's likewise, from the independent routing's exponent; the one
            # keeps a pass without reads defined
            y = math.log((reads + 1.0) / (self.m1 + 1.0))
            self._sxx = self.FIT_DECAY * self._sxx + x * x
            self._sxy = self.FIT_DECAY * self._sxy + x * y
            self.gamma = min(1.5, max(0.0, self._sxy / self._sxx))
        elif self.miss_s <= 0:
            # no store to price: the step anchored on every pass, its width's curve taken out
            self.u1 = _mix(self.u1, compute / max(1e-9, self.ratio(k)), self.TIME_DECAY)
        if draft_s is not None:
            self.draft_s = _mix(self.draft_s, max(0.0, float(draft_s)), self.TIME_DECAY)
        # the width's measured seconds (`curve_now`), decayed: a burst - another program's slice of the card - is taken
        # as twice what they were at most, a faster pass (the card freed again) as it is
        w = self.width(k)
        old = self.live.get(w)
        self.live[w] = compute if old is None else _mix(old, min(compute, 2.0 * old), self.TIME_DECAY)

    def calibration(self, tag: str) -> list[float]:
        """`tag`'s calibration by depth, 1 .. DEPTHS"""
        return [round(self.cal(tag, d), 3) for d in range(1, self.DEPTHS + 1)]

    def report(self) -> dict[str, Any]:
        """the call's pricing: whether it sized the passes, the rows its passes carried, the model's seconds
        against the measured, and the figures behind them"""
        tags = sorted({t for t, _d in self.cal_p})
        return {
            "active": self.active(),
            "rows": dict(sorted(self.rows.items())),
            "predicted_s": round(self.pred_s, 3),
            "measured_s": round(self.meas_s, 3),
            "priced_passes": self.priced,
            "step_s": round(self.c1 or 0.0, 4),
            "step_wait_s": round(self.w1, 4),
            "compute_exponent": round(self.h, 3),
            "draft_s": round(self.draft_s or 0.0, 4),
            "miss_ms": round(self.miss_s * 1e3, 3),
            "reads_step": round(self.m1 or 0.0, 1),
            "reads_exponent": round(self.gamma, 3),
            "calibration": {t: self.calibration(t) for t in tags},
        }

    def line(self) -> str:
        """the report as the loop's log says it; '' when the pricing sized nothing"""
        if not self.rows or not (self.active() or self.m1):
            return ""
        rows = " ".join(f"{k}:{n}" for k, n in sorted(self.rows.items()))
        return (
            f"; priced by the rows' expert reads: passes by rows {rows}, the model {self.pred_s:.2f} s against "
            f"{self.meas_s:.2f} s measured over {self.priced} passes (a step's compute {self.u1 or 0.0:.3f} s x "
            f"k^{self.h:.2f} + its wait {self.w1:.3f} s + {self.miss_s * 1e3:.2f} ms a read x {self.m1 or 0.0:.0f} x "
            f"(k^{self.gamma:.2f} - 1); the drafter "
            f"{self.draft_s or 0.0:.3f} s)"
        )
