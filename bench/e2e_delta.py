#!/usr/bin/env python3
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Turn two pytest-benchmark JSON runs (main baseline, this PR) into the {id, delta, lo, hi} list the PR-comment
renderer (.github/scripts/bench_compare.py --e2e) folds into its device sections.

pytest-benchmark did the timing and the statistics; this only takes the ratio of the two medians and a 95% band
from the two runs' standard errors of that median - no timing of its own.

    e2e_delta.py BASE_JSON PR_JSON [--out e2e.json]

A benchmark present in only one run is skipped (nothing to compare). Output ids are `<device>/<model>`, taken
from each benchmark's params so they land in the right device section.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import TypedDict


# the slices of pytest-benchmark's JSON this reads (total=False: real files carry more; a run may omit a key)
class _Stats(TypedDict, total=False):
    median: float
    stddev: float
    rounds: int


class _Bench(TypedDict, total=False):
    fullname: str
    name: str
    params: dict[str, str]
    group: str
    stats: _Stats


class _Doc(TypedDict, total=False):
    benchmarks: list[_Bench]
    machine_info: dict[str, object]


class _Entry(TypedDict):
    label: str
    section: str
    stats: _Stats


# The point estimate is a ratio of MEDIANS, so the band has to be the standard error of the MEDIAN. For a roughly
# normal sample that is sqrt(pi/2) times the standard error of the mean - the one distributional assumption here,
# and the reason the mean is not used instead: micro-bench timings are heavy-tailed to the right (a descheduled
# round costs milliseconds), which drags the mean but not the median.
_MEDIAN_SE = math.sqrt(math.pi / 2)


class _Delta(TypedDict):
    id: str
    section: str
    delta: float | None  # None when the two runs are not comparable (different ISA tiers)
    lo: float
    hi: float
    isa: str
    baseline_isa: str


def isa_of(doc: _Doc) -> str:
    """the ISA tier the run's native/cpu paths executed at (bench/conftest.py records it), "" if unrecorded"""
    return str((doc.get("machine_info") or {}).get("isa") or "")


def _entries(doc: _Doc) -> dict[str, _Entry]:
    """each benchmark keyed by its stable fullname, carrying a readable label, its section (the group the bench
    set - `cpu`/`mlx` for e2e, `mlx-quant`/`mlx-attn` for the MLX ops) and its stats. The label reads
    `<device>/<model>` for the e2e paths, else the param values joined."""
    out: dict[str, _Entry] = {}
    for b in doc.get("benchmarks", []):
        key = b.get("fullname") or b.get("name", "")
        params = b.get("params") or {}
        if {"device", "model"} <= params.keys():
            label = f"{params['device']}/{params['model']}"
        elif params:
            label = "-".join(str(v) for v in params.values())
        else:
            label = b.get("name", key)
        out[key] = {"label": label, "section": b.get("group") or "", "stats": b.get("stats", {})}
    return out


def deltas(base: _Doc, pr: _Doc) -> list[_Delta]:
    b, p = _entries(base), _entries(pr)
    base_isa, pr_isa = isa_of(base), isa_of(pr)
    # a shared runner's CPU differs run to run (avx2 vs avx512): two tiers are two kernels, not a delta
    comparable = not (base_isa and pr_isa and base_isa != pr_isa)
    out: list[_Delta] = []
    for key in sorted(b.keys() & p.keys()):
        bs, ps = b[key]["stats"], p[key]["stats"]
        bm, pm = float(bs.get("median", 0.0)), float(ps.get("median", 0.0))
        if bm <= 0:
            continue
        delta = (pm - bm) / bm
        # the band is a standard error of the statistic (stddev / sqrt(rounds), scaled to the median), not the raw
        # per-sample stddev: the raw spread does not shrink with n - it is a per-sample prediction band, not a CI -
        # so using it made a 15k-round bench read ±100% when its centre was pinned to ±0.3%. Quadrature over the
        # two runs, since base and PR are independent samples.
        bn, pn = max(1, int(bs.get("rounds", 1))), max(1, int(ps.get("rounds", 1)))
        bse = (float(bs.get("stddev", 0.0)) / math.sqrt(bn)) / bm if bm else 0.0
        pse = (float(ps.get("stddev", 0.0)) / math.sqrt(pn)) / pm if pm else 0.0
        half = 1.96 * _MEDIAN_SE * math.sqrt(bse * bse + pse * pse)
        sec = p[key]["section"]
        label = p[key]["label"]
        out.append(
            {
                "id": f"{sec}/{label}" if sec and "/" not in label else label,
                "section": sec,
                "delta": delta if comparable else None,
                "lo": delta - half if comparable else 0.0,
                "hi": delta + half if comparable else 0.0,
                "isa": pr_isa,
                "baseline_isa": base_isa,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_json")
    ap.add_argument("pr_json")
    ap.add_argument("--out", default="e2e.json")
    a = ap.parse_args(argv)
    base: _Doc
    pr: _Doc
    try:
        base = json.load(open(a.base_json, encoding="utf-8"))
        pr = json.load(open(a.pr_json, encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"e2e_delta: cannot read inputs ({e}); writing empty", file=sys.stderr)
        base = pr = {"benchmarks": []}
    result = deltas(base, pr)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"e2e_delta: {len(result)} paths compared -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
