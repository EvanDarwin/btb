# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Two pytest-benchmark JSON runs (the base branch, this PR) as the {id, delta, lo, hi} list bench/report.py folds
into its device sections: the ratio of the two medians and a 95% band from the two runs' standard errors of that
median. Each entry carries the PR run's ISA tier.

A benchmark the baseline lacks lists with delta None (new); one only the baseline has is dropped. Output ids are
`<device>/<model>`, taken from each benchmark's params so they land in the right device section.
"""

from __future__ import annotations

import math
from typing import TypedDict


# the slices of pytest-benchmark's JSON this reads (total=False: real files carry more; a run may omit a key)
class Stats(TypedDict, total=False):
    median: float
    stddev: float
    rounds: int


class Bench(TypedDict, total=False):
    fullname: str
    name: str
    params: dict[str, str]
    group: str
    stats: Stats


class Doc(TypedDict, total=False):
    benchmarks: list[Bench]
    machine_info: dict[str, object]


class _Entry(TypedDict):
    label: str
    section: str
    stats: Stats


# The point estimate is a ratio of MEDIANS, so the band has to be the standard error of the MEDIAN. For a roughly
# normal sample that is sqrt(pi/2) times the standard error of the mean - the one distributional assumption here,
# and the reason the mean is not used instead: micro-bench timings are heavy-tailed to the right (a descheduled
# round costs milliseconds), which drags the mean but not the median.
_MEDIAN_SE = math.sqrt(math.pi / 2)


class _Delta(TypedDict):
    id: str
    section: str
    delta: float | None  # None when the baseline has no such benchmark
    lo: float
    hi: float
    isa: str


def isa_of(doc: Doc) -> str:
    """the ISA tier the run's native/cpu paths executed at (bench/conftest.py records it), "" if unrecorded"""
    return str((doc.get("machine_info") or {}).get("isa") or "")


def _entries(doc: Doc) -> dict[str, _Entry]:
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


def _id(e: _Entry) -> str:
    """the output id: the label, under its section unless the label already names one (`<device>/<model>`)"""
    sec, label = e["section"], e["label"]
    return f"{sec}/{label}" if sec and "/" not in label else label


class Timing(TypedDict):
    id: str
    section: str
    median_s: float
    lo_s: float  # the median's 95% band, from the same standard error `deltas` uses
    hi_s: float


def timings(doc: Doc) -> list[Timing]:
    """each benchmark's own median in seconds with its 95% band, under the ids `deltas` gives it"""
    out: list[Timing] = []
    for e in _entries(doc).values():
        st = e["stats"]
        median = float(st.get("median", 0.0))
        half = 1.96 * _MEDIAN_SE * float(st.get("stddev", 0.0)) / math.sqrt(max(1, int(st.get("rounds", 1))))
        out.append(
            {"id": _id(e), "section": e["section"], "median_s": median, "lo_s": median - half, "hi_s": median + half}
        )
    return out


def deltas(base: Doc, pr: Doc) -> list[_Delta]:
    b, p = _entries(base), _entries(pr)
    pr_isa = isa_of(pr)
    out: list[_Delta] = []
    for key in sorted(p.keys()):
        sec = p[key]["section"]
        entry_id = _id(p[key])
        bs: Stats = b[key]["stats"] if key in b else {}
        ps = p[key]["stats"]
        bm, pm = float(bs.get("median", 0.0)), float(ps.get("median", 0.0))
        if key not in b or bm <= 0:
            out.append(
                {
                    "id": entry_id,
                    "section": sec,
                    "delta": None,
                    "lo": 0.0,
                    "hi": 0.0,
                    "isa": pr_isa,
                }
            )
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
        out.append(
            {
                "id": entry_id,
                "section": sec,
                "delta": delta,
                "lo": delta - half,
                "hi": delta + half,
                "isa": pr_isa,
            }
        )
    return out
