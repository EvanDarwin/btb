#!/usr/bin/env python3
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Render a PR comment from two raw benchmark result sets, grouped into collapsible sections.

A result root holds what its bench runs wrote at its repo paths: native/target/criterion/**/<run>/estimates.json
(the criterion benches, `cargo bench -- --save-baseline <run>`) and e2e-<run>.json (bench/e2e_bench.py), for the
interleaved runs `a` and `b` (base a, PR a, PR b, base b); a root without them is one run, criterion's `new` and
e2e.json. Each run is compared with the baseline's run of the same name - two pairs adjacent in time, whose mean
cancels a drift across the job - and without a baseline every row lists as new.

The comment flags a change only when its whole 95% CI clears the noise floor in every pair. `--gate` adds the
perf-gate exit code (1 when a benchmark's whole CI clears the regression threshold in every pair). Where the PR's
run holds e2e_bench's placement group (a CUDA box), the comment ends with its crossover: each op's lanes by rows
from the run's own times, and where the winner changes - a lane winning a row by the same rule (its lead's whole CI
clear of the noise floor in every run), a tie otherwise.

    python -m bench.report PR_ROOT [--baseline BASE_ROOT] [--noise FRAC] [--out comment.md] [--gate [--gate-threshold FRAC]]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import NotRequired, TypedDict, cast

from bench.e2e_delta import Doc, Timing, band, rel_se
from bench.e2e_delta import deltas as e2e_deltas
from bench.e2e_delta import timings as e2e_timings
from btb.kinds import Json
from tests.cert.native_ops import op_families
from tests.cert.spec import Hardware

MARKER = "<!-- btb-bench-compare -->"
CRITERION = os.path.join("native", "target", "criterion")
E2E = "e2e.json"
RUNS = ("a", "b")  # the interleaved runs of each side, in the order bench.yml's pairs name them
OP_FAMILIES = tuple(op_families())
DEVICES = tuple(h.value for h in Hardware)
# a benchmark id or group is named by the PR's own test code and lands in this comment inside a code span and an
# HTML <summary>: one line and a plain charset, so a crafted name cannot close the span, add table cells, inject
# markup, @-mention a user, reference an issue (#N), or autolink a URL from the bot's comment
_UNSAFE = re.compile(r"[^A-Za-z0-9_./ :+-]")
_LINK = re.compile(r"://|www\.", re.I)  # the two autolink triggers; ':' itself is harmless


def _safe(s: str) -> str:
    return _UNSAFE.sub("?", _LINK.sub("?", s.replace("\n", " ").replace("\r", " ")))[:200]


class Entry(TypedDict):
    """one benchmark's change against the baseline, whichever source it came from. `delta` is None for a bench
    the baseline has never seen (nothing to compare), and `section` is set only by the e2e side - the criterion
    ids carry theirs in the id, which _section() reads."""

    id: str
    section: NotRequired[str]
    delta: float | None
    lo: float
    hi: float
    isa: NotRequired[str]  # the ISA tier the e2e side recorded for this run


def _read_json(path: str) -> object | None:
    """the parsed document, or None when the file is absent or not JSON. The shape is whatever was on disk, so
    callers narrow it themselves rather than trusting an annotation over a file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _num(doc: object, *path: str) -> float:
    """a number at a nested key path, 0.0 when any step is missing or not a number."""
    cur = doc
    for key in path:
        if not isinstance(cur, dict):
            return 0.0
        cur = cur.get(key)
    return float(cur) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else 0.0


def _criterion_runs(root: str, run: str | None = None) -> dict[str, object]:
    """{bench id -> its <run>/estimates.json (`new` for a root's one run)} under a result root's criterion tree;
    the id is the bench's path (`gemv_bf16_group/neon/4`)"""
    out: dict[str, object] = {}
    tree = os.path.join(root, CRITERION)
    if not os.path.isdir(tree):
        return out
    for dirpath, _dirs, files in os.walk(tree):
        if os.path.basename(dirpath) != (run or "new") or "estimates.json" not in files:
            continue
        rel = os.path.relpath(os.path.dirname(dirpath), tree).replace(os.sep, "/")
        if rel.startswith("report"):
            continue
        out[rel] = _read_json(os.path.join(dirpath, "estimates.json"))
    return out


def _e2e_path(root: str, run: str | None) -> str:
    return os.path.join(root, f"e2e-{run}.json" if run else E2E)


def runs(root: str) -> tuple[str | None, ...]:
    """the interleaved runs a result root holds, or (None,) for a root of one run"""
    held = tuple(r for r in RUNS if os.path.exists(_e2e_path(root, r)) or _criterion_runs(root, r))
    return held or (None,)


def _from_criterion(pr_root: str, base_root: str | None, run: str | None = None) -> list[Entry]:
    """the ratio of the two roots' `run` medians per bench, with the two 95% CIs (criterion's bootstrap) combined
    in quadrature as relative half-widths; a bench the baseline lacks lists as new."""
    base = _criterion_runs(base_root, run) if base_root else {}
    out: list[Entry] = []
    for rel, pr in sorted(_criterion_runs(pr_root, run).items()):
        pm = _num(pr, "median", "point_estimate")
        bm = _num(base.get(rel), "median", "point_estimate")
        if rel not in base or pm <= 0 or bm <= 0:
            out.append({"id": rel, "delta": None, "lo": 0.0, "hi": 0.0})
            continue
        delta = (pm - bm) / bm
        half = math.hypot(_half(pr, pm), _half(base[rel], bm))
        out.append({"id": rel, "delta": delta, "lo": delta - half, "hi": delta + half})
    return out


def _half(est: object, median: float) -> float:
    """a criterion run's 95% half-width on its median, relative to that median"""
    ci = ("median", "confidence_interval")
    return (_num(est, *ci, "upper_bound") - _num(est, *ci, "lower_bound")) / 2 / median


def _from_e2e(pr_root: str, base_root: str | None, run: str | None = None) -> list[Entry]:
    """bench/e2e_delta's deltas between the two roots' `run` e2e JSON; an absent or unreadable file is an empty
    run"""
    pr = _read_json(_e2e_path(pr_root, run))
    base = _read_json(_e2e_path(base_root, run)) if base_root else None
    out: list[Entry] = []
    for d in e2e_deltas(_doc(base), _doc(pr)):
        entry: Entry = {"id": d["id"], "delta": d["delta"], "lo": d["lo"], "hi": d["hi"]}
        if d["section"]:
            entry["section"] = d["section"]
        if d["isa"]:
            entry["isa"] = d["isa"]
        out.append(entry)
    return out


def compare(pr_root: str, base_root: str | None) -> list[Entry]:
    """every benchmark's change, each interleaved pair compared on its own and then folded: the mean of the
    pairs' deltas, the band from the lowest bound to the highest, so a flag needs every pair to clear it. A
    bench new in any pair lists as new."""
    folded: dict[str, list[Entry]] = {}
    for run in runs(pr_root):
        for e in _from_criterion(pr_root, base_root, run) + _from_e2e(pr_root, base_root, run):
            folded.setdefault(e["id"], []).append(e)
    out: list[Entry] = []
    for pairs in folded.values():
        e = pairs[0].copy()
        if any(p["delta"] is None for p in pairs):
            e["delta"], e["lo"], e["hi"] = None, 0.0, 0.0
        else:
            e["delta"] = sum(p["delta"] or 0.0 for p in pairs) / len(pairs)
            e["lo"], e["hi"] = min(p["lo"] for p in pairs), max(p["hi"] for p in pairs)
        out.append(e)
    return out


def _criterion_timings(root: str, run: str | None) -> list[Timing]:
    """a root's `run` criterion medians in seconds (criterion writes nanoseconds) with their 95% bootstrap CI"""
    out: list[Timing] = []
    for rel, est in sorted(_criterion_runs(root, run).items()):
        ci = ("median", "confidence_interval")
        out.append(
            {
                "id": rel,
                "section": _section(rel),
                "median_s": _num(est, "median", "point_estimate") / 1e9,
                "lo_s": _num(est, *ci, "lower_bound") / 1e9,
                "hi_s": _num(est, *ci, "upper_bound") / 1e9,
            }
        )
    return out


def timings(root: str) -> list[Timing]:
    """every benchmark's own time on a result root, its runs folded as `compare` folds them: the mean of the
    runs' medians, the band from the lowest bound to the highest"""
    folded: dict[str, list[Timing]] = {}
    for run in runs(root):
        e2e = e2e_timings(_doc(_read_json(_e2e_path(root, run))))
        for t in _criterion_timings(root, run) + e2e:
            folded.setdefault(t["id"], []).append(t)
    return [
        {
            "id": ts[0]["id"],
            "section": ts[0]["section"] or _section(ts[0]["id"]),
            "median_s": sum(t["median_s"] for t in ts) / len(ts),
            "lo_s": min(t["lo_s"] for t in ts),
            "hi_s": max(t["hi_s"] for t in ts),
        }
        for ts in folded.values()
    ]


def results(root: str, entries: list[Entry], sha: str | None, base_sha: str | None, run_url: str | None) -> Json:
    """the run as data: each benchmark's own time with its change against the base where there is one, and the
    commit, ISA tier and CPU they ran on - what `--json` writes for a page to render"""
    change = {e["id"]: e for e in entries}
    e2e = _doc(_read_json(_e2e_path(root, runs(root)[0])))
    cpu = (e2e.get("machine_info") or {}).get("cpu")
    benches: list[Json] = []
    for t in sorted(timings(root), key=lambda t: (t["section"], t["id"])):
        c = change.get(t["id"])
        delta = None if c is None or c["delta"] is None else {"delta": c["delta"], "lo": c["lo"], "hi": c["hi"]}
        benches.append({**t, "change": delta})
    return {
        "sha": sha if sha and _SHA.fullmatch(sha) else None,
        "base_sha": base_sha if base_sha and _SHA.fullmatch(base_sha) else None,
        "run_url": run_url,
        "isa": next((e["isa"] for e in entries if e.get("isa")), None),
        "cpu": cpu.get("brand_raw") if isinstance(cpu, dict) else None,
        "pairs": len(runs(root)),
        "benches": benches,
    }


def _doc(parsed: object) -> Doc:
    """a run's e2e.json as e2e_delta reads it: what was on disk when it is an object, else an empty run"""
    return cast(Doc, parsed) if isinstance(parsed, dict) else {}


def _section(entry_id: str) -> str:
    """the collapsible section an entry belongs to: its device when the id is device-prefixed (the end-to-end
    paths, e.g. `cpu/tiny_qwen3`), else its op family (the kernels, e.g. `gemv_bf16/b16` -> gemv)."""
    head = entry_id.split("/", 1)[0]
    if head in DEVICES:
        return head
    for fam in OP_FAMILIES:
        if head == fam or head.startswith(fam + "_"):
            return fam
    return head


def regressions(entries: list[Entry], threshold: float) -> list[Entry]:
    """entries whose WHOLE 95% CI sits above +threshold - a real slowdown, not a CI that straddles zero (noise).
    Same lower-bound-clears-the-floor test the comment's verdict uses, exposed for the perf gate's exit code."""
    return [e for e in entries if e["delta"] is not None and e["lo"] > threshold]


def _verdict(e: Entry, noise: float) -> tuple[str, bool]:
    if e["delta"] is None:
        return "new", False
    if e["lo"] > noise:
        return "🔴 slower", True
    if e["hi"] < -noise:
        return "🟢 faster", False
    return "≈ noise", False


def _fmt(e: Entry) -> str:
    if e["delta"] is None:
        return "_new_"
    return f"{e['delta'] * 100:+.1f}% [{e['lo'] * 100:+.1f}%, {e['hi'] * 100:+.1f}%]"


_SHA = re.compile(r"[0-9a-f]{7,40}")


def _against(base_ref: str | None, base_sha: str | None) -> str:
    """the line naming what the change is compared against: the base's branch and the commit it was at, the
    commit only when it is a plain hex SHA (GitHub links one in a comment); '' without one"""
    if not base_sha or not _SHA.fullmatch(base_sha):
        return ""
    return f"Compared against `{_safe(base_ref)}` at {base_sha}." if base_ref else f"Compared against {base_sha}."


def render(
    entries: list[Entry], noise: float, pairs: int = 1, base_ref: str | None = None, base_sha: str | None = None
) -> tuple[str, bool]:
    head = [MARKER, "## Benchmark comparison (base → PR)", ""]
    against = _against(base_ref, base_sha)
    if against:
        head += [against, ""]
    isa = next((e["isa"] for e in entries if e.get("isa")), "")
    if isa:
        head += [f"Native and cpu benches at ISA tier `{_safe(isa)}`.", ""]
    if not entries:
        return "\n".join(head + ["_No benchmark results found._", ""]), False
    sections: dict[str, list[Entry]] = {}
    for e in entries:
        # e2e/MLX entries carry their section (the bench group); criterion entries infer it from the id
        sections.setdefault(e.get("section") or _section(e["id"]), []).append(e)
    any_reg = False
    n_changed = sum(1 for e in entries if e["delta"] is not None)
    interleaved = (
        f"Base and PR ran interleaved ({pairs} pairs, each adjacent in time); a change is flagged only when its 95% "
        f"CI clears ±{noise * 100:.0f}% in every pair"
        if pairs > 1
        else f"A change is flagged only when its 95% CI clears ±{noise * 100:.0f}%"
    )
    lines = head + [
        f"{n_changed} benchmarks compared vs the base. {interleaved} (else it reads as runner noise). The perf gate "
        "is advisory on the shared runner. Sections collapsed below.",
        "",
    ]
    # a section is open when it holds a flagged regression, so the interesting ones are visible without a click
    for name in sorted(sections):
        rows = sorted(sections[name], key=lambda e: (e["delta"] is None, -abs(e["delta"] or 0)))
        flags = [_verdict(e, noise) for e in rows]
        section_reg = any(reg for _lbl, reg in flags)
        any_reg = any_reg or section_reg
        worst = max((abs(e["delta"]) for e in rows if e["delta"] is not None), default=0.0)
        summary = f"{_safe(name)} · {len(rows)} benches · worst {worst * 100:+.1f}%" + (" · 🔴" if section_reg else "")
        lines.append(f"<details{' open' if section_reg else ''}><summary>{summary}</summary>\n")
        band = "mean of the pairs, 95% CI across them" if pairs > 1 else "95% CI"
        lines.append(f"| benchmark | Δ (median), {band} | |")
        lines.append("|---|---:|:--|")
        for e, (lbl, _reg) in zip(rows, flags):
            lines.append(f"| `{_safe(e['id'])}` | {_fmt(e)} | {lbl} |")
        lines.append("\n</details>\n")
    return "\n".join(lines) + "\n", any_reg


PLACEMENT = "placement"  # e2e_bench's group whose own times the crossover reads
CPU_LANE, HELD_LANE, FED_LANE = "cpu/ram", "cuda/vram", "cuda/ram"  # e2e_bench.LANES: device / where the weights sit
# a lane's time at a row count in each run it ran in: {run: (median seconds, the median's relative standard error)}
Runs = dict[str | None, tuple[float, float]]
Placement = dict[tuple[str, str], dict[tuple[int, str], dict[str, Runs]]]


def _placement(root: str) -> Placement:
    """the placement group on a result root, {(op, shape): {(rows, row label): {lane: {run: (median, rel se)}}}},
    read off each bench's `extra_info`, every run kept apart so a winner is held to each of them"""
    got: Placement = {}
    for run in runs(root):
        for b in _doc(_read_json(_e2e_path(root, run))).get("benchmarks", []):
            x = b.get("extra_info") or {}
            op, shape, lane, row, rows = (x.get(k) for k in ("op", "shape", "lane", "row", "rows"))
            median = _num(b, "stats", "median")
            if b.get("group") != PLACEMENT or median <= 0 or not isinstance(rows, int):
                continue
            if not (isinstance(op, str) and isinstance(shape, str) and isinstance(lane, str) and isinstance(row, str)):
                continue
            lanes = got.setdefault((op, shape), {}).setdefault((rows, row), {})
            lanes.setdefault(lane, {})[run] = (median, rel_se(b.get("stats", {})))
    return got


def _mean(t: Runs) -> float:
    """a lane's time as the table shows it: the mean of its runs' medians, as `timings` folds the runs"""
    return sum(m for m, _se in t.values()) / len(t)


def _lead(lanes: dict[str, Runs], noise: float) -> tuple[str, list[str]]:
    """a row's fastest lane (by its runs' mean median) and the lanes it does not beat: those whose median it does not
    undercut by more than `noise` with the whole 95% band of the difference, in every run of the row - the rule the
    comment holds a change to. A lane that did not run in every run the row has (a cell that failed, a lane skipped)
    cannot be told from: beating it in the runs it has proves nothing of the one it lacks. An empty list is a win;
    otherwise the row is a tie among the fastest and those"""
    every = {r for t in lanes.values() for r in t}
    best = min(lanes, key=lambda ln: _mean(lanes[ln]))
    tied = []
    for ln, t in lanes.items():
        if ln == best:
            continue
        clears = set(t) == every and set(lanes[best]) == every
        for r in every if clears else ():
            (mb, sb), (mo, so) = lanes[best][r], t[r]
            if (mb - mo) / mo + band(sb, so) >= -noise:
                clears = False
        if not clears:
            tied.append(ln)
    return best, sorted(tied, key=_lane_key)


def _outcome(lanes: dict[str, Runs], noise: float) -> str:
    """a row's outcome as markdown-safe text: the winning lane, `≈ a, b` for the lanes it cannot be told from (each
    lane's name made safe on its own, the marks between them the report's), or `a alone` where only one lane was
    timed - nothing it could have won against"""
    if len(lanes) < 2:
        return f"{_safe(next(iter(lanes)))} alone"
    best, tied = _lead(lanes, noise)
    names = [_safe(ln) for ln in sorted([best, *tied], key=_lane_key)]
    return names[0] if not tied else "≈ " + ", ".join(names)


def _lane_key(lane: str) -> tuple[int, int, str]:
    """the lanes in a table's order: the CPU, the card over held weights, over shipped ones, then the rest by the
    number they carry (the routing replay's cuts)"""
    known = (CPU_LANE, HELD_LANE, FED_LANE)
    if lane in known:
        return known.index(lane), 0, lane
    digits = "".join(c for c in lane if c.isdigit())
    return len(known), int(digits) if digits else 0, lane


def _winners(by_rows: dict[tuple[int, str], dict[str, Runs]], noise: float) -> list[str]:
    """each row's outcome in runs along the rows, markdown-safe - where the crossover is, and where it is too close
    to call: `cpu/ram r1 to r8, ≈ cpu/ram, cuda/vram at r16, cuda/vram r24 to r4096`"""
    spans: list[tuple[str, str, str]] = []  # the outcome, its first row, its last
    for (_rows, row), lanes in sorted(by_rows.items()):
        v, row = _outcome(lanes, noise), _safe(row)
        if spans and spans[-1][0] == v:
            spans[-1] = (v, spans[-1][1], row)
        else:
            spans.append((v, row, row))
    return [f"{v} at {a}" if a == b else f"{v} {a} to {b}" for v, a, b in spans]


def crossover(root: str, noise: float = 0.05) -> str:
    """The placement group's section of the comment: for each op and shape, its lanes' times by rows (ms, the mean of
    the runs' medians), the row's outcome, and the card's best against the CPU with the weights shipped from RAM (fed)
    and held in VRAM (held) - above 1 the CPU wins - under a line naming the outcomes along the rows. A lane wins a row
    only when it beats every other by more than `noise` with the whole 95% band of the difference, in every run, as
    the comment holds a change; otherwise the row is a tie (`≈`) among the lanes it cannot be told from. '' where the
    run has no placement."""
    pts = _placement(root)
    if not pts:
        return ""
    lines = [f"<details><summary>placement · where each op wins, by rows · {len(pts)} ops</summary>", ""]
    lines.append(
        f"A lane wins a row only when its lead clears ±{noise * 100:.0f}% with its whole 95% CI in every run; "
        "`≈` marks the lanes a row cannot tell apart.\n"
    )
    for (op, shape), by_rows in sorted(pts.items()):
        lanes = sorted({ln for v in by_rows.values() for ln in v}, key=_lane_key)
        ratios = [(name, ln) for name, ln in (("fed ÷ cpu", FED_LANE), ("held ÷ cpu", HELD_LANE)) if ln in lanes]
        ratios = ratios if CPU_LANE in lanes else []
        lines += [f"**{_safe(op)} {_safe(shape)}**: " + ", ".join(_winners(by_rows, noise)), ""]
        lines.append("| rows | " + " | ".join(f"`{_safe(ln)}` ms" for ln in lanes) + " | best |")
        lines[-1] += "".join(f" {name} |" for name, _ln in ratios)
        lines.append("|---" + "|---:" * len(lanes) + "|:--|" + "---:|" * len(ratios))
        for (_rows, row), runs_of in sorted(by_rows.items()):
            t = {ln: _mean(r) for ln, r in runs_of.items()}
            cells = [f"{t[ln] * 1e3:.3f}" if ln in t else "" for ln in lanes]
            v = _outcome(runs_of, noise)
            vs = [f"{t[ln] / t[CPU_LANE]:.2f}x" if ln in t and CPU_LANE in t else "" for _name, ln in ratios]
            lines.append(f"| {_safe(row)} | " + " | ".join(cells) + f" | `{v}` |" + "".join(f" {x} |" for x in vs))
        lines.append("")
    lines.append("</details>\n")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pr_root", help="this run's result root (the repo, after the benches ran)")
    ap.add_argument("--baseline", default=None, metavar="ROOT", help="the base branch run's result root")
    # 0.05: a change is flagged only when its whole 95% CI lower bound clears ±5%. Measured native criterion
    # baselines drift run-to-run up to ~+3.3% (CI lower bound ~+2.4%) on identical code - a 2% floor flagged that
    # as a regression; 5% sits above the observed drift so between-run noise reads as noise.
    ap.add_argument("--noise", type=float, default=0.05, metavar="FRAC")
    ap.add_argument("--out", default=None, metavar="PATH")
    ap.add_argument("--base-ref", default=None, help="the base's branch, named in the comment")
    ap.add_argument("--base-sha", default=None, help="the commit benched as the base, named in the comment")
    ap.add_argument("--json", default=None, metavar="PATH", help="also write the run as data: times and changes")
    ap.add_argument("--sha", default=None, help="the commit benched, recorded in --json")
    ap.add_argument("--run-url", default=None, help="the CI run's page, recorded in --json")
    ap.add_argument(
        "--gate",
        action="store_true",
        help="exit nonzero when a benchmark's whole 95%% CI clears the regression threshold (the perf gate)",
    )
    ap.add_argument(
        "--gate-threshold",
        type=float,
        default=0.10,
        metavar="FRAC",
        help="the regression trip line for --gate (default 0.10; ~4x the ~2.4%% between-run drift measured on the "
        "native baselines, and above the 0.05 comment noise floor)",
    )
    a = ap.parse_args(argv)
    entries = compare(a.pr_root, a.baseline)
    body, regressed = render(entries, a.noise, len(runs(a.pr_root)), a.base_ref, a.base_sha)
    body += crossover(a.pr_root, a.noise)
    sys.stdout.write(body)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(body)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results(a.pr_root, entries, a.sha, a.base_sha, a.run_url), f, indent=1)
    if regressed:
        sys.stderr.write("note: at least one benchmark regressed past the noise floor (informational)\n")
    if a.gate:
        reg = regressions(entries, a.gate_threshold)
        pct = a.gate_threshold * 100
        for e in reg:
            sys.stderr.write(f"regression: {e['id']} {_fmt(e)} (whole CI above +{pct:.0f}%)\n")
        if reg:
            sys.stderr.write(f"FAIL: {len(reg)} benchmark(s) regressed past the +{pct:.0f}% perf gate\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
