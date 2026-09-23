#!/usr/bin/env python3
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Render a PR comment from base-vs-PR benchmark deltas, grouped into collapsible sections.

Two sources of bench data, both with their own confidence intervals:
  - Rust / criterion (the native per-op benches)
  - Python / pytest bench (bench/e2e_bench.py, bench/e2e_delta.py)

The PR comment will only flag changes that are statistically significant of our 95% CI.

    bench_compare.py CRITERION_DIR [--e2e e2e.json] [--noise FRAC] [--out comment.md] [--gate [--gate-threshold FRAC]]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import NotRequired, TypedDict

# the op families and devices are not re-listed here: the families are the native bench files on disk
# (tests.cert.native_ops.op_families) and the devices are the cert hardware axis (tests.cert.spec.Hardware), so
# a new bench file or hardware becomes a section with no edit here. Imported from the repo root, which this
# script (in .github/scripts/) adds to the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tests.cert.native_ops import op_families
from tests.cert.spec import Hardware

MARKER = "<!-- btb-bench-compare -->"
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
    isa: NotRequired[str]  # the e2e side records the run's ISA tier and the baseline's
    baseline_isa: NotRequired[str]


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


def _from_criterion(root: str) -> list[Entry]:
    """one entry per bench that has a change/ against the baseline; new benches (no change/) are tagged with
    delta=None so they still list."""
    out: list[Entry] = []
    if not os.path.isdir(root):
        return out
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath) != "new" or "estimates.json" not in files:
            continue
        bench = os.path.dirname(dirpath)
        rel = os.path.relpath(bench, root).replace(os.sep, "/")
        if rel.startswith("report"):
            continue
        change = _read_json(os.path.join(bench, "change", "estimates.json"))
        if not isinstance(change, dict):
            out.append({"id": rel, "delta": None, "lo": 0.0, "hi": 0.0})
            continue
        out.append({
            "id": rel,
            "delta": _num(change, "mean", "point_estimate"),
            "lo": _num(change, "mean", "confidence_interval", "lower_bound"),
            "hi": _num(change, "mean", "confidence_interval", "upper_bound"),
        })  # fmt: skip
    return out


def _from_e2e(doc: object) -> list[Entry]:
    """the pytest-bench deltas (bench/e2e_delta.py's JSON list). An item without a string id, or a document that
    is not a list at all, is dropped: a half-written e2e file costs its own rows, not the whole comment."""
    out: list[Entry] = []
    if not isinstance(doc, list):
        return out
    for item in doc:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            continue
        delta = item.get("delta")
        entry: Entry = {
            "id": item["id"],
            "delta": _num(item, "delta") if isinstance(delta, (int, float)) else None,
            "lo": _num(item, "lo"),
            "hi": _num(item, "hi"),
        }
        section = item.get("section")
        if isinstance(section, str) and section:
            entry["section"] = section
        for key in ("isa", "baseline_isa"):
            tier = item.get(key)
            if isinstance(tier, str) and tier:
                entry[key] = tier
        out.append(entry)
    return out


def tiers(entries: list[Entry]) -> tuple[str, str]:
    """(this run's ISA tier, the baseline's) as the e2e entries recorded them, "" when unrecorded"""
    for e in entries:
        if e.get("isa") or e.get("baseline_isa"):
            return e.get("isa", ""), e.get("baseline_isa", "")
    return "", ""


def comparable(entries: list[Entry]) -> list[Entry]:
    """the entries with every delta voided when the two runs' tiers differ: the criterion rows ran on the same
    CPU as the e2e rows, so one mismatch voids the whole comparison (and the gate with it)"""
    isa, base = tiers(entries)
    if not (isa and base and isa != base):
        return entries
    out: list[Entry] = []
    for e in entries:
        voided: Entry = {**e, "delta": None, "lo": 0.0, "hi": 0.0}
        out.append(voided)
    return out


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


def render(entries: list[Entry], noise: float) -> tuple[str, bool]:
    head = [MARKER, "## Benchmark comparison (main → PR)", ""]
    isa, base = tiers(entries)
    mismatch = bool(isa and base and isa != base)
    if mismatch:
        head += [
            f"⚠️ This run's native/cpu benches ran at ISA tier `{_safe(isa)}`, the main baseline's at "
            f"`{_safe(base)}` (a different runner CPU): the numbers are not comparable, so no deltas below.",
            "",
        ]
    elif isa:
        head += [f"Native and cpu benches at ISA tier `{_safe(isa)}`.", ""]
    if not entries:
        return "\n".join(head + ["_No benchmark results found._", ""]), False
    sections: dict[str, list[Entry]] = {}
    for e in entries:
        # e2e/MLX entries carry their section (the bench group); criterion entries infer it from the id
        sections.setdefault(e.get("section") or _section(e["id"]), []).append(e)
    any_reg = False
    n_changed = sum(1 for e in entries if e["delta"] is not None)
    lines = head + [
        f"{n_changed} benchmarks compared vs the main baseline; a change is flagged only when its 95% CI clears "
        f"±{noise * 100:.0f}% (else it reads as runner noise). Not a gate — the hard perf gate runs on dedicated "
        "hardware. Sections collapsed below.",
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
        lines.append("| benchmark | Δ (median), 95% CI | |")
        lines.append("|---|---:|:--|")
        for e, (lbl, _reg) in zip(rows, flags):
            cell, verdict = ("—", "not comparable") if mismatch else (_fmt(e), lbl)
            lines.append(f"| `{_safe(e['id'])}` | {cell} | {verdict} |")
        lines.append("\n</details>\n")
    return "\n".join(lines) + "\n", any_reg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("criterion_dir")
    ap.add_argument("--e2e", default=None, metavar="JSON", help="the pytest bench deltas (bench/e2e_delta.py output)")
    # 0.05: a change is flagged only when its whole 95% CI lower bound clears ±5%. Measured native criterion
    # baselines drift run-to-run up to ~+3.3% (CI lower bound ~+2.4%) on identical code - a 2% floor flagged that
    # as a regression; 5% sits above the observed drift so between-run noise reads as noise.
    ap.add_argument("--noise", type=float, default=0.05, metavar="FRAC")
    ap.add_argument("--out", default=None, metavar="PATH")
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
    entries = _from_criterion(a.criterion_dir)
    if a.e2e:
        entries += _from_e2e(_read_json(a.e2e))
    entries = comparable(entries)
    body, regressed = render(entries, a.noise)
    sys.stdout.write(body)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(body)
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
