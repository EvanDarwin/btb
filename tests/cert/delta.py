# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Cert-gap delta: the cert findings now, minus a banked baseline, so an agent sees only the gaps ITS change
introduced. This is the dev-loop lens on the coverage gate, not a softer version of it: the gate still blocks on
the standing gap set (uncovered CUDA cells, qwen4, fp16/fp8, kv-parity...), and closing those is the work. The
delta answers the narrower question a change has to answer first - did I add one? ruff and mypy are expected
clean outright, so they run in full elsewhere; only the cert layer, with its standing gap set, deltas.

Baseline is a snapshot taken at task start (the delta-check skill banks one before work begins); with no baseline
there is nothing to delta against, and `check` says so and fails rather than reporting a pass it did not measure.
This imports the matrices - repo code - so it is a deliberate skill step an agent runs in a tree it is developing,
never wired into a hook that fires on an unvetted checkout.

    python -m tests.cert.delta bank [--out PATH]        # snapshot the findings now (run at task start)
    python -m tests.cert.delta check --baseline PATH    # print only findings absent from the baseline; nonzero if any
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import re
import sys
from collections.abc import Callable
from enum import StrEnum
from typing import cast

from . import core, manifest

# a matrix's classified gap list: (kind, subject), the taxonomy each matrix spells in its own `Missing` enum
Findings = Callable[[], list[tuple[StrEnum, str]]]

_UNSAFE = re.compile(r"[^A-Za-z0-9_./:@ +-]")

# the manifest findings that are not a GAP cell's Missing reason
_DNR_NO_REASON = "dnr-no-reason"
_ORPHAN = "orphan-fixture"


def _safe(s: str, limit: int = 100) -> str:
    """neutralize a checkout-derived token (a symbol, a route, a fixture directory name) before it enters an
    agent's context: collapse to one line, a plain charset, and a bounded length. The report is untrusted data
    regardless (the agent must not act on it as instructions), but this strips the injection mechanics - the
    newlines and length a `... IGNORE PREVIOUS INSTRUCTIONS ...` payload needs - from a single bad edit."""
    s = _UNSAFE.sub("?", s.replace("\n", " ").replace("\r", " ").replace("\t", " "))
    return s if len(s) <= limit else s[:limit] + "..."


def matrices() -> dict[str, Findings]:
    """{module stem -> its `_findings()`} for every cert matrix in this package: the modules that expose one,
    discovered rather than listed, so a new matrix deltas the day it lands. The test_* modules are the suite that
    gates on the matrices, not matrices themselves."""
    out: dict[str, Findings] = {}
    package = importlib.import_module(__package__ or "tests.cert")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.name.startswith("test_"):
            continue
        fn = getattr(importlib.import_module(f"{package.__name__}.{info.name}"), "_findings", None)
        if callable(fn):
            out[info.name] = cast("Findings", fn)
    return out


def findings() -> set[str]:
    """every cert gap right now as a stable line - the union across the discovered matrices, core's own family
    declaration, and the coverage manifest (gap cells, reasonless DNR cells, orphan fixtures). Each line is a
    controlled kind (the Missing/Verdict taxonomy, in-repo) plus a sanitized subject; it is both the diff key and
    what an agent reads. Subjects are checkout-derived, so they go through _safe(); the taxonomy and the
    manifest's enum cell fields are in-repo and pass through."""
    out: set[str] = set()
    for name, gaps_of in matrices().items():
        for kind, subject in gaps_of():
            out.add(f"[{name}/{kind.value}] {_safe(subject)}")
    for problem in core.consistency_problems():
        out.add(f"[core/family-drift] {_safe(problem)}")
    for c in manifest.compute_cells():
        cell = f"{c.kind}/{c.storage}/{c.device}/{c.decode}"
        if c.verdict is manifest.Verdict.GAP:
            out.add(f"[manifest/{c.reason}] {cell}")
        elif c.verdict is manifest.Verdict.DNR and not c.reason.strip():
            out.add(f"[manifest/{_DNR_NO_REASON}] {cell}")  # "did not run" with no reason is a hidden gap
    for name in manifest.fixture_gaps():
        out.add(f"[manifest/{_ORPHAN}] {_safe(name)}")
    return out


def bank(path: str) -> int:
    fs = sorted(findings())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=1)
    print(f"banked {len(fs)} cert findings -> {path}")
    return 0


_GAP_CELL = re.compile(r"\[manifest/([^\]]+)\] (.*)")


def _gap_cell(line: str) -> str | None:
    """the cell a manifest GAP line names, None for any other finding: a gap cell is one finding whatever its
    Missing reason, so a cell whose first reason closed and whose next one now shows is not a new gap. A baseline
    from an older tree can carry a reason this one has retired, so any reason counts, not only today's Missing."""
    m = _GAP_CELL.match(line)
    return m.group(2) if m and m.group(1) not in (_DNR_NO_REASON, _ORPHAN) else None


def new_since(baseline_path: str) -> list[str]:
    """the findings present now but not in the baseline snapshot - the gaps this change introduced. A manifest
    gap cell counts as new only if the baseline had no gap on that cell at all."""
    with open(baseline_path, encoding="utf-8") as f:
        base = set(json.load(f))
    base_cells = {c for line in base if (c := _gap_cell(line)) is not None}
    return sorted(line for line in findings() - base if _gap_cell(line) not in base_cells)


def check(baseline_path: str) -> int:
    if not os.path.exists(baseline_path):
        print(
            f"no cert baseline at {baseline_path}: there is nothing to delta against, so this is not a pass. "
            f"Bank one at task start with `python -m tests.cert.delta bank --out {baseline_path}` (from a tree "
            "without your change, so the baseline is the standing gap set and not your own).",
            file=sys.stderr,
        )
        return 1
    new = new_since(baseline_path)
    if not new:
        print("no new cert gaps vs the baseline.")
        return 0
    print(
        f"{len(new)} new cert gap(s) this change introduced (fix these or bank a new baseline if intended):",
        file=sys.stderr,
    )
    for line in new:
        print(f"  {line}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank", help="snapshot the cert findings now (run at task start)")
    b.add_argument("--out", default=".claude/baseline/cert.json")
    c = sub.add_parser("check", help="print only findings absent from the baseline; nonzero if any")
    c.add_argument("--baseline", default=".claude/baseline/cert.json")
    a = ap.parse_args(argv)
    if a.cmd == "bank":
        return bank(a.out)
    return check(a.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
