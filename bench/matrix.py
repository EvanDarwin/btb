#!/usr/bin/env python
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The benchmark matrix: the cross-product of the axes (device, dtype, pack-12, sampling, mega, tool) this machine can run against every complete model on it, one `btb bench` process per cell, a table and a JSON file at the end. Select with --only / --not / --config; see --help."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sys
from collections.abc import Sequence
from typing import cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import COMPARE_PY, QUESTIONS, RESULTS, git_commit, say
from lib.env import ensure_compare_env
from lib.host import host_specs
from lib.plan import (
    axes_manifest,
    machine_devices,
    machine_tools,
    parse_selection,
    pick_models,
    plan_cells,
    resume_cells,
)
from lib.records import BenchGuard, BenchRunDoc, BenchStatus
from lib.run import run_cell
from lib.table import cell_label, cell_line, specs_line, tables

_KINDS = "a device (cpu, cpu+gpu, gpu, cpu+mlx, mlx), a dtype (bf16, fp32), pack12/nopack12, mega/nomega, a tool (btb, mlx-lm, airllm, llama-cpp), greedy, t<T> (e.g. t0.7), or axis=value (the only form for model=GLOB)"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--models",
        default=None,
        metavar="LIST",
        help="models to run, comma-separated: names from the cache (see `btb serve` / `/v1/models`), repo ids, "
        "local paths, or a glob over the cache names/repo ids (qwen3-*, *4b*); default: every complete model",
    )
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="REGEX",
        help="keep the cached models whose name or repo id matches (repeatable; any match keeps)",
    )
    ap.add_argument(
        "--exclude-models",
        action="append",
        default=[],
        metavar="GLOB",
        help="drop models matching these globs over the name/repo id (comma-separated, repeatable)",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KINDS",
        help="keep only cells whose axis carries one of these kinds (repeatable, comma-separated): " + _KINDS,
    )
    ap.add_argument(
        "--not",
        dest="nots",
        action="append",
        default=[],
        metavar="KINDS",
        help="drop cells carrying one of these kinds (same vocabulary as --only); applied after --only",
    )
    ap.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="A+B+...",
        help="run exactly this combination, kinds joined by + (mlx+fp32+nomega); repeatable, the plan is their "
        "union - unnamed axes stay free",
    )
    ap.add_argument(
        "--new", default="64,256,1024", metavar="N,N,...", help="answer lengths to time (default: 64,256,1024)"
    )
    ap.add_argument(
        "--prompts",
        default=QUESTIONS,
        metavar="JSONL",
        help='one {"prompt": ...} per line (default: bench/questions.jsonl, the one question set every platform benches on)',
    )
    ap.add_argument("--rows", default="", metavar="I,J,...", help="which prompt rows to run (default: all)")
    ap.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="the JSON file with the specs and every cell's detail (default: bench/results/<host>-<date>.json)",
    )
    ap.add_argument(
        "--logs", default=None, metavar="DIR", help="per-cell logs and raw records (default: next to --out)"
    )
    ap.add_argument(
        "--timeout", type=float, default=4 * 3600, metavar="S", help="per-cell limit in seconds (default: 4 hours)"
    )
    ap.add_argument(
        "--mem-floor",
        type=float,
        default=8.0,
        metavar="PCT",
        help="kill a cell when the kernel's free-memory level falls under this percentage (macOS; default 8)",
    )
    ap.add_argument(
        "--swap-cap",
        type=float,
        default=12.0,
        metavar="GB",
        help="kill a cell when the swap in use passes this (macOS; default 12 GB - a leak once filled the disk)",
    )
    ap.add_argument(
        "--disk-floor",
        type=float,
        default=8.0,
        metavar="GB",
        help="kill a cell when the results' volume has less than this free (default 8 GB)",
    )
    ap.add_argument("--markdown", action="store_true", help="also print the table in the README's markdown shape")
    ap.add_argument("--strict", action="store_true", help="treat any skipped cell as a failure (exit non-zero); for CI")
    ap.add_argument(
        "--compare-python",
        default=COMPARE_PY,
        metavar="PY",
        help="the interpreter with mlx-lm / airllm for the comparison rows (default: .venv-compare's; "
        "a comparison engine that is not there is left out)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the specs, run nothing")
    ap.add_argument(
        "--render", default=None, metavar="JSON", help="print the tables of an earlier run's JSON file, run nothing"
    )
    ap.add_argument(
        "--resume",
        default=None,
        metavar="JSON",
        help="an earlier run's JSON file: its finished cells are reused as they are and only the rest run",
    )
    a = ap.parse_args(argv)

    # WARN: the table's box drawing is UTF-8; a stdout redirected to a file on Windows is cp1252 otherwise
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if sys.platform == "win32" and reconfigure is not None:
        with contextlib.suppress(Exception):
            reconfigure(encoding="utf-8")

    if a.render:
        doc = cast(BenchRunDoc, json.load(open(a.render, encoding="utf-8")))
        say(specs_line(doc["specs"]))
        say(f"btb {doc.get('btb')}, {doc.get('started')}")
        tables(doc["cells"], ",".join(str(n) for n in doc["new"]), a.markdown)
        return 0

    allow, deny, points = parse_selection(a.only, a.nots, a.config, ap.error)
    # the comparison environment is looked at (and its prompt shown) only when a comparison row could run: a
    # selection pinned to btb alone wants none
    tool_only = allow.get("tool", set()) | {p["tool"] for p in points if "tool" in p}
    wants_compare = not (tool_only and tool_only <= {"btb"})
    compare_py = ensure_compare_env(a.compare_python, ask=not a.dry_run) if wants_compare else None
    devices = machine_devices()
    tools = machine_tools(compare_py) if wants_compare else []
    names = [n.strip() for n in a.models.split(",") if n.strip()] if a.models else []
    entries = pick_models(names, a.filter, a.exclude_models)
    if not entries:
        say("no models to run (nothing complete in the Hugging Face cache matched)")
        return 1
    specs = host_specs()
    new = [int(x) for x in a.new.split(",") if x.strip()]
    cells = plan_cells(entries, devices, tools, allow, deny, points, compare_py=compare_py)
    if not cells:
        say("no cells to run (the selection matched nothing on this machine)")
        return 1
    if a.resume:
        resume_cells(cells, json.load(open(a.resume, encoding="utf-8")), new, a.resume)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    out = a.out or os.path.join(RESULTS, f"{specs['host'].split('.')[0]}-{stamp}.json")
    logdir = a.logs or os.path.join(os.path.dirname(os.path.abspath(out)), "logs-" + os.path.basename(out)[:-5])
    say(specs_line(specs))
    say(
        f"btb {git_commit()}; {len(entries)} model(s) x {', '.join(devices)}"
        f"{' + ' + ', '.join(tools) if tools else ''}: "
        f"{sum(1 for c in cells if not c.get('status') and not c.get('resumed'))} cells to run, "
        f"{sum(1 for c in cells if c.get('status') == BenchStatus.DNR)} not run, "
        f"{sum(1 for c in cells if c.get('resumed'))} reused; answers of {a.new} tokens"
    )
    for c in cells:
        what = (
            f"DNR: {c.get('reason')}"
            if c.get("status") == BenchStatus.DNR
            else "reused from the earlier run"
            if c.get("resumed")
            else " ".join(c["args"])
        )
        say(f"  {c['model']:<24} {cell_label(c):<18} {c['dtype']:<5} {what}")
    if a.dry_run:
        return 0
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    started = dt.datetime.now().isoformat(timespec="seconds")

    def save() -> None:
        doc: BenchRunDoc = {
            "specs": specs,
            "btb": git_commit(),
            "started": started,
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
            "new": new,
            "prompts": os.path.abspath(a.prompts),
            "rows": a.rows or "all",
            "axes": axes_manifest(cells),
            "cells": cells,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, ensure_ascii=False)

    for i, c in enumerate(cells):
        if c.get("resumed") or c.get("status") == BenchStatus.DNR:  # reused, or ruled out at planning
            continue
        say(f"[{i + 1}/{len(cells)}] {c['model']} {cell_label(c)} {c['dtype']} ...")
        run_cell(
            c,
            a.prompts,
            a.new,
            a.rows,
            logdir,
            a.timeout,
            compare_py=compare_py,
            guard=BenchGuard(mem_floor=a.mem_floor, swap_cap=a.swap_cap, disk_floor=a.disk_floor),
        )
        say("  " + cell_line(c))
        if c.get("memory"):
            m = c["memory"]
            say(
                f"  memory over the cell: free level down to {m['min_level']:.0f}%, swap up to {m['max_swap_gb']:.1f} GB"
                + (f", disk down to {m['min_disk_gb']:.1f} GB" if "min_disk_gb" in m else "")
            )
        save()
    tables(cells, a.new, a.markdown)
    say(out)
    # OOM / DNF / DNR are valid recorded outcomes, so a plain run reports them and still succeeds; --strict
    # (for CI) turns any cell that produced no numbers into a non-zero exit
    incomplete = [c for c in cells if c.get("status") and c.get("status") != BenchStatus.OK]
    if a.strict and incomplete:
        say(f"strict: {len(incomplete)} cell(s) did not produce numbers")
    return 1 if (a.strict and incomplete) else 0


if __name__ == "__main__":
    raise SystemExit(main())
