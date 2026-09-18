# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Running one cell: a `btb bench` or a `bench/compare.py` in a child process, its record read back,
the machine watched while it runs
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import cast

from lib import HERE, ROOT
from lib.host import memory_state
from lib.records import BenchGuard, BenchMatrixCell, BenchRecord, BenchStatus
from lib.table import cell_label


def run_cell(
    cell: BenchMatrixCell,
    prompts: str,
    new: str,
    rows: str,
    logdir: str,
    timeout: float,
    compare_py: str | None = None,
    guard: BenchGuard | None = None,
) -> BenchMatrixCell:
    """
    One `btb bench` (or one `bench/compare.py`) in a child process; fills the cell with its numbers, or
    with what went wrong. `guard` watches the machine while the cell runs - `mem_floor` (the kernel's free
    memory %), `swap_cap` (GB) and `disk_floor` (GB) - and kills the cell's process group at the first one
    crossed, before the machine goes down with it (a leaking cell once swapped until the disk was full)
    """
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{cell['model']}-{cell_label(cell)}-{cell['dtype']}")
    out = os.path.join(logdir, tag + ".jsonl")
    log = os.path.join(logdir, tag + ".log")
    if os.path.exists(out):
        os.remove(out)
    label = f"{cell['model']} {cell_label(cell)} {cell['dtype']}"
    if cell.get("tool"):
        argv = [compare_py or sys.executable, os.path.join(HERE, "compare.py"), cell["path"]]
    else:
        argv = [sys.executable, "-m", "btb.cli", "bench", cell["path"]]
    argv += ["--prompts", prompts, "--new", new, "--out", out, "--label", label, *cell["args"]]
    if rows:
        argv += ["--rows", rows]
    cell["command"] = " ".join(argv)
    cell["log"] = log
    t0 = time.time()
    g: BenchGuard = guard or {}
    worst = {"min_level": 100.0, "max_swap_gb": 0.0, "min_disk_gb": float("inf")}
    kind: BenchStatus | None = None  # the guard's verdict, when it stops the cell before it finishes
    why = ""
    with open(log, "w", encoding="utf-8") as f:
        f.write(cell["command"] + "\n\n")
        f.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=(os.name != "nt"))
        while proc.poll() is None:
            time.sleep(1.0)
            st = memory_state()
            worst["min_level"] = min(worst["min_level"], st["level"])
            worst["max_swap_gb"] = max(worst["max_swap_gb"], st["swap_gb"])
            worst["min_disk_gb"] = min(worst["min_disk_gb"], st["disk_gb"])
            if time.time() - t0 > timeout:
                kind, why = BenchStatus.DNF, f"did not finish within {timeout:.0f}s"
            elif st["level"] < g.get("mem_floor", 0):
                kind, why = (
                    BenchStatus.OOM,
                    f"the kernel's free memory fell to {st['level']:.0f}% (floor {g['mem_floor']:.0f}%)",
                )
            elif st["swap_gb"] > g.get("swap_cap", float("inf")):
                kind, why = BenchStatus.OOM, f"swap in use reached {st['swap_gb']:.1f} GB (cap {g['swap_cap']:.0f} GB)"
            elif st["disk_gb"] < g.get("disk_floor", 0):
                kind, why = (
                    BenchStatus.DNR,
                    f"stopped: {st['disk_gb']:.1f} GB left on the disk (floor {g['disk_floor']:.0f} GB)",
                )
            if kind:
                if sys.platform == "win32":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
        rc = proc.returncode
    cell["seconds"] = round(time.time() - t0, 1)
    cell["memory"] = {k: round(v, 1) for k, v in worst.items() if v != float("inf")}
    rec: BenchRecord | None = None
    if os.path.exists(out):
        lines = [l for l in open(out, encoding="utf-8") if l.strip()]
        if lines:
            rec = cast(BenchRecord, json.loads(lines[-1]))
    if rec is not None:
        # the numbers off the engine's own ledger (placement, peaks included); kept even after a kill, so a
        # DNF/OOM still shows the lengths it did reach. A comparison engine's record has no report
        cell["cells"] = rec["cells"]
        cell["report"] = rec.get("report")
    if kind is not None:
        cell["status"] = kind
        cell["reason"] = why
        return cell
    if rec is None:
        tail = ""
        with contextlib.suppress(Exception):
            tail = "".join(open(log, encoding="utf-8").readlines()[-6:]).strip()
        cell["status"] = BenchStatus.DNR
        cell["reason"] = f"errored (exit {rc}): {tail[-500:]}" if tail else f"errored (exit {rc})"
        return cell
    cell["status"] = BenchStatus.OK
    return cell
