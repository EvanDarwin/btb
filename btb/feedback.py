# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What btb tells the user when a run is worth reporting: the crash report a failed run prints as Markdown, ready
to paste into an issue, and the `--profile` folder (report.json, events.npz) a run writes for the maintainer to
attach. The CLI drives it; the presentation lives here."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import sys
import traceback
from collections.abc import Sequence
from typing import Any

from .kinds import Json, Tier
from .options import Device

ISSUES = "https://github.com/EvanDarwin/btb/issues/new"


def device_detail(d: Json) -> str:
    """One device's model and what the machine reports of it, without the engines: shared by the `devices`
    listing and the diagnostics report (which gives the engines their own column)."""
    det = d["details"]
    if d["kind"] == "gpu":
        return f"{det['model']}, {det['memory_gb']:.1f} GB VRAM, compute {det['compute_capability']}"
    if d["kind"] == Device.MLX:
        return f"{det['model']}, {det['memory_gb']:.1f} GB unified memory"
    return f"{det['model']}, {det['cores']} cores, {det['memory_gb']:.1f} GB RAM"


_OOM_MARKS = (
    "out of memory",
    "outofmemory",
    "can't allocate memory",
    "cannot allocate memory",
    "failed to allocate",
    "metal::malloc",
    "attempting to allocate",
    "kiogpucommandbuffercallbackerroroutofmemory",
    "insufficient memory",
    "std::bad_alloc",
    "mmap failed",
    "enomem",
)


def is_oom(e: BaseException) -> bool:
    """Whether `e` is the machine running out of memory: MemoryError, torch's OutOfMemoryError, or an allocator's
    message (torch CPU, MLX Metal, MPS, mmap, ENOMEM), anywhere in the chain of causes."""
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, MemoryError) or type(cur).__name__ in ("OutOfMemoryError", "MlxOutOfMemory"):
            return True
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(m in text for m in _OOM_MARKS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def not_found_report(e: BaseException, model: Any = None) -> str:
    """The message for a model that would not resolve. A bare `btb <cmd> NAME` where NAME is neither a local
    directory nor a Hugging Face repo id gets the repo-id hint; any other missing path is named plainly."""
    path = str((e.args[0] if e.args else None) or getattr(e, "filename", None) or "?")
    a_name = "/" not in path and "\\" not in path and not os.path.exists(path)
    if model is not None and path == str(model) and a_name:
        return (
            f"model {path!r} not found: it is neither a local model directory nor a Hugging Face repo id. "
            f"Repo ids are 'Owner/Name' (for example 'Qwen/Qwen3.8-Flash-Next'); pass one, or a path to a "
            f"local model directory."
        )
    if model is not None and path == str(model) and "/" in path:
        return f"model {path!r} not found: no such directory, and it is not in the local Hugging Face cache"
    return f"{path!r}: no such file or directory"


def _environment(argv: Sequence[str] | None = None) -> Json:
    """The machine, the versions and the command line, for a bug report - shared by the out-of-memory notice
    and the diagnostics bundle so both read the box the same way. `argv` is the invocation to record; None reads
    the process's own."""
    from .sysinfo import host_cpu_name, host_free_bytes, host_total_bytes

    home = os.path.expanduser("~")
    cmd = [
        (("~" + a[len(home) :]) if a.startswith(home) else a)
        for a in (list(argv) if argv is not None else sys.argv[1:])
    ]

    versions: dict[str, str] = {}
    try:
        from importlib.metadata import version

        versions["btb"] = version("btb")
    except Exception:
        versions["btb"] = "source checkout"
    for name, attr in (("torch", "torch"), ("transformers", "transformers"), ("mlx", "mlx.core")):
        mod = sys.modules.get(attr)  # mlx keeps its version on mlx.core
        if mod is not None:
            versions[name] = str(getattr(mod, "__version__", "?"))
    gpu = ""
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            gpu = f"{p.name}, {p.total_memory / 2**30:.1f} GB"
    mem = "unknown"
    with contextlib.suppress(Exception):
        mem = f"{host_free_bytes() / 2**30:.1f} GB free of {host_total_bytes() / 2**30:.1f} GB"
    cpu = "unknown"
    with contextlib.suppress(Exception):
        cpu = host_cpu_name() or "unknown"
    return {
        "command": "btb " + shlex.join(cmd),
        "platform": f"{platform.platform()} {platform.machine()}",
        "python": platform.python_version(),
        "cpu": cpu,
        "memory": mem,
        "gpu": gpu,
        "versions": versions,
    }


def crash_notice(e: BaseException) -> str:
    """the lines before the report: where it goes, and for an out-of-memory failure what to try meanwhile"""
    oom = is_oom(e)
    out = [
        "",
        "[btb] "
        + ("the machine ran out of memory." if oom else "the run failed.")
        + f" This is worth reporting: open an issue at {ISSUES} and paste the report below.",
    ]
    if oom:
        out.append(
            "Meanwhile: a larger --ram-reserve (or --vram-reserve) makes the plan stream more from the drive; "
            "--expert-cache-gb caps a mixture of experts' store; --context caps the cache."
        )
    return "\n".join(out)


def _devices_table(devices: list[Json]) -> list[str]:
    """The machine's compute devices as a Markdown table: the `--device` id, its engines, and what the machine
    reports of it - the `devices` command's listing, one row each."""
    if not devices:
        return ["- device probe unavailable"]
    rows = ["| ID | Engine(s) | Details |", "| --- | --- | --- |"]
    for d in devices:
        eng = ", ".join(f"`{b}`" for b in (d["details"].get("backends") or [])) or "—"
        rows.append(f"| {d['name']} | {eng} | {device_detail(d)} |")
    return rows


def _reproduction_command(a: argparse.Namespace, sm: Any, report: Json) -> str:
    """The run's configuration as it resolved at load, as one runnable command: the model as named, the prompt
    source, and every knob that shaped the placement, the scheduler and the speculation at its resolved value,
    so a report reproduces on another machine. Wrapped across lines by concern."""
    pl = report.get("placement") or {}
    plan = report.get("plan") or {}
    caps = plan.get("caps") or {}
    spec = report.get("speculation") or {}
    device = str(report.get("device") or getattr(a, "device", "") or "")
    cuda = device.startswith(Device.CUDA)
    head = ["btb", getattr(a, "cmd", "run")]
    model = getattr(a, "_model", None) or getattr(a, "path", None)
    if model:
        head.append(shlex.quote(str(model)))
    if getattr(a, "file", None):
        head += ["--file", shlex.quote(str(a.file))]
    elif getattr(a, "prompt", None) is not None:
        head += ["-p", shlex.quote(str(a.prompt))]
    if getattr(a, "new", None) is not None:
        head += ["--new", str(a.new)]
    place = ["--device", device] if device else []
    cfg = getattr(sm, "cfg", None)
    ctx = getattr(sm, "context", None) or int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if ctx:
        place += ["--context", str(ctx)]
    place += ["--fp32", "0" if pl.get("compute_dtype", "bf16") == "bf16" else "1"]
    place += ["--kv-host", "1" if pl.get("kv") == Tier.HOST else "0"]
    if pl.get("kv_bits"):
        place += ["--kv-bits", str(pl["kv_bits"])]
    place += ["--resident-head", "1" if plan.get("head_on_card") else "0"]
    sched = []
    rr = getattr(sm, "ram_reserve", None)
    if rr:
        sched += ["--ram-reserve", f"{rr / 2**30:.2f}"]
    if cuda and caps.get("vram_reserve_gb") is not None:
        sched += ["--vram-reserve", f"{caps['vram_reserve_gb']:.2f}"]
    sched += ["--adapt", "1" if getattr(sm, "adapt", False) else "0"]
    spec = [
        "--tree-budget",
        str(spec.get("tree_budget", 0)),
        "--v-max",
        str(spec.get("v_max", 0)),
        "--draft-vocab",
        str(spec.get("draft_vocab", 0)),
    ]
    groups = [g for g in (head, place, sched, spec) if g]
    return " \\\n  ".join(" ".join(g) for g in groups)


def issue_report(
    e: BaseException,
    a: argparse.Namespace | None = None,
    sm: Any = None,
    argv: Sequence[str] | None = None,
    profile_dir: str | None = None,
    expert_summary: str | None = None,
) -> str:
    """The crash report as Markdown for an issue: the error, the machine and versions, the devices, the command as
    run and one to reproduce it, the placement, the scheduler's memory, the inference settings and timings, the
    expert-store summary where a profile ran, the traceback's tail, and which folder to attach. `sm` and `a` are
    None when the run failed before an engine was up: those sections are left out."""
    env = _environment(argv)
    devices: list[Json] = []
    with contextlib.suppress(Exception):
        from . import available_devices

        devices = available_devices()
    err = str(e).strip().splitlines()[0] if str(e).strip() else ""
    out = [
        "## btb crash report",
        "",
        f"**Error:** `{type(e).__name__}: {err}`",
        "",
        "### Environment",
        "",
        "```sh",
        env["command"],
        "```",
        "",
        f"* **Machine:** {env['platform']}, {env['cpu']}, memory {env['memory']}"
        + (f", gpu {env['gpu']}" if env["gpu"] else ""),
        f"* **Python:** {env['python']}",
        "* **Versions:** " + ", ".join(f"{k} {v}" for k, v in env["versions"].items()),
        "",
        *_devices_table(devices),
    ]
    report: Json = {}
    if sm is not None:
        with contextlib.suppress(Exception):
            report = sm.report()
    if sm is not None and a is not None:
        out += ["", "### Reproduction", "", "```sh", _reproduction_command(a, sm, report), "```"]
    if report:
        p = report.get("placement", {})
        free = (report.get("plan") or {}).get("free") or {}
        peak = report.get("peak") or {}
        growth = report.get("growth") or {}
        spec = report.get("speculation") or {}
        ctr = report.get("counters") or {}
        out += [
            "",
            "### Placement",
            "",
            f"- device {report.get('device')}: {len(p.get('resident', []))} resident, {len(p.get('host', []))} host, "
            f"{len(p.get('cold', []))} from the drive; head {p.get('head')}, drafter {p.get('drafter')}, "
            f"kv {p.get('kv')}, compute {p.get('compute_dtype')}, packed {p.get('packed')}",
            "",
            "### Memory",
            "",
            f"- free at plan: VRAM {free.get('vram_gb', 0):.1f} GB, RAM {free.get('ram_gb', 0):.1f} GB",
            f"- peak: RAM {peak.get('ram_gb', 0):.1f} GB, VRAM {peak.get('vram_reserved_gb', 0):.1f} GB, "
            f"MLX {peak.get('mlx_gb', 0):.1f} GB",
        ]
        if growth:
            out.append(
                f"- growth past the plan: {growth.get('past_plan_gb', 0):.2f} GB "
                f"(estimate {growth.get('estimate_gb', 0):.2f} GB)"
            )
        out += [
            "",
            "### Inference",
            "",
            f"- speculation: proposer {spec.get('proposer')}, tree_budget {spec.get('tree_budget')}, "
            f"v_max {spec.get('v_max')}, draft_vocab {spec.get('draft_vocab')}",
            f"- timings: load {ctr.get('load_s', 0)} s, compute {ctr.get('compute_s', 0)} s, "
            f"cold wait {ctr.get('cold_wait_s', 0)} s",
        ]
    if expert_summary:
        out += ["", "### Expert store", "", "```", expert_summary.strip(), "```"]
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__)).strip()
    out += ["", "<details>", "<summary>Traceback</summary>", "", "```", tb[-3000:], "```", "", "</details>", ""]
    if profile_dir:
        out.append(f"Attach the profile folder `{profile_dir}` (report.json, events.npz).")
    else:
        out.append("For the ledger and the expert-store trace, run again with `--profile DIR` and attach the folder.")
    return "\n".join(out) + "\n"


def expert_summary(profile_dir: str) -> str | None:
    """the expert-store summary of the trace in `profile_dir`, or None where there is no trace or no expert
    traffic in it (a dense model leaves the watchdog's events only)"""
    with contextlib.suppress(Exception):
        import numpy as np

        from .engine.experts import ExpertProfile

        ev = np.load(os.path.join(profile_dir, "events.npz"))["events"]
        kinds = ev[:, 2]
        if any(int((kinds == k).sum()) for k in (ExpertProfile.HIT, ExpertProfile.MISS, ExpertProfile.CALL)):
            return ExpertProfile.summary(ev)
    return None


def write_profile(a: argparse.Namespace, sm: Any, log: Any) -> None:
    """Write the profile folder at `a.profile`: the expert-store trace (events.npz), flushed here if the run
    ended before close() did, and the engine's ledger (report.json). Best-effort; never raises."""
    out_dir = a.profile
    with contextlib.suppress(Exception):
        os.makedirs(out_dir, exist_ok=True)
    prof = getattr(sm, "expert_profile", None)
    if prof is not None:
        sm.expert_profile = None
        with contextlib.suppress(Exception):
            prof.save()
    report: Json = {}
    with contextlib.suppress(Exception):
        report = sm.report()
    with contextlib.suppress(Exception), open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, default=str))
    log(f"[profile] -> {out_dir} (report.json, events.npz)")
