# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The plan: the device configurations this machine can run, the models to bench, and every cell of the
matrix with its arguments or the reason it is skipped
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch

from btb.hf import ModelEntry
from btb.kinds import Json
from lib import say
from lib.env import compare_tools
from lib.records import BenchMatrixCell
from lib.tools import TOOLS

CONFIGS = ("cpu", "cpu+gpu", "gpu", "cpu+mlx", "mlx", *TOOLS)


def machine_configs(compare_py: str | None = None) -> list[str]:
    """
    The device configurations this machine can run, in the default order; the comparison engines last
    """
    from btb import mlx_available

    out = ["cpu"]
    if torch.cuda.is_available():
        out += ["cpu+gpu", "gpu"]
    if mlx_available():
        out += ["cpu+mlx", "mlx"]
    tools = compare_tools(compare_py)
    if "mlx-lm" in tools and mlx_available():
        out.append("mlx-lm")
    if "airllm" in tools:
        out.append("airllm")
    if "llama-cpp" in tools:
        out.append("llama-cpp")
    return out


def chosen_configs(devices: str | None, have: Sequence[str], error: Callable[[str], Any]) -> list[str]:
    """
    The configurations to run: the ones named (an unknown name is an error, one this machine cannot run is
    left out with a word), or every one the machine has
    """
    if not devices:
        return list(have)
    configs = [c.strip() for c in devices.split(",") if c.strip()]
    bad = [c for c in configs if c not in CONFIGS]
    if bad:
        error(f"unknown device configuration(s) {bad}; choose from {', '.join(CONFIGS)}")
    cannot = [c for c in configs if c not in have]
    if cannot:
        why = (
            "no comparison environment"
            if all(c in TOOLS for c in cannot)
            else "no card"
            if any("gpu" in c for c in cannot)
            else "no MLX"
        )
        say(f"this machine cannot run {', '.join(cannot)} ({why}); left out")
    return [c for c in configs if c in have]


def pick_models(names: Sequence[str], filters: Sequence[str]) -> list[ModelEntry]:
    """
    The models to bench: the cache's complete models narrowed by the filters, or the ones named
    """
    from btb import _model_type, available_models, is_packed, resolve, serve_name

    have = available_models()
    if not names:
        if not filters:
            return have
        rx = [re.compile(f, re.IGNORECASE) for f in filters]
        return [e for e in have if any(r.search(e["name"]) or r.search(e["repo"]) for r in rx)]
    out: list[ModelEntry] = []
    for n in names:
        hit = next((e for e in have if e["name"] == serve_name(n) or e["repo"].lower() == n.lower()), None)
        if hit is None:
            d = resolve(n)
            hit = ModelEntry(
                name=serve_name(os.path.basename(os.path.normpath(n))),
                repo=n,
                path=d,
                type=_model_type(d),
                size=0,
                packed=is_packed(d),
            )
        out.append(hit)
    return out


def plan_cells(entries: Sequence[ModelEntry], configs: Sequence[str], fp32_too: bool) -> list[BenchMatrixCell]:
    """
    Every (model, configuration, dtype) to time, with its `btb bench` arguments or the reason it is skipped.
    The placement is the engine's own: `--device` names the machine, and the planner splits the model across
    the card, the host and the drive from the free memory it measures. A comparison tool is one cell per device
    regime btb runs (the card, the CPU), forced there by its own flags, never handicapped (`compare_variants`)
    """
    cells: list[BenchMatrixCell] = []
    for e in entries:
        for c in configs:
            if c in TOOLS:
                for v in compare_variants(c, e):
                    cell = base_cell(e)
                    cell.update(v)
                    cells.append(cell)
                continue
            args, note, dtype = [], None, "bf16"
            if c == "cpu":
                args = ["--device", "cpu"]
                dtype = "fp32"
            elif c == "cpu+gpu":
                # the planner's own placement across the card and the host (a model the card holds whole is the card)
                args = ["--device", "cuda"]
            elif c == "gpu":
                args = ["--device", "cuda", "--cpu-layers", "0"]
            elif c == "cpu+mlx":
                # a CPU/GPU split on unified memory is named on the command line (--cpu-layers N); with none named
                # the planner places every layer for the GPU, which is the `mlx` cell
                note = "cpu+mlx needs an explicit --cpu-layers split (see mlx for the planner's placement)"
            elif c == "mlx":
                args = ["--device", "mlx"]
            variants = [(dtype, args)]
            if fp32_too and dtype == "bf16" and note is None:
                variants.append(("fp32", [*args, "--fp32", "1"]))
            for d, a in variants:
                cell = base_cell(e)
                cell.update(BenchMatrixCell(config=c, dtype=d, args=a, skip=note))
                cells.append(cell)
    return cells


def base_cell(e: ModelEntry) -> BenchMatrixCell:
    return BenchMatrixCell(
        model=e["name"],
        repo=e["repo"],
        path=e["path"],
        type=e["type"],
        packed=bool(e.get("packed")),
        tool=None,
        dtype="bf16",
        skip=None,
    )


def compare_variants(c: str, e: ModelEntry) -> list[BenchMatrixCell]:
    """
    A comparison tool runs in the same device regimes btb does, each forced through the tool's own flags
    (`BenchTool.regimes`: the card through AirLLM `--device cuda` / llama.cpp `--n-gpu-layers -1`, the CPU through
    `--device cpu` / `--n-gpu-layers 0`). It is not handicapped to one device and it is not the device that
    flatters btb; it is whatever the column under test uses. A regime the tool cannot honestly hold (a model
    past the card's memory with everything forced onto the card) carries the note, not a silent drop
    """
    tool = TOOLS[c]
    cuda = torch.cuda.is_available()
    return [
        BenchMatrixCell(
            config=f"{c}-{regime}" if regime != "mlx" else c,
            tool=c,
            device_regime=regime,
            dtype="bf16",
            args=["--tool", c, *flags],
            skip=tool.cannot(regime, e["type"], cuda),
        )
        for regime, flags in tool.regimes
    ]


def resume_cells(cells: list[BenchMatrixCell], prev: Json, new: Sequence[int], source: str) -> int:
    """
    Take over the finished cells of an earlier run's document `prev` (same model, configuration and dtype,
    the same answer lengths) with their numbers, command and log, marking them `resumed`; the rest run
    """
    if list(prev.get("new", [])) != list(new):
        return 0
    done = {(c["model"], c["config"], c["dtype"]): c for c in prev.get("cells", []) if c.get("status") == "ok"}
    n = 0
    for c in cells:
        old = done.get((c["model"], c["config"], c["dtype"]))
        if old is not None and not c["skip"]:
            c.update(cast(BenchMatrixCell, {k: v for k, v in old.items() if k not in ("skip", "args")}))
            c["resumed"] = source
            n += 1
    return n
