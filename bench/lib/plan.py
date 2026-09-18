# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The plan: the models to bench, the axis values this machine supports, and every cell of the cross-product
with its `btb bench` arguments or the reason a gate skips it. Devices and options no longer share a namespace
- each axis (device, dtype, pack-12, sampling, mega, tool) is selected and gated on its own, then composed.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch

from btb.hf import ModelEntry, ModelInfo, _config, is_gguf, model_info
from lib import say
from lib.env import compare_tools, llama_cpp_supported
from lib.records import Axes, BenchDevice, BenchDtype, BenchMatrixCell, BenchRunDoc, BenchStatus
from lib.tools import TOOLS

# per device, its `--device`/`--cpu-layers` arguments and the dtype it runs in natively - the CPU tier is fp32
# arithmetic over the bf16 weights, the card and MLX are bf16. cpu+mlx is a split that needs an explicit
# `--cpu-layers N`, so it is a note rather than a run (see mlx for the placement).
DEVICE_ARGS: dict[BenchDevice, list[str]] = {
    BenchDevice.CPU: ["--device", "cpu"],
    BenchDevice.CPU_GPU: ["--device", "cuda"],
    BenchDevice.GPU: ["--device", "cuda", "--cpu-layers", "0"],
    BenchDevice.MLX: ["--device", "mlx"],
}
NATIVE_DTYPE: dict[BenchDevice, BenchDtype] = {
    BenchDevice.CPU: BenchDtype.FP32,
    BenchDevice.CPU_GPU: BenchDtype.BF16,
    BenchDevice.GPU: BenchDtype.BF16,
    BenchDevice.CPU_MLX: BenchDtype.BF16,
    BenchDevice.MLX: BenchDtype.BF16,
}

# the families whose speculation is off by default (`v_max 0`): the megakernel does not lay them out either
MOE_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text", "gpt_oss"})

TOOL_NAMES = tuple(TOOLS)

# the closed axes and their kinds; the value spaces are disjoint, so a bare kind names its axis on its own
CLOSED_AXES: dict[str, tuple[str, ...]] = {
    "device": tuple(BenchDevice),
    "dtype": tuple(BenchDtype),
    "pack12": ("pack12", "nopack12"),
    "mega": ("mega", "nomega"),
    "tool": ("btb", *TOOL_NAMES),
}
KIND_AXIS = {v: ax for ax, vals in CLOSED_AXES.items() for v in vals}
_TEMP = re.compile(r"t\d+(\.\d+)?\Z")

# a parsed kind token: the axis it names and the kind value on it
AxisKind = tuple[str, str]
# a --only allowlist or a --not denylist: each axis mapped to the kinds named on it (the open "model" axis to
# its globs)
Kinds = dict[str, set[str]]
# a --config point: each axis it pins mapped to that one kind; the axes it leaves out stay free
Point = dict[str, str]


def _is_temp(tok: str) -> bool:
    return tok == "greedy" or bool(_TEMP.match(tok))


def parse_kind(tok: str, error: Callable[[str], Any]) -> AxisKind:
    """One kind token to (axis, value): a bare kind names its axis by the vocabulary, `axis=value` is
    explicit and the only form for the open `model` axis. An unknown kind is an error listing the axes."""
    tok = tok.strip()
    if "=" in tok:
        ax, _, val = tok.partition("=")
        ax, val = ax.strip().lower(), val.strip()
        if ax == "model":
            return ("model", val)
        if ax == "sampling":
            if not _is_temp(val):
                error(f"{val!r} is not a sampling kind (greedy, or t<T> like t0.7)")
            return ("sampling", val)
        if ax in CLOSED_AXES:
            if val not in CLOSED_AXES[ax]:
                error(f"{val!r} is not a {ax} kind; choose from {', '.join(CLOSED_AXES[ax])}")
            return (ax, val)
        error(f"unknown axis {ax!r}; choose from device, dtype, pack12, sampling, mega, tool, model")
    if tok in KIND_AXIS:
        return (KIND_AXIS[tok], tok)
    if _is_temp(tok):
        return ("sampling", tok)
    error(f"unknown kind {tok!r}; a device/dtype/pack12/mega/tool kind, greedy, t<T>, or axis=value")
    raise AssertionError  # error() does not return


def _split(tokens: str) -> list[str]:
    return [t.strip() for t in tokens.split(",") if t.strip()]


def parse_selection(
    only: Sequence[str], nots: Sequence[str], configs: Sequence[str], error: Callable[[str], Any]
) -> tuple[Kinds, Kinds, list[Point]]:
    """The `--only` allowlist, the `--not` denylist, and the `--config` points. A `--config` naming two values
    of one axis is an error (a point cannot be two places on one axis)."""
    allow: Kinds = {}
    deny: Kinds = {}
    for group, dst in ((only, allow), (nots, deny)):
        for spec in group:
            for tok in _split(spec):
                ax, val = parse_kind(tok, error)
                dst.setdefault(ax, set()).add(val)
    points: list[Point] = []
    for spec in configs:
        point: Point = {}
        for tok in spec.split("+"):
            if not tok.strip():
                continue
            ax, val = parse_kind(tok, error)
            if ax in point and point[ax] != val:
                error(f"--config {spec!r} names two {ax} values ({point[ax]}, {val})")
            point[ax] = val
        if point:
            points.append(point)
    return allow, deny, points


def machine_devices() -> list[BenchDevice]:
    """The btb device regimes this machine can run, in the default order."""
    from btb import mlx_available

    out = [BenchDevice.CPU]
    if torch.cuda.is_available():
        out += [BenchDevice.CPU_GPU, BenchDevice.GPU]
    if mlx_available():
        out += [BenchDevice.CPU_MLX, BenchDevice.MLX]
    return out


def machine_tools(compare_py: str | None) -> list[str]:
    """The comparison engines whose environment is present (and whose device this machine has)."""
    from btb import mlx_available

    have = compare_tools(compare_py)
    out = []
    if "mlx-lm" in have and mlx_available():
        out.append("mlx-lm")
    if "airllm" in have:
        out.append("airllm")
    if "llama-cpp" in have:
        out.append("llama-cpp")
    return out


def _matches(name: str, repo: str, pat: str) -> bool:
    """A model matches a pattern by a case-insensitive glob on its serve name or its repo id."""
    p = pat.lower()
    return fnmatch.fnmatch(name.lower(), p) or fnmatch.fnmatch(repo.lower(), p)


def pick_models(names: Sequence[str], filters: Sequence[str], exclude: Sequence[str]) -> list[ModelEntry]:
    """
    The models to bench: the cache's complete models, or the ones named. A name is a cache name / repo id, a
    glob over them (`qwen3-*`, `*4b*`), or a path or repo id resolved and added; `--filter` keeps regex
    matches; `--exclude-models` drops glob matches from whatever the above chose.
    """
    from btb import _model_type, available_models, is_packed, resolve, serve_name

    have = available_models()
    out: list[ModelEntry] = []
    if names:
        for n in names:
            hits = [e for e in have if _matches(e["name"], e["repo"], n) or e["name"] == serve_name(n)]
            if hits:
                out += [h for h in hits if h not in out]
            elif any(c in n for c in "*?["):
                say(f"no cached model matched {n!r}")
            else:
                d = resolve(n)
                out.append(
                    ModelEntry(
                        name=serve_name(os.path.basename(os.path.normpath(n))),
                        repo=n,
                        path=d,
                        type=_model_type(d),
                        size=0,
                        packed=is_packed(d),
                    )
                )
    elif filters:
        rx = [re.compile(f, re.IGNORECASE) for f in filters]
        out = [e for e in have if any(r.search(e["name"]) or r.search(e["repo"]) for r in rx)]
    else:
        out = list(have)
    if exclude:
        pats = [p for spec in exclude for p in _split(spec)]
        out = [e for e in out if not any(_matches(e["name"], e["repo"], p) for p in pats)]
    return out


def _store_path(e: ModelEntry) -> str | None:
    """The 12-bit store to load for this model's pack-12 cell: the model itself when it is already packed, or
    the `<path>-pack12` a `btb pack` writes beside it; None when there is none."""
    if e.get("packed"):
        return e["path"]
    store = e["path"] + "-pack12"
    return store if os.path.isdir(store) else None


def mega_capable(device: str, dtype: str, model_type: str | None) -> bool:
    """Whether the megakernel applies: the MLX device, bf16 weights, a dense family it lays out (not a MoE)."""
    return device == BenchDevice.MLX and dtype == BenchDtype.BF16 and model_type not in MOE_TYPES


# the cell-field value on each side of a boolean axis
_BOOL_KIND = {"pack12": True, "nopack12": False, "mega": True, "nomega": False}


def _kinds(axis: str, default: Iterable[str], allow: Kinds, points: Sequence[Point]) -> set[str]:
    """The kinds to enumerate on an axis: `--only` replaces the default, otherwise the default widened by any
    the `--config` points name. The points then filter the grid down to their own combinations, so a config
    naming `fp32` adds those cells beside the default bf16 ones rather than replacing them."""
    if allow.get(axis):
        return set(allow[axis])
    return set(default) | {p[axis] for p in points if axis in p}


def cell_kind(c: BenchMatrixCell, axis: str) -> str:
    """The kind value a cell carries on an axis, for matching a `--config` point or a `--not` denial."""
    if axis == "pack12":
        return "pack12" if c.get("pack12") else "nopack12"
    if axis == "mega":
        return "mega" if c.get("mega") else "nomega"
    if axis == "tool":
        return c.get("tool") or "btb"
    return cast(str, c.get(axis, ""))


def _denied(c: BenchMatrixCell, deny: Kinds) -> bool:
    for axis, vals in deny.items():
        if axis == "model":
            if any(_matches(c["model"], c["repo"], p) for p in vals):
                return True
        elif cell_kind(c, axis) in vals:
            return True
    return False


def _in_points(c: BenchMatrixCell, points: Sequence[Point]) -> bool:
    """With `--config` points given, a cell runs only when it matches one on every axis that point names."""
    if not points:
        return True
    for p in points:
        if all(
            (_matches(c["model"], c["repo"], v) if ax == "model" else cell_kind(c, ax) == v) for ax, v in p.items()
        ):
            return True
    return False


def _gate(c: BenchMatrixCell, e: ModelEntry) -> str | None:
    """None to run the cell, else why a gate skips it (recorded, never spawned)."""
    device, dtype = c["device"], c["dtype"]
    if device == BenchDevice.CPU_MLX:
        return "cpu+mlx needs an explicit --cpu-layers split (see mlx for the planner's placement)"
    if dtype == BenchDtype.BF16 and device == BenchDevice.CPU:
        return "bf16 is the card/MLX path; the CPU tier is fp32"
    if dtype == BenchDtype.FP32 and device == BenchDevice.MLX:
        return "MLX runs bf16 only for now until support is added"
    if c.get("mega"):
        if device != BenchDevice.MLX:
            return "the megakernel is the MLX path"
        if dtype != BenchDtype.BF16:
            return "the megakernel lays out bf16 weights"
        if e["type"] in MOE_TYPES:
            return "the megakernel is for a dense model, not a mixture of experts"
    if c.get("pack12") and _store_path(e) is None:
        return "no 12-bit store beside this model (btb pack writes one)"
    return None


# the storage description of each path, read once from the files and reused across a model's cells
_INFO: dict[str, ModelInfo] = {}


def _info(path: str) -> ModelInfo:
    """The model's storage description (format, quant, parameters), read once per path; a fresh copy each call
    so a cell that swaps its path (a pack-12 store) never rewrites another cell's."""
    if path not in _INFO:
        _INFO[path] = model_info(path)
    return cast(ModelInfo, dict(_INFO[path]))


def base_cell(e: ModelEntry) -> BenchMatrixCell:
    return BenchMatrixCell(
        model=e["name"],
        repo=e["repo"],
        path=e["path"],
        type=e["type"],
        packed=bool(e.get("packed")),
        info=_info(e["path"]),
        tool=None,
    )


def _btb_cell(
    e: ModelEntry, device: BenchDevice, dtype: BenchDtype, pack: bool, mega: bool, samp: str
) -> BenchMatrixCell:
    args = list(DEVICE_ARGS.get(device, []))
    if dtype == BenchDtype.FP32 and NATIVE_DTYPE[device] != BenchDtype.FP32:
        args += ["--fp32", "1"]
    if mega_capable(device, dtype, e["type"]) and not mega:
        args += ["--mlx-mega", "0"]  # the off-variant of a cell the megakernel would otherwise lay out
    if samp != "greedy":
        args += ["--temperature", samp[1:]]
    cell = base_cell(e)
    cell.update(BenchMatrixCell(device=device, dtype=dtype, pack12=pack, mega=mega, sampling=samp, args=args))
    if pack and (store := _store_path(e)):
        cell["path"] = store  # load the 12-bit store rather than the bf16 weights
        cell["info"] = _info(store)
    return cell


def _arch(e: ModelEntry) -> str | None:
    """The checkpoint's architectures[0] (`Qwen3ForCausalLM`, ...), which llama.cpp keys its converter on;
    None for a GGUF or a model without a config."""
    c = _config(e["path"])
    archs = (c or {}).get("architectures") or []
    return str(archs[0]) if archs else None


def compare_cells(e: ModelEntry, tool: str, supported: frozenset[str] | None) -> list[BenchMatrixCell]:
    """
    A comparison tool runs in the same device regimes btb does, each forced through the tool's own flags
    (`BenchTool.regimes`). Its `device` axis is that regime in btb's names; a regime it cannot honestly hold
    (no CUDA, a GGUF a rival cannot read, an architecture llama.cpp's converter lacks) carries the gate note
    rather than a silent drop. `supported` is llama.cpp's convertible-architecture set (None when unknown).
    """
    spec = TOOLS[tool]
    cuda = torch.cuda.is_available()
    gguf = is_gguf(e["path"])
    arch = None if gguf else _arch(e)
    cells = []
    for regime, flags in spec.regimes:
        cell = base_cell(e)
        cell.update(
            BenchMatrixCell(
                tool=tool,
                device=BenchDevice(regime),
                dtype=BenchDtype.BF16,
                pack12=False,
                mega=False,
                sampling="greedy",
                args=["--tool", tool, *flags],
            )
        )
        if r := spec.cannot(regime, e["type"], cuda, gguf, arch, supported):
            cell["status"] = BenchStatus.DNR
            cell["reason"] = r
        cells.append(cell)
    return cells


def plan_cells(
    entries: Sequence[ModelEntry],
    devices: Sequence[str],
    tools: Sequence[str],
    allow: Kinds,
    deny: Kinds,
    points: Sequence[Point],
    compare_py: str | None = None,
) -> list[BenchMatrixCell]:
    """
    Every cell of the cross-product the selection asks for, each with its `btb bench` arguments or the reason
    a gate skips it. `devices`/`tools` are what the machine has; `allow` (--only) and `points` (--config)
    widen or narrow each axis, `deny` (--not) subtracts, then the gates mark what cannot run. `compare_py`
    reads llama.cpp's convertible-architecture set once, so a checkpoint's llama.cpp cell says whether the
    converter knows its architecture rather than only that a GGUF is needed.
    """
    want_btb = "btb" in allow["tool"] if allow.get("tool") else True
    supported = llama_cpp_supported(compare_py) if "llama-cpp" in tools else None
    cells: list[BenchMatrixCell] = []
    for e in entries:
        if want_btb:
            for d in devices:
                if allow.get("device") and d not in allow["device"]:
                    continue
                device = BenchDevice(d)
                packs = sorted(_kinds("pack12", {"nopack12"}, allow, points))
                samps = sorted(_kinds("sampling", {"greedy"}, allow, points))
                for dt in sorted(_kinds("dtype", {NATIVE_DTYPE[device]}, allow, points)):
                    dtype = BenchDtype(dt)
                    default_mega = "mega" if mega_capable(device, dtype, e["type"]) else "nomega"
                    megas = sorted(_kinds("mega", {default_mega}, allow, points))
                    for pk in packs:
                        for mk in megas:
                            for samp in samps:
                                c = _btb_cell(e, device, dtype, _BOOL_KIND[pk], _BOOL_KIND[mk], samp)
                                if _denied(c, deny) or not _in_points(c, points):
                                    continue
                                if r := _gate(c, e):
                                    c["status"] = BenchStatus.DNR
                                    c["reason"] = r
                                cells.append(c)
        for tool in tools:
            if allow.get("tool") and tool not in allow["tool"]:
                continue
            for c in compare_cells(e, tool, supported):
                if not _denied(c, deny) and _in_points(c, points):
                    cells.append(c)
    return cells


def axes_manifest(cells: Sequence[BenchMatrixCell]) -> Axes:
    """The axis values the plan actually covers, for the run document: what the grid spanned, skips included."""
    out: Axes = {}
    for axis in ("device", "dtype", "pack12", "sampling", "mega", "tool"):
        seen = {cell_kind(c, axis) for c in cells}
        out[axis] = sorted(v for v in seen if v)
    return out


def _key(c: BenchMatrixCell) -> tuple[object, ...]:
    """A cell's identity for resume: its model and its axis coordinate."""
    m, d, dt = c.get("model"), c.get("device"), c.get("dtype")
    return (m, d, dt, c.get("pack12"), c.get("sampling"), c.get("mega"), c.get("tool"))


def resume_cells(cells: list[BenchMatrixCell], prev: BenchRunDoc, new: Sequence[int], source: str) -> int:
    """
    Take over the finished cells of an earlier run's document `prev` (the same model and axis coordinate, the
    same answer lengths) with their numbers, command and log, marking them `resumed`; the rest run. `prev` is
    read back as a (total=False) run document, every field maybe-absent - a cell that does not match reruns.
    """
    if list(prev.get("new") or []) != list(new):
        return 0
    done = {_key(c): c for c in prev.get("cells") or [] if c.get("status") == BenchStatus.OK}
    n = 0
    for c in cells:
        old = done.get(_key(c))
        if old is not None and c.get("status") != BenchStatus.DNR:
            c.update(cast(BenchMatrixCell, {k: v for k, v in old.items() if k != "args"}))
            c["resumed"] = source
            n += 1
    return n
