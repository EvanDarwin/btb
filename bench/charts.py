#!/usr/bin/env python
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""One bar chart per model and machine out of the bench's result files (the matrix's and compare.py's JSON,
`btb bench`'s JSONL), plain SVG with no dependencies: tok/s at 64 / 256 / 1024 new tokens as the bars, the
cell's other numbers as its caption; the newest result of a configuration wins. Give it result files or a
glob and it writes one SVG per model and machine under --out. See --help."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import RESULTS, ROOT
from lib.plan import mega_capable
from lib.records import (
    BenchCell,
    BenchDevice,
    BenchDtype,
    BenchMatrixCell,
    BenchRecord,
    BenchRunDoc,
    BenchSpecs,
    BenchStatus,
    Placement,
)
from lib.table import placement_line

OUT = os.path.join(ROOT, "assets", "bench")
LENGTHS = (64, 256, 1024)
ENGINES = {"btb": "btb", "mlx-lm": "mlx-lm", "llama-cpp": "llama.cpp", "airllm": "AirLLM"}
# deep-field ground; bars and labels are colored by engine (a neutral key - btb the teal primary, the rivals
# the accents), and the winner of each length is marked by an underline, never by color
GROUND, TEAL, IRIS, GOLD, CORAL = "#04050c", "#7de3d3", "#8b7bff", "#e6c27d", "#ef8f7a"
INK, MUTED, TRACK, PANEL, LINE = "#eaf0f6", "#7e8aa0", "#0e1019", "#0e1019", "#1a1e2c"
ENGINE_COLOR = {"btb": TEAL, "mlx-lm": IRIS, "llama.cpp": GOLD, "AirLLM": CORAL}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
W, PAD, LABEL_W, LEN_W, BAR_H, BAR_GAP, VALUE_W, HEADER_H = 960, 28, 196, 40, 16, 7, 58, 112
BARS_X = PAD + LABEL_W + LEN_W  # the label column, then the length gutter, then the bars
CAP_COLS = (0, 190, 330, 450)  # the aligned stat columns, offset from the bars' left


@dataclass
class Machine:
    section: str  # the README section: "Windows", "Apple silicon", "Linux"
    line: str  # "Apple M3 Pro, 36 GB unified memory, macOS 26.6.2"


@dataclass
class Row:
    model: str
    engine: str
    device: str
    dtype: str
    packed: bool  # the weights from the 12-bit store
    variant: str  # the off-default axes as prose (the megakernel off, a temperature), "" for the default
    how: str
    speeds: list[float | None]  # tok/s per answer length, None where the length was not run
    per_pass: list[float]
    first: float | None
    ram: float
    vram: float
    note: str  # "no result" for a cell that ran and failed
    when: str  # the run's finish time; the newest row of a configuration wins


SECTIONS = {"darwin": "Apple silicon", "win32": "Windows", "linux": "Linux"}

# a chart row's identity, for keeping the newest of a configuration: section, model, engine, device, dtype,
# the 12-bit store, the variant, and whether it is a failed run
RowKey = tuple[str, str, str, str, str, bool, str, bool]


def _section(specs: BenchSpecs) -> str:
    return SECTIONS.get(str(specs.get("platform")), "Unknown")


def _machine(specs: BenchSpecs) -> Machine:
    bits = [specs.get("cpu") or specs.get("machine", "")]
    if specs.get("unified_memory_gb"):
        bits.append(f"{specs['unified_memory_gb']} GB unified memory")
    elif specs.get("ram_gb"):
        bits.append(f"{specs['ram_gb']} GB RAM")
    if specs.get("gpu") and specs["gpu"] != specs.get("cpu"):
        bits.append(specs["gpu"] + (f" {specs['vram_gb']:g} GB" if specs.get("vram_gb") else ""))
    bits.append(specs.get("os", ""))
    return Machine(_section(specs), ", ".join(b for b in bits if b))


def _device_from_placement(pl: Placement) -> BenchDevice | None:
    """The device regime a standalone `btb bench` ran in, read off the engine's placement - its slimmed record
    no longer carries the axis, the matrix owns that. The MLX tier is MLX, a card's resident tier the GPU,
    host-only the CPU; a split that also fills the host is the cpu+ regime."""
    if pl.get("mlx"):
        return BenchDevice.CPU_MLX if pl.get("host") else BenchDevice.MLX
    if pl.get("resident"):
        return BenchDevice.CPU_GPU if pl.get("host") else BenchDevice.GPU
    if pl.get("host") or pl.get("cold"):
        return BenchDevice.CPU
    return None


def _cell(rec: BenchRecord) -> BenchMatrixCell:
    """A standalone `btb bench` record in the matrix cell's shape. The slimmed record carries only the numbers
    and the report, so the device and dtype are read back off the placement; the run is taken as the default
    (the megakernel on, greedy)."""
    report = rec.get("report")
    pl = report.get("placement") if report else None
    cell = BenchMatrixCell(
        model=str(rec.get("model") or rec.get("label") or ""),
        tool=None,
        pack12=False,
        mega=True,
        sampling="greedy",
        status=BenchStatus.OK,
        cells=rec.get("cells") or [],
        report=report,
    )
    if pl:
        if (dev := _device_from_placement(pl)) is not None:
            cell["device"] = dev
        if pl.get("compute_dtype") in (BenchDtype.BF16, BenchDtype.FP32):
            cell["dtype"] = BenchDtype(pl["compute_dtype"])
    return cell


def _variant(cell: BenchMatrixCell) -> str:
    """The off-default axes as prose, the way an old cell's `variant` named a setup: the megakernel turned off
    where it would apply, and a sampling temperature. Empty for the default (greedy, the megakernel on)."""
    bits: list[str] = []
    # the megakernel is btb's own MLX path; a rival has no such thing, so the variant is meaningless there
    if (
        not cell.get("tool")
        and not cell.get("mega")
        and mega_capable(cell.get("device") or "", cell.get("dtype") or "", cell.get("type"))
    ):
        bits.append("no megakernel")
    samp = cell.get("sampling") or "greedy"
    if samp != "greedy":
        bits.append(f"temperature {samp[1:]}")
    return ", ".join(bits)


def _rows(cell: BenchMatrixCell, when: str) -> list[Row]:
    """A cell's row: its decode as the configuration runs (btb's speculation included), or a failed run's
    'no result'. The cell's axes name it - the engine (`tool`), the device, the dtype, the 12-bit store, and
    the off-default toggles as the variant."""
    tool = cell.get("tool")
    engine = ENGINES.get(tool or "btb", tool or "btb")
    model = str(cell.get("repo") or cell.get("model") or "")
    dtype = str(cell.get("dtype") or "")
    packed = bool(cell.get("pack12")) and not tool
    variant = _variant(cell)
    device = " + ".join(p.upper() for p in str(cell.get("device") or "").split("+"))
    status = cell.get("status")
    # a cell that was launched (it has a wall time) but produced no numbers gets a labeled note row: the state
    # and the reason it carries. A planning-stage skip never ran, so it is not charted at all
    if status in (BenchStatus.OOM, BenchStatus.DNF, BenchStatus.DNR) and cell.get("seconds") is not None:
        why = str(cell.get("reason") or "")
        note = f"{str(status).upper()}: {why}" if why else str(status).upper()
        return [Row(model, engine, device, dtype, packed, variant, dtype, [], [], None, 0.0, 0.0, note, when)]
    cs: list[BenchCell] = sorted(cell.get("cells") or [], key=lambda c: c.get("new", 0))
    if status != BenchStatus.OK or not cs:
        return []
    by_new = {c.get("new", 0): c for c in cs}
    mid = by_new.get(256) or cs[min(1, len(cs) - 1)]
    first = mid.get("first_s")
    ram = max(c.get("peak_ram_gb", 0.0) for c in cs)
    vram = max(c.get("peak_vram_gb", 0.0) for c in cs)
    speeds: list[float | None] = []
    for n in LENGTHS:
        c = by_new.get(n)
        s_tok = (c.get("spec_s_tok") or c.get("base_s_tok")) if c is not None else None
        speeds.append(1.0 / s_tok if s_tok else None)
    tpp = [by_new[n].get("tokens_per_pass") or 1.0 for n in LENGTHS if n in by_new]
    per_pass = tpp if any(abs(t - 1.0) >= 0.005 for t in tpp) else []
    placement = placement_line(cell.get("report")) if engine == "btb" else ""
    how = ", ".join(b for b in (dtype, "the 12-bit store" if packed else "", placement, variant) if b)
    return [Row(model, engine, device, dtype, packed, variant, how, speeds, per_pass, first, ram, vram, "", when)]


def load(paths: Sequence[str], specs_for_jsonl: BenchSpecs | None = None) -> dict[str, tuple[Machine, list[Row]]]:
    """{section: (machine, rows)} out of run docs and bench records; a JSONL's machine is `specs_for_jsonl`
    (this one, by default)"""
    found: dict[RowKey, tuple[Machine, Row]] = {}
    for p in paths:
        cells: list[BenchMatrixCell]
        with open(p, encoding="utf-8") as fh:
            if p.endswith(".jsonl"):
                if specs_for_jsonl is None:
                    from lib.host import host_specs

                    specs_for_jsonl = host_specs()
                machine = _machine(specs_for_jsonl)
                when = dt.datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")
                cells = [_cell(cast(BenchRecord, json.loads(ln))) for ln in fh if ln.strip()]
            else:
                doc = cast(BenchRunDoc, json.load(fh))
                machine = _machine(doc["specs"])
                when = doc.get("finished") or doc.get("started") or ""
                cells = doc["cells"]
        for c in cells:
            for r in _rows(c, when):
                key = (machine.section, r.model, r.engine, r.device, r.dtype, r.packed, r.variant, bool(r.note))
                if key not in found or found[key][1].when < r.when:
                    found[key] = (machine, r)
    # a rival's failure is kept only from its newest attempt at that model
    newest: dict[tuple[str, str, str], str] = {}
    for machine, r in found.values():
        k = (machine.section, r.model, r.engine)
        newest[k] = max(newest.get(k, ""), r.when)
    out: dict[str, tuple[Machine, list[Row]]] = {}
    for machine, r in found.values():
        if r.note and r.when < newest[(machine.section, r.model, r.engine)]:
            continue
        out.setdefault(machine.section, (machine, []))[1].append(r)
    for _, rows in out.values():
        rows.sort(key=lambda r: (r.engine != "btb", r.engine, r.note != "", r.device, r.how))
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _wrap(s: str, width: int, lines: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    if len(out) > lines:
        out = out[: lines - 1] + [out[lines - 1][: width - 1] + "…"]
    return out


def _fmt(v: float) -> str:
    return f"{v:.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def _color(engine: str) -> str:
    """The engine's bar and label color; an unknown engine falls back to the muted ink."""
    return ENGINE_COLOR.get(engine, MUTED)


def _legend(rows: Sequence[Row]) -> tuple[str, int]:
    """The engine color key for the header panel - the engines present, in the order they appear - as markup
    relative to the panel's origin, and the panel width to size and place it."""
    seen: list[str] = []
    for r in rows:
        if r.engine not in seen:
            seen.append(r.engine)
    out, x = [], 12
    for eng in seen:
        out.append(f'<rect x="{x}" y="9" width="11" height="11" rx="3" fill="{_color(eng)}"/>')
        out.append(f'<text x="{x + 16}" y="19" font-size="12" fill="{INK}">{escape(eng)}</text>')
        x += 16 + len(eng) * 7 + 18
    return "".join(out), x


def _winners(rows: Sequence[Row]) -> dict[int, int]:
    """The row that wins each context-length bracket: the fastest at that length, one per length. The brackets
    are independent - one engine can take some lengths and another the rest, each a real win - so this is a
    map per length, never a single overall 'fastest'."""
    out: dict[int, int] = {}
    for i in range(len(LENGTHS)):
        ranked = [(r.speeds[i], ri) for ri, r in enumerate(rows) if r.speeds and r.speeds[i] is not None]
        if ranked:
            out[i] = max(ranked)[1]
    return out


def chart(model: str, machine: Machine, rows: Sequence[Row]) -> str:
    """The SVG of one model's rows: a block a row, the three lengths' tok/s as bars colored by engine, the
    winner of each length underlined, and the other numbers in an aligned stat row. Dark deep-field ground."""
    memory = "MLX" if machine.section == "Apple silicon" else "VRAM"
    span = W - BARS_X - PAD - VALUE_W
    top = max((v for r in rows for v in r.speeds if v is not None), default=0.0) or 1.0
    winner = _winners(rows)
    body: list[str] = []
    y = HEADER_H
    for ri, r in enumerate(rows):
        if ri:
            body.append(f'<line x1="{PAD}" y1="{y - 9}" x2="{W - PAD}" y2="{y - 9}" stroke="{LINE}"/>')
        name = f'<tspan fill="{_color(r.engine)}" font-weight="600">{escape(r.engine)}</tspan>'
        dev = f'<tspan fill="{MUTED}"> · {escape(r.device)}</tspan>' if r.device else ""
        g = [f'<text x="{PAD}" y="{y + 14}" font-size="15">{name}{dev}</text>']
        for i, ln in enumerate(_wrap(r.how, 26, 2)):
            g.append(f'<text x="{PAD}" y="{y + 34 + i * 15}" font-size="13" fill="{MUTED}">{escape(ln)}</text>')
        if not r.speeds:
            for i, ln in enumerate(_wrap(r.note, 74, 2)):
                g.append(
                    f'<text x="{BARS_X}" y="{y + 14 + i * 16}" font-size="13" font-style="italic" fill="{MUTED}">{escape(ln)}</text>'
                )
            body.append("".join(g))
            y += 46
            continue
        for i, v in enumerate(r.speeds):
            by = y + i * (BAR_H + BAR_GAP)
            g.append(
                f'<text x="{BARS_X - 10}" y="{by + BAR_H - 4}" font-size="11" fill="{MUTED}" text-anchor="end">{LENGTHS[i]}</text>'
            )
            g.append(f'<rect x="{BARS_X}" y="{by}" width="{span}" height="{BAR_H}" rx="4" fill="{TRACK}"/>')
            if v is None:
                g.append(f'<text x="{BARS_X + 8}" y="{by + BAR_H - 4}" font-size="11" fill="{MUTED}">no run</text>')
                continue
            w = max(3.0, v / top * span)
            g.append(f'<rect x="{BARS_X}" y="{by}" width="{w:.1f}" height="{BAR_H}" rx="4" fill="{_color(r.engine)}"/>')
            deco = ' text-decoration="underline"' if winner.get(i) == ri else ""
            g.append(
                f'<text x="{BARS_X + w + 8:.1f}" y="{by + BAR_H - 4}" font-size="14" fill="{INK}"{deco}>{_fmt(v)}</text>'
            )
        caps = [
            ("tok/pass", " · ".join(f"{t:.2f}" for t in r.per_pass) if r.per_pass else "1.00"),
            ("first", f"{r.first:.2f}s" if r.first is not None else "—"),
            ("RAM", f"{r.ram:.1f} GB"),
            (memory, f"{r.vram:.1f} GB" if r.vram > 0.05 else "0"),
        ]
        cy = y + len(r.speeds) * (BAR_H + BAR_GAP) + 8
        for (lbl, val), off in zip(caps, CAP_COLS):
            g.append(
                f'<text x="{BARS_X + off}" y="{cy}" font-size="12" fill="{MUTED}">{lbl} <tspan fill="{INK}">{escape(val)}</tspan></text>'
            )
        body.append("".join(g))
        y += len(r.speeds) * (BAR_H + BAR_GAP) + 32
    height = y + 6
    legend, lw = _legend(rows)
    head = (
        f'<rect x="0" y="0" width="{W}" height="{height}" rx="16" fill="{GROUND}"/>'
        f'<text x="{PAD}" y="46" font-size="24" font-weight="700" fill="{INK}">{escape(model)}</text>'
        f'<text x="{PAD}" y="70" font-size="14" fill="{MUTED}">{escape(machine.line)}</text>'
        f'<text x="{PAD}" y="90" font-size="12" fill="{MUTED}">tok/s after the first token · the fastest at each length is underlined</text>'
        f'<g transform="translate({W - PAD - lw},34)"><rect width="{lw}" height="30" rx="8" fill="{PANEL}"/>{legend}</g>'
        f'<line x1="{PAD}" y1="102" x2="{W - PAD}" y2="102" stroke="{LINE}"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
        f'font-family="{FONT}">\n{head}\n' + "\n".join(body) + "\n</svg>\n"
    )


def write(
    paths: Sequence[str], out: str, specs_for_jsonl: BenchSpecs | None = None
) -> dict[str, list[tuple[str, str]]]:
    """every chart written under `out`; {section: [(model, the SVG's path)]}"""
    os.makedirs(out, exist_ok=True)
    made: dict[str, list[tuple[str, str]]] = {}
    for section, (machine, rows) in load(paths, specs_for_jsonl).items():
        models: dict[str, list[Row]] = {}
        for r in sorted(rows, key=lambda r: r.model.lower()):
            models.setdefault(r.model, []).append(r)
        for model, mrows in models.items():
            path = os.path.join(out, f"{slug(section)}-{slug(model)}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(chart(model, machine, mrows))
            made.setdefault(section, []).append((model, path))
    return made


def _expand(results: Sequence[str]) -> list[str]:
    """the result files to read: the arguments with any glob expanded, or every .json/.jsonl under bench/results."""
    globs = list(results) or [os.path.join(RESULTS, "*.json"), os.path.join(RESULTS, "*.jsonl")]
    paths: list[str] = []
    for g in globs:
        paths += sorted(glob.glob(g)) if glob.has_magic(g) else [g]
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "results", nargs="*", help="result files or globs (default: every .json and .jsonl under bench/results)"
    )
    ap.add_argument("--out", default=OUT, help="directory the SVGs are written to (default assets/bench)")
    a = ap.parse_args(argv)
    paths = _expand(a.results)
    if not paths:
        print("no result files matched", file=sys.stderr)
        return 1
    made = write(paths, a.out)
    for section, items in made.items():
        for model, path in items:
            print(f"{section}: {model} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
