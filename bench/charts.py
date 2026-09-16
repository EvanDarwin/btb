#!/usr/bin/env python
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""One bar chart per model and machine out of the bench's result files (the matrix's and compare.py's JSON,
`btb bench`'s JSONL), plain SVG with no dependencies: tok/s at 64 / 256 / 1024 new tokens as the bars, the
cell's other numbers as its caption; the newest result of a configuration wins. `--embed` puts the charts into
the README under each machine's section, above its table. See --help."""

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
from lib.records import BenchCell, BenchMatrixCell, BenchRecord, BenchRunDoc, BenchSpecs
from lib.table import placement_line

README = os.path.join(ROOT, "README.md")
OUT = os.path.join(ROOT, "assets", "bench")
LENGTHS = (64, 256, 1024)
ENGINES = {"btb": "btb", "mlx-lm": "mlx-lm", "llama-cpp": "llama.cpp", "airllm": "AirLLM"}
COLORS = ("#9ecae1", "#4292c6", "#08519c")
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
W, LEFT, RIGHT, BAR, GAP, PAD, LINE = 960, 350, 80, 11, 1, 14, 14
_SECTION = re.compile(r"^## Benchmarks \((.+)\)\s*$")


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
    variant: str  # the cell's named setup within its configuration, "" for the matrix's own cells
    how: str
    speeds: list[float | None]  # tok/s per answer length, None where the length was not run
    per_pass: list[float]
    first: float | None
    ram: float
    vram: float
    note: str  # "no result" for a cell that ran and failed
    when: str  # the run's finish time; the newest row of a configuration wins


SECTIONS = {"darwin": "Apple silicon", "win32": "Windows", "linux": "Linux"}


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


def _cell(rec: BenchRecord) -> BenchMatrixCell:
    """a bench record (one `btb bench` or compare.py run) in the matrix cell's shape"""
    return BenchMatrixCell(
        model=str(rec.get("model") or rec.get("label") or ""),
        tool=rec.get("tool"),
        config=str(rec.get("device") or ""),
        device_regime=str(rec.get("device_regime") or ""),
        dtype=str(rec.get("dtype") or ""),
        status="ok",
        device=rec.get("device"),
        cells=rec.get("cells") or [],
        report=rec.get("report"),
    )


def _rows(cell: BenchMatrixCell, when: str) -> list[Row]:
    """a cell's row: its decode as the configuration runs (btb's speculation included), or a failed run's
    'no result'; the cell's own fields name the model (`repo`/`model`), the engine (`tool`), the device
    (btb's `config`, a rival's `device_regime`), the dtype, the 12-bit store and the variant"""
    tool = cell.get("tool")
    engine = ENGINES.get(tool or "btb", tool or "btb")
    model = cell.get("repo") or cell.get("model") or ""
    dtype = cell.get("dtype") or ""
    packed = bool(cell.get("packed")) and not tool
    variant = cell.get("variant") or ""
    regime = cell.get("device_regime") if tool else cell.get("config")
    device = " + ".join(p.upper() for p in (regime or "").split("+"))
    if cell.get("status") == "failed":
        return [Row(model, engine, device, dtype, packed, variant, dtype, [], [], None, 0.0, 0.0, "no result", when)]
    cs: list[BenchCell] = sorted(cell.get("cells") or [], key=lambda c: c.get("new", 0))
    if cell.get("status") != "ok" or not cs:
        return []
    by_new = {c.get("new", 0): c for c in cs}
    mid = by_new.get(256) or cs[min(1, len(cs) - 1)]
    first = mid.get("first_s")
    ram = max(c.get("peak_ram_gb", 0.0) for c in cs)
    vram = max(c.get("peak_vram_gb", 0.0) for c in cs)
    speeds: list[float | None] = []
    for n in LENGTHS:
        c = by_new.get(n)
        s_tok = (c.get("spec_s_tok") or c.get("greedy_s_tok")) if c is not None else None
        speeds.append(1.0 / s_tok if s_tok else None)
    tpp = [by_new[n].get("tokens_per_pass") or 1.0 for n in LENGTHS if n in by_new]
    per_pass = tpp if any(abs(t - 1.0) >= 0.005 for t in tpp) else []
    placement = placement_line(cell.get("report")) if engine == "btb" else ""
    how = ", ".join(b for b in (dtype, "the 12-bit store" if packed else "", placement, variant) if b)
    return [Row(model, engine, device, dtype, packed, variant, how, speeds, per_pass, first, ram, vram, "", when)]


def load(paths: Sequence[str], specs_for_jsonl: BenchSpecs | None = None) -> dict[str, tuple[Machine, list[Row]]]:
    """{section: (machine, rows)} out of run docs and bench records; a JSONL's machine is `specs_for_jsonl`
    (this one, by default)"""
    found: dict[tuple[str, str, str, str, str, bool, str, bool], tuple[Machine, Row]] = {}
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


def chart(model: str, machine: Machine, rows: Sequence[Row]) -> str:
    """the SVG of one model's rows: a block a row, three bars of tok/s and the caption of its other numbers"""
    memory = "MLX" if machine.section == "Apple silicon" else "VRAM"
    span = W - LEFT - RIGHT
    top = max((v for r in rows for v in r.speeds if v is not None), default=0.0) or 1.0
    blocks: list[str] = []
    y = 58
    for r in rows:
        how = _wrap(r.how, 52, 3)
        n_bars = len(r.speeds) if r.speeds else 1
        h_bars = n_bars * (BAR + GAP) + LINE
        h = max(h_bars, (len(how) + 1) * LINE) + PAD
        g = [f'<text x="{PAD}" y="{y + 12}" font-weight="600">{escape(r.engine)} · {escape(r.device)}</text>']
        for i, ln in enumerate(how):
            g.append(f'<text x="{PAD}" y="{y + 12 + (i + 1) * LINE}" fill="#555">{escape(ln)}</text>')
        if r.speeds:
            for i, v in enumerate(r.speeds):
                by = y + 3 + i * (BAR + GAP)
                if v is None:
                    g.append(f'<text x="{LEFT + 4}" y="{by + BAR - 2}" fill="#999">—</text>')
                    continue
                w = max(1.0, v / top * span)
                g.append(f'<rect x="{LEFT}" y="{by}" width="{w:.1f}" height="{BAR}" fill="{COLORS[i]}"/>')
                g.append(f'<text x="{LEFT + w + 4:.1f}" y="{by + BAR - 2}">{_fmt(v)}</text>')
        else:
            g.append(f'<text x="{LEFT + 4}" y="{y + 12}" fill="#999" font-style="italic">{escape(r.note)}</text>')
        cap: list[str] = []
        if r.per_pass:
            cap.append("tokens per pass " + " / ".join(f"{t:.2f}" for t in r.per_pass))
        if r.first is not None:
            cap.append(f"first token {r.first:.2f} s")
        if r.speeds:
            cap.append(f"RAM {r.ram:.1f} GB")
            cap.append(f"{memory} {r.vram:.1f} GB" if r.vram > 0.05 else f"{memory} 0")
        if cap:
            cy = y + 3 + n_bars * (BAR + GAP) + 10
            g.append(f'<text x="{LEFT}" y="{cy}" fill="#777" font-size="11">{escape(" · ".join(cap))}</text>')
        blocks.append("\n".join(g))
        y += h
        blocks.append(f'<line x1="{PAD}" y1="{y - 6}" x2="{W - PAD}" y2="{y - 6}" stroke="#eee"/>')
    height = y + 4
    legend = "".join(
        f'<rect x="{LEFT + i * 130}" y="34" width="12" height="11" fill="{COLORS[i]}"/>'
        f'<text x="{LEFT + i * 130 + 16}" y="44" fill="#555">{n} new tokens</text>'
        for i, n in enumerate(LENGTHS)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" '
        f'font-family="{FONT}" font-size="12" fill="#222">\n'
        f'<rect width="{W}" height="{height}" fill="#fff" rx="6"/>\n'
        f'<text x="{PAD}" y="24" font-size="16" font-weight="700">{escape(model)} on {escape(machine.line)}</text>\n'
        f'<text x="{PAD}" y="44" fill="#555">tok/s after the first token</text>\n'
        f"{legend}\n" + "\n".join(blocks) + "\n</svg>\n"
    )


def write(
    paths: Sequence[str], out: str, readme: str = README, specs_for_jsonl: BenchSpecs | None = None
) -> dict[str, list[tuple[str, str]]]:
    """every chart written under `out`; {section: [(model, path relative to the README)]}"""
    base = os.path.dirname(os.path.abspath(readme))
    os.makedirs(out, exist_ok=True)
    made: dict[str, list[tuple[str, str]]] = {}
    for section, (machine, rows) in load(paths, specs_for_jsonl).items():
        models: dict[str, list[Row]] = {}
        for r in sorted(rows, key=lambda r: r.model.lower()):
            models.setdefault(r.model, []).append(r)
        for model, mrows in models.items():
            name = f"{slug(section)}-{slug(model)}.svg"
            with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
                fh.write(chart(model, machine, mrows))
            made.setdefault(section, []).append((model, os.path.relpath(os.path.join(out, name), base)))
    return made


def embed(text: str, made: dict[str, list[tuple[str, str]]]) -> str:
    """the README text with each section's charts above its model table, the table folded; a section with no
    charts keeps its plain table; run again, the same text comes out"""
    lines = text.splitlines()
    out: list[str] = []
    section = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _SECTION.match(line)
        if m:
            section = m.group(1)
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if section is not None and cells[:3] == ["model", "engine", "device"]:
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            # whatever an earlier embedding put above the table (charts, the fold) goes, back to the intro
            k = len(out)
            while k > 0 and (
                out[k - 1] == ""
                or out[k - 1].startswith("![")
                or out[k - 1] in ("<details>",)
                or out[k - 1].startswith("<summary>")
            ):
                k -= 1
            out = out[:k]
            out.append("")
            charts = made.get(section) or []
            for model, path in charts:
                out.append(f"![{model} on {section}]({path.replace(os.sep, '/')})")
                out.append("")
            if charts:
                out.extend(["<details>", "<summary>The table</summary>", ""])
            out.extend(lines[i:j])
            if charts:
                out.extend(["", "</details>"])
            i = j
            if i + 1 < len(lines) and lines[i] == "" and lines[i + 1] == "</details>":
                i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="*", help="result files (default: every .json and .jsonl under bench/results)")
    ap.add_argument("--readme", default=README, help="the README the charts go into and are addressed from")
    ap.add_argument("--out", default=OUT, help="directory the SVGs are written to (default assets/bench)")
    ap.add_argument("--embed", action="store_true", help="rewrite the README with the charts above each table")
    a = ap.parse_args(argv)
    paths = a.results or sorted(
        glob.glob(os.path.join(RESULTS, "*.json")) + glob.glob(os.path.join(RESULTS, "*.jsonl"))
    )
    made = write(paths, a.out, a.readme)
    for section, items in made.items():
        for model, path in items:
            print(f"{section}: {model} -> {path}")
    if a.embed:
        with open(a.readme, encoding="utf-8") as fh:
            text = fh.read()
        with open(a.readme, "w", encoding="utf-8") as fh:
            fh.write(embed(text, made))
    return 0


if __name__ == "__main__":
    sys.exit(main())
