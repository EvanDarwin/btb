# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The table: a row per cell out of its numbers, on the console and in the README's markdown shape
"""

from __future__ import annotations

from collections.abc import Sequence

from btb.kinds import Json
from lib import say
from lib.records import BenchCell, BenchMatrixCell, BenchSpecs


def _tps(c: BenchCell) -> float:
    s = c.get("spec_s_tok") or c.get("greedy_s_tok")
    return (1.0 / s) if s else 0.0


def _fmt(v: float) -> str:
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def summarize(cell: BenchMatrixCell) -> dict[str, str] | None:
    """
    One row's columns out of a cell: tok/s per answer length (as the configuration runs, speculation
    included when it is on), the greedy loop's tok/s, tokens per pass, first token, peaks
    """
    if cell.get("status") != "ok":
        return None
    cs = cell["cells"]
    tps = " / ".join(_fmt(_tps(c)) for c in cs)
    spec = any(c.get("spec_s_tok") for c in cs)
    greedy = " / ".join(_fmt(1.0 / c["greedy_s_tok"]) if c.get("greedy_s_tok") else "" for c in cs) if spec else ""
    tpp = [c.get("tokens_per_pass") or 1.0 for c in cs]
    tpp_s = "1.00" if all(abs(t - 1.0) < 0.005 for t in tpp) else " / ".join(f"{t:.2f}" for t in tpp)
    mid = cs[min(1, len(cs) - 1)]
    first = f"{mid['first_s']:.2f} s"
    ram = f"{max(c['peak_ram_gb'] for c in cs):.1f} GB"
    vram = max(c["peak_vram_gb"] for c in cs)
    vram_s = f"{vram:.1f} GB" if vram > 0.05 else "0"
    ident = mid.get("identical") or ""
    return {
        "tok/s": tps,
        "greedy tok/s": greedy,
        "tok/pass": tpp_s,
        "first token": first,
        "peak RAM": ram,
        "peak VRAM/MLX": vram_s,
        "identical": ident,
        "placement": placement_line(cell.get("report")),
    }


def placement_line(report: Json | None) -> str:
    """
    Where the layers lived, off the engine's report: `28 resident, head card` / `64 host, head card`
    """
    pl = (report or {}).get("placement") or {}
    if not pl:
        return ""
    parts = [f"{len(pl[k])} {k}" for k in ("resident", "host", "cold", "mlx") if pl.get(k)]
    if pl.get("head"):
        parts.append(f"head {pl['head']}")
    if pl.get("drafter") and pl["drafter"] != "none":
        parts.append(f"drafter {pl['drafter']}")
    return ", ".join(parts)


def render(cells: Sequence[BenchMatrixCell], new: str) -> str:
    """
    The console table: a row per cell, the bench columns, and what happened to the cells that have none
    """
    lens = new.replace(",", " / ")
    heads = [
        "model",
        "config",
        "dtype",
        "placement",
        f"tok/s at {lens}",
        "greedy tok/s",
        "tok/pass",
        "first token",
        "peak RAM",
        "peak VRAM/MLX",
        "status",
    ]
    rows = []
    for c in cells:
        s = summarize(c)
        if s is None:
            why = c.get("skip") or c.get("error") or c.get("status") or ""
            rows.append(
                [c["model"], c["config"], c["dtype"], *[""] * 7, ("skipped: " if c.get("skip") else "") + why[:60]]
            )
        else:
            ok = "ok" + (f" (spec identical {s['identical']})" if s["identical"] else "")
            rows.append(
                [
                    c["model"],
                    c["config"],
                    c["dtype"],
                    s["placement"],
                    s["tok/s"],
                    s["greedy tok/s"],
                    s["tok/pass"],
                    s["first token"],
                    s["peak RAM"],
                    s["peak VRAM/MLX"],
                    ok,
                ]
            )
    w = [max(len(str(r[i])) for r in [heads, *rows]) for i in range(len(heads))]
    line = lambda r: "│ " + " │ ".join(str(x).ljust(w[i]) for i, x in enumerate(r)) + " │"
    top = "┌─" + "─┬─".join("─" * x for x in w) + "─┐"
    mid = "├─" + "─┼─".join("─" * x for x in w) + "─┤"
    bot = "└─" + "─┴─".join("─" * x for x in w) + "─┘"
    return "\n".join([top, line(heads), mid] + [line(r) for r in rows] + [bot])


def render_markdown(cells: Sequence[BenchMatrixCell], new: str) -> str:
    """
    The README's table shape: model | how it runs | tok/s | tokens per pass | first token | peak RAM | peak VRAM
    """
    out = [
        f"| model | configuration | dtype | tok/s at {new.replace(',', ' / ')} | greedy tok/s | tokens per weight pass | first token | peak RAM | peak VRAM/MLX |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        s = summarize(c)
        if s is None:
            continue
        out.append(
            f"| {c['model']} | {c['config']} | {c['dtype']} | {s['tok/s']} | {s['greedy tok/s']} | {s['tok/pass']} | "
            f"{s['first token']} | {s['peak RAM']} | {s['peak VRAM/MLX']} |"
        )
    return "\n".join(out)


def specs_line(s: BenchSpecs) -> str:
    bits = [s.get("cpu") or s.get("machine", "")]
    if s.get("cores"):
        bits[-1] += f" ({s['cores']} cores)"
    if s.get("ram_gb"):
        bits.append(
            f"{s['ram_gb']} GB RAM" if not s.get("unified_memory_gb") else f"{s['unified_memory_gb']} GB unified memory"
        )
    if s.get("gpu") and s["gpu"] != s.get("cpu"):
        bits.append(s["gpu"] + (f" {s['vram_gb']} GB" if s.get("vram_gb") else ""))
    bits.append(s.get("os", ""))
    soft = [f"python {s.get('python')}", f"torch {s.get('torch')}", f"transformers {s.get('transformers')}"]
    if s.get("mlx"):
        soft.append(f"mlx {s['mlx']}")
    return ", ".join(b for b in bits if b) + " — " + ", ".join(soft)


def cell_line(c: BenchMatrixCell) -> str:
    """
    What a cell came to, for the progress line
    """
    s = summarize(c)
    if s is None:
        return f"failed after {c['seconds']}s: {c.get('error', '')[:200]}"
    return (
        f"{s['tok/s']} tok/s, {s['tok/pass']} tok/pass, first {s['first token']}, RAM {s['peak RAM']}, "
        f"VRAM/MLX {s['peak VRAM/MLX']} ({c['seconds']}s)"
    )


def tables(cells: Sequence[BenchMatrixCell], new: str, markdown: bool) -> None:
    """
    The console table, and the README's markdown shape when asked
    """
    say()
    print(render(cells, new), flush=True)
    if markdown:
        say()
        print(render_markdown(cells, new), flush=True)
