# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The table: a row per cell out of its numbers, on the console and in the README's markdown shape
"""

from __future__ import annotations

from collections.abc import Sequence

from lib import say
from lib.records import BenchCell, BenchMatrixCell, BenchSpecs, BenchStatus, Report

# a rendered table row: each column header to its formatted value
Row = dict[str, str]


def _tps(c: BenchCell) -> float:
    s = c.get("spec_s_tok") or c.get("base_s_tok")
    return (1.0 / s) if s else 0.0


def _fmt(v: float) -> str:
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def cell_label(c: BenchMatrixCell) -> str:
    """The cell's display name, composed from its axes; the identity is the axes themselves, never this. A
    comparison tool is its name and, where it distinguishes the regime, the device; a btb cell is the device
    and whatever it turns off or on from the default (`nomega`, `pack12`, a temperature)."""
    from lib.plan import mega_capable

    if c.get("tool"):
        dev = c.get("device") or ""
        return f"{c['tool']}-{dev}" if dev and dev != "mlx" else str(c["tool"])
    parts = [c.get("device") or "?"]
    if not c.get("mega") and mega_capable(c.get("device") or "", c.get("dtype") or "", c.get("type")):
        parts.append("nomega")
    if c.get("pack12"):
        parts.append("pack12")
    samp = c.get("sampling") or "greedy"
    if samp != "greedy":
        parts.append(samp)
    return "·".join(parts)


def summarize(cell: BenchMatrixCell) -> Row | None:
    """
    One row's columns out of a cell: tok/s after the first token per answer length (speculation included
    when it is on), the no-spec baseline's tok/s, tokens per pass, first token (TTFT), the cell's total wall
    time, and the peaks
    """
    if cell.get("status") != BenchStatus.OK:
        return None
    cs = cell["cells"]
    tps = " / ".join(_fmt(_tps(c)) for c in cs)
    spec = any(c.get("spec_s_tok") for c in cs)
    base = " / ".join(_fmt(1.0 / c["base_s_tok"]) if c.get("base_s_tok") else "" for c in cs) if spec else ""
    tpp = [c.get("tokens_per_pass") or 1.0 for c in cs]
    tpp_s = "1.00" if all(abs(t - 1.0) < 0.005 for t in tpp) else " / ".join(f"{t:.2f}" for t in tpp)
    mid = cs[min(1, len(cs) - 1)]
    first = f"{mid['first_s']:.2f} s"
    ram = f"{max(c['peak_ram_gb'] for c in cs):.1f} GB"
    vram = max(c["peak_vram_gb"] for c in cs)
    vram_s = f"{vram:.1f} GB" if vram > 0.05 else "0"
    ident = mid.get("identical") or ""
    secs = cell.get("seconds")
    return {
        "tok/s": tps,
        "base tok/s": base,
        "tok/pass": tpp_s,
        "first token": first,
        "total": f"{secs:g} s" if secs is not None else "",
        "peak RAM": ram,
        "peak VRAM/MLX": vram_s,
        "identical": ident,
        "placement": placement_line(cell.get("report")),
    }


def placement_line(report: Report | None) -> str:
    """Where the model ran, off the engine's report, in words rather than the tier names: `28 layers on the
    GPU`, `30 layers on the card, 6 on the CPU`, `36 layers on the CPU`. MLX counts the layers on its GPU and
    runs the rest of the RAM-resident ones on the CPU; a CUDA card counts its resident layers, the host the
    ones the CPU runs, cold the ones streamed from the drive each pass (resident/host/cold are disjoint, `mlx`
    is a second axis over the same layers - so the raw counts double up and are not shown)."""
    pl = report.get("placement") if report else None
    if not pl:
        return ""
    resident, host = len(pl.get("resident") or ()), len(pl.get("host") or ())
    cold, mlx = len(pl.get("cold") or ()), len(pl.get("mlx") or ())
    tiers: list[tuple[int, str]] = []
    if mlx:
        tiers.append((mlx, "on the GPU"))
        if host - mlx > 0:
            tiers.append((host - mlx, "on the CPU"))
    elif resident:
        tiers.append((resident, "on the card"))
        if host:
            tiers.append((host, "on the CPU"))
    elif host:
        tiers.append((host, "on the CPU"))
    if cold:
        tiers.append((cold, "streamed from the drive"))
    parts = [f"{n} layers {where}" if i == 0 else f"{n} {where}" for i, (n, where) in enumerate(tiers)]
    if pl.get("head") == "card":
        parts.append("the head on the card")
    elif pl.get("head") == "packed":
        parts.append("the head from the 12-bit store")
    if pl.get("drafter") == "card":
        parts.append("the drafter on the card")
    elif pl.get("drafter") == "host":
        parts.append("the drafter in RAM")
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
        "base tok/s",
        "tok/pass",
        "first token",
        "total",
        "peak RAM",
        "peak VRAM/MLX",
        "status",
    ]
    rows = []
    for c in cells:
        s = summarize(c)
        if s is None:
            st = str(c.get("status") or "").upper()
            why = (c.get("reason") or "")[:60]
            rows.append([c["model"], cell_label(c), c["dtype"], *[""] * 8, f"{st}: {why}" if st else why])
        else:
            ok = "ok" + (f" (spec identical {s['identical']})" if s["identical"] else "")
            rows.append(
                [
                    c["model"],
                    cell_label(c),
                    c["dtype"],
                    s["placement"],
                    s["tok/s"],
                    s["base tok/s"],
                    s["tok/pass"],
                    s["first token"],
                    s["total"],
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
        f"| model | configuration | dtype | tok/s at {new.replace(',', ' / ')} | base tok/s | tokens per weight pass | first token | total | peak RAM | peak VRAM/MLX |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        s = summarize(c)
        if s is None:
            continue
        out.append(
            f"| {c['model']} | {cell_label(c)} | {c['dtype']} | {s['tok/s']} | {s['base tok/s']} | {s['tok/pass']} | "
            f"{s['first token']} | {s['total']} | {s['peak RAM']} | {s['peak VRAM/MLX']} |"
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
        st = str(c.get("status") or "?").upper()
        return f"{st} after {c['seconds']}s: {(c.get('reason') or '')[:200]}"
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
