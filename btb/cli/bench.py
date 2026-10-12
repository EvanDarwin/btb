# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb bench`: the hardware report (btb/machine.py) with no model, or prompt-set timings with one.

Each mode rejects the other's flags. Both print JSON with --json and write it with --out."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from ..kinds import Json
from ..options import BadValue, Device, DeviceName
from . import args
from .console import say, say_err, title_name
from .model import open_model

NEW = (64, 256, 1024)  # default answer lengths for a model bench
# flags (by dest) that belong to one mode only. A model's load options belong to the model bench too, but for SHARED.
MODEL_FLAGS = ("prompts", "rows", "new", "label")
MACHINE_FLAGS = ("ci", "max_seconds", "drive", "skip")
SHARED = ("device", "native")  # load options the machine report reads too


def sections(s: str) -> list[str]:
    """Parse --skip: comma-separated report section names."""
    from ..machine import SECTIONS

    out = [x.strip().lower() for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in SECTIONS]
    if bad:
        raise argparse.ArgumentTypeError(f"{', '.join(bad)}: the sections are {', '.join(SECTIONS)}")
    return out


def add(sub: Any) -> None:
    p = sub.add_parser(
        "bench",
        help="report this machine and time every op btb runs on it (or, given a model, time a prompt set)",
        description="Without a model: the machine report - the CPU, its cores and features, the memory and its "
        "modules' speed, each card, its memory and its PCIe link, the drives - then every op btb runs timed where it "
        "runs and swept across the sizes it runs at: the host's memory from its caches to DRAM and its native "
        "kernels by weight size, rows and context; the card's memory, kernels and bus by the same; Apple's GPU; and "
        "the drive under the models by read size and depth, read through btb's own reader. Each point is sampled "
        "until its median's 95%% interval is within --ci of it, or for --max-seconds, whichever comes first. With a "
        "model and --prompts: greedy (and speculative, when configured) decoding timed over the prompt set, "
        "reporting first-token latency, s/token, peak RAM and peak VRAM per answer length.",
    )
    args.model_options(
        p,
        path_required=False,
        path_help="the model to time over --prompts (a directory, Hugging Face repo id, 12-bit model or GGUF file); "
        "omit it for the machine report",
    )
    m = p.add_argument_group("the machine report (no model)")
    m.add_argument(
        "--ci",
        type=args.fraction,
        default=None,
        metavar="FRAC",
        help="a point is sampled until its median's 95%% interval is within this share of the median (default 0.02: "
        "+-2%%)",
    )
    m.add_argument(
        "--max-seconds",
        dest="max_seconds",
        type=args.seconds,
        default=None,
        metavar="S",
        help="the most one point is sampled for, its interval reported as it stands then (default 15)",
    )
    m.add_argument(
        "--drive",
        action="append",
        type=args.existing_dir,
        default=None,
        metavar="DIR",
        help="time the drive under DIR through a file written there and removed (repeatable; default: the Hugging "
        "Face cache, where models are read from)",
    )
    m.add_argument(
        "--skip",
        type=sections,
        default=None,
        metavar="LIST",
        help="sections to leave out: memory, cpu, card, bus, mlx, drive (comma-separated)",
    )
    mb = p.add_argument_group("a model's bench")
    mb.add_argument(
        "--prompts", default=None, type=args.existing_file, metavar="JSONL", help='one {"prompt": ...} per line'
    )
    mb.add_argument(
        "--rows", type=args.ints(0), default=None, metavar="I,J,...", help="which rows to run (default: all)"
    )
    mb.add_argument(
        "--new",
        type=args.ints(1),
        default=None,
        metavar="N,N,...",
        help=f"answer lengths to time (default: {','.join(map(str, NEW))})",
    )
    mb.add_argument("--label", default=None, help="name for this configuration in the output")
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help="a model's bench: append the cells as JSON lines; the machine report: write it as JSON",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="the result as one JSON object on stdout - the machine report, or a model's bench record as --out "
        "appends it - and the lines as they land on stderr",
    )
    p.set_defaults(fn=cmd_bench)


def cmd_bench(a: argparse.Namespace) -> None:
    if a.path is None:
        name = "name the model (btb bench MODEL --prompts JSONL)"
        _refuse(a, dict.fromkeys(MODEL_FLAGS), f"times a model's prompt set: {name}")
        _refuse(a, _load_defaults(), f"sets how a model loads: {name}")
        _write(machine_report(a), a, append=False)
        return
    _refuse(a, dict.fromkeys(MACHINE_FLAGS), "is the machine report's: run `btb bench` without a model for it")
    record = model_bench(a)
    if record is not None:
        _write(record, a, append=True)


def _load_defaults() -> Json:
    """The model load options' defaults by dest, those the machine report reads (`SHARED`) left out."""
    p = argparse.ArgumentParser(add_help=False)
    args.model_options(p, path_required=False)
    return {d: v for d, v in vars(p.parse_args([])).items() if d not in ("path", *SHARED)}


def _refuse(a: argparse.Namespace, flags: Json, why: str) -> None:
    """Raise on the first of `flags` (dest to default) set to something other than its default."""
    stray = [d for d, default in flags.items() if getattr(a, d) != default]
    if stray:
        raise BadValue(stray[0], why)


def _progress(a: argparse.Namespace) -> Any:
    """Where progress lines go: stderr under --json, so stdout holds only the JSON."""
    return say_err if a.as_json else say


def _write(record: Json, a: argparse.Namespace, append: bool) -> None:
    """Print the result as JSON under --json, and write it to --out (a model bench appends)."""
    text = json.dumps(record, ensure_ascii=False, default=str)
    if a.as_json:
        say(text)
    if a.out:
        with open(a.out, "a" if append else "w", encoding="utf-8") as f:
            f.write(text + "\n")
        _progress(a)(f"[bench] as JSON -> {a.out}")


# -- the machine report ---------------------------------------------------------------------------------------------


def _card(dev: DeviceName | None) -> int | None:
    """The card to benchmark: the one -d names, else the first. None for -d cpu or mlx, or without CUDA."""
    import torch

    if not torch.cuda.is_available() or (dev is not None and dev.kind is not Device.CUDA):
        return None
    card = dev.index if dev is not None and dev.index is not None else 0
    if card >= torch.cuda.device_count():
        raise BadValue("device", f"this machine has {torch.cuda.device_count()} card(s)", str(dev))
    return card


def machine_report(a: argparse.Namespace) -> Json:
    """Run the hardware report (btb/machine.py) with the --native library on the -d card."""
    from ..engine.native import Native
    from ..machine import Sampler, default_drive, run

    if a.native is not None:  # bind the library the same way a model load would
        nat = a.native.strip()
        if nat.lower() in ("", "none", "off"):
            Native.unbind()
        else:
            Native.load_gemv(nat)
    sampler = Sampler(ci=a.ci or Sampler.ci, budget_s=a.max_seconds or Sampler.budget_s)
    drives = a.drive or [default_drive()]
    return run(sampler, _progress(a), card=_card(a.device), drives=drives, skip=a.skip or [])


# -- a model's bench ------------------------------------------------------------------------------------------------


def _timed(sm: Any, ids: list[int], n_new: int, sampling: Any, speculate: bool) -> tuple[list[int], Json, float, float]:
    """Generate one answer. Returns its tokens, its stats, the time to the first token, and s/token after that."""
    marks: list[float] = []
    t0 = time.perf_counter()
    gen = sm.generate(
        ids,
        n_new,
        eos=(),
        speculate=speculate,
        on_token=lambda _t: marks.append(time.perf_counter()),
        sampling=sampling,
    )
    t1 = time.perf_counter()
    return gen.tokens, gen.stats, marks[0] - t0, (t1 - marks[0]) / max(1, len(gen.tokens) - 1)


def _length(sm: Any, prompts: list[tuple[int, str]], n_new: int, sampling: Any, label: str, log: Any) -> Json:
    """Time one answer length over the (row, prompt) pairs: plain decoding, plus speculative decoding when the
    model speculates. Logs a line per prompt and a summary; returns the summary as a cell."""
    from .. import peak_memory
    from ..sysinfo import hard_page_faults, page_faults, process_read_bytes

    spec = sm.v_max > 0
    firsts, base, fast, tpp, same = [], [], [], [], 0
    faults0, hard0, read0 = page_faults(), hard_page_faults(), process_read_bytes()
    for ri, prompt in prompts:
        ids = sm.prompt_ids(prompt)
        g, _, first, tg = _timed(sm, ids, n_new, sampling, speculate=False)
        firsts.append(first)
        base.append(tg)
        line = (
            f"[bench] {label} new={n_new} row {ri}: prompt {len(ids)}, first token {first:.2f}s, then {tg:.3f} s/token"
        )
        if spec:
            o, c, _, ts = _timed(sm, ids, n_new, sampling, speculate=True)
            fast.append(ts)
            tpp.append(len(o) / max(1, c.get("forwards", 1)))
            same += int(o == g)
            line += f" | speculative {ts:.3f} s/token, {tpp[-1]:.2f} tokens/pass, identical {o == g}"
        # peak RSS on every row, so a leak shows as a climbing number
        log(line + f" | peak RSS {peak_memory(sm)[0] / 2**30:.1f} GB")
    k = len(prompts)
    rss, vram = peak_memory(sm)
    cell: Json = {
        "new": n_new,
        "first_s": sum(firsts) / k,
        "base_s_tok": sum(base) / k,
        "spec_s_tok": (sum(fast) / k) if spec else None,
        "tokens_per_pass": (sum(tpp) / k) if spec else 1.0,
        "identical": f"{same}/{k}" if spec else "",
        "peak_ram_gb": rss / 2**30,
        "peak_vram_gb": vram / 2**30,
        "page_faults": page_faults() - faults0,
        "hard_faults": hard_page_faults() - hard0,
        "read_gb": (process_read_bytes() - read0) / 2**30,
    }
    speculative = (
        f"; speculative {cell['spec_s_tok']:.3f} s/token = {1 / cell['spec_s_tok']:.2f} tok/s, "
        f"{cell['tokens_per_pass']:.2f} tokens/pass, identical {cell['identical']}"
        if spec
        else ""
    )
    log(
        f"[bench] {label} new={n_new}: first token {cell['first_s']:.2f}s; baseline {cell['base_s_tok']:.3f} s/token "
        f"= {1 / cell['base_s_tok']:.2f} tok/s{speculative}; peak RAM {cell['peak_ram_gb']:.1f} GB, peak VRAM "
        f"{cell['peak_vram_gb']:.1f} GB, page faults {cell['page_faults']} ({cell['hard_faults']} hard), "
        f"read {cell['read_gb']:.1f} GB"
    )
    return cell


def model_bench(a: argparse.Namespace) -> Json | None:
    """Time the --prompts set at each --new answer length (greedy unless --temperature is given). Returns the
    record (cells and the engine's report) when --out or --json needs it, else None."""
    from ..sampling import GREEDY, Sampling

    if a.prompts is None:
        raise BadValue("prompts", 'a model\'s bench times a prompt set: a JSONL of {"prompt": ...} lines')
    with open(a.prompts, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise BadValue("prompts", "no prompts in it", a.prompts)
    pick = list(a.rows) if a.rows else list(range(len(rows)))
    bad = [i for i in pick if i >= len(rows)]
    if bad:
        raise BadValue("rows", f"{a.prompts} has {len(rows)} rows (0..{len(rows) - 1})", ",".join(str(i) for i in bad))
    prompts = [(i, rows[i]["prompt"]) for i in pick]
    sm, _tok = open_model(a)
    # seeded, so plain and speculative decoding can match token for token at any temperature
    sampling = Sampling.from_request(
        {f: getattr(a, f, None) for f in ("temperature", "top_p", "top_k", "seed")}, GREEDY
    ).seeded()
    label = a.label or title_name(a.path)
    log = _progress(a)
    cells = [_length(sm, prompts, n, sampling, label, log) for n in a.new or NEW]
    record = None
    if a.out or a.as_json:
        # include the engine's report, but don't lose the timings if it fails
        try:
            rep = sm.report()
        except Exception as e:
            log(f"[report] failed: {e!r}")
            rep = None
        record = {"label": label, "path": a.path, "model": title_name(a.path), "report": rep, "cells": cells}
    sm.close()
    return record
