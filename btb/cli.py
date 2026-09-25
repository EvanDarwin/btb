# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from typing import Any

from .kinds import Json
from .options import BadValue, Device, DeviceName, OptionError, env_help

ART = "\n".join(
    [
        "",
        "       .94 ⋅ -.31   1.7 ⋅  .5 ⋅ -.07",
        "     -.12 ⋅ .38  ⋅  -.4  ⋅  .21 ⋅ .08",
        "         .43 ⋅ ░▒▒▒▒▒▒▒░ -.6 ⋅ .9",
        "            ░▒▓█-.31█▓█.07█▓▒░",
        "             ▒▓██1.1██-.5█▓▒",
        "              ▀▀▓███▓▀▀",
        "                 ╲▓╱",
        "            ▟█████┴█████▙",
        "            ▐ ▚▚▚▚▚▚▚▚▚ ▌",
        "            ▐ ▚▚▚▚▚▚▚▚▚ ▌",
        "            ▜███████████▛",
        "        beyond the box (btb)",
        "",
    ]
)


def _p(*a: Any) -> None:
    print(*a, flush=True)


def _e(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


def _utf8_streams() -> None:
    # the model's text is UTF-8, and a piped prompt may be too; a Windows console defaults to cp1252
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def banner(err: bool = False) -> None:
    with contextlib.suppress(Exception):
        print(ART, file=sys.stderr if err else sys.stdout, flush=True)


def _license_notice() -> None:
    """Print the FSL commercial-use notice to stderr while this build is inside its two-year window (btb.fsl);
    silent once the window has passed, when the code is Apache 2.0, and for anyone with `~/.config/btb.friend`"""
    from .fsl import restricted

    if restricted():
        _e("btb is free for personal and academic use. For commercial use, consider a paid license.")


PLACEMENT = (
    "cpu_layers",
    "resident_last",
    "tree_budget",
    "v_max",
    "fp32",
    "resident_head",
    "kv_host",
    "prefill_card",
    "context",
    "kv_bits",
    "tree_min_prob",
    "mlx_mega",
    "gguf_packed",
    "tree_step_mass",
    "draft_vocab",
    "draft_bits",
    "draft_model",
    "draft_ks",
    "ngram_p",
    "expert_cache_gb",
    "ram_reserve_gb",
    "vram_reserve_gb",
    "vram_watch",
    "cold_slots",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "draft_temp_ratio",
)


def _device(s: str) -> DeviceName | None:
    """--device as typed: cpu, mlx, cuda or cuda:N (mlx off Apple silicon refused here, before any import)"""
    from .options import OptionError, check_device

    try:
        return check_device(s)
    except OptionError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _positive(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a whole number above 0") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{s!r}: a whole number above 0")
    return n


def _port(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a port, 1 to 65535") from None
    if not 1 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"{s!r}: a port, 1 to 65535")
    return n


def _ints(least: int) -> Callable[[str], list[int]]:
    """a comma-separated list of whole numbers, each at least `least`"""

    def parse(s: str) -> list[int]:
        out = []
        for x in s.split(","):
            if not x.strip():
                continue
            try:
                n = int(x)
            except ValueError:
                raise argparse.ArgumentTypeError(f"{s!r}: whole numbers separated by commas") from None
            if n < least:
                raise argparse.ArgumentTypeError(f"{s!r}: each at least {least}")
            out.append(n)
        if not out:
            raise argparse.ArgumentTypeError(f"{s!r}: at least one number")
        return out

    return parse


def _file(s: str) -> str:
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f"{s!r}: no such file")
    return s


def _dir(s: str) -> str:
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f"{s!r}: no such directory")
    return s


def _regex(s: str) -> str:
    try:
        re.compile(s)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"{s!r}: not a regular expression ({e})") from None
    return s


def _kw(a: argparse.Namespace) -> tuple[Json, Any]:
    kw = {}
    for k in PLACEMENT:
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    nat = a.native
    if nat is not None and nat.strip().lower() in ("", "none", "off"):
        nat = ""
    return kw, nat


def _confirm_or_exit(path: str | None) -> None:
    """Consent before a command may download a model from the Hub: a no-op for a missing, local, or
    already-cached path; exits when the user declines an uncached repo (see hf.confirm_download)."""
    if not path:
        return
    from .hf import confirm_download

    if not confirm_download(path):
        _e("[btb] aborted; nothing downloaded")
        raise SystemExit(1)


def _pick_gguf_quant(path: str) -> str:
    """`path` unchanged for a local model, a normal Hub repo, or an explicit repo/id:file.gguf; a bare GGUF repo
    id (a repo of .gguf files and no config.json) turned into repo/id:file.gguf by asking which quant. On a
    terminal that is a menu; with no terminal to ask, the file list is printed and the run exits, since there is
    no safe default quant to pick."""
    from . import resolve
    from .confirm import select
    from .hf import _gb, is_gguf, repo_gguf_files

    _, _, tail = path.rpartition(":")
    if is_gguf(tail) or os.path.exists(path) or "/" not in path:
        return path  # an explicit .gguf file, a local path, or nothing with a repo id's shape
    # already cached as a normal HF model (it has a config.json): load it as-is, and offline, with no Hub probe
    with contextlib.suppress(Exception):
        if os.path.exists(os.path.join(resolve(path, local=True), "config.json")):
            return path
    files = repo_gguf_files(path)
    if not files:
        return path  # not a GGUF repo, or the Hub is unreachable: let the normal resolve/download path handle it
    labels = [f"{n}  ({_gb(sz)})" if sz else n for n, sz in files]
    i = select(f"{path} is a GGUF repo - choose a quant to download:", labels)
    if i is None:
        _e(f"[btb] {path} is a GGUF repo; name a file, e.g. {path}:{files[0][0]}")
        _e("[btb] available files:")
        for n, sz in files:
            _e(f"         {n}" + (f"  ({_gb(sz)})" if sz else ""))
        raise SystemExit(1)
    return f"{path}:{files[i][0]}"


def _open(a: argparse.Namespace) -> Any:
    from . import load, resolve

    a.path = _pick_gguf_quant(a.path)  # a bare GGUF repo id becomes repo/id:file.gguf (asks which quant on a tty)
    a._model = a.path  # the model as the user named it, for the reproduction command (resolve rewrites a.path)
    _confirm_or_exit(a.path)
    if not a.draft_model and not getattr(a, "quiet", False):
        from .hf import draft_notice

        draft_notice(a._model)
    a.path = resolve(a.path)
    kw, nat = _kw(a)
    out = _e if a.cmd == "run" else _p  # run keeps stdout for the model's text
    sm = load(a.path, device=a.device, native=nat, log=(out if a.verbose else None), **kw)
    if getattr(a, "profile", None):
        from .engine.experts import ExpertProfile

        os.makedirs(a.profile, exist_ok=True)
        sm.expert_profile = ExpertProfile(os.path.join(a.profile, "events.npz"))
        sm.expert_profile.watch()
        a._sm = sm  # main() writes the diagnostics bundle around the command, on success or on a crash
    if a.verbose:
        # the placement as the engine itself has it, once the load has settled
        try:
            out(sm.report_line(sm.report()))
        except Exception as e:
            out(f"[report] unavailable: {e!r}")
    return sm, sm.tokenizer


def _common(ap: argparse.ArgumentParser, path_required: bool = True) -> None:
    if path_required:
        ap.add_argument(
            "path",
            help="model directory, a Hugging Face repo id (downloaded into the local cache), a 12-bit model "
            "written by `btb pack`, or a GGUF file: a .gguf path, or repo/id:file.gguf fetched from the Hub",
        )
    else:
        ap.add_argument(
            "path",
            nargs="?",
            default=None,
            help="model directory, Hugging Face repo id, 12-bit model or GGUF file (.gguf, or repo/id:file.gguf) "
            "to load at startup; omit to start with "
            "nothing loaded and serve whatever is already on the machine, on request",
        )
    ap.add_argument(
        "-d",
        "--device",
        type=_device,
        default=None,
        help="cuda, cuda:N (a specific card), mlx or cpu; a device named must be here (mlx off Apple silicon, "
        "cuda without a card: an error). Default: the card, else mlx on Apple silicon, else cpu",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="print the placement, the tiers and each turn's timings"
    )
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="answer yes to prompts (e.g. downloading a model from the Hub); required to fetch one in a "
        "non-interactive run",
    )
    ap.add_argument(
        "--profile",
        default=None,
        metavar="DIR",
        help="write the run's profile to DIR: report.json (the engine's ledger) and events.npz (the expert-store "
        "trace, for a mixture of experts). Attach the folder to an issue beside the crash report a failed run prints.",
    )
    ap.add_argument(
        "--native",
        default=None,
        metavar="LIB",
        help="path to the native library (found on its own by default); 'none' runs on torch alone",
    )
    g = ap.add_argument_group("placement (override the configuration)")
    g.add_argument(
        "--cpu-layers",
        dest="cpu_layers",
        type=int,
        metavar="N",
        default=None,
        help="the first N layers run on the CPU from RAM",
    )
    g.add_argument(
        "--cold-slots",
        dest="cold_slots",
        type=int,
        metavar="N",
        default=None,
        help="slots the reader streams layers from the drive into ahead of the compute (default: as many as keep "
        "the drive busy through the resident layers, within the memory the plan leaves; at least 2)",
    )
    g.add_argument(
        "--resident-last",
        dest="resident_last",
        type=int,
        metavar="N",
        default=None,
        help="the last N layers stay on the GPU",
    )
    g.add_argument(
        "--resident-head",
        dest="resident_head",
        type=int,
        choices=(0, 1),
        default=None,
        help="keep the output head on the GPU (1) or on the CPU (0)",
    )
    g.add_argument(
        "--fp32",
        dest="fp32",
        type=int,
        choices=(0, 1),
        default=None,
        help="float32 arithmetic over the bf16 weights (1) or the model's own bf16 (0)",
    )
    g.add_argument(
        "--kv-host",
        dest="kv_host",
        type=int,
        choices=(0, 1),
        default=None,
        help="the attention cache in RAM with the weights on the GPU (1) or on the card (0); unset, the plan "
        "prices both and keeps it on the card unless the layers it would evict cost more to stream",
    )
    g.add_argument(
        "--prefill-card",
        dest="prefill_card",
        type=int,
        choices=(0, 1),
        default=None,
        help="where layers run on the CPU, set aside card memory for one layer of each kind so a long prompt's "
        "prefill runs those layers on the GPU (1); off by default (0), that memory holds more layers "
        "instead, which every generated token is faster for",
    )
    g.add_argument(
        "--kv-bits",
        dest="kv_bits",
        type=int,
        choices=(8,),
        default=None,
        help="MLX: keep the attention cache as int8 with a scale per row (half the bytes; the model's "
        "bf16 by default). Lossy: the rows carry about 0.4%% of their largest magnitude in error",
    )
    g.add_argument(
        "--context",
        dest="context",
        type=int,
        metavar="N",
        default=None,
        help="context window in tokens; past the model's own window YaRN scaling is applied",
    )
    g.add_argument(
        "--expert-cache-gb",
        dest="expert_cache_gb",
        type=float,
        metavar="GB",
        default=None,
        help="RAM the expert store of a mixture-of-experts model may grow to (default: the free RAM above the reserve)",
    )
    m = ap.add_argument_group("scheduler")
    m.add_argument(
        "--ram-reserve",
        dest="ram_reserve_gb",
        metavar="GB|%",
        default=None,
        help="RAM left to other programs, in GB or a percent of total RAM (e.g. 12 or 10%%); the run still gives "
        "memory back under pressure (default: 10%% of the RAM free at load, or the OS's floor plus the run's growth "
        "where that is more)",
    )
    m.add_argument(
        "--vram-reserve",
        dest="vram_reserve_gb",
        metavar="GB|%",
        default=None,
        help="GPU memory reserved for other programs, in GB or %% of the card "
        "(default: 0.5 GB, or 8%% of a smaller card)",
    )
    m.add_argument(
        "--vram-watch",
        dest="vram_watch",
        type=int,
        choices=(0, 1),
        default=1,
        help="move layers off the card when another program takes it, and back when it frees (default 1); "
        "--vram-watch 0 pins the placement taken at load",
    )
    s = ap.add_argument_group("speculation (models with a drafting head, or the n-gram drafter)")
    s.add_argument(
        "--tree-budget",
        dest="tree_budget",
        type=int,
        metavar="N",
        default=None,
        help="draft tree size per step; 0 turns speculation off",
    )
    s.add_argument(
        "--tree-min-prob",
        dest="tree_min_prob",
        type=float,
        metavar="P",
        default=None,
        help="minimum draft probability kept in the tree",
    )
    s.add_argument(
        "--mlx-mega",
        dest="mlx_mega",
        type=int,
        choices=(0, 1),
        default=None,
        help="the dense pass as one Metal dispatch (the megakernel; bit-exact with the fused path; the small "
        "models 15-40%% ahead of the fused path on an M3 Pro, the 4B at parity; 600 MB of arena and scratch); "
        "1 by default where it builds (dense Qwen3, every layer resident)",
    )
    s.add_argument(
        "--gguf-packed",
        dest="gguf_packed",
        type=int,
        choices=(0, 1),
        default=None,
        help="a GGUF's Q4_0 / Q4_1 / Q4_K / Q8_0 tensors multiplied as stored by the packed kernels (MLX), "
        "the file's own numbers; 0 dequantizes them to bf16 at load, where the megakernel applies (default 1)",
    )
    s.add_argument(
        "--draft-bits",
        dest="draft_bits",
        type=int,
        choices=(4, 8, 16),
        default=None,
        help="the MTP drafter's weights (its layer, fc and head over the draft ids) packed in memory at first use to "
        "4 or 8 bits and multiplied as such (4 runs as 8 on the torch tiers); the model's files untouched, the "
        "drafter's output only ever verified; 8 by default on MLX, 16 on the torch tiers",
    )
    s.add_argument(
        "--tree-step-mass",
        dest="tree_step_mass",
        type=float,
        metavar="P",
        default=None,
        help="a drafting step runs only when the nodes it would extend carry at least this much path "
        "probability; 0 always steps",
    )
    s.add_argument(
        "--draft-vocab",
        dest="draft_vocab",
        type=int,
        metavar="N",
        default=None,
        help="the drafting head scores only the first N token ids (the frequent part of a BPE "
        "vocabulary): a smaller read per draft step, the verified output unchanged; 0 scores all",
    )
    s.add_argument(
        "--v-max",
        dest="v_max",
        type=int,
        metavar="N",
        default=None,
        help="drafted tokens verified per step; 0 decodes one token at a time",
    )
    s.add_argument(
        "--ngram-p",
        dest="ngram_p",
        type=float,
        metavar="P",
        default=None,
        help="acceptance threshold of the n-gram drafter",
    )
    s.add_argument(
        "--no-spec",
        action="store_true",
        help="turn speculation off: decode one token at a time. The off switch; the flags above tune the tree",
    )
    s.add_argument(
        "--draft-model",
        dest="draft_model",
        metavar="PATH",
        default=None,
        help="what drafts for this model, its output verified so the answer is unchanged: 'auto' picks and "
        "downloads the curated small model for it (a one-time HF-cache download), a model directory or Hugging "
        "Face repo id is a draft model, and a single file is a custom MTP drafting head (unsupported - it must "
        "match this model). --draft-ks sets a draft model's branching, --draft-bits packs it. Without one the "
        "n-gram drafter proposes.",
    )
    s.add_argument(
        "--draft-ks",
        dest="draft_ks",
        type=_ints(1),
        metavar="K,K,...",
        default=None,
        help="the draft model's branching per tree depth (default 4,3,2): pass one reads the root's top-K0, "
        "each later pass the leaves' top-Kd",
    )
    d = ap.add_argument_group("sampling (how every token is picked; on the servers a request's own fields win)")
    d.add_argument(
        "--temperature",
        type=float,
        metavar="T",
        default=None,
        help="0 (the default) takes the likeliest token; above it the logits are scaled by T and a token drawn",
    )
    d.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        metavar="P",
        default=None,
        help="draw from the fewest likeliest tokens whose probability reaches P (default 1: every token)",
    )
    d.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        metavar="K",
        default=None,
        help="draw from the K likeliest tokens (default 0: every token)",
    )
    d.add_argument(
        "--seed",
        type=int,
        metavar="N",
        default=None,
        help="the draws' seed: a prompt and a seed repeat their answer (default: drawn per call, in the stats)",
    )
    d.add_argument(
        "--draft-temperature",
        dest="draft_temp_ratio",
        type=float,
        metavar="R",
        default=None,
        help="under a temperature the drafting head draws its tree at R times it (default 1): the verified answer's "
        "distribution is the same at any R, the accepted drafts a pass are not",
    )


def cmd_pack(a: argparse.Namespace) -> None:
    from . import resolve
    from .engine import pack_model

    _confirm_or_exit(a.path)
    out = pack_model(resolve(a.path), a.out, log=_p)
    _p(f"[pack] done -> {out}")


def cmd_devices(a: argparse.Namespace) -> None:
    from . import available_devices
    from .feedback import device_detail

    devs = available_devices()
    if a.as_json:
        _p(json.dumps(devs, ensure_ascii=False))
        return
    if a.quiet:
        for d in devs:
            _p(d["name"])
        return
    w = max((len(d["name"]) for d in devs), default=0)
    for d in devs:
        info = device_detail(d)
        eng = d["details"].get("backends") or []
        if eng:
            info += f" ({', '.join(eng)})"
        _p(f"{d['name']:<{w}}  {d['kind']:<3}  {info}")


DEFAULT_PROMPT = "What is the capital of France? Answer in one word."


def _run_prompts(a: argparse.Namespace) -> list[str]:
    """The prompts `run` answers: a JSONL --file's, or the one message from -p, else the prompt piped in on
    stdin, else a sample."""
    if a.file:
        out = []
        with open(a.file, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    out.append(str(rec["prompt"]))
                except (ValueError, KeyError, TypeError) as e:
                    raise BadValue("file", f'line {i}: {{"prompt": ...}} expected ({e})', a.file) from None
        if not out:
            raise BadValue("file", "no prompts in it", a.file)
        return out
    prompt = a.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip() or None  # the prompt piped in
    return [prompt if prompt is not None else DEFAULT_PROMPT]


def cmd_run(a: argparse.Namespace) -> None:
    from . import answer

    sm, tok = _open(a)
    es = sm.stop_ids
    eset = set(es)
    prompts = _run_prompts(a)
    fh = open(a.out, "a", encoding="utf-8") if a.out else None
    try:
        for idx, prompt in enumerate(prompts):
            ids = sm.prompt_ids(prompt)
            t0 = time.perf_counter()
            out, c = sm.generate(ids, a.new, eos=es, speculate=not a.no_spec)
            s = time.perf_counter() - t0
            stop = "eos" if (out and out[-1] in eset) else "length"
            tpp = round(len(out) / c["forwards"], 3) if (c and "forwards" in c) else 1.0
            if a.raw:
                text, think = tok.decode(out, skip_special_tokens=False), ""
            else:
                text, think = answer(tok, [t for t in out if t not in eset])
            rec: Json = {"answer": text}
            if think:
                rec["reasoning"] = think
            rec.update(
                {
                    "prompt": prompt,
                    "index": idx,
                    "took": round(s, 3),
                    "stop": stop,
                    "stats": {
                        "tokens_per_sec": round(len(out) / s, 1) if s > 0 else 0.0,
                        "tokens_in": len(ids),
                        "tokens_out": len(out),
                        "tokens_per_pass": tpp,
                    },
                }
            )
            if a.as_json:
                _p(json.dumps(rec, ensure_ascii=False))
            else:
                _p(rec["answer"])
                if not a.quiet:
                    _e(
                        f"[btb] {len(ids)} in, {len(out)} out, stop={stop}, {s:.1f}s = "
                        f"{s / max(1, len(out)):.2f} s/token"
                        + (f", {tpp:.2f} tokens/pass" if c and "forwards" in c else "")
                    )
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
    finally:
        if fh:
            fh.close()
    sm.close()


def cmd_chat(a: argparse.Namespace) -> None:
    sm, _tok = _open(a)
    ch = sm.chat(max_new=a.new)
    turns = None
    if a.file:
        turns = [json.loads(l) for l in open(a.file, encoding="utf-8") if l.strip()]
    k = 0
    while True:
        if turns is not None:
            if k >= len(turns):
                break
            t = turns[k]
            msg = t.get("user", "")
            if t.get("file"):
                txt = open(t["file"], encoding="utf-8").read()
                msg = t.get("prefix", "") + txt[: int(t.get("chars", 4000))] + t.get("suffix", "")
            _p(f"> {msg[:200]!r}{'...' if len(msg) > 200 else ''}")
        else:
            try:
                msg = input("> ")
            except EOFError:
                break
            if not msg.strip():
                continue
            if msg.strip() == "/reset":
                ch.reset()
                continue
            if msg.strip() in ("/quit", "/exit"):
                break
        t0 = time.perf_counter()
        for piece in ch.stream(msg):
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        s = time.perf_counter() - t0
        c = ch.last
        _p(
            f"[btb] turn {k + 1}: {c['prompt']} in (reused {c.get('reused', 0)}), {c['new']} out in {s:.1f}s = "
            f"{s / max(1, c['new']):.2f} s/token; {c['new'] / max(1, c.get('forwards', 1)):.2f} tokens/pass"
        )
        k += 1
    sm.close()


def cmd_bench(a: argparse.Namespace) -> None:
    from . import peak_memory
    from .sampling import GREEDY, Sampling

    sm, _tok = _open(a)
    rows = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
    pick = list(a.rows) if a.rows else list(range(len(rows)))
    bad = [i for i in pick if i >= len(rows)]
    if bad:
        raise BadValue("rows", f"{a.prompts} has {len(rows)} rows (0..{len(rows) - 1})", ",".join(str(i) for i in bad))
    counts = list(a.new)
    spec = sm.v_max > 0
    # both passes run at the cell's sampling (greedy unless --temperature was given), seeded so the baseline
    # and the speculative pass agree token for token and `identical` stays meaningful at any temperature
    samp = Sampling.from_request(
        {f: getattr(a, f, None) for f in ("temperature", "top_p", "top_k", "seed")}, GREEDY
    ).seeded()
    label = a.label or os.path.basename(os.path.normpath(a.path))
    cells = {}
    from .sysinfo import hard_page_faults, page_faults, process_read_bytes

    for n_new in counts:
        g_all, s_all, tpp, firsts, same = [], [], [], [], 0
        faults0, hard0, read0 = page_faults(), hard_page_faults(), process_read_bytes()
        for ri in pick:
            ids = sm.prompt_ids(rows[ri]["prompt"])
            marks: list[float] = []

            def mark(_t: int, m: list[float] = marks) -> None:
                m.append(time.perf_counter())

            t0 = time.perf_counter()
            g, _ = sm.generate(ids, n_new, eos=(), speculate=False, on_token=mark, sampling=samp)
            t1 = time.perf_counter()
            firsts.append(marks[0] - t0)
            tg = (t1 - marks[0]) / max(1, len(g) - 1)
            g_all.append(tg)
            line = f"[bench] {label} new={n_new} row {ri}: prompt {len(ids)}, first token {firsts[-1]:.2f}s, then {tg:.3f} s/token"
            if spec:
                # timed as every engine's baseline row is: the first token from the prompt, the rate after it
                marks_s: list[float] = []
                t0 = time.perf_counter()
                o, c = sm.generate(
                    ids, n_new, eos=(), on_token=lambda _t, m=marks_s: m.append(time.perf_counter()), sampling=samp
                )
                t1 = time.perf_counter()
                ts = (t1 - marks_s[0]) / max(1, len(o) - 1)
                s_all.append(ts)
                tpp.append(len(o) / max(1, c.get("forwards", 1)))
                same += int(o == g)
                line += f" | speculative {ts:.3f} s/token, {tpp[-1]:.2f} tokens/pass, identical {o == g}"
            # the process's peak so far on every row: a leak shows as a figure that climbs row after row
            line += f" | peak RSS {peak_memory(sm)[0] / 2**30:.1f} GB"
            _p(line)
        k = len(pick)
        rss, vram = peak_memory(sm)
        cell: Json = {
            "new": n_new,
            "first_s": sum(firsts) / k,
            "base_s_tok": sum(g_all) / k,
            "spec_s_tok": (sum(s_all) / k) if spec else None,
            "tokens_per_pass": (sum(tpp) / k) if spec else 1.0,
            "identical": f"{same}/{k}" if spec else "",
            "peak_ram_gb": rss / 2**30,
            "peak_vram_gb": vram / 2**30,
            "page_faults": page_faults() - faults0,
            "hard_faults": hard_page_faults() - hard0,
            "read_gb": (process_read_bytes() - read0) / 2**30,
        }
        cells[n_new] = cell
        _p(
            f"[bench] {label} new={n_new}: first token {cell['first_s']:.2f}s; baseline {cell['base_s_tok']:.3f} s/token "
            f"= {1 / cell['base_s_tok']:.2f} tok/s"
            + (
                f"; speculative {cell['spec_s_tok']:.3f} s/token = {1 / cell['spec_s_tok']:.2f} tok/s, "
                f"{cell['tokens_per_pass']:.2f} tokens/pass, identical {cell['identical']}"
                if spec
                else ""
            )
            + f"; peak RAM {cell['peak_ram_gb']:.1f} GB, peak VRAM {cell['peak_vram_gb']:.1f} GB, "
            f"page faults {cell['page_faults']} ({cell['hard_faults']} hard), read {cell['read_gb']:.1f} GB"
        )
    if a.out:
        # the engine's ledger beside the timings; never at the cost of the record itself
        try:
            rep = sm.report()
        except Exception as e:
            _p(f"[report] failed: {e!r}")
            rep = None
        from .hf import cache_repo_id

        with open(a.out, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "label": label,
                        "path": a.path,
                        "model": cache_repo_id(a.path) or os.path.basename(os.path.normpath(a.path)),
                        "report": rep,
                        "cells": list(cells.values()),
                    }
                )
                + "\n"
            )
    sm.close()


def cmd_serve(a: argparse.Namespace) -> Any:
    from .serve import serve

    _confirm_or_exit(a.path)
    if not a.draft_model:
        from .hf import draft_notice

        draft_notice(a.path)
    kw, nat = _kw(a)
    return serve(
        a.path,
        host=a.host,
        port=a.port,
        device=a.device,
        max_new=a.new,
        log=(_p if a.verbose else None),
        native=nat,
        extra_paths=a.models_dir or (),
        pattern=a.models_filter,
        api_key=a.api_key,
        **kw,
    )


def cmd_ollama(a: argparse.Namespace) -> Any:
    from .serve import ollama

    _confirm_or_exit(a.path)
    kw, nat = _kw(a)
    return ollama(
        a.path,
        host=a.host,
        port=a.port,
        device=a.device,
        max_new=a.new,
        log=(_p if a.verbose else None),
        native=nat,
        args=a.args,
        extra_paths=a.models_dir or (),
        pattern=a.models_filter,
        gui=a.gui,
        api_key=a.api_key,
        **kw,
    )


def cmd_pi(a: argparse.Namespace) -> Any:
    from .serve import pi

    _confirm_or_exit(a.path)
    kw, nat = _kw(a)
    return pi(
        a.path,
        host=a.host,
        port=a.port,
        device=a.device,
        max_new=a.new,
        log=(_p if a.verbose else None),
        native=nat,
        extra_paths=a.models_dir or (),
        pattern=a.models_filter,
        configure=a.configure,
        config_path=a.pi_config,
        api_key=a.api_key,
        **kw,
    )


def _serve_discovery(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model discovery (serve and ollama list every complete model, loaded on request)")
    g.add_argument(
        "--models-dir",
        dest="models_dir",
        action="append",
        type=_dir,
        metavar="DIR",
        default=None,
        help="also offer the model(s) under DIR (a model directory, or a directory of them); repeatable. "
        "The Hugging Face cache is always scanned.",
    )
    g.add_argument(
        "--models-filter",
        dest="models_filter",
        type=_regex,
        metavar="REGEX",
        default=None,
        help="only offer models whose name or repo id matches REGEX (the launch model is always offered)",
    )


def _require_deps(device: DeviceName | None) -> None:
    """Fail early and clearly on a missing package (find_spec, so torch is not imported to check for it);
    `mlx` is required only when the device is asked for by name."""
    import importlib.util as u

    need = ["torch", "transformers", "huggingface_hub", "numpy", "safetensors"]
    missing = [m for m in need if u.find_spec(m) is None]
    if device is not None and device.kind is Device.MLX and u.find_spec("mlx") is None:
        missing.append("mlx")
    if not missing:
        return
    _p(f"[btb] missing required package(s): {', '.join(missing)}")
    raise SystemExit(2)


def title_name(model: str | None) -> str:
    """A model as a person would name it: a repo id as given (`Qwen/Qwen3-4B`), a Hugging Face cache snapshot
    by its repo id rather than the commit hash the directory is named for, any other directory by its name."""
    s = str(model or "")
    if not os.path.exists(s):
        return s
    from .hf import cache_repo_id

    return cache_repo_id(s) or os.path.basename(os.path.normpath(os.path.abspath(s)))


def set_title(model: str | None, mode: str) -> str | None:
    """The process title as ps/top show it: `btb [model: <name>] (<mode>)`; silent where setproctitle is absent."""
    title = f"btb [model: {title_name(model)}] ({mode})"
    try:
        import setproctitle

        setproctitle.setproctitle(title)
        return title
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/comm", "w") as f:
                f.write(title[:15])
            return title
        except Exception:
            pass
    return None


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_streams()
    ap = argparse.ArgumentParser(
        prog="btb",
        description="Run big models on small machines, exactly.",
        epilog=env_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=False, metavar="command")
    p = sub.add_parser(
        "run",
        help="answer one or more prompts",
        description="Answer a prompt (given with -p, piped in on stdin, or a sample), or a JSONL file of "
        "prompts each with a fresh context (model loaded once); --json / --out make it machine-readable "
        "for batch eval.",
    )
    _common(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="the user message (default: the prompt piped in on stdin, else a sample prompt)",
    )
    g.add_argument(
        "--file",
        type=_file,
        default=None,
        metavar="JSONL",
        help='run each {"prompt": ...} line as an independent prompt with a fresh context (model loaded once)',
    )
    p.add_argument(
        "--new",
        type=_positive,
        metavar="N",
        default=None,
        help="cap on new tokens (default: until the model ends its turn or the context is full)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="keep special tokens in the output (<think>, the turn-end, eos) instead of scrubbing them",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="nothing but the model's text on stdout; no banner, no [btb] line",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="one JSON record per prompt on stdout (text and counts; implies --quiet); "
        "on an error, null and a non-zero exit code",
    )
    p.add_argument("--out", default=None, metavar="PATH", help="also append each prompt's JSON record to this file")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser(
        "chat",
        help="multi-turn chat in the terminal",
        description="Chat in the terminal; each turn reuses the previous turns' cache. /reset clears, /quit exits.",
    )
    _common(p)
    p.add_argument(
        "--file", type=_file, default=None, metavar="JSONL", help='scripted turns, one {"user": ...} per line'
    )
    p.add_argument(
        "--new", type=_positive, metavar="N", default=None, help="cap on new tokens per turn (default: none)"
    )
    p.set_defaults(fn=cmd_chat)
    p = sub.add_parser(
        "bench",
        help="time a prompt set at several answer lengths",
        description="Time greedy (and speculative, when configured) decoding over a prompt set, "
        "reporting first-token latency, s/token, peak RAM and peak VRAM per answer length.",
    )
    _common(p)
    p.add_argument("--prompts", required=True, type=_file, metavar="JSONL", help='one {"prompt": ...} per line')
    p.add_argument("--rows", type=_ints(0), default=None, metavar="I,J,...", help="which rows to run (default: all)")
    p.add_argument(
        "--new",
        type=_ints(1),
        default=[64, 256, 1024],
        metavar="N,N,...",
        help="answer lengths to time (default: 64,256,1024)",
    )
    p.add_argument("--label", default=None, help="name for this configuration in the output")
    p.add_argument("--out", default=None, metavar="JSONL", help="append the cells as JSON lines")
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser(
        "serve",
        help="OpenAI- and Ollama-compatible server",
        description="Serve the OpenAI API (POST /v1/chat/completions, streaming and not; GET /v1/models) "
        "and the Ollama API (POST /api/chat, /api/generate, /api/show; GET /api/tags). "
        "An answer runs until the model ends its turn or the context window is full "
        "unless the request sets max_tokens (OpenAI) or options.num_predict (Ollama).",
    )
    _common(p, path_required=False)
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default 127.0.0.1; 0.0.0.0 for every interface, which needs --api-key)",
    )
    p.add_argument("--port", type=_port, default=8000, help="port to listen on (default 8000)")
    p.add_argument(
        "--new", type=_positive, metavar="N", default=None, help="server-side ceiling on new tokens per request"
    )
    p.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        default=os.environ.get("BTB_API_KEY") or None,
        help="require this key on every request (Authorization: Bearer KEY); default: BTB_API_KEY from the "
        "environment, else none - set one before binding beyond 127.0.0.1",
    )
    _serve_discovery(p)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser(
        "ollama",
        help="serve the model and open it in the Ollama CLI or desktop app",
        description="Start the server, point the Ollama CLI at it (OLLAMA_HOST) and run `ollama run <name>`; "
        "the server stops when the CLI exits. --gui opens the Ollama desktop app pointed at "
        "this server instead. Without the ollama command on PATH the server "
        "stays up and prints the line to run elsewhere.",
    )
    _common(p, path_required=False)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        default=os.environ.get("BTB_API_KEY") or None,
        help="require this key on every request (Authorization: Bearer KEY); default: BTB_API_KEY from the "
        "environment, else none - set one before binding beyond 127.0.0.1",
    )
    p.add_argument(
        "--port", type=_port, default=11435, help="port for the server (default 11435, clear of a local Ollama)"
    )
    p.add_argument(
        "--new", type=_positive, metavar="N", default=None, help="server-side ceiling on new tokens per request"
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="open the Ollama desktop app pointed at this server (OLLAMA_HOST) instead of the terminal chat",
    )
    _serve_discovery(p)
    p.epilog = "Anything btb does not recognize is passed to `ollama run` after the model name, e.g. a one-shot prompt in quotes."
    p.set_defaults(fn=cmd_ollama)
    p = sub.add_parser(
        "pi",
        help="serve the model and register it with the pi coding agent (pi.dev)",
        description="Start the OpenAI-compatible server (with tool calling) and register a 'btb' provider in "
        "pi's ~/.pi/agent/models.json pointing at it; then run `pi --provider btb --model <name>` in another "
        "terminal. The server stays up until Ctrl-C. --no-configure prints the provider entry instead of "
        "writing the file; --config writes it elsewhere.",
    )
    _common(p, path_required=False)
    p.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1)")
    p.add_argument("--port", type=_port, default=8000, help="port to listen on (default 8000)")
    p.add_argument(
        "--new", type=_positive, metavar="N", default=None, help="server-side ceiling on new tokens per request"
    )
    p.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        default=os.environ.get("BTB_API_KEY") or None,
        help="require this key on every request (Authorization: Bearer KEY); default: BTB_API_KEY from the "
        "environment, else none - set one before binding beyond 127.0.0.1",
    )
    p.add_argument(
        "--no-configure",
        dest="configure",
        action="store_false",
        help="print the provider entry instead of writing ~/.pi/agent/models.json",
    )
    p.add_argument(
        "--config",
        dest="pi_config",
        default=None,
        metavar="PATH",
        help="write the provider to this file instead of ~/.pi/agent/models.json",
    )
    _serve_discovery(p)
    p.set_defaults(fn=cmd_pi)
    p = sub.add_parser(
        "pack",
        help="write the 12-bit model for a model (optional, lossless)",
        description="Write the lossless 12-bit model beside the model: for a Hugging Face repo a `<repo>-pack12` "
        "entry in the cache, for a directory `<model>-pack12`. It is a model like any other (pass it as the "
        "path); its config names the parent, which stays where it is and is not modified.",
    )
    p.add_argument("path", help="model directory, or a Hugging Face repo id")
    p.add_argument("out", nargs="?", default=None, help="write here instead (a plain model directory)")
    p.add_argument(
        "--confirm",
        action="store_true",
        help="answer yes to prompts; required to fetch a repo in a non-interactive run",
    )
    p.set_defaults(fn=cmd_pack)
    p = sub.add_parser(
        "devices",
        help="list the compute devices on this machine",
        description="List the devices a model can be placed on - the CUDA card(s), Apple's MLX, and the CPU - "
        "with what the machine reports about each. Each device's name is what --device takes.",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="one device name per line, nothing else")
    p.add_argument("--json", dest="as_json", action="store_true", help="the devices as a JSON array")
    p.set_defaults(fn=cmd_devices)
    cmdline: list[str] = list(argv) if argv is not None else sys.argv[1:]
    a, rest = ap.parse_known_args(argv)
    if a.cmd is None:  # no subcommand: print the help, as `btb --help` does
        ap.print_help()
        return 0
    if a.cmd == "ollama":
        a.args = rest
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if a.cmd != "devices":  # devices reports the hardware; it needs no model stack
        _require_deps(getattr(a, "device", None))
    if getattr(a, "device", None) is not None and a.device.kind is Device.CPU:
        from . import cpu_only

        cpu_only()
    as_json = getattr(a, "as_json", False)
    if as_json:
        a.quiet = True
    quiet = as_json or getattr(a, "quiet", False)
    from .confirm import set_assume_yes

    set_assume_yes(getattr(a, "confirm", False))
    run = a.cmd == "run"  # run's stdout carries the model's text alone; the rest goes to stderr
    if a.cmd not in ("pack", "devices"):  # the model-loading commands: title and banner
        set_title(a.path or "(btb)", "openai" if a.cmd == "serve" else a.cmd)  # serve speaks the OpenAI API
        if not quiet:
            banner(err=run)
    if a.cmd not in ("pack", "devices") and getattr(a, "native", None) is None and not quiet:
        from . import native_path

        if native_path() is None:
            (_e if run else _p)("[btb] no native library found; torch fallback")
    if not as_json:  # stderr, past --quiet, but never in a JSON run's machine output
        _license_notice()
    from . import feedback

    err: BaseException | None = None
    try:
        rc = a.fn(a)
    except KeyboardInterrupt:
        return 130
    except OptionError as e:  # a value btb cannot take: the line, no traceback; an option named by its flag
        msg = str(e)
        if isinstance(e, BadValue):
            for act in sub.choices[a.cmd]._actions:
                if act.option_strings and act.dest == e.name:
                    msg = e.shown_as(act.option_strings[-1])
                    break
        _e(f"[btb] {msg}")
        if as_json:
            _p("null")
        return 2
    except FileNotFoundError as e:
        model = getattr(a, "path", None)
        path = str((e.args[0] if e.args else None) or getattr(e, "filename", None) or "")
        if model is None or path != str(model):
            raise  # a missing file deeper in a command is a real fault; keep its traceback
        err = e
        _e(f"[btb] {feedback.not_found_report(e, model)}")
        if as_json:
            _p("null")
        return 1
    except SystemExit:
        raise
    except BaseException as e:
        # any other failure: the traceback, then the report for an issue (with the profile written first, so the
        # report carries the expert-store summary where a trace was kept)
        err = e
        traceback.print_exc()
        sm = getattr(a, "_sm", None)
        summary = None
        if getattr(a, "profile", None) and sm is not None:
            feedback.write_profile(a, sm, _e)
            summary = feedback.expert_summary(a.profile)
        _e(feedback.crash_notice(e))
        _e(feedback.issue_report(e, a, sm, cmdline, getattr(a, "profile", None), summary))
        if as_json:
            _p("null")
        return 2 if feedback.is_oom(e) else 1
    finally:
        sm = getattr(a, "_sm", None)
        if err is None and getattr(a, "profile", None) and sm is not None:
            feedback.write_profile(a, sm, _e)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
