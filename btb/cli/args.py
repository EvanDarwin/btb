# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Argument types for the CLI, and the options shared by every command that loads a model."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable
from typing import Any

from ..kinds import Json
from ..options import DeviceName

# options passed straight through to `load` when given
PLACEMENT = (
    "cpu_layers",
    "resident_last",
    "tree_budget",
    "v_max",
    "fp32",
    "resident_head",
    "kv_host",
    "context",
    "kv_bits",
    "tree_min_prob",
    "mlx_mega",
    "sparse",
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
    "adapt",
    "cold_slots",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "draft_temp_ratio",
)


def device(s: str) -> DeviceName | None:
    """--device as typed: cpu, mlx, cuda or cuda:N (mlx off Apple silicon refused here, before any import)"""
    from ..options import OptionError, check_device

    try:
        return check_device(s)
    except OptionError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def positive(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a whole number above 0") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{s!r}: a whole number above 0")
    return n


def port(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a port, 1 to 65535") from None
    if not 1 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"{s!r}: a port, 1 to 65535")
    return n


def fraction(s: str) -> float:
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: a share between 0 and 1, such as 0.02") from None
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError(f"{s!r}: a share between 0 and 1, such as 0.02")
    return x


def seconds(s: str) -> float:
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r}: seconds above 0") from None
    if not x > 0:
        raise argparse.ArgumentTypeError(f"{s!r}: seconds above 0")
    return x


def ints(least: int) -> Callable[[str], list[int]]:
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


def existing_file(s: str) -> str:
    if not os.path.isfile(s):
        raise argparse.ArgumentTypeError(f"{s!r}: no such file")
    return s


def existing_dir(s: str) -> str:
    if not os.path.isdir(s):
        raise argparse.ArgumentTypeError(f"{s!r}: no such directory")
    return s


def regex(s: str) -> str:
    try:
        re.compile(s)
    except re.error as e:
        raise argparse.ArgumentTypeError(f"{s!r}: not a regular expression ({e})") from None
    return s


def load_args(a: argparse.Namespace) -> tuple[Json, Any]:
    """Return the PLACEMENT options that were given, and the --native library ('' for none, None to search)."""
    kw = {}
    for k in PLACEMENT:
        v = getattr(a, k, None)
        if v is not None:
            kw[k] = v
    nat = a.native
    if nat is not None and nat.strip().lower() in ("", "none", "off"):
        nat = ""
    return kw, nat


def model_options(ap: argparse.ArgumentParser, path_required: bool = True, path_help: str | None = None) -> None:
    """Add the model path and its load options: device, placement, scheduler, speculation and sampling."""
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
            help=path_help
            or "model directory, Hugging Face repo id, 12-bit model or GGUF file (.gguf, or repo/id:file.gguf) "
            "to load at startup; omit to start with "
            "nothing loaded and serve whatever is already on the machine, on request",
        )
    ap.add_argument(
        "-d",
        "--device",
        type=device,
        default=None,
        help="cuda, cuda:N (a specific card), mlx or cpu; a device named must be here (mlx off Apple silicon, "
        "cuda without a card: an error). Default: the card, else mlx on Apple silicon, else cpu",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print the placement, the tiers and each turn's timings; -vv also traces every automatic decision as it "
        "is taken - a layer moved between the card, RAM and the drive, memory given back to another program, the path "
        "a pass takes, the tuner's lane - on stderr, never holding the passes up (BTB_TRACE=1 in a library)",
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
        "--adapt",
        dest="adapt",
        type=int,
        choices=(0, 1),
        default=1,
        help="give layers up (off the card, from RAM to the drive) when another program needs the memory, and take "
        "them back when it frees (default 1); --adapt 0 pins the placement taken at load",
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
        "--sparse",
        dest="sparse",
        type=int,
        choices=(0, 1),
        default=None,
        help="Qwen4's sparse attention scores every row's blocks in one pass: faster prefill, but a near-tie at the "
        "budget's edge may keep another block than the reference's indexer (not the certified path); 0 by default",
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
        type=ints(1),
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
