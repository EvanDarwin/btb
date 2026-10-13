# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb run`: answer one prompt, or each prompt in a JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from ..kinds import Json
from ..options import BadValue
from . import args
from .console import say, say_err
from .model import open_model

DEFAULT_PROMPT = "What is the capital of France? Answer in one word."


def add(sub: Any) -> None:
    p = sub.add_parser(
        "run",
        help="answer one or more prompts",
        description="Answer a prompt (given with -p, piped in on stdin, or a sample), or a JSONL file of "
        "prompts each with a fresh context (model loaded once); --json / --out make it machine-readable "
        "for batch eval.",
    )
    args.model_options(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="the user message (default: the prompt piped in on stdin, else a sample prompt)",
    )
    g.add_argument(
        "--file",
        type=args.existing_file,
        default=None,
        metavar="JSONL",
        help='run each {"prompt": ...} line as an independent prompt with a fresh context (model loaded once)',
    )
    p.add_argument(
        "--new",
        type=args.positive,
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


def prompts(a: argparse.Namespace) -> list[str]:
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
    from .. import answer

    sm, tok = open_model(a)
    es = sm.stop_ids
    eset = set(es)
    todo = prompts(a)
    fh = open(a.out, "a", encoding="utf-8") if a.out else None
    try:
        for idx, prompt in enumerate(todo):
            ids = sm.prompt_ids(prompt)
            t0 = time.perf_counter()
            gen = sm.generate(ids, a.new, eos=es, speculate=not a.no_spec)
            out, c = gen.tokens, gen.stats
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
                say(json.dumps(rec, ensure_ascii=False))
            else:
                say(rec["answer"])
                if not a.quiet:
                    say_err(
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
