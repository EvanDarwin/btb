# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb chat`: a multi-turn chat in the terminal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from . import args
from .console import say
from .model import open_model


def add(sub: Any) -> None:
    p = sub.add_parser(
        "chat",
        help="multi-turn chat in the terminal",
        description="Chat in the terminal; each turn reuses the previous turns' cache. /reset clears, /quit exits.",
    )
    args.model_options(p)
    p.add_argument(
        "--file",
        type=args.existing_file,
        default=None,
        metavar="JSONL",
        help='scripted turns, one {"user": ...} per line',
    )
    p.add_argument(
        "--new", type=args.positive, metavar="N", default=None, help="cap on new tokens per turn (default: none)"
    )
    p.set_defaults(fn=cmd_chat)


def _scripted(t: dict[str, Any]) -> str:
    """The message for a scripted turn: `user`, or the first `chars` of `file` wrapped in `prefix` and `suffix`."""
    if not t.get("file"):
        return str(t.get("user", ""))
    with open(t["file"], encoding="utf-8") as f:
        txt = f.read()
    return t.get("prefix", "") + txt[: int(t.get("chars", 4000))] + t.get("suffix", "")


def cmd_chat(a: argparse.Namespace) -> None:
    sm, _tok = open_model(a)
    ch = sm.chat(max_new=a.new)
    turns = None
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            turns = [json.loads(line) for line in f if line.strip()]
    k = 0
    while True:
        if turns is not None:
            if k >= len(turns):
                break
            msg = _scripted(turns[k])
            say(f"> {msg[:200]!r}{'...' if len(msg) > 200 else ''}")
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
        say(
            f"[btb] turn {k + 1}: {c['prompt']} in (reused {c.get('reused', 0)}), {c['new']} out in {s:.1f}s = "
            f"{s / max(1, c['new']):.2f} s/token; {c['new'] / max(1, c.get('forwards', 1)):.2f} tokens/pass"
        )
        k += 1
    sm.close()
