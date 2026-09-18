# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A yes/no prompt that waits for the user before an action worth stopping over - a large download, an
overwrite. It answers itself where a person cannot: --confirm (or BTB_CONFIRM=1) says yes without asking, and
a non-interactive run with neither says no rather than blocking on input no one is there to type. Torch-free."""

from __future__ import annotations

import os
import sys

_ASSUME_YES = False


def set_assume_yes(yes: bool) -> None:
    """Answer every later confirm() yes without asking - the CLI wires this to --confirm."""
    global _ASSUME_YES
    _ASSUME_YES = bool(yes)


def _say(line: str) -> None:
    sys.stderr.write(line)
    sys.stderr.flush()


def confirm(question: str, *, default: bool = False) -> bool:
    """Ask `question` on the terminal and wait for y/n, an empty line taking `default`. Return True at once
    under --confirm or BTB_CONFIRM=1. In a non-interactive run with neither, return False after a one-line
    note on stderr that --confirm would allow it, so a piped or scripted call fails closed instead of hanging."""
    if _ASSUME_YES or os.environ.get("BTB_CONFIRM") == "1":
        return True
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        _say(f"[btb] {question}  pass --confirm to allow this without a terminal\n")
        return False
    prompt = f"{question} [{'Y/n' if default else 'y/N'}] "
    while True:
        _say(prompt)
        try:
            reply = input().strip().lower()
        except EOFError:
            return False
        if not reply:
            return default
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        _say("[btb] please answer y or n\n")


def select(question: str, options: list[str]) -> int | None:
    """Ask the user to pick one of `options` on the terminal, returning its index, or None where a person cannot
    be asked - a non-interactive run, an empty line, or EOF. A list has no safe default the way a yes/no does, so
    --confirm / BTB_CONFIRM=1 does not answer it either: an unattended run bails rather than guess a choice."""
    if not options:
        return None
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return None
    _say(question + "\n")
    for i, opt in enumerate(options, 1):
        _say(f"  {i:>3}. {opt}\n")
    while True:
        _say(f"[1-{len(options)}, or blank to cancel] ")
        try:
            reply = input().strip()
        except EOFError:
            return None
        if not reply:
            return None
        if reply.isdigit() and 1 <= int(reply) <= len(options):
            return int(reply) - 1
        _say(f"[btb] please enter a number from 1 to {len(options)}\n")
