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
