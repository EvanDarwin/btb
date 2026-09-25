# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
What bench/matrix.py and bench/compare.py share: the paths, one progress line, one subprocess call.
The rest is by phase: records (the shapes), host (the machine), env (the comparison environment),
tools (the rival engines), plan, run, table.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
QUESTIONS = os.path.join(HERE, "questions.jsonl")
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
COMPARE_VENV = os.path.join(ROOT, ".venv-compare")
COMPARE_PY = os.path.join(
    COMPARE_VENV,
    "Scripts" if sys.platform == "win32" else "bin",
    "python.exe" if sys.platform == "win32" else "python",
)


def say(msg: str = "", tag: str = "matrix") -> None:
    """
    One line of progress, tagged and flushed (a cell's own output goes to its log)
    """
    print(f"[{tag}] {msg}" if msg else "", flush=True)


def capture(cmd: Sequence[str], timeout: float = 20) -> str:
    """
    A command's stdout, stripped; "" when it fails or is not there
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False).stdout.strip()
    except Exception:
        return ""


def git_commit() -> str:
    return capture(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"])
