# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Console output for the CLI: stdout and stderr helpers, the banner, the license notice and the process title."""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

ART = "\n".join(
    [
        "",
        "               ██████",
        "           ████▓▓▓▒▒▒▒▒▒██████",
        "       ████▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒████▒▒",
        "     ▓▓▓▓██████▓▓▓▒      ████▒▒▒▒▒▒",
        "     ▓▓▓▓▓▓▓▓▓ ▓▓▓▒██████▒▒▒▒▒▒▓▓▒▒",
        "  █████████  ▓▓▓▓▓▒▓▓▓▓▒▒▒▒   ▓▓▓▒▒",
        "  ░░░░░░░░░████▓▓▓▒█▓▓▓▒▒     ▓▓▓▒▒",
        "     ▓▓▓▒▒ ░░░░▓▓▓▒░█████████ ▓▓▓▒▒",
        "     ▓▓▓▒▒     ▓▓▓▒ ░░░░░░░░░█▓▓▓▒▒███",
        "     ▓▓▓▒▒    █▓▓▓▒██████████░▓▓▓▒▒░░░",
        "     ▓▓▓▒█████▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▓▓▓▒▒",
        "     ▓▓▓█▓▓▓▓▓      ▓▓▓▒▒  ███▓▒▒▒▒",
        "     ▓▓▓████████████▓▓▓▒▒██▒▒▒▒▒▒",
        "          ▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒",
        "                   ▓▓▓▓▒▒",
        "",
        "          beyond the box (btb)",
        "",
    ]
)


def say(*a: Any) -> None:
    """Print a line to stdout and flush."""
    print(*a, flush=True)


def say_err(*a: Any) -> None:
    """Print a line to stderr and flush."""
    print(*a, file=sys.stderr, flush=True)


def utf8_streams() -> None:
    # the model's text is UTF-8, and a piped prompt may be too; a Windows console defaults to cp1252
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def banner(err: bool = False) -> None:
    with contextlib.suppress(Exception):
        print(ART, file=sys.stderr if err else sys.stdout, flush=True)


def license_notice() -> None:
    """Print the FSL commercial-use notice to stderr while this build is inside its two-year window (btb.fsl);
    silent once the window has passed, when the code is Apache 2.0, and for anyone with `~/.config/btb.friend`"""
    from ..fsl import restricted

    if restricted():
        say_err("btb is free for personal and academic use. For commercial use, consider a paid license.")


def title_name(model: str | None) -> str:
    """A model as a person would name it: a repo id as given (`Qwen/Qwen3-4B`), a Hugging Face cache snapshot
    by its repo id rather than the commit hash the directory is named for, any other directory by its name."""
    s = str(model or "")
    if not os.path.exists(s):
        return s
    from ..hf import cache_repo_id

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
