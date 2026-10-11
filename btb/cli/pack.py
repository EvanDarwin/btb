# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb pack`: write a lossless 12-bit copy of a model."""

from __future__ import annotations

import argparse
from typing import Any

from .console import say
from .model import confirm_or_exit


def add(sub: Any) -> None:
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


def cmd_pack(a: argparse.Namespace) -> None:
    from .. import resolve
    from ..engine import pack_model

    confirm_or_exit(a.path)
    written = pack_model(resolve(a.path), a.out, log=say)
    say(f"[pack] done -> {written}")
