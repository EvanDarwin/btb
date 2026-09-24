# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The FSL window: two years from the day this wheel was built, after which the code is Apache 2.0. The CLI's
commercial-use notice keys off it."""

from __future__ import annotations

import datetime
import os

FRIEND_FILE = os.path.expanduser("~/.config/btb.friend")


def build_date() -> datetime.date | None:
    """The day this wheel was built, baked into btb/_build.py at build time (setup.py). None in a source
    checkout, where the module is never generated - the window then can't be measured, so it counts as open."""
    try:
        from importlib import import_module

        return datetime.date.fromisoformat(import_module("btb._build").BUILD_DATE)
    except Exception:
        return None


def restricted() -> bool:
    """whether this build is inside its FSL window: before the change date two years after the build, and not
    for anyone who has `~/.config/btb.friend`"""
    if os.path.exists(FRIEND_FILE):
        return False
    built = build_date()
    if built is None:
        return True
    try:
        change = built.replace(year=built.year + 2)
    except ValueError:  # a Feb 29 build
        change = built.replace(year=built.year + 2, day=28)
    return datetime.date.today() < change
