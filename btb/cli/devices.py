# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb devices`: list the devices a model can run on, by the names --device takes."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .console import say


def add(sub: Any) -> None:
    p = sub.add_parser(
        "devices",
        help="list the compute devices on this machine",
        description="List the devices a model can be placed on - the CUDA card(s), Apple's MLX, and the CPU - "
        "with what the machine reports about each. Each device's name is what --device takes.",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="one device name per line, nothing else")
    p.add_argument("--json", dest="as_json", action="store_true", help="the devices as a JSON array")
    p.set_defaults(fn=cmd_devices)


def cmd_devices(a: argparse.Namespace) -> None:
    from .. import available_devices
    from ..feedback import device_detail

    devs = available_devices()
    if a.as_json:
        say(json.dumps(devs, ensure_ascii=False))
        return
    if a.quiet:
        for d in devs:
            say(d["name"])
        return
    w = max((len(d["name"]) for d in devs), default=0)
    for d in devs:
        info = device_detail(d)
        eng = d["details"].get("backends") or []
        if eng:
            info += f" ({', '.join(eng)})"
        say(f"{d['name']:<{w}}  {d['kind']:<3}  {info}")
