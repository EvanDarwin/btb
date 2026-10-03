# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""pytest-benchmark hooks for bench/e2e_bench.py, and the placement group's routing replay's options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("btb placement")
    g.addoption(
        "--routing",
        default=None,
        metavar="EVENTS",
        help="a `btb ... --profile DIR` run's DIR/events.npz: the placement group replays its expert calls, each "
        "with its recorded picks, over stand-in weights of --routing-expert's shape",
    )
    g.addoption(
        "--routing-expert",
        default="2560x640",
        metavar="HxI",
        help="the recorded model's hidden width and expert intermediate width (gate_up is 2 * I wide)",
    )


def pytest_benchmark_update_machine_info(config: pytest.Config, machine_info: dict[str, object]) -> None:
    """the ISA tier the native/cpu paths ran at, into the run's JSON, for the report to state"""
    from btb.engine.native import isa

    machine_info["isa"] = isa()
