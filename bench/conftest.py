# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""pytest-benchmark hooks for bench/e2e_bench.py."""

from __future__ import annotations

import pytest


def pytest_benchmark_update_machine_info(config: pytest.Config, machine_info: dict[str, object]) -> None:
    """the ISA tier the native/cpu paths ran at, into the run's JSON, for the report to state"""
    from btb.engine.native import isa

    machine_info["isa"] = isa()
