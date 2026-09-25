# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""pytest-benchmark hooks for bench/e2e_bench.py."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import btb


@pytest.fixture(autouse=True)
def _keep_the_cards_visibility() -> Iterator[None]:
    """a cpu `load()` calls `cpu_only()`, which clears the package's CUDA flag, and every cuda cell after it was
    refused (BadDevice) - as tests/conftest.py does, each bench leaves the flag as it found it"""
    was = btb.CUDA
    yield
    btb.CUDA = was


def pytest_benchmark_update_machine_info(config: pytest.Config, machine_info: dict[str, object]) -> None:
    """the ISA tier the native/cpu paths ran at, into the run's JSON, for the report to state"""
    from btb.engine.native import isa

    machine_info["isa"] = isa()
