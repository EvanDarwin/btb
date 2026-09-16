"""Fixtures shared across the suite."""

from collections.abc import Iterator

import pytest

import btb


@pytest.fixture(autouse=True)
def _keep_the_cards_visibility() -> Iterator[None]:
    """`load(device="cpu")` / `run -d cpu` calls `cpu_only()`, which clears the package's CUDA flag and decides
    where every later load goes. Each test leaves it as it found it, so the suite reads the same in any order."""
    was = btb.CUDA
    yield
    btb.CUDA = was
