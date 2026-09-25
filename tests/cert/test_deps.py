"""transformers version-drift guard (cert-coverage-gaps #29). The floor `transformers>=5.16,<6` lets any minor
inside the major install silently, and minor releases are where transformers changes model/family code - the
qwen4 router-dtype failure IS a transformers-version issue. So the cert records the major.minor it was last
validated against and this test fails when the installed version differs, forcing a dep bump to be a deliberate,
re-certified event instead of silent drift.

Granularity is major.minor on purpose: a patch bump (bugfixes, no model-code change) would be noise, but a minor
or major bump is exactly where family behavior moves and MUST re-run the cert. Torch-free - it reads the installed
distribution's metadata, never imports transformers - so it fits cert.yml's torch-free gate.
"""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

# The transformers major.minor the family cert was last validated against. Bump ONLY after re-running the cert
# on the new version (see the failure message). A patch difference (5.17.0 -> 5.17.3) is not a change here.
CERT_TRANSFORMERS: tuple[int, int] = (5, 17)


def _installed() -> tuple[int, int] | None:
    """the installed transformers major.minor from its distribution metadata, or None when it is not installed
    (a torch-free checkout with nothing to guard)."""
    try:
        raw = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return None
    parts = raw.split(".")
    return int(parts[0]), int(parts[1])


def _pyproject_bounds() -> tuple[tuple[int, int], tuple[int, int] | None]:
    """the (floor, ceiling) major.minor declared for transformers in pyproject, so the recorded constant can be
    cross-checked against the package's own requirement rather than trusting it blind. Ceiling is None when only
    a floor is pinned."""
    root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = [r for r in data["project"]["dependencies"] if r.replace(" ", "").startswith("transformers")]
    assert reqs, "no transformers requirement in pyproject dependencies"
    spec = reqs[0]
    floor = re.search(r">=\s*(\d+)(?:\.(\d+))?", spec)
    assert floor, f"no >= floor in {spec!r}"
    lo = (int(floor.group(1)), int(floor.group(2) or 0))
    ceil = re.search(r"<\s*(\d+)(?:\.(\d+))?", spec)
    hi = (int(ceil.group(1)), int(ceil.group(2) or 0)) if ceil else None
    return lo, hi


def test_transformers_matches_cert() -> None:
    """the drift gate: the installed transformers must be the major.minor the cert was validated against."""
    installed = _installed()
    if installed is None:
        pytest.skip("transformers not installed; nothing to guard (the engine cert installs it)")
    assert installed == CERT_TRANSFORMERS, (
        f"transformers {installed[0]}.{installed[1]} differs from the cert-validated "
        f"{CERT_TRANSFORMERS[0]}.{CERT_TRANSFORMERS[1]}. A minor/major bump can change family/model behavior "
        f"(e.g. the qwen4 router dtype). Re-run the family cert on this version, then set CERT_TRANSFORMERS in "
        f"tests/cert/test_deps.py to {installed}."
    )


def test_recorded_version_is_within_pyproject_bounds() -> None:
    """keeps the recorded constant and the pyproject pin from drifting apart: the cert-validated version must sit
    at or above the floor and below the ceiling the package declares."""
    lo, hi = _pyproject_bounds()
    assert CERT_TRANSFORMERS >= lo, f"CERT_TRANSFORMERS {CERT_TRANSFORMERS} is below the pyproject floor {lo}"
    if hi is not None:
        assert CERT_TRANSFORMERS < hi, f"CERT_TRANSFORMERS {CERT_TRANSFORMERS} is not below the pyproject ceiling {hi}"
