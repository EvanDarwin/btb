"""Fixtures shared across the suite, the prereq warning, and the skip ledger a full run banks."""

from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

import btb

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.reports import TestReport

# every skip of a session, collected across phases and written at the end; a full `tests` run banks the ledger
_SKIPS: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _keep_the_cards_visibility() -> Iterator[None]:
    """`load(device="cpu")` / `run -d cpu` calls `cpu_only()`, which clears the package's CUDA flag and decides
    where every later load goes. Each test leaves it as it found it, so the suite reads the same in any order."""
    was = btb.CUDA
    yield
    btb.CUDA = was


def _release_cuda_cache() -> None:
    """the engine's own vram_trim idiom (synchronize, then empty_cache), when a test has loaded torch at all - the
    torch-free cert gate (cert.yml) runs this conftest without it"""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def _release_allocator_caches() -> Iterator[None]:
    """Every allocator keeps freed buffers cached, and the scheduler's ledger counts them as held, so a long session
    starves later loads whatever the backend - the draft-model test refused a 16 MB KV grow at the end of the cert
    runner with "0 KiB free, 7.49 GiB held by MLX". After each test: drop the cyclic garbage that pins tensors, then
    return the CUDA and MLX caches, so memory reads the same in any order."""
    yield
    gc.collect()
    _release_cuda_cache()
    mx = sys.modules.get("mlx.core")  # only a test that loaded MLX has an MLX cache to return
    if mx is not None:
        mx.clear_cache()


def pytest_configure(config: Config) -> None:
    _require_native()
    _warn_missing_prereqs(config)


def _require_native() -> None:
    """no run without the native library: its kernel, store and receipt tests would take the torch path or skip, and
    the run would pass having certified none of them. A file check, so the torch-free cert gate can run it."""
    if btb.native_path() is None:
        raise pytest.UsageError("no native library built for this machine: `python build.py build --no-wheel`")


def pytest_report_header() -> str | None:
    """print the cert coverage summary at the top of a run, so fixture gaps and missing cached models (and the
    tests they mute) are visible before anything skips - never fails a run if the check itself cannot import.
    The repo forces -q (which hides this), so `_warn_missing_prereqs` also surfaces the missing case under -q."""
    try:
        from tests.cert.manifest import summary_line

        return summary_line()
    except Exception:
        return None


def _full_run(config: Config) -> bool:
    """True for a whole-directory run (not a single .py file or nodeid): the same gate the ledger uses, so a
    focused TDD run pays neither the prereq check nor the ledger write."""
    return all(not arg.split("::", 1)[0].endswith(".py") for arg in config.args or ())


def _warn_missing_prereqs(config: Config) -> None:
    """On a full run, print one visible line to stderr when real models are missing - this shows even under the
    repo's forced -q, which hides pytest_report_header. Silent when all are cached (no noise)."""
    if not _full_run(config):
        return
    try:
        from tests.cert.manifest import target_status

        missing = [t for t, ok in target_status() if not ok]
    except Exception:
        return
    if missing:
        sys.stderr.write(
            f"\n[prereqs] {len(missing)} real model(s) not cached; {len(missing)} test group(s) will skip. "
            "Details: python -m tests.cert.manifest --report\n\n"
        )


def pytest_runtest_logreport(report: TestReport) -> None:
    """note every skipped test and its reason once, for the ledger written at session end"""
    if report.skipped and report.nodeid not in _SKIPS:
        lr = report.longrepr
        reason = str(lr[2]) if isinstance(lr, tuple) and len(lr) == 3 else str(lr)
        _SKIPS[report.nodeid] = reason.removeprefix("Skipped: ").strip()


def _ledger_path(config: Config) -> str | None:
    """where the ledger goes: `BTB_SKIP_LEDGER` when set (always), else `.pytest_cache/btb-skip-ledger.json`
    (already gitignored) but only for a full `tests` run, so a single-file run does not clobber the baseline"""
    env = os.environ.get("BTB_SKIP_LEDGER")
    if env:
        return env
    if not _full_run(config):
        return None  # a single-file (or single-test) run: leave the baseline alone
    return os.path.join(str(config.rootpath), ".pytest_cache", "btb-skip-ledger.json")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """write the collected skips to the ledger (see `_ledger_path`), and, when BTB_CERT_RECEIPT names a path,
    this machine's cert coverage receipt (what the cert runner ran and passed here) for the cross-machine union"""
    receipt_path = os.environ.get("BTB_CERT_RECEIPT")
    if receipt_path:
        from tests.cert import receipt

        receipt.emit(receipt_path)
    path = _ledger_path(session.config)
    if path is None:
        return
    ledger = {
        "count": len(_SKIPS),
        "skips": [{"nodeid": nid, "reason": _SKIPS[nid]} for nid in sorted(_SKIPS)],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")
