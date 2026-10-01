"""Fixtures shared across the suite, the prereq warning, and the skip ledger a full run banks."""

from __future__ import annotations

import gc
import json
import os
import sys
from collections.abc import Generator, Iterator
from typing import TYPE_CHECKING

import pytest

import btb

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.reports import TestReport

# every skip of a session, collected across phases and written at the end; a full `tests` run banks the ledger
_SKIPS: dict[str, str] = {}

# every engine a test builds is checked for what it keeps once closed (btb/engine/leaks.py): armed before any is built
os.environ["BTB_LEAK_HARNESS"] = "1"


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
    return the CUDA and MLX caches, so memory reads the same in any order. First, before the collection that would
    free it, what a closed engine kept - still reached from it or btb, or held only by a cycle - fails the test at
    its teardown, apart from the test's own outcome."""
    yield
    leaks = sys.modules.get("btb.engine.leaks")  # only a test that built an engine has one to check
    found = leaks.verify() if leaks is not None else []
    helpers = sys.modules.get("tests.helpers")
    if leaks is not None and helpers is not None and helpers._SHARED:
        # a model shared with the next test is still alive, so the census waits for it; what only a cycle keeps is
        # found now, before the collection below frees it unseen
        found += leaks.cycles()
    gc.collect()
    _release_cuda_cache()
    mx = sys.modules.get("mlx.core")  # only a test that loaded MLX has an MLX cache to return
    if mx is not None:
        mx.clear_cache()
    if found:
        pytest.fail("a closed engine kept memory:\n" + "\n".join(found), pytrace=False)


def _shared_keys(item: pytest.Item | None) -> frozenset[tuple[object, ...]]:
    """the shared models a test declares (its module's `shared_model_keys`, tests.helpers.shared_model); none for a
    test that is skipped before it runs"""
    if item is None or item.get_closest_marker("skip") is not None:
        return frozenset()
    fn = getattr(getattr(item, "module", None), "shared_model_keys", None)
    return frozenset(fn(item)) if fn is not None else frozenset()


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    """the tests of a module that share a model run one after another, so it is loaded once for them all
    (tests.helpers.shared_model): within each module, and only among its tests that declare one - every other test
    keeps its place, so a module's fixtures and the tests that read what earlier ones recorded are left as they were"""
    start = 0
    while start < len(items):
        mod = getattr(items[start], "module", None)
        end = start
        while end < len(items) and getattr(items[end], "module", None) is mod:
            end += 1
        slots = [i for i in range(start, end) if _shared_keys(items[i])]
        if len(slots) > 1:
            ordered = sorted((items[i] for i in slots), key=lambda it: sorted(map(repr, _shared_keys(it))))
            for i, it in zip(slots, ordered, strict=True):
                items[i] = it
        start = end


# a test whose call failed: the shared models it used are let go at its teardown, never handed to the next test in
# whatever state the failure left them (a pass cut short, the card graph turned off, a session still bound)
_FAILED = pytest.StashKey[bool]()
# the first test after each that is not skipped before it runs, by the test's id (`_next_to_run`)
_NEXT_RUN: dict[int, pytest.Item | None] = {}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, TestReport, TestReport]:
    report = yield
    if report.when == "call" and report.failed:
        item.stash[_FAILED] = True
    return report


def _next_to_run(item: pytest.Item, nextitem: pytest.Item | None) -> pytest.Item | None:
    """the next test that runs: `nextitem`, or past it the first not skipped before it runs - a skipped cell between
    two tests sharing a model otherwise let the model go and the second loaded it again"""
    if nextitem is None or nextitem.get_closest_marker("skip") is None:
        return nextitem
    if not _NEXT_RUN:
        after: pytest.Item | None = None
        for it in reversed(item.session.items):
            _NEXT_RUN[id(it)] = after
            if it.get_closest_marker("skip") is None:
                after = it
    return _NEXT_RUN.get(id(nextitem))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """a shared model the next test to run does not declare closed now, before this test's fixtures are torn down -
    so the leak check of `_release_allocator_caches` covers it at this test; every one this test used, where it
    failed. A model that would not close is this test's teardown error, raised after pytest's own teardown has run
    (raised before it, the test's fixtures were never torn down and the next test failed its setup)"""
    helpers = sys.modules.get("tests.helpers")
    errors: list[BaseException] = []
    if helpers is not None and helpers._SHARED:
        keep = frozenset() if item.stash.get(_FAILED, False) else _shared_keys(_next_to_run(item, nextitem))
        errors = helpers.release_shared(keep)
    result = yield
    if errors:
        raise ExceptionGroup("a shared model would not close", errors)  # type: ignore[type-var]
    return result


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
    helpers = sys.modules.get("tests.helpers")
    if helpers is not None:
        # a run stopped part way (-x, a keyboard interrupt) leaves no shared model open; a close that raised is said,
        # the receipt and the ledger written all the same
        for e in helpers.release_shared():
            sys.stderr.write(f"\n[shared model] would not close at the session's end: {e!r}\n")
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
