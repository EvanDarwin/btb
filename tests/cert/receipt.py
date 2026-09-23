"""Coverage receipts: what one machine actually ran and passed. No single machine has every device - CI has no
GPU, the Mac has no CUDA, the CUDA box is not the Mac - so a cell that skips here is not a gap if another machine
ran it. Each machine emits a receipt (the cert runner writes one when BTB_CERT_RECEIPT names a path), and coverage
is the UNION of receipts: the dev's CUDA box, the dev's Mac, and CI together. A cell in no receipt is a cell no
machine has ever exercised - a real, visible gap.

Torch-free: a receipt is just JSON, so it merges anywhere. `record()` collects ids during a run; the conftest
`pytest_sessionfinish` calls `emit()` when BTB_CERT_RECEIPT is set.

    python -m tests.cert.receipt [<dir-or-file>...]  # print the merged coverage (the union of every receipt)

With no paths, `merged()` and the CLI read `receipts_dir()`: `$BTB_CERT_RECEIPTS` when set, else
`tests/cert/receipts` - the one place CI and the matrices agree receipts land.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import socket
import time

# ids the current process has run and passed; the runner appends via record(), conftest emits at session end.
_COVERED: set[str] = set()


def record(cell_id: str) -> None:
    """mark one cell id as run-and-passed on this machine (called at the end of a passing runner cell)."""
    _COVERED.add(cell_id)


def machine_label() -> str:
    """a stable-ish name for the machine a receipt came from: host + arch + the accelerators present."""
    acc = []
    try:
        import torch

        if torch.cuda.is_available():
            acc.append("cuda")
    except Exception:
        pass
    try:
        import btb

        if btb.mlx_available():
            acc.append("mlx")
    except Exception:
        pass
    return f"{socket.gethostname()}/{platform.machine()}" + (f"/{'+'.join(acc)}" if acc else "")


def emit(path: str, covered: set[str] | None = None) -> None:
    """write this machine's receipt: the ids it ran and passed, with what machine and when."""
    ids = sorted(_COVERED if covered is None else covered)
    doc = {
        "machine": machine_label(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "covered": ids,
        "source": "run",  # emitted by a cert run that asserted each cell's tag; a pasted PR-body one says "pr-body"
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)


def receipts_dir() -> str:
    """where receipts land when no path is given: `$BTB_CERT_RECEIPTS`, else `tests/cert/receipts` beside this
    module. CI points the env var at its downloaded artifacts; a local run gets the in-tree directory."""
    env = os.environ.get("BTB_CERT_RECEIPTS")
    return env if env else os.path.join(os.path.dirname(os.path.abspath(__file__)), "receipts")


def _receipt_files(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths or [receipts_dir()]:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.json"))))
        else:
            out.append(p)
    return out


def merged(*paths: str) -> set[str]:
    """the union of covered ids across every receipt under the given files/dirs - the real cross-machine coverage.
    With no paths, the receipts directory (`receipts_dir()`)."""
    out: set[str] = set()
    for fn in _receipt_files(list(paths)):
        try:
            with open(fn, encoding="utf-8") as f:
                out |= set(json.load(f).get("covered", []))
        except (OSError, ValueError):
            continue
    return out


def by_source(*paths: str) -> dict[str, set[str]]:
    """covered ids grouped by the receipt's `source`: "run" for one a cert run emitted (each cell's tag was
    asserted), "pr-body" for one a contributor pasted into the PR - a claim CI cannot verify - so a report can say
    which coverage is verified and which is only claimed. A receipt without the field predates it: "run"."""
    out: dict[str, set[str]] = {}
    for fn in _receipt_files(list(paths)):
        try:
            with open(fn, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        out.setdefault(str(doc.get("source") or "run"), set()).update(doc.get("covered", []))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help=f"receipt files/dirs to merge and print (default: {receipts_dir()})")
    ap.add_argument(
        "--emit",
        metavar="PATH",
        help="write a receipt at PATH covering the --id ids, for a certifier that is not the pytest runner",
    )
    ap.add_argument("--id", action="append", default=[], metavar="ID", help="a covered cell id (repeatable)")
    a = ap.parse_args(argv)
    if a.emit:
        if not a.id:
            ap.error("--emit needs at least one --id; an empty receipt claims nothing and would read as coverage")
        emit(a.emit, set(a.id))
        print(f"emitted {len(a.id)} id(s) -> {a.emit}")
        return 0
    files = _receipt_files(a.paths)
    ids = merged(*a.paths)
    print(f"merged coverage: {len(ids)} cells run+passed across {len(files)} receipt(s)")
    for i in sorted(ids):
        print(f"  {i}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
