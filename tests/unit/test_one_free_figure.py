# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""One figure for free memory: every decision btb makes about what fits reads the device's ledger (`Device.free`,
through a grant, a room, `_make_room` or `prefill_room`), which knows the margin, what is spoken for, and what btb
itself just took. A direct read of the OS's or the driver's figure anywhere else is a second figure: something sized
against it disagrees with what the ledger will grant, and the two refuse, or run out, at different points (a chunk
priced by `mem_get_info` that the ledger then could not make room for; the expert store sizing its blocks against the
OS's free RAM while the ledger's reservations stood).

The ledger's own home reads the raw figures (`device.py`, and `sysinfo.py`, which wraps the OS). Anywhere else a
direct read carries a note on its line, or the line above, saying why it is not a decision the ledger should make -
a report or a log, a plan made before the model and its ledger exist - as `# free-read: <why>`. An unnoted read
fails here."""

from __future__ import annotations

import ast
import os

from tests.helpers import checkout

# the calls that read free memory, or what torch holds, directly
READS = frozenset(
    {
        "mem_get_info",
        "host_free_bytes",
        "host_commit_bytes",
        "memory_reserved",
        "memory_allocated",
        "max_memory_reserved",
        "max_memory_allocated",
        "free_bytes",
        "_physical_free_bytes",
        "virtual_memory",
        "darwin_available_bytes",
    }
)
# the ledger's own home: the raw figures are read here, and turned into the one btb decides by
HOME = frozenset({"btb/engine/device.py", "btb/sysinfo.py"})
NOTE = "free-read:"


def _reads() -> list[tuple[str, int, str, str]]:
    """every direct read outside the ledger's home: (path, line, the call, the function it is in)"""
    root = checkout()
    out = []
    for dirpath, _dirs, files in os.walk(os.path.join(root, "btb")):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if rel in HOME:
                continue
            tree = ast.parse(open(path, encoding="utf-8").read())

            def walk(node: ast.AST, scope: list[str], rel: str = rel) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        walk(child, [*scope, child.name])
                        continue
                    if isinstance(child, ast.Call):
                        fn = child.func
                        name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                        if name in READS:
                            out.append((rel, child.lineno, name, ".".join(scope) or "<module>"))
                    walk(child, scope)

            walk(tree, [])
    return sorted(out)


def _noted(rel: str, line: int, lines: dict[str, list[str]]) -> bool:
    if rel not in lines:
        lines[rel] = open(os.path.join(checkout(), rel), encoding="utf-8").read().splitlines()
    src = lines[rel]
    return any(NOTE in src[i] for i in (line - 1, line - 2) if 0 <= i < len(src))


def test_every_free_memory_read_is_the_ledgers_or_says_why_not() -> None:
    lines: dict[str, list[str]] = {}
    bare = [f"{rel}:{line} {fn} ({name})" for rel, line, name, fn in _reads() if not _noted(rel, line, lines)]
    assert not bare, (
        "free memory read past the ledger (read `self.device.free(...)`, or note why it is not a decision the "
        f"ledger should make with `# {NOTE} <why>`):\n  " + "\n  ".join(bare)
    )


def test_the_scan_sees_the_reads_it_is_meant_to() -> None:
    """the scan finds a known read outside the ledger's home, so a quiet pass means the reads are noted, not that
    the scan went blind"""
    assert any(rel == "btb/engine/tiers.py" and name == "max_memory_allocated" for rel, _l, name, _f in _reads())
