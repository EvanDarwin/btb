"""Coverage receipts: what one machine actually ran and passed. No single machine has every device - CI has no
GPU, the Mac has no CUDA, the CUDA box is not the Mac - so a cell that skips here is not a gap if another machine
ran it. Each machine emits a receipt (the cert runner writes one when BTB_CERT_RECEIPT names a path), and coverage
is the UNION of receipts: the dev's CUDA box, the dev's Mac, and CI together. A cell in no receipt is a cell no
machine has ever exercised - a real, visible gap.

Torch-free: a receipt is just JSON, so it merges anywhere. `record()` collects ids during a run; the conftest's
`pytest_sessionstart` calls `begin()` and its `pytest_sessionfinish` `emit()` when BTB_CERT_RECEIPT is set.

    python -m tests.cert.receipt [<dir-or-file>...]  # print the merged coverage (the union of every receipt)
    python -m tests.cert.receipt --pr-block PATH     # a receipt as a PR comment, folded in a <details> block

With no paths, `merged()` and the CLI read `receipts_dir()`: `$BTB_CERT_RECEIPTS` when set, else
`tests/cert/receipts` - the one place CI and the matrices agree receipts land. A receipt reaches CI pasted into the
PR: its body, or a comment by its author or a maintainer (`collect`).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import glob
import gzip
import json
import os
import platform
import re
import socket
import subprocess
import time
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypedDict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ids the current process has run and passed; the runner appends via record(), conftest emits at session end.
_COVERED: set[str] = set()


class Receipt(TypedDict, total=False):
    """a receipt: the machine, when it finished, the cells it ran and passed and where it came from; a cert run's
    also the commit it certified, whether that tree had uncommitted changes and the seconds it took, and a pasted
    one who pasted it"""

    machine: str
    when: str
    covered: list[str]
    source: str
    sha: str | None
    dirty: bool | None
    seconds: int
    by: str


@dataclass(frozen=True)
class _Start:
    """a cert run's start (`begin`): the commit it certifies, whether its tree had uncommitted changes, the time"""

    sha: str | None
    dirty: bool | None
    at: float


_START: _Start | None = None


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


def _git(*args: str) -> str | None:
    """git's output in the checkout, None where git or the checkout is not there"""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def begin() -> None:
    """mark this run's start: the commit it certifies (the checkout's HEAD, else CI's GITHUB_SHA), whether its tree
    had uncommitted changes, the time. Taken at the start, as a run can outlast the commit it began on"""
    global _START
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    sha = head.strip() if head else os.environ.get("GITHUB_SHA")
    _START = _Start(sha=sha or None, dirty=bool(status.strip()) if status is not None else None, at=time.time())


def emit(path: str, covered: set[str] | None = None) -> None:
    """write this machine's receipt: the ids it ran and passed, with what machine and when, and after `begin()` the
    commit it certified, whether its tree had changes and the run's seconds"""
    doc: Receipt = {
        "machine": machine_label(),
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "covered": sorted(_COVERED if covered is None else covered),
        "source": "run",  # emitted by a cert run that asserted each cell's tag; a pasted one says "pr-body"/"pr-comment"
    }
    if _START is not None:
        doc["sha"], doc["dirty"], doc["seconds"] = _START.sha, _START.dirty, round(time.time() - _START.at)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)


# a receipt in a PR: a fenced block of plain JSON, or of its gzip in base64 (a long one, folded in a <details> block),
# at most MAX_BYTES once decoded and MAX_IDS cells
MAX_BYTES = 1 << 20
MAX_IDS = 20000
COMMENT_CHARS = 65536  # GitHub's cap on a comment or PR body
_BLOCK = re.compile(r"```(json btb-receipt|btb-receipt-gz)[^\n]*\n(.*?)\n?```", re.S)
# besides the PR's author, who may hand CI a receipt: a maintainer, by the comment's author_association
MAINTAINERS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
# a pasted receipt's strings are rendered into the job summary and compared to cell ids: one line, a plain charset,
# bounded, so a crafted one can neither inject markdown nor grant anything (a mangled id matches no cell)
_UNSAFE = re.compile(r"[^A-Za-z0-9 ._/+:-]")
_LINK = re.compile(r"://|www\.", re.I)  # the two autolink triggers; ':' itself stays for the timestamp
_SHA = re.compile(r"[0-9a-f]{7,40}")


def pr_block(path: str) -> str:
    """the receipt at `path` as a PR comment: its gzip in base64 in a ```btb-receipt-gz block, folded in a <details>
    block that names the machine, the cells and the commit"""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    raw = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError(f"[receipt] {len(raw)} bytes, past the {MAX_BYTES} a PR's receipt may decode to")
    text = base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode()
    lines = "\n".join(text[i : i + 76] for i in range(0, len(text), 76))
    sha = doc.get("sha")
    at = f" at {sha[:12]}{' with uncommitted changes' if doc.get('dirty') else ''}" if isinstance(sha, str) else ""
    took = f" in {doc['seconds'] // 60} min" if isinstance(doc.get("seconds"), int) else ""
    summary = f"btb cert receipt: {doc.get('machine')}, {len(doc.get('covered', []))} cells{at}{took}"
    block = f"<details><summary>{summary}</summary>\n\n```btb-receipt-gz\n{lines}\n```\n\n</details>\n"
    if len(block) > COMMENT_CHARS:
        raise ValueError(f"[receipt] {len(block)} characters, past the {COMMENT_CHARS} a GitHub comment holds")
    return block


def _decode(kind: str, text: str) -> object:
    """a block's receipt, None where it is malformed or decodes past MAX_BYTES (a gzip bomb stops at the bound)"""
    if kind == "btb-receipt-gz":
        if len(text) > 2 * MAX_BYTES:
            return None
        try:
            packed = base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError):
            return None
        d = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
        try:
            raw = d.decompress(packed, MAX_BYTES + 1)
        except zlib.error:
            return None
        if len(raw) > MAX_BYTES or not d.eof:
            return None
    else:
        raw = text.encode()
        if len(raw) > MAX_BYTES:
            return None
    try:
        doc: object = json.loads(raw)
    except (ValueError, RecursionError):  # malformed, or nested past the interpreter's limit
        return None
    return doc


def _safe(s: str, limit: int) -> str:
    return _UNSAFE.sub("?", _LINK.sub("?", s.replace("\n", " ").replace("\r", " ")))[:limit]


def _clean(doc: object, source: str, by: str) -> Receipt | None:
    """a pasted receipt checked and made safe (`_UNSAFE`), labelled with where it came from and who pasted it"""
    if not isinstance(doc, dict):
        return None
    machine, when, covered = doc.get("machine"), doc.get("when"), doc.get("covered")
    if not isinstance(machine, str) or not isinstance(when, str):
        return None
    if not isinstance(covered, list) or len(covered) > MAX_IDS or not all(isinstance(c, str) for c in covered):
        return None
    out: Receipt = {
        "machine": _safe(machine, 120),
        "when": _safe(when, 40),
        "covered": [_safe(c, 200) for c in covered],
        "source": source,
        "by": _safe(by, 40),
    }
    sha, dirty, seconds = doc.get("sha"), doc.get("dirty"), doc.get("seconds")
    if isinstance(sha, str) and _SHA.fullmatch(sha):
        out["sha"] = sha
    if isinstance(dirty, bool):
        out["dirty"] = dirty
    if isinstance(seconds, int) and not isinstance(seconds, bool) and 0 <= seconds < 10**7:
        out["seconds"] = seconds
    return out


def collect(dest: str, body: str, author: str, comments: Iterable[Mapping[str, object]]) -> list[Receipt]:
    """the receipts a PR hands CI, written to `dest` a file each: its body's (its author's), and those in comments by
    its author or a maintainer, each (who, machine)'s latest: a later run of a machine replaces its earlier receipt,
    other machines and other contributors add theirs. `comments` as the API lists them, each its `login`,
    `association` (author_association), `body` and `at` (updated_at); the body counts as the earliest"""
    latest: dict[tuple[str, str], Receipt] = {}

    def take(text: str, by: str, source: str) -> None:
        for kind, block in _BLOCK.findall(text):
            r = _clean(_decode(kind, block), source, by)
            if r is not None:
                latest[(r["by"], r["machine"])] = r

    take(body, author, "pr-body")
    rows = [c for c in comments if all(isinstance(c.get(k), str) for k in ("login", "association", "body", "at"))]
    for c in sorted(rows, key=lambda c: str(c["at"])):
        login, text = str(c["login"]), str(c["body"])
        if login == author or c["association"] in MAINTAINERS:
            take(text, login, "pr-comment")
    os.makedirs(dest, exist_ok=True)
    out = list(latest.values())
    for k, r in enumerate(out, 1):
        with open(os.path.join(dest, f"pr-{k}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, sort_keys=True)
    return out


def api_comments(text: str) -> list[dict[str, object]]:
    """the issue comments in `gh api --paginate` output (a JSON array a page, back to back) as `collect` takes them"""
    dec = json.JSONDecoder()
    out: list[dict[str, object]] = []
    text = text.strip()
    i = 0
    while i < len(text):
        page, i = dec.raw_decode(text, i)
        for c in page:
            user = c.get("user") or {}
            at = c.get("updated_at")
            out.append(
                {
                    "login": user.get("login"),
                    "association": c.get("author_association"),
                    "body": c.get("body"),
                    "at": at,
                }
            )
        while i < len(text) and text[i].isspace():
            i += 1
    return out


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


def arch_of(machine: str) -> str:
    """the CPU architecture a receipt's machine label names, in Rust's `target_arch` spelling (the crate's cfg
    blocks tie each ISA tier to one): macOS says arm64 and Windows AMD64 where Linux says aarch64 and x86_64"""
    raw = machine.split("/")[1].lower() if machine.count("/") else ""
    return {"arm64": "aarch64", "amd64": "x86_64"}.get(raw, raw)


def by_arch(*paths: str) -> dict[str, set[str]]:
    """covered ids grouped by the CPU architecture of the machine that ran them (`arch_of`), so a CPU cell proven
    on an arm64 Mac is not read as proof for x86-64"""
    out: dict[str, set[str]] = {}
    for fn in _receipt_files(list(paths)):
        try:
            with open(fn, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        out.setdefault(arch_of(str(doc.get("machine") or "")), set()).update(doc.get("covered", []))
    return out


def by_source(*paths: str) -> dict[str, set[str]]:
    """covered ids grouped by the receipt's `source`: "run" for one a cert run emitted (each cell's tag was
    asserted), "pr-body"/"pr-comment" for one pasted into the PR - a claim CI cannot verify - so a report can say
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
    ap.add_argument("--pr-block", metavar="PATH", help="print the receipt at PATH as a PR comment to paste")
    ap.add_argument(
        "--collect",
        metavar="DIR",
        help="write the receipts the PR hands CI to DIR: from $PR_BODY (by $PR_AUTHOR) and the --comments",
    )
    ap.add_argument("--comments", metavar="FILE", help="the PR's comments, as `gh api --paginate` printed them")
    a = ap.parse_args(argv)
    if a.pr_block:
        print(pr_block(a.pr_block), end="")
        return 0
    if a.collect:
        text = ""
        if a.comments:
            with open(a.comments, encoding="utf-8") as f:
                text = f.read()
        env = os.environ
        head = env.get("PR_HEAD") or ""
        kept = collect(a.collect, env.get("PR_BODY") or "", env.get("PR_AUTHOR") or "", api_comments(text))
        for r in kept:
            sha = r.get("sha")
            at = f" at {sha[:12]}" if sha else " at no named commit"
            if sha and head and sha != head:
                at += " (not the PR head)"
            if r.get("dirty"):
                at += " with uncommitted changes"
            took = f", {r['seconds'] // 60} min" if "seconds" in r else ""
            print(f"  {r['by']} {r['machine']}: {len(r['covered'])} cells{at}{took} ({r['source']})")
        print(f"pasted receipts kept: {len(kept)}")
        return 0
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
