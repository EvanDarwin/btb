# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The cert state as a PR comment: a coverage table, what the PR closed, one table of what to fix (the problem,
where, and the fix), and the full text folded away. Built from what the gates read (`manifest.blocking()`, each
matrix's `_findings()`, `gaps()`, `MISSING`, `TITLE` and `coverage()`), so the comment and the gates agree.

    python -m tests.cert.comment [--base BASE_FINDINGS_JSON] [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType

from . import delta, manifest, receipt, spec

MARKER = "<!-- btb-cert-gaps -->"
RECEIPT_HOWTO = (
    "On a machine with that device, run `BTB_CERT_RECEIPT=receipt.json python -m pytest "
    "tests/cert/test_cert_runner.py` and paste what `python -m tests.cert.receipt --pr-block receipt.json` "
    "prints into a comment on the PR (yours, or a maintainer's)."
)
MODELS = "Models"  # the manifest's area: the model cells

# the text comes from the PR's own code and the bot posts it: a subject keeps a plain charset, and every line is
# defanged so it cannot mention a user, reference an issue, autolink, open markup or split a table cell
_SUBJECT = re.compile(r"[^A-Za-z0-9_./ :+-]")
_ZW = "\u200b"


def _subject(s: str) -> str:
    return "`" + _SUBJECT.sub("?", s)[:120] + "`"


def _text(s: str, limit: int = 1500) -> str:
    s = " ".join(s.split())[:limit]
    s = s.replace("<", "&lt;").replace(">", "&gt;").replace("@", "@" + _ZW).replace("|", "\\|")
    s = s.replace("://", ":/" + _ZW + "/").replace("www.", "www" + _ZW + ".")
    return re.sub(r"#(?=\d)", "#" + _ZW, s)


def _period(s: str) -> str:
    s = s.rstrip()
    return s + ("" if s.endswith((".", "!", "?")) else ".")


def _sentence(s: str) -> str:
    return _period(s[:1].upper() + s[1:])


@dataclass(frozen=True)
class Row:
    """one thing to fix: a row of the table, and its full text for the details"""

    area: str
    problem: str
    where: str
    fix: str
    detail: str = ""

    def line(self) -> str:
        return f"| {self.area} | {self.problem} | {self.where} | {self.fix} |"


@dataclass(frozen=True)
class Matrix:
    title: str
    module: ModuleType
    blocking: list[Row]  # what the matrix's --check fails on
    improvements: list[Row]  # what it reports but never fails on
    coverage: list[tuple[int, int, str]]  # (done, total, what) from the matrix's coverage()


def title(module: ModuleType, kind: StrEnum) -> tuple[str, str]:
    """a gap kind's (problem, fix) from the module's TITLE; its first clause and "see details" where it has none"""
    if kind in getattr(module, "TITLE", {}):
        problem, fix = module.TITLE[kind]
        return problem, fix
    what = re.split(r": | - | \(|; ", module.MISSING[kind][0], maxsplit=1)[0]
    return _sentence(what.replace("{s}", "it")).rstrip("."), "See details"


def _row(area: str, module: ModuleType, kind: StrEnum, subject: str) -> Row:
    problem, fix = title(module, kind)
    what, how = module.MISSING[kind]
    s = _subject(subject)
    detail = f"{_sentence(what.format(s=s))} To fix: {_period(how.format(s=s))}"
    return Row(area, _text(problem), s, _text(fix), _text(detail))


def matrices() -> list[Matrix]:
    """every cert matrix in this package (a module with `_findings`, `gaps` and `MISSING`), discovered: a finding
    is blocking when the matrix's own gaps() - what its --check fails on - carries it"""
    out: list[Matrix] = []
    package = importlib.import_module(__package__ or "tests.cert")
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda i: i.name):
        if info.name.startswith("test_"):
            continue
        mod = importlib.import_module(f"{package.__name__}.{info.name}")
        if not all(hasattr(mod, a) for a in ("_findings", "gaps", "MISSING")):
            continue
        name = re.split(r"[.:]", (mod.__doc__ or info.name).strip(), maxsplit=1)[0].removeprefix("The ")
        name = _text(name[:1].upper() + name[1:]).removesuffix(" matrix")
        gating = set(mod.gaps())
        blocking: list[Row] = []
        improvements: list[Row] = []
        for kind, subject in mod._findings():
            gates = (subject, mod.MISSING[kind][0].format(s=subject)) in gating
            (blocking if gates else improvements).append(_row(name, mod, kind, subject))
        coverage = mod.coverage() if hasattr(mod, "coverage") else []
        out.append(Matrix(name, mod, blocking, improvements, coverage))
    return out


def _by_device(ids: Iterable[str]) -> dict[str, int]:
    """cell ids counted by the device sub-path each names"""
    keys = {d.key for d in spec.DEVICE_SUBPATHS}
    by: dict[str, int] = {}
    for cid in ids:
        dev = next((p for p in cid.split("/") if p in keys), "other")
        by[dev] = by.get(dev, 0) + 1
    return by


def _mark(done: int, total: int) -> str:
    return "✅" if total and done == total else ("🟡" if done else "⬜")


def coverage_rows(mats: list[Matrix]) -> list[tuple[int, int, str]]:
    """(done, total, what): the runnable cells a receipt proves, then each matrix's own coverage()"""
    runnable = manifest.runnable_ids()
    rows = [(len(runnable & receipt.merged()), len(runnable), "runnable cells proven by a receipt")]
    return rows + [row for m in mats for row in m.coverage]


def _coverage(mats: list[Matrix]) -> list[str]:
    lines = ["| Coverage | Done |", "|---|---|"]
    lines += [
        f"| {_mark(d, t)} {_text(what[:1].upper() + what[1:])} | {d} / {t} |" for d, t, what in coverage_rows(mats)
    ]
    return lines


_FINDING = re.compile(r"\[([^/\]]+)/([^\]]+)\] (.*)")


def _closed(base: list[str], mats: list[Matrix]) -> list[str]:
    """the base branch's findings this PR no longer has, struck through: a matrix's by its title, the manifest's
    cells tallied by kind, anything else as recorded"""
    modules = {m.module.__name__.rsplit(".", 1)[-1]: m.module for m in mats}
    cells: dict[str, int] = {}
    out: list[str] = []
    for line in sorted(set(base) - delta.findings()):
        m = _FINDING.match(line)
        stem, kind, subject = m.groups() if m else ("", "", line)
        if stem == "manifest" and kind in {k.value for k in manifest.Missing}:
            cells[kind] = cells.get(kind, 0) + 1
            continue
        mod = modules.get(stem)
        k = next((k for k in mod.MISSING if k.value == kind), None) if mod else None
        if mod is not None and k is not None:
            out.append(f"- ✅ ~~{_text(title(mod, k)[0])}~~ {_subject(subject)}")
        else:
            out.append(f"- ✅ ~~{_subject(line)}~~")
    for kind, n in sorted(cells.items()):
        out.append(f"- ✅ ~~{_text(title(manifest, manifest.Missing(kind))[0])}~~ ({n} cells)")
    return out


def _families(fams: list[str]) -> str:
    return ", ".join(_subject(f) for f in fams)


def manifest_rows(b: manifest.Blocking) -> list[Row]:
    """the manifest's blocking set as rows, in the order its --check reports them"""
    rows: list[Row] = []
    for p in b.problems:
        rows.append(Row(MODELS, "btb.kinds and the cert axes disagree", _text(p, 200), "Make tests/cert/spec.py match"))
    if b.reasonless:
        cells = ", ".join(_subject(f"{c.kind}/{c.storage}/{c.device}/{c.decode}") for c in b.reasonless)
        rows.append(
            Row(
                MODELS, "Cells marked did-not-run with no reason", f"{len(b.reasonless)} cells",
                "Give each a reason in manifest.dnr()", f"The cells: {cells}.",
            )
        )  # fmt: skip
    for kind, n, fams in b.gaps:
        problem, fix = title(manifest, kind)
        what, how = manifest.MISSING[kind]
        detail = _text(f"{_sentence(what)} To fix: {_period(how)}")
        rows.append(Row(MODELS, _text(problem), f"{_families(fams)} ({n} cells)", _text(fix), detail))
    if b.unproven:
        have, want = _by_device(manifest.runnable_ids() & receipt.merged()), _by_device(manifest.runnable_ids())
        table = [f"| {_subject(d)} | {have.get(d, 0)} / {n} |" for d, n in sorted(want.items())]
        detail = "\n".join(["| Device | Proven |", "|---|---|", *table, "", RECEIPT_HOWTO])
        rows.append(
            Row(
                MODELS,
                "Cells no receipt proves",
                f"{len(b.unproven)} cells",
                "Run the cert on each device, paste the receipt",
                detail,
            )
        )
    for name in b.orphans:
        rows.append(Row(MODELS, "Fixture no cell uses", _subject(name), "Bind it in tests/cert/spec.py, or delete it"))
    for fork in b.untagged:
        where = f"{_subject(fork.name)} at {_subject(fork.where)}"
        rows.append(
            Row(MODELS, "Engine branch with no PassTag", where, "Record a PassTag on it", _text(_sentence(fork.why)))
        )
    if b.uncertified:
        table = [f"| {_subject(t.value)} | {_text(_sentence(note))} |" for t, note in b.uncertified.items()]
        detail = "\n".join(["| Branch | Why no cell asserts it |", "|---|---|", *table])
        rows.append(
            Row(
                MODELS, "Engine branches no cell asserts", f"{len(b.uncertified)} branches",
                "Add a sub-path in tests/cert/spec.py that asserts it", detail,
            )
        )  # fmt: skip
    for tag in b.unnoted:
        rows.append(
            Row(MODELS, "Engine branch neither asserted nor explained", _subject(tag.value), "Assert it, or note why")
        )
    for tag in b.stale:
        rows.append(
            Row(MODELS, "Fork note for a branch now asserted", _subject(tag.value), "Drop its FORK_NOTES entry")
        )
    return rows


def _table(rows: list[Row]) -> list[str]:
    return ["| Area | Problem | Where | Fix |", "|---|---|---|---|", *[r.line() for r in rows]]


def _details(summary: str, rows: list[Row], table: bool = False) -> list[str]:
    out = [f"<details><summary>{summary}</summary>", "", *(_table(rows) + [""] if table else [])]
    for r in rows:
        if r.detail:
            out += [f"**{r.problem}** · {r.where}", "", r.detail, ""]
    return [*out, "</details>", ""]


def render(base: list[str] | None = None) -> str:
    """the comment body; `base` is the base branch's findings (`tests.cert.delta bank`), for what this PR closed"""
    mats = matrices()
    todo = manifest_rows(manifest.blocking()) + [r for m in mats for r in m.blocking]
    ideas = [r for m in mats for r in m.improvements]
    lines = [MARKER, "## Certification", ""]
    lines += [f"**{len(todo)} to fix**" + (f", {len(ideas)} optional ideas" if ideas else "") + ".", ""]
    lines += [*_coverage(mats), ""]
    closed = _closed(base, mats) if base is not None else []
    if closed:
        lines += [f"### Closed by this PR ({len(closed)})", "", *closed, ""]
    if todo:
        lines += [f"### To fix ({len(todo)})", "", *_table(todo), ""]
        lines += _details("How to fix each", todo)
    if ideas:
        lines += _details(f"Optional ideas ({len(ideas)})", ideas, table=True)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, metavar="JSON", help="the base branch's `tests.cert.delta bank` findings")
    ap.add_argument("--out", default=None, metavar="PATH", help="also write the comment here")
    a = ap.parse_args(argv)
    base: list[str] | None = None
    if a.base:
        with open(a.base, encoding="utf-8") as f:
            base = [str(line) for line in json.load(f)]
    body = render(base)
    sys.stdout.write(body)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
