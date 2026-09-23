# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The cert gaps as a PR comment, in prose: what is missing and how to close it, for everything a cert gate
fails on, then the non-blocking findings as potential improvements. Rendered from the same structures the gates
read (`manifest.blocking()`, each matrix's `_findings()`/`gaps()`/`MISSING`), so the comment and the gates
cannot disagree; a new matrix or gap kind appears here without an edit.

    python -m tests.cert.comment [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from types import ModuleType

from . import manifest, spec

MARKER = "<!-- btb-cert-gaps -->"
RECEIPT_HOWTO = (
    "run `BTB_CERT_RECEIPT=receipt.json python -m pytest tests/cert/test_cert_runner.py` on a machine with that "
    "device and paste `receipt.json` into this PR's description as a ```` ```json btb-receipt ```` block"
)

# the text is built by the PR's own code (its MISSING tables, its symbol names) and posted by the bot: a subject
# keeps a plain charset, and every line is defanged so it cannot mention a user, reference an issue, autolink, or
# open markup of its own
_SUBJECT = re.compile(r"[^A-Za-z0-9_./ :+-]")
_ZW = "\u200b"


def _subject(s: str) -> str:
    return "`" + _SUBJECT.sub("?", s)[:120] + "`"


def _text(s: str, limit: int = 1500) -> str:
    s = " ".join(s.split())[:limit]
    s = s.replace("<", "&lt;").replace(">", "&gt;").replace("@", "@" + _ZW)
    s = s.replace("://", ":/" + _ZW + "/").replace("www.", "www" + _ZW + ".")
    return re.sub(r"#(?=\d)", "#" + _ZW, s)


def _period(s: str) -> str:
    s = s.rstrip()
    return s + ("" if s.endswith((".", "!", "?")) else ".")


def _sentence(s: str) -> str:
    return _period(s[:1].upper() + s[1:])


@dataclass(frozen=True)
class Matrix:
    title: str
    blocking: list[str]  # one prose paragraph per gap the matrix's --check fails on
    improvements: list[str]  # findings it surfaces but never blocks on


def _prose(module: ModuleType, kind: StrEnum, subject: str) -> str:
    what, how = module.MISSING[kind]
    s = _subject(subject)
    return _text(f"{_sentence(what.format(s=s))} To close: {_period(how.format(s=s))}")


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
        gating = set(mod.gaps())
        blocking: list[str] = []
        improvements: list[str] = []
        for kind, subject in mod._findings():
            gates = (subject, mod.MISSING[kind][0].format(s=subject)) in gating
            (blocking if gates else improvements).append(_prose(mod, kind, subject))
        title = re.split(r"[.:]", (mod.__doc__ or info.name).strip(), maxsplit=1)[0].removeprefix("The ")
        out.append(Matrix(_text(title[:1].upper() + title[1:]), blocking, improvements))
    return out


def _unproven(ids: list[str]) -> list[str]:
    keys = {d.key for d in spec.DEVICE_SUBPATHS}
    by: dict[str, list[str]] = {}
    for cid in ids:
        dev = next((p for p in cid.split("/") if p in keys), "other")
        by.setdefault(dev, []).append(cid)
    counts = ", ".join(f"{_subject(d)} {len(v)}" for d, v in sorted(by.items()))
    return [
        "- "
        + _text(
            f"**{len(ids)} runnable cells have no receipt**: no machine has run and passed them, so nothing proves "
            f"these paths work. By device sub-path: {counts}. To close: {RECEIPT_HOWTO}."
        ),
        "",
        f"  <details><summary>The {len(ids)} cells</summary>",
        "",
        *[f"  - {_subject(i)}" for i in ids],
        "",
        "  </details>",
        "",
    ]


def _manifest(b: manifest.Blocking) -> list[str]:
    """the manifest's blocking set as list items, in the order its --check reports them"""
    out: list[str] = []
    for p in b.problems:
        out.append(
            "- " + _text(f"**Core drift**: {p}. To close: make btb.kinds and the cert axes (tests/cert/spec.py) agree.")
        )
    if b.reasonless:
        cells = ", ".join(_subject(f"{c.kind}/{c.storage}/{c.device}/{c.decode}") for c in b.reasonless)
        out.append(
            "- "
            + _text(
                f"**{len(b.reasonless)} cells are marked did-not-run with no reason** ({cells}), which hides a gap. "
                "To close: give each a proof in manifest.dnr(), or cover it."
            )
        )
    for kind, n, fams in b.gaps:
        what, how = manifest.MISSING[kind]
        out.append(
            "- "
            + _text(
                f"**{n} cells** ({', '.join(_subject(f) for f in fams)}): {_sentence(what)} To close: {_period(how)}"
            )
        )
    if b.unproven:
        out += _unproven(b.unproven)
    for name in b.orphans:
        out.append(
            "- "
            + _text(
                f"**Orphan fixture** {_subject(name)}: it is on disk but no cell binds it. To close: bind it in "
                "tests/cert/spec.py or delete it."
            )
        )
    for fork in b.untagged:
        out.append(
            "- "
            + _text(
                f"**Untagged engine fork** {_subject(fork.name)} at {_subject(fork.where)}, selected by "
                f"{_subject(fork.selects)}: {_sentence(fork.why)} To close: record a PassTag on the branch."
            )
        )
    if b.uncertified:
        out.append(
            "- "
            + _text(
                f"**{len(b.uncertified)} engine forks no cell certifies**: each is a branch the engine takes that no "
                "device sub-path asserts, so a run down it would pass unexamined. To close one: add a sub-path in "
                "tests/cert/spec.py whose `expect` asserts its tag, and drop its manifest.FORK_NOTES entry."
            )
        )
        out += [f"  - {_subject(t.value)}: {_text(_sentence(note))}" for t, note in b.uncertified.items()]
    for tag in b.unnoted:
        out.append(
            "- "
            + _text(
                f"**Unexplained fork** {_subject(tag.value)}: no sub-path asserts it and FORK_NOTES does not say "
                "why. To close: assert it in a sub-path, or note why the grid cannot reach it."
            )
        )
    for tag in b.stale:
        out.append(
            "- "
            + _text(
                f"**Stale fork note** {_subject(tag.value)}: a sub-path asserts it now. To close: drop its "
                "manifest.FORK_NOTES entry."
            )
        )
    return out


def render() -> str:
    """the comment body"""
    b = manifest.blocking()
    mats = matrices()
    sections: list[tuple[str, list[str]]] = [("Coverage manifest", _manifest(b))]
    sections += [(m.title, [f"- {p}" for p in m.blocking]) for m in mats]
    sections = [(t, body) for t, body in sections if body]
    lines = [MARKER, "## Certification gaps", ""]
    if not sections:
        lines += ["Nothing blocks: every path the cert knows of is covered and proven by a receipt.", ""]
    else:
        lines += [
            "These implementation gaps must be closed before this change is certified; each says what is missing "
            "and how to close it. The cert gates fail until they are.",
            "",
        ]
        for title, body in sections:
            lines += [f"### {title}", "", *body, ""]
    improvements = [(m.title, m.improvements) for m in mats if m.improvements]
    if improvements:
        total = sum(len(i) for _t, i in improvements)
        lines += [f"<details><summary>Potential improvements ({total}, not blocking)</summary>", ""]
        for title, items in improvements:
            lines += [f"#### {title}", "", *[f"- {p}" for p in items], ""]
        lines += ["</details>", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, metavar="PATH", help="also write the comment here")
    a = ap.parse_args(argv)
    body = render()
    sys.stdout.write(body)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
