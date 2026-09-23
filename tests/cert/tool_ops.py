# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The tool-calling matrix: every family's tool-call parser and chat-template shaping in btb/tools.py, crossed
with the test that exercises it and the served family it stands for. A `ToolFormat` subclass owns both halves -
it parses a call out of the answer and shapes the conversation for its family's chat template (`prepare`) - so
certifying the class certifies the templating too. A new parser with no test, a new parser no registry row
claims, a stale row, or a served family with no parser is a reported gap, so a tool-call convention cannot land
uncertified and a new family is forced to declare its tool-calling story.

Imported, not source-parsed: btb.tools is torch-free and import-light (it pulls in btb.kinds and btb.text, both
torch-free), so reflecting the live module gates nothing behind a runtime. mlx_ops/cuda_ops parse source only
because their modules import mlx.core / torch; that constraint does not apply here, so the parser set comes from
`ToolFormat.__subclasses__()` (exact classes, not a regex guess) and the family->parser mapping is MEASURED by
calling `tools.tool_format(kind)` for each served family - what the engine really dispatches, not a re-derivation
that would agree with a broken dispatch. The row declares only the parser and its test.

    python -m tests.cert.tool_ops --report    # the parser matrix and the plain-language gaps
    python -m tests.cert.tool_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.tool_ops --check     # nonzero on registry/source drift, a parser with no test, or an unmapped family
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import StrEnum

from btb import tools
from btb.kinds import FamilyKind

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, "tests")


class Missing(StrEnum):
    """what a tool-calling gap needs, as a stable key; `MISSING` maps each to (what is absent, how to close it).
    A parser or a served family with no test gates, same as native_ops/mlx_ops - a convention exercised only
    indirectly is not certified. A new gap kind is one member here and one MISSING row."""

    NO_TEST = "no-test"
    PARSER_UNCOVERED = "parser-uncovered"
    ROW_STALE = "row-stale"
    FAMILY_UNMAPPED = "family-unmapped"
    PARSER_UNREACHED = "parser-unreached"


# kind -> (what is missing about {s}, how to close it). {s} is the subject: a parser class or a family.
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.NO_TEST: (
        "parser {s} is claimed by a row but its test file does not mention it - nothing asserts it parses a call "
        "or shapes its family's template, so it is uncertified",
        "add a test to that row's file exercising {s} (its calls()/split()/prepare()), or point the row at the "
        "file that does",
    ),
    Missing.PARSER_UNCOVERED: (
        "{s} is a ToolFormat subclass in btb/tools.py that no registry row claims - a new parser with no matrix row",
        "add a Parser row naming {s} and the test that exercises it",
    ),
    Missing.ROW_STALE: (
        "PARSERS claims the parser {s} but btb/tools.py defines no such ToolFormat subclass",
        "drop the {s} row (the parser was renamed or removed)",
    ),
    Missing.FAMILY_UNMAPPED: (
        "family {s} is served by btb (btb.kinds.FamilyKind) but `tools.tool_format()` falls through to the AnyText "
        "guesser for it and it is not in NO_TOOL_SUPPORT - a new family with no tool-calling story",
        "give {s} a parser (add it to a ToolFormat's `kinds`), or record it in NO_TOOL_SUPPORT with why it has none",
    ),
    Missing.PARSER_UNREACHED: (
        "parser {s} declares families in its `kinds` but `tools.tool_format()` never returns it - the parser is "
        "dead code (it is missing from the dispatch set, or an earlier parser claims the same families)",
        "make {s} reachable: register it in the dispatch set and give it families no earlier parser already claims",
    ),
}


@dataclass(frozen=True)
class Parser:
    name: str  # the ToolFormat subclass (a report row); the families it serves are read from the class
    test: str  # the tests/<path> that exercises its parsing and template shaping


# Every tool-call parser the matrix certifies, each naming the ToolFormat subclass it covers and the test that
# exercises it. The claimed set is cross-checked against btb/tools.py by `parser_gaps()`, so a new ToolFormat
# subclass no row covers - or a row for a class that is gone - is a reported gap. AnyText is the fallback a
# family none names falls to; it carries no `kinds` and so maps no family.
PARSERS: tuple[Parser, ...] = (
    Parser("HermesJson", "unit/test_tools.py"),
    Parser("QwenXml", "unit/test_tools.py"),
    Parser("PhiJson", "unit/test_tools.py"),
    Parser("Harmony", "unit/test_tools.py"),
    Parser("GemmaJson", "unit/test_tools.py"),
    Parser("AnyText", "unit/test_tools.py"),
)

# Families btb serves that deliberately have no tool-call parser, each with why. A served family absent here and
# from every parser's `kinds` is a gap, so a new family cannot silently ship without a tool-calling decision.
NO_TOOL_SUPPORT: dict[FamilyKind, str] = {}


def defined_parsers() -> dict[str, type[tools.ToolFormat]]:
    """{class name -> class} for every ToolFormat subclass defined in btb/tools.py (recursively), the authoritative
    parser set the registry is checked against. The base class and any subclass defined elsewhere are excluded."""
    out: dict[str, type[tools.ToolFormat]] = {}
    stack = list(tools.ToolFormat.__subclasses__())
    while stack:
        cls = stack.pop()
        stack += cls.__subclasses__()
        if cls.__module__ == "btb.tools":
            out[cls.__name__] = cls
    return out


def family_parser() -> dict[FamilyKind, str]:
    """{served family -> the parser the engine hands it}, MEASURED by calling `tools.tool_format(kind)` rather
    than re-deriving it from the classes: a parser the dispatch set never reaches shows here as what the engine
    really does. A family that falls through to the AnyText guesser is absent."""
    out: dict[FamilyKind, str] = {}
    for k in FamilyKind:
        fmt = tools.tool_format(k)
        if type(fmt) is not tools.AnyText:
            out[k] = type(fmt).__name__
    return out


def _test_covers(row: Parser, cls_exists: bool) -> bool:
    """whether the row's test file mentions the parser class by name (a whole-word match) - the file exists and
    exercises it. A row for a class that no longer exists is not judged here (ROW_STALE covers that)."""
    if not cls_exists:
        return True
    path = os.path.join(TESTS, row.test)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return re.search(rf"\b{re.escape(row.name)}\b", f.read()) is not None


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, subject); everything the matrix reports derives from it, so
    gaps()/parser_gaps()/family_gaps() and the plain-language render never disagree."""
    out: list[tuple[Missing, str]] = []
    defined = defined_parsers()
    claimed = {row.name for row in PARSERS}
    out += [(Missing.PARSER_UNCOVERED, n) for n in sorted(set(defined) - claimed)]
    out += [(Missing.ROW_STALE, n) for n in sorted(claimed - set(defined))]
    for row in PARSERS:
        if not _test_covers(row, row.name in defined):
            out.append((Missing.NO_TEST, row.name))
    mapped = family_parser()
    out += [(Missing.FAMILY_UNMAPPED, str(f)) for f in FamilyKind if f not in mapped and f not in NO_TOOL_SUPPORT]
    reached = set(mapped.values())
    out += [(Missing.PARSER_UNREACHED, n) for n, cls in sorted(defined.items()) if cls.kinds and n not in reached]
    return out


_PARSER_AXIS = (Missing.PARSER_UNCOVERED, Missing.ROW_STALE, Missing.PARSER_UNREACHED)


def parser_gaps() -> list[tuple[str, str]]:
    """parser-axis drift: a ToolFormat subclass no row claims (a new parser with no matrix row), a row naming a
    class that is gone, or a parser the engine's dispatch never reaches."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings() if k in _PARSER_AXIS]


def family_gaps() -> list[tuple[str, str]]:
    """family-axis drift: a served family with no parser and no explicit NO_TOOL_SUPPORT record."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings() if k == Missing.FAMILY_UNMAPPED]


def gaps() -> list[tuple[str, str]]:
    """(subject, what) for every gap: parser/registry drift, a parser with no test, or an unmapped family. A new
    parser cannot land, and a new family cannot ship without a tool-calling story, without a gap here."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings()]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no tool-calling gaps: every parser has a row and a test, and every served family maps to a parser."]
    lines = [f"{len(items)} tool-calling gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    defined = defined_parsers()
    mapped = family_parser()
    print(
        f"tool-calling matrix: {len(PARSERS)} parser rows, {len(defined)} ToolFormat subclasses, "
        f"{len(mapped)}/{len(FamilyKind)} served families mapped"
    )
    print(f"{'parser':<12} {'families':<24} {'test':<22} {'status':<6}")
    for row in PARSERS:
        cls = defined.get(row.name)
        fams = ",".join(str(k) for k, p in mapped.items() if p == row.name)  # what dispatch hands it, not its `kinds`
        if cls is None:
            status = "STALE"
        elif not _test_covers(row, True):
            status = "GAP"
        elif cls.kinds and not fams:
            status = "DEAD"
        else:
            status = "ok"
        print(f"{row.name:<12} {(fams or '(fallback)'):<24} {row.test:<22} {status:<6}")
    for f in FamilyKind:
        if f not in mapped:
            where = NO_TOOL_SUPPORT.get(f)
            print(f"  family {f}: {'no tool support - ' + where if where else 'UNMAPPED (gap)'}")
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()
    if g:
        print(f"FAIL: {len(g)} tool-calling gaps", file=sys.stderr)
        for subject, what in g:
            print(f"  {subject}: {what}", file=sys.stderr)
    return 1 if g else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--report", action="store_true", help="print the parser matrix and the plain-language gaps (default)"
    )
    ap.add_argument(
        "--missing", action="store_true", help="print only the gaps in plain language: what is missing and how to close"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero on registry/source drift, a parser with no test, or an unmapped family",
    )
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
