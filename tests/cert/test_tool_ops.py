# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The tool-calling matrix as a suite gate: the ToolFormat subclasses btb/tools.py defines and the ones the
registry claims must be the same set, every parser must have a test that exercises it, and every served family
must map to a parser (or be recorded as having none). A new parser the registry does not cover, a stale row, a
parser with no test, or a served family with no tool-calling story fails here."""

from __future__ import annotations

from btb import tools

from . import tool_ops


def test_no_gaps() -> None:
    """the full gate: registry/source drift, a parser with no test, AND a served family with no parser all fail
    here - a convention exercised only indirectly is uncertified and gates until it gets a row and a test."""
    assert tool_ops.gaps() == [], tool_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a reflection that silently found nothing would make the gate vacuous."""
    assert tool_ops.defined_parsers(), "reflected no ToolFormat subclasses from btb/tools.py"
    assert tool_ops.family_parser(), "found no family->parser mapping"


def test_formats_are_the_declared_parsers() -> None:
    """FORMATS is the registered subclasses that declare `kinds`, in definition order - not a second hand list a
    new parser could be left out of while this matrix certified it as mapped."""
    assert tuple(type(f) for f in tools.FORMATS) == tuple(c for c in tools._SUBCLASSES if c.kinds)
    assert tools.AnyText not in {type(f) for f in tools.FORMATS}, "the fallback is not a dispatch target"


def test_anytext_tries_every_in_text_parser() -> None:
    """AnyText's forms come from the same registration, filtered by `in_text`, so an unknown family gets every
    parser whose call can sit in an answer's text and nothing has to be listed twice."""
    assert tuple(type(f) for f in tools.AnyText.forms) == tuple(type(f) for f in tools.FORMATS if f.in_text)
    assert not any(f.in_text for f in tools.FORMATS if isinstance(f, tools.Harmony)), "harmony is channel tokens"


def test_dispatch_reaches_every_declared_family() -> None:
    """the measured mapping equals what the classes declare: a `kinds=` entry the engine's tool_format() never
    reaches (the drift this matrix exists to catch) shows up here as a family it hands to the AnyText guesser."""
    declared = {k: name for name, cls in tool_ops.defined_parsers().items() for k in cls.kinds}
    assert tool_ops.family_parser() == declared
