# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The cert gaps PR comment (tests/cert/comment.py): it names exactly what the gates fail on, keeps the
non-blocking findings apart, and cannot be steered into markup or mentions. Torch-free."""

from __future__ import annotations

from . import comment, manifest


def _blocking_part(body: str) -> str:
    return body.split("<details><summary>Potential improvements", 1)[0]


def test_the_comment_is_marked_for_its_sticky_upsert() -> None:
    assert comment.render().startswith(comment.MARKER + "\n")


def test_every_matrix_gate_finding_is_named_as_blocking() -> None:
    """whatever a matrix's --check fails on appears before the potential improvements, and whatever it surfaces
    without failing appears only after them"""
    body = comment.render()
    head = _blocking_part(body)
    for m in comment.matrices():
        for p in m.blocking:
            assert p in head, p
        for p in m.improvements:
            assert p in body and p not in head, p


def test_the_manifest_section_is_what_its_gate_fails_on() -> None:
    """the comment and `manifest --check` read one structure: the section is present exactly when the gate fails"""
    b = manifest.blocking()
    assert (manifest.check() == 1) == bool(b)
    assert ("### Coverage manifest" in comment.render()) == bool(b)
    head = _blocking_part(comment.render())
    for kind, n, _fams in b.gaps:
        assert f"**{n} cells**" in head, kind
    if b.unproven:
        assert f"**{len(b.unproven)} runnable cells have no receipt**" in head


def test_text_from_the_pr_cannot_mention_link_reference_or_open_markup() -> None:
    s = comment._text("ping @someone see https://evil.example www.evil.example #12 <img src=x> and\nnext line")
    assert "@someone" not in s and "https://" not in s and "www.evil" not in s and "#12" not in s
    assert "<img" not in s and "\n" not in s


def test_a_subject_keeps_a_plain_charset_inside_its_code_span() -> None:
    s = comment._subject("route`|<b>@x")
    assert s.startswith("`") and s.endswith("`") and s.count("`") == 2
    assert not any(c in s[1:-1] for c in "|<>@")
