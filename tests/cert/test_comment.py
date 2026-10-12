# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The cert PR comment (tests/cert/comment.py): one table row per thing to fix, exactly what the gates fail on,
the optional ideas kept apart, a coverage table, what the PR closed, a short body, and no way for the PR's text to
open markup or mentions. Torch-free."""

from __future__ import annotations

from . import comment, delta, manifest


def test_the_comment_is_marked_for_its_sticky_upsert() -> None:
    assert comment.render().startswith(comment.MARKER + "\n")


def test_every_gap_kind_has_a_short_problem_and_fix() -> None:
    """the table's words: every MISSING kind of the manifest and each matrix has a TITLE, a few plain words each"""
    for mod in [manifest, *(m.module for m in comment.matrices())]:
        assert set(mod.TITLE) == set(mod.MISSING), mod.__name__
        for problem, fix in mod.TITLE.values():
            assert len(problem) <= 60 and len(fix) <= 70 and "{s}" not in problem + fix, (problem, fix)


def test_every_finding_is_a_row_of_its_table() -> None:
    """what a matrix's --check fails on is a row of the to-fix table; what it only reports is an optional idea"""
    todo, _, ideas = comment.render().partition("<details><summary>Optional ideas")
    for m in comment.matrices():
        for r in m.blocking:
            assert r.line() in todo, r
        for r in m.improvements:
            assert r.line() in ideas and r.line() not in todo, r


def test_the_manifest_rows_are_what_its_gate_fails_on() -> None:
    """the comment and `manifest --check` read one structure: rows exactly when the gate fails, one per gap kind"""
    b = manifest.blocking()
    assert (manifest.check() == 1) == bool(b)
    rows = comment.manifest_rows(b)
    assert bool(rows) == bool(b)
    body = comment.render()
    for r in rows:
        assert r.line() in body, r
    for _kind, n, _fams in b.gaps:
        assert f"({n} cells)" in body
    if b.unproven:
        assert f"| {len(b.unproven)} cells |" in body


def test_coverage_is_a_table_with_marks() -> None:
    body = comment.render()
    for done, total, what in comment.coverage_rows(comment.matrices()):
        mark = "✅" if done == total else ("🟡" if done else "⬜")
        assert f"| {mark} {comment._text(what[:1].upper() + what[1:])} | {done} / {total} |" in body, what


def test_the_body_stays_short() -> None:
    """no list of every unproven cell: per device counts, and the long text folded into details"""
    assert len(comment.render()) < 40_000


def test_closed_credits_only_what_the_base_had_and_this_tree_does_not() -> None:
    """a matrix finding is struck through by its title, manifest cells are tallied by kind, a finding still present
    is never credited, and with no base the section is left out"""
    now = sorted(delta.findings())
    base = [*now, "[mlx_ops/no-parity] fakefam", "[manifest/fp16-fixture] a/b/c/d", "[manifest/fp16-fixture] e/f/g/h"]
    closed = comment.render(base).split("### Closed by this PR (2)", 1)[1].split("###", 1)[0]
    assert "~~Kernel family has no parity test~~ `fakefam`" in closed
    assert "~~No fp16 fixture~~ (2 cells)" in closed
    assert "Closed by this PR" not in comment.render(now)
    assert "Closed by this PR" not in comment.render()


def test_text_from_the_pr_cannot_mention_link_reference_or_open_markup() -> None:
    s = comment._text("ping @someone see https://evil.example www.evil.example #12 <img src=x> a|b and\nnext line")
    assert "@someone" not in s and "https://" not in s and "www.evil" not in s and "#12" not in s
    assert "<img" not in s and "\n" not in s and "a\\|b" in s


def test_a_subject_keeps_a_plain_charset_inside_its_code_span() -> None:
    s = comment._subject("route`|<b>@x")
    assert s.startswith("`") and s.endswith("`") and s.count("`") == 2
    assert not any(c in s[1:-1] for c in "|<>@")
