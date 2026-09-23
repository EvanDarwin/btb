# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The HTTP-server route matrix: every route btb's OpenAI/Ollama server dispatches crossed with the test that
exercises it. The routes are string literals the handler methods branch on (every `do_<METHOD>` on btb/serve.py's
request handler), so a new route - or a new method on an existing one - that no registry row claims is a reported
gap, and a row naming a route the handlers no longer dispatch is a stale gap. A route with no test gates, same
as the other matrices: an endpoint reachable but unexercised is uncertified.

Parsed, not imported: `import btb.serve` pulls torch, transformers and numpy (the engine it serves), so
reflecting the live handlers would gate this whole matrix behind the runtime. cuda_ops parses native.py source
and mlx_ops parses btb/mlx source for the same reason, all three in the torch-free cert.yml gate; this follows
that precedent and stays import-light. The handler methods are parsed from the BaseHTTPRequestHandler subclass
(`def do_<METHOD>`) and the dispatch set from the `r == "/..."` / `r in (...)` / `self._route() in (...)` branches
in their bodies, plus the routes of any `self.do_<METHOD>()` they answer through (do_HEAD is do_GET without the
body), the authoritative set a request can reach - a new verb brings its routes with it.

    python -m tests.cert.serve_ops --report    # the route table plus the plain-language gaps
    python -m tests.cert.serve_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.serve_ops --check     # nonzero when a route lacks a test or drifts from the handlers
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import StrEnum

from .core import BTB_SRC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the tests/ tree
SERVE_PY = os.path.join(BTB_SRC, "serve.py")


class Missing(StrEnum):
    """what a route gap needs, as a stable key. `MISSING` maps each to (what is absent, how to close it) in
    plain language; the taxonomy lives in one place, so a new gap kind is one member here and one MISSING row."""

    ROUTE_UNCOVERED = "route-uncovered"
    ROUTE_STALE = "route-stale"
    NO_TEST = "no-test"
    TEST_MISSING = "test-missing"


# kind -> (what is missing about {s}, how to close it). {s} is the subject: a "METHOD /path" pair, or a route name.
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.ROUTE_UNCOVERED: (
        "the server dispatches {s} (a branch in one of btb/serve.py's do_<METHOD> handlers) but no Route in OPS claims "
        "it - a new route, or a new method on one, with no matrix row",
        "add or extend a Route whose `methods`/`paths` cover {s}, naming a test that exercises it",
    ),
    Missing.ROUTE_STALE: (
        "OPS claims the route {s} but no handler in btb/serve.py dispatches it",
        "drop {s} from that Route's `methods`/`paths` (the route was renamed or removed)",
    ),
    Missing.NO_TEST: (
        "route {s} has no test - nothing sends a request to it, so the route is reachable but uncertified",
        "add a test under tests/ that requests the route and asserts its answer, and name it in the Route's `test`",
    ),
    Missing.TEST_MISSING: (
        "route {s} names a test function that is not defined on disk",
        "create the named test (file::func), or correct the Route's `test` (renamed or removed)",
    ),
}


@dataclass(frozen=True)
class Route:
    name: str  # the logical endpoint (a report row)
    methods: tuple[str, ...]  # the HTTP methods whose handler dispatches these paths
    paths: tuple[str, ...]  # the request paths (serve.py string literals) this row covers, cross-checked below
    test: str | None  # a "tests/<file>.py::<func>" that exercises the route, or None when nothing does


# Every route the server dispatches, each claiming the (method, path) pairs it stands for. The claims are
# cross-checked against btb/serve.py's handlers by `route_gaps()`, so a new route - or a new method on one - that
# no row covers is a reported gap; the OPS table cannot silently fall behind the handlers. Routes that share one
# dispatch branch and behavior are one row with a `paths` tuple (as native_ops groups a kernel's exports), the
# `test` a request that exercises the branch.
OPS: tuple[Route, ...] = (
    Route("discovery", ("GET",), ("/", "/health"), "tests/unit/test_cli.py::test_server_routes_without_a_model"),
    Route(
        "head",
        ("HEAD",),
        ("/", "/health", "/v1/models", "/api/tags", "/api/ps", "/api/version"),
        "tests/unit/test_cli.py::test_server_head_is_get_without_the_body",
    ),
    Route("openai_models", ("GET",), ("/v1/models",), "tests/unit/test_cli.py::test_server_routes_without_a_model"),
    Route(
        "openai_chat",
        ("POST",),
        ("/v1/chat/completions",),
        "tests/unit/test_pi.py::test_openai_stays_plain_text_without_tools",
    ),
    Route("ollama_tags", ("GET",), ("/api/tags",), "tests/unit/test_cli.py::test_server_routes_without_a_model"),
    Route("ollama_ps", ("GET",), ("/api/ps",), "tests/unit/test_cli.py::test_ollama_ps_lists_the_loaded_models"),
    Route("ollama_version", ("GET",), ("/api/version",), "tests/unit/test_cli.py::test_server_routes_without_a_model"),
    Route(
        "ollama_chat", ("POST",), ("/api/chat",), "tests/unit/test_pi.py::test_a_bad_request_field_is_a_400_naming_it"
    ),
    Route(
        "ollama_generate",
        ("POST",),
        ("/api/generate",),
        "tests/unit/test_pi.py::test_a_bad_request_field_is_a_400_naming_it",
    ),
    Route("ollama_show", ("POST",), ("/api/show",), "tests/unit/test_pi.py::test_show_names_the_model_not_its_path"),
    Route(
        "ollama_unsupported",
        ("POST",),
        ("/api/pull", "/api/push", "/api/create", "/api/copy", "/api/delete", "/api/embed", "/api/embeddings"),
        "tests/unit/test_cli.py::test_server_routes_without_a_model",
    ),
)


_CLASS_RE = re.compile(r"\nclass \w+\(BaseHTTPRequestHandler\):(.*?)(?=\nclass |\Z)", re.S)  # the request handler
_DO_RE = re.compile(r"\n    def (do_([A-Z]+))\s*\(")  # a do_<METHOD> request handler; the method is the verb

_EQ_RE = re.compile(r'\br\s*==\s*"(/[^"]*)"')  # a `r == "/path"` branch
_IN_RE = re.compile(r"(?:\br|self\._route\(\))\s+in\s+\(([^)]*)\)", re.S)  # a `r in (...)` / `_route() in (...)` branch
_LIT_RE = re.compile(r'"(/[^"]*)"')  # a "/path" literal inside an `in (...)` tuple
_DELEGATE_RE = re.compile(r"\bself\.(do_[A-Z]+)\(")  # a handler answering through another verb's handler
_TESTREF_RE = re.compile(r"^(.+\.py)::(\w+)$")  # a Route.test, "tests/<file>.py::<func>"


def _serve_source() -> str:
    with open(SERVE_PY, encoding="utf-8") as f:
        return f.read()


def _handler_class(src: str) -> str:
    """the body of btb/serve.py's BaseHTTPRequestHandler subclass - the only class whose methods answer a request."""
    m = _CLASS_RE.search(src)
    return m.group(1) if m else ""


def handler_methods(src: str) -> dict[str, str]:
    """{do_<METHOD> -> METHOD} for every request handler `src`'s class defines, parsed rather than listed: a new
    do_PUT/do_DELETE/do_PATCH brings its routes into the dispatch set instead of dispatching what nothing checks."""
    return {m.group(1): m.group(2) for m in _DO_RE.finditer(_handler_class(src))}


def _handler_body(src: str, name: str) -> str:
    """the source of the Handler method `name`, from its `def` to the next method or class - the region whose
    route literals are that method's dispatch branches (nothing outside a handler dispatches a request)."""
    m = re.search(rf"\n    def {name}\(.*?(?=\n    def |\nclass |\Z)", src, re.S)
    return m.group(0) if m else ""


def _handler_paths(src: str, handler: str, seen: frozenset[str] = frozenset()) -> set[str]:
    """the paths `handler` dispatches: its own `r == ...` / `r in (...)` branches, plus every path of a
    do_<METHOD> it delegates to (do_HEAD answering through `self.do_GET()` dispatches all of GET's routes)"""
    body = _handler_body(src, handler)
    out = set(_EQ_RE.findall(body))
    for group in _IN_RE.findall(body):
        out.update(_LIT_RE.findall(group))
    for callee in set(_DELEGATE_RE.findall(body)) - seen - {handler}:
        out |= _handler_paths(src, callee, seen | {handler})
    return out


def dispatched_routes(full: str | None = None) -> set[tuple[str, str]]:
    """(method, path) for every route the server dispatches (btb/serve.py's, or `full` a stand-in source),
    parsed from each do_<METHOD> handler's branches and delegations - the truth the OPS table's claims are
    checked against. A request reaches exactly these; anything else falls to the handlers' default 404."""
    full = _serve_source() if full is None else full
    src = _handler_class(full)
    return {(method, p) for handler, method in handler_methods(full).items() for p in _handler_paths(src, handler)}


def _test_exists(ref: str | None) -> bool:
    """whether `ref` ("tests/<file>.py::<func>") names a test function defined on disk - a renamed or removed
    test is caught here, not left as a dead claim."""
    m = _TESTREF_RE.match(ref or "")
    if m is None:
        return False
    path = os.path.join(ROOT, m.group(1))
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        return re.search(rf"\bdef {re.escape(m.group(2))}\s*\(", f.read()) is not None


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, subject). Everything the matrix reports derives from this,
    so gaps()/route_gaps()/test_gaps() and the plain-language render never disagree. A route with no (or an
    absent) test is a gap here, so it gates - not a soft note."""
    out: list[tuple[Missing, str]] = []
    claimed = {(m, p) for op in OPS for m in op.methods for p in op.paths}
    actual = dispatched_routes()
    out += [(Missing.ROUTE_UNCOVERED, f"{m} {p}") for m, p in sorted(actual - claimed)]
    out += [(Missing.ROUTE_STALE, f"{m} {p}") for m, p in sorted(claimed - actual)]
    for op in OPS:
        if op.test is None:
            out.append((Missing.NO_TEST, op.name))
        elif not _test_exists(op.test):
            out.append((Missing.TEST_MISSING, f"{op.name} ({op.test})"))
    return out


def _what(kind: Missing, subject: str) -> str:
    return MISSING[kind][0].format(s=subject)


def route_gaps() -> list[tuple[str, str]]:
    """handler drift: a (method, path) the handlers dispatch that no Route claims (a new route with no matrix
    row), or a Route claiming one the handlers no longer dispatch."""
    return [(s, _what(k, s)) for k, s in _findings() if k in (Missing.ROUTE_UNCOVERED, Missing.ROUTE_STALE)]


def test_gaps() -> list[tuple[str, str]]:
    """test drift: a route no test exercises, or a route naming a test not defined on disk."""
    return [(s, _what(k, s)) for k, s in _findings() if k in (Missing.NO_TEST, Missing.TEST_MISSING)]


def gaps() -> list[tuple[str, str]]:
    """(subject, what) for every gap: handler drift, or a route with no (or an absent) test. A new route cannot
    land, and a route cannot go untested, without a gap here."""
    return [(s, _what(k, s)) for k, s in _findings()]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary"""
    tested = sum(1 for op in OPS if op.test is not None and _test_exists(op.test))
    return [(tested, len(OPS), "server routes with a test")]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no route gaps: every dispatched route is claimed by a row, and every row has a test."]
    lines = [f"{len(items)} route gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    claimed = {(m, p) for op in OPS for m in op.methods for p in op.paths}
    actual = dispatched_routes()
    covered = sum(1 for op in OPS if _test_exists(op.test))
    print(
        f"serve route matrix: {len(OPS)} routes over {len(claimed)} (method, path) pairs "
        f"({len(actual)} dispatched), {covered}/{len(OPS)} with a test"
    )
    print(f"{'route':<20} {'methods':<12} {'paths':<40} {'test':<8}")
    for op in OPS:
        t = "ok" if _test_exists(op.test) else "GAP"
        print(f"{op.name:<20} {','.join(op.methods):<12} {','.join(op.paths):<40} {t:<8}")
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()
    if g:
        print(f"FAIL: {len(g)} route gaps", file=sys.stderr)
        for subject, what in g:
            print(f"  {subject}: {what}", file=sys.stderr)
    return 1 if g else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--report", action="store_true", help="print the route matrix and the plain-language gaps (default)"
    )
    ap.add_argument(
        "--missing", action="store_true", help="print only the gaps in plain language: what is missing and how to close"
    )
    ap.add_argument("--check", action="store_true", help="exit nonzero on a route with no test or handler drift")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
