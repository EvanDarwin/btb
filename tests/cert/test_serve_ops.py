# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The HTTP-server route matrix as a suite gate. The routes the handlers dispatch and the ones the registry
claims must be the same set, and every route must have a test. A new route the registry does not cover, a stale
row, or a route no test exercises fails here."""

from __future__ import annotations

from . import serve_ops


def test_no_gaps() -> None:
    """the full gate: handler/registry drift AND a route with no test both fail here - a route reachable but
    unexercised (e.g. the HEAD liveness probe, /api/ps) is uncertified and gates until it gets a test."""
    assert serve_ops.gaps() == [], serve_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a parse that silently found nothing would make the gate vacuous."""
    assert serve_ops.dispatched_routes(), "parsed no routes from btb/serve.py's handlers"
    assert serve_ops.OPS, "the route registry is empty"


# a handler carrying a verb the matrix never listed: its routes must reach the dispatch set on their own
_SYNTHETIC = '''
class Handler(BaseHTTPRequestHandler):
    """a stand-in for btb/serve.py's handler"""

    def do_GET(self) -> None:
        r = self._route()
        if r == "/health":
            return self._ok()

    def do_PUT(self) -> None:
        if self._route() in ("/v1/files", "/api/blobs"):
            return self._ok()
'''


def test_a_new_verb_is_parsed_from_the_handler() -> None:
    """the verbs come from the handler class, so a do_PUT (or do_DELETE/do_PATCH) added to btb/serve.py brings
    its routes into the checked set instead of dispatching requests no row claims."""
    assert serve_ops.handler_methods(_SYNTHETIC) == {"do_GET": "GET", "do_PUT": "PUT"}
    assert set(serve_ops.handler_methods(serve_ops._serve_source())) >= {"do_GET", "do_HEAD", "do_POST"}
