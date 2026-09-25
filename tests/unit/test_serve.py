# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The server `serve` / `ollama` run: its routes, what a request may ask, who it answers (peers, keys, hosts and
origins), its connections under load, and how a request in flight stops - a Ctrl-C, a client that leaves, a write
that fails. Over a stubbed engine but for one generate on the tiny fixture; everything here runs in seconds. The
OpenAI endpoint's tool calling is the pi harness's, in test_pi.py."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import threading
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import cast

import pytest
from pytest import CaptureFixture

from btb.draft import SpanBank
from btb.engine.text import Generation, RowGeneration
from btb.kinds import Json, Tokens
from btb.sampling import Sampling
from btb.serve import Engine, Handler, Message, _public
from tests.helpers import (
    FakeEngine,
    FakeReg,
    FakeRun,
    bare_registry,
    fixture,
    post_chat,
    request,
    request_json,
    stub_server,
)

# --- routes -------------------------------------------------------------------------------------------------


def test_server_routes_without_a_model() -> None:
    """the server's routes over an empty registry: discovery, version, tags, and the errors for a missing model"""
    from btb.serve import start

    server = start(None, host="127.0.0.1", port=0, pattern="^$").start()
    base = server.url
    try:
        assert request_json(base, "GET", "/health")[1]["ok"] is True
        assert request_json(base, "GET", "/v1/models")[1]["object"] == "list"
        assert request_json(base, "GET", "/api/tags")[1]["models"] == []
        assert request_json(base, "GET", "/api/version")[1]["version"] == "0.0.0"
        code, body = request_json(base, "POST", "/api/show", {"name": "no-such-model"})
        assert code == 404 and "not found" in body["error"]
        code, body = request_json(
            base,
            "POST",
            "/v1/chat/completions",
            {"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert code == 404 and "error" in body
        code, body = request_json(base, "POST", "/api/pull", {"name": "x"})
        assert code == 404 and "not supported" in body["error"]
    finally:
        server.close()


def _raw(host: str, port: int, method: str, path: str, extra: str = "") -> tuple[int, dict[str, str], bytes]:
    """one request over a raw socket, read to the close: (status, the headers lower-cased, every byte after
    them). http.client never reads a HEAD response's body, so it cannot show that one was sent."""
    with socket.create_connection((host, port), timeout=10) as s:
        s.sendall(f"{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\n{extra}Connection: close\r\n\r\n".encode())
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    head, _, rest = raw.partition(b"\r\n\r\n")
    status_line, *lines = head.decode("latin-1").split("\r\n")
    fields = (line.split(":", 1) for line in lines)
    return int(status_line.split()[1]), {k.strip().lower(): v.strip() for k, v in fields}, rest


def test_server_head_is_get_without_the_body() -> None:
    """HEAD on every route it dispatches, on a path nothing serves, and on a request the guard refuses: GET's
    status and headers (Content-Length the GET body's, as RFC 9110 8.6 requires) and not one byte of body"""
    from btb.serve import start
    from tests.cert.serve_ops import dispatched_routes

    paths = sorted(p for m, p in dispatched_routes() if m == "HEAD")
    assert paths, "the server dispatches no HEAD route"
    cases = [(p, "") for p in [*paths, "/no-such-route"]] + [("/health", "Origin: http://example.com\r\n")]
    server = start(None, host="127.0.0.1", port=0, pattern="^$").start()
    try:
        for path, extra in cases:
            status_get, hs_get, body = _raw(server.host, server.port, "GET", path, extra)
            status, hs, rest = _raw(server.host, server.port, "HEAD", path, extra)
            assert int(hs_get["content-length"]) == len(body) > 0, (path, hs_get)
            assert status == status_get and rest == b"", (path, status, rest)
            hs.pop("date"), hs_get.pop("date")  # the clock may tick between the two requests
            assert hs == hs_get, (path, hs, hs_get)
    finally:
        server.close()


def test_ollama_ps_lists_the_loaded_models() -> None:
    """Ollama's /api/ps, the models in memory: none before a request loads one, then the fixture a generate
    loaded - its tag, size and family, with Ollama's expires_at and size_vram (0 on the CPU it runs on)"""
    from btb.serve import start

    fx = fixture("tiny_qwen3")
    only = f"^{re.escape(os.path.basename(fx))}$"
    server = start(None, host="127.0.0.1", port=0, device="cpu", extra_paths=[fx], pattern=only).start()
    base = server.url
    try:
        ((name, entry),) = server.registry.refresh().items()
        assert request_json(base, "GET", "/api/ps") == (200, {"models": []})
        gen = {"model": name, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}
        code, body = request_json(base, "POST", "/api/generate", gen)
        assert code == 200, body
        code, body = request_json(base, "GET", "/api/ps")
        assert code == 200
        (m,) = body["models"]
        assert m["name"] == m["model"] == f"{name}:latest" and m["loaded"] is True
        assert m["size"] == entry["size"] > 0 and isinstance(m["digest"], str)
        assert m["details"]["family"] == entry["type"] and m["details"]["families"] == [entry["type"]]
        assert datetime.fromisoformat(m["expires_at"]).utcoffset() == timedelta(0), m["expires_at"]
        assert m["size_vram"] == 0
    finally:
        server.close()


def test_show_names_the_model_not_its_path() -> None:
    server = stub_server("ok")
    try:
        code, _, data = request(server.url, "POST", "/api/show", json.dumps({"model": "fake"}).encode())
        assert code == 200 and json.loads(data)["modelfile"] == "# btb\nFROM fake\n"
    finally:
        server.close()


# --- requests: what a request may ask -----------------------------------------------------------------------


def test_a_bad_request_field_is_a_400_naming_it() -> None:
    """a temperature that is not a number, a top_p past 1, a max_tokens of 0, messages as a string, a body that
    is not an object, Ollama options of the wrong shape: each a 400 with the line, never a dropped connection"""
    server = stub_server("ok")
    try:
        msgs = [{"role": "user", "content": "hi"}]
        body: Json
        for body, word in (
            ({"messages": msgs, "temperature": "hot"}, "temperature="),
            ({"messages": msgs, "top_p": 5}, "top_p="),
            ({"messages": msgs, "top_k": -1}, "top_k="),
            ({"messages": msgs, "seed": 1.5}, "seed="),
            ({"messages": msgs, "max_tokens": 0}, "max_tokens="),
            ({"messages": msgs, "max_tokens": "many"}, "max_tokens="),
            ({"messages": "hi"}, "messages:"),
            ({"messages": [1, 2]}, "messages:"),
        ):
            code, data = post_chat(server, body)
            assert code == 400 and word in json.loads(data)["error"], (body, code, data)
        code, _, data = request(server.url, "POST", "/v1/chat/completions", [1, 2])
        assert code == 400 and "object" in json.loads(data)["error"]
        for body, word in (
            ({"model": "fake", "messages": msgs, "options": {"temperature": -1}}, "temperature="),
            ({"model": "fake", "messages": msgs, "options": {"num_predict": "x"}}, "num_predict="),
            ({"model": "fake", "messages": msgs, "options": 3}, "options:"),
            ({"model": "fake", "messages": "hi"}, "messages:"),
        ):
            code, _, data = request(server.url, "POST", "/api/chat", body)
            assert code == 400 and word in json.loads(data)["error"], (body, code, data)
        code, _, data = request(
            server.url, "POST", "/api/generate", {"model": "fake", "prompt": "hi", "options": {"top_p": 2}}
        )
        assert code == 400 and "top_p=" in json.loads(data)["error"]
        # the good request still answers, with num_predict -1 (no cap) accepted as Ollama means it
        code, data = post_chat(server, {"messages": msgs, "temperature": 0.5, "top_p": 0.9, "max_tokens": 8})
        assert code == 200, data
        code, _, data = request(
            server.url,
            "POST",
            "/api/chat",
            {"model": "fake", "messages": msgs, "stream": False, "options": {"num_predict": -1}},
        )
        assert code == 200, data
    finally:
        server.close()


def test_a_nonsense_field_is_a_400_never_a_500() -> None:
    """values a client should never send - infinities, a list for a string, an object for a flag - each answered
    400 naming the field, before anything reaches a template or the engine"""
    server = stub_server("ok")
    msgs = [{"role": "user", "content": "hi"}]
    inf = 1e999  # json.dumps writes Infinity, which json.loads reads back
    try:
        for path, body, word in (
            ("/v1/chat/completions", {"messages": msgs, "max_tokens": inf}, "max_tokens="),
            ("/v1/chat/completions", {"messages": msgs, "max_tokens": float("nan")}, "max_tokens="),
            ("/v1/chat/completions", {"messages": [{"role": 1, "content": "hi"}]}, "role"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": 5}]}, "content"),
            ("/v1/chat/completions", {"messages": [{"role": "user", "content": ["x"]}]}, "content"),
            ("/v1/chat/completions", {"messages": msgs, "model": ["a"]}, "model="),
            ("/v1/chat/completions", {"messages": msgs, "stream_options": [1]}, "stream_options="),
            ("/v1/chat/completions", {"messages": msgs, "tools": [{"type": "function"}]}, "tools:"),
            (
                "/v1/chat/completions",
                {"messages": [{"role": "assistant", "content": None, "tool_calls": [{"function": "x"}]}]},
                "tool_calls[].function",
            ),
            ("/api/chat", {"messages": msgs, "options": {"num_predict": inf}}, "num_predict="),
            ("/api/chat", {"messages": [{"role": "user", "content": {"a": 1}}]}, "content"),
            ("/api/generate", {"prompt": ["hi"]}, "prompt="),
            ("/api/generate", {"prompt": "hi", "system": 3}, "system="),
            ("/api/generate", {"prompt": "hi", "raw": "yes"}, "raw="),
            ("/api/show", {"model": {"x": 1}}, "model="),
        ):
            code, _, data = request(server.url, "POST", path, body)
            assert code == 400 and word in json.loads(data)["error"], (path, body, code, data)
    finally:
        server.close()


def test_a_stream_flag_is_a_boolean() -> None:
    from btb.options import BadValue
    from btb.serve import _flag

    assert _flag({"stream": True}, "stream", False) is True and _flag({}, "stream", True) is True
    assert _flag({"stream": 0}, "stream", True) is False
    with pytest.raises(BadValue):
        _flag({"stream": "false"}, "stream", False)


def test_the_body_is_bounded_and_its_length_checked() -> None:
    from btb.serve import MAX_BODY

    server = stub_server("ok")
    try:
        code, _, data = request(server.url, "POST", "/v1/chat/completions", b"{}", {"Content-Length": "abc"})
        assert code == 400 and "Content-Length" in json.loads(data)["error"]
        code, _, data = request(
            server.url, "POST", "/v1/chat/completions", b"{}", {"Content-Length": str(MAX_BODY + 1)}
        )
        assert code == 413 and str(MAX_BODY) in json.loads(data)["error"]
    finally:
        server.close()


def test_a_request_never_sizes_a_decode_past_the_window() -> None:
    """a max_tokens of 10**30 is capped at what fits after the prompt: the engine reserves a cache for the whole
    decode, so an uncapped one is refused as an impossible allocation"""
    asked: list[int | None] = []

    class _Model:
        window = 64

        def generate(self, ids: Tokens, max_new: int | None, **kw: object) -> RowGeneration:
            asked.append(max_new)
            return Generation([], {"cap": max_new or 0})

    eng = Engine.__new__(Engine)
    eng.sm, eng.max_new, eng.eos, eng.session = _Model(), None, (), None  # type: ignore[assignment]
    eng.bank, eng.sampling = SpanBank(), Sampling()
    eng.run([], 10**30, ids=[1, 2, 3, 4])
    eng.run([], 8, ids=[1, 2, 3, 4])
    assert asked == [60, 8]


def test_a_request_cannot_name_a_path_on_the_machine() -> None:
    """the registry serves what it discovered: a directory a request names is not loaded, even a real model"""
    from btb.serve import ModelRegistry

    reg = ModelRegistry.__new__(ModelRegistry)
    reg.entries = {
        "fake": {"name": "fake", "repo": "org/fake", "path": "/models/fake", "type": None, "size": 0, "packed": False}
    }
    reg.primary_name = "fake"
    assert reg.resolve_name("fake:latest") == "fake" and reg.resolve_name("org/fake") == "fake"
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "tiny_q35")
    assert reg.resolve_name(here) is None
    assert reg.resolve_name("/models/fake") == "fake"  # the discovered path itself still names it


# --- access: who the server answers -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "peer,public",
    [
        ("8.8.8.8", True),
        ("2001:4860:4860::8888", True),
        ("::ffff:8.8.8.8", True),  # an IPv4-mapped peer is judged as its IPv4
        ("not-an-address", True),  # fails closed
        ("127.0.0.1", False),
        ("::1", False),
        ("192.168.1.20", False),
        ("10.0.0.5", False),
        ("172.20.1.1", False),
        ("fe80::1%en0", False),
        ("100.101.102.103", False),  # Tailscale's IPv4 range (carrier-grade NAT)
        ("fd7a:115c:a1e0::1", False),  # Tailscale's IPv6 range (unique local)
        ("::ffff:192.168.1.20", False),
    ],
)
def test_only_a_public_peer_is_public(peer: str, public: bool) -> None:
    assert _public(peer) is public


def test_a_public_peer_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """a keyless server answers a public peer 403 before it holds a request thread or a slot, and the peer reads
    that answer even while it is still sending a body; with a key the peer must present it (401), and a private
    peer needs a key only when one is set"""
    server = stub_server("ok")
    srv = server._srv
    try:
        assert request(server.url, "GET", "/health")[0] == 200  # a private peer is served
        monkeypatch.setattr("btb.serve._public", lambda host: True)
        code, headers, data = request(server.url, "GET", "/health")
        assert code == 403 and headers.get("connection") == "close", (code, headers)
        assert json.loads(data)["error"] == "An API_KEY must be provided"
        assert request(server.url, "POST", "/v1/chat/completions", {"messages": []})[0] == 403
        assert request(server.url, "POST", "/v1/chat/completions", {"messages": [], "pad": "x" * (1 << 20)})[0] == 403
        srv.api_key = "k"
        assert request(server.url, "GET", "/health")[0] == 401
        assert request(server.url, "GET", "/health", headers={"Authorization": "Bearer k"})[0] == 200
        monkeypatch.setattr("btb.serve._public", lambda host: False)
        assert request(server.url, "GET", "/health")[0] == 401  # a set key binds a private peer too
        srv.api_key = None
        assert request(server.url, "GET", "/health")[0] == 200
    finally:
        server.close()


def test_binding_every_interface_or_a_public_address_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from btb import serve

    monkeypatch.setattr(serve, "ModelRegistry", lambda *a, **kw: FakeReg(FakeEngine("ok")))
    serve.start(None, host="127.0.0.1", port=0)._srv.server_close()
    with pytest.raises(serve.KeyRequired, match=r"^An API_KEY must be provided to bind 0\.0\.0\.0$"):
        serve.start(None, host="0.0.0.0", port=0)
    serve.start(None, host="0.0.0.0", port=0, api_key="k")._srv.server_close()
    monkeypatch.setattr(serve, "_public", lambda host: True)
    with pytest.raises(serve.KeyRequired):
        serve.start(None, host="127.0.0.1", port=0)
    serve.start(None, host="127.0.0.1", port=0, api_key="k")._srv.server_close()


def test_binding_beyond_loopback_says_so(capsys: CaptureFixture[str]) -> None:
    from btb.serve import _exposure

    _exposure("serve", "127.0.0.1", None)
    assert capsys.readouterr().out == ""
    _exposure("serve", "192.168.1.20", None)
    assert "without an API key" in capsys.readouterr().out
    _exposure("serve", "0.0.0.0", "k")
    assert "with an API key" in capsys.readouterr().out


def test_the_server_refuses_a_foreign_host_or_origin() -> None:
    """bound to loopback, a Host that is not this machine (DNS rebinding: a page's name resolving to 127.0.0.1)
    and a cross-origin browser request are 403; local names pass"""
    from btb.serve import _host_of

    assert _host_of("[::1]:8000") == "::1" and _host_of("localhost:8000") == "localhost"
    assert _host_of("https://evil.example:443/x") == "evil.example" and _host_of("127.0.0.1") == "127.0.0.1"
    server = stub_server("ok")
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        code, _, data = request(server.url, "POST", "/v1/chat/completions", body, {"Host": "evil.example"})
        assert code == 403 and "Host" in json.loads(data)["error"]
        code, _, _ = request(server.url, "GET", "/v1/models", b"", {"Host": "evil.example:8000"})
        assert code == 403
        code, _, _ = request(server.url, "POST", "/v1/chat/completions", body, {"Origin": "https://evil.example"})
        assert code == 403
        # `Origin: null` is what a sandboxed frame on any site sends: cross-origin, refused like the rest
        code, _, _ = request(server.url, "POST", "/v1/chat/completions", body, {"Origin": "null"})
        assert code == 403
        for ok in (
            {"Host": f"localhost:{server.port}"},
            {"Host": "[::1]:1"},
            {"Origin": "http://localhost:3000"},
        ):
            code, _, data = request(server.url, "POST", "/v1/chat/completions", body, ok)
            assert code == 200, (ok, code, data)
    finally:
        server.close()


def test_an_api_key_gates_every_route() -> None:
    server = stub_server("ok")
    server._srv.api_key = "s3cret"
    try:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        code, hs, data = request(server.url, "POST", "/v1/chat/completions", body)
        assert code == 401 and hs.get("www-authenticate") == "Bearer" and "API key" in json.loads(data)["error"]
        assert request(server.url, "GET", "/v1/models", b"")[0] == 401
        assert request(server.url, "POST", "/v1/chat/completions", body, {"Authorization": "Bearer wrong"})[0] == 401
        assert request(server.url, "POST", "/v1/chat/completions", body, {"Authorization": "Bearer s3cret"})[0] == 200
        assert request(server.url, "GET", "/v1/models", b"", {"Authorization": "bearer s3cret"})[0] == 200
    finally:
        server.close()


# --- connections: a burst, and the cap ----------------------------------------------------------------------


def test_a_burst_of_connections_waits_to_be_accepted() -> None:
    """more connections at once than the default accept queue (5) holds, none accepted yet, all finish their
    handshake: a full queue would reset the rest on macOS and leave them hanging on Linux"""
    from btb.serve import start

    server = start(None, host="127.0.0.1", port=0, pattern="^$")  # listening, never serving: nothing is accepted
    socks: list[socket.socket] = []
    try:
        for _ in range(32):
            socks.append(socket.create_connection((server.host, server.port), timeout=2))
    finally:
        for s in socks:
            s.close()
        server.close()


@pytest.mark.timing
def test_connections_past_the_cap_are_shed() -> None:
    import time

    server = stub_server("ok")
    server._srv._slots = threading.BoundedSemaphore(1)
    hold = socket.create_connection((server.host, server.port), timeout=10)  # takes the one slot, sends nothing
    try:
        time.sleep(0.3)
        code, hs, _ = request(server.url, "GET", "/v1/models", b"")
        assert code == 503 and hs.get("connection") == "close"
    finally:
        hold.close()
        time.sleep(0.3)
        assert request(server.url, "GET", "/v1/models", b"")[0] == 200
        server.close()


# --- stopping a request in flight ---------------------------------------------------------------------------


def test_serve_ctrl_c_aborts_the_request_then_closes_gracefully() -> None:
    # a Ctrl-C must abort any in-flight generation at once (so its handler frees the lock) and still run close(),
    # which tears the tiers down; the OS delivering SIGINT is simulated by invoking the installed handler
    import signal

    from btb import serve as S

    class _SM:
        def __init__(self) -> None:
            self.abort = threading.Event()

    class _Eng:
        def __init__(self) -> None:
            self.sm = _SM()

    class _Reg:
        loaded = {"a": _Eng(), "b": _Eng()}

    closed = []

    class _Srv:
        registry = _Reg()

        def serve_forever(self) -> None:
            handler = signal.getsignal(signal.SIGINT)
            assert callable(handler)  # the test installed one
            handler(signal.SIGINT, None)  # the OS's Ctrl-C: run the installed handler

        def close(self) -> None:
            closed.append(True)

    S._serve_until_interrupt(_Srv())  # type: ignore[arg-type]
    assert all(e.sm.abort.is_set() for e in _Reg.loaded.values()), "every loaded engine is told to stop"
    assert closed == [True], "the graceful teardown still runs"


def test_registry_close_stops_the_request_in_flight_before_the_model_closes() -> None:
    """a Ctrl-C mid-answer: the registry's close tells the running generation to stop and closes the engine only
    once the request's handler has released the lock (an engine closed under a running pass is a segfault)"""
    order = []

    class Model:
        def __init__(self) -> None:
            self.abort = threading.Event()

        def close(self) -> None:
            order.append("closed")

    class Eng:
        def __init__(self) -> None:
            self.sm = Model()

    reg = bare_registry()
    eng = Eng()
    reg.loaded["m"] = cast("Engine", eng)

    def handling() -> None:
        # the handler holds the lock for the whole request and its loop ends once abort is set
        with reg.lock:
            order.append("running")
            eng.sm.abort.wait(timeout=10)
            order.append("stopped")

    th = threading.Thread(target=handling)
    th.start()
    while "running" not in order:
        pass
    reg.close()
    th.join(timeout=10)
    assert order == ["running", "stopped", "closed"] and not reg.loaded


@pytest.mark.timing
def test_openai_stream_client_disconnect_stops_the_model_and_frees_the_next_turn() -> None:
    # a cancelled stream must stop the generation at its next step, not finish the answer while the next request
    # waits on the registry's lock; the abort is cleared afterwards so the next turn runs
    import time

    class _SM:
        def __init__(self) -> None:
            self.abort = threading.Event()

    class _Slow(FakeEngine):
        def __init__(self) -> None:
            super().__init__("x")
            self.sm = _SM()
            self.emitted = 0
            self.done = threading.Event()

        def run(
            self,
            messages: Sequence[Message],
            max_new: int | None,
            on_token: Callable[[int], object] | None = None,
            ids: Tokens | None = None,
            sampling: Sampling | None = None,
            **_hooks: object,
        ) -> FakeRun:
            toks = []
            try:
                for _ in range(int(max_new) if max_new else 1000):
                    if self.sm.abort.is_set():
                        break
                    toks.append(ord("a"))
                    self.emitted += 1
                    if on_token is not None:
                        on_token(ord("a"))
                    time.sleep(0.01)
                return (ids or [1, 2, 3]), toks, {"cap": 9999, "forwards": 0}, None
            finally:
                self.done.set()

    eng = _Slow()
    server = stub_server(engine=eng)
    try:
        c = http.client.HTTPConnection(server.host, server.port, timeout=10)
        body = {"model": "fake", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        assert r.status == 200
        assert r.read(1)  # the stream is flowing
        r.close()  # the client cancels: the socket closes with the response that holds it
        c.close()
        assert eng.done.wait(10), "the generation kept running after the client left"
        assert eng.emitted < 900, eng.emitted
        deadline = time.perf_counter() + 5
        while eng.sm.abort.is_set() and time.perf_counter() < deadline:  # the handler clears it after the join
            time.sleep(0.01)
        assert not eng.sm.abort.is_set(), "the abort is cleared for the next request"
        eng.done.clear()
        t0 = time.perf_counter()
        status, data = post_chat(
            server, {"model": "fake", "messages": [{"role": "user", "content": "again"}], "max_tokens": 5}
        )
        assert status == 200, data
        assert time.perf_counter() - t0 < 5  # not queued behind the cancelled answer (1000 tokens at 10 ms)
    finally:
        server.close()


def test_a_write_that_raises_mid_stream_stops_the_model_before_the_lock_goes() -> None:
    """a client that stops reading raises TimeoutError out of the write (an OSError `_write` absorbs); an emit
    that raises anything else still stops the decode and joins it before the exception leaves `_decode`, so the
    next request never runs on an engine still decoding; the abort is cleared afterwards"""
    import time

    class _SM:
        def __init__(self) -> None:
            self.abort = threading.Event()

    class _Slow(FakeEngine):
        def __init__(self) -> None:
            super().__init__("x")
            self.sm = _SM()
            self.emitted = 0

        def run(
            self,
            messages: Sequence[Message],
            max_new: int | None,
            on_token: Callable[[int], object] | None = None,
            ids: Tokens | None = None,
            sampling: Sampling | None = None,
            **_hooks: object,
        ) -> FakeRun:
            toks = []
            for _ in range(1000):
                if self.sm.abort.is_set():
                    break
                toks.append(ord("a"))
                self.emitted += 1
                if on_token is not None:
                    on_token(ord("a"))
                time.sleep(0.005)
            return (ids or [1, 2, 3]), toks, {"cap": 9999, "forwards": 0}, None

    eng = _Slow()
    h = Handler.__new__(Handler)
    h.command = "POST"  # what parse_request sets on a real handler: the streams answer a POST

    def emit(delta: str, kind: str) -> bool:
        raise RuntimeError("the callback failed")

    with pytest.raises(RuntimeError, match="callback"):
        Handler._decode(h, cast("Engine", eng), [{"role": "user", "content": "hi"}], None, emit)
    assert not eng.sm.abort.is_set() and eng.emitted < 900

    class _Wf:
        def write(self, b: bytes) -> None:
            raise TimeoutError("timed out")

        def flush(self) -> None:
            pass

    h.wfile = _Wf()  # type: ignore[assignment]
    assert Handler._write(h, "data") is False
