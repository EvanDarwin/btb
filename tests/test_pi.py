"""The pi harness (pi.dev): the OpenAI endpoint's tool calling, over a stubbed model on the real server, and
the provider entry btb writes into pi's models.json."""

from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from pytest import CaptureFixture

from btb.kinds import Json, Tokens
from btb.sampling import Sampling
from btb.serve import Engine, Handler, Message, Server, _content_text, _for_template, _Server, _write_pi_config
from btb.tools import tool_calls_of
from tests.helpers import request

TOOL_OUT = 'Reading it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>'
REQ = {
    "model": "fake",
    "messages": [{"role": "user", "content": "read main.py"}],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}


class _FakeTok:
    """a token per character: enough for the stream to decode text and for Channels to see no markers"""

    unk_token_id = -1

    def convert_tokens_to_ids(self, s: str) -> int:
        return -1

    def decode(self, ids: Tokens, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids)


# what a fake engine's `run` answers: the prompt ids, the tokens, the census
_Run = tuple[Tokens, list[int], Json]


class FakeEngine:
    name = "fake"
    eos: tuple[int, ...] = ()
    sampling = Sampling()

    def __init__(self, out: str) -> None:
        self._out = out
        self.tok = _FakeTok()

    def ids_for(self, messages: Sequence[Message], tools: object = None) -> list[int]:
        return [1, 2, 3]

    def run(
        self,
        messages: Sequence[Message],
        max_new: int | None,
        on_token: Callable[[int], object] | None = None,
        ids: Tokens | None = None,
        sampling: Sampling | None = None,
    ) -> _Run:
        toks = [ord(ch) for ch in self._out]
        if on_token is not None:
            for t in toks:
                on_token(t)
        return (ids or [1, 2, 3]), toks, {"cap": 9999, "forwards": 0}

    def text(self, toks: Tokens) -> tuple[str, str]:
        return self._out, ""


class FakeReg:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.lock = threading.Lock()
        self.loaded: dict[str, FakeEngine] = {}
        self.entries: dict[str, Json] = {"fake": {"name": "fake", "path": "fake"}}
        self.primary_name = "fake"

    def acquire(self, name: str) -> FakeEngine:
        return self.engine

    def refresh(self, force: bool = False) -> dict[str, Json]:
        return self.entries

    def loaded_names(self) -> list[str]:
        return list(self.loaded)

    def loaded_get(self, name: str) -> FakeEngine | None:
        return self.loaded.get(name)

    def resolve_name(self, n: str) -> str:
        return "fake"

    def close(self) -> None:
        pass


def _server(out: str = TOOL_OUT) -> Server:
    srv = _Server(("127.0.0.1", 0), Handler)
    srv.reg = FakeReg(FakeEngine(out))  # type: ignore[assignment]
    return Server(srv, srv.reg).start()


def _post(server: Server, body: Json) -> tuple[int, str]:
    """a chat completion request: (status, the body's text)"""
    status, _, data = request(server.url, "POST", "/v1/chat/completions", body)
    return status, data


def test_openai_returns_tool_calls_when_the_model_emits_one() -> None:
    server = _server()
    try:
        status, data = _post(server, REQ)
        assert status == 200, data
        choice = json.loads(data)["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        call = choice["message"]["tool_calls"][0]
        assert call["type"] == "function" and call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "main.py"}
        assert choice["message"]["content"] == "Reading it."  # the prose kept, the <tool_call> block stripped
    finally:
        server.close()


def test_openai_streams_the_tool_calls() -> None:
    server = _server()
    try:
        status, data = _post(server, {**REQ, "stream": True})
        assert status == 200, data
        chunks = [
            json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: ") and ln[6:].strip() != "[DONE]"
        ]
        assert any(ch["choices"][0]["delta"].get("tool_calls") for ch in chunks), data
        assert any(ch["choices"][0]["finish_reason"] == "tool_calls" for ch in chunks), data
        assert data.rstrip().endswith("data: [DONE]")
        # the gate holds the call back: the <tool_call> block must never leak into the streamed content
        content = "".join(c for ch in chunks if (c := ch["choices"][0]["delta"].get("content")))
        assert "<tool_call>" not in content and "<function=" not in content, content
        assert content.strip() == "Reading it.", content  # only the prose before the call streamed
    finally:
        server.close()


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


def test_openai_tools_stream_prose_in_pieces() -> None:
    # with tools present the prose must still stream token by token, not arrive in one decode-whole chunk
    server = _server(out="Line one.\nLine two.\nLine three.\n")
    try:
        status, data = _post(server, {**REQ, "stream": True})
        assert status == 200, data
        chunks = [
            json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: ") and ln[6:].strip() != "[DONE]"
        ]
        contents = [c for ch in chunks if (c := ch["choices"][0]["delta"].get("content"))]
        assert len(contents) >= 2, contents  # streamed in pieces, not one blob
        assert "".join(contents) == "Line one.\nLine two.\nLine three.\n"
        assert any(ch["choices"][0]["finish_reason"] == "stop" for ch in chunks), data
    finally:
        server.close()


def test_openai_stays_plain_text_without_tools() -> None:
    server = _server(out="Just an answer.")
    try:
        status, data = _post(server, {"model": "fake", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200, data
        choice = json.loads(data)["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "Just an answer."
        assert "tool_calls" not in choice["message"]
    finally:
        server.close()


def test_openai_streams_usage_when_the_client_asks_for_it() -> None:
    server = _server()  # REQ carries tools, so this exercises the tools stream path
    try:
        status, data = _post(server, {**REQ, "stream": True, "stream_options": {"include_usage": True}})
        assert status == 200, data
        chunks = [
            json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: ") and ln[6:].strip() != "[DONE]"
        ]
        used = [ch for ch in chunks if ch.get("usage")]
        assert used, data
        assert used[-1]["usage"]["total_tokens"] > 0 and used[-1]["choices"] == []
    finally:
        server.close()


def test_tool_calls_of_parses_and_strips() -> None:
    prose, calls = tool_calls_of(TOOL_OUT)
    assert prose == "Reading it."
    assert calls == [{"name": "read_file", "arguments": '{"path": "main.py"}'}]


def test_tool_calls_of_reads_the_xml_function_form() -> None:
    # Qwen3.5's <function=..><parameter=..> form; a value is typed by the tool's schema, and stays text without one
    out = (
        "Retry?\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
        "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"
    )
    prose, calls = tool_calls_of(out)
    assert prose == "Retry?"
    assert calls == [{"name": "bash", "arguments": '{"command": "ls -la", "timeout": "30"}'}]
    schema = {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}
    _prose, calls = tool_calls_of(out, [{"type": "function", "function": {"name": "bash", "parameters": schema}}])
    assert calls == [{"name": "bash", "arguments": '{"command": "ls -la", "timeout": 30}'}]


def test_tool_calls_of_keeps_nested_arguments() -> None:
    out = '<tool_call>{"name": "edit", "arguments": {"loc": {"line": 5}}}</tool_call>'
    _prose, calls = tool_calls_of(out)
    assert calls == [{"name": "edit", "arguments": '{"loc": {"line": 5}}'}]


def test_openai_content_parts_are_flattened_to_text() -> None:
    # OpenAI clients (pi among them) send content as an array of typed parts; the chat template needs a string
    assert _content_text([{"type": "text", "text": "Explain "}, {"type": "text", "text": "this?"}]) == "Explain this?"
    got = _for_template([{"role": "user", "content": [{"type": "text", "text": "hi there"}]}])
    assert got[0]["content"] == "hi there"
    assert _for_template([{"role": "user", "content": "plain"}])[0]["content"] == "plain", "a string is untouched"
    passthrough = _for_template([{"role": "assistant", "content": None, "tool_calls": []}])
    assert passthrough[0]["content"] == "", "a tool-call turn's null content is the empty string the templates take"


def test_write_pi_config_merges_and_preserves_other_providers(tmp_path: Path) -> None:
    cfg = str(tmp_path / "models.json")
    with open(cfg, "w") as f:
        json.dump({"providers": {"openai": {"api": "openai-completions"}}}, f)
    _write_pi_config(cfg, "http://127.0.0.1:8000/v1", ["a", "b"])
    with open(cfg) as f:
        got = json.load(f)
    assert set(got["providers"]) == {"openai", "btb"}
    assert got["providers"]["btb"]["baseUrl"] == "http://127.0.0.1:8000/v1"
    assert got["providers"]["btb"]["api"] == "openai-completions"
    assert [m["id"] for m in got["providers"]["btb"]["models"]] == ["a", "b"]


def test_openai_streams_the_prose_after_a_call_and_never_a_partial_opener() -> None:
    # the gate holds from the opener, then releases the prose that followed the call once the block is struck
    out = (
        'Reading it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>\nDone reading.'
    )
    server = _server(out=out)
    try:
        status, data = _post(server, {**REQ, "stream": True})
        assert status == 200, data
        chunks = [
            json.loads(ln[6:]) for ln in data.splitlines() if ln.startswith("data: ") and ln[6:].strip() != "[DONE]"
        ]
        content = "".join(c for ch in chunks if (c := ch["choices"][0]["delta"].get("content")))
        assert content == "Reading it.\nDone reading.", content
        assert "<" not in content
        calls = [tc for ch in chunks for tc in (ch["choices"][0]["delta"].get("tool_calls") or [])]
        assert [c["function"]["name"] for c in calls] == ["read_file"]
        assert any(ch["choices"][0]["finish_reason"] == "tool_calls" for ch in chunks), data
    finally:
        server.close()


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
        ) -> _Run:
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
                return (ids or [1, 2, 3]), toks, {"cap": 9999, "forwards": 0}
            finally:
                self.done.set()

    eng = _Slow()
    srv = _Server(("127.0.0.1", 0), Handler)
    srv.reg = FakeReg(eng)  # type: ignore[assignment]
    server = Server(srv, srv.reg).start()
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
        status, data = _post(
            server, {"model": "fake", "messages": [{"role": "user", "content": "again"}], "max_tokens": 5}
        )
        assert status == 200, data
        assert time.perf_counter() - t0 < 5  # not queued behind the cancelled answer (1000 tokens at 10 ms)
    finally:
        server.close()


def test_a_bad_request_field_is_a_400_naming_it() -> None:
    """a temperature that is not a number, a top_p past 1, a max_tokens of 0, messages as a string, a body that
    is not an object, Ollama options of the wrong shape: each a 400 with the line, never a dropped connection"""
    server = _server("ok")
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
            code, data = _post(server, body)
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
        code, data = _post(server, {"messages": msgs, "temperature": 0.5, "top_p": 0.9, "max_tokens": 8})
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


def test_the_server_refuses_a_foreign_host_or_origin() -> None:
    """bound to loopback, a Host that is not this machine (DNS rebinding: a page's name resolving to 127.0.0.1)
    and a cross-origin browser request are 403; local names pass"""
    from btb.serve import _host_of

    assert _host_of("[::1]:8000") == "::1" and _host_of("localhost:8000") == "localhost"
    assert _host_of("https://evil.example:443/x") == "evil.example" and _host_of("127.0.0.1") == "127.0.0.1"
    server = _server("ok")
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
    server = _server("ok")
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


def test_the_body_is_bounded_and_its_length_checked() -> None:
    from btb.serve import MAX_BODY

    server = _server("ok")
    try:
        code, _, data = request(server.url, "POST", "/v1/chat/completions", b"{}", {"Content-Length": "abc"})
        assert code == 400 and "Content-Length" in json.loads(data)["error"]
        code, _, data = request(
            server.url, "POST", "/v1/chat/completions", b"{}", {"Content-Length": str(MAX_BODY + 1)}
        )
        assert code == 413 and str(MAX_BODY) in json.loads(data)["error"]
    finally:
        server.close()


def test_a_request_cannot_name_a_path_on_the_machine() -> None:
    """the registry serves what it discovered: a directory a request names is not loaded, even a real model"""
    from btb.serve import ModelRegistry

    reg = ModelRegistry.__new__(ModelRegistry)
    reg.entries = {
        "fake": {"name": "fake", "repo": "org/fake", "path": "/models/fake", "type": "", "size": 0, "packed": False}
    }
    reg.primary_name = "fake"
    assert reg.resolve_name("fake:latest") == "fake" and reg.resolve_name("org/fake") == "fake"
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "tiny_q35")
    assert reg.resolve_name(here) is None
    assert reg.resolve_name("/models/fake") == "fake"  # the discovered path itself still names it


def test_show_names_the_model_not_its_path() -> None:
    server = _server("ok")
    try:
        code, _, data = request(server.url, "POST", "/api/show", json.dumps({"model": "fake"}).encode())
        assert code == 200 and json.loads(data)["modelfile"] == "# btb\nFROM fake\n"
    finally:
        server.close()


@pytest.mark.timing
def test_connections_past_the_cap_are_shed() -> None:
    import socket
    import time

    server = _server("ok")
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


def test_binding_beyond_loopback_says_so(capsys: CaptureFixture[str]) -> None:
    from btb.serve import _exposure, _pi_provider

    _exposure("serve", "127.0.0.1", None)
    assert capsys.readouterr().out == ""
    _exposure("serve", "0.0.0.0", None)
    assert "no API key set" in capsys.readouterr().out
    _exposure("serve", "0.0.0.0", "k")
    assert "with an API key" in capsys.readouterr().out
    assert _pi_provider("http://127.0.0.1:8000/v1", ["a"], "k")["apiKey"] == "k"
    assert _pi_provider("http://127.0.0.1:8000/v1", ["a"])["apiKey"] == "btb"


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
        ) -> _Run:
            toks = []
            for _ in range(1000):
                if self.sm.abort.is_set():
                    break
                toks.append(ord("a"))
                self.emitted += 1
                if on_token is not None:
                    on_token(ord("a"))
                time.sleep(0.005)
            return (ids or [1, 2, 3]), toks, {"cap": 9999, "forwards": 0}

    eng = _Slow()
    h = Handler.__new__(Handler)

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
