# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The command line and its individual commands: how `run` finds its prompt (a pipe, -p, a --file, or a
sample), and the server `serve` / `ollama` run - its routes, and the close that stops a request in flight.
Model-free but for one end-to-end `run --json` over a piped prompt on the tiny fixture; everything here runs
in seconds. The out-of-memory notice and the --profile bundle are the feedback module's, in test_feedback.py."""

import argparse
import json
import os
import re
import socket
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from pytest import CaptureFixture, MonkeyPatch

from tests.helpers import bare_registry, fixture, request_json


class _Stdin:
    """a stand-in for sys.stdin: a piped stream (isatty False) carrying `data`, or an interactive one (isatty True)"""

    def __init__(self, data: str, tty: bool) -> None:
        self.data, self.tty = data, tty

    def isatty(self) -> bool:
        return self.tty

    def read(self) -> str:
        return self.data


# --- run: the prompt sources --------------------------------------------------------------------------------


def test_run_reads_the_prompt_from_a_pipe_explicit_or_a_sample(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """`btb run` takes its one prompt from -p when given, else from a pipe on stdin, else a sample; a JSONL
    --file overrides all. Platform-independent: stdin is faked, never shell-piped."""
    from btb.cli import DEFAULT_PROMPT, _run_prompts

    def prompts(prompt: str | None = None, file: str | None = None, data: str = "", tty: bool = True) -> list[str]:
        monkeypatch.setattr(sys, "stdin", _Stdin(data, tty))
        return _run_prompts(argparse.Namespace(prompt=prompt, file=file))

    assert prompts(data="What is 2+2?\n", tty=False) == ["What is 2+2?"], "a piped prompt is taken and trimmed"
    assert prompts(prompt="explicit", data="piped", tty=False) == ["explicit"], "-p wins over a pipe"
    assert prompts(data="junk", tty=True) == [DEFAULT_PROMPT], "an interactive stdin falls back to the sample"
    assert prompts(data="  \n ", tty=False) == [DEFAULT_PROMPT], "an empty pipe falls back to the sample"
    f = tmp_path / "p.jsonl"
    f.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    assert prompts(prompt="x", file=str(f), data="x", tty=False) == ["a", "b"], "--file overrides -p and the pipe"


def test_run_json_answers_a_prompt_piped_on_stdin(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """the machine-readable path end to end: a prompt piped on stdin, `btb run --json` writes exactly one JSON
    record whose `prompt` is the piped text (trimmed) and nothing else on stdout."""
    from btb.cli import main

    fx = fixture("tiny_qwen3")
    if not os.path.exists(os.path.join(fx, "tokenizer.json")):
        pytest.skip("the tiny_qwen3 fixture has no tokenizer: these tests run from a checkout")
    monkeypatch.setattr(sys, "stdin", _Stdin("  what is 2 plus 2?  \n", tty=False))
    rc = main(["run", fx, "-d", "cpu", "--new", "2", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out.strip())  # one record, nothing else on stdout
    assert rec["prompt"] == "what is 2 plus 2?" and isinstance(rec["answer"], str)


# --- serve / ollama: the server -----------------------------------------------------------------------------


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


def test_registry_close_stops_the_request_in_flight_before_the_model_closes() -> None:
    """a Ctrl-C mid-answer: the registry's close tells the running generation to stop and closes the engine only
    once the request's handler has released the lock (an engine closed under a running pass is a segfault)"""
    import threading

    from btb.serve import Engine

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

    def request() -> None:
        # the handler holds the lock for the whole request and its loop ends once abort is set
        with reg.lock:
            order.append("running")
            eng.sm.abort.wait(timeout=10)
            order.append("stopped")

    th = threading.Thread(target=request)
    th.start()
    while "running" not in order:
        pass
    reg.close()
    th.join(timeout=10)
    assert order == ["running", "stopped", "closed"] and not reg.loaded
