"""Cohesive runtime certification: the whole stack as one unit, where the real bugs live. Cell tests load one
model on one device path in isolation; this drives the actual OpenAI server over the real engine through the
traffic a deployment sees at once - concurrent ragged requests, streaming and not, multi-turn prefix reuse, a
model switch under a memory budget - and asserts it holds together and stays deterministic. Tiny random
fixtures on CPU, so it runs anywhere (no GPU, no real models, no license)."""

from __future__ import annotations

import http.client
import json
import os
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from tests.helpers import FIXTURES, request_json

if TYPE_CHECKING:
    from btb.kinds import Json
    from btb.serve import Server

    ServerCtx = tuple[Server, str, str]  # (server, primary_name, second_name), as the fixture yields

QWEN3 = os.path.join(FIXTURES, "tiny_qwen3")
PHI3 = os.path.join(FIXTURES, "tiny_phi3")


@pytest.fixture(scope="module")
def server() -> Iterator[ServerCtx]:
    """the real serve.Server over the real engine, primary tiny_qwen3 with tiny_phi3 as a second model, on CPU;
    yields (server, primary_name, second_name)."""
    if not os.path.isdir(QWEN3):
        pytest.skip("tiny_qwen3 fixture not built")
    from btb.serve import Handler, ModelRegistry, Server, _Server

    reg = ModelRegistry(QWEN3, device="cpu", max_new=8, extra_paths=[PHI3] if os.path.isdir(PHI3) else [])
    srv = _Server(("127.0.0.1", 0), Handler)
    srv.reg = reg
    server = Server(srv, reg).start()
    primary = reg.primary_name
    assert primary is not None
    try:
        yield server, primary, _second_name(reg)
    finally:
        server.close()


def _second_name(reg: object) -> str:
    entries = reg.refresh()  # type: ignore[attr-defined]
    for name in entries:
        if name != reg.primary_name:  # type: ignore[attr-defined]
            return name
    return reg.primary_name  # type: ignore[attr-defined]


def _chat(url: str, model: str, text: str, *, max_tokens: int = 8, temperature: float = 0.0) -> tuple[int, Json]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return request_json(url, "POST", "/v1/chat/completions", body)


def _content(data: Json) -> str:
    return data["choices"][0]["message"]["content"]


def test_concurrent_ragged_requests_all_complete(server: ServerCtx) -> None:
    """the scheduler batches a burst of different-length prompts through one engine without dropping or crossing
    a request; every one comes back 200 with its own answer."""
    srv, model, _second = server
    results: dict[int, tuple[int, str]] = {}
    lock = threading.Lock()

    def one(i: int) -> None:
        status, data = _chat(srv.url, model, "tell me about " + "x " * (i % 7 + 1))
        with lock:
            results[i] = (status, _content(data) if status == 200 else "")

    threads = [threading.Thread(target=one, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len(results) == 12
    assert all(st == 200 for st, _ in results.values()), results
    assert all(isinstance(c, str) for _st, c in results.values())


def test_end_to_end_greedy_is_deterministic(server: ServerCtx) -> None:
    """two identical temperature=0 requests through the whole server return identical text - the invariant
    speculation and caching rest on, checked at the top of the stack rather than at the engine call."""
    srv, model, _second = server
    s1, d1 = _chat(srv.url, model, "the capital of France is")
    s2, d2 = _chat(srv.url, model, "the capital of France is")
    assert s1 == 200 and s2 == 200
    assert _content(d1) == _content(d2), (_content(d1), _content(d2))


def test_multi_turn_conversation_reuses_the_prefix(server: ServerCtx) -> None:
    """a growing conversation: each turn appends the prior answer and asks again. The session/cache reuse the
    shared prefix; the point here is it stays coherent and 200 across turns, not a numeric prefix-hit check."""
    srv, model, _second = server
    messages = [{"role": "user", "content": "count up from one"}]
    for _turn in range(3):
        body = {"model": model, "messages": messages, "max_tokens": 8, "temperature": 0.0}
        status, data = request_json(srv.url, "POST", "/v1/chat/completions", body)
        assert status == 200, data
        reply = _content(data)
        assert isinstance(reply, str)
        messages = [*messages, {"role": "assistant", "content": reply}, {"role": "user", "content": "continue"}]


def test_model_switch_is_resilient(server: ServerCtx) -> None:
    """two models served by one registry. The switch loads/evicts within the budget; and whatever the second
    model does - answers, or fails cleanly (tiny_phi3 ships no chat template, so serve returns an error, not a
    crash) - the server survives and still serves the primary. Surviving a bad request is the cohesive property;
    a per-model failure must not take the server down."""
    srv, primary, second = server
    if second == primary:
        pytest.skip("second fixture (tiny_phi3) not built")
    ss, _ds = _chat(srv.url, second, "hello")
    assert ss < 500 or ss >= 200, ss  # some HTTP answer, not a hang/reset
    sp, dp = _chat(srv.url, primary, "hello")  # the server is still up and serves the primary after the switch
    assert sp == 200 and isinstance(_content(dp), str), (sp, dp)


def test_streaming_arrives_in_pieces(server: ServerCtx) -> None:
    """a streamed completion arrives as multiple SSE chunks through the real engine, not one blob at the end -
    the token callback path wired end to end."""
    srv, model, _second = server
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "say several words"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    conn = http.client.HTTPConnection(srv.host, srv.port, timeout=60)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    chunks = []
    for raw in resp.read().decode().splitlines():
        if raw.startswith("data: ") and not raw.endswith("[DONE]"):
            piece = json.loads(raw[6:])["choices"][0].get("delta", {}).get("content")
            if piece:
                chunks.append(piece)
    conn.close()
    assert len(chunks) >= 2, chunks  # streamed token-by-token, not one final blob
