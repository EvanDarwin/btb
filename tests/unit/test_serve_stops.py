# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Stop strings, tool calls and several answers over the server's routes: the same request answered alike streamed
or not, and the decode stopped once nothing more of it is wanted - a stop string in every row, a row ended at its
stop token beside them, a client gone while a call is still being buffered."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Sequence

import pytest

from btb.engine.hooks import LogitsProcessor
from btb.sampling import Sampling
from btb.serve import Handler, Message
from tests.helpers import FakeEngine, FakeRun, post_chat, request, stub_server

EOS = 1
CALL = 'Reading it.\n<tool_call>\n{"name":"read_file","arguments":{"path":"main.py"}}\n</tool_call>'
TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}}}]


class Scripted(FakeEngine):
    """rows decoded a step at a time: row r's script (a character a token, EOS ending it), then `runaway` to the
    cap; a step checks the request's cancel, a row leaves once `until(row)` says so, as the engine's loops do;
    `steps` counts the steps and `drawn[r]` row r's tokens"""

    eos = (EOS,)

    def __init__(self, scripts: Sequence[str | list[int]], cap: int = 300, pause: float = 0.0) -> None:
        super().__init__("")
        self.scripts = [[ord(c) for c in s] if isinstance(s, str) else list(s) for s in scripts]
        self.cap, self.pause = cap, pause
        self.steps = 0
        self.drawn: list[int] = []
        self.done = threading.Event()

    def run_rows(
        self,
        ids: Sequence[int],
        n: int,
        max_new: int | None,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        on_token: Callable[[int, int], object] | None = None,
        until: Callable[[int], object] | None = None,
        cancel: threading.Event | None = None,
    ) -> tuple[list[list[int]], list[bool], dict[str, int], None]:
        try:
            return self._rows(n, max_new, on_token, until, cancel)
        finally:
            self.done.set()

    def _rows(
        self,
        n: int,
        max_new: int | None,
        on_token: Callable[[int, int], object] | None,
        until: Callable[[int], object] | None,
        cancel: threading.Event | None,
    ) -> tuple[list[list[int]], list[bool], dict[str, int], None]:
        cap = min(self.cap, max_new or self.cap)
        outs: list[list[int]] = [[] for _ in range(n)]
        self.drawn = [0] * n
        live = list(range(n))
        for k in range(cap):
            if (cancel is not None and cancel.is_set()) or not live:
                break
            self.steps += 1
            for r in list(live):
                sc = self.scripts[r % len(self.scripts)]
                t = sc[k] if k < len(sc) else ord("z")
                outs[r].append(t)
                self.drawn[r] += 1
                if on_token is not None:
                    on_token(r, t)
                if t == EOS or (until is not None and until(r)):
                    live.remove(r)
            time.sleep(self.pause)
        rows = [[t for t in o if t != EOS] for o in outs]
        return rows, [bool(o) and o[-1] == EOS for o in outs], {"cap": cap, "forwards": self.steps}, None

    def run(
        self,
        messages: Sequence[Message],
        max_new: int | None,
        on_token: Callable[[int], object] | None = None,
        ids: Sequence[int] | None = None,
        sampling: Sampling | None = None,
        cancel: threading.Event | None = None,
        **_hooks: object,
    ) -> FakeRun:
        rows, _ended, c, _ = self.run_rows(
            ids or [1, 2, 3],
            1,
            max_new,
            on_token=None if on_token is None else (lambda _r, t: on_token(t)),
            cancel=cancel,
        )
        return list(ids or [1, 2, 3]), rows[0], c, None

    def text(self, toks: Sequence[int]) -> tuple[str, str]:
        return "".join(chr(t) for t in toks if t != EOS), ""


def _body(stream: bool, **kw: object) -> dict[str, object]:
    return {"model": "fake", "messages": [{"role": "user", "content": "go"}], "stream": stream, **kw}


def _answers(stream: bool, data: str) -> list[tuple[str, list[str], str | None]]:
    """each choice's (content, tool call names and arguments, finish reason), streamed or not: the content trimmed,
    as a stream sends the prose before a call as it comes and cannot take its trailing space back"""
    if not stream:
        out = []
        for ch in json.loads(data)["choices"]:
            m = ch["message"]
            calls = [c["function"]["name"] + c["function"]["arguments"] for c in m.get("tool_calls") or ()]
            out.append(((m["content"] or "").strip(), calls, ch["finish_reason"]))
        return out
    text: dict[int, str] = {}
    called: dict[int, list[str]] = {}
    finish: dict[int, str | None] = {}
    for line in data.splitlines():
        if not line.startswith("data: {"):
            continue
        for ch in json.loads(line[6:])["choices"]:
            i, d = ch["index"], ch["delta"]
            text[i] = text.get(i, "") + (d.get("content") or "")
            called.setdefault(i, []).extend(
                c["function"]["name"] + c["function"]["arguments"] for c in d.get("tool_calls") or ()
            )
            finish[i] = ch["finish_reason"] or finish.get(i)
    return [(text.get(i, "").strip(), called.get(i, []), finish.get(i)) for i in sorted(text.keys() | called.keys())]


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("stop", [None, ["</tool_call>"], ["}}"], ["<tool_call>"], ["\n"]])
def test_a_stop_string_and_a_tool_call_answer_alike_streamed_or_not(stop: list[str] | None, n: int) -> None:
    """the stop string cuts the answer first and the calls are read from what is left, on every route: a stop
    before or inside a call leaves no call, and no call is ever sent twice (as its text and as a call)"""
    answers = []
    for stream in (False, True):
        server = stub_server(engine=Scripted([CALL + chr(EOS)]))
        try:
            body = _body(stream, tools=TOOLS, n=n, **({"stop": stop} if stop else {}))
            status, data = post_chat(server, body)
        finally:
            server.close()
        assert status == 200, data
        answers.append(_answers(stream, data))
    assert answers[0] == answers[1]
    for content, calls, finish in answers[0]:
        assert not (calls and "<tool_call>" in content), "a call sent as text and as a call"
        if stop is None:
            assert calls and finish == "tool_calls" and content == "Reading it."
        else:
            assert not calls and finish == "stop"


def test_rows_all_at_a_stop_string_or_their_stop_token_end_the_decode() -> None:
    """a row ended at its stop token wants nothing more: once every other row has hit a stop string the decode
    stops, streamed or not, and the usage counts the tokens up to each stop"""
    for stream in (False, True):
        eng = Scripted(["hi" + chr(EOS), "abSTOP"], pause=0.002)  # a step's time: the decode runs ahead
        server = stub_server(engine=eng)
        try:
            status, data = post_chat(server, _body(stream, n=2, stop=["STOP"], stream_options={"include_usage": True}))
        finally:
            server.close()
        assert status == 200, data
        assert eng.steps == 6 and eng.drawn == [3, 6], (stream, eng.steps, eng.drawn)
        assert [a[0] for a in _answers(stream, data)] == ["hi", "ab"]
        usage = json.loads(data)["usage"] if not stream else json.loads(data.split("data: ")[-2])["usage"]
        assert usage["completion_tokens"] == 2 + 6, usage


def _leave_after(server: object, body: dict[str, object], chunks: int) -> None:
    """a streamed request whose client reads chunks chunks and goes"""
    c = http.client.HTTPConnection(server.host, server.port, timeout=10)  # type: ignore[attr-defined]
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200
    got = 0
    while got < chunks:
        line = r.fp.readline()
        assert line, "the stream ended early"
        got += line.startswith(b"data: {")
    r.close()
    c.close()


@pytest.mark.timing
@pytest.mark.parametrize(
    "scripts,extra,chunks",
    [
        # row 0 ended at its stop token, then the client goes: row 1's writes fail, row 0 wants nothing more
        (["hi" + chr(EOS), "a" * 400], {"n": 2}, 6),
        # the prose sent, then the client goes while the call is buffered and nothing is written
        (['Hi.\n<tool_call>\n{"name":"read_file","arguments":{"path":"' + "a" * 400], {"tools": TOOLS}, 4),
    ],
)
def test_a_client_gone_stops_the_decode(
    scripts: list[str], extra: dict[str, object], chunks: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a client that leaves mid-answer stops the decode at its next step: beside a row already ended, or while a
    tool call is buffered and nothing is being written (the keepalive finds it gone)"""
    monkeypatch.setattr(Handler, "KEEPALIVE", 0.05)
    eng = Scripted(scripts, cap=400, pause=0.005)
    server = stub_server(engine=eng)
    try:
        _leave_after(server, _body(True, **extra), chunks)
        assert eng.done.wait(10)
        assert eng.steps < 300, eng.steps
    finally:
        server.close()


def test_a_row_cut_to_nothing_is_still_the_assistants() -> None:
    server = stub_server(engine=Scripted(["STOP and more"]))
    try:
        status, data = post_chat(server, _body(True, stop=["STOP"]))
    finally:
        server.close()
    assert status == 200, data
    deltas = [json.loads(ln[6:])["choices"][0]["delta"] for ln in data.splitlines() if ln.startswith("data: {")]
    assert deltas and deltas[0].get("role") == "assistant"


def test_ollama_names_a_stop_string_on_the_last_token_a_stop() -> None:
    server = stub_server(engine=Scripted(["abSTOP and more"]))
    try:
        body = {
            "model": "fake",
            "messages": [{"role": "user", "content": "go"}],
            "stream": False,
            "options": {"num_predict": 6, "stop": ["STOP"]},
        }
        status, _, data = request(server.url, "POST", "/api/chat", body)
    finally:
        server.close()
    assert status == 200, data
    got = json.loads(data)
    assert got["message"]["content"] == "ab" and got["done_reason"] == "stop"


LONG_ROW = "the other row goes on to its own end" + chr(EOS)


@pytest.mark.parametrize("stream", [False, True])
def test_one_row_at_its_stop_string_leaves_the_others_decoding(stream: bool) -> None:
    """row 0 comes to a stop string at its sixth token; row 1 has thirty more to go to its stop token: row 1 is
    answered whole, row 0 cut and out of the batch at the step that cut it, and the decode ends once both are
    done, not at the cap"""
    eng = Scripted(["abSTOP" + "z" * 200, LONG_ROW], pause=0.002)
    server = stub_server(engine=eng)
    try:
        status, data = post_chat(server, _body(stream, n=2, stop=["STOP"]))
    finally:
        server.close()
    assert status == 200, data
    (c0, _, f0), (c1, _, f1) = _answers(stream, data)
    assert (c0, f0) == ("ab", "stop") and (c1, f1) == (LONG_ROW[:-1], "stop")
    assert eng.drawn == [6, len(LONG_ROW)] and eng.steps == len(LONG_ROW), (eng.drawn, eng.steps)


def test_a_request_stopped_early_leaves_the_next_one_whole() -> None:
    """a request whose every row came to its stop string stopped the decode; the next request, with none, decodes
    to its own end - the stop is not carried over to it"""
    eng = Scripted(["abSTOP" + "c" * 30 + chr(EOS), "xySTOP" + "d" * 30 + chr(EOS)], pause=0.002)
    server = stub_server(engine=eng)
    try:
        for stream in (True, False):
            status, data = post_chat(server, _body(stream, n=2, stop=["STOP"]))
            assert status == 200, data
            assert [a[0] for a in _answers(stream, data)] == ["ab", "xy"]
            status, data = post_chat(server, _body(stream, n=2))
            assert status == 200, data
            assert [a[0] for a in _answers(stream, data)] == ["abSTOP" + "c" * 30, "xySTOP" + "d" * 30]
    finally:
        server.close()


# one request, every way (docs/streaming.md): each row's script, and what its answer must be - its content, its
# finish, the tokens its usage counts (up to its stop string), and the tokens the decode drew for it (the step its
# stop came at: a row leaves there)
X = chr(EOS)
WAYS = {
    "its stop token": [("hello there" + X, "hello there", "stop", 11, 12)],
    "a stop string": [("say STOP then more" + X, "say", "stop", 8, 8)],
    "a stop string at once": [("STOP and on", "", "stop", 4, 4)],
    "a stop string's head held back, then let go": [("aSTxSTOP", "aSTx", "stop", 8, 8)],
    "the cap": [("runs on", "runs onzzzzz", "length", 12, 12)],
    "rows apart": [
        ("ab" + X, "ab", "stop", 2, 3),
        ("cdSTOPef", "cd", "stop", 6, 6),
        ("gh", "ghzzzzzzzzzz", "length", 12, 12),
    ],
}


def _ollama(stream: bool, data: str) -> tuple[str, str, int]:
    """an Ollama chat's (content, done reason, eval count), streamed or not"""
    lines = [json.loads(ln) for ln in data.splitlines() if ln.strip()] if stream else [json.loads(data)]
    content = "".join(ln["message"]["content"] for ln in lines)
    return content.strip(), lines[-1]["done_reason"], lines[-1]["eval_count"]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("way", WAYS)
def test_one_request_answered_alike_every_way(way: str, n: int) -> None:
    """the same request streamed and whole, one answer or three, on OpenAI's route and Ollama's: the same content,
    finish and usage, and each row drawn up to the step its stop came at and no further"""
    rows = [WAYS[way][r % len(WAYS[way])] for r in range(n)]
    want: list[tuple[str, list[str], str | None]] = [(c, [], f) for _s, c, f, _u, _d in rows]
    for stream in (False, True):
        eng = Scripted([r[0] for r in rows])
        server = stub_server(engine=eng)
        try:
            body = _body(stream, n=n, stop=["STOP"], max_tokens=12, stream_options={"include_usage": True})
            status, data = post_chat(server, body)
            assert status == 200, data
            assert _answers(stream, data) == want, (stream, data)
            usage = json.loads(data)["usage"] if not stream else json.loads(data.split("data: ")[-2])["usage"]
            assert usage["completion_tokens"] == sum(r[3] for r in rows), (stream, usage)
            assert eng.drawn == [r[4] for r in rows], (stream, eng.drawn)
            if n == 1:
                body = {
                    "model": "fake",
                    "messages": [{"role": "user", "content": "go"}],
                    "stream": stream,
                    "options": {"num_predict": 12, "stop": ["STOP"]},
                }
                status, _, data = request(server.url, "POST", "/api/chat", body)
                assert status == 200, data
                assert _ollama(stream, data) == (rows[0][1], rows[0][2], rows[0][3]), (stream, data)
                assert eng.drawn == [rows[0][4]], (stream, eng.drawn)
        finally:
            server.close()
