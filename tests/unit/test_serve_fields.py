# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a request may ask of its answer beyond the sampling: OpenAI's `n`, `logprobs`, `stop`, `logit_bias`, the
penalties and `response_format`, and Ollama's `format` and `options.stop` - served from the tiny fixture on the
CPU, plus the processors they map to on their own."""

from __future__ import annotations

import http.client
import json
import os
import re
from collections.abc import Iterator

import pytest
import torch

from btb.engine.constrain import JsonObjectPrefix, LogitBias, Penalties, PrefixConstraint
from btb.kinds import Json
from btb.serve import Server
from tests.helpers import CharTokenizer, fixture, request, request_json

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def served() -> Iterator[tuple[Server, str]]:
    from btb.serve import start

    fx = fixture("tiny_qwen3")
    only = f"^{re.escape(os.path.basename(fx))}$"
    server = start(None, host="127.0.0.1", port=0, device="cpu", extra_paths=[fx], pattern=only).start()
    try:
        ((name, _entry),) = server.registry.refresh().items()
        yield server, name
    finally:
        server.close()


def chat(served: tuple[Server, str], **fields: object) -> Json:
    server, name = served
    code, body = request_json(server.url, "POST", "/v1/chat/completions", {"model": name, "messages": MSGS, **fields})
    assert code == 200, body
    return body


def events(served: tuple[Server, str], **fields: object) -> list[Json]:
    """a streamed chat completion's chunks, in order"""
    server, name = served
    body = {"model": name, "messages": MSGS, "stream": True, **fields}
    code, _, text = request(server.url, "POST", "/v1/chat/completions", body)
    assert code == 200, text
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: ") :]))
    return out


def test_logprobs_come_a_token_with_their_alternatives(served: tuple[Server, str]) -> None:
    body = chat(served, max_tokens=6, logprobs=True, top_logprobs=2)
    (choice,) = body["choices"]
    content = choice["logprobs"]["content"]
    assert len(content) == body["usage"]["completion_tokens"]
    for e in content:
        assert e["logprob"] <= 0 and e["bytes"] == list(e["token"].encode("utf-8"))
        assert len(e["top_logprobs"]) == 2 and e["top_logprobs"][0]["logprob"] >= e["logprob"] - 1e-6
    assert "".join(e["token"] for e in content) == choice["message"]["content"]
    assert "logprobs" not in chat(served, max_tokens=2)["choices"][0]


def test_streamed_logprobs_come_before_the_finish(served: tuple[Server, str]) -> None:
    chunks = events(served, max_tokens=5, logprobs=True)
    carried = [c["choices"][0]["logprobs"] for c in chunks if c["choices"] and "logprobs" in c["choices"][0]]
    assert len(carried) == 1 and len(carried[0]["content"]) >= 1
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_n_answers_are_drawn_apart(served: tuple[Server, str]) -> None:
    body = chat(served, max_tokens=8, n=3, temperature=0.9, seed=7, logprobs=True)
    choices = body["choices"]
    assert [c["index"] for c in choices] == [0, 1, 2]
    assert len({c["message"]["content"] for c in choices}) > 1
    assert all(c["finish_reason"] in ("stop", "length") for c in choices)
    assert body["usage"]["completion_tokens"] == sum(len(c["logprobs"]["content"]) for c in choices)
    assert chat(served, max_tokens=8, n=3, temperature=0.9, seed=7)["choices"] == [
        {k: v for k, v in c.items() if k != "logprobs"} for c in choices
    ]


def test_n_answers_stream_by_index(served: tuple[Server, str]) -> None:
    chunks = events(served, max_tokens=6, n=2, temperature=0.9, seed=3)
    text: dict[int, str] = {0: "", 1: ""}
    finished = set()
    for c in chunks:
        for ch in c["choices"]:
            text[ch["index"]] += ch["delta"].get("content", "")
            if ch["finish_reason"]:
                finished.add(ch["index"])
    assert finished == {0, 1}
    whole = chat(served, max_tokens=6, n=2, temperature=0.9, seed=3)["choices"]
    assert [text[0], text[1]] == [c["message"]["content"] for c in whole]


@pytest.mark.parametrize("stream", [False, True])
def test_a_stop_string_ends_the_answer_before_it(served: tuple[Server, str], stream: bool) -> None:
    full = chat(served, max_tokens=24)["choices"][0]["message"]["content"]
    assert len(full) > 6, full
    stop = full[3:6]
    want = full[: full.find(stop)]
    if stream:
        chunks = events(served, max_tokens=24, stop=[stop])
        got = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        fin = chunks[-1]["choices"][0]["finish_reason"]
    else:
        choice = chat(served, max_tokens=24, stop=stop)["choices"][0]
        got, fin = choice["message"]["content"], choice["finish_reason"]
    assert got == want and fin == "stop"


def test_logit_bias_decides_the_pick(served: tuple[Server, str]) -> None:
    body = chat(served, max_tokens=3, logit_bias={"65": 100}, logprobs=True)
    assert all(
        e["token"] == body["choices"][0]["logprobs"]["content"][0]["token"]
        for e in body["choices"][0]["logprobs"]["content"]
    )
    server, _ = served
    tok = next(iter(server.registry.loaded.values())).tok
    assert body["choices"][0]["logprobs"]["content"][0]["token"] == tok.decode([65])


def test_json_mode_holds_the_answer_to_a_json_object(served: tuple[Server, str]) -> None:
    choice = chat(served, max_tokens=40, response_format={"type": "json_object"})["choices"][0]
    text = choice["message"]["content"]
    assert JsonObjectPrefix().feed(text) is not None, text
    if choice["finish_reason"] == "stop":
        assert isinstance(json.loads(text), dict)


def test_ollama_takes_format_and_stop(served: tuple[Server, str]) -> None:
    server, name = served
    code, body = request_json(
        server.url,
        "POST",
        "/api/generate",
        {"model": name, "prompt": "hi", "stream": False, "format": "json", "options": {"num_predict": 20}},
    )
    assert code == 200 and JsonObjectPrefix().feed(body["response"]) is not None, body
    code, plain = request_json(
        server.url,
        "POST",
        "/api/generate",
        {"model": name, "prompt": "hi", "stream": False, "options": {"num_predict": 20}},
    )
    stop = plain["response"][2:4]
    code, cut = request_json(
        server.url,
        "POST",
        "/api/generate",
        {"model": name, "prompt": "hi", "stream": False, "options": {"num_predict": 20, "stop": [stop]}},
    )
    assert code == 200 and cut["response"] == plain["response"][: plain["response"].find(stop)]


@pytest.mark.parametrize(
    "fields,word",
    [
        ({"top_logprobs": 2}, "top_logprobs="),
        ({"logprobs": True, "top_logprobs": 21}, "top_logprobs="),
        ({"stop": ["a", "b", "c", "d", "e"]}, "stop="),
        ({"stop": [""]}, "stop="),
        ({"response_format": {"type": "json_schema", "json_schema": {}}}, "response_format="),
        ({"logit_bias": {"x": 1}}, "logit_bias="),
        ({"logit_bias": {"1": 101}}, "logit_bias="),
        ({"presence_penalty": 3}, "presence_penalty="),
        ({"frequency_penalty": "a"}, "frequency_penalty="),
        ({"n": 0}, "n="),
    ],
)
def test_a_field_out_of_range_is_a_400(served: tuple[Server, str], fields: Json, word: str) -> None:
    server, name = served
    code, body = request_json(server.url, "POST", "/v1/chat/completions", {"model": name, "messages": MSGS, **fields})
    assert code == 400 and word in body["error"], (fields, body)


# -- the processors on their own -----------------------------------------------------------------------------------


def test_penalties_follow_openais_formula() -> None:
    lg = torch.zeros(6)
    out = Penalties(2, presence=0.5, frequency=0.25)([9, 9, 1, 1, 1, 4], lg)
    assert out.tolist() == [0.0, -1.25, 0.0, 0.0, -0.75, 0.0]
    assert torch.equal(Penalties(2, 1.0, 1.0)([1, 2], lg), lg)


def test_logit_bias_adds_to_the_chosen_tokens() -> None:
    out = LogitBias({1: 5.0, 3: -100.0})([], torch.zeros(4))
    assert out.tolist() == [0.0, 5.0, 0.0, -100.0]


class _Chars(CharTokenizer):
    """the char tokenizer with a vocabulary: token i is chr(i)"""

    all_special_ids: list[int] = []

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


def test_json_mode_keeps_every_prefix_valid_and_ends_at_a_stop() -> None:
    """random logits over printable characters, steered only by the processor: every text it lets through is a
    JSON object's prefix, and once the object closes only the stop token is left"""
    stop = 3
    proc = PrefixConstraint(_Chars(), 2, [stop], JsonObjectPrefix(), width=8, scan=128)
    gen = torch.Generator().manual_seed(0)
    ids = [1, 2]
    for _ in range(400):
        lg = torch.randn(128, generator=gen)
        lg[ord("}")] += 1.5  # a nudge so the object closes within the run
        t = int(proc(ids, lg).argmax())
        if t == stop:
            break
        ids.append(t)
        assert JsonObjectPrefix().feed("".join(map(chr, ids[2:]))) is not None
    text = "".join(map(chr, ids[2:]))
    assert isinstance(json.loads(text), dict), text
    after = proc(ids, torch.zeros(128))
    assert torch.isfinite(after).nonzero().flatten().tolist() == [stop]


class _Digits:
    """a language of its own: two or more digits, complete from the second"""

    def __init__(self, n: int = 0) -> None:
        self.n = n

    @property
    def complete(self) -> bool:
        return self.n >= 2

    def feed(self, text: str) -> _Digits | None:
        return _Digits(self.n + len(text)) if text.isdigit() else None


def test_a_prefix_constraint_takes_any_recognizer() -> None:
    """the constraint knows no JSON: any `TextPrefix` holds the answer to its language, a stop token allowed once
    the text is complete and not before"""
    stop = 3
    proc = PrefixConstraint(_Chars(), 0, [stop], _Digits(), width=128, scan=128)
    allowed = torch.isfinite(proc([ord("7")], torch.zeros(128))).nonzero().flatten().tolist()
    assert allowed == [ord(c) for c in "0123456789"]
    allowed = torch.isfinite(proc([ord("7"), ord("1")], torch.zeros(128))).nonzero().flatten().tolist()
    assert allowed == [stop, *(ord(c) for c in "0123456789")]


def _leave_after(served: tuple[Server, str], chunks: int, **fields: object) -> None:
    """a streamed request whose client reads `chunks` chunks and goes"""
    server, name = served
    host, port = server.url.split("//", 1)[1].split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=30)
    body = {"model": name, "messages": MSGS, "stream": True, **fields}
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


def test_requests_cut_short_leave_the_session_to_answer_the_next_whole(served: tuple[Server, str]) -> None:
    """the server's session goes on across requests; one cut short - a client gone mid-answer at n=2 and at n=1,
    every row at a stop string - leaves it holding what it can reuse, so the next request answers as it would
    have with none of them before it"""
    ref = chat(served, max_tokens=8)["choices"][0]["message"]["content"]
    _leave_after(served, 3, n=2, max_tokens=40)
    _leave_after(served, 3, max_tokens=40)
    first = chat(served, max_tokens=8)["choices"][0]["message"]["content"]
    cut = chat(served, n=2, max_tokens=40, stop=[first[1:3]] if len(first) > 2 else None)
    assert all(c["finish_reason"] in ("stop", "length") for c in cut["choices"])
    assert first == ref
    assert chat(served, max_tokens=8)["choices"][0]["message"]["content"] == ref
