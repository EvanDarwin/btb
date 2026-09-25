"""The pi harness (pi.dev): the OpenAI endpoint's tool calling, over a stubbed model on the real server, and
the provider entry btb writes into pi's models.json."""

from __future__ import annotations

import json
from pathlib import Path

from btb.serve import _content_text, _for_template, _write_pi_config
from btb.tools import tool_calls_of
from tests.helpers import post_chat, stub_server

TOOL_OUT = 'Reading it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>'
REQ = {
    "model": "fake",
    "messages": [{"role": "user", "content": "read main.py"}],
    "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
}


def test_openai_returns_tool_calls_when_the_model_emits_one() -> None:
    server = stub_server(TOOL_OUT)
    try:
        status, data = post_chat(server, REQ)
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
    server = stub_server(TOOL_OUT)
    try:
        status, data = post_chat(server, {**REQ, "stream": True})
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


def test_openai_tools_stream_prose_in_pieces() -> None:
    # with tools present the prose must still stream token by token, not arrive in one decode-whole chunk
    server = stub_server(out="Line one.\nLine two.\nLine three.\n")
    try:
        status, data = post_chat(server, {**REQ, "stream": True})
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
    server = stub_server(out="Just an answer.")
    try:
        status, data = post_chat(server, {"model": "fake", "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200, data
        choice = json.loads(data)["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "Just an answer."
        assert "tool_calls" not in choice["message"]
    finally:
        server.close()


def test_openai_streams_usage_when_the_client_asks_for_it() -> None:
    server = stub_server(TOOL_OUT)  # REQ carries tools, so this exercises the tools stream path
    try:
        status, data = post_chat(server, {**REQ, "stream": True, "stream_options": {"include_usage": True}})
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


def test_the_pi_provider_carries_the_servers_key() -> None:
    from btb.serve import _pi_provider

    assert _pi_provider("http://127.0.0.1:8000/v1", ["a"], "k")["apiKey"] == "k"
    assert _pi_provider("http://127.0.0.1:8000/v1", ["a"])["apiKey"] == "btb"


def test_openai_streams_the_prose_after_a_call_and_never_a_partial_opener() -> None:
    # the gate holds from the opener, then releases the prose that followed the call once the block is struck
    out = (
        'Reading it.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>\nDone reading.'
    )
    server = stub_server(out=out)
    try:
        status, data = post_chat(server, {**REQ, "stream": True})
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
