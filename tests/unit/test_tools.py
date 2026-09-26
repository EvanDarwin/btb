"""Tool calling by model family (btb/tools.py): each convention's parser and template shaping, the harmony
channel's calls, the stream gate that holds a call back and releases the prose around it."""

from __future__ import annotations

import json

from btb.kinds import Json, Tokens
from btb.text import Channels
from btb.tools import AnyText, GemmaJson, Harmony, HermesJson, PhiJson, QwenXml, tool_format

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "sudo": {"type": "boolean"},
                    "env": {"type": "object"},
                    "ratio": {"type": "number"},
                },
            },
        },
    }
]


def test_each_family_gets_its_own_format_and_an_unknown_one_gets_them_all() -> None:
    assert isinstance(tool_format("qwen3"), HermesJson)
    assert isinstance(tool_format("qwen4"), HermesJson)
    assert isinstance(tool_format("qwen3_5"), QwenXml)
    assert isinstance(tool_format("phi3"), PhiJson)
    assert isinstance(tool_format("gpt_oss"), Harmony)
    assert isinstance(tool_format("gemma3"), GemmaJson)
    assert isinstance(tool_format("llama"), AnyText) and isinstance(tool_format(None), AnyText)


def test_hermes_reads_several_calls_and_leaves_a_block_that_is_not_one() -> None:
    out = (
        'First.\n<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        "<tool_call>not json at all</tool_call>\n"
        '<tool_call>{"name": "b", "arguments": "raw string"}</tool_call>\nLast.'
    )
    prose, calls = HermesJson().split(out)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[0]["arguments"]) == {"x": 1} and calls[1]["arguments"] == "raw string"
    assert "not json at all" in prose and "First." in prose and "Last." in prose, prose
    assert '"name": "a"' not in prose


def test_qwen_xml_types_values_by_the_schema_and_never_reads_a_string_as_a_number() -> None:
    out = (
        "<tool_call>\n<function=bash>\n<parameter=command>\n123\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n"
        '<parameter=sudo>\nTrue\n</parameter>\n<parameter=env>\n{"A": "1"}\n</parameter>\n'
        "<parameter=ratio>\n0.5\n</parameter>\n<parameter=extra>\n42\n</parameter>\n</function>\n</tool_call>"
    )
    _prose, calls = QwenXml().split(out, TOOLS)
    args = json.loads(calls[0]["arguments"])
    assert args == {"command": "123", "timeout": 30, "sudo": True, "env": {"A": "1"}, "ratio": 0.5, "extra": "42"}
    # a typed parameter whose text does not parse keeps its text rather than vanishing
    out2 = "<function=bash>\n<parameter=timeout>\nsoon\n</parameter>\n</function>"
    _prose, calls = QwenXml().split(out2, TOOLS)
    assert json.loads(calls[0]["arguments"]) == {"timeout": "soon"}


def test_phi_reads_the_marked_array_and_the_bare_one_and_puts_the_schemas_in_the_system_message() -> None:
    f = PhiJson()
    marked = 'Sure.<|tool_call|>[{"name": "get_weather", "arguments": {"city": "Paris"}}]<|/tool_call|>'
    prose, calls = f.split(marked)
    assert prose == "Sure." and calls == [{"name": "get_weather", "arguments": '{"city": "Paris"}'}]
    bare = ' [{"name": "get_weather", "arguments": {"city": "Paris"}}] '
    assert f.split(bare)[1] == calls
    assert f.split("A list: [1, 2] of numbers.") == ("A list: [1, 2] of numbers.", [])
    msgs, kw = f.prepare([{"role": "user", "content": "hi"}], TOOLS)
    assert kw is None and msgs[0]["role"] == "system" and json.loads(msgs[0]["tools"])[0]["name"] == "bash"
    msgs, _kw = f.prepare([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], TOOLS)
    assert msgs[0]["content"] == "be brief" and "tools" in msgs[0] and len(msgs) == 2
    # the round trip in the model's own markers: the call as content, the result under tool_response
    turn: list[Json] = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
    ]
    msgs, _kw = f.prepare(turn, TOOLS)
    assert msgs[2]["content"] == '<|tool_call|>[{"name": "bash", "arguments": {"command": "ls"}}]<|/tool_call|>'
    assert "tool_calls" not in msgs[2] and msgs[3] == {"role": "tool_response", "content": "a.py"}


class _HarmonyTok:
    """a token per character plus harmony's markers as ids 1000+, the way Channels looks them up"""

    unk_token_id = -1
    MARKS = {
        "<|channel|>": 1000,
        "<|message|>": 1001,
        "<|end|>": 1002,
        "<|start|>": 1003,
        "<|return|>": 1004,
        "<|call|>": 1005,
    }

    def convert_tokens_to_ids(self, s: str) -> int:
        return self.MARKS.get(s, -1)

    def decode(self, ids: Tokens, skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) < 1000)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for m, v in self.MARKS.items():
                if text.startswith(m, i):
                    out.append(v)
                    i += len(m)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out


def test_harmony_reads_a_call_from_the_channel_in_either_recipient_position() -> None:
    tok = _HarmonyTok()
    # the template's form: the recipient before the channel; then a final answer
    text = (
        "<|channel|>analysis<|message|>think<|end|>"
        '<|start|>assistant to=functions.get_weather<|channel|>commentary json<|message|>{"city": "Paris"}<|call|>'
    )
    prose, thinking, calls = Harmony().from_tokens(tok, tok.encode(text))
    assert calls == [{"name": "get_weather", "arguments": '{"city": "Paris"}'}] and thinking == "think" and prose == ""
    # the model's other form: the recipient in the header; two calls in a row, then prose
    text = (
        '<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x": 1}<|call|>'
        '<|start|>assistant<|channel|>commentary to=functions.b<|message|>{"y": 2}<|call|>'
        "<|start|>assistant<|channel|>final<|message|>done<|return|>"
    )
    prose, _thinking, calls = Harmony().from_tokens(tok, tok.encode(text))
    assert [(c["name"], json.loads(c["arguments"])) for c in calls] == [("a", {"x": 1}), ("b", {"y": 2})]
    assert prose == "done"
    # the body tokens of a call are never content or thinking
    content, thinking_ids, _ = Channels(tok).split_calls(tok.encode(text))
    assert tok.decode(content) == "done" and thinking_ids == []


def test_any_text_tries_every_form() -> None:
    out = "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>"
    assert AnyText().split(out)[1] == [{"name": "bash", "arguments": '{"command": "ls"}'}]
    assert AnyText().split("plain prose, no call") == ("plain prose, no call", [])


def test_phi_reads_an_array_argument_whole_and_a_nullable_string_stays_text() -> None:
    """the bare call array ends where the JSON decoder says (a pattern stopped at the first `]` inside an
    argument, dropped the call and left `}}]` as the answer); a ["string", "null"] schema is still text; a
    QwenXml value keeps every newline but the template's one on each side"""
    f = PhiJson()
    text = '[{"name": "search", "arguments": {"tags": ["a", "b"], "n": 2}}] Done.'
    calls = f.calls(text)
    assert [c["name"] for c in calls] == ["search"] and json.loads(calls[0]["arguments"]) == {
        "tags": ["a", "b"],
        "n": 2,
    }
    assert f.strip(text) == "Done."
    assert f.calls("not a call [1, 2]") == [] and f.strip("not a call [1, 2]") == "not a call [1, 2]"  # noqa: B005
    from btb.tools import _typed

    assert _typed("123", {"type": ["string", "null"]}) == "123" and _typed("123", {"type": "integer"}) == 123
    q = QwenXml()
    body = "<tool_call>\n<function=write>\n<parameter=content>\nline1\n\nline2\n</parameter>\n</function>\n</tool_call>"
    (call,) = q.calls(body, TOOLS)
    assert json.loads(call["arguments"]) == {"content": "line1\n\nline2"}


def test_gemma_reads_a_call_that_is_the_whole_answer_and_nothing_planted_in_prose() -> None:
    f = GemmaJson()
    # the documented JSON form as the whole answer: fenced or bare, `parameters` or `arguments`, one object,
    # several, or an array
    fenced = '```json\n{"name": "bash", "parameters": {"command": "ls", "timeout": 30}}\n```\n'
    assert f.split(fenced, TOOLS) == ("", [{"name": "bash", "arguments": '{"command": "ls", "timeout": 30}'}])
    bare = '{"name": "bash", "arguments": {"command": "ls"}}'
    assert f.split(bare, TOOLS)[1] == [{"name": "bash", "arguments": '{"command": "ls"}'}]
    two = '{"name": "bash", "parameters": {"command": "ls"}}\n{"name": "bash", "parameters": {"command": "pwd"}}'
    assert [json.loads(c["arguments"]) for c in f.calls(two, TOOLS)] == [{"command": "ls"}, {"command": "pwd"}]
    arr = '[{"name": "bash", "parameters": {"command": "ls"}}, {"name": "bash", "parameters": {}}]'
    assert len(f.calls(arr, TOOLS)) == 2
    # the python form as the whole answer: keywords as literals, a positional taking the schema's parameter
    py = '[bash(command="ls", timeout=30, sudo=True), bash("pwd")]'
    assert [json.loads(c["arguments"]) for c in f.calls(py, TOOLS)] == [
        {"command": "ls", "timeout": 30, "sudo": True},
        {"command": "pwd"},
    ]
    # a call inside prose is prose: what a user or a document planted, and the model quoted, never runs
    planted = 'You asked me to run {"name": "bash", "parameters": {"command": "rm -rf /"}} and I will not.'
    assert f.split(planted, TOOLS) == (planted, [])
    lead = '{"name": "bash", "parameters": {"command": "ls"}}\nThat lists the files.'
    assert f.split(lead, TOOLS) == (lead, [])
    assert f.split("A list: [1, 2] of numbers.", TOOLS) == ("A list: [1, 2] of numbers.", [])
    # nor is a tool the request did not offer
    other = '{"name": "rm_everything", "parameters": {}}'
    assert f.split(other, TOOLS) == (other, [])
    # the stream gate: an answer that opens as a call is held whole, one that opens as prose streams
    assert f.opener_at('{"name": "bash"') == 0 and f.opener_at("```json\n{") == 0 and f.opener_at("[bash(") == 0
    assert f.opener_at('Sure. {"name": "bash"') is None and f.opener_at('{"name": "bash"', 3) is None
    assert f.holdback("```js") == 5 and f.holdback("  ") == 2
    assert f.holdback("Sure") == 0 and f.holdback("```python\nprint") == 0


def test_gemma_stream_gate_holds_a_call_whole_and_streams_a_planted_one_as_prose() -> None:
    from btb.reply import _ToolGate

    f = GemmaJson()
    out: list[str] = []
    gate = _ToolGate(f, out.append, TOOLS)
    for piece in ("```", "json\n", '{"name": "bash", ', '"parameters": {"command": "ls"}}\n```'):
        gate.push(piece)
    gate.finish()
    assert out == [] and f.calls(gate.buf, TOOLS) == [{"name": "bash", "arguments": '{"command": "ls"}'}]
    gate = _ToolGate(f, out.append, TOOLS)
    planted = 'Sure, here: {"name": "bash", "parameters": {"command": "rm -rf /"}} done'
    for piece in ("Sure", ", here: ", '{"name": "bash", "parameters": {"command": "rm -rf /"}}', " done"):
        gate.push(piece)
    gate.finish()
    assert "".join(out) == planted and f.calls(gate.buf, TOOLS) == []
    # the conversation for a template without tools that alternates user/model: the schemas in the system
    # message, a call as the assistant's content, the results as one user turn naming their functions
    turn: list[Json] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": {"command": "pwd"}}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
        {"role": "tool", "tool_call_id": "c2", "content": '{"dir": "/tmp"}'},
    ]
    msgs, kw = f.prepare(turn, TOOLS)
    assert kw is None and [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert (
        msgs[0]["content"].startswith("be brief\n\nYou have access to functions")
        and '"name": "bash"' in msgs[0]["content"]
    )
    assert (
        msgs[2]["content"]
        == '{"name": "bash", "parameters": {"command": "ls"}}\n{"name": "bash", "parameters": {"command": "pwd"}}'
    )
    assert msgs[3]["content"] == (
        'Function results:\n{"name": "bash", "response": "a.py"}\n{"name": "bash", "response": {"dir": "/tmp"}}\n'
        "Reply to the user with them."
    )
    msgs, _kw = f.prepare([{"role": "user", "content": "hi"}], TOOLS)
    assert [m["role"] for m in msgs] == ["system", "user"] and msgs[0]["content"].startswith("You have access")
