# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Tool calling by model family. A `ToolFormat` knows how its family's chat template takes the tool schemas
(`prepare`), where a call sits in the answer - a text block the stream holds back from its opener, or gpt-oss's
harmony channel, which never reaches the content at all - and turns it into OpenAI's shape: `{name, arguments}`
with `arguments` a JSON string. `tool_format(kind)` picks the family's; an unknown family gets every text form."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, TypedDict

from .kinds import FamilyKind, Json, Tokens
from .text import Channels, answer


class ToolCall(TypedDict):
    name: str
    arguments: str


def _schema_of(tools: Any, name: str) -> Json:
    """the named tool's `parameters.properties`, or {} when the request carries no schema for it"""
    for t in tools or ():
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            params = fn.get("parameters") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            return props if isinstance(props, dict) else {}
    return {}


def _typed(value: str, schema: Json | None) -> Any:
    """A parameter's text as the value its schema declares: a string stays the text it is (never re-read as a
    number or a bool); the other JSON types parse, and fall back to the text where they do not."""
    kind = (schema or {}).get("type")
    kinds = kind if isinstance(kind, list) else [kind]  # a nullable string is ["string", "null"]: still text
    if kind is None or "string" in kinds:
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        low = value.strip().lower()
        if "boolean" in kinds and low in ("true", "false"):
            return low == "true"
        return value
    if "integer" in kinds and isinstance(parsed, float) and parsed.is_integer():
        return int(parsed)
    return parsed


def _unframed(text: str) -> str:
    """a parameter's text without the one newline the template puts on each side; the rest is the value's"""
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    return text


def _arguments(args: Any) -> str:
    """arguments as the wire carries them: a JSON string (an object serialized, a string kept as it is)"""
    return args if isinstance(args, str) else json.dumps(args if args is not None else {})


class ToolFormat:
    """One family's convention. `kinds` names the engine families it serves; `openers` the text at which the
    streamed answer stops and buffers, since a call is not content."""

    kinds: tuple[str, ...] = ()
    openers: tuple[str, ...] = ()

    def prepare(self, messages: Sequence[Json], tools: Any) -> tuple[list[Json], Any]:
        """(messages, tools) as the family's chat template takes them; the default hands `tools` to the template"""
        return [dict(m) for m in messages], tools

    def calls(self, text: str, tools: Any = None) -> list[ToolCall]:
        return []

    def strip(self, text: str) -> str:
        """the answer with its call blocks removed, the prose around them untouched"""
        return text

    def split(self, text: str, tools: Any = None) -> tuple[str, list[ToolCall]]:
        """(prose, calls): the blocks parsed and struck from the prose, which is trimmed only when a call was found"""
        calls = self.calls(text, tools)
        return (self.strip(text).strip() if calls else text), calls

    def from_tokens(self, tok: Any, toks: Tokens, tools: Any = None) -> tuple[str, str, list[ToolCall]]:
        """(prose, reasoning, calls) of a generated token list"""
        content, thinking = answer(tok, toks)
        prose, calls = self.split(content, tools)
        return prose, thinking, calls

    def opener_at(self, text: str, start: int = 0) -> int | None:
        """the index of the first complete opener at or after `start`, or None"""
        hits = [p for p in (text.find(op, start) for op in self.openers) if p >= 0]
        return min(hits) if hits else None

    def holdback(self, text: str) -> int:
        """how many trailing characters of `text` could be an opener still forming, and are not emitted yet"""
        k = 0
        for op in self.openers:
            for j in range(1, min(len(op) - 1, len(text)) + 1):
                if text.endswith(op[:j]):
                    k = max(k, j)
        return k


class HermesJson(ToolFormat):
    """`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`: Qwen3, Qwen4 and the Hermes lineage. The body is
    the call object itself, nested arguments and all; a block that is not JSON, or names no tool, is left in the
    prose rather than guessed at."""

    kinds = (
        FamilyKind.QWEN3,
        FamilyKind.QWEN4,
    )
    openers = ("<tool_call>",)
    block = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def _call(self, body: str) -> ToolCall | None:
        try:
            obj = json.loads(body)
        except ValueError:
            return None
        if not isinstance(obj, dict) or not obj.get("name"):
            return None
        return {"name": str(obj["name"]), "arguments": _arguments(obj.get("arguments"))}

    def calls(self, text: str, tools: Any = None) -> list[ToolCall]:
        return [c for m in self.block.finditer(text) if (c := self._call(m.group(1))) is not None]

    def strip(self, text: str) -> str:
        return self.block.sub(lambda m: "" if self._call(m.group(1)) is not None else m.group(0), text)


class QwenXml(ToolFormat):
    """`<tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>`: Qwen3.5. Values are
    text; each is typed by the tool's schema (a string parameter is never re-read as a number), and an object or
    array parameter parses from the JSON the template told the model to write."""

    kinds = (FamilyKind.QWEN3_5,)
    openers = ("<tool_call>", "<function=")
    block = re.compile(
        r"(?:<tool_call>\s*)?<function\s*=\s*([^>\s]+)\s*>(.*?)</function>(?:\s*</tool_call>)?", re.DOTALL
    )
    _param = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

    def _call(self, m: re.Match[str], tools: Any) -> ToolCall | None:
        name = m.group(1).strip()
        if not name:
            return None
        schema = _schema_of(tools, name)
        args = {
            p.group(1).strip(): _typed(_unframed(p.group(2)), schema.get(p.group(1).strip()))
            for p in self._param.finditer(m.group(2))
        }
        return {"name": name, "arguments": json.dumps(args)}

    def calls(self, text: str, tools: Any = None) -> list[ToolCall]:
        return [c for m in self.block.finditer(text) if (c := self._call(m, tools)) is not None]

    def strip(self, text: str) -> str:
        return self.block.sub("", text)


class PhiJson(ToolFormat):
    """`<|tool_call|>[{"name": ..., "arguments": {...}}]<|/tool_call|>`: Phi-4-mini. Its template reads the schemas
    from the system message's `tools` field rather than the `tools` argument, and its markers are special tokens
    the decode drops, so a bare array of calls opening the answer is read as well."""

    kinds = (FamilyKind.PHI3,)
    openers = ("<|tool_call|>", '[{"name"')
    marked = re.compile(r"<\|tool_call\|>\s*(\[.*?\])\s*(?:<\|/tool_call\|>|$)", re.DOTALL)

    def prepare(self, messages: Sequence[Json], tools: Any) -> tuple[list[Json], Any]:
        """The template renders `<|role|>content<|end|>` for any role and ignores `tool_calls`, so the conversation
        is written in the model's own markers: the schemas in the system message's `tools` field, an assistant
        call as `<|tool_call|>[...]<|/tool_call|>` content, and a tool's result under the `tool_response` role."""
        out = []
        for m in messages:
            m = dict(m)
            tcs = m.pop("tool_calls", None)
            if m.get("role") == "assistant" and tcs:
                calls = [
                    {
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments", {}),
                    }
                    for tc in tcs
                ]
                m["content"] = (m.get("content") or "") + f"<|tool_call|>{json.dumps(calls)}<|/tool_call|>"
            elif m.get("role") == "tool":
                m = {"role": "tool_response", "content": str(m.get("content") or "")}
            out.append(m)
        if not tools:
            return out, None
        schemas = json.dumps([t.get("function", t) for t in tools if isinstance(t, dict)])
        for m in out:
            if m.get("role") == "system":
                m["tools"] = schemas
                break
        else:
            out.insert(0, {"role": "system", "content": "", "tools": schemas})
        return out, None

    def _calls_in(self, body: str) -> list[ToolCall]:
        try:
            arr = json.loads(body)
        except ValueError:
            return []
        if not isinstance(arr, list):
            return []
        return [
            {"name": str(o["name"]), "arguments": _arguments(o.get("arguments"))}
            for o in arr
            if isinstance(o, dict) and o.get("name")
        ]

    def _match(self, text: str) -> tuple[int, int, str] | None:
        """(start, end, the array's text) of the call array, marked or bare; a bare array's end is where the
        JSON decoder finds it (a pattern would stop at the first `]` inside an argument)"""
        m = self.marked.search(text)
        if m:
            return m.start(), m.end(), m.group(1)
        i = len(text) - len(text.lstrip())
        if not text.startswith("[", i):
            return None
        try:
            arr, end = json.JSONDecoder().raw_decode(text, i)
        except ValueError:
            return None
        if not (isinstance(arr, list) and arr and isinstance(arr[0], dict) and "name" in arr[0]):
            return None
        while end < len(text) and text[end].isspace():
            end += 1
        return i, end, text[i:end]

    def calls(self, text: str, tools: Any = None) -> list[ToolCall]:
        m = self._match(text)
        return self._calls_in(m[2]) if m else []

    def strip(self, text: str) -> str:
        m = self._match(text)
        return text[: m[0]] + text[m[1] :] if m else text


class Harmony(ToolFormat):
    """gpt-oss: a call is a commentary message addressed `to=functions.<name>` whose body is the arguments JSON,
    ended by `<|call|>`; it lives in the channel tokens, never in the content, so nothing is held back and nothing
    is parsed out of the prose."""

    kinds = (FamilyKind.GPT_OSS,)

    def from_tokens(self, tok: Any, toks: Tokens, tools: Any = None) -> tuple[str, str, list[ToolCall]]:
        content, thinking, calls = Channels(tok).split_calls(toks)
        dec = lambda x: str(tok.decode(x, skip_special_tokens=True)) if x else ""
        out: list[ToolCall] = []
        for name, body in calls:
            args = dec(body).strip()
            try:
                json.loads(args)
            except ValueError:
                args = json.dumps({"input": args})  # a body that is not JSON reaches the tool whole, not lost
            out.append({"name": name, "arguments": args})
        return dec(content).strip() if out else dec(content), dec(thinking), out


class AnyText(ToolFormat):
    """A family this module does not know: every text form tried in turn, the first that finds a call wins."""

    forms: tuple[ToolFormat, ...] = (HermesJson(), QwenXml(), PhiJson())
    openers = tuple(dict.fromkeys(op for f in forms for op in f.openers))

    def calls(self, text: str, tools: Any = None) -> list[ToolCall]:
        for f in self.forms:
            found = f.calls(text, tools)
            if found:
                return found
        return []

    def strip(self, text: str) -> str:
        for f in self.forms:
            if f.calls(text):
                return f.strip(text)
        return text


FORMATS: tuple[ToolFormat, ...] = (HermesJson(), QwenXml(), PhiJson(), Harmony())


def tool_format(kind: FamilyKind | str | None) -> ToolFormat:
    """the family's format, or every text form for a family none names"""
    for f in FORMATS:
        if kind in f.kinds:
            return f
    return AnyText()


def tool_calls_of(text: str, tools: Any = None) -> tuple[str, list[ToolCall]]:
    """(prose, calls) of an answer whose family is not known: every text form tried"""
    return AnyText().split(text, tools)
