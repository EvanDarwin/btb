# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Text on either side of the engine, with any tokenizer that speaks the transformers interface: a prompt
through the chat template, an answer out of a token list (gpt-oss's channels split), and a stream that
releases text only once its bytes are whole. Nothing here imports torch.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .kinds import Json, Tokens

Messages = Sequence[Json]


def messages_of(prompt: str | Messages) -> list[dict[str, str]]:
    """
    A conversation: the messages as given, or one user turn from a string
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in prompt]


def template(
    tok: Any, history: Messages, thinking: bool = False, tools: Any = None, continue_final: bool = False
) -> str:
    """
    The conversation rendered through the tokenizer's chat template with the generation prompt appended;
    `thinking` reaches the templates that have the switch (Qwen3's) and `tools` the ones that render function
    schemas, both dropped for a template whose signature lacks them. `continue_final` leaves the last message
    (the assistant's, begun) open for the model to go on from, where a new turn would start otherwise.
    """
    if continue_final and history and history[-1].get("role") == "assistant":
        kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False, "continue_final_message": True}
    else:
        kw = {"tokenize": False, "add_generation_prompt": True}
    kw["enable_thinking"] = thinking
    if tools:
        kw["tools"] = tools
    while True:
        try:
            return str(tok.apply_chat_template(history, **kw))
        except TypeError:
            if "enable_thinking" in kw:
                del kw["enable_thinking"]
            elif "tools" in kw:
                del kw["tools"]
            elif kw.get("continue_final_message"):
                # a tokenizer without the switch: the turn opened as a new one, the begun text after it
                head = template(tok, history[:-1], thinking, tools)
                return head + str(history[-1].get("content") or "")
            else:
                raise


def prompt_ids(
    tok: Any, prompt: str | Messages, thinking: bool = False, tools: Any = None, continue_final: bool = False
) -> list[int]:
    """
    The prompt as the model takes it: the template rendered and tokenized, no special tokens added on top
    """
    text = template(tok, messages_of(prompt), thinking, tools, continue_final)
    return [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]


def probe_tail(tok: Any) -> int:
    """
    How many tokens of a chat template's generation prompt the next turn renders differently (Qwen3's empty
    think block, kept on the final turn alone): a session that knows it resumes the first follow-up turn from
    its snapshot instead of learning the tail from that turn's mismatch and prefilling the whole context again
    """
    try:
        ids = prompt_ids(tok, [{"role": "user", "content": "y"}])
        probe = [
            {"role": "user", "content": "y"},
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "y"},
        ]
        kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
        try:
            text = tok.apply_chat_template(probe, **kw)
        except TypeError:
            del kw["enable_thinking"]
            text = tok.apply_chat_template(probe, **kw)
        other = [int(i) for i in tok(str(text), add_special_tokens=False)["input_ids"]]
    except Exception:
        return 0
    m, lim = 0, min(len(ids), len(other))
    while m < lim and ids[m] == other[m]:
        m += 1
    return max(0, len(ids) - m)


class Channels:
    """
    gpt-oss's harmony channels, a token at a time: `push(t)` names the token's part ("content", "thinking",
    "call" for the body of a tool call, None for a marker), `split(ids)` sorts an output into (content ids,
    thinking ids), `split_calls(ids)` adds the calls. No markers: all content
    """

    header: Any
    recipient: Any
    kind: Any

    _TO = re.compile(r"to=functions\.([\w.-]+)")

    def __init__(self, tok: Any) -> None:
        ids = {}
        unk = getattr(tok, "unk_token_id", None)
        for n in ("channel", "message", "end", "start", "return", "call"):
            i = tok.convert_tokens_to_ids(f"<|{n}|>")
            ids[n] = int(i) if isinstance(i, int) and i >= 0 and i != unk else None
        self.on = all(ids[n] is not None for n in ("channel", "message", "end", "start"))
        self.ids = ids
        self.tok = tok
        self.stops = {i for i in (ids["end"], ids["return"], ids["call"]) if i is not None}
        self.header = None
        # the tokens between <|start|> and <|channel|>: the message's recipient, `assistant to=functions.x` for a
        # call written the template's way (the model may put the recipient in the header instead; both are read)
        self.recipient = None
        # what a body token is until a header says otherwise: an output that opens without one is the answer
        self.kind = "content"
        self.call_name: str | None = None
        self.fresh = False  # a header just resolved: the next body token opens a new message

    def push(self, t: int) -> Any:
        if not self.on:
            return "content"
        t = int(t)
        if self.header is not None:
            if t == self.ids["message"]:
                name = self.tok.decode(self.header, skip_special_tokens=True).strip()
                lead = self.tok.decode(self.recipient or [], skip_special_tokens=True)
                m = self._TO.search(lead + " " + name)
                if m is not None:
                    self.kind, self.call_name = "call", m.group(1)
                else:
                    self.kind, self.call_name = ("content" if name.startswith("final") else "thinking"), None
                self.fresh = True
                self.header = None
                self.recipient = None
            else:
                self.header.append(t)
            return None
        if t == self.ids["channel"]:
            self.header = []
            return None
        if t == self.ids["start"]:
            self.recipient = []
            self.kind = None
            return None
        if self.recipient is not None:
            self.recipient.append(t)
            return None
        if t in self.stops:
            # between messages nothing is text
            self.kind = None
            return None
        return self.kind

    def split(self, ids: Tokens) -> tuple[list[int], list[int]]:
        content, thinking, _calls = self.split_calls(ids)
        return content, thinking

    def split_calls(self, ids: Tokens) -> tuple[list[int], list[int], list[tuple[str, list[int]]]]:
        """(content ids, thinking ids, calls as (tool name, body ids))"""
        content: list[int] = []
        thinking: list[int] = []
        calls: list[tuple[str, list[int]]] = []
        for t in ids:
            k = self.push(t)
            if k == "content":
                content.append(int(t))
            elif k == "thinking":
                thinking.append(int(t))
            elif k == "call":
                if self.fresh or not calls:
                    calls.append((str(self.call_name), []))
                calls[-1][1].append(int(t))
            if k is not None:
                self.fresh = False
        return content, thinking, calls


def answer(tok: Any, toks: Tokens) -> tuple[str, str]:
    """
    (answer, reasoning) as text for a generated token list: the final channel and the rest for gpt-oss,
    everything and "" for every other family
    """
    c, th = Channels(tok).split(toks)
    dec = lambda x: str(tok.decode(x, skip_special_tokens=True)) if x else ""
    return dec(c), dec(th)


class TextStream:
    """
    Tokens in, whole text out. A token can end mid-character (a byte of a multi-byte sequence), so the text
    is decoded over the tokens since the last line and a piece is released only when it decodes clean; the
    window rolls at a newline, where a tokenizer's leading-space rules cannot change what came before
    """

    def __init__(self, tok: Any, window: int = 2048) -> None:
        self.tok = tok
        self.window = int(window)
        self.buf: list[int] = []
        self.sent = ""

    def push(self, t: int) -> str:
        """
        The text this token released, "" when it is still incomplete
        """
        self.buf.append(int(t))
        text = str(self.tok.decode(self.buf, skip_special_tokens=True))
        if text.endswith("�"):
            return ""
        piece = text[len(self.sent) :]
        self.sent = text
        if text.endswith("\n") or len(self.buf) >= self.window:
            self.buf.clear()
            self.sent = ""
        return piece

    def flush(self) -> str:
        """
        Whatever is still held at the end, replacement characters included
        """
        if not self.buf:
            return ""
        text = str(self.tok.decode(self.buf, skip_special_tokens=True))
        piece = text[len(self.sent) :]
        self.buf.clear()
        self.sent = ""
        return piece
