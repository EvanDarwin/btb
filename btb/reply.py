# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""One request's answers as they decode (docs/streaming.md): tokens in, a row each, and out the events a streamed
route writes - a piece of content or of reasoning, a row's tool calls, a row's finish - and, once the decode is
over, each row whole, which a route answering at once reads. The one place a stop string cuts an answer, a tool call
is read out of one, and the tokens and logprobs an answer counts are counted: every route, streamed or not, one row
or `n`, answers through it alike. A row knows when it wants no more tokens (its stop token, a stop string, the
client gone), and the decode asks it."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .text import Channels, TextStream

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from .engine.hooks import TokenLogprob
    from .tools import ToolCall, ToolFormat


def _cut(text: str, stops: Sequence[str]) -> tuple[str, bool]:
    """the text up to the first stop string in it, and whether there was one"""
    at = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
    return (text[:at], True) if at >= 0 else (text, False)


class _Stops:
    """Stop strings over text arriving in pieces: `push(piece)` releases what is safe (a tail that could open a
    stop string is held back) and sets `hit` once one appears, cutting there; `flush()` the held tail at the end"""

    def __init__(self, stops: Sequence[str]) -> None:
        self.stops = list(stops)
        self.buf, self.hit = "", False

    def push(self, piece: str) -> str:
        if self.hit or not self.stops:
            return "" if self.hit else piece
        self.buf += piece
        text, self.hit = _cut(self.buf, self.stops)
        if self.hit:
            self.buf = ""
            return text
        hold = next(
            (
                k
                for k in range(min(len(self.buf), max(map(len, self.stops)) - 1), 0, -1)
                if any(s.startswith(self.buf[-k:]) for s in self.stops)
            ),
            0,
        )
        out, self.buf = self.buf[: len(self.buf) - hold], self.buf[len(self.buf) - hold :]
        return out

    def flush(self) -> str:
        out, self.buf = ("" if self.hit else self.buf), ""
        return out


class _Pieces:
    """One answer's tokens as text pieces as they decode: gpt-oss's channels split (`push(t)` returns
    [(kind, delta)], kind "content" or "thinking"), a stop token and a call's body never text"""

    def __init__(self, tok: PreTrainedTokenizerBase, eos: Sequence[int]) -> None:
        self.ch = Channels(tok)
        self.streams = {"content": TextStream(tok), "thinking": TextStream(tok)}
        self.eos = set(eos)

    def push(self, t: int) -> list[tuple[str, str]]:
        if t in self.eos:
            return []
        kind = self.ch.push(t)
        if kind is None or kind == "call":  # a harmony call's body is never content
            return []
        delta = self.streams[kind].push(t)
        return [(kind, delta)] if delta else []

    def flush(self) -> list[tuple[str, str]]:
        return [(kind, tail) for kind, st in self.streams.items() if (tail := st.flush())]


class _ToolGate:
    """The prose of a tool-calling turn: pieces go out as they decode until a call's opener, from which the text is
    buffered (a partial call never leaks as content); `finish` sends the prose that followed the call once the
    blocks are known and struck"""

    def __init__(self, fmt: ToolFormat, send: Callable[[str], None], tools: Any = None) -> None:
        self.fmt, self.send, self.tools = fmt, send, tools
        self.buf, self.sent, self.in_call = "", 0, False

    def push(self, delta: str) -> None:
        self.buf += delta
        if self.in_call:
            return
        p = self.fmt.opener_at(self.buf, self.sent)
        if p is not None:
            self.in_call = True
            out, self.sent = self.buf[self.sent : p], p
            if out:
                self.send(out)
            return
        upto = len(self.buf) - self.fmt.holdback(self.buf)
        if upto > self.sent:
            out, self.sent = self.buf[self.sent : upto], upto
            self.send(out)

    def finish(self) -> None:
        """the prose after the last call, and whatever a holdback kept, with the call blocks struck"""
        prose = self.fmt.strip(self.buf) if self.fmt.calls(self.buf, self.tools) else self.buf
        sent = self.buf[: self.sent]
        tail = prose[len(sent) :] if prose.startswith(sent) else ""
        if self.in_call:
            tail = tail.lstrip()
        if tail.strip():
            self.send(tail)


@dataclass
class Row:
    """one answer: its tokens as decoded (stop tokens out), how many of them it counts (up to its stop string), its
    content and reasoning as sent, its tool calls, and how it ended - `finish` None while it decodes, else "stop"
    (its stop token or a stop string), "length" (the cap) or "tool_calls"""

    index: int
    tokens: list[int] = field(default_factory=list)
    spent: int = 0
    content: str = ""
    thinking: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    logprobs: list[TokenLogprob] | None = None
    ended: bool = False  # at its stop token
    cut: bool = False  # at a stop string
    finish: str | None = None


# an event a streamed route writes, in the order the rows make them: ("text", row, kind, piece) - kind "content" or
# "thinking"; ("calls", row, [ToolCall]); ("done", row, finish reason)
Event = tuple[Any, ...]


class Reply:
    """`n` answers to one request as they decode. `token(r, t)` on the decode's thread with each token drawn;
    `done(r)` whether row r wants no more (the decode stops it, or it leaves the batch); `finish(...)` once the
    decode is over. `emit(event)` receives the events as they come (a streamed route's writer); the rows are read
    whole after `finish` (a route answering at once). `leave()` - the client gone - makes every row done. `cancel`
    is set once every row is done: one answer's decode stops at its next step on it."""

    def __init__(
        self,
        tok: PreTrainedTokenizerBase,
        eos: Sequence[int],
        n: int,
        stops: Sequence[str] = (),
        tools: Any = None,
        fmt: ToolFormat | None = None,
        emit: Callable[[Event], object] | None = None,
    ) -> None:
        self.tok, self.eos, self.stops, self.tools, self.fmt = tok, set(eos), list(stops), tools, fmt
        self.emit = emit
        self.rows = [Row(i) for i in range(int(n))]
        self.gone = False
        self.cancel = threading.Event()
        self._pieces = [_Pieces(tok, eos) for _ in range(n)]
        self._stops = [_Stops(stops) for _ in range(n)]
        # the prose streams as it decodes; with tools, from a call's opener the text is buffered and read whole
        self._gates = [
            _ToolGate(fmt, functools.partial(self._send_content, r), tools) if tools and fmt else None for r in range(n)
        ]

    def _event(self, *ev: Any) -> None:
        if self.emit is not None:
            self.emit(ev)

    def _send_content(self, r: int, piece: str) -> None:
        self.rows[r].content += piece
        self._event("text", r, "content", piece)

    def _content(self, r: int, piece: str) -> None:
        """a piece of row r's content, through its stop strings and its tool gate"""
        st, g = self._stops[r], self._gates[r]
        piece = st.push(piece)
        if st.hit:
            self.rows[r].cut = True
        if piece:
            if g is not None:
                g.push(piece)
            else:
                self._send_content(r, piece)

    def leave(self) -> None:
        """the client is gone: no row wants another token"""
        self.gone = True
        self.cancel.set()

    def token(self, r: int, t: int) -> None:
        """a token drawn for row r"""
        self._token(r, t)
        if self.done(r) and self.all_done:
            self.cancel.set()

    def _token(self, r: int, t: int) -> None:
        row = self.rows[r]
        if row.ended:
            return
        if t in self.eos:
            row.ended = True
            return
        row.tokens.append(int(t))
        if not row.cut:
            row.spent += 1
        for kind, piece in self._pieces[r].push(int(t)):
            if kind == "content":
                if not row.cut:
                    self._content(r, piece)
            elif not row.cut:
                row.thinking += piece
                self._event("text", r, "thinking", piece)

    def done(self, r: int) -> bool:
        """whether row r wants no more tokens: its stop token, a stop string, or the client gone"""
        row = self.rows[r]
        return self.gone or row.ended or row.cut

    @property
    def all_done(self) -> bool:
        return all(self.done(r) for r in range(len(self.rows)))

    def finish(self, logprobs: Sequence[Sequence[TokenLogprob] | None] | None = None) -> None:
        """the decode is over: each row's held text sent, its calls read (from the text its stop string left; a
        channel format's from its tokens), its logprobs counted up to its stop, its finish decided"""
        fmt, tools = self.fmt, self.tools
        for r, row in enumerate(self.rows):
            for kind, tail in self._pieces[r].flush():
                if row.cut:
                    continue
                if kind == "content":
                    self._content(r, tail)
                else:
                    row.thinking += tail
                    self._event("text", r, "thinking", tail)
            st, g = self._stops[r], self._gates[r]
            if not st.hit and (tail := st.flush()):
                if g is not None:
                    g.push(tail)
                else:
                    self._send_content(r, tail)
            if g is not None and fmt is not None:
                row.calls = list(
                    fmt.calls(g.buf, tools) if fmt.in_text else fmt.from_tokens(self.tok, row.tokens, tools)[2]
                )
                if row.calls and fmt.in_text:
                    # the message's content is the prose around the calls, trimmed as a whole answer is
                    prose, _ = fmt.split(g.buf, tools)
                    g.finish()
                    row.content = prose
                else:
                    g.finish()
                    if row.calls:
                        row.content = row.content.strip()  # a channel format's prose beside its calls, trimmed
                if row.calls:
                    self._event("calls", r, row.calls)
            if logprobs is not None and r < len(logprobs) and logprobs[r] is not None:
                row.logprobs = list(logprobs[r] or ())[: row.spent]
            row.finish = "tool_calls" if row.calls else ("stop" if row.cut or row.ended else "length")
            self._event("done", r, row.finish)
