# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Logits processors for what a request may ask of an answer: a bias on chosen tokens, OpenAI's presence and
frequency penalties, and the answer's text held to a language (`PrefixConstraint` over a `TextPrefix`; JSON mode
is `JsonObjectPrefix`). Each reads the answer as the ids after `start`, the prompt's length."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol

import torch

NEG = float("-inf")


class LogitBias:
    """a fixed amount added to chosen tokens' logits (OpenAI's `logit_bias`: -100 bans a token, 100 all but forces
    it)"""

    def __init__(self, bias: Mapping[int, float]) -> None:
        self.idx = torch.tensor([int(t) for t in bias], dtype=torch.long)
        self.val = torch.tensor([float(v) for v in bias.values()], dtype=torch.float32)

    def __call__(self, ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        out = logits.clone()
        out[self.idx] += self.val.to(out.dtype)
        return out


class Penalties:
    """OpenAI's penalties over the answer so far: `presence` off every token it holds, `frequency` times the times
    it holds it"""

    def __init__(self, start: int, presence: float = 0.0, frequency: float = 0.0) -> None:
        self.start, self.presence, self.frequency = int(start), float(presence), float(frequency)

    def __call__(self, ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        ans = ids[self.start :]
        if not ans:
            return logits
        V = int(logits.shape[-1])
        n = torch.bincount(torch.tensor([t for t in ans if 0 <= t < V], dtype=torch.long), minlength=V).to(logits.dtype)
        return logits - self.frequency * n - self.presence * (n > 0).to(logits.dtype)


class TextTokenizer(Protocol):
    """what a constraint reads of a tokenizer: a text's ids, ids' text, and the special ids (no text of theirs)"""

    @property
    def all_special_ids(self) -> list[int]: ...

    def __call__(self, text: str, /, *, add_special_tokens: bool) -> Mapping[str, Sequence[int]]: ...

    def decode(self, ids: list[int], /) -> str | list[str]: ...


class TextPrefix(Protocol):
    """A recognizer's state over the text read so far: `feed(text)` is the state after `text`, None where no text
    of the language starts that way; `complete` when the text read is a whole one, which a stop token may end"""

    @property
    def complete(self) -> bool: ...

    def feed(self, text: str) -> TextPrefix | None: ...


class PrefixConstraint:
    """The answer's text held to a language: `initial` is its recognizer before any text, a token is allowed while
    its text keeps the answer a prefix the recognizer accepts, and a stop token once the answer is complete (or no
    token can go on). The pick is made among the `width` likeliest allowed tokens (greedy is exact; a draw loses
    the mass past them), scanning at most `scan` in logit order."""

    def __init__(
        self,
        tok: TextTokenizer,
        start: int,
        stop_ids: Sequence[int],
        initial: TextPrefix,
        width: int = 64,
        scan: int = 4096,
    ) -> None:
        self.tok, self.start = tok, int(start)
        self.stop = [int(t) for t in stop_ids]
        self.width, self.scan = int(width), int(scan)
        self._piece: dict[int, str] = {}
        self._state: dict[tuple[int, ...], TextPrefix | None] = {(): initial}
        anchor = tok("a", add_special_tokens=False)["input_ids"]
        self._anchor = [int(anchor[0])] if anchor else []
        self._anchor_text = str(tok.decode(self._anchor)) if anchor else ""
        self._special = {int(t) for t in tok.all_special_ids}

    def piece(self, t: int) -> str:
        """token t's text as it reads inside a longer text (a leading space kept)"""
        p = self._piece.get(t)
        if p is None:
            p = "" if t in self._special else str(self.tok.decode([*self._anchor, t]))[len(self._anchor_text) :]
            self._piece[t] = p
        return p

    def state(self, ans: tuple[int, ...]) -> TextPrefix | None:
        """the recognizer after the answer `ans` (ids), None once it strayed"""
        if ans in self._state:
            return self._state[ans]
        prev = self.state(ans[:-1])
        st = None if prev is None else prev.feed(self.piece(ans[-1]))
        self._state[ans] = st
        return st

    def __call__(self, ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        st = self.state(tuple(int(t) for t in ids[self.start :]))
        out = torch.full_like(logits, NEG)
        if st is None or st.complete:
            out[self.stop] = logits[self.stop]
        if st is None:
            return out
        kept = 0
        for t in torch.argsort(logits, descending=True)[: self.scan].tolist():
            p = self.piece(t)
            if p and st.feed(p) is not None:
                out[t] = logits[t]
                kept += 1
                if kept >= self.width:
                    break
        if not kept:
            out[self.stop] = logits[self.stop]
        return out


_WS = " \t\n\r"


class _Mode(StrEnum):
    """what a JSON object's text may go on with"""

    START = "start"  # before the object: whitespace, then its '{'
    KEY_OR_END = "key_or_end"  # an object's first key, or its '}'
    KEY_EXPECTED = "key_expected"  # after an object's ',': a key
    KEY = "key"  # inside a key
    COLON = "colon"
    VALUE = "value"
    VALUE_OR_END = "value_or_end"  # an array's first value, or its ']'
    STR = "str"  # inside a string value
    NUM = "num"
    LITERAL = "literal"  # inside true, false or null
    AFTER = "after"  # after a value: ',' or its container's close
    DONE = "done"  # the object closed


class _Escape(StrEnum):
    """a string's state after a backslash (a `\\u` escape's is the count of hex digits left)"""

    PENDING = "pending"


class _Num(StrEnum):
    """a number's states: -?(0|[1-9]\\d*)(\\.\\d+)?([eE][+-]?\\d+)?"""

    MINUS = "minus"
    ZERO = "zero"
    INT = "int"
    DOT = "dot"
    FRAC = "frac"
    E = "e"
    E_SIGN = "e_sign"
    EXP = "exp"


class _Char(StrEnum):
    """the classes of character that move a number"""

    ZERO = "0"
    DIGIT = "d"  # 1-9
    MINUS = "-"
    PLUS = "+"
    DOT = "."
    E = "e"  # e or E
    OTHER = ""


# where a number goes on from each state with each class of character
_NUM: dict[tuple[_Num, _Char], _Num] = {
    (_Num.MINUS, _Char.ZERO): _Num.ZERO,
    (_Num.MINUS, _Char.DIGIT): _Num.INT,
    (_Num.ZERO, _Char.DOT): _Num.DOT,
    (_Num.ZERO, _Char.E): _Num.E,
    (_Num.INT, _Char.ZERO): _Num.INT,
    (_Num.INT, _Char.DIGIT): _Num.INT,
    (_Num.INT, _Char.DOT): _Num.DOT,
    (_Num.INT, _Char.E): _Num.E,
    (_Num.DOT, _Char.ZERO): _Num.FRAC,
    (_Num.DOT, _Char.DIGIT): _Num.FRAC,
    (_Num.FRAC, _Char.ZERO): _Num.FRAC,
    (_Num.FRAC, _Char.DIGIT): _Num.FRAC,
    (_Num.FRAC, _Char.E): _Num.E,
    (_Num.E, _Char.PLUS): _Num.E_SIGN,
    (_Num.E, _Char.MINUS): _Num.E_SIGN,
    (_Num.E, _Char.ZERO): _Num.EXP,
    (_Num.E, _Char.DIGIT): _Num.EXP,
    (_Num.E_SIGN, _Char.ZERO): _Num.EXP,
    (_Num.E_SIGN, _Char.DIGIT): _Num.EXP,
    (_Num.EXP, _Char.ZERO): _Num.EXP,
    (_Num.EXP, _Char.DIGIT): _Num.EXP,
}
# the states a number may end in
_NUM_ENDS = frozenset({_Num.ZERO, _Num.INT, _Num.FRAC, _Num.EXP})
_LITERALS = ("true", "false", "null")

# what a mode carries beside it: a string's escape (or hex digits left), a number's state, a literal and how far
# into it the text is
_Aux = _Escape | int | _Num | tuple[str, int] | None


def _char_class(ch: str) -> _Char:
    if ch == "0":
        return _Char.ZERO
    if ch in "123456789":
        return _Char.DIGIT
    if ch in "eE":
        return _Char.E
    return _Char(ch) if ch in "-+." else _Char.OTHER


class JsonObjectPrefix:
    """A prefix of one JSON object, read a character at a time (leading whitespace allowed, nothing after the
    object closes). `stack` holds the open containers ('{' or '['); `mode` what may come next."""

    __slots__ = ("aux", "mode", "stack")

    def __init__(self, stack: tuple[str, ...] = (), mode: _Mode = _Mode.START, aux: _Aux = None) -> None:
        self.stack, self.mode, self.aux = stack, mode, aux

    @property
    def complete(self) -> bool:
        return self.mode is _Mode.DONE

    def feed(self, text: str) -> JsonObjectPrefix | None:
        st: JsonObjectPrefix | None = JsonObjectPrefix(self.stack, self.mode, self.aux)
        for ch in text:
            assert st is not None
            st = st._step(ch)
            if st is None:
                return None
        return st

    def _step(self, ch: str) -> JsonObjectPrefix | None:
        m, stack, aux = self.mode, self.stack, self.aux
        if m is _Mode.START:
            if ch in _WS:
                return self
            return JsonObjectPrefix(("{",), _Mode.KEY_OR_END) if ch == "{" else None
        if m is _Mode.DONE:
            return None
        if m in (_Mode.STR, _Mode.KEY):
            if aux is _Escape.PENDING:
                if ch == "u":
                    return JsonObjectPrefix(stack, m, 4)
                return JsonObjectPrefix(stack, m) if ch in '"\\/bfnrt' else None
            if isinstance(aux, int):
                if ch not in "0123456789abcdefABCDEF":
                    return None
                return JsonObjectPrefix(stack, m, aux - 1 if aux > 1 else None)
            if ch == "\\":
                return JsonObjectPrefix(stack, m, _Escape.PENDING)
            if ch == '"':
                return JsonObjectPrefix(stack, _Mode.COLON if m is _Mode.KEY else _Mode.AFTER)
            return None if ord(ch) < 0x20 else self
        if m is _Mode.NUM:
            assert isinstance(aux, _Num)
            nxt = _NUM.get((aux, _char_class(ch)))
            if nxt is not None:
                return JsonObjectPrefix(stack, _Mode.NUM, nxt)
            if aux not in _NUM_ENDS:
                return None
            return JsonObjectPrefix(stack, _Mode.AFTER)._step(ch)
        if m is _Mode.LITERAL:
            assert isinstance(aux, tuple)
            word, i = aux
            if ch != word[i]:
                return None
            if i + 1 == len(word):
                return JsonObjectPrefix(stack, _Mode.AFTER)
            return JsonObjectPrefix(stack, _Mode.LITERAL, (word, i + 1))
        if ch in _WS:
            return self
        if m is _Mode.KEY_OR_END:
            if ch == '"':
                return JsonObjectPrefix(stack, _Mode.KEY)
            return self._close(ch)
        if m is _Mode.COLON:
            return JsonObjectPrefix(stack, _Mode.VALUE) if ch == ":" else None
        if m in (_Mode.VALUE, _Mode.VALUE_OR_END):
            if m is _Mode.VALUE_OR_END and ch == "]":
                return self._close(ch)
            if ch == '"':
                return JsonObjectPrefix(stack, _Mode.STR)
            if ch in "{[":
                return JsonObjectPrefix((*stack, ch), _Mode.KEY_OR_END if ch == "{" else _Mode.VALUE_OR_END)
            if ch == "-":
                return JsonObjectPrefix(stack, _Mode.NUM, _Num.MINUS)
            if ch == "0":
                return JsonObjectPrefix(stack, _Mode.NUM, _Num.ZERO)
            if ch in "123456789":
                return JsonObjectPrefix(stack, _Mode.NUM, _Num.INT)
            for word in _LITERALS:
                if ch == word[0]:
                    return JsonObjectPrefix(stack, _Mode.LITERAL, (word, 1))
            return None
        if m is _Mode.AFTER:
            if ch == ",":
                return JsonObjectPrefix(stack, _Mode.KEY_EXPECTED if stack[-1] == "{" else _Mode.VALUE)
            return self._close(ch)
        if m is _Mode.KEY_EXPECTED:
            return JsonObjectPrefix(stack, _Mode.KEY) if ch == '"' else None
        return None

    def _close(self, ch: str) -> JsonObjectPrefix | None:
        if not self.stack or ch != ("}" if self.stack[-1] == "{" else "]"):
            return None
        rest = self.stack[:-1]
        return JsonObjectPrefix(rest, _Mode.AFTER) if rest else JsonObjectPrefix((), _Mode.DONE)
