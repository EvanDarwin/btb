# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Logits processors for what a request may ask of an answer: a bias on chosen tokens, OpenAI's presence and
frequency penalties, and the answer's text held to a language (`PrefixConstraint` over a `TextPrefix`; JSON mode
is `JsonObjectPrefix`). Each reads the answer as the ids after `start`, the prompt's length."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

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
        tok: Any,
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
        self._special = {int(t) for t in getattr(tok, "all_special_ids", []) or []}

    def piece(self, t: int) -> str:
        """token t's text as it reads inside a longer text (a leading space kept)"""
        p = self._piece.get(t)
        if p is None:
            p = "" if t in self._special else str(self.tok.decode([*self._anchor, t]))[len(self._anchor_text) :]
            self._piece[t] = p
        return p

    def state(self, ans: tuple[int, ...]) -> TextPrefix | None:
        """the recognizer after the answer `ans` (ids), None once it strayed"""
        st = self._state.get(ans, _MISSING)
        if st is _MISSING:
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


_MISSING: Any = object()
_WS = " \t\n\r"


class JsonObjectPrefix:
    """A prefix of one JSON object, read a character at a time (leading whitespace allowed, nothing after the
    object closes). `stack` holds the open containers ('{' or '['); `mode` what may come next."""

    __slots__ = ("aux", "mode", "stack")

    def __init__(self, stack: tuple[str, ...] = (), mode: str = "start", aux: Any = None) -> None:
        self.stack, self.mode, self.aux = stack, mode, aux

    @property
    def complete(self) -> bool:
        return self.mode == "done"

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
        if m == "start":
            if ch in _WS:
                return self
            return JsonObjectPrefix(("{",), "key_or_end") if ch == "{" else None
        if m == "done":
            return None
        if m in ("str", "key"):
            if aux == "esc":
                if ch == "u":
                    return JsonObjectPrefix(stack, m, 4)
                return JsonObjectPrefix(stack, m) if ch in '"\\/bfnrt' else None
            if isinstance(aux, int):
                if ch not in "0123456789abcdefABCDEF":
                    return None
                return JsonObjectPrefix(stack, m, aux - 1 if aux > 1 else None)
            if ch == "\\":
                return JsonObjectPrefix(stack, m, "esc")
            if ch == '"':
                return JsonObjectPrefix(stack, "colon" if m == "key" else "after")
            return None if ord(ch) < 0x20 else self
        if m == "num":
            nxt = _NUM.get((aux, _num_class(ch)))
            if nxt is not None:
                return JsonObjectPrefix(stack, "num", nxt)
            if aux not in ("zero", "int", "frac", "exp"):
                return None
            return JsonObjectPrefix(stack, "after")._step(ch)
        if m == "lit":
            word, i = aux
            if ch != word[i]:
                return None
            return (
                JsonObjectPrefix(stack, "after")
                if i + 1 == len(word)
                else JsonObjectPrefix(stack, "lit", (word, i + 1))
            )
        if ch in _WS:
            return self
        if m == "key_or_end":
            if ch == '"':
                return JsonObjectPrefix(stack, "key")
            return self._close(ch)
        if m == "colon":
            return JsonObjectPrefix(stack, "value") if ch == ":" else None
        if m in ("value", "value_or_end"):
            if m == "value_or_end" and ch == "]":
                return self._close(ch)
            if ch == '"':
                return JsonObjectPrefix(stack, "str")
            if ch in "{[":
                return JsonObjectPrefix((*stack, ch), "key_or_end" if ch == "{" else "value_or_end")
            if ch == "-":
                return JsonObjectPrefix(stack, "num", "minus")
            if ch == "0":
                return JsonObjectPrefix(stack, "num", "zero")
            if ch in "123456789":
                return JsonObjectPrefix(stack, "num", "int")
            for word in ("true", "false", "null"):
                if ch == word[0]:
                    return JsonObjectPrefix(stack, "lit", (word, 1))
            return None
        if m == "after":
            if ch == ",":
                return JsonObjectPrefix(stack, "key_expected" if stack[-1] == "{" else "value")
            return self._close(ch)
        if m == "key_expected":
            return JsonObjectPrefix(stack, "key") if ch == '"' else None
        return None

    def _close(self, ch: str) -> JsonObjectPrefix | None:
        if not self.stack or ch != ("}" if self.stack[-1] == "{" else "]"):
            return None
        rest = self.stack[:-1]
        return JsonObjectPrefix(rest, "after") if rest else JsonObjectPrefix((), "done")


def _num_class(ch: str) -> str:
    if ch == "0":
        return "0"
    if ch in "123456789":
        return "d"
    return {"-": "-", "+": "+", ".": ".", "e": "e", "E": "e"}.get(ch, "")


# a number's states and the character classes that move them: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
_NUM: dict[tuple[str, str], str] = {
    ("minus", "0"): "zero",
    ("minus", "d"): "int",
    ("zero", "."): "dot",
    ("zero", "e"): "e",
    ("int", "0"): "int",
    ("int", "d"): "int",
    ("int", "."): "dot",
    ("int", "e"): "e",
    ("dot", "0"): "frac",
    ("dot", "d"): "frac",
    ("frac", "0"): "frac",
    ("frac", "d"): "frac",
    ("frac", "e"): "e",
    ("e", "+"): "esign",
    ("e", "-"): "esign",
    ("e", "0"): "exp",
    ("e", "d"): "exp",
    ("esign", "0"): "exp",
    ("esign", "d"): "exp",
    ("exp", "0"): "exp",
    ("exp", "d"): "exp",
}
