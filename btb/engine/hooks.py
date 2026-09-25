# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a caller hooks into a decode: logits processors ahead of every pick, the drawn tokens' log-probabilities,
hidden states from chosen layers, and a callback per pass. Every loop applies them at its one pick, so a hooked
decode is the same with speculation or without."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

import torch


class LogitsProcessor(Protocol):
    """`(ids, logits) -> logits`: the ids the row holds so far (prompt and answer) and the next token's logits
    [V]. Called on every candidate row, a speculative pass's drafts included, so it must be a function of `ids`:
    a grammar keeps its own memo by prefix."""

    def __call__(self, ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class TokenLogprob:
    """a drawn token's log-probability under the distribution it was picked from (the processed logits, before
    the temperature), and the `top` most likely tokens there with theirs"""

    token: int
    logprob: float
    top: tuple[tuple[int, float], ...] = ()


class PassStats(TypedDict):
    """one pass of a decode (the prefill, which picks the first token, is pass 0): its drafted and accepted
    tokens, the tokens it committed, and its wall time"""

    index: int
    drafted: int
    accepted: int
    tokens: int
    seconds: float


@dataclass
class Hooks:
    """one call's hooks and, per row, what they collected"""

    processors: tuple[LogitsProcessor, ...] = ()
    logprobs: int | None = None  # None: none; 0: the drawn token's alone; k: and the k most likely
    taps: tuple[int, ...] = ()
    on_pass: Callable[[PassStats], Any] | None = None
    lp: list[list[TokenLogprob]] = field(default_factory=list)
    hidden: list[dict[int, list[torch.Tensor]]] = field(default_factory=list)

    @property
    def needs_logits(self) -> bool:
        """the pick must run here over logits: the fused paths that pick in their graph stand aside"""
        return bool(self.processors) or self.logprobs is not None

    @property
    def active(self) -> bool:
        """the logits or the layers are wanted in hand: the fused paths stand aside"""
        return self.needs_logits or bool(self.taps)

    @property
    def any(self) -> bool:
        """anything to call or collect at all (a pass callback alone leaves the fused paths be)"""
        return self.active or self.on_pass is not None

    def per_token(self, on_token: Callable[[int], Any] | None) -> Callable[[int], Any]:
        """a one-token loop's token callback that also reports each token as its own pass"""
        on_pass = self.on_pass
        assert on_pass is not None
        k, last = 0, time.perf_counter()

        def cb(t: int) -> Any:
            nonlocal k, last
            now = time.perf_counter()
            on_pass({"index": k, "drafted": 0, "accepted": 0, "tokens": 1, "seconds": now - last})
            k, last = k + 1, now
            return on_token(t) if on_token is not None else None

        return cb

    def rows(self, n: int) -> None:
        while len(self.lp) < n:
            self.lp.append([])
            self.hidden.append({i: [] for i in self.taps})

    def child(self) -> Hooks:
        """the same hooks, collecting afresh (a sub-batch whose rows `extend` appends)"""
        return Hooks(self.processors, self.logprobs, self.taps, self.on_pass)

    def extend(self, other: Hooks) -> None:
        self.lp.extend(other.lp)
        self.hidden.extend(other.hidden)

    def process(self, ctx: Sequence[Sequence[int]], logits: torch.Tensor) -> torch.Tensor:
        """the processors over each row of `logits` [N, V], row n with ids `ctx[n]`"""
        if not self.processors:
            return logits
        out = logits.clone()
        for n, ids in enumerate(ctx):
            row = out[n]
            for p in self.processors:
                row = p(ids, row)
            out[n] = row
        return out

    def record(self, row: int, logits: torch.Tensor, token: int) -> None:
        """the drawn token's log-probability out of its row's processed logits [V]"""
        if self.logprobs is None:
            return
        lp = torch.log_softmax(logits.float(), dim=-1)
        top: tuple[tuple[int, float], ...] = ()
        if self.logprobs > 0:
            v, i = torch.topk(lp, min(int(self.logprobs), int(lp.shape[-1])))
            top = tuple((int(a), float(b)) for a, b in zip(i.tolist(), v.tolist()))
        self.lp[row].append(TokenLogprob(int(token), float(lp[int(token)]), top))

    def tap(self, row: int, layer: int, h: torch.Tensor) -> None:
        """one position's hidden state [H] of a tapped layer"""
        if layer in self.hidden[row]:
            self.hidden[row][layer].append(h.detach().float().cpu())

    def collector(self) -> tuple[Callable[[int, torch.Tensor], None], dict[int, torch.Tensor]]:
        """an `on_layer` that keeps the tapped layers' output of one pass, and the dict it fills"""
        seen: dict[int, torch.Tensor] = {}

        def on_layer(i: int, h: torch.Tensor) -> None:
            if i in self.taps:
                seen[i] = h

        return on_layer, seen
