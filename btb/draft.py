from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
class NGramProposer:
    """n-gram drafts from the prompt and the tokens so far: per order n in [n_min, n_max], each n-gram's followers
    with their count and latest position; `add_sequence` banks other texts. A chain copies the continuation of
    the best match; `propose_chains` gives the top `followers` at every order for a tree."""

    def __init__(
        self,
        corpus_ids: Iterable[int],
        n_max: int = 4,
        n_min: int = 2,
        followers: int = 4,
        bank: SpanBank | None = None,
    ) -> None:
        self.corpus: list[int] = []
        self.n_max, self.n_min = n_max, n_min
        self.followers = max(1, int(followers))
        # order -> n-gram -> follower token -> [count, position after its latest occurrence]
        self.maps: dict[int, dict[tuple[int, ...], dict[int, list[int]]]] = {n: {} for n in range(n_min, n_max + 1)}
        self.ext: dict[int, dict[tuple[int, ...], tuple[int, int]]] = {n: {} for n in range(n_min, n_max + 1)}
        self.seqs: list[list[int]] = []
        self.tags: list[Any] = []
        self.bank = bank  # a shared index, kept by the bank across requests, looked up after this one's own
        for t in corpus_ids:
            self.extend(int(t))

    def _banked(self, n: int, key: tuple[int, ...]) -> tuple[list[int], int, Any, int] | None:
        """(sequence, position after the match, tag, sid) of the latest banked continuation of `key` at order n:
        this proposer's own sequences first, then the bank's"""
        hit = self.ext[n].get(key)
        if hit is not None:
            sid, pos = hit
            return self.seqs[sid], pos, self.tags[sid], sid
        if self.bank is not None:
            return self.bank.lookup(n, key)
        return None

    def add_sequence(self, ids: Iterable[int], tag: Any) -> int:
        ids = [int(t) for t in ids]
        sid = len(self.seqs)
        self.seqs.append(ids)
        self.tags.append(tag)
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(ids) - n):
                self.ext[n][tuple(ids[i : i + n])] = (sid, i + n)
        return sid

    def extend(self, tok_id: int) -> None:
        self.corpus.append(tok_id)
        hi = len(self.corpus)
        for n in range(self.n_min, self.n_max + 1):
            i = hi - n - 1
            if i >= 0:
                key = tuple(self.corpus[i : i + n])
                e = self.maps[n].setdefault(key, {}).get(tok_id)
                if e is None:
                    self.maps[n][key][tok_id] = [1, i + n]
                else:
                    e[0] += 1
                    e[1] = i + n

    def _followers(self, n: int, key: tuple[int, ...]) -> list[tuple[int, int]]:
        """the tokens that followed `key` at order n, as (count, position after the latest occurrence), the
        most frequent first and the most recent among equals, at most `followers` of them"""
        d = self.maps[n].get(key)
        if not d:
            return []
        return sorted(((c, p) for c, p in d.values()), reverse=True)[: self.followers]

    def propose_chains(self, v: int) -> list[tuple[list[int], str]]:
        """Distinct continuations of up to `v` tokens for the tail: longest order first, most frequent follower
        first, sequences ahead of the corpus; a chain equal to or a prefix of an earlier one is dropped."""
        out: list[tuple[list[int], str]] = []
        if v <= 0:
            return out
        for n in range(self.n_max, self.n_min - 1, -1):
            if len(self.corpus) < n:
                continue
            key = tuple(self.corpus[-n:])
            cands = []
            hit = self._banked(n, key)
            if hit is not None:
                seq, pos, _tag, _sid = hit
                cands.append((seq[pos : pos + v], f"seq{n}"))
            for _c, at in self._followers(n, key):
                cands.append((self.corpus[at : at + v], f"ngram{n}"))
            for c, tag in cands:
                if c and not any(o[: len(c)] == c for o, _ in out):
                    out.append((list(c), tag))
        return out

    def propose_with_source(self, v: int) -> tuple[list[int], Any]:
        if v <= 0:
            return [], None
        for n in range(self.n_max, self.n_min - 1, -1):
            if len(self.corpus) < n:
                continue
            key = tuple(self.corpus[-n:])
            hit = self._banked(n, key)
            if hit is not None:
                seq, pos, tag, sid = hit
                return seq[pos : pos + v], (tag, sid, pos)
            fl = self._followers(n, key)
            if fl:
                at = fl[0][1]
                return self.corpus[at : at + v], at
        return [], None


class SpanBank:
    """(tag, token ids) spans a server banks for later requests' n-gram proposer, bounded by `max_tokens`,
    oldest out first, with the n-gram index (orders 1..`n_max`) kept here and extended as spans arrive, so a
    request's proposer looks it up instead of rebuilding it from every span. One bank per engine: key per tenant
    or conversation on a shared server."""

    def __init__(self, max_tokens: int = 1 << 20, n_max: int = 4) -> None:
        self.max_tokens = int(max_tokens)
        self.n_max = max(1, int(n_max))
        self._spans: dict[int, tuple[str, list[int]]] = {}  # sid -> (tag, ids), insertion order = age
        self._next = 0
        self.tokens = 0
        # order -> n-gram -> (sid, position after the latest occurrence); an evicted sid's entries are skipped on
        # lookup and swept when they outnumber the live spans' tokens
        self.ext: dict[int, dict[tuple[int, ...], tuple[int, int]]] = {n: {} for n in range(1, self.n_max + 1)}
        self._stale = 0

    def add(self, tag: str, ids: Iterable[int]) -> None:
        ids = [int(t) for t in ids]
        if not ids or len(ids) > self.max_tokens:
            return
        sid = self._next
        self._next += 1
        self._spans[sid] = (tag, ids)
        self.tokens += len(ids)
        for n in range(1, self.n_max + 1):
            ext = self.ext[n]
            for i in range(len(ids) - n):
                ext[tuple(ids[i : i + n])] = (sid, i + n)
        while self.tokens > self.max_tokens and self._spans:
            old_sid = next(iter(self._spans))
            _, old = self._spans.pop(old_sid)
            self.tokens -= len(old)
            self._stale += len(old)
        if self._stale > self.tokens:
            self._rebuild()

    def _rebuild(self) -> None:
        self.ext = {n: {} for n in range(1, self.n_max + 1)}
        for sid, (_tag, ids) in self._spans.items():
            for n in range(1, self.n_max + 1):
                ext = self.ext[n]
                for i in range(len(ids) - n):
                    ext[tuple(ids[i : i + n])] = (sid, i + n)
        self._stale = 0

    def lookup(self, n: int, key: tuple[int, ...]) -> tuple[list[int], int, Any, int] | None:
        """(sequence, position after the match, tag, sid) of the latest banked span continuing `key` at order n"""
        hit = self.ext.get(n, {}).get(key)
        if hit is None:
            return None
        sid, pos = hit
        span = self._spans.get(sid)
        if span is None:  # evicted; the entry goes at the next sweep
            return None
        return span[1], pos, span[0], sid

    def spans(self) -> list[tuple[str, list[int]]]:
        return list(self._spans.values())

    def __len__(self) -> int:
        return len(self._spans)


# what a generate call takes for the n-gram proposer: (tag, ids) pairs, or a standing bank it looks up
Spans = Sequence[Any] | SpanBank
