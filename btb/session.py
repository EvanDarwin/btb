# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's state between calls, and what a new prompt reuses of it."""

from __future__ import annotations

from typing import Any

from .kinds import LayerKind, Tokens

# a chat template's generation tail is a handful of tokens (Qwen3's empty think block is four); a prompt that
# parts from the previous one further back than this is another conversation, and teaches no tail
TAIL_MAX = 64


class Session:
    """The tokens the cache holds and the cache, a hybrid's DeltaNet snapshots (`anchor`), the drafter's tail
    (`dr`, `dr_len`, `pend_h`) and the template's generation tail (`tail`, learned or given)."""

    def __init__(self, tail: int = 0) -> None:
        self.ids: list[int] = []
        self.n_prompt = 0
        self.cache: Any = None
        self.anchor: list[dict[str, Any]] = []
        self.dr: Any = None
        self.dr_len = 0
        self.pend_h: Any = None
        self.tail = int(tail)

    @property
    def fresh(self) -> bool:
        return self.cache is None

    def match(self, prompt: Tokens) -> int:
        """The prefix the new prompt shares with the cache's tokens (at most n - 1); learns `tail`, the distance
        before the previous prompt's end at which its re-rendering diverged."""
        n = len(prompt)
        lim = min(len(self.ids), n - 1)
        m = 0
        while m < lim and prompt[m] == self.ids[m]:
            m += 1
        if self.n_prompt and 0 < m < self.n_prompt <= len(self.ids) and self.n_prompt - m <= TAIL_MAX:
            self.tail = int(self.n_prompt - m)
        return m

    def anchored(self, m: int) -> dict[str, Any] | None:
        """the latest DeltaNet snapshot at or before the matched prefix, None when there is none"""
        best = None
        for a in self.anchor:
            if 0 < a["n"] <= m and (best is None or a["n"] > best["n"]):
                best = a
        return best

    def open(self, engine: Any, prompt: Tokens) -> tuple[Any, int, dict[str, Any] | None]:
        """What a new prompt reuses: (cache, rows reused, the snapshot restored), or (None, 0, None). A dense
        cache is cropped to the shared prefix; a hybrid's DeltaNet states cannot be cropped, so they come
        back from the latest snapshot inside it."""
        if self.cache is None:
            return None, 0, None
        n = len(prompt)
        m = self.match(prompt)
        cache, reuse, anchored = None, 0, None
        if 0 < len(self.ids) < n and m == len(self.ids):
            cache, reuse = self.cache, len(self.ids)
        elif LayerKind.LINEAR in engine.layer_types:
            anchored = self.anchored(m)
            if anchored is not None:
                cache, reuse = self.cache, int(anchored["n"])
                for i, snap in anchored["states"].items():
                    engine._lin_restore(cache.layers[i], snap)
        elif m > 0:
            cache, reuse = self.cache, m
        if cache is None:
            # nothing of the last conversation serves this one: its cache goes now, not when the new turn's
            # `keep` replaces it, or the new prefill runs beside a full cache of the old (7 GB twice at 40k); the
            # drafter's own cache, which the engine holds, goes with it (a gigabyte at 120k rows)
            dr = self.dr
            self.cache, self.anchor, self.ids, self.n_prompt = None, [], [], 0
            self.dr, self.dr_len, self.pend_h = None, 0, None
            if dr is not None and hasattr(dr, "reset"):
                dr.reset()
        if cache is not None:
            for i in range(engine.L):
                cl = cache.layers[i]
                if getattr(cl, "keys", None) is not None and cl.keys.shape[-2] > reuse:
                    cl.keys = cl.keys[..., :reuse, :]
                    cl.values = cl.values[..., :reuse, :]
                    if hasattr(cl, "cumulative_length"):
                        cl.cumulative_length = int(cl.keys.shape[-2])
                    if getattr(cl, "indexer_keys", None) is not None and cl.indexer_keys.numel():
                        cl.indexer_keys = cl.indexer_keys[:, :reuse]
        return cache, reuse, anchored

    def keep(
        self,
        prompt: Tokens,
        out: Tokens,
        cache: Any,
        anchors: Any,
        dr: Any = None,
        dr_len: int = 0,
        pend_h: Any = None,
    ) -> None:
        """what the next call finds: the cache holds the prompt and the answer but its last token"""
        self.ids = [*prompt, *out[:-1]]
        self.n_prompt = len(prompt)
        self.cache = cache
        self.anchor = list(anchors or ())
        self.dr, self.dr_len = dr, int(dr_len)
        self.pend_h = pend_h.detach().clone() if pend_h is not None else None
