# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's prefix cache: every conversation its sessions decode, in one pool of pages (btb/engine/paged.py) under
one tree over their tokens (btb/engine/radix.py). A session's commit puts its tokens and rows in the tree; a prompt
then opens on the longest prefix held - the session's own, or another conversation's read in place - so a side request
between two turns leaves the conversation where it was, and a system prompt two conversations share is held once.

`why_not(sm)` says why an engine has none yet, tier by tier as the phases land: the engine's sessions then keep the
contiguous cache they always had, never a copied prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..kinds import LayerKind
from ..options import Device
from .native import Native
from .paged import HostPool, PagedCache
from .radix import RadixTree

if TYPE_CHECKING:
    from .state import _State


class PrefixCache:
    def __init__(self, sm: _State) -> None:
        self.cfg = c = sm.cfg
        self.layer_types = list(sm.layer_types)
        sched = getattr(sm, "scheduler", None)
        layers = [i for i, lt in enumerate(self.layer_types) if lt != LayerKind.LINEAR]
        hq = int(c.num_attention_heads)
        hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        # the rows in the dtype a contiguous cache keeps them in (`new_cache`): a bf16 host's bf16, else the rows' own
        kv = torch.bfloat16 if sm.host_kv_dtype() == torch.bfloat16 else None
        self.pool = HostPool(layers, hk, d, sched.grant if sched is not None else None, dtype=kv)
        self.tree = RadixTree(self.pool.pages)
        self.pool.tree = self.tree

    @staticmethod
    def why_not(sm: Any) -> str | None:
        """why the engine's sessions keep a contiguous cache of their own instead, None where the prefix cache serves
        them: the tiers and families whose attention reads the pages land phase by phase"""
        if sm.dev.type != Device.CPU or getattr(sm, "mlx", None) is not None:
            return "the card's and MLX's attention read no pages yet: the prefix cache serves the CPU's alone"
        if LayerKind.LINEAR in sm.layer_types:
            return "a hybrid's recurrent states resume only from snapshots at message boundaries, which come later"
        if sm.fam.own:
            return "a family that brings its own layer (Qwen4) reads its rows through programs of its own"
        if not sm.fam.fast:
            return "the family's attention is its own module's (gpt-oss's sinks), which reads one contiguous buffer"
        if not sm.flat_cache():
            from transformers.cache_utils import DynamicCache, DynamicSlidingWindowLayer

            if any(isinstance(cl, DynamicSlidingWindowLayer) for cl in DynamicCache(config=sm.cfg).layers):
                return (
                    "the contiguous cache lets a sliding layer's rows past its window go (Phi-3-mini-4k's 2047) and reads "
                    "what is left, where the pages keep every row"
                )
        if Native.attn_spans is None or Native.attn_nodes is None:
            return "the native library's attention, which reads the pages by row, is not loaded"
        return None

    def new(self, rows: Sequence[int] = ()) -> PagedCache:
        """a cache over the pool, opened on `rows` - a prefix the tree holds, read in place - or empty"""
        return PagedCache(self, rows)

    def insert(self, ids: Sequence[int], rows: Sequence[int]) -> None:
        self.tree.insert(ids, rows)

    def room(self, cache: PagedCache, T: int, free: Callable[[], int | None]) -> None:
        """the pages `cache`'s next T rows take, made of the conversations the tree holds where the pool's growth
        would not be granted (`free()` the room the grant reads): least recently used first, until the free pages
        hold the rows or the growth fits - before a pass, so the engine gives up nothing else of its own for rows
        an old conversation's pages can take"""
        while True:
            need = cache.growth(T)
            room = free() if need else None
            if room is None or need <= room:
                return
            if not self.tree.evict(enough=lambda freed: freed >= 1):
                return

    def close(self) -> None:
        """everything let go with the engine: the tree's conversations, and the pool's rows whoever reads them"""
        self.tree.evict()
        self.pool.close()
