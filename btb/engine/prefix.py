# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's prefix cache: every conversation its sessions decode, in one pool of pages (btb/engine/paged.py) under
one tree over their tokens (btb/engine/radix.py). A session's commit puts its tokens and rows in the tree; a prompt
then opens on the longest prefix held - the session's own, or another conversation's read in place - so a side request
between two turns leaves the conversation where it was, and a system prompt two conversations share is held once.

The rows of the layers the card runs lie on the card, in the reserved room the conversation the card decodes takes
(`CardRegion`); the other conversations' rows of those layers wait in pinned RAM until one of them is decoded again.
The host's layers' rows lie in RAM (`HostRegion`). A layer moving between the two takes its rows along (`place`).

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
from .paged import KvPool, PagedCache
from .radix import RadixTree

if TYPE_CHECKING:
    from .state import _State

# the head widths the card's attention kernels are built for (btb_attn_prefill_d*, btb_attn_split_tbl_d*)
CARD_HEAD_DIMS = (64, 128, 256)


def card_runs(sm: Any, i: int) -> bool:
    """whether layer i's rows lie on the card: an attention layer the card runs, its rows not kept in RAM (`kv_host`)"""
    return (
        sm.dev.type == Device.CUDA
        and sm.layer_types[i] != LayerKind.LINEAR
        and i in sm.resident
        and not getattr(sm, "kv_host", False)
    )


class PrefixCache:
    def __init__(self, sm: _State) -> None:
        self.cfg = c = sm.cfg
        self.layer_types = list(sm.layer_types)
        sched = getattr(sm, "scheduler", None)
        layers = [i for i, lt in enumerate(self.layer_types) if lt != LayerKind.LINEAR]
        card = [i for i in layers if card_runs(sm, i)]
        hq = int(c.num_attention_heads)
        hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        on_card = sm.dev.type == Device.CUDA
        # the host's rows in the dtype a contiguous cache keeps them in (`new_cache`): on a card engine its host
        # layers' (`host_kv_dtype`), elsewhere a bf16 host's bf16, else the rows' own
        if on_card:
            kv: torch.dtype | None = sm.host_kv_dtype()
        else:
            kv = torch.bfloat16 if sm.host_kv_dtype() == torch.bfloat16 else None
        ceiling = int(sm._card_ceiling()) if on_card and hasattr(sm, "_card_ceiling") else 1
        self.pool = KvPool(
            [i for i in layers if i not in card],
            hk,
            d,
            sched.grant if sched is not None else None,
            dtype=kv,
            card_layers=card,
            dev=sm.dev if on_card else None,
            ceiling=ceiling,
        )
        self.tree = RadixTree(self.pool.pages)
        self.pool.tree = self.tree

    @staticmethod
    def why_not(sm: Any) -> str | None:
        """why the engine's sessions keep a contiguous cache of their own instead, None where the prefix cache serves
        them: the tiers and families whose attention reads the pages land phase by phase"""
        if getattr(sm, "mlx", None) is not None:
            return "MLX's attention reads no pages yet: the prefix cache serves the CPU's and the card's"
        if sm.dev.type not in (Device.CPU, Device.CUDA):
            return f"a {sm.dev.type} engine's attention reads no pages, only the host's and the card's do"
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
        if sm.dev.type == Device.CUDA:
            return PrefixCache._card_why_not(sm)
        return None

    @staticmethod
    def _card_why_not(sm: Any) -> str | None:
        """the card's own conditions: its attention through btb's kernels, which read the pages through the row map"""
        c = sm.cfg
        hq = int(c.num_attention_heads)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        if d not in CARD_HEAD_DIMS:
            return f"the card's paged attention kernels take heads of {CARD_HEAD_DIMS} dims, not {d}"
        if sm.compute_dtype not in (None, torch.bfloat16) or getattr(sm, "resident_fp32", False) or sm.shadow:
            return "the card's paged attention kernels read bf16 rows, and this engine computes its card layers wider"
        if getattr(c, "_attn_implementation", None) != "btb_sdpa":
            return "the card layers' modules attend through another attention than btb's, which reads no pages"
        k = Native.card_kernels()
        if k is None:
            return f"the card's kernels are not loaded ({Native.cuda_reason})"
        missing = [n for n in (f"btb_attn_prefill_d{d}", f"btb_attn_split_tbl_d{d}") if n not in k.fn]
        if missing:
            return f"the card's kernels lack {', '.join(missing)}"
        return None

    def new(self, rows: Sequence[int] = ()) -> PagedCache:
        """a cache over the pool, opened on `rows` - a prefix the tree holds, read in place - or empty"""
        return PagedCache(self, rows)

    def insert(self, ids: Sequence[int], rows: Sequence[int]) -> None:
        self.tree.insert(ids, rows)

    def place(self, sm: Any, i: int) -> None:
        """layer i's rows to the region where the engine now runs it (a shed's, a regrow's): every conversation's at
        once"""
        if self.layer_types[i] != LayerKind.LINEAR:
            self.pool.rehome(i, card_runs(sm, i))

    def room(self, cache: PagedCache, T: int, free: Callable[[], int | None]) -> None:
        """the pages `cache`'s next T rows take, made of the conversations the tree holds where the pool's growth
        in RAM would not be granted (`free()` the room the grant reads): least recently used first, until the free
        pages hold the rows or the growth fits - before a pass, so the engine gives up nothing else of its own for
        rows an old conversation's pages can take"""
        while True:
            need = cache.growth(T).get("cpu", 0)
            room = free() if need else None
            if room is None or need <= room:
                return
            if not self.tree.evict(enough=lambda freed: freed >= 1):
                return

    def close(self) -> None:
        """everything let go with the engine: the tree's conversations, and the pool's rows whoever reads them"""
        self.tree.evict()
        self.pool.close()
