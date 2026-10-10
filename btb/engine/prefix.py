# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's prefix cache: every conversation its sessions decode, in one pool of pages (btb/engine/paged.py) under
one tree over their tokens (btb/engine/radix.py). A session's commit puts its tokens and rows in the tree - held back
until the tree is next read, so a session cut back before then never froze what it cuts; a prompt then opens on the
longest prefix held - the session's own, or another conversation's read in place - so a side request
between two turns leaves the conversation where it was, and a system prompt two conversations share is held once.

The rows of the layers the card runs lie on the card, in the reserved room the conversation the card decodes takes
(`CardRegion`); the other conversations' rows of those layers stay there until their room is wanted, then wait in
pinned RAM, least recently used first, until one of them is decoded again.
The host's layers' rows lie in RAM (`HostRegion`). A layer moving between the two takes its rows along (`place`).

`why_not(sm)` says why an engine has none yet, tier by tier as the phases land: the engine's sessions then keep the
contiguous cache they always had, never a copied prefix.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ..kinds import LayerKind
from ..options import Device
from .native import Native
from .paged import KvPool, PagedCache, Table
from .radix import Match, RadixTree
from .scheduler import MemoryGrantError

if TYPE_CHECKING:
    from .state import _State

# the head widths the card's one attention is built for (btb_attn_flash.cuh: its prefill and decode forms)
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
        self.grant = sched.grant if sched is not None else None
        self.tree = RadixTree(self.pool.pages)
        self.pool.tree = self.tree
        # the sessions' commits held back for the tree on their tables (`PagedCache.commit`), put in when it is next
        # read (`flush`)
        self._held: dict[int, Table] = {}
        self._asking: Table | None = None
        self.tree.before = self.flush
        # the row lists the host's attention reads a pass's rows by (`_CudaMixin._pass_lists`): the last conversation's
        # alone - its table, the map's version they were made for, the lists
        self._lists: tuple[weakref.ref[Table], int, dict[Any, Any]] | None = None

    @staticmethod
    def why_not(sm: Any) -> str | None:
        """why the engine's sessions keep a contiguous cache of their own instead, None where the prefix cache serves
        them: the tiers and families whose attention reads the pages land phase by phase"""
        if getattr(sm, "mlx", None) is not None:
            return "MLX's attention reads no pages yet: the prefix cache serves the CPU's and the card's"
        if sm.dev.type not in (Device.CPU, Device.CUDA):
            return f"a {sm.dev.type} engine's attention reads no pages, only the host's and the card's do"
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
        # loaded, they are all there (the loader takes every kernel or none), every head width above among them
        return None

    def new(self, rows: Sequence[int] = ()) -> PagedCache:
        """a cache over the pool, opened on `rows` - a prefix the tree holds, read in place - or empty"""
        return PagedCache(self, rows)

    def insert(self, ids: Sequence[int], rows: Sequence[int], snaps: Mapping[int, Any] | None = None) -> None:
        self.tree.insert(ids, rows, snaps)

    def hold(self, table: Table) -> None:
        """the commit on `table` (`held_ids`) held back for the tree until it is read (`flush`): put in then, the rows
        frozen for every reader; a session cut back meanwhile (a rewind, a regenerate) cuts what it holds back too, and
        writes on in its own page - frozen at once, every cut left that page to the tree and took a fresh one (a
        mark/feed/rewind loop wrote two rows a page). A session's commits come in once for many feeds, not its whole
        path each feed. The table is held, which holds the pages: the session's cache is collected as it would be"""
        self._held[id(table)] = table

    def let_go(self, table: Table) -> None:
        self._held.pop(id(table), None)

    def put(self, table: Table) -> None:
        """the commit held back on `table` into the tree now: its rows made on the route alone (`Table.exact`). A
        hybrid's up to its deepest snapshot made cold (`Table.cold`), each snapshot at its node's end: the rows past
        it no prompt resumes on - its states are kept nowhere further"""
        self._held.pop(id(table), None)
        ids, table.held_ids = table.held_ids, None
        snaps, table.held_snaps = table.held_snaps, None
        if ids and table.exact is not None:
            ids = ids[: table.exact]
        if ids and table.cold is not None:
            snaps = {k: s for k, s in (snaps or {}).items() if k <= min(len(ids), table.cold)}
            ids = ids[: max(snaps, default=0)]
        if ids:
            self.insert(ids, table.rows()[: len(ids)].tolist(), snaps)

    def snap_room(self, nbytes: int) -> bool:
        """room in RAM for a hybrid's snapshot of `nbytes` for the tree: granted, the snapshots the tree keeps let go
        least recently used first while the grant refuses it (their nodes stay, read to an earlier one). False where
        none is left to let go: the snapshot is not kept, and a prompt resumes from an earlier one"""
        if self.grant is None:
            return True
        while True:
            try:
                self.grant(
                    nbytes, "kv", requester="a hybrid's states at a block's end, for the prefix tree", device="cpu"
                )
                return True
            except MemoryGrantError:
                if not self.tree.drop_snaps(lambda dropped: dropped >= 1):
                    return False

    def reading(self, table: Table) -> None:
        """a pass over `table` begins: the row lists another conversation's pass left (`_lists`) let go. Made only by
        the host's attention, a pass whose layers never read them (the card's, under kv_host or a split placement)
        left the idle conversation's held"""
        held = self._lists
        if held is not None and held[0]() is not table:
            self._lists = None

    def flush(self) -> None:
        """every commit held back put in the tree, but the asking session's own (`match`): what it parts from it keeps
        or drops itself"""
        for t in list(self._held.values()):
            if t is not self._asking:
                self.put(t)

    def match(self, tokens: Sequence[int], asking: PagedCache | None = None) -> Match:
        """how much of `tokens` the tree holds (`RadixTree.match`), every session's commits in it first - but
        `asking`'s, the session asking (its own rows it reads itself)"""
        self._asking = asking.table if asking is not None else None
        try:
            return self.tree.match(tokens)
        finally:
            self._asking = None

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
        self._held.clear()
        self.pool.close()
