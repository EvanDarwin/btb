# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A conversation's KV rows in the pages of a pool every conversation on the engine shares (btb/engine/kvpool.py).

`HostPool` holds the rows on the host: per attention layer a K and a V region [Hk, rows, D], head-major as the native
kernels read a cache (head g's row r at g * rows * D + r * D), every layer grown together as the ledger grants it. A
conversation's `Table` is its row map - position j of the sequence at row `rows()[j]` of every layer's region - and the
pages it holds a reference on; a prefix two conversations share is the same rows, read in place by both. `PagedCache`
is the cache a session decodes over: its attention layers `PagedLayer`s over one table. The attention reads a layer's
rows through the map (`attn_nodes` takes each query's rows as a list, the decode step's bits for the rows in order), so a
row is never copied to be read; a path that would read the layer as one contiguous buffer meets a `PagedError` rather
than a silent gather.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from ..kinds import LayerKind
from .kvpool import PAGE, Page, PagePool
from .scheduler import MemoryGrantError

if TYPE_CHECKING:
    from .prefix import PrefixCache
    from .radix import RadixTree


class PagedError(RuntimeError):
    """a paged cache read as one contiguous buffer: a path the paged reader does not reach (yet)"""


class HostPool:
    """the pages' rows on the host, every attention layer's: K and V [Hk, rows, D] a layer (`hk` heads of `d`), in
    the rows' own dtype unless one is given, grown by GROW as the ledger grants it - a buffer at a time, as a layer
    cache grows its own, each let go once its rows are copied over. A growth the ledger refuses lets go of the tree's
    least recently used conversations first, their freed pages taking the new rows"""

    GROW = 1.5
    MIN_PAGES = 16

    def __init__(
        self,
        layers: Sequence[int],
        hk: int,
        d: int,
        grant: Callable[..., None] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.layers = [int(i) for i in layers]
        self.hk, self.d = int(hk), int(d)
        self.grant = grant
        self.pages = PagePool(grow=self._grow)
        self.cap = 0  # the pages the regions hold rows for
        self.kv_dtype = dtype  # the rows' dtype where it is not their own (a bf16 host's)
        self.dtype: torch.dtype | None = None  # the regions' dtype, from the first rows written
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.tree: RadixTree | None = None

    def _el(self) -> int:
        return torch.empty(0, dtype=self.dtype or self.kv_dtype or torch.float32).element_size()

    def _target(self, pages: int) -> int:
        """the pages a growth to hold `pages` makes room for: GROW past what the regions hold"""
        return max(int(pages), int(self.cap * self.GROW), self.MIN_PAGES)

    def nbytes(self) -> int:
        """what the regions hold"""
        return 2 * len(self.layers) * self.hk * self.cap * PAGE * self.d * self._el()

    def growth(self, pages: int) -> int:
        """the free memory a growth to hold `pages` pages takes at its peak: what the regions add, and the buffer a
        new one replaces, held until its rows are copied over (the last one's, every other let go by then); 0 where
        they hold them already"""
        if pages <= self.cap:
            return 0
        new = self._target(pages)
        row = self.hk * PAGE * self.d * self._el()
        return 2 * len(self.layers) * (new - self.cap) * row + self.cap * row

    def shape(self, k: torch.Tensor) -> None:
        """the rows' dtype from the first ones written, [B, Hk, T, D] - their heads and width the pool's"""
        if int(k.shape[1]) != self.hk or int(k.shape[-1]) != self.d:
            raise PagedError(
                f"rows of {int(k.shape[1])} heads of {int(k.shape[-1])} into a pool of {self.hk} of {self.d}"
            )
        if self.dtype is None:
            self.dtype = self.kv_dtype or k.dtype

    def _grow(self, pages: int) -> None:
        if pages <= self.cap:
            return
        if self.dtype is None:
            raise PagedError("the pool grew before its rows' dtype was known")
        new = self._target(pages)
        rows, el = new * PAGE, self._el()
        for i in self.layers:
            for store in (self.k, self.v):
                old = store.get(i)
                if old is not None and old.shape[1] >= rows:
                    continue  # grown by an attempt the ledger refused part way
                if self.grant is not None:
                    # free RAM alone: the expert store is a cache too, and the conversations least recently used go
                    # before its blocks do (`alloc`, `PrefixCache.room`)
                    self.grant(
                        self.hk * rows * self.d * el,
                        "kv",
                        requester=f"the prefix cache's layer {i}, {new} pages of {PAGE} rows",
                        device="cpu",
                        held=old.numel() * el if old is not None else 0,
                    )
                buf = torch.empty(self.hk, rows, self.d, dtype=self.dtype)
                if old is not None:
                    buf[:, : old.shape[1]] = old
                store[i] = buf
        self.cap = new

    def alloc(self, writer: object) -> Page:
        """a page for `writer` to append to: a free one, else the regions grown; refused, conversations the tree
        holds let go, least recently used first, until a page frees"""
        while True:
            try:
                return self.pages.alloc(writer)
            except MemoryGrantError:
                if self.tree is None or not self.tree.evict(enough=lambda freed: freed >= 1):
                    raise

    def write(self, i: int, rows: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """layer i's `k`, `v` [Hk, T, D] at `rows` (T of them)"""
        self.k[i][:, rows] = k.to(self.dtype)
        self.v[i][:, rows] = v.to(self.dtype)

    def move(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """every layer's rows at `src` copied to `dst` (the source read whole first, so the two may overlap)"""
        for store in (self.k, self.v):
            for buf in store.values():
                buf[:, dst] = buf[:, src]

    def close(self) -> None:
        """the regions let go with the engine: nothing reads the pool after"""
        self.k.clear()
        self.v.clear()
        self.cap = 0


class Table:
    """A conversation's rows: `rows()[j]` is the pool row holding its position j, in every layer. It holds a
    reference on each page it reads (`count` its rows there), and appends only to a page it is the writer of, past
    the rows another holder froze (kvpool's rules); a prefix taken from the tree is read in place, the conversation's
    own rows going on in pages of its own. Let go (`release`, or collected), its references go with it."""

    def __init__(self, pool: HostPool, rows: Sequence[int] = ()) -> None:
        self.pool = pool
        self.token = object()  # the pages' writer: not the table, which no page may keep alive
        self.n = 0
        self.version = 0  # moved by every change to the map: what is built from it (a pass's row lists) is keyed by it
        self._buf = torch.empty(0, dtype=torch.int64)
        self.held: dict[int, Page] = {}
        self.count: dict[int, int] = {}
        self.tail: Page | None = None
        self._fin = weakref.finalize(self, Table._let_go, pool.pages, self.held, self.count)
        self._take(rows)

    @staticmethod
    def _let_go(pages: PagePool, held: dict[int, Page], count: dict[int, int]) -> None:
        for p in list(held.values()):
            pages.unref(p)
        held.clear()
        count.clear()

    def release(self) -> None:
        """every page let go: the table reads nothing after (and what it holds again it lets go when collected)"""
        self._fin()
        self._fin = weakref.finalize(self, Table._let_go, self.pool.pages, self.held, self.count)
        self.n, self.tail = 0, None
        self.version += 1

    def __len__(self) -> int:
        return self.n

    def rows(self) -> torch.Tensor:
        """the row map: position j's pool row, int64, a view the next append may move"""
        return self._buf[: self.n]

    def _reserve(self, n: int) -> None:
        if n > len(self._buf):
            buf = torch.empty(max(n, 2 * len(self._buf), 256), dtype=torch.int64)
            buf[: self.n] = self._buf[: self.n]
            self._buf = buf

    def _take(self, rows: Sequence[int]) -> None:
        """`rows` read from another holder (the tree) appended: a hold on each page they lie in"""
        rows = [int(r) for r in rows]
        if not rows:
            return
        for r in rows:
            pid = r // PAGE
            if pid not in self.held:
                p = self.pool.pages.pages[pid]
                self.pool.pages.ref(p)
                self.held[pid] = p
            self.count[pid] = self.count.get(pid, 0) + 1
        self._reserve(self.n + len(rows))
        self._buf[self.n : self.n + len(rows)] = torch.tensor(rows, dtype=torch.int64)
        self.n += len(rows)
        self.version += 1

    def _own_tail(self) -> Page | None:
        """the page the table appends to: its tail, while it is the writer there and the page has room"""
        p = self.tail
        if p is None or p.writer is not self.token or p.fill >= PAGE or p.id not in self.held:
            return None
        return p

    def pages_for(self, T: int) -> int:
        """the new pages T more rows take: what the tail's room does not hold"""
        p = self._own_tail()
        past = int(T) - (PAGE - p.fill if p is not None else 0)
        return max(0, -(-past // PAGE))

    def extend(self, n: int) -> None:
        """rows for positions up to `n`: on the table's own tail page while it has room, else on new pages"""
        if n <= self.n:
            return
        self.version += 1
        self._reserve(n)
        while self.n < n:
            p = self._own_tail()
            if p is None:
                p = self.pool.alloc(self.token)
                self.held[p.id] = p
                self.tail = p
            take = min(PAGE - p.fill, n - self.n)
            first = p.id * PAGE + p.fill
            self._buf[self.n : self.n + take] = torch.arange(first, first + take, dtype=torch.int64)
            self.count[p.id] = self.count.get(p.id, 0) + take
            p.fill += take
            self.n += take

    def crop(self, n: int) -> None:
        """positions past `n` let go: a page the table no longer reads is unheld, and its own tail page takes its
        rows back where no other holder froze them (else it writes there no more)"""
        n = max(0, int(n))
        if n >= self.n:
            return
        self.version += 1
        gone = self._buf[n : self.n].tolist()
        self.n = n
        for r in gone:
            pid = r // PAGE
            c = self.count[pid] - 1
            if c:
                self.count[pid] = c
                continue
            del self.count[pid]
            self.pool.pages.unref(self.held.pop(pid))
        t = self.tail
        if t is None:
            return
        if t.id not in self.held:
            self.tail = None
            return
        last = int(self._buf[n - 1]) if n else -1
        end = last % PAGE + 1 if last >= 0 and last // PAGE == t.id else 0
        if t.writer is self.token and t.frozen <= end:
            t.fill = end
        else:
            self.tail = None

    def path_rows(self, base: int, path: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor] | None:
        """a verify pass's accepted path (`path` its nodes, the pass's rows from `base`): the rows to copy where the
        path's positions are, (from, to), None where they are there already"""
        if list(path) == list(range(len(path))):
            return None
        src = self._buf[torch.tensor([base + j for j in path], dtype=torch.int64)].clone()
        dst = self._buf[base : base + len(path)].clone()
        return src, dst


class PagedLayer(DynamicLayer):
    """one attention layer of a `PagedCache`: its rows the table's, written into its region of the pool. `append` is
    the paged reader's (the pool's region and the table's map back); `update` serves the first rows of a fresh prompt
    to a module, whose attention then reads them as they came - past them a module's read of the layer as one buffer
    is a `PagedError`, as `keys`/`values` are: `gather()` is the explicit copy (a fork's prefix, `Session.rows`)"""

    paged = True

    def __init__(self, table: Table, pool: HostPool, i: int) -> None:
        super().__init__()
        self.table, self.pool, self.i = table, pool, int(i)
        self.n = 0
        self.is_initialized = True

    @property
    def keys(self) -> torch.Tensor:
        raise PagedError(f"layer {self.i}'s rows are paged: read them through the row map, or gather() a copy")

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise PagedError(f"layer {self.i}'s rows are paged: written by append, not assigned")

    @property
    def values(self) -> torch.Tensor:
        raise PagedError(f"layer {self.i}'s rows are paged: read them through the row map, or gather() a copy")

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise PagedError(f"layer {self.i}'s rows are paged: written by append, not assigned")

    def get_seq_length(self) -> int:
        return self.n

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`k`, `v` [1, Hk, T, D] after the layer's rows: their rows on the table (made by whichever layer reaches
        them first), the pool's K and V regions [1, Hk, rows, D] back, read through `table.rows()`"""
        if int(k.shape[0]) != 1:
            raise PagedError(f"a paged cache holds one sequence, not {int(k.shape[0])}")
        T = int(k.shape[-2])
        self.pool.shape(k)
        self.table.extend(self.n + T)
        self.pool.write(self.i, self.table.rows()[self.n : self.n + T], k[0], v[0])
        self.n += T
        return self.pool.k[self.i][None], self.pool.v[self.i][None]

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self.n:
            raise PagedError(
                f"layer {self.i}: a module read a paged cache past its first rows, as one buffer - a path the paged "
                "reader does not take"
            )
        self.append(key_states, value_states)
        return key_states.to(self.pool.dtype), value_states.to(self.pool.dtype)

    def gather(self) -> tuple[torch.Tensor, torch.Tensor]:
        """the layer's rows in order as [1, Hk, n, D] tensors of their own: a copy"""
        rows = self.table.rows()[: self.n]
        return self.pool.k[self.i][:, rows][None], self.pool.v[self.i][:, rows][None]

    def crop(self, n: int) -> None:
        self.n = min(self.n, max(0, int(n)))


class PagedCache(DynamicCache):
    """a session's cache over the engine's pool: its attention layers `PagedLayer`s over one table, opened on the
    rows `rows` of a prefix the tree holds (read in place). `commit` gives the tree what the session made"""

    paged = True

    def __init__(self, prefix: PrefixCache, rows: Sequence[int] = ()) -> None:
        super().__init__(config=prefix.cfg)
        self.prefix = prefix
        self.table = Table(prefix.pool, rows)
        for i, lt in enumerate(prefix.layer_types):
            if lt != LayerKind.LINEAR:
                pl = PagedLayer(self.table, prefix.pool, i)
                pl.n = len(self.table)
                self.layers[i] = pl

    def growth(self, T: int) -> int:
        """the free memory the pool's growth for the next T rows takes (`HostPool.growth`): 0 where its free pages
        and the table's tail hold them"""
        pool = self.prefix.pool
        short = self.table.pages_for(T) - len(pool.pages.free)
        return pool.growth(len(pool.pages.pages) + short) if short > 0 else 0

    def crop(self, max_length: int) -> None:
        """transformers' crop (a negative length counting from the end), the table cut with the layers: a layer cut
        alone would leave the table's rows past it to be written again, rows the tree may read"""
        n = len(self.table)
        self.crop_to(n + max_length if max_length < 0 else max_length)

    def crop_to(self, n: int) -> None:
        """the sequence's first `n` positions kept, every layer's and the table's"""
        for cl in self.layers:
            if isinstance(cl, PagedLayer):
                cl.crop(n)
        self.table.crop(n)

    def keep_path(self, base: int, path: Sequence[int]) -> None:
        """a verify pass's accepted path into place (`path` its nodes, the pass's rows from `base`): each layer's
        accepted rows copied down to the path's positions in its own pages, the rest let go"""
        moved = self.table.path_rows(base, path)
        if moved is not None:
            self.prefix.pool.move(*moved)
        self.crop_to(base + len(path))

    def commit(self, ids: Sequence[int]) -> None:
        """the session's tokens and the rows holding them into the tree: what a later prompt opens on"""
        n = min(len(ids), len(self.table))
        if n:
            self.prefix.insert(ids[:n], self.table.rows()[:n].tolist())

    def release(self) -> None:
        self.table.release()
        for cl in self.layers:
            if isinstance(cl, PagedLayer):
                cl.n = 0
