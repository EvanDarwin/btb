# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The attention cache layer grown in place, in torch's memory or in MLX's (the shared buffer the GPU appends to),
and a fork's layer over another cache's rows."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from .. import mlx as mlxdev

if TYPE_CHECKING:
    import mlx.core as mx_
    from transformers.cache_utils import CacheLayerMixin, DynamicCache, LinearAttentionCacheLayerMixin
    from transformers.cache_utils import DynamicLayer as _DynamicLayer

    # a sequence's cache: transformers' DynamicCache over the engine's layers (GrowLayer, a fork's, a hybrid's)
    KvCache = DynamicCache
    # one layer of it: an attention layer's rows, or a linear-attention layer's recurrent states
    CacheLayer = CacheLayerMixin | LinearAttentionCacheLayerMixin
else:
    # transformers is imported by name here and not at the top: the engine package is imported for its
    # discovery and planning too, where transformers' import time is not wanted
    _DynamicLayer = __import__("transformers").cache_utils.DynamicLayer

# a linear-attention layer's states copied out (a mark's, an anchor's): its conv and recurrent states, per state
# index where the layer keeps several
LinearStates = tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor]


def linear_layer(cl: CacheLayer) -> LinearAttentionCacheLayerMixin:
    """`cl` as the linear-attention layer a hybrid's linear index holds; a TypeError for any other"""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    if not isinstance(cl, LinearAttentionCacheLayerMixin):
        raise TypeError(f"a linear-attention layer's states were asked of a {type(cl).__name__}")
    return cl


def attention_rows(cl: CacheLayer) -> tuple[torch.Tensor, torch.Tensor]:
    """an attention layer's keys and values; a TypeError for a layer that holds none"""
    from transformers.cache_utils import CacheLayerMixin

    if not isinstance(cl, CacheLayerMixin) or cl.keys is None or cl.values is None:
        raise TypeError(f"an attention layer's rows were asked of a {type(cl).__name__} holding none")
    return cl.keys, cl.values


def indexer_keys(cl: CacheLayer) -> torch.Tensor | None:
    """a sparse-attention layer's indexer keys [B, n, d_index]; None for any other layer"""
    from transformers.cache_utils import DynamicIndexedLayer

    return cl.indexer_keys if isinstance(cl, DynamicIndexedLayer) else None


@runtime_checkable
class GraphStates(Protocol):
    """a linear layer whose states the pipelined MLX decode carries in its graph between steps (btb's attributes
    on transformers' layer): this step's (conv, recurrent) pair and the step before's"""

    _mx_pending: tuple[mx_.array, mx_.array] | None
    _mx_prev: tuple[mx_.array, mx_.array] | None


# the caches a fork or a batch built (`forked`), held weakly
_FORKS: weakref.WeakSet[KvCache] = weakref.WeakSet()


def mark_forked(cache: KvCache) -> KvCache:
    """`cache` as a fork's or a batch's: the single-row paths stand aside for it"""
    _FORKS.add(cache)
    return cache


class GrowLayer(_DynamicLayer):
    """One layer's attention cache grown in place: a [B, Hk, cap, d] buffer with room ahead. With `shared=True`
    the buffer is MLX's in unified memory and torch sees it through transient views (`keys`/`values`); no
    view may be held across an MLX append, and a crop or gather assigned from torch is written in by the
    next append."""

    # the buffers and their bookkeeping, None until the first append (their shapes come with the first rows)
    _mx: Any
    _an: int | None  # the front-row count when the buffer is the card's arena, else None
    _mx2: Any
    _shape: Any
    _keys_t: Any
    _values_t: Any
    _tk: Any
    _tv: Any
    _buf: Any
    _tmp: Any
    _ptr: Any
    _ns: Any
    _seg: Any
    _offs: Any
    _row: Any
    _n: int
    _b: int
    _t: int
    _dec_cap: int
    _flat: bool
    _flat_len: int
    _rows_total: int

    def __init__(
        self,
        reserve: int = 0,
        shared: bool = False,
        bits: int | None = None,
        cap_hint: int = 0,
        grant: Callable[..., None] | None = None,
        bound: int = 0,
        arena: tuple[Any, int, int, int] | None = None,
    ) -> None:
        # `grant` is the scheduler's gate (BatchScheduler.grant), asked on the growth branch alone - never per
        # token - so a buffer too large for the card, or one growing past `bound` (the length these rows can
        # reach), is refused here with a diagnostic instead of OOM-ing inside torch's allocator
        self.grant = grant
        self.bound = int(bound)
        self.shared = bool(shared)
        # bits = 8 (shared layers): int8 rows with a float32 scale each (`_mx = [k, v, ks, vs]`); torch's view is a
        # dequantized copy (`_view`)
        self.bits = int(bits) if bits else None
        self.cap_hint = int(cap_hint)
        # (buffer, K offset, V offset, rows a head): the layer's K and V are views of one shared array (the
        # megakernel's arena), written in place by kernels (`btb.mlx.kv`)
        self.arena = arena
        # a batched cache (the MLX device): `_ns[b]` is row b's own length, `_row` the row a single-sequence
        # forward is writing (its prefill), `_rows_total` the batch the buffer is sized for
        self._ns = None
        self._row = None
        self._rows_total = 0
        # a flat batched cache: every row's prompt end to end in one buffer (row b's rows from `_offs[b]`),
        # the decode steps in the step buffer `_mx2` (slot t of every row is step t), `_seg` the boundary
        self._flat = False
        self._offs = None
        self._flat_len = 0
        self._seg = None
        self._mx2 = None
        self._dec_cap = 0
        self._t = 0
        self._buf = None
        self._keys_t = None
        self._values_t = None
        # the rows' count when they are the front of `_buf` (the card's arena): the views are cut on read,
        # so a graph step advances every layer by one integer instead of building three views a layer
        self._an = None
        self._mx = None
        self._n = 0
        self._tk = None
        self._tv = None
        self._ptr = (0, 0)
        self._tmp = [None, None]
        self._shape = None
        self._b = 0
        super().__init__()
        self.reserve = int(reserve)

    # -- torch's view of the cache --
    @property
    def keys(self) -> torch.Tensor:
        if self._an is not None:
            return self._buf[0][..., : self._an, :]
        if not self.shared or self._mx is None:
            return self._keys_t
        if self._tk is not None:
            return self._tk
        return self._view(0)[..., : self._n, :]

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        self._put(0, self._owned(0, t))

    @property
    def values(self) -> torch.Tensor:
        if self._an is not None:
            return self._buf[1][..., : self._an, :]
        if not self.shared or self._mx is None:
            return self._values_t
        if self._tv is not None:
            return self._tv
        return self._view(1)[..., : self._n, :]

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        self._put(1, self._owned(1, t))

    def _set_rows(self, k: torch.Tensor | None, v: torch.Tensor | None) -> None:
        """btb's own write of the rows, taken as they are: tensors it just made, or cut from this layer's buffer"""
        self._put(0, k)
        self._put(1, v)

    def _put(self, which: int, t: torch.Tensor | None) -> None:
        if not self.shared or self._mx is None:
            if which == 0:
                self._keys_t = t
            else:
                self._values_t = t
            self._an = self._front_len(which, t)
        else:
            self._assign(which, t)

    def _owned(self, which: int, t: torch.Tensor | None) -> torch.Tensor | None:
        """`t` as rows the layer may keep: a cut from the front of its own buffer, or of rows it holds apart from
        one, stays a view; anything else (another cache's rows, the card's arena, MLX memory) is copied, so no
        later write elsewhere, reallocation or change of the arena's owner can reach the layer's rows"""
        if t is None or not t.numel():
            return t
        if t.ndim != 4:
            raise ValueError(f"a cache layer's rows are [batch, kv heads, positions, head dim], got {tuple(t.shape)}")
        if self.shared and self._mx is not None:
            keep = self._mx_prefix(which, t)
        elif self._front_len(which, t) is not None:
            keep = True
        else:
            cur = self._keys_t if which == 0 else self._values_t
            keep = (
                self._an is None
                and isinstance(cur, torch.Tensor)
                and bool(cur.numel())
                and not (self._buf is not None and _same_storage(t, self._buf[which]))
                and _inside(t, cur)
            )
        return t if keep else t.clone(memory_format=torch.contiguous_format)

    def _front_len(self, which: int, t: torch.Tensor | None) -> int | None:
        """the rows' count when `t` is the front of the buffer (a view from its first row), else None"""
        b = self._buf
        if b is None or t is None or not t.numel():
            return None
        kb = b[which]
        if (
            t.data_ptr() == kb.data_ptr()
            and t.device == kb.device
            and t.dtype == kb.dtype
            and t.shape[:2] == kb.shape[:2]
            and t.shape[-1] == kb.shape[-1]
            and t.stride() == kb.stride()
        ):
            return int(t.shape[-2])
        return None

    def get_seq_length(self) -> int:
        if self._an is not None:
            return int(self._an)
        if self.shared and self._mx is not None:
            return int(self._tk.shape[-2]) if self._tk is not None else int(self._n)
        return super().get_seq_length()

    def _view(self, which: int) -> torch.Tensor:
        if self.bits:
            # the live rows dequantized to bf16, a copy; kept so that a prefix of it handed back through the
            # setter is recognized as a crop and not written back
            t = mlxdev.from_mx(
                mlxdev.kv_dequantize(
                    self._mx[which][: self._b, :, : self._n], self._mx[2 + which][: self._b, :, : self._n]
                )
            )
            self._tmp[which] = t
            return t
        return mlxdev.from_mx(self._mx[which])[: self._b]

    def mx_kv(self, start: int, n: int, row: int = 0) -> tuple[mx_.array, mx_.array]:
        """the cache rows [start, n) of every head of batch row `row` as bf16 MLX arrays (K, V): views for a
        bf16 layer, a dequantized copy for an int8 one"""
        if self._flat:
            # the row's rows are a stretch of the one flat buffer
            off = int(self._offs[int(row)])
            r, start, n = slice(0, 1), off + int(start), off + int(n)
        else:
            r = slice(int(row), int(row) + 1)
        if self.bits:
            return (
                mlxdev.kv_dequantize(self._mx[0][r, :, start:n], self._mx[2][r, :, start:n]),
                mlxdev.kv_dequantize(self._mx[1][r, :, start:n], self._mx[3][r, :, start:n]),
            )
        return self._mx[0][r, :, start:n], self._mx[1][r, :, start:n]

    def _store(self, which: int, n0: int, a: mx_.array, row: int | None = None) -> None:
        """rows n0.. of buffer `which` from an MLX [B, Hk, T, d] array (into batch row `row` when given, at
        the row's offset of a flat buffer), quantized for an int8 layer"""
        B, T = int(a.shape[0]), int(a.shape[2])
        if self._flat:
            if row is not None:
                n0 = int(n0) + int(self._offs[int(row)])
            r = slice(0, 1)
        else:
            r = slice(0, B) if row is None else slice(int(row), int(row) + 1)
        if self.bits:
            q, s = mlxdev.kv_quantize(a)
            self._mx[which][r, :, n0 : n0 + T, :] = q
            self._mx[2 + which][r, :, n0 : n0 + T] = s
            return
        if self._mx[which].dtype != a.dtype:
            a = a.astype(self._mx[which].dtype)
        if self.arena is not None:
            # in place, evaluated now: the rows have no graph dependency to order them before their readers
            mlxdev.mx().eval(mlxdev.kv_store(self._mx[which], a[0], int(n0)))
            return
        self._mx[which][r, :, n0 : n0 + T, :] = a

    # -- a batched cache: B rows, each at its own length --
    def batch_rows(self, B: int, dec_cap: int = 0, lens: Any = None) -> None:
        """Size the layer for B rows at their own lengths. `dec_cap`: the decode steps go to a step buffer where
        step t is slot t for every row; `lens`: the prompts share one flat buffer, row b's from its offset."""
        self._ns = [0] * int(B)
        self._rows_total = int(B)
        self._row = None
        self._n = 0
        self._seg = None
        self._mx2 = None
        self._dec_cap = int(dec_cap)
        self._t = 0
        self._flat = lens is not None
        self._offs = None
        self._flat_len = 0
        if self._flat:
            offs, tot = [], 0
            for L in lens:
                offs.append(tot)
                tot += int(L)
            self._offs, self._flat_len = offs, tot
            self._mx = None
            self._shape = None
            self._tk = self._tv = None

    def _ensure_flat(self, Hk: int, d: int, dtype: torch.dtype) -> None:
        """the flat buffer of a batched cache, allocated on its first write (the rows' total length is known)"""
        m = mlxdev.mx()
        if self._mx is not None:
            return
        mdt = m.int8 if self.bits else mlxdev._dtypes()[dtype]
        n = max(1, int(self._flat_len))
        arrays = [m.zeros((1, Hk, n, d), dtype=mdt), m.zeros((1, Hk, n, d), dtype=mdt)]
        if self.bits:
            arrays += [m.ones((1, Hk, n), dtype=m.float32), m.ones((1, Hk, n), dtype=m.float32)]
        m.eval(*arrays)
        self._mx = arrays
        self._shape = (1, Hk, n, d)
        self._b = 1
        self._ptr = (
            mlxdev.from_mx(arrays[0]).untyped_storage().data_ptr(),
            mlxdev.from_mx(arrays[1]).untyped_storage().data_ptr(),
        )

    def forest_store(self, k: mx_.array, v: mx_.array, node0: int) -> None:
        """a batched prefill's rows for this layer: `k`/`v` [T, Hk, d], the T prompt tokens of consecutive
        rows laid end to end from flat row `node0` - one slab write per buffer for the whole group"""
        if not self.is_initialized:
            empty = torch.empty(0, dtype=mlxdev.torch_dtype(k.dtype))
            self.lazy_initialization(empty, empty)
        _T, Hk, d = (int(x) for x in k.shape)
        self._ensure_flat(Hk, d, mlxdev.torch_dtype(k.dtype))
        self._store(0, int(node0), k.transpose(1, 0, 2)[None])
        self._store(1, int(node0), v.transpose(1, 0, 2)[None])

    def select_row(self, b: int | None) -> None:
        """make a single-sequence forward read and write row b alone (its prefill); None returns the layer
        to the batch, its length the longest row's, the rows' prefill lengths fixed as the segment boundary"""
        self._row = None if b is None else int(b)
        if self._ns:
            self._n = self._ns[self._row] if self._row is not None else max(self._ns)
            if self._row is None and self._dec_cap:
                self._seg = list(self._ns)

    def mx_update_rows(self, k: mx_.array, v: mx_.array) -> None:
        """One decode step for every row: `k`/`v` [B, Hk, 1, d] into the step buffer's slot t (one write per
        buffer), else each row at its own length by scatter. Every length grows by one."""
        m = mlxdev.mx()
        dt = mlxdev.torch_dtype(k.dtype)
        if not self.is_initialized:
            empty = torch.empty(0, dtype=dt)
            self.lazy_initialization(empty, empty)
        B, Hk, T, d = (int(x) for x in k.shape)
        if self._dec_cap:
            if self._mx2 is None:
                mdt = m.int8 if self.bits else self._mx[0].dtype
                cap2 = max(self._dec_cap, T)
                self._mx2 = [m.zeros((B, Hk, cap2, d), dtype=mdt), m.zeros((B, Hk, cap2, d), dtype=mdt)]
                if self.bits:
                    self._mx2 += [m.ones((B, Hk, cap2), dtype=m.float32), m.ones((B, Hk, cap2), dtype=m.float32)]
            t = self._t
            if t + T > int(self._mx2[0].shape[2]):
                # a fork's rows stepped past the room they were given: twice the room, the steps so far kept
                cap2 = max(t + T, 2 * int(self._mx2[0].shape[2]))
                grown = []
                for x in self._mx2:
                    y = (m.ones if x.ndim == 3 else m.zeros)((*x.shape[:2], cap2, *x.shape[3:]), dtype=x.dtype)
                    y[:, :, :t] = x[:, :, :t]
                    grown.append(y)
                m.eval(*grown)
                self._mx2 = grown
            if self.bits:
                qk, sk = mlxdev.kv_quantize(k)
                qv, sv = mlxdev.kv_quantize(v)
                self._mx2[0][:, :, t : t + T, :] = qk
                self._mx2[1][:, :, t : t + T, :] = qv
                self._mx2[2][:, :, t : t + T] = sk
                self._mx2[3][:, :, t : t + T] = sv
            else:
                if self._mx2[0].dtype != k.dtype:
                    k, v = k.astype(self._mx2[0].dtype), v.astype(self._mx2[0].dtype)
                self._mx2[0][:, :, t : t + T, :] = k
                self._mx2[1][:, :, t : t + T, :] = v
            self._t = t + T
            self._ns = [n + T for n in self._ns]
            self._n = max(self._ns)
            return
        self._ensure(B, Hk, max(self._ns) + T, d, dt)
        self._apply_overrides()
        ns = m.array(self._ns, dtype=m.uint32)
        idx = ns.reshape(B, 1, 1, 1) * m.ones((B, Hk, T, d), dtype=m.uint32)
        if self.bits:
            qk, sk = mlxdev.kv_quantize(k)
            qv, sv = mlxdev.kv_quantize(v)
            sidx = ns.reshape(B, 1, 1) * m.ones((B, Hk, T), dtype=m.uint32)
            self._mx[0] = m.put_along_axis(self._mx[0], idx, qk, axis=2)
            self._mx[1] = m.put_along_axis(self._mx[1], idx, qv, axis=2)
            self._mx[2] = m.put_along_axis(self._mx[2], sidx, sk, axis=2)
            self._mx[3] = m.put_along_axis(self._mx[3], sidx, sv, axis=2)
        else:
            if self._mx[0].dtype != k.dtype:
                k, v = k.astype(self._mx[0].dtype), v.astype(self._mx[0].dtype)
            self._mx[0] = m.put_along_axis(self._mx[0], idx, k, axis=2)
            self._mx[1] = m.put_along_axis(self._mx[1], idx, v, axis=2)
        self._ns = [n + T for n in self._ns]
        self._n = max(self._ns)

    def mx_kv_row(self, b: int) -> tuple[mx_.array, mx_.array]:
        """row b's live rows as bf16 (K, V) [1, Hk, n_b, d]: its prefill slice, and its decode steps from
        the step buffer when the cache has one (a copy then)"""
        m = mlxdev.mx()
        if self._mx2 is None or self._seg is None:
            return self.mx_kv(0, int(self._ns[b]), row=b)
        s, t = int(self._seg[b]), int(self._ns[b]) - int(self._seg[b])
        K0, V0 = self.mx_kv(0, s, row=b)
        if self.bits:
            K1 = mlxdev.kv_dequantize(self._mx2[0][b : b + 1, :, :t], self._mx2[2][b : b + 1, :, :t])
            V1 = mlxdev.kv_dequantize(self._mx2[1][b : b + 1, :, :t], self._mx2[3][b : b + 1, :, :t])
        else:
            K1, V1 = self._mx2[0][b : b + 1, :, :t], self._mx2[1][b : b + 1, :, :t]
        return m.concatenate([K0, K1], axis=2), m.concatenate([V0, V1], axis=2)

    def mx_advance(self, T: int, Hk: int, d: int, dtype: Any) -> None:
        """T rows appended by the megakernel: the buffers exist and the length moves"""
        if not self.is_initialized:
            import torch

            empty = torch.empty(0, dtype=dtype)
            self.lazy_initialization(empty, empty)
        self._ensure(1, Hk, self._n + T, d, dtype)
        self._n += int(T)

    def crop(self, n: int) -> None:
        """keep the first n rows (n past the length keeps them all); a negative n drops the last -n, as
        transformers' `DynamicLayer.crop` takes it. A shared layer's crop is a new length (an int8 layer's is off
        the torch view); a torch layer's rows are cut, the card's arena front with them"""
        have = self.get_seq_length()
        n = max(0, have + int(n)) if n < 0 else min(int(n), have)
        if self.shared and self._mx is not None:
            self._tk = self._tv = None
            self._n = n
            return
        if n < have:
            k, v = self.keys, self.values
            self._set_rows(k[..., :n, :], v[..., :n, :])

    def gather(self, keep: Sequence[int], base: int = 0, lazy: bool = False) -> list[Any]:
        """The rows `keep` become the cache: gathered on the GPU and written back, an int8 layer's scales with
        them. The first `base` rows are left in place; only the rows past them move. An arena layer's write is
        one kernel whose flag `lazy` hands back for the caller to evaluate with the other layers'."""
        m = mlxdev.mx()
        keep = list(keep)
        n = len(keep)
        if base >= n or keep[base:] == list(range(base, n)):
            self.crop(n)
            return []
        if self.arena is not None:
            flags = [mlxdev.kv_gather(x, keep[base:], base) for x in self._mx[:2]]
            self._tk = self._tv = None
            self._n = n
            if lazy:
                return flags
            m.eval(*flags)
            return []
        idx = m.array(keep[base:], dtype=m.int32)
        bufs = [x for x in self._mx if x is not None]
        parts = [m.take(x[: self._b], idx, axis=2) for x in bufs]
        m.eval(*parts)
        for x, p in zip(bufs, parts):
            if x.ndim == 4:
                x[: self._b, :, base:n, :] = p
            else:
                x[: self._b, :, base:n] = p
        self._tk = self._tv = None
        self._n = n
        return []

    def _assign(self, which: int, t: torch.Tensor | None) -> None:
        # a prefix of the buffer is just a new length; anything else (a gathered tree path, rows from elsewhere)
        # waits as an override until the next append writes it into the buffer
        if t is None or t.numel() == 0:
            self._n = 0
            if which == 0:
                self._tk = None
            else:
                self._tv = None
            return
        if self._mx_prefix(which, t):
            self._n = int(t.shape[-2])
            if which == 0:
                self._tk = None
            else:
                self._tv = None
            return
        if t.untyped_storage().data_ptr() == self._mx_base(which):
            t = t.clone()
        if which == 0:
            self._tk = t
        else:
            self._tv = t

    def _mx_base(self, which: int) -> int:
        """the storage torch's view of a shared buffer lives in (an int8 layer's: its last dequantized copy)"""
        if self.bits:
            return self._tmp[which].untyped_storage().data_ptr() if self._tmp[which] is not None else -1
        return int(self._ptr[which])

    def _mx_prefix(self, which: int, t: torch.Tensor) -> bool:
        """`t` is the first rows of torch's view of a shared buffer, strides and all"""
        rows = self._tmp[which].shape[-2] if self.bits and self._tmp[which] is not None else self._shape[2]
        d = self._shape[-1]
        return bool(
            t.untyped_storage().data_ptr() == self._mx_base(which)
            and t.storage_offset() == 0
            and tuple(t.shape[:2]) == (self._b, self._shape[1])
            and int(t.shape[-1]) == d
            and tuple(t.stride()[1:]) == (int(rows) * d, d, 1)
        )

    # -- storage --
    def _grant_bound(self) -> int:
        """The capacity the scheduler should still call plausible: the ceiling these rows can reach plus the
        one growth step that lands over it. A growth reserves `have + max(4096, have // 8)`, so the last
        legitimate growth of a full-length context asks for more rows than the context has - checking against
        the bare ceiling would refuse the top of every long run. 0 where no ceiling is known (no check)."""
        return self.bound + max(4096, self.bound // 8) if self.bound else 0

    def _ensure(self, B: int, Hk: int, need: int, d: int, dtype: torch.dtype) -> None:
        m = mlxdev.mx()
        mdt = m.int8 if self.bits else mlxdev._dtypes()[dtype]
        old = self._mx
        same = old is not None and self._shape[1] == Hk and self._shape[-1] == d and old[0].dtype == mdt
        if same and self._shape[0] >= B and self._shape[2] >= need:
            # the buffer holds the largest batch seen; a smaller batch uses its first rows
            self._b = B
            return
        # rows grow by the usual step; a batch change alone keeps the row capacity (the drafter's tree steps
        # switch batch sizes several times a pass: growing the rows on each switch ran away to gigabytes)
        have = self._shape[2] if same else 0
        if same and have >= need:
            cap = have
        elif self.cap_hint:
            # the caller knows how far these rows run (prompt + max_new): reserved once, no regrowth
            cap = max(need, self.cap_hint)
        else:
            cap = max(need, have + max(4096, have // 8), 4096)
        Bc = max(B, self._shape[0] if same else 0)
        if self.arena is not None and (Bc != 1 or self.bits or need > self.arena[3]):
            # past the arena: a buffer of the layer's own, the rows copied below; the pass leaves the megakernel
            self.arena = None
            self._in_arena = False
            same = same and old is not None
        if self.arena is not None:
            buf, ok, ov, acap = self.arena
            if old is None or not getattr(self, "_in_arena", False):
                # the views of the arena's stretch; rows a buffer of the layer's own held move in
                nb = Hk * acap * d * 2
                kb = buf[ok : ok + nb].view(mdt).reshape(1, Hk, acap, d)
                vb = buf[ov : ov + nb].view(mdt).reshape(1, Hk, acap, d)
                m.eval(kb, vb)
                if old is not None and self._n:
                    m.eval(
                        mlxdev.kv_store(kb, old[0][0, :, : self._n].astype(mdt), 0),
                        mlxdev.kv_store(vb, old[1][0, :, : self._n].astype(mdt), 0),
                    )
                self._mx = [kb, vb]
                self._in_arena = True
                self._shape = (1, Hk, acap, d)
                self._ptr = (
                    mlxdev.from_mx(kb).untyped_storage().data_ptr(),
                    mlxdev.from_mx(vb).untyped_storage().data_ptr(),
                )
            self._b = B
            return
        if self.grant is not None:
            # k and v; an int8 row carries a float32 scale a head beside its d bytes
            el = 1 if self.bits else torch.empty(0, dtype=dtype).element_size()
            self.grant(
                2 * Bc * Hk * cap * (d * el + (4 if self.bits else 0)),
                "kv",
                requester=f"GrowLayer(layer cache) n={self._n} T={max(0, int(need) - self._n)} have={have}",
                B=Bc,
                cap=cap,
                bound=self._grant_bound() or None,
            )
        kb = m.zeros((Bc, Hk, cap, d), dtype=mdt)
        vb = m.zeros((Bc, Hk, cap, d), dtype=mdt)
        arrays = [kb, vb]
        if self.bits:
            arrays += [m.ones((Bc, Hk, cap), dtype=m.float32), m.ones((Bc, Hk, cap), dtype=m.float32)]
        n = self._n if same else 0
        if n:
            ob = self._shape[0]
            kb[:ob, :, :n, :] = old[0][..., :n, :].astype(mdt)
            vb[:ob, :, :n, :] = old[1][..., :n, :].astype(mdt)
            if self.bits:
                arrays[2][:ob, :, :n] = old[2][..., :n]
                arrays[3][:ob, :, :n] = old[3][..., :n]
        m.eval(*arrays)
        self._mx = arrays
        self._n = n
        self._b = B
        self._shape = (Bc, Hk, cap, d)
        self._ptr = (mlxdev.from_mx(kb).untyped_storage().data_ptr(), mlxdev.from_mx(vb).untyped_storage().data_ptr())

    def _apply_overrides(self) -> None:
        if self._tk is None and self._tv is None:
            return
        t = self._tk if self._tk is not None else self._tv
        n = int(t.shape[-2])
        if self.bits:
            # rows handed over in bf16 are quantized into the buffer (a gathered tree path's rows came out
            # of it, and a row's own scale takes them back to the same codes)
            for which, o in ((0, self._tk), (1, self._tv)):
                if o is not None:
                    self._store(which, 0, mlxdev.to_mx(o.contiguous()))
            self._n = n
            self._tk = self._tv = None
            return
        if self._tk is not None:
            self._view(0)[..., :n, :].copy_(self._tk)
        if self._tv is not None:
            self._view(1)[..., :n, :].copy_(self._tv)
        self._n = n
        self._tk = self._tv = None

    def mx_update(self, k: mx_.array, v: mx_.array) -> tuple[mx_.array, mx_.array]:
        """The MLX device's append: `k`/`v` MLX arrays [B, Hk, T, d]; returns the used prefixes as lazy views,
        the append itself in place at the next eval."""
        dt = mlxdev.torch_dtype(k.dtype)
        if not self.is_initialized:
            empty = torch.empty(0, dtype=dt)
            self.lazy_initialization(empty, empty)
        B, Hk, T, d = (int(x) for x in k.shape)
        n = self.get_seq_length()
        row = self._row
        if self._flat:
            self._ensure_flat(Hk, d, dt)
        else:
            self._ensure(self._rows_total if row is not None else B, Hk, n + T, d, dt)
        self._apply_overrides()
        if row is not None:
            # a batched cache with one row selected (its prefill): the append is that row's alone, at its
            # own length, and the prefix handed back is that row's
            self._store(0, n, k, row=row)
            self._store(1, n, v, row=row)
            self._ns[row] = self._n = n + T
            return self.mx_kv(0, n + T, row=row)
        if self.bits:
            # quantized on the way in; the used prefix comes back dequantized (a prefill's attention reads it;
            # the one-row step and the verify pass read the int8 rows through the node kernel instead)
            self._store(0, n, k)
            self._store(1, n, v)
            self._n = n + T
            return self.mx_kv(0, self._n)
        if self._mx[0].dtype != k.dtype:
            k, v = k.astype(self._mx[0].dtype), v.astype(self._mx[0].dtype)
        if self.arena is not None:
            # in place through the arena's kernel: a slice assignment would rebind the layer to a copy
            self._store(0, n, k)
            self._store(1, n, v)
            self._n = n + T
            return self.mx_kv(0, self._n)
        if self._shape[0] == B:
            self._mx[0][..., n : n + T, :] = k
            self._mx[1][..., n : n + T, :] = v
        else:
            self._mx[0][:B, :, n : n + T, :] = k
            self._mx[1][:B, :, n : n + T, :] = v
        self._n = n + T
        return self._mx[0][:B, :, : self._n, :], self._mx[1][:B, :, : self._n, :]

    def attach(self, kb: torch.Tensor, vb: torch.Tensor) -> None:
        """The layer's rows move into `(kb, vb)` - a [B, Hk, cap, d] pair the engine owns (the card's arena) -
        and every append grows there: the rows it has are copied in and its views re-cut on the new buffer."""
        n = int(self.keys.shape[-2]) if (self.is_initialized and self.keys is not None and self.keys.numel()) else 0
        if n and self.keys.data_ptr() != kb.data_ptr():
            kb[..., :n, :].copy_(self.keys)
            vb[..., :n, :].copy_(self.values)
        self._buf = (kb, vb)
        self._an = None
        if n:
            self._set_rows(kb[..., :n, :], vb[..., :n, :])

    def detach(self) -> None:
        """The layer gives its buffer up (another cache takes the arena): its rows become its own copy."""
        if self._buf is None:
            return
        if self.is_initialized and self.keys is not None and self.keys.numel() and self._attached():
            k, v = self.keys.clone(), self.values.clone()
            self._an = None
            self._keys_t, self._values_t = k, v
        self._an = None
        self._buf = None

    def set_front(self, n: int) -> None:
        """the rows are the buffer's first n (a graph step wrote them in place): no views, one integer"""
        self._an = int(n)

    def _attached(self) -> bool:
        """the rows are the front of the buffer itself (a view from its first row), not a copy elsewhere. The
        test is the data pointer, not the storage: the card's arena is one storage for every layer's buffer,
        so a buffer that is a slice of it starts at an offset of its own"""
        if self._an is not None:
            return True
        b = self._buf
        if b is None or not self.is_initialized or self.keys is None or not self.keys.numel():
            return False
        k = self.keys
        return bool(
            k.data_ptr() == b[0].data_ptr()
            and k.device == b[0].device
            and k.dtype == b[0].dtype
            and k.shape[:2] == b[0].shape[:2]
            and k.shape[-1] == b[0].shape[-1]
        )

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.shared and key_states.device.type == "cpu":
            B, Hk, T, d = key_states.shape
            n = self.get_seq_length()
            self._ensure(B, Hk, n + T, d, key_states.dtype)
            self._apply_overrides()
            if self.bits:
                self._store(0, n, mlxdev.to_mx(key_states.contiguous()))
                self._store(1, n, mlxdev.to_mx(value_states.contiguous()))
                self._n = n + T
                return self.keys, self.values
            kb, vb = self._view(0), self._view(1)
            kb[..., n : n + T, :].copy_(key_states)
            vb[..., n : n + T, :].copy_(value_states)
            self._n = n + T
            return kb[..., : self._n, :], vb[..., : self._n, :]
        n = self.keys.shape[-2] if self.keys.numel() else 0
        B, Hk, T, d = key_states.shape
        fits = (
            self._buf is not None
            and self._buf[0].shape[-2] >= n + T
            and tuple(self._buf[0].shape[:2]) == (B, Hk)
            and self._buf[0].shape[-1] == d
            and self._buf[0].dtype == key_states.dtype
            and self._buf[0].device == key_states.device
        )
        if fits and not self._attached() and n and self.keys.data_ptr() != self._buf[0].data_ptr():
            # the rows were detached from the buffer (an accepted tree path gathered them, or they came back
            # from the card): copy them into the buffer that already holds room for them instead of growing it
            self._buf[0][..., :n, :].copy_(self.keys)
            self._buf[1][..., :n, :].copy_(self.values)
            self._set_rows(self._buf[0][..., :n, :], self._buf[1][..., :n, :])
        if not fits:
            have = self._buf[0].shape[-2] if self._buf is not None else 0
            if self.cap_hint:
                # the caller knows how far this sequence runs (prompt + max_new): reserve exactly that once,
                # so a short decode does not take the 4096-position floor times the batch and OOM
                cap = max(n + T, self.cap_hint)
            elif have >= n + T:
                # only the placement changed - the batch, the dtype or the device (a layer shed to the host and
                # regrown, its rows cast on each move): the rows keep their capacity and the buffer is re-cut
                # at the same size where they now live. Growing an eighth on every move compounded a 0.6B
                # model's cache to gigabytes over one answer
                cap = have
            else:
                cap = max(n + T, have + max(4096, have // 8), 4096)
                if key_states.device.type == "cpu" and self.reserve:
                    cap = max(cap, self.reserve + 1024)
            if self.grant is not None:
                self.grant(
                    2 * B * Hk * cap * d * key_states.element_size(),
                    "kv",
                    requester=f"GrowLayer(layer cache) n={n} T={T} have={have}",
                    B=B,
                    cap=cap,
                    bound=self._grant_bound() or None,
                    device=key_states.device,
                )
            kb = torch.empty(B, Hk, cap, d, dtype=key_states.dtype, device=key_states.device)
            vb = torch.empty_like(kb)
            if n:
                kb[..., :n, :].copy_(self.keys)
                vb[..., :n, :].copy_(self.values)
            self._buf = (kb, vb)
        kb, vb = self._buf
        kb[..., n : n + T, :].copy_(key_states)
        vb[..., n : n + T, :].copy_(value_states)
        self._set_rows(kb[..., : n + T, :], vb[..., : n + T, :])
        return self.keys, self.values


def set_rows(layer: CacheLayerMixin, k: torch.Tensor, v: torch.Tensor) -> None:
    """btb's own write of a layer's rows (see `GrowLayer._set_rows`); any other layer takes them as assigned"""
    if isinstance(layer, GrowLayer):
        layer._set_rows(k, v)
    else:
        layer.keys, layer.values = k, v


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.device == b.device and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _inside(t: torch.Tensor, b: torch.Tensor) -> bool:
    """every element of `t` lies within the span of memory `b` covers"""
    if t.dtype != b.dtype or not _same_storage(t, b):
        return False

    def span(x: torch.Tensor) -> tuple[int, int]:
        last = sum((n - 1) * s for n, s in zip(x.shape, x.stride()))
        return x.data_ptr(), x.data_ptr() + (last + 1) * x.element_size()

    (lo, hi), (blo, bhi) = span(t), span(b)
    return blo <= lo and hi <= bhi


def forked(cache: KvCache) -> bool:
    """a fork's or a batch's cache: the single-row paths (the fused MLX step, the card graph) cannot read its
    layers"""
    return cache in _FORKS


class ForkLayer(_DynamicLayer):
    """One attention layer of B rows going on from given rows: a prefix `[1, Hk, P, d]` every row shares (a fork's,
    another cache's rows, never written) or `[B, Hk, P, d]` one a row (a batch's, left-padded), then each row's
    own rows in a buffer of their own. A pass reads the two joined, built once a step."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor, B: int) -> None:
        super().__init__()
        self._pk, self._pv = k, v
        self._B = int(B)
        self._tk: torch.Tensor | None = None
        self._tv: torch.Tensor | None = None
        self._t = 0
        self._cat: tuple[torch.Tensor, torch.Tensor] | None = None
        self.dtype, self.device = k.dtype, k.device
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.is_initialized = True

    def _joined(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cat is None:
            pk = self._pk.expand(self._B, -1, -1, -1)
            pv = self._pv.expand(self._B, -1, -1, -1)
            if self._t and self._tk is not None and self._tv is not None:
                pk = torch.cat([pk, self._tk[..., : self._t, :]], dim=-2)
                pv = torch.cat([pv, self._tv[..., : self._t, :]], dim=-2)
            self._cat = (pk, pv)
        return self._cat

    @property
    def keys(self) -> torch.Tensor:
        return self._joined()[0]

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise TypeError("a fork's rows are cut by its Branches, not through the cache")

    @property
    def values(self) -> torch.Tensor:
        return self._joined()[1]

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise TypeError("a fork's rows are cut by its Branches, not through the cache")

    def get_seq_length(self) -> int:
        return int(self._pk.shape[-2]) + self._t

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Hk, T, d = key_states.shape
        need = self._t + T
        if self._tk is None or self._tv is None or self._tk.shape[-2] < need:
            cap = max(need, 2 * (self._tk.shape[-2] if self._tk is not None else 0), 64)
            tk = key_states.new_empty(B, Hk, cap, d)
            tv = value_states.new_empty(B, Hk, cap, d)
            if self._t and self._tk is not None and self._tv is not None:
                tk[..., : self._t, :].copy_(self._tk[..., : self._t, :])
                tv[..., : self._t, :].copy_(self._tv[..., : self._t, :])
            self._tk, self._tv = tk, tv
        self._tk[..., self._t : need, :].copy_(key_states)
        self._tv[..., self._t : need, :].copy_(value_states)
        self._t = need
        self._cat = None
        return self._joined()

    def select(self, idx: torch.Tensor) -> None:
        """the rows `idx` become the batch, in that order"""
        if self._pk.shape[0] > 1:
            self._pk = self._pk.index_select(0, idx.to(self._pk.device))
            self._pv = self._pv.index_select(0, idx.to(self._pv.device))
        if self._tk is not None and self._tv is not None:
            self._tk = self._tk.index_select(0, idx.to(self._tk.device))
            self._tv = self._tv.index_select(0, idx.to(self._tv.device))
        self._B = int(idx.numel())
        self._cat = None

    def row(self, b: int) -> tuple[torch.Tensor, ...] | None:
        """row b's own rows [1, Hk, t, d], copied out"""
        if not self._t or self._tk is None or self._tv is None:
            return None
        return self._tk[b : b + 1, :, : self._t].clone(), self._tv[b : b + 1, :, : self._t].clone()

    def to(self, dev: str | torch.device) -> None:
        """every row to `dev`, where the layer now runs: the prefix becomes a copy of its own there"""
        self._pk, self._pv = self._pk.to(dev), self._pv.to(dev)
        if self._tk is not None and self._tv is not None:
            self._tk, self._tv = self._tk.to(dev), self._tv.to(dev)
        self.device = torch.device(dev)
        self._cat = None


class ForkIndexedLayer(ForkLayer):
    """a `ForkLayer` of a sparse-attention layer, which caches its indexer's keys `[B, n, d_index]` beside K and V:
    the prefix's shared (or one a row) and each row's own after them, as the attention's rows are"""

    def __init__(self, k: torch.Tensor, v: torch.Tensor, ik: torch.Tensor, B: int) -> None:
        super().__init__(k, v, B)
        self._pi = ik
        self._ti: torch.Tensor | None = None
        self.is_indexer_initialized = True

    @property
    def indexer_keys(self) -> torch.Tensor:
        pi = self._pi.expand(self._B, -1, -1)
        return pi if self._ti is None else torch.cat([pi, self._ti], dim=1)

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        t = indexer_key_states
        self._ti = t.clone() if self._ti is None else torch.cat([self._ti, t], dim=1)
        return self.indexer_keys

    def select(self, idx: torch.Tensor) -> None:
        if self._pi.shape[0] > 1:
            self._pi = self._pi.index_select(0, idx.to(self._pi.device))
        if self._ti is not None:
            self._ti = self._ti.index_select(0, idx.to(self._ti.device))
        super().select(idx)

    def row(self, b: int) -> tuple[torch.Tensor, ...] | None:
        kv = super().row(b)
        if kv is None or self._ti is None:
            return kv
        return (*kv, self._ti[b : b + 1].clone())

    def to(self, dev: str | torch.device) -> None:
        super().to(dev)
        self._pi = self._pi.to(dev)
        if self._ti is not None:
            self._ti = self._ti.to(dev)
