# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The attention cache layer grown in place, in torch's memory or in MLX's (the shared buffer the GPU appends to),
and a fork's layer over another cache's rows."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from .. import mlx as mlxdev
from .device import DeviceSpec, Where, where

if TYPE_CHECKING:
    import mlx.core as mx_
    from transformers.cache_utils import CacheLayerMixin, DynamicCache, LinearAttentionCacheLayerMixin
    from transformers.cache_utils import DynamicIndexedLayer as _DynamicIndexedLayer
    from transformers.cache_utils import DynamicLayer as _DynamicLayer

    # a sequence's cache: transformers' DynamicCache over the engine's layers (GrowLayer, a fork's, a hybrid's)
    KvCache = DynamicCache
    # one layer of it: an attention layer's rows, or a linear-attention layer's recurrent states
    CacheLayer = CacheLayerMixin | LinearAttentionCacheLayerMixin
else:
    # transformers is imported by name here and not at the top: the engine package is imported for its
    # discovery and planning too, where transformers' import time is not wanted
    _DynamicLayer = __import__("transformers").cache_utils.DynamicLayer
    _DynamicIndexedLayer = __import__("transformers").cache_utils.DynamicIndexedLayer


def linear_layer(cl: CacheLayer) -> LinearAttentionCacheLayerMixin:
    """`cl` as the linear-attention layer a hybrid's linear index holds; a TypeError for any other"""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    if not isinstance(cl, LinearAttentionCacheLayerMixin):
        raise TypeError(f"a linear-attention layer's states were asked of a {type(cl).__name__}")
    return cl


def conv_states_as(cl: Any, dtype: torch.dtype) -> None:
    """a linear-attention layer's conv states in `dtype`, the one its layer computes in where it runs now. They
    are kept in the dtype of the pass that made them, and a layer that moves between the host (float32) and the
    card (bf16) - a host layer's prefill chunks, some on the card and some not; a layer shed and grown back - would
    join its state onto the new rows promoted, and the card's bf16 conv refuses a float32 input. Nothing for any
    other layer, or a state already so."""
    states = getattr(cl, "conv_states", None)
    if not isinstance(states, dict):
        return
    for k, v in states.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != dtype:
            states[k] = v.to(dtype)


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
        shared: bool = False,
        bits: int | None = None,
        cap_hint: int = 0,
        grant: Callable[..., None] | None = None,
        bound: int = 0,
        arena: tuple[Any, int, int, int] | None = None,
        kv_dtype: torch.dtype | None = None,
    ) -> None:
        # `grant` is the scheduler's gate (BatchScheduler.grant), asked on the growth branch alone - never per
        # token - so a buffer too large for the card, or one growing past `bound` (the length these rows can
        # reach), is refused here with a diagnostic instead of OOM-ing inside torch's allocator
        self.grant = grant
        # the dtype the rows are kept in wherever they are made (`update` casts what comes in): a card engine's
        # model dtype, so a host layer's rows - the prompt's made on the card, the answer's on the host - are held
        # as every card layer's are, not widened to the host's float32 at twice the bytes. None: as they come
        self.kv_dtype = kv_dtype
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

    def _ceiling(self, cap: int, need: int) -> int:
        """`cap` rows held to the most these rows can reach where the caller named it (`cap_hint`: the prompt and
        its answer's cap) - a ceiling, never a reservation. Reserved up front, an answer left uncapped (`btb run` with
        no --new: the window's 262144 positions on Qwen3-4B) asked 36 GB of cache for a 22-token prompt"""
        return min(cap, max(need, self.cap_hint)) if self.cap_hint else cap

    def _step(self, need: int, have: int) -> int:
        """the rows a growth past `have` makes to hold `need`: an eighth more, at least 4096, held to the ceiling - so
        a short decode's buffer is its own few rows, not the 4096-position floor times the batch"""
        return self._ceiling(max(need, have + max(4096, have // 8), 4096), need)

    def _mx_cap(self, need: int, have: int) -> int:
        """the rows a new shared buffer holds for `need` rows, `have` the rows of the one it replaces"""
        if have >= need:
            return have
        return self._step(need, have)

    def _torch_cap(self, need: int, have: int) -> int:
        """the rows a new torch buffer holds for `need` rows, `have` the rows of the one it replaces"""
        if have >= need:
            # only the placement changed - the batch, the dtype or the device (a layer shed to the host and
            # regrown, its rows cast on each move): the rows keep their capacity and the buffer is re-cut at the
            # same size where they now live. Growing an eighth on every move compounded a 0.6B model's cache to
            # gigabytes over one answer
            return have
        return self._step(need, have)

    def growth(self, B: int, T: int, Hk: int, d: int, dtype: torch.dtype) -> int:
        """The bytes the next append of T rows to B sequences allocates: a new buffer where they do not fit the
        layer's own, 0 where they do. A fork's or a batch's layer grows its step buffer by its own and reads 0."""
        if self._ns is not None or self._flat:
            return 0
        need = self.get_seq_length() + T if self.is_initialized else T
        if self.shared:
            same = self._mx is not None and self._shape[1] == Hk and self._shape[-1] == d
            have = int(self._shape[2]) if same else 0
            if same and self._shape[0] >= B and have >= need:
                return 0
            if self.arena is not None and B == 1 and not self.bits and need <= self.arena[3]:
                return 0
            Bc = max(B, int(self._shape[0]) if same else 0)
            row = d * (1 if self.bits else dtype.itemsize) + (4 if self.bits else 0)
            return 2 * Bc * Hk * self._mx_cap(need, have) * row
        buf = self._buf
        if buf is not None and tuple(buf[0].shape[:2]) == (B, Hk) and buf[0].shape[-1] == d:
            if buf[0].shape[-2] >= need:
                return 0
        have = int(buf[0].shape[-2]) if buf is not None else 0
        return 2 * B * Hk * self._torch_cap(need, have) * d * dtype.itemsize

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
        cap = self._mx_cap(need, have)
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
            row = d * el + (4 if self.bits else 0)
            self.grant(
                2 * Bc * Hk * cap * row,
                "kv",
                requester=f"GrowLayer(layer cache) n={self._n} T={max(0, int(need) - self._n)} have={have}",
                B=Bc,
                cap=cap,
                bound=self._grant_bound() or None,
                # the buffer this one replaces, let go once its rows are copied over
                held=2 * int(self._shape[0]) * Hk * have * row if same else 0,
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

    def presize(self, rows: int, B: int, Hk: int, d: int, dtype: torch.dtype, dev: Where) -> None:
        """The layer's buffer `rows` long at once, the rows it holds copied in: a prefill sweep's whole prompt, so
        the cache does not grow while the sweep's working set is live around it - a buffer grown then would sit
        inside the one block the pass's own buffers are cut from, and split it for good. Granted as a growth is,
        the buffer it replaces counted; nothing where the buffer holds that many rows already, and nothing for rows
        of another shape (left to their own growth). With a growth step's room past the prompt, held to the
        sequence's reach where the caller named it (`_step`): sized to the prompt alone, the answer's first token grew
        every layer again - a second whole buffer each, copied, shedding layers to make the room."""
        if self.shared:
            return
        rows = self._step(int(rows), int(rows))
        b = self._buf
        if (
            b is not None
            and b[0].shape[-2] >= rows
            and b[0].dtype == dtype
            and b[0].device == dev
            and tuple(b[0].shape[:2]) == (B, Hk)
        ):
            return
        k = self.keys if self.is_initialized else None
        n = int(k.shape[-2]) if (k is not None and k.numel()) else 0
        if k is not None and n and (tuple(k.shape[:2]) != (B, Hk) or int(k.shape[-1]) != d):
            return
        el = torch.empty(0, dtype=dtype).element_size()
        if self.grant is not None:
            self.grant(
                2 * B * Hk * int(rows) * d * el,
                "kv",
                requester=f"GrowLayer(the prompt's rows at once) n={n} rows={rows}",
                B=B,
                cap=int(rows),
                bound=self._grant_bound() or None,
                device=dev,
                held=2 * b[0].numel() * b[0].element_size() if b is not None else 0,
            )
        kb = torch.empty(B, Hk, int(rows), d, dtype=dtype, device=dev)
        vb = torch.empty_like(kb)
        if n:
            kb[..., :n, :].copy_(self.keys)
            vb[..., :n, :].copy_(self.values)
        self._buf, self._an = (kb, vb), None
        if n:
            self._set_rows(kb[..., :n, :], vb[..., :n, :])

    def hop(self, kb: torch.Tensor, vb: torch.Tensor) -> None:
        """The rows into `(kb, vb)` - [B, Hk, cap, d] buffers the caller owns on another device, room for every row
        the caller's passes bring - and the buffer they were in let go: a host layer's prefill chunks on the card
        then write their rows in place there, the cache neither grown nor granted (the prefill's reservation holds
        the buffers, `_hop_bytes`). The rows are cast as a growth into the buffers' dtype would cast them; a layer
        with no rows yet (the prompt's first chunk) starts in them. Moved elsewhere (`_cache_to`), the rows become
        a copy of their own and the buffers are the caller's again"""
        k = self.keys if self.is_initialized else None
        n = int(k.shape[-2]) if k is not None and k.numel() else 0
        self._buf, self._an = (kb, vb), None
        if k is not None and n:
            kb[..., :n, :].copy_(k)
            vb[..., :n, :].copy_(self.values)
            self._set_rows(kb[..., :n, :], vb[..., :n, :])

    def land(self, dev: Where, dtype: torch.dtype) -> None:
        """The rows back from a hop onto `dev` in a buffer of `dtype` with a growth step's room past them (`_step`,
        held to the sequence's reach where it is named), granted as that growth is: the layer's next append writes in
        place. Copied back as they were (a tight copy in the card's dtype), the answer's first token on
        the host grew every host layer again in float32 at once, beside the rows it replaced - 2.6 GB asked in one
        step of a 40k prompt's answer, refused"""
        k, v = self.keys, self.values
        B, Hk, n, d = (int(x) for x in k.shape)
        cap = self._step(n, n)
        el = torch.empty(0, dtype=dtype).element_size()
        if self.grant is not None:
            self.grant(
                2 * B * Hk * cap * d * el,
                "kv",
                requester=f"GrowLayer(the rows back from the card) n={n} cap={cap}",
                B=B,
                cap=cap,
                bound=self._grant_bound() or None,
                device=dev,
            )
        kb = torch.empty(B, Hk, cap, d, dtype=dtype, device=dev)
        vb = torch.empty_like(kb)
        kb[..., :n, :].copy_(k)
        vb[..., :n, :].copy_(v)
        self._buf, self._an = (kb, vb), None
        self._set_rows(kb[..., :n, :], vb[..., :n, :])

    def detach_bytes(self) -> int:
        """what `detach` copies"""
        if self._buf is None or not (self.is_initialized and self.keys is not None and self._attached()):
            return 0
        return 2 * self.keys.numel() * self.keys.element_size()

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
        kd = self.kv_dtype
        if kd is not None and not self.shared and key_states.dtype != kd:
            key_states, value_states = key_states.to(kd), value_states.to(kd)
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
            cap = self._torch_cap(n + T, have)
            if self.grant is not None:
                self.grant(
                    2 * B * Hk * cap * d * key_states.element_size(),
                    "kv",
                    requester=f"GrowLayer(layer cache) n={n} T={T} have={have}",
                    B=B,
                    cap=cap,
                    bound=self._grant_bound() or None,
                    device=key_states.device,
                    # the buffer this one replaces, let go once its rows are copied over
                    held=2 * self._buf[0].numel() * self._buf[0].element_size() if self._buf is not None else 0,
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
    own rows. The first step joins the two in one buffer `[B, Hk, P + cap, d]`, and every step after writes its
    rows in place: a pass reads a slice of it, copying nothing. The buffer is granted as the main cache's growth
    is (`grant`, the scheduler's gate), counting the one it replaces."""

    def __init__(self, k: torch.Tensor, v: torch.Tensor, B: int, grant: Callable[..., None] | None = None) -> None:
        super().__init__()
        # the prefix, until the first step joins it into the buffer (a batch's own padded copy is then let go)
        self._pk: torch.Tensor | None = k
        self._pv: torch.Tensor | None = v
        self._P = int(k.shape[-2])
        self._B = int(B)
        self._kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self._t = 0
        self.grant = grant
        self.dtype = k.dtype
        self.device: Where = where(k.device)
        self.is_initialized = True

    @classmethod
    def over(
        cls, kb: torch.Tensor, vb: torch.Tensor, P: int, t: int, grant: Callable[..., None] | None = None
    ) -> ForkLayer:
        """a fork's layer over a buffer `[B, Hk, cap, d]` of its own, already granted and holding its rows: `P`
        of prefix, then `t` of each row's"""
        fl = cls(kb[:1, :, :0], vb[:1, :, :0], int(kb.shape[0]), grant)
        fl._pk = fl._pv = None
        fl._P, fl._t, fl._kv = int(P), int(t), (kb, vb)
        return fl

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.is_initialized = True

    def _ask(
        self, nbytes: int, held: int, what: str, dev: torch.device, B: int, cap: int, draws: str | None = None
    ) -> None:
        """the scheduler's grant for a buffer of the layer's: its growth draws on the epoch's KV, a copy or a move of
        rows it holds already (`draws=""`) on nothing"""
        if self.grant is not None:
            self.grant(
                nbytes,
                "kv",
                requester=f"{type(self).__name__}({what})",
                B=B,
                cap=cap,
                device=dev,
                held=held,
                draws=draws,
            )

    def _prefix(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._pk is not None and self._pv is not None  # held until the buffer takes them
        return self._pk, self._pv

    def _joined(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._kv is None:
            pk, pv = self._prefix()
            return pk.expand(self._B, -1, -1, -1), pv.expand(self._B, -1, -1, -1)
        n = self._P + self._t
        return self._kv[0][..., :n, :], self._kv[1][..., :n, :]

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
        return self._P + self._t

    def _grow(self, need: int, like: torch.Tensor) -> None:
        """a buffer of `need` rows or more, the rows so far copied in: the prefix, on the first step"""
        B, Hk, _, d = like.shape
        old = self._kv
        have = int(old[0].shape[-2]) - self._P if old is not None else 0
        # a cast alone keeps the room it had; a growth doubles it
        cap = self._P + (have if have >= need - self._P else max(need - self._P, 2 * have, 64))
        held = 2 * old[0].numel() * old[0].element_size() if old is not None else 0
        self._ask(
            2 * B * Hk * cap * d * like.element_size(), held, f"B={B} P={self._P} t={self._t}", like.device, B, cap
        )
        kb = like.new_empty(B, Hk, cap, d)
        vb = like.new_empty(B, Hk, cap, d)
        if old is None:
            pk, pv = self._prefix()
            kb[..., : self._P, :].copy_(pk.expand(B, -1, -1, -1))
            vb[..., : self._P, :].copy_(pv.expand(B, -1, -1, -1))
            self._pk = self._pv = None
        else:
            n = self._P + self._t
            kb[..., :n, :].copy_(old[0][..., :n, :])
            vb[..., :n, :].copy_(old[1][..., :n, :])
        self._kv = (kb, vb)

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = int(key_states.shape[-2])
        if key_states.device != self.device:
            # the layer runs elsewhere now (given up to the host, or grown back onto the card): its rows follow it
            self.to(key_states.device)
        need = self._P + self._t + T
        if self._kv is None or self._kv[0].shape[-2] < need or self._kv[0].dtype != key_states.dtype:
            # grown, or cast: the layer computes in another dtype where it runs now (float32 on the host, the card's
            # bf16), and its rows are made that dtype as a growth into it makes them, as `GrowLayer`'s are
            self._grow(need, key_states)
        assert self._kv is not None
        self._kv[0][..., need - T : need, :].copy_(key_states)
        self._kv[1][..., need - T : need, :].copy_(value_states)
        self._t += T
        return self._joined()

    def _pick(self, x: torch.Tensor, idx: torch.Tensor, what: str) -> torch.Tensor:
        """`x`'s rows at `idx`, a buffer of their own granted before it is made (the old one let go after)"""
        B = int(idx.numel())
        per = x[0].numel() * x.element_size()
        self._ask(B * per, x.numel() * x.element_size(), what, x.device, B, int(x.shape[-2]), draws="")
        return x.index_select(0, idx.to(x.device))

    def select(self, idx: torch.Tensor) -> None:
        """the rows `idx` become the batch, in that order"""
        if self._kv is not None:
            self._kv = (self._pick(self._kv[0], idx, "select"), self._pick(self._kv[1], idx, "select"))
        else:
            pk, pv = self._prefix()
            if pk.shape[0] > 1:
                self._pk, self._pv = self._pick(pk, idx, "select prefix"), self._pick(pv, idx, "select prefix")
        self._B = int(idx.numel())

    def row(self, b: int) -> tuple[torch.Tensor, ...] | None:
        """row b's own rows [1, Hk, t, d], a view the caller copies into its own cache before the next step"""
        if not self._t or self._kv is None:
            return None
        s = slice(self._P, self._P + self._t)
        return self._kv[0][b : b + 1, :, s], self._kv[1][b : b + 1, :, s]

    def to(self, device: DeviceSpec) -> None:
        """every row to `device`, where the layer now runs: the prefix becomes a copy of its own there"""
        dev = where(device)
        if self._kv is not None:
            k, v = self._kv
            self._ask(2 * k.numel() * k.element_size(), 0, f"to {dev}", dev, int(k.shape[0]), int(k.shape[-2]), "")
            self._kv = (k.to(dev), v.to(dev))
        else:
            pk, pv = self._prefix()
            self._ask(2 * pk.numel() * pk.element_size(), 0, f"prefix to {dev}", dev, int(pk.shape[0]), self._P, "")
            self._pk, self._pv = pk.to(dev), pv.to(dev)
        self.device = dev


class ForkIndexedLayer(ForkLayer):
    """a `ForkLayer` of a sparse-attention layer, which caches its indexer's keys `[B, n, d_index]` beside K and V:
    the prefix's shared (or one a row) and each row's own after them in one buffer, as the attention's rows are"""

    def __init__(
        self, k: torch.Tensor, v: torch.Tensor, ik: torch.Tensor, B: int, grant: Callable[..., None] | None = None
    ) -> None:
        super().__init__(k, v, B, grant)
        self._pi: torch.Tensor | None = ik
        self._Pi = int(ik.shape[1])
        self._ti: torch.Tensor | None = None
        # the indexer's own step count: its keys are written apart from K and V, in the attention's own pass
        self._it = 0
        self.is_indexer_initialized = True

    @property
    def indexer_keys(self) -> torch.Tensor:
        if self._ti is None:
            assert self._pi is not None
            return self._pi.expand(self._B, -1, -1)
        return self._ti[:, : self._Pi + self._it]

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        t = indexer_key_states
        if t.device != self.device:
            self.to(t.device)
        B, T, d = t.shape
        need = self._Pi + self._it + T
        if self._ti is None or self._ti.shape[1] < need or self._ti.dtype != t.dtype:
            old = self._ti
            have = int(old.shape[1]) - self._Pi if old is not None else 0
            cap = self._Pi + (have if have >= need - self._Pi else max(need - self._Pi, 2 * have, 64))
            held = old.numel() * old.element_size() if old is not None else 0
            self._ask(B * cap * d * t.element_size(), held, f"indexer B={B} P={self._Pi}", t.device, B, cap)
            buf = t.new_empty(B, cap, d)
            if old is None:
                assert self._pi is not None
                buf[:, : self._Pi].copy_(self._pi.expand(B, -1, -1))
                self._pi = None
            else:
                n = self._Pi + self._it
                buf[:, :n].copy_(old[:, :n])
            self._ti = buf
        self._ti[:, need - T : need].copy_(t)
        self._it += T
        return self.indexer_keys

    def select(self, idx: torch.Tensor) -> None:
        if self._ti is not None:
            self._ti = self._pick(self._ti, idx, "select indexer")
        elif self._pi is not None and self._pi.shape[0] > 1:
            self._pi = self._pick(self._pi, idx, "select indexer prefix")
        super().select(idx)

    def row(self, b: int) -> tuple[torch.Tensor, ...] | None:
        kv = super().row(b)
        if kv is None or self._ti is None or not self._it:
            return kv
        return (*kv, self._ti[b : b + 1, self._Pi : self._Pi + self._it])

    def to(self, device: DeviceSpec) -> None:
        super().to(device)
        x = self._ti if self._ti is not None else self._pi
        assert x is not None
        self._ask(x.numel() * x.element_size(), 0, f"indexer to {self.device}", self.device, int(x.shape[0]), 0, "")
        if self._ti is not None:
            self._ti = self._ti.to(self.device)
        else:
            self._pi = x.to(self.device)


class CardRowsLayer(_DynamicLayer):
    """One attention layer of B rows in the card's arena, stepped by the card graph's rows pass: row b's prefix is
    `lens[b]` rows from slot `offs[b]` (a fork's rows share their session's, a batch's lie end to end), then its
    own steps - step i of every row in the stretch of `W` slots from `base + i * W`, row b at column `cols[b]`.
    The kernels read each row's keys where they lie, so a step joins and copies nothing; torch reads a row out
    (`row`), or the lot as a fork's layer (`to_fork`) when the rows leave the card."""

    def __init__(
        self,
        kb: torch.Tensor,
        vb: torch.Tensor,
        offs: Sequence[int],
        lens: Sequence[int],
        base: int,
        W: int,
        grant: Callable[..., None] | None = None,
    ) -> None:
        super().__init__()
        # the scheduler's gate, asked before the rows are copied out of the arena (`to_fork`)
        self.grant = grant
        # this layer's [Hk, cap, d] slices of the arena; None while another cache holds it, the rows then in `_own`
        self._buf: tuple[torch.Tensor, torch.Tensor] | None = (kb, vb)
        self._own: tuple[torch.Tensor, torch.Tensor] | None = None
        self.offs, self.lens = [int(x) for x in offs], [int(x) for x in lens]
        self.cols = list(range(len(self.offs)))
        self.base, self.W = int(base), int(W)
        self._t = 0
        self.dtype, self.device = kb.dtype, kb.device
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.is_initialized = True

    @property
    def used(self) -> int:
        """the arena's slots the rows hold, [0, used)"""
        return self.base + self._t * self.W

    def _rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[Hk, >= used, d]: the arena's slices, or the copy made when another cache took the arena"""
        src = self._buf if self._buf is not None else self._own
        assert src is not None  # a layer holds its rows in one place or the other
        return src

    def _steps(self, cols: Sequence[int], W: int | None = None) -> torch.Tensor:
        """the slots of the rows at `cols`, [len(cols), t]: step i of the row at column c is base + i * W + c"""
        W = self.W if W is None else W
        steps = torch.arange(self._t, device=self.device) * W + self.base
        return steps[None, :] + torch.tensor(list(cols), device=self.device)[:, None]

    def get_seq_length(self) -> int:
        return max(self.lens) + self._t

    # the rows as a fork's layer holds them, a copy (the kernels read the arena; this is torch's view)
    @property
    def keys(self) -> torch.Tensor:
        return self._joined(0)

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise TypeError("a fork's rows are cut by its Branches, not through the cache")

    @property
    def values(self) -> torch.Tensor:
        return self._joined(1)

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        if t is not None:
            raise TypeError("a fork's rows are cut by its Branches, not through the cache")

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object) -> Any:
        raise TypeError("a card rows layer is written by the card graph's rows pass alone")

    def _prefix(self, which: int) -> torch.Tensor:
        """every row's prefix, [1, Hk, P, d] when the rows share one, else left-padded [B, Hk, max len, d]"""
        x = self._rows()[which]
        if len(set(zip(self.offs, self.lens))) == 1:
            return x[:, self.offs[0] : self.offs[0] + self.lens[0]][None].clone()
        P = max(self.lens)
        out = x.new_zeros(len(self.offs), x.shape[0], P, x.shape[-1])
        for b, (off, n) in enumerate(zip(self.offs, self.lens)):
            out[b, :, P - n :] = x[:, off : off + n]
        return out

    def _tails(self, which: int, cols: Sequence[int]) -> torch.Tensor:
        """the steps of the rows at `cols`, [len(cols), Hk, t, d]"""
        x = self._rows()[which]
        return x[:, self._steps(cols)].permute(1, 0, 2, 3).contiguous()

    def _own_steps(self, which: int, c: int) -> torch.Tensor:
        """the steps of the row at column `c`, [Hk, t, d]: a view of every W-th slot from its first, no copy"""
        x = self._rows()[which]
        return x[:, self.base + c : self.base + c + self._t * self.W : self.W]

    def _joined(self, which: int) -> torch.Tensor:
        p = self._prefix(which).expand(len(self.cols), -1, -1, -1)
        return torch.cat([p, self._tails(which, self.cols)], dim=-2) if self._t else p.clone()

    def row(self, b: int) -> tuple[torch.Tensor, ...] | None:
        """row b's own rows [1, Hk, t, d], a view the caller copies into its own cache before the next step"""
        if not self._t:
            return None
        c = self.cols[b]
        return self._own_steps(0, c)[None], self._own_steps(1, c)[None]

    def to_fork(self, dev: str | torch.device | None = None) -> ForkLayer:
        """the rows as a fork's layer holds them (each row's prefix, a batch's left-padded, then its own steps) in
        one buffer of its own on `dev`, granted before it is made and filled from the arena directly, for the
        torch pass once the card cannot take them; a batch's pass masks the padding"""
        dev = self.device if dev is None else torch.device(dev)
        k, v = self._rows()
        B, P, t = len(self.cols), max(self.lens), self._t
        Hk, d = int(k.shape[0]), int(k.shape[-1])
        cap = P + max(t, 64)
        if self.grant is not None:
            self.grant(
                2 * B * Hk * cap * d * k.element_size(),
                "kv",
                requester=f"CardRowsLayer(to a fork's layer) B={B} P={P} t={t}",
                B=B,
                cap=cap,
                device=dev,
                draws="",  # the rows the arena holds, copied out: the epoch counted them once already
            )
        bufs = []
        for which, x in enumerate((k, v)):
            buf = x.new_empty((B, Hk, cap, d), device=dev)
            for b, (off, n) in enumerate(zip(self.offs, self.lens)):
                if n < P:
                    buf[b, :, : P - n].zero_()
                buf[b, :, P - n : P].copy_(x[:, off : off + n])
                if t:
                    buf[b, :, P : P + t].copy_(self._own_steps(which, self.cols[b]))
            bufs.append(buf)
        return ForkLayer.over(bufs[0], bufs[1], P, t, self.grant)

    def detach_bytes(self) -> int:
        """what `detach` copies"""
        if self._buf is None:
            return 0
        return 2 * self._buf[0][:, : self.used].numel() * self._buf[0].element_size()

    def select(self, slots: Sequence[int]) -> None:
        """the rows at `slots` become the rows, in that order (a row may repeat: a beam's survivor kept twice). A
        row keeps its column; a repeat takes a free one, its steps copied there - or, with too few columns free,
        every row's steps move to a wider stretch, which the arena must have room for (`used_after`)."""
        slots = [int(s) for s in slots]
        cols, taken = self._plan(slots)
        if cols is None:
            # a stretch as wide as the rows: row j at column j, every step moved (the gather reads before it writes)
            W = len(slots)
            src = self._steps([self.cols[s] for s in slots])
            dst = self._steps(range(W), W)
            for x in self._rows():
                x[:, dst] = x[:, src]
            self.W, self.cols = W, list(range(W))
        else:
            src = self._steps([self.cols[s] for s, c in zip(slots, cols) if c not in taken])
            dst = self._steps([c for c in cols if c not in taken])
            if src.numel():
                for x in self._rows():
                    x[:, dst] = x[:, src]
            self.cols = cols
        self.offs = [self.offs[s] for s in slots]
        self.lens = [self.lens[s] for s in slots]

    def _plan(self, slots: Sequence[int]) -> tuple[list[int] | None, set[int]]:
        """each new row's column: a row's first appearance keeps its own (`taken`), a repeat takes a free one;
        None when the columns run out"""
        taken: set[int] = set()
        cols: list[int | None] = []
        for s in slots:
            c = self.cols[s]
            cols.append(None if c in taken else c)
            taken.add(c)
        free = [c for c in range(self.W) if c not in taken]
        if cols.count(None) > len(free):
            return None, taken
        it = iter(free)
        return [c if c is not None else next(it) for c in cols], taken

    def used_after(self, slots: Sequence[int]) -> int:
        """the slots `select(slots)` leaves the rows holding"""
        cols, _ = self._plan(slots)
        return self.base + self._t * (len(slots) if cols is None else self.W)

    def detach(self) -> None:
        """another cache takes the arena: the rows held so far become a copy of their own"""
        if self._buf is None:
            return
        n = self.used
        self._own = (self._buf[0][:, :n].clone(), self._buf[1][:, :n].clone())
        self._buf = None

    def attach(self, kb: torch.Tensor, vb: torch.Tensor) -> None:
        """the rows back in the arena's slices `(kb, vb)` (copied in when they were held apart)"""
        if self._own is not None:
            n = self.used
            kb[:, :n].copy_(self._own[0])
            vb[:, :n].copy_(self._own[1])
            self._own = None
        self._buf = (kb, vb)
        self.device = kb.device


class GrantedIndexedLayer(_DynamicIndexedLayer):
    """A sparse-attention layer's cache as transformers keeps it - its keys, values and the indexer's keys grown by
    concatenation, which a tree's branches assign and restore as they walk - each growth past what it was granted
    asked of the scheduler first (`grant`, as `kind`), for twice what the rows reach: a decode's per-token growth
    reads the ledger once a doubling, and a concatenation's copy beside the rows it replaces fits what was asked. The
    epoch's KV is drawn on for what the rows allocate - what they reach less what the ledger counted of them already -
    not for the room ahead. The rows' own device is asked for, and what was asked there is kept: rows that came to a
    device some other way (a shed's or a regrow's move, a copy out of a card program's arena) were granted by what
    brought them, and a layer hopping back to a device it grew on asks nothing until it passes what it had there.
    `growth` prices an append before the pass (`cache_growth`), as the grant will."""

    def __init__(
        self, grant: Callable[..., None] | None, kind: str = "kv", what: str = "a sparse-attention layer's cache"
    ) -> None:
        super().__init__()
        self.grant = grant  # a fork of the layer asks the same (`branches._fork_layer`)
        self._kind, self._what = kind, what
        self._dev: torch.device | None = None  # where the rows grew last
        self._granted: dict[torch.device, int] = {}  # the room asked for on each device: twice what the rows reached
        self._drawn: dict[torch.device, int] = {}  # what of the rows the ledger counts on each device
        self._ik_row = 0  # an indexer key's bytes, once the layer has seen one: `growth` prices them before the pass

    @staticmethod
    def _nbytes(*ts: Any) -> int:
        return sum(t.numel() * t.element_size() for t in ts if isinstance(t, torch.Tensor))

    def _held_bytes(self) -> int:
        """the bytes of the rows held now: keys, values and the indexer's keys"""
        return self._nbytes(self.keys, self.values, self.indexer_keys)

    def _seen(self, dev: torch.device) -> tuple[int, int]:
        """what was asked for on `dev` and what the ledger counts there: the rows held now at least, where they came
        to `dev` since the layer last grew (moved or copied, and granted, by what brought them)"""
        granted, drawn = self._granted.get(dev, 0), self._drawn.get(dev, 0)
        if dev != self._dev:
            held = self._held_bytes()
            granted, drawn = max(granted, held), max(drawn, held)
        return granted, drawn

    def _room(self, dev: torch.device, need: int) -> None:
        """room asked for on `dev` for the rows reaching `need` bytes, where that passes what was asked there"""
        granted, drawn = self._seen(dev)
        self._dev = dev
        if need > granted:
            want = 2 * need
            if self.grant is not None:
                # the room checked is the doubling's; what comes off the epoch is what the rows allocate past what was
                # counted of them (`held`: the rest of the room, which nothing allocates)
                self.grant(want, self._kind, requester=self._what, device=dev, held=want - max(0, need - drawn))
            granted, drawn = want, max(drawn, need)
        self._granted[dev], self._drawn[dev] = granted, drawn

    def growth(self, B: int, T: int, Hk: int, d: int, dtype: torch.dtype, dev: torch.device) -> int:
        """The bytes the grant is asked for as the next append of T rows to B sequences lands on `dev` (`_room`):
        twice what the rows then reach, where that passes the room asked for there; 0 where it does not. The rows'
        own shape and dtype where the layer holds any, else the ones given; the indexer's keys once it has seen one."""
        k, ik = self.keys, self.indexer_keys
        if isinstance(k, torch.Tensor) and k.dim() == 4 and k.numel():
            B, Hk, d, el = int(k.shape[0]), int(k.shape[1]), int(k.shape[-1]), k.element_size()
        else:
            el = torch.empty(0, dtype=dtype).element_size()
        if isinstance(ik, torch.Tensor) and ik.dim() == 3 and ik.numel():
            self._ik_row = int(ik.shape[-1]) * ik.element_size()
        need = (self.get_seq_length() + T) * B * (2 * Hk * d * el + self._ik_row)
        return 2 * need if need > self._seen(dev)[0] else 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        T = max(1, int(key_states.shape[-2]))
        per = key_states.numel() // T * key_states.element_size()  # a position's bytes, every row and head
        # the rows reached: keys and values with these, and the indexer's keys as they stand
        self._room(key_states.device, 2 * (self.get_seq_length() + T) * per + self._nbytes(self.indexer_keys))
        return super().update(key_states, value_states, *args, **kwargs)

    def update_indexer(self, indexer_key_states: torch.Tensor) -> Any:
        T = max(1, int(indexer_key_states.shape[1]))
        have = self.indexer_keys
        n = int(have.shape[1]) if isinstance(have, torch.Tensor) and have.dim() == 3 else 0
        self._ik_row = int(indexer_key_states.shape[-1]) * indexer_key_states.element_size()
        ik = (n + T) * (indexer_key_states.numel() // T) * indexer_key_states.element_size()
        self._room(indexer_key_states.device, ik + self._nbytes(self.keys, self.values))
        return super().update_indexer(indexer_key_states)


class ArenaIndexedLayer(GrantedIndexedLayer):
    """A sparse-attention layer's cache - its keys and values, and the indexer's raw keys - held in an arena a card
    program owns, not grown by concatenation: rows are written in place at the front (`update`, `update_indexer`)
    and the views handed out are the front, so the program's captured kernels and the torch modules read and write
    the same rows. The program binds a sequence's cache to its arena (`attach`, rows held apart copied in) and lets
    it go (`detach`: the rows become a copy of their own, grown after as transformers' layer grows them, each growth
    asked of the scheduler as `GrantedIndexedLayer`'s are); `grow(need)` asks the program for room past the arena's capacity (it reallocates, copies and attaches every
    layer again, this one included). `keep_path` keeps a speculative pass's accepted path. Rows assigned from
    elsewhere are copied to the front; rows on another device (the layer given up to the host) detach the layer,
    every row moved straight there.
    `low` is the lowest indexer row written since the program last read it: its pooled keys of the blocks from
    there on are stale."""

    def __init__(self, grow: Callable[[int], None], grant: Callable[..., None] | None = None) -> None:
        # set before transformers' init, which assigns `keys`, `values` and `indexer_keys` through the setters below
        self._arena: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # K, V [Hk, cap, d]; raw [cap, di]
        self._own: list[torch.Tensor | None] = [None, None, None]  # detached: keys, values, indexer keys
        self._n = [0, 0, 0]  # the rows at the arena's front: keys, values, raw keys (set one at a time)
        self.low = 0
        super().__init__(grant, "kv", "a sparse-attention layer's cache, let go by the card program's arena")
        self._grow = grow

    # -- the rows ----------------------------------------------------------------------------------------------

    def _view(self, which: int) -> torch.Tensor | None:
        a = self._arena
        if a is None:
            return self._own[which]
        if which == 2:
            return a[2][None, : self._n[2]]
        return a[which][None, :, : self._n[which]]

    @property
    def keys(self) -> torch.Tensor | None:
        return self._view(0)

    @keys.setter
    def keys(self, t: torch.Tensor | None) -> None:
        self._put(0, t)

    @property
    def values(self) -> torch.Tensor | None:
        return self._view(1)

    @values.setter
    def values(self, t: torch.Tensor | None) -> None:
        self._put(1, t)

    @property
    def indexer_keys(self) -> torch.Tensor | None:
        return self._view(2)

    @indexer_keys.setter
    def indexer_keys(self, t: torch.Tensor | None) -> None:
        self._put(2, t)

    def _front(self, which: int, t: torch.Tensor) -> bool:
        """whether `t` is the arena's own front (a view the layer handed out, cut shorter or not)"""
        a = self._arena
        assert a is not None
        b = a[which]
        return (
            t.device == b.device
            and t.dtype == b.dtype
            and t.data_ptr() == b.data_ptr()
            and t.dim() == b.dim() + 1
            and int(t.shape[0]) == 1
            and tuple(t.shape[1:-2]) == tuple(b.shape[:-2])
            and int(t.shape[-1]) == int(b.shape[-1])
            and tuple(t.stride()[1:]) == tuple(b.stride())
        )

    def _put(self, which: int, t: torch.Tensor | None) -> None:
        a = self._arena
        if a is None:
            self._own[which] = t
            return
        if t is None:
            return  # transformers' init: the arena's front stands
        n = int(t.shape[-2])
        if self._front(which, t):
            self._set_len(which, n)
            return
        if t.device != a[which].device and not (self.in_ram and t.is_cuda):
            # rows moved to another device (the layer given up to the host): the layer leaves the arena, its other rows
            # moved straight there - never copied on the card first, in the room a shed is short of. Rows from the
            # card to an arena kept in RAM are copied in: RAM is where they live
            self.detach(t.device, {which: t})
            return
        if int(t.shape[0]) != 1:
            # a batch's rows: the layer leaves the arena
            self.detach()
            self._own[which] = t
            return
        if n > int(a[which].shape[-2]):
            self._grow(n)
            a = self._arena
            assert a is not None
        a[which][..., :n, :].copy_(t[0])
        self._set_len(which, n)
        if which == 2:
            self.low = 0

    def _set_len(self, which: int, n: int) -> None:
        self._n[which] = n
        if which == 2:
            self.low = min(self.low, n)

    def get_seq_length(self) -> int:
        if self._arena is None:
            return int(super().get_seq_length())
        return self._n[0]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: object, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._arena is None:
            return super().update(key_states, value_states, *args, **kwargs)
        T = int(key_states.shape[-2])
        n = self._n[0]
        need = n + T
        if need > int(self._arena[0].shape[-2]):
            self._grow(need)
        a = self._arena
        a[0][:, n:need].copy_(key_states[0])
        a[1][:, n:need].copy_(value_states[0])
        self._n[0] = self._n[1] = need
        k, v = self.keys, self.values
        assert k is not None and v is not None
        return k, v

    def update_indexer(self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        if self._arena is None:
            return super().update_indexer(indexer_key_states)
        T = int(indexer_key_states.shape[1])
        n = self._n[2]
        need = n + T
        if need > int(self._arena[2].shape[0]):
            self._grow(need)
        a = self._arena
        a[2][n:need].copy_(indexer_key_states[0])
        self.low = min(self.low, n)
        self._n[2] = need
        ik = self.indexer_keys
        assert ik is not None
        return ik

    def keep_path(self, base: int, path: Sequence[int]) -> None:
        """a speculative pass's rows past `base` cut to its accepted `path` (node indices, root first), moved into
        place: what the engine's commit (`ad`) asks of each attention layer that keeps its own rows"""
        path = [int(p) for p in path]
        n = len(path)
        base = int(base)
        a = self._arena
        prefix = path == list(range(n))
        if a is not None:
            if not prefix:
                self._settle()
                idx = torch.tensor(path, device=a[0].device) + base
                for b in a[:2]:
                    b[:, base : base + n] = b.index_select(1, idx)
                a[2][base : base + n] = a[2].index_select(0, idx)
                self.low = min(self.low, base)
            self._n = [base + n] * 3
            return
        keep = list(range(base)) + [base + p for p in path]
        for which in range(3):
            t = self._own[which]
            if t is None or not t.numel():
                continue
            dim = 1 if which == 2 else -2
            if prefix:
                t = t.narrow(dim, 0, len(keep))
            else:
                t = t.index_select(dim, torch.tensor(keep, device=t.device))
            self._own[which] = t

    def set_front(self, n: int) -> None:
        """the rows at the arena's front after the program's kernels wrote them: its keys, values and raw keys"""
        self._n = [int(n)] * 3

    # -- binding -----------------------------------------------------------------------------------------------

    def attached_to(self, k: torch.Tensor) -> bool:
        """whether the layer's keys are the arena slice `k`"""
        a = self._arena
        return a is not None and a[0].data_ptr() == k.data_ptr() and a[0].shape == k.shape

    def attach(self, k: torch.Tensor, v: torch.Tensor, raw: torch.Tensor) -> None:
        """the arena's slices (K and V [Hk, cap, d], raw [cap, di]) as the layer's: rows it held apart copied to the
        front, rows already in an arena left where the caller put them (a regrowth copies the arena whole)"""
        own = self._own if self._arena is None else [None, None, None]
        self._arena = (k, v, raw)
        self._own = [None, None, None]
        self.dtype, self.device = k.dtype, k.device
        self.is_initialized = True
        self.is_indexer_initialized = True
        self.indexer_dtype, self.indexer_device = raw.dtype, raw.device
        if own[0] is not None or own[2] is not None:
            self.load(own[0], own[1], own[2])

    def load(self, keys: torch.Tensor | None, values: torch.Tensor | None, ik: torch.Tensor | None) -> None:
        """rows [1, Hk, n, d] (and the indexer's [1, n, di]) copied to the attached arena's front"""
        a = self._arena
        assert a is not None
        n = int(keys.shape[-2]) if keys is not None and keys.numel() else 0
        ni = int(ik.shape[1]) if ik is not None and ik.numel() else 0
        if max(n, ni) > int(a[0].shape[-2]):
            self._grow(max(n, ni))
            a = self._arena
            assert a is not None
        if n:
            assert keys is not None and values is not None
            a[0][:, :n].copy_(keys[0])
            a[1][:, :n].copy_(values[0])
        if ni:
            assert ik is not None
            a[2][:ni].copy_(ik[0])
        self._n, self.low = [n, n, ni], 0

    @property
    def attached(self) -> bool:
        """whether the layer's rows are an arena's (not a copy of their own)"""
        return self._arena is not None

    @property
    def in_ram(self) -> bool:
        """whether the layer's arena is kept in RAM (`kv_host`), pinned, the card's kernels reading it in place"""
        a = self._arena
        return a is not None and a[0].device.type == "cpu"

    def _settle(self) -> None:
        """the card's kernels done with an arena in RAM before the host reads or moves its rows: they write it in
        place, behind the host's back (a verify pass's rows, still landing as the commit asks for them)"""
        if self.in_ram and torch.cuda.is_initialized():  # no kernel ran without the card started: nothing to wait for
            torch.cuda.synchronize()

    def growth(self, B: int, T: int, Hk: int, d: int, dtype: torch.dtype, dev: torch.device) -> int:
        """nothing while attached: the arena grows by the program's own grant (`grow`); detached, as
        `GrantedIndexedLayer`'s"""
        return 0 if self._arena is not None else super().growth(B, T, Hk, d, dtype, dev)

    def detach(self, device: DeviceSpec | None = None, given: dict[int, torch.Tensor] | None = None) -> None:
        """the program lets the arena go (another sequence takes it, a fork takes the rows, or the layer leaves the
        card): the rows held so far become a copy of their own, asked of the scheduler first - on `device` where they
        leave for another (the layer given up to the host), moved straight there and nothing copied where they were.
        `given`: rows the caller made there already ({0: keys, 1: values, 2: indexer keys}), taken as they are. A
        refusal leaves the layer attached, as it was"""
        a = self._arena
        if a is None:
            return
        self._settle()
        given = given or {}
        k, v, raw = a
        nk, nv, ni = self._n
        dev = k.device if device is None else where(device)
        rows = [k[None, :, :nk], v[None, :, :nv], raw[None, :ni]]
        nbytes = sum(int(t.numel()) * t.element_size() for w, t in enumerate(rows) if w not in given)
        if self.grant is not None and nbytes:
            what = "copied out" if dev == k.device else f"moved to {dev}"
            self.grant(nbytes, "kv", requester=f"a card program's arena rows, {what}", device=dev, draws="")
        self._own = [given[w] if w in given else t.to(dev, copy=True) for w, t in enumerate(rows)]
        self._arena = None
        self._dev = None  # the rows the copy made are counted where they are, at the next growth (`_seen`)
