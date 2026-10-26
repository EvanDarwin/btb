# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MTP drafter: a model's own drafting head run as a small model of its own, proposing the speculative tree."""

from __future__ import annotations

import heapq
import math
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, TypeAlias, TypedDict, Unpack, cast, overload

import torch

from .. import mlx as mlxdev
from ..gguf import QuantWeight
from ..kinds import Json, Tokens
from .cache import GrowLayer, indexer_keys, set_rows
from .fixed_rows import RowLinear
from .host import _HostLinear
from .native import Native
from .stored import CardStored

if TYPE_CHECKING:
    import mlx.core as mx_
    from transformers.cache_utils import CacheLayerMixin, DynamicCache
    from transformers.configuration_utils import PreTrainedConfig

    from ..sampling import Sampling
    from ..session import Session
    from .model import StreamedTextModel

    # the drafting layer's K/V written to its cache on MLX (a layer's `mx_update`): the step's rows in, the layer's
    # whole K and V out
    MxAppend: TypeAlias = Callable[[mx_.array, mx_.array], tuple[mx_.array, mx_.array]]
    # the attention the caller computes over the shared cache (the tree's node kernel): q, k, v in, [B, Hq, T, D] out
    MxAttend: TypeAlias = Callable[[mx_.array, mx_.array, mx_.array], mx_.array]
    # a drafted node's distribution over the drafter's ids [Vd]: a row of the MLX graph's on the MLX tree, else torch's
    DraftProbs: TypeAlias = mx_.array | torch.Tensor

# hidden rows under a run of tokens [B, T, H] (Qwen4's streams, [B, T, S * H]): the target's last layer's, which
# `prefill`, `extend` and a tree's root take, or those a step of the drafter's own leaves for the next depth
HiddenRows: TypeAlias = torch.Tensor
# the drafter's float32 logits [B, T, V] over the ids it drafts
DraftLogits: TypeAlias = torch.Tensor
# a chain the tree takes beside its own drafts (the n-gram's, a caller's): its tokens, their probability and the tag
# its nodes carry ("extra" without one)
ExtraChain: TypeAlias = tuple[Tokens, float] | tuple[Tokens, float, str]


def mx_rows(a: mx_.array) -> list[list[float]]:
    """a 2-D MLX array's rows as `tolist` reads them: Python numbers, an id array's ints"""
    if a.ndim != 2:
        raise ValueError(f"[mtp] rows of a 2-D array, not a {a.ndim}-D one")
    return cast("list[list[float]]", a.tolist())


def mx_ids(a: mx_.array) -> list[list[int]]:
    """a 2-D MLX id array's rows as ints"""
    return [[int(t) for t in r] for r in mx_rows(a)]


def mlx_topk_ids(logits: mx_.array, k: int) -> mx_.array:
    """the top-k ids [T, k] a row of an MLX logits array, sorted by value descending, the reduction (argpartition
    then sort) on the graph so a caller reads k ids a row instead of the whole vocab; lazy."""
    m = mlxdev.mx
    k = min(int(k), int(logits.shape[-1]))
    part = m.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    vals = m.take_along_axis(logits, part, axis=-1)
    order = m.argsort(-vals, axis=-1)
    return m.take_along_axis(part, order, axis=-1)


def drafter_weights(path: str | None) -> tuple[dict[str, torch.Tensor], Json | None]:
    """a drafter's own weights file (`weights`, in the checkpoint's place): its tensors by checkpoint name, and its
    `__meta__` record where it has one"""
    if path is None:
        return {}, None
    src = torch.load(path, map_location="cpu", weights_only=True)
    meta = src.pop("__meta__", None)
    return src, meta


def _dense_head(sm: StreamedTextModel) -> torch.Tensor:
    """the output head as one matrix: the module's weight, else the file's (a head on the card as a GGUF type's
    blocks, `CardStored`, holds none)"""
    head = sm.head
    return (head.weight if head is not None and not isinstance(head, CardStored) else sm._get(sm.head_key)).detach()


class _Int8Linear(torch.nn.Module):
    """A linear over int8 weights with one scale a row, packed in memory from a bf16/float linear: the
    drafter's weights on the torch tiers (`draft_bits` 8; 4 runs as 8 here). torch's packed int8 matmul where
    the device has it (CPU, MPS), the row-scaled product from the int8 tensor elsewhere - a card kernel is
    the card's to add."""

    def __init__(self, weight: torch.Tensor, rows: int = 1024) -> None:
        super().__init__()
        # quantize in row blocks to bound the float32 temporaries
        w0 = weight.detach()
        w8 = torch.empty(w0.shape, dtype=torch.int8, device=w0.device)
        scale = torch.empty(int(w0.shape[0]), dtype=torch.float32, device=w0.device)
        for r in range(0, int(w0.shape[0]), int(rows)):
            w = w0[r : r + rows].float()
            s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            w8[r : r + rows] = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
            scale[r : r + rows] = s
        self.w8 = torch.nn.Parameter(w8, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.packed_mm = torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_weight_int8pack_mm", weight.device.type.upper()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.packed_mm:
            y = torch._weight_int8pack_mm(x2, self.w8, self.scale.to(x2.dtype))
        else:
            y = x2 @ (self.w8.to(x2.dtype) * self.scale.to(x2.dtype)[:, None]).T
        return y.reshape(*shape[:-1], y.shape[-1])


# the head a torch tier drafts with (`_torch_head`): the model's rows (a view, or a copy on the drafter's device),
# packed to int8 (`draft_bits` below 16), or a GGUF type's blocks on the card
TorchHead: TypeAlias = "torch.Tensor | _Int8Linear | CardStored"


class DraftAttention(Protocol):
    """the drafting layer's attention as the tree reads it: a head's width and the scores' scale"""

    @property
    def head_dim(self) -> int: ...

    @property
    def scaling(self) -> float: ...


class DraftLayer(Protocol):
    """the drafting layer: a decoder layer of the family's, its linears the tier's (torch's, the host's, int8)"""

    @property
    def self_attn(self) -> DraftAttention: ...

    @property
    def mlp(self) -> torch.nn.Module: ...

    def modules(self) -> Iterator[torch.nn.Module]: ...

    def named_modules(self) -> Iterator[tuple[str, torch.nn.Module]]: ...


class DraftTree(NamedTuple):
    """a drafted tree's nodes in the drafter's order: each one's token, its parent node (-1 the root) and depth"""

    tokens: list[int]
    parents: list[int]
    depths: list[int]


class TaggedTree(NamedTuple):
    """`DraftTree` with each node's source: "mtp", or the tag of the chain it came from"""

    tokens: list[int]
    parents: list[int]
    depths: list[int]
    tags: list[str]


class SampledTree(NamedTuple):
    """`TaggedTree` drawn under a temperature: each expanded node's distribution (the root's at -1) and its draws in
    draw order, which the verify pass accepts against"""

    tokens: list[int]
    parents: list[int]
    depths: list[int]
    tags: list[str]
    probs: dict[int, DraftProbs]
    draws: dict[int, list[int]]


class TwoChains(NamedTuple):
    """`at`'s fixed tree: the first guess (None asked for none), then the chains from its top two children"""

    first: int | None
    a: list[int]
    b: list[int]


class TreeKnobs(TypedDict, total=False):
    """`au`'s shaping of the tree, each its parameter's"""

    max_depth: int
    top_k: int
    expand_k: int
    min_prob: float
    pop_ratio: float
    fan_ratio: float
    extra_chains: Sequence[ExtraChain]


class _Cand(NamedTuple):
    """a node the tree's heap may pop: its path's negative log-probability (the heap's key, `tie` its push order),
    parent node (-1 the root), token, depth, source and the tokens from the root to it"""

    neg: float
    tie: int
    parent: int
    tok: int
    depth: int
    tag: str
    path: list[int]


class _Node(NamedTuple):
    """a popped node a depth steps: its index among the tree's nodes, its path's negative log-probability, its
    parent node (-1 the root) and its token"""

    idx: int
    neg: float
    parent: int
    tok: int


class MTPDrafter:
    build_s: float
    # the heads it drafts with, made on first use: a torch tier's (`_torch_head`) and the host's (`_host_head`, False
    # where it has none)
    _head_t: TorchHead | None = None
    _head_h: _HostLinear | Literal[False] | None = None
    fc8: _Int8Linear | None = None  # the fc packed to int8 in its place (`_pack_torch`)
    _packed_torch: bool = False
    cache: DynamicCache | None
    cd: torch.dtype
    dev: torch.device
    fc: torch.Tensor | None
    fc_host: _HostLinear | None
    layer: DraftLayer
    mcfg: PreTrainedConfig
    sm: StreamedTextModel
    step_s: float
    steps: int
    train_mode: bool
    # the session whose rows the cache holds: set by that session's commit, let go by the next decode that takes the
    # drafter. A session reuses the drafter's rows only while it is theirs - the engine has one drafter, and two
    # sessions taking turns on it would each crop the other's rows as their own
    follows: weakref.ref[Session] | None = None

    def __init__(
        self,
        sm: StreamedTextModel,
        train: bool = False,
        weights: str | None = None,
        dev: str | torch.device | None = None,
    ) -> None:
        """the engine it drafts for and the device and dtype it runs in; a family's drafter builds its layer after,
        from the checkpoint's drafting head or, given `weights`, from that file in its place"""
        self.sm = sm
        self.dev = torch.device(dev) if dev is not None else sm.dev
        self.cd = torch.float32 if train else torch.bfloat16
        self.train_mode = train
        self.step_s = 0.0
        self.steps = 0
        # the last tree's nodes' path probabilities, in its order: what the verify pass's pricing weighs each by
        self.last_p: list[float] = []
        # the weights packed for `draft_bits` on MLX (codes, scales, biases), by (the weight's id, the dtype)
        self._q: dict[tuple[int, str], tuple[mx_.array, mx_.array, mx_.array]] = {}
        self.fc: torch.Tensor | None = None
        self.fc_host: _HostLinear | None = None
        self.cache = None
        # the position of the cache's first row, set by the first rows a reset cache takes: a resumed turn feeds the
        # drafter from its reuse point on, so its rows and the positions its callers name differ by it
        self.first: int | None = None

    def _kv(self) -> DynamicCache:
        """the drafter's cache, made by `reset` (a prefill's first)"""
        if self.cache is None:
            raise RuntimeError("[mtp] the drafter's cache was read before its reset made it")
        return self.cache

    def _layer0(self) -> CacheLayerMixin:
        """the cache's one layer, the drafting layer's attention rows"""
        from transformers.cache_utils import CacheLayerMixin

        cl = self._kv().layers[0]
        if not isinstance(cl, CacheLayerMixin):
            raise TypeError(f"[mtp] the drafter's cache layer is a {type(cl).__name__}, not an attention layer's")
        return cl

    def _mlx_ready(self) -> bool:
        """whether this drafter runs its own MLX graph (a family's drafter that has one says so, and then carries
        the graph's parts below)"""
        return False

    def _tree_kernel(self) -> bool:
        """whether the MLX tree steps its nodes through the node kernel over the shared cache"""
        return False

    def _mx_dtype(self) -> mx_.Dtype:
        """the MLX graph's activation dtype"""
        raise NotImplementedError

    def _mlx_body(
        self,
        em: mx_.array,
        hm: mx_.array,
        pos0: int,
        append: MxAppend | None,
        attn: MxAttend | None = None,
        layer: GrowLayer | None = None,
    ) -> mx_.array:
        """the drafting layer over MLX rows `em`/`hm` [B, T, H]: its output [B, T, H], lazily; `append(kh, vh)`
        gives the attention its K/V, or `attn(qh, kh, vh)` computes the attention itself"""
        raise NotImplementedError

    def _mlx_draw(
        self, out: mx_.array, k: int, sampling: Sampling, keys: Sequence[int]
    ) -> tuple[mx_.array, mx_.array, mx_.array]:
        """`k` children a row of `out` [B, 1, H] drawn under `sampling`: their log-probabilities, ids and the
        distribution, lazily"""
        raise NotImplementedError

    def _mlx_topk(self, out: mx_.array, k: int) -> tuple[mx_.array, mx_.array]:
        """the top-k log-probabilities and ids [B, k] of `out` [B, 1, H]'s next token, lazily"""
        raise NotImplementedError

    def _pass(
        self, tok_ids: torch.Tensor, h: HiddenRows, pos0: int, need_logits: bool
    ) -> tuple[DraftLogits | None, HiddenRows]:
        """the drafting head over `tok_ids` [B, T] and the hidden rows `h` under them at `pos0`: (the logits, or None
        without `need_logits`; the hidden rows it leaves, which the next step reads). Each family's own"""
        raise NotImplementedError

    def _step(self, tok_ids: torch.Tensor, h: HiddenRows, pos0: int) -> tuple[DraftLogits, HiddenRows]:
        """`_pass` with its logits"""
        self._anchor(pos0)
        logits, out = self._pass(tok_ids, h, pos0, True)
        assert logits is not None  # asked for
        return logits, out

    def _anchor(self, pos0: int) -> None:
        """rows written at `pos0` on: where the cache holds none yet, its first row's position (`first`)"""
        if self.first is None:
            self.first = int(pos0)

    def _rows_before(self, pos: int) -> int:
        """the rows the cache holds before position `pos`"""
        return max(0, int(pos) - (self.first or 0))

    def reset(self) -> None:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=self.mcfg)
        if self.sm.mlx is not None:
            cache.layers[0] = GrowLayer(shared=True)
        self.cache = cache
        self.first = None

    def _torch_head(self) -> TorchHead | None:
        """Return rows `[:draft_vocab]` of the lm_head for drafting, int8-quantized if `draft_bits` < 16.
        Cached on the instance; None if the head is on another device and there is no slice."""
        head = self._head_t
        if head is None:
            sm = self.sm
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            if isinstance(sm.head, CardStored):
                if sm.head.raw.device.type != self.dev.type:
                    return None
                # a GGUF head kept as its type's blocks on the card: its first n rows' blocks, a view
                rows, cols = sm.head.shape
                w = sm.head.stored
                if 0 < n < rows:
                    w = QuantWeight(w.raw[: w.raw.numel() // rows * n], w.q, (n, cols), w.grid, w.ksigns)
                self._head_t = CardStored(w, sm.head_key, scratch=sm.scratch)
                return self._head_t
            W = _dense_head(sm)
            whole = not (0 < n < int(W.shape[0]))
            if whole and W.device.type != self.dev.type:
                return None
            if not whole:
                W = W[:n]
            bits = int(getattr(self.sm, "draft_bits", 16) or 16)
            if bits < 16 and not self.train_mode:
                q8 = _Int8Linear(W.to(self.dev))
                sm.log(
                    f"[draft] the drafter's head: {'the first ' + str(n) + ' ids' if not whole else 'every id'} packed to "
                    f"8 bits in memory on {self.dev} ({q8.w8.numel() / 2**20:.0f} MB)"
                )
                head = q8
            elif W.device.type != self.dev.type:
                head = W.to(self.dev, self.cd)
            else:
                head = W  # the model's own rows, a view
            self._head_t = head
        return head

    def _host_head(self) -> _HostLinear | None:
        """Return rows `[:draft_vocab]` of the lm_head as a native `_HostLinear` for drafting on the CPU.
        Cached on the instance; None without a slice, without the native kernel, or with MLX."""
        head = self._head_h
        if head is None:
            sm = self.sm
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            head = False
            if sm.mlx is None and Native.gemv is not None and n > 0:
                w = sm._head_host().quant if sm._head_stored() is not None else None
                rows = int(sm._shape(sm.head_key)[0])
                if n < rows and w is not None:
                    # the first n rows of a head stored as a GGUF type: its first n rows of blocks, no copy
                    lin = _HostLinear(torch.zeros((), dtype=torch.bfloat16).expand(n, w.shape[1]), key=sm.head_key)
                    lin.quant = QuantWeight.of(w.raw[: w.raw.numel() // rows * n], w.q, n, w.shape[1])
                    lin.scratch = sm.scratch
                    sm.log(f"[draft] the drafter's head on the host: the first {n} ids ({w.q.value}, as stored)")
                    head = lin
                elif n < rows:
                    W = sm._get(sm.head_key)
                    head = _HostLinear(W[:n], key=sm.head_key)
                    sm.log(
                        f"[draft] the drafter's head on the host: the first {n} ids "
                        f"(bf16, {n * int(W.shape[1]) * 2 / 2**20:.0f} MB)"
                    )
            self._head_h = head
        return head or None

    def _head_logits(self, hn: HiddenRows) -> DraftLogits:
        """float32 logits of the drafter's last rows `hn` [B, T, H] over the head it drafts with: on the host the
        native slice (`_host_head`), else the model's own host head; on a torch tier the slice or the model's rows
        on the drafter's device (`_torch_head`), chunked across the vocabulary where neither is there"""
        sm = self.sm
        if self.dev.type == "cpu" and (sm.dev.type != "cpu" or Native.gemv is not None or sm.mlx is not None):
            host = self._host_head()
            return (sm._head_host() if host is None else host)(hn.float()).float()
        head = self._torch_head()
        if head is None:
            W = _dense_head(sm)
            step = 32768
            return torch.cat(
                [hn.to(self.cd) @ W[c : c + step].to(self.dev, self.cd).T for c in range(0, W.shape[0], step)], dim=-1
            ).float()
        if isinstance(head, (_Int8Linear, CardStored)):
            return head(hn).float()
        return (hn.to(head.dtype) @ head.T).float()

    def _pack_torch(self) -> int:
        """the drafter's layer on a torch tier with `draft_bits` below 16: every plain linear of it (and the fc
        tensor) replaced by an int8 one packed in memory; returns how many were packed (the host tier's
        native linears keep their own format)"""
        bits = int(getattr(self.sm, "draft_bits", 16) or 16)
        if bits >= 16 or self.train_mode or self._packed_torch:
            return 0
        self._packed_torch = True
        n = 0
        for _mname, mod in list(self.layer.named_modules()):
            for cname, child in list(mod.named_children()):
                # a plain linear, or the card layers' fixed-row one (`RowLinear`, the same parameters)
                if isinstance(child, torch.nn.Linear) and type(child) in (torch.nn.Linear, RowLinear):
                    setattr(mod, cname, _Int8Linear(child.weight.data))
                    n += 1
        if self.fc_host is None and self.fc is not None:
            self.fc8 = _Int8Linear(self.fc)
            self.fc = None
            n += 1
        if n:
            self.sm.log(
                f"[draft] the drafter's {n} linears packed to 8 bits in memory on {self.dev}"
                + (" (4 asked; 8 on this tier)" if bits < 8 else "")
            )
        return n

    @staticmethod
    def layer_rows(layer: CacheLayerMixin) -> tuple[torch.Tensor, ...]:
        """a drafter cache layer's per-row state, each tensor a row's along its first dim and a position's along its
        second last: its keys and values, and a sparse-attention layer's indexer keys [B, n, d_index] with them"""
        ik = indexer_keys(layer)
        k, v = layer.keys, layer.values
        if k is None or v is None:
            raise TypeError(f"[mtp] the rows of a {type(layer).__name__} that holds none")
        return (k, v) if ik is None or ik.dim() != 3 else (k, v, ik)

    @staticmethod
    def set_layer_rows(layer: CacheLayerMixin, rows: Sequence[torch.Tensor]) -> None:
        """`layer_rows`'s state written back: the tree's branches, a crop, the root restored"""
        from transformers.cache_utils import DynamicIndexedLayer

        set_rows(layer, rows[0], rows[1])
        if len(rows) > 2:
            if not isinstance(layer, DynamicIndexedLayer):
                raise TypeError(f"[mtp] indexer keys for a {type(layer).__name__}, which keeps none")
            layer.indexer_keys = rows[2]

    def crop(self, keep: int) -> None:
        """the cache cut back to the positions before `keep` (a position, as its callers count: its rows past a
        resumed turn's reuse point are fewer by it)"""
        from transformers.cache_utils import CacheLayerMixin

        rows = self._rows_before(keep)
        for layer in self._kv().layers:
            if isinstance(layer, CacheLayerMixin) and layer.keys is not None and layer.keys.shape[-2] > rows:
                self.set_layer_rows(layer, [t[..., :rows, :] for t in self.layer_rows(layer)])
        if rows == 0:
            self.first = None  # nothing held: the next rows anchor it again

    def prefill(self, prompt_ids: Tokens, h_all: HiddenRows) -> None:
        self.reset()
        n = len(prompt_ids)
        if n >= 2:
            self.extend(prompt_ids[1:n], h_all[:, : n - 1], 0)

    def extend(self, toks: Tokens, h: HiddenRows, pos0: int) -> None:
        """The drafter's cache fed `toks` at `pos0` on, in chunks the way the engine's own prefill runs: its
        attention over a long prompt at once materialized the scores of every row against every key (a 13k-row
        turn asked Metal for 33 GB), so a chunk is sized by the same rule, against the keys already in its cache."""
        toks = [int(t) for t in toks]
        self._anchor(pos0)
        a = 0
        while a < len(toks):
            C = int(self.sm.prefill_chunk or self.sm._auto_chunk(self._rows_before(pos0 + a)))
            b = min(len(toks), a + C)
            self._pass(torch.tensor([toks[a:b]]), h[:, a:b], pos0 + a, need_logits=False)
            a = b

    def ar(self, toks: Tokens, h: HiddenRows, pos0: int, k: int) -> list[int]:
        if k <= 0:
            return []
        logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
        t = int(logits[0, -1].argmax())
        out = [t]
        hp = hout[:, -1:]
        pos = pos0 + len(toks)
        for _ in range(k - 1):
            logits, hout = self._step(torch.tensor([[t]]), hp, pos)
            t = int(logits[0, -1].argmax())
            out.append(t)
            hp = hout
            pos += 1
        return out

    def at(self, toks: Tokens, h: HiddenRows, pos0: int, k: int) -> TwoChains:
        if k <= 1:
            g = self.ar(toks, h, pos0, k)
            return TwoChains(g[0] if g else None, [], [])
        logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
        g1 = int(logits[0, -1].argmax())
        hp = hout[:, -1:]
        pos = pos0 + len(toks)
        logits2, hout2 = self._step(torch.tensor([[g1]]), hp, pos)
        top2 = torch.topk(logits2[0, -1], 2).indices.tolist()
        a2, b2 = int(top2[0]), int(top2[1])
        after_g1 = pos + 1  # a position, as `crop` takes: g1's row is at `pos`
        chains: list[list[int]] = []
        for x2 in (a2, b2):
            chain = [x2]
            hx, px, t = hout2, pos + 1, x2
            for _ in range(k - 2):
                lg, hx = self._step(torch.tensor([[t]]), hx, px)
                t = int(lg[0, -1].argmax())
                chain.append(t)
                px += 1
            chains.append(chain)
            self.crop(after_g1)
        return TwoChains(g1, chains[0], chains[1])

    @overload
    def au(
        self,
        toks: Tokens,
        h: HiddenRows,
        pos0: int,
        budget: int,
        *,
        with_tags: Literal[False] = ...,
        sampling: Sampling | None = ...,
        **knobs: Unpack[TreeKnobs],
    ) -> DraftTree: ...

    @overload
    def au(
        self,
        toks: Tokens,
        h: HiddenRows,
        pos0: int,
        budget: int,
        *,
        with_tags: Literal[True],
        sampling: None = ...,
        **knobs: Unpack[TreeKnobs],
    ) -> TaggedTree: ...

    @overload
    def au(
        self,
        toks: Tokens,
        h: HiddenRows,
        pos0: int,
        budget: int,
        *,
        with_tags: Literal[True],
        sampling: Sampling,
        **knobs: Unpack[TreeKnobs],
    ) -> TaggedTree | SampledTree: ...

    @overload
    def au(
        self,
        toks: Tokens,
        h: HiddenRows,
        pos0: int,
        budget: int,
        *,
        with_tags: bool,
        sampling: Sampling | None = ...,
        **knobs: Unpack[TreeKnobs],
    ) -> DraftTree | TaggedTree | SampledTree: ...

    def au(
        self,
        toks: Tokens,
        h: HiddenRows,
        pos0: int,
        budget: int,
        max_depth: int = 8,
        top_k: int = 8,
        expand_k: int = 8,
        min_prob: float = 0.0,
        pop_ratio: float = 0.5,
        fan_ratio: float = 0.05,
        extra_chains: Sequence[ExtraChain] = (),
        with_tags: bool = False,
        sampling: Sampling | None = None,
    ) -> DraftTree | TaggedTree | SampledTree:
        """the tree of drafts; under a `sampling` (a temperature) every node's children are drawn without
        replacement from the drafter's sampled distribution, and the verify pass accepts them against it in draw
        order (`with_tags` then also returns {node: the distribution} and {node: the draws in order} for the root,
        -1, and each expanded node); the n-gram chains are left out of a sampled tree"""
        smp = sampling if (sampling is not None and not sampling.greedy) else None
        if smp is not None:
            # the proposal is the drafter's distribution at a multiple of the temperature: any proposal keeps the
            # verify pass exact (its ratio uses it), and a flatter one overlaps a target this head is sharper than
            from dataclasses import replace

            smp = replace(smp, temperature=smp.temperature * float(getattr(self.sm, "draft_temp_ratio", 1.0)))
        qrows: dict[int, DraftProbs] = {}
        draws: dict[int, list[int]] = {}  # under sampling: every draw of a node in draw order, for the verify pass
        nodes: list[tuple[int, int, int]] = []  # each node's token, parent and depth
        tags: dict[int, str] = {}

        def tree() -> DraftTree | TaggedTree | SampledTree:
            res = ([n[0] for n in nodes], [n[1] for n in nodes], [n[2] for n in nodes])
            if not with_tags:
                return DraftTree(*res)
            tg = [tags[i] for i in range(len(nodes))]
            return SampledTree(*res, tg, qrows, draws) if smp is not None else TaggedTree(*res, tg)

        self.last_p = []
        if budget <= 0:
            return tree()
        floor = -math.log(min_prob) if min_prob > 0 else float("inf")
        fan_gap = -math.log(fan_ratio) if fan_ratio > 0 else float("inf")
        chains = [
            ([int(t) for t in c[0]], math.log(max(1e-6, min(1.0, float(c[1])))), (c[2] if len(c) > 2 else "extra"))
            for c in extra_chains
            if c and len(c[0]) > 0 and smp is None
        ]
        self._anchor(pos0)
        root_pos = pos0 + len(toks) - 1  # the position the root's children are proposed at; a depth-d node steps d on
        layer = self._layer0()
        grow = layer if self._mlx_ready() and isinstance(layer, GrowLayer) and layer.shared and not layer.bits else None
        m = mlxdev.mx
        # the tree on one side, MLX's graph (`grow`) or torch's: each stepped node's output row, which its children
        # step from (the root's at -1), and, where a depth copies its nodes' prefixes, each one's K/V through it
        out_mx: dict[int, mx_.array] = {}
        out_t: dict[int, torch.Tensor] = {}
        kv_mx: dict[int, tuple[mx_.array, mx_.array]] = {}
        kv_t: dict[int, tuple[torch.Tensor, ...]] = {}
        kernel_tree = False
        row_parent: list[int] = []  # under the node kernel: a tree row's parent row (-1: the prefix)
        row_of: dict[int, int] = {}  # node index -> tree row
        root_vals: list[float]
        root_inds: list[int]
        if grow is not None:
            t0 = time.time()
            dt = self._mx_dtype()
            e = self.sm.embed(torch.tensor([[int(t) for t in toks]]))
            em = mlxdev.to_mx(e).astype(dt)
            hm = mlxdev.to_mx(h.detach().contiguous()).astype(dt)
            out = self._mlx_body(em, hm, pos0, grow.mx_update)
            if smp is not None:
                vals0, ids0, q0 = self._mlx_draw(out[:, -1:], top_k, smp, [smp.key_for(root_pos, salt=1)])
                qrows[-1] = q0[0]
            else:
                vals0, ids0 = self._mlx_topk(out[:, -1:], top_k)
            m.eval(vals0, ids0)
            self.step_s += time.time() - t0
            self.steps += 1
            root_len = self._kv().get_seq_length()
            kernel_tree = self._tree_kernel()
            if not kernel_tree:
                kv_mx[-1] = grow.mx_kv(0, root_len)
            root_vals, root_inds, out_mx[-1] = mx_rows(vals0)[0], mx_ids(ids0)[0], out[:, -1:]
            if kernel_tree:
                # the tree's rows go after the prefix, in stepping order; the prefix is never copied
                Hk_ = (
                    int(grow._shape[1])
                    if grow._shape is not None
                    else int(getattr(self.sm.cfg, "num_key_value_heads", 0))
                )
                hd_ = int(self.layer.self_attn.head_dim)
                grow._ensure(1, Hk_, root_len + budget + 1, hd_, mlxdev.torch_dtype(grow.mx_bufs()[0].dtype))
        else:
            logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
            root_len = self._kv().get_seq_length()
            # every per-row tensor of the layer (a sparse attention's indexer keys with its K/V) branches with the tree
            kv_t[-1] = tuple(t.clone() for t in self.layer_rows(layer))
            if smp is not None:
                ids_t, q_t = smp.draw_torch(logits[0, -1:].float(), [smp.key_for(root_pos, salt=1)], top_k)
                qrows[-1] = q_t[0]
                root_vals, root_inds = torch.log(q_t[0][ids_t[0]]).tolist(), ids_t[0].tolist()
            else:
                tb0 = torch.topk(torch.log_softmax(logits[0, -1].float(), dim=-1), min(top_k, logits.shape[-1]))
                root_vals, root_inds = tb0.values.tolist(), tb0.indices.tolist()
            out_t[-1] = hout[:, -1:]
        paths: dict[int, list[int]] = {}
        heap: list[_Cand] = []
        tie = 0

        def az(
            parent_idx: int,
            parent_neg: float,
            parent_depth: int,
            vals: Sequence[float],
            inds: Sequence[int],
            parent_path: list[int],
        ) -> None:
            nonlocal tie
            if smp is not None:
                # the verify pass tries all of a node's draws in order whether or not the tree holds them (an accepted
                # draw the tree lacks ends the walk), so the heap takes them by probability like the greedy tree
                draws[parent_idx] = [int(t) for lv, t in zip(vals, inds) if math.isfinite(lv)]  # no zero-mass draw
            best = max(vals)
            seen: dict[int, tuple[float, str]] = {}
            for lv, t in zip(vals, inds):
                if best - lv > fan_gap:
                    continue
                seen[int(t)] = (parent_neg - lv, "mtp")
            for ctoks, clp, ctag in chains:
                d = len(parent_path)
                if d < len(ctoks) and ctoks[:d] == parent_path:
                    cand = parent_neg - clp
                    if ctoks[d] not in seen or cand < seen[ctoks[d]][0]:
                        seen[ctoks[d]] = (cand, ctag)
            for t, (neg, tag) in seen.items():
                heapq.heappush(heap, _Cand(neg, tie, parent_idx, t, parent_depth + 1, tag, [*parent_path, t]))
                tie += 1

        az(-1, 0.0, 0, root_vals, root_inds, [])
        # a confident root over-builds: shrink the budget as it sharpens (size affects speed only, never a token), never
        # past the budget asked
        p1 = math.exp(float(max(root_vals)))
        cmin = int(getattr(self.sm, "tree_cap_min", 5))
        lo = float(getattr(self.sm, "tree_conf_lo", 0.5))
        hi = float(getattr(self.sm, "tree_conf_hi", 0.9))
        frac = min(1.0, max(0.0, (p1 - lo) / max(1e-6, hi - lo)))
        eff_budget = min(budget, max(min(cmin, budget), round(budget - (budget - cmin) * frac)))
        pop_gap = -math.log(pop_ratio)
        # a depth is stepped only when its nodes' path probability can pay for the drafter's step
        step_mass = float(getattr(self.sm, "tree_step_mass", 0.0) or 0.0)
        while heap and len(nodes) < eff_budget and heap[0].neg <= floor:
            popped: list[_Cand] = []
            while (
                heap
                and len(popped) < expand_k
                and len(nodes) + len(popped) < eff_budget
                and heap[0].neg <= floor
                and (not popped or heap[0].neg - popped[0].neg <= pop_gap)
            ):
                popped.append(heapq.heappop(heap))
            first = len(nodes)
            for c in popped:
                idx = len(nodes)
                nodes.append((c.tok, c.parent, c.depth))
                self.last_p.append(math.exp(-c.neg))
                paths[idx], tags[idx] = c.path, c.tag
            if len(nodes) >= eff_budget:
                break
            by_depth: dict[int, list[_Node]] = {}
            for j, c in enumerate(popped):
                if c.depth < max_depth:
                    by_depth.setdefault(c.depth, []).append(_Node(first + j, c.neg, c.parent, c.tok))
            for depth, group in sorted(by_depth.items()):
                if step_mass > 0 and sum(math.exp(-n.neg) for n in group) < step_mass:
                    continue
                ts = torch.tensor([[n.tok] for n in group])
                vals_l: list[list[float]]
                inds_l: list[list[int]]
                if grow is not None and kernel_tree:
                    # the group's rows appended to the shared cache where their paths say, the attention one node
                    # kernel call over the prefix and the tree's rows: nothing copied, nothing crosses to torch
                    t0 = time.time()
                    first_row = len(row_parent)
                    for j, n in enumerate(group):
                        row_of[n.idx] = first_row + j
                        row_parent.append(-1 if n.parent < 0 else row_of[n.parent])
                    meta_all, path_all, splits = mlxdev.tree_meta(root_len, row_parent)
                    meta_g, path_g = meta_all[first_row:], path_all[first_row:]
                    hs = m.concatenate([out_mx[n.parent] for n in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    bufs = grow.mx_bufs()
                    kbuf, vbuf = bufs[0], bufs[1]
                    scale_ = float(self.layer.self_attn.scaling)

                    def node_attn(
                        qh: mx_.array,
                        kh: mx_.array,
                        vh: mx_.array,
                        kbuf: mx_.array = kbuf,
                        vbuf: mx_.array = vbuf,
                        n0: int = root_len + first_row,
                        meta: mx_.array = meta_g,
                        path: mx_.array = path_g,
                        splits: int = splits,
                        scale_: float = scale_,
                    ) -> mx_.array:
                        # the rows written in place, the flag evaluated before the kernel reads them
                        fk_ = mlxdev.kv_store(kbuf, kh[:, :, 0, :].transpose(1, 0, 2), n0)
                        fv_ = mlxdev.kv_store(vbuf, vh[:, :, 0, :].transpose(1, 0, 2), n0)
                        m.eval(fk_, fv_)
                        a = mlxdev.attn_nodes(
                            qh[:, :, 0, :], kbuf, vbuf, meta, path, scale_, splits, odt=qh.dtype, shared=True
                        )
                        return a[:, :, None, :]

                    ho = self._mlx_body(em, hs, root_pos + depth, None, attn=node_attn)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + n.idx) for n in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, n in enumerate(group):
                            qrows[n.idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = mx_rows(vals), mx_ids(inds)
                    for b, n in enumerate(group):
                        out_mx[n.idx] = ho[b : b + 1, -1:]
                        az(n.idx, n.neg, depth, vals_l[b], inds_l[b], paths[n.idx])
                    continue
                if grow is not None:
                    # the group's prefixes and hidden rows concatenated on the graph, the node's K/V a lazy slice
                    # of the group's, the top-k read as k numbers a node: nothing crosses to torch
                    t0 = time.time()
                    kp = m.concatenate([kv_mx[n.parent][0] for n in group], axis=0)
                    vp = m.concatenate([kv_mx[n.parent][1] for n in group], axis=0)
                    hs = m.concatenate([out_mx[n.parent] for n in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    kn: list[mx_.array] = []

                    def cat_kv(
                        kh: mx_.array,
                        vh: mx_.array,
                        kp: mx_.array = kp,
                        vp: mx_.array = vp,
                        kn: list[mx_.array] = kn,
                    ) -> tuple[mx_.array, mx_.array]:
                        K = m.concatenate([kp, kh], axis=2)
                        V = m.concatenate([vp, vh], axis=2)
                        kn.extend((K, V))
                        return K, V

                    ho = self._mlx_body(em, hs, root_pos + depth, cat_kv)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + n.idx) for n in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, n in enumerate(group):
                            qrows[n.idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = mx_rows(vals), mx_ids(inds)
                    for b, n in enumerate(group):
                        kv_mx[n.idx] = (kn[0][b : b + 1], kn[1][b : b + 1])
                        out_mx[n.idx] = ho[b : b + 1, -1:]
                        az(n.idx, n.neg, depth, vals_l[b], inds_l[b], paths[n.idx])
                    continue
                self.set_layer_rows(
                    layer, [torch.cat([kv_t[n.parent][j] for n in group]) for j in range(len(kv_t[-1]))]
                )
                hs_t = torch.cat([out_t[n.parent] for n in group], dim=0)
                lg, ho_t = self._step(ts, hs_t, root_pos + depth)
                if smp is not None:
                    gk = [smp.key_for(root_pos + depth, salt=2 + n.idx) for n in group]
                    ids_g, qg_t = smp.draw_torch(lg[:, -1].float(), gk, top_k)
                    vals_l = torch.log(torch.gather(qg_t, 1, ids_g)).tolist()
                    inds_l = ids_g.tolist()
                    for b, n in enumerate(group):
                        qrows[n.idx] = qg_t[b]
                else:
                    tb = torch.topk(torch.log_softmax(lg[:, -1].float(), dim=-1), min(top_k, lg.shape[-1]), dim=-1)
                    vals_l, inds_l = tb.values.tolist(), tb.indices.tolist()
                rows_g = self.layer_rows(layer)
                for b, n in enumerate(group):
                    kv_t[n.idx] = tuple(t[b : b + 1].clone() for t in rows_g)
                    out_t[n.idx] = ho_t[b : b + 1, -1:]
                    az(n.idx, n.neg, depth, vals_l[b], inds_l[b], paths[n.idx])
        if grow is None:
            self.set_layer_rows(layer, kv_t[-1])
        return tree()
