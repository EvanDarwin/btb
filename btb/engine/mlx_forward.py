# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX device: the fused forward as one Metal graph per token (dense families and the Qwen3.5 hybrid),
the batched decode step and the forest prefill, and the pipelined greedy loops."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .. import mlx as mlxdev
from .. import pool
from ..kinds import (
    LayerKind,
    NodePath,
    Parents,
    PassTag,
    Quant,
    QuantClass,
    TokenRows,
    Tokens,
    latt_backend_key,
    quants_of,
)
from ..mlx import fused as fk
from ..mlx.legacyq import KINDS as LEGACY_KINDS
from ..mlx.q6k import gather_q6k
from ..sampling import GREEDY
from ..session import Session
from .cache import GraphStates, GrowLayer, forked
from .families import act_name
from .host import _HostLinear, copy_bytes
from .native import Native
from .state import _State

if TYPE_CHECKING:
    import mlx.core as mx_

    from ..mlx import Shared


@dataclass
class MlxState:
    """The MLX path's own state: its switches (shipped at these defaults; a caller may set one on the engine's
    `mlx_state`), what it holds in unified memory, the read-ahead buffer, the pool blocks it took, and the
    caches it fills on first use (the fused weights, the family check, the rope table, the embedding)."""

    batch: bool = True
    pipeline: bool = True
    rope_rows: bool = True
    forest_kernel: bool = False
    forest_max: int = 1024
    forest_tokens: int = 8192
    delta_mode: str = "recurrent"
    bytes: int = 0
    ahead: Shared | None = None
    pool_blocks: set[Any] = field(default_factory=set)
    weights: dict[Any, Any] | None = None
    family_ok: bool | None = None
    affine: bool = False  # linears bound as a GGUF's own quant blocks (their kernels): no megakernel, no fused step
    rope_cache: tuple[Any, ...] | None = None
    embed_w: Any = None
    embed_lin: _HostLinear | None = None


@dataclass
class _AttnParams:
    """A forward's `attn_params` for the decode row over `n` cache rows, one per window the layers read (None for
    a full-attention layer): built once a forward, not once a layer."""

    n: int
    by_window: dict[int | None, tuple[mx_.array, int]] = field(default_factory=dict)


def _pick_keys(pick: Any, positions: Any, last_only: bool) -> list[int] | None:
    """the noise keys of a pass's rows (the cache positions the rows decide at), the last row's alone under
    `last_only`; None when no sample is drawn"""
    if pick is None or pick.greedy:
        return None
    pos = [int(p) for p in positions]
    return [pick.key_for(p) for p in (pos[-1:] if last_only else pos)]


class _MxCheckpoints:
    """A tree pass's DeltaNet state kept in MLX memory: the [T, C, K] conv windows, the heads (q, k, v, g, beta)
    and the state the pass started from. `restore(path)` gives the accepted node's conv window as a torch view
    and its recurrent state recomputed along the path (lazy), the way the commit writes them in."""

    def __init__(self, conv: Any, heads: Any, state0: Any) -> None:
        self.conv, self.heads, self.state0 = conv, heads, state0

    def restore(self, path: NodePath) -> tuple[Any, Any]:
        """(the accepted node's conv window as a torch view, the kernel output to evaluate): the state is written
        into the cache's own buffer by the sequence kernel over the path"""
        # a torch view of the whole (evaluated) array: an MLX slice would be a kernel and a sync
        return mlxdev.from_mx(self.conv)[path[-1]][None], mlxdev.delta_chain_state(
            self.heads, self.state0, path, inplace=True
        )


class _MlxMixin(_State):
    def _mlx_fuse(self, layer: Any) -> None:
        be = self.mlx
        if be is None:
            return

        def fused(mod: Any, names: Sequence[str]) -> Any:
            if mod is None or not all(hasattr(mod, n) for n in names):
                return None
            ws = [getattr(getattr(mod, n), "mx", None) for n in names]
            return be.weight_fused(ws) if all(w is not None for w in ws) else None

        # the DeltaNet's four projections sit next to each other in the layer's buffer (module order), so one
        # matmul serves them: two of them are 32 rows, a launch each otherwise
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            la._mx_in = fused(la, ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))
        # q/k/v and gate/up as one GEMM for the batched decode (a view, no copy); the one-row kernel is row-wise, so
        # up to 16 rows the bits are the separate matmuls'
        at = getattr(layer, "self_attn", None)
        if at is not None:
            at._mx_qkv = fused(at, ("q_proj", "k_proj", "v_proj"))
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            mlp._mx_gu = fused(mlp, ("gate_proj", "up_proj"))

    def _shared_ahead(self, nbytes: int) -> tuple[Shared, int]:
        """A `Shared` of `nbytes` for a layer's linears, filled on MLX's CPU stream with the next layer's buffer
        started before this layer's weights are read (one touch of fresh memory, overlapped with the read);
        a slice of a pool block when the pool has one. Returns (Shared, offset)."""
        got = pool.POOL.take(nbytes)
        if got is not None:
            self.mlx_state.pool_blocks.add(got[0])
            return got
        m = mlxdev.mx()
        sh = self.mlx_state.ahead
        if sh is None or sh.nbytes != nbytes:
            sh = mlxdev.Shared(nbytes, stream=m.cpu)
        self.mlx_state.ahead = mlxdev.Shared(nbytes, stream=m.cpu, defer=True)
        return sh, 0

    def _bind_mlx_resident(self, layer: Any, checkpoint: bool = True) -> Any:
        # the layer's linears get their weights in unified memory; the module's other tensors (norms, biases,
        # the DeltaNet's small ones) stay torch and run on the CPU
        return self._bind_mlx_linears([m for m in layer.modules() if isinstance(m, _HostLinear)], checkpoint=checkpoint)

    def _bind_mlx_linears(self, lins: Sequence[Any], checkpoint: bool = True) -> Any:
        """Weights for the GPU: one MLX byte buffer per call, read straight from the checkpoint by the direct reader;
        a copy of the tensors when the reader is missing or the weights are not the checkpoint's."""
        from concurrent.futures import ThreadPoolExecutor

        lins = [m for m in lins if m.mx is None]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_q4k(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_q5k(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_q2k(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_q3k(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_iq4(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_lattice(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_legacy(m)]
        if not lins:
            return 0
        lins = [m for m in lins if not self._bind_gguf_q6k(m)]
        if not lins:
            return 0
        be = self.mlx
        assert be is not None
        rd = Native.read_direct
        direct = checkpoint and rd is not None and all(m.key in self.weight_map for m in lins)
        if not direct:
            # no direct reader (or the weights are not the checkpoint's, e.g. a dequantized GGUF): copy the
            # tensors, but into a pool block like the reader path, not a fresh MLX array each. A fresh array
            # beside the seeded-but-unused pool was the reader-less path's whole model held twice in MLX.
            sized, cur = [], 0
            for m in lins:
                rows, cols = (int(x) for x in m.weight.shape)
                nb = rows * cols * 2  # bf16
                sized.append((m, rows, cols, nb, cur))
                cur += (nb + 63) // 64 * 64
            sh, base = self._shared_ahead(cur)
            for m, rows, cols, nb, so in sized:
                w = m.weight.data
                if w.dtype != torch.bfloat16:
                    w = w.to(torch.bfloat16)
                region = sh.torch[base + so : base + so + nb]
                region.copy_(w.reshape(-1).view(torch.uint8))
                m.mx = be.weight_slot(sh, base + so, nb, (rows, cols))
                # the torch weight becomes a view of the pool block: the original tensor's bytes are freed, the
                # shape kept for any host path that still reads it
                m.weight = torch.nn.Parameter(region.view(torch.bfloat16).view(rows, cols), requires_grad=False)
            self.mlx_state.bytes += cur
            return cur
        items, cur = self._layer_items(lins)
        sh, base = self._shared_ahead(cur)
        chunk = getattr(self, "cold_chunk", 16 << 20)

        def read(it: Any) -> Any:
            m, path, off, nb, so = it
            if path is None:  # not on the drive as bf16 (a GGUF tensor of another type, a cast float): from memory
                copy_bytes(sh.torch[base + so : base + so + nb], m.weight.data)
            else:
                rd(path, off, nb, sh.torch[base + so : base + so + nb], chunk)
            return nb

        pool = getattr(self, "_mlx_pool", None)
        if pool is None:
            pool = self._mlx_pool = ThreadPoolExecutor(max_workers=4)
        for nb in pool.map(read, items):
            self.bytes_streamed += nb
        for m, _path, _off, nb, so in items:
            m.mx = be.weight_slot(sh, base + so, nb, tuple(m.weight.shape))
        self.mlx_state.bytes += cur
        return cur

    def _bind_gguf_legacy(self, m: Any) -> bool:
        """a GGUF Q4_0/Q4_1/Q8_0 tensor bound as its own blocks for `matvec_legacy` (`gguf_packed`): batch-invariant,
        so a verify pass computes each row as the one-row step does. The torch weight becomes a shape-only
        placeholder, so the bf16 copy `_get` made is released; False when the linear is not such a tensor."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        kind = t.tensor_type.name.lower() if t is not None else ""
        shape = tuple(m.weight.shape)
        if kind not in LEGACY_KINDS or len(shape) != 2 or shape[1] % 32:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_legacy(kind, raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_q4k(self, m: Any) -> bool:
        """a GGUF Q4_K tensor bound as its own native bytes for `matvec_q4k` (batch-invariant, so affine
        speculation is bit-exact) instead of the bf16-scale affine repack: the 144 B/256 superblocks stay packed
        and the torch weight becomes a placeholder, so the bf16 copy `_get` made is released. Runs before the
        affine binder, which would otherwise repack Q4_K. False when the linear is not a Q4_K tensor."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name != Quant.Q4_K:
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_q4k(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_q5k(self, m: Any) -> bool:
        """a GGUF Q5_K tensor bound as its own native bytes for `matvec_q5k` (batch-invariant, and beats the
        bf16-dequant path that lost to llama.cpp): the 176 B/256 superblocks stay packed, the torch weight a
        placeholder. False when the linear is not a Q5_K tensor."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name != Quant.Q5_K:
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_q5k(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_q2k(self, m: Any) -> bool:
        """a GGUF Q2_K tensor bound as its own native bytes for `matvec_q2k` (batch-invariant, beats the
        bf16-dequant path): the 84 B/256 superblocks stay packed, the torch weight a placeholder."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name != Quant.Q2_K:
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_q2k(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_q3k(self, m: Any) -> bool:
        """a GGUF Q3_K tensor bound as its own native bytes for `matvec_q3k` (batch-invariant, beats the
        bf16-dequant path): the 110 B/256 superblocks stay packed, the torch weight a placeholder."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name != Quant.Q3_K:
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_q3k(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_iq4(self, m: Any) -> bool:
        """a GGUF IQ4_NL/IQ4_XS tensor bound as its own native bytes for `matvec_iq4nl`/`matvec_iq4xs`
        (batch-invariant, beats the bf16-dequant path): the codebook blocks stay packed, the torch weight a
        placeholder."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name not in quants_of(QuantClass.IQ4):
            return False
        shape = tuple(m.weight.shape)
        blkw = 32 if t.tensor_type.name == Quant.IQ4_NL else 256
        if len(shape) != 2 or shape[1] % blkw:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_iq4nl(raw, shape) if t.tensor_type.name == Quant.IQ4_NL else be.weight_iq4xs(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    # GGUF IQ lattice type name -> the backend's lattice-kernel kind, from the LATTICE members of kinds.Quant
    # (the backend's `_LATT` carries the matching keys and their byte layouts); a new lattice quant is one member.
    _LATT_KINDS = {q.value: latt_backend_key(q) for q in quants_of(QuantClass.LATTICE)}

    def _bind_gguf_lattice(self, m: Any) -> bool:
        """a GGUF IQ lattice tensor (the grid-codebook quants behind the Unsloth dynamic mixes) bound as its own
        native bytes for `matvec_lattice` (batch-invariant, beats the bf16-dequant path); the torch weight a
        placeholder."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None:
            return False
        kind = self._LATT_KINDS.get(t.tensor_type.name)
        if kind is None:
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_lattice(kind, raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _bind_gguf_q6k(self, m: Any) -> bool:
        """a GGUF Q6_K tensor bound as its own bytes for `matvec_q6k` (the head and every ffn_down of a k-quant
        file): the 6-bit blocks stay packed instead of a bf16 copy, and the torch weight becomes a placeholder so
        the bf16 copy `_get` made is released. False when the linear is not a Q6_K tensor."""
        gg = getattr(self, "gguf", None)
        if gg is None or not getattr(self, "gguf_packed", True) or m.key not in getattr(self, "_gguf_names", {}):
            return False
        name = self._gguf_names[m.key]
        t = gg.tensors.get(name)
        if t is None or t.tensor_type.name != "Q6_K":
            return False
        shape = tuple(m.weight.shape)
        if len(shape) != 2 or shape[1] % 256:
            return False
        be = self.mlx
        assert be is not None
        raw = gg.raw(name).numpy()
        m.mx = be.weight_q6k(raw, shape)
        m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)
        self.mlx_state.affine = True  # not a plain bf16 slot: stays on the per-op path, off the megakernel
        self.mlx_state.bytes += int(raw.nbytes)
        return True

    def _mlx_act(self) -> Callable[[mx_.array], mx_.array] | None:
        if self.mlx is None:
            return None
        return self.mlx.act(act_name(self.cfg))

    def _mlx_ok(
        self, cache: Any, B: int, T: int, am: torch.Tensor | None, positions: torch.Tensor | None, n_layers: int
    ) -> bool:
        if self.mlx is None or B != 1 or am is not None or forked(cache):
            return False
        if not getattr(self, "mlx_fused", True) or getattr(self, "_probe", None) is not None:
            return False
        tree = bool(getattr(self, "aq", False)) and getattr(self, "ap", None) is not None
        if (tree or positions is not None) and T > 16:
            return False  # the node kernels and the fused step take 16 rows; a wider pass keeps the host path
        fam = self.fam
        ok = self.mlx_state.family_ok
        if ok is None:
            if fam.dense or fam.sandwich:
                attn = next((self.host[i].self_attn for i in self.host if hasattr(self.host[i], "self_attn")), None)
                # a sandwich family (Gemma 3) runs its sliding layers through the fused path too, windowed per layer;
                # the plain dense path has no window, so it stays whole-prefix only
                ok = (
                    self._mlx_act() is not None
                    and attn is not None
                    and (
                        fam.sandwich
                        or (
                            all(lt == LayerKind.FULL for lt in self.layer_types)
                            and getattr(attn, "sliding_window", None) is None
                        )
                    )
                )
            elif fam.hybrid:
                # the DeltaNet step per position runs through the native kernel; without it the host path
                ok = self._mlx_act() is not None and Native.delta_step is not None
            else:
                ok = False
            self.mlx_state.family_ok = ok
        if not ok:
            return False
        if not fam.hybrid and (tree or positions is not None) and (cache is None or not self._mlx_tree_able(cache)):
            # a tree on the dense path attends through the node kernel (each node walks its own ancestry), so
            # every layer's cache must be one the kernel reads; otherwise chains only
            return False
        return all(i in self.mlx_layers and i in self.host for i in range(n_layers))

    def _mlx_tree_prep(self, cache: Any, past: int, parents: Parents) -> tuple[Any, list[int]]:
        """What a tree verify pass needs once, not per layer: the node kernel's metadata (a cache with no row
        selected) and every node's position, past + depth, for the one-launch rope."""
        T = len(parents)
        depth = [0] * T
        for j in range(T):
            depth[j] = 0 if parents[j] < 0 else depth[parents[j]] + 1
        cl0 = cache.layers[0] if getattr(cache, "layers", None) else None
        prep = mlxdev.tree_meta(past, parents) if (cl0 is None or getattr(cl0, "_row", None) is None) else None
        return prep, [int(past) + d for d in depth]

    def mlx_attn_cost(self) -> None:
        """The node attention's cost per node and row for this model's heads, timed over synthetic caches of
        4096 and 16384 rows (a tree of 8 nodes against one): `_mlx_attn_slope` = [(rows, seconds a layer per
        node-row)], what `_spec_budget` adds to the short curve for a pass over `past` rows."""
        m = mlxdev.mx()
        c = self.cfg
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        # the first layer with attention (a hybrid's first layers are DeltaNet ones)
        attn = next((lay.self_attn for lay in self.host.values() if hasattr(lay, "self_attn")), None)
        hd = int(attn.head_dim) if attn is not None else 0
        if hd not in (64, 128, 256) or Hq // Hk > mlxdev.ATTN_MAXG:
            return
        slopes = []
        parents = [-1, 0, 1, 2, 0, 4, 5, 1]
        for rows in (4096, 16384):
            k = m.zeros((1, Hk, rows + 16, hd), dtype=m.bfloat16)
            v = m.zeros((1, Hk, rows + 16, hd), dtype=m.bfloat16)
            q8 = m.zeros((8, Hq, hd), dtype=m.float32)
            m.eval(k, v, q8)
            pa = mlxdev.attn_params(rows)
            prep = mlxdev.tree_meta(rows, parents)

            def one(q8: Any = q8, k: Any = k, v: Any = v, rows: int = rows, pa: Any = pa) -> Any:
                return mlxdev.attn_decode(q8[0], k, v, rows, 1.0, params=pa)

            def tree(q8: Any = q8, k: Any = k, v: Any = v, rows: int = rows, prep: Any = prep) -> Any:
                return mlxdev.attn_tree(q8, k, v, rows, parents, 1.0, prepared=prep)

            times = []
            for fn in (one, tree):
                m.eval(fn())
                best = float("inf")
                for _ in range(8):
                    t0 = time.perf_counter()
                    m.eval([fn() for _ in range(8)])
                    best = min(best, (time.perf_counter() - t0) / 8)
                times.append(best)
            slopes.append((rows, max(0.0, (times[1] - times[0]) / (7 * rows))))
        self._mlx_attn_slope = slopes
        self.log("[mlx] attention per node-row: " + ", ".join(f"{r} rows {s * 1e9:.1f} ns" for r, s in slopes))

    def mlx_warm(self, ids: Tokens, t_max: int | None = None) -> int:
        """The cost of a verify pass of 1 .. `t_max` rows on the fused MLX path, timed over a throwaway cache of
        `ids` into `_mlx_cost`, the curve the speculative loop weighs the tree against: every width's pass is
        paired with a one-row pass in the same breath, the fastest of each kept, so a busy machine moves both
        sides of the ratio alike (a lone slow reading once shut the tree for a whole run). Returns the widths
        measured; 0 where the fused tree path does not apply."""
        if self.mlx is None or not getattr(self, "mlx_fused", True):
            return 0
        t_max = int(t_max or self._spec_full())
        t_max = max(1, min(t_max, 16))
        ids_t = torch.as_tensor(list(ids), dtype=torch.long).view(1, -1)
        cost: dict[int, float] = {}
        with torch.inference_mode():
            cache = self.new_cache(max_len=int(ids_t.shape[1]) + t_max + 2)
            if not self._mlx_ok(cache, 1, 1, None, None, self.L):
                return 0
            self._prefill(ids_t, cache)
            tok = int(ids_t[0, -1])

            def timed(T: int) -> float:
                base = cache.get_seq_length()
                self.aa(None)
                t0 = time.perf_counter()
                try:
                    self.forward([[tok] * T], cache=cache, last_only=False, pick=True)
                finally:
                    self.ab()
                dt = time.perf_counter() - t0
                # the pass's rows dropped as the loop drops a rejected draft: the attention layers cropped, a
                # hybrid's DeltaNet states restored from the pass's checkpoints
                self.ad(cache, base, [0])
                for cl in cache.layers:
                    if isinstance(cl, GrowLayer) and cl.get_seq_length() > base:
                        cl.crop(base)
                return dt

            one = timed(1)
            for T in range(1, t_max + 1):
                if T > 1 and not self._mlx_ok(cache, 1, T, None, None, self.L):
                    break
                best = float("inf")
                for _rep in range(3 if T <= 8 else 2):
                    one = min(one, timed(1))
                    best = min(best, timed(T))
                cost[T] = best
            cost[1] = min(cost.get(1, one), one)
        if cost:
            self._mlx_cost = cost
            c1 = cost[1]
            self.log(
                "[mlx] pass cost by rows: "
                + ", ".join(f"{T}:{c / c1:.2f}x" for T, c in cost.items() if T in (1, 2, 4, 8, 12, t_max))
                + f" (one row {c1 * 1e3:.1f} ms)"
            )
        return len(cost)

    def _mlx_tree_able(self, cache: Any) -> bool:
        """Whether a tree of drafts can verify on the fused dense path: a shared cache (bf16 or int8), a head size
        and query grouping the node kernel takes, the kernel on."""
        if self.mlx is None or not self.fam.dense:
            return False
        cl = cache.layers[0] if getattr(cache, "layers", None) else None
        c = self.cfg
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        hd = int(getattr(c, "head_dim", None) or int(c.hidden_size) // Hq)
        return (
            isinstance(cl, GrowLayer)
            and cl.shared
            and cl._mx is not None
            and hd in (128, 256)
            and Hq // Hk <= mlxdev.ATTN_MAXG
            and bool(getattr(self, "mlx_attn_kernel", True))
        )

    def _mlx_consts(self, i: int, tmpl: Any, dt: torch.dtype) -> dict[str, Any]:
        """The layer's small tensors (norm weights) as MLX arrays, made once. Qwen3.5's RMSNorm scales by
        (1 + weight) in float32, so its weights are kept as float32 with the one added."""
        d = self.mlx_state.weights
        if d is None:
            d = self.mlx_state.weights = {}
        key = (i, dt)
        w = d.get(key)
        if w is not None:
            return w
        plus = self.fam.norm_centered

        def norm(mod: Any) -> Any:
            t = mod.weight.data.detach().float()
            if plus:
                t = 1.0 + t
            else:
                t = t.to(dt)
            return mlxdev.to_mx(t.contiguous()), float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)))

        if i < 0 and tmpl is None:
            a, e = norm(self.norm)
            w = {"norm": a, "eps": e}
        else:
            w = {}
            w["ln1"], w["eps1"] = norm(tmpl.input_layernorm)
            w["ln2"], w["eps2"] = norm(tmpl.post_attention_layernorm)
            if self.fam.sandwich:
                # the sandwich block norms the attention output (post_attention == ln2, reused), the MLP input, and
                # the MLP output before each residual add
                w["post_attn"], w["eps_pa"] = w["ln2"], w["eps2"]
                w["pre_ff"], w["eps_pf"] = norm(tmpl.pre_feedforward_layernorm)
                w["post_ff"], w["eps_pf2"] = norm(tmpl.post_feedforward_layernorm)
            at = getattr(tmpl, "self_attn", None)
            if at is not None and getattr(at, "q_norm", None) is not None:
                w["qn"], w["epsq"] = norm(at.q_norm)
                w["kn"], _ = norm(at.k_norm)
            la = getattr(tmpl, "linear_attn", None)
            if la is not None:
                f = lambda t: mlxdev.to_mx(t.data.detach().float().contiguous())
                w["delta"] = {
                    "conv_w": f(la.conv1d.weight.squeeze(1)),
                    "conv_b": None if la.conv1d.bias is None else f(la.conv1d.bias),
                    "a_log": f(la.A_log),
                    "dt_bias": f(la.dt_bias),
                    "norm_w": f(la.norm.weight),
                    "eps": float(getattr(la.norm, "variance_epsilon", getattr(la.norm, "eps", 1e-6))),
                    "hk": int(la.num_k_heads),
                    "hv": int(la.num_v_heads),
                    "dk": int(la.head_k_dim),
                    "dv": int(la.head_v_dim),
                    "key_dim": int(la.key_dim),
                    "value_dim": int(la.value_dim),
                }
        d[key] = w
        arrays = [x for x in w.values() if not isinstance(x, (float, dict))]
        arrays += [x for x in w.get("delta", {}).values() if not isinstance(x, (float, int)) and x is not None]
        mlxdev.mx().eval(*arrays)
        return w

    def _mlx_lin_adopt(self, cl: Any) -> Any:
        """The DeltaNet layer's states moved once into shared memory, so the GPU step and every torch path write
        the same tensors. Returns their MLX twins."""
        pend = getattr(cl, "_mx_pending", None)
        if pend is not None:
            # the pipelined decode carries the states in the graph between steps
            return pend
        lin: Any = self._lin(cl)
        c, r = lin
        if getattr(c, "_mx_twin", None) is None or c.dtype != torch.float32:
            cs = mlxdev.shared_tensor(tuple(c.shape), torch.float32)
            cs.copy_(c)
            c = cs
        if getattr(r, "_mx_twin", None) is None or r.dtype != torch.float32:
            rs = mlxdev.shared_tensor(tuple(r.shape), torch.float32)
            rs.copy_(r)
            r = rs
        self._lin_set(cl, c, r)
        return c._mx_twin, r._mx_twin

    def _mlx_flush_states(self, cache: Any) -> None:
        """DeltaNet states the pipelined decode kept in the graph are written into the cache's tensors, so every
        torch path (trees, snapshots, the module) sees them."""
        m = mlxdev.mx()
        todo = [cl for cl in cache.layers if getattr(cl, "_mx_pending", None) is not None]
        if not todo:
            return
        m.eval(*[a for cl in todo for a in cl._mx_pending])
        for cl in todo:
            cn, rn = cl._mx_pending
            cl._mx_pending = None
            cl._mx_prev = None
            c, r = self._lin(cl)
            c.copy_(mlxdev.from_mx(cn))
            r.copy_(mlxdev.from_mx(rn))

    def _mlx_delta_proj(self, la: Any, x: mx_.array) -> Any:
        """The DeltaNet's projections of x [T, H]: (mixed, z, b, a), one matmul when the weights are fused."""
        be = self.mlx
        assert be is not None
        f = getattr(la, "_mx_in", None)
        if f is None:
            return (
                be.matmul(x, la.in_proj_qkv.mx),
                be.matmul(x, la.in_proj_z.mx),
                be.matmul(x, la.in_proj_b.mx),
                be.matmul(x, la.in_proj_a.mx),
            )
        y = be.matmul(x, f)
        c, zn, hv = int(la.in_proj_qkv.mx.shape[0]), int(la.in_proj_z.mx.shape[0]), int(la.in_proj_b.mx.shape[0])
        return y[:, :c], y[:, c : c + zn], y[:, c + zn : c + zn + hv], y[:, c + zn + hv :]

    def _mlx_delta_step(
        self,
        w: dict[str, Any],
        mixed: mx_.array,
        z: mx_.array,
        a: mx_.array,
        b: mx_.array,
        conv_mx: mx_.array,
        rec_mx: mx_.array,
    ) -> Any:
        """One DeltaNet position in MLX ops as the module's recurrent rule (float32), compiled once per shape.
        Returns (core [Hv*dv], conv_new [C, K], state_new [Hv, dk, dv]), lazily."""
        d = w["delta"]
        fn = d.get("step")
        if fn is None:
            fn = d["step"] = mlxdev.delta_step_fn(
                d["hk"], d["hv"], d["dk"], d["dv"], d["key_dim"], d["eps"], d["conv_b"] is not None
            )
        conv_b = d["conv_b"] if d["conv_b"] is not None else d["a_log"]
        core, conv_new, S = fn(
            mixed, z, a, b, conv_mx[0], rec_mx[0], d["conv_w"], conv_b, d["a_log"], d["dt_bias"], d["norm_w"]
        )
        return core, conv_new, S

    def _mlx_rope(self) -> tuple[mx_.array | dict[str, mx_.array], int, float]:
        """The rotary module's frequencies for `Backend.rope_fast`: (freqs, rotary dims, attention scaling),
        refreshed when the module swaps its table (longrope switches past the original window). For a dual-rope
        family (Gemma 3) `freqs` is a dict keyed by layer type; rotary dims and scaling are shared across types."""
        if self.fam.dual_rope:
            return self._mlx_rope_dual()
        inv = self.rotary.inv_freq
        cur = self.mlx_state.rope_cache
        if cur is None or cur[3] is not inv:
            freqs = mlxdev.to_mx((1.0 / inv.detach().float()).contiguous())
            mlxdev.mx().eval(freqs)
            cur = self.mlx_state.rope_cache = (
                freqs,
                int(inv.shape[0]) * 2,
                float(getattr(self.rotary, "attention_scaling", 1.0)),
                inv,
            )
        return cur[:3]

    def _mlx_rope_dual(self) -> tuple[dict[str, mx_.array], int, float]:
        """Gemma 3's local/global rope: one freqs array per layer type, keyed by the type the layer reads; the
        rotary dims and the attention scaling must be one for every type (the fused rope takes a single pair)."""
        types = sorted(set(self.layer_types))
        invs = tuple(getattr(self.rotary, f"{lt}_inv_freq") for lt in types)
        cur = self.mlx_state.rope_cache
        if cur is None or len(cur[3]) != len(invs) or any(a is not b for a, b in zip(cur[3], invs, strict=False)):
            freqs = {
                lt: mlxdev.to_mx((1.0 / inv.detach().float()).contiguous()) for lt, inv in zip(types, invs, strict=True)
            }
            mlxdev.mx().eval(*freqs.values())
            dims = {int(inv.shape[0]) * 2 for inv in invs}
            scales = {float(getattr(self.rotary, f"{lt}_attention_scaling", 1.0)) for lt in types}
            assert len(dims) == 1 and len(scales) == 1, f"the layer types' ropes differ: dims {dims}, scaling {scales}"
            cur = self.mlx_state.rope_cache = (freqs, dims.pop(), scales.pop(), invs)
        return cur[:3]

    def _mlx_norm(self, a: mx_.array, wt: mx_.array, eps: float) -> mx_.array:
        m: Any = self.mlx and mlxdev.mx()
        if wt.dtype != a.dtype:
            return m.fast.rms_norm(a.astype(wt.dtype), wt, eps).astype(a.dtype)
        return m.fast.rms_norm(a, wt, eps)

    def _mlx_prefill_able(self, cl: Any, T: int, hd: int, Hq: int, Hk: int) -> bool:
        """The fused prefill kernel's gate: a causal chunk of more than 16 rows at head size 256 (the size MLX's own
        fused attention lacks) over a shared bf16 cache layer with no row selected, `mlx_attn_prefill` on."""
        return (
            T > 16
            and hd == 256
            and isinstance(cl, GrowLayer)
            and cl.shared
            and not cl.bits
            and cl._mx is not None
            and cl._mx[0].dtype == mlxdev.mx().bfloat16
            and Hq // Hk <= mlxdev.ATTN_MAXG
            and 8 % (Hq // Hk) == 0
            and getattr(cl, "_row", None) is None
            and not getattr(cl, "_flat", False)
            and bool(getattr(self, "mlx_attn_kernel", True))
            and bool(getattr(self, "mlx_attn_prefill", True))
        )

    def _mlx_attend(
        self,
        qh: mx_.array,
        K: mx_.array,
        V: mx_.array,
        cl: Any,
        T: int,
        hd: int,
        Hq: int,
        Hk: int,
        scale: float,
        mask: Any,
        attn_pa: _AttnParams | None,
        nodes: Any = None,
        prepared: Any = None,
        win: int | None = None,
    ) -> tuple[mx_.array, _AttnParams | None]:
        """Attention for the fused paths: the node kernel for a decode row past `mlx_attn_rows` (0 with speculation
        on: every row and node computes as the one-row step would) and for `nodes` = (past, parents) of a
        verify pass; MLX's fused attention otherwise. `win` is a sliding layer's window (Gemma 3), passed to the
        node kernels and, on the SDPA fallback (a small-cache decode, a sliding prefill), applied as the mask.
        Returns ([T, Hq*hd], attn_pa)."""
        m = mlxdev.mx()
        able = (
            cl is not None
            and isinstance(cl, GrowLayer)
            and cl.shared
            and hd in (128, 256)
            and Hq // Hk <= mlxdev.ATTN_MAXG
            and (cl._mx[0].dtype == m.bfloat16 or cl.bits)
            and getattr(self, "mlx_attn_kernel", True)
        )
        # an int8 layer's rows and their scales go to the kernel as they are
        kq: dict[str, Any] = {"ks": cl._mx[2], "vs": cl._mx[3]} if (able and cl.bits) else {}
        # a batched cache with one row selected reads that row's slice of the buffer (or, of a flat buffer,
        # its stretch from its offset)
        b = int(cl._row) if (able and getattr(cl, "_row", None) is not None) else 0
        place: dict[str, Any] = {"batch": b}
        if b and getattr(cl, "_flat", False):
            place = {"batch": b, "pbase": int(cl._offs[b]), "kbatch": 0}
        if able and nodes is not None:
            past, parents = nodes
            a = mlxdev.attn_tree(
                qh[0].transpose(1, 0, 2).astype(m.float32),
                cl._mx[0],
                cl._mx[1],
                past,
                parents,
                scale,
                window=win,
                prepared=prepared if not b else None,
                **place,
                **kq,
            )
            return a.astype(qh.dtype).reshape(T, Hq * hd), attn_pa
        if able and T == 1 and mask is None and cl._n >= self.mlx_attn_rows:
            if b:
                pa = mlxdev.attn_params(cl._n, win, **place)
            else:
                if attn_pa is None or attn_pa.n != cl._n:
                    attn_pa = _AttnParams(cl._n)
                if win not in attn_pa.by_window:
                    attn_pa.by_window[win] = mlxdev.attn_params(cl._n, win)
                pa = attn_pa.by_window[win]
            a = mlxdev.attn_decode(qh[0, :, 0].astype(m.float32), cl._mx[0], cl._mx[1], cl._n, scale, params=pa, **kq)
            return a.astype(qh.dtype).reshape(1, Hq * hd), attn_pa
        if able and mask == "causal" and win is None and self._mlx_prefill_able(cl, T, hd, Hq, Hk):
            # the chunk's rows are appended already: row t at cache row past + t
            a = mlxdev.attn_prefill(qh[0].transpose(1, 0, 2), cl._mx[0], cl._mx[1], cl._n - T, scale, odt=qh.dtype)
            return a.reshape(T, Hq * hd), attn_pa
        if win is not None and (mask is None or isinstance(mask, str)):
            # SDPA over the whole cache, masked to a sliding window: row t (at past + t) keeps keys (p - win, p]
            n = int(K.shape[-2])
            p = (n - T) + m.arange(T, dtype=m.int32)[:, None]
            j = m.arange(n, dtype=m.int32)[None, :]
            mask = ((j <= p) & (j > p - win))[None, None]
        self._tag(PassTag.MLX_SDPA)
        a = m.fast.scaled_dot_product_attention(qh, K, V, scale=scale, mask=mask)
        return a[0].transpose(1, 0, 2).reshape(T, Hq * hd), attn_pa

    def _mlx_cache(self, cache: Any, i: int, kh: mx_.array, vh: mx_.array) -> tuple[Any, mx_.array, mx_.array]:
        """Append this layer's K/V rows: in place on the GPU for a shared cache layer, through torch otherwise."""
        m = mlxdev.mx()
        if cache is None:
            return None, kh, vh
        cl = cache.layers[i]
        if isinstance(cl, GrowLayer) and cl.shared:
            K, V = cl.mx_update(kh, vh)
            return cl, K, V
        m.eval(kh, vh)
        kf, vf = cache.update(mlxdev.from_mx(kh), mlxdev.from_mx(vh), i)
        return None, mlxdev.to_mx(kf), mlxdev.to_mx(vf)

    def _mlx_finish(
        self,
        hm: mx_.array,
        last_only: bool,
        head: bool,
        n_layers: int,
        t0: float,
        extra: Sequence[Any] = (),
        after: Callable[[], Any] | None = None,
        lazy: bool = False,
        pick: Any = None,
        keys: Sequence[int] | None = None,
    ) -> Any:
        """The tail of the fused paths: the final norm and the head in the graph when the head is on the GPU, else
        the host tail. `extra` is evaluated with it and `after()` runs then; `lazy` returns the logits
        unevaluated; `pick` (a Sampling) returns [1, T] int32 ids picked in the graph instead of logits, the
        rows' noise under `keys`."""
        m = mlxdev.mx()
        be = self.mlx
        assert be is not None
        fused_head = (
            n_layers == self.L
            and head
            and self.norm is not None
            and self.head_host is not None
            and self.head_host.mx is not None
            and self.head is None
        )
        if fused_head:
            assert self.head_host is not None  # fused_head requires the resident head
            w = self._mlx_consts(-1, None, torch.float32)
            hn = m.fast.rms_norm((hm[-1:] if last_only else hm).astype(m.float32), w["norm"], w["eps"])
            logits = be.matmul(hn, self.head_host.mx)
            if lazy:
                return logits
            if pick is not None:
                am = pick.pick_mx(logits, keys or [0] * int(logits.shape[0]))
                m.eval(am, *extra)
            else:
                m.eval(logits, *extra)
        else:
            m.eval(hm, *extra)
        if after is not None:
            after()
        be.stat["layers"] += n_layers
        be.stat["layer_s"] += time.time() - t0
        be.stat["evals"] += 1
        self.compute_s += time.time() - t0
        if n_layers < self.L:
            return None
        if fused_head:
            return mlxdev.from_mx(am if pick is not None else logits)[None]
        return self._finish(mlxdev.from_mx(hm)[None], last_only, head)

    def _mlx_rope_rows(self, x: mx_.array, rows: Sequence[int], rd: int, freqs: mx_.array, rscale: float) -> mx_.array:
        """`rope_fast` over B rows x [B, heads, d], row b at position rows[b], one call per distinct position (the
        one-row step's bits); returns [heads, B, d] in row order."""
        m = mlxdev.mx()
        be = self.mlx
        assert be is not None
        if self.mlx_state.rope_rows:
            # one launch for every row at its own position, the bits of rope_fast at that position
            return mlxdev.rope_rows(x, rd, freqs, rscale, rows)
        groups: dict[int, list[int]] = {}
        for b, p in enumerate(rows):
            groups.setdefault(int(p), []).append(b)
        if len(groups) == 1:
            p = next(iter(groups))
            return be.rope_fast(x[:, :, None, :], rd, freqs, rscale, p)[:, :, 0, :]
        parts, order = [], []
        for p, idx in groups.items():
            xg = m.take(x, m.array(idx, dtype=m.int32), axis=0)
            parts.append(be.rope_fast(xg[:, :, None, :], rd, freqs, rscale, p)[:, :, 0, :])
            order.extend(idx)
        inv = [0] * len(order)
        for i, b in enumerate(order):
            inv[b] = i
        return m.take(m.concatenate(parts, axis=0), m.array(inv, dtype=m.int32), axis=0)

    def _mlx_attend_rows(self, qh: mx_.array, cl: GrowLayer, scale: float, prepared: Any = None) -> mx_.array:
        """One decode step's attention for B rows: `qh` [B, Hq, hd], each row over its own cache slice at its own
        length through the node kernel; a head size the kernel lacks runs MLX's attention per row."""
        m = mlxdev.mx()
        B, Hq, hd = (int(x) for x in qh.shape)
        Hk = int(cl._mx[0].shape[1])
        able = (
            hd in (128, 256)
            and Hq // Hk <= mlxdev.ATTN_MAXG
            and (cl._mx[0].dtype == m.bfloat16 or cl.bits)
            and getattr(self, "mlx_attn_kernel", True)
        )
        if able:
            kq: dict[str, Any] = {"ks": cl._mx[2], "vs": cl._mx[3]} if cl.bits else {}
            if getattr(cl, "_mx2", None) is not None:
                kq.update({"segs": cl._seg, "k2": cl._mx2[0], "v2": cl._mx2[1]})
                if cl.bits:
                    kq.update({"ks2": cl._mx2[2], "vs2": cl._mx2[3]})
            # q in its own dtype, the scale applied in the kernel, the output in q's dtype: no casts, no op
            a = mlxdev.attn_rows(
                qh,
                cl._mx[0],
                cl._mx[1],
                list(cl._ns),
                scale,
                prepared=prepared,
                odt=qh.dtype,
                pbases=cl._offs if cl._flat else None,
                **kq,
            )
            return a.reshape(B, Hq * hd)
        self._tag(PassTag.MLX_SDPA)
        outs = []
        for b in range(B):
            K, V = cl.mx_kv_row(b)
            a = m.fast.scaled_dot_product_attention(qh[b][None, :, None, :].astype(K.dtype), K, V, scale=scale)
            outs.append(a[0, :, 0, :])
        return m.stack(outs, axis=0).reshape(B, Hq * hd)

    def _mlx_attend_forest(
        self, qh: mx_.array, kh: mx_.array, vh: mx_.array, cl: GrowLayer, scale: float, forest: dict[str, Any]
    ) -> mx_.array:
        """A batched prefill's attention over a group of rows' tokens end to end: gathered into a right-padded
        [G, heads, Lmax, hd] batch, MLX's causal attention once (padding never enters a real token's result),
        scattered back; `mlx_forest_kernel` routes the group through the node kernel instead."""
        m = mlxdev.mx()
        T, Hq, hd = (int(x) for x in qh.shape)
        Hk = int(kh.shape[1])
        if self.mlx_state.forest_kernel and hd in (128, 256) and Hq // Hk <= mlxdev.ATTN_MAXG:
            kq: dict[str, Any] = {"ks": cl._mx[2], "vs": cl._mx[3]} if cl.bits else {}
            a = mlxdev.attn_nodes(
                qh, cl._mx[0], cl._mx[1], forest["meta"], forest["path"], scale, forest["splits"], odt=qh.dtype, **kq
            )
            return a.reshape(T, Hq * hd)
        self._tag(PassTag.MLX_SDPA)
        G, Lm = forest["G"], forest["Lmax"]
        gi = forest["gather"]

        def padded(x: mx_.array, H: int) -> mx_.array:
            return m.take(x, gi, axis=0).reshape(G, Lm, H, hd).transpose(0, 2, 1, 3)

        a = m.fast.scaled_dot_product_attention(
            padded(qh, Hq), padded(kh, Hk).astype(qh.dtype), padded(vh, Hk).astype(qh.dtype), scale=scale, mask="causal"
        )
        a = a.transpose(0, 2, 1, 3).reshape(G * Lm, Hq * hd)
        return m.take(a, forest["scatter"], axis=0)

    def _mlx_forest(self, rows: TokenRows, offs: Sequence[int]) -> tuple[list[int], dict[str, Any]]:
        """The plan of one batched-prefill pass over `rows` at flat-buffer offsets `offs`: the tokens end to end,
        their positions, the gather/scatter to the right-padded batch, the node metadata, each row's last token."""
        m = mlxdev.mx()
        lens = [len(r) for r in rows]
        G, Lm = len(rows), max(lens)
        toks = [t for r in rows for t in r]
        pos = [p for L in lens for p in range(L)]
        gather: list[int] = []
        scatter: list[int] = []
        last: list[int] = []
        grp: list[tuple[int, int, int]] = []
        s = 0
        for b, L in enumerate(lens):
            # a pad slot points at the row's last token (a valid row; its result is dropped)
            gather.extend(range(s, s + L))
            gather.extend([s + L - 1] * (Lm - L))
            scatter.extend(range(b * Lm, b * Lm + L))
            last.append(s + L - 1)
            grp.append((b, s, L))
            s += L
        meta, path, splits = mlxdev.forest_meta(lens, offs)
        forest = {
            "pos": m.array(pos, dtype=m.uint32),
            "node0": int(offs[0]),
            "meta": meta,
            "path": path,
            "splits": splits,
            "last": m.array(last, dtype=m.int32),
            "rows": grp,
            "G": G,
            "Lmax": Lm,
            "gather": m.array(gather, dtype=m.int32),
            "scatter": m.array(scatter, dtype=m.int32),
        }
        return toks, forest

    def _forward_mlx(
        self,
        h: Any,
        pe: Any,
        cache: Any,
        on_layer: Any,
        last_only: bool,
        head: bool,
        n_layers: int,
        hm: Any = None,
        lazy: bool = False,
        rows: Sequence[int] | None = None,
        forest: dict[str, Any] | None = None,
        pick: Any = None,
    ) -> Any:
        """The dense families on the GPU: one MLX graph per forward, the K/V appended in unified memory inside it.
        `hm` an MLX hidden state in place of `h`; `lazy` returns the logits unevaluated; `rows` a batched decode
        step (hm[b] at position rows[b]); `forest` a batched prefill (the logits each row's last token's); `pick`
        the ids picked in the graph (a chain's rows at past + j, a tree's at past + depth)."""
        self._tag_tiers(n_layers)
        self._tag_quant()
        if self.fam.hybrid:
            return self._forward_mlx_hybrid(
                h, pe, cache, on_layer, last_only, head, n_layers, hm=hm, lazy=lazy, pick=pick
            )
        self._tag(PassTag.MLX_STEP)
        m = mlxdev.mx()
        be = self.mlx
        assert be is not None
        c = self.cfg
        if hm is None:
            T = int(h.shape[1])
            dt = h.dtype
            hm = mlxdev.to_mx(h[0])
        else:
            T = int(hm.shape[0])
            dt = mlxdev.torch_dtype(hm.dtype)
        act = self._mlx_act()
        assert act is not None  # _mlx_ok admits the family only with an MLX activation
        silu = act_name(c) in ("silu", "swish")  # the fused gate/up kernel is silu's; gelu keeps `act`
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        past = cache.get_seq_length() if cache is not None else 0
        freqs, rd, rscale = self._mlx_rope()
        # a sandwich family's sliding layers attend through their window; the dense families have none
        sw = int(getattr(c, "sliding_window", 0) or 0) if self.fam.sandwich else 0
        if self.cold:
            self._cold_start(n_layers)
        t0 = time.time()
        attn_pa: _AttnParams | None = None
        taps = []
        # under speculation a chain or a tree of drafted tokens attends through the decode kernel node by node, as the
        # one-row step would, so its verdicts are the greedy loop's
        spec_chain = bool(getattr(self, "aq", False)) and past > 0
        parents: Any = getattr(self, "ap", None) if spec_chain else None
        tree = parents is not None and any(parents[j] != j - 1 for j in range(T))
        if not tree:
            parents = None
        tree_prep, tree_pos = self._mlx_tree_prep(cache, past, parents) if tree else (None, [])
        keys = _pick_keys(pick, tree_pos if tree else (rows if rows is not None else range(past, past + T)), last_only)
        # the fused kernels serve the one-row step, a chain and a tree (<= 16 rows, the matvec tile) of the dense
        # families with a full rotary; the batched and forest passes keep their own launches
        fuse = (
            bool(getattr(self, "mlx_fused_kernels", True))
            and rows is None
            and forest is None
            and T <= 16
            and (self.fam.kernel_layout or self.fam.sandwich)
            and rd == int(self.host[0].self_attn.head_dim)
            and hm.dtype == m.bfloat16
        )
        self._tag(PassTag.MLX_STEP_FUSED if fuse else PassTag.MLX_STEP_UNFUSED)
        x_next: Any = None
        rows_prep = None
        batched = rows is not None or forest is not None
        if rows is not None:
            # every layer's attention this step reads the same lengths: the node metadata once, not per layer
            cl0 = cache.layers[0]
            rows_prep = mlxdev.rows_meta(
                [int(r) + 1 for r in rows],
                segs=cl0._seg if getattr(cl0, "_dec_cap", 0) else None,
                pbases=cl0._offs if getattr(cl0, "_flat", False) else None,
            )
        for i in range(n_layers):
            tmpl = self.host[i]
            if i in self.cold:
                self._cold_wait(i)
            w = self._mlx_consts(i, tmpl, dt)
            at, mlp = tmpl.self_attn, tmpl.mlp
            hd = int(at.head_dim)
            fq = freqs[self.layer_types[i]] if isinstance(freqs, dict) else freqs
            win = sw if sw and self.layer_types[i] == LayerKind.SLIDING else None
            x = x_next if x_next is not None else self._mlx_norm(hm, w["ln1"], w["eps1"])
            roped = False
            if hasattr(at, "qkv_proj"):  # q, k and v as one projection (Phi-3's layout)
                qkv = be.matmul(x, at.qkv_proj.mx)
                nq, nk = Hq * hd, Hk * hd
                q = qkv[:, :nq].reshape(T, Hq, hd)
                k = qkv[:, nq : nq + nk].reshape(T, Hk, hd)
                v = qkv[:, nq + nk :].reshape(T, Hk, hd)
            elif getattr(at, "_mx_qkv", None) is not None and (batched or T <= 16):
                qkv = be.matmul(x, at._mx_qkv).reshape(T, Hq + 2 * Hk, hd)
                if fuse and not batched:
                    # the q/k norms and the rope in one launch: row t at past + t (a chain), or at its depth (a tree)
                    qk_pos = tree_pos if tree else [past + t for t in range(T)]
                    q, k = fk.qk_norm_rope(
                        qkv[:, :Hq], qkv[:, Hq : Hq + Hk], w["qn"], w["kn"], w["epsq"], rd, fq, rscale, qk_pos
                    )
                    v = qkv[:, Hq + Hk :]
                    roped = True
                else:
                    q = self._mlx_norm(qkv[:, :Hq], w["qn"], w["epsq"])
                    k = self._mlx_norm(qkv[:, Hq : Hq + Hk], w["kn"], w["epsq"])
                    v = qkv[:, Hq + Hk :]
            else:
                # separate q/k/v projections (a GGUF file's tensors, or an unfused layout): the q/k norms and the
                # rope in the one launch fk.qk_norm_rope serves, exactly the fused-qkv branch's on the same rows
                q = be.matmul(x, at.q_proj.mx).reshape(T, Hq, hd)
                k = be.matmul(x, at.k_proj.mx).reshape(T, Hk, hd)
                v = be.matmul(x, at.v_proj.mx).reshape(T, Hk, hd)
                if fuse and not batched:
                    qk_pos = tree_pos if tree else [past + t for t in range(T)]
                    q, k = fk.qk_norm_rope(q, k, w["qn"], w["kn"], w["epsq"], rd, fq, rscale, qk_pos)
                    roped = True
                else:
                    q = self._mlx_norm(q, w["qn"], w["epsq"])
                    k = self._mlx_norm(k, w["kn"], w["epsq"])
            if rows is not None:
                # B rows, one token each at its own position: the weights read once for all, each row rotated,
                # appended and attended over its own slice as its single decode would be
                cl = cache.layers[i]
                if self.mlx_state.rope_rows:
                    qr, kr = mlxdev.rope_rows2(q, k, rd, fq, rscale, rows)  # both in one launch
                else:
                    qr = self._mlx_rope_rows(q, rows, rd, fq, rscale)
                    kr = self._mlx_rope_rows(k, rows, rd, fq, rscale)
                cl.mx_update_rows(kr[:, :, None, :], v[:, :, None, :])
                a = self._mlx_attend_rows(qr, cl, float(at.scaling), prepared=rows_prep)
            elif forest is not None:
                # a group of rows' prompts end to end: one gemm at the group's size, each token rotated at its
                # position in its row, the rows one slab in the flat buffer, every token attending its row's prefix
                cl = cache.layers[i]
                qr, kr = mlxdev.rope_rows2(q, k, rd, fq, rscale, forest["pos"])
                cl.forest_store(kr, v, forest["node0"])
                a = self._mlx_attend_forest(qr, kr, v, cl, float(at.scaling), forest)
            elif roped:
                qh, kh = q.transpose(1, 0, 2)[None], k.transpose(1, 0, 2)[None]
                vh = v.transpose(1, 0, 2)[None]
            elif tree:
                # every node rotated at its own position (past + depth) in one launch for q and k, the bits
                # of the one-row step's rope at that position
                qr, kr = mlxdev.rope_rows2(q, k, rd, fq, rscale, tree_pos)
                qh, kh = qr.transpose(1, 0, 2)[None], kr.transpose(1, 0, 2)[None]
                vh = v.transpose(1, 0, 2)[None]
            else:
                qh = be.rope_fast(q.transpose(1, 0, 2), rd, fq, rscale, past)[None]
                kh = be.rope_fast(k.transpose(1, 0, 2), rd, fq, rscale, past)[None]
                vh = v.transpose(1, 0, 2)[None]
            if not batched:
                cl, K, V = self._mlx_cache(cache, i, kh, vh)
                a, attn_pa = self._mlx_attend(
                    qh,
                    K,
                    V,
                    cl,
                    T,
                    hd,
                    Hq,
                    Hk,
                    float(at.scaling),
                    "causal" if T > 1 else None,
                    attn_pa,
                    nodes=(past, parents if tree else list(range(-1, T - 1))) if (spec_chain and T > 1) else None,
                    prepared=tree_prep,
                    win=win,
                )
            attn_out = be.matmul(a, at.o_proj.mx)
            if self.fam.sandwich:
                # Gemma norms the attention output, then adds it to the residual; the MLP reads its own pre-norm
                hm = (
                    fk.sandwich_add(hm, attn_out, w["post_attn"], w["eps_pa"])
                    if fuse
                    else hm + self._mlx_norm(attn_out, w["post_attn"], w["eps_pa"])
                )
                x2 = self._mlx_norm(hm, w["pre_ff"], w["eps_pf"])
            elif fuse:
                hm, x2 = fk.add_rmsnorm(hm, attn_out, w["ln2"], w["eps2"])
            else:
                hm = hm + attn_out
                x2 = self._mlx_norm(hm, w["ln2"], w["eps2"])
            gu_w = getattr(mlp, "_mx_gu", None)
            if hasattr(mlp, "gate_up_proj"):  # gate and up as one projection (Phi-3's layout)
                gate, up = m.split(be.matmul(x2, mlp.gate_up_proj.mx), 2, axis=-1)
                mid = up * act(gate)
                down = be.matmul(mid, mlp.down_proj.mx)
            elif gu_w is not None and (batched or T <= 16):
                # gate and up as one matvec
                gu = be.matmul(x2, gu_w)
                if fuse and silu:
                    mid = fk.silu_mul(gu)
                else:
                    half = int(gu.shape[1]) // 2
                    mid = act(gu[:, :half]) * gu[:, half:]
                down = be.matmul(mid, mlp.down_proj.mx)
            else:
                mid = act(be.matmul(x2, mlp.gate_proj.mx)) * be.matmul(x2, mlp.up_proj.mx)
                down = be.matmul(mid, mlp.down_proj.mx)
            if self.fam.sandwich:
                # norm the MLP output, then add; the next layer takes its own input norm (no fusion across the add)
                hm = (
                    fk.sandwich_add(hm, down, w["post_ff"], w["eps_pf2"])
                    if fuse
                    else hm + self._mlx_norm(down, w["post_ff"], w["eps_pf2"])
                )
                x_next = None
            elif fuse and i + 1 < n_layers and self.layer_types[i + 1] == LayerKind.FULL:
                # the residual add fused with the next layer's input norm
                w_next = self._mlx_consts(i + 1, self.host[i + 1], dt)
                hm, x_next = fk.add_rmsnorm(hm, down, w_next["ln1"], w_next["eps1"])
            else:
                hm = hm + down
                x_next = None
            if i in self.cold:
                m.eval(hm)
                self._cold_release(i)
            if on_layer is not None:
                # handed over after the one eval at the end: an eval here would sync the GPU every layer
                taps.append((i, hm))
        if taps:
            return self._mlx_finish(
                hm,
                last_only,
                head,
                n_layers,
                t0,
                extra=[h_ for _, h_ in taps],
                after=lambda: [on_layer(i_, mlxdev.from_mx(h_)[None]) for i_, h_ in taps],
                pick=pick,
                keys=keys,
            )
        if forest is not None:
            # only each row's last token reaches the norm and the head
            hm = m.take(hm, forest["last"], axis=0)
        return self._mlx_finish(hm, last_only, head, n_layers, t0, lazy=lazy, pick=pick, keys=keys)

    def _mega_ok(self, cache: Any, T: int, on_layer: Any, head: bool, pick: Any, positions: Any) -> bool:
        """the megakernel takes a pass of 1..16 rows over an arena cache of the dense family, the head and the
        pick in the graph (the argmax in the kernel; a sample over the logits it leaves), every layer resident"""
        mg = getattr(self, "_mega", None)
        if mg is None or cache is None or not (1 <= T <= 16) or on_layer is not None or not head or pick is None:
            return False
        if self.cold or not self.fam.kernel_layout or self.fam.sandwich or self.mlx_state.affine:
            return False
        if positions is not None and not self._mega_positions_ok(cache, T, positions):
            return False
        if self.compute_dtype is not None and self.compute_dtype != torch.bfloat16:
            return False
        layers = getattr(cache, "layers", None)
        if not layers or any(not isinstance(cl, GrowLayer) or cl.arena is None for cl in layers):
            return False
        past = cache.get_seq_length()
        cap = layers[0].arena[3]
        from ..mlx.attn import ATTN_BLOCK

        return past + T <= cap and past + T <= mg.SPL * ATTN_BLOCK

    def _mega_positions_ok(self, cache: Any, T: int, positions: Any) -> bool:
        """the tree's natural positions (past + a node's depth), the only ones the kernel computes"""
        try:
            pos = [int(v) for v in np.asarray(positions).reshape(-1).tolist()]
        except (TypeError, ValueError):
            return False
        if len(pos) != T:
            return False
        past = cache.get_seq_length()
        parents = getattr(self, "ap", None) if getattr(self, "aq", False) and past > 0 else None
        if parents is None:
            parents = list(range(-1, T - 1))
        depth = [0] * T
        for j in range(T):
            depth[j] = 0 if parents[j] < 0 else depth[parents[j]] + 1
        return pos == [past + d for d in depth]

    def _forward_mega(self, ids: Any, cache: Any, T: int, pick: Any = None) -> torch.Tensor:
        """The pass as one dispatch: the picked ids [1, T] int32 (the kernel's argmax, or a sample over the logits
        it leaves in its scratch); the cache's rows appended in the kernel."""
        self._tag(PassTag.MLX_MEGA)
        self._tag_tiers(self.L)
        self._tag_quant()
        m = mlxdev.mx()
        mg = self._mega
        assert mg is not None  # _forward_mega runs only when the megakernel is built
        t0 = time.time()
        past = cache.get_seq_length()
        spec_chain = bool(getattr(self, "aq", False)) and past > 0
        parents = getattr(self, "ap", None) if spec_chain else None
        if parents is None:
            parents = list(range(-1, T - 1))
        meta, path, splits = mlxdev.tree_meta(past, parents)
        depth = [0] * T
        for j in range(T):
            depth[j] = 0 if parents[j] < 0 else depth[parents[j]] + 1
        cl0 = cache.layers[0]
        buf, _ok, _ov, cap = cl0.arena
        Hk, hd = mg.Hk, mg.hd

        def kv_off(i: int, which: int) -> int:
            return int(cache.layers[i].arena[1 + which])

        ids_a = m.array(np.asarray(ids, dtype=np.uint32))
        pos = m.array(np.asarray([past + d for d in depth], dtype=np.uint32))
        sampled = pick is not None and not pick.greedy
        out, cnt = mg.run(buf, cap, kv_off, ids_a, pos, meta, path, splits, past, argmax=not sampled)
        if sampled:
            # the sample over the logits the kernel's head op left in its scratch, ordered after the pass by
            # taking its output as an input, evaluated with it: one sync a pass
            V = mg.V
            raw = mg.scr[mg.toff["logits"] : mg.toff["logits"] + T * V * 4].view(m.float32).reshape(T, V)
            out = pick.pick_mx(raw, [pick.key_for(past + d) for d in depth], after=out)
        m.eval(out, cnt)
        if int(cnt[-1]):
            raise RuntimeError("[mega] the pass's grid barrier timed out")
        for cl in cache.layers:
            cl.mx_advance(T, Hk, hd, torch.bfloat16)
        be = self.mlx
        assert be is not None
        be.stat["layers"] += self.L
        be.stat["layer_s"] += time.time() - t0
        be.stat["evals"] += 1
        self.compute_s += time.time() - t0
        return torch.from_numpy(np.asarray(out).astype(np.int32)).view(1, T)

    def _mlx_embed(self) -> mx_.array:
        """The token embedding table as an MLX array (the head's, when tied; a copy in unified memory else)."""
        w = self.mlx_state.embed_w
        if w is None:
            if (
                self.head_host is not None
                and self.head_host.mx is not None
                and self.head_key == self.prefix + "embed_tokens.weight"
            ):
                w = self.head_host.mx.get()
            else:
                assert self.embed_table is not None  # the untied embedding table backs the head
                lin = _HostLinear(self.embed_table, key=self.prefix + "embed_tokens.weight")
                self._bind_mlx_linears([lin])
                self.mlx_state.embed_lin = lin
                w = lin.mx.get()
            self.mlx_state.embed_w = w
        return w

    def _mlx_embed_rows(self, tok: Any) -> mx_.array:
        """the embedding rows for token ids `tok` (an MLX int array): gathered straight from the packed bytes when
        the tied head is Q6_K (no bf16 copy of the whole table), else a take on the resident table; scaled as
        `embed` scales them (Gemma's sqrt(hidden)), so every graph entry embeds alike"""
        hh = self.head_host
        if (
            hh is not None
            and getattr(hh.mx, "q6k", None) is not None
            and self.head_key == self.prefix + "embed_tokens.weight"
        ):
            raw, _rows, cols = hh.mx.q6k
            rows = gather_q6k(raw, tok, cols)
        else:
            rows = mlxdev.mx().take(self._mlx_embed(), tok, axis=0)
        return rows if self.embed_scale is None else rows * self.embed_scale

    def _mlx_batch_ok(self, B: int, on_layer: Any, prefill_only: bool) -> bool:
        """B > 1 rows decode together on the MLX device: the dense families through the fused forward, each
        row at its own length. A mask is fine (a left-padded batch is unpadded to its rows)."""
        return (
            self.mlx is not None
            and B > 1
            and self.mlx_state.batch
            and on_layer is None
            and not prefill_only
            and self.fam.dense
            and not self.cold
            and self.norm is not None
            and self.head_host is not None
            and self.head_host.mx is not None
            and self.head is None
            and self._mlx_ok(None, 1, 1, None, None, self.L)
        )

    def _generate_greedy_mlx_batch(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: Any,
        attention_mask: torch.Tensor | None,
        t0: float,
        sampling: Any = None,
    ) -> list[list[int]]:
        """B rows decoded together at their own lengths (no padding, no mask): the prompts prefilled as a forest
        into one flat buffer (short rows in groups, one fused forward each; long rows alone), then one fused
        forward a step for every row. A batch past memory runs as fixed-size epochs. Returns a list per row."""
        m = mlxdev.mx()
        smp = sampling or GREEDY
        am = None if attention_mask is None else torch.as_tensor(attention_mask)
        rows: list[list[int]] = []
        for b in range(int(ids.shape[0])):
            toks = ids[b].tolist()
            keep = am[b].tolist() if am is not None else [1] * len(toks)
            rows.append([int(t) for t, k in zip(toks, keep) if k])
        B = len(rows)
        target = max(len(r) for r in rows) + int(max_new)
        mb = self.scheduler.max_batch(target)
        if mb is not None and mb < B:
            out = []
            for s in range(0, B, mb):
                out.extend(
                    self._generate_greedy_mlx_batch(
                        ids[s : s + mb], max_new, eos, None if am is None else am[s : s + mb], t0, smp
                    )
                )
            return out
        # short rows first: a forest group is a run of consecutive rows, so its rows are one stretch of the
        # flat buffer and its tokens one stretch of the forward
        order = sorted(range(B), key=lambda b: len(rows[b]))
        rows = [rows[b] for b in order]
        lens = [len(r) for r in rows]
        offs, tot = [], 0
        for L in lens:
            offs.append(tot)
            tot += L
        # the flat buffer holds the prompts (row b's from offs[b]); the decode steps go to the step buffer,
        # one slot a step for every row, so no step copies a cache buffer
        cache = self.new_cache(max_len=max(lens))
        layers = [cl for cl in cache.layers if isinstance(cl, GrowLayer)]
        for cl in layers:
            cl.batch_rows(B, dec_cap=int(max_new) + 1, lens=lens)
        out = [[] for _ in range(B)]
        done = [False] * B
        cur = [0] * B
        t_pre = time.perf_counter()
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        fmax = self.mlx_state.forest_max
        ftok = max(1, self.mlx_state.forest_tokens)
        b = 0
        while b < B and lens[b] <= fmax:
            # a group: consecutive rows up to `ftok` tokens, the longest at most twice the shortest (the
            # attention pads the group to its longest row)
            e, n_tok = b, 0
            while e < B and lens[e] <= fmax and (e == b or (n_tok + lens[e] <= ftok and lens[e] <= 2 * lens[b])):
                n_tok += lens[e]
                e += 1
            toks, forest = self._mlx_forest(rows[b:e], offs[b:e])
            hm = self._mlx_embed_rows(m.array(toks, dtype=m.int32))
            if fp32:
                hm = hm.astype(m.float32)
            lg = self._forward_mlx(None, None, cache, None, False, True, self.L, hm=hm, lazy=True, forest=forest)
            # the first token off the prefill's logits in torch, as every loop's is (the same draw a row gets alone)
            first = smp.pick_torch(mlxdev.from_mx(lg), [smp.key_for(lens[bb] - 1) for bb in range(b, e)]).tolist()
            for j, bb in enumerate(range(b, e)):
                out[bb].append(int(first[j]))
                cur[bb] = int(first[j])
                done[bb] = int(first[j]) in eos
                for cl in layers:
                    cl._ns[bb] = lens[bb]
            b = e
        for bb in range(b, B):
            for cl in layers:
                cl.select_row(bb)
            lg = self._prefill(torch.tensor([rows[bb]], dtype=torch.long), cache)
            t = int(smp.pick_torch(lg[0, -1:], [smp.key_for(lens[bb] - 1)])[0])
            out[bb].append(t)
            cur[bb] = t
            done[bb] = t in eos
        for cl in layers:
            cl.select_row(None)
        t_dec = time.perf_counter()

        def build(tok: Any) -> Any:
            # the next step's graph from the previous step's unread pick: the GPU runs step t while Python builds
            # t + 1; row b's token lands at its own length, the row the pick is keyed by
            hm = self._mlx_embed_rows(tok)
            if fp32:
                hm = hm.astype(m.float32)
            ns = [int(p) for p in layers[0]._ns]
            lg = self._forward_mlx(None, None, cache, None, False, True, self.L, hm=hm, lazy=True, rows=ns)
            return smp.pick_mx(lg, [smp.key_for(p) for p in ns])

        steps = 0
        pending: Any = None
        if int(max_new) > 1:
            pending = build(m.array(cur, dtype=m.int32))
            m.async_eval(pending)
        for step in range(1, int(max_new)):
            if self.abort.is_set():
                m.eval(pending)
                break
            steps += 1
            ahead = build(pending) if step + 1 < int(max_new) else None
            if ahead is not None:
                m.async_eval(ahead)
            toks = pending.tolist()
            for b in range(B):
                if not done[b]:
                    out[b].append(toks[b])
                    done[b] = toks[b] in eos
            if all(done):
                break
            pending = ahead
        n_tok = sum(len(o) for o in out)
        dec = time.perf_counter() - t_dec
        self.log(
            f"[stream] generated {n_tok} tokens over {B} rows in {time.time() - t0:.1f}s (batched: "
            f"prefill {t_dec - t_pre:.2f}s over {tot} tokens ({b} rows as a forest), {steps} steps {dec:.2f}s = "
            f"{dec / max(1, steps) * 1e3:.1f} ms/step, {B * steps / max(1e-9, dec):.0f} tok/s decoding)"
        )
        # back in the caller's order
        back: list[Any] = [None] * B
        for j, b_ in enumerate(order):
            back[b_] = out[j]
        return back

    def _mlx_greedy_ok(self, B: int, attention_mask: torch.Tensor | None, on_layer: Any, prefill_only: bool) -> bool:
        return (
            self.mlx is not None
            and self.mlx_state.pipeline
            and B == 1
            and attention_mask is None
            and on_layer is None
            and not prefill_only
            and (self.fam.dense or self.fam.hybrid or self.fam.sandwich)
            and not self.cold
            and self.norm is not None
            and self.head_host is not None
            and self.head_host.mx is not None
            and self.head is None
            and self._mlx_ok(None, 1, 1, None, None, self.L)
        )

    def _generate_greedy_mlx(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: Any,
        on_token: Callable[[int], Any] | None,
        t0: float,
        session: Session | None = None,
        sampling: Any = None,
    ) -> tuple[list[int], dict[str, Any]]:
        """Plain decoding with one token of lookahead: step t+1's graph is built from step t's unread pick (the
        argmax, or the sample keyed by the row) and queued before t is read. A `session` reuses its cache when
        the prompt extends the tokens it holds."""
        m = mlxdev.mx()
        smp = sampling or GREEDY
        prompt = ids[0].tolist()
        cache, reuse, _anchored = session.open(self, prompt) if session is not None else (None, 0, None)
        if cache is None:
            cache = self.new_cache()
        logits, anchors = self._session_prefill(ids, cache, reuse, session)
        self.vram_trim("prefill")
        first = int(smp.pick_torch(logits[0, -1:], [smp.key_for(len(prompt) - 1)])[0])
        out = [first]
        census = {"forwards": 1, "reused": reuse, "prefill_s": round(time.time() - t0, 3)}

        def keep() -> tuple[list[int], dict[str, Any]]:
            if session is not None:
                session.keep(prompt, out, cache, anchors)
            census["seconds"] = time.time() - t0
            census["forwards"] = len(out)
            return out, census

        if on_token:
            on_token(first)
        if first in eos or max_new <= 1:
            return keep()
        if self._mega_ok(cache, 1, None, True, smp, None):
            # one dispatch a token: nothing to build ahead of the GPU
            cur_t = first
            for _step in range(1, max_new):
                cur_t = int(self.forward([[cur_t]], cache=cache, pick=smp)[0, 0])
                out.append(cur_t)
                if on_token:
                    on_token(cur_t)
                if cur_t in eos:
                    break
            return keep()
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16

        def build(tok: Any) -> Any:
            hm = self._mlx_embed_rows(tok)
            if fp32:
                hm = hm.astype(m.float32)
            pos = cache.get_seq_length()  # the row `tok` lands in, the pick's key
            lg = self._forward_mlx(None, None, cache, None, True, True, self.L, hm=hm, lazy=True)
            return smp.pick_mx(lg, [smp.key_for(pos)]).reshape(1)

        cur = build(m.array([first], dtype=m.int32))
        m.async_eval(cur)
        for step in range(1, max_new):
            if self.abort.is_set():
                m.eval(cur)  # the queued step lands (its rows are the last token's) and nothing stays in flight
                break
            ahead = step + 1 < max_new
            nxt: Any = None
            if ahead:
                nxt = build(cur)
                m.async_eval(nxt)
            t = int(cur.item())
            out.append(t)
            if on_token:
                on_token(t)
            if t in eos:
                if nxt is not None:
                    # the lookahead step is undone: its cache rows and its DeltaNet states
                    m.eval(nxt)
                    for cl in cache.layers:
                        if isinstance(cl, GrowLayer) and cl.shared and cl._mx is not None:
                            cl._n -= 1
                        if isinstance(cl, GraphStates) and cl._mx_pending is not None:
                            cl._mx_pending, cl._mx_prev = cl._mx_prev, None
                break
            cur = nxt
        self._mlx_flush_states(cache)
        self.log(
            f"[stream] generated {len(out)} tokens over 1 rows in {time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, len(out)):.1f} s/step incl. prefill; pipelined, reused {reuse})"
        )
        return keep()

    def _forward_mlx_hybrid(
        self,
        h: Any,
        pe: Any,
        cache: Any,
        on_layer: Any,
        last_only: bool,
        head: bool,
        n_layers: int,
        hm: Any = None,
        lazy: bool = False,
        pick: Any = None,
    ) -> Any:
        """The Qwen3.5 hybrid on the GPU: attention layers and MLPs as MLX graphs, the DeltaNet as one dispatch per
        layer (or per position on the CPU kernel with per-node checkpoints under speculation); a tree attends
        through one mask. `lazy` keeps the new DeltaNet states in the graph."""
        self._tag(PassTag.MLX_HYBRID)
        m = mlxdev.mx()
        be = self.mlx
        assert be is not None
        c = self.cfg
        if hm is None:
            T = int(h.shape[1])
            dt = h.dtype
            hm = mlxdev.to_mx(h[0])
        else:
            T = int(hm.shape[0])
            dt = mlxdev.torch_dtype(hm.dtype)
        act = self._mlx_act()
        assert act is not None  # _mlx_ok admits the family only with an MLX activation
        silu = act_name(c) in ("silu", "swish")
        past = cache.get_seq_length() if cache is not None else 0
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        spec_on = bool(getattr(self, "aq", False))
        parents: Any = getattr(self, "ap", None) if spec_on else None
        tree = parents is not None and any(parents[j] != j - 1 for j in range(T))
        freqs, rd, rscale = self._mlx_rope()
        assert not isinstance(freqs, dict)  # the hybrid families (Qwen3.5) carry a single rope, never Gemma's dual
        tree_prep, tree_pos = self._mlx_tree_prep(cache, past, parents) if tree else (None, [])
        keys = _pick_keys(pick, tree_pos if tree else range(past, past + T), last_only)
        # as the host path: after a prefix, positions step one at a time through the kernel (a chain, a tree, a
        # short continuation) - up to the 16 rows the node kernels take; a prefill from scratch, a chunked one and
        # any longer continuation (a session's next turn) go through the module's chunked rule from the stored
        # states. Past 16 rows the node step would build a checkpoint per row per layer, which for a long prompt
        # is tens of GB before the kernel refuses it
        step_nodes = (
            cache is not None
            and past > 0
            and T <= 16
            and (spec_on or T == 1 or not getattr(self, "_batched_cont", False))
        )
        tree_mask = None
        if tree:
            allow = torch.zeros(T, past + T, dtype=torch.bool)
            allow[:, :past] = True
            for p in range(T):
                allow[p, past + p] = True
                q_ = parents[p]
                while q_ >= 0:
                    allow[p, past + q_] = True
                    q_ = parents[q_]
            tree_mask = mlxdev.to_mx(allow)[None, None]
        if self.cold:
            self._cold_start(n_layers)
        t0 = time.time()
        attn_pa: _AttnParams | None = None
        pending: list[Any] = []
        pend_inplace: set[int] = set()
        fresh: list[Any] = []
        ckpts: list[Any] = []
        taps = []

        def write_back() -> None:
            for cl_, cn, rn in pending:
                c_, r_ = self._lin(cl_)
                c_[0].copy_(mlxdev.from_mx(cn))
                if id(cl_) not in pend_inplace:
                    r_[0].copy_(mlxdev.from_mx(rn))
            for cl_, cn, rn, K_ in fresh:
                # through the cache's own setters: they initialize a fresh layer and copy into a live one
                cl_.update_conv_state(mlxdev.from_mx(cn)[None].clone(), 0, conv_kernel_size=K_)
                cl_.update_recurrent_state(mlxdev.from_mx(rn)[None].clone())

        for i in range(n_layers):
            tmpl = self.host[i]
            lt = self.layer_types[i]
            if i in self.cold:
                self._cold_wait(i)
            w = self._mlx_consts(i, tmpl, dt)
            x = self._mlx_norm(hm, w["ln1"], w["eps1"])
            if lt == LayerKind.FULL:
                at = tmpl.self_attn
                hd = int(at.head_dim)
                qkv_w = getattr(at, "_mx_qkv", None)
                if qkv_w is not None:
                    # q (with its gate), k and v as one matvec: the weights one stream, the rows' bits the separate
                    # matvecs' (a row's chain runs over K alone)
                    qkv = be.matmul(x, qkv_w)
                    nq, nk = Hq * 2 * hd, Hk * hd
                    qg, kk, vv = qkv[:, :nq], qkv[:, nq : nq + nk], qkv[:, nq + nk :]
                else:
                    qg, kk, vv = be.matmul(x, at.q_proj.mx), be.matmul(x, at.k_proj.mx), be.matmul(x, at.v_proj.mx)
                q, gate = m.split(qg.reshape(T, Hq, 2 * hd), 2, axis=-1)
                gate = gate.reshape(T, Hq * hd)
                q = self._mlx_norm(q, w["qn"], w["epsq"])
                k = self._mlx_norm(kk.reshape(T, Hk, hd), w["kn"], w["epsq"])
                v = vv.reshape(T, Hk, hd)
                if tree:
                    # every node rotated at past + depth in one launch for q and k, the one-row step's bits (a table
                    # rotation would round the K rows differently)
                    qr, kr = mlxdev.rope_rows2(q, k, rd, freqs, rscale, tree_pos)
                    qh, kh = qr.transpose(1, 0, 2)[None], kr.transpose(1, 0, 2)[None]
                else:
                    qh = be.rope_fast(q.transpose(1, 0, 2), rd, freqs, rscale, past)[None]
                    kh = be.rope_fast(k.transpose(1, 0, 2), rd, freqs, rscale, past)[None]
                vh = v.transpose(1, 0, 2)[None]
                cl, K, V = self._mlx_cache(cache, i, kh, vh)
                a, attn_pa = self._mlx_attend(
                    qh,
                    K,
                    V,
                    cl,
                    T,
                    hd,
                    Hq,
                    Hk,
                    float(at.scaling),
                    tree_mask if tree else ("causal" if T > 1 else None),
                    attn_pa,
                    prepared=tree_prep,
                    nodes=(past, parents if parents is not None else list(range(-1, T - 1)))
                    if (step_nodes and spec_on)
                    else None,
                )
                a = a * m.sigmoid(gate)
                hm = hm + be.matmul(a, at.o_proj.mx)
            else:
                la = tmpl.linear_attn
                if step_nodes and T == 1 and not spec_on and getattr(self, "mlx_delta", True):
                    # the plain decode position in MLX ops: the states stay in the graph, no round trip
                    mixed, z, b, a_ = (t[0].astype(m.float32) for t in self._mlx_delta_proj(la, x))
                    conv_mx, rec_mx = self._mlx_lin_adopt(cache.layers[i])
                    if self.mlx_state.delta_mode == "recurrent" and w["delta"]["dk"] % 32 == 0:
                        # the same Metal dispatch as the prefill, over one position
                        d = w["delta"]
                        # the state written into the cache's buffer by the kernel unless the graph carries it
                        core, conv_new, rec_new = mlxdev.delta_prefill(
                            mixed[None],
                            z[None],
                            a_[None],
                            b[None],
                            conv_mx[0][:, 1:],
                            rec_mx,
                            d["conv_w"],
                            d["conv_b"],
                            d["a_log"],
                            d["dt_bias"],
                            d["norm_w"],
                            d["eps"],
                            d["hk"],
                            d["hv"],
                            d["dk"],
                            d["dv"],
                            d["key_dim"],
                            inplace=not lazy,
                        )
                        core = core[0]
                    else:
                        core, conv_new, rec_new = self._mlx_delta_step(w, mixed, z, a_, b, conv_mx, rec_mx)
                    pending.append((cache.layers[i], conv_new, rec_new))
                    if rec_new is None:
                        pend_inplace.add(id(cache.layers[i]))
                    hm = hm + be.matmul(core.astype(hm.dtype)[None], la.out_proj.mx)
                elif step_nodes and getattr(self, "mlx_delta", True) and w["delta"]["dk"] % 32 == 0:
                    # a chain or tree of nodes in one dispatch per layer: each node's conv reads its ancestry, its
                    # rule starts from its parent's checkpoint; under speculation the checkpoints stay for the commit
                    d = w["delta"]
                    cl = cache.layers[i]
                    mixed, z, b, a_ = (t.astype(m.float32) for t in self._mlx_delta_proj(la, x))
                    conv_mx, rec_mx = self._mlx_lin_adopt(cl)
                    core, conv_ck, heads = mlxdev.delta_tree_step(
                        mixed,
                        z,
                        a_,
                        b,
                        conv_mx[0][:, 1:],
                        rec_mx,
                        parents,
                        d["conv_w"],
                        d["conv_b"],
                        d["a_log"],
                        d["dt_bias"],
                        d["norm_w"],
                        d["eps"],
                        d["hk"],
                        d["hv"],
                        d["dk"],
                        d["dv"],
                        d["key_dim"],
                    )
                    if spec_on:
                        ckpts.extend((conv_ck, *heads))
                        self.al[i] = _MxCheckpoints(conv_ck, heads, rec_mx)
                        self.am[i] = self._lin(cl)
                    elif lazy:
                        pending.append((cl, conv_ck[T - 1], mlxdev.delta_chain_state(heads, rec_mx)))
                    else:
                        pending.append((cl, conv_ck[T - 1], mlxdev.delta_chain_state(heads, rec_mx, inplace=True)))
                        pend_inplace.add(id(cl))
                    hm = hm + be.matmul(core.astype(hm.dtype), la.out_proj.mx)
                elif step_nodes:
                    mixed, z, b, a_ = self._mlx_delta_proj(la, x)
                    m.eval(mixed, z, b, a_)
                    f32 = lambda t: mlxdev.from_mx(t).float()[None]
                    core = self._delta_nodes(
                        tmpl, i, cache.layers[i], f32(mixed), f32(z), f32(a_), f32(b), T, parents, spec_on
                    )
                    hm = hm + be.matmul(mlxdev.to_mx(core[0].to(dt).contiguous()), la.out_proj.mx)
                elif cache is not None and getattr(self, "mlx_delta_prefill", True):
                    # a prefill in MLX: the chunked rule from the cache's states (fresh, or the stored ones for a
                    # chunked continuation); the new states are written into the cache after the eval
                    d = w["delta"]
                    cl = cache.layers[i]
                    mixed, z, b, a_ = (t.astype(m.float32) for t in self._mlx_delta_proj(la, x))
                    kw_ = int(d["conv_w"].shape[1])
                    if past > 0:
                        conv_mx, rec_mx = self._mlx_lin_adopt(cl)
                        conv_prev, state0 = conv_mx[0], rec_mx[0]
                    else:
                        conv_prev, state0 = m.zeros((int(d["conv_w"].shape[0]), kw_ - 1), dtype=m.float32), None
                    core, conv_new, rec_new = mlxdev.delta_prefill(
                        mixed,
                        z,
                        a_,
                        b,
                        conv_prev,
                        state0,
                        d["conv_w"],
                        d["conv_b"],
                        d["a_log"],
                        d["dt_bias"],
                        d["norm_w"],
                        d["eps"],
                        d["hk"],
                        d["hv"],
                        d["dk"],
                        d["dv"],
                        d["key_dim"],
                        mode=self.mlx_state.delta_mode,
                    )
                    fresh.append((cl, conv_new, rec_new, kw_))
                    hm = hm + be.matmul(core.astype(hm.dtype), la.out_proj.mx)
                else:
                    # the module's projections on the GPU (bf16), its conv and chunked rule on the CPU in
                    # float32 (they widen on their own); the states it leaves are widened by the step paths
                    m.eval(x)
                    mix = la(mlxdev.from_mx(x)[None], cache_params=cache, attention_mask=None)
                    hm = hm + mlxdev.to_mx(mix[0].to(dt).contiguous())
            x2 = self._mlx_norm(hm, w["ln2"], w["eps2"])
            mlp = tmpl.mlp
            gu_w = getattr(mlp, "_mx_gu", None)
            if gu_w is not None and silu:
                mid = fk.silu_mul(be.matmul(x2, gu_w))
            else:
                mid = act(be.matmul(x2, mlp.gate_proj.mx)) * be.matmul(x2, mlp.up_proj.mx)
            hm = hm + be.matmul(mid, mlp.down_proj.mx)
            if i in self.cold:
                m.eval(hm)
                self._cold_release(i)
            if on_layer is not None:
                # handed over after the one eval at the end: an eval here would sync the GPU every layer
                taps.append((i, hm))
        if lazy and not fresh:
            for cl_, cn, rn in pending:
                # kept with the batch axis, the shape the cache's tensors and `_mlx_lin_adopt` use
                cl_._mx_prev = getattr(cl_, "_mx_pending", None)
                cl_._mx_pending = (cn[None], rn[None])
            return self._mlx_finish(hm, last_only, head, n_layers, t0, lazy=True)
        extra = [a for _, cn, rn in pending for a in (cn, rn) if a is not None]
        extra += [a for _, cn, rn, _ in fresh for a in (cn, rn)]

        def after() -> None:
            if extra:
                write_back()
            for i_, h_ in taps:
                on_layer(i_, mlxdev.from_mx(h_)[None])

        return self._mlx_finish(
            hm,
            last_only,
            head,
            n_layers,
            t0,
            extra=extra + ckpts + [h_ for _, h_ in taps],
            after=after if (extra or taps) else None,
            pick=pick,
            keys=keys,
        )
