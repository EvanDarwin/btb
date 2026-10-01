# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's card program: its resident layers' one-token step and speculative verify pass (a chain or a tree of up to
32 rows) through btb's row-invariant card kernels (native/cuda/btb_kernels.cu), so every node of a verify pass
computes bit for bit as the one-token step of its path - the step and the pass run the same kernels, and no kernel's
arithmetic for a row depends on the rows beside it.

A pass is one captured graph a layer and one for the tail, replayed in turn by the engine's runner (cuda.py
`_forward_card_program`), keyed by (layer, rows, mode, taps) and captured on first use; the rows are padded to the
gemv's widths (1, 2, 4, 8, 16, 32), the padding rows computed and never read. Graph i runs from the previous layer's
mixture close - the routed experts' rows back from the host, the shared expert's gated add, the write into the
streams - through layer i's hyper-connection read, its mixer (the sparse attention or the gated DeltaNet, the
per-layer n-gram embedding before it where the layer has one), the MLP's read, and the router, whose picks and whose
rows it publishes to pinned host memory (`btb_moe_route`'s flag); the shared expert runs while the host serves the
routed experts through the expert store (`between`), whose sum comes back through pinned memory at the next graph's
start. The tail closes the last layer's mixture and runs the final mixer and the head.

The program runs the model as it is placed: each run of consecutive resident layers is a segment of its graphs, and a
layer on the host between them runs the host path (the engine's `run_layer`, as the torch path's pass runs it). A
segment's first graph starts from the streams as they come in - no mixture of the layer before it to close - and a
segment a host layer follows ends in a close (`close_body`: its last layer's mixture into the streams). The streams cross
the edge through pinned rows (`leave`, `enter`): the program's bf16 rows [T, G H] to the host's float32 and the host
layer's float32 back to bf16, each row cast as the torch path's crossing casts it. A host layer takes the step and the
verify pass through the same host code, so a node there computes as its path's steps as well; its states, rows and
commits are its own cache layer's, as on the torch path.

`step` (a one-row decode) writes the cache as it goes: the key and value rows at their slots, the DeltaNet's states
and conv window, the n-gram embedding's kept window. `tree` (a verify pass) writes only rows past the committed ones
- each node's key, value and raw indexer key at its own slot - and leaves every state as it was: each DeltaNet node
steps its parent's state in scratch slots, each node's n-gram window is read along its path. The commit (`ad` runs
`commit` with the accepted path) moves the path's rows into place (`ArenaIndexedLayer.keep_path`), re-steps the
DeltaNet states along the path from the pass's own projections (one captured graph), and shifts the path's n-gram
inputs into the kept window, so the cache stands where the path's one-token steps would have left it.

The program's buffers are its own and fixed, so a graph captured once serves every pass: the attention's rows in one
arena (keys, values and the indexer's raw keys a sparse layer, its pooled keys caught up at each pass), the states
the cache's linear layers hold (bound into the cache: its torch paths write the same tensors), one set of pass
buffers sized for 32 rows. Every one is asked of the scheduler before it is made, and let go at the engine's close.
Where the program cannot run a model or a pass (`why_not`), the engine takes its torch path."""

from __future__ import annotations

import ctypes
import math
import sys
import time
import types
import weakref
from typing import Any

import torch

from ....kinds import LayerKind
from ...cache import ArenaIndexedLayer, indexer_keys
from ...forward import path_of
from ...host import _Experts
from ...hostmem import pinned
from ...native import Native, _Cuda
from ...scheduler import MemoryGrantError
from .verify import _slots

STEP, TREE = "step", "tree"
ROWS = 32  # the widest pass: the gemvs' widest M and the kernels' ancestor walks
# the kernels every model the program runs needs (the head-width ones are named for its dims in `why_not`)
_KERNELS = (
    "btb_hc_rmsnorm",
    "btb_hc_act",
    "btb_hc_mix",
    "btb_moe_route",
    "btb_moe_combine",
    "btb_gemv_sgate_bf16_m1",
    "btb_delta_nodes_bf16",
    "btb_conv_window",
    "btb_ple_gate",
    "btb_ple_conv",
)


class _NGram:
    """A PLE layer's n-gram embedding rows for a pass's nodes, made on the host: the reference module's hashing
    (`Qwen4ExpTextNGramEmbedding.forward`, the module's own code) over host copies of its tables - the ids are
    integers, so the host's are the card's - its rows through the engine's row table (`_NGramRows`), as the torch
    path's step (verify.py `_ple_forward`) makes them."""

    def __init__(self, emb: Any) -> None:
        self._forward = type(emb).forward
        self.ngram_size = int(emb.ngram_size)
        self.context_len = int(emb.context_len)
        self.heads_per_ngram = int(emb.heads_per_ngram)
        self.eos_token_id = int(emb.eos_token_id)
        self.layer_multipliers = emb.layer_multipliers.detach().cpu()
        self.ngram_heads_vocab_sizes = emb.ngram_heads_vocab_sizes.detach().cpu()
        self.ngram_heads_offsets = emb.ngram_heads_offsets.detach().cpu()
        self.ngram_embedding = emb.ngram_embedding
        self._shift_right_ignore_eos = types.MethodType(type(emb)._shift_right_ignore_eos, self)

    def rows(self, hist: torch.Tensor) -> torch.Tensor:
        """[T, E] the embedding of each node's last position over its last `ngram_size` ids `hist` [T, n]"""
        return self._forward(self, hist, None)[:, -1]


class Qwen4Card:
    """The card program of one engine (held weakly: the engine owns the program). `ok` says whether it runs the
    engine's model as placed now; the engine's runner then drives a pass through `begin`, the layer graphs
    (`layer_body`, `tail_body`) with `between` after each layer, and `end`."""

    ROWS = ROWS
    ARENA_MIN = 4096  # rows the arena starts at
    SPIN_S = 60.0  # the longest the host waits for a layer's router to publish

    def __init__(self, sm: Any) -> None:
        self._sm = weakref.ref(sm)
        cfg = sm.cfg
        self.L = int(sm.L)
        self.types = list(sm.layer_types)
        ple_ids = list(getattr(cfg, "ple_layer_ids", None) or [])
        self.ple_all = [i for i in range(self.L) if (i + 1) in ple_ids]
        # the layers as placed (`_layout`): the resident ones by kind, the segments they run in, the host's
        self.res: tuple[int, ...] | None = None
        self.at: set[int] = set()
        self.segs: list[tuple[int, int]] = []
        self.host_layers: list[int] = []
        self.sparse: list[int] = []
        self.linear: list[int] = []
        self.others: list[int] = []
        self.ple: list[int] = []
        self.sj: dict[int, int] = {}
        self.lj: dict[int, int] = {}
        self.pj: dict[int, int] = {}
        g = lambda k, d=0: int(getattr(cfg, k, None) or d)
        self.H, self.G, self.R = g("hidden_size"), g("hc_count"), g("hc_lowrank")
        self.E, self.topk, self.Is = g("num_experts"), g("num_experts_per_tok"), g("shared_expert_intermediate_size")
        self.Hq, self.Hk = g("num_attention_heads"), g("num_key_value_heads")
        self.D = g("head_dim") or self.H // max(1, self.Hq)
        self.Hi, self.di, self.r = g("indexer_n_heads"), g("indexer_head_dim"), g("indexer_compress_ratio", 1)
        self.ktop = g("indexer_budget") // max(1, self.r)
        self.hk, self.hv = g("linear_num_key_heads"), g("linear_num_value_heads")
        self.dk, self.dv, self.Kc = g("linear_key_head_dim"), g("linear_value_head_dim"), g("linear_conv_kernel_dim")
        self.C = 2 * self.hk * self.dk + self.hv * self.dv
        self.ps = self.C + self.hv * self.dv + 2 * self.hv  # the merged DeltaNet projection: q|k|v, z, b, a
        self.off = (0, self.C, self.C + self.hv * self.dv, self.C + self.hv * self.dv + self.hv)
        self.Ep, self.Kp, self.dil = g("ple_embed_dim", self.H), g("ple_conv_kernel_size"), g("ngram_size")
        self.Lp = max(0, (self.Kp - 1) * self.dil)
        self.n_ids = max(0, self.dil - 1)
        self.eps = float(cfg.rms_norm_eps)
        # the merged attention projection's columns: q interleaved with its gate, k, v, the indexer's q heads, its key
        self.ko = 2 * self.Hq * self.D
        self.vo = self.ko + self.Hk * self.D
        self.iq0 = self.vo + self.Hk * self.D
        self.ik0 = self.iq0 + self.Hi * self.di
        self.width = self.ik0 + self.di
        self.pool_new = ROWS // max(1, self.r) + 2  # the blocks a pass's graph pools: one pass's committed growth
        self.k: Any = None
        self.version: Any = None
        self._why: str | None = "not checked"
        self.W: dict[Any, dict[str, Any]] = {}
        self.B: dict[str, Any] | None = None  # the pass buffers
        self.S: dict[str, list[torch.Tensor]] | None = None  # the states bound into the cache
        self.A: dict[str, Any] | None = None  # the arena
        self.X: dict[str, torch.Tensor] | None = None  # the edge's pinned rows, where host layers run between segments
        self.P: dict[str, torch.Tensor] | None = None  # a torch-path slice's tree over the rows in RAM (`attend_rows`)
        self.held: dict[str, int] = {}  # the bytes granted, by what they hold
        self.graphs: dict[Any, torch.cuda.CUDAGraph] = {}
        self.pool: Any = None
        self.stream: torch.cuda.Stream | None = None
        self.owner: weakref.ref[Any] | None = None
        self.bound: list[weakref.ref[ArenaIndexedLayer] | None] = []
        self.pooled: list[int] = []
        self.ctx: list[list[int] | None] = []
        self.seen: int | None = None  # the cache's length after the program's own last write
        # the pass in flight
        self.T = self.M = self.n0 = 0
        self.mode = STEP
        self.tap = False
        self.ids: list[int] = []
        self.parents: list[int] = []
        self.staged = False
        self.clean = True  # the last pass ran to its end: no publish of it is still to land
        self.passes = 0  # passes begun: a commit is its own pass's, never a later one's
        self.to_commit = -1

    # -- whether it runs --------------------------------------------------------------------------------------

    def ok(self) -> bool:
        """whether the program runs the engine's model as placed now (its weights read afresh for a new
        placement: a layer that moved is another module)"""
        sm = self._sm()
        if sm is None:
            return False
        place = sm.device.snapshot()
        if place.version != self.version:
            self._reset_weights()
            try:
                self._layout(place.resident)
            except MemoryGrantError as e:
                # the bound cache's rows and states, copied out of the buffers the old layout sized, refused: the
                # program declines this placement (its passes take the torch path) with the old buffers kept, so a
                # layer still attached reads rows that stand, and the next look lays the placement out again
                self._why = f"the bound cache's rows could not be copied out for the new placement: {e}"
                self.version = None
                sm.log(f"[card] Qwen4's card program off for this placement: {self._why}")
                return False
            self.version = place.version
            self._why = self.why_not(sm)
        if self._why is None and self.host_layers and self.X is None:
            # the edge's rows before the pass, not at its first crossing: the ledger refusing them here declines the
            # program for this placement (its passes take the torch path) instead of failing a pass part way
            try:
                self.edge()
            except MemoryGrantError as e:
                self._why = f"the edge's rows refused: {e}"
                sm.log(f"[card] Qwen4's card program off for this placement: {self._why}")
        return self._why is None

    def _layout(self, resident: tuple[int, ...]) -> None:
        """the program's layers as placed: the resident ones by kind and the segments they run in, the rest the
        host's. A placement that moves a layer on or off the card lets go of every buffer sized by the resident
        layers - the cache bound to them keeping copies of its own - and they are made again for the new one"""
        res = tuple(sorted(int(i) for i in resident))
        if res == self.res:
            return
        owner = self.owner() if self.owner is not None else None
        if owner is not None:
            self._evict(owner)
        self._drop_graphs()
        self.B = self.S = self.A = self.X = self.P = None
        self.owner = None
        for what in [w for w in self.held if not w.startswith("layer ")]:
            del self.held[what]
        self.res = res
        self.at = set(res)
        self.host_layers = [i for i in range(self.L) if i not in self.at]
        self.segs = []
        for i in res:
            if self.segs and self.segs[-1][1] == i:
                self.segs[-1] = (self.segs[-1][0], i + 1)
            else:
                self.segs.append((i, i + 1))
        self.sparse = [i for i in res if self.types[i] == LayerKind.QWEN_SPARSE]
        self.linear = [i for i in res if self.types[i] == LayerKind.LINEAR]
        self.others = [i for i in res if self.types[i] not in (LayerKind.QWEN_SPARSE, LayerKind.LINEAR)]
        self.ple = [i for i in self.ple_all if i in self.at]
        self.sj = {i: j for j, i in enumerate(self.sparse)}
        self.lj = {i: j for j, i in enumerate(self.linear)}
        self.pj = {i: j for j, i in enumerate(self.ple)}
        self.bound = [None] * len(self.sparse)
        self.pooled = [0] * len(self.sparse)
        self.ctx = [None] * len(self.ple)
        self.seen = None

    def seg_end(self, a: int) -> int:
        """the end of the segment that starts at resident layer `a`"""
        return next(b for s, b in self.segs if s == a)

    def why_not(self, sm: Any) -> str | None:
        """what keeps the program off this engine's model, or None: the card and its kernels, a bf16 model the card
        holds the head of and some layers of - the rest on the host, none streamed through the card - and shapes the
        kernels are written for"""
        from .. import act_name

        k = Native.card_kernels()
        if sm.dev.type != "cuda" or k is None:
            return "no card kernels"
        missing = [n for n in _KERNELS if n not in k.fn]
        if missing:
            return f"the kernels lack {missing[0]}"
        if sm.compute_dtype not in (None, torch.bfloat16) or getattr(sm, "resident_fp32", False) or sm.shadow:
            return "a float32 compute"
        if sm.mlx is not None:
            return "not the card's tier"
        if not self.res:
            return "no layer on the card"
        streamed = [i for i in self.host_layers if i not in sm.host]
        if streamed:
            return f"layer {streamed[0]} streamed through the card, neither resident nor on the host"
        if self.others:
            return "a layer type the program has no body for"
        head = sm.head
        if head is None or head.weight.device.type != "cuda" or head.weight.dtype != torch.bfloat16:
            return "the head not on the card in bf16"
        mixer = sm.mixer
        if mixer is None or any(p.device.type != "cuda" or p.dtype != torch.bfloat16 for p in mixer.parameters()):
            return "the final mixer not on the card in bf16"
        cfg = sm.cfg
        if act_name(cfg) != "silu" or not bool(getattr(cfg, "norm_topk_prob", True)):
            return "an activation or a routing the kernels are not written for"
        rot = self._rot(sm)
        H, G, R, D, di = self.H, self.G, self.R, self.D, self.di

        def rope_ok(d: int) -> bool:
            e = d // 32
            half = rot // (2 * e) if e else 0
            return 0 < rot <= d and rot % (2 * e) == 0 and half > 0 and not half & (half - 1)

        shapes = [
            H % 8 == 0 and (G * H) % 8 == 0 and R % 8 == 0 and R > 0 and G > 1,
            self.Is % 8 == 0 and 0 < self.topk <= _Cuda.ROUTE_MAX_K and self.E * 4 <= 48 * 1024,
        ]
        if self.sparse:
            shapes += [
                D in (128, 256) and di in (128, 256) and rope_ok(D) and rope_ok(di),
                self.Hk > 0 and self.Hq % self.Hk == 0 and (self.Hq * D) % 8 == 0 and self.Hi > 0 and self.ktop > 0,
                (self.Hi + 33) * di * 2 + 2048 <= 48 * 1024,
            ]
        if self.linear:
            shapes += [0 < self.Kc <= 8 and self.hk > 0 and self.hv % self.hk == 0 and (self.hv * self.dv) % 8 == 0]
        if self.ple:
            shapes += [self.Ep % 8 == 0 and 0 < self.Lp < 32 and all(i in self.lj for i in self.ple)]
        if not all(shapes):
            return "shapes the kernels are not written for"
        if self.sparse:
            names = [f"btb_norm_rope_part_d{d}" for d in {D, di}]
            names += [f"btb_qsa_pool_d{di}", f"btb_qsa_select_d{di}", f"btb_qsa_attn_split_d{D}"]
            missing = [n for n in names if n not in k.fn]
            if missing:
                return f"the kernels lack {missing[0]}"
        for i in self.res:
            layer = sm.resident[i]
            if not isinstance(getattr(layer.mlp, "experts", None), _Experts):
                return "a mixture not served through the store"
            if (layer.ple is not None) != (i in self.pj):
                return "a PLE layer the config does not name"
        return None

    def _rot(self, sm: Any) -> int:
        """the rope's rotated dims (the table's width)"""
        x = torch.zeros(1, 1, self.H, dtype=torch.bfloat16, device=sm.dev)
        cos, _sin = sm.rotary(x, torch.zeros(3, 1, 1, dtype=torch.long, device=sm.dev))
        return int(cos.shape[-1])

    # -- memory -----------------------------------------------------------------------------------------------

    def _grant(self, what: str, nbytes: int, kind: str, device: Any, held: int = 0, **kw: Any) -> None:
        sm = self._sm()
        sched = getattr(sm, "scheduler", None) if sm is not None else None
        if sched is not None and nbytes:
            sched.grant(int(nbytes), kind, requester=f"Qwen4's card program: {what}", device=device, held=held, **kw)
        self.held[what] = self.held.get(what, 0) + int(nbytes) - int(held)

    def _alloc(
        self,
        what: str,
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
        kind: str,
        pinned: bool = False,
        reclaim: bool = False,
    ) -> dict[str, torch.Tensor]:
        """zeroed tensors of `specs` {name: (shape, dtype)} on the card (or pinned in RAM), asked of the scheduler
        together before any is made; `reclaim` lets a host request take room back from the expert store (a caller
        outside any store call: see `BatchScheduler.grant`)"""
        sm = self._sm()
        assert sm is not None
        dev = torch.device("cpu") if pinned else sm.dev
        nbytes = sum(math.prod(s) * torch.empty(0, dtype=dt).element_size() for s, dt in specs.values())
        self._grant(what, nbytes, kind, dev, reclaim=reclaim)
        return {n: torch.zeros(s, dtype=dt, device=dev, pin_memory=pinned) for n, (s, dt) in specs.items()}

    def held_bytes(self) -> int:
        """the bytes the program holds now, as allocated (the test's check against what it was granted)"""
        n = 0
        seen: set[int] = set()

        def add(t: Any) -> None:
            nonlocal n
            if isinstance(t, torch.Tensor):
                key = t.untyped_storage().data_ptr()
                if key not in seen:
                    seen.add(key)
                    n += t.untyped_storage().nbytes()
            elif isinstance(t, dict):
                for v in t.values():
                    add(v)
            elif isinstance(t, (list, tuple)):
                for v in t:
                    add(v)

        add(self.B)
        add(self.S)
        add(self.A)
        add(self.X)
        add(self.P)
        for W in self.W.values():
            add(W.get("_own", []))
        return n

    # -- the weights ------------------------------------------------------------------------------------------

    def let_go(self) -> None:
        """the merged blocks, their float32 operands and the graphs let go now - what a new placement's `ok()` does
        on the next pass - the arena and a bound cache's rows kept: a layer the engine sheds meanwhile frees its
        blocks on the card at once (the modules only view them; held here, a shed freed nothing and a yield gave up
        every layer and the head for one cut of the budget). Read again, lazily, as the next pass asks"""
        self._reset_weights()

    def _reset_weights(self) -> None:
        self.W.clear()
        # the operands the weights' reading made go with them (a merged block is the modules' own, granted as moved)
        for what in [w for w in self.held if w.endswith("operands in float32")]:
            del self.held[what]
        self._drop_graphs()

    def _merge(self, what: str, parts: list[tuple[Any, str]]) -> torch.Tensor:
        """the modules' matrices `parts` [(module, name)] as one row block on the card, the modules keeping views of
        it (a block merged before - a placement's second reading - is taken as it lies)"""
        sm = self._sm()
        assert sm is not None
        ts = [getattr(m, n) for m, n in parts]
        for t in ts:
            if t.dtype != torch.bfloat16 or t.device.type != "cuda" or t.dim() != 2:
                raise RuntimeError(
                    f"[card] {what}: the program takes bf16 matrices on the card, got {t.dtype} {t.device}"
                )
        rows = sum(int(t.shape[0]) for t in ts)
        cols = int(ts[0].shape[1])
        st = ts[0].untyped_storage()
        at = ts[0].storage_offset()
        whole = all(t.is_contiguous() and int(t.shape[1]) == cols for t in ts)
        for t in ts:
            whole = whole and t.untyped_storage().data_ptr() == st.data_ptr() and t.storage_offset() == at
            at += t.numel()
        if whole:
            return torch.empty(0, dtype=torch.bfloat16, device=ts[0].device).set_(
                st, ts[0].storage_offset(), (rows, cols), (cols, 1)
            )
        nbytes = rows * cols * 2
        self._grant(f"{what}, merged", nbytes, "weights", sm.dev, held=nbytes)
        W = torch.cat([t.detach() for t in ts], 0).contiguous()
        a = 0
        for (m, n), t in zip(parts, ts, strict=True):
            b = a + int(t.shape[0])
            sm._set_param(m, n, W[a:b])
            a = b
        return W

    @staticmethod
    def _bf(t: torch.Tensor, what: str) -> torch.Tensor:
        if t.dtype != torch.bfloat16 or t.device.type != "cuda" or not t.is_contiguous():
            raise RuntimeError(f"[card] {what}: the program takes a contiguous bf16 tensor on the card")
        return t

    def _hc(self, mod: Any, what: str) -> dict[str, torch.Tensor]:
        """a hyper-connection's weights: its norm, the [down | inject] block the gemv reads in one, and up"""
        inject = getattr(mod, "block_inject_weight", None)
        if inject is not None:
            dn = self._merge(what, [(mod.input_mix_weight_down, "weight"), (inject, "weight")])
        else:
            dn = self._bf(mod.input_mix_weight_down.weight, what)
        return {
            "norm": self._bf(mod.hc_norm.weight, what),
            "dn": dn,
            "up": self._bf(mod.input_mix_weight_up.weight, what),
        }

    def weights(self, i: int) -> dict[str, Any]:
        """layer `i`'s weights as the kernels take them (the tail's for `i` == L): the projections a gemv reads
        together merged into one row block, the modules keeping views of it; the DeltaNet's float32 operands"""
        W = self.W.get(i)
        if W is not None:
            return W
        sm = self._sm()
        assert sm is not None
        if i == self.L:
            assert sm.head is not None and sm.mixer is not None
            W = {"hc": self._hc(sm.mixer, "the final mixer"), "head": self._bf(sm.head.weight, "the head")}
            self.W[i] = W
            return W
        layer = sm.resident[i]
        mlp = layer.mlp
        se = mlp.shared_expert
        W = {
            "hcA": self._hc(layer.attn_hyper_connection, f"layer {i}'s attention hyper-connection"),
            "hcM": self._hc(layer.mlp_hyper_connection, f"layer {i}'s MLP hyper-connection"),
            "router": self._merge(f"layer {i}'s router", [(mlp.gate, "weight"), (mlp.shared_expert_gate, "weight")]),
            "sgu": self._merge(f"layer {i}'s shared expert", [(se.gate_proj, "weight"), (se.up_proj, "weight")]),
            "sdn": self._bf(se.down_proj.weight, f"layer {i}'s shared expert"),
        }
        # the store's lookahead reads each layer's router off its module: its cached weight is the one merged away
        store = sm.expert_store
        if store is not None:
            getattr(store, "_routers", {}).pop(i, None)
        if i in self.sj:
            at = layer.self_attn
            ix = at.indexer
            W["qkv"] = self._merge(
                f"layer {i}'s attention",
                [(at.q_proj, "weight"), (at.k_proj, "weight"), (at.v_proj, "weight"), (ix.index_qk_proj, "weight")],
            )
            W["o"] = self._bf(at.o_proj.weight, f"layer {i}'s o_proj")
            W["wq"], W["wk"] = self._bf(at.q_norm.weight, "q_norm"), self._bf(at.k_norm.weight, "k_norm")
            W["iq"] = self._bf(ix.q_layernorm.weight, "the indexer's q norm")
            W["ik"] = self._bf(ix.k_layernorm.weight, "the indexer's k norm")
            W["scale"] = float(at.scaling)
        else:
            la = layer.linear_attn
            W["proj"] = self._merge(
                f"layer {i}'s DeltaNet",
                [
                    (la.in_proj_qkv, "weight"),
                    (la.in_proj_z, "weight"),
                    (la.in_proj_b, "weight"),
                    (la.in_proj_a, "weight"),
                ],
            )
            W["out"] = self._bf(la.out_proj.weight, f"layer {i}'s out_proj")
            f32 = lambda t: None if t is None else t.detach().to(sm.dev, torch.float32).contiguous()
            parts = [la.conv1d.weight.squeeze(1), la.conv1d.bias, la.A_log, la.dt_bias, la.norm.weight]
            self._grant(
                f"layer {i}'s DeltaNet operands in float32",
                sum(4 * int(t.numel()) for t in parts if t is not None),
                "weights",
                sm.dev,
            )
            own = [f32(t) for t in parts]
            W["conv_w"], W["conv_b"], W["a_log"], W["dt_bias"], W["norm_w"] = own
            W["_own"] = [t for t in own if t is not None]
            W["eps_d"] = float(getattr(la.norm, "variance_epsilon", getattr(la.norm, "eps", 1e-6)))
            W["gate"] = 1 if getattr(la.norm, "activation", "silu") == "sigmoid" else 0
        if i in self.pj:
            p = layer.ple
            W["kv"] = self._merge(f"layer {i}'s n-gram embedding", [(p.key_proj, "weight"), (p.value_proj, "weight")])
            W["nq"], W["nk"] = self._bf(p.norm_query.weight, "norm_query"), self._bf(p.norm_key.weight, "norm_key")
            W["nc"] = self._bf(p.norm_conv.weight, "norm_conv")
            W["conv"] = self._bf(p.conv1d.weight, "the n-gram conv").view(self.G * self.H, self.Kp)
            W["ngram"] = _NGram(p.ple_embedding)
        self.W[i] = W
        return W

    # -- the buffers ------------------------------------------------------------------------------------------

    def buffers(self) -> dict[str, Any]:
        """the pass buffers, sized for 32 rows and shared by every graph (a graph of fewer rows reads their
        front), and the pinned rows the host serves the experts through"""
        if self.B is not None:
            return self.B
        sm = self._sm()
        assert sm is not None
        H, G, R, M = self.H, self.G, self.R, ROWS
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
        S = _Cuda.qsa_splits(self.r, self.ktop) if self.sparse else 1
        V = int(sm.head.weight.shape[0])
        spec: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "h": ((M, G * H), bf),
            "xn": ((M, G * H), bf),
            "up": ((M, G * H), bf),
            "dn": ((M, R + G), bf),
            "dn_mix": ((M, R), bf),
            "act": ((M, R), bf),
            "mixed": ((M, H), bf),
            "y": ((M, H), bf),
            "yr": ((M, H), bf),
            "ys": ((M, H), bf),
            "inj_a": ((M, G), bf),
            "inj_m": ((M, G), bf),
            "logits_r": ((M, self.E + 1), bf),
            "idx": ((M, self.topk), i32),
            "wts": ((M, self.topk), bf),
            "gu_s": ((M, 2 * self.Is), bf),
            "logits": ((M, V), bf),
            "meta": ((1 + 4 * M,), i32),
            "crow": ((M,), i32),
            "seq": ((1,), i32),
            "cnt": ((1,), i32),
        }
        if self.sparse:
            spec |= {
                "qkv": ((M, self.width), bf),
                "q": ((M, self.Hq, self.D), bf),
                "qi": ((M, self.Hi, self.di), bf),
                "att": ((M, self.Hq, self.D), bf),
                "sel": ((M, self.ktop), i32),
                "nsel": ((M,), i32),
                "part_m": ((S * M * self.Hq,), f32),
                "part_l": ((S * M * self.Hq,), f32),
                "part_acc": ((S * M * self.Hq * self.D,), f32),
                "acnt": ((M * self.Hq,), i32),
            }
        if self.linear:
            spec |= {
                "core": ((M, self.hv * self.dv), bf),
                "scratch": ((M, self.hv, self.dk, self.dv), f32),
                "proj": ((len(self.linear), M, self.ps), bf),
            }
        if self.ple:
            spec |= {
                "emb": ((M, self.Ep), bf),
                "kvp": ((M, (G + 1) * H), bf),
                "gated": ((len(self.ple), M, G * H), bf),
                "normed": ((len(self.ple), M, G * H), bf),
            }
        B: dict[str, Any] = self._alloc("the pass buffers", spec, "scratch")
        pinned: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
            "xh": ((M, H), bf),
            "yh": ((M, H), bf),
            "hidx": ((M, self.topk), i32),
            "hw": ((M, self.topk), bf),
            "hseq": ((1,), i32),
            "meta_h": ((1 + 4 * M,), i32),
            "crow_h": ((M,), i32),
        }
        if self.ple:
            pinned["emb_h"] = ((len(self.ple), M, self.Ep), bf)
        B |= self._alloc("the host's pinned rows", pinned, "scratch", pinned=True)
        B["seq"].fill_(1)
        B["hseq_np"] = B["hseq"].numpy()
        # the pass's tree, one upload: its length, then each row's depth, parent (-2 padding), proj row (-1 past
        # the pass) and DeltaNet scratch slot
        m = B["meta"]
        B["n0"], B["depth"], B["par"] = m[0:1], m[1 : 1 + M], m[1 + M : 1 + 2 * M]
        B["rows"], B["slots"] = m[1 + 2 * M : 1 + 3 * M], m[1 + 3 * M : 1 + 4 * M]
        self.B = B
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=sm.dev)
            self.pool = torch.cuda.graph_pool_handle()
        self.k = Native.card_kernels()
        return B

    def _hout(self) -> torch.Tensor:
        """each layer's output streams for a pass whose caller reads them (`on_layer`): made on first use"""
        B = self.buffers()
        t = B.get("hout")
        if t is None:
            t = self._alloc(
                "each layer's output rows", {"hout": ((self.L, ROWS, self.G * self.H), torch.bfloat16)}, "scratch"
            )["hout"]
            B["hout"] = t
        return t

    def states(self) -> dict[str, list[torch.Tensor]]:
        """the linear layers' states as the cache holds them - the DeltaNet's conv window and recurrent state in
        float32, the n-gram embedding's kept inputs and ids - bound into the cache the program runs, so its torch
        paths write the same tensors"""
        if self.S is not None:
            return self.S
        C, K = self.C, self.Kc
        spec: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for j in range(len(self.linear)):
            spec[f"conv{j}"] = ((1, C, K), torch.float32)
            spec[f"rec{j}"] = ((1, self.hv, self.dk, self.dv), torch.float32)
        for j in range(len(self.ple)):
            spec[f"pre{j}"] = ((1, self.G * self.H, self.Lp), torch.bfloat16)
            spec[f"ctx{j}"] = ((1, self.n_ids), torch.long)
        t = self._alloc("the linear layers' states", spec, "kv")
        self.S = {
            "conv": [t[f"conv{j}"] for j in range(len(self.linear))],
            "rec": [t[f"rec{j}"] for j in range(len(self.linear))],
            "pre": [t[f"pre{j}"] for j in range(len(self.ple))],
            "ctx": [t[f"ctx{j}"] for j in range(len(self.ple))],
        }
        return self.S

    # positions of the rope tables made on the card at a time, for an arena kept in RAM: the tables' float32 rows of a
    # million positions at once were half a GiB of the card's
    ROPE_CHUNK = 1 << 16

    def arena(self, need: int) -> dict[str, Any]:
        """the sparse layers' rows - keys and values [layers, 2, Hk, cap, D], the indexer's raw keys [layers, cap,
        di] and pooled keys - and the rope over `cap` positions, at least `need` rows deep: grown by reallocation (the
        rows copied over, every bound layer attached again, the graphs dropped).

        With the attention's rows kept in RAM (`kv_host`: a context the card has no room for) the rows and the rope
        tables are pinned host memory the kernels read and write in place (`hostmem.pinned`: its exact size, freed
        with it) - a pass reads only the rows its picks name, the indexer's budget whatever the context - and the
        arena is made for the load's whole context at once, a regrowth copying every row; the pooled keys and the
        scores, which every pass reads whole, stay the card's"""
        A = self.A
        if A is not None and A["cap"] >= need:
            return A
        sm = self._sm()
        assert sm is not None
        ram = bool(getattr(sm, "kv_host", False))
        old = A["cap"] if A is not None else 0
        whole = int(getattr(sm, "context", 0) or 0) + ROWS if ram else 0
        cap = max(int(need) + ROWS, self.ARENA_MIN, old + old // 4, whole)
        cap = (cap + 1023) // 1024 * 1024
        n = len(self.sparse)
        r = max(1, self.r)
        nb = cap // r + 1
        rot = self._rot(sm)
        # the rows and the rope tables: on the card, or in RAM
        rows = lambda c: n * (2 * self.Hk * c * self.D + c * self.di) * 2 + 2 * c * rot * 2
        # the pooled keys, their lengths and the scores: the card's
        card = lambda c: n * (c // r + 1) * self.di * 2 + ROWS * (c // r + 1) * 4 + n * 2 * 4
        cpu = torch.device("cpu")
        if ram:
            self._grant(f"the arena's rows in RAM, {cap} rows", rows(cap), "kv", cpu, held=rows(old) if old else 0)
            self._grant(
                f"the arena, {cap} rows", card(cap), "kv", sm.dev, held=card(old) if old else 0, cap=cap, bound=None
            )
        else:
            per = lambda c: rows(c) + card(c)
            self._grant(
                f"the arena, {cap} rows", per(cap), "kv", sm.dev, held=per(old) if old else 0, cap=cap, bound=None
            )
        bf = torch.bfloat16
        if ram:
            kv = pinned((n, 2, self.Hk, cap, self.D), bf)
            raw = pinned((n, cap, self.di), bf)
        else:
            kv = torch.zeros(n, 2, self.Hk, cap, self.D, dtype=bf, device=sm.dev)
            raw = torch.zeros(n, cap, self.di, dtype=bf, device=sm.dev)
        pk = torch.zeros(n, nb, self.di, dtype=bf, device=sm.dev)
        pk_len = torch.zeros(n, 2, dtype=torch.int32, device=sm.dev)
        if A is not None:
            if ram:
                torch.cuda.synchronize(sm.dev)  # every kernel's write to the old rows landed before they are copied
            kv[:, :, :, :old].copy_(A["kv"])
            raw[:, :old].copy_(A["raw"])
            pk[:, : A["pk"].shape[1]].copy_(A["pk"])
            pk_len.copy_(A["pk_len"])
        cos, sin = self._rope(sm, cap, rot, A, ram)
        self.A = {
            "cap": cap,
            "kv": kv,
            "raw": raw,
            "pk": pk,
            "pk_len": pk_len,
            "cos": cos,
            "sin": sin,
            "scores": torch.zeros(ROWS, nb, dtype=torch.float32, device=sm.dev),
        }
        for j, ref in enumerate(self.bound):
            layer = ref() if ref is not None else None
            if layer is not None and layer.attached:
                # a layer that let the arena go since (given up to the host, its rows a fork's) keeps its own rows: the
                # next bind attaches it again where the program runs it
                layer.attach(kv[j, 0], kv[j, 1], raw[j])
        self._drop_graphs()
        if old:
            sm.log(f"[card] Qwen4's arena grown to {cap} rows ({per(cap) / 2**20:.0f} MB); graphs dropped")
        return self.A

    def _rope(
        self, sm: Any, cap: int, rot: int, A: dict[str, Any] | None, ram: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """the rope's cos and sin [cap, rot] bf16 at every position of the arena: on the card in one go, or in RAM
        (pinned) made a chunk at a time on the card, an old arena's positions copied over"""
        x = torch.zeros(1, 1, self.H, dtype=torch.bfloat16, device=sm.dev)
        if not ram:
            pos = torch.arange(cap, device=sm.dev).view(1, 1, -1).expand(3, 1, -1)
            cos, sin = sm.rotary(x, pos)
            return cos[0].to(torch.bfloat16).contiguous(), sin[0].to(torch.bfloat16).contiguous()
        out = pinned((cap, rot), torch.bfloat16), pinned((cap, rot), torch.bfloat16)
        start = 0
        if A is not None and A["cos"].device.type == "cpu":
            start = int(A["cap"])
            out[0][:start].copy_(A["cos"])
            out[1][:start].copy_(A["sin"])
        for a in range(start, cap, self.ROPE_CHUNK):
            b = min(cap, a + self.ROPE_CHUNK)
            pos = torch.arange(a, b, device=sm.dev).view(1, 1, -1).expand(3, 1, -1)
            c, s = sm.rotary(x, pos)
            out[0][a:b].copy_(c[0].to(torch.bfloat16))
            out[1][a:b].copy_(s[0].to(torch.bfloat16))
        return out

    def _arena_grow(self, need: int) -> None:
        self.arena(need)

    # -- binding a cache --------------------------------------------------------------------------------------

    def adopt(self, cache: Any, need: int) -> None:
        """`cache`'s sparse layers bound to the program's arena, at least `need` rows deep: the cache the program
        holds already only checked (a layer given up the arena is attached again), another one taking it over - the
        one holding it before keeping copies of its own. The rows only: the linear layers' states are bound by a
        pass's `begin` (`_bind`). A torch-path pass over an arena in RAM adopts its cache first (`attend_rows`), so
        its rows go where the program's kernels read them"""
        sm = self._sm()
        assert sm is not None
        owner = self.owner() if self.owner is not None else None
        if owner is not cache:
            if owner is not None:
                self._evict(owner)
            self.owner = weakref.ref(cache)
            self.ctx = [None] * len(self.ple)
            self.seen = None
        A = self.arena(need)
        grant = sm.scheduler.grant if getattr(sm, "scheduler", None) is not None else None
        for j, i in enumerate(self.sparse):
            cl = cache.layers[i]
            k, v, raw = A["kv"][j, 0], A["kv"][j, 1], A["raw"][j]
            if isinstance(cl, ArenaIndexedLayer) and cl.attached_to(k):
                self.bound[j] = weakref.ref(cl)
                continue
            if isinstance(cl, ArenaIndexedLayer):
                cl.attach(k, v, raw)
            else:
                # the layer bound before its rows go in: an arena that grows under them attaches it again
                new = ArenaIndexedLayer(self._arena_grow, grant)
                new.attach(k, v, raw)
                self.bound[j] = weakref.ref(new)
                new.load(cl.keys, cl.values, indexer_keys(cl))
                cache.layers[i] = new
                cl = new
            self.bound[j] = weakref.ref(cl)
            A["pk_len"][j].zero_()
            self.pooled[j] = 0

    def _bind(self, cache: Any, need: int) -> None:
        """`cache`'s layers bound to the program's arena and states: a cache the program holds already only
        checked (a torch path may have replaced a state tensor, or a layer given up the arena); another one
        takes them over, the one holding them before keeping copies of its own"""
        sm = self._sm()
        assert sm is not None
        self.adopt(cache, need)
        S = self.states()
        for j, i in enumerate(self.linear):
            cl = cache.layers[i]
            conv, rec = sm._lin(cl)
            mc, mr = S["conv"][j], S["rec"][j]
            if conv is not mc or rec is not mr:
                if conv is not mc:
                    mc.copy_(conv)
                if rec is not mr:
                    mr.copy_(rec)
                sm._lin_set(cl, mc, mr)
            if i in self.pj:
                p = self.pj[i]
                states = cl.conv_states
                for idx, mine in ((1, S["pre"][p]), (2, S["ctx"][p])):
                    t = states[idx]
                    if t is not mine:
                        mine.copy_(t)
                        states[idx] = mine
                        self.ctx[p] = None

    def _evict(self, owner: Any) -> None:
        """the cache holding the program's buffers lets them go: its rows and states become copies of its own"""
        sm = self._sm()
        assert sm is not None
        S = self.S
        mine = {id(t) for ts in (S or {}).values() for t in ts}
        # a linear layer's states by index, as transformers' cache keeps them (the program binds its own into them)
        states = [d for i in self.linear for d in (owner.layers[i].conv_states, owner.layers[i].recurrent_states)]
        held = [(d, key) for d in states for key, t in d.items() if isinstance(t, torch.Tensor) and id(t) in mine]
        nbytes = sum(d[key].numel() * d[key].element_size() for d, key in held)
        if nbytes and getattr(sm, "scheduler", None) is not None:
            sm.scheduler.grant(
                nbytes, "kv", requester="Qwen4's card program: a cache's states, copied out", device=sm.dev, draws=""
            )
        for d, key in held:
            d[key] = d[key].clone()
        for j, i in enumerate(self.sparse):
            cl = owner.layers[i]
            if isinstance(cl, ArenaIndexedLayer) and self.A is not None and cl.attached_to(self.A["kv"][j, 0]):
                cl.detach()
            self.bound[j] = None

    def touched(self, cache: Any) -> None:
        """a pass off the program over `cache`: what the host mirrors of it are read again"""
        owner = self.owner() if self.owner is not None else None
        if owner is cache:
            self.seen = None

    # -- a pass -----------------------------------------------------------------------------------------------

    def begin(
        self,
        cache: Any,
        h: torch.Tensor,
        ids: list[int],
        past: int,
        parents: list[int],
        depth: list[int],
        mode: str,
        tap: bool,
    ) -> int:
        """a pass of T = len(ids) rows over `cache` (`past` rows long): the cache bound, the tree uploaded, the
        pooled keys caught up where the graphs' own pooling would fall short, the rows `h` [1, T, G H] in (the
        n-gram rows are staged as the segment of their layer is entered, `enter`). Returns the graphs' row count
        M."""
        sm = self._sm()
        assert sm is not None
        T = len(ids)
        M = sm._card_m(T)
        self.T, self.M, self.n0, self.mode, self.tap = T, M, int(past), mode, bool(tap)
        self.passes += 1
        self.ids, self.parents = [int(t) for t in ids], [int(p) for p in parents]
        B = self.buffers()
        # every layer's weights read before any graph is captured: a merge recorded into a capture would run only at
        # the replay, reading the parts it replaced after they were let go
        for i in (*(self.res or ()), self.L):
            self.weights(i)
        if not self.clean:
            # a pass that raised part way: its graphs may still publish, so the card is let finish first
            torch.cuda.synchronize()
        B["hseq_np"][0] = 0
        self.clean = False
        self._bind(cache, past + ROWS)
        A = self.A
        assert A is not None
        if tap:
            self._hout()
        self._tree(past, depth, mode)
        self._catch_up(past)
        self._ple_context()
        self.staged = False
        if self.host_layers:
            self.edge()
        hb = B["h"]
        hb[:T].copy_(h.reshape(T, -1))
        if T < M:
            hb[T:M].zero_()  # padding rows start from zero every pass, never from a growing residue
        return M

    def _tree(self, past: int, depth: list[int], mode: str) -> None:
        """the pass's tree, one upload: depth and parent a row (padding -2, which every kernel skips), the proj rows
        the DeltaNet nodes read (the pass's own, -1 past them), the scratch slot each node steps in"""
        B, T = self.B, self.T
        assert B is not None
        mh = B["meta_h"]
        mh.zero_()
        mh[0] = past
        mh[1 : 1 + T] = torch.tensor(depth, dtype=torch.int32)
        par = torch.full((ROWS,), -2, dtype=torch.int32)
        par[:T] = torch.tensor(self.parents, dtype=torch.int32)
        mh[1 + ROWS : 1 + 2 * ROWS] = par
        rows = torch.full((ROWS,), -1, dtype=torch.int32)
        rows[:T] = torch.arange(T, dtype=torch.int32)
        mh[1 + 2 * ROWS : 1 + 3 * ROWS] = rows
        if mode == TREE:
            mh[1 + 3 * ROWS : 1 + 3 * ROWS + T] = torch.tensor(_slots(self.parents)[0], dtype=torch.int32)
        B["meta"].copy_(mh, non_blocking=True)

    def rehearse(self, T: int, n0: int) -> int:
        """a timing pass of a T-row chain verified at `n0` rows of context (`_card_program_time`), over the
        program's own buffers and no cache: the graphs the verify pass of that width replays - captured now if
        they were not - with `between` rehearsed (`dry`: no expert read, the routed rows as they lie), the n-gram
        rows as they lie (no table read) and nothing a cache holds written - a verify pass writes only rows past
        the committed ones, and the cache the program held is let go first (it keeps copies of its own). The
        pooled keys stand at `n0` for it, as a live cache's would. Returns the graphs' row count M; `rehearsed`
        ends it."""
        sm = self._sm()
        assert sm is not None
        owner = self.owner() if self.owner is not None else None
        if owner is not None:
            self._evict(owner)
        self.owner = None
        self.seen = None
        M = sm._card_m(T)
        self.T, self.M, self.n0, self.mode, self.tap = T, M, int(n0), TREE, False
        self.passes += 1
        self.ids, self.parents = [0] * T, list(range(-1, T - 1))
        B = self.buffers()
        for i in (*(self.res or ()), self.L):
            self.weights(i)
        if not self.clean:
            torch.cuda.synchronize()
        B["hseq_np"][0] = 0
        self.clean = False
        A = self.arena(int(n0) + ROWS)
        self.states()
        self._tree(int(n0), list(range(T)), TREE)
        A["pk_len"][:, 0].fill_(int(n0) // max(1, self.r))
        self.staged = True
        return M

    def rehearsed(self) -> None:
        """a timing pass ended: the pooled keys it stood at let go (the next cache bound pools its own)"""
        A = self.A
        assert A is not None  # `rehearse` made it
        A["pk_len"].zero_()
        self.pooled = [0] * len(self.sparse)
        self.clean = True

    def _catch_up(self, n0: int) -> None:
        """each sparse layer's pooled keys brought to where the graph's own pool (a pass's growth at most) reaches:
        rows a torch path rewrote (a crop, a prefill) pooled again, a long stretch pooled here at once"""
        A, B, k = self.A, self.B, self.k
        assert A is not None and B is not None
        r = max(1, self.r)
        for j in range(len(self.sparse)):
            ref = self.bound[j]
            layer = ref() if ref is not None else None
            done = min(self.pooled[j], n0 // r)
            if layer is not None:
                done = min(done, int(layer.low) // r)
                layer.low = sys.maxsize
            if done != self.pooled[j]:
                A["pk_len"][j, 0].fill_(done)
                self.pooled[j] = done
            lag = n0 // r - done
            if lag > self.pool_new:
                k.qsa_pool(
                    A["raw"][j], A["pk"][j], A["pk_len"][j], B["n0"], A["cos"], A["sin"], self.weights(self.sparse[j])["ik"],
                    self.eps, r, lag,
                )  # fmt: skip
            self.pooled[j] = n0 // r

    def _ple_context(self) -> None:
        """the n-gram embedding's id context on the host: the program's own mirror where it wrote the cache last,
        else read off the card"""
        sm = self._sm()
        owner = self.owner() if self.owner is not None else None
        if not self.ple or owner is None or sm is None:
            return
        S = self.S
        assert S is not None
        fresh = self.seen is not None and self.seen == owner.get_seq_length()
        for p in range(len(self.ple)):
            if not fresh or self.ctx[p] is None:
                self.ctx[p] = [int(t) for t in S["ctx"][p][0].tolist()]

    def _stage(self) -> None:
        """each PLE layer's n-gram rows for the pass's nodes, into pinned staging (the layer's graph copies them in):
        a node's last ids along its path, the context before the pass beneath them - verify.py's `_ple_forward`"""
        if self.staged:
            return
        B = self.B
        assert B is not None
        n = self.dil
        T = self.T
        paths = [path_of(self.parents, j) for j in range(T)]
        for p, i in enumerate(self.ple):
            ctx = self.ctx[p] or []
            hist = torch.empty(T, n, dtype=torch.long)
            for j, pth in enumerate(paths):
                for s in range(n):  # s steps back from node j
                    hist[j, n - 1 - s] = self.ids[pth[s]] if s < len(pth) else ctx[len(ctx) - (s - len(pth)) - 1]
            rows = self.weights(i)["ngram"].rows(hist)
            B["emb_h"][p][:T].copy_(rows)
        self.staged = True

    def _gemv(self, W: torch.Tensor, x: torch.Tensor, y: torch.Tensor, act: bool = False) -> None:
        k = self.k
        M = int(x.shape[0])
        R, C = (int(s) for s in W.shape)
        P, ci = k.ptr, ctypes.c_int
        name = f"btb_gemv_silu_bf16_m{M}" if act else f"btb_gemv_bf16_m{M}"
        k.launch(name, ((R + 3) // 4, 1, 1), (128, 1, 1), [P(W), P(x), P(y), ci(R), ci(C)])

    def _hc_read(self, Wh: dict[str, torch.Tensor], M: int, inj: torch.Tensor | None) -> None:
        """a hyper-connection's read of the normed streams `xn`: the mixed row into `mixed`, the inject weights
        into `inj` (None: the final mixer)"""
        B, k, G, R = self.B, self.k, self.G, self.R
        assert B is not None
        dn = B["dn"][:M] if inj is not None else B["dn_mix"][:M]
        self._gemv(Wh["dn"], B["xn"][:M], dn)
        k.hc_act(dn, B["act"][:M], G)
        self._gemv(Wh["up"], B["act"][:M], B["up"][:M])
        k.hc_mix(B["xn"][:M], B["up"][:M], dn if inj is not None else None, B["mixed"][:M], inj, G, R)

    def _moe_close(self, M: int) -> None:
        """the previous layer's mixture closed: the routed experts' rows the host wrote, plus the shared expert's
        gated output, into `y`"""
        B, k = self.B, self.k
        assert B is not None
        B["yr"][:M].copy_(B["yh"][:M], non_blocking=True)
        k.moe_combine(B["yr"][:M], B["ys"][:M], B["logits_r"][:M], self.E, B["y"][:M])

    def layer_body(self, i: int, M: int, mode: str, tap: bool) -> None:
        """graph i: the previous layer's mixture closed into the streams, layer i through its router's publish and
        its shared expert (a segment's first layer takes the streams as they came in: the layer before it is the
        host's, or there is none)"""
        B, k, W = self.B, self.k, self.weights(i)
        assert B is not None
        first = (i - 1) not in self.at
        if not first:
            self._moe_close(M)
        y = None if first else B["y"][:M]
        inj = None if first else B["inj_m"][:M]
        h, xn = B["h"][:M], B["xn"][:M]
        if i in self.pj:
            k.hc_rmsnorm(h, y, inj, W["nq"], self.eps, xn, self.G)
            if tap and not first:
                B["hout"][i - 1][:M].copy_(h)
            self._ple(i, M, mode)
            k.hc_rmsnorm(h, None, None, W["hcA"]["norm"], self.eps, xn, self.G)
        else:
            k.hc_rmsnorm(h, y, inj, W["hcA"]["norm"], self.eps, xn, self.G)
            if tap and not first:
                B["hout"][i - 1][:M].copy_(h)
        self._hc_read(W["hcA"], M, B["inj_a"][:M])
        if i in self.sj:
            self._attention(i, M)
        else:
            self._delta(i, M, mode)
        k.hc_rmsnorm(h, B["y"][:M], B["inj_a"][:M], W["hcM"]["norm"], self.eps, xn, self.G)
        self._hc_read(W["hcM"], M, B["inj_m"][:M])
        # the mixture: the rows to the host, then the router's picks published behind them, the shared expert while
        # the host serves the routed experts
        B["xh"][:M].copy_(B["mixed"][:M], non_blocking=True)
        self._gemv(W["router"], B["mixed"][:M], B["logits_r"][:M])
        k.moe_route(
            B["logits_r"][:M], self.E, self.topk, B["idx"][:M], B["wts"][:M],
            B["hidx"][:M], B["hw"][:M], B["cnt"], B["seq"], B["hseq"],
        )  # fmt: skip
        self._gemv(W["sgu"], B["mixed"][:M], B["gu_s"][:M])
        self._gemv(W["sdn"], B["gu_s"][:M], B["ys"][:M], act=True)

    def close_body(self, i: int, M: int, mode: str, tap: bool) -> None:
        """a segment's close: its last layer `i`'s mixture into the streams, which leave for the host layer after
        it (the stream write is the read's kernel, its normed rows unread)"""
        B, k, W = self.B, self.k, self.weights(i)
        assert B is not None
        self._moe_close(M)
        h = B["h"][:M]
        k.hc_rmsnorm(h, B["y"][:M], B["inj_m"][:M], W["hcM"]["norm"], self.eps, B["xn"][:M], self.G)
        if tap:
            B["hout"][i][:M].copy_(h)

    def tail_body(self, M: int, mode: str, tap: bool) -> None:
        """the tail: the last layer's mixture closed into the streams (a last layer on the host hands them in
        whole), the final mixer, the head"""
        B, k, W = self.B, self.k, self.weights(self.L)
        assert B is not None
        h = B["h"][:M]
        if (self.L - 1) in self.at:
            self._moe_close(M)
            k.hc_rmsnorm(h, B["y"][:M], B["inj_m"][:M], W["hc"]["norm"], self.eps, B["xn"][:M], self.G)
            if tap:
                B["hout"][self.L - 1][:M].copy_(h)
        else:
            k.hc_rmsnorm(h, None, None, W["hc"]["norm"], self.eps, B["xn"][:M], self.G)
        self._hc_read(W["hc"], M, None)
        self._gemv(W["head"], B["mixed"][:M], B["logits"][:M])

    # -- the edge: the streams to a host layer and back ---------------------------------------------------------

    def edge(self) -> dict[str, torch.Tensor]:
        """the pinned rows the streams cross the edge through: bf16 as the card holds them, float32 as the host
        layers take them. Made by `ok` before a pass of a placement with host layers, where the expert store may
        give room back for them (it grows into the RAM the ledger shows free); never in the middle of a pass"""
        if self.X is None:
            GH = self.G * self.H
            self.X = self._alloc(
                "the edge's rows",
                {"hx": ((ROWS, GH), torch.bfloat16), "hf": ((ROWS, GH), torch.float32)},
                "scratch",
                pinned=True,
                reclaim=True,
            )
        return self.X

    def leave(self) -> torch.Tensor:
        """the pass's streams [1, T, G H] off the card for a host layer, float32 on the host (the copy waits for the
        graphs before it; the host layer's input is the edge's own rows, read by the next crossing only)"""
        B, X, T = self.B, self.edge(), self.T
        assert B is not None
        X["hx"][:T].copy_(B["h"][:T])
        hf = X["hf"][:T]
        hf.copy_(X["hx"][:T])
        return hf.view(1, T, -1)

    def enter(self, i: int, h: torch.Tensor | None) -> None:
        """the segment that starts at layer `i` (the tail at L) entered: a host layer's streams `h` [1, T, G H] in,
        cast to bf16 on the host and copied behind the graphs before it; its layer's n-gram rows staged"""
        B = self.B
        assert B is not None
        if h is not None:
            X, T = self.edge(), self.T
            X["hx"][:T].copy_(h.reshape(T, -1))
            B["h"][:T].copy_(X["hx"][:T], non_blocking=True)
        if i in self.pj:
            self._stage()

    def _attention(self, i: int, M: int) -> None:
        """the sparse attention: the pooled keys caught up, q | k | v | the indexer's q and key in one gemv, the
        rows normed and roped into the arena, each node's picks, the attention over them, o_proj through the gate"""
        A, B, k, W = self.A, self.B, self.k, self.weights(i)
        assert A is not None and B is not None
        j = self.sj[i]
        r = max(1, self.r)
        K, V, raw = A["kv"][j, 0], A["kv"][j, 1], A["raw"][j]
        cos, sin = A["cos"], A["sin"]
        n0, depth, par = B["n0"], B["depth"][:M], B["par"][:M]
        k.qsa_pool(raw, A["pk"][j], A["pk_len"][j], n0, cos, sin, W["ik"], self.eps, r, self.pool_new)
        qkv = B["qkv"][:M]
        self._gemv(W["qkv"], B["mixed"][:M], qkv)
        D, Hq, Hk = self.D, self.Hq, self.Hk
        k.norm_rope_part(
            qkv, B["q"][:M], K, V, cos, sin, n0, depth, Hq, Hk, 2 * D, self.ko, self.vo, W["wq"], W["wk"], self.eps
        )
        k.norm_rope_part(
            qkv, B["qi"][:M], raw.unsqueeze(0), None, cos, sin, n0, depth, self.Hi, 1, self.di, self.ik0, self.ik0,
            W["iq"], None, self.eps, True, True, self.iq0,
        )  # fmt: skip
        k.qsa_select(
            B["qi"][:M], A["pk"][j], raw, n0, par, cos, sin, W["ik"], self.eps, r, self.ktop,
            A["scores"][:M], B["sel"][:M], B["nsel"][:M],
        )  # fmt: skip
        k.qsa_attn_split(
            B["q"][:M], K, V, B["att"][:M], n0, par, B["sel"][:M], B["nsel"][:M], r, self.ktop, W["scale"],
            B["part_m"], B["part_l"], B["part_acc"], B["acnt"],
        )  # fmt: skip
        k.gemv_sgate(W["o"], B["att"][:M].view(M, Hq * D), qkv, B["y"][:M], D, D, 2 * D)

    def attend_rows(
        self, attn: Any, hs: torch.Tensor, pe: tuple[torch.Tensor, torch.Tensor], cache: Any, parents: list[int]
    ) -> torch.Tensor:
        """A torch-path pass's sparse attention for resident layer `attn.layer_idx` over the arena in RAM (`kv_host`):
        the queries, the gate, the keys and values, and the indexer's query and raw key made as the reference makes
        them (attend.py, qsa.py), the rows written into the arena, then the program's own kernels a slice of up to
        ROWS rows at a time - the pooled keys caught up to the slice, its picks, the attention over them - so the card
        reads the indexer's budget of rows a query whatever the context. A prefill's chunk is a chain of slices;
        `parents` a verify pass's tree (ROWS rows at most) or a chain's. `hs` [1, T, H]; returns the output rows
        [1, T, H] (o_proj through the gate), as the reference module returns them"""
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

        sm = self._sm()
        assert sm is not None
        B, T, _ = hs.shape
        chain = parents == list(range(-1, T - 1))
        if B != 1 or (T > ROWS and not chain):
            raise RuntimeError(
                f"[card] Qwen4's attention over its rows in RAM takes one sequence and a tree of {ROWS} rows at most "
                f"(a batch of {B}, {T} rows)"
            )
        i = int(attn.layer_idx)
        j = self.sj[i]
        r = max(1, self.r)
        past = int(cache.layers[i].get_seq_length())
        self.adopt(cache, past + T + ROWS)
        cl = cache.layers[i]
        past = int(cl.get_seq_length())
        # the queries, the gate, the keys and the values, and the indexer's query and raw key, as the reference
        cos, sin = (x[:, -T:, :] for x in pe)
        hd = int(attn.head_dim)
        q, gate = torch.chunk(attn.q_proj(hs).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = attn.q_norm(q.reshape(B, T, -1, hd)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hs).view(B, T, -1, hd)).transpose(1, 2)
        v = attn.v_proj(hs).view(B, T, -1, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        ix = attn.indexer
        d = int(ix.index_head_dim)
        qi, raw = torch.split(ix.index_qk_proj(hs), [int(ix.index_n_heads) * d, int(ix.index_kv_heads) * d], dim=-1)
        qi = apply_rotary_pos_emb(ix.q_layernorm(qi.reshape(B, T, -1, d)), cos=cos, sin=sin, unsqueeze_dim=2)
        # the rows into the arena (a growth there reallocates it: read after), then every slice over them
        cl.update(k, v)
        cl.update_indexer(raw.reshape(B, T, -1, d).squeeze(2))
        A, Bf, kern, W = self.A, self.buffers(), self.k, self.weights(i)
        assert A is not None
        P = self._tree_rows()
        qs = q[0].transpose(0, 1)
        out = torch.empty(T, self.Hq, hd, dtype=hs.dtype, device=hs.device)
        for s in range(0, T, ROWS):
            e = min(T, s + ROWS)
            t = e - s
            n0 = past + s
            # the pooled keys of every complete block before the slice: those its rows rewrote pooled again
            done = min(self.pooled[j], n0 // r, int(cl.low) // r)
            cl.low = sys.maxsize
            A["pk_len"][j, 0].fill_(done)
            P["n0"].fill_(n0)
            if n0 // r > done:
                kern.qsa_pool(
                    A["raw"][j],
                    A["pk"][j],
                    A["pk_len"][j],
                    P["n0"],
                    A["cos"],
                    A["sin"],
                    W["ik"],
                    self.eps,
                    r,
                    n0 // r - done,
                )
            self.pooled[j] = n0 // r
            par = list(range(-1, t - 1)) if chain else parents
            P["par"][:t].copy_(torch.tensor(par, dtype=torch.int32))
            kern.qsa_select(
                qi[0, s:e].contiguous(), A["pk"][j], A["raw"][j], P["n0"], P["par"][:t], A["cos"], A["sin"], W["ik"],
                self.eps, r, self.ktop, A["scores"][:t], Bf["sel"][:t], Bf["nsel"][:t],
            )  # fmt: skip
            kern.qsa_attn_split(
                qs[s:e].contiguous(), A["kv"][j, 0], A["kv"][j, 1], out[s:e], P["n0"], P["par"][:t], Bf["sel"][:t],
                Bf["nsel"][:t], r, self.ktop, W["scale"], Bf["part_m"], Bf["part_l"], Bf["part_acc"], Bf["acnt"],
            )  # fmt: skip
        a = out.view(B, T, -1)
        return attn.o_proj(a * torch.sigmoid(gate))

    def _tree_rows(self) -> dict[str, torch.Tensor]:
        """the length and the parents a torch-path slice over the arena in RAM hands the kernels (`attend_rows`)"""
        P = self.P
        if P is None:
            P = self.P = self._alloc(
                "the torch path's slices over the rows in RAM",
                {"n0": ((1,), torch.int32), "par": ((ROWS,), torch.int32)},
                "scratch",
            )
        return P

    def _delta(self, i: int, M: int, mode: str) -> None:
        """the gated DeltaNet: q|k|v|z|b|a in one gemv into the layer's own rows (the commit steps them again),
        the nodes stepped - a step's in place, a verify pass's in scratch from the cache's state - out_proj"""
        B, S, k, W = self.B, self.S, self.k, self.weights(i)
        assert B is not None and S is not None
        j = self.lj[i]
        proj = B["proj"][j][:M]
        self._gemv(W["proj"], B["mixed"][:M], proj)
        conv, rec = S["conv"][j][0], S["rec"][j][0]
        tree = mode == TREE
        k.delta_nodes_bf16(
            proj, self.off, B["rows"][:M], W["conv_w"], W["conv_b"], conv, rec,
            B["scratch"] if tree else None, B["slots"][:M] if tree else None, B["par"][:M] if tree else None,
            W["a_log"], W["dt_bias"], W["norm_w"], W["eps_d"], W["gate"], self.hk, self.hv, self.dk, self.dv,
            B["core"][:M],
        )  # fmt: skip
        if not tree:
            k.conv_window(conv, proj, 0, B["rows"][:M], M)
        self._gemv(W["out"], B["core"][:M], B["y"][:M])

    def _ple(self, i: int, M: int, mode: str) -> None:
        """the per-layer n-gram embedding: the staged rows in, [key | value] in one gemv, the nodes' gate, gated
        rows and conv into the streams (a step's kept window shifted in place)"""
        B, S, k, W = self.B, self.S, self.k, self.weights(i)
        assert B is not None and S is not None
        p = self.pj[i]
        B["emb"][:M].copy_(B["emb_h"][p][:M], non_blocking=True)
        self._gemv(W["kv"], B["emb"][:M], B["kvp"][:M])
        k.ple_nodes(
            B["kvp"][:M], B["xn"][:M], W["nk"], W["nc"], W["conv"], S["pre"][p][0], B["par"][:M], B["h"][:M],
            B["gated"][p][:M], B["normed"][p][:M], self.G, self.dil, self.eps, update=mode == STEP,
        )  # fmt: skip

    def between(self, i: int, dry: bool = False) -> None:
        """the host's part between graph i and graph i + 1: layer i's routed experts through the store, once its
        router has published the rows' picks, the sum into pinned memory for the next graph (`dry`, a timing
        pass's: the wait for the publish alone - no store call, the routed rows left as they lie)"""
        sm = self._sm()
        B = self.B
        assert sm is not None and B is not None
        nxt = i + 1
        if nxt < self.L and nxt in self.pj and not dry:
            self._stage()  # the n-gram rows while the card runs the layer
        flag = B["hseq_np"]
        t0 = time.perf_counter()
        while int(flag[0]) != 1:
            if time.perf_counter() - t0 > self.SPIN_S:
                raise RuntimeError(f"[card] layer {i}'s router did not publish within {self.SPIN_S:.0f} s")
        flag[0] = 0
        if dry:
            return
        T = self.T
        x = B["xh"][:T].clone()
        idx = B["hidx"][:T].to(torch.long)
        w = B["hw"][:T].clone()
        # an expert seated on the card reads the rows where the program holds them (`_Experts._seated`)
        y = sm.resident[i].mlp.experts(x, idx, w, x_card=B["mixed"][:T])
        B["yh"][:T].copy_(y.reshape(T, -1))

    def end(self) -> None:
        """the pass's bookkeeping: a step's rows committed (the lengths, the n-gram context), a verify pass's commit
        handed to the engine's (`ad` runs it with the accepted path)"""
        sm = self._sm()
        assert sm is not None
        self.clean = True
        n = self.n0 + (1 if self.mode == STEP else self.T)
        for ref in self.bound:
            layer = ref() if ref is not None else None
            if layer is not None:
                layer.set_front(n)
        if self.mode == STEP:
            self._context(self.ids)
            self.seen = self.n0 + 1
        else:
            self.seen = None
            self.to_commit = self.passes
            sm.spec_commits.append(self.commit)

    def _context(self, ids: list[int]) -> None:
        """the n-gram context after `ids` committed: the host's mirror and the cache's tensor"""
        S = self.S
        assert S is not None
        for p in range(len(self.ple)):
            ctx = ((self.ctx[p] or []) + list(ids))[-self.n_ids :] if self.n_ids else []
            self.ctx[p] = ctx
            if ctx:
                S["ctx"][p][0].copy_(torch.tensor(ctx, dtype=torch.long))

    def commit(self, path: list[int]) -> None:
        """a verify pass's accepted `path` (node indices, root first) stepped into the states: the DeltaNet along the
        path from the pass's own projections, the n-gram embedding's kept inputs and context (the attention's rows
        move with `ArenaIndexedLayer.keep_path`)"""
        sm = self._sm()
        B, S = self.B, self.S
        if sm is None or B is None or S is None or not path:
            return
        if self.to_commit != self.passes:
            # the pass's projections and n-gram rows are another pass's by now
            raise RuntimeError("[card] a verify pass's commit asked after another pass ran on the program")
        self.to_commit = -1
        path = [int(p) for p in path]
        n = len(path)
        with torch.inference_mode():
            ch = B["crow_h"]
            ch.fill_(-1)
            ch[:n] = torch.tensor(path, dtype=torch.int32)
            B["crow"].copy_(ch, non_blocking=True)
            if self.linear:
                sm._card_graph_run(self, ("commit",), self._commit_body)
            for p in range(len(self.ple)):
                pre = S["pre"][p][0]
                rows = B["normed"][p].index_select(0, B["crow"][:n].long())
                pre.copy_(torch.cat([pre, rows.t()], dim=1)[:, -self.Lp :])
            self._context([self.ids[q] for q in path])
        self.seen = self.n0 + n

    def _commit_body(self) -> None:
        """each DeltaNet layer's state and conv window stepped along the accepted rows (`crow`, -1 past them)"""
        B, S, k = self.B, self.S, self.k
        assert B is not None and S is not None
        for j, i in enumerate(self.linear):
            W = self.weights(i)
            conv, rec = S["conv"][j][0], S["rec"][j][0]
            k.delta_nodes_bf16(
                B["proj"][j], self.off, B["crow"], W["conv_w"], W["conv_b"], conv, rec, None, None, None,
                W["a_log"], W["dt_bias"], W["norm_w"], W["eps_d"], W["gate"], self.hk, self.hv, self.dk, self.dv, None,
            )  # fmt: skip
            k.conv_window(conv, B["proj"][j], 0, B["crow"], ROWS)

    def logits(self, T: int) -> torch.Tensor:
        """the pass's logits [1, T, V] bf16, a view of the program's buffer (read before the next pass)"""
        B = self.B
        assert B is not None
        return B["logits"][:T].view(1, T, -1)

    def hidden(self, T: int) -> torch.Tensor:
        """the final mixer's rows [1, T, H] (a pass asked without the head)"""
        B = self.B
        assert B is not None
        return B["mixed"][:T].view(1, T, -1)

    def tap_rows(self, i: int, T: int) -> torch.Tensor:
        """layer i's output streams [1, T, G H] of the last tapped pass"""
        B = self.B
        assert B is not None
        return B["hout"][i][:T].view(1, T, -1)

    # -- letting go -------------------------------------------------------------------------------------------

    def _drop_graphs(self) -> None:
        if self.graphs:
            torch.cuda.synchronize()
            for g in self.graphs.values():
                g.reset()
            self.graphs.clear()

    def close(self) -> None:
        """the graphs reset and every buffer let go (a cache still bound keeps the arena's rows it views)"""
        self._drop_graphs()
        self.W.clear()
        self.B = self.S = self.A = self.X = self.P = None
        self.owner = None
        self.bound = [None] * len(self.sparse)
        self.held.clear()
