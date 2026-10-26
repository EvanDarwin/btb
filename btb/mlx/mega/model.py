# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A model's pass as its program replayed on the GPU (`replay`): the weight buffers and constants its program reads,
gathered once, and a program per pass shape built from the family's layers (`Family.mega_layer`), each composed of a
`Block`'s parts, every instruction a dispatch of as many threadgroups as its work fills."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeAlias, TypeGuard

import numpy as np
import torch

from ...kinds import LayerKind, Quant, quant_key
from ...metal import paged
from .. import aluminum
from ..attn import tree_meta
from ..backend import Weight
from ..core import mx, to_mx
from ..stored import STORED
from .kernel import EMBED_OP, MV_OP, OpName, head_groups, stored_groups
from .program import CONSTS, SCR, Program, Refs, Tensor, align, kv
from .replay import THREADS, KvSlot, Replay

if TYPE_CHECKING:
    import mlx.core as mx_

    from ...engine.model import StreamedTextModel
    from ..delta import Heads

ROWS_MAX = 16  # a pass's rows: the matvec's tile holds 16
SPL = 16  # the attention segments a pass may read, 16 x ATTN_BLOCK cache rows
MV_KT = 64  # the matvec's columns a step
AMP = 64  # the argmax's parts a row
TG_MEMORY = 32768  # a threadgroup's memory on every Apple GPU family (MTLDevice.maxThreadgroupMemoryLength)

ConstKey: TypeAlias = tuple[str, int]  # a packed constant: what it is, and the module it is read from
# a weight's references: its first stream (the instruction's A), then the rest of what its kernels read (D, E, F)
WeightRefs: TypeAlias = tuple[Tensor, ...]


def refs_of(w: WeightRefs, x: Tensor | None, out: Tensor) -> Refs:
    """an instruction's references over weight `w` beside `x` (B) and `out` (C)"""
    return (w[0], x, out, *w[1:])


class NormModule(Protocol):
    """a block's RMSNorm as the megakernel reads it: its weight (its epsilon `variance_epsilon` or `eps`)"""

    weight: torch.Tensor


class LinearModule(Protocol):
    """a host linear, its weight on the GPU as `mx` (`_HostLinear`)"""

    mx: Weight | None


Linear: TypeAlias = Weight | LinearModule  # a weight, or a host linear holding one
Linears: TypeAlias = Linear | tuple[Linear, ...]  # one, or several into one output


class AttentionModule(Protocol):
    """a layer's attention as the dense block reads it (its q/k norms, where it has them, as `q_norm`, `k_norm`)"""

    scaling: float
    o_proj: LinearModule
    _mx_qkv: Weight | None  # q, k and v's adjacent slots as one weight (`_mlx_fuse`)


class MlpModule(Protocol):
    """a layer's gated MLP as the dense block reads it"""

    down_proj: LinearModule
    _mx_gu: Weight | None  # gate and up's adjacent slots as one weight (`_mlx_fuse`)


class DenseLayer(Protocol):
    """a layer of the plain pre-norm block"""

    input_layernorm: NormModule
    post_attention_layernorm: NormModule
    self_attn: AttentionModule
    mlp: MlpModule


class SandwichLayer(DenseLayer, Protocol):
    """a layer of the sandwich block (Gemma's): the attention's output normed (post_attention_layernorm) before its
    add, and the MLP's input and output normed besides"""

    pre_feedforward_layernorm: NormModule
    post_feedforward_layernorm: NormModule


class ConvModule(Protocol):
    """the DeltaNet's depthwise conv: its weight [C, 1, K], its bias where it has one"""

    weight: torch.Tensor
    bias: torch.Tensor | None


class DeltaNetModule(Protocol):
    """a gated DeltaNet's mixer as the megakernel reads it (Qwen3.5's `linear_attn`): its four input projections (one
    weight where their slots are adjacent, `_mx_in`), its conv, decay and gated norm, its out projection"""

    in_proj_qkv: LinearModule
    in_proj_z: LinearModule
    in_proj_b: LinearModule
    in_proj_a: LinearModule
    out_proj: LinearModule
    conv1d: ConvModule
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: NormModule
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    _mx_in: Weight | None


class DeltaLayer(Protocol):
    """a hybrid's DeltaNet layer: the pre-norm block with the DeltaNet in the attention's place"""

    input_layernorm: NormModule
    post_attention_layernorm: NormModule
    linear_attn: DeltaNetModule
    mlp: MlpModule


class DeltaShape(NamedTuple):
    """a hybrid's DeltaNet layers' sizes, every layer's alike: the conv's channels and width, the value and key heads
    and their dims, the keys' width"""

    C: int
    K: int
    Hv: int
    Hk: int
    dk: int
    dv: int
    key_dim: int


class DeltaRefs(NamedTuple):
    """where a DeltaNet layer's pass leaves what a verify's commit reads (scratch offsets): the conv's output xc [T, C]
    (q, k and v), g and beta [T, 2Hv], each node's conv window [T, C, K]"""

    xc: int
    gb: int
    win: int


def _holds(layer: object, proto: type) -> bool:
    """`layer` has every member `proto` and its protocol bases declare. hasattr, not a runtime protocol check: that
    reads attributes statically, and a torch module's submodules sit in its `_modules` behind `__getattr__`"""
    return all(hasattr(layer, n) for c in proto.__mro__ for n in inspect.get_annotations(c))


def dense_layer(layer: object) -> TypeGuard[DenseLayer]:
    """`layer` is the plain block's"""
    return _holds(layer, DenseLayer)


def sandwich_layer(layer: object) -> TypeGuard[SandwichLayer]:
    """`layer` is the sandwich block's"""
    return _holds(layer, SandwichLayer)


def delta_layer(layer: object) -> TypeGuard[DeltaLayer]:
    """`layer` is a hybrid's DeltaNet layer"""
    if not _holds(layer, DeltaLayer):
        return False
    la: object = getattr(layer, "linear_attn", None)
    return _holds(la, DeltaNetModule)


def mv_items(rows: int, per: int) -> int:
    """the threadgroups a matvec of `rows` takes, its simdgroups `per` rows each (the bf16 tile's 8): every simdgroup
    its own rows, once"""
    tiles = -(-rows // per)
    return max(1, -(-tiles // 8))


class Block:
    """What a family's layer is made of in the megakernel (`Family.mega_layer`): the residual stream, read through a
    norm and added to by `add`, and the ops over it. An add waits for the next norm, which makes both in one op."""

    def __init__(self, mp: MegaPass, prog: Program, splits: int, keep: bool) -> None:
        self.mp, self.p = mp, prog
        self.T = prog.T
        self.splits = splits
        self.keep = keep  # a verify pass: the DeltaNet's states left for its commit (`delta`)
        self.h: Tensor | None = None  # the residual
        self.y: Tensor | None = None  # an add to it not yet made

    def dense(self, i: int) -> DenseLayer:
        """layer i's host modules, read as the plain block's"""
        layer = self.mp.sm.host[i]
        if not dense_layer(layer):
            raise TypeError(f"[mega] layer {i} ({type(layer).__name__}) is not the plain block's")
        return layer

    def deltanet(self, i: int) -> DeltaLayer:
        """layer i's host modules, read as a hybrid's DeltaNet layer"""
        layer = self.mp.sm.host[i]
        if not delta_layer(layer):
            raise TypeError(f"[mega] layer {i} ({type(layer).__name__}) is not a DeltaNet layer")
        return layer

    def sandwich(self, i: int) -> SandwichLayer:
        """layer i's host modules, read as the sandwich block's"""
        layer = self.mp.sm.host[i]
        if not sandwich_layer(layer):
            raise TypeError(f"[mega] layer {i} ({type(layer).__name__}) is not the sandwich block's")
        return layer

    @staticmethod
    def qkv(at: AttentionModule) -> Linears:
        """q, k and v's weight as one where it is (their adjacent slots, or the module's own fused projection: Phi-3's),
        else the three into one output (a GGUF's v of another type than q and k)"""
        if at._mx_qkv is not None:
            return at._mx_qkv
        return _one_or_parts(at, "qkv_proj", ("q_proj", "k_proj", "v_proj"))

    @staticmethod
    def delta_in(la: DeltaNetModule) -> Linears:
        """the DeltaNet's four input projections (the conv's input, z, b and a) as one weight where their slots are
        adjacent, else the four into one output"""
        if la._mx_in is not None:
            return la._mx_in
        return la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a

    @staticmethod
    def gate_up(mlp: MlpModule) -> Linears:
        """gate's and up's weight as one where it is (adjacent slots, or Phi-3's fused projection), else the two into
        one output"""
        if mlp._mx_gu is not None:
            return mlp._mx_gu
        return _one_or_parts(mlp, "gate_up_proj", ("gate_proj", "up_proj"))

    def embed(self) -> None:
        """the rows' embeddings, scaled where the family scales them (Gemma's sqrt(hidden))"""
        mp, H = self.mp, self.mp.H
        h = self.p.scratch(H)
        scale = mp.sm.embed_scale
        n = (H, int(scale is not None), 0)
        op = OpName.EMBED if mp.embed_q is None else EMBED_OP[mp.embed_q]
        self.p.emit(op, self.T, refs_of(mp.embed, None, h), n, scale or 0.0, writes=[h])
        self.h = h

    def norm(self, mod: NormModule) -> Tensor:
        """the residual through norm `mod` (an add pending made with it, in one op)"""
        assert self.h is not None  # the pass starts at `embed`
        H = self.mp.H
        w, eps, f32 = self.mp.norm_weight(mod)
        x = self.p.scratch(H)
        refs: Refs
        if self.y is None:
            refs = (self.h, w, x)
            self.p.emit(OpName.RMSNORM, self.T, refs, (H, int(f32), 0), eps, reads=[self.h], writes=[x])
        else:
            h2 = self.p.scratch(H)
            refs = (self.h, self.y, h2, x, w)
            n = (H, int(not self.mp.fused) | (int(f32) << 1), 0)
            self.p.emit(OpName.ADD_RMSNORM, self.T, refs, n, eps, reads=[self.h, self.y], writes=[h2, x])
            self.h, self.y = h2, None
        return x

    def add_normed(self, y: Tensor, mod: NormModule) -> None:
        """norm `mod` of y added to the residual at once (the sandwich block's: the delta normed, not the sum)"""
        assert self.h is not None
        if self.y is not None:
            raise ValueError("[mega] a sandwich add over an add not yet made")
        H = self.mp.H
        w, eps, f32 = self.mp.norm_weight(mod)
        out = self.p.scratch(H)
        refs = (self.h, y, out, None, w)
        self.p.emit(OpName.SANDWICH_ADD, self.T, refs, (H, int(f32), 0), eps, reads=[self.h, y], writes=[out])
        self.h = out

    def add(self, y: Tensor) -> None:
        """y added to the residual, by the next norm"""
        if self.y is not None:
            raise ValueError("[mega] two adds to the residual with no norm between them")
        self.y = y

    def linear(self, x: Tensor, w: Linears) -> Tensor:
        """x . w^T for a slot weight `w` (bf16, or a stored type the kernel multiplies as stored; or a host linear
        holding one), several end to end into one output, a piece of each row a weight"""
        got = [self.mp.weight(v) for v in (w if isinstance(w, tuple) else (w,))]
        width = sum(rows for _, rows, _, _ in got)
        y = self.p.scratch(width)
        pieces, at = [], 0
        for wt, rows, cols, q in got:
            if x.cols != cols or (q is None and cols % 8):
                raise ValueError(f"[mega] a linear of {cols} columns over rows of {x.cols}")
            out = y if len(got) == 1 else y.piece(at, rows)
            ld = 0 if len(got) == 1 else width
            op, items = (OpName.MATVEC, mv_items(rows, 8)) if q is None else (MV_OP[q], stored_groups(rows, self.T))
            self.p.emit(op, items, refs_of(wt, x, out), (rows, cols, ld << 1), reads=[x], writes=[out])
            pieces.append(out)
            at += rows
        if len(pieces) > 1:
            y.parts = tuple(pieces)
        return y

    def attention(self, i: int, qkv: Tensor, at: AttentionModule) -> Tensor:
        """layer i's attention over q, k and v end to end in `qkv`: the q/k norms where it has them and the rope (its
        layer type's), the pass's rows into the arena, each node over its cache rows and its path, the partials
        folded; [T, Hq * hd]"""
        mp, p, T = self.mp, self.p, self.T
        Hq, Hk, hd = mp.Hq, mp.Hk, mp.hd
        qk = mp.qk_weights(at)
        if qk is not None and mp.rscale != 1.0:
            raise ValueError("[mega] q/k norms with a scaled rope: the fused step scales after the kernel")
        if qkv.cols != (Hq * (1 + mp.gated) + 2 * Hk) * hd:
            raise ValueError(f"[mega] a q/k/v projection of {qkv.cols} columns")
        q = p.scratch(Hq * hd)
        K, V = Tensor((kv(i, 0), 0)), Tensor((kv(i, 1), 0))
        row = mp.rope_row.get(mp.sm.layer_types[i], 0)
        f32 = qk is not None and qk[2]
        # the norms the fused kernel's where the step takes its kernels, else MLX's own, as the step's unfused ops
        fk, mlx = qk is not None and mp.fused, qk is not None and not mp.fused
        n = (
            (mp.rd // 2) | (fk << 31),
            Hq | (row << 16) | (mp.gated << 30) | (mlx << 31),
            Hk | (hd << 16) | (f32 << 31),
        )
        f0 = qk[1] if qk is not None else mp.rscale
        refs = (qkv, qk[0] if qk is not None else None, q, K, V)
        p.emit(OpName.QKROPE, T * (Hq + 2 * Hk), refs, n, f0, reads=[qkv], writes=[q, K, V])
        # the partials, and right behind them (A_OFF) the rows of the nodes of one segment as the attention writes them:
        # a pass of one segment's output, else the fold's to copy beside the others' folded partials
        part = p.scratch(mp.part_bytes, 1, rows=1)
        pl = Tensor((SCR, part.ref[1] + mp.pl_off), scratch=True)
        one = p.scratch(Hq * hd)
        refs = (q, K, part, pl, V)
        scale = float(at.scaling)
        wrow = mp.wins.index(mp.windows[i]) << 16  # the layer's window's row of the metadata
        tm = T | wrow
        pick = aluminum.tree_pick(hd, mp.G, Hk, T, self.splits, THREADS // 32)
        if pick.kind is aluminum.TreeKind.NODES:
            nhg = mp.G // mp.gh
            n = (Hk * nhg, self.splits, tm)
            p.emit(OpName.ATTN, Hk * nhg * self.splits * T, refs, n, scale, reads=[q, K, V], writes=[part, pl, one])
        else:
            phased = pick.kind is aluminum.TreeKind.PHASED
            tiles = -(-(T * mp.G) // pick.rows)
            n = (tiles, Hk, self.splits | wrow | ((phased and pick.rows == 16) << 31))
            op = OpName.ATTN_PHASED if phased else OpName.ATTN_TREE
            p.emit(op, tiles * Hk * self.splits, refs, n, scale, reads=[q, K, V], writes=[part, pl, one])
        # the fold's output placed whatever the segments, so every shape lays the scratch out alike (`final`)
        a = p.scratch(Hq * hd)
        if self.splits == 1:
            return one
        n = (Hk, self.splits, tm)
        p.emit(OpName.FOLD, (T * Hq * hd + 255) // 256, (part, pl, a), n, reads=[part, pl, one], writes=[a])
        return a

    def gate(self, a: Tensor, qkv: Tensor) -> Tensor:
        """the attention's output `a` times the sigmoid of its heads' gates (Qwen3.5's output gate), each q head's
        gate the second half of its columns in `qkv`"""
        mp = self.mp
        W = mp.Hq * mp.hd
        y = self.p.scratch(W)
        n = (W, mp.hd, qkv.cols)
        self.p.emit(OpName.GATE, (self.T * W + 255) // 256, (a, qkv, y), n, reads=[a, qkv], writes=[y])
        return y

    def delta(self, i: int, p: Tensor, la: DeltaNetModule) -> Tensor:
        """layer i's DeltaNet over its projections `p` (the conv's input, z, b and a end to end): the conv along each
        node's path, the gated delta rule from each node's parent's state, the gated norm; [T, Hv * dv] bf16 for the out
        projection. A pass that is not a verify writes its last node's states back over the layer's (the cache's, bound
        at kv(i, 0) and kv(i, 1)); a verify leaves them, its windows and the rule's inputs in the scratch
        (`DeltaRefs`) for its commit"""
        mp, prog, T = self.mp, self.p, self.T
        d = mp.delta
        if d is None:
            raise ValueError("[mega] a DeltaNet layer in a model without them")
        if p.cols != d.C + d.Hv * d.dv + 2 * d.Hv:
            raise ValueError(f"[mega] DeltaNet projections of {p.cols} columns")
        consts, bias = mp.delta_consts(la)
        conv, rec = Tensor((kv(i, 0), 0)), Tensor((kv(i, 1), 0))
        # placed whatever the pass, so every shape lays the scratch out alike
        xc, gb, win = prog.scratch(d.C, 4), prog.scratch(2 * d.Hv, 4), prog.scratch(d.C * d.K, 4)
        mp.delta_refs(i, DeltaRefs(xc.ref[1], gb.ref[1], win.ref[1]))
        n = (d.C, p.cols, d.Hv | (bias << 16) | (self.keep << 17))
        items = (max(d.C, T * d.Hv) + 255) // 256
        prog.emit(OpName.DELTA_CONV, items, (p, consts, xc, gb, conv, win), n, reads=[p], writes=[xc, gb])
        core = prog.scratch(d.Hv * d.dv, 4)
        n = (d.Hv, d.Hk, int(self.keep))
        prog.emit(OpName.DELTA_RULE, d.Hv * d.dv // 8, (xc, gb, core, None, rec), n, reads=[xc, gb], writes=[core])
        y = prog.scratch(d.Hv * d.dv)
        w, eps = mp.delta_norm(la)
        n = (d.Hv, p.cols, d.C)
        items = (T * d.Hv + 7) // 8
        prog.emit(OpName.DELTA_NORM, items, (core, p, y, None, w), n, eps, reads=[core, p], writes=[y])
        return y

    def gated(self, gu: Tensor) -> Tensor:
        """act(gate) * up over gate and up end to end in `gu`, as the step makes it: silu in the fused kernel (gate and
        up one weight) or in MLX's own ops (the step's split gate and up, an unfused family), gelu-tanh in MLX's"""
        if self.mp.act in ("silu", "swish"):
            how = 0 if self.mp.fused_mlp and not gu.parts else 1
        elif self.mp.act in ("gelu_pytorch_tanh", "gelu_new"):
            how = 2
        else:
            raise ValueError(f"[mega] the gated MLP's {self.mp.act} is not in the kernel")
        In = gu.cols // 2
        mid = self.p.scratch(In)
        items, n = (self.T * In + 255) // 256, (In, self.T, how)
        self.p.emit(OpName.GATED, items, (gu, None, mid), n, reads=[gu], writes=[mid])
        return mid

    def final(self, argmax: bool) -> None:
        """the final norm (in float32, the pending add made in it), the head, and the argmax a row"""
        assert self.h is not None
        mp, p, T, H = self.mp, self.p, self.T, self.mp.H
        w, eps = mp.final_norm
        hn = p.scratch(H, 4)
        res = p.scratch(H)  # the last layer's output, a layer hook's (`MegaPass.hidden`)
        if mp._frozen and res.ref[1] != mp.hidden_off:
            raise ValueError("[mega] a pass shape places the last layer's output where the widest pass does not")
        mp.hidden_off = res.ref[1]
        y = self.y
        reads = [self.h] if y is None else [self.h, y]
        n = (H, int(y is not None), 0)
        p.emit(OpName.FNORM, T, (self.h, w, hn, res, y), n, eps, reads=reads, writes=[hn, res])
        logits = p.scratch(mp.V, 4)
        if mp._frozen and logits.ref[1] != mp.logits_off:
            # a replayed shape's logits are read at the widest pass's offset (`logits`)
            raise ValueError("[mega] a pass shape places its logits where the widest pass does not")
        mp.logits_off = logits.ref[1]
        if mp.head_q is None:  # the step's own float32 matvec over the bf16 head (`kernel.head_source`)
            items = head_groups(mp.V, T)
            p.emit(OpName.HEAD, items, refs_of(mp.head, hn, logits), (mp.V, H, 0), reads=[hn], writes=[logits])
        else:  # the type's own chain over float32 rows, as the step multiplies a stored head
            items = stored_groups(mp.V, T)
            refs = refs_of(mp.head, hn, logits)
            p.emit(MV_OP[mp.head_q], items, refs, (mp.V, H, 1), reads=[hn], writes=[logits])
        amax = p.scratch(2 * AMP, 4)
        if argmax:
            n = (mp.V, AMP, p.R * AMP)
            p.emit(OpName.ARGMAX_PART, T * AMP, (logits, None, amax), n, reads=[logits], writes=[amax])
            p.emit(OpName.ARGMAX, T, (amax,), (0, AMP, p.R * AMP), reads=[amax])


class MegaPass:
    """A model's pass of 1..ROWS_MAX rows over an arena or a paged cache replayed on the GPU, bit-exact with the fused
    path: its weights in slot buffers, its constants packed once, a write-once scratch for the widest pass, and a
    program per pass shape (rows, attention segments, argmax) built from the family's layers, encoded once and replayed
    over whichever cache a pass binds."""

    def __init__(self, sm: StreamedTextModel) -> None:
        from ...engine.families import act_name

        if not sm.fam.mega:
            raise ValueError(f"[mega] {sm.fam.name}'s layers are not composed for the megakernel")
        hh = sm.head_host
        hw = hh.mx if hh is not None else None
        if hw is None:
            raise ValueError("[mega] the head is not on the host")
        m = mx
        c = sm.cfg
        self.sm = sm
        self.L = int(sm.L)
        self.H = int(c.hidden_size)
        self.Hq = int(c.num_attention_heads)
        self.Hk = int(getattr(c, "num_key_value_heads", None) or self.Hq)
        # a hybrid's DeltaNet layers beside its attention ones, the attention's sizes read off the first of those
        self.linear = [sm.layer_types[i] == LayerKind.LINEAR for i in range(self.L)]
        att = next((i for i in range(self.L) if not self.linear[i]), None)
        if att is None:
            raise ValueError("[mega] no attention layer to read the heads off")
        self.hd = int(sm.host[att].self_attn.head_dim)
        self.gated = bool(sm.fam.attn_gate)  # each q head's output gate beside it in the q projection (Qwen3.5's)
        self.delta = self._delta_shape() if any(self.linear) else None
        self._delta_refs: dict[int, DeltaRefs] = {}
        self.V = int(hw.shape[0])
        self.SPL, self.ROWS = SPL, ROWS_MAX
        if self.hd not in aluminum.HEADS or self.Hq % self.Hk or self.Hq // self.Hk > aluminum.MAXG or self.H % 8:
            raise ValueError(
                f"[mega] the kernel takes heads of {aluminum.HEADS} dims, up to {aluminum.MAXG} query heads a kv head, "
                "and a width of 8k"
            )
        # the rope: a row of frequencies a layer type (one for most families), turning the first rd dims of a head
        freqs, rd, rscale = sm._mlx_rope()
        if rd > self.hd or rd % 2:
            raise ValueError(f"[mega] a rotary of {rd} dims over a head of {self.hd}")
        self.rd, self.rscale = int(rd), float(rscale)
        types = sorted(freqs) if isinstance(freqs, dict) else []
        self.rope_row = {lt: j for j, lt in enumerate(types)}
        table = m.concatenate([freqs[lt] for lt in types]) if isinstance(freqs, dict) else freqs
        self.freqs = table.astype(m.float32).reshape(-1)
        if self.freqs.size < 8:
            self.freqs = m.pad(self.freqs, (0, 8 - int(self.freqs.size)))
        # the step this pass is bit-exact with: the fused kernels (`_forward_mlx`'s `fuse`), or MLX's own ops; the
        # hybrid's step (`_forward_mlx_hybrid`) takes MLX's ops but the fused silu * up wherever gate and up are one
        self.fused = bool(getattr(sm, "mlx_fused_kernels", True)) and sm.fam.fused_step and self.rd == self.hd
        self.fused_mlp = self.fused or sm.fam.hybrid
        self.act = act_name(c)
        # each layer's attention window, as the step reads it (a sandwich family's sliding layers), and the distinct
        # ones: a row of the node metadata each (`metas`)
        sw = int(getattr(c, "sliding_window", 0) or 0) if sm.fam.sandwich else 0
        self.windows = [sw if sw and sm.layer_types[i] == LayerKind.SLIDING else None for i in range(self.L)]
        self.wins = sorted(set(self.windows), key=lambda w: -1 if w is None else w)
        self.G = self.Hq // self.Hk
        # aluminum's decode: the query heads a threadgroup and V's keys a stage, its threadgroup memory's layout
        self.gh, self.ch = aluminum.decode_shape(self.hd, self.G)
        self.vt, self.attn_tgm = aluminum.tgm_layout(self.hd, self.gh, self.ch)
        # the attention's partials a layer, for the widest pass: m and l a (row, query head, segment), then o behind them
        pm = ROWS_MAX * self.Hq * SPL * 4
        self.pl_off = align(pm)
        self.pacc_off = 2 * align(pm)
        self.part_bytes = self.pacc_off + ROWS_MAX * self.Hq * SPL * self.hd * 4
        self.bufs: list[mx_.array] = []
        self._bid: dict[int, int] = {}
        self._cparts: list[mx_.array] = []
        self._csize = 0
        self._consts: dict[ConstKey, Tensor] = {}
        self._frozen = False
        self.logits_off = 0
        self.hidden_off = 0
        self.head, self.head_q = self._ref(hw), hw.q
        if sm.head_key == sm.prefix + "embed_tokens.weight":
            self.embed, self.embed_q = self.head, self.head_q
        else:
            sm._mlx_embed()
            lin = sm.mlx_state.embed_lin
            if lin is None or lin.mx is None or getattr(lin.mx, "sh", None) is None:
                raise ValueError("[mega] the embedding table is not in a slot buffer")
            self.embed, self.embed_q = self._ref(lin.mx), lin.mx.q
        fn: NormModule = sm.norm
        centered = sm.fam.norm_centered
        self.final_norm = (self._const(("final", 0), lambda: _norm_array(fn, True, centered)), _eps(fn))
        # the widest pass, built once: what it reads gathers the weights and the constants, its scratch the size
        prog = self._program(ROWS_MAX, SPL, True)
        self._frozen = True
        self.nb = len(self.bufs)
        # the constants and the scratch whole pages, as the replay aliases them
        consts = [p.reshape(-1).view(m.uint8) for p in self._cparts]
        self.consts = m.concatenate([*consts, m.zeros((paged(self._csize) - self._csize,), dtype=m.uint8)])
        self.scr = m.zeros((paged(prog.size),), dtype=m.uint8)
        m.eval(self.consts, self.scr, self.freqs)
        # every op a pass shape compiles to: the widest's, a one-block pass's (its attention folding its own), and every
        # attention (a shape's the one `aluminum.tree_pick` picks; the phased one at head 256 alone)
        attns = {OpName.ATTN, OpName.ATTN_TREE} | ({OpName.ATTN_PHASED} if self.hd == 256 else set())
        self.ops = frozenset(prog.ops) | self._program(ROWS_MAX, 1, True).ops | attns
        self.instructions = len(prog.rows)  # the most a pass has: the widest one's
        self.replay = Replay(self)

    # -- what the program reads, gathered by the widest pass's build --

    def _ref(self, w: Weight) -> WeightRefs:
        """a weight in a slot or pool buffer, bf16 or of a stored type the kernel reads (`STORED`): a reference each
        of what its kernels read (`kernel.stored_inputs`), its streams in its buffer and its tables in the constants"""
        sh = getattr(w, "sh", None)
        if sh is None or w.packed is not None:
            raise ValueError("[mega] a weight outside a slot buffer")
        if w.q is not None and w.q not in STORED:
            raise ValueError(f"[mega] {w.q} weights are not in the kernel's ops")
        k = None if w.q is None else STORED[w.q]
        offs = w.offs or (w.off,)
        if k is not None and len(offs) != len(k.streams):
            raise ValueError(f"[mega] {w.q} weights outside a pool block: their streams are repacked in the graph")
        i = self._bid.get(id(sh.mx))
        if i is None:
            if self._frozen:
                raise ValueError("[mega] a pass reads a weight buffer the widest pass did not")
            i = self._bid[id(sh.mx)] = len(self.bufs)
            self.bufs.append(sh.whole)
        cols = int(w.shape[1])
        streams = tuple(Tensor((i, int(o)), cols) for o in offs)
        if w.q is None or k is None:
            return streams
        return streams + tuple(self._const((f"{quant_key(w.q)}_{t.name}", 0), t.make) for t in k.tables)

    def weight(self, w: Linear | None) -> tuple[WeightRefs, int, int, Quant | None]:
        """a slot weight (or a host linear holding one): its references, rows, columns and stored type (None: bf16)"""
        wt = w if w is None or isinstance(w, Weight) else w.mx
        if wt is None:
            raise ValueError("[mega] a linear with no weight on the GPU")
        return self._ref(wt), int(wt.shape[0]), int(wt.shape[1]), wt.q

    def _const(self, key: ConstKey, make: Callable[[], mx_.array]) -> Tensor:
        """a constant packed once (4-byte aligned) behind the others, by `key`"""
        t = self._consts.get(key)
        if t is None:
            if self._frozen:
                raise ValueError("[mega] a pass reads a constant the widest pass did not")
            a = make()
            mx.eval(a)
            t = self._consts[key] = Tensor((CONSTS, self._csize))
            self._cparts.append(a)
            self._csize += a.nbytes
            if self._csize % 4:
                pad = mx.zeros((4 - self._csize % 4,), dtype=mx.uint8)
                self._cparts.append(pad)
                self._csize += pad.nbytes
        return t

    def _delta_shape(self) -> DeltaShape:
        """the DeltaNet layers' sizes, read off the first (every layer's alike), held to what the ops take: the rule's
        key dim a multiple of 32 (its lanes), the value dim a multiple of 8 (its simdgroups) up to 128 (a simdgroup's
        norm), the conv's channels the keys twice and the values"""
        layer = next(self.sm.host[i] for i in range(self.L) if self.linear[i])
        if not delta_layer(layer):
            raise ValueError(f"[mega] a linear-attention layer ({type(layer).__name__}) that is not a DeltaNet's")
        la = layer.linear_attn
        w = la.conv1d.weight
        d = DeltaShape(
            C=int(w.shape[0]),
            K=int(w.shape[-1]),
            Hv=int(la.num_v_heads),
            Hk=int(la.num_k_heads),
            dk=int(la.head_k_dim),
            dv=int(la.head_v_dim),
            key_dim=int(la.key_dim),
        )
        if d.dk % 32 or d.dv % 8 or d.dv > 128 or d.Hv % d.Hk or d.K < 2:
            raise ValueError(f"[mega] a DeltaNet of {d} is not in the kernel's ops")
        if d.key_dim != d.Hk * d.dk or d.C != 2 * d.key_dim + d.Hv * d.dv:
            raise ValueError(f"[mega] a DeltaNet conv of {d.C} channels over keys of {d.key_dim}")
        return d

    def delta_consts(self, la: DeltaNetModule) -> tuple[Tensor, bool]:
        """a DeltaNet layer's constants end to end as `op_delta_conv` reads them, float32 as the step keeps them: the
        conv's weight [C, K] and bias [C] (where it has one), a_log and dt_bias [Hv]; and whether it has the bias"""
        bias = la.conv1d.bias is not None

        def make() -> mx_.array:
            parts = [la.conv1d.weight.squeeze(1), *([la.conv1d.bias] if la.conv1d.bias is not None else [])]
            parts += [la.A_log, la.dt_bias]
            return to_mx(torch.cat([t.data.detach().float().reshape(-1) for t in parts]).contiguous())

        return self._const(("delta", id(la)), make), bias

    def delta_norm(self, la: DeltaNetModule) -> tuple[Tensor, float]:
        """a DeltaNet layer's gated norm's weight, float32 as the step keeps it (not centered), and its epsilon"""
        return self._const(("delta_norm", id(la)), lambda: _norm_array(la.norm, True, False)), _eps(la.norm)

    def delta_refs(self, i: int, refs: DeltaRefs) -> None:
        """where layer i's pass leaves its verify's state (`checkpoints`): recorded by the widest pass, every other
        shape held to it"""
        got = self._delta_refs.setdefault(i, refs)
        if got != refs:
            raise ValueError(f"[mega] a pass shape places layer {i}'s DeltaNet state where the widest pass does not")

    def checkpoints(self, i: int, T: int) -> tuple[mx_.array, Heads]:
        """what the last pass (a verify of T nodes) left for DeltaNet layer i's commit, views of the scratch the next
        pass overwrites: each node's conv window [T, C, K] and the rule's heads (q, k [T, Hk, dk] raw, v [T, Hv, dv], g
        and beta [T, Hv]), as delta.py `delta_tree_step` hands them to `delta_chain_state`"""
        d, r = self.delta, self._delta_refs[i]
        assert d is not None  # a layer with refs is a DeltaNet layer
        f32 = mx.float32

        def view(off: int, *shape: int) -> mx_.array:
            n = math.prod(shape) * 4
            return self.scr[off : off + n].view(f32).reshape(*shape)

        xc = view(r.xc, T, d.C)
        gb = view(r.gb, T, 2 * d.Hv)
        kd = d.key_dim
        q = xc[:, :kd].reshape(T, d.Hk, d.dk)
        k = xc[:, kd : 2 * kd].reshape(T, d.Hk, d.dk)
        v = xc[:, 2 * kd :].reshape(T, d.Hv, d.dv)
        return view(r.win, T, d.C, d.K), (q, k, v, gb[:, : d.Hv], gb[:, d.Hv :])

    def norm_weight(self, mod: NormModule) -> tuple[Tensor, float, bool]:
        """a block norm's weight as the step keeps it (bf16, or a centered norm's 1 + w in float32), its epsilon, and
        whether it is the float32 one"""
        f32 = self.sm.fam.norm_centered
        return self._const(("norm", id(mod)), lambda: _norm_array(mod, f32, f32)), _eps(mod), f32

    def qk_weights(self, at: AttentionModule) -> tuple[Tensor, float, bool] | None:
        """the q and k norms' weights end to end as the step keeps them (bf16, or a centered norm's 1 + w in float32),
        their epsilon, and whether they are the float32 ones; None for an attention without them"""
        qn, kn = getattr(at, "q_norm", None), getattr(at, "k_norm", None)
        if qn is None or kn is None:
            return None
        f32 = self.sm.fam.norm_centered
        t = self._const(("qk", id(at)), lambda: mx.concatenate([_norm_array(qn, f32, f32), _norm_array(kn, f32, f32)]))
        return t, _eps(qn), f32

    def metas(self, past: int, parents: Sequence[int]) -> tuple[mx_.array, mx_.array, int]:
        """a pass's node metadata, a row of it a distinct window (`wins`) end to end, the nodes' paths, and the
        attention blocks the widest row reads"""
        rows = [tree_meta(past, parents, window=w) for w in self.wins]
        meta = rows[0][0] if len(rows) == 1 else mx.concatenate([r[0] for r in rows])
        return meta, rows[0][1], max(r[2] for r in rows)

    # -- the kernel --

    def defs(self, xr: int) -> dict[str, int]:
        """the ops' compile-time constants for passes of up to xr rows"""
        return {
            "XROWS": xr,
            "MV_KT": MV_KT,
            "ATT_G": self.G,
            "ATT_D": self.hd,
            "ATT_GH": self.gh,
            "ATT_NHG": self.G // self.gh,
            "ATT_CH": self.ch,
            "ATT_VT": self.vt,
            "PACC_OFF": self.pacc_off,
            "A_OFF": align(self.part_bytes),
            **(
                {}
                if self.delta is None
                else {
                    "DELTA_K": self.delta.K,
                    "DELTA_DK": self.delta.dk,
                    "DELTA_DV": self.delta.dv,
                    "DELTA_NPT": self.delta.dk // 32,
                }
            ),
        }

    def tgm(self, xr: int) -> dict[OpName, int]:
        """each op's threadgroup memory at xr rows: the bf16 matvec's tiles staged a simdgroup each, the attention's
        (`aluminum.tgm_layout`, `aluminum.tree_tgm`, `aluminum.phased_tgm`), the rest their reductions' (a head kernel
        declares its own, the step's)"""
        small = max((8 + self.hd) * 4, 2 * 256 * 4)
        need = dict.fromkeys(OpName, small)
        need[OpName.MATVEC] = max(small, 8 * (8 + xr) * MV_KT * 2)
        need[OpName.ATTN] = max(small, self.attn_tgm)
        need[OpName.ATTN_TREE] = max(small, aluminum.tree_tgm(self.hd, THREADS // 32))
        need[OpName.ATTN_PHASED] = max(small, aluminum.phased_tgm(16))
        if max(need.values()) > TG_MEMORY:
            raise ValueError(
                f"[mega] an op takes {max(need.values())} bytes of threadgroup memory, a threadgroup has {TG_MEMORY}"
            )
        return need

    def _program(self, T: int, splits: int, argmax: bool, keep: bool = False) -> Program:
        """a pass's program: every layer and the head; `keep` a verify's, its DeltaNet states left for its commit"""
        p = Program(T, ROWS_MAX)
        b = Block(self, p, splits, keep)
        b.embed()
        for i in range(self.L):
            self.sm.fam.mega_layer(b, i)
        b.final(argmax)
        return p

    def table(self, T: int, splits: int, argmax: bool = True, keep: bool = False) -> np.ndarray:
        """the program of a pass of T rows (`splits` attention blocks; `keep` a verify's) as the table the replay
        encodes, uint32 [instructions, W]"""
        return self._program(T, splits, argmax, keep).table(self.nb)

    def logits(self, T: int) -> mx_.array:
        """the logits [T, V] float32 the last pass left in the scratch (a pass without its argmax)"""
        o = self.logits_off
        return self.scr[o : o + T * self.V * 4].view(mx.float32).reshape(T, self.V)

    def hidden(self, T: int) -> mx_.array:
        """the last layer's output [T, H] bf16 the last pass left in the scratch, as the step hands a layer hook"""
        o = self.hidden_off
        return self.scr[o : o + T * self.H * 2].view(mx.bfloat16).reshape(T, self.H)

    def run(
        self,
        kvs: Sequence[KvSlot],
        cap: int,
        rmap: np.ndarray | None,
        ids: Sequence[int],
        pos: Sequence[int],
        meta: mx_.array,
        path: mx_.array,
        splits: int,
        past: int,
        argmax: bool = True,
        keep: bool = False,
    ) -> np.ndarray:
        """the pass, run and waited for: `kvs` each layer's K then V rows (2L of them) [Hk, cap, D] bf16 - a DeltaNet
        layer's conv and recurrent states in their place, float32 - `rmap` a paged cache's row map (uint32, the
        conversation's rows to the regions' through the pass's own: rows are read and written through it) or None for
        rows where they lie (an arena), `ids` and `pos` the rows' tokens and positions, `meta`/`path` the node
        metadata; the argmax ids uint32 [T] (zeros with `argmax` off, the logits then in the scratch). `keep`: a
        verify, its DeltaNet states left for its commit (`checkpoints`)"""
        if len(kvs) != 2 * self.L:
            raise ValueError(f"[mega] {len(kvs)} K and V regions for {self.L} layers")
        return self.replay.run(kvs, cap, rmap, ids, pos, meta, path, splits, past, argmax, keep)


def _one_or_parts(mod: AttentionModule | MlpModule, fused: str, names: tuple[str, ...]) -> Linears:
    """`mod`'s fused projection `fused` where it has one, else its projections `names`, into one output"""
    one: LinearModule | None = getattr(mod, fused, None)
    if one is not None:
        return one
    parts: list[Linear] = []
    for n in names:
        p: LinearModule | None = getattr(mod, n, None)
        if p is None:
            raise ValueError(f"[mega] neither {fused} nor {n}")
        parts.append(p)
    return tuple(parts)


def _eps(mod: NormModule) -> float:
    return float(getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6)))


def _norm_array(mod: NormModule, f32: bool, centered: bool) -> mx_.array:
    """a norm's weight as the fused path keeps it: bf16, or float32 (1 + weight for a centered norm)"""
    t = mod.weight.data.detach().float()
    if centered:
        t = 1.0 + t
    return to_mx((t if f32 else t.to(torch.bfloat16)).contiguous())
