# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A model's pass as its program replayed on the GPU (`replay`): the weight buffers and constants its program reads,
gathered once, and a program per pass shape built from the family's layers (`Family.mega_layer`), each composed of a
`Block`'s parts, every instruction a dispatch of as many threadgroups as its work fills."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
import torch

from ...kinds import LayerKind, Quant
from ...metal import paged
from ..attn import tree_meta
from ..backend import Weight
from ..core import mx, to_mx
from .kernel import EMBED_OP, MV_OP, STORED, OpName, stored_groups
from .program import CONSTS, SCR, Program, Refs, Tensor, align, kv
from .replay import KvSlot, Replay

if TYPE_CHECKING:
    import mlx.core as mx_

    from ...engine.model import StreamedTextModel

ROWS_MAX = 16  # a pass's rows: the matvec's tile holds 16
SPL = 8  # the attention blocks a pass may read, 8 x ATTN_BLOCK cache rows
MV_KT = 64  # the matvec's columns a step
AMP = 64  # the argmax's parts a row
TG_MEMORY = 32768  # a threadgroup's memory on every Apple GPU family (MTLDevice.maxThreadgroupMemoryLength)

ConstKey: TypeAlias = tuple[str, int]  # a packed constant: what it is, and the module it is read from


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


@dataclass(frozen=True)
class Fusions:
    """the ops a pass makes inside a neighbour instead of as instructions of their own, bit-exact either way: each
    saves a wait between instructions, and costs what the neighbour then does again in every threadgroup and the
    registers its code adds to the whole kernel"""

    fold_in_attention: bool = True  # a one-block pass's partials folded by the attention's own threadgroup


def mv_items(rows: int, per: int) -> int:
    """the threadgroups a matvec of `rows` takes, its simdgroups `per` rows each (the bf16 tile's 8): every simdgroup
    its own rows, once"""
    tiles = -(-rows // per)
    return max(1, -(-tiles // 8))


class Block:
    """What a family's layer is made of in the megakernel (`Family.mega_layer`): the residual stream, read through a
    norm and added to by `add`, and the ops over it. An add waits for the next norm, which makes both in one op."""

    def __init__(self, mp: MegaPass, prog: Program, splits: int) -> None:
        self.mp, self.p = mp, prog
        self.T = prog.T
        self.splits = splits
        self.h: Tensor | None = None  # the residual
        self.y: Tensor | None = None  # an add to it not yet made

    def dense(self, i: int) -> DenseLayer:
        """layer i's host modules, read as the plain block's"""
        return cast("DenseLayer", self.mp.sm.host[i])

    def sandwich(self, i: int) -> SandwichLayer:
        """layer i's host modules, read as the sandwich block's"""
        return cast("SandwichLayer", self.mp.sm.host[i])

    @staticmethod
    def qkv(at: AttentionModule) -> Linears:
        """q, k and v's weight as one where it is (their adjacent slots, or the module's own fused projection: Phi-3's),
        else the three into one output (a GGUF's v of another type than q and k)"""
        if at._mx_qkv is not None:
            return at._mx_qkv
        return _one_or_parts(at, "qkv_proj", ("q_proj", "k_proj", "v_proj"))

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
        self.p.emit(op, self.T, (mp.embed, None, h), n, scale or 0.0, writes=[h])
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
            op, items = (OpName.MATVEC, mv_items(rows, 8)) if q is None else (MV_OP[q], stored_groups(q, rows, self.T))
            self.p.emit(op, items, (wt, x, out), (rows, cols, ld << 1), reads=[x], writes=[out])
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
        q = p.scratch(Hq * hd)
        K, V = Tensor((kv(i, 0), 0)), Tensor((kv(i, 1), 0))
        row = mp.rope_row.get(mp.sm.layer_types[i], 0)
        f32 = qk is not None and qk[2]
        n = ((mp.rd // 2) | ((qk is not None) << 31), Hq | (row << 16), Hk | (hd << 16) | (f32 << 31))
        f0 = qk[1] if qk is not None else mp.rscale
        refs = (qkv, qk[0] if qk is not None else None, q, K, V)
        p.emit(OpName.QKROPE, T * (Hq + 2 * Hk), refs, n, f0, reads=[qkv], writes=[q, K, V])
        # the partials, the folded output right behind them (A_OFF): a pass of one block a node folds in the attention
        part = p.scratch(mp.part_bytes, 1, rows=1)
        pl = Tensor((SCR, part.ref[1] + mp.pl_off), scratch=True)
        a = p.scratch(Hq * hd)
        refs = (q, K, part, pl, V)
        scale = float(at.scaling)
        tm = T | (mp.wins.index(mp.windows[i]) << 16)  # the rows and the layer's window's row of the metadata
        if self.splits == 1 and mp.fusions.fold_in_attention:
            n = (Hk, 1, tm)
            p.emit(OpName.ATTN_FOLD, Hk * T, refs, n, scale, reads=[q, K, V], writes=[part, pl, a])
        else:
            n = (Hk, self.splits, tm)
            p.emit(OpName.ATTN, Hk * self.splits * T, refs, n, scale, reads=[q, K, V], writes=[part, pl])
            p.emit(OpName.FOLD, (T * Hq + 7) // 8, (part, pl, a), n, reads=[part, pl], writes=[a])
        return a

    def gated(self, gu: Tensor) -> Tensor:
        """act(gate) * up over gate and up end to end in `gu`, as the step makes it: silu in the fused kernel (gate and
        up one weight) or in MLX's own ops (the step's split gate and up, an unfused family), gelu-tanh in MLX's"""
        if self.mp.act in ("silu", "swish"):
            how = 0 if self.mp.fused and not gu.parts else 1
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
        y = self.y
        reads = [self.h] if y is None else [self.h, y]
        n = (H, int(y is not None), 0)
        p.emit(OpName.FNORM, T, (self.h, w, hn, None, y), n, eps, reads=reads, writes=[hn])
        logits = p.scratch(mp.V, 4)
        mp.logits_off = logits.ref[1]
        if mp.head_q is None:
            p.emit(OpName.HEAD, mv_items(mp.V, 8), (mp.head, hn, logits), (mp.V, H, 0), reads=[hn], writes=[logits])
        else:  # the type's own chain over float32 rows, as the step multiplies a stored head
            items = stored_groups(mp.head_q, mp.V, T)
            p.emit(MV_OP[mp.head_q], items, (mp.head, hn, logits), (mp.V, H, 1), reads=[hn], writes=[logits])
        amax = p.scratch(2 * AMP, 4)
        if argmax:
            n = (mp.V, AMP, p.R * AMP)
            p.emit(OpName.ARGMAX_PART, T * AMP, (logits, None, amax), n, reads=[logits], writes=[amax])
            p.emit(OpName.ARGMAX, T, (amax,), (0, AMP, p.R * AMP), reads=[amax])


class MegaPass:
    """A model's pass of 1..ROWS_MAX rows over an arena or a paged cache replayed on the GPU, bit-exact with the fused
    path: its weights in slot buffers, its constants packed once, a write-once scratch for the widest pass, and a
    program per pass shape (rows, attention blocks, argmax) built from the family's layers, encoded once and replayed
    over whichever cache a pass binds."""

    def __init__(self, sm: StreamedTextModel, fusions: Fusions | None = None) -> None:
        from ...engine.families import act_name

        self.fusions = fusions or Fusions()
        if not sm.fam.mega:
            raise ValueError(f"[mega] {sm.fam.name}'s layers are not composed for the megakernel")
        hh = sm.head_host
        if hh is None:
            raise ValueError("[mega] the head is not on the host")
        m = mx()
        c = sm.cfg
        self.sm = sm
        self.L = int(sm.L)
        self.H = int(c.hidden_size)
        self.Hq = int(c.num_attention_heads)
        self.Hk = int(getattr(c, "num_key_value_heads", None) or self.Hq)
        self.hd = int(sm.host[0].self_attn.head_dim)
        self.V = int(hh.mx.shape[0])
        self.SPL, self.ROWS = SPL, ROWS_MAX
        if self.hd % 32 or self.hd > 256 or self.H % 8:
            raise ValueError("[mega] the kernel takes a head of 32k dims up to 256 and a width of 8k")
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
        # the step this pass is bit-exact with: the fused kernels (`_forward_mlx`'s `fuse`), or MLX's own ops
        self.fused = bool(getattr(sm, "mlx_fused_kernels", True)) and sm.fam.fused_step and self.rd == self.hd
        self.act = act_name(c)
        # each layer's attention window, as the step reads it (a sandwich family's sliding layers), and the distinct
        # ones: a row of the node metadata each (`metas`)
        sw = int(getattr(c, "sliding_window", 0) or 0) if sm.fam.sandwich else 0
        self.windows = [sw if sw and sm.layer_types[i] == LayerKind.SLIDING else None for i in range(self.L)]
        self.wins = sorted(set(self.windows), key=lambda w: -1 if w is None else w)
        G = self.Hq // self.Hk
        # the attention's partials a layer, for the widest pass: m and l, then acc behind them
        pm = ROWS_MAX * self.Hk * SPL * 8 * G * 4
        self.pl_off = align(pm)
        self.pacc_off = 2 * align(pm)
        self.part_bytes = self.pacc_off + ROWS_MAX * self.Hk * SPL * 8 * G * self.hd * 4
        self.bufs: list[mx_.array] = []
        self._bid: dict[int, int] = {}
        self._cparts: list[mx_.array] = []
        self._csize = 0
        self._consts: dict[ConstKey, Tensor] = {}
        self._frozen = False
        self.logits_off = 0
        self.head, self.head_q = self._ref(hh.mx), hh.mx.q
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
        # every op a pass shape compiles to: the widest's and a one-block pass's (its attention folding its own)
        self.ops = frozenset(prog.ops) | self._program(ROWS_MAX, 1, True).ops
        self.instructions = len(prog.rows)  # the most a pass has: the widest one's
        self.replay = Replay(self)

    # -- what the program reads, gathered by the widest pass's build --

    def _ref(self, w: Weight) -> Tensor:
        """a weight in a slot or pool buffer, bf16 or of a stored type the kernel reads (`STORED`): its reference"""
        sh = getattr(w, "sh", None)
        if sh is None or w.packed is not None:
            raise ValueError("[mega] a weight outside a slot buffer")
        if w.q is not None and w.q not in STORED:
            raise ValueError(f"[mega] {w.q} weights are not in the kernel's ops")
        i = self._bid.get(id(sh.mx))
        if i is None:
            if self._frozen:
                raise ValueError("[mega] a pass reads a weight buffer the widest pass did not")
            i = self._bid[id(sh.mx)] = len(self.bufs)
            self.bufs.append(sh.whole)
        return Tensor((i, int(w.off)), int(w.shape[1]))

    def weight(self, w: Linear | None) -> tuple[Tensor, int, int, Quant | None]:
        """a slot weight (or a host linear holding one): its reference, rows, columns and stored type (None: bf16)"""
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
            mx().eval(a)
            t = self._consts[key] = Tensor((CONSTS, self._csize))
            self._cparts.append(a)
            self._csize += a.nbytes
            if self._csize % 4:
                pad = mx().zeros((4 - self._csize % 4,), dtype=mx().uint8)
                self._cparts.append(pad)
                self._csize += pad.nbytes
        return t

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
        if not self.fused:
            raise ValueError("[mega] q/k norms outside the fused kernels: the step norms them with MLX's own op")
        f32 = self.sm.fam.norm_centered
        t = self._const(
            ("qk", id(at)), lambda: mx().concatenate([_norm_array(qn, f32, f32), _norm_array(kn, f32, f32)])
        )
        return t, _eps(qn), f32

    def metas(self, past: int, parents: Sequence[int]) -> tuple[mx_.array, mx_.array, int]:
        """a pass's node metadata, a row of it a distinct window (`wins`) end to end, the nodes' paths, and the
        attention blocks the widest row reads"""
        rows = [tree_meta(past, parents, window=w) for w in self.wins]
        meta = rows[0][0] if len(rows) == 1 else mx().concatenate([r[0] for r in rows])
        return meta, rows[0][1], max(r[2] for r in rows)

    # -- the kernel --

    def defs(self, xr: int) -> dict[str, int]:
        """the ops' compile-time constants for passes of up to xr rows"""
        from ..attn import ATTN_BLOCK

        return {
            "XROWS": xr,
            "MV_KT": MV_KT,
            "ATT_G": self.Hq // self.Hk,
            "ATT_D": self.hd,
            "ATT_P": ATTN_BLOCK,
            "PACC_OFF": self.pacc_off,
            "A_OFF": align(self.part_bytes),
        }

    def tgm(self, xr: int) -> dict[OpName, int]:
        """each op's threadgroup memory at xr rows: the bf16 matvec's and the head's tiles staged a simdgroup each, the
        rest their reductions' and the attention's queries"""
        G = self.Hq // self.Hk
        small = max(G * self.hd * 4, (8 + self.hd) * 4, 2 * 256 * 4)
        need = dict.fromkeys(OpName, small)
        need[OpName.MATVEC] = max(small, 8 * (8 + xr) * MV_KT * 2)
        need[OpName.HEAD] = max(small, 8 * (8 * 32 + xr * 32) * 4)
        if max(need.values()) > TG_MEMORY:
            raise ValueError(
                f"[mega] an op takes {max(need.values())} bytes of threadgroup memory, a threadgroup has {TG_MEMORY}"
            )
        return need

    def _program(self, T: int, splits: int, argmax: bool) -> Program:
        """a pass's program: every layer and the head"""
        p = Program(T, ROWS_MAX)
        b = Block(self, p, splits)
        b.embed()
        for i in range(self.L):
            self.sm.fam.mega_layer(b, i)
        b.final(argmax)
        return p

    def table(self, T: int, splits: int, argmax: bool = True) -> np.ndarray:
        """the program of a pass of T rows (`splits` attention blocks) as the table the replay encodes, uint32
        [instructions, W]"""
        return self._program(T, splits, argmax).table(self.nb)

    def logits(self, T: int) -> mx_.array:
        """the logits [T, V] float32 the last pass left in the scratch (a pass without its argmax)"""
        o = self.logits_off
        return self.scr[o : o + T * self.V * 4].view(mx().float32).reshape(T, self.V)

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
    ) -> np.ndarray:
        """the pass, run and waited for: `kvs` each layer's K then V rows (2L of them) [Hk, cap, D] bf16, `rmap` a paged
        cache's row map (uint32, the conversation's rows to the regions' through the pass's own: rows are read and
        written through it) or None for rows where they lie (an arena), `ids` and `pos` the rows' tokens and positions,
        `meta`/`path` the node metadata; the argmax ids uint32 [T] (zeros with `argmax` off, the logits then in the
        scratch)"""
        if len(kvs) != 2 * self.L:
            raise ValueError(f"[mega] {len(kvs)} K and V regions for {self.L} layers")
        return self.replay.run(kvs, cap, rmap, ids, pos, meta, path, splits, past, argmax)


def _one_or_parts(mod: object, fused: str, names: tuple[str, ...]) -> Linears:
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
