# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3.5's MTP drafter: its drafting head (`mtp.fc` over the normed embedding and hidden state, one full-attention
layer, `mtp.norm`) as the step the shared drafter (btb/engine/drafter.py) grows its trees from, on torch or as one
MLX graph. Qwen3.5-MoE's layer is the family's own, a mixture's: its experts are the expert store's, read under
`mtp.layers.0.mlp.experts.` (one tensor an expert in its checkpoints) as the store's layer `L`, one past the trunk's,
as Qwen4's drafter reads its own."""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from .... import mlx as mlxdev
from ....kinds import LayerKind
from ....mlx import fused as fk
from ....options import Device
from ...cache import GrowLayer
from ...drafter import MTPDrafter, _Int8Linear, mlx_topk_ids
from ...host import _HostLinear
from ...native import Native

if TYPE_CHECKING:
    import mlx.core as mx_

    from ....mlx.backend import Weight
    from ....sampling import Sampling
    from ...drafter import MxAppend, MxAttend

# the drafting head's layer, and its own tensors around it: the fc over [embedding; hidden], their two norms, and
# the norm before the head
LAYER = "mtp.layers.0."
FC = "mtp.fc.weight"
NORM_E = "mtp.pre_fc_norm_embedding.weight"
NORM_H = "mtp.pre_fc_norm_hidden.weight"
NORM = "mtp.norm.weight"


class _Norms(TypedDict):
    """the drafter's own norm weights (1 + w) in its MLX activation dtype `dt`: the embedding's, the hidden rows' and
    the output's, and their epsilon"""

    eps: float
    dt: mx_.Dtype
    e: mx_.array
    h: mx_.array
    n: mx_.array


class Qwen35Drafter(MTPDrafter):
    """the drafting head of Qwen3.5 and the families sharing its `mtp.*` layout"""

    def __init__(
        self, sm: Any, train: bool = False, weights: str | None = None, dev: str | torch.device | None = None
    ) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        if train and sm.fam.moe:
            raise ValueError("btb does not train Qwen3.5-MoE's drafting head: its experts are the expert store's")
        super().__init__(sm, train=train, weights=weights, dev=dev)
        cfg = sm.cfg
        idx = sm.layer_types.index(LayerKind.FULL)
        L = int(sm.L)
        self.layer = sm._new_layer(idx, base=LAYER, at=L)
        src = {} if weights is None else torch.load(weights, map_location="cpu", weights_only=True)
        get = lambda k: src[k] if k in src else sm._get(k)
        for name, p in list(self.layer.named_parameters()):
            t = get(f"{LAYER}{name}")
            if t.shape != p.shape:
                raise RuntimeError(
                    f"[mtp] {LAYER}{name} is {tuple(t.shape)}; the drafter's layer takes {tuple(p.shape)}"
                )
            sm._set_param(self.layer, name, t.to(self.dev, dtype=self.cd))
        self.layer.self_attn.layer_idx = 0
        self.fc = get(FC).to(self.dev, dtype=self.cd)
        self.fc_host = None
        if self.dev.type == Device.CPU and not train and (Native.gemv is not None or sm.mlx is not None):
            for mname, m in list(self.layer.named_modules()):
                for cname, child in list(m.named_children()):
                    if isinstance(child, torch.nn.Linear) and child.weight.dtype == torch.bfloat16:
                        key = LAYER + (f"{mname}.{cname}" if mname else cname) + ".weight"
                        setattr(m, cname, _HostLinear(child.weight.data, key=key))
            # the family's own part of a host layer: a mixture's router through the host's gemv
            sm.fam.finish_host_layer(sm, self.layer, L, base=LAYER)
            if getattr(sm, "_packed", None):
                sm._bind_host_packed_layer(self.layer)
            self.fc_host = _HostLinear(self.fc, key=FC)
            if getattr(sm, "_packed", None):
                pk = sm._get_packed(FC)
                if pk is not None:
                    blob, tbl, e = pk
                    a1 = e["lo"]
                    a2 = a1 + e["hi4"] + e["pad"]
                    a3 = a2 + 4 * e["esc"]
                    self.fc_host.packed = (
                        blob[:a1],
                        blob[a1 : a1 + e["hi4"]],
                        tbl,
                        blob[a2:a3].view(torch.int32) if e["esc"] else torch.zeros(0, dtype=torch.int32),
                        blob[a3:] if e["esc"] else torch.zeros(0, dtype=torch.uint8),
                        int(e["esc"]),
                    )
            if sm.mlx is None and weights is None:
                # a GGUF's stored types multiplied in their blocks, as a host layer's are (MLX's binder takes them)
                heads = (*self.layer.modules(), self.fc_host)
                sm._bind_stored({m.key: m for m in heads if isinstance(m, _HostLinear) and m.key})
            self.cd = torch.float32
            if sm.mlx is not None:
                sm._bind_mlx_resident(self.layer, checkpoint=weights is None)
                sm._bind_mlx_linears([self.fc_host], checkpoint=weights is None)
        self.norm_e = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_e.weight.data = get(NORM_E).to(self.dev, dtype=self.cd)
        self.norm_h = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_h.weight.data = get(NORM_H).to(self.dev, dtype=self.cd)
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm.weight.data = get(NORM).to(self.dev, dtype=self.cd)
        # the rope over the drafter's own positions, on its device: a drafter on the host (shed there, or the plan put
        # none on the card) handed the card engine's rope its positions in RAM
        self.rotary = sm.fam.rotary(config=cfg).to(self.dev)
        if weights is not None:
            sm.log(
                f"[mtp] drafter weights <- {weights} ({sum(1 for k in src if k != '__meta__')} tensors"
                + (f"; epochs {src['__meta__'].get('epochs_done')}" if "__meta__" in src else "")
                + ")"
            )
        if train:
            for name, p in list(self.layer.named_parameters()):
                sm._set_param(self.layer, name, torch.nn.Parameter(p.detach().clone().float(), requires_grad=True))
            assert self.fc is not None  # set in __init__, nulled only later by int8 quant
            self.fc = torch.nn.Parameter(self.fc.detach().clone().float(), requires_grad=True)
            for m in (self.norm_e, self.norm_h, self.norm):
                m.weight = torch.nn.Parameter(m.weight.detach().clone().float(), requires_grad=True)
            self.layer.train(False)
        mcfg = copy.deepcopy(cfg)
        mcfg.num_hidden_layers = 1
        mcfg.layer_types = [LayerKind.FULL]
        self.mcfg = mcfg
        self.cache = None

    def named_tensors(self) -> dict[str, torch.Tensor]:
        out = {f"{LAYER}{n}": p for n, p in self.layer.named_parameters()}
        if self.fc is not None:
            out[FC] = self.fc
        else:
            fc8 = self.fc8
            out[FC] = (fc8.w8.float() * fc8.scale[:, None]).to(self.cd)
        out[NORM_E] = self.norm_e.weight
        out[NORM_H] = self.norm_h.weight
        out[NORM] = self.norm.weight
        return out

    def _mlx_ready(self) -> bool:
        sm = self.sm
        return (
            sm.mlx is not None
            and not self.train_mode
            and self.fc_host is not None
            and self.fc_host.mx is not None
            and getattr(self.layer.self_attn.q_proj, "mx", None) is not None
            and sm._mlx_act() is not None
        )

    def _mx_dtype(self) -> mx_.Dtype:
        """The drafter's activation dtype on MLX: bf16 where the tree runs through the node kernel (it reads bf16
        rows), float32 otherwise (faster end to end). The verified output does not depend on it."""
        m = mlxdev.mx()
        return m.bfloat16 if self._tree_kernel() else m.float32

    def _tree_kernel(self) -> bool:
        """the tree over the shared cache through the node kernel: the prefix kept once in the kernel's form, each
        depth one call; False for a head shape the kernel does not take (every node's prefix is copied then)"""
        sm = self.sm
        if sm.mlx is None or self.train_mode:
            return False
        hd = int(self.layer.self_attn.head_dim)
        Hq = int(sm.cfg.num_attention_heads)
        Hk = int(getattr(sm.cfg, "num_key_value_heads", None) or Hq)
        return hd in (64, 128, 256) and Hq // Hk <= mlxdev.ATTN_MAXG

    def _mm(self, x: mx_.array, w: Weight) -> mx_.array:
        """x [b, cols] against the drafter's weight `w`: the engine's matmul, or, with `draft_bits` 4 or 8, MLX's
        quantized matmul over a copy of the weight packed in memory at first use (groups of 64, affine) - the
        model's files untouched, the drafter's output only ever verified"""
        bits = int(getattr(self.sm, "draft_bits", 16) or 16)
        if bits >= 16:
            return self.sm.mlx.matmul(x, w)
        m = mlxdev.mx()
        q = self._q
        key = (id(w), str(x.dtype))
        packed = q.get(key)
        if packed is None:
            W = w.get()
            W = W.astype(x.dtype) if W.dtype != x.dtype else W
            packed = q[key] = m.quantize(W, group_size=64, bits=bits)
            m.eval(*packed)
        wq, sc, bi = packed
        return m.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=bits)

    def _mx_head(self) -> Weight:
        """The drafter's head: the model's, or its first `draft_vocab` rows (ids follow frequency rank). A proposal
        outside them is a missed draft; the verified output is unchanged."""
        w = getattr(self, "_mx_head_w", None)
        if w is None:
            sm = self.sm
            w = sm.head_host.mx
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            if 0 < n < int(w.shape[0]) and w.packed is None:
                w = mlxdev.Weight(w.get()[:n])
            self._mx_head_w = w
        return w

    def _mx_norms(self) -> _Norms:
        """the drafter's own norm weights (1 + w) in its activation dtype, made once"""
        m = mlxdev.mx()
        dt = self._mx_dtype()
        wd: _Norms | None = getattr(self, "_mx_consts", None)
        if wd is None or wd["dt"] != dt:
            e, h, n = (
                mlxdev.to_mx((1.0 + mod.weight.data.detach().float()).contiguous()).astype(dt)
                for mod in (self.norm_e, self.norm_h, self.norm)
            )
            m.eval(e, h, n)
            wd = _Norms(eps=float(self.sm.cfg.rms_norm_eps), dt=dt, e=e, h=h, n=n)
            self._mx_consts = wd
        return wd

    def _mlx_step(
        self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """The drafter's step as one MLX graph (norms, fc, the attention layer over its shared cache, the head)
        over bf16 weights in the drafter's dtype."""
        m = mlxdev.mx()
        sm = self.sm
        c = sm.cfg
        t0 = time.time()
        e = sm.embed(tok_ids)
        B, T = int(e.shape[0]), int(e.shape[1])
        wd = self._mx_norms()
        dt = self._mx_dtype()
        em = mlxdev.to_mx(e).astype(dt)
        hm = mlxdev.to_mx(h.detach().contiguous()).astype(dt)
        cl = self.cache.layers[0]
        append: MxAppend
        if isinstance(cl, GrowLayer) and cl.shared:
            append = cl.mx_update
        else:

            def _append(kh: mx_.array, vh: mx_.array) -> tuple[mx_.array, mx_.array]:
                m.eval(kh, vh)
                kf, vf = self.cache.update(mlxdev.from_mx(kh), mlxdev.from_mx(vh), 0)
                return mlxdev.to_mx(kf), mlxdev.to_mx(vf)

            append = _append

        out = self._mlx_body(em, hm, pos0, append, layer=cl if isinstance(cl, GrowLayer) and cl.shared else None)
        if not need_logits:
            m.eval(out)
            self.step_s += time.time() - t0
            self.steps += 1
            return None, mlxdev.from_mx(out)
        hn = m.fast.rms_norm(out, wd["n"], wd["eps"])
        H = int(c.hidden_size)
        if sm.head_host is not None and sm.head_host.mx is not None:
            lm = self._mm(hn.reshape(B * T, H), self._mx_head()).reshape(B, T, -1).astype(m.float32)
            m.eval(lm, out)
            logits = mlxdev.from_mx(lm)
        else:
            m.eval(hn, out)
            logits = sm._head_host()(mlxdev.from_mx(hn)).float()
        self.step_s += time.time() - t0
        self.steps += 1
        return logits, mlxdev.from_mx(out)

    def _mlx_body(
        self,
        em: mx_.array,
        hm: mx_.array,
        pos0: int,
        append: MxAppend | None,
        attn: MxAttend | None = None,
        layer: GrowLayer | None = None,
    ) -> mx_.array:
        """The drafter's layer over `em`/`hm` [B, T, H] MLX arrays in the drafter's dtype: returns `out` [B, T, H],
        lazily; `append(kh, vh)` gives the attention its K/V (the cache's append), or `attn(qh, kh, vh)` computes
        the attention itself (the tree's node kernel over the shared cache) and returns [B, Hq, T, D]; `layer` the
        shared cache layer `append` writes, for a chunk's attention through the fused prefill kernel."""
        m = mlxdev.mx()
        sm = self.sm
        be = sm.mlx
        c = sm.cfg
        wd = self._mx_norms()
        B, T = int(em.shape[0]), int(em.shape[1])
        xin = m.concatenate([m.fast.rms_norm(em, wd["e"], wd["eps"]), m.fast.rms_norm(hm, wd["h"], wd["eps"])], axis=-1)
        H = int(c.hidden_size)
        assert self.fc_host is not None  # the MLX body runs with the resident head bound
        x = self._mm(xin.reshape(B * T, -1), self.fc_host.mx).reshape(B, T, H)
        freqs, rd, rscale = sm._mlx_rope()
        w = sm._mlx_consts(-2, self.layer, mlxdev.torch_dtype(wd["dt"]))
        at, mlp = self.layer.self_attn, self.layer.mlp
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        hd = int(at.head_dim)
        act = sm._mlx_act()
        xn = m.fast.rms_norm(x, w["ln1"], w["eps1"]).reshape(B * T, H)
        qkv_w = getattr(at, "_mx_qkv", None)
        if qkv_w is not None:
            qkv = self._mm(xn, qkv_w)
            nq, nk = Hq * 2 * hd, Hk * hd
            qg, kk, vv = qkv[:, :nq], qkv[:, nq : nq + nk], qkv[:, nq + nk :]
        else:
            qg, kk, vv = self._mm(xn, at.q_proj.mx), self._mm(xn, at.k_proj.mx), self._mm(xn, at.v_proj.mx)
        q, gate = m.split(qg.reshape(B, T, Hq, 2 * hd), 2, axis=-1)
        gate = gate.reshape(B, T, Hq * hd)
        k = kk.reshape(B, T, Hk, hd)
        v = vv.reshape(B, T, Hk, hd)
        fuse = rd == hd and B * T <= 16
        if fuse:
            # the q/k norms and the rope in one launch, row (b, t) at pos0 + t
            pos = [pos0 + t for _b in range(B) for t in range(T)]
            qr, kr = fk.qk_norm_rope(
                q.reshape(B * T, Hq, hd), k.reshape(B * T, Hk, hd), w["qn"], w["kn"], w["epsq"], rd, freqs, rscale, pos
            )
            qh = qr.reshape(B, T, Hq, hd).transpose(0, 2, 1, 3)
            kh = kr.reshape(B, T, Hk, hd).transpose(0, 2, 1, 3)
        else:
            q = m.fast.rms_norm(q, w["qn"], w["epsq"])
            k = m.fast.rms_norm(k, w["kn"], w["epsq"])
            qh = be.rope_fast(q.transpose(0, 2, 1, 3), rd, freqs, rscale, pos0)
            kh = be.rope_fast(k.transpose(0, 2, 1, 3), rd, freqs, rscale, pos0)
        vh = v.transpose(0, 2, 1, 3)
        if self._tree_kernel():
            # the cache in the drafter's dtype from the prefill on: the node kernel reads bf16 rows, and a
            # layer asked to grow in another dtype starts over empty
            kh, vh = kh.astype(em.dtype), vh.astype(em.dtype)
        if attn is not None:
            a = attn(qh, kh, vh)
        else:
            assert append is not None  # the attention's K/V from the caller's cache when it computes none itself
            K, V = append(kh, vh)
            if B == 1 and layer is not None and sm._mlx_prefill_able(layer, T, hd, Hq, Hk):
                a = mlxdev.attn_prefill(
                    qh[0].transpose(1, 0, 2), layer._mx[0], layer._mx[1], layer._n - T, float(at.scaling), odt=qh.dtype
                )
                a = a.transpose(1, 0, 2)[None]
            else:
                a = m.fast.scaled_dot_product_attention(
                    qh, K, V, scale=float(at.scaling), mask="causal" if T > 1 else None
                )
        a = a.transpose(0, 2, 1, 3).reshape(B, T, Hq * hd) * m.sigmoid(gate)
        o = self._mm(a.reshape(B * T, -1), at.o_proj.mx)
        if sm.fam.moe:
            # a mixture's block through its modules, as the trunk's run on MLX: the router and the shared expert's
            # linears on MLX, the experts the store's (layer L), between the graph's halves
            x = x + o.reshape(B, T, H)
            x2 = m.fast.rms_norm(x, w["ln2"], w["eps2"]).astype(m.float32)
            m.eval(x2)
            y = mlp(mlxdev.from_mx(x2))
            return x + mlxdev.to_mx(y.float().contiguous()).astype(x.dtype)
        gu_w = getattr(mlp, "_mx_gu", None)
        if fuse:
            x_flat, x2 = fk.add_rmsnorm(x.reshape(B * T, H), o, w["ln2"], w["eps2"])
            if gu_w is not None:
                down = self._mm(fk.silu_mul(self._mm(x2, gu_w)), mlp.down_proj.mx)
            else:
                mid = act(self._mm(x2, mlp.gate_proj.mx)) * self._mm(x2, mlp.up_proj.mx)
                down = self._mm(mid, mlp.down_proj.mx)
            return (x_flat + down).reshape(B, T, H)
        x = x + o.reshape(B, T, H)
        x2 = m.fast.rms_norm(x, w["ln2"], w["eps2"]).reshape(B * T, H)
        mid = act(self._mm(x2, mlp.gate_proj.mx)) * self._mm(x2, mlp.up_proj.mx)
        return x + self._mm(mid, mlp.down_proj.mx).reshape(B, T, H)

    def _mlx_draw(
        self, out: mx_.array, k: int, sampling: Sampling, keys: Sequence[int]
    ) -> tuple[mx_.array, mx_.array, mx_.array]:
        """The drafter's head over `out` [B, 1, H] under `sampling`: `k` children a row drawn without replacement
        from the sampled distribution (in draw order), their log-probabilities, and the distribution itself
        [B, Vd]; lazily."""
        m = mlxdev.mx()
        sm = self.sm
        wd = self._mx_norms()
        H = int(sm.cfg.hidden_size)
        B = int(out.shape[0])
        hn = m.fast.rms_norm(out, wd["n"], wd["eps"]).reshape(B, H)
        lg = self._mm(hn, self._mx_head()).astype(m.float32)
        ids, q = sampling.draw_mx(lg, keys, min(int(k), int(lg.shape[-1])))
        vals = m.log(m.take_along_axis(q, ids, axis=-1))
        return vals, ids, q

    def _mlx_topk(self, out: mx_.array, k: int) -> tuple[mx_.array, mx_.array]:
        """The drafter's head over `out` [B, 1, H]: the top-k log-probs and ids [B, k], sorted, lazily - the
        whole reduction on the graph, so a tree depth reads k numbers a node instead of the logits."""
        m = mlxdev.mx()
        sm = self.sm
        wd = self._mx_norms()
        H = int(sm.cfg.hidden_size)
        B = int(out.shape[0])
        hn = m.fast.rms_norm(out, wd["n"], wd["eps"]).reshape(B, H)
        lp = self._mm(hn, self._mx_head()).astype(m.float32)
        lp = lp - m.logsumexp(lp, axis=-1, keepdims=True)
        ids = mlx_topk_ids(lp, k)
        return m.take_along_axis(lp, ids, axis=-1), ids

    def _pass(
        self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self._mlx_ready():
            return self._mlx_step(tok_ids, h, pos0, need_logits)
        from transformers.masking_utils import create_causal_mask

        self._pack_torch()
        t0 = time.time()
        sm = self.sm
        e = sm.embed(tok_ids).to(self.dev, self.cd)
        xin = torch.cat([self.norm_e(e), self.norm_h(h.to(self.dev, self.cd))], dim=-1)
        fc8 = getattr(self, "fc8", None)
        if self.fc_host is not None:
            x = self.fc_host(xin.float())
        elif fc8 is not None:
            x = fc8(xin)
        else:
            assert self.fc is not None  # no head backend means the raw fc weight
            x = xin @ self.fc.T
        T = x.shape[1]
        pos = (torch.arange(T, device=self.dev) + pos0).view(1, 1, -1).expand(4, x.shape[0], -1)
        text_pos, rope_pos = pos[0], pos[1:]
        pe = self.rotary(x, rope_pos)
        mask = create_causal_mask(
            config=self.mcfg, inputs_embeds=x, attention_mask=None, past_key_values=self.cache, position_ids=text_pos
        )
        out = self.layer(
            x,
            position_embeddings=pe,
            attention_mask=mask,
            position_ids=text_pos,
            past_key_values=self.cache,
            use_cache=True,
        )
        if not need_logits:
            sm._sync()
            self.step_s += time.time() - t0
            self.steps += 1
            return None, out
        hn = self.norm(out)
        if self.dev.type == Device.CPU and (sm.dev.type != Device.CPU or Native.gemv is not None or sm.mlx is not None):
            head = self._host_head()
            logits = (sm._head_host() if head is None else head)(hn.float()).float()
        else:
            head = self._torch_head()
            if head is None:
                W = (sm.head.weight if sm.head is not None else sm._get(sm.head_key)).detach()
                step = 32768
                logits = torch.cat(
                    [hn.to(self.cd) @ W[c : c + step].to(self.dev, self.cd).T for c in range(0, W.shape[0], step)],
                    dim=-1,
                ).float()
            elif isinstance(head, _Int8Linear):
                logits = head(hn).float()
            else:
                logits = (hn.to(head.dtype) @ head.T).float()
        sm._sync()
        self.step_s += time.time() - t0
        self.steps += 1
        return logits, out
