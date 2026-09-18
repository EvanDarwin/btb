# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MTP drafter: a model's own drafting head run as a small model of its own, proposing the speculative tree."""

from __future__ import annotations

import copy
import heapq
import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from .. import mlx as mlxdev
from ..kinds import LayerKind, Tokens
from ..mlx import fused as fk
from ..options import Device
from .cache import GrowLayer
from .host import _HostLinear
from .native import Native

if TYPE_CHECKING:
    pass


def mlx_topk_ids(logits: Any, k: int) -> Any:
    """the top-k ids [T, k] a row of an MLX logits array, sorted by value descending, the reduction (argpartition
    then sort) on the graph so a caller reads k ids a row instead of the whole vocab; lazy."""
    m = mlxdev.mx()
    k = min(int(k), int(logits.shape[-1]))
    part = m.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    vals = m.take_along_axis(logits, part, axis=-1)
    order = m.argsort(-vals, axis=-1)
    return m.take_along_axis(part, order, axis=-1)


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


class MTPDrafter:
    build_s: float
    _head_h: Any
    _head_t: Any
    fc8: Any
    _mx_consts: Any
    _mx_head_w: Any
    cache: Any
    cd: torch.dtype
    dev: torch.device
    fc: torch.Tensor | None
    fc_host: _HostLinear | None
    layer: Any
    mcfg: Any
    norm: Any
    norm_e: Any
    norm_h: Any
    sm: Any
    step_s: float
    steps: int
    train_mode: bool

    def __init__(
        self, sm: Any, train: bool = False, weights: str | None = None, dev: str | torch.device | None = None
    ) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        self.sm = sm
        cfg = sm.cfg
        self.dev = torch.device(dev) if dev is not None else sm.dev
        self.cd = torch.float32 if train else torch.bfloat16
        self.train_mode = train
        self.step_s = 0.0
        self.steps = 0
        self._q: dict[Any, Any] = {}  # the weights packed for `draft_bits`, by (weight, dtype)
        idx = sm.layer_types.index(LayerKind.FULL)
        self.layer = sm._new_layer(idx)
        src = {} if weights is None else torch.load(weights, map_location="cpu", weights_only=True)
        get = lambda k: src[k] if k in src else sm._get(k)
        for name, _ in list(self.layer.named_parameters()):
            sm._set_param(self.layer, name, get(f"mtp.layers.0.{name}").to(self.dev, dtype=self.cd))
        self.layer.self_attn.layer_idx = 0
        self.fc = get("mtp.fc.weight").to(self.dev, dtype=self.cd)
        self.fc_host = None
        if self.dev.type == Device.CPU and not train and (Native.gemv is not None or sm.mlx is not None):
            base = "mtp.layers.0."
            for mname, m in list(self.layer.named_modules()):
                for cname, child in list(m.named_children()):
                    if isinstance(child, torch.nn.Linear) and child.weight.dtype == torch.bfloat16:
                        key = base + (f"{mname}.{cname}" if mname else cname) + ".weight"
                        setattr(m, cname, _HostLinear(child.weight.data, key=key))
            if getattr(sm, "_packed", None):
                sm._bind_host_packed_layer(self.layer)
            self.fc_host = _HostLinear(self.fc, key="mtp.fc.weight")
            if getattr(sm, "_packed", None):
                pk = sm._get_packed("mtp.fc.weight")
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
            self.cd = torch.float32
            if sm.mlx is not None:
                sm._bind_mlx_resident(self.layer, checkpoint=weights is None)
                sm._bind_mlx_linears([self.fc_host], checkpoint=weights is None)
        self.norm_e = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_e.weight.data = get("mtp.pre_fc_norm_embedding.weight").to(self.dev, dtype=self.cd)
        self.norm_h = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_h.weight.data = get("mtp.pre_fc_norm_hidden.weight").to(self.dev, dtype=self.cd)
        self.norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm.weight.data = get("mtp.norm.weight").to(self.dev, dtype=self.cd)
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
        out = {f"mtp.layers.0.{n}": p for n, p in self.layer.named_parameters()}
        if self.fc is not None:
            out["mtp.fc.weight"] = self.fc
        else:
            fc8 = self.fc8
            out["mtp.fc.weight"] = (fc8.w8.float() * fc8.scale[:, None]).to(self.cd)
        out["mtp.pre_fc_norm_embedding.weight"] = self.norm_e.weight
        out["mtp.pre_fc_norm_hidden.weight"] = self.norm_h.weight
        out["mtp.norm.weight"] = self.norm.weight
        return out

    def reset(self) -> None:
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache(config=self.mcfg)
        if self.sm.mlx is not None:
            self.cache.layers[0] = GrowLayer(shared=True)

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

    def _mx_dtype(self) -> Any:
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

    def _mm(self, x: Any, w: Any) -> Any:
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

    def _mx_head(self) -> Any:
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

    def _mx_norms(self) -> dict[str, Any]:
        """the drafter's own norm weights (1 + w) in its activation dtype, made once"""
        m = mlxdev.mx()
        dt = self._mx_dtype()
        wd = getattr(self, "_mx_consts", None)
        if wd is None or wd["dt"] != dt:
            wd = self._mx_consts = {"eps": float(self.sm.cfg.rms_norm_eps), "dt": dt}
            for name, mod in (("e", self.norm_e), ("h", self.norm_h), ("n", self.norm)):
                wd[name] = mlxdev.to_mx((1.0 + mod.weight.data.detach().float()).contiguous()).astype(dt)
            m.eval(wd["e"], wd["h"], wd["n"])
        return wd

    def _mlx_step(self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool = True) -> tuple[Any, Any]:
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
        append: Callable[[Any, Any], tuple[Any, Any]]
        if isinstance(cl, GrowLayer) and cl.shared:
            append = cl.mx_update
        else:

            def _append(kh: Any, vh: Any) -> tuple[Any, Any]:
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
            logits = self._mm(hn.reshape(B * T, H), self._mx_head()).reshape(B, T, -1).astype(m.float32)
            m.eval(logits, out)
            logits = mlxdev.from_mx(logits)
        else:
            m.eval(hn, out)
            logits = sm._head_host()(mlxdev.from_mx(hn)).float()
        self.step_s += time.time() - t0
        self.steps += 1
        return logits, mlxdev.from_mx(out)

    def _mlx_body(self, em: Any, hm: Any, pos0: int, append: Any, attn: Any = None, layer: Any = None) -> Any:
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
            K, V = append(kh, vh)
            if B == 1 and sm._mlx_prefill_able(layer, T, hd, Hq, Hk):
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

    def _mlx_draw(self, out: Any, k: int, sampling: Any, keys: Sequence[int]) -> tuple[Any, Any, Any]:
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

    def _mlx_topk(self, out: Any, k: int) -> tuple[Any, Any]:
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

    def _torch_head(self) -> Any:
        """Return rows `[:draft_vocab]` of the lm_head for drafting, int8-quantized if `draft_bits` < 16.
        Cached on the instance; None if the head is on another device and there is no slice."""
        head = getattr(self, "_head_t", None)
        if head is None:
            sm = self.sm
            W = (sm.head.weight if sm.resident_head else sm._get(sm.head_key)).detach()
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            whole = not (0 < n < int(W.shape[0]))
            if whole and W.device.type != self.dev.type:
                return None
            if not whole:
                W = W[:n]
            bits = int(getattr(self.sm, "draft_bits", 16) or 16)
            if bits < 16 and not self.train_mode:
                head = _Int8Linear(W.to(self.dev))
                sm.log(
                    f"[draft] the drafter's head: {'the first ' + str(n) + ' ids' if not whole else 'every id'} packed to "
                    f"8 bits in memory on {self.dev} ({head.w8.numel() / 2**20:.0f} MB)"
                )
            elif W.device.type != self.dev.type:
                head = W.to(self.dev, self.cd)
            else:
                head = W  # the model's own rows, a view
            self._head_t = head
        return head

    def _host_head(self) -> Any:
        """Return rows `[:draft_vocab]` of the lm_head as a native `_HostLinear` for drafting on the CPU.
        Cached on the instance; None without a slice, without the native kernel, or with MLX."""
        head = getattr(self, "_head_h", None)
        if head is None:
            sm = self.sm
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            head = False
            if sm.mlx is None and Native.gemv is not None and n > 0:
                W = sm._get(sm.head_key)
                if n < int(W.shape[0]):
                    head = _HostLinear(W[:n], key=sm.head_key)
                    sm.log(
                        f"[draft] the drafter's head on the host: the first {n} ids "
                        f"(bf16, {n * int(W.shape[1]) * 2 / 2**20:.0f} MB)"
                    )
            self._head_h = head
        return head or None

    def _pack_torch(self) -> int:
        """the drafter's layer on a torch tier with `draft_bits` below 16: every plain linear of it (and the fc
        tensor) replaced by an int8 one packed in memory; returns how many were packed (the host tier's
        native linears keep their own format)"""
        bits = int(getattr(self.sm, "draft_bits", 16) or 16)
        if bits >= 16 or self.train_mode or getattr(self, "_packed_torch", False):
            return 0
        self._packed_torch = True
        n = 0
        for _mname, mod in list(self.layer.named_modules()):
            for cname, child in list(mod.named_children()):
                if type(child) is torch.nn.Linear:
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

    def _step(self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool = True) -> tuple[Any, Any]:
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
        pe = sm.rotary(x, rope_pos)
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
                W = (sm.head.weight if sm.resident_head else sm._get(sm.head_key)).detach()
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

    def crop(self, keep: int) -> None:
        for layer in self.cache.layers:
            if getattr(layer, "keys", None) is not None and layer.keys.shape[-2] > keep:
                layer.keys = layer.keys[..., :keep, :]
                layer.values = layer.values[..., :keep, :]
                if hasattr(layer, "cumulative_length"):
                    layer.cumulative_length = int(layer.keys.shape[-2])

    def prefill(self, prompt_ids: Any, h_all: torch.Tensor) -> None:
        self.reset()
        n = len(prompt_ids)
        if n >= 2:
            self.extend(prompt_ids[1:n], h_all[:, : n - 1], 0)

    def extend(self, toks: Tokens, h: torch.Tensor, pos0: int) -> None:
        """The drafter's cache fed `toks` at `pos0` on, in chunks the way the engine's own prefill runs: its
        attention over a long prompt at once materialized the scores of every row against every key (a 13k-row
        turn asked Metal for 33 GB), so a chunk is sized by the same rule, against the keys already in its cache."""
        toks = [int(t) for t in toks]
        a = 0
        while a < len(toks):
            C = int(self.sm.prefill_chunk or self.sm._auto_chunk(pos0 + a))
            b = min(len(toks), a + C)
            self._step(torch.tensor([toks[a:b]]), h[:, a:b], pos0 + a, need_logits=False)
            a = b

    def ar(self, toks: Tokens, h: torch.Tensor, pos0: int, k: int) -> Any:
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

    def at(self, toks: Tokens, h: torch.Tensor, pos0: int, k: int) -> Any:
        if k <= 1:
            return (self.ar(toks, h, pos0, k) or [None])[0], [], []
        logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
        g1 = int(logits[0, -1].argmax())
        hp = hout[:, -1:]
        pos = pos0 + len(toks)
        logits2, hout2 = self._step(torch.tensor([[g1]]), hp, pos)
        top2 = torch.topk(logits2[0, -1], 2).indices.tolist()
        a2, b2 = int(top2[0]), int(top2[1])
        after_g1 = self.cache.get_seq_length()
        chains = []
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
        return g1, chains[0], chains[1]

    def au(
        self,
        toks: Tokens,
        h: torch.Tensor,
        pos0: int,
        budget: int,
        max_depth: int = 8,
        top_k: int = 8,
        expand_k: int = 8,
        min_prob: float = 0.0,
        pop_ratio: float = 0.5,
        fan_ratio: float = 0.05,
        extra_chains: Any = (),
        with_tags: bool = False,
        sampling: Any = None,
    ) -> Any:
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
        qrows: dict[int, Any] = {}
        empty: Any = (
            ([], [], [], [], qrows, {})
            if (with_tags and smp is not None)
            else ([], [], [], [])
            if with_tags
            else ([], [], [])
        )
        if budget <= 0:
            return empty
        floor = -math.log(min_prob) if min_prob > 0 else float("inf")
        fan_gap = -math.log(fan_ratio) if fan_ratio > 0 else float("inf")
        chains = [
            ([int(t) for t in c[0]], math.log(max(1e-6, min(1.0, float(c[1])))), (c[2] if len(c) > 2 else "extra"))
            for c in extra_chains
            if c and len(c[0]) > 0 and smp is None
        ]
        root_pos = pos0 + len(toks) - 1  # the cache row the root's children are proposed at
        layer = self.cache.layers[0]
        mlx_tree = self._mlx_ready() and isinstance(layer, GrowLayer) and layer.shared and not layer.bits
        if mlx_tree:
            m = mlxdev.mx()
            t0 = time.time()
            dt = self._mx_dtype()
            e = self.sm.embed(torch.tensor([[int(t) for t in toks]]))
            em = mlxdev.to_mx(e).astype(dt)
            hm = mlxdev.to_mx(h.detach().contiguous()).astype(dt)
            out = self._mlx_body(em, hm, pos0, layer.mx_update)
            if smp is not None:
                vals0, ids0, q0 = self._mlx_draw(out[:, -1:], top_k, smp, [smp.key_for(root_pos, salt=1)])
                qrows[-1] = q0[0]
            else:
                vals0, ids0 = self._mlx_topk(out[:, -1:], top_k)
            m.eval(vals0, ids0)
            self.step_s += time.time() - t0
            self.steps += 1
            root_len = self.cache.get_seq_length()
            kernel_tree = self._tree_kernel()
            kv_root: Any = None if kernel_tree else layer.mx_kv(0, root_len)
            root_vals, root_inds, root_hin = vals0.tolist()[0], ids0.tolist()[0], out[:, -1:]
            if kernel_tree:
                # the tree's rows go after the prefix, in stepping order; the prefix is never copied
                Hk_ = (
                    int(layer._shape[1])
                    if layer._shape is not None
                    else int(getattr(self.sm.cfg, "num_key_value_heads", 0))
                )
                hd_ = int(self.layer.self_attn.head_dim)
                layer._ensure(1, Hk_, root_len + budget + 1, hd_, mlxdev.torch_dtype(layer._mx[0].dtype))
                row_parent: list[int] = []  # a tree row's parent row (-1: the prefix)
                row_of: dict[int, int] = {}  # node index -> tree row
        else:
            logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
            root_len = self.cache.get_seq_length()
            kv_root = (layer.keys.clone(), layer.values.clone())
            if smp is not None:
                ids0, q0 = smp.draw_torch(logits[0, -1:].float(), [smp.key_for(root_pos, salt=1)], top_k)
                qrows[-1] = q0[0]
                root_vals = torch.log(q0[0][ids0[0]]).tolist()
                root_inds, root_hin = ids0[0].tolist(), hout[:, -1:]
            else:
                tb0 = torch.topk(torch.log_softmax(logits[0, -1].float(), dim=-1), min(top_k, logits.shape[-1]))
                root_vals, root_inds, root_hin = tb0.values.tolist(), tb0.indices.tolist(), hout[:, -1:]
        nodes: list[Any] = []
        kv_after: dict[Any, Any] = {}
        paths: dict[Any, Any] = {}
        tags: dict[Any, Any] = {}
        heap: list[Any] = []
        tie = 0
        draws: dict[int, list[int]] = {}  # under sampling: every draw of a node in draw order, for the verify pass

        def az(
            parent_idx: int, parent_neg: float, parent_depth: int, vals: Any, inds: Any, hin: Any, parent_path: Any
        ) -> None:
            nonlocal tie
            if smp is not None:
                # the verify pass tries all of a node's draws in order whether or not the tree holds them (an accepted
                # draw the tree lacks ends the walk), so the heap takes them by probability like the greedy tree
                draws[parent_idx] = [int(t) for lv, t in zip(vals, inds) if math.isfinite(lv)]  # no zero-mass draw
            best = max(vals)
            seen = {}
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
                heapq.heappush(heap, (neg, tie, parent_idx, t, parent_depth + 1, hin, tag, [*parent_path, t]))
                tie += 1

        az(-1, 0.0, 0, root_vals, root_inds, root_hin, [])
        # a confident root over-builds: shrink the budget as it sharpens (size affects speed only, never a token)
        p1 = math.exp(float(max(root_vals)))
        cmin = int(getattr(self.sm, "tree_cap_min", 5))
        lo = float(getattr(self.sm, "tree_conf_lo", 0.5))
        hi = float(getattr(self.sm, "tree_conf_hi", 0.9))
        frac = min(1.0, max(0.0, (p1 - lo) / max(1e-6, hi - lo)))
        eff_budget = max(min(cmin, budget), round(budget - (budget - cmin) * frac))
        pop_gap = -math.log(pop_ratio)
        # a depth is stepped only when its nodes' path probability can pay for the drafter's step
        step_mass = float(getattr(self.sm, "tree_step_mass", 0.0) or 0.0)
        while heap and len(nodes) < eff_budget and heap[0][0] <= floor:
            popped: list[Any] = []
            while (
                heap
                and len(popped) < expand_k
                and len(nodes) + len(popped) < eff_budget
                and heap[0][0] <= floor
                and (not popped or heap[0][0] - popped[0][0] <= pop_gap)
            ):
                popped.append(heapq.heappop(heap))
            first = len(nodes)
            for _neg, _, parent, tok, depth, _hin, tag, path in popped:
                idx = len(nodes)
                nodes.append((tok, parent, depth))
                paths[idx], tags[idx] = path, tag
            if len(nodes) >= eff_budget:
                break
            by_depth: dict[Any, Any] = {}
            for j, (neg, _, parent, tok, depth, hin, _tag, _path) in enumerate(popped):
                if depth < max_depth:
                    by_depth.setdefault(depth, []).append((first + j, neg, parent, tok, hin))
            for depth, group in sorted(by_depth.items()):
                if step_mass > 0 and sum(math.exp(-neg) for _, neg, _, _, _ in group) < step_mass:
                    continue
                ts = torch.tensor([[tok] for _, _, _, tok, _ in group])
                if mlx_tree and kernel_tree:
                    # the group's rows appended to the shared cache where their paths say, the attention one node
                    # kernel call over the prefix and the tree's rows: nothing copied, nothing crosses to torch
                    t0 = time.time()
                    first_row = len(row_parent)
                    for j, (idx, _neg, parent, _tok, _hin) in enumerate(group):
                        row_of[idx] = first_row + j
                        row_parent.append(-1 if parent < 0 else row_of[parent])
                    meta_all, path_all, splits = mlxdev.tree_meta(root_len, row_parent)
                    meta_g, path_g = meta_all[first_row:], path_all[first_row:]
                    hs = m.concatenate([hin for _, _, _, _, hin in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    kbuf, vbuf = layer._mx[0], layer._mx[1]
                    scale_ = float(self.layer.self_attn.scaling)

                    def node_attn(
                        qh: Any,
                        kh: Any,
                        vh: Any,
                        kbuf: Any = kbuf,
                        vbuf: Any = vbuf,
                        n0: int = root_len + first_row,
                        meta: Any = meta_g,
                        path: Any = path_g,
                        splits: int = splits,
                        scale_: float = scale_,
                    ) -> Any:
                        # the rows written in place, the flag evaluated before the kernel reads them
                        fk_ = mlxdev.kv_store(kbuf, kh[:, :, 0, :].transpose(1, 0, 2), n0)
                        fv_ = mlxdev.kv_store(vbuf, vh[:, :, 0, :].transpose(1, 0, 2), n0)
                        m.eval(fk_, fv_)
                        a = mlxdev.attn_nodes(qh[:, :, 0, :], kbuf, vbuf, meta, path, scale_, splits, odt=qh.dtype)
                        return a[:, :, None, :]

                    ho = self._mlx_body(em, hs, root_len + depth - 1, None, attn=node_attn)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, (idx, _, _, _, _) in enumerate(group):
                            qrows[idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = vals.tolist(), inds.tolist()
                    for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                        az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
                    continue
                if mlx_tree:
                    # the group's prefixes and hidden rows concatenated on the graph, the node's K/V a lazy slice
                    # of the group's, the top-k read as k numbers a node: nothing crosses to torch
                    t0 = time.time()
                    kp = m.concatenate([kv_root[0] if p < 0 else kv_after[p][0] for _, _, p, _, _ in group], axis=0)
                    vp = m.concatenate([kv_root[1] if p < 0 else kv_after[p][1] for _, _, p, _, _ in group], axis=0)
                    hs = m.concatenate([hin for _, _, _, _, hin in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    kn: list[Any] = []

                    def cat_kv(kh: Any, vh: Any, kp: Any = kp, vp: Any = vp, kn: list[Any] = kn) -> tuple[Any, Any]:
                        K = m.concatenate([kp, kh], axis=2)
                        V = m.concatenate([vp, vh], axis=2)
                        kn.extend((K, V))
                        return K, V

                    ho = self._mlx_body(em, hs, root_len + depth - 1, cat_kv)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, (idx, _, _, _, _) in enumerate(group):
                            qrows[idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = vals.tolist(), inds.tolist()
                    for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                        kv_after[idx] = (kn[0][b : b + 1], kn[1][b : b + 1])
                        az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
                    continue
                ks = torch.cat(
                    [kv_root[0] if parent < 0 else kv_after[parent][0] for _, _, parent, _, _ in group], dim=0
                )
                vs = torch.cat(
                    [kv_root[1] if parent < 0 else kv_after[parent][1] for _, _, parent, _, _ in group], dim=0
                )
                layer.keys, layer.values = ks, vs
                hs = torch.cat([hin for _, _, _, _, hin in group], dim=0)
                lg, ho = self._step(ts, hs, root_len + depth - 1)
                if smp is not None:
                    gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                    ids_g, qg = smp.draw_torch(lg[:, -1].float(), gk, top_k)
                    vals_l = torch.log(torch.gather(qg, 1, ids_g)).tolist()
                    inds_l = ids_g.tolist()
                    for b, (idx, _, _, _, _) in enumerate(group):
                        qrows[idx] = qg[b]
                else:
                    tb = torch.topk(torch.log_softmax(lg[:, -1].float(), dim=-1), min(top_k, lg.shape[-1]), dim=-1)
                    vals_l, inds_l = tb.values.tolist(), tb.indices.tolist()
                for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                    kv_after[idx] = (layer.keys[b : b + 1].clone(), layer.values[b : b + 1].clone())
                    az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
        if not mlx_tree:
            layer.keys, layer.values = kv_root
        out = ([n[0] for n in nodes], [n[1] for n in nodes], [n[2] for n in nodes])
        if with_tags and smp is not None:
            return (*out, [tags[i] for i in range(len(nodes))], qrows, draws)
        return (*out, [tags[i] for i in range(len(nodes))]) if with_tags else out
