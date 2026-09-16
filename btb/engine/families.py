# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The model families the engine serves: their attention forwards (the engine's SDPA, sinks and sliding
windows), their layer shapes and how a host layer is built from the checkpoint."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .. import mlx as mlxdev
from ..kinds import FamilyKind, LayerKind
from .cache import GrowLayer
from .fused import _fuse_mlp_cls, _fuse_norm_cls
from .host import _Experts, _HostLinear, _NGramRows, _Router
from .state import _State


def attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kw: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if query.device.type != "cuda":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw
        )
    g = int(getattr(module, "num_key_value_groups", 1) or 1)
    if g > 1:
        b, hk, t, d = key.shape
        key = key[:, :, None].expand(b, hk, g, t, d).reshape(b, hk * g, t, d)
        value = value[:, :, None].expand(b, hk, g, t, d).reshape(b, hk * g, t, d)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    causal = attention_mask is None and query.shape[2] > 1 if is_causal is None else bool(is_causal)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, is_causal=causal, scale=scaling
    )
    return out.transpose(1, 2).contiguous(), None


def attention_sinks(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    s_aux: Any = None,
    **kw: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """gpt-oss's attention: a sink logit per query head in the softmax's denominator and, on alternate layers,
    a `sliding_window`. Over an MLX cache the nodes of a speculative pass (and every decode row once
    speculation is on) run through the node kernel, the rest through MLX's fused attention; elsewhere the
    reference's arithmetic without its copies. Returns what the reference returns."""
    B, Hq, T, d = (int(v) for v in query.shape)
    Hk = int(key.shape[1])
    g = Hq // Hk
    scale = float(d**-0.5 if scaling is None else scaling)
    win = int(sliding_window) if sliding_window else None
    sm: Any = getattr(module, "_sm", None)
    cl = None
    if sm is not None and sm.mlx is not None and B == 1 and getattr(module, "layer_idx", None) is not None:
        ctx = getattr(sm, "_attn_ctx", None)
        cl = ctx.layers[module.layer_idx] if ctx is not None and module.layer_idx < len(ctx.layers) else None
        if not (
            isinstance(cl, GrowLayer)
            and cl.shared
            and cl._mx is not None
            and cl._tk is None
            and cl._tv is None
            and int(cl._n) == int(key.shape[-2])
            and s_aux is not None
        ):
            cl = None
    if cl is not None:
        m = mlxdev.mx()
        n = int(cl._n)
        past = n - T
        spec = bool(getattr(sm, "aq", False)) and past > 0
        sinks = getattr(module, "_mx_sinks", None)
        if sinks is None:
            sinks = mlxdev.to_mx(s_aux.detach().float().contiguous())
            m.eval(sinks)
            module._mx_sinks = sinks
        K, V = cl._mx[0], cl._mx[1]
        kernel = (
            d in (64, 128, 256)
            and g <= mlxdev.ATTN_MAXG
            and (K.dtype == m.bfloat16 or cl.bits)
            and getattr(sm, "mlx_attn_kernel", True)
            and ((T > 1 and spec) or (T == 1 and (spec or n >= sm.mlx_attn_rows)))
        )
        kq = {"ks": cl._mx[2], "vs": cl._mx[3]} if cl.bits else {}
        parents = getattr(sm, "ap", None) if spec else None
        if parents is None or len(parents) != T:
            parents = list(range(-1, T - 1))
        if kernel:
            q = mlxdev.to_mx(query[0].transpose(0, 1).float().contiguous())
            if T == 1:
                out = mlxdev.attn_decode(
                    q[0], K, V, n, scale, params=mlxdev.attn_params(n, window=win), sinks=sinks, **kq
                )[None]
            else:
                out = mlxdev.attn_tree(q, K, V, past, list(parents), scale, sinks=sinks, window=win, **kq)
            return mlxdev.from_mx(out).to(query.dtype)[None], None
        qh = mlxdev.to_mx(query[0].contiguous())[None]
        if T == 1:
            start = mlxdev.attn_window_start(n, win)
            Kv, Vv = cl.mx_kv(start, n)
            mask = None
        else:
            tree = any(parents[j] != j - 1 for j in range(T))
            if tree:
                Kv, Vv = cl.mx_kv(0, n)
                depth = [0] * T
                allow = torch.zeros(T, n, dtype=torch.bool)
                for t in range(T):
                    depth[t] = 0 if parents[t] < 0 else depth[parents[t]] + 1
                    allow[t, mlxdev.attn_window_start(past + depth[t] + 1, win) : past] = True
                    cur = t
                    while cur >= 0:
                        allow[t, past + cur] = True
                        cur = parents[cur]
                mask = mlxdev.to_mx(allow)
            else:
                # a sliding layer's chunk needs only the window's worth of keys behind it: slicing there is exact and
                # makes the prefill O(T*win)
                start = max(0, past - win + 1) if win is not None else 0
                Kv, Vv = cl.mx_kv(start, n)
                p = past + m.arange(T, dtype=m.int32)[:, None]
                j = start + m.arange(n - start, dtype=m.int32)[None]
                mask = (j <= p) if win is None else ((j <= p) & (j > p - win))
        a = m.fast.scaled_dot_product_attention(
            qh.astype(Kv.dtype), Kv, Vv, scale=scale, mask=mask, sinks=sinks.astype(Kv.dtype)
        )
        return mlxdev.from_mx(a[0].transpose(1, 0, 2)).to(query.dtype)[None], None
    n = int(key.shape[-2])
    # a sliding layer's prefill chunk sees [n-T-win+1, n): dropping the rest is exact (the mask zeros it) and
    # O(T*win); decode (T == 1) is unchanged
    start = max(0, (n - T) - win + 1) if (win and T > 1) else 0
    if start:
        key, value = key[..., start:, :], value[..., start:, :]
    nk = n - start
    scores = torch.matmul(query.reshape(B, Hk, g * T, d), key.transpose(-1, -2)).reshape(B, Hq, T, nk) * scale
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, start:n]
    elif T > 1:
        # no mask handed over (a tier that owns the attention elsewhere): the causal one, and the window
        pos = torch.arange(n - T, n, device=scores.device)[:, None]
        j = start + torch.arange(nk, device=scores.device)[None]
        allow = (j <= pos) if win is None else ((j <= pos) & (j > pos - win))
        scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
    sinks = s_aux.reshape(1, -1, 1, 1).expand(B, -1, T, -1)
    combined = torch.cat([scores, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1].to(value.dtype)
    res = torch.matmul(probs.reshape(B, Hk, g * T, nk), value).reshape(B, Hq, T, d)
    return res.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    from transformers.masking_utils import AttentionMaskInterface, eager_mask, sdpa_mask
    from transformers.modeling_utils import AttentionInterface

    if "btb_sdpa" not in AttentionInterface._global_mapping:
        AttentionInterface.register("btb_sdpa", attention)
    if "btb_sdpa" not in AttentionMaskInterface._global_mapping:
        AttentionMaskInterface.register("btb_sdpa", sdpa_mask)
    if "btb_sinks" not in AttentionInterface._global_mapping:
        # the sink family's attention takes the reference's float mask (every row a tensor, never skipped)
        AttentionInterface.register("btb_sinks", attention_sinks)
    if "btb_sinks" not in AttentionMaskInterface._global_mapping:
        AttentionMaskInterface.register("btb_sinks", eager_mask)


@dataclass(frozen=True)
class Family:
    """What the engine knows of a model family: its kind, the transformers module and classes it builds layers
    from, and the capabilities the paths key on (never the kind): `dense` - the pre-norm block of GQA attention
    with rotary and a gated MLP, so the fused MLX forward, the tree verify, the batched and pipelined loops and
    the card graph apply; `hybrid` - linear-attention layers through the DeltaNet step and the hybrid forward;
    `norm_centered` - RMSNorm scales by 1 + weight; `kernel_layout` - separate q/k/v/o with q/k norms and
    gate/up/down, the layout the compiled kernels (the MLX megakernel and fused step, the CUDA step graph) are
    written for; `own`: the engine drives the layer; `fast`: the per-position host paths know the attention;
    `moe`: experts through the store; `mxfp4`: as MXFP4; `eager`: the module's own attention; `flat_cache`:
    one cache row per layer. Fused q/k/v and gate/up projections are read off the module, not declared."""

    kind: FamilyKind
    mod: Any = None
    layer: Any = None
    norm: Any = None
    rotary: Any = None
    mrope: bool = False
    attn_gate: bool = False
    own: bool = False
    streams: int = 1
    attn: str = LayerKind.FULL
    fast: bool = True
    moe: bool = False
    mxfp4: bool = False
    eager: bool = False
    flat_cache: bool = False
    dense: bool = False
    hybrid: bool = False
    norm_centered: bool = False
    kernel_layout: bool = False


def family(cfg: Any) -> Family:
    import importlib

    mt = str(getattr(cfg, "model_type", "") or "")
    if mt in ("qwen3_5", "qwen3_5_text"):
        mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        # Qwen3.5's norm scales by 1 + weight (weights stored around zero), unlike the families below
        _fuse_norm_cls(mod.Qwen3_5RMSNorm, centered=True)
        return Family(
            kind=FamilyKind.QWEN3_5,
            mod=mod,
            layer=mod.Qwen3_5DecoderLayer,
            norm=mod.Qwen3_5RMSNorm,
            rotary=mod.Qwen3_5TextRotaryEmbedding,
            mrope=True,
            attn_gate=True,
            hybrid=True,
            norm_centered=True,
        )
    if mt == "qwen3":
        mod = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
        _fuse_norm_cls(mod.Qwen3RMSNorm)
        _fuse_mlp_cls(mod.Qwen3MLP)
        return Family(
            kind=FamilyKind.QWEN3,
            mod=mod,
            layer=mod.Qwen3DecoderLayer,
            norm=mod.Qwen3RMSNorm,
            rotary=mod.Qwen3RotaryEmbedding,
            mrope=False,
            attn_gate=False,
            dense=True,
            kernel_layout=True,
        )
    if mt == "phi3":
        mod = importlib.import_module("transformers.models.phi3.modeling_phi3")
        _fuse_norm_cls(mod.Phi3RMSNorm)
        _fuse_mlp_cls(mod.Phi3MLP)
        return Family(
            kind=FamilyKind.PHI3,
            mod=mod,
            layer=mod.Phi3DecoderLayer,
            norm=mod.Phi3RMSNorm,
            rotary=mod.Phi3RotaryEmbedding,
            mrope=False,
            attn_gate=False,
            dense=True,
        )
    if mt in ("qwen4_exp", "qwen4_exp_text"):
        mod = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
        return Family(
            kind=FamilyKind.QWEN4,
            mod=mod,
            layer=mod.Qwen4ExpTextDecoderLayer,
            norm=None,
            rotary=mod.Qwen4ExpTextRotaryEmbedding,
            mrope=True,
            attn_gate=True,
            own=True,
            moe=True,
            streams=int(cfg.hc_count),
            attn=LayerKind.QWEN_SPARSE,
        )
    if mt == "gpt_oss":
        mod = importlib.import_module("transformers.models.gpt_oss.modeling_gpt_oss")
        # an attention sink per query head (a logit with no value) is not expressible through sdpa: the module's eager
        # attention runs (`fast` off)
        return Family(
            kind=FamilyKind.GPT_OSS,
            mod=mod,
            layer=mod.GptOssDecoderLayer,
            norm=mod.GptOssRMSNorm,
            rotary=mod.GptOssRotaryEmbedding,
            mrope=False,
            attn_gate=False,
            fast=False,
            moe=True,
            mxfp4=True,
            eager=True,
            flat_cache=True,
        )
    raise RuntimeError(f"unsupported model_type {mt!r}")


class _FamiliesMixin(_State):
    attention = staticmethod(attention)
    attention_sinks = staticmethod(attention_sinks)
    register_attention = staticmethod(register_attention)
    family = staticmethod(family)

    @staticmethod
    def _dense_key(key: str) -> bool:
        if ".mlp.experts." in key:
            # gpt-oss keeps one bias per expert; it is small and rides with the layer, the matrices stream
            return key.endswith("_proj_bias")
        return ".ngram_embedding." not in key

    def _shape_layer(self, layer: Any, i: int) -> Any:
        base = f"{self.prefix}layers.{i}."
        if self.fam.kind is FamilyKind.GPT_OSS:
            ex = layer.mlp.experts
            layer.mlp.experts = _Experts(
                self,
                base + "mlp.experts.",
                ex.num_experts,
                None,
                layer=i,
                mx=True,
                biases=True,
                gate=_Experts.gpt_oss_gate,
                alpha=ex.alpha,
                limit=ex.limit,
            )
            return layer
        if self.fam.kind is not FamilyKind.QWEN4:
            return layer
        ex = layer.mlp.experts
        layer.mlp.experts = _Experts(self, base + "mlp.experts.", ex.num_experts, ex.act_fn, layer=i)
        if getattr(layer, "ple", None) is not None:
            out_dtype = self.compute_dtype if self.compute_dtype is not None else torch.bfloat16
            layer.ple.ple_embedding.ngram_embedding = _NGramRows(
                self, base + "ple.ple_embedding.ngram_embedding.", int(self.cfg.split_ngram_parts), out_dtype
            )
        return layer

    @staticmethod
    def _named_tensors(module: Any) -> Iterable[tuple[str, torch.Tensor, bool]]:
        return [(n, p, False) for n, p in module.named_parameters()] + [(n, b, True) for n, b in module.named_buffers()]

    def _make_host_layer(self, i: int) -> Any:
        with self._meta:
            layer = self.fam.layer(self.cfg, i).eval()
        layer = self._shape_layer(layer, i)
        base = f"{self.prefix}layers.{i}."
        for name, _p, is_buf in self._named_tensors(layer):
            t = self._get(base + name)
            # widened once: norms, biases, sinks, the conv, Qwen4's router, gpt-oss's per-expert biases (a few MB,
            # added in float32)
            wide = t.is_floating_point() and (
                t.dim() <= 1 or "conv1d" in name or ".experts." in name or name.endswith("mlp.gate.weight")
            )
            self._set_param(layer, name, t.float() if wide else t, buffer=is_buf)
        for mname, m in list(layer.named_modules()):
            for cname, child in list(m.named_children()):
                if isinstance(child, torch.nn.Linear) and child.weight.dtype == torch.bfloat16:
                    key = base + (f"{mname}.{cname}" if mname else cname) + ".weight"
                    setattr(
                        m,
                        cname,
                        _HostLinear(child.weight.data, key=key, bias=None if child.bias is None else child.bias.data),
                    )
        if self.fam.kind is FamilyKind.GPT_OSS:
            # the attention finds the engine (its cache, the speculative pass's tree) through the module
            layer.self_attn._sm = self
            # the router's matvec through the batch-invariant kernels on the bf16 weight (F.linear's sums depend on
            # how many rows travel together)
            r = layer.mlp.router
            layer.mlp.router = _Router(
                _HostLinear(r.weight.data, key=base + "mlp.router.weight", bias=r.bias.data), int(r.top_k)
            )
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.layer_idx = i
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = i
        if self.mlx is not None and i in self.mlx_layers and i not in self.cold:
            self._bind_mlx_resident(layer)
            self._mlx_fuse(layer)
        if getattr(self, "_packed", None):
            self._bind_host_packed_layer(layer)
        return layer
