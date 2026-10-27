# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3.5 (`qwen3_5`): the hybrid of gated DeltaNet and gated full-attention layers, its norms scaling by one
plus their weight, and the MTP drafting head its checkpoints ship (`drafter.py`)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Self

import torch

from ....kinds import FamilyKind, LayerKind
from ...fused import _fuse_norm_cls, chunk_gated_delta_rule, fast_causal_conv1d
from ..base import Family, _flags
from .drafter import FC, LAYER, NORM, NORM_E, NORM_H, Qwen35Drafter

if TYPE_CHECKING:
    from ....mlx.mega import Block
    from ...drafter import MTPDrafter
    from ...state import _State


class Qwen35Family(Family):
    """Qwen3.5's layers as transformers makes them; its own are the convolution the MLX tier runs and the
    drafter over its `mtp.*` head"""

    KIND = FamilyKind.QWEN3_5

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        # Qwen3.5's norm scales by 1 + weight (weights stored around zero), unlike the plain block's
        _fuse_norm_cls(mod.Qwen3_5RMSNorm, centered=True)
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Qwen3_5DecoderLayer,
            norm=mod.Qwen3_5RMSNorm,
            rotary=mod.Qwen3_5TextRotaryEmbedding,
            **_flags(cls.KIND),
        )

    def prepare(self, sm: _State) -> None:
        # the DeltaNet's chunked rule a block at a time (`chunk_gated_delta_rule`): a prompt's bits whatever its chunks
        # where they cut at its blocks, and whatever the thread count
        self.mod.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        if sm.mlx is not None:
            # the DeltaNet's convolution as shifted multiply-adds, not F.conv1d at its cost on the CPU build
            self.mod.causal_conv1d_fn = fast_causal_conv1d

    @property
    def mega(self) -> bool:
        # the hybrid block composed below; a mixture's MLP (Qwen3.5-MoE's) is not in the megakernel
        return not self.moe

    def mega_layer(self, b: Block, i: int) -> None:
        # a DeltaNet layer or a gated attention one (its output times the sigmoid of the gate beside each q head), each
        # pre-normed, then the dense MLP
        if b.mp.linear[i]:
            dl = b.deltanet(i)
            la = dl.linear_attn
            x = b.norm(dl.input_layernorm)
            b.add(b.linear(b.delta(i, b.linear(x, b.delta_in(la)), la), la.out_proj))
            post, mlp = dl.post_attention_layernorm, dl.mlp
        else:
            layer = b.dense(i)
            at = layer.self_attn
            x = b.norm(layer.input_layernorm)
            qkv = b.linear(x, b.qkv(at))
            b.add(b.linear(b.gate(b.attention(i, qkv, at), qkv), at.o_proj))
            post, mlp = layer.post_attention_layernorm, layer.mlp
        x2 = b.norm(post)
        b.add(b.linear(b.gated(b.linear(x2, b.gate_up(mlp))), mlp.down_proj))

    def drafter_cls(self) -> type[MTPDrafter] | None:
        return Qwen35Drafter

    def drafting_names(self, cfg: Any) -> list[str]:
        # the head's own tensors, and its layer's: one of the trunk's full-attention kind (a mixture's experts stacked,
        # as the store reads them)
        idx = [LayerKind.of(t) for t in cfg.layer_types].index(LayerKind.FULL)
        with torch.device("meta"):
            layer = self.layer(cfg, idx)
        return [FC, NORM_E, NORM_H, NORM] + [LAYER + n for n, _ in (*layer.named_parameters(), *layer.named_buffers())]
