# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""gpt-oss: a mixture of MXFP4 experts behind a biased router, and an attention with a sink logit per query head
that sdpa cannot express, so its layers run their own attention (`sinks.py`) over the engine's cache."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Self

import torch

from ....kinds import FamilyKind
from ...host import _Experts, _HostLinear, _Router
from ..attention import register_attention
from ..base import Family, _flags
from .sinks import ScoresWorkspace, close_scores, open_scores

if TYPE_CHECKING:
    from ...state import _State


class GptOssFamily(Family):
    """gpt-oss's layers: the experts through the store with their per-expert biases and gated activation, the
    router through the batch-invariant kernels, the attention its own"""

    KIND = FamilyKind.GPT_OSS

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.gpt_oss.modeling_gpt_oss")
        # an attention sink per query head is not expressible through sdpa, so its eager attention runs (fast off)
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.GptOssDecoderLayer,
            norm=mod.GptOssRMSNorm,
            rotary=mod.GptOssRotaryEmbedding,
            **_flags(cls.KIND),
        )

    def prepare(self, sm: _State) -> None:
        # the sinks are not expressible through sdpa: `attention_sinks` runs the reference's arithmetic on the CPU,
        # the engine's kernels over an MLX cache
        register_attention()
        sm.cfg._attn_implementation = "btb_sinks"

    def dense_key(self, key: str) -> bool:
        # one bias per expert: small, it rides with the layer while the expert matrices stream
        if ".mlp.experts." in key and key.endswith("_proj_bias"):
            return True
        return super().dense_key(key)

    def shape_layer(self, sm: _State, layer: Any, i: int) -> Any:
        ex = layer.mlp.experts
        layer.mlp.experts = _Experts(
            sm,
            f"{sm.prefix}layers.{i}.mlp.experts.",
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

    def finish_host_layer(self, sm: _State, layer: Any, i: int) -> None:
        # the attention finds the engine (its cache, the speculative pass's tree) through the module
        layer.self_attn._sm = sm
        # the router's matvec through the batch-invariant kernels on the bf16 weight (F.linear's sums depend on
        # how many rows travel together)
        r = layer.mlp.router
        layer.mlp.router = _Router(
            _HostLinear(r.weight.data, key=f"{sm.prefix}layers.{i}.mlp.router.weight", bias=r.bias.data), int(r.top_k)
        )

    def open_sweep(self, sm: _State, C: int, keys: int, B: int) -> bool:
        # the sink attention's scores - the buffers that grow with every chunk's position - taken once, sized for
        # the last chunk: every chunk's are views of them, never the allocator's cache (`ScoresWorkspace`)
        open_scores(sm.dev, ScoresWorkspace(self._scores_bytes(sm, C, keys, B), sm.dev))
        return True

    def close_sweep(self, sm: _State) -> None:
        close_scores(sm.dev)

    @staticmethod
    def _scores_bytes(sm: _State, C: int, keys: int, B: int) -> int:
        """one of a sweep's two score buffers (`ScoresWorkspace`): a chunk of `C` rows' scores against `keys` keys and
        the sink column, every query head, in the dtype the sinks' join promotes them to"""
        cd = sm.compute_dtype
        nb = 4 if (cd is not None and cd != torch.bfloat16) else 2
        return int(B) * int(sm.cfg.num_attention_heads) * int(C) * (int(keys) + 1) * max(nb, sm._sinks_bytes())
