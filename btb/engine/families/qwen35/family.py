# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3.5 (`qwen3_5`): the hybrid of gated DeltaNet and gated full-attention layers, its norms scaling by one
plus their weight, and the MTP drafting head its checkpoints ship (`drafter.py`)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Self

from ....kinds import FamilyKind
from ...fused import _fuse_norm_cls, chunk_gated_delta_rule, fast_causal_conv1d
from ..base import Family, _flags
from .drafter import Qwen35Drafter

if TYPE_CHECKING:
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

    def drafter_cls(self) -> type[MTPDrafter] | None:
        return Qwen35Drafter
