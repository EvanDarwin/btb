# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3 (`qwen3`): the plain pre-norm block in the layout the compiled kernels are written for."""

from __future__ import annotations

import importlib
from typing import Any, Self

from ....kinds import FamilyKind
from ...fused import _fuse_mlp_cls, _fuse_norm_cls
from ..base import Family, _flags


class Qwen3Family(Family):
    """Qwen3's layers, the plain block's behaviour throughout"""

    KIND = FamilyKind.QWEN3

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.qwen3.modeling_qwen3")
        _fuse_norm_cls(mod.Qwen3RMSNorm)
        _fuse_mlp_cls(mod.Qwen3MLP)
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Qwen3DecoderLayer,
            norm=mod.Qwen3RMSNorm,
            rotary=mod.Qwen3RotaryEmbedding,
            **_flags(cls.KIND),
        )
