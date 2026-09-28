# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Phi-3 (`phi3`): the plain pre-norm block with q, k and v as one projection and gate and up as another, its
rotary over part of each head where the config says so."""

from __future__ import annotations

import importlib
from typing import Any, Self

from ....kinds import FamilyKind
from ...fused import _fuse_mlp_cls, _fuse_norm_cls
from ..base import Family, _flags


class Phi3Family(Family):
    """Phi-3's layers, the plain block's behaviour throughout (its fused projections are read off the module)"""

    KIND = FamilyKind.PHI3

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.phi3.modeling_phi3")
        _fuse_norm_cls(mod.Phi3RMSNorm)
        _fuse_mlp_cls(mod.Phi3MLP)
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Phi3DecoderLayer,
            norm=mod.Phi3RMSNorm,
            rotary=mod.Phi3RotaryEmbedding,
            **_flags(cls.KIND),
        )
