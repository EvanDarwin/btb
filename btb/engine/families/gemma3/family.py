# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Gemma 3 (`gemma3`): the sandwich-normed block with q/k norms, its input embedding scaled by sqrt(hidden), and
sliding local layers beside the global ones, each kind with its own rope."""

from __future__ import annotations

import importlib
from typing import Any, Self

from ....kinds import FamilyKind
from ..base import Family, _flags


class Gemma3Family(Family):
    """Gemma 3's layers as transformers makes them; what sets them apart the paths read off the flags"""

    KIND = FamilyKind.GEMMA3

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.gemma3.modeling_gemma3")
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Gemma3DecoderLayer,
            norm=mod.Gemma3RMSNorm,
            rotary=mod.Gemma3RotaryEmbedding,
            **_flags(cls.KIND),
        )
