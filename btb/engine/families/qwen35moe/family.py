# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen3.5-MoE (`qwen3_5_moe`): Qwen3.5's hybrid of gated DeltaNet and gated full-attention layers, each layer's MLP
a mixture of experts - a top-k router over the routed experts, and a shared expert behind a sigmoid gate - and
Qwen3.5's drafter over its `mtp.*` head, whose layer is a mixture too."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Self

from ....kinds import FamilyKind
from ....mxfp4 import stored_mxfp4
from ...fused import _fuse_norm_cls
from ...host import _Experts
from ..base import _flags
from ..qwen4.router import install as install_router
from ..qwen4.router import install_card as install_card_router
from ..qwen4.router import mixture
from ..qwen35.family import Qwen35Family

if TYPE_CHECKING:
    import torch

    from ...state import _State
    from ..base import Mixture


class Qwen35MoeFamily(Qwen35Family):
    """Qwen3.5's layers with the experts through the engine's store (`_Experts`) and the router row-invariant, as
    Qwen4's mixture runs (qwen4/router.py): a host layer's logits through the host's gemv, the card's at the fixed
    shape"""

    KIND = FamilyKind.QWEN3_5_MOE

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
        _fuse_norm_cls(mod.Qwen3_5MoeRMSNorm, centered=True)
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Qwen3_5MoeDecoderLayer,
            norm=mod.Qwen3_5MoeRMSNorm,
            rotary=mod.Qwen3_5MoeTextRotaryEmbedding,
            **_flags(cls.KIND),
        )

    def widened(self, name: str) -> bool:
        # the router's matrix stays as stored: `finish_host_layer` multiplies it through the host's gemv
        return not name.endswith("mlp.gate.weight") and super().widened(name)

    def shape_layer(self, sm: _State, layer: Any, i: int, base: str | None = None) -> Any:
        base = f"{sm.prefix}layers.{i}." if base is None else base
        ex = layer.mlp.experts
        layer.mlp.experts = _Experts(
            sm,
            base + "mlp.experts.",
            ex.num_experts,
            ex.act_fn,
            layer=i,
            mx=stored_mxfp4(self.mxfp4, sm.gguf),
            f8=sm.fp8_experts,
        )
        install_card_router(layer.mlp)
        return layer

    def finish_host_layer(self, sm: _State, layer: Any, i: int, base: str | None = None) -> None:
        install_router(layer.mlp, (f"{sm.prefix}layers.{i}." if base is None else base) + "mlp.gate.weight")

    def mixture(self, mlp: torch.nn.Module) -> Mixture | None:
        return mixture(mlp)
