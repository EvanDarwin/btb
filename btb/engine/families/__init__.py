# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The model families the engine serves, each a `Family` subclass in its own package here (base.py the plain
block they build on): `family()` makes the one a config names, and the engine's mixin below builds a host layer
through it. The attention forwards the engine registers with transformers are attention.py."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from ...kinds import FAMILY_NAMES, KIND_OF, FamilyKind, ModelType
from ...options import UnsupportedModelType
from ..host import _HostLinear, compute_fp32
from ..native import Native
from ..state import _State
from .attention import attention, register_attention
from .base import Family
from .gemma3.family import Gemma3Family
from .gpt_oss.family import GptOssFamily
from .gpt_oss.sinks import attention_sinks
from .phi3.family import Phi3Family
from .qwen3.family import Qwen3Family
from .qwen4.family import Qwen4Family
from .qwen35.family import Qwen35Family


def act_name(cfg: Any) -> str:
    """the MLP activation's name: Gemma's config calls it `hidden_activation` (gelu_pytorch_tanh), the rest
    `hidden_act`"""
    return str(getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", "silu"))


# the model_type view of those names, through KIND_OF (the "_text" variants read as their base family)
NAME_OF: dict[ModelType, str] = {mt: FAMILY_NAMES[fk] for mt, fk in KIND_OF.items()}
SUPPORTED_MODEL_TYPES = tuple(KIND_OF)  # the served model_types, from the one declaration in kinds.py

# each family's class, by the kind it builds
FAMILIES: dict[FamilyKind, type[Family]] = {
    cls.KIND: cls for cls in (Qwen3Family, Qwen35Family, Phi3Family, Qwen4Family, GptOssFamily, Gemma3Family)
}


def family(cfg: Any) -> Family:
    raw = str(getattr(cfg, "model_type", "") or "")
    try:
        fk = KIND_OF[ModelType(raw)]  # route by FamilyKind, not a model_type string match; "_text" shares a base
    except (ValueError, KeyError):
        raise UnsupportedModelType(raw, list(FAMILY_NAMES.values())) from None
    cls = FAMILIES.get(fk)
    if cls is None:
        # `raw` is served (KIND_OF answered it) but no class here builds its family: a table this package has
        # fallen behind, not a model the user cannot run
        raise RuntimeError(f"{fk} is declared in kinds.KIND_OF but no Family subclass in families/ builds it")
    return cls.build(cfg)


class _FamiliesMixin(_State):
    attention = staticmethod(attention)
    attention_sinks = staticmethod(attention_sinks)
    register_attention = staticmethod(register_attention)
    family = staticmethod(family)

    @staticmethod
    def _named_tensors(module: Any) -> Iterable[tuple[str, torch.Tensor, bool]]:
        return [(n, p, False) for n, p in module.named_parameters()] + [(n, b, True) for n, b in module.named_buffers()]

    def _make_host_layer(self, i: int) -> Any:
        with self._meta:
            layer = self.fam.layer(self.cfg, i).eval()
        layer = self.fam.shape_layer(self, layer, i)
        base = f"{self.prefix}layers.{i}."
        fp32_owners: set[str] = set()
        # an FP8 linear of a layer the host kernels run is multiplied as stored (`_HostLinear.f8`), its weight a
        # shape with no bytes behind it; an MLX-bound or cold layer reads a widened copy into its slots
        f8_host = (
            Native.gemv_fp8 is not None and i not in self.cold and not (self.mlx is not None and i in self.mlx_layers)
        )
        f8_keys: set[str] = set()
        for name, _p, is_buf in self._named_tensors(layer):
            key = base + name
            f8 = self._fp8(key)
            # widened once: norms, biases and sinks (below), and the matrices the family runs in float32 - the conv,
            # a router, gpt-oss's per-expert biases
            widened = self.fam.widened(name)
            if f8 and f8_host and not widened and len(shape := self._shape(key)) == 2:
                f8_keys.add(key)
                self._set_param(layer, name, torch.zeros((), dtype=torch.bfloat16).expand(*shape), buffer=is_buf)
                continue
            t = self._get(key, stored=True)
            wide = t.is_floating_point() and (t.dim() <= 1 or widened)
            self._set_param(layer, name, t.float() if wide else t.bfloat16() if f8 else self._held(t), buffer=is_buf)
            # a widened matrix is its module's conv or linear operand; the per-expert biases are added by
            # `_Experts`, which casts them itself
            if wide and t.dim() >= 2 and ".experts." not in name:
                fp32_owners.add(name.rpartition(".")[0])
        for owner in sorted(fp32_owners):
            compute_fp32(layer.get_submodule(owner))
        for mname, m in list(layer.named_modules()):
            for cname, child in list(m.named_children()):
                if isinstance(child, torch.nn.Linear) and child.weight.dtype == torch.bfloat16:
                    key = base + (f"{mname}.{cname}" if mname else cname) + ".weight"
                    setattr(
                        m,
                        cname,
                        _HostLinear(child.weight.data, key=key, bias=None if child.bias is None else child.bias.data),
                    )
        self.fam.finish_host_layer(self, layer, i)
        if f8_keys:
            for m in layer.modules():
                if isinstance(m, _HostLinear) and m.key in f8_keys:
                    m.f8 = self._f8_weights(m.key)[0]
            self.fp8_layers.add(i)
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
