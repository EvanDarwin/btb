# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The attention forwards the engine registers with transformers: its SDPA (`btb_sdpa`), which every family's
modules run through, and gpt-oss's sinks (`btb_sinks`, gpt_oss/sinks.py)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .gpt_oss.sinks import attention_sinks


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
