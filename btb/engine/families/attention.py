# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The attention forwards the engine registers with transformers: its SDPA (`btb_sdpa`), which every family's
modules run through, and gpt-oss's sinks (`btb_sinks`, gpt_oss/sinks.py)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .gpt_oss.sinks import attention_sinks


class ChunkCausal:
    """a prefill chunk's causal mask as what it is rather than as a tensor: rows at positions past .. past + T - 1,
    row t seeing keys 0 .. past + t. The layer-by-layer sweep hands it to a card chunk's layers in place of the
    [T, past + T] mask transformers builds from the same rule, which `attention` materializes only where its own
    call cannot take it"""

    __slots__ = ("past",)

    def __init__(self, past: int) -> None:
        self.past = int(past)

    def mask(self, T: int, S: int, device: torch.device) -> torch.Tensor:
        """the rule as transformers' bool mask [1, 1, T, S]"""
        rows = torch.arange(self.past, self.past + T, device=device)[:, None]
        return (torch.arange(S, device=device)[None, :] <= rows)[None, None]


def _grouped_chunk(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: int, scaling: float | None
) -> torch.Tensor:
    """one sequence's chunk [1, Hq, T, D] over its keys [1, Hk, S, D] (S = past + T) through the efficient kernel's
    own call: the group's g query heads in the batch - batch i holds head h g + i of every group h - over the keys
    and values as the cache lays them out, a stride-0 batch, and the lower-right causal rule as the kernel's mask
    type. sdpa took the keys widened to every head (a copy of the whole prefix a layer and chunk: 335 MB at 40k)
    and the mask materialized, and ran the same kernel over them: the same bits, 32-62% faster on Qwen3-0.6B's
    heads. Returns [1, T, Hq, D]"""
    _, Hq, T, D = query.shape
    Hk, S = key.shape[1], key.shape[2]
    qg = query[0].reshape(Hk, g, T, D).permute(1, 2, 0, 3)
    kg = key[0].permute(1, 0, 2)[None].expand(g, S, Hk, D)
    vg = value[0].permute(1, 0, 2)[None].expand(g, S, Hk, D)
    out = torch.ops.aten._efficient_attention_forward(
        qg,
        kg,
        vg,
        bias=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=0.0,
        custom_mask_type=2,  # causal from the bottom right: row t sees keys 0 .. S - T + t
        compute_log_sumexp=False,
        scale=scaling,
        seqlen_k=None,
    )[0]
    # [g, T, Hk, D] -> [1, T, Hq, D], head h g + i at h * g + i
    return out.permute(1, 2, 0, 3).reshape(1, T, Hq, D)


def attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | ChunkCausal | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kw: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if query.device.type != "cuda":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        if isinstance(attention_mask, ChunkCausal):
            attention_mask = attention_mask.mask(query.shape[2], key.shape[-2], query.device)
        if key.dtype != query.dtype:
            # a host layer's rows kept in the card's bf16 (`host_kv_dtype`), its queries the host's float32: widened
            # for the pass, as sdpa takes one dtype (the native kernels read them as kept)
            key, value = key.to(query.dtype), value.to(query.dtype)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw
        )
    g = int(getattr(module, "num_key_value_groups", 1) or 1)
    if isinstance(attention_mask, ChunkCausal):
        T, S = query.shape[2], key.shape[-2]
        if (
            query.shape[0] == 1
            and S == attention_mask.past + T
            and dropout == 0.0
            and query.dtype in (torch.bfloat16, torch.float16)
            and key.dtype == query.dtype
            and query.shape[-1] % 8 == 0
            and query.shape[-1] <= 256
        ):
            return _grouped_chunk(query, key, value, g, scaling), None
        attention_mask = attention_mask.mask(T, S, query.device)
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
