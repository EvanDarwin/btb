# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The attention forwards the engine registers with transformers: its SDPA (`btb_sdpa`), which every family's
modules run through, and gpt-oss's sinks (`btb_sinks`, gpt_oss/sinks.py)."""

from __future__ import annotations

import contextvars
from typing import Any

import torch
import torch.nn.functional as F

from ..fixed_rows import KeyRows
from .gpt_oss.sinks import attention_sinks

# the engine whose card layer's module attends now (`_run_card_layer`): past a prompt's first rows its attention runs
# on btb's kernels (`_card_attention`) - a paged layer's rows through the card's row map, a contiguous one's where they
# lie, one set of bits for the two
CARD_ATTENTION: contextvars.ContextVar[Any] = contextvars.ContextVar("btb_card_attention", default=None)


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


def one_query_rows(
    module: Any, n: int, attention_mask: torch.Tensor | KeyRows | None, device: torch.device
) -> torch.Tensor | None:
    """The cache rows a one-query call attends: as `KeyRows` names them, or as its layer's window leaves them (the
    last `sliding_window` of its `n`, as transformers' sliding mask leaves a step), or None for all `n`"""
    if isinstance(attention_mask, KeyRows):
        return attention_mask.idx
    win = int(getattr(module, "sliding_window", None) or 0)
    if win and n > win:
        return torch.arange(n - win, n, device=device)
    return None


def attend_one(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | KeyRows | None,
    g: int,
    scaling: float | None,
) -> torch.Tensor | None:
    """A one-query row's attention as the greedy step at its position computes it, whoever calls: the rows it attends
    gathered in order (`one_query_rows`), its keys and values made contiguous at [1, Hq, rows, D] - every head's -
    and sdpa over them with no mask. A step and a speculative pass's node over the same rows then make the same call
    on the same shapes, so the same bits (a node's bool mask moved sdpa off its flash path, and the node saw its
    siblings' rows masked where the step never had them). None where the call is not one this form takes: a batch,
    or a mask tensor on a layer with no window (a caller's padding), which the general path reads."""
    if query.shape[0] != 1 or query.shape[2] != 1:
        return None
    if isinstance(attention_mask, torch.Tensor) and not int(getattr(module, "sliding_window", None) or 0):
        return None
    n = int(key.shape[-2])
    idx = one_query_rows(module, n, attention_mask, key.device)
    if idx is not None:
        key, value = key.index_select(-2, idx), value.index_select(-2, idx)
    b, hk, t, d = key.shape
    if g > 1:
        key = key[:, :, None].expand(b, hk, g, t, d).reshape(b, hk * g, t, d)
        value = value[:, :, None].expand(b, hk, g, t, d).reshape(b, hk * g, t, d)
    else:
        key, value = key.contiguous(), value.contiguous()
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False, scale=scaling)
    return out.transpose(1, 2).contiguous()


def _rows_apart(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    g: int,
    scaling: float | None,
) -> torch.Tensor:
    """Each query row of one sequence's pass through its own one-row call (`attend_one`) over the cache rows its mask
    row lets it see, in ascending order: the call a one-token step makes over the rows it sees, so a speculative pass's
    node and its step make the same call on the same rows and agree to the bit. For a layer whose mask is its own -
    Qwen4's sparse attention, the indexer's picks per row (`btb_rows_apart`, which says for which passes): sdpa over the
    rows together with that mask picks its kernel by how many rows travel together. Returns [1, T, Hq, D]"""
    T, S = int(query.shape[2]), int(key.shape[-2])
    rows = attention_mask[0, 0, :, :S]
    outs = []
    for t in range(T):
        one = attend_one(module, query[:, :, t : t + 1], key, value, KeyRows(rows[t].nonzero().flatten()), g, scaling)
        assert one is not None  # one row of one sequence, its rows named: attend_one's own form
        outs.append(one)
    return outs[0] if T == 1 else torch.cat(outs, dim=1)


def grouped_chunk_ok(dtype: torch.dtype, head_dim: int, device: torch.device) -> bool:
    """whether a chunk's queries of `dtype` and `head_dim` on `device` take the efficient kernel's own call
    (`_grouped_chunk`): a half dtype, the head a multiple of 8 dims up to 256, and bf16 on a card that has the
    kernel's bf16 build (compute capability 8.0 on: sdpa checks this before choosing the kernel, a direct call does
    not)"""
    if dtype not in (torch.bfloat16, torch.float16) or head_dim % 8 or head_dim > 256:
        return False
    return dtype == torch.float16 or torch.cuda.get_device_capability(device)[0] >= 8


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
    from ..paged import PagedError, PagedKV

    paged = isinstance(key, PagedKV)
    sm = CARD_ATTENTION.get()
    if (paged or sm is not None) and query.device.type == "cuda" and dropout == 0.0 and not kw.get("softcap"):
        out = sm._card_attention(module, query, key, value, attention_mask, scaling) if sm is not None else None
        if out is not None:
            return out, None
    if paged:
        raise PagedError(
            f"layer {getattr(module, 'layer_idx', '?')}: a paged card layer's rows reached an attention that cannot "
            "read them through the card's row map"
        )
    if query.device.type != "cuda":
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        if isinstance(attention_mask, ChunkCausal):
            attention_mask = attention_mask.mask(query.shape[2], key.shape[-2], query.device)
        elif isinstance(attention_mask, KeyRows):
            # off the card the rows as the bool mask they stand for: the host's sdpa takes the node that way
            allow = torch.zeros(int(key.shape[-2]), dtype=torch.bool, device=query.device)
            allow[attention_mask.idx.to(query.device)] = True
            attention_mask = allow.view(1, 1, 1, -1)
        if key.dtype != query.dtype:
            # a host layer's rows kept in the card's bf16 (`host_kv_dtype`), its queries the host's float32: widened
            # for the pass, as sdpa takes one dtype (the native kernels read them as kept)
            key, value = key.to(query.dtype), value.to(query.dtype)
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw
        )
    g = int(getattr(module, "num_key_value_groups", 1) or 1)
    apart = getattr(module, "btb_rows_apart", None)
    if (
        apart is not None
        and dropout == 0.0
        and query.shape[0] == 1
        and isinstance(attention_mask, torch.Tensor)
        and attention_mask.dim() == 4
        and attention_mask.dtype == torch.bool
        and apart(int(query.shape[2]))
    ):
        return _rows_apart(module, query, key, value, attention_mask, g, scaling), None
    one = (
        attend_one(module, query, key, value, attention_mask, g, scaling)
        if dropout == 0.0 and not isinstance(attention_mask, ChunkCausal)
        else None
    )
    if one is not None:
        return one, None
    if isinstance(attention_mask, KeyRows):
        raise RuntimeError("[attention] a KeyRows mask reached a call of more than one query row")
    if isinstance(attention_mask, ChunkCausal):
        T, S = query.shape[2], key.shape[-2]
        if (
            query.shape[0] == 1
            and S == attention_mask.past + T
            and dropout == 0.0
            and key.dtype == query.dtype
            and grouped_chunk_ok(query.dtype, int(query.shape[-1]), query.device)
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
