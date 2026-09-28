# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's sparse attention on a host layer through the native kernel, one list of cache rows per query row
(`Native.attn_nodes`): the one-token step and a speculative verify pass alike, so a tree node attends the rows its
committed step will, in the same order, through the same arithmetic - bit for bit. The reference runs sdpa over
the pass's rows together, whose sums change with how many rows travel together.

Each row's list is what the reference lets it see: the pass's mask (causal, or a verify pass's tree - the prefix,
the row's ancestors, itself) combined with the indexer's selection (`qsa.select`, exact row for row), its cache
rows in ascending order. A tree's parents precede their children, so a node's rows ascend in its path's order, the
order its steps lay them in the cache. The queries, keys and values are made as the reference makes them, and the
keys and values written to the cache are the reference's. A prefill (several rows outside speculation), a batch,
a pass on the card or a library without the kernel takes the module's own forward."""

from __future__ import annotations

import types
import weakref
from typing import Any

import torch

from ...native import Native
from .rows import each


def install(attn: Any, sm: Any) -> None:
    """`attn` (a host layer's `Qwen4ExpTextAttention`) forwarding through `_forward`; `sm` held weakly (the engine
    owns the layer)"""
    attn._btb_sm = weakref.ref(sm)
    attn.forward = types.MethodType(_forward, attn)


def _forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kw: Any,
) -> tuple[torch.Tensor, None]:
    sm = self._btb_sm()
    B, T, _ = hidden_states.shape
    ours = (
        sm is not None
        and past_key_values is not None
        and attention_mask is not None
        and attention_mask.dim() == 4
        and B == 1
        and hidden_states.device.type == "cpu"
        and Native.attn_nodes is not None
        and (T == 1 or bool(getattr(sm, "aq", False)))
    )
    if not ours:
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    assert attention_mask is not None  # `ours` held
    selected = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    chosen = selected if selected.dtype == torch.bool else selected == 0
    allowed = (visible & chosen)[0, 0]  # [T, kv]: row t's cache rows
    counts = allowed.sum(dim=-1)
    if not bool((counts > 0).all()):
        # the reference's softmax over no key is NaN; the kernel takes no empty list
        raise RuntimeError(f"Qwen4's sparse attention, layer {self.layer_idx}: a query row sees no key")

    # the queries, the gate, the keys and the values exactly as the reference makes them
    cos, sin = (x[:, -T:, :] for x in position_embeddings)
    hd = int(self.head_dim)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, hd)
    q, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    k_all, v_all = past_key_values.update(k, v, self.layer_idx)
    kc, vc = k_all[0], v_all[0]  # [hk, n, d]
    n = int(kc.shape[-2])
    if int(allowed.shape[-1]) != n:
        raise RuntimeError(
            f"Qwen4's sparse attention, layer {self.layer_idx}: the mask covers {allowed.shape[-1]} keys, the "
            f"cache holds {n}"
        )
    if kc.dtype not in (torch.bfloat16, torch.float32) or vc.dtype != kc.dtype:
        kc, vc = kc.float(), vc.float()
    # the kernel reads a head's rows packed one after another, the heads at a stride of their own (the cache's
    # buffer, cut to its rows, is laid so)
    if kc.stride(-1) != 1 or kc.stride(-2) != hd:
        kc = kc.contiguous()
    if vc.stride(-1) != 1 or vc.stride(-2) != hd:
        vc = vc.contiguous()

    # every row's list at once: the allowed (row, key) pairs row-major are each row's keys ascending, row by row
    idx = allowed.nonzero()[:, 1].to(torch.int32).contiguous()
    offs = torch.zeros(T + 1, dtype=torch.int32)
    offs[1:] = counts.cumsum(0).to(torch.int32)
    qf = q[0].transpose(0, 1).float().contiguous()  # [T, hq, d]
    out = torch.empty(qf.shape, dtype=torch.float32)
    Native.attn_nodes(qf, kc, vc, offs, idx, float(self.scaling), out)
    a = out.view(*input_shape, -1).to(hidden_states.dtype)
    # the gate's sigmoid a row at a time, as the step's (rows.py)
    return self.o_proj(a * each(torch.sigmoid, gate, T)), None
