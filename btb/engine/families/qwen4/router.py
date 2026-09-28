# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's mixture router on a host layer, row-invariant: a row's logits, weights and experts are the same whether
it travels alone (the one-token step) or among a verify pass's rows. The reference (`Qwen4ExpTextTopKRouter`) runs
`F.linear`, whose sums change with how many rows travel together; here the logits come through the host's gemv
(`_HostLinear`) on the checkpoint's bf16 matrix with a float32 accumulate - the same function as the reference's
float32 linear over the widened matrix (widening bf16 is exact), its sums in another order but a row's own - and
the rest is the reference's routing: a float32 softmax over every expert, the top k, renormalised where the config
says so. `install` puts it in a mixture block's place of the reference's, for the trunk's layers and the MTP
layer's alike."""

from __future__ import annotations

from typing import Any

import torch

from ...host import _HostLinear, compute_fp32


class Qwen4Router(torch.nn.Module):
    """Qwen4's top-k router over `lin`, the router's matrix through the host's gemv. Returns (logits, weights,
    experts) as the reference does, the logits and weights in the dtype of the rows it is given (the dtype the
    engine's float32 router handed back before)."""

    lin: _HostLinear
    top_k: int
    num_experts: int
    norm_topk_prob: bool
    hidden_dim: int

    def __init__(self, lin: _HostLinear, top_k: int, norm_topk_prob: bool) -> None:
        super().__init__()
        self.lin = lin
        self.top_k = int(top_k)
        self.num_experts, self.hidden_dim = (int(s) for s in lin.weight.shape)
        self.norm_topk_prob = bool(norm_topk_prob)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = hidden_states.reshape(-1, self.hidden_dim)
        logits = self.lin(x.float())
        probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
        top, experts = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top = top / top.sum(dim=-1, keepdim=True)
        weights = top.to(logits.dtype)
        return logits.to(hidden_states.dtype), weights.to(hidden_states.dtype), experts


def install(mlp: Any, key: str) -> None:
    """`mlp.gate` (a `Qwen4ExpTextTopKRouter` whose weight the host read as stored, bf16 - see
    `Qwen4Family.widened`) as a `Qwen4Router` over it; `key` is the matrix's checkpoint name, which binds an FP8
    checkpoint's bytes to it after. A router held in another precision (a float32 one kept as stored) stays the
    reference's module, computing in float32 as the engine ran it before, not row-invariant."""
    gate: Any = getattr(mlp, "gate", None)
    if not isinstance(getattr(gate, "weight", None), torch.Tensor):
        return  # a dense block, or a router already installed
    w = gate.weight.data
    if w.dtype != torch.bfloat16:
        gate.weight.data = w.float()
        compute_fp32(gate)
        return
    mlp.gate = Qwen4Router(_HostLinear(w, key=key), int(gate.top_k), bool(gate.norm_topk_prob))
