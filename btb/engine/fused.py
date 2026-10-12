# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Fused replacements for transformers' modules: the RMSNorm, rotary and MLP forwards the engine installs on the
families' classes, the DeltaNet's causal conv as shifted multiply-adds, and its chunked rule a block at a time."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F


def _fused_rms_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    # one F.rms_norm kernel for the module's seven; CUDA only (it multiplies by the weight in fp32: within the GPU
    # receipts' 1e-4, not the CPU's exact 0). BTB_FUSED_NORM=0 restores the module's
    if x.is_cuda:
        eps = getattr(self, "variance_epsilon", None)
        if eps is None:
            eps = getattr(self, "eps", 1e-6)
        if type(self)._btb_centered:
            # a zero-centred norm (Qwen3.5's): the scale is 1 + weight, with weights stored around zero, and the
            # reference multiplies in float32 before the cast back. `rms_norm(x, weight)` would scale by the
            # raw weight - about zero - and the residual stream collapses to a constant token (the card and the
            # host disagreed by 0.79 in the logits at prompt position 0 on the hybrid fixture, whatever the
            # tiering, because the final norm runs on the card)
            out = torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), None, eps)
            return (out * (1.0 + self.weight.float())).type_as(x)
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, eps)
    return type(self)._btb_orig_rms(self, x)


def _fuse_norm_cls(cls: Any, centered: bool = False) -> None:
    """Install the fused forward on a family's RMSNorm class. `centered` names the convention the class
    scales by: `weight * normed` (Llama, Qwen3, Phi3; weights around one) or `(1 + weight) * normed` (Qwen3.5;
    weights around zero) - the two are not interchangeable, and the family that imports the class says which."""
    if cls is None or os.environ.get("BTB_FUSED_NORM", "1") == "0" or getattr(cls, "_btb_fused", False):
        return
    cls._btb_orig_rms = cls.forward
    cls._btb_fused = True
    cls._btb_centered = bool(centered)
    cls.forward = _fused_rms_forward


def _fused_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    # apply_rotary_pos_emb with addcmul folding the last mul and add: two fewer nodes a layer, bit for bit the
    # reference on the full-rotary path
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rd = cos.shape[-1]
    d = q.shape[-1]
    if rd == d:
        h = d // 2
        qr = torch.cat((-q[..., h:], q[..., :h]), dim=-1)
        kr = torch.cat((-k[..., h:], k[..., :h]), dim=-1)
        return torch.addcmul(q * cos, qr, sin), torch.addcmul(k * cos, kr, sin)
    # partial rotary (phi3 with rotary_dim < head_dim): rotate the first rd dims, pass the rest through
    h = rd // 2
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    qr = torch.cat((-q_rot[..., h:], q_rot[..., :h]), dim=-1)
    kr = torch.cat((-k_rot[..., h:], k_rot[..., :h]), dim=-1)
    q_embed = torch.cat((torch.addcmul(q_rot * cos, qr, sin), q_pass), dim=-1)
    k_embed = torch.cat((torch.addcmul(k_rot * cos, kr, sin), k_pass), dim=-1)
    return q_embed, k_embed


def _fused_mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    # SwiGLU with the product written in place (mul_): one allocation fewer between the GEMMs, bit for bit the
    # module's act(gate) * up. CUDA only; BTB_FUSED_MLP=0 restores it
    if x.is_cuda:
        if hasattr(self, "gate_up_proj"):  # phi3 packs gate and up into one projection
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
            act = getattr(self, "activation_fn", None) or self.act_fn
            return self.down_proj(act(gate).mul_(up))
        act = getattr(self, "act_fn", None) or self.activation_fn
        return self.down_proj(act(self.gate_proj(x)).mul_(self.up_proj(x)))
    return type(self)._btb_orig_mlp(self, x)


def _fuse_mlp_cls(cls: Any) -> None:
    if cls is None or os.environ.get("BTB_FUSED_MLP", "1") == "0" or getattr(cls, "_btb_fused_mlp", False):
        return
    cls._btb_orig_mlp = cls.forward
    cls._btb_fused_mlp = True
    cls.forward = _fused_mlp_forward


def fast_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """The DeltaNet's causal depthwise conv as K shifted multiply-adds (the module's F.conv1d, without its cost
    on the CPU build); installed on the module when the MLX device is active."""
    from transformers.activations import ACT2FN

    x = hidden_states.to(weight.dtype)
    K = int(weight.shape[-1])
    L = int(x.shape[-1])
    xp = F.pad(x, (K - 1, 0))
    out = xp[..., :L] * weight[:, 0, None]
    for k in range(1, K):
        out = out + xp[..., k : k + L] * weight[:, k, None]
    if bias is not None:
        out = out + bias[:, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


# the gated DeltaNet's block: its chunked rule sums a block's rows in one order, and carries the state from block to
# block. A prompt cut where its blocks are cut - multiples of DELTA_BLOCK rows from a call's start - is the same bits
DELTA_BLOCK = 64


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = DELTA_BLOCK,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The gated delta rule in its chunked form (transformers' `torch_chunk_gated_delta_rule`, its arguments and
    results), a block at a time: every product batched over the heads alone, each block's operands tensors of their
    own. transformers' batches each product over every block of the call, and the math library splits a batch of a
    few blocks across its threads otherwise than one of many - a prompt resumed from a block's state, or swept in
    chunks of blocks, parted from the prompt taken whole (by 3e-8 in a state). A block here is a function of its own
    rows and the state it starts from: the same bits however the call is cut at its blocks, at any thread count.
    query, key [B, n, H, Dk]; value [B, n, H, Dv]; g, beta [B, n, H]; the state [B, H, Dk, Dv]"""
    dtype = query.dtype
    B, n, H, Dk = key.shape
    Dv = int(value.shape[-1])
    C = int(chunk_size)
    q, k, v, b, gl = (
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format) for x in (query, key, value, beta, g)
    )
    if use_qk_l2norm_in_kernel:
        q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    q = q * Dk**-0.5
    dev = q.device
    if initial_state is None:
        state = torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=dev)
    else:
        state = initial_state.to(device=dev, dtype=torch.float32, copy=True)
    out = torch.empty(B, H, n, Dv, dtype=torch.float32, device=dev)
    upper = torch.ones(C, C, dtype=torch.bool, device=dev).triu(1)
    for a in range(0, n, C):
        e = min(n, a + C)
        pad = C - (e - a)
        qc, kc, vc = (F.pad(x[:, :, a:e], (0, 0, 0, pad)).contiguous() for x in (q, k, v))
        bc, gc = (F.pad(x[:, :, a:e], (0, pad)).contiguous() for x in (b, gl))
        kb, vb = kc * bc[..., None], vc * bc[..., None]
        cum = gc.cumsum(dim=-1)
        pair = (cum[..., :, None] - cum[..., None, :]).masked_fill(upper, float("-inf")).exp()
        ut = (kb @ kc.transpose(-1, -2)) * pair
        intra = (qc @ kc.transpose(-1, -2)) * pair
        new_v = torch.linalg.solve_triangular(ut, vb, upper=False, unitriangular=True)
        k_cum = torch.linalg.solve_triangular(ut, kb * cum.exp()[..., None], upper=False, unitriangular=True)
        v_new = new_v - k_cum @ state
        o = (qc * cum.exp()[..., None]) @ state + intra @ v_new
        kd = kc * (cum[..., -1:] - cum).exp()[..., None]
        state = state * cum[..., -1].exp()[..., None, None] + kd.transpose(-1, -2) @ v_new
        out[:, :, a:e] = o[:, :, : e - a]
    return out.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format), state if output_final_state else None
