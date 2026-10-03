# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's elementwise activations on the host, one token row at a time in a speculative verify pass, so a node's
row is computed as its one-token step computes it. Torch's CPU kernels run a sigmoid or a silu through a vector
body and a scalar tail (and past 32k elements, across threads at arbitrary cuts): which of the two an element takes
depends on how many elements travel together, and the two differ in the last bit. A row of a verify pass among
T rows lands in the vector body where its step's few elements took the tail - Qwen4's hyper-connections inject
through a sigmoid of `hc_count` (4) elements a row. Each row alone is the step's own call, bit for bit.

The linears around them already are a row's own (the host's gemv, `_HostLinear`), so only the activations go
row by row: the hyper-connections (`Qwen4ExpTextGatedResidual`), the mixture block's shared expert and its gate,
and the closing mixer, whose float32 linears are torch's, run a row at a time whole. The attention (`attend.py`)
and the n-gram embedding (`verify.py`) use `each` for their own. Outside a verify pass - the one-token step, a
prefill - every module runs as the reference's."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from . import bind_forward


def apart(sm: Any, x: torch.Tensor, T: int) -> bool:
    """whether `T` token rows of `x` compute one at a time: several rows of a verify pass on the host"""
    return T > 1 and x.device.type == "cpu" and sm is not None and bool(getattr(sm, "aq", False))


def each(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, T: int, rows: bool = True) -> torch.Tensor:
    """`fn(x)` with `x`'s `T` token rows (its first non-unit dimension, the rows' elements contiguous after it) taken
    one at a time when `rows`, each call the elements the one-token step's call is"""
    if not rows or T <= 1:
        return fn(x)
    flat = x.reshape(T, -1)
    return torch.cat([fn(flat[t : t + 1]) for t in range(T)]).view(x.shape)


def install(layer: Any, sm: Any) -> None:
    """a host layer's hyper-connections and mixture block computing their activations row by row in a verify pass,
    and the engine's closing mixer a row at a time; `sm` held weakly (the engine owns the modules). Every Qwen4
    layer, the drafting head's too, has both hyper-connections and a mixture block with its shared expert"""
    ref = weakref.ref(sm)
    for m in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
        m._btb_sm = ref
        bind_forward(m, _residual_forward)
    layer.mlp._btb_sm = ref
    bind_forward(layer.mlp, _moe_forward)
    mixer = getattr(sm, "mixer", None)
    if mixer is not None and getattr(mixer, "_btb_sm", None) is None:
        mixer._btb_sm = ref
        bind_forward(mixer, _mixer_forward)


def _residual_forward(self: Any, hyper_input: torch.Tensor) -> Any:
    T = int(hyper_input.numel() // hyper_input.shape[-1])
    if not apart(self._btb_sm(), hyper_input, T):
        return type(self).forward(self, hyper_input)
    # the reference's arithmetic (`Qwen4ExpTextGatedResidual.forward`), its activations row by row
    hc, H = int(self.hc_count), int(self.hidden_size)
    if hyper_input.shape[-1] != hc * H:
        raise ValueError(f"Expected {hc * H} hyper-connection features, got {hyper_input.shape[-1]}.")
    normed = self.hc_norm(hyper_input)
    mix = each(F.silu, self.input_mix_weight_down(normed) / hc, T)
    mix = each(torch.sigmoid, self.input_mix_weight_up(mix), T).unflatten(-1, (hc, H))
    mixed = (mix * normed.unflatten(-1, (hc, H))).mean(dim=-2)
    # a layer's hyper-connections always combine (`use_combine`); the closing mixer, which does not, is `_mixer_forward`
    inject = 2 * each(torch.sigmoid, self.block_inject_weight(normed) / hc, T)
    return mixed, hyper_input, inject


def _moe_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    T = B * S
    if not apart(self._btb_sm(), hidden_states, T):
        return type(self).forward(self, hidden_states)
    # the reference's arithmetic (`Qwen4ExpTextSparseMoeBlock.forward`), the shared expert's activation and its
    # gate row by row; the routed experts' own run a row's elements as its step's do (host.py `_Experts`)
    x = hidden_states.view(-1, D)
    se = self.shared_expert
    shared = se.down_proj(each(se.act_fn, se.gate_proj(x), T) * se.up_proj(x))
    _, weights, experts = self.gate(x)
    out = self.experts(x, experts, weights)
    shared = each(torch.sigmoid, self.shared_expert_gate(x), T) * shared
    return (out + shared).reshape(B, S, D)


def _mixer_forward(self: Any, hyper_input: torch.Tensor) -> Any:
    # the closing mixer's linears are torch's float32 ones, whose sums change with the rows too: each row whole
    T = int(hyper_input.numel() // hyper_input.shape[-1])
    if not apart(self._btb_sm(), hyper_input, T) or hyper_input.dim() != 3 or hyper_input.shape[0] != 1:
        return type(self).forward(self, hyper_input)
    return torch.cat([type(self).forward(self, hyper_input[:, t : t + 1]) for t in range(T)], dim=1)
