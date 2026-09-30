# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Row-invariant torch on the card: a small pass's rows through the matmuls and norms at one fixed shape.

cuBLAS picks a GEMM's kernel by its shape, and torch's reductions pick their launch by how many rows they reduce, so
a token's value on the card's torch path moved with how many tokens travelled with it: a one-row greedy step and a
speculative verify pass of several rows over the same token parted by a bf16 step now and then, and speculation's
answer parted from greedy's. Here every pass of at most `ROWS_MAX` tokens runs its matmuls and norms in chunks of
exactly `ROW_W` tokens - the last one padded with zero rows - so a token meets the same kernel whatever the pass: a
GEMM's row, and a norm's, depends only on that row and the weights once the shape is fixed. A pass of more tokens (a
prefill) keeps torch's own shapes; its rows never meet a step's.

Installed on the modules btb owns on the card: every `nn.Linear` of a card layer and the head (`fix_linears`, a
`RowLinear` of the same parameters), and the family's norm class (`fix_rows_cls`). The host and MLX never take
it: `fixed_rows` passes a tensor off the card straight through."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

ROW_W = 16  # the rows a small pass's chunk carries
ROWS_MAX = 64  # the most tokens a pass may carry and take the fixed chunks (a verify pass's rows are at most 33)


def _tokens(x: torch.Tensor) -> int:
    """the leading dims that count a pass's tokens: [N, ...] for a 2-d tensor, [B, T, ...] otherwise"""
    return int(x.shape[0]) if x.dim() <= 2 else int(x.shape[0]) * int(x.shape[1])


def fixed_rows(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """`fn` over `x`'s tokens in chunks of exactly ROW_W, each contiguous, the last padded with zero tokens, and the
    tokens' rows put back in `x`'s leading shape; `fn(x)` itself off the card or past ROWS_MAX tokens. `fn` must act
    on each token alone (a matmul's row, a norm over the last dims)"""
    if not x.is_cuda or x.dim() < 2:
        return fn(x)
    n = _tokens(x)
    if n == 0 or n > ROWS_MAX:
        return fn(x)
    lead = tuple(x.shape[:1]) if x.dim() <= 2 else tuple(x.shape[:2])
    rest = tuple(x.shape[len(lead) :])
    x2 = x.reshape(n, *rest)
    outs = []
    for a in range(0, n, ROW_W):
        m = min(ROW_W, n - a)
        c = x2[a : a + m]
        if m < ROW_W:
            c = torch.cat([c, c.new_zeros(ROW_W - m, *rest)])
        y = fn(c.contiguous())
        outs.append(y[:m] if m < ROW_W else y)
    y = outs[0] if len(outs) == 1 else torch.cat(outs)
    return y.reshape(*lead, *y.shape[1:])


class KeyRows:
    """the cache rows one query row attends, in the order it attends them, handed to a layer in place of its mask:
    a speculative pass's node (the prefix - its window's worth under a window - its ancestors and itself) as the
    greedy step at its position reads them once its path is committed (`forward.py` `node_mask`), so its attention
    makes the step's own call (`families/attention.py` `attend_one`, gpt-oss's sinks). `idx` a long tensor on the
    rows' device"""

    __slots__ = ("idx",)

    def __init__(self, idx: torch.Tensor) -> None:
        self.idx = idx


class RowLinear(torch.nn.Linear):
    """an `nn.Linear` whose small passes run at the fixed shape (`fixed_rows`): its parameters are the module's own,
    only its forward is"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fixed_rows(lambda c: F.linear(c, self.weight, self.bias), x)


def fix_linears(module: Any) -> None:
    """every plain `nn.Linear` in `module` (itself too) made a `RowLinear` in place"""
    for m in module.modules():
        if type(m) is torch.nn.Linear:
            m.__class__ = RowLinear


def fix_rows_cls(cls: Any) -> None:
    """a norm class's forward (fused or the reference's, whichever it has) run at the fixed shape: its arithmetic
    as it is, over chunks of ROW_W tokens. Once per class"""
    if cls is None or getattr(cls, "_btb_fixed_rows", False):
        return
    inner = cls.forward

    def forward(self: Any, x: torch.Tensor, *a: Any, **kw: Any) -> torch.Tensor:
        if a or kw:  # a gated norm's second operand: not a lone row's function of `x`
            return inner(self, x, *a, **kw)
        return fixed_rows(lambda c: inner(self, c), x)

    cls._btb_fixed_rows = True
    cls.forward = forward
