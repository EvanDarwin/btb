# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A linear held on the card as a GGUF type's blocks (btb/gguf.py `QuantWeight`), multiplied by the type's own
kernels (native/cuda/btb_gemv_quant.cuh) instead of a torch Linear over a bf16 copy that holds and reads 2-4x the
bytes."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from ..gguf import QuantWeight
from ..kinds import CARD_ROWS, PassTag, quant_tag
from .native import Native

# a card kernel's projection: one bf16 weight [R, C], or where a GGUF type's blocks are on the card its pieces in
# output-row order (a type's blocks, or a bf16 weight where the file kept one so)
CardMat = torch.Tensor | tuple[QuantWeight | torch.Tensor, ...]


def mat_rows(W: CardMat) -> int:
    """a projection's output rows"""
    return int(W.shape[0]) if isinstance(W, torch.Tensor) else sum(int(p.shape[0]) for p in W)


def card_loose(mats: Iterable[CardMat]) -> int:
    """the rows of the widest bf16 piece among stored projections' pieces: what a kernel's output of it takes before
    its rows land in the projection's (0 where there is none)"""
    return max(
        (int(p.shape[0]) for W in mats if not isinstance(W, torch.Tensor) for p in W if isinstance(p, torch.Tensor)),
        default=0,
    )


def held_bytes(m: torch.nn.Linear | CardStored) -> int:
    """the bytes a linear's weight holds: a stored one's blocks and tables, a torch one's weight"""
    if isinstance(m, CardStored):
        return sum(int(t.numel()) * t.element_size() for t in (m.raw, m.grid, m.ksigns) if t is not None)
    return int(m.weight.numel()) * m.weight.element_size()


class CardStored(torch.nn.Module):
    """`w`'s blocks and tables on the card as buffers (moved with the module), `key` its checkpoint name. A pass of
    up to 32 rows runs the type's gemv padded to a kernel width - batch-invariant, the card graph's own kernel, so
    a verify row is the one-row step's; a longer one (a prefill) decodes the weight to bf16 once and takes one
    GEMM. bf16 in, bf16 out, as a card linear's."""

    raw: torch.Tensor
    grid: torch.Tensor | None
    ksigns: torch.Tensor | None

    def __init__(self, w: QuantWeight, key: str, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("raw", w.raw)
        self.register_buffer("grid", w.grid)
        self.register_buffer("ksigns", w.ksigns)
        self.q, self.shape, self.key = w.q, w.shape, key
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)

    @property
    def stored(self) -> QuantWeight:
        """the blocks as the kernels take them, wherever the module is now"""
        return QuantWeight(self.raw, self.q, self.shape, self.grid, self.ksigns)

    def stored_tag(self) -> PassTag:
        return quant_tag(self.q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = Native.card_kernels()
        if k is None:
            raise RuntimeError(
                f"[card] {self.key}: a {self.q.value} weight needs the card's kernels ({Native.cuda_reason})"
            )
        (R, C), shp = self.shape, x.shape
        x2 = x.reshape(-1, C).to(torch.bfloat16).contiguous()
        T, w = int(x2.shape[0]), self.stored
        if T > CARD_ROWS[-1]:
            W = torch.empty(R, C, dtype=torch.bfloat16, device=x2.device)
            k.dequant_stored(w, W)
            y = F.linear(x2, W)
        else:
            M = next(m for m in CARD_ROWS if T <= m)
            xp = x2 if M == T else torch.cat([x2, x2.new_zeros(M - T, C)])
            yp = torch.empty(M, R, dtype=torch.bfloat16, device=x2.device)
            k.gemv_stored(w, xp, yp)
            y = yp[:T]
        out = y.reshape(*shp[:-1], R)
        return out if self.bias is None else out + self.bias.to(out.dtype)
