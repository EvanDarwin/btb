# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's one-pass MXFP4 widening (`btb_mx4_widen`, `_Cuda.mx4_widen`) against the torch widening it replaces in
a grouped expert call (`dequant_blocks`, itself transformers' dequantizer bit for bit: test_mxfp4_torch.py): every
byte under every scale, compared as bits so a signed zero, an infinity and a NaN are each held to their own
pattern; the experts taken at seats out of order and repeated, as a wave takes them from a depot's stacks."""

from __future__ import annotations

import torch

from btb.mxfp4_torch import dequant_blocks
from tests.helpers import need_card_kernels


def _stacks(seats: int, rows: int, groups: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """a depot's two stacks for one MXFP4 part: blocks [seats, rows, groups, 16] and scales [seats, rows, groups]"""
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randint(0, 256, (seats, rows, groups, 16), dtype=torch.uint8, generator=g)
    scales = torch.randint(0, 256, (seats, rows, groups), dtype=torch.uint8, generator=g)
    return blocks.cuda(), scales.cuda()


def _same_bits(got: torch.Tensor, want: torch.Tensor) -> None:
    assert got.shape == want.shape and got.dtype == want.dtype == torch.bfloat16
    diff = got.view(torch.int16) != want.view(torch.int16)
    if bool(diff.any()):
        at = diff.nonzero()[0].tolist()
        raise AssertionError(
            f"{int(diff.sum())} of {diff.numel()} values differ; first at {at}: "
            f"{float(got[tuple(at)])} against {float(want[tuple(at)])}"
        )


def test_every_byte_under_every_scale_widens_as_torch_widens_it() -> None:
    cu = need_card_kernels()
    # seat s holds every byte value once in each of its 16 blocks' lanes, under scale s: 256 x 256 pairs
    lanes = torch.arange(256, dtype=torch.uint8).view(16, 16)
    blocks = lanes.expand(256, 16, 16).contiguous().view(256, 1, 16, 16).cuda()
    scales = torch.arange(256, dtype=torch.uint8).view(256, 1, 1).expand(256, 1, 16).contiguous().cuda()
    seats = torch.arange(256, dtype=torch.int32, device="cuda")
    got = cu.mx4_widen(blocks, scales, seats).view(256, 1, 16, 32)
    _same_bits(got, dequant_blocks(blocks, scales))


def test_experts_at_seats_out_of_order_and_repeated() -> None:
    cu = need_card_kernels()
    # a stack's row of 32-value blocks past one launch block's 256 threads, and not a multiple of it
    blocks, scales = _stacks(seats=7, rows=45, groups=9, seed=1)
    for order in ([6, 0, 3], [2, 2, 5, 0, 2], [4]):
        seats = torch.tensor(order, dtype=torch.int32, device="cuda")
        got = cu.mx4_widen(blocks, scales, seats).view(len(order), 45, 9 * 32)
        _same_bits(got, dequant_blocks(blocks[seats.long()], scales[seats.long()]).view(len(order), 45, 9 * 32))


def test_a_given_buffer_is_written_and_a_malformed_stack_refused() -> None:
    cu = need_card_kernels()
    blocks, scales = _stacks(seats=3, rows=4, groups=2, seed=2)
    seats = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    out = torch.full((2, 4 * 2 * 32), float("nan"), dtype=torch.bfloat16, device="cuda")
    assert cu.mx4_widen(blocks, scales, seats, out=out) is out
    _same_bits(out.view(2, 4, 2, 32), dequant_blocks(blocks[1:], scales[1:]))
    for bad in (blocks[:, :, :, :8].contiguous(), blocks.transpose(1, 2), blocks.to(torch.int16)):
        try:
            cu.mx4_widen(bad, scales, seats)
        except ValueError:
            continue
        raise AssertionError(f"a stack of {tuple(bad.shape)} {bad.dtype} was taken")
