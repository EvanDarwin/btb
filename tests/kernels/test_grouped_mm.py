# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A grouped expert call's matmul on the card (`torch._grouped_mm`, as `_Experts._card_grouped` calls it: the
experts' rows in slot order, their weights a stack viewed transposed) against the per-expert loop it replaced,
each expert's rows through `F.linear` on their own - bit for bit, the claim the grouped path stands on: a slot's
product over its rows is the loop's, so grouping changes no bit of a prefill. At gpt-oss-120b's expert widths (the
tiling is the shape's), over uneven rows an expert and an expert with none. The widening that feeds it from the
depot's MXFP4 bytes is held to torch's in test_mx4_widen.py."""

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from tests.helpers import need_cuda

H, I = 2880, 2880  # gpt-oss-120b: hidden, expert intermediate (gate_up is 2 * I wide)
COUNTS = (1, 7, 0, 32, 130, 5)  # each expert's rows in the call: one, a few, none, many


def test_the_grouped_matmul_is_each_experts_linear_bit_for_bit() -> None:
    dev = need_cuda()
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("this torch has no _grouped_mm: the engine keeps the per-expert loop")
    g = torch.Generator(device=dev).manual_seed(0)
    n, total = len(COUNTS), sum(COUNTS)
    ends = torch.tensor(list(accumulate(COUNTS)), dtype=torch.int32, device=dev)
    for rows, cols in ((2 * I, H), (H, I)):  # gate_up, then down
        w = torch.randn(n, rows, cols, device=dev, generator=g).mul_(cols**-0.5).bfloat16()
        x = torch.randn(total, cols, device=dev, generator=g).bfloat16()
        got = torch._grouped_mm(x, w.transpose(1, 2), offs=ends)
        assert got.shape == (total, rows) and got.dtype == torch.bfloat16
        a = 0
        for e, c in enumerate(COUNTS):
            want = F.linear(x[a : a + c], w[e])
            same = got[a : a + c].view(torch.int16) == want.view(torch.int16)
            assert bool(same.all()), f"[{rows}, {cols}] expert {e} ({c} rows): {int((~same).sum())} values differ"
            a += c
