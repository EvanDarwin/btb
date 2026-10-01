# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The host's MXFP4 matvec on the card (`btb_gemv_lane16_mx4_f32_m{M}`, `_Cuda.gemv_lane16_mx4`) against the host's
own (`Native.gemv_mx4`), bit for bit: an MXFP4 expert seated on the card must compute what it computes in RAM, or a
token's value would depend on which tier its experts sat in. Held at gpt-oss's shapes and at ragged small ones, under
scales from the subnormal end (2^-127 is built as 2^-126 * 0.5) to large, at every launch width, and row by row
whatever travels with the row. Compared as bits, so a signed zero is held to its sign."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from btb import mxfp4
from btb.engine.native import Native
from tests.helpers import mxfp4_random, native_library, need_card_kernels


def _same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    diff = got.view(torch.int32) != want.view(torch.int32)
    if bool(diff.any()):
        at = diff.nonzero()[0].tolist()
        raise AssertionError(
            f"{what}: {int(diff.sum())} of {diff.numel()} values differ; first at {at}: "
            f"{float(got[tuple(at)])} against the host's {float(want[tuple(at)])}"
        )


@pytest.mark.parametrize(
    "rows,k,lo,hi",
    [
        (5760, 2880, 110, 135),  # gpt-oss's gate_up
        (2880, 2880, 110, 135),  # its down
        (7, 32, 0, 160),  # one block a row, scales down to the subnormal end
        (33, 96, 0, 8),  # every weight subnormal or zero
        (129, 1024, 100, 160),
        (1, 64, 120, 130),
    ],
)
def test_the_cards_mx4_gemv_is_the_hosts(rows: int, k: int, lo: int, hi: int) -> None:
    cu = need_card_kernels()
    native_library()
    assert Native.gemv_mx4 is not None, "the native library has no MXFP4 gemv: rebuild it"
    rng = np.random.default_rng(rows * 7919 + k + lo)
    blocks, scales = mxfp4_random(rng, rows, k, lo, hi)
    b, s = torch.from_numpy(blocks.reshape(-1)), torch.from_numpy(scales.reshape(-1))
    x = torch.from_numpy((rng.standard_normal((32, k)) * rng.uniform(0.01, 4, (32, 1))).astype(np.float32))
    host = torch.empty(32, rows, dtype=torch.float32)
    Native.gemv_mx4(mxfp4.MxWeight(b, s, rows, k), x, host)
    w = mxfp4.MxWeight(b.cuda(), s.cuda(), rows, k)
    xc = x.cuda()
    for M in cu.GEMV_ROWS:
        y = torch.full((M, rows), 7.0, device="cuda")
        cu.gemv_lane16_mx4(w, xc[:M].contiguous(), y)
        _same_bits(y.cpu(), host[:M], f"[{rows}, {k}] M {M}")
    # row m of the 32-row launch is the one-row launch's
    for m in (0, 13, 31):
        one = torch.empty(1, rows, device="cuda")
        cu.gemv_lane16_mx4(w, xc[m : m + 1].contiguous(), one)
        _same_bits(one[0].cpu(), host[m], f"[{rows}, {k}] row {m} alone")


def test_a_malformed_call_is_refused() -> None:
    cu = need_card_kernels()
    blocks, scales = mxfp4_random(3, 8, 64)
    w = mxfp4.MxWeight(torch.from_numpy(blocks.reshape(-1)).cuda(), torch.from_numpy(scales.reshape(-1)).cuda(), 8, 64)
    ok = torch.zeros(1, 64, device="cuda"), torch.zeros(1, 8, device="cuda")
    cu.gemv_lane16_mx4(w, *ok)
    with pytest.raises(ValueError, match="float32"):
        cu.gemv_lane16_mx4(w, ok[0].bfloat16(), ok[1])
    with pytest.raises(ValueError, match="shapes"):
        cu.gemv_lane16_mx4(w, torch.zeros(3, 64, device="cuda"), torch.zeros(3, 8, device="cuda"))  # M 3
    with pytest.raises(ValueError, match="shapes"):
        cu.gemv_lane16_mx4(w, ok[0], torch.zeros(1, 9, device="cuda"))
    assert w.scales is not None  # the checkpoint's layout: its scales apart
    with pytest.raises(ValueError, match="uint8"):
        cu.gemv_lane16_mx4(mxfp4.MxWeight(w.blocks.cpu(), w.scales.cpu(), 8, 64), *ok)  # the bytes in RAM
    with pytest.raises(ValueError, match="layout"):
        raw = torch.zeros(mxfp4.ggml_bytes(8, 64), dtype=torch.uint8, device="cuda")
        cu.gemv_lane16_mx4(mxfp4.MxWeight.from_ggml(raw, 8, 64), *ok)
