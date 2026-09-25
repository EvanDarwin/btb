# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`btb/mxfp4.py` against transformers' own dequantizer, and the native matvec against both.

This is where the nibble order, the e2m1 table and the e8m0 scale are pinned: the shipped gpt-oss weights
are read through them, and getting one of the three wrong is a model that produces plausible nonsense
rather than an error.
"""

import numpy as np
import torch

from btb import mxfp4
from btb.engine import StreamedTextModel
from tests.helpers import mxfp4_random, mxfp4_slot, native_library


def test_table_is_the_transformers_table() -> None:
    from transformers.integrations.mxfp4 import FP4_VALUES

    assert tuple(FP4_VALUES) == mxfp4.FP4_VALUES


def test_dequantize_matches_transformers_to_the_bit() -> None:
    """Over the whole e8m0 range, including the zero scale and the values that overflow to infinity."""
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    rng = np.random.default_rng(7)
    for rows, k, lo, hi in ((5, 64, 0, 256), (9, 96, 118, 130), (1, 32, 0, 1), (3, 2880, 100, 155)):
        b, s = mxfp4_random(rng, rows, k, lo, hi)
        ours = mxfp4.dequantize(b[None], s[None])[0]
        # the transformers dequantizer returns [K, rows] (its experts multiply x @ W); the checkpoint's
        # own orientation, and this module's, is [rows, K]
        ref = convert_moe_packed_tensors(torch.from_numpy(b[None]), torch.from_numpy(s[None]))[0].T.float().numpy()
        assert np.array_equal(ours.view(np.uint32), np.ascontiguousarray(ref).view(np.uint32)), (rows, k, lo, hi)


def test_mxweight_wraps_slot_bytes_without_copying() -> None:
    rng = np.random.default_rng(11)
    rows, k = 6, 64
    b, s = mxfp4_random(rng, rows, k, 118, 130)
    slot = torch.from_numpy(mxfp4_slot(b, s))
    nb = b.size
    w = mxfp4.MxWeight(slot[:nb], slot[nb:], rows, k)
    assert w.shape == (rows, k)
    assert np.array_equal(w.dequantize().numpy(), mxfp4.dequantize(b, s))


def test_native_matvec_matches_the_numpy_reference() -> None:
    native_library()
    assert StreamedTextModel.gemv_mx4 is not None, "the native library has no MXFP4 gemv: rebuild it"
    rng = np.random.default_rng(13)
    for rows, k, b in ((1, 32, 1), (5, 64, 1), (128, 64, 3), (17, 2880, 8), (64, 96, 9)):
        blocks, scales = mxfp4_random(rng, rows, k, 118, 130)
        w = mxfp4.MxWeight(torch.from_numpy(blocks.reshape(-1)), torch.from_numpy(scales.reshape(-1)), rows, k)
        x = torch.from_numpy(rng.standard_normal((b, k)).astype(np.float32))
        y = torch.empty(b, rows, dtype=torch.float32)
        StreamedTextModel.gemv_mx4(w, x, y)
        want = torch.from_numpy(mxfp4.dequantize(blocks, scales)).double() @ x.double().T
        scale = (torch.from_numpy(mxfp4.dequantize(blocks, scales)).double().abs() @ x.double().abs().T).clamp(
            min=1e-30
        )
        assert float(((y.double().T - want).abs() / scale).max()) < 1e-6, (rows, k, b)
        # a row's value does not depend on how many rows travel with it
        one = torch.empty(1, rows, dtype=torch.float32)
        StreamedTextModel.gemv_mx4(w, x[:1].contiguous(), one)
        assert torch.equal(one[0], y[0]), (rows, k, b)


def test_native_group_matches_the_single_calls() -> None:
    native_library()
    assert StreamedTextModel.gemv_mx4_group is not None, "the native library has no MXFP4 group gemv: rebuild it"
    rng = np.random.default_rng(17)
    shapes = [(12, 64), (7, 128), (20, 96)]
    ws, xs, want = [], [], []
    for rows, k in shapes:
        blocks, scales = mxfp4_random(rng, rows, k, 118, 130)
        ws.append(mxfp4.MxWeight(torch.from_numpy(blocks.reshape(-1)), torch.from_numpy(scales.reshape(-1)), rows, k))
        xs.append(torch.from_numpy(rng.standard_normal((1, k)).astype(np.float32)))
        y = torch.empty(1, rows, dtype=torch.float32)
        StreamedTextModel.gemv_mx4(ws[-1], xs[-1], y)
        want.append(y)
    got = [torch.empty(1, rows, dtype=torch.float32) for rows, _ in shapes]
    StreamedTextModel.gemv_mx4_group(ws, xs, got)
    for a, b in zip(want, got):
        assert torch.equal(a, b)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
