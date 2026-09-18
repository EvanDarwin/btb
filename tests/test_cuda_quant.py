# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's k-quant gemv kernels (native/cuda/btb_kernels.cu), certified on the CPU: each decoder below is a
numpy twin of exactly what `btb_gemv_q*k_rows` computes for a superblock, held bit-for-bit to gguf's own
dequantization, and the Q4_K matvec twin (the kernel's lane layout and fp32 accumulation) to x . dequant(W)^T,
identical for a row whether the pass carries 1 row or 16. No card is needed: the .cu is a transcription of
these twins, and a card only confirms them at speed."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from btb.quant import QUANTS

gguf = pytest.importorskip("gguf")
from gguf.constants import GGMLQuantizationType as GT


def _f16(b: bytes) -> np.float32:
    return np.frombuffer(b, np.float16)[0].astype(np.float32)


def _scale_min(s: bytes, j: int) -> tuple[float, float]:
    """llama.cpp's get_scale_min_k4: the 6-bit scale and min of sub-block j from the 12 packed bytes"""
    if j < 4:
        return float(s[j] & 63), float(s[j + 4] & 63)
    return float((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)), float((s[j + 4] >> 4) | ((s[j] >> 6) << 4))


def _decode_q4k(blk: bytes) -> np.ndarray:
    d, dm, s, qs = _f16(blk[0:2]), _f16(blk[2:4]), blk[4:16], blk[16:144]
    o = np.zeros(256, np.float32)
    for k in range(4):
        sclo, mnlo = _scale_min(s, 2 * k)
        schi, mnhi = _scale_min(s, 2 * k + 1)
        for lane in range(32):
            byte = qs[32 * k + lane]
            o[(2 * k) * 32 + lane] = d * sclo * (byte & 0xF) - dm * mnlo
            o[(2 * k + 1) * 32 + lane] = d * schi * (byte >> 4) - dm * mnhi
    return o


def _decode_q5k(blk: bytes) -> np.ndarray:
    d, dm, s, qh, qs = _f16(blk[0:2]), _f16(blk[2:4]), blk[4:16], blk[16:48], blk[48:176]
    o = np.zeros(256, np.float32)
    for lane in range(32):
        hbit = qh[lane]
        for k in range(4):
            byte = qs[32 * k + lane]
            qlo = (byte & 0xF) | (((hbit >> (2 * k)) & 1) << 4)
            qhi = (byte >> 4) | (((hbit >> (2 * k + 1)) & 1) << 4)
            sclo, mnlo = _scale_min(s, 2 * k)
            schi, mnhi = _scale_min(s, 2 * k + 1)
            o[(2 * k) * 32 + lane] = d * sclo * qlo - dm * mnlo
            o[(2 * k + 1) * 32 + lane] = d * schi * qhi - dm * mnhi
    return o


def _decode_q6k(blk: bytes) -> np.ndarray:
    ql, qh, sc, d = blk[0:128], blk[128:192], np.frombuffer(blk[192:208], np.int8), _f16(blk[208:210])
    o = np.zeros(256, np.float32)
    for h in range(2):
        base, qlo, qho, sco = h * 128, h * 64, h * 32, h * 8
        for lane in range(32):
            isc = sco + lane // 16
            l0, l1, hb = ql[qlo + lane], ql[qlo + lane + 32], qh[qho + lane]
            o[base + lane] = d * sc[isc + 0] * (((l0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32)
            o[base + lane + 32] = d * sc[isc + 2] * (((l1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32)
            o[base + lane + 64] = d * sc[isc + 4] * (((l0 >> 4) | (((hb >> 4) & 3) << 4)) - 32)
            o[base + lane + 96] = d * sc[isc + 6] * (((l1 >> 4) | (((hb >> 6) & 3) << 4)) - 32)
    return o


def _decode_q2k(blk: bytes) -> np.ndarray:
    scales, qs, d, dm = blk[0:16], blk[16:80], _f16(blk[80:82]), _f16(blk[82:84])
    o = np.zeros(256, np.float32)
    for h in range(2):
        for lane in range(32):
            byte, sub = qs[h * 32 + lane], lane >> 4
            for j in range(4):
                s = scales[h * 8 + 2 * j + sub]
                o[h * 128 + j * 32 + lane] = d * (s & 0xF) * ((byte >> (2 * j)) & 3) - dm * (s >> 4)
    return o


def _decode_q3k(blk: bytes) -> np.ndarray:
    hmask, qs, sc, d = blk[0:32], blk[32:96], blk[96:108], _f16(blk[108:110])
    a0, a1, a2 = (int.from_bytes(sc[i : i + 4], "little") for i in (0, 4, 8))
    k1, k2 = 0x03030303, 0x0F0F0F0F
    aux = [
        (a0 & k2) | (((a2 >> 0) & k1) << 4),
        (a1 & k2) | (((a2 >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
    ]
    scl = b"".join(v.to_bytes(4, "little") for v in aux)  # the 16 scales, 0..63
    o = np.zeros(256, np.float32)
    for h in range(2):
        for lane in range(32):
            hm, sub, byte = hmask[lane], lane >> 4, qs[h * 32 + lane]
            for j in range(4):
                q = (byte >> (2 * j)) & 3
                if not (hm & (1 << (h * 4 + j))):
                    q -= 4
                o[h * 128 + j * 32 + lane] = d * (scl[h * 8 + 2 * j + sub] - 32) * q
    return o


# a kernel's per-superblock decode as numpy: the block's bytes to its 256 weights
Decoder = Callable[[bytes], np.ndarray]
# the byte offsets of a block's f16 scale fields, made finite in a random block (random bytes as f16 may be inf/nan)
ScaleFields = tuple[int, ...]
# by type; the block sizes are the registry's (pinned to gguf's table by test_quant.py)
DECODERS: dict[str, tuple[Decoder, ScaleFields]] = {
    "Q2_K": (_decode_q2k, (80, 82)),
    "Q3_K": (_decode_q3k, (108,)),
    "Q4_K": (_decode_q4k, (0, 2)),
    "Q5_K": (_decode_q5k, (0, 2)),
    "Q6_K": (_decode_q6k, (208,)),
}
_Q4K = QUANTS["Q4_K"].block_bytes


def _blocks(name: str, n: int, seed: int) -> bytes:
    nb = QUANTS[name].block_bytes
    _decode, scale_offs = DECODERS[name]
    rng = np.random.default_rng(seed)
    buf = bytearray(rng.integers(0, 256, size=n * nb, dtype=np.uint8).tobytes())
    for b in range(n):
        for off in scale_offs:
            buf[b * nb + off : b * nb + off + 2] = np.float16(rng.uniform(0.001, 0.05)).tobytes()
    return bytes(buf)


def test_every_card_type_has_a_certified_decoder() -> None:
    assert set(DECODERS) == {q.name for q in QUANTS.values() if q.card is not None}


@pytest.mark.parametrize("name", sorted(DECODERS))
def test_the_card_decoder_matches_gguf_bit_for_bit(name: str) -> None:
    nb = QUANTS[name].block_bytes
    decode, _ = DECODERS[name]
    raw = np.frombuffer(_blocks(name, 7, hash(name) & 0xFFFF), np.uint8)
    ref = gguf.quants.dequantize(raw.copy(), GT[name]).reshape(7, 256).astype(np.float32)
    mine = np.stack([decode(bytes(raw[b * nb : (b + 1) * nb])) for b in range(7)])
    np.testing.assert_array_equal(mine, ref)


def _q4k_matvec_twin(W: bytes, x_bf: torch.Tensor, R: int, nsb: int) -> np.ndarray:
    """btb_gemv_q4k_rows's arithmetic: lane l owns bytes [4l, 4l+3] of every superblock (sub-block pair l>>3 at
    inner offset (l&7)*4), eight weights decoded once, fp32 fma over the pass's rows, the lane partials summed"""
    M = x_bf.shape[0]
    xf = x_bf.float().numpy()  # bf2f, what the kernel reads
    y = np.zeros((M, R), np.float32)
    for r in range(R):
        row = W[r * nsb * _Q4K : (r + 1) * nsb * _Q4K]
        for m in range(M):
            acc = np.float32(0)
            for lane in range(32):
                k, inner = lane >> 3, (lane & 7) << 2
                for c in range(nsb):
                    blk = row[c * _Q4K : (c + 1) * _Q4K]
                    d, dm, s = _f16(blk[0:2]), _f16(blk[2:4]), blk[4:16]
                    sclo, mnlo = _scale_min(s, 2 * k)
                    schi, mnhi = _scale_min(s, 2 * k + 1)
                    packed = blk[16 + lane * 4 : 16 + lane * 4 + 4]
                    colL, colH = c * 256 + (2 * k) * 32 + inner, c * 256 + (2 * k + 1) * 32 + inner
                    for i in range(4):
                        wl = np.float32(d * sclo * (packed[i] & 0xF) - dm * mnlo)
                        wh = np.float32(d * schi * (packed[i] >> 4) - dm * mnhi)
                        acc = np.float32(acc + wl * np.float32(xf[m, colL + i]))
                        acc = np.float32(acc + wh * np.float32(xf[m, colH + i]))
            y[m, r] = acc
    return y


def test_the_q4k_matvec_is_the_dequantized_product_and_batch_invariant() -> None:
    R, nsb = 12, 2
    C = nsb * 256
    W = b"".join(_blocks("Q4_K", nsb, 100 + r) for r in range(R))
    Wf = gguf.quants.dequantize(np.frombuffer(W, np.uint8).copy(), GT.Q4_K).reshape(R, C).astype(np.float32)
    g = torch.Generator().manual_seed(1)
    x1 = torch.randn(1, C, generator=g).bfloat16()
    x16 = torch.cat([x1, torch.randn(15, C, generator=g).bfloat16()], 0)
    ref = x1.float().numpy() @ Wf.T
    one, sixteen = _q4k_matvec_twin(W, x1, R, nsb), _q4k_matvec_twin(W, x16, R, nsb)
    assert np.abs(one - ref).max() <= 2e-3 * np.abs(ref).max()
    # row 0 of a 16-row pass is row 0 of the 1-row step, bit for bit: what a verify pass needs
    np.testing.assert_array_equal(sixteen[0], one[0])
