from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .gguf import GGUFModel

# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""MXFP4 as gpt-oss stores it, and the one definition of the expert store's slot.

A block is 32 weights along K: 16 bytes of packed fp4 (e2m1) plus one uint8 e8m0 exponent, and the weight
is `fp4 * 2**(scale - 127)`. Two weights share a byte, the earlier one in the LOW nibble (weight `2j` is
`byte & 0x0F`, weight `2j+1` is `byte >> 4`), and the sixteen fp4 codes are :data:`FP4_VALUES`. Both of
those are read off transformers' own dequantizer (`transformers.integrations.mxfp4`, `FP4_VALUES` and
`_convert_moe_packed_tensors`) and `tests/test_mxfp4.py` checks this module against it.

The checkpoint keeps `<proj>_blocks` as `[experts, rows, K/32, 16]` uint8 and `<proj>_scales` as
`[experts, rows, K/32]` uint8, so a matrix dequantizes to `[rows, K]` row-major: `y = W @ x` reads it the
way the bf16 matvec kernels read a bf16 matrix, with no transpose. (transformers' own dequantizer ends
with a `.transpose(1, 2)` because `GptOssExperts` multiplies `x @ W_t`; the same numbers, the other
orientation.)

Slot layout, shared with the MLX tier: for expert `e` the slot holds, in order,

    gate_up blocks[e] | gate_up scales[e] | down blocks[e] | down scales[e]

each exactly the checkpoint's bytes for that expert, so a slot is filled by four unbuffered reads at the
expert's offset in each tensor and nothing is expanded on the way in. The biases are small and stay with
the layer.

A GGUF holds the same weights in ggml's layout (`GGML_BLOCK_BYTES`): 17-byte blocks, the e8m0 scale first,
then 16 bytes whose LOW nibbles are weights 0..15 and HIGH nibbles 16..31, and gate and up as two tensors.
The kernels read that layout as stored (`MxWeight.ggml`), so a GGUF's slot is three reads:

    gate[e] | up[e] | down[e]
"""

FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

#: weights per block, and the bytes that hold them.
BLOCK = 32
BLOCK_BYTES = 16
#: ggml's block: the scale byte, then the 16 nibble bytes.
GGML_BLOCK_BYTES = 17

#: the exponent bias of the e8m0 scale: `2**(scale - BIAS)`.
BIAS = 127


def stored_mxfp4(mxfp4_family: bool, gguf: GGUFModel | None) -> bool:
    """whether a model's experts are MXFP4 blocks the matvec multiplies as stored: an MXFP4 family's checkpoint
    (gpt-oss's), or a GGUF whose expert tensors llama.cpp stored as MXFP4 (its MXFP4_MOE file type, any MoE)"""
    return mxfp4_family if gguf is None else gguf.mxfp4_experts()


def blocks_per_row(k: int) -> int:
    if k % BLOCK:
        raise ValueError(f"[mxfp4] K = {k} is not a multiple of {BLOCK}")
    return k // BLOCK


def matrix_bytes(rows: int, k: int) -> tuple[int, int]:
    """`(blocks, scales)` byte counts of one `[rows, k]` matrix in the slot."""
    g = blocks_per_row(k)
    return rows * g * BLOCK_BYTES, rows * g


def ggml_bytes(rows: int, k: int) -> int:
    """the bytes one `[rows, k]` matrix takes in ggml's layout"""
    return rows * blocks_per_row(k) * GGML_BLOCK_BYTES


def ggml_to_hf(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ggml's blocks (`[..., 17 * K/32]` bytes) as the checkpoint's: blocks `[..., K/32, 16]` of consecutive
    weight pairs, scales `[..., K/32]`"""
    raw = np.asarray(raw, dtype=np.uint8).reshape(*raw.shape[:-1], -1, GGML_BLOCK_BYTES)
    scales = raw[..., 0].copy()
    lo, hi = raw[..., 1:] & np.uint8(0x0F), raw[..., 1:] >> np.uint8(4)
    blocks = np.empty(raw[..., 1:].shape, dtype=np.uint8)
    blocks[..., :8] = lo[..., 0::2] | (lo[..., 1::2] << np.uint8(4))
    blocks[..., 8:] = hi[..., 0::2] | (hi[..., 1::2] << np.uint8(4))
    return blocks, scales


def hf_to_ggml(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """the inverse of `ggml_to_hf`: the checkpoint's blocks and scales as ggml's, `[..., 17 * K/32]`"""
    blocks = np.asarray(blocks, dtype=np.uint8)
    el = np.empty((*blocks.shape[:-1], BLOCK), dtype=np.uint8)
    el[..., 0::2], el[..., 1::2] = blocks & np.uint8(0x0F), blocks >> np.uint8(4)
    qs = el[..., :BLOCK_BYTES] | (el[..., BLOCK_BYTES:] << np.uint8(4))
    out = np.concatenate([np.asarray(scales, dtype=np.uint8)[..., None], qs], axis=-1)
    return out.reshape(*blocks.shape[:-2], -1)


def _np_table() -> np.ndarray:
    return np.array(FP4_VALUES, dtype=np.float32)


def dequantize(blocks: Any, scales: Any, dtype: Any = None) -> np.ndarray:
    """`[..., rows, K/32, 16]` uint8 blocks and `[..., rows, K/32]` uint8 scales to `[..., rows, K]` floats; pure
    numpy, the reference every other path is checked against (one rounding per weight, as `torch.ldexp`)."""
    blocks = np.asarray(blocks, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"[mxfp4] blocks {blocks.shape} do not match scales {scales.shape}")
    if blocks.shape[-1] != BLOCK_BYTES:
        raise ValueError(f"[mxfp4] a block is {BLOCK_BYTES} bytes, got {blocks.shape[-1]}")
    lut = _np_table()
    out = np.empty((*blocks.shape[:-1], BLOCK), dtype=np.float32)
    out[..., 0::2] = lut[blocks & 0x0F]
    out[..., 1::2] = lut[blocks >> 4]
    out *= scale_factors(scales)[..., None]
    out = out.reshape((*blocks.shape[:-2], blocks.shape[-2] * BLOCK))
    return out if dtype is None else out.astype(dtype)


def scale_factors(scales: Any) -> np.ndarray:
    """`2**(scale - 127)` as float32, exactly, for every uint8 scale."""
    s = np.asarray(scales, dtype=np.uint8).astype(np.uint32)
    f = (np.maximum(s, 1) << 23).view(np.float32)
    return np.where(s == 0, f * np.float32(0.5), f).astype(np.float32)


class MxWeight:
    """One MXFP4 matrix as stored, views never copied, `shape` = (rows, K). The checkpoint's layout: `blocks`
    (rows * K/32 * 16 bytes) and `scales` (rows * K/32). ggml's (`ggml` True, from `MxWeight.ggml`): `blocks`
    the 17-byte blocks (rows * K/32 * 17), `scales` None."""

    __slots__ = ("blocks", "ggml", "scales", "shape")

    def __init__(self, blocks: torch.Tensor, scales: torch.Tensor | None, rows: int, k: int, ggml: bool = False):
        if ggml:
            if scales is not None or blocks.numel() != ggml_bytes(rows, k):
                raise ValueError(f"[mxfp4] ggml [{rows}, {k}] needs {ggml_bytes(rows, k)} bytes, got {blocks.numel()}")
        else:
            nb, ns = matrix_bytes(rows, k)
            if scales is None or blocks.numel() != nb or scales.numel() != ns:
                raise ValueError(f"[mxfp4] [{rows}, {k}] needs {nb} + {ns} bytes, got {blocks.numel()} + {scales}")
        self.blocks = blocks
        self.scales = scales
        self.ggml = bool(ggml)
        self.shape = (int(rows), int(k))

    @classmethod
    def from_ggml(cls, raw: torch.Tensor, rows: int, k: int) -> MxWeight:
        """a matrix in ggml's layout: `raw` its 17-byte blocks"""
        return cls(raw, None, rows, k, ggml=True)

    def __repr__(self) -> str:
        return f"MxWeight[{self.shape[0]}, {self.shape[1]}{', ggml' if self.ggml else ''}]"

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        rows, k = self.shape
        g = blocks_per_row(k)
        if self.ggml:
            blocks, scales = ggml_to_hf(self.blocks.reshape(rows, -1).numpy())
        else:
            assert self.scales is not None
            blocks, scales = self.blocks.reshape(rows, g, BLOCK_BYTES).numpy(), self.scales.reshape(rows, g).numpy()
        t = torch.from_numpy(dequantize(blocks, scales))
        return t if dtype is None else t.to(dtype)


class MxGateUp:
    """gpt-oss's gate_up as a GGUF keeps it: `gate` and `up` two `[I, H]` matrices whose outputs interleave
    into the checkpoint's `[2I]` (gate at the even positions); `shape` = (2I, H) as one matrix would be."""

    __slots__ = ("gate", "shape", "up")

    def __init__(self, gate: MxWeight, up: MxWeight) -> None:
        if gate.shape != up.shape:
            raise ValueError(f"[mxfp4] gate {gate.shape} and up {up.shape} differ")
        self.gate, self.up = gate, up
        self.shape = (2 * gate.shape[0], gate.shape[1])

    def __repr__(self) -> str:
        return f"MxGateUp[{self.shape[0]}, {self.shape[1]}]"

    @staticmethod
    def interleave(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """`[..., I]` gate and up outputs as the checkpoint's `[..., 2I]`"""
        return torch.stack([gate, up], dim=-1).reshape(*gate.shape[:-1], 2 * gate.shape[-1])

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """the `[2I, H]` matrix the checkpoint would hold: gate and up rows interleaved"""
        g, u = self.gate.dequantize(dtype), self.up.dequantize(dtype)
        return torch.stack([g, u], dim=1).reshape(self.shape)
