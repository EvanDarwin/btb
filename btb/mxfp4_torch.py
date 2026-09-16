# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""MXFP4 experts on a torch device (CPU or CUDA) for openai/gpt-oss, over the store's slot layout: `blocks`
uint8 [rows, G, 16] then `scales` uint8 [rows, G], G = K // 32, never copied. `dequant_slot` is bit-identical to
transformers' `convert_moe_packed_tensors`; `matvec_slot` / `expert_step` hand the product to `torch.matmul`, so
they match the Rust matvec to a tolerance, not the bit, with the experts summed in ascending expert order."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .mxfp4 import FP4_VALUES, MxWeight  # one table (tests/test_mxfp4_torch.py holds it to transformers')

# gpt-oss swiglu constants (GptOssExperts.alpha / .limit)
ALPHA = 1.702
LIMIT = 7.0

_BYTE_LUT: dict[Any, torch.Tensor] = {}


def _byte_lut(device: Any, dtype: torch.dtype) -> torch.Tensor:
    """[256, 2] table: byte -> (low nibble value, high nibble value), the reference's lane order in one lookup."""
    key = (device.type, device.index, dtype)
    t = _BYTE_LUT.get(key)
    if t is None:
        lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)
        b = torch.arange(256, device=device)
        t = torch.stack((lut[b & 0x0F], lut[b >> 4]), dim=1)
        _BYTE_LUT[key] = t
    return t


def slot_bytes(rows: int, K: int) -> int:
    """Bytes one expert matrix occupies in a store slot."""
    if K % 32:
        raise ValueError(f"K must be a multiple of 32, got {K}")
    return rows * K // 2 + rows * (K // 32)


def split_slot(
    slot_u8: torch.Tensor | MxWeight, rows: int | None = None, K: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(blocks [rows, G, 16], scales [rows, G]) uint8 views of a slot, no copy; a uint8 tensor of
    `slot_bytes(rows, K)` or an `MxWeight` (then `rows`/`K` may be omitted)."""
    if not isinstance(slot_u8, torch.Tensor):
        b, s = slot_u8.blocks, slot_u8.scales
        r, k = slot_u8.shape
        if rows is not None and (rows, K) != (r, k):
            raise ValueError(f"MxWeight is [{r}, {k}], not [{rows}, {K}]")
        G = k // 32
        if s is None:
            raise ValueError(f"{slot_u8} is in ggml's layout: no scales to split")
        return b.reshape(r, G, 16), s.reshape(r, G)
    if rows is None or K is None:
        raise ValueError("rows and K are required for a raw uint8 slot")
    if slot_u8.dtype != torch.uint8:
        raise TypeError(f"slot must be uint8, got {slot_u8.dtype}")
    flat = slot_u8.reshape(-1) if slot_u8.dim() != 1 else slot_u8
    need = slot_bytes(rows, K)
    if flat.numel() != need:
        raise ValueError(f"slot has {flat.numel()} bytes, expected {need} for rows={rows} K={K}")
    G = K // 32
    n_blocks = rows * G * 16
    blocks = flat[:n_blocks].view(rows, G, 16)
    scales = flat[n_blocks:].view(rows, G)
    return blocks, scales


def dequant_blocks(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize `blocks [.., G, 16]` + `scales [.., G]` to `[.., G*32]`, bit-identical to transformers'
    `convert_moe_packed_tensors`."""
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"blocks {tuple(blocks.shape)} do not match scales {tuple(scales.shape)}")
    lut = _byte_lut(blocks.device, dtype)
    vals = lut[blocks.to(torch.int)]  # [.., G, 16, 2]
    vals = vals.reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)
    exp = scales.to(torch.int32) - 127
    return torch.ldexp(vals, exp.unsqueeze(-1))


def dequant_slot(
    slot_u8: torch.Tensor | MxWeight,
    rows: int | None = None,
    K: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize one expert matrix from its slot bytes to `[rows, K]` on the slot's device, no copy of the slot;
    `rows_per_chunk` bounds the transient memory, `out` [rows, K] receives the result."""
    blocks, scales = split_slot(slot_u8, rows, K)
    rows, K = blocks.shape[0], blocks.shape[1] * 32
    if out is None:
        out = torch.empty(rows, K, dtype=dtype, device=blocks.device)
    elif tuple(out.shape) != (rows, K) or out.dtype != dtype or out.device != blocks.device:
        raise ValueError(f"out must be [{rows}, {K}] {dtype} on {blocks.device}")
    step = rows if rows_per_chunk is None else max(1, int(rows_per_chunk))
    G = K // 32
    for r0 in range(0, rows, step):
        r1 = min(r0 + step, rows)
        v = dequant_blocks(blocks[r0:r1], scales[r0:r1], dtype=dtype)
        out[r0:r1] = v.reshape(r1 - r0, G * 32)
    return out


def matvec_slot(
    slot_u8: torch.Tensor | MxWeight,
    rows: int | torch.Tensor | None = None,
    K: int | None = None,
    x: Any = None,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x [B, K] -> [B, rows] float32`, the slot dequantized to `dtype` (bf16: the reference's values) and
    multiplied in float32 on the slot's device; `rows_per_chunk` bounds the live slice. A tolerance-level match
    to the Rust matvec. `slot_u8` may be an `MxWeight` (`matvec_slot(w, x)`)."""
    if x is None and isinstance(rows, torch.Tensor):
        rows, K, x = None, None, rows  # matvec_slot(mx_weight, x)
    if x is None:
        raise ValueError("matvec_slot needs x")
    assert not isinstance(rows, torch.Tensor)  # the MxWeight form moved the Tensor to x above
    blocks, scales = split_slot(slot_u8, rows, K)
    rows, K = blocks.shape[0], blocks.shape[1] * 32
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[-1] != K:
        raise ValueError(f"x has {x.shape[-1]} columns, expected K={K}")
    xf = x.to(torch.float32)
    if xf.device != blocks.device:
        xf = xf.to(blocks.device)
    B = xf.shape[0]
    if out is None:
        out = torch.empty(B, rows, dtype=torch.float32, device=blocks.device)
    step = rows if rows_per_chunk is None else max(1, int(rows_per_chunk))
    G = K // 32
    for r0 in range(0, rows, step):
        r1 = min(r0 + step, rows)
        w = dequant_blocks(blocks[r0:r1], scales[r0:r1], dtype=dtype).reshape(r1 - r0, G * 32)
        out[:, r0:r1] = torch.matmul(xf, w.to(torch.float32).t())
    return out


def swiglu(gate_up: torch.Tensor, alpha: float = ALPHA, limit: float = LIMIT) -> torch.Tensor:
    """gpt-oss's gated activation over an interleaved [.., 2*I] row as `GptOssExperts._apply_gate`: gate the even
    lane clamped above at `limit`, up the odd lane clamped to +-limit, (up + 1) * gate * sigmoid(alpha * gate)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def expert_step(
    x: torch.Tensor,
    slots_gate_up: Sequence[torch.Tensor | MxWeight],
    slots_down: Sequence[torch.Tensor | MxWeight],
    biases_gate_up: Any,
    biases_down: Any,
    weights: torch.Tensor,
    expert_of: Any = None,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int | None = None,
    alpha: float = ALPHA,
    limit: float = LIMIT,
) -> torch.Tensor:
    """The gpt-oss MoE step for T tokens: `x` [T, K], `slots_gate_up`/`slots_down` E slot views in expert order,
    the biases ([E, .] or lists), `weights` [T, k], `expert_of` [T, k] (default: every slot in order, E == k).
    Returns [T, K] float32 on x's device, summed in ascending expert order on the slots' device."""
    E = len(slots_gate_up)
    if len(slots_down) != E:
        raise ValueError(f"{E} gate_up slots but {len(slots_down)} down slots")
    if x.dim() != 2:
        raise ValueError(f"x must be [T, K], got {tuple(x.shape)}")
    T, K = x.shape
    weights = weights.to(torch.float32)
    if weights.dim() != 2 or weights.shape[0] != T:
        raise ValueError(f"weights must be [T, k], got {tuple(weights.shape)}")
    k = weights.shape[1]
    if expert_of is None:
        if k != E:
            raise ValueError(f"expert_of is required unless E == k ({E} != {k})")
        expert_of = torch.arange(k, device="cpu").unsqueeze(0).expand(T, k)
    expert_of = expert_of.to("cpu", torch.long)
    if tuple(expert_of.shape) != (T, k):
        raise ValueError(f"expert_of must be [T, k], got {tuple(expert_of.shape)}")

    out_dev = x.device
    s0 = slots_gate_up[0]
    dev = (s0 if isinstance(s0, torch.Tensor) else s0.blocks).device  # slots are the big objects
    xf = x.to(device=dev, dtype=torch.float32)
    final = torch.zeros(T, K, dtype=torch.float32, device=dev)
    w_cpu = weights.cpu()
    for e in range(E):
        pos = (expert_of == e).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        token_idx = pos[:, 0].to(dev)
        gu_rows = _rows_from_slot(slots_gate_up[e], K)
        cur = xf.index_select(0, token_idx)
        gate_up = matvec_slot(slots_gate_up[e], gu_rows, K, cur, dtype=dtype, rows_per_chunk=rows_per_chunk)
        gate_up = gate_up + _bias(biases_gate_up, e, dev)
        h = swiglu(gate_up, alpha=alpha, limit=limit)
        inter = h.shape[-1]
        dn_rows = _rows_from_slot(slots_down[e], inter)
        y = matvec_slot(slots_down[e], dn_rows, inter, h, dtype=dtype, rows_per_chunk=rows_per_chunk)
        y = y + _bias(biases_down, e, dev)
        scale = w_cpu[pos[:, 0], pos[:, 1]].to(dev).unsqueeze(-1)
        final.index_add_(0, token_idx, y * scale)
    return final if final.device == out_dev else final.to(out_dev)


def _bias(biases: Any, e: int, device: Any) -> Any:
    b = biases[e]
    return b.to(device=device, dtype=torch.float32)


def _rows_from_slot(slot_u8: torch.Tensor | MxWeight, K: int) -> int:
    """Recover ``rows`` from a slot's byte count: rows * (K/2 + K/32)."""
    if K % 32:
        raise ValueError(f"K must be a multiple of 32, got {K}")
    if not isinstance(slot_u8, torch.Tensor):
        r, k = slot_u8.shape
        if k != K:
            raise ValueError(f"MxWeight has K={k}, expected {K}")
        return r
    n = slot_u8.reshape(-1).numel()
    per_row = K // 2 + K // 32
    if n % per_row:
        raise ValueError(f"slot of {n} bytes is not a whole number of rows at K={K}")
    return n // per_row
