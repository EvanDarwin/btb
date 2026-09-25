# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Fine-grained FP8 as checkpoints store it: e4m3fn weights and an f32 scale per block of a grid over each matrix
(`weight_scale_inv`; a fused expert tensor's `<name>_scale_inv` carries a grid per expert), or one per-tensor
`weight_scale` (an FP8 embedding table). A weight is its e4m3 value times its block's scale; the block is the
matrix's shape over the grid's, so every layout reads by the one rule transformers' `Fp8Dequantize` applies."""

from __future__ import annotations

from collections.abc import Container

import torch

E4M3 = torch.float8_e4m3fn
FP8_MAX = 448.0  # the largest finite e4m3fn magnitude


def scale_key(key: str, keys: Container[str]) -> str | None:
    """the scale a stored FP8 tensor is multiplied by, among `keys`: `X.weight_scale_inv` (a block grid) or
    `X.weight_scale` (per-tensor) for `X.weight`, `X_scale_inv` for a fused expert tensor `X`"""
    cands: tuple[str, ...]
    if key.endswith(".weight"):
        base = key[: -len(".weight")]
        cands = (base + ".weight_scale_inv", base + ".weight_scale")
    else:
        cands = (key + "_scale_inv",)
    return next((k for k in cands if k in keys), None)


def scale_grid(s: torch.Tensor) -> torch.Tensor:
    """a stored scale as f32 with its grid in the last two dims: e8m0 exponents (a uint8 or float8_e8m0fnu
    tensor) as `2^(e - 127)`, and a per-tensor scalar or `(1,)` as a 1x1 grid"""
    f = torch.exp2(s.float() - 127.0) if s.dtype == torch.uint8 else s.float()
    return (f.reshape(1, 1) if f.dim() < 2 else f).contiguous()


def widen(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """`q * scale` blockwise in f32, then `dtype`: `q` e4m3fn (or its bytes as uint8) with the matrix in its last
    two dims, `s` its stored scale (`scale_grid`), one grid per leading index or one for all"""
    rows, cols = q.shape[-2:]
    g = scale_grid(s)
    sr, sc = g.shape[-2:]
    if rows % sr or cols % sc:
        raise ValueError(f"[fp8] a {rows}x{cols} matrix does not split into a {sr}x{sc} scale grid")
    qf = (q.view(E4M3) if q.dtype == torch.uint8 else q).float()
    out = qf.reshape(-1, sr, rows // sr, sc, cols // sc) * g.reshape(-1, sr, 1, sc, 1)
    return out.to(dtype).reshape(q.shape)


def quantize(t: torch.Tensor, block: tuple[int, int] | None) -> tuple[torch.Tensor, torch.Tensor]:
    """`t`'s matrices (its last two dims) as e4m3fn and their f32 scales, transformers' `Fp8Quantize`: each block
    of `block` rows by columns (the whole matrix when None) scaled so its largest magnitude is `FP8_MAX`, the scale
    stored as its inverse. The grid is `[..., rows / bm, cols / bn]`."""
    rows, cols = t.shape[-2:]
    bm, bn = block if block is not None else (rows, cols)
    if rows % bm or cols % bn:
        raise ValueError(f"[fp8] a {rows}x{cols} matrix does not split into {bm}x{bn} blocks")
    lead = t.shape[:-2]
    tiles = t.float().reshape(*lead, rows // bm, bm, cols // bn, bn)
    peak = tiles.abs().amax(dim=(-3, -1))
    scale = torch.where(peak > 0, FP8_MAX / torch.where(peak > 0, peak, 1.0), 1.0)
    q = (tiles * scale.unsqueeze(-1).unsqueeze(-3)).clamp(-FP8_MAX, FP8_MAX).to(E4M3)
    return q.reshape(t.shape), (1.0 / scale).float()


class F8Weight:
    """One FP8 matrix as stored, views never copied: `w` its `rows * cols` e4m3 bytes (uint8) row-major, `scales`
    its f32 grid `[sr, sc]` (contiguous), `shape` = (rows, K)."""

    __slots__ = ("scales", "shape", "w")

    def __init__(self, w: torch.Tensor, scales: torch.Tensor, rows: int, k: int) -> None:
        if w.numel() != rows * k:
            raise ValueError(f"[fp8] [{rows}, {k}] needs {rows * k} bytes, got {w.numel()}")
        g = scale_grid(scales)
        if g.dim() != 2 or rows % g.shape[0] or k % g.shape[1]:
            raise ValueError(f"[fp8] a {tuple(g.shape)} grid does not divide [{rows}, {k}]")
        self.w = w.view(torch.uint8) if w.dtype == E4M3 else w
        self.scales = g
        self.shape = (int(rows), int(k))

    @property
    def grid(self) -> tuple[int, int]:
        return int(self.scales.shape[0]), int(self.scales.shape[1])

    @property
    def nbytes(self) -> int:
        return self.w.numel() + self.scales.numel() * 4

    def __repr__(self) -> str:
        return f"F8Weight[{self.shape[0]}, {self.shape[1]}, grid {self.grid[0]}x{self.grid[1]}]"

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """the matrix as the engine holds it, `held(e4m3 * scale)`, in `dtype`: what the native kernel reads"""
        return held(self.w.reshape(self.shape), self.scales).to(dtype)


def held(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """an FP8 matrix as the engine holds every checkpoint's weights: `widen` to bf16, each weight the nearest bf16
    to its e4m3 value times its scale - the value the native kernel, a bf16 MLX slot and a card all read"""
    return widen(q, s, torch.bfloat16)
