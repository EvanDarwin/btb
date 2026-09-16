# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The 12-bit model format, version 1, readable with torch and safetensors alone.

A bf16 tensor is a low byte per element and a 4-bit code per element into a 15-entry table of high bytes; the
elements whose high byte is not in the table escape by index. On disk, little-endian throughout, a packed tensor
is two safetensors entries: `<name>.pack12`, uint8 - the low bytes (n), the codes two to a byte (ceil(n/2), the
first element in the low nibble), zero padding to a multiple of 4, the escape indices (int32, ascending), the
escape high bytes (uint8), zero padding to a multiple of 4 - and `<name>.pack12.meta`, uint8 - the table (16
bytes, entries past the fifteenth zero), the escape count (uint32), the rank (uint32), the shape (int64 x rank).
The table lists the high bytes by frequency, ties by value, so a tensor packs to the same bytes every time. The
escape indices being int32, a tensor packs up to 2**31 - 1 elements; `pack_bf16` refuses a larger one. A model's
layers are written this way into shards beside a `config.json` whose `btb` key names the format, its version and
the parent checkpoint every other tensor is read from (`{"format": "pack12", "version": 1, "source": ...}`); a
store of another version is refused on read. `entries` reads a model's shards; `load_state_dict` widens it."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .hf import PACK12_FORMAT as FORMAT
from .hf import pack_format, resolve, shard_map
from .kinds import Json
from .options import BadPack

VERSION = 1
ESC = 15
MAX_ELEMENTS = 2**31 - 1  # the escape indices are int32
SUFFIX = "-pack12"
SHARD = "pack12-{:05d}.safetensors"
META = 24  # the meta's fixed part: the table (16), the escape count (uint32), the rank (uint32); the shape follows


def check_size(n: int) -> None:
    """a tensor of `n` elements packs, or ValueError: past MAX_ELEMENTS the escape indices cannot address it"""
    if int(n) > MAX_ELEMENTS:
        raise ValueError(
            f"{int(n)} elements: the format's escape indices are int32, so a tensor packs up to {MAX_ELEMENTS}"
        )


def pack_bf16(t: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    check_size(t.numel())
    v = t.detach().contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)
    lo = (v & 0xFF).astype(np.uint8)
    hi = (v >> 8).astype(np.uint8)
    counts = np.bincount(hi, minlength=256)
    top = np.argsort(-counts, kind="stable")[:ESC]
    top = top[counts[top] > 0]
    lut = np.full(256, ESC, dtype=np.uint8)
    lut[top] = np.arange(top.size, dtype=np.uint8)
    code = lut[hi]
    esc = np.nonzero(code == ESC)[0]
    esc_idx = esc.astype(np.int32)
    esc_val = hi[esc]
    if code.size % 2:
        code = np.concatenate([code, np.zeros(1, dtype=np.uint8)])
    hi4 = (code[0::2] | (code[1::2] << 4)).astype(np.uint8)
    tbl = np.zeros(16, dtype=np.uint8)
    tbl[: top.size] = top
    return lo, hi4, tbl, esc_idx, esc_val


def unpack_bf16(
    lo: torch.Tensor,
    hi4: torch.Tensor,
    table: torch.Tensor,
    n: int,
    shape: Sequence[int],
    esc_idx: Any = None,
    esc_val: Any = None,
    out: torch.Tensor | None = None,
    chunk: int = 1 << 23,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=lo.device)
    o16 = out.view(torch.int16).view(-1)
    tbl16 = table.to(torch.int16) << 8
    for c in range(0, n, chunk):
        e = min(n, c + chunk)
        h4 = hi4[c // 2 : (e + 1) // 2]
        codes = torch.stack([h4 & 15, h4 >> 4], dim=-1).reshape(-1)
        if c % 2:
            codes = codes[1:]
        codes = codes[: e - c].to(torch.int32)
        hi = torch.index_select(tbl16, 0, codes)
        o16[c:e] = hi | lo[c:e].to(torch.int16)
    if esc_idx is not None and esc_idx.numel():
        ev = (esc_val.to(torch.int16) << 8) | lo[esc_idx.long()].to(torch.int16)
        o16[esc_idx.long()] = ev
    return out


def blob(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 tensor as its packed bytes - lo, hi4, padding to 4, escape indices (int32), escape values, padding
    to 4 - and its meta: the table, the escape count, the rank and the shape. Every tensor a multiple of 4 bytes
    keeps the int32 escape indices aligned wherever safetensors places it."""
    lo, hi4, tbl, esc_idx, esc_val = pack_bf16(t)
    pad = (-(lo.nbytes + hi4.nbytes)) % 4
    body = lo.tobytes() + hi4.tobytes() + b"\0" * pad + esc_idx.tobytes() + esc_val.tobytes()
    body += b"\0" * ((-len(body)) % 4)
    shape = [int(d) for d in t.shape]
    meta = tbl.tobytes() + struct.pack(f"<II{len(shape)}q", int(esc_idx.size), len(shape), *shape)
    return torch.frombuffer(bytearray(body), dtype=torch.uint8), torch.frombuffer(bytearray(meta), dtype=torch.uint8)


def entry(shard: str, off: int, meta: bytes) -> Json:
    """The engine's record of one packed tensor from its meta: its shard and byte offset, the layout within (lo,
    hi4, pad, esc), its size and shape, and the table."""
    esc, rank = struct.unpack("<II", meta[16:24])
    shape = [int(d) for d in struct.unpack(f"<{rank}q", meta[24 : 24 + 8 * rank])]
    n = 1
    for d in shape:
        n *= d
    lo, hi4 = n, (n + 1) // 2
    return {
        "shard": shard,
        "off": off,
        "raw": False,
        "shape": shape,
        "n": n,
        "lo": lo,
        "hi4": hi4,
        "pad": (-(lo + hi4)) % 4,
        "esc": int(esc),
        "table": list(meta[:16]),
    }


def _header(path: str) -> tuple[Json, int]:
    """A safetensors file's header and the offset its data begins at."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def parent_dir(model_dir: str) -> str:
    """The directory of the parent a 12-bit model names: a path relative to the model (or absolute) where that
    directory exists, else a repo id through the cache."""
    fmt = pack_format(model_dir)
    if fmt is None:
        raise ValueError(f"{model_dir} is not a {FORMAT} model")
    src = str(fmt["source"])
    # absolute, so a shard path built on it survives `os.path.join(model_dir, shard)` whatever form `model_dir` took
    beside = os.path.normpath(os.path.join(os.path.abspath(model_dir), src))
    return beside if os.path.isdir(beside) else os.path.abspath(resolve(src))


def check_version(model_dir: str) -> None:
    """the store at `model_dir` is one this reader knows, or BadPack naming its version"""
    fmt = pack_format(model_dir)
    if fmt is None:
        raise BadPack(model_dir, f"not a {FORMAT} model")
    v = fmt.get("version")
    if v != VERSION:
        raise BadPack(
            model_dir,
            f"{FORMAT} version {v if v is not None else 'unversioned, a pre-release pack'}; this btb reads version "
            f"{VERSION}: re-pack it with `btb pack`",
        )


def entries(model_dir: str) -> dict[str, Json]:
    """Every packed tensor of a 12-bit model by weight name, as `entry` records, from the shards alone."""
    check_version(model_dir)
    wm = shard_map(model_dir)
    headers: dict[str, tuple[Json, int]] = {}

    def header(shard: str) -> tuple[Json, int]:
        if shard not in headers:
            headers[shard] = _header(os.path.join(model_dir, shard))
        return headers[shard]

    out: dict[str, Json] = {}
    tail = "." + FORMAT
    for name, shard in wm.items():
        if not name.endswith(tail):
            continue
        key = name[: -len(tail)]
        hdr, base = header(shard)
        a, _b = hdr[name]["data_offsets"]
        mshard = wm[name + ".meta"]
        mh, mbase = header(mshard)
        ma, mb = mh[name + ".meta"]["data_offsets"]
        with open(os.path.join(model_dir, mshard), "rb") as f:
            f.seek(mbase + ma)
            meta = f.read(mb - ma)
        out[key] = entry(shard, base + a, meta)
    return out


def load_state_dict(model_dir: str, device: Any = "cpu") -> dict[str, torch.Tensor]:
    """A 12-bit model as bf16 tensors by weight name, for any torch program: its packed layers widened, every other
    tensor read from the parent as it is."""
    from safetensors.torch import load_file

    ents = entries(model_dir)
    tail = "." + FORMAT
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(shard_map(model_dir).values())):
        for name, t in load_file(os.path.join(model_dir, shard), device=str(device)).items():
            if not name.endswith(tail):
                continue
            e = ents[name[: -len(tail)]]
            a1 = e["lo"]
            a2 = a1 + e["hi4"] + e["pad"]
            a3 = a2 + 4 * e["esc"]
            esc_idx = t[a2:a3].clone().view(torch.int32) if e["esc"] else None
            esc_val = t[a3 : a3 + e["esc"]] if e["esc"] else None
            table = torch.tensor(e["table"], dtype=torch.uint8, device=t.device)
            out[name[: -len(tail)]] = unpack_bf16(t[:a1], t[a1:a2], table, e["n"], e["shape"], esc_idx, esc_val)
    parent = parent_dir(model_dir)
    for shard in sorted(set(shard_map(parent).values())):
        for name, t in load_file(os.path.join(parent, shard), device=str(device)).items():
            out.setdefault(name, t)
    return out
