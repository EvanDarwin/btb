# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The pack command: `pack_model` writes a model's layers as a 12-bit model beside it (`btb.pack12` the format)."""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import time
from typing import Any

import torch

from ..hf import MODEL_FILES, cache_repo_id, is_gguf
from ..kinds import Log
from ..options import NotPackable
from ..pack12 import ESC, FORMAT, SHARD, SUFFIX, VERSION, blob, pack_bf16, unpack_bf16  # noqa: F401

SHARD_BYTES = 2 * 2**30


def pack_dir(model_dir: str, suffix: str = SUFFIX) -> str:
    """Where a pack writes for `model_dir`: a snapshot in the Hugging Face cache gets a sibling `<repo><suffix>`
    entry there; any other directory gets `<dir><suffix>` beside it."""
    snap = os.path.normpath(os.path.abspath(model_dir))
    if cache_repo_id(snap) is None:
        return snap + suffix
    root = os.path.dirname(os.path.dirname(snap))  # .../models--org--name
    return os.path.join(root + suffix, "snapshots", os.path.basename(snap))


def source_for(model_dir: str, out_dir: str) -> str:
    """How a 12-bit model at `out_dir` names its parent: the repo id of a cache snapshot, else the parent's path
    relative to `out_dir` (forward slashes, so the pair moves together across machines), absolute across drives."""
    snap = os.path.normpath(os.path.abspath(model_dir))
    repo = cache_repo_id(snap)
    if repo is not None:
        return repo
    try:
        return os.path.relpath(snap, start=os.path.abspath(out_dir)).replace(os.sep, "/")
    except ValueError:
        return snap


def _free_bytes(path: str) -> int:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def _copy_small_files(model_dir: str, out_dir: str) -> None:
    """The model's files that are not weights, copied."""
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src) and any(fnmatch.fnmatch(name, pat) for pat in MODEL_FILES):
            shutil.copy2(src, os.path.join(out_dir, name))


def _stamp(out_dir: str, record: dict[str, Any]) -> None:
    """config.json gains the `btb` record: written last, so a pack that stopped half way is not a 12-bit model"""
    p = os.path.join(out_dir, "config.json")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    c["btb"] = record
    with open(p, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=2)
        f.write("\n")


def _write_refs(out_dir: str) -> None:
    """A cache sibling's refs/main, so the hub resolves its snapshot: the snapshot directory's own name."""
    refs = os.path.join(os.path.dirname(os.path.dirname(out_dir)), "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w", encoding="utf-8") as f:
        f.write(os.path.basename(out_dir))


def pack_model(model_dir: str, out_dir: str | None = None, prefix_filter: str = "layers.", log: Log = print) -> str:
    """Write the 12-bit model for `model_dir`: its bf16 layer tensors packed into safetensors shards, its small
    files copied, its config.json naming the format and the parent every other tensor is read from. Beside the
    model by default (`pack_dir`), or at `out_dir`. The checkpoint is not modified. Returns the directory written."""
    from safetensors.torch import save_file

    if is_gguf(model_dir):
        raise NotPackable(os.path.basename(model_dir), "a GGUF file cannot be packed; as it is already packed")

    # here, not at the top: the engine's tiers import this module for the format, and the store needs the
    # engine to walk the checkpoint - a cycle at import time, none at call time
    from .model import StreamedTextModel

    in_cache = out_dir is None and cache_repo_id(model_dir) is not None
    out_dir = os.path.normpath(out_dir) if out_dir else pack_dir(model_dir)
    source = source_for(model_dir, out_dir)
    sm = StreamedTextModel(model_dir, device="cpu", resident_head=False, log=lambda *_a: None)
    try:
        # an MXFP4 expert is already 4.25 bits and the expert store reads it out of the checkpoint as it is:
        # the 12-bit store has nothing to add and would write the model's bulk a second time
        skip = lambda k: bool(sm.fam.mxfp4) and ".experts." in k and not k.endswith("_bias")
        keys = sorted(k for k in sm.weight_map if prefix_filter in k and not skip(k))
        src = 0
        for k in keys:
            _, hdr, _ = sm._shard(sm.weight_map[k])
            a, b = hdr[k]["data_offsets"]
            src += b - a
        free = _free_bytes(out_dir)
        log(
            f"[pack] {len(keys)} tensors - {src / 2**30:.1f} GB uncompressed -> ~{0.76 * src / 2**30:.1f} GB compressed"
            f"- ({free / 2**30:.0f} GB free)"
        )
        if 0.76 * src > free:
            raise RuntimeError(
                f"[pack] not enough free space in {out_dir}: ~{0.76 * src / 2**30:.0f} GB needed, but only "
                f"{free / 2**30:.0f} GB is free"
            )
        os.makedirs(out_dir, exist_ok=True)
        _copy_small_files(model_dir, out_dir)
        if in_cache:
            _write_refs(out_dir)
        weight_map: dict[str, str] = {}
        pending: dict[str, torch.Tensor] = {}
        shard_i = shard_bytes = 0
        t0 = time.time()
        tot_in = tot_out = n_pack = 0

        def flush() -> None:
            nonlocal shard_i, shard_bytes, pending
            if not pending:
                return
            name = SHARD.format(shard_i)
            save_file(pending, os.path.join(out_dir, name), metadata={"format": "pt"})
            for k in pending:
                weight_map[k] = name
            shard_i += 1
            shard_bytes = 0
            pending = {}

        for j, k in enumerate(keys):
            t = sm._get(k)
            if t.dtype == torch.bfloat16:  # anything else is read from the parent as it is
                tot_in += t.numel() * t.element_size()
                try:
                    body, meta = blob(t)
                except ValueError as e:  # past the format's element cap
                    raise NotPackable(os.path.basename(model_dir), f"{k}: {e}") from None
                pending[f"{k}.{FORMAT}"] = body
                pending[f"{k}.{FORMAT}.meta"] = meta
                shard_bytes += body.numel() + meta.numel()
                tot_out += body.numel() + meta.numel()
                n_pack += 1
                if shard_bytes >= SHARD_BYTES:
                    flush()
            if (j + 1) % 50 == 0 or j + 1 == len(keys):
                log(
                    f"[pack] {j + 1}/{len(keys)} tensors, {tot_in / 2**30:.1f} -> {tot_out / 2**30:.1f} GB "
                    f"({tot_in / max(1, tot_out):.3f}x) in {time.time() - t0:.0f}s"
                )
        flush()
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as fi:
            json.dump({"metadata": {"total_size": tot_out}, "weight_map": weight_map}, fi, indent=2)
        _stamp(out_dir, {"format": FORMAT, "version": VERSION, "source": source})
    finally:
        sm.close()
    name = source + SUFFIX if in_cache else out_dir
    log(
        f"[pack] {n_pack} tensors -> {out_dir} ({tot_out / 2**30:.2f} GB packed from {tot_in / 2**30:.2f}); load it as {name}"
    )
    return out_dir
