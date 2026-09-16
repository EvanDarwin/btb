# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Model discovery: the complete, servable models in the Hugging Face cache and in given directories, their
names as the server reports them, a 12-bit model's record of its parent, and resolving a repo id to a local
snapshot."""

from __future__ import annotations

import glob
import json
import os
import re
import struct
import sys
from collections.abc import Iterable
from typing import Any, TypedDict

from .kinds import Json

# model_types the engine can actually serve (mirrors StreamedTextModel.family); a downloaded repo of any other
# type is skipped by the discovery below so it never shows up as a servable model
SERVE_TYPES = frozenset({"qwen3", "qwen3_5", "qwen3_5_text", "phi3", "qwen4_exp", "qwen4_exp_text", "gpt_oss"})

# the files a model is: what `resolve` downloads of a repo, and what `btb pack` copies beside its packed weights
MODEL_FILES = ("*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken")
GGUF_EXT = ".gguf"
# btb's families by their llama.cpp architecture name (general.architecture), an explicit table
ARCH_MODEL_TYPES = {"qwen3": "qwen3", "phi3": "phi3", "gpt-oss": "gpt_oss"}
# the GGUF storage types whose blocks are an affine quantization at group 32 (a scale and an offset per 32
# values), the form the packed kernels multiply as stored; the value is the bits an integer takes
AFFINE_TYPES = {"Q4_0": 4, "Q4_1": 4, "Q8_0": 8, "Q4_K": 4}


def is_gguf(path: Any) -> bool:
    """whether `path` names a GGUF file (llama.cpp's format)"""
    return isinstance(path, str) and path.lower().endswith(GGUF_EXT)


def model_stem(path: str) -> str:
    """what a model is called from its path: a directory's name, a GGUF file's name without the extension"""
    base = os.path.basename(os.path.normpath(str(path).rstrip("/")))
    return base[: -len(GGUF_EXT)] if is_gguf(base) else base


def gguf_lib() -> Any:
    import gguf

    return gguf


# the byte width of each fixed-size GGUF metadata value type (uint8/int8/bool 1, uint16 2, u32/i32/f32 4,
# u64/i64/f64 8); STRING (8) and ARRAY (9) carry their own length and are handled apart
_GGUF_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_skip(f: Any, vt: int) -> None:
    """advance past one metadata value of type `vt` without materializing it; raises on a type this reader does
    not know, so `gguf_arch` falls back to the full library reader"""
    if vt in _GGUF_SCALAR:
        f.seek(_GGUF_SCALAR[vt], 1)
    elif vt == 8:  # string: a u64 length then that many bytes
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif vt == 9:  # array: element type, count, then the payload skipped by its own width
        et, n = struct.unpack("<IQ", f.read(12))
        if et == 8:
            for _ in range(n):  # a run of strings (the tokenizer vocab): each a u64 length then its bytes
                f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        elif et in _GGUF_SCALAR:
            f.seek(_GGUF_SCALAR[et] * n, 1)
        else:
            raise ValueError(f"gguf array of type {et}")
    else:
        raise ValueError(f"gguf value type {vt}")


def _gguf_arch_fast(path: str) -> str | None:
    """`general.architecture` read by scanning the metadata key-values and stopping at it, skipping every other
    value (the tokenizer's 150k-string array included) by its byte length rather than decoding it. None when the
    format is not the v2/v3 little-endian this scanner handles, so the caller falls back to the library reader."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF" or struct.unpack("<I", f.read(4))[0] not in (2, 3):
            return None
        f.seek(8, 1)  # tensor count, unused here
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            key = f.read(struct.unpack("<Q", f.read(8))[0])
            vt = struct.unpack("<I", f.read(4))[0]
            if key == b"general.architecture":
                if vt != 8:
                    return None
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
            _gguf_skip(f, vt)
    return None


def gguf_arch(path: str) -> str | None:
    """a GGUF file's architecture off its header (the whole file is not read); None when it cannot be read"""
    try:
        arch = _gguf_arch_fast(path)
        if arch is not None:
            return arch
    except Exception:
        pass  # an unusual header (v1, big-endian, a type the scanner skips): the library reader below handles it
    try:
        g = gguf_lib()
        return str(g.GGUFReader(path).get_field(g.Keys.General.ARCHITECTURE).contents())
    except Exception:
        return None


# the format btb writes as a sibling model that reads the rest of its weights from a parent: the 12-bit store
# (`btb pack`)
PACK12_FORMAT = "pack12"
PACK_FORMATS = frozenset({PACK12_FORMAT})


class ModelEntry(TypedDict):
    """One servable model: its served name, repo id, snapshot directory, model_type, weight bytes, and whether
    it is a 12-bit model (`btb pack`)."""

    name: str
    repo: str
    path: str
    type: str | None
    size: int
    packed: bool


def serve_name(raw: str | None) -> str:
    """The Ollama/OpenAI id for a model: its name lowercased with anything but [A-Za-z0-9_.-] turned to a dash
    (so `Qwen/Qwen3.5-4B` -> `qwen3.5-4b`). One rule everywhere, so a name from /api/tags round-trips back."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw or "")).strip("-").lower() or "btb"


def _config(d: str) -> Json | None:
    p = os.path.join(d, "config.json")
    if not os.path.isfile(p):
        return None
    try:
        return dict(json.load(open(p, encoding="utf-8")))
    except Exception:
        return None


def _model_type(d: str) -> str | None:
    if is_gguf(d):
        return ARCH_MODEL_TYPES.get(gguf_arch(d) or "")
    c = _config(d)
    return None if c is None else str(c.get("model_type") or "")


def pack_format(path: str) -> Json | None:
    """The `btb` record of a sibling model's config.json - `{"format": "pack12", "version": 1, "source": <the
    parent: a repo id or a directory>}` - or None for a plain model. The version is checked where the store is
    read (`pack12.check_version`)."""
    c = _config(path)
    b = c.get("btb") if c else None
    return dict(b) if isinstance(b, dict) and b.get("format") in PACK_FORMATS else None


def is_packed(path: str) -> bool:
    """Whether `path` is a 12-bit packed model (`btb pack`)."""
    return (pack_format(path) or {}).get("format") == PACK12_FORMAT


def local_parent(path: str) -> str | None:
    """The local directory a 12-bit model's parent lives in, found without a download: a path beside the sibling,
    or the parent's cache entry at the same snapshot revision. None when only a download would find it."""
    fmt = pack_format(path)
    src = str(fmt.get("source") or "") if fmt else ""
    if not src:
        return None
    beside = os.path.normpath(os.path.join(os.path.abspath(path), src))
    if os.path.isdir(beside):
        return beside
    if "/" in src:  # a repo id: the parent cache entry beside this one, at the same snapshot revision
        snap = os.path.abspath(path)
        hub = os.path.dirname(os.path.dirname(os.path.dirname(snap)))
        org, _, name = src.partition("/")
        cand = os.path.join(hub, f"models--{org}--{name}", "snapshots", os.path.basename(snap))
        if os.path.isdir(cand):
            return cand
    return None


def cache_repo_id(path: str) -> str | None:
    """The repo id of a snapshot directory in the Hugging Face cache (`.../models--Qwen--Qwen3-4B/snapshots/<rev>`
    -> `Qwen/Qwen3-4B`), else None."""
    p = os.path.normpath(os.path.abspath(str(path))) + os.sep
    m = re.search(r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]", p)
    return f"{m.group(1)}/{m.group(2)}" if m else None


def shard_map(model_dir: str) -> dict[str, str]:
    """Tensor name -> safetensors file, from the index or, for an unsharded checkpoint, off the files' headers."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"])
    out: dict[str, str] = {}
    for name in sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors")):
        with open(os.path.join(model_dir, name), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            for k in json.loads(fh.read(n)):
                if k != "__metadata__":
                    out[k] = name
    return out


def _model_complete(d: str) -> bool:
    """True when `d` holds a full set of weights: config.json, and either a single safetensors file or an index
    whose every shard is present (a partly-downloaded repo fails here), and no leftover *.incomplete blobs."""
    if is_gguf(d):
        return os.path.isfile(d)
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    if glob.glob(os.path.join(d, "*.incomplete")):
        return False
    if pack_format(d) is not None and local_parent(d) is None:
        # a 12-bit model reads the rest of its tensors from its parent: not servable until the parent is here
        return False
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        try:
            wm = json.load(open(idx, encoding="utf-8")).get("weight_map", {})
        except Exception:
            return False
        return bool(wm) and all(os.path.isfile(os.path.join(d, f)) for f in set(wm.values()))
    if os.path.isfile(os.path.join(d, "model.safetensors")):
        return True
    return bool(glob.glob(os.path.join(d, "*.safetensors")))


def _model_bytes(d: str) -> int:
    if is_gguf(d):
        return os.path.getsize(d) if os.path.isfile(d) else 0
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.safetensors")) if os.path.isfile(f))


def _cache_model_dirs() -> list[tuple[str, str]]:
    """Every model snapshot in the Hugging Face cache as (repo_id, directory). A missing huggingface_hub raises;
    a scan failure yields nothing."""
    from huggingface_hub import scan_cache_dir

    out: list[tuple[str, str]] = []
    try:
        info = scan_cache_dir()
    except Exception:
        return out
    for repo in info.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        # newest first, so the first revision that is complete wins (a repo can hold an older full snapshot
        # beside a newer half-downloaded one)
        out.extend(
            (repo.repo_id, str(rev.snapshot_path))
            for rev in sorted(repo.revisions, key=lambda r: r.last_modified or 0, reverse=True)
        )
    return out


def available_models(paths: Iterable[str] = (), pattern: str | None = None) -> list[ModelEntry]:
    """The complete, servable models: the Hugging Face cache plus each of `paths` (a model directory, or a
    directory of them), as `ModelEntry` records sorted by name; `pattern` a case-insensitive regex searched
    against the name or repo id."""
    rx = re.compile(pattern, re.IGNORECASE) if pattern else None
    cands: list[tuple[str, str]] = []
    # the paths named first: a model the user pointed at outranks a cache entry of the same name
    for p in paths:
        if not p:
            continue
        p = os.path.normpath(p)
        if is_gguf(p) and os.path.isfile(p):
            cands.append((model_stem(p), p))
        elif os.path.isfile(os.path.join(p, "config.json")):
            cands.append((os.path.basename(p), p))
        elif os.path.isdir(p):
            for sub in sorted(os.listdir(p)):
                d = os.path.join(p, sub)
                if is_gguf(sub) and os.path.isfile(d):
                    cands.append((model_stem(d), d))
                elif os.path.isfile(os.path.join(d, "config.json")):
                    cands.append((sub, d))
    for repo, d in _cache_model_dirs():
        cands.append((repo, d))
        # a repository of GGUF files (Qwen/Qwen3-4B-GGUF): every file its own model, named for the file
        cands.extend((f"{repo}:{f}", os.path.join(d, f)) for f in sorted(os.listdir(d)) if is_gguf(f))
    out: dict[str, ModelEntry] = {}
    for repo, d in cands:
        if not _model_complete(d):
            continue
        mt = _model_type(d)
        if mt not in SERVE_TYPES:
            continue
        # name from the repo id (the snapshot directory is named for a commit hash, not the model), a GGUF
        # file for the file
        name = serve_name(model_stem(d) if is_gguf(d) else os.path.basename(str(repo).rstrip("/")))
        if rx and not (rx.search(name) or rx.search(repo)):
            continue
        if name in out:
            continue
        out[name] = ModelEntry(name=name, repo=repo, path=d, type=mt, size=_model_bytes(d), packed=is_packed(d))
    return [out[k] for k in sorted(out)]


def _download(path: str) -> str:
    from huggingface_hub import snapshot_download

    # the root of the repo only: a pattern matches across "/", and gpt-oss keeps a second copy of the
    # weights under original/ (61 GB) and a third under metal/. A cached repo is taken as it is: the hub's
    # revalidation rewrites its refs/main, and a load must not touch a file under the model's directory
    kw: dict[str, Any] = {"allow_patterns": list(MODEL_FILES), "ignore_patterns": ["*/*"]}
    try:
        return snapshot_download(path, local_files_only=True, **kw)
    except Exception:
        return snapshot_download(path, **kw)


def _hard_exit(code: int) -> None:  # abandons the download threads at once, past concurrent.futures' atexit join
    os._exit(code)


def _fetch(path: str) -> str:
    """Download `path` from the Hub on a daemon thread while the main thread waits interruptibly, so Ctrl-C is
    answered at once (a synchronous parallel download otherwise holds the main thread until every in-flight file
    finishes). On Ctrl-C the process exits 130 and the partial files stay resumable."""
    import threading

    got: list[str] = []
    err: list[BaseException] = []

    def work() -> None:
        try:
            got.append(_download(path))
        except BaseException as e:  # re-raised on the caller's thread below
            err.append(e)

    t = threading.Thread(target=work, name="btb-download", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.2)
    except KeyboardInterrupt:
        sys.stderr.write("\n\n[btb] download cancelled\n")
        sys.stderr.flush()
        _hard_exit(130)
    if err:
        raise err[0]
    return got[0]


def resolve(path: str, local: bool = False) -> str:
    """A model directory for `path`: the directory itself, or a repo id's snapshot in the cache, downloaded when
    it is not there. `local` never downloads: a repo id that is not in the cache is FileNotFoundError."""
    if os.path.isdir(path):
        return path
    if is_gguf(path) and os.path.isfile(path):
        return os.path.abspath(path)
    head, _, tail = path.rpartition(":")
    if is_gguf(tail) and "/" in head and not os.path.exists(path) and not os.path.isabs(head):
        # `repo/id:file.gguf`: one file of a repository of GGUF files, fetched into the cache
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(head, tail, local_files_only=local)
        except Exception as e:
            if local:
                raise FileNotFoundError(path) from e
            raise
    if "/" in path and not os.path.exists(path) and not os.path.isdir(path.split("/", 1)[0]):
        # a repo id, not a mistyped local path: `tests/nothing-here` has the shape of one, but its first segment is
        # a directory here, so it is answered as a missing path rather than sent to the Hub
        if not local:
            return _fetch(path)
        from huggingface_hub import snapshot_download

        try:
            return snapshot_download(
                path, local_files_only=True, allow_patterns=list(MODEL_FILES), ignore_patterns=["*/*"]
            )
        except Exception as e:  # the bare path: the CLI's not-found report keys on it
            raise FileNotFoundError(path) from e
    raise FileNotFoundError(path)
