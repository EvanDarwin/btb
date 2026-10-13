# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Loading a model for a CLI command: the dependency check, download consent, GGUF quant choice and the load."""

from __future__ import annotations

import argparse
import contextlib
import os
from collections.abc import Callable
from typing import Any

from ..options import Device, DeviceName
from .args import load_args
from .console import say, say_err


def require_deps(device: DeviceName | None) -> None:
    """Fail early and clearly on a missing package (find_spec, so torch is not imported to check for it);
    `mlx` is required only when the device is asked for by name."""
    import importlib.util as u

    need = ["torch", "transformers", "huggingface_hub", "numpy", "safetensors"]
    missing = [m for m in need if u.find_spec(m) is None]
    if device is not None and device.kind is Device.MLX and u.find_spec("mlx") is None:
        missing.append("mlx")
    if not missing:
        return
    say(f"[btb] missing required package(s): {', '.join(missing)}")
    raise SystemExit(2)


def confirm_or_exit(path: str | None) -> None:
    """Consent before a command may download a model from the Hub: a no-op for a missing, local, or
    already-cached path; exits when the user declines an uncached repo (see hf.confirm_download)."""
    if not path:
        return
    from ..hf import confirm_download

    if not confirm_download(path):
        say_err("[btb] aborted; nothing downloaded")
        raise SystemExit(1)


def pick_gguf_quant(path: str) -> str:
    """`path` unchanged for a local model, a normal Hub repo, or an explicit repo/id:file.gguf; a bare GGUF repo
    id (a repo of .gguf files and no config.json) turned into repo/id:file.gguf by asking which quant. On a
    terminal that is a menu; with no terminal to ask, the file list is printed and the run exits, since there is
    no safe default quant to pick."""
    from .. import resolve
    from ..confirm import select
    from ..hf import _gb, is_gguf, repo_gguf_files

    _, _, tail = path.rpartition(":")
    if is_gguf(tail) or os.path.exists(path) or "/" not in path:
        return path  # an explicit .gguf file, a local path, or nothing with a repo id's shape
    # already cached as a normal HF model (it has a config.json): load it as-is, and offline, with no Hub probe
    with contextlib.suppress(Exception):
        if os.path.exists(os.path.join(resolve(path, local=True), "config.json")):
            return path
    files = repo_gguf_files(path)
    if not files:
        return path  # not a GGUF repo, or the Hub is unreachable: let the normal resolve/download path handle it
    labels = [f"{n}  ({_gb(sz)})" if sz else n for n, sz in files]
    i = select(f"{path} is a GGUF repo - choose a quant to download:", labels)
    if i is None:
        say_err(f"[btb] {path} is a GGUF repo; name a file, e.g. {path}:{files[0][0]}")
        say_err("[btb] available files:")
        for n, sz in files:
            say_err(f"         {n}" + (f"  ({_gb(sz)})" if sz else ""))
        raise SystemExit(1)
    return f"{path}:{files[i][0]}"


def log_for(a: argparse.Namespace, out: Callable[..., object]) -> Callable[..., object] | None:
    """the engine's log for -v (`out`, as it is written) and -vv (the trace's: each line said once, timestamped
    among the decisions, off the passes' thread); none without either"""
    verbose = int(getattr(a, "verbose", 0) or 0)
    if verbose >= 2:
        from .. import trace

        return trace.say
    return out if verbose else None


def open_model(a: argparse.Namespace) -> Any:
    """Load the model at `a.path` with the options in `a`. Returns (model, tokenizer)."""
    from .. import load, resolve

    a.path = pick_gguf_quant(a.path)  # a bare GGUF repo id becomes repo/id:file.gguf (asks which quant on a tty)
    a._model = a.path  # the model as the user named it, for the reproduction command (resolve rewrites a.path)
    confirm_or_exit(a.path)
    if not a.draft_model and not getattr(a, "quiet", False):
        from ..hf import draft_notice

        draft_notice(a._model)
    a.path = resolve(a.path)
    kw, nat = load_args(a)
    # run keeps stdout for the model's text, and a --json run for its JSON
    out = say_err if a.cmd == "run" or getattr(a, "as_json", False) else say
    sm = load(a.path, device=a.device, native=nat, log=log_for(a, out), **kw)
    if getattr(a, "profile", None):
        from ..engine.experts import ExpertProfile

        os.makedirs(a.profile, exist_ok=True)
        sm.expert_profile = ExpertProfile(os.path.join(a.profile, "events.npz"))
        sm.expert_profile.watch()
        a._sm = sm  # main() writes the diagnostics bundle around the command, on success or on a crash
    if a.verbose:
        # the placement as the engine itself has it, once the load has settled
        try:
            out(sm.report_line(sm.report()))
        except Exception as e:
            out(f"[report] unavailable: {e!r}")
    return sm, sm.tokenizer
