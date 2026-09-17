#!/usr/bin/env python
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Get a machine ready to bench the rivals: the comparison venv (.venv-compare, with mlx-lm / airllm /
llama-cpp-python pinned to bench/requirements.txt) and, only if you want to bench a plain checkpoint that has
no GGUF, a llama.cpp checkout for its HF->GGUF converter. Idempotent - re-running only does the missing steps.
Run from the repo root: `python bench/prepare.py`. GGUF models (the quant repos, a repo's BF16 GGUF) need no
converter, so `--skip-llama-cpp` is the common case."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import COMPARE_PY, ROOT, say
from lib.env import compare_env_mismatch, compare_tools, setup_compare_env

# the upstream repo; its convert_hf_to_gguf.py is the only piece llama-cpp-python does not ship (the runtime
# loads a GGUF, it does not build one)
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def _say(msg: str) -> None:
    say(msg, "prepare")


def prepare_compare_env() -> bool:
    """Make .venv-compare match bench/requirements.txt; a no-op when it already does."""
    missing = compare_env_mismatch(COMPARE_PY)
    if not missing:
        _say(f".venv-compare is ready: {', '.join(compare_tools(COMPARE_PY))}")
        return True
    _say(f".venv-compare needs {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}; setting it up")
    if not setup_compare_env(COMPARE_PY):
        return False
    _say(f".venv-compare is ready: {', '.join(compare_tools(COMPARE_PY))}")
    return True


def prepare_llama_cpp(dest: str) -> bool:
    """A llama.cpp checkout at `dest` for the HF->GGUF converter; a shallow clone, skipped if it is already
    there. Only needed to bench a checkpoint that has no GGUF - the matrix finds the converter here when `dest`
    is ./llama.cpp (its default) or LLAMA_CPP_DIR points at it."""
    if os.path.exists(os.path.join(dest, "convert_hf_to_gguf.py")):
        _say(f"llama.cpp is ready at {dest}")
        return True
    _say(f"cloning llama.cpp into {dest} (shallow; for HF->GGUF conversion) ...")
    rc = subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, dest], check=False).returncode
    if rc != 0 or not os.path.exists(os.path.join(dest, "convert_hf_to_gguf.py")):
        _say("that clone failed; converting a plain checkpoint will be unavailable (GGUF models are unaffected)")
        return False
    _say(f"llama.cpp is ready at {dest}")
    if dest != os.path.join(ROOT, "llama.cpp"):
        _say(f"  set LLAMA_CPP_DIR={dest} so the matrix finds the converter")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--llama-cpp",
        default=os.environ.get("LLAMA_CPP_DIR") or os.path.join(ROOT, "llama.cpp"),
        metavar="DIR",
        help="where the llama.cpp checkout goes (default: LLAMA_CPP_DIR or ./llama.cpp, where the matrix looks)",
    )
    ap.add_argument(
        "--skip-llama-cpp",
        action="store_true",
        help="only set up the venv; skip the llama.cpp checkout (fine unless you bench a checkpoint with no GGUF)",
    )
    a = ap.parse_args(argv)
    ok = prepare_compare_env()
    if not a.skip_llama_cpp:
        ok = prepare_llama_cpp(a.llama_cpp) and ok
    _say("ready to bench" if ok else "setup incomplete; see the messages above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
