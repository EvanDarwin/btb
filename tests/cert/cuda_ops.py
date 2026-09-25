"""The CUDA card-kernel matrix. The card runs only where a GPU is present (CI has none), so this certifies the
kernel surface structurally, source-parsed and torch-free: every kernel the engine loads (`_Cuda.KERNELS` in
btb/engine/native.py) must be defined in native/cuda/*.cu|.cuh, and every kernel defined there must be one the
engine loads. A new .cu kernel the engine does not load - or a loaded name with no definition - is a reported
gap, so a card kernel cannot land uncertified.

    python -m tests.cert.cuda_ops --report    # the kernel matrix and the plain-language gaps
    python -m tests.cert.cuda_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.cuda_ops --check      # nonzero when the loaded set and the defined set disagree
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from enum import StrEnum

from .core import BTB_SRC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUDA_DIR = os.path.join(ROOT, "native", "cuda")
NATIVE_PY = os.path.join(BTB_SRC, "engine", "native.py")


class Missing(StrEnum):
    """what a CUDA-kernel gap needs, as a stable key; `MISSING` maps each to (what is absent, how to close it)."""

    DEFINED_NOT_LOADED = "defined-not-loaded"
    LOADED_NOT_DEFINED = "loaded-not-defined"


# kind -> (what is missing about kernel {s}, how to close it).
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.DEFINED_NOT_LOADED: (
        "kernel {s} is defined in native/cuda/*.cu|.cuh but the engine never loads it (not in _Cuda.KERNELS) - "
        "a new card kernel with no wiring or cert",
        "add {s} to _Cuda.KERNELS in btb/engine/native.py and launch it where it belongs, or remove the "
        "definition if the kernel is dead",
    ),
    Missing.LOADED_NOT_DEFINED: (
        "the engine loads {s} (_Cuda.KERNELS) but native/cuda defines no such kernel - the fatbin load would "
        "fail on a GPU",
        "define {s} in native/cuda (or its generating macro), or drop it from _Cuda.KERNELS if it was renamed",
    ),
}

# a kernel entry point: `extern "C" __global__ void [__launch_bounds__(...)] btb_<name>(`, the name possibly on
# the next line (the MMA kernel), and NOT a macro template (a name ending at `##` is caught by _macro_kernels).
_EXTERN = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?(btb_\w+)\s*\(')
_TEMPLATE = re.compile(r"(btb_\w+)##")  # a macro body's templated name(s), e.g. btb_gemv_bf16_m##M
_DEFINE = re.compile(r"#define\s+(\w+)\(")  # a macro definition head
_INST = re.compile(r"^(\w+)\((\d+)\)", re.M)  # a macro instantiation, e.g. GEMV(16)
_KERNELS_BLOCK = re.compile(r"KERNELS\s*=\s*\((.*?)\)", re.S)


def _cuda_source() -> str:
    """every native/cuda/*.cu|*.cuh joined, with C line-continuations folded so a macro body is one logical line."""
    text = []
    for fn in sorted(os.listdir(CUDA_DIR)):
        if fn.endswith((".cu", ".cuh")):
            with open(os.path.join(CUDA_DIR, fn), encoding="utf-8") as f:
                text.append(f.read())
    return "\n".join(text).replace("\\\n", " ")


def loaded_kernels() -> set[str]:
    """the card kernels the engine loads by name, from `_Cuda.KERNELS` in native.py (parsed as source, so no
    torch import); a missing one fails at load on a GPU, so this is the authoritative required set."""
    with open(NATIVE_PY, encoding="utf-8") as f:
        block = _KERNELS_BLOCK.search(f.read())
    if block is None:
        return set()
    return set(re.findall(r'"(btb_\w+)"', block.group(1)))


def defined_kernels() -> set[str]:
    """the card kernels defined in native/cuda: the literal `extern "C" __global__` entry points plus the
    macro-generated ones (each `#define MACRO(P) ... btb_x##P ...` expanded over its `MACRO(v)` instantiations)."""
    src = _cuda_source()
    out = set(_EXTERN.findall(src))
    templates: dict[str, list[str]] = {}
    for line in src.splitlines():
        d = _DEFINE.match(line.strip())
        if d:
            templates[d.group(1)] = _TEMPLATE.findall(line)
    for macro, value in _INST.findall(src):
        for prefix in templates.get(macro, ()):
            out.add(f"{prefix}{value}")
    return out


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, kernel name); gaps() and render_missing() both derive from it."""
    loaded, defined = loaded_kernels(), defined_kernels()
    out = [(Missing.DEFINED_NOT_LOADED, k) for k in sorted(defined - loaded)]
    out += [(Missing.LOADED_NOT_DEFINED, k) for k in sorted(loaded - defined)]
    return out


def gaps() -> list[tuple[str, str]]:
    """(kernel, what) for each drift between the loaded set and the defined set."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings()]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary"""
    loaded, defined = loaded_kernels(), defined_kernels()
    return [(len(loaded & defined), len(loaded | defined), "card kernels both loaded by the engine and defined")]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no CUDA kernel gaps: every loaded kernel is defined and every defined kernel is loaded."]
    lines = [f"{len(items)} CUDA kernel gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    loaded, defined = loaded_kernels(), defined_kernels()
    print(f"CUDA card-kernel matrix: {len(loaded)} loaded, {len(defined)} defined, {len(loaded & defined)} matched")
    for k in sorted(loaded & defined):
        print(f"  ok   {k}")
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()
    if g:
        print(f"FAIL: {len(g)} CUDA kernel gaps", file=sys.stderr)
        for subject, what in g:
            print(f"  {subject}: {what}", file=sys.stderr)
    return 1 if g else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--report", action="store_true", help="print the kernel matrix and the plain-language gaps (default)"
    )
    ap.add_argument(
        "--missing", action="store_true", help="print only the gaps in plain language: what is missing and how to close"
    )
    ap.add_argument("--check", action="store_true", help="exit nonzero when loaded and defined sets disagree")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
