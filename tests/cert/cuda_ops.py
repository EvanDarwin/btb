"""The CUDA card-kernel matrix. The card runs only where a GPU is present (CI has none), so this certifies the
kernel surface structurally, source-parsed and torch-free. The kernels' Python bindings are written from native/cuda
(`python build.py bindings`: each parameter's role a typedef of native/cuda/btb_roles.cuh, a NewType in btb/kinds.py,
and a typed launcher a kernel in btb/engine/card_launch.py), and mypy holds every launch to its kernel's parameters -
their count, their order and their roles. What the types cannot see is held here: the checked-in bindings are what
native/cuda's signatures write now (a signature edited without them is a binding the type check would trust while the
kernel reads something else), nothing in btb/ launches a kernel but through its launcher (a `.launch(...)` anywhere
else hands the driver a list no type checked), and every kernel's launcher is reached from btb/ (a kernel nothing
launches any more is a dead path, compiled and loaded for nothing).

    python -m tests.cert.cuda_ops --report    # the kernel matrix and the plain-language gaps
    python -m tests.cert.cuda_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.cuda_ops --check      # nonzero on any gap
"""

from __future__ import annotations

import argparse
import ast
import functools
import glob
import importlib.util
import os
import sys
from enum import StrEnum
from types import ModuleType

from .core import BTB_SRC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAUNCH_PY = "card_launch.py"  # btb/engine's generated launchers: the one module that calls `launch`


class Missing(StrEnum):
    """what a CUDA-kernel gap needs, as a stable key; `MISSING` maps each to (what is absent, how to close it)."""

    BINDING_RULE = "binding-rule"
    BINDINGS_STALE = "bindings-stale"
    RAW_LAUNCH = "raw-launch"
    UNLAUNCHED = "unlaunched"


# kind -> (what is missing about {s}, how to close it).
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.BINDING_RULE: (
        "native/cuda breaks a rule the bindings rest on: {s}",
        "give each parameter of the kernel a role of native/cuda/btb_roles.cuh as the message says, then run "
        "`python build.py bindings`",
    ),
    Missing.BINDINGS_STALE: (
        "{s} is not what native/cuda's signatures write now - a kernel's parameters or a role changed without the "
        "bindings, so the type check holds launches to parameters the kernel no longer has",
        "run `python build.py bindings` and check the launches mypy then names",
    ),
    Missing.RAW_LAUNCH: (
        "the `.launch(...)` at {s} hands the driver an argument list no type checked",
        "launch the kernel through its method (`k.btb_<kernel>(grid, block, ...)`, btb/engine/card_launch.py)",
    ),
    Missing.UNLAUNCHED: (
        "nothing in btb/ reaches {s}'s launcher - a dead kernel path, compiled and loaded for nothing",
        "launch {s} where it belongs, or remove its definition from native/cuda (and the tests and benches that "
        "drive it alone), then run `python build.py bindings`",
    ),
}


@functools.cache
def bindings() -> ModuleType:
    """build.py, the bindings' writer, loaded from the checkout by path (standard library only; never the PyPA
    `build` package a plain import could find)"""
    spec = importlib.util.spec_from_file_location("btb_build", os.path.join(ROOT, "build.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["btb_build"] = mod  # the module's dataclasses look themselves up here as they are made
    spec.loader.exec_module(mod)
    return mod


def launchers() -> dict[str, list[str]]:
    """each launcher's name (`btb_attn_flash`) and the kernels it launches (`btb_attn_flash_d64`, ..)"""
    return {k.method: k.names() for k in bindings().card_kernels()}


def defined_kernels() -> set[str]:
    """the card kernels native/cuda defines, a macro's at each of its instantiations"""
    return {n for names in launchers().values() for n in names}


def _sources(root: str) -> list[tuple[str, ast.Module]]:
    """every module of the package under `root` but the launchers themselves (btb/mlx's Metal kernels share names with
    the card's, not launches)"""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if os.path.join(root, "mlx") + os.sep in path or os.path.basename(path) == LAUNCH_PY:
            continue
        with open(path, encoding="utf-8") as f:
            out.append((path, ast.parse(f.read(), filename=path)))
    return out


def _where(path: str, line: int) -> str:
    """a site's `file:line`: a module of the package (`_sources`, absolute) by its path from the package's parent -
    `btb/engine/cuda.py` from the checkout and the installed wheel alike - and one handed by name (a test's own module)
    as named"""
    if not os.path.isabs(path):
        return f"{path}:{line}"
    return f"{os.path.relpath(path, os.path.dirname(BTB_SRC))}:{line}"


def raw_launches(sources: list[tuple[str, ast.Module]] | None = None) -> list[str]:
    """every `.launch(...)` call in btb/ outside the launchers, as `file:line`"""
    sources = _sources(BTB_SRC) if sources is None else sources
    return [
        _where(path, node.lineno)
        for path, tree in sources
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch"
    ]


def reached_launchers(sources: list[tuple[str, ast.Module]] | None = None) -> set[str]:
    """the launchers btb/ reaches: each named as a function (`btb_silu_mul(k, ...)`) or as the card module's method
    (`k.btb_silu_mul(...)`), called there or picked and called later (`glu = k.btb_gemv_mma_glu_gelu if gelu else
    k.btb_gemv_mma_glu_silu`)"""
    sources = _sources(BTB_SRC) if sources is None else sources
    names = set(launchers())
    out: set[str] = set()
    for _, tree in sources:
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
            if name in names:
                out.add(name)
    return out


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, subject); gaps() and render_missing() both derive from it."""
    B = bindings()
    try:
        names = launchers()
        stale = B.stale_bindings(BTB_SRC)  # the package under test: the wheel certify moves the checkout's aside
    except B.BindingError as e:
        return [(Missing.BINDING_RULE, str(e))]
    sources = _sources(BTB_SRC)
    reached = reached_launchers(sources)
    out = [(Missing.BINDINGS_STALE, f) for f in stale]
    out += [(Missing.RAW_LAUNCH, s) for s in raw_launches(sources)]
    out += [(Missing.UNLAUNCHED, m) for m in sorted(names) if m not in reached]
    return out


def gaps() -> list[tuple[str, str]]:
    """(subject, what) for each gap."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings()]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary"""
    names = launchers()
    reached = reached_launchers()
    return [(len(set(names) & reached), len(names), "card kernels launched from btb/ through their typed launchers")]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no CUDA kernel gaps: the bindings are current and every kernel is launched through its launcher."]
    lines = [f"{len(items)} CUDA kernel gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    names = launchers()
    reached = reached_launchers()
    kernels = sum(len(v) for v in names.values())
    print(f"CUDA card-kernel matrix: {kernels} kernels, {len(names)} launchers, {len(set(names) & reached)} reached")
    for m in sorted(set(names) & reached):
        print(f"  ok   {m}: {', '.join(names[m])}")
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
    ap.add_argument("--check", action="store_true", help="exit nonzero on any gap")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
