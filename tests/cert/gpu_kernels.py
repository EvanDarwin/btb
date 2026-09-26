"""The GPU kernels btb ships, per backend, read from the source: the Metal kernels `btb/mlx` builds with
`metal_kernel(name=...)` and the CUDA kernels `native/cuda` declares `__global__`. Names are templated per shape
and type, so each is reduced to its stem, and the stems sort into op categories by pattern; a stem no category
claims is a finding, so a new kernel cannot land outside the table. Torch-free.

    python -m tests.cert.gpu_kernels    # each category's kernels per backend
"""

from __future__ import annotations

import glob
import importlib.util
import os
import re

from btb.kinds import Json

from . import spec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUDA_SRC = os.path.join(ROOT, "native", "cuda")


def _mlx_src() -> str:
    """where `btb.mlx` is imported from: the checkout's, or the installed wheel's once CI moves the source tree
    aside (`mv btb btb.src`) so the tests run against the wheel. Found without importing it (and torch)"""
    found = importlib.util.find_spec("btb.mlx")
    if found is None or found.origin is None:
        raise ModuleNotFoundError("btb.mlx is neither in the checkout nor installed")
    return os.path.dirname(found.origin)


_METAL_NAME = re.compile(r'metal_kernel\(\s*name=f?"([^"]+)"')
_METAL_HELPER = re.compile(r'\b_kernel\(\s*"(btb_\w+)"')  # fused.py builds through one cached helper
_CUDA_NAME = re.compile(r"__global__\s+void\s+(?:__launch_bounds__\s*\([^)]*\)\s*)?(\w+?)(##\w+)?\s*\(")

# op category -> the stem pattern that claims it, first match wins
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("Matrix-vector: bf16", r"gemv_(bf16|silu|gelu|mma)|gemm16"),
    ("Matrix-vector: 12-bit", r"p12"),
    ("Matrix-vector: MXFP4", r"mxfp4"),
    ("Matrix-vector: GGUF quants", r"<kind>|q\dk|iq4nl"),
    ("Attention", r"attn"),
    ("Norm and RoPE", r"norm|rope|sandwich"),
    ("Activation", r"silu_mul|gelu_mul"),
    ("DeltaNet", r"delta"),
    ("Sampling", r"sample"),
    ("KV cache", r"kv_store"),
    ("Whole-pass kernels", r"mega|publish"),
)


def _stem(name: str) -> str:
    """a templated kernel name without its shape and type suffixes: `btb_{kind}_mv_{rows}` -> `btb_<kind>_mv`"""
    name = name.replace("{kind}", "<kind>")
    head, templated, _ = name.partition("{")
    # a one-letter prefix before the placeholder names the value (`_x{xb}`, `_g{g}`), not the kernel
    return (re.sub(r"_[a-z]$", "", head) if templated else head).rstrip("_")


def mlx_kernels() -> set[str]:
    out: set[str] = set()
    for fn in glob.glob(os.path.join(_mlx_src(), "*.py")):
        with open(fn, encoding="utf-8") as f:
            src = f.read()
        out |= {_stem(n) for n in _METAL_NAME.findall(src) + _METAL_HELPER.findall(src)}
    return out


def cuda_kernels() -> set[str]:
    """the `__global__` names; a macro-stamped one (`btb_gemv_bf16_m##M`) loses the stamped letter"""
    out: set[str] = set()
    for fn in glob.glob(os.path.join(CUDA_SRC, "*.cu*")):
        with open(fn, encoding="utf-8") as f:
            for name, stamped in _CUDA_NAME.findall(f.read()):
                out.add(re.sub(r"_[A-Za-z]$", "", name) if stamped else name)
    return out


KERNELS = {spec.Hardware.MLX: mlx_kernels, spec.Hardware.CUDA: cuda_kernels}


def category(stem: str) -> str | None:
    return next((c for c, pat in CATEGORIES if re.search(pat, stem)), None)


def orphans() -> list[str]:
    """kernel stems no category claims"""
    return sorted(s for read in KERNELS.values() for s in read() if category(s) is None)


def table() -> Json:
    """per GPU backend (every non-CPU hardware), each category's kernel stems: a list, empty where the backend
    runs the framework's own ops, None where btb has no backend at all"""
    backends = [h for h in spec.Hardware if h is not spec.Hardware.CPU]
    stems = {h: KERNELS[h]() if h in KERNELS else set() for h in backends}
    return {
        "backends": [h.value for h in backends],
        "rows": [
            {
                "category": c,
                "kernels": {
                    h.value: None if h in spec.NO_BACKEND else sorted(s for s in stems[h] if category(s) == c)
                    for h in backends
                },
            }
            for c, _ in CATEGORIES
        ],
    }


def main() -> int:
    for row in table()["rows"]:
        print(row["category"])
        for backend, ks in row["kernels"].items():
            print(f"  {backend:5} {'no backend' if ks is None else ', '.join(ks) or '-'}")
    for s in orphans():
        print(f"ORPHAN {s}: no category claims it")
    return 1 if orphans() else 0


if __name__ == "__main__":
    raise SystemExit(main())
