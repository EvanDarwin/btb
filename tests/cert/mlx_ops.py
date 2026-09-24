"""The MLX Metal-kernel matrix. The Metal kernels compile and run only on Apple hardware (CI has none), so this
certifies the kernel surface structurally, source-parsed and torch/mlx-free: every launcher function
(`matvec_*`/`dequant_*`/`repack_*`/`weight_*`) in btb/mlx and every `btb_*` Metal-kernel name a module compiles
must be claimed by a family in `OPS`, and every family must name a launcher/kernel that still exists. A new
kernel with no matrix row - or a row for a launcher/kernel that is gone - is a reported gap, so a Metal kernel
cannot land uncertified.

Parsed, not imported: the kernels come from `btb.mlx.*`, which imports `mlx.core`, so reflecting the live
modules would gate this whole matrix behind an MLX runtime. cuda_ops parses .cu source for the same reason and
runs in the torch-free cert.yml gate; this follows that precedent and stays import-light. The kernel names are
Python f-strings keyed by module: the leading `{kind}` of a `name=f"btb_{kind}_mv_{rows}..."` is EXPANDED over the
finite set the module dispatches on (the keys of the dicts that `kind` indexes), and only the remaining
interpolations fold to `*`. Folding the kind too would make one row pre-claim every kernel the module can ever
name, so a new quant's kernels could never be reported - hence a row whose kind segment is `*` is itself a gap.

    python -m tests.cert.mlx_ops --report    # the kernel matrix and the plain-language gaps
    python -m tests.cert.mlx_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.mlx_ops --check      # nonzero on registry/source drift OR a family with no parity test
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum

from .core import BTB_SRC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MLX_DIR = os.path.join(BTB_SRC, "mlx")
TESTS = os.path.join(ROOT, "tests")


class Missing(StrEnum):
    """what an MLX-kernel gap needs, as a stable key; `MISSING` maps each to (what is absent, how to close it).
    A family with no parity test is a gap that gates, same as native_ops - a kernel exercised only indirectly is
    not certified."""

    FUNC_UNCOVERED = "func-uncovered"
    FUNC_STALE = "func-stale"
    KERNEL_UNCOVERED = "kernel-uncovered"
    KERNEL_STALE = "kernel-stale"
    KERNEL_WILDCARD = "kernel-wildcard"
    NO_PARITY = "no-parity"
    PARITY_MISSING = "parity-missing"


# kind -> (what is missing about {s}, how to close it). {s} is the subject: module.launcher, module:kernel, or family.
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.FUNC_UNCOVERED: (
        "{s} is a launcher (matvec_/dequant_/repack_/weight_) in btb/mlx that no family in OPS covers - a new "
        "kernel launcher with no matrix row",
        "add or extend an Op whose `funcs` lists it, pointing at a parity test under tests/kernels",
    ),
    Missing.FUNC_STALE: (
        "OPS claims the launcher {s} but btb/mlx no longer defines it",
        "drop it from that family's `funcs` (renamed or removed)",
    ),
    Missing.KERNEL_UNCOVERED: (
        "{s} is a btb_* Metal kernel a btb/mlx module compiles that no family covers - a new kernel with no matrix row",
        "add or extend an Op whose `kernels` lists its template, with a parity test",
    ),
    Missing.KERNEL_STALE: (
        "OPS claims the Metal kernel {s} but no btb/mlx module compiles it",
        "drop it from that family's `kernels` (renamed or removed)",
    ),
    Missing.KERNEL_WILDCARD: (
        "OPS claims the template {s}, whose kind segment is a wildcard - it pre-claims every kernel the module "
        "can ever name, so a new quant's kernels could never be reported as uncovered",
        "replace {s} with one entry per kind the module dispatches on, each naming its kind",
    ),
    Missing.NO_PARITY: (
        "family {s} has no parity test - nothing asserts its kernels match a reference (it runs only indirectly, "
        "e.g. through the engine cache/drafter), so it is uncertified",
        "add a parity test under tests/kernels comparing the family's kernels to a reference, and name it in the "
        "Op's `parity`",
    ),
    Missing.PARITY_MISSING: (
        "family {s} names a parity file that is not on disk",
        "create the named tests/kernels file, or correct the Op's `parity`",
    ),
}


@dataclass(frozen=True)
class Op:
    name: str  # the logical kernel family (a report row)
    module: str  # the btb/mlx/<module>.py that defines it
    kernels: tuple[str, ...]  # the btb_* Metal-kernel name templates ({...} -> *) it compiles
    parity: str | None  # the tests/<path> asserting parity against a reference, or None if none exists
    funcs: tuple[str, ...] = ()  # the matvec_/dequant_/repack_/weight_ launchers it covers (quant families only)
    note: str = field(default="")


# Every MLX kernel family the matrix certifies, each claiming the launcher functions and Metal-kernel name
# templates it stands for. The claims are cross-checked against btb/mlx by `func_gaps()`/`kernel_gaps()`, so a
# new launcher or a new `btb_*` kernel that no family covers is a reported gap - the table cannot fall behind.
OPS: tuple[Op, ...] = (
    Op("rope", "attn", ("btb_rope_rows", "btb_rope_rows2"), "kernels/test_mlx.py"),
    Op("attn_decode", "attn", ("btb_attn_nodes_**_d*_g*_n*", "btb_attn_fold"), "kernels/test_mlx.py"),
    Op("attn_prefill", "attn", ("btb_attn_prefill_d256_g*",), "kernels/test_mlx.py"),
    Op("delta", "delta", ("btb_delta_recurrent*", "btb_delta_tree"), "kernels/test_mlx.py"),
    Op(
        "fused_dense",
        "fused",
        ("btb_add_rmsnorm", "btb_sandwich_add", "btb_qk_norm_rope", "btb_silu_mul"),
        "kernels/test_mlx.py",
    ),
    Op("p12_unpack", "gemv", ("btb_unpack_p12",), "kernels/test_mlx.py"),
    Op("gemm_bf16", "gemv", ("btb_gemm16_x*",), "kernels/test_mlx.py"),
    Op("gemv_mxfp4", "gemv", ("btb_gemv_mxfp4**_x*", "btb_mxfp4_dequant"), "kernels/test_mxfp4.py"),
    Op(
        "kquant_q45",
        "kquant",
        ("btb_q4k_mv_*_*_*", "btb_q4k_dequant", "btb_q5k_mv_*_*_*", "btb_q5k_dequant"),
        "kernels/test_gguf.py",
        funcs=("matvec_q4k", "dequant_q4k", "matvec_q5k", "dequant_q5k"),
    ),
    Op(
        "kquant_q2k",
        "kquant",
        ("btb_q2k_mv_*_*_*", "btb_q2k_dequant"),
        "kernels/test_gguf.py",
        funcs=("matvec_q2k", "dequant_q2k"),
    ),
    Op(
        "kquant_q3k",
        "kquant",
        ("btb_q3k_mv_*_*_*", "btb_q3k_dequant"),
        "kernels/test_gguf.py",
        funcs=("matvec_q3k", "dequant_q3k"),
    ),
    Op(
        "q6k",
        "q6k",
        ("btb_q6k_mv_*_*_*", "btb_q6k_gather_*", "btb_q6k_dequant"),
        "kernels/test_gguf.py",
        funcs=("matvec_q6k", "dequant_q6k"),
    ),
    Op(
        "iq4nl",
        "iquant",
        ("btb_iq4nl_mv_*_*_*", "btb_iq4nl_dequant"),
        "kernels/test_gguf.py",
        funcs=("repack_iq4nl", "matvec_iq4nl", "dequant_iq4nl"),
    ),
    Op(
        "iq4xs",
        "iquant",
        ("btb_iq4xs_mv_*_*_*", "btb_iq4xs_dequant"),
        "kernels/test_gguf.py",
        funcs=("matvec_iq4xs", "dequant_iq4xs"),
    ),
    Op(
        "iquant_lattice",
        "iquant",
        (
            "btb_iq1m_mv_*_*_*",
            "btb_iq1m_dequant",
            "btb_iq1s_mv_*_*_*",
            "btb_iq1s_dequant",
            "btb_iq2s_mv_*_*_*",
            "btb_iq2s_dequant",
            "btb_iq2xs_mv_*_*_*",
            "btb_iq2xs_dequant",
            "btb_iq2xxs_mv_*_*_*",
            "btb_iq2xxs_dequant",
            "btb_iq3s_mv_*_*_*",
            "btb_iq3s_dequant",
            "btb_iq3xxs_mv_*_*_*",
            "btb_iq3xxs_dequant",
        ),
        "kernels/test_gguf.py",
        funcs=("repack_lattice", "matvec_lattice", "dequant_lattice"),
    ),
    Op("kv", "kv", ("btb_kv_store",), "kernels/test_kv.py"),
    Op("mega", "mega", ("btb_mega_*_*_*_*",), "kernels/test_mega.py"),
    Op("sample", "sample", ("btb_sample_pick", "btb_sample_verify"), "kernels/test_sampling.py"),
    Op(
        "binder",
        "backend",
        (),
        "kernels/test_gguf.py",
        funcs=(
            "weight_slot",
            "weight_fused",
            "weight_slot_packed",
            "weight_affine",
            "weight_q6k",
            "weight_q5k",
            "weight_q2k",
            "weight_q3k",
            "weight_iq4nl",
            "weight_iq4xs",
            "weight_lattice",
            "weight_q4k",
        ),
        note="binder methods build Weight objects; the matvec/dequant families carry the kernel parity",
    ),
)


_FUNC_RE = re.compile(r"^\s*def\s+((?:matvec|dequant|repack|weight)_\w+)\s*\(", re.M)
_KERNEL_RE = re.compile(r'(?:f)?"(btb_[^"]*)"')  # every btb_* string literal in btb/mlx is a Metal kernel name
_BRACE_RE = re.compile(r"\{[^{}]*\}")  # an f-string interpolation; folded to `*` to make a template
_LEAD_RE = re.compile(r"^btb_\{(\w+)\}")  # the kind segment of a name, the one interpolation that is expanded
_WILDCARD_RE = re.compile(r"^btb_\*")  # a template that names no kind: it would claim every kernel of its module


def _mlx_modules() -> list[tuple[str, str]]:
    """(module stem, source text) for each btb/mlx/*.py, so a new module is parsed with no list to edit."""
    out = []
    for fn in sorted(os.listdir(MLX_DIR)):
        if fn.endswith(".py") and fn != "__init__.py":
            with open(os.path.join(MLX_DIR, fn), encoding="utf-8") as f:
                out.append((fn[:-3], f.read()))
    return out


def source_funcs() -> set[tuple[str, str]]:
    """(module, launcher) for every matvec_/dequant_/repack_/weight_ function defined in btb/mlx - the launcher
    axis, exact names, the authoritative set each family's `funcs` is checked against."""
    return {(mod, fn) for mod, src in _mlx_modules() for fn in _FUNC_RE.findall(src)}


def dispatch_kinds(src: str) -> dict[str, set[str]]:
    """{variable -> the string keys of the module dicts that variable indexes}: `_KQ[kind]` in kquant.py answers
    {"kind": {"q4k", "q5k"}}. That is the finite set a `btb_{kind}_...` name can expand over, read from the table
    the module itself dispatches on rather than a list here."""
    tree = ast.parse(src)
    tables: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):  # an annotated table (`_LATT: dict[...] = {...}`) counts
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not isinstance(node.value, ast.Dict) or len(targets) != 1 or not isinstance(targets[0], ast.Name):
                continue
            tables[targets[0].id] = {
                k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    out: dict[str, set[str]] = {}
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and isinstance(sub.slice, ast.Name):
            out.setdefault(sub.slice.id, set()).update(tables.get(sub.value.id, set()))
    return out


def expand(name: str, kinds: dict[str, set[str]]) -> list[str]:
    """the kernel-name templates one f-string stands for: its leading `{kind}` expanded over `kinds`, every other
    interpolation folded to `*`. An unresolvable kind stays `*`, which no row may claim - a loud gap, not a
    silent blanket claim."""
    m = _LEAD_RE.match(name)
    found = sorted(kinds.get(m.group(1), ())) if m is not None else []
    names = [f"btb_{k}{name[m.end() :]}" for k in found] if m is not None and found else [name]
    return [_BRACE_RE.sub("*", n) for n in names]


def source_kernels() -> set[tuple[str, str]]:
    """(module, template) for every btb_* Metal-kernel name a module compiles, each f-string expanded over its
    kind table and folded elsewhere; the entry-point axis a family's `kernels` is checked against."""
    out: set[tuple[str, str]] = set()
    for mod, src in _mlx_modules():
        kinds = dispatch_kinds(src)
        for raw in _KERNEL_RE.findall(src):
            out |= {(mod, t) for t in expand(raw, kinds)}
    return out


def _exists(name: str | None) -> bool:
    return bool(name) and os.path.exists(os.path.join(TESTS, name))  # type: ignore[arg-type]


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, subject); everything the matrix reports derives from it. A
    family with no parity test (or one naming an absent file) is a gap here, so it gates - not a soft note."""
    out: list[tuple[Missing, str]] = []
    claimed_f = {(op.module, fn) for op in OPS for fn in op.funcs}
    actual_f = source_funcs()
    out += [(Missing.FUNC_UNCOVERED, f"{m}.{fn}") for m, fn in sorted(actual_f - claimed_f)]
    out += [(Missing.FUNC_STALE, f"{m}.{fn}") for m, fn in sorted(claimed_f - actual_f)]
    claimed_k = {(op.module, k) for op in OPS for k in op.kernels}
    actual_k = source_kernels()
    out += [(Missing.KERNEL_UNCOVERED, f"{m}:{k}") for m, k in sorted(actual_k - claimed_k)]
    out += [(Missing.KERNEL_STALE, f"{m}:{k}") for m, k in sorted(claimed_k - actual_k)]
    out += [(Missing.KERNEL_WILDCARD, f"{m}:{k}") for m, k in sorted(claimed_k) if _WILDCARD_RE.match(k)]
    for op in OPS:
        if op.parity is None:
            out.append((Missing.NO_PARITY, op.name))
        elif not _exists(op.parity):
            out.append((Missing.PARITY_MISSING, f"{op.name} ({op.parity})"))
    return out


def func_gaps() -> list[tuple[str, str]]:
    """launcher-axis drift: a matvec_/dequant_/repack_/weight_ function no family claims, or a stale claim."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings() if k in (Missing.FUNC_UNCOVERED, Missing.FUNC_STALE)]


_KERNEL_AXIS = (Missing.KERNEL_UNCOVERED, Missing.KERNEL_STALE, Missing.KERNEL_WILDCARD)


def kernel_gaps() -> list[tuple[str, str]]:
    """entry-point-axis drift: a btb_* Metal kernel no family claims, a family naming one no module compiles, or
    a claim so wide it names no kind."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings() if k in _KERNEL_AXIS]


def gaps() -> list[tuple[str, str]]:
    """(subject, what) for every gap: launcher/kernel drift, or a family with no (or an absent) parity test. A
    new MLX kernel cannot land, and a kernel cannot go uncertified, without a gap here."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings()]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary"""
    tested = sum(1 for op in OPS if op.parity is not None and _exists(op.parity))
    return [(tested, len(OPS), "MLX kernel families with a parity test")]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no MLX-kernel gaps: every launcher and kernel is covered, and every family has a parity test."]
    lines = [f"{len(items)} MLX-kernel gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    funcs, kerns = source_funcs(), source_kernels()
    cf = {(op.module, fn) for op in OPS for fn in op.funcs}
    ck = {(op.module, k) for op in OPS for k in op.kernels}
    print(
        f"MLX Metal-kernel matrix: {len(OPS)} families, {len(cf & funcs)}/{len(funcs)} launchers and "
        f"{len(ck & kerns)}/{len(kerns)} kernels covered"
    )
    print(f"{'family':<16} {'module':<9} {'kernels':<3} {'funcs':<3} {'parity':<22}")
    for op in OPS:
        p = op.parity if _exists(op.parity) else ("(NONE - gap)" if op.parity is None else f"MISSING {op.parity}")
        print(f"{op.name:<16} {op.module:<9} {len(op.kernels):<3} {len(op.funcs):<3} {p:<22}")
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()
    if g:
        print(f"FAIL: {len(g)} MLX-kernel gaps", file=sys.stderr)
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
    ap.add_argument(
        "--check", action="store_true", help="exit nonzero on registry/source drift or a family with no parity test"
    )
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
