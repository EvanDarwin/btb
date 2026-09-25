"""The native (Rust) op matrix: every fused C-ABI op crossed with the guards it should carry - a parity test
against the scalar reference, a guard-page test where it does pointer math, and a per-op benchmark. Bound by
file convention under native/tests and native/benches, so a new op without a parity test or a bench is a
reported GAP. Every gap carries a `Missing` kind whose plain-language what/how lives in `MISSING`, so `--missing`
(and the report) read as a to-do an agent can act on.

The ISA axis is per kernel family: which of the Rust `Isa` tiers (native/src/gemv.rs) each family implements,
read from the crate's `<kernel>_<tier>` naming in native/src/<family>.rs. A tier a family lacks is surfaced as
a non-blocking finding (on that CPU the family runs its next narrower path), never a silent fallback. Tiers are
certified by wheels.yml, which runs the parity and guard-page suites on every shipped platform at the tier that CPU
detects.

    python -m tests.cert.native_ops --report    # the op table plus the plain-language gaps
    python -m tests.cert.native_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.native_ops --check     # nonzero when an op lacks a parity test/bench or drifts from the crate
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import StrEnum

from btb.kinds import Quant

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NATIVE_TESTS = os.path.join(ROOT, "native", "tests")
NATIVE_BENCHES = os.path.join(ROOT, "native", "benches")


class Missing(StrEnum):
    """what a native-op gap needs, as a stable key. `MISSING` maps each to (what is absent, how to close it) in
    plain language; the taxonomy lives in one place, so a new gap kind is one member here and one MISSING row."""

    NO_PARITY = "no-parity"
    NO_GUARD = "no-guard"
    EXPORT_UNCOVERED = "export-uncovered"
    EXPORT_STALE = "export-stale"
    BENCH_UNCOVERED = "bench-uncovered"
    BENCH_MISSING = "bench-missing"
    STORED_UNDECLARED = "stored-undeclared"
    STORED_UNUSED = "stored-unused"
    TIER_UNIMPLEMENTED = "tier-unimplemented"


# kind -> (what is missing about {s}, how to close it). {s} is the subject: an op name, a btb_* export, or a file.
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.NO_PARITY: (
        "op {s} has no parity test - nothing asserts its fused/SIMD output equals the scalar reference",
        "add native/tests/<op>.rs comparing every ISA tier to the scalar oracle, and name it in that Op's `parity`",
    ),
    Missing.NO_GUARD: (
        "op {s} walks raw pointers but has no guard-page test fencing the walk",
        "add a native/tests/guard_*.rs that runs the op against an mmap'd guard page, and name it in the Op's `guard`",
    ),
    Missing.EXPORT_UNCOVERED: (
        "the crate exports {s} (native/src/lib.rs) but no Op covers it - a new kernel with no matrix row",
        "add or extend an Op whose `exports` lists {s}, giving it a parity test, a guard test, and a bench",
    ),
    Missing.EXPORT_STALE: (
        "OPS claims the export {s} but the crate no longer defines it",
        "drop {s} from that Op's `exports` (the kernel was renamed or removed)",
    ),
    Missing.BENCH_UNCOVERED: (
        "the bench file {s} exists under native/benches but no Op is timed in it - a kernel benched but uncertified",
        "add an Op whose family is {s}'s stem (or delete the bench file if the kernel is gone)",
    ),
    Missing.BENCH_MISSING: (
        "op {s} names a bench file that is not on disk",
        "create the native/benches/<family>.rs it expects, or correct the op's family",
    ),
    Missing.STORED_UNDECLARED: (
        "{s} is a Stored member that is neither a btb.kinds.Quant type nor recorded in NON_QUANT - the op table's "
        "storage column would carry a name nothing in btb defines",
        "spell {s} as its Quant member's name, or record it in NON_QUANT with what btb stores under it",
    ),
    Missing.STORED_UNUSED: (
        "{s} is a Stored member no op reads - a storage form the native crate no longer has a kernel for",
        "drop {s} (and its NON_QUANT row), or give the kernel that reads it an Op",
    ),
    Missing.TIER_UNIMPLEMENTED: (
        "kernel family {s} has no path for that ISA tier - a CPU of that tier runs the family's next narrower path",
        "add the <kernel>_<tier> variants in native/src/<family>.rs and dispatch to them where the family selects by Isa",
    ),
}


class Stored(StrEnum):
    """what a native kernel reads off the weight (or the file): a GGUF quant type spelled as its btb.kinds.Quant
    name lowercased, or one of btb's own storage forms, which `NON_QUANT` records. `stored_gaps()` holds the two
    halves together, so the op table's storage column cannot name a form btb does not have."""

    BF16 = "bf16"
    F16 = "f16"
    F32 = "f32"
    P12 = "p12"
    MXFP4 = "mxfp4"
    MXFP4_GGML = "mxfp4_ggml"
    FP8 = "fp8"
    BYTES = "bytes"


# the Stored forms that are not GGUF quant types, each with what btb stores under it. A member absent here and
# from btb.kinds.Quant is a gap: the column may not carry a name nothing defines.
NON_QUANT: dict[Stored, str] = {
    Stored.F32: "a plain fp32 tensor (an attention/delta state), never a stored quant",
    Stored.P12: "btb's 12-bit packed head store (hf.PACK12_FORMAT)",
    Stored.MXFP4_GGML: "Quant.MXFP4 in GGML's block layout, which needs its own kernel",
    Stored.FP8: "a fine-grained FP8 safetensors weight: e4m3fn bytes and an f32 block-scale grid",
    Stored.BYTES: "raw file bytes: a direct-IO read has no element type",
}


@dataclass(frozen=True)
class Op:
    name: str  # the logical op name (a report row)
    quants: tuple[Stored, ...]  # the stored dtypes/quants this op reads
    exports: tuple[str, ...]  # the btb_* C-ABI functions (native/src/lib.rs) this op covers
    parity: str | None  # the native/tests file asserting scalar-parity, or None if none exists
    guard: str | None  # the guard-page fenced test, or None when the op does no raw pointer walk
    bench: str  # the native/benches file the op is timed in (its stem is the op family / criterion section)
    pointer_math: bool = True  # whether the op walks raw pointers (so a guard test is required)


# Every logical op the matrix certifies, each claiming the crate's C-ABI exports it stands for. The claimed
# `exports` are cross-checked against native/src/lib.rs by `export_gaps()`, so a new btb_* export that no op
# covers is a reported gap - the OPS table cannot silently fall behind the crate.
OPS: tuple[Op, ...] = (
    Op(
        "gemv_bf16",
        (Stored.BF16, Stored.F16, Stored.F32),
        ("btb_gemv_bf16_rows",),
        "gemv.rs",
        "guard_gemv.rs",
        "gemv.rs",
    ),
    Op("gemv_p12", (Stored.P12,), ("btb_gemv_p12_rows",), "gemv.rs", "guard_gemv.rs", "gemv.rs"),
    Op(
        "gemv_mxfp4",
        (Stored.MXFP4, Stored.MXFP4_GGML),
        ("btb_gemv_mxfp4_rows", "btb_gemv_mxfp4_ggml_rows"),
        "gemv.rs",
        "guard_mxfp4.rs",
        "gemv.rs",
    ),
    Op("gemv_fp8", (Stored.FP8,), ("btb_gemv_fp8_rows",), "guard_scalar.rs", "guard_fp8.rs", "gemv.rs"),
    Op(
        "gemv_group",
        (Stored.BF16, Stored.MXFP4, Stored.FP8),
        ("btb_gemv_bf16_group", "btb_gemv_mxfp4_group", "btb_gemv_mxfp4_ggml_group", "btb_gemv_fp8_group"),
        "gemv_group.rs",
        "guard_gemv_group.rs",
        "gemv.rs",
    ),
    Op(
        "attn_decode",
        (Stored.BF16, Stored.F32),
        ("btb_attn_decode_bf16", "btb_attn_decode_f32"),
        "attn.rs",
        "guard_attn.rs",
        "attn.rs",
    ),
    Op("delta_step", (Stored.F32,), ("btb_delta_step",), "delta.rs", "guard_delta.rs", "delta.rs"),
    Op("sample_pick", (Stored.F32,), ("btb_sample_pick",), "sample.rs", "guard_sample.rs", "sample.rs"),
    Op(
        "read_direct",
        (Stored.BYTES,),
        ("btb_read_direct", "btb_open", "btb_close"),
        "direct.rs",
        "guard_direct.rs",
        "direct.rs",
    ),
    Op("read_at", (Stored.BYTES,), ("btb_read_at",), "handle.rs", "guard_direct.rs", "direct.rs"),
)

# crate exports that are not kernels, so no Op claims them: the tier probe the bench records its numbers under
META_EXPORTS: tuple[str, ...] = ("btb_isa",)


def op_families() -> list[str]:
    """the native op families, from the bench files on disk (native/benches/<family>.rs) - the same files
    criterion runs, so bench/report.py groups by exactly these and a new family's bench file is a new section with
    no list to edit. `bench_files_without_ops()` fails if such a file has no Op certifying it."""
    return sorted(f[:-3] for f in os.listdir(NATIVE_BENCHES) if f.endswith(".rs"))


LIB_RS = os.path.join(ROOT, "native", "src", "lib.rs")
GEMV_RS = os.path.join(ROOT, "native", "src", "gemv.rs")
_EXPORT_RE = re.compile(r'pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(btb_\w+)')
_ISA_RE = re.compile(r"pub enum Isa \{(.*?)\n\}", re.S)
_VARIANT_RE = re.compile(r"^\s{4}(\w+),?\s*$", re.M)
SCALAR = "scalar"


def isa_tiers() -> tuple[str, ...]:
    """the ISA tiers, from the Rust `pub enum Isa` in native/src/gemv.rs with each variant lowercased - the
    names `BTB_NATIVE_ISA` takes and the crate suffixes its kernels with. A new tier (SVE, AVX10) is a column
    the moment the enum grows one, with no list here to edit."""
    with open(GEMV_RS, encoding="utf-8") as f:
        body = _ISA_RE.search(f.read())
    return tuple(v.lower() for v in _VARIANT_RE.findall(body.group(1))) if body else ()


ISA_TIERS: tuple[str, ...] = isa_tiers()

_DETECT_RE = re.compile(r"fn detect_isa\(\) -> Isa \{(.*?)\n\}", re.S)
_ARCH_RE = re.compile(r'#\[cfg\(target_arch = "(\w+)"\)\]')


def tier_arch() -> dict[str, str | None]:
    """each ISA tier's CPU architecture, from the `#[cfg(target_arch = ...)]` block of `detect_isa` in
    native/src/gemv.rs that returns it (Rust's spelling: x86_64, aarch64); None for a tier no such block names,
    the portable scalar one"""
    with open(GEMV_RS, encoding="utf-8") as f:
        body = _DETECT_RE.search(f.read())
    out: dict[str, str | None] = dict.fromkeys(ISA_TIERS)
    if body is None:
        return out
    marks = list(_ARCH_RE.finditer(body.group(1)))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body.group(1))
        segment = body.group(1)[m.end() : end].split("#[cfg(not(")[0]
        for tier in re.findall(r"Isa::(\w+)", segment):
            if tier.lower() != SCALAR and out.get(tier.lower()) is None:
                out[tier.lower()] = m.group(1)
    return out


def family_tiers(family: str) -> frozenset[str]:
    """the ISA tiers native/src/<family>.rs implements: scalar (the oracle every family has) plus every tier the
    file names a `<kernel>_<tier>` for - the crate's convention for a SIMD variant (task_neon, range_avx512)."""
    with open(os.path.join(ROOT, "native", "src", f"{family}.rs"), encoding="utf-8") as f:
        src = f.read()
    named = {t for t in ISA_TIERS if t != SCALAR and re.search(rf"\b\w+_{t}\b", src)}
    return frozenset(named | {SCALAR})


def crate_exports() -> set[str]:
    """the btb_* C-ABI functions the crate actually exports, read from native/src/lib.rs - the truth the OPS
    table's `exports` claims are checked against."""
    with open(LIB_RS, encoding="utf-8") as f:
        return set(_EXPORT_RE.findall(f.read()))


def _exists(base: str, name: str | None) -> bool:
    return bool(name) and os.path.exists(os.path.join(base, name))  # type: ignore[arg-type]


def _vector_families() -> list[str]:
    """the families with an element type to vectorize: a family whose ops only read raw bytes (direct IO) has
    no SIMD path to have, so it is n/a in the tier table rather than a column of gaps."""
    return [f for f in op_families() if any(op.quants != (Stored.BYTES,) for op in OPS if op.bench == f"{f}.rs")]


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, subject). Everything the matrix reports derives from this,
    so gaps()/export_gaps()/bench_gaps() and the plain-language render never disagree."""
    out: list[tuple[Missing, str]] = []
    for op in OPS:
        if not _exists(NATIVE_TESTS, op.parity):
            out.append((Missing.NO_PARITY, op.name))
        if op.pointer_math and not _exists(NATIVE_TESTS, op.guard):
            out.append((Missing.NO_GUARD, op.name))
    claimed = {e for op in OPS for e in op.exports} | set(META_EXPORTS)
    actual = crate_exports()
    out += [(Missing.EXPORT_UNCOVERED, e) for e in sorted(actual - claimed)]
    out += [(Missing.EXPORT_STALE, e) for e in sorted(claimed - actual)]
    on_disk = {f"{fam}.rs" for fam in op_families()}
    referenced = {op.bench for op in OPS}
    out += [(Missing.BENCH_UNCOVERED, f) for f in sorted(on_disk - referenced)]
    out += [(Missing.BENCH_MISSING, op.name) for op in OPS if op.bench not in on_disk]
    quants = {q.value.lower() for q in Quant}
    read = {s for op in OPS for s in op.quants}
    out += [(Missing.STORED_UNDECLARED, s.value) for s in Stored if s not in NON_QUANT and s.value not in quants]
    out += [(Missing.STORED_UNUSED, s.value) for s in Stored if s not in read]
    for fam in _vector_families():
        have = family_tiers(fam)
        out += [(Missing.TIER_UNIMPLEMENTED, f"{fam}/{t}") for t in ISA_TIERS if t not in have]
    return out


def _what(kind: Missing, subject: str) -> str:
    return MISSING[kind][0].format(s=subject)


def export_gaps() -> list[tuple[str, str]]:
    """crate drift: a btb_* export no op claims (a new kernel with no matrix row), or an op claiming an export
    the crate no longer defines."""
    return [(s, _what(k, s)) for k, s in _findings() if k in (Missing.EXPORT_UNCOVERED, Missing.EXPORT_STALE)]


def bench_gaps() -> list[tuple[str, str]]:
    """bench-file drift: a native/benches/*.rs file no op is timed in (what would have caught the gemm
    placeholder), or an op naming a bench file that is not on disk."""
    return [(s, _what(k, s)) for k, s in _findings() if k in (Missing.BENCH_UNCOVERED, Missing.BENCH_MISSING)]


def stored_gaps() -> list[tuple[str, str]]:
    """storage-column drift: a Stored form btb itself does not define (no Quant member, no NON_QUANT row), or one
    no op reads - so the op table's quants column is a checked claim rather than decoration."""
    return [(s, _what(k, s)) for k, s in _findings() if k in (Missing.STORED_UNDECLARED, Missing.STORED_UNUSED)]


def tier_gaps() -> list[tuple[str, str]]:
    """the ISA tiers a kernel family lacks, as family/tier - work to find, not drift: a family is correct on
    every CPU through its narrower paths, so these are surfaced (--report/--missing) and never block."""
    return [(s, _what(k, s)) for k, s in _findings() if k is Missing.TIER_UNIMPLEMENTED]


def gaps() -> list[tuple[str, str]]:
    """the blocking set: a missing parity/guard/bench, or crate/bench/storage drift - what must be empty for a
    new kernel to land, and what `check()`/`test_no_native_gaps` assert. Tier gaps are deliberately not here."""
    return [(s, _what(k, s)) for k, s in _findings() if k is not Missing.TIER_UNIMPLEMENTED]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary: the ops carrying every guard they owe, and the
    ISA tiers implemented across the vector families"""
    ops = sum(
        1
        for op in OPS
        if _exists(NATIVE_TESTS, op.parity)
        and (_exists(NATIVE_TESTS, op.guard) or not op.pointer_math)
        and _exists(NATIVE_BENCHES, op.bench)
    )
    fams = _vector_families()
    tiers = sum(len(family_tiers(f)) for f in fams)
    return [
        (ops, len(OPS), "native ops with a parity test, a guard and a bench"),
        (tiers, len(fams) * len(ISA_TIERS), "ISA tiers implemented across the native kernel families"),
    ]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no native-op gaps: every op has a parity test, a guard, a bench, and matches the crate."]
    lines = [f"{len(items)} native-op gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    claimed = {e for op in OPS for e in op.exports}
    fams = op_families()
    print(
        f"native op matrix: {len(OPS)} ops in {len(fams)} families, "
        f"{len(claimed)}/{len(crate_exports())} crate exports covered"
    )
    print(f"{'op':<14} {'family':<8} {'quants':<22} {'parity':<10} {'guard':<10} {'bench':<8}")
    for fam in fams:
        for op in (o for o in OPS if o.bench == f"{fam}.rs"):
            p = "ok" if _exists(NATIVE_TESTS, op.parity) else "GAP"
            g = "ok" if _exists(NATIVE_TESTS, op.guard) else ("GAP" if op.pointer_math else "n/a")
            b = "ok" if _exists(NATIVE_BENCHES, op.bench) else "GAP"
            print(f"{op.name:<14} {fam:<8} {','.join(op.quants):<22} {p:<10} {g:<10} {b:<8}")
    print()
    vector = set(_vector_families())
    print("ISA tiers per family (a missing tier runs the next narrower path; wheels.yml certifies each shipped")
    print("platform at the tier its CPU detects - x86_64 at avx2, avx512 only on a runner that has it):")
    print(f"{'family':<8} " + " ".join(f"{t:<8}" for t in ISA_TIERS))
    for fam in fams:
        have = family_tiers(fam) if fam in vector else frozenset()
        cells = ("ok" if t in have else ("n/a" if fam not in vector else "GAP") for t in ISA_TIERS)
        print(f"{fam:<8} " + " ".join(f"{c:<8}" for c in cells))
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()  # drift only; tier gaps are surfaced by --report/--missing and never block
    if g:
        print(f"FAIL: {len(g)} native-op gaps", file=sys.stderr)
        for subject, what in g:
            print(f"  {subject}: {what}", file=sys.stderr)
    return 1 if g else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="print the op matrix and the plain-language gaps (default)")
    ap.add_argument(
        "--missing", action="store_true", help="print only the gaps in plain language: what is missing and how to close"
    )
    ap.add_argument("--check", action="store_true", help="exit nonzero on any missing parity test/bench or crate drift")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
