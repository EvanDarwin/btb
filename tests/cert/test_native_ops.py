"""The native op matrix as a suite gate: every op carries a parity test, a bench, and a guard-page test where
it walks pointers, and every btb_* export in native/src/lib.rs is covered by an op."""

from __future__ import annotations

from . import native_ops


def test_matrix_binds_to_real_files() -> None:
    """Every op that claims a parity/guard/bench file must point at one that exists (a typo in the table is a
    silent hole); a declared-None is a real gap, caught by test_no_native_gaps."""
    import os

    for op in native_ops.OPS:
        for base, name in (
            (native_ops.NATIVE_TESTS, op.parity),
            (native_ops.NATIVE_TESTS, op.guard),
            (native_ops.NATIVE_BENCHES, op.bench),
        ):
            if name is not None:
                assert os.path.exists(os.path.join(base, name)), f"{op.name} names a missing file {name!r}"


def test_every_crate_export_is_covered() -> None:
    """the drift guard: every btb_* C-ABI export in native/src/lib.rs is claimed by some op, and no op claims a
    stale one. A new native kernel exposed to Python with no matrix row fails here."""
    assert native_ops.export_gaps() == [], native_ops.export_gaps()


def test_meta_exports_exist_and_are_not_ops() -> None:
    """a META export is in the crate (else it is a stale claim) and no Op also claims it"""
    actual = native_ops.crate_exports()
    op_claimed = {e for op in native_ops.OPS for e in op.exports}
    for e in native_ops.META_EXPORTS:
        assert e in actual, e
        assert e not in op_claimed, e


def test_every_bench_file_is_covered() -> None:
    """the other drift guard: every native/benches/*.rs file is a family some op is certified in (a kernel
    benched but not in the matrix - the gemm placeholder was exactly this), and no op names a missing file."""
    assert native_ops.bench_gaps() == [], native_ops.bench_gaps()


def test_no_native_gaps() -> None:
    """Every op carries a parity test, a bench, and a guard-page test where it walks pointers - and no crate
    export is uncovered. Tier gaps are not in gaps(): see test_tier_gaps_are_surfaced_not_blocking."""
    assert native_ops.gaps() == [], native_ops.gaps()


def test_isa_tiers_come_from_the_rust_enum() -> None:
    """the tiers are the `pub enum Isa` variants lowercased, read from native/src/gemv.rs, cross-checked against
    the variants the crate constructs - a tier declared and never returned, or a stale parse, fails here."""
    import re

    with open(native_ops.GEMV_RS, encoding="utf-8") as f:
        constructed = {m.lower() for m in re.findall(r"Isa::(\w+)", f.read())}
    assert native_ops.ISA_TIERS == native_ops.isa_tiers()
    assert set(native_ops.ISA_TIERS) == constructed, (native_ops.ISA_TIERS, constructed)
    assert native_ops.SCALAR in native_ops.ISA_TIERS


def test_family_tiers_match_the_dispatch_sites() -> None:
    """the family-level tier claim holds per op: every `by_isa!` site in gemv.rs names a distinct kernel per
    tier (a site passing the scalar task for neon would be a fallback the table calls 'ok'), and a tier the
    table lists for a family is one its file carries a `#[target_feature]` or aarch64 kernel for."""
    import re

    with open(native_ops.GEMV_RS, encoding="utf-8") as f:
        src = f.read()
    for site in re.findall(r"by_isa!\(\s*isa,\s*([^()]*?)\(", src, re.S):
        idents = [i.strip() for i in site.split(",") if i.strip()]
        assert len(idents) == len(native_ops.ISA_TIERS), idents
        assert len(set(idents)) == len(idents), f"by_isa! site reuses a kernel across tiers: {idents}"
    for fam in native_ops._vector_families():
        with open(f"{native_ops.ROOT}/native/src/{fam}.rs", encoding="utf-8") as f:
            body = f.read()
        have = native_ops.family_tiers(fam)
        assert ("avx2" in have) == ('target_feature(enable = "avx2' in body), fam
        assert ("avx512" in have) == ('target_feature(enable = "avx512' in body), fam
        assert ("neon" in have) == ('cfg(target_arch = "aarch64")' in body), fam


def test_tier_gaps_are_surfaced_not_blocking() -> None:
    """a tier a family lacks is a finding the report and --missing show (work to find), and never in the
    blocking set: a family is correct on every CPU through its narrower paths."""
    surfaced = {s for k, s in native_ops._findings() if k is native_ops.Missing.TIER_UNIMPLEMENTED}
    assert surfaced == {s for s, _what in native_ops.tier_gaps()}
    assert surfaced.isdisjoint(s for s, _what in native_ops.gaps())
    for subject in surfaced:
        fam, tier = subject.split("/")
        assert fam in native_ops._vector_families() and tier in native_ops.ISA_TIERS, subject
        assert tier not in native_ops.family_tiers(fam)


def test_every_storage_form_is_declared_and_read() -> None:
    """the op table's quants column is checked, not decorative: each Stored form is a btb.kinds.Quant type or a
    NON_QUANT row saying what btb stores under it, and each is read by some op."""
    assert native_ops.stored_gaps() == [], native_ops.stored_gaps()
    assert {s for op in native_ops.OPS for s in op.quants} == set(native_ops.Stored)
