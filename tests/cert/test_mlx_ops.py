"""The MLX Metal-kernel matrix as a suite gate. The Metal kernels compile only on Apple hardware, so this is the
structural half: the launchers and kernels btb/mlx defines and the ones the registry claims must be the same
set, and every family must have a parity test. A new MLX kernel the registry does not cover, a stale row, or a
family with no (or an absent) parity test fails here."""

from __future__ import annotations

import pytest

from . import mlx_ops


@pytest.mark.cert_gap
def test_no_gaps() -> None:
    """the full gate: registry/source drift AND a family with no parity test both fail here - a kernel covered
    only indirectly (e.g. kv, run through the engine cache) is uncertified and gates until it gets a parity test."""
    assert mlx_ops.gaps() == [], mlx_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a parse that silently found nothing would make the gate vacuous."""
    assert mlx_ops.source_funcs(), "parsed no launcher functions from btb/mlx"
    assert mlx_ops.source_kernels(), "parsed no btb_* Metal kernels from btb/mlx"


def test_a_kind_is_expanded_not_wildcarded() -> None:
    """a `btb_{kind}_mv_...` name becomes one template per kind the module dispatches on, so every kernel is a
    distinct claim; only the remaining interpolations fold to `*`."""
    kinds = {"kind": {"q4k", "q5k"}}
    assert mlx_ops.expand("btb_{kind}_mv_{rows}_{nsb}_{tr}", kinds) == ["btb_q4k_mv_*_*_*", "btb_q5k_mv_*_*_*"]
    assert mlx_ops.expand("btb_rope_rows2", kinds) == ["btb_rope_rows2"]
    assert mlx_ops.expand("btb_{unknown}_mv_{rows}", kinds) == ["btb_*_mv_*"]  # unresolvable: a loud gap, not a claim
    assert not [k for _m, k in mlx_ops.source_kernels() if k.startswith("btb_*")]


def test_a_wildcard_kind_in_a_row_is_a_gap() -> None:
    """a row claiming `btb_*_mv_*_*_*` would pre-claim every kernel its module can ever name, so KERNEL_UNCOVERED
    could never fire for that module again - the matrix reports the claim itself."""
    wide = mlx_ops.Op("wide", "kquant", ("btb_*_mv_*_*_*",), "kernels/test_gguf.py")
    assert not [k for k in wide.kernels if not mlx_ops._WILDCARD_RE.match(k)]
    assert all(not mlx_ops._WILDCARD_RE.match(k) for op in mlx_ops.OPS for k in op.kernels), "a row names no kind"


def test_the_lattice_kinds_are_the_supported_quants() -> None:
    """the kind table btb/mlx/iquant.py dispatches on and btb.kinds' LATTICE quants are the same set: a lattice
    quant declared in kinds with no kernel (or the reverse) is drift between the two."""
    from btb.kinds import QuantClass, latt_backend_key, quants_of

    src = dict(mlx_ops._mlx_modules())["iquant"]
    parsed = mlx_ops.dispatch_kinds(src)["kind"]
    assert {latt_backend_key(q) for q in quants_of(QuantClass.LATTICE)} <= parsed
