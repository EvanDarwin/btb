"""The cert drift guards that need the engine (torch/MLX), so they run against the built wheel in wheels.yml,
not in the torch-free cert.yml gate. Each pins a core declaration to its runtime: every declared family builds,
every capability has a Family field, and every declared quant is reachable by a live bind path. The torch-free
halves (the family set agrees, every quant is classified, SUPPORTED_QUANTS == the enum) are in test_manifest."""

from __future__ import annotations

import types

import pytest

from btb.kinds import CAPS, KIND_OF, Cap, QuantClass, quant_key, quants_of


def test_every_served_type_builds() -> None:
    """every model_type in kinds.KIND_OF actually builds a Family - a family declared in the table (so the cert
    iterates it) but missing its family() branch fails here, not at a user's load."""
    from btb.engine.families import family

    for mt in KIND_OF:
        fam = family(types.SimpleNamespace(model_type=mt, hc_count=2))
        assert fam.kind is KIND_OF[mt], f"{mt} built {fam.kind}, expected {KIND_OF[mt]}"


def test_every_family_has_a_display_name() -> None:
    """every served model_type reads out a family name (the list an unsupported load prints) - a family added to
    kinds.KIND_OF without a name fails here instead of at a KeyError in the error path."""
    from btb.engine.families import FAMILY_NAMES, NAME_OF

    assert set(NAME_OF) == set(KIND_OF), set(NAME_OF) ^ set(KIND_OF)
    assert all(NAME_OF[mt] == FAMILY_NAMES[fk] for mt, fk in KIND_OF.items())
    assert set(FAMILY_NAMES) == set(KIND_OF.values()), set(FAMILY_NAMES) ^ set(KIND_OF.values())


def test_family_flags_cover_every_cap() -> None:
    """the Flags TypedDict family() spreads has one field per Cap, and _flags() fills each from CAPS - a
    capability added to kinds.Cap without a Flags field (so family() would silently drop it), or a flag that
    disagrees with the CAPS table, fails here."""
    from btb.engine.families.base import Family, Flags, _flags

    names = {c.value for c in Cap}
    assert set(Flags.__annotations__) == names, set(Flags.__annotations__) ^ names
    assert names <= set(Family.__dataclass_fields__), names - set(Family.__dataclass_fields__)
    for kind, caps in CAPS.items():
        flags = _flags(kind)
        assert set(flags) == names, (kind, set(flags) ^ names)
        assert flags == {c.value: c in caps for c in Cap}, kind


def test_backend_lattice_registry_matches_enum() -> None:
    """kills the two-list mirror: btb/mlx/iquant._LATT (the per-kind byte layouts) carries exactly the backend
    keys the LATTICE members of kinds.Quant derive. A lattice quant added to one list and not the other fails."""
    iquant = pytest.importorskip("btb.mlx.iquant")
    derived = {quant_key(q) for q in quants_of(QuantClass.LATTICE)}
    assert set(iquant._LATT) == derived, set(iquant._LATT) ^ derived


def test_every_stored_type_has_its_kernel_on_every_host_backend() -> None:
    """each type kinds.stored_quants() names reaches its own kernel: the MLX backend's per-type table holds the
    k-quant and IQ4 types, the affine and lattice families their own tables, and the native library binds a gemv
    for every type. A type added to the declaration with a kernel missing on a backend fails here."""
    from btb.engine.native import Native
    from btb.kinds import stored_quants

    backend = pytest.importorskip("btb.mlx.backend")
    iquant = pytest.importorskip("btb.mlx.iquant")
    legacyq = pytest.importorskip("btb.mlx.legacyq")
    own = {quant_key(q) for q in quants_of(QuantClass.KQUANT) + quants_of(QuantClass.IQ4)}
    assert set(backend._MATVEC) == set(backend._DEQUANT) == own, set(backend._MATVEC) ^ own
    assert {q.value.lower() for q in quants_of(QuantClass.AFFINE)} == set(legacyq.KINDS)
    assert {quant_key(q) for q in quants_of(QuantClass.LATTICE)} == set(iquant._LATT)
    gq = Native.gemv_quant
    assert gq is not None and all(q in gq for q in stored_quants()), [q for q in stored_quants() if gq and q not in gq]
