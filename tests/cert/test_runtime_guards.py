"""The cert drift guards that need the engine (torch/MLX), so they run against the built wheel in wheels.yml,
not in the torch-free cert.yml gate. Each pins a core declaration to its runtime: every declared family builds,
every capability has a Family field, and every declared quant is reachable by a live bind path. The torch-free
halves (the family set agrees, every quant is classified, SUPPORTED_QUANTS == the enum) are in test_manifest."""

from __future__ import annotations

import types

import pytest

from btb.hf import AFFINE_TYPES
from btb.kinds import CAPS, KIND_OF, Cap, QuantClass, latt_backend_key, quants_of


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
    from btb.engine.families import Family, Flags, _flags

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
    derived = {latt_backend_key(q) for q in quants_of(QuantClass.LATTICE)}
    assert set(iquant._LATT) == derived, set(iquant._LATT) ^ derived


def test_engine_lattice_map_matches_enum() -> None:
    """the engine's GGUF-name -> backend-key map is exactly the LATTICE members (it is derived, so this pins
    that the derivation and the backend agree on the same kinds)."""
    mf = pytest.importorskip("btb.engine.mlx_forward")
    assert mf._MlxMixin._LATT_KINDS == {q.value: latt_backend_key(q) for q in quants_of(QuantClass.LATTICE)}


def test_affine_types_are_enum_members() -> None:
    """AFFINE_TYPES (name -> bits) is keyed by Quant members - the affine decoder's set cannot name a type the
    enum does not know, and every AFFINE-class quant has a bits entry."""
    from btb.kinds import Quant

    assert all(k in set(Quant) for k in AFFINE_TYPES), [k for k in AFFINE_TYPES if k not in set(Quant)]
    assert set(quants_of(QuantClass.AFFINE)) <= set(AFFINE_TYPES), "an AFFINE quant with no bits entry"
