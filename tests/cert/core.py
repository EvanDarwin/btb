"""The single bridge from btb's core family declaration into the cert matrix. Every family and flag is read
from `btb.kinds` (KIND_OF, CAPS) - the one torch-free declaration family() itself builds from - so nothing is
re-listed here and a family added to core appears in the matrix on its own. Torch-free: reading the family set
or its flags needs no engine (family() builds the runtime layers from the same table).

`consistency_problems()` guards what still could diverge: a served family with no capabilities, or an
ARCH_MODEL_TYPES GGUF target that is not served. That every declared family actually builds a runtime Family is
a separate runtime check (test_manifest.test_every_served_type_builds)."""

from __future__ import annotations

import importlib.util

from btb.kinds import CAPS, KIND_OF, Cap, FamilyKind


def btb_src() -> str:
    """the btb package directory the matrices reflect on - the checkout's, or the installed wheel's when the
    checkout's is moved aside (the wheel certify) - located without importing the package (no torch)"""
    spec = importlib.util.find_spec("btb")
    assert spec is not None and spec.submodule_search_locations, "btb is not importable"
    return next(iter(spec.submodule_search_locations))


BTB_SRC = btb_src()

# the family clause of the card step graph's gate, `_CudaMixin._card_family_ok` (btb/engine/cuda.py:284): the
# kernels are written for the dense kernel layout and the sandwich layout, and never for a family that brings
# its own layer. The rest of that gate (the activation, the shapes, the compute dtype) is not a capability;
# the manifest models the shape half on its own. test_manifest holds this set against cuda.py's source.
CARD_FAMILY_CAPS: frozenset[Cap] = frozenset({Cap.KERNEL_LAYOUT, Cap.SANDWICH, Cap.OWN})


def served_kinds() -> list[FamilyKind]:
    """the families core serves, from kinds.KIND_OF - the set the matrix iterates, never a hand list."""
    return sorted(set(KIND_OF.values()), key=lambda k: k.value)


def flags(kind: FamilyKind) -> frozenset[Cap]:
    """the capabilities core turns on for a family, from kinds.CAPS (the same table family() reads)."""
    return CAPS.get(kind, frozenset())


def card_family_ok(kind: FamilyKind) -> bool:
    """whether the card step graph is written for this family, by cuda.py's own capability clause. A family
    this rejects runs `PassTag.CUDA_TORCH_FALLBACK` on a card (forward.py:471), not the graph."""
    fl = flags(kind)
    return (Cap.KERNEL_LAYOUT in fl or Cap.SANDWICH in fl) and Cap.OWN not in fl


def gguf_kinds() -> set[FamilyKind]:
    """the families core can load from a GGUF: each hf.ARCH_MODEL_TYPES target mapped to its FamilyKind."""
    from btb.hf import ARCH_MODEL_TYPES

    return {KIND_OF[mt] for mt in ARCH_MODEL_TYPES.values() if mt in KIND_OF}


def consistency_problems() -> list[str]:
    """where core's family declaration is internally inconsistent: a served family with no CAPS, or a GGUF arch
    target that is not a served model_type. (SERVE_TYPES and SUPPORTED_MODEL_TYPES now derive from KIND_OF, so
    they cannot drift from it.)"""
    from btb.hf import ARCH_MODEL_TYPES, SERVE_TYPES

    problems: list[str] = []
    for kind in sorted(set(KIND_OF.values()), key=lambda k: k.value):
        if kind not in CAPS:
            problems.append(f"{kind.value!r} is served but has no CAPS entry")
    for mt in sorted(ARCH_MODEL_TYPES.values()):
        if mt not in SERVE_TYPES:
            problems.append(f"{mt!r} is an ARCH_MODEL_TYPES GGUF target but not in SERVE_TYPES")
    return problems
