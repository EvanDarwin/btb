# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The GGUF storage types (kinds.Quant) the card and the native CPU gemv multiply as stored: each type's block
(gguf's own sizes, pinned to its table by tests/unit/test_quant.py) and its kernel stems. The entries a fatbin may
carry derive from `QUANTS`; the Metal and CUDA sources keep their own block literals, which are the kernels'
knowledge of the format, not the engine's."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .kinds import Quant

# the row counts the card's gemv kernels are compiled for (btb_gemv_*_m{M}); a pass pads to the next one up
CARD_WIDTHS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def card_width(rows: int) -> int:
    """the kernel width a `rows`-row pass launches at: the smallest of CARD_WIDTHS that holds it"""
    for m in CARD_WIDTHS:
        if rows <= m:
            return m
    raise ValueError(f"{rows} rows: the card's gemv kernels take at most {CARD_WIDTHS[-1]}")


@dataclass(frozen=True)
class QuantType:
    """One ggml storage type: `name` the type, `block_weights` weights a block of `block_bytes`. `card` is the card
    kernel's stem (btb_gemv_<card>_bf16_m{M}) or None where the card reads the tensor dequantized; `cpu` the native
    CPU gemv stem (`btb_gemv_<cpu>_rows`, `Native.gemv_<cpu>`) that multiplies the type as stored."""

    name: Quant
    block_weights: int
    block_bytes: int
    card: str | None = None
    cpu: str | None = None

    @property
    def bits(self) -> float:
        """bits a weight as stored"""
        return self.block_bytes * 8 / self.block_weights

    def packable(self, shape: tuple[int, ...]) -> bool:
        """whether a tensor of `shape` is a matrix whose rows are whole blocks - the kernels' one requirement"""
        return len(shape) == 2 and shape[1] % self.block_weights == 0

    def nbytes(self, rows: int, cols: int) -> int:
        """the bytes a [rows, cols] tensor takes as stored"""
        return rows * cols // self.block_weights * self.block_bytes

    def card_kernel(self, m: int) -> str:
        """the fatbin entry for an m-row pass"""
        if self.card is None:
            raise ValueError(f"{self.name} has no card kernel")
        return f"btb_gemv_{self.card}_bf16_m{m}"


QUANTS: Mapping[Quant, QuantType] = MappingProxyType(
    {
        t.name: t
        for t in (
            QuantType(Quant.Q2_K, 256, 84, card="q2k", cpu="q2k"),
            QuantType(Quant.Q3_K, 256, 110, card="q3k", cpu="q3k"),
            QuantType(Quant.Q4_K, 256, 144, card="q4k", cpu="q4k"),
            QuantType(Quant.Q5_K, 256, 176, card="q5k", cpu="q5k"),
            QuantType(Quant.Q6_K, 256, 210, card="q6k", cpu="q6k"),
            QuantType(Quant.Q4_0, 32, 18, cpu="q40"),
            QuantType(Quant.Q4_1, 32, 20, cpu="q41"),
            QuantType(Quant.Q8_0, 32, 34, cpu="q80"),
            QuantType(Quant.IQ4_NL, 32, 18, cpu="iq4nl"),
            QuantType(Quant.IQ4_XS, 256, 136, cpu="iq4xs"),
            QuantType(Quant.IQ1_S, 256, 50, cpu="iq1s"),
            QuantType(Quant.IQ1_M, 256, 56, cpu="iq1m"),
            QuantType(Quant.IQ2_XXS, 256, 66, cpu="iq2xxs"),
            QuantType(Quant.IQ2_XS, 256, 74, cpu="iq2xs"),
            QuantType(Quant.IQ2_S, 256, 82, cpu="iq2s"),
            QuantType(Quant.IQ3_XXS, 256, 98, cpu="iq3xxs"),
            QuantType(Quant.IQ3_S, 256, 110, cpu="iq3s"),
        )
    }
)


def quant_of(name: str) -> QuantType | None:
    """the type gguf names `name` (a reader tensor's `tensor_type.name`), or None for one btb reads dequantized"""
    try:
        return QUANTS.get(Quant(name))
    except ValueError:
        return None


# every fatbin entry the card's packed path may launch; a fatbin built before a type was added lacks its entries,
# and Native binds them only when present
CARD_KERNELS: frozenset[str] = frozenset(
    q.card_kernel(m) for q in QUANTS.values() if q.card is not None for m in CARD_WIDTHS
)

# the grid-codebook types (QuantClass.LATTICE) whose native CPU gemv also takes the shared 128-entry sign table;
# the others carry explicit signs, or none. The engine reads the table from the gguf package.
CPU_KSIGNS: frozenset[Quant] = frozenset({Quant.IQ3_XXS, Quant.IQ2_XXS, Quant.IQ2_XS})
