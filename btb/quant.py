# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The GGUF storage types btb multiplies as stored, in one place: each type's block (gguf's own sizes, pinned to
its table by tests/test_quant.py) and which backend carries a kernel for it - the card's gemv stem, the MLX binder
kind, the bit width of its affine repack. Every list the engine used to spell by hand (the types a device packs,
the entries a fatbin may carry, the affine types) derives from `QUANTS`; the Metal and CUDA sources keep their own
block literals, which are the kernels' knowledge of the format, not the engine's."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
    """One ggml storage type: `name` as gguf spells it, `block_weights` weights a block of `block_bytes`. `card`
    is the card kernel's stem (btb_gemv_<card>_bf16_m{M}) or None where the card reads the tensor dequantized;
    `mlx` the MLX binder kind (`Backend.weight_packed`) or None where MLX does; `cpu` the native CPU gemv stem
    (`btb_gemv_<cpu>_rows`, `Native.gemv_<cpu>`) that multiplies the type as stored; `affine_bits` the bit width
    of the type's affine repack (`gguf.affine_of`, MLX's `quantized_matmul`) where the format has one."""

    name: str
    block_weights: int
    block_bytes: int
    card: str | None = None
    mlx: str | None = None
    cpu: str | None = None
    affine_bits: int | None = None

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


QUANTS: Mapping[str, QuantType] = MappingProxyType(
    {
        t.name: t
        for t in (
            QuantType("Q2_K", 256, 84, card="q2k", mlx="q2k", cpu="q2k"),
            QuantType("Q3_K", 256, 110, card="q3k", mlx="q3k", cpu="q3k"),
            QuantType("Q4_K", 256, 144, card="q4k", mlx="q4k", cpu="q4k", affine_bits=4),
            QuantType("Q5_K", 256, 176, card="q5k", mlx="q5k", cpu="q5k"),
            QuantType("Q6_K", 256, 210, card="q6k", mlx="q6k", cpu="q6k"),
            QuantType("Q4_0", 32, 18, mlx="affine", cpu="q40", affine_bits=4),
            QuantType("Q4_1", 32, 20, mlx="affine", cpu="q41", affine_bits=4),
            QuantType("Q8_0", 32, 34, mlx="affine", cpu="q80", affine_bits=8),
            QuantType("IQ4_NL", 32, 18, mlx="iq4nl", cpu="iq4nl"),
            QuantType("IQ4_XS", 256, 136, mlx="iq4xs", cpu="iq4xs"),
            QuantType("IQ1_S", 256, 50, mlx="iq1s", cpu="iq1s"),
            QuantType("IQ1_M", 256, 56, mlx="iq1m", cpu="iq1m"),
            QuantType("IQ2_XXS", 256, 66, mlx="iq2xxs", cpu="iq2xxs"),
            QuantType("IQ2_XS", 256, 74, mlx="iq2xs", cpu="iq2xs"),
            QuantType("IQ2_S", 256, 82, mlx="iq2s", cpu="iq2s"),
            QuantType("IQ3_XXS", 256, 98, mlx="iq3xxs", cpu="iq3xxs"),
            QuantType("IQ3_S", 256, 110, mlx="iq3s", cpu="iq3s"),
        )
    }
)


# the one type the engine refers to by identity: a tied Q6_K head doubles as the embedding table through a gather
# of its packed bytes (the only type with a gather kernel)
Q6_K: QuantType = QUANTS["Q6_K"]


def quant_of(name: str) -> QuantType | None:
    """the type gguf names `name` (a reader tensor's `tensor_type.name`), or None for one btb reads dequantized"""
    return QUANTS.get(name)


# the affine repack's types and bit widths (gguf.affine_of, pool's packed sizing)
AFFINE_TYPES: Mapping[str, int] = MappingProxyType(
    {q.name: q.affine_bits for q in QUANTS.values() if q.affine_bits is not None}
)

# every fatbin entry the card's packed path may launch; a fatbin built before a type was added lacks its entries,
# and Native binds them only when present
CARD_KERNELS: frozenset[str] = frozenset(
    q.card_kernel(m) for q in QUANTS.values() if q.card is not None for m in CARD_WIDTHS
)

# the CPU grid-codebook types: their native gemv takes the type's int8 grid as a buffer, and the ksigns subset
# also the shared 128-entry sign table (the others carry explicit signs, or none). The engine reads both from
# the gguf package and passes them to `Native.gemv_<cpu>`.
CPU_LATTICE: frozenset[str] = frozenset({"iq3xxs", "iq2xxs", "iq2xs", "iq2s", "iq1s", "iq1m", "iq3s"})
CPU_KSIGNS: frozenset[str] = frozenset({"iq3xxs", "iq2xxs", "iq2xs"})
