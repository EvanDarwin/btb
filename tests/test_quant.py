# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The quant registry (btb.quant) against gguf's own table: every block size it holds is gguf's, and the lists the
engine derives from it are the ones it used to spell by hand."""

from __future__ import annotations

import pytest

from btb.quant import (
    AFFINE_TYPES,
    CARD_KERNELS,
    CARD_WIDTHS,
    CPU_KSIGNS,
    CPU_LATTICE,
    QUANTS,
    card_width,
    quant_of,
)

gguf = pytest.importorskip("gguf")


def test_every_block_size_is_ggufs() -> None:
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

    for q in QUANTS.values():
        weights, nbytes = GGML_QUANT_SIZES[GGMLQuantizationType[q.name]]
        assert (q.block_weights, q.block_bytes) == (weights, nbytes), q.name
        assert q.nbytes(3, q.block_weights * 2) == 3 * 2 * q.block_bytes
        assert q.packable((3, q.block_weights * 2)) and not q.packable((3, q.block_weights * 2 + 1))
        assert not q.packable((q.block_weights,))


def test_the_derived_lists_are_the_hand_written_ones() -> None:
    assert dict(AFFINE_TYPES) == {"Q4_0": 4, "Q4_1": 4, "Q8_0": 8, "Q4_K": 4}
    card = sorted(q.name for q in QUANTS.values() if q.card is not None)
    assert card == ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
    assert all(QUANTS[n].block_weights == 256 for n in card)  # the card kernels walk 256-weight superblocks
    assert len(CARD_KERNELS) == len(card) * len(CARD_WIDTHS)
    assert "btb_gemv_q4k_bf16_m32" in CARD_KERNELS and QUANTS["Q4_K"].card_kernel(1) == "btb_gemv_q4k_bf16_m1"
    with pytest.raises(ValueError):
        QUANTS["Q8_0"].card_kernel(1)
    assert quant_of("BF16") is None and quant_of("Q6_K") is QUANTS["Q6_K"]
    # every type has a native CPU gemv; the lattice subset takes the grid buffer, the ksigns subset the sign table
    assert all(q.cpu is not None for q in QUANTS.values())
    assert CPU_KSIGNS < CPU_LATTICE and CPU_LATTICE < {q.cpu for q in QUANTS.values()}
    assert {q.name for q in QUANTS.values() if q.cpu in CPU_LATTICE} == {
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ3_XXS",
        "IQ3_S",
    }
    assert abs(QUANTS["Q4_K"].bits - 4.5) < 1e-9
    assert [card_width(t) for t in (1, 2, 3, 5, 9, 17, 32)] == [1, 2, 4, 8, 16, 32, 32]
    with pytest.raises(ValueError):
        card_width(33)
