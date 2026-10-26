# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Every stored type's launch as the step's matvec builds it (`matvec.StoredKernel`), by type: what the megakernel's
stored ops and a mixture's expert gather compile from, so each multiplies a type as its step does."""

from __future__ import annotations

from ..kinds import MEGA_STORED, Quant
from . import iquant, kquant, legacyq, q6k
from .matvec import StoredKernel

STORED: dict[Quant, StoredKernel] = {**kquant.KERNELS, **q6k.KERNELS, **legacyq.KERNELS, **iquant.KERNELS}
assert set(STORED) == MEGA_STORED, f"kinds.MEGA_STORED and the stored types' launches differ: {set(STORED)}"
