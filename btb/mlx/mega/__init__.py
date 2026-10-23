# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel: a decode or verify pass of up to 16 rows as one Metal dispatch. `kernel` is the base (the ops
as device functions under a ticket scheduler, no co-residency assumed), `program` the table a pass is built into,
`model` the parts a family composes its layers from (`Family.mega_layer`) and the pass itself, `gpu` what it
measures of the device under it."""

from .model import ROWS_MAX, Block, MegaPass

__all__ = ["ROWS_MAX", "Block", "MegaPass"]
