# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A decode or verify pass of up to 16 rows as its program replayed on the GPU. `kernel` is the base (the ops as
device functions, a kernel each), `program` the table a pass is built into, `model` the parts a family composes its
layers from (`Family.mega_layer`) and the pass itself, `replay` the program encoded once a shape into a Metal indirect
command buffer and replayed."""

from .model import ROWS_MAX, Block, MegaPass

__all__ = ["ROWS_MAX", "Block", "MegaPass"]
