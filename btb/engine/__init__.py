# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine: `StreamedTextModel` runs a model across a machine's tiers and decodes greedily or through a
speculative tree; `GrowLayer` the cache layer, `BatchScheduler` the batch size and the gate every large allocation
asks (`MemoryGrantError` its refusal), `Native` the kernel handles, `pack_model` / `pack_bf16` / `unpack_bf16` the
12-bit store."""

from ..session import Session
from .cache import GrowLayer
from .fused import fast_causal_conv1d
from .model import StreamedTextModel
from .native import Native
from .pack import ESC, pack_bf16, pack_model, unpack_bf16
from .scheduler import BatchScheduler, MemoryGrantError

__all__ = [
    "ESC",
    "BatchScheduler",
    "GrowLayer",
    "MemoryGrantError",
    "Native",
    "Session",
    "StreamedTextModel",
    "fast_causal_conv1d",
    "pack_bf16",
    "pack_model",
    "unpack_bf16",
]
