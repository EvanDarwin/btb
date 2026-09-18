# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's named vocabularies as StrEnums, each spelled once: a member equals its string (a config's
`layer_types` entry, a report's field), hashes like it and JSON-encodes as it, so the enums drop in where the
strings were and the names stop being scattered. Torch-free."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Any

# token ids, the currency of the engine's inputs and outputs; TokenRows a batch of them (ragged prompts, a
# tree's children per node)
Tokens = Sequence[int]
TokenRows = Sequence[Tokens]
# a verify tree's parent map: parents[j] is node j's parent, -1 the prefix; a chain reads parents[j] == j - 1
Parents = Sequence[int]
# the nodes of one root-to-leaf branch of a verify tree, the path a commit keeps
NodePath = Sequence[int]
# a decoded JSON object: a config, a report, a request body, a receipt
Json = dict[str, Any]
# the engine's log callback: print-shaped, its return never read
Log = Callable[..., object]


class UnknownKind(ValueError):
    """a config's layer type `text` is not one transformers names; `kinds` are"""

    def __init__(self, text: Any, kinds: Iterable[str]) -> None:
        self.text, self.kinds = str(text), list(kinds)
        super().__init__(f"layer type {text!r} is not one transformers names; the kinds are {', '.join(self.kinds)}")


class LayerKind(StrEnum):
    """A decoder layer's kind, as transformers' `layer_types` names it: every entry of its
    ALLOWED_ATTN_LAYER_TYPES (a test holds the two lists equal), so any config transformers accepts reads into
    the enum. The engine runs FULL, SLIDING, LINEAR and QWEN_SPARSE; a family the engine does not drive is
    refused at the family table, never here."""

    FULL = "full_attention"
    SLIDING = "sliding_attention"
    CHUNKED = "chunked_attention"
    WINDOW = "window_attention"
    COMPRESSED_SPARSE = "compressed_sparse_attention"
    HEAVILY_COMPRESSED = "heavily_compressed_attention"
    MINIMAX_M3_SPARSE = "minimax_m3_sparse"
    CONV = "conv"
    MOE = "moe"
    HYBRID = "hybrid"
    HYBRID_SLIDING = "hybrid_sliding"
    DEEPSEEK_SPARSE = "deepseek_sparse_attention"
    QWEN_SPARSE = "qwen_sparse_attention"
    LINEAR = "linear_attention"

    @classmethod
    def of(cls, text: str) -> LayerKind:
        """the kind of a config's entry; a name transformers itself does not allow is an UnknownKind naming it"""
        try:
            return cls(str(text))
        except ValueError:
            raise UnknownKind(text, cls) from None


class FamilyKind(StrEnum):
    """the model families btb drives, keyed from the config's model_type (btb/engine/families.py)"""

    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"
    QWEN4 = "qwen4"
    PHI3 = "phi3"
    GPT_OSS = "gpt_oss"


class Tier(StrEnum):
    """where a part of the model lives: the head, the drafter and the attention cache in a Placement and a report"""

    CARD = "card"
    HOST = "host"
    PACKED = "packed"  # the head read from the 12-bit store
    NONE = "none"  # no such part (a model without a drafting head)


class LayerTier(StrEnum):
    """a layer's tier in a Placement: on the card, in RAM, streamed from the drive each pass, or in flight"""

    RESIDENT = "resident"
    HOST = "host"
    COLD = "cold"
    STREAMED = "streamed"


class SlotKind(StrEnum):
    """what a cold ring slot holds for one linear: the checkpoint's bf16 bytes read off the drive, a 12-bit store
    entry read off the drive, or bf16 the engine dequantizes into the slot each pass (a GGUF tensor of another
    storage type, which is not on the drive as bf16)"""

    BF16 = "bf16"
    P12 = "p12"
    MEM = "mem"
