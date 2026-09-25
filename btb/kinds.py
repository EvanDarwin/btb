# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's named vocabularies as StrEnums, each spelled once: a member equals its string (a config's
`layer_types` entry, a report's field), hashes like it and JSON-encodes as it, so the enums drop in where the
strings were and the names stop being scattered. Torch-free."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
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
    GEMMA3 = "gemma3"


# the name a user would recognize for each family; an unsupported load lists these values
FAMILY_NAMES: dict[FamilyKind, str] = {
    FamilyKind.QWEN3: "Qwen3",
    FamilyKind.QWEN3_5: "Qwen3.5",
    FamilyKind.PHI3: "Phi-3",
    FamilyKind.QWEN4: "Qwen4",
    FamilyKind.GPT_OSS: "GPT-OSS",
    FamilyKind.GEMMA3: "Gemma 3",
}


class ModelType(StrEnum):
    """a config's `model_type`, as transformers and GGUF name it - the string the loader reads and `family()`
    switches on. `KIND_OF` maps each to its FamilyKind (the "_text" GGUF variants share a base family). Spelled
    once here so the model_type strings stop being scattered across hf/families/gguf."""

    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"
    QWEN3_5_TEXT = "qwen3_5_text"
    PHI3 = "phi3"
    QWEN4_EXP = "qwen4_exp"
    QWEN4_EXP_TEXT = "qwen4_exp_text"
    GPT_OSS = "gpt_oss"
    GEMMA3 = "gemma3"
    GEMMA3_TEXT = "gemma3_text"


class Cap(StrEnum):
    """a family capability - a boolean on `Family` that selects a code path. Spelled once here (the value is the
    field name); `families.family()` builds a Family's flags from `CAPS`, so this table is the single truth for
    what a family does, torch-free, readable without the engine."""

    DENSE = "dense"
    KERNEL_LAYOUT = "kernel_layout"
    HYBRID = "hybrid"
    MOE = "moe"
    MXFP4 = "mxfp4"
    MROPE = "mrope"
    ATTN_GATE = "attn_gate"
    OWN = "own"
    NORM_CENTERED = "norm_centered"
    EMBED_SCALE = "embed_scale"
    DUAL_ROPE = "dual_rope"
    SANDWICH = "sandwich"
    FLAT_CACHE = "flat_cache"
    EAGER = "eager"
    FAST = "fast"


# every model_type btb serves -> the family that drives it (the "_text" GGUF variants share the base family).
# hf.SERVE_TYPES and families.SUPPORTED_MODEL_TYPES derive from this; a new family is one entry here.
KIND_OF: dict[ModelType, FamilyKind] = {
    ModelType.QWEN3: FamilyKind.QWEN3,
    ModelType.QWEN3_5: FamilyKind.QWEN3_5,
    ModelType.QWEN3_5_TEXT: FamilyKind.QWEN3_5,
    ModelType.PHI3: FamilyKind.PHI3,
    ModelType.QWEN4_EXP: FamilyKind.QWEN4,
    ModelType.QWEN4_EXP_TEXT: FamilyKind.QWEN4,
    ModelType.GPT_OSS: FamilyKind.GPT_OSS,
    ModelType.GEMMA3: FamilyKind.GEMMA3,
    ModelType.GEMMA3_TEXT: FamilyKind.GEMMA3,
}

# each family's capabilities, the code paths it selects. family() builds the runtime Family's flags from this
# (it adds the transformers classes); the cert reads it directly. `fast` off (gpt-oss) is simply its absence.
CAPS: dict[FamilyKind, frozenset[Cap]] = {
    FamilyKind.QWEN3: frozenset({Cap.DENSE, Cap.KERNEL_LAYOUT, Cap.FAST}),
    FamilyKind.QWEN3_5: frozenset({Cap.HYBRID, Cap.MROPE, Cap.ATTN_GATE, Cap.NORM_CENTERED, Cap.FAST}),
    FamilyKind.PHI3: frozenset({Cap.DENSE, Cap.FAST}),
    FamilyKind.QWEN4: frozenset({Cap.MOE, Cap.MROPE, Cap.ATTN_GATE, Cap.OWN, Cap.FAST}),
    FamilyKind.GPT_OSS: frozenset({Cap.MOE, Cap.MXFP4, Cap.EAGER, Cap.FLAT_CACHE}),
    FamilyKind.GEMMA3: frozenset(
        {Cap.NORM_CENTERED, Cap.EMBED_SCALE, Cap.DUAL_ROPE, Cap.SANDWICH, Cap.FLAT_CACHE, Cap.FAST}
    ),
}


class Quant(StrEnum):
    """A stored weight type btb can read, spelled once as its GGUF type name (a tensor's `tensor_type.name`),
    so `t.tensor_type.name == Quant.Q4_K` and `name in AFFINE_TYPES` work with the member in the string's place.
    This is the single set of supported types: the engine's binders and the cert's quant coverage both read it,
    so a type is declared here and nowhere else. Torch-free."""

    BF16 = "BF16"
    F16 = "F16"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"
    Q2_K = "Q2_K"
    Q3_K = "Q3_K"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    IQ4_NL = "IQ4_NL"
    IQ4_XS = "IQ4_XS"
    IQ3_XXS = "IQ3_XXS"
    IQ3_S = "IQ3_S"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    MXFP4 = "MXFP4"


class QuantClass(StrEnum):
    """the bind path that reads a Quant - the engine's dispatch group and the cert's storage bucket. Q4_K is
    KQUANT (it binds via its k-quant kernel first); `AFFINE_TYPES` separately lists the types the affine decoder
    can also produce, Q4_K among them."""

    FLOAT = "float"  # BF16 / F16: dequantized, no as-stored kernel
    AFFINE = "affine"  # Q4_0 / Q4_1 / Q8_0: the packed affine kernel
    KQUANT = "kquant"  # Q2_K..Q6_K: the k-quant as-stored kernels
    IQ4 = "iq4"  # IQ4_NL / IQ4_XS: the non-linear 4-bit codebook
    LATTICE = "lattice"  # the grid-codebook i-quants (the Unsloth dynamic mixes)
    MXFP4 = "mxfp4"  # gpt-oss experts


# each supported type -> the bind path that reads it. `families`-style single truth: the lattice member list
# drives the engine's IQ-name map and the cert reads the whole set. A new kernel is one member plus one row.
QUANT_KIND: dict[Quant, QuantClass] = {
    Quant.BF16: QuantClass.FLOAT,
    Quant.F16: QuantClass.FLOAT,
    Quant.Q4_0: QuantClass.AFFINE,
    Quant.Q4_1: QuantClass.AFFINE,
    Quant.Q8_0: QuantClass.AFFINE,
    Quant.Q2_K: QuantClass.KQUANT,
    Quant.Q3_K: QuantClass.KQUANT,
    Quant.Q4_K: QuantClass.KQUANT,
    Quant.Q5_K: QuantClass.KQUANT,
    Quant.Q6_K: QuantClass.KQUANT,
    Quant.IQ4_NL: QuantClass.IQ4,
    Quant.IQ4_XS: QuantClass.IQ4,
    Quant.IQ3_XXS: QuantClass.LATTICE,
    Quant.IQ3_S: QuantClass.LATTICE,
    Quant.IQ2_XXS: QuantClass.LATTICE,
    Quant.IQ2_XS: QuantClass.LATTICE,
    Quant.IQ2_S: QuantClass.LATTICE,
    Quant.IQ1_S: QuantClass.LATTICE,
    Quant.IQ1_M: QuantClass.LATTICE,
    Quant.MXFP4: QuantClass.MXFP4,
}


def quants_of(cls: QuantClass) -> list[Quant]:
    """the supported types bound by one path, from QUANT_KIND - the list the engine and cert derive, never a
    hand copy (e.g. the lattice kinds, or the affine-kernel types)."""
    return [q for q in Quant if QUANT_KIND[q] is cls]


def latt_backend_key(q: Quant) -> str:
    """the backend lattice-kernel key for a LATTICE quant: its GGUF name lowercased, underscores dropped
    (IQ3_XXS -> iq3xxs), the one rule the engine's name map and the backend's `_LATT` both follow."""
    return q.value.lower().replace("_", "")


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


class Proposer(StrEnum):
    """what proposes a speculative pass's drafts (btb/engine/generate.py): the n-gram continuations, or a
    drafting head as a chain, a fixed tree or the dynamic tree it draws itself. A sibling draft model is not a
    member - it rides behind whichever of these the engine runs (`draft_engine`)."""

    NGRAM = "ngram"
    MTP = "mtp"
    MTP_TREE = "mtp_tree"
    MTP_DYN = "mtp_dyn"

    @property
    def mtp(self) -> bool:
        """whether this proposer draws from the checkpoint's `mtp.*` drafting head, the one rule the placement
        and tier code key on too"""
        return self.value.startswith("mtp")

    @classmethod
    def of(cls, text: str) -> Proposer:
        """the proposer `text` names; an unknown name raises rather than silently decoding as n-gram"""
        try:
            return cls(str(text))
        except ValueError:
            raise ValueError(f"proposer {text!r} is not one btb has; the proposers are {', '.join(cls)}") from None


class PassTag(StrEnum):
    """Every path fork a decode pass can take, so a `PassReport` can assert the pass ran the intended code path
    rather than a silent fallback. One member per fork in the engine; the engine records the tag at the fork and
    a lint (tests/cert/test_provenance.py) fails when a new `_forward_*` entry point carries none. Torch-free."""

    # the compute path: which forward implementation ran (btb/engine/mlx_forward.py, cuda.py, forward.py)
    MLX_MEGA = "mlx_mega"  # the pass as one Metal dispatch (_forward_mega, kernel_layout families)
    MLX_STEP = "mlx_step"  # the per-token graph (_forward_mlx, dense/sandwich)
    MLX_STEP_FUSED = "mlx_step_fused"  # that graph's fused kernels: the q/k norms, the rope and the attention in one
    MLX_STEP_UNFUSED = "mlx_step_unfused"  # the same graph as separate ops: no kernel/sandwich layout, a partial
    # rotary, a non-bf16 state, or a batched pass
    MLX_HYBRID = "mlx_hybrid"  # the Qwen3.5 DeltaNet fused path (_forward_mlx_hybrid)
    MLX_SDPA = "mlx_sdpa"  # MLX's own fused attention, where the node kernel does not apply
    CUDA_GRAPH = "cuda_graph"  # a captured card graph ran the pass (_forward_card_segment/_forward_fast/step graph)
    CUDA_TORCH_FALLBACK = "cuda_torch_fallback"  # a card layer through torch modules, btb's kernels absent
    CPU_NATIVE = "cpu_native"  # a host layer through the native CPU gemv kernels
    MLX_PEROP = "mlx_perop"  # a host layer's linears through Native.mlx.linear (the non-fused MLX path: MoE, offload)
    # the stored weight path a linear read (mlx_state.affine; only on MLX)
    QUANT_ASSTORED = "quant_asstored"  # a GGUF quant bound as its own bytes (matvec kernels)
    QUANT_DEQUANT = "quant_dequant"  # a bf16 slot (float weights, or a GGUF dequantized to bf16)
    FP8_ASSTORED = "fp8_asstored"  # FP8 weights multiplied as stored, e4m3 bytes and scale grid (the native kernel)
    FP8_WIDENED = "fp8_widened"  # FP8 weights widened by their scales to bf16 before the matmul
    # where a MoE layer's experts came from, and how their blocks were multiplied (host.py, experts.py)
    EXPERT_TABLES = "expert_tables"  # the checkpoint's whole expert tables, no store
    EXPERT_STORE = "expert_store"  # the store served the layer's experts out of its slots
    EXPERT_LINE = "expert_line"  # the store's plain line of riders
    EXPERT_BUS_PASS = "expert_bus_pass"  # the store's Bus Pass residency policy (the `bus_pass` option)
    EXPERT_VRAM_SEAT = "expert_vram_seat"  # an expert multiplied from its seat on the card (`vram_experts_gb`)
    EXPERT_MXFP4_ASSTORED = "expert_mxfp4_asstored"  # MXFP4 experts multiplied in their stored blocks
    EXPERT_MXFP4_DEQUANT = "expert_mxfp4_dequant"  # MXFP4 experts widened to float before the matmul
    # the head, and the route a long prefill took (btb/engine/forward.py)
    HEAD_RESIDENT = "head_resident"  # the head multiplied where the model runs
    HEAD_STREAMED = "head_streamed"  # the head read from the checkpoint for the pass (`resident_head` off)
    PREFILL_CARD = "prefill_card"  # a prefill ran the host layers on the card (`prefill_card_min` rows or more)
    # the token draw (btb/sampling.py)
    SAMPLE_GREEDY = "sample_greedy"  # the argmax
    SAMPLE_STOCHASTIC = "sample_stochastic"  # the Gumbel-max draw under a temperature
    # the placement tiers a pass ran layers on (btb/engine/device.py Placement.tier)
    TIER_RESIDENT = "tier_resident"
    TIER_HOST = "tier_host"
    TIER_COLD = "tier_cold"
    TIER_STREAMED = "tier_streamed"
    # speculation: the proposer, and whether a pass's drafts were accepted (btb/engine/generate.py)
    SPEC_OFF = "spec_off"  # the plain one-token loop, no drafts
    SPEC_MTP = "spec_mtp"  # a drafting head proposes a chain
    SPEC_MTP_TREE = "spec_mtp_tree"  # a drafting head proposes a fixed tree
    SPEC_MTP_DYN = "spec_mtp_dyn"  # a drafting head draws its own tree
    SPEC_DRAFT = "spec_draft"  # a sibling draft model proposes
    SPEC_NGRAM = "spec_ngram"  # the n-gram proposer
    SPEC_ACCEPT = "spec_accept"  # a pass accepted at least one drafted token
    SPEC_REJECT = "spec_reject"  # a pass rejected at least one drafted token


# the tag a speculative decode records for the proposer it ran (a sibling draft model's decode records SPEC_DRAFT
# in the n-gram proposer's place)
PROPOSER_TAG: dict[Proposer, PassTag] = {
    Proposer.NGRAM: PassTag.SPEC_NGRAM,
    Proposer.MTP: PassTag.SPEC_MTP,
    Proposer.MTP_TREE: PassTag.SPEC_MTP_TREE,
    Proposer.MTP_DYN: PassTag.SPEC_MTP_DYN,
}


@dataclass(frozen=True)
class PassReport:
    """What a decode pass (or a whole `generate()`, whose tags accumulate) took: the set of `PassTag`s its forks
    recorded, and the speculation counts. The cert asserts a pass carries the tag of the path it meant to
    exercise, so a fallback to torch or the per-op path is caught instead of certified. Torch-free."""

    tags: frozenset[PassTag] = field(default_factory=frozenset)
    spec_proposed: int = 0
    spec_accepted: int = 0

    def __contains__(self, tag: PassTag) -> bool:
        return tag in self.tags
