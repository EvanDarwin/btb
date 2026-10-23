# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's named vocabularies as StrEnums, each spelled once: a member equals its string (a config's
`layer_types` entry, a report's field), hashes like it and JSON-encodes as it, so the enums drop in where the
strings were and the names stop being scattered. The card kernels' parameter roles close the file, a NewType each.
Torch-free."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, NewType

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
    ALLOWED_ATTN_LAYER_TYPES, read through its legacy names (`LEGACY`; a test holds the lists equal), so any config
    transformers accepts reads into the enum. The engine runs FULL, SLIDING, LINEAR and QWEN_SPARSE (Qwen4's
    indexed sparse attention); a family the engine does not drive is refused at the family table, never here."""

    FULL = "full_attention"
    SLIDING = "sliding_attention"
    CHUNKED = "chunked_attention"
    WINDOW = "window_attention"
    # an indexer's attention: Qwen4's QSA, DeepSeek's DSA. transformers 5.18 names it so; 5.16-5.17 named each its own
    # (`LEGACY`), and a config read under them, or saved by them, carries those
    QWEN_SPARSE = "indexed_attention"
    COMPRESSED_SPARSE = "compressed_sparse_attention"
    HEAVILY_COMPRESSED = "heavily_compressed_attention"
    MINIMAX_M3_SPARSE = "minimax_m3_sparse"
    CONV = "conv"
    MOE = "moe"
    HYBRID = "hybrid"
    HYBRID_SLIDING = "hybrid_sliding"
    LINEAR = "linear_attention"

    @classmethod
    def of(cls, text: str) -> LayerKind:
        """the kind of a config's entry, its legacy names too (`LEGACY`); a name transformers itself does not allow
        is an UnknownKind naming it"""
        text = str(text)
        try:
            return cls(LEGACY.get(text, text))
        except ValueError:
            raise UnknownKind(text, cls) from None

    def hf_name(self) -> str:
        """the kind as the installed transformers spells it in a config it is handed (a drafter's config built
        here): its cache picks the layer's form by that name - under another spelling Qwen4's attention got a plain
        cache layer, with no indexer keys. btb takes transformers 5.16 on, which named Qwen4's indexed attention
        `qwen_sparse_attention` until 5.18"""
        if self is LayerKind.QWEN_SPARSE:
            from transformers.configuration_utils import ALLOWED_ATTN_LAYER_TYPES

            if self.value not in ALLOWED_ATTN_LAYER_TYPES:
                return "qwen_sparse_attention"
        return self.value


# a config's layer type by a name transformers used before (and still reads, mapping it on): its kind's name now
LEGACY: dict[str, str] = {
    "qwen_sparse_attention": "indexed_attention",  # transformers <= 5.17: Qwen4's QSA
    "deepseek_sparse_attention": "indexed_attention",  # transformers <= 5.17: DeepSeek's DSA
    "mamba": "linear_attention",
    "attention": "full_attention",
}


class FamilyKind(StrEnum):
    """the model families btb drives, keyed from the config's model_type (a class each in btb/engine/families/)"""

    QWEN3 = "qwen3"
    QWEN3_5 = "qwen3_5"
    QWEN3_5_MOE = "qwen3_5_moe"
    QWEN4 = "qwen4"
    PHI3 = "phi3"
    GPT_OSS = "gpt_oss"
    GEMMA3 = "gemma3"


# the name a user would recognize for each family; an unsupported load lists these values
FAMILY_NAMES: dict[FamilyKind, str] = {
    FamilyKind.QWEN3: "Qwen3",
    FamilyKind.QWEN3_5: "Qwen3.5",
    FamilyKind.QWEN3_5_MOE: "Qwen3.5-MoE",
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
    QWEN3_5_MOE = "qwen3_5_moe"
    QWEN3_5_MOE_TEXT = "qwen3_5_moe_text"
    PHI3 = "phi3"
    QWEN4_EXP = "qwen4_exp"
    QWEN4_EXP_TEXT = "qwen4_exp_text"
    GPT_OSS = "gpt_oss"
    GEMMA3 = "gemma3"
    GEMMA3_TEXT = "gemma3_text"


class Cap(StrEnum):
    """a family capability - a boolean on `Family` that selects a code path. Spelled once here (the value is the
    field name); each family's `build` takes its Family's flags from `CAPS`, so this table is the single truth for
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
    ModelType.QWEN3_5_MOE: FamilyKind.QWEN3_5_MOE,
    ModelType.QWEN3_5_MOE_TEXT: FamilyKind.QWEN3_5_MOE,
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
    FamilyKind.QWEN3_5_MOE: frozenset({Cap.HYBRID, Cap.MOE, Cap.MROPE, Cap.ATTN_GATE, Cap.NORM_CENTERED, Cap.FAST}),
    FamilyKind.PHI3: frozenset({Cap.DENSE, Cap.FAST}),
    FamilyKind.QWEN4: frozenset({Cap.MOE, Cap.MROPE, Cap.ATTN_GATE, Cap.OWN, Cap.FAST}),
    FamilyKind.GPT_OSS: frozenset({Cap.MOE, Cap.MXFP4, Cap.EAGER, Cap.FLAT_CACHE}),
    FamilyKind.GEMMA3: frozenset(
        {Cap.NORM_CENTERED, Cap.EMBED_SCALE, Cap.DUAL_ROPE, Cap.SANDWICH, Cap.FLAT_CACHE, Cap.FAST}
    ),
}


class Quant(StrEnum):
    """A stored weight type btb can read, spelled once as its GGUF type name (a tensor's `tensor_type.name`),
    so `t.tensor_type.name == Quant.Q4_K` works with the member in the string's place.
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
    """the bind path that reads a Quant - the engine's dispatch group and the cert's storage bucket"""

    FLOAT = "float"  # BF16 / F16: a linear read as bf16; an expert multiplied as stored (`EXPERT_FLOATS`)
    AFFINE = "affine"  # Q4_0 / Q4_1 / Q8_0: one scale (and Q4_1's min) a 32-weight block
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


# the classes whose linears every backend multiplies in their stored blocks, a gemv per type (CPU, MLX, card);
# FLOAT is read into a bf16 slot, MXFP4 is the expert store's
STORED_CLASSES: frozenset[QuantClass] = frozenset(
    {QuantClass.AFFINE, QuantClass.KQUANT, QuantClass.IQ4, QuantClass.LATTICE}
)


def stored_quants() -> list[Quant]:
    """the types a linear multiplies as stored, in declaration order"""
    return [q for q in Quant if QUANT_KIND[q] in STORED_CLASSES]


# the stored types the MLX megakernel multiplies as stored, each through its launch's own chain: the affine and
# k-quant classes whole; its registry (`mlx.mega.kernel.STORED`) is held to this at import, so the torch-free cert
# reads the same set
MEGA_STORED: frozenset[Quant] = frozenset(q for q in Quant if QUANT_KIND[q] in (QuantClass.AFFINE, QuantClass.KQUANT))


def stored_in_place(q: Quant) -> bool:
    """whether the MLX kernels read a stored type's blocks as the file lays them out, one stream and no repack:
    IQ4_NL splits its blocks into two and the lattice types add side streams (`mlx.backend.stored_streams`)"""
    return q is not Quant.IQ4_NL and QUANT_KIND[q] is not QuantClass.LATTICE


# the float types a GGUF's expert store also holds as the file's and multiplies with its own kernel, so a quantizer's
# float fallback beside a stored type keeps the experts stored (a Q2_K file's down at 400 columns, no whole block)
EXPERT_FLOATS: tuple[Quant, ...] = (Quant.BF16, Quant.F16)


def expert_quants() -> list[Quant]:
    """the types an expert store multiplies as stored: the stored types and `EXPERT_FLOATS`"""
    return [*stored_quants(), *EXPERT_FLOATS]


def quant_key(q: Quant) -> str:
    """the stem every backend names a type's kernels by: its GGUF name lowercased, underscores dropped (Q4_K ->
    q4k, IQ3_XXS -> iq3xxs) - `btb_gemv_<key>_rows` natively, `btb_gemv_<key>_bf16_m<M>` on the card, the MLX
    backend's kernel keys."""
    return q.value.lower().replace("_", "")


# the row counts the card's gemv kernels are built for (btb_gemv_*_bf16_m{M}); a pass pads to the next one up
CARD_ROWS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


class Tier(StrEnum):
    """where a part of the model lives: the head, the drafter and the attention cache in a Placement and a report"""

    CARD = "card"
    GPU = "gpu"  # Apple silicon's GPU (MLX), over the memory the host shares
    HOST = "host"
    PACKED = "packed"  # the head read from the 12-bit store
    NONE = "none"  # no such part (a model without a drafting head)


class LayerTier(StrEnum):
    """a layer's tier in a Placement: on the card, in RAM, streamed from the drive each pass, or in flight"""

    RESIDENT = "resident"
    HOST = "host"
    COLD = "cold"
    STREAMED = "streamed"


class HostCache(StrEnum):
    """The host CPU's cache figures in bytes (`sysinfo.host_cache_sizes`), two per level.

    `_INSTANCE` is the smallest single instance: what a thread running alone gets, on any core. `_SHARE` is the
    smallest share a thread gets with every thread busy: an instance's size over the threads sharing it,
    hyperthreads counted. A consumer picks by how many threads it runs. Neither assumes where a thread runs."""

    L1D_INSTANCE = "host_l1d_instance"
    L1D_SHARE = "host_l1d_share"
    L2_INSTANCE = "host_l2_instance"
    L2_SHARE = "host_l2_share"
    L3_INSTANCE = "host_l3_instance"
    L3_SHARE = "host_l3_share"


class HostCacheTotal(StrEnum):
    """A cache level's every instance added up, in bytes (`sysinfo.host_cache_totals`): the size a working set must
    pass to miss the level everywhere. Only the bench's past-the-cache pools use it."""

    L1D = "host_l1d_total"
    L2 = "host_l2_total"
    L3 = "host_l3_total"


class SlotKind(StrEnum):
    """what a cold ring slot holds for one linear: the checkpoint's bf16 off the drive, a 12-bit store entry off the
    drive, bf16 the engine dequantizes into it each pass (a GGUF tensor no native gemv takes), a float of another
    precision off the drive rewritten as bf16 where it lands, or a GGUF tensor's own blocks for its type's gemv - off
    the drive, or gathered in memory where the converter reordered them"""

    BF16 = "bf16"
    P12 = "p12"
    MEM = "mem"
    CAST = "cast"
    QUANT = "quant"


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
    # the engine's attention kernels over the cache buffers (btb/mlx/attn.py: decode, tree, rows, forest, head-256
    # prefill), read from the buffer itself rather than the views an append hands back
    MLX_ATTN_KERNEL = "mlx_attn_kernel"
    CUDA_GRAPH = "cuda_graph"  # a captured card graph ran the pass (_forward_card_segment/_forward_fast/step graph)
    # a prompt's chunk through the card graph's kernels as they come, each row the row its step makes
    # (_forward_card_prefill)
    CARD_PREFILL = "card_prefill"
    CUDA_TORCH_FALLBACK = "cuda_torch_fallback"  # a card layer through torch modules, btb's kernels absent
    CPU_NATIVE = "cpu_native"  # a host layer through the native CPU gemv kernels
    MLX_PEROP = "mlx_perop"  # a host layer's linears through Native.mlx.linear (the non-fused MLX path: MoE, offload)
    # the stored weight path a linear read, on every tier: a bf16 slot, or a GGUF type multiplied in its own
    # blocks - one member a type of `stored_quants()`, valued "quant_<type>" (`quant_tag`), held equal to it by
    # tests/cert/test_runtime_guards.py
    QUANT_DEQUANT = "quant_dequant"  # a bf16 slot (float weights, or a GGUF dequantized to bf16)
    QUANT_BF16 = "quant_bf16"  # a GGUF's bf16 expert multiplied as stored (`EXPERT_FLOATS`)
    QUANT_F16 = "quant_f16"  # a GGUF's f16 expert multiplied as stored, widened exactly
    QUANT_Q4_0 = "quant_q4_0"
    QUANT_Q4_1 = "quant_q4_1"
    QUANT_Q8_0 = "quant_q8_0"
    QUANT_Q2_K = "quant_q2_k"
    QUANT_Q3_K = "quant_q3_k"
    QUANT_Q4_K = "quant_q4_k"
    QUANT_Q5_K = "quant_q5_k"
    QUANT_Q6_K = "quant_q6_k"
    QUANT_IQ4_NL = "quant_iq4_nl"
    QUANT_IQ4_XS = "quant_iq4_xs"
    QUANT_IQ3_XXS = "quant_iq3_xxs"
    QUANT_IQ3_S = "quant_iq3_s"
    QUANT_IQ2_XXS = "quant_iq2_xxs"
    QUANT_IQ2_XS = "quant_iq2_xs"
    QUANT_IQ2_S = "quant_iq2_s"
    QUANT_IQ1_S = "quant_iq1_s"
    QUANT_IQ1_M = "quant_iq1_m"
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
    # a GGUF's experts of another stored type: read into the store's slots as the file's blocks and multiplied by
    # their type's own kernel (each type's `quant_<type>` tag beside it), or dequantized to bf16 slots on read
    EXPERT_QUANT_ASSTORED = "expert_quant_asstored"
    EXPERT_QUANT_DEQUANT = "expert_quant_dequant"
    EXPERT_FP8_ASSTORED = "expert_fp8_asstored"  # FP8 experts multiplied as stored, e4m3 bytes and scale grid
    EXPERT_FP8_WIDENED = "expert_fp8_widened"  # FP8 experts widened by their scales to bf16 before the matmul
    # a prefill's expert call as grouped matmuls over the depot's slots on the card (`_card_grouped`)
    EXPERT_CARD_GROUPED = "expert_card_grouped"
    # the head, and the route a long prefill took (btb/engine/forward.py)
    HEAD_RESIDENT = "head_resident"  # the head multiplied where the model runs
    HEAD_STREAMED = "head_streamed"  # the head read from the checkpoint for the pass (`resident_head` off)
    PREFILL_CARD = "prefill_card"  # a prefill ran the host layers on the card (`prefill_card_min` rows or more)
    # the token draw (btb/sampling.py)
    SAMPLE_GREEDY = "sample_greedy"  # the argmax
    SAMPLE_STOCHASTIC = "sample_stochastic"  # the Gumbel-max draw under a temperature
    PICK_HOOKED = "pick_hooked"  # the pick over logits a caller's hooks saw: the in-graph picks stood aside
    ROWS_FLAT = "rows_flat"  # a fork's or a batch's rows on the MLX batched step, over one flat buffer
    ROWS_JOINED = "rows_joined"  # a fork's or a batch's rows on the torch pass, prefix and own rows joined
    ROWS_CARD = "rows_card"  # a fork's or a batch's rows on the card graph's rows pass, read in place in its arena
    # the reader a pass's attention took its cache rows through (btb/engine/state.py `_tag_kv`)
    KV_CONTIGUOUS = "kv_contiguous"  # the conversation's own buffers, its rows in order from the first
    KV_PAGED = "kv_paged"  # the engine's pool of pages, read through the conversation's row map (btb/engine/paged.py)
    KV_PARK = "kv_park"  # the conversation's pages brought back onto the card from pinned RAM (`CardRegion.bind`)
    # what a decode over a session kept of the cache for its prompt (btb/session.py `_begin_decode`)
    PREFIX_HIT = "prefix_hit"  # rows of the prompt the cache held: prefilled from there on
    PREFIX_MISS = "prefix_miss"  # nothing kept: the whole prompt prefilled
    PREFIX_SHARED = "prefix_shared"  # the prompt opened on rows the prefix tree holds, read in place (`_open_paged`)
    SNAPSHOT_RESUME = "snapshot_resume"  # a hybrid's recurrent states restored from a kept snapshot to resume
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
    # the programmatic API, one member per public method of a class `btb.api.api(owner)` declares, valued
    # "owner.method": the class refuses to build with a public method missing here, and each call records its member
    API_MODEL_PEAK_MEMORY = "model.peak_memory"
    API_MODEL_PROMPT_IDS = "model.prompt_ids"
    API_MODEL_GENERATE = "model.generate"
    API_MODEL_HIDDEN = "model.hidden"
    API_MODEL_PROJECT = "model.project"
    API_MODEL_ENCODE = "model.encode"
    API_MODEL_ASK = "model.ask"
    API_MODEL_STREAM = "model.stream"
    API_MODEL_SESSION = "model.session"
    API_MODEL_BATCH = "model.batch"
    API_MODEL_CHAT = "model.chat"
    API_MODEL_ASK_MANY = "model.ask_many"
    API_MODEL_RESERVE = "model.reserve"
    API_MODEL_EMPTY = "model.empty"
    API_MODEL_ZEROS = "model.zeros"
    API_MODEL_FULL = "model.full"
    API_MODEL_ROOM = "model.room"
    API_MODEL_MEMORY = "model.memory"
    API_ROOM_RELEASE = "room.release"
    API_ROOM_EMPTY = "room.empty"
    API_ROOM_ZEROS = "room.zeros"
    API_ROOM_FULL = "room.full"
    API_SESSION_FEED = "session.feed"
    API_SESSION_NEXT_LOGITS = "session.next_logits"
    API_SESSION_MARK = "session.mark"
    API_SESSION_REWIND = "session.rewind"
    API_SESSION_CROP = "session.crop"
    API_SESSION_ROWS = "session.rows"
    API_SESSION_SYNC = "session.sync"
    API_SESSION_FORK = "session.fork"
    API_SESSION_GENERATE = "session.generate"
    API_ROWS_TOKENS = "rows.tokens"
    API_ROWS_STEP = "rows.step"
    API_ROWS_ADVANCE = "rows.advance"
    API_ROWS_NEXT_LOGITS = "rows.next_logits"
    API_ROWS_LEAVE = "rows.leave"
    API_ROWS_GENERATE = "rows.generate"
    API_BRANCHES_REORDER = "branches.reorder"
    API_BRANCHES_KEEP = "branches.keep"
    API_BRANCHES_CLOSE = "branches.close"
    API_BATCH_JOIN = "batch.join"
    API_BATCH_CLOSE = "batch.close"


def quant_tag(q: Quant) -> PassTag:
    """the tag a linear of type `q` multiplied in its stored blocks records"""
    return PassTag(f"quant_{q.value.lower()}")


def api_tags(owner: str) -> frozenset[PassTag]:
    """the declared API calls of `owner` (a class's `api(owner)`), from the PassTag values"""
    return frozenset(t for t in PassTag if t.value.partition(".")[0] == owner and "." in t.value)


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


# --- card kernel roles: written by `python build.py bindings` from native/cuda/btb_roles.cuh ---
# Each card kernel parameter's role as its own type (btb/engine/card_launch.py types the launchers with
# them): a pointer role's value is a card address of its element, a scalar role's the value itself. So two
# of a launch's arguments swapped, or a value of one role where another goes, fail the type check.

# --- a pass's rows ---
# int: T: the rows a launch carries (a pass's tokens, a matvec's M)
Rows = NewType("Rows", int)
# int*: [1]: the cache's rows before the pass, read on the card at each replay
PrefixLen = NewType("PrefixLen", int)
# int*: [T]: each row's depth in the pass's tree
Depths = NewType("Depths", int)
# int*: [T]: each row's parent in the pass's tree, -1 the prefix, -2 padding; null a chain
TreeParents = NewType("TreeParents", int)
# int*: [3 + 3 T]: the rows as T sequences a token each (base, step, W, then col, off, len)
RowsLayout = NewType("RowsLayout", int)
# int*: [cap]: the cache's row map, a logical row to its slot in its page; null none
RowMap = NewType("RowMap", int)

# --- matvecs and GEMMs ---
# bf16*: [R, C] row-major: a matvec's or GEMM's weight
Weight = NewType("Weight", int)
# bf16*: [T, C]: the rows a matvec or GEMM reads
MatIn = NewType("MatIn", int)
# bf16*: [T, R]: the rows it writes
MatOut = NewType("MatOut", int)
# float*: [T, C] float32: the host gemv's rows, read by its card twin
MatInF32 = NewType("MatInF32", int)
# float*: [T, R] float32: the rows the host gemv's card twin writes
MatOutF32 = NewType("MatOutF32", int)
# int: R: the weight's rows, the output's width
OutDim = NewType("OutDim", int)
# int: C: the weight's columns, the input's width
InDim = NewType("InDim", int)
# bf16*: [T, 2 I]: the MLP's gate then up projection
GateUp = NewType("GateUp", int)
# bf16*: [T, I]: act(gate) * up, the down projection's input
MlpAct = NewType("MlpAct", int)
# int: I: the MLP's intermediate width
Intermediate = NewType("Intermediate", int)
# unsigned char*: [R, C / block * bytes]: a GGUF type's blocks as the file holds them
QuantBlocks = NewType("QuantBlocks", int)
# signed char*: a lattice type's int8 codebook; null for the rest
QuantGrid = NewType("QuantGrid", int)
# unsigned char*: the lattice types' 128-entry sign table; null for the rest
QuantSigns = NewType("QuantSigns", int)
# int: the elements between two rows of a matvec's output (a merged projection's piece)
OutStride = NewType("OutStride", int)
# bf16*: [R, C]: a GGUF type's weight as bf16
Dequant = NewType("Dequant", int)
# long long: R * C: the values a dequant writes
QuantValues = NewType("QuantValues", int)
# unsigned char*: [seats, groups, 16]: MXFP4 experts' 32-value blocks, two fp4 codes a byte
Mx4Blocks = NewType("Mx4Blocks", int)
# unsigned char*: [seats, groups]: each block's power-of-two scale, biased by 127
Mx4Scales = NewType("Mx4Scales", int)
# int*: [n]: the experts' seats in a depot's stacks
Seats = NewType("Seats", int)
# long long: the 32-value blocks of one expert
Mx4Groups = NewType("Mx4Groups", int)
# bf16*: [n, groups * 32]: the experts widened to bf16
Widened = NewType("Widened", int)
# int*: [1]: the share of each weight row warmed into L2, in 256ths, read on the card
WarmShare = NewType("WarmShare", int)
# int*: [1]: where the warm-up's loads land, so they are kept
WarmSink = NewType("WarmSink", int)
# int*: [1]: the tensor cores' self-check's mismatches, zeroed before it
MmaMisses = NewType("MmaMisses", int)

# --- norms, rope and the cache ---
# bf16*: [T, H]: the residual stream, added to in place
Residual = NewType("Residual", int)
# bf16*: [T, H]: a block's output, added to the residual; null adds nothing
BlockOut = NewType("BlockOut", int)
# bf16*: [H]: an RMSNorm's weight
NormWeight = NewType("NormWeight", int)
# float: an RMSNorm's epsilon
NormEps = NewType("NormEps", float)
# int: 1: the norm's weight is zero-centred (times 1 + w); 0: times w
NormCentered = NewType("NormCentered", int)
# bf16*: [T, H]: an RMSNorm's output
Normed = NewType("Normed", int)
# int: H: the hidden width
Hidden = NewType("Hidden", int)
# bf16*: [T, (Hq + 2 Hk) D]: the merged q, k and v projection's rows
Qkv = NewType("Qkv", int)
# int: a Qkv row's width, in elements
QkvWidth = NewType("QkvWidth", int)
# int: the elements between two query heads of a Qkv row
QHeadStride = NewType("QHeadStride", int)
# int: where a Qkv row's keys start
KColumn = NewType("KColumn", int)
# int: where a Qkv row's values start
VColumn = NewType("VColumn", int)
# bf16*: [D]: the query heads' norm weight; null no norm
QNorm = NewType("QNorm", int)
# bf16*: [D]: the key heads' norm weight; null no norm
KNorm = NewType("KNorm", int)
# bf16*: [positions, ts]: the model's rotary cosines
RopeCos = NewType("RopeCos", int)
# bf16*: [positions, ts]: the model's rotary sines
RopeSin = NewType("RopeSin", int)
# int: ts: a rotary table's row width
RopeStride = NewType("RopeStride", int)
# int: the dims a head's rope turns; the rest pass through
RopeDims = NewType("RopeDims", int)
# int: 1: the keys go into the cache as projected (the indexer's raw keys)
KeepRawKey = NewType("KeepRawKey", int)
# bf16*: [T, Hq, D]: the query heads, normed and roped
Query = NewType("Query", int)
# bf16*: a layer's key rows: head g's row j at g * hs + j * rs
KCache = NewType("KCache", int)
# bf16*: a layer's value rows, laid out as its keys
VCache = NewType("VCache", int)
# int: Hq: the query heads
QHeads = NewType("QHeads", int)
# int: Hk: the key and value heads
KvHeads = NewType("KvHeads", int)
# int: hs: the elements between two heads of the cache
CacheHeadStride = NewType("CacheHeadStride", int)
# int: rs: the elements between two rows of the cache
CacheRowStride = NewType("CacheRowStride", int)

# --- attention ---
# bf16*: [T, Hq, D]: the attention's output
AttnOut = NewType("AttnOut", int)
# float: the scores' scale, 1 / sqrt(D) unless the model says otherwise
AttnScale = NewType("AttnScale", float)
# int: a sliding layer's window, in keys; 0 the whole cache
Window = NewType("Window", int)
# float*: [S, T, Hq]: each split's running max, for the last block to fold
PartMax = NewType("PartMax", int)
# float*: [S, T, Hq]: each split's running sum of exponentials
PartSum = NewType("PartSum", int)
# float*: [S, T, Hq, D]: each split's running output
PartAcc = NewType("PartAcc", int)
# int*: [T, Hq]: the splits arrived so far, zero between launches
Arrivals = NewType("Arrivals", int)
# int: n0: a prompt chunk's first position
ChunkStart = NewType("ChunkStart", int)
# float*: [T, Hq, D]: each row's running output across its key groups
PrefillRun = NewType("PrefillRun", int)
# bf16*: the output gate's logits, at their own place in the query projection's rows
OutGate = NewType("OutGate", int)
# int: the elements between two rows of the gate
GateRowStride = NewType("GateRowStride", int)
# int: the elements between two heads of the gate
GateHeadStride = NewType("GateHeadStride", int)
# int: D: a head's width
HeadDim = NewType("HeadDim", int)
# bf16*: [T, Hq D]: the attention's output times sigmoid(gate)
GatedAttn = NewType("GatedAttn", int)
# int: Hq D: an attention output row's width
AttnWidth = NewType("AttnWidth", int)

# --- Qwen4's sparse attention's indexer ---
# bf16*: [T, Hi, DI]: the indexer's query heads
IdxQuery = NewType("IdxQuery", int)
# bf16*: [cap, DI]: the indexer's raw keys, a row a position
IdxRawKey = NewType("IdxRawKey", int)
# bf16*: [NBcap, DI]: the committed complete blocks' pooled keys
IdxPooled = NewType("IdxPooled", int)
# int*: [2]: the pooled blocks, then the pool's arrival count
IdxPooledLen = NewType("IdxPooledLen", int)
# bf16*: [DI]: the indexer's key norm weight
IdxKNorm = NewType("IdxKNorm", int)
# int: Hi: the indexer's heads
IdxHeads = NewType("IdxHeads", int)
# int: r: the positions a block pools
BlockRatio = NewType("BlockRatio", int)
# int: the blocks a row keeps
TopBlocks = NewType("TopBlocks", int)
# int: NBcap: the pooled keys' rows
PooledCap = NewType("PooledCap", int)
# int: the most blocks a pool adds
MaxNewBlocks = NewType("MaxNewBlocks", int)
# float*: the indexer's block scores, its scratch
IdxScores = NewType("IdxScores", int)
# int*: [T, ktop]: each row's picked blocks
PickedBlocks = NewType("PickedBlocks", int)
# int*: [T]: each row's picked block count
PickedCount = NewType("PickedCount", int)

# --- the step's handoff to the host ---
# long long*: [1]: the step's token, on the card
StepToken = NewType("StepToken", int)
# long long*: pinned host memory: the step's token
HostToken = NewType("HostToken", int)
# int*: pinned host memory: the cache's length after the step
HostLen = NewType("HostLen", int)
# long long*: pinned host memory: the card's clock at the step's end, ns
HostClock = NewType("HostClock", int)

# --- the gated DeltaNet ---
# float*: [T, C]: the q | k | v projection before its conv
DeltaMixed = NewType("DeltaMixed", int)
# float*: [T, Hv dv]: the output gate's input
DeltaZ = NewType("DeltaZ", int)
# float*: [T, Hv]: the decay's input
DeltaA = NewType("DeltaA", int)
# float*: [T, Hv]: the write strength's input
DeltaB = NewType("DeltaB", int)
# bf16*: [T, ps]: the card program's merged projection, the nodes' inputs as its columns
DeltaProj = NewType("DeltaProj", int)
# int: ps: a DeltaProj row's width
ProjStride = NewType("ProjStride", int)
# int: where a DeltaProj row's q | k | v starts
QkvColumn = NewType("QkvColumn", int)
# int: where a DeltaProj row's z starts
ZColumn = NewType("ZColumn", int)
# int: where a DeltaProj row's b starts
BColumn = NewType("BColumn", int)
# int: where a DeltaProj row's a starts
AColumn = NewType("AColumn", int)
# int*: [T]: the DeltaProj row a node reads; null row j; negative ends the pass
ProjRows = NewType("ProjRows", int)
# float*: [C, K]: the conv's weight
ConvWeight = NewType("ConvWeight", int)
# float*: [C]: the conv's bias; null none
ConvBias = NewType("ConvBias", int)
# float*: [C, K]: the conv's window
ConvState = NewType("ConvState", int)
# float*: [Hv, dk, dv]: the recurrent state
DeltaState = NewType("DeltaState", int)
# float*: [slots, Hv, dk, dv]: the tree's node states; null steps the state in place
DeltaScratch = NewType("DeltaScratch", int)
# int*: [T]: the scratch slot a node's state goes to; null a slot a node
ScratchSlots = NewType("ScratchSlots", int)
# float*: [Hv]: the decay's log rate
DeltaALog = NewType("DeltaALog", int)
# float*: [Hv]: the decay's time-step bias
DeltaDtBias = NewType("DeltaDtBias", int)
# float*: [dv]: the output norm's weight
DeltaNormWeight = NewType("DeltaNormWeight", int)
# int: the output gate: 1 sigmoid, else silu
DeltaGate = NewType("DeltaGate", int)
# int: C: a conv's channels
ConvChannels = NewType("ConvChannels", int)
# int: K: a conv's taps
ConvTaps = NewType("ConvTaps", int)
# int: a conv's dilation
ConvDilation = NewType("ConvDilation", int)
# int: the DeltaNet's key heads
DeltaKHeads = NewType("DeltaKHeads", int)
# int: Hv: the DeltaNet's value heads
DeltaVHeads = NewType("DeltaVHeads", int)
# int: dk: a key head's width
DeltaKDim = NewType("DeltaKDim", int)
# int: dv: a value head's width
DeltaVDim = NewType("DeltaVDim", int)
# float*: [T, Hv dv] float32: the DeltaNet's output
DeltaOutF32 = NewType("DeltaOutF32", int)
# bf16*: [T, Hv dv]: the DeltaNet's output, rounded to bf16
DeltaOut = NewType("DeltaOut", int)

# --- Qwen4's hyper-connection streams ---
# bf16*: [T, G, H]: the streams, added to in place
Streams = NewType("Streams", int)
# bf16*: [T, G]: the block's inject weights
Inject = NewType("Inject", int)
# int: G: the streams
StreamCount = NewType("StreamCount", int)
# bf16*: [T, ds]: the mixer's down projection rows (then the inject logits)
MixDown = NewType("MixDown", int)
# int: ds: a MixDown row's width
MixDownStride = NewType("MixDownStride", int)
# bf16*: [T, R]: silu(down / G), the mixer's low-rank activation
MixAct = NewType("MixAct", int)
# int: R: the mixer's rank
MixRank = NewType("MixRank", int)
# bf16*: [T, G, H]: the mixer's up projection's logits
MixUp = NewType("MixUp", int)
# bf16*: [T, H]: the streams mixed
Mixed = NewType("Mixed", int)

# --- Qwen4's MoE ---
# bf16*: [T, ls]: the experts' logits, then the shared expert's gate logit
RouterLogits = NewType("RouterLogits", int)
# int: ls: a RouterLogits row's width
RouterStride = NewType("RouterStride", int)
# int: E: the routed experts
Experts = NewType("Experts", int)
# int: k: the experts a row takes
TopExperts = NewType("TopExperts", int)
# int*: [T, k]: each row's experts
ExpertIds = NewType("ExpertIds", int)
# bf16*: [T, k]: each row's expert weights
ExpertWeights = NewType("ExpertWeights", int)
# int*: pinned host memory: each row's experts; null none
HostExpertIds = NewType("HostExpertIds", int)
# bf16*: pinned host memory: each row's expert weights
HostExpertWeights = NewType("HostExpertWeights", int)
# unsigned*: [1]: the router's blocks arrived, zero between launches
RouteCount = NewType("RouteCount", int)
# int*: [1]: the routing's sequence number, on the card
RouteSeq = NewType("RouteSeq", int)
# int*: pinned host memory: the sequence number of the last rows out; null none
HostRouteSeq = NewType("HostRouteSeq", int)
# bf16*: [T, H]: the routed experts' sum
Routed = NewType("Routed", int)
# bf16*: [T, H]: the shared expert's output
SharedOut = NewType("SharedOut", int)
# bf16*: the shared expert's gate logit, row t at t * ls
SharedGate = NewType("SharedGate", int)
# bf16*: [T, H]: the MoE's output; may be Routed's buffer
MoeOut = NewType("MoeOut", int)

# --- Qwen4's per-layer n-gram embedding ---
# bf16*: [T, ks]: the n-gram embeddings' key then value projection
PleKv = NewType("PleKv", int)
# int: ks: a PleKv row's width
PleKvStride = NewType("PleKvStride", int)
# bf16*: [T, G H]: the streams through the query norm
PleQuery = NewType("PleQuery", int)
# bf16*: [G H]: the key norm's weight
PleKNorm = NewType("PleKNorm", int)
# bf16*: [G H]: the conv norm's weight
PleConvNorm = NewType("PleConvNorm", int)
# float: 1 / sqrt(H)
InvSqrtHidden = NewType("InvSqrtHidden", float)
# bf16*: [T, G H]: sigmoid(gate) * value
PleGated = NewType("PleGated", int)
# bf16*: [T, G H]: the gated rows through the conv norm
PleNormed = NewType("PleNormed", int)
# bf16*: [C, K]: the dilated conv's weight
PleConvWeight = NewType("PleConvWeight", int)
# bf16*: [C, (K - 1) dil]: the conv's kept window, oldest first
PleWindow = NewType("PleWindow", int)
# int: 1: a one-row step shifts its row into the window
PleUpdate = NewType("PleUpdate", int)

# --- sampling ---
# float*: [R, V] float32: the rows' logits
Logits = NewType("Logits", int)
# int: V: a Logits row's width
Vocab = NewType("Vocab", int)
# float*: [2]: 1 / temperature, top_p
SampleParams = NewType("SampleParams", int)
# unsigned*: [R]: each row's max scaled logit, as ordered bits
SampleMax = NewType("SampleMax", int)
# int: the radix level, 0 to 2
RadixLevel = NewType("RadixLevel", int)
# int: 0 counts (top-k), 1 masses (top-p)
SampleMode = NewType("SampleMode", int)
# unsigned*: [R]: the threshold resolved so far, as ordered bits
RadixPrefix = NewType("RadixPrefix", int)
# unsigned*: [R]: the floor a token must clear, as ordered bits
RadixFloor = NewType("RadixFloor", int)
# unsigned long long*: [R, 2048]: a level's histogram
RadixHist = NewType("RadixHist", int)
# unsigned long long*: [R]: the remainder a level carries to the next
RadixCarry = NewType("RadixCarry", int)
# int: the tokens top-k keeps; 0 all
TopK = NewType("TopK", int)
# unsigned*: [R]: the top-k threshold, as ordered bits
TopKFloor = NewType("TopKFloor", int)
# unsigned*: [R]: the top-p threshold, as ordered bits
TopPFloor = NewType("TopPFloor", int)
# unsigned long long*: [R]: each row's best draw, packed
SampleBest = NewType("SampleBest", int)
# unsigned*: [R]: each row's token
SampledTokens = NewType("SampledTokens", int)
# unsigned*: [2]: the seed, low word first
NoiseSeed = NewType("NoiseSeed", int)
# int*: [1]: the position the noise is keyed to
NoisePos = NewType("NoisePos", int)
# int: the key's salt; 0 none
NoiseSalt = NewType("NoiseSalt", int)
# unsigned*: [R, 2]: each row's noise key, low word first
NoiseKeys = NewType("NoiseKeys", int)
# float*: [T, Vd]: the drafter's distribution a node
DraftProbs = NewType("DraftProbs", int)
# int*: [T, C]: each node's drafted children in draw order, -1 padding
DraftKids = NewType("DraftKids", int)
# unsigned*: [T]: 0 where the drafts are point masses
HasDraftProbs = NewType("HasDraftProbs", int)
# int: Vd: a DraftProbs row's width
DraftVocab = NewType("DraftVocab", int)
# int: C: a node's drafted children
DraftWidth = NewType("DraftWidth", int)
# unsigned*: [T]: each node's pick, (slot + 1) << 24 | token
VerifyPicks = NewType("VerifyPicks", int)
# --- end of the card kernel roles ---
