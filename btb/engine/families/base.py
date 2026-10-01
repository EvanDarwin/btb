# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`Family`, what the engine knows of a model family and the part of its behaviour the engine asks it for: the
capability flags the paths key on, the transformers classes it builds layers from, and the methods a family
overrides where its layers, its closing module, its checkpoint or its drafting head are its own. Each family is a
subclass in its own package beside this module; the base is the plain pre-norm block, what a family that
overrides nothing runs as."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypedDict, cast

import torch

from ...kinds import CAPS, FAMILY_NAMES, Cap, FamilyKind, LayerKind, PassTag

if TYPE_CHECKING:
    from ..drafter import MTPDrafter
    from ..state import _State


class Flags(TypedDict):
    """the capability booleans a family's `build` spreads into it, one field per `Cap` (the field name is the
    Cap's value). Typed so the spread checks against Family's fields; the field set is held equal to `Cap` by
    tests/cert/test_runtime_guards.py, which `_flags()` below builds from."""

    dense: bool
    kernel_layout: bool
    hybrid: bool
    moe: bool
    mxfp4: bool
    mrope: bool
    attn_gate: bool
    own: bool
    norm_centered: bool
    embed_scale: bool
    dual_rope: bool
    sandwich: bool
    flat_cache: bool
    eager: bool
    fast: bool


def _flags(kind: FamilyKind) -> Flags:
    """the Family capability flags for a family, projected from `CAPS` - the single source of which caps a family
    has. A family's `build` adds the transformers classes (the runtime); the booleans are read here so they never
    drift."""
    caps = CAPS[kind]
    # one entry per Cap, keyed by the Cap's value (the Family field name), so Flags cannot fall behind the enum
    return cast(Flags, {c.value: c in caps for c in Cap})


@dataclass(frozen=True)
class Family:
    """What the engine knows of a model family: its kind, the transformers module and classes it builds layers
    from, and the capabilities the paths key on (never the kind): `dense` - the pre-norm block of GQA attention
    with rotary and a gated MLP, so the fused MLX forward, the tree verify, the batched and pipelined loops and
    the card graph apply; `hybrid` - linear-attention layers through the DeltaNet step and the hybrid forward;
    `norm_centered` - RMSNorm scales by 1 + weight; `kernel_layout` - separate q/k/v/o with q/k norms and
    gate/up/down, the layout the compiled kernels (the MLX megakernel and fused step, the CUDA step graph) are
    written for; `own`: the engine drives the layer; `fast`: the per-position host paths know the attention;
    `moe`: experts through the store; `mxfp4`: as MXFP4; `eager`: the module's own attention; `flat_cache`:
    one cache row per layer; `embed_scale`: the input embedding is scaled by sqrt(hidden_size) (Gemma);
    `dual_rope`: rope frequencies differ by layer type, so the pass carries one rope per type and each layer
    reads its own (Gemma 3's local/global split); `sandwich`: the block norms the attention and MLP outputs
    before their residual add (four norms a layer) rather than the pre-norm block's two, so the fused MLX path
    takes its own norm ordering. Fused q/k/v and gate/up projections are read off the module, not declared.

    The flags are read where a path forks on what a family can do; the methods below are where a family's own
    code runs, each overridden by the family whose layers, checkpoint or drafting head need it. `build` makes a
    family's instance from a config; the base built directly (`Family(kind=...)`) is a family of flags alone,
    the plain block's behaviour."""

    # the family a subclass builds, the key `family()` finds it under
    KIND: ClassVar[FamilyKind]

    kind: FamilyKind
    mod: Any = None
    layer: Any = None
    norm: Any = None
    rotary: Any = None
    mrope: bool = False
    attn_gate: bool = False
    own: bool = False
    streams: int = 1
    attn: str = LayerKind.FULL
    fast: bool = True
    moe: bool = False
    mxfp4: bool = False
    eager: bool = False
    flat_cache: bool = False
    dense: bool = False
    hybrid: bool = False
    norm_centered: bool = False
    kernel_layout: bool = False
    embed_scale: bool = False
    dual_rope: bool = False
    sandwich: bool = False

    @classmethod
    def build(cls, cfg: Any) -> Self:
        """the family of `cfg`: its transformers module imported (the engine's fused forwards installed on its
        classes where they apply), its layer, norm and rotary classes, and its flags from `CAPS`"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """the name a user would recognize the family by"""
        return FAMILY_NAMES[self.kind]

    # the paths the family's block can take, from its flags: the engine's gates read these, and so can a caller
    # asking which rows a family runs (the fixture's shapes and the load's options decide the rest)

    @property
    def card_graph(self) -> bool:
        """the card graph's kernels are written for the family's block: the kernel layout or the sandwich one, in a
        family that does not run its layers as its own (`own`); cuda.py `_card_family_ok` checks the activation,
        the tier and the shapes after"""
        return (self.kernel_layout or self.sandwich) and not self.own

    @property
    def chunk_causal(self) -> bool:
        """the family's layers hand the pass's mask to the engine's sdpa untouched: the plain dense block, neither run
        as the engine's own (`own`) nor through its module's own attention (`eager`) - so a prefill chunk's causal
        mask can go to them as the rule it is (`attention.ChunkCausal`) and never be built"""
        return self.dense and not self.own and not self.eager

    @property
    def fused_step(self) -> bool:
        """the MLX step's fused kernels (the q/k norms, the rope and the attention in one) are written for the
        family's block: the kernel layout or the sandwich one; the step takes them for a rotary over the whole head
        in a bf16 state (mlx_forward.py `_forward_mlx`)"""
        return self.kernel_layout or self.sandwich

    @property
    def mega(self) -> bool:
        """the MLX megakernel is written for the family's block: the kernel layout, not the sandwich's norm order
        (mlx/mega.py; `_mega_ok` checks the cache, the storage and the pass after)"""
        return self.kernel_layout and not self.sandwich

    @property
    def mlx_path(self) -> PassTag:
        """the MLX forward the family's layers take: the per-token graph (a dense or sandwich block), the hybrid's
        DeltaNet forward, or else each host layer's linears through the per-op path (a mixture's)"""
        if self.dense or self.sandwich:
            return PassTag.MLX_STEP
        if self.hybrid:
            return PassTag.MLX_HYBRID
        return PassTag.MLX_PEROP

    def speculates(self, mlx: bool) -> bool:
        """whether a speculative pass verifies for this family on the MLX tier (`mlx`) or on the others: every
        tier verifies the plain block's"""
        return True

    def verify_exact(self, sm: _State, cache: Any) -> bool:
        """whether the next speculative pass over `cache` verifies its rows exactly - each node as the one-token step
        of its path - on the paths the engine would run it down now; the speculative loop takes a pass that would not
        as a plain one-row pass. Asked a pass at a time, so a load that `speculates` keeps every pass that can verify.
        The plain block's verifies on every path it speculates on"""
        return True

    def card_program(self) -> type[Any] | None:
        """the class of the family's own card program - its layers' step and verify pass as graphs over btb's card
        kernels, the engine's runner replaying them in turn (cuda.py `_forward_card_program`) - or None where the
        family's layers take the card graph or the torch path; the plain block's take those. A program answers
        `ok()` (whether it runs the model as placed now), `let_go()` (its weight blocks and graphs dropped before a
        shed, read again as asked) and `close()`"""
        return None

    def drafter_cls(self) -> type[MTPDrafter] | None:
        """the drafter over the checkpoint's `mtp.*` drafting head, or None for a family btb builds none for (the
        plain block's checkpoints carry no such head)"""
        return None

    def attn_index_bytes(self, cfg: Any, rows: int) -> tuple[int, int]:
        """what a resident attention layer's index adds to its cache at `rows` positions, as the placement prices
        it: (the bytes that stay on the card wherever the rows live - what every pass reads whole; the bytes that
        live with the rows). The plain block's attention keeps no index: (0, 0)"""
        return 0, 0

    def prepare(self, sm: _State) -> None:
        """what the family sets on the engine's config and on its transformers module once the engine knows its
        device, before any layer is made: nothing for the plain block"""

    def dense_key(self, key: str) -> bool:
        """whether a layer's checkpoint tensor `key` is read with the layer: not an FP8 tensor's scale, read with
        the tensor it scales (btb/fp8.py), nor a mixture's expert, which streams through the store"""
        if key.endswith(("_scale_inv", ".weight_scale")):
            return False
        return ".mlp.experts." not in key

    def widened(self, name: str) -> bool:
        """whether a host layer's matrix `name` (its name in the layer) is read widened to float32 and its module
        run in float32, the way transformers runs it: the conv, a mixture's router, and gpt-oss's per-expert biases
        (a few MB, added in float32). A family whose router multiplies through the host's own kernels keeps it as
        stored"""
        return "conv1d" in name or ".experts." in name or name.endswith("mlp.gate.weight")

    def shape_layer(self, sm: _State, layer: Any, i: int) -> Any:
        """layer `i` as the engine runs it, made on the meta device and not yet loaded: the family's modules the
        engine replaces with its own (a mixture's experts through the store, say) swapped in. The plain block's
        layer runs as transformers made it"""
        return layer

    def finish_host_layer(self, sm: _State, layer: Any, i: int) -> None:
        """the family's own part of host layer `i`, once its tensors are read and its linears are the host's: the
        plain block's has none"""

    def closing(self, cfg: Any) -> tuple[str, torch.nn.Module]:
        """the module the last layer's rows pass through before the head, made on the current device (the engine
        makes it on the meta device and reads its tensors in after), and the name its tensors sit under in the
        checkpoint: the family's final norm"""
        return "norm", self.norm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def layer_kw(self, lt: str, causal: Any, linear_mask: Any, pos: torch.Tensor, ple_ids: Any) -> dict[str, Any]:
        """the keywords a layer of type `lt` takes beside its rows, rope and cache: its mask - the causal one (its
        own type's where the pass made one a type, as gpt-oss's alternating windows need), or a linear-attention
        layer's padding - and the rows' positions"""
        if isinstance(causal, dict):
            causal = causal.get(lt)
        return {"attention_mask": linear_mask if lt == LayerKind.LINEAR else causal, "position_ids": pos}

    def ple_ids(self, cfg: Any, ids: torch.Tensor, linear_mask: torch.Tensor | None) -> torch.Tensor | None:
        """the token ids a pass hands its layers' per-layer embedding, where the family's layers have one; None
        for the plain block"""
        return None

    def open_sweep(self, sm: _State, C: int, keys: int, B: int) -> bool:
        """take what the family's attention holds for a layer-by-layer prefill on the card, sized for its last
        chunk of `C` rows of `B` against `keys` keys; whether it took anything, which `close_sweep` then lets go.
        The plain block's attention holds nothing across chunks"""
        return False

    def close_sweep(self, sm: _State) -> None:
        """let go of what `open_sweep` took"""
