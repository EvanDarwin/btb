# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4 (`qwen4_exp`): hyper-connected residual streams closed by a gated mixer, a sparse attention that picks
its key blocks through an indexer, gated DeltaNet layers, a per-layer n-gram embedding and a mixture of experts.
Its layers take the pass whole, so the engine hands them their own keywords and drives their node steps."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Self

import torch

from ....kinds import FamilyKind, LayerKind
from ....mxfp4 import stored_mxfp4
from ...host import _Experts, _NGramRows
from ...native import Native
from ..base import Family, _flags

if TYPE_CHECKING:
    from ...drafter import MTPDrafter
    from ...state import _State


class Qwen4Family(Family):
    """Qwen4's layers: the experts and the n-gram embedding's rows through the engine's stores, the indexer's
    block selection and the DeltaNet and n-gram steps the engine's own (`qsa.py`, `verify.py`), and on the host
    the router, the attention and the activations row-invariant (`router.py`, `attend.py`, `rows.py`)"""

    KIND = FamilyKind.QWEN4

    @classmethod
    def build(cls, cfg: Any) -> Self:
        mod = importlib.import_module("transformers.models.qwen4_exp.modeling_qwen4_exp")
        # streams (hc_count) is per-config runtime; the sparse attention is the family's own
        return cls(
            kind=cls.KIND,
            mod=mod,
            layer=mod.Qwen4ExpTextDecoderLayer,
            norm=None,
            rotary=mod.Qwen4ExpTextRotaryEmbedding,
            streams=int(cfg.hc_count),
            attn=LayerKind.QWEN_SPARSE,
            **_flags(cls.KIND),
        )

    def dense_key(self, key: str) -> bool:
        # the n-gram embedding's table is read a row at a time as a pass asks for them (`_NGramRows`)
        return ".ngram_embedding." not in key and super().dense_key(key)

    def widened(self, name: str) -> bool:
        # the router's matrix stays as stored: `finish_host_layer` multiplies it through the host's gemv (router.py)
        return not name.endswith("mlp.gate.weight") and super().widened(name)

    def finish_host_layer(self, sm: _State, layer: Any, i: int, base: str | None = None) -> None:
        # the router, the sparse attention and the activations row-invariant on the host, so a verify pass's node
        # computes as the one-token step of its path: the router's logits through the host's gemv (router.py), the
        # attention through the native kernel over each row's own list of cache rows (attend.py), the activations a
        # row at a time (rows.py). `base` as `shape_layer`'s
        from .attend import install as install_attention
        from .router import install as install_router
        from .rows import install as install_rows

        install_router(layer.mlp, (f"{sm.prefix}layers.{i}." if base is None else base) + "mlp.gate.weight")
        if getattr(layer, "self_attn", None) is not None:
            install_attention(layer.self_attn, sm)
        install_rows(layer, sm)

    def shape_layer(self, sm: _State, layer: Any, i: int, base: str | None = None) -> Any:
        # `i` is the layer's id in the expert store and `base` its checkpoint prefix, the trunk's layer `i` unless
        # named: the drafter's layer reads its experts under `mtp.layers.0.` as the store's layer `L` (drafter.py)
        base = f"{sm.prefix}layers.{i}." if base is None else base
        ex = layer.mlp.experts
        layer.mlp.experts = _Experts(
            sm,
            base + "mlp.experts.",
            ex.num_experts,
            ex.act_fn,
            layer=i,
            mx=stored_mxfp4(self.mxfp4, sm.gguf),
            f8=sm.fp8_experts,
        )
        if getattr(layer, "ple", None) is not None:
            out_dtype = sm.compute_dtype if sm.compute_dtype is not None else torch.bfloat16
            layer.ple.ple_embedding.ngram_embedding = _NGramRows(
                sm, base + "ple.ple_embedding.ngram_embedding.", int(sm.cfg.split_ngram_parts), out_dtype
            )
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if indexer is not None:
            # the sparse attention's block selection without the reference's per-query rebuild of the keys: the
            # reference's mask bit for bit, or with `sparse` the rows scored in one pass (qsa.py)
            from .qsa import install

            install(indexer, sparse=bool(getattr(sm, "sparse", False)))
            # a resident layer's attention over rows the card program keeps in RAM through the program's kernels
            # (ram.py); a host layer's takes attend.py's over it (`finish_host_layer`)
            from .ram import install as install_ram

            install_ram(layer.self_attn, sm)
        # the DeltaNet and the n-gram embedding stepped node by node: the one-token step and a verify pass alike
        from .verify import install as install_steps

        install_steps(layer, sm)
        return layer

    def closing(self, cfg: Any) -> tuple[str, torch.nn.Module]:
        # the streams mixed back into one in the final norm's place
        return "hyper_connection_mixer", self.mod.Qwen4ExpTextGatedResidual(cfg, use_combine=False)

    def layer_kw(self, lt: str, causal: Any, linear_mask: Any, pos: torch.Tensor, ple_ids: Any) -> dict[str, Any]:
        # the layer takes the pass's one mask whatever its type, the rows' padding as its convolution's mask, and
        # the ids its n-gram embedding reads
        return {"attention_mask": causal, "conv_mask": linear_mask, "ple_input_ids": ple_ids}

    def ple_ids(self, cfg: Any, ids: torch.Tensor, linear_mask: torch.Tensor | None) -> torch.Tensor | None:
        # a padded position hands the n-gram embedding the end-of-text id in its token's place
        if linear_mask is None:
            return ids
        eos = cfg.eos_token_id
        eos = eos[0] if isinstance(eos, (list, tuple)) else eos
        return torch.where(linear_mask.bool(), ids, torch.full_like(ids, int(eos)))

    def card_program(self) -> type[Any] | None:
        # the resident layers' step and verify pass through the row-invariant card kernels (card.py)
        from .card import Qwen4Card

        return Qwen4Card

    def attn_index_bytes(self, cfg: Any, rows: int) -> tuple[int, int]:
        # the sparse attention's indexer: its pooled keys, one a block of `indexer_compress_ratio` rows, on the card
        # whatever the rows' home (every pass scores them all: a million positions' worth read over the bus a pass
        # was milliseconds a layer); a raw key a row, beside the keys and values (card.py `arena`)
        r = int(getattr(cfg, "indexer_compress_ratio", 0) or 0)
        di = int(getattr(cfg, "indexer_head_dim", 0) or 0)
        if r <= 0 or di <= 0:
            return 0, 0
        return (int(rows) // r + 1) * di * 2, int(rows) * di * 2

    def drafter_cls(self) -> type[MTPDrafter] | None:
        from .drafter import Qwen4Drafter

        return Qwen4Drafter

    def speculates(self, mlx: bool) -> bool:
        # a verify pass runs through the layers' node steps (verify.py), which the MLX tier does not: there the
        # decode is the plain loop. Elsewhere whether a pass verifies exactly is asked a pass at a time
        # (`verify_exact`)
        return not mlx

    def verify_exact(self, sm: _State, cache: Any) -> bool:
        # a node computes as its path's step only through btb's own kernels: on the host the DeltaNet step and the
        # attention over each row's cache rows (verify.py, attend.py; without them a host layer runs the reference
        # modules, which step a tree's rows as one chain); every other layer only inside the card program, whose
        # own gate (`_card_program`, the program's `why_not`) says whether it takes this cache's pass - where it
        # declines (a float32 compute, the KV on the host, a streamed layer, a fork, ...) the layers run the
        # reference router, attention and activations, which are not row-invariant
        if sm.host and (Native.delta_step is None or Native.attn_nodes is None):
            return False
        if all(i in sm.host for i in range(int(sm.L))):
            return True
        return sm._card_program(cache, 1, 1, cache.get_seq_length(), None, None, None) is not None
