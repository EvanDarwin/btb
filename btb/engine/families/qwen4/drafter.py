# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's MTP drafter: its drafting head as the step the shared drafter (btb/engine/drafter.py) grows its trees
from. No reference implementation of the head exists; its layout is read off the checkpoint's `mtp.*` tensors and
Qwen's architecture diagram:

    x_s = fc_hidden(norm_h(h)_s) + fc_embedding(norm_e(e))      for each of the `hc_count` streams s
    out = mtp.layers.0(x)                                          one sparse-attention decoder layer, a mixture's
    logits = lm_head(hyper_connection_mixer(out))                  the streams mixed back into one, no final norm

`e` is the embedding of the token after the one the main model's hidden rows `h` were computed at, and `h` the
main model's residual streams [.., hc_count * H] leaving its last layer before its own mixer (what the speculative
loop's layer tap reads). `norm_h` normalises each stream on its own (a `Qwen4ExpTextRMSNorm` over hc_count * H
with group H, as the model's own mixers and PLE norm their streams - inferred, not read off a reference), and
fc_embedding's one row is added into every stream. The layer's output streams are the hidden rows the next depth
reads, as the main model's are.

The layer is the family's own, shaped as a trunk layer is (`Qwen4Family.shape_layer`): its mixture's experts are
the expert store's, read under `mtp.layers.0.mlp.experts.` as the store's layer `L` - one past the trunk's, which
the store's lookahead never reaches - in the ascending waves every MoE call is served in, never placed with the
drafter. Only the dense tensors are placed: 0.17 GB of the 180B's 4.86 GB head. The layer's sparse attention keeps
its own one-layer cache with the indexer's keys beside K and V, which the tree's branches carry with them
(`MTPDrafter.layer_rows`)."""

from __future__ import annotations

import copy
import time
from typing import Any

import torch

from ....kinds import LayerKind
from ...drafter import MTPDrafter
from ...host import _HostLinear, compute_fp32
from ...native import Native

# the drafting head's tensors, and its layer's
HEAD = "mtp."
LAYER = "mtp.layers.0."


class _Head(torch.nn.Module):
    """the drafter's own modules around its layer, named as the checkpoint names them under `mtp.`"""

    def __init__(self, mod: Any, cfg: Any) -> None:
        super().__init__()
        H, S, eps = int(cfg.hidden_size), int(cfg.hc_count), float(cfg.rms_norm_eps)
        self.pre_fc_norm_embedding = mod.Qwen4ExpTextRMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = mod.Qwen4ExpTextRMSNorm(S * H, group_size=H, eps=eps)
        self.fc_embedding = torch.nn.Linear(H, H, bias=False)
        self.fc_hidden = torch.nn.Linear(H, H, bias=False)
        self.hyper_connection_mixer = mod.Qwen4ExpTextGatedResidual(cfg, use_combine=False)


def _indexed_layer() -> type[Any]:
    from transformers.cache_utils import DynamicIndexedLayer

    return DynamicIndexedLayer


class _GrantedIndexedLayer(_indexed_layer()):  # type: ignore[misc]
    """The drafter's sparse-attention cache layer: transformers' own - its keys, values and the indexer's keys grown
    by concatenation, which the tree's branches assign and restore as they walk - each growth past what it was
    granted asked of the scheduler first, for twice what the rows reach: a decode's per-token growth reads the
    ledger once a doubling, and a concatenation's copy beside the rows it replaces fits what was asked."""

    def __init__(self, grant: Any, dev: torch.device) -> None:
        super().__init__()
        self._ask = grant
        self._dev = dev
        self._granted = 0
        self._kv = 0  # the bytes the keys and values reach, and the indexer's keys
        self._ik = 0

    def _room(self) -> None:
        need = self._kv + self._ik
        if need > self._granted:
            want = 2 * need
            self._ask(
                want, "drafter", requester="Qwen4's drafter: its layer's cache", device=self._dev, held=self._granted
            )
            self._granted = want

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        T = max(1, int(key_states.shape[-2]))
        per = key_states.numel() // T * key_states.element_size()  # a position's bytes, every row and head
        self._kv = 2 * (self.get_seq_length() + T) * per
        self._room()
        return super().update(key_states, value_states, *args, **kwargs)

    def update_indexer(self, indexer_key_states: torch.Tensor) -> Any:
        T = max(1, int(indexer_key_states.shape[1]))
        have = self.indexer_keys
        n = int(have.shape[1]) if isinstance(have, torch.Tensor) and have.dim() == 3 else 0
        self._ik = (n + T) * (indexer_key_states.numel() // T) * indexer_key_states.element_size()
        self._room()
        return super().update_indexer(indexer_key_states)


class Qwen4Drafter(MTPDrafter):
    """the drafting head of Qwen4 (`qwen4_exp`)"""

    head: _Head
    dense_keys: list[str]
    dense_numel: int

    def __init__(
        self, sm: Any, train: bool = False, weights: str | None = None, dev: str | torch.device | None = None
    ) -> None:
        if train:
            raise ValueError("btb does not train Qwen4's drafting head")
        super().__init__(sm, train=False, weights=weights, dev=dev)
        cfg, fam = sm.cfg, sm.fam
        self.H, self.S = int(cfg.hidden_size), int(fam.streams)
        # the drafter's one layer as a model of its own: one sparse-attention layer, no n-gram embedding
        mcfg = copy.deepcopy(cfg)
        mcfg.num_hidden_layers = 1
        mcfg.layer_types = [LayerKind.QWEN_SPARSE.value]
        mcfg.ple_layer_ids = []
        self.mcfg = mcfg
        src = {} if weights is None else torch.load(weights, map_location="cpu", weights_only=True)
        # the host's kernels where the drafter runs on the host with them; the torch modules in bf16 elsewhere
        self.host = self.dev.type == "cpu" and Native.gemv is not None and sm.mlx is None
        self.cd = torch.float32 if self.host else torch.bfloat16
        with sm._meta:
            layer = fam.layer(mcfg, 0).eval()
            head = _Head(fam.mod, mcfg).eval()
        # the store's layer past the trunk's: its experts under `mtp.layers.0.`, served as every MoE layer's are
        L = int(sm.L)
        layer = fam.shape_layer(sm, layer, L, base=LAYER)
        self.dense_keys, self.dense_numel = [], 0
        self._grant([(LAYER, layer), (HEAD, head)], src)
        self.layer = self._load(layer, LAYER, src)
        self.head = self._load(head, HEAD, src)
        if self.host:
            fam.finish_host_layer(sm, self.layer, L, base=LAYER)
        self.layer.self_attn.layer_idx = 0
        self.layer.self_attn.indexer.layer_idx = 0
        if getattr(sm, "_packed", None):
            sm._bind_host_packed_layer(self.layer)
            sm._bind_host_packed_layer(self.head)
        # the rope over the drafter's own positions, on its device (the engine's may sit on the card)
        self.rotary = fam.rotary(config=cfg).to(self.dev)
        if weights is not None:
            sm.log(f"[mtp] drafter weights <- {weights} ({sum(1 for k in src if k != '__meta__')} tensors)")
        sm.log(
            f"[mtp] Qwen4's drafter on {self.dev}: {len(self.dense_keys)} dense tensors "
            f"({2 * self.dense_numel / 2**30:.2f} GB at bf16); its experts through the expert store as layer {L}"
        )
        self.cache = None

    def _grant(self, parts: list[tuple[str, torch.nn.Module]], src: dict[str, Any]) -> None:
        """the bytes the placement allocates, asked of the scheduler first: the dense tensors on a torch tier; on
        the host the norms widened to float32 (its matrices are views of the checkpoint's map)"""
        sched = getattr(self.sm, "scheduler", None)
        if sched is None:
            return
        n = 0
        for _base, mod in parts:
            for name, p, _b in self.sm._named_tensors(mod):
                if self.host:
                    n += 4 * p.numel() if p.dim() <= 1 or self.sm.fam.widened(name) else 0
                else:
                    n += 2 * p.numel()
        sched.grant(n, "drafter", requester="Qwen4's drafter: its dense tensors", device=self.dev)

    def _load(self, module: Any, base: str, src: dict[str, Any]) -> Any:
        """`module`'s tensors under `base` read and placed: on the host as a host layer holds them (norms and the
        family's widened matrices in float32, the rest as stored behind the host's linears, an FP8 matrix
        multiplied as stored), elsewhere in bf16 on the drafter's device"""
        sm = self.sm
        fp32_owners: set[str] = set()
        f8_keys: set[str] = set()
        for name, _p, is_buf in sm._named_tensors(module):
            key = base + name
            self.dense_keys.append(key)
            self.dense_numel += int(_p.numel())
            own = key in src
            if not self.host:
                t = src[key] if own else sm._get(key)
                sm._set_param(module, name, t.to(self.dev, torch.bfloat16), buffer=is_buf)
                continue
            f8 = not own and sm._fp8(key)
            widened = sm.fam.widened(name)
            if f8 and Native.gemv_fp8 is not None and not widened and len(shape := sm._shape(key)) == 2:
                f8_keys.add(key)
                sm._set_param(module, name, torch.zeros((), dtype=torch.bfloat16).expand(*shape), buffer=is_buf)
                continue
            t = src[key] if own else sm._get(key, stored=True)
            wide = t.is_floating_point() and (t.dim() <= 1 or widened)
            sm._set_param(module, name, t.float() if wide else t.bfloat16() if f8 else sm._held(t), buffer=is_buf)
            if wide and t.dim() >= 2 and ".experts." not in name:
                fp32_owners.add(name.rpartition(".")[0])
        if not self.host:
            return module
        for owner in sorted(fp32_owners):
            compute_fp32(module.get_submodule(owner))
        for mname, m in list(module.named_modules()):
            for cname, child in list(m.named_children()):
                if isinstance(child, torch.nn.Linear) and child.weight.dtype == torch.bfloat16:
                    key = base + (f"{mname}.{cname}" if mname else cname) + ".weight"
                    setattr(m, cname, _HostLinear(child.weight.data, key=key))
        for m in module.modules():
            if isinstance(m, _HostLinear) and m.key in f8_keys:
                m.f8 = sm._f8_weights(m.key)[0]
        return module

    def named_tensors(self) -> dict[str, torch.Tensor]:
        """the drafter's dense tensors by checkpoint name, as placed (a host linear's under its key)"""
        out: dict[str, torch.Tensor] = {}
        for base, mod in ((LAYER, self.layer), (HEAD, self.head)):
            for name, t, _b in self.sm._named_tensors(mod):
                lin = mod.get_submodule(name.rpartition(".")[0]) if "." in name else mod
                key = lin.key if isinstance(lin, _HostLinear) and lin.key else base + name
                out[key] = t
        return out

    def reset(self) -> None:
        from transformers.cache_utils import DynamicCache

        # one sparse-attention layer: K, V and the indexer's keys, grown as transformers grows them, the growth asked
        # of the scheduler (`_GrantedIndexedLayer`)
        self.cache = DynamicCache(config=self.mcfg)
        sched = getattr(self.sm, "scheduler", None)
        if sched is not None:
            self.cache.layers[0] = _GrantedIndexedLayer(sched.grant, self.dev)

    def _rope(self, n: int, B: int, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """the rope of positions [0, n) - the indexer pools its key blocks at their own positions - for `B` rows
        sharing them (a tree's depth: every node at the same position), one table viewed `B` times"""
        pos = torch.arange(n, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.rotary(like, pos)
        return cos.expand(B, -1, -1), sin.expand(B, -1, -1)

    def _step(self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool = True) -> tuple[Any, Any]:
        from transformers.masking_utils import create_causal_mask

        self._pack_torch()
        t0 = time.time()
        sm = self.sm
        hd = self.head
        e = sm.embed(tok_ids).to(self.dev, self.cd)
        hs = h.to(self.dev, self.cd)
        B, T = int(e.shape[0]), int(e.shape[1])
        emb = hd.fc_embedding(hd.pre_fc_norm_embedding(e))  # [B, T, H], into every stream
        streams = hd.fc_hidden(hd.pre_fc_norm_hidden(hs).unflatten(-1, (self.S, self.H)))  # [B, T, S, H]
        x = (streams + emb.unsqueeze(-2).to(streams.dtype)).flatten(-2)
        pe = self._rope(pos0 + T, B, x)
        text_pos = (torch.arange(T, device=self.dev) + pos0).view(1, -1).expand(B, -1)
        mask = create_causal_mask(
            config=self.mcfg,
            inputs_embeds=x,
            attention_mask=None,
            past_key_values=self.cache,
            position_ids=text_pos,
            allow_is_causal_skip=False,
        )
        out = self.layer(x, position_embeddings=pe, attention_mask=mask, past_key_values=self.cache, use_cache=True)
        if not need_logits:
            sm._sync()
            self.step_s += time.time() - t0
            self.steps += 1
            return None, out
        logits = self._head_logits(hd.hyper_connection_mixer(out))
        sm._sync()
        self.step_s += time.time() - t0
        self.steps += 1
        return logits, out
