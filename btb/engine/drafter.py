# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MTP drafter: a model's own drafting head run as a small model of its own, proposing the speculative tree."""

from __future__ import annotations

import heapq
import math
import time
import weakref
from collections.abc import Sequence
from typing import Any

import torch

from .. import mlx as mlxdev
from ..kinds import Tokens
from .cache import GrowLayer, indexer_keys, set_rows
from .fixed_rows import RowLinear
from .host import _HostLinear
from .native import Native


def mlx_topk_ids(logits: Any, k: int) -> Any:
    """the top-k ids [T, k] a row of an MLX logits array, sorted by value descending, the reduction (argpartition
    then sort) on the graph so a caller reads k ids a row instead of the whole vocab; lazy."""
    m = mlxdev.mx()
    k = min(int(k), int(logits.shape[-1]))
    part = m.argpartition(-logits, kth=k - 1, axis=-1)[:, :k]
    vals = m.take_along_axis(logits, part, axis=-1)
    order = m.argsort(-vals, axis=-1)
    return m.take_along_axis(part, order, axis=-1)


class _Int8Linear(torch.nn.Module):
    """A linear over int8 weights with one scale a row, packed in memory from a bf16/float linear: the
    drafter's weights on the torch tiers (`draft_bits` 8; 4 runs as 8 here). torch's packed int8 matmul where
    the device has it (CPU, MPS), the row-scaled product from the int8 tensor elsewhere - a card kernel is
    the card's to add."""

    def __init__(self, weight: torch.Tensor, rows: int = 1024) -> None:
        super().__init__()
        # quantize in row blocks to bound the float32 temporaries
        w0 = weight.detach()
        w8 = torch.empty(w0.shape, dtype=torch.int8, device=w0.device)
        scale = torch.empty(int(w0.shape[0]), dtype=torch.float32, device=w0.device)
        for r in range(0, int(w0.shape[0]), int(rows)):
            w = w0[r : r + rows].float()
            s = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
            w8[r : r + rows] = (w / s[:, None]).round().clamp(-127, 127).to(torch.int8)
            scale[r : r + rows] = s
        self.w8 = torch.nn.Parameter(w8, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        self.packed_mm = torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::_weight_int8pack_mm", weight.device.type.upper()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.packed_mm:
            y = torch._weight_int8pack_mm(x2, self.w8, self.scale.to(x2.dtype))
        else:
            y = x2 @ (self.w8.to(x2.dtype) * self.scale.to(x2.dtype)[:, None]).T
        return y.reshape(*shape[:-1], y.shape[-1])


class MTPDrafter:
    build_s: float
    _head_h: Any
    _head_t: Any
    fc8: Any
    _mx_consts: Any
    _mx_head_w: Any
    cache: Any
    cd: torch.dtype
    dev: torch.device
    fc: torch.Tensor | None
    fc_host: _HostLinear | None
    layer: Any
    mcfg: Any
    norm: Any
    norm_e: Any
    norm_h: Any
    sm: Any
    step_s: float
    steps: int
    train_mode: bool
    # the session whose rows the cache holds: set by that session's commit, let go by the next decode that takes the
    # drafter. A session reuses the drafter's rows only while it is theirs - the engine has one drafter, and two
    # sessions taking turns on it would each crop the other's rows as their own
    follows: weakref.ref[Any] | None = None

    def __init__(
        self, sm: Any, train: bool = False, weights: str | None = None, dev: str | torch.device | None = None
    ) -> None:
        """the engine it drafts for and the device and dtype it runs in; a family's drafter builds its layer after,
        from the checkpoint's drafting head or, given `weights`, from that file in its place"""
        self.sm = sm
        self.dev = torch.device(dev) if dev is not None else sm.dev
        self.cd = torch.float32 if train else torch.bfloat16
        self.train_mode = train
        self.step_s = 0.0
        self.steps = 0
        # the last tree's nodes' path probabilities, in its order: what the verify pass's pricing weighs each by
        self.last_p: list[float] = []
        self._q: dict[Any, Any] = {}  # the weights packed for `draft_bits`, by (weight, dtype)
        self.fc: torch.Tensor | None = None
        self.fc_host: _HostLinear | None = None
        self.cache = None

    def _mlx_ready(self) -> bool:
        """whether this drafter runs its own MLX graph (a family's drafter that has one says so, and then carries
        the graph's parts below)"""
        return False

    def _tree_kernel(self) -> bool:
        """whether the MLX tree steps its nodes through the node kernel over the shared cache"""
        return False

    def _mx_dtype(self) -> Any:
        """the MLX graph's activation dtype"""
        raise NotImplementedError

    def _mlx_body(self, em: Any, hm: Any, pos0: int, append: Any, attn: Any = None, layer: Any = None) -> Any:
        """the drafting layer over MLX rows `em`/`hm` [B, T, H]: its output [B, T, H], lazily; `append(kh, vh)`
        gives the attention its K/V, or `attn(qh, kh, vh)` computes the attention itself"""
        raise NotImplementedError

    def _mlx_draw(self, out: Any, k: int, sampling: Any, keys: Sequence[int]) -> tuple[Any, Any, Any]:
        """`k` children a row of `out` [B, 1, H] drawn under `sampling`: their log-probabilities, ids and the
        distribution, lazily"""
        raise NotImplementedError

    def _mlx_topk(self, out: Any, k: int) -> tuple[Any, Any]:
        """the top-k log-probabilities and ids [B, k] of `out` [B, 1, H]'s next token, lazily"""
        raise NotImplementedError

    def _step(self, tok_ids: torch.Tensor, h: torch.Tensor, pos0: int, need_logits: bool = True) -> tuple[Any, Any]:
        """the drafting head over `tok_ids` [B, T] and the hidden rows `h` under them at `pos0`: (the logits, or None
        without `need_logits`; the hidden rows it leaves, which the next step reads). Each family's own"""
        raise NotImplementedError

    def reset(self) -> None:
        from transformers.cache_utils import DynamicCache

        self.cache = DynamicCache(config=self.mcfg)
        if self.sm.mlx is not None:
            self.cache.layers[0] = GrowLayer(shared=True)

    def _torch_head(self) -> Any:
        """Return rows `[:draft_vocab]` of the lm_head for drafting, int8-quantized if `draft_bits` < 16.
        Cached on the instance; None if the head is on another device and there is no slice."""
        head = getattr(self, "_head_t", None)
        if head is None:
            sm = self.sm
            W = (sm.head.weight if sm.resident_head else sm._get(sm.head_key)).detach()
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            whole = not (0 < n < int(W.shape[0]))
            if whole and W.device.type != self.dev.type:
                return None
            if not whole:
                W = W[:n]
            bits = int(getattr(self.sm, "draft_bits", 16) or 16)
            if bits < 16 and not self.train_mode:
                head = _Int8Linear(W.to(self.dev))
                sm.log(
                    f"[draft] the drafter's head: {'the first ' + str(n) + ' ids' if not whole else 'every id'} packed to "
                    f"8 bits in memory on {self.dev} ({head.w8.numel() / 2**20:.0f} MB)"
                )
            elif W.device.type != self.dev.type:
                head = W.to(self.dev, self.cd)
            else:
                head = W  # the model's own rows, a view
            self._head_t = head
        return head

    def _host_head(self) -> Any:
        """Return rows `[:draft_vocab]` of the lm_head as a native `_HostLinear` for drafting on the CPU.
        Cached on the instance; None without a slice, without the native kernel, or with MLX."""
        head = getattr(self, "_head_h", None)
        if head is None:
            sm = self.sm
            n = int(getattr(sm, "draft_vocab", 0) or 0)
            head = False
            if sm.mlx is None and Native.gemv is not None and n > 0:
                W = sm._get(sm.head_key)
                if n < int(W.shape[0]):
                    head = _HostLinear(W[:n], key=sm.head_key)
                    sm.log(
                        f"[draft] the drafter's head on the host: the first {n} ids "
                        f"(bf16, {n * int(W.shape[1]) * 2 / 2**20:.0f} MB)"
                    )
            self._head_h = head
        return head or None

    def _head_logits(self, hn: torch.Tensor) -> torch.Tensor:
        """float32 logits of the drafter's last rows `hn` [B, T, H] over the head it drafts with: on the host the
        native slice (`_host_head`), else the model's own host head; on a torch tier the slice or the model's rows
        on the drafter's device (`_torch_head`), chunked across the vocabulary where neither is there"""
        sm = self.sm
        if self.dev.type == "cpu" and (sm.dev.type != "cpu" or Native.gemv is not None or sm.mlx is not None):
            head = self._host_head()
            return (sm._head_host() if head is None else head)(hn.float()).float()
        head = self._torch_head()
        if head is None:
            W = (sm.head.weight if sm.resident_head else sm._get(sm.head_key)).detach()
            step = 32768
            return torch.cat(
                [hn.to(self.cd) @ W[c : c + step].to(self.dev, self.cd).T for c in range(0, W.shape[0], step)], dim=-1
            ).float()
        if isinstance(head, _Int8Linear):
            return head(hn).float()
        return (hn.to(head.dtype) @ head.T).float()

    def _pack_torch(self) -> int:
        """the drafter's layer on a torch tier with `draft_bits` below 16: every plain linear of it (and the fc
        tensor) replaced by an int8 one packed in memory; returns how many were packed (the host tier's
        native linears keep their own format)"""
        bits = int(getattr(self.sm, "draft_bits", 16) or 16)
        if bits >= 16 or self.train_mode or getattr(self, "_packed_torch", False):
            return 0
        self._packed_torch = True
        n = 0
        for _mname, mod in list(self.layer.named_modules()):
            for cname, child in list(mod.named_children()):
                # a plain linear, or the card layers' fixed-row one (`RowLinear`, the same parameters)
                if type(child) in (torch.nn.Linear, RowLinear):
                    setattr(mod, cname, _Int8Linear(child.weight.data))
                    n += 1
        if self.fc_host is None and self.fc is not None:
            self.fc8 = _Int8Linear(self.fc)
            self.fc = None
            n += 1
        if n:
            self.sm.log(
                f"[draft] the drafter's {n} linears packed to 8 bits in memory on {self.dev}"
                + (" (4 asked; 8 on this tier)" if bits < 8 else "")
            )
        return n

    @staticmethod
    def layer_rows(layer: Any) -> tuple[torch.Tensor, ...]:
        """a drafter cache layer's per-row state, each tensor a row's along its first dim and a position's along its
        second last: its keys and values, and a sparse-attention layer's indexer keys [B, n, d_index] with them"""
        ik = indexer_keys(layer)
        return (layer.keys, layer.values) if ik is None or ik.dim() != 3 else (layer.keys, layer.values, ik)

    @staticmethod
    def set_layer_rows(layer: Any, rows: Sequence[torch.Tensor]) -> None:
        """`layer_rows`'s state written back: the tree's branches, a crop, the root restored"""
        set_rows(layer, rows[0], rows[1])
        if len(rows) > 2:
            layer.indexer_keys = rows[2]

    def crop(self, keep: int) -> None:
        for layer in self.cache.layers:
            if getattr(layer, "keys", None) is not None and layer.keys.shape[-2] > keep:
                self.set_layer_rows(layer, [t[..., :keep, :] for t in self.layer_rows(layer)])

    def prefill(self, prompt_ids: Any, h_all: torch.Tensor) -> None:
        self.reset()
        n = len(prompt_ids)
        if n >= 2:
            self.extend(prompt_ids[1:n], h_all[:, : n - 1], 0)

    def extend(self, toks: Tokens, h: torch.Tensor, pos0: int) -> None:
        """The drafter's cache fed `toks` at `pos0` on, in chunks the way the engine's own prefill runs: its
        attention over a long prompt at once materialized the scores of every row against every key (a 13k-row
        turn asked Metal for 33 GB), so a chunk is sized by the same rule, against the keys already in its cache."""
        toks = [int(t) for t in toks]
        a = 0
        while a < len(toks):
            C = int(self.sm.prefill_chunk or self.sm._auto_chunk(pos0 + a))
            b = min(len(toks), a + C)
            self._step(torch.tensor([toks[a:b]]), h[:, a:b], pos0 + a, need_logits=False)
            a = b

    def ar(self, toks: Tokens, h: torch.Tensor, pos0: int, k: int) -> Any:
        if k <= 0:
            return []
        logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
        t = int(logits[0, -1].argmax())
        out = [t]
        hp = hout[:, -1:]
        pos = pos0 + len(toks)
        for _ in range(k - 1):
            logits, hout = self._step(torch.tensor([[t]]), hp, pos)
            t = int(logits[0, -1].argmax())
            out.append(t)
            hp = hout
            pos += 1
        return out

    def at(self, toks: Tokens, h: torch.Tensor, pos0: int, k: int) -> Any:
        if k <= 1:
            return (self.ar(toks, h, pos0, k) or [None])[0], [], []
        logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
        g1 = int(logits[0, -1].argmax())
        hp = hout[:, -1:]
        pos = pos0 + len(toks)
        logits2, hout2 = self._step(torch.tensor([[g1]]), hp, pos)
        top2 = torch.topk(logits2[0, -1], 2).indices.tolist()
        a2, b2 = int(top2[0]), int(top2[1])
        after_g1 = self.cache.get_seq_length()
        chains = []
        for x2 in (a2, b2):
            chain = [x2]
            hx, px, t = hout2, pos + 1, x2
            for _ in range(k - 2):
                lg, hx = self._step(torch.tensor([[t]]), hx, px)
                t = int(lg[0, -1].argmax())
                chain.append(t)
                px += 1
            chains.append(chain)
            self.crop(after_g1)
        return g1, chains[0], chains[1]

    def au(
        self,
        toks: Tokens,
        h: torch.Tensor,
        pos0: int,
        budget: int,
        max_depth: int = 8,
        top_k: int = 8,
        expand_k: int = 8,
        min_prob: float = 0.0,
        pop_ratio: float = 0.5,
        fan_ratio: float = 0.05,
        extra_chains: Any = (),
        with_tags: bool = False,
        sampling: Any = None,
    ) -> Any:
        """the tree of drafts; under a `sampling` (a temperature) every node's children are drawn without
        replacement from the drafter's sampled distribution, and the verify pass accepts them against it in draw
        order (`with_tags` then also returns {node: the distribution} and {node: the draws in order} for the root,
        -1, and each expanded node); the n-gram chains are left out of a sampled tree"""
        smp = sampling if (sampling is not None and not sampling.greedy) else None
        if smp is not None:
            # the proposal is the drafter's distribution at a multiple of the temperature: any proposal keeps the
            # verify pass exact (its ratio uses it), and a flatter one overlaps a target this head is sharper than
            from dataclasses import replace

            smp = replace(smp, temperature=smp.temperature * float(getattr(self.sm, "draft_temp_ratio", 1.0)))
        qrows: dict[int, Any] = {}
        empty: Any = (
            ([], [], [], [], qrows, {})
            if (with_tags and smp is not None)
            else ([], [], [], [])
            if with_tags
            else ([], [], [])
        )
        self.last_p = []
        if budget <= 0:
            return empty
        floor = -math.log(min_prob) if min_prob > 0 else float("inf")
        fan_gap = -math.log(fan_ratio) if fan_ratio > 0 else float("inf")
        chains = [
            ([int(t) for t in c[0]], math.log(max(1e-6, min(1.0, float(c[1])))), (c[2] if len(c) > 2 else "extra"))
            for c in extra_chains
            if c and len(c[0]) > 0 and smp is None
        ]
        root_pos = pos0 + len(toks) - 1  # the cache row the root's children are proposed at
        layer = self.cache.layers[0]
        mlx_tree = self._mlx_ready() and isinstance(layer, GrowLayer) and layer.shared and not layer.bits
        if mlx_tree:
            m = mlxdev.mx()
            t0 = time.time()
            dt = self._mx_dtype()
            e = self.sm.embed(torch.tensor([[int(t) for t in toks]]))
            em = mlxdev.to_mx(e).astype(dt)
            hm = mlxdev.to_mx(h.detach().contiguous()).astype(dt)
            out = self._mlx_body(em, hm, pos0, layer.mx_update)
            if smp is not None:
                vals0, ids0, q0 = self._mlx_draw(out[:, -1:], top_k, smp, [smp.key_for(root_pos, salt=1)])
                qrows[-1] = q0[0]
            else:
                vals0, ids0 = self._mlx_topk(out[:, -1:], top_k)
            m.eval(vals0, ids0)
            self.step_s += time.time() - t0
            self.steps += 1
            root_len = self.cache.get_seq_length()
            kernel_tree = self._tree_kernel()
            kv_root: Any = None if kernel_tree else layer.mx_kv(0, root_len)
            root_vals, root_inds, root_hin = vals0.tolist()[0], ids0.tolist()[0], out[:, -1:]
            if kernel_tree:
                # the tree's rows go after the prefix, in stepping order; the prefix is never copied
                Hk_ = (
                    int(layer._shape[1])
                    if layer._shape is not None
                    else int(getattr(self.sm.cfg, "num_key_value_heads", 0))
                )
                hd_ = int(self.layer.self_attn.head_dim)
                layer._ensure(1, Hk_, root_len + budget + 1, hd_, mlxdev.torch_dtype(layer._mx[0].dtype))
                row_parent: list[int] = []  # a tree row's parent row (-1: the prefix)
                row_of: dict[int, int] = {}  # node index -> tree row
        else:
            logits, hout = self._step(torch.tensor([[int(t) for t in toks]]), h, pos0)
            root_len = self.cache.get_seq_length()
            # every per-row tensor of the layer (a sparse attention's indexer keys with its K/V) branches with the tree
            kv_root = tuple(t.clone() for t in self.layer_rows(layer))
            if smp is not None:
                ids0, q0 = smp.draw_torch(logits[0, -1:].float(), [smp.key_for(root_pos, salt=1)], top_k)
                qrows[-1] = q0[0]
                root_vals = torch.log(q0[0][ids0[0]]).tolist()
                root_inds, root_hin = ids0[0].tolist(), hout[:, -1:]
            else:
                tb0 = torch.topk(torch.log_softmax(logits[0, -1].float(), dim=-1), min(top_k, logits.shape[-1]))
                root_vals, root_inds, root_hin = tb0.values.tolist(), tb0.indices.tolist(), hout[:, -1:]
        nodes: list[Any] = []
        kv_after: dict[Any, Any] = {}
        paths: dict[Any, Any] = {}
        tags: dict[Any, Any] = {}
        heap: list[Any] = []
        tie = 0
        draws: dict[int, list[int]] = {}  # under sampling: every draw of a node in draw order, for the verify pass

        def az(
            parent_idx: int, parent_neg: float, parent_depth: int, vals: Any, inds: Any, hin: Any, parent_path: Any
        ) -> None:
            nonlocal tie
            if smp is not None:
                # the verify pass tries all of a node's draws in order whether or not the tree holds them (an accepted
                # draw the tree lacks ends the walk), so the heap takes them by probability like the greedy tree
                draws[parent_idx] = [int(t) for lv, t in zip(vals, inds) if math.isfinite(lv)]  # no zero-mass draw
            best = max(vals)
            seen = {}
            for lv, t in zip(vals, inds):
                if best - lv > fan_gap:
                    continue
                seen[int(t)] = (parent_neg - lv, "mtp")
            for ctoks, clp, ctag in chains:
                d = len(parent_path)
                if d < len(ctoks) and ctoks[:d] == parent_path:
                    cand = parent_neg - clp
                    if ctoks[d] not in seen or cand < seen[ctoks[d]][0]:
                        seen[ctoks[d]] = (cand, ctag)
            for t, (neg, tag) in seen.items():
                heapq.heappush(heap, (neg, tie, parent_idx, t, parent_depth + 1, hin, tag, [*parent_path, t]))
                tie += 1

        az(-1, 0.0, 0, root_vals, root_inds, root_hin, [])
        # a confident root over-builds: shrink the budget as it sharpens (size affects speed only, never a token)
        p1 = math.exp(float(max(root_vals)))
        cmin = int(getattr(self.sm, "tree_cap_min", 5))
        lo = float(getattr(self.sm, "tree_conf_lo", 0.5))
        hi = float(getattr(self.sm, "tree_conf_hi", 0.9))
        frac = min(1.0, max(0.0, (p1 - lo) / max(1e-6, hi - lo)))
        eff_budget = max(min(cmin, budget), round(budget - (budget - cmin) * frac))
        pop_gap = -math.log(pop_ratio)
        # a depth is stepped only when its nodes' path probability can pay for the drafter's step
        step_mass = float(getattr(self.sm, "tree_step_mass", 0.0) or 0.0)
        while heap and len(nodes) < eff_budget and heap[0][0] <= floor:
            popped: list[Any] = []
            while (
                heap
                and len(popped) < expand_k
                and len(nodes) + len(popped) < eff_budget
                and heap[0][0] <= floor
                and (not popped or heap[0][0] - popped[0][0] <= pop_gap)
            ):
                popped.append(heapq.heappop(heap))
            first = len(nodes)
            for neg, _, parent, tok, depth, _hin, tag, path in popped:
                idx = len(nodes)
                nodes.append((tok, parent, depth))
                self.last_p.append(math.exp(-neg))
                paths[idx], tags[idx] = path, tag
            if len(nodes) >= eff_budget:
                break
            by_depth: dict[Any, Any] = {}
            for j, (neg, _, parent, tok, depth, hin, _tag, _path) in enumerate(popped):
                if depth < max_depth:
                    by_depth.setdefault(depth, []).append((first + j, neg, parent, tok, hin))
            for depth, group in sorted(by_depth.items()):
                if step_mass > 0 and sum(math.exp(-neg) for _, neg, _, _, _ in group) < step_mass:
                    continue
                ts = torch.tensor([[tok] for _, _, _, tok, _ in group])
                if mlx_tree and kernel_tree:
                    # the group's rows appended to the shared cache where their paths say, the attention one node
                    # kernel call over the prefix and the tree's rows: nothing copied, nothing crosses to torch
                    t0 = time.time()
                    first_row = len(row_parent)
                    for j, (idx, _neg, parent, _tok, _hin) in enumerate(group):
                        row_of[idx] = first_row + j
                        row_parent.append(-1 if parent < 0 else row_of[parent])
                    meta_all, path_all, splits = mlxdev.tree_meta(root_len, row_parent)
                    meta_g, path_g = meta_all[first_row:], path_all[first_row:]
                    hs = m.concatenate([hin for _, _, _, _, hin in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    kbuf, vbuf = layer._mx[0], layer._mx[1]
                    scale_ = float(self.layer.self_attn.scaling)

                    def node_attn(
                        qh: Any,
                        kh: Any,
                        vh: Any,
                        kbuf: Any = kbuf,
                        vbuf: Any = vbuf,
                        n0: int = root_len + first_row,
                        meta: Any = meta_g,
                        path: Any = path_g,
                        splits: int = splits,
                        scale_: float = scale_,
                    ) -> Any:
                        # the rows written in place, the flag evaluated before the kernel reads them
                        fk_ = mlxdev.kv_store(kbuf, kh[:, :, 0, :].transpose(1, 0, 2), n0)
                        fv_ = mlxdev.kv_store(vbuf, vh[:, :, 0, :].transpose(1, 0, 2), n0)
                        m.eval(fk_, fv_)
                        a = mlxdev.attn_nodes(qh[:, :, 0, :], kbuf, vbuf, meta, path, scale_, splits, odt=qh.dtype)
                        return a[:, :, None, :]

                    ho = self._mlx_body(em, hs, root_len + depth - 1, None, attn=node_attn)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, (idx, _, _, _, _) in enumerate(group):
                            qrows[idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = vals.tolist(), inds.tolist()
                    for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                        az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
                    continue
                if mlx_tree:
                    # the group's prefixes and hidden rows concatenated on the graph, the node's K/V a lazy slice
                    # of the group's, the top-k read as k numbers a node: nothing crosses to torch
                    t0 = time.time()
                    kp = m.concatenate([kv_root[0] if p < 0 else kv_after[p][0] for _, _, p, _, _ in group], axis=0)
                    vp = m.concatenate([kv_root[1] if p < 0 else kv_after[p][1] for _, _, p, _, _ in group], axis=0)
                    hs = m.concatenate([hin for _, _, _, _, hin in group], axis=0)
                    em = mlxdev.to_mx(self.sm.embed(ts)).astype(hs.dtype)
                    kn: list[Any] = []

                    def cat_kv(kh: Any, vh: Any, kp: Any = kp, vp: Any = vp, kn: list[Any] = kn) -> tuple[Any, Any]:
                        K = m.concatenate([kp, kh], axis=2)
                        V = m.concatenate([vp, vh], axis=2)
                        kn.extend((K, V))
                        return K, V

                    ho = self._mlx_body(em, hs, root_len + depth - 1, cat_kv)
                    if smp is not None:
                        gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                        vals, inds, qg = self._mlx_draw(ho[:, -1:], top_k, smp, gk)
                        for b, (idx, _, _, _, _) in enumerate(group):
                            qrows[idx] = qg[b]
                    else:
                        vals, inds = self._mlx_topk(ho[:, -1:], top_k)
                    m.eval(vals, inds)
                    self.step_s += time.time() - t0
                    self.steps += 1
                    vals_l, inds_l = vals.tolist(), inds.tolist()
                    for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                        kv_after[idx] = (kn[0][b : b + 1], kn[1][b : b + 1])
                        az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
                    continue
                self.set_layer_rows(
                    layer,
                    [
                        torch.cat([kv_root[j] if parent < 0 else kv_after[parent][j] for _, _, parent, _, _ in group])
                        for j in range(len(kv_root))
                    ],
                )
                hs = torch.cat([hin for _, _, _, _, hin in group], dim=0)
                lg, ho = self._step(ts, hs, root_len + depth - 1)
                if smp is not None:
                    gk = [smp.key_for(root_pos + depth, salt=2 + idx) for idx, _, _, _, _ in group]
                    ids_g, qg = smp.draw_torch(lg[:, -1].float(), gk, top_k)
                    vals_l = torch.log(torch.gather(qg, 1, ids_g)).tolist()
                    inds_l = ids_g.tolist()
                    for b, (idx, _, _, _, _) in enumerate(group):
                        qrows[idx] = qg[b]
                else:
                    tb = torch.topk(torch.log_softmax(lg[:, -1].float(), dim=-1), min(top_k, lg.shape[-1]), dim=-1)
                    vals_l, inds_l = tb.values.tolist(), tb.indices.tolist()
                for b, (idx, neg, _parent, _tok, _hin) in enumerate(group):
                    kv_after[idx] = tuple(t[b : b + 1].clone() for t in self.layer_rows(layer))
                    az(idx, neg, depth, vals_l[b], inds_l[b], ho[b : b + 1, -1:], paths[idx])
        if not mlx_tree:
            self.set_layer_rows(layer, kv_root)
        out = ([n[0] for n in nodes], [n[1] for n in nodes], [n[2] for n in nodes])
        if with_tags and smp is not None:
            return (*out, [tags[i] for i in range(len(nodes))], qrows, draws)
        return (*out, [tags[i] for i in range(len(nodes))]) if with_tags else out
