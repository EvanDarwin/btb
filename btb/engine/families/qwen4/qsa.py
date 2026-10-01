# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's sparse attention indexer (`Qwen4ExpTextQSAIndexer`) without the reference's per-query rebuild of the
keys. The reference, for every query row: finds its visible tokens (`nonzero`, a host sync), pools every complete
block of `compress_ratio` keys among them (mean, norm, rope at the block's start), scores the blocks against the
row's index heads and keeps the top `block_topk` blocks and the partial tail. At 16k rows that is 16k syncs and
16k poolings of up to 4k blocks a layer, and it was most of the 180B's prefill.

Under a plain causal mask (no padding, no window) the pooled key of a block is the same for every query that sees
it, so it is pooled once; a query's visible tokens are its prefix; a query whose complete blocks all fit the
budget keeps its whole prefix (the reference's top-k over all of them selects all of them), so only the rows past
the budget are scored. The default scores each of those rows as the reference does - its index heads against its
own complete blocks, one matmul of the reference's shape - so the mask is the reference's bit for bit. A speculative
pass's tree (every row the whole prefix, and of the pass's rows itself and its ancestors) pools the prefix's complete
blocks once and each row's blocks past them - the prefix's tail and its ancestors - for that row alone
(`_select_tree`), the reference's selection row for row; any other mask takes the reference's own forward. `sparse` (the `--sparse` option) scores every such row in one matmul
over all the blocks, the invalid ones masked: far fewer launches, and a sum in another order, so a near-tie at the
budget's edge may choose the other block. It is held against the reference indexer's choices, not the receipts.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from . import bind_forward


def install(indexer: Any, sparse: bool = False) -> None:
    """`indexer` (a `Qwen4ExpTextQSAIndexer`) forwarding through `select`; `sparse` fixed for its life"""
    indexer.btb_sparse = bool(sparse)
    bind_forward(indexer, select)


def _plain_causal(visible: torch.Tensor, S: int) -> bool:
    """whether every query row sees exactly its prefix: [B, 1, S, kv], row p the first kv - S + p + 1 positions"""
    if visible.dim() != 4 or visible.shape[-2] != S:
        return False
    kv = int(visible.shape[-1])
    off = kv - S
    if off < 0:
        return False
    rows = torch.arange(S, device=visible.device) + off
    causal = torch.arange(kv, device=visible.device)[None, :] <= rows[:, None]
    return all(torch.equal(visible[b, 0], causal) for b in range(int(visible.shape[0])))


def select(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Any,
) -> torch.Tensor:
    """the indexer's selected-token mask, [B, 1, S, kv]: bool, or additive float where the mask it is given is"""
    visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    B, S, _ = hidden_states.shape
    plain = _plain_causal(visible, S)
    tree = None if plain else _prefix_tree(visible, S)
    if not plain and tree is None:
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    d = int(self.index_head_dim)
    full_cos, full_sin = position_embeddings
    cur_cos, cur_sin = full_cos[:, -S:, :], full_sin[:, -S:, :]
    # the query and the new keys exactly as the reference makes them
    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(qk, [self.index_n_heads * d, self.index_kv_heads * d], dim=-1)
    shape = (B, S, -1, d)
    q, raw_keys = q.reshape(*shape), token_k.reshape(*shape).squeeze(2)
    q = self.q_layernorm(q)
    q = apply_rotary_pos_emb(q, cos=cur_cos, sin=cur_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    r, k_top = int(self.compress_ratio), int(self.block_topk)
    kv = int(visible.shape[-1])
    off = kv - S
    if tree is not None:
        mask = _select_tree(self, q, raw_keys, full_cos, full_sin, visible, tree, off, r, k_top, d)
        if attention_mask.is_floating_point():
            return torch.where(mask, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
        return mask
    NB = (off + S) // r  # the complete blocks the last row sees
    # every row whose complete blocks fit the budget keeps its whole prefix; the first row past it is p0
    p0 = max(0, (k_top + 1) * r - 1 - off)
    mask = visible.clone()
    if p0 < S and NB > k_top:
        dev = hidden_states.device
        blocks = torch.arange(NB * r, device=dev).view(NB, r)
        starts = blocks[:, 0]
        ps = torch.arange(p0, S, device=dev)
        n_vis = ps + off + 1
        nbs = n_vis // r
        at = torch.arange(kv, device=dev)[None, :]
        tail = (at >= (nbs * r)[:, None]) & (at < n_vis[:, None])  # the partial block each row keeps whole
        for b in range(B):
            # each complete block's key once, by the reference's own arithmetic for it
            groups = raw_keys[b].index_select(0, blocks.flatten()).view(NB, r, d)
            pooled = self.k_layernorm(groups.float().mean(dim=1).to(raw_keys.dtype))
            keys = apply_rotary_pos_emb(
                pooled.unsqueeze(1),
                cos=full_cos[b].index_select(0, starts),
                sin=full_sin[b].index_select(0, starts),
            ).squeeze(1)
            keys_f = keys.float()
            if bool(getattr(self, "btb_sparse", False)):
                sel = _scores_batched(q[b, p0:].float(), keys_f, nbs, k_top, d)
            else:
                picks = []
                for p in range(p0, S):
                    nb = (off + p + 1) // r
                    scores = torch.matmul(q[b, p].float(), keys_f[:nb].transpose(-1, -2)).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(d)
                    picks.append(scores.topk(k_top, dim=0).indices)
                sel = torch.stack(picks)
            tokens = (sel[:, :, None] * r + torch.arange(r, device=dev)).reshape(sel.shape[0], -1)
            rows = torch.zeros(sel.shape[0], kv, dtype=torch.bool, device=dev).scatter_(1, tokens, True)
            mask[b, 0, p0:] = rows | tail
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        return torch.where(mask, attention_mask.new_zeros(()), min_dtype)
    return mask


def _prefix_tree(visible: torch.Tensor, S: int) -> list[list[torch.Tensor]] | None:
    """A speculative pass's mask - every row sees the whole prefix, and of the pass's own rows itself and some of
    those before it (a tree's ancestors, parents ahead of their children) - as each row's pass rows it sees, by
    batch row; None for any other mask. [B, 1, S, kv]"""
    if visible.dim() != 4 or visible.shape[-2] != S:
        return None
    kv = int(visible.shape[-1])
    off = kv - S
    if off < 0 or not bool(visible[..., :off].all()):
        return None
    block = visible[..., off:]
    upper = torch.ones(S, S, dtype=torch.bool, device=visible.device).triu(1)
    diag = torch.eye(S, dtype=torch.bool, device=visible.device)
    if bool((block & upper).any()) or not bool((block | ~diag).all()):
        return None
    return [[block[b, 0, p].nonzero().flatten() for p in range(S)] for b in range(int(visible.shape[0]))]


def _pooled(self: Any, raw: torch.Tensor, blocks: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """blocks' keys [n, d] by the reference's arithmetic: the mean of each block's `r` raw keys, normed, roped at the
    block's first position. `raw` [kv, d]; `blocks` [n, r] cache rows; `cos`/`sin` [kv, rd] a row's position each"""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    n, r = int(blocks.shape[0]), int(blocks.shape[1])
    groups = raw.index_select(0, blocks.flatten()).view(n, r, -1)
    pooled = self.k_layernorm(groups.float().mean(dim=1).to(raw.dtype))
    starts = blocks[:, 0]
    return apply_rotary_pos_emb(
        pooled.unsqueeze(1), cos=cos.index_select(0, starts), sin=sin.index_select(0, starts)
    ).squeeze(1)


def _select_tree(
    self: Any,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    full_cos: torch.Tensor,
    full_sin: torch.Tensor,
    visible: torch.Tensor,
    tree: list[list[torch.Tensor]],
    off: int,
    r: int,
    k_top: int,
    d: int,
) -> torch.Tensor:
    """The indexer's selection for a speculative pass's rows ([B, 1, S, kv] bool). Row p sees the prefix and its own
    pass rows, so its complete blocks are the prefix's first `off // r` - pooled once, the same for every row -
    then the blocks of what follows them in its sequence: the prefix's partial tail and its ancestors, pooled for
    the row alone. A row whose blocks fit the budget keeps everything it sees (the reference's top-k over all of
    them selects all of them); the others score their blocks as the reference does, one matmul of its shape."""
    dev = q.device
    mask = visible.clone()
    kv = int(visible.shape[-1])
    n_pre = off // r
    pre_blocks = torch.arange(n_pre * r, device=dev).view(n_pre, r)
    lead = torch.arange(n_pre * r, off, device=dev)  # the prefix's rows past its last complete block
    for b, rows in enumerate(tree):
        shared: torch.Tensor | None = None
        for p, own in enumerate(rows):
            n_vis = off + int(own.numel())
            if n_vis // r <= k_top:
                continue
            if shared is None:
                shared = _pooled(self, raw_keys[b], pre_blocks, full_cos[b], full_sin[b])
            rest = torch.cat([lead, off + own.to(dev)])
            n_own = int(rest.numel()) // r
            own_blocks = rest[: n_own * r].view(n_own, r)
            keys = shared
            if n_own:
                keys = torch.cat([shared, _pooled(self, raw_keys[b], own_blocks, full_cos[b], full_sin[b])])
            scores = torch.matmul(q[b, p].float(), keys.float().transpose(-1, -2)).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(d)
            sel = scores.topk(k_top, dim=0).indices
            tokens = torch.cat([pre_blocks, own_blocks]).index_select(0, sel).flatten()
            row = torch.zeros(kv, dtype=torch.bool, device=dev)
            row[tokens] = True
            row[rest[n_own * r :]] = True  # the partial block kept whole
            mask[b, 0, p] = row
    return mask


def _scores_batched(q: torch.Tensor, keys: torch.Tensor, nbs: torch.Tensor, k_top: int, d: int) -> torch.Tensor:
    """`--sparse`: every row's top blocks in one matmul over all the blocks, those a row cannot see masked below
    any score (a score is a sum of relus, never negative); tiled over the rows so the scores stay under ~256 MB"""
    n, H, _ = q.shape
    NB = int(keys.shape[0])
    step = max(1, (256 << 20) // max(1, H * NB * 4))
    out = []
    kt = keys.transpose(-1, -2)
    for a in range(0, n, step):
        s = torch.matmul(q[a : a + step], kt)  # [rows, H, NB]
        s = torch.relu(s).sum(dim=1) / math.sqrt(d)
        s.masked_fill_(torch.arange(NB, device=s.device)[None, :] >= nbs[a : a + step, None], -1.0)
        out.append(s.topk(k_top, dim=-1).indices)
    return torch.cat(out)
