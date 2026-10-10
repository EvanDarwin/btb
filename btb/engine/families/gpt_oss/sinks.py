# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""gpt-oss's attention (`attention_sinks`): a sink logit per query head in the softmax's denominator and a sliding
window on alternate layers, which sdpa cannot express; and the workspace a prefill sweep's scores are views of
(`ScoresWorkspace`)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .... import mlx as mlxdev
from ...cache import GrowLayer
from ...device import Where, where
from ...fixed_rows import KeyRows


class ScoresWorkspace:
    """Two buffers a layer-by-layer prefill makes once, each its last chunk's scores in the dtype the sinks' join
    promotes them to, that `attention_sinks` takes its big tensors from as views, in turn: the scores (B), their
    promoted copy (A), the softmax (B, over the scores), the probabilities in the values' dtype (A, over the copy).
    Made per call, they grow with every chunk's position, and a buffer freed by one chunk is too small for the next:
    the card fills with blocks the allocator must empty its cache to reuse, or cannot. As views of these the
    largest of a pass's buffers never come from the cache at all."""

    def __init__(self, nbytes: int, device: torch.device) -> None:
        self.nbytes = int(nbytes)
        self.bufs = {
            "a": torch.empty(self.nbytes, dtype=torch.uint8, device=device),
            "b": torch.empty(self.nbytes, dtype=torch.uint8, device=device),
        }

    def view(self, which: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor | None:
        """buffer `which`'s front as a contiguous `shape` of `dtype`; None where it does not hold that much"""
        n = 1
        for s in shape:
            n *= int(s)
        n *= torch.empty(0, dtype=dtype).element_size()
        buf = self.bufs[which]
        return buf[:n].view(dtype).view(shape) if n <= buf.numel() else None


# the open workspace a device's sink attention takes its scores from (`ScoresWorkspace`): a prefill sweep's, while
# it runs; none otherwise, the tensors then made per call
_SCORES: dict[Where, ScoresWorkspace] = {}


def open_scores(device: Any, ws: ScoresWorkspace) -> None:
    _SCORES[where(device)] = ws


def close_scores(device: Any) -> None:
    _SCORES.pop(where(device), None)


def attention_sinks(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    s_aux: Any = None,
    **kw: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """gpt-oss's attention: a sink logit per query head in the softmax's denominator and, on alternate layers,
    a `sliding_window`. Over an MLX cache the nodes of a speculative pass (and every decode row once
    speculation is on) run through the node kernel, the rest through MLX's fused attention; elsewhere the
    reference's arithmetic without its copies. Returns what the reference returns."""
    B, Hq, T, d = (int(v) for v in query.shape)
    Hk = int(key.shape[1])
    g = Hq // Hk
    scale = float(d**-0.5 if scaling is None else scaling)
    win = int(sliding_window) if sliding_window else None
    sm: Any = getattr(module, "_sm", None)
    cl = None
    if sm is not None and sm.mlx is not None and B == 1 and getattr(module, "layer_idx", None) is not None:
        ctx = getattr(sm, "_attn_ctx", None)
        cl = ctx.layers[module.layer_idx] if ctx is not None and module.layer_idx < len(ctx.layers) else None
        if not (
            isinstance(cl, GrowLayer)
            and cl.shared
            and cl._mx is not None
            and cl._tk is None
            and cl._tv is None
            and int(cl._n) == int(key.shape[-2])
            and s_aux is not None
        ):
            cl = None
    if cl is not None:
        m = mlxdev.mx()
        n = int(cl._n)
        past = n - T
        spec = bool(getattr(sm, "aq", False)) and past > 0
        sinks = getattr(module, "_mx_sinks", None)
        if sinks is None:
            sinks = mlxdev.to_mx(s_aux.detach().float().contiguous())
            m.eval(sinks)
            module._mx_sinks = sinks
        K, V = cl._mx[0], cl._mx[1]
        kernel = (
            d in (64, 128, 256)
            and g <= mlxdev.ATTN_MAXG
            and (K.dtype == m.bfloat16 or cl.bits)
            and getattr(sm, "mlx_attn_kernel", True)
            and ((T > 1 and spec) or (T == 1 and (spec or n >= sm.mlx_attn_rows)))
        )
        kq = {"ks": cl._mx[2], "vs": cl._mx[3]} if cl.bits else {}
        parents = getattr(sm, "ap", None) if spec else None
        if parents is None or len(parents) != T:
            parents = list(range(-1, T - 1))
        if kernel:
            q = mlxdev.to_mx(query[0].transpose(0, 1).float().contiguous())
            if T == 1:
                out = mlxdev.attn_decode(
                    q[0], K, V, n, scale, params=mlxdev.attn_params(n, window=win), sinks=sinks, **kq
                )[None]
            else:
                out = mlxdev.attn_tree(q, K, V, past, list(parents), scale, sinks=sinks, window=win, **kq)
            return mlxdev.from_mx(out).to(query.dtype)[None], None
        qh = mlxdev.to_mx(query[0].contiguous())[None]
        if T == 1:
            start = mlxdev.attn_window_start(n, win)
            Kv, Vv = cl.mx_kv(start, n)
            mx_mask = None
        else:
            tree = any(parents[j] != j - 1 for j in range(T))
            if tree:
                Kv, Vv = cl.mx_kv(0, n)
                depth = [0] * T
                allow = torch.zeros(T, n, dtype=torch.bool)
                for t in range(T):
                    depth[t] = 0 if parents[t] < 0 else depth[parents[t]] + 1
                    allow[t, mlxdev.attn_window_start(past + depth[t] + 1, win) : past] = True
                    cur = t
                    while cur >= 0:
                        allow[t, past + cur] = True
                        cur = parents[cur]
                mx_mask = mlxdev.to_mx(allow)
            else:
                # a sliding layer's chunk needs only the window's worth of keys behind it: slicing there is exact and
                # makes the prefill O(T*win)
                start = max(0, past - win + 1) if win is not None else 0
                Kv, Vv = cl.mx_kv(start, n)
                p = past + m.arange(T, dtype=m.int32)[:, None]
                j = start + m.arange(n - start, dtype=m.int32)[None]
                mx_mask = (j <= p) if win is None else ((j <= p) & (j > p - win))
        a = m.fast.scaled_dot_product_attention(
            qh.astype(Kv.dtype), Kv, Vv, scale=scale, mask=mx_mask, sinks=sinks.astype(Kv.dtype)
        )
        return mlxdev.from_mx(a[0].transpose(1, 0, 2)).to(query.dtype)[None], None
    if T == 1 and B == 1 and (sm is None or getattr(sm, "_attn_ctx", None) is not None):
        # one query row - a greedy step, or a speculative pass's node (`KeyRows`), on the card or the host - over the rows it
        # attends gathered in order and nothing masked: a step and a node over the same rows make the same call on the
        # same shapes, so the same bits. The pass carries no caller's mask (`_attn_ctx`), so a mask tensor here is the
        # causal one, all of the prefix or its window
        idx = attention_mask.idx if isinstance(attention_mask, KeyRows) else None
        n = int(key.shape[-2])
        if idx is None and win and n > win:
            idx = torch.arange(n - win, n, device=key.device)
        if idx is not None:
            key, value = key.index_select(-2, idx), value.index_select(-2, idx)
        key, value = key.contiguous(), value.contiguous()
        attention_mask, win = None, None
    elif isinstance(attention_mask, KeyRows):
        raise RuntimeError("[attention] a KeyRows mask reached a sink call it cannot take")
    n = int(key.shape[-2])
    # a sliding layer's prefill chunk sees [n-T-win+1, n): dropping the rest is exact (the mask zeros it) and
    # O(T*win); decode (T == 1) is unchanged
    start = max(0, (n - T) - win + 1) if (win and T > 1) else 0
    if start:
        key, value = key[..., start:, :], value[..., start:, :]
    nk = n - start
    # the reference's steps - scaled, masked, the sinks joined on, shifted, softmaxed - each computed in the dtype it
    # computes in there (the scores', or the float mask's the add promotes them to), written into one buffer of the
    # dtype the join promotes to, in place: the same values, so the same bits, at the buffer beside the scores and
    # then beside the softmax instead of a new tensor a step, several alive at once; views of a sweep's workspace
    # where one is open here (`ScoresWorkspace`)
    mask = attention_mask[:, :, :, start:n] if attention_mask is not None else None
    raw_dt = torch.promote_types(query.dtype, key.dtype)
    lhs = raw_dt if mask is None else torch.promote_types(raw_dt, mask.dtype)
    cdt = torch.promote_types(lhs, s_aux.dtype)
    ws = _SCORES.get(where(query.device)) if _SCORES else None
    shape = (B, Hq, T, nk + 1)
    held: list[torch.Tensor] | None = None
    if ws is not None:
        views = [
            ws.view("b", (B, Hk, g * T, nk), raw_dt),
            ws.view("a", shape, cdt),
            ws.view("b", shape, cdt),
            ws.view("a", (B, Hq, T, nk), value.dtype),
        ]
        # past what the sweep sized it for: made per call
        held = [t for t in views if t is not None] if all(t is not None for t in views) else None
    if held is not None:
        raw = torch.matmul(query.reshape(B, Hk, g * T, d), key.transpose(-1, -2), out=held[0]).reshape(B, Hq, T, nk)
    else:
        raw = torch.matmul(query.reshape(B, Hk, g * T, d), key.transpose(-1, -2)).reshape(B, Hq, T, nk)
    raw.mul_(scale)
    combined = held[1] if held is not None else raw.new_empty(shape, dtype=cdt)
    scores = combined[..., :nk]
    if mask is not None:
        torch.add(raw, mask, out=scores)  # computed in the two's dtype, then made the buffer's, as the join makes it
    else:
        # no mask handed over (a tier that owns the attention elsewhere): the causal one, and the window - a decode
        # row's too, whose keys past the window a sliding layer must not see
        pos = torch.arange(n - T, n, device=raw.device)[:, None]
        j = start + torch.arange(nk, device=raw.device)[None]
        allow = (j <= pos) if win is None else ((j <= pos) & (j > pos - win))
        raw.masked_fill_(~allow, torch.finfo(raw.dtype).min)
        scores.copy_(raw)
    del raw
    combined[..., nk:] = s_aux.reshape(1, -1, 1, 1)
    combined.sub_(combined.max(dim=-1, keepdim=True).values)
    if held is not None:
        # the kernel `F.softmax` runs, into the buffer the scores were in; then the probabilities in the values' dtype
        # into the one the promoted copy was in, each read before it is written over
        probs = torch.ops.aten._softmax.out(combined, -1, False, out=held[2])
        del combined, scores
        p = held[3]
        p.copy_(probs[..., :-1])
    else:
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        del combined, scores
        p = probs[..., :-1].to(value.dtype)
    res = torch.matmul(p.reshape(B, Hk, g * T, nk), value).reshape(B, Hq, T, d)
    return res.transpose(1, 2).contiguous(), None
