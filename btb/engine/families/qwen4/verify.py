# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's two layers that carry state along the sequence - the gated DeltaNet and the per-layer n-gram embedding
(PLE) - through the one-token step and the speculative verify pass. Everything else in a Qwen4 layer computes a row
as its own (on the host: the linears through the host's gemv, the router `router`, the activations `rows`) or
through the mask it is given (the sparse attention, `attend` on the host, and its indexer, `qsa`), so a verify pass
over a tree is the module's own forward with the tree's mask (`forward.tree_mask`) and these two layers stepped node
by node, each from its parent's state.

The step and the verify pass run the same code - the host's `btb_delta_step` or the card's `btb_delta_nodes` for the
DeltaNet, the same gathered window for PLE's convolution - so a node computes as the one-token step of its path.
A verify pass leaves the cache's states as they were; the accepted path is stepped into them at the commit
(`ad`), which spares a state kept per node for every layer until then. A prefill (or any pass of several rows
outside speculation) takes the module's own forward."""

from __future__ import annotations

import math
import types
import weakref
from typing import Any

import torch
import torch.nn.functional as F

from ...forward import chain_of, path_of
from ...native import Native
from .rows import each


def _slots(parents: list[int]) -> tuple[list[int], list[bool]]:
    """each node's state slot and whether it starts from a copy: a node steps its parent's slot in place when it is
    the parent's last child (every earlier child took its copy first); otherwise, and at a root, a fresh slot
    starts as a copy of the parent's state (the cache's, at a root)"""
    last: dict[int, int] = {}
    for j, p in enumerate(parents):
        last[p] = j
    slot: list[int] = []
    fresh: list[bool] = []
    n = 0
    for j, p in enumerate(parents):
        if p >= 0 and last[p] == j:
            slot.append(slot[p])
            fresh.append(False)
        else:
            slot.append(n)
            fresh.append(True)
            n += 1
    return slot, fresh


# -- the gated DeltaNet --------------------------------------------------------------------------------------------


def install(layer: Any, sm: Any) -> None:
    """layer's DeltaNet and PLE forwarding through this module; `sm` held weakly (the engine owns the layer)"""
    ref = weakref.ref(sm)
    la = getattr(layer, "linear_attn", None)
    if la is not None:
        la._btb_sm = ref
        la.forward = types.MethodType(_delta_forward, la)
    ple = getattr(layer, "ple", None)
    if ple is not None:
        ple._btb_sm = ref
        ple.forward = types.MethodType(_ple_forward, ple)


def _consts(la: Any, dev: torch.device) -> dict[str, Any]:
    """the layer's float32 operands on `dev`, made once per layer the module holds: a streamed template or a float32
    shadow is refilled in place (its storage unchanged) with the next layer's weights and pointed at that layer
    (`_retarget` sets its `layer_idx`), so the key is the layer beside the weights' storage - one build per layer
    switch, never a first layer's operands kept for the next. A float32 weight on `dev` is its own operand (no copy):
    it follows a refill by itself"""
    key = (dev, int(la.layer_idx), la.conv1d.weight.data_ptr(), la.A_log.data_ptr(), la.norm.weight.data_ptr())
    c = getattr(la, "_btb_consts", None)
    if c is None or c["key"] != key:
        f = lambda t: None if t is None else t.detach().to(dev, torch.float32).contiguous()
        c = la._btb_consts = {
            "key": key,
            "conv_w": f(la.conv1d.weight.squeeze(1)),
            "conv_b": f(la.conv1d.bias),
            "a_log": f(la.A_log),
            "dt_bias": f(la.dt_bias),
            "norm_w": f(la.norm.weight),
            "eps": float(getattr(la.norm, "variance_epsilon", getattr(la.norm, "eps", 1e-6))),
            "gate": 1 if getattr(la.norm, "activation", "silu") == "sigmoid" else 0,
        }
    return c


def _states(sm: Any, cl: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """the layer's (conv [1, C, K], recurrent [1, Hv, dk, dv]) states, the recurrent one float32 and contiguous (a
    prefill may leave it otherwise; it is widened once, in place in the cache, a value the step would compute in
    float32 anyway)"""
    conv, rec = sm._lin(cl)
    if rec.dtype != torch.float32 or not rec.is_contiguous():
        rec = rec.float().contiguous()
        sm._lin_set(cl, conv, rec)
    return conv, rec


def _delta_forward(
    self: Any,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    **kw: Any,
) -> torch.Tensor:
    sm = self._btb_sm()
    B, T, _ = hidden_states.shape
    i = int(self.layer_idx)
    stepping = (
        sm is not None and cache_params is not None and B == 1 and cache_params.has_previous_state(i, state_idx=0)
    )
    spec = stepping and bool(getattr(sm, "aq", False))
    if not stepping or (T > 1 and not spec) or not _kernels_for(hidden_states.device):
        return type(self).forward(self, hidden_states, cache_params, attention_mask, **kw)
    x = hidden_states
    mixed = self.in_proj_qkv(x)[0].float().contiguous()
    z = self.in_proj_z(x)[0].float().contiguous()
    b = self.in_proj_b(x)[0].float().contiguous()
    a = self.in_proj_a(x)[0].float().contiguous()
    cl = cache_params.layers[i]
    core = _nodes(sm, self, cl, mixed, z, a, b, chain_of(sm.ap, T) if spec else None)
    if spec:
        sm.al[i] = _PathStep(sm, self, cl, mixed, z, a, b, _kept(sm, self, i, _consts(self, x.device)))
    return self.out_proj(core.view(1, T, -1).to(x.dtype))


_OPERANDS = ("conv_w", "conv_b", "a_log", "dt_bias", "norm_w")


def _kept(sm: Any, la: Any, i: int, c: dict[str, Any]) -> dict[str, Any]:
    """layer `i`'s operands `c` as the commit (`ad`, after every later layer has run) must read them. A module that
    is the layer's own (a host or resident layer) keeps its weights to the commit, and a rebuild of its operands is
    a new set, so `c` itself. A module that serves other layers too - a streamed template, a float32 shadow - is
    refilled with them before the commit, which a float32 operand aliasing its weight follows, and which rebuilds
    the operands the commit would read off it: those are copied here into the layer's own scratch (a few rows: the
    conv's taps, the heads' decay and bias, the norm)"""
    own = any(getattr(m, "linear_attn", None) is la for m in (sm.host.get(i), sm.resident.get(i)))
    if own:
        return c
    dev = c["conv_w"].device
    ts = [c[k] for k in _OPERANDS if c[k] is not None]
    n = sum(t.numel() for t in ts)
    buf = sm.scratch.take(
        f"qwen4 delta operands {i}", (n,), torch.float32, dev, "Qwen4's DeltaNet: a verify pass's layer operands"
    )
    kept = dict(c)
    at = 0
    for k in _OPERANDS:
        t = c[k]
        if t is None:
            continue
        kept[k] = buf[at : at + t.numel()].view_as(t).copy_(t)
        at += t.numel()
    return kept


def _kernels_for(dev: torch.device) -> bool:
    """whether the step's kernel runs where the layer is: the native library's on the host, the card's there"""
    if dev.type == "cpu":
        return Native.delta_step is not None
    return dev.type == "cuda" and Native.card_kernels() is not None


def _nodes(
    sm: Any,
    la: Any,
    cl: Any,
    mixed: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    parents: list[int] | None,
    c: dict[str, Any] | None = None,
) -> torch.Tensor:
    """the DeltaNet's gated-normed output [T, Hv * dv] float32 for T nodes: a chain stepped into the cache's states
    (`parents` None: the one-token step, or a commit), or a tree stepped in scratch from them (a verify pass); over
    the operands `c` (a commit's, kept at its pass: `_kept`), else the module's now"""
    dev = mixed.device
    c = _consts(la, dev) if c is None else c
    conv, rec = _states(sm, cl)
    T, C = (int(s) for s in mixed.shape)
    hk, hv, dk, dv = int(la.num_k_heads), int(la.num_v_heads), int(la.head_k_dim), int(la.head_v_dim)
    K = int(c["conv_w"].shape[1])
    out = torch.empty(T, hv * dv, dtype=torch.float32, device=dev)
    who = "Qwen4's DeltaNet: a verify pass's node states"
    if dev.type == "cpu":
        step = Native.delta_step
        w, cb, a_log, dt, nw, eps, gate = (
            c[k] for k in ("conv_w", "conv_b", "a_log", "dt_bias", "norm_w", "eps", "gate")
        )
        if parents is None:
            if conv.dtype != torch.float32 or not conv.is_contiguous():
                conv = conv.float().contiguous()
                sm._lin_set(cl, conv, rec)
            for j in range(T):
                step(
                    mixed[j].clone(),
                    conv[0],
                    w,
                    cb,
                    z[j],
                    a[j],
                    b[j],
                    a_log,
                    dt,
                    rec[0],
                    hk,
                    hv,
                    dk,
                    dv,
                    nw,
                    eps,
                    out[j],
                    gate,
                )
            return out
        slot, fresh = _slots(parents)
        n = max(slot) + 1
        conv_s = sm.scratch.take("qwen4 delta conv", (n, C, K), torch.float32, dev, who)
        rec_s = sm.scratch.take("qwen4 delta state", (n, hv, dk, dv), torch.float32, dev, who)
        for j, p in enumerate(parents):
            s = slot[j]
            if fresh[j]:
                conv_s[s].copy_(conv[0] if p < 0 else conv_s[slot[p]])
                rec_s[s].copy_(rec[0] if p < 0 else rec_s[slot[p]])
            step(
                mixed[j].clone(),
                conv_s[s],
                w,
                cb,
                z[j],
                a[j],
                b[j],
                a_log,
                dt,
                rec_s[s],
                hk,
                hv,
                dk,
                dv,
                nw,
                eps,
                out[j],
                gate,
            )
        return out
    kern = Native.card_kernels()
    assert kern is not None  # `_kernels_for` held
    conv0 = conv[0].float().contiguous()
    scratch = par = None
    if parents is not None:
        scratch = sm.scratch.take("qwen4 delta nodes", (T, hv, dk, dv), torch.float32, dev, who)
        par = torch.tensor(parents, dtype=torch.int32).to(dev, non_blocking=True)
    kern.delta_nodes(
        mixed=mixed,
        z=z,
        a=a,
        b=b,
        conv_w=c["conv_w"],
        conv_b=c["conv_b"],
        conv0=conv0,
        state=rec[0],
        scratch=scratch,
        parents=par,
        a_log=c["a_log"],
        dt_bias=c["dt_bias"],
        norm_w=c["norm_w"],
        eps=c["eps"],
        gate=c["gate"],
        hk=hk,
        hv=hv,
        dk=dk,
        dv=dv,
        out=out,
    )
    if parents is None:
        # a chain stepped the cache's state in place: its window's last K rows, the starting window's then the chain's
        # inputs
        conv[0].copy_(torch.cat([conv0, mixed.t()], dim=1)[:, -K:])
    return out


class _PathStep:
    """a verify pass's DeltaNet inputs, kept to step the accepted path into the cache's states at the commit
    (`ad` calls `restore`): the path's rows as a chain from the states the pass started from, which it left as they
    were, over the layer's operands as the pass read them (`c`, from `_kept`: the module may hold another layer's
    weights by the commit)"""

    __slots__ = ("a", "b", "c", "cl", "la", "mixed", "sm", "z")

    def __init__(self, sm: Any, la: Any, cl: Any, mixed: Any, z: Any, a: Any, b: Any, c: dict[str, Any]) -> None:
        self.sm, self.la, self.cl, self.c = weakref.ref(sm), la, cl, c
        self.mixed, self.z, self.a, self.b = mixed, z, a, b

    def restore(self, path: list[int]) -> None:
        sm = self.sm()
        if sm is None:
            return None
        rows = torch.tensor(path, dtype=torch.long, device=self.mixed.device)
        pick = lambda t: t.index_select(0, rows).contiguous()
        _nodes(sm, self.la, self.cl, pick(self.mixed), pick(self.z), pick(self.a), pick(self.b), None, self.c)
        return None


# -- the per-layer n-gram embedding --------------------------------------------------------------------------------


def _ple_forward(
    self: Any,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    past_key_values: Any,
    conv_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    sm = self._btb_sm()
    B, T, _ = hidden_states.shape
    i = int(self.layer_idx)
    stepping = (
        sm is not None
        and past_key_values is not None
        and B == 1
        and conv_mask is None
        and past_key_values.has_previous_state(i, state_idx=1)
        and past_key_values.has_previous_state(i, state_idx=2)
    )
    spec = stepping and bool(getattr(sm, "aq", False))
    if not stepping or (T > 1 and not spec):
        return type(self).forward(self, hidden_states, input_ids, past_key_values, conv_mask)
    cl = past_key_values.layers[i]
    parents = chain_of(sm.ap if spec else None, T)
    ids = input_ids[0].long()
    ctx = cl.conv_states[2][0].long()  # the context's last ngram_size - 1 ids
    pre = cl.conv_states[1][0]  # [4H, (K - 1) * dilation] the convolution's last inputs, oldest first
    emb_mod = self.ple_embedding
    n = int(emb_mod.ngram_size)
    # each node's last n ids along its path, the context before the pass beneath them: the embedding of the
    # reference's last position over them (its eos segments read off the same n ids)
    hist = torch.empty(T, n, dtype=torch.long, device=ids.device)
    paths = [path_of(parents, j) for j in range(T)]
    for j, pth in enumerate(paths):
        for s in range(n):  # s steps back from node j
            hist[j, n - 1 - s] = ids[pth[s]] if s < len(pth) else ctx[len(ctx) - (s - len(pth)) - 1]
    emb = emb_mod(hist, None)[:, -1:, :].reshape(1, T, -1)
    H, hc = int(self.hidden_size), int(self.hc_count)
    key_normed = self.norm_key(self.key_proj(emb)).unflatten(-1, (hc, H))
    value = self.value_proj(emb)
    query_normed = self.norm_query(hidden_states).unflatten(-1, (hc, H))
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    # the activations a node at a time, as its step's (rows.py)
    gated = each(torch.sigmoid, gate, T) * value.unsqueeze(-2)
    normed = self.norm_conv(gated.flatten(-2))[0]  # [T, 4H]
    gated = gated.flatten(-2)
    # the dilated depthwise convolution over each node's path: tap k reads the input dilation * (K - 1 - k) steps
    # back, the pass's own rows first and the kept inputs past them; the taps summed in index order, the same
    # arithmetic for the step and for every node of a verify pass
    w = self.conv1d.weight[:, 0, :].to(normed.dtype)  # [4H, K]
    K, dil = int(w.shape[1]), int(self.conv1d.dilation[0])
    L = int(pre.shape[-1])
    acc = None
    for k in range(K):
        s = dil * (K - 1 - k)
        rows = torch.stack(
            [normed[pth[s]] if s < len(pth) else pre[:, L - (s - len(pth)) - 1].to(normed.dtype) for pth in paths]
        )
        term = w[:, k] * rows
        acc = term if acc is None else acc + term
    assert acc is not None
    out = gated + each(F.silu, acc, T).unsqueeze(0).to(gated.dtype)
    if spec:
        sm.spec_commits.append(_PleCommit(cl, ids, normed, n - 1, L))
    else:
        _ple_keep(cl, ids, normed, n - 1, L)
    return out


def _ple_keep(cl: Any, ids: torch.Tensor, rows: torch.Tensor, n_ids: int, n_rows: int) -> None:
    """PLE's kept state after `ids` [t] and their normed inputs `rows` [t, 4H]: the last `n_ids` ids and the last
    `n_rows` inputs, written into the tensors the cache holds"""
    ctx, pre = cl.conv_states[2], cl.conv_states[1]
    ctx.copy_(torch.cat([ctx[0].long(), ids.to(ctx.device)])[-n_ids:].view_as(ctx))
    pre.copy_(torch.cat([pre[0], rows.t().to(pre.dtype)], dim=1)[:, -n_rows:].view_as(pre))


class _PleCommit:
    """a verify pass's PLE inputs, written into the kept state along the accepted path at the commit"""

    __slots__ = ("cl", "ids", "n_ids", "n_rows", "rows")

    def __init__(self, cl: Any, ids: torch.Tensor, rows: torch.Tensor, n_ids: int, n_rows: int) -> None:
        self.cl, self.ids, self.rows, self.n_ids, self.n_rows = cl, ids, rows, n_ids, n_rows

    def __call__(self, path: list[int]) -> None:
        at = torch.tensor(path, dtype=torch.long, device=self.ids.device)
        _ple_keep(
            self.cl,
            self.ids.index_select(0, at),
            self.rows.index_select(0, at.to(self.rows.device)),
            self.n_ids,
            self.n_rows,
        )
