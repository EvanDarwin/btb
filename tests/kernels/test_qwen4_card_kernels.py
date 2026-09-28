# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's card kernels (native/cuda/btb_kernels.cu): each against the transformers module's own math at bf16 - bit
for bit where the kernel reproduces torch's rounding sequence, within an ulp where a reduction runs in another
order - and each row-invariant: row t of a 32-row launch (padding rows included) is the one-row launch of row t,
bit for bit, as the speculative verify needs. A malformed call is refused before it launches."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from btb.engine.forward import path_of
from btb.engine.native import Native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the card's kernels")

T = 32  # a verify pass's width
REAL = 27  # rows past it are padding
H, G, R = 2560, 4, 320  # the 180B's hidden size, streams, hc_lowrank
E, TOPK = 512, 10  # experts, picked a token
HQ, HK, D, ROT = 24, 2, 256, 64  # attention heads, kv heads, head_dim, rotary dims (partial_rotary_factor 0.25)
EPS = 1e-6


@pytest.fixture(scope="module")
def kern() -> Any:
    k = Native.card_kernels()
    if k is None:
        pytest.skip(f"no card kernels: {Native.cuda_reason}")
    return k


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _bf(g: torch.Generator, *shape: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=g) * scale).to("cuda", torch.bfloat16)


def _pad(x: torch.Tensor) -> torch.Tensor:
    """rows REAL.. of a pass replaced by padding (zeros, as a captured width's spare rows may hold anything)"""
    x = x.clone()
    x[REAL:] = 0
    return x


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """the largest distance between two bf16 tensors in units in the last place (signs apart count as far)"""
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    same = (ia < 0) == (ib < 0)
    d = torch.where(same, (ia - ib).abs(), torch.full_like(ia, 1 << 16))
    d = torch.where(a == b, torch.zeros_like(d), d)  # +0 and -0
    return int(d.max()) if d.numel() else 0


def _rows_match(
    full: torch.Tensor | tuple[torch.Tensor, ...], one: Callable[[int], torch.Tensor | tuple[torch.Tensor, ...]]
) -> None:
    """row t of the full launch's outputs is the one-row launch of row t, bit for bit, for every row"""
    fulls = full if isinstance(full, tuple) else (full,)
    for t in range(T):
        ones = one(t)
        ones = ones if isinstance(ones, tuple) else (ones,)
        for f, o in zip(fulls, ones, strict=True):
            assert torch.equal(f[t : t + 1], o), f"row {t} of the {T}-row launch parts from its one-row launch"


def _gemv(kern: Any, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> None:
    """btb_gemv_bf16_m{M}: y [M, R] = x [M, C] w^T"""
    M, C = (int(s) for s in x.shape)
    Rw = int(w.shape[0])
    P, ci = kern.ptr, ctypes.c_int
    kern.launch(f"btb_gemv_bf16_m{M}", ((Rw + 3) // 4, 1, 1), (128, 1, 1), [P(w), P(x), P(y), ci(Rw), ci(C)])


def _rmsnorm(dim: int, w: torch.Tensor, group: int | None = None) -> torch.nn.Module:
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    m = Qwen4ExpTextRMSNorm(dim, group_size=group, eps=EPS).to("cuda", torch.bfloat16)
    with torch.no_grad():
        m.weight.copy_(w)
    return m


# -- the hyper-connections ------------------------------------------------------------------------------------------


def _hc_inputs(seed: int) -> dict[str, torch.Tensor]:
    g = _gen(seed)
    return {
        "h": _pad(_bf(g, T, G * H)),
        "y": _pad(_bf(g, T, H)),
        "inj": _pad((2 * torch.sigmoid(_bf(g, T, G))).bfloat16()),
        "w": _bf(g, G * H, scale=0.1),
    }


def test_hc_rmsnorm_is_the_layers_write_then_the_hc_norm(kern: Any) -> None:
    x = _hc_inputs(1)
    h, xo = x["h"].clone(), torch.empty_like(x["h"])
    kern.hc_rmsnorm(h, x["y"], x["inj"], x["w"], EPS, xo, G)
    # the decoder layer's write, torch's two bf16 ops: bit for bit
    ref_h = x["h"] + (x["y"].unsqueeze(-2) * x["inj"].unsqueeze(-1)).flatten(-2)
    assert torch.equal(h, ref_h)
    # the hc_norm: the mean's sum runs in another order than torch's, so within an ulp
    ref_x = _rmsnorm(G * H, x["w"], group=H)(ref_h)
    assert _ulps(xo, ref_x) <= 1
    # no write: h stays, x is its norm
    h2, xo2 = ref_h.clone(), torch.empty_like(xo)
    kern.hc_rmsnorm(h2, None, None, x["w"], EPS, xo2, G)
    assert torch.equal(h2, ref_h) and torch.equal(xo2, xo)

    def one(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        h1, x1 = x["h"][t : t + 1].clone(), torch.empty(1, G * H, dtype=torch.bfloat16, device="cuda")
        kern.hc_rmsnorm(h1, x["y"][t : t + 1], x["inj"][t : t + 1], x["w"], EPS, x1, G)
        return h1, x1

    _rows_match((h, xo), one)


def test_hc_act_is_silu_of_the_down_projection_over_the_streams(kern: Any) -> None:
    g = _gen(2)
    dn = _pad(_bf(g, T, R + G, scale=4.0))
    act = torch.empty(T, R, dtype=torch.bfloat16, device="cuda")
    kern.hc_act(dn, act, G)
    assert torch.equal(act, F.silu(dn[:, :R] / G))

    def one(t: int) -> torch.Tensor:
        a1 = torch.empty(1, R, dtype=torch.bfloat16, device="cuda")
        kern.hc_act(dn[t : t + 1], a1, G)
        return a1

    _rows_match(act, one)


def test_hc_mix_is_the_streams_mean_and_the_inject_weights(kern: Any) -> None:
    g = _gen(3)
    xn, up = _pad(_bf(g, T, G * H)), _pad(_bf(g, T, G * H, scale=2.0))
    dn = _pad(_bf(g, T, R + G, scale=4.0))
    mixed = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    inj = torch.empty(T, G, dtype=torch.bfloat16, device="cuda")
    kern.hc_mix(xn, up, dn, mixed, inj, G, R)
    ref = (torch.sigmoid(up).unflatten(-1, (G, H)) * xn.unflatten(-1, (G, H))).mean(dim=-2)
    assert torch.equal(mixed, ref)
    assert torch.equal(inj, 2 * torch.sigmoid(dn[:, R : R + G] / G))
    # the final mixer: no inject
    m2 = torch.empty_like(mixed)
    kern.hc_mix(xn, up, None, m2, None, G, R)
    assert torch.equal(m2, mixed)

    def one(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        m1 = torch.empty(1, H, dtype=torch.bfloat16, device="cuda")
        i1 = torch.empty(1, G, dtype=torch.bfloat16, device="cuda")
        kern.hc_mix(xn[t : t + 1], up[t : t + 1], dn[t : t + 1], m1, i1, G, R)
        return m1, i1

    _rows_match((mixed, inj), one)


def test_the_kernels_make_the_gated_residual_modules_output(kern: Any) -> None:
    """Qwen4ExpTextGatedResidual end to end: hc_rmsnorm, the merged [down | inject] gemv, hc_act, the up gemv,
    hc_mix - against the module, within the gemvs' reassociation (their sums run in another order than cuBLAS's)"""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    cfg = Qwen4ExpTextConfig(hidden_size=H, hc_count=G, hc_lowrank=R, rms_norm_eps=EPS)
    torch.manual_seed(4)
    mod = Qwen4ExpTextGatedResidual(cfg).to("cuda", torch.bfloat16)
    with torch.no_grad():
        mod.hc_norm.weight.normal_(0, 0.1)
    h = _bf(_gen(4), T, G * H)
    ref_mixed, _, ref_inj = mod(h)
    x = torch.empty_like(h)
    kern.hc_rmsnorm(h.clone(), None, None, mod.hc_norm.weight.data, EPS, x, G)
    assert mod.block_inject_weight is not None
    w_dn = torch.cat([mod.input_mix_weight_down.weight, mod.block_inject_weight.weight]).contiguous()
    dn = torch.empty(T, R + G, dtype=torch.bfloat16, device="cuda")
    _gemv(kern, w_dn, x, dn)
    act = torch.empty(T, R, dtype=torch.bfloat16, device="cuda")
    kern.hc_act(dn, act, G)
    up = torch.empty(T, G * H, dtype=torch.bfloat16, device="cuda")
    _gemv(kern, mod.input_mix_weight_up.weight.data.contiguous(), act, up)
    mixed = torch.empty(T, H, dtype=torch.bfloat16, device="cuda")
    inj = torch.empty(T, G, dtype=torch.bfloat16, device="cuda")
    kern.hc_mix(x, up, dn, mixed, inj, G, R)
    torch.testing.assert_close(mixed.float(), ref_mixed.float(), rtol=0.03, atol=0.03)
    torch.testing.assert_close(inj.float(), ref_inj.float(), rtol=0.02, atol=0.02)


# -- the MoE --------------------------------------------------------------------------------------------------------


def _pinned(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, pin_memory=True)


def test_moe_route_is_the_routers_pick_and_reaches_the_host(kern: Any) -> None:
    g = _gen(5)
    # distinct logits a row (random bf16 would tie often, and torch's topk breaks a tie its own way): a shuffle of
    # k / 32, exact in bf16; the padding rows every logit equal, a tie of all E
    perm = torch.stack([torch.randperm(E + 1, generator=g) for _ in range(T)])
    logits = _pad(((perm - 256) / 32).to("cuda", torch.bfloat16))
    logits[0, 5] = logits[0, 3] = 9.0  # row 0's top two tie: the lower index ranks first
    idx = torch.empty(T, TOPK, dtype=torch.int32, device="cuda")
    w = torch.empty(T, TOPK, dtype=torch.bfloat16, device="cuda")
    hidx, hw, hseq = (
        _pinned(T, TOPK, dtype=torch.int32),
        _pinned(T, TOPK, dtype=torch.bfloat16),
        _pinned(1, dtype=torch.int32),
    )
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    seq = torch.tensor([7], dtype=torch.int32, device="cuda")
    kern.moe_route(logits, E, TOPK, idx, w, hidx, hw, cnt, seq, hseq)
    torch.cuda.synchronize()
    probs = torch.softmax(logits[:REAL, :E].float(), dim=-1)
    v, i = torch.topk(probs, TOPK, dim=-1)
    ref_w = (v / v.sum(dim=-1, keepdim=True)).bfloat16()
    assert idx[0, 0] == 3 and idx[0, 1] == 5 and w[0, 0] == w[0, 1]
    assert torch.equal(idx[1:REAL].long(), i[1:])
    # the softmax's and the renormalisation's sums run in another order than torch's: within an ulp
    assert _ulps(w[:REAL], ref_w) <= 1
    # a tie picks the lowest indices, in order
    assert torch.equal(idx[REAL:].cpu(), torch.arange(TOPK, dtype=torch.int32).expand(T - REAL, TOPK))
    # the host's copy, and the sequence behind it; the count left zero for the next launch
    assert torch.equal(hidx, idx.cpu()) and torch.equal(hw, w.cpu())
    assert int(hseq[0]) == 7 and int(cnt.item()) == 0

    def one(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        i1 = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")
        w1 = torch.empty(1, TOPK, dtype=torch.bfloat16, device="cuda")
        kern.moe_route(logits[t : t + 1], E, TOPK, i1, w1)
        return i1, w1

    _rows_match((idx, w), one)


def test_moe_combine_is_the_shared_experts_gated_add(kern: Any) -> None:
    g = _gen(6)
    yr, ys = _pad(_bf(g, T, H)), _pad(_bf(g, T, H))
    logits = _pad(_bf(g, T, E + 1, scale=2.0))
    y = torch.empty_like(yr)
    kern.moe_combine(yr, ys, logits, E, y)
    assert torch.equal(y, yr + torch.sigmoid(logits[:, E : E + 1]) * ys)
    inplace = yr.clone()
    kern.moe_combine(inplace, ys, logits, E, inplace)
    assert torch.equal(inplace, y)

    def one(t: int) -> torch.Tensor:
        y1 = torch.empty(1, H, dtype=torch.bfloat16, device="cuda")
        kern.moe_combine(yr[t : t + 1], ys[t : t + 1], logits[t : t + 1], E, y1)
        return y1

    _rows_match(y, one)


# -- the attention --------------------------------------------------------------------------------------------------

WIDTH = HQ * 2 * D + 2 * HK * D  # the merged q (interleaved with its gate) | k | v projection
KOFF, VOFF = HQ * 2 * D, HQ * 2 * D + HK * D


def _gate_ref(att: torch.Tensor, qkv: torch.Tensor) -> torch.Tensor:
    """Qwen4ExpTextAttention's `attn_output * torch.sigmoid(gate)`, the gate chunked from q_proj's rows"""
    n = int(att.shape[0])
    _, gate = torch.chunk(qkv[:, :KOFF].view(n, HQ, 2 * D), 2, dim=-1)
    return att * torch.sigmoid(gate.reshape(n, -1))


def test_the_output_gate_alone_and_folded_into_o_proj(kern: Any) -> None:
    g = _gen(7)
    att, qkv = _pad(_bf(g, T, HQ * D)), _pad(_bf(g, T, WIDTH, scale=2.0))
    w = _bf(g, H, HQ * D, scale=0.02)
    x = torch.empty_like(att)
    kern.sigmoid_mul(att, qkv, x, D, D, 2 * D)
    assert torch.equal(x, _gate_ref(att, qkv))
    for M in kern.GEMV_ROWS:
        y = torch.empty(M, H, dtype=torch.bfloat16, device="cuda")
        kern.gemv_sgate(w, att[:M].contiguous(), qkv[:M].contiguous(), y, D, D, 2 * D)
        plain = torch.empty_like(y)
        _gemv(kern, w, x[:M].contiguous(), plain)
        assert torch.equal(y, plain), f"M {M}: the folded gate parts from btb_gemv over btb_sigmoid_mul"
        if M == T:
            full = y
    # against torch's o_proj: the gemv's sum runs in another order than cuBLAS's
    torch.testing.assert_close(full.float(), F.linear(x, w).float(), rtol=0.02, atol=0.02)

    def one(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        y1 = torch.empty(1, H, dtype=torch.bfloat16, device="cuda")
        kern.gemv_sgate(w, att[t : t + 1], qkv[t : t + 1], y1, D, D, 2 * D)
        x1 = torch.empty(1, HQ * D, dtype=torch.bfloat16, device="cuda")
        kern.sigmoid_mul(att[t : t + 1], qkv[t : t + 1], x1, D, D, 2 * D)
        return y1, x1

    _rows_match((full, x), one)


def _tables(rot: int, positions: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """the model's rotary tables for the first `rot` dims: cat(freqs, freqs), cos/sin in bf16"""
    inv = 1.0 / (1e7 ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    emb = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    emb = torch.cat([emb, emb], dim=-1)
    return emb.cos().to("cuda", torch.bfloat16), emb.sin().to("cuda", torch.bfloat16)


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """apply_rotary_pos_emb on x [T, heads, dim] at positions `pos` [T]: the module's partial rope"""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    q = x.permute(1, 0, 2).unsqueeze(0)
    out = apply_rotary_pos_emb(q, cos=cos[pos].unsqueeze(0), sin=sin[pos].unsqueeze(0))
    return out.squeeze(0).permute(1, 0, 2)


N0, CAP = 100, 256
DEPTH = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0]


def _nrp(
    kern: Any,
    qkv: torch.Tensor,
    n0: int,
    depth: list[int],
    wq: torch.Tensor | None,
    wk: torch.Tensor | None,
    heads: int,
    kv_heads: int,
    dim: int,
    q_stride: int,
    k_off: int,
    v_off: int,
    raw_key: bool,
    tables: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n = int(qkv.shape[0])
    qo = torch.full((n, heads, dim), 7.0, dtype=torch.bfloat16, device="cuda")
    K = torch.zeros(kv_heads, CAP, dim, dtype=torch.bfloat16, device="cuda")
    V = None if raw_key else torch.zeros_like(K)
    kern.norm_rope_part(
        qkv,
        qo,
        K,
        V,
        *tables,
        torch.tensor([n0], dtype=torch.int32, device="cuda"),
        torch.tensor(depth, dtype=torch.int32, device="cuda"),
        heads,
        kv_heads,
        q_stride,
        k_off,
        v_off,
        wq,
        wk,
        EPS,
        True,
        raw_key,
    )
    return qo, K, V


@pytest.mark.parametrize("normed", [False, True])
def test_norm_rope_part_is_the_modules_q_k_and_partial_rope(kern: Any, normed: bool) -> None:
    g = _gen(8)
    qkv = _pad(_bf(g, T, WIDTH))
    wq, wk = (_bf(g, D, scale=0.1), _bf(g, D, scale=0.1)) if normed else (None, None)
    tables = _tables(ROT)
    qo, K, V = _nrp(kern, qkv, N0, DEPTH, wq, wk, HQ, HK, D, 2 * D, KOFF, VOFF, False, tables)
    pos = torch.tensor(DEPTH, device="cuda") + N0
    q = qkv[:, :KOFF].view(T, HQ, 2 * D)[..., :D]
    k = qkv[:, KOFF:VOFF].view(T, HK, D)
    if wq is not None and wk is not None:
        q, k = _rmsnorm(D, wq)(q), _rmsnorm(D, wk)(k)
    ref_q, ref_k = _rope_ref(q, *tables, pos), _rope_ref(k, *tables, pos)
    kc = K[:, N0 : N0 + T].permute(1, 0, 2)
    if normed:  # the norm's sum runs in another order than torch's mean: an ulp there, carried through the rope
        torch.testing.assert_close(qo.float(), ref_q.float(), rtol=0.02, atol=0.02)
        torch.testing.assert_close(kc.float(), ref_k.float(), rtol=0.02, atol=0.02)
    else:  # the rope alone is torch's three bf16 roundings, bit for bit
        assert torch.equal(qo, ref_q) and torch.equal(kc, ref_k)
    assert V is not None
    assert torch.equal(V[:, N0 : N0 + T].permute(1, 0, 2), qkv[:, VOFF:].view(T, HK, D))
    assert not K[:, :N0].any() and not K[:, N0 + T :].any(), "a row landed outside its slot"

    def one(t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # the same position (n0 + depth) and the same slot (n0 + t) as a one-row pass
        q1, K1, V1 = _nrp(
            kern, qkv[t : t + 1], N0 + t, [DEPTH[t] - t], wq, wk, HQ, HK, D, 2 * D, KOFF, VOFF, False, tables
        )
        assert V1 is not None
        return q1, K1[:, N0 + t].unsqueeze(0), V1[:, N0 + t].unsqueeze(0)

    full = (qo, K[:, N0 : N0 + T].permute(1, 0, 2), V[:, N0 : N0 + T].permute(1, 0, 2))
    _rows_match(tuple(f.contiguous() for f in full), one)


def test_norm_rope_part_as_the_indexer(kern: Any) -> None:
    """the QSA indexer: its q heads normed and roped over the first ROT dims, its raw key copied to the arena"""
    hi, di = 8, 128
    g = _gen(9)
    qkv = _pad(_bf(g, T, (hi + 1) * di))
    wq = _bf(g, di, scale=0.1)
    tables = _tables(ROT)
    for w in (None, wq):
        qo, K, V = _nrp(kern, qkv, N0, DEPTH, w, None, hi, 1, di, di, hi * di, 0, True, tables)
        assert V is None
        q = qkv[:, : hi * di].view(T, hi, di)
        ref_q = _rope_ref(q if w is None else _rmsnorm(di, w)(q), *tables, torch.tensor(DEPTH, device="cuda") + N0)
        if w is None:
            assert torch.equal(qo, ref_q)
        else:
            torch.testing.assert_close(qo.float(), ref_q.float(), rtol=0.02, atol=0.02)
        assert torch.equal(K[0, N0 : N0 + T], qkv[:, hi * di :])

        def one(t: int, w: torch.Tensor | None = w) -> tuple[torch.Tensor, torch.Tensor]:
            q1, K1, _ = _nrp(
                kern, qkv[t : t + 1], N0 + t, [DEPTH[t] - t], w, None, hi, 1, di, di, hi * di, 0, True, tables
            )
            return q1, K1[0, N0 + t].unsqueeze(0)

        _rows_match((qo, K[0, N0 : N0 + T].contiguous()), one)


# -- the gated DeltaNet ---------------------------------------------------------------------------------------------

DELTA_SHAPES = [(2, 6, 24, 40, 4), (16, 48, 128, 128, 4)]  # (key heads, value heads, key dim, value dim, conv taps)
PARENTS = [-1, 0, 1, 2, 1, 4, 0, 6, 7, 3]


def _slots(parents: list[int]) -> list[int]:
    """families/qwen4/verify.py `_slots`: a node steps in its parent's slot when it is the parent's last child"""
    last = {p: j for j, p in enumerate(parents)}
    slot: list[int] = []
    n = 0
    for j, p in enumerate(parents):
        if p >= 0 and last[p] == j:
            slot.append(slot[p])
        else:
            slot.append(n)
            n += 1
    return slot


def _delta_inputs(shape: tuple[int, ...], rows: int, seed: int) -> dict[str, Any]:
    hk, hv, dk, dv, K = shape
    g = _gen(seed)
    C = 2 * hk * dk + hv * dv
    ps = C + hv * dv + 2 * hv + 8  # q|k|v, z, b, a, and a few columns of something else
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to("cuda")
    return {
        "proj": r(rows, ps).bfloat16(),
        "offsets": (0, C, C + hv * dv, C + hv * dv + hv),
        "conv_w": r(C, K, scale=0.5),
        "conv0": r(C, K),
        "state": r(hv, dk, dv, scale=0.1),
        "a_log": r(hv, scale=0.5),
        "dt_bias": r(hv, scale=0.5),
        "norm_w": 1.0 + r(dv, scale=0.1),
    }


def _widened(x: dict[str, Any], shape: tuple[int, ...], rows: list[int]) -> dict[str, torch.Tensor]:
    """the f32 kernel's inputs: the projection's columns at `rows`, widened"""
    hk, hv, dk, dv, _K = shape
    oq, oz, ob, oa = x["offsets"]
    C = 2 * hk * dk + hv * dv
    p = x["proj"][rows].float()
    return {
        "mixed": p[:, oq : oq + C].contiguous(),
        "z": p[:, oz : oz + hv * dv].contiguous(),
        "b": p[:, ob : ob + hv].contiguous(),
        "a": p[:, oa : oa + hv].contiguous(),
    }


def _f32(
    kern: Any, x: dict[str, Any], shape: tuple[int, ...], rows: list[int], parents: list[int] | None, gate: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """btb_delta_nodes on the widened rows: (out, the per-node scratch with `parents`, else the chain's state)"""
    hk, hv, dk, dv, _K = shape
    w = _widened(x, shape, rows)
    n = len(rows)
    out = torch.empty(n, hv * dv, device="cuda")
    state = x["state"].clone()
    scratch = torch.empty(n, hv, dk, dv, device="cuda") if parents is not None else None
    par = torch.tensor(parents, dtype=torch.int32, device="cuda") if parents is not None else None
    kern.delta_nodes(
        w["mixed"], w["z"], w["a"], w["b"], x["conv_w"], None, x["conv0"], state, scratch, par,
        x["a_log"], x["dt_bias"], x["norm_w"], EPS, gate, hk, hv, dk, dv, out,
    )  # fmt: skip
    return out, (scratch if scratch is not None else state)


def _b16(
    kern: Any,
    x: dict[str, Any],
    shape: tuple[int, ...],
    rows: list[int] | None,
    parents: list[int] | None,
    slots: list[int] | None,
    gate: int = 1,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """btb_delta_nodes_bf16: (out, the slots' scratch with `parents`, else the chain's state)"""
    hk, hv, dk, dv, _K = shape
    n = len(rows) if rows is not None else int(x["proj"].shape[0])
    i32 = lambda v: None if v is None else torch.tensor(v, dtype=torch.int32, device="cuda")
    out = torch.full((n, hv * dv), 3.0, dtype=torch.bfloat16, device="cuda")
    st = (x["state"] if state is None else state).clone()
    scratch = torch.empty(max(slots or [n - 1]) + 1, hv, dk, dv, device="cuda") if parents is not None else None
    kern.delta_nodes_bf16(
        x["proj"], x["offsets"], i32(rows), x["conv_w"], None, x["conv0"], st, scratch, i32(slots), i32(parents),
        x["a_log"], x["dt_bias"], x["norm_w"], EPS, gate, hk, hv, dk, dv, out,
    )  # fmt: skip
    return out, (scratch if scratch is not None else st)


@pytest.mark.parametrize("gate", [0, 1])
@pytest.mark.parametrize("shape", DELTA_SHAPES)
def test_delta_nodes_bf16_is_the_f32_kernel_on_the_widened_rows(kern: Any, shape: tuple[int, ...], gate: int) -> None:
    x = _delta_inputs(shape, 5, seed=11 + gate)
    out, state = _b16(kern, x, shape, None, None, None, gate)
    ref, ref_state = _f32(kern, x, shape, list(range(5)), None, gate)
    assert torch.equal(out, ref.bfloat16()) and torch.equal(state, ref_state)


@pytest.mark.parametrize("shape", DELTA_SHAPES)
def test_delta_nodes_bf16_steps_a_tree_in_its_slots(kern: Any, shape: tuple[int, ...]) -> None:
    x = _delta_inputs(shape, len(PARENTS), seed=7)
    slots = _slots(PARENTS)
    out, scratch = _b16(kern, x, shape, None, PARENTS, slots)
    ref, ref_scratch = _f32(kern, x, shape, list(range(len(PARENTS))), PARENTS)
    assert torch.equal(out, ref.bfloat16())
    last = {s: j for j, s in enumerate(slots)}
    for s, j in last.items():
        assert torch.equal(scratch[s], ref_scratch[j]), f"slot {s} is not node {j}'s state"
    # every node is the chain of its own path, stepped alone from the cache's state (a pass of one node's width)
    for j in range(len(PARENTS)):
        path = path_of(PARENTS, j)[::-1]
        o, _ = _b16(kern, x, shape, path, None, None)
        assert torch.equal(out[j], o[-1]), f"node {j} parts from its path's chain"


@pytest.mark.parametrize("shape", DELTA_SHAPES)
def test_delta_commit_restep_and_conv_window(kern: Any, shape: tuple[int, ...]) -> None:
    """the commit: the accepted path's rows of the verify pass's projection re-stepped as a chain into the state
    (padding rows after it), then the conv window over the same rows"""
    hk, hv, dk, dv, K = shape
    x = _delta_inputs(shape, len(PARENTS), seed=13)
    path = path_of(PARENTS, 9)[::-1]
    rows = path + [-1] * (8 - len(path))
    out, state = _b16(kern, x, shape, rows, None, None)
    ref, ref_state = _f32(kern, x, shape, path, None)
    assert torch.equal(state, ref_state)
    assert torch.equal(out[: len(path)], ref.bfloat16())
    assert (out[len(path) :] == 3.0).all(), "a padding row was written"
    C = 2 * hk * dk + hv * dv
    for rw, n in ((rows, len(rows)), (path[:1], 1), (path[:2], 2), (None, 1), (None, len(PARENTS))):
        conv = x["conv0"].clone()
        kern.conv_window(
            conv, x["proj"], 0, None if rw is None else torch.tensor(rw, dtype=torch.int32, device="cuda"), n
        )
        picked = [r for r in (rw if rw is not None else list(range(n)))[:n] if r >= 0]
        ref_conv = torch.cat([x["conv0"], x["proj"][picked, :C].float().t()], dim=1)[:, -K:]
        assert torch.equal(conv, ref_conv), f"the window after rows {rw} (n {n})"


def test_delta_node_does_not_depend_on_the_pass_width_or_its_padding(kern: Any) -> None:
    shape = DELTA_SHAPES[1]
    g = torch.Generator().manual_seed(17)
    parents = [-1] + [int(torch.randint(0, j, (1,), generator=g)) for j in range(1, REAL)] + [0] * (T - REAL)
    x = _delta_inputs(shape, T, seed=17)
    rows = list(range(REAL)) + [-1] * (T - REAL)
    full, _ = _b16(kern, x, shape, rows, parents, _slots(parents[:REAL]) + [0] * (T - REAL))
    assert (full[REAL:] == 3.0).all(), "a padding row was written"
    for n in (1, 2, 9, REAL):
        part, _ = _b16(kern, x, shape, rows[:n], parents[:n], _slots(parents[:n]))
        assert torch.equal(part, full[:n]), f"a pass of {n} nodes parts from the padded {T}-row one"
    # and node j is its path's chain, the one-token steps
    for j in (0, 5, REAL - 1):
        o, _ = _b16(kern, x, shape, path_of(parents, j)[::-1], None, None)
        assert torch.equal(full[j], o[-1]), f"node {j} parts from its path's chain"


# -- the per-layer n-gram embedding ---------------------------------------------------------------------------------

PH, PK, PDIL = 256, 4, 3  # a stream's width, the conv's taps and dilation (ngram_size)
PLP = (PK - 1) * PDIL  # the kept window


def _ple_inputs(seed: int) -> dict[str, torch.Tensor]:
    g = _gen(seed)
    C = G * PH
    return {
        "kv": _bf(g, T, C + PH),
        "xq": _bf(g, T, C),
        "wk": _bf(g, C, scale=0.1),
        "wc": _bf(g, C, scale=0.1),
        "conv": _bf(g, C, PK, scale=0.5),
        "pre": _bf(g, C, PLP),
        "h": _bf(g, T, C),
    }


def _ple(
    kern: Any, x: dict[str, torch.Tensor], rows: list[int], par: list[int], pre: torch.Tensor, update: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """btb_ple_gate + btb_ple_conv over the nodes at `rows` of the inputs: (h after, gated, normed)"""
    n = len(rows)
    h = x["h"][rows].clone()
    gated = torch.full((n, G * PH), 3.0, dtype=torch.bfloat16, device="cuda")
    normed = torch.full_like(gated, 3.0)
    kern.ple_nodes(
        x["kv"][rows].contiguous(), x["xq"][rows].contiguous(), x["wk"], x["wc"], x["conv"], pre,
        torch.tensor(par, dtype=torch.int32, device="cuda"), h, gated, normed, G, PDIL, EPS, update,
    )  # fmt: skip
    return h, gated, normed


def test_ple_nodes_are_the_modules_math(kern: Any) -> None:
    """the gate, the gated rows and their norm, the conv along a chain and the add into the streams against the
    torch path's step (families/qwen4/verify.py `_ple_forward`, on the card): within the norms' and the gate's
    reassociation (their sums run in the block's order, not torch's)"""
    x = _ple_inputs(31)
    n = 5
    rows = list(range(n))
    par = [-1, 0, 1, 2, 3]
    h, gated, normed = _ple(kern, x, rows, par, x["pre"].clone())
    kn = _rmsnorm(G * PH, x["wk"], group=PH)(x["kv"][:n, : G * PH]).unflatten(-1, (G, PH))
    q = x["xq"][:n].unflatten(-1, (G, PH))
    gate = (kn * q).sum(dim=-1, keepdim=True) / math.sqrt(PH)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    ref_g = (torch.sigmoid(gate) * x["kv"][:n, G * PH :].unsqueeze(-2)).flatten(-2)
    ref_n = _rmsnorm(G * PH, x["wc"], group=PH)(ref_g)
    torch.testing.assert_close(gated.float(), ref_g.float(), rtol=0.03, atol=0.03)
    torch.testing.assert_close(normed.float(), ref_n.float(), rtol=0.03, atol=0.03)
    # the conv over the kernel's own normed rows, as the host sums its taps: each tap rounded, added in index order
    pre = x["pre"]
    acc = None
    for k in range(PK):
        s = PDIL * (PK - 1 - k)
        tap = torch.stack([normed[j - s] if s <= j else pre[:, PLP - (s - j)] for j in range(n)])
        term = x["conv"][:, k] * tap
        acc = term if acc is None else acc + term
    assert acc is not None
    ref_h = x["h"][:n] + (gated + F.silu(acc))
    assert torch.equal(h, ref_h), "the conv and the adds are the host's roundings"


def test_ple_node_is_its_paths_steps(kern: Any) -> None:
    """every node of a tree (padding rows past it) is its path's one-row steps, each shifting its normed row into
    the kept window, bit for bit; the steps' window is the path's normed rows after the one the pass started from"""
    x = _ple_inputs(32)
    g = torch.Generator().manual_seed(33)
    par = [-1] + [int(torch.randint(max(0, j - 4), j, (1,), generator=g)) for j in range(1, REAL)]
    par = par + [-2] * (T - REAL)
    h, gated, normed = _ple(kern, x, list(range(T)), par, x["pre"].clone())
    assert torch.equal(h[REAL:], x["h"][REAL:]) and (gated[REAL:] == 3.0).all(), "a padding row was written"
    for j in range(REAL):
        path = path_of(par, j)[::-1]
        pre = x["pre"].clone()
        for q in path:
            h1, _g1, n1 = _ple(kern, x, [q], [-1], pre, update=True)
        assert torch.equal(h1[0], h[j]), f"node {j} (depth {len(path) - 1}) parts from its path's steps"
        assert torch.equal(n1[0], normed[j])
        want = torch.cat([x["pre"], normed[path].t()], dim=1)[:, -PLP:]
        assert torch.equal(pre, want), f"node {j}: the steps' window is not the path's normed rows"


# -- misuse ---------------------------------------------------------------------------------------------------------


def test_a_malformed_call_is_refused(kern: Any) -> None:
    k = kern
    g = _gen(21)
    bf = lambda *s: _bf(g, *s)
    h, y, inj, w = bf(2, G * 64), bf(2, 64), bf(2, G), bf(G * 64)
    with pytest.raises(ValueError, match="bfloat16"):
        k.hc_rmsnorm(h, y.float(), inj, w, EPS, torch.empty_like(h), G)
    with pytest.raises(ValueError, match="shapes"):
        k.hc_rmsnorm(h, y, None, w, EPS, torch.empty_like(h), G)
    with pytest.raises(ValueError, match="shapes"):
        k.hc_rmsnorm(h, y, inj, w, EPS, torch.empty_like(h), 3)
    with pytest.raises(ValueError, match="contiguous"):
        k.hc_act(bf(8, 2).t(), bf(2, 4), G)
    with pytest.raises(ValueError, match="shapes"):
        k.hc_act(bf(2, 4), bf(2, 8), G)
    with pytest.raises(ValueError, match="shapes"):
        k.hc_mix(h, h, bf(2, 8), bf(2, 64), bf(2, G), G, 8)  # dn narrower than R + G
    logits = bf(2, 17)
    i2, w2 = torch.empty(2, 4, dtype=torch.int32, device="cuda"), torch.empty(2, 4, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="int32"):
        k.moe_route(logits, 16, 4, i2.long(), w2)
    with pytest.raises(ValueError, match="shapes"):
        k.moe_route(logits, 18, 4, i2, w2)
    with pytest.raises(ValueError, match="shapes"):
        k.moe_route(logits, 16, 33, i2, w2)
    with pytest.raises(ValueError, match="together"):
        k.moe_route(logits, 16, 4, i2, w2, hidx=torch.zeros(2, 4, dtype=torch.int32))
    cnt, seq = torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="pinned"):
        k.moe_route(
            logits, 16, 4, i2, w2, torch.zeros(2, 4, dtype=torch.int32), _pinned(2, 4, dtype=torch.bfloat16), cnt, seq,
            _pinned(1, dtype=torch.int32),
        )  # fmt: skip
    with pytest.raises(ValueError, match="shapes"):
        k.moe_combine(y, y, logits, 17, torch.empty_like(y))
    att, qkv = bf(2, 4 * 16), bf(2, 4 * 32)
    with pytest.raises(ValueError, match="shapes"):
        k.sigmoid_mul(att, qkv, torch.empty_like(att), 16, 20, 32)  # a misaligned gate
    with pytest.raises(ValueError, match="shapes"):
        k.sigmoid_mul(att, qkv, torch.empty_like(att), 16, 32, 32)  # past the row
    with pytest.raises(ValueError, match="shapes"):
        k.gemv_sgate(bf(8, 64), bf(3, 64), bf(3, 128), bf(3, 8), 16, 16, 32)  # M 3
    n0, dep = torch.zeros(1, dtype=torch.int32, device="cuda"), torch.zeros(2, dtype=torch.int32, device="cuda")
    cos, sin = _tables(ROT, 16)
    Kc = torch.zeros(1, 8, 128, dtype=torch.bfloat16, device="cuda")
    q2 = torch.empty(2, 1, 128, dtype=torch.bfloat16, device="cuda")
    qkv2 = bf(2, 3 * 128)
    with pytest.raises(ValueError, match="shapes"):
        k.norm_rope_part(
            qkv2, q2, Kc, Kc.clone(), cos, sin, n0, dep, 1, 1, 128, 128, 256, None, None, EPS, True, True
        )  # a raw key with V
    with pytest.raises(ValueError, match="shapes"):
        k.norm_rope_part(
            qkv2,
            q2,
            Kc,
            None,
            cos[:, :24].contiguous(),
            sin[:, :24].contiguous(),
            n0,
            dep,
            1,
            1,
            128,
            128,
            256,
            None,
            None,
            EPS,
        )  # ROT/2E not a power of two
    with pytest.raises(ValueError, match="int32"):
        k.norm_rope_part(qkv2, q2, Kc, None, cos, sin, n0.long(), dep, 1, 1, 128, 128, 256, None, None, EPS)
    shape = DELTA_SHAPES[0]
    x = _delta_inputs(shape, 2, seed=1)
    hk, hv, dk, dv, _K = shape
    call = lambda **kw: k.delta_nodes_bf16(
        **{
            "proj": x["proj"], "offsets": x["offsets"], "rows": None, "conv_w": x["conv_w"], "conv_b": None,
            "conv0": x["conv0"], "state": x["state"], "scratch": None, "slots": None, "parents": None,
            "a_log": x["a_log"], "dt_bias": x["dt_bias"], "norm_w": x["norm_w"], "eps": EPS, "gate": 1, "hk": hk,
            "hv": hv, "dk": dk, "dv": dv, "out": torch.empty(2, hv * dv, dtype=torch.bfloat16, device="cuda"), **kw,
        }
    )  # fmt: skip
    with pytest.raises(ValueError, match="bfloat16"):
        call(proj=x["proj"].float())
    with pytest.raises(ValueError, match="float32"):
        call(state=x["state"].bfloat16())
    with pytest.raises(ValueError, match="shapes"):
        call(offsets=(0, 1, 2, int(x["proj"].shape[1])))
    with pytest.raises(ValueError, match="slots"):
        call(slots=torch.zeros(2, dtype=torch.int32, device="cuda"))  # slots without a scratch
    with pytest.raises(ValueError, match="int32"):
        call(rows=torch.zeros(2, dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="shapes"):
        k.conv_window(x["conv0"].clone(), x["proj"], 0, None, 3)  # past the rows
    with pytest.raises(ValueError, match="float32"):
        k.conv_window(x["conv0"].bfloat16(), x["proj"], 0, None, 1)
    p = _ple_inputs(34)
    ple = lambda **kw: k.ple_nodes(
        **{
            "kv": p["kv"][:2].contiguous(), "xq": p["xq"][:2].contiguous(), "wk": p["wk"], "wc": p["wc"],
            "conv_w": p["conv"], "pre": p["pre"].clone(), "par": None, "h": p["h"][:2].clone(),
            "gated": torch.empty_like(p["h"][:2]), "normed": torch.empty_like(p["h"][:2]), "streams": G,
            "dilation": PDIL, "eps": EPS, **kw,
        }
    )  # fmt: skip
    with pytest.raises(ValueError, match="shapes"):
        ple(update=True)  # a step's window shifted by two rows
    with pytest.raises(ValueError, match="shapes"):
        ple(pre=p["pre"][:, :4].contiguous())
    with pytest.raises(ValueError, match="bfloat16"):
        ple(h=p["h"][:2].float())
