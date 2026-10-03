# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's sparse attention on the card (native/cuda/btb_kernels.cu: btb_qsa_pool, btb_qsa_select,
btb_qsa_attn_split) and the host's gemv on the card (btb_gemv_lane16_f32).

The indexer's picks are held two ways. Against btb's host indexer (families/qwen4/qsa.py `_select_tree`, the
reference's selection row for row): the pooled keys within an ulp (k_layernorm's sum of squares runs in the warp's
order, not torch's mean's) and the picks' agreement reported - the scores' dot products run in the warp's order
where the host's run in cuBLAS's, so a near-tie at the budget's edge may go the other way. And EXACTLY against
themselves: every node's picks, and its attention's bits, are those the one-token step computes over its path once
committed (the prefix and its ancestors as real positions, the node the step's row) - the row invariance the
speculative verify stands on. Under the budget the attention is btb_attn_split's, bit for bit. The lane16 gemv is
the host's `Native.gemv`, bit for bit."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from btb.engine.native import Native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the card's kernels")

T, REAL = 32, 27  # a verify pass's width; rows past REAL are padding (par -2)
HI, DI, ROT = 8, 128, 64  # index heads, index head dim, rotary dims
HQ, HK, D = 24, 2, 256  # the attention's heads, kv heads, head_dim
SCALE = 1.0 / 16
EPS = 1e-6
dev = "cuda"
bf = torch.bfloat16

# (prefix n0, compress ratio r, block_topk): short prefixes where the shallow nodes keep everything and the deep ones
# pick, a prefix whose blocks all overflow the budget, and long ones (a budget of 2048 tokens)
CONFIGS = [(16, 4, 5), (37, 4, 10), (40, 8, 5), (40, 8, 4), (4000, 4, 512), (16000, 8, 256)]


@pytest.fixture(scope="module")
def kern() -> Any:
    k = Native.card_kernels()
    if k is None:
        pytest.skip(f"no card kernels: {Native.cuda_reason}")
    return k


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _i32(v: list[int]) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.int32, device=dev)


def _tables(rot: int, positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """the model's rotary tables for the first `rot` dims: cat(freqs, freqs), cos/sin in bf16"""
    inv = 1.0 / (1e7 ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    emb = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    emb = torch.cat([emb, emb], dim=-1)
    return emb.cos().to(dev, bf).contiguous(), emb.sin().to(dev, bf).contiguous()


def _parents(seed: int) -> list[int]:
    """a tree of REAL nodes (parents ahead of children, deep enough to complete blocks in flight), then padding"""
    g = _gen(seed)
    par = [-1]
    for j in range(1, REAL):
        lo = max(0, j - 3)
        par.append(int(torch.randint(lo, j, (1,), generator=g)))
    return par + [-2] * (T - REAL)


def _anc(par: list[int], t: int) -> list[int]:
    """node t's path among the pass's rows, root first"""
    out = [t]
    while par[out[-1]] >= 0:
        out.append(par[out[-1]])
    return out[::-1]


@dataclass
class Pass:
    n0: int
    r: int
    k: int
    cap: int
    par: list[int]
    raw: torch.Tensor  # [cap, DI] the raw-key arena: the prefix, then the pass's rows at n0 + t
    qi: torch.Tensor  # [T, HI, DI]
    kw: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    pk: torch.Tensor  # the committed blocks' keys after the pass's pool
    pk_len: torch.Tensor
    sel: torch.Tensor = field(init=False)  # the tree's picks
    nsel: torch.Tensor = field(init=False)


def _select(
    kern: Any, x: Pass, qi: torch.Tensor, raw: torch.Tensor, pk: torch.Tensor, n0: int, par: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(qi.shape[0])
    sel = torch.full((n, x.k), -7, dtype=torch.int32, device=dev)
    nsel = torch.full((n,), -7, dtype=torch.int32, device=dev)
    scores = torch.empty(n, int(pk.shape[0]), device=dev)
    kern.qsa_select(qi, pk, raw, _i32([n0]), _i32(par), x.cos, x.sin, x.kw, EPS, x.r, x.k, scores, sel, nsel)
    return sel, nsel


_PASSES: dict[tuple[int, int, int], Pass] = {}


def _pass(kern: Any, n0: int, r: int, k: int) -> Pass:
    """the tree's pool and picks for a config (made once a module)"""
    key = (n0, r, k)
    if key in _PASSES:
        return _PASSES[key]
    g = _gen(n0 * 31 + r * 7 + k)
    cap = n0 + T + 40
    cos, sin = _tables(ROT, cap + 8)
    raw = (torch.randn(cap, DI, generator=g)).to(dev, bf)
    qi = (torch.randn(T, HI, DI, generator=g)).to(dev, bf)
    kw = (torch.randn(DI, generator=g) * 0.1).to(dev, bf)
    pk = torch.zeros(cap // r + 1, DI, dtype=bf, device=dev)
    pk_len = torch.zeros(2, dtype=torch.int32, device=dev)
    kern.qsa_pool(raw, pk, pk_len, _i32([n0]), cos, sin, kw, EPS, r, n0 // r + 1)
    x = Pass(n0, r, k, cap, _parents(n0 + r), raw, qi, kw, cos, sin, pk, pk_len)
    x.sel, x.nsel = _select(kern, x, qi, raw, pk, n0, x.par)
    _PASSES[key] = x
    return x


def _keys_of(x: Pass, t: int, sel: torch.Tensor, nsel: torch.Tensor) -> list[int]:
    """node t's attended cache rows, in its list's order: its picked blocks' positions, then its partial tail"""
    anc = _anc(x.par, t)
    n = x.n0 + len(anc)
    nb = n // x.r
    ns = int(nsel[t])
    pos = [int(b) * x.r + i for b in sel[t, :ns].tolist() for i in range(x.r)] + list(range(nb * x.r, n))
    return [p if p < x.n0 else x.n0 + anc[p - x.n0] for p in pos]


def _host(x: Pass) -> tuple[torch.Tensor, SimpleNamespace]:
    """btb's host indexer (`_select_tree`) over the same keys and queries: [REAL, n0 + REAL] bool"""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    from btb.engine.families.qwen4.qsa import _prefix_tree, _select_tree

    norm = Qwen4ExpTextRMSNorm(DI, eps=EPS).to(dev, bf)
    with torch.no_grad():
        norm.weight.copy_(x.kw)
    me = SimpleNamespace(k_layernorm=norm)
    kv = x.n0 + REAL
    vis = torch.zeros(REAL, kv, dtype=torch.bool, device=dev)
    vis[:, : x.n0] = True
    depth = []
    for t in range(REAL):
        anc = _anc(x.par, t)
        depth.append(len(anc) - 1)
        vis[t, [x.n0 + a for a in anc]] = True
    vis = vis[None, None]
    tree = _prefix_tree(vis, REAL)
    assert tree is not None
    pos = torch.cat([torch.arange(x.n0), x.n0 + torch.tensor(depth)]).to(dev)
    with torch.no_grad():
        mask = _select_tree(
            me, x.qi[None, :REAL], x.raw[None, :kv], x.cos[pos][None], x.sin[pos][None], vis, tree, x.n0, x.r, x.k, DI
        )
    return mask[0, 0], me


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    ia, ib = a.view(torch.int16).int(), b.view(torch.int16).int()
    same = (ia < 0) == (ib < 0)
    d = torch.where(same, (ia - ib).abs(), torch.full_like(ia, 1 << 16))
    d = torch.where(a == b, torch.zeros_like(d), d)
    return int(d.max()) if d.numel() else 0


# -- the pool -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n0,r,k", CONFIGS)
def test_pooled_keys_are_the_host_indexers(kern: Any, n0: int, r: int, k: int) -> None:
    from btb.engine.families.qwen4.qsa import _pooled

    x = _pass(kern, n0, r, k)
    nc = n0 // r
    assert x.pk_len.tolist() == [nc, 0], "the pool leaves the committed blocks' count and a zero arrival count"
    _, me = _host(x)
    blocks = torch.arange(nc * r, device=dev).view(nc, r)
    with torch.no_grad():
        ref = _pooled(me, x.raw, blocks, x.cos, x.sin)
    got = x.pk[:nc]
    same = float((got == ref).float().mean())
    print(f"\n[qsa pool n0 {n0} r {r}] pooled keys equal to the host's: {same:.4%}, worst {_ulps(got, ref)} ulp")
    # the mean and the rope are the module's roundings; the norm's sum of squares runs in another order than torch's
    # mean, an ulp of the normed value, carried through the rope's three roundings
    assert _ulps(got, ref) <= 2
    assert not x.pk[nc:].any(), "a block past the committed ones was written"


def test_the_pool_catches_up_in_steps_and_rewinds(kern: Any) -> None:
    x = _pass(kern, 4000, 4, 512)
    pk = torch.zeros_like(x.pk)
    pk_len = torch.zeros(2, dtype=torch.int32, device=dev)
    n0 = torch.tensor([37], dtype=torch.int32, device=dev)
    kern.qsa_pool(x.raw, pk, pk_len, n0, x.cos, x.sin, x.kw, EPS, x.r, 3)
    assert pk_len.tolist() == [3, 0], "a launch pools at most max_new blocks"
    for n in (37, 100, 1000, 4000, 4000):
        n0.fill_(n)
        while int(pk_len[0]) < n // x.r:
            kern.qsa_pool(x.raw, pk, pk_len, n0, x.cos, x.sin, x.kw, EPS, x.r, 57)
    assert pk_len.tolist() == [1000, 0]
    assert torch.equal(pk, x.pk), "blocks pooled a few a launch part from one launch's"
    n0.fill_(401)  # a rewind: the length comes back, the next growth pools again
    kern.qsa_pool(x.raw, pk, pk_len, n0, x.cos, x.sin, x.kw, EPS, x.r, 8)
    assert pk_len.tolist() == [100, 0]


# -- the picks ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n0,r,k", CONFIGS)
def test_picks_against_the_host_indexer(kern: Any, n0: int, r: int, k: int) -> None:
    x = _pass(kern, n0, r, k)
    mask, _ = _host(x)
    kv = n0 + REAL
    rows_same, picked, shared, full = 0, 0, 0, 0
    for t in range(REAL):
        card = torch.zeros(kv, dtype=torch.bool, device=dev)
        card[_keys_of(x, t, x.sel, x.nsel)] = True
        n = n0 + len(_anc(x.par, t))
        if n // r <= k:
            full += 1
            assert torch.equal(card, mask[t]), f"node {t} keeps everything it sees, as the host does"
        rows_same += int(torch.equal(card, mask[t]))
        picked += int(mask[t].sum())
        shared += int((card & mask[t]).sum())
    print(
        f"\n[qsa picks n0 {n0} r {r} k {k}] {full}/{REAL} nodes under the budget; rows equal to the host's "
        f"{rows_same}/{REAL}; tokens the host keeps that the card keeps {shared / picked:.4%}"
    )
    # the scores' sums run in another order than the host's matmul: a near-tie at the budget's edge may go the other
    # way, never more than a sliver of the picks
    assert shared / picked >= 0.97
    # every list is ascending, of k blocks past the budget
    for t in range(REAL):
        ns = int(x.nsel[t])
        s = x.sel[t, :ns].tolist()
        assert s == sorted(set(s)) and ns == min(k, (n0 + len(_anc(x.par, t))) // r)
    assert (x.nsel[REAL:] == 0).all(), "a padding row picked blocks"


@pytest.mark.parametrize("n0,r,k", CONFIGS)
def test_each_nodes_picks_are_its_committed_paths_one_token_step(kern: Any, n0: int, r: int, k: int) -> None:
    """node t's picks == the T = 1 launch's over its path committed: the prefix and its ancestors real positions
    (their raw keys at slots n0 .., pooled into pk by the pool), the node the step's row at slot n0 + depth"""
    x = _pass(kern, n0, r, k)
    for t in range(REAL):
        anc = _anc(x.par, t)
        d = len(anc) - 1
        raw1 = x.raw.clone()
        raw1[n0 : n0 + d + 1] = x.raw[[n0 + a for a in anc]]
        pk1, pl1 = x.pk.clone(), x.pk_len.clone()
        kern.qsa_pool(raw1, pk1, pl1, _i32([n0 + d]), x.cos, x.sin, x.kw, EPS, r, d // r + 2)
        sel1, nsel1 = _select(kern, x, x.qi[t : t + 1], raw1, pk1, n0 + d, [-1])
        ns = int(x.nsel[t])
        assert int(nsel1[0]) == ns and torch.equal(sel1[0, :ns], x.sel[t, :ns]), (
            f"node {t} (depth {d}) picks other blocks than its committed path's one-token step"
        )


# -- the attention --------------------------------------------------------------------------------------------------


def _attn(
    kern: Any,
    x: Pass,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n0: int,
    par: list[int],
    sel: torch.Tensor,
    nsel: torch.Tensor,
) -> torch.Tensor:
    n = int(q.shape[0])
    S = kern.qsa_splits(x.r, x.k)
    out = torch.full((n, HQ, D), 3.0, dtype=bf, device=dev)
    pm, pl = torch.zeros(S * n * HQ, device=dev), torch.zeros(S * n * HQ, device=dev)
    pa = torch.zeros(S * n * HQ * D, device=dev)
    cnt = torch.zeros(n * HQ, dtype=torch.int32, device=dev)
    kern.qsa_attn_split(q, K, V, out, _i32([n0]), _i32(par), sel, nsel, x.r, x.k, SCALE, pm, pl, pa, cnt)
    assert int(cnt.abs().sum()) == 0, "every (head, row) count is reset by its last block"
    return out


def _kv(x: Pass, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = _gen(seed)
    K = torch.randn(HK, x.cap, D, generator=g).to(dev, bf)
    V = torch.randn(HK, x.cap, D, generator=g).to(dev, bf)
    q = torch.randn(T, HQ, D, generator=g).to(dev, bf)
    return q, K, V


def _attn_ref(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rows: list[int]) -> torch.Tensor:
    kk = K[:, rows].float().repeat_interleave(HQ // HK, 0)
    vv = V[:, rows].float().repeat_interleave(HQ // HK, 0)
    s = torch.einsum("hd,hnd->hn", q.float(), kk) * SCALE
    return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), vv)


@pytest.mark.parametrize("n0,r,k", CONFIGS)
def test_attention_over_the_picks(kern: Any, n0: int, r: int, k: int) -> None:
    x = _pass(kern, n0, r, k)
    q, K, V = _kv(x, n0 + 1)
    out = _attn(kern, x, q, K, V, n0, x.par, x.sel, x.nsel)
    assert (out[REAL:] == 3.0).all(), "a padding row was written"
    for t in range(REAL):
        ref = _attn_ref(q[t], K, V, _keys_of(x, t, x.sel, x.nsel))
        # within the output's bf16 rounding of the fp32 reference (its sums in another order)
        torch.testing.assert_close(
            out[t].float(), ref, rtol=8e-3, atol=2e-3, msg=f"node {t}: not the attention over its picks"
        )
    assert torch.equal(out, _attn(kern, x, q, K, V, n0, x.par, x.sel, x.nsel)), "the same bits again"
    # each node's bits are the one-token step's over its committed path: its ancestors' rows at slots n0 .., its own
    # at n0 + depth, its picks the T = 1 select's
    for t in range(REAL):
        anc = _anc(x.par, t)
        d = len(anc) - 1
        rows = [n0 + a for a in anc]
        K1, V1 = K.clone(), V.clone()
        K1[:, n0 : n0 + d + 1], V1[:, n0 : n0 + d + 1] = K[:, rows], V[:, rows]
        raw1 = x.raw.clone()
        raw1[n0 : n0 + d + 1] = x.raw[rows]
        pk1, pl1 = x.pk.clone(), x.pk_len.clone()
        kern.qsa_pool(raw1, pk1, pl1, _i32([n0 + d]), x.cos, x.sin, x.kw, EPS, r, d // r + 2)
        sel1, nsel1 = _select(kern, x, x.qi[t : t + 1], raw1, pk1, n0 + d, [-1])
        one = _attn(kern, x, q[t : t + 1].contiguous(), K1, V1, n0 + d, [-1], sel1, nsel1)
        assert torch.equal(one[0], out[t]), f"node {t} (depth {d}) parts from its committed path's one-token step"


def _attn_split(kern: Any, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n0: int, par: list[int]) -> torch.Tensor:
    n = int(q.shape[0])
    cap = int(K.shape[1])
    S = (cap + 1023) // 1024
    out = torch.empty(n, HQ, D, dtype=bf, device=dev)
    pm, pl = torch.zeros(S * n * HQ, device=dev), torch.zeros(S * n * HQ, device=dev)
    pa = torch.zeros(S * n * HQ * D, device=dev)
    cnt = torch.zeros(n * HQ, dtype=torch.int32, device=dev)
    n0t, part = _i32([n0]), _i32(par)  # held past the launch: a freed temporary's block is the next one's
    P, ci = kern.ptr, ctypes.c_int
    kern.launch(
        f"btb_attn_split_d{D}",
        (HQ, n, S),
        (256, 1, 1),
        [
            P(q), P(K), P(V), P(out), P(n0t), P(part), ci(n), ci(HQ), ci(HK), ci(K.stride(0)), ci(K.stride(1)),
            ctypes.c_float(SCALE), P(pm), P(pl), P(pa), P(cnt), ci(S), ci(0),
        ],
    )  # fmt: skip
    return out


@pytest.mark.parametrize("n0,r,k", [(300, 4, 200), (3000, 4, 800), (2040, 8, 260)])
def test_under_the_budget_it_is_btb_attn_split(kern: Any, n0: int, r: int, k: int) -> None:
    """every node keeping its whole sequence: the list is the positions in order, walked as btb_attn_split walks
    them - across splits too (3000 + the tree: three splits of 1024)"""
    x = _pass(kern, n0, r, k)
    assert all(int(v) == (n0 + len(_anc(x.par, t))) // r for t, v in enumerate(x.nsel[:REAL].tolist()))
    q, K, V = _kv(x, n0 + 2)
    out = _attn(kern, x, q, K, V, n0, x.par, x.sel, x.nsel)
    plain = _attn_split(kern, q, K, V, n0, x.par)
    assert torch.equal(out[:REAL], plain[:REAL])


# -- the host's gemv on the card ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "R,C", [(2560, 640), (640, 2560), (1280, 2560), (7, 5), (33, 17), (129, 1000), (1, 16), (300, 529)]
)
def test_gemv_lane16_is_the_hosts_gemv(kern: Any, R: int, C: int) -> None:
    gemv = Native.gemv
    if gemv is None:
        pytest.skip("no native CPU gemv")
    g = _gen(R * 131 + C)
    w = torch.randn(R, C, generator=g).bfloat16()
    x = torch.randn(32, C, generator=g) * torch.rand(32, 1, generator=g) * 4
    host = torch.empty(32, R)
    gemv(w, x, host)
    wc, xc = w.to(dev), x.to(dev)
    for M in kern.GEMV_ROWS:
        y = torch.full((M, R), 7.0, device=dev)
        kern.gemv_lane16(wc, xc[:M].contiguous(), y)
        assert torch.equal(y.cpu(), host[:M]), f"M {M}: the card's lane16 gemv parts from the host's bits"
    # row m of the 32-row launch is the one-row launch's
    full = torch.empty(32, R, device=dev)
    kern.gemv_lane16(wc, xc, full)
    for m in (0, 5, 31):
        one = torch.empty(1, R, device=dev)
        kern.gemv_lane16(wc, xc[m : m + 1].contiguous(), one)
        assert torch.equal(one[0], full[m])


# -- misuse ---------------------------------------------------------------------------------------------------------


def test_a_malformed_call_is_refused(kern: Any) -> None:
    x = _pass(kern, 16, 4, 5)
    k, pk, pl, n0 = kern, x.pk, x.pk_len.clone(), _i32([16])
    with pytest.raises(ValueError, match="bfloat16"):
        k.qsa_pool(x.raw.float(), pk, pl, n0, x.cos, x.sin, x.kw, EPS, 4, 4)
    with pytest.raises(ValueError, match="int32"):
        k.qsa_pool(x.raw, pk, pl.long(), n0, x.cos, x.sin, x.kw, EPS, 4, 4)
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_pool(x.raw, pk[:3].contiguous(), pl, n0, x.cos, x.sin, x.kw, EPS, 4, 4)  # pk short of cap // r
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_pool(x.raw, pk, pl[:1], n0, x.cos, x.sin, x.kw, EPS, 4, 4)  # no arrival count
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_pool(x.raw, pk, pl, n0, x.cos[:, :24].contiguous(), x.sin[:, :24].contiguous(), x.kw, EPS, 4, 4)
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_pool(x.raw, pk, pl, n0, x.cos, x.sin, x.kw, EPS, 0, 4)
    sc = torch.empty(T, int(pk.shape[0]), device=dev)
    sel, nsel = torch.empty(T, 5, dtype=torch.int32, device=dev), torch.empty(T, dtype=torch.int32, device=dev)
    par = _i32(x.par)
    call = lambda **kw: k.qsa_select(
        **{
            "qi": x.qi, "pk": pk, "raw": x.raw, "n0": n0, "par": par, "cos": x.cos, "sin": x.sin, "kw": x.kw,
            "eps": EPS, "ratio": 4, "k_top": 5, "scores": sc, "sel": sel, "nsel": nsel, **kw,
        }
    )  # fmt: skip
    call()
    with pytest.raises(ValueError, match="float32"):
        call(scores=sc.bfloat16())
    with pytest.raises(ValueError, match="shapes"):
        call(k_top=6)  # sel is [T, 5]
    with pytest.raises(ValueError, match="shapes"):
        call(scores=sc[:, :-1].contiguous())
    with pytest.raises(ValueError, match="shapes"):
        big = torch.zeros(40, HI, DI, dtype=bf, device=dev)  # past the walk's 32 rows
        call(qi=big, par=_i32([-1] * 40))
    with pytest.raises(ValueError, match="shapes"):
        call(qi=torch.zeros(T, 200, DI, dtype=bf, device=dev))  # the queries past the block's shared memory
    q, K, V = _kv(x, 1)
    out = torch.empty_like(q)
    S = k.qsa_splits(4, 5)
    pm, pa = torch.zeros(S * T * HQ, device=dev), torch.zeros(S * T * HQ * D, device=dev)
    cnt = torch.zeros(T * HQ, dtype=torch.int32, device=dev)
    k.qsa_attn_split(q, K, V, out, n0, par, x.sel, x.nsel, 4, 5, SCALE, pm, pm.clone(), pa, cnt)
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_attn_split(q, K, V, out, n0, par, x.sel, x.nsel, 4, 5, SCALE, pm[:-1], pm, pa, cnt)
    with pytest.raises(ValueError, match="shapes"):
        k.qsa_attn_split(q, K[:1].contiguous(), V[:1].contiguous(), out, n0, par, x.sel, x.nsel, 4, 6, SCALE, pm, pm,
                         pa, cnt)  # fmt: skip
    with pytest.raises(ValueError, match="int32"):
        k.qsa_attn_split(q, K, V, out, n0, par.long(), x.sel, x.nsel, 4, 5, SCALE, pm, pm, pa, cnt)
    w = torch.zeros(8, 16, dtype=bf, device=dev)
    with pytest.raises(ValueError, match="float32"):
        k.gemv_lane16(w, torch.zeros(1, 16, dtype=bf, device=dev), torch.zeros(1, 8, device=dev))
    with pytest.raises(ValueError, match="shapes"):
        k.gemv_lane16(w, torch.zeros(3, 16, device=dev), torch.zeros(3, 8, device=dev))  # M 3
    with pytest.raises(ValueError, match="shapes"):
        k.gemv_lane16(w, torch.zeros(2, 16, device=dev), torch.zeros(2, 9, device=dev))
