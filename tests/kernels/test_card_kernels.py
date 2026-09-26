# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's decode kernels against torch's own ops, and the determinism they promise: bit-identical across
runs, row r of a many-row pass equal to the one-row step's, one-row steps equal to one verify pass of the
same path. Needs a CUDA card and the built fatbin; skipped otherwise."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from pytest import CaptureFixture

from tests.helpers import need_card_kernels

if TYPE_CHECKING:
    from btb.engine.native import _Cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")


@pytest.fixture(scope="module")
def cu() -> _Cuda:
    return need_card_kernels()


P = lambda t: ctypes.c_void_p(0 if t is None else t.data_ptr())
I = ctypes.c_int
Fl = ctypes.c_float
dev = "cuda"
bf = torch.bfloat16


def _gemv(cu: _Cuda, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    M, C = x.shape
    R = W.shape[0]
    Mk = next(m for m in (1, 2, 4, 8, 16, 32) if M <= m)
    xp = x if M == Mk else torch.cat([x, torch.zeros(Mk - M, C, device=dev, dtype=x.dtype)])
    y = torch.empty(Mk, R, device=dev, dtype=bf)
    cu.launch(f"btb_gemv_bf16_m{Mk}", ((R + 3) // 4, 1, 1), (128, 1, 1), [P(W), P(xp), P(y), I(R), I(C)])
    return y[:M]


@pytest.mark.parametrize("R,C", [(4096, 1024), (1024, 3072), (2560, 9728), (151936, 1024)])
def test_gemv_matches_linear_and_is_row_deterministic(cu: _Cuda, R: int, C: int) -> None:
    torch.manual_seed(0)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(32, C, device=dev, dtype=bf)
    ref = F.linear(x, W).float()
    y32 = _gemv(cu, W, x)
    # within one bf16 ulp of cuBLAS at the outputs' magnitude
    tol = ref.abs().max().item() * 2**-7
    assert (y32.float() - ref).abs().max().item() <= tol
    # the same bits again, and row r the same whether the pass carries 1, 4 or 32 rows
    assert torch.equal(y32, _gemv(cu, W, x))
    assert torch.equal(y32[:1], _gemv(cu, W, x[:1]))
    assert torch.equal(y32[:4], _gemv(cu, W, x[:4]))


def _attn_ref(q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rows: Sequence[int], scale: float) -> torch.Tensor:
    """query q [Hq, D] over the cache rows `rows` (indices into the cap axis), in fp32"""
    Hq = q.shape[0]
    kk = K[:, rows].float().repeat_interleave(Hq // K.shape[0], 0)
    vv = V[:, rows].float().repeat_interleave(Hq // V.shape[0], 0)
    s = torch.einsum("hd,hnd->hn", q.float(), kk) * scale
    return torch.einsum("hn,hnd->hd", torch.softmax(s, -1), vv).to(bf)


@pytest.mark.parametrize("D", [64, 128, 256])
def test_attention_chain_matches_reference_and_one_row_steps(cu: _Cuda, D: int) -> None:
    torch.manual_seed(1)
    Hq, Hk, cap, n0, T = 8, 4, 640, 300, 6
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    chain = list(range(-1, T - 1))
    out = _attn_split(cu, q, K, V, n0, chain, scale)
    for t in range(T):
        ref = _attn_ref(q[t], K, V, list(range(n0 + t + 1)), scale)
        assert (out[t].float() - ref.float()).abs().max().item() < 4e-3
        # the same row reached by a one-row step at that position
        one = _attn_split(cu, q[t : t + 1], K, V, n0 + t, [-1], scale)
        assert torch.equal(one[0], out[t])
    assert torch.equal(out, _attn_split(cu, q, K, V, n0, chain, scale))


def test_attention_tree_masks_siblings(cu: _Cuda) -> None:
    torch.manual_seed(2)
    Hq, Hk, D, cap, n0 = 16, 8, 128, 512, 200
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(5, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    # nodes 3 and 4 are both children of 2: node 4 sees the prefix, 0, 1, 2 and itself - never node 3
    out = _attn_split(cu, q, K, V, n0, [-1, 0, 1, 2, 2], scale)
    ref4 = _attn_ref(q[4], K, V, list(range(n0 + 3)) + [n0 + 4], scale)
    assert (out[4].float() - ref4.float()).abs().max().item() < 4e-3
    chain = _attn_split(cu, q, K, V, n0, [-1, 0, 1, 2, 3], scale)
    assert torch.equal(out[:4], chain[:4])
    # the branch node's bits are those of a one-row step at its position: the same rows laid out as that
    # step would have them (node 4's row at slot n0 + 3, where the step at depth 3 writes its own)
    K2, V2 = K.clone(), V.clone()
    K2[:, n0 + 3], V2[:, n0 + 3] = K[:, n0 + 4], V[:, n0 + 4]
    one = _attn_split(cu, q[4:5], K2, V2, n0 + 3, [-1], scale)
    assert torch.equal(one[0], out[4])
    # and a second branch that repeats the first branch's rows at other slots gives the first branch's bits
    K3, V3 = K.clone(), V.clone()
    K3[:, n0 + 3], V3[:, n0 + 3] = K[:, n0 + 1], V[:, n0 + 1]
    K3[:, n0 + 4], V3[:, n0 + 4] = K[:, n0 + 2], V[:, n0 + 2]
    q3 = q.clone()
    q3[3], q3[4] = q[1], q[2]
    twin = _attn_split(cu, q3, K3, V3, n0, [-1, 0, 1, 0, 3], scale)
    assert torch.equal(twin[3], out[1]) and torch.equal(twin[4], out[2])


def _attn_split(
    cu: _Cuda,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n0: int,
    par: Sequence[int],
    scale: float,
    split: int = 1024,
    win: int = 0,
) -> torch.Tensor:
    T, Hq, D = q.shape
    Hk, cap = K.shape[0], K.shape[1]
    S = (cap + split - 1) // split
    out = torch.empty(T, Hq, D, device=dev, dtype=bf)
    n0t = torch.tensor([n0], dtype=torch.int32, device=dev)
    pt = torch.tensor(par, dtype=torch.int32, device=dev)
    pm = torch.zeros(S * T * Hq, device=dev)
    pl = torch.zeros(S * T * Hq, device=dev)
    pa = torch.zeros(S * T * Hq * D, device=dev)
    cnt = torch.zeros(T * Hq, dtype=torch.int32, device=dev)
    cu.launch(
        f"btb_attn_split_d{D}",
        (Hq, T, S),
        (256, 1, 1),
        [
            P(q),
            P(K),
            P(V),
            P(out),
            P(n0t),
            P(pt),
            I(T),
            I(Hq),
            I(Hk),
            I(cap),
            Fl(scale),
            P(pm),
            P(pl),
            P(pa),
            P(cnt),
            I(S),
            # the kernel's last parameter (0: the whole prefix, no sliding window); left off, the driver read the
            # argument array past its end - an access violation on Windows
            I(win),
        ],
    )
    assert int(cnt.abs().sum()) == 0, "every (head, row) count is reset by its last block"
    return out


@pytest.mark.parametrize("n0", [300, 1500, 3000])
def test_split_attention_matches_the_reference_across_splits_and_reproduces_one_row_steps(cu: _Cuda, n0: int) -> None:
    torch.manual_seed(7)
    Hq, Hk, D, cap, T = 16, 8, 128, 4096, 4
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    # a tree: root, a wrong branch 1 -> 2, the right node 3 off the root
    par = [-1, 0, 1, 0]
    out = _attn_split(cu, q, K, V, n0, par, scale)
    ref0 = _attn_ref(q[0], K, V, list(range(n0 + 1)), scale)
    ref3 = _attn_ref(q[3], K, V, list(range(n0)) + [n0, n0 + 3], scale)
    assert (out[0].float() - ref0.float()).abs().max().item() < 4e-3
    assert (out[3].float() - ref3.float()).abs().max().item() < 4e-3
    # repeat-identical, and node 3's bits are a one-row step's at position n0 + 1 with its row at slot n0 + 1
    assert torch.equal(out, _attn_split(cu, q, K, V, n0, par, scale))
    K2, V2 = K.clone(), V.clone()
    K2[:, n0 + 1], V2[:, n0 + 1] = K[:, n0 + 3], V[:, n0 + 3]
    one = _attn_split(cu, q[3:4], K2, V2, n0 + 1, [-1], scale)
    assert torch.equal(one[0], out[3])


# ---------------------------------------------------------------------------------------------------------
# the rows kernels: T sequences a token each over one cache - a prefix per row (shared by a fork's rows, end to
# end for a batch's), then every row's step i in the stretch base + i * W, row at its column. Each row must
# come out as its own one-row step over its keys laid end to end, whatever rows step beside it.
# ---------------------------------------------------------------------------------------------------------

RowSpec = tuple[int, int, int] | None  # (column, prefix offset, prefix length); None a padding row


def _rows_layout(base: int, step: int, W: int, rows: Sequence[RowSpec]) -> torch.Tensor:
    flat = [base, step, W]
    for r in rows:
        flat += [0, 0, -1] if r is None else list(r)
    return torch.tensor(flat, dtype=torch.int32, device=dev)


def _row_slots(base: int, step: int, W: int, r: tuple[int, int, int]) -> list[int]:
    """a row's keys in logical order, as slots of the shared cache: its prefix, then its steps to this one"""
    col, off, n = r
    return list(range(off, off + n)) + [base + i * W + col for i in range(step + 1)]


def _attn_rows(
    cu: _Cuda, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rw: torch.Tensor, scale: float, win: int = 0
) -> torch.Tensor:
    T, Hq, D = q.shape
    Hk, cap = K.shape[0], K.shape[1]
    S = (cap + 1023) // 1024
    out = torch.full((T, Hq, D), 7.0, device=dev, dtype=bf)  # a padding row's stays as it was
    pm = torch.zeros(S * T * Hq, device=dev)
    pl = torch.zeros(S * T * Hq, device=dev)
    pa = torch.zeros(S * T * Hq * D, device=dev)
    cnt = torch.zeros(T * Hq, dtype=torch.int32, device=dev)
    cu.launch(
        f"btb_attn_rows_d{D}",
        (Hq, T, S),
        (256, 1, 1),
        [
            P(q),
            P(K),
            P(V),
            P(out),
            P(rw),
            I(T),
            I(Hq),
            I(Hk),
            I(cap),
            Fl(scale),
            P(pm),
            P(pl),
            P(pa),
            P(cnt),
            I(S),
            I(win),
        ],
    )
    assert int(cnt.abs().sum()) == 0, "every (head, row) count is reset by its last block"
    return out


# a fork: one prefix across two splits, the rows' columns shuffled (rows left and a beam reordered), a padding
# row last; a batch: three prompts end to end, their steps past 2K keys, one row padding in the middle
FORK = (1500, 37, 4, [(2, 0, 1500), (0, 0, 1500), (3, 0, 1500), (1, 0, 1500), None])
BATCH = (1900, 700, 3, [(0, 0, 700), None, (1, 700, 1100), (2, 1800, 90)])


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", [FORK, BATCH], ids=["fork", "batch"])
@pytest.mark.parametrize("win", [0, 512])
def test_rows_attention_is_each_rows_own_one_row_step(
    cu: _Cuda, D: int, layout: tuple[int, int, int, list[RowSpec]], win: int
) -> None:
    torch.manual_seed(8)
    base, step, W, rows = layout
    Hq, Hk, T = 8, 4, len(rows)
    cap = (base + (step + 1) * W + 1023) // 1024 * 1024
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    rw = _rows_layout(base, step, W, rows)
    out = _attn_rows(cu, q, K, V, rw, scale, win)
    assert torch.equal(out, _attn_rows(cu, q, K, V, rw, scale, win))
    for t, r in enumerate(rows):
        if r is None:
            assert bool((out[t] == 7.0).all()), "a padding row computes nothing"
            continue
        slots = _row_slots(base, step, W, r)
        seen = slots[-win:] if win else slots
        ref = _attn_ref(q[t], K, V, seen, scale)
        assert (out[t].float() - ref.float()).abs().max().item() < 4e-3
        # the row alone: its keys end to end from slot 0, the one-row step at its position
        n = len(slots)
        K1 = torch.zeros(Hk, (n + 1023) // 1024 * 1024, D, device=dev, dtype=bf)
        V1 = torch.zeros_like(K1)
        K1[:, :n], V1[:, :n] = K[:, slots], V[:, slots]
        one = _attn_split(cu, q[t : t + 1], K1, V1, n - 1, [-1], scale, win=win)
        assert torch.equal(one[0], out[t])


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("layout", [FORK, BATCH], ids=["fork", "batch"])
def test_norm_rope_kv_rows_writes_each_row_as_its_own_step(
    cu: _Cuda, D: int, layout: tuple[int, int, int, list[RowSpec]]
) -> None:
    torch.manual_seed(9)
    base, step, W, rows = layout
    Hq, Hk, T, eps = 8, 4, len(rows), 1e-6
    cap = (base + (step + 1) * W + 1023) // 1024 * 1024
    wq = torch.rand(D, device=dev, dtype=bf) + 0.5
    wk = torch.rand(D, device=dev, dtype=bf) + 0.5
    cos_t, sin_t = _rope_tables(cap, D)
    qkv = torch.randn(T, (Hq + 2 * Hk) * D, device=dev, dtype=bf)
    K = torch.zeros(Hk, cap, D, device=dev, dtype=bf)
    V = torch.zeros_like(K)
    qo = torch.zeros(T, Hq, D, device=dev, dtype=bf)
    rw = _rows_layout(base, step, W, rows)
    common = [I(T), I(Hq), I(Hk), I(cap), I(0)]
    cu.launch(
        f"btb_norm_rope_kv_rows_d{D}",
        (Hq + 2 * Hk, T, 1),
        (32, 1, 1),
        [P(qkv), P(wq), P(wk), Fl(eps), P(cos_t), P(sin_t), P(rw), P(K), P(V), P(qo), *common],
    )
    written = torch.zeros(cap, dtype=torch.bool, device=dev)
    for t, r in enumerate(rows):
        if r is None:
            assert bool((qo[t] == 0).all()), "a padding row writes nothing"
            continue
        pos = r[2] + step
        slot = _row_slots(base, step, W, r)[-1]
        written[slot] = True
        # the tree kernel's one-row step at the row's position, its row at slot `pos` of a cache of its own
        K1, V1 = torch.zeros_like(K), torch.zeros_like(V)
        q1 = torch.empty(1, Hq, D, device=dev, dtype=bf)
        x1 = qkv[t : t + 1].contiguous()
        n0t = torch.tensor([pos], dtype=torch.int32, device=dev)
        d0 = torch.zeros(1, dtype=torch.int32, device=dev)
        cu.launch(
            f"btb_norm_rope_kv_d{D}",
            (Hq + 2 * Hk, 1, 1),
            (32, 1, 1),
            [
                P(x1),
                P(wq),
                P(wk),
                Fl(eps),
                P(cos_t),
                P(sin_t),
                P(n0t),
                P(d0),
                P(K1),
                P(V1),
                P(q1),
                I(1),
                I(Hq),
                I(Hk),
                I(cap),
                I(0),
            ],
        )
        assert torch.equal(qo[t], q1[0])
        assert torch.equal(K[:, slot], K1[:, pos]) and torch.equal(V[:, slot], V1[:, pos])
    assert bool((K[:, ~written] == 0).all()) and bool((V[:, ~written] == 0).all()), "no other slot is touched"


def _rope_tables(cap: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=dev).float() / D))
    fr = torch.outer(torch.arange(cap, device=dev).float(), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos().to(bf).contiguous(), emb.sin().to(bf).contiguous()


@pytest.mark.parametrize("D", [64, 128])
def test_norm_rope_kv_is_bit_exact_against_the_fused_ops(cu: _Cuda, D: int) -> None:
    from btb.engine.fused import _fused_rope

    torch.manual_seed(3)
    Hq, Hk, cap, n0, T, eps = 8, 4, 512, 100, 5, 1e-6
    wq = torch.rand(D, device=dev, dtype=bf) + 0.5
    wk = torch.rand(D, device=dev, dtype=bf) + 0.5
    cos_t, sin_t = _rope_tables(cap, D)
    qkv = torch.randn(T, (Hq + 2 * Hk) * D, device=dev, dtype=bf)
    depth = torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device=dev)
    n0t = torch.tensor([n0], dtype=torch.int32, device=dev)
    K = torch.zeros(Hk, cap, D, device=dev, dtype=bf)
    V = torch.zeros_like(K)
    qo = torch.empty(T, Hq, D, device=dev, dtype=bf)
    cu.launch(
        f"btb_norm_rope_kv_d{D}",
        (Hq + 2 * Hk, T, 1),
        (32, 1, 1),
        [
            P(qkv),
            P(wq),
            P(wk),
            Fl(eps),
            P(cos_t),
            P(sin_t),
            P(n0t),
            P(depth),
            P(K),
            P(V),
            P(qo),
            I(T),
            I(Hq),
            I(Hk),
            I(cap),
            I(0),
        ],
    )
    qs = qkv[:, : Hq * D].view(T, Hq, D)
    ks = qkv[:, Hq * D : (Hq + Hk) * D].view(T, Hk, D)
    vs = qkv[:, (Hq + Hk) * D :].view(T, Hk, D)
    qn, kn = F.rms_norm(qs, (D,), wq, eps), F.rms_norm(ks, (D,), wk, eps)
    pos = n0 + depth.long()
    c, s = cos_t[pos].view(1, T, D), sin_t[pos].view(1, T, D)
    qr, kr = _fused_rope(qn.permute(1, 0, 2)[None], kn.permute(1, 0, 2)[None], c, s)
    assert torch.equal(qo, qr[0].permute(1, 0, 2).contiguous())
    assert torch.equal(K[:, n0 : n0 + T], kr[0])
    assert torch.equal(V[:, n0 : n0 + T], vs.permute(1, 0, 2))
    assert bool((K[:, :n0] == 0).all()) and bool((K[:, n0 + T :] == 0).all()) and bool((V[:, :n0] == 0).all())


@pytest.mark.parametrize("H,centered", [(1024, 0), (2560, 0), (1024, 1)])
def test_add_rmsnorm_is_bit_exact(cu: _Cuda, H: int, centered: bool) -> None:
    torch.manual_seed(4)
    T, eps = 3, 1e-6
    h = torch.randn(T, H, device=dev, dtype=bf)
    y = torch.randn(T, H, device=dev, dtype=bf)
    w = (torch.rand(H, device=dev, dtype=bf) + 0.5) if not centered else (torch.rand(H, device=dev, dtype=bf) - 0.5)
    h2, x = h.clone(), torch.empty_like(h)
    cu.launch("btb_add_rmsnorm", (T, 1, 1), (256, 1, 1), [P(h2), P(y), P(w), Fl(eps), P(x), I(H), I(centered)])
    hr = h + y
    if centered:
        xr = (F.rms_norm(hr.float(), (H,), None, eps) * (1.0 + w.float())).to(bf)
    else:
        xr = F.rms_norm(hr, (H,), w, eps)
    assert torch.equal(h2, hr)
    assert torch.equal(x, xr)
    # without a residual: h untouched
    h3, x0 = h.clone(), torch.empty_like(h)
    cu.launch("btb_add_rmsnorm", (T, 1, 1), (256, 1, 1), [P(h3), P(None), P(w), Fl(eps), P(x0), I(H), I(centered)])
    assert torch.equal(h3, h)


@pytest.mark.parametrize("M", [1, 4, 32])
def test_gemv_with_silu_folded_in_matches_the_two_kernels(cu: _Cuda, M: int) -> None:
    torch.manual_seed(6)
    H, Ii = 1024, 3072
    W = torch.randn(H, Ii, device=dev, dtype=bf)
    gu = torch.randn(M, 2 * Ii, device=dev, dtype=bf)
    m = torch.empty(M, Ii, device=dev, dtype=bf)
    cu.launch("btb_silu_mul", (min(4096, (M * Ii + 255) // 256), 1, 1), (256, 1, 1), [P(gu), P(m), I(M), I(Ii)])
    two = _gemv(cu, W, m)
    one = torch.empty(M, H, device=dev, dtype=bf)
    cu.launch(f"btb_gemv_silu_bf16_m{M}", ((H + 3) // 4, 1, 1), (128, 1, 1), [P(W), P(gu), P(one), I(H), I(Ii)])
    assert torch.equal(one, two)


def test_silu_mul_is_bit_exact(cu: _Cuda) -> None:
    torch.manual_seed(5)
    T, Ii = 4, 3072
    gu = torch.randn(T, 2 * Ii, device=dev, dtype=bf)
    m = torch.empty(T, Ii, device=dev, dtype=bf)
    cu.launch("btb_silu_mul", (min(4096, (T * Ii + 255) // 256), 1, 1), (256, 1, 1), [P(gu), P(m), I(T), I(Ii)])
    assert torch.equal(m, F.silu(gu[:, :Ii]).mul_(gu[:, Ii:]))


def test_scheduler_reports_the_hierarchy(cu: _Cuda) -> None:
    from btb.sysinfo import host_cache_sizes

    lim, win = cu.persist_limit()
    assert 0 < lim <= cu.l2_bytes() <= win or win == 0
    hc = host_cache_sizes()
    assert set(hc) == {"host_l2", "host_l3"}
    assert all(v >= 0 for v in hc.values())


# ---------------------------------------------------------------------------------------------------------
# the tensor-core matvec: one kernel for every row count, the caller padding x to the 32 rows it always
# carries. A row group is 16 weight rows to a 64-thread block, so the grid is ceil(R / 16).
# ---------------------------------------------------------------------------------------------------------

MMA_SHAPES = [(151936, 1024), (19456, 2560), (2560, 9728), (4096, 1024), (6144, 1024), (1024, 3072), (6144, 2560)]


def _mma(cu: _Cuda, W: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    M, C = x.shape
    R = W.shape[0]
    xp = x if M == 32 else torch.cat([x, torch.zeros(32 - M, C, device=dev, dtype=x.dtype)])
    y = torch.empty(32, R, device=dev, dtype=bf)
    cu.launch("btb_gemv_mma_bf16", ((R + 15) // 16, 1, 1), (64, 1, 1), [P(W), P(xp), P(y), I(R), I(C), I(M)])
    return y[:M]


@pytest.mark.parametrize("R,C", MMA_SHAPES + [(1020, 1032), (17, 8), (16, 32)])
def test_gemv_mma_matches_linear_within_one_ulp(cu: _Cuda, R: int, C: int) -> None:
    torch.manual_seed(0)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(32, C, device=dev, dtype=bf)
    ref = F.linear(x, W).float()
    # within one bf16 ulp of cuBLAS at the outputs' magnitude - the k inside a super-tile is permuted, so the
    # fp32 sums differ from cuBLAS's in the last bits and can round to the neighbouring bf16
    assert (_mma(cu, W, x).float() - ref).abs().max().item() <= ref.abs().max().item() * 2**-7


@pytest.mark.parametrize("R,C", [(151936, 1024), (2560, 9728), (1024, 3072), (1020, 1032)])
def test_gemv_mma_is_repeat_identical_and_carries_no_row_across(cu: _Cuda, R: int, C: int) -> None:
    torch.manual_seed(1)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(32, C, device=dev, dtype=bf)
    y = _mma(cu, W, x)
    assert torch.equal(y, _mma(cu, W, x))
    # row 0 is the one-row step's bits whatever stands in the other 31: mma accumulates each output element
    # from its own row of x, so a 1-row step and a 32-row verify pass agree bit for bit
    zeros = x.clone()
    zeros[1:] = 0
    assert torch.equal(y[:1], _mma(cu, W, zeros)[:1])
    other = x.clone()
    other[1:] = torch.randn(31, C, device=dev, dtype=bf)
    assert torch.equal(y[:1], _mma(cu, W, other)[:1])
    # and a 4-row pass is the first four rows of the 32-row one
    assert torch.equal(y[:4], _mma(cu, W, x[:4]))


def test_gemv_mma_streams_the_weights_at_the_cards_ceiling(cu: _Cuda, capsys: CaptureFixture[str]) -> None:
    """Prints GB/s at M=1 and the M=32 / M=1 ratio; asserts nothing about speed - the card is shared."""
    free = torch.cuda.mem_get_info()[0]
    rows = []
    for R, C in MMA_SHAPES:
        nb = R * C * 2
        n = max(2, min(40, int(free * 0.55) // nb))
        if n < 2:
            continue
        Ws: list[torch.Tensor] | None = [torch.randn(R, C, device=dev, dtype=bf) for _ in range(n)]
        x1 = torch.zeros(32, C, device=dev, dtype=bf)
        x1[0] = torch.randn(C, device=dev, dtype=bf)
        x32 = torch.randn(32, C, device=dev, dtype=bf)
        y = torch.empty(32, R, device=dev, dtype=bf)

        def timed(
            xs: torch.Tensor, m: int, R: int = R, C: int = C, Ws: list[torch.Tensor] | None = Ws, y: torch.Tensor = y
        ) -> float:
            def body() -> None:
                assert Ws is not None
                for W in Ws:
                    cu.launch(
                        "btb_gemv_mma_bf16", ((R + 15) // 16, 1, 1), (64, 1, 1), [P(W), P(xs), P(y), I(R), I(C), I(m)]
                    )

            for _ in range(3):
                body()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(6):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                g.replay()
                b.record()
                torch.cuda.synchronize()
                best = min(best, a.elapsed_time(b))
            return best

        t1, t32 = timed(x1, 1), timed(x32, 32)
        rows.append(f"  {R:>7} x {C:<5} {n:>3} mats  M=1 {n * nb / t1 / 1e6:7.1f} GB/s   M=32/M=1 {t32 / t1:.3f}")
        Ws = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with capsys.disabled():
        print("\n[mma gemv] weights streamed from DRAM, 40 distinct matrices where they fit:")
        for r in rows:
            print(r)
