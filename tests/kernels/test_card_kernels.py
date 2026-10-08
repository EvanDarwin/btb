# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's decode kernels against torch's own ops, and the determinism they promise: bit-identical across
runs, row r of a many-row pass equal to the one-row step's, one-row steps equal to one verify pass of the
same path. Needs a CUDA card and the built fatbin; skipped otherwise."""

from __future__ import annotations

import ctypes
import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn.functional as F
from pytest import CaptureFixture

from btb.engine.cuda import _CudaMixin
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


PAGE = 64  # the prefix cache's page: rows a page of every layer (btb/engine/kvpool.py)


def _paged(K: torch.Tensor, V: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`K`/`V` [Hk, cap, D]'s rows moved into PAGE-row pages of a position-major buffer twice the size (the pool's
    layout) in a shuffled order, the rest of it noise: (the row map, the paged K, the paged V)"""
    g = torch.Generator().manual_seed(seed)
    Hk, cap, D = K.shape
    pages = cap // PAGE
    order = torch.randperm(2 * pages, generator=g)[:pages]
    tbl = (order[:, None] * PAGE + torch.arange(PAGE)[None]).flatten().to(dev, torch.int32)
    Kp = _position_major(torch.randn(Hk, 2 * cap, D, device=dev, dtype=bf))
    Vp = _position_major(torch.randn(Hk, 2 * cap, D, device=dev, dtype=bf))
    Kp[:, tbl.long()], Vp[:, tbl.long()] = K, V
    return tbl, Kp, Vp


@pytest.mark.parametrize("D", [64, 128, 256])
def test_a_row_map_writes_each_row_into_its_page(cu: _Cuda, D: int) -> None:
    """the norm/rope write through a row map (`btb_norm_rope_kv_tbl_d{D}`): each of the pass's rows lands at the
    slot its map names - the bits the plain write leaves at its logical slot, every other row untouched - and the
    query rows are the plain write's"""
    torch.manual_seed(14)
    Hq, Hk, cap, n0, T, eps = 8, 4, 512, 100, 5, 1e-6
    wq = torch.rand(D, device=dev, dtype=bf) + 0.5
    wk = torch.rand(D, device=dev, dtype=bf) + 0.5
    cos_t, sin_t = _rope_tables(cap, D)
    qkv = torch.randn(T, (Hq + 2 * Hk) * D, device=dev, dtype=bf)
    depth = torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32, device=dev)
    n0t = torch.tensor([n0], dtype=torch.int32, device=dev)
    zeros = torch.zeros(Hk, cap, D, device=dev, dtype=bf)
    tbl, Kp, Vp = _paged(zeros, zeros, D)
    Kp.zero_()
    Vp.zero_()
    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for K, V, mapped in ((zeros.clone(), zeros.clone(), None), (Kp, Vp, tbl)):
        qo = torch.empty(T, Hq, D, device=dev, dtype=bf)
        cu.launch(
            f"btb_norm_rope_kv{'_tbl' if mapped is not None else ''}_d{D}",
            (Hq + 2 * Hk, T, 1),
            (32, 1, 1),
            [P(qkv), P(wq), P(wk), Fl(eps), P(cos_t), P(sin_t), P(n0t), P(depth), P(K), P(V), P(qo)]
            + [I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), I(0)]
            + ([] if mapped is None else [P(mapped)]),
        )
        out.append((K, V, qo))
    (k0, v0, q0), (k1, v1, q1) = out
    assert torch.equal(q0, q1)
    rows = tbl[n0 : n0 + T].long()
    assert torch.equal(k1[:, rows], k0[:, n0 : n0 + T]) and torch.equal(v1[:, rows], v0[:, n0 : n0 + T])
    touched = torch.zeros(Kp.shape[1], dtype=torch.bool, device=dev)
    touched[rows] = True
    assert bool((k1[:, ~touched] == 0).all()) and bool((v1[:, ~touched] == 0).all()), "no other row is touched"


def _position_major(K: torch.Tensor) -> torch.Tensor:
    """`K` [Hk, cap, D] laid out position-major - row by row, a row's heads side by side, as the card arena keeps
    them - and viewed back as [Hk, cap, D]"""
    return K.permute(1, 0, 2).contiguous().permute(1, 0, 2)


@pytest.mark.parametrize("Hq,Hk,D", [(16, 8, 128), (32, 8, 128), (16, 8, 64)])
def test_position_major_rows_are_written_as_head_major_ones(cu: _Cuda, Hq: int, Hk: int, D: int) -> None:
    """The card arena keeps its rows position-major so they grow at its end alone. The norm/rope write takes head g's
    row r at g * hs + r * rs, and over either layout lands a pass's rows at the same logical slots, the same bits (the
    one attention reads either layout alike: `test_one_attention_reads_rows_where_they_lie`)"""
    torch.manual_seed(11)
    cap, n0, T, eps = 4096, 1500, 4, 1e-6
    Kp = _position_major(torch.zeros(Hk, cap, D, device=dev, dtype=bf))
    assert (Kp.stride(0), Kp.stride(1)) == (D, Hk * D)
    wq = torch.rand(D, device=dev, dtype=bf) + 0.5
    wk = torch.rand(D, device=dev, dtype=bf) + 0.5
    cos_t, sin_t = _rope_tables(cap, D)
    qkv = torch.randn(T, (Hq + 2 * Hk) * D, device=dev, dtype=bf)
    depth = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device=dev)
    n0t = torch.tensor([n0], dtype=torch.int32, device=dev)
    written = []
    for lay in (lambda t: t, _position_major):
        Kw, Vw = lay(torch.zeros(Hk, cap, D, device=dev, dtype=bf)), lay(torch.zeros(Hk, cap, D, device=dev, dtype=bf))
        qo = torch.empty(T, Hq, D, device=dev, dtype=bf)
        cu.launch(
            f"btb_norm_rope_kv_d{D}",
            (Hq + 2 * Hk, T, 1),
            (32, 1, 1),
            [P(qkv), P(wq), P(wk), Fl(eps), P(cos_t), P(sin_t), P(n0t), P(depth), P(Kw), P(Vw), P(qo)]
            + [I(T), I(Hq), I(Hk), I(Kw.stride(0)), I(Kw.stride(1)), I(0)],
        )
        written.append((Kw, Vw, qo))
    (k0, v0, q0), (k1, v1, q1) = written
    assert torch.equal(k0, k1) and torch.equal(v0, v1) and torch.equal(q0, q1)


# ---------------------------------------------------------------------------------------------------------
# the rows layout: T sequences a token each over one cache - a prefix per row (shared by a fork's rows, end to
# end for a batch's), then every row's step i in the stretch base + i * W, row at its column. Each row must
# come out as its own one-row step over its keys laid end to end, whatever rows step beside it: the norm/rope write
# below, the one attention's rows form further down (`test_one_attention_takes_each_rows_own_keys`).
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


# a fork: one prefix across two splits, the rows' columns shuffled (rows left and a beam reordered), a padding
# row last; a batch: three prompts end to end, their steps past 2K keys, one row padding in the middle
FORK = (1500, 37, 4, [(2, 0, 1500), (0, 0, 1500), (3, 0, 1500), (1, 0, 1500), None])
BATCH = (1900, 700, 3, [(0, 0, 700), None, (1, 700, 1100), (2, 1800, 90)])


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
    common = [I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), I(0)]
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
                I(K1.stride(0)),
                I(K1.stride(1)),
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
            I(K.stride(0)),
            I(K.stride(1)),
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
# carries. A row group is 16 weight rows to a block of 2 to 8 warps (k split between them), so the grid is
# ceil(R / 16).
# ---------------------------------------------------------------------------------------------------------

MMA_SHAPES = [(151936, 1024), (19456, 2560), (2560, 9728), (4096, 1024), (6144, 1024), (1024, 3072), (6144, 2560)]


def _mma(cu: _Cuda, W: torch.Tensor, x: torch.Tensor, warps: int = 2) -> torch.Tensor:
    M, C = x.shape
    R = W.shape[0]
    xp = x if M == 32 else torch.cat([x, torch.zeros(32 - M, C, device=dev, dtype=x.dtype)])
    y = torch.empty(32, R, device=dev, dtype=bf)
    cu.launch("btb_gemv_mma_bf16", ((R + 15) // 16, 1, 1), (32 * warps, 1, 1), [P(W), P(xp), P(y), I(R), I(C), I(M)])
    return y[:M]


@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("R,C", MMA_SHAPES + [(1020, 1032), (17, 8), (16, 32)])
def test_gemv_mma_matches_linear_within_one_ulp(cu: _Cuda, R: int, C: int, warps: int) -> None:
    torch.manual_seed(0)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(32, C, device=dev, dtype=bf)
    ref = F.linear(x, W).float()
    # within one bf16 ulp of cuBLAS at the outputs' magnitude - the k inside a super-tile is permuted, so the
    # fp32 sums differ from cuBLAS's in the last bits and can round to the neighbouring bf16
    assert (_mma(cu, W, x, warps).float() - ref).abs().max().item() <= ref.abs().max().item() * 2**-7


@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("R,C", [(151936, 1024), (2560, 9728), (1024, 3072), (1020, 1032)])
def test_gemv_mma_is_repeat_identical_and_carries_no_row_across(cu: _Cuda, R: int, C: int, warps: int) -> None:
    torch.manual_seed(1)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(32, C, device=dev, dtype=bf)
    y = _mma(cu, W, x, warps)
    assert torch.equal(y, _mma(cu, W, x, warps))
    # row 0 is the one-row step's bits whatever stands in the other 31: mma accumulates each output element
    # from its own row of x, so a 1-row step and a 32-row verify pass agree bit for bit
    zeros = x.clone()
    zeros[1:] = 0
    assert torch.equal(y[:1], _mma(cu, W, zeros, warps)[:1])
    other = x.clone()
    other[1:] = torch.randn(31, C, device=dev, dtype=bf)
    assert torch.equal(y[:1], _mma(cu, W, other, warps)[:1])
    # and a 4-row pass is the first four rows of the 32-row one
    assert torch.equal(y[:4], _mma(cu, W, x[:4], warps))


@pytest.mark.parametrize("M", [1, 5, 32])
@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("R,C", [(1024, 2048), (1024, 3072), (1020, 1032), (17, 8), (4096, 1024)])
def test_the_eight_row_matvec_is_the_sixteen_row_one_bit_for_bit(cu: _Cuda, R: int, C: int, warps: int, M: int) -> None:
    """`btb_gemv_mma8_bf16`, a block 8 weight rows, against btb_gemv_mma_bf16's 16: each output the same mma, k
    slices and fold, so the same bits at any warps and any live rows"""
    torch.manual_seed(12)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(M, C, device=dev, dtype=bf)
    xp = x if M == 32 else torch.cat([x, torch.zeros(32 - M, C, device=dev, dtype=bf)])
    y = torch.empty(32, R, device=dev, dtype=bf)
    cu.launch("btb_gemv_mma8_bf16", ((R + 7) // 8, 1, 1), (32 * warps, 1, 1), [P(W), P(xp), P(y), I(R), I(C), I(M)])
    assert torch.equal(y[:M], _mma(cu, W, x, warps))


@pytest.mark.parametrize("act", ["silu", "gelu"])
@pytest.mark.parametrize("M", [1, 5, 32])
@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("Ii,C", [(3072, 1024), (1000, 1032), (24, 64)])
def test_the_glu_matvec_is_the_matvec_and_the_activation_bit_for_bit(
    cu: _Cuda, Ii: int, C: int, warps: int, M: int, act: str
) -> None:
    """`btb_gemv_mma_glu_{act}`, gate and up in one kernel and the activation in its epilogue, against
    btb_gemv_mma_bf16 over the merged [2I, C] weight and btb_{act}_mul over its output - at any I (a row group's
    gate and up rows need not line up with the plain kernel's groups), any warps, any live rows"""
    torch.manual_seed(9)
    W = torch.randn(2 * Ii, C, device=dev, dtype=bf)
    x = torch.randn(M, C, device=dev, dtype=bf)
    gu = _mma(cu, W, x, warps)
    want = torch.empty(M, Ii, device=dev, dtype=bf)
    cu.launch(f"btb_{act}_mul", (min(4096, (M * Ii + 255) // 256), 1, 1), (256, 1, 1), [P(gu), P(want), I(M), I(Ii)])
    xp = x if M == 32 else torch.cat([x, torch.zeros(32 - M, C, device=dev, dtype=bf)])
    m = torch.empty(32, Ii, device=dev, dtype=bf)
    cu.launch(
        f"btb_gemv_mma_glu_{act}", ((Ii + 15) // 16, 1, 1), (32 * warps, 1, 1), [P(W), P(xp), P(m), I(Ii), I(C), I(M)]
    )
    assert torch.equal(m[:M], want)


# ---------------------------------------------------------------------------------------------------------
# a prompt's matmuls (btb_gemm.cuh): each row of a chunk computed as the step's matvec computes it, so a prompt
# prefilled on the card makes the rows its steps would have made
# ---------------------------------------------------------------------------------------------------------


def _gemm(cu: _Cuda, W: torch.Tensor, x: torch.Tensor, warps: int = 0) -> torch.Tensor:
    """`btb_gemm_mma_bf16` at the matvec's `warps`, or `btb_gemm_f32_bf16` (warps 0), over every row of x"""
    T, C = x.shape
    R = W.shape[0]
    y = torch.full((T, R), float("nan"), device=dev, dtype=bf)
    grid = _CudaMixin._card_gemm_grid(bool(warps), R, T)  # the engine's launch: its tiles in groups
    if warps:
        cu.launch("btb_gemm_mma_bf16", grid, (128, 1, 1), [P(W), P(x), P(y), I(R), I(C), I(T), I(warps)])
    else:
        cu.launch("btb_gemm_f32_bf16", grid, (128, 1, 1), [P(W), P(x), P(y), I(R), I(C), I(T)])
    return y


@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("R,C", [(4096, 1024), (1024, 3072), (6144, 1024), (2560, 9728), (1020, 1032), (17, 8)])
def test_a_prompts_matmul_gives_the_steps_rows_on_tensor_cores(cu: _Cuda, R: int, C: int, warps: int) -> None:
    """`btb_gemm_mma_bf16` at the matvec's warps: every row of a chunk bit for bit the row btb_gemv_mma_bf16 gives
    a step, whatever the chunk's length and wherever in a block the row falls"""
    torch.manual_seed(13)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(300, C, device=dev, dtype=bf)
    y = _gemm(cu, W, x, warps)
    for a in range(0, 300, 32):
        assert torch.equal(y[a : a + 32], _mma(cu, W, x[a : a + 32], warps)), a
    assert torch.equal(_gemm(cu, W, x[:1], warps), y[:1])
    assert torch.equal(_gemm(cu, W, x[100:229], warps), y[100:229])


@pytest.mark.parametrize("R,C", [(4096, 1024), (1024, 3072), (2560, 9728), (1020, 1032), (17, 8)])
def test_a_prompts_matmul_gives_the_steps_rows_in_the_fp32_chain(cu: _Cuda, R: int, C: int) -> None:
    """`btb_gemm_f32_bf16`: every row of a chunk bit for bit the row btb_gemv_bf16_m{M} gives a step - its lanes'
    chains and its butterfly - whatever the chunk's length"""
    torch.manual_seed(14)
    W = torch.randn(R, C, device=dev, dtype=bf)
    x = torch.randn(77, C, device=dev, dtype=bf)
    y = _gemm(cu, W, x)
    for a in range(0, 77, 32):
        assert torch.equal(y[a : a + 32], _gemv(cu, W, x[a : a + 32])), a
    assert torch.equal(_gemm(cu, W, x[5:6]), y[5:6])


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


# ---------------------------------------------------------------------------------------------------------
# the one attention (btb_attn_flash.cuh): every pass's rows - a step, a verify chain or tree, a fork's or a batch's
# rows, a prompt's chunk - each the same operations over its own keys. Its bits are the row's alone: a prompt's chunk
# gives the rows the steps at those positions give, so a cached conversation's next turn decodes as the prompt cold.
# ---------------------------------------------------------------------------------------------------------


# the decode form's shape as the engine launches it (FaPick in btb_attn_flash.cuh): a group's keys, a block's threads
_fa_group = _CudaMixin._attn_group


def _fa_threads(D: int) -> int:
    return 32 * _CudaMixin._attn_warps(D)


def _flash(
    cu: _Cuda,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n0: int,
    par: Sequence[int],
    scale: float,
    win: int = 0,
    tbl: torch.Tensor | ctypes.c_void_p | None = None,
    cap: int | None = None,
) -> torch.Tensor:
    """the decode form `btb_attn_flash_d{D}`: T tokens after n0 rows, token t parented to par[t], every row's group
    states stored and folded by the last of its groups to arrive; `cap` the positions the launch's groups cover (the
    graph's), K's own by default; `tbl` a row map, or a pointer to one (a node's map set back to its window's start)"""
    T, Hq, D = q.shape
    Hk = K.shape[0]
    G = Hq // Hk
    S = ((cap or K.shape[1]) + _fa_group(D) - 1) // _fa_group(D)
    out = torch.full((T, Hq, D), float("nan"), device=dev, dtype=bf)
    n0t = torch.tensor([n0], dtype=torch.int32, device=dev)
    pt = torch.tensor(par, dtype=torch.int32, device=dev)
    pm = torch.zeros(S * T * Hq, device=dev)
    pl = torch.zeros(S * T * Hq, device=dev)
    pa = torch.zeros(S * T * Hq * D, device=dev)
    cnt = torch.zeros(T * Hq, dtype=torch.int32, device=dev)
    cu.launch(
        f"btb_attn_flash_d{D}",
        ((T * G + 7) // 8, S, Hk),  # a block 8 rows of a group, its warps its tiles (four at the widest head)
        (_fa_threads(D), 1, 1),
        [P(q), P(K), P(V), P(out), P(n0t), P(pt), I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), Fl(scale)]
        + [P(pm), P(pl), P(pa), P(cnt), I(win), tbl if isinstance(tbl, ctypes.c_void_p) else P(tbl)],
    )
    assert int(cnt.abs().sum()) == 0, "every row's count is reset by the block that folds it"
    return out


def _flash_rows(
    cu: _Cuda, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, rw: torch.Tensor, scale: float, win: int = 0
) -> torch.Tensor:
    """the decode form over a fork's or a batch's rows, `btb_attn_flash_rows_d{D}`: a block 8 heads of a token a group"""
    T, Hq, D = q.shape
    Hk, cap = K.shape[0], K.shape[1]
    S = (cap + _fa_group(D) - 1) // _fa_group(D)
    out = torch.full((T, Hq, D), 7.0, device=dev, dtype=bf)  # a padding row's stays as it was
    pm = torch.zeros(S * T * Hq, device=dev)
    pl = torch.zeros(S * T * Hq, device=dev)
    pa = torch.zeros(S * T * Hq * D, device=dev)
    cnt = torch.zeros(T * Hq, dtype=torch.int32, device=dev)
    cu.launch(
        f"btb_attn_flash_rows_d{D}",
        (T * ((Hq // Hk + 7) // 8), S, Hk),
        (_fa_threads(D), 1, 1),
        [P(q), P(K), P(V), P(out), P(rw), I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), Fl(scale)]
        + [P(pm), P(pl), P(pa), P(cnt), I(win)],
    )
    assert int(cnt.abs().sum()) == 0, "every row's count is reset by the block that folds it"
    return out


def _flash_prefill(
    cu: _Cuda,
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    n0: int,
    scale: float,
    win: int = 0,
    tbl: torch.Tensor | None = None,
    kq: bool = False,
) -> torch.Tensor:
    """the prefill form: a chunk's T tokens at positions n0 .., four warps a block, the tiles and groups folded as the
    walk goes, the row states in a float32 buffer of the chunk's rows - the rows as the mma's M
    (`btb_attn_flash_prefill_d{D}`), or `kq` the decode form's orientation (`btb_attn_flash_prefill_kq_d{D}`, a
    card's whose tensor cores fail `btb_mma_roles`), whichever this card picks: both are tested on every card"""
    T, Hq, D = q.shape
    Hk = K.shape[0]
    tpb = cu.flash_prefill_rows(D) // (Hq // Hk)
    out = torch.full((T, Hq, D), float("nan"), device=dev, dtype=bf)
    run = torch.empty(T * Hq * D, device=dev)
    cu.launch(
        cu.flash_prefill_kernel(D, kq),
        ((T + tpb - 1) // tpb, Hk, 1),
        (128, 1, 1),
        [P(q), P(K), P(V), P(out), I(n0), I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), Fl(scale), I(win)]
        + [P(tbl), P(run)],
        shared=cu.flash_prefill_smem(D, kq),
    )
    return out


def _prefill_forms(cu: _Cuda) -> list[bool]:
    """the prefill's orientations this card may run (`kq` for `_flash_prefill`): the decode form's always, the
    rows-as-M one where its tensor cores pass `btb_mma_roles` - elsewhere the card never takes it"""
    return [True, False] if cu.mma_roles else [True]


def test_the_prefill_takes_the_form_its_card_passes(cu: _Cuda) -> None:
    """the card's prefill kernel is the rows-as-M form where its mma passed `btb_mma_roles` at bind, else the decode
    form's orientation; launched again the check gives the same verdict"""
    for D in (64, 128, 256):
        want = "btb_attn_flash_prefill_kq_d" if not cu.mma_roles else "btb_attn_flash_prefill_d"
        assert cu.flash_prefill_kernel(D) == f"{want}{D}"
        assert cu.flash_prefill_smem(D) == cu.flash_prefill_smem(D, not cu.mma_roles)
    assert cu._mma_roles_hold() == cu.mma_roles


def _close_to_reference(out: torch.Tensor, ref: torch.Tensor) -> bool:
    """within the bf16 weights' rounding at the row's magnitude (a row of two keys averages to values near 3)"""
    return (out.float() - ref.float()).abs().max().item() <= 2**-6 * max(1.0, ref.float().abs().max().item())


@pytest.mark.parametrize("win", [0, 300])
@pytest.mark.parametrize("Hq,Hk,D", [(16, 8, 128), (32, 8, 64), (8, 2, 256), (12, 12, 128), (40, 8, 128)])
def test_one_attention_matches_the_reference_and_a_prompt_gives_the_steps_rows(
    cu: _Cuda, Hq: int, Hk: int, D: int, win: int
) -> None:
    """a prompt's rows through the prefill form are each within bf16 of the float32 reference, and bit for bit the
    row a one-row step at that position computes - across splits, under a window, whatever chunk the row came in
    and wherever in it, in either orientation the card may take - so the rows a conversation's steps made are the
    rows its prompt cold would make"""
    torch.manual_seed(31)
    cap = 2304
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(cap, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    steps = {
        p: _flash(cu, q[p : p + 1], K, V, p, [-1], scale, win)[0] for p in (0, 1, 63, 64, 511, 512, 513, 1100, 2099)
    }
    for kq in _prefill_forms(cu):
        whole = _flash_prefill(cu, q[:2100], K, V, 0, scale, win, kq=kq)
        for p, step in steps.items():
            first = max(0, p + 1 - win) if win else 0
            assert _close_to_reference(whole[p], _attn_ref(q[p], K, V, list(range(first, p + 1)), scale)), (kq, p)
            assert torch.equal(step, whole[p]), f"row {p} (kq {kq}): the prompt's row is not its step's"
        # the same prompt cut in chunks anywhere: the same rows
        for cuts in ([0, 64, 700, 2100], [0, 1, 37, 513, 514, 2100], [0, 2100]):
            parts = [_flash_prefill(cu, q[a:b], K, V, a, scale, win, kq=kq) for a, b in itertools.pairwise(cuts)]
            assert torch.equal(torch.cat(parts), whole), (kq, cuts)
        assert torch.equal(_flash_prefill(cu, q[:2100], K, V, 0, scale, win, kq=kq), whole)


@pytest.mark.parametrize("win", [0, 200])
@pytest.mark.parametrize("Hq,Hk,D", [(16, 8, 128), (32, 8, 64), (8, 2, 256), (40, 8, 128)])
def test_one_attention_verifies_a_tree_as_its_steps(cu: _Cuda, Hq: int, Hk: int, D: int, win: int) -> None:
    """a verify pass's chain and tree: each node bit for bit the one-row step at its committed position over its
    committed path's rows, whatever splits the prefix spans - the speculation's answer is the plain loop's"""
    torch.manual_seed(32)
    cap, T = 3072, 8
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    tree = [-1, 0, 1, 0, 3, 4, 1, 6]
    for n0 in (40, 500, 1020, 2500):
        out = _flash(cu, q, K, V, n0, tree, scale, win)
        assert torch.equal(out, _flash(cu, q, K, V, n0, tree, scale, win))
        for t in range(T):
            path = [t]  # node t's path, root first
            while tree[path[0]] >= 0:
                path.insert(0, tree[path[0]])
            d = len(path) - 1
            K1, V1 = K.clone(), V.clone()
            K1[:, n0 : n0 + d + 1], V1[:, n0 : n0 + d + 1] = K[:, [n0 + a for a in path]], V[:, [n0 + a for a in path]]
            one = _flash(cu, q[t : t + 1], K1, V1, n0 + d, [-1], scale, win)
            assert torch.equal(one[0], out[t]), (n0, t)
            first = max(0, n0 + d + 1 - win) if win else 0
            ref = _attn_ref(q[t], K1, V1, list(range(first, n0 + d + 1)), scale)
            assert _close_to_reference(out[t], ref), (n0, t)
        # a chain's rows are the prompt's: the prefill form over the same positions, either orientation
        chain = _flash(cu, q, K, V, n0, list(range(-1, T - 1)), scale, win)
        for kq in _prefill_forms(cu):
            assert torch.equal(chain, _flash_prefill(cu, q, K, V, n0, scale, win, kq=kq)), (n0, kq)


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", [FORK, BATCH], ids=["fork", "batch"])
@pytest.mark.parametrize("win", [0, 512])
@pytest.mark.parametrize("Hq,Hk", [(8, 4), (40, 4)], ids=["G2", "G10"])
def test_one_attention_takes_each_rows_own_keys(
    cu: _Cuda, D: int, layout: tuple[int, int, int, list[RowSpec]], win: int, Hq: int, Hk: int
) -> None:
    """a fork's or a batch's rows: each within bf16 of the float32 reference over its own keys, and its own
    sequence's one-row step, bit for bit, whatever rows step beside it - the same bits again; a token's heads past
    the 8 rows of a block (G 10) in blocks of their own"""
    torch.manual_seed(33)
    base, step, W, rows = layout
    T = len(rows)
    cap = (base + (step + 1) * W + 1023) // 1024 * 1024
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    rw = _rows_layout(base, step, W, rows)
    out = _flash_rows(cu, q, K, V, rw, scale, win)
    assert torch.equal(out, _flash_rows(cu, q, K, V, rw, scale, win))
    for t, r in enumerate(rows):
        if r is None:
            assert bool((out[t] == 7.0).all()), "a padding row computes nothing"
            continue
        slots = _row_slots(base, step, W, r)
        assert _close_to_reference(out[t], _attn_ref(q[t], K, V, slots[-win:] if win else slots, scale)), t
        n = len(slots)
        K1 = torch.zeros(Hk, (n + 1023) // 1024 * 1024, D, device=dev, dtype=bf)
        V1 = torch.zeros_like(K1)
        K1[:, :n], V1[:, :n] = K[:, slots], V[:, slots]
        one = _flash(cu, q[t : t + 1], K1, V1, n - 1, [-1], scale, win)
        assert torch.equal(one[0], out[t]), t


@pytest.mark.parametrize("win", [0, 128])
@pytest.mark.parametrize("Hq,Hk,D", [(16, 8, 128), (8, 2, 256), (32, 8, 64)])
def test_one_attention_reads_rows_where_they_lie(cu: _Cuda, Hq: int, Hk: int, D: int, win: int) -> None:
    """a row's bits are its keys' and nothing else: through a paged row map over a position-major pool (an identity
    map the plain read), through either layout, under a window, and whichever query heads share its key head's tiles
    (a head's row alone, its group's G - 1 others dropped, is the same row)"""
    torch.manual_seed(34)
    cap, n0, T = 2048, 1100, 5
    K = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    V = torch.randn(Hk, cap, D, device=dev, dtype=bf)
    q = torch.randn(T, Hq, D, device=dev, dtype=bf)
    scale = 1.0 / math.sqrt(D)
    tbl, Kp, Vp = _paged(K, V, D + win)
    ident = torch.arange(cap, dtype=torch.int32, device=dev)
    par = [-1, 0, 1, 0, 3]
    want = _flash(cu, q, K, V, n0, par, scale, win)
    assert torch.equal(_flash(cu, q, Kp, Vp, n0, par, scale, win, tbl=tbl, cap=cap), want)
    assert torch.equal(_flash(cu, q, K, V, n0, par, scale, win, tbl=ident), want)
    assert torch.equal(_flash(cu, q, _position_major(K), _position_major(V), n0, par, scale, win), want)
    G = Hq // Hk
    for kq in _prefill_forms(cu):
        pf = _flash_prefill(cu, q, K, V, n0, scale, win, kq=kq)
        assert torch.equal(_flash_prefill(cu, q, Kp, Vp, n0, scale, win, tbl=tbl, kq=kq), pf)
        assert torch.equal(_flash_prefill(cu, q, K, V, n0, scale, win, tbl=ident, kq=kq), pf)
        assert torch.equal(_flash_prefill(cu, q[:, ::G].contiguous(), K, V, n0, scale, win, kq=kq), pf[:, ::G])
    # one head of each group alone, its key head's rows its own
    alone = _flash(cu, q[:, ::G].contiguous(), K, V, n0, par, scale, win)
    assert torch.equal(alone, want[:, ::G])
    # a prompt over several of the prefill's blocks through the map, and a one-row chunk through it - the step's row
    qp = torch.randn(150, Hq, D, device=dev, dtype=bf)
    for kq in _prefill_forms(cu):
        pf = _flash_prefill(cu, qp, K, V, n0, scale, win, kq=kq)
        assert torch.equal(_flash_prefill(cu, qp, Kp, Vp, n0, scale, win, tbl=tbl, kq=kq), pf), kq
        one = _flash_prefill(cu, qp[:1], Kp, Vp, n0, scale, win, tbl=tbl, kq=kq)
        assert torch.equal(one, _flash(cu, qp[:1], K, V, n0, [-1], scale, win)), kq
    # a node's keys named one by one from its window's start (`_card_attention`'s KeyRows): the map's pointer set back
    # by the window's first position, never read before it - the node's row as its step's through the whole map
    pos = n0 + 3
    first = max(0, pos + 1 - win) if win else 0
    rows = tbl[first : pos + 1].clone()
    via = ctypes.c_void_p(int(rows.data_ptr()) - 4 * first)
    node = _flash(cu, q[:1], Kp, Vp, pos, [-1], scale, win, tbl=via, cap=cap)
    assert torch.equal(node, _flash(cu, q[:1], K, V, pos, [-1], scale, win)), "a node through its named keys"
