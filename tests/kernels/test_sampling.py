# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The sampler (btb/sampling.py): greedy is the argmax, the masks keep the right set, a key repeats its draw, the
draws follow the distribution, and the torch pre-cut agrees with the full sort. No model; MLX where present."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import pytest
import torch

from btb.sampling import PRECUT, Sampling, as_pick
from tests.helpers import card_kernels, mlx_core, native_library

# the CPU native sampler's shape: (logits [R, V] float32, a key a row, temperature, top_k, top_p) -> [R] tokens
_CpuPick = Callable[[torch.Tensor, Sequence[int], float, int, float], torch.Tensor]


def _cpu() -> _CpuPick:
    """the CPU native sampler (Native.sample_pick, the sample.rs port)"""
    from btb.engine.native import Native

    native_library()
    assert Native.sample_pick is not None, "the native library has no sampler: rebuild it"
    return Native.sample_pick


def test_temperature_zero_is_the_argmax_everywhere() -> None:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 300, generator=g)
    s = Sampling()
    assert s.greedy and not s.masked and s.seeded() is s
    assert s.pick_torch(x, [1, 2, 3, 4, 5]).tolist() == x.argmax(-1).tolist()
    p = as_pick(True)
    assert p is not None and p.greedy and as_pick(None) is None and as_pick(False) is None
    mx = mlx_core()
    if mx is not None:
        ids = s.pick_mx(mx.array(x.numpy()), [1, 2, 3, 4, 5])
        assert ids.tolist() == x.argmax(-1).tolist()


def _mask_of(s: Sampling, x: torch.Tensor, draws: int = 400) -> set[int]:
    """the set of tokens a sampling ever returns over `draws` keys: the mask it applies"""
    return {int(s.pick_torch(x[None], [k])[0]) for k in range(draws)}


def test_top_k_and_top_p_keep_the_right_set() -> None:
    # a row whose probabilities are 0.5, 0.25, 0.125, ... : top-p 0.8 keeps the first three (0.875 crosses it)
    x = torch.tensor([math.log(0.5 / 2**i) for i in range(8)])
    assert _mask_of(Sampling(temperature=1.0, top_k=2), x) == {0, 1}
    assert _mask_of(Sampling(temperature=1.0, top_p=0.8), x) == {0, 1, 2}
    assert _mask_of(Sampling(temperature=1.0, top_p=0.5), x) == {0}  # the first token alone reaches the mass
    # top-k first, top-p over what it kept: over the top four (mass 0.9375) the third token's cumulative mass
    # before it is 0.8 exactly, so it falls out
    assert _mask_of(Sampling(temperature=1.0, top_k=4, top_p=0.8), x) == {0, 1}
    assert _mask_of(Sampling(temperature=1.0, top_k=4, top_p=0.81), x) == {0, 1, 2}
    assert _mask_of(Sampling(temperature=1.0, top_k=2, top_p=0.99), x) == {0, 1}
    mx = mlx_core()
    if mx is not None:
        for s, want in (
            (Sampling(temperature=1.0, top_k=2), {0, 1}),
            (Sampling(temperature=1.0, top_p=0.8), {0, 1, 2}),
            (Sampling(temperature=1.0, top_k=4, top_p=0.81), {0, 1, 2}),
        ):
            got = {int(s.pick_mx(mx.array(x.numpy())[None], [k])[0]) for k in range(400)}
            assert got == want, (s, got)


def test_a_key_repeats_its_draw_and_seeds_differ() -> None:
    x = torch.randn(1, 5000, generator=torch.Generator().manual_seed(1))
    a, b = Sampling(temperature=1.0, seed=7), Sampling(temperature=1.0, seed=8)
    assert a.key_for(12) == a.key_for(12) and a.key_for(12) != a.key_for(13) and a.key_for(12) != b.key_for(12)
    assert a.pick_torch(x, [a.key_for(12)]).tolist() == a.pick_torch(x, [a.key_for(12)]).tolist()
    picks = {int(a.pick_torch(x, [a.key_for(p)])[0]) for p in range(64)}
    assert len(picks) > 1, "a flat-ish row sampled at many positions does not sit on one token"
    s = Sampling(temperature=0.7).seeded()
    assert s.seed is not None and Sampling(temperature=0.7, seed=5).seeded().seed == 5


def test_the_draws_follow_the_distribution() -> None:
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    x = torch.log(p)[None]
    n = 20000
    s = Sampling(temperature=1.0, seed=11)
    c = Counter(int(s.pick_torch(x, [s.key_for(i)])[0]) for i in range(n))
    for t, want in enumerate(p.tolist()):
        assert abs(c[t] / n - want) < 0.02, (t, c[t] / n, want)
    # temperature 0.5 sharpens: p^2 renormalized
    q = p**2 / (p**2).sum()
    s2 = Sampling(temperature=0.5, seed=11)
    c2 = Counter(int(s2.pick_torch(x, [s2.key_for(i)])[0]) for i in range(n))
    for t, want in enumerate(q.tolist()):
        assert abs(c2[t] / n - want) < 0.02, (t, c2[t] / n, want)
    mx = mlx_core()
    if mx is not None:
        xm = mx.array(x.numpy())
        ids = s.pick_mx(mx.repeat(xm, n, axis=0), [s.key_for(i) for i in range(n)]).tolist()
        cm = Counter(ids)
        for t, want in enumerate(p.tolist()):
            assert abs(cm[t] / n - want) < 0.02, (t, cm[t] / n, want)


def test_the_torch_path_widens_its_cut_and_keeps_the_nucleus() -> None:
    """the torch draw (the fallback where the native kernel is not bound, and every CUDA row): its top-p works
    over the top PRECUT tokens and widens the cut by fours when they do not hold the nucleus; the picks stay in
    the nucleus either way and a key repeats its draw"""
    V = 4 * PRECUT
    s = Sampling(temperature=1.0, top_p=0.9, seed=3)
    for x in (torch.zeros(V) + 0.01 * torch.arange(V).flip(0), torch.zeros(V) + 0.5 * torch.arange(V).flip(0)):
        p = torch.softmax(x, -1)
        cum = torch.cumsum(p, -1)
        kept = int((cum - p < 0.9).sum())  # the row is sorted: the nucleus is the first `kept` tokens
        got = [int(s._row_torch(x, s.key_for(i))) for i in range(50)]
        assert max(got) < kept, (max(got), kept)
        assert got == [int(s._row_torch(x, s.key_for(i))) for i in range(50)]
    assert kept < PRECUT < 4 * PRECUT  # the second row's nucleus fits the cut, the first row's needed widening


def test_the_mlx_kernel_picks_as_the_ops_do() -> None:
    """the fused Metal pick: greedy is the argmax; a draw sits inside the top-k / top-p mask the plain-ops
    formulation keeps (ties aside, the same set); a key repeats its draw; a row picks the same alone and in a
    batch; the draws follow the distribution"""
    mx = mlx_core()
    if mx is None:
        pytest.skip("no Metal sampler")
    from btb.mlx import sample

    g = torch.Generator().manual_seed(4)
    x = torch.randn(16, 3000, generator=g) * 3
    xm = mx.array(x.numpy())
    assert sample.pick(xm, list(range(16)), 0.0, 0, 1.0).tolist() == x.argmax(-1).tolist()
    for s in (
        Sampling(temperature=0.8, seed=1),
        Sampling(temperature=0.8, top_k=40, seed=1),
        Sampling(temperature=0.8, top_p=0.9, seed=1),
        Sampling(temperature=0.8, top_k=40, top_p=0.9, seed=1),
    ):
        picks = [s.pick_mx(xm, [s.key_for(i * 16 + r) for r in range(16)]) for i in range(100)]
        mx.eval(*picks)
        for r in range(16):
            row = x[r] / s.temperature
            keep = set(torch.topk(row, s.top_k).indices.tolist()) if s.top_k > 0 else set(range(3000))
            if s.top_p < 1.0:
                vals, idx = torch.sort(row, descending=True)
                if s.top_k > 0:
                    m = torch.zeros(3000, dtype=torch.bool)
                    m[list(keep)] = True
                    vals = vals.masked_fill(~m[idx], float("-inf"))
                p = torch.softmax(vals, -1)
                keep &= set(idx[(torch.cumsum(p, -1) - p) < s.top_p].tolist())
            got = {int(q[r]) for q in picks}
            assert got <= keep, (s, r, got - keep)
        one = s.pick_mx(xm[3:4], [s.key_for(3)])
        both = s.pick_mx(xm, [s.key_for(r) for r in range(16)])
        again = s.pick_mx(xm, [s.key_for(r) for r in range(16)])
        mx.eval(one, both, again)
        assert int(one[0]) == int(both[3]) and both.tolist() == again.tolist()
    # a 4-way row, 20k keys: the frequencies of the distribution
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    s = Sampling(temperature=1.0, seed=2)
    ids = s.pick_mx(
        mx.repeat(mx.array(torch.log(p).numpy())[None], 20000, axis=0), [s.key_for(i) for i in range(20000)]
    )
    c = Counter(ids.tolist())
    for t, want in enumerate(p.tolist()):
        assert abs(c[t] / 20000 - want) < 0.02, (t, c[t] / 20000, want)


def test_the_cuda_kernel_picks_as_the_ops_do() -> None:
    """the fused card pick (the multi-block pipeline): greedy is the argmax; a draw sits inside the top-k / top-p
    mask the plain-ops formulation keeps; a key repeats its draw; a row picks the same alone and in a batch; the
    draws follow the distribution; and where the CPU port is here, the pick is its token on every key (the exact
    32-bit threshold, three radix levels)"""
    cu = card_kernels()
    if cu is None:
        pytest.skip("no CUDA sampler")
    cpu = _cpu()  # the CPU port, for the token-for-token A/B

    g = torch.Generator().manual_seed(4)
    x = torch.randn(16, 3000, generator=g) * 3
    xc = x.cuda()
    assert Sampling().pick_torch(xc, list(range(16))).cpu().tolist() == x.argmax(-1).tolist()  # greedy = argmax
    for s in (
        Sampling(temperature=0.8, seed=1),
        Sampling(temperature=0.8, top_k=40, seed=1),
        Sampling(temperature=0.8, top_p=0.9, seed=1),
        Sampling(temperature=0.8, top_k=40, top_p=0.9, seed=1),
    ):
        keysets = [[s.key_for(i * 16 + r) for r in range(16)] for i in range(100)]
        picks = [cu.pick(xc, ks, s.temperature, s.top_k, s.top_p).cpu() for ks in keysets]
        # the card pick equals the CPU port token-for-token (the exact 3-level threshold)
        for ks, pk in zip(keysets, picks):
            assert pk.tolist() == cpu(x, ks, s.temperature, s.top_k, s.top_p).tolist(), (s, ks[0])
        for r in range(16):
            row = x[r] / s.temperature
            keep = set(torch.topk(row, s.top_k).indices.tolist()) if s.top_k > 0 else set(range(3000))
            if s.top_p < 1.0:
                vals, idx = torch.sort(row, descending=True)
                if s.top_k > 0:
                    m = torch.zeros(3000, dtype=torch.bool)
                    m[list(keep)] = True
                    vals = vals.masked_fill(~m[idx], float("-inf"))
                p = torch.softmax(vals, -1)
                keep &= set(idx[(torch.cumsum(p, -1) - p) < s.top_p].tolist())
            got = {int(q[r]) for q in picks}
            assert got <= keep, (s, r, got - keep)
        one = cu.pick(xc[3:4], [s.key_for(3)], s.temperature, s.top_k, s.top_p).cpu()
        both = cu.pick(xc, [s.key_for(r) for r in range(16)], s.temperature, s.top_k, s.top_p).cpu()
        again = cu.pick(xc, [s.key_for(r) for r in range(16)], s.temperature, s.top_k, s.top_p).cpu()
        assert int(one[0]) == int(both[3]) and both.tolist() == again.tolist()  # batch-invariant, deterministic
    # a 4-way row, 20k keys: the frequencies of the distribution
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    s = Sampling(temperature=1.0, seed=2)
    ids = cu.pick(torch.log(p)[None].repeat(20000, 1).cuda(), [s.key_for(i) for i in range(20000)], 1.0, 0, 1.0)
    c = Counter(ids.cpu().tolist())
    for t, want in enumerate(p.tolist()):
        assert abs(c[t] / 20000 - want) < 0.02, (t, c[t] / 20000, want)


def test_the_cuda_pick_equals_the_cpu_pick_to_the_token() -> None:
    """the card pick is the CPU port's pick token-for-token, over temperature / top-k / top-p / both, a small vocab
    up to the Qwen production vocab (151936), and many keys: the exact 32-bit radix threshold (three levels,
    11+11+10) and the shared fast_exp make the card's kept set and its Gumbel-max draw identical to sample.rs -
    accuracy is not traded for the multi-block speed, and the top-p mass sum cast to f32 matches even where the
    ordered-uint sum runs to ~2^47"""
    cu = card_kernels()
    cpu = _cpu()
    if cu is None:
        pytest.skip("no CUDA sampler")
    configs = ((0.8, 0, 1.0), (0.7, 40, 1.0), (0.9, 0, 0.9), (0.8, 40, 0.9), (1.0, 0, 0.95), (0.6, 100, 0.8))
    for V, R, draws in ((3000, 8, 128), (32768, 4, 64), (151936, 2, 48)):
        g = torch.Generator().manual_seed(V)
        x = torch.randn(R, V, generator=g) * 3
        xc = x.cuda()
        s = Sampling(seed=17)
        for temp, k, tp in configs:
            for i in range(draws):
                keys = [s.key_for(i * R + r) for r in range(R)]
                got = cu.pick(xc, keys, temp, k, tp).cpu().tolist()
                ref = cpu(x, keys, temp, k, tp).tolist()
                assert got == ref, (V, R, temp, k, tp, i, got, ref)


def test_the_cuda_pick_writes_its_own_rows_and_nothing_past_them() -> None:
    """the pick's finalize (btb_sample_out) runs R rows as whole 256-thread blocks, as `pick` launches it; the threads
    past R leave. Unguarded, a one-row pick's 255 spare threads wrote ~1 KB past `out` into whatever the allocator
    had put there - in the card graph tests a step graph's state, which then stalled. gbest and out are the fronts of
    sentinel-filled buffers here, so a stray write shows whatever the allocator's layout"""
    import ctypes

    cu = card_kernels()
    if cu is None:
        pytest.skip("no card or no btb_kernels.fatbin")
    pad, sentinel = 512, -12345
    for R in (1, 3, 256, 257):
        toks = [7 * r + 1 for r in range(R)]
        gbest = torch.zeros(R + pad, dtype=torch.int64, device="cuda")  # past R: packs of token ~0, never sentinel
        gbest[:R] = torch.tensor([0xFFFFFFFF - t for t in toks], dtype=torch.int64)  # the packed low word is ~token
        out = torch.full((R + pad,), sentinel, dtype=torch.int32, device="cuda")
        cu.launch(
            "btb_sample_out", ((R + 255) // 256, 1, 1), (256, 1, 1), [cu.ptr(gbest), cu.ptr(out), ctypes.c_int(R)]
        )
        torch.cuda.synchronize()
        assert out[:R].tolist() == toks, R
        assert bool((out[R:] == sentinel).all()), f"R={R}: the finalize wrote past its rows"


def test_the_verify_kernel_matches_its_reference_and_never_runs_dry() -> None:
    """the drawn-tree verify: rows whose drafter nucleus is one to three tokens with eight draws asked (the tail
    of the draw list has no mass), on a 12-token target: the kernel's outcome is the reference's on every key,
    always a token of the row, and the emitted token follows the target over the keys (exact: the residual rule
    ends at the first massless draw instead of subtracting the proposal again)"""
    from btb.sampling import Verify

    mx = mlx_core()
    V, R, n = 12, 4, 3000
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(R, V, generator=g) * 3
    dl = logits + torch.randn(R, V, generator=g) * 0.7  # the drafter: the target, noised
    s = Sampling(temperature=0.8, top_p=0.9, seed=5)
    p = s.dist_torch(logits)
    seen: list[Counter[int]] = [Counter() for _ in range(R)]
    for i in range(n):
        keys = [s.key_for(i * R + r) for r in range(R)]
        ids, q = s.draw_torch(dl, [s.key_for(i * R + r, salt=1) for r in range(R)], 8)
        kids = ids.tolist()
        assert any((q[r][c] == 0) for r in range(R) for c in kids[r]), "the case needs massless draws"
        v = Verify(s, q, kids, [True] * R)
        ref = v.pick_torch(logits, keys).tolist()
        if mx is not None:
            vm = Verify(s, mx.array(q.numpy()), kids, [True] * R)
            got = vm.pick_mx(mx.array(logits.numpy()), keys).tolist()
            assert got == ref, (i, got, ref)
        for r in range(R):
            slot, t = Verify.unpack(ref[r])
            assert 0 <= t < V, (i, r, ref[r])
            assert slot < 0 or (kids[r][slot] == t and q[r][t] > 0), (i, r, slot, t)
            seen[r][t] += 1
    for r in range(R):
        for t in range(V):
            assert abs(seen[r][t] / n - float(p[r][t])) < 0.03, (r, t, seen[r][t] / n, float(p[r][t]))


def test_the_cuda_verify_follows_the_target() -> None:
    """the card verify (btb_sample_verify): its outcome is the torch reference's token-for-token (the exact 3-level
    threshold and the shared fast_exp), every outcome is a token of the row (a child of the node or a residual
    draw), and the emitted token follows the target p over the keys - the property that makes the drawn-tree
    speculation exact"""
    cu = card_kernels()
    if cu is None:
        pytest.skip("no CUDA sampler")
    from btb.sampling import Verify

    V, R, n = 12, 4, 2500
    g = torch.Generator().manual_seed(9)
    logits = torch.randn(R, V, generator=g) * 3
    dl = logits + torch.randn(R, V, generator=g) * 0.7
    s = Sampling(temperature=0.8, top_p=0.9, seed=5)
    p = s.dist_torch(logits)
    seen: list[Counter[int]] = [Counter() for _ in range(R)]
    for i in range(n):
        keys = [s.key_for(i * R + r) for r in range(R)]
        ids, q = s.draw_torch(dl, [s.key_for(i * R + r, salt=1) for r in range(R)], 8)
        kids = ids.tolist()
        got = Verify(s, q, kids, [True] * R).pick_torch(logits.cuda(), keys).tolist()  # routes to the card kernel
        ref = Verify(s, q, kids, [True] * R).pick_torch(logits, keys).tolist()  # the torch reference on the CPU
        assert got == ref, (i, got, ref)  # bit-for-bit with the reference
        for r in range(R):
            slot, t = Verify.unpack(got[r])
            assert 0 <= t < V, (i, r, got[r])
            assert slot < 0 or (kids[r][slot] == t and q[r][t] > 0), (i, r, slot, t)
            seen[r][t] += 1
    for r in range(R):
        for t in range(V):
            assert abs(seen[r][t] / n - float(p[r][t])) < 0.04, (r, t, seen[r][t] / n, float(p[r][t]))


def test_the_uniform_stays_inside_the_unit_interval_and_top_p_zero_keeps_the_top_token() -> None:
    """the hashed uniform takes 23 bits of the hash because the top of a 24-bit range rounds to 1.0 in float32
    (an infinite Gumbel, a token winning from anywhere): a token whose hash tops the 24-bit range, 25 nats
    under the top, is not drawn by the CPU, the Metal or the card kernel; and a top-p that rounds to no mass
    keeps the top token, the argmax, on all three"""
    from btb.sampling import _hash64, _shr

    cpu = _cpu()
    mx = mlx_core()
    cu = card_kernels()
    V = 151936
    s = Sampling(temperature=1.0, seed=7)
    found = None
    for pos in range(4000):
        key = s.key_for(pos)
        bad = (_shr(_hash64(key, torch.arange(V)), 40) == 0xFFFFFF).nonzero()
        if len(bad):
            found = (key, int(bad[0]))
            break
    assert found is not None
    key, tok = found
    x = torch.full((1, V), -60.0)
    x[0, 5] = 40.0
    x[0, tok] = 15.0
    assert int(cpu(x, [key], 1.0, 0, 1.0)[0]) == 5
    if mx is not None:
        assert int(s.pick_mx(mx.array(x.numpy()), [key])[0]) == 5
    if cu is not None:
        assert int(cu.pick(x.cuda(), [key], 1.0, 0, 1.0)[0]) == 5
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 3000, generator=g) * 4
    for tp in (0.0, 1e-12):
        z = Sampling(temperature=0.8, top_p=tp, seed=3)
        keys = [z.key_for(r) for r in range(4)]
        assert cpu(x, keys, 0.8, 0, tp).tolist() == x.argmax(-1).tolist()
        if mx is not None:
            assert z.pick_mx(mx.array(x.numpy()), keys).tolist() == x.argmax(-1).tolist()
        if cu is not None:
            assert cu.pick(x.cuda(), keys, 0.8, 0, tp).cpu().tolist() == x.argmax(-1).tolist()


def test_the_cpu_and_metal_picks_are_one_pick() -> None:
    """the CPU kernel (native/src/sample.rs) and the Metal kernel token for token at the production vocabulary,
    over temperature / top-k / top-p / both: the same thresholds, the same hash, so a row picks the same token
    on either (the card kernel is held to the CPU one the same way)"""
    cpu = _cpu()
    mx = mlx_core()
    if mx is None:
        pytest.skip("no Metal sampler")
    g = torch.Generator().manual_seed(11)
    x = torch.randn(8, 151936, generator=g) * 4
    xm = mx.array(x.numpy())
    for s in (
        Sampling(temperature=0.8, seed=3),
        Sampling(temperature=0.8, top_k=40, seed=3),
        Sampling(temperature=0.8, top_p=0.9, seed=3),
        Sampling(temperature=1.3, top_k=40, top_p=0.5, seed=3),
    ):
        for i in range(20):
            keys = [s.key_for(i * 8 + r) for r in range(8)]
            a = cpu(x, keys, s.temperature, s.top_k, s.top_p).tolist()
            b = s.pick_mx(xm, keys).tolist()
            assert a == b, (s, i, a, b)
