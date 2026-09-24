# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel arena's in-place writes (btb/mlx/kv.py) against a numpy copy of the arena: `kv_store` and
`kv_gather` put exactly the reference's bf16 bits where the cache and the drafter aim them, through the layer's
views of the one shared array, and leave every other byte of the arena as it was. Bit-exact: the kernels move
16-bit rows and do no arithmetic. Skips when MLX is not available."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from btb import mlx_available

if TYPE_CHECKING:
    import mlx.core as mx_

    from btb.engine.cache import GrowLayer
    from btb.mlx import Shared

Bits = NDArray[np.uint16]  # bf16 values as their raw 16-bit patterns

# (Hk, head_dim) of the arena models: the megakernel takes head_dim in multiples of 64 (Qwen3's 8 x 128, the
# 256-wide heads of Qwen3.5 and Gemma 3)
SHAPES = ((8, 128), (2, 256), (4, 64))
LAYERS = 3


@pytest.fixture(autouse=True)
def _mlx() -> None:
    if not mlx_available():
        pytest.skip("no Metal device")


def _bits(t: torch.Tensor) -> Bits:
    """a torch tensor's bf16 bits as uint16 (a float tensor rounds to bf16 first, as the launcher's cast does)"""
    return t.bfloat16().contiguous().view(torch.int16).numpy().view(np.uint16)


def _arena(Hk: int, d: int, cap: int) -> tuple[Shared, list[GrowLayer], Bits]:
    """an arena of LAYERS layers laid out as the engine's cache builds it (K then V a layer, `cap` rows a head),
    each layer's views cut by GrowLayer itself; with the arena's bytes as [2 * LAYERS, Hk, cap, d] bf16 bits"""
    from btb import mlx as mlxdev
    from btb.engine.cache import GrowLayer

    per = Hk * cap * d * 2
    sh = mlxdev.Shared(2 * LAYERS * per)
    layers = []
    for i in range(LAYERS):
        layer = GrowLayer(shared=True, arena=(sh.mx, 2 * i * per, (2 * i + 1) * per, cap))
        empty = torch.empty(0, dtype=torch.bfloat16)
        layer.lazy_initialization(empty, empty)
        layer._ensure(1, Hk, 1, d, torch.bfloat16)
        assert layer.arena is not None and getattr(layer, "_in_arena", False)
        layers.append(layer)
    return sh, layers, sh.np.view(np.uint16).reshape(2 * LAYERS, Hk, cap, d)


def _store(buf: mx_.array, rows: mx_.array, n0: int) -> None:
    from btb import mlx as mlxdev

    flag = mlxdev.kv_store(buf, rows, n0)
    mlxdev.mx().eval(flag)
    assert flag.tolist() == [1]


def _view_bits(buf: mx_.array) -> Bits:
    """the layer's MLX view read back as the attention reads it, [Hk, cap, d] bits"""
    from btb import mlx as mlxdev

    m = mlxdev.mx()
    return np.array(buf[0].view(m.uint16))


@pytest.mark.parametrize("Hk,d", SHAPES)
@pytest.mark.parametrize("cap", [1500, 8192])
def test_store_writes_the_rows_in_place_and_nothing_else(Hk: int, d: int, cap: int) -> None:
    """each write form the engine issues - a prefill, a forest slab, decode steps, the drafter's tree nodes and
    their rewrite, float rows cast on the way, a layer's own buffer moving in, the last rows of the stretch - into
    the first, a middle and the last layer's K and V; after every write the whole arena equals the reference"""
    from btb import mlx as mlxdev

    _sh, layers, arena = _arena(Hk, d, cap)
    ref = np.zeros_like(arena)
    g = torch.Generator().manual_seed(Hk * 1000 + d + cap)

    def rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * 4).to(dtype)

    for li in (0, LAYERS // 2, LAYERS - 1):
        for which in (0, 1):
            buf = layers[li]._mx[which]
            slot = 2 * li + which

            def check(n0: int, want: Bits, buf: mx_.array = buf, slot: int = slot) -> None:
                T = want.shape[1]
                ref[slot, :, n0 : n0 + T] = want
                assert np.array_equal(arena, ref), f"layer slot {slot} write at {n0}+{T}: arena differs"
                assert np.array_equal(_view_bits(buf)[:, n0 : n0 + T], want)

            # prefill: GrowLayer._store hands a[0] of [1, Hk, T, d]
            t = rand(1, Hk, 37, d)
            _store(buf, mlxdev.to_mx(t)[0], 0)
            check(0, _bits(t[0]))
            # forest_store: [T, Hk, d] transposed, a strided input
            t = rand(23, Hk, d)
            _store(buf, mlxdev.to_mx(t).transpose(1, 0, 2), 37)
            check(37, _bits(t.transpose(0, 1)))
            # decode steps, one row each
            for n0 in range(60, 64):
                t = rand(1, Hk, 1, d)
                _store(buf, mlxdev.to_mx(t)[0], n0)
                check(n0, _bits(t[0]))
            # the drafter's tree nodes: kh [B, Hk, 1, d] as kh[:, :, 0, :].transpose(1, 0, 2), then a later
            # group's nodes written over the same rows
            for B in (5, 3):
                t = rand(B, Hk, 1, d)
                _store(buf, mlxdev.to_mx(t)[:, :, 0, :].transpose(1, 0, 2), 64)
                check(64, _bits(t[:, :, 0, :].transpose(0, 1)))
            # float32 rows: kv_store casts to the buffer's bf16 (round to nearest even, as torch's cast)
            t = rand(1, Hk, 9, d, dtype=torch.float32)
            _store(buf, mlxdev.to_mx(t)[0], 70)
            check(70, _bits(t[0]))
            # a layer's own buffer moving into the arena (GrowLayer._ensure): old[0][0, :, :n] of a wider cap
            own = rand(1, Hk, cap + 64, d)
            _store(buf, mlxdev.to_mx(own)[0, :, :90], 0)
            check(0, _bits(own[0, :, :90]))
            # the stretch's last rows, n0 + T == cap
            t = rand(1, Hk, 11, d)
            _store(buf, mlxdev.to_mx(t)[0], cap - 11)
            check(cap - 11, _bits(t[0]))


@pytest.mark.parametrize("Hk,d", SHAPES)
def test_gather_moves_the_kept_rows_in_place(Hk: int, d: int) -> None:
    """kv_gather over the keep lists GrowLayer.gather hands it: an accepted tree path (the root untouched, the
    path's rows packed after it) and a full permutation from row 0 (rows moving back and forth, so every row is
    read before any is overwritten); the rows past the kept ones are left as they were"""
    from btb import mlx as mlxdev

    m = mlxdev.mx()
    cap = 1024
    _sh, layers, arena = _arena(Hk, d, cap)
    rng = random.Random(Hk * 31 + d)
    n = 300
    fill = (torch.randn(2 * LAYERS, Hk, n, d, generator=torch.Generator().manual_seed(d)) * 4).bfloat16()
    for slot in range(2 * LAYERS):
        _store(layers[slot // 2]._mx[slot % 2], mlxdev.to_mx(fill[slot]), 0)
    ref = arena.copy()
    assert np.array_equal(ref[:, :, :n], _bits(fill))
    root = 200
    path = sorted(rng.sample(range(root, n), 40))
    perm = list(range(n))
    rng.shuffle(perm)
    for keep, base in ((list(range(root)) + path, root), (perm, 0)):
        for slot in range(2 * LAYERS):
            buf = layers[slot // 2]._mx[slot % 2]
            before = ref[slot].copy()
            flag = mlxdev.kv_gather(buf, keep[base:], base)
            m.eval(flag)
            assert flag.tolist() == [1]
            ref[slot, :, base : len(keep)] = before[:, keep[base:]]
            assert np.array_equal(arena, ref), f"slot {slot} gather from {base}: arena differs"
            assert np.array_equal(_view_bits(buf), ref[slot])


@pytest.mark.parametrize("Hk,d", SHAPES)
def test_arena_layer_appends_gathers_and_grows_out(Hk: int, d: int) -> None:
    """a GrowLayer in the arena driven as the engine drives it - a prefill, decode steps, a verify pass, the
    accepted path gathered (eagerly and with the flags handed back), then an append past the stretch that moves
    the layer to a buffer of its own - its K/V after each step against a plain concatenation of the rows"""
    from btb import mlx as mlxdev

    m = mlxdev.mx()
    cap = 256
    _sh, layers, arena = _arena(Hk, d, cap)
    layer = layers[1]
    g = torch.Generator().manual_seed(Hk + d)
    ref_k = torch.empty(1, Hk, 0, d, dtype=torch.bfloat16)
    ref_v = torch.empty(1, Hk, 0, d, dtype=torch.bfloat16)

    def agree() -> None:
        n = ref_k.shape[2]
        assert layer.get_seq_length() == n
        k, v = layer.mx_kv(0, n)
        assert np.array_equal(np.array(k.view(m.uint16)), _bits(ref_k))
        assert np.array_equal(np.array(v.view(m.uint16)), _bits(ref_v))
        assert torch.equal(layer.keys, ref_k) and torch.equal(layer.values, ref_v)

    def append(T: int) -> None:
        nonlocal ref_k, ref_v
        k = (torch.randn(1, Hk, T, d, generator=g) * 4).bfloat16()
        v = (torch.randn(1, Hk, T, d, generator=g) * 4).bfloat16()
        pk, pv = layer.mx_update(mlxdev.to_mx(k), mlxdev.to_mx(v))
        m.eval(pk, pv)
        ref_k, ref_v = torch.cat([ref_k, k], 2), torch.cat([ref_v, v], 2)
        assert np.array_equal(np.array(pk.view(m.uint16)), _bits(ref_k))
        assert np.array_equal(np.array(pv.view(m.uint16)), _bits(ref_v))
        agree()

    append(50)
    for _ in range(3):
        append(1)
    append(8)
    keep = list(range(53)) + [54, 57, 60]
    assert layer.gather(keep, base=53) == []
    ref_k, ref_v = ref_k[:, :, keep], ref_v[:, :, keep]
    agree()
    append(6)
    keep = list(range(56)) + [58, 61]
    flags = layer.gather(keep, base=56, lazy=True)
    assert len(flags) == 2
    m.eval(*flags)
    ref_k, ref_v = ref_k[:, :, keep], ref_v[:, :, keep]
    agree()
    # the arena's copy of the layer is what the growth carries out; past it the arena is no longer written
    assert np.array_equal(arena[2, :, :58], _bits(ref_k[0])) and np.array_equal(arena[3, :, :58], _bits(ref_v[0]))
    frozen = arena.copy()
    append(cap)
    assert layer.arena is None
    assert np.array_equal(arena, frozen)
    append(1)
