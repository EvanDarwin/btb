# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX pool's blocks: one engine's at a time, idle and unwrapped once given back, let go by a trim."""

from __future__ import annotations

from btb.pool import ALIGN, Pool
from tests.helpers import need_mlx

MiB = 2**20


def test_a_block_serves_one_owner_until_given_back_and_a_trim_lets_idle_ones_go() -> None:
    need_mlx()
    p, a, b = Pool(), object(), object()
    p._seed_cuts([MiB])
    p._settle()
    assert p.free_bytes() == MiB
    sh, off = p.take(1000, a) or (None, -1)
    assert sh is not None and off == 0 and p.free_bytes() == 0
    assert p.take(1000, b) is None  # a's block, though it has room: b's close could never free a's bytes
    again = p.take(1000, a)
    assert again is not None and again[0] is sh and again[1] == (1000 + ALIGN - 1) // ALIGN * ALIGN
    assert p.trim() == 0 and len(p.blocks) == 1  # held, so kept
    p.give(a)
    assert p.free_bytes() == MiB and p.blocks[0][3] is None  # idle, its torch view gone with a
    got = p.take(1000, b)
    assert got is not None and got[1] == 0
    p.give(b)
    assert p.trim() == MiB and not p.blocks and p.free_bytes() == 0
