# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The gated DeltaNet's chunked rule a block at a time (btb/engine/fused.py `chunk_gated_delta_rule`): transformers'
rule within float32's reach, and its bits a function of each block's rows and the state it starts from - a prompt
resumed from a block's state, or cut into calls of whole blocks, is the prompt taken whole, at any thread count.
Model-free: random rows at the heads' shapes."""

from __future__ import annotations

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as reference

from btb.engine.fused import DELTA_BLOCK, chunk_gated_delta_rule

B, H, D = 1, 4, 32
# a prompt's q, k, v [B, n, H, D] and its decay and beta [B, n, H]
Rows = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def rows(n: int, seed: int, dtype: torch.dtype = torch.float32) -> Rows:
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, n, H, D, generator=g).to(dtype) for _ in range(3))
    decay = -torch.rand(B, n, H, generator=g) * 0.5
    beta = torch.rand(B, n, H, generator=g).to(dtype)
    return q, k, v, decay, beta


def run(xs: Rows, a: int, b: int, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v, g, beta = (x[:, a:b] for x in xs)
    out, st = chunk_gated_delta_rule(
        q, k, v, g, beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert st is not None
    return out, st


def test_the_rule_is_transformers_rule() -> None:
    """the same rule as transformers' reference - only the order of its float32 sums differs - its state too, from
    no state and from one of its own"""
    xs = rows(150, 1)
    q, k, v, g, beta = xs
    got, st = run(xs, 0, 150, None)
    ref, ref_st = reference(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert ref_st is not None
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), float((got - ref).abs().max())
    assert torch.allclose(st, ref_st, atol=1e-6, rtol=1e-4), float((st - ref_st).abs().max())
    s0 = torch.randn(B, H, D, D) * 0.1
    got, _ = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=s0, use_qk_l2norm_in_kernel=True)
    ref, _ = reference(q, k, v, g, beta, initial_state=s0, use_qk_l2norm_in_kernel=True)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), "from a state of its own"


def test_a_prompt_cut_at_its_blocks_is_the_prompt_whole() -> None:
    """resumed from the state at any block's start, or taken in calls of whole blocks (a sweep's chunks), the rows
    and the final state are the prompt's taken whole - bit for bit, at the machine's thread count and at one (the
    reference batched its products over the call's blocks, and the math library split a batch of a few blocks
    across its threads otherwise: a state resumed late in a prompt parted by 3e-8)"""
    n = 300  # a short block last
    xs = rows(n, 2, torch.bfloat16)
    threads = torch.get_num_threads()
    try:
        for t in sorted({threads, 1}):
            torch.set_num_threads(t)
            whole, whole_st = run(xs, 0, n, None)
            for cut in range(DELTA_BLOCK, n, DELTA_BLOCK):
                _, s = run(xs, 0, cut, None)
                rest, st = run(xs, cut, n, s)
                assert torch.equal(rest, whole[:, cut:]) and torch.equal(st, whole_st), (t, cut)
            for size in (DELTA_BLOCK, 2 * DELTA_BLOCK, 3 * DELTA_BLOCK):
                state: torch.Tensor | None = None
                outs: list[torch.Tensor] = []
                for a in range(0, n, size):
                    o, state = run(xs, a, min(n, a + size), state)
                    outs.append(o)
                assert state is not None
                assert torch.equal(torch.cat(outs, 1), whole) and torch.equal(state, whole_st), (t, size)
    finally:
        torch.set_num_threads(threads)


def test_a_cut_inside_a_block_is_not_the_prompt_whole() -> None:
    """why a resume starts at a block's start: cut inside one, the rows after the cut sum otherwise (and a block's
    rows recomputed from its start are the prompt's again)"""
    n = 200
    xs = rows(n, 3, torch.bfloat16)
    whole, _ = run(xs, 0, n, None)
    _, s = run(xs, 0, 100, None)
    rest, _ = run(xs, 100, n, s)
    assert not torch.equal(rest, whole[:, 100:])
    _, s = run(xs, 0, DELTA_BLOCK, None)
    rest, _ = run(xs, DELTA_BLOCK, n, s)
    assert torch.equal(rest[:, 100 - DELTA_BLOCK :], whole[:, 100:])
