# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""gpt-oss's attention with sinks over torch (`attention_sinks`), written into one buffer in place, against the
arithmetic it replaced - the product, the mask added, the sinks joined on, the shift and the softmax, each a new
tensor - bit for bit: the same ops on the same values in the same dtype. On the host in float32 and on the card in
bf16, a prefill chunk past earlier rows, with a float mask, with none (the causal one built), and a window."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from btb.engine.families import ScoresWorkspace, attention_sinks, close_scores, open_scores

B, HQ, HK, T, PAST, D = 1, 8, 2, 12, 20, 16


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
    sinks: torch.Tensor,
    win: int | None,
) -> torch.Tensor:
    """the path as it stood: a new tensor at every step"""
    Bq, Hq, Tq, d = q.shape
    g = Hq // k.shape[1]
    n = int(k.shape[-2])
    start = max(0, (n - Tq) - win + 1) if (win and Tq > 1) else 0
    k, v = k[..., start:, :], v[..., start:, :]
    nk = n - start
    scores = torch.matmul(q.reshape(Bq, k.shape[1], g * Tq, d), k.transpose(-1, -2)).reshape(Bq, Hq, Tq, nk) * scale
    if mask is not None:
        scores = scores + mask[:, :, :, start:n]
    else:
        pos = torch.arange(n - Tq, n, device=scores.device)[:, None]
        j = start + torch.arange(nk, device=scores.device)[None]
        allow = (j <= pos) if win is None else ((j <= pos) & (j > pos - win))
        scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
    combined = torch.cat([scores, sinks.reshape(1, -1, 1, 1).expand(Bq, -1, Tq, -1)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1].to(v.dtype)
    res = torch.matmul(probs.reshape(Bq, k.shape[1], g * Tq, nk), v).reshape(Bq, Hq, Tq, d)
    return res.transpose(1, 2).contiguous()


def _inputs(dev: str, dt: torch.dtype, mask_dt: torch.dtype, sink_dt: torch.dtype) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(3)
    n = PAST + T
    q = torch.randn(B, HQ, T, D, generator=g)
    k = torch.randn(B, HK, n, D, generator=g)
    v = torch.randn(B, HK, n, D, generator=g)
    sinks = torch.randn(HQ, generator=g)
    pos = torch.arange(PAST, n)[:, None]
    j = torch.arange(n)[None]
    mask = torch.zeros(T, n).masked_fill(j > pos, torch.finfo(mask_dt).min).view(1, 1, T, n)
    return (*(t.to(dev, dt) for t in (q, k, v)), sinks.to(dev, sink_dt), mask.to(dev, mask_dt))


# (the pass's dtype, the mask's, the sinks'): the host's float32 throughout; the card's bf16 with transformers'
# float32 mask (made in the dtype of the rows it is handed), and with every dtype its own
DTYPES = {
    "host": ("cpu", torch.float32, torch.float32, torch.float32),
    "card": ("cuda", torch.bfloat16, torch.bfloat16, torch.bfloat16),
    "card, float32 mask": ("cuda", torch.bfloat16, torch.float32, torch.bfloat16),
    "card, float32 mask and sinks": ("cuda", torch.bfloat16, torch.float32, torch.float32),
    "card, float32 sinks": ("cuda", torch.bfloat16, torch.bfloat16, torch.float32),
}


# a sweep's workspace: none (tensors made per call), one that holds the scores, one too small (made per call)
WORKSPACE = {"none": 0, "open": 1 << 20, "too small": 64}


@pytest.mark.parametrize("masked", [True, False])
@pytest.mark.parametrize("win", [None, 7])
@pytest.mark.parametrize("case", list(DTYPES))
@pytest.mark.parametrize("workspace", list(WORKSPACE))
def test_the_sinks_in_one_buffer_are_the_steps_they_replaced(
    case: str, win: int | None, masked: bool, workspace: str
) -> None:
    where, dt, mask_dt, sink_dt = DTYPES[case]
    if where == "cuda" and not torch.cuda.is_available():
        pytest.skip("no card")
    q, k, v, sinks, mask = _inputs(where, dt, mask_dt, sink_dt)
    scale = D**-0.5
    # the workspace opened as a sweep opens it, on the engine's device (`cuda`, not the tensors' `cuda:0`)
    dev = torch.device(where)
    ws = ScoresWorkspace(WORKSPACE[workspace], q.device) if WORKSPACE[workspace] else None
    if ws is not None:
        for buf in ws.bufs.values():
            buf.fill_(0xA5)
        open_scores(dev, ws)
    try:
        got, _ = attention_sinks(
            types.SimpleNamespace(), q, k, v, mask if masked else None, scaling=scale, sliding_window=win, s_aux=sinks
        )
    finally:
        close_scores(dev)
    if ws is not None:
        used = bool((ws.bufs["a"] != 0xA5).any())
        assert used == (workspace == "open"), f"the workspace {'was not' if not used else 'was'} taken"
    want = _reference(q, k, v, mask if masked else None, scale, sinks, win)
    assert got.dtype == want.dtype and torch.equal(got, want)
