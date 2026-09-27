# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A fork's layer (`ForkLayer`): its rows in one buffer, the shared prefix copied in once and each step written in
place; the buffer granted as it grows, the one it replaces counted; and the rows made the layer's dtype when it
runs somewhere that computes in another (float32 on the host, the card's bf16), as a `GrowLayer`'s are. No model."""

from __future__ import annotations

from typing import Any

import torch

from btb.engine.cache import ForkIndexedLayer, ForkLayer

B, HK, P, D = 3, 2, 5, 4


def _prefix(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, HK, P, D, generator=g).to(dtype), torch.randn(1, HK, P, D, generator=g).to(dtype)


def test_a_step_is_written_after_the_shared_prefix_in_one_buffer() -> None:
    pk, pv = _prefix()
    fl = ForkLayer(pk, pv, B)
    assert fl.keys.shape == (B, HK, P, D) and fl.get_seq_length() == P
    k1, v1 = torch.ones(B, HK, 1, D), torch.full((B, HK, 1, D), 2.0)
    k, v = fl.update(k1, v1)
    assert k.shape == (B, HK, P + 1, D) and fl.get_seq_length() == P + 1
    assert torch.equal(k[:, :, :P], pk.expand(B, -1, -1, -1)) and torch.equal(k[:, :, P:], k1)
    buf = fl._kv[0] if fl._kv is not None else None
    fl.update(k1 * 3, v1 * 3)
    assert fl._kv is not None and fl._kv[0] is buf, "a step with room is written in place: nothing joined"
    row = fl.row(1)
    assert row is not None and torch.equal(row[0], torch.cat([k1, k1 * 3], dim=2)[1:2])


def test_the_buffer_is_granted_counting_the_one_it_replaces() -> None:
    asked: list[dict[str, Any]] = []

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        asked.append({"nbytes": nbytes, "kind": kind, **kw})

    fl = ForkLayer(*_prefix(), B, grant=grant)
    fl.update(torch.ones(B, HK, 1, D), torch.ones(B, HK, 1, D))
    assert len(asked) == 1 and asked[0]["held"] == 0 and asked[0]["draws"] is None, "a growth draws on the epoch"
    first = asked[0]["nbytes"]
    assert first == 2 * B * HK * (P + 64) * D * 4
    fl.update(torch.ones(B, HK, 70, D), torch.ones(B, HK, 70, D))
    assert len(asked) == 2 and asked[1]["held"] == first, "the old buffer, let go once copied, is counted"


def test_the_rows_take_the_dtype_the_layer_computes_in_where_it_runs_now() -> None:
    """a fork stepped with its layer on the host (float32), then with the layer on the card (bf16): the rows are
    made bf16 - the attention takes its keys in the queries' dtype - and keep the room they had"""
    fl = ForkLayer(*_prefix(), B)
    fl.update(torch.ones(B, HK, 1, D), torch.ones(B, HK, 1, D))
    cap = fl._kv[0].shape[-2] if fl._kv is not None else 0
    k, v = fl.update(torch.ones(B, HK, 1, D, dtype=torch.bfloat16), torch.ones(B, HK, 1, D, dtype=torch.bfloat16))
    assert k.dtype == v.dtype == torch.bfloat16 and fl.get_seq_length() == P + 2
    assert fl._kv is not None and fl._kv[0].shape[-2] == cap, "a cast alone keeps the room it had"
    pk, _pv = _prefix()
    assert torch.equal(k[:, :, :P], pk.to(torch.bfloat16).expand(B, -1, -1, -1)), "cast as a growth into bf16 casts"


def test_an_indexed_layers_keys_follow_the_same_rule() -> None:
    ik = torch.randn(1, P, 3)
    fl = ForkIndexedLayer(*_prefix(), ik, B)
    fl.update_indexer(torch.ones(B, 1, 3))
    got = fl.update_indexer(torch.ones(B, 1, 3, dtype=torch.bfloat16))
    assert got.dtype == torch.bfloat16 and got.shape == (B, P + 2, 3)
    assert torch.equal(got[:, :P], ik.to(torch.bfloat16).expand(B, -1, -1))
