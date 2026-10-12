# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A fork's layer (`ForkLayer`): its rows in one buffer, the shared prefix copied in once and each step written in
place; the buffer granted as it grows, the one it replaces counted; and the rows made the layer's dtype when it
runs somewhere that computes in another (float32 on the host, the card's bf16), as a `GrowLayer`'s are. And a
sparse-attention layer grown by concatenation (`GrantedIndexedLayer`, and a card program's `ArenaIndexedLayer` once it
lets the arena go): priced before the pass as its grant asks, drawing on the epoch what it allocates, keeping what it
asked on each device; its rows leaving the arena for another device moved straight there. No model."""

from __future__ import annotations

import types
from typing import Any, cast

import torch

from btb.engine.cache import ArenaIndexedLayer, ForkIndexedLayer, ForkLayer, GrantedIndexedLayer
from btb.engine.device import where

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


# -- a sparse-attention layer grown by concatenation --------------------------------------------------------------

DI = 3  # the indexer's key width
KV_ROW, IK_ROW = 2 * HK * D * 4, DI * 4  # a position's bytes in float32: keys and values, the indexer's key


def _recorder() -> tuple[list[dict[str, Any]], Any]:
    asked: list[dict[str, Any]] = []

    def grant(nbytes: int, kind: str, **kw: Any) -> None:
        asked.append({"nbytes": nbytes, "kind": kind, **kw})

    return asked, grant


def _drawn(asked: list[dict[str, Any]]) -> int:
    """what the grants took off the epoch's KV: each one's bytes less the ones it replaces"""
    return sum(a["nbytes"] - a.get("held", 0) for a in asked if a.get("draws") is None)


def _append(cl: Any, T: int, dev: str = "cpu") -> None:
    cl.update(torch.ones(1, HK, T, D, device=dev), torch.ones(1, HK, T, D, device=dev))
    cl.update_indexer(torch.ones(1, T, DI, device=dev))


def test_a_concatenated_layer_is_priced_as_it_asks_and_draws_what_it_allocates() -> None:
    """the growth priced before a pass is what the grant then asks (twice what the rows reach, once a doubling);
    what comes off the epoch is what the rows allocate, never the doubling's room ahead"""
    from btb.engine.memory import _MemoryMixin

    asked, grant = _recorder()
    cl = GrantedIndexedLayer(grant)
    cpu = where("cpu")
    probe = types.SimpleNamespace(
        cfg=types.SimpleNamespace(num_attention_heads=HK, num_key_value_heads=HK, head_dim=D, hidden_size=HK * D),
        resident={0: None},
        dev=cpu,
        compute_dtype=torch.float32,
    )
    probe._kv_home = lambda i, layer: _MemoryMixin._kv_home(cast("_MemoryMixin", probe), i, layer)
    cache = types.SimpleNamespace(layers=[cl])
    growth = lambda T: _MemoryMixin.cache_growth(cast("_MemoryMixin", probe), cast(Any, cache), 1, T)
    assert growth(P) == {cpu: 2 * P * KV_ROW}, "an empty layer: its keys' growth, from the shape the config gives"
    _append(cl, P)
    assert [a["nbytes"] for a in asked] == [2 * P * KV_ROW]
    assert growth(1) == {cpu: 0}, "an append inside the room asked for asks nothing"
    n = P
    while not (priced := growth(1)[cpu]):
        _append(cl, 1)
        n += 1
    assert priced == 2 * (n + 1) * (KV_ROW + IK_ROW), "the doubling, the indexer's keys counted once seen"
    before = len(asked)
    _append(cl, 1)
    assert len(asked) == before + 1 and asked[-1]["nbytes"] <= priced
    assert _drawn(asked) <= cl._held_bytes(), f"drew {_drawn(asked)} on the epoch for {cl._held_bytes()} bytes of rows"
    assert _drawn(asked) >= (n + 1) * KV_ROW


def test_a_layer_hopping_back_to_a_device_asks_nothing_it_had_there() -> None:
    """rows moved to another device (a shed, a prefill's hop) are counted there as the move granted them: the growth
    there draws only its own rows; moved back, the layer grows into the room it asked for before"""
    asked, grant = _recorder()
    cl = GrantedIndexedLayer(grant)
    _append(cl, P)
    assert len(asked) == 1 and _drawn(asked) == P * KV_ROW
    assert cl.keys is not None and cl.values is not None and cl.indexer_keys is not None
    meta = torch.device("meta")
    cl.keys, cl.values = cl.keys.to(meta), cl.values.to(meta)
    cl.indexer_keys = cl.indexer_keys.to(meta)
    _append(cl, 1, "meta")
    assert len(asked) == 2 and asked[1]["device"] == meta
    assert asked[1]["nbytes"] - asked[1]["held"] == KV_ROW, "the moved rows drawn on again"
    # back where it grew (fresh rows standing in for the move: a meta tensor holds none to copy)
    cl.keys, cl.values = torch.ones(1, HK, P + 1, D), torch.ones(1, HK, P + 1, D)
    cl.indexer_keys = torch.ones(1, P + 1, DI)
    _append(cl, 1)
    assert len(asked) == 2, "a hop back asked the whole layer again"


def test_an_arena_layer_leaving_for_another_device_moves_its_rows_straight_there() -> None:
    """a card program's layer given up to another device (a shed moving its keys there): every row goes straight to
    that device, granted there - nothing copied, or asked for, where the arena is (the room a shed is short of) -
    and the layer, detached, grows through the grant as a `GrantedIndexedLayer` does; while attached it prices
    nothing (the arena grows by the program's own grant)"""
    asked, grant = _recorder()

    def grow(need: int) -> None:
        raise AssertionError(f"the arena was asked to grow to {need}")

    cap = 16
    arena = torch.zeros(HK, cap, D), torch.zeros(HK, cap, D), torch.zeros(cap, DI)
    cl = ArenaIndexedLayer(grow, grant)
    cl.attach(*arena)
    _append(cl, P)
    assert cl.attached and not asked and cl.get_seq_length() == P
    assert cl.growth(1, cap, HK, D, torch.float32, torch.device("cpu")) == 0
    meta = torch.device("meta")
    assert cl.keys is not None
    cl.keys = cl.keys.to(meta)
    assert not cl.attached
    assert [(a["device"], a["draws"], a["nbytes"]) for a in asked] == [(meta, "", P * (KV_ROW // 2 + IK_ROW))]
    rows = (cl.keys, cl.values, cl.indexer_keys)
    assert all(t is not None and t.device == meta and t.shape[-2] == P for t in rows)
    _append(cl, 1, "meta")
    assert len(asked) == 2 and asked[1]["device"] == meta and asked[1].get("draws") is None
    assert asked[1]["nbytes"] - asked[1]["held"] == KV_ROW, "detached, a growth draws its own rows"
    # a batch's rows (same device) let the arena go as a copy of the rows, granted where they are
    asked.clear()
    cl2 = ArenaIndexedLayer(grow, grant)
    cl2.attach(*arena)
    cl2.update(torch.ones(1, HK, P, D), torch.ones(1, HK, P, D))
    cl2.detach()
    assert [(a["device"], a["draws"]) for a in asked] == [(torch.device("cpu"), "")]
    assert cl2.keys is not None and cl2.keys.data_ptr() != arena[0].data_ptr()


def test_a_rows_step_the_card_has_no_room_for_goes_on_through_the_torch_pass() -> None:
    """a fork's or a batch's step on the card graph whose buffers the ledger refuses (another program took the card)
    turns the card graph off as a single step's refusal does (`_card_oom`) and takes the step through the torch pass,
    the rows leaving the arena: the answer goes on - raised, it failed mid-decode, and every step after was refused
    alike. Any other failure is the caller's, the rows where the step began"""
    from btb.engine.branches import _Rows
    from btb.engine.cuda import _CudaMixin
    from btb.engine.scheduler import MemoryGrantError

    oomed: list[BaseException] = []
    eng = types.SimpleNamespace(
        _is_card_oom=_CudaMixin._is_card_oom,
        _card_oom=oomed.append,
        _card_rows_ok=lambda B, cache=None: True,
        _card_rows_release=lambda cache: None,
    )

    class _Stepped(_Rows):  # the rows alone: what a fork and a batch share
        def close(self) -> None:
            pass

    rows = _Stepped(cast(Any, eng))
    rows.cache = cast(Any, types.SimpleNamespace(layers=[]))
    seen: list[str] = []
    fail: list[BaseException] = []

    def step(cache: Any, toks: list[int], taps: Any, host: bool) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        seen.append(rows.mode)
        if rows.mode == "card" or fail:
            raise fail[0] if fail else MemoryGrantError("[grant] REFUSED the card graph's buffers for 2 rows")
        return torch.zeros(len(toks), 4), {}

    rows._step_rows = step  # type: ignore[method-assign]
    rows.mode = "card"
    lg, _ = rows._advance([1, 2])
    assert seen == ["card", "fork"] and rows.mode == "fork" and len(oomed) == 1 and lg.shape == (2, 4)
    rows.mode = "card"
    fail.append(ValueError("a token past the vocabulary"))
    try:
        rows._advance([1, 2])
    except ValueError:
        pass
    else:
        raise AssertionError("a failure that is not the card's room was taken for one")
    assert rows.mode == "card" and len(oomed) == 1
