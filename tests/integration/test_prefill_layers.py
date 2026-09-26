# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A mixture's prefill layer by layer against the same prompt's chunks taken through every layer in turn: the last
row's logits and every layer's cache - attention rows, the DeltaNet and convolution states, the sparse indexer's
keys - are the same bits, on the host tier, on a card pass that streams the host layers in as templates, and on a
card pass whose short last chunk stays on the host."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from btb.engine import StreamedTextModel
from tests.helpers import NO_LOG, fixture, layer_count, need_cuda, receipts

CACHE_ATTRS = ("keys", "values", "conv_states", "recurrent_states", "indexer_keys")


def _tensors(cache: Any) -> Iterator[tuple[str, torch.Tensor]]:
    for i, cl in enumerate(cache.layers):
        for attr in CACHE_ATTRS:
            t = getattr(cl, attr, None)
            if isinstance(t, dict):
                for k, v in t.items():
                    if isinstance(v, torch.Tensor):
                        yield f"layer {i} {attr}[{k}]", v
            elif isinstance(t, torch.Tensor):
                yield f"layer {i} {attr}", t


def _prefill(layers: bool, ids: list[int], **kw: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    """the prompt's last-row logits and its cache's tensors, prefilled with `prefill_layers` as given"""
    log: list[str] = []
    sm = StreamedTextModel(
        fixture("tiny_q4"),
        resident_head=True,
        log=lambda *a, **k: log.append(" ".join(str(x) for x in a)),
        compute_dtype=torch.float32,
        **kw,
    )
    try:
        sm.prefill_layers = layers
        with torch.inference_mode():
            cache = sm.new_cache()
            lg = sm._prefill(torch.tensor([ids]), cache)[0, -1].float().cpu()
            got = {name: t.detach().float().cpu().clone() for name, t in _tensors(cache)}
    finally:
        sm.close()
    return lg, got, log


def _prompt(n: int = 20) -> list[int]:
    """the receipts' prompt tokens repeated to `n`: several chunks, a short one last"""
    base = [int(t) for t in torch.as_tensor(receipts("q4")["host"]["prompt"]).flatten().tolist()]
    return (base * (n // len(base) + 1))[:n]


def _same(kw: dict[str, Any], chunk: int) -> list[str]:
    """the layer-by-layer prefill against the chunks, bit for bit; the layer-by-layer run's log"""
    ids = _prompt()
    assert len(ids) > 2 * chunk, "the prompt must span at least three chunks"
    ref, ref_cache, ref_log = _prefill(False, ids, prefill_chunk=chunk, **kw)
    got, got_cache, got_log = _prefill(True, ids, prefill_chunk=chunk, **kw)
    assert any("layer by layer" in line for line in got_log), got_log
    assert not any("layer by layer" in line for line in ref_log), ref_log
    assert torch.equal(got, ref), f"logits part by {float((got - ref).abs().max()):.3e}"
    assert set(got_cache) == set(ref_cache)
    for name, t in ref_cache.items():
        assert torch.equal(got_cache[name], t), f"{name} parts by {float((got_cache[name] - t).abs().max()):.3e}"
    return got_log


def test_host_layers_by_layer_are_the_chunks_bits() -> None:
    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    _same({"device": "cpu", "cpu_layers": range(L)}, chunk=3)


def test_card_pass_by_layer_is_the_chunks_bits() -> None:
    dev = need_cuda()
    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    # the last two layers resident, the rest streamed onto the card as templates for every chunk
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    _same({**kw, "prefill_card_min": 1, "prefetch": True}, chunk=3)
    _same({**kw, "prefill_card_min": 1, "prefetch": False}, chunk=3)


def test_the_depot_holds_a_layers_experts_on_the_card(monkeypatch: pytest.MonkeyPatch) -> None:
    """the experts multiplied on the card for every chunk (the rows past the host floor): a layer's experts are
    seated once in the depot and reused by the later chunks, and the bits stay the chunked path's"""
    dev = need_cuda()
    from btb.engine.native import Native

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    # a 3-row chunk takes the card's expert path, as a 64-row one does at the default floor
    monkeypatch.setattr(Native, "gemm_rows", 2)
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    got_log = _same({**kw, "prefill_card_min": 1, "prefetch": True}, chunk=3)
    line = next((ln for ln in got_log if "depot:" in ln), None)
    assert line is not None, got_log
    seated, reused = (int(re.search(rf"(\d+) {what}", line).group(1)) for what in ("experts seated", "reused"))  # type: ignore[union-attr]
    assert seated > 0 and reused > 0, line


def test_experts_past_the_depot_ride_the_scratch_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """a depot of one seat: every other expert is uploaded through the two scratch slots, each chunk again, and the
    bits stay the chunked path's (a scratch slot is rewritten only once the matmuls handed it are done)"""
    dev = need_cuda()
    from btb.engine.experts import LayerDepot
    from btb.engine.native import Native

    def one_seat(self: LayerDepot, gu: torch.Tensor, dn: torch.Tensor) -> bool:
        n = self.SCRATCH + 1
        self.gu = torch.empty((n, *gu.shape), dtype=gu.dtype, device=self.dev)
        self.dn = torch.empty((n, *dn.shape), dtype=dn.dtype, device=self.dev)
        self.n_seats = 1
        return True

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    monkeypatch.setattr(LayerDepot, "_open", one_seat)
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    got_log = _same({**kw, "prefill_card_min": 1, "prefetch": True}, chunk=3)
    line = next((ln for ln in got_log if "depot:" in ln), None)
    assert line is not None, got_log
    scratch = int(re.search(r"(\d+) through scratch", line).group(1))  # type: ignore[union-attr]
    assert scratch > 0, line


def test_grouped_picks_are_the_per_expert_lookup() -> None:
    """one sort of a call's picks gives every expert the (pick, row) pairs `torch.where` finds for it, in its order"""
    from btb.engine.host import group_picks

    g = torch.Generator().manual_seed(0)
    for T, E, k in ((1, 8, 2), (7, 8, 2), (64, 512, 10), (333, 64, 6)):
        top = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
        pos_s, row_s, offs, counts = group_picks(top, E)
        mask = torch.nn.functional.one_hot(top, num_classes=E).permute(2, 1, 0)
        for e in range(E):
            pos, row = torch.where(mask[e])
            a, n = offs[e], counts[e]
            assert torch.equal(pos_s[a : a + n], pos) and torch.equal(row_s[a : a + n], row), (T, E, k, e)


def test_short_last_chunk_stays_on_the_host() -> None:
    dev = need_cuda()
    L = layer_count(fixture("tiny_q4"))
    ids = _prompt()
    chunk = 3
    if len(ids) % chunk == 0:
        pytest.skip("the prompt divides into whole chunks: no short last chunk")
    StreamedTextModel.register_attention()
    # a chunk under the card pass's floor runs its host layers on the host, the others on the card
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    _same({**kw, "prefill_card_min": chunk, "prefetch": True}, chunk=chunk)


def test_the_switch_names_the_reference() -> None:
    """BTB_PREFILL_LAYERS=0 is read at load: the chunked path, the one the bits above are held against"""
    import os

    old = os.environ.get("BTB_PREFILL_LAYERS")
    os.environ["BTB_PREFILL_LAYERS"] = "0"
    try:
        sm = StreamedTextModel(fixture("tiny_q4"), resident_head=True, log=NO_LOG, device="cpu")
        try:
            assert sm.prefill_layers is False
        finally:
            sm.close()
    finally:
        if old is None:
            os.environ.pop("BTB_PREFILL_LAYERS", None)
        else:
            os.environ["BTB_PREFILL_LAYERS"] = old
