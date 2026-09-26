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
        fixture(kw.pop("fx", "tiny_q4")),
        resident_head=True,
        log=lambda *a, **k: log.append(" ".join(str(x) for x in a)),
        compute_dtype=kw.pop("compute_dtype", torch.float32),
        **{k: v for k, v in kw.items() if k != "grouped"},
    )
    try:
        sm.prefill_layers = layers
        if "grouped" in kw:
            sm.grouped_experts = bool(kw["grouped"])
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

    def one_seat(self: LayerDepot, parts: tuple[torch.Tensor, ...]) -> bool:
        n = self.SCRATCH + 1
        self.stacks = [torch.empty((n, *p.shape), dtype=p.dtype, device=self.dev) for p in parts]
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


def _grouped_vs_loop(kw: dict[str, Any]) -> int:
    """the layer-by-layer prefill with a call's experts as grouped matmuls against the per-expert loop, bit for bit;
    how many calls went grouped"""
    from btb.engine.host import _Experts

    calls = [0]
    inner = _Experts._card_grouped

    def counted(self: Any, *a: Any, **k: Any) -> Any:
        calls[0] += 1
        return inner(self, *a, **k)

    ids = _prompt()
    ref, ref_cache, _ = _prefill(True, ids, prefill_chunk=3, grouped=False, **kw)
    _Experts._card_grouped = counted  # type: ignore[method-assign]
    try:
        got, got_cache, _ = _prefill(True, ids, prefill_chunk=3, grouped=True, **kw)
    finally:
        _Experts._card_grouped = inner  # type: ignore[method-assign]
    assert torch.equal(got, ref), f"logits part by {float((got - ref).abs().max()):.3e}"
    assert set(got_cache) == set(ref_cache)
    for name, t in ref_cache.items():
        assert torch.equal(got_cache[name], t), f"{name} parts by {float((got_cache[name] - t).abs().max()):.3e}"
    return calls[0]


def test_grouped_experts_are_the_loops_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """a call's experts as grouped matmuls over the depot's slots give the per-expert loop's bits: the logits and
    every cache tensor of a bf16 prefill whose expert calls all take the card's path"""
    dev = need_cuda()
    from btb.engine.native import Native

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    n = _grouped_vs_loop({**kw, "prefill_card_min": 1, "prefetch": True, "compute_dtype": torch.bfloat16})
    assert n > 0, "no call took the grouped path"


def test_grouped_waves_through_a_small_scratch_are_the_loops_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """a depot of one seat and two scratch slots: a call's experts go in several waves, each taking the scratch over
    once the card is done with the last, and the bits stay the loop's"""
    dev = need_cuda()
    from btb.engine.experts import LayerDepot
    from btb.engine.native import Native

    def one_seat(self: LayerDepot, parts: tuple[torch.Tensor, ...]) -> bool:
        n = self.SCRATCH + 1
        self.stacks = [torch.empty((n, *p.shape), dtype=p.dtype, device=self.dev) for p in parts]
        self.n_seats = 1
        return True

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    monkeypatch.setattr(LayerDepot, "SCRATCH", 2)
    monkeypatch.setattr(LayerDepot, "_open", one_seat)
    kw = {"device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    n = _grouped_vs_loop({**kw, "prefill_card_min": 1, "prefetch": True, "compute_dtype": torch.bfloat16})
    assert n > 0, "no call took the grouped path"


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


def test_the_prefill_lookahead_never_takes_the_calls_own_slots() -> None:
    """layer 0's call holds its slots, the store has no free one and the ring may grow over all of it: the sweep's
    predictions for layer 1 evict other riders, never the call's own - a prediction read into one would land in the
    bytes the call is about to multiply (it once did, and a 180B prefill's routing collapsed)"""
    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    sm = StreamedTextModel(
        fixture("tiny_q4"),
        resident_head=True,
        log=NO_LOG,
        device="cpu",
        cpu_layers=range(L),
        compute_dtype=torch.float32,
        prefill_chunk=3,
    )
    try:
        with torch.inference_mode():
            sm._prefill(torch.tensor([_prompt()]), sm.new_cache())
            store = sm.expert_store
            assert store is not None and store.per is not None
            ex = sm.host[0].mlp.experts
            ids = list(range(int(ex.num_experts)))
            views, pending = store.get(0, ex.base, ids, keep=True, rows=len(ids))
            store.wait(pending)
            held = set(store.last_slots.values())
            # the store down to layer 0's call alone: every other rider off it, no free slot, no room to grow - the
            # ring can grow only by evicting, and only the call's own riders are left to evict
            for key, _s in list(store.res.items()):
                if key[0] != 0:
                    store.res.pop(key)
            store.free.clear()
            store.n_slots = store.live()
            store.ring_n = store.n_slots
            h = torch.randn(len(ids), int(sm.cfg.hidden_size))
            made = store.lookahead(0, h, sweep=len(ids))
            assert not (set(store.ring) & held), (sorted(set(store.ring) & held), made)
    finally:
        sm.close()


@pytest.mark.parametrize("fx", ["tiny_q4", "tiny_gpt_oss"])
def test_the_lookahead_finds_each_familys_router(fx: str) -> None:
    """the store reads the next layer's routing off its router whatever the family names it - Qwen's `mlp.gate`,
    gpt-oss's `mlp.router` with its bias - and so predicts on both (on gpt-oss it once found none and read nothing)"""
    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    sm = StreamedTextModel(
        fixture(fx),
        resident_head=True,
        log=NO_LOG,
        device="cpu",
        cpu_layers=range(L),
        prefill_chunk=3,
        compute_dtype=torch.float32,
    )
    try:
        with torch.inference_mode():
            sm._prefill(torch.tensor([_prompt() if fx == "tiny_q4" else list(range(3, 23))]), sm.new_cache())
            store = sm.expert_store
            assert store is not None
            wb = store._router(1)
            assert wb is not None, "no router found for layer 1"
            w, bias = wb
            if fx == "tiny_gpt_oss":
                assert bias is not None, "gpt-oss's router bias was not read"
            for key in [k for k, _s in list(store.res.items()) if k[0] == 1]:
                store.res.pop(key)  # layer 1's experts off the store: something to predict
            made = store.lookahead(0, torch.randn(4, int(sm.cfg.hidden_size)), sweep=4)
            assert made > 0, "the lookahead predicted nothing"
    finally:
        sm.close()


# gpt-oss's FP8 twin keeps its MXFP4 experts (an FP8 trunk beside them); tiny_q4's has FP8 experts
@pytest.mark.parametrize("fx", ["tiny_gpt_oss", "tiny_gpt_oss-f8_e4m3", "tiny_q4-f8_e4m3"])
def test_stored_experts_on_the_card_match_the_hosts_kernel(monkeypatch: pytest.MonkeyPatch, fx: str) -> None:
    """experts stored narrow (gpt-oss's MXFP4, an FP8 checkpoint's e4m3) on a card pass, widened on the card and
    multiplied as grouped matmuls, against the per-expert loop (each expert on the host's kernel for its form, handed
    back to the card as bf16): the same steps dtype for dtype, so the logits agree to bf16's rounding and the greedy
    token is the same; the cache's every tensor too"""
    dev = need_cuda()
    from btb.engine.host import _Experts
    from btb.engine.native import Native

    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    calls = [0]
    inner = _Experts._card_grouped

    def counted(self: Any, *a: Any, **k: Any) -> Any:
        calls[0] += 1 if self.mx or self.f8 else 0
        return inner(self, *a, **k)

    kw = {
        "fx": fx,
        "device": dev,
        "cpu_layers": range(L - 1),
        "resident_layers": range(L - 1, L),
        "prefill_card": True,
        "prefill_card_min": 1,
        "prefetch": True,
        "compute_dtype": torch.bfloat16,
        "prefill_chunk": 3,
    }
    ids = list(range(3, 23))
    ref, ref_cache, _ = _prefill(True, ids, grouped=False, **dict(kw))
    monkeypatch.setattr(_Experts, "_card_grouped", counted)
    got, got_cache, _ = _prefill(True, ids, grouped=True, **dict(kw))
    assert calls[0] > 0, f"no {fx} call took the card's grouped path"
    scale = float(ref.abs().max())
    d = float((got - ref).abs().max())
    print(f"[{fx}] logits part by {d:.3e} of a {scale:.3e} scale; argmax {int(got.argmax())} vs {int(ref.argmax())}")
    assert int(got.argmax()) == int(ref.argmax())
    assert d <= 2e-2 * scale, (d, scale)
    for name, t in ref_cache.items():
        g = got_cache[name]
        assert g.shape == t.shape, name
        assert float((g - t).abs().max()) <= 5e-2 * max(1.0, float(t.abs().max())), name
