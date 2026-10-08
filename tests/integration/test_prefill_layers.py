# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A prefill layer by layer against the same prompt's chunks taken through every layer in turn - a mixture's, a
dense model's, a hybrid's: the last row's logits and every layer's cache - attention rows, the DeltaNet and
convolution states, the sparse indexer's keys - are the same bits, on the host tier, on the drive's tier (each drive
layer's ring slot held across its chunks), on a card pass that streams the host layers in as templates, and on a
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
    """the prompt's last-row logits and its cache's tensors, prefilled with `prefill_layers` as given. A hybrid's
    chunks without it are the chunked loop's own passes, driven here: the engine takes a hybrid's prompt whole
    outside the layer-by-layer path"""
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
            t = torch.tensor([ids])
            if not layers and sm.fam.hybrid and not sm.fam.own and sm.prefill_chunk:
                # the chunked loop's passes (`_prefill`): each chunk one forward, continuing the one before
                C, sm._batched_cont = int(sm.prefill_chunk), True
                try:
                    for a in range(0, t.shape[1], C):
                        out = sm.forward(t[:, a : a + C], cache=cache, last_only=True)
                finally:
                    sm._batched_cont = False
                assert out is not None  # a whole pass returns its logits
                lg = out[0, -1].float().cpu()
            else:
                lg = sm._prefill(t, cache)[0, -1].float().cpu()
            # a host layer's rows where it runs once the prompt is in, whichever path took it: a card pass that hopped
            # them onto the card puts them back, its last run carrying the head as much as one ending short of it
            away = [
                i
                for i in sm.host
                if isinstance(getattr(cache.layers[i], "keys", None), torch.Tensor)
                and cache.layers[i].keys.device.type != "cpu"
            ]
            assert not away, f"host layers {away}: their rows left on the card past the prefill"
            got = {name: t.detach().float().cpu().clone() for name, t in _tensors(cache)}
    finally:
        sm.close()
    return lg, got, log


def _prompt(n: int = 20) -> list[int]:
    """the receipts' prompt tokens repeated to `n`: several chunks, a short one last"""
    base = [int(t) for t in torch.as_tensor(receipts("q4")["host"]["prompt"]).flatten().tolist()]
    return (base * (n // len(base) + 1))[:n]


def _same(kw: dict[str, Any], chunk: int, ids: list[int] | None = None) -> list[str]:
    """the layer-by-layer prefill against the chunks, bit for bit; the layer-by-layer run's log"""
    ids = _prompt() if ids is None else ids
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


# layer by layer is not a mixture's alone: a dense model's and a hybrid's streamed layers are read once a prompt too
DENSE = ["tiny_qwen3", "tiny_q35"]


def _ids(n: int = 20) -> list[int]:
    """`n` tokens inside every fixture's vocabulary (256 the smallest): six chunks of three and a short one"""
    return [(7 * i + 3) % 200 + 2 for i in range(n)]


@pytest.mark.parametrize("fx", DENSE)
def test_dense_host_layers_by_layer_are_the_chunks_bits(fx: str) -> None:
    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    _same({"fx": fx, "device": "cpu", "cpu_layers": range(L)}, chunk=3, ids=_ids())


@pytest.mark.parametrize("fx", DENSE)
def test_drive_layers_by_layer_hold_their_ring_slot(fx: str) -> None:
    """the drive's tier on the host: a drive layer waits for its ring slot once, holds it across all the layer's
    chunks and frees it after the last, two slots for the layers between - the bits the chunks' own passes give"""
    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    kw = {"fx": fx, "device": "cpu", "cpu_layers": range(L), "cold_layers": range(1, L - 1), "cold_slots": 2}
    _same(kw, chunk=3, ids=_ids())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fx", DENSE)
def test_dense_card_pass_by_layer_is_the_chunks_bits(fx: str, dtype: torch.dtype) -> None:
    """in bf16 too: the chunks' resident layers take the card graph's kernels (`_forward_card_prefill`), the
    chunked loop's a run of them a chunk, the sweep's a layer at a time - each row its step's either way, an engine's
    first chunk too (its kernels were read before they were loaded, and its first chunk took torch's)"""
    dev = need_cuda()
    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    kw = {"fx": fx, "device": dev, "cpu_layers": range(L - 2), "resident_layers": range(L - 2, L), "prefill_card": True}
    _same({**kw, "prefill_card_min": 1, "prefetch": True, "compute_dtype": dtype}, chunk=3, ids=_ids())


def test_a_card_sweep_under_the_causal_rule_is_the_masks_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    """a dense model's bf16 card sweep, its chunks' full-attention layers taking their causal mask as the rule
    (`ChunkCausal`: the efficient kernel's grouped call, no mask built, no keys widened), against the same sweep
    with the mask built and the keys widened: the same logits and cache, bit for bit"""
    dev = need_cuda()
    L = layer_count(fixture("tiny_qwen3"))
    StreamedTextModel.register_attention()
    kw = {
        "fx": "tiny_qwen3",
        "device": dev,
        "cpu_layers": range(L - 2),
        "resident_layers": range(L - 2, L),
        "prefill_card": True,
        "prefill_card_min": 1,
        "prefill_chunk": 3,
        "compute_dtype": torch.bfloat16,
    }
    took: list[bool] = []
    orig = StreamedTextModel._chunk_rule

    def rule(self: StreamedTextModel, B: int) -> bool:
        took.append(orig(self, B))
        return took[-1]

    monkeypatch.setattr(StreamedTextModel, "_chunk_rule", rule)
    got, got_cache, _ = _prefill(True, _ids(), **kw)
    assert any(took), "the sweep's chunks took the rule"
    monkeypatch.setattr(StreamedTextModel, "_chunk_rule", lambda self, B: False)
    ref, ref_cache, _ = _prefill(True, _ids(), **kw)
    assert torch.equal(got, ref), f"logits part by {float((got - ref).abs().max()):.3e}"
    for name, t in ref_cache.items():
        assert torch.equal(got_cache[name], t), f"{name} parts by {float((got_cache[name] - t).abs().max()):.3e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("past,T", [(0, 64), (0, 1), (300, 64), (1000, 7)])
@pytest.mark.parametrize("Hq,Hk,D", [(16, 8, 128), (16, 16, 128), (32, 8, 64), (8, 1, 256), (8, 4, 80)])
def test_a_chunks_causal_rule_is_its_masks_bits(
    Hq: int, Hk: int, D: int, past: int, T: int, dtype: torch.dtype
) -> None:
    """`btb_sdpa` over a chunk's causal rule (`ChunkCausal`: the group's heads batched over the keys as they lie,
    the kernel's own lower-right mask) against the same chunk under transformers' bool mask with the keys widened to
    every head: the same bits, over group widths 1 to 8 and head sizes 64 to 256; and where the rule's call cannot
    take the chunk - the keys past the rule's reach - the rule materialized, which is the mask"""
    from transformers.masking_utils import sdpa_mask

    from btb.engine.families.attention import ChunkCausal, attention

    dev = need_cuda()
    torch.manual_seed(3)
    S = past + T
    mod = type("M", (), {"num_key_value_groups": Hq // Hk})()
    q = torch.randn(1, Hq, T, D, device=dev, dtype=dtype)
    k = torch.randn(1, Hk, S, D, device=dev, dtype=dtype)
    v = torch.randn(1, Hk, S, D, device=dev, dtype=dtype)
    mask = sdpa_mask(batch_size=1, q_length=T, kv_length=S, q_offset=past, allow_is_causal_skip=False, device=dev)
    assert mask is not None
    want, _ = attention(mod, q, k, v, mask, scaling=D**-0.5)
    got, _ = attention(mod, q, k, v, ChunkCausal(past), scaling=D**-0.5)
    assert torch.equal(got, want)
    # keys past the rule's reach (a cache longer than past + T): the rule's call steps aside for the rule's mask
    k2 = torch.cat([k, torch.randn_like(k[:, :, :5])], dim=2)
    v2 = torch.cat([v, torch.randn_like(v[:, :, :5])], dim=2)
    wide = torch.cat([mask, torch.zeros_like(mask[..., :5])], dim=-1)
    want2, _ = attention(mod, q, k2, v2, wide, scaling=D**-0.5)
    got2, _ = attention(mod, q, k2, v2, ChunkCausal(past), scaling=D**-0.5)
    assert torch.equal(got2, want2)


@pytest.mark.parametrize("fx", DENSE)
def test_drive_layers_on_a_card_pass_by_layer(fx: str) -> None:
    """drive layers on a card pass whose last chunk is short: the full chunks take each drive layer as the card's
    template, the short one takes it from the ring on the host, which the prefill holds for it"""
    dev = need_cuda()
    L = layer_count(fixture(fx))
    StreamedTextModel.register_attention()
    kw = {
        "fx": fx,
        "device": dev,
        "cpu_layers": range(L - 2),
        "cold_layers": range(1, L - 2),
        "resident_layers": range(L - 2, L),
        "prefill_card": True,
        "prefill_card_min": 3,
        "prefetch": True,
    }
    _same(kw, chunk=3, ids=_ids())


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

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    monkeypatch.setattr(LayerDepot, "MAX_SEATS", 1)
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

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    monkeypatch.setattr(LayerDepot, "SCRATCH", 2)
    monkeypatch.setattr(LayerDepot, "MAX_SEATS", 1)
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


def test_a_depot_with_no_room_yet_opens_once_there_is() -> None:
    """the depot opens at the first call's form, with its scratch slots, where the ledger has them: a call that
    finds no room leaves it closed, not stuck at a form with no slots - the next call asks again and opens it"""
    import types

    from btb.engine.experts import LayerDepot

    dev = need_cuda()
    room = [0]
    ledger = types.SimpleNamespace(
        free=lambda *a, **k: room[0],
        lend=lambda make, nbytes, device, counted, **k: make(),
    )
    depot = LayerDepot(torch.device(dev), ledger)
    try:
        parts = (torch.zeros(4, 8, dtype=torch.bfloat16), torch.zeros(8, 4, dtype=torch.bfloat16))
        assert not depot.takes(parts), "no room: the loop takes the call"
        assert depot.form is None and depot.stat["refused"] == 1
        room[0] = 1 << 30
        assert depot.takes(parts), "room now: the depot opens"
        assert depot.blocks and depot.form is not None
    finally:
        depot.close()


def test_a_depot_opened_up_front_grows_to_a_layers_seats_as_far_as_the_ledger_lets_it() -> None:
    """the sweep opens its depot before its working set is cut: its scratch and a layer's seats, a block at a time,
    until the ledger has no block to give - the seats it holds, and none past the room"""
    import types

    from btb.engine.experts import LayerDepot

    dev = need_cuda()
    form = ((torch.Size([4, 8]), torch.bfloat16), (torch.Size([8, 4]), torch.bfloat16))
    per = 2 * 4 * 8 * 2
    room = [(LayerDepot.SCRATCH + 2 * LayerDepot.BLOCK) * per]

    def lend(make: Any, nbytes: int, device: Any, counted: bool, **k: Any) -> Any:
        if torch.device(device).type == "cuda":
            room[0] -= int(nbytes)  # what the depot takes is gone from the card's room, as its free reading shows it
        return make()

    ledger = types.SimpleNamespace(free=lambda *a, **k: room[0], lend=lend)
    depot = LayerDepot(torch.device(dev), ledger)
    try:
        assert depot.open_at(form, 128) == 2 * LayerDepot.BLOCK, "two blocks' room, two blocks of seats"
        assert depot.form == form and len(depot.blocks) == 3
        assert depot.open_at(form, 128) == 2 * LayerDepot.BLOCK, "opened already: nothing more taken"
    finally:
        depot.close()


def test_a_sweep_under_store_pressure_is_the_chunks_bits() -> None:
    """the store held to a little past a layer's experts, the read-ahead free to take half of it: the sweep's
    predictions, the evictions they make and the slots they recycle must never land in bytes a call is about to
    multiply - the layer-by-layer prefill stays the chunked one's, bit for bit, at every store size (a read-ahead
    that did once collapsed a 120B's routing to a handful of experts, and no roomy store here showed it)"""
    L = layer_count(fixture("tiny_q4"))
    kw: dict[str, Any] = {"device": "cpu", "cpu_layers": range(L), "prefill_chunk": 3}
    ids = _prompt(24)
    ref, ref_cache, _log = _prefill(False, ids, **kw)
    StreamedTextModel.register_attention()
    sm = StreamedTextModel(fixture("tiny_q4"), resident_head=True, log=NO_LOG, compute_dtype=torch.float32, **kw)
    try:
        sm.prefill_layers = True
        with torch.inference_mode():
            sm._prefill(torch.tensor([ids]), sm.new_cache())  # the store sized, its layout read
            store = sm.expert_store
            assert store is not None and store.per is not None
            n_exp = int(sm.n_experts)
            # below one call's experts (served in waves), then a little past a layer's
            for n_slots in (3, 5, n_exp + 2, n_exp + 4, 2 * n_exp):
                for b in list(store.blocks):
                    store._release_block(b)
                store.n_slots = n_slots
                store.block_max = n_slots * int(store.per)  # a block of exactly these seats, no more
                store._grow(n_slots)
                assert store.live() == n_slots, (store.live(), n_slots)
                store.ring_n = max(1, n_slots // 2)
                cache = sm.new_cache()
                got = sm._prefill(torch.tensor([ids]), cache)[0, -1].float().cpu()
                assert torch.equal(got, ref), f"{n_slots} slots: logits part by {float((got - ref).abs().max()):.3e}"
                for name, t in _tensors(cache):
                    want = ref_cache[name]
                    assert torch.equal(t.float().cpu(), want), f"{n_slots} slots: {name} parts"
    finally:
        sm.close()


@pytest.mark.parametrize("layout", ["host layers on the card pass", "every layer resident"])
def test_a_card_sweep_under_store_pressure_is_the_chunks_bits(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    """the store under the same pressure, the chunks' experts on the card's grouped path through the depot: the
    read-ahead's reads, the depot's uploads out of the store's slots and the store's evictions never cross - the
    sweep stays the chunked prefill's, bit for bit. And the depot is the tier a chunk asks first: an expert seated
    on the card by an earlier chunk of its layer is not asked of the store again, so however small the store, a
    layer's experts are read from the drive once for the prompt, not once a chunk"""
    dev = need_cuda()
    from btb.engine.native import Native

    L = layer_count(fixture("tiny_q4"))
    StreamedTextModel.register_attention()
    monkeypatch.setattr(Native, "gemm_rows", 2)
    place: dict[str, Any] = (
        {"cpu_layers": range(L - 2), "resident_layers": range(L - 2, L)}
        if layout.startswith("host")
        else {"resident_layers": range(L)}
    )
    kw: dict[str, Any] = {
        "device": dev,
        **place,
        "prefill_card": True,
        "prefill_card_min": 1,
        "prefetch": True,
        "compute_dtype": torch.bfloat16,
        "prefill_chunk": 3,
    }
    ids = _prompt(24)
    ref, ref_cache, _log = _prefill(False, ids, **kw)
    sm = StreamedTextModel(fixture("tiny_q4"), resident_head=True, log=NO_LOG, **kw)
    try:
        sm.prefill_layers = True
        with torch.inference_mode():
            sm._prefill(torch.tensor([ids]), sm.new_cache())
            store = sm.expert_store
            if store is None or store.per is None:
                pytest.skip("no expert went through the store in this layout")
            n_exp = int(sm.n_experts)
            # below one call's experts (served in waves), then a little past a layer's
            for n_slots in (3, 5, n_exp + 2, n_exp + 4, 2 * n_exp):
                for b in list(store.blocks):
                    store._release_block(b)
                store.n_slots = n_slots
                store.block_max = n_slots * int(store.per)  # a block of exactly these seats, no more
                store._grow(n_slots)
                assert store.live() == n_slots, (store.live(), n_slots)
                store.ring_n = max(1, n_slots // 2)
                cache = sm.new_cache()
                m0 = store.stat["miss"]
                got = sm._prefill(torch.tensor([ids]), cache)[0, -1].float().cpu()
                reads = store.stat["miss"] - m0
                assert reads <= L * n_exp, (
                    f"{n_slots} slots: {reads} reads, past every layer's experts once ({L * n_exp})"
                )
                assert torch.equal(got, ref), f"{n_slots} slots: logits part by {float((got - ref).abs().max()):.3e}"
                for name, t in _tensors(cache):
                    assert torch.equal(t.float().cpu(), ref_cache[name]), f"{n_slots} slots: {name} parts"
    finally:
        sm.close()
