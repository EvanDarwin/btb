# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A mixture's experts on the card, below the passes that use them: the layer depot a prefill sweeps its chunks
through (btb/engine/experts.py `LayerDepot`) against a stub ledger - its blocks refused by the scheduler or the card,
an upload with no pinned ring, an expert it cannot take handed back, a call after a wave waiting on the wave's
scratch - each slot holding its expert's bytes; the expert store's first-class seats on the card (`VramSeats`,
`_open_vram`) asked of the scheduler, filled as experts ride and served with no bit changed, and multiplied without
the card's kernels within float32's reach; a grouped call's MXFP4 widening by torch where the card has no kernel,
bit for bit the kernel's; a wave with no place on the card refused by name; and the GPU-recovery backoff."""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from btb.engine.experts import ExpertProfile, LayerDepot
from btb.engine.scheduler import MemoryGrantError
from tests.helpers import NO_LOG, fixture, forward_logits, host_model, layer_count, need_cuda, rel_err

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")

H, INTER = 8, 4  # a stub expert: gate_up [2 * INTER, H], down [H, INTER]
PER = (2 * INTER * H + H * INTER) * 2
PROMPT = [3, 17, 42, 5, 99, 120, 7, 7, 200, 12, 45, 8]


def _expert(seed: int, inter: int = INTER) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2 * inter, H, generator=g).bfloat16(), torch.randn(H, inter, generator=g).bfloat16()


def _ledger(pin: bool = True) -> types.SimpleNamespace:
    """a ledger with room to spare; `pin` False: a machine that will not pin the depot's ring"""

    def lend(make: Any, nbytes: int, device: Any, counted: bool, **_k: Any) -> Any:
        if not pin and torch.device(device).type == "cpu":
            raise RuntimeError("the test's machine will not pin")
        return make()

    return types.SimpleNamespace(free=lambda *_a, **_k: 1 << 30, lend=lend)


def _slot_is(depot: LayerDepot, s: int, parts: tuple[torch.Tensor, ...]) -> bool:
    torch.cuda.synchronize()
    stacks, row = depot.at(s)
    return all(torch.equal(st[row].cpu(), p) for st, p in zip(stacks, parts, strict=True))


# -- the layer depot ------------------------------------------------------------------------------------------------


def test_a_block_the_scheduler_or_the_card_refuses_seats_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """the depot's blocks are asked of the scheduler before they are made: refused, the depot does not open (the
    loop takes the call); a block the card itself cannot make grows no seat, and the expert rides a scratch slot;
    a closed depot makes nothing and asks for nothing"""
    dev = torch.device(need_cuda())
    asked: list[tuple[int, str]] = []
    refuse = [True]

    def grant(nbytes: int, kind: str, **_kw: Any) -> None:
        asked.append((int(nbytes), kind))
        if refuse[0]:
            raise MemoryGrantError("[grant] REFUSED the test's depot")

    depot = LayerDepot(dev, _ledger(), types.SimpleNamespace(grant=grant))
    try:
        gu, dn = _expert(0)
        assert not depot.takes((gu, dn)) and depot.form is None and depot.stat["refused"] == 1
        assert asked == [(LayerDepot.SCRATCH * PER, "depot")]
        refuse[0] = False
        assert depot.takes((gu, dn)) and depot.n_seats == 0
        real = torch.empty

        def empty(*a: Any, **k: Any) -> torch.Tensor:
            if k.get("device") is not None and torch.device(k["device"]).type == "cuda":
                raise torch.OutOfMemoryError("the test's card is full")
            return real(*a, **k)

        with monkeypatch.context() as mp:
            mp.setattr(torch, "empty", empty)
            assert depot._block(LayerDepot.BLOCK) is None
            got = depot.get(0, 3, gu, dn)
        assert depot.n_seats == 0 and depot.stat["scratch"] == 1 and depot.stat["seated"] == 0
        assert all(t.device.type == "cuda" for t in got) and _slot_is(depot, 0, (gu, dn))
    finally:
        depot.close()
    n = len(asked)
    assert depot._block(LayerDepot.BLOCK) is None and len(asked) == n, "a closed depot asked for a block"


def test_uploads_with_no_pinned_ring_go_from_the_stores_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """a machine that will not pin the depot's ring, or `BTB_PREFILL_STAGE=0`: each expert crosses from the store's
    own pages, and its slot holds its bytes all the same"""
    dev = torch.device(need_cuda())
    for how in ("no pin", "stage off"):
        with monkeypatch.context() as mp:
            if how == "stage off":
                mp.setenv("BTB_PREFILL_STAGE", "0")
            depot = LayerDepot(dev, _ledger(pin=how != "no pin"))
            try:
                experts = {e: _expert(e) for e in (1, 4)}
                assert depot.takes(experts[1]) and depot.stage is None, how
                slots = depot.place(0, [(e, *p) for e, p in experts.items()])
                assert depot.stat["seated"] == 2 and all(s is not None for s in slots), how
                for s, p in zip(slots, experts.values(), strict=True):
                    assert s is not None and _slot_is(depot, s, p), f"{how}: a seat does not hold its expert"
            finally:
                depot.close()


def test_an_expert_the_depot_cannot_take_is_handed_back() -> None:
    """an expert of another form than the depot opened at, or one already on the card: the loop's `get` hands its
    own views back, a wave's `place` gives it no slot - the rest of the wave placed around it"""
    dev = torch.device(need_cuda())
    depot = LayerDepot(dev, _ledger())
    try:
        gu, dn = _expert(0)
        assert depot.takes((gu, dn))
        other = _expert(1, inter=2 * INTER)
        got = depot.get(0, 5, *other)
        assert got[0] is other[0] and got[1] is other[1] and depot.stat["passed"] == 1
        on_card = (gu.to(dev), dn.to(dev))
        slots = depot.place(1, [(2, gu, dn), (3, *on_card), (6, *other)])
        assert slots[0] is not None and slots[1:] == [None, None] and depot.stat["passed"] == 3
        assert _slot_is(depot, slots[0], (gu, dn))
    finally:
        depot.close()


def test_a_loop_call_after_a_wave_waits_for_the_waves_scratch(monkeypatch: pytest.MonkeyPatch) -> None:
    """with no seats to grow, a wave's experts ride the scratch slots; a per-expert call after it takes a scratch
    slot only once the card is done with the wave, and the slot then holds the new expert's bytes"""
    dev = torch.device(need_cuda())
    monkeypatch.setattr(LayerDepot, "MAX_SEATS", 0)
    depot = LayerDepot(dev, _ledger())
    try:
        wave = {e: _expert(e) for e in range(3)}
        assert depot.takes(wave[0])
        slots = depot.place(0, [(e, *p) for e, p in wave.items()])
        assert slots == [0, 1, 2] and depot.fence_scratch and depot.stat["scratch"] == 3
        late = _expert(7)
        depot.get(0, 7, *late)
        assert not depot.fence_scratch, "the call did not wait for the wave"
        assert _slot_is(depot, 0, late) and _slot_is(depot, 1, wave[1])
    finally:
        depot.close()


# -- the store's seats on the card ------------------------------------------------------------------------------


def test_a_seat_is_given_only_to_a_rider_that_earned_it() -> None:
    """`VramSeats.offer` seats a rider that has ridden `min_rides` times while the pass has a promotion left, copying
    its bytes; it refuses one already seated, one short of its rides, one past the pass's promotions, and every one
    where there are no seats at all; with the seats full the least recently ridden gives its seat up"""
    from btb.engine.experts import VramSeats

    dev = torch.device(need_cuda())
    shapes = (2 * INTER * H * 2, (2 * INTER, H), (H, INTER))
    region = lambda seed: torch.cat([p.reshape(-1).view(torch.uint8) for p in _expert(seed)])
    assert not VramSeats(0, PER, shapes, dev).offer("a", 9, region(0)), "no seats, yet one was given"
    v = VramSeats(2, PER, shapes, dev, min_rides=2, per_pass=2)
    assert not v.offer("a", 1, region(0)) and "a" not in v, "seated short of its rides"
    assert v.offer("a", 2, region(0)) and not v.offer("a", 5, region(0)), "seated twice"
    assert v.offer("b", 3, region(1)) and not v.offer("c", 3, region(2)), "past the pass's promotions"
    v.new_pass()
    v.views("a")  # "a" ridden since "b": "b" is the least recently ridden
    assert v.offer("c", 3, region(2)) and "b" not in v and set(v.seat_of) == {"a", "c"} and v.copies == 3
    for key, seed in (("a", 0), ("c", 2)):
        gu, dn = v.views(key)
        want = _expert(seed)
        assert torch.equal(gu.cpu(), want[0]) and torch.equal(dn.cpu(), want[1]), f"seat {key} holds other bytes"


@pytest.fixture(scope="module")
def q4() -> Iterator[Any]:
    """tiny_q4 on the card's engine with every layer on the host, its store reading into unpadded slots (the
    bounce, `BTB_STORE_PADDED=0`): a seat is the slot's bytes as they lie"""
    import os

    from btb.engine import StreamedTextModel

    need_cuda()
    StreamedTextModel.register_attention()
    path = fixture("tiny_q4")
    was = os.environ.get("BTB_STORE_PADDED")
    os.environ["BTB_STORE_PADDED"] = "0"
    try:
        sm = host_model(path, device="cuda", cpu_layers=range(layer_count(path)))
        with torch.inference_mode():
            forward_logits(sm, [PROMPT], sm.new_cache())  # the store's recipe read under the variable
    finally:
        if was is None:
            os.environ.pop("BTB_STORE_PADDED", None)
        else:
            os.environ["BTB_STORE_PADDED"] = was
    try:
        yield sm
    finally:
        sm.close()


def _steps(sm: Any, toks: list[int]) -> list[torch.Tensor]:
    cache = sm.new_cache()
    out = [forward_logits(sm, [PROMPT], cache)[0, -1].clone()]
    return out + [forward_logits(sm, [[t]], cache)[0, -1].clone() for t in toks]


def test_the_stores_seats_on_the_card_are_granted_and_change_no_bit(
    q4: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """`vram_experts_gb` as a figure or "auto" (the card's free memory less a gigabyte) opens that many seats, each
    asked of the scheduler (refused, none made); too little room for one opens none. Experts that ride enough are copied from their
    unpadded slots into the seats and multiplied there as the host multiplies them (the profile records each such
    hit), so a decode's every logit is the one without seats; without the card's kernels the seats' float32 matmuls
    stay within float32's reach of it"""
    import btb.engine.host as host_mod
    from btb.engine.native import Native

    sm = q4
    store = sm.expert_store
    assert store is not None and store.per is not None and not store.padded
    toks = [77, 5, 9]
    with torch.inference_mode():
        want = _steps(sm, toks)
        per = store._held(store.per)
        store.vram = None
        for gb, n in ((1e-12, 0), ((3 * per + per // 2) / 2**30, 3), ("auto", 2)):
            with monkeypatch.context() as mp:
                mp.setattr(sm, "vram_experts_gb", gb, raising=False)
                # "auto" reads free as the grant does, less every reservation: an epoch's KV is not the seats'
                mp.setattr(sm.scheduler, "free_for", lambda device=None, draws=None: (1 << 30) + 2 * per)
                before = sm.scheduler.granted.get("experts@cuda", 0)
                store.vram = None
                store._open_vram()
                got_n = store.vram.n if store.vram is not None else 0
                assert got_n == n, f"{gb} GB: {got_n} seats, not {n}"
                assert sm.scheduler.granted.get("experts@cuda", 0) - before == n * per, "the seats were not asked for"
        with monkeypatch.context() as mp:
            logs: list[str] = []
            mp.setattr(sm, "vram_experts_gb", 1.0, raising=False)
            mp.setattr(sm, "log", lambda *a, **_k: logs.append(" ".join(str(x) for x in a)))

            def refuse(nbytes: int, kind: str, **_kw: Any) -> None:
                raise MemoryGrantError("[grant] REFUSED the test's seats")

            mp.setattr(sm.scheduler, "grant", refuse)
            kept = store.vram
            store._open_vram()
            assert store.vram is kept, "seats the scheduler refused were made"
            assert any("no seats on the card: [grant] REFUSED the test's seats" in ln for ln in logs), logs
        # a seat for every expert of the model, so none the decode seats is given up before its next ride
        every = int(sm.L) * int(sm.cfg.num_experts)
        with monkeypatch.context() as mp:
            mp.setattr(sm, "vram_experts_gb", (every * per + per // 2) / 2**30, raising=False)
            store.vram = None
            store._open_vram()
        assert store.vram is not None and store.vram.n == every
        store.vram.min_rides, store.vram.per_pass = 1, every
        prof = ExpertProfile(str(tmp_path / "events.npz"))
        monkeypatch.setattr(sm, "expert_profile", prof)
        _steps(sm, toks)  # the first rides seat the experts
        seated = _steps(sm, toks)
        assert store.vram.copies > 0 and store.vram.seat_of, "no expert was seated on the card"
        ev = prof.a[: prof.n]
        assert bool(((ev[:, 2] == prof.HIT) & (ev[:, 9] == -2) & (ev[:, 10] == 2)).any()), "no hit from a seat"
        proxy = types.SimpleNamespace(**{k: getattr(Native, k) for k in dir(Native) if not k.startswith("__")})
        proxy.card_kernels = lambda: None
        monkeypatch.setattr(host_mod, "Native", proxy)
        torch_seats = _steps(sm, toks)
        monkeypatch.setattr(sm, "expert_profile", None)
        store.vram = None
    for j, (a, b) in enumerate(zip(want, seated, strict=True)):
        assert torch.equal(a, b), f"pass {j} parts with experts seated on the card"
    for j, (a, b) in enumerate(zip(want, torch_seats, strict=True)):
        assert rel_err(b, a) < 1e-4, f"pass {j}: the seats' float32 matmuls are {rel_err(b, a):.2e} from the host's"


def test_mxfp4_experts_seated_on_the_card_change_no_bit(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """gpt-oss's MXFP4 experts take seats on the card as stored (four parts, the checkpoint's layout) and are
    multiplied there by the host's MXFP4 gemv's bits (`gemv_lane16_mx4`), their gate/up bias before the gate and their
    down bias before the row's rounding as the host adds them - so a prompt's rows (the per-expert loop) and each
    decode step's (the one-row path) are the logits without seats, bit for bit"""
    from btb.engine import StreamedTextModel
    from tests.helpers import need_card_kernels

    need_card_kernels()
    StreamedTextModel.register_attention()
    path = fixture("tiny_gpt_oss")
    sm = host_model(path, device="cuda", cpu_layers=range(layer_count(path)))
    try:
        store = sm.expert_store
        toks = [77, 5, 9, 31]
        with torch.inference_mode():
            want = _steps(sm, toks)
            assert store is not None and store.mx and store.per is not None
            every = int(sm.L) * int(sm.n_experts)
            with monkeypatch.context() as mp:
                mp.setattr(sm, "vram_experts_gb", (every * store.per + store.per // 2) / 2**30, raising=False)
                store.vram = None
                store._open_vram()
            assert store.vram is not None and store.vram.n == every, "no MXFP4 seats were opened"
            store.vram.min_rides, store.vram.per_pass = 1, every
            prof = ExpertProfile(str(tmp_path / "events.npz"))
            monkeypatch.setattr(sm, "expert_profile", prof)
            _steps(sm, toks)  # the first rides seat the experts
            seated = _steps(sm, toks)
            monkeypatch.setattr(sm, "expert_profile", None)
            assert store.vram.copies > 0 and store.vram.seat_of, "no expert was seated on the card"
            gu, dn = store.vram.views(next(iter(store.vram.seat_of)))
            assert gu.blocks.is_cuda and gu.scales is not None and dn.blocks.is_cuda, "a seat is not MXFP4 on the card"
            ev = prof.a[: prof.n]
            assert bool(((ev[:, 2] == prof.HIT) & (ev[:, 9] == -2) & (ev[:, 10] == 2)).any()), "no hit from a seat"
            store.vram = None
        for j, (a, b) in enumerate(zip(want, seated, strict=True)):
            assert torch.equal(a, b), f"pass {j} parts with MXFP4 experts seated on the card"
    finally:
        sm.close()


def test_the_layout_is_read_off_the_first_layer_with_experts(q4: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """the form a prefill's depot opens at, read off the store's layout with no expert read: a layer with no experts
    (a dense one: its recipe has no such tensors) is passed over for the next, and a model with none has no form"""
    store = q4.expert_store
    assert store is not None
    form = store.form()
    assert form is not None and all(dt == torch.bfloat16 for _shape, dt in form)
    real = store._recipe

    def dense(first_dense: int) -> Any:
        def recipe(layer: int, base: str) -> Any:
            if layer < first_dense:
                raise KeyError(f"{base}gate_up_proj")
            return real(layer, base)

        return recipe

    for first, want in ((1, form), (int(q4.L), None)):
        with monkeypatch.context() as mp:
            # the layout read afresh (the store's first recipe), as the store read it: into unpadded slots
            mp.setenv("BTB_STORE_PADDED", "0")
            mp.setattr(store, "shapes", None)
            mp.setattr(store, "recipes", {})
            mp.setattr(store, "_recipe", dense(first))
            assert store.form() == want, f"the first {first} layers dense"


# -- a grouped call on the card -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt_oss() -> Iterator[Any]:
    """tiny_gpt_oss swept layer by layer on the card, its last layer resident: a chunk's MXFP4 experts go to the
    depot and are widened there for grouped matmuls"""
    from btb.engine import StreamedTextModel

    dev = need_cuda()
    StreamedTextModel.register_attention()
    path = fixture("tiny_gpt_oss")
    L = layer_count(path)
    sm = StreamedTextModel(
        path,
        resident_head=True,
        log=NO_LOG,
        device=dev,
        cpu_layers=range(L - 1),
        resident_layers=range(L - 1, L),
        prefill_card=True,
        prefill_card_min=1,
        prefetch=True,
        compute_dtype=torch.bfloat16,
        prefill_chunk=3,
    )
    try:
        yield sm
    finally:
        sm.close()


def test_mxfp4_widened_by_torch_is_the_card_kernels_widening(gpt_oss: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """a grouped call's MXFP4 experts widened on the card by its kernel (`mx4_widen`) or, with no kernel, by torch
    (`dequant_blocks`): the same bf16 weights, so a prefill's logits and every cache tensor are the same bits"""
    import btb.engine.host as host_mod
    from btb.engine.host import _Experts
    from btb.engine.native import Native

    sm = gpt_oss
    monkeypatch.setattr(Native, "gemm_rows", 2)  # a 3-row chunk takes the card's expert path
    calls = [0]
    inner = _Experts._card_grouped

    def counted(self: Any, *a: Any, **k: Any) -> Any:
        calls[0] += 1 if self.mx else 0
        return inner(self, *a, **k)

    monkeypatch.setattr(_Experts, "_card_grouped", counted)
    ids = torch.tensor([list(range(3, 23))])
    out = {}
    for kernel in (True, False):
        with monkeypatch.context() as mp:
            if not kernel:
                proxy = types.SimpleNamespace(**{k: getattr(Native, k) for k in dir(Native) if not k.startswith("__")})
                proxy.card_kernels = lambda: None
                mp.setattr(host_mod, "Native", proxy)
            calls[0] = 0
            with torch.inference_mode():
                cache = sm.new_cache()
                lg = sm._prefill(ids, cache)[0, -1].float().cpu()
                state = [
                    t.detach().float().cpu()
                    for cl in cache.layers
                    for t in (getattr(cl, "keys", None), getattr(cl, "values", None))
                    if isinstance(t, torch.Tensor)
                ]
            assert calls[0] > 0, f"kernel {kernel}: no MXFP4 call took the grouped path"
            assert state, "the cache holds no rows to compare"
            out[kernel] = (lg, state)
    (lg1, st1), (lg0, st0) = out[True], out[False]
    assert torch.equal(lg1, lg0), f"the logits part by {float((lg1 - lg0).abs().max()):.3e}"
    for j, (a, b) in enumerate(zip(st1, st0, strict=True)):
        assert torch.equal(a, b), f"cache tensor {j} parts"


def test_a_family_with_no_card_program_runs_none(gpt_oss: Any) -> None:
    """gpt-oss has no card program of its own: a step over its cache asks for none, and makes none"""
    sm = gpt_oss
    with torch.inference_mode():
        cache = sm.new_cache()
        forward_logits(sm, [PROMPT], cache)
        assert sm._card_program(cache, 1, 1, cache.get_seq_length(), None, None, None) is None
        assert getattr(sm, "_cp", None) is None
        assert sm._card_program_warm(torch.tensor([PROMPT]), 2) == 0


def test_a_wave_with_no_place_on_the_card_is_refused_by_name(gpt_oss: Any) -> None:
    """a wave the depot finds no slot for at all (no seat, no scratch, no form it takes) is an error naming the
    layer, not a call that loops on it"""
    sm = gpt_oss
    L = int(sm.L)
    ex = sm.resident[L - 1].mlp.experts
    dev = sm.dev
    Hd = int(sm.cfg.hidden_size)
    depot = types.SimpleNamespace(place=lambda layer, items: [None] * len(items), settle=lambda: None)
    top = torch.tensor([[0, 1]], device=dev)
    with torch.inference_mode(), pytest.raises(RuntimeError, match=rf"layer {L - 1}: no expert of the wave"):
        ex._card_grouped(
            torch.zeros(1, Hd, dtype=torch.bfloat16, device=dev),
            torch.ones(1, 2, dtype=torch.bfloat16, device=dev),
            top,
            [0, 1],
            {0: (None, None), 1: (None, None)},
            [],
            sm.expert_store,
            depot,
            torch.zeros(1, Hd, dtype=torch.bfloat16, device=dev),
        )


# -- the GPU's recoveries -------------------------------------------------------------------------------------------


def test_gpu_recoveries_back_off_and_a_quiet_stretch_forgets_them(monkeypatch: pytest.MonkeyPatch) -> None:
    """each transient GPU recovery in a streak doubles the pause before the retry, from a tenth of a second to two at
    most; five quiet seconds past the last pause and the next recovery starts the streak again"""
    import btb.engine.scheduler as sched_mod

    now = [100.0]
    monkeypatch.setattr(sched_mod, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    s = sched_mod.BatchScheduler(types.SimpleNamespace())
    waits = []
    for _ in range(7):
        waits.append(s.gpu_recovered())
        now[0] += 0.01
    assert waits == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 2.0, 2.0])
    now[0] += 4.0
    assert s.gpu_recovered() == pytest.approx(2.0), "a stretch inside the cool-down kept the streak"
    now[0] += 7.5
    assert s.gpu_recovered() == pytest.approx(0.1), "a quiet stretch did not forget the streak"
