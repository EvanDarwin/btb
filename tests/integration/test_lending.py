# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Memory for the caller's own tensors beside a model: `empty`/`zeros`/`full` make room before allocating,
`room` holds room for allocations btb does not make, `memory()` says what can be had - and a layer btb moves
takes every live cache's rows with it, an idle session's and a fork's included."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine import device as device_mod
from btb.engine.cache import ForkLayer, GrowLayer
from btb.engine.device import Device
from btb.engine.host import _HostLinear
from btb.engine.memory import Room
from btb.engine.scheduler import EPOCH, MemoryGrantError
from tests.helpers import fixture, loaded_model, need_cuda, need_mlx

if TYPE_CHECKING:
    from btb.engine.cache import KvCache

PROMPT = [5, 17, 99, 3, 42, 8, 61, 7, 12, 30]
MiB = 2**20


@pytest.fixture(params=["cpu", "mlx"])
def sm(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    if request.param == "mlx":
        need_mlx()
    with loaded_model(fixture("tiny_qwen3"), device=request.param) as m:
        yield m


def squeeze(sm: StreamedTextModel, left: int, mp: pytest.MonkeyPatch) -> None:
    """leave `left` bytes free on the host, as btb counts it: the host's free RAM pinned at its reading here, as
    MLX's ledger is its own already - measured against a live reading, any other process's allocation took the
    last bytes and a decode the test expects to run was refused"""
    if sm.mlx is not None:
        sm.ram_reserve = int(sm.mem_start) - int(sm.mlx.held_bytes()) - left
    else:
        now = device_mod.host_free_bytes()
        mp.setattr(device_mod, "host_free_bytes", lambda: now)
        sm.ram_reserve += max(0, sm.memory()["cpu"].free - left)


def test_rooms_add_up_and_come_back(sm: StreamedTextModel) -> None:
    with sm.room(4 * MiB) as a, sm.room(2 * MiB, name="b"):
        assert sm.memory()["cpu"].reserved == 6 * MiB and a.held
        a.release()
        a.release()
        assert sm.memory()["cpu"].reserved == 2 * MiB and not a.held
    assert sm.memory()["cpu"].reserved == 0
    with sm.reserve("the reranker", MiB):
        assert sm.memory()["cpu"].reserved == MiB
    assert sm.memory()["cpu"].reserved == 0


def test_a_room_dropped_unreleased_is_given_back(sm: StreamedTextModel) -> None:
    r = sm.room(MiB)
    assert sm.memory()["cpu"].reserved == MiB
    del r
    gc.collect()
    assert sm.memory()["cpu"].reserved == 0


def test_a_lent_tensor_is_a_plain_tensor_counted_while_it_lives(sm: StreamedTextModel) -> None:
    """what btb's ledger cannot see (a torch tensor on MLX's unified memory) it counts until the last view is gone;
    where it sees torch's allocations (the CPU, a card) the tensor counts itself"""
    t = sm.full((256, 1024), 2.0, dtype=torch.float32)
    assert isinstance(t, torch.Tensor) and t.device.type == "cpu" and float(t[0, 0]) == 2.0
    counted = 256 * 1024 * 4 if sm.mlx is not None else 0
    assert sm.memory()["cpu"].reserved == counted
    view = t[10:]
    del t
    gc.collect()
    assert sm.memory()["cpu"].reserved == counted
    del view
    gc.collect()
    assert sm.memory()["cpu"].reserved == 0
    assert torch.equal(sm.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.int64))


def test_a_refusal_names_the_request_and_leaves_the_model_answering(
    sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """asked for more than btb can give, it gives up what it holds - warm layers to the drive - and refuses,
    naming the request; the model decodes the same tokens from the drive"""
    ref = list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens)
    squeeze(sm, MiB, monkeypatch)
    with pytest.raises(MemoryGrantError, match=r"shape \[67108864\].*64\.0 MiB"):
        sm.empty(64 * MiB, dtype=torch.uint8)
    assert sm.cold == set(range(sm.L))
    with pytest.raises(MemoryGrantError, match="room 'workspace'"):
        sm.room(64 * MiB, name="workspace")
    assert list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens) == ref


def _cache_bytes(cache: KvCache | None) -> int:
    """what a cache's grown buffers hold, k and v (and an int8 layer's scales)"""
    assert cache is not None
    n = 0
    for cl in cache.layers:
        if isinstance(cl, GrowLayer) and cl._buf is not None:
            n += sum(int(t.nbytes) for t in cl._buf)
        elif isinstance(cl, GrowLayer) and cl._mx is not None and cl.arena is None:
            n += sum(int(x.nbytes) for x in cl._mx)
    return n


def test_a_cache_growth_is_priced_before_the_pass(sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """the room a pass's cache growth needs is asked for before the pass, as the bytes its buffers then hold; a
    pass the buffers already fit asks for nothing"""
    asked: list[int] = []
    make = sm._make_room

    def record(dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
        asked.append(nbytes)
        return make(dev, nbytes, what, own)

    monkeypatch.setattr(sm, "_make_room", record)
    s = sm.session(PROMPT)
    assert sum(asked) == _cache_bytes(s.cache)
    before = len(asked)
    s.feed([1, 2])
    assert len(asked) == before


def test_a_cache_growth_makes_room_instead_of_refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """a prefill whose cache growth finds no room: the grant alone refuses it (what the room a policy took left);
    asked for before the pass, btb gives up what it holds and the prefill runs as it would have"""
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        ref = sm.session(PROMPT).logits
        assert ref is not None
        keep = sm.ram_reserve
        squeeze(sm, 0, monkeypatch)
        with monkeypatch.context() as mp:
            mp.setattr(sm, "cache_room", lambda cache, B, T: None)
            with pytest.raises(MemoryGrantError, match="only"):
                sm.session(PROMPT)

        def give(dev: torch.device, short: int, tried: set[str]) -> bool:
            # a shed's freed room, stood in for: the tiny fixture's layers are too small to free a cache's worth
            sm.ram_reserve = keep
            return True

        monkeypatch.setattr(sm, "_give_up_one", give)
        got = sm.session(PROMPT).logits
        assert got is not None and torch.equal(got, ref)


def test_a_pinned_placement_makes_no_room_for_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    """`adapt` off pins the placement: neither memory policy runs, a cache's growth gives nothing up, and the grant
    refuses what does not fit"""
    with loaded_model(fixture("tiny_qwen3"), device="cpu", adapt=False) as sm:
        assert not sm.adapt and not sm.ram_watch and not sm.vram_watch
        squeeze(sm, 0, monkeypatch)
        gave: list[int] = []

        def give(dev: torch.device, short: int, tried: set[str]) -> bool:
            gave.append(short)
            return True

        monkeypatch.setattr(sm, "_give_up_one", give)
        with pytest.raises(MemoryGrantError, match="only"):
            sm.session(PROMPT)
        assert not gave


def test_a_rooms_own_tensors_count_against_it(sm: StreamedTextModel) -> None:
    """tensors taken through a room fill it: where btb's free reading sees them (the CPU) the room holds only what
    they leave, so neither is counted twice; on MLX, which cannot, the whole room. A tensor past what is left is
    refused, and one gone gives its bytes back to the room"""
    seen = sm.mlx is None
    with sm.room(4 * MiB) as r:
        t = r.zeros(MiB, dtype=torch.uint8)
        assert t.device.type == "cpu" and int(t.sum()) == 0 and r.used == MiB
        assert sm.memory()["cpu"].reserved == (3 * MiB if seen else 4 * MiB)
        with pytest.raises(MemoryGrantError, match="MiB left"):
            r.empty(4 * MiB, dtype=torch.uint8)
        del t
        gc.collect()
        assert r.used == 0 and sm.memory()["cpu"].reserved == 4 * MiB
    assert sm.memory()["cpu"].reserved == 0
    with pytest.raises(ValueError, match="released"):
        r.empty(1)


def test_the_grant_prices_what_rooms_hold_but_not_the_epochs_own_room(sm: StreamedTextModel) -> None:
    """a room is memory promised away: the grant and the batch sizing read it as taken, as `memory()` does - all
    but the epoch's own KV, which the cache's growth inside the epoch draws on"""

    def near(a: int | None, b: int) -> bool:  # two readings of the host's free RAM, a moment apart
        return a is not None and abs(a - b) < 64 * MiB

    with sm.room(256 * MiB):
        assert near(sm.scheduler.free_for("cpu"), sm.memory()["cpu"].free)
        with pytest.raises(MemoryGrantError):
            sm.scheduler.grant(sm.memory()["cpu"].free + 128 * MiB, "scratch", device="cpu")
        sm.device.reserve(EPOCH, 256 * MiB, "cpu")
        try:
            assert near(sm.scheduler.free_for("cpu"), sm.memory()["cpu"].free + 256 * MiB)
        finally:
            sm.device.release(EPOCH)


@pytest.mark.parametrize("order", ["released first", "tensor first"])
@pytest.mark.parametrize("where", ["here", "another thread"])
def test_a_room_released_as_its_tensor_goes_holds_nothing_after(sm: StreamedTextModel, order: str, where: str) -> None:
    """whichever goes first, the room's release or its tensor, and on whichever thread: the room holds nothing
    after, and the tensor's bytes are nobody's (docs/lending.md: nothing is re-held when a tensor goes)"""
    r = sm.room(4 * MiB)
    held = [r.zeros(MiB, dtype=torch.uint8)]
    steps = [r.release, lambda: let_go(held)]
    for step in steps if order == "released first" else steps[::-1]:
        if where == "here":
            step()
        else:
            th = threading.Thread(target=step)
            th.start()
            th.join()
    assert not r.held and r.used == 0 and sm.device.reserved("cpu") == 0


def test_two_threads_filling_a_room_never_take_more_than_it_holds(
    sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = sm.room(3 * MiB)
    empty = torch.empty
    inside, go = threading.Event(), threading.Event()

    def slow(*shape: object, **kw: object) -> torch.Tensor:
        if shape == ((2 * MiB,),) and not inside.is_set():
            inside.set()
            go.wait(0.3)
        return empty(*shape, **kw)  # type: ignore[call-overload]

    monkeypatch.setattr(torch, "empty", slow)
    got: list[object] = []
    a = threading.Thread(target=lambda: got.append(r.empty(2 * MiB, dtype=torch.uint8)))
    a.start()
    assert inside.wait(5)
    try:
        got.append(r.empty(2 * MiB, dtype=torch.uint8))
    except MemoryGrantError as e:
        got.append(e)
    go.set()
    a.join()
    assert sum(isinstance(g, torch.Tensor) for g in got) == 1 and r.used == 2 * MiB
    r.release()


def test_a_room_outliving_its_model_keeps_no_ledger_alive() -> None:
    class Stub:
        dev = torch.device("cpu")

        def _called(self, tag: object) -> None:
            pass

    eng = Stub()
    led = Device(eng)
    gone = weakref.ref(led)
    r = Room(led, "orphan#1", MiB, torch.device("cpu"), True, eng)  # type: ignore[arg-type]
    del led
    gc.collect()
    assert gone() is None
    r.release()
    with pytest.raises(ValueError, match="released"):
        r.empty(1)


def test_memory_read_while_a_layer_moves_is_not_a_crash(sm: StreamedTextModel, monkeypatch: pytest.MonkeyPatch) -> None:
    """`memory()` takes no decode lock: a shed or regrow on the decode's thread can move a layer as it counts"""
    stored = sm._layer_bytes_stored
    moved: list[int] = []

    def moving(i: int, packed: bool = False) -> int:
        if not moved:
            moved.append(i)
            sm.host[sm.L + 7] = sm.host[i]  # a layer arriving mid-count
        return stored(i, packed)

    monkeypatch.setattr(sm, "_layer_bytes_stored", moving)
    try:
        assert sm.memory()["cpu"].sheddable > 0
    finally:
        sm.host.pop(sm.L + 7, None)


def test_a_host_shed_for_a_tensor_grows_back_once_it_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """where the ledger sees torch's allocations (the CPU off MLX) and no RAM policy runs, the layer given up for a
    lent tensor still grows back once the tensor is gone"""
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        sm.ram_watch = False
        make = sm._make_room

        def one_step(dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
            assert sm._give_up_one(dev, nbytes, set())
            return make(dev, 0, what, own)

        monkeypatch.setattr(sm, "_make_room", one_step)
        t = sm.empty(MiB, dtype=torch.uint8)
        monkeypatch.undo()
        assert sm.cold
        del t
        gc.collect()
        assert _until(lambda: not sm.cold, sm), f"still from the drive: {sorted(sm.cold)}"


def test_memory_names_each_device_btb_runs_on(sm: StreamedTextModel) -> None:
    mem = sm.memory()
    assert list(mem) == ["cpu"]
    m = mem["cpu"]
    assert m.free > 0 and m.reserved == 0 and m.sheddable > 0 and m.lendable == m.free + m.sheddable


def _until(cond: Callable[[], bool], sm: StreamedTextModel, seconds: float = 20.0) -> bool:
    """decode now and then (each pass runs the memory policies) until `cond()` or `seconds` pass"""
    deadline = time.time() + seconds
    while not cond() and time.time() < deadline:
        sm.generate(PROMPT, 2, eos=(), speculate=False)
        time.sleep(0.25)
    return bool(cond())


def test_a_layer_grown_back_on_mlx_holds_its_own_weights() -> None:
    """a layer back from the drive holds MLX weights of its own, not views of a ring slot the rebuilt ring fills
    with another layer's - which decoded right only while the pool happened not to hand that slot out again"""
    need_mlx()
    with loaded_model(fixture("tiny_qwen3"), device="mlx") as sm:
        ref = list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens)
        while sm.ram_shed("test") is not None:
            pass
        slots = list(sm.cold_ring.shared or [])
        i = sm.ram_regrow()
        assert i is not None
        slots += list(sm.cold_ring.shared or [])
        views = [getattr(m.mx, "sh", None) for m in sm.host[i].modules() if isinstance(m, _HostLinear)]
        assert views and not any(v is s for v in views for s in slots), "a regrown layer reads a ring slot"
        assert list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens) == ref


def test_what_a_refusal_shed_grows_back_once_there_is_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """MLX runs no host memory policy, so the layers making room shed are grown back by the lending policy"""
    need_mlx()
    with loaded_model(fixture("tiny_qwen3"), device="mlx") as sm:
        keep = sm.ram_reserve
        squeeze(sm, MiB, monkeypatch)
        with pytest.raises(MemoryGrantError):
            sm.empty(64 * MiB, dtype=torch.uint8)
        assert sm.cold
        sm.ram_reserve = keep
        assert _until(lambda: not sm.cold, sm), f"still from the drive: {sorted(sm.cold)}"


def test_a_card_gives_layers_up_for_a_tensor_and_every_cache_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    """a card short of room sheds layers for a tensor; an idle session's rows follow each layer to the host (its
    next turn runs, no device mismatch); once the tensor is gone the layers grow back with no memory policy running.
    The shortage is stood in for: the tiny fixture's layers free less than an allocator block, which no free reading
    shows, so the steps making room are taken as a short card would take them"""
    dev = need_cuda()
    with loaded_model(fixture("tiny_qwen3"), device=dev, adapt=False) as sm:
        idle = sm.session(PROMPT)
        before = set(sm.resident)
        make = sm._make_room

        def short(dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
            for _ in range(2):
                assert sm._give_up_one(dev, nbytes, set())
            return make(dev, 0, what, own)

        monkeypatch.setattr(sm, "_make_room", short)
        t = sm.empty(MiB, dtype=torch.uint8)
        monkeypatch.undo()
        assert t.device.type == "cuda" and set(sm.resident) < before
        for i in before - set(sm.resident):
            assert idle.rows(i)[0].device.type == "cpu", i
        assert len(idle.generate(2, eos=(), speculate=False).tokens) == 2
        del t
        gc.collect()
        assert _until(lambda: set(sm.resident) == before, sm), f"still shed: {sorted(before - set(sm.resident))}"


def test_a_layer_move_reaches_every_live_cache() -> None:
    """a layer btb moves takes its rows in every live cache along: an idle session's, a fork's shared prefix and its
    own rows - not only the running pass's; a cache nobody holds any more is not kept alive for it"""
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        gone = weakref.ref(sm.session(PROMPT).cache)
        idle = sm.session(PROMPT)
        forked = sm.session(PROMPT)
        br = forked.fork(2)
        br.step([1, 2])
        gc.collect()
        assert gone() is None and all(c is not None for c in sm._live_caches)
        sm._caches_to(0, "meta")
        assert idle.rows(0)[0].device.type == "meta"
        fl = br._check().layers[0]
        assert isinstance(fl, ForkLayer) and fl.keys.device.type == "meta" and fl._tk is not None
        assert fl._tk.device.type == "meta"
        assert idle.rows(1)[0].device.type == "cpu"
        br.close()


# -- lifetimes: late, on another thread, through views, out of order -----------------------------------------------


@pytest.mark.parametrize("gone", ["released", "dropped"])
def test_a_rooms_tensor_btb_cannot_see_stays_counted_after_the_room_goes(sm: StreamedTextModel, gone: str) -> None:
    """where btb's free reading cannot see torch's allocations (MLX's ledger) a room's tensor is counted by the room;
    the room released or dropped while the tensor lives, the tensor is still counted until it is gone"""
    base = sm.device.reserved("cpu")
    r = Room(sm.device, "unseen#1", 4 * MiB, torch.device("cpu"), False, sm)
    t = r.empty(MiB, dtype=torch.uint8)
    assert sm.device.reserved("cpu") - base == 4 * MiB
    if gone == "released":
        r.release()
    else:
        del r
        gc.collect()
    assert sm.device.reserved("cpu") - base == MiB, "the tensor outlived its room and went uncounted"
    del t
    gc.collect()
    assert sm.device.reserved("cpu") == base


def let_go(refs: list[torch.Tensor]) -> None:
    """the last references dropped on this thread, and a collection run here: the finalizers run on it"""
    refs.clear()
    gc.collect()


def test_a_rooms_tensor_let_go_on_another_thread_comes_back(sm: StreamedTextModel) -> None:
    with sm.room(4 * MiB) as r:
        held = [r.zeros(MiB, dtype=torch.uint8)]
        view = [held[0][10:]]
        th = threading.Thread(target=let_go, args=(held,))
        th.start()
        th.join()
        assert r.used == MiB, "a view keeps the storage, and the room's count, alive"
        th = threading.Thread(target=let_go, args=(view,))
        th.start()
        th.join()
        assert r.used == 0 and sm.memory()["cpu"].reserved == 4 * MiB
    assert sm.memory()["cpu"].reserved == 0


def test_rooms_and_tensors_asked_for_at_once_add_up(sm: StreamedTextModel) -> None:
    errors: list[BaseException] = []

    def ask() -> None:
        try:
            for _ in range(20):
                with sm.room(MiB) as r:
                    kept = [r.zeros(1024, dtype=torch.uint8), sm.empty(4096, dtype=torch.uint8)]
                    assert r.used == 1024
                    del kept
        except BaseException as e:  # handed to the test's thread
            errors.append(e)

    threads = [threading.Thread(target=ask) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    gc.collect()
    assert not errors, errors[0]
    assert sm.memory()["cpu"].reserved == 0


def test_what_a_refusal_shed_on_the_host_grows_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """a refused tensor gave up every warm layer on the way; with no RAM policy running the lending policy grows
    them back once there is room again"""
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        sm.ram_watch = False
        keep = sm.ram_reserve
        with monkeypatch.context() as mp:
            squeeze(sm, MiB, mp)
            with pytest.raises(MemoryGrantError):
                sm.empty(64 * MiB, dtype=torch.uint8)
            assert sm.cold
        sm.ram_reserve = keep
        assert _until(lambda: not sm.cold, sm), f"still from the drive: {sorted(sm.cold)}"


def test_layers_shed_for_two_tensors_grow_back_whichever_goes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    with loaded_model(fixture("tiny_qwen3"), device="cpu") as sm:
        sm.ram_watch = False
        make = sm._make_room

        def one_step(dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
            assert sm._give_up_one(dev, nbytes, set())
            return make(dev, 0, what, own)

        with monkeypatch.context() as mp:
            mp.setattr(sm, "_make_room", one_step)
            a = sm.empty(MiB, dtype=torch.uint8)
            b = sm.empty(MiB, dtype=torch.uint8)
        assert len(sm.cold) == 2
        del b
        gc.collect()
        del a
        gc.collect()
        assert _until(lambda: not sm.cold, sm), f"still from the drive: {sorted(sm.cold)}"
