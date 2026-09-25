# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Memory for the caller's own tensors beside a model: `empty`/`zeros`/`full` make room before allocating,
`room` holds room for allocations btb does not make, `memory()` says what can be had - and a layer btb moves
takes every live cache's rows with it, an idle session's and a fork's included."""

from __future__ import annotations

import gc
import time
import weakref
from collections.abc import Callable, Iterator

import pytest
import torch

from btb.engine import StreamedTextModel
from btb.engine.cache import ForkLayer
from btb.engine.scheduler import MemoryGrantError
from tests.helpers import fixture, loaded_model, need_cuda, need_mlx

PROMPT = [5, 17, 99, 3, 42, 8, 61, 7, 12, 30]
MiB = 2**20


@pytest.fixture(params=["cpu", "mlx"])
def sm(request: pytest.FixtureRequest) -> Iterator[StreamedTextModel]:
    if request.param == "mlx":
        need_mlx()
    with loaded_model(fixture("tiny_qwen3"), device=request.param) as m:
        yield m


def squeeze(sm: StreamedTextModel, left: int) -> None:
    """leave `left` bytes free on the host, as btb counts it"""
    if sm.mlx is not None:
        sm.ram_reserve = int(sm.mem_start) - int(sm.mlx.held_bytes()) - left
    else:
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


def test_a_refusal_names_the_request_and_leaves_the_model_answering(sm: StreamedTextModel) -> None:
    """asked for more than btb can give, it gives up what it holds - warm layers to the drive - and refuses,
    naming the request; the model decodes the same tokens from the drive"""
    ref = list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens)
    squeeze(sm, MiB)
    with pytest.raises(MemoryGrantError, match=r"shape \[67108864\].*64\.0 MiB"):
        sm.empty(64 * MiB, dtype=torch.uint8)
    assert sm.cold == set(range(sm.L))
    with pytest.raises(MemoryGrantError, match="room 'workspace'"):
        sm.room(64 * MiB, name="workspace")
    assert list(sm.generate(PROMPT, 6, eos=(), speculate=False).tokens) == ref


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


def test_what_a_refusal_shed_grows_back_once_there_is_room() -> None:
    """MLX runs no host memory policy, so the layers making room shed are grown back by the lending policy"""
    need_mlx()
    with loaded_model(fixture("tiny_qwen3"), device="mlx") as sm:
        keep = sm.ram_reserve
        squeeze(sm, MiB)
        with pytest.raises(MemoryGrantError):
            sm.empty(64 * MiB, dtype=torch.uint8)
        assert sm.cold
        sm.ram_reserve = keep
        assert _until(lambda: not sm.cold, sm), f"still from the drive: {sorted(sm.cold)}"


def test_a_card_gives_layers_up_for_a_tensor_and_every_cache_follows() -> None:
    """a squeezed card sheds layers for a tensor; an idle session's rows follow each layer to the host (its next
    turn runs, no device mismatch); once the tensor is gone the layers grow back with no memory policy running"""
    dev = need_cuda()
    with loaded_model(fixture("tiny_qwen3"), device=dev, vram_watch=False) as sm:
        idle = sm.session(PROMPT)
        card = str(sm.dev)
        before, margin = set(sm.resident), sm.vram_margin
        sm.vram_margin += max(0, sm.memory()[card].free - MiB)
        want = sm.memory()[card].sheddable // 2
        t = sm.empty(want, dtype=torch.uint8)
        assert t.device.type == "cuda" and set(sm.resident) < before
        for i in before - set(sm.resident):
            assert idle.rows(i)[0].device.type == "cpu", i
        assert len(idle.generate(2, eos=(), speculate=False).tokens) == 2
        del t
        gc.collect()
        sm.vram_margin = margin
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
