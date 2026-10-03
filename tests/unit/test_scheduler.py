# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
# mypy: check-untyped-defs=false
"""The batch scheduler and the two consumers that live off it: `generate_greedy`'s epoch split and `serve`'s
ragged queue, plus the VRAM policy's batched-decode gate. Everything here is device-free and model-free - a
stub engine carries a toy model whose answer is a pure function of a row's real (unpadded) prompt, so a row
decoded alone and the same row decoded inside an epoch must agree token for token, and the CUDA memory
readings are three patched functions. The suite runs in seconds on any machine, with or without a card."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from pytest import MonkeyPatch

from btb.engine import BatchScheduler
from btb.engine import device as device_mod
from btb.engine import memory as memory_mod
from btb.engine.device import Device, Where
from btb.engine.families import Family
from btb.engine.forward import _ForwardMixin
from btb.engine.generate import _GenerateMixin
from btb.engine.memory import RamPolicyState, VramPolicyState, _MemoryMixin
from btb.engine.scheduler import MemoryGrantError
from btb.kinds import FamilyKind, Json, LayerKind, Log, TokenRows, Tokens
from tests.helpers import (
    ABSENT,
    GB,
    KB,
    MB,
    FakeRoute,
    MlxLedger,
    SchedulerModel,
    bare_registry,
    expert_store,
    model_config,
    seat,
    slot_size,
    stub_engine,
    stub_ledger,
)

if TYPE_CHECKING:
    from btb.engine.experts import _ExpertStore
    from btb.serve import Engine

TOTAL_VRAM = 12 * GB


# --- the stubs ----------------------------------------------------------------------------------------------


def _where(dev: str) -> Where:
    """`dev` as the engine names its device (`device.where`): a card always by its index - the card 0, no card here
    to ask which is current"""
    d = torch.device(dev)
    return Where(torch.device("cuda", 0) if d.type == "cuda" and d.index is None else d)


@contextlib.contextmanager
def cuda_stats(
    free: int | list[int] = 8 * GB,
    reserved: int = 0,
    allocated: int = 0,
    raises: BaseException | None = None,
    physical: int | None = 1 << 62,
    budget: int | None = None,
) -> Iterator[dict[str, int]]:
    """`torch.cuda`'s three memory readings, patched, plus the cross-process physical free (`free_bytes` clamps
    the per-process reading to it; a huge default makes the clamp a no-op so these price the mem_get_info
    arithmetic alone) and the WDDM budget's room (None: no budget, as off Windows). `free` may be a list: one
    reading per call, the last repeating, staging a run whose free memory moves between plans."""
    saved = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
    saved_phys, saved_budget = device_mod._physical_free_bytes, device_mod._wddm_room
    seen = {"mem_get_info": 0}
    seq = list(free) if isinstance(free, (list, tuple)) else None

    def mem_get_info(*a: object, **k: object) -> tuple[int, int]:
        seen["mem_get_info"] += 1
        if raises is not None:
            raise raises
        f = seq[min(seen["mem_get_info"] - 1, len(seq) - 1)] if seq is not None else free
        assert not isinstance(f, list)  # a list free set seq, so f is one of its ints
        return (int(f), TOTAL_VRAM)

    torch.cuda.mem_get_info = mem_get_info
    torch.cuda.memory_reserved = lambda *a, **k: int(reserved)
    torch.cuda.memory_allocated = lambda *a, **k: int(allocated)
    device_mod._physical_free_bytes = lambda dev: None if physical is None else int(physical)
    device_mod._wddm_room = lambda dev: None if budget is None else int(budget)
    try:
        yield seen
    finally:
        torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = saved
        device_mod._physical_free_bytes, device_mod._wddm_room = saved_phys, saved_budget


V = 64  # the toy vocabulary


def toy_tokens(prompt: Sequence[int], n: int) -> list[int]:
    """What the stub engine answers a row whose real prompt is `prompt`: a pure function of the prompt's
    tokens and the step, so padding, batching and epoch splitting cannot change it."""
    s = 0
    for t in prompt:
        s = (s * 131 + int(t) + 1) % 1000003
    return [int((s * 7919 * (k + 1) + 13) % V) for k in range(n)]


class _StubCache:
    def __init__(self, max_len: int | None = None) -> None:
        self.max_len = max_len
        self.step = 0
        self.rows: list[list[int]] = []
        self.layers = [types.SimpleNamespace(keys=None, values=None)]


class _StubEngine(_GenerateMixin):
    """the decode loop of the real engine over a toy forward: every call it makes is recorded"""

    pad_left = staticmethod(_ForwardMixin.pad_left)

    def __init__(
        self,
        dev: str = "cuda",
        layer_types: Sequence[str] = ("full_attention",) * 4,
        vram_margin: int = 0,
        cfg: types.SimpleNamespace | None = None,
    ) -> None:
        self.cfg = model_config() if cfg is None else cfg
        self.layer_types = list(layer_types)
        self.compute_dtype = None
        self.kv_bits = None
        self.dev = _where(dev)
        self.mlx = None
        self.mem_start = 0
        self.ram_reserve = 0
        self.vram_margin = int(vram_margin)
        self.vram_watch = False
        self.host = {}
        self.abort = threading.Event()
        self._in_epoch = False
        self.lines: list[str] = []
        self.prefills: list[Json] = []  # one record per prefill: the shape, the mask, the rows it decoded
        self.policy_calls = 0
        self.scheduler = BatchScheduler(self)

    # -- what the engine provides and the decode loop calls --
    def log(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    def vram_trim(self, tag: str = "") -> tuple[float, float]:
        return 0.0, 0.0

    def vram_policy(self, cache: Any = None, log: Log | None = None) -> None:
        self.policy_calls += 1

    def ram_policy(self, log: Log | None = None) -> None:
        pass

    def _mlx_greedy_ok(self, B: int, attention_mask: torch.Tensor | None, on_layer: Any, prefill_only: bool) -> bool:
        return False

    def _mlx_batch_ok(self, B: int, on_layer: Any, prefill_only: bool) -> bool:
        return False

    def new_cache(self, max_len: int | None = None) -> _StubCache:
        return _StubCache(max_len)

    def _logits(self, rows: TokenRows, k: int) -> torch.Tensor:
        out = torch.zeros(len(rows), 1, V)
        for b, r in enumerate(rows):
            out[b, 0, toy_tokens(r, k + 1)[k]] = 1.0
        return out

    def _prefill(
        self,
        ids: torch.Tensor,
        cache: Any,
        on_layer: Any = None,
        attention_mask: torch.Tensor | None = None,
        last_only: bool = True,
    ) -> torch.Tensor:
        ids = torch.as_tensor(ids, dtype=torch.long)
        B, T = int(ids.shape[0]), int(ids.shape[1])
        rows = []
        for b in range(B):
            row = ids[b].tolist()
            if attention_mask is not None:
                row = [int(t) for t, m in zip(row, attention_mask[b].tolist(), strict=True) if m]
            rows.append([int(t) for t in row])
        cache.rows, cache.step = rows, 0
        self.prefills.append(
            {
                "B": B,
                "T": T,
                "mask": attention_mask is not None,
                "rows": rows,
                "ids": ids.clone(),
                "max_len": cache.max_len,
            }
        )
        return self._logits(rows, 0)

    def forward(  # type: ignore[override]
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        cache: Any = None,
        on_layer: Callable[[int, torch.Tensor], object] | None = None,
        last_only: bool = True,
        attention_mask: torch.Tensor | None = None,
        **k: object,
    ) -> torch.Tensor:
        ids = torch.as_tensor(ids, dtype=torch.long)
        assert int(ids.shape[0]) == len(cache.rows), "the batch never resizes mid-decode"
        cache.step += 1
        return self._logits(cache.rows, cache.step)


def test_the_epochs_kv_on_a_card_is_only_the_resident_layers() -> None:
    """the epoch's KV is reserved on the card, so it is priced for the layers whose rows live there: a host layer's
    rows are on the host, and `kv_host` keeps even a resident layer's there. Priced for every layer it was a
    phantom reservation that shrank the depot and the chunk on a card holding a few layers"""
    m = SchedulerModel(
        dev="cuda", layer_types=("full_attention", "linear_attention", "full_attention", "full_attention")
    )
    every = BatchScheduler(m)._kv_bytes_per_row_token()
    m.resident = {2: None}  # type: ignore[attr-defined]
    assert BatchScheduler(m)._kv_bytes_per_row_token() * 3 == every, "one of the three attention layers on the card"
    m.kv_host = True  # type: ignore[attr-defined]
    assert BatchScheduler(m)._kv_bytes_per_row_token() == 0, "every layer's rows on the host"
    host = SchedulerModel(dev="cpu", layer_types=m.layer_types)
    host.resident = {2: None}  # type: ignore[attr-defined]
    assert BatchScheduler(host)._kv_bytes_per_row_token() == every, "one device: every layer's rows are its own"


def test_a_batchs_host_rows_count_what_the_store_would_give_back() -> None:
    """on a card whose attention rows live on the host, a batch is held to the host's room for them: its free memory
    and what the expert store - a cache grown into free RAM, which gives blocks back as rows grow - would release.
    A warm store at its margin no longer sizes every epoch at one row; and the epoch reserves those rows there"""
    from btb.engine.scheduler import EPOCH

    m = SchedulerModel(dev="cuda", layer_types=("full_attention",) * 4)
    m.resident = {}  # type: ignore[attr-defined]  # every attention layer's rows on the host, float32
    reserved: dict[tuple[str, str], int] = {}

    class _Ledger:
        def free(self, device: Any = None, unreserved: bool = False, own: Any = None, pooled: bool = False) -> int:
            return 1 * GB if device is not None and torch.device(device).type == "cpu" else 64 * GB

        def reserve(self, tag: str, nbytes: int, device: Any = None, used: Any = None) -> None:
            reserved[(tag, "cpu" if device is not None else "card")] = int(nbytes)

    m.device = _Ledger()  # type: ignore[attr-defined]
    s = BatchScheduler(m)
    host = s.host_kv_row_bytes(4096)
    assert host > 0 and s.kv_row_bytes(4096) == 0, "no row on the card, every one on the host"
    assert s.max_batch(4096) == GB // host, "the store's margin alone"
    m.expert_store = types.SimpleNamespace(releasable=lambda: 8 * GB)  # type: ignore[attr-defined]
    assert s.max_batch(4096) == 9 * GB // host, "and what the store would give back for them"
    batch, _ = s.plan(10**6, 4096)
    assert reserved[(EPOCH, "cpu")] == host * batch, "the host's rows reserved on the host"


def _counting_max_batch(engine: _StubEngine, values: Sequence[int]) -> dict[str, int]:
    """replace the scheduler's max_batch with one that returns `values` in order (the last repeating) and
    counts how often it was consulted - the `_in_epoch` guard says: once per split"""
    seen = {"n": 0}

    def mb(target_len: int) -> int:
        seen["n"] += 1
        return values[min(seen["n"] - 1, len(values) - 1)]

    engine.scheduler.max_batch = mb  # type: ignore[method-assign]
    return seen


def _recording_plan(engine: _StubEngine) -> list[Json]:
    """wrap the real plan, keeping the (n_pending, target_len) -> (batch, reserve) it was asked and answered"""
    real = engine.scheduler.plan
    seen: list[Json] = []

    def plan(n_pending: int, target_len: int) -> tuple[int, int]:
        out = real(n_pending, target_len)
        seen.append({"n_pending": n_pending, "target_len": target_len, "batch": out[0], "reserve": out[1]})
        return out

    engine.scheduler.plan = plan  # type: ignore[method-assign]
    return seen


# --- the KV arithmetic ---------------------------------------------------------------------------------------


def test_kv_bytes_are_k_and_v_over_every_attention_layer() -> None:
    """guards the row cost: 2 (k and v) x layers x kv heads x head dim x element size, and its linearity"""
    s = BatchScheduler(SchedulerModel(layer_types=("full_attention",) * 4))
    assert s._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2
    assert s.kv_row_bytes(1000) == 1000 * (2 * 4 * 2 * 64 * 2)


def test_kv_bytes_use_the_query_heads_when_there_is_no_grouping() -> None:
    """guards MHA: a config with no num_key_value_heads (or None) caches one k/v per query head, not per group"""
    gqa = BatchScheduler(SchedulerModel(cfg=model_config(num_attention_heads=8, num_key_value_heads=2)))
    mha_absent = BatchScheduler(SchedulerModel(cfg=model_config(num_attention_heads=8, num_key_value_heads=ABSENT)))
    mha_none = BatchScheduler(SchedulerModel(cfg=model_config(num_attention_heads=8, num_key_value_heads=None)))
    assert mha_absent._kv_bytes_per_row_token() == 2 * 4 * 8 * 64 * 2
    assert mha_none._kv_bytes_per_row_token() == mha_absent._kv_bytes_per_row_token()
    assert gqa._kv_bytes_per_row_token() * 4 == mha_absent._kv_bytes_per_row_token(), "8 heads over 2 = 4x the KV"


def test_kv_bytes_derive_the_head_dim_from_the_hidden_size() -> None:
    """guards the fallback for a config that does not state head_dim: hidden_size // query heads"""
    derived = BatchScheduler(SchedulerModel(cfg=model_config(head_dim=ABSENT, hidden_size=512, num_attention_heads=8)))
    assert derived._kv_bytes_per_row_token() == 2 * 4 * 2 * (512 // 8) * 2
    zero = BatchScheduler(SchedulerModel(cfg=model_config(head_dim=0, hidden_size=512, num_attention_heads=8)))
    assert zero._kv_bytes_per_row_token() == derived._kv_bytes_per_row_token(), "a 0 head_dim falls back too"
    stated = BatchScheduler(SchedulerModel(cfg=model_config(head_dim=128, hidden_size=512, num_attention_heads=8)))
    assert stated._kv_bytes_per_row_token() == 2 * 4 * 2 * 128 * 2, "a stated head_dim wins over the division"


def test_kv_bytes_exclude_the_linear_attention_layers() -> None:
    """guards the hybrid: a DeltaNet layer holds a fixed recurrent state, not a per-position cache, so it must
    not be charged per token - counting it would undersize every batch on a hybrid model"""
    hybrid = BatchScheduler(
        SchedulerModel(layer_types=("full_attention", "linear_attention", "linear_attention", "full_attention"))
    )
    dense = BatchScheduler(SchedulerModel(layer_types=("full_attention", "full_attention")))
    assert hybrid._kv_bytes_per_row_token() == dense._kv_bytes_per_row_token() == 2 * 2 * 2 * 64 * 2
    mixed = BatchScheduler(SchedulerModel(layer_types=("full_attention", "linear_attention", "sliding_attention")))
    assert mixed._kv_bytes_per_row_token() == 2 * 2 * 2 * 64 * 2, "only 'linear_attention' is free of a cache"


def test_kv_bytes_of_an_all_linear_model_do_not_divide_by_zero() -> None:
    """guards the degenerate model: no attention layer at all -> a zero row cost. max_batch must not divide by
    zero, and the plan it gives must be sane (every pending row, since no KV binds the batch)."""
    s = BatchScheduler(SchedulerModel(dev="cuda", layer_types=("linear_attention",) * 8))
    assert s._kv_bytes_per_row_token() == 0
    assert s.kv_row_bytes(4096) == 0
    with cuda_stats(free=4 * GB):
        assert s.max_batch(4096) == 4 * GB, "the guard is max(1, free // max(1, 0)): free bytes, not a crash"
        assert s.plan(37, 4096) == (37, 4096), "every pending row: nothing about KV holds the batch back"


def test_kv_bytes_follow_the_compute_dtype() -> None:
    """guards the element size: an unset compute_dtype means bf16 (the engine's default), not float32"""
    assert BatchScheduler(SchedulerModel(compute_dtype=None))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2
    assert BatchScheduler(SchedulerModel(compute_dtype=torch.bfloat16))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2
    assert BatchScheduler(SchedulerModel(compute_dtype=torch.float16))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2
    assert BatchScheduler(SchedulerModel(compute_dtype=torch.float32))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 4


def test_kv_bytes_of_an_int8_cache_are_one_byte_a_value() -> None:
    """guards the quantized cache: kv_bits 8 is one byte a value whatever the compute dtype; anything else
    (None, and any other width) is charged at the dtype's size"""
    assert BatchScheduler(SchedulerModel(kv_bits=8))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 1
    fp32_int8 = BatchScheduler(SchedulerModel(kv_bits=8, compute_dtype=torch.float32))
    assert fp32_int8._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 1, "kv_bits wins over the compute dtype"
    assert BatchScheduler(SchedulerModel(kv_bits=None))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2
    assert BatchScheduler(SchedulerModel(kv_bits=16))._kv_bytes_per_row_token() == 2 * 4 * 2 * 64 * 2


def test_kv_row_bytes_charge_at_least_one_position() -> None:
    """guards the floor: a zero or negative target length is one position, never zero bytes (which would let
    max_batch answer 'the whole free memory' for a real cache)"""
    s = BatchScheduler(SchedulerModel())
    one = s._kv_bytes_per_row_token()
    assert s.kv_row_bytes(0) == one
    assert s.kv_row_bytes(-5) == one
    assert s.kv_row_bytes(1) == one
    assert s.kv_row_bytes(2) == 2 * one


# --- free_vram -------------------------------------------------------------------------------------------


def test_free_vram_counts_torchs_reclaimable_pool() -> None:
    """guards the reclaimable term: torch's reserved-but-unallocated blocks are available to the next
    allocation and never reach the driver, so leaving them out undercounts and sizes a tiny epoch"""
    s = BatchScheduler(SchedulerModel(dev="cuda", vram_margin=1 * GB))
    with cuda_stats(free=4 * GB, reserved=3 * GB, allocated=1 * GB):
        assert s.free_vram() == 4 * GB + 2 * GB - 1 * GB


def test_free_vram_with_nothing_reclaimable_is_the_driver_reading_less_the_margin() -> None:
    s = BatchScheduler(SchedulerModel(dev="cuda", vram_margin=512 * MB))
    with cuda_stats(free=4 * GB, reserved=2 * GB, allocated=2 * GB):
        assert s.free_vram() == 4 * GB - 512 * MB
    with cuda_stats(free=4 * GB, reserved=0, allocated=0):
        assert s.free_vram() == 4 * GB - 512 * MB


def test_free_vram_clamps_at_zero_when_the_margin_is_larger_than_free() -> None:
    """guards the clamp: a card already inside its margin reports 0 free, never a negative that would make
    max_batch's floor division give a negative batch"""
    s = BatchScheduler(SchedulerModel(dev="cuda", vram_margin=6 * GB))
    with cuda_stats(free=1 * GB, reserved=0, allocated=0):
        assert s.free_vram() == 0
        assert s.max_batch(1024) == 1, "the floor still offers one row"
        assert s.plan(8, 1024) == (1, 1024)


def test_free_vram_on_unified_memory_is_the_engines_own_ledger() -> None:
    """guards the MLX arithmetic: mem_start - ram_reserve - held, not the OS free count (a Metal buffer reads
    free until its pages are faulted), and its clamp when MLX holds more than the ledger allows"""
    s = BatchScheduler(SchedulerModel(dev="cpu", mlx=MlxLedger(10 * GB), mem_start=36 * GB, ram_reserve=4 * GB))
    assert s.free_vram() == 22 * GB
    tight = BatchScheduler(SchedulerModel(dev="cpu", mlx=MlxLedger(40 * GB), mem_start=36 * GB, ram_reserve=4 * GB))
    assert tight.free_vram() == 0, "held past the ledger clamps to 0, it does not go negative"
    assert tight.max_batch(1024) == 1
    empty = BatchScheduler(SchedulerModel(dev="cpu", mlx=MlxLedger(0), mem_start=8 * GB, ram_reserve=0))
    assert empty.free_vram() == 8 * GB


def test_free_vram_is_none_on_the_host_tier() -> None:
    """guards the host tier's opt-out: bound by CPU throughput, not KV memory, so it does not size a batch by
    memory at all - None all the way through max_batch, and plan takes every pending row"""
    s = BatchScheduler(SchedulerModel(dev="cpu"))
    assert s.free_vram() is None
    assert s.max_batch(1_000_000) is None
    assert s.plan(37, 200) == (37, 200)
    assert (s.batch, s.reserve) == (37, 200)


def test_free_vram_lets_a_driver_error_surface() -> None:
    """The scheduler does not catch mem_get_info: a driver that cannot answer is not 'no memory free' (which
    would silently decode one row at a time forever), it is a broken context, and it must reach the caller.
    This test pins that decision - if a future patch wants a fallback, it has to change this line on purpose."""
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    boom = RuntimeError("CUDA driver error: invalid device context")
    with cuda_stats(raises=boom):
        with pytest.raises(RuntimeError) as e:
            s.free_vram()
        assert e.value is boom
        with pytest.raises(RuntimeError):
            s.max_batch(1024)
        with pytest.raises(RuntimeError):
            s.plan(4, 1024)
    assert (s.batch, s.reserve) == (None, None), "a failed plan leaves no half-set epoch size behind"


# --- max_batch and plan ------------------------------------------------------------------------------------


def test_max_batch_is_the_free_memory_divided_by_one_rows_kv() -> None:
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    row = s.kv_row_bytes(1024)
    with cuda_stats(free=10 * row, reserved=0, allocated=0):
        assert s.max_batch(1024) == 10
    with cuda_stats(free=10 * row + row // 2, reserved=0, allocated=0):
        assert s.max_batch(1024) == 10, "a partial row is not a row"


def test_max_batch_at_exactly_one_row_and_one_byte_short() -> None:
    """guards the boundary and the floor: one row exactly is one row; a byte short is still one row, because
    the scheduler always offers a single sequence rather than refusing to decode"""
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    row = s.kv_row_bytes(2048)
    with cuda_stats(free=row, reserved=0, allocated=0):
        assert s.max_batch(2048) == 1
    with cuda_stats(free=row - 1, reserved=0, allocated=0):
        assert s.max_batch(2048) == 1, "the floor is 1: a row that does not fit is still attempted"
    with cuda_stats(free=0, reserved=0, allocated=0):
        assert s.max_batch(2048) == 1
    with cuda_stats(free=2 * row - 1, reserved=0, allocated=0):
        assert s.max_batch(2048) == 1


def test_max_batch_over_the_target_length_edges() -> None:
    """guards the length axis: 0 and 1 cost the same (one position), and a huge length shrinks the batch to
    the floor instead of overflowing or reaching zero"""
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    with cuda_stats(free=1 * GB, reserved=0, allocated=0):
        assert s.max_batch(0) == s.max_batch(1) == 1 * GB // s.kv_row_bytes(1)
        assert s.max_batch(-7) == s.max_batch(1), "a negative length is one position, not a negative row"
        assert s.max_batch(1 << 40) == 1
        assert s.max_batch(1024) == 1 * GB // (1024 * s._kv_bytes_per_row_token())


def test_plan_is_capped_by_the_pending_rows_and_is_sticky() -> None:
    """guards the epoch handle: plan answers min(pending, what fits) and leaves the size on the instance, which
    is what the run reports and what a later assertion about 'the batch never resized' reads"""
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    row = s.kv_row_bytes(512)
    with cuda_stats(free=8 * row, reserved=0, allocated=0):
        assert s.plan(3, 512) == (3, 512), "fewer pending than fit: the pending count wins"
        assert (s.batch, s.reserve) == (3, 512)
        assert s.plan(100, 512) == (8, 512), "more pending than fit: the memory wins"
        assert (s.batch, s.reserve) == (8, 512)
        assert s.plan(1, 512) == (1, 512)
        assert (s.batch, s.reserve) == (1, 512)


def test_plan_with_no_pending_rows_differs_on_and_off_the_card() -> None:
    """Pins an asymmetry: off the card plan(0) is 0, on the card the max(1, ...) floor makes it 1. `serve`
    never asks with an empty queue, so this is a documented shape, not a live bug - but a future caller that
    loops on plan()'s answer would spin on the card and not off it."""
    off = BatchScheduler(SchedulerModel(dev="cpu"))
    assert off.plan(0, 128) == (0, 128)
    on = BatchScheduler(SchedulerModel(dev="cuda"))
    with cuda_stats(free=8 * GB, reserved=0, allocated=0):
        assert on.plan(0, 128) == (1, 128)


def test_plan_takes_a_target_length_under_a_prompts_own_length() -> None:
    """guards the arithmetic under a caller that reserves less than a prompt holds (the scheduler reserves what
    it is told; it does not silently grow the reservation to the prompt)"""
    s = BatchScheduler(SchedulerModel(dev="cuda"))
    with cuda_stats(free=1 * GB, reserved=0, allocated=0):
        b, r = s.plan(64, 8)
        assert r == 8 and b == min(64, 1 * GB // s.kv_row_bytes(8))
        assert s.plan(64, 8.9)[1] == 8, "the target length is taken as an int"  # type: ignore[arg-type]


# --- the epoch split in generate_greedy ---------------------------------------------------------------------


def test_greedy_runs_fixed_epochs_and_sizes_them_once() -> None:
    """guards the `_in_epoch` guard: `mb` is computed ONCE for the whole split. A sub-call that re-sized itself
    off the (now shifted) free memory would drift the epoch size and re-shed every layer's KV buffer."""
    e = _StubEngine(dev="cuda")
    seen = _counting_max_batch(e, [2, 1, 7, 99])  # only the first must ever be read
    prompts = [[10 + i, 20 + i, 30 + i] for i in range(5)]
    ids = torch.tensor(prompts, dtype=torch.long)
    out = e.generate_greedy(ids, 4, eos_ids=())
    assert seen["n"] == 1, "max_batch was consulted once for the split, never again inside it"
    assert [p["B"] for p in e.prefills] == [2, 2, 1], "fixed epochs of 2, the remainder alone"
    assert len(out) == 5
    for r, p in zip(out, prompts, strict=True):
        assert r == toy_tokens(p, 4)


def test_greedy_epoch_rows_equal_the_single_row_decode() -> None:
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [3])
    prompts = [[7, 8, 9], [1, 2, 3], [4, 5, 6], [9, 9, 9], [2, 4, 8], [0, 1, 0], [5, 5, 5]]
    batched = e.generate_greedy(torch.tensor(prompts, dtype=torch.long), 5, eos_ids=())
    singles = []
    for p in prompts:
        solo = _StubEngine(dev="cuda")
        _counting_max_batch(solo, [3])
        singles.append(solo.generate_greedy([p], 5, eos_ids=()))
    assert batched == singles, "an epoch changes nothing about a row's answer"


def test_greedy_reads_the_free_memory_once_for_the_whole_split() -> None:
    """the same guard through the real scheduler: the driver reading falls to nothing after the first plan,
    and the epochs still run at the size the first reading gave. A second reading would give epochs of 1."""
    e = _StubEngine(dev="cuda")
    row = e.scheduler.kv_row_bytes(3 + 2)
    prompts = [[i, i + 1, i + 2] for i in range(5)]
    with cuda_stats(free=[2 * row, 0], reserved=0, allocated=0) as seen:
        out = e.generate_greedy(torch.tensor(prompts, dtype=torch.long), 2, eos_ids=())
    assert seen["mem_get_info"] == 1, "one memory reading sized the whole split"
    assert [p["B"] for p in e.prefills] == [2, 2, 1]
    assert out == [toy_tokens(p, 2) for p in prompts]


def test_greedy_clears_the_epoch_guard_after_a_split() -> None:
    """guards the finally: `_in_epoch` is dropped when the split ends, so the next top-level call sizes itself
    again. A leaked flag would make every later batch skip the scheduler and OOM at the first big one."""
    e = _StubEngine(dev="cuda")
    seen = _counting_max_batch(e, [2])
    e.generate_greedy(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long), 2, eos_ids=())
    assert e._in_epoch is False
    assert seen["n"] == 1
    e.generate_greedy(torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.long), 2, eos_ids=())
    assert seen["n"] == 2, "the next batch was sized on its own"


def test_greedy_does_not_split_a_batch_that_fits_exactly() -> None:
    """the boundary: the split runs only when max_batch is strictly under B, so a batch that fits exactly is
    one epoch (an off-by-one there would halve every full batch)"""
    e = _StubEngine(dev="cuda")
    seen = _counting_max_batch(e, [3])
    out = e.generate_greedy(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long), 3, eos_ids=())
    assert seen["n"] == 1 and [p["B"] for p in e.prefills] == [3], "one batch, one prefill"
    assert len(out) == 3
    roomy = _StubEngine(dev="cuda")
    _counting_max_batch(roomy, [8])
    roomy.generate_greedy(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long), 3, eos_ids=())
    assert [p["B"] for p in roomy.prefills] == [3]


def test_greedy_of_one_row_never_asks_the_scheduler() -> None:
    """a single sequence is not a batch: the split is a B > 1 guard, so a one-row decode costs no memory read"""
    e = _StubEngine(dev="cuda")
    seen = _counting_max_batch(e, [1])
    out = e.generate_greedy([[4, 5, 6]], 3, eos_ids=())
    assert seen["n"] == 0
    assert out == toy_tokens([4, 5, 6], 3), "one row answers as a flat token list, not a list of rows"


def test_greedy_never_splits_off_the_card() -> None:
    """the split is a CUDA-only guard: the host tier decodes the whole batch, max_batch is not consulted"""
    e = _StubEngine(dev="cpu")
    seen = _counting_max_batch(e, [1])
    out = e.generate_greedy(torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long), 2, eos_ids=())
    assert seen["n"] == 0 and [p["B"] for p in e.prefills] == [4]
    assert len(out) == 4


def test_greedy_splits_a_huge_batch_into_many_single_row_epochs() -> None:
    """guards the extreme: 64 rows and room for one - 64 epochs, every row still its own single-row answer,
    and the B == 1 sub-call's flat list wrapped back into a row (not spliced token by token)"""
    e = _StubEngine(dev="cuda")
    seen = _counting_max_batch(e, [1])
    prompts = [[i, i + 1] for i in range(64)]
    out = e.generate_greedy(torch.tensor(prompts, dtype=torch.long), 3, eos_ids=())
    assert seen["n"] == 1
    assert len(e.prefills) == 64 and {p["B"] for p in e.prefills} == {1}
    assert len(out) == 64 and all(isinstance(r, list) and len(r) == 3 for r in out)
    assert out == [toy_tokens(p, 3) for p in prompts]


def test_greedy_epochs_carry_each_rows_own_mask_slice() -> None:
    """guards the ragged split: the attention mask is sliced with the rows, so a padded row keeps its own real
    tokens. A mis-sliced mask would give a row another row's padding and a different answer."""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    prompts = [[1, 2, 3, 4], [5, 6], [7], [8, 9, 10]]
    ids, mask = e.pad_left(prompts, pad_id=0)
    out = e.generate_greedy(ids, 3, eos_ids=(), attention_mask=mask)
    assert [p["B"] for p in e.prefills] == [2, 2]
    assert [r for p in e.prefills for r in p["rows"]] == prompts, "every epoch saw its own rows unpadded"
    assert out == [toy_tokens(p, 3) for p in prompts]


def test_greedy_streams_tokens_only_from_the_first_epoch() -> None:
    """guards the on_token contract under a split: the callback is a single stream, so it follows row 0 of the
    first epoch only - a callback fired from every epoch would interleave several answers into one stream"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    got: list[int] = []
    prompts = [[1, 1], [2, 2], [3, 3], [4, 4]]
    e.generate_greedy(torch.tensor(prompts, dtype=torch.long), 3, eos_ids=(), on_token=got.append)
    assert got == toy_tokens(prompts[0], 3)


def test_greedy_stops_a_row_at_its_eos_inside_an_epoch() -> None:
    """guards per-row stopping inside a fixed batch: one row's eos ends that row, the others run on"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [3])
    prompts = [[1, 2], [3, 4], [5, 6]]
    eos = toy_tokens(prompts[1], 1)[0]  # row 1 stops on its very first token
    out = e.generate_greedy(torch.tensor(prompts, dtype=torch.long), 4, eos_ids=(eos,))
    assert out[1] == [eos], "an eos on the first step is the whole answer"
    for b in (0, 2):
        expect = toy_tokens(prompts[b], 4)
        cut = next((i + 1 for i, t in enumerate(expect) if t == eos), len(expect))
        assert out[b] == expect[:cut]


def test_greedy_with_no_new_tokens_returns_empty_rows() -> None:
    """guards max_new 0: a prefill happens, no step does, and every row is an empty list (not a missing row)"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    out = e.generate_greedy(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long), 0, eos_ids=())
    assert out == [[], [], []]
    solo = _StubEngine(dev="cuda")
    _counting_max_batch(solo, [2])
    assert solo.generate_greedy([[1, 2]], 0, eos_ids=()) == []


def test_greedy_reserves_the_prompt_plus_the_new_tokens() -> None:
    """guards the target length the epoch is sized and the cache built for: prompt + max_new, per sub-call"""
    e = _StubEngine(dev="cuda")
    lens = []

    def mb(target_len: int) -> int:
        lens.append(target_len)
        return 2

    e.scheduler.max_batch = mb  # type: ignore[method-assign]
    e.generate_greedy(torch.tensor([[1, 2, 3]] * 3, dtype=torch.long), 7, eos_ids=())
    assert lens == [3 + 7], "sized once, at the prompt's width plus the new tokens"
    assert {p["max_len"] for p in e.prefills} == {10}, "every epoch's cache is built for the same length"


# --- serve ---------------------------------------------------------------------------------------------------


def test_serve_returns_one_list_per_prompt_in_input_order() -> None:
    """guards the queue's bookkeeping: results land at their input index whatever epoch decoded them"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    prompts = [[1, 2, 3, 4, 5], [6], [7, 8], [9, 10, 11], [12]]
    out = e.serve(prompts, 4, eos_ids=())
    assert len(out) == len(prompts)
    assert out == [toy_tokens(p, 4) for p in prompts]


def test_serve_matches_the_single_row_decode_of_every_prompt() -> None:
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [3])
    prompts = [[5], [1, 2, 3], [4, 4, 4, 4], [7, 7], [8], [2, 9]]
    served = e.serve(prompts, 3, eos_ids=())
    for p, r in zip(prompts, served, strict=True):
        solo = _StubEngine(dev="cuda")
        _counting_max_batch(solo, [3])
        assert r == solo.generate_greedy([p], 3, eos_ids=())


def test_serve_sizes_each_epoch_for_the_longest_still_waiting() -> None:
    """guards the reserve: an epoch is planned at (longest prompt still queued + max_new), so a later epoch of
    short prompts reserves less than the first. A reserve taken from the epoch's own rows would under-reserve
    when a longer prompt is still ahead of it in the queue."""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    seen = _recording_plan(e)
    prompts = [[1] * 3, [2] * 4, [3] * 9, [4] * 2, [5] * 1]
    e.serve(prompts, 6, eos_ids=())
    # epoch 0 takes the 3 and the 4 but reserves for the 9 still queued; epoch 1 takes the 9 and the 2 and
    # still reserves for the 9; epoch 2 is the lone 1-token prompt, and only then does the reserve fall
    assert [s["target_len"] for s in seen] == [9 + 6, 9 + 6, 1 + 6], "the longest of what is still waiting"
    assert [s["n_pending"] for s in seen] == [5, 3, 1]
    assert [s["batch"] for s in seen] == [2, 2, 1]


def test_serve_reserves_for_a_prompt_that_is_not_in_this_epoch() -> None:
    """Pins a conservative shape: the reserve is the longest of everything still queued, not the longest of the
    rows this epoch actually takes. Epoch 0 here reserves for a 40-token prompt it will not decode until epoch
    1 - it over-reserves (a smaller batch than the memory would allow), it never under-reserves."""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [1])
    seen = _recording_plan(e)
    e.serve([[1, 2], [3] * 40], 5, eos_ids=())
    assert seen[0]["target_len"] == 40 + 5, "epoch 0 decodes the 2-token prompt but reserves for the 40"
    assert seen[1]["target_len"] == 40 + 5


def test_serve_sends_a_lone_leftover_down_the_single_row_path() -> None:
    """guards the leftover: one prompt is decoded unpadded and unmasked (no pad row to attend around), and its
    answer is still returned as a list of tokens, not spliced into the results"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    prompts = [[1, 1], [2, 2], [3, 3]]
    out = e.serve(prompts, 2, eos_ids=())
    assert [p["B"] for p in e.prefills] == [2, 1]
    assert e.prefills[0]["mask"] is True and e.prefills[-1]["mask"] is False, "a single row needs no mask"
    assert e.prefills[-1]["rows"] == [[3, 3]]
    assert out == [toy_tokens(p, 2) for p in prompts]


def test_serve_of_one_prompt_is_one_row() -> None:
    """the whole queue is a single prompt: the single-row path, one flat answer at index 0"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [4])
    out = e.serve([[3, 1, 4]], 3, eos_ids=())
    assert out == [toy_tokens([3, 1, 4], 3)]
    assert [p["B"] for p in e.prefills] == [1] and e.prefills[0]["mask"] is False


def test_serve_takes_any_sequence_of_tokens() -> None:
    """the queue is copied to lists on the way in: tuples and range objects are prompts too"""
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    out = e.serve([(1, 2, 3), range(4, 7), [8, 9]], 2, eos_ids=())
    assert out == [toy_tokens([1, 2, 3], 2), toy_tokens([4, 5, 6], 2), toy_tokens([8, 9], 2)]


def test_serve_of_nothing_is_nothing() -> None:
    """guards the empty queue: no plan, no prefill, no max() over an empty range"""
    e = _StubEngine(dev="cuda")
    seen = _recording_plan(e)
    assert e.serve([], 8, eos_ids=()) == []
    assert seen == [] and e.prefills == []


def test_serve_derives_the_pad_id_when_it_is_not_given() -> None:
    """guards the pad id: the config's pad_token_id first, else the first eos, else 0 - a wrong pad id would
    be attended as a real token by any path that reads the ids without the mask"""
    with_cfg = _StubEngine(dev="cuda", cfg=model_config(pad_token_id=99))
    _counting_max_batch(with_cfg, [4])
    with_cfg.serve([[1], [2, 3, 4]], 1, eos_ids=(7,))
    assert int(with_cfg.prefills[0]["ids"][0, 0]) == 99, "the config's pad id wins"

    from_eos = _StubEngine(dev="cuda")
    _counting_max_batch(from_eos, [4])
    from_eos.serve([[1], [2, 3, 4]], 1, eos_ids=(7,))
    assert int(from_eos.prefills[0]["ids"][0, 0]) == 7, "no pad id in the config: the first eos"

    from_zero = _StubEngine(dev="cuda")
    _counting_max_batch(from_zero, [4])
    from_zero.serve([[1], [2, 3, 4]], 1, eos_ids=())
    assert int(from_zero.prefills[0]["ids"][0, 0]) == 0, "no pad id and no eos: 0"

    given = _StubEngine(dev="cuda", cfg=model_config(pad_token_id=99))
    _counting_max_batch(given, [4])
    given.serve([[1], [2, 3, 4]], 1, eos_ids=(7,), pad_id=5)
    assert int(given.prefills[0]["ids"][0, 0]) == 5, "an explicit pad id wins over both"


def test_serve_with_no_new_tokens_still_answers_one_row_per_prompt() -> None:
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [2])
    assert e.serve([[1, 2], [3], [4, 5, 6]], 0, eos_ids=()) == [[], [], []]


def test_serve_stops_a_prompt_on_an_eos_at_the_first_step() -> None:
    e = _StubEngine(dev="cuda")
    _counting_max_batch(e, [3])
    prompts = [[1, 2], [3, 4, 5], [6]]
    eos = toy_tokens(prompts[2], 1)[0]
    out = e.serve(prompts, 5, eos_ids=(eos,))
    assert out[2] == [eos]
    for b in (0, 1):
        expect = toy_tokens(prompts[b], 5)
        cut = next((i + 1 for i, t in enumerate(expect) if t == eos), len(expect))
        assert out[b] == expect[:cut]


def test_serve_resizes_between_epochs_but_never_inside_one() -> None:
    """guards the whole point of the scheduler: free memory that moves between epochs gives different epoch
    sizes, and no epoch ever changes size while it decodes (the stub's forward asserts the batch dim holds)"""
    e = _StubEngine(dev="cuda")
    seen = _recording_plan(e)
    row = e.scheduler.kv_row_bytes(2 + 3)
    # the free memory is read once by serve's plan and once by the decode's own guard; it holds for the first
    # epoch and then falls to one row's worth, so epoch 0 runs 3 rows and everything after runs alone
    with cuda_stats(free=[3 * row, 3 * row, 1 * row], reserved=0, allocated=0):
        out = e.serve([[i, i + 1] for i in range(6)], 3, eos_ids=())
    assert [s["batch"] for s in seen] == [3, 1, 1, 1], "the first epoch got 3 rows, the shrunk card 1 at a time"
    assert [p["B"] for p in e.prefills] == [3, 1, 1, 1], "each epoch decoded at the size it was planned"
    assert out == [toy_tokens([i, i + 1], 3) for i in range(6)]


def test_serve_epoch_is_split_again_when_memory_falls_between_the_plan_and_the_decode() -> None:
    """Pins the second guard: `serve` plans an epoch, then `generate_greedy` re-reads the free memory before
    it allocates and splits that epoch further if the card has shrunk in between. serve's plan is therefore
    an upper bound, not a promise - which is what keeps a queued stream off an OOM when another tenant takes
    memory between the plan and the prefill. The rows' answers are unchanged either way."""
    e = _StubEngine(dev="cuda")
    seen = _recording_plan(e)
    row = e.scheduler.kv_row_bytes(2 + 3)
    with cuda_stats(free=[3 * row, 1 * row], reserved=0, allocated=0):  # falls before the first decode
        out = e.serve([[i, i + 1] for i in range(6)], 3, eos_ids=())
    assert [s["batch"] for s in seen] == [3, 1, 1, 1], "serve still planned 3 for the first epoch"
    assert [p["B"] for p in e.prefills] == [1] * 6, "the decode split that epoch of 3 into three of 1"
    assert out == [toy_tokens([i, i + 1], 3) for i in range(6)]


def test_serve_off_the_card_takes_every_prompt_in_one_epoch() -> None:
    e = _StubEngine(dev="cpu")
    seen = _recording_plan(e)
    prompts = [[1], [2, 3], [4, 5, 6]]
    out = e.serve(prompts, 2, eos_ids=())
    assert [s["batch"] for s in seen] == [3] and [p["B"] for p in e.prefills] == [3]
    assert out == [toy_tokens(p, 2) for p in prompts]


# --- the VRAM policy's gates (btb/engine/memory.py) -----------------------------------------------------------


class _PolicyEngine(_MemoryMixin):
    """the policy's state, with the two actions it can take recorded instead of run"""

    def __init__(self, vram_margin: int = 1 * GB, shed: Sequence[str] = (), watch: bool = True) -> None:
        self.dev = _where("cuda")
        self.vram_watch = watch
        self.vram_margin = int(vram_margin)
        # the policy reads free memory and applies its moves through the engine's device, as the real one does
        self.device = Device(self)
        # a floor already learned, so any shared byte reads as bled; the realloc arm already spent, since these
        # tests are about the shed gate
        self.vram_state = VramPolicyState(shared_floor=0, realloc_tried=True)
        self._last_card_ms = 0.0
        self._card_ms_min = None
        self._shed = list(shed)
        self.lines: list[str] = []
        self.shed_calls: list[types.SimpleNamespace | None] = []
        self.regrow_calls: list[types.SimpleNamespace | None] = []
        self.realloc_calls: list[bool] = []

    def log(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    def vram_shed(self, cache: Any = None, log: Log | None = None) -> str:
        self.shed_calls.append(cache)
        return "layer 27"

    def vram_regrow(self, cache: Any = None, log: Log | None = None) -> str:
        self.regrow_calls.append(cache)
        return "layer 27"

    def vram_realloc(self, log: Log | None = None) -> list[str]:
        self.realloc_calls.append(True)
        return []

    def _regrow_bytes(self) -> int:
        return MB


@contextlib.contextmanager
def pressure(shared_mb: int) -> Iterator[None]:
    """the PDH sensor, patched at the module the policy reads it from"""
    saved = memory_mod.vram_pressure
    memory_mod.vram_pressure = lambda pid=None: {
        "adapters": {"a": {"dedicated": 0, "shared": 0}},
        "process": {"dedicated": 8 * GB, "shared": int(shared_mb) << 20},
    }
    try:
        yield
    finally:
        memory_mod.vram_pressure = saved


def _batched_cache(B: int) -> types.SimpleNamespace:
    layer = types.SimpleNamespace(keys=torch.zeros(B, 2, 4, 8), values=torch.zeros(B, 2, 4, 8))
    return types.SimpleNamespace(layers=[layer])


def test_vram_policy_holds_the_layers_while_the_card_has_room() -> None:
    """guards the gate that stopped the 0.6B model shedding and regrowing its last layer four times with 9 GB
    free: a bled reading with free memory above the margin is logged once and held, never acted on"""
    e = _PolicyEngine(vram_margin=1 * GB)
    with pressure(512), cuda_stats(free=9 * GB):
        e.vram_policy(_batched_cache(1))
        e.vram_policy(_batched_cache(1))
    assert e.shed_calls == [] and e.realloc_calls == []
    held = [ln for ln in e.lines if "holding the card's layers" in ln]
    assert len(held) == 1, "the same reason is logged once, then held silently"
    assert e.vram_state.held == "bled"


def test_vram_policy_sheds_when_the_card_is_inside_its_margin() -> None:
    """the other side of the gate: bled AND less free than the margin is a real memory shortage, and the
    resident layer leaves the card"""
    e = _PolicyEngine(vram_margin=2 * GB)
    with pressure(512), cuda_stats(free=1 * GB):
        e.vram_policy(_batched_cache(1))
    assert len(e.shed_calls) == 1
    assert any("reason: bled" in ln for ln in e.lines)
    assert e.vram_state.held is None and e._card_ms_min is None


def test_vram_policy_stands_aside_for_a_batched_decode() -> None:
    """guards the batched gate: a cache whose batch dim is past 1 is already sized by the scheduler, so the
    per-step policy returns before it even reads the sensor - shedding mid-epoch mismatches a layer's mask
    against its cache on the next epoch's prefill and collapses throughput"""
    e = _PolicyEngine(vram_margin=8 * GB)
    calls = {"n": 0}
    saved = memory_mod.vram_pressure

    def counted(pid: int | None = None) -> Json:
        calls["n"] += 1
        return {"adapters": {}, "process": {"dedicated": 0, "shared": GB}}

    memory_mod.vram_pressure = counted
    try:
        with cuda_stats(free=0):
            e.vram_policy(_batched_cache(4))
    finally:
        memory_mod.vram_pressure = saved
    assert calls["n"] == 0, "the sensor is not even read for a batched cache"
    assert e.shed_calls == [] and e.lines == []


def test_vram_policy_does_not_stand_aside_for_a_single_row_cache() -> None:
    """the gate is the batch dim, not the presence of a cache: one row still gets the streaming policy"""
    e = _PolicyEngine(vram_margin=2 * GB)
    with pressure(512), cuda_stats(free=1 * GB):
        e.vram_policy(_batched_cache(1))
    assert len(e.shed_calls) == 1


@contextlib.contextmanager
def budget_room(start: int, freed_per_shed: int, e: _PolicyEngine, trim_frees: int = 0) -> Iterator[list[int]]:
    """the WDDM budget's room for this process, `start` bytes (negative: past the budget), rising by
    `freed_per_shed` with every layer the stub engine sheds and by `trim_frees` with every emptying of torch's cache
    (the trim tried before any shed); the PDH sensor made to fail the test if read"""
    room = [int(start)]
    saved_room, saved_pdh, saved_info = device_mod._wddm_room, memory_mod.vram_pressure, device_mod._wddm_info
    saved_cuda = (
        torch.cuda.synchronize,
        torch.cuda.empty_cache,
        torch.cuda.memory_reserved,
        torch.cuda.memory_allocated,
    )
    real_shed = e.vram_shed

    def shed(cache: Any = None, log: Log | None = None) -> str:
        room[0] += int(freed_per_shed)
        return real_shed(cache, log)

    def no_pdh(pid: int | None = None) -> Json:
        raise AssertionError("the budget's answer needs no PDH read")

    device_mod._wddm_room = lambda dev: room[0]
    # the budget and the usage apart, the budget 11 GB as the engine came up: a room below zero is a smaller budget
    device_mod._wddm_info = lambda dev: (11 * GB + min(0, room[0]), 11 * GB - max(0, room[0]))
    e.vram_state.budget_hi = 11 * GB
    memory_mod.vram_pressure = no_pdh
    e.vram_shed = shed  # type: ignore[method-assign]
    # the trim's own torch calls: no card here (CUDA hidden), its cache a figure the budget answers to
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda: room.__setitem__(0, room[0] + int(trim_frees))
    torch.cuda.memory_reserved = lambda *a, **k: 0
    torch.cuda.memory_allocated = lambda *a, **k: 0
    try:
        yield room
    finally:
        device_mod._wddm_room, memory_mod.vram_pressure, device_mod._wddm_info = saved_room, saved_pdh, saved_info
        (torch.cuda.synchronize, torch.cuda.empty_cache, torch.cuda.memory_reserved, torch.cuda.memory_allocated) = (
            saved_cuda
        )


def test_vram_policy_gives_the_card_back_the_moment_its_budget_shrinks() -> None:
    """a game started beside btb: Windows cuts this process's budget below what it holds, and the next pass gives
    back as much as the budget asks - four layers in one call, not a layer a second, and with no PDH read - leaving
    the margin kept for other programs; adapt never reacted to this before (it waited for its memory to be paged
    out, or its step to slow, and even then shed one layer a second: the game failed to start meanwhile)"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    with budget_room(-3 * GB // 2, 600 * MB, e) as room:
        e.vram_policy(_batched_cache(1))
        assert len(e.shed_calls) == 4, f"{len(e.shed_calls)} layers given back"
        assert room[0] >= e.vram_margin, "the margin for other programs is not kept"
        e.vram_state.last_t = time.time()  # the PDH read is not due: only the budget is read this pass
        e.vram_policy(_batched_cache(1))
    assert len(e.shed_calls) == 4, "a budget with room sheds nothing more"
    assert any("another program took 1.50 GiB of the card" in ln for ln in e.lines), e.lines


def test_vram_yield_names_its_own_growth_past_the_budget_apart_from_another_programs() -> None:
    """the budget as it was when the engine came up, the process's own use past it: the log says the plan fell short,
    not that another program took the card (Qwen3-4B alone on the card read as a game for hours)"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    with budget_room(-GB, 600 * MB, e):
        device_mod._wddm_info = lambda dev: (11 * GB, 12 * GB)  # the budget unmoved, the process past it
        e.vram_yield(_batched_cache(1))
    assert any("grew past its own budget" in ln for ln in e.lines), e.lines
    assert not any("another program" in ln for ln in e.lines), e.lines


def test_vram_policy_within_its_budget_gives_nothing_back() -> None:
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    e.vram_state.last_t = time.time()  # the PDH read is not due: only the budget is read this pass
    with budget_room(2 * GB, 600 * MB, e):
        e.vram_policy(_batched_cache(1))
    assert e.shed_calls == [] and e.lines == []


def test_vram_yield_gives_all_it_can_and_asks_again_only_when_the_budget_moves() -> None:
    """a game taking the card as it frees (Windows cuts the budget with every layer given back): the card graphs let
    go first, then every layer the engine holds; with nothing left, the next passes do not ask again (the log is not
    flooded) until the budget is cut deeper or recovers"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 36
    left = [3]  # layers the stub still holds
    real_shed = e.vram_shed

    def shed(cache: Any = None, log: Log | None = None) -> str | None:
        if not left[0]:
            return None
        left[0] -= 1
        return real_shed(cache, log)

    let_go: list[bool] = []
    e._card_let_go = lambda: let_go.append(True)  # type: ignore[method-assign]
    with budget_room(-GB, 0, e) as room:  # the game takes each freed chunk: the budget does not move
        e.vram_shed = shed  # type: ignore[method-assign,assignment]
        e.vram_policy(_batched_cache(1))
        assert len(e.shed_calls) == 3 and left[0] == 0, f"{len(e.shed_calls)} layers given back of 3"
        assert let_go, "the card graphs were not let go before the sheds"
        e.vram_state.last_t = time.time()
        e.vram_policy(_batched_cache(1))
        said = [ln for ln in e.lines if "past this process's budget" in ln]
        assert len(said) == 1, "asked again with nothing more to give"
        room[0] = -2 * GB  # the game takes more
        e.vram_policy(_batched_cache(1))
        said = [ln for ln in e.lines if "past this process's budget" in ln]
        assert len(said) == 2, "a deeper cut was not answered"
        room[0] = GB  # the game gone: inside the budget again
        e.vram_state.last_t = time.time()
        e.vram_policy(_batched_cache(1))
        assert e.vram_state.spent == 0


def test_torchs_cached_blocks_answer_a_budget_before_any_placement_change() -> None:
    """a warm-up's 1.5 GB of freed blocks still in torch's cache when another program asks (Qwen3-4B, measured by
    the -vv trace): emptying the cache is enough - no layer leaves the card, no card graph is let go, and the
    placement's version does not move (a request moves it, and every card graph and card program is rebuilt for
    the new one: for nothing, where the cache was enough)"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    let_go: list[bool] = []
    e._card_let_go = lambda: let_go.append(True)  # type: ignore[method-assign]
    with budget_room(-GB, 600 * MB, e, trim_frees=2 * GB):
        v0 = e.device.version
        e.vram_policy(_batched_cache(1))
        assert e.device.version == v0, "the placement moved for torch's own cache"
    assert e.shed_calls == [] and not let_go, "a layer or the graphs given up for torch's own cache"
    assert e.vram_state.spent == 0
    assert any("torch's cached blocks" in ln for ln in e.lines), e.lines


def test_a_budget_past_again_soon_after_a_trim_is_answered_by_a_shed() -> None:
    """a pass's own transients refilling the cache: the trim answers the first time; past the budget again within
    TRIM_AGAIN_S the policy yields and sheds a layer at least - lasting room, where a trim a pass had emptied the
    cache and moved the placement on every pass without ever making any"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    e._card_let_go = lambda: None  # type: ignore[method-assign]
    with budget_room(-GB, 600 * MB, e, trim_frees=2 * GB) as room:
        e.vram_policy(_batched_cache(1))
        assert e.shed_calls == [], "the first trim was enough"
        room[0] = -GB // 4  # the next pass's transients: past the budget again
        e.vram_policy(_batched_cache(1))
    assert len(e.shed_calls) >= 1, "past again within TRIM_AGAIN_S and no layer shed"


def test_a_cut_back_to_the_budget_at_load_is_another_programs_once_the_budget_rose() -> None:
    """btb loaded beside a game (its budget then 5.8 GB), the game gone (11 GB given, the watcher noting it), then
    back: the cut is the game's - read against the largest budget given, where the budget at load called it btb's
    own growth past its plan"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.vram_state.budget_hi = 11 * GB
    saved = device_mod._wddm_info
    device_mod._wddm_info = lambda dev: (int(5.8 * GB), int(6.2 * GB))
    try:
        why = e._vram_why()
    finally:
        device_mod._wddm_info = saved
    assert why.startswith("another program took"), why


def test_the_memory_policies_stand_aside_for_the_warm_up_and_answer_once_it_is_done() -> None:
    """the load's warm-up times passes: a yield between two of its captures let the card graphs go and moved the
    placement (its kernel choice written into a state no pass read). The watchers' try at the decode lock fails during
    it, the warm-up's own passes' policies stand aside (`warming`), and once it is done each policy reads its budget
    - the watcher's try then succeeds"""
    from btb.engine import StreamedTextModel

    seen: list[bool] = []
    policies: list[tuple[str, bool]] = []
    stub = types.SimpleNamespace(
        dev=torch.device("cuda"), mlx=None, _decode_lock=threading.RLock(), vram_trim=lambda tag="": None
    )
    stub.warming = False
    stub.vram_policy = lambda cache=None, log=None: policies.append(("vram", stub.warming))
    stub.ram_policy = lambda log=None: policies.append(("ram", stub.warming))
    stub._warming = lambda: StreamedTextModel._warming(cast(Any, stub))

    def watcher_try() -> None:
        got = stub._decode_lock.acquire(blocking=False)
        seen.append(got)
        if got:
            stub._decode_lock.release()

    def card_warm(ids: Tokens) -> int:
        assert stub.warming, "the warm-up's passes ran with the policies acting"
        t = threading.Thread(target=watcher_try)
        t.start()
        t.join()
        return 3

    stub.card_warm = card_warm
    assert StreamedTextModel.warm(cast(Any, stub)) == 3
    watcher_try()
    assert seen == [False, True], f"the watcher's tries during and after the warm-up: {seen}"
    assert policies == [("vram", False), ("ram", False)], f"the policies once the warm-up is done: {policies}"
    assert not stub.warming


def test_a_warm_up_pass_gives_nothing_back_however_short_the_budget() -> None:
    """a pass of the warm-up (`warming`) past its budget: nothing shed, trimmed or requested - the placement stays
    as the warm-up is timing it; the budget is answered once it is done (`_warming`)"""
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    e.warming = True
    with budget_room(-2 * GB, 600 * MB, e):
        v0 = e.device.version
        e.vram_policy(_batched_cache(1))
        assert e.device.version == v0 and e.shed_calls == [] and e.vram_state.trim_t == 0.0


def test_vram_yield_stops_when_nothing_is_left_to_shed() -> None:
    e = _PolicyEngine(vram_margin=GB // 2)
    e.L = 8
    with budget_room(-GB, 0, e):
        e.vram_shed = lambda cache=None, log=None: None  # type: ignore[method-assign,assignment,return-value]
        assert e.vram_yield(_batched_cache(1)) == []
    assert any("nothing left to shed" in ln for ln in e.lines), e.lines


class _YieldEngine(_MemoryMixin):
    """the RAM yield's view of an engine: a machine whose free RAM and commit the test sets (commit as RAM unless
    staged apart), an expert store and host layers whose giving-back moves them - a shed by `gain` (RAM, commit), the
    OS's figures moved by it unless `lag` (the pages not yet taken back)"""

    def __init__(self, free: int, reserve: int, per_block: int, blocks: int, host: int, per_layer: int) -> None:
        self.dev = _where("cuda")
        self.adapt, self.mlx, self.L, self.ram_reserve = True, None, 8, int(reserve)
        self.machine = {"free": int(free), "low": False}
        self.lines: list[str] = []
        self.host = dict.fromkeys(range(host))
        self.cold: set[int] = set()
        self.shed: list[int] = []
        self.gain = (int(per_layer), int(per_layer))
        self.lag = False
        self.warming = False
        self.ram_state = RamPolicyState()
        self.requests: list[str] = []
        eng = self

        class Store:
            n, hold = int(blocks), (0, 0.0)

            def hold_for(self, nbytes: int, seconds: float) -> None:
                self.hold = (int(nbytes), float(seconds))

            def release(self, want: int = 1) -> int:
                freed = 0
                while self.n and eng.machine["free"] - eng.ram_reserve < want:
                    self.n -= 1
                    eng.machine["free"] += per_block
                    if "commit" in eng.machine:
                        eng.machine["commit"] += per_block
                    freed += 1
                return freed

        class Dev:
            version = 0

            def request(self, what: str, fn: Callable[[], Any]) -> Any:
                self.version += 1
                eng.requests.append(what)
                return fn()

        self.store = Store()
        self.expert_store = cast(Any, self.store)
        self.device = cast(Any, Dev())

    def log(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    def _shed_gain(self, i: int) -> tuple[int, int]:
        return self.gain

    def _shed_warm(self, why: str = "", log: Log | None = None) -> tuple[int, int, int] | None:
        i = self._next_warm()
        if i is None:
            return None
        self.cold.add(i)
        self.shed.append(i)
        if not self.lag:
            self.machine["free"] += self.gain[0]
            if "commit" in self.machine:
                self.machine["commit"] += self.gain[1]
        return i, *self.gain


@contextlib.contextmanager
def machine(e: _YieldEngine, total: int = 64 * GB) -> Iterator[None]:
    """the OS's memory reads the RAM yield takes, answered from the engine's staged machine"""
    names = ("host_free_bytes", "host_commit_bytes", "memory_pressure", "host_total_bytes")
    saved = {n: getattr(memory_mod, n) for n in names}
    memory_mod.host_free_bytes = lambda: e.machine["free"]
    memory_mod.host_commit_bytes = lambda: e.machine.get("commit", e.machine["free"])
    memory_mod.memory_pressure = lambda: {"low": e.machine["low"], "level": 1.0 if e.machine["low"] else 0.0}
    memory_mod.host_total_bytes = lambda: int(total)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(memory_mod, n, f)


def test_ram_short_is_the_reserve_and_a_launchs_headroom_once_another_program_takes_memory() -> None:
    e = _YieldEngine(free=10 * GB, reserve=4 * GB, per_block=GB, blocks=8, host=0, per_layer=GB)
    with machine(e):
        assert e._ram_headroom() == 4 * GB, "a sixteenth of 64 GB"
        assert e._ram_short() == 0, "plenty above the reserve: nothing is short"
        e.machine["free"] = 3 * GB  # a game took 7 GB: 1 GB into the reserve
        assert e._ram_short() == 5 * GB, "the reserve's gigabyte back, and a launch's headroom"
        e.machine["free"] = 5 * GB  # above the reserve, but the OS says memory is low
        assert e._ram_short() == 0
        e.machine["low"] = True
        assert e._ram_short() == 3 * GB, "the OS's word counts: the headroom above what is left"
        e.adapt = False
        assert e._ram_short() == 0, "adapt off: nothing is ever given back"


def test_ram_yield_gives_the_store_back_first_then_host_layers_and_holds_the_store() -> None:
    """a game took RAM into the reserve: the store's blocks go back until the reserve and a launch's headroom are
    free, and the store is held off growing back; host layers go to the drive only past what the store could give"""
    e = _YieldEngine(free=3 * GB, reserve=4 * GB, per_block=GB, blocks=3, host=4, per_layer=GB)
    with machine(e):
        gained = e.ram_yield()
        assert e.store.n == 0, "the store's blocks were not given back first"
        assert e.shed == [3, 2], f"host layers to the drive past the store: {e.shed}"
        assert e._ram_left() == e._ram_headroom() and gained == 5 * GB, "the reserve and the headroom are free"
        assert e.store.hold == (e._ram_headroom(), e.RAM_HOLD_S), "the store is not held off growing back"
    assert any("another program wants memory" in ln for ln in e.lines), e.lines


def test_ram_yield_stops_when_nothing_is_left_to_give() -> None:
    e = _YieldEngine(free=1 * GB, reserve=4 * GB, per_block=GB, blocks=1, host=0, per_layer=GB)
    with machine(e):
        e.ram_yield()
    assert any("short of the headroom, nothing more a shed frees" in ln for ln in e.lines), e.lines


def test_ram_yield_counts_what_each_shed_frees_not_the_lagging_os_figure() -> None:
    """the OS takes a shed layer's pages back only as it gets round to them: read again after each shed, the figure
    did not move and one yield sent every warm layer to the drive. Counted by what each shed let go, the yield stops
    once the reserve and the headroom are covered"""
    e = _YieldEngine(free=3 * GB, reserve=4 * GB, per_block=GB, blocks=0, host=8, per_layer=GB)
    e.lag = True
    with machine(e):
        e.ram_yield()
    assert e.shed == [7, 6, 5, 4, 3], f"5 GB short, 1 GB a layer: {e.shed}"


def test_ram_yield_sheds_no_mapped_layer_for_a_commit_shortfall() -> None:
    """a layer on the checkpoint's mapping holds RAM and no commit, and its ring slot takes both: with commit short
    and RAM to spare a shed of it only makes things worse, and none is made; a layer of its own copies frees both"""
    e = _YieldEngine(free=20 * GB, reserve=4 * GB, per_block=GB, blocks=0, host=8, per_layer=GB)
    e.machine["commit"] = 3 * GB
    e.gain = (GB, -GB // 4)  # mapped: its pages back, the ring's slot taken
    with machine(e):
        e.ram_yield()
        assert e.shed == [], f"a mapped layer shed for commit: {e.shed}"
        e.ram_state.spent = 0
        e.gain = (GB, GB)  # its own copies: RAM and commit both freed
        e.ram_yield()
    assert e.shed == [7, 6, 5, 4, 3], e.shed


def test_a_shortfall_a_yield_cannot_answer_is_asked_once_not_every_pass() -> None:
    """with nothing left to give and another program still in the reserve, every pass asked again - a request
    moves the placement's version (the card graphs rebuilt) and the yield logged two lines - and the watcher four
    times a second. Asked once; again only when the other program wants RAM_AGAIN more, or after the host was clear"""
    e = _YieldEngine(free=1 * GB, reserve=4 * GB, per_block=GB, blocks=0, host=0, per_layer=GB)
    e._decode_lock = threading.RLock()
    abort = threading.Event()
    with machine(e):
        for _ in range(3):
            e.ram_policy()
            e._ram_watch_once(abort)
        assert e.requests == ["ram-yield"], e.requests
        assert sum("another program wants memory" in ln for ln in e.lines) == 1, e.lines
        e.machine["free"] -= GB  # the other program takes a gigabyte more
        e.ram_policy()
        e._ram_watch_once(abort)
        assert e.requests == ["ram-yield"] * 2, "a deeper shortfall was not answered"
        e.machine["free"] = 20 * GB  # gone: clear again
        e.ram_policy()
        assert e.ram_state.spent == 0 and e.requests == ["ram-yield"] * 2
        e.machine["free"] = 1 * GB  # back: a fresh shortfall is answered
        e._ram_watch_once(abort)
        assert e.requests == ["ram-yield"] * 3, e.requests


def test_the_adapt_watchers_outlive_an_exception(monkeypatch: MonkeyPatch) -> None:
    """a read or a yield that raised ended a watcher's thread, and adapt was off for the engine's life, unsaid: each
    watcher says the error once and keeps watching"""
    e = _YieldEngine(free=20 * GB, reserve=4 * GB, per_block=GB, blocks=0, host=0, per_layer=GB)
    e.abort = threading.Event()
    e.vram_watch = True
    calls = {"ram": 0, "vram": 0}
    again = {"ram": threading.Event(), "vram": threading.Event()}

    def once(kind: str) -> Callable[..., Any]:
        def f(*a: object) -> bool:
            calls[kind] += 1
            if calls[kind] <= 2:
                raise RuntimeError(f"{kind} read failed")
            again[kind].set()
            return False

        return f

    e._ram_watch_once = once("ram")  # type: ignore[method-assign]
    e._vram_watch_once = once("vram")  # type: ignore[method-assign]
    reg = types.SimpleNamespace(event=0)
    monkeypatch.setattr(memory_mod, "wddm_budget_event", lambda *a: reg)
    monkeypatch.setattr(memory_mod, "wddm_budget_unregister", lambda r: None)

    def wait_event(handle: int, timeout_s: float) -> bool:
        time.sleep(0.01)
        return False

    monkeypatch.setattr(memory_mod, "wait_event", wait_event)
    monkeypatch.setattr(device_mod, "_wddm_info", lambda dev: None)
    monkeypatch.setattr(device_mod, "card_ids", lambda dev: ("card", GB, 0))
    try:
        e.watch_ram()
        e.watch_vram_budget()
        assert again["ram"].wait(5.0) and again["vram"].wait(5.0), f"a watcher stopped after raising: {calls}"
    finally:
        e.abort.set()
        # ended before the patched OS calls are put back
        for t in threading.enumerate():
            if t.name in ("btb-ram-watch", "btb-vram-budget"):
                t.join(5.0)
    assert sum("the memory watcher: RuntimeError" in ln for ln in e.lines) == 1, e.lines
    assert sum("the budget watcher: RuntimeError" in ln for ln in e.lines) == 1, e.lines


def test_a_shed_whose_host_copy_is_refused_leaves_the_layer_on_the_card() -> None:
    """the host's copy is made before the card's leaves: refused, the layer stays resident, never in neither tier"""

    def refuse(i: int) -> Any:
        raise MemoryGrantError("no room on the host")

    stub = types.SimpleNamespace(
        aj=None,
        resident={3: object()},
        host={},
        dev=torch.device("cpu"),
        resident_head=False,
        _shed=[],
        log=lambda *a: None,
        _layer_bytes=lambda i: GB,
        _card_let_go=lambda: None,
        _make_host_layer=refuse,
    )
    with pytest.raises(MemoryGrantError):
        _MemoryMixin.vram_shed(cast(Any, stub))
    assert 3 in stub.resident and not stub.host


def test_the_widest_speculative_pass_is_held_to_what_the_family_verifies() -> None:
    """a tree budget past what the family verifies exactly (Qwen4's card program: 32 rows) is held to it - a wider
    pass took the torch path, and its pricing raised past the program's widths - and a pass the pricer is handed past
    them is priced at its own rows"""
    from btb.engine.cuda import _CudaMixin

    stub = types.SimpleNamespace(v_max=4, tree_budget=40, fam=types.SimpleNamespace(verify_rows=lambda sm: 32))
    assert _CudaMixin._spec_full(cast(Any, stub)) == 32
    stub.fam = types.SimpleNamespace(verify_rows=lambda sm: None)
    assert _CudaMixin._spec_full(cast(Any, stub)) == 41
    w = types.SimpleNamespace(CARD_T_MAX=32, _card_m=_CudaMixin._card_m)
    assert _CudaMixin._card_width(cast(Any, w), 3) == 4 and _CudaMixin._card_width(cast(Any, w), 41) == 41


def test_one_burst_does_not_price_a_width_out_for_good() -> None:
    """a width's first pass caught in another program's burst is taken as twice what the curve says of it at most,
    and a width not run since takes the warm-up's figure again after LIVE_STALE passes: measured once at ten times
    its cost, it was priced out and never run again. With a store priced, one burst no longer drives the rows'
    compute exponent to its ceiling"""
    from btb.engine.spec_cost import SpecCost

    curve = {1: 0.010, 2: 0.011, 4: 0.012, 8: 0.015}
    pc = SpecCost()
    pc.price(9, curve, 0.0)
    pc.record_pass(4, 0.12, 0.0, None)
    assert pc.curve_now()[4] == pytest.approx(0.024), pc.curve_now()
    for _ in range(pc.LIVE_STALE + 1):
        pc.record_pass(1, 0.010, 0.0, None)
    assert pc.curve_now()[4] == pytest.approx(0.012), pc.curve_now()
    priced = SpecCost()
    priced.price(9, curve, 0.001)
    for _ in range(priced.WARM):
        priced.record_pass(1, 0.010, 10.0, None)
    priced.record_pass(2, 0.080, 12.0, None)  # eight steps' time for two rows: a burst
    assert priced.h < 1.0, f"one burst drove the exponent to {priced.h}"


def test_a_card_graph_the_card_had_no_room_for_is_tried_again(monkeypatch: MonkeyPatch) -> None:
    """an out-of-memory build turned the card graph off until the placement moved - for the engine's life where none
    did (adapt off, a budget a trim answered, a model wholly on the card). It is tried again after CARD_RETRY_S,
    doubling with each refusal at one placement; the driver's own out-of-memory errors count as torch's do"""
    from btb.engine import cuda as cuda_mod
    from btb.engine.cuda import _CudaMixin

    now = [100.0]
    # cuda.py's clock alone: frozen for the whole process, every wait on it elsewhere stood still
    monkeypatch.setattr(cuda_mod, "time", types.SimpleNamespace(monotonic=lambda: now[0]))
    dev = types.SimpleNamespace(version=3)
    dev.snapshot = lambda: types.SimpleNamespace(version=dev.version)
    stub = types.SimpleNamespace(
        device=dev, CARD_RETRY_S=10.0, CARD_RETRY_MAX_S=300.0, _card_let_go=lambda: None, log=lambda *a: None
    )
    e = cast(Any, stub)
    oom = RuntimeError("CUDA error: out of memory (cudaGraphInstantiate)")
    assert _CudaMixin._is_card_oom(oom) and _CudaMixin._is_card_oom(torch.OutOfMemoryError("x"))
    assert not _CudaMixin._is_card_oom(RuntimeError("an index out of range"))
    _CudaMixin._card_oom(e, oom)
    assert _CudaMixin._card_off_now(e)
    now[0] += 11
    assert not _CudaMixin._card_off_now(e), "not tried again once its retry was due"
    _CudaMixin._card_oom(e, oom)
    now[0] += 11
    assert _CudaMixin._card_off_now(e), "a second refusal at one placement waits twice as long"
    dev.version += 1
    assert not _CudaMixin._card_off_now(e), "a placement moved: tried at once"


def test_a_let_go_drops_the_step_graphs_embedding_table(monkeypatch: MonkeyPatch) -> None:
    """the step graph's table - a granted copy, or a tied head's weight - went with nothing: a yield measured with it
    held shed layers past what the budget asked, and a tied head's shed freed nothing"""
    from btb.engine.cuda import _CudaMixin

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    st = {"graphs": {}, "layers": {"x": 1}, "table": torch.zeros(4)}
    stub = types.SimpleNamespace(_cg=st, _cp=None, dev=torch.device("cpu"), _trace_mem=lambda: (0, 0))
    _CudaMixin._card_let_go(cast(Any, stub))
    assert "table" not in st and not st["layers"]


def test_a_layers_rows_stay_in_ram_when_it_comes_to_the_card_under_kv_host() -> None:
    """under `kv_host` a layer regrown onto the card leaves its attention rows in RAM (a DeltaNet's states come
    with it), and its regrowth is priced without them; the card pass's move of the host layers' rows is the plain
    move (`_rows_to`), refused whole before a row moves"""
    from btb.engine.tiers import _TiersMixin

    moved: list[Any] = []
    stub = types.SimpleNamespace(
        kv_host=True,
        layer_types=[LayerKind.LINEAR, LayerKind.QWEN_SPARSE, LayerKind.FULL],
        _rows_to=lambda caches, layers, dev: moved.append(list(layers)),
        __dict__={},
    )
    e = cast(Any, stub)
    stub._rows_stay = types.MethodType(_TiersMixin._rows_stay, stub)
    assert _TiersMixin._rows_stay(e, 1, "cuda") and _TiersMixin._rows_stay(e, 2, "cuda")
    assert not _TiersMixin._rows_stay(e, 0, "cuda") and not _TiersMixin._rows_stay(e, 1, "cpu")
    _TiersMixin._caches_to(e, 1, "cuda")
    _TiersMixin._caches_to(e, 0, "cuda")
    assert moved == [[0]], moved


def test_a_float32_chunk_is_not_priced_under_the_chunk_rule() -> None:
    """the rule's sweep priced a card chunk with neither the mask nor the keys widened to every head; under --fp32
    (or a head wider than 256) the attention builds both after all, a gigabyte a layer at 32k unpriced"""
    from btb.engine.forward import _ForwardMixin

    cfg = types.SimpleNamespace(num_attention_heads=8, head_dim=128, hidden_size=1024, _attn_implementation="btb_sdpa")
    stub = types.SimpleNamespace(
        cfg=cfg,
        fam=types.SimpleNamespace(chunk_causal=True),
        layer_types=[LayerKind.FULL] * 2,
        dev=torch.device("cuda"),
        compute_dtype=torch.float32,
    )
    assert not _ForwardMixin._chunk_rule(cast(Any, stub), 1)
    stub.compute_dtype, cfg.head_dim = torch.bfloat16, 512
    assert not _ForwardMixin._chunk_rule(cast(Any, stub), 1)


def test_a_drafting_engines_passes_hold_its_decode_lock() -> None:
    """a --draft-model engine runs its own adapt watchers, which take a free decode lock for an idle engine: its
    passes, driven by the proposer, hold the lock as a decode does, so a yield waits for them"""
    from btb.engine.propose import ModelProposer

    lock = threading.RLock()
    seen: list[bool] = []

    class Eng:
        _decode_lock, _mega = lock, None
        layer_types: list[str] = []

        def new_cache(self) -> Any:
            return types.SimpleNamespace(layers=[])

        def _prefill(self, ids: Any, cache: Any) -> None:
            seen.append(lock._is_owned())  # type: ignore[attr-defined]

        def aa(self, parents: Any) -> None:
            pass

        def ab(self) -> None:
            pass

        def forward(self, ids: Any, cache: Any = None, last_only: bool = True, **kw: Any) -> torch.Tensor:
            seen.append(lock._is_owned())  # type: ignore[attr-defined]
            return torch.zeros(1, len(ids[0]), 8)

    p = ModelProposer(cast(Any, Eng()), [1, 2, 3])
    p._topk([4], [-1], 2)
    assert seen == [True, True], seen


def test_the_cuda_runtime_to_pin_with_is_found() -> None:
    """the RAM arena's pinning needs the runtime torch loaded: found beside torch, where the process maps it, or in
    the nvidia-cuda-runtime package (a Linux wheel keeps it there); with none, the program declines kv_host rather
    than failing the load"""
    from btb.engine import hostmem

    if torch.version.cuda is None:
        pytest.skip("a torch without CUDA")
    path = hostmem._runtime_path()
    assert path is not None and "cudart" in os.path.basename(path), path
    assert hostmem.can_pin()


def test_a_budget_event_registered_on_a_lost_adapter_is_stale(monkeypatch: MonkeyPatch) -> None:
    """after a driver reset the card is found afresh as another adapter: the event registered on the old one
    signals nothing, and the watcher registers again"""
    import ctypes

    from btb import sysinfo

    if sys.platform != "win32":
        pytest.skip("Windows' budget events")
    key = ("a card", GB, 0)
    reg = sysinfo.BudgetEvent(0, 0, ctypes.c_void_p(4))
    monkeypatch.setitem(sysinfo._WDDM, key, (ctypes.c_void_p(4), 0.0))
    assert not sysinfo.wddm_budget_stale(reg, *key)
    monkeypatch.setitem(sysinfo._WDDM, key, (ctypes.c_void_p(5), 0.0))
    assert sysinfo.wddm_budget_stale(reg, *key)


def test_a_failed_dxgi_lookup_is_tried_again_after_the_retry_not_every_read(monkeypatch: MonkeyPatch) -> None:
    """a lookup DXGI raised for was never kept, so it ran again on every read - every pass and every watcher second -
    each one leaking the factory and adapters it enumerated"""
    from btb import sysinfo

    n = [0]

    def lookup(*a: object) -> Any:
        n[0] += 1
        raise OSError("GetDesc1 failed")

    monkeypatch.setattr(sysinfo, "_wddm_adapter", lookup)
    key = ("no such card", GB, None)
    try:
        for _ in range(3):
            assert sysinfo._wddm_for("no such card", GB, None) is None
        assert n[0] == 1, f"looked up {n[0]} times within the retry"
    finally:
        sysinfo._WDDM.pop(key, None)


def test_vram_policy_is_off_when_it_is_not_watching_or_not_on_a_card() -> None:
    e = _PolicyEngine(watch=False)
    with pressure(512), cuda_stats(free=0):
        e.vram_policy(_batched_cache(1))
    assert e.shed_calls == [] and e.lines == []
    host = _PolicyEngine()
    host.dev = _where("cpu")
    with pressure(512), cuda_stats(free=0):
        host.vram_policy(_batched_cache(1))
    assert host.shed_calls == []


# --- the grant gate (SPEC: not landed yet) ---------------------------------------------------------------------
#
# BatchScheduler.grant(nbytes, kind, *, requester, B, cap, bound) is the one place an allocation past the free
# memory is refused instead of OOM'ing the process. These tests are written against that signature and are
# expected to fail with AttributeError until the patch lands; when it does they run for real, and a failure
# then is a failure of the gate, not of the test.


def _grant_api() -> tuple[ModuleType, type[MemoryGrantError]]:
    from btb.engine import scheduler as sched

    return sched, MemoryGrantError


def _grant_scheduler(free: int = 4 * GB) -> tuple[BatchScheduler, int]:
    return BatchScheduler(SchedulerModel(dev="cuda")), free


def test_grant_refuses_a_request_past_the_free_memory() -> None:
    _sched, err = _grant_api()
    s, free = _grant_scheduler()
    with cuda_stats(free=free, reserved=0, allocated=0), pytest.raises(err):
        s.grant(free + 1, "kv", requester="test", B=1, cap=1024, bound=4096)


def test_grant_refuses_a_kv_cap_past_its_bound() -> None:
    """a KV grant is refused when the cap asked for is past the bound the caller declared, whatever the size"""
    _sched, err = _grant_api()
    s, free = _grant_scheduler()
    with cuda_stats(free=free, reserved=0, allocated=0):
        with pytest.raises(err):
            s.grant(MB, "kv", requester="test", B=1, cap=8192, bound=4096)
        s.grant(MB, "kv", requester="test", B=1, cap=4096, bound=4096)  # cap == bound is legal


def test_grant_warns_past_the_warn_fraction_of_free() -> None:
    """a grant that is a large share of the free memory is granted, and says so in the log"""
    _sched, _err = _grant_api()
    sm = SchedulerModel(dev="cuda")
    s = BatchScheduler(sm)
    with cuda_stats(free=4 * GB, reserved=0, allocated=0):
        s.grant(int(0.30 * 4 * GB), "kv", requester="test", B=1, cap=1024, bound=4096)
        assert any("warn" in ln.lower() or "large" in ln.lower() for ln in sm.lines), sm.lines
        n = len(sm.lines)
        s.grant(int(0.05 * 4 * GB), "kv", requester="test", B=1, cap=1024, bound=4096)
        assert len(sm.lines) == n, "a small request is granted quietly"


def test_a_host_grant_takes_room_back_from_the_expert_store_only_when_the_caller_lets_it(
    monkeypatch: MonkeyPatch,
) -> None:
    """The expert store grows into whatever RAM the ledger shows free, so a host request made after it has is short.
    Without `reclaim` the request is refused and the store left alone (a caller mid-call could lose experts it is
    multiplying); with it the store gives blocks back for the request - asked once, for what the request needs -
    and the grant goes through. A request that fits never asks."""
    from btb.engine import scheduler as S

    _sched, err = _grant_api()
    free = {"now": 64 * MB}
    monkeypatch.setattr(S, "host_free_bytes", lambda: free["now"])

    class Store:
        def __init__(self) -> None:
            self.asked: list[int] = []

        def release(self, want: int = 1) -> int:
            self.asked.append(int(want))
            free["now"] += 256 * MB
            return 1

    sm = SchedulerModel(dev="cpu")
    store = Store()
    monkeypatch.setattr(sm, "expert_store", store, raising=False)
    s = BatchScheduler(sm)
    with pytest.raises(err):
        s.grant(128 * MB, "scratch", requester="test", device="cpu")
    assert store.asked == [], "a grant that may not reclaim leaves the store alone"
    s.grant(128 * MB, "scratch", requester="test", device="cpu", reclaim=True)
    assert store.asked == [128 * MB]
    s.grant(MB, "scratch", requester="test", device="cpu", reclaim=True)
    assert store.asked == [128 * MB], "a request the room holds asks nothing back"


def test_grant_allows_a_plausible_request() -> None:
    _sched, _err = _grant_api()
    s, free = _grant_scheduler()
    with cuda_stats(free=free, reserved=0, allocated=0):
        s.grant(MB, "kv", requester="test", B=2, cap=1024, bound=4096)
        s.grant(MB, "weights", requester="test", B=2, cap=1024, bound=4096)


# -- the expert store's memory: bounded blocks, a margin above the reserve, no thrash at the reserve --


def _store(
    monkeypatch: MonkeyPatch, free: int, reserve: int = 4 * GB, per: int = 64 * KB, block_max: int = 4 * MB
) -> tuple[_ExpertStore, dict[str, int]]:
    """a store over a stub engine with a controllable free-RAM reading; slots of `per` bytes, blocks of at most
    `block_max`; a released block hands its bytes back to the reading, as the machine would see it"""
    from btb.engine import experts as experts_mod

    state = {"free": int(free)}
    sm = stub_engine(
        mlx=None,
        fam=Family(kind=FamilyKind.QWEN3),
        cold_chunk=0,
        expert_profile=None,
        device=stub_ledger(lambda: state["free"], reserve),
    )
    st = experts_mod._ExpertStore(sm, budget_bytes=(1 << 16) * per, reserve_bytes=reserve)
    st.per, st.n_slots = per, 1 << 16
    st.block_max = int(block_max)
    st.margin = max(st.block_max, st.reserve // 4)
    release_block = st._release_block

    def give_back(b: int) -> None:
        assert st.per is not None
        state["free"] += len(st.blocks[b][1]) * st.per
        release_block(b)

    st._release_block = give_back  # type: ignore[method-assign]
    return st, state


def test_store_blocks_are_bounded(monkeypatch: MonkeyPatch) -> None:
    st, state = _store(monkeypatch, free=0)
    state["free"] = st.reserve + st.margin + 100 * st.block_max
    k = st._grow(1)
    assert 0 < k * slot_size(st) <= st.block_max, "a block is at most block_max, however much room there is"
    assert k * slot_size(st) == st.block_max, "with room to spare the block is a whole block_max"


def test_store_grows_only_a_margin_above_the_reserve(monkeypatch: MonkeyPatch) -> None:
    st, state = _store(monkeypatch, free=0)
    state["free"] = st.reserve + st.margin + slot_size(st)
    assert st._grow(1) == 0, "just above the margin there is no room for a growth"
    state["free"] = st.reserve + st.margin + 4 * slot_size(st)
    assert st._grow(1) >= 1, "past the margin the store grows"
    state["free"] = st.reserve + st.margin // 2
    assert st._grow(1) == 0, "inside the margin the store does not grow, though free RAM is above the reserve"


def test_store_does_not_thrash_at_the_reserve(monkeypatch: MonkeyPatch) -> None:
    st, state = _store(monkeypatch, free=0)
    state["free"] = st.reserve + st.margin + 20 * st.block_max
    for _ in range(4):
        assert st._grow(1) > 0
    grown = st.live()
    # every slot in use, the oldest first
    for i in range(grown):
        seat(st, (0, i))
    st.max_call = 1
    # the machine takes memory: free RAM dips under the reserve
    state["free"] = st.reserve - slot_size(st)
    freed = st.release()
    assert 0 < freed * slot_size(st) <= st.block_max, "one block goes back, not the store"
    assert state["free"] >= st.reserve
    released_after_first = st.stat["released"]
    # the reading now sits between the reserve and the margin: no regrowth, no further release, ten times over
    for _ in range(10):
        assert st._grow(1) == 0
        assert st.release() == 0
    assert st.stat["released"] == released_after_first
    assert st.live() == grown - freed
    # the memory comes back past the margin: the store grows again
    state["free"] = st.reserve + st.margin + 4 * slot_size(st)
    assert st._grow(1) > 0


def test_store_release_holds_the_floor_for_the_largest_call(monkeypatch: MonkeyPatch) -> None:
    st, state = _store(monkeypatch, free=0)
    state["free"] = st.reserve + st.margin + 20 * st.block_max
    for _ in range(2):
        assert st._grow(1) > 0
    n = st.live()
    for i in range(n):
        seat(st, (0, i))
    st.max_call = n - 1  # freeing any block would leave fewer slots than the largest call served
    state["free"] = st.reserve - slot_size(st)
    assert st.release() == 0, "the store holds rather than fall below what a call needs"
    assert st.stat.get("floor") == 1


# -- the Route: the queue's order, dedupe and drop, the profile's rule, a worker end to end --


def _route(
    monkeypatch: MonkeyPatch, reads: list[tuple[str, int, int]], delay: float = 0.0
) -> tuple[BatchScheduler, dict[str, Any]]:
    """a scheduler over a stub engine whose reader appends (path, off, n) to `reads` instead of touching a drive"""
    import time as _time

    from btb.engine import native as native_mod

    def fake_read(path: str, off: int, n: int, dst: torch.Tensor, chunk: int = 0) -> None:
        if delay:
            _time.sleep(delay)
        reads.append((path, int(off), int(n)))

    monkeypatch.setattr(native_mod.Native, "read_direct", staticmethod(fake_read))
    for name in ("open", "read_at", "close"):
        monkeypatch.setattr(native_mod.Native, name, None, raising=False)
    s = BatchScheduler(stub_engine())
    st = s._disk_state()
    st["depth"] = 1
    return s, st


def test_route_serves_the_waiting_layer_first_then_by_file_and_offset(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]  # no reader thread: the order is read off the queue by hand
    dst = torch.empty(8, dtype=torch.uint8)
    s.disk_read("b.st", 4096, 8, dst, s.DISK_SWEEP)
    s.disk_read("a.st", 8192, 8, dst, s.DISK_AHEAD + 1)
    s.disk_read("a.st", 0, 8, dst, s.DISK_AHEAD)
    s.disk_read("b.st", 0, 8, dst, s.DISK_DEMAND)
    s.disk_read("a.st", 4096, 8, dst, s.DISK_DEMAND)
    order = []
    while True:
        t = s._disk_take(st)
        if t is None:
            break
        order.append((t[0]["path"], t[0]["off"]))
        st["inflight_demand"] = st["inflight_ahead"] = 0  # each read lands before the next is taken
    assert order == [("a.st", 4096), ("b.st", 0), ("a.st", 0), ("a.st", 8192), ("b.st", 4096)]


def test_route_queues_the_same_bytes_once_and_raises_their_priority(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    dst = torch.empty(8, dtype=torch.uint8)
    f1 = s.disk_read("a.st", 0, 8, dst, s.DISK_AHEAD + 1, key="k1")
    f2 = s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND, key="k2")
    assert f1 is f2
    assert len(st["reqs"]) == 1
    t = s._disk_take(st)
    assert t is not None and t[0]["priority"] == s.DISK_DEMAND
    assert s._disk_take(st) is None, "the stale entry of the raised request is skipped"


def test_route_drops_a_lapsed_prediction_before_it_is_issued(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    dst = torch.empty(8, dtype=torch.uint8)
    f = s.disk_read("a.st", 0, 8, dst, s.DISK_AHEAD, key=(3, 7))
    g = s.disk_read("a.st", 4096, 8, dst, s.DISK_DEMAND, key=(3, 9))
    assert s.disk_drop((3, 7)) == 1
    assert f.cancelled()
    t = s._disk_take(st)
    assert t is not None and t[0]["future"] is g
    assert s._disk_take(st) is None


def test_route_merges_adjacent_reads_only_where_the_drive_seeks(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    dst = torch.empty(8, dtype=torch.uint8)
    s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND)
    s.disk_read("a.st", 8, 8, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and t[1] == [], "no merging on a drive that does not seek"
    assert s._disk_take(st) is not None
    st["merge"] = True
    s.disk_read("a.st", 16, 8, dst, s.DISK_DEMAND)
    s.disk_read("a.st", 24, 8, dst, s.DISK_DEMAND)
    s.disk_read("a.st", 40, 8, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [24], "two adjacent reads go as one"
    t = s._disk_take(st)
    assert t is not None and t[1] == [], "a gap is not bridged"
    # two padded spans share their boundary sector: they overlap, and go as one read of the union
    d1, d2 = torch.empty(8, dtype=torch.uint8), torch.empty(8, dtype=torch.uint8)
    s.disk_read("a.st", 64, 8, d1, s.DISK_DEMAND)
    s.disk_read("a.st", 68, 8, d2, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [68]
    # a read inside another is not a partner (nothing past the end to add)
    s.disk_read("a.st", 128, 16, dst, s.DISK_DEMAND)
    s.disk_read("a.st", 132, 4, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and t[1] == []
    assert s._disk_take(st) is not None
    # a run of five adjacent experts is one read, and a chain stops at the merged span's cap
    for i in range(5):
        s.disk_read("a.st", 256 + 8 * i, 8, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [264, 272, 280, 288], "the whole run as one read"
    monkeypatch.setattr(BatchScheduler, "DISK_MERGE_MAX", 20)
    for i in range(5):
        s.disk_read("a.st", 512 + 8 * i, 8, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [520], "16 bytes fit the cap, 24 would not"
    t = s._disk_take(st)
    assert t is not None and t[0]["off"] == 528 and [q["off"] for q in t[1]] == [536]


def test_route_never_merges_a_cached_read_with_a_direct_one(monkeypatch: MonkeyPatch) -> None:
    """a merged run is read through one handle: reads through the file cache and reads around it (an expert and a
    cold layer can share a shard) never go as one, while each kind still merges with its own"""
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    st["merge"] = True
    dst = torch.empty(8, dtype=torch.uint8)
    s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND, cached=True)
    s.disk_read("a.st", 8, 8, dst, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and t[1] == [], "a cached read and a direct one went as one"
    assert s._disk_take(st) is not None
    s.disk_read("a.st", 16, 8, dst, s.DISK_DEMAND, cached=True)
    s.disk_read("a.st", 24, 8, dst, s.DISK_DEMAND, cached=True)
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [24], "two adjacent cached reads go as one"


def test_route_reads_a_cached_request_through_a_cached_handle(monkeypatch: MonkeyPatch) -> None:
    """a reader keeps a handle a file and a mode: a read through the file cache on one opened by `open_cached`, a
    read around it on one opened by `open`, never the one for the other"""
    from btb.engine import native as native_mod

    reads: list[tuple[str, int, int]] = []
    s, _st = _route(monkeypatch, reads)
    opened: list[tuple[str, str]] = []
    used: list[int] = []
    ids = iter(range(1, 100))

    def opener(kind: str) -> Callable[[str], int]:
        def open_(path: str) -> int:
            opened.append((kind, path))
            return next(ids)

        return open_

    monkeypatch.setattr(native_mod.Native, "open", staticmethod(opener("direct")))
    monkeypatch.setattr(native_mod.Native, "open_cached", staticmethod(opener("cached")))
    monkeypatch.setattr(
        native_mod.Native, "read_at", staticmethod(lambda h, off, n, dst, chunk, depth: used.append(int(h)))
    )
    monkeypatch.setattr(native_mod.Native, "close", staticmethod(lambda h: None))
    dst = torch.empty(8, dtype=torch.uint8)
    try:
        s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND, cached=True).result(timeout=10)
        s.disk_read("a.st", 4096, 8, dst, s.DISK_DEMAND).result(timeout=10)
        s.disk_read("a.st", 8192, 8, dst, s.DISK_DEMAND, cached=True).result(timeout=10)
    finally:
        s.disk_close()
    handle = {kind: i + 1 for i, (kind, _p) in enumerate(opened)}
    assert sorted(opened) == [("cached", "a.st"), ("direct", "a.st")], opened
    assert used == [handle["cached"], handle["direct"], handle["cached"]], (used, handle)


def test_route_merged_read_lands_each_destination_from_the_union(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    from btb.engine import native as native_mod

    def fake_read(path: str, off: int, n: int, dst: torch.Tensor, chunk: int = 0) -> None:
        reads.append((path, int(off), int(n)))
        dst.copy_(torch.arange(off, off + n, dtype=torch.int64).to(torch.uint8))

    monkeypatch.setattr(native_mod.Native, "read_direct", staticmethod(fake_read))
    st["workers"] = [None]
    st["merge"] = True
    d1, d2, d3 = (torch.zeros(8, dtype=torch.uint8) for _ in range(3))
    f1 = s.disk_read("a.st", 64, 8, d1, s.DISK_DEMAND)
    f2 = s.disk_read("a.st", 68, 8, d2, s.DISK_DEMAND)
    f3 = s.disk_read("a.st", 76, 8, d3, s.DISK_DEMAND)
    t = s._disk_take(st)
    assert t is not None and len(t[1]) == 2
    s._disk_serve(st, t, lambda path, off, n, dst, chunk, depth, cached: fake_read(path, off, n, dst, chunk))
    assert reads == [("a.st", 64, 20)], "one read of the union"
    assert d1.tolist() == list(range(64, 72)) and d2.tolist() == list(range(68, 76))
    assert d3.tolist() == list(range(76, 84))
    assert f1.done() and f2.done() and f3.done() and st["inflight"] == 0 and not st["reqs"]


def test_route_rule_from_the_profile() -> None:
    rule = BatchScheduler._disk_rule
    nvme = {
        "fixed_ms": 0.15,
        "single_ms": 1.3,
        "copy_ms": 0.4,
        "reps": 3,
        "rates": {1: (4.8, 0.1), 4: (5.6, 0.1), 16: (6.2, 0.3)},
    }
    assert rule(nvme) == {"depth": 16, "merge": False, "ahead": 8}
    # sixteen readers tie with four (the drive saturates at four): the tie goes to the deeper queue
    tie = dict(nvme, rates={1: (5.0, 0.2), 4: (6.47, 0.1), 16: (6.45, 0.06)})
    assert rule(tie) == {"depth": 16, "merge": False, "ahead": 8}
    # sixteen readers measurably below four (a queue the drive thrashes on): four
    worse = dict(nvme, rates={1: (4.8, 0.1), 4: (5.6, 0.1), 16: (5.0, 0.1)})
    assert rule(worse) == {"depth": 4, "merge": False, "ahead": 2}
    # a drop of three percent with three bursts a depth is inside the floor: still deeper
    flat = dict(nvme, rates={1: (5.0, 0.05), 4: (4.9, 0.05), 16: (4.85, 0.05)})
    assert rule(flat)["depth"] == 16
    # a seek costs four bounce copies many times over: merged; fifty milliseconds of the drive's time is one read
    hdd = {
        "fixed_ms": 9.0,
        "single_ms": 55.0,
        "copy_ms": 0.6,
        "reps": 2,
        "rates": {1: (0.12, 0.0), 4: (0.13, 0.0), 16: (0.12, 0.0)},
    }
    # ... and one read outlasts the fifty milliseconds a prediction may hold, so none is allowed: a prediction
    # there is a read taken from the layer waiting now, and it cannot be recalled in flight
    assert rule(hdd) == {"depth": 16, "merge": True, "ahead": 0}
    # with two bursts a depth the floor is ten percent: a fifteen-percent loss at sixteen is a loss
    hdd4 = dict(hdd, rates={1: (0.12, 0.0), 4: (0.135, 0.0), 16: (0.11, 0.0)})
    assert rule(hdd4) == {"depth": 4, "merge": True, "ahead": 0}
    # a disk whose read fits the window once keeps one prediction in flight
    assert rule(dict(hdd, single_ms=40.0))["ahead"] == 1
    # a controller's fixed cost under four copies: separate
    sata = {
        "fixed_ms": 0.3,
        "single_ms": 13.0,
        "copy_ms": 0.6,
        "reps": 3,
        "rates": {1: (0.5, 0.01), 4: (0.52, 0.02), 16: (0.53, 0.02)},
    }
    assert rule(sata) == {"depth": 16, "merge": False, "ahead": 3}
    assert rule({}) == {"depth": 4, "merge": False, "ahead": 2}
    assert rule({"rates": {4: (1.0, 0.0)}}) == {"depth": 4, "merge": False, "ahead": 2}
    # a profile that stopped short of sixteen readers keeps the deepest it measured
    assert rule({"rates": {1: (1.0, 0.0), 4: (1.1, 0.0)}}) == {"depth": 4, "merge": False, "ahead": 2}


def test_route_queue_keeps_one_reader_for_the_prediction_class_where_the_rule_allows_none(
    monkeypatch: MonkeyPatch,
) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    hdd = {
        "measured": True,
        "fixed_ms": 9.0,
        "single_ms": 55.0,
        "single_gbs": 0.12,
        "copy_ms": 0.6,
        "reps": 2,
        "rates": {1: (0.12, 0.0), 4: (0.13, 0.0), 16: (0.12, 0.0)},
        "seq_gbs": 0.15,
        "big_mb": 6.25,
        "cost_s": 1.0,
    }
    monkeypatch.setattr(BatchScheduler, "measure_drive", staticmethod(lambda path, clock=None: dict(hdd)))
    monkeypatch.setattr(s, "_volume", lambda path: "X:")  # a path that exists on no OS
    p = s.disk("x.st")
    assert p["ahead"] == 0 and p["merge"] and p["depth"] == 16, "the rule: no predictions, merged, deep"
    assert st["ahead_cap"] == 1 and st["depth"] == 16 and st["merge"], "the queue keeps one for the cold ring"


def test_lookahead_withheld_on_a_drive_with_no_room(monkeypatch: MonkeyPatch) -> None:
    st, _r, _sm = _store_with_routers(monkeypatch, lookahead=(2, 1))
    assert st._grow(64) > 0
    x = torch.ones(1, 4)
    st.drive = {"ahead": 0}
    assert st.lookahead(0, x) == 0, "the rule allowed no prediction on this drive"
    st.drive = {"ahead": 2}
    st.saturated = True
    assert st.lookahead(0, x) == 0, "the first pass found the drive saturated"
    st.saturated = False
    assert st.lookahead(0, x) > 0, "the NVMe profile leaves it on"


def test_lookahead_withheld_while_the_route_reports_the_drive_slow(monkeypatch: MonkeyPatch) -> None:
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 1))
    assert st._grow(64) > 0
    x = torch.ones(1, 4)
    route.slow = True
    assert st.lookahead(0, x) == 0, "a drive under half the probe's rate gets no prediction"
    route.slow = False
    assert st.lookahead(0, x) > 0, "and gets them back when it recovers"


def test_the_drive_report_and_the_first_pass_verdict(monkeypatch: MonkeyPatch) -> None:
    from btb.engine.experts import _ExpertStore

    per = 9830400
    nvme = {"fixed_ms": 0.13, "single_ms": 1.3, "big_mb": 6.25}
    hdd = {"fixed_ms": 9.0, "single_ms": 55.0, "big_mb": 6.25}
    # two fixed costs and the bytes at the single-stream rate: 2.2 ms on the NVMe drive, 100 ms on the disk
    assert 0.002 < _ExpertStore.expert_s(nvme, per) < 0.0025
    assert 0.09 < _ExpertStore.expert_s(hdd, per) < 0.11
    assert _ExpertStore.expert_s({}, per) == 0.0
    line = _ExpertStore.drive_report(hdd, 3300, 24576, per)
    assert "ms" in line and "3300 of 24576" in line
    assert "not measured" in _ExpertStore.drive_report({}, 1, 2, per)
    st, _state = _store(monkeypatch, free=64 * GB)
    st.drive = hdd
    # the median of sixteen passes decides, not the first: one inflated pass (the reload after a prefill)
    # does not call a drive saturated by itself, and fifteen steady ones outvote it
    assert st._pass_closed(2.0, 1.5, 900) == "" and not st._decided
    verdict = ""
    for _ in range(15):
        verdict = st._pass_closed(2.0, 1.5, 90)
    assert "saturated" in verdict and st.saturated and st._decided, "90 misses outlast half a second of compute"
    st2, _state2 = _store(monkeypatch, free=64 * GB)
    st2.drive = nvme
    st2._pass_closed(0.5, 0.3, 900)
    for _ in range(15):
        verdict = st2._pass_closed(2.0, 1.5, 90)
    assert "has room" in verdict and not st2.saturated
    st3, _state3 = _store(monkeypatch, free=64 * GB)
    st3.drive = None
    assert st3._pass_closed(2.0, 1.5, 90) == "" and not st3._pass_samples
    # the verdict is live: a saturated drive that eases (or a machine that frees up) turns it back, said once
    turns = [st._pass_closed(2.0, 0.1, 1) for _ in range(16)]
    assert not st.saturated and sum("has room" in t for t in turns) == 1
    assert all(t == "" for t in turns if "has room" not in t), "silent between turns"
    assert len(st._pass_samples) == 16, "a rolling window"


def test_the_first_one_row_pass_decides_and_a_long_prompt_is_announced(monkeypatch: MonkeyPatch) -> None:
    st, _r, sm = _store_with_routers(monkeypatch, lookahead=(0, 0))
    lines: list[str] = []
    sm.log = lines.append
    st.drive = {"fixed_ms": 9.0, "single_ms": 55.0, "big_mb": 6.25}
    assert st._grow(8) > 0
    st.get(0, "layers.0.mlp.experts.", [1, 2], rows=51)
    assert not st._decided and not st._pass_samples, "a prefill's pass is not one that counts"
    for i in range(16):
        st.get(0, "layers.0.mlp.experts.", [1 + (i % 3)], rows=1)
        assert len(st._pass_samples) == i, "a one-row pass counts when the next layer 0 closes it"
    assert not st._decided
    st.get(0, "layers.0.mlp.experts.", [3], rows=1)
    assert st._decided and any("one-row passes" in x for x in lines), "the sixteenth closes, the medians decide"
    st2, _r2, sm2 = _store_with_routers(monkeypatch, lookahead=(0, 0))
    lines2: list[str] = []
    sm2.log = lines2.append
    # the fixture's expert is 64 KB: a read of 1,000 s a span makes it 10 s an expert, 80 s a layer of two
    st2.drive = {"fixed_ms": 9.0, "single_ms": 1e6, "big_mb": 6.25}
    assert st2._grow(8) > 0
    st2.get(0, "layers.0.mlp.experts.", [1, 2], rows=51)
    assert sum("minutes to its first token" in x for x in lines2) == 1
    st2.get(0, "layers.0.mlp.experts.", [3, 4], rows=51)
    assert sum("minutes to its first token" in x for x in lines2) == 1, "said once"


def test_route_reads_land_with_their_duration_and_callbacks(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads, delay=0.02)
    st["workers"] = [None]  # queue everything first, then let one reader loose
    dst = torch.empty(8, dtype=torch.uint8)
    seen = []
    f0 = s.disk_read("a.st", 0, 8, dst, s.DISK_SWEEP, on_done=lambda d: seen.append(("sweep", d)))
    f1 = s.disk_read("a.st", 8192, 8, dst, s.DISK_SWEEP, on_done=lambda d: seen.append(("sweep2", d)))
    f2 = s.disk_read("a.st", 4096, 8, dst, s.DISK_DEMAND, on_done=lambda d: seen.append(("demand", d)))
    st["workers"] = []
    with st["cv"]:
        s._disk_start(st)
        st["cv"].notify_all()
    for f in (f0, f1, f2):
        assert f.result(timeout=5) > 0
    assert [r[1] for r in reads] == [4096, 0, 8192], "the layer waiting now goes first, then the sweep in offset order"
    assert [t for t, _ in seen] == ["demand", "sweep", "sweep2"]
    assert all(d > 0 for _, d in seen)
    s.disk_close()
    assert st["workers"] == []


def test_route_settles_a_failed_read_and_a_failing_callback_and_keeps_both_callers(monkeypatch: MonkeyPatch) -> None:
    """the same bytes asked twice while queued are one read that runs both callers' `on_done`; a callback that
    raises is logged and its read's future still lands; a read that raises settles every future of its merged run
    - a prediction's pair here - with the error, and the in-flight counts go back to nothing"""
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    lines: list[str] = []
    s.sm.log = lines.append
    st["workers"] = [None]
    st["ahead_cap"] = 4
    dst = torch.empty(8, dtype=torch.uint8)
    seen: list[str] = []
    f = s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND, on_done=lambda d: seen.append("first"))
    g = s.disk_read("a.st", 0, 8, dst, s.DISK_DEMAND, on_done=lambda d: seen.append("second"))
    assert f is g and len(st["reqs"]) == 1
    s._disk_serve(st, s._disk_take(st), lambda *a: None)
    assert f.result() >= 0 and seen == ["first", "second"]

    def boom(dur: int) -> None:
        raise ValueError("boom")

    h = s.disk_read("a.st", 4096, 8, dst, s.DISK_DEMAND, on_done=boom)
    s._disk_serve(st, s._disk_take(st), lambda *a: None)
    assert h.result() >= 0 and any("a read's on_done raised: ValueError('boom')" in x for x in lines), lines
    st["merge"] = True
    a = s.disk_read("a.st", 64, 8, dst, s.DISK_AHEAD, key=(1, 2))
    b = s.disk_read("a.st", 72, 8, dst, s.DISK_AHEAD, key=(1, 3))
    t = s._disk_take(st)
    assert t is not None and [q["off"] for q in t[1]] == [72] and st["inflight_ahead"] == 2

    def gone(*args: object) -> None:
        raise OSError("the drive went away")

    s._disk_serve(st, t, gone)
    for fut in (a, b):
        with pytest.raises(OSError, match="the drive went away"):
            fut.result()
    assert st["inflight"] == st["inflight_ahead"] == st["inflight_demand"] == 0 and not st["reqs"]


def test_route_pulse_reads_the_drives_busy_time_and_profiles_its_turns(tmp_path: Path) -> None:
    """the live rate: reads that took no time rate nothing; the last reads' bytes over their busy time under half
    the probe's rate is a slowed drive and above four fifths a recovered one, each turn logged and profiled"""
    from btb.engine.experts import ExpertProfile

    s = BatchScheduler(stub_engine())
    lines: list[str] = []
    s.sm.log = lines.append
    prof = s.sm.expert_profile = ExpertProfile(str(tmp_path / "p.npz"))
    st = s._disk_state()
    st["live"] = [(t, t, MB) for t in range(32)]
    assert s._disk_pulse(st) is None, "no busy time, no rate"
    st["expect_gbs"] = 10.0
    for busy_ns, what, aux in ((500_000, "slowed", 1), (100_000, "recovered", 0)):
        st["live"] = [(i * 1_000_000, i * 1_000_000 + busy_ns, MB) for i in range(32)]
        turn = s._disk_pulse(st)
        assert turn is not None and turn[0] == what and turn[1] == pytest.approx(MB / busy_ns)
        s._disk_turned(st, turn)
        assert f"the drive {what}" in lines[-1] and (("predictions withheld" in lines[-1]) == (what == "slowed"))
        row = prof.a[prof.n - 1]
        assert int(row[2]) == prof.DRIVE and int(row[10]) == aux and int(row[8]) == int(turn[1] * 1e9)
    assert s._disk_pulse(st) is None, "no turn while the rate holds"


def test_route_depth_forced_and_a_reader_that_outlives_the_close(monkeypatch: MonkeyPatch) -> None:
    """`BTB_ROUTE_DEPTH` forces the readers in flight over the probe's rule, the predictions capped at half of them;
    a reader still in a read when the queue closes is said so and kept, and the queue stays stopped for it"""
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    lines: list[str] = []
    s.sm.log = lines.append
    nvme = {
        "measured": True,
        "fixed_ms": 0.15,
        "single_ms": 1.3,
        "single_gbs": 4.8,
        "copy_ms": 0.4,
        "reps": 3,
        "rates": {1: (4.8, 0.1), 4: (5.6, 0.1), 16: (6.2, 0.3)},
        "seq_gbs": 6.0,
        "big_mb": 6.25,
        "cost_s": 1.0,
    }
    monkeypatch.setattr(BatchScheduler, "measure_drive", staticmethod(lambda path, clock=None: dict(nvme)))
    monkeypatch.setattr(s, "_volume", lambda path: "Y:")
    monkeypatch.setenv("BTB_ROUTE_DEPTH", "2")
    p = s.disk("y.st")
    assert p["depth"] == 2 and p["ahead"] == 1 and st["depth"] == 2, p
    assert any("2 in flight (forced)" in x for x in lines), lines

    class Stuck:
        """a reader deep in a read: the join's timeout passes and it is still there"""

        def join(self, timeout: float | None = None) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    stuck = Stuck()
    st["workers"] = [stuck]
    s.disk_close()
    assert any("1 reader(s) still in a read at close" in x for x in lines)
    assert st["workers"] == [stuck] and st["stop"]
    st["workers"], st["stop"] = [], False


def test_the_schedulers_small_arithmetic() -> None:
    """a size at the unit that shows it; a card's recovery pauses doubling with a streak, capped, and a streak
    forgotten after a clean stretch; the drive file the probe reads is the largest weight file that is there"""
    import time as _time

    from btb.engine.scheduler import _size

    assert (_size(1000), _size(3 * MB), _size(5 * GB)) == ("1.0 KiB", "3.00 MiB", "5.00 GiB")
    s = BatchScheduler(stub_engine())
    assert [s.gpu_recovered() for _ in range(6)] == pytest.approx([0.1, 0.2, 0.4, 0.8, 1.6, 2.0])
    s._gpu_cool_until = _time.monotonic() - 6.0
    assert s.gpu_recovered() == pytest.approx(0.1), "a clean stretch clears the streak"


def test_the_probe_reads_the_largest_weight_file_there_is(tmp_path: Path) -> None:
    """`_drive_file`: the largest of the model's weight files on disk, a file the map names but the disk lacks
    passed over; nothing for a probe with no files"""
    (tmp_path / "a.st").write_bytes(b"x" * 10)
    (tmp_path / "c.st").write_bytes(b"x" * 20)
    probe = types.SimpleNamespace(dir=str(tmp_path), weight_map={"a": "a.st", "b": "gone.st", "c": "c.st"})
    assert BatchScheduler._drive_file(probe) == os.path.join(str(tmp_path), "c.st")
    assert BatchScheduler._drive_file(types.SimpleNamespace(dir="", weight_map={})) is None


# -- the Timetable: the next layers' picks read ahead into the ring, promoted on use, withdrawn when lapsed --


def test_store_falls_back_to_pageable_blocks_when_the_machine_will_not_pin(monkeypatch: MonkeyPatch) -> None:
    st, _state = _store(monkeypatch, free=64 * GB)
    real_empty = torch.empty

    def no_pin(*a: int, **k: Any) -> torch.Tensor:  # torch.empty's own keywords, passed on
        if k.get("pin_memory"):
            raise RuntimeError("pinning refused")
        return real_empty(*a, **k)

    monkeypatch.setattr(torch, "empty", no_pin)
    lines: list[str] = []
    st.sm.log = lines.append
    st.pin = True
    assert st._grow(1) > 0
    assert st.pin is False and any("would not pin" in x for x in lines)
    buf, _ids = st.blocks[0]
    assert buf.data_ptr() % 4096 == 0 and not buf.is_pinned()
    assert st._grow(1) > 0, "the next block is pageable without another attempt"
    assert sum("would not pin" in x for x in lines) == 1


@pytest.mark.timing
def test_profile_watch_sees_a_thread_holding_the_gil_and_names_it(tmp_path: Path) -> None:
    import threading
    import time

    import numpy as np

    from btb.engine.experts import ExpertProfile

    prof = ExpertProfile(str(tmp_path / "p.npz"))
    prof.watch(every_s=0.001, late_ms=3.0)
    stop = time.perf_counter() + 0.4

    def hog() -> None:
        n = 0
        while time.perf_counter() < stop:
            n += 1

    th = threading.Thread(target=hog, name="hog", daemon=True)
    th.start()
    th.join()
    time.sleep(0.02)
    prof.save()
    ev = prof.a[: prof.n]
    g = ev[ev[:, 2] == prof.GIL]
    # idle the sleeper wakes every 1.5 ms; under a hog every 10 ms or so, each wake held past the switch interval
    assert g.shape[0] > 10
    assert (g[:, 8] > 3e6).any(), "a wake held past the switch interval while the hog ran"
    snaps = "\n".join(str(x) for x in np.load(tmp_path / "p.npz")["snapshots"])
    assert "hog: " in snaps, "the snapshot, saved with the trace, lists the hog's frame"
    assert sorted(q.name for q in tmp_path.iterdir()) == ["p.npz"], "nothing beside the trace"


def test_profile_keeps_each_calls_picks_and_saves_them(tmp_path: Path) -> None:
    """a call's picks copied row by row into the profile's own array, the offset of its first row returned (the
    `call` event's offset); the array doubles past its rows and widens once for a call of more picks, a narrower
    call's rows -1 past its k; and `save` writes them beside the events"""
    import numpy as np

    from btb.engine.experts import ExpertProfile

    prof = ExpertProfile(str(tmp_path / "p.npz"))
    a = torch.randint(0, 128, (5, 8))
    offs = [prof.keep_picks(a)]
    cap = prof.p.shape[0]
    b = torch.randint(0, 128, (cap, 10), dtype=torch.int32)  # past the rows, and wider: one growth covers both
    offs.append(prof.keep_picks(b))
    c = torch.randint(0, 128, (3, 8))
    offs.append(prof.keep_picks(c))
    assert offs == [0, 5, 5 + cap] and prof.m == 8 + cap and prof.p.shape == (2 * cap, 10)
    prof.add(prof.CALL, 3, expert=3, offset=offs[2])
    prof.save()
    z = np.load(tmp_path / "p.npz")
    picks, (call,) = z["picks"], z["events"]
    assert picks.shape == (8 + cap, 10) and picks.dtype == np.int16
    assert (picks[:5, :8] == a.numpy()).all() and (picks[:5, 8:] == -1).all()
    assert (picks[5 : 5 + cap] == b.numpy()).all()
    off, rows = int(call[7]), int(call[4])
    assert (picks[off : off + rows, :8] == c.numpy()).all() and (picks[off : off + rows, 8:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_store_pins_its_blocks_on_a_page_boundary_beside_a_card(monkeypatch: MonkeyPatch) -> None:
    st, _state = _store(monkeypatch, free=64 * GB)
    st.pin = True
    assert st._grow(1) > 0
    buf, _ids = st.blocks[0]
    assert buf.is_pinned() and buf.data_ptr() % 4096 == 0
    st._release_block(0)
    assert 0 not in st.blocks


def _store_with_routers(
    monkeypatch: MonkeyPatch, lookahead: tuple[int, ...] = (2, 1), ring_n: int = 64
) -> tuple[_ExpertStore, FakeRoute, types.SimpleNamespace]:
    """four layers of eight routed experts over the fake Route: (the store, the route, the stub engine)"""
    route = FakeRoute()
    st, sm = expert_store(monkeypatch, route, n_layers=4, n_experts=8, lookahead=lookahead, ring_n=ring_n, routers=True)
    return st, route, sm


def test_lookahead_reads_the_next_layers_top_picks_that_are_not_resident(monkeypatch: MonkeyPatch) -> None:
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 1))
    _sm.lookahead_rows = (2, 1)
    h = torch.ones(1, 4)
    # expert 7 of layer 1 is resident already: it is not read again
    st._grow(1)
    seat(st, (1, 7))
    n = st.lookahead(0, h)
    assert n == 2, "layer 1's top-2 less the resident one, and layer 2's top-1"
    keys = [r["key"] for r in route.reads]
    assert keys == [(1, 6), (1, 6), (2, 7), (2, 7)], "two reads an expert (gate_up, down), the next layer first"
    pri = [r["priority"] for r in route.reads]
    assert pri == [route.DISK_AHEAD, route.DISK_AHEAD, route.DISK_AHEAD + 1, route.DISK_AHEAD + 1]
    assert set(st.ahead) == {(1, 6), (2, 7)} and len(st.ring) == 2
    assert st.lookahead(0, h) == 0, "predicted already: nothing queued twice"


def test_lookahead_hit_is_promoted_and_a_lapsed_prediction_withdrawn(monkeypatch: MonkeyPatch) -> None:
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0))
    h = torch.ones(1, 4)
    assert st.lookahead(0, h) == 2  # (1, 7) and (1, 6)
    route.land((1, 7))
    # layer 1 asks for 7 (predicted, landed) and 3 (a miss); 6 lapses with its reads still queued
    ready, pending = st.get(1, "layers.1.mlp.experts.", [7, 3])
    assert (1, 7) in st.lru and (1, 7) not in st.ahead, "the used prediction lives in the store now"
    assert st.stat["ahead_used"] == 1 and st.stat["hit"] == 1 and st.stat["miss"] == 1
    assert 7 in ready and [e for e, _f, _s in pending] == [3]
    assert (1, 6) not in st.ahead and st.stat["ahead_dropped"] == 1
    assert (1, 6) in route.dropped, "the lapsed prediction's reads were withdrawn from the queue"
    slot_6 = next(r for r in route.reads if r["key"] == (1, 6))
    assert slot_6["future"].cancelled()
    demand = [r for r in route.reads if r["key"] == (1, 3)]
    assert len(demand) == 2 and all(r["priority"] == route.DISK_DEMAND for r in demand)


def test_lookahead_hit_still_in_flight_is_waited_for_not_read_again(monkeypatch: MonkeyPatch) -> None:
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(1, 0))
    h = torch.ones(1, 4)
    assert st.lookahead(0, h) == 1  # (1, 7)
    n_reads = len(route.reads)
    ready, pending = st.get(1, "layers.1.mlp.experts.", [7])
    assert len(route.reads) == n_reads, "no demand read for an expert already on its way"
    assert ready == {} and len(pending) == 1 and pending[0][0] == 7
    assert not pending[0][1].done()
    route.land((1, 7))
    pending[0][1].result()
    assert pending[0][1].done()


def test_ring_wraps_over_landed_predictions_and_holds_at_in_flight_ones(monkeypatch: MonkeyPatch) -> None:
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0), ring_n=2)
    h = torch.ones(1, 4)
    assert st.lookahead(0, h) == 2  # ring full: (1, 7), (1, 6)
    assert st.lookahead(1, h) == 0, "both ring slots hold reads in flight: no room for layer 2's picks"
    route.land((1, 7))
    assert st.lookahead(1, h) == 1, "the oldest landed slot is reused for layer 2's first pick"
    assert (1, 7) not in st.ahead and (2, 7) in st.ahead and st.stat["ahead_recycled"] == 1
    assert len(st.ring) == 2


def test_a_store_below_a_calls_experts_serves_it_in_waves(monkeypatch: MonkeyPatch) -> None:
    """a store held below one call's experts (the machine's commit can hold it near a layer's, and a long prompt's
    call asks for all of one) serves the call in turn - the longest prefix it can seat, then the rest once those
    are done with - instead of refusing it; every expert once, in ascending order, never more seats than it has"""
    route = FakeRoute()
    st, _sm = expert_store(monkeypatch, route, n_layers=2, n_experts=8)
    assert st.per is not None
    st.block_max = 3 * st.per
    st._grow(3)
    st.n_slots = st.live()  # three seats, no growth
    assert st.live() == 3
    ids = list(range(8))
    served: list[int] = []
    call = st.call(0, "layers.0.mlp.experts.", ids, rows=4)
    while not call.done:
        rest = list(call.rest)
        ready, pending = call.wave()
        wave = sorted([*ready, *(e for e, _f, _s in pending)])
        assert wave == rest[: len(wave)] and wave, f"a wave is a prefix of what is left: {wave} of {rest}"
        assert len({s for _e, _f, s in pending} | set(st.last_slots.values())) <= 3
        for e in wave:
            route.land((0, e))  # the wave's reads in; the call multiplies it and asks for the rest
        served += wave
    assert served == ids, "every expert once, in ascending order"


def test_a_cold_store_on_a_tight_machine_serves_a_call_in_waves(monkeypatch: MonkeyPatch) -> None:
    """a store that has grown nothing yet, on a machine with room for a few slots, serves a long prompt's call in
    waves of what it can seat. Its first wave was sized to what the room holds without the two slots' slack a
    growth keeps, the growth for the whole wave failed, and the call was refused with the room for most of it
    free: a 120B's cold 4096-token prefill died at its first layer"""
    route = FakeRoute()
    st, _sm = expert_store(monkeypatch, route, n_layers=2, n_experts=16)
    per = st.per
    assert per is not None
    room = 5  # slots the machine holds above the margin, the store's own included
    monkeypatch.setattr(st, "_host_free", lambda: st.margin + (room - st.live()) * per)
    assert st.live() == 0
    ids = list(range(14))
    served: list[int] = []
    call = st.call(0, "layers.0.mlp.experts.", ids, rows=4096)
    while not call.done:
        rest = list(call.rest)
        ready, pending = call.wave()
        wave = sorted([*ready, *(e for e, _f, _s in pending)])
        assert wave == rest[: len(wave)] and wave, f"a wave is a prefix of what is left: {wave} of {rest}"
        assert st.live() <= room
        for e in wave:
            route.land((0, e))
        served += wave
    assert served == ids, "every expert once, in ascending order"


def test_residents_whose_block_goes_back_are_read_again_in_the_calls_order(monkeypatch: MonkeyPatch) -> None:
    """a release at a wave's start takes back the block the call's residents sit in: they are misses of that wave,
    read again where they fall in the call's order, and the wave is the prefix before the first expert without a
    seat. (A release once ran inside the call, after its hits were handed out: re-read and appended after the
    misses, they were left unseated before the cut, or seated after misses the cut dropped, which stayed in the
    line with no read behind them.) Every expert of every wave is read, none taken for resident"""
    route = FakeRoute()
    st, _sm = expert_store(monkeypatch, route, n_layers=2, n_experts=8)
    base = "layers.0.mlp.experts."
    assert st.per is not None
    st.block_max = 2 * st.per
    for _ in range(3):
        st._grow(2)
    st.n_slots = st.live()  # six seats
    _ready, pending = st.get(0, base, [0, 1], rows=1)  # 0 and 1 resident, together in one block
    for e, _f, _s in pending:
        route.land((0, e))
    route.reads.clear()
    monkeypatch.setattr(st, "_grow", lambda need: 0)  # nothing grows back: the call has the seats left
    real, calls = st.release, [0]

    def release(want: int = 1) -> Any:
        calls[0] += 1
        if calls[0] == 1:  # the first wave's start: the block 0 and 1 sit in goes back
            st._release_block(st.slots[dict(st.res.items())[(0, 0)]].block)
            return 0
        return real(want)

    monkeypatch.setattr(st, "release", release)
    ids = list(range(8))
    served: list[int] = []
    call = st.call(0, base, ids, rows=64)
    while not call.done:
        rest = list(call.rest)
        ready, pending = call.wave()
        wave = sorted([*ready, *(e for e, _f, _s in pending)])
        assert wave == rest[: len(wave)] and wave, f"a wave is a prefix of what is left: {wave} of {rest}"
        read = {r["key"][1] for r in route.reads if r["key"][0] == 0}
        assert all(e in read for e in wave), f"every expert of the wave read, none taken for resident: {wave}"
        for e in wave:
            route.land((0, e))
        route.reads.clear()
        served += wave
    assert served == ids, "every expert once, in ascending order"


def test_closing_a_store_lets_every_block_go(monkeypatch: MonkeyPatch) -> None:
    """`close` gives the store's blocks back whole - residents, free seats and the lookahead's ring alike - once
    every read into them has landed; the store holds no slot afterwards"""
    route = FakeRoute()
    st, _sm = expert_store(monkeypatch, route, n_layers=2, n_experts=8)
    assert st.per is not None
    st.block_max = 3 * st.per
    st._grow(3)
    st._grow(3)
    _ready, pending = st.get(0, "layers.0.mlp.experts.", [0, 1, 2], rows=1)
    for e, _f, _s in pending:
        route.land((0, e))
    bufs = [weakref.ref(buf) for buf, _ids in st.blocks.values()]
    assert len(bufs) == 2
    st.close()
    assert st.live() == 0 and not st.res and not st.slots and not st.free
    assert all(b() is None for b in bufs), "a closed store's block is still held"


def test_a_prediction_with_a_part_in_flight_is_never_withdrawn_in_part(monkeypatch: MonkeyPatch) -> None:
    """an expert is read as several parts: a prediction that lapses with one of them already in flight is left to
    land whole - no part withdrawn - and a later call that asks for it waits for every part. Withdrawn in part, the
    slot was a mix of two experts that read as landed: a 120B's routing collapsed on it"""
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0))
    h = torch.ones(1, 4)
    assert st.lookahead(0, h) == 2  # (1, 7) and (1, 6), two reads each
    route.land((1, 7))
    route.start((1, 6), parts=1)  # a reader has taken its first part; the second is still queued
    st.get(1, "layers.1.mlp.experts.", [7, 3])  # layer 1 asks for 7 and 3: 6 lapses
    six = [r["future"] for r in route.reads if r["key"] == (1, 6)]
    assert not any(f.cancelled() for f in six), "a part of an expert in flight was withdrawn"
    assert (1, 6) in st.ahead, "its record stays until it lands"
    ready, pending = st.get(1, "layers.1.mlp.experts.", [6])
    assert 6 not in ready and [e for e, _f, _s in pending] == [6], "asked for, it is waited for: every part"
    route.land((1, 6))
    assert all(f.done() and not f.cancelled() for f in six)


def test_the_lookahead_gives_a_call_no_slot_still_being_written(monkeypatch: MonkeyPatch) -> None:
    """a call with no seat left takes a prediction's slot back only whole: one with a part in flight keeps its
    slot (and every part), and the call takes the next"""
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0), ring_n=2)
    assert st.lookahead(0, torch.ones(1, 4)) == 2  # (1, 7), (1, 6) in the ring
    route.start((1, 7), parts=1)
    s7 = st.ahead[(1, 7)]
    got = st._ring_take()
    assert got is not None and got != s7, "the slot being written was given away"
    assert (1, 7) in st.ahead and not any(
        f.cancelled() for r in route.reads if r["key"] == (1, 7) for f in [r["future"]]
    )


def test_an_experts_reads_are_settled_before_its_slot_goes_back() -> None:
    """withdrawing a prediction cancels its queued parts; a part in flight still writes into the slot, so the slot
    is given back once that one has finished too - not at the first cancelled part"""
    from btb.engine.experts import Parts

    queued: Future[float] = Future()
    flying: Future[float] = Future()
    queued.cancel()
    threading.Timer(0.05, lambda: flying.set_result(1.0)).start()
    Parts([queued, flying]).settle()
    assert flying.done()


def test_a_call_with_no_seat_left_takes_one_back_from_the_lookahead(monkeypatch: MonkeyPatch) -> None:
    """a store that cannot grow (the host's commit ran out long before its ceiling) and whose every seat is the
    call's own: the call takes the lookahead's slots back, landed predictions and queued ones alike, instead of
    failing - the call before a guess"""
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0), ring_n=2)
    assert st.per is not None
    st.block_max = 2 * st.per  # a store of two seats, grown by the lookahead: every one the ring's
    h = torch.ones(1, 4)
    assert st.lookahead(0, h) == 2  # layer 1's (1, 7) and (1, 6) in the ring
    route.land((1, 7))  # one landed; (1, 6) still queued
    st.n_slots = st.live()  # no growth past the seats it has
    assert not st.free and len(st.ring) == st.live() == 2
    ready, pending = st.get(2, "layers.2.mlp.experts.", [0, 1])
    assert sorted(e for e, _f, _s in pending) == [0, 1], "both of the call's experts have a seat"
    assert not st.ring and st.ahead == {} and st.stat["ahead_dropped"] == 2
    assert (1, 6) in route.dropped, "the queued prediction's reads were withdrawn, not left to land in a used slot"


def test_riders_store_line_bumps_the_oldest_and_a_ride_renews() -> None:
    from btb.engine.experts import Riders

    r = Riders(lambda: 3)
    r.admit("a", 1)
    r.admit("b", 2)
    r.admit("c", 3)
    assert r.get("a") == 1, "a ride renews the seat"
    assert r.victim() == ("b", 2), "the oldest ride gives up its seat"
    assert "b" not in r and len(r) == 2
    assert r.victim(skip={3}) == ("a", 1), "a seat in `skip` is passed over"
    assert r.oldest_slot() == 3


def test_bus_pass_regulars_outlast_day_riders_and_ghosts_move_the_split() -> None:
    from btb.engine.experts import BusPass

    r = BusPass(lambda: 2)
    r.admit("a", 1)
    assert r.get("a") == 1 and "a" in r.t2, "a second ride makes a regular"
    r.admit("b", 2)
    assert r.victim() == ("b", 2), "the day rider goes before the regular"
    assert "b" in r.b1, "and is remembered as a ghost"
    r.admit("b", 2)
    assert "b" in r.t2 and r.p >= 1, "the ghost's return seats it as a regular and grows the day riders' share"
    r.admit("c", 3)
    assert r.victim() == ("a", 1), "the split moved to the day riders: the oldest regular gives way, not the day rider"
    assert "a" in r.b2, "and is remembered as a regular ghost"
    assert r.pop("b") == 2 and "b" not in r and "b" not in r.b1 and "b" not in r.b2, "a pop leaves no ghost"


def test_landed_yields_experts_as_their_reads_complete(monkeypatch: MonkeyPatch) -> None:
    import threading
    import time as _time

    from btb.engine.experts import Parts

    st, _route, _sm = _store_with_routers(monkeypatch)
    fa, fb, fc = Future[float](), Future[float](), Future[float]()
    pending = [(1, Parts([fa]), 11), (2, Parts([fb]), 12), (3, Parts([fc]), 13)]

    def later() -> None:
        _time.sleep(0.02)
        fc.set_result(0.0)
        _time.sleep(0.02)
        fa.set_result(0.0)
        fb.set_result(0.0)

    threading.Thread(target=later).start()
    order = [[e for e, _p, _s in batch] for batch in st.landed(pending)]
    assert order[0] == [3], "the first to land is computed first, whatever the submission order"
    assert sorted(e for batch in order[1:] for e in batch) == [1, 2]
    assert st.stat["wait_s"] > 0
    assert list(st.landed([])) == []


def test_lookahead_over_a_verify_pass_unions_the_rows_picks_up_to_twice_k(monkeypatch: MonkeyPatch) -> None:
    st, _route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0))
    _sm.lookahead_rows = (2, 0)
    # row one ranks the high experts first, row two the low ones: the union is both pairs
    h = torch.stack([torch.ones(4), -torch.ones(4)])
    assert st.lookahead(0, h) == 4
    assert {k[1] for k in st.ahead} == {7, 6, 0, 1}
    st2, _route2, _sm2 = _store_with_routers(monkeypatch, lookahead=(2, 0))
    _sm2.lookahead_rows = (2, 0)
    h3 = torch.stack([torch.ones(4), -torch.ones(4), torch.tensor([1.0, 1.0, 1.0, -1.0])])
    assert st2.lookahead(0, h3) <= 4, "never more than twice k, whatever the rows"


def test_route_caps_the_lookaheads_share_of_the_readers(monkeypatch: MonkeyPatch) -> None:
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    st["ahead_cap"] = 1
    dst = torch.empty(8, dtype=torch.uint8)
    s.disk_read("a.st", 0, 8, dst, s.DISK_AHEAD, key="p1")
    s.disk_read("a.st", 4096, 8, dst, s.DISK_AHEAD, key="p2")
    t1 = s._disk_take(st)
    assert t1 is not None and t1[0]["key"] == "p1" and st["inflight_ahead"] == 1
    assert s._disk_take(st) is None, "the second prediction waits: the cap holds one reader for the lookahead"
    s.disk_read("a.st", 8192, 8, dst, s.DISK_DEMAND, key="d")
    t2 = s._disk_take(st)
    assert t2 is not None and t2[0]["key"] == "d", "a layer waiting now is not held by the cap"
    assert s._disk_take(st) is None
    assert len(st["queue"]) == 1, "the deferred prediction is back in the queue"
    # with a demand read in flight no prediction starts, whatever the cap; once it lands the prediction goes
    st["inflight_ahead"] = 0
    st["ahead_cap"] = 4
    assert st["inflight_demand"] == 1 and s._disk_take(st) is None, (
        "the lookahead takes the gaps, not a share of a burst"
    )
    st["inflight_demand"] = 0
    t3 = s._disk_take(st)
    assert t3 is not None and t3[0]["key"] == "p2"


def test_padded_slots_read_the_aligned_span_straight_in_and_the_views_find_the_bytes(monkeypatch: MonkeyPatch) -> None:
    """A padded slot gives every part a region on a sector with two sectors of slack: the read is the aligned
    span around the expert's bytes, into the region, and the expert then sits its file offset's misalignment
    into it, where the views pick it up."""
    from btb.engine.experts import _ExpertStore

    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(0, 0))
    per = slot_size(st)
    st.padded = True
    st.stride, st.part_at = _ExpertStore._layout(st.sizes, True)
    assert st.part_at == (0, 4096 * -(-(per // 2 + 8192) // 4096)) and st.stride % 4096 == 0
    st.n_slots = 64
    # a part whose bytes start 1000 bytes into a sector, in a file long enough for the aligned span
    parts = [("gu.st", 1000, per // 2, (1,)), ("dn.st", 4096 * 3 + 500, per // 2, (1,))]
    st._sizes_of = {"gu.st": GB, "dn.st": GB}
    monkeypatch.setattr(st, "_recipe", lambda layer, base: parts)
    assert st._grow(1) > 0
    s = seat(st, (0, 1))
    st._submit(route, parts, 1, s, 0, route.DISK_DEMAND)
    r0, r1 = route.reads[-2], route.reads[-1]
    for r in (r0, r1):
        assert r["off"] % 4096 == 0 and r["n"] % 4096 == 0, "the span read is sector-aligned"
        assert r["dst"].data_ptr() % 4096 == 0, "and lands on a sector of the slot"
    # the file offset of expert 1's gate_up is 1000 + per // 2: its delta is that modulo 4096
    delta0 = (1000 + per // 2) % 4096
    assert r0["off"] == 1000 + per // 2 - delta0 and st.slots[s].delta[0] == delta0
    assert r0["n"] >= per // 2 and r0["n"] - per // 2 < 8192
    # the bytes the drive would write: a pattern at the expert's place inside the span
    r0["dst"].fill_(0)
    r0["dst"][delta0 : delta0 + per // 2] = torch.arange(per // 2, dtype=torch.int64).to(torch.uint8)
    gu, _dn = st._views(s)
    want = torch.arange(per // 2, dtype=torch.int64).to(torch.uint8)
    assert torch.equal(gu.reshape(-1).view(torch.uint8), want), "the view is the expert's own bytes, not the slack"
    # a span that would run past the file's end is cut at it (the reader bounces that one)
    st._sizes_of["dn.st"] = 4096 * 3 + 500 + per // 2 + 100
    st._submit(route, parts, 0, s, 0, route.DISK_DEMAND)
    r = route.reads[-1]
    assert r["off"] + r["n"] == st._sizes_of["dn.st"] and r["n"] % 4096 != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_vram_seats_hold_the_most_ridden_experts_and_serve_them_from_the_card(monkeypatch: MonkeyPatch) -> None:
    from btb.engine.experts import VramSeats

    st, _route, _sm = _store_with_routers(monkeypatch, lookahead=(0, 0))
    per = slot_size(st)
    st.vram = VramSeats(2, per, st.shapes, torch.device("cuda"), min_rides=3, per_pass=1)
    assert st._grow(1) > 0
    s = seat(st, (0, 5))
    st._region(s).fill_(7)
    for _ in range(2):
        ready, _pending = st.get(0, "layers.0.mlp.experts.", [5])
        assert ready[5][0].device.type == "cpu", "not yet ridden enough for a seat"
    ready, _pending = st.get(0, "layers.0.mlp.experts.", [5])
    assert st.rides[(0, 5)] == 3 and (0, 5) in st.vram, "the third ride earns the seat"
    assert ready[5][0].device.type == "cpu", "the ride that earned it is still served from RAM"
    # a second rider, a third, at layer 1: one promotion a pass (a pass turns at layer 0), the seats by last ride
    for e in (6, 7):
        seat(st, (1, e))  # through the store's own transitions: its slot table checks after every wave
        st.rides[(1, e)] = 5
    st.get(1, "layers.1.mlp.experts.", [6])
    assert (1, 6) not in st.vram, "the pass's one promotion was spent on 5"
    ready, _pending = st.get(0, "layers.0.mlp.experts.", [5])
    assert ready[5][0].device.type == "cuda" and int(ready[5][0].view(torch.uint8)[0]) == 7, "then from the card"
    assert st.stat["hit"] == 5
    st.get(1, "layers.1.mlp.experts.", [6])
    assert (1, 6) in st.vram, "the next pass's promotion"
    st.get(0, "layers.0.mlp.experts.", [])
    st.get(1, "layers.1.mlp.experts.", [7])
    assert (1, 7) in st.vram and (0, 5) not in st.vram, "the least recently ridden seat was given up"
    assert st.lookahead(0, torch.ones(1, 4)) == 0 or all(k not in st.vram for k in st.ahead)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_seated_expert_serves_a_multi_row_call_from_the_card(monkeypatch: MonkeyPatch) -> None:
    """the prefill's path (several rows, the host kernel) met a seated expert's card views and handed the host
    kernel a card pointer; the rows of a seated expert go to the card and come back with the same answer - the same
    bits, where the card has btb's kernels: the host's gemv on the card (`gemv_lane16`), the activation on the host"""
    from btb.engine.experts import VramSeats
    from btb.engine.host import _Experts

    st, _route, sm = _store_with_routers(monkeypatch, lookahead=(0, 0))
    hidden, inter = 4, 8
    st.per, st.sizes = (2 * inter * hidden + hidden * inter) * 2, (2 * inter * hidden * 2, hidden * inter * 2)
    st.shapes = (st.sizes[0], (2 * inter, hidden), (hidden, inter))
    st.stride, st.part_at = st._layout(st.sizes, False)
    assert st._grow(2) > 0
    torch.manual_seed(3)
    for e in (5, 6):
        s = seat(st, (0, e))
        st._region(s).view(torch.bfloat16).copy_((torch.randn(st.per // 2) * 0.2).bfloat16())
    st.vram = VramSeats(2, st.per, st.shapes, torch.device("cuda"), min_rides=1, per_pass=4)
    sm.expert_store = st
    sm.expert_stat = {"experts": 0, "bytes": 0, "calls": 0, "s": 0.0}
    sm.expert_trace = None
    mod = _Experts(sm, "layers.0.mlp.experts.", 8, act_fn=torch.nn.functional.silu, layer=0)
    x = (torch.randn(2, hidden) * 0.5).bfloat16().cuda()
    top = torch.tensor([[5, 6], [6, 5]], device="cuda")
    w = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.bfloat16, device="cuda")
    y0 = mod(x, top, w)
    assert (0, 5) in st.vram and (0, 6) in st.vram, "the first ride earned the seats"
    y1 = mod(x, top, w)
    assert y1.shape == x.shape
    from btb.engine.native import Native

    if Native.card_kernels() is not None:
        assert torch.equal(y1, y0), "a seated expert's rows part from its rows off the card"
    torch.testing.assert_close(y1.float().cpu(), y0.float().cpu(), rtol=2e-2, atol=2e-2)


def test_route_two_experts_sharing_a_sector_are_two_reads(monkeypatch: MonkeyPatch) -> None:
    """padded spans of neighbouring small parts are the same bytes of the file into different slots: the queue
    must not fold the second into the first (it did, and the second expert's scales were another expert's)"""
    reads: list[tuple[str, int, int]] = []
    s, st = _route(monkeypatch, reads)
    st["workers"] = [None]
    a = torch.empty(4096, dtype=torch.uint8)
    b = torch.empty(4096, dtype=torch.uint8)
    f1 = s.disk_read("a.st", 0, 4096, a, s.DISK_DEMAND, key=(0, 1))
    f2 = s.disk_read("a.st", 0, 4096, b, s.DISK_DEMAND, key=(0, 2))
    assert f1 is not f2 and len(st["reqs"]) == 2
    f3 = s.disk_read("a.st", 0, 4096, a, s.DISK_AHEAD, key=(0, 1))
    assert f3 is f1, "the same bytes into the same place is one read"


def test_grant_keeps_a_ledger_of_what_it_gave() -> None:
    """every grant is banked by kind and device for the report; a refused request is not"""
    from btb.engine.scheduler import MemoryGrantError

    e = _StubEngine()
    with cuda_stats(free=8 * GB):
        e.scheduler.grant(GB, "kv", requester="a cache", device="cuda")
        e.scheduler.grant(GB // 2, "kv", device="cuda")
        e.scheduler.grant(GB // 4, "table", device="cuda")
        with pytest.raises(MemoryGrantError):
            e.scheduler.grant(9 * GB, "kv", device="cuda")
    assert e.scheduler.granted == {"kv@cuda": GB + GB // 2, "table@cuda": GB // 4}


def test_the_card_reading_is_nvmls_own_for_the_card_at_this_pci_address(monkeypatch: MonkeyPatch) -> None:
    """NVML read in the process, every reading fresh, for the card at CUDA's card's PCI address: NVML numbers the
    cards in its own order, so under CUDA_VISIBLE_DEVICES=1 CUDA's card 0 is NVML's 1 - by index the free memory read
    was another card's"""
    reads = iter([4 * GB, 3 * GB])
    asked: list[str] = []

    def nvml(bus_id: str) -> int:
        asked.append(bus_id)
        return next(reads)

    monkeypatch.setattr(device_mod, "nvml_free_bytes", nvml)
    props = types.SimpleNamespace(pci_domain_id=0, pci_bus_id=0x2B, pci_device_id=0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev=None: props)
    dev = torch.device("cuda", 0)
    assert device_mod._physical_free_bytes(dev) == 4 * GB
    assert device_mod._physical_free_bytes(dev) == 3 * GB, "a second reading is NVML's own, not the first one cached"
    assert asked == ["00000000:2B:00.0"] * 2, asked


def test_a_reservation_allocated_out_of_piecemeal_holds_only_what_is_not_in_use() -> None:
    """a pass's working set is allocated by its own ops, which the ledger never sees one by one: its reservation,
    given a measure of what is live, holds only the rest - the device's free reading counts the live part already,
    and counting it twice once refused a cache's growth mid-chunk the room the card had"""
    e = _StubEngine()
    e.device = Device(e)  # the stub's ledger
    live = [0]
    with cuda_stats(free=8 * GB):
        e.device.reserve("pass", 4 * GB, "cuda", used=lambda: live[0])
        assert e.device.reserved("cuda") == 4 * GB
        live[0] = GB  # a GB of its activations allocated: the free reading has them, the reservation the other 3
        assert e.device.reserved("cuda") == 3 * GB and e.device.spoken_for("cuda") == {"pass": 3 * GB}
        live[0] = 5 * GB  # past what it reserved: it holds nothing, never less
        assert e.device.reserved("cuda") == 0 and e.device.spoken_for("cuda") == {}
        live[0] = -GB  # less than at its start: all of it, never more
        assert e.device.reserved("cuda") == 4 * GB
        e.device.reserve("pass", 4 * GB, "cuda")  # reserved again with no measure: whole, whatever is live
        live[0] = GB
        assert e.device.reserved("cuda") == 4 * GB
        e.device.reserve("pass", 4 * GB, "cuda", used=lambda: live[0])
        e.device.release("pass")
        assert e.device.reserved("cuda") == 0 and not e.device._uses, "a released tag's measure goes with it"


def test_a_grant_draws_on_the_reservation_it_names() -> None:
    """a reservation is room nothing else may take, and the allocation it was made for draws on it: its room the
    caller's own, and what the allocation adds spent from it, so the memory is counted once. A `kv` grant draws on
    the epoch's KV unless it names another, `""` on none (a copy of rows the epoch counted already); one tag holds
    room on the card and the host at once, and lets both go together"""
    from btb.engine.scheduler import EPOCH, MemoryGrantError

    e = _StubEngine()
    e.device = Device(e)  # the stub's ledger, the one the grant reads
    with cuda_stats(free=8 * GB):
        e.device.reserve("sweep", 6 * GB, "cuda")
        e.device.reserve("sweep", GB, "cpu")
        assert e.device.reserved("cuda") == 6 * GB and e.device.reserved("cpu") == GB
        with pytest.raises(MemoryGrantError, match=r"spoken for there: sweep 6.00 GiB"):
            e.scheduler.grant(3 * GB, "table", device="cuda")  # 2 GB free past the sweep's room, named in the refusal
        e.scheduler.grant(3 * GB, "work", device="cuda", draws="sweep")
        assert e.device.reserved("cuda") == 3 * GB, "what it adds comes off the room it drew on"
        e.device.reserve(EPOCH, 4 * GB, "cuda")
        with pytest.raises(MemoryGrantError):
            e.scheduler.grant(3 * GB, "kv", device="cuda", draws="")  # 8 less the sweep's 3 and the epoch's 4
        e.scheduler.grant(3 * GB, "kv", device="cuda", held=GB)  # the epoch's own: 2 GB added, spent from it
        assert e.device.reserved("cuda", but="sweep") == 2 * GB
        e.device.release("sweep")
        assert e.device.reserved("cpu") == 0 and e.device.reserved("cuda") == 2 * GB


def test_the_cold_ring_survives_a_pass_of_the_same_order() -> None:
    """the reader goes on into the next pass: a second `_cold_start` over the same layers finds the ring running
    and keeps it (no second thread, no re-read of a slot still holding its layer); a different order restarts
    it, and `_cold_stop` joins the thread it started"""
    from btb.engine.tiers import ColdRing, _TiersMixin

    class Ring(_TiersMixin):
        def __init__(self, cold: Iterable[int], slots: int) -> None:
            self.cold = set(cold)
            self.mlx = None
            self.scheduler = None  # type: ignore[assignment]
            self.cold_chunk = 0
            self.host: dict[int, Any] = {}
            self.reads: list[str] = []
            self.cold_ring = ColdRing(
                slots=[torch.empty(4096, dtype=torch.uint8) for _ in range(slots)],
                slot_of={i: k % slots for k, i in enumerate(sorted(cold))},
                recipe={i: [(None, f"layer{i}.bin", 0, 4096, 0, "bf16", None)] for i in cold},
            )

        def _cold_read(self, path: str, off: int, nb: int, dst: torch.Tensor) -> None:
            self.reads.append(path)

    r = Ring([1, 3, 5], slots=3)
    r._cold_start(6)
    t1 = r.cold_ring.thread
    assert t1 is not None
    for i in (1, 3, 5):
        assert r.cold_ring.ready[i].wait(5)
    assert len(r.reads) == 3
    for i in (1, 3, 5):
        r._cold_release(i)
    r._cold_start(6)
    assert r.cold_ring.thread is t1 and t1.is_alive(), "the same order keeps the ring running"
    for i in (1, 3, 5):
        assert r.cold_ring.ready[i].wait(5)
    assert len(r.reads) == 3, "a slot still holding its layer is not read again"
    r._cold_start(4)  # layers 1 and 3 only: another shape
    t2 = r.cold_ring.thread
    assert t2 is not None
    assert t2 is not t1 and not t1.is_alive(), "a different order stops the old reader before starting the new"
    r._cold_stop()
    assert r.cold_ring.thread is None and not t2.is_alive()


def _host_readings(
    monkeypatch: MonkeyPatch, *, total: int, free: int, commit: int, working_set: int, pool_held: int = 0
) -> None:
    """the scheduler's host sensors, patched to one reading of the box"""
    from btb.engine import scheduler as S

    monkeypatch.setattr(S, "host_total_bytes", lambda: int(total))
    monkeypatch.setattr(S, "host_free_bytes", lambda: int(free))
    monkeypatch.setattr(S, "host_commit_bytes", lambda: int(commit))
    monkeypatch.setattr(S, "process_working_set_bytes", lambda: int(working_set))
    monkeypatch.setattr(S.pool.POOL, "free_bytes", lambda: int(pool_held))


def test_the_card_margin_is_the_smaller_of_half_a_gb_and_eight_percent() -> None:
    """the default VRAM kept free uses the card as fully as its size allows: 0.5 GB on any card of 6.25 GB or
    more, 8% of a smaller one"""
    assert BatchScheduler.vram_margin_gb(24 * GB) == 0.5 and BatchScheduler.vram_margin_gb(80 * GB) == 0.5
    assert BatchScheduler.vram_margin_gb(4 * GB) == 4 * 0.08 and BatchScheduler.vram_margin_gb(0) == 0.0


def test_the_host_floor_is_a_share_of_the_available_ram_or_the_os_figure_plus_growth(monkeypatch: MonkeyPatch) -> None:
    """the RAM a plan leaves to other programs is a tenth of what is available at load - so it follows the
    free memory, not the box's size - unless the OS's own figure plus the run's growth is more; the available
    memory is the OS's plus the pool's blocks; --ram-reserve names another floor"""
    from btb.engine import scheduler as S
    from btb.engine.scheduler import HostBudget

    MB = 2**20
    monkeypatch.setattr(S, "os_memory_floor", lambda: 100 * MB)
    for total in (64, 512):
        _host_readings(monkeypatch, total=total * GB, free=40 * GB, commit=30 * GB, working_set=2 * GB, pool_held=GB)
        hb = BatchScheduler.measure_host(growth=50 * MB)
        assert isinstance(hb, HostBudget)
        assert (hb.total, hb.available, hb.commit, hb.footprint) == (total * GB, 41 * GB, 30 * GB, 2 * GB)
        assert (hb.os_floor, hb.growth, hb.floor) == (100 * MB, 50 * MB, int(4.1 * GB)) and hb.reserve == hb.floor
        assert hb.spendable == 41 * GB - int(4.1 * GB)
    # little free: the OS's figure plus the growth is the larger and stands
    _host_readings(monkeypatch, total=64 * GB, free=GB, commit=30 * GB, working_set=2 * GB)
    assert BatchScheduler.measure_host(growth=50 * MB).floor == 150 * MB
    monkeypatch.setattr(S, "os_memory_floor", lambda: 0)
    assert BatchScheduler.measure_host(growth=50 * MB).floor == int(0.1 * GB)
    _host_readings(monkeypatch, total=64 * GB, free=40 * GB, commit=30 * GB, working_set=2 * GB, pool_held=GB)
    assert BatchScheduler.measure_host().floor == int(4.1 * GB)
    assert BatchScheduler.measure_host(3.0, growth=50 * MB).floor == 3 * GB
    hb = HostBudget(total=GB, available=GB // 2, commit=GB, footprint=0, os_floor=GB, growth=0, floor=GB)
    assert hb.spendable == 0


def test_the_growth_estimate_comes_from_the_models_shape() -> None:
    """the run's growth past the plan is priced from the checkpoint's own shape: the widest pass's float32
    activations over two layers and its logits, no constant"""
    import types

    dense = types.SimpleNamespace(cfg=types.SimpleNamespace(hidden_size=1024, intermediate_size=3072, vocab_size=50000))
    assert BatchScheduler.growth_estimate(dense) == 2 * 16 * (3 * 1024 + 2 * 3072) * 4 + 16 * 50000 * 4
    moe = types.SimpleNamespace(cfg=types.SimpleNamespace(hidden_size=1024, moe_intermediate_size=512, vocab_size=10))
    assert BatchScheduler.growth_estimate(moe) == 2 * 16 * (3 * 1024 + 2 * 512) * 4 + 16 * 10 * 4
    bare = types.SimpleNamespace(cfg=types.SimpleNamespace(hidden_size=8, vocab_size=1))
    assert BatchScheduler.growth_estimate(bare, rows=1) == 2 * (3 * 8 + 2 * 32) * 4 + 4


class _PlanProbe:
    """a model opened on the CPU as the planner sees it: `plan_budget` records what it was asked and answers
    with a fixed two-layer host placement"""

    weight_map: dict[str, str] = {}
    fam = Family(kind=FamilyKind.QWEN3)
    cfg = types.SimpleNamespace(hidden_size=64, intermediate_size=256, vocab_size=100)

    def __init__(self) -> None:
        self.asked: list[tuple[float, Json]] = []

    def plan_budget(self, ram_gb: float, vram_gb: float, **kw: float | None) -> Json:
        self.asked.append((ram_gb, kw))
        return {
            "head_on_card": True,
            "drafter_on_card": False,
            "resident": [],
            "host": [0, 1],
            "cold": [],
            "warm": [0, 1],
            "prefill_card": False,
            "predicted_ms_per_token": 1.0,
            "bytes": {
                "vram_layers": 0,
                "head": 0,
                "drafter": 0,
                "warm": 10,
                "cold": 0,
                "slots": 2,
                "shadow": 0,
                "templates": 0,
            },
            "caps": {
                "ram_gb": ram_gb,
                "vram_gb": vram_gb,
                "os_reserve_gb": kw["os_reserve_gb"],
                "vram_reserve_gb": kw["vram_reserve_gb"],
            },
        }


class _KvProbe(_PlanProbe):
    """the planner's two prices for one model: with the cache on the card the layers it evicts stream at
    `stream_ms` a token; in RAM every layer is resident and the host's attention reads it at `kv_read_ms`"""

    def __init__(self, stream_ms: float, kv_read_ms: float) -> None:
        super().__init__()
        self.stream_ms, self.kv_read_ms = stream_ms, kv_read_ms

    def plan_budget(self, ram_gb: float, vram_gb: float, **kw: float | None) -> Json:
        out = super().plan_budget(ram_gb, vram_gb, **kw)
        if kw["kv_host"]:
            out.update(resident=[0, 1], host=[], warm=[], predicted_ms_per_token=6.0, kv_read_ms=self.kv_read_ms)
        else:
            out.update(predicted_ms_per_token=self.stream_ms, kv_read_ms=0.0)
        return out


def test_the_plan_puts_the_cache_in_ram_only_where_it_beats_streaming() -> None:
    """unasked, the cache goes to RAM when the layers it evicts stream at more a token than the host's attention
    reads at the full context, else stays on the card and RAM is never priced twice; asked, the answer is the ask"""
    from btb.engine.scheduler import HostBudget

    hb = HostBudget(
        total=64 * GB, available=40 * GB, commit=30 * GB, footprint=2 * GB, os_floor=GB, growth=0, floor=3 * GB
    )
    plan = lambda p, **kw: BatchScheduler.plan_placement(
        p, "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, budget=hb, settle_s=0.0, **kw
    )
    p = _KvProbe(stream_ms=100.0, kv_read_ms=20.0)
    pl = plan(p)
    assert pl.kv_host and pl.resident == (0, 1) and [kw["kv_host"] for _, kw in p.asked] == [False, True]
    assert "cache in RAM" in str(pl)
    p = _KvProbe(stream_ms=20.0, kv_read_ms=100.0)
    pl = plan(p)
    assert not pl.kv_host and pl.host == (0, 1) and "cache in RAM" not in str(pl)
    p = _KvProbe(stream_ms=100.0, kv_read_ms=20.0)
    assert not plan(p, kv_host=False).kv_host and [kw["kv_host"] for _, kw in p.asked] == [False]
    assert plan(_KvProbe(20.0, 100.0), kv_host=True).kv_host


class _ColdProbe(_PlanProbe):
    """a model with a layer on the drive, its one weight file this test module (the drive probe reads it)"""

    def __init__(self, on_disk: bool = True) -> None:
        super().__init__()
        self.dir = ""
        self.weight_map = {"w": __file__} if on_disk else {}

    def plan_budget(self, ram_gb: float, vram_gb: float, **kw: float | None) -> Json:
        out = super().plan_budget(ram_gb, vram_gb, **kw)
        out.update(host=[0, 1], warm=[0], cold=[1], predicted_ms_per_token=1.0 + (kw.get("drive_bps") or 0) / 1e12)
        return out


NVME = {
    "measured": True,
    "fixed_ms": 0.08,
    "single_ms": 2.0,
    "single_gbs": 3.2,
    "copy_ms": 0.4,
    "reps": 3,
    "rates": {1: (3.2, 0.1), 4: (6.0, 0.2), 16: (6.1, 0.3)},
    "seq_gbs": 6.5,
    "big_mb": 6.25,
    "cost_s": 0.4,
}


def test_the_drive_is_probed_once_and_the_plan_and_the_route_share_it(monkeypatch: MonkeyPatch) -> None:
    """a placement that streams layers probes the drive on the model's file, prices the cold tier at its rate
    and keeps the measurement by volume; the engine's Route on the same volume binds it instead of measuring
    again; a simulated drive is probed every time and never kept; a model without files leaves the plan
    unmeasured"""
    from btb.engine.scheduler import DriveBenchmark, HostBudget

    calls = []
    monkeypatch.setattr(DriveBenchmark, "_kept", {})

    def _measure_drive(path: str, clock: Callable[[], float] | None = None) -> Json:
        calls.append(path)
        return dict(NVME)

    monkeypatch.setattr(BatchScheduler, "measure_drive", staticmethod(_measure_drive))
    hb = HostBudget(
        total=64 * GB, available=40 * GB, commit=30 * GB, footprint=2 * GB, os_floor=GB, growth=0, floor=3 * GB
    )
    plan = lambda p, **kw: BatchScheduler.plan_placement(
        p, "cpu", 0.0, packed=False, fp32=True, vram_reserve_gb=0.0, budget=hb, settle_s=0.0, **kw
    )
    p = _ColdProbe()
    pl = plan(p)
    assert calls == [__file__] and pl.drive is not None and pl.drive.measured
    assert pl.drive.bps == 6.1e9, "the burst rate at the rule's depth (16: within the spread of 4's)"
    assert p.asked[-1][1]["drive_bps"] == pl.drive.bps and "the drive 6.10 GB/s" in str(pl)
    assert DriveBenchmark.kept(BatchScheduler._volume(__file__)) is pl.drive
    logs: list[str] = []
    s = BatchScheduler(types.SimpleNamespace(log=logs.append, dev=torch.device("cpu")))
    prof = s.disk(__file__)
    assert calls == [__file__], "the Route bound the plan's measurement, it did not measure again"
    assert prof["depth"] == 16 and s._disk["expect_gbs"] == 6.1 and any("the plan's measurement" in x for x in logs)
    assert plan(_ColdProbe()).drive is pl.drive and calls == [__file__], "a second plan on the volume: kept"
    sim = BatchScheduler(types.SimpleNamespace(log=logs.append, dev=torch.device("cpu")))
    sim._disk_clock = lambda: 0.0
    sim.disk(__file__)
    assert calls == [__file__, __file__] and len(DriveBenchmark._kept) == 1, "simulated: probed, not kept"
    assert plan(_ColdProbe(on_disk=False)).drive is None and plan(_PlanProbe()).drive is None
    assert DriveBenchmark(None, {"measured": False}).bps == 0.0
    assert DriveBenchmark(None, {"measured": True, "rates": {}, "seq_gbs": 2.0, "depth": 4}).bps == 2.0e9


def test_plan_placement_draws_on_the_host_budget(monkeypatch: MonkeyPatch) -> None:
    """the plan's RAM is the budget's available figure, its reserve the budget's floor and nothing else (no
    working room: the footprint one stood for is already out of the available figure), and the plan carries the
    budget it was drawn on; without one the scheduler measures it, with the floor a caller names"""
    from btb.engine.scheduler import HostBudget

    hb = HostBudget(
        total=64 * GB, available=40 * GB, commit=30 * GB, footprint=2 * GB, os_floor=GB, growth=0, floor=3 * GB
    )
    p = _PlanProbe()
    pl = BatchScheduler.plan_placement(
        p, "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, budget=hb, settle_s=0.0
    )
    ram_gb, kw = p.asked[-1]
    assert ram_gb == 40.0 and kw["os_reserve_gb"] == 3.0 and kw["working_ram_gb"] == 0.0
    assert pl.budget is hb and pl.caps.os_reserve_gb == 3.0 and pl.free.ram_gb == 40.0
    _host_readings(monkeypatch, total=64 * GB, free=20 * GB, commit=30 * GB, working_set=GB)
    from btb.engine import scheduler as S

    monkeypatch.setattr(S, "os_memory_floor", lambda: 0)
    pl = BatchScheduler.plan_placement(
        p, "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, os_reserve_gb=1.5, settle_s=0.0
    )
    assert pl.budget is not None and pl.budget.floor == int(1.5 * GB) and pl.free.ram_gb == 20.0
    assert p.asked[-1][1]["os_reserve_gb"] == 1.5
    # nothing named: a tenth of the 20 GB free (the OS's figure is none here and the probe's growth is less)
    pl = BatchScheduler.plan_placement(p, "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, settle_s=0.0)
    assert pl.budget is not None and pl.budget.os_floor == 0
    assert pl.budget.growth == BatchScheduler.growth_estimate(p) < pl.budget.floor == 2 * GB


def test_a_plan_the_host_cannot_carry_is_refused_by_name() -> None:
    """a box with less above its floor than the smallest working set (the ring's slots, the head and drafter on
    the host, the cache's first rows, the staging) gets a refusal that names the figures, not a plan with every
    layer on the drive and nowhere to land them"""
    from btb.engine.scheduler import HostBudget, PlanError

    class Probe(_PlanProbe):
        def plan_budget(self, ram_gb: float, vram_gb: float, **kw: float | None) -> Json:
            out = super().plan_budget(ram_gb, vram_gb, **kw)
            out["head_on_card"] = False
            out["bytes"]["head"] = GB
            out["bytes"]["slots"] = GB // 2
            return out

    short = HostBudget(
        total=4 * GB, available=int(1.9 * GB), commit=int(1.9 * GB), footprint=0, os_floor=0, growth=0, floor=GB
    )
    with pytest.raises(PlanError) as e:
        BatchScheduler.plan_placement(
            Probe(), "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, budget=short, settle_s=0.0
        )
    assert "REFUSED" in str(e.value) and "1.90 GB available" in str(e.value) and "1.50 GB in RAM" in str(e.value)
    enough = HostBudget(total=4 * GB, available=3 * GB, commit=3 * GB, footprint=0, os_floor=0, growth=0, floor=GB)
    pl = BatchScheduler.plan_placement(
        Probe(), "cuda", 11.0, packed=False, fp32=False, vram_reserve_gb=0.5, budget=enough, settle_s=0.0
    )
    assert pl.budget is enough


def test_a_host_grant_past_the_floor_is_refused(monkeypatch: MonkeyPatch) -> None:
    """the floor is the OS's (or --ram-reserve's), not a tenth of the box, so there is no door through it: a host
    request past the RAM free above the floor is refused, one within it is granted and banked"""
    from btb.engine import scheduler as S
    from btb.engine.scheduler import MemoryGrantError

    e = _StubEngine(dev="cpu")
    e.ram_reserve = GB
    monkeypatch.setattr(S, "host_free_bytes", lambda: 2 * GB)
    e.scheduler.grant(GB // 2, "kv", device="cpu")
    with pytest.raises(MemoryGrantError):
        e.scheduler.grant(int(1.5 * GB), "kv", device="cpu")
    assert e.scheduler.granted == {"kv@cpu": GB // 2}


class _RamDevice:
    """the engine's device as `ram_policy` uses it: a free reading under test, and the moves it was asked for"""

    def __init__(self, free: int) -> None:
        self.free_b = free
        self.did: list[str] = []

    def request(self, what: str, fn: Callable[[], object]) -> object:
        self.did.append(what)
        return fn()

    def free(self, device: object = None, unreserved: bool = False) -> int:
        return self.free_b


class _RamEngine(_MemoryMixin):
    """Stub engine for `ram_policy`: three warm layers, a recording ring, and a device whose `free` is the value
    under test."""

    # the stubs stand where the engine's device and cold ring do, so the policy's moves can be read back
    device: _RamDevice  # type: ignore[assignment]
    cold_ring: tuple[str, tuple[int, ...]] | None  # type: ignore[assignment]

    def __init__(self, stored_bytes: int, free: int) -> None:
        self.stored_bytes = stored_bytes
        self.mlx = None
        self.ram_watch = True
        self.ram_state = RamPolicyState(period=0.0)
        self.host = {i: torch.nn.Module() for i in range(3)}
        self.cold = set()
        self._packed = None
        self.device = _RamDevice(free=free)
        self.lines: list[str] = []
        self.cold_ring = None
        self.rebound: list[int] = []

    def log(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))

    def _cold_stop(self) -> None:
        pass

    def _bind_cold(self) -> None:
        self.cold_ring = ("ring", tuple(sorted(self.cold)))

    def _rebind_warm(self, i: int) -> None:
        self.rebound.append(i)

    def _layer_bytes_stored(self, i: int, packed: bool = False) -> int:
        return self.stored_bytes


def _ram_engine(layer_bytes: int = GB, free: int = 10 * GB) -> _RamEngine:
    return _RamEngine(layer_bytes, free)


def _sensors(monkeypatch: MonkeyPatch, deltas: Sequence[int] = (), low: bool = False) -> None:
    """Patch the policy's sensors: a fault counter that grows by each of `deltas` in turn, and a fixed `low`."""
    from btb.engine import memory as M

    total = [0]
    it = iter(deltas)

    def read() -> int:
        total[0] += next(it, 0)
        return total[0]

    monkeypatch.setattr(M, "hard_page_faults", read)
    monkeypatch.setattr(M, "memory_pressure", lambda: {"low": low, "level": 1.0 if low else 0.0})


def test_the_ram_policy_leaves_a_box_with_room_alone_whatever_its_faults(monkeypatch: MonkeyPatch) -> None:
    """Background hard faults (1-6 a reading) with room above the reserve must not shed. A fault-count trigger
    once shed every host layer of the 27B on a box that was not short."""
    e = _ram_engine(free=3 * GB)
    _sensors(monkeypatch, [0] + [1, 4, 2, 6, 3, 1] * 10)
    for _ in range(61):
        e.ram_policy()
    assert not e.cold and not e.device.did and e.ram_state.shed == []
    assert e.ram_state.clean == 61 and e.ram_state.paging == 0


def test_the_ram_policy_sheds_when_the_ledger_reads_no_room_on_two_readings(monkeypatch: MonkeyPatch) -> None:
    """Two consecutive readings of no free memory above the reserve shed the last warm layer; one reading does
    not. Half a layer's room is not enough to regrow."""
    e = _ram_engine(free=0)
    _sensors(monkeypatch, [0, 2, 1, 3])
    e.ram_policy()
    assert not e.cold, "one short reading is not a verdict"
    e.ram_policy()
    assert e.cold == {2} and e.ram_state.shed == [2] and e.device.did == ["ram-shed"]
    assert e.cold_ring == ("ring", (2,)) and "[ram] SHED layer 2" in e.lines[-1]
    assert "0.00 GB free above the reserve, hard faults +2" in e.lines[-1]
    e.device.free_b = GB // 2
    for _ in range(6):
        e.ram_policy()
    assert e.cold == {2} and e.rebound == [], "regrown into half a layer's room"
    e.device.free_b = 0
    e.ram_policy()
    e.ram_policy()
    assert e.cold == {1, 2} and e.ram_state.shed == [2, 1] and e.device.did == ["ram-shed", "ram-shed"]


def test_the_ram_policy_sheds_on_the_os_signal_alone(monkeypatch: MonkeyPatch) -> None:
    """The OS low-memory signal sheds on its own, whatever the ledger reads."""
    e = _ram_engine(free=10 * GB)
    _sensors(monkeypatch, low=True)
    e.ram_policy()
    e.ram_policy()
    assert e.cold == {2} and "the OS short of memory" in e.lines[-1]
    # nothing left to shed
    e.cold = {0, 1, 2}
    e.ram_state.paging = 0
    e.ram_policy()
    e.ram_policy()
    assert e.device.did.count("ram-shed") == 1


def test_the_ram_policy_regrows_after_clean_readings_with_room(monkeypatch: MonkeyPatch) -> None:
    """Regrow after five consecutive readings with room, background faults included. A short reading restarts the
    count."""
    e = _ram_engine(free=0)
    _sensors(monkeypatch, [0, 0, 0, 2, 1, 3, 2, 4, 1, 5, 2, 3, 1, 2, 4, 6])
    e.ram_policy()
    e.ram_policy()
    assert e.cold == {2}
    e.device.free_b = 2 * GB
    for _ in range(4):
        e.ram_policy()
    assert e.cold == {2} and e.rebound == [], "regrown before five clean readings"
    e.device.free_b = 0
    e.ram_policy()  # short
    e.device.free_b = 2 * GB
    for _ in range(4):
        e.ram_policy()
    assert e.cold == {2} and e.rebound == []
    e.ram_policy()  # fifth clean reading
    assert e.cold == set() and e.rebound == [2] and e.device.did[-1] == "ram-regrow" and e.ram_state.shed == []
    assert "[ram] REGROW layer 2" in e.lines[-1]


# --- the serve registry frees the card ---------------------------------------------------------------------
# `ModelRegistry._make_room` evicts before a load on two budgets that coexist: the host RAM the tiered layers
# live in, and the card's VRAM, so a model no longer in use never holds the GPU while another is prompted. The
# VRAM reading is patched; eviction stops as soon as the card fits, so models that fit together stay resident.


class _RegModel:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name, self.closed = name, closed

    def close(self) -> None:
        self.closed.append(self.name)


class _RegEngine:
    """a served engine as the registry's eviction sees it: a footprint and a model to close"""

    def __init__(self, name: str, footprint: int, closed: list[str]) -> None:
        self.footprint = footprint
        self.sm = _RegModel(name, closed)


def _reg_engine(name: str, footprint: int, closed: list[str]) -> Engine:
    return cast("Engine", _RegEngine(name, footprint, closed))


def test_make_room_evicts_the_lru_model_to_free_the_card(monkeypatch: MonkeyPatch) -> None:
    reg = bare_registry()
    closed: list[str] = []
    reg.loaded["a"] = _reg_engine("a", 5 * GB, closed)  # least recently used
    reg.loaded["b"] = _reg_engine("b", 5 * GB, closed)
    monkeypatch.setattr(reg, "_card_free", lambda: (1 + 5 * len(closed)) * GB)  # 1 GB free, +5 GB per eviction
    reg._make_room(6 * GB)  # one eviction frees enough; the second model stays resident
    assert closed == ["a"]
    assert list(reg.loaded) == ["b"]


def test_make_room_leaves_the_card_alone_when_the_new_model_fits(monkeypatch: MonkeyPatch) -> None:
    reg = bare_registry()
    closed: list[str] = []
    reg.loaded["a"] = _reg_engine("a", 2 * GB, closed)
    monkeypatch.setattr(reg, "_card_free", lambda: 8 * GB)
    reg._make_room(3 * GB)
    assert closed == [] and list(reg.loaded) == ["a"]


def test_make_room_does_not_evict_off_the_card_when_vram_cannot_be_read(monkeypatch: MonkeyPatch) -> None:
    reg = bare_registry()
    closed: list[str] = []
    reg.loaded["a"] = _reg_engine("a", 2 * GB, closed)
    monkeypatch.setattr(reg, "_card_free", lambda: None)  # not a CUDA device / reading failed
    reg._make_room(9 * GB)
    assert closed == [] and list(reg.loaded) == ["a"], "no card reading means no card eviction, not evict-all"


def test_make_room_evicts_on_the_host_ram_budget_independently(monkeypatch: MonkeyPatch) -> None:
    reg = bare_registry(budget=8 * GB)  # host RAM budget of 8 GB
    closed: list[str] = []
    reg.loaded["a"] = _reg_engine("a", 5 * GB, closed)
    monkeypatch.setattr(reg, "_card_free", lambda: 100 * GB)  # card is roomy: only the RAM budget should bite
    reg._make_room(5 * GB)  # 5 + 5 > 8 -> evict a
    assert closed == ["a"] and list(reg.loaded) == []


def test_eviction_releases_the_engine_so_the_card_is_actually_reclaimed(monkeypatch: MonkeyPatch) -> None:
    # the OOM regression: closing a model is not enough - the engine's reference cycles hold its VRAM until they
    # are collected, so the next model plans against a card it wrongly thinks is full. Eviction must drop the
    # engine entirely; a live weakref after it means the VRAM is still pinned.
    reg = bare_registry()
    closed: list[str] = []
    reg.loaded["a"] = _reg_engine("a", 5 * GB, closed)
    ref = weakref.ref(reg.loaded["a"])
    monkeypatch.setattr(reg, "_card_free", lambda: 0)  # card full: force the eviction
    reg._make_room(1 * GB)
    assert closed == ["a"], "the evicted engine was closed"
    assert list(reg.loaded) == []
    assert ref() is None, "the engine is collected on eviction, so its VRAM can be reclaimed"


# --- free_bytes clamps the per-process reading to the physical free (Windows WDDM over-reports) ---------------


def test_free_bytes_clamps_the_per_process_view_to_the_physical_free() -> None:
    # mem_get_info reads high (WDDM's per-process figure while another process holds the card); physical is truth
    with cuda_stats(free=10 * GB, physical=5 * GB):
        assert device_mod.free_bytes(torch.device("cuda:0")) == 5 * GB


def test_free_bytes_adds_this_processs_own_reclaimable_pool_over_the_physical_free() -> None:
    with cuda_stats(free=10 * GB, reserved=3 * GB, allocated=1 * GB, physical=5 * GB):
        # the physical 5 GB plus this process's own 2 GB reserved-but-unallocated pool it can reuse
        assert device_mod.free_bytes(torch.device("cuda:0")) == 5 * GB + 2 * GB


def test_free_bytes_holds_the_card_to_what_the_wddm_budget_leaves_this_process() -> None:
    # a game in the foreground: the physical free still reads 5 GB, but Windows keeps only 1 GB more of this
    # process resident before paging (it, or the game) out - the budget's room, this process's own reusable pool on
    # top (the budget's usage counts it whole)
    with cuda_stats(free=10 * GB, reserved=3 * GB, allocated=1 * GB, physical=5 * GB, budget=1 * GB):
        assert device_mod.free_bytes(torch.device("cuda:0")) == 1 * GB + 2 * GB
    # past its budget already: nothing more, but what it holds unused it can still reuse
    with cuda_stats(free=10 * GB, reserved=3 * GB, allocated=1 * GB, physical=5 * GB, budget=-2 * GB):
        assert device_mod.free_bytes(torch.device("cuda:0")) == 2 * GB
    # a budget with more room than the card has free: the physical free stands
    with cuda_stats(free=10 * GB, physical=5 * GB, budget=9 * GB):
        assert device_mod.free_bytes(torch.device("cuda:0")) == 5 * GB


def test_free_bytes_falls_back_to_mem_get_info_when_the_physical_free_is_unreadable() -> None:
    with cuda_stats(free=4 * GB, physical=None):  # no nvidia-smi / not NVIDIA: keep the per-process reading
        assert device_mod.free_bytes(torch.device("cuda:0")) == 4 * GB


def test_a_foreign_model_on_mlx_is_priced_against_the_ram_the_gpu_shares() -> None:
    """`for_model(device="mlx")` prices a transformers model as `btb.plan` prices an MLX load: the cache against
    the host's RAM (Apple silicon's GPU spends the same), no card margin; its grants and free reads take 'mlx'"""
    if not device_mod.mlx_available():
        pytest.skip("MLX is Apple-silicon only")
    sched = BatchScheduler.for_model(model_config(num_hidden_layers=2, vocab_size=1000), device="mlx")
    assert sched.sm.dev == torch.device("cpu") and sched.sm.vram_margin == 0
    assert device_mod.torch_device("mlx") == torch.device("cpu")
    free = sched.free_for("mlx")
    assert free is not None and free > 0
    sched.grant(MB, "kv", requester="a row", device="mlx")
    assert sched.granted == {"kv@cpu": MB}
    assert sched.kv_row_bytes(64) > 0


def test_a_foreign_model_on_the_host_is_priced_in_its_own_dtype_and_prices_no_card() -> None:
    """`for_model` over a torch module on the CPU: its compute dtype is its parameters', no card margin; asked for
    room on a card it has none to price, so the request is not refused; the memory hierarchy is the host's alone
    (the card's levels 0) and read once, and nothing is pinned in a card's L2"""

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = model_config(num_hidden_layers=2, vocab_size=1000)
            self.w = torch.nn.Parameter(torch.zeros(2, dtype=torch.float16))

    sched = BatchScheduler.for_model(Tiny(), device="cpu")
    assert sched.sm.compute_dtype == torch.float16 and sched.sm.vram_margin == 0
    assert sched.free_for("cuda") is None
    sched.grant(GB, "kv", requester="a row", device="cuda")
    assert sched.granted == {}, "a request nothing here can price is let through, not counted"
    cs = sched.caches()
    assert cs["gpu_l2"] == cs["gpu_l2_persist"] == cs["gpu_l2_window"] == 0 and sched.caches() is cs
    assert sched.pin_bytes(MB) == 0


def test_a_foreign_model_on_a_device_btb_cannot_run_is_an_option_error() -> None:
    """a name that is no device, or a card this machine lacks, is refused as `btb.load` refuses it - naming the
    device and what runs here - never a torch error from inside the pricing"""
    from btb.options import BadDevice

    cfg = model_config(num_hidden_layers=2, vocab_size=1000)
    with pytest.raises(BadDevice, match="tpu"):
        BatchScheduler.for_model(cfg, device="tpu")
    if not torch.cuda.is_available():
        with pytest.raises(BadDevice, match="runs here"):
            BatchScheduler.for_model(cfg, device="cuda")


def test_the_store_knows_an_experts_form_before_reading_one(monkeypatch: MonkeyPatch) -> None:
    """a prefill's depot opens before the sweep's first call, at the form of the experts the store will read: the
    store gives it off its layout alone, the same shapes and dtypes a read expert's views have"""
    from btb.engine.host import stored_parts

    st, _sm = expert_store(monkeypatch, object(), n_layers=2, n_experts=4)
    form = st.form()
    assert form is not None
    st._grow(1)
    s = st.free.pop()
    read = stored_parts(*st._views(s))
    assert read is not None and form == tuple((p.shape, p.dtype) for p in read)


# --- the store's rarer transitions ----------------------------------------------------------------------------


def test_the_lines_demote_pop_and_name_their_oldest_seat() -> None:
    """either residency policy: a demoted rider is the next to go, a popped one leaves no trace, `oldest_slot` is
    the next victim's slot (None on an empty line), `items` every seat; a line whose every seat is the call's own
    bumps nobody. A regular demoted goes before the other regulars."""
    from btb.engine.experts import BusPass, Riders

    for cls in (Riders, BusPass):
        line = cls(lambda: 8)
        assert line.oldest_slot() is None, cls.__name__
        for key, s in (("a", 0), ("b", 1), ("c", 2)):
            line.admit(key, s)
        line.demote("c")
        line.demote("nobody")  # not seated: nothing moves
        assert line.oldest_slot() == 2, cls.__name__
        assert dict(line.items()) == {"a": 0, "b": 1, "c": 2}, cls.__name__
        assert line.pop("b") == 1 and "b" not in line and line.pop("b") is None, cls.__name__
        assert line.victim(skip={0, 2}) is None, f"{cls.__name__}: every seat the call's own, yet one was bumped"
        assert line.victim() == ("c", 2), cls.__name__
    bp = BusPass(lambda: 8)
    for key, s in (("x", 5), ("y", 6)):
        bp.admit(key, s)
        assert bp.get(key) == s  # a second ride: a regular
    bp.demote("y")
    assert bp.oldest_slot() == 6 and bp.victim() == ("y", 6), "the demoted regular goes first"


def test_the_slot_tables_check_names_each_broken_invariant(monkeypatch: MonkeyPatch) -> None:
    """`check` over a store holding seats on the line, predictions in the ring and free slots: each invariant
    broken on its own is refused by name, and the table put back passes again"""
    from btb.engine.experts import SlotState

    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0))
    assert st.lookahead(0, torch.ones(1, 4)) == 2  # (1, 7) and (1, 6) predicted
    _ready, pending = st.get(0, "layers.0.mlp.experts.", [0, 1])
    for e, _f, _s in pending:
        route.land((0, e))
    st.check()
    seat0, seat1 = st.ahead[(1, 7)], dict(st.res.items())[(0, 0)]
    free = st.free[0]

    def refused(match: str) -> None:
        with pytest.raises(AssertionError, match=match):
            st.check()

    rec = st.slots.pop(free)
    refused(r"slots recorded \[\d+\] differ from the blocks' live ones")
    st.slots[free] = rec
    st.free.append(free)
    refused("a slot twice in the free list")
    st.free.pop()
    rec.state = SlotState.RESIDENT
    refused(f"slot {free} is resident but free")
    rec.state = SlotState.FREE
    st.ring[free] = None
    refused(f"slot {free} is free but in the ring")
    del st.ring[free]
    st.ahead[(1, 6)], was = seat1, st.ahead[(1, 6)]
    refused(r"prediction \(1, 6\) names slot \d+, which is resident for \(0, 0\)")
    st.ahead[(1, 6)] = was
    st.res.admit((0, 5), seat0)
    refused(r"line seat \(0, 5\) names slot \d+, which is predicted for \(1, 7\)")
    st.res.pop((0, 5))
    st.res.pop((0, 0))
    refused(r"2 predicted slots for 2 predictions, 2 resident for 1 seats on the line")
    st.res.admit((0, 0), seat1)
    st.check()


def test_a_seat_given_up_is_profiled_to_the_call_or_to_the_machine(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """a block given back to the machine takes every seat and prediction in it off its line - an eviction for the
    machine (aux 1) each seat, a prediction's queued reads withdrawn - and is one release in the profile; a seat
    the line gives a later call's miss is that call's eviction (aux 0)"""
    from btb.engine.experts import ExpertProfile

    st, route, sm = _store_with_routers(monkeypatch, lookahead=(2, 0))
    prof = sm.expert_profile = ExpertProfile(str(tmp_path / "p.npz"))
    assert st.lookahead(0, torch.ones(1, 4)) == 2
    _ready, pending = st.get(0, "layers.0.mlp.experts.", [0, 1])
    for e, _f, _s in pending:
        route.land((0, e))
    (b, (_buf, ids)), *more = st.blocks.items()
    assert not more, "the store grew one block"
    st._release_block(b)
    assert not st.slots and not st.res and not st.ahead and not st.ring and not st.free
    assert {(1, 7), (1, 6)} <= set(route.dropped), "a prediction's queued reads outlived its block"
    ev = prof.a[: prof.n]
    assert {(int(r[3]), int(r[4])) for r in ev if r[2] == prof.EVICT and r[10] == 1} == {(0, 0), (0, 1)}
    rel = [r for r in ev if r[2] == prof.RELEASE]
    assert len(rel) == 1 and int(rel[0][4]) == len(ids) and int(rel[0][5]) == len(ids) * slot_size(st)
    # two seats and three experts: the third's miss takes the oldest seat, an eviction for the call
    st.block_max = 2 * slot_size(st)
    assert st._grow(2) == 2
    st.n_slots = st.live()
    for e in (0, 1, 2):
        _ready, pending = st.get(0, "layers.0.mlp.experts.", [e])
        for pe, _f, _s in pending:
            route.land((0, pe))
    ev = prof.a[: prof.n]
    assert [(int(r[3]), int(r[4])) for r in ev if r[2] == prof.EVICT and r[10] == 0] == [(0, 0)]


def test_a_store_with_nobody_seated_gives_back_its_first_block(monkeypatch: MonkeyPatch) -> None:
    """a release under the reserve with the line empty (nothing but free seats) gives back the store's first block,
    and stops once the machine has the bytes"""
    st, state = _store(monkeypatch, free=0)
    state["free"] = st.reserve + st.margin + 10 * st.block_max
    k0, k1 = st._grow(1), st._grow(1)
    assert k0 > 0 and k1 > 0 and list(st.blocks) == [0, 1] and not st.res
    state["free"] = st.reserve  # nothing above the reserve
    assert st.release() == k0
    assert list(st.blocks) == [1], "the first block went back, and only it"


def test_a_missed_experts_price_is_the_probes_else_the_reads_timed(monkeypatch: MonkeyPatch) -> None:
    """`miss_s`: nothing before a read or a probe; the reads timed so far spread over the readers that ran them;
    the drive's probe over both, once there is one"""
    st, _sm = expert_store(monkeypatch, FakeRoute(), n_layers=2, n_experts=4)
    assert st.miss_s() == 0.0
    st.stat["read_s"], st.stat["read_n"] = 0.8, 4
    assert st.miss_s() == pytest.approx(0.8 / 4 / st.readers)
    st.drive = {"big_mb": 6.25, "single_ms": 40.0, "fixed_ms": 5.0}
    assert st.miss_s() == pytest.approx(2 * 0.005 + slot_size(st) * 0.040 / (6.25 * MB))


def test_the_lookahead_passes_over_a_layer_whose_experts_it_cannot_read(monkeypatch: MonkeyPatch) -> None:
    """a layer after this one with a router but no experts the store can read (no checkpoint prefix): its picks
    are skipped and the layer after it is still read ahead"""
    st, route, sm = _store_with_routers(monkeypatch, lookahead=(2, 1))
    del sm.resident[1].mlp.experts.base
    assert st.lookahead(0, torch.ones(1, 4)) == 1
    assert {r["key"] for r in route.reads} == {(2, 7)} and set(st.ahead) == {(2, 7)}


def test_a_ring_slot_the_call_holds_is_never_taken_back(monkeypatch: MonkeyPatch) -> None:
    """`_ring_take` with a ring slot in `skip` (the wave's own): that one is passed over and the next landed
    prediction's slot is given, its prediction forgotten"""
    st, route, _sm = _store_with_routers(monkeypatch, lookahead=(2, 0), ring_n=2)
    assert st.lookahead(0, torch.ones(1, 4)) == 2
    route.land((1, 7))
    route.land((1, 6))
    held, other = list(st.ring)
    other_key = st.slots[other].key
    assert st._ring_take(skip={held}) == other
    assert held in st.ring and other not in st.ring and other_key not in st.ahead
    assert st.stat["ahead_dropped"] == 1
    st._resident(other, (0, 3))  # the call seats its expert there
    st.check()


def test_a_sweeps_end_gives_the_rings_extra_slots_back_and_withdraws_their_reads(monkeypatch: MonkeyPatch) -> None:
    """a layer-by-layer prefill's lookahead grows the ring past `ring_n`, one slot a prediction; the sweep's end
    shrinks it back, newest first: a prediction still queued is withdrawn and forgotten, its slot free again"""
    st, route, sm = _store_with_routers(monkeypatch, lookahead=(1, 0), ring_n=2)
    sm.cfg = types.SimpleNamespace(num_experts_per_tok=6)
    st.sweep_layer = 0
    assert st.lookahead(0, torch.ones(3, 4), sweep=2) == 6, "a sweep reads every expert some row picks"
    assert len(st.ring) == 6
    newest = [st.slots[s].key for s in list(st.ring)[2:]]
    free0 = len(st.free)
    st.sweep_end()
    assert st.sweep_layer == -1 and len(st.ring) == 2
    assert route.dropped == newest[::-1], "the newest predictions withdrawn, newest first"
    assert not any(k in st.ahead for k in newest) and len(st.ahead) == 2
    assert st.stat["ahead_dropped"] == 4 and len(st.free) == free0 + 4
    st.check()


def test_a_call_the_store_cannot_seat_is_refused_whole_or_at_its_first_expert(monkeypatch: MonkeyPatch) -> None:
    """a store of three seats that cannot grow: a call that must be seated whole (the paths that multiply its
    experts together) asking five is refused, naming the count; a store with no seat at all refuses a wave's first
    expert, since a wave must seat one to go on"""
    route = FakeRoute()
    st, _sm = expert_store(monkeypatch, route, n_layers=2, n_experts=8)
    st.block_max = 3 * slot_size(st)
    assert st._grow(3) == 3
    st.n_slots = st.live()
    with pytest.raises(RuntimeError, match="one call needs 5 experts and the store seats 3 of them"):
        st.call(0, "layers.0.mlp.experts.", [0, 1, 2, 3, 4], rows=4).whole()
    empty, _sm2 = expert_store(monkeypatch, FakeRoute(), n_layers=2, n_experts=8)
    empty.n_slots = 0
    with pytest.raises(RuntimeError, match="one call needs 2 experts and the store seats 0 of them"):
        empty.call(0, "layers.0.mlp.experts.", [0, 1], rows=4).wave()
    done = st.call(1, "layers.1.mlp.experts.", [], rows=1)
    assert done.done and done.wave() == ({}, []), "a call with nothing left seats nothing"


def _layered_store(tmp_path: Path, experts_at: Sequence[int]) -> _ExpertStore:
    """a store over a stub model of two layers whose experts (bf16, four of them) are in the checkpoint's header at
    `experts_at` only - a dense layer has none - read through the store's own recipe"""
    from btb.engine import StreamedTextModel
    from btb.engine import experts as experts_mod
    from btb.engine.experts import ExpertProfile

    E, inter, H = 4, 8, 16
    gu, dn = 2 * inter * H * 2, H * inter * 2
    hdr: dict[str, Json] = {}
    for i in experts_at:
        hdr[f"layers.{i}.mlp.experts.gate_up_proj"] = {
            "dtype": "BF16",
            "shape": [E, 2 * inter, H],
            "data_offsets": [0, E * gu],
        }
        hdr[f"layers.{i}.mlp.experts.down_proj"] = {
            "dtype": "BF16",
            "shape": [E, H, inter],
            "data_offsets": [E * gu, E * (gu + dn)],
        }
    sm = stub_engine(
        mlx=None,
        fam=Family(kind=FamilyKind.QWEN3),
        cold_chunk=0,
        expert_profile=ExpertProfile(str(tmp_path / "p.npz")),
        L=2,
        n_experts=E,
        prefix="",
        dir=str(tmp_path),
        weight_map=dict.fromkeys(hdr, "experts.safetensors"),
        _shard=lambda shard: (None, hdr, 8),
        ST_DTYPES=StreamedTextModel.ST_DTYPES,
        scheduler=None,
        resident={},
        host={},
        device=stub_ledger(lambda: 64 * GB, GB),
    )
    return experts_mod._ExpertStore(sm, budget_bytes=64 * MB, reserve_bytes=GB)


def test_a_model_whose_first_layer_is_dense_starts_its_pass_at_the_first_with_experts(tmp_path: Path) -> None:
    """a dense first layer has no experts in the checkpoint: the pass starts at the first layer that has them (the
    profile's step taken there, once a pass), and the experts' form is read off that layer before any call; a model
    with none at all has no form to give"""
    from btb.engine.host import stored_parts

    st = _layered_store(tmp_path, experts_at=[1])
    form = st.form()
    assert form == ((torch.Size([16, 16]), torch.bfloat16), (torch.Size([16, 8]), torch.bfloat16))
    assert st._first_layer() == 1
    prof = st.sm.expert_profile
    st.call(1, "layers.1.mlp.experts.", [0])
    st.call(1, "layers.1.mlp.experts.", [2])
    assert prof.step == 2, "every call at the first layer with experts starts a pass"
    st._grow(1)
    read = stored_parts(*st._views(st.free[-1]))
    assert read is not None and form == tuple((p.shape, p.dtype) for p in read)
    assert _layered_store(tmp_path, experts_at=[]).form() is None


def test_the_expert_shapes_the_card_multiplies_are_read_off_each_layout(monkeypatch: MonkeyPatch) -> None:
    """`mx_shapes` and `f8_shapes`: the logical [2I, H] and [H, I] of an MXFP4 expert in the checkpoint's layout
    (blocks of 32) and in ggml's (gate and up apart), and of an FP8 one beside its scale grids"""
    st, _sm = expert_store(monkeypatch, FakeRoute(), n_layers=1, n_experts=4)
    st.mx, st.ggml = True, False
    st.shapes = ((64, 2, 16), (64, 2), (32, 1, 16), (32, 1))
    assert st.mx_shapes() == ((64, 64), (32, 32))
    st.ggml = True
    st.shapes = ((32, 64), (32, 64), (32, 32))
    assert st.mx_shapes() == ((64, 64), (32, 32))
    st.mx, st.ggml, st.f8 = False, False, True
    st.shapes = ((64, 32), (4, 2), (32, 32), (2, 2))
    assert st.f8_shapes() == ((64, 32), (32, 32))


def test_an_fp8_scale_left_off_its_alignment_reads_the_same_scales(monkeypatch: MonkeyPatch) -> None:
    """an FP8 expert's scale grid sitting a byte off its float32 alignment in the slot (where a direct read's
    padding would leave a misaligned file's bytes): the view reads it through an aligned copy, the same values"""
    from btb.engine.experts import _ExpertStore

    st, _sm = expert_store(monkeypatch, FakeRoute(), n_layers=1, n_experts=4)
    st.mx, st.f8, st.f8_sdt = False, True, torch.float32
    st.shapes = ((32, 16), (2, 1), (16, 16), (1, 1))
    st.sizes = (32 * 16, 2 * 4, 16 * 16, 1 * 4)
    st.padded = True
    st.stride, st.part_at = _ExpertStore._layout(st.sizes, True)
    assert st._grow(1) > 0
    s = st.free[-1]
    st.slots[s].delta = (0, 1, 0, 1)
    region = st._region(s)
    want_gu, want_dn = torch.tensor([[0.5], [2.0]]), torch.tensor([[0.25]])
    for p, w in ((1, want_gu), (3, want_dn)):
        at = st.part_at[p] + 1
        region[at : at + w.numel() * 4] = w.reshape(-1).view(torch.uint8)
    gu, dn = st._views(s)
    assert torch.equal(gu.scales, want_gu) and torch.equal(dn.scales, want_dn)
    assert gu.shape == (32, 16) and dn.shape == (16, 16)


@pytest.mark.parametrize("dt", [torch.float32, torch.float16])
def test_a_store_without_a_route_reads_on_its_own_readers_and_holds_a_wide_expert_as_bf16(
    monkeypatch: MonkeyPatch, tmp_path: Path, dt: torch.dtype
) -> None:
    """A store over a checkpoint whose experts are float32 or float16 and a scheduler with no Route: its own
    readers read each miss off the drive, and every use reads the expert as the bf16 of its values - a waited-for
    read rewritten as it is collected, and a landed one a later call hits rewritten there, once"""
    from btb.engine.native import Native
    from tests.helpers import native_library

    native_library()
    assert Native.read_direct is not None
    st, _sm = expert_store(monkeypatch, object(), n_layers=1, n_experts=4, files=str(tmp_path) + os.sep)
    per = slot_size(st)
    n = per // 2 // dt.itemsize  # a part's values
    st.dt = dt
    st.shapes = (per // 2, (n,), (n,))
    torch.manual_seed(7)
    parts = {}
    for name in ("gu0", "dn0"):
        vals = torch.randn(4 * n).to(dt)
        (tmp_path / f"{name}.st").write_bytes(vals.view(torch.uint8).numpy().tobytes())
        parts[name] = vals.view(4, n).to(torch.bfloat16)
    base = "layers.0.mlp.experts."
    ready, pending = st.get(0, base, [1, 3])
    assert not ready and [e for e, _f, _s in pending] == [1, 3]
    got = st.wait(pending)
    for e in (1, 3):
        assert torch.equal(got[e][0], parts["gu0"][e]) and torch.equal(got[e][1], parts["dn0"][e]), e
    _ready, pending = st.get(0, base, [2])
    pending[0][1].result()  # landed; nobody has collected it
    assert not st.slots[pending[0][2]].bf16
    ready, again = st.get(0, base, [2])
    assert not again and torch.equal(ready[2][0], parts["gu0"][2]) and torch.equal(ready[2][1], parts["dn0"][2])
    assert st.slots[pending[0][2]].bf16, "the hit rewrote its expert as bf16"
    st.close()


def test_the_profile_grows_its_event_array_and_watches_once(tmp_path: Path) -> None:
    """the profile's array doubles when full and keeps every event in order; a second `watch` starts no second
    watchdog"""
    from btb.engine.experts import ExpertProfile

    prof = ExpertProfile(str(tmp_path / "p.npz"), cap=2)
    for i in range(5):
        prof.add(prof.HIT, layer=i)
    assert prof.n == 5 and prof.a.shape[0] >= 5 and [int(x) for x in prof.a[:5, 3]] == [0, 1, 2, 3, 4]

    def watchers() -> int:
        return sum(1 for t in threading.enumerate() if t.name == "gil-watch")

    try:
        prof.watch(every_s=0.05)
        n = watchers()
        prof.watch(every_s=0.05)
        assert watchers() == n
    finally:
        prof._watching = False
