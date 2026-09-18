# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
# mypy: check-untyped-defs=false
"""The batch scheduler and the two consumers that live off it: `generate_greedy`'s epoch split and `serve`'s
ragged queue, plus the VRAM policy's batched-decode gate. Everything here is device-free and model-free - a
stub engine carries a toy model whose answer is a pure function of a row's real (unpadded) prompt, so a row
decoded alone and the same row decoded inside an epoch must agree token for token, and the CUDA memory
readings are three patched functions. The suite runs in seconds on any machine, with or without a card."""

from __future__ import annotations

import contextlib
import threading
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
from btb.engine.device import Device
from btb.engine.families import Family
from btb.engine.forward import _ForwardMixin
from btb.engine.generate import _GenerateMixin
from btb.engine.memory import RamPolicyState, VramPolicyState, _MemoryMixin
from btb.engine.scheduler import MemoryGrantError
from btb.kinds import FamilyKind, Json, Log, TokenRows, Tokens
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
    slot_size,
    stub_engine,
)

if TYPE_CHECKING:
    from btb.engine.experts import _ExpertStore
    from btb.serve import Engine

TOTAL_VRAM = 12 * GB


# --- the stubs ----------------------------------------------------------------------------------------------


@contextlib.contextmanager
def cuda_stats(
    free: int | list[int] = 8 * GB,
    reserved: int = 0,
    allocated: int = 0,
    raises: BaseException | None = None,
    physical: int | None = 1 << 62,
) -> Iterator[dict[str, int]]:
    """`torch.cuda`'s three memory readings, patched, plus the cross-process physical free (`free_bytes` clamps
    the per-process reading to it; a huge default makes the clamp a no-op so these price the mem_get_info
    arithmetic alone). `free` may be a list: one reading per call, the last repeating, staging a run whose free
    memory moves between plans."""
    saved = (torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated)
    saved_phys = device_mod._physical_free_bytes
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
    try:
        yield seen
    finally:
        torch.cuda.mem_get_info, torch.cuda.memory_reserved, torch.cuda.memory_allocated = saved
        device_mod._physical_free_bytes = saved_phys


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
        self.dev = torch.device(dev)
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
        tap: int | None = None,
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
        self.dev = torch.device("cuda")
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


def test_vram_policy_is_off_when_it_is_not_watching_or_not_on_a_card() -> None:
    e = _PolicyEngine(watch=False)
    with pressure(512), cuda_stats(free=0):
        e.vram_policy(_batched_cache(1))
    assert e.shed_calls == [] and e.lines == []
    host = _PolicyEngine()
    host.dev = torch.device("cpu")
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
    monkeypatch.setattr(experts_mod, "host_free_bytes", lambda: state["free"])
    sm = stub_engine(mlx=None, fam=Family(kind=FamilyKind.QWEN3), cold_chunk=0, expert_profile=None)
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
    for i, s in enumerate(range(grown)):
        st.lru[(0, i)] = s
    st.free = []
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
        st.lru[(0, i)] = i
    st.free = []
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
    s._disk_serve(st, t, lambda path, off, n, dst, chunk, depth: fake_read(path, off, n, dst, chunk))
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
    s = st.free.pop()
    st.lru[(1, 7)] = s
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


# -- the Bus Pass: day riders, regulars and the ghosts that move the split --


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
    s = st.free.pop()
    st._submit(route, parts, 1, s, 0, route.DISK_DEMAND)
    r0, r1 = route.reads[-2], route.reads[-1]
    for r in (r0, r1):
        assert r["off"] % 4096 == 0 and r["n"] % 4096 == 0, "the span read is sector-aligned"
        assert r["dst"].data_ptr() % 4096 == 0, "and lands on a sector of the slot"
    # the file offset of expert 1's gate_up is 1000 + per // 2: its delta is that modulo 4096
    delta0 = (1000 + per // 2) % 4096
    assert r0["off"] == 1000 + per // 2 - delta0 and st.slot_delta[s][0] == delta0
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
def test_vram_seats_hold_the_most_ridden_experts_and_serve_them_from_the_card(monkeypatch: MonkeyPatch) -> None:
    from btb.engine.experts import VramSeats

    st, _route, _sm = _store_with_routers(monkeypatch, lookahead=(0, 0))
    per = slot_size(st)
    st.vram = VramSeats(2, per, st.shapes, torch.device("cuda"), min_rides=3, per_pass=1)
    assert st._grow(1) > 0
    s = st.free.pop()
    st.res.admit((0, 5), s)
    st._region(s).fill_(7)
    for _ in range(2):
        ready, _pending = st.get(0, "layers.0.mlp.experts.", [5])
        assert ready[5][0].device.type == "cpu", "not yet ridden enough for a seat"
    ready, _pending = st.get(0, "layers.0.mlp.experts.", [5])
    assert st.rides[(0, 5)] == 3 and (0, 5) in st.vram, "the third ride earns the seat"
    assert ready[5][0].device.type == "cpu", "the ride that earned it is still served from RAM"
    # a second rider, a third, at layer 1: one promotion a pass (a pass turns at layer 0), the seats by last ride
    for e in (6, 7):
        if st.free:
            s2 = st.free.pop()
        else:
            victim = st.res.victim()
            assert victim is not None
            s2 = victim[1]
        st.res.admit((1, e), s2)
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
    kernel a card pointer; the rows of a seated expert go to the card and come back with the same answer"""
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
        s = st.free.pop()
        st.res.admit((0, e), s)
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
        self.host = {0: None, 1: None, 2: None}
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


def test_free_bytes_falls_back_to_mem_get_info_when_the_physical_free_is_unreadable() -> None:
    with cuda_stats(free=4 * GB, physical=None):  # no nvidia-smi / not NVIDIA: keep the per-process reading
        assert device_mod.free_bytes(torch.device("cuda:0")) == 4 * GB
