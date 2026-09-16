# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Fast, model-free checks of the engine's plumbing contracts, each with a plain answer: device resolution,
the 12-bit format, the scheduler's arithmetic, the native library's boundary checks, the wheel's package list,
the n-gram tree and span bank, the plan's cold-slot and cache pricing, and the session's prefix reuse. Fakes,
not a loaded model, so everything runs in seconds; the bench tool is in test_bench.py, the machine sensors in
test_sysinfo.py, the commands in test_cli.py, and model discovery in test_hf.py."""

import ctypes
import os
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from pytest import CaptureFixture, MonkeyPatch

from btb import resolve_device
from btb.engine import BatchScheduler, pack_bf16, unpack_bf16
from tests.helpers import ROOT, SchedulerModel, checkout, need_native

if TYPE_CHECKING:
    from btb.engine.tiers import _TiersMixin

# --- device resolution (btb/engine/device.py) ---------------------------------------------------------------


def test_resolve_device_names_itself() -> None:
    assert str(resolve_device("cpu")) == "cpu"
    assert str(resolve_device("CPU")) == "cpu"
    from btb.engine.device import mlx_available
    from btb.options import OptionError

    if mlx_available():
        assert str(resolve_device("mlx")) == "mlx"
    else:  # a device named must be here: never a silent fall to the CPU
        with pytest.raises(OptionError):
            resolve_device("mlx")
    assert str(resolve_device(None)) in ("cuda", "mlx", "cpu")
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            resolve_device("cuda:notanumber")
        with pytest.raises(ValueError):
            resolve_device(f"cuda:{torch.cuda.device_count() + 5}")


# --- the 12-bit format (btb/engine/pack.py) -----------------------------------------------------------------


@pytest.mark.parametrize("shape,scale", [((64, 96), 1.0), ((7, 33), 50.0), ((1, 5), 1e-3), ((128, 128), 0.02)])
def test_pack_unpack_round_trip_is_exact(shape: tuple[int, int], scale: float) -> None:
    g = torch.Generator().manual_seed(int(shape[0] * 1000 + shape[1]))
    t = (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)
    lo, hi4, tbl, esc_idx, esc_val = pack_bf16(t)
    n = t.numel()
    assert lo.size == n and hi4.size == (n + 1) // 2 and tbl.size == 16
    assert esc_idx.size == esc_val.size
    out = unpack_bf16(
        torch.from_numpy(lo),
        torch.from_numpy(hi4),
        torch.from_numpy(tbl),
        n,
        shape,
        torch.from_numpy(esc_idx),
        torch.from_numpy(esc_val),
    )
    assert torch.equal(out.view(torch.int16), t.view(torch.int16)), "the round trip is bit-exact"


def test_pack_escapes_the_rare_high_bytes() -> None:
    # 17 distinct high bytes (a bf16 high byte is the sign and seven exponent bits, so exponents two apart
    # differ in it): 15 fit the table, the 2 rarest are escapes
    vals = torch.tensor([float(2 ** (2 * k)) for k in range(17)] * 4, dtype=torch.bfloat16)
    lo, hi4, tbl, esc_idx, esc_val = pack_bf16(vals)
    assert esc_idx.size == 8 and esc_val.size == 8
    out = unpack_bf16(
        torch.from_numpy(lo),
        torch.from_numpy(hi4),
        torch.from_numpy(tbl),
        vals.numel(),
        vals.shape,
        torch.from_numpy(esc_idx),
        torch.from_numpy(esc_val),
    )
    assert torch.equal(out, vals)


# --- the scheduler (btb/engine/scheduler.py) ---------------------------------------------------------------


def test_scheduler_kv_bytes_count_the_attention_layers_only() -> None:
    s = BatchScheduler(
        SchedulerModel(layer_types=("full_attention", "linear_attention", "full_attention", "linear_attention"))
    )
    # 2 attention layers x (k + v) x 2 kv heads x 64 dims x 2 bytes (bf16)
    assert s._kv_bytes_per_row_token() == 2 * 2 * 2 * 64 * 2
    assert s.kv_row_bytes(100) == 100 * s._kv_bytes_per_row_token()
    assert s.kv_row_bytes(0) == s._kv_bytes_per_row_token(), "at least one position"
    s8 = BatchScheduler(SchedulerModel(layer_types=("full_attention",), kv_bits=8))
    assert s8._kv_bytes_per_row_token() == 2 * 1 * 2 * 64 * 1


def test_scheduler_off_the_card_takes_every_pending_row() -> None:
    s = BatchScheduler(SchedulerModel())
    assert s.free_vram() is None and s.max_batch(1000) is None
    assert s.plan(37, 200) == (37, 200)
    assert (s.batch, s.reserve) == (37, 200)


# --- the native library's boundary checks (native/src/*.rs) ------------------------------------------------


def _native() -> Callable[..., int]:
    lib = ctypes.CDLL(need_native())
    f = lib.btb_gemv_bf16_rows
    f.restype = ctypes.c_int32
    f.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    return f


def test_native_refuses_absurd_thread_counts() -> None:
    f = _native()
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    x = torch.zeros(1, 8)
    y = torch.zeros(1, 4)
    rc = f(w.data_ptr(), 4, 8, x.data_ptr(), 1, y.data_ptr(), 100_000)
    assert rc != 0, (
        "the built library predates the threads limit; rebuild it (python build.py)"
    )  # a refusal that vanished is a regression, not a skip
    assert rc == -6, "ERR_DOMAIN"
    assert f(w.data_ptr(), 4, 8, x.data_ptr(), 1, y.data_ptr(), 0) == 0
    assert f(w.data_ptr(), 4, 8, x.data_ptr(), 1, y.data_ptr(), 2) == 0


def test_native_refuses_misaligned_buffers() -> None:
    f = _native()
    w = torch.zeros(4, 8, dtype=torch.bfloat16)
    y = torch.zeros(1, 4)
    raw = torch.zeros(8 * 4 + 4, dtype=torch.uint8)
    x_off = raw.data_ptr() + 2  # a float32 buffer starting 2 bytes in: not naturally aligned
    rc = f(w.data_ptr(), 4, 8, x_off, 1, y.data_ptr(), 1)
    assert rc != 0, (
        "the built library predates the alignment check; rebuild it (python build.py)"
    )  # a refusal that vanished is a regression, not a skip
    assert rc == -6
    assert f(0, 4, 8, raw.data_ptr(), 1, y.data_ptr(), 1) == -1, "ERR_NULL"
    assert f(w.data_ptr(), 0, 8, raw.data_ptr(), 1, y.data_ptr(), 1) == -6, "a zero dimension"


# --- the wheel's package list (pyproject.toml) --------------------------------------------------------------


def test_every_package_directory_is_in_the_wheel() -> None:
    """pyproject lists the packages by hand: a new subpackage (btb/engine, btb/mlx) that is not in the list imports
    from the checkout and is missing from the wheel - the Docker verification found exactly that."""
    import tomllib

    listed = set(tomllib.load(open(checkout("pyproject.toml"), "rb"))["tool"]["setuptools"]["packages"])
    if not os.path.isdir(os.path.join(ROOT, "btb")):
        pytest.skip("the source tree is not here (an installed wheel): nothing to walk")
    found = set()
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "btb")):
        if "__init__.py" in files:
            found.add(os.path.relpath(dirpath, ROOT).replace(os.sep, "."))
    assert found <= listed, f"packages in the tree but not in pyproject: {sorted(found - listed)}"


# --- the n-gram proposer and the span bank (btb/draft.py) ---------------------------------------------------


def test_ngram_chains_merge_into_one_tree() -> None:
    """The n-gram proposer offers a continuation per order and per recent follower, and the loop verifies them
    as one tree: shared prefixes share nodes, node j carries guesses[j - 1], depth counts from the root, the
    budget caps the nodes."""
    from btb.draft import NGramProposer
    from btb.engine.generate import _chains_tree

    # the tail "1 2 3": order 3 matched at index 5 (continuing 9 9 2 3) and at 0 (4 5 1 2), once each, so
    # the newer first; order 2 ("2 3") continued as 7 (latest), 9 and 4, the 9 and 4 chains repeating the
    # order-3 ones; order 4 ("7 1 2 3") never; the chain proposer takes the top at the longest order
    p = NGramProposer([1, 2, 3, 4, 5, 1, 2, 3, 9, 9, 2, 3, 7, 1, 2, 3], n_max=4, n_min=2)
    chains = p.propose_chains(4)
    assert [c for c, _ in chains] == [[9, 9, 2, 3], [4, 5, 1, 2], [7, 1, 2, 3]], chains
    assert [tag for _, tag in chains] == ["ngram3", "ngram3", "ngram2"]
    assert p.propose_with_source(4)[0] == [9, 9, 2, 3]
    # how often a token followed outranks how recently: 5 was followed by 6 twice and by 7 once (the latest)
    q = NGramProposer([5, 6, 5, 6, 5, 7, 5], n_max=2, n_min=1, followers=2)
    assert [c for c, _ in q.propose_chains(2)] == [[6, 5], [7, 5]]
    assert q.propose_with_source(2)[0] == [6, 5]
    guesses, parents, depth, children, tags = _chains_tree([([9, 9, 2], "a"), ([9, 7], "b"), ([4], "c")], 14)
    assert guesses == [9, 9, 2, 7, 4]
    assert parents == [-1, 0, 1, 2, 1, 0]
    assert depth == [0, 1, 2, 3, 2, 1]
    assert children == {0: [1, 5], 1: [2, 4], 2: [3]}
    assert tags == ["root", "a", "a", "a", "b", "c"]
    g2, p2, d2, _, _ = _chains_tree([([1, 2, 3], "a"), ([4, 5, 6], "b")], 4)
    assert g2 == [1, 2, 3, 4] and p2 == [-1, 0, 1, 2, 0] and d2 == [0, 1, 2, 3, 1]


def test_span_bank_keeps_the_newest_within_its_budget() -> None:
    from btb.draft import NGramProposer, SpanBank

    b = SpanBank(max_tokens=10)
    b.add("a", [1, 2, 3, 4])
    b.add("b", [5, 6, 7])
    b.add("c", [8, 9, 10, 11])  # 11 tokens: the oldest span leaves
    assert [tag for tag, _ in b.spans()] == ["b", "c"] and b.tokens == 7
    b.add("d", list(range(20)))  # larger than the budget: ignored
    assert len(b) == 2
    # the bank drafts: a proposer given the spans continues a seen answer
    p = NGramProposer([5, 6], n_max=4, n_min=2)
    for tag, ids in b.spans():
        p.add_sequence(ids, tag)
    assert p.propose_with_source(3)[0] == [7]


# --- the plan's cold-slot and cache pricing (btb/__init__.py, btb/engine/scheduler.py) ----------------------


def test_cold_slots_follow_the_plan() -> None:
    """the cold reader's ring is as deep as the warm layers' compute needs, within the memory the plan left"""
    from btb.engine.scheduler import Plan, PlanBytes, PlanCaps, PlanFree

    slot = 140 * 2**20

    def plan_with(cold: Sequence[int], ram_gb: float = 12.0) -> Plan:
        # head and drafter on the card, 20 warm layers, two slots reserved, a 1 GB OS reserve; only the cold
        # set and the RAM cap vary between the cases
        cold = tuple(cold)
        return Plan(
            device="cuda",
            resident=(),
            host=cold,
            cold=cold,
            warm=(),
            head_on_card=True,
            drafter_on_card=True,
            prefill_card=False,
            kv_host=False,
            has_mtp=False,
            moe=False,
            predicted_ms_per_token=0.0,
            bytes=PlanBytes(
                vram_layers=0, head=0, drafter=0, warm=20 * slot, cold=0, slots=2 * slot, shadow=0, templates=0
            ),
            caps=PlanCaps(ram_gb=ram_gb, vram_gb=0.0, os_reserve_gb=1.0, vram_reserve_gb=0.0),
            free=PlanFree(vram_gb=0.0, ram_gb=ram_gb, ram_gb_first=ram_gb, settle_s=0.0),
        )

    # 20 warm layers compute in ~56 ms at the plan's 50 GB/s; a cold layer reads in ~40 ms: 2 more slots wanted,
    # 12 - 2 - 1 - 2.7 (warm) - 0.27 (slots) = ~5.7 GB spare holds them
    assert plan_with(range(12)).cold_slots() == 4
    assert plan_with(range(12), ram_gb=6.0).cold_slots() == 2  # ~-0.3 GB spare: nothing extra
    assert plan_with([3, 9]).cold_slots() == 2  # two cold layers: two slots hold them
    assert plan_with([]).cold_slots() == 2


def test_cold_slots_under_a_named_cpu_share() -> None:
    """with --cpu-layers N the ring is the RAM tier of the streamed layers: every cold layer the spare memory holds
    keeps a slot (a slot still holding its layer is not read again), not the two-plus-overlap of a planned
    placement"""
    from btb.engine.scheduler import Plan, PlanBytes, PlanCaps, PlanFree

    slot = 140 * 2**20
    pl = Plan(
        device="cuda",
        resident=(),
        host=(),
        cold=(),
        warm=(),
        head_on_card=True,
        drafter_on_card=True,
        prefill_card=False,
        kv_host=False,
        has_mtp=False,
        moe=False,
        predicted_ms_per_token=0.0,
        bytes=PlanBytes(
            vram_layers=0, head=0, drafter=0, warm=20 * slot, cold=0, slots=2 * slot, shadow=0, templates=0
        ),
        caps=PlanCaps(ram_gb=12.0, vram_gb=0.0, os_reserve_gb=1.0, vram_reserve_gb=0.0),
        free=PlanFree(vram_gb=0.0, ram_gb=12.0, ram_gb_first=12.0, settle_s=0.0),
    )
    # no host share: 12 - 2 - 1 - 0.27 GB spare holds 63 slots, and 61 cold layers each keep one
    assert pl.cold_slots(warm_bytes=0, n_cold=61) == 61
    # the host keeps 20 layers (2.7 GB): 6 GB spare, 43 slots more than the two
    assert pl.cold_slots(warm_bytes=20 * slot, n_cold=61) == 2 + (12 * 2**30 - 3 * 2**30 - 22 * slot) // slot
    assert pl.cold_slots(warm_bytes=0, n_cold=2) == 2


def _probe(L: int = 8, layer_bytes: int = 64 * 2**20, hk: int = 2, hd: int = 64) -> types.SimpleNamespace:
    """a model opened on the CPU as the planner sees it: sizes off the headers, nothing loaded"""
    import types

    cfg = types.SimpleNamespace(
        vocab_size=1000, hidden_size=256, num_attention_heads=4, num_key_value_heads=hk, head_dim=hd
    )
    return types.SimpleNamespace(
        L=L,
        cfg=cfg,
        layer_types=["full_attention"] * L,
        weight_map={},
        _layer_bytes=lambda i: layer_bytes,
        _layer_bytes_stored=lambda i, packed: layer_bytes,
    )


def test_plan_prices_the_cache_for_the_context() -> None:
    """a resident layer costs its weights and its cache: at a long context the card holds fewer layers, and with
    the cache on the host (kv_host) the layers fit again while the RAM is charged for every layer's cache"""
    from btb.engine.tiers import _TiersMixin

    MB = 2**20
    budget = lambda **kw: _TiersMixin.plan_budget(
        cast("_TiersMixin", _probe()),
        ram_gb=64.0,
        vram_gb=1.5 + 0.5 + 0.55,
        packed=False,
        fp32=False,
        drafter=False,
        prefill_card=False,
        os_reserve_gb=1.0,
        vram_reserve_gb=0.5,
        **kw,
    )
    # 0.55 GB for layers: eight at 64 MB with a 2 MB cache each (4096 positions of 2 x 2 heads x 64 x bf16)
    out = budget()
    assert len(out["resident"]) == 8 and out["bytes"]["kv_card"] == 8 * 2 * MB and out["bytes"]["kv_host"] == 0
    # 131072 positions: 64 MB of cache a layer, so a layer costs 128 MB and four fit; the other four's cache is RAM
    out = budget(context=131072)
    assert len(out["resident"]) == 4
    assert out["bytes"]["kv_card"] == 4 * 64 * MB and out["bytes"]["kv_host"] == 4 * 64 * MB
    # the cache on the host: the card holds its eight layers again, the RAM carries every layer's cache
    out = budget(context=131072, kv_host=True)
    assert len(out["resident"]) == 8 and out["bytes"]["kv_card"] == 0 and out["bytes"]["kv_host"] == 8 * 64 * MB
    # what the host's attention reads a token at the full context, priced at the plan's RAM figure
    from btb.engine.tiers import RAM_BPS

    assert out["kv_read_ms"] == 8 * 64 * MB / RAM_BPS * 1e3 and budget()["kv_read_ms"] == 0.0


def test_plan_prices_the_staging_a_streamed_layer_crosses() -> None:
    """once a layer streams to the card, its pinned staging (one layer a type in bf16, and the stored form beside
    it under the 12-bit store) is charged to the RAM: a plan that ends with cold layers is priced again with it"""
    from btb.engine.tiers import _TiersMixin

    MB = 2**20
    # no room on the card for a layer; RAM for seven of the eight 64 MB layers once the working room, the
    # reserve, two slots and the eight caches (16 MB) are taken: 3216 MB spoken for, 480 MB left
    ram_gb = (3216 + 480) / 1024
    run = lambda card, **kw: _TiersMixin.plan_budget(
        cast("_TiersMixin", _probe()),
        ram_gb=ram_gb,
        vram_gb=2.001,
        fp32=False,
        drafter=False,
        prefill_card=card,
        os_reserve_gb=1.0,
        vram_reserve_gb=0.5,
        **kw,
    )
    cpu = run(False, packed=False)
    assert len(cpu["cold"]) == 1 and cpu["bytes"]["staging"] == 0
    card = run(True, packed=False)
    assert card["bytes"]["staging"] == 64 * MB and len(card["cold"]) == 2
    packed = run(True, packed=True)
    assert packed["bytes"]["staging"] == 2 * 64 * MB and len(packed["cold"]) == 3


def test_cold_slots_with_a_host_budget_keep_its_floor_only() -> None:
    """a plan drawn on a host budget leaves the budget's floor free and nothing else: the working room and reserve
    a plan without one kept were standing in for the footprint the available figure already leaves out"""
    from btb.engine.scheduler import HostBudget, Plan, PlanBytes, PlanCaps, PlanFree

    slot = 140 * 2**20

    def plan_with(budget: HostBudget | None, ram_gb: float = 6.0) -> Plan:
        return Plan(
            device="cuda",
            resident=(),
            host=(),
            cold=(),
            warm=(),
            head_on_card=True,
            drafter_on_card=True,
            prefill_card=False,
            kv_host=False,
            has_mtp=False,
            moe=False,
            predicted_ms_per_token=0.0,
            bytes=PlanBytes(vram_layers=0, head=0, drafter=0, warm=0, cold=0, slots=2 * slot, shadow=0, templates=0),
            caps=PlanCaps(ram_gb=ram_gb, vram_gb=0.0, os_reserve_gb=1.0, vram_reserve_gb=0.0),
            free=PlanFree(vram_gb=0.0, ram_gb=ram_gb, ram_gb_first=ram_gb, settle_s=0.0),
            budget=budget,
        )

    # a named share over 61 cold layers, 6 GB of RAM: without a budget 2 GB of working room and the 1 GB reserve
    # are kept (2.7 GB spare, 19 slots more); on a budget with a 1 GB floor only the floor is (4.7 GB, 34 more)
    assert plan_with(None).cold_slots(warm_bytes=0, n_cold=61) == 2 + (6 * 2**30 - 3 * 2**30 - 2 * slot) // slot
    hb = HostBudget(
        total=64 * 2**30, available=6 * 2**30, commit=6 * 2**30, footprint=0, os_floor=2**30, growth=0, floor=2**30
    )
    assert plan_with(hb).cold_slots(warm_bytes=0, n_cold=61) == 2 + (6 * 2**30 - 2**30 - 2 * slot) // slot


# --- the session's prefix reuse (btb/session.py) ------------------------------------------------------------


def test_probe_tail_reads_the_generation_prompt_a_template_keeps_for_the_last_turn_only() -> None:
    """a template that closes the final turn's generation prompt with an empty think block, dropped when the
    turn is followed by another: the probe counts those tokens; a plain template gives 0"""
    from btb.text import probe_tail

    class Tok:
        def __init__(self, think: bool) -> None:
            self.think = think

        def apply_chat_template(
            self,
            msgs: list[dict[str, str]],
            tokenize: bool = False,
            add_generation_prompt: bool = False,
            enable_thinking: bool | None = None,
        ) -> str:
            s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
            if add_generation_prompt:
                s += "<assistant>" + ("<think></think>" if self.think else "")
            return s

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": [ord(c) for c in text]}

    assert probe_tail(Tok(think=True)) == len("<think></think>")
    assert probe_tail(Tok(think=False)) == 0


def test_session_reuses_the_shared_prefix_and_learns_the_tail() -> None:
    """a session opens a new prompt on what its cache holds: the whole cache when the prompt extends it, a crop to
    the shared prefix otherwise, nothing on a fresh one; the template's tail is learned from the divergence"""
    import types

    from btb.session import Session

    class Layer:
        def __init__(self, n: int) -> None:
            self.keys = torch.zeros(1, 2, n, 4)
            self.values = torch.zeros(1, 2, n, 4)

    def engine_and_cache(n: int) -> tuple[types.SimpleNamespace, types.SimpleNamespace]:
        cache = types.SimpleNamespace(layers=[Layer(n), Layer(n)])
        eng = types.SimpleNamespace(layer_types=["full_attention", "full_attention"], L=2)
        return eng, cache

    s = Session()
    assert s.fresh and s.open(None, [1, 2, 3]) == (None, 0, None)
    eng, cache = engine_and_cache(6)
    s.keep([1, 2, 3, 4], [9, 8, 7], cache, None)  # the cache holds the prompt and the answer but its last token
    assert s.ids == [1, 2, 3, 4, 9, 8] and s.n_prompt == 4 and not s.fresh
    # the next turn extends the previous text: the whole cache is reused
    c, reuse, anc = s.open(eng, [1, 2, 3, 4, 9, 8, 5, 6])
    assert c is cache and reuse == 6 and anc is None and cache.layers[0].keys.shape[-2] == 6
    # a prompt that diverges two tokens before the previous prompt's end: a crop to the shared prefix, tail learned
    c, reuse, _ = s.open(eng, [1, 2, 7, 7, 7])
    assert reuse == 2 and cache.layers[1].keys.shape[-2] == 2 and s.tail == 2
    # nothing shared: nothing reused, and the old cache is released before the new prefill, not after it, the
    # drafter's cache with it
    drafter = types.SimpleNamespace(resets=0)
    drafter.reset = lambda: setattr(drafter, "resets", drafter.resets + 1)
    s.dr = drafter
    assert s.open(eng, [5, 5, 5]) == (None, 0, None) and s.cache is None and s.ids == [] and s.fresh
    assert s.dr is None and drafter.resets == 1
    # a prompt that parts from a long previous prompt far from its end is another conversation: the crop still
    # happens, the tail is not re-learned from it (a learned tail of thousands once resumed a whole prompt as one)
    eng, cache = engine_and_cache(300)
    s.keep(list(range(1, 201)), [9, 8], cache, None)
    s.tail = 2
    c, reuse, _ = s.open(eng, [1, 2, 3, 7, 7, 7])
    assert reuse == 3 and s.tail == 2


def test_the_engine_vocabularies_are_spelled_once() -> None:
    """the layer kinds are transformers' own `layer_types` names, the whole of its ALLOWED_ATTN_LAYER_TYPES, so
    any config it accepts reads into the enum; the family kinds and tiers equal their strings, so a report
    reads as before"""
    import json

    from transformers.configuration_utils import ALLOWED_ATTN_LAYER_TYPES

    from btb.kinds import FamilyKind, LayerKind, LayerTier, Tier, UnknownKind

    assert set(LayerKind) == set(ALLOWED_ATTN_LAYER_TYPES), "transformers' list moved: add the new kinds"
    assert LayerKind.of("full_attention") is LayerKind.FULL and LayerKind.of("conv") is LayerKind.CONV
    with pytest.raises(UnknownKind) as e:
        LayerKind.of("not_a_kind")
    assert e.value.text == "not_a_kind" and set(e.value.kinds) == set(LayerKind)
    # a StrEnum member and its string are the same dict key (equal and same hash); mypy cannot express it
    assert {LayerKind.FULL: 1}["full_attention"] == 1 and {"linear_attention": 2}[LayerKind.LINEAR] == 2  # type: ignore[index]
    assert json.dumps({"head": Tier.CARD, "kind": FamilyKind.QWEN3, "tier": LayerTier.COLD}) == (
        '{"head": "card", "kind": "qwen3", "tier": "cold"}'
    )


def test_the_package_root_and_the_cli_stay_torch_free() -> None:
    """`import btb`, the CLI module and the bug-report module load without torch: the memory pool seeds before
    torch is imported, and a missing torch is one line from the CLI, not a traceback (the sampler's export is
    lazy; the vocabularies live in btb/kinds.py, not under the engine package)"""
    import subprocess

    code = (
        "import sys; import btb, btb.cli, btb.feedback, btb.options, btb.kinds, btb.tools; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


def test_the_span_bank_keeps_the_index_and_a_proposer_looks_it_up() -> None:
    """a request's proposer finds a banked continuation without re-adding every span; an evicted span is not
    served; the sweep after enough evictions drops its entries"""
    from btb.draft import NGramProposer, SpanBank

    bank = SpanBank(max_tokens=12, n_max=3)
    bank.add("answer", [7, 8, 9, 10, 11])
    p = NGramProposer([1, 2, 7, 8], n_max=3, n_min=2, bank=bank)
    chains = p.propose_chains(3)
    assert ([9, 10, 11], "seq2") in chains, chains
    ids, src = p.propose_with_source(2)
    assert ids == [9, 10] and src[0] == "answer"
    bank.add("prompt", [20, 21, 22, 23, 24, 25, 26, 27])  # 13 tokens: the first span goes
    assert len(bank) == 1 and bank.lookup(2, (7, 8)) is None
    q = NGramProposer([1, 2, 7, 8], n_max=3, n_min=2, bank=bank)
    assert q.propose_with_source(2) == ([], None)
    bank.add("x", [30, 31, 32, 33])  # 12 tokens: fits; the first span's 5 stale entries stay until a sweep
    assert (7, 8) in bank.ext[2]
    bank.add("y", [40, 41, 42, 43, 44, 45, 46])  # 19: the 8-token span goes, 13 stale > 11 live: swept
    assert (7, 8) not in bank.ext[2] and bank.lookup(2, (21, 22)) is None and bank.lookup(2, (30, 31)) is not None


def test_a_seed_keeps_its_bits_and_a_stream_flag_is_a_boolean() -> None:
    from btb.options import BadValue, check_value
    from btb.serve import _flag

    assert check_value("seed", 2**60 + 1) == 2**60 + 1 and check_value("seed", "7") == 7
    assert _flag({"stream": True}, "stream", False) is True and _flag({}, "stream", True) is True
    assert _flag({"stream": 0}, "stream", True) is False
    with pytest.raises(BadValue):
        _flag({"stream": "false"}, "stream", False)


def test_the_card_without_its_kernels_is_a_reason_kept_and_one_warning(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """no fatbin: `Native.card_kernels()` is None with the reason kept and the path asked once; the engine's
    warning prints once a process, on stderr, whatever the log callback"""
    from btb.engine import cuda as cudamod
    from btb.engine import native as natmod
    from btb.engine.native import Native

    asked = []
    monkeypatch.setattr(Native, "cuda", None)
    monkeypatch.setattr(Native, "cuda_reason", None)
    monkeypatch.setattr(natmod, "kernels_path", lambda: asked.append(1))
    assert Native.card_kernels() is None and "fatbin" in str(Native.cuda_reason)
    assert Native.card_kernels() is None and asked == [1]
    monkeypatch.setattr(cudamod, "_KERNELS_WARNED", False)
    cudamod._card_warning(str(Native.cuda_reason))
    cudamod._card_warning(str(Native.cuda_reason))
    err = capsys.readouterr().err
    assert err.count("WARNING") == 1 and "fatbin" in err


def test_a_packed_tensors_meta_describes_it_and_the_store_is_versioned(tmp_path: Path) -> None:
    """the meta carries the table, the escape count, the rank and the shape, so an entry is read from the shard
    alone; a tensor past the escape indices' reach is refused; a store of another version, or none, is refused
    with a line naming it"""
    import json

    from btb.options import BadPack
    from btb.pack12 import META, VERSION, blob, check_size, check_version, entry

    t = torch.randn(3, 5, 7).to(torch.bfloat16)
    body, meta = blob(t)
    assert body.numel() % 4 == 0 and meta.numel() == META + 8 * 3
    e = entry("pack12-00000.safetensors", 128, bytes(meta.numpy()))
    assert e["shape"] == [3, 5, 7] and e["n"] == 105 and e["lo"] == 105 and e["hi4"] == 53 and e["pad"] == 2
    assert e["off"] == 128 and len(e["table"]) == 16 and e["esc"] * 5 + e["lo"] + e["hi4"] + e["pad"] <= body.numel()
    check_size(2**31 - 1)
    with pytest.raises(ValueError):
        check_size(2**31)
    d = tmp_path / "m-pack12"
    d.mkdir()
    for record, ok in (
        ({"format": "pack12", "version": VERSION, "source": "m"}, True),
        ({"format": "pack12", "source": "m"}, False),
        ({"format": "pack12", "version": VERSION + 1, "source": "m"}, False),
    ):
        (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "btb": record}))
        if ok:
            check_version(str(d))
        else:
            with pytest.raises(BadPack) as ex:
                check_version(str(d))
            assert "re-pack" in str(ex.value)
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    with pytest.raises(BadPack):
        check_version(str(d))
