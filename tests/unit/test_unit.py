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
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from pytest import CaptureFixture, MonkeyPatch

from btb import resolve_device
from btb.engine import BatchScheduler, pack_bf16, unpack_bf16
from tests.helpers import ROOT, SchedulerModel, checkout, native_library

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


# --- a widened host module (btb/engine/host.py) -------------------------------------------------------------


def test_a_widened_module_computes_in_float32_and_answers_in_the_callers_dtype() -> None:
    """`compute_fp32` on a float32 router: a bf16 activation is multiplied in float32 and its floating results
    come back bf16 (the indices untouched); a float32 activation passes through bit for bit."""
    from btb.engine.host import compute_fp32

    class Router(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 16), requires_grad=False)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.nn.functional.linear(x, self.weight)
            return logits, logits.argmax(-1)

    torch.manual_seed(0)
    router, x = Router(), torch.randn(3, 16)
    want_fp32 = router(x)
    want_bf16 = router(x.bfloat16().float())
    with pytest.raises(RuntimeError, match="same dtype"):
        router(x.bfloat16())
    compute_fp32(router)
    got = router(x)
    assert got[0].dtype == torch.float32 and torch.equal(got[0], want_fp32[0]) and torch.equal(got[1], want_fp32[1])
    logits, idx = router(x.bfloat16())
    assert logits.dtype == torch.bfloat16 and torch.equal(logits, want_bf16[0].bfloat16())
    assert idx.dtype == torch.int64 and torch.equal(idx, want_bf16[1])
    from btb.engine.host import _cast_floats

    # what is neither a floating tensor nor a sequence of them (an int, None, an index tensor) passes through
    t, n, i = torch.ones(2), 3, torch.tensor([1, 2])
    out = _cast_floats((t, n, None, [i, t]), torch.bfloat16)
    assert isinstance(out, tuple) and out[0].dtype == torch.bfloat16 and out[1:3] == (3, None)
    assert out[3][0] is i and out[3][1].dtype == torch.bfloat16


def test_a_reader_thread_writes_an_inference_buffer_and_widens_it_in_place() -> None:
    """`copy_bytes` and `bf16_in_place` from a thread outside inference mode, into a buffer made inside it (a
    bind, or the store's slot, then a reader writing it): torch refuses that write to a plain `copy_`; through
    them it lands, and the float32 or float16 values are rewritten as their bf16 from the buffer's start"""
    import threading

    from btb.engine.host import bf16_in_place, copy_bytes

    for dt in (torch.float32, torch.float16):
        vals = torch.randn(64, generator=torch.Generator().manual_seed(5)).to(dt)
        with torch.inference_mode():
            buf = torch.zeros(vals.numel() * dt.itemsize, dtype=torch.uint8)
        errors: list[BaseException] = []

        def reader(
            vals: torch.Tensor = vals,
            buf: torch.Tensor = buf,
            dt: torch.dtype = dt,
            errors: list[BaseException] = errors,
        ) -> None:
            try:
                with pytest.raises(RuntimeError, match=r"[Ii]nference"):
                    buf.copy_(vals.view(torch.uint8))
                copy_bytes(buf, vals)
                assert torch.equal(buf.view(dt), vals)
                bf16_in_place(buf, dt)
            except BaseException as e:  # carried back to the test's thread
                errors.append(e)

        th = threading.Thread(target=reader)
        th.start()
        th.join()
        assert not errors, errors
        assert torch.equal(buf[: vals.numel() * 2].view(torch.bfloat16), vals.to(torch.bfloat16)), dt


def test_stored_parts_are_the_bytes_the_card_takes_of_each_form() -> None:
    """`stored_parts`: a bf16 expert's two matrices, an MXFP4 one in the checkpoint's layout as its blocks and
    scales, an FP8 one as its bytes and scale grids; nothing for ggml's MXFP4 (no card path takes it), a pair of
    two forms, or an expert already off the host"""
    from btb.engine.host import stored_parts
    from btb.fp8 import F8Weight, quantize
    from btb.mxfp4 import MxWeight, hf_to_ggml
    from tests.helpers import mxfp4_random

    gu, dn = torch.zeros(8, 4, dtype=torch.bfloat16), torch.zeros(4, 4, dtype=torch.bfloat16)
    assert stored_parts(gu, dn) == (gu, dn)
    b, s = mxfp4_random(1, 8, 64)
    mx = MxWeight(torch.from_numpy(b).reshape(-1), torch.from_numpy(s).reshape(-1), 8, 64)
    assert stored_parts(mx, mx) == (mx.blocks, mx.scales, mx.blocks, mx.scales)
    gg = MxWeight.from_ggml(torch.from_numpy(hf_to_ggml(b, s)).reshape(-1), 8, 64)
    assert stored_parts(gg, gg) is None
    q, sc = quantize(torch.randn(32, 32), (16, 16))
    f8 = F8Weight(q, sc, 32, 32)
    assert stored_parts(f8, f8) == (f8.w, f8.scales, f8.w, f8.scales)
    assert stored_parts(gu, mx) is None and stored_parts(f8, mx) is None
    meta = torch.empty(8, 4, dtype=torch.bfloat16, device="meta")
    assert stored_parts(meta, dn) is None, "an expert off the host is the card's already"


def test_the_experts_linear_widens_what_no_kernel_multiplies(monkeypatch: MonkeyPatch) -> None:
    """`_Experts._linear` without the kernel for a form (a native library built without it): MXFP4 in either layout,
    a GGUF's gate and up apart, and FP8 are widened and multiplied in float32 - the dequantized matrix's product,
    and the kernel's to its float32 sums - each call tagged so; a bf16 matrix goes to torch's float32 product at
    `gemm_rows` rows or past, and without the library, which the gemv's rows equal to its sums"""
    import torch.nn.functional as F

    from btb.engine.host import _Experts
    from btb.engine.native import Native
    from btb.fp8 import F8Weight, quantize
    from btb.kinds import PassTag
    from btb.mxfp4 import MxGateUp, MxWeight, hf_to_ggml
    from tests.helpers import mxfp4_random

    native_library()
    tags: set[object] = set()
    ex = _Experts(types.SimpleNamespace(_tag=lambda *t: tags.update(t)), "x.", 2, F.silu)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 64, generator=g)
    b, s = mxfp4_random(5, 32, 64, lo=118, hi=130)
    mx = MxWeight(torch.from_numpy(b).reshape(-1), torch.from_numpy(s).reshape(-1), 32, 64)
    gg = MxWeight.from_ggml(torch.from_numpy(hf_to_ggml(b, s)).reshape(-1), 32, 64)
    q, sc = quantize(torch.randn(32, 64, generator=g), (16, 16))
    f8 = F8Weight(q, sc, 32, 64)
    forms: dict[str, Any] = {"mxfp4": mx, "ggml": gg, "fp8": f8, "gate_up": MxGateUp(gg, gg)}
    kernel = {name: ex._linear(x, w) for name, w in forms.items()}
    assert tags == {PassTag.EXPERT_MXFP4_ASSTORED, PassTag.EXPERT_FP8_ASSTORED}
    for name in ("gemv_mx4", "gemv_mx4_ggml", "gemv_fp8"):
        monkeypatch.setattr(Native, name, None)
    tags.clear()
    for name, w in forms.items():
        got = ex._linear(x, w)
        if isinstance(w, MxGateUp):
            # the gate and the up are each their own widened product, joined: one product of the two stacked is
            # not the same bits on every host's BLAS (arm64's sums a row of a taller matrix otherwise)
            half = F.linear(x, mx.dequantize(torch.float32))
            want = torch.cat([half, half], dim=-1)
        else:
            want = F.linear(x, w.dequantize(torch.float32))
        assert torch.equal(got, want), f"{name}: not the widened matrix's product"
        torch.testing.assert_close(got, kernel[name], rtol=1e-5, atol=1e-5, msg=f"{name}: parts from the kernel")
    assert tags == {PassTag.EXPERT_MXFP4_DEQUANT, PassTag.EXPERT_FP8_WIDENED}
    w = torch.randn(32, 64, generator=g).bfloat16()
    gemv = ex._linear(x, w)
    monkeypatch.setattr(Native, "gemm_rows", 2)
    wide = ex._linear(x, w)
    monkeypatch.setattr(Native, "gemv", None)
    assert torch.equal(ex._linear(x, w), wide) and torch.equal(wide, F.linear(x, w.float()))
    torch.testing.assert_close(gemv, wide, rtol=1e-5, atol=1e-5)


# --- the scheduler (btb/engine/scheduler.py) ---------------------------------------------------------------


def test_gpt_oss_experts_give_a_row_the_same_bits_alone_or_among_others() -> None:
    """gpt-oss's MXFP4 experts with their two biases and clamped gate: a row's routed sum is the same bits in a pass
    of one row (the greedy step's path, `_one_row`) as in a pass of several (a verify's or a prompt's, the per-expert
    loop) - the row invariance speculation stands on. It did not hold: the one-row path added the down bias to the
    float32 product before rounding, the loop (and transformers' reference) to the bf16 product, and gpt-oss-120b's
    speculative answer parted from its greedy one. Compared as bits: a token test on a tiny random model misses an
    ulp that a real model's argmax finds dozens of tokens in."""
    from btb.engine.host import _Experts
    from btb.mxfp4 import BLOCK
    from tests.helpers import mxfp4_random

    native_library()
    E, H, inter, k = 8, 64, 64, 4
    blocks, scales = {}, {}
    for name, (rows, cols) in (("gate_up_proj", (2 * inter, H)), ("down_proj", (H, inter))):
        b, s = mxfp4_random(11 + rows, (E, rows), cols, lo=118, hi=130)
        blocks[name], scales[name] = torch.from_numpy(b), torch.from_numpy(s)
    tensors = {f"x.{n}_blocks": blocks[n] for n in blocks} | {f"x.{n}_scales": scales[n] for n in scales}
    sm = types.SimpleNamespace(
        _tag=lambda *t: None,
        _get=lambda key, *a, **kw: tensors[key],
        expert_store=None,
        expert_probe=None,
        expert_profile=None,
        expert_trace=None,
        mlx=None,
        expert_stat={"experts": 0, "bytes": 0, "calls": 0, "s": 0.0},
    )
    ex = _Experts(sm, "x.", E, None, layer=0, mx=True, biases=True, gate=_Experts.gpt_oss_gate)
    g = torch.Generator().manual_seed(5)
    ex.gate_up_proj_bias = torch.nn.Parameter(torch.randn(E, 2 * inter, generator=g).bfloat16(), requires_grad=False)
    ex.down_proj_bias = torch.nn.Parameter(torch.randn(E, H, generator=g).bfloat16(), requires_grad=False)
    assert blocks["down_proj"].shape[-2] * BLOCK == inter
    T = 6
    x = (torch.randn(T, H, generator=g) * 2).bfloat16()
    top = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), dim=-1).bfloat16()
    with torch.no_grad():
        together = ex(x, top, w)
        for r in range(T):
            alone = ex(x[r : r + 1], top[r : r + 1], w[r : r + 1])
            assert torch.equal(alone[0].view(torch.int16), together[r].view(torch.int16)), (
                f"row {r}: {int((alone[0] != together[r]).sum())} of {H} values part alone from among {T} rows"
            )


def test_the_step_tuner_engages_the_deep_queue_releases_it_and_engages_it_again() -> None:
    """the step loop's tuner (`_Tuner`) over synthetic replay times on its two lanes: alone on the card the usual
    replays are faster and it stays there, saying nothing; when another program's load makes the deep queue faster
    by more than its margin it moves there and says the card is shared, holds while the deep queue stays faster by
    less than the margin, moves back and says so as soon as the usual lane is as fast, and engages again after; a
    free card's drift inside the margin moves nothing; each lane still tried a window in every `explore`; the
    switches written only when the arm changes"""
    from btb.engine.cuda import _Tuner

    arms = [("usual", [1, 0, 0], ("usual", False)), ("queued deep", [2, 0, 0], ("deep", False))]
    tu = _Tuner(arms, torch.device("cpu"), window=2, rounds=2, explore=4, keep=5, margin=0.10, least=3, dwell=6)
    switch = torch.zeros(3, dtype=torch.int32)
    speed = {0: 4.2, 1: 4.4}  # ms a token: alone, the usual lane a little faster
    said: list[str] = []
    writes, runs = 0, [0, 0]

    def run(n: int) -> None:
        nonlocal writes
        for _ in range(n):
            before = tu.cur
            arm = tu.before_replay(switch)
            writes += before != arm
            assert int(switch[0]) == arms[arm][1][0]  # the arm's switches stand on the card
            runs[arm] += 1
            s = tu.after_replay(arm, speed[arm] * 1e-3)
            if s:
                assert s[1]  # a move of the queue is the console's
                said.append(s[0])

    run(80)
    assert tu.chosen == 0 and not said and runs[1] >= 8  # alone: usual, the deep lane still tried
    assert writes < 40  # a switch write at a change of arm, not at every replay
    # a deep lane that loses clearly is explored less and less, and as often as at first when it comes close
    before = runs[1]
    speed.update({1: 6.0})
    run(400)
    assert tu.chosen == 0 and tu.every == 64 and runs[1] - before < 400 // 8
    speed.update({0: 4.6, 1: 4.35})  # a free card's drift: the deep lane 6% ahead, inside the margin - no move
    run(120)
    assert tu.chosen == 0 and not said
    speed.update({0: 9.0, 1: 7.8})  # a game on the card: the deep queue runs in its gaps
    run(80)
    assert tu.chosen == 1 and len(said) == 1 and "shared" in said[0]
    speed.update({0: 8.0})  # the deep lane still faster, by less than the margin: the choice holds
    run(80)
    assert tu.chosen == 1 and len(said) == 1
    speed.update({0: 4.2, 1: 4.4})  # the game gone
    run(120)
    assert tu.chosen == 0 and len(said) == 2 and "no longer pays" in said[1]
    speed.update({0: 9.0, 1: 7.8})  # and back
    run(120)
    assert tu.chosen == 1 and len(said) == 3 and "shared" in said[2]
    assert "*queued deep 7.80" in tu.report()
    # an answer over another length: windows timed at one prefix never meet another's. A short prompt's deep
    # windows against a long one's usual ones read as a shared card on a free one
    speed.update({0: 4.2, 1: 4.4})
    tu.at_context(100)
    run(120)
    assert tu.chosen == 0 and len(said) == 4
    tu.at_context(4000)
    assert all(not m for m in tu.meds)  # the short prompt's windows are gone
    speed.update({0: 5.5, 1: 5.6})  # a free card at 4k: every step slower, the usual lane still the faster
    run(120)
    assert tu.chosen == 0 and len(said) == 4
    tu.at_context(4050)
    assert any(tu.meds)  # the same power of two: the windows stand


def test_the_step_tuner_takes_the_fused_kernels_where_they_win_and_only_logs_it() -> None:
    """the tuner over four lanes - the usual and the deep queue, each in the plain and the fused kernels: alone on
    the card the usual plain lane stands; where another program's load makes the fused kernels the faster (fewer
    edges where it takes the card), it takes the usual fused lane and says so to the log alone; where the deep queue
    in the fused kernels wins, it moves there and the console hears of the queue"""
    from btb.engine.cuda import _Tuner

    lanes = [("usual", False), ("usual", True), ("deep", False), ("deep", True)]
    tu = _Tuner(
        [(str(ln), [0, 0], ln) for ln in lanes], torch.device("cpu"), window=2, rounds=2, explore=4, least=3, dwell=6
    )
    switch = torch.zeros(2, dtype=torch.int32)
    said: list[tuple[str, bool]] = []

    def run(n: int, speed: list[float]) -> None:
        for _ in range(n):
            arm = tu.before_replay(switch)
            s = tu.after_replay(arm, speed[arm] * 1e-3)
            if s:
                said.append(s)

    run(200, [4.2, 4.3, 4.4, 4.5])  # alone
    assert tu.chosen == 0 and not said
    run(200, [9.0, 7.6, 8.8, 8.0])  # beside a game: the fused kernels win, on the usual queue
    assert tu.chosen == 1 and len(said) == 1 and not said[0][1] and "fused kernels run" in said[0][0]
    run(300, [9.0, 7.6, 8.8, 6.5])  # and then the deep queue in them
    assert tu.chosen == 3 and len(said) == 2 and said[1][1] and "shared" in said[1][0] and "fused" in said[1][0]


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
    lib = ctypes.CDLL(native_library())
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_attn_nodes_rows_are_the_one_row_steps(dtype: torch.dtype) -> None:
    """a verify pass's tree nodes through `Native.attn_nodes`: each node over its list of cache rows (the prefix,
    its ancestors, itself) is `Native.attn_decode` over those rows copied out, bit for bit, on a cache sliced out
    of a longer one as the engine slices it; a list of the first n rows is the decode step itself; and a list
    naming a row past the cache is refused with ERR_DOMAIN before anything is written"""
    from btb.engine.native import Native, NativeError

    native_library()
    hq, hk, d, cap, prefix = 24, 2, 256, 96, 70
    g = torch.Generator().manual_seed(7)
    parents = [-1, 0, 1, -1, 3, 1]  # two branches off the prefix, one of them forked
    n_rows = prefix + len(parents)
    kc = torch.randn(hk, cap, d, generator=g).to(dtype)
    vc = torch.randn(hk, cap, d, generator=g).to(dtype)
    k, v = kc[:, :n_rows], vc[:, :n_rows]
    lists = []
    for j in range(len(parents)):
        path, at = [], j
        while at >= 0:
            path.append(prefix + at)
            at = parents[at]
        lists.append(list(range(prefix)) + path[::-1])
    lists.append(list(range(n_rows)))  # the identity list
    T = len(lists)
    q = torch.randn(T, hq, d, generator=g)
    offs = torch.tensor([0] + [sum(len(li) for li in lists[: i + 1]) for i in range(T)], dtype=torch.int32)
    idx = torch.tensor([r for li in lists for r in li], dtype=torch.int32)
    out = torch.full((T, hq, d), float("nan"))
    Native.attn_nodes(q, k, v, offs, idx, 0.0625, out)
    for t, li in enumerate(lists):
        rows = torch.tensor(li)
        step = torch.empty(hq, d)
        Native.attn_decode(q[t], k[:, rows].contiguous(), v[:, rows].contiguous(), 0.0625, step)
        assert torch.equal(out[t], step), f"node {t} is not its one-row step"
    whole = torch.empty(hq, d)
    Native.attn_decode(q[T - 1], k, v, 0.0625, whole)
    assert torch.equal(out[T - 1], whole), "a list of the first n rows is not the decode step over n rows"

    bad = idx.clone()
    bad[-1] = n_rows
    kept = torch.full((T, hq, d), 7.0)
    with pytest.raises(NativeError) as e:
        Native.attn_nodes(q, k, v, offs, bad, 0.0625, kept)
    assert e.value.rc == -6, "a row past the cache is ERR_DOMAIN"
    assert bool((kept == 7.0).all()), "a refused call wrote its output"


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_qwen4_router_rows_are_the_one_row_calls(norm_topk_prob: bool) -> None:
    """Qwen4's host router (families/qwen4/router.py): a row's logits, weights and experts from a call of many rows
    are the one-row call's bit for bit - the verify pass's rows route as their one-token steps do - and they are
    the reference router's over the widened matrix, its routing to the expert and its numbers to float32's
    rounding"""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    from btb.engine.families.qwen4.router import Qwen4Router, install

    native_library()
    E, H, k, T = 24, 192, 4, 17
    g = torch.Generator().manual_seed(11)
    cfg = types.SimpleNamespace(num_experts_per_tok=k, num_experts=E, norm_topk_prob=norm_topk_prob, hidden_size=H)
    ref = Qwen4ExpTextTopKRouter(cfg)
    ref.weight.data = torch.randn(E, H, generator=g).bfloat16()
    mlp = types.SimpleNamespace(gate=ref)
    install(mlp, "model.layers.0.mlp.gate.weight")
    ours = mlp.gate
    assert isinstance(ours, Qwen4Router) and ours.lin.weight.dtype == torch.bfloat16, "the matrix is kept as stored"
    x = torch.randn(T, H, generator=g)
    logits, weights, experts = ours(x)
    for t in range(T):
        lg1, w1, e1 = ours(x[t : t + 1])
        assert torch.equal(logits[t], lg1[0]) and torch.equal(weights[t], w1[0]) and torch.equal(experts[t], e1[0]), (
            f"row {t} of a {T}-row call is not its one-row call"
        )
    install(mlp, "model.layers.0.mlp.gate.weight")
    assert mlp.gate is ours, "a router installed twice is the first one"
    ref.weight.data = ref.weight.data.float()
    r_logits, r_weights, r_experts = ref(x)
    assert torch.equal(experts, r_experts), "the reference router picks other experts"
    torch.testing.assert_close(logits, r_logits, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(weights, r_weights, rtol=1e-5, atol=1e-6)
    # a router held at float32 (a GGUF keeps its routers so) stays the reference's module, computing in float32 and
    # answering in the rows' dtype
    wide = types.SimpleNamespace(gate=ref)
    install(wide, "model.layers.0.mlp.gate.weight")
    assert wide.gate is ref and ref.weight.dtype == torch.float32
    b_logits, b_weights, b_experts = ref(x.bfloat16())
    f_logits, f_weights, f_experts = ref(x.bfloat16().float())
    assert b_logits.dtype == torch.bfloat16 and torch.equal(b_logits, f_logits.bfloat16())
    assert torch.equal(b_weights, f_weights.bfloat16()) and torch.equal(b_experts, f_experts)


def test_the_indexers_masks_are_told_apart_by_shape() -> None:
    """qsa.py: a plain causal mask is every row its prefix, a speculative pass's tree every row the whole prefix
    and its ancestors; a mask of another rank, of rows other than the pass's, or narrower than its rows is
    neither, and a tree whose prefix a row cannot see is no tree - each takes the reference indexer"""
    from btb.engine.families.qwen4.qsa import _plain_causal, _prefix_tree

    S, off = 3, 4
    causal = torch.ones(1, 1, S, off + S, dtype=torch.bool).tril(off)
    assert _plain_causal(causal, S) and _prefix_tree(causal, S) is not None
    tree = causal.clone()
    tree[0, 0, 2, off + 1] = False  # row 2 a sibling of row 1: both under row 0
    assert not _plain_causal(tree, S)
    rows = _prefix_tree(tree, S)
    assert rows is not None and [r.tolist() for r in rows[0]] == [[0], [0, 1], [0, 2]]
    for bad in (causal[0], causal[:, :, :2], torch.ones(1, 1, S, S - 1, dtype=torch.bool)):
        assert not _plain_causal(bad, S) and _prefix_tree(bad, S) is None, tuple(bad.shape)
    hidden = tree.clone()
    hidden[0, 0, 1, 0] = False  # row 1 misses a prefix row
    assert _prefix_tree(hidden, S) is None


def test_the_shared_drafter_leaves_its_steps_to_each_family() -> None:
    """btb/engine/drafter.py's base: no MLX graph of its own, and the step and the MLX graph's parts each family's
    drafter supplies - asked of the base, they refuse"""
    from btb.engine.drafter import MTPDrafter

    dr = MTPDrafter(types.SimpleNamespace(dev=torch.device("cpu")))
    assert not dr._mlx_ready() and not dr._tree_kernel()
    for call in (
        lambda: dr._step(torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, 4), 0),
        dr._mx_dtype,
        lambda: dr._mlx_body(None, None, 0, None),
        lambda: dr._mlx_draw(None, 1, None, ()),
        lambda: dr._mlx_topk(None, 1),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_a_verify_pass_kept_for_an_engine_that_is_gone_restores_nothing() -> None:
    """the DeltaNet's kept inputs of a verify pass hold the engine weakly: after the engine is gone the commit's
    restore is a no-op, not an error"""
    import gc

    from btb.engine.families.qwen4.verify import _PathStep

    class Engine:
        pass

    sm = Engine()
    t = torch.zeros(2, 4)
    step = _PathStep(sm, None, None, t, t, t, t, {})
    del sm
    gc.collect()
    assert step.sm() is None
    step.restore([0, 1])  # nothing to step, and nothing raised


def test_a_deltanet_template_refilled_for_another_layer_rebuilds_its_operands() -> None:
    """a streamed template or a float32 shadow is refilled in place with the next layer's weights and pointed at it
    (`layer_idx`), its storage unchanged: the step's float32 operands are that layer's, rebuilt, never the first
    layer's kept (bf16 weights, copied), and a float32 weight is its own operand, following the refill"""
    from btb.engine.families.qwen4.verify import _consts

    def module(dt: torch.dtype) -> Any:
        ns = types.SimpleNamespace
        t = lambda *s: torch.randn(*s).to(dt)
        return ns(
            layer_idx=3,
            conv1d=ns(weight=t(8, 1, 4), bias=None),
            A_log=t(2),
            dt_bias=t(2),
            norm=ns(weight=t(4), variance_epsilon=1e-6, activation="silu"),
        )

    cpu = torch.device("cpu")
    for dt in (torch.bfloat16, torch.float32):
        la = module(dt)
        first = _consts(la, cpu)["conv_w"].clone()
        other = module(dt)
        for mine, theirs in ((la.conv1d.weight, other.conv1d.weight), (la.A_log, other.A_log)):
            mine.copy_(theirs)
        la.norm.weight.copy_(other.norm.weight)
        la.layer_idx = 11  # `_retarget`
        c = _consts(la, cpu)
        assert torch.equal(c["conv_w"], other.conv1d.weight.squeeze(1).float()), dt
        assert torch.equal(c["a_log"], other.A_log.float()) and torch.equal(c["norm_w"], other.norm.weight.float())
        assert not torch.equal(c["conv_w"], first), dt


def test_the_family_table_and_the_plain_blocks_answers() -> None:
    """families/: a model type btb does not serve is refused by name, and one the kinds declare but no class here
    builds is the table's fault, said so; the activation's name under either key; the plain block's own answers -
    no build of its own, its name, the fused paths its flags give, no drafter, the tensors read with a layer (not
    an FP8 scale, not an expert) and no sweep to open"""
    from btb.engine import families
    from btb.engine.families import FAMILIES, Family, act_name, family
    from btb.kinds import FAMILY_NAMES, FamilyKind
    from btb.options import UnsupportedModelType

    with pytest.raises(UnsupportedModelType):
        family(types.SimpleNamespace(model_type="not_a_model"))
    with pytest.raises(UnsupportedModelType):
        family(types.SimpleNamespace())
    kept = dict(FAMILIES)
    try:
        del families.FAMILIES[FamilyKind.QWEN3]
        with pytest.raises(RuntimeError, match="no Family subclass in families/ builds it"):
            family(types.SimpleNamespace(model_type="qwen3"))
    finally:
        families.FAMILIES.clear()
        families.FAMILIES.update(kept)
    assert act_name(types.SimpleNamespace(hidden_activation="gelu_pytorch_tanh", hidden_act="silu")) == (
        "gelu_pytorch_tanh"
    )
    assert act_name(types.SimpleNamespace(hidden_act="relu")) == "relu"
    assert act_name(types.SimpleNamespace()) == "silu"
    base = Family(kind=FamilyKind.QWEN3)
    with pytest.raises(NotImplementedError):
        Family.build(types.SimpleNamespace())
    assert base.name == FAMILY_NAMES[FamilyKind.QWEN3]
    assert not base.fused_step and not base.card_graph and not base.mega
    assert Family(kind=FamilyKind.QWEN3, kernel_layout=True).fused_step
    assert Family(kind=FamilyKind.QWEN3, sandwich=True, own=True).fused_step
    assert not Family(kind=FamilyKind.QWEN3, sandwich=True, own=True).card_graph
    assert Family(kind=FamilyKind.QWEN3, kernel_layout=True).mega
    assert not Family(kind=FamilyKind.QWEN3, kernel_layout=True, sandwich=True).mega
    from btb.kinds import PassTag

    assert Family(kind=FamilyKind.QWEN3, dense=True).mlx_path is PassTag.MLX_STEP
    assert Family(kind=FamilyKind.QWEN3, sandwich=True).mlx_path is PassTag.MLX_STEP
    assert Family(kind=FamilyKind.QWEN3, hybrid=True).mlx_path is PassTag.MLX_HYBRID
    assert base.mlx_path is PassTag.MLX_PEROP, "a mixture's layers: the per-op path"
    assert base.drafter_cls() is None and base.open_sweep(None, 1, 1, 1) is False  # type: ignore[arg-type]
    assert base.verify_exact(None, None)  # type: ignore[arg-type]
    assert base.dense_key("model.layers.0.self_attn.q_proj.weight")
    for key in (
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.ngram.shard_0.weight_scale",
        "model.layers.0.mlp.experts.gate_up_proj",
    ):
        assert not base.dense_key(key), key


@pytest.mark.parametrize("width", [1, 4, 8, 20, 64])
def test_qwen4_activations_row_by_row_are_the_one_row_calls(width: int) -> None:
    """families/qwen4/rows.py `each`: a verify pass's activations a token row at a time, every row the one-row call
    bit for bit - torch's sigmoid and silu take a vector body or a scalar tail by how many elements travel
    together, which a row among many and a row alone need not share"""
    import torch.nn.functional as F

    from btb.engine.families.qwen4.rows import each

    T = 16
    x = torch.randn(1, T, width, generator=torch.Generator().manual_seed(width)) * 3
    for fn in (torch.sigmoid, F.silu):
        rows = each(fn, x, T)
        assert rows.shape == x.shape
        for t in range(T):
            assert torch.equal(rows[0, t], fn(x[:, t : t + 1].clone())[0, 0]), f"{fn.__name__} row {t} of {width}"


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

    from btb.engine.families import Family
    from btb.kinds import FamilyKind

    cfg = types.SimpleNamespace(
        vocab_size=1000, hidden_size=256, num_attention_heads=4, num_key_value_heads=hk, head_dim=hd
    )
    return types.SimpleNamespace(
        L=L,
        cfg=cfg,
        fam=Family(kind=FamilyKind.QWEN3),  # the plain block's flags: no mixture, no attention index
        layer_types=["full_attention"] * L,
        weight_map={},
        _layer_bytes=lambda i: layer_bytes,
        _layer_bytes_stored=lambda i, packed: layer_bytes,
        _cast_growth=lambda i: 0,  # a bf16 checkpoint: a cold layer lands no wider than it is held
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


def test_plan_keeps_a_sparse_attentions_pooled_keys_on_the_card_with_the_rows_in_ram() -> None:
    """Qwen4's sparse attention indexes its rows: a raw key a row, which lives with the rows, and a pooled key a
    block, which every pass scores whole and so stays on the card - under `kv_host` too, where nothing else of the
    cache is the card's. At a million positions the pooled keys are the card's share of the context"""
    from btb.engine.families.base import _flags
    from btb.engine.families.qwen4.family import Qwen4Family
    from btb.engine.tiers import _TiersMixin
    from btb.kinds import FamilyKind, LayerKind

    MB = 2**20
    p = _probe()
    p.fam = Qwen4Family(kind=FamilyKind.QWEN4, streams=4, **_flags(FamilyKind.QWEN4))
    p.layer_types = [LayerKind.QWEN_SPARSE, LayerKind.LINEAR] * 4
    p.cfg.indexer_compress_ratio, p.cfg.indexer_head_dim, p.cfg.indexer_budget = 4, 128, 2048
    rows = 1 << 20
    pooled, raw = (rows // 4 + 1) * 128 * 2, rows * 128 * 2
    assert p.fam.attn_index_bytes(p.cfg, rows) == (pooled, raw)
    out = _TiersMixin.plan_budget(
        cast("_TiersMixin", p),
        ram_gb=256.0,
        vram_gb=64.0,
        packed=False,
        fp32=False,
        drafter=False,
        prefill_card=False,
        os_reserve_gb=1.0,
        vram_reserve_gb=0.5,
        context=rows,
        kv_host=True,
    )
    kv_row = 2 * 2 * 64 * 2 * rows  # keys and values: 2 heads of 64, bf16
    assert len(out["resident"]) == 8
    assert out["bytes"]["kv_card"] == 4 * pooled, "the four sparse layers' pooled keys, and nothing else"
    assert out["bytes"]["kv_host"] == 4 * (kv_row + raw), "their rows and raw keys in RAM; DeltaNet keeps none"
    # the card's share: 64 MiB a layer at a million positions, a twelfth of what the layer keeps in RAM
    assert pooled == 64 * MB + 256 and (kv_row + raw) / pooled == pytest.approx(12.0, rel=1e-4)
    # a token reads the indexer's budget of those rows and a block's tail - 2052 of a million - not every one
    from btb.engine.tiers import RAM_BPS

    assert p.fam.attn_read_rows(p.cfg, rows) == 2048 + 4
    assert out["kv_read_ms"] == pytest.approx(4 * (kv_row + raw) * 2052 / rows / RAM_BPS * 1e3)
    assert out["kv_read_ms"] < 0.2, "a million positions in RAM priced as a whole read a token"
    # a sparse layer on the host reads every one of its rows (its path grows and re-pools them whole): the budget's
    # discount is the card program's, for the resident layers' rows alone
    few = _TiersMixin.plan_budget(
        cast("_TiersMixin", p),
        ram_gb=256.0,
        vram_gb=1.5 + 0.5 + 0.55,
        packed=False,
        fp32=False,
        drafter=False,
        prefill_card=False,
        os_reserve_gb=1.0,
        vram_reserve_gb=0.5,
        context=rows,
        kv_host=True,
    )
    on_host = [i for i in range(8) if i not in few["resident"] and i % 2 == 0]
    on_card = [i for i in few["resident"] if i % 2 == 0]
    assert on_host and on_card, few["resident"]
    whole, read = len(on_host) * (kv_row + raw), len(on_card) * (kv_row + raw) * 2052 / rows
    assert few["kv_read_ms"] == pytest.approx((whole + read) / RAM_BPS * 1e3)


def test_an_mlx_plan_prices_the_gpu_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """on Apple silicon a pass reads its layers and the head on the GPU from the one memory: the prediction is those
    bytes at the GPU's measured read rate (not the host tier's CPU price and its host head), and the summary puts
    the layers and the head on the GPU; a card's plan reads as it did"""
    import btb.mlx
    from btb.engine.scheduler import BatchScheduler, HostBudget
    from btb.engine.tiers import _TiersMixin

    GB = 2**30
    monkeypatch.setattr(btb.mlx, "read_bps", lambda: 100e9)
    p = _probe()
    p.plan_budget = lambda *a, **kw: _TiersMixin.plan_budget(cast("_TiersMixin", p), *a, **kw)
    hb = HostBudget(total=64 * GB, available=48 * GB, commit=64 * GB, footprint=0, os_floor=0, growth=0, floor=4 * GB)
    pl = BatchScheduler.plan_placement(p, "mlx", 0.0, packed=False, fp32=False, vram_reserve_gb=0.0, budget=hb)
    assert len(pl.warm) == 8 and not pl.cold and pl.gpu_bps == 100e9
    assert pl.predicted_ms_per_token == pytest.approx((pl.bytes.warm + pl.bytes.head) / 100e9 * 1e3)
    s = str(pl)
    assert "8 on the GPU" in s and "head on the GPU" in s and "the GPU reads 100 GB/s" in s and "host" not in s
    cpu = BatchScheduler.plan_placement(p, "cpu", 0.0, packed=False, fp32=False, vram_reserve_gb=0.0, budget=hb)
    assert cpu.gpu_bps is None and "8 host" in str(cpu) and "head host" in str(cpu)


# Qwen3.8-Flash-Next's drafting head as its headers give it (H 2560, four streams, 512 experts): the experts are
# 4.69 GiB of its 4.86, the dense rest 0.17
_Q4_MTP = {
    "mtp.fc_embedding.weight": [2560, 2560],
    "mtp.fc_hidden.weight": [2560, 2560],
    "mtp.hyper_connection_mixer.hc_norm.weight": [10240],
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": [320, 10240],
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.mlp.experts.down_proj": [512, 2560, 640],
    "mtp.layers.0.mlp.experts.gate_up_proj": [512, 1280, 2560],
    "mtp.layers.0.mlp.gate.weight": [512, 2560],
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": [2560, 640],
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": [640, 2560],
    "mtp.layers.0.mlp.shared_expert_gate.weight": [1, 2560],
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": [4, 10240],
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": [10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": [320, 10240],
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": [10240, 320],
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": [640, 2560],
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": [128],
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": [128],
    "mtp.layers.0.self_attn.k_norm.weight": [256],
    "mtp.layers.0.self_attn.k_proj.weight": [512, 2560],
    "mtp.layers.0.self_attn.o_proj.weight": [2560, 6144],
    "mtp.layers.0.self_attn.q_norm.weight": [256],
    "mtp.layers.0.self_attn.q_proj.weight": [12288, 2560],
    "mtp.layers.0.self_attn.v_proj.weight": [512, 2560],
    "mtp.pre_fc_norm_embedding.weight": [2560],
    "mtp.pre_fc_norm_hidden.weight": [10240],
}


def test_the_plan_prices_only_the_drafter_it_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """a drafting head whose layer is a mixture is priced at its dense tensors alone (the experts are the expert
    store's, as every MoE layer's are): the 180B's 0.17 GB, not its 4.86; and it is priced only where the load runs
    it - speculation on, with a family that drafts and verifies on the tier - else nowhere, and the plan says none"""
    import btb.mlx
    from btb.engine.families.base import _flags
    from btb.engine.families.qwen4.family import Qwen4Family
    from btb.engine.scheduler import BatchScheduler, HostBudget
    from btb.engine.state import DRAFT_VOCAB
    from btb.engine.tiers import _TiersMixin
    from btb.kinds import FamilyKind

    GB = 2**30
    monkeypatch.setattr(btb.mlx, "read_bps", lambda: 100e9)
    p = _probe()
    p.cfg.hidden_size = 2560
    p.fam = Qwen4Family(kind=FamilyKind.QWEN4, streams=4, **_flags(FamilyKind.QWEN4))
    hdr = {k: {"shape": shp, "dtype": "BF16"} for k, shp in _Q4_MTP.items()}
    p.weight_map = dict.fromkeys(hdr, "mtp.safetensors")
    p._shard = lambda _f: (None, hdr, 0)
    p._drafter_bytes = lambda: _TiersMixin._drafter_bytes(cast("_TiersMixin", p))
    p.plan_budget = lambda *a, **kw: _TiersMixin.plan_budget(cast("_TiersMixin", p), *a, **kw)
    whole = sum(2 * int(torch.Size(s).numel()) for s in _Q4_MTP.values())
    dense = sum(2 * int(torch.Size(s).numel()) for k, s in _Q4_MTP.items() if ".mlp.experts." not in k)
    assert (round(whole / GB, 2), round(dense / GB, 2)) == (4.86, 0.17)
    assert p._drafter_bytes() == dense == 181_136_896
    slice_b = min(DRAFT_VOCAB, int(p.cfg.vocab_size)) * 2560 * 2
    hb = HostBudget(total=64 * GB, available=48 * GB, commit=64 * GB, footprint=0, os_floor=0, growth=0, floor=4 * GB)
    plan = lambda dev, **kw: BatchScheduler.plan_placement(
        p,
        dev,
        11.0 if dev == "cuda" else 0.0,
        packed=False,
        fp32=False,
        vram_reserve_gb=0.5,
        budget=hb,
        settle_s=0.0,
        **kw,
    )
    on = plan("cuda")
    assert on.has_mtp and on.drafter_on_card and on.bytes.drafter == dense + slice_b
    assert "drafter card" in str(on)
    # the room it leaves is the room the card's layers get: a head priced at its whole 4.86 GB would take it
    assert on.bytes.drafter < 0.2 * GB
    for off in (plan("cuda", speculate=False), plan("mlx"), plan("cpu", speculate=False)):
        assert not off.has_mtp and not off.drafter_on_card and off.bytes.drafter == 0
        assert "drafter none" in str(off)
    host = plan("cpu")
    assert host.has_mtp and not host.drafter_on_card and host.bytes.drafter == dense + slice_b
    assert "drafter host" in str(host)
    # what the card can give up (`memory()`'s sheddable) counts the same head on the card: its dense tensors off the
    # headers, never every `mtp.*` tensor read through `_get` (the store's experts too, an FP8 one widened each call)
    from btb.engine.memory import _MemoryMixin

    card = torch.device("cuda")
    p.dev, p.compute_dtype, p.resident, p.head, p.resident_head = card, None, {}, None, False
    p.aj = types.SimpleNamespace(dev=card)
    p._get = lambda k, *a, **kw: pytest.fail(f"the sheddable bytes read {k}")
    assert _MemoryMixin._sheddable(cast("_MemoryMixin", p), card) == dense


def test_the_report_line_puts_mlx_layers_the_head_and_the_cache_on_the_gpu() -> None:
    from btb.engine.tiers import report_line
    from btb.kinds import Tier

    placement = {"resident": [], "host": [0, 1, 2], "cold": [2], "mlx": [0, 1], "head": Tier.GPU, "kv": Tier.GPU}
    line = report_line({"device": "mlx", "placement": {**placement, "drafter": Tier.NONE}})
    assert "resident 0, gpu 2, host 1, cold 1; head gpu; drafter none; kv gpu" in line


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
    """a decode over a session opens its prompt on what the cache holds: the whole cache when the prompt extends it,
    a crop to the shared prefix otherwise, nothing on a fresh one; the template's tail is learned from the
    divergence. Each opening is a transaction left uncommitted here, as a decode failing there would leave it: the
    session goes back to the point the opening kept"""
    import types

    from transformers.cache_utils import DynamicCache, DynamicLayer

    from btb.engine.drafter import MTPDrafter
    from btb.engine.state import _State
    from btb.session import Session, State

    class Layer(DynamicLayer):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.keys = torch.zeros(1, 2, n, 4)
            self.values = torch.zeros(1, 2, n, 4)
            self.is_initialized = True

    def engine_and_cache(n: int) -> tuple[_State, DynamicCache]:
        cache = DynamicCache()
        cache.layers = [Layer(n), Layer(n)]
        eng = types.SimpleNamespace(layer_types=["full_attention", "full_attention"], L=2)
        return cast("_State", eng), cache

    def held(cache: DynamicCache, i: int) -> int:
        """the rows layer i of the cache holds"""
        cl = cache.layers[i]
        assert isinstance(cl, Layer) and cl.keys is not None
        return int(cl.keys.shape[-2])

    def decoded(s: Session, eng: _State, prompt: list[int], out: list[int], cache: DynamicCache) -> None:
        with s._decoding(eng):
            s._begin_decode(eng, prompt)
            s._commit_decode(prompt, out, cache, None)

    s = Session()
    eng0 = engine_and_cache(1)[0]
    with s._decoding(eng0):
        assert s.fresh and s._begin_decode(eng0, [1, 2, 3]) == (None, 0, None)
    eng, cache = engine_and_cache(6)
    decoded(s, eng, [1, 2, 3, 4], [9, 8, 7], cache)  # the cache holds the prompt and the answer but its last token
    assert s.ids == [1, 2, 3, 4, 9, 8] and s._pending == 7 and s.n_prompt == 4 and s.state is State.PENDING
    # the next turn extends the previous text: the whole cache is reused
    with s._decoding(eng):
        c, reuse, anc = s._begin_decode(eng, [1, 2, 3, 4, 9, 8, 7, 5, 6])
        assert c is cache and reuse == 6 and anc is None and held(cache, 0) == 6
    assert s.tokens == [1, 2, 3, 4, 9, 8, 7], "an opening left uncommitted changed the session"
    # a prompt that diverges two tokens before the previous prompt's end: a crop to the shared prefix, tail learned;
    # left uncommitted, the session is the shared prefix with its last token drawn (the rows past it were replaced)
    with s._decoding(eng):
        c, reuse, _ = s._begin_decode(eng, [1, 2, 7, 7, 7])
        assert reuse == 2 and held(cache, 1) == 2 and s.tail == 2
    assert s.tokens == [1, 2] and s.state is State.PENDING and held(cache, 1) == 1
    # nothing shared: nothing reused, and the old cache is released before the new prefill, not after it, the
    # drafter's cache with it
    drafter = types.SimpleNamespace(resets=0)
    drafter.reset = lambda: setattr(drafter, "resets", drafter.resets + 1)
    s.dr = cast("MTPDrafter", drafter)
    with s._decoding(eng):
        assert s._begin_decode(eng, [5, 5, 5]) == (None, 0, None) and s.cache is None and s.ids == [] and s.fresh
    assert s.dr is None and drafter.resets == 1 and s.state is State.EMPTY
    # a prompt that parts from a long previous prompt far from its end is another conversation: the crop still
    # happens, the tail is not re-learned from it (a learned tail of thousands once resumed a whole prompt as one)
    eng, cache = engine_and_cache(201)
    decoded(s, eng, list(range(1, 201)), [9, 8], cache)
    s.tail = 2
    with s._decoding(eng):
        c, reuse, _ = s._begin_decode(eng, [1, 2, 3, 7, 7, 7])
        assert reuse == 3 and s.tail == 2


def test_the_engine_vocabularies_are_spelled_once() -> None:
    """the layer kinds are transformers' own `layer_types` names, the whole of its ALLOWED_ATTN_LAYER_TYPES read
    through their legacy names, so any config it accepts - under the names of transformers 5.16-5.17 or 5.18 on -
    reads into the enum, and a kind handed back to it is the installed version's name; the family kinds and tiers
    equal their strings, so a report reads as before"""
    import json

    from transformers.configuration_utils import ALLOWED_ATTN_LAYER_TYPES

    from btb.kinds import LEGACY, FamilyKind, LayerKind, LayerTier, Tier, UnknownKind

    assert {LayerKind.of(t) for t in ALLOWED_ATTN_LAYER_TYPES} == set(LayerKind), (
        "transformers' list moved: add the new kinds"
    )
    assert all(LayerKind.of(old) is LayerKind(new) for old, new in LEGACY.items())
    assert LayerKind.of("qwen_sparse_attention") is LayerKind.of("indexed_attention") is LayerKind.QWEN_SPARSE
    assert LayerKind.QWEN_SPARSE.hf_name() in ALLOWED_ATTN_LAYER_TYPES, "a drafter's config named its layer unknown"
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


def test_the_host_memory_a_call_frees_goes_back_to_the_machine() -> None:
    """`import btb` has torch's CPU allocator give freed memory back as it goes (`MIMALLOC_PURGE_DELAY`, set before
    torch loads): by default mimalloc kept it committed to the process for good, and a 40k prompt's 2.5 GB of host
    rows outlived the answer, refusing the next call's prefill on a machine short of commit. A value the caller set
    stands. On Windows, where torch's allocator is mimalloc, 1 GB freed is back in the machine's commit at once"""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "MIMALLOC_PURGE_DELAY"}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    code = (
        "import gc, os, btb, torch; from btb.sysinfo import host_commit_bytes as c; "
        "a = c(); t = torch.ones(1 << 28); held = c(); del t; gc.collect(); "
        "print(os.environ['MIMALLOC_PURGE_DELAY'], a - held, a - c())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, check=True)
    delay, took, kept = out.stdout.split()
    assert delay == "0", out.stdout + out.stderr
    if sys.platform == "win32":
        GB = 1 << 30
        assert int(took) > GB // 2, "the 1 GB tensor was never committed: the reading cannot tell"
        assert int(kept) < GB // 4, f"{int(kept) / GB:.2f} GB of the freed tensor stayed committed"
    env["MIMALLOC_PURGE_DELAY"] = "7"
    out = subprocess.run(
        [sys.executable, "-c", "import os, btb; print(os.environ['MIMALLOC_PURGE_DELAY'])"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert out.stdout.strip() == "7", "a value the caller set was overridden"


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


def test_a_seed_keeps_its_bits() -> None:
    from btb.options import check_value

    assert check_value("seed", 2**60 + 1) == 2**60 + 1 and check_value("seed", "7") == 7


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


def test_a_sampled_draw_is_held_to_the_oracle_only_where_the_engine_computes_in_fp32() -> None:
    """the reference computes in fp32; in bf16 a flat-logit near-tie at the sampler's cut falls either way"""
    from types import SimpleNamespace

    from btb.engine.model import StreamedTextModel
    from tests.cert import oracle

    def model(dtype: torch.dtype | None) -> StreamedTextModel:
        return cast(StreamedTextModel, SimpleNamespace(compute_dtype=dtype))

    assert oracle.holds_sampled(model(torch.float32))
    assert not oracle.holds_sampled(model(None))
    assert not oracle.holds_sampled(model(torch.bfloat16))
