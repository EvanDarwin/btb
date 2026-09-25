# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The one bench a person or an agent runs on any machine - it times whatever the box can, skipping the rest:

    pytest bench/e2e_bench.py --benchmark-json=e2e.json -q      # (needs the tiny fixtures + a built btb)

Two kinds of timing, all through pytest-benchmark (a stable, statistically-sound utility - no hand-rolled loop):

  * the end-to-end generate loop per family (dense, hybrid, the two MoE layouts, Gemma's sandwich/dual-rope,
    gpt-oss sinks) on each device the box has - cpu always, mlx on Apple silicon, cuda on a card;
  * the per-op kernels: the MLX Metal quant matvecs and attention decode (Apple silicon) and the CUDA card
    gemv (a CUDA box). A kernel's runtime does not depend on the weight VALUES, so these use random inputs - no
    fixture or real model needed.

The native CPU ops live in Rust criterion (`native/benches/*.rs`, `cargo bench`) - a different toolchain, not
pytest. `python -m bench.report` compares two runs of both (the base-vs-PR deltas the PR comment renders),
grouped by each entry's bench group.

Not part of `pytest tests` (bench/ is outside testpaths); the bench workflow runs it explicitly.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

import btb
from btb import mlx as mlxdev
from btb.engine.native import Native
from btb.kinds import Quant, QuantClass, latt_backend_key, quants_of
from tests.cert.spec import FIXTURE_STEM, Hardware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if TYPE_CHECKING:
    import mlx.core as mx_

    from btb.engine.native import _Cuda

    Matvec = Callable[[mx_.array, mx_.array, int, int], mx_.array]  # (w_bytes, x, rows, cols) -> out

# per-op benches gate on their backend; the e2e loop runs on every device the box has (below)
mlx_only = pytest.mark.skipif(not btb.mlx_available(), reason="MLX is Apple-silicon only")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="the card kernels need a CUDA GPU")

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
# one fixture per served family, so the sweep hits every architecture's forward in ~4 tokens
FAMILIES = tuple(FIXTURE_STEM.values())
DEVICES: list[tuple[str, dict[str, str]]] = [(Hardware.CPU, {"device": Hardware.CPU})]
if btb.mlx_available():
    DEVICES.append((Hardware.MLX, {"device": Hardware.MLX}))
if torch.cuda.is_available():
    DEVICES.append((Hardware.CUDA, {"device": Hardware.CUDA}))  # the card path, on a CUDA box (empty elsewhere)
PROMPT = [1, 2, 3, 4]
N = 4
OP_WIDTH = 2048  # a realistic layer width for the op benches; a multiple of the 256-weight superblock


# Pin the run so n does not float per-runner: at least 25 rounds (matching the suite's n>=16 discipline), a
# warmup round, and GC off during timing. pytest-benchmark still calibrates iterations-per-round and caps total
# time (the workflow passes --benchmark-max-time); it never runs unbounded.
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_generate(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    path = os.path.join(FIXTURES, model)
    if not os.path.isdir(path):
        pytest.skip(f"{model} fixture not built")
    benchmark.group = device  # type: ignore[attr-defined]  # group the report/JSON by device
    benchmark.name = f"{device}/{model}"  # type: ignore[attr-defined]
    with btb.load(path, log=None, v_max=0, **knobs) as sm:

        def once() -> object:
            return sm.generate(list(PROMPT), N, speculate=False)

        once()  # warm: compile kernels and settle allocation so the timed rounds are steady state
        benchmark(once)  # type: ignore[operator]  # pytest-benchmark times this over many rounds


def _api(benchmark: object, model: str, device: str, knobs: dict[str, str], what: str) -> btb.StreamedTextModel:
    path = os.path.join(FIXTURES, model)
    if not os.path.isdir(path):
        pytest.skip(f"{model} fixture not built")
    benchmark.group = f"{device}/{what}"  # type: ignore[attr-defined]
    benchmark.name = f"{device}/{model}/{what}"  # type: ignore[attr-defined]
    return btb.load(path, log=None, v_max=0, **knobs)


# The programmatic API beside the plain loop, per family and device: what each costs over `test_generate`'s
# decode of the same tokens - the hooked pick (logits in Python, the in-graph picks aside), a session driven by
# hand, a fork's rows over one prefill, sessions of their own lengths batched, and memory lent to the caller.
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_generate_hooked(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    with _api(benchmark, model, device, knobs, "hooked") as sm:

        def once() -> object:
            return sm.generate(list(PROMPT), N, speculate=False, processors=[lambda ids, lg: lg], logprobs=2)

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_session_feed_rewind(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    with _api(benchmark, model, device, knobs, "feed-rewind") as sm:
        s = sm.session(list(PROMPT))
        mark = s.mark()

        def once() -> object:
            out = s.feed(list(range(5, 5 + N)))
            s.rewind(mark)
            return out

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_session_feed_tapped(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    with _api(benchmark, model, device, knobs, "feed-tapped") as sm:
        s = sm.session(list(PROMPT))
        mark = s.mark()

        def once() -> object:
            out = s.feed(list(range(5, 5 + N)), last_only=True, taps=(-1,))
            s.rewind(mark)
            return out

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_fork_step_leave(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    with _api(benchmark, model, device, knobs, "fork-step") as sm:
        s = sm.session(list(PROMPT))

        def once() -> object:
            with s.fork(4) as br:
                br.step([1, 2, 3, 4])
                br.step([5, 6, 7, 8], taps=(-1,))
                br.leave(1)
                return br.step([9, 10, 11])

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_fork_generate(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    with _api(benchmark, model, device, knobs, "fork4") as sm:
        s = sm.session(list(PROMPT))
        smp = btb.Sampling(temperature=0.8, seed=1)

        def once() -> object:
            with s.fork(4) as br:
                return br.generate(N, eos=(), sampling=smp)

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_batch_generate(benchmark: object, model: str, device: str, knobs: dict[str, str]) -> None:
    rows = [list(PROMPT), [9, 8], list(range(1, 11))]  # three sessions of their own lengths
    with _api(benchmark, model, device, knobs, "batch3") as sm:

        def once() -> object:
            with sm.batch([sm.session(r) for r in rows]) as bt:
                return bt.generate(N, eos=())

        once()
        benchmark(once)  # type: ignore[operator]


@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_lend(benchmark: object, device: str, knobs: dict[str, str]) -> None:
    """a tensor and a room lent beside a loaded model with room to spare: the ledger's own cost"""
    with _api(benchmark, FAMILIES[0], device, knobs, "lend") as sm:

        def once() -> object:
            t = sm.empty((1024, 1024), dtype=torch.bfloat16)
            with sm.room(1 << 20):
                return t

        once()
        benchmark(once)  # type: ignore[operator]


# the one piece of hand data: bytes per 256-weight superblock, as the GGUF layout stores it. IQ4_NL has no entry
# because it is not a superblock type - 32-weight blocks with the scales in a separate array, so its launcher
# takes (d, q, x, ...) rather than one as-stored buffer and does not fit this bench's call shape.
SUPERBLOCK: dict[Quant, int] = {
    Quant.Q2_K: 84, Quant.Q3_K: 110, Quant.Q4_K: 144, Quant.Q5_K: 176, Quant.Q6_K: 210, Quant.IQ4_XS: 136,
}  # fmt: skip
# every as-stored quant kernel btb binds, keyed by the backend key its launcher is named after - a new k-quant
# is benched once it has a superblock size above, with no list to update here
QUANTS: dict[str, int] = {
    latt_backend_key(q): SUPERBLOCK[q]
    for q in quants_of(QuantClass.KQUANT) + quants_of(QuantClass.IQ4)
    if q in SUPERBLOCK
}


def _matvec(kind: str) -> Matvec:
    """the `matvec_<kind>` launcher, from whichever kernel module defines it."""
    from btb.mlx import iquant, kquant, q6k

    for mod in (kquant, q6k, iquant):
        fn = getattr(mod, "matvec_" + kind, None)
        if fn is not None:
            return cast("Matvec", fn)
    raise AttributeError(f"no matvec_{kind}() in btb.mlx.{{kquant,q6k,iquant}}")


@mlx_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("kind", list(QUANTS))
@pytest.mark.parametrize("t", [1, 16])  # decode (1 row) and a verify pass (16 rows)
def test_mlx_quant_matvec(benchmark: object, kind: str, t: int) -> None:
    m = mlxdev.mx()
    nbytes = (OP_WIDTH * OP_WIDTH // 256) * QUANTS[kind]
    w = m.array(np.random.randint(0, 256, nbytes, dtype=np.uint8))
    x = m.random.normal((t, OP_WIDTH)).astype(m.bfloat16)
    m.eval(w, x)
    fn = _matvec(kind)
    benchmark.group = "mlx-quant"  # type: ignore[attr-defined]

    def run() -> object:
        return m.eval(fn(w, x, OP_WIDTH, OP_WIDTH))  # eval forces the lazy kernel to actually run

    run()  # warm
    benchmark(run)  # type: ignore[operator]


@mlx_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("n", [512, 2048])  # context lengths the decode kernel reads over
def test_mlx_attn_decode(benchmark: object, n: int) -> None:
    from btb.mlx.attn import attn_decode

    m = mlxdev.mx()
    hq, hk, d = 32, 8, 128
    # the K/V cache layout the kernel reads: [1, Hk, cap, d] bf16, `n` rows live; q is [Hq, d] float32
    q = m.random.normal((hq, d)).astype(m.float32)
    kb = m.random.normal((1, hk, n, d)).astype(m.bfloat16)
    vb = m.random.normal((1, hk, n, d)).astype(m.bfloat16)
    m.eval(q, kb, vb)
    scale = float(d**-0.5)
    benchmark.group = "mlx-attn"  # type: ignore[attr-defined]

    def run() -> object:
        return m.eval(attn_decode(q, kb, vb, n, scale))

    run()
    benchmark(run)  # type: ignore[operator]


def _cuda_kernels() -> _Cuda:
    """the loaded card kernels, or a skip naming why they are absent (the fatbin is built by `python build.py`)."""
    k = Native.card_kernels()
    if k is None:
        pytest.skip(f"card kernels not loaded ({Native.cuda_reason}); build them with `python build.py`")
    return k


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("m", [1, 16, 32])
def test_cuda_gemv_bf16(benchmark: object, m: int) -> None:
    """the bf16 matvec `btb_gemv_bf16_m{m}` at m rows: the one-row step (m=1) and the batched/verify widths."""
    k = _cuda_kernels()
    w = torch.randn(OP_WIDTH, OP_WIDTH, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(m, OP_WIDTH, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(m, OP_WIDTH, dtype=torch.bfloat16, device="cuda")
    args = [k.ptr(w), k.ptr(x), k.ptr(y), ctypes.c_int(OP_WIDTH), ctypes.c_int(OP_WIDTH)]

    def run() -> None:
        k.launch(f"btb_gemv_bf16_m{m}", ((OP_WIDTH + 3) // 4, 1, 1), (128, 1, 1), args)
        torch.cuda.synchronize()

    benchmark.group = "cuda-gemv"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
