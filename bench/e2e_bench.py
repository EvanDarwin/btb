# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The one bench a person or an agent runs on any machine - it times whatever the box can, skipping the rest:

    pytest bench/e2e_bench.py --benchmark-json=e2e.json -q      # (needs the tiny fixtures + a built btb)

Three kinds of timing, all through pytest-benchmark (a stable, statistically-sound utility - no hand-rolled loop):

  * the end-to-end generate loop per family (dense, hybrid, the two MoE layouts, Gemma's sandwich/dual-rope,
    gpt-oss sinks) on each device the box has - cpu always, mlx on Apple silicon, cuda on a card;
  * the per-op kernels: the MLX Metal quant matvecs and attention decode (Apple silicon) and the CUDA card
    gemv, attention and grouped expert call (a CUDA box). A kernel's runtime does not depend on the weight
    VALUES, so these use random inputs - no fixture or real model needed;
  * the placement group (a CUDA box): each op at the engine's shapes and row counts on the CPU, on the card over
    weights held in VRAM, and on the card over weights shipped from RAM each call - where an op wins - and, with
    `--routing DIR/events.npz` (a `--profile DIR` run's), a MoE prefill's recorded expert calls replayed split
    between the two sides.

The native CPU ops live in Rust criterion (`native/benches/*.rs`, `cargo bench`) - a different toolchain, not
pytest. `python -m bench.report` compares two runs of both (the base-vs-PR deltas the PR comment renders),
grouped by each entry's bench group, and renders the placement group's crossover from a run's own times.

Not part of `pytest tests` (bench/ is outside testpaths); the bench workflow runs it explicitly.
"""

from __future__ import annotations

import ctypes
import gc
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import btb
from btb import mlx as mlxdev
from btb.engine.cuda import _CudaMixin
from btb.engine.native import Native
from btb.kinds import Quant, QuantClass, latt_backend_key, quants_of
from btb.mxfp4 import MxWeight
from btb.mxfp4_torch import dequant_blocks
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


# A conversation's next turn as a server sees it, per family and device: after another request on the same session,
# the turn opening on the rows the conversation left (`hit`: the prefix cache's, where the engine has one), against
# the same prompt with nothing kept (`cold`). The turn alone is timed; the conversation and the request between
# are each round's setup. A tier the prefix cache is not on yet drops the conversation at the request between: its
# hit is its cold
PREFIX_FIRST = [3 + (7 * i) % 190 for i in range(300)]  # the conversation's first prompt, inside every vocabulary
PREFIX_SIDE = [200 + i % 50 for i in range(20)]  # a request sharing nothing with it


@pytest.mark.parametrize("arm", ["hit", "cold"])
@pytest.mark.parametrize("model", FAMILIES)
@pytest.mark.parametrize("device,knobs", DEVICES, ids=[d for d, _ in DEVICES])
def test_prefix_turn(benchmark: object, model: str, device: str, knobs: dict[str, str], arm: str) -> None:
    with _api(benchmark, model, device, knobs, f"prefix-{arm}") as sm:
        out = list(sm.generate(PREFIX_FIRST, N, eos=(), speculate=False).tokens)
        turn = [*PREFIX_FIRST, *out, *range(5, 25)]

        def setup() -> tuple[tuple[object, ...], dict[str, object]]:
            pc = sm._prefix_cache()
            if pc is not None:
                pc.tree.evict()  # the rounds before let go: each starts from nothing kept
            s = sm.session()
            if arm == "hit":
                sm.generate(PREFIX_FIRST, N, eos=(), session=s, speculate=False)
                sm.generate(PREFIX_SIDE, 1, eos=(), session=s, speculate=False)
            return (s,), {}

        def once(s: object) -> object:
            return sm.generate(turn, 1, eos=(), session=cast("btb.Session", s), speculate=False)

        benchmark.pedantic(once, setup=setup, rounds=25, warmup_rounds=1)  # type: ignore[attr-defined]


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


# the decode's weight shapes, [R, C], on Qwen3-0.6B: q/k/v as one, the attention's output, gate/up as one, down
MMA_SHAPES = {"qkv": (4096, 1024), "o": (1024, 2048), "gu": (6144, 1024), "down": (1024, 3072)}


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("rows", [16, 8])
@pytest.mark.parametrize("warps", [2, 4, 8])
@pytest.mark.parametrize("shape", list(MMA_SHAPES))
def test_cuda_gemv_mma(benchmark: object, shape: str, warps: int, rows: int) -> None:
    """the tensor-core matvec at one row over a decode's weight shapes, by warps and rows a block
    (`btb_gemv_mma_bf16` at 16, `btb_gemv_mma8_bf16` at 8 - the same bits): a round streams distinct weights past
    the card's L2 back to back, as a step reads its layers, and the rate is the weights' bytes over the round
    (`gbps`). A short weight (1024 rows: 64 groups of 16 on a 60-SM card) streams at its best with more warps to a
    group than a tall one, or more groups"""
    k = _cuda_kernels()
    name = "btb_gemv_mma_bf16" if rows == 16 else "btb_gemv_mma8_bf16"
    if name not in k.fn:
        pytest.skip(f"{name} is not in this build")
    R, C = MMA_SHAPES[shape]
    n = -(-(160 << 20) // (R * C * 2))  # past a 48 MB L2 several times, so every read is the DRAM's
    ws = [torch.randn(R, C, dtype=torch.bfloat16, device="cuda") for _ in range(n)]
    x = torch.randn(32, C, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(32, R, dtype=torch.bfloat16, device="cuda")
    grid, block = ((R + rows - 1) // rows, 1, 1), (32 * warps, 1, 1)
    args = [[k.ptr(w), k.ptr(x), k.ptr(y), ctypes.c_int(R), ctypes.c_int(C), ctypes.c_int(1)] for w in ws]

    def body() -> None:
        for a in args:
            k.launch(name, grid, block, a)

    # the round as one captured graph, as a step replays its layers: the host's launches out of the time
    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()

    def run() -> None:
        g.replay()
        torch.cuda.synchronize()

    benchmark.group = f"cuda-gemv-mma-{shape}"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
    mean = float(benchmark.stats.stats.mean)  # type: ignore[attr-defined]
    benchmark.extra_info["gbps"] = round(n * R * C * 2 / mean / 1e9, 1)  # type: ignore[attr-defined]


@cuda_only
@pytest.mark.benchmark(min_rounds=10, warmup=True, disable_gc=True)
@pytest.mark.parametrize("how", ["mma", "f32", "cublas"])
@pytest.mark.parametrize("T", [512, 4096])  # a prompt chunk's rows
@pytest.mark.parametrize("shape", list(MMA_SHAPES))
def test_cuda_gemm(benchmark: object, shape: str, T: int, how: str) -> None:
    """a prompt chunk's matmul over a decode's weight shapes (Qwen3-0.6B's): `btb_gemm_mma_bf16` - each row the
    tensor-core matvec's bits, at its warps for the shape - and `btb_gemm_f32_bf16` - each row the fp32-chain matvec's
    - against torch's matmul (cuBLAS), what a prompt's chunk took before its rows had to be its steps'. The rate is
    the multiply-adds over the time (`tflops`)."""
    k = _cuda_kernels()
    if "btb_gemm_mma_bf16" not in k.fn:
        pytest.skip("the prompt's GEMMs are not in this build")
    R, C = MMA_SHAPES[shape]
    w = torch.randn(R, C, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(T, C, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(T, R, dtype=torch.bfloat16, device="cuda")
    take = _CudaMixin._card_gemm_takes("cuda")  # a split tail's buffers, kept from round to round as the engine's

    def run() -> None:
        if how == "cublas":
            torch.matmul(x, w.t(), out=y)
        else:
            # the engine's launch: the step's warps for the shape, its last wave split where the card's plan splits it
            _CudaMixin._card_gemm_launch(k, how == "mma", w, x, y, R, C, T, take)
        torch.cuda.synchronize()

    benchmark.group = f"cuda-gemm-{shape}-t{T}"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
    mean = float(benchmark.stats.stats.mean)  # type: ignore[attr-defined]
    benchmark.extra_info["tflops"] = round(2 * T * R * C / mean / 1e12, 1)  # type: ignore[attr-defined]


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("how", ["flash", "sdpa"])
@pytest.mark.parametrize("T", [1, 5])  # the one-row step, a speculative tree's verify
# the context the rows attend over, to a million keys (one layer's cache there: 4 GB)
@pytest.mark.parametrize("n", [128, 512, 1024, 2048, 4096, 8192, 16384, 40960, 262144, 1048576])
def test_cuda_attn_decode(benchmark: object, n: int, T: int, how: str) -> None:
    """a decode step's attention at Qwen3-0.6B's heads (16 q, 8 kv, d 128): the one attention every pass takes, in its
    decode form (`btb_attn_flash_d128`, tensor cores, a row's bits its own whatever the pass), against torch's sdpa
    over the same rows (the tree's mask where T > 1). A round runs one call a layer over distinct caches past the L2,
    as a step does, and the rate is the caches' live K/V bytes over the round (`gbps`)."""
    k = _cuda_kernels()
    if "btb_attn_flash_d128" not in k.fn:
        pytest.skip("the one attention is not in this build")
    Hq, Hk, D = 16, 8, 128
    cap = (n + T + 1023) // 1024 * 1024
    S = cap // _CudaMixin._attn_group(D)  # the groups the launch covers
    live = 2 * Hk * (n + T) * D * 2
    layers = min(28, -(-(160 << 20) // live))
    KV = [torch.randn(2, Hk, cap, D, dtype=torch.bfloat16, device="cuda") for _ in range(layers)]
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(T, Hq, D, dtype=torch.bfloat16, device="cuda")
    pm = torch.zeros(S * T * Hq, device="cuda")
    pl = torch.zeros(S * T * Hq, device="cuda")
    pa = torch.zeros(S * T * Hq * D, device="cuda")
    cnt = torch.zeros(T * Hq, dtype=torch.int32, device="cuda")
    tree = [-1, 0, 1, 0, 3][:T]
    n0 = torch.tensor([n], dtype=torch.int32, device="cuda")
    par = torch.tensor(tree, dtype=torch.int32, device="cuda")
    P, I, Fl = k.ptr, ctypes.c_int, ctypes.c_float
    # head g's row r at g * hs + r * rs: the caches here head-major [Hk, cap, D]; a block 8 rows of a group, a warp a
    # tile
    grid, block = ((T * (Hq // Hk) + 7) // 8, S, Hk), (32 * _CudaMixin._attn_warps(D), 1, 1)
    tail = [I(T), I(Hq), I(Hk), I(cap * D), I(D), Fl(D**-0.5), P(pm), P(pl), P(pa), P(cnt), I(0), P(None)]
    args = [[P(q), P(kv[0]), P(kv[1]), P(out), P(n0), P(par), *tail] for kv in KV]
    # sdpa's rows: each over the prefix, its ancestors and itself
    seen = torch.zeros(T, n + T, dtype=torch.bool, device="cuda")
    seen[:, :n] = True
    for t in range(T):
        p = t
        while p >= 0:
            seen[t, n + p] = True
            p = tree[p]
    qs = q.transpose(0, 1)[None]  # [1, Hq, T, D], the module's layout

    def body() -> None:
        if how == "flash":
            for a in args:
                k.launch("btb_attn_flash_d128", grid, block, a)
        else:
            for kv in KV:
                F.scaled_dot_product_attention(
                    qs, kv[0, None, :, : n + T], kv[1, None, :, : n + T], attn_mask=seen, enable_gqa=True, scale=D**-0.5
                )

    body()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()

    def run() -> None:
        g.replay()
        torch.cuda.synchronize()

    benchmark.group = f"cuda-attn-decode-n{n}-t{T}"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
    mean = float(benchmark.stats.stats.mean)  # type: ignore[attr-defined]
    benchmark.extra_info["gbps"] = round(layers * live / mean / 1e9, 1)  # type: ignore[attr-defined]


@cuda_only
@pytest.mark.benchmark(min_rounds=10, warmup=True, disable_gc=True)
@pytest.mark.parametrize("how", ["flash", "sdpa"])
@pytest.mark.parametrize("n", [0, 4096, 16384, 262144, 1048576])  # the rows before the chunk
@pytest.mark.parametrize("T", [512, 4096])  # the chunk's rows
def test_cuda_attn_prefill(benchmark: object, T: int, n: int, how: str) -> None:
    """a prompt chunk's attention at Qwen3-0.6B's heads (16 q, 8 kv, d 128): the one attention every pass takes in its
    prefill form (`btb_attn_flash_prefill_d128`, each row over the rows before it and itself on tensor cores, the bits
    its step makes), against torch's sdpa over the same rows with the causal mask the chunk takes. The rate is the
    attention's multiply-adds over the time (`tflops`)."""
    k = _cuda_kernels()
    if "btb_attn_flash_prefill_d128" not in k.fn:
        pytest.skip("the one attention is not in this build")
    if how == "sdpa" and n > 16384:
        pytest.skip("sdpa takes the chunk's causal mask as a T x (n + T) tensor: 4 GB at a million keys")
    Hq, Hk, D = 16, 8, 128
    K = torch.randn(Hk, n + T, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(Hk, n + T, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(T, Hq, D, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(T, Hq, D, dtype=torch.bfloat16, device="cuda")
    P, I, Fl = k.ptr, ctypes.c_int, ctypes.c_float
    states = torch.empty(T * Hq * D, device="cuda")  # the row states (its `run`)
    args = [P(q), P(K), P(V), P(out), I(n), I(T), I(Hq), I(Hk), I(K.stride(0)), I(K.stride(1)), Fl(D**-0.5)]
    args += [I(0), P(None), P(states)]
    qs = q.transpose(0, 1)[None]  # [1, Hq, T, D], the module's layout
    mask = torch.ones(T, n + T, dtype=torch.bool, device="cuda").tril(diagonal=n)
    tpb = k.flash_prefill_rows(D) // (Hq // Hk)  # the flash form's tokens a block: their G heads each

    def run() -> None:
        if how == "flash":
            k.launch(
                k.flash_prefill_kernel(D),  # the orientation this card takes (`btb_mma_roles`)
                ((T + tpb - 1) // tpb, Hk, 1),
                (128, 1, 1),
                args,
                shared=k.flash_prefill_smem(D),
            )
        else:
            F.scaled_dot_product_attention(qs, K[None], V[None], attn_mask=mask, enable_gqa=True, scale=D**-0.5)
        torch.cuda.synchronize()

    benchmark.group = f"cuda-attn-prefill-t{T}-n{n}"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
    mean = float(benchmark.stats.stats.mean)  # type: ignore[attr-defined]
    macs = Hq * D * 2 * sum(n + t + 1 for t in range(T))  # q.k and p.v, the causal half
    benchmark.extra_info["tflops"] = round(2 * macs / mean / 1e12, 1)  # type: ignore[attr-defined]


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("how", ["rows", "per-row"])
@pytest.mark.parametrize("n", [512, 4096])  # the shared prefix the rows attend over
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_cuda_attn_rows(benchmark: object, rows: int, n: int, how: str) -> None:
    """a fork's decode attention at Qwen3-0.6B's heads (16 q, 8 kv, d 128), each row 16 steps past a shared prefix:
    the one attention's rows form (`btb_attn_flash_rows_d128`) over all rows in one launch, against its decode form
    (`btb_attn_flash_d128`) a launch a row over the same keys (the one-row steps the rows would take one session at a
    time)."""
    k = _cuda_kernels()
    if "btb_attn_flash_rows_d128" not in k.fn:
        pytest.skip("the one attention is not in this build")
    Hq, Hk, D, steps = 16, 8, 128, 16
    G = Hq // Hk
    cap = (n + (steps + 1) * rows + 1023) // 1024 * 1024
    S = cap // _CudaMixin._attn_group(D)  # the groups the launch covers
    K = torch.randn(Hk, cap, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(Hk, cap, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(rows, Hq, D, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, Hq, D, dtype=torch.bfloat16, device="cuda")
    pm = torch.zeros(S * rows * Hq, device="cuda")
    pl = torch.zeros(S * rows * Hq, device="cuda")
    pa = torch.zeros(S * rows * Hq * D, device="cuda")
    cnt = torch.zeros(rows * Hq, dtype=torch.int32, device="cuda")
    # the rows' layout (base, steps, W, then col/offset/length a row): every row over the one prefix
    rw = torch.tensor([n, steps, rows, *[x for c in range(rows) for x in (c, 0, n)]], dtype=torch.int32, device="cuda")
    n0 = torch.tensor([n + steps], dtype=torch.int32, device="cuda")
    root = torch.tensor([-1], dtype=torch.int32, device="cuda")
    P, I, Fl = k.ptr, ctypes.c_int, ctypes.c_float
    # head g's row r at g * hs + r * rs: the cache head-major [Hk, cap, D]
    dims = [I(Hq), I(Hk), I(cap * D), I(D), Fl(D**-0.5)]
    states = [P(pm), P(pl), P(pa), P(cnt), I(0)]
    if how == "rows":
        launches = [
            (
                "btb_attn_flash_rows_d128",
                (rows * ((G + 7) // 8), S, Hk),
                [P(q), P(K), P(V), P(out), P(rw), I(rows), *dims, *states],
            )
        ]
    else:
        launches = [
            (
                "btb_attn_flash_d128",
                ((G + 7) // 8, S, Hk),
                [P(q[r]), P(K), P(V), P(out[r]), P(n0), P(root), I(1), *dims, *states, P(None)],
            )
            for r in range(rows)
        ]

    def run() -> None:
        for name, grid, args in launches:
            k.launch(name, grid, (32 * _CudaMixin._attn_warps(D), 1, 1), args)
        torch.cuda.synchronize()

    benchmark.group = f"cuda-attn-rows/n{n}/rows{rows}"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize("form", ["bf16", "mxfp4"])
@pytest.mark.parametrize("rows", [8, 32, 128])  # each expert's rows: a prefill chunk's spread over its experts
def test_cuda_grouped_experts(benchmark: object, rows: int, form: str) -> None:
    """One wave of a grouped expert call on the card (`_Experts._card_grouped`) at gpt-oss-120b's expert shape: 16
    experts of `rows` rows each, gate_up and down as `torch._grouped_mm` over the experts' rows in slot order - the
    MXFP4 form first widened from the depot's stacks in one pass (`btb_mx4_widen`), as a wave's batch is."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("this torch has no _grouped_mm: the engine keeps the per-expert loop")
    n, H, I = 16, 2880, 2880
    x = torch.randn(n * rows, H, dtype=torch.bfloat16, device=CUDA)
    ends = torch.arange(1, n + 1, dtype=torch.int32, device=CUDA) * rows
    shapes = ((2 * I, H), (H, I))
    if form == "mxfp4":
        k = _cuda_kernels()
        seats = torch.arange(n, dtype=torch.int32, device=CUDA)
        stacks = [
            (
                torch.randint(0, 256, (n, r, c // 32, 16), dtype=torch.uint8, device=CUDA),
                torch.randint(118, 124, (n, r, c // 32), dtype=torch.uint8, device=CUDA),
            )
            for r, c in shapes
        ]
        wide = [torch.empty(n, r * c, dtype=torch.bfloat16, device=CUDA) for r, c in shapes]

        def weights() -> list[torch.Tensor]:
            return [
                k.mx4_widen(b, s, seats, out=w).view(n, r, c)
                for (b, s), w, (r, c) in zip(stacks, wide, shapes, strict=True)
            ]

    else:
        held = [torch.randn(n, r, c, device=CUDA).mul_(c**-0.5).bfloat16() for r, c in shapes]

        def weights() -> list[torch.Tensor]:
            return held

    def run() -> None:
        gu, dn = weights()
        h = torch._grouped_mm(x, gu.transpose(1, 2), offs=ends)
        torch._grouped_mm(_act(h), dn.transpose(1, 2), offs=ends)
        torch.cuda.synchronize()

    benchmark.group = "cuda-grouped"  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]


# -- where an op wins ------------------------------------------------------------------------------------------
#
# The placement group: each op at the engine's shapes and row counts, timed where it could run - the CPU over its
# weights in RAM (the engine's host path: the native kernels, torch's float32 GEMM from `Native.gemm_rows` rows),
# the card over weights held in VRAM, and the card over weights shipped from RAM each call. The question is the
# placement, not the kernel: every lane is wall time as the engine would pay it, the crossing of the tier's edge
# included - a card lane takes its rows from the host and hands its answer back. Weights rotate through pools past
# the CPU's L3 and the card's L2, so every call reads memory as a decode's layers do, and each lane's output is held
# against a float32 reference before it is timed. `python -m bench.report` renders the crossover - where each op's
# winner changes with the rows - from the run's JSON (each bench's `extra_info`).

CUDA = "cuda"
CPU_POOL = 384 << 20  # past a desktop CPU's L3 many times over
CARD_POOL = 256 << 20  # past the card's L2 (48 MB on an AD104)
TOLERANCE = 0.03  # a lane's largest error against the float32 reference, relative to the reference's largest value
LANES = ("cpu/ram", "cuda/vram", "cuda/ram")  # where the op computes / where its weights sit
ROWS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 2048, 4096)
GLUE_ROWS = (1, 16, 256, 4096)


@dataclass(frozen=True)
class Point:
    """one op at one shape: its dims, the row counts it is timed at and the lanes it runs in. `per_rows`: its
    operands are the rows' own (attention's cache is `rows` long), so they are built afresh for each count."""

    op: str
    shape: str
    dims: tuple[int, ...]
    rows: tuple[int, ...]
    lanes: tuple[str, ...] = LANES
    per_rows: bool = False


POINTS: dict[tuple[str, str], Point] = {
    (p.op, p.shape): p
    for p in (
        Point("linear", "qkv-6144x2560", (6144, 2560), ROWS),
        Point("linear", "down-2560x9728", (2560, 9728), ROWS),
        Point("linear", "down-5120x25600", (5120, 25600), tuple(r for r in ROWS if r <= 1024)),
        Point("linear", "head-151936x2560", (151936, 2560), (1, 4, 16)),
        Point("expert", "moe-2560x640", (2560, 640), tuple(r for r in ROWS if r <= 512)),
        Point("expert_mx4", "oss-2880x2880", (2880, 2880), tuple(r for r in ROWS if r <= 512)),
        # one row's attention over a cache of `rows` rows
        Point("attn", "q32-kv8-d128", (32, 8, 128), (1024, 4096, 16384, 65536, 131072), per_rows=True),
        # the glue between the matmuls: its weights are small, so they are held where it runs; the rows still cross
        Point("rmsnorm", "2560", (2560,), GLUE_ROWS, LANES[:2]),
        Point("router", "512x2560-top10", (512, 2560, 10), GLUE_ROWS, LANES[:2]),
        Point("silu_mul", "2560", (2560,), GLUE_ROWS, LANES[:2]),
        Point("pick", "151936-t0.8-p0.9", (151936,), (1, 16), LANES[:2]),
    )
}


class Pool:
    """copies of one operand, enough to pass `nbytes`, handed out in turn: a call reads its weights from memory,
    never from a cache the last call filled"""

    def __init__(self, make: Callable[[], Any], each: int, nbytes: int, cap: int = 64) -> None:
        n = max(2, min(cap, -(-nbytes // max(1, each))))
        self.items = [make() for _ in range(n)]
        self.i = 0

    @classmethod
    def one(cls, item: Any) -> Pool:
        """one operand handed out every call: the glue's small weights, which a cache holds whatever the bench does"""
        return cls(lambda: item, 0, 0, cap=1)

    def next(self) -> Any:
        self.i = (self.i + 1) % len(self.items)
        return self.items[self.i]

    @property
    def first(self) -> Any:
        return self.items[0]


class _Operands:
    """A point's operands in each tier, built on first use and shared by its row counts and lanes; one point's at a
    time - the last point's are let go (and the allocators' cached blocks given back) before the next one's."""

    held: ClassVar[_Operands | None] = None

    def __init__(self, key: tuple[object, ...]) -> None:
        self.key = key
        self.parts: dict[str, Any] = {}

    @classmethod
    def of(cls, *key: object) -> _Operands:
        if cls.held is None or cls.held.key != key:
            cls.held = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                host_empty = getattr(torch._C, "_host_emptyCache", None)
                if host_empty is not None:
                    host_empty()  # the pinned pools' blocks, which the host allocator keeps otherwise
            cls.held = cls(key)
        return cls.held

    def get(self, name: str, make: Callable[[], Any]) -> Any:
        if name not in self.parts:
            self.parts[name] = make()
        return self.parts[name]


# a lane: the pool its weights rotate through, the call over one of them, and the float32 answer (None: unchecked)
Lane = tuple[Pool, Callable[[Any], Any], torch.Tensor | None]


def _rows(n: int, width: int) -> torch.Tensor:
    """`n` rows of activations, float32 on the host, the same in every lane of a point"""
    return torch.randn(n, width, generator=torch.Generator().manual_seed(n))


def _act(gu: torch.Tensor) -> torch.Tensor:
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


def _host_linear(w: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """the engine's host matmul (`_HostLinear`): the native kernel under `Native.gemm_rows` rows, else torch's
    float32 GEMM over the weight widened"""
    if Native.gemv is not None and x.shape[0] < Native.gemm_rows:
        Native.gemv(w, x, y)
        return y
    return F.linear(x, w.float())


def _to_card(x: torch.Tensor) -> torch.Tensor:
    return x.to(CUDA, torch.bfloat16)


def _card(
    ops: _Operands, cpu: Pool, each: int, lane: str, caps: tuple[int, int], on_card: Callable[[Any], Any]
) -> tuple[Pool, Callable[[Any], Any]]:
    """a card lane over `cpu.first`'s tensors: held in VRAM (`cuda/vram`), a pool of copies there; or shipped from
    RAM each call (`cuda/ram`), a pool of pinned copies, each landing in one set of card buffers"""
    if lane == "cuda/vram":
        return ops.get(
            "card", lambda: Pool(lambda: tuple(t.to(CUDA) for t in cpu.first), each, CARD_POOL, caps[0])
        ), on_card
    pin = ops.get("pin", lambda: Pool(lambda: tuple(t.pin_memory() for t in cpu.first), each, CARD_POOL, caps[1]))
    bufs = ops.get("buf", lambda: tuple(torch.empty_like(t, device=CUDA) for t in cpu.first))

    def fed(w: tuple[torch.Tensor, ...]) -> Any:
        for b, t in zip(bufs, w, strict=True):
            b.copy_(t, non_blocking=True)
        return on_card(bufs)

    return pin, fed


def _linear(p: Point, ops: _Operands, rows: int, lane: str) -> Lane:
    R, C = p.dims
    each = R * C * 2
    cpu = ops.get("cpu", lambda: Pool(lambda: (torch.randn(R, C).mul_(C**-0.5).bfloat16(),), each, CPU_POOL))
    x = _rows(rows, C)
    ref = F.linear(x, cpu.first[0].float())
    if lane == "cpu/ram":
        y = torch.empty(rows, R)
        return cpu, lambda w: _host_linear(w[0], x, y), ref
    return (*_card(ops, cpu, each, lane, (64, 8), lambda w: F.linear(_to_card(x), w[0]).float().cpu()), ref)


def _expert(p: Point, ops: _Operands, rows: int, lane: str) -> Lane:
    """one expert of a bf16 mixture: gate_up, the gate, down"""
    H, I = p.dims
    each = 3 * H * I * 2

    def make() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(2 * I, H).mul_(H**-0.5).bfloat16(), torch.randn(H, I).mul_(I**-0.5).bfloat16()

    cpu = ops.get("cpu", lambda: Pool(make, each, CPU_POOL, cap=128))
    x = _rows(rows, H)
    gu0, dn0 = cpu.first
    ref = F.linear(_act(F.linear(x, gu0.float())), dn0.float())
    if lane == "cpu/ram":
        gy, dy = torch.empty(rows, 2 * I), torch.empty(rows, H)
        return cpu, lambda w: _host_linear(w[1], _act(_host_linear(w[0], x, gy)).contiguous(), dy), ref

    def on_card(w: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return F.linear(_act(F.linear(_to_card(x), w[0])), w[1]).float().cpu()

    return (*_card(ops, cpu, each, lane, (64, 32), on_card), ref)


def _stack(w: MxWeight) -> tuple[torch.Tensor, torch.Tensor]:
    """an MXFP4 matrix as a depot's stack of one seat: blocks [1, rows, groups, 16], scales [1, rows, groups]"""
    assert w.scales is not None  # the checkpoint's layout, as the bench builds it
    return w.blocks.view(1, w.shape[0], -1, 16), w.scales.view(1, w.shape[0], -1)


def _expert_mx4(p: Point, ops: _Operands, rows: int, lane: str) -> Lane:
    """one expert of an MXFP4 mixture as stored: the CPU's own matvec over the blocks, against the card widening
    them (`btb_mx4_widen`; torch's `dequant_blocks` without the card kernels) from VRAM or from the bytes shipped -
    a quarter of bf16's over the bus"""
    if lane == "cpu/ram" and Native.gemv_mx4 is None:
        pytest.skip("this library has no MXFP4 matvec")
    H, I = p.dims
    shapes = ((2 * I, H), (H, I))

    def mat(r: int, k: int) -> MxWeight:
        g = k // 32
        blocks = torch.randint(0, 256, (r * g * 16,), dtype=torch.uint8)
        return MxWeight(blocks, torch.randint(118, 124, (r * g,), dtype=torch.uint8), r, k)

    each = sum(r * k // 32 * 17 for r, k in shapes)
    cpu = ops.get("cpu", lambda: Pool(lambda: tuple(mat(r, k) for r, k in shapes), each, CPU_POOL, cap=128))
    x = _rows(rows, H)
    gu0, dn0 = (w.dequantize(torch.float32) for w in cpu.first)
    ref = F.linear(_act(F.linear(x, gu0)), dn0)
    if lane == "cpu/ram":
        gy, dy = torch.empty(rows, 2 * I), torch.empty(rows, H)

        def host(w: tuple[MxWeight, ...]) -> torch.Tensor:
            Native.gemv_mx4(w[0], x, gy)
            Native.gemv_mx4(w[1], _act(gy).contiguous(), dy)
            return dy

        return cpu, host, ref
    stacks = ops.get("stacks", lambda: Pool.one(tuple(t for w in cpu.first for t in _stack(w))))
    wide = ops.get("wide", lambda: [torch.empty(1, r * k, dtype=torch.bfloat16, device=CUDA) for r, k in shapes])
    seat = ops.get("seat", lambda: torch.zeros(1, dtype=torch.int32, device=CUDA))
    kern = Native.card_kernels()

    def widen(j: int, w: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r, k = shapes[j]
        b, s = w[2 * j], w[2 * j + 1]
        return (kern.mx4_widen(b, s, seat, out=wide[j]) if kern is not None else dequant_blocks(b, s)).view(r, k)

    def on_card(w: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return F.linear(_act(F.linear(_to_card(x), widen(0, w))), widen(1, w)).float().cpu()

    return (*_card(ops, stacks, each, lane, (64, 64), on_card), ref)


def _attn(p: Point, ops: _Operands, n: int, lane: str) -> Lane:
    """one row's decode attention over a cache of `n` rows (K and V [Hk, n, d] bf16): the host's kernel, against
    torch's SDPA on the card over the cache held there or shipped"""
    if lane == "cpu/ram" and Native.attn_decode is None:
        pytest.skip("this library has no attention kernel")
    Hq, Hk, d = p.dims
    each = 2 * Hk * n * d * 2

    def make() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(Hk, n, d).bfloat16(), torch.randn(Hk, n, d).bfloat16()

    cpu = ops.get("cpu", lambda: Pool(make, each, CPU_POOL, cap=32))
    q = _rows(Hq, d)
    scale = d**-0.5
    k0, v0 = cpu.first
    ref = F.scaled_dot_product_attention(
        q.view(1, Hq, 1, d), k0.float().unsqueeze(0), v0.float().unsqueeze(0), scale=scale, enable_gqa=True
    ).view(Hq, d)
    if lane == "cpu/ram":
        out = torch.empty(Hq, d)

        def host(kv: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            Native.attn_decode(q, kv[0], kv[1], scale, out)
            return out

        return cpu, host, ref

    def on_card(kv: tuple[torch.Tensor, ...]) -> torch.Tensor:
        qd = _to_card(q).view(1, Hq, 1, d)
        y = F.scaled_dot_product_attention(qd, kv[0].unsqueeze(0), kv[1].unsqueeze(0), scale=scale, enable_gqa=True)
        return y.float().cpu().view(Hq, d)

    return (*_card(ops, cpu, each, lane, (16, 4), on_card), ref)


def _norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    v = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(v + 1e-6)).to(x.dtype) * w


def _glue(p: Point, ops: _Operands, rows: int, lane: str) -> Lane:
    """the small ops between the matmuls, their weights held where they run; the card lane's rows still cross"""
    host = lane == "cpu/ram"
    if p.op == "pick":
        (V,) = p.dims
        lg = torch.randn(rows, V, generator=torch.Generator().manual_seed(rows))
        keys = list(range(rows))
        if host:
            if Native.sample_pick is None:
                pytest.skip("this library has no sampler")
            return Pool.one(lg), lambda t: Native.sample_pick(t, keys, 0.8, 0, 0.9), None
        k = _cuda_kernels()  # the logits are the card's own there: only the picks come back
        return Pool.one(lg.to(CUDA)), lambda t: k.pick(t, keys, 0.8, 0, 0.9).cpu(), None
    H = p.dims[1] if p.op == "router" else p.dims[0]
    x = _rows(rows, 2 * H if p.op == "silu_mul" else H)

    def fn(a: torch.Tensor, w: Any) -> torch.Tensor:
        if p.op == "rmsnorm":
            return _norm(a, w)
        if p.op == "router":  # the softmax over the experts' logits, its k largest
            return torch.topk(torch.softmax(F.linear(a, w).float(), -1), p.dims[2], dim=-1).values
        return _act(a)

    w: torch.Tensor | None = None
    if p.op == "rmsnorm":
        w = ops.get("w", lambda: torch.rand(H) + 0.5)
    elif p.op == "router":
        w = ops.get("w", lambda: torch.randn(p.dims[0], H).mul_(H**-0.5))
    ref = fn(x, w)
    if host:
        return Pool.one(w), lambda wt: fn(x, wt), ref
    wd = None if w is None else w.to(CUDA, torch.bfloat16)
    return Pool.one(wd), lambda wt: fn(_to_card(x), wt).float().cpu(), ref


BUILD: dict[str, Callable[[Point, _Operands, int, str], Lane]] = {
    "linear": _linear, "expert": _expert, "expert_mx4": _expert_mx4, "attn": _attn,
    "rmsnorm": _glue, "router": _glue, "silu_mul": _glue, "pick": _glue,
}  # fmt: skip


def _check(what: str, y: Any, ref: torch.Tensor | None) -> float | None:
    """a lane's answer against the float32 reference: its largest error relative to the reference's largest value,
    refused past `TOLERANCE`; None for a lane with nothing to hold it to (a sampled pick)"""
    if ref is None:
        return None
    y, ref = y.detach().float().cpu(), ref.detach().float().cpu()
    err = float((y - ref).abs().max() / ref.abs().max().clamp(min=1e-30))
    assert err < TOLERANCE, f"{what}: {err:.3g} of the reference's largest value off it"
    return err


def _placed(benchmark: Any, lane: str, op: str, shape: str, rows: int, err: float | None, row: str = "") -> None:
    """the bench's group and id (`<lane>/<op>/<shape>/<row>`, the lane naming its device first), and what the
    report's crossover reads back out of the JSON"""
    row = row or f"r{rows}"
    bid = f"{lane}/{op}/{shape}/{row}"
    benchmark.group = "placement"
    benchmark.name = bid
    benchmark.extra_info.update(id=bid, op=op, shape=shape, rows=rows, row=row, lane=lane, err=err)


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
@pytest.mark.parametrize(
    "op,shape,rows,lane",
    [
        pytest.param(p.op, p.shape, r, lane, id=f"{p.op}-{p.shape}-r{r}-{lane.replace('/', '-')}")
        for p in POINTS.values()
        for r in p.rows
        for lane in p.lanes
    ],
)
def test_placement(benchmark: object, op: str, shape: str, rows: int, lane: str) -> None:
    """one op at one shape and row count, in one lane: its answer checked on its pool's first weight, then each
    call timed on the next"""
    p = POINTS[(op, shape)]
    ops = _Operands.of(op, shape, rows if p.per_rows else None)
    pool, run, ref = BUILD[op](p, ops, rows, lane)
    err = _check(f"{lane} {op} {shape} at {rows} rows", run(pool.first), ref)
    _placed(benchmark, lane, op, shape, rows, err)
    benchmark(lambda: run(pool.next()))  # type: ignore[operator]


# -- a recorded prefill's expert calls, replayed ------------------------------------------------------------------
#
# The Express's question over real routing: a MoE layer's prefill call with its crowded experts on the card and its
# sparse ones on the CPU at once. A `--profile DIR` run records every call's picks (its `call` events and `picks`,
# btb/engine/experts.py `ExpertProfile`); `--routing DIR/events.npz` replays every sixth layer's calls over stand-in
# weights of the model's expert shape (`--routing-expert`): every expert on the CPU (grouped native matvecs), every
# expert fed to the card through pinned double buffers, and the split at each cut (the experts with at least `cut`
# rows on the card, the rest on the CPU in a thread beside it). Every lane leaves the call's answer on the card.

ROUTING_EVERY = 6  # every sixth layer's calls: the layers route alike, and all of them is hours of bench
ROUTING_CUTS = (4, 8, 12, 16, 20, 24, 32, 48, 64)
ROUTING_POOL = 128  # distinct experts held pinned (1.26 GB at 2560x640), past the caches: expert e reads pool[e % 128]
ROUTING_GROUP = 16  # experts one native grouped matvec takes: a pool barrier each, not one an expert


class _Routing:
    """a profile's expert calls as the replay reads them: each `call` event's (layer, rows, offset), its rows'
    picks the profile's `picks` rows from the offset on (-1 past a narrower call's k)"""

    loaded: ClassVar[dict[str, _Routing]] = {}

    def __init__(self, path: str) -> None:
        from btb.engine.experts import ExpertProfile

        with np.load(path) as z:
            if "picks" not in z.files:
                raise ValueError(f"{path} predates the profile's picks: record it again with `--profile`")
            ev, self.picks = z["events"], z["picks"]
        self.calls = [(int(c[3]), int(c[4]), int(c[7])) for c in ev[ev[:, 2] == ExpertProfile.CALL]]

    @classmethod
    def of(cls, path: str) -> _Routing:
        if path not in cls.loaded:
            cls.loaded[path] = cls(path)
        return cls.loaded[path]

    def top(self, i: int) -> torch.Tensor:
        _layer, rows, off = self.calls[i]
        return torch.from_numpy(self.picks[off : off + rows].astype(np.int64))

    def most(self, i: int) -> int:
        """the rows of the call's most crowded expert"""
        e, n = np.unique(self.top(i).numpy(), return_counts=True)
        return int(n[e >= 0].max(initial=0))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """the routing replay's cases, one a (recorded call, lane), from `--routing`'s profile; a skip without one"""
    if metafunc.function.__name__ != "test_placement_routing":
        return
    path = metafunc.config.getoption("routing")
    if not path:
        why = "no --routing profile (a `--profile DIR` run's DIR/events.npz)"
        metafunc.parametrize("call,lane", [pytest.param(-1, "", marks=pytest.mark.skip(reason=why), id="no-routing")])
        return
    rt = _Routing.of(str(path))
    cases = []
    for i, (layer, rows, _off) in enumerate(rt.calls):
        if layer % ROUTING_EVERY:
            continue
        cuts = [f"split/c{c}" for c in ROUTING_CUTS if 1 < c <= rt.most(i)]
        for lane in ("cpu/ram", "cuda/ram", *cuts):
            cases.append(pytest.param(i, lane, id=f"L{layer}-c{i}-r{rows}-{lane.replace('/', '-')}"))
    metafunc.parametrize("call,lane", cases)


@cuda_only
@pytest.mark.benchmark(min_rounds=25, warmup=True, disable_gc=True)
def test_placement_routing(benchmark: object, request: pytest.FixtureRequest, call: int, lane: str) -> None:
    """one recorded call in one lane: its answer checked against the float32 sum of its experts' products, then
    timed"""
    if Native.gemv_group is None and lane != "cuda/ram":
        pytest.skip("this library has no grouped matvec")
    rt = _Routing.of(str(request.config.getoption("routing")))
    H, I = (int(v) for v in str(request.config.getoption("routing_expert")).lower().split("x"))
    layer, T, _off = rt.calls[call]
    top = rt.top(call)
    ops = _Operands.of("express", H, I)

    def expert() -> tuple[torch.Tensor, torch.Tensor]:
        gu = torch.randn(2 * I, H).mul_(H**-0.5).bfloat16().pin_memory()
        return gu, torch.randn(H, I).mul_(I**-0.5).bfloat16().pin_memory()

    def buffers() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.empty(2 * I, H, dtype=torch.bfloat16, device=CUDA), torch.empty(
            H, I, dtype=torch.bfloat16, device=CUDA
        )

    pin = ops.get("pin", lambda: [expert() for _ in range(ROUTING_POOL)])
    bufs = ops.get("buf", lambda: [buffers() for _ in range(2)])
    free = ops.get("free", lambda: [torch.cuda.Event() for _ in range(2)])
    copy = ops.get("copy", torch.cuda.Stream)
    x = _rows(T, H)
    xd = _to_card(x)
    calls = []  # (expert, its rows on the host, on the card), the most rows first
    for e in torch.unique(top).tolist():
        if e >= 0:
            idx = (top == e).any(-1).nonzero().flatten()
            calls.append((e, idx, idx.to(CUDA)))
    calls.sort(key=lambda c: -len(c[1]))
    ref = torch.zeros(T, H)
    for e, idx, _ in calls:
        gu, dn = pin[e % ROUTING_POOL]
        ref.index_add_(0, idx, F.linear(_act(F.linear(x[idx], gu.float())), dn.float()))
    out_cpu = torch.zeros(T, H)

    def cpu_side(part: list[Any]) -> None:
        # grouped: a pool barrier a matvec for every ROUTING_GROUP experts, and one scatter for the call
        out_cpu.zero_()
        idxs, ys = [], []
        for s in range(0, len(part), ROUTING_GROUP):
            grp = part[s : s + ROUTING_GROUP]
            gys = [torch.empty(len(idx), 2 * I) for _, idx, _ in grp]
            Native.gemv_group([pin[e % ROUTING_POOL][0] for e, _, _ in grp], [x[idx] for _, idx, _ in grp], gys)
            dys = [torch.empty(len(idx), H) for _, idx, _ in grp]
            Native.gemv_group([pin[e % ROUTING_POOL][1] for e, _, _ in grp], [_act(g).contiguous() for g in gys], dys)
            idxs += [idx for _, idx, _ in grp]
            ys += dys
        if idxs:
            out_cpu.index_add_(0, torch.cat(idxs), torch.cat(ys))

    def card_side(part: list[Any], out: torch.Tensor) -> None:
        # the next expert's copy under the current one's products: two buffers, a copy stream, an event each way
        main = torch.cuda.current_stream()
        for j, (e, _, idd) in enumerate(part):
            b = j % 2
            ready = torch.cuda.Event()
            with torch.cuda.stream(copy):
                copy.wait_event(free[b])
                bufs[b][0].copy_(pin[e % ROUTING_POOL][0], non_blocking=True)
                bufs[b][1].copy_(pin[e % ROUTING_POOL][1], non_blocking=True)
                ready.record(copy)
            main.wait_event(ready)
            out.index_add_(0, idd, F.linear(_act(F.linear(xd[idd], bufs[b][0])), bufs[b][1]))
            free[b].record(main)

    cut = {"cpu/ram": 1 << 30, "cuda/ram": 0}.get(lane)
    cut = int(lane.split("/c", 1)[1]) if cut is None else cut
    on_card = [c for c in calls if len(c[1]) >= cut]
    on_host = [c for c in calls if len(c[1]) < cut]

    def run() -> torch.Tensor:
        out = torch.zeros(T, H, dtype=torch.bfloat16, device=CUDA)
        th = threading.Thread(target=cpu_side, args=(on_host,)) if on_host else None
        if th is not None:
            th.start()
        card_side(on_card, out)
        if th is not None:
            th.join()
            out += out_cpu.to(CUDA, torch.bfloat16)
        torch.cuda.synchronize()
        return out

    err = _check(f"{lane} L{layer} call {call}", run(), ref)
    _placed(benchmark, lane, "express", f"{H}x{I}", T, err, row=f"L{layer}-c{call}-r{T}")
    rows_max = len(calls[0][1]) if calls else 0
    benchmark.extra_info.update(experts=len(calls), on_card=len(on_card), rows_max=rows_max)  # type: ignore[attr-defined]
    benchmark(run)  # type: ignore[operator]
