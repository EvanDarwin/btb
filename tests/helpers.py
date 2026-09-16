# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The suite's library: the fixture paths and the machine gates, engines built the way the tests build them, the
receipt tree every banked comparison walks, random MXFP4 matrices, the scheduler's stub engines and an HTTP client
for the served routes. A suite run as a script (`python tests/test_receipts.py`) puts the checkout on the path
before importing this."""

from __future__ import annotations

import collections
import http.client
import json
import os
import threading
import types
from collections.abc import Iterable, Sequence
from concurrent.futures import Future
from types import ModuleType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import pytest
import torch

from btb.kinds import Json, Parents, TokenRows, Tokens

if TYPE_CHECKING:
    from pytest import MonkeyPatch
    from transformers.cache_utils import DynamicCache

    from btb.engine.experts import _ExpertStore
    from btb.engine.model import StreamedTextModel
    from btb.engine.native import _Cuda
    from btb.serve import ModelRegistry

# --- the checkout and its fixtures ---------------------------------------------------------------------------

TESTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS)
FIXTURES = os.path.join(TESTS, "fixtures")
GGUF_FIXTURES = os.path.join(FIXTURES, "gguf")

# byte sizes, so the suite reads 4 * MB instead of 4 << 20
KB = 1 << 10
MB = 1 << 20
GB = 1 << 30


def fixture(name: str) -> str:
    """tests/fixtures/<name>; one that is not here skips the test (tests/make_fixtures.py builds them)"""
    p = os.path.join(FIXTURES, name)
    if not os.path.exists(p):
        pytest.skip(f"fixture {name} is not here (tests/make_fixtures.py)")
    return p


def checkout(*rel: str) -> str:
    """A file of the repository checkout; the wheel's verification runs these tests from a plain copy of tests/
    against the installed package, where the bench and the build files are not present."""
    p = os.path.join(ROOT, *rel)
    if not os.path.exists(p):
        pytest.skip(f"{os.path.join(*rel)} is not here: these tests run from a checkout")
    return p


def receipts(tag: str) -> Json:
    """the receipts make_fixtures banked for a family: receipts_<tag>.pt, the q35 hybrid's receipts.pt"""
    return torch.load(fixture("receipts.pt" if tag == "q35" else f"receipts_{tag}.pt"))


def safetensors_state(model_dir: str) -> dict[str, torch.Tensor]:
    """every tensor of a checkpoint's shards, by name"""
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for fn in sorted(os.listdir(model_dir)):
        if fn.endswith(".safetensors"):
            state.update(load_file(os.path.join(model_dir, fn)))
    return state


def cached(spec: str) -> str | None:
    """the local path of a cached Hub model or GGUF file, None when it is not on this machine (never a download)"""
    from btb import resolve

    try:
        return resolve(spec, local=True)
    except FileNotFoundError:
        return None


def layer_count(path: str) -> int:
    from transformers import AutoConfig

    full = AutoConfig.from_pretrained(path)
    return int(getattr(full, "text_config", full).num_hidden_layers)


# --- the machine ---------------------------------------------------------------------------------------------


def NO_LOG(*a: object, **k: object) -> None:
    """The engine's `log` callback silenced; passed where a test does not read the log."""


def need_cuda() -> str:
    """the card's device name, or a skip. A cpu `load()` earlier in the session called `cpu_only()`, which clears
    the package's flag and would send the test to the CPU without saying so; torch still holds the device, so
    the flag is put back (conftest restores it after the test)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import btb

    btb.CUDA = True
    return "cuda"


def need_mlx() -> None:
    from btb import mlx_available

    if not mlx_available():
        pytest.skip("MLX is not available")


def mlx_core() -> ModuleType | None:
    """mlx.core where MLX is installed, else None"""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx


def native_library() -> str | None:
    """the native library's path with its kernels bound to the engine, None where it is not built (the torch
    path answers then)"""
    from btb import native_path
    from btb.engine import StreamedTextModel

    p = native_path()
    if p:
        StreamedTextModel.load_gemv(p)
    return p


def need_native() -> str:
    p = native_library()
    if p is None:
        pytest.skip("no native library built")
    return p


def card_kernels() -> _Cuda | None:
    """the card's fused kernels where a card and btb_kernels.fatbin are here, else None"""
    if not torch.cuda.is_available():
        return None
    from btb import kernels_path
    from btb.engine.native import Native

    p = kernels_path()
    return None if p is None else Native.load_cuda(p)


def need_card_kernels() -> _Cuda:
    cu = card_kernels()
    if cu is None:
        pytest.skip("no CUDA card, or btb_kernels.fatbin not built (run `python build.py`)")
    return cu


# --- engines -------------------------------------------------------------------------------------------------


def host_model(
    path: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    packed: bool = False,
    cpu_layers: Iterable[int] | None = None,
    resident_layers: Iterable[int] = (),
    cold_layers: Iterable[int] = (),
    expert_cache_gb: float | None = None,
    prefill_chunk: int | None = None,
    kv_bits: int | None = None,
) -> StreamedTextModel:
    """the engine as the receipts run it: the head resident, every layer on the host tier unless placed
    otherwise, quiet; `packed` binds the 12-bit store"""
    from btb.engine import StreamedTextModel

    sm = StreamedTextModel(
        path,
        device=device,
        resident_head=True,
        log=NO_LOG,
        compute_dtype=dtype,
        cpu_layers=range(layer_count(path)) if cpu_layers is None else cpu_layers,
        resident_layers=resident_layers,
        cold_layers=cold_layers,
        expert_cache_gb=expert_cache_gb,
        prefill_chunk=prefill_chunk,
        kv_bits=kv_bits,
    )
    if packed:
        sm.open_packed()
        sm.bind_host_packed()
    return sm


def speculation(
    sm: StreamedTextModel,
    *,
    tree_budget: int | None = None,
    tree_min_prob: float | None = None,
    ngram_p: float | None = None,
    tree_read: str | None = None,
    v_max: int | None = None,
) -> None:
    """the speculative loop's knobs a test pins, each left alone when not given; the drafter never read from
    beside the model"""
    if tree_budget is not None:
        sm.tree_budget = tree_budget
    if tree_min_prob is not None:
        sm.tree_min_prob = tree_min_prob
    if ngram_p is not None:
        sm.ngram_p = ngram_p
    if tree_read is not None:
        sm.tree_read = tree_read
    if v_max is not None:
        sm.v_max = v_max
    sm.drafter_weights = None


def forward_logits(
    sm: StreamedTextModel,
    ids: torch.Tensor | Tokens | TokenRows,
    cache: DynamicCache | None = None,
    *,
    last_only: bool = True,
    positions: TokenRows | None = None,
) -> torch.Tensor:
    """`sm.forward` with the head returns logits, never None; asserted once here for the type checker"""
    out = sm.forward(ids, cache=cache, last_only=last_only, positions=positions)
    assert out is not None
    return out


# --- the receipt tree ----------------------------------------------------------------------------------------

# the tree every receipt walks (make_fixtures banks it, the receipt and device suites replay it): eight drafted
# tokens over the prompt, node j under PARENTS[j] at depth DEPTH[j], PATH the branch the commit keeps
PROMPT_Q35: TokenRows = [[3, 17, 42, 5, 99, 120, 7, 7, 300, 12, 45, 8, 3, 17, 60, 61]]
PROMPT_DENSE: TokenRows = [[3, 17, 42, 5, 99, 120, 7, 7, 200, 12, 45, 8, 3, 17, 60, 61]]
CHUNK: Tokens = [9, 33, 14, 71, 33, 5, 90, 2]
PARENTS: Parents = [-1, 0, 1, 2, 1, 4, 0, 6]
DEPTH: Sequence[int] = [0, 1, 2, 3, 2, 3, 1, 2]
PATH: Sequence[int] = [0, 1, 4, 5]


def tree_pass(sm: StreamedTextModel, cache: DynamicCache) -> tuple[torch.Tensor, int]:
    """the receipts' tree over `cache` (the prompt prefilled): the chunk verified as one pass at the tree's
    positions and the accepted path committed; (the pass's logits [T, V], the prompt's length)"""
    base = cache.get_seq_length()
    sm.aa(PARENTS)
    try:
        lg = forward_logits(sm, [CHUNK], cache, last_only=False, positions=[[base + d for d in DEPTH]])[0]
    finally:
        sm.ab()
    sm.ad(cache, base, PATH)
    return lg, base


def tree_next(sm: StreamedTextModel, cache: DynamicCache) -> torch.Tensor:
    """the step after the committed path: the logits over the last accepted token"""
    return forward_logits(sm, [[CHUNK[PATH[-1]]]], cache)[0, -1]


# --- comparisons ---------------------------------------------------------------------------------------------


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """the largest elementwise distance, as the receipts are held"""
    return float((a - b).abs().max())


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """the largest distance from `b` over `b`'s largest magnitude"""
    a, b = a.detach(), b.detach()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))


def bits(t: torch.Tensor) -> torch.Tensor:
    """a tensor as its bit pattern (bf16 as int16, else int32): an equality that tells NaN payloads and signed
    zeros apart"""
    t = t.contiguous()
    return t.view(torch.int16 if t.dtype is torch.bfloat16 else torch.int32)


# --- MXFP4 matrices ------------------------------------------------------------------------------------------


def mxfp4_random(
    rng: np.random.Generator | int, rows: int | Sequence[int], k: int, lo: int = 0, hi: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """random MXFP4 bytes for a [rows, k] matrix (or a stack of them, `rows` a shape prefix) in the checkpoint's
    layout: blocks [.., k/32, 16] uint8, then scales [.., k/32] uint8 drawn from [lo, hi); a seed or a generator"""
    from btb.mxfp4 import BLOCK, BLOCK_BYTES

    g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    shape = (rows,) if isinstance(rows, int) else tuple(rows)
    blocks = g.integers(0, 256, size=(*shape, k // BLOCK, BLOCK_BYTES), dtype=np.uint8)
    scales = g.integers(lo, hi, size=(*shape, k // BLOCK)).astype(np.uint8)
    return blocks, scales


def mxfp4_slot(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """one slot's bytes as the store holds them: the blocks, then the scales, flat"""
    return np.concatenate([blocks.reshape(-1), scales.reshape(-1)])


# --- the scheduler's stub engines ----------------------------------------------------------------------------

ABSENT = object()  # a config attribute that is not there at all, as against one that is there and None


def model_config(**kw: object) -> types.SimpleNamespace:
    """A model config: the four fields the scheduler reads, any of them replaceable by ABSENT (not present)."""
    base: dict[str, object] = {"num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "hidden_size": 512}
    base.update(kw)
    return types.SimpleNamespace(**{k: v for k, v in base.items() if v is not ABSENT})


class MlxLedger:
    """the unified-memory backend's ledger, the only part of it the scheduler asks about"""

    def __init__(self, held: int) -> None:
        self._held = int(held)

    def held_bytes(self) -> int:
        return self._held


class SchedulerModel:
    """the engine as the scheduler sees it: a config, the layer list, a device and the memory reserves"""

    def __init__(
        self,
        dev: str = "cpu",
        layer_types: Sequence[str] = ("full_attention",) * 4,
        kv_bits: int | None = None,
        compute_dtype: torch.dtype | None = None,
        cfg: types.SimpleNamespace | None = None,
        mlx: MlxLedger | None = None,
        mem_start: int = 0,
        ram_reserve: int = 0,
        vram_margin: int = 0,
    ) -> None:
        self.cfg = model_config() if cfg is None else cfg
        self.layer_types = list(layer_types)
        self.compute_dtype = compute_dtype
        self.dev = torch.device(dev)
        self.mlx = mlx
        self.kv_bits = kv_bits
        self.mem_start = int(mem_start)
        self.ram_reserve = int(ram_reserve)
        self.vram_margin = int(vram_margin)
        self.lines: list[str] = []

    def log(self, *a: object, **k: object) -> None:
        self.lines.append(" ".join(str(x) for x in a))


def stub_engine(**attrs: object) -> types.SimpleNamespace:
    """the least an engine the scheduler or the expert store is built over: a quiet log and the CPU, plus
    whatever the test adds"""
    return types.SimpleNamespace(**{"log": NO_LOG, "dev": torch.device("cpu"), **attrs})


class FakeRoute:
    """a scheduler whose reads never touch a drive: each `disk_read` is recorded and its future settled by hand"""

    DISK_DEMAND, DISK_AHEAD, DISK_SWEEP = 0, 1, 3

    def __init__(self) -> None:
        self.reads: list[Json] = []
        self.dropped: list[object] = []
        self.slow = False  # what `disk_slow` reports: a test turns it on and off

    def disk(self, path: str) -> Json:
        return {}

    def disk_slow(self) -> bool:
        return self.slow

    def disk_read(
        self,
        path: str,
        off: int,
        n: int,
        dst: torch.Tensor,
        priority: int = 0,
        key: object = None,
        on_done: Any = None,
        chunk: int = 0,
    ) -> Future[float]:
        f: Future[float] = Future()
        self.reads.append(
            {
                "path": path,
                "off": int(off),
                "n": int(n),
                "dst": dst,
                "priority": priority,
                "key": key,
                "on_done": on_done,
                "future": f,
            }
        )
        return f

    def disk_drop(self, key: object) -> int:
        n = 0
        for r in self.reads:
            if r["key"] == key and not r["future"].done():
                r["future"].cancel()
                n += 1
        self.dropped.append(key)
        return n

    def land(self, key: object) -> None:
        for r in self.reads:
            if r["key"] == key and not r["future"].done():
                if r["on_done"] is not None:
                    r["on_done"](1000)
                r["future"].set_result(1e-6)


def slot_size(st: _ExpertStore) -> int:
    """a store's slot size, once the store is sized"""
    assert st.per is not None
    return st.per


def expert_store(
    monkeypatch: MonkeyPatch,
    scheduler: object,
    *,
    n_layers: int,
    n_experts: int,
    hidden: int = 4,
    lookahead: tuple[int, ...] = (2, 1),
    lookahead_rows: tuple[int, ...] | None = None,
    ring_n: int = 64,
    routers: bool = False,
    files: str = "",
) -> tuple[_ExpertStore, types.SimpleNamespace]:
    """a store of 4096 slots of 64 KB over a stub engine on a box with 64 GB free, its two-part experts read from
    `files`gu<layer>.st and `files`dn<layer>.st (a recipe that needs no checkpoint); `routers` gives every layer a
    router whose top picks for a state are fixed (weight rows scaled so row e scores e for the all-ones input)"""
    from btb.engine import experts as experts_mod
    from btb.engine.families import Family
    from btb.kinds import FamilyKind

    monkeypatch.setattr(experts_mod, "host_free_bytes", lambda: 64 * GB)
    per = 64 * KB
    resident: dict[int, types.SimpleNamespace] = {}
    if routers:
        for L in range(n_layers):
            w = torch.zeros(n_experts, hidden)
            for e in range(n_experts):
                w[e] = float(e)  # the all-ones input ranks expert n_experts-1 first, then n_experts-2, ...
            gate = types.SimpleNamespace(weight=w)
            experts = types.SimpleNamespace(base=f"layers.{L}.mlp.experts.")
            resident[L] = types.SimpleNamespace(mlp=types.SimpleNamespace(gate=gate, experts=experts))
    attrs: dict[str, object] = {"lookahead_rows": lookahead_rows} if lookahead_rows is not None else {}
    sm = stub_engine(
        mlx=None,
        fam=Family(kind=FamilyKind.QWEN3),
        cold_chunk=0,
        expert_profile=None,
        resident=resident,
        host={},
        L=n_layers,
        n_experts=n_experts,
        lookahead=lookahead,
        scheduler=scheduler,
        **attrs,
    )
    st = experts_mod._ExpertStore(sm, budget_bytes=4096 * per, reserve_bytes=GB)
    st.per, st.n_slots, st.ring_n = per, 4096, ring_n
    st.shapes, st.sizes = (per // 2, (per // 4,), (per // 4,)), (per // 2, per // 2)
    st.stride, st.part_at = st._layout(st.sizes, False)
    st.block_max = 64 * per
    st.margin = st.block_max

    def recipe(layer: int, base: str) -> list[tuple[str, int, int, tuple[int, ...]]]:
        r = st.recipes.get(layer)
        if r is None:
            r = st.recipes[layer] = [
                (f"{files}gu{layer}.st", 0, per // 2, (1,)),
                (f"{files}dn{layer}.st", 0, per // 2, (1,)),
            ]
        return r

    monkeypatch.setattr(st, "_recipe", recipe)
    return st, sm


def bare_registry(device: str = "cuda:0", budget: int = 1 << 60) -> ModelRegistry:
    """a model registry with nothing scanned or loaded: the locks, the budget and an empty table"""
    from btb.options import DeviceName
    from btb.serve import ModelRegistry

    reg = ModelRegistry.__new__(ModelRegistry)
    name = DeviceName.parse(device)
    assert name is not None
    reg.device = name
    reg.budget = budget
    reg.log = None
    reg.loaded = collections.OrderedDict()
    reg.lock = threading.Lock()
    reg.state = threading.Lock()
    return reg


# --- the served routes ---------------------------------------------------------------------------------------


def request(
    base: str,
    method: str,
    path: str,
    body: bytes | Json | list[object] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], str]:
    """one request to a served endpoint at `base` (its URL): (status, the headers lower-cased, the body's text);
    a dict or list body is sent as JSON, and a 4xx is an answer, never an exception"""
    u = urlsplit(base)
    c = http.client.HTTPConnection(u.hostname or "127.0.0.1", u.port, timeout=10)
    data = body if body is None or isinstance(body, bytes) else json.dumps(body)
    c.request(method, path, data, {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    text = r.read().decode()
    hs = {k.lower(): v for k, v in r.getheaders()}
    c.close()
    return r.status, hs, text


def request_json(base: str, method: str, path: str, body: Json | None = None) -> tuple[int, Json]:
    """`request` whose answer is decoded as JSON: (status, the object)"""
    status, _, text = request(base, method, path, body)
    return status, json.loads(text)
