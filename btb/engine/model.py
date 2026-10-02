# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`StreamedTextModel`: the engine. Its behaviour is split by concern over the mixins of this package; this module
holds its construction and lifetime."""

from __future__ import annotations

import contextlib
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Concatenate, ParamSpec, Protocol, TypeVar

import torch

from .. import mlx as mlxdev
from .. import pool as _pool
from .. import trace
from ..gguf import GGUFModel
from ..hf import is_gguf, pack_format, shard_map
from ..kinds import LayerKind, Log, Tokens
from ..mlx.q6k import gather_q6k
from ..options import Device as DeviceKind
from ..options import DeviceName
from ..pack12 import parent_dir
from ..sysinfo import host_free_bytes, vram_pressure_close
from . import device as device_mod
from .cache import GrantedIndexedLayer, GrowLayer
from .cuda import _CudaMixin
from .device import Device, where
from .drafter import MTPDrafter
from .experts import _ExpertStore
from .families import _FamiliesMixin, family, register_attention
from .fixed_rows import RowLinear, fix_linears, fix_rows_cls
from .forward import _ForwardMixin
from .generate import _GenerateMixin
from .holdings import Holdings, Stage, last_on_card, on_card
from .host import _Experts, _HostLinear, _NGramRows, _Router
from .leaks import closed as _leaks_closed
from .leaks import track as _leaks_track
from .memory import RamPolicyState, VramPolicyState, _LendMixin
from .mlx_forward import MlxState, _MlxMixin
from .native import Native
from .scheduler import BatchScheduler, _size
from .scratch import Scratch
from .text import _TextMixin
from .tiers import ColdRing, _TiersMixin


class _Closes(Protocol):
    def close(self) -> None: ...


_P = ParamSpec("_P")
_E = TypeVar("_E", bound=_Closes)


def _closes_on_failure(init: Callable[Concatenate[_E, _P], None]) -> Callable[Concatenate[_E, _P], None]:
    """an engine's construction made whole or nothing: a failure part way closes what it had taken (its holdings
    so far) before the error goes on, rather than leaving it to the collector"""

    def build(self: _E, /, *args: _P.args, **kwargs: _P.kwargs) -> None:
        try:
            init(self, *args, **kwargs)
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise

    build.__doc__, build.__qualname__ = init.__doc__, init.__qualname__
    return build


class StreamedTextModel(
    _FamiliesMixin, _TiersMixin, _LendMixin, _MlxMixin, _CudaMixin, _ForwardMixin, _GenerateMixin, _TextMixin
):
    """The engine over one model. Construction and lifetime live here; the forward, the tiers, memory, the
    MLX and CUDA paths and decoding are the mixins (one module each in this package)."""

    ST_DTYPES = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,  # fine-grained FP8 weights (btb/fp8.py)
        "F8_E8M0": torch.float8_e8m0fnu,  # their scales stored as exponents
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    # the tests reach the helpers as `StreamedTextModel._HostLinear`; the native handles live on `Native` alone
    _HostLinear = _HostLinear
    _Router = _Router
    _Experts = _Experts
    _NGramRows = _NGramRows
    _ExpertStore = _ExpertStore
    MTPDrafter = MTPDrafter
    mlx: Any = None
    gemm_rows = Native.gemm_rows
    cpu_gemm_rows = Native.cpu_gemm_rows

    @classmethod
    def load_gemv(cls, dll_path: str | os.PathLike[str], threads: int = 0) -> Callable[..., Any]:
        """bind the native library's kernels (once per process); see `Native.load_gemv`"""
        return Native.load_gemv(dll_path, threads)

    @_closes_on_failure
    def __init__(
        self,
        model_dir: str,
        device: Any = None,
        resident_head: bool = True,
        attn_impl: str = "btb_sdpa",
        log: Log = print,
        prefetch: bool = False,
        resident_layers: Iterable[int] = (),
        compute_dtype: torch.dtype | None = None,
        cpu_layers: Iterable[int] = (),
        cold_layers: Iterable[int] = (),
        cold_slots: int = 2,
        cold_chunk_mb: int = 16,
        prefill_card: bool = False,
        prefill_card_min: int = 64,
        resident_fp32: bool = False,
        kv_host: bool = False,
        prefill_chunk: int | None = None,
        context: int | None = None,
        rope_scaling: Any = None,
        original_max_position_embeddings: int | None = None,
        expert_cache_gb: float | None = None,
        ram_reserve_gb: float | None = None,
        vram_reserve_gb: float | None = None,
        adapt: bool = True,
        mlx_layers: Iterable[int] | None = None,
        kv_bits: int | None = None,
        gguf_packed: bool = True,
        host_budget: Any = None,
        bus_pass: bool = True,
        store_pin: int = 0,
        sparse: bool = False,
        weights: bool = True,
    ) -> None:
        from transformers import AutoConfig

        t0 = time.time()
        # everything the engine holds past a pass, each registered where it is made: `close` is this run
        self.holdings = Holdings()
        _leaks_track(self)
        self.scratch = Scratch(self)
        self.gguf = GGUFModel(model_dir) if is_gguf(model_dir) else None
        self.gguf_packed = bool(gguf_packed)  # its Q4/Q8 tensors on the packed kernels as stored (MLX), else bf16
        self.dir = self.gguf.dir if self.gguf is not None else model_dir
        self.mlx = None
        self.mlx_state = MlxState()
        # the RAM this load started with (pool.py samples it before anything is allocated for the load; a
        # model built directly samples now): what the engine may use, against which it counts what it holds
        # free-read: the MLX ledger's baseline, sampled before the ledger exists
        self.mem_start = _pool.MEM_START if _pool.MEM_START is not None else host_free_bytes() + _pool.POOL.free_bytes()
        self.mlx_layers = set()
        # the attention cache's rows as int8 with a scale each (the MLX device; `GrowLayer(bits=)`): half the
        # bytes of bf16, so every attention row goes through the node kernel that reads them as they are
        self.kv_bits = int(kv_bits) if kv_bits else None
        if self.kv_bits not in (None, 8):
            raise ValueError(f"[stream] kv_bits must be 8 or unset, got {kv_bits}")
        self.mlx_attn_rows = 0 if self.kv_bits else 8192
        # a prefill chunk's attention at head size 256 through the engine's fused prefill kernel instead of MLX's
        # fallback (its fused attention takes 64, 80 and 128): in situ at a 40k position the attention runs 1.2x
        # faster and the chunk 1.08-1.16x, the same math
        self.mlx_attn_prefill = True
        devname = DeviceName.parse(device) or DeviceName(DeviceKind.CPU)
        if devname.kind is DeviceKind.MLX:
            if not mlxdev.available():
                raise RuntimeError("[mlx] MLX is not available in this Python (pip install mlx; Apple silicon only)")
            self.mlx = Native.mlx = mlxdev.Backend(self, gemm_rows=Native.gemm_rows, log=log)
            self.holdings.own(Stage.VIEWS, "the MLX tier", self._mlx_teardown)
            self.mlx_layers = (
                {int(x) for x in mlx_layers}
                if mlx_layers is not None
                else {int(x) for x in cpu_layers} | {int(x) for x in cold_layers}
            )
            devname = DeviceName(DeviceKind.CPU)  # MLX's tensors on the torch side are host tensors
        # the card by its index, as its tensors name it (a bare 'cuda' is the current card)
        self.dev = where(str(devname))
        if self.dev.type == DeviceKind.CUDA:
            # make the chosen card the process's current device, so ops that take no explicit device (a
            # library's default allocation, the default stream) land on it too, not on cuda:0
            torch.cuda.set_device(self.dev)
        self.log = log
        self.bytes_streamed = 0
        self.load_s = 0.0
        self.compute_s = 0.0
        self.prefetch = bool(prefetch) and self.dev.type == DeviceKind.CUDA
        self.compute_dtype = compute_dtype
        full = self.gguf.config() if self.gguf is not None else AutoConfig.from_pretrained(model_dir)
        cfg = getattr(full, "text_config", full)
        if attn_impl == "btb_sdpa":
            register_attention()
        cfg._attn_implementation = attn_impl
        self.cfg = cfg
        self.fam = family(cfg)
        # the fused rope and the in-place SwiGLU, fixed here for the engine's life: every path that rotates q and k
        # itself reads one rope (`_rope_fn`), whichever of them runs first
        self._frope = os.environ.get("BTB_FUSED_ROPE", "1") != "0"
        self._fmlp = os.environ.get("BTB_FUSED_MLP", "1") != "0"
        # a mixture's chunked prefill layer by layer (each expert read once a prompt); 0 takes the chunks through
        # every layer in turn, the path it replaced and its bits' reference
        self.prefill_layers = os.environ.get("BTB_PREFILL_LAYERS", "1") != "0"
        # a prefill's expert calls on the card as grouped matmuls over the depot's slots, the per-expert loop's bits;
        # 0 keeps the loop
        self.grouped_experts = os.environ.get("BTB_GROUPED_EXPERTS", "1") != "0"
        # Qwen4's sparse attention picks its blocks for every row in one pass (`--sparse`): not the reference's
        # mask bit for bit at a near-tie; off, the selection is the reference's exactly (qsa.py)
        self.sparse = bool(sparse)
        # Gemma scales the input embedding by sqrt(hidden); the engine gathers rows itself, so it applies the
        # scale the module's scaled embedding would (the tied head's output projection stays unscaled)
        self.embed_scale = float(cfg.hidden_size) ** 0.5 if self.fam.embed_scale else None
        self.fam.prepare(self)
        self.n_experts = int(getattr(cfg, "num_local_experts", 0) or getattr(cfg, "num_experts", 0) or 0)
        self.L = int(cfg.num_hidden_layers)
        self.layer_types = [LayerKind.of(t) for t in (getattr(cfg, "layer_types", None) or [LayerKind.FULL] * self.L)]
        self.pack = None if self.gguf is not None else pack_format(model_dir)
        if self.gguf is not None:
            # the family's tensor names against the file's, through llama.cpp's table; the file is the one shard
            self._gguf_names, self._gguf_hdr = self.gguf.weight_map(self._family_tensor_names(cfg), self.L)
            self.weight_map = dict.fromkeys(self._gguf_names, self.gguf.file)
            self.log(f"[gguf] {self.gguf.describe()}")
        elif self.pack is None:
            self.weight_map = shard_map(model_dir)
        else:
            # a 12-bit model reads the rest of its tensors from the parent it names, by a path relative to the
            # model (a sibling beside it, or its cache entry), absolute only where no relative path exists
            parent = parent_dir(model_dir)
            try:
                rel = os.path.relpath(parent, os.path.abspath(model_dir))
            except ValueError:  # another drive
                rel = parent
            self.weight_map = {k: os.path.join(rel, f) for k, f in shard_map(parent).items()}
        emb_keys = [k for k in self.weight_map if k.endswith("embed_tokens.weight") and "visual" not in k]
        if len(emb_keys) != 1:
            raise RuntimeError(f"[stream] expected one text embedding key, found {emb_keys}")
        self.prefix = emb_keys[0][: -len("embed_tokens.weight")]
        self.head_key = "lm_head.weight" if "lm_head.weight" in self.weight_map else emb_keys[0]
        self._maps = {}
        self.holdings.own(Stage.FILES, "the checkpoint's maps and handles", self._close_files)
        self.holdings.own(Stage.MEMORY, "the weights", self._drop_weights)
        # the embedding carries the checkpoint's own weight precision: an fp16 or fp32 one is held as bf16
        self.held_cast = False
        if self.gguf is None:
            _mm, hdr, _ = self._shard(self.weight_map[emb_keys[0]])
            self.held_cast = self.ST_DTYPES[hdr[emb_keys[0]]["dtype"]] != torch.bfloat16
        self.fp8_experts = any(self._fp8(k) for k in self.weight_map if k.endswith(".mlp.experts.gate_up_proj"))
        self.fp8_layers = set()
        self.fp8_widened = False
        # set by close(): every decode loop ends at its next step, so no thread is mid-pass when the buffers go
        self.abort = threading.Event()
        self._decode_lock = threading.RLock()  # one decode at a time on the engine (MLX's worker serializes too)
        self._meta = torch.device("meta")
        self.expert_stat = {"experts": 0, "bytes": 0, "calls": 0, "s": 0.0}
        self.expert_trace = None
        self.expert_profile = None
        self.expert_store = None
        with self._meta:
            closing, last = self.fam.closing(cfg)
        # a family's final norm is `norm`, which the fused paths fold into their graphs; a family without one closes
        # with its own module (Qwen4's mixer of its streams), `mixer`, run as its module runs
        self.norm, self.mixer = (last, None) if self.fam.norm is not None else (None, last)
        fix_rows_cls(self.fam.norm)  # the final norm's small passes at the fixed shape (fixed_rows.py)
        # a closing mixer's linears too: its small passes' rows on the card a row's own whatever travels with them
        # (Qwen4's, whose plain linears parted a verify row from its step)
        fix_linears(last)
        # the parameters are widened to a float32 compute dtype below: read at the checkpoint's own precision for it
        wide = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        for name, _, is_buf in self._named_tensors(last):
            t = self._get(f"{self.prefix}{closing}.{name}", stored=wide and not is_buf)
            self._adopt(last, name, t, buffer=is_buf)
        self.resident_head = resident_head
        self.head = None
        self.head_host = None
        if resident_head and self.mlx is not None:
            # `gguf_shortcut`: this weight rebinds packed right below, so a wasted bf16 dequant would be
            # discarded immediately -- unlike the embed table below, which needs real values for row-gather
            self.head_host = _HostLinear(self._get(self.head_key, gguf_shortcut=True), key=self.head_key)
            self._bind_mlx_linears([self.head_host])
        elif resident_head:
            with self._meta:
                # the head's small passes at the fixed shape, as the card layers' (fixed_rows.py)
                self.head = RowLinear(cfg.hidden_size, cfg.vocab_size, bias=False)
            self._adopt(self.head, "weight", self._get(self.head_key))
        # the embedding table, unless the tied head is Q6_K: then its packed bytes are the table, gathered on demand
        # (embed / _mlx_embed_rows) instead of a bf16 copy of the whole vocabulary
        tied_q6k = self.head_host is not None and getattr(self.head_host.mx, "q6k", None) is not None
        if tied_q6k and self.head_key == self.prefix + "embed_tokens.weight":
            self.embed_table = None
        else:
            self.embed_table = self._get(self.prefix + "embed_tokens.weight")
        native = int(getattr(cfg, "max_position_embeddings", 0) or 0)
        self.context = int(context) if context else None
        rp = dict(getattr(cfg, "rope_parameters", None) or {})
        if rope_scaling:
            rp.update(rope_scaling)
            cfg.rope_parameters = rp
        elif self.context and native and self.context > native and rp.get("rope_type", "default") == "default":
            orig = int(original_max_position_embeddings or native)
            factor = float(math.ceil(self.context / orig))
            rp.update({"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": orig})
            cfg.rope_parameters = rp
            cfg.max_position_embeddings = max(native, int(orig * factor))
            self.log(
                f"[rope] yarn factor {factor:g} over {orig} positions for a context of {self.context} "
                f"(the model's window is {native})"
            )
        self.rotary = self.fam.rotary(config=cfg).to(self.dev)
        _res_set = {int(x) for x in resident_layers}
        _host_set = {int(x) for x in cpu_layers} - _res_set
        # `weights` off: an engine opened to price its layers from the checkpoint's headers (`btb.plan`), which
        # runs no pass - so it reads no layer into a template (on the CPU a copy of the whole layer) and builds no
        # expert store
        self._streamed_any = weights and any(i not in _res_set and i not in _host_set for i in range(self.L))
        self.prefill_card = bool(prefill_card) and self.dev.type == DeviceKind.CUDA
        self.prefill_card_min = int(prefill_card_min)
        if self.prefill_card:
            self._streamed_any = True
            self.prefetch = True
        if not self._streamed_any:
            self.prefetch = False
        self.templates = {}
        self._toggle = {}
        self._staging = {}
        fp32_compute = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        for lt in sorted(set(self.layer_types)) if self._streamed_any else []:
            idx = self.layer_types.index(lt)
            n_t = 1 if (self.prefetch and fp32_compute) else (2 if self.prefetch else 1)
            self.templates[lt] = []
            for _ in range(n_t):
                tmpl = self._new_layer(idx)
                self._load_layer(idx, tmpl, first=True)
                self.templates[lt].append(tmpl)
            self._toggle[lt] = 0
            if self.prefetch:
                self._staging[lt] = {
                    name: torch.empty_like(p, device="cpu").pin_memory()
                    for name, p in self.templates[lt][0].named_parameters()
                }
        if self.prefetch:
            self._copy_stream = torch.cuda.Stream(device=self.dev)
            self._pending = None
            self._thread_mod = threading
            self.wait_s = 0.0
        self._dma_done = {}
        self._events = []
        self.resident = {}
        # initializing: the plan still bends to another program taking the card while the layers load (a layer not
        # yet placed goes to the host instead); a layer placed stays until the engine is up, when `adapt` takes over
        given_up = self._place_planned(sorted({int(x) for x in resident_layers}), bool(adapt))
        self.cold = {int(x) for x in cold_layers} - set(self.resident)
        self.cold_slots = int(cold_slots)
        self._mega = None
        self._mlx_attn_slope = None
        self.cold_chunk = int(cold_chunk_mb) << 20
        self.cold_ring = ColdRing()
        self.host = {}
        for i in sorted({int(x) for x in cpu_layers} | self.cold | given_up):
            if i in self.resident:
                continue
            self.host[i] = self._make_host_layer(i)
            trace.event(
                "load: layer %d on the host (%s)", i, "read from the drive each pass" if i in self.cold else "in RAM"
            )
        self.mlx_state.ahead = None
        self._bind_cold()
        # the RAM kept free of the engine's own grants and the store's budget: the plan's host budget where the
        # load drew one, else taken now - the scheduler's one floor
        if ram_reserve_gb is not None:
            self.ram_reserve = int(float(ram_reserve_gb) * 2**30)
        else:
            self.ram_reserve = int((host_budget if host_budget is not None else BatchScheduler.measure_host()).reserve)
        if self.dev.type == DeviceKind.CUDA:
            total_vram = torch.cuda.get_device_properties(self.dev).total_memory
            # the margin the load's plan left free (`vram_margin_gb`), so the engine does not read itself as
            # short of the card the moment it is up
            self.vram_margin = int(
                (float(vram_reserve_gb) if vram_reserve_gb is not None else BatchScheduler.vram_margin_gb(total_vram))
                * 2**30
            )
        else:
            self.vram_margin = 0
        self.scheduler = BatchScheduler(self)
        self.plan = None
        self.device = Device(self)
        if self.dev.type == DeviceKind.CUDA:
            on_card(self)
            self.holdings.own(Stage.MEMORY, "the card graphs and their arena", self._graphs_close)
        # the expert store reads both at construction, so they are set here and never afterwards: the Bus Pass by
        # default (1-8% on the token over two pairs on NVMe, 11% fewer misses on the replay's warm passes,
        # bookkeeping its only cost), and the store's pages pageable unless `store_pin` asks for pinned ones
        self.bus_pass = bool(bus_pass)
        self.store_pin = int(store_pin)
        if weights and self.fam.moe and Native.read_direct is not None and expert_cache_gb != 0:
            if expert_cache_gb is None:
                # the MLX tier's ledger: the RAM the load started with, less the reserve, less what MLX holds; the
                # OS's free count reads high for untouched Metal buffers
                if self.mlx is not None:
                    # the base re-read with the trunk and the pool bound and touched: the figure sampled at import was
                    # taken while the previous process was still releasing memory, and capped the store 5-10 GB low
                    # free-read: the MLX ledger's own baseline, set here
                    self.mem_start = max(self.mem_start, host_free_bytes() + self.mlx.held_bytes())
                # what the ledger has free on the host above the reserve (the commit left included on Windows)
                room = int(self.device.free(torch.device("cpu"), unreserved=True) or 0)
                budget = max(0, room)  # the store keeps its scratch slots whatever the room
            else:
                budget = int(float(expert_cache_gb) * 2**30)
            self.expert_store = _ExpertStore(self, budget, self.ram_reserve)
            self.holdings.own(Stage.MEMORY, "the expert store", self._close_store)
        # the memory policies give layers up when another program needs the memory and take them back after;
        # `adapt` off pins the placement taken at load
        self.adapt = bool(adapt)
        self.vram_watch = self.adapt and self.dev.type == DeviceKind.CUDA
        self.vram_state = VramPolicyState()
        # the host tier's policy: on where layers run from RAM off unified memory (which keeps its own ledger)
        self.ram_watch = self.adapt and bool(self.host) and self.mlx is None
        self.ram_state = RamPolicyState()
        self._shed = []
        self.drafter_dev = None
        self.pinned = {}
        self.shadow = {}
        self.kv_host = bool(kv_host) and self.dev.type == DeviceKind.CUDA
        self.prefill_chunk = int(prefill_chunk) if prefill_chunk else None
        self.kv_block = 4096
        self._kv_stage = None
        self._batched_cont = False
        self.resident_fp32 = bool(resident_fp32)
        if self.compute_dtype is not None and self.compute_dtype != torch.bfloat16:
            # one float32 shadow per layer kind and structure, built from a layer of that structure: layers of one
            # kind can differ (tiny_q4's layer 1 carries a `ple` block its kind's other layers lack), and a shadow
            # of the wrong one had the upcast copy a (256, 64) weight into a 64-wide slot. The kind stays in the
            # key: two kinds can share a structure (gemma3's sliding and full layers) and still run apart
            srcs: dict[tuple[Any, tuple[Any, ...]], tuple[int, Any]] = {}
            for lt in self.templates:
                tmpl = self.templates[lt][0]
                srcs.setdefault((lt, self._structure(tmpl)), (self.layer_types.index(lt), tmpl))
            if not self.resident_fp32:
                for i, tmpl in self.resident.items():
                    srcs.setdefault((self.layer_types[i], self._structure(tmpl)), (i, tmpl))
            for (lt, key), (idx, src_mod) in srcs.items():
                sh = self._new_layer(idx)
                # the buffers too, at their own dtype: a layer's buffer (tiny_q4's `ple.layer_multipliers`) left as
                # `_new_layer` made it stayed on the meta device
                for name, src, is_buf in list(self._named_tensors(src_mod)):
                    t = src.data if is_buf else src.data.to(self.compute_dtype)
                    self._set_param(sh, name, t.clone() if is_buf else t, buffer=is_buf)
                self.shadow.setdefault(lt, {})[key] = sh
            for m in (self.norm, self.mixer):
                if m is not None:
                    for p in m.parameters():
                        p.data = p.data.to(self.compute_dtype)
            if self.resident_fp32:
                for tmpl in self.resident.values():
                    for p in tmpl.parameters():
                        p.data = p.data.to(self.compute_dtype)
        # the threads writing into the engine's buffers, stopped before any of them is let go (the last registered
        # first: the cold ring's reader, which waits on the drive's, before them), and the record they complete,
        # written once they have stopped
        self.holdings.own(Stage.STOP, "the expert store's and the MLX tier's workers", self._stop_workers)
        self.holdings.own(Stage.STOP, "the drive's readers", self.scheduler.disk_close)
        self.holdings.own(Stage.STOP, "the cold ring's reader", self._cold_stop)
        self.holdings.own(Stage.RECORD, "the expert profile", self._save_profile)
        self.holdings.own(Stage.MEMORY, "a verify pass's recurrent-state checkpoints", self._drop_spec_state)
        # free-read: the load's log line
        res = torch.cuda.memory_allocated(self.dev) / 2**30 if self.dev.type == DeviceKind.CUDA else 0.0
        n_templates = sum(len(v) for v in self.templates.values())
        self.log(
            f"[stream] {model_dir}: {self.L} layers ({n_templates} templates, prefetch "
            f"{'on' if self.prefetch else 'off'}), prefix '{self.prefix}', head "
            f"{'resident' if resident_head else 'streamed'}; resident {res:.2f} GB; init {time.time() - t0:.1f}s"
        )
        # the placement taken: from here another program asking for the card, or for memory, gets it back at once
        # (`adapt`)
        self.watch_vram_budget()
        self.watch_ram()
        if self.mlx is not None:
            n_cpu = len([i for i in self.host if i not in self.mlx_layers])
            n_cold = len([i for i in self.cold if i in self.mlx_layers])
            self.log(
                f"[mlx] {len(self.mlx_layers) - n_cold} layers and the head in unified memory ({self.mlx_state.bytes / 2**30:.2f} GB"
                f"), {n_cold} streamed from the drive each pass, {n_cpu} on the CPU kernels; "
                f"MLX active {self.mlx.active_bytes() / 2**30:.2f} GB"
            )

    # what the card's budget may move by, beyond the engine's own allocations, before a layer still to place is given
    # up: the driver's own bookkeeping moves the reading by tens of MB
    INIT_SLACK = 256 << 20

    def _place_planned(self, planned: Sequence[int], adapt: bool) -> set[int]:
        """The plan's card layers placed one at a time while the engine initializes, the plan still open to the card
        changing under it: before each, this process's WDDM budget room is read (microseconds) against what it was
        when placing began less what the layers placed since took. Where it fell further - another program took the
        card meanwhile (a game launched during a load of minutes) - the layers still to place that the gap would take
        (the last ones, as a shed would give them up) go to the host instead, and the plan is the smaller one from
        there. A layer already placed is never moved while initializing; past the budget still once the engine is
        up, the VRAM policy gives it back (`vram_yield`). Returns the layers given up. With `adapt` off, or no
        budget to read (off Windows), the plan as it was"""
        dev = self.dev
        r0 = device_mod._wddm_room(dev) if (adapt and dev.type == DeviceKind.CUDA) else None
        # the engine's own growth while it places its layers, to tell it from another program's in the budget's room:
        # the ledger is not made yet (initializing), and this is no allocation's decision
        # free-read: the engine's own growth while initializing, before its ledger exists
        base = torch.cuda.memory_reserved(dev) if r0 is not None else 0
        todo = list(planned)
        sizes: list[int] = []
        given_up: set[int] = set()
        while todo:
            if r0 is not None and sizes:
                room = device_mod._wddm_room(dev)
                # free-read: the engine's own growth since placing began (above), set against the budget's room
                taken = 0 if room is None else (r0 - (torch.cuda.memory_reserved(dev) - base)) - room
                if taken > self.INIT_SLACK:
                    per = max(1, sum(sizes) // len(sizes))
                    k = min(len(todo), -(-taken // per))
                    todo, give = todo[: len(todo) - k], todo[len(todo) - k :]
                    given_up.update(give)
                    r0 -= taken  # the smaller card is the plan's from here
                    self.log(
                        f"[plan] another program took {taken / 2**30:.2f} GB of the card while loading: "
                        f"{k} layer{'s' if k != 1 else ''} ({give[0]}-{give[-1]}) to the host instead"
                    )
                    if not todo:
                        break
            i = todo.pop(0)
            sized = r0 is not None or (trace.ON and dev.type == DeviceKind.CUDA)
            # free-read: a placed layer's own size on the card, to size the layers still to place (above)
            before = torch.cuda.memory_reserved(dev) if sized else 0
            tmpl = self._new_layer(i)
            self._load_layer(i, tmpl, first=True)
            self.resident[i] = tmpl
            if sized:
                # free-read: the same layer's size, after
                size = max(0, torch.cuda.memory_reserved(dev) - before)
                trace.event("load: layer %d on the card (%s, +%s)", i, dev, _size(size))
                if r0 is not None:
                    sizes.append(size)
            else:
                trace.event("load: layer %d on %s", i, dev)
        return given_up

    def embed(self, ids: Any) -> torch.Tensor:
        rows = self._embed_rows(torch.as_tensor(ids, dtype=torch.long))
        return rows if self.embed_scale is None else rows * self.embed_scale

    def _embed_rows(self, ids: torch.Tensor) -> torch.Tensor:
        head = self.head
        if (
            head is not None
            and self.head_key == self.prefix + "embed_tokens.weight"
            and head.weight.device.type != "cpu"
        ):
            # tied embeddings: the head on the card is the table - one gather there, no host round trip
            return torch.nn.functional.embedding(ids.to(head.weight.device), head.weight)
        if self.embed_table is None:  # tied Q6_K: gather the rows straight from the head's packed bytes
            hh = self.head_host
            assert hh is not None
            raw, _rows, cols = hh.mx.q6k
            tok = mlxdev.mx().array(ids.reshape(-1).to(torch.int32).cpu().numpy())
            rows = mlxdev.from_mx(gather_q6k(raw, tok, cols))
            return rows.view(*ids.shape, cols).to(self.dev)
        rows = self.embed_table[ids.reshape(-1).cpu()]
        return rows.view(*ids.shape, self.embed_table.shape[1]).to(self.dev)

    def warm(self) -> int:
        """Ready as part of the load: the card's graphs and pass-cost curve where the card graph applies, the MLX
        pass-cost curve where the fused tree applies, over a throwaway prompt. Returns what was captured or
        timed (0 elsewhere)."""
        ids = [1] * 8
        if self.dev.type == DeviceKind.CUDA:
            with self._warming():
                n = int(self.card_warm(ids))
                # the warm-up's passes and its throwaway cache let go: an engine up and idle does not sit on 1.5 GB
                # of torch's freed blocks (Qwen3-4B, 12 layers on the card) until its first pass, a budget's worth
                # that a program asking for the card meanwhile saw a layer shed for instead
                self.vram_trim("warm-up")
            return n
        if self.mlx is not None:
            n = int(self.mlx_warm(ids))
            if n:
                # a pass's attention grows with the rows a node reads: its cost per node and row, timed on
                # the node kernel over synthetic caches, is what the budget adds for the session's length
                self.mlx_attn_cost()
            return n
        if self.dev.type == DeviceKind.CPU:
            with self._warming():
                return int(self.host_warm(ids))
        return 0

    @contextlib.contextmanager
    def _warming(self) -> Iterator[None]:
        """The warm-up's timing, the memory policies standing aside for it and answering once it is done: the decode
        lock held as a decode holds it, so the watchers (`watch_vram_budget`, `watch_ram`) leave it be, and
        `warming` set, so its own passes' `vram_policy` and `ram_policy` give nothing back mid-timing - a yield there
        let the card graphs go and moved the placement between two captures (the warm-up's kernel choice written
        into a card state no pass read), a ram-yield moved a host layer to the drive inside a timed pass. Room made
        on demand (`_give_up_one`) still moves what it must. Once done, each policy reads its budget: a program
        that asked meanwhile (its signal kept by the watcher) is answered now, not at the first request"""
        with self._decode_lock:
            self.warming = True
            try:
                yield
            finally:
                self.warming = False
            self.vram_policy(None)
            self.ram_policy()

    def host_warm(self, ids: Tokens, t_max: int | None = None) -> int:
        """The cost of a verify pass of 1 .. `t_max` rows on the host tier, timed over a throwaway cache of
        `ids` into `_host_cost` (each width paired with a one-row pass, the fastest kept, as the MLX curve):
        the budget weighs a tree's rows against it, and the loop verifies trees on this tier once it exists.
        Returns the widths timed; 0 where the host does not hold every layer."""
        if self.dev.type != DeviceKind.CPU or self.cold or self.mlx is not None:
            return 0
        t_max = int(t_max or self._spec_full())
        t_max = max(1, min(t_max, 16))
        ids_t = torch.as_tensor(list(ids), dtype=torch.long).view(1, -1)
        cost: dict[int, float] = {}
        with torch.inference_mode():
            cache = self.new_cache(max_len=int(ids_t.shape[1]) + t_max + 2)
            self._prefill(ids_t, cache)
            tok = int(ids_t[0, -1])

            def timed(T: int) -> float:
                base = cache.get_seq_length()
                self.aa(list(range(-1, T - 1)))
                t0 = time.perf_counter()
                try:
                    self.forward([[tok] * T], cache=cache, last_only=False, positions=[[base + t for t in range(T)]])
                finally:
                    self.ab()
                dt = time.perf_counter() - t0
                self.ad(cache, base, [0])
                for cl in cache.layers:
                    if isinstance(cl, GrowLayer) and cl.get_seq_length() > base:
                        cl.crop(base)
                    elif getattr(cl, "keys", None) is not None and cl.keys.shape[-2] > base:
                        cl.keys = cl.keys[..., :base, :]
                        cl.values = cl.values[..., :base, :]
                        if hasattr(cl, "cumulative_length"):
                            cl.cumulative_length = base
                return dt

            one = timed(1)
            for T in range(1, t_max + 1):
                best = float("inf")
                for _rep in range(3 if T <= 8 else 2):
                    one = min(one, timed(1))
                    best = min(best, timed(T))
                cost[T] = best
            cost[1] = min(cost.get(1, one), one)
        if cost:
            self._host_cost = cost
            c1 = cost[1]
            self.log(
                "[host] pass cost by rows: "
                + ", ".join(f"{T}:{c / c1:.2f}x" for T, c in cost.items() if T in (1, 2, 4, 8, 12, t_max))
                + f" (one row {c1 * 1e3:.1f} ms)"
            )
        return len(cost)

    def _family_tensor_names(self, cfg: Any) -> list[str]:
        """every tensor the family's layers, embedding, norm (Qwen4's closing mixer) and head are loaded from, by
        its HF name: what a GGUF file's tensors are matched against. Each layer is its own: a hybrid family's
        layers differ in kind."""
        names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        with torch.device("meta"):
            closing, last = self.fam.closing(cfg)
            # a closing module other than the norm named above (Qwen4's mixer) adds its own
            names += [n for t, _, _ in self._named_tensors(last) if (n := f"model.{closing}.{t}") not in names]
            for i in range(self.L):
                names += [f"model.layers.{i}.{n}" for n, _, _ in self._named_tensors(self.fam.layer(cfg, i))]
        return names

    def close(self) -> None:
        """Stop any decode in flight and let go of everything the engine holds (`holdings`): the threads writing
        into its buffers first, then the records they complete, the arrays over other buffers, the buffers, the
        files. Each release runs on its own; one that raises stays held for the next call to try again, and the
        errors are raised together once every other release has run. Safe to call twice; the engine is unusable
        afterwards. `with btb.load(...) as model:` calls it on exit. What is still held after is logged
        (`leaks.closed`)."""
        if getattr(self, "_closed", False):
            return
        draft = getattr(self, "draft_engine", None)
        draft_error: Exception | None = None
        if draft is not None:
            self.draft_engine = None  # the draft model is its own engine (a --draft-model load): released first
            try:
                draft.close()
            except Exception as e:
                # on its own, as every release is: the engine's own holdings are still let go, the error raised with
                # theirs, and the draft kept for the next close to try again
                e.add_note("[close] releasing the draft model")
                draft_error = e
        if getattr(self, "abort", None) is not None:
            self.abort.set()
        # the MLX tier's teardown runs on the model's worker thread, where its arrays were built (see `_on_worker`),
        # then the worker itself is retired, whether or not the teardown raised
        w = getattr(self, "_worker", None)
        if w is not None and threading.current_thread() is not getattr(self, "_worker_thread", None):
            try:
                fut = w.submit(self.close)
            except RuntimeError:
                # the interpreter is exiting and shut the executor first: the teardown runs here, off the worker
                self._worker = None
                return self.close()
            try:
                fut.result()
            finally:
                w.shutdown(wait=True)
                self._worker = None
                if draft_error is not None:
                    self.draft_engine, self._closed = draft, False
            if draft_error is not None:
                raise draft_error
            return
        if getattr(self, "_pending", None) is not None:
            self._pending[3].join()
            self._pending = None
        holdings = getattr(self, "holdings", None)
        errors = holdings.release_all() if holdings is not None else []
        self._closed = not holdings
        if getattr(self, "dev", None) is not None and self.dev.type == DeviceKind.CUDA:
            torch.cuda.empty_cache()
            if last_on_card(self):
                # what torch keeps for the process: a cuBLAS workspace for every stream it multiplied on (a closed
                # engine's streams are never used again) and the pinned host buffers it caches, each pinned byte
                # the machine's commit on Windows. Let go by the last engine on a card only: another may be mid-GEMM
                torch._C._cuda_clearCublasWorkspaces()
                torch._C._host_emptyCache()
                # and what btb keeps for the card: the GPU counters' query, and the card's kernels - the module
                # out of the driver - once nothing that launches them is left (a graph a failed release kept
                # still holds them)
                vram_pressure_close()
                if self._closed:
                    Native.unload_cuda()
        if draft_error is not None:
            self.draft_engine, self._closed = draft, False
            errors = [draft_error, *errors]
        _leaks_closed(self)
        if errors:
            raise ExceptionGroup(f"[close] {len(errors)} of the engine's holdings would not let go", errors)

    def _stop_workers(self) -> None:
        """the expert store's readers and the MLX tier's pool, drained: nothing of theirs still writes"""
        store = getattr(self, "expert_store", None)
        if store is not None:
            store.pool.shutdown(wait=True)
        pool = getattr(self, "_mlx_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)

    def _save_profile(self) -> None:
        """every reader has finished: the expert profile's record is complete"""
        prof = getattr(self, "expert_profile", None)
        if prof is not None:
            self.expert_profile = None
            prof.save()

    def _mlx_teardown(self) -> None:
        """MLX's arrays let go - the host layers' packed weights, the head's, the cold ring's shared slots - before
        the store's blocks they sit over, then the tier itself; the pool's blocks go back whole, so the next model
        this process loads reads into touched memory"""
        for layer in self.host.values():
            for m in layer.modules():
                if isinstance(m, _HostLinear):
                    m.mx = None
        self.head_host = None
        self.cold_ring.slots, self.cold_ring.shared = [], None
        self.mlx_state.weights = {}
        if self.mlx is not None:
            self.mlx.close()
        for sh in self.mlx_state.pool_blocks:
            _pool.POOL.give(sh)
        self.mlx_state.pool_blocks = set()

    def _close_store(self) -> None:
        """the expert store's blocks let go now, not when its last reference goes: a caller's `with` leaves the
        engine bound, and the store's gigabytes with it"""
        store, self.expert_store = self.expert_store, None
        if store is not None:
            store.close()

    def _drop_weights(self) -> None:
        """the weights on every tier, the drafter's state and the last pass's cache (the caller's to keep, not the
        closed engine's)"""
        for attr in ("templates", "shadow", "resident", "pinned", "_staging", "host", "_spec_slot_bufs"):
            setattr(self, attr, {})
        self.embed_table = self.norm = self.head = self.head_host = None
        self.aj = None
        self._attn_ctx = None
        # the recipe with the slots: its entries hold the cold layers' modules, whose weights are views of the slots
        self.cold_ring.slots, self.cold_ring.shared, self.cold_ring.recipe = [], None, None

    def _close_files(self) -> None:
        """the checkpoint's maps and the drive handles the cold ring read through"""
        for mm, _, _ in self._maps.values():
            if mm is not None:  # the GGUF file's map is the gguf package's own
                with contextlib.suppress(BufferError):
                    mm.close()
        self._maps = {}
        for mm in getattr(self, "_packed_maps", {}).values():
            with contextlib.suppress(BufferError):
                mm.close()
        self._packed_maps = {}
        for fh in self.cold_ring.fh.values():
            with contextlib.suppress(OSError):
                fh.close()
        self.cold_ring.fh = {}
        self.gguf = None

    def new_cache(self, max_len: int | None = None) -> Any:
        from transformers.cache_utils import DynamicCache, DynamicIndexedLayer, DynamicLayer

        cache = DynamicCache(config=self.cfg)
        # the engine's layer keeps every row of a sliding layer (the window lives in the mask), so every layer crops,
        # rolls back and reads alike
        flat = bool(self.fam.flat_cache)
        # a window the model can never fill (Phi-4-mini: 262144 against a 131072 context) is no window; transformers'
        # evicting layer would cost a torch round trip per layer per token (78 -> 68 ms/token)
        win = getattr(self.cfg, "sliding_window", None)
        span = max(int(getattr(self.cfg, "max_position_embeddings", 0) or 0), int(self.context or 0))
        if win and span and int(win) >= span:
            flat = True
        # the ceiling a row of this cache can reach: what the caller reserved for, else the context (or the
        # model's own position limit). A buffer growing past it is a bug the scheduler refuses before it is
        # allocated; 0 where neither is known, and the size check stands alone
        bound = int(max_len) if max_len else span
        sched = getattr(self, "scheduler", None)
        # the megakernel's arena: every layer's K and V as views of one shared array, `cap` rows a head (a layer
        # a session grows past it takes a buffer of its own and the pass leaves the megakernel)
        arena = None
        mg = getattr(self, "_mega", None)
        # the arena and its megakernel are bf16-only (`per` counts two bytes a row, and `_mega_ok` refuses any
        # other compute dtype); an fp32 pass would store float32 into these bf16-sized slots, so it keeps its own
        # per-layer buffers instead
        bf16_compute = self.compute_dtype in (None, torch.bfloat16)
        if mg is not None and bf16_compute and self.mlx is not None and not self.kv_bits and not flat:
            from .. import mlx as mlxdev
            from ..mlx.attn import ATTN_BLOCK

            cap = min(int(max_len) if max_len else (span or 4096), mg.SPL * ATTN_BLOCK)
            per = mg.Hk * cap * mg.hd * 2
            sh = mlxdev.Shared(2 * self.L * per)
            setattr(cache, "_mega_arena", sh)  # noqa: B010  not a DynamicCache slot: the arena pinned to the cache's lifetime
            arena = [(sh.mx, (2 * i) * per, (2 * i + 1) * per, cap) for i in range(self.L)]
        # a bf16 card engine keeps every layer's rows in bf16, a host layer's too (`host_kv_dtype`)
        kv_dtype = torch.bfloat16 if self.host_kv_dtype() == torch.bfloat16 else None
        for i, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer or (flat and isinstance(layer, DynamicLayer)):
                cache.layers[i] = GrowLayer(
                    shared=self.mlx is not None,
                    bits=self.kv_bits if self.mlx is not None else None,
                    cap_hint=int(max_len or 0),
                    grant=None if sched is None else sched.grant,
                    bound=bound,
                    arena=arena[i] if arena is not None else None,
                    kv_dtype=kv_dtype,
                )
            elif type(layer) is DynamicIndexedLayer and sched is not None:
                # a sparse-attention layer's rows, grown as transformers grows them, each growth asked of the ledger
                # (the epoch's KV it draws on) before it is made
                cache.layers[i] = GrantedIndexedLayer(sched.grant)
        if self.dev.type == DeviceKind.CUDA:
            self._card_adopt_cache(cache)
        return self._track(cache)

    def reset_stats(self) -> None:
        self.bytes_streamed = 0
        self.load_s = 0.0
        self.compute_s = 0.0
        self.wait_s = 0.0
