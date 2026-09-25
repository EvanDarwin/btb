# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`StreamedTextModel`: the engine. Its behaviour is split by concern over the mixins of this package; this module
holds its construction and lifetime."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .. import mlx as mlxdev
from .. import pool as _pool
from ..gguf import GGUFModel
from ..hf import is_gguf, pack_format, shard_map
from ..kinds import LayerKind, Log, Tokens
from ..mlx.q6k import gather_q6k
from ..options import Device as DeviceKind
from ..options import DeviceName
from ..pack12 import parent_dir
from ..sysinfo import host_commit_bytes, host_free_bytes
from .cache import GrowLayer
from .cuda import _CudaMixin
from .device import Device
from .drafter import MTPDrafter
from .experts import _ExpertStore
from .families import _FamiliesMixin, family, register_attention
from .forward import _ForwardMixin
from .fused import fast_causal_conv1d
from .generate import _GenerateMixin
from .host import _Experts, _HostLinear, _NGramRows, _Router
from .memory import RamPolicyState, VramPolicyState, _MemoryMixin
from .mlx_forward import MlxState, _MlxMixin
from .native import Native
from .scheduler import BatchScheduler
from .text import _TextMixin
from .tiers import ColdRing, _TiersMixin


class StreamedTextModel(
    _FamiliesMixin, _TiersMixin, _MemoryMixin, _MlxMixin, _CudaMixin, _ForwardMixin, _GenerateMixin, _TextMixin
):
    """The engine over one model. Construction and lifetime live here; the forward, the tiers, memory, the
    MLX and CUDA paths and decoding are the mixins (one module each in this package)."""

    ST_DTYPES = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }

    # the tests reach the helpers as `StreamedTextModel._HostLinear`; the native handles are mirrored from
    # `Native` by `load_gemv`
    _HostLinear = _HostLinear
    _Router = _Router
    _Experts = _Experts
    _NGramRows = _NGramRows
    _ExpertStore = _ExpertStore
    MTPDrafter = MTPDrafter
    gemv = gemv_p12 = gemv_group = gemv_mx4 = gemv_mx4_group = attn_decode = delta_step = read_direct = None
    read_open = read_at = read_close = None
    # `Native.open` and `Native.close` are the reader's file handle; `close` here is the engine's own teardown,
    # so those two are mirrored under the names above
    _HANDLE_NAMES = {"open": "read_open", "close": "read_close"}
    mlx: Any = None
    gemm_rows = Native.gemm_rows
    cpu_gemm_rows = Native.cpu_gemm_rows

    @classmethod
    def load_gemv(cls, dll_path: str | os.PathLike[str], threads: int = 0) -> Callable[..., Any]:
        """bind the native library's kernels (once per process); see `Native.load_gemv`"""
        gemv = Native.load_gemv(dll_path, threads)
        for name in Native.HANDLES:
            setattr(cls, cls._HANDLE_NAMES.get(name, name), getattr(Native, name))
        return gemv

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
        vram_watch: bool = True,
        mlx_layers: Iterable[int] | None = None,
        kv_bits: int | None = None,
        gguf_packed: bool = True,
        host_budget: Any = None,
        bus_pass: bool = True,
        store_pin: int = 0,
    ) -> None:
        from transformers import AutoConfig

        t0 = time.time()
        self.gguf = GGUFModel(model_dir) if is_gguf(model_dir) else None
        self.gguf_packed = bool(gguf_packed)  # its Q4/Q8 tensors on the packed kernels as stored (MLX), else bf16
        self.dir = self.gguf.dir if self.gguf is not None else model_dir
        self.mlx = None
        self.mlx_state = MlxState()
        # the RAM this load started with (pool.py samples it before anything is allocated for the load; a
        # model built directly samples now): what the engine may use, against which it counts what it holds
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
            self.mlx_layers = (
                {int(x) for x in mlx_layers}
                if mlx_layers is not None
                else {int(x) for x in cpu_layers} | {int(x) for x in cold_layers}
            )
            devname = DeviceName(DeviceKind.CPU)  # MLX's tensors on the torch side are host tensors
        self.dev = torch.device(str(devname))
        if self.dev.type == DeviceKind.CUDA and self.dev.index is not None:
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
        # Gemma scales the input embedding by sqrt(hidden); the engine gathers rows itself, so it applies the
        # scale the module's scaled embedding would (the tied head's output projection stays unscaled)
        self.embed_scale = float(cfg.hidden_size) ** 0.5 if self.fam.embed_scale else None
        if self.fam.eager:
            # gpt-oss's sinks are not expressible through sdpa: `attention_sinks` runs the reference's arithmetic on
            # the CPU, the engine's kernels over an MLX cache
            register_attention()
            cfg._attn_implementation = "btb_sinks"
        self.n_experts = int(getattr(cfg, "num_local_experts", 0) or getattr(cfg, "num_experts", 0) or 0)
        if self.mlx is not None and self.fam.hybrid:
            self.fam.mod.causal_conv1d_fn = fast_causal_conv1d
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
        # the embedding carries the checkpoint's own weight precision: an fp16 or fp32 one is held as bf16
        self.held_cast = False
        if self.gguf is None:
            _mm, hdr, _ = self._shard(self.weight_map[emb_keys[0]])
            self.held_cast = self.ST_DTYPES[hdr[emb_keys[0]]["dtype"]] != torch.bfloat16
        # set by close(): every decode loop ends at its next step, so no thread is mid-pass when the buffers go
        self.abort = threading.Event()
        self._decode_lock = threading.RLock()  # one decode at a time on the engine (MLX's worker serializes too)
        self._meta = torch.device("meta")
        self.expert_stat = {"experts": 0, "bytes": 0, "calls": 0, "s": 0.0}
        self.expert_trace = None
        self.expert_profile = None
        self.expert_store = None
        self.norm = self.mixer = None
        with self._meta:
            if self.fam.norm is not None:
                self.norm = self.fam.norm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            else:
                self.mixer = self.fam.mod.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        # the parameters are widened to a float32 compute dtype below: read at the checkpoint's own precision for it
        wide = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        if self.norm is not None:
            self._adopt(self.norm, "weight", self._get(self.prefix + "norm.weight", stored=wide))
        else:
            for name, _, is_buf in self._named_tensors(self.mixer):
                t = self._get(self.prefix + "hyper_connection_mixer." + name, stored=wide and not is_buf)
                self._adopt(self.mixer, name, t, buffer=is_buf)
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
                self.head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
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
        self._streamed_any = any(i not in _res_set and i not in _host_set for i in range(self.L))
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
        for i in sorted({int(x) for x in resident_layers}):
            tmpl = self._new_layer(i)
            self._load_layer(i, tmpl, first=True)
            self.resident[i] = tmpl
        self.cold = {int(x) for x in cold_layers} - set(self.resident)
        self.cold_slots = int(cold_slots)
        self._mega = None
        self._mlx_attn_slope = None
        self.cold_chunk = int(cold_chunk_mb) << 20
        self.cold_ring = ColdRing()
        self.host = {}
        for i in sorted({int(x) for x in cpu_layers} | self.cold):
            if i in self.resident:
                continue
            self.host[i] = self._make_host_layer(i)
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
            self.vram_margin = (
                int(float(vram_reserve_gb) * 2**30)
                if vram_reserve_gb is not None
                else max(1 << 29, total_vram * 8 // 100)
            )
        else:
            self.vram_margin = 0
        self.scheduler = BatchScheduler(self)
        self.plan = None
        self.device = Device(self)
        # the expert store reads both at construction, so they are set here and never afterwards: the Bus Pass by
        # default (1-8% on the token over two pairs on NVMe, 11% fewer misses on the replay's warm passes,
        # bookkeeping its only cost), and the store's pages pageable unless `store_pin` asks for pinned ones
        self.bus_pass = bool(bus_pass)
        self.store_pin = int(store_pin)
        if self.fam.moe and Native.read_direct is not None and expert_cache_gb != 0:
            if expert_cache_gb is None:
                # the MLX tier's ledger: the RAM the load started with, less the reserve, less what MLX holds; the
                # OS's free count reads high for untouched Metal buffers
                if self.mlx is not None:
                    # the base re-read with the trunk and the pool bound and touched: the figure sampled at import was
                    # taken while the previous process was still releasing memory, and capped the store 5-10 GB low
                    self.mem_start = max(self.mem_start, host_free_bytes() + self.mlx.held_bytes())
                    room = self.mem_start - self.ram_reserve - self.mlx.held_bytes()
                else:
                    room = host_free_bytes() - self.ram_reserve
                if sys.platform == "win32":
                    # the commit charge: the page file bounds what can be allocated at all, above the same floor
                    room = min(room, host_commit_bytes() - self.ram_reserve)
                budget = max(0, room)  # the store keeps its scratch slots whatever the room
            else:
                budget = int(float(expert_cache_gb) * 2**30)
            self.expert_store = _ExpertStore(self, budget, self.ram_reserve)
        self.vram_watch = bool(vram_watch) and self.dev.type == DeviceKind.CUDA
        self.vram_state = VramPolicyState()
        # the host tier's policy: on where layers run from RAM off unified memory (which keeps its own ledger)
        self.ram_watch = bool(self.host) and self.mlx is None
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
        res = torch.cuda.memory_allocated(self.dev) / 2**30 if self.dev.type == DeviceKind.CUDA else 0.0
        n_templates = sum(len(v) for v in self.templates.values())
        self.log(
            f"[stream] {model_dir}: {self.L} layers ({n_templates} templates, prefetch "
            f"{'on' if self.prefetch else 'off'}), prefix '{self.prefix}', head "
            f"{'resident' if resident_head else 'streamed'}; resident {res:.2f} GB; init {time.time() - t0:.1f}s"
        )
        if self.mlx is not None:
            n_cpu = len([i for i in self.host if i not in self.mlx_layers])
            n_cold = len([i for i in self.cold if i in self.mlx_layers])
            self.log(
                f"[mlx] {len(self.mlx_layers) - n_cold} layers and the head in unified memory ({self.mlx_state.bytes / 2**30:.2f} GB"
                f"), {n_cold} streamed from the drive each pass, {n_cpu} on the CPU kernels; "
                f"MLX active {self.mlx.active_bytes() / 2**30:.2f} GB"
            )

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
            return int(self.card_warm(ids))
        if self.mlx is not None:
            n = int(self.mlx_warm(ids))
            if n:
                # a pass's attention grows with the rows a node reads: its cost per node and row, timed on
                # the node kernel over synthetic caches, is what the budget adds for the session's length
                self.mlx_attn_cost()
            return n
        if self.dev.type == DeviceKind.CPU:
            return int(self.host_warm(ids))
        return 0

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
        """every tensor the family's layers, embedding, norm and head are loaded from, by its HF name: what a
        GGUF file's tensors are matched against"""
        with torch.device("meta"):
            layer = self.fam.layer(cfg, 0)
        per = [n for n, _ in layer.named_parameters()] + [n for n, _ in layer.named_buffers()]
        names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
        return names + [f"model.layers.{i}.{n}" for i in range(self.L) for n in per]

    def close(self) -> None:
        """Stop any decode in flight and release the model's memory on every tier; safe to call twice. The engine
        is unusable afterwards. `with btb.load(...) as model:` calls it on exit."""
        if getattr(self, "_closed", False):
            return
        draft = getattr(self, "draft_engine", None)
        if draft is not None:
            self.draft_engine = None
            draft.close()
        self.abort.set()
        # the MLX tier's teardown runs on the model's worker thread, where its arrays were built (see `on_worker`),
        # then the worker itself is retired
        w = getattr(self, "_worker", None)
        if w is not None and threading.current_thread() is not getattr(self, "_worker_thread", None):
            try:
                fut = w.submit(self.close)
            except RuntimeError:
                # the interpreter is exiting and shut the executor first: the teardown runs here, off the worker
                self._worker = None
                return self.close()
            fut.result()
            w.shutdown(wait=True)
            self._worker = None
            return
        self._closed = True
        draft = getattr(self, "draft_engine", None)
        if draft is not None:
            self.draft_engine = None  # the draft model is its own engine (a --draft-model load): release it too
            draft.close()
        if getattr(self, "_pending", None) is not None:
            self._pending[3].join()
            self._pending = None
        # no thread may still be writing into a buffer this is about to free: the cold reader is told to stop
        # and joined, the expert store's and the loader's pools are drained
        self._cold_stop()
        store = getattr(self, "expert_store", None)
        if store is not None:
            store.pool.shutdown(wait=True)
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            sched.disk_close()
        prof = getattr(self, "expert_profile", None)
        if prof is not None:
            # every reader has finished: the record is complete
            self.expert_profile = None
            prof.save()
        pool = getattr(self, "_mlx_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        if self.mlx is not None:
            for layer in self.host.values():
                for m in layer.modules():
                    if isinstance(m, _HostLinear):
                        m.mx = None
            self.head_host = None
            self.expert_store = None
            self.cold_ring.slots, self.cold_ring.shared = [], None
            self.mlx_state.weights = {}
            self.mlx.close()
        # the pool's blocks go back whole: the next model this process loads reads into touched memory
        for sh in self.mlx_state.pool_blocks:
            _pool.POOL.give(sh)
        self.mlx_state.pool_blocks = set()
        for attr in ("templates", "shadow", "resident", "pinned", "_staging", "host", "_spec_slot_bufs"):
            setattr(self, attr, {})
        self.embed_table = self.norm = self.head = None
        self.aj = None
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
        self.cold_ring.slots, self.cold_ring.shared = [], None
        self.expert_store = None
        self.gguf = None
        if self.dev.type == DeviceKind.CUDA:
            torch.cuda.empty_cache()

    def new_cache(self, max_len: int | None = None) -> Any:
        from transformers.cache_utils import DynamicCache, DynamicLayer

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
        for i, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer or (flat and isinstance(layer, DynamicLayer)):
                cache.layers[i] = GrowLayer(
                    reserve=self.context or 0,
                    shared=self.mlx is not None,
                    bits=self.kv_bits if self.mlx is not None else None,
                    cap_hint=int(max_len or 0),
                    grant=None if sched is None else sched.grant,
                    bound=bound,
                    arena=arena[i] if arena is not None else None,
                )
        if self.dev.type == DeviceKind.CUDA:
            self._card_adopt_cache(cache)
        return cache

    def reset_stats(self) -> None:
        self.bytes_streamed = 0
        self.load_s = 0.0
        self.compute_s = 0.0
        self.wait_s = 0.0
