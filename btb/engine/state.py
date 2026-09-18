# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's declared state: the attributes every mixin reads and stubs of the methods they call on each
other, so each part type-checks alone. Declarations only; the real methods come earlier in the MRO. A knob the
engine reads as `getattr(self, name, default)` is declared without a value: unset until a caller sets it."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch

# rows of the lm_head the drafter proposes from
DRAFT_VOCAB = 32768

if TYPE_CHECKING:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import ModuleType

    from ..draft import Spans
    from ..gguf import GGUFModel
    from ..kinds import Json, Log, NodePath, Parents, TokenRows, Tokens
    from ..mlx import Backend, Shared
    from ..mlx.mega import MegaPass
    from ..sampling import Sampling
    from ..session import Session
    from .device import Device
    from .drafter import MTPDrafter
    from .experts import ExpertProfile, _ExpertStore
    from .families import Family
    from .host import _HostLinear
    from .memory import RamPolicyState, VramPolicyState
    from .mlx_forward import MlxState
    from .model import StreamedTextModel
    from .scheduler import BatchScheduler, Plan
    from .tiers import ColdRing


class _State:
    """see the module docstring"""

    # -- what the checkpoint, its config and the device fix --
    L: int
    ST_DTYPES: dict[str, torch.dtype]
    cfg: Any
    compute_dtype: torch.dtype | None
    context: int | None
    dev: torch.device
    device: Device
    dir: str
    fam: Family
    gguf: GGUFModel | None
    gguf_packed: bool  # a GGUF's quantized tensors on the packed kernels as stored (else dequantized to bf16)
    _gguf_names: dict[str, str]
    _gguf_hdr: dict[str, Any]
    head_key: str
    kv_bits: int | None
    kv_block: int
    kv_host: bool
    layer_types: list[str]
    log: Log
    n_experts: int
    prefill_chunk: int | None
    prefix: str
    weight_map: dict[str, str]
    _meta: torch.device

    # -- where the model's parts live, and the tiers' working state --
    cold: set[int]
    drafter_dev: torch.device | None
    embed_table: torch.Tensor | None
    embed_scale: float | None
    head: torch.nn.Linear | None
    head_host: _HostLinear | None
    host: dict[int, Any]
    mixer: torch.nn.Module | None
    mlx_layers: set[int]
    norm: Any
    pinned: dict[int, dict[str, torch.Tensor]]
    prefetch: bool
    prefill_card: bool
    prefill_card_min: int
    resident: dict[int, Any]
    resident_fp32: bool
    resident_head: bool
    rotary: Any
    shadow: dict[str, Any]
    templates: dict[str, Any]
    _attn_ctx: Any
    _batched_cont: bool
    _worker: ThreadPoolExecutor | None
    _worker_thread: threading.Thread
    _copy_stream: torch.cuda.Stream
    _dma_done: dict[str, Any]
    _events: list[tuple[torch.cuda.Event, torch.cuda.Event]]
    _host_cost: dict[int, float]
    _kv_stage: list[Any] | None
    _maps: dict[Any, Any]
    _packed: Any
    _packed_maps: dict[Any, Any]
    _pending: Any
    _pstaging: dict[Any, Any]
    _staging: dict[Any, Any]
    _streamed_any: bool
    _sweep_keep: bool
    _thread_mod: ModuleType
    _toggle: dict[str, int]

    # -- the cold ring: the layers read from the drive each pass --
    cold_chunk: int
    cold_ring: ColdRing
    cold_slots: int

    # -- the memory policy: reserves, margins, and what has been shed --
    mem_start: int
    ram_reserve: int
    ram_state: RamPolicyState
    ram_watch: bool
    vram_margin: int
    vram_state: VramPolicyState
    vram_watch: bool
    _card_ms_min: float | None
    _last_card_ms: float | None
    _shed: list[str]

    # -- the scheduler, and the run's counters --
    abort: threading.Event
    _decode_lock: threading.RLock
    bytes_streamed: int
    compute_s: float
    load_s: float
    plan: Plan | None
    scheduler: BatchScheduler
    wait_s: float
    _closed: bool
    _in_epoch: bool

    # -- the card graph (cuda.py); the mechanism keeps its letters --
    card_pipeline: bool
    fast_decode: bool
    aj: MTPDrafter | None
    al: dict[int, Any]
    am: dict[int, Any]
    an: dict[int, Any]
    ap: Parents | None
    aq: bool
    ay: dict[int, tuple[torch.Tensor, torch.Tensor]]
    _fmlp: bool
    _frope: bool
    _g: dict[str, Any]

    # -- MLX --
    mlx: Backend | None
    mlx_attn_kernel: bool
    mlx_attn_prefill: bool
    mlx_attn_rows: int
    mlx_state: MlxState
    _mega: MegaPass | None
    _mlx_attn_slope: list[tuple[int, float]] | None
    _mlx_cost: dict[int, float]
    _mlx_pool: ThreadPoolExecutor

    # -- the text layer --
    _tokenizer: Any

    # -- speculation: the tree, the drafter, the n-gram proposer --
    draft_bits: int
    draft_engine: StreamedTextModel | None
    draft_temp_ratio: float  # under sampling the drafter draws its children at this multiple of the temperature
    draft_ks: tuple[int, ...]
    draft_vocab: int
    drafter_weights: str | None
    eos_ids: tuple[int, ...]
    ngram_p: float
    ngram_tree: bool
    proposer: str
    sampling: Sampling  # the engine's default: greedy unless loaded with temperature/top_p/top_k/seed
    tree_budget: int
    tree_min_prob: float
    tree_read: str
    tree_step_mass: float
    v_max: int

    # -- the expert store --
    bus_pass: bool
    expert_profile: ExpertProfile | None
    expert_stat: dict[str, Any]
    expert_store: _ExpertStore | None
    expert_trace: list[tuple[int, torch.Tensor]] | None
    lookahead: tuple[int, ...]
    lookahead_rows: tuple[int, ...]
    store_pin: int
    vram_experts_gb: str | float

    # -- tiers.py --
    def _adopt(self, module: Any, dotted: str, t: torch.Tensor, writable: bool = False, buffer: bool = False) -> None:
        raise NotImplementedError

    def _bind_cold(self) -> None:
        raise NotImplementedError

    def _bind_host_packed_layer(self, layer: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def _cache_to(cache: Any, i: int, dev: str | torch.device) -> None:
        raise NotImplementedError

    def _cold_release(self, i: int) -> None:
        raise NotImplementedError

    def _cold_start(self, n_layers: int) -> None:
        raise NotImplementedError

    def _cold_stop(self) -> None:
        raise NotImplementedError

    def _layer_bytes_stored(self, i: int, packed: bool = False) -> int:
        raise NotImplementedError

    def _rebind_warm(self, i: int) -> None:
        raise NotImplementedError

    def _cold_wait(self, i: int) -> None:
        raise NotImplementedError

    def _flush_events(self) -> None:
        raise NotImplementedError

    def _get(self, key: str) -> torch.Tensor:
        raise NotImplementedError

    def _head_host(self) -> Any:
        raise NotImplementedError

    def _layer_items(self, lins: Sequence[Any]) -> tuple[list[tuple[Any, str | None, int, int, int]], int]:
        raise NotImplementedError

    def _load_layer(self, i: int, tmpl: Any, first: bool = False) -> None:
        raise NotImplementedError

    def _new_layer(self, idx: int) -> Any:
        raise NotImplementedError

    def _next_template(self, lt: str) -> Any:
        raise NotImplementedError

    def _realloc_bytes(self) -> int:
        raise NotImplementedError

    def _regrow_bytes(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _set_param(module: Any, dotted: str, t: torch.Tensor, buffer: bool = False) -> None:
        raise NotImplementedError

    def _shard(self, shard: str) -> tuple[Any, dict[str, Any], Any]:
        raise NotImplementedError

    def _start_prefetch(self, i: int, tmpl: Any) -> None:
        raise NotImplementedError

    def _sync(self) -> None:
        raise NotImplementedError

    def _upcast(self, lt: str, tmpl: Any, i: int) -> Any:
        raise NotImplementedError

    def _wait_prefetch(self, i: int) -> Any:
        raise NotImplementedError

    # -- cuda.py --
    def _card_generate_greedy(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: set[int],
        on_token: Callable[[int], Any] | None,
        t0: float,
        sampling: Any = None,
    ) -> list[int]:
        raise NotImplementedError

    def _card_greedy_ok(
        self, cache: Any, B: int, am: Any, on_layer: Any, prefill_only: bool, session: Any
    ) -> bool:  # an engine built without the card mixin has no step graph
        return False

    def _card_pass_ok(
        self, cache: Any, B: int, T: int, past: int, am: torch.Tensor | None, on_layer: Any, stop_after: int | None
    ) -> bool:
        raise NotImplementedError

    def _card_segment_at(self, i: int, n_layers: int) -> tuple[int, int] | None:
        raise NotImplementedError

    def _delta_nodes(
        self,
        layer: Any,
        i: int,
        cl: Any,
        mixed_all: torch.Tensor,
        z_all: torch.Tensor,
        a_all: torch.Tensor,
        b_all: torch.Tensor,
        T: int,
        parents: Parents,
        spec_on: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _fast_ok(
        self, cache: Any, B: int, T: int, past: int, am: torch.Tensor | None, on_layer: Any, stop_after: int | None
    ) -> bool:
        raise NotImplementedError

    def _forward_card_segment(
        self, a: int, b: int, h: torch.Tensor, pas: Any, tail: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        raise NotImplementedError

    def _forward_fast(self, h: torch.Tensor, pe: Any, cache: Any, last_only: bool, head: bool) -> torch.Tensor:
        raise NotImplementedError

    def _spec_budget(self, ema_tokens: float, passes: int, v_max: int | None = None, past: int = 0) -> int:
        raise NotImplementedError

    def _spec_full(self, v_max: int | None = None) -> int:
        raise NotImplementedError

    def aa(self, parents: Parents | None = None) -> None:
        raise NotImplementedError

    def ab(self) -> None:
        raise NotImplementedError

    def ac(self, tmpl: Any, i: int, h: torch.Tensor, pe: Any, text_pos: Any, cache: Any) -> torch.Tensor:
        raise NotImplementedError

    def ad(self, cache: Any, base_len: int, path: NodePath) -> None:
        raise NotImplementedError

    def ai(self, layer: Any, i: int, h: torch.Tensor, pe: Any, text_pos: Any, cache: Any) -> torch.Tensor:
        raise NotImplementedError

    # -- mlx_forward.py --
    def _bind_mlx_resident(self, layer: Any, checkpoint: bool = True) -> Any:
        raise NotImplementedError

    def _forward_mega(self, ids: Any, cache: Any, T: int, pick: Any = None) -> torch.Tensor:
        raise NotImplementedError

    def _forward_mlx(
        self,
        h: Any,
        pe: Any,
        cache: Any,
        on_layer: Any,
        last_only: bool,
        head: bool,
        n_layers: int,
        hm: Any = None,
        lazy: bool = False,
        rows: Sequence[int] | None = None,
        forest: dict[str, Any] | None = None,
        pick: Any = None,
    ) -> Any:
        raise NotImplementedError

    def _generate_greedy_mlx(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: Any,
        on_token: Callable[[int], Any] | None,
        t0: float,
        session: Session | None = None,
        sampling: Any = None,
    ) -> tuple[list[int], Json]:
        raise NotImplementedError

    def _generate_greedy_mlx_batch(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: Any,
        attention_mask: torch.Tensor | None,
        t0: float,
        sampling: Any = None,
    ) -> list[list[int]]:
        raise NotImplementedError

    def _mega_ok(self, cache: Any, T: int, on_layer: Any, head: bool, pick: Any, positions: Any) -> bool:
        raise NotImplementedError

    def _mlx_batch_ok(self, B: int, on_layer: Any, prefill_only: bool) -> bool:
        raise NotImplementedError

    def _mlx_embed(self) -> Any:
        raise NotImplementedError

    def _mlx_embed_rows(self, tok: Any) -> Any:
        raise NotImplementedError

    def _mlx_flush_states(self, cache: Any) -> None:
        raise NotImplementedError

    def _mlx_fuse(self, layer: Any) -> None:
        raise NotImplementedError

    def _mlx_greedy_ok(self, B: int, attention_mask: torch.Tensor | None, on_layer: Any, prefill_only: bool) -> bool:
        raise NotImplementedError

    def _mlx_ok(
        self, cache: Any, B: int, T: int, am: torch.Tensor | None, positions: torch.Tensor | None, n_layers: int
    ) -> bool:
        raise NotImplementedError

    def _mlx_tree_able(self, cache: Any) -> bool:
        raise NotImplementedError

    def _shared_ahead(self, nbytes: int) -> tuple[Shared, int]:
        raise NotImplementedError

    # -- forward.py --
    def _finish(self, h: torch.Tensor, last_only: bool, head: bool) -> torch.Tensor:
        raise NotImplementedError

    def _prefill(
        self, ids: torch.Tensor, cache: Any, on_layer: Any = None, attention_mask: torch.Tensor | None = None
    ) -> Any:
        raise NotImplementedError

    @torch.inference_mode()
    def forward(
        self,
        ids: Any,
        cache: Any = None,
        on_layer: Callable[[int, torch.Tensor], Any] | None = None,
        last_only: bool = True,
        stop_after: int | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: Any = None,
        head: bool = True,
        pick: Any = None,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def pad_left(rows: TokenRows, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    # -- generate.py --
    @staticmethod
    def _lin(cl: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def generate_greedy(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        on_token: Callable[[int], Any] | None = None,
        attention_mask: torch.Tensor | None = None,
        on_layer: Any = None,
        prefill_only: bool = False,
        session: Any = None,
        sampling: Any = None,
    ) -> Any:
        raise NotImplementedError

    def generate_speculative(
        self,
        ids: torch.Tensor | Tokens | TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        v_max: int = 8,
        n_min: int = 2,
        n_max: int = 4,
        on_token: Callable[[int], Any] | None = None,
        proposer: str = "ngram",
        spans: Spans = (),
        session: Session | None = None,
        sampling: Any = None,
    ) -> tuple[list[int], Json]:
        raise NotImplementedError

    def serve(
        self,
        prompts: TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        pad_id: int | None = None,
        sampling: Any = None,
    ) -> list[list[int]]:
        raise NotImplementedError

    @staticmethod
    def _lin_set(cl: Any, conv: torch.Tensor, rec: torch.Tensor) -> None:
        raise NotImplementedError

    def _session_prefill(
        self, ids: torch.Tensor, cache: Any, reuse: int, session: Session | None, on_layer: Any = None
    ) -> tuple[Any, Any]:
        raise NotImplementedError

    # -- families.py --
    @staticmethod
    def _dense_key(key: str) -> bool:
        raise NotImplementedError

    def _make_host_layer(self, i: int) -> Any:
        raise NotImplementedError

    @staticmethod
    def _named_tensors(module: Any) -> Iterable[tuple[str, torch.Tensor, bool]]:
        raise NotImplementedError

    def _shape_layer(self, layer: Any, i: int) -> Any:
        raise NotImplementedError

    # -- model.py --
    def close(self) -> None:
        raise NotImplementedError

    def embed(self, ids: Any) -> torch.Tensor:
        raise NotImplementedError

    def new_cache(self, max_len: int | None = None) -> Any:
        raise NotImplementedError

    # -- memory.py --
    def ram_policy(self, log: Log | None = None) -> None:
        raise NotImplementedError

    def vram_policy(self, cache: Any = None, log: Log | None = None) -> None:
        raise NotImplementedError

    def vram_trim(self, tag: str = "") -> Any:
        raise NotImplementedError
