# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The engine's declared state: the attributes every mixin reads and stubs of the methods they call on each
other, so each part type-checks alone. Declarations only; the real methods come earlier in the MRO. A knob the
engine reads as `getattr(self, name, default)` is declared without a value: unset until a caller sets it."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import torch

from ..kinds import LayerTier, PassReport, PassTag, Proposer

# rows of the lm_head the drafter proposes from
DRAFT_VOCAB = 32768

# a call handed to the decode's thread (`_serial`): its parameters and its result
P = ParamSpec("P")
R = TypeVar("R")

# LayerTier -> the PassTag a pass records for a layer on that tier (btb/engine/device.py Placement.tier)
_TIER_TAG: dict[LayerTier, PassTag] = {
    LayerTier.RESIDENT: PassTag.TIER_RESIDENT,
    LayerTier.HOST: PassTag.TIER_HOST,
    LayerTier.COLD: PassTag.TIER_COLD,
    LayerTier.STREAMED: PassTag.TIER_STREAMED,
}


@dataclass
class _PassRecorder:
    """The mutable accumulator behind `StreamedTextModel.last_pass_report()`: the tags a generate's forks record
    and its running speculation counts. Reset once a `generate()`; frozen into a `PassReport` on read."""

    tags: set[PassTag] = field(default_factory=set)
    spec_proposed: int = 0
    spec_accepted: int = 0

    def report(self) -> PassReport:
        return PassReport(frozenset(self.tags), self.spec_proposed, self.spec_accepted)


if TYPE_CHECKING:
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import ModuleType

    from ..draft import Spans
    from ..fp8 import F8Weight
    from ..gguf import GGUFModel
    from ..kinds import Json, Log, NodePath, Parents, TokenRows, Tokens
    from ..mlx import Backend, Shared
    from ..mlx.mega import MegaPass
    from ..sampling import Sampling
    from ..session import Session
    from .cache import KvCache
    from .device import Device, DeviceSpec
    from .drafter import MTPDrafter
    from .experts import ExpertProfile, _ExpertStore
    from .families import Family
    from .generate import LinLayer, LinSnap
    from .hooks import Hooks
    from .host import _HostLinear
    from .memory import RamPolicyState, Room, VramPolicyState
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
    held_cast: bool  # the checkpoint stores its weights at a float precision other than bf16 (tiers._held)
    fp8_experts: bool  # the checkpoint's fused experts are FP8 (btb/fp8.py), multiplied as stored
    fp8_layers: set[int]  # the host layers whose FP8 linears are multiplied as stored (`_HostLinear.f8`)
    fp8_widened: bool  # some FP8 tensor was widened to bf16 for a path that reads it so (MLX slots, a card, cold)
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
    adapt: bool  # the memory policies give way to other programs (`--adapt`); off, the placement is pinned
    ram_watch: bool
    vram_margin: int
    vram_state: VramPolicyState
    vram_watch: bool
    _card_ms_min: float | None
    _last_card_ms: float | None
    _shed: list[str]
    _live_caches: weakref.WeakSet[KvCache]  # every cache a layer's move reaches (`_track`)

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
    _pass_rec: _PassRecorder | None = None  # the provenance accumulator, created on first tag or reset
    _calls: frozenset[PassTag] = frozenset()  # every API call made on the model (btb.api), never reset

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
    proposer: Proposer
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

    def _track(self, cache: KvCache) -> KvCache:
        raise NotImplementedError

    def _caches_to(self, i: int, dev: str | torch.device, cache: KvCache | None = None) -> None:
        raise NotImplementedError

    def _cold_release(self, i: int) -> None:
        raise NotImplementedError

    def _cold_start(self, n_layers: int) -> None:
        raise NotImplementedError

    def _cold_stop(self) -> None:
        raise NotImplementedError

    def _layer_bytes(self, i: int) -> int:
        raise NotImplementedError

    def _layer_bytes_stored(self, i: int, packed: bool = False) -> int:
        raise NotImplementedError

    def _rebind_warm(self, i: int) -> None:
        raise NotImplementedError

    def _cold_wait(self, i: int) -> None:
        raise NotImplementedError

    def _flush_events(self) -> None:
        raise NotImplementedError

    def _get(self, key: str, gguf_shortcut: bool = False, stored: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def _held(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _cast_on_read(self, info: dict[str, Any]) -> bool:
        raise NotImplementedError

    def _fp8(self, key: str) -> bool:
        raise NotImplementedError

    def _f8_weights(self, key: str) -> list[F8Weight]:
        raise NotImplementedError

    def _shape(self, key: str) -> tuple[int, ...]:
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

    def _final_norm(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _apply_head(self, hf: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prefill(
        self,
        ids: torch.Tensor,
        cache: Any,
        on_layer: Any = None,
        attention_mask: torch.Tensor | None = None,
        last_only: bool = True,
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

    @staticmethod
    def _lin_snap(cl: LinLayer) -> LinSnap:
        raise NotImplementedError

    @staticmethod
    def _lin_restore(cl: LinLayer, snap: LinSnap) -> None:
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
        hooks: Hooks | None = None,
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
        proposer: Proposer | str = Proposer.NGRAM,
        spans: Spans = (),
        session: Session | None = None,
        sampling: Any = None,
        hooks: Hooks | None = None,
    ) -> tuple[list[int], Json]:
        raise NotImplementedError

    def serve(
        self,
        prompts: TokenRows,
        max_new: int,
        eos_ids: Tokens = (),
        pad_id: int | None = None,
        sampling: Any = None,
        hooks: Hooks | None = None,
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
    def lend_policy(self) -> None:
        raise NotImplementedError

    def cache_room(self, cache: KvCache | None, B: int, T: int) -> None:
        raise NotImplementedError

    def room(self, nbytes: int, device: DeviceSpec | None = None, name: str = "room") -> Room:
        raise NotImplementedError

    def _serial(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        raise NotImplementedError

    def ram_policy(self, log: Log | None = None) -> None:
        raise NotImplementedError

    def vram_policy(self, cache: Any = None, log: Log | None = None) -> None:
        raise NotImplementedError

    def vram_trim(self, tag: str = "") -> Any:
        raise NotImplementedError

    # -- pass provenance: the tags a generate's forks record, read as a PassReport. Recording only - a fork
    # calls `_tag`, it never changes which path runs. `last_pass_report()` is the cert's window on the path a
    # pass actually took, so a silent fallback is caught rather than certified. --
    def _pass_reset(self) -> None:
        """start a fresh report; called once a `generate()` so its passes' tags accumulate into one report"""
        self._pass_rec = _PassRecorder()

    def _tag(self, *tags: PassTag) -> None:
        """record the forks a pass took; the recorder is created lazily so a bare `forward()` reports too"""
        rec = self._pass_rec
        if rec is None:
            rec = self._pass_rec = _PassRecorder()
        rec.tags.update(tags)

    def _tag_tiers(self, n_layers: int) -> None:
        """record the placement tiers the pass's layers live on, from the same `Placement.tier` every path reads"""
        place = self.device.snapshot()
        self._tag(*{_TIER_TAG[place.tier(i)] for i in range(n_layers)})
        if self.fp8_widened:
            self._tag(PassTag.FP8_WIDENED)

    def _tag_quant(self) -> None:
        """record the stored-weight path on MLX: as-stored quant bytes (`mlx_state.affine`) vs a bf16 slot"""
        if self.mlx is not None:
            self._tag(PassTag.QUANT_ASSTORED if self.mlx_state.affine else PassTag.QUANT_DEQUANT)

    def _tag_spec(self, proposed: int, accepted: int) -> None:
        """record one speculative pass's outcome: the accept/reject tags and the running counts"""
        rec = self._pass_rec
        if rec is None:
            rec = self._pass_rec = _PassRecorder()
        rec.spec_proposed += int(proposed)
        rec.spec_accepted += int(accepted)
        if accepted > 0:
            rec.tags.add(PassTag.SPEC_ACCEPT)
        if proposed > accepted:
            rec.tags.add(PassTag.SPEC_REJECT)

    def _called(self, tag: PassTag) -> None:
        """record an API call (btb.api): a call is not one pass's fork, so a generate's reset keeps it"""
        self._calls = self._calls | {tag}

    def last_pass_report(self) -> PassReport:
        """The provenance of the most recent `generate()` (its passes' tags accumulated) or of a bare
        `forward()`: the `PassTag`s its forks recorded and the speculation counts, with every API call made on
        the model so far. Empty before any pass or call."""
        rec = self._pass_rec
        rep = rec.report() if rec is not None else PassReport()
        return PassReport(rep.tags | self._calls, rep.spec_proposed, rep.spec_accepted)
