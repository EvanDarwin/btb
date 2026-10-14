# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The weight tiers: layers resident on the card, pinned in RAM, on the host kernels, in the 12-bit store, or
cold on the drive; how each is bound, streamed, prefetched and released."""

from __future__ import annotations

import dataclasses
import json
import mmap
import os
import struct
import threading
import time
import weakref
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .. import fp8
from .. import mlx as mlxdev
from ..fp8 import F8Weight
from ..gguf import QuantWeight
from ..kinds import Json, LayerKind, Quant, SlotKind, Tier
from ..options import Device
from ..pack12 import entries, unpack_bf16
from ..sysinfo import process_working_set_bytes
from .cache import CardRowsLayer, ForkIndexedLayer, ForkLayer, GrowLayer
from .device import where
from .fixed_rows import RowLinear, fix_linears, fix_rows_cls
from .host import _Experts, _HostLinear, _NGramRows, bf16_in_place, copy_bytes
from .native import Native
from .state import DRAFT_VOCAB, Shard, _State
from .stored import CardStored, held_bytes

if TYPE_CHECKING:
    from ..mlx import Shared
    from .cache import KvCache

_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}
RAM_BPS = 50 * 2**30  # the read bandwidth the plan prices a RAM tier at, weights and cache alike
DRIVE_BPS = int(3.4 * 2**30)  # the cold tier's rate until the drive is probed (scheduler.DriveBenchmark)
# the card tier: its weights and cache at the share of the card's own memory rate its matvecs and attention reach
# (Qwen3-0.6B on a 4070 Ti, 504 GB/s: 415-477 GB/s for the matvecs, 400-480 for the attention), and the rate
# where the card cannot be asked (`card_bps`)
CARD_EFF = 0.85
CARD_BPS = 400 * 10**9
# a card layer's fixed cost past its bytes: its nine kernels' launches, ramps and tails (Qwen3-0.6B's 28 layers
# stepped in 4.2 ms where their bytes alone take 2.8)
CARD_LAYER_MS = 0.05


def card_bps(dev: Any) -> float:
    """the rate the plan prices the card's reads at: the memory clock the driver reports, double-pumped, over its
    bus, at `CARD_EFF` of it; `CARD_BPS` where the card cannot be asked"""
    try:
        p = torch.cuda.get_device_properties(dev)
        peak = float(p.memory_clock_rate) * 1e3 * 2 * int(p.memory_bus_width) / 8
    except Exception:
        return float(CARD_BPS)
    return peak * CARD_EFF if peak > 0 else float(CARD_BPS)


@dataclass(frozen=True)
class ColdItem:
    """One linear's place in a cold ring slot: `nb` bytes at slot offset `at` as its kernel reads them, filled per
    `kind` from the drive at `path`/`off`, or from memory each pass where `path` is None (MEM: `_get` of the
    linear's key; QUANT: the GGUF tensor `src` gathered as stored). A P12 item carries its store entry, a CAST
    item the (bytes, dtype) it is read as before it is rewritten as bf16 in place."""

    m: _HostLinear
    kind: SlotKind
    at: int
    nb: int
    path: str | None = None
    off: int = 0
    p12: Json | None = None
    cast: tuple[int, torch.dtype] | None = None
    src: str | None = None

    @property
    def read_nb(self) -> int:
        """the bytes this item lands in its slot: a cast float's as stored"""
        return self.cast[0] if self.cast is not None else self.nb


@dataclass
class ColdRing:
    """The cold layers' ring, built by `_cold_start` and torn down by `_cold_stop`; its slots are shared with
    MLX where a cold layer runs there."""

    fh: dict[str, Any] = field(default_factory=dict)
    shared: list[Shared] | None = None
    slots: list[torch.Tensor] = field(default_factory=list)
    slot_of: dict[int, int] = field(default_factory=dict)
    recipe: dict[int, list[ColdItem]] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)
    ready: dict[int, threading.Event] = field(default_factory=dict)
    free: list[threading.Event] = field(default_factory=list)
    holds: list[int] = field(default_factory=list)
    abort: bool = False
    thread: threading.Thread | None = None
    bytes_per_pass: int = 0
    bytes: int = 0
    wait_s: float = 0.0


def report_line(report: Json) -> str:
    """`report()` as one line for the verbose log: the device, the tiers, the head, the drafter, the cache."""
    pl = report.get("placement") or {}
    gpu = set(pl.get("mlx") or ())  # the host layers Apple silicon's GPU runs
    host = [i for i in pl.get("host") or () if i not in gpu]
    parts = [f"resident {len(pl.get('resident') or ())}", *([f"gpu {len(gpu)}"] if gpu else [])]
    parts += [f"host {len(host)}", f"cold {len(pl.get('cold') or ())}"]
    return (
        f"[report] {report.get('device')}: {', '.join(parts)}; head {pl.get('head')}; "
        f"drafter {pl.get('drafter')}; kv {pl.get('kv')}; {pl.get('compute_dtype')}"
    )


class _TiersMixin(_State):
    report_line = staticmethod(report_line)

    def _layer_bytes_stored(self, i: int, packed: bool = False) -> int:
        if not packed or not getattr(self, "_packed", None):
            return self._layer_bytes(i)
        base = f"{self.prefix}layers.{i}."
        n = 0
        for k in self.weight_map:
            if k.startswith(base) and self.fam.dense_key(k):
                e = self._packed.get(k)
                if e is not None and not e["raw"]:
                    n += e["lo"] + e["hi4"] + e["pad"] + 5 * e["esc"]
                else:
                    _mm, hdr, _ = self._shard(self.weight_map[k])
                    a, b = hdr[k]["data_offsets"]
                    n += b - a
        return n

    def plan_budget(
        self,
        ram_gb: float,
        vram_gb: float,
        packed: bool = False,
        fp32: bool = True,
        working_vram_gb: float = 1.5,
        working_ram_gb: float = 2.0,
        slots: int = 2,
        drafter: bool = False,
        os_reserve_gb: float = 1.0,
        vram_reserve_gb: float = 0.5,
        resident_fp32: bool = False,
        prefill_card: bool = False,
        context: int = 0,
        kv_host: bool = False,
        drive_bps: float | None = None,
        gpu_bps: float | None = None,
    ) -> Json:
        L = self.L
        cfg = self.cfg
        card = bool(prefill_card)
        head_b = cfg.vocab_size * cfg.hidden_size * 2
        # the head's host size is its packed size where the store holds it
        pk = (getattr(self, "_packed", None) or {}).get(getattr(self, "head_key", "")) if packed else None
        head_host_b = (pk["lo"] + pk["hi4"] + pk["pad"] + 5 * pk["esc"]) if (pk and not pk["raw"]) else head_b
        # 4096 positions at least: a cache's first growth reaches that far (cache.py, the card's arena); a
        # linear-attention layer keeps a fixed state, no per-position cache
        kv_rows = max(4096, int(context or 0))
        hq = int(cfg.num_attention_heads)
        hk = int(getattr(cfg, "num_key_value_heads", None) or hq)
        hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // hq)
        # an attention layer's index (Qwen4's sparse attention): its raw keys live with the rows, its pooled keys on
        # the card wherever the rows are - under `kv_host` too, where nothing else of the cache is the card's
        index_card, index_rows = self.fam.attn_index_bytes(cfg, kv_rows)
        kv_layer = {
            i: 0 if self.layer_types[i] == LayerKind.LINEAR else 2 * hk * hd * (4 if fp32 else 2) * kv_rows + index_rows
            for i in range(L)
        }
        index_layer = {i: 0 if self.layer_types[i] == LayerKind.LINEAR else index_card for i in range(L)}
        # the drafter the load runs (`drafter`: speculation on, with the drafting head as its proposer), its dense
        # tensors and its head slice at bf16; a mixture in its layer is the expert store's, never placed with it
        drafter_b = (
            self._drafter_bytes() + min(DRAFT_VOCAB, int(cfg.vocab_size)) * int(cfg.hidden_size) * 2 if drafter else 0
        )
        bf16 = {i: self._layer_bytes(i) for i in range(L)}
        stored = {i: self._layer_bytes_stored(i, packed) for i in range(L)}
        vram = int(vram_gb * 2**30) - int(working_vram_gb * 2**30) - int(vram_reserve_gb * 2**30)
        head_on_card = vram >= head_b
        if head_on_card:
            vram -= head_b
        drafter_on_card = drafter and vram >= drafter_b
        if drafter_on_card:
            vram -= drafter_b
        types = sorted(set(self.layer_types))
        big = {lt: max(bf16[i] for i in range(L) if self.layer_types[i] == lt) for lt in types}
        big_stored = {lt: max(stored[i] for i in range(L) if self.layer_types[i] == lt) for lt in types}
        # a float32 shadow per layer structure (model.py): a layer kind's layers can differ, and those that do differ
        # in size, so each distinct (kind, size) is one shadow at twice its bf16 bytes
        shapes = {(self.layer_types[i], bf16[i]) for i in range(L)}
        shadow_b = sum(2 * b for _, b in shapes) if (fp32 and not resident_fp32) else 0
        tmpl_b = sum(big[lt] * (1 if fp32 else 2) for lt in types)
        if fp32 and not resident_fp32 and vram >= shadow_b:
            vram -= shadow_b
        else:
            shadow_b = 0
        prefill_card = bool(prefill_card) and vram >= tmpl_b + (
            0 if shadow_b else sum(2 * big[lt] for lt in types) * (1 if fp32 else 0)
        )
        if prefill_card:
            vram -= tmpl_b
        else:
            tmpl_b = 0
        resident = []
        for i in reversed(range(L)):
            b = bf16[i] * (2 if (fp32 and resident_fp32) else 1) + (0 if kv_host else kv_layer[i]) + index_layer[i]
            if vram >= b:
                resident.append(i)
                vram -= b
            else:
                break
        resident = sorted(resident)
        # the card has room for the templates: a split the caller forces (`cpu_layers`) prefills its host layers
        # there, where a plan that holds every layer on the card has no host layer to prefill
        prefill_card_room = prefill_card
        if len(resident) == L:
            prefill_card = False
        rest = [i for i in range(L) if i not in resident]
        # what the cold reader lands: a float32 layer as stored, before it is rewritten as bf16
        read = {i: stored[i] + self._cast_growth(i) for i in rest}
        slot_b = max(read[i] for i in rest) if rest else 0
        ram = int(ram_gb * 2**30) - int(working_ram_gb * 2**30) - int(os_reserve_gb * 2**30) - slots * slot_b
        if not head_on_card:
            ram -= head_host_b
        if drafter and not drafter_on_card:
            ram -= drafter_b
        kv_card_b = (0 if kv_host else sum(kv_layer[i] for i in resident)) + sum(index_layer[i] for i in resident)
        kv_host_b = sum(kv_layer[i] for i in rest) + (sum(kv_layer[i] for i in resident) if kv_host else 0)
        ram -= kv_host_b
        total_rest = sum(stored[i] for i in rest)

        def cold_for(room: int) -> int:
            n = 0
            while n < len(rest) and total_rest - sum(stored[i] for i in rest[:n]) > room:
                n += 1
            return n

        # pinned staging (`_staging`, `_pstaging`): one layer of each type, its stored form beside it under the
        # 12-bit store; the buffers exist once anything streams, so a plan ending with cold layers is priced again
        stage_b = sum(big[lt] for lt in types) + (sum(big_stored[lt] for lt in types) if packed else 0)
        staging_b = stage_b if (card and prefill_card and rest) else 0
        n_cold = cold_for(ram - staging_b)
        if card and n_cold and not staging_b:
            staging_b = stage_b
            n_cold = cold_for(ram - staging_b)
        cold = sorted({rest[round(j * len(rest) / n_cold)] for j in range(n_cold)}) if n_cold else []
        k = 0
        while len(cold) < n_cold and k < len(rest):
            if rest[k] not in cold:
                cold.append(rest[k])
            k += 1
        cold = sorted(cold)
        warm = [i for i in rest if i not in cold]
        cold_b = sum(read[i] for i in cold)
        warm_b = sum(stored[i] for i in warm)
        warm_ms = warm_b / RAM_BPS * 1e3
        cold_ms = cold_b / (drive_bps or DRIVE_BPS) * 1e3
        # what a token's attention reads of the cache at the full context: every row for the plain block's, the
        # indexer's picks for a sparse attention's on the card (`attn_read_rows`) - priced whole, a million positions
        # of Qwen4's rows in RAM read as 20 GB a token, where its card program reads a budget of them. A host layer's
        # attention reads every row whatever the family (Qwen4's host path grows and re-pools its rows whole)
        reads = min(1.0, self.fam.attn_read_rows(cfg, kv_rows) / kv_rows)
        rest_kv_b = sum(kv_layer[i] for i in rest)
        # the host's attention over the cache at the full context: the host layers' rows whole, the resident layers'
        # kept in RAM (`kv_host`) as their attention reads them
        kv_read_ms = (rest_kv_b + (kv_host_b - rest_kv_b) * reads) / RAM_BPS * 1e3
        # the card's share of a token, priced as the host's is, by what it reads: its layers, the head where it holds
        # it and the cache it holds at the full context (what its attention reads of it, and an index's share whole),
        # at the card's rate, and each layer's kernels' fixed cost. A flat 3 ms a resident layer priced Qwen3-0.6B at
        # 89 ms a token (it runs at 4) and made a layer more on the card look dearer, where the cache's placement
        # (`choose`) weighs one against the other
        vram_layers_b = sum(bf16[i] for i in resident) * (2 if (fp32 and resident_fp32) else 1)
        index_card_b = sum(index_layer[i] for i in resident)
        card_b = vram_layers_b + (head_b if head_on_card else 0) + (kv_card_b - index_card_b) * reads + index_card_b
        card_ms = card_b / (gpu_bps or CARD_BPS) * 1e3 + CARD_LAYER_MS * len(resident)
        head_host_ms = 0.0 if head_on_card else head_host_b / RAM_BPS * 1e3
        return {
            "head_on_card": head_on_card,
            "drafter_on_card": drafter_on_card,
            "resident": resident,
            "host": warm + cold,
            "cold": cold,
            "warm": warm,
            "prefill_card": prefill_card,
            "prefill_card_room": prefill_card_room,
            "bytes": {
                "vram_layers": vram_layers_b,
                "head": head_b if head_on_card else head_host_b,
                "drafter": drafter_b,
                "warm": warm_b,
                "cold": cold_b,
                "slots": slots * slot_b,
                "shadow": shadow_b,
                "templates": tmpl_b,
                "kv_card": kv_card_b,
                "kv_host": kv_host_b,
                "staging": staging_b,
            },
            # the host's layers (the RAM's and the drive's streaming side by side), then the card's, one after the
            # other as a token passes them
            "predicted_ms_per_token": max(warm_ms, cold_ms) + head_host_ms + card_ms,
            "kv_read_ms": kv_read_ms,
            "caps": {
                "ram_gb": ram_gb,
                "vram_gb": vram_gb,
                "os_reserve_gb": os_reserve_gb,
                "vram_reserve_gb": vram_reserve_gb,
            },
        }

    def report(self) -> Json:
        """The engine's ledger as one JSON record: where each layer ended up, what each tier holds, the peaks
        the process reached, what it decodes with. resident/host/cold count every layer once (`self.host` holds
        the cold layers too; the record does not); `mlx` is a second axis over the same layers. Every read
        defaults, so a partially built, CPU or MLX engine reports rather than raising."""
        from .. import device_name, peak_memory

        dev = getattr(self, "dev", torch.device("cpu"))
        cuda = dev.type == Device.CUDA
        mlx = getattr(self, "mlx", None)
        cfg = getattr(self, "cfg", None)
        types = list(getattr(self, "layer_types", ()) or ())
        L = min(int(getattr(self, "L", 0) or 0), len(types))
        on_drive = getattr(self, "cold", None) or set()
        cold = sorted(on_drive)
        resident = sorted(getattr(self, "resident", {}) or {})
        host = [i for i in sorted(getattr(self, "host", {}) or {}) if i not in on_drive]
        packed = bool(getattr(self, "_packed", None))
        cdt = getattr(self, "compute_dtype", None)
        fp32 = cdt is not None and cdt != torch.bfloat16
        wide = 2 if (fp32 and getattr(self, "resident_fp32", False)) else 1
        # unpacked, the stored size is the bf16 size: the second header pass is paid only with the 12-bit store open
        bf16 = {i: self._layer_bytes(i) for i in range(L)} if getattr(self, "weight_map", None) else {}
        stored = {i: self._layer_bytes_stored(i, True) for i in range(L)} if (packed and bf16) else bf16
        head_b = int(cfg.vocab_size) * int(cfg.hidden_size) * 2 if cfg is not None else 0
        # off the headers, never through `_get`: a report may not move the engine's counters
        mtp = [k for k in getattr(self, "weight_map", {}) if k.startswith("mtp.")]
        drafter_b = self._drafter_bytes() if mtp else 0
        tmpl_b = 0
        for lt, mods in (getattr(self, "templates", {}) or {}).items():
            tmpl_b += len(mods) * max((bf16.get(i, 0) for i in range(L) if types[i] == lt), default=0)
        # where the head, the drafter and the cache sit: the live placement's reading, the one the passes hold
        dv = getattr(self, "device", None)
        snap = dv.snapshot() if dv is not None else None
        head, drafter, kv = (snap.head, snap.drafter, snap.kv) if snap is not None else (Tier.NONE,) * 3
        rss = peak_memory()[0]
        pl = getattr(self, "plan", None)
        # the run's growth past its plan: the working set now, less the footprint the plan was drawn at and the
        # bytes it priced into RAM - the figure the plan's growth estimate is read against
        growth = None
        if pl is not None and pl.budget is not None:
            pb = pl.bytes
            in_ram = (
                pb.warm
                + pb.slots
                + pb.staging
                + pb.kv_host
                + (0 if pl.head_on_card else pb.head)
                + (0 if pl.drafter_on_card else pb.drafter)
            )
            ws = process_working_set_bytes()
            growth = {
                "working_set_gb": ws / 2**30,
                "priced_gb": in_ram / 2**30,
                "past_plan_gb": (ws - pl.budget.footprint - in_ram) / 2**30,
                "estimate_gb": pl.budget.growth / 2**30,
            }
        return {
            "device": device_name(self),
            "plan": dataclasses.asdict(pl) if pl is not None else None,
            "granted": dict(getattr(getattr(self, "scheduler", None), "granted", None) or {}),
            "growth": growth,
            "model": {
                "path": getattr(self, "dir", ""),
                "layers": int(getattr(self, "L", 0) or 0),
                "family": getattr(getattr(self, "fam", None), "kind", None),
                "gguf": (
                    {
                        "file": self.gguf.file,
                        "types": dict(self.gguf.types()),
                        "packed": bool(getattr(self, "gguf_packed", True)) and self.gguf.packable(),
                    }
                    if self.gguf is not None
                    else None
                ),
                "has_drafter": bool(mtp),
            },
            "placement": {
                "resident": resident,
                "host": host,
                "cold": cold,
                "mlx": sorted(getattr(self, "mlx_layers", ()) or ()),
                "head": head,
                "drafter": drafter,
                "kv": kv,
                "kv_bits": getattr(self, "kv_bits", None),
                "packed": packed,
                "templates": sum(len(v) for v in (getattr(self, "templates", {}) or {}).values()),
                "prefetch": bool(getattr(self, "prefetch", False)),
                "compute_dtype": _DTYPE_NAME.get(cdt, str(cdt).rsplit(".", 1)[-1]) if cdt is not None else "bf16",
                "shadow": sorted(getattr(self, "shadow", {}) or {}),
            },
            "bytes": {
                "resident": sum(bf16.get(i, 0) for i in resident) * wide,
                "host": sum(stored.get(i, 0) for i in host),
                "cold": sum(stored.get(i, 0) for i in cold),
                "head": head_b,
                "drafter": drafter_b,
                "templates": tmpl_b,
            },
            "peak": {
                "ram_gb": rss / 2**30,
                # free-read: the run's report
                "vram_reserved_gb": (torch.cuda.max_memory_reserved(dev) / 2**30) if cuda else 0.0,
                # free-read: the run's report
                "vram_allocated_gb": (torch.cuda.max_memory_allocated(dev) / 2**30) if cuda else 0.0,
                "mlx_gb": (mlx.peak_bytes() / 2**30) if mlx is not None else 0.0,
            },
            "speculation": {
                "proposer": getattr(self, "proposer", None),
                "tree_budget": int(getattr(self, "tree_budget", 0) or 0),
                "v_max": int(getattr(self, "v_max", 0) or 0),
                "ngram_tree": bool(getattr(self, "ngram_tree", True)),
                "draft_vocab": int(getattr(self, "draft_vocab", 0) or 0),
            },
            "counters": {
                "load_s": round(float(getattr(self, "load_s", 0.0) or 0.0), 3),
                "compute_s": round(float(getattr(self, "compute_s", 0.0) or 0.0), 3),
                "wait_s": round(float(getattr(self, "wait_s", 0.0) or 0.0), 3),
                "cold_bytes": int(getattr(getattr(self, "cold_ring", None), "bytes", 0) or 0),
                "cold_wait_s": round(float(getattr(getattr(self, "cold_ring", None), "wait_s", 0.0) or 0.0), 3),
                "expert_stat": {
                    k: (round(v, 3) if isinstance(v, float) else int(v))
                    for k, v in (getattr(self, "expert_stat", None) or {}).items()
                    if isinstance(v, (int, float))
                },
                "expert_store": {
                    k: (round(v, 4) if isinstance(v, float) else int(v))
                    for k, v in (getattr(getattr(self, "expert_store", None), "stat", None) or {}).items()
                    if isinstance(v, (int, float))
                },
            },
        }

    def _stored_name(self, key: str) -> str | None:
        """the GGUF tensor of checkpoint weight `key` the host's native kernels multiply in its blocks: a stored
        type the library has the gemv for, under `gguf_packed`; None for any other"""
        gg, gq = self.gguf, Native.gemv_quant
        if gg is None or gq is None or not self.gguf_packed:
            return None
        name = self._gguf_names.get(key)
        return name if name is not None and gg.storable(name) and gg.tensors[name].tensor_type.name in gq else None

    def _host_stored_keys(self, i: int, linears: Iterable[str]) -> dict[str, str]:
        """layer `i`'s linear weights (`linears`, named in the layer) that its native kernels multiply in their GGUF
        blocks (`_stored_name`), by checkpoint key -> file tensor. None on an MLX layer, whose binders take the
        blocks to its own kernels (`_bind_mlx_linears`)"""
        if self.mlx is not None and i in self.mlx_layers:
            return {}
        base = f"{self.prefix}layers.{i}."
        return {base + n: name for n in linears if (name := self._stored_name(base + n)) is not None}

    def _mlx_slot_keys(self, i: int, linears: Iterable[str]) -> dict[str, str]:
        """MLX layer `i`'s linear weights (`linears`, named in the layer) a cold slot holds as their GGUF blocks for
        the type's Metal kernel to read in place (`Backend.weight_stored_slot`), by checkpoint key -> file tensor"""
        gg = self.gguf
        if gg is None or self.mlx is None or not self.gguf_packed:
            return {}
        base = f"{self.prefix}layers.{i}."
        out: dict[str, str] = {}
        for n in linears:
            name = self._gguf_names.get(base + n)
            if name is not None and gg.storable(name):
                out[base + n] = name
        return out

    def _stored_type(self, m: _HostLinear) -> Quant:
        """the GGUF type of a stored linear's file tensor"""
        assert self.gguf is not None and m.key is not None  # a stored linear is a GGUF's checkpoint linear
        return Quant(self.gguf.tensors[self._gguf_names[m.key]].tensor_type.name)

    def _bind_host_stored(self, i: int, layer: Any) -> None:
        """host layer `i`'s linears its native kernels multiply as stored (`_host_stored_keys`) bound to the file's
        blocks (`_bind_stored`)"""
        lins = {n + ".weight": m for n, m in layer.named_modules() if isinstance(m, _HostLinear)}
        keys = self._host_stored_keys(i, lins)
        base = f"{self.prefix}layers.{i}."
        self._bind_stored({base + n: m for n, m in lins.items() if base + n in keys})

    def _bind_stored(self, lins: Mapping[str, _HostLinear]) -> None:
        """host linears, by checkpoint key, bound to the GGUF blocks their native kernels multiply (`_stored_name`,
        `_HostLinear.quant`), each weight a shape with no bytes behind it; a key with none keeps its weight"""
        for key, m in lins.items():
            name = self._stored_name(key)
            if name is None:
                continue
            assert self.gguf is not None  # a stored name is a GGUF's
            m.quant, m.scratch = self.gguf.stored(name), self.scratch
            shape = m.weight.shape
            m.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16).expand(*shape), requires_grad=False)

    def _bind_host_packed_layer(self, layer: Any) -> Any:
        n_b = n_all = 0
        for m in layer.modules():
            if isinstance(m, _HostLinear) and m.key:
                n_all += 1
                pk = self._get_packed(m.key)
                if pk is None:
                    continue
                blob, tbl, e = pk
                a1 = e["lo"]
                a2 = a1 + e["hi4"] + e["pad"]
                a3 = a2 + 4 * e["esc"]
                esc_idx = blob[a2:a3].view(torch.int32) if e["esc"] else torch.zeros(0, dtype=torch.int32)
                esc_val = blob[a3:] if e["esc"] else torch.zeros(0, dtype=torch.uint8)
                m.packed = (blob[:a1], blob[a1 : a1 + e["hi4"]], tbl, esc_idx, esc_val, int(e["esc"]))
                n_b += 1
        return n_b, n_all

    def bind_host_packed(self) -> Any:
        n_b = n_all = 0
        for layer in self.host.values():
            b, a = self._bind_host_packed_layer(layer)
            n_b += b
            n_all += a
        self.log(f"[stream] host linears bound to the packed store: {n_b} of {n_all}")
        self._bind_cold()
        return n_b

    def _head_stored(self) -> str | None:
        """the head's GGUF tensor where the native kernels multiply it as stored (a type the library has the gemv
        for, under `gguf_packed`), else None"""
        gg, gq = self.gguf, Native.gemv_quant
        name = self._gguf_names.get(self.head_key) if gg is not None else None
        if gg is None or gq is None or name is None or not self.gguf_packed or not gg.storable(name):
            return None
        return name if gg.tensors[name].tensor_type.name in gq else None

    def _head_host(self) -> Any:
        """the head as a host linear, made once: on its stored blocks where the native kernels take its type
        (`_head_stored`), else the checkpoint's bf16"""
        if self.head_host is None:
            name = self._head_stored() if self.mlx is None else None
            if name is not None:
                assert self.gguf is not None  # a stored name is a GGUF's
                w = self.gguf.stored(name)
                self.head_host = _HostLinear(torch.zeros((), dtype=torch.bfloat16).expand(*w.shape), key=self.head_key)
                self.head_host.quant, self.head_host.scratch = w, self.scratch
            else:
                self.head_host = _HostLinear(self._get(self.head_key), key=self.head_key)
        return self.head_host

    def _span(self, key: str) -> tuple[str | None, int, int]:
        """where a tensor's bf16 bytes are read from: (file path, offset, bytes), or (None, 0, bytes) when they
        are not on the drive as bf16 (a GGUF tensor of another storage type, or a safetensors float of another
        precision: `_get` has them in memory, cast)"""
        shard = self.weight_map[key]
        _mm, hdr, base = self._shard(shard)
        info = hdr[key]
        a, b = info["data_offsets"]
        gg = info.get("gguf")
        if gg is not None:
            if gg["type"] == "BF16":
                return os.path.join(self.dir, shard), int(gg["offset"]), int(gg["nbytes"])
            return None, 0, b - a
        if self._cast_on_read(info):
            return None, 0, self._held_nbytes(info)
        return os.path.join(self.dir, shard), base + a, b - a

    def _cast_on_read(self, info: dict[str, Any]) -> bool:
        """whether `_held` casts this safetensors tensor: any fp16 one, and every float at another precision in a
        checkpoint that stores its weights so; a bf16 checkpoint's own float32 tensors (norms, routers) are read
        as stored"""
        if info.get("gguf") is not None:
            return False
        dt = self.ST_DTYPES[info["dtype"]]
        return dt in (torch.float16, fp8.E4M3) or (self.held_cast and dt.is_floating_point and dt != torch.bfloat16)

    def _shape(self, key: str) -> tuple[int, ...]:
        """a tensor's shape from its header, nothing read"""
        _mm, hdr, _ = self._shard(self.weight_map[key])
        return tuple(int(x) for x in hdr[key]["shape"])

    def _fp8(self, key: str) -> bool:
        """whether a safetensors tensor is stored as FP8, a scale beside it (btb/fp8.py)"""
        shard = self.weight_map.get(key)
        if shard is None or self.gguf is not None:
            return False
        _mm, hdr, _ = self._shard(shard)
        return self.ST_DTYPES.get(hdr[key].get("dtype", "")) == fp8.E4M3

    def _fp8_scale(self, key: str) -> torch.Tensor:
        """the stored scale of FP8 tensor `key`, as the checkpoint holds it"""
        sk = fp8.scale_key(key, self.weight_map)
        if sk is None:
            raise KeyError(f"[stream] FP8 tensor {key!r} has no scale beside it")
        return self._get(sk, stored=True)

    def _f8_weights(self, key: str) -> list[F8Weight]:
        """FP8 tensor `key` as stored, views of the checkpoint's bytes: one F8Weight, or one per expert of a fused
        `[E, rows, cols]` expert tensor"""
        mm, hdr, base = self._shard(self.weight_map[key])
        info = hdr[key]
        shape = [int(x) for x in info["shape"]]
        rows, cols = shape[-2:]
        n = rows * cols
        count = n * (shape[0] if len(shape) == 3 else 1)
        raw = torch.frombuffer(mm, dtype=torch.uint8, count=count, offset=base + info["data_offsets"][0])
        s = self._fp8_scale(key)
        if len(shape) == 2:
            return [F8Weight(raw, s, rows, cols)]
        per = s.reshape(-1, *s.shape[-2:]) if s.dim() >= 2 else s.reshape(1, 1, 1)
        return [
            F8Weight(raw[e * n : (e + 1) * n], per[e if per.shape[0] > 1 else 0], rows, cols) for e in range(shape[0])
        ]

    def _stored_span(self, key: str) -> tuple[str, int, tuple[int, torch.dtype]] | None:
        """where a tensor `_held` casts sits on the drive as stored: (file path, offset, (bytes, dtype)); None
        for any other, and for an FP8 one, whose widening needs its scale (`_get` widens it)"""
        shard = self.weight_map[key]
        _mm, hdr, base = self._shard(shard)
        info = hdr[key]
        if not self._cast_on_read(info) or self.ST_DTYPES[info["dtype"]] == fp8.E4M3:
            return None
        a, b = info["data_offsets"]
        return os.path.join(self.dir, shard), base + a, (int(b - a), self.ST_DTYPES[info["dtype"]])

    def _cast_growth(self, i: int) -> int:
        """the bytes layer `i`'s float32 tensors read as stored take beyond their bf16 (the cold slot holds them
        as they land)"""
        if not self.held_cast:
            return 0
        base = f"{self.prefix}layers.{i}."
        n = 0
        for k in self.weight_map:
            if k.startswith(base) and self.fam.dense_key(k):
                _mm, hdr, _ = self._shard(self.weight_map[k])
                a, b = hdr[k]["data_offsets"]
                n += int(b - a) - self._held_nbytes(hdr[k])
        return n

    def _held_nbytes(self, info: dict[str, Any]) -> int:
        """a safetensors tensor's bytes as the engine holds it (`_held`)"""
        a, b = info["data_offsets"]
        if not self._cast_on_read(info):
            return int(b - a)
        return int(b - a) // self.ST_DTYPES[info["dtype"]].itemsize * 2

    def _held(self, t: torch.Tensor) -> torch.Tensor:
        """a checkpoint tensor as the engine holds it: an fp16 one, and in a checkpoint stored at another float
        precision every float one, as bf16 - the one weight precision the host gemv, the CPU GEMM, the MLX slots,
        the cold ring and the 12-bit store read - and anything else as stored"""
        if t.dtype == torch.float16 or (self.held_cast and t.is_floating_point() and t.dtype != torch.bfloat16):
            return t.to(torch.bfloat16)
        return t

    def _layer_items(self, lins: Sequence[Any]) -> tuple[list[tuple[Any, str | None, int, int, int]], int]:
        """(module, file path or None, offset, bytes, slot offset) per linear, 64-byte aligned, and the bytes in
        all; a None path means the bytes come from the module's weight in memory, not the drive."""
        items, cur = [], 0
        for m in lins:
            path, off, nb = self._span(m.key)
            items.append((m, path, off, nb, cur))
            cur += (nb + 63) // 64 * 64
        return items, cur

    def _track(self, cache: KvCache) -> KvCache:
        """`cache` among those a layer's move reaches while it lives: an idle session's, a fork's, the running pass's"""
        live: weakref.WeakSet[KvCache] = self.__dict__.setdefault("_live_caches", weakref.WeakSet())
        live.add(cache)
        return cache

    def _caches_to(self, i: int, dev: str | torch.device, cache: KvCache | None = None) -> None:
        """layer i's rows in every live cache (and `cache`) moved to `dev`, where the layer now runs - an attention
        layer coming to the card under `kv_host` leaves its rows in RAM, where they live whatever the placement (its
        linear states, small and its own step's, come with it). The prefix cache's pages move once for every
        conversation, to the region where the layer now runs (`PrefixCache.place`)"""
        pc = self.__dict__.get("_kv")
        if pc is not None:
            pc.place(self, i)
        if self._rows_stay(i, dev):
            return
        live: list[KvCache] = list(self.__dict__.get("_live_caches", ()))
        caches = list({id(c): c for c in [*live, *([cache] if cache is not None else [])]}.values())
        self._rows_to(caches, [i], dev)

    def _rows_stay(self, i: int, dev: str | torch.device) -> bool:
        """whether layer `i`'s rows stay where they are as the layer moves to `dev`: an attention layer's coming to
        the card under `kv_host`"""
        return (
            torch.device(dev).type == Device.CUDA
            and bool(getattr(self, "kv_host", False))
            and self.layer_types[i] != LayerKind.LINEAR
        )

    def _rows_to(self, caches: Sequence[Any], layers: Sequence[int], dev: str | torch.device) -> None:
        """the rows of `layers` in `caches` moved to `dev`, granted together first where they come onto the card:
        a move the card cannot take is refused whole, before a row has moved"""
        dev = where(torch.device(dev))  # torch's own names too ('meta'), not only the ones --device takes
        sched = getattr(self, "scheduler", None)
        if dev.type == Device.CUDA and sched is not None:
            nbytes = sum(_rows_bytes(c, i, dev) for c in caches for i in layers)
            if nbytes:
                sched.grant(nbytes, "kv", requester=f"layers {list(layers)}'s rows onto the card", device=dev, draws="")
        for c in caches:
            for i in layers:
                self._cache_to(c, i, dev)

    @staticmethod
    def _cache_to(cache: Any, i: int, device: str | torch.device) -> None:
        if cache is None or i >= len(cache.layers):
            return
        # compared with the rows' own devices below: a card by its index, as they name it
        dev = where(torch.device(device))
        cl = cache.layers[i]
        if getattr(cl, "paged", False):
            # its rows live in the engine's pool of pages, which the host's attention reads whatever the layer's
            # placement (btb/engine/paged.py): nothing of the cache's own moves
            return
        if isinstance(cl, ForkLayer):
            cl.to(dev)
            return
        if isinstance(cl, CardRowsLayer):
            # a fork's or a batch's rows leaving the card's arena: a fork's layer where the layer now runs (the
            # batch takes the torch pass from its next step)
            if dev.type != cl.device.type:
                cache.layers[i] = cl.to_fork(dev)
            return
        if isinstance(cl, GrowLayer) and not cl.shared and cl._buf is not None and cl._buf[0].device != dev:
            # the rows copied out and the buffer they grew in let go now, not at the cache's next append
            k, v = cl.keys, cl.values
            cl._buf, cl._an = None, None
            cl._set_rows(k, v)
        if isinstance(cl, GrowLayer) and cl.is_initialized and isinstance(cl.keys, torch.Tensor):
            if cl.keys.device != dev:
                cl._set_rows(cl.keys.to(dev), cl.values.to(dev))
        for attr in ("keys", "values", "conv_states", "recurrent_states", "indexer_keys"):
            t = getattr(cl, attr, None)
            if isinstance(t, dict):
                for k, v in t.items():
                    if isinstance(v, torch.Tensor) and v.device != dev:
                        t[k] = v.to(dev)
            elif isinstance(t, torch.Tensor) and t.device != dev:
                setattr(cl, attr, t.to(dev))

    def _layer_bytes(self, i: int) -> int:
        base = f"{self.prefix}layers.{i}."
        n = 0
        for k in self.weight_map:
            if k.startswith(base) and self.fam.dense_key(k):
                _mm, hdr, _ = self._shard(self.weight_map[k])
                n += self._held_nbytes(hdr[k])
        return n

    def _drafter_bytes(self) -> int:
        """the drafting head's bytes where it is placed, at bf16 off the headers: its `mtp.*` tensors the family
        reads with a layer (`dense_key`) - a mixture's experts in its drafting layer stream through the expert
        store as every MoE layer's do, and an FP8 scale is read with the tensor it scales"""
        n = 0
        for k in self.weight_map:
            if k.startswith("mtp.") and self.fam.dense_key(k):
                _mm, hdr, _ = self._shard(self.weight_map[k])
                c = 2
                for d in hdr[k]["shape"]:
                    c *= int(d)
                n += c
        return n

    def _regrow_bytes(self) -> int:
        if not self._shed:
            return 0
        what = self._shed[-1]
        if what == "head":
            return self.cfg.vocab_size * self.cfg.hidden_size * 2
        if what == "drafter":
            return self._drafter_bytes()
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        i = int(what.split()[1])
        # and its rows in every live cache, which follow it onto the card (not under `kv_host`: `_rows_stay`) - the
        # prefix cache's pages' once for every conversation, an arena of the card region's slots
        live = [] if self._rows_stay(i, self.dev) else list(self.__dict__.get("_live_caches", ()))
        rows = sum(_rows_bytes(c, i, self.dev) for c in live)
        pc = self.__dict__.get("_kv")
        card = pc.pool.card if pc is not None else None
        if card is not None and not self._rows_stay(i, self.dev) and self.layer_types[i] != LayerKind.LINEAR:
            rows += card.layer_bytes()
        return self._layer_bytes(i) * (2 if (fp32 and self.resident_fp32) else 1) + rows

    def _realloc_bytes(self) -> int:
        n = 0
        if self.resident_head and self.head is not None:
            n += held_bytes(self.head)
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            n += self._drafter_bytes()
        return n

    def _cold_item(self, m: _HostLinear, at: int, gguf: str | None) -> ColdItem:
        """where linear `m` of a cold layer sits in its slot from `at`, and how the ring fills it (`SlotKind`);
        `gguf` its file tensor where the layer's native kernels multiply it as stored (`_host_stored_keys`)"""
        assert m.key is not None  # a ring item is a checkpoint linear
        if gguf is not None:
            assert self.gguf is not None  # a stored name is a GGUF's
            span = self.gguf.stored_span(gguf)
            nb = int(self.gguf.tensors[gguf].n_bytes)
            if span is None:
                return ColdItem(m, SlotKind.QUANT, at, nb, src=gguf)
            return ColdItem(m, SlotKind.QUANT, at, nb, span[0], span[1])
        pk = getattr(self, "_packed", {}).get(m.key)
        if pk is not None and not pk["raw"]:
            nb = pk["lo"] + pk["hi4"] + pk["pad"] + 5 * pk["esc"]
            return ColdItem(m, SlotKind.P12, at, nb, os.path.join(self.dir, pk["shard"]), int(pk["off"]), p12=pk)
        path, off, nb = self._span(m.key)
        stored = self._stored_span(m.key)
        if stored is not None:
            # a float at another precision, read as stored into its region and rewritten as bf16 there as it lands
            # (`_cold_cast`)
            spath, soff, cast = stored
            return ColdItem(m, SlotKind.CAST, at, nb, spath, soff, cast=cast)
        # not on the drive as bf16 (a GGUF tensor of a type no native gemv takes here): each pass takes it from
        # `_get`, which reads and dequantizes it
        return ColdItem(m, SlotKind.BF16 if path is not None else SlotKind.MEM, at, nb, path, off)

    def _cold_fill(self, it: ColdItem, slot: torch.Tensor) -> None:
        """a memory item's bytes into its region: a GGUF tensor's blocks gathered as stored, or `_get`'s bf16"""
        if it.kind is SlotKind.QUANT:
            assert self.gguf is not None and it.src is not None  # a QUANT item from memory names its tensor
            copy_bytes(slot[it.at : it.at + it.nb], self.gguf.raw(it.src))
        else:
            assert it.m.key is not None  # a ring item is a checkpoint linear
            copy_bytes(slot[it.at : it.at + it.nb], self._get(it.m.key))

    def _bind_cold(self) -> None:
        if not self.cold:
            return
        recipes: dict[int, list[ColdItem]] = {}
        sizes: dict[int, int] = {}
        for i in sorted(self.cold):
            items: list[ColdItem] = []
            cur = 0
            layer = self.host[i]
            lins = {n + ".weight": m for n, m in layer.named_modules() if isinstance(m, _HostLinear) and m.key}
            stored = self._mlx_slot_keys(i, lins) if i in self.mlx_layers else self._host_stored_keys(i, lins)
            for m in lins.values():
                cur = (cur + 63) // 64 * 64
                assert m.key is not None  # filtered above
                it = self._cold_item(m, cur, stored.get(m.key))
                items.append(it)
                cur += max(it.nb, it.read_nb)
            recipes[i], sizes[i] = items, cur
        slot_bytes = max(sizes.values())
        R = max(1, min(self.cold_slots, len(self.cold)))
        self.cold_ring.shared = None
        if self.mlx is not None:
            self.cold_ring.shared = [mlxdev.Shared(slot_bytes) for _ in range(R)]
            self.cold_ring.slots = [sh.torch for sh in self.cold_ring.shared]
        else:
            self.cold_ring.slots = [torch.empty(slot_bytes, dtype=torch.uint8) for _ in range(R)]
        order = sorted(self.cold)
        self.cold_ring.slot_of = {i: k % R for k, i in enumerate(order)}
        self.cold_ring.recipe = recipes
        for i, items in recipes.items():
            slot = self.cold_ring.slots[self.cold_ring.slot_of[i]]
            for it in items:
                m, region = it.m, slot[it.at : it.at + it.nb]
                rows, cols = m.weight.shape
                shared = self.cold_ring.shared
                mlx_slot = shared is not None and i in self.mlx_layers
                if shared is not None and mlx_slot:
                    sh = shared[self.cold_ring.slot_of[i]]
                    assert self.mlx is not None  # the ring is shared only on the MLX device
                    # MEM lands in the slot as bf16 too; a 12-bit record and a GGUF type's blocks are read as stored
                    if it.p12 is not None:
                        m.mx = self.mlx.weight_slot_packed(sh, it.at, it.p12, (rows, cols))
                    elif it.kind is SlotKind.QUANT:
                        m.mx = self.mlx.weight_stored_slot(sh, it.at, it.nb, self._stored_type(m), (rows, cols))
                    else:
                        m.mx = self.mlx.weight_slot(sh, it.at, it.nb, (rows, cols))
                m.packed = m.quant = None
                if it.kind not in (SlotKind.P12, SlotKind.QUANT):
                    m.weight = torch.nn.Parameter(region.view(torch.bfloat16).view(rows, cols), requires_grad=False)
                    continue
                # a bf16 view bound earlier (the ring before the pack was opened) would hold that ring's slot: the
                # shape stays (the host path reads it), over no memory
                m.weight = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.bfloat16).expand(rows, cols), requires_grad=False
                )
                if it.kind is SlotKind.QUANT:
                    if not mlx_slot:  # the native kernels read the slot's blocks; an MLX layer's its view (above)
                        m.quant = QuantWeight.of(region, self._stored_type(m), rows, cols)
                        m.scratch = self.scratch
                    continue
                e = it.p12
                assert e is not None  # a P12 item carries its store entry
                a1 = e["lo"]
                a2 = a1 + e["hi4"] + e["pad"]
                a3 = a2 + 4 * e["esc"]
                esc_idx = region[a2:a3].view(torch.int32) if e["esc"] else torch.zeros(0, dtype=torch.int32)
                esc_val = region[a3:] if e["esc"] else torch.zeros(0, dtype=torch.uint8)
                m.packed = (
                    region[:a1],
                    region[a1 : a1 + e["hi4"]],
                    torch.tensor(e["table"], dtype=torch.uint8),
                    esc_idx,
                    esc_val,
                    int(e["esc"]),
                )
        for i in recipes:
            if self.cold_ring.shared is not None and i in self.mlx_layers:
                self._mlx_fuse(self.host[i])
            self._bind_tags(i, self.host[i])
        self.cold_ring.bytes_per_pass = sum(sizes.values())
        kinds = sorted({it.kind for items in recipes.values() for it in items})
        self.log(
            f"[stream] cold tier: {len(order)} layers {order[:6]}{'...' if len(order) > 6 else ''} x "
            f"{self.cold_ring.bytes_per_pass / 2**30:.2f} GB per pass ({'/'.join(kinds)}), {R} slots x "
            f"{slot_bytes / 2**30:.2f} GB, reader {'natives (unbuffered)' if Native.read_direct else 'buffered fallback (through the page cache)'}"
        )

    def _cold_read(self, path: str, off: int, nb: int, dst: torch.Tensor) -> None:
        if Native.read_direct is not None:
            Native.read_direct(path, off, nb, dst, self.cold_chunk)
            return
        fh = self.cold_ring.fh.get(path)
        if fh is None:
            fh = self.cold_ring.fh[path] = open(path, "rb", buffering=0)
        fh.seek(off)
        view = memoryview(dst.numpy())
        got = 0
        while got < nb:
            n = fh.readinto(view[got:])
            if not n:
                raise RuntimeError(f"[cold] short read: {path} @ {off}+{got} of {nb}")
            got += n

    def _cold_cast(self, i: int, slot: torch.Tensor) -> None:
        """layer `i`'s floats read as stored rewritten as bf16 where they landed, once all its reads are in"""
        for it in self.cold_ring.recipe[i]:
            if it.cast is not None:
                bf16_in_place(slot[it.at : it.at + it.cast[0]], it.cast[1])

    def _cold_start(self, n_layers: int) -> None:
        """The cold reader as a ring across passes: one thread reads the cold layers into their slots as they free
        and goes on into the next pass (the layer order never depends on the tokens). A pass over the same
        layers finds it running; a different shape restarts it. A slot still holding its layer is not re-read."""
        order = [i for i in sorted(self.cold) if i < n_layers]
        if not order:
            return
        th = self.cold_ring.thread
        if th is not None and th.is_alive() and self.cold_ring.order == order:
            return
        self._cold_stop()
        self.cold_ring.order = order
        self.cold_ring.ready = {i: threading.Event() for i in order}
        self.cold_ring.free = [threading.Event() for _ in self.cold_ring.slots]
        for ev in self.cold_ring.free:
            ev.set()
        self.cold_ring.holds = [-1] * len(self.cold_ring.slots)
        ready, free, holds = self.cold_ring.ready, self.cold_ring.free, self.cold_ring.holds

        stat: dict[Any, Any] = {}

        self.cold_ring.abort = False
        sched = getattr(self, "scheduler", None)
        # one read per tensor, so each stays a sequential stream and the Route's readers supply the concurrency;
        # no route.disk(): the profile's depth is for expert-sized reads, and whole layers at that depth slam a HDD
        route = (
            sched if (sched is not None and hasattr(sched, "disk_read") and Native.read_direct is not None) else None
        )

        def submit(i: int, slot: torch.Tensor) -> list[Any]:
            assert route is not None  # submit runs only where the Route reads the cold tier
            futs = []
            for it in self.cold_ring.recipe[i]:
                if it.path is None:
                    self._cold_fill(it, slot)
                    continue
                futs.append(
                    route.disk_read(
                        it.path,
                        it.off,
                        it.read_nb,
                        slot[it.at : it.at + it.read_nb],
                        route.DISK_AHEAD,
                        key=("cold", i),
                        chunk=self.cold_chunk,
                        depth=1,
                    )
                )
                self.cold_ring.bytes += it.read_nb
            return futs

        def land(i: int, s: int, t_sub: float, futs: list[Any], nbytes: int) -> None:
            for f in futs:
                f.result()
            self._cold_cast(i, self.cold_ring.slots[s])
            stat[i] = {"t0": t_sub, "t1": time.time(), "bytes": nbytes}
            holds[s] = i
            ready[i].set()

        def run() -> None:
            queued: list[tuple[int, int, float, list[Any], int]] = []
            try:
                while not self.cold_ring.abort:
                    for i in order:
                        s = self.cold_ring.slot_of[i]
                        if not free[s].is_set():
                            # the slot frees when the pass is done with its layer, and the pass may be waiting on
                            # a layer whose reads are queued here: those land first, or the two wait on each other
                            while queued:
                                land(*queued.pop(0))
                        free[s].wait()
                        if self.cold_ring.abort:
                            return
                        free[s].clear()
                        if holds[s] == i:
                            ready[i].set()
                            continue
                        slot = self.cold_ring.slots[s]
                        a = time.time()
                        if route is None:
                            nbytes = 0
                            for it in self.cold_ring.recipe[i]:
                                if it.path is None:
                                    self._cold_fill(it, slot)
                                else:
                                    self._cold_read(it.path, it.off, it.read_nb, slot[it.at : it.at + it.read_nb])
                                self.cold_ring.bytes += it.read_nb
                                nbytes += it.read_nb
                            self._cold_cast(i, slot)
                            stat[i] = {"t0": a, "t1": time.time(), "bytes": nbytes}
                            holds[s] = i
                            ready[i].set()
                            continue
                        queued.append((i, s, a, submit(i, slot), sum(it.read_nb for it in self.cold_ring.recipe[i])))
                        while len(queued) > 1:
                            land(*queued.pop(0))
                    while queued:
                        land(*queued.pop(0))
            except CancelledError:
                return

        t = threading.Thread(target=run, daemon=True, name="cold-reader")
        t.start()
        self.cold_ring.thread = t

    def _cold_stop(self) -> None:
        """stop the ring and wait for its read in flight (a pass of another shape, or the close)"""
        th = self.cold_ring.thread
        if th is None or not th.is_alive():
            return
        self.cold_ring.abort = True
        sched = getattr(self, "scheduler", None)
        if sched is not None and hasattr(sched, "disk_drop"):
            for i in self.cold_ring.order:
                sched.disk_drop(("cold", i))
        for ev in self.cold_ring.free:
            ev.set()
        th.join(timeout=60)
        if th.is_alive():
            # still in a read into a ring slot after a minute: the ring is not reused or let go while it writes -
            # the thread stays the ring's, and the pass (or the close, which tries again) is told
            raise RuntimeError("[stream] the cold ring's reader is still in a read after 60 s; its slots stay held")
        self.cold_ring.thread = None

    def _cold_wait(self, i: int) -> None:
        t0 = time.time()
        ev = self.cold_ring.ready[i]
        while not ev.wait(0.5):
            if self.abort.is_set():  # a close() withdrew the ring's reads: nothing will set the event
                raise RuntimeError(f"[stream] the decode was stopped while waiting for layer {i}")
        w = time.time() - t0
        self.cold_ring.wait_s += w
        if self.mlx is not None:
            for m in self.host[i].modules():
                if isinstance(m, _HostLinear) and m.mx is not None:
                    m.mx.invalidate()

    def _cold_release(self, i: int) -> None:
        if self.mlx is not None:
            for m in self.host[i].modules():
                if isinstance(m, _HostLinear) and m.mx is not None:
                    m.mx.drop()
        self.cold_ring.ready[i].clear()
        self.cold_ring.free[self.cold_ring.slot_of[i]].set()

    def _rebind_warm(self, i: int) -> None:
        """layer `i` back on the store it came from: every linear's weight the checkpoint's mapped bytes again, its
        packed form beside it under the 12-bit store, a GGUF type its native gemv takes on the file's blocks again,
        no slot of the ring behind it"""
        layer = self.host[i]
        on_mlx = self.mlx is not None and i in self.mlx_layers
        lins = {n + ".weight": m for n, m in layer.named_modules() if isinstance(m, _HostLinear) and m.key}
        stored = self._host_stored_keys(i, lins)
        for m in lins.values():
            m.packed = m.quant = None
            if m.key not in stored:
                m.weight = torch.nn.Parameter(self._get(m.key or ""), requires_grad=False)
            if on_mlx:
                m.mx = None  # a view of a ring slot, which the rebuilt ring fills with another layer
        self._bind_host_stored(i, layer)
        if on_mlx:
            self._bind_mlx_resident(layer)
            self._mlx_fuse(layer)
        if getattr(self, "_packed", None):
            self._bind_host_packed_layer(layer)
        self._bind_tags(i, layer)

    def _next_template(self, lt: str) -> Any:
        k = self._toggle[lt]
        self._toggle[lt] = 1 - k if len(self.templates[lt]) > 1 else 0
        return self.templates[lt][k]

    def _retarget(self, module: Any, i: int) -> Any:
        """Point a template or shadow at layer `i`: every cache index its modules hold and, for a mixture of experts,
        where the expert store reads its experts from. Every submodule that carries a `layer_idx` is one: beside the
        attention's own, Qwen4's sparse attention indexer and its PLE and n-gram convolutions each keep theirs, and a
        template built from layer 3 and run as layer 11 wrote layer 11's indexer keys into layer 3's cache."""
        for m in module.modules():
            if isinstance(getattr(m, "layer_idx", None), int):
                m.layer_idx = i
            if isinstance(m, _Experts) and m.layer != i:
                m.layer = i
                m.base = f"{self.prefix}layers.{i}.mlp.experts."
                m.gate_up = m.down = None
            if isinstance(m, _NGramRows):
                # a layer's own lookup table read from the checkpoint by name (Qwen4's PLE n-grams): layer i's, and
                # the shards opened for the layer it was built from dropped
                at = f"{self.prefix}layers."
                if m.base.startswith(at):
                    base = f"{at}{i}.{m.base[len(at) :].partition('.')[2]}"
                    if base != m.base:
                        m.base, m.shards, m.f8 = base, None, None
        return module

    def _structure(self, module: Any) -> tuple[Any, ...]:
        """a layer module's parameters and buffers as (name, shape): what a float32 shadow must match to run it"""
        return tuple((name, tuple(t.shape), is_buf) for name, t, is_buf in self._named_tensors(module))

    def _upcast(self, lt: str, tmpl: Any, i: int) -> Any:
        """layer `i`'s bf16 weights, and its buffers (a layer's own values), into the float32 shadow of its
        structure, copied by name"""
        sh = self.shadow[lt][self._structure(tmpl)]
        src = {name: t for name, t, _ in self._named_tensors(tmpl)}
        for name, t, _ in self._named_tensors(sh):
            t.data.copy_(src[name].data)
        return self._retarget(sh, i)

    def _host_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src)

    def _start_prefetch(self, i: int, tmpl: Any) -> None:
        ev = torch.cuda.Event()
        gate = torch.cuda.Event()
        gate.record(torch.cuda.current_stream(self.dev))
        lt = self.layer_types[i]
        stage = self._staging[lt]
        base = f"{self.prefix}layers.{i}."
        stats: dict[str, Any] = {"t": 0.0}

        pinned = self.pinned.get(i)
        prev_dma = self._dma_done.get(lt)

        def work() -> None:
            try:
                copy()
            except BaseException as e:  # a thread's exception is otherwise lost: `_wait_prefetch` re-raises it
                stats["err"] = e

        def copy() -> None:
            t0 = time.time()
            b0 = self.bytes_streamed
            if prev_dma is not None:
                prev_dma.synchronize()
            pstage: Any = getattr(self, "_pstaging", {}).get(lt)
            order = sorted(tmpl.named_parameters(), key=lambda kv: -kv[1].numel())
            with torch.cuda.stream(self._copy_stream):
                self._copy_stream.wait_event(gate)
                for name, p in order:
                    if pinned is not None:
                        s = pinned[name]
                        self.bytes_streamed += s.numel() * s.element_size()
                        p.data.copy_(s, non_blocking=True)
                        continue
                    pk = self._get_packed(base + name) if pstage is not None else None
                    if pk is not None:
                        blob, tbl, e = pk
                        if tuple(e["shape"]) != tuple(p.shape) or p.dtype != torch.bfloat16:
                            raise RuntimeError(f"[stream] packed drift at layer {i} {name}")
                        nb = blob.numel()
                        buf = pstage[name]
                        self._host_copy(buf[:nb], blob)
                        d = buf[:nb].to(self.dev, non_blocking=True)
                        a1 = e["lo"]
                        a2 = a1 + e["hi4"] + e["pad"]
                        a3 = a2 + 4 * e["esc"]
                        esc_idx = d[a2:a3].view(torch.int32) if e["esc"] else None
                        esc_val = d[a3:nb] if e["esc"] else None
                        unpack_bf16(
                            d[:a1],
                            d[a1:a2],
                            tbl.to(self.dev, non_blocking=True),
                            e["n"],
                            tuple(e["shape"]),
                            esc_idx,
                            esc_val,
                            out=p.data,
                        )
                        continue
                    t = self._get(base + name)
                    if p.shape != t.shape or p.dtype != t.dtype:
                        raise RuntimeError(f"[stream] drift at layer {i} {name}")
                    s = stage[name]
                    self._host_copy(s, t)
                    p.data.copy_(s, non_blocking=True)
                ev.record(self._copy_stream)
            stats["t"] = time.time() - t0
            stats["t0"], stats["t1"], stats["bytes"] = t0, time.time(), self.bytes_streamed - b0

        th = self._thread_mod.Thread(target=work, daemon=True)
        th.start()
        self._dma_done[lt] = ev
        self._pending = (i, tmpl, ev, th, stats)

    def _wait_prefetch(self, i: int) -> Any:
        j, tmpl, ev, th, stats = self._pending
        if j != i:
            raise RuntimeError(f"[stream] prefetch order broken: wanted {i}, pending {j}")
        t0 = time.time()
        th.join()
        w = time.time() - t0
        self.wait_s += w
        if "err" in stats:
            self._pending = None
            raise RuntimeError(f"[stream] the prefetch of layer {i} failed") from stats["err"]
        torch.cuda.current_stream(self.dev).wait_event(ev)
        self.load_s += stats["t"]
        self._pending = None
        return self._retarget(tmpl, i)

    def _sync(self) -> None:
        if self.dev.type == Device.CUDA:
            torch.cuda.synchronize(self.dev)

    def _flush_events(self) -> None:
        if self._events:
            self._events[-1][1].synchronize()
            ms = sum(a.elapsed_time(b) for a, b in self._events)
            self.compute_s += ms / 1000.0
            self._last_card_ms = ms
            prev = getattr(self, "_card_ms_min", None)
            self._card_ms_min = ms if prev is None else min(prev, ms)
            self._events = []

    def _new_layer(self, idx: int, base: str | None = None, at: int | None = None) -> Any:
        """a layer of trunk layer `idx`'s kind; `base` and `at` one whose tensors sit under another prefix (a
        drafting head's `mtp.layers.0.`), its experts the store's layer `at`"""
        with self._meta:
            layer = self.fam.layer(self.cfg, idx).eval()
        layer = self.fam.shape_layer(self, layer, idx if at is None else at, base=base)
        # a card layer's torch path row-invariant: its matmuls and norms at the fixed shape for a small pass, so a
        # step and a verify pass over the same token agree to the bit (fixed_rows.py); a no-op off the card
        fix_linears(layer)
        fix_rows_cls(self.fam.norm)
        base = f"{self.prefix}layers.{idx}." if base is None else base
        for name, _b in layer.named_buffers():
            if base + name not in self.weight_map:
                raise RuntimeError(f"[stream] layer has a buffer {name!r} the loader would not fill")
        return layer

    def _shard(self, shard: str) -> Shard:
        if shard not in self._maps:
            gg = getattr(self, "gguf", None)
            if gg is not None and shard == gg.file:
                # the GGUF file: no mmap of our own (the package holds it), a header the size readers take
                self._maps[shard] = (None, self._gguf_hdr, 0)
                return self._maps[shard]
            with open(os.path.join(self.dir, shard), "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = struct.unpack("<Q", mm[:8])[0]
            hdr = json.loads(mm[8 : 8 + n].decode("utf-8"))
            hdr.pop("__metadata__", None)
            self._maps[shard] = (mm, hdr, 8 + n)
        return self._maps[shard]

    def open_packed(self) -> None:
        """Bind the 12-bit model's packed layers: the entries of its own shards (`btb.pack12.entries`)."""
        self._packed = entries(self.dir)
        self._packed_maps = {}
        self.log(f"[stream] packed transport: {len(self._packed)} tensors from {self.dir}")
        if self.prefetch:
            self._pstaging = {}
            for lt, tmpls in self.templates.items():
                sizes: dict[int, int] = {}
                for i in range(self.L):
                    if self.layer_types[i] != lt:
                        continue
                    base = f"{self.prefix}layers.{i}."
                    for name, _ in tmpls[0].named_parameters():
                        e = self._packed.get(base + name)
                        if e is None or e["raw"]:
                            continue
                        sizes[name] = max(sizes.get(name, 0), e["lo"] + e["hi4"] + e["pad"] + 5 * e["esc"])
                self._pstaging[lt] = {
                    name: torch.empty(nb, dtype=torch.uint8).pin_memory() for name, nb in sizes.items()
                }

    def _packed_blob(self, shard: str) -> Any:
        if shard not in self._packed_maps:
            with open(os.path.join(self.dir, shard), "rb") as f:
                self._packed_maps[shard] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._packed_maps[shard]

    def _get_packed(self, key: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
        import warnings

        e = getattr(self, "_packed", {}).get(key)
        if e is None or e["raw"]:
            return None
        mm = self._packed_blob(e["shard"])
        nb = e["lo"] + e["hi4"] + e["pad"] + 5 * e["esc"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blob = torch.frombuffer(mm, dtype=torch.uint8, count=nb, offset=e["off"])
        self.bytes_streamed += nb
        return blob, torch.tensor(e["table"], dtype=torch.uint8), e

    def _gguf_binds_packed(self, name: str) -> bool:
        """Whether the GGUF tensor `name` binds to its type's own MLX kernel (mlx_forward.py's `_bind_gguf_stored`)
        instead of a plain bf16 slot, so `_get` can skip the bf16 dequant the binder would discard for its own
        raw-byte read: the binder's gate, nothing read. A synthesized name (gpt-oss's interleaved expert biases) is
        no file tensor and reads for real."""
        assert self.gguf is not None
        return self.mlx is not None and self.gguf_packed and self.gguf.storable(name)

    def _gguf_layer_index(self, key: str) -> int | None:
        """The layer index a weight key names, or None for a non-layer tensor (head, embed, norm, ...)."""
        pre = self.prefix + "layers."
        if not key.startswith(pre):
            return None
        head, _, _ = key[len(pre) :].partition(".")
        return int(head) if head.isdigit() else None

    def _get(self, key: str, gguf_shortcut: bool = False, stored: bool = False) -> torch.Tensor:
        """a checkpoint tensor as the engine holds it (`_held`), or with `stored` at the checkpoint's own
        precision - for a caller that widens it to float32 itself, so an fp32 checkpoint's values reach it whole"""
        import warnings

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"[stream] tensor {key!r} is not in the index")
        mm, hdr, base = self._shard(shard)
        if mm is None:  # the GGUF file: dequantized on read by the gguf package
            assert self.gguf is not None  # a shard with no mmap is the GGUF's own tensor
            name = self._gguf_names[key]
            i = self._gguf_layer_index(key)
            # a layer bound to MLX always runs its linears through `_bind_mlx_resident` right after loading
            # (`_make_host_layer`'s `i in self.mlx_layers and i not in self.cold`); `gguf_shortcut` lets a caller that
            # knows its own tensor rebinds immediately (the resident head) claim the same skip
            resolved = gguf_shortcut or (i is not None and i in self.mlx_layers and i not in getattr(self, "cold", ()))
            if resolved and self._gguf_binds_packed(name):
                shape = tuple(int(x) for x in hdr[key]["shape"])
                return torch.zeros((), dtype=torch.bfloat16).expand(*shape)
            t = self.gguf.get(name)
            self.bytes_streamed += t.numel() * t.element_size()
            return t
        info = hdr[key]
        dt = self.ST_DTYPES[info["dtype"]]
        a, _ = info["data_offsets"]
        shape = tuple(int(x) for x in info["shape"])
        n = 1
        for d in shape:
            n *= d
        if n == 0:
            t = torch.empty(shape, dtype=dt)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = torch.frombuffer(mm, dtype=dt, count=n, offset=base + a).view(shape)
        self.bytes_streamed += n * t.element_size()
        if dt == fp8.E4M3:
            # widened blockwise by its scale to the bf16 every path holds (`fp8.held`), in f32 for a caller that
            # widens it itself
            self.fp8_widened = True
            w = fp8.held(t, self._fp8_scale(key))
            return w.float() if stored else w
        return t if stored else self._held(t)

    @staticmethod
    def _set_param(module: Any, dotted: str, t: torch.Tensor, buffer: bool = False) -> None:
        *path, leaf = dotted.split(".")
        m = module
        for p in path:
            m = getattr(m, p)
        setattr(m, leaf, t if buffer else torch.nn.Parameter(t, requires_grad=False))

    def _adopt(self, module: Any, dotted: str, t: torch.Tensor, writable: bool = False, buffer: bool = False) -> None:
        if self.dev.type != Device.CPU:
            t = t.to(self.dev)
        elif writable:
            t = t.clone()
        self._set_param(module, dotted, t, buffer=buffer)

    def _card_stores(self) -> bool:
        """whether the card's kernels multiply a GGUF's stored types here: under `gguf_packed`, with the kernels
        loaded and a bf16 compute (they take bf16 rows; a float32 one keeps the dequantized weights)"""
        if self.gguf is None or not self.gguf_packed or self.dev.type != Device.CUDA or self.mlx is not None:
            return False
        return self.compute_dtype in (None, torch.bfloat16) and Native.card_kernels() is not None

    def _card_stored_keys(self, i: int, linears: Iterable[str]) -> dict[str, str]:
        """card layer `i`'s linear weights (`linears`, named in the layer) that the card's kernels multiply in
        their GGUF blocks (`_card_stores`), by checkpoint key -> file tensor"""
        gg = self.gguf
        if gg is None or not self._card_stores():
            return {}
        base = f"{self.prefix}layers.{i}."
        out: dict[str, str] = {}
        for n in linears:
            name = self._gguf_names.get(base + n)
            if name is not None and gg.storable(name):
                out[base + n] = name
        return out

    def _make_head(self) -> RowLinear | CardStored:
        """the resident head, built and rebuilt (the VRAM policy's shed and regrow) the one way: its GGUF blocks on
        the card where the card's kernels take its type (a Q*_K_M file keeps its output in Q6_K), else a bf16
        linear whose small passes run at the fixed shape, as the card layers' (fixed_rows.py)"""
        gg = self.gguf
        name = self._gguf_names.get(self.head_key) if gg is not None else None
        if gg is not None and name is not None and self._card_stores() and gg.storable(name):
            return CardStored(gg.stored(name).to(self.dev), self.head_key, scratch=self.scratch)
        with self._meta:
            head = RowLinear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        self._adopt(head, "weight", self._get(self.head_key))
        return head

    def _bind_card_stored(self, tmpl: Any, stored: dict[str, str], base: str) -> None:
        """each linear of a card layer `stored` names replaced by its GGUF blocks on the card (`CardStored`), the
        one-element stand-in the load adopted going with the module"""
        assert self.gguf is not None  # stored keys are a GGUF's
        for key, name in stored.items():
            *path, leaf = key[len(base) : -len(".weight")].split(".")
            parent = tmpl
            for p in path:
                parent = getattr(parent, p)
            bias = getattr(getattr(parent, leaf), "bias", None)
            w = self.gguf.stored(name).to(self.dev)
            setattr(parent, leaf, CardStored(w, key, scratch=self.scratch, bias=None if bias is None else bias.data))

    def _load_layer(self, i: int, tmpl: Any, first: bool = False) -> None:
        t0 = time.time()
        base = f"{self.prefix}layers.{i}."
        seen = 0
        # a resident layer's GGUF linears the card multiplies as stored: a one-element stand-in adopted in place of
        # the bf16 dequant, then the module swapped for the blocks on the card - nothing expanded is read or moved
        stored = (
            self._card_stored_keys(
                i, [n + ".weight" for n, m in tmpl.named_modules() if isinstance(m, torch.nn.Linear)]
            )
            if first
            else {}
        )
        for name, p, is_buf in self._named_tensors(tmpl):
            t = torch.zeros(1, dtype=torch.bfloat16) if base + name in stored else self._get(base + name)
            if first:
                self._adopt(tmpl, name, t, writable=True, buffer=is_buf)
            else:
                if p.shape != t.shape or p.dtype != t.dtype:
                    raise RuntimeError(
                        f"[stream] drift at layer {i} {name}: {tuple(p.shape)}/{p.dtype} vs {tuple(t.shape)}/{t.dtype}"
                    )
                p.data.copy_(t)
            seen += 1
        extra = [k for k in self.weight_map if k.startswith(base) and self.fam.dense_key(k)]
        if len(extra) != seen:
            raise RuntimeError(f"[stream] layer {i}: template consumed {seen} tensors, the index holds {len(extra)}")
        self._retarget(tmpl, i)
        if first:
            tmpl.to(self.dev)
            if stored:
                self._bind_card_stored(tmpl, stored, base)
            self._bind_tags(i, tmpl)
        else:
            self.weight_tags.pop(i, None)  # a template streamed in: bf16 slots
        self._sync()
        self.load_s += time.time() - t0


def _rows_bytes(cache: Any, i: int, dev: torch.device) -> int:
    """the bytes layer i's rows in `cache` take once moved to `dev` (`_cache_to`): nothing for those there already"""
    if cache is None or i >= len(cache.layers):
        return 0
    cl = cache.layers[i]
    if getattr(cl, "paged", False):
        return 0  # the pool's rows, moved once for every conversation and granted there (`KvPool.rehome`)
    if isinstance(cl, CardRowsLayer):
        return 0  # the arena's rows: a fork's layer made of them grants its own buffer (`to_fork`)
    if isinstance(cl, ForkLayer):
        held = [*(cl._kv or (cl._pk, cl._pv))]
        if isinstance(cl, ForkIndexedLayer):
            held.append(cl._ti if cl._ti is not None else cl._pi)
    else:
        held = [getattr(cl, a, None) for a in ("keys", "values", "indexer_keys")]
        for a in ("conv_states", "recurrent_states"):
            d = getattr(cl, a, None)
            held += list(d.values()) if isinstance(d, dict) else [d]
    return sum(
        t.numel() * t.element_size()
        for t in held
        if isinstance(t, torch.Tensor) and t.device.type != dev.type and t.device.type != "meta"
    )
