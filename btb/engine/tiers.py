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
from collections.abc import Sequence
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .. import mlx as mlxdev
from ..gguf import GROUP
from ..hf import AFFINE_TYPES
from ..kinds import Json, LayerKind, Tier
from ..options import Device
from ..pack12 import entries, unpack_bf16
from ..sysinfo import process_working_set_bytes
from .cache import ForkLayer, GrowLayer
from .host import _Experts, _HostLinear, bf16_in_place, copy_bytes
from .native import Native
from .state import DRAFT_VOCAB, _State

if TYPE_CHECKING:
    from ..mlx import Shared

_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}
RAM_BPS = 50 * 2**30  # the read bandwidth the plan prices a RAM tier at, weights and cache alike
DRIVE_BPS = int(3.4 * 2**30)  # the cold tier's rate until the drive is probed (scheduler.DriveBenchmark)


@dataclass
class ColdRing:
    """The cold layers' ring, built by `_cold_start` and torn down by `_cold_stop`; its slots are shared with
    MLX where a cold layer runs there."""

    fh: dict[str, Any] = field(default_factory=dict)
    shared: list[Shared] | None = None
    slots: list[torch.Tensor] = field(default_factory=list)
    slot_of: dict[int, int] = field(default_factory=dict)
    recipe: Any = None
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
            if k.startswith(base) and self._dense_key(k):
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
        kv_layer = {
            i: 0 if self.layer_types[i] == LayerKind.LINEAR else 2 * hk * hd * (4 if fp32 else 2) * kv_rows
            for i in range(L)
        }
        # the drafter plus its head slice, both at bf16
        drafter_b = (
            sum(self._get(k).numel() * 2 for k in self.weight_map if k.startswith("mtp."))
            + min(DRAFT_VOCAB, int(cfg.vocab_size)) * int(cfg.hidden_size) * 2
            if drafter
            else 0
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
        shadow_b = sum(2 * big[lt] for lt in types) if (fp32 and not resident_fp32) else 0
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
            b = bf16[i] * (2 if (fp32 and resident_fp32) else 1) + (0 if kv_host else kv_layer[i])
            if vram >= b:
                resident.append(i)
                vram -= b
            else:
                break
        resident = sorted(resident)
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
        kv_card_b = 0 if kv_host else sum(kv_layer[i] for i in resident)
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
        kv_read_ms = kv_host_b / RAM_BPS * 1e3  # the host's attention over the whole cache, at the full context
        return {
            "head_on_card": head_on_card,
            "drafter_on_card": drafter_on_card,
            "resident": resident,
            "host": warm + cold,
            "cold": cold,
            "warm": warm,
            "prefill_card": prefill_card,
            "bytes": {
                "vram_layers": sum(bf16[i] for i in resident) * (2 if (fp32 and resident_fp32) else 1),
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
            "predicted_ms_per_token": max(warm_ms, cold_ms)
            + (3.0 * len(resident))
            + (50.0 if not head_on_card else 5.0),
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
        drafter_b = 0
        for k in mtp:
            _mm, hdr, _ = self._shard(self.weight_map[k])
            n = 1
            for d in hdr[k]["shape"]:
                n *= int(d)
            drafter_b += n * 2
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
                "vram_reserved_gb": (torch.cuda.max_memory_reserved(dev) / 2**30) if cuda else 0.0,
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

    def _head_host(self) -> Any:
        if self.head_host is None:
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
        return dt == torch.float16 or (self.held_cast and dt.is_floating_point and dt != torch.bfloat16)

    def _stored_span(self, key: str) -> tuple[str, int, tuple[int, torch.dtype]] | None:
        """where a tensor `_held` casts sits on the drive as stored: (file path, offset, (bytes, dtype)); None
        for any other"""
        shard = self.weight_map[key]
        _mm, hdr, base = self._shard(shard)
        info = hdr[key]
        if not self._cast_on_read(info):
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
            if k.startswith(base) and self._dense_key(k):
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

    def bind_cpu_gemm(self) -> int:
        """The CPU tier's host linears in shared torch/MLX memory: each layer's weights read once into one MLX byte
        buffer, the torch weight a view of it, `cpu_gemm` a bf16 view for the prefill's GEMM. Cold and packed
        layers are left alone. Returns the bytes bound."""
        from concurrent.futures import ThreadPoolExecutor

        rd = Native.read_direct
        total = 0
        pool = getattr(self, "_mlx_pool", None)
        if pool is None:
            pool = self._mlx_pool = ThreadPoolExecutor(max_workers=4)
        groups = [
            [m for m in layer.modules() if isinstance(m, _HostLinear)]
            for i, layer in self.host.items()
            if i not in self.cold
        ]
        if self.head_host is not None:
            groups.append([self.head_host])
        for lins in groups:
            lins = [
                m
                for m in lins
                if m.mx is None
                and m.packed is None
                and m.cpu_gemm is None
                and m.key in self.weight_map
                and m.weight.dtype == torch.bfloat16
            ]
            if not lins:
                continue
            items, cur = self._layer_items(lins)
            sh, base = self._shared_ahead(cur)
            chunk = getattr(self, "cold_chunk", 16 << 20)

            def read(it: Any) -> Any:
                # consumed by pool.map within this iteration: `sh`, `base` and `chunk` are this layer's (B023)
                m, path, off, nb, so = it
                if rd is not None and path is not None:
                    rd(path, off, nb, sh.torch[base + so : base + so + nb], chunk)  # noqa: B023
                else:
                    copy_bytes(sh.torch[base + so : base + so + nb], m.weight.data)  # noqa: B023
                return nb

            for nb in pool.map(read, items):
                self.bytes_streamed += nb
            for m, _path, _off, nb, so in items:
                shape = tuple(m.weight.shape)
                m.weight = torch.nn.Parameter(sh.view_torch(base + so, nb, torch.bfloat16, shape), requires_grad=False)
                m.cpu_gemm = sh.view_mx(base + so, nb, mlxdev.mx().bfloat16, shape)
                m._cpu_shared = sh
            # the views evaluated here, on the loading thread: MLX keeps a lazy op's stream per thread, and a
            # server's first prefill runs on a request thread, where the loader's stream does not exist
            mlxdev.mx().eval(*[m.cpu_gemm for m, *_ in items])
            total += cur
        return total

    def _track(self, cache: Any) -> Any:
        """`cache` among those a layer's move reaches while it lives: an idle session's, a fork's, the running pass's"""
        live = self.__dict__.setdefault("_live_caches", weakref.WeakSet())
        live.add(cache)
        return cache

    def _caches_to(self, i: int, dev: str | torch.device, cache: Any = None) -> None:
        """layer i's rows in every live cache (and `cache`) moved to `dev`, where the layer now runs"""
        live = list(self.__dict__.get("_live_caches", ()))
        for c in {id(c): c for c in [*live, *([cache] if cache is not None else [])]}.values():
            self._cache_to(c, i, dev)

    @staticmethod
    def _cache_to(cache: Any, i: int, dev: str | torch.device) -> None:
        if cache is None or i >= len(cache.layers):
            return
        cl = cache.layers[i]
        if isinstance(cl, ForkLayer):
            cl.to(dev)
            return
        if (
            isinstance(cl, GrowLayer)
            and not cl.shared
            and cl._buf is not None
            and cl._buf[0].device != torch.device(dev)
        ):
            # the rows copied out and the buffer they grew in let go now, not at the cache's next append
            k, v = cl.keys, cl.values
            cl._buf, cl._an = None, None
            cl._set_rows(k, v)
        if isinstance(cl, GrowLayer) and cl.is_initialized and isinstance(cl.keys, torch.Tensor):
            if cl.keys.device != torch.device(dev):
                cl._set_rows(cl.keys.to(dev), cl.values.to(dev))
        for attr in ("keys", "values", "conv_states", "recurrent_states", "indexer_keys"):
            t = getattr(cl, attr, None)
            if isinstance(t, dict):
                for k, v in t.items():
                    if isinstance(v, torch.Tensor) and v.device != torch.device(dev):
                        t[k] = v.to(dev)
            elif isinstance(t, torch.Tensor) and t.device != torch.device(dev):
                setattr(cl, attr, t.to(dev))

    def _layer_bytes(self, i: int) -> int:
        base = f"{self.prefix}layers.{i}."
        n = 0
        for k in self.weight_map:
            if k.startswith(base) and self._dense_key(k):
                _mm, hdr, _ = self._shard(self.weight_map[k])
                n += self._held_nbytes(hdr[k])
        return n

    def _regrow_bytes(self) -> int:
        if not self._shed:
            return 0
        what = self._shed[-1]
        if what == "head":
            return self.cfg.vocab_size * self.cfg.hidden_size * 2
        if what == "drafter":
            return sum(self._get(k).numel() * 2 for k in self.weight_map if k.startswith("mtp."))
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        return self._layer_bytes(int(what.split()[1])) * (2 if (fp32 and self.resident_fp32) else 1)

    def _realloc_bytes(self) -> int:
        n = 0
        if self.resident_head and self.head is not None:
            n += self.head.weight.numel() * self.head.weight.element_size()
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            n += sum(self._get(k).numel() * 2 for k in self.weight_map if k.startswith("mtp."))
        return n

    def _bind_cold(self) -> None:
        if not self.cold:
            return
        recipes, sizes = {}, {}
        for i in sorted(self.cold):
            items, cur = [], 0
            for m in self.host[i].modules():
                if not (isinstance(m, _HostLinear) and m.key):
                    continue
                pk = getattr(self, "_packed", {}).get(m.key)
                path: str | None
                if pk is not None and not pk["raw"]:
                    path = os.path.join(self.dir, pk["shard"])
                    off, nb, kind, e = int(pk["off"]), pk["lo"] + pk["hi4"] + pk["pad"] + 5 * pk["esc"], "p12", pk
                else:
                    path, off, nb = self._span(m.key)
                    # "mem": the bytes are not on the drive as bf16 (a GGUF tensor of another type): each pass
                    # takes them from `_get`, which reads and dequantizes them
                    kind, e = ("bf16", None) if path is not None else ("mem", m.key)
                    stored = self._stored_span(m.key)
                    if stored is not None:
                        # "cast": a float at another precision, read as stored into its region and rewritten as
                        # bf16 there as it lands (`_cold_cast`); e is (stored bytes, dtype)
                        path, off, e = stored
                        kind = "cast"
                cur = (cur + 63) // 64 * 64
                items.append((m, path, off, nb, cur, kind, e))
                cur += e[0] if kind == "cast" else nb
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
            for m, _path, _off, nb, so, kind, e in items:
                region = slot[so : so + nb]
                rows, cols = m.weight.shape
                if self.cold_ring.shared is not None and i in self.mlx_layers:
                    sh = self.cold_ring.shared[self.cold_ring.slot_of[i]]
                    assert self.mlx is not None  # the ring is shared only on the MLX device
                    # "mem" lands in the slot as bf16 too; only a 12-bit record is read packed
                    m.mx = (
                        self.mlx.weight_slot_packed(sh, so, e, (rows, cols))
                        if kind == "p12"
                        else self.mlx.weight_slot(sh, so, nb, (rows, cols))
                    )
                if kind != "p12":
                    m.weight = torch.nn.Parameter(region.view(torch.bfloat16).view(rows, cols), requires_grad=False)
                    m.packed = None
                else:
                    # a bf16 view bound earlier (the ring before the pack was opened) would hold that ring's slot:
                    # the shape stays (the host path reads it), over no memory
                    m.weight = torch.nn.Parameter(
                        torch.zeros((), dtype=torch.bfloat16).expand(rows, cols), requires_grad=False
                    )
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
        if self.cold_ring.shared is not None:
            for i in recipes:
                if i in self.mlx_layers:
                    self._mlx_fuse(self.host[i])
        self.cold_ring.bytes_per_pass = sum(sizes.values())
        kinds = sorted({it[5] for items in recipes.values() for it in items})
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

    @staticmethod
    def _cold_nb(it: Any) -> int:
        """the bytes a cold recipe item reads: a cast float's as stored"""
        return int(it[6][0]) if it[5] == "cast" else int(it[3])

    def _cold_cast(self, i: int, slot: torch.Tensor) -> None:
        """layer `i`'s floats read as stored rewritten as bf16 where they landed, once all its reads are in"""
        for _m, _path, _off, _nb, so, kind, e in self.cold_ring.recipe[i]:
            if kind == "cast":
                bf16_in_place(slot[so : so + e[0]], e[1])

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
                _m, path, off, nb, so, kind, e = it
                if kind == "mem":
                    copy_bytes(slot[so : so + nb], self._get(e))
                    continue
                nb = self._cold_nb(it)
                futs.append(
                    route.disk_read(
                        path,
                        off,
                        nb,
                        slot[so : so + nb],
                        route.DISK_AHEAD,
                        key=("cold", i),
                        chunk=self.cold_chunk,
                        depth=1,
                    )
                )
                self.cold_ring.bytes += nb
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
                                _m, path, off, _nb, so, kind, e = it
                                nb = self._cold_nb(it)
                                if kind == "mem":
                                    copy_bytes(slot[so : so + nb], self._get(e))
                                else:
                                    self._cold_read(path, off, nb, slot[so : so + nb])
                                self.cold_ring.bytes += nb
                                nbytes += nb
                            self._cold_cast(i, slot)
                            stat[i] = {"t0": a, "t1": time.time(), "bytes": nbytes}
                            holds[s] = i
                            ready[i].set()
                            continue
                        queued.append((i, s, a, submit(i, slot), sum(map(self._cold_nb, self.cold_ring.recipe[i]))))
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
        """layer `i` back on the store it came from: every linear's weight the checkpoint's mapped bytes again,
        its packed form beside it under the 12-bit store, no slot of the ring behind it"""
        layer = self.host[i]
        for m in layer.modules():
            if isinstance(m, _HostLinear) and m.key:
                m.weight = torch.nn.Parameter(self._get(m.key), requires_grad=False)
                m.packed = None
        if getattr(self, "_packed", None):
            self._bind_host_packed_layer(layer)

    def _next_template(self, lt: str) -> Any:
        k = self._toggle[lt]
        self._toggle[lt] = 1 - k if len(self.templates[lt]) > 1 else 0
        return self.templates[lt][k]

    def _retarget(self, module: Any, i: int) -> Any:
        """Point a template or shadow at layer `i`: the attention's cache index and, for a mixture of experts, where
        the expert store reads its experts from."""
        if hasattr(module, "linear_attn"):
            module.linear_attn.layer_idx = i
        if hasattr(module, "self_attn"):
            module.self_attn.layer_idx = i
        for m in module.modules():
            if isinstance(m, _Experts) and m.layer != i:
                m.layer = i
                m.base = f"{self.prefix}layers.{i}.mlp.experts."
                m.gate_up = m.down = None
        return module

    def _upcast(self, lt: str, tmpl: Any, i: int) -> Any:
        sh = self.shadow[lt]
        for (_, p32), (_, p16) in zip(sh.named_parameters(), tmpl.named_parameters()):
            p32.data.copy_(p16.data)
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

    def _new_layer(self, idx: int) -> Any:
        with self._meta:
            layer = self.fam.layer(self.cfg, idx).eval()
        layer = self._shape_layer(layer, idx)
        base = f"{self.prefix}layers.{idx}."
        for name, _b in layer.named_buffers():
            if base + name not in self.weight_map:
                raise RuntimeError(f"[stream] layer has a buffer {name!r} the loader would not fill")
        return layer

    def _shard(self, shard: str) -> tuple[Any, dict[str, Any], Any]:
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
        """Whether the GGUF tensor `name` binds to a packed MLX kernel (mlx_forward.py's `_bind_gguf_q4k`/
        `_affine`/`_q6k`) instead of a plain bf16 slot, so `_get` can skip the bf16 dequant they would
        immediately discard for their own raw-byte read. The type/shape gate `affine_of` (`gguf.py`) and
        `_bind_gguf_q6k` apply, without `affine_of`'s block decode: that decode is wasted work here since
        `_bind_gguf_affine`/`_bind_gguf_q4k` run it again from the raw bytes right after this returns."""
        if self.mlx is None or not bool(int(getattr(self, "gguf_packed", 1))):
            return False
        assert self.gguf is not None
        t = self.gguf.tensors.get(name)
        if t is None:  # a synthesized name, not a file tensor (gpt-oss's interleaved expert biases): read for real
            return False
        shape = tuple(int(x) for x in reversed(list(t.shape)))
        if len(shape) != 2:
            return False
        kind = t.tensor_type.name
        # own-kernel binders' shape gate; no fallback binder catches a miss
        if kind in (
            "Q6_K",
            "Q5_K",
            "Q3_K",
            "Q2_K",
            "IQ4_XS",
            "IQ3_XXS",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ1_S",
            "IQ3_S",
            "IQ1_M",
        ):
            return shape[1] % 256 == 0
        if kind == "IQ4_NL":
            return shape[1] % 32 == 0
        return kind in AFFINE_TYPES and shape[1] % GROUP == 0

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
            # (`families.py`'s `i in self.mlx_layers and i not in self.cold`); `gguf_shortcut` lets a caller that
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

    def _load_layer(self, i: int, tmpl: Any, first: bool = False) -> None:
        t0 = time.time()
        base = f"{self.prefix}layers.{i}."
        seen = 0
        for name, p, is_buf in self._named_tensors(tmpl):
            t = self._get(base + name)
            if first:
                self._adopt(tmpl, name, t, writable=True, buffer=is_buf)
            else:
                if p.shape != t.shape or p.dtype != t.dtype:
                    raise RuntimeError(
                        f"[stream] drift at layer {i} {name}: {tuple(p.shape)}/{p.dtype} vs {tuple(t.shape)}/{t.dtype}"
                    )
                p.data.copy_(t)
            seen += 1
        extra = [k for k in self.weight_map if k.startswith(base) and self._dense_key(k)]
        if len(extra) != seen:
            raise RuntimeError(f"[stream] layer {i}: template consumed {seen} tensors, the index holds {len(extra)}")
        self._retarget(tmpl, i)
        if first:
            tmpl.to(self.dev)
        self._sync()
        self.load_s += time.time() - t0
