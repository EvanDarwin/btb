# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The device: one owner of where the model's parts live, what memory is free for them, and when placement may
change. Every path that asks "what is resident", "how much is free" or "may this move now" asks here, so the
answers agree."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from ..kinds import LayerTier, Tier
from ..options import BadDevice, DeviceName, check_device
from ..options import Device as DeviceKind
from ..sysinfo import host_free_bytes

_PHYS_FREE: dict[int, tuple[float, int | None]] = {}


def _physical_free_bytes(dev: torch.device) -> int | None:
    """The card's free VRAM across every process, from nvidia-smi (NVML), or None when it cannot be read.
    `torch.cuda.mem_get_info` is a per-process figure on Windows - WDDM virtualizes VRAM, so it reads high while
    another process holds the card - and only NVML sees the true physical free. Cached for a second: this is an
    allocation-path read, not a per-token one, so one subprocess a second is nothing."""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    now = time.monotonic()
    hit = _PHYS_FREE.get(idx)
    if hit is not None and now - hit[0] < 1.0:
        return hit[1]
    free: int | None = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout
        free = int(out.strip().splitlines()[0]) * (1 << 20)  # MiB -> bytes
    except Exception:
        free = None
    _PHYS_FREE[idx] = (now, free)
    return free


def free_bytes(dev: torch.device, margin: int = 0) -> int | None:
    """Memory free on `dev` above `margin`. On a card: the physical free VRAM (nvidia-smi's, so a card shared with
    another process is priced by what is truly free, not the per-process figure WDDM hands `mem_get_info`) plus
    torch's own reserved-but-unallocated pool - blocks the next allocation reuses without reaching the driver, so
    counting them keeps a later epoch from reading the previous epoch's retained pool as "used". On the host: the
    available RAM. None where nothing here can price the device."""
    if dev.type == DeviceKind.CUDA:
        free, _ = torch.cuda.mem_get_info(dev)
        phys = _physical_free_bytes(dev)
        if phys is not None:
            free = min(int(free), phys)  # never trust the per-process view over the true physical free
        reclaimable = torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        return max(0, int(free) + int(reclaimable) - int(margin))
    if dev.type == DeviceKind.CPU:
        return max(0, int(host_free_bytes()) - int(margin))
    return None


def mlx_available() -> bool:
    """Check if MLX is available on this machine"""
    try:
        from .. import mlx as mlxdev

        return mlxdev.available()
    except Exception:
        return False


def mlx_reason() -> str:
    """why MLX is not available here, for an error message"""
    if sys.platform != "darwin":
        return f"it is only available on Apple hardware, but this machine runs {sys.platform}"
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return "the mlx package is not installed (pip install mlx)"
    return "Metal reports no device (an Intel Mac, or a headless session)"


def resolve_device(device: Any) -> DeviceName:
    """The device a load runs on: None (or 'auto') picks the card when one is visible, else MLX on Apple
    silicon, else the CPU; a device named must exist here - 'mlx' off Apple silicon or 'cuda' without a card
    is an OptionError saying why and what to use, never a silent fall to the CPU. 'cuda:N' names a card."""
    # the gate `btb.cpu_only()` closes lives in the package root, which must stay torch-free; read it live
    from .. import CUDA

    d = check_device(device)
    card = CUDA and torch.cuda.is_available()
    if d is None:
        if card:
            return DeviceName(DeviceKind.CUDA)
        return DeviceName(DeviceKind.MLX if (sys.platform == "darwin" and mlx_available()) else DeviceKind.CPU)
    if d.kind is DeviceKind.CPU:
        return d
    if d.kind is DeviceKind.MLX:
        if mlx_available():
            return d
        raise BadDevice(d, f"{mlx_reason()}; --device {DeviceKind.CPU} runs here")
    if not card:
        why = "the CPU was chosen (btb.cpu_only)" if not CUDA else "no CUDA device is visible to torch"
        if CUDA and torch.version.cuda is None:
            why += f" (torch {torch.__version__} is a CPU build)"
        alt = (
            f"{DeviceKind.CPU}, or {DeviceKind.MLX}"
            if (sys.platform == "darwin" and mlx_available())
            else str(DeviceKind.CPU)
        )
        raise BadDevice(d, f"{why}; --device {alt} runs here")
    if d.index is not None:
        n = torch.cuda.device_count()
        if d.index >= n:
            raise BadDevice(
                d,
                f"{n} CUDA device(s) are visible ({DeviceKind.CUDA}:0{f'..{DeviceKind.CUDA}:{n - 1}' if n > 1 else ''}); "
                "CUDA_VISIBLE_DEVICES chooses which cards those are",
            )
    return d


@dataclass(frozen=True)
class Placement:
    """Where every part of the model lives, as a pass sees it: immutable, and stamped with the version of the
    placement it was taken from. The tiers are disjoint (a cold layer is not counted under host, though it
    streams through a host module) so resident + host + cold names every layer once; `mlx` is a second axis
    over the same layers on unified memory."""

    version: int
    resident: tuple[int, ...]
    host: tuple[int, ...]
    cold: tuple[int, ...]
    mlx: tuple[int, ...]
    head: Tier
    drafter: Tier
    kv: Tier

    def tier(self, i: int) -> LayerTier:
        if i in self.resident:
            return LayerTier.RESIDENT
        if i in self.cold:
            return LayerTier.COLD
        if i in self.host:
            return LayerTier.HOST
        return LayerTier.STREAMED


class Device:
    """The engine's placement and memory ledger. Placement is read through `snapshot()`; a pass that must
    not see it change takes it through `hold()`, and a change asked for meanwhile (`request`) waits at the
    boundary where the last holder lets go - a shed no longer moves a layer under a pass that is half way
    through it. Memory is read through `free()`, one arithmetic for the plan, the scheduler and the policy;
    an epoch's KV is `reserve()`d so the policy does not count memory the batch is about to take as free."""

    def __init__(self, sm: Any) -> None:
        self.sm = sm
        self.version = 0
        self._holds = 0
        self._pending: list[tuple[str, Callable[[], Any]]] = []
        self._reserved: dict[str, tuple[str, int]] = {}

    # -- placement -----------------------------------------------------------------------------------------

    def snapshot(self) -> Placement:
        sm = self.sm
        cold = set(getattr(sm, "cold", None) or ())
        resident = tuple(sorted(getattr(sm, "resident", {}) or {}))
        host = tuple(i for i in sorted(getattr(sm, "host", {}) or {}) if i not in cold)
        mlx = tuple(sorted(getattr(sm, "mlx_layers", ()) or ()))
        cuda = sm.dev.type == DeviceKind.CUDA
        hh = getattr(sm, "head_host", None)
        if hh is not None and getattr(hh, "packed", None) is not None:
            head = Tier.PACKED
        elif getattr(sm, "head", None) is not None and cuda:
            head = Tier.CARD
        else:
            head = Tier.HOST
        aj = getattr(sm, "aj", None)
        drafter = Tier.NONE
        if aj is not None:
            drafter = Tier.CARD if aj.dev.type == DeviceKind.CUDA else Tier.HOST
        elif str(getattr(sm, "proposer", "") or "").startswith("mtp") and any(
            k.startswith("mtp.") for k in (getattr(sm, "weight_map", None) or {})
        ):
            dd = getattr(sm, "drafter_dev", None)
            drafter = (
                Tier.HOST if (dd is not None and dd.type == DeviceKind.CPU) else (Tier.CARD if cuda else Tier.HOST)
            )
        kv = Tier.CARD if (cuda and not getattr(sm, "kv_host", False)) else Tier.HOST
        return Placement(self.version, resident, host, tuple(sorted(cold)), mlx, head, drafter, kv)

    @contextlib.contextmanager
    def hold(self) -> Iterator[Placement]:
        """The placement as it stands, held for the block: changes requested inside wait for its end."""
        self._holds += 1
        try:
            yield self.snapshot()
        finally:
            self._holds -= 1
            if self._holds == 0 and self._pending:
                self._apply_pending()

    def held(self) -> bool:
        return self._holds > 0

    def request(self, what: str, fn: Callable[[], Any]) -> Any:
        """A placement change: applied now when no pass holds the placement, else at the next boundary.
        Returns what the change returned when it ran now, None when it was queued."""
        if self._holds:
            self._pending.append((what, fn))
            return None
        self.version += 1
        return fn()

    def _apply_pending(self) -> None:
        pending, self._pending = self._pending, []
        for _what, fn in pending:
            self.version += 1
            fn()

    # -- the layer dispatch ----------------------------------------------------------------------------------

    def run_layer(self, i: int, h: torch.Tensor, pas: Any) -> torch.Tensor:
        """Layer `i` on the tier it lives on, the activation moved and cast at the tier's edge: fp32 on the
        CPU for a host layer, the layer's own dtype on the card for a resident or streamed one. The one place
        a tier boundary is crossed, so no path crosses it its own way (a host layer's fp32 output reaching a
        resident bf16 projection uncast was the `--cpu-layers N` failure). The tier is the pass's snapshot:
        a shed asked for meanwhile waits until the pass lets the placement go."""
        sm = self.sm
        tier = pas.place.tier(i)
        if tier in (LayerTier.HOST, LayerTier.COLD) and not pas.card_pass:
            if h.device.type != "cpu":
                h = h.detach().float().cpu()
            return sm._run_host_layer(i, h, pas)
        tmpl = sm._card_layer(i, pas)
        wd = sm._layer_dtype(tmpl)
        if h.device != sm.dev or (wd is not None and h.dtype != wd):
            h = h.to(sm.dev, wd) if wd is not None else h.to(sm.dev)
        return sm._run_card_layer(i, tmpl, h, pas)

    # -- memory --------------------------------------------------------------------------------------------

    def free(self, device: Any = None, unreserved: bool = False) -> int | None:
        """Memory free for an allocation on `device` (the engine's own when None), above the engine's margin
        there; with `unreserved`, less what `reserve()` has spoken for on it. On unified memory the ledger is
        the engine's own: the RAM the load started with, less the reserve, less everything MLX holds."""
        sm = self.sm
        dev = sm.dev if device is None else torch.device(device)
        if dev.type == DeviceKind.CUDA:
            if sm.dev.type != DeviceKind.CUDA:
                return None
            out = free_bytes(dev, int(getattr(sm, "vram_margin", 0) or 0))
        elif getattr(sm, "mlx", None) is not None:
            out = max(0, int(sm.mem_start) - int(getattr(sm, "ram_reserve", 0) or 0) - int(sm.mlx.held_bytes()))
        else:
            out = free_bytes(torch.device("cpu"), int(getattr(sm, "ram_reserve", 0) or 0))
        if out is None or not unreserved:
            return out
        return max(0, out - self.reserved(dev))

    def reserve(self, tag: str, nbytes: int, device: Any = None) -> None:
        """Memory spoken for under `tag` on `device` (the engine's own when None); a tag reserved again is
        replaced, not added."""
        dev = self.sm.dev if device is None else torch.device(device)
        self._reserved[tag] = (dev.type, max(0, int(nbytes)))

    def release(self, tag: str) -> None:
        self._reserved.pop(tag, None)

    def reserved(self, device: Any = None) -> int:
        dev = self.sm.dev if device is None else torch.device(device)
        return sum(n for t, n in self._reserved.values() if t == dev.type)
