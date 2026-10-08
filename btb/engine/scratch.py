# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The buffers a pass reuses from one call to the next (a verify pass's per-node states, a kernel's workspace): one
per name and device, each asked of the scheduler before it is allocated or grown, and all of them let go at the
engine's close."""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from typing import Any

import torch

from .holdings import Stage


class Scratch:
    """An engine's reusable buffers. `take` hands back a view of the buffer named `name` on `device`, grown - through
    `BatchScheduler.grant`, which refuses what the device cannot afford - when the shape asked for is past it; the
    old buffer is let go before the new one is priced, so the ledger reads the room the growth really needs. The
    engine is held weakly: the registry is the engine's, and a reference back would keep it alive past its close."""

    def __init__(self, sm: Any) -> None:
        self._sm = weakref.ref(sm)
        self._bufs: dict[tuple[str, str], torch.Tensor] = {}
        sm.holdings.own(Stage.MEMORY, "the passes' scratch buffers", self.clear)

    def take(
        self, name: str, shape: Sequence[int], dtype: torch.dtype, device: Any, requester: str, zeroed: bool = False
    ) -> torch.Tensor:
        """buffer `name` on `device` as `shape` of `dtype` (its contents whatever the last user left). `zeroed`: a
        buffer made new is zeros - for a user that leaves it at zero itself (a kernel's counters), so a buffer it
        takes again is zeros too"""
        dev = torch.device(device)
        key = (name, str(dev))
        n = math.prod(int(s) for s in shape)
        t = self._bufs.get(key)
        if t is None or t.dtype != dtype or t.numel() < n:
            self._bufs.pop(key, None)
            t = None
            # a pass takes its buffers on a live engine, whose scheduler prices every allocation
            sm = self._sm()
            assert sm is not None
            nbytes = n * torch.empty(0, dtype=dtype).element_size()
            sm.scheduler.grant(nbytes, "scratch", requester=requester, device=dev)
            t = (torch.zeros if zeroed else torch.empty)(n, dtype=dtype, device=dev)
            self._bufs[key] = t
        return t[:n].view(*shape)

    def release(self, prefix: str = "", device: Any = None) -> int:
        """the buffers whose names start with `prefix` (every one where empty) on `device` (every device where None)
        let go - a view a pass still holds keeps its memory until the pass lets it go - and the bytes they held"""
        dev = str(torch.device(device)) if device is not None else None
        gone = 0
        for key in [k for k in self._bufs if k[0].startswith(prefix) and (dev is None or k[1] == dev)]:
            t = self._bufs.pop(key)
            gone += t.numel() * t.element_size()
        return gone

    def clear(self) -> None:
        self._bufs.clear()
