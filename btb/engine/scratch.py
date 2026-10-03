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

    def take(self, name: str, shape: Sequence[int], dtype: torch.dtype, device: Any, requester: str) -> torch.Tensor:
        """buffer `name` on `device` as `shape` of `dtype` (its contents whatever the last user left)"""
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
            t = torch.empty(n, dtype=dtype, device=dev)
            self._bufs[key] = t
        return t[:n].view(*shape)

    def clear(self) -> None:
        self._bufs.clear()
