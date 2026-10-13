# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card kernels' generated launchers (btb/engine/card_launch.py) without a card: every launcher hands the
launcher it is given - here one of the test's own, no base class - the kernel's name and one ctypes value a
parameter, each as wide as native/cuda's signature declares it; and the card module's methods are those functions."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import torch

from btb.engine import card_launch
from btb.engine.native import _Cuda
from btb.kinds import Query
from tests.cert.cuda_ops import bindings

WIDTH = {"int": 4, "unsigned": 4, "float": 4, "long long": 8, "unsigned long long": 8, "unsigned char": 1}


class Recorder:
    """a launcher of one's own: the launches it is handed, as (kernel, argument widths, stream, shared)"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], torch.cuda.Stream | None, int]] = []

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[Any],
        stream: torch.cuda.Stream | None = None,
        shared: int = 0,
    ) -> None:
        self.calls.append((name, tuple(ctypes.sizeof(a) for a in args), stream, shared))


def test_every_launcher_hands_its_launcher_the_kernels_parameters() -> None:
    roles = {r.name: r.ctype for r in bindings().card_roles()}
    for k in bindings().card_kernels():
        fn = getattr(card_launch, k.method)
        want = tuple(8 if p.pointer else WIDTH[roles[p.role]] for p in k.params)
        args = [0 if p.pointer or roles[p.role] != "float" else 0.0 for p in k.params]
        for value, name in zip(k.values or ("",), k.names(), strict=True):
            rec = Recorder()
            variant = [int(value)] if k.variant else []
            fn(rec, *variant, (1, 1, 1), (1, 1, 1), *args, shared=48)
            assert rec.calls == [(name, want, None, 48)], name


def test_the_card_modules_methods_are_the_launchers() -> None:
    """`k.btb_attn_flash(...)` is `btb_attn_flash(k, ...)`: the card module carries each function itself, and `ptr`"""
    for k in bindings().card_kernels():
        assert getattr(_Cuda, k.method) is getattr(card_launch, k.method), k.method
    assert _Cuda.ptr is card_launch.ptr
    assert card_launch.ptr(Query, None) == 0
