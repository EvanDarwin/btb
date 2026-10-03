# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4 (`qwen4_exp`): the family (`family`), its sparse attention's indexer (`qsa`), its decode and speculative
verify through the gated DeltaNet and the per-layer n-gram embedding (`verify`), its host layers' router and sparse
attention row for row as the one-token step (`router`, `attend`), its card program - the resident layers' step and
verify pass through btb's row-invariant card kernels (`card`) - and its MTP drafter (`drafter`)."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from typing import Any


def bind_forward(module: Any, fn: Callable[..., Any]) -> None:
    """`fn(module, ...)` as `module`'s forward, the module held weakly: a method bound to it and kept on it held the
    module in a cycle with itself, so a layer let go - shed, or its engine closed - kept its weights until the cyclic
    collector came round (hundreds of MB a layer at a real model's widths). The class's own forward stays
    `type(module).forward`, which the installed forwards fall back on"""
    ref = weakref.ref(module)

    def forward(*args: Any, **kw: Any) -> Any:
        return fn(ref(), *args, **kw)

    module.forward = forward
