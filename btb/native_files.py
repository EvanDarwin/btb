# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Where the built library and the CUDA kernels ship: btb/native/<platform-tag>/. Torch-free, so a check for the
library costs no engine import."""

from __future__ import annotations

import os
import platform
import sys

_LIB_NAMES = {"windows": "btb_native.dll", "linux": "libbtb_native.so", "macos": "libbtb_native.dylib"}
_PKG = os.path.dirname(os.path.abspath(__file__))  # the package directory (btb/)


def native_tag() -> str:
    """Calculate the native platform tag"""
    system = {"win32": "windows", "linux": "linux", "darwin": "macos"}.get(sys.platform, sys.platform)
    m = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "arm64": "aarch64", "aarch64": "aarch64"}.get(m, m)
    return f"{system}-{arch}"


def native_path() -> str | None:
    tag = native_tag()
    own = _LIB_NAMES.get(tag.split("-")[0])
    names = [own] if own else list(_LIB_NAMES.values())
    cands = [os.path.join(_PKG, "native", tag, n) for n in names]
    cands += [os.path.join(_PKG, "native", n) for n in _LIB_NAMES.values()]
    for p in cands:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def kernels_path() -> str | None:
    """the CUDA decode kernels (btb_kernels.fatbin)"""
    tag = native_tag()
    for p in (
        os.path.join(_PKG, "native", tag, "btb_kernels.fatbin"),
        os.path.join(_PKG, "native", "btb_kernels.fatbin"),
    ):
        if os.path.exists(p):
            return os.path.abspath(p)
    return None
