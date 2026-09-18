# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Touched memory, kept: a process-level pool of unified-memory blocks whose pages were faulted once (a zero
fill on a helper thread, seeded from the checkpoint's headers before torch imports), cut into a model's weight
buffers and taken back whole when it closes. torch-free on purpose."""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import os
import struct
import sys
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from .hf import is_gguf
from .kinds import Log
from .quant import AFFINE_TYPES

if TYPE_CHECKING:
    from .mlx import Shared

CAP = 2**31 - 4096  # an MLX dimension is an int32
ALIGN = 64


def mlx_ok() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except Exception:
        return False


# the darwin reader's libc, struct class and host port, bound on the first call: the memory policy reads the
# figure once a second and a fresh CDLL and class per call cost 140 us against the read's few
_DARWIN_VM: list[Any] = []


def darwin_available_bytes() -> int:
    """free + inactive + speculative pages (what the kernel hands out without compressing or swapping); the
    file cache counts as inactive, so the mmapped checkpoint pages of the host tier are reclaimable"""
    if not _DARWIN_VM:
        _DARWIN_VM.extend(_darwin_vm_bind())
    libc, VMStat64, host, page = _DARWIN_VM
    st = VMStat64()
    count = ctypes.c_uint32(ctypes.sizeof(VMStat64) // 4)
    if libc.host_statistics64(host, 4, ctypes.byref(st), ctypes.byref(count)) != 0:  # HOST_VM_INFO64
        return 0
    return int(st.free_count + st.inactive_count + st.speculative_count) * page


def _darwin_vm_bind() -> tuple[Any, Any, int, int]:
    """(libc, the vm_statistics64 struct, this host's port, the page size)"""
    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libSystem.B.dylib")

    class VMStat64(ctypes.Structure):
        _fields_ = [
            ("free_count", ctypes.c_uint32),
            ("active_count", ctypes.c_uint32),
            ("inactive_count", ctypes.c_uint32),
            ("wire_count", ctypes.c_uint32),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint32),
            ("speculative_count", ctypes.c_uint32),
            ("decompressions", ctypes.c_uint64),
            ("compressions", ctypes.c_uint64),
            ("swapins", ctypes.c_uint64),
            ("swapouts", ctypes.c_uint64),
            ("compressor_page_count", ctypes.c_uint32),
            ("throttled_count", ctypes.c_uint32),
            ("external_page_count", ctypes.c_uint32),
            ("internal_page_count", ctypes.c_uint32),
            ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ]

    libc.mach_host_self.restype = ctypes.c_uint32
    host = libc.mach_host_self()
    page = ctypes.c_size_t(0)
    if libc.host_page_size(host, ctypes.byref(page)) != 0 or not page.value:
        page.value = int(os.sysconf("SC_PAGE_SIZE"))
    return libc, VMStat64, int(host), int(page.value)


def _headers(model_dir: str) -> dict[str, tuple[str, list[int], int]]:
    """every tensor's (dtype, shape, nbytes) from the safetensors headers; no weight is read"""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = [f for f in sorted(os.listdir(model_dir)) if f.endswith(".safetensors")]
    out = {}
    for name in files:
        with open(os.path.join(model_dir, name), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                a, b = v["data_offsets"]
                out[k] = (v["dtype"], v["shape"], b - a)
    return out


def _gguf_sizes(path: str, packed: bool) -> list[int]:
    """`sizes_for` of a GGUF file off its tensor list (llama.cpp's names: `token_embd`, `output`, `blk.N.*`): the
    2-D tensors as bf16, less those the packed kernels take as stored when `packed`"""
    import gguf

    al = lambda nb: (nb + ALIGN - 1) // ALIGN * ALIGN
    per: dict[int, int] = {}
    head = 0
    for t in gguf.GGUFReader(path).tensors:
        if len(t.shape) != 2 or (packed and t.tensor_type.name in AFFINE_TYPES):
            continue
        nb = 2 * int(t.n_elements)
        if t.name == "output.weight" or (t.name == "token_embd.weight" and not head):
            head = al(nb)
        elif t.name.startswith("blk."):
            i = int(t.name.split(".")[1])
            per[i] = per.get(i, 0) + al(nb)
    return [head] + [per[i] for i in sorted(per)] if per else []


def sizes_for(model_dir: str, gguf_packed: bool = True) -> list[int]:
    """The byte size of each buffer the resident binding asks for, in order: the head, then each layer's 2-D bf16
    linears (not norms, biases, convs, expert tensors or a router). A miss only means a fresh allocation."""
    if is_gguf(model_dir):
        return _gguf_sizes(model_dir, gguf_packed)
    h = _headers(model_dir)
    emb = [k for k in h if k.endswith("embed_tokens.weight")]
    if not emb:
        return []
    prefix = emb[0][: -len("embed_tokens.weight")]
    head = "lm_head.weight" if "lm_head.weight" in h else emb[0]
    al = lambda nb: (nb + ALIGN - 1) // ALIGN * ALIGN
    per: dict[int, int] = {}
    for k, (dt, shape, nb) in h.items():
        if dt != "BF16" or len(shape) != 2 or not k.startswith(prefix + "layers."):
            continue
        if ".experts." in k or "conv1d" in k or k.endswith("mlp.gate.weight"):
            continue
        i = int(k[len(prefix) + len("layers.") :].split(".")[0])
        per[i] = per.get(i, 0) + al(nb)
    return [al(h[head][2])] + [per[i] for i in sorted(per)]


class Pool:
    blocks: Any
    lock: Any
    streams: Any
    threads: Any

    def __init__(self) -> None:
        self.blocks = []  # [mlx array, numpy view, bytes used, Shared or None]
        self.lock = threading.Lock()
        self.threads = []  # seeds still filling
        self.streams = []  # the fills' own CPU streams, kept alive with their arrays

    def free_bytes(self) -> int:
        with self.lock:
            return sum(len(v) - used for _, v, used, _ in self.blocks)

    def seed(self, sizes: Iterable[int]) -> None:
        """Blocks cut so that the buffers of `sizes`, taken in order, never straddle one; only what the pool's
        free bytes do not already cover is allocated, on a helper thread, its pages faulted by the fill."""
        cuts, cur = [], 0
        for s in sizes:
            if s > CAP:
                return
            if cur + s > CAP:
                cuts.append(cur)
                cur = 0
            cur += s
        if cur:
            cuts.append(cur)
        held = self.free_bytes()
        while cuts and held >= cuts[0]:
            held -= cuts.pop(0)
        self._seed_cuts(cuts)

    def _seed_cuts(self, cuts: list[int]) -> None:
        if not cuts:
            return

        def run() -> None:
            import mlx.core as mx

            with contextlib.suppress(Exception):
                mx.set_wired_limit(int(mx.device_info().get("max_recommended_working_set_size", 0)))
            # the fills on up to four CPU streams at once (a stream is a thread of MLX's own): 7.7 GB in
            # ~0.25 s instead of ~0.9 on one, so a short window hides them
            streams = [mx.new_stream(mx.cpu) for _ in range(min(4, len(cuts)))]
            self.streams.extend(streams)
            xs = [mx.zeros((nb,), dtype=mx.uint8, stream=streams[i % len(streams)]) for i, nb in enumerate(cuts)]
            mx.eval(*xs)
            for x in xs:
                v = np.array(x, copy=False)
                with self.lock:
                    self.blocks.append([x, v, 0, None])

        t = threading.Thread(target=run, name="btb-pool", daemon=True)
        self.threads.append(t)
        t.start()

    def _settle(self) -> None:
        """wait for every seed still filling"""
        while self.threads:
            self.threads.pop(0).join()

    def take(self, nbytes: int) -> tuple[Shared, int] | None:
        """(Shared, offset) of `nbytes` in a block with room, or None; waits for a seed still filling"""
        from .mlx import Shared

        self._settle()
        need = (int(nbytes) + ALIGN - 1) // ALIGN * ALIGN
        with self.lock:
            for e in self.blocks:
                if len(e[1]) - e[2] >= need:
                    if e[3] is None:
                        e[3] = Shared.wrap(e[0], e[1])
                    off = e[2]
                    e[2] += need
                    return e[3], off
        return None

    def give(self, shared: Shared) -> None:
        """a block back, whole, once the model that read into it is closed"""
        with self.lock:
            for e in self.blocks:
                if e[3] is shared:
                    e[2] = 0


POOL = Pool()

# the RAM a load started with - the machine's free count plus the pool's own blocks - sampled before anything
# is allocated for it: the engine's ledger of what it may use, against which it counts what it holds
MEM_START = None


def seed_for(model_dir: str, log: Log | None = None, gguf_packed: bool = True) -> None:
    """At the top of a load: cut and fault the model's buffers if it fits in free RAM with a margin; a no-op off
    Apple silicon or with BTB_POOL=0."""
    global MEM_START
    if sys.platform == "darwin":
        if "torch" in sys.modules and log:
            # nothing can stop a caller importing torch first; the baseline then counts torch's own footprint
            # as memory in use, so the plan sees less RAM than the model really has
            log(
                "[pool] torch was imported before this load seeded the pool: the free-RAM baseline counts torch's "
                "footprint as used (import btb before torch to keep it exact)"
            )
        MEM_START = darwin_available_bytes() + POOL.free_bytes()
    if os.environ.get("BTB_POOL", "1") == "0" or not mlx_ok():
        return
    try:
        sizes = sizes_for(model_dir, gguf_packed)
    except Exception:
        return
    if not sizes:
        return
    need = sum(sizes) - POOL.free_bytes()
    avail = darwin_available_bytes()
    if need <= 0 or need > avail - max(4 << 30, avail // 8):
        return
    if log:
        log(
            f"[pool] faulting {sum(sizes) / 2**30:.2f} GB of weight buffers under the imports "
            f"({POOL.free_bytes() / 2**30:.2f} GB already held)"
        )
    POOL.seed(sizes)
