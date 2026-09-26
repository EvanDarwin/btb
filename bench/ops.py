# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
# ruff: noqa: B023 - every lane is a closure over its loop iteration's operands, measured and dropped inside that
# same iteration, never called after the loop moves on
"""Where an op wins: each op timed on the CPU, on the card over operands already in VRAM, and on the card over
weights shipped from RAM each call, at the shapes and row counts the engine runs. The question is the placement,
not the kernel: every lane is wall time as the engine would pay it, the crossing of the tier's edge included.

    python bench/ops.py --native btb_native.dll --kernels btb_kernels.fatbin --out ops.json [--quick]

Weights rotate through pools past the CPU's L3 and the card's L2, so every call reads from memory as a decode
does. The lanes of a point run interleaved in rounds (the order shuffled a round), so a drift of the machine
lands on every lane alike; a lane's figure is the median of its rounds. Each lane's output is held against a
float32 reference and its error recorded beside its time.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from btb.engine.native import Native  # noqa: E402
from btb.mxfp4 import MxWeight  # noqa: E402
from btb.mxfp4_torch import dequant_slot  # noqa: E402

CPU_POOL = 384 << 20  # past the 13700K's 30 MB L3 many times over
CARD_POOL = 256 << 20  # past the card's L2 (48 MB on an AD104)
ROUNDS = 7
TARGET_S = 0.35  # a point's time budget per lane, over all its rounds

Lane = Callable[[], Any]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def sync() -> None:
    torch.cuda.synchronize()


def measure(lanes: dict[str, Lane], target_s: float = TARGET_S, rounds: int = ROUNDS) -> dict[str, dict[str, float]]:
    """each lane's seconds a call: one warm call apiece, a batch size from it, then `rounds` rounds of one batch a
    lane in a shuffled order; the median of the rounds' means, and the spread of the rounds"""
    per: dict[str, float] = {}
    for name, fn in lanes.items():
        fn()
        t0 = time.perf_counter()
        fn()
        per[name] = max(time.perf_counter() - t0, 1e-6)
    slow = max(per.values())
    r = rounds if slow < 0.5 else 3
    batch = {n: max(1, min(200, int(target_s / r / per[n]))) for n in lanes}
    got: dict[str, list[float]] = {n: [] for n in lanes}
    order = list(lanes)
    for k in range(r):
        random.shuffle(order)
        for n in order:
            if per[n] > 2.0 and k > 0:  # a lane seconds a call is plainly lost: one round says so
                continue
            fn, b = lanes[n], batch[n]
            t0 = time.perf_counter()
            for _ in range(b):
                fn()
            got[n].append((time.perf_counter() - t0) / b)
    out = {}
    for n, xs in got.items():
        xs.sort()
        out[n] = {"s": statistics.median(xs), "lo": xs[0], "hi": xs[-1], "calls": batch[n] * len(xs)}
    return out


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    y, ref = y.detach().float().cpu(), ref.detach().float().cpu()
    return float((y - ref).abs().max() / ref.abs().max().clamp(min=1e-30))


class Pool:
    """copies of one weight, enough to pass `nbytes`, handed out in turn"""

    def __init__(self, make: Callable[[], Any], each: int, nbytes: int, cap: int = 64) -> None:
        n = max(2, min(cap, -(-nbytes // max(1, each))))
        self.items = [make() for _ in range(n)]
        self.i = 0

    def next(self) -> Any:
        self.i = (self.i + 1) % len(self.items)
        return self.items[self.i]

    @property
    def first(self) -> Any:
        return self.items[0]


class Bench:
    def __init__(self, native: str, kernels: str | None, quick: bool) -> None:
        self.quick = quick
        Native.load_gemv(native)
        self.native = native
        self.k = Native.load_cuda(kernels) if kernels else None
        self.dev = torch.device("cuda")
        self.records: list[dict[str, Any]] = []
        self.copy = torch.cuda.Stream()

    def add(
        self, op: str, shape: Any, rows: int, res: dict[str, dict[str, float]], err: dict[str, float], **kw: Any
    ) -> None:
        for lane, r in res.items():
            rec = {"op": op, "shape": shape, "rows": rows, "lane": lane, **r, "err": err.get(lane), **kw}
            self.records.append(rec)
        best = min(res, key=lambda n: res[n]["s"])
        cells = "  ".join(f"{n} {res[n]['s'] * 1e3:.3f}" for n in res)
        print(f"{op:10} {shape!s:>16} r={rows:<5} best={best:22} | {cells}", flush=True)

    # -- transfers ------------------------------------------------------------------------------------------------

    def transfers(self) -> None:
        sizes = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20] if not self.quick else [4 << 10, 16 << 20]
        for nb in sizes:
            hp = torch.empty(nb, dtype=torch.uint8).pin_memory()
            hq = torch.empty(nb, dtype=torch.uint8)
            hq.fill_(1)
            d = torch.empty(nb, dtype=torch.uint8, device=self.dev)

            def h2d_pinned() -> None:
                d.copy_(hp, non_blocking=True)
                sync()

            def h2d_pageable() -> None:
                d.copy_(hq)
                sync()

            def d2h_pinned() -> None:
                hp.copy_(d, non_blocking=True)
                sync()

            def d2h_pageable() -> None:
                hq.copy_(d)

            res = measure(
                {
                    "h2d_pinned": h2d_pinned,
                    "h2d_pageable": h2d_pageable,
                    "d2h_pinned": d2h_pinned,
                    "d2h_pageable": d2h_pageable,
                }
            )
            for r in res.values():
                r["gbps"] = nb / r["s"] / 1e9
            self.add("transfer", nb, 1, res, {})
        # the handoff: a hidden row to the card, one small kernel, back, as a tier edge costs
        for H in (2560, 5120):
            x = torch.randn(1, H)

            def handoff() -> torch.Tensor:
                return (x.to(self.dev, torch.bfloat16) * 2).float().cpu()

            xd = x.to(self.dev)

            def launch_only() -> None:
                (xd * 2)
                sync()

            self.add("handoff", H, 1, measure({"roundtrip": handoff, "kernel_sync": launch_only}), {})

    # -- a bf16 linear --------------------------------------------------------------------------------------------

    def linear(self, name: str, R: int, C: int, rows_list: list[int]) -> None:
        each = R * C * 2
        cpu = Pool(lambda: torch.randn(R, C).mul_(C**-0.5).bfloat16(), each, CPU_POOL)
        pin = Pool(lambda: cpu.first.clone().pin_memory(), each, CARD_POOL, cap=8)
        card = Pool(lambda: cpu.first.to(self.dev), each, CARD_POOL)
        bufs = [torch.empty(R, C, dtype=torch.bfloat16, device=self.dev) for _ in range(2)]
        free = [torch.cuda.Event() for _ in range(2)]
        for rows in rows_list:
            x = torch.randn(rows, C)
            # float64 on the card: a head's float64 copy is 3 GB, and the card's float64 is fast enough here
            ref = (x.to(self.dev, torch.float64) @ card.first.double().T).cpu()
            xd = x.to(self.dev, torch.bfloat16)
            lanes: dict[str, Lane] = {}
            outs: dict[str, Callable[[], torch.Tensor]] = {}

            if rows <= 512:
                y = torch.empty(rows, R)

                def cpu_native() -> None:
                    Native.gemv(cpu.next(), x, y)

                lanes["cpu_native"] = cpu_native
                outs["cpu_native"] = lambda: (Native.gemv(cpu.first, x, y), y)[1]
            if rows >= 16:
                # the engine's multi-row host path: the weight widened to float32 each call
                lanes["cpu_torch_f32"] = lambda: F.linear(x, cpu.next().float())
                outs["cpu_torch_f32"] = lambda: F.linear(x, cpu.first.float())
                xb = x.bfloat16()
                lanes["cpu_torch_bf16"] = lambda: F.linear(xb, cpu.next())
                outs["cpu_torch_bf16"] = lambda: F.linear(xb, cpu.first)

            def card_resident() -> torch.Tensor:
                return F.linear(x.to(self.dev, torch.bfloat16), card.next()).float().cpu()

            def card_kernel() -> None:
                F.linear(xd, card.next())
                sync()

            lanes["card_resident"] = card_resident
            lanes["card_kernel"] = card_kernel
            outs["card_resident"] = lambda: F.linear(x.to(self.dev, torch.bfloat16), card.first).float()
            k = self.k
            if k is not None and rows <= 32:
                M = next(m for m in (1, 2, 4, 8, 16, 32) if rows <= m)
                xm = torch.zeros(M, C, dtype=torch.bfloat16, device=self.dev)
                xm[:rows] = xd
                ym = torch.empty(M, R, dtype=torch.bfloat16, device=self.dev)
                x32 = torch.zeros(32, C, dtype=torch.bfloat16, device=self.dev)
                x32[:rows] = xd
                y32 = torch.empty(32, R, dtype=torch.bfloat16, device=self.dev)
                P, ci = k.ptr, ctypes.c_int

                def btb_fp32(w: torch.Tensor | None = None) -> torch.Tensor:
                    w = card.next() if w is None else w
                    k.launch(
                        f"btb_gemv_bf16_m{M}", ((R + 3) // 4, 1, 1), (128, 1, 1), [P(w), P(xm), P(ym), ci(R), ci(C)]
                    )
                    sync()
                    return ym[:rows]

                def btb_mma(w: torch.Tensor | None = None) -> torch.Tensor:
                    w = card.next() if w is None else w
                    k.launch(
                        "btb_gemv_mma_bf16",
                        ((R + 15) // 16, 1, 1),
                        (64, 1, 1),
                        [P(w), P(x32), P(y32), ci(R), ci(C), ci(rows)],
                    )
                    sync()
                    return y32[:rows]

                lanes["card_btb_fp32"] = btb_fp32
                outs["card_btb_fp32"] = lambda: btb_fp32(card.first)
                if "btb_gemv_mma_bf16" in k.fn:
                    lanes["card_btb_mma"] = btb_mma
                    outs["card_btb_mma"] = lambda: btb_mma(card.first)

            def stream_pinned() -> torch.Tensor:
                bufs[0].copy_(pin.next(), non_blocking=True)
                return F.linear(x.to(self.dev, torch.bfloat16), bufs[0]).float().cpu()

            def stream_pageable() -> torch.Tensor:
                bufs[0].copy_(cpu.next())
                return F.linear(x.to(self.dev, torch.bfloat16), bufs[0]).float().cpu()

            def stream_piped(k: int = 8) -> None:
                # the Express's card side in its steady state: the next weight's copy under the current matmul, the
                # rows already on the card, outputs left there
                main = torch.cuda.current_stream()
                for j in range(k):
                    b = j % 2
                    ready = torch.cuda.Event()
                    with torch.cuda.stream(self.copy):
                        self.copy.wait_event(free[b])
                        bufs[b].copy_(pin.next(), non_blocking=True)
                        ready.record(self.copy)
                    main.wait_event(ready)
                    F.linear(xd, bufs[b])
                    free[b].record(main)
                sync()

            lanes["card_stream_pinned"] = stream_pinned
            lanes["card_stream_pageable"] = stream_pageable
            lanes["card_stream_piped"] = stream_piped
            res = measure(lanes)
            res["card_stream_piped"] = {
                k2: (v / 8 if k2 in ("s", "lo", "hi") else v) for k2, v in res["card_stream_piped"].items()
            }
            err = {}
            for n, f in outs.items():
                err[n] = rel_err(f(), ref)
            self.add("linear", f"{name} {R}x{C}", rows, res, err, bytes=each)

    # -- one expert: gate_up, the gate, down ------------------------------------------------------------------------

    def expert_bf16(self, name: str, H: int, I: int, rows_list: list[int], group: int = 16) -> None:
        each = 3 * H * I * 2

        def mk() -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.randn(2 * I, H).mul_(H**-0.5).bfloat16(),
                torch.randn(H, I).mul_(I**-0.5).bfloat16(),
            )

        cpu = Pool(mk, each, CPU_POOL, cap=128)
        pin = Pool(lambda: tuple(t.clone().pin_memory() for t in cpu.first), each, CARD_POOL, cap=32)
        card = Pool(lambda: tuple(t.to(self.dev) for t in cpu.first), each, CARD_POOL, cap=64)
        bufs = [
            (
                torch.empty(2 * I, H, dtype=torch.bfloat16, device=self.dev),
                torch.empty(H, I, dtype=torch.bfloat16, device=self.dev),
            )
            for _ in range(2)
        ]
        free = [torch.cuda.Event() for _ in range(2)]

        def act(gu: torch.Tensor) -> torch.Tensor:
            g, u = gu.chunk(2, dim=-1)
            return F.silu(g) * u

        for rows in rows_list:
            x = torch.randn(rows, H)
            gu0, dn0 = cpu.first
            ref = act(x.double() @ gu0.double().T) @ dn0.double().T
            xd = x.to(self.dev, torch.bfloat16)
            lanes: dict[str, Lane] = {}
            outs: dict[str, Callable[[], torch.Tensor]] = {}
            gu_y = torch.empty(rows, 2 * I)
            dn_y = torch.empty(rows, H)

            def cpu_native(w: Any = None) -> torch.Tensor:
                gu, dn = cpu.next() if w is None else w
                Native.gemv(gu, x, gu_y)
                h = act(gu_y).contiguous()
                Native.gemv(dn, h, dn_y)
                return dn_y

            if rows <= 512:
                lanes["cpu_native"] = cpu_native
                outs["cpu_native"] = lambda: cpu_native(cpu.first)
                if Native.gemv_group is not None:
                    gys = [torch.empty(rows, 2 * I) for _ in range(group)]
                    dys = [torch.empty(rows, H) for _ in range(group)]

                    def cpu_group() -> None:
                        ws = [cpu.next() for _ in range(group)]
                        Native.gemv_group([w[0] for w in ws], [x] * group, gys)
                        hs = [act(g).contiguous() for g in gys]
                        Native.gemv_group([w[1] for w in ws], hs, dys)

                    lanes["cpu_group"] = cpu_group
            if rows >= 16:
                lanes["cpu_torch_f32"] = lambda: (lambda w: F.linear(act(F.linear(x, w[0].float())), w[1].float()))(
                    cpu.next()
                )

            def on_card(w: Any, xin: torch.Tensor) -> torch.Tensor:
                return F.linear(act(F.linear(xin, w[0])), w[1])

            lanes["card_resident"] = lambda: on_card(card.next(), x.to(self.dev, torch.bfloat16)).float().cpu()
            outs["card_resident"] = lambda: on_card(card.first, x.to(self.dev, torch.bfloat16)).float()

            def stream_pinned() -> torch.Tensor:
                w = pin.next()
                bufs[0][0].copy_(w[0], non_blocking=True)
                bufs[0][1].copy_(w[1], non_blocking=True)
                return on_card(bufs[0], x.to(self.dev, torch.bfloat16)).float().cpu()

            def stream_pageable() -> torch.Tensor:
                w = cpu.next()
                bufs[0][0].copy_(w[0])
                bufs[0][1].copy_(w[1])
                return on_card(bufs[0], x.to(self.dev, torch.bfloat16)).float().cpu()

            def stream_piped(k: int = group) -> None:
                main = torch.cuda.current_stream()
                for j in range(k):
                    b = j % 2
                    ready = torch.cuda.Event()
                    w = pin.next()
                    with torch.cuda.stream(self.copy):
                        self.copy.wait_event(free[b])
                        bufs[b][0].copy_(w[0], non_blocking=True)
                        bufs[b][1].copy_(w[1], non_blocking=True)
                        ready.record(self.copy)
                    main.wait_event(ready)
                    on_card(bufs[b], xd)
                    free[b].record(main)
                sync()

            lanes["card_stream_pinned"] = stream_pinned
            lanes["card_stream_pageable"] = stream_pageable
            lanes["card_stream_piped"] = stream_piped
            res = measure(lanes)
            for n in ("card_stream_piped", "cpu_group"):
                if n in res:
                    res[n] = {k2: (v / group if k2 in ("s", "lo", "hi") else v) for k2, v in res[n].items()}
            err = {n: rel_err(f(), ref) for n, f in outs.items()}
            self.add("expert", f"{name} H{H} I{I}", rows, res, err, bytes=each)

    def expert_mxfp4(self, name: str, H: int, I: int, rows_list: list[int]) -> None:
        """MXFP4 as stored: the CPU's own matvec over the blocks, against the card fed the blocks (a quarter of the
        bf16 bytes over the bus) and widening them there"""
        if Native.gemv_mx4 is None:
            print("[ops] no MXFP4 matvec in this library")
            return

        def mat(rows: int, k: int) -> MxWeight:
            G = k // 32
            blocks = torch.randint(0, 256, (rows * G * 16,), dtype=torch.uint8)
            scales = torch.randint(118, 124, (rows * G,), dtype=torch.uint8)
            return MxWeight(blocks, scales, rows, k)

        def slot(w: MxWeight) -> torch.Tensor:
            assert w.scales is not None  # the checkpoint's layout, built above
            return torch.cat([w.blocks.reshape(-1), w.scales.reshape(-1)])

        mk = lambda: (mat(2 * I, H), mat(H, I))
        each = mk()[0].nbytes + mk()[1].nbytes
        cpu = Pool(mk, each, CPU_POOL, cap=128)
        pin = Pool(lambda: tuple(slot(w).pin_memory() for w in cpu.first), each, CARD_POOL, cap=64)
        dslot = [torch.empty(slot(w).numel(), dtype=torch.uint8, device=self.dev) for w in cpu.first]
        wide = [
            torch.empty(2 * I, H, dtype=torch.bfloat16, device=self.dev),
            torch.empty(H, I, dtype=torch.bfloat16, device=self.dev),
        ]

        def act(gu: torch.Tensor) -> torch.Tensor:
            g, u = gu.chunk(2, dim=-1)
            return F.silu(g) * u

        for rows in rows_list:
            x = torch.randn(rows, H)
            gu0, dn0 = cpu.first
            ref = act(x.double() @ gu0.dequantize(torch.float32).double().T) @ dn0.dequantize(torch.float32).double().T
            gu_y = torch.empty(rows, 2 * I)
            dn_y = torch.empty(rows, H)

            def cpu_native(w: Any = None) -> torch.Tensor:
                gu, dn = cpu.next() if w is None else w
                Native.gemv_mx4(gu, x, gu_y)
                h = act(gu_y).contiguous()
                Native.gemv_mx4(dn, h, dn_y)
                return dn_y

            def stream_pinned(w: Any = None) -> torch.Tensor:
                s = pin.next() if w is None else w
                for j, (src, dims) in enumerate(zip(s, ((2 * I, H), (H, I)), strict=True)):
                    dslot[j].copy_(src, non_blocking=True)
                    dequant_slot(dslot[j], dims[0], dims[1], out=wide[j])
                xd = x.to(self.dev, torch.bfloat16)
                return F.linear(act(F.linear(xd, wide[0])), wide[1]).float().cpu()

            res = measure({"cpu_native": cpu_native, "card_stream_pinned": stream_pinned})
            err = {
                "cpu_native": rel_err(cpu_native(cpu.first), ref),
                "card_stream_pinned": rel_err(stream_pinned(pin.first), ref),
            }
            self.add("expert_mx4", f"{name} H{H} I{I}", rows, res, err, bytes=each)

    # -- attention over the cache, one row ---------------------------------------------------------------------------

    def attn_decode(self, Hq: int, Hk: int, d: int, ctxs: list[int]) -> None:
        if Native.attn_decode is None:
            print("[ops] no attention kernel in this library")
            return
        for n in ctxs:
            each = 2 * Hk * n * d * 2
            mk = lambda: (torch.randn(Hk, n, d).bfloat16(), torch.randn(Hk, n, d).bfloat16())
            cpu = Pool(mk, each, CPU_POOL, cap=32)
            card = Pool(lambda: tuple(t.to(self.dev) for t in cpu.first), each, CARD_POOL, cap=16)
            pin = Pool(lambda: tuple(t.clone().pin_memory() for t in cpu.first), each, CARD_POOL, cap=4)
            kb = torch.empty(Hk, n, d, dtype=torch.bfloat16, device=self.dev)
            vb = torch.empty(Hk, n, d, dtype=torch.bfloat16, device=self.dev)
            q = torch.randn(Hq, d)
            out = torch.empty(Hq, d)
            scale = d**-0.5
            k0, v0 = cpu.first
            ref = F.scaled_dot_product_attention(
                q.double().view(1, Hq, 1, d),
                k0.double().unsqueeze(0),
                v0.double().unsqueeze(0),
                scale=scale,
                enable_gqa=True,
            ).view(Hq, d)

            def cpu_native(kv: Any = None) -> torch.Tensor:
                k, v = cpu.next() if kv is None else kv
                Native.attn_decode(q, k, v, scale, out)
                return out

            def sdpa(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                qd = q.to(self.dev, torch.bfloat16).view(1, Hq, 1, d)
                return F.scaled_dot_product_attention(qd, k.unsqueeze(0), v.unsqueeze(0), scale=scale, enable_gqa=True)

            def card_resident(kv: Any = None) -> torch.Tensor:
                k, v = card.next() if kv is None else kv
                return sdpa(k, v).float().cpu().view(Hq, d)

            def stream_pinned() -> torch.Tensor:
                k, v = pin.next()
                kb.copy_(k, non_blocking=True)
                vb.copy_(v, non_blocking=True)
                return sdpa(kb, vb).float().cpu()

            res = measure(
                {"cpu_native": cpu_native, "card_resident": card_resident, "card_stream_pinned": stream_pinned}
            )
            err = {
                "cpu_native": rel_err(cpu_native(cpu.first), ref),
                "card_resident": rel_err(card_resident(card.first), ref),
            }
            self.add("attn_dec", f"q{Hq} kv{Hk} d{d}", n, res, err, bytes=each)

    # -- the glue between the matmuls -------------------------------------------------------------------------------

    def glue(self, H: int, rows_list: list[int]) -> None:
        eps = 1e-6
        wn = torch.rand(H) + 0.5
        wnd = wn.to(self.dev, torch.bfloat16)
        E, topk = 512, 10
        wr = torch.randn(E, H).mul_(H**-0.5)
        wrd = wr.to(self.dev, torch.bfloat16)

        def norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            v = x.float().pow(2).mean(-1, keepdim=True)
            return (x.float() * torch.rsqrt(v + eps)).to(x.dtype) * w

        def route(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            p = torch.softmax(F.linear(x, w).float(), -1)
            return torch.topk(p, topk, dim=-1)

        for rows in rows_list:
            x = torch.randn(rows, H)
            xd = x.to(self.dev, torch.bfloat16)
            gu = torch.randn(rows, 2 * H)
            gud = gu.to(self.dev, torch.bfloat16)
            for op, cpu_fn, card_fn, xin, xcard in (
                ("rmsnorm", lambda a: norm(a, wn), lambda a: norm(a, wnd), x, xd),
                ("router", lambda a: route(a, wr), lambda a: route(a, wrd), x, xd),
                ("silu_mul", lambda a: F.silu(a[:, :H]) * a[:, H:], lambda a: F.silu(a[:, :H]) * a[:, H:], gu, gud),
            ):

                def on_card(f: Any = card_fn, a: Any = xcard) -> None:
                    f(a)
                    sync()

                def handoff(f: Any = card_fn, a: Any = xin) -> None:
                    y = f(a.to(self.dev, torch.bfloat16))
                    y = y[0] if isinstance(y, tuple) else y
                    y.float().cpu()

                def on_host(f: Any = cpu_fn, a: Any = xin) -> None:
                    f(a)

                res = measure({"cpu": on_host, "card_on_card": on_card, "card_handoff": handoff})
                self.add(op, H, rows, res, {})
        # the pick over a vocabulary's logits
        for V in (151936,):
            for R in (1, 16):
                lg = torch.randn(R, V)
                lgd = lg.to(self.dev)
                keys = list(range(R))

                def argmax_on_card(lgd: torch.Tensor = lgd) -> None:
                    lgd.argmax(-1)
                    sync()

                def argmax_host(lg: torch.Tensor = lg) -> None:
                    lg.argmax(-1)

                def argmax_handoff(lg: torch.Tensor = lg) -> None:
                    lg.to(self.dev).argmax(-1).cpu()

                def pick_host(lg: torch.Tensor = lg, keys: list[int] = keys) -> None:
                    Native.sample_pick(lg, keys, 0.8, 0, 0.9)

                lanes: dict[str, Lane] = {
                    "cpu_argmax": argmax_host,
                    "card_argmax_on_card": argmax_on_card,
                    "card_argmax_handoff": argmax_handoff,
                }
                if Native.sample_pick is not None:
                    lanes["cpu_pick_p0.9"] = pick_host
                k = self.k
                if k is not None:

                    def pick_on_card(lgd: torch.Tensor = lgd, keys: list[int] = keys, k: Any = k) -> None:
                        k.pick(lgd, keys, 0.8, 0, 0.9)
                        sync()

                    lanes["card_pick_p0.9_on_card"] = pick_on_card
                self.add("pick", V, R, measure(lanes), {})

    # -- the Express: one MoE layer's prefill, the crowded experts on the card and the sparse ones on the CPU ----

    def express(self, trace: str, H: int, I: int, cuts: list[int], every: int = 6) -> None:
        """A recorded prefill's expert calls replayed (`bench/route_trace.py`): every `every`th layer's calls, each
        with its real picks - which rows go to which expert - over stand-in weights of the model's expert shape.
        Lanes: today's path, every expert fed to the card through pinned double buffers, every expert on the CPU,
        and the Express at each cut (experts with at least `cut` rows on the card, the rest on the CPU at once)."""
        import numpy as np

        tr = np.load(trace)
        meta = json.loads(str(tr["meta"]))
        run, layer, off, picks = tr["run"], tr["layer"], tr["off"], tr["picks"]
        print(
            f"[express] {len(run)} recorded calls; placement {meta.get('placement', {}).get('resident', '?')}",
            flush=True,
        )
        P = 128  # distinct experts in memory, 1.26 GB a pool: past the caches, and expert e reads pool[e % P]
        pin = [
            (
                torch.randn(2 * I, H).mul_(H**-0.5).bfloat16().pin_memory(),
                torch.randn(H, I).mul_(I**-0.5).bfloat16().pin_memory(),
            )
            for _ in range(P)
        ]
        page = [(g.clone(), d.clone()) for g, d in pin]
        bufs = [
            (
                torch.empty(2 * I, H, dtype=torch.bfloat16, device=self.dev),
                torch.empty(H, I, dtype=torch.bfloat16, device=self.dev),
            )
            for _ in range(2)
        ]
        free = [torch.cuda.Event() for _ in range(2)]

        def act(gu: torch.Tensor) -> torch.Tensor:
            g, u = gu.chunk(2, dim=-1)
            return F.silu(g) * u

        for i in range(len(run)):
            if int(layer[i]) % every:
                continue
            top = torch.from_numpy(picks[off[i] : off[i + 1]].astype(np.int64))
            T = int(top.shape[0])
            label = f"T{meta['runs'][int(run[i])]['T']} L{int(layer[i])} c{i}"
            x = torch.randn(T, H)
            xd = x.to(self.dev, torch.bfloat16)
            calls = []  # (expert, its rows on the host, on the card), the most rows first
            for e in torch.unique(top).tolist():
                idx = (top == e).any(-1).nonzero().flatten()
                calls.append((e, idx, idx.to(self.dev)))
            calls.sort(key=lambda c: -len(c[1]))
            counts = [len(c[1]) for c in calls]
            out_cpu = torch.zeros(T, H)

            def cpu_side(part: list[Any], G: int = 16) -> None:
                # grouped: one pool barrier per G experts a matvec, and one scatter for the layer - an expert at a
                # time paid ~50 us of dispatch a matvec and 0.15-0.3 ms of index_add_ (scratchpad cpu_side_diag)
                out_cpu.zero_()
                idxs, ys = [], []
                for s in range(0, len(part), G):
                    grp = part[s : s + G]
                    xs = [x[idx] for _, idx, _ in grp]
                    gys = [torch.empty(len(idx), 2 * I) for _, idx, _ in grp]
                    Native.gemv_group([pin[e % P][0] for e, _, _ in grp], xs, gys)
                    hs = [act(g).contiguous() for g in gys]
                    dys = [torch.empty(len(idx), H) for _, idx, _ in grp]
                    Native.gemv_group([pin[e % P][1] for e, _, _ in grp], hs, dys)
                    idxs += [idx for _, idx, _ in grp]
                    ys += dys
                if idxs:
                    out_cpu.index_add_(0, torch.cat(idxs), torch.cat(ys))

            def card_side(part: list[Any], out: torch.Tensor) -> None:
                main = torch.cuda.current_stream()
                for j, (e, _, idd) in enumerate(part):
                    b = j % 2
                    ready = torch.cuda.Event()
                    with torch.cuda.stream(self.copy):
                        self.copy.wait_event(free[b])
                        bufs[b][0].copy_(pin[e % P][0], non_blocking=True)
                        bufs[b][1].copy_(pin[e % P][1], non_blocking=True)
                        ready.record(self.copy)
                    main.wait_event(ready)
                    out.index_add_(0, idd, F.linear(act(F.linear(xd[idd], bufs[b][0])), bufs[b][1]))
                    free[b].record(main)

            def split(cut: int) -> Callable[[], torch.Tensor]:
                card = [c for c in calls if len(c[1]) >= cut]
                host = [c for c in calls if len(c[1]) < cut]

                def run_() -> torch.Tensor:
                    out = torch.zeros(T, H, dtype=torch.bfloat16, device=self.dev)
                    th = threading.Thread(target=cpu_side, args=(host,)) if host else None
                    if th is not None:
                        th.start()
                    card_side(card, out)
                    if th is not None:
                        th.join()
                        out += out_cpu.to(self.dev, torch.bfloat16)
                    sync()
                    return out

                return run_

            def today_card() -> torch.Tensor:
                # `_Experts.forward` at >= 64 rows: each expert's weights to the card from the store's pageable
                # pages, one expert after another
                out = torch.zeros(T, H, dtype=torch.bfloat16, device=self.dev)
                for e, _, idd in calls:
                    gu, dn = page[e % P]
                    h = act(F.linear(xd[idd], gu.to(self.dev, non_blocking=True)))
                    out.index_add_(0, idd, F.linear(h, dn.to(self.dev, non_blocking=True)))
                sync()
                return out

            # below 64 rows the engine keeps the call on the host (`on_host`): today is the CPU side, one by one
            today = today_card if T >= Native.gemm_rows else split(1 << 30)
            lanes: dict[str, Lane] = {"today": today, "all_card_piped": split(0), "all_cpu": split(1 << 30)}
            for c in cuts:
                if 1 < c <= counts[0]:
                    lanes[f"cut{c}"] = split(c)
            ref = split(1 << 30)().float()
            err = {n: rel_err(f(), ref) for n, f in lanes.items() if n != "all_cpu"}
            res = measure(lanes)
            q = lambda p: counts[min(len(counts) - 1, int(p * len(counts)))]
            self.add(
                "express",
                label,
                T,
                res,
                err,
                run=int(run[i]),
                layer=int(layer[i]),
                hit=len(counts),
                rows_max=counts[0],
                rows_p50=q(0.5),
                rows_p90=q(0.1),
                rows_mean=sum(counts) / len(counts),
                card_at={c: sum(1 for n in counts if n >= c) for c in cuts},
            )
            print(
                f"           {label}: {len(counts)} experts hit, rows max {counts[0]} p90 {q(0.1)} median {q(0.5)}; "
                f"experts at or past each cut {[(c, sum(1 for n in counts if n >= c)) for c in cuts]}",
                flush=True,
            )

    # -- the two sides at once --------------------------------------------------------------------------------------

    def interference(self) -> None:
        """the CPU's matvec rate with the bus idle and with the card pulling pinned RAM flat out, and the bus's rate
        alone and beside the matvec: both read the same DRAM"""
        R, C = 2560, 5120
        cpu = Pool(lambda: torch.randn(R, C).bfloat16(), R * C * 2, 1 << 30, cap=64)
        x = torch.randn(1, C)
        y = torch.empty(1, R)
        src = torch.empty(256 << 20, dtype=torch.uint8).pin_memory()
        dst = torch.empty(256 << 20, dtype=torch.uint8, device=self.dev)
        stop = threading.Event()
        moved = [0, 0.0]

        def pump() -> None:
            with torch.cuda.stream(self.copy):
                t0 = time.perf_counter()
                while not stop.is_set():
                    dst.copy_(src, non_blocking=True)
                    self.copy.synchronize()
                    moved[0] += src.numel()
                moved[1] = time.perf_counter() - t0

        def cpu_rate(seconds: float = 2.0) -> float:
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                Native.gemv(cpu.next(), x, y)
                n += 1
            return n * R * C * 2 / (time.perf_counter() - t0) / 1e9

        def bus_alone(seconds: float = 2.0) -> float:
            stop.clear()
            moved[:] = [0, 0.0]
            th = threading.Thread(target=pump)
            th.start()
            time.sleep(seconds)
            stop.set()
            th.join()
            return moved[0] / moved[1] / 1e9

        for rep in range(3):
            alone_cpu = cpu_rate()
            alone_bus = bus_alone()
            stop.clear()
            moved[:] = [0, 0.0]
            th = threading.Thread(target=pump)
            th.start()
            both_cpu = cpu_rate()
            stop.set()
            th.join()
            both_bus = moved[0] / moved[1] / 1e9
            rec = {
                "op": "interference",
                "rep": rep,
                "cpu_gbps_alone": alone_cpu,
                "bus_gbps_alone": alone_bus,
                "cpu_gbps_both": both_cpu,
                "bus_gbps_both": both_bus,
            }
            self.records.append(rec)
            print(
                f"interference rep {rep}: cpu {alone_cpu:.1f} -> {both_cpu:.1f} GB/s, bus {alone_bus:.1f} -> "
                f"{both_bus:.1f} GB/s, together {both_cpu + both_bus:.1f}",
                flush=True,
            )

    def threads(self) -> None:
        """the one-row matvec's rate against the worker count (the 13700K's 8 P-cores, 16 cores, 24 threads)"""
        R, C = 2560, 5120
        cpu = Pool(lambda: torch.randn(R, C).bfloat16(), R * C * 2, 1 << 30, cap=64)
        x = torch.randn(1, C)
        y = torch.empty(1, R)
        for n in (4, 8, 12, 16, 24, 0):
            Native.load_gemv(self.native, threads=n)
            res = measure({f"t{n}": lambda: Native.gemv(cpu.next(), x, y)})
            res[f"t{n}"]["gbps"] = R * C * 2 / res[f"t{n}"]["s"] / 1e9
            self.add("threads", f"{R}x{C}", 1, res, {})
        Native.load_gemv(self.native)


def host() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "card": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        info["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        pass
    return info


CPU_LANES = ("cpu_native", "cpu_group", "cpu_torch_f32", "cpu_torch_bf16", "cpu")
FED_LANES = ("card_stream_pinned", "card_stream_pageable", "card_stream_piped")  # weights from RAM each call
HELD_LANES = ("card_resident", "card_btb_fp32", "card_btb_mma")  # weights in VRAM, the rows crossing the edge


def report(path: str) -> None:
    """the bench's JSON as tables: each point's lanes in ms, and where the CPU's best beats the card's best with the
    weights in RAM (fed) and with them in VRAM (held)"""
    with open(path) as f:
        doc = json.load(f)
    h = doc["host"]
    print(f"# ops on {h.get('processor')} / {h.get('card')} (torch {h.get('torch')}, cuda {h.get('cuda')})\n")
    recs = doc["records"]
    pts: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for r in recs:
        if "lane" not in r:
            continue
        pts.setdefault((r["op"], str(r["shape"])), {}).setdefault(int(r["rows"]), {})[r["lane"]] = r["s"]
    for (op, shape), by_rows in pts.items():
        lanes = sorted({ln for v in by_rows.values() for ln in v}, key=lambda s: (not s.startswith("cpu"), s))
        print(f"## {op} {shape}\n")
        print("| rows | " + " | ".join(lanes) + " | best | cpu vs fed | cpu vs held |")
        print("|---" * (len(lanes) + 4) + "|")
        for rows in sorted(by_rows):
            v = by_rows[rows]
            cells = [f"{v[ln] * 1e3:.3f}" if ln in v else "" for ln in lanes]
            best = min(v, key=v.get)  # type: ignore[arg-type]
            cpu = min((v[ln] for ln in CPU_LANES if ln in v), default=None)
            fed = min((v[ln] for ln in FED_LANES if ln in v), default=None)
            held = min((v[ln] for ln in HELD_LANES if ln in v), default=None)
            vs = lambda o: f"{o / cpu:.2f}x" if (cpu and o) else ""
            print(f"| {rows} | " + " | ".join(cells) + f" | {best} | {vs(fed)} | {vs(held)} |")
        print()
    for r in recs:
        if r.get("op") == "interference":
            print(
                f"interference {r['rep']}: cpu {r['cpu_gbps_alone']:.1f} -> {r['cpu_gbps_both']:.1f} GB/s, "
                f"bus {r['bus_gbps_alone']:.1f} -> {r['bus_gbps_both']:.1f} GB/s"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native")
    ap.add_argument("--kernels")
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma list of sections")
    ap.add_argument("--report", help="print the tables of a bench's JSON and exit")
    ap.add_argument("--routing", help="a bench/route_trace.py trace, for the express section")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return
    from btb.native_files import kernels_path, native_path

    a.native = a.native or native_path()
    a.kernels = a.kernels or kernels_path()
    if not (a.native and a.out):
        ap.error("--out is required, and --native where this tree has no built library")
    import btb

    print(f"[ops] btb from {btb.__file__}; native {a.native} ({sha(a.native)})", flush=True)
    random.seed(0)
    torch.manual_seed(0)
    b = Bench(a.native, a.kernels, a.quick)
    q = a.quick
    rows_lin = [1, 4, 16, 64, 256] if q else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rows_exp = [1, 8, 64] if q else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    def linear() -> None:
        b.linear("qkv", 6144, 2560, rows_lin)
        b.linear("down", 2560, 9728, rows_lin)
        if not q:
            b.linear("down-wide", 5120, 25600, [r for r in rows_lin if r <= 1024])
        b.linear("head", 151936, 2560, [1, 4, 16] if not q else [1])

    def fine() -> None:
        b.expert_bf16("moe", 2560, 640, [12, 16, 20, 24, 28, 32, 40, 48])
        b.linear("qkv", 6144, 2560, [12, 16, 20, 24, 28, 32, 40, 48])
        b.linear("down", 2560, 9728, [12, 16, 20, 24, 28, 32, 40, 48])

    sections: dict[str, Callable[[], None]] = {
        "transfers": b.transfers,
        "linear": linear,
        "expert": lambda: b.expert_bf16("moe", 2560, 640, rows_exp),
        "expert_mx4": lambda: b.expert_mxfp4("oss", 2880, 2880, rows_exp),
        "attn": lambda: b.attn_decode(32, 8, 128, [1024, 8192] if q else [1024, 4096, 16384, 65536, 131072]),
        "glue": lambda: b.glue(2560, [1, 16] if q else [1, 16, 256, 4096]),
        "interference": b.interference,
        "threads": b.threads,
        # the crossover's neighbourhood, row by row (not in a default run)
        "fine": fine,
        # a recorded prefill's expert calls (bench/route_trace.py), with --routing
        "express": lambda: b.express(a.routing, 2560, 640, [4, 8, 12, 16, 20, 24, 32, 48, 64]),
    }
    default_off = {"fine", "express"}
    only = [s for s in a.only.split(",") if s]
    t0 = time.time()
    for name, fn in sections.items():
        if (only and name not in only) or (not only and name in default_off):
            continue
        print(f"== {name}", flush=True)
        fn()
        with open(a.out, "w") as f:
            json.dump(
                {
                    "host": host(),
                    "native": {"path": a.native, "sha": sha(a.native)},
                    "kernels": {"path": a.kernels, "sha": sha(a.kernels)} if a.kernels else None,
                    "records": b.records,
                },
                f,
                indent=1,
            )
    print(f"[ops] {len(b.records)} records in {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
