# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a grouped expert call costs on this card, piece by piece, at gpt-oss-120b's shapes: torch's `_grouped_mm`
(a native grouped GEMM only on sm_90 to sm_110; ATen's fallback below that), a plain `mm` per expert over the same
sorted rows, and a padded `bmm`; and the widening of one batch of MXFP4 experts (`dequant_blocks`). Each product's
bits against `_grouped_mm`'s, so a replacement is known to be the same arithmetic before it is timed.

    python bench/grouped_mm_sm89.py [--experts 16] [--rows 8,32,128] [--reps 20]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

import torch

# this checkout's btb, not whichever one the interpreter has installed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from btb.engine.native import Native  # noqa: E402
from btb.mxfp4_torch import dequant_blocks  # noqa: E402

H, I = 2880, 2880  # gpt-oss-120b: hidden, expert intermediate (gate_up is 2 * I wide)


def timed(fn: Callable[[], object], reps: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--rows", default="8,32,128")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)
    n = args.experts
    print(
        f"{torch.cuda.get_device_name(dev)} sm_{''.join(map(str, torch.cuda.get_device_capability(dev)))}, torch {torch.__version__}"
    )

    # the widening: n experts' gate_up and down as stored (MXFP4 blocks of 32 values, a scale each) to bf16
    shapes = [(2 * I, H), (H, I)]
    stored = [
        (
            torch.randint(0, 256, (n, r, c // 32, 16), dtype=torch.uint8, device=dev),
            torch.randint(118, 130, (n, r, c // 32), dtype=torch.uint8, device=dev),
        )
        for r, c in shapes
    ]
    ms = timed(lambda: [dequant_blocks(b, s).reshape(b.shape[0], b.shape[1], -1) for b, s in stored], args.reps)
    gb = sum(b.numel() + s.numel() for b, s in stored) / 1e9
    wide = n * 3 * H * I * 2 / 1e9
    print(f"widen {n} experts (MXFP4 -> bf16), torch: {ms:.2f} ms ({gb:.2f} GB as stored, {wide:.2f} GB widened)")
    kern = Native.card_kernels()
    if kern is not None:
        # the card's one pass over a depot's stacks, the experts taken at seats as a wave takes them
        seats = torch.arange(n, dtype=torch.int32, device=dev)
        km = timed(lambda: [kern.mx4_widen(b, s, seats) for b, s in stored], args.reps)
        same = all(
            torch.equal(
                kern.mx4_widen(b, s, seats).view(torch.int16), dequant_blocks(b, s).reshape(n, -1).view(torch.int16)
            )
            for b, s in stored
        )
        print(
            f"widen {n} experts, btb_mx4_widen: {km:.2f} ms, {(gb + wide) * 1000 / km:.0f} GB/s moved, "
            f"{ms / km:.1f}x{'' if same else ' (bits differ)'}"
        )
    else:
        print(f"no card kernels: {Native.cuda_reason}")
    gu = dequant_blocks(*stored[0]).reshape(n, 2 * I, H)
    for rows in (int(r) for r in args.rows.split(",")):
        print(f"gate_up over {n} experts x {rows} rows: " + _gate_up(gu, rows, args.reps))


def _gate_up(gu: torch.Tensor, rows: int, reps: int) -> str:
    """gate_up's product over `rows` rows an expert, four ways, each timed and its bits held to `_grouped_mm`'s"""
    n, dev = int(gu.shape[0]), gu.device
    M = n * rows
    x = torch.randn(M, H, dtype=torch.bfloat16, device=dev)
    ends = torch.tensor([rows * (e + 1) for e in range(n)], dtype=torch.int32, device=dev)
    wt = gu.transpose(1, 2)  # what `_card_grouped` hands it: [n, H, 2I] as a view

    def grouped() -> torch.Tensor:
        return torch._grouped_mm(x, wt, offs=ends)

    def per_expert() -> torch.Tensor:
        out = torch.empty(M, 2 * I, dtype=torch.bfloat16, device=dev)
        for e in range(n):
            torch.mm(x[e * rows : (e + 1) * rows], wt[e], out=out[e * rows : (e + 1) * rows])
        return out

    def per_expert_linear() -> torch.Tensor:
        out = torch.empty(M, 2 * I, dtype=torch.bfloat16, device=dev)
        for e in range(n):
            out[e * rows : (e + 1) * rows] = torch.nn.functional.linear(x[e * rows : (e + 1) * rows], gu[e])
        return out

    def padded() -> torch.Tensor:
        return torch.bmm(x.view(n, rows, H), wt)

    ref = grouped()
    line = []
    for name, fn in (
        ("_grouped_mm", grouped),
        ("mm/expert", per_expert),
        ("linear/expert", per_expert_linear),
        ("bmm padded", padded),
    ):
        t = timed(fn, reps)
        same = torch.equal(fn().reshape(M, -1), ref)
        line.append(f"{name} {t:.2f} ms{'' if same else ' (bits differ)'}")
    return ", ".join(line)


if __name__ == "__main__":
    main()
