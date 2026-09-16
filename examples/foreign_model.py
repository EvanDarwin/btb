# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The memory system over a model that is not btb's. `btb.BatchScheduler.for_model` prices the cache from a
transformers model's config and measures the host as a plan would; then your own decode has the grant
gate (`grant`, refused by name), the epoch sizing (`plan` / `release`) and the KV pricing
(`kv_row_bytes`) over it. The layer swapping is the engine's own and does not apply here.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--new", type=int, default=256)
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(btb.resolve(a.model))
    sched = btb.BatchScheduler.for_model(hf, device=a.device, ram_reserve_gb=1.0)
    row = sched.kv_row_bytes(1024 + a.new)
    print(f"one sequence's cache at {1024 + a.new} positions: {row / 2**20:.1f} MiB")
    free = sched.free_for(a.device)
    print(f"free on {a.device} above the reserve: {free / 2**30:.1f} GB")
    sched.grant(row * 4, "kv", requester="my four rows", device=a.device)
    refused = None
    try:
        sched.grant(free + (1 << 30), "kv", requester="my impossible cache", device=a.device)
    except btb.MemoryGrantError as e:
        refused = str(e)
        print(refused)
    batch, reserve = sched.plan(8, 1024 + a.new)
    print(f"an epoch for 8 pending sequences: {batch} rows at {reserve} positions; granted so far {sched.granted}")
    sched.release()
    return {"row_bytes": row, "batch": batch, "refused": refused, "granted": dict(sched.granted)}


if __name__ == "__main__":
    main()
