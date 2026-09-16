# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The memory system from the outside. `btb.host_budget()` reads the host as a plan would, with nothing
loaded. A load takes its reserves (`ram_reserve_gb`, `vram_reserve_gb`) and keeps a ledger
(`model.report()`: the bytes granted, the run's growth against the estimate). Every large allocation asks
the gate first (`model.scheduler.grant`), which refuses by name before anything is allocated instead of
letting torch OOM from inside the allocator.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=0)
    ap.add_argument("--ram-reserve-gb", type=float, default=1.0)
    a = ap.parse_args(argv)
    before = btb.host_budget()
    print(
        f"host: {before.available / 2**30:.1f} GB available of {before.total / 2**30:.0f}, floor {before.floor / 2**30:.2f} GB"
    )
    with btb.load(a.model, device=a.device, ram_reserve_gb=a.ram_reserve_gb) as model:
        b = model.plan.budget
        print(
            f"the engine keeps {model.ram_reserve / 2**30:.2f} GB free (named); the run's growth estimate is {b.growth / 2**30:.3f} GB"
        )
        rep = model.report()
        print("granted at load:", {k: f"{v / 2**20:.1f} MiB" for k, v in rep["granted"].items()})
        print(f"growth: {rep['growth']['past_plan_gb']:.3f} GB past the plan")
        free = model.scheduler.free_for("cpu")
        model.scheduler.grant(16 << 20, "scratch", requester="my 16 MiB buffer", device="cpu")
        print(f"granted 16 MiB of {free / 2**30:.1f} GB free on the host; ledger {model.scheduler.granted}")
        refused = None
        try:
            model.scheduler.grant(free + (1 << 30), "scratch", requester="my impossible buffer", device="cpu")
        except btb.MemoryGrantError as e:
            refused = str(e)
            print(refused)
        return {
            "before": before,
            "budget": b,
            "granted": dict(model.scheduler.granted),
            "refused": refused,
            "report": rep,
        }


if __name__ == "__main__":
    main()
