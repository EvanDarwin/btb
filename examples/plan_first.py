# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The placement before the load. `btb.plan` prices the model against the memory free now: where each
layer would run (the card, RAM, streamed from the drive each pass), what a token would cost, what the
host budget keeps free. A machine that cannot carry the least working set is refused by name
(`btb.PlanError`) before a byte is loaded. Then the load, with the placement the plan chose or your own.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=0)
    a = ap.parse_args(argv)
    plan = btb.plan(a.model, device=a.device)
    print(plan)
    if plan.budget is not None:
        b = plan.budget
        print(
            f"host budget: {b.available / 2**30:.1f} GB available, {b.spendable / 2**30:.1f} GB spendable above the floor"
        )
    refused = None
    try:
        # a floor past what the machine has: a box that cannot hold the least working set
        btb.plan(a.model, device=a.device, os_reserve_gb=plan.caps.ram_gb + 1)
    except btb.PlanError as e:
        refused = str(e)
        print("refused, as it should be:", refused[:120], "...")
    # the plan's own placement, or one you name: here the plan's host share on the CPU kernels
    cpu_layers = len(plan.host) if a.device != "cpu" else None
    with btb.load(a.model, device=a.device, cpu_layers=cpu_layers) as model:
        print("loaded:", model.report_line(model.report()))
        return {"plan": plan, "refused": refused, "host": sorted(model.host), "resident": sorted(model.resident)}


if __name__ == "__main__":
    main()
