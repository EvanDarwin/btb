# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Many prompts at once. `ask_many` decodes ragged prompts as epochs the scheduler sizes from the memory
free for their cache (on a card, as many rows as the free VRAM holds at the longest prompt's final length;
on the host, every prompt at once), each epoch run to completion, one answer per prompt in input order.
"""

import argparse

import btb

PROMPTS = (
    "A short question?",
    "A much longer prompt that asks for a longer and more considered answer than the others do.",
    "Third.",
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=32)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device, v_max=0) as model:
        rows = [model.prompt_ids(p) for p in PROMPTS]
        batch, reserve = model.scheduler.plan(len(rows), max(map(len, rows)) + a.new)
        model.scheduler.release()
        print(f"the scheduler's epoch for these: {batch} rows, {reserve} positions of cache each")
        answers = model.ask_many(PROMPTS, max_new=a.new)
        for p, ans in zip(PROMPTS, answers, strict=True):
            print(f"> {p}\n{ans}")
        return {"answers": answers, "batch": batch}


if __name__ == "__main__":
    main()
