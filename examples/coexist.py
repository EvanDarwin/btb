# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Your own tensors beside the engine in one process. Ask the gate before a large allocation of your own
(`model.scheduler.grant`, refused by name when it does not fit), and hold the memory under a name while
you use it (`model.reserve`), so the policy does not shed layers to make room you are about to take. The
ledger (`model.report()["granted"]`) then shows both sides.
"""

import argparse

import torch

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--mb", type=int, default=256, help="the working set of your own, in MiB")
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        nbytes = a.mb << 20
        dev = "cuda" if model.dev.type == "cuda" else "cpu"
        model.scheduler.grant(nbytes, "scratch", requester="my working set", device=dev)
        with model.reserve("my working set", nbytes, device=dev):
            mine = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
            mine.fill_(1.0)
            text = model.ask("What is a working set?", max_new=a.new)
            print(text)
            held = model.device.reserved(dev)
        after = model.device.reserved(dev)
        ledger = model.report()["granted"]
        print(f"held {held / 2**20:.0f} MiB during the block, {after / 2**20:.0f} after; ledger {ledger}")
        return {"text": text, "held": held, "after": after, "ledger": ledger}


if __name__ == "__main__":
    main()
