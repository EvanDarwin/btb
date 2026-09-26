# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Memory of your own beside the model. `model.memory()` says what each device has free and what btb could give up.
`model.empty(shape)` (and `zeros`, `full`) is for a tensor you make: btb gives up what it holds there first,
cheapest first, instead of your allocation running out, and takes the room back once the tensor is gone.
`model.room(nbytes)` is for memory btb does not allocate - a second model, a library's workspace - kept from btb
until released (a `with` block); tensors taken through it (`room.zeros`) count against it, not beside it. Either
refuses by name (`MemoryGrantError`) when even giving everything up is not enough.
"""

import argparse

import torch

import btb

MiB = 1 << 20


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        where = str(model.dev) if model.dev.type == "cuda" else "cpu"
        before = model.memory()[where]
        print(f"{where}: {before.free / MiB:.0f} MiB free, {before.sheddable / MiB:.0f} MiB btb could give up")
        mine = model.zeros((256, 1024), dtype=torch.float32)  # 1 MiB of your own, a plain tensor
        with model.room(8 * MiB, name="workspace") as room:  # memory another library will allocate
            part = room.zeros(2 * MiB, dtype=torch.uint8)  # a tensor inside the room
            held = model.memory()[where].reserved  # what the room keeps from btb: what its tensors have not taken
            print(f"the room holds {held / MiB:.0f} MiB beside its {room.used / MiB:.0f} MiB tensor")
            del part
        after = model.memory()[where].reserved
        print(f"released: {after / MiB:.0f} MiB kept from btb; your tensor is {tuple(mine.shape)}")
        try:
            model.empty(1 << 50, dtype=torch.uint8)
            refused = None
        except btb.MemoryGrantError as e:
            refused = str(e)
            print(f"refused: {refused}")
        return {"held": held, "after": after, "mine": tuple(mine.shape), "refused": refused}


if __name__ == "__main__":
    main()
