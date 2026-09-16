# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The memory policies' own moves, by hand. On the host tier a warm layer is shed to the ring (its weights
read from the store each pass, its RAM given back) and taken back: `ram_shed` / `ram_regrow`, what
`ram_policy` does under the OS's pressure signal. On a card a resident layer is shed to the host and
regrown: `vram_shed` / `vram_regrow`, what `vram_policy` does when another program takes the card. The
answer is the same at every step: a layer's tier is a placement, never a different computation.

WARN: the host move needs a 12-bit model (`btb pack <model>` writes `<model>-pack12`; pass it as --model)
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=12)
    a = ap.parse_args(argv)
    path = btb.resolve(a.model)
    packed = btb.pack_format(path) is not None
    with btb.load(path, device=a.device, v_max=0) as model:
        ids = model.prompt_ids("The same answer, wherever the layer lives:")
        decode = lambda toks: model.tokenizer.decode(toks, skip_special_tokens=True)
        before = model.generate(ids, a.new, greedy=True).tokens
        print(f"{'from RAM:':<26} {decode(before)}")
        moves = []
        if model.host and packed:
            i = model.ram_shed("the example")
            during = model.generate(ids, a.new, greedy=True).tokens
            print(f"{f'layer {i} on the ring:':<26} {decode(during)}")
            back = model.ram_regrow()
            after = model.generate(ids, a.new, greedy=True).tokens
            print(f"{f'layer {back} back in RAM:':<26} {decode(after)}")
            moves.append(("ram", i, during == before, back, after == before))
        elif not packed:
            print("no 12-bit store beside the model: the host move is skipped")
        if model.resident and model.dev.type == "cuda":
            what = model.vram_shed()
            during = model.generate(ids, a.new, greedy=True).tokens
            print(f"{f'{what} off the card:':<26} {decode(during)}")
            model.vram_regrow()
            after = model.generate(ids, a.new, greedy=True).tokens
            print(f"{'regrown:':<26} {decode(after)}")
            moves.append(("vram", what, during == before, None, after == before))
        return {"before": before, "moves": moves}


if __name__ == "__main__":
    main()
