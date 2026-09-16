# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
One answer, then one streamed. `btb.load` places the model across the machine's tiers from the memory
free now; `ask` decodes with the engine's configured loop (speculative where it pays, the greedy answer
either way); `stream` releases text as it is decoded, whole across multi-byte tokens.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=64)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        text = model.ask("Name three uses of a paperclip.", max_new=a.new)
        print(text)
        stream = model.stream("Now name a fourth.", max_new=a.new)
        for piece in stream:
            print(piece, end="", flush=True)
        print(f"\n[{len(stream.tokens)} tokens in {stream.stats.get('forwards', '?')} passes]")
        return {"text": text, "streamed": stream.tokens, "stats": stream.stats}


if __name__ == "__main__":
    main()
