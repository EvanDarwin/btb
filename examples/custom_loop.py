# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Your own tokens and your own loop. The engine is ids in, ids out: `generate` takes any ids and any stop
ids, and `forward` with a cache gives the next token's logits with the model placed across the tiers as it
was loaded, the swapping policies included. Here: a first step constrained to an allowed set (a grammar,
a tool name, a choice), then temperature sampling with a seed.
"""

import argparse

import torch

import btb

CHOICES = (" cherry", " orange", " kiwi")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        tok = model.tokenizer  # or assign your own: model.tokenizer = mine
        ids = tok("Continue the list: apple, banana,", add_special_tokens=False)["input_ids"]
        allowed = [int(tok.encode(w, add_special_tokens=False)[0]) for w in CHOICES]
        stop = set(model.stop_ids)
        g = torch.Generator().manual_seed(a.seed)
        out = []
        with torch.inference_mode():
            cache = model.new_cache()
            logits = model.forward([ids], cache=cache)[0, -1].float().cpu()
            for step in range(a.new):
                if step == 0:
                    pick = allowed[int(torch.argmax(logits[allowed]))]
                else:
                    pick = int(torch.multinomial(torch.softmax(logits / a.temperature, dim=-1), 1, generator=g))
                out.append(pick)
                if pick in stop or step + 1 == a.new:
                    break
                logits = model.forward([[pick]], cache=cache)[0, -1].float().cpu()
        print(tok.decode(out, skip_special_tokens=True))
        print(f"first token from {allowed}: {out[0]}; the cache holds {cache.get_seq_length()} positions")
        return {"prompt": ids, "tokens": out, "allowed": allowed, "positions": int(cache.get_seq_length())}


if __name__ == "__main__":
    main()
