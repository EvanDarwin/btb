# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
What you can hook into a decode. `processors` rewrite every pick's logits - `(ids, logits) -> logits`, called on
every row a speculative pass verifies too, so a decode is the same with speculation or without; `logprobs` returns
each token's log-probability and its `k` likeliest alternatives; `taps` the chosen layers' state at each new token;
`on_pass` is called with each pass's counts. What comes back is a `Generation`: it unpacks as `(tokens, stats)` and
carries `logprobs`, `hidden` and `report` (the paths the decode took). A callback runs between two steps of the
decode: it may read (the model's `memory()`), not call back into the model.
"""

import argparse
import math

import torch

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=12)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        ids = model.prompt_ids("Write one sentence about the sea.")
        plain = list(model.generate(ids, a.new, eos=()).tokens)
        banned = plain[0]  # the token the plain answer opens with, never drawn below

        def never(_ids, logits: torch.Tensor) -> torch.Tensor:
            out = logits.clone()
            out[banned] = -math.inf
            return out

        passes = []
        last = model.L - 1
        g = model.generate(ids, a.new, eos=(), processors=[never], logprobs=2, taps=[last], on_pass=passes.append)
        tokens, stats = g  # a Generation unpacks as the pair
        print(model.tokenizer.decode(tokens, skip_special_tokens=True))
        for t, lp in list(zip(tokens, g.logprobs))[:3]:
            alts = ", ".join(f"{model.tokenizer.decode([i])!r} {v:.2f}" for i, v in lp.top)
            print(f"{model.tokenizer.decode([t])!r}: {lp.logprob:.2f} (alternatives {alts})")
        print(f"{len(passes)} passes for {len(tokens)} tokens; the last layer's states: {tuple(g.hidden[last].shape)}")
        return {
            "banned": banned,
            "tokens": list(tokens),
            "stats": stats,
            "logprobs": g.logprobs,
            "hidden": tuple(g.hidden[last].shape),
            "passed": sum(p["tokens"] for p in passes),
            "report": g.report,
        }


if __name__ == "__main__":
    main()
