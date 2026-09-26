# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Sessions decoded together. `model.batch(sessions)` steps a row a session as one batch, each row drawing what its
session's own decode would; `join(session)` adds a row between steps and `leave(row)` lets one go, written back
into its session at once. Leaving the `with` block (or `close()`) writes every row still in the batch back into
its session, which goes on from there. Meanwhile the sessions are the batch's: a feed on one is refused until then.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=4)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        prompts = [model.prompt_ids(q) for q in ("Say hello.", "Count to three.", "Name a colour.")]
        a_, b_ = model.session(prompts[0]), model.session(prompts[1])
        with model.batch([a_, b_]) as bt:
            first = bt.generate(a.new, eos=())  # a list of tokens a row
            c_ = model.session(prompts[2])
            joined = bt.join(c_)  # c's row number
            second = bt.generate(a.new, eos=())
            bt.leave(0)  # a goes free now, its rows written back
            refused = False
            try:
                b_.feed([1])
            except ValueError:
                refused = True  # b is still the batch's
        # the block's end wrote b's and c's rows back: every session goes on from its row's end
        for s in (a_, b_, c_):
            print(model.tokenizer.decode(s.tokens[-2 * a.new :], skip_special_tokens=True))
        return {
            "prompts": prompts,
            "first": first.tokens,
            "second": second.tokens,
            "joined": joined,
            "refused": refused,
            "sessions": [a_.tokens, b_.tokens, c_.tokens],
            "alone": [list(model.generate(p, 2 * a.new, eos=(), speculate=False).tokens) for p in prompts[:2]],
        }


if __name__ == "__main__":
    main()
