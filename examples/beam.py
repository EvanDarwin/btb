# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
A beam search of your own over a fork. `session.fork(n)` starts `n` rows at the session's end, sharing its cache;
`logits` holds the live rows' next-token logits [rows, V]; `reorder(rows)` re-forms the rows as copies of the
survivors (row j a copy of row rows[j], repeats allowed); `step(tokens)` feeds one token a row and returns the
logits after them. `keep(r)` writes row r into the session and closes the fork: the session goes on from the
beam's end as if it had been fed those tokens itself. A width of one is the greedy answer.
"""

import argparse

import torch

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--new", type=int, default=8)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        ids = model.prompt_ids("Finish the proverb: a stitch in time")
        s = model.session(ids)
        br = s.fork(a.width)
        scores = [0.0] * a.width
        for step in range(a.new):
            lp = torch.log_softmax(br.logits, dim=-1)  # [rows, V]
            rows = 1 if step == 0 else len(lp)  # the rows start alike: one of them proposes the first tokens
            cand = []
            for j in range(rows):
                v, t = lp[j].topk(a.width)
                cand += [(scores[j] + float(x), j, int(k)) for x, k in zip(v, t)]
            best = sorted(cand, reverse=True)[: a.width]
            br.reorder([j for _, j, _ in best])  # the survivors' rows, copied
            br.step([t for _, _, t in best])  # each fed its token
            scores = [sc for sc, _, _ in best]
        top = max(range(len(scores)), key=scores.__getitem__)
        kept = br.keep(top)  # the session now holds the prompt and the best beam, and goes on from there
        beam = kept.tokens[len(ids) :]
        print(model.tokenizer.decode(beam, skip_special_tokens=True))
        print(f"best of {a.width} beams: log-probability {scores[top]:.3f}")
        return {
            "tokens": beam,
            "score": scores[top],
            "greedy": list(model.generate(ids, a.new, eos=(), speculate=False).tokens),
            # the kept session goes on as one fed the prompt and the beam would
            "on": list(kept.generate(4, eos=(), speculate=False).tokens),
            "fed_on": list(model.session(ids + beam).generate(4, eos=(), speculate=False).tokens),
        }


if __name__ == "__main__":
    main()
