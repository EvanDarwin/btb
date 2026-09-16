# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
Speculation from text you already hold. `btb.SpanBank` banks token sequences (a document the answer will
quote, an earlier answer, a tool's output), `generate(spans=...)` hands them to the n-gram proposer as
drafts, and the verify pass keeps only what the greedy loop would have produced: the tokens are identical,
the stats say how many came from the drafts.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=48)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        ids = model.prompt_ids("Repeat the passage exactly as written.")
        greedy = model.generate(ids, a.new, greedy=True).tokens
        bank = btb.SpanBank()
        bank.add("the passage", greedy)  # what the answer will say, banked ahead; a real use banks the document quoted
        spec, stats = model.generate(ids, a.new, spans=bank.spans())
        print(model.tokenizer.decode(spec, skip_special_tokens=True))
        print(
            f"identical to the greedy answer: {spec == greedy}; {stats['accepted']} of {stats['proposed']} drafted "
            f"tokens accepted over {stats['forwards']} passes"
        )
        return {"identical": spec == greedy, "stats": stats, "tokens": spec}


if __name__ == "__main__":
    main()
