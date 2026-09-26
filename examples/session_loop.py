# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
A sequence you drive by hand. `model.session(ids)` feeds a prompt and keeps its cache; `feed(ids)` appends tokens
and returns the logits after each of them ([T, V]; with `last_only`, the last one's alone, [1, V]; with `taps`,
also the chosen layers' states, as `(logits, {layer: [T, H]})`); `mark()` remembers a point and `rewind(mark)` goes
back to it - the tokens and the cache as they were, a hybrid's recurrent states too. `generate` decodes on from
where the session stands. Its last token is drawn but not fed yet (`session.pending`): the next call feeds it
first, so nothing is computed twice and nothing is lost.
"""

import argparse

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=8)
    a = ap.parse_args(argv)
    with btb.load(a.model, device=a.device) as model:
        ids = model.prompt_ids("Name three rivers.")
        s = model.session(ids)  # the prompt fed; s.logits is the next token's, [V]
        first = int(s.logits.argmax())
        here = s.mark()
        tried = s.feed([first])  # [1, V]: the logits after the token fed
        _logits, states = s.feed([int(tried[-1].argmax())], taps=[-1])  # and the last layer's state at it, [1, H]
        s.rewind(here)  # back to the mark: the two tokens are gone from the tokens and the cache
        again = s.feed([first])  # the same logits as the first time
        s.rewind(here)
        g = s.generate(a.new, eos=(), speculate=False)  # decodes on from the mark
        print(model.tokenizer.decode(g.tokens, skip_special_tokens=True))
        print(
            f"{len(s)} tokens in the session, the last ({s.pending}) drawn and not fed yet; the feed after the "
            f"rewind gave the same logits: {bool((again == tried).all())}"
        )
        return {
            "prompt": ids,
            "tokens": list(g.tokens),
            "session": s.tokens,
            "pending": s.pending,
            "same": bool((again == tried).all()),
            "tap": tuple(next(iter(states.values())).shape),
            "hidden": int(model.cfg.hidden_size),
            "plain": list(model.generate(ids, a.new, eos=(), speculate=False).tokens),
        }


if __name__ == "__main__":
    main()
