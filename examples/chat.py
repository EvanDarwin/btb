# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
A conversation. `model.chat()` keeps the history and the session the cache lives in, so each turn pays
only for its new tokens; `chat.last["reused"]` is how many positions came back from the cache.
"""

import argparse

import btb

TURNS = ("Pick a colour.", "Now a fruit of that colour.", "Why that one?")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=48)
    a = ap.parse_args(argv)
    reused = []
    with btb.load(a.model, device=a.device) as model:
        chat = model.chat(max_new=a.new)
        for text in TURNS:
            answer = chat.ask(text)
            reused.append(chat.last.get("reused", 0))
            print(f"> {text}\n{answer}\n  ({chat.last['prompt']} prompt tokens, {reused[-1]} reused from the cache)")
        return {"reused": reused, "history": chat.history}


if __name__ == "__main__":
    main()
