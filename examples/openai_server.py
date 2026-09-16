# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The OpenAI-compatible server inside your own process. `btb.serve.start` binds the port and loads the
model; as a `with` block it serves on a background thread and stops when the block ends. The endpoints
are `btb serve`'s: `/v1/models`, `/v1/chat/completions` (streamed or not), Ollama's `/api/chat`.
"""

import argparse
import json
import urllib.request

import btb.serve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--port", type=int, default=0, help="0: any free port")
    a = ap.parse_args(argv)
    with btb.serve.start(a.model, port=a.port, device=a.device, max_new=a.new) as server:
        name = server.registry.primary_name
        body = {
            "model": name,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": a.new,
        }
        req = urllib.request.Request(
            f"{server.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.load(urllib.request.urlopen(req))
        text = resp["choices"][0]["message"]["content"]
        print(f"{name} at {server.url}: {text!r} {resp.get('usage')}")
        return {"model": name, "text": text, "usage": resp.get("usage"), "url": server.url}


if __name__ == "__main__":
    main()
