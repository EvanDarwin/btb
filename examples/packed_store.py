# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The 12-bit model. `btb.pack_model` writes a model's layers packed as a model of its own beside the source
(`<model>-pack12`; for a Hugging Face repo a `<repo>-pack12` entry in the cache), its config naming the parent
every other tensor is read from. It loads like any model: pass it as the path. The arithmetic is bf16 as before
and the tokens are the same; `btb.pack12` reads the format with torch alone.
"""

import argparse
import os
import tempfile

import btb


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda", choices=("cuda", "mlx", "cpu"))
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--out", default=None, help="where the 12-bit model is written (default: a temp directory)")
    a = ap.parse_args(argv)
    src = btb.resolve(a.model)
    out = a.out or os.path.join(tempfile.mkdtemp(prefix="btb-"), os.path.basename(src) + "-pack12")
    out = btb.pack_model(src, out, log=print)
    packed_bytes = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".safetensors"))
    print(f"packed the layers into {packed_bytes / 2**20:.1f} MB of shards at {out}")
    with btb.load(src, device=a.device, v_max=0) as plain:
        ids = plain.prompt_ids("From the checkpoint and from the store:")
        base = plain.generate(ids, a.new, greedy=True).tokens
    with btb.load(out, device=a.device, v_max=0) as packed:
        from_store = packed.generate(ids, a.new, greedy=True).tokens
        print(f"identical: {from_store == base}; the parent it names: {btb.pack_format(out)['source']}")
    return {"out": out, "identical": from_store == base, "packed_bytes": packed_bytes, "format": btb.pack_format(out)}


if __name__ == "__main__":
    main()
