# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A mixture of experts' routing through real prefills: every expert call's picks, layer by layer, recorded off
the engine's own `expert_trace` while the model prefills the repo's own docs cut to each length. The picks are
what `bench/ops.py --routing` replays; the prefill's wall time and the store's counters ride along as today's
figure.

    python bench/route_trace.py MODEL --out trace.npz [--lens 64,256,1024,4096]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", default="64,256,1024,4096")
    ap.add_argument("--native")
    ap.add_argument("--sparse", type=int, choices=(0, 1), default=0, help="the engine's --sparse")
    a = ap.parse_args()
    import numpy as np
    import torch

    import btb

    print(f"[trace] btb from {btb.__file__}", flush=True)
    text = "\n\n".join(
        open(p, encoding="utf-8").read()
        for p in [os.path.join(ROOT, "README.md"), *sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")))]
    )
    runs, layers, offs, picks, meta = [], [], [0], [], []
    with btb.load(a.model, device="cuda", native=a.native, log=print, sparse=a.sparse) as m:
        ids = m.tokenizer(text, add_special_tokens=False)["input_ids"]
        print(f"[trace] {len(ids)} tokens of text", flush=True)
        for r, T in enumerate(int(x) for x in a.lens.split(",")):
            if T > len(ids):
                print(f"[trace] skip {T}: the text is {len(ids)} tokens", flush=True)
                continue
            m.expert_trace = []
            st0 = dict(m.expert_stat)
            store = getattr(m, "expert_store", None)
            ss0 = dict(store.stat) if store is not None else {}
            cache = m.new_cache()
            t0 = time.time()
            with torch.inference_mode():
                m._prefill(torch.tensor([ids[:T]]), cache)
            wall = time.time() - t0
            tr, m.expert_trace = m.expert_trace, None
            for layer, top in tr:
                runs.append(r)
                layers.append(int(layer))
                picks.append(top.reshape(-1, top.shape[-1]).to(torch.int16).numpy())
                offs.append(offs[-1] + picks[-1].shape[0])
            st = {
                k: (m.expert_stat[k] - st0.get(k, 0))
                for k in m.expert_stat
                if isinstance(m.expert_stat[k], (int, float))
            }
            # the store's own ledger over the prefill: misses are reads, `bytes` what they moved, `wait_s` the time
            # the calls stood waiting on the drive, `ahead*` the lookahead's predictions and their use
            ss = {
                k: round(store.stat[k] - ss0.get(k, 0), 3)
                for k in ("hit", "miss", "bytes", "wait_s", "ahead", "ahead_used", "ahead_dropped", "calls")
                if store is not None and isinstance(store.stat.get(k), (int, float))
            }
            meta.append({"T": T, "wall_s": wall, "calls": len(tr), "expert_stat": st, "store": ss})
            print(f"[trace] T={T}: prefill {wall:.2f}s, {len(tr)} expert calls, {st}; store {ss}", flush=True)
            del cache
        report = m.report()
    np.savez_compressed(
        a.out,
        run=np.array(runs, dtype=np.int32),
        layer=np.array(layers, dtype=np.int32),
        off=np.array(offs, dtype=np.int64),
        picks=np.concatenate(picks) if picks else np.zeros((0, 10), np.int16),
        meta=json.dumps({"runs": meta, "placement": report.get("placement"), "device": report.get("device")}),
    )
    print(f"[trace] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
