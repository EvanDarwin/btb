# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
A script for benchmarking btb against other engines in their own environment, currently
supported: mlx-lm, airllm, llama-cpp.

A JSON file is created in the results/ root with the machine's name, and current
date/time; and also creates a folder within results/ that contains the logs and jsonl
results for each individual run.

A venv should be set up at `.venv-compare` to organize all the vendor deps from the
project's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib import COMPARE_VENV, say
from lib.records import BenchRecord
from lib.tools import TOOLS, bench, prompts


def main(argv: Sequence[str] | None = None) -> int:
    """Comparison benchmark entrypoint"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="model directory")
    ap.add_argument("--tool", required=True, choices=tuple(TOOLS))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--rows", default="")
    ap.add_argument("--new", default="64,256,1024")
    ap.add_argument("--out", required=True, help="append the record here (JSON lines)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--dtype", default=None, help="airllm: float32 or bfloat16 (default: the model's own)")
    ap.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="airllm: the device to run on, set by the matrix to match the column under test (the card for the "
        "gpu column, the CPU for the cpu column); llama-cpp expresses the same through --n-gpu-layers",
    )
    ap.add_argument(
        "--airllm-split-dir",
        default=None,
        help="airllm: where the per-layer splits are written (default: BTB_AIRLLM_DIR or F:\\_airllmcache\\<model>)",
    )
    ap.add_argument(
        "--keep-splits",
        action="store_true",
        help="airllm: keep the splits this run created (default: remove them after)",
    )
    ap.add_argument(
        "--budget", type=float, default=1800.0, help="seconds after which the remaining lengths are left out"
    )
    ap.add_argument("--gguf", default=None, help="llama-cpp: an existing GGUF to time (default: convert `path` to one)")
    ap.add_argument(
        "--gguf-dir",
        default=None,
        help="llama-cpp: where a converted GGUF is written (default: BTB_GGUF_DIR or F:\\_ggufcache)",
    )
    ap.add_argument(
        "--gguf-type", default="bf16", help="llama-cpp: convert outtype (bf16 keeps the stored weights; f16, q8_0, ...)"
    )
    ap.add_argument(
        "--keep-gguf", action="store_true", help="llama-cpp: keep a GGUF this run converted (default: remove it after)"
    )
    ap.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="llama-cpp: layers offloaded to the card (-1 all, 0 CPU-only)"
    )
    ap.add_argument(
        "--llama-cpp-dir",
        default=None,
        help="llama-cpp: the llama.cpp checkout with convert_hf_to_gguf.py (default: LLAMA_CPP_DIR or F:\\llama.cpp)",
    )
    a = ap.parse_args(argv)
    log = lambda s: say(s, "compare")
    # WARN: the rivals' deps live in .venv-compare; another interpreter is a different measurement
    if os.path.basename(sys.prefix) != os.path.basename(COMPARE_VENV):
        log(f"not running from .venv-compare ({sys.prefix}); see bench/requirements.txt")
    counts = [int(x) for x in a.new.split(",") if x.strip()]
    tool = TOOLS[a.tool](
        a.path,
        log,
        dtype=a.dtype,
        split_dir=a.airllm_split_dir,
        keep_splits=a.keep_splits,
        device=a.device,
        gguf=a.gguf,
        gguf_dir=a.gguf_dir,
        keep_gguf=a.keep_gguf,
        ngl=a.n_gpu_layers,
        gguf_type=a.gguf_type,
        llama_cpp_dir=a.llama_cpp_dir,
        max_new=max(counts),
    )
    try:
        tool.load()
        cells = bench(tool, prompts(a.prompts, a.rows or None), counts, a.budget)
    finally:
        tool.close()
    for c in cells:
        log(
            f"{a.tool} new={c['new']}: first token {c['first_s']:.2f}s; {c['base_s_tok']:.3f} s/token = "
            f"{1 / c['base_s_tok']:.2f} tok/s; peak RAM {c['peak_ram_gb']:.1f} GB, peak MLX {c['peak_vram_gb']:.1f} GB"
        )
    from btb.hf import cache_repo_id  # after the rival has run: the checkout's btb is last on the path

    model = cache_repo_id(a.path) or os.path.basename(os.path.normpath(a.path))
    # the matrix owns the axes (tool and its regime are the cell's own); the record carries only the numbers
    rec = BenchRecord(label=a.label or a.tool, path=a.path, model=model, cells=cells)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
