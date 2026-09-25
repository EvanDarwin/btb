# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Certification: the engine's greedy next token equals stock transformers' (a float32 `AutoModelForCausalLM`,
its KV in host RAM through `_host_cache`) at log-spaced checkpoints as the context grows to BTB_CERT_CTX
tokens. bf16 against float32: a disagreement is allowed only where the reference itself is at a near-tie the
size of bf16's rounding; any other fails. Skipped unless BTB_CERT or BTB_CERT_CTX is set.

    BTB_CERT=1 .venv/Scripts/python -m pytest tests/test_context_cert.py -s     # the 32k default, minutes
    BTB_CERT_CTX=131072 ...                                                     # 128k
    BTB_CERT_CTX=4096 ...                                                       # a quick pass
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest
import torch

from btb.kinds import Json
from tests.helpers import NO_LOG

if TYPE_CHECKING:
    from transformers import DynamicCache, PretrainedConfig


REQUESTED = os.environ.get("BTB_CERT") or os.environ.get("BTB_CERT_CTX")
CTX = int(os.environ.get("BTB_CERT_CTX", "32768"))
CHUNK = int(os.environ.get("BTB_CERT_CHUNK", "4096"))
MODEL = os.environ.get("BTB_CERT_MODEL", "qwen3-0.6b")
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("BTB_CERT_DTYPE", "fp32")]
# the reference is fed in smaller steps than the engine so its attention scores (one [step x context] block a
# head, materialized by sdpa under a mask) stay small while its KV is off in host RAM
REF_CHUNK = int(os.environ.get("BTB_CERT_REF_CHUNK", "512"))
# a greedy disagreement is allowed only where the reference's own top-two gap is this small (bf16's rounding at
# logit magnitudes of order ten; a real divergence opens a far wider gap)
TIE_TOL = float(os.environ.get("BTB_CERT_TIE_TOL", "0.1"))

pytestmark = [
    pytest.mark.skipif(not REQUESTED, reason="set BTB_CERT=1 (heavy, on-demand certification)"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def _model_path() -> str | None:
    import btb

    e = {x["name"]: x for x in btb.available_models()}.get(MODEL)
    return e["path"] if e else None


def _checkpoints(ctx: int) -> list[int]:
    cs = [c for c in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072) if c <= ctx]
    if not cs or cs[-1] != ctx:
        cs.append(ctx)
    return cs


def _ids(ctx: int, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, int(vocab), (ctx,), generator=g, dtype=torch.long)


def _rope_scaling(cfg: PretrainedConfig, ctx: int) -> Json | None:
    """the engine turns on YaRN when the context passes the checkpoint's native length; the reference must
    use the very same factor or the two would part on rope alone, not on any defect"""
    native = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    if not native or ctx <= native:
        return None
    factor = float(math.ceil(ctx / native))
    return {"rope_type": "yarn", "factor": factor, "original_max_position_embeddings": native}


def _btb_checkpoints(path: str, ids: torch.Tensor, ctx: int, checkpoints: Sequence[int]) -> list[torch.Tensor]:
    """the engine over the sequence, its next-token logits at each checkpoint; freed before it returns"""
    import btb

    kv_host = ctx > 49152  # the engine's own KV to host RAM past ~48k, where the bf16 KV would crowd the card
    sm = btb.load(path, device="cuda", context=ctx, kv_host=kv_host, log=NO_LOG)
    out = []
    try:
        with torch.inference_mode():
            cache = sm.new_cache()
            fed = 0
            for c in checkpoints:
                while fed < c:
                    step = min(CHUNK, c - fed)
                    lg = sm.forward(ids[fed : fed + step].tolist(), cache=cache)
                    fed += step
                assert lg is not None
                out.append(lg[0, -1].detach().float().cpu())
        return out
    finally:
        sm.close()
        torch.cuda.empty_cache()


def _host_cache() -> DynamicCache:
    """a transformers cache that keeps every layer's K/V in host RAM and lends the card only the layer being
    computed -- the engine's own "cold bytes in RAM, only what runs on the card", handed to torch. Subclasses
    the stock `DynamicCache` so it keeps that whole interface (masks, offsets, lengths); the one change is that
    each layer's growing store is pushed back to the CPU right after its attention, so the reference's full KV
    (float32, tens of GB at high context) never sits on the 12 GB card, only one layer's worth at a time."""
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    class HostCache(DynamicCache):
        def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            *args: object,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # a layer's store (transformers' DynamicLayer holds it as keys/values) moves to the card for its
            # attention and back to host RAM after it
            layer = self.layers[layer_idx] if layer_idx < len(self.layers) else None
            if isinstance(layer, DynamicLayer) and layer.is_initialized:
                assert layer.keys is not None and layer.values is not None
                layer.keys = layer.keys.to(key_states.device)
                layer.values = layer.values.to(key_states.device)
            k, v = super().update(key_states, value_states, layer_idx, *args, **kwargs)
            layer = self.layers[layer_idx]
            assert isinstance(layer, DynamicLayer) and layer.keys is not None and layer.values is not None
            layer.keys = layer.keys.to("cpu")
            layer.values = layer.values.to("cpu")
            return k, v  # this layer's K/V on the card for its attention; the store now back in host RAM

    return HostCache()


def _ref_checkpoints(path: str, ids: torch.Tensor, ctx: int, checkpoints: Sequence[int]) -> list[torch.Tensor]:
    """stock transformers, its next-token logits at each checkpoint. It runs on the card, but its KV is held in
    host RAM (`_host_cache`) so it cannot OOM the card at any length; the head keeps only the last row and the
    feed is chunked so the attention scores stay small."""
    from transformers import AutoConfig, AutoModelForCausalLM

    dev = torch.device("cuda")
    cfg = AutoConfig.from_pretrained(path)
    rs = _rope_scaling(getattr(cfg, "text_config", cfg), ctx)
    if rs is not None:
        tgt = getattr(cfg, "text_config", cfg)
        tgt.rope_scaling = rs
        tgt.max_position_embeddings = max(int(tgt.max_position_embeddings), ctx)
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=DTYPE).to(dev).eval()  # type: ignore[arg-type]  # transformers' partial stubs mistype .to(device)
    print(f"[cert] reference on cuda, KV in host RAM, {DTYPE} at {ctx}")
    out = []
    try:
        with torch.inference_mode():
            cache = _host_cache()
            fed = 0
            for c in checkpoints:
                while fed < c:
                    step = min(REF_CHUNK, c - fed)
                    chunk = ids[fed : fed + step].view(1, -1).to(dev)
                    pos = torch.arange(fed, fed + step, device=dev)
                    # only the last row's logits: the head over a whole chunk in float32 is gigabytes, and every
                    # position but the last is thrown away here anyway
                    o = model(
                        input_ids=chunk, past_key_values=cache, use_cache=True, cache_position=pos, logits_to_keep=1
                    )
                    fed += step
                out.append(o.logits[0, -1].detach().float().cpu())
        return out
    finally:
        del model
        torch.cuda.empty_cache()


def test_no_divergence_from_torch_over_a_long_context() -> None:
    path = _model_path()
    if path is None:
        pytest.skip(f"{MODEL} is not in the cache")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(path)
    vocab = int(getattr(cfg, "text_config", cfg).vocab_size)
    checkpoints = _checkpoints(CTX)
    ids = _ids(CTX, vocab)

    btb_lg = _btb_checkpoints(path, ids, CTX, checkpoints)
    ref_lg = _ref_checkpoints(path, ids, CTX, checkpoints)

    diverged: list[tuple[int, int, int, float]] = []
    worst = 0.0
    for c, b, r in zip(checkpoints, btb_lg, ref_lg):
        worst = max(worst, float((b - r).abs().max()))
        bt, rt = int(b.argmax()), int(r.argmax())
        if bt != rt:
            gap = float(r[rt] - r[bt])  # how much the reference itself prefers its token over the engine's
            (diverged if gap > TIE_TOL else []).append((c, bt, rt, gap))
        print(
            f"[cert] {c:>7} tokens: greedy {'ok' if bt == rt else f'ref={rt} btb={bt} gap={r[rt] - r[bt]:.3f}'}"
            f"  max|dlogit|={float((b - r).abs().max()):.3f}"
        )
    print(
        f"[cert] {MODEL} to {CTX} tokens: {len(checkpoints)} checkpoints, worst |dlogit| {worst:.3f}, "
        f"{len(diverged)} non-tie divergences"
    )
    assert not diverged, f"greedy token diverged from torch beyond a tie: {diverged}"
