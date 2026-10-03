# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A template or shadow pointed at another layer (`_retarget`) holds that layer's cache index in every submodule
that keeps one: Qwen4's sparse attention indexer and its PLE and n-gram convolutions carry their own, beside the
attention's."""

from __future__ import annotations

from btb.engine import StreamedTextModel
from tests.helpers import NO_LOG, fixture


def test_a_retargeted_layer_holds_no_other_layers_cache_index() -> None:
    """a template built from the first sparse-attention layer, pointed at the last: every cache index in it names
    the last - the indexer's too, which once kept the first's and wrote the last layer's keys into its cache"""
    StreamedTextModel.register_attention()
    sm = StreamedTextModel(fixture("tiny_q4"), resident_head=True, log=NO_LOG, device="cpu")
    try:
        sparse = [i for i, lt in enumerate(sm.layer_types) if lt == sm.layer_types[3]]
        first, last = sparse[0], sparse[-1]
        assert first != last, sparse
        tmpl = sm._retarget(sm._new_layer(first), last)
        held = {name: m.layer_idx for name, m in tmpl.named_modules() if isinstance(getattr(m, "layer_idx", None), int)}
        assert any("indexer" in name for name in held), held
        assert set(held.values()) == {last}, held
    finally:
        sm.close()
