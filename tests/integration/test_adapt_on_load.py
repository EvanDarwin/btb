# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""`adapt` while the engine initializes: another program taking the card during a load (a game launched while a model
of minutes loads) is seen before each layer is placed, through this process's WDDM budget, and the layers still to
place that the card no longer has room for go to the host instead - the plan bends while nothing has moved yet. A
layer placed stays until the engine is up. With `adapt` off the plan is kept as drawn."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from btb.engine import device as device_mod
from tests.helpers import fixture, loaded_model, need_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA card")

GB = 1 << 30


@pytest.mark.parametrize("adapt", [1, 0])
def test_a_program_taking_the_card_during_a_load_moves_the_layers_not_yet_placed(
    adapt: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from btb.engine import StreamedTextModel

    dev = need_cuda()
    # the budget's room: 8 GB until two layers are on the card, then 4 GB less - a game started mid-load (the
    # planner's own reads before the load see the 8 GB)
    # the layers placed, by index (a layer's load reads it in more than one call)
    placed: set[int] = set()
    load = StreamedTextModel._load_layer

    def counted(self: Any, i: int, tmpl: Any, *a: Any, **kw: Any) -> Any:
        out = load(self, i, tmpl, *a, **kw)
        placed.add(int(i))
        return out

    monkeypatch.setattr(StreamedTextModel, "_load_layer", counted)
    monkeypatch.setattr(device_mod, "_wddm_room", lambda d: 8 * GB if len(placed) < 2 else 4 * GB)
    lines: list[str] = []
    path = fixture("tiny_qwen3")
    with loaded_model(path, device=dev, adapt=adapt, log=lambda *a, **k: lines.append(" ".join(map(str, a)))) as sm:
        L = int(sm.L)
        said = [ln for ln in lines if "another program took" in ln]
        if adapt:
            assert said, f"the drop was not seen: {lines}"
            assert len(sm.resident) == 2 and set(sm.host) == set(range(2, L)), (
                f"resident {sorted(sm.resident)}, host {sorted(sm.host)}: the layers still to place were not moved"
            )
        else:
            assert not said and len(sm.resident) == L, "adapt off: the plan as drawn"
        out: Any = sm.generate([3, 17, 42, 5], 4, eos=(), speculate=False)
        assert len(out.tokens) >= 4, "the engine does not decode after the plan bent"
