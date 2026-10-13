# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card kernels' dtype checks: a kernel gets each tensor as an 8-byte pointer, so the launch's width check can't
see an fp32 tensor handed to a bf16 kernel. The engine checks where tensors enter - a graph build, a prompt chunk, a
GEMM - before any launch. Decode steps replay graphs and never run the check."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from pytest import MonkeyPatch

from btb.engine.cuda import _CudaMixin
from btb.engine.native import NativeDtypeError, _Cuda
from tests.helpers import fixture, loaded_model, need_cuda

IDS = [(7 * i + 3) % 200 + 2 for i in range(20)]  # inside every fixture's vocabulary


def test_a_wrong_dtype_names_its_kernel_and_tensor() -> None:
    bf = torch.ones(4, dtype=torch.bfloat16)
    _CudaMixin._card_need_bf16([("btb_add_rmsnorm", "layer 0's input norm", bf), ("btb_x", "an absent buffer", None)])
    with pytest.raises(NativeDtypeError) as e:
        _CudaMixin._card_need_bf16([("btb_add_rmsnorm", "layer 0's input norm", bf.float())])
    assert (e.value.call, e.value.arg, e.value.got, e.value.want) == (
        "btb_add_rmsnorm", "layer 0's input norm", torch.float32, (torch.bfloat16,)
    )  # fmt: skip
    unloaded = _Cuda.__new__(_Cuda)  # no module loaded: a launch through it would fail, the check comes first
    with pytest.raises(NativeDtypeError, match=r"btb_gemm_f32_bf16: the weight is torch\.float32"):
        _CudaMixin._card_gemm(unloaded, False, bf.float().view(2, 2), bf.view(2, 2), bf.view(2, 2), 2, 2, 2)


def _card_model() -> Any:
    """tiny_qwen3 with every layer on the card, skipped where the card graph doesn't run it"""
    return loaded_model(fixture("tiny_qwen3"), device=need_cuda(), cpu_layers=0)


def test_an_fp32_norm_stops_the_pass_before_any_launch(monkeypatch: MonkeyPatch) -> None:
    """layer 0's input norm in fp32: its matmul weights still pass `_card_weights`, so only the dtype check catches
    it, and it does before a kernel is launched"""
    with _card_model() as sm:
        if not sm._card_ready():
            pytest.skip("the card graph does not run this model here")
        k = sm._card_kernels()  # loaded now: their own load launches a probe kernel
        launched: list[str] = []
        real = type(k).launch

        def launch(self: Any, name: str, *a: Any, **kw: Any) -> None:
            launched.append(name)
            real(self, name, *a, **kw)

        monkeypatch.setattr(type(k), "launch", launch)
        norm = sm.resident[0].input_layernorm.weight
        norm.data = norm.data.float()
        with (
            torch.inference_mode(),
            pytest.raises(NativeDtypeError, match=r"btb_add_rmsnorm: layer 0's input norm is torch\.float32"),
        ):
            sm.generate(IDS, 4, eos=(), speculate=False)
        assert launched == [], f"launched before the check: {launched[:5]}"


def test_decode_steps_never_run_the_check(monkeypatch: MonkeyPatch) -> None:
    """after a warm-up that builds every graph, a 64-token answer runs the check as often as an 8-token one: only
    the prompt's pass runs it, never a decode step"""
    with torch.inference_mode(), _card_model() as sm:
        if not sm._card_ready():
            pytest.skip("the card graph does not run this model here")
        calls = [0]
        real = _CudaMixin._card_need_bf16

        def counted(reads: Any) -> None:
            calls[0] += 1
            real(reads)

        monkeypatch.setattr(_CudaMixin, "_card_need_bf16", staticmethod(counted))
        sm.generate(IDS, 64, eos=(), speculate=False)
        assert calls[0] > 0, "the warm-up ran no check: the card path under test never ran"

        def checks(n: int) -> int:
            calls[0] = 0
            sm.generate(IDS, n, eos=(), speculate=False)
            return calls[0]

        assert checks(8) == checks(64), "a decode step ran the dtype check"
