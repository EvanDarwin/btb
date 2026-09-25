"""Fine-grained FP8 (btb/fp8.py and the native `btb_gemv_fp8_*`): the format is transformers' own to the bit, the
native matvec is the held matrix's (each weight the nearest bf16 to e4m3 * scale, as every path holds it), and an
FP8 checkpoint decodes on the engine's kernels as transformers does over those held weights."""

from __future__ import annotations

import os
import shutil
import types
from pathlib import Path

import pytest
import torch

from btb import fp8
from btb.engine.native import Native
from btb.fp8 import F8Weight
from tests.helpers import PROMPT_DENSE, fixture, forward_logits, host_model, native_library, safetensors_state


@pytest.mark.parametrize(
    "shape,block",
    [((64, 96), (16, 32)), ((4, 32, 64), (16, 16)), ((13, 7), None), ((256, 128), (128, 128))],
)
def test_quantize_and_widen_are_transformers_fine_grained_fp8(
    shape: tuple[int, ...], block: tuple[int, int] | None
) -> None:
    """`fp8.quantize` writes the bytes and inverse scales transformers' `Fp8Quantize` does, and `fp8.widen` reads
    them back as its `Fp8Dequantize` does, bit for bit; an e8m0 exponent scale reads as `2^(e - 127)`"""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize, Fp8Quantize

    t = torch.randn(*shape, generator=torch.Generator().manual_seed(7)) * 0.05
    hq = types.SimpleNamespace(quantization_config=types.SimpleNamespace(weight_block_size=block, scale_fmt=None))
    ref = Fp8Quantize(hq)._quantize_one("x.weight", t)
    q, s = fp8.quantize(t, block)
    assert torch.equal(ref["x.weight"].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(ref["x.weight_scale_inv"], s)
    assert torch.equal(Fp8Dequantize(hq)._dequantize_one(q, s, output_dtype=torch.float32), fp8.widen(q, s))
    e = torch.tensor([[120, 127], [130, 1]], dtype=torch.uint8)
    assert torch.equal(fp8.scale_grid(e), torch.exp2(e.float() - 127))
    assert torch.equal(fp8.scale_grid(e.view(torch.float8_e8m0fnu)), torch.exp2(e.float() - 127))


def test_the_native_fp8_matvec_is_the_held_matrix() -> None:
    """`Native.gemv_fp8` is `x @ held(w).T` to f32 rounding, a batch's row is bit-identical to its own call, and a
    group's task to its own call"""
    native_library()
    g = torch.Generator().manual_seed(3)
    mats = []
    for rows, cols, block in ((96, 160, (32, 32)), (7, 13, None), (48, 64, (16, 64))):
        q, s = fp8.quantize(torch.randn(rows, cols, generator=g) * 0.1, block)
        w = F8Weight(q.reshape(-1), s, rows, cols)
        x = torch.randn(5, cols, generator=g)
        y = torch.empty(5, rows)
        Native.gemv_fp8(w, x, y)
        wd = w.dequantize(torch.float64)
        want, mag = x.double() @ wd.T, x.double().abs() @ wd.abs().T  # the error against each row's conditioning
        assert float(((y.double() - want).abs() / mag).max()) < 1e-6, (rows, cols, block)
        for i in range(5):
            one = torch.empty(1, rows)
            Native.gemv_fp8(w, x[i : i + 1].contiguous(), one)
            assert torch.equal(one[0], y[i]), (rows, cols, i)
        mats.append((w, x[:2].contiguous(), y[:2]))
    ys = [torch.empty(2, w.shape[0]) for w, _, _ in mats]
    Native.gemv_fp8_group([w for w, _, _ in mats], [x for _, x, _ in mats], ys)
    for (_, _, want), got in zip(mats, ys, strict=True):
        assert torch.equal(got, want)


def _widened_checkpoint(base: str, twin: str, out: str) -> str:
    """the FP8 twin's weights as the engine holds them (`fp8.held`), as an f32 checkpoint beside `base`'s config
    and tokenizer (so transformers maps the names as it maps `base`'s); gpt-oss's MXFP4 experts widened by
    transformers, as `make_fixtures.bank_gpt_oss` does, since a checkpoint read cannot map them without a
    quantization config"""
    from safetensors.torch import save_file
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    state = safetensors_state(twin)
    wide = {}
    for k, t in state.items():
        if k.endswith(("_scale_inv", ".weight_scale", "_proj_scales")):
            continue
        if k.endswith("_proj_blocks"):
            wide[k.removesuffix("_blocks")] = convert_moe_packed_tensors(t, state[k.removesuffix("blocks") + "scales"])
            continue
        sk = fp8.scale_key(k, state) if t.dtype == fp8.E4M3 else None
        wide[k] = fp8.held(t, state[sk]).float() if sk is not None else t.float() if t.is_floating_point() else t
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(base):
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copyfile(os.path.join(base, name), os.path.join(out, name))
    save_file({k: v.contiguous() for k, v in wide.items()}, os.path.join(out, "model.safetensors"))
    return out


@pytest.mark.parametrize("stem", ["tiny_qwen3", "tiny_phi3", "tiny_q35", "tiny_q4", "tiny_gpt_oss", "tiny_gemma3"])
def test_an_fp8_checkpoint_decodes_as_transformers_over_its_held_weights(stem: str, tmp_path: Path) -> None:
    """the FP8 twin on the engine's CPU tier - its linears, fused experts and n-gram tables multiplied as stored -
    gives transformers' float32 logits over the same checkpoint's held weights (`fp8.held`), and its greedy tokens"""
    from transformers import AutoModelForCausalLM

    native_library()
    base, twin = fixture(stem), fixture(f"{stem}-f8_e4m3")
    wide = _widened_checkpoint(base, twin, str(tmp_path / stem))
    ref = AutoModelForCausalLM.from_pretrained(wide, dtype=torch.float32).eval()
    sm = host_model(twin)
    try:
        assert sm.fp8_layers, "no layer bound its FP8 linears as stored"
        with torch.inference_mode():
            want = ref(torch.tensor(PROMPT_DENSE)).logits[0, -1]
            got = forward_logits(sm, PROMPT_DENSE, sm.new_cache())[0, -1]
            d = float((got - want).abs().max())
            assert d < 1e-4, f"{stem}: engine vs transformers {d:.2e}"
            ids = torch.tensor(PROMPT_DENSE)
            ref_greedy = ref.generate(ids, max_new_tokens=6, do_sample=False)[0, ids.shape[1] :].tolist()
        assert sm.generate_greedy(PROMPT_DENSE, 6) == ref_greedy
        assert os.path.basename(twin).endswith("-f8_e4m3")
    finally:
        sm.close()
