# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""GGUF models (btb/gguf.py): the fixtures' GGUF twins (tests/make_fixtures.py gguf) read through the gguf
package - the config off the metadata, every tensor of the family mapped through llama.cpp's table, a bf16
file decoding token for token as its safetensors twin, an f16 file within f16's rounding of it, a quantized
file as the numbers llama.cpp dequantizes, a gpt-oss file's MXFP4 experts streamed and multiplied as stored
(ggml's layout, in the CPU and Metal kernels), discovery and resolution of .gguf paths. The tokenizer conversion (transformers') is not exercised
here: the fixtures' byte-level vocabulary has no merges, which that converter refuses; it is checked on a real
release."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.helpers import (
    FIXTURES,
    GGUF_FIXTURES,
    NO_LOG,
    assert_close,
    cached,
    loaded_model,
    max_abs,
    mxfp4_random,
    need_cached,
    need_mlx,
    need_native,
    safetensors_state,
)

GGUF = GGUF_FIXTURES
PROMPT = [3, 17, 42, 99, 7, 250, 11, 64]


def _need() -> None:
    if not os.path.isfile(os.path.join(GGUF, "tiny_qwen3-bf16.gguf")):
        pytest.skip("the GGUF fixtures are not here (tests/make_fixtures.py gguf)")


def _tokens(path: str, device: str = "cpu") -> list[int]:
    with loaded_model(path, device=device) as sm:
        out, _ = sm.generate(PROMPT, 8, eos=(), speculate=False)
    return [int(t) for t in out]


def test_the_reader_maps_every_tensor_and_reads_the_same_numbers() -> None:
    """the bf16 twin: config numbers the fixture's (the head size included, which transformers' table lacks),
    the vocabulary and the chat template in the metadata, every tensor of the safetensors fixture found under
    llama.cpp's name and equal as bf16; the f16 twin within f16's rounding, and not equal (bf16's smallest
    values are f16 subnormals); Phi-3's fused projections through the same table"""
    _need()
    from gguf import Keys

    from btb.gguf import GGUFModel

    g = GGUFModel(os.path.join(GGUF, "tiny_qwen3-bf16.gguf"))
    cfg = g.config()
    want = json.load(open(os.path.join(FIXTURES, "tiny_qwen3", "config.json")))
    for k in (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "intermediate_size",
    ):
        assert getattr(cfg, k) == want[k], (k, getattr(cfg, k), want[k])
    assert cfg.model_type == "qwen3" and g.model_type == "qwen3"
    tokens = g.reader.get_field(Keys.Tokenizer.LIST)
    assert tokens is not None and len(tokens.data) == want["vocab_size"]
    assert g.reader.get_field(Keys.Tokenizer.CHAT_TEMPLATE) is not None
    state = safetensors_state(os.path.join(FIXTURES, "tiny_qwen3"))
    names, hdr = g.weight_map(list(state), cfg.num_hidden_layers)
    assert set(names) == set(state), set(state) ^ set(names)
    for hf, gn in names.items():
        t = g.get(gn)
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == tuple(state[hf].shape) == tuple(hdr[hf]["shape"]), hf
        assert torch.equal(t, state[hf].to(torch.bfloat16)), hf
    h = GGUFModel(os.path.join(GGUF, "tiny_qwen3-f16.gguf"))
    names16, _ = h.weight_map(list(state), cfg.num_hidden_layers)
    diffs = [max_abs(h.get(gn).float(), state[hf].float()) for hf, gn in names16.items()]
    assert max(diffs) < 1e-3 and any(d > 0 for d in diffs), max(diffs)
    p = GGUFModel(os.path.join(GGUF, "tiny_phi3-bf16.gguf"))
    pn, _ = p.weight_map(["model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.mlp.gate_up_proj.weight"], 1)
    assert set(pn.values()) == {"blk.0.attn_qkv.weight", "blk.0.ffn_up.weight"}


def test_mxfp4_blocks_repack_to_the_checkpoint_layout() -> None:
    """ggml's 17-byte MXFP4 blocks (the scale first, elements 0-15 in the low nibbles, 16-31 in the high) and the
    checkpoint's blocks + scales (consecutive element pairs) hold the same numbers: the repack dequantizes
    (btb.mxfp4) to what gguf-py dequantizes from the raw blocks, and round-trips byte for byte"""
    _need()
    import gguf.quants
    from gguf import GGMLQuantizationType

    from btb import mxfp4
    from btb.mxfp4 import ggml_to_hf, hf_to_ggml

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (3, 5, 4 * 17), dtype=np.uint8)
    raw.reshape(3, 5, 4, 17)[..., 0] = rng.integers(100, 140, (3, 5, 4))
    blocks, scales = ggml_to_hf(raw)
    assert blocks.shape == (3, 5, 4, 16) and scales.shape == (3, 5, 4)
    ref = gguf.quants.dequantize(raw, GGMLQuantizationType.MXFP4)
    np.testing.assert_array_equal(mxfp4.dequantize(blocks, scales), ref)
    np.testing.assert_array_equal(hf_to_ggml(blocks, scales), raw)
    w = mxfp4.MxWeight.from_ggml(torch.from_numpy(raw[0, 0]), 1, 128)
    np.testing.assert_array_equal(w.dequantize().numpy()[0], ref[0, 0])


def _mxfp4_pair(rows: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """random MXFP4 bytes for a [rows, k] matrix in both layouts: (blocks, scales) and ggml's raw blocks"""
    from btb.mxfp4 import hf_to_ggml

    blocks, scales = mxfp4_random(seed, rows, k, 112, 142)
    return blocks, scales, hf_to_ggml(blocks, scales)


def test_the_cpu_mxfp4_kernel_reads_ggml_blocks_as_the_checkpoints() -> None:
    """the native matvec over ggml's layout gives the checkpoint layout's bits for the same weights, alone
    and grouped, at every batch width the host path uses; skipped without the native library"""
    _need()
    from btb.engine.native import Native
    from btb.mxfp4 import MxWeight

    need_native()
    if Native.gemv_mx4_ggml is None:
        pytest.skip("the native library predates ggml's layout")
    rows, k = 96, 256
    for b in (1, 5, 16):
        blocks, scales, raw = _mxfp4_pair(rows, k, 7 + b)
        hf = MxWeight(torch.from_numpy(blocks.reshape(-1)), torch.from_numpy(scales.reshape(-1)), rows, k)
        gg = MxWeight.from_ggml(torch.from_numpy(raw.reshape(-1)), rows, k)
        x = torch.randn(b, k, generator=torch.Generator().manual_seed(b))
        want, got = torch.empty(b, rows), torch.empty(b, rows)
        Native.gemv_mx4(hf, x, want)
        Native.gemv_mx4_ggml(gg, x, got)
        assert torch.equal(want, got), f"b={b}"
        want2, got2 = torch.empty(b, rows), torch.empty(b, rows)
        Native.gemv_mx4_group([hf, hf], [x, x], [want, want2])
        Native.gemv_mx4_ggml_group([gg, gg], [x, x], [got, got2])
        assert torch.equal(want, got) and torch.equal(want2, got2), f"group b={b}"


def test_the_metal_mxfp4_kernels_read_ggml_blocks_as_the_checkpoints() -> None:
    """the Metal matvec and dequantizer over ggml's layout give the checkpoint layout's bits, at every batch
    width, in bf16 and float32, grouped too; skipped without MLX"""
    _need()
    need_mlx()
    import mlx.core as mx

    from btb import mlx as mlxdev

    rows, k = 96, 256
    for b in (1, 5, 16):
        blocks, scales, raw = _mxfp4_pair(rows, k, 30 + b)
        slot = mx.array(np.concatenate([blocks.reshape(-1), scales.reshape(-1)]))
        gg = mx.array(raw.reshape(-1))
        mx.eval(slot, gg)
        for dtype in (mx.float32, mx.bfloat16):
            x = mx.random.normal((b, k), key=mx.random.key(b)).astype(dtype)
            want = mlxdev.gemv_mxfp4(slot, rows, k, x)
            got = mlxdev.gemv_mxfp4(gg, rows, k, x, ggml=True)
            mx.eval(want, got)
            assert bool(
                mx.array_equal(
                    want.view(mx.uint32 if dtype == mx.float32 else mx.uint16),
                    got.view(mx.uint32 if dtype == mx.float32 else mx.uint16),
                )
            ), (b, dtype)
            wg = mlxdev.gemv_mxfp4_group([slot, slot], rows, k, mx.broadcast_to(x[None], (2, b, k)))
            gr = mlxdev.gemv_mxfp4_group([gg, gg], rows, k, mx.broadcast_to(x[None], (2, b, k)), ggml=True)
            mx.eval(wg, gr)
            assert bool(mx.array_equal(wg.astype(mx.float32), gr.astype(mx.float32))), ("group", b, dtype)
        dq = mlxdev.mxfp4_dequant(slot, rows, k, mx.float32)
        dg = mlxdev.mxfp4_dequant(gg, rows, k, mx.float32, ggml=True)
        mx.eval(dq, dg)
        assert bool(mx.array_equal(dq, dg))
        # gate and up apart (the even and odd rows) through the pair kernel: the interleaved gate_up's bits
        from btb.mxfp4 import hf_to_ggml

        gate = mx.array(hf_to_ggml(blocks[0::2], scales[0::2]).reshape(-1))
        up = mx.array(hf_to_ggml(blocks[1::2], scales[1::2]).reshape(-1))
        x = mx.random.normal((b, k), key=mx.random.key(b + 50)).astype(mx.bfloat16)
        want = mlxdev.gemv_mxfp4(slot, rows, k, x)
        got = mlxdev.gemv_mxfp4_pair(gate, up, rows // 2, k, x)
        big = mlxdev.matmul_mxfp4_pair(gate, up, rows // 2, k, mx.broadcast_to(x[:1], (20, k)))
        ref = mlxdev.matmul_mxfp4(slot, rows, k, mx.broadcast_to(x[:1], (20, k)))
        mx.eval(want, got, big, ref)
        assert bool(mx.array_equal(want.view(mx.uint16), got.view(mx.uint16))), ("pair", b)
        assert bool(mx.array_equal(big.view(mx.uint16), ref.view(mx.uint16))), ("pair gemm", b)


def test_a_gpt_oss_gguf_streams_its_experts_as_stored_and_decodes_as_its_twin(tmp_path: Path) -> None:
    """the file's MXFP4 experts are read by the expert store straight out of the GGUF, gate, up and down in
    ggml's layout, and multiplied as stored; the config carries the file's expert counts, window and yarn
    scaling; the file decodes as its safetensors twin on the CPU and on MLX; nothing is written beside it"""
    _need()
    src = os.path.join(GGUF, "tiny_gpt_oss-mxfp4.gguf")
    if not os.path.isfile(src):
        pytest.skip("the gpt-oss GGUF fixture is not here (tests/make_fixtures.py gguf)")
    from btb.engine.device import mlx_available
    from btb.gguf import GGUFModel
    from btb.mxfp4 import MxGateUp

    path = str(tmp_path / "tiny_gpt_oss-mxfp4.gguf")
    shutil.copy(src, path)
    g = GGUFModel(path)
    cfg = g.config()
    twin = os.path.join(FIXTURES, "tiny_gpt_oss")
    want = json.load(open(os.path.join(twin, "config.json")))
    for k in ("num_hidden_layers", "hidden_size", "num_local_experts", "num_experts_per_tok", "sliding_window"):
        assert getattr(cfg, k) == want[k], (k, getattr(cfg, k), want[k])
    assert cfg.head_dim == want["head_dim"] and cfg.layer_types == want["layer_types"]
    rope = cfg.rope_parameters
    assert rope["rope_type"] == "yarn" and rope["factor"] == want["rope_scaling"]["factor"]
    assert rope["original_max_position_embeddings"] == want["rope_scaling"]["original_max_position_embeddings"]
    with loaded_model(path, device="cpu") as sm:
        assert "blk.0.ffn_gate_exps.weight" in sm.weight_map
        bias = sm._get("model.layers.0.mlp.experts.gate_up_proj_bias")
        assert tuple(bias.shape) == (want["num_local_experts"], 2 * want["intermediate_size"])
        out, _ = sm.generate(PROMPT, 8, eos=(), speculate=False)
        store = sm.expert_store
        if store is not None:
            assert store.ggml and isinstance(store._views(store.last_slots[next(iter(store.last_slots))])[0], MxGateUp)
            assert sm.expert_stat["experts"] > 0
    assert [int(t) for t in out] == _tokens(twin)
    if mlx_available():
        assert _tokens(path, "mlx") == _tokens(twin, "mlx")
    assert sorted(os.listdir(str(tmp_path))) == ["tiny_gpt_oss-mxfp4.gguf"]


def test_the_o200k_pre_tokenizer_matches_the_hf_gpt_oss_tokenizer() -> None:
    """the pre-tokenizer a gpt-oss GGUF gets (its `pre` type names o200k, which the file does not carry) splits
    and encodes as the HF gpt-oss tokenizer.json does, on text with contractions, numbers, punctuation runs,
    mixed scripts and whitespace; skipped when no gpt-oss tokenizer is in the cache"""
    _need()
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
    from tokenizers import Tokenizer

    from btb.gguf import o200k_pre_tokenizer

    try:
        repos = scan_cache_dir().repos
    except CacheNotFound:  # a machine that never downloaded a model has no cache directory at all
        pytest.skip("no Hugging Face cache on this machine")
    snaps = [
        str(rev.snapshot_path)
        for r in repos
        if r.repo_id.startswith("openai/gpt-oss")
        for rev in r.revisions
        if os.path.isfile(os.path.join(str(rev.snapshot_path), "tokenizer.json"))
    ]
    if not snaps:
        pytest.skip("no gpt-oss tokenizer.json in the cache")
    hf = Tokenizer.from_file(os.path.join(snaps[0], "tokenizer.json"))
    ours = Tokenizer.from_file(os.path.join(snaps[0], "tokenizer.json"))
    ours.pre_tokenizer = o200k_pre_tokenizer()
    texts = [
        "Hello world, it's 12345 o'clock!!  Don't  panic\n\n  ok",
        "the price is $1,234.56 (USD) - or 9.99/month",
        "  leading spaces and\ttabs\r\nCRLF lines",
        "I'VE GOT 3 APPLES and 42 pears; y'all'd've known",
        "def f(x):\n    return x**2  # comment\n",
        "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 and English mixed 123456",
    ]
    assert hf.pre_tokenizer is not None and ours.pre_tokenizer is not None
    for text in texts:
        assert hf.pre_tokenizer.pre_tokenize_str(text) == ours.pre_tokenizer.pre_tokenize_str(text), text
        assert hf.encode(text).ids == ours.encode(text).ids, text


def test_a_bf16_gguf_decodes_as_its_safetensors_twin() -> None:
    _need()
    assert _tokens(os.path.join(GGUF, "tiny_qwen3-bf16.gguf")) == _tokens(os.path.join(FIXTURES, "tiny_qwen3"))


def test_a_quantized_gguf_decodes_as_its_dequantized_weights(tmp_path: Path) -> None:
    """Q8_0 and Q4_0: the file decodes exactly as a safetensors model built from the numbers the gguf package
    dequantizes (llama.cpp's own arithmetic), and the numbers differ from the bf16 twin's as a quantization does"""
    _need()
    from safetensors.torch import save_file

    from btb.gguf import GGUFModel

    src = os.path.join(FIXTURES, "tiny_qwen3")
    state = safetensors_state(os.path.join(FIXTURES, "tiny_qwen3"))
    for qt in ("q8_0", "q4_0"):
        path = os.path.join(GGUF, f"tiny_qwen3-{qt}.gguf")
        g = GGUFModel(path)
        names, _ = g.weight_map(list(state), g.config().num_hidden_layers)
        deq = {hf: g.get(gn).contiguous() for hf, gn in names.items()}
        assert any(not torch.equal(deq[k], state[k].to(torch.bfloat16)) for k in deq), qt
        twin = tmp_path / f"twin-{qt}"
        twin.mkdir()
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(src, fn), twin / fn)
        save_file(deq, str(twin / "model.safetensors"), metadata={"format": "pt"})
        assert _tokens(path) == _tokens(str(twin)), qt


def _need_q35() -> None:
    if not os.path.isfile(os.path.join(GGUF, "tiny_q35-bf16.gguf")):
        pytest.skip("the Qwen3.5 GGUF fixtures are not here (tests/make_fixtures.py gguf_q35)")


def _q35_trunk() -> dict[str, torch.Tensor]:
    """the tiny_q35 checkpoint without its MTP head, which llama.cpp's converter leaves out under --no-mtp"""
    state = safetensors_state(os.path.join(FIXTURES, "tiny_q35"))
    return {k: v for k, v in state.items() if not k.startswith("mtp.")}


def test_a_qwen35_gguf_reads_back_its_checkpoint() -> None:
    """Qwen3.5 through llama.cpp's converter and back: the config off the typed keys (the hybrid's layer pattern,
    linear attention widths, partial MRoPE), every trunk tensor found and equal as bf16 once the converter's
    rewrites are inverted (the tiled value heads, -exp(A_log), 1 + norm, the squeezed conv), and a quantized
    file's reordered blocks (what the packed kernels bind) the same numbers as its reordered dequantization"""
    _need_q35()
    from gguf import dequantize

    from btb.gguf import GGUFModel

    g = GGUFModel(os.path.join(GGUF, "tiny_q35-bf16.gguf"))
    cfg = g.config()
    want = json.load(open(os.path.join(FIXTURES, "tiny_q35", "config.json")))
    assert cfg.model_type == want["model_type"] == g.model_type
    for k in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "layer_types",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "partial_rotary_factor",
        "tie_word_embeddings",
    ):
        assert getattr(cfg, k) == want[k], (k, getattr(cfg, k), want[k])
    for k in ("rope_theta", "mrope_section", "mrope_interleaved"):
        assert cfg.rope_parameters[k] == want["rope_parameters"][k], k
    state = _q35_trunk()
    names, hdr = g.weight_map(list(state), cfg.num_hidden_layers)
    assert set(names) == set(state), set(state) ^ set(names)
    for hf, gn in names.items():
        t = g.get(gn)
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == tuple(state[hf].shape) == tuple(hdr[hf]["shape"]), hf
        assert torch.equal(t, state[hf].to(torch.bfloat16)), hf
    for qt in ("q8_0", "q4_0", "q4_1"):
        q = GGUFModel(os.path.join(GGUF, f"tiny_q35-{qt}.gguf"))
        qnames, _ = q.weight_map(list(state), cfg.num_hidden_layers)
        reordered = [gn for gn in qnames.values() if gn in q.undo and gn in q.tensors]
        assert any(q.undo[gn][1].cols is not None for gn in reordered), qt  # out_proj binds as stored too
        for gn in reordered:
            t = q.tensors[gn]
            got = dequantize(q.raw(gn).numpy(), t.tensor_type).reshape(q.get(gn).shape)
            assert torch.equal(torch.from_numpy(np.array(got, dtype=np.float32)).bfloat16(), q.get(gn)), (qt, gn)


def test_a_qwen35_bf16_gguf_decodes_as_its_safetensors_twin() -> None:
    _need_q35()
    twin = os.path.join(FIXTURES, "tiny_q35")
    assert _tokens(os.path.join(GGUF, "tiny_q35-bf16.gguf")) == _tokens(twin)


def test_a_qwen35_bf16_gguf_decodes_as_its_safetensors_twin_on_mlx() -> None:
    _need_q35()
    need_mlx()
    twin = os.path.join(FIXTURES, "tiny_q35")
    assert _tokens(os.path.join(GGUF, "tiny_q35-bf16.gguf"), "mlx") == _tokens(twin, "mlx")


def test_a_quantized_qwen35_gguf_decodes_as_its_dequantized_weights(tmp_path: Path) -> None:
    """each affine type: the file decodes exactly as a safetensors model built from the numbers the reader
    dequantizes and un-rewrites, and those numbers differ from the bf16 twin's as a quantization does"""
    _need_q35()
    from safetensors.torch import save_file

    from btb.gguf import GGUFModel

    src = os.path.join(FIXTURES, "tiny_q35")
    state = _q35_trunk()
    for qt in ("q8_0", "q4_0", "q4_1"):
        path = os.path.join(GGUF, f"tiny_q35-{qt}.gguf")
        g = GGUFModel(path)
        names, _ = g.weight_map(list(state), g.config().num_hidden_layers)
        deq = {hf: g.get(gn).contiguous() for hf, gn in names.items()}
        assert any(not torch.equal(deq[k], state[k].to(torch.bfloat16)) for k in deq), qt
        twin = tmp_path / f"twin-{qt}"
        twin.mkdir()
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(src, fn), twin / fn)
        save_file(deq, str(twin / "model.safetensors"), metadata={"format": "pt"})
        assert _tokens(path) == _tokens(str(twin)), qt


def test_a_qwen35_gguf_multiplies_its_reordered_blocks_as_stored_on_mlx() -> None:
    """the value-head reordering applied to the stored blocks (rows of in_proj_*, whole blocks of out_proj's
    columns): the packed path's first-step logits sit within bf16 rounding of the dequantized path's"""
    _need_q35()
    need_mlx()
    logits = {}
    for packed in (1, 0):
        with loaded_model(os.path.join(GGUF, "tiny_q35-q8_0.gguf"), device="mlx", gguf_packed=packed) as sm:
            rep = sm.report()["model"]["gguf"]
            assert (rep["packed"] > 0) == bool(packed), rep
            lg = sm.forward([PROMPT], cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
    assert_close(logits[1], logits[0])


def test_gguf_paths_are_discovered_resolved_and_not_packed() -> None:
    _need()
    from btb import available_models, model_stem, resolve
    from btb.engine import pack_model
    from btb.options import NotPackable

    f16 = os.path.join(GGUF, "tiny_qwen3-f16.gguf")
    assert resolve(f16) == os.path.abspath(f16) and model_stem(f16) == "tiny_qwen3-f16"
    names = {e["name"]: e for e in available_models([GGUF], pattern="tiny_")}
    assert {"tiny_qwen3-bf16", "tiny_qwen3-f16", "tiny_qwen3-q8_0", "tiny_qwen3-q4_0", "tiny_phi3-bf16"} <= set(names)
    assert names["tiny_qwen3-q4_0"]["type"] == "qwen3" and names["tiny_qwen3-q4_0"]["path"].endswith(".gguf")
    assert names["tiny_qwen3-q4_0"]["size"] == os.path.getsize(names["tiny_qwen3-q4_0"]["path"])
    with pytest.raises(NotPackable) as e:
        pack_model(f16, log=NO_LOG)
    assert e.value.path == "tiny_qwen3-f16.gguf"


REAL_REPO = "unsloth/Qwen3-0.6B-GGUF"
REAL_TYPES = (
    "BF16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
    "Q4_0",
    "Q4_1",
    "Q3_K_M",
    "Q3_K_S",
    "Q2_K",
    "IQ4_NL",
    "IQ4_XS",
    "UD-IQ3_XXS",
    "UD-IQ2_M",
    "UD-IQ1_M",
)


def test_real_qwen3_gguf_files_when_cached() -> None:
    """the release files (every quant depth unsloth ships for Qwen3-0.6B), whichever are in the cache: the
    tokenizer and chat template rebuilt from the metadata, a greedy answer from each, and the BF16 file the
    same tokens as the safetensors release when that is cached too; skipped where none are"""
    _need()
    files = {q: cached(f"{REAL_REPO}:Qwen3-0.6B-{q}.gguf") for q in REAL_TYPES}
    files = {q: p for q, p in files.items() if p is not None}
    if not files:
        pytest.skip("no Qwen3-0.6B GGUF release file is cached")
    ref = cached("Qwen/Qwen3-0.6B")
    prompt = "What is 2 + 2? Answer with the number only."
    answers = {}
    for q, path in files.items():
        assert path is not None
        with loaded_model(path, device="cpu") as sm:
            tok = sm.tokenizer
            assert tok.chat_template and tok("hello world")["input_ids"], q
            assert sm.stop_ids, q
            ids = sm.prompt_ids(prompt)
            out, _ = sm.generate(ids, 8, speculate=False)
            answers[q] = (ids, [int(t) for t in out], sm.tokenizer.decode(out, skip_special_tokens=True))
    print({q: a[2] for q, a in answers.items()})
    if ref and "BF16" in answers:
        with loaded_model(ref, device="cpu") as sm:
            ids = sm.prompt_ids(prompt)
            out, _ = sm.generate(ids, 8, speculate=False)
        assert answers["BF16"][0] == ids and answers["BF16"][1] == [int(t) for t in out]


def test_a_q6k_gguf_multiplies_its_blocks_as_stored_on_mlx() -> None:
    """the Q6_K release, when cached: on MLX its Q6_K tensors (every weight of a pure-Q6_K file, the head and
    ffn_down of a mixed one) bind to the Q6_K matvec as stored, no bf16 copy, and the first-step logits sit
    within bf16 rounding of the dequantized path's - the same greedy tokens"""
    need_mlx()
    path = need_cached(f"{REAL_REPO}:Qwen3-0.6B-Q6_K.gguf", "Qwen3-0.6B-Q6_K.gguf is not cached")
    logits, toks = {}, {}
    for packed in (1, 0):
        with loaded_model(path, device="mlx", gguf_packed=packed) as sm:
            if packed:
                assert any(getattr(m.mx, "q6k", None) is not None for m in sm.host[0].modules() if hasattr(m, "mx"))
            lg = sm.forward(PROMPT, cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
            out, _ = sm.generate(PROMPT, 8, speculate=False)
            toks[packed] = [int(t) for t in out]
    a, b = logits[1], logits[0]
    assert_close(a, b)
    assert toks[1] == toks[0]


@pytest.mark.parametrize(
    ("quant", "kind", "stable"),
    [
        ("Q2_K", "q2k", True),
        ("Q3_K_M", "q3k", True),
        ("Q5_K_M", "q5k", True),
        ("IQ4_NL", "iq4nl", True),
        ("IQ4_XS", "iq4xs", True),
        ("UD-IQ3_XXS", "latt", True),
        ("UD-IQ2_M", "latt", True),
        ("UD-IQ1_M", "latt", False),
    ],
)
def test_a_native_kernel_gguf_multiplies_its_blocks_as_stored_on_mlx(quant: str, kind: str, stable: bool) -> None:
    """each release whose tensors have their own Metal matvec (the k-quants, the IQ4 codebooks, the IQ lattice
    mixes behind the Unsloth dynamic files), when cached. The kernel contract is checked directly: one tensor of
    every quantized type in the file multiplied as stored equals the package's fp32 dequantization to fp32
    rounding. Then on MLX the tensors bind as their raw bytes to that kernel (no bf16 copy), and on a numerically
    stable file the first-step logits sit within bf16 rounding of the dequantized path's with the same greedy
    tokens. `stable` is False for the 1-bit file: with exact kernels its two paths still drift 43% of max-logit
    by the first token (2% on the 2- and 3-bit siblings on the same code), a smooth per-layer amplification of
    rounding order that the CPU path also fails to agree with - token equality is no implementation's property
    there, so it asks only that the paths agree on the first token."""
    need_mlx()
    import mlx.core as mx
    from gguf import dequantize

    from btb.gguf import GGUFModel
    from btb.mlx.iquant import matvec_iq4nl, matvec_iq4xs, matvec_lattice, repack_iq4nl, repack_lattice
    from btb.mlx.kquant import matvec_q2k, matvec_q3k, matvec_q4k, matvec_q5k
    from btb.mlx.q6k import matvec_q6k

    path = need_cached(f"{REAL_REPO}:Qwen3-0.6B-{quant}.gguf", f"Qwen3-0.6B-{quant}.gguf is not cached")
    latt = {
        "IQ3_XXS": "iq3xxs",
        "IQ2_XXS": "iq2xxs",
        "IQ2_XS": "iq2xs",
        "IQ2_S": "iq2s",
        "IQ1_S": "iq1s",
        "IQ3_S": "iq3s",
        "IQ1_M": "iq1m",
    }
    kq = {"Q2_K": matvec_q2k, "Q3_K": matvec_q3k, "Q4_K": matvec_q4k, "Q5_K": matvec_q5k, "Q6_K": matvec_q6k}
    seen: set[str] = set()
    for name, t in GGUFModel(path).tensors.items():
        tn = t.tensor_type.name
        if tn in seen or tn == "F32" or len(t.shape) != 2:
            continue
        rows, cols = int(t.shape[1]), int(t.shape[0])
        if cols % (32 if tn == "IQ4_NL" else 256):
            continue
        raw = np.ascontiguousarray(np.asarray(t.data)).view(np.uint8).reshape(-1)
        ref = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(rows, cols)
        x = np.random.default_rng(0).standard_normal((1, cols)).astype(np.float32)
        wb, xm = mx.array(raw), mx.array(x)
        if tn in latt:
            y = matvec_lattice(latt[tn], wb, xm, rows, cols, repack_lattice(latt[tn], raw))
        elif tn in kq:
            y = kq[tn](wb, xm, rows, cols)
        elif tn == "IQ4_NL":
            d, q = repack_iq4nl(raw)
            y = matvec_iq4nl(d, q, xm, rows, cols)
        elif tn == "IQ4_XS":
            y = matvec_iq4xs(wb, xm, rows, cols)
        else:
            continue
        gold = x @ ref.T
        rel = float(np.abs(np.array(y)[0] - gold[0]).max() / (np.abs(gold[0]).max() + 1e-9))
        assert rel < 1e-4, (quant, tn, name, rel)
        seen.add(tn)
    assert seen, quant

    logits, toks = {}, {}
    for packed in (1, 0):
        with loaded_model(path, device="mlx", gguf_packed=packed) as sm:
            if packed:
                bound = any(
                    getattr(m.mx, kind, None) is not None
                    for layer in sm.host.values()
                    for m in layer.modules()
                    if hasattr(m, "mx")
                )
                assert bound, f"{quant}: no tensor bound to the {kind} kernel"
            lg = sm.forward(PROMPT, cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
            out, _ = sm.generate(PROMPT, 8, speculate=False)
            toks[packed] = [int(t) for t in out]
    a, b = logits[1], logits[0]
    assert a.argmax().item() == b.argmax().item()
    if stable:
        assert_close(a, b)
        assert toks[1] == toks[0]


def test_a_draft_model_proposes_a_tree_verified_exactly_on_mlx() -> None:
    """Qwen3-0.6B drafting for Qwen3-4B (both cached): `--draft-model` loads the sibling and wires the
    ModelProposer, the speculative tree it proposes verifies to the plain greedy tokens exactly, and the pass
    count drops well below one a token (drafts accepted). Skipped without both files or MLX."""
    need_mlx()
    draft = cached(f"{REAL_REPO}:Qwen3-0.6B-Q4_K_M.gguf")
    target = cached("unsloth/Qwen3-4B-GGUF:Qwen3-4B-Q4_K_M.gguf")
    if draft is None or target is None:
        pytest.skip("Qwen3-4B and 0.6B Q4_K_M GGUFs are not both cached")
    with loaded_model(target, device="mlx", gguf_packed=1, draft_model=draft) as sm:
        assert sm.draft_engine is not None and sm.draft_ks == (4, 3, 2)
        ids = sm.prompt_ids("Write a detailed paragraph about the ocean and its depths.")
        greedy = sm.generate(ids, 48, speculate=False)
        spec = sm.generate(ids, 48, speculate=True)
    assert list(spec.tokens) == list(greedy.tokens), (spec.tokens, greedy.tokens)
    assert spec.stats["forwards"] < len(spec.tokens), spec.stats["forwards"]


def test_a_q4k_self_draft_speculates_to_the_greedy_tokens_on_mlx() -> None:
    """Qwen3-0.6B-Q4_K_M drafting for itself (`--draft-model` the same file): with Q4_K on the batch-invariant
    native matvec a verify pass computes each row exactly as the one-row draft step did, so the speculative
    tree verifies to the plain greedy tokens - the divergence the non-invariant `quantized_matmul` caused is
    gone. Skipped without the file or MLX."""
    need_mlx()
    path = need_cached(f"{REAL_REPO}:Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q4_K_M.gguf is not cached")
    with loaded_model(path, device="mlx", gguf_packed=1, draft_model=path) as sm:
        assert sm.draft_engine is not None
        assert any(getattr(m.mx, "q4k", None) is not None for m in sm.host[0].modules() if hasattr(m, "mx"))
        ids = sm.prompt_ids("Write a detailed paragraph about the ocean and its depths.")
        greedy = sm.generate(ids, 48, speculate=False)
        spec = sm.generate(ids, 48, speculate=True)
    assert list(spec.tokens) == list(greedy.tokens), (spec.tokens, greedy.tokens)


def test_the_packed_mlx_path_multiplies_as_stored() -> None:
    """on MLX a Q4_0 / Q8_0 fixture binds its matrices in the affine form and the packed kernel multiplies them
    as stored: the affine repack equals the package's dequantization to the number, and Q4_K (which leaves the
    affine path for its own native matvec) is within bf16 rounding of the dequantization and batch-invariant.
    The report counts the packed tensors, and the first-step logits sit within bf16 rounding of the dequantized
    path's"""
    _need()
    need_mlx()
    from gguf import dequantize

    from btb.gguf import GGUFModel

    g = GGUFModel(os.path.join(GGUF, "tiny_qwen3-q4_0.gguf"))
    for name, t in g.tensors.items():
        aff = g.affine(name)
        if aff is None:
            continue
        import mlx.core as mx

        deq = mx.dequantize(
            mx.array(aff.wq), mx.array(aff.scales), mx.array(aff.biases), group_size=aff.group, bits=aff.bits
        )
        ref = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
        assert np.array_equal(np.array(deq.astype(mx.float32)), ref), name

    # Q4_K leaves the affine repack for its own native matvec (batch-invariant, so affine speculation is exact):
    # over a synthetic superblock weight the matvec sits within bf16 rounding of the package's dequantization,
    # and a row is bit-identical alone and inside a 16-row tile - what `quantized_matmul` does not guarantee.
    import mlx.core as mx
    from gguf import GGMLQuantizationType

    from btb.mlx.kquant import matvec_q4k

    rng = np.random.default_rng(0)
    rows, cols, nsb = 40, 512, 512 // 256
    blk = np.zeros((rows, nsb, 144), np.uint8)
    d = (rng.random((rows, nsb)).astype(np.float32) * 0.05 + 0.01).astype(np.float16)
    dm = (rng.random((rows, nsb)).astype(np.float32) * 0.03).astype(np.float16)
    blk[:, :, 0:2] = np.frombuffer(np.ascontiguousarray(d).tobytes(), np.uint8).reshape(rows, nsb, 2)
    blk[:, :, 2:4] = np.frombuffer(np.ascontiguousarray(dm).tobytes(), np.uint8).reshape(rows, nsb, 2)
    blk[:, :, 4:] = rng.integers(0, 256, size=(rows, nsb, 140), dtype=np.uint8)
    raw = blk.reshape(-1)
    ref_w = dequantize(raw, GGMLQuantizationType.Q4_K).astype(np.float32).reshape(rows, cols)
    xf = rng.standard_normal((16, cols)).astype(np.float32)
    want = xf @ ref_w.T
    wb = mx.array(raw)
    xb = mx.array(xf).astype(mx.bfloat16)
    got = np.array(matvec_q4k(wb, xb, rows, cols).astype(mx.float32))
    rel = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
    assert rel < 0.02, rel  # bf16 rounding of x and the weight
    one = np.array(matvec_q4k(wb, xb[:1], rows, cols).astype(mx.float32))
    assert np.array_equal(one[0], got[0]), "Q4_K matvec row 0 differs alone vs in a tile"

    ids = [PROMPT]
    logits = {}
    for packed in (1, 0):
        with loaded_model(os.path.join(GGUF, "tiny_qwen3-q8_0.gguf"), device="mlx", gguf_packed=packed) as sm:
            rep = sm.report()["model"]["gguf"]
            assert (rep["packed"] > 0) == bool(packed), rep
            lg = sm.forward(ids, cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
    a, b = logits[1], logits[0]
    assert_close(a, b)
