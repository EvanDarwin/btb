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
    cached,
    max_abs,
    mxfp4_random,
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
    import btb

    with btb.load(path, device=device, log=None) as sm:
        out, _ = sm.generate(PROMPT, 8, eos=(), greedy=True)
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
    import btb
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
    with btb.load(path, device="cpu", log=None) as sm:
        assert "blk.0.ffn_gate_exps.weight" in sm.weight_map
        bias = sm._get("model.layers.0.mlp.experts.gate_up_proj_bias")
        assert tuple(bias.shape) == (want["num_local_experts"], 2 * want["intermediate_size"])
        out, _ = sm.generate(PROMPT, 8, eos=(), greedy=True)
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
    import btb

    files = {q: cached(f"{REAL_REPO}:Qwen3-0.6B-{q}.gguf") for q in REAL_TYPES}
    files = {q: p for q, p in files.items() if p is not None}
    if not files:
        pytest.skip("no Qwen3-0.6B GGUF release file is cached")
    ref = cached("Qwen/Qwen3-0.6B")
    prompt = "What is 2 + 2? Answer with the number only."
    answers = {}
    for q, path in files.items():
        assert path is not None
        with btb.load(path, device="cpu", log=None) as sm:
            tok = sm.tokenizer
            assert tok.chat_template and tok("hello world")["input_ids"], q
            assert sm.stop_ids, q
            ids = sm.prompt_ids(prompt)
            out, _ = sm.generate(ids, 8, greedy=True)
            answers[q] = (ids, [int(t) for t in out], sm.tokenizer.decode(out, skip_special_tokens=True))
    print({q: a[2] for q, a in answers.items()})
    if ref and "BF16" in answers:
        with btb.load(ref, device="cpu", log=None) as sm:
            ids = sm.prompt_ids(prompt)
            out, _ = sm.generate(ids, 8, greedy=True)
        assert answers["BF16"][0] == ids and answers["BF16"][1] == [int(t) for t in out]


def test_the_packed_mlx_path_multiplies_as_stored() -> None:
    """on MLX a Q4_0 / Q8_0 fixture binds its matrices in the affine form and the packed kernel multiplies them
    as stored: the affine repack equals the package's dequantization to the number, the report counts the
    packed tensors, and the first-step logits sit within bf16 rounding of the dequantized path's"""
    _need()
    need_mlx()
    from gguf import dequantize

    import btb
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
    ids = [PROMPT]
    logits = {}
    for packed in (1, 0):
        with btb.load(os.path.join(GGUF, "tiny_qwen3-q8_0.gguf"), device="mlx", gguf_packed=packed, log=None) as sm:
            rep = sm.report()["model"]["gguf"]
            assert (rep["packed"] > 0) == bool(packed), rep
            logits[packed] = sm.forward(ids, cache=sm.new_cache()).float()[0, -1]
    a, b = logits[1], logits[0]
    assert torch.allclose(a, b, atol=0.05 * b.abs().max().item()), (a - b).abs().max().item()
