# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""GGUF models (btb/gguf.py): the fixtures' GGUF twins (tests/make_fixtures.py gguf) read through the gguf
package - the config off the metadata, every tensor of the family mapped through llama.cpp's table, a bf16
file decoding token for token as its safetensors twin, an f16 file within f16's rounding of it, a quantized
file as the numbers llama.cpp dequantizes, a gpt-oss or qwen4exp file's MXFP4 experts streamed and multiplied
as stored (ggml's layout, in the CPU and Metal kernels), discovery and resolution of .gguf paths. The tokenizer
conversion (transformers') is not exercised here: the fixtures' byte-level vocabulary has no merges, which that
converter refuses; it is checked on a real release."""

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import torch

from btb.kinds import PassTag, Quant, quant_tag, stored_quants
from tests.cert.manifest import QWEN3_06B_GGUF

if TYPE_CHECKING:
    import mlx.core as mx_

    from btb.engine.scratch import Scratch
    from btb.gguf import GGUFModel, QuantWeight
    from btb.mlx.backend import Streams

from tests.helpers import (
    F16_AT,
    FIXTURES,
    GGUF_FIXTURES,
    NO_LOG,
    assert_close,
    cached,
    fixture,
    forward_logits,
    host_model,
    layer_count,
    loaded_model,
    max_abs,
    mxfp4_random,
    native_library,
    need_cached,
    need_mlx,
    safetensors_state,
    speculation,
    stored_blocks,
)

GGUF = GGUF_FIXTURES
PROMPT = [3, 17, 42, 99, 7, 250, 11, 64]


def _need() -> None:
    if not os.path.isfile(os.path.join(GGUF, "tiny_qwen3-bf16.gguf")):
        pytest.skip("the GGUF fixtures are not here (tests/make_fixtures.py gguf)")


def _tokens(path: str, device: str = "cpu", **load_kw: Any) -> list[int]:
    with loaded_model(path, device=device, **load_kw) as sm:
        out = sm.generate(PROMPT, 8, eos=(), speculate=False).tokens
    return [int(t) for t in out]


def _agrees_where_decisive(path: str, ref: str, device: str = "cpu") -> None:
    """the file's greedy decode against `ref`, a model of the numbers it dequantizes to: over `ref`'s own tokens,
    every step whose two leaders sit further apart than the two models' logits differ picks the same one. The
    as-stored path decodes each weight exactly where `ref` holds its bf16 rounding, so a near-tie may go either way"""
    want = _tokens(ref, device)
    forced = []
    for p in (ref, path):
        with loaded_model(p, device=device) as sm:
            lf = sm.forward(PROMPT + want, cache=sm.new_cache(), last_only=False)
            assert lf is not None
            forced.append(lf.float()[0, len(PROMPT) - 1 :])
    b, a = forced
    for j in range(len(want)):
        gap = float(b[j].topk(2).values.diff().abs())
        if gap > 2 * max_abs(a[j], b[j]):
            assert a[j].argmax().item() == b[j].argmax().item(), (path, device, j, gap)


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


def _tier_matvec(device: str, w: "QuantWeight", x: np.ndarray) -> np.ndarray:
    """y = x . W^T over `w`'s stored blocks on a tier's own kernel for its type, in float32"""
    if device == "cpu":
        from btb.engine.native import Native

        gq = Native.gemv_quant
        assert gq is not None and w.q in gq, f"the native library has no {w.q.value} gemv: rebuild it"
        y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32)
        gq(w, torch.from_numpy(x).contiguous(), y)
        return y.numpy()
    need_mlx()
    import mlx.core as mx

    from btb.mlx.backend import stored_matvec, stored_streams

    streams = stored_streams(w.q, w.raw.numpy())
    return np.array(stored_matvec(w.q, streams, mx.array(x), *w.shape).astype(mx.float32))


@pytest.mark.parametrize("device", ["cpu", "mlx"])
@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
@pytest.mark.parametrize("tiny", [False, True], ids=["wide", "tiny"])
def test_each_stored_types_gemv_is_ggufs_dequantization(device: str, q: Quant, tiny: bool) -> None:
    """every type kinds.stored_quants() names, on each tier's own kernel for it: the matvec over the file's
    blocks equals `x @ gguf.dequantize(blocks)` to f32 rounding against each output's conditioning, and a row
    of a pass of any width up to 16 - each launches its own shape, some padded - is bit-for-bit the row alone (a
    verify pass computes each row as the one-row step does). `tiny`: one row of two blocks, whose one-a-block
    streams MLX would pass in constant memory"""
    from gguf import GGMLQuantizationType, dequantize

    from btb.gguf import QuantWeight, block_of

    native_library()
    assert set(F16_AT) == set(stored_quants()), set(F16_AT) ^ set(stored_quants())
    rows, cols = (1, 2 * block_of(q)[0]) if tiny else (40, 512)
    raw = stored_blocks(q, rows, cols, stored_quants().index(q))
    w = QuantWeight.of(torch.from_numpy(raw), q, rows, cols)
    ref = dequantize(raw.reshape(rows, -1), GGMLQuantizationType[q.value]).astype(np.float64).reshape(rows, cols)
    assert np.isfinite(ref).all()
    x = np.random.default_rng(1).standard_normal((16, cols)).astype(np.float32)
    y = _tier_matvec(device, w, x)
    want, mag = x.astype(np.float64) @ ref.T, np.abs(x.astype(np.float64)) @ np.abs(ref.T)
    assert float(np.max(np.abs(y - want) / np.maximum(mag, 1e-30))) < 1e-5, q.value
    for t in range(1, len(x)):
        assert np.array_equal(_tier_matvec(device, w, x[:t]), y[:t]), f"{q.value}: a {t}-row pass differs"
    for i in (7, 15):
        assert np.array_equal(_tier_matvec(device, w, x[i : i + 1])[0], y[i]), f"{q.value}: row {i} alone differs"


@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
def test_joined_stored_weights_are_their_members_in_one_launch(q: Quant) -> None:
    """resident stored weights of one type joined for one launch (q/k/v, gate/up on MLX): each row of the join's
    matvec is bit-for-bit its member's at every pass width; a member, now a view of the join, gives what it gave
    before; a slot's weights are never joined (the slot is rewritten in place)"""
    need_mlx()
    import mlx.core as mx

    from btb.mlx.backend import Weight, _join_stored, stored_matvec, stored_streams

    cols = 512
    ws = [
        Weight(q=q, streams=stored_streams(q, stored_blocks(q, r, cols, i)), shape=(r, cols))
        for i, r in enumerate((40, 24, 16))
    ]
    x = mx.array(np.random.default_rng(3).standard_normal((16, cols)).astype(np.float32))
    before = [np.array(stored_matvec(q, w.streams, x, *w.shape)) for w in ws]
    slot = Weight(q=q, streams=ws[0].streams, shape=ws[0].shape, slot=True)
    assert _join_stored([slot, ws[1]]) is None
    j = _join_stored(ws)
    assert j is not None and j.shape == (80, cols)
    for t in range(1, 17):
        want = np.concatenate([b[:t] for b in before], axis=1)
        assert np.array_equal(np.array(stored_matvec(q, j.streams, x[:t], *j.shape)), want), (q.value, t)
    for w, b in zip(ws, before, strict=True):
        assert np.array_equal(np.array(stored_matvec(q, w.streams, x, *w.shape)), b), q.value


@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
def test_a_stored_weights_first_rows_are_its_rows(q: Quant) -> None:
    """a stored weight's first rows (`Weight.first_rows`, a drafter's vocabulary slice of the head) are a weight over
    a prefix of each of its streams: its matvec the whole one's first outputs bit for bit, no bf16 copy made"""
    need_mlx()
    import mlx.core as mx

    from btb.mlx.backend import Weight, stored_matvec, stored_streams

    rows, cols, n = 40, 512, 24
    w = Weight(q=q, streams=stored_streams(q, stored_blocks(q, rows, cols, 5)), shape=(rows, cols))
    x = mx.array(np.random.default_rng(6).standard_normal((3, cols)).astype(np.float32)).astype(mx.bfloat16)
    head = w.first_rows(n)
    assert head.shape == (n, cols) and head.a is None and w.a is None
    want = stored_matvec(q, w.streams, x, rows, cols)[:, :n]
    assert mx.array_equal(stored_matvec(q, head.streams, x, n, cols), want), q.value


@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
def test_pooled_stored_weights_join_as_a_view_of_their_block(q: Quant) -> None:
    """stored weights of each type laid out in one pool block as a GGUF binds them (`kinds.pooled_layout`, then
    `weight_stored_pool`: the blocks in place, a split type's streams made once): each matvec is the type's over its
    own streams, and their join (q/k/v, gate/up) a view a stream over the same bytes, each of its rows bit-for-bit its
    member's at every pass width; members out of order or of another block are not joined at all (a copy would hold
    the block's bytes twice)"""
    need_mlx()
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.kinds import pooled_layout
    from btb.mlx import Shared
    from btb.mlx.backend import Weight, stored_matvec, stored_streams

    be = mlxdev.Backend.__new__(mlxdev.Backend)
    cols, base, heights = 512, 64, (40, 24, 16)

    def pool(raws: list[np.ndarray], heights: Sequence[int]) -> tuple[Shared, list[Weight]]:
        offs, size = pooled_layout([(q, r.size) for r in raws])
        sh = mlxdev.Shared(base + size)
        ws = []
        for raw, rows, at in zip(raws, heights, offs, strict=True):
            parts = [np.asarray(s).reshape(-1).view(np.uint8) for s in stored_streams(q, raw)]
            for o, p in zip(at, parts, strict=True):
                sh.torch[base + o : base + o + p.size] = torch.from_numpy(p.copy())
            ws.append(be.weight_stored_pool(sh, [base + o for o in at], [p.size for p in parts], q, (rows, cols)))
        return sh, ws

    raws = [stored_blocks(q, r, cols, i) for i, r in enumerate(heights)]
    sh, ws = pool(raws, heights)
    x = mx.array(np.random.default_rng(3).standard_normal((16, cols)).astype(np.float32))
    before = [np.array(stored_matvec(q, w.streams, x, *w.shape)) for w in ws]
    for raw, w, b in zip(raws, ws, before, strict=True):
        assert np.array_equal(np.array(stored_matvec(q, stored_streams(q, raw), x, *w.shape)), b), q.value
    j = be.weight_fused(ws)
    assert j is not None and j.shape == (80, cols) and j.sh is sh and j.offs == ws[0].offs
    assert j.nbs == tuple(sum(w.nbs[i] for w in ws) for i in range(len(j.nbs))), q.value
    for t in range(1, 17):
        want = np.concatenate([b[:t] for b in before], axis=1)
        assert np.array_equal(np.array(stored_matvec(q, j.streams, x[:t], *j.shape)), want), (q.value, t)
    _, (apart,) = pool([raws[1]], heights[1:2])
    for bad in ([ws[1], ws[0]], [ws[0], apart]):
        assert be.weight_fused(bad) is None, q.value


@pytest.mark.parametrize("q", [*stored_quants(), Quant.F16, Quant.BF16], ids=lambda q: q.value)
def test_a_stored_prefills_rows_are_its_steps_on_mlx(q: Quant) -> None:
    """every stored type (and a GGUF's float experts) through `Backend.matmul`, x bf16 and float32: a pass of each
    width - the type's matvec, past `gemm_rows` its dequant and the gemm - is its rows' one-row steps bit for bit (a
    prefix hit's tail and a cold prefill, a verify pass and its steps, agree), the matvec is gemv over the dequant's
    weights and x in the chain's column order (a 32-wide type's row a half group past whole 64-groups), and the
    embedding gather is the dequant's rows"""
    need_mlx()
    import importlib

    import mlx.core as mx

    from btb.mlx.backend import Backend, Stat, Weight, stored_dequant, stored_matvec, stored_streams
    from btb.mlx.matvec import chain_x
    from btb.mlx.stored import STORED

    gemv = importlib.import_module("btb.mlx.gemv").gemv
    be = Backend.__new__(Backend)
    be.gemm_rows = 64
    be.stat = Stat(linears=0, evals=0, s=0.0, layers=0, layer_s=0.0)
    floats = q in (Quant.F16, Quant.BF16)
    rows, widths = 72, (2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100)
    cols = 544 if not floats and STORED[q].bw == 32 else 512
    if floats:
        f = np.random.default_rng(9).standard_normal((rows, cols)).astype(np.float32) * 0.05
        bits = f.astype(np.float16) if q is Quant.F16 else (f.view(np.uint32) >> 16).astype(np.uint16)
        raw = bits.view(np.uint8).reshape(-1)
    else:
        raw = stored_blocks(q, rows, cols, 9)
    streams = stored_streams(q, raw)
    w = Weight(q=q, streams=streams, shape=(rows, cols))
    for dt in (mx.bfloat16, mx.float32):
        x = mx.array(np.random.default_rng(10).standard_normal((max(widths), cols)).astype(np.float32)).astype(dt)
        one = mx.concatenate([stored_matvec(q, streams, x[i : i + 1], rows, cols) for i in range(max(widths))])
        for t in widths:
            assert mx.array_equal(be.matmul(x[:t], w), one[:t]), (q.value, str(dt), t)
        if not floats:
            deq = stored_dequant(q, streams, rows, cols, dt, chain=True)
            got = gemv(deq, chain_x(x[:16], cols))
            assert mx.array_equal(got, one[:16]), (q.value, str(dt), "matvec against gemv(dequant)")
    if not floats:
        tok = mx.array([3, 0, rows - 1, 5], dtype=mx.int32)
        assert mx.array_equal(Backend.gather_rows(w, tok), stored_dequant(q, streams, rows, cols)[tok]), q.value


@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
def test_each_stored_types_dequant_feeds_the_prefills_gemm(q: Quant) -> None:
    """the native decode of a whole matrix a CPU prefill's GEMM takes, every stored type: the package's
    dequantization to f32 rounding; and `stored_matmul` over 64 rows (the GEMM over those weights)
    `x @ dequantize(blocks)` to bf16 rounding against each output's conditioning"""
    from gguf import GGMLQuantizationType, dequantize

    from btb.engine.host import stored_matmul
    from btb.engine.native import Native
    from btb.gguf import QuantWeight

    native_library()
    gq = Native.gemv_quant
    assert gq is not None and q in gq.dequants, f"the native library has no {q.value} dequant: rebuild it"
    rows, cols = 40, 512
    raw = stored_blocks(q, rows, cols, stored_quants().index(q))
    w = QuantWeight.of(torch.from_numpy(raw), q, rows, cols)
    ref = dequantize(raw.reshape(rows, -1), GGMLQuantizationType[q.value]).astype(np.float64).reshape(rows, cols)
    f32 = torch.empty(rows, cols)
    gq.dequant(w, f32)
    err = np.abs(f32.numpy() - ref).max(axis=1) / np.maximum(np.abs(ref).max(axis=1), 1e-30)
    assert float(err.max()) < 1e-6, q.value
    x = np.random.default_rng(2).standard_normal((64, cols)).astype(np.float32)
    y = stored_matmul(w, torch.from_numpy(x), None).numpy()
    want, mag = x.astype(np.float64) @ ref.T, np.abs(x.astype(np.float64)) @ np.abs(ref.T)
    assert float(np.max(np.abs(y - want) / np.maximum(mag, 1e-30))) < 1e-2, q.value


@pytest.mark.parametrize("q", [Quant.BF16, Quant.F16, *stored_quants()], ids=lambda q: q.value)
def test_the_reader_decodes_each_type_to_the_packages_numbers(q: Quant) -> None:
    """every tensor of a type's twin, and each expert of a mixture's, decoded by btb (`GGUFModel._float32`): the
    gguf package's numbers to the bit but a zero's sign"""
    from gguf import dequantize

    from btb.gguf import GGUFModel

    native_library()
    for stem in ("tiny_qwen3", "tiny_q35moe"):
        g = GGUFModel(os.path.join(GGUF, f"{stem}-{q.value.lower()}.gguf"))
        n = 0
        for name, t in g.tensors.items():
            if t.tensor_type.name != q.value:
                continue
            data = np.asarray(t.data)
            for e in range(data.shape[0]) if name.endswith("_exps.weight") else (None,):
                got = g._float32(name, e)
                want = np.asarray(dequantize(data if e is None else data[e], t.tensor_type))
                assert got.shape == want.shape and np.array_equal(got, want), (stem, name, e)
                assert not np.any((got.view(np.uint32) != want.view(np.uint32)) & (want != 0)), (stem, name, e)
                n += 1
        assert n, f"{stem}: no {q.value} tensor"


@pytest.mark.parametrize("device", ["cpu", "mlx"])
def test_a_mixture_with_f16_down_experts_keeps_them_stored(device: str, tmp_path: Path) -> None:
    """a Q4_0 file whose down experts are F16, as llama.cpp's quantizer leaves a type whose rows are no whole blocks
    (Qwen3.5-MoE 0.87B's 400 columns under Q2_K): the store holds every expert as the file's, down multiplied in
    f16 exactly; the logits are those of the experts decoded into bf16 slots to bf16's rounding, and a speculative
    decode, its verify rows through the same kernels, is the plain one"""
    from btb.gguf import GGUFModel
    from tests.make_fixtures import retyped_gguf

    if device == "mlx":
        need_mlx()
    native_library()
    src = os.path.join(GGUF, "tiny_q35moe-q4_0.gguf")
    out = str(tmp_path / "tiny_q35moe-q4_0-down-f16.gguf")
    retyped_gguf(src, out, {n for n in GGUFModel(src).tensors if n.endswith("ffn_down_exps.weight")}, Quant.F16)
    logits = []
    for packed in (1, 0):
        with loaded_model(out, device=device, gguf_packed=packed) as sm:
            cache = sm.new_cache()
            lg = [forward_logits(sm, PROMPT, cache)[0, -1]]
            for _ in range(4):
                lg.append(forward_logits(sm, [[int(lg[-1].argmax())]], cache)[0, -1])
            logits.append(torch.stack(lg).float())
            store = sm.expert_store
            assert store is not None
            if not packed:
                assert store.dequant and not store.types, "gguf_packed off: the experts decoded into bf16 slots"
                continue
            assert store.types and all(t == (Quant.Q4_0, Quant.Q4_0, Quant.F16) for t in store.types.values()), (
                store.types
            )
            plain = sm.generate_greedy(PROMPT, 12)
            speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, v_max=4, price=False)
            toks, census = sm.generate_speculative(PROMPT, 12, proposer="mtp_tree", v_max=4)
            assert toks == plain and census["proposed"] > 0, (toks, plain, census["proposed"])
    # bf16's rounding of the decoded weights: ~0.3% of the largest logit on the cold-slot test's model
    assert_close(logits[0], logits[1], atol_frac=0.02, msg=device)


def test_a_refused_prefill_decode_takes_the_types_gemv() -> None:
    """a prefill whose decoded weight the scheduler has no room for: `stored_matmul`'s rows through the type's gemv,
    each row the bits its step gives, instead of the refusal escaping"""
    from btb.engine.host import stored_matmul
    from btb.engine.native import Native
    from btb.engine.scheduler import MemoryGrantError
    from btb.gguf import QuantWeight

    native_library()
    gq = Native.gemv_quant
    assert gq is not None and Quant.Q4_K in gq.dequants

    class Refusing:
        def take(self, *_a: object) -> torch.Tensor:
            raise MemoryGrantError("no room for the decoded weight")

    rows, cols = 40, 512
    w = QuantWeight.of(torch.from_numpy(stored_blocks(Quant.Q4_K, rows, cols, 3)), Quant.Q4_K, rows, cols)
    x = torch.from_numpy(np.random.default_rng(4).standard_normal((Native.gemm_rows, cols)).astype(np.float32))
    want = torch.empty(Native.gemm_rows, rows)
    gq(w, x, want)
    assert torch.equal(stored_matmul(w, x, cast("Scratch", Refusing())), want)


@pytest.mark.parametrize("q", [Quant.IQ4_NL, Quant.IQ1_M], ids=lambda q: q.value)
def test_a_cold_layer_repacks_its_types_streams_from_each_fill_on_mlx(q: Quant) -> None:
    """IQ4_NL's split streams and IQ1_M's side stream are repacked from a cold layer's slot each time the ring
    refills it (`Weight.invalidate`), so two cold layers taking turns in one slot give the logits they give resident
    (the tiny model's greedy tokens can hide a layer read from the other's weights)"""
    need_mlx()
    path = os.path.join(GGUF, f"tiny_qwen3-{q.value.lower()}.gguf")
    L = layer_count(fixture("tiny_qwen3"))
    logits = []
    for cold in ((), range(1, L - 1)):
        sm = host_model(path, device="mlx", dtype=torch.bfloat16, cpu_layers=range(L), cold_layers=cold, cold_slots=1)
        try:
            assert sorted(sm.cold) == list(cold), sm.cold
            assert not cold or len(sm.cold_ring.shared or ()) == 1, "the cold layers do not share one MLX slot"
            cache = sm.new_cache()
            lg = [forward_logits(sm, PROMPT, cache)[0, -1]]
            for _ in range(4):
                lg.append(forward_logits(sm, [[int(lg[-1].argmax())]], cache)[0, -1])
            logits.append(torch.stack(lg).float())
        finally:
            sm.close()
    # the two paths' bf16 rounding is ~0.3% of the largest logit; a layer read from the other's weights ~70%
    assert_close(logits[1], logits[0], atol_frac=0.02, msg=q.value)


@pytest.mark.parametrize("device", ["cpu", "mlx"])
def test_a_long_gguf_prefill_takes_the_gemm_as_a_bf16_one_does(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """a prompt past both GEMM row counts: the as-stored file's prefill decodes its weights once for the GEMM (CPU
    `QuantGemv.dequant`, MLX `stored_dequant`) instead of its type's matvec a row tile at a time, and its last
    logits sit within bf16 rounding of the dequantized path's"""
    native_library()
    if device == "mlx":
        need_mlx()
    import btb.mlx.backend as backend
    from btb.engine.native import QuantGemv
    from btb.gguf import QuantWeight

    path = os.path.join(GGUF, "tiny_qwen3-q4_0.gguf")
    prompt = [(7 * i + 3) % 250 + 1 for i in range(80)]
    decoded: list[int] = []
    deq, sdq = QuantGemv.dequant, backend.stored_dequant

    def native_dequant(gq: QuantGemv, w: QuantWeight, out: torch.Tensor) -> None:
        decoded.append(w.shape[0])
        deq(gq, w, out)

    def mlx_dequant(
        q: Quant, streams: "Streams", rows: int, cols: int, dtype: "mx_.Dtype | None" = None, chain: bool = False
    ) -> "mx_.array":
        decoded.append(rows)
        return sdq(q, streams, rows, cols, dtype, chain)

    monkeypatch.setattr(QuantGemv, "dequant", native_dequant)
    monkeypatch.setattr(backend, "stored_dequant", mlx_dequant)
    logits = {}
    for packed in (1, 0):
        with loaded_model(path, device=device, gguf_packed=packed) as sm:
            decoded.clear()  # the pass's decodes: the load decodes the file's other tensors through `dequant` too
            lg = sm.forward(prompt, cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
            assert (quant_tag(Quant.Q4_0) in sm.last_pass_report().tags) is bool(packed)
        assert bool(decoded) is bool(packed), (device, packed, len(decoded))
    assert_close(logits[1], logits[0])


def _mxfp4_pair(rows: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """random MXFP4 bytes for a [rows, k] matrix in both layouts: (blocks, scales) and ggml's raw blocks"""
    from btb.mxfp4 import hf_to_ggml

    blocks, scales = mxfp4_random(seed, rows, k, 112, 142)
    return blocks, scales, hf_to_ggml(blocks, scales)


def test_the_cpu_mxfp4_kernel_reads_ggml_blocks_as_the_checkpoints() -> None:
    """the native matvec over ggml's layout gives the checkpoint layout's bits for the same weights, alone
    and grouped, at every batch width the host path uses"""
    _need()
    from btb.engine.native import Native
    from btb.mxfp4 import MxWeight

    native_library()
    assert Native.gemv_mx4_ggml is not None, "the native library predates ggml's layout: rebuild it"
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
    from btb.mxfp4 import GateUp

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
        out = sm.generate(PROMPT, 8, eos=(), speculate=False).tokens
        store = sm.expert_store
        if store is not None:
            assert store.ggml and isinstance(store._views(store.last_slots[next(iter(store.last_slots))])[0], GateUp)
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
    """Q8_0 and Q4_0: the file read as bf16 slots (`gguf_packed=0`) decodes exactly as a safetensors model built from
    the numbers the gguf package dequantizes (llama.cpp's own arithmetic), and as stored agrees wherever a step is
    decisive; the numbers differ from the bf16 twin's as a quantization does"""
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
        assert _tokens(path, gguf_packed=0) == _tokens(str(twin)), qt
        _agrees_where_decisive(path, str(twin))


# Qwen3.5 and its mixture, both through the Qwen3.5 converter's rewrites
Q35_STEMS = ("tiny_q35", "tiny_q35moe")
# the mixture's MLP widths, in place of the dense one's intermediate_size
MOE_KEYS = ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size")


def _need_q35(stem: str = "tiny_q35") -> None:
    if not os.path.isfile(os.path.join(GGUF, f"{stem}-bf16.gguf")):
        pytest.skip(f"the {stem} GGUF fixtures are not here (tests/make_fixtures.py gguf_twins)")


def _q35_trunk(stem: str = "tiny_q35") -> dict[str, torch.Tensor]:
    """the checkpoint's trunk, its MTP head aside"""
    state = safetensors_state(os.path.join(FIXTURES, stem))
    return {k: v for k, v in state.items() if not k.startswith("mtp.")}


def _q35_named(stem: str) -> dict[str, torch.Tensor]:
    """the checkpoint as the family names its tensors (`_family_tensor_names`): the trunk's and the drafting head's,
    a mixture head's one-tensor-an-expert matrices stacked as the store reads them (gate's rows, then up's)"""
    state = safetensors_state(os.path.join(FIXTURES, stem))
    pre = "mtp.layers.0.mlp.experts."
    n = len({k.split(".")[5] for k in state if k.startswith(pre)})
    if n:
        get = lambda e, p: state.pop(f"{pre}{e}.{p}.weight")
        state[pre + "gate_up_proj"] = torch.stack(
            [torch.cat([get(e, "gate_proj"), get(e, "up_proj")]) for e in range(n)]
        )
        state[pre + "down_proj"] = torch.stack([get(e, "down_proj") for e in range(n)])
    return state


@pytest.mark.parametrize("stem", Q35_STEMS)
def test_a_qwen35_gguf_reads_back_its_checkpoint(stem: str) -> None:
    """Qwen3.5 through llama.cpp's converter and back: the config off the typed keys (the hybrid's layer pattern,
    linear attention widths, partial MRoPE, a mixture's experts), every tensor - the drafting head's too, kept as a
    NextN block past the trunk's - found and equal as bf16 once the converter's rewrites are inverted (the tiled
    value heads, -exp(A_log), 1 + norm, the squeezed conv, the split experts, the head's renames), and a quantized
    file's reordered blocks (what the packed kernels bind) the same numbers as its reordered dequantization"""
    _need_q35(stem)
    from gguf import dequantize

    from btb.gguf import GGUFModel

    g = GGUFModel(os.path.join(GGUF, f"{stem}-bf16.gguf"))
    cfg = g.config()
    want = json.load(open(os.path.join(FIXTURES, stem, "config.json")))
    assert cfg.model_type == want["model_type"] == g.model_type
    for k in (
        *(MOE_KEYS if want.get("num_experts") else ("intermediate_size",)),
        "num_hidden_layers",
        "hidden_size",
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
    state = _q35_named(stem)
    assert g.nextn() == 1 and any(k.startswith("mtp.") for k in state)
    names, hdr = g.weight_map(list(state), cfg.num_hidden_layers)
    assert set(names) == set(state), set(state) ^ set(names)
    for hf, gn in names.items():
        t = g.get(gn)
        assert t.dtype == torch.bfloat16 and tuple(t.shape) == tuple(state[hf].shape) == tuple(hdr[hf]["shape"]), hf
        assert torch.equal(t, state[hf].to(torch.bfloat16)), hf
    for qt in ("q8_0", "q4_0", "q4_1"):
        q = GGUFModel(os.path.join(GGUF, f"{stem}-{qt}.gguf"))
        qnames, _ = q.weight_map(list(state), cfg.num_hidden_layers)
        reordered = [gn for gn in qnames.values() if gn in q.undo and gn in q.tensors]
        assert any(q.undo[gn][1].cols is not None for gn in reordered), qt  # out_proj binds as stored too
        for gn in reordered:
            t = q.tensors[gn]
            got = dequantize(q.raw(gn).numpy(), t.tensor_type).reshape(q.get(gn).shape)
            assert torch.equal(torch.from_numpy(np.array(got, dtype=np.float32)).bfloat16(), q.get(gn)), (qt, gn)


@pytest.mark.parametrize("stem", Q35_STEMS)
def test_a_qwen35_bf16_gguf_decodes_as_its_safetensors_twin(stem: str) -> None:
    _need_q35(stem)
    twin = os.path.join(FIXTURES, stem)
    assert _tokens(os.path.join(GGUF, f"{stem}-bf16.gguf")) == _tokens(twin)


@pytest.mark.parametrize("stem", Q35_STEMS)
def test_a_qwen35_bf16_gguf_decodes_as_its_safetensors_twin_on_mlx(stem: str) -> None:
    _need_q35(stem)
    need_mlx()
    twin = os.path.join(FIXTURES, stem)
    assert _tokens(os.path.join(GGUF, f"{stem}-bf16.gguf"), "mlx") == _tokens(twin, "mlx")


def test_a_squeezed_qwen35moe_gguf_reads_and_decodes_as_the_converters(tmp_path: Path) -> None:
    """Qwen3.5-MoE as an older converter wrote it, each size-1 axis squeezed (the shared-expert gate a vector):
    every tensor reads back as the unsqueezed file's at its checkpoint shape, and the file decodes as its
    safetensors twin on the CPU and on MLX"""
    stem = "tiny_q35moe"
    _need_q35(stem)
    from btb.gguf import GGUFModel
    from tests.make_fixtures import squeezed_gguf

    src, out = os.path.join(GGUF, f"{stem}-bf16.gguf"), str(tmp_path / f"{stem}-squeezed.gguf")
    squeezed_gguf(src, out)
    a, b = GGUFModel(src), GGUFModel(out)
    assert any(len(b._shape(n)) < len(a._shape(n)) for n in a.tensors)
    state, layers = _q35_named(stem), a.config().num_hidden_layers
    (na, ha), (nb, hb) = a.weight_map(list(state), layers), b.weight_map(list(state), layers)
    assert set(na) == set(nb) == set(state)
    for hf in na:
        assert hb[hf]["shape"] == ha[hf]["shape"], hf
        assert torch.equal(b.get(nb[hf]), a.get(na[hf])), hf
    twin = os.path.join(FIXTURES, stem)
    assert _tokens(out) == _tokens(twin)
    need_mlx()
    assert _tokens(out, "mlx") == _tokens(twin, "mlx")


@pytest.mark.parametrize("stem", Q35_STEMS)
def test_a_quantized_qwen35_gguf_decodes_as_its_dequantized_weights(tmp_path: Path, stem: str) -> None:
    """each affine type: the file read as bf16 slots (`gguf_packed=0`, a mixture's experts too) decodes exactly as a
    safetensors model built from the numbers the reader dequantizes and un-rewrites, and as stored agrees wherever
    a step is decisive; those numbers differ from the bf16 twin's as a quantization does"""
    _need_q35(stem)
    from safetensors.torch import save_file

    from btb.gguf import GGUFModel

    src = os.path.join(FIXTURES, stem)
    state = _q35_trunk(stem)
    for qt in ("q8_0", "q4_0", "q4_1"):
        path = os.path.join(GGUF, f"{stem}-{qt}.gguf")
        g = GGUFModel(path)
        names, _ = g.weight_map(list(state), g.config().num_hidden_layers)
        deq = {hf: g.get(gn).contiguous() for hf, gn in names.items()}
        assert any(not torch.equal(deq[k], state[k].to(torch.bfloat16)) for k in deq), qt
        twin = tmp_path / f"twin-{qt}"
        twin.mkdir()
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(os.path.join(src, fn), twin / fn)
        save_file(deq, str(twin / "model.safetensors"), metadata={"format": "pt"})
        assert _tokens(path, gguf_packed=0) == _tokens(str(twin)), qt
        _agrees_where_decisive(path, str(twin))


@pytest.mark.parametrize("stem", Q35_STEMS)
def test_a_qwen35_gguf_multiplies_its_reordered_blocks_as_stored_on_mlx(stem: str) -> None:
    """the value-head reordering applied to the stored blocks (rows of in_proj_*, whole blocks of out_proj's
    columns): the packed path's first-step logits sit within bf16 rounding of the dequantized path's"""
    _need_q35(stem)
    need_mlx()
    logits = {}
    for packed in (1, 0):
        with loaded_model(os.path.join(GGUF, f"{stem}-q8_0.gguf"), device="mlx", gguf_packed=packed) as sm:
            rep = sm.report()["model"]["gguf"]
            assert (rep["packed"] > 0) == bool(packed), rep
            lg = sm.forward([PROMPT], cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
    assert_close(logits[1], logits[0])


# --- qwen4exp: tiny_q4's twins as llama.cpp's own converter writes them (tests/make_fixtures.py make_q4_gguf) ---

Q4 = os.path.join(FIXTURES, "tiny_q4")
NGRAM = ".ple.ple_embedding.ngram_embedding.shard_"


def _q4_state() -> dict[str, torch.Tensor]:
    """tiny_q4's trunk with its n-gram shards as the one table the GGUF stores (`shard_0`): the drafting head
    (`mtp.*`) left out, as llama.cpp's converter leaves it (make_fixtures strips it before converting)"""
    state = {k: v for k, v in safetensors_state(Q4).items() if not k.startswith("mtp.")}
    for pre in sorted({k.partition(NGRAM)[0] for k in state if NGRAM in k}):
        parts = sorted((k for k in state if k.startswith(pre + NGRAM)), key=lambda k: (len(k), k))
        state[pre + NGRAM + "0.weight"] = torch.cat([state.pop(k) for k in parts])
    return state


def _q4_names(g: "GGUFModel") -> dict[str, str]:
    names, _ = g.weight_map(
        [k for k in safetensors_state(Q4) if not k.startswith("mtp.")], g.config().num_hidden_layers
    )
    return names


def test_the_qwen4exp_table_is_llama_cpps() -> None:
    """btb's own qwen4exp names and keys are the gguf package's once a release knows the architecture; until
    then they name exactly the tensors llama.cpp's converter wrote, every one of them read"""
    _need()
    from gguf import MODEL_ARCH_NAMES, MODEL_TENSOR, TENSOR_NAMES, TensorNameMap

    from btb.gguf import OWN_KEYS, OWN_NAMES, OWN_TENSOR_NAMES, GGUFModel, knows_arch, meta_key
    from btb.hf import QWEN4EXP

    g = GGUFModel(os.path.join(GGUF, "tiny_q4-bf16.gguf"))
    if knows_arch(QWEN4EXP):
        L = int(g.config().num_hidden_layers)
        tmap = TensorNameMap(next(a for a, n in MODEL_ARCH_NAMES.items() if n == QWEN4EXP), L)
        for hf, name in OWN_NAMES[QWEN4EXP].items():
            assert tmap.get_name(hf.format(bid=1)) == name.format(bid=1), hf
        for member, name in OWN_TENSOR_NAMES[QWEN4EXP].items():
            assert TENSOR_NAMES[MODEL_TENSOR[member]] == name, member
        for path, key in OWN_KEYS.items():
            assert meta_key(QWEN4EXP, path) == key.format(arch=QWEN4EXP), path
    names = _q4_names(g)
    read = {g.undo[n][0] if n in g.undo else n for n in names.values() if n not in g.derived}
    read |= {f for fs, _ in g.derived.values() for f in fs}
    assert read == set(g.tensors), set(g.tensors) ^ read


def test_the_qwen4exp_reader_inverts_llama_cpps_converter() -> None:
    """the bf16 twin: the config transformers reads off the checkpoint, rebuilt from llama.cpp's keys; every
    tensor of the checkpoint equal as bf16 once the converter's rewrites are undone (V heads regrouped, A_log,
    the +1 gammas, the squeezed convolutions, the split indexer and experts, the n-gram table and its hash
    constants); the f16 twin within f16's rounding of it"""
    _need()
    from transformers import AutoConfig

    from btb.gguf import GGUFModel

    g = GGUFModel(os.path.join(GGUF, "tiny_q4-bf16.gguf"))
    cfg, want = g.config(), AutoConfig.from_pretrained(Q4)
    assert g.model_type == cfg.model_type == want.model_type
    # the n-gram hash's seed and the shard count are not in the file (its constants and its one table stand in),
    # nor the checkpoint's provenance; the file holds the epsilon as a float32
    unread = {"seed", "split_ngram_parts", "ngram_vocab_size_base", "dtype", "architectures", "transformers_version"}
    same = set(want.to_diff_dict()) - unread - {"rms_norm_eps"}
    assert {k: getattr(cfg, k) for k in same} == {k: getattr(want, k) for k in same}
    assert np.float32(cfg.rms_norm_eps) == np.float32(want.rms_norm_eps)
    state = _q4_state()
    names = _q4_names(g)
    assert set(names) == set(state), set(state) ^ set(names)
    for hf, gn in names.items():
        t = g.get(gn)
        assert t.dtype == state[hf].dtype and tuple(t.shape) == tuple(state[hf].shape), hf
        assert torch.equal(t, state[hf]), hf
    h = GGUFModel(os.path.join(GGUF, "tiny_q4-f16.gguf"))
    diffs = [max_abs(h.get(gn).float(), state[hf].float()) for hf, gn in _q4_names(h).items()]
    assert max(diffs) < 1e-3 and any(d > 0 for d in diffs), max(diffs)


def test_a_bf16_qwen4exp_gguf_decodes_as_its_safetensors_twin() -> None:
    _need()
    assert _tokens(os.path.join(GGUF, "tiny_q4-bf16.gguf")) == _tokens(Q4)


def _q4_dequantized_twin(path: str, dest: Path) -> str:
    """a tiny_q4 checkpoint of the numbers the GGUF at `path` dequantizes to, written at `dest`: the n-gram table
    split back into the checkpoint's shards, and experts the file keeps as MXFP4 blocks (read by the store, not
    mapped to an HF name) rebuilt as the checkpoint's gate/up and down tables"""
    from safetensors.torch import save_file

    from btb.gguf import GGUFModel

    g = GGUFModel(path)
    deq = {hf: g.get(gn).contiguous() for hf, gn in _q4_names(g).items() if not hf.endswith("_exps.weight")}
    assert any(not torch.equal(deq[k], v) for k, v in _q4_state().items() if k in deq), path
    for i in range(int(g.config().num_hidden_layers)):
        pre, exps = f"model.layers.{i}.mlp.experts.", f"blk.{i}.ffn_{{}}_exps.weight"
        if exps.format("gate") in g.tensors and pre + "gate_up_proj" not in deq:
            deq[pre + "gate_up_proj"] = torch.cat([g.get(exps.format("gate")), g.get(exps.format("up"))], dim=1)
            deq[pre + "down_proj"] = g.get(exps.format("down")).contiguous()
    shards: dict[str, list[int]] = {}  # the checkpoint's shard rows, in order, by PLE layer
    for k, v in sorted(safetensors_state(Q4).items(), key=lambda kv: (len(kv[0]), kv[0])):
        if NGRAM in k:
            shards.setdefault(k.partition(NGRAM)[0], []).append(int(v.shape[0]))
    for pre, rows in shards.items():
        for j, part in enumerate(torch.split(deq.pop(pre + NGRAM + "0.weight"), rows)):
            deq[pre + NGRAM + f"{j}.weight"] = part.contiguous()
    dest.mkdir()
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(Q4, fn), dest / fn)
    save_file(deq, str(dest / "model.safetensors"), metadata={"format": "pt"})
    return str(dest)


def test_a_quantized_qwen4exp_gguf_decodes_as_its_dequantized_weights(tmp_path: Path) -> None:
    """Q8_0, Q4_0, Q4_1 and IQ4_NL: the file decodes as a checkpoint of the numbers it dequantizes to - exactly with
    `gguf_packed` off, its experts dequantized into bf16 slots on read; as stored, its experts read by the store as
    the file's blocks and multiplied by their type's own kernel (on MLX IQ4_NL's streams repacked in the graph
    from the slot), agreeing wherever a step is decisive"""
    _need()
    from btb.engine.device import mlx_available
    from btb.mxfp4 import GateUp

    for qt in ("q8_0", "q4_0", "q4_1", "iq4_nl"):
        path = os.path.join(GGUF, f"tiny_q4-{qt}.gguf")
        twin = _q4_dequantized_twin(path, tmp_path / f"twin-{qt}")
        want = _tokens(twin)
        for packed in (1, 0):
            with loaded_model(path, device="cpu", gguf_packed=packed) as sm:
                out = [int(t) for t in sm.generate(PROMPT, 8, eos=(), speculate=False).tokens]
                store, tags = sm.expert_store, sm.last_pass_report().tags
                assert store is not None and bool(store.types) is bool(packed) and store.widened is not bool(packed)
                if packed:
                    assert isinstance(store._views(store.last_slots[next(iter(store.last_slots))])[0], GateUp)
                    types = {q for t in store.types.values() for q in t}
                    assert {PassTag.EXPERT_QUANT_ASSTORED, *map(quant_tag, types)} <= tags, qt
                else:
                    assert PassTag.EXPERT_QUANT_DEQUANT in tags, qt
                    assert out == want, qt
        _agrees_where_decisive(path, twin)
        if mlx_available():
            _agrees_where_decisive(path, twin, "mlx")


def test_a_qwen4exp_mxfp4_gguf_multiplies_its_experts_as_stored(tmp_path: Path) -> None:
    """llama.cpp's MXFP4_MOE of tiny_q4 (its experts MXFP4, every other matrix Q8_0): the expert store reads the
    experts straight out of the file in ggml's layout and multiplies them as stored through the family's own
    SiLU gate, and the file decodes as a checkpoint of its dequantized numbers on the CPU and on MLX"""
    _need()
    from btb.engine.device import mlx_available
    from btb.mxfp4 import GateUp

    path = os.path.join(GGUF, "tiny_q4-mxfp4.gguf")
    with loaded_model(path, device="cpu") as sm:
        assert "blk.0.ffn_gate_exps.weight" in sm.weight_map
        out = sm.generate(PROMPT, 8, eos=(), speculate=False).tokens
        store = sm.expert_store
        assert store is not None and store.ggml
        assert isinstance(store._views(store.last_slots[next(iter(store.last_slots))])[0], GateUp)
        assert PassTag.EXPERT_MXFP4_ASSTORED in sm.last_pass_report()
    twin = _q4_dequantized_twin(path, tmp_path / "twin")
    assert [int(t) for t in out] == _tokens(twin)
    if mlx_available():
        assert _tokens(path, "mlx") == _tokens(twin, "mlx")


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
            out = sm.generate(ids, 8, speculate=False).tokens
            answers[q] = (ids, [int(t) for t in out], sm.tokenizer.decode(out, skip_special_tokens=True))
    print({q: a[2] for q, a in answers.items()})
    if ref and "BF16" in answers:
        with loaded_model(ref, device="cpu") as sm:
            ids = sm.prompt_ids(prompt)
            out = sm.generate(ids, 8, speculate=False).tokens
        assert answers["BF16"][0] == ids and answers["BF16"][1] == [int(t) for t in out]


# the releases whose two paths no implementation holds to the same tokens: with exact kernels the 1-bit file's
# as-stored and dequantized forwards drift 43% of max-logit by the first token (2% on its 2- and 3-bit siblings),
# a smooth per-layer amplification of rounding order on every tier - so it is held to the first token alone
UNSTABLE = frozenset({"UD-IQ1_M"})


@pytest.mark.parametrize("device", ["cpu", "mlx"])
@pytest.mark.parametrize(
    "release", [n for n, gq in QWEN3_06B_GGUF.items() if gq.name in {q.value for q in stored_quants()}]
)
def test_a_release_multiplies_its_blocks_as_stored(device: str, release: str) -> None:
    """each Qwen3-0.6B release of a stored type, when cached, on each tier: one tensor of every stored type in
    the file multiplied by the tier's kernel equals the package's dequantization (real blocks carry the subnormal
    deltas random ones seldom do); loaded as stored, the pass reports the file's own type and its linears hold
    no bf16 copy; and the first-step logits sit within bf16 rounding of the dequantized path's, with the same
    greedy tokens on a stable file"""
    from gguf import dequantize

    from btb.engine.host import _HostLinear
    from btb.gguf import GGUFModel

    native_library()
    path = need_cached(f"{REAL_REPO}:Qwen3-0.6B-{release}.gguf", f"Qwen3-0.6B-{release}.gguf is not cached")
    g = GGUFModel(path)
    seen: set[str] = set()
    for name, t in g.tensors.items():
        if t.tensor_type.name in seen or not g.storable(name):
            continue
        w = g.stored(name)
        x = np.random.default_rng(0).standard_normal((1, w.shape[1])).astype(np.float32)
        ref = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float64).reshape(w.shape)
        y, want = _tier_matvec(device, w, x)[0], (x.astype(np.float64) @ ref.T)[0]
        assert float(np.abs(y - want).max() / (np.abs(want).max() + 1e-9)) < 1e-4, (release, name)
        seen.add(t.tensor_type.name)
    assert seen, release
    main = Quant(QWEN3_06B_GGUF[release].name)
    logits, toks, forced = {}, {}, {}
    for packed in (0, 1):
        with loaded_model(path, device=device, gguf_packed=packed) as sm:
            lg = sm.forward(PROMPT, cache=sm.new_cache())
            assert lg is not None
            logits[packed] = lg.float()[0, -1]
            out = sm.generate(PROMPT, 8, speculate=False)
            toks[packed] = [int(t) for t in out.tokens]
            tags = sm.last_pass_report().tags
            # both paths over the dequantized path's own tokens, every step's logits
            lf = sm.forward(PROMPT + toks[0], cache=sm.new_cache(), last_only=False)
            assert lf is not None
            forced[packed] = lf.float()[0, len(PROMPT) - 1 :]
            if packed:
                assert quant_tag(main) in tags, sorted(t.value for t in tags)
                lins = [m for layer in sm.host.values() for m in layer.modules() if isinstance(m, _HostLinear)]
                held = [m for m in lins if m.stored_tag() not in (None, PassTag.QUANT_DEQUANT)]
                assert held and all(m.weight.stride() == (0, 0) for m in held), "a stored linear kept a bf16 copy"
            else:
                assert PassTag.QUANT_DEQUANT in tags and quant_tag(main) not in tags
    # the dequantized path rounds every weight to bf16 and the as-stored one decodes it exactly, so a step whose two
    # leaders sit closer than the paths differ may go either way (IQ4_XS's first 11 and 8, 0.012 apart); every
    # step over the same tokens picks the same leader where it is decisive
    for j in range(len(toks[0])):
        a, b = forced[1][j], forced[0][j]
        gap = float(b.topk(2).values.diff().abs())
        if gap > 2 * max_abs(a, b):
            assert a.argmax().item() == b.argmax().item(), (release, j, gap)
    if release not in UNSTABLE:
        assert_close(logits[1], logits[0])


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
        assert any(getattr(m.mx, "q", None) is Quant.Q4_K for m in sm.host[0].modules() if hasattr(m, "mx"))
        ids = sm.prompt_ids("Write a detailed paragraph about the ocean and its depths.")
        greedy = sm.generate(ids, 48, speculate=False)
        spec = sm.generate(ids, 48, speculate=True)
    assert list(spec.tokens) == list(greedy.tokens), (spec.tokens, greedy.tokens)


def test_the_packed_mlx_path_multiplies_as_stored() -> None:
    """on MLX a Q4_0 / Q4_1 / Q8_0 fixture binds its matrices as their own blocks and `matvec_legacy` multiplies
    them as stored: the dequant is the package's numbers in bf16, the fp32 matvec sits within f32 rounding of the
    package's dequantized product, and a row is bit-identical alone and inside a 16-row tile (so a verify pass
    computes each row as the one-row step does). Q4_K's own native matvec is held to the same. The report counts
    the packed tensors, and the first-step logits sit within bf16 rounding of the dequantized path's"""
    _need()
    need_mlx()
    import mlx.core as mx
    from gguf import dequantize

    from btb.gguf import GGUFModel
    from btb.mlx.legacyq import KINDS, dequant_legacy, matvec_legacy

    rng = np.random.default_rng(1)
    for kind in KINDS:
        g = GGUFModel(os.path.join(GGUF, f"tiny_qwen3-{kind}.gguf"))
        mats = [(n, t) for n, t in g.tensors.items() if t.tensor_type.name.lower() == kind and len(t.shape) == 2]
        assert mats, f"tiny_qwen3-{kind}.gguf holds no {kind} matrix"
        for name, t in mats:
            rows, cols = (int(x) for x in reversed(list(t.shape)))
            ref = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(rows, cols)
            wb = mx.array(np.asarray(t.data).reshape(-1).view(np.uint8))
            deq = np.array(dequant_legacy(kind, wb, rows, cols).astype(mx.float32))
            assert np.array_equal(deq, np.array(mx.array(ref).astype(mx.bfloat16).astype(mx.float32))), name
            x = mx.array(rng.standard_normal((16, cols)).astype(np.float32))
            y = np.array(matvec_legacy(kind, wb, x, rows, cols))
            want, mag = np.array(x) @ ref.T.astype(np.float64), np.abs(np.array(x)) @ np.abs(ref.T).astype(np.float64)
            # against each output's conditioning; a zero weight row (the pad token's embedding) must give exact 0
            assert float(np.max(np.abs(y - want) / np.maximum(mag, 1e-30))) < 1e-6, name
            for i in (0, 7, 15):
                one = np.array(matvec_legacy(kind, wb, x[i : i + 1], rows, cols))
                assert np.array_equal(one[0], y[i]), f"{name}: row {i} alone differs from the 16-row tile"

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
