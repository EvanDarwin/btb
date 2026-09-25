# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""btb.mxfp4_torch against transformers' own MXFP4 dequantizer and GptOssExperts.

CPU only; every check here is device-agnostic torch.  Run with -s to see the
timings for one gpt-oss-120b-sized expert.
"""

import time
from collections.abc import Callable

import pytest
import torch
from pytest import CaptureFixture
from transformers.integrations.mxfp4 import FP4_VALUES, convert_moe_packed_tensors
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from btb import mxfp4_torch as M
from btb.mxfp4 import MxWeight
from tests.helpers import bits, mxfp4_random, rel_err


def pack_slot(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One slot's bytes from a checkpoint's `blocks [rows, G, 16]` and `scales [rows, G]`: how these tests build
    their inputs; the engine reads stored MXFP4 and never packs it."""
    assert blocks.shape[:-1] == scales.shape, f"blocks {tuple(blocks.shape)} do not match scales {tuple(scales.shape)}"
    return torch.cat((blocks.reshape(-1).to(torch.uint8), scales.reshape(-1).to(torch.uint8)))


# gpt-oss-120b's own numbers, for the timing case
BIG_ROWS, BIG_K = 5760, 2880


def _rand(rows: int, K: int, seed: int, scale_lo: int = 0, scale_hi: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """`mxfp4_random` as torch tensors"""
    blocks, scales = mxfp4_random(seed, rows, K, scale_lo, scale_hi)
    return torch.from_numpy(blocks), torch.from_numpy(scales)


def test_fp4_table_matches_transformers() -> None:
    assert list(M.FP4_VALUES) == list(FP4_VALUES)


def test_slot_bytes() -> None:
    assert M.slot_bytes(BIG_ROWS, BIG_K) == BIG_ROWS * BIG_K // 2 + BIG_ROWS * (BIG_K // 32)
    assert M.slot_bytes(BIG_ROWS, BIG_K) == 8_812_800  # gate_up expert
    assert M.slot_bytes(BIG_K, BIG_K) == 4_406_400  # down expert
    with pytest.raises(ValueError):
        M.slot_bytes(4, 30)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dequant_slot_bitwise_against_transformers(dtype: torch.dtype) -> None:
    """Random nibbles over the whole e8m0 range, including 0 and 255."""
    rows, K = 96, 256
    blocks, scales = _rand(rows, K, seed=11)
    # pin the extremes explicitly so the case is always covered
    scales[0, :] = 0  # exponent -127
    scales[1, :] = 255  # exponent +128 (OCP calls this NaN; transformers ldexps it)
    scales[2, :] = 254
    scales[3, :] = 127  # exponent 0
    blocks[4, :, :] = 0x77  # +6.0 in both nibbles, overflows at scale 255
    scales[4, :] = 255

    ref = convert_moe_packed_tensors(blocks.unsqueeze(0), scales.unsqueeze(0), dtype=dtype)
    assert tuple(ref.shape) == (1, K, rows)  # transformers returns [E, K, rows]
    ref = ref[0].t()  # -> [rows, K], our orientation

    slot = pack_slot(blocks, scales)
    got = M.dequant_slot(slot, rows, K, dtype=dtype)
    assert got.shape == (rows, K) and got.dtype is dtype
    assert torch.equal(bits(got), bits(ref))
    assert torch.equal(got.isnan(), ref.isnan())
    assert torch.equal(got.isinf(), ref.isinf())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dequant_slot_chunking_is_identical(dtype: torch.dtype) -> None:
    rows, K = 71, 128
    blocks, scales = _rand(rows, K, seed=12)
    slot = pack_slot(blocks, scales)
    whole = M.dequant_slot(slot, rows, K, dtype=dtype)
    for chunk in (1, 7, 32, 1000):
        part = M.dequant_slot(slot, rows, K, dtype=dtype, rows_per_chunk=chunk)
        assert torch.equal(bits(whole), bits(part)), chunk


def test_dequant_slot_into_out_buffer() -> None:
    rows, K = 40, 96
    blocks, scales = _rand(rows, K, seed=13)
    slot = pack_slot(blocks, scales)
    buf = torch.zeros(rows, K, dtype=torch.bfloat16)
    got = M.dequant_slot(slot, rows, K, out=buf)
    assert got.data_ptr() == buf.data_ptr()
    assert torch.equal(bits(buf), bits(M.dequant_slot(slot, rows, K)))


def test_slot_is_a_view_into_a_slab_and_is_not_copied() -> None:
    """The store hands out a slice of a shared slab; nothing here may copy it."""
    rows, K = 48, 128
    blocks, scales = _rand(rows, K, seed=14)
    n = M.slot_bytes(rows, K)
    slab = torch.empty(n * 3 + 7, dtype=torch.uint8)
    slab.fill_(0xAB)
    off = n + 7  # deliberately unaligned start
    view = slab[off : off + n]
    view.copy_(pack_slot(blocks, scales))
    assert view.data_ptr() == slab.data_ptr() + off
    assert view.is_contiguous() and view.storage_offset() == off

    b, s = M.split_slot(view, rows, K)
    assert b.data_ptr() == view.data_ptr()
    assert s.data_ptr() == view.data_ptr() + rows * (K // 32) * 16
    assert b.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()
    assert s.untyped_storage().data_ptr() == slab.untyped_storage().data_ptr()

    ref = convert_moe_packed_tensors(blocks.unsqueeze(0), scales.unsqueeze(0))[0].t()
    assert torch.equal(bits(M.dequant_slot(view, rows, K)), bits(ref))

    # a 2-D non-contiguous base (a row of a strided slab) still gives the right answer
    slab2 = torch.zeros(3, n + 5, dtype=torch.uint8)
    slab2[1, :n] = pack_slot(blocks, scales)
    row = slab2[1, :n]
    assert row.data_ptr() == slab2.data_ptr() + (n + 5)
    assert torch.equal(bits(M.dequant_slot(row, rows, K)), bits(ref))

    # neighbours untouched
    assert int(slab[:off].min()) == 0xAB and int(slab[off + n :].min()) == 0xAB


def test_split_slot_rejects_bad_input() -> None:
    rows, K = 8, 64
    slot = pack_slot(*_rand(rows, K, seed=15))
    with pytest.raises(ValueError):
        M.split_slot(slot, rows + 1, K)
    with pytest.raises(TypeError):
        M.split_slot(slot.to(torch.int8), rows, K)
    with pytest.raises(ValueError):  # ggml's layout carries no scales to split
        M.split_slot(MxWeight.from_ggml(torch.zeros(rows * (K // 32) * 17, dtype=torch.uint8), rows, K))


def test_matvec_slot_matches_dequant_then_matmul() -> None:
    rows, K = 128, 256
    blocks, scales = _rand(rows, K, seed=16, scale_lo=120, scale_hi=135)
    slot = pack_slot(blocks, scales)
    w = M.dequant_slot(slot, rows, K).float()
    x = torch.randn(5, K)
    ref = x @ w.t()
    # the weights are bit-identical; the sum over K is torch's, so the chunk
    # size (which changes the matmul shape and therefore its kernel and its
    # accumulation order) moves the last bits - a tolerance, not the bit
    for chunk in (None, 1, 33, 512):
        got = M.matvec_slot(slot, rows, K, x, rows_per_chunk=chunk)
        assert got.shape == (5, rows) and got.dtype is torch.float32
        assert rel_err(got, ref) < 1e-6, (chunk, rel_err(got, ref))
    one = M.matvec_slot(slot, rows, K, x[0])  # 1-D input is promoted to [1, K]
    assert one.shape == (1, rows)
    assert rel_err(one, ref[:1]) < 1e-6


# ---------------------------------------------------------------- expert_step


def _tiny(
    E: int = 6, K: int = 128, I: int = 64, seed: int = 20
) -> tuple[GptOssConfig, GptOssExperts, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Random MXFP4 experts plus the reference GptOssExperts built from them."""
    g = torch.Generator().manual_seed(seed)
    gu_blocks = torch.randint(0, 256, (E, 2 * I, K // 32, 16), dtype=torch.uint8, generator=g)
    gu_scales = torch.randint(122, 132, (E, 2 * I, K // 32), dtype=torch.uint8, generator=g)
    dn_blocks = torch.randint(0, 256, (E, K, I // 32, 16), dtype=torch.uint8, generator=g)
    dn_scales = torch.randint(122, 132, (E, K, I // 32), dtype=torch.uint8, generator=g)
    gu_bias = torch.randn(E, 2 * I, generator=g).to(torch.bfloat16)
    dn_bias = torch.randn(E, K, generator=g).to(torch.bfloat16)

    cfg = GptOssConfig(
        hidden_size=K,
        intermediate_size=I,
        num_local_experts=E,
        num_experts_per_tok=3,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        vocab_size=64,
    )
    ref = GptOssExperts(cfg)
    with torch.no_grad():
        # transformers dequantizes on load; use its own function, then run fp32
        ref.gate_up_proj.copy_(convert_moe_packed_tensors(gu_blocks, gu_scales).float())
        ref.down_proj.copy_(convert_moe_packed_tensors(dn_blocks, dn_scales).float())
        ref.gate_up_proj_bias.copy_(gu_bias.float())
        ref.down_proj_bias.copy_(dn_bias.float())
    slots_gu = [pack_slot(gu_blocks[e], gu_scales[e]) for e in range(E)]
    slots_dn = [pack_slot(dn_blocks[e], dn_scales[e]) for e in range(E)]
    return cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias


def test_swiglu_matches_reference_apply_gate() -> None:
    _, ref, _, _, _, _ = _tiny()
    x = torch.randn(7, 2 * ref.intermediate_size) * 6
    assert torch.equal(M.swiglu(x), ref._apply_gate(x))


def test_expert_step_matches_gpt_oss_experts() -> None:
    E, K, I, k, T = 6, 128, 64, 3, 9
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I)
    g = torch.Generator().manual_seed(21)
    x = torch.randn(T, K, generator=g)
    logits = torch.randn(T, E, generator=g)
    top_v, idx = torch.topk(logits, k, dim=-1)
    w = torch.softmax(top_v, dim=1)

    want = ref(x, router_indices=idx, routing_weights=w)
    got = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w, expert_of=idx)
    assert got.shape == (T, K) and got.dtype is torch.float32
    assert rel_err(got, want) < 1e-5, rel_err(got, want)


def test_expert_step_decode_default_expert_of() -> None:
    """T == 1 with E == k: every slot is used once, no expert_of needed."""
    E, K, I = 4, 128, 64
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=22)
    g = torch.Generator().manual_seed(23)
    x = torch.randn(1, K, generator=g)
    w = torch.softmax(torch.randn(1, E, generator=g), dim=1)
    idx = torch.arange(E).unsqueeze(0)
    want = ref(x, router_indices=idx, routing_weights=w)
    got = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w)
    assert rel_err(got, want) < 1e-5, rel_err(got, want)


def test_expert_step_sums_in_expert_order() -> None:
    """The outer sum is fixed: ascending slot-list order, whatever the routing."""
    E, K, I, k, T = 5, 128, 64, 3, 4
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=24)
    g = torch.Generator().manual_seed(25)
    x = torch.randn(T, K, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), dim=1)
    a = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w, expert_of=idx)
    # permuting each token's k choices (and its weights with them) must not move a bit
    perm = torch.tensor([k - 1 - j for j in range(k)])
    b = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w[:, perm], expert_of=idx[:, perm])
    assert torch.equal(a, b)
    assert rel_err(a, ref(x, router_indices=idx, routing_weights=w)) < 1e-5


def test_expert_step_over_slab_views() -> None:
    """The same result when the slots are windows into one shared slab."""
    E, K, I, k, T = 4, 128, 64, 2, 6
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=26)
    per = slots_gu[0].numel() + slots_dn[0].numel()
    slab = torch.zeros(E * per, dtype=torch.uint8)
    view_gu, view_dn = [], []
    for e in range(E):
        o = e * per
        n = slots_gu[e].numel()
        slab[o : o + n] = slots_gu[e]
        slab[o + n : o + per] = slots_dn[e]
        view_gu.append(slab[o : o + n])
        view_dn.append(slab[o + n : o + per])
        assert view_gu[-1].data_ptr() == slab.data_ptr() + o
    g = torch.Generator().manual_seed(27)
    x = torch.randn(T, K, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), dim=1)
    a = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w, expert_of=idx)
    b = M.expert_step(x, view_gu, view_dn, gu_bias, dn_bias, w, expert_of=idx)
    assert torch.equal(a, b)
    assert rel_err(b, ref(x, router_indices=idx, routing_weights=w)) < 1e-5


def test_expert_step_chunked_weights() -> None:
    E, K, I, k, T = 4, 128, 64, 2, 5
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=28)
    g = torch.Generator().manual_seed(29)
    x = torch.randn(T, K, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), dim=1)
    want = ref(x, router_indices=idx, routing_weights=w)
    for chunk in (17, 64, None):
        got = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w, expert_of=idx, rows_per_chunk=chunk)
        assert rel_err(got, want) < 1e-5, (chunk, rel_err(got, want))


def test_expert_step_rejects_mismatched_shapes() -> None:
    E, K, I = 3, 128, 64
    _, _, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=30)
    w = torch.softmax(torch.randn(2, 2), dim=1)
    with pytest.raises(ValueError):  # E != k and no expert_of
        M.expert_step(torch.randn(2, K), slots_gu, slots_dn, gu_bias, dn_bias, w)
    with pytest.raises(ValueError):  # unequal slot lists
        M.expert_step(
            torch.randn(2, K),
            slots_gu,
            slots_dn[:2],
            gu_bias,
            dn_bias,
            w,
            expert_of=torch.zeros(2, 2, dtype=torch.long),
        )


def test_accepts_an_mxweight_without_rows_and_k() -> None:
    rows, K = 64, 128
    blocks, scales = _rand(rows, K, seed=32, scale_lo=122, scale_hi=132)
    slot = pack_slot(blocks, scales)
    mx = MxWeight(blocks, scales, rows, K)
    assert torch.equal(bits(M.dequant_slot(mx)), bits(M.dequant_slot(slot, rows, K)))
    x = torch.randn(3, K)
    assert torch.equal(M.matvec_slot(mx, x), M.matvec_slot(slot, rows, K, x))
    b, s = M.split_slot(mx)
    assert b.data_ptr() == blocks.data_ptr() and s.data_ptr() == scales.data_ptr()


def test_expert_step_over_mxweights() -> None:
    E, K, I, k, T = 4, 128, 64, 2, 5
    _cfg, ref, slots_gu, slots_dn, gu_bias, dn_bias = _tiny(E=E, K=K, I=I, seed=33)
    nb_gu, ns_gu = 2 * I * (K // 32) * 16, 2 * I * (K // 32)
    nb_dn, ns_dn = K * (I // 32) * 16, K * (I // 32)
    mx_gu = [MxWeight(s[:nb_gu], s[nb_gu : nb_gu + ns_gu], 2 * I, K) for s in slots_gu]
    mx_dn = [MxWeight(s[:nb_dn], s[nb_dn : nb_dn + ns_dn], K, I) for s in slots_dn]
    g = torch.Generator().manual_seed(34)
    x = torch.randn(T, K, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), dim=1)
    a = M.expert_step(x, slots_gu, slots_dn, gu_bias, dn_bias, w, expert_of=idx)
    b = M.expert_step(x, mx_gu, mx_dn, gu_bias, dn_bias, w, expert_of=idx)
    assert torch.equal(a, b)
    assert rel_err(b, ref(x, router_indices=idx, routing_weights=w)) < 1e-5


# ------------------------------------------------------------------- timing


def test_timing_one_gpt_oss_expert(capsys: CaptureFixture[str]) -> None:
    """One 5760x2880 gate_up expert: dequantize, then a decode-shaped matvec."""
    blocks, scales = _rand(BIG_ROWS, BIG_K, seed=31, scale_lo=122, scale_hi=132)
    slot = pack_slot(blocks, scales)
    assert slot.numel() == 8_812_800
    x = torch.randn(1, BIG_K)

    def timeit(fn: Callable[[], object], n: int = 3) -> float:
        fn()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n

    d_whole = timeit(lambda: M.dequant_slot(slot, BIG_ROWS, BIG_K))
    d_chunk = timeit(lambda: M.dequant_slot(slot, BIG_ROWS, BIG_K, rows_per_chunk=1024))
    m_whole = timeit(lambda: M.matvec_slot(slot, BIG_ROWS, BIG_K, x))
    m_chunk = timeit(lambda: M.matvec_slot(slot, BIG_ROWS, BIG_K, x, rows_per_chunk=1024))
    x8 = torch.randn(8, BIG_K)
    m8 = timeit(lambda: M.matvec_slot(slot, BIG_ROWS, BIG_K, x8, rows_per_chunk=1024))

    with capsys.disabled():
        print(
            f"\n  gate_up expert 5760x2880: slot {slot.numel() / 2**20:.2f} MiB, bf16 {BIG_ROWS * BIG_K * 2 / 2**20:.2f} MiB"
        )
        print(f"  dequant_slot        whole {d_whole * 1e3:7.1f} ms   chunk=1024 {d_chunk * 1e3:7.1f} ms")
        print(f"  matvec_slot  B=1    whole {m_whole * 1e3:7.1f} ms   chunk=1024 {m_chunk * 1e3:7.1f} ms")
        print(f"  matvec_slot  B=8             chunk=1024 {m8 * 1e3:7.1f} ms")
    assert d_whole > 0 and m_chunk > 0
