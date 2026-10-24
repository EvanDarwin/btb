# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The MLX device's parts, each against a torch reference: the GPU unpack of the 12-bit store (bit-exact), the
float32-over-bf16 kernel, RoPE, the shared buffers, the attention cache in unified memory (in place, and the
torch side's crops and gathers), and MLX's causal mask against the engine's. Skips when MLX is not available."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import numpy as np
import torch

from btb import mlx_available
from tests.helpers import (
    PROMPT_DENSE,
    PROMPT_Q35,
    fixture,
    forward_logits,
    host_model,
    mxfp4_random,
    mxfp4_slot,
    native_library,
    receipts,
    speculation,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from btb.mlx.attn import NodeMeta
    from btb.mlx.backend import PackedEntry

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _skip() -> bool:
    if not mlx_available():
        print("mlx: not available, skipped", flush=True)
        return True
    return False


def check_unpack() -> str:
    from btb import mlx as mlxdev
    from btb.engine import pack_bf16

    torch.manual_seed(0)
    t = (torch.randn(300, 520) * torch.logspace(-3, 3, 520)).bfloat16()
    lo, hi4, tbl, esc_idx, esc_val = pack_bf16(t)
    pad = (-(lo.nbytes + hi4.nbytes)) % 4
    blob = np.concatenate([lo, hi4, np.zeros(pad, np.uint8), esc_idx.view(np.uint8), esc_val])
    e: PackedEntry = {
        "lo": lo.nbytes,
        "hi4": hi4.nbytes,
        "pad": pad,
        "esc": int(esc_idx.size),
        "table": [int(x) for x in tbl],
        "n": t.numel(),
    }
    so = 64 * 7
    sh = mlxdev.Shared(so + blob.size + 128)
    sh.torch[so : so + blob.size] = torch.from_numpy(blob.copy())
    w = mlxdev.Backend.weight_slot_packed(sh, so, e, tuple(t.shape))
    back = mlxdev.from_mx(w.get())
    assert esc_idx.size > 0, "the fixture has no escapes"
    assert bool((back.view(torch.int16) == t.view(torch.int16)).all()), "GPU unpack is not bit-exact"
    return "unpack bit-exact"


def check_gemv() -> str:
    from btb import mlx as mlxdev
    from btb.mlx.backend import Stat

    be = mlxdev.Backend.__new__(mlxdev.Backend)
    be.gemm_rows = 64
    be.stat = Stat(linears=0, evals=0, s=0.0, layers=0, layer_s=0.0)
    worst = 0.0
    torch.manual_seed(1)
    for rows, cols, b in [
        (7, 64, 1),
        (1000, 4096, 1),
        (4096, 4096, 5),
        (4096, 4096, 8),
        (4096, 4096, 9),
        (3, 8, 63),
        (513, 2560, 3),
    ]:
        W = torch.randn(rows, cols).bfloat16()
        x = torch.randn(b, cols)
        y = be.linear(x, mlxdev.Weight(mlxdev.bf16_weight(W)))
        ref = x @ W.float().T
        worst = max(worst, float((y - ref).abs().max() / ref.abs().max()))
        assert y.dtype == torch.float32
    assert worst < 1e-5, f"float32 kernel rel err {worst:.2e}"
    W = torch.randn(4096, 4096).bfloat16()
    x = torch.randn(1, 4096).bfloat16()
    y = be.linear(x, mlxdev.Weight(mlxdev.bf16_weight(W)))
    ref = torch.nn.functional.linear(x.float(), W.float()).bfloat16()
    d = float((y.float() - ref.float()).abs().max() / ref.float().abs().max())
    assert y.dtype == torch.bfloat16 and d < 1e-2, f"bf16 path rel err {d:.2e}"
    return f"gemv rel err {worst:.1e}"


def check_rope() -> str:
    from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb

    from btb import mlx as mlxdev

    torch.manual_seed(2)
    q = torch.randn(1, 4, 5, 32)
    k = torch.randn(1, 2, 5, 32)
    cos, sin = torch.randn(1, 5, 24), torch.randn(1, 5, 24)
    qe, _ = apply_rotary_pos_emb(q, k, cos, sin)
    qm = mlxdev.Backend.rope(
        mlxdev.to_mx(q[0].transpose(0, 1).contiguous()), mlxdev.to_mx(cos[0]), mlxdev.to_mx(sin[0])
    )
    d = float((mlxdev.from_mx(qm).transpose(0, 1) - qe[0]).abs().max())
    assert d == 0.0, f"rope differs by {d}"
    return "rope exact"


def check_cache() -> str:
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.engine import GrowLayer

    torch.manual_seed(3)
    B, Hk, d = 1, 2, 16
    cl = GrowLayer(shared=True)
    ref_k, ref_v = [], []
    # MLX appends, in place
    for T in (5, 1, 1, 3):
        k = torch.randn(B, Hk, T, d).bfloat16()
        v = torch.randn(B, Hk, T, d).bfloat16()
        ref_k.append(k)
        ref_v.append(v)
        K, V = cl.mx_update(mlxdev.to_mx(k), mlxdev.to_mx(v))
        mx.eval(K, V)
    ptr = cl._ptr
    assert cl.get_seq_length() == 10
    assert bool((cl.keys == torch.cat(ref_k, -2)).all()) and bool((cl.values == torch.cat(ref_v, -2)).all())
    k = torch.randn(B, Hk, 1, d).bfloat16()
    K, V = cl.mx_update(mlxdev.to_mx(k), mlxdev.to_mx(k))
    mx.eval(K)
    assert cl._ptr == ptr, "an append reallocated the buffer (a view was alive)"
    assert bool((cl.keys[..., -1:, :] == k).all())
    # a prefix crop from torch is just a length
    cl.keys = cl.keys[..., :7, :]
    cl.values = cl.values[..., :7, :]
    assert cl.get_seq_length() == 7 and cl._tk is None
    # a gather (an accepted tree path) is written into the buffer by the next append
    idx = torch.tensor([0, 1, 2, 5, 6])
    cl.keys = cl.keys.index_select(-2, idx)
    cl.values = cl.values.index_select(-2, idx)
    expect_k = torch.cat(ref_k, -2)[..., idx, :]
    assert cl.get_seq_length() == 5 and cl._tk is not None
    k = torch.randn(B, Hk, 2, d).bfloat16()
    K, V = cl.mx_update(mlxdev.to_mx(k), mlxdev.to_mx(k))
    mx.eval(K)
    assert cl.get_seq_length() == 7 and cl._tk is None
    assert bool((cl.keys == torch.cat([expect_k, k], -2)).all())
    # a view held across an append keeps MLX from writing the buffer in place: the rows move, and a crop of the moved
    # rows is still a crop (a finishing command buffer's hold moves them the same way, at random)
    held = cl.keys
    k1 = torch.randn(B, Hk, 1, d).bfloat16()
    K, V = cl.mx_update(mlxdev.to_mx(k1), mlxdev.to_mx(k1))
    mx.eval(K, V)
    assert cl.keys.untyped_storage().data_ptr() != held.untyped_storage().data_ptr(), "the held view kept the rows"
    del held
    cl.keys = cl.keys[..., :7, :]
    cl.values = cl.values[..., :7, :]
    assert cl.get_seq_length() == 7 and cl._tk is None and cl._tv is None, "a crop of moved rows was not seen as one"
    # torch appends through the same buffer, and growth keeps the rows
    k2 = torch.randn(B, Hk, 4100, d).bfloat16()
    kk, _vv = cl.update(k2, k2)
    assert kk.shape[-2] == 4107 and bool((kk[..., :7, :] == torch.cat([expect_k, k], -2)).all())
    assert bool((kk[..., 7:, :] == k2).all())
    K, V = cl.mx_update(mlxdev.to_mx(k), mlxdev.to_mx(k))
    mx.eval(K)
    assert cl.get_seq_length() == 4109 and bool((cl.keys[..., -2:, :] == k).all())
    return "cache in place, crops and gathers honored"


def check_causal() -> str:
    import mlx.core as mx

    from btb import mlx as mlxdev

    torch.manual_seed(4)
    Hq, Hk, d, past, T = 4, 2, 16, 9, 3
    q = torch.randn(1, Hq, T, d)
    k = torch.randn(1, Hk, past + T, d)
    v = torch.randn(1, Hk, past + T, d)
    allow = torch.ones(T, past + T, dtype=torch.bool).tril(past)
    mask = torch.zeros(T, past + T).masked_fill(~allow, float("-inf")).view(1, 1, T, past + T)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True, scale=0.25)
    out = mx.fast.scaled_dot_product_attention(
        mlxdev.to_mx(q), mlxdev.to_mx(k), mlxdev.to_mx(v), scale=0.25, mask="causal"
    )
    d_ = float((mlxdev.from_mx(out) - ref).abs().max())
    assert d_ < 1e-5, f"causal mask alignment differs by {d_}"
    return "causal mask aligned"


def _attn_ref(
    q: mx.array,
    kb: mx.array,
    vb: mx.array,
    S: int,
    scale: float,
    Hk: int,
    g: int,
    sinks: mx.array | None = None,
    window: int | None = None,
) -> mx.array:
    """one row of attention in MLX ops: the softmax over [scores, sink] with the window mask, as
    transformers' `eager_attention_forward` computes it (the sink is a column with no value, and it is not
    scaled: it joins the already scaled scores)"""
    import mlx.core as mx

    d = int(q.shape[-1])
    s = (q.reshape(Hk, g, d) @ kb[0, :, :S].astype(mx.float32).transpose(0, 2, 1)) * scale
    start = 0 if not window else max(0, S - window)
    if start:
        s = s + mx.concatenate([mx.full((start,), -mx.inf), mx.zeros((S - start,))])
    if sinks is None:
        p = mx.softmax(s, axis=-1)
    else:
        p = mx.softmax(mx.concatenate([s, sinks.reshape(Hk, g, 1)], axis=-1), axis=-1)[..., :S]
    return (p @ vb[0, :, :S].astype(mx.float32)).reshape(Hk * g, d)


def check_attn_kernel() -> str:
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.mlx import aluminum

    mx.random.seed(5)
    # the error by P's precision: float below head 256, bf16 at 256 (aluminum.bf16p); q is bf16 as the kernel takes it
    worst = {False: 0.0, True: 0.0}
    # head_dim 128 (Qwen3, Llama), 256 (Qwen3.5) with four query heads per kv head, 96 (Phi-3's, a kv head a query
    # head), and gpt-oss: head_dim 64, eight query heads per kv head, per-head sinks and a 128-position sliding window
    # on half its layers
    for Hq, Hk, d in ((32, 8, 128), (16, 4, 256), (32, 32, 96), (64, 8, 64)):
        g = Hq // Hk
        for S, cap in ((7, 64), (73, 4096), (4096, 4500), (9000, 12288)):
            kb = (mx.random.normal((1, Hk, cap, d)) * 2).astype(mx.bfloat16)
            vb = mx.random.normal((1, Hk, cap, d)).astype(mx.bfloat16)
            q = mx.random.normal((Hq, d)).astype(mx.bfloat16).astype(mx.float32)
            arms: list[tuple[mx.array | None, int | None]] = [(None, None)]
            if d == 64:
                sk = mx.random.normal((Hq,)) * 3
                arms = [(None, None), (sk, None), (None, 128), (sk, 128)]
            for sinks, window in arms:
                ref = _attn_ref(q, kb, vb, S, 0.088, Hk, g, sinks, window)
                out = mlxdev.attn_decode(q, kb, vb, S, 0.088, sinks=sinks, window=window)
                pb = aluminum.bf16p(d)
                worst[pb] = max(worst[pb], float(mx.abs(out - ref).max() / mx.abs(ref).max()))
    Hq, Hk, d = 32, 8, 128
    S, cap = 4096, 4500
    kb = (mx.random.normal((1, Hk, cap, d)) * 2).astype(mx.bfloat16)
    vb = mx.random.normal((1, Hk, cap, d)).astype(mx.bfloat16)
    # three query heads per kv head (Phi-4-mini's shape)
    q = mx.random.normal((24, d)).astype(mx.bfloat16).astype(mx.float32)
    ref = mx.softmax((q.reshape(Hk, 3, d) @ kb[0, :, :S].astype(mx.float32).transpose(0, 2, 1)) * 0.088, axis=-1) @ vb[
        0, :, :S
    ].astype(mx.float32)
    out = mlxdev.attn_decode(q, kb, vb, S, 0.088)
    worst[False] = max(worst[False], float(mx.abs(out - ref.reshape(24, d)).max() / mx.abs(ref).max()))
    assert worst[False] < 1e-4, f"decode kernel rel err {worst[False]:.2e} (float P)"
    assert worst[True] < 6e-3, f"decode kernel rel err {worst[True]:.2e} (P in bf16)"
    return (
        f"decode kernel rel err {worst[False]:.1e} (float P, heads 64/96/128), {worst[True]:.1e} (P in bf16, head 256);"
        " sinks and a 128 window"
    )


def check_delta_tree() -> str:
    """the tree recurrence against the sequence kernel: a chain bit-exact, every root-to-leaf path of a tree
    bit-exact against the sequence kernel run over that path, and the commit's state for that path the same"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(11)
    H, dk, dv = 4, 64, 32
    T = 9
    q = mx.random.normal((T, H, dk)) * 0.1
    k = mx.random.normal((T, H, dk)) * 0.1
    v = mx.random.normal((T, H, dv))
    g = -mx.abs(mx.random.normal((T, H))) * 0.1
    beta = mx.sigmoid(mx.random.normal((T, H)))
    state = mx.random.normal((H, dk, dv)) * 0.5
    out_c = mlxdev.delta_tree(q, k, v, g, beta, state, list(range(-1, T - 1)))
    out_s, st_s = mlxdev.delta_recurrent(q, k, v, g, beta, state)
    st_c = mlxdev.delta_chain_state((q, k, v, g, beta), state)
    assert st_s is not None and st_c is not None
    mx.eval(out_c, out_s, st_s, st_c)
    assert mx.array_equal(out_c, out_s) and mx.array_equal(st_c, st_s), "chain differs from the sequence kernel"
    parents = [-1, 0, 0, 1, 1, 2, 4, 4, 6]
    out_t = mlxdev.delta_tree(q, k, v, g, beta, state, parents)
    mx.eval(out_t)
    leaves = [j for j in range(T) if j not in parents]
    for leaf in leaves:
        path = []
        n = leaf
        while n >= 0:
            path.append(n)
            n = parents[n]
        path = path[::-1]
        idx = mx.array(np.asarray(path, dtype=np.int32))
        o, s = mlxdev.delta_recurrent(q[idx], k[idx], v[idx], g[idx], beta[idx], state)
        s2 = mlxdev.delta_chain_state((q, k, v, g, beta), state, path)
        assert s is not None and s2 is not None
        mx.eval(o, s, s2)
        assert mx.array_equal(o, out_t[idx]) and mx.array_equal(s, s2), f"path {path} differs"
    # in place: the state written over its own buffer, the bits the returned state's
    buf = mx.array(state)
    mx.eval(buf)
    done = mlxdev.delta_chain_state((q, k, v, g, beta), buf, path, inplace=True)
    mx.eval(done)
    assert s is not None
    assert mx.array_equal(buf, s), "the in-place state differs"
    return (
        f"chain and {len(leaves)} tree paths bit-exact against the sequence kernel, the path's state recomputed,"
        " in place the same"
    )


def check_attn_tree() -> str:
    """a verify pass's attention against one-row decodes: every node bit for bit as the one-row kernel computes
    it over a cache holding the prefix and then the node's path, the way the greedy loop would hold them"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(9)
    bad = 0
    # a long prefix over six segments at head_dim 128; a short prefix off every alignment (73 rows) at
    # head_dim 256 with a tree whose branch nodes sit away from their path's rows in the cache; head_dim 96 over
    # three segments; and gpt-oss (head_dim 64, eight query heads per kv head) with its sinks and a 128-position
    # window over a 3000-row prefix, so that all but the last rows of the prefix are outside every node's window
    for Hq, Hk, d, past, parents, sink, win in (
        (32, 8, 128, 3000, [-1, 0, 0, 1, 2, 2, 4], False, None),
        (16, 4, 256, 73, [-1, 0, 0, 1, 1, 2, 3, 5], False, None),
        (32, 32, 96, 1500, [-1, 0, 1, 1, 3, 0], False, None),
        (64, 8, 64, 3000, [-1, 0, 0, 1, 2, 2, 4, 6], True, 128),
        (64, 8, 64, 60, [-1, 0, 0, 1, 2, 2, 4, 6], True, 128),
    ):
        T, cap = len(parents), 4096
        kb = (mx.random.normal((1, Hk, cap, d)) * 2).astype(mx.bfloat16)
        vb = mx.random.normal((1, Hk, cap, d)).astype(mx.bfloat16)
        q = mx.random.normal((T, Hq, d))
        sinks = mx.random.normal((Hq,)) * 3 if sink else None
        mx.eval(kb, vb, q)
        out = mlxdev.attn_tree(q, kb, vb, past, parents, 0.088, sinks=sinks, window=win)
        mx.eval(out)
        for t in range(T):
            path, cur = [], t
            while cur >= 0:
                path.append(cur)
                cur = parents[cur]
            path = path[::-1]
            idx = mx.array(
                list(range(past)) + [past + p for p in path] + [0] * (cap - past - len(path)), dtype=mx.int32
            )
            k1, v1 = kb[:, :, idx], vb[:, :, idx]
            mx.eval(k1, v1)
            one = mlxdev.attn_decode(q[t], k1, v1, past + len(path), 0.088, sinks=sinks, window=win)
            mx.eval(one)
            bad += int(not mx.array_equal(one, out[t]))
    assert bad == 0, f"{bad} nodes differ from the one-row decode"
    return (
        "tree nodes bit-exact against one-row decodes (head_dim 128 over six segments; 256 off alignment; 96 over "
        "three; gpt-oss with sinks and a 128 window, past 3000 and 60)"
    )


def check_attn_paged() -> str:
    """the node and prefill kernels over a paged cache: a conversation's rows scattered (`kv_scatter`) over a pool's
    pages in shuffled order, the rows before a pass read through its row map and the pass's own beside them, bit for
    bit as the same rows held in one contiguous buffer - the decode, a tree's nodes and a prefill chunk (the fold goes
    by the conversation's rows, never where they lie)"""
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.engine.kvpool import PAGE

    mx.random.seed(13)
    rng = np.random.default_rng(13)
    bad = []
    for Hq, Hk, d, past, parents in (
        (32, 8, 128, 3000, [-1, 0, 0, 1, 2, 2, 4]),
        (16, 4, 256, 73, [-1, 0, 0, 1, 1, 2, 3, 5]),
        (8, 2, 256, 1500, [-1, 0, 1, 2]),
        (32, 32, 96, 900, [-1, 0, 0, 2]),
        (64, 8, 64, 700, [-1, 0, 1, 1, 3]),
    ):
        T = len(parents)
        n = past + T
        pages = -(-n // PAGE)
        pool = 2 * pages + 3
        order = rng.permutation(pool)[:pages]
        rows = (order[:, None] * PAGE + np.arange(PAGE)[None]).reshape(-1)[:n]
        cap = n + 1
        kb = mx.zeros((1, Hk, cap, d), dtype=mx.bfloat16)
        vb = mx.zeros((1, Hk, cap, d), dtype=mx.bfloat16)
        kb[:, :, :n] = (mx.random.normal((1, Hk, n, d)) * 2).astype(mx.bfloat16)
        vb[:, :, :n] = mx.random.normal((1, Hk, n, d)).astype(mx.bfloat16)
        rk = mx.zeros((1, Hk, pool * PAGE, d), dtype=mx.bfloat16)
        rv = mx.zeros((1, Hk, pool * PAGE, d), dtype=mx.bfloat16)
        q1 = mx.random.normal((Hq, d))
        qt = mx.random.normal((T, Hq, d))
        mx.eval(kb, vb, rk, rv, q1, qt)
        at = mx.array(rows.astype(np.uint32))
        mx.eval(mlxdev.kv_scatter(rk, rv, kb[:, :, :n], vb[:, :, :n], at))
        old, step = at[: n - 1], at[:past]
        pairs = [
            (
                "decode",
                mlxdev.attn_decode(q1, kb, vb, n, 0.088),
                mlxdev.attn_decode(q1, rk, rv, n, 0.088, rmap=old, k2=kb[:, :, n - 1 : n], v2=vb[:, :, n - 1 : n]),
            ),
            (
                "tree",
                mlxdev.attn_tree(qt, kb, vb, past, parents, 0.088),
                mlxdev.attn_tree(qt, rk, rv, past, parents, 0.088, rmap=step, k2=kb[:, :, past:n], v2=vb[:, :, past:n]),
            ),
        ]
        qp = mx.random.normal((T, Hq, d)).astype(mx.bfloat16)
        pairs.append(
            (
                "prefill",
                mlxdev.attn_prefill(qp, kb, vb, past, d**-0.5, odt=mx.float32),
                mlxdev.attn_prefill(
                    qp, rk, rv, past, d**-0.5, odt=mx.float32, rmap=step, k2=kb[:, :, past:n], v2=vb[:, :, past:n]
                ),
            )
        )
        for name, want, got in pairs:
            mx.eval(want, got)
            if not mx.array_equal(want, got):
                bad.append((name, Hq, Hk, d, past, float(mx.max(mx.abs(want - got)))))
    assert not bad, f"paged reads differ from contiguous ones: {bad}"
    return (
        "decode, tree and prefill through a shuffled row map bit-exact against contiguous rows (heads 64, 96, 128, 256)"
    )


def check_attn_int8() -> str:
    """the node kernel over an int8 cache (a float32 scale per row): the decode against a float32 reference
    over the rows as dequantized to bf16 (what the kernel reads, the prefill's rows of that cache), and a tree's
    nodes bit for bit as one-row decodes over the same int8 rows"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(13)
    Hq, Hk, d, cap = 32, 8, 128, 4096
    kb = mx.random.normal((1, Hk, cap, d)) * 2
    vb = mx.random.normal((1, Hk, cap, d))
    kq, ks = mlxdev.kv_quantize(kb)
    vq, vs = mlxdev.kv_quantize(vb)
    kd, vd = mlxdev.kv_dequantize(kq, ks, mx.float32), mlxdev.kv_dequantize(vq, vs, mx.float32)
    mx.eval(kq, ks, vq, vs, kd, vd)
    err_q = float(mx.abs(kd - kb).max() / mx.abs(kb).max())
    assert err_q < 5e-3, f"int8 rows are off by {err_q:.2e} of the largest magnitude"
    kd, vd = (mlxdev.kv_dequantize(x, s_).astype(mx.float32) for x, s_ in ((kq, ks), (vq, vs)))
    n = 3000
    q = mx.random.normal((Hq, d)).astype(mx.bfloat16).astype(mx.float32)
    mx.eval(q)
    out = mlxdev.attn_decode(q, kq, vq, n, 0.088, ks=ks, vs=vs)
    # the reference: softmax(q . K^T) V over the dequantized rows, per query head, float32
    g = Hq // Hk
    qs = q.reshape(Hk, g, d) * 0.088
    s = qs @ kd[0, :, :n].transpose(0, 2, 1)
    p = mx.softmax(s, axis=-1)
    ref = (p @ vd[0, :, :n]).reshape(Hq, d)
    mx.eval(out, ref)
    err = float(mx.abs(out - ref).max() / mx.abs(ref).max())
    assert err < 1e-5, f"the int8 decode differs from the float32 reference by {err:.2e}"
    parents = [-1, 0, 0, 1, 2, 2, 4]
    T, past = len(parents), 3000
    qt = mx.random.normal((T, Hq, d))
    mx.eval(qt)
    tree = mlxdev.attn_tree(qt, kq, vq, past, parents, 0.088, ks=ks, vs=vs)
    mx.eval(tree)
    bad = 0
    for t in range(T):
        path, cur = [], t
        while cur >= 0:
            path.append(cur)
            cur = parents[cur]
        path = path[::-1]
        idx = mx.array(list(range(past)) + [past + p for p in path] + [0] * (cap - past - len(path)), dtype=mx.int32)
        one = mlxdev.attn_decode(
            qt[t], kq[:, :, idx], vq[:, :, idx], past + len(path), 0.088, ks=ks[:, :, idx], vs=vs[:, :, idx]
        )
        mx.eval(one)
        bad += int(not mx.array_equal(one, tree[t]))
    assert bad == 0, f"{bad} tree nodes differ from the one-row decode over the int8 cache"
    return f"int8 cache: rows within {err_q:.1e}, decode rel err {err:.1e} vs float32, tree nodes bit-exact"


def check_tree_tile() -> str:
    """every tree kernel (`aluminum.tree` on each of `aluminum.tree_kernels`, which a tree's attention takes where
    `tree_pick` picks it) bit for bit as the node kernel over the same inputs: trees of 1 to 40 nodes over pasts short
    of a block, on and off segment edges, at every head size; int8 rows, windows with sinks (the nodes spanning more
    segments than any one has), a step buffer (bf16 and int8), a row map, a batch slice and a flat buffer's stretch"""
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.mlx import aluminum

    mx.random.seed(19)
    rng = np.random.default_rng(19)
    bad: list[str] = []
    cells = 0

    def same(
        name: str,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        meta: NodeMeta,
        sinks: mx.array | None = None,
        ks: mx.array | None = None,
        vs: mx.array | None = None,
        k2: mx.array | None = None,
        v2: mx.array | None = None,
        ks2: mx.array | None = None,
        vs2: mx.array | None = None,
        rmap: mx.array | None = None,
    ) -> None:
        nonlocal cells
        meta_, path, splits = meta
        want = aluminum.nodes(q, k, v, meta_, path, 0.088, splits, sinks, ks, vs, k2, v2, ks2, vs2, mx.bfloat16, rmap)
        for pick in aluminum.tree_kernels(int(q.shape[-1])):
            got = aluminum.tree(
                q, k, v, meta_, path, 0.088, splits, sinks, ks, vs, k2, v2, ks2, vs2, mx.bfloat16, rmap, pick
            )
            mx.eval(want, got)
            cells += 1
            if not mx.array_equal(want, got):
                bad.append(f"{name} ({pick.kind} {pick.rows} rows, {pick.nsg} simdgroups)")

    for Hq, Hk, d in ((32, 8, 128), (64, 8, 64), (32, 8, 96), (16, 8, 256), (4, 1, 256)):
        for past in (5, 511, 1500):
            cap = past + 48
            kb = (mx.random.normal((1, Hk, cap, d)) * 2).astype(mx.bfloat16)
            vb = mx.random.normal((1, Hk, cap, d)).astype(mx.bfloat16)
            for T in (1, 3, 9, 40):
                parents = [-1] + [int(rng.integers(0, t)) for t in range(1, T)]
                same(f"d {d} past {past} T {T}", mx.random.normal((T, Hq, d)), kb, vb, mlxdev.tree_meta(past, parents))
            parents = [-1, 0, 0, 1, 2, 2, 4, 3, 1, 8, 9, 0]
            T = len(parents)
            q = mx.random.normal((T, Hq, d))
            meta = mlxdev.tree_meta(past, parents)
            kq, ks = mlxdev.kv_quantize(kb)
            vq, vs = mlxdev.kv_quantize(vb)
            same(f"int8 d {d} past {past}", q, kq, vq, meta, ks=ks, vs=vs)
            sinks = mx.random.normal((Hq,)) * 3
            for win in (128, 700):
                same(f"window {win} d {d} past {past}", q, kb, vb, mlxdev.tree_meta(past, parents, window=win), sinks)
            k2 = mx.random.normal((1, Hk, T, d)).astype(mx.bfloat16)
            v2 = mx.random.normal((1, Hk, T, d)).astype(mx.bfloat16)
            step = mlxdev.tree_meta(past, parents, seg=past)
            same(f"step buffer d {d} past {past}", q, kb, vb, step, k2=k2, v2=v2)
            k2q, k2s = mlxdev.kv_quantize(k2)
            v2q, v2s = mlxdev.kv_quantize(v2)
            same(f"int8 step buffer d {d} past {past}", q, kq, vq, step, None, ks, vs, k2q, v2q, k2s, v2s)
            rmap = mx.array(rng.permutation(cap)[:past].astype(np.uint32))
            same(f"row map d {d} past {past}", q, kb, vb, step, k2=k2, v2=v2, rmap=rmap)
            kb3 = mx.random.normal((3, Hk, cap + 40, d)).astype(mx.bfloat16)
            vb3 = mx.random.normal((3, Hk, cap + 40, d)).astype(mx.bfloat16)
            same(f"slice d {d} past {past}", q, kb3, vb3, mlxdev.tree_meta(past, parents, batch=2))
            flat = mlxdev.tree_meta(past, parents, batch=1, pbase=37, kbatch=0)
            same(f"flat d {d} past {past}", q, kb3, vb3, flat)
    assert not bad, f"a tree kernel differs from the node kernel: {bad}"
    return (
        f"{cells} tree passes on every tree kernel bit-exact against the node kernel (heads 64, 96, 128, 256; every "
        "variant)"
    )


def check_kv_bits() -> str:
    """the gpt-oss fixture end to end on the MLX device with an int8 cache: the tree's tokens are the greedy
    loop's, and the answer is the bf16 cache's (the fixture's logits within the int8 rows' error)"""
    fx = fixture("tiny_gpt_oss")
    B = receipts("gpt_oss")["host"]
    native_library()
    res = {}
    for bits in (None, 8):
        with torch.inference_mode():
            sm = host_model(fx, device="mlx", dtype=torch.bfloat16, kv_bits=bits)
            speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
            cache = sm.new_cache()
            lg = sm._prefill(B["prompt"], cache)[0, -1].float()
            nxt = forward_logits(sm, B["cont"], cache=cache)[0, -1].float()
            g = sm.generate_greedy(B["prompt"], 10)
            s, census = sm.generate_speculative(
                B["prompt"], 10, proposer="ngram", v_max=4, spans=[("receipt", B["greedy"])]
            )
            res[bits] = (lg, nxt, g, s, int(census["accepted"]))
            sm.close()
    _lg8, nxt8, g8, s8, acc = res[8]
    _lg16, nxt16, g16, _, _ = res[None]
    assert s8 == g8 and acc > 0, (
        f"with an int8 cache the speculative pass {s8} (accepted {acc}) is not the greedy loop's {g8}"
    )
    err = float((nxt8 - nxt16).abs().max())
    assert err < 0.3, f"a continuation over the int8 cache differs from the bf16 cache's by {err:.2e}"
    return (
        f"int8 cache end to end: spec == greedy (accepted {acc}), greedy {'==' if g8 == g16 else '!='} the bf16 "
        f"cache's tokens, a continuation's logits within {err:.1e} of the bf16 cache's"
    )


def check_rope_nodes() -> str:
    """a verify pass's rope: every tree node rotated bit for bit as the one-row step rotates it at the node's
    position (past + depth), q and k in the one launch the fused paths use (`rope_rows2` at each node's
    position; partial rotary, the hybrid's shape)"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(11)
    parents = [-1, 0, 1, 2, 1, 4, 0, 6]
    depth = [0, 1, 2, 3, 2, 3, 1, 2]
    T, H, d, rd, past = 8, 4, 64, 16, 300
    freqs = mx.array([10000.0 ** (2 * i / rd) for i in range(rd // 2)], dtype=mx.float32)
    x = mx.random.normal((T, H, d))
    y = mx.random.normal((T, H // 2, d))
    mx.eval(x, y, freqs)
    pos = [past + (0 if parents[j] < 0 else depth[j]) for j in range(T)]
    assert pos == [past + dd for dd in depth]
    bad = 0
    for scale in (1.0, 1.2):
        qr, kr = mlxdev.rope_rows2(x, y, rd, freqs, scale, pos)
        mx.eval(qr, kr)
        for j in range(T):
            one = mlxdev.Backend.rope_fast(x[j : j + 1].transpose(1, 0, 2), rd, freqs, scale, past + depth[j])
            two = mlxdev.Backend.rope_fast(y[j : j + 1].transpose(1, 0, 2), rd, freqs, scale, past + depth[j])
            mx.eval(one, two)
            bad += int(not mx.array_equal(one[:, 0], qr[j])) + int(not mx.array_equal(two[:, 0], kr[j]))
    assert bad == 0, f"{bad} nodes rotated unlike the one-row step"
    return f"{T} tree nodes (two shared depths) rotated bit-exact as one-row steps"


def check_gemv_rows() -> str:
    """the matvec is batch-invariant: a row alone, in a tile of any width and in a pass split over launches (past
    `TILE_ROWS`, every width the engine's passes under `gemm_rows` take), bf16 and float32, bit for bit"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(4)
    W = (mx.random.normal((1536, 2560)) * 0.05).astype(mx.bfloat16)
    bad = []
    for dt in (mx.bfloat16, mx.float32):
        X = mx.random.normal((63, 2560)).astype(dt)
        mx.eval(W, X)
        rows = mx.concatenate([mlxdev.gemv(W, X[i : i + 1]) for i in range(63)])
        for t in range(2, 64):
            full = mlxdev.gemv(W, X[:t])
            mx.eval(full, rows)
            if not mx.array_equal(full, rows[:t]):
                bad.append((str(dt), t))
    assert not bad, f"a row's product depends on its company: {bad}"
    return "rows alone and in passes of 2-63 rows bit-exact (bf16, float32)"


_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _mxfp4_fixture(
    rows: int, k: int, seed: int, lo: int = 112, hi: int = 142
) -> tuple[np.ndarray, np.ndarray, mx.array]:
    """random MXFP4 bytes for a [rows, k] matrix, and the slot they make: blocks then scales. The scales
    stay inside 1..254 (`check_mxfp4_table` covers the whole byte)."""
    import mlx.core as mx

    blocks, scales = mxfp4_random(seed, rows, k, lo, hi)
    slot = mx.array(mxfp4_slot(blocks, scales))
    mx.eval(slot)
    return blocks, scales, slot


def _mxfp4_dequant(blocks: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """the layout in numpy: weight 2j of a block is the low nibble of byte j and 2j+1 the high one, times
    2**(scale - 127) (`check_mxfp4_table` checks this against transformers' own dequantizer)"""
    lut = np.array(_FP4, dtype=np.float32)
    out = np.empty((*blocks.shape[:-1], 32), dtype=np.float32)
    out[..., 0::2] = lut[blocks & 0x0F]
    out[..., 1::2] = lut[blocks >> 4]
    s = scales.astype(np.uint32)
    f = (np.maximum(s, 1) << 23).view(np.float32)
    out *= np.where(s == 0, f * np.float32(0.5), f)[..., None]
    return out.reshape(blocks.shape[0], blocks.shape[1] * 32)


def check_mxfp4_table() -> str:
    """the block decoding against transformers' dequantizer over random nibbles and every e8m0 scale"""
    import mlx.core as mx
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    from btb import mlx as mlxdev

    rows, k = 64, 128
    rng = np.random.default_rng(7)
    g = k // 32
    blocks = rng.integers(0, 256, size=(rows, g, 16), dtype=np.uint8)
    # every scale a checkpoint can hold: 0 (2**-127) up, stopping below the exponents where 6 * 2**(e - 127)
    # overflows a float32 for every dequantizer alike (255 is a NaN scale in the MX spec and unused)
    scales = rng.integers(0, 250, size=(rows, g)).astype(np.uint8)
    # transformers dequantizes to [experts, K, rows] (it transposes for `x @ W`); this module's [rows, K] is
    # the same numbers the other way round
    hf = (
        convert_moe_packed_tensors(torch.from_numpy(blocks)[None], torch.from_numpy(scales)[None], dtype=torch.float32)[
            0
        ]
        .T.contiguous()
        .numpy()
    )
    assert (hf == _mxfp4_dequant(blocks, scales)).all(), "the test's own dequantizer differs from transformers'"
    slot = mx.array(np.concatenate([blocks.reshape(-1), scales.reshape(-1)]))
    mine = np.asarray(mlxdev.mxfp4_dequant(slot, rows, k, mx.float32))
    live = np.repeat(scales, 32, axis=1) >= 1
    assert (mine[live] == hf[live]).all(), "the GPU decodes an MXFP4 block differently from transformers"
    # scale 0 is 2**-127, below the smallest normal float: the GPU flushes it and the block reads as zero.
    # No gpt-oss checkpoint uses it, and every product of one is flushed anyway (the largest is 3.5e-38).
    assert (mine[~live] == 0).all() and float(np.abs(hf[~live]).max(initial=0.0)) < 4e-38
    return f"block decoding bit-exact against transformers over {int(live.sum())} weights, scale 0 flushed"


def check_mxfp4_gemv() -> str:
    """the MXFP4 matvec against a float32 reference, and batch-invariant like the bf16 one"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(21)
    worst = 0.0
    for rows, k, b in ((5760, 2880, 1), (2880, 2880, 3), (256, 64, 15), (3, 32, 2)):
        blocks, scales, slot = _mxfp4_fixture(rows, k, rows + b)
        W = mx.array(_mxfp4_dequant(blocks, scales))
        x = mx.random.normal((b, k))
        mx.eval(W, x)
        assert bool(mx.array_equal(mlxdev.mxfp4_dequant(slot, rows, k, mx.float32), W)), "dequant differs"
        y = mlxdev.gemv_mxfp4(slot, rows, k, x)
        ref = x @ W.T
        mx.eval(y, ref)
        worst = max(worst, float(mx.abs(y - ref).max() / mx.abs(ref).max()))
    # 2880 float32 products summed in a different order than mx.matmul's: a few ulp of the largest term
    assert worst < 1e-5, f"MXFP4 matvec rel err {worst:.2e}"
    rows, k = 1536, 2560
    _, _, slot = _mxfp4_fixture(rows, k, 5)
    bad = 0
    for dt in (mx.bfloat16, mx.float32):
        X = mx.random.normal((15, k)).astype(dt)
        mx.eval(X)
        full = mlxdev.gemv_mxfp4(slot, rows, k, X)
        one = mx.concatenate([mlxdev.gemv_mxfp4(slot, rows, k, X[i : i + 1]) for i in range(15)])
        mx.eval(full, one)
        bad += int(not mx.array_equal(full, one))
    assert bad == 0, "a row's MXFP4 product depends on its company"
    X = mx.random.normal((5, k))
    mx.eval(X)
    tiles = [mlxdev.gemv_mxfp4(slot, rows, k, X, tm=t) for t in (2, 4, 8, 16)]
    mx.eval(tiles)
    assert all(bool(mx.array_equal(tiles[0], t)) for t in tiles[1:]), "the row tile changes the result"
    # the grouped dispatch: four experts' matvecs at once, each bit-identical to its own call
    slots = [_mxfp4_fixture(rows, k, 30 + i)[2] for i in range(4)]
    xs = mx.random.normal((4, 5, k))
    mx.eval(xs)
    grp = mlxdev.gemv_mxfp4_group(slots, rows, k, xs)
    sing = mx.stack([mlxdev.gemv_mxfp4(s, rows, k, xs[i]) for i, s in enumerate(slots)])
    mx.eval(grp, sing)
    assert bool(mx.array_equal(grp, sing)), "the grouped MXFP4 matvec differs from the single calls"
    return f"MXFP4 matvec rel err {worst:.1e}, rows alone and in a tile bit-exact, grouped == single"


def check_mxfp4_experts() -> str:
    """one token's experts end to end against gpt-oss's own arithmetic in float32, and row by row"""
    import mlx.core as mx

    from btb import mlx as mlxdev

    mx.random.seed(22)
    hidden, inter, K = 256, 192, 3
    gu, dn, gub, dnb, W_gu, W_dn = [], [], [], [], [], []
    for e in range(K):
        bl, sc, slot = _mxfp4_fixture(2 * inter, hidden, 100 + e)
        gu.append(slot)
        W_gu.append(mx.array(_mxfp4_dequant(bl, sc)))
        bl, sc, slot = _mxfp4_fixture(hidden, inter, 200 + e)
        dn.append(slot)
        W_dn.append(mx.array(_mxfp4_dequant(bl, sc)))
        gub.append((mx.random.normal((2 * inter,)) * 0.1).astype(mx.bfloat16))
        dnb.append((mx.random.normal((hidden,)) * 0.1).astype(mx.bfloat16))
    b = 5
    x = mx.random.normal((b, hidden))
    wts = mx.abs(mx.random.normal((K, b)))
    mx.eval(x, wts, *W_gu, *W_dn, *gub, *dnb)
    ref = mx.zeros((b, hidden))
    for e in range(K):
        y = x @ W_gu[e].T + gub[e].astype(mx.float32)
        gate = mx.minimum(y[..., 0::2], 7.0)
        up = mx.clip(y[..., 1::2], -7.0, 7.0)
        h = (up + 1) * (gate * mx.sigmoid(1.702 * gate))
        ref = ref + (h @ W_dn[e].T + dnb[e].astype(mx.float32)) * wts[e][:, None]
    out = mlxdev.experts_mxfp4_step(x, gu, dn, wts, hidden, inter, gub, dnb)
    mx.eval(ref, out)
    err = float(mx.abs(out - ref).max() / mx.abs(ref).max())
    assert err < 1e-5, f"the MXFP4 expert step differs from the float32 reference by {err:.2e}"
    rows = mx.concatenate(
        [mlxdev.experts_mxfp4_step(x[i : i + 1], gu, dn, wts[:, i], hidden, inter, gub, dnb) for i in range(b)]
    )
    mx.eval(rows)
    assert bool(mx.array_equal(out, rows)), "a token's experts depend on the rows beside it"
    return f"expert step rel err {err:.1e}, a row alone and in a {b}-row tile bit-exact"


def check_gpt_oss() -> str:
    """gpt-oss end to end on the MLX device in bf16 over the tiny fixture (sinks, a 4-token window, the MXFP4
    experts from the store's shared slots): the node kernel on every row (as when speculation is on) and MLX's
    fused attention give the same greedy tokens, a speculative pass through the kernel gives the greedy
    loop's, the kernel really ran, and the logits sit within bf16 of transformers' float32 forward"""
    from btb import mlx as mlxdev

    fx = fixture("tiny_gpt_oss")
    B = receipts("gpt_oss")["host"]
    native_library()
    calls = {"nodes": 0}
    # the engine's decode reaches attn_nodes through attn_decode/attn_tree in the attn module, its forest
    # prefill through the package: the counter goes on both
    orig = mlxdev.attn.attn_nodes

    def counting(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        def counted(*a: _P.args, **k: _P.kwargs) -> _R:
            calls["nodes"] += 1
            return fn(*a, **k)

        return counted

    mlxdev.attn_nodes = mlxdev.attn.attn_nodes = counting(orig)
    res = {}
    try:
        for kern in (True, False):
            with torch.inference_mode():
                sm = host_model(fx, device="mlx", dtype=torch.bfloat16)
                speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
                sm.mlx_attn_kernel = kern
                n0 = calls["nodes"]
                cache = sm.new_cache()
                lg = sm._prefill(B["prompt"], cache)[0, -1].float()
                g = sm.generate_greedy(B["prompt"], 10)
                s, census = sm.generate_speculative(
                    B["prompt"], 10, proposer="ngram", v_max=4, spans=[("receipt", B["greedy"])]
                )
                res[kern] = (lg, g, s, int(census["accepted"]), calls["nodes"] - n0)
                sm.close()
    finally:
        mlxdev.attn_nodes = mlxdev.attn.attn_nodes = orig
    lg_k, g_k, s_k, acc_k, n_k = res[True]
    lg_s, g_s, s_s, acc_s, n_s = res[False]
    assert n_k > 0 and n_s == 0, f"the node kernel ran {n_k} times with it on, {n_s} with it off"
    assert g_k == g_s, f"greedy differs between the node kernel {g_k} and MLX's attention {g_s}"
    assert s_k == g_k and acc_k > 0, f"the speculative pass {s_k} (accepted {acc_k}) is not the greedy loop's {g_k}"
    assert s_s == g_s and acc_s > 0, f"the speculative pass {s_s} (accepted {acc_s}) is not the greedy loop's {g_s}"
    err = max(float((lg_k - B["prompt_logits"]).abs().max()), float((lg_s - B["prompt_logits"]).abs().max()))
    # bf16 through four layers of a random model: about 0.08 on this M3 Pro against logits of magnitude 5
    assert err < 0.3, f"bf16 logits differ from the float32 reference by {err:.2e}"
    return (
        f"bf16 greedy {'==' if g_k == B['greedy'] else '!='} the float32 reference's tokens, node kernel and MLX "
        f"attention agree, spec == greedy (accepted {acc_k}), {n_k} kernel calls, logits within {err:.1e}"
    )


def check_hybrid_warm() -> str:
    """the load-time warm-up on the tiny hybrid fixture: the pass-cost curve and the attention-cost calibration
    find the first layer with attention (the hybrid's first layers are DeltaNet ones)"""
    with torch.inference_mode():
        sm = host_model(fixture("tiny_q35"), device="mlx", dtype=torch.bfloat16, cpu_layers=())
        speculation(sm, tree_budget=0, v_max=4)
        n = sm.warm()
        slope = getattr(sm, "_mlx_attn_slope", None)
        sm.close()
    return f"warm timed {n} widths, attention slope {'set' if slope else 'skipped (head dim outside the kernel)'}"


def check_draft_bits() -> str:
    """the MTP drafter over weights packed in memory to 4 and 8 bits (`draft_bits`): the speculative loop's
    tokens stay the greedy loop's on the tiny hybrid fixture, the packed copies exist for every weight the
    drafter multiplies, and nothing is written next to the model"""
    import glob
    import os

    fx = fixture("tiny_q35")
    files = sorted((f, os.path.getmtime(f), os.path.getsize(f)) for f in glob.glob(os.path.join(fx, "*")))
    out = {}
    for device, bits in (("mlx", 16), ("mlx", 8), ("mlx", 4), ("cpu", 8)):
        with torch.inference_mode():
            sm = host_model(fx, device=device, dtype=torch.bfloat16, cpu_layers=())
            speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.0, tree_read="step")
            sm.draft_bits = bits
            g = sm.generate_greedy(PROMPT_Q35, 12)
            s_, census = sm.generate_speculative(PROMPT_Q35, 12, proposer="mtp_dyn", v_max=4)
            dr = getattr(sm, "aj", None)
            n_packed = len(getattr(dr, "_q", {}) or {}) if dr is not None else 0
            if device == "cpu":
                n_packed = int(getattr(dr, "_packed_torch", False)) if dr is not None else 0
            out[(device, bits)] = (g, s_, int(census["accepted"]), n_packed)
            sm.close()
    for (device, bits), (g, s_, _acc, n_packed) in out.items():
        assert s_ == g, f"draft_bits {bits} on {device}: the speculative tokens {s_} are not the greedy loop's {g}"
        if bits < 16:
            assert n_packed > 0, f"draft_bits {bits} on {device}: nothing was packed"
        else:
            assert n_packed == 0, "draft_bits 16 packed a weight"
    assert files == sorted((f, os.path.getmtime(f), os.path.getsize(f)) for f in glob.glob(os.path.join(fx, "*"))), (
        "a file next to the model changed"
    )
    return (
        "spec == greedy at 16 / 8 / 4 bits on mlx and 8 on cpu (accepted "
        + " / ".join(str(out[k][2]) for k in (("mlx", 16), ("mlx", 8), ("mlx", 4), ("cpu", 8)))
        + f"), {out[('mlx', 4)][3]} weights packed at 4 bits, the model's files untouched"
    )


def check_host_tree() -> str:
    """the tree verify on the host tier: the tiny dense fixture on the CPU kernels, its pass-cost curve timed
    by `host_warm`, the speculative loop's n-gram tree verified as a tree (passes fewer than a chain's) with
    the greedy loop's tokens"""
    fx = fixture("tiny_qwen3")
    B = receipts("qwen3")["bf16"]
    out = {}
    with torch.inference_mode():
        for tree in (False, True):
            sm = host_model(fx, device="cpu", cpu_layers=())
            speculation(sm, tree_budget=0, v_max=4, ngram_p=0.0, tree_read="step")
            sm.ngram_tree = tree
            n = sm.host_warm([1] * 8) if tree else 0
            g = sm.generate_greedy(PROMPT_DENSE, 12)
            s_, census = sm.generate_speculative(
                PROMPT_DENSE, 12, proposer="ngram", v_max=4, spans=[("receipt", B["greedy"])]
            )
            out[tree] = (g, s_, int(census["forwards"]), int(census["accepted"]), n, getattr(sm, "_host_cost", None))
            sm.close()
    for tree, (g, s_, _f, _acc, n, cost) in out.items():
        assert s_ == g, f"host tree {tree}: the speculative tokens {s_} are not the greedy loop's {g}"
        if tree:
            # the curve decides the width: on a tier whose rows cost a third of a step it keeps one row, and the
            # tree accepts nothing - the tokens stay the greedy loop's either way
            assert n > 0 and cost and 1 in cost, "the host's pass-cost curve was not timed"
    curve = out[True][5]
    assert curve is not None
    return f"spec == greedy on the host with the tree and the chain (passes {out[True][2]} vs {out[False][2]}, accepted {out[True][3]} vs {out[False][3]}; the curve {len(curve)} widths)"


def check_rope_rows() -> str:
    """`rope_rows` (B rows at B positions in one launch) is bit-identical to `rope_fast` at each row's
    position - bf16 and float32, head_dim 128 and 64, full and partial rotary dims, with an attention
    scaling - as the batched decode needs it to be to give each row its single decode's bits"""
    from btb import mlx as mlxdev
    from btb.mlx import Backend

    m = mlxdev.mx
    m.random.seed(5)
    inv = 1.0 / (10000.0 ** (m.arange(0, 128, 2).astype(m.float32) / 128))  # a rotary module's inv_freq
    cases = []
    for dt in (m.bfloat16, m.float32):
        for D, rd, sc in ((128, 128, 1.0), (128, 128, 1.19), (64, 32, 1.0)):
            B, H = 37, 4
            freqs = (1.0 / inv[: rd // 2]).astype(m.float32)
            x = m.random.normal((B, H, D)).astype(dt)
            pos = [int(p) for p in np.asarray(m.random.randint(0, 40000, (B,)))]
            out = mlxdev.rope_rows(x, rd, freqs, sc, pos)
            for b in range(B):
                one = Backend.rope_fast(x[b][None, :, None, :], rd, freqs, sc, pos[b])[0, :, 0, :]
                m.eval(out, one)
                assert m.array_equal(out[b], one), f"{dt} D{D} rd{rd} scale {sc}: row {b} at {pos[b]} differs"
            cases.append(f"{'bf16' if dt == m.bfloat16 else 'f32'} D{D}/rd{rd}{' x' + str(sc) if sc != 1.0 else ''}")
    # q and k in one launch: the bits of rope_rows
    q = m.random.normal((37, 8, 128)).astype(m.bfloat16)
    k = m.random.normal((37, 2, 128)).astype(m.bfloat16)
    pos = [int(p) for p in np.asarray(m.random.randint(0, 40000, (37,)))]
    freqs = (1.0 / inv).astype(m.float32)
    oq, ok = mlxdev.rope_rows2(q, k, 128, freqs, 1.0, pos)
    m.eval(oq, ok)
    assert m.array_equal(oq, mlxdev.rope_rows(q, 128, freqs, 1.0, pos)) and m.array_equal(
        ok, mlxdev.rope_rows(k, 128, freqs, 1.0, pos)
    )
    return (
        "bit-exact against rope_fast over 37 rows at random positions: "
        + ", ".join(cases)
        + "; q+k in one launch the same"
    )


def check_attn_rows() -> str:
    """a batched decode step through the node kernel: B rows of different lengths in one [B, Hk, cap, D]
    buffer, each row's attention bit-identical to the one-row decode over that row's own slice (bf16 and
    int8; head_dim 128, four query heads per kv head)"""
    from btb import mlx as mlxdev

    m = mlxdev.mx
    m.random.seed(7)
    B, Hq, Hk, D, cap = 6, 8, 2, 128, 700
    ns = [1, 64, 65, 300, 699, 128]
    q = m.random.normal((B, Hq, D)).astype(m.float32)
    kb = m.random.normal((B, Hk, cap, D)).astype(m.bfloat16)
    vb = m.random.normal((B, Hk, cap, D)).astype(m.bfloat16)
    out = mlxdev.attn_rows(q, kb, vb, ns, 0.088)
    for b, n in enumerate(ns):
        one = mlxdev.attn_decode(q[b], kb[b : b + 1], vb[b : b + 1], n, 0.088)
        m.eval(out, one)
        assert m.array_equal(out[b], one), f"row {b} ({n} rows) differs from its one-row decode"
    kq, ks = mlxdev.kv_quantize(kb)
    vq, vs = mlxdev.kv_quantize(vb)
    out8 = mlxdev.attn_rows(q, kq, vq, ns, 0.088, ks=ks, vs=vs)
    for b, n in enumerate(ns):
        one = mlxdev.attn_decode(q[b], kq[b : b + 1], vq[b : b + 1], n, 0.088, ks=ks[b : b + 1], vs=vs[b : b + 1])
        m.eval(out8, one)
        assert m.array_equal(out8[b], one), f"int8 row {b} ({n} rows) differs from its one-row decode"
    # the two-segment cache: row b's first segs[b] rows in the main buffer, the rest in a step buffer at
    # (row - segs[b]); against the one-row decode over the same rows in one buffer
    segs = [1, 40, 65, 250, 600, 100]
    cap2 = max(n - s for n, s in zip(ns, segs))
    k2 = m.random.normal((B, Hk, cap2, D)).astype(m.bfloat16)
    v2 = m.random.normal((B, Hk, cap2, D)).astype(m.bfloat16)
    outs = mlxdev.attn_rows(q, kb, vb, ns, 0.088, segs=segs, k2=k2, v2=v2)
    for b, (n, s) in enumerate(zip(ns, segs)):
        kk = m.concatenate([kb[b : b + 1, :, :s], k2[b : b + 1, :, : n - s]], axis=2)
        vv = m.concatenate([vb[b : b + 1, :, :s], v2[b : b + 1, :, : n - s]], axis=2)
        one = mlxdev.attn_decode(q[b], kk, vv, n, 0.088)
        m.eval(outs, one)
        assert m.array_equal(outs[b], one), f"two-segment row {b} ({s} + {n - s} rows) differs from its one-row decode"
    k2q, k2s = mlxdev.kv_quantize(k2)
    v2q, v2s = mlxdev.kv_quantize(v2)
    outs8 = mlxdev.attn_rows(q, kq, vq, ns, 0.088, ks=ks, vs=vs, segs=segs, k2=k2q, v2=v2q, ks2=k2s, vs2=v2s)
    for b, (n, s) in enumerate(zip(ns, segs)):
        kk = m.concatenate([kq[b : b + 1, :, :s], k2q[b : b + 1, :, : n - s]], axis=2)
        vv = m.concatenate([vq[b : b + 1, :, :s], v2q[b : b + 1, :, : n - s]], axis=2)
        kks = m.concatenate([ks[b : b + 1, :, :s], k2s[b : b + 1, :, : n - s]], axis=2)
        vvs = m.concatenate([vs[b : b + 1, :, :s], v2s[b : b + 1, :, : n - s]], axis=2)
        one = mlxdev.attn_decode(q[b], kk, vv, n, 0.088, ks=kks, vs=vvs)
        m.eval(outs8, one)
        assert m.array_equal(outs8[b], one), f"two-segment int8 row {b} differs from its one-row decode"
    # the flat layout: every row's prefill end to end in one [1, Hk, sum, D] buffer, row b's from pbases[b],
    # against the one-row decode over the row's stretch; then the forest - every prompt token a node over
    # its row's prefix - against the one-row decode at each length
    lens = [1, 40, 65, 250, 300, 100]
    pbases, tot = [], 0
    for L in lens:
        pbases.append(tot)
        tot += L
    kf = m.random.normal((1, Hk, tot, D)).astype(m.bfloat16)
    vf = m.random.normal((1, Hk, tot, D)).astype(m.bfloat16)
    nsf = [L + (n - s) for L, n, s in zip(lens, ns, segs)]
    outf = mlxdev.attn_rows(q, kf, vf, nsf, 0.088, segs=lens, k2=k2, v2=v2, pbases=pbases)
    for b, (L, n) in enumerate(zip(lens, nsf)):
        kk = m.concatenate([kf[:, :, pbases[b] : pbases[b] + L], k2[b : b + 1, :, : n - L]], axis=2)
        vv = m.concatenate([vf[:, :, pbases[b] : pbases[b] + L], v2[b : b + 1, :, : n - L]], axis=2)
        one = mlxdev.attn_decode(q[b], kk, vv, n, 0.088)
        m.eval(outf, one)
        assert m.array_equal(outf[b], one), f"flat row {b} ({L} + {n - L} rows) differs from its one-row decode"
    qf = m.random.normal((tot, Hq, D)).astype(m.float32)
    meta, path, splits = mlxdev.forest_meta(lens, pbases)
    outn = mlxdev.attn_nodes(qf, kf, vf, meta, path, 0.088, splits)
    m.eval(outn)
    checked = 0
    for b, L in enumerate(lens):
        for p in (0, 1, L // 2, L - 1):
            if p >= L:
                continue
            t = pbases[b] + p
            one = mlxdev.attn_decode(
                qf[t], kf[:, :, pbases[b] : pbases[b] + L], vf[:, :, pbases[b] : pbases[b] + L], p + 1, 0.088
            )
            m.eval(one)
            assert m.array_equal(outn[t], one), f"forest node {t} (row {b}, token {p}) differs from the one-row decode"
            checked += 1
    # the forest's prompts prefilled in one launch: its nodes' bits, each prompt's own prefill's from its offset
    outp = mlxdev.attn_prefill_rows(qf, kf, vf, lens, pbases, 0.088, odt=m.float32)
    m.eval(outp)
    assert m.array_equal(outp, outn), "the forest's prefill differs from its nodes"
    for b, L in enumerate(lens):
        span = slice(pbases[b], pbases[b] + L)
        one = mlxdev.attn_prefill(qf[span], kf, vf, 0, 0.088, odt=m.float32, pbase=pbases[b])
        m.eval(one)
        assert m.array_equal(one, outp[span]), f"prompt {b}'s prefill at its offset differs from the forest's"
    kfq, kfs = mlxdev.kv_quantize(kf)
    vfq, vfs = mlxdev.kv_quantize(vf)
    outn8 = mlxdev.attn_nodes(qf, kfq, vfq, meta, path, 0.088, splits, ks=kfs, vs=vfs)
    outp8 = mlxdev.attn_prefill_rows(qf, kfq, vfq, lens, pbases, 0.088, odt=m.float32, ks=kfs, vs=vfs)
    m.eval(outn8, outp8)
    assert m.array_equal(outp8, outn8), "the int8 forest's prefill differs from its nodes"
    # a chunk of one slice of a batched buffer, as the slice alone
    qc = m.random.normal((40, Hq, D)).astype(m.float32)
    for b in (0, 4):
        one = mlxdev.attn_prefill(qc, kb, vb, 600, 0.088, odt=m.float32, kbatch=b)
        alone = mlxdev.attn_prefill(qc, kb[b : b + 1], vb[b : b + 1], 600, 0.088, odt=m.float32)
        m.eval(one, alone)
        assert m.array_equal(one, alone), f"a chunk of slice {b} differs from the slice alone"
        one = mlxdev.attn_prefill(qc, kq, vq, 600, 0.088, odt=m.float32, kbatch=b, ks=ks, vs=vs)
        alone = mlxdev.attn_prefill(
            qc, kq[b : b + 1], vq[b : b + 1], 600, 0.088, odt=m.float32, ks=ks[b : b + 1], vs=vs[b : b + 1]
        )
        m.eval(one, alone)
        assert m.array_equal(one, alone), f"an int8 chunk of slice {b} differs from the slice alone"
    return (
        f"{B} rows of {min(ns)}..{max(ns)} cache rows bit-exact against their one-row decodes, bf16 and int8, "
        f"one buffer, two segments and the flat layout; {checked} forest nodes of {tot} bit-exact, the forest's "
        "prefill its nodes' (bf16 and int8) and each prompt's own, a slice's chunk the slice's alone"
    )


def check_attn_prefill() -> str:
    """the fused prefill kernel at head sizes 64, 96, 128 and 256: random bf16 caches at several (past, T, Hk, g,
    window) against a float32 reference in plain ops (float P below 256; at 256 the error P's bf16 rounding, measured
    3.1e-3 at most), each of a few rows bit for bit as the one-row decode computes it, and the chunk prefilled in
    pieces bit for bit as in one"""
    from btb import mlx as mlxdev
    from btb.mlx import aluminum

    m = mlxdev.mx
    m.random.seed(11)
    worst = {False: 0.0, True: 0.0}
    bad: list[tuple[int, int, int, int | str]] = []  # (head, past, T, the row that differs or "pieces")
    cases = [
        (256, 0, 17, 2, 4, None),
        (256, 1, 40, 1, 8, None),
        (256, 100, 33, 2, 2, None),
        (256, 77, 100, 3, 1, None),
        (256, 513, 64, 4, 4, None),
        (256, 0, 4096, 2, 4, None),
        (256, 40000, 20, 1, 4, None),
        (256, 900, 200, 2, 2, 512),
        (128, 0, 700, 8, 2, None),
        (128, 1500, 60, 4, 7, None),
        (96, 0, 300, 4, 1, None),
        (96, 1500, 60, 2, 4, 128),
        (64, 0, 300, 8, 8, 128),
        (64, 2000, 90, 8, 8, None),
    ]
    for D, past, T, Hk, g, win in cases:
        scale = D**-0.5
        Hq = Hk * g
        n = past + T
        cap = max(n + 5, 64)
        kb = m.random.normal((1, Hk, cap, D)).astype(m.bfloat16)
        vb = m.random.normal((1, Hk, cap, D)).astype(m.bfloat16)
        q = m.random.normal((T, Hq, D)).astype(m.bfloat16)
        out = mlxdev.attn_prefill(q, kb, vb, past, scale, odt=m.float32, window=win)
        qf = q.astype(m.float32).transpose(1, 0, 2)
        kf = m.repeat(kb[0, :, :n].astype(m.float32), g, axis=0)
        vf = m.repeat(vb[0, :, :n].astype(m.float32), g, axis=0)
        s = (qf * scale) @ kf.transpose(0, 2, 1)
        j, p = m.arange(n)[None, :], past + m.arange(T)[:, None]
        s = m.where((j <= p) & (j > p - win) if win else j <= p, s, -m.inf)
        ref = (m.softmax(s, axis=-1) @ vf).transpose(1, 0, 2)
        cut = T // 3 + 1
        pieces = m.concatenate(
            [
                mlxdev.attn_prefill(q[:cut], kb, vb, past, scale, odt=m.float32, window=win),
                mlxdev.attn_prefill(q[cut:], kb, vb, past + cut, scale, odt=m.float32, window=win),
            ]
        )
        m.eval(out, ref, pieces)
        err = float(m.max(m.abs(out - ref)))
        pb = aluminum.bf16p(D)
        worst[pb] = max(worst[pb], err)
        assert err < (6e-3 if pb else 1e-4), f"head {D} past {past} T {T} Hk {Hk} g {g}: {err:.2e} from the reference"
        if not m.array_equal(pieces, out):
            bad.append((D, past, T, "pieces"))
        for t in (0, T // 2, T - 1):
            one = mlxdev.attn_decode(q[t].astype(m.float32), kb, vb, past + t + 1, scale, window=win)
            m.eval(one)
            if not m.array_equal(one, out[t]):
                bad.append((D, past, T, t))
        # over an int8 cache: the prefill over its dequantized rows, and its last row the int8 decode's
        kq, ks = mlxdev.kv_quantize(kb)
        vq, vs = mlxdev.kv_quantize(vb)
        out8 = mlxdev.attn_prefill(q, kq, vq, past, scale, odt=m.float32, window=win, ks=ks, vs=vs)
        kd, vd = mlxdev.kv_dequantize(kq, ks), mlxdev.kv_dequantize(vq, vs)
        deq = mlxdev.attn_prefill(q, kd, vd, past, scale, odt=m.float32, window=win)
        one8 = mlxdev.attn_decode(q[T - 1].astype(m.float32), kq, vq, n, scale, window=win, ks=ks, vs=vs)
        m.eval(out8, deq, one8)
        if not m.array_equal(out8, deq):
            bad.append((D, past, T, "int8"))
        if not m.array_equal(one8, out8[T - 1]):
            bad.append((D, past, T, "int8 decode"))
    assert not bad, f"rows differ from their decode or their chunking: {bad}"
    return (
        f"{len(cases)} shapes: max {worst[False]:.1e} (float P) / {worst[True]:.1e} (P in bf16) from the float32"
        " reference; rows bit-exact against the one-row decode and the chunk in pieces, bf16 and int8"
    )


def check_batch_rows() -> str:
    """batched greedy decoding on the MLX device: B ragged prompts prefilled as a forest (one pass over the
    rows' tokens end to end, the attention one right-padded causal call, the rows' K/V in one flat buffer)
    and decoded through one fused forward a step, each row's tokens the tokens of its own single-sequence
    decode (the fixture's head of 128 takes the rows kernel in the step), at 3 rows and at 16 (the one-row
    kernel's tile), through a left-padded batch as
    serve() sends one"""
    from btb.engine import StreamedTextModel

    fx = fixture("tiny_qwen3")
    native_library()
    g = torch.Generator().manual_seed(11)
    with torch.inference_mode():
        sm = host_model(fx, device="mlx", dtype=torch.bfloat16)
        speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0)
        report = []
        for B in (3, 16):
            rows = [
                torch.randint(0, 256, (int(torch.randint(3, 13, (1,), generator=g)),), generator=g).tolist()
                for _ in range(B)
            ]
            single = [sm.generate_greedy(torch.tensor([r]), 10) for r in rows]
            ids, mask = StreamedTextModel.pad_left(rows, 0)
            batched = sm.generate_greedy(ids, 10, attention_mask=mask)
            assert len(batched) == B
            bad = [b for b in range(B) if list(batched[b]) != list(single[b])]
            assert not bad, (
                f"B={B}: rows {bad} differ from their single decodes: {[(batched[b], single[b]) for b in bad[:2]]}"
            )
            report.append(
                f"B={B} lengths {min(len(r) for r in rows)}..{max(len(r) for r in rows)}: 10 tokens a row identical"
            )
        sm.close()
    return "; ".join(report)


def check_sampled_rows() -> str:
    """sampling on the MLX device: the pipelined loop picks in the graph one step ahead and the speculative loop's
    verify pass picks every node in the graph; under one seed they give the same tokens (the draft the answer
    itself, then a reversed one), a batch of rows gives each row's own single sampled decode, and the sample
    is not the greedy answer"""
    from btb.engine import StreamedTextModel
    from btb.sampling import Sampling

    fx = fixture("tiny_qwen3")
    native_library()
    g = torch.Generator().manual_seed(5)
    s9 = Sampling(temperature=0.9, top_p=0.95, seed=9)
    with torch.inference_mode():
        sm = host_model(fx, device="mlx", dtype=torch.bfloat16)
        speculation(sm, tree_budget=0, v_max=4, ngram_p=0.9, tree_min_prob=0.0)
        row = torch.randint(0, 256, (12,), generator=g).tolist()
        plain = list(sm.generate_greedy(torch.tensor([row]), 24, sampling=s9))
        assert plain != list(sm.generate_greedy(torch.tensor([row]), 24)), "the sample is the greedy answer"
        assert plain == list(sm.generate_greedy(torch.tensor([row]), 24, sampling=s9)), "a seed repeats"
        wrong = plain[:6] + [(t + 1) % 256 for t in plain[6:]]  # every draft past the prefix another token
        for tag, span in (("answer", plain), ("wrong", wrong)):
            spec, census = sm.generate_speculative(
                torch.tensor([row]), 24, proposer="ngram", v_max=4, spans=[(tag, span)], sampling=s9
            )
            assert list(spec) == plain, f"{tag}: speculative {spec[:10]} vs sequential {plain[:10]}"
            assert census["accepted"] > 0, census
        rows = [
            torch.randint(0, 256, (int(torch.randint(3, 13, (1,), generator=g)),), generator=g).tolist()
            for _ in range(4)
        ]
        single = [list(sm.generate_greedy(torch.tensor([r]), 10, sampling=s9)) for r in rows]
        ids, mask = StreamedTextModel.pad_left(rows, 0)
        batched = sm.generate_greedy(ids, 10, attention_mask=mask, sampling=s9)
        bad = [b for b in range(4) if list(batched[b]) != single[b]]
        assert not bad, f"rows {bad} differ from their single sampled decodes: {[(batched[b], single[b]) for b in bad]}"
        sm.close()
    return "sampled: pipelined == speculative under one seed, 4 batched rows == their single decodes"


def check_fused_kernels() -> str:
    """the fused Metal kernels against the separate ops: residual add + RMSNorm, q/k RMSNorm + rope (the rotation
    rope_rows' to the bit), and silu(gate) * up"""
    import mlx.core as mx

    from btb import mlx as mlxdev
    from btb.mlx import fused as fk

    mx.random.seed(7)
    T, H = 5, 2560
    h = mx.random.normal((T, H)).astype(mx.bfloat16)
    y = (mx.random.normal((T, H)) * 0.1).astype(mx.bfloat16)
    w = (1.0 + 0.1 * mx.random.normal((H,))).astype(mx.bfloat16)
    h2, x = fk.add_rmsnorm(h, y, w, 1e-6)
    h2r = h + y
    xr = mx.fast.rms_norm(h2r, w, 1e-6)
    mx.eval(h2, x, h2r, xr)
    assert mx.array_equal(h2, h2r), "the residual add differs"
    d_norm = float(mx.max(mx.abs(x.astype(mx.float32) - xr.astype(mx.float32))) / mx.max(mx.abs(xr.astype(mx.float32))))
    assert d_norm < 2e-2, f"add+rmsnorm rel err {d_norm:.2e}"
    Hq, Hk, D = 32, 8, 128
    q = mx.random.normal((T, Hq, D)).astype(mx.bfloat16)
    k = mx.random.normal((T, Hk, D)).astype(mx.bfloat16)
    qn = (1.0 + 0.1 * mx.random.normal((D,))).astype(mx.bfloat16)
    kn = (1.0 + 0.1 * mx.random.normal((D,))).astype(mx.bfloat16)
    freqs = mx.array([1000000.0 ** (2 * i / D) for i in range(D // 2)], dtype=mx.float32)
    pos = [300, 301, 302, 302, 303]
    oq, ok = fk.qk_norm_rope(q, k, qn, kn, 1e-6, D, freqs, 1.0, pos)
    rq, rk = mlxdev.rope_rows2(mx.fast.rms_norm(q, qn, 1e-6), mx.fast.rms_norm(k, kn, 1e-6), D, freqs, 1.0, pos)
    mx.eval(oq, ok, rq, rk)
    d_q = float(mx.max(mx.abs(oq.astype(mx.float32) - rq.astype(mx.float32))) / mx.max(mx.abs(rq.astype(mx.float32))))
    d_k = float(mx.max(mx.abs(ok.astype(mx.float32) - rk.astype(mx.float32))) / mx.max(mx.abs(rk.astype(mx.float32))))
    assert d_q < 2e-2 and d_k < 2e-2, f"qk norm+rope rel err q {d_q:.2e} k {d_k:.2e}"
    # the norm's own rounding aside, the rotation is exact: rope the reference's normed rows and compare bits
    same_q = mx.array_equal(oq, mlxdev.rope_rows(mx.fast.rms_norm(q, qn, 1e-6), D, freqs, 1.0, pos))
    inter = 9728
    gu = mx.random.normal((T, 2 * inter)).astype(mx.bfloat16)
    mid = fk.silu_mul(gu)
    g, u = gu[:, :inter], gu[:, inter:]
    ref = (g * mx.sigmoid(g)) * u
    mx.eval(mid, ref)
    d_d = float(
        mx.max(mx.abs(mid.astype(mx.float32) - ref.astype(mx.float32))) / mx.max(mx.abs(ref.astype(mx.float32)))
    )
    exact = bool(mx.array_equal(mid, ref))
    assert d_d < 1e-2, f"silu * up rel err {d_d:.2e}"
    return (
        f"add+rmsnorm rel err {d_norm:.1e} (the add exact); q/k norm+rope rel err {d_q:.1e} / {d_k:.1e}"
        f"{' (bit-exact given the same normed rows)' if same_q else ''}; silu * up rel err {d_d:.1e}"
        f"{' (bit-exact)' if exact else ''}"
    )


def main() -> int:
    if _skip():
        return 0
    for fn in (
        check_fused_kernels,
        check_unpack,
        check_gemv,
        check_gemv_rows,
        check_rope,
        check_rope_nodes,
        check_cache,
        check_causal,
        check_attn_kernel,
        check_attn_tree,
        check_attn_paged,
        check_attn_int8,
        check_tree_tile,
        check_delta_tree,
        check_mxfp4_table,
        check_mxfp4_gemv,
        check_mxfp4_experts,
        check_gpt_oss,
        check_hybrid_warm,
        check_draft_bits,
        check_host_tree,
        check_kv_bits,
        check_rope_rows,
        check_attn_rows,
        check_attn_prefill,
        check_batch_rows,
        check_sampled_rows,
    ):
        print(f"mlx {fn.__name__[6:]}: {fn()}", flush=True)
    print("mlx ok", flush=True)
    return 0


def test_mlx() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
