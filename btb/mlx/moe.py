# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A mixture's pass on Metal with its picks on the GPU: the router's logits to each row's top k experts and their
weights (`route`), the picked experts' matvecs straight from the expert store's slots (`gather`: each (row, pick) its
type's one-row chain, so a row's bits never depend on the pass), and the weighted picks summed with the shared expert
(`combine`). Their math is the row functions of `FUNCS`, which the megakernel's ops call too; each spells out what a
fast-math compile would change (`precise::exp`, `precise::divide`, no multiply-add), so the two compile alike."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from ..kinds import Quant, RouteMode, quant_key
from . import matvec
from .core import metal_kernel, mx
from .fused import SILU_MUL_FN, _kernel
from .gemv import GEMM_SRC, gemm_sgn
from .stored import STORED

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

# a route mode as `ROUTE_FN` takes it
ROUTE_CODE: dict[RouteMode, int] = {m: i for i, m in enumerate(RouteMode)}

ROUTE_FN = r"""
// row l's E logits (float32) to its K experts in ascending order (ids) and their weights (w, in the pass's dtype O).
// One simdgroup a row, lane j holding logits j, j + 32, ...: the picks in K rounds by (logit desc, index asc), equal
// logits going to the lower index. mode (`ROUTE_CODE`) 0: a softmax over every expert; 1: that renormalized over the
// picks; 2: a softmax over the picks' logits alone. L a template parameter: MLX binds an input under 8 elements as
// `constant`
template <uint E, uint K, typename L, typename O>
inline void btb_route_row(L l, device uint* ids, device O* w, uint mode, uint lane) {
    constexpr uint PER = (E + 31) / 32;
    float v[PER];
    for (uint i = 0; i < PER; i++) {
        uint e = lane + 32 * i;
        v[i] = e < E ? l[e] : -INFINITY;
    }
    uint taken = 0;
    uint pick[K];
    float pl[K];
    for (uint k = 0; k < K; k++) {
        float best = -INFINITY;
        uint bi = 0xFFFFFFFFu;
        for (uint i = 0; i < PER; i++) {
            uint e = lane + 32 * i;
            if (e < E && !((taken >> i) & 1u) && (bi == 0xFFFFFFFFu || v[i] > best)) {
                best = v[i];
                bi = e;
            }
        }
        float m = simd_max(best);
        uint e = simd_min(best == m ? bi : 0xFFFFFFFFu);
        if (e % 32 == lane) taken |= 1u << (e / 32);
        pick[k] = e;
        pl[k] = m;
    }
    float p[K];
    if (mode == 2) {
        float s = 0.0f;
        for (uint k = 0; k < K; k++) {
            p[k] = metal::precise::exp(pl[k] - pl[0]);
            s = s + p[k];
        }
        for (uint k = 0; k < K; k++) p[k] = metal::precise::divide(p[k], s);
    } else {
        float s = 0.0f;
        for (uint i = 0; i < PER; i++) {
            if (lane + 32 * i < E) s = s + metal::precise::exp(v[i] - pl[0]);
        }
        s = simd_sum(s);
        float t = 0.0f;
        for (uint k = 0; k < K; k++) {
            p[k] = metal::precise::divide(metal::precise::exp(pl[k] - pl[0]), s);
            t = t + p[k];
        }
        if (mode == 1) {
            for (uint k = 0; k < K; k++) p[k] = metal::precise::divide(p[k], t);
        }
    }
    if (lane != 0) return;
    // the picks by expert, the order they are summed in
    for (uint a = 1; a < K; a++) {
        uint ea = pick[a];
        float pa = p[a];
        uint b = a;
        for (; b > 0 && pick[b - 1] > ea; b--) {
            pick[b] = pick[b - 1];
            p[b] = p[b - 1];
        }
        pick[b] = ea;
        p[b] = pa;
    }
    for (uint k = 0; k < K; k++) {
        ids[k] = pick[k];
        w[k] = static_cast<O>(p[k]);
    }
}
"""

COMBINE_FN = r"""
// column c of row t of a mixture's output, in the pass's dtype T: its K experts' values y (H apart, in ascending
// expert order), each times its weight, summed from zero in that order; plus, where `shared`, sigmoid(g) times the
// shared expert's value. Each operation rounded to T as separate ops round it. The pointers' types template
// parameters: MLX binds an input under 8 elements (a row's weights) as `constant`
template <typename T, typename Y, typename W>
inline T btb_moe_combine(Y y, W w, uint K, uint H, bool shared, float g, T sh) {
    T acc = static_cast<T>(0.0f);
    for (uint k = 0; k < K; k++) {
        T yw = static_cast<T>(static_cast<float>(y[(size_t)k * H]) * static_cast<float>(w[k]));
        acc = static_cast<T>(static_cast<float>(acc) + static_cast<float>(yw));
    }
    if (!shared) return acc;
    float s = metal::precise::divide(1.0f, 1.0f + metal::precise::exp(-g));
    T gs = static_cast<T>(static_cast<float>(static_cast<T>(s)) * static_cast<float>(sh));
    return static_cast<T>(static_cast<float>(acc) + static_cast<float>(gs));
}
"""

FUNCS = ROUTE_FN + COMBINE_FN

_ROUTE_SRC = """
    uint row = threadgroup_position_in_grid.x;
    btb_route_row<E, K>(logits + (size_t)row * E, ids + (size_t)row * K, w + (size_t)row * K, MODE,
                        thread_index_in_simdgroup);
"""

_COMBINE_SRC = """
    // out[t, c] of y [NT, K, H], w [NT, K], g [NT] (the shared expert's gate logits) and sh [NT, H]
    uint i = thread_position_in_grid.x;
    if (i >= NT * H) return;
    uint t = i / H, c = i - t * H;
    out[i] = btb_moe_combine<T>(y + (size_t)t * K * H + c, w + (size_t)t * K, K, H, SHARED, static_cast<float>(g[t]),
                                static_cast<T>(sh[SHARED ? i : 0]));
"""

_SILU_APART_SRC = """
    // mid[i] = silu(g[i]) * u[i]: the gate and up apart (a GGUF expert's two matrices)
    uint i = thread_position_in_grid.x;
    if (i >= g_shape[0] * g_shape[1]) return;
    out[i] = btb_silu_mul<T>(static_cast<float>(g[i]), static_cast<float>(u[i]));
"""


def route(logits: mx_.array, k: int, mode: RouteMode, dtype: mx_.Dtype) -> tuple[mx_.array, mx_.array]:
    """a router's float32 logits [T, E] to each row's k experts, ascending (uint32 [T, k]), and their weights ([T, k],
    the pass's `dtype`, as the reference hands them back) by `mode`, lazily"""
    m = mx
    T, E = (int(s) for s in logits.shape)
    kern = _kernel("btb_moe_route", ["logits"], ["ids", "w"], ROUTE_FN, _ROUTE_SRC)
    ids, w = kern(
        inputs=[logits],
        template=[("E", E), ("K", int(k)), ("MODE", ROUTE_CODE[mode])],
        grid=(32 * T, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(T, int(k)), (T, int(k))],
        output_dtypes=[m.uint32, dtype],
    )
    return ids, w


def combine(y: mx_.array, w: mx_.array, g: mx_.array | None = None, sh: mx_.array | None = None) -> mx_.array:
    """a mixture's rows [T, H] in y's dtype: `y` [T, k, H] the picks' expert rows in ascending expert order, `w`
    [T, k] their weights, and the shared expert's gate logits `g` [T] and rows `sh` [T, H] where it has one, lazily"""
    T, K, H = (int(s) for s in y.shape)
    shared = g is not None and sh is not None
    kern = _kernel("btb_moe_combine", ["y", "w", "g", "sh"], ["out"], COMBINE_FN, _COMBINE_SRC)
    return kern(
        inputs=[y, w, g.reshape(-1) if g is not None else w, sh if sh is not None else y],
        template=[("T", y.dtype), ("NT", T), ("K", K), ("H", H), ("SHARED", int(shared))],
        grid=(-(-T * H // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, H)],
        output_dtypes=[y.dtype],
    )[0]


def silu_apart(g: mx_.array, u: mx_.array) -> mx_.array:
    """silu(g) * u for a gate and up apart, each element as `fused.silu_mul` computes it over the two side by side"""
    kern = _kernel("btb_moe_silu_apart", ["g", "u"], ["out"], SILU_MUL_FN, _SILU_APART_SRC)
    return kern(
        inputs=[g, u],
        template=[("T", g.dtype)],
        grid=(-(-int(g.size) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(int(s) for s in g.shape)],
        output_dtypes=[g.dtype],
    )[0]


# the pass rows a gather serves, a step's or a verify pass's; a prefill's experts each take their rows' own matmuls
ROWS_MAX = matvec.ROWS_MAX
# the store's blocks one gather binds: a launch's buffer table holds 31, beside the table, the picks, x, the output and
# a type's codebooks
BLOCKS_MAX = 24
# the streams a part of an expert may have (a stored type's `kinds.stored_stream_bytes`), and a table row's words:
# the bound block, then each part's streams' offsets
STREAMS_MAX = 3
TAB_W = 1 + 3 * STREAMS_MAX


class Gather(NamedTuple):
    """a layer's experts as a gather reads them: the store's blocks it binds (`BLOCKS_MAX` at most), and a row of
    `tab` (uint32 [E, TAB_W]) an expert: the bound block it lies in, then each part's streams' byte offsets in that
    block, `STREAMS_MAX` words a part (a stored type's gate, up and down; a bf16 one's gate_up and down)"""

    blocks: tuple[mx_.array, ...]
    tab: mx_.array


# the float types a gather multiplies through the step's float32 matvec (`gemv`): bf16 slots, a GGUF's float experts
FLOATS: frozenset[Quant | None] = frozenset({None, Quant.BF16, Quant.F16})


def _bind(nb: int, part: int, item: str, streams: Sequence[tuple[str, str]]) -> str:
    """item `item`'s expert's part `part`: its table row, the bound block it lies in, and each of its `streams` (name,
    element type) declared where it lies there"""
    cases = "".join(f"        case {b}: base = B{b}; break;\n" for b in range(1, nb))
    decl = "".join(
        f"    const device {t}* {n} = (const device {t}*)(base + te[{1 + STREAMS_MAX * part + j}]);\n"
        for j, (n, t) in enumerate(streams)
    )
    return (
        f"    const device uint* te = tab + (size_t)ids[{item}] * {TAB_W};\n"
        "    const device uint8_t* base = B0;\n"
        f"    switch (te[0]) {{\n{cases}        default: break;\n    }}\n" + decl
    )


_GEMV_HEAD = """
    uint item = threadgroup_position_in_grid.y;
BIND
    const device uint16_t* w = (const device uint16_t*)W;
    const device XT* x = xin + (size_t)(item / KP) * COLS;
    device YT* y = yout + (size_t)item * ROWS;
    const int w_shape[2] = {ROWS, COLS};
"""

_lock = threading.Lock()
_kernels: dict[tuple[object, ...], MetalKernel] = {}


def _compiled(key: tuple[object, ...], name: str, inputs: Sequence[str], outputs: Sequence[str], header: str,
              source: str) -> MetalKernel:  # fmt: skip
    k = _kernels.get(key)
    if k is None:
        with _lock:
            k = _kernels.get(key)
            if k is None:
                k = _kernels[key] = metal_kernel(
                    name=name, input_names=inputs, output_names=outputs, header=header, source=source
                )
    return k


def gather(
    q: Quant | None, g: Gather, ids: mx_.array, x: mx_.array, rows: int, cols: int, part: int, kp: int
) -> mx_.array:
    """y [items, rows] in x's dtype: item i the matvec of x's row i // kp (kp a row's picks; 1 where x holds each
    item's own row) through expert ids[i]'s part `part` [rows, cols] where its streams lie in `g`'s blocks, its type's
    one-row chain (`stored.STORED`, or the step's float32 matvec for a float type), lazily"""
    items, nb = int(ids.size), len(g.blocks)
    if q in FLOATS:
        return _gather_float(q, g, ids, x, rows, cols, part, kp)
    assert q is not None
    k = STORED[q]
    assert len(k.streams) <= STREAMS_MAX, f"{q} reads {len(k.streams)} streams, past a table row's"
    rp = (k.shapes or matvec.SHAPES).get(1, (4, 1))[0]
    shapes = {items: (rp, 1)}  # an item a simdgroup's pass row: its own expert's weights
    key = ("stored", q, rows, cols, items, nb, part, kp)
    d = matvec.defines(rows, cols, k.bw, items, shapes)
    if kp > 1:
        d += f"#undef XROW\n#define XROW(t) (x + (size_t)((t) / {kp}) * COLS)\n"
    kern = _compiled(
        key,
        f"btb_moe_gather_{quant_key(q)}_{rows}_{cols}_{items}_{nb}_{part}_{kp}",
        [*(f"B{b}" for b in range(nb)), "tab", "ids", *(t.name for t in k.tables), "x"],
        ["out"],
        k.defs + k.header + d,
        matvec.blocks(k.frags, _bind(nb, part, "tr0", k.streams)),
    )
    return kern(
        inputs=[*g.blocks, g.tab, ids.reshape(-1), *(t.make() for t in k.tables), x],
        grid=(-(-matvec.threads(rows, items, shapes) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(items, rows)],
        output_dtypes=[x.dtype],
        template=[("T", x.dtype)],
    )[0]


def _gather_float(
    q: Quant | None, g: Gather, ids: mx_.array, x: mx_.array, rows: int, cols: int, part: int, kp: int
) -> mx_.array:
    """`gather` over float experts: the step's float32 matvec (gemv.py `GEMM_SRC`) an item a launch's threadgroup row,
    as `gemv` takes x: bf16 tiles over a bf16 weight and a bf16 x, float32 tiles over a float32 x (the weight widened at
    the stage), an f16 weight's over x widened to float32 and y rounded back to x's dtype"""
    m = mx
    items, nb = int(ids.size), len(g.blocks)
    f16 = q is Quant.F16
    xb = x.dtype == m.bfloat16 and not f16
    xin = x.view(m.uint16) if xb else x.astype(m.float32)
    sgn = gemm_sgn(1, xb)
    xt = "uint16_t" if xb else "float"
    head = _GEMV_HEAD.replace("BIND", _bind(nb, part, "item", (("W", "uint8_t"),))).replace("XT", xt).replace("YT", xt)
    defs = (
        f"#include <metal_simdgroup_matrix>\n#define NB 1\n#define SGN {sgn}\n#define XB {int(xb)}\n"
        f"#define WF16 {int(f16)}\n#define WF32 0\n#define ROWS {rows}\n#define COLS {cols}\n#define KP {kp}\n"
    )
    kern = _compiled(
        ("float", f16, xb, rows, cols, items, nb, part, kp),
        f"btb_moe_gather_{'f16' if f16 else 'bf16'}_x{int(xb)}_{rows}_{cols}_{items}_{nb}_{part}_{kp}",
        [*(f"B{b}" for b in range(nb)), "tab", "ids", "xin"],
        ["yout"],
        defs,
        head + GEMM_SRC,
    )
    y = kern(
        inputs=[*g.blocks, g.tab, ids.reshape(-1), xin],
        grid=(-(-rows // (8 * sgn)) * 32 * sgn, items, 1),
        threadgroup=(32 * sgn, 1, 1),
        output_shapes=[(items, rows)],
        output_dtypes=[m.uint16 if xb else m.float32],
    )[0]
    return y.view(m.bfloat16) if xb else y.astype(x.dtype)
