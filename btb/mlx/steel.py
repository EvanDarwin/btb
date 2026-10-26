# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Prefill-width matmuls with the matvec's chain: each output 8-wide matrix multiply-accumulates in ascending K into a
float accumulator, the chain of `gemv`'s tiles, so a row has the bits here that it has in a one-row step. MLX's own
gemm where its dispatch keeps that chain, else its steel GEMMKernel (never its split-K) at pinned tiles, a whole
tile's K steps fed by btb's prefetching loader. The headers are the installed mlx package's, inlined: Metal's runtime
compiler takes no include path."""

from __future__ import annotations

import functools
import os
import re
import threading
from typing import TYPE_CHECKING

from .core import metal_kernel, mx
from .gemv import gemv

if TYPE_CHECKING:
    import mlx.core as mx_

    from .core import MetalKernel

# (BM, BN, BK, WM, WN): an output tile of BM x BN, K staged BK at a time, WM x WN simdgroups
Tile = tuple[int, int, int, int, int]
# largest first; every tile gives the same bits, so a launch takes the fastest (`pick`). On an 18-core M3 Pro, against
# MLX's gemm over eight shapes at 8 to 4096 rows: within noise of it or up to 50% faster where MLX splits K
TILES: tuple[Tile, ...] = ((64, 64, 16, 2, 2), (64, 32, 32, 2, 2), (32, 32, 32, 1, 2))
# the threadgroups a launch needs to keep that GPU busy: 64 x 32 tiles at 80 beat 64 x 64 at 40 by 6 points
# ([2560, 9728] at 64 rows), 64 x 64 at 80 tie 64 x 32 at 160
FILL = 80
# a bf16 pass's rows from which the gemm beats the matvec: 7 to 15 points faster at 32 rows, a tie at 16 and 24
FROM_ROWS = 32

_INCLUDE = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"\s*$', re.M)
# a string literal (kept), or a comment (dropped)
_COMMENT = re.compile(r'("(?:\\.|[^"\\])*")|/\*.*?\*/|//[^\n]*', re.S)

_FETCH = """
// steel's BlockLoader split in two: `fetch` a K step's elements into registers, `stash` them where the loader puts
// them, so the next step's reads are in flight while this one's multiplies run
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct BtbFetch {
    STEEL_CONST short n_reads = (BCOLS * BROWS) / tgp_size;
    STEEL_CONST short TCOLS = BCOLS / n_reads;
    STEEL_CONST short TROWS = tgp_size / TCOLS;
    STEEL_CONST short n_rows = BROWS / TROWS;
    static_assert(n_rows > 0 && BROWS % TROWS == 0, "a tile's rows split evenly over its threads");
    struct alignas(sizeof(T)) ReadVector { uint8_t v[sizeof(T) * n_reads]; };
    const int src_ld;
    threadgroup T* dst;
    const device T* src;
    ReadVector r[n_rows];
    METAL_FUNC BtbFetch(const device T* src_, int src_ld_, threadgroup T* dst_, ushort sg, ushort lane) thread
        : src_ld(src_ld_),
          dst(dst_ + ((sg * 32 + lane) / TCOLS) * dst_ld + n_reads * ((sg * 32 + lane) % TCOLS)),
          src(src_ + ((sg * 32 + lane) / TCOLS) * src_ld_ + n_reads * ((sg * 32 + lane) % TCOLS)) {}
    METAL_FUNC void fetch() thread {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < n_rows; i++) r[i] = *((const device ReadVector*)(&src[i * TROWS * src_ld]));
    }
    METAL_FUNC void stash() const thread {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < n_rows; i++) *((threadgroup ReadVector*)(&dst[i * TROWS * dst_ld])) = r[i];
    }
    METAL_FUNC void next() thread { src += BCOLS; }
};
"""

_SRC = """
    using namespace mlx::steel;
    using gemm_t = GEMMKernel<T, T, BM_, BN_, BK_, WM_, WN_, false, true, false, KA>;
    threadgroup T As[gemm_t::tgp_mem_size_a];
    threadgroup T Bs[gemm_t::tgp_mem_size_b];
    const int M = x_shape[0], N = w_shape[0], K = w_shape[1];
    const int c_row = threadgroup_position_in_grid.y * BM_, c_col = threadgroup_position_in_grid.x * BN_;
    ushort sg = simdgroup_index_in_threadgroup, lane = thread_index_in_simdgroup;
    const device T* xa = (const device T*)x + (size_t)c_row * K;
    const device T* wb = (const device T*)w + (size_t)c_col * K;
    thread typename gemm_t::loader_a_t loader_a(xa, K, As, sg, lane);
    thread typename gemm_t::loader_b_t loader_b(wb, K, Bs, sg, lane);
    thread typename gemm_t::mma_t mma_op(sg, lane);
    device T* D = (device T*)y + (size_t)c_row * N + c_col;
    // MA / NA: every tile whole along M / N, so a launch of whole tiles compiles none of the bounded paths
    const short tgp_bm = MA ? BM_ : min(BM_, M - c_row), tgp_bn = NA ? BN_ : min(BN_, N - c_col), lbk = K % BK_;
    const int iters = K / BK_;
    if ((MA || tgp_bm == BM_) && (NA || tgp_bn == BN_)) {
        // steel's K steps and multiplies in its order, each step's loads issued before the last one's multiply
        // (2-6% faster than steel's own loop from 256 rows, within noise under it)
        BtbFetch<T, BM_, BK_, BK_ + gemm_t::tgp_padding_a, WM_ * WN_ * 32> fa(xa, K, As, sg, lane);
        BtbFetch<T, BN_, BK_, BK_ + gemm_t::tgp_padding_b, WM_ * WN_ * 32> fb(wb, K, Bs, sg, lane);
        if (iters > 0) { fa.fetch(); fb.fetch(); }
        for (int k = 0; k < iters; k++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            fa.stash();
            fb.stash();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k + 1 < iters) { fa.next(); fb.next(); fa.fetch(); fb.fetch(); }
            mma_op.mma(As, Bs);
        }
        if (!KA) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            loader_a.src += (size_t)iters * BK_;
            loader_b.src += (size_t)iters * BK_;
            loader_a.load_safe(short2(lbk, BM_));
            loader_b.load_safe(short2(lbk, BN_));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mma_op.mma(As, Bs);
        }
        mma_op.store_result(D, N);
    } else if (NA || tgp_bn == BN_) {
        gemm_t::template gemm_loop<false, true, KA>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
        mma_op.store_result_safe(D, N, short2(tgp_bn, tgp_bm));
    } else if (MA || tgp_bm == BM_) {
        gemm_t::template gemm_loop<true, false, KA>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
        mma_op.store_result_safe(D, N, short2(tgp_bn, tgp_bm));
    } else {
        gemm_t::template gemm_loop<false, false, KA>(As, Bs, iters, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
        mma_op.store_result_safe(D, N, short2(tgp_bn, tgp_bm));
    }
"""

_kernels: dict[Tile, MetalKernel] = {}
_lock = threading.Lock()


def _inline(path: str, seen: set[str]) -> str:
    if path in seen:
        return ""
    seen.add(path)
    with open(os.path.join(os.path.dirname(mx.__file__), "include", path)) as f:
        text = f.read().replace("#pragma once", "")
    return _INCLUDE.sub(lambda mt: _inline(mt.group(1), seen), text)


@functools.cache
def _header() -> str:
    # without comments or indentation: MLX copies a kernel's whole text at every launch, 55 KB of it 3.3 us of the
    # launch's 12; ends on a newline, as MLX appends the kernel's declaration straight after it
    gemm = _COMMENT.sub(lambda mt: mt.group(1) or " ", _inline("mlx/backend/metal/kernels/steel/gemm/gemm.h", set()))
    lines = (ln.strip() for ln in gemm.splitlines())
    body = "\n".join(ln for ln in lines if ln)
    return "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n" + body + _FETCH


def _kernel(tile: Tile) -> MetalKernel:
    with _lock:
        k = _kernels.get(tile)
        if k is None:
            bm, bn, bk, wm, wn = tile
            defines = "".join(f"#define {n}_ {v}\n" for n, v in zip(("BM", "BN", "BK", "WM", "WN"), tile, strict=True))
            k = _kernels[tile] = metal_kernel(
                name=f"btb_steel_{bm}_{bn}_{bk}_{wm}_{wn}",
                input_names=["w", "x"],
                output_names=["y"],
                header=_header() + defines,
                source=_SRC,
            )
    return k


def pick(b: int, rows: int) -> Tile:
    """the tile for b rows of x over `rows` outputs: the largest that pads b to within a quarter of the fewest rows
    any tile pads it to (96 rows take 32-row tiles, not 64-row ones a third empty) and still launches `FILL`
    threadgroups"""
    padded = {t: -(-b // t[0]) * t[0] for t in TILES}
    least = min(padded.values())
    for t in TILES:
        if 4 * padded[t] <= 5 * least and padded[t] // t[0] * -(-rows // t[1]) >= FILL:
            return t
    return TILES[-1]


def _launch(w: mx_.array, x: mx_.array) -> mx_.array:
    """x w^T for w and x of one dtype (bf16 or float32), y in it"""
    rows, cols, b = int(w.shape[0]), int(w.shape[1]), int(x.shape[0])
    tile = pick(b, rows)
    bm, bn, bk, wm, wn = tile
    return _kernel(tile)(
        inputs=[w, x],
        template=[("T", x.dtype), ("MA", b % bm == 0), ("NA", rows % bn == 0), ("KA", cols % bk == 0)],
        grid=(-(-rows // bn) * wm * wn * 32, -(-b // bm), 1),
        threadgroup=(wm * wn * 32, 1, 1),
        output_shapes=[(b, rows)],
        output_dtypes=[x.dtype],
    )[0]


@functools.cache
def mlx_regular() -> bool:
    """MLX's gemm runs the matvec's chain where it takes its regular kernel: one probe at such a shape, bit for bit.
    False on a GPU whose regular kernel is another (NAX tiles), and then MLX's gemm is never taken."""
    m = mx
    w = (m.random.normal((2048, 512), key=m.random.key(0)) * 0.05).astype(m.bfloat16)
    x = m.random.normal((64, 512), key=m.random.key(1)).astype(m.bfloat16)
    return bool(m.array_equal(m.matmul(x, w.T), gemv(w, x)))


def gemm(w: mx_.array, x: mx_.array) -> mx_.array:
    """y[b, rows] = x[b, cols] w[rows, cols]^T with `gemv`'s bits and dtypes: a bf16 weight and x in bf16, else
    everything widened to float32, y float32 but for an f16 weight's bf16 x, rounded back to bf16. Needs cols % 8 ==
    0, x bf16 or float32."""
    m = mx
    if w.dtype == m.bfloat16 and x.dtype == m.bfloat16:
        rows, cols, b = int(w.shape[0]), int(w.shape[1]), int(x.shape[0])
        # MLX's own gemm where it is that chain: past 15 rows (its wide matvec), K under max(M, N) (it splits K only
        # from there, MLX 0.32's matmul.cpp) in whole 32-steps (it runs a K remainder first). Its launch costs half
        # of steel's (6 us against 12), which is 2-17% at short K and 64-512 rows. Its tiles take 64 rows, so only
        # where they pad b within a quarter of 32-row ones (32 rows on a vocabulary head: +12-18% on its tiles)
        pads_well = 4 * (-(-b // 64) * 64) <= 5 * (-(-b // 32) * 32)
        if cols < max(b, rows) and cols % 32 == 0 and pads_well and mlx_regular():
            return m.matmul(x, w.T)
        return _launch(w, x)
    y = _launch(w.astype(m.float32), x.astype(m.float32))
    return y.astype(m.bfloat16) if w.dtype == m.float16 and x.dtype == m.bfloat16 else y
