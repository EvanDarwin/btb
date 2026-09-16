# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The megakernel: a dense decode or verify pass of up to 16 rows as one Metal dispatch. A persistent grid of
threadgroups walks an instruction table (op, five buffer references, dims) with a grid barrier per instruction;
the ops are the fused path's kernels as device functions; every scratch location is written once a pass, read
after its barrier (what the hardware makes coherent inside a dispatch). Bit-exact with the fused path."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from .core import mx

OPS = {
    "rmsnorm": 1,
    "matvec": 2,
    "silu": 3,
    "add_rmsnorm": 4,
    "embed": 5,
    "qkrope": 6,
    "attn": 7,
    "fold": 8,
    "argmax": 9,
    "argmax_part": 12,
    "fnorm": 10,
    "head": 11,
}


GRID = 36  # threadgroups a dispatch: every one resident, or the grid barrier never clears
KT = 64  # the matvec's columns a step (the stage is 8 simdgroups x (8 + rows) rows x KT bf16, 32 KB at most)


ROWS_MAX = 16


_SRC = r"""
    // buffers by id: 0..3 the pool blocks, 4 the scratch (written in place), 5 the constants, 6 the K/V arena
    // (written in place), 7 freqs, 8 ids, 9 pos, 10 meta, 11 path, 12 the argmax out
    BUFFER_TABLE
    uint g = threadgroup_position_in_grid.x;
    uint G = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint N = tab_shape[0];
    device atomic_uint* cnt = (device atomic_uint*)counters;
    threadgroup uint8_t tgm[TGM_BYTES];
    const uint NB = dims[0];      // rows of the tile (<= XROWS)
    const uint CAPR = dims[2];    // the arena's rows a head
    for (uint ins = 0; ins < N; ++ins) {
        const device uint* f = tab + ins * 16;
        uint op = f[0], items = f[1];
        const device uint8_t* A = B[f[2]] + f[3];
        const device uint8_t* Bp = B[f[4]] + f[5];
        device uint8_t* C = (device uint8_t*)(B[f[6]] + f[7]);
        device uint8_t* Dp = (device uint8_t*)(B[f[8]] + f[9]);
        const device uint8_t* E = B[f[10]] + f[11];
        uint n0 = f[12], n1 = f[13], n2 = f[14];
        float f0 = as_type<float>(f[15]);
        for (uint item = g; item < items; item += G) {
            if (op == 1) {

                // rmsnorm: row `item` of x [T, H] (A) with weight w [H] (Bp) -> y (C); H = n0, eps = f0
                uint H = n0;
                const device bfloat* xr = (const device bfloat*)A + (size_t)item * H;
                const device bfloat* w = (const device bfloat*)Bp;
                device bfloat* yr = (device bfloat*)C + (size_t)item * H;
                threadgroup float* part = (threadgroup float*)tgm;
                float ss = 0.0f;
                for (uint i = tid; i < H; i += 256) { float v = (float)xr[i]; ss = fma(v, v, ss); }
                ss = simd_sum(ss);
                if (lane == 0) part[sg] = ss;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = 0.0f;
                for (uint q = 0; q < 8; ++q) tot += part[q];
                float scale = metal::rsqrt(tot / (float)H + f0);
                // MLX's rms_norm rounds x * scale to the dtype before the weight
                for (uint i = tid; i < H; i += 256) yr[i] = (bfloat)((float)(bfloat)((float)xr[i] * scale) * (float)w[i]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (op == 2) {

                // matvec: y [T, rows] (C) = x [T, cols] (Bp) . W [rows, cols]^T (A); rows = n0, cols = n1, T = n2;
                // item -> 64 rows, a simdgroup 8 of them (the tile kernel's body, NB a runtime value)
                const uint KT = KT_VALUE;
                uint Nr = n0, K = n1;
                uint ntile = (Nr + 7) / 8;
                uint t0 = (uint)(((ulong)item * ntile) / items), t1 = (uint)(((ulong)(item + 1) * ntile) / items);
                for (uint tile = t0 + sg; tile < t1; tile += 8) {
                uint nb0 = tile * 8;
                {
                    const device uint16_t* w = (const device uint16_t*)A;
                    const device uint16_t* x = (const device uint16_t*)Bp;
                    device uint16_t* y = (device uint16_t*)C;
                    threadgroup bfloat* ws = (threadgroup bfloat*)tgm + sg * (8 * KT);
                    threadgroup bfloat* xs = (threadgroup bfloat*)tgm + 8 * (8 * KT) + sg * (XROWS_VALUE * KT);
                    const uint PL = KT / 8;
                    uint wr0 = lane / PL, wr1 = (32 / PL) + lane / PL, wc = (lane % PL) * 8;
                    const device uint4* wp0 = (const device uint4*)(w + (size_t)min(nb0 + wr0, Nr - 1) * K + wc);
                    const device uint4* wp1 = (const device uint4*)(w + (size_t)min(nb0 + wr1, Nr - 1) * K + wc);
                    bool ok0 = nb0 + wr0 < Nr, ok1 = (32 / PL) < 8 && nb0 + wr1 < Nr;
                    const uint XR = XROWS_VALUE * PL / 32;   // x pieces a lane: rows x PL columns over 32 lanes
                    uint xr[XR]; bool xok[XR]; const device uint4* xp[XR];
                    for (uint i = 0; i < XR; ++i) {
                        xr[i] = lane / PL + (32 / PL) * i;
                        xok[i] = xr[i] < NB;
                        xp[i] = (const device uint4*)(x + (size_t)min(xr[i], NB - 1) * K + wc);
                    }
                    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
                    uint nstep = (K + KT - 1) / KT;
                    // a step's loads are issued one step ahead, into registers (deeper measured slower)
                    uint4 w0v = (0 < nstep && ok0 && wc < K) ? wp0[0] : uint4(0u);
                    uint4 w1v = (0 < nstep && ok1 && wc < K) ? wp1[0] : uint4(0u);
                    uint4 xv[XR];
                    for (uint i = 0; i < XR; ++i) xv[i] = (0 < nstep && xok[i] && wc < K) ? xp[i][0] : uint4(0u);
                    for (uint st = 0; st < nstep; ++st) {
                        *((threadgroup uint4*)(ws + wr0 * KT + wc)) = w0v;
                        if ((32 / PL) < 8) *((threadgroup uint4*)(ws + wr1 * KT + wc)) = w1v;
                        for (uint i = 0; i < XR; ++i) *((threadgroup uint4*)(xs + xr[i] * KT + wc)) = xv[i];
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        uint sn = st + 1;
                        uint kn = sn * KT;
                        bool more = sn < nstep;
                        w0v = (more && ok0 && kn + wc < K) ? wp0[sn * (KT / 8)] : uint4(0u);
                        w1v = (more && ok1 && kn + wc < K) ? wp1[sn * (KT / 8)] : uint4(0u);
                        for (uint i = 0; i < XR; ++i) xv[i] = (more && xok[i] && kn + wc < K) ? xp[i][sn * (KT / 8)] : uint4(0u);
                        for (uint ks = 0; ks < KT / 8; ++ks) {
                            simdgroup_bfloat8x8 A0, A1, Bt;
                            simdgroup_load(Bt, ws + ks * 8, KT, ulong2(0, 0), true);
                            simdgroup_load(A0, xs + ks * 8, KT);
                            simdgroup_multiply_accumulate(acc0, A0, Bt, acc0);
                            if (XROWS_VALUE > 8) {
                                simdgroup_load(A1, xs + 8 * KT + ks * 8, KT);
                                simdgroup_multiply_accumulate(acc1, A1, Bt, acc1);
                            }
                        }
                        // no barrier after the fragments are read: the next step's stores follow them in the
                        // simdgroup's own program order (a second barrier measured 1-2% slower; none at all the same)
                    }
                    threadgroup float* out = (threadgroup float*)xs;
                    simdgroup_store(acc0, out, 8);
                    if (XROWS_VALUE > 8) simdgroup_store(acc1, out + 64, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint o = lane; o < NB * 8; o += 32) {
                        uint b = o / 8, n = o % 8;
                        if (nb0 + n < Nr) {
                            if (n2 & 1u) {
                                ((device float*)C)[(size_t)b * Nr + nb0 + n] = out[o];
                            } else {
                                uint u = as_type<uint>(out[o]);
                                u += 0x7FFFu + ((u >> 16) & 1u);
                                y[(size_t)b * Nr + nb0 + n] = (uint16_t)(u >> 16);
                            }
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                }
            } else if (op == 3) {

                // silu: mid [T, I] (C) = silu(gu[:, :I]) * gu[:, I:] (A); I = n0, T = n1; item -> 256 elements
                uint I = n0;
                uint e = item * 256 + tid;
                if (e < n1 * I) {
                    uint b = e / I, c = e - b * I;
                    const device bfloat* r = (const device bfloat*)A + (size_t)b * 2 * I;
                    float gv = (float)r[c], u = (float)r[I + c];
                    float s = (float)(bfloat)(1.0f / (1.0f + metal::exp(-gv)));
                    float si = (float)(bfloat)(gv * s);
                    ((device bfloat*)C)[e] = (bfloat)(si * u);
                }
            } else if (op == 4) {

                // add + rmsnorm on row `item`: h2 (C) = h (A) + y (Bp); x (Dp) = rmsnorm(h2) * w (E); H = n0, eps = f0
                uint H = n0;
                const device bfloat* hr = (const device bfloat*)A + (size_t)item * H;
                const device bfloat* yr = (const device bfloat*)Bp + (size_t)item * H;
                device bfloat* h2r = (device bfloat*)C + (size_t)item * H;
                device bfloat* xr = (device bfloat*)Dp + (size_t)item * H;
                const device bfloat* w = (const device bfloat*)E;
                threadgroup float* part = (threadgroup float*)tgm;
                float ss = 0.0f;
                for (uint i = tid; i < H; i += 256) {
                    float v = (float)hr[i] + (float)yr[i];
                    bfloat vt = (bfloat)v;
                    h2r[i] = vt;
                    float vf = (float)vt;
                    ss = fma(vf, vf, ss);
                }
                ss = simd_sum(ss);
                if (lane == 0) part[sg] = ss;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = 0.0f;
                for (uint q = 0; q < 8; ++q) tot += part[q];
                float scale = metal::rsqrt(tot / (float)H + f0);
                for (uint i = tid; i < H; i += 256) xr[i] = (bfloat)((float)h2r[i] * scale * (float)w[i]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (op == 5) {
                // embed: row `item` of hm (C) = the table (A, [V, H]) row ids[item]; H = n0
                uint H = n0;
                uint id = ids[item];
                const device bfloat* src = (const device bfloat*)A + (size_t)id * H;
                device bfloat* dst = (device bfloat*)C + (size_t)item * H;
                for (uint i = tid; i < H; i += 256) dst[i] = src[i];
            } else if (op == 6) {
                // q/k norm + rope and the append: item -> (t, head hh) over Hq q heads, Hk k heads, Hk v heads of
                // qkv (A, [T, (Hq + 2Hk) * D]); q -> C [T, Hq, D]; k, v -> the arena rows (Dp = K base, E = V base of
                // this layer, [Hk, cap, D]) at row n0 + t; Hq = n1, Hk = n2 & 0xffff, D = n2 >> 16, eps = f0; the norm
                // weights qn / kn in the constants at f[10..11]... (E holds V; the weights ride behind pos: see wq / wk)
                uint Hq = n1, Hk = n2 & 0xffffu, D = n2 >> 16;
                uint layer = n0 >> 20;      // the instruction's n0 carries the layer above the row (never modified: the item loop reuses it)
                uint row0 = n0 & 0xfffffu;
                uint per = Hq + 2 * Hk;
                uint t = item / per, hh = item - t * per;
                uint hd2 = D / 2;
                const device bfloat* src = (const device bfloat*)A + ((size_t)t * per + hh) * D;
                if (hh >= Hq + Hk) {
                    // v: copied into the arena
                    device bfloat* dst = (device bfloat*)E + ((size_t)(hh - Hq - Hk) * CAPR + row0 + t) * D;
                    for (uint i = tid; i < D; i += 256) dst[i] = src[i];
                } else {
                    bool isq = hh < Hq;
                    const device bfloat* wv = (const device bfloat*)(consts + (isq ? QN_OFF : KN_OFF));
                    device bfloat* dst = isq ? ((device bfloat*)C + ((size_t)t * Hq + hh) * D)
                                             : ((device bfloat*)Dp + ((size_t)(hh - Hq) * CAPR + row0 + t) * D);
                    threadgroup float* part = (threadgroup float*)tgm;
                    threadgroup float* normed = (threadgroup float*)tgm + 8;
                    uint i = tid;
                    float v = i < D ? (float)src[i] : 0.0f;
                    float ss = simd_sum(v * v);
                    if (lane == 0) part[sg] = ss;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float tot = 0.0f;
                    uint nsg = (D + 31) / 32;
                    for (uint q = 0; q < nsg; ++q) tot += part[q];
                    float scale = metal::rsqrt(tot / (float)D + f0);
                    if (i < D) normed[i] = (float)(bfloat)(v * scale * (float)wv[i]);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (i < D) {
                        if (i < hd2) {
                            float inv_freq = 1.0f / freqs[i];
                            float Lp = 1.0f * (float)(pos[t]);
                            float theta = Lp * inv_freq;
                            float costheta = metal::fast::cos(theta);
                            float sintheta = metal::fast::sin(theta);
                            float x1 = normed[i];
                            float x2 = normed[i + hd2];
                            dst[i] = (bfloat)(x1 * costheta - x2 * sintheta);
                            dst[i + hd2] = (bfloat)(x1 * sintheta + x2 * costheta);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else if (op == 7) {
                // node attention: item -> (kv head h, block s, node); q (A, [T, Hq, D] bf16), K / V (Bp, E arena bases
                // of this layer), meta / path buffers; partials pm / pl / pacc in the scratch at C (pm), Dp (pl),
                // and pacc at C + PACC_OFF... : n0 = Hk, n1 = splits, n2 = T, f0 = scale
                const uint G_ = ATT_G;
                const uint D = ATT_D;
                const uint P = ATTN_P;
                const uint DL = D / 32;
                const uint KC = D / 8;
                uint Hk = n0, ns_max = n1, Tn = n2;
                uint h = item % Hk;
                uint s = (item / Hk) % ns_max;
                uint node = item / (Hk * ns_max);
                if (node < Tn) {
                    uint Hq = Hk * G_;
                    uint nrows = meta[node * 8], nb = meta[node * 8 + 1], past = meta[node * 8 + 2], start = meta[node * 8 + 3];
                    uint PD = dims[1];
                    uint lo = 0, hi = 0;
                    if (s < nb) {
                        uint origin = (start / P + s) * P;
                        lo = max(start, origin);
                        hi = min(nrows, origin + P);
                    }
                    bool has = lo < hi;
                    const device uint16_t* kb16 = (const device uint16_t*)Bp + (size_t)h * CAPR * D;
                    const device uint16_t* vb16 = (const device uint16_t*)E + (size_t)h * CAPR * D;
                    device float* pm = (device float*)C;
                    device float* pl = (device float*)Dp;
                    device float* pacc = (device float*)(C + PACC_OFF);
                    threadgroup float4* qs = (threadgroup float4*)tgm;   // [G_][D / 4]
                    for (uint e = tid; e < G_ * (D / 4); e += 256) {
                        uint i = e / (D / 4), c = e - i * (D / 4);
                        const device bfloat* qi = (const device bfloat*)A + ((size_t)node * Hq + h * G_ + i) * D + c * 4;
                        qs[e] = float4((float)qi[0], (float)qi[1], (float)qi[2], (float)qi[3]) * f0;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    float m[G_]; float l[G_]; float acc[G_][DL];
                    for (uint i = 0; i < G_; ++i) { m[i] = -INFINITY; l[i] = 0.0f; for (uint w = 0; w < DL; ++w) acc[i][w] = 0.0f; }
                    if (has) {
                        for (uint b0 = (lo & ~31u) + sg * 32; b0 < hi; b0 += 256) {
                            uint j = b0 + lane;
                            bool live = j >= lo && j < hi;
                            float sd[G_];
                            for (uint i = 0; i < G_; ++i) sd[i] = live ? 0.0f : -INFINITY;
                            if (live) {
                                uint pj = j < past ? j : past + path[(size_t)node * PD + (j - past)];
                                const device uint4* kr = (const device uint4*)(kb16 + (size_t)pj * D);
                                #pragma clang loop unroll(disable)
                                for (uint c = 0; c < KC; ++c) {
                                    uint4 pk = kr[c];
                                    float k0 = as_type<float>(pk.x << 16), k1 = as_type<float>(pk.x & 0xffff0000u);
                                    float k2_ = as_type<float>(pk.y << 16), k3 = as_type<float>(pk.y & 0xffff0000u);
                                    float k4 = as_type<float>(pk.z << 16), k5 = as_type<float>(pk.z & 0xffff0000u);
                                    float k6 = as_type<float>(pk.w << 16), k7 = as_type<float>(pk.w & 0xffff0000u);
                                    for (uint i = 0; i < G_; ++i) {
                                        float4 qa = qs[i * (D / 4) + 2 * c];
                                        float4 qb = qs[i * (D / 4) + 2 * c + 1];
                                        sd[i] += qa.x * k0 + qa.y * k1 + qa.z * k2_ + qa.w * k3 + qb.x * k4 + qb.y * k5 + qb.z * k6 + qb.w * k7;
                                    }
                                }
                            }
                            for (uint i = 0; i < G_; ++i) {
                                float bm = simd_max(sd[i]);
                                if (bm == -INFINITY) { sd[i] = 0.0f; continue; }
                                float mn = max(m[i], bm);
                                float corr = exp(m[i] - mn);
                                float p = live ? exp(sd[i] - mn) : 0.0f;
                                l[i] = l[i] * corr + simd_sum(p);
                                for (uint w = 0; w < DL; ++w) acc[i][w] *= corr;
                                m[i] = mn;
                                sd[i] = p;
                            }
                            uint nrow = min(32u, hi - b0);
                            #pragma clang loop unroll(disable)
                            for (uint r = 0; r < nrow; ++r) {
                                uint rr = b0 + r;
                                if (rr < lo) continue;
                                uint pr = rr < past ? rr : past + path[(size_t)node * PD + (rr - past)];
                                const device uint* vw = (const device uint*)(vb16 + (size_t)pr * D + lane * DL);
                                float vv[DL];
                                for (uint w = 0; w < DL / 2; ++w) {
                                    uint x = vw[w];
                                    vv[2 * w] = as_type<float>(x << 16);
                                    vv[2 * w + 1] = as_type<float>(x & 0xffff0000u);
                                }
                                for (uint i = 0; i < G_; ++i) {
                                    float p = simd_shuffle(sd[i], r);
                                    for (uint w = 0; w < DL; ++w) acc[i][w] += p * vv[w];
                                }
                            }
                        }
                    }
                    size_t pb = ((((size_t)node * Hk + h) * ns_max + s) * 8 + sg) * G_;
                    for (uint i = 0; i < G_; ++i) {
                        size_t p = pb + i;
                        if (lane == 0) { pm[p] = m[i]; pl[p] = l[i]; }
                        device float* po = pacc + p * D + lane * DL;
                        for (uint w = 0; w < DL; ++w) po[w] = acc[i][w];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            } else if (op == 8) {
                // the fold: a simdgroup per (node, query head): item * 8 + sg; pm (A), pl (Bp), pacc (A + PACC_OFF) ->
                // out (C, [T, Hq * D] bf16); n0 = Hk, n1 = splits, n2 = T
                const uint G_ = ATT_G;
                const uint D = ATT_D;
                const uint DL = D / 32;
                uint Hk = n0, NP = n1 * 8, Tn = n2;
                uint j = item * 8 + sg;
                if (j < Tn * Hk * G_) {
                    uint node = j / (Hk * G_);
                    uint hq = j - node * Hk * G_;
                    uint h = hq / G_;
                    uint i = hq - h * G_;
                    const device float* pm = (const device float*)A;
                    const device float* pl = (const device float*)Bp;
                    const device float* pacc = (const device float*)(A + PACC_OFF);
                    uint np_ = meta[node * 8 + 1] * 8;
                    size_t base = ((size_t)node * Hk + h) * NP;
                    float M = -INFINITY;
                    for (uint s = 0; s < np_; ++s) M = max(M, pm[(base + s) * G_ + i]);
                    float Lsum = 0.0f;
                    float o[DL];
                    for (uint t = 0; t < DL; ++t) o[t] = 0.0f;
                    for (uint s = 0; s < np_; ++s) {
                        size_t p = (base + s) * G_ + i;
                        float w = exp(pm[p] - M);
                        Lsum += w * pl[p];
                        const device float* pa = pacc + p * D + lane * DL;
                        for (uint t = 0; t < DL; ++t) o[t] += w * pa[t];
                    }
                    float inv = 1.0f / Lsum;
                    device bfloat* outp = (device bfloat*)C + (size_t)j * D + lane * DL;
                    for (uint t = 0; t < DL; ++t) outp[t] = (bfloat)(o[t] * inv);
                }
            } else if (op == 12) {
                // argmax partials: item -> (row t = item / P, part p): the best of the logits (A, [T, V] float32) in
                // part p of row t, its value and index into C ([T, P] float, [T, P] uint behind it); V = n0, P = n1
                uint V = n0, P = n1;
                uint t = item / P, p = item - t * P;
                uint lo = (uint)(((ulong)p * V) / P), hi = (uint)(((ulong)(p + 1) * V) / P);
                const device float* row = (const device float*)A + (size_t)t * V;
                threadgroup float* bv = (threadgroup float*)tgm;
                threadgroup uint* bi = (threadgroup uint*)tgm + 256;
                float best = -INFINITY; uint bidx = lo;
                for (uint i = lo + tid; i < hi; i += 256) { float v = row[i]; if (v > best) { best = v; bidx = i; } }
                bv[tid] = best; bi[tid] = bidx;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    float b = bv[0]; uint bi0 = bi[0];
                    for (uint k = 1; k < 256; ++k) if (bv[k] > b || (bv[k] == b && bi[k] < bi0)) { b = bv[k]; bi0 = bi[k]; }
                    ((device float*)C)[item] = b;
                    ((device uint*)C)[n2 + item] = bi0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (op == 9) {
                // argmax of row `item`: the partials (A: [T, P] float, [T, P] uint at n2) folded in part order,
                // the first of equal values winning (the lowest index); P = n1
                uint P = n1;
                if (tid == 0) {
                    const device float* pv = (const device float*)A + (size_t)item * P;
                    const device uint* pi = (const device uint*)A + n2 + (size_t)item * P;
                    float b = pv[0]; uint bi0 = pi[0];
                    for (uint p = 1; p < P; ++p) if (pv[p] > b) { b = pv[p]; bi0 = pi[p]; }
                    out[item] = bi0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (op == 10) {
                // the final norm in float32: x (C, [T, H] float32) = rmsnorm(h (A, bf16)) * w (Bp, float32); H = n0
                uint H = n0;
                const device bfloat* hr = (const device bfloat*)A + (size_t)item * H;
                const device float* w = (const device float*)Bp;
                device float* xr = (device float*)C + (size_t)item * H;
                threadgroup float* part = (threadgroup float*)tgm;
                float ss = 0.0f;
                for (uint i = tid; i < H; i += 256) { float v = (float)hr[i]; ss = fma(v, v, ss); }
                ss = simd_sum(ss);
                if (lane == 0) part[sg] = ss;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float tot = 0.0f;
                for (uint q = 0; q < 8; ++q) tot += part[q];
                float scale = metal::rsqrt(tot / (float)H + f0);
                for (uint i = tid; i < H; i += 256) xr[i] = (float)hr[i] * scale * w[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else if (op == 11) {
                // the head: y [T, rows] float32 (C) = x [T, cols] float32 (Bp) . W [rows, cols]^T (A) as float 8x8
                // tiles (the weight widened at the stage, K steps of 32: the tile kernel's float path, the same chain)
                const uint KT = 32;
                uint Nr = n0, K = n1;
                uint ntile = (Nr + 7) / 8;
                uint t0 = (uint)(((ulong)item * ntile) / items), t1 = (uint)(((ulong)(item + 1) * ntile) / items);
                for (uint tile = t0 + sg; tile < t1; tile += 8) {
                uint nb0 = tile * 8;
                {
                    const device uint16_t* w = (const device uint16_t*)A;
                    const device float* x = (const device float*)Bp;
                    threadgroup float* ws = (threadgroup float*)tgm + sg * (8 * KT);
                    threadgroup float* xs = (threadgroup float*)tgm + 8 * (8 * KT) + sg * (16 * KT);
                    uint wr = lane / 4, wc = (lane % 4) * 8;
                    const device uint4* wp = (const device uint4*)(w + (size_t)min(nb0 + wr, Nr - 1) * K + wc);
                    bool okw = nb0 + wr < Nr;
                    const uint XR = XROWS_VALUE / 4;
                    uint xr[4]; bool xok[4]; const device uint4* xp[4];
                    for (uint i = 0; i < XR; ++i) {
                        uint piece = lane + 32 * i;
                        xr[i] = piece / 8;
                        uint xc = (piece % 8) * 4;
                        xok[i] = xr[i] < NB;
                        xp[i] = (const device uint4*)(x + (size_t)min(xr[i], NB - 1) * K + xc);
                    }
                    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);
                    uint nstep = (K + KT - 1) / KT;
                    for (uint st = 0; st < nstep; ++st) {
                        uint k0 = st * KT;
                        uint4 pk = (okw && k0 + wc < K) ? wp[st * (KT / 8)] : uint4(0u);
                        threadgroup float4* d0 = (threadgroup float4*)(ws + wr * KT + wc);
                        d0[0] = float4(as_type<float>(pk.x << 16), as_type<float>(pk.x & 0xffff0000u), as_type<float>(pk.y << 16), as_type<float>(pk.y & 0xffff0000u));
                        d0[1] = float4(as_type<float>(pk.z << 16), as_type<float>(pk.z & 0xffff0000u), as_type<float>(pk.w << 16), as_type<float>(pk.w & 0xffff0000u));
                        for (uint i = 0; i < XR; ++i) {
                            uint piece = lane + 32 * i;
                            uint xc = (piece % 8) * 4;
                            uint4 xv = (xok[i] && k0 + xc < K) ? xp[i][st * (KT / 4)] : uint4(0u);
                            *((threadgroup uint4*)(xs + xr[i] * KT + xc)) = xv;
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        for (uint ks = 0; ks < KT / 8; ++ks) {
                            simdgroup_float8x8 A0, A1, Bt;
                            simdgroup_load(Bt, ws + ks * 8, KT, ulong2(0, 0), true);
                            simdgroup_load(A0, xs + ks * 8, KT);
                            simdgroup_multiply_accumulate(acc0, A0, Bt, acc0);
                            if (XROWS_VALUE > 8) {
                                simdgroup_load(A1, xs + 8 * KT + ks * 8, KT);
                                simdgroup_multiply_accumulate(acc1, A1, Bt, acc1);
                            }
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    threadgroup float* outp = xs;
                    simdgroup_store(acc0, outp, 8);
                    if (XROWS_VALUE > 8) simdgroup_store(acc1, outp + 64, 8);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint o = lane; o < NB * 8; o += 32) {
                        uint b = o / 8, n = o % 8;
                        if (nb0 + n < Nr) ((device float*)C)[(size_t)b * Nr + nb0 + n] = outp[o];
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (tid == 0) {
            atomic_fetch_add_explicit(cnt + ins, 1u, memory_order_relaxed);
            uint spins = 0;
            while (atomic_load_explicit(cnt + ins, memory_order_relaxed) < G) {
                if (++spins > 3000000u) { atomic_fetch_add_explicit(cnt + N, 1u, memory_order_relaxed); break; }
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
"""


_lock = threading.Lock()
_kernels: dict[Any, Any] = {}


def _kernel(names: list[str], xrows: int, shape: tuple[int, ...]) -> Any:
    """the kernel for `names` weight buffers and a tile of up to `xrows` rows, compiled once per model shape:
    `shape` = (G, D, P, qn_off, kn_off, stride, pacc_off)"""
    G, D, P, qn_off, kn_off, stride, pacc_off = shape
    if 8 * (8 + xrows) * KT * 2 > 32768:
        raise ValueError(f"[mega] a stage of {xrows} rows x {KT} columns does not fit the threadgroup memory")
    key = (len(names), xrows, shape)
    with _lock:
        k = _kernels.get(key)
        if k is None:
            table = f"const device uint8_t* B[{len(names) + 3}] = {{{', '.join(names)}, scr, consts, kv}};"
            src = (
                _SRC.replace("BUFFER_TABLE", table)
                .replace("XROWS_VALUE", str(xrows))
                .replace("KT_VALUE", str(KT))
                .replace("TGM_BYTES", str(max(8 * (8 + xrows) * KT * 2, 8 * (8 * 32 + 16 * 32) * 4)))
                .replace("ATT_G", str(G))
                .replace("ATT_D", str(D))
                .replace("ATTN_P", str(P))
                .replace("PACC_OFF", str(pacc_off))
                .replace("(isq ? QN_OFF : KN_OFF)", f"(isq ? {qn_off}u : {kn_off}u) + {stride}u * layer")
            )
            k = _kernels[key] = mx().fast.metal_kernel(
                name=f"btb_mega_{len(names)}_{xrows}_{G}_{D}",
                input_names=[*names, "scr", "consts", "kv", "freqs", "ids", "pos", "meta", "path", "tab", "dims"],
                output_names=["out", "counters"],
                header="#include <metal_simdgroup_matrix>\n#define WORK 0\n",
                source=src,
            )
    return k


def fbits(x: float) -> int:
    """a float's bits as the table carries them"""
    return int(np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0])


def _align(n: int) -> int:
    return (n + 255) // 256 * 256


def mv_items(rows: int) -> int:
    """the threadgroups a matvec of `rows` takes: the fewest whose simdgroups still cover the tiles in the same
    number of rounds as the whole grid, so the last round is as full as the others (a round run by a few
    streams is latency-bound while the rest of the grid waits at the barrier)"""
    ntile = (rows + 7) // 8
    rounds = -(-ntile // (GRID * 8))
    streams = -(-ntile // rounds)
    return max(1, min(GRID, -(-streams // 8)))


class MegaPass:
    """One dense model's pass in the megakernel: the weight buffers of every layer and the head, the constants,
    a write-once scratch for `ROWS_MAX` rows, and the instruction table of a pass. Needs every projection in a
    slot weight (the pool's blocks) and the head resident on the GPU."""

    def __init__(self, sm: Any) -> None:
        import torch

        from . import attn as attn_mod

        m = mx()
        self.sm = sm
        c = sm.cfg
        self.L = int(sm.L)
        self.H = int(c.hidden_size)
        self.I = int(c.intermediate_size)
        self.Hq = int(c.num_attention_heads)
        self.Hk = int(getattr(c, "num_key_value_heads", None) or self.Hq)
        self.hd = int(sm.host[0].self_attn.head_dim)
        self.V = int(sm.head_host.mx.shape[0])
        freqs, rd, rscale = sm._mlx_rope()
        if rd != self.hd or rscale != 1.0:
            raise ValueError("[mega] a partial or scaled rotary is not in the kernel")
        if self.hd % 64 or self.H % 8 or self.I % 8:
            raise ValueError("[mega] the kernel takes a head of 64k dims and widths of 8k")
        self.freqs = freqs.astype(m.float32)
        self.scaling = float(sm.host[0].self_attn.scaling)
        # the weight buffers, by backing array
        self.bufs: list[Any] = []
        bid: dict[int, int] = {}

        def ref(w: Any) -> tuple[int, int]:
            sh = getattr(w, "sh", None)
            if sh is None or w.packed is not None:
                raise ValueError("[mega] a weight outside a slot buffer")
            i = bid.get(id(sh.mx))
            if i is None:
                i = bid[id(sh.mx)] = len(self.bufs)
                self.bufs.append(sh.mx)
            return i, int(w.off)

        self.w: list[dict[str, tuple[int, int]]] = []
        for i in range(self.L):
            at, mlp = sm.host[i].self_attn, sm.host[i].mlp
            qkv, gu = getattr(at, "_mx_qkv", None), getattr(mlp, "_mx_gu", None)
            if qkv is None or gu is None:
                raise ValueError("[mega] the q/k/v and gate/up projections are not adjacent slots")
            self.w.append({"qkv": ref(qkv), "o": ref(at.o_proj.mx), "gu": ref(gu), "down": ref(mlp.down_proj.mx)})
        self.head = ref(sm.head_host.mx)
        if sm.head_key == sm.prefix + "embed_tokens.weight":
            self.embed = self.head
        else:
            sm._mlx_embed()
            lin = sm.mlx_state.embed_lin
            if lin is None or getattr(lin.mx, "sh", None) is None:
                raise ValueError("[mega] the embedding table is not in a slot buffer")
            self.embed = ref(lin.mx)
        if len(self.bufs) > 20:
            raise ValueError(f"[mega] {len(self.bufs)} weight buffers, the kernel binds 20")
        self.names = [f"w{i}" for i in range(len(self.bufs))]
        self.nb = len(self.bufs)
        self._tabs: dict[Any, tuple[Any, list[int]]] = {}  # the kept tables by (T, splits, arena)
        self.SCR, self.CONSTS, self.KV = self.nb, self.nb + 1, self.nb + 2
        # the constants: ln1, ln2, qn, kn a layer in bf16 (one stride), the final norm in float32 after them
        parts: list[Any] = []
        self.coff: dict[Any, int] = {}
        self.eps: dict[str, float] = {}
        for i in range(self.L):
            w = sm._mlx_consts(i, sm.host[i], torch.bfloat16)
            for name in ("ln1", "ln2", "qn", "kn"):
                self.coff[(i, name)] = sum(int(a.size) for a in parts) * 2
                parts.append(w[name].astype(m.bfloat16))
            self.eps = {"ln1": float(w["eps1"]), "ln2": float(w["eps2"]), "qk": float(w["epsq"])}
        wn = sm._mlx_consts(-1, None, torch.float32)
        self.eps["final"] = float(wn["eps"])
        bf = m.concatenate(parts).view(m.uint8)
        self.fnorm_off = int(bf.size)
        self.consts = m.concatenate([bf, wn["norm"].astype(m.float32).view(m.uint8)])
        m.eval(self.consts, self.freqs)
        stride = self.coff[(1, "qn")] - self.coff[(0, "qn")] if self.L > 1 else 0
        # the scratch: a slot per layer, sized for ROWS_MAX rows; the tail after the layers
        R, H, I, Hq, Hk, hd = ROWS_MAX, self.H, self.I, self.Hq, self.Hk, self.hd
        self.G = Hq // Hk
        self.SPL = 8  # blocks of ATTN_BLOCK rows a pass may attend over (8192)
        per = [
            ("x", R * H * 2),
            ("qkv", R * (Hq + 2 * Hk) * hd * 2),
            ("q", R * Hq * hd * 2),
            ("pm", R * Hk * self.SPL * 8 * self.G * 4),
            ("pl", R * Hk * self.SPL * 8 * self.G * 4),
            ("pacc", R * Hk * self.SPL * 8 * self.G * hd * 4),
            ("a", R * Hq * hd * 2),
            ("o", R * H * 2),
            ("h2", R * H * 2),
            ("x2", R * H * 2),
            ("gu", R * 2 * I * 2),
            ("mid", R * I * 2),
            ("down", R * H * 2),
            ("h3", R * H * 2),
            ("x3", R * H * 2),
        ]
        self.offs: dict[str, int] = {}
        o = 0
        for name, nb in per:
            self.offs[name] = o
            o += _align(nb)
        self.layer_bytes = o
        self.toff: dict[str, int] = {}
        o = self.L * self.layer_bytes
        self.AMP = 64  # argmax parts a row
        for name, nb in (("hm", R * H * 2), ("hn", R * H * 4), ("logits", R * self.V * 4), ("amax", R * self.AMP * 8)):
            self.toff[name] = o
            o += _align(nb)
        self.scr = m.zeros((o,), dtype=m.uint8)
        m.eval(self.scr)
        self.shape = (
            self.G,
            hd,
            attn_mod.ATTN_BLOCK,
            self.coff[(0, "qn")],
            self.coff[(0, "kn")],
            stride,
            self.offs["pacc"] - self.offs["pm"],
        )

    def soff(self, i: int, name: str) -> tuple[int, int]:
        return self.SCR, i * self.layer_bytes + self.offs[name]

    def table(self, T: int, past: int, splits: int, kv_off: Any, argmax: bool = True) -> Any:
        """the instruction table of a pass of T rows over a cache of `past` rows: `kv_off(i, which)` the arena
        offset of layer i's K (0) or V (1); `argmax` off leaves the logits in the scratch for a sample instead.
        Built once per (T, splits, arena, argmax) and kept; `past` sits in the qk-rope rows alone and is written
        into the kept copy."""
        L = self.L
        key = (T, splits, kv_off(0, 0), kv_off(0, 1), kv_off(L - 1, 1), argmax)
        kept = self._tabs.get(key)
        if kept is None:
            kept = self._tabs[key] = self._table(T, splits, kv_off, argmax)
        base, qk = kept
        t = base.copy()
        t[qk, 12] = np.asarray([past | (i << 20) for i in range(L)], dtype=np.uint32)
        return mx().array(t)

    def _table(self, T: int, splits: int, kv_off: Any, argmax: bool = True) -> tuple[Any, list[int]]:
        """the table's rows as uint32 [n, 16] with `past` 0, and the qk-rope rows' indices"""
        rows: list[list[int]] = []
        qk_rows: list[int] = []
        past = 0
        H, I, Hq, Hk, hd, V, L = self.H, self.I, self.Hq, self.Hk, self.hd, self.V, self.L
        C, S = self.CONSTS, self.SCR

        Ref = tuple[int, int]

        def add(
            op: str,
            items: int,
            a: Ref = (0, 0),
            b: Ref = (0, 0),
            c: Ref = (0, 0),
            d: Ref = (0, 0),
            e: Ref = (0, 0),
            n0: int = 0,
            n1: int = 0,
            n2: int = 0,
            f0: float = 0.0,
        ) -> None:
            rows.append(
                [OPS[op], items, a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1], e[0], e[1], n0, n1, n2, fbits(f0)]
            )

        add("embed", T, a=self.embed, c=(S, self.toff["hm"]), n0=H)
        hsrc = (S, self.toff["hm"])
        add("rmsnorm", T, a=hsrc, b=(C, self.coff[(0, "ln1")]), c=self.soff(0, "x"), n0=H, f0=self.eps["ln1"])
        for i in range(L):
            w = self.w[i]
            add(
                "matvec",
                mv_items((Hq + 2 * Hk) * hd),
                a=w["qkv"],
                b=self.soff(i, "x"),
                c=self.soff(i, "qkv"),
                n0=(Hq + 2 * Hk) * hd,
                n1=H,
            )
            qk_rows.append(len(rows))
            add(
                "qkrope",
                T * (Hq + 2 * Hk),
                a=self.soff(i, "qkv"),
                c=self.soff(i, "q"),
                d=(self.KV, kv_off(i, 0)),
                e=(self.KV, kv_off(i, 1)),
                n0=past | (i << 20),
                n1=Hq,
                n2=Hk | (hd << 16),
                f0=self.eps["qk"],
            )
            add(
                "attn",
                Hk * splits * T,
                a=self.soff(i, "q"),
                b=(self.KV, kv_off(i, 0)),
                c=self.soff(i, "pm"),
                d=self.soff(i, "pl"),
                e=(self.KV, kv_off(i, 1)),
                n0=Hk,
                n1=splits,
                n2=T,
                f0=self.scaling,
            )
            add(
                "fold",
                (T * Hq + 7) // 8,
                a=self.soff(i, "pm"),
                b=self.soff(i, "pl"),
                c=self.soff(i, "a"),
                n0=Hk,
                n1=splits,
                n2=T,
            )
            add("matvec", mv_items(H), a=w["o"], b=self.soff(i, "a"), c=self.soff(i, "o"), n0=H, n1=Hq * hd)
            add(
                "add_rmsnorm",
                T,
                a=hsrc,
                b=self.soff(i, "o"),
                c=self.soff(i, "h2"),
                d=self.soff(i, "x2"),
                e=(C, self.coff[(i, "ln2")]),
                n0=H,
                f0=self.eps["ln2"],
            )
            add("matvec", mv_items(2 * I), a=w["gu"], b=self.soff(i, "x2"), c=self.soff(i, "gu"), n0=2 * I, n1=H)
            add("silu", (T * I + 255) // 256, a=self.soff(i, "gu"), c=self.soff(i, "mid"), n0=I, n1=T)
            add("matvec", mv_items(H), a=w["down"], b=self.soff(i, "mid"), c=self.soff(i, "down"), n0=H, n1=I)
            if i + 1 < L:
                add(
                    "add_rmsnorm",
                    T,
                    a=self.soff(i, "h2"),
                    b=self.soff(i, "down"),
                    c=self.soff(i, "h3"),
                    d=self.soff(i + 1, "x"),
                    e=(C, self.coff[(i + 1, "ln1")]),
                    n0=H,
                    f0=self.eps["ln1"],
                )
            else:
                add(
                    "add_rmsnorm",
                    T,
                    a=self.soff(i, "h2"),
                    b=self.soff(i, "down"),
                    c=self.soff(i, "h3"),
                    d=self.soff(i, "x3"),
                    e=(C, self.coff[(i, "ln2")]),
                    n0=H,
                    f0=self.eps["ln2"],
                )
            hsrc = self.soff(i, "h3")
        add(
            "fnorm",
            T,
            a=self.soff(L - 1, "h3"),
            b=(C, self.fnorm_off),
            c=(S, self.toff["hn"]),
            n0=H,
            f0=self.eps["final"],
        )
        add("head", GRID, a=self.head, b=(S, self.toff["hn"]), c=(S, self.toff["logits"]), n0=V, n1=H)
        if argmax:
            P = self.AMP
            add("argmax_part", T * P, a=(S, self.toff["logits"]), c=(S, self.toff["amax"]), n0=V, n1=P, n2=ROWS_MAX * P)
            add("argmax", T, a=(S, self.toff["amax"]), n1=P, n2=ROWS_MAX * P)
        return np.asarray(rows, dtype=np.uint32), qk_rows

    def run(
        self,
        kv: Any,
        cap: int,
        kv_off: Any,
        ids: Any,
        pos: Any,
        meta: Any,
        path: Any,
        splits: int,
        past: int,
        argmax: bool = True,
    ) -> Any:
        """the pass: `kv` the arena (uint8), `cap` its rows a head, `ids`/`pos` uint32 [T], `meta`/`path` the node
        metadata; returns (the argmax ids uint32 [T], the barrier counters), lazily; `argmax` off leaves the ids
        zero and the logits in the scratch"""
        m = mx()
        T = int(ids.shape[0])
        xrows = 8 if T <= 8 else 16
        tab = self.table(T, past, splits, kv_off, argmax)
        dims = m.array([T, int(path.shape[1]), int(cap), 0, 0, 0, 0, 0], dtype=m.uint32)
        k = _kernel(self.names, xrows, self.shape)
        return k(
            inputs=[*self.bufs, self.scr, self.consts, kv, self.freqs, ids, pos, meta, path, tab, dims],
            grid=(GRID * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(T,), (int(tab.shape[0]) + 1,)],
            output_dtypes=[m.uint32, m.uint32],
            init_value=0,
        )
