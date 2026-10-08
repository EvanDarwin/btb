// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The prompt's matmuls in the step's bits: y[m][r] = sum_c w[r][c] * x[m][c] over T rows of x, each output computed
// by exactly the operations the step's matvec computes it with - so a prompt prefilled on the card makes the rows a
// conversation's steps made, and the conversation's next turn read from the cache decodes as the prompt cold does.
// There are two matvecs and the warm-up picks one for the engine (`_card_mma_for`); each has its GEMM here.
//
//   btb_gemm_mma_bf16(w [R, C], x [T, C], y [T, R], R, C, T, nw)  ceil(T / 128) * ceil(R / 64) blocks, block 128
//     btb_gemv_mma_bf16's bits at `nw` warps (`_card_mma_warps(R, C)`, the step's): the super-tiles of 32 k (the
//     permuted k), cut in nw slices of ceil(nst / nw) as its warps cut them, each slice's partial an mma chain from
//     zero in super-tile order and the partials added in slice order - the matvec's warp-ordered fold. x as the mma's
//     A operand and w as its B, as there.
//   btb_gemm_f32_bf16(w [R, C], x [T, C], y [T, R], R, C, T)  ceil(T / 8) * ceil(R / 32) blocks, block 128
//     btb_gemv_bf16_m{M}'s bits: lane l's fp32 chain over chunks l, l + 32, .. of 8 elements in index order, then
//     the butterfly over the lanes (xor 16, 8, 4, 2, 1), each lane here the same lane of 64 outputs at once.
//
// The mma GEMM reuses a tile of w over 128 rows of x (the matvec reads w once a step; a prompt's chunk reads it once
// a 128 rows), staged in shared memory by asynchronous copies three super-tiles deep; a lane's fragments are one
// 16-byte shared load each, a row's 64 bytes a super-tile, two rows filling the banks once. Both launch their tiles
// in groups of x tiles (`gm_tile`): which block takes which tile moves no bit, each output its own block's.

#define GM_BM 128  // x rows a block
#define GM_BR 64   // weight rows a block
#define GM_STAGES 3
#define GM_GROUP 8      // the mma GEMM's x tiles a group: 1024 x rows
#define GM_GROUP_F32 32  // the fp32 chain's: 256 x rows

// a block's tiles (x tile, weight tile) of a 1-D launch of nx * nw blocks: a group of G x tiles walks the weight
// tiles together, so the blocks on the card at once read a stretch of the weights for every x tile of the group out
// of the L2 and the group's x rows stay there. With the weight tiles fastest, each x tile read every weight row again
// from DRAM (a 4096-row chunk's 32 x tiles: the weights 32 times over); with the x tiles fastest, the x rows again
// for each weight tile
__device__ __forceinline__ void gm_tile(int nx, int nw, int G, int& xt, int& wt) {
    const int per = G * nw;
    const int g = blockIdx.x / per, k = blockIdx.x % per;
    const int first = g * G, size = min(nx - first, G);
    xt = first + k % size;
    wt = k / size;
}

extern "C" __global__ void __launch_bounds__(128)
    btb_gemm_mma_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C,
                      int T, int nw) {
    __shared__ __align__(16) bf16 sx[GM_STAGES][GM_BM * 32];
    __shared__ __align__(16) bf16 sw[GM_STAGES][GM_BR * 32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = warp >> 1, wr = warp & 1;  // the warp's tile: x rows wm * 64 .., weight rows wr * 32 ..
    int xt, wt;
    gm_tile((T + GM_BM - 1) / GM_BM, (R + GM_BR - 1) / GM_BR, GM_GROUP, xt, wt);
    const int m0 = xt * GM_BM, r0 = wt * GM_BR;
    const int nst = (C + 31) >> 5;
    const int per = (nst + nw - 1) / nw;
    // a super-tile's x and w rows into stage buffer b: x 128 rows of 64 bytes, w 64; a chunk past C or a row past
    // the matrix is zeros, as the matvec's zero loads
    auto issue = [&](int st, int b) {
#pragma unroll
        for (int it = 0; it < GM_BM * 4 / 128; ++it) {
            const int c = it * 128 + threadIdx.x, row = c >> 2, ch = c & 3;
            const int k = (st << 5) + (ch << 3);
            const bool ok = m0 + row < T && k < C;
            cp16(&sx[b][row * 32 + ch * 8], x + (size_t)(ok ? m0 + row : 0) * C + (ok ? k : 0), ok);
        }
#pragma unroll
        for (int it = 0; it < GM_BR * 4 / 128; ++it) {
            const int c = it * 128 + threadIdx.x, row = c >> 2, ch = c & 3;
            const int k = (st << 5) + (ch << 3);
            const bool ok = r0 + row < R && k < C;
            cp16(&sw[b][row * 32 + ch * 8], w + (size_t)(ok ? r0 + row : 0) * C + (ok ? k : 0), ok);
        }
        cp_commit();
    };
    float tot[4][4][4], acc[4][4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) tot[mt][nt][e] = acc[mt][nt][e] = 0.f;
    issue(0, 0);
    if (nst > 1) {
        issue(1, 1);
    } else {
        cp_commit();
    }
    int slice = 0;
    for (int st = 0; st < nst; ++st) {
        cp_wait1();
        __syncthreads();  // super-tile st landed for every thread, and st - 1's buffer is free
        if (st + 2 < nst) {
            issue(st + 2, (st + 2) % GM_STAGES);
        } else {
            cp_commit();
        }
        const int b = st % GM_STAGES;
        uint4 wv[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            wv[nt] = *reinterpret_cast<const uint4*>(&sw[b][(wr * 32 + nt * 8 + gid) * 32 + tig * 8]);
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            const uint4 xa = *reinterpret_cast<const uint4*>(&sx[b][(wm * 64 + mt * 16 + gid) * 32 + tig * 8]);
            const uint4 xb = *reinterpret_cast<const uint4*>(&sx[b][(wm * 64 + mt * 16 + gid + 8) * 32 + tig * 8]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                float* a = acc[mt][nt];
                btb_mma16816(a[0], a[1], a[2], a[3], xa.x, xb.x, xa.y, xb.y, wv[nt].x, wv[nt].y);
                btb_mma16816(a[0], a[1], a[2], a[3], xa.z, xb.z, xa.w, xb.w, wv[nt].z, wv[nt].w);
            }
        }
        if ((st + 1) % per == 0 || st + 1 == nst) {
            // the slice's end: its partial into the total, in slice order (the first a copy, as warp 0's is kept)
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        tot[mt][nt][e] = slice == 0 ? acc[mt][nt][e] : tot[mt][nt][e] + acc[mt][nt][e];
                        acc[mt][nt][e] = 0.f;
                    }
            ++slice;
        }
    }
    // the matvec's warps past the last super-tile add their zero partials too
    for (; slice < nw; ++slice) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) tot[mt][nt][e] = tot[mt][nt][e] + 0.f;
    }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = m0 + wm * 64 + mt * 16 + gid + h * 8;
            if (m >= T) continue;
            bf16* const p = y + (size_t)m * R;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                btb_mma_st2(p, r0 + wr * 32 + nt * 8 + tig * 2, R, tot[mt][nt][2 * h], tot[mt][nt][2 * h + 1]);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// the fp32 chain's GEMM: a warp a tile of 8 x rows by 8 weight rows, lane l the matvec's lane l of all 64 (its
// chunks l, l + 32, .. of each, eight elements in index order, one fmaf a product into the output's running sum), a
// block four warps along the weight rows over the same x rows. The butterfly is the matvec's tree (each step a lane's
// sum plus its partner's, the partner's the same sum the other way round), its halves dealt out as it goes: a lane
// keeps the half of the outputs its bit of the step selects and trades the other half with its partner, so the five
// steps cost 62 shuffles for the 64 outputs, not 320, and each lane ends holding two finished sums.
// ---------------------------------------------------------------------------------------------------------
template <int N>
__device__ __forceinline__ void gm_fold(float* a, int lane) {
    // one butterfly step over N values: the lane's bit at this step chooses which half it finishes
    constexpr int H = N / 2;
    constexpr int S = N / 4;  // the step's lane distance: N = 64 is the first step (16), N = 4 the last (1)
    const bool up = (lane & S) != 0;
#pragma unroll
    for (int k = 0; k < H; ++k) {
        const float send = up ? a[k] : a[H + k];
        const float keep = up ? a[H + k] : a[k];
        const float got = __shfl_xor_sync(0xffffffffu, send, S);
        a[k] = keep + got;
    }
}

extern "C" __global__ void __launch_bounds__(128)
    btb_gemm_f32_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C,
                      int T) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int xt, wt;
    gm_tile((T + 7) / 8, (R + 31) / 32, GM_GROUP_F32, xt, wt);
    const int m0 = xt * 8, r0 = wt * 32 + warp * 8;
    const int nchunk = C >> 3;
    const uint4* xr[8];
    const uint4* wr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        xr[i] = reinterpret_cast<const uint4*>(x) + (size_t)min(m0 + i, T - 1) * nchunk;
        wr[i] = reinterpret_cast<const uint4*>(w) + (size_t)min(r0 + i, R - 1) * nchunk;
    }
    float a[64];  // a[i * 8 + j]: x row m0 + i by weight row r0 + j
#pragma unroll
    for (int e = 0; e < 64; ++e) a[e] = 0.f;
    for (int c = lane; c < nchunk; c += 32) {
        uint4 wv[8], xv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = __ldg(wr[j] + c);
#pragma unroll
        for (int i = 0; i < 8; ++i) xv[i] = __ldg(xr[i] + c);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const bf16* xp = reinterpret_cast<const bf16*>(&xv[i]);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const bf16* wp = reinterpret_cast<const bf16*>(&wv[j]);
                float s = a[i * 8 + j];
#pragma unroll
                for (int e = 0; e < 8; ++e) s = fmaf(bf2f(wp[e]), bf2f(xp[e]), s);
                a[i * 8 + j] = s;
            }
        }
    }
    gm_fold<64>(a, lane);
    gm_fold<32>(a, lane);
    gm_fold<16>(a, lane);
    gm_fold<8>(a, lane);
    gm_fold<4>(a, lane);
    // the lane's two finished sums: output index o = 2 * (its bits 4, 3, 2, 1, 0 as chosen) + {0, 1} of the 64,
    // the halves taken from the top: bit 4 chose 32 of 64, bit 3 16 of those, ..
    const int o = (((lane >> 4) & 1) << 5) | (((lane >> 3) & 1) << 4) | (((lane >> 2) & 1) << 3) |
                  (((lane >> 1) & 1) << 2) | ((lane & 1) << 1);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
        const int i = (o + u) >> 3, j = (o + u) & 7;
        if (m0 + i < T && r0 + j < R) y[(size_t)(m0 + i) * R + r0 + j] = f2bf(a[u]);
    }
}
