// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The prompt's matmuls in the step's bits: y[m][r] = sum_c w[r][c] * x[m][c] over T rows of x, each output computed
// by exactly the operations the step's matvec computes it with - so a prompt prefilled on the card makes the rows a
// conversation's steps made, and the conversation's next turn read from the cache decodes as the prompt cold does.
// There are two matvecs and the warm-up picks one for the engine (`_card_mma_for`); each has its GEMM here.
//
//   btb_gemm_mma_bf16(w [R, C], x [T, C], y [T, R], R, C, T)  ceil(T / 128) * ceil(R / 64) blocks, block 128
//     btb_gemv_mma_bf16's bits: each output one mma chain from zero over the super-tiles of 32 k (the permuted k) in
//     order, x as the mma's A operand and w as its B, as there. Four warps of 64 x rows by 32 weight rows
//   btb_gemm_mma_small_bf16(w, x, y, R, C, T)  ceil(T / 64) * ceil(R / 64) blocks, block 128
//     the same chain on tiles of 64 x 64 (four warps of 32 x 32), for a chunk too few tiles at 128 x 64 to fill the
//     card: a chain can use no more SMs than there are tiles
//   btb_gemm_f32_bf16(w [R, C], x [T, C], y [T, R], R, C, T)  ceil(T / 8) * ceil(R / 32) blocks, block 128
//     btb_gemv_bf16_m{M}'s bits: lane l's fp32 chain over chunks l, l + 32, .. of 8 elements in index order, then
//     the butterfly over the lanes (xor 16, 8, 4, 2, 1), each lane here the same lane of 64 outputs at once.
//
// The mma GEMM reuses a tile of w over the tile's x rows (the matvec reads w once a step; a prompt's chunk reads it
// once a tile), staged in shared memory by asynchronous copies a super-tile a stage; a lane's fragments are one
// 16-byte shared load each, a row's 64 bytes a super-tile, two rows filling the banks once. With nothing folded an
// output needs one accumulator, so a thread holds 64 at a 64 x 32 warp tile and four blocks stand on an SM. Both
// kernels take their tiles in groups of x tiles (`gm_tile`): which block takes which tile moves no bit.

#define GM_GROUP 8       // the mma GEMM's x tiles a group
#define GM_GROUP_F32 32  // the fp32 chain's: 256 x rows
#define GM_BIG_STAGES 2  // the 128 x 64 kernel's stages, and the blocks an SM it is built for
#define GM_BIG_BLOCKS 4
#define GM_SMALL_STAGES 3
#define GM_SMALL_BLOCKS 4

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

// a block's tile of WM x WN warps, each MT m-tiles (16 x rows) by NT n-tiles (8 weight rows), NS stages of a
// super-tile: its whole chain
template <int WM, int WN, int MT, int NT, int NS>
__device__ __forceinline__ void gm_mma(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y,
                                       int R, int C, int T, bf16* sm) {
    constexpr int NTH = 32 * WM * WN, BM = 16 * MT * WM, BR = 8 * NT * WN;
    constexpr int XS = BM * 32, WS = BR * 32, STG = XS + WS;  // a stage: the tile's x and w rows, 64 B each
    constexpr int XC = BM * 4 / NTH, WC = BR * 4 / NTH, RSTEP = NTH / 4;  // a thread's copies a stage
    static_assert(XC * NTH == BM * 4 && WC * NTH == BR * 4, "a tile's copies split evenly over its threads");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int wm = warp / WN, wr = warp % WN;
    int xt, wt;
    gm_tile((T + BM - 1) / BM, (R + BR - 1) / BR, GM_GROUP, xt, wt);
    const int m0 = xt * BM, r0 = wt * BR;
    const int nst = (C + 31) >> 5;
    // a thread's copies: the 16-byte chunk ch of rows crow + RSTEP i, their row starts found once (a row past the
    // matrix reads row 0's address, copied as zeros), a super-tile's k added to them
    const int crow = threadIdx.x >> 2, ch = threadIdx.x & 3;
    const bf16* xs[XC];
    bool xok[XC];
#pragma unroll
    for (int i = 0; i < XC; ++i) {
        const int row = crow + i * RSTEP;
        xok[i] = m0 + row < T;
        xs[i] = x + (size_t)(xok[i] ? m0 + row : 0) * C + ch * 8;
    }
    const bf16* ws[WC];
    bool wok[WC];
#pragma unroll
    for (int i = 0; i < WC; ++i) {
        const int row = crow + i * RSTEP;
        wok[i] = r0 + row < R;
        ws[i] = w + (size_t)(wok[i] ? r0 + row : 0) * C + ch * 8;
    }
    const int dst = crow * 32 + ch * 8;
    // super-tile st's rows into stage b; a chunk past C or a row past the matrix is zeros, as the matvec's
    auto issue = [&](int st, int b) {
        bf16* const base = sm + b * STG;
        const int kk = st << 5;
        const bool kin = kk + ch * 8 < C;
        const int ko = kin ? kk : 0;
#pragma unroll
        for (int i = 0; i < XC; ++i) cp16(base + dst + i * RSTEP * 32, xs[i] + ko, xok[i] && kin);
#pragma unroll
        for (int i = 0; i < WC; ++i) cp16(base + XS + dst + i * RSTEP * 32, ws[i] + ko, wok[i] && kin);
        cp_commit();
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mt][nt][e] = 0.f;
#pragma unroll
    for (int g = 0; g < NS - 1; ++g) {
        if (g < nst) {
            issue(g, g);
        } else {
            cp_commit();
        }
    }
    int rb = 0, wb = NS - 1;  // the stage read, the stage written
    for (int st = 0; st < nst; ++st) {
        cp_waitn<NS - 2>();
        __syncthreads();  // super-tile st landed for every thread, and the stage read before it is free
        if (st + NS - 1 < nst) {
            issue(st + NS - 1, wb);
        } else {
            cp_commit();
        }
        wb = wb + 1 == NS ? 0 : wb + 1;
        const bf16* const bx = sm + rb * STG;
        const bf16* const bw = bx + XS;
        rb = rb + 1 == NS ? 0 : rb + 1;
        uint4 wv[NT];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            wv[nt] = *reinterpret_cast<const uint4*>(&bw[(wr * NT * 8 + nt * 8 + gid) * 32 + tig * 8]);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const uint4 xa = *reinterpret_cast<const uint4*>(&bx[(wm * MT * 16 + mt * 16 + gid) * 32 + tig * 8]);
            const uint4 xb = *reinterpret_cast<const uint4*>(&bx[(wm * MT * 16 + mt * 16 + gid + 8) * 32 + tig * 8]);
            // the first k-tile across the n-tiles, then the second: an output's two mma in order, others between
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float* a = acc[mt][nt];
                btb_mma16816(a[0], a[1], a[2], a[3], xa.x, xb.x, xa.y, xb.y, wv[nt].x, wv[nt].y);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                float* a = acc[mt][nt];
                btb_mma16816(a[0], a[1], a[2], a[3], xa.z, xb.z, xa.w, xb.w, wv[nt].z, wv[nt].w);
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int m = m0 + wm * MT * 16 + mt * 16 + gid + h * 8;
            if (m >= T) continue;
            bf16* const p = y + (size_t)m * R;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                btb_mma_st2(p, r0 + wr * NT * 8 + nt * 8 + tig * 2, R, acc[mt][nt][2 * h], acc[mt][nt][2 * h + 1]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, GM_BIG_BLOCKS)
    btb_gemm_mma_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R, int C,
                      int T) {
    __shared__ __align__(16) bf16 sm[GM_BIG_STAGES * (128 + 64) * 32];
    gm_mma<2, 2, 4, 4, GM_BIG_STAGES>(w, x, y, R, C, T, sm);
}

extern "C" __global__ void __launch_bounds__(128, GM_SMALL_BLOCKS)
    btb_gemm_mma_small_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R,
                            int C, int T) {
    __shared__ __align__(16) bf16 sm[GM_SMALL_STAGES * (64 + 64) * 32];
    gm_mma<2, 2, 2, 4, GM_SMALL_STAGES>(w, x, y, R, C, T, sm);
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
