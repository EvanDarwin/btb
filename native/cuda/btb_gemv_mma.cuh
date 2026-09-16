// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// The tensor-core matvec: y[m][r] = sum_c w[r][c] * x[m][c] for M <= 32 rows of x through bf16 mma.sync with
// fp32 accumulation, one kernel for every M (the caller pads x to 32 rows), so a one-row step and a 32-row
// verify pass compute row r from the same instruction sequence and agree bit for bit.
//
//   btb_gemv_mma_bf16(w [R, C], x [32, C], y [32, R], R, C, M)  grid ceil(R / 16) x 1 x 1, block 64 x 1 x 1
//   (M: the live rows of x, 1..32; the rest are zeros and are not read)
//
// The pass is y^T[R, 32] = W[R, C] x^T[C, 32] read as a GEMM with M = 32 (the x rows), N = R (the weight
// rows), K = C, run on mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32. A warp owns 16 weight rows (two
// n = 8 tiles) by all 32 x rows (two m = 16 tiles) and marches k in super-tiles of 32 (two mma k-tiles), so
// its output tile is 32 x 16 in fp32 - 16 registers a lane, the whole 32-row pass, never a per-M variant.
// The mma accumulates each output element from its own row of x alone, so the 31 zero rows a one-row step
// pads with cannot reach row 0's bits; that is checked, not assumed.
//
// The k inside a super-tile is PERMUTED, and that is what buys the 16-byte load. The fragment a lane holds
// is four bf16 of B (two pairs, eight k apart) and eight of A; laid out in k order those are 4-byte scraps.
// Numbering the super-tile's 32 k so that lane q = lane & 3 owns k 8q .. 8q + 7 makes every fragment one
// aligned uint4: B's two k-tiles are the load's (x, y) and (z, w) halves, A's the same halves of the four x
// rows the lane carries. The map k-slot -> k is one fixed bijection applied to W and to x alike, so the
// products summed are exactly the C products of the row; only their order inside a super-tile differs from
// the fp32 kernel's, which no caller can observe (the two were never bit-equal - each is bit-stable in
// itself, which is what a verify pass needs).
//
// W is read once and never twice: one __ldcs uint4 a lane a super-tile a n-tile, the quad covering 64
// contiguous bytes of a weight row - evict-first the whole way, so the stream displaces neither the cache
// nor x. Four super-tiles go as one burst (BTB_MMA_U): eight loads standing together, 256 contiguous bytes
// of each of the sixteen rows, which is the whole difference between 264 GB/s and 477 on the 0.6B head -
// one super-tile at a time leaves the two halves of a 128 B line an iteration apart, and the evict-first
// hint has dropped the line by then. x rides the normal policy (__ldg); its four rows a lane are twice the
// weight bytes, and against a variant that fabricates x in registers they cost nothing (the down
// projection: 455.4 GB/s against 456.1).
//
// A block is two warps over ONE group of 16 weight rows, k split in halves - warp 0 takes super-tiles
// 0 .. ceil(nst/2), warp 1 the rest - and the fp32 partials meet in shared memory, warp 1's added into warp
// 0's. The half-split is not about parallelism: at one warp a block (the same 16 rows, k whole) the same
// code reads 264 GB/s, at two 477. Every other split measured worse - four warps to a row group 458 GB/s on
// the down projection, eight 449, and the eight-warp block that spreads its warps over eight row groups
// 258. The layout is a function of R and C alone, never of M and never of the grid, so a shape's launches
// all give the same bits; blockIdx.x only strides the row groups (lb += gridDim.x), which makes a short
// grid slow, never wrong.

#define BTB_MMA_WARPS 2
#define BTB_MMA_ROWS 16
#define BTB_MMA_U 4

__device__ __forceinline__ void btb_mma16816(float& d0, float& d1, float& d2, float& d3, unsigned a0,
                                             unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 btb_mma_ldw(const bf16* p, bool ok) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (ok) v = __ldcs(reinterpret_cast<const uint4*>(p));
    return v;
}

__device__ __forceinline__ uint4 btb_mma_ldx(const bf16* p, bool ok) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (ok) v = __ldg(reinterpret_cast<const uint4*>(p));
    return v;
}

// one super-tile: the lane's four x rows, then the eight mma of the 32 x 16 tile over the two k-tiles. Only
// the M live rows of x are loaded - the pass's rows; the padded rows are zeros without a load. Every block
// reads the whole of x, so at one row that is 1/32 of the traffic (a 4B step read 14 GB of x from L2 with
// all 32), and the instruction sequence, hence the bits, is the same for every M.
__device__ __forceinline__ void btb_mma_step(float* acc, const bf16* xl, int C, int k0, int st, int gid,
                                             uint4 ca, uint4 cb, int M) {
    const bf16* const xp = xl + (st << 5);
    const bool okx = ((st << 5) + k0) < C;
    const uint4 x0 = btb_mma_ldx(xp + (size_t)gid * C, okx && gid < M);
    const uint4 x1 = btb_mma_ldx(xp + (size_t)(gid + 8) * C, okx && gid + 8 < M);
    const uint4 x2 = btb_mma_ldx(xp + (size_t)(gid + 16) * C, okx && gid + 16 < M);
    const uint4 x3 = btb_mma_ldx(xp + (size_t)(gid + 24) * C, okx && gid + 24 < M);
    btb_mma16816(acc[0], acc[1], acc[2], acc[3], x0.x, x1.x, x0.y, x1.y, ca.x, ca.y);
    btb_mma16816(acc[0], acc[1], acc[2], acc[3], x0.z, x1.z, x0.w, x1.w, ca.z, ca.w);
    btb_mma16816(acc[4], acc[5], acc[6], acc[7], x0.x, x1.x, x0.y, x1.y, cb.x, cb.y);
    btb_mma16816(acc[4], acc[5], acc[6], acc[7], x0.z, x1.z, x0.w, x1.w, cb.z, cb.w);
    btb_mma16816(acc[8], acc[9], acc[10], acc[11], x2.x, x3.x, x2.y, x3.y, ca.x, ca.y);
    btb_mma16816(acc[8], acc[9], acc[10], acc[11], x2.z, x3.z, x2.w, x3.w, ca.z, ca.w);
    btb_mma16816(acc[12], acc[13], acc[14], acc[15], x2.x, x3.x, x2.y, x3.y, cb.x, cb.y);
    btb_mma16816(acc[12], acc[13], acc[14], acc[15], x2.z, x3.z, x2.w, x3.w, cb.z, cb.w);
}

// the tile's two columns of one x row: adjacent r, so one 4-byte store where R keeps y's rows even
__device__ __forceinline__ void btb_mma_st2(bf16* p, int c, int R, float a, float b) {
    if ((R & 1) == 0 && c + 1 < R) {
        *reinterpret_cast<__nv_bfloat162*>(p + c) = __halves2bfloat162(f2bf(a), f2bf(b));
    } else {
        if (c < R) p[c] = f2bf(a);
        if (c + 1 < R) p[c + 1] = f2bf(b);
    }
}

extern "C" __global__ void __launch_bounds__(32 * BTB_MMA_WARPS)
    btb_gemv_mma_bf16(const bf16* __restrict__ w, const bf16* __restrict__ x, bf16* __restrict__ y, int R,
                      int C, int M) {
    const int lane = threadIdx.x & 31;
    const int slice = threadIdx.x >> 5;
    const int tig = lane & 3;
    const int gid = lane >> 2;

    const int nst = (C + 31) >> 5;
    const int nrg = (R + BTB_MMA_ROWS - 1) / BTB_MMA_ROWS;
    const int per = (nst + BTB_MMA_WARPS - 1) / BTB_MMA_WARPS;
    const int s0 = min(slice * per, nst), s1 = min(s0 + per, nst);

    __shared__ float red[256];

    const int k0 = tig << 3;
    const bf16* const xl = x + k0;

    for (int lb = blockIdx.x; lb < nrg; lb += gridDim.x) {
        const int r0 = lb * BTB_MMA_ROWS;
        const int rowa = r0 + gid, rowb = r0 + 8 + gid;
        const bool oka = rowa < R, okb = rowb < R;
        const bf16* const wa = w + (size_t)(oka ? rowa : 0) * C + k0;
        const bf16* const wb = w + (size_t)(okb ? rowb : 0) * C + k0;

        float acc[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) acc[i] = 0.f;

        int st = s0;
        for (; st + BTB_MMA_U - 1 < s1; st += BTB_MMA_U) {
            uint4 wv[2 * BTB_MMA_U];
#pragma unroll
            for (int u = 0; u < BTB_MMA_U; ++u)
                wv[u] = btb_mma_ldw(wa + ((st + u) << 5), oka && (((st + u) << 5) + k0) < C);
#pragma unroll
            for (int u = 0; u < BTB_MMA_U; ++u)
                wv[BTB_MMA_U + u] = btb_mma_ldw(wb + ((st + u) << 5), okb && (((st + u) << 5) + k0) < C);
#pragma unroll
            for (int u = 0; u < BTB_MMA_U; ++u)
                btb_mma_step(acc, xl, C, k0, st + u, gid, wv[u], wv[BTB_MMA_U + u], M);
        }
        for (; st < s1; ++st) {
            const bool okk = ((st << 5) + k0) < C;
            btb_mma_step(acc, xl, C, k0, st, gid, btb_mma_ldw(wa + (st << 5), oka && okk),
                         btb_mma_ldw(wb + (st << 5), okb && okk), M);
        }

        // the k halves meet: warp 1's partials into warp 0's, eight of the sixteen at a time
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (slice == 1) {
#pragma unroll
                for (int i = 0; i < 8; ++i) red[(i << 5) + lane] = acc[(h << 3) + i];
            }
            __syncthreads();
            if (slice == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[(h << 3) + i] += red[(i << 5) + lane];
            }
            __syncthreads();
        }

        if (slice == 0) {
            const int c0 = r0 + (tig << 1);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                bf16* const p0 = y + (size_t)((mt << 4) + gid) * R;
                bf16* const p1 = y + (size_t)((mt << 4) + gid + 8) * R;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int b = (mt << 3) + (nt << 2);
                    btb_mma_st2(p0, c0 + (nt << 3), R, acc[b], acc[b + 1]);
                    btb_mma_st2(p1, c0 + (nt << 3), R, acc[b + 2], acc[b + 3]);
                }
            }
        }
    }
}
