// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// One attention for every pass on the card - a decode step, a verify pass's chain or tree, a fork's or a batch's
// rows, a prompt's chunk - each query row computed by the same operations over its own keys whatever rows share its
// launch, so a row's bits are fixed by the row alone: a prompt prefilled makes the rows a conversation's steps made,
// and a conversation's next turn read from the cache decodes as the same prompt decoded cold does.
//
// The row: its keys are the logical positions [first, last] (first = last + 1 - win under a window, else 0), cut at
// fixed boundaries into tiles of BN keys and the tiles into groups of GT. Its arithmetic is three folds, each in a
// fixed order and none of them a chain through the keys:
//   a TILE's state from nothing - its scores on tensor cores, scaled to log2 units, masked to the row's keys, their
//     max, the weights 2^(s - max) summed and multiplied into V on tensor cores;
//   a GROUP's state, its tiles' states folded in tile order (`fa_fold_ml`: the first copied, each next rescaled to the
//     larger max and added);
//   the ROW's, its groups' states folded in group order; the row is O / L in bf16.
// No tile waits on the tile before it, so a decode computes a group's tiles all at once and folds them - its warps'
// tiles side by side, folded in tile order, its groups by the last block to arrive - while a prefill walks the same
// tiles in order and folds as it goes. Every float operation is spelled out (no contraction left to the compiler), so
// the folds are the same operations wherever they run.
//
// Keys are the mma's M and the rows its N (S^T = K Q^T, then O^T = V^T P^T, mma.sync m16n8k16): a row tile is 8 rows,
// so a one-row step of a group's 2 to 8 heads wastes no half-empty tile. The scores' C fragment becomes the weights'
// B fragment by an 8x8 transpose in registers (movmatrix). The dot product's k is PERMUTED as the tensor-core
// matvec's is (btb_gemv_mma.cuh): in each super-tile of 32 dims lane tig owns dims 8 tig .. 8 tig + 7, its pairs
// (0,1) (2,3) the fragment's k (tig*2, +1) (tig*2+8, +9) of the even k-tile and (4,5) (6,7) of the odd one - so a
// lane's K fragment of a key is one aligned 16-byte load, and q's one. The weights take the keys in order; only the
// output's dims are dealt to lanes as the loads suit (an output element's sum is its own whichever slot of the tile
// computes it). A row's sum of weights is its lane's keys in key order (the lanes holding it: gid, its keys gid,
// gid + 8, gid + 16, ..), then the lanes' sums across gid (xor 4, 8, 16).
//
// A row's arithmetic in an mma tile is its own: the rows beside it, masked or live, never reach it. A key a row does
// not see is masked; a tile it does not reach leaves it an empty state, and folding an empty state changes nothing -
// so a prompt's chunk and a step agree whatever tiles their launches walk.
//
// `tbl` (null: none) maps a logical slot to its cache row, wherever a page put it: an address, never a bit.

#define FA_TMAX 32  // a decode pass's rows at most (the card graph's widest pass)

// the one attention's shape at each head width: a tile's keys and a group's tiles - the arithmetic, the same in every
// form - and the decode form's warps a block, the prefill form's warps a block and row tiles a warp
template <int D>
struct FaPick {
    static constexpr int BN = D >= 256 ? 32 : 64;
    static constexpr int GT = 8;
    static constexpr int WPB = D >= 256 ? 4 : 8;  // the warps' item states in shared memory: 35 KB at the widest head
    // the prefill form: four warps, the rows the mma's M (`attn_flash_prefill_rm`), a row tile of 16 to SPL warps - two
    // at the widest head, where one warp's q, group output and tile accumulators pass its registers. Its fallback in the
    // decode form's orientation (`attn_flash_prefill`, for a card failing `btb_mma_roles`): NR row tiles of 8 a warp
    static constexpr int PFW = 4;
    // the prefill's blocks an SM, the registers' bound: three at the narrowest head (its shared memory allows six),
    // else what the registers give (two)
    static constexpr int PFB = D <= 64 ? 3 : 1;
    static constexpr int SPL = D >= 256 ? 2 : 1;
    static constexpr int NR = D >= 256 ? 1 : 2;
};

template <int D, int BN_, int GT_>
struct FaTile {
    static constexpr int BN = BN_;      // a tile's keys
    static constexpr int GT = GT_;      // a group's tiles
    static constexpr int GK = GT * BN;  // a group's keys
    static constexpr int KT = D / 16;   // the dot product's k-tiles
    static constexpr int ST = D / 32;   // its super-tiles (two k-tiles, the permuted k)
    static constexpr int MT = BN / 16;  // a tile's key m-tiles
    static constexpr int MD = D / 16;   // the output's dim m-tiles
    static constexpr int RUN = D / 16;  // the decode form's dims a lane takes of a V row, twice
    static constexpr int SL = BN / 32;  // the decode form's cache rows a lane looks up a tile
    static constexpr int SROW = D + 8;  // a V row in shared memory, padded so ldmatrix's rows miss each other
    static constexpr int CH = D / 8;    // 16-byte chunks a row
};

// a lane's rows - row r = 2 nr + e is slot 2 tig + e of row tile nr - their max, sum and output: o[nr][md][c] is row
// 2 nr + (c & 1) at dim slot gid (c < 2) or gid + 8 (c >= 2) of output m-tile md (which dims those are is the form's)
template <int MD, int NR>
struct FaState {
    float m[2 * NR], l[2 * NR];
    float o[NR][MD][4];
    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int r = 0; r < 2 * NR; ++r) {
            m[r] = NEG_INF;
            l[r] = 0.f;
        }
#pragma unroll
        for (int nr = 0; nr < NR; ++nr)
#pragma unroll
            for (int md = 0; md < MD; ++md) o[nr][md][0] = o[nr][md][1] = o[nr][md][2] = o[nr][md][3] = 0.f;
    }
};

// 2^x on the special function unit: the one exponential of the softmax and the folds, whose maxima and scores are in
// log2 units (the scale carries log2 e) - one instruction where expf's range reduction took several, a weight a tile
// for every key of every row. Flushed to zero below 2^-126 (a weight that small is nothing beside the row's largest,
// which is 1): without .ftz the instruction is wrapped in a denormal range check a weight
__device__ __forceinline__ float fa_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

#define FA_LOG2E 1.4426950408889634f

// a state (m, l, o) folded into the running one (M, L, O): each side scaled to the larger max - the larger side's
// factor exactly 1, the smaller's 2^(smaller - larger) - and the two summed in one fma, O' = fma(o, co, O * cO) and L'
// alike (`fa_fold_e`). An empty side's factor is 0 (its values zeros, never garbage: a fresh accumulator's, a cleared
// state's, its l 0), so a fold into an empty state copies and an empty state's fold changes nothing. (M, L) here, the
// factors out; one exponential and selects, no branch - as branches, a warp whose rows split between keeping their max
// and taking the state's ran both arms, an exponential and thirty-odd instructions a row each tile
__device__ __forceinline__ void fa_fold_ml(float& M, float& L, float m, float l, float& cO, float& co) {
    const bool em = m == NEG_INF, eM = M == NEG_INF;
    const float hi = fmaxf(M, m), lo = fminf(M, m);
    const float e = fa_ex2(em || eM ? 0.f : __fsub_rn(lo, hi));
    cO = em ? 1.f : eM ? 0.f : (M >= m ? 1.f : e);
    co = em ? 0.f : eM ? 1.f : (M > m ? e : 1.f);
    L = __fmaf_rn(l, co, __fmul_rn(L, cO));
    M = hi;
}

// an output element's fold (`fa_fold_ml`'s factors): where the running side's factor is 1 the product is O itself, so
// a caller whose rows all keep their max may leave it out - the same bits
__device__ __forceinline__ float fa_fold_e(float O, float o, float cO, float co) {
    return __fmaf_rn(o, co, __fmul_rn(O, cO));
}

// the row's output element from its folded state
__device__ __forceinline__ float fa_out(float O, float L) { return __fmul_rn(O, __fdiv_rn(1.f, L)); }

// an 8x8 bf16 matrix a register a lane (lane gid's row, columns 2 tig, 2 tig + 1) transposed across the warp
__device__ __forceinline__ unsigned fa_trans(unsigned x) {
    unsigned y;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}

// two floats as a bf16 pair in one register, the first in the low half: an mma fragment's two k (or two rows)
__device__ __forceinline__ unsigned fa_pack(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&v);
}

// four 8x8 bf16 matrices out of shared memory, each transposed into a lane's mma fragment register: lane i supplies the
// address of row i & 7 of matrix i >> 3
__device__ __forceinline__ void fa_ldsm_t4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const bf16* p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

// a row tile's q as the scores' B fragments, the k permuted: row slot gid's dims 8 tig .. + 7 of each super-tile, one
// load; a row with no query (null) zeros
template <int D>
__device__ __forceinline__ void fa_load_q(unsigned (&qb)[D / 16][2], const bf16* qr, int tig) {
#pragma unroll
    for (int st = 0; st < D / 32; ++st) {
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (qr != nullptr) v = *reinterpret_cast<const uint4*>(qr + st * 32 + tig * 8);
        qb[2 * st][0] = v.x;
        qb[2 * st][1] = v.y;
        qb[2 * st + 1][0] = v.z;
        qb[2 * st + 1][1] = v.w;
    }
}

// a key m-tile's super-tile into a row tile's scores: keys gid (ka) and gid + 8 (kb), the lane's eight dims of each;
// the even k-tile, then the odd
__device__ __forceinline__ void fa_qk(float (&s)[4], uint4 ka, uint4 kb, const unsigned (&q0)[2],
                                      const unsigned (&q1)[2]) {
    btb_mma16816(s[0], s[1], s[2], s[3], ka.x, kb.x, ka.y, kb.y, q0[0], q0[1]);
    btb_mma16816(s[0], s[1], s[2], s[3], ka.z, kb.z, ka.w, kb.w, q1[0], q1[1]);
}

// a tile's scores (s[mt][nr][2h + e]: key j0 + mt * 16 + 8 h + gid of row 2 nr + e) into the state: scaled to log2
// units (`scale2` = scale log2 e) and masked - row r of the lane sees the keys [first[r], last[r]] where `live[r]` -
// the rows' max over the eight lanes holding them, the weights 2^(s - max), summed in the lane's key order and then
// across the lanes; (m, l) the tile's own - its state from nothing. `full`: the caller found the tile inside every live
// row's keys (the warp's alike), so no mask is computed - a masked score was the same scaled score
template <class F, int NR>
__device__ __forceinline__ void fa_softmax(FaState<F::MD, NR>& st, float (&s)[F::MT][NR][4], int j0,
                                           const int (&first)[2 * NR], const int (&last)[2 * NR],
                                           const bool (&live)[2 * NR], float scale2, int gid, bool full) {
    float mx[2 * NR];
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) mx[r] = NEG_INF;
    if (full) {
#pragma unroll
        for (int mt = 0; mt < F::MT; ++mt)
#pragma unroll
            for (int nr = 0; nr < NR; ++nr)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = 2 * nr + (i & 1);
                    s[mt][nr][i] = __fmul_rn(s[mt][nr][i], scale2);
                    mx[r] = fmaxf(mx[r], s[mt][nr][i]);
                }
    } else {
#pragma unroll
        for (int mt = 0; mt < F::MT; ++mt) {
#pragma unroll
            for (int nr = 0; nr < NR; ++nr) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = 2 * nr + (i & 1);
                    const int j = j0 + mt * 16 + (i >> 1) * 8 + gid;
                    const bool ok = live[r] && j >= first[r] && j <= last[r];
                    s[mt][nr][i] = ok ? __fmul_rn(s[mt][nr][i], scale2) : NEG_INF;
                    mx[r] = fmaxf(mx[r], s[mt][nr][i]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 4));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 8));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 16));
    }
    float sum[2 * NR];
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) sum[r] = 0.f;
#pragma unroll
    for (int mt = 0; mt < F::MT; ++mt) {
#pragma unroll
        for (int nr = 0; nr < NR; ++nr) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = 2 * nr + (i & 1);
                const float p = mx[r] == NEG_INF ? 0.f : fa_ex2(__fsub_rn(s[mt][nr][i], mx[r]));
                s[mt][nr][i] = p;
                sum[r] = __fadd_rn(sum[r], p);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) {
        sum[r] = __fadd_rn(sum[r], __shfl_xor_sync(0xffffffffu, sum[r], 4));
        sum[r] = __fadd_rn(sum[r], __shfl_xor_sync(0xffffffffu, sum[r], 8));
        sum[r] = __fadd_rn(sum[r], __shfl_xor_sync(0xffffffffu, sum[r], 16));
        st.l[r] = sum[r];
        st.m[r] = mx[r];
    }
}

// key m-tile mt's weights of a row tile in bf16 as the P . V mma's B fragments: the scores' C fragment (keys gid,
// gid + 8 by rows 2 tig, + 1) transposed 8x8 into rows gid by keys 2 tig, + 1 (and + 8, + 9)
__device__ __forceinline__ void fa_pv_b(unsigned (&b)[2], const float (&s)[4]) {
    b[0] = fa_trans(fa_pack(s[0], s[1]));
    b[1] = fa_trans(fa_pack(s[2], s[3]));
}

__device__ __forceinline__ uint4 fa_ld16(const bf16* __restrict__ base, int row, int rs, int off) {
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (row >= 0) v = __ldg(reinterpret_cast<const uint4*>(base + (size_t)row * rs + off));
    return v;
}

// a run of RUN dims of a V row into words (two dims a word), a zero row where `row` < 0
template <int RUN>
__device__ __forceinline__ void fa_ld_run(unsigned (&w)[RUN / 2], const bf16* __restrict__ base, int row, int rs,
                                          int off) {
    if constexpr (RUN == 4) {
        uint2 v = make_uint2(0u, 0u);
        if (row >= 0) v = __ldg(reinterpret_cast<const uint2*>(base + (size_t)row * rs + off));
        w[0] = v.x;
        w[1] = v.y;
    } else {
#pragma unroll
        for (int c = 0; c < RUN / 8; ++c) {
            const uint4 v = fa_ld16(base, row, rs, off + c * 8);
            w[4 * c] = v.x;
            w[4 * c + 1] = v.y;
            w[4 * c + 2] = v.z;
            w[4 * c + 3] = v.w;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// the decode form: one graph for every length. Block (x, grp, g) takes group grp of key head g for one row tile of
// the pass - TREE: row i is token i / G at query head g * G + i % G, rows x * 8 .. + 7; ROWS: token x / ZR's heads, ZR
// = ceil(G / 8) blocks a token - its WPB warps the group's tiles (warp w tiles w, w + WPB, ..), each a state from
// nothing; the block's threads fold them in tile order through shared memory, a thread a share of every lane's
// state. A row reaching one group writes its output there; a row reaching several stores the group's state, and the
// last of its groups' blocks to arrive folds them in group order (`cnt`, reset by it), the block's threads a run of
// four dims each. TREE: *n0p rows stand before the pass; token t's key at logical position n0 + d is its ancestor at
// depth d (par[], -1 at the root), so a verify pass's node walks its committed step's keys. ROWS: token t is a
// sequence of its own (`RowsLayout`). `part_*` hold S x T x Hq states, S the groups the graph's capacity reaches.
//
// A lane's tile: the cache rows of keys j0 + lane (+ 32) looked up once and passed round by shuffles; K of keys
// mt * 16 + gid (+ 8), their super-tiles' dims 8 tig .. a load each; V of keys kk * 16 + 2 tig (+ 1, + 8, + 9), two
// runs of RUN dims - gid * RUN .. and D / 2 + gid * RUN .. - a pair of keys' same dim packed into an A fragment. So
// output m-tile md's slot gid is dim gid * RUN + md and slot gid + 8 dim D / 2 + gid * RUN + md.
// ---------------------------------------------------------------------------------------------------------
#define FA_TREE 0
#define FA_ROWS 1

// one tile's state from nothing into `st`, a warp's row tile, its keys' cache rows `slot(j)` (-1: a zero row, a key no
// row walks)
template <class F, int D, typename Slot>
__device__ __forceinline__ void fa_pass(FaState<F::MD, 1>& st, const unsigned (&qb)[D / 16][2],
                                        const bf16* __restrict__ Kg, const bf16* __restrict__ Vg, int rs, int j0,
                                        Slot slot, const int (&first)[2], const int (&last)[2], const bool (&live)[2],
                                        float scale2, int lane, bool full) {
    const int gid = lane >> 2, tig = lane & 3;
    int sl[F::SL];
#pragma unroll
    for (int c = 0; c < F::SL; ++c) sl[c] = slot(j0 + c * 32 + lane);
    // the tile's K rows, and its V rows beside them where both fit a lane's registers (a short tile: one trip to the
    // memory a tile, not two); a long tile's V after its scores, a k-tile's keys at a time as its P . V takes them -
    // the whole tile's would pass the registers beside the accumulators (the widest head: 128 of V, 64 of
    // accumulators, 32 of q - spilled)
    constexpr bool EARLY_V = F::MT * F::ST * 8 + F::MT * 4 * F::RUN <= 160;
    constexpr bool V_BY_KK = !EARLY_V;
    uint4 kv[F::MT][F::ST][2];
#pragma unroll
    for (int mt = 0; mt < F::MT; ++mt) {
        const int ra = __shfl_sync(0xffffffffu, sl[mt >> 1], ((mt & 1) << 4) + gid);
        const int rb = __shfl_sync(0xffffffffu, sl[mt >> 1], ((mt & 1) << 4) + 8 + gid);
#pragma unroll
        for (int t = 0; t < F::ST; ++t) {
            kv[mt][t][0] = fa_ld16(Kg, ra, rs, t * 32 + tig * 8);
            kv[mt][t][1] = fa_ld16(Kg, rb, rs, t * 32 + tig * 8);
        }
    }
    unsigned vw[V_BY_KK ? 1 : F::MT][4][2][F::RUN / 2];
    // k-tile kk's keys' V runs into vw[at]
    auto load_vk = [&](int kk, int at) {
        const int k0 = ((kk & 1) << 4) + tig * 2;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int rk = __shfl_sync(0xffffffffu, sl[kk >> 1], k0 + (k & 1) + (k >> 1) * 8);
            fa_ld_run<F::RUN>(vw[at][k][0], Vg, rk, rs, gid * F::RUN);
            fa_ld_run<F::RUN>(vw[at][k][1], Vg, rk, rs, D / 2 + gid * F::RUN);
        }
    };
    if constexpr (EARLY_V) {
#pragma unroll
        for (int kk = 0; kk < F::MT; ++kk) load_vk(kk, kk);
    }
    float s[F::MT][1][4];
#pragma unroll
    for (int mt = 0; mt < F::MT; ++mt) {
        s[mt][0][0] = s[mt][0][1] = s[mt][0][2] = s[mt][0][3] = 0.f;
#pragma unroll
        for (int t = 0; t < F::ST; ++t) fa_qk(s[mt][0], kv[mt][t][0], kv[mt][t][1], qb[2 * t], qb[2 * t + 1]);
    }
    fa_softmax<F, 1>(st, s, j0, first, last, live, scale2, gid, full);
    {
        // the tile's accumulators from zero here, not before its loads and scores: zeros held through them are
        // registers the loads could have had
#pragma unroll
        for (int md = 0; md < F::MD; ++md) st.o[0][md][0] = st.o[0][md][1] = st.o[0][md][2] = st.o[0][md][3] = 0.f;
    }
#pragma unroll
    for (int kk = 0; kk < F::MT; ++kk) {
        const int at = V_BY_KK ? 0 : kk;
        if constexpr (V_BY_KK) load_vk(kk, 0);
        unsigned b[2];
        fa_pv_b(b, s[kk][0]);
#pragma unroll
        for (int md = 0; md < F::MD; ++md) {
            const unsigned sel = (md & 1) ? 0x7632u : 0x5410u;
            const int x = md >> 1;
            const unsigned a0 = __byte_perm(vw[at][0][0][x], vw[at][1][0][x], sel);
            const unsigned a1 = __byte_perm(vw[at][0][1][x], vw[at][1][1][x], sel);
            const unsigned a2 = __byte_perm(vw[at][2][0][x], vw[at][3][0][x], sel);
            const unsigned a3 = __byte_perm(vw[at][2][1][x], vw[at][3][1][x], sel);
            btb_mma16816(st.o[0][md][0], st.o[0][md][1], st.o[0][md][2], st.o[0][md][3], a0, a1, a2, a3, b[0], b[1]);
        }
    }
}

template <class F, int D, int MODE, int WPB>
__device__ __forceinline__ void attn_flash(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                           const bf16* __restrict__ V, bf16* __restrict__ out,
                                           const int* __restrict__ n0p, const int* __restrict__ par,
                                           const int* __restrict__ rw, int T, int Hq, int Hk, int hs, int rs,
                                           float scale, int win, float* __restrict__ part_m,
                                           float* __restrict__ part_l, float* __restrict__ part_acc,
                                           int* __restrict__ cnt, const int* __restrict__ tbl) {
    constexpr int NF = 4 + F::MD * 4;               // an item's state's floats a lane
    constexpr int OE = F::MD * 4 / WPB;             // a thread's share of a lane's output elements
    constexpr int V4 = D / 4;                       // a row's runs of four dims
    constexpr int R4 = (8 * V4 + WPB * 32 - 1) / (WPB * 32);  // a thread's runs of the row tile
    __shared__ float fs[WPB][NF][32];  // the warps' item states, a lane's floats side by side
    __shared__ __align__(16) float gso[8][D];  // the group's state, row-major: [row slot][dim]
    __shared__ float gsm[8], gsl[8];
    __shared__ int sflag[8];
    [[maybe_unused]] __shared__ int anc[FA_TMAX][FA_TMAX];
    [[maybe_unused]] __shared__ int dep[FA_TMAX];
    [[maybe_unused]] __shared__ int s_chain;
    const int grp = blockIdx.y, g = blockIdx.z;
    const int G = Hq / Hk;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int lo = grp * F::GK, hi = lo + F::GK - 1;
    int n0 = 0, rbase, tok_b = 0;
    [[maybe_unused]] RowsLayout rl;
    if constexpr (MODE == FA_ROWS) {
        const int ZR = (G + 7) / 8;
        tok_b = blockIdx.x / ZR;
        rbase = (blockIdx.x % ZR) * 8;
        rl = RowsLayout(rw, tok_b);
        if (rl.len < 0 || lo > rl.len + rl.step) return;  // a padding row, or a group past the row's keys
    } else {
        n0 = *n0p;
        if (lo >= n0 + T) return;  // past every row's keys: nothing stored, nothing counted
        rbase = blockIdx.x * 8;
        // each token's path: its ancestors root first, and whether the pass is a chain (every path the identity) -
        // walked twice, its length and then its nodes from the end, so no array a thread indexes at run time (a
        // local array: memory, not registers)
        if (threadIdx.x == 0) s_chain = 1;
        __syncthreads();
        for (int t = threadIdx.x; t < T; t += blockDim.x) {
            int len = 0;
            for (int p = t; p >= 0 && p < T && len < FA_TMAX; p = par[p]) ++len;
            int k = len;
            for (int p = t; k > 0; p = par[p]) {
                anc[t][--k] = p;
                if (p != k) s_chain = 0;
            }
            dep[t] = len - 1;
        }
        __syncthreads();
    }
    const int rows = MODE == FA_ROWS ? G : T * G;
    if (rbase >= rows) return;
    // row slot r's token, query head and keys (the lane's two rows are slots 2 tig, + 1)
    auto row_at = [&](int r, int& tok, int& head, int& first, int& last) -> bool {
        const int i = rbase + r;
        const bool ex = i < rows;
        tok = MODE == FA_ROWS ? tok_b : (ex ? i / G : 0);
        head = g * G + (ex ? i % G : 0);
        if constexpr (MODE == FA_ROWS) {
            last = ex ? rl.len + rl.step : -1;
        } else {
            last = ex ? n0 + dep[tok] : -1;
        }
        first = win > 0 ? max(last + 1 - win, 0) : 0;
        return ex;
    };
    int tok[2], first[2], last[2];
    bool exists[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        int head;  // the row's query head: its output's, which the threads storing it look up again
        exists[e] = row_at(2 * tig + e, tok[e], head, first[e], last[e]);
    }
    // the block's reach in this group: the keys any of its rows sees here (every warp the same)
    int b_lo = min(exists[0] ? first[0] : 0x7fffffff, exists[1] ? first[1] : 0x7fffffff);
    int b_hi = max(last[0], last[1]);
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        b_lo = min(b_lo, __shfl_xor_sync(0xffffffffu, b_lo, o));
        b_hi = max(b_hi, __shfl_xor_sync(0xffffffffu, b_hi, o));
    }
    b_lo = max(b_lo, lo);
    b_hi = min(b_hi, hi);
    if (b_lo > b_hi) return;  // no row of the block reaches this group
    // the keys every row of the block sees: a tile inside them takes no mask
    int v_lo = max(exists[0] ? first[0] : 0, exists[1] ? first[1] : 0);
    int v_hi = min(exists[0] ? last[0] : 0x7fffffff, exists[1] ? last[1] : 0x7fffffff);
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        v_lo = max(v_lo, __shfl_xor_sync(0xffffffffu, v_lo, o));
        v_hi = min(v_hi, __shfl_xor_sync(0xffffffffu, v_hi, o));
    }
    const float scale2 = __fmul_rn(scale, FA_LOG2E);
    // row slot gid's query
    const int iq = rbase + gid;
    const int tq = MODE == FA_ROWS ? tok_b : (iq < rows ? iq / G : 0);
    const bf16* qr = iq < rows ? q + ((size_t)tq * Hq + g * G + iq % G) * D : nullptr;
    unsigned qb[D / 16][2];
    fa_load_q<D>(qb, qr, tig);
    const bf16* Kg = K + (size_t)g * hs;
    const bf16* Vg = V + (size_t)g * hs;
    // the block's work: an item a tile it reaches, each a state from nothing - and a tree's tail tile an item a token
    // of the block's rows, the token's own path walked (its rows live, the others left empty). Each row is live in
    // exactly one of a tail tile's items and folding an empty state changes nothing, so the items folded one after
    // another in the tile's place are the tile's state folded: the tail's tokens run on the block's warps side by
    // side, where they ran one after another on the warp that held the tile. Shared tiles come first (the tail
    // tiles end the reach), so an item's tile and token are arithmetic
    const int w_lo = (b_lo - lo) / F::BN, w_hi = (b_hi - lo) / F::BN;
    const int tok_lo = MODE == FA_ROWS ? tok_b : rbase / G;
    const int ntok = MODE == FA_ROWS ? 1 : min(T - 1, (rbase + 7) / G) - tok_lo + 1;
    int w_tail = w_hi + 1;  // the first tile walked a token at a time
    if constexpr (MODE == FA_TREE) {
        if (!s_chain) w_tail = max(w_lo, min(w_hi + 1, (n0 - lo) / F::BN));
    }
    const int n_items = (w_tail - w_lo) + (w_hi + 1 - w_tail) * ntok;
    // the group's state, each thread its share of every lane's - OE of its output elements, element f = warp + WPB i
    // of md * 4 + c, all of them row ge = warp & 1's (WPB even), whose (M, L) the thread folds beside them. One row a
    // thread, its fold's factors scalars: indexed by a row the thread picks at run time they were an array in memory
    static_assert(WPB % 2 == 0, "a thread's output elements are one row's");
    const int ge = warp & 1;
    float gm = NEG_INF, gl = 0.f, go[OE];
#pragma unroll
    for (int i = 0; i < OE; ++i) go[i] = 0.f;
    // the items a warp each, WPB at a time: each into shared memory, then every thread folds them in item order
    for (int c0 = 0; c0 < n_items; c0 += WPB) {
        const int it = c0 + warp;
        if (it < n_items) {
            // the item's state from nothing: (m, l) by its softmax, o zeroed just before its P . V (a tail's token's
            // rows of the other tokens masked, so empty)
            FaState<F::MD, 1> st;
            st.m[0] = st.m[1] = NEG_INF;
            st.l[0] = st.l[1] = 0.f;
            if (it < w_tail - w_lo) {
                const int j0 = lo + (w_lo + it) * F::BN;
                fa_pass<F, D>(
                    st, qb, Kg, Vg, rs, j0,
                    [&](int j) {
                        // a key outside the block's reach is masked for every row: a zero row, nothing read - a
                        // node's keys before its window come as a map that starts at the window (`_card_attention`)
                        if (j > b_hi || j < b_lo) return -1;
                        if constexpr (MODE == FA_ROWS) {
                            return rl.slot(j);
                        } else {
                            return tbl != nullptr ? tbl[j] : j;
                        }
                    },
                    first, last, exists, scale2, lane, j0 >= v_lo && j0 + F::BN - 1 <= v_hi);
            } else {
                const int k = it - (w_tail - w_lo);
                const int j0 = lo + (w_tail + k / ntok) * F::BN, u = tok_lo + k % ntok;
                const int top = min(n0 + dep[u], b_hi);
                if (top < j0) {
                    st.clear();  // the token's path ends before the tile: its rows' keys here all masked, empty
                } else {
                    // a state from nothing, as a shared tile's - once cleared and rescaled by the softmax instead
                    // (the same bits: its factor 0 or 1 on zeros), its 64 accumulators stood live through the
                    // tile's loads, past the registers at the widest head
                    const bool live[2] = {exists[0] && tok[0] == u, exists[1] && tok[1] == u};
                    fa_pass<F, D>(
                        st, qb, Kg, Vg, rs, j0,
                        [&](int j) {
                            if (j > top) return -1;
                            const int sl = j < n0 ? j : n0 + anc[u][j - n0];
                            return tbl != nullptr ? tbl[sl] : sl;
                        },
                        first, last, live, scale2, lane, false);
                }
            }
            float(*f)[32] = fs[warp];
            f[0][lane] = st.m[0];
            f[1][lane] = st.m[1];
            f[2][lane] = st.l[0];
            f[3][lane] = st.l[1];
#pragma unroll
            for (int md = 0; md < F::MD; ++md)
#pragma unroll
                for (int c = 0; c < 4; ++c) f[4 + md * 4 + c][lane] = st.o[0][md][c];
        }
        __syncthreads();
        for (int w = 0; w < WPB && c0 + w < n_items; ++w) {
            const float(*f)[32] = fs[w];
            float cO, co;
            fa_fold_ml(gm, gl, f[ge][lane], f[2 + ge][lane], cO, co);
#pragma unroll
            for (int i = 0; i < OE; ++i) go[i] = fa_fold_e(go[i], f[4 + warp + WPB * i][lane], cO, co);
        }
        __syncthreads();
    }
    // the group's state row-major in shared memory: the decode form's dims (slot gid of m-tile md: dim gid * RUN + md,
    // slot gid + 8: D / 2 + gid * RUN + md)
#pragma unroll
    for (int i = 0; i < OE; ++i) {
        const int fe = warp + WPB * i, md = fe >> 2, c = fe & 3;
        gso[2 * tig + ge][(c >> 1) * (D / 2) + gid * F::RUN + md] = go[i];
    }
    if (warp < 2 && gid == 0) {
        gsm[2 * tig + ge] = gm;
        gsl[2 * tig + ge] = gl;
    }
    __syncthreads();
    // each row's group state: its output where the group is its only one, else stored and counted; a thread's runs
    // are row r = u / V4's dims 4 (u % V4) .. + 3
#pragma unroll
    for (int k = 0; k < R4; ++k) {
        const int u = threadIdx.x + k * WPB * 32;
        if (u >= 8 * V4) break;
        const int r = u / V4, d4 = (u % V4) * 4;
        int rt, rh, rf, rl_;
        if (!row_at(r, rt, rh, rf, rl_)) continue;
        const int s_first = rf / F::GK, s_last = rl_ / F::GK;
        if (grp < s_first || grp > s_last) continue;
        const size_t row = (size_t)rt * Hq + rh;
        const float4 v = *reinterpret_cast<const float4*>(&gso[r][d4]);
        if (s_first == s_last) {
            // the fold of one group is its copy: O / L of the group's own state
            const float L = gsl[r];
            const __nv_bfloat162 b0 = __floats2bfloat162_rn(fa_out(v.x, L), fa_out(v.y, L));
            const __nv_bfloat162 b1 = __floats2bfloat162_rn(fa_out(v.z, L), fa_out(v.w, L));
            *reinterpret_cast<uint2*>(out + row * D + d4) =
                make_uint2(*reinterpret_cast<const unsigned*>(&b0), *reinterpret_cast<const unsigned*>(&b1));
            continue;
        }
        const size_t idx = (size_t)grp * T * Hq + row;
        if (d4 == 0) {
            part_m[idx] = gsm[r];
            part_l[idx] = gsl[r];
        }
        *reinterpret_cast<float4*>(part_acc + idx * D + d4) = v;
    }
    __threadfence();
    __syncthreads();
    // the block's arrival for each of its rows; the last group's block of a row folds it in group order
    if (threadIdx.x < 8) {
        const int r = threadIdx.x;
        int rt, rh, rf, rl_;
        int fold = 0;
        if (row_at(r, rt, rh, rf, rl_)) {
            const int s_first = rf / F::GK, s_last = rl_ / F::GK;
            if (grp >= s_first && grp <= s_last && s_first != s_last)
                fold = atomicAdd(cnt + (size_t)rt * Hq + rh, 1) == s_last - s_first ? 1 : 0;
        }
        sflag[r] = fold;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < R4; ++k) {
        const int u = threadIdx.x + k * WPB * 32;
        if (u >= 8 * V4) break;
        const int r = u / V4, d4 = (u % V4) * 4;
        if (!sflag[r]) continue;
        __threadfence();
        int rt, rh, rf, rl_;
        row_at(r, rt, rh, rf, rl_);
        const int s_first = rf / F::GK, s_last = rl_ / F::GK;
        const size_t row = (size_t)rt * Hq + rh;
        float M = NEG_INF, L = 0.f, O[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 8
        for (int sp = s_first; sp <= s_last; ++sp) {
            const size_t iu = (size_t)sp * T * Hq + row;
            float cO, co;
            fa_fold_ml(M, L, __ldcg(part_m + iu), __ldcg(part_l + iu), cO, co);
            const float4 v = __ldcg(reinterpret_cast<const float4*>(part_acc + iu * D + d4));
            O[0] = fa_fold_e(O[0], v.x, cO, co);
            O[1] = fa_fold_e(O[1], v.y, cO, co);
            O[2] = fa_fold_e(O[2], v.z, cO, co);
            O[3] = fa_fold_e(O[3], v.w, cO, co);
        }
        const __nv_bfloat162 b0 = __floats2bfloat162_rn(fa_out(O[0], L), fa_out(O[1], L));
        const __nv_bfloat162 b1 = __floats2bfloat162_rn(fa_out(O[2], L), fa_out(O[3], L));
        *reinterpret_cast<uint2*>(out + row * D + d4) =
            make_uint2(*reinterpret_cast<const unsigned*>(&b0), *reinterpret_cast<const unsigned*>(&b1));
        if (d4 == 0) cnt[row] = 0;
    }
}

// the decode form's two kernels a head width, one launch shape (FaPick): btb_attn_flash_d* a tree of T rows after the
// cache's *n0p rows (`par` their parents; `tbl` the cache's row map, null for none), btb_attn_flash_rows_d* T rows
// each a sequence of its own (`rw` their layout)
#define ATTN_FLASH_K(D)                                                                                      \
    extern "C" __global__ void __launch_bounds__(FaPick<D>::WPB * 32) btb_attn_flash_d##D(                   \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ n0p, const int* __restrict__ par, int T, int Hq,     \
        int Hk, int hs, int rs, float scale, float* __restrict__ part_m, float* __restrict__ part_l,         \
        float* __restrict__ part_acc, int* __restrict__ cnt, int win, const int* __restrict__ tbl) {         \
        attn_flash<FaTile<D, FaPick<D>::BN, FaPick<D>::GT>, D, FA_TREE, FaPick<D>::WPB>(                     \
            q, K, V, out, n0p, par, nullptr, T, Hq, Hk, hs, rs, scale, win, part_m, part_l, part_acc, cnt,   \
            tbl);                                                                                            \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(FaPick<D>::WPB * 32) btb_attn_flash_rows_d##D(              \
        const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                  \
        bf16* __restrict__ out, const int* __restrict__ rw, int T, int Hq, int Hk, int hs, int rs,           \
        float scale, float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,   \
        int* __restrict__ cnt, int win) {                                                                    \
        attn_flash<FaTile<D, FaPick<D>::BN, FaPick<D>::GT>, D, FA_ROWS, FaPick<D>::WPB>(                     \
            q, K, V, out, nullptr, nullptr, rw, T, Hq, Hk, hs, rs, scale, win, part_m, part_l, part_acc,     \
            cnt, nullptr);                                                                                   \
    }
ATTN_FLASH_K(64)
ATTN_FLASH_K(128)
ATTN_FLASH_K(256)

// ---------------------------------------------------------------------------------------------------------
// the prefill form: a prompt's chunk, T tokens at positions n0 .. n0 + T - 1, token t over the keys [first, n0 + t].
// Block (z, g) takes PFW * 8 NR / G tokens' rows of key head g (their G query heads each), NR row tiles a warp, and
// walks every tile from its first row's first key to its last row's own position: each tile's state from nothing,
// folded into the group's as the walk goes, the group's into the row's as the walk leaves the group - the decode
// form's arithmetic. The three states: the tile's and the group's in registers (the tile's the mma's accumulators, a
// row tile's at a time, folded into the group's as its P . V ends), the row's in `run` ([T, Hq, D] float32 on the
// card, touched at the groups' ends alone); their (max, sum)s in registers. A tile lands in shared memory by
// asynchronous copies, the next tile's K while this one's V is used and its V while the next scores run; K's rows
// swizzled (a row's 16-byte chunk c at c ^ 4 (row & 1), so a quarter-warp's eight fragment loads fall in distinct
// banks), V's padded for ldmatrix. Output m-tile md's slot r is dim md * 16 + r.
// The block's shared memory is dynamic (`FaPfSmem`, past the 48 KB a static array may take): the host sets the
// kernel's ceiling to it and launches with it.
// ---------------------------------------------------------------------------------------------------------

// a prefill block's key head and first token, grid (ceil(T / tpb), Hk): a key head's blocks dispatched together, so
// the blocks on the card at once walk one head's keys side by side and read them from the L2 (numbered heads fastest,
// they spread over every head and read each head's keys from DRAM again and again: 20 times the existing prefill's
// DRAM traffic at a long prefix, the DRAM saturated), and within a head the chunk's last tokens first - the longest
// walks (a token's keys are its position) dispatched first, the launch's tail short ones
__device__ __forceinline__ void fa_pf_block(int tpb, int& g, int& t0) {
    g = blockIdx.y;
    t0 = (gridDim.x - 1 - blockIdx.x) * tpb;
}

// the prefill form's shared memory, bytes: K's tile and V's
template <class F, int D>
struct FaPfSmem {
    static constexpr int K = F::BN * D * 2;
    static constexpr int V = F::BN * F::SROW * 2;
    static constexpr int BYTES = K + V;
};

// the group state's output in registers, the P . V and its fold a row tile at a time (the row tiles' V fragments read
// again for each, the tile's accumulators of one row tile live at once) - the block's shared memory only the tile it
// walks, two blocks to an SM, whose phases drift apart where one block's warps all stood in step at its barriers
template <class F, int D, int PFW, int NR>
__device__ __forceinline__ void attn_flash_prefill(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                                   const bf16* __restrict__ V, bf16* __restrict__ out, int n0, int T,
                                                   int Hq, int Hk, int hs, int rs, float scale, int win,
                                                   const int* __restrict__ tbl, float* __restrict__ run) {
    using SM = FaPfSmem<F, D>;
    constexpr int NTH = PFW * 32;
    constexpr int RW = 8 * NR;  // a warp's rows
    extern __shared__ __align__(16) unsigned char fa_smem[];
    {
        // launched with less than the layout takes, the kernel would write past its block's memory: a fault, not that
        unsigned have;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
        if (have < SM::BYTES) __trap();
    }
    bf16* const sK = reinterpret_cast<bf16*>(fa_smem);
    bf16* const sV = reinterpret_cast<bf16*>(fa_smem + SM::K);
    const int G = Hq / Hk;
    // a block's rows a whole token's heads at least (the host takes the kernel only there: `_card_family_ok`)
    if (G > PFW * RW) __trap();
    const int tpb = PFW * RW / G;  // tokens a block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    int g, t0;
    fa_pf_block(tpb, g, t0);
    // the lane's rows: r = 2 nr + e, slot nr * 8 + 2 tig + e of the warp's - each its row of out (and of run), its keys
    int orow[2 * NR], first[2 * NR], last[2 * NR];
    bool exists[2 * NR];
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) {
        const int i = warp * RW + (r >> 1) * 8 + 2 * tig + (r & 1);
        const int t = t0 + i / G;
        exists[r] = i < tpb * G && t < T;
        orow[r] = exists[r] ? t * Hq + g * G + i % G : 0;
        last[r] = exists[r] ? n0 + t : -1;
        first[r] = win > 0 ? max(n0 + t + 1 - win, 0) : 0;
    }
    const int t_hi = min(t0 + tpb, T) - 1;
    const int b_hi = n0 + t_hi;
    const int b_lo = win > 0 ? max(n0 + t0 + 1 - win, 0) : 0;
    // the keys every row of the warp sees: a tile inside them takes no mask
    int v_lo = 0, v_hi = 0x7fffffff;
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) {
        if (!exists[r]) continue;
        v_lo = max(v_lo, first[r]);
        v_hi = min(v_hi, last[r]);
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
        v_lo = max(v_lo, __shfl_xor_sync(0xffffffffu, v_lo, o));
        v_hi = min(v_hi, __shfl_xor_sync(0xffffffffu, v_hi, o));
    }
    const float scale2 = __fmul_rn(scale, FA_LOG2E);
    // each row tile's query (row slot gid), its fragments read again each tile (from L1: the registers are the states')
    const bf16* qr[NR];
#pragma unroll
    for (int nr = 0; nr < NR; ++nr) {
        const int iq = warp * RW + nr * 8 + gid, tq = t0 + iq / G;
        qr[nr] = (iq < tpb * G && tq < T) ? q + ((size_t)tq * Hq + g * G + iq % G) * D + tig * 8 : nullptr;
    }
    // the tile's state (registers); the group's (m, l) and its output (`gso`, element (nr, md, c) as the tile state's)
    // in registers; the row's (m, l) here and its output in `run`. The group's output is always finite (zeros before any
    // tile), as a fold scaling it by 0 needs
    FaState<F::MD, NR> st;
    float gM[2 * NR], gL[2 * NR], rM[2 * NR], rL[2 * NR];
#pragma unroll
    for (int r = 0; r < 2 * NR; ++r) {
        gM[r] = rM[r] = NEG_INF;
        gL[r] = rL[r] = 0.f;
    }
    float gso[NR][F::MD][4];
#pragma unroll
    for (int nr = 0; nr < NR; ++nr)
#pragma unroll
        for (int md = 0; md < F::MD; ++md) gso[nr][md][0] = gso[nr][md][1] = gso[nr][md][2] = gso[nr][md][3] = 0.f;
    // row tile nr's tile state into the group's: a row's factors once, then its elements
    auto fold_tile = [&](int nr) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int r = 2 * nr + e;
            float cO, co;
            fa_fold_ml(gM[r], gL[r], st.m[r], st.l[r], cO, co);
#pragma unroll
            for (int md = 0; md < F::MD; ++md) {
                gso[nr][md][e] = fa_fold_e(gso[nr][md][e], st.o[nr][md][e], cO, co);
                gso[nr][md][2 + e] = fa_fold_e(gso[nr][md][2 + e], st.o[nr][md][2 + e], cO, co);
            }
        }
    };
    // the group's state into the row's, and the group's emptied (its max: its output stays finite, and the next fold
    // scales it by 0): a row tile at a time, its two rows' `run` values loaded first, then folded, then stored - a load
    // consumed before the next issued left the block's warps a memory round trip an element at every group's end, and
    // every row's at once were 64 registers beside the group's output, spilled. The walk's last fold writes the rows'
    // output from its registers instead (read back from `run`, each load waited on the store to `out` before it, which
    // the compiler could not tell apart)
    auto fold_row = [&](bool last_fold) {
#pragma unroll
        for (int nr = 0; nr < NR; ++nr) {
            float v[2][F::MD][2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                // a row's first group finds `run` holding nothing of it (whatever is there, a fold into an empty
                // state scales by 0, which must meet zeros); a padding row never has a state, so never reads
                const int r = 2 * nr + e;
                const bool had = rM[r] != NEG_INF;
                const float* rp = run + (size_t)orow[r] * D + gid;
#pragma unroll
                for (int md = 0; md < F::MD; ++md) {
                    v[e][md][0] = had ? rp[md * 16] : 0.f;
                    v[e][md][1] = had ? rp[md * 16 + 8] : 0.f;
                }
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int r = 2 * nr + e;
                float cO, co;
                fa_fold_ml(rM[r], rL[r], gM[r], gL[r], cO, co);
#pragma unroll
                for (int md = 0; md < F::MD; ++md) {
                    v[e][md][0] = fa_fold_e(v[e][md][0], gso[nr][md][e], cO, co);
                    v[e][md][1] = fa_fold_e(v[e][md][1], gso[nr][md][2 + e], cO, co);
                }
                gM[r] = NEG_INF;
                gL[r] = 0.f;
                if (!exists[r]) continue;  // a padding row's state would land on another row's, in `run`
                if (last_fold) {
                    bf16* op = out + (size_t)orow[r] * D + gid;
#pragma unroll
                    for (int md = 0; md < F::MD; ++md) {
                        op[md * 16] = f2bf(fa_out(v[e][md][0], rL[r]));
                        op[md * 16 + 8] = f2bf(fa_out(v[e][md][1], rL[r]));
                    }
                } else {
                    float* rp = run + (size_t)orow[r] * D + gid;
#pragma unroll
                    for (int md = 0; md < F::MD; ++md) {
                        rp[md * 16] = v[e][md][0];
                        rp[md * 16 + 8] = v[e][md][1];
                    }
                }
            }
        }
    };
    const bf16* Kg = K + (size_t)g * hs;
    const bf16* Vg = V + (size_t)g * hs;
    const bool warp_rows = warp * RW < tpb * G && t0 + (warp * RW) / G < T;
    // a tile's K or V rows into shared memory (a key past the block's last row a zero row), one commit group. A thread
    // copies the same 16-byte chunk ch0 of rows r0, r0 + RSTEP, ..: its places in shared memory one base and fixed
    // steps (RSTEP even, so a K row's swizzle is the thread's own), and in the cache a row's start plus ch0 - computed
    // per chunk, the copies' addresses were a loop's worth of values kept across the walk
    constexpr int RSTEP = NTH / F::CH;
    static_assert(NTH % F::CH == 0 && RSTEP % 2 == 0 && F::BN % RSTEP == 0, "a thread's chunks are one column");
    const int r0 = threadIdx.x / F::CH, ch0 = threadIdx.x % F::CH;
    bf16* const sk0 = sK + r0 * D + ((ch0 ^ ((r0 & 1) << 2)) << 3);
    bf16* const sv0 = sV + r0 * F::SROW + ch0 * 8;
    const bf16* const Kc = Kg + ch0 * 8;
    const bf16* const Vc = Vg + ch0 * 8;
    auto issue = [&](bool is_k, int j0) {
#pragma unroll
        for (int it = 0; it < F::BN / RSTEP; ++it) {
            const int j = j0 + r0 + it * RSTEP;
            const bool ok = j <= b_hi;
            const int sl = ok ? (tbl != nullptr ? tbl[j] : j) : 0;
            if (is_k) {
                cp16(sk0 + it * RSTEP * D, Kc + (size_t)sl * rs, ok);
            } else {
                cp16(sv0 + it * RSTEP * F::SROW, Vc + (size_t)sl * rs, ok);
            }
        }
        cp_commit();
    };
    const int jf = b_lo / F::BN * F::BN;
    issue(true, jf);
    issue(false, jf);
    int grp = jf / F::GK;
    for (int j0 = jf; j0 <= b_hi; j0 += F::BN) {
        const bool more = j0 + F::BN <= b_hi;
        if (j0 / F::GK != grp) {  // the walk leaves a group: its state into the row's
            if (warp_rows) fold_row(false);
            grp = j0 / F::GK;
        }
        cp_wait1();  // this tile's K (its V may still be on the way)
        __syncthreads();
        // the tile's state is made in its own phases: (m, l) by the softmax, o zeroed just before P . V - zeroed here,
        // its 64 accumulators stood live and idle through the scores and the softmax
        float s[F::MT][NR][4];
        if (warp_rows) {
#pragma unroll
            for (int mt = 0; mt < F::MT; ++mt)
#pragma unroll
                for (int nr = 0; nr < NR; ++nr) s[mt][nr][0] = s[mt][nr][1] = s[mt][nr][2] = s[mt][nr][3] = 0.f;
            // a super-tile at a time: its q fragments read once and used against every key m-tile (each score still
            // takes its k-tiles in order) - with the key m-tiles outside, each fragment was wanted four times and the
            // compiler kept all of them standing
            const int sw = (gid & 1) << 2;
#pragma unroll
            for (int t = 0; t < F::ST; ++t) {
                const int off = ((t * 4 + tig) ^ sw) << 3;
                unsigned q0[NR][2], q1[NR][2];
#pragma unroll
                for (int nr = 0; nr < NR; ++nr) {
                    uint4 qv = make_uint4(0u, 0u, 0u, 0u);
                    if (qr[nr] != nullptr) qv = __ldg(reinterpret_cast<const uint4*>(qr[nr] + t * 32));
                    q0[nr][0] = qv.x;
                    q0[nr][1] = qv.y;
                    q1[nr][0] = qv.z;
                    q1[nr][1] = qv.w;
                }
#pragma unroll
                for (int mt = 0; mt < F::MT; ++mt) {
                    const bf16* ka = sK + (mt * 16 + gid) * D;
                    const uint4 a = *reinterpret_cast<const uint4*>(ka + off);
                    const uint4 b = *reinterpret_cast<const uint4*>(ka + 8 * D + off);
#pragma unroll
                    for (int nr = 0; nr < NR; ++nr) fa_qk(s[mt][nr], a, b, q0[nr], q1[nr]);
                }
            }
        }
        __syncthreads();  // K's tile consumed: the next one's copies may land
        if (more) {
            issue(true, j0 + F::BN);
        } else {
            cp_commit();
        }
        // the weights packed into the P . V B fragments as soon as they are made: bf16 pairs, half the scores' registers
        // through the P . V, where the tile's accumulators and the group's state stand beside them
        unsigned pb[F::MT][NR][2];
        if (warp_rows) {
            fa_softmax<F, NR>(st, s, j0, first, last, exists, scale2, gid, j0 >= v_lo && j0 + F::BN - 1 <= v_hi);
#pragma unroll
            for (int kk = 0; kk < F::MT; ++kk)
#pragma unroll
                for (int nr = 0; nr < NR; ++nr) fa_pv_b(pb[kk][nr], s[kk][nr]);
        }
        cp_wait1();  // this tile's V (the next K may still be on the way)
        __syncthreads();
        if (warp_rows) {
            // the weights times V: a key m-tile a step, V^T's A fragments transposed out of shared memory, a row tile at
            // a time, folded into the group's as it ends
#pragma unroll
            for (int nr = 0; nr < NR; ++nr) {
#pragma unroll
                for (int md = 0; md < F::MD; ++md)
#pragma unroll
                    for (int c = 0; c < 4; ++c) st.o[nr][md][c] = 0.f;
#pragma unroll
                for (int kk = 0; kk < F::MT; ++kk) {
                    const bf16* vr = sV + (kk * 16 + (lane >> 4) * 8 + (lane & 7)) * F::SROW + ((lane >> 3) & 1) * 8;
#pragma unroll
                    for (int md = 0; md < F::MD; ++md) {
                        unsigned a0, a1, a2, a3;
                        fa_ldsm_t4(a0, a1, a2, a3, vr + md * 16);
                        btb_mma16816(st.o[nr][md][0], st.o[nr][md][1], st.o[nr][md][2], st.o[nr][md][3], a0, a1, a2,
                                     a3, pb[kk][nr][0], pb[kk][nr][1]);
                    }
                }
                fold_tile(nr);
            }
        }
        __syncthreads();  // V's tile consumed
        if (more) {
            issue(false, j0 + F::BN);
        } else {
            cp_commit();
        }
    }
    if (warp_rows) fold_row(true);
}

// ---------------------------------------------------------------------------------------------------------
// the prefill form with the rows as the mma's M (D <= 128; RM): the decode form's arithmetic with the operands' roles
// swapped back - S = Q K^T and O = P V, a warp's 16 rows one m-tile, the keys and the dims its n-tiles. A tensor-core
// output element is its own k-ordered products and accumulator whichever operand holds which factor and wherever in
// the tile it sits (checked bit for bit over wide exponents, cancellations and zero accumulators), and the k orders
// are the decode form's (the dot product's dims permuted as there, the weights' keys in order), so a row is the decode
// form's row. What the roles buy: q is the A operand, its fragments built once and held; a key's B fragments for a
// super-tile are one aligned 16-byte load whose words are the even and odd k-tiles' pairs as the mma takes them (as the
// A operand a fragment was two keys' words interleaved: four register moves an mma, a sixth of the kernel's
// instructions); the weights' A fragments are the scores' C fragments packed in place, no transpose; V's B fragments
// come out of shared memory by ldmatrix.trans, each read once. A row's sum of weights keeps the decode form's order:
// lane tig holds the key classes 2 tig, 2 tig + 1 (class c the keys c, c + 8, .., summed in order), paired in the lane
// and then across tig (xor 1, 2) - the decode form's partners across gid (xor 4, 8, 16).
//
// Block (z, g): PFW warps of 16 rows, 16 PFW / G tokens of key head g (`fa_pf_block`). The tile's state is the
// mma's accumulators, made a half of the dims at a time and folded into the group's (registers) as each half ends - at
// a group's first tile copied - the group's into the row's (`run`) as the walk leaves it. Two barriers a tile: the
// tile's K landed (every warp past the last P . V: V's copies for this tile issued) and its V landed (every warp's
// scores done: the next K's copies issued), each copy given half a tile's compute to land. K and V unpadded and
// swizzled: K's 16-byte chunk c of row r at c ^ 4 (r & 1) (a quarter-warp's fragment loads, two keys' 64 bytes each,
// in distinct banks), V's at c ^ (r & 7) (ldmatrix's eight rows of a chunk in distinct banks).
// ---------------------------------------------------------------------------------------------------------
// the block's shared memory: K's and V's tiles, and - SPL > 1 - each warp's trade with its partner: its keys' scaled,
// masked scores (NT / SPL n-tiles of 4 a lane)
template <class F, int D, int PFW, int SPL>
struct FaRmSmem {
    static constexpr int K = F::BN * D * 2;
    static constexpr int XW = SPL > 1 ? F::BN / 8 / SPL * 4 * 32 : 0;  // a warp's trade, floats
    static constexpr int BYTES = 2 * K + PFW * XW * 4;
};

// SPL warps share a row tile of 16 (SPL 2 at the widest head, where one warp's q, group output and tile accumulators
// pass its registers): warp half wh scores the tile's keys n-tiles [wh NT / 2, ..) and owns the output's dims n-tiles
// [wh OT / 2, ..). The pair trades its scores through shared memory across the barrier that lands the tile's V (the
// block's own, no sync of the pair's), and each then makes the whole tile's softmax - the same max, weights and sum,
// in the decode form's order - and both k-tiles of the P . V, and folds its own dims. Traded as the max and then the
// weights, the pair stood at two syncs of its own a tile, its warps on different schedulers drifting apart between
template <class F, int D, int PFW, int SPL>
__device__ __forceinline__ void attn_flash_prefill_rm(const bf16* __restrict__ q, const bf16* __restrict__ K,
                                                      const bf16* __restrict__ V, bf16* __restrict__ out, int n0,
                                                      int T, int Hq, int Hk, int hs, int rs, float scale, int win,
                                                      const int* __restrict__ tbl, float* __restrict__ run) {
    using SM = FaRmSmem<F, D, PFW, SPL>;
    constexpr int NTH = PFW * 32;
    constexpr int NT = F::BN / 8;   // a tile's key n-tiles
    constexpr int NTW = NT / SPL;   // the warp's
    constexpr int OT = D / 8;       // the output's dim n-tiles
    constexpr int OTW = OT / SPL;   // the warp's
    constexpr int OH = OTW / 2;     // half of them: a P . V and its fold
    constexpr int KK = F::BN / 16;  // the P . V's k-tiles
    constexpr int RG = PFW / SPL;   // the block's row tiles
    static_assert(OH % 2 == 0 && OTW % 8 == 0 && NT % SPL == 0 && PFW % SPL == 0 && SPL <= 2, "the split's shape");
    extern __shared__ __align__(16) unsigned char fa_smem[];
    {
        unsigned have;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(have));
        if (have < SM::BYTES) __trap();
    }
    bf16* const sK = reinterpret_cast<bf16*>(fa_smem);
    bf16* const sV = reinterpret_cast<bf16*>(fa_smem + SM::K);
    const int G = Hq / Hk;
    if (G > RG * 16) __trap();  // a block's rows a whole token's heads at least (`_card_family_ok`)
    const int tpb = RG * 16 / G;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const int wr = warp / SPL, wh = warp % SPL;  // the warp's row tile and its half of it
    // the warp's trade and its partner's (SPL 2): its keys' scores [nt][i][lane]
    [[maybe_unused]] float* const xs = reinterpret_cast<float*>(fa_smem + 2 * SM::K) + warp * SM::XW + lane;
    [[maybe_unused]] const float* const xp = reinterpret_cast<float*>(fa_smem + 2 * SM::K) + (warp ^ 1) * SM::XW + lane;
    int g, t0;
    fa_pf_block(tpb, g, t0);
    // the lane's rows: r = 0 the warp's row gid, r = 1 its row gid + 8 - each a row of out (and of run) and its keys
    int orow[2], first[2], last[2];
    bool exists[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int i = wr * 16 + gid + 8 * r;
        const int t = t0 + i / G;
        exists[r] = i < tpb * G && t < T;
        orow[r] = exists[r] ? t * Hq + g * G + i % G : 0;
        last[r] = exists[r] ? n0 + t : -1;
        first[r] = win > 0 ? max(n0 + t + 1 - win, 0) : 0;
    }
    const int b_hi = n0 + min(t0 + tpb, T) - 1;
    const int b_lo = win > 0 ? max(n0 + t0 + 1 - win, 0) : 0;
    // the keys every row of the warp sees (a tile inside them takes no mask) and those any of them sees (a tile outside
    // them leaves the warp's states as they are: it skips it)
    int v_lo = 0, v_hi = 0x7fffffff, w_lo = 0x7fffffff, w_hi = -1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (!exists[r]) continue;
        v_lo = max(v_lo, first[r]);
        v_hi = min(v_hi, last[r]);
        w_lo = min(w_lo, first[r]);
        w_hi = max(w_hi, last[r]);
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
        v_lo = max(v_lo, __shfl_xor_sync(0xffffffffu, v_lo, o));
        v_hi = min(v_hi, __shfl_xor_sync(0xffffffffu, v_hi, o));
        w_lo = min(w_lo, __shfl_xor_sync(0xffffffffu, w_lo, o));
        w_hi = max(w_hi, __shfl_xor_sync(0xffffffffu, w_hi, o));
    }
    const float scale2 = __fmul_rn(scale, FA_LOG2E);
    // q's A fragments, held: super-tile st's k-tiles 2 st (words x, y of row gid's dims 32 st + 8 tig .. + 7, one load,
    // and row gid + 8's) and 2 st + 1 (words z, w)
    unsigned qa[F::KT][4];
    {
        const bf16* qp0 = q + (size_t)orow[0] * D + tig * 8;
        const bf16* qp1 = q + (size_t)orow[1] * D + tig * 8;
#pragma unroll
        for (int st = 0; st < F::ST; ++st) {
            uint4 u = make_uint4(0u, 0u, 0u, 0u), v = make_uint4(0u, 0u, 0u, 0u);
            if (exists[0]) u = __ldg(reinterpret_cast<const uint4*>(qp0 + st * 32));
            if (exists[1]) v = __ldg(reinterpret_cast<const uint4*>(qp1 + st * 32));
            qa[2 * st][0] = u.x;
            qa[2 * st][1] = v.x;
            qa[2 * st][2] = u.y;
            qa[2 * st][3] = v.y;
            qa[2 * st + 1][0] = u.z;
            qa[2 * st + 1][1] = v.z;
            qa[2 * st + 1][2] = u.w;
            qa[2 * st + 1][3] = v.w;
        }
    }
    // the group's (m, l) and output (go[ot][2 r + e]: row r's dim d0 + 8 ot + 2 tig + e, d0 the warp's first), the
    // row's (m, l) here and its output in `run`. The output is always finite (zeros before any tile), as a fold scaling
    // it by 0 needs
    const int d0 = wh * OTW * 8;
    float gM[2], gL[2], rM[2], rL[2];
    float go[OTW][4];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        gM[r] = rM[r] = NEG_INF;
        gL[r] = rL[r] = 0.f;
    }
#pragma unroll
    for (int ot = 0; ot < OTW; ++ot) go[ot][0] = go[ot][1] = go[ot][2] = go[ot][3] = 0.f;
    // the group's state into the row's, the group's emptied: a row at a time, its `run` values loaded, then folded,
    // then stored (the walk's last fold writes the output from its registers). A row's first group finds nothing of it
    // in `run` and reads nothing; a padding row has no row to fold into. Both rows' values at once were 64 registers
    // beside the group's output and q, and the compiler spilled the walk's own values around every fold
    auto fold_row = [&](bool last_fold) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const bool had = exists[r] && rM[r] != NEG_INF;
            float cO, co;
            fa_fold_ml(rM[r], rL[r], gM[r], gL[r], cO, co);
            gM[r] = NEG_INF;
            gL[r] = 0.f;
            if (!exists[r]) continue;
            float2 v[OTW];
            const float* rp = run + (size_t)orow[r] * D + d0 + 2 * tig;
#pragma unroll
            for (int ot = 0; ot < OTW; ++ot)
                v[ot] = had ? *reinterpret_cast<const float2*>(rp + ot * 8) : make_float2(0.f, 0.f);
#pragma unroll
            for (int ot = 0; ot < OTW; ++ot) {
                v[ot].x = fa_fold_e(v[ot].x, go[ot][2 * r], cO, co);
                v[ot].y = fa_fold_e(v[ot].y, go[ot][2 * r + 1], cO, co);
            }
            if (last_fold) {
                bf16* op = out + (size_t)orow[r] * D + d0 + 2 * tig;
#pragma unroll
                for (int ot = 0; ot < OTW; ++ot)
                    *reinterpret_cast<__nv_bfloat162*>(op + ot * 8) =
                        __floats2bfloat162_rn(fa_out(v[ot].x, rL[r]), fa_out(v[ot].y, rL[r]));
            } else {
                float* wp = run + (size_t)orow[r] * D + d0 + 2 * tig;
#pragma unroll
                for (int ot = 0; ot < OTW; ++ot) *reinterpret_cast<float2*>(wp + ot * 8) = v[ot];
            }
        }
    };
    const bf16* Kg = K + (size_t)g * hs;
    const bf16* Vg = V + (size_t)g * hs;
    const bool warp_rows = w_hi >= 0;
    // a tile's K or V rows into shared memory (a key past the block's last row a zero row), one commit group: thread's
    // 16-byte chunk ch0 of rows r0, r0 + RSTEP, .. - RSTEP even, so a K row's swizzle is the thread's own; V's is where
    // RSTEP is a multiple of 8, and at the widest head (4 rows a step) alternates between two
    constexpr int RSTEP = NTH / F::CH;
    constexpr int VB = RSTEP % 8 == 0 ? 1 : 8 / RSTEP;  // V's swizzles a thread's rows take in turn
    static_assert(NTH % F::CH == 0 && RSTEP % 2 == 0 && (RSTEP % 8 == 0 || 8 % RSTEP == 0) &&
                      F::BN % (RSTEP * VB) == 0,
                  "a thread's chunks are one column");
    const int r0 = threadIdx.x / F::CH, ch0 = threadIdx.x % F::CH;
    bf16* const sk0 = sK + r0 * D + ((ch0 ^ ((r0 & 1) << 2)) << 3);
    bf16* sv[VB];
#pragma unroll
    for (int b = 0; b < VB; ++b) {
        const int r = r0 + b * RSTEP;
        sv[b] = sV + r * D + ((ch0 ^ (r & 7)) << 3);
    }
    auto issue = [&](bool is_v, const bf16* src, int j0) {
#pragma unroll
        for (int it = 0; it < F::BN / RSTEP; ++it) {
            const int j = j0 + r0 + it * RSTEP;
            const bool ok = j <= b_hi;
            const int sl = ok ? (tbl != nullptr ? tbl[j] : j) : 0;
            bf16* dst = is_v ? sv[it % VB] + (it / VB) * VB * RSTEP * D : sk0 + it * RSTEP * D;
            cp16(dst, src + (size_t)sl * rs, ok);
        }
        cp_commit();
    };
    // the lane's fragment addresses: K's row 8 nt + gid at super-tile st's chunk 4 st + tig, swizzled - st's low bit
    // flipped where the row is odd, so two bases by that bit and the rest a constant (the warp's first key n-tile in
    // the bases); V's ldmatrix row kk * 16 + (lane & 7) + 8 ((lane >> 3) & 1), at dim n-tile pair (dn, dn + 1)'s chunk
    // dn + (lane >> 4), swizzled by lane & 7 - four bases by dn & 7 and the rest a constant (the warp's dims in them)
    const bf16* kb[2];
#pragma unroll
    for (int b = 0; b < 2; ++b) kb[b] = sK + (wh * NTW * 8 + gid) * D + ((4 * (b ^ (gid & 1)) + tig) << 3);
    const bf16* vb[4];
    {
        const int y = (lane >> 4) ^ (lane & 7);
        const bf16* vr = sV + ((lane & 7) + ((lane >> 3) & 1) * 8) * D + d0;
#pragma unroll
        for (int b = 0; b < 4; ++b) vb[b] = vr + (((2 * b) ^ y) << 3);
    }
    const int jf = b_lo / F::BN * F::BN;
    issue(false, Kg + ch0 * 8, jf);
    int grp = jf / F::GK;
    for (int j0 = jf; j0 <= b_hi; j0 += F::BN) {
        if (j0 / F::GK != grp) {  // the walk leaves a group: its state into the row's
            if (warp_rows) fold_row(false);
            grp = j0 / F::GK;
        }
        cp_wait0();  // this tile's K
        __syncthreads();
        issue(true, Vg + ch0 * 8, j0);
        const bool mine = warp_rows && j0 <= w_hi && j0 + F::BN - 1 >= w_lo;
        // the tile's state: its scores, then (m, l) and the weights as P . V's A fragments
        unsigned pa[KK][4];
        float tm[2], tl[2];
        // the whole tile's scaled, masked scores (s[nt][2 r + e]: row r's key j0 + 8 nt + 2 tig + e) into the rows'
        // max over the quad holding them, the weights, their sum in the decode form's order, the A fragments
        auto softmax = [&](float (&s)[NT][4]) {
            float mx[2] = {NEG_INF, NEG_INF};
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) mx[i >> 1] = fmaxf(mx[i >> 1], s[nt][i]);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
                mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            }
            float part[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = i >> 1;
                    s[nt][i] = mx[r] == NEG_INF ? 0.f : fa_ex2(__fsub_rn(s[nt][i], mx[r]));
                    part[r][i & 1] = __fadd_rn(part[r][i & 1], s[nt][i]);
                }
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float l = __fadd_rn(part[r][0], part[r][1]);
                l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 1));
                l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 2));
                tm[r] = mx[r];
                tl[r] = l;
            }
#pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                pa[kk][0] = fa_pack(s[2 * kk][0], s[2 * kk][1]);
                pa[kk][1] = fa_pack(s[2 * kk][2], s[2 * kk][3]);
                pa[kk][2] = fa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                pa[kk][3] = fa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            }
        };
        // the warp's keys' scores (sw[nt]: key n-tile kn + nt, kn its first): a pair's, traded through the barrier
        // that lands V (written before it, read after it - no sync of the pair's own), each warp then the whole
        // tile's softmax; a lone warp's, its softmax before the barrier
        [[maybe_unused]] float sw[NTW][4];
        if (mine) {
            const int kn = wh * NTW;
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) sw[nt][0] = sw[nt][1] = sw[nt][2] = sw[nt][3] = 0.f;
#pragma unroll
            for (int st = 0; st < F::ST; ++st) {
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    const uint4 b = *reinterpret_cast<const uint4*>(kb[st & 1] + nt * 8 * D + (st >> 1) * 64);
                    btb_mma16816(sw[nt][0], sw[nt][1], sw[nt][2], sw[nt][3], qa[2 * st][0], qa[2 * st][1],
                                 qa[2 * st][2], qa[2 * st][3], b.x, b.y);
                    btb_mma16816(sw[nt][0], sw[nt][1], sw[nt][2], sw[nt][3], qa[2 * st + 1][0], qa[2 * st + 1][1],
                                 qa[2 * st + 1][2], qa[2 * st + 1][3], b.z, b.w);
                }
            }
            // scaled to log2 units and masked
            if (j0 >= v_lo && j0 + F::BN - 1 <= v_hi) {
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) sw[nt][i] = __fmul_rn(sw[nt][i], scale2);
            } else {
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int r = i >> 1, j = j0 + (kn + nt) * 8 + 2 * tig + (i & 1);
                        const bool ok = exists[r] && j >= first[r] && j <= last[r];
                        sw[nt][i] = ok ? __fmul_rn(sw[nt][i], scale2) : NEG_INF;
                    }
            }
            if constexpr (SPL > 1) {
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) xs[(nt * 4 + i) * 32] = sw[nt][i];
            } else {
                softmax(sw);
            }
        }
        cp_wait0();  // this tile's V
        __syncthreads();
        if (j0 + F::BN <= b_hi) issue(false, Kg + ch0 * 8, j0 + F::BN);
        if (mine) {
            if constexpr (SPL > 1) {
                // the whole tile's scores in key order: the warp's own and its partner's
                float s[NT][4];
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt)
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float other = xp[(nt * 4 + i) * 32];
                        s[nt][i] = wh == 0 ? sw[nt][i] : other;
                        s[NTW + nt][i] = wh == 0 ? other : sw[nt][i];
                    }
                softmax(s);
            }
            // each row's factors once; where every row of the warp keeps its group max the group side's factor is 1
            // and its product left out (the same bits). The warp's verdict by shuffles across the rows (gid): a vote
            // here, under branches the compiler cannot see are the warp's alike, was a call into the runtime's
            // convergence helper. A group's first tile is a fold into the emptied state like any other (its factor
            // 0 on the group side) - copied instead, the copy's branch left the group's output in other registers
            // than the fold's, and the compiler moved all 64 of them back at every tile
            float cO[2], co[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) fa_fold_ml(gM[r], gL[r], tm[r], tl[r], cO[r], co[r]);
            int scale_go = cO[0] != 1.f || cO[1] != 1.f;
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) scale_go |= __shfl_xor_sync(0xffffffffu, scale_go, o);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float o[OH][4];
#pragma unroll
                for (int ot = 0; ot < OH; ++ot) o[ot][0] = o[ot][1] = o[ot][2] = o[ot][3] = 0.f;
#pragma unroll
                for (int kk = 0; kk < KK; ++kk) {
#pragma unroll
                    for (int p = 0; p < OH / 2; ++p) {
                        const int dn = h * OH + 2 * p;  // the pair's first dim n-tile: chunk dn
                        unsigned b0, b1, b2, b3;
                        fa_ldsm_t4(b0, b1, b2, b3, vb[(dn & 7) >> 1] + kk * 16 * D + (dn >> 3) * 64);
                        btb_mma16816(o[2 * p][0], o[2 * p][1], o[2 * p][2], o[2 * p][3], pa[kk][0], pa[kk][1],
                                     pa[kk][2], pa[kk][3], b0, b1);
                        btb_mma16816(o[2 * p + 1][0], o[2 * p + 1][1], o[2 * p + 1][2], o[2 * p + 1][3], pa[kk][0],
                                     pa[kk][1], pa[kk][2], pa[kk][3], b2, b3);
                    }
                }
                if (scale_go) {
#pragma unroll
                    for (int ot = 0; ot < OH; ++ot)
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                            go[h * OH + ot][c] = fa_fold_e(go[h * OH + ot][c], o[ot][c], cO[c >> 1], co[c >> 1]);
                } else {
#pragma unroll
                    for (int ot = 0; ot < OH; ++ot)
#pragma unroll
                        for (int c = 0; c < 4; ++c)
                            go[h * OH + ot][c] = __fmaf_rn(o[ot][c], co[c >> 1], go[h * OH + ot][c]);
                }
            }
        }
    }
    if (warp_rows) fold_row(true);
}

// ---------------------------------------------------------------------------------------------------------
// the property the rows-as-M prefill stands on, checked on the card that runs it (the host, once at bind): a
// tensor-core output element is its own k-ordered products and accumulator, whichever operand holds which factor and
// wherever in the tile it sits. A warp a trial: A [16 x 16], B [16 x 8], C [16 x 8] hashed from (trial, index) - by
// trial, small exponents, wide ones, cancelling pairs (A's k-halves equal, B's opposite) or a zero accumulator -
// D = A B + C as the mma takes it, then D^T = B^T A^T + C^T with B^T's rows at the top of the m-tile and again at its
// bottom (the other rows filler); each element's bits against D's, the mismatches counted into *bad. On the 4070 Ti
// (Ada) none, over 25 million elements; a card that counts any takes the decode form's orientation for its prefill.
// ---------------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned fa_hash(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// a trial's bf16 value at (which matrix, index): sign and 7 mantissa bits hashed, the exponent within +-span of 1
__device__ __forceinline__ unsigned short fa_roles_bf(unsigned trial, unsigned which, unsigned idx, int span) {
    const unsigned h = fa_hash(trial * 0x9e3779b9u ^ (which << 24) ^ idx);
    const unsigned e = 127u + (unsigned)((int)(h >> 8) % (2 * span + 1) - span);
    return (unsigned short)(((h & 1u) << 15) | (e << 7) | ((h >> 1) & 0x7fu));
}

extern "C" __global__ void __launch_bounds__(256) btb_mma_roles(int* __restrict__ bad) {
    __shared__ float d1s[8][16][8];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, tig = lane & 3;
    const unsigned trial = blockIdx.x * 8 + w, mode = trial & 3;
    const int span = mode == 0 ? 3 : 20;
    auto a_at = [&](int m, int k) -> unsigned short {
        return fa_roles_bf(trial, 0, m * 16 + (mode == 2 ? (k & 7) : k), span);  // cancelling: k-halves alike
    };
    auto b_at = [&](int k, int n) -> unsigned short {
        const unsigned short v = fa_roles_bf(trial, 1, (mode == 2 ? (k & 7) : k) * 8 + n, span);
        return mode == 2 && k >= 8 ? (unsigned short)(v ^ 0x8000u) : v;  // .. and B's opposite
    };
    auto c_at = [&](int m, int n) -> float {
        if (mode == 3) return 0.f;
        const unsigned h = fa_hash(trial * 0x9e3779b9u ^ (2u << 24) ^ (unsigned)(m * 8 + n));
        const unsigned e = 127u + (unsigned)((int)(h >> 8) % 61 - 30);
        return __uint_as_float(((h & 1u) << 31) | (e << 23) | (fa_hash(h) & 0x7fffffu));
    };
    auto pk = [](unsigned short lo, unsigned short hi) -> unsigned { return (unsigned)lo | ((unsigned)hi << 16); };
    // D as the mma takes it
    {
        float d0 = c_at(gid, 2 * tig), d1 = c_at(gid, 2 * tig + 1), d2 = c_at(gid + 8, 2 * tig),
              d3 = c_at(gid + 8, 2 * tig + 1);
        btb_mma16816(d0, d1, d2, d3, pk(a_at(gid, 2 * tig), a_at(gid, 2 * tig + 1)),
                     pk(a_at(gid + 8, 2 * tig), a_at(gid + 8, 2 * tig + 1)),
                     pk(a_at(gid, 2 * tig + 8), a_at(gid, 2 * tig + 9)),
                     pk(a_at(gid + 8, 2 * tig + 8), a_at(gid + 8, 2 * tig + 9)),
                     pk(b_at(2 * tig, gid), b_at(2 * tig + 1, gid)), pk(b_at(2 * tig + 8, gid), b_at(2 * tig + 9, gid)));
        d1s[w][gid][2 * tig] = d0;
        d1s[w][gid][2 * tig + 1] = d1;
        d1s[w][gid + 8][2 * tig] = d2;
        d1s[w][gid + 8][2 * tig + 1] = d3;
    }
    __syncwarp();
    int miss = 0;
#pragma unroll
    for (int top = 0; top < 2; ++top) {
        const int r0 = top ? 0 : 8;  // the m-tile rows B^T takes
        auto at = [&](int m, int k) -> unsigned short {
            return m >= r0 && m < r0 + 8 ? b_at(k, m - r0) : fa_roles_bf(trial, 3, m * 16 + k, span);
        };
        const unsigned a0 = pk(at(gid, 2 * tig), at(gid, 2 * tig + 1));
        const unsigned a1 = pk(at(gid + 8, 2 * tig), at(gid + 8, 2 * tig + 1));
        const unsigned a2 = pk(at(gid, 2 * tig + 8), at(gid, 2 * tig + 9));
        const unsigned a3 = pk(at(gid + 8, 2 * tig + 8), at(gid + 8, 2 * tig + 9));
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int n = nt * 8 + gid;  // B'[k][n'] = A[n'][k]
            const int r = top ? gid : gid + 8;
            float c[4] = {0.f, 0.f, 0.f, 0.f};
            c[top ? 0 : 2] = c_at(nt * 8 + 2 * tig, r - r0);
            c[top ? 1 : 3] = c_at(nt * 8 + 2 * tig + 1, r - r0);
            btb_mma16816(c[0], c[1], c[2], c[3], a0, a1, a2, a3, pk(a_at(n, 2 * tig), a_at(n, 2 * tig + 1)),
                         pk(a_at(n, 2 * tig + 8), a_at(n, 2 * tig + 9)));
            const float e0 = top ? c[0] : c[2], e1 = top ? c[1] : c[3];
            miss += __float_as_uint(e0) != __float_as_uint(d1s[w][nt * 8 + 2 * tig][r - r0]);
            miss += __float_as_uint(e1) != __float_as_uint(d1s[w][nt * 8 + 2 * tig + 1][r - r0]);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) miss += __shfl_xor_sync(0xffffffffu, miss, o);
    if (lane == 0 && miss) atomicAdd(bad, miss);
}

// the prefill's two kernels a head width, the same arguments and launch shape (FaPick: 16 PFW / SPL rows a block, as
// 8 PFW NR): btb_attn_flash_prefill_d* the rows as the mma's M, and btb_attn_flash_prefill_kq_d* the decode form's
// own orientation - the keys the M - for a card whose tensor cores fail `btb_mma_roles` (the host's pick:
// `_Cuda.flash_prefill_kernel`). Each bound names the blocks an SM it is built for (FaPick::PFB, the fallback one):
// with the thread count alone, ptxas picked a register cap of its own, which spilled a d64 form on one target and not
// another (the fallback's held to 168 registers: 211 without the cap, none). The fallback's d128 form is at the
// ceiling itself (q re-read each tile, the compiler's prefetch of it, the group state and the scores): some bytes
// spilled on Ada and Blackwell - the price of the fallback's orientation
#define ATTN_FLASH_PF_ARGS                                                                                   \
    const bf16* __restrict__ q, const bf16* __restrict__ K, const bf16* __restrict__ V,                      \
        bf16* __restrict__ out, int n0, int T, int Hq, int Hk, int hs, int rs, float scale, int win,         \
        const int* __restrict__ tbl, float* __restrict__ run
#define ATTN_FLASH_PREFILL_K(D)                                                                              \
    extern "C" __global__ void __launch_bounds__(FaPick<D>::PFW * 32, FaPick<D>::PFB)                        \
        btb_attn_flash_prefill_d##D(ATTN_FLASH_PF_ARGS) {                                                    \
        attn_flash_prefill_rm<FaTile<D, FaPick<D>::BN, FaPick<D>::GT>, D, FaPick<D>::PFW, FaPick<D>::SPL>(   \
            q, K, V, out, n0, T, Hq, Hk, hs, rs, scale, win, tbl, run);                                      \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(FaPick<D>::PFW * 32, 1)                                     \
        btb_attn_flash_prefill_kq_d##D(ATTN_FLASH_PF_ARGS) {                                                 \
        attn_flash_prefill<FaTile<D, FaPick<D>::BN, FaPick<D>::GT>, D, FaPick<D>::PFW, FaPick<D>::NR>(       \
            q, K, V, out, n0, T, Hq, Hk, hs, rs, scale, win, tbl, run);                                      \
    }
ATTN_FLASH_PREFILL_K(64)
ATTN_FLASH_PREFILL_K(128)
ATTN_FLASH_PREFILL_K(256)
