// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
// Each GGUF type a linear multiplies as stored (btb.kinds.stored_quants()) as one `decode` of weight 32*ib + lane
// of a 256-weight unit (a k-quant or i-quant superblock, or eight 32-weight blocks), ported from native/src/gemv.rs's
// scalar definitions; btb_gemv_quant.cuh's kernels read the blocks through them. Plain C++ under the CUDA
// qualifiers, so tests/kernels/test_card_decode.py compiles them for the host and holds them to gguf's numbers.

#pragma once

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

// a lattice type's int8 codebook (entries x values) and the shared 128-entry sign table
struct QTab {
    const int8_t* grid;
    const uint8_t* ks;
};

__device__ __forceinline__ uint32_t qb(const uint8_t* p) { return __ldcs(p); }
__device__ __forceinline__ uint32_t q16(const uint8_t* p) { return qb(p) | (qb(p + 1) << 8); }
__device__ __forceinline__ uint32_t q32(const uint8_t* p) { return q16(p) | (q16(p + 2) << 16); }
__device__ __forceinline__ float qhalf(uint32_t bits) { return __half2float(__ushort_as_half((unsigned short)bits)); }
__device__ __forceinline__ float qf16(const uint8_t* p) { return qhalf(q16(p)); }
__device__ __forceinline__ float qsign(uint32_t byte, int b) { return (byte >> b) & 1 ? -1.f : 1.f; }

__constant__ int8_t IQ4_KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// llama.cpp's get_scale_min_k4: the 6-bit scale and min of sub-block j from the 12 packed bytes at s
__device__ __forceinline__ void q_scale_min_k4(const uint8_t* s, int j, float& sc, float& mn) {
    if (j < 4) {
        sc = (float)(qb(s + j) & 63);
        mn = (float)(qb(s + j + 4) & 63);
    } else {
        sc = (float)((qb(s + j + 4) & 0xF) | ((qb(s + j - 4) >> 6) << 4));
        mn = (float)((qb(s + j + 4) >> 4) | ((qb(s + j) >> 6) << 4));
    }
}

// each type: BW weights a block of BB bytes, `decode` weight 32 * ib + lane of the 256-weight unit at u
#define QTYPE(NAME, BW_, BB_) \
    struct NAME {             \
        static constexpr int BW = BW_, BB = BB_, UB = 256 / BW_ * BB_;
#define QEND \
    }        \
    ;

QTYPE(q40, 32, 18)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const uint8_t* b = u + ib * BB;
    const uint32_t c = qb(b + 2 + (lane & 15));
    return qf16(b) * (float)((int)(lane < 16 ? c & 0xF : c >> 4) - 8);
}
QEND

QTYPE(q41, 32, 20)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const uint8_t* b = u + ib * BB;
    const uint32_t c = qb(b + 4 + (lane & 15));
    return fmaf(qf16(b), (float)(lane < 16 ? c & 0xF : c >> 4), qf16(b + 2));
}
QEND

QTYPE(q80, 32, 34)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const uint8_t* b = u + ib * BB;
    return qf16(b) * (float)(int8_t)qb(b + 2 + lane);
}
QEND

QTYPE(iq4nl, 32, 18)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const uint8_t* b = u + ib * BB;
    const uint32_t c = qb(b + 2 + (lane & 15));
    return qf16(b) * (float)IQ4_KV[lane < 16 ? c & 0xF : c >> 4];
}
QEND

QTYPE(q2k, 256, 84)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const int h = ib >> 2, j = ib & 3;
    const uint32_t sc = qb(u + h * 8 + 2 * j + (lane >> 4));
    const float q = (float)((qb(u + 16 + h * 32 + lane) >> (2 * j)) & 3);
    return fmaf(q, qf16(u + 80) * (float)(sc & 0xF), -(qf16(u + 82) * (float)(sc >> 4)));
}
QEND

QTYPE(q3k, 256, 110)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const int h = ib >> 2, j = ib & 3, i = h * 8 + 2 * j + (lane >> 4);
    // scale i of the 16 signed 6-bit scales packed in the 12 bytes at 96 (llama.cpp's kmask recombination)
    const uint32_t lo = (qb(u + 96 + (i & 7)) >> (4 * (i >> 3))) & 0xF;
    const uint32_t hi = (qb(u + 104 + (i & 3)) >> (2 * (i >> 2))) & 3;
    const float dsc = qf16(u + 108) * (float)((int)(lo | (hi << 4)) - 32);
    const uint32_t q2 = (qb(u + 32 + h * 32 + lane) >> (2 * j)) & 3;
    const uint32_t bit = (qb(u + lane) >> (h * 4 + j)) & 1;
    return fmaf((float)(q2 + 4 * bit), dsc, -4.f * dsc);
}
QEND

QTYPE(q4k, 256, 144)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    float sc, mn;
    q_scale_min_k4(u + 4, ib, sc, mn);
    const uint32_t c = qb(u + 16 + 32 * (ib >> 1) + lane);
    return fmaf((float)(ib & 1 ? c >> 4 : c & 0xF), qf16(u) * sc, -(qf16(u + 2) * mn));
}
QEND

QTYPE(q5k, 256, 176)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    float sc, mn;
    q_scale_min_k4(u + 4, ib, sc, mn);
    const uint32_t c = qb(u + 48 + 32 * (ib >> 1) + lane);
    const uint32_t q = (ib & 1 ? c >> 4 : c & 0xF) | (((qb(u + 16 + lane) >> ib) & 1) << 4);
    return fmaf((float)q, qf16(u) * sc, -(qf16(u + 2) * mn));
}
QEND

QTYPE(q6k, 256, 210)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const int h = ib >> 2, t = ib & 3;
    const uint32_t l = qb(u + 64 * h + lane + 32 * (t & 1));
    const uint32_t hb = qb(u + 128 + 32 * h + lane);
    const uint32_t c = (t < 2 ? l & 0xF : l >> 4) | (((hb >> (2 * t)) & 3) << 4);
    const float dsc = qf16(u + 208) * (float)(int8_t)qb(u + 192 + 8 * h + (lane >> 4) + 2 * t);
    return fmaf((float)c, dsc, -32.f * dsc);
}
QEND

QTYPE(iq4xs, 256, 136)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab) {
    const uint32_t ls = ((qb(u + 4 + (ib >> 1)) >> (4 * (ib & 1))) & 0xF) | (((q16(u + 2) >> (2 * ib)) & 3) << 4);
    const uint32_t c = qb(u + 8 + 16 * ib + (lane & 15));
    return qf16(u) * (float)((int)ls - 32) * (float)IQ4_KV[lane < 16 ? c & 0xF : c >> 4];
}
QEND

QTYPE(iq3xxs, 256, 98)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const uint32_t aux = q32(u + 66 + 4 * ib);
    const float db = qf16(u) * (0.5f + (float)(aux >> 28)) * 0.5f;
    const int l = lane >> 3, m = lane & 7;
    const uint32_t idx = qb(u + 2 + 8 * ib + 2 * l + (m >= 4 ? 1 : 0));
    const float g = (float)__ldg(t.grid + idx * 4 + (m & 3));
    return db * g * qsign(__ldg(t.ks + ((aux >> (7 * l)) & 127)), m);
}
QEND

QTYPE(iq3s, 256, 110)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const int qbi = 8 * ib + (lane >> 2);
    const uint32_t idx = qb(u + 2 + qbi) | (((qb(u + 66 + (qbi >> 3)) >> (qbi & 7)) & 1) << 8);
    const uint32_t sc4 = (qb(u + 106 + (ib >> 1)) >> (4 * (ib & 1))) & 0xF;
    const float db = qf16(u) * (float)(1 + 2 * sc4);
    const float g = (float)__ldg(t.grid + idx * 4 + (lane & 3));
    return db * g * qsign(qb(u + 74 + 4 * ib + (lane >> 3)), lane & 7);
}
QEND

QTYPE(iq2xxs, 256, 66)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const uint8_t* grp = u + 2 + 8 * ib;
    const uint32_t w1 = q32(grp + 4);
    const float db = qf16(u) * (0.5f + (float)(w1 >> 28)) * 0.25f;
    const int l = lane >> 3, j = lane & 7;
    const float g = (float)__ldg(t.grid + qb(grp + l) * 8 + j);
    return db * g * qsign(__ldg(t.ks + ((w1 >> (7 * l)) & 127)), j);
}
QEND

QTYPE(iq2xs, 256, 74)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const int l = lane >> 3, j = lane & 7, grp = 4 * ib + l;
    const uint32_t v = q16(u + 2 + 2 * grp);
    const int sci = grp >> 1;
    const uint32_t sc4 = (qb(u + 66 + (sci >> 1)) >> (4 * (sci & 1))) & 0xF;
    const float db = qf16(u) * (0.5f + (float)sc4) * 0.25f;
    const float g = (float)__ldg(t.grid + (v & 511) * 8 + j);
    return db * g * qsign(__ldg(t.ks + (v >> 9)), j);
}
QEND

QTYPE(iq2s, 256, 82)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const int l = lane >> 3, j = lane & 7, grp = 4 * ib + l;
    const uint32_t idx = qb(u + 2 + grp) | (((qb(u + 66 + (grp >> 2)) >> (2 * (grp & 3))) & 3) << 8);
    const int b = grp >> 1;
    const uint32_t sc4 = (qb(u + 74 + (b >> 1)) >> (4 * (b & 1))) & 0xF;
    const float db = qf16(u) * (0.5f + (float)sc4) * 0.25f;
    const float g = (float)__ldg(t.grid + idx * 8 + j);
    return db * g * qsign(qb(u + 34 + grp), j);
}
QEND

QTYPE(iq1s, 256, 50)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const int l = lane >> 3, j = lane & 7;
    const uint32_t qh = q16(u + 34 + 2 * ib);
    const uint32_t idx = qb(u + 2 + 4 * ib + l) | (((qh >> (3 * l)) & 7) << 8);
    const float dl = qf16(u) * (float)(2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -0.125f : 0.125f;
    return dl * ((float)__ldg(t.grid + idx * 8 + j) + delta);
}
QEND

QTYPE(iq1m, 256, 56)
__device__ static float decode(const uint8_t* u, int ib, int lane, QTab t) {
    const uint32_t s0 = q16(u + 48), s1 = q16(u + 50), s2 = q16(u + 52), s3 = q16(u + 54);
    const float d = qhalf((s0 >> 12) | ((s1 >> 12) << 4) | ((s2 >> 12) << 8) | ((s3 >> 12) << 12));
    const int jj = 4 * ib + (lane >> 3), e = lane & 7;
    const uint32_t nib = (qb(u + 32 + (jj >> 1)) >> (4 * (jj & 1))) & 0xF;
    const uint32_t idx = qb(u + jj) | ((nib & 7) << 8);
    const float delta = nib & 8 ? -0.125f : 0.125f;
    const int si = 2 * ib + (lane >> 4);
    const uint32_t sw = si < 8 ? (si < 4 ? s0 : s1) : (si < 12 ? s2 : s3);  // sub-scale si: word si >> 2
    const float dl = d * (float)(2 * ((sw >> (3 * (si & 3))) & 7) + 1);
    return dl * ((float)__ldg(t.grid + idx * 8 + e) + delta);
}
QEND
