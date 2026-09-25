// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The pick of a token from a row of logits on the CPU: greedy (the argmax), or a draw under a temperature
//! with top-k and top-p, the thresholds found by radix select over histograms (three levels of a float's
//! ordered bits, no sort), the masses in 64-bit fixed point, the draw the argmax of the kept tokens' logits
//! plus Gumbel noise hashed from the row's key and the token (the Metal kernel's formula: a draw only moves
//! when two candidates tie to the ulp). Deterministic for a (key, row); rows across the thread pool. The NEON
//! and AVX2 passes pick bit-for-bit what the scalar ones pick (an AVX-512 CPU runs the AVX2 passes).

use crate::codes::*;
use crate::gemv::Isa;
use rayon::prelude::*;

const BINS: usize = 2048;
const MASS_SCALE: f32 = 1_073_741_824.0; // 2^30 a unit of mass

/// the ordered-uint image of a float: monotone in the value (NaN excluded)
#[inline(always)]
fn ordered(f: f32) -> u32 {
    let u = f.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// exp(t) for t <= 0 to a few ulps: t = n ln2 + r with ln2 in two parts (n's product with the high part exact),
/// e^r through its degree-7 series (|r| <= 0.35: the next term below 1e-8); 0 below -87
#[inline(always)]
fn fast_exp(t: f32) -> f32 {
    if t < -87.0 {
        return 0.0;
    }
    let n = (t * std::f32::consts::LOG2_E).round();
    let r = t - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668
                        + r * (0.008_333_334 + r * (0.001_388_889 + r * 0.000_198_412_7))))));
    let e = (n as i32 + 127).clamp(1, 254);
    f32::from_bits((e as u32) << 23) * p
}

#[derive(Clone, Copy)]
struct Cfg {
    inv_t: f32,
    top_k: u32,
    top_p: f32,
}

fn argmax(row: &[f32]) -> u32 {
    argmax_from(row, 0, f32::NEG_INFINITY, 0)
}

/// the lowest-index maximum over `row[from..]`, carried on from `best` at `bi`
#[inline(always)]
fn argmax_from(row: &[f32], from: usize, mut best: f32, mut bi: usize) -> u32 {
    for (i, &v) in row.iter().enumerate().skip(from) {
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi as u32
}

/// the scaled logits and their maximum (a NaN never wins it)
fn scale_max(x: &[f32], inv_t: f32, s: &mut Vec<f32>) -> f32 {
    s.clear();
    s.extend(x.iter().map(|&a| a * inv_t));
    s.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// a token's mass in fixed point: none under `floor` or 21 nats below the top (a 2^-30 chance)
#[inline(always)]
fn mass(a: f32, m: f32, floor: u32) -> u64 {
    if ordered(a) >= floor && a - m >= -21.0 {
        (fast_exp(a - m) * MASS_SCALE) as u64
    } else {
        0
    }
}

fn masses(s: &[f32], m: f32, floor: u32, w: &mut Vec<u64>) {
    w.clear();
    w.extend(s.iter().map(|&a| mass(a, m, floor)));
}

/// token `i`'s logit plus its Gumbel noise under `key`
#[inline(always)]
fn gumbel(key: u64, i: usize, a: f32) -> f32 {
    let mut h = key ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 32;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 29;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 32;
    // 23 bits: the top of a 24-bit range rounds to 1.0 in f32, an infinite Gumbel that wins the row
    let uf = ((h >> 41) as f32 + 0.5) * (1.0 / 8_388_608.0);
    a - (-uf.ln()).ln()
}

/// the race over `s[from..]`, carried on from `best` at `bi`: a token under `floor` or 21 nats under
/// the top (a 2^-30 chance) is left out
#[inline(always)]
fn draw_from(
    s: &[f32],
    m: f32,
    floor: u32,
    key: u64,
    from: usize,
    mut best: f32,
    mut bi: usize,
) -> (f32, usize) {
    for (i, &a) in s.iter().enumerate().skip(from) {
        if ordered(a) < floor || a - m < -21.0 {
            continue;
        }
        let v = gumbel(key, i, a);
        if v > best {
            best = v;
            bi = i;
        }
    }
    (best, bi)
}

fn draw(s: &[f32], m: f32, floor: u32, key: u64) -> u32 {
    draw_from(s, m, floor, key, 0, f32::NEG_INFINITY, 0).1 as u32
}

// NEON: the passes over a whole row four lanes at a time, each lane the scalar operation (no fused
// multiply-add; max as fmaxnm, which is order-free), so a pick is bit-for-bit the scalar one. The
// radix selects scatter into histograms and stay shared.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ordered_neon(a: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::uint32x4_t {
    use std::arch::aarch64::*;
    let u = vreinterpretq_u32_f32(a);
    let neg = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_u32(u)));
    veorq_u32(u, vorrq_u32(neg, vdupq_n_u32(0x8000_0000)))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fast_exp_neon(t: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let n = vrndaq_f32(vmulq_f32(t, vdupq_n_f32(std::f32::consts::LOG2_E)));
    let r = vsubq_f32(
        vsubq_f32(t, vmulq_f32(n, vdupq_n_f32(0.693_145_75))),
        vmulq_f32(n, vdupq_n_f32(1.428_606_8e-6)),
    );
    let mut p = vmulq_f32(r, vdupq_n_f32(0.000_198_412_7));
    for c in [
        0.001_388_889f32,
        0.008_333_334,
        0.041_666_668,
        0.166_666_67,
        0.5,
        1.0,
    ] {
        p = vmulq_f32(r, vaddq_f32(vdupq_n_f32(c), p));
    }
    p = vaddq_f32(vdupq_n_f32(1.0), p);
    let e = vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127));
    let e = vminq_s32(vmaxq_s32(e, vdupq_n_s32(1)), vdupq_n_s32(254));
    let y = vmulq_f32(vreinterpretq_f32_s32(vshlq_n_s32::<23>(e)), p);
    vbslq_f32(vcltq_f32(t, vdupq_n_f32(-87.0)), vdupq_n_f32(0.0), y)
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
unsafe fn argmax_neon(row: &[f32]) -> u32 {
    use std::arch::aarch64::*;
    let n = row.len();
    if n > u32::MAX as usize {
        return argmax(row);
    }
    let p = row.as_ptr();
    // sixteen lanes, each the lowest-index maximum of its own column of the row
    let mut best = [vdupq_n_f32(f32::NEG_INFINITY); 4];
    let mut idx = [vdupq_n_u32(0); 4];
    let base: [u32; 4] = [0, 1, 2, 3];
    let lane = vld1q_u32(base.as_ptr());
    let mut cur = [0u32, 4, 8, 12].map(|o| vaddq_u32(lane, vdupq_n_u32(o)));
    let mut i = 0;
    while i + 16 <= n {
        for k in 0..4 {
            let v = vld1q_f32(p.add(i + 4 * k));
            let gt = vcgtq_f32(v, best[k]);
            best[k] = vbslq_f32(gt, v, best[k]);
            idx[k] = vbslq_u32(gt, cur[k], idx[k]);
            cur[k] = vaddq_u32(cur[k], vdupq_n_u32(16));
        }
        i += 16;
    }
    let (mut bv, mut bx) = ([0.0f32; 16], [0u32; 16]);
    for k in 0..4 {
        vst1q_f32(bv.as_mut_ptr().add(4 * k), best[k]);
        vst1q_u32(bx.as_mut_ptr().add(4 * k), idx[k]);
    }
    // no lane holds a NaN; -0 == +0 here as in the scalar compare, so the first zero wins
    let m = bv
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let bi = (0..16)
        .filter(|&l| bv[l] == m)
        .map(|l| bx[l])
        .min()
        .unwrap_or(0);
    argmax_from(row, i, m, bi as usize)
}

#[cfg(target_arch = "aarch64")]
unsafe fn scale_max_neon(x: &[f32], inv_t: f32, s: &mut Vec<f32>) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    s.clear();
    s.reserve(n);
    let (src, dst) = (x.as_ptr(), s.as_mut_ptr());
    let it = vdupq_n_f32(inv_t);
    let mut mx = [vdupq_n_f32(f32::NEG_INFINITY); 4];
    let mut i = 0;
    while i + 16 <= n {
        for (k, acc) in mx.iter_mut().enumerate() {
            let v = vmulq_f32(vld1q_f32(src.add(i + 4 * k)), it);
            vst1q_f32(dst.add(i + 4 * k), v);
            *acc = vmaxnmq_f32(*acc, v);
        }
        i += 16;
    }
    while i + 4 <= n {
        let v = vmulq_f32(vld1q_f32(src.add(i)), it);
        vst1q_f32(dst.add(i), v);
        mx[0] = vmaxnmq_f32(mx[0], v);
        i += 4;
    }
    let mut m = vmaxnmvq_f32(vmaxnmq_f32(
        vmaxnmq_f32(mx[0], mx[1]),
        vmaxnmq_f32(mx[2], mx[3]),
    ));
    while i < n {
        let v = *src.add(i) * inv_t;
        *dst.add(i) = v;
        m = m.max(v);
        i += 1;
    }
    s.set_len(n);
    m
}

#[cfg(target_arch = "aarch64")]
unsafe fn masses_neon(s: &[f32], m: f32, floor: u32, w: &mut Vec<u64>) {
    use std::arch::aarch64::*;
    let n = s.len();
    w.clear();
    w.reserve(n);
    let (src, dst) = (s.as_ptr(), w.as_mut_ptr());
    let (mv, fl) = (vdupq_n_f32(m), vdupq_n_u32(floor));
    let mut i = 0;
    while i + 4 <= n {
        let a = vld1q_f32(src.add(i));
        let t = vsubq_f32(a, mv);
        let keep = vandq_u32(
            vcgeq_u32(ordered_neon(a), fl),
            vcgeq_f32(t, vdupq_n_f32(-21.0)),
        );
        let e = vbslq_f32(keep, fast_exp_neon(t), vdupq_n_f32(0.0));
        // f32 to f64 is exact and the fixed-point convert scales by 2^30 before truncating, as
        // `(e * MASS_SCALE) as u64` does, saturating alike
        vst1q_u64(
            dst.add(i),
            vcvtq_n_u64_f64::<30>(vcvt_f64_f32(vget_low_f32(e))),
        );
        vst1q_u64(dst.add(i + 2), vcvtq_n_u64_f64::<30>(vcvt_high_f64_f32(e)));
        i += 4;
    }
    while i < n {
        *dst.add(i) = mass(*src.add(i), m, floor);
        i += 1;
    }
    w.set_len(n);
}

#[cfg(target_arch = "aarch64")]
unsafe fn draw_neon(s: &[f32], m: f32, floor: u32, key: u64) -> u32 {
    use std::arch::aarch64::*;
    let n = s.len();
    let p = s.as_ptr();
    let (mv, fl, lim) = (vdupq_n_f32(m), vdupq_n_u32(floor), vdupq_n_f32(-21.0));
    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
    let mut i = 0;
    while i + 16 <= n {
        let mut skip = vdupq_n_u32(u32::MAX);
        for k in 0..4 {
            let a = vld1q_f32(p.add(i + 4 * k));
            let out = vorrq_u32(
                vcltq_u32(ordered_neon(a), fl),
                vcltq_f32(vsubq_f32(a, mv), lim),
            );
            skip = vandq_u32(skip, out);
        }
        // a lane some token of the sixteen stays in: race them in order, as the scalar loop does
        if vminvq_u32(skip) == 0 {
            (best, bi) = draw_from(&s[..i + 16], m, floor, key, i, best, bi);
        }
        i += 16;
    }
    draw_from(s, m, floor, key, i, best, bi).1 as u32
}

// AVX2: the NEON passes' shape eight lanes at a time, each lane the scalar operation - no fused multiply-add,
// `f32::round`'s ties away from zero rebuilt from a truncate (AVX2 rounds ties to even), the unsigned compares
// done signed on sign-flipped operands - so a pick is bit-for-bit the scalar one.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn ordered_avx2(a: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let u = _mm256_castps_si256(a);
    let neg = _mm256_srai_epi32::<31>(u);
    _mm256_xor_si256(u, _mm256_or_si256(neg, _mm256_set1_epi32(i32::MIN)))
}

/// lanes where `a < b` as unsigned 32-bit
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lt_u32_avx2(
    a: std::arch::x86_64::__m256i,
    b: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let s = _mm256_set1_epi32(i32::MIN);
    _mm256_cmpgt_epi32(_mm256_xor_si256(b, s), _mm256_xor_si256(a, s))
}

/// `f32::round` per lane: the truncation, one step away from zero where the dropped part is at least a half
/// (x - trunc(x) is exact; a lane with less keeps its truncation, sign of zero included)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn round_away_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let t = _mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(x);
    let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
    let half =
        _mm256_cmp_ps::<_CMP_GE_OQ>(_mm256_and_ps(_mm256_sub_ps(x, t), abs), _mm256_set1_ps(0.5));
    let step = _mm256_or_ps(_mm256_set1_ps(1.0), _mm256_andnot_ps(abs, x)); // 1 with x's sign
    _mm256_blendv_ps(t, _mm256_add_ps(t, step), half)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fast_exp_avx2(t: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let n = round_away_avx2(_mm256_mul_ps(t, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm256_sub_ps(
        _mm256_sub_ps(t, _mm256_mul_ps(n, _mm256_set1_ps(0.693_145_75))),
        _mm256_mul_ps(n, _mm256_set1_ps(1.428_606_8e-6)),
    );
    let mut p = _mm256_mul_ps(r, _mm256_set1_ps(0.000_198_412_7));
    for c in [
        0.001_388_889f32,
        0.008_333_334,
        0.041_666_668,
        0.166_666_67,
        0.5,
        1.0,
    ] {
        p = _mm256_mul_ps(r, _mm256_add_ps(_mm256_set1_ps(c), p));
    }
    p = _mm256_add_ps(_mm256_set1_ps(1.0), p);
    let e = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
    let e = _mm256_min_epi32(
        _mm256_max_epi32(e, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(254),
    );
    let y = _mm256_mul_ps(_mm256_castsi256_ps(_mm256_slli_epi32::<23>(e)), p);
    let under = _mm256_cmp_ps::<_CMP_LT_OQ>(t, _mm256_set1_ps(-87.0));
    _mm256_blendv_ps(y, _mm256_setzero_ps(), under)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::needless_range_loop)]
unsafe fn argmax_avx2(row: &[f32]) -> u32 {
    use std::arch::x86_64::*;
    let n = row.len();
    if n > u32::MAX as usize {
        return argmax(row);
    }
    let p = row.as_ptr();
    // thirty-two lanes, each the lowest-index maximum of its own column of the row
    let mut best = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    let mut idx = [_mm256_setzero_si256(); 4];
    let lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let mut cur = [0i32, 8, 16, 24].map(|o| _mm256_add_epi32(lane, _mm256_set1_epi32(o)));
    let mut i = 0;
    while i + 32 <= n {
        for k in 0..4 {
            let v = _mm256_loadu_ps(p.add(i + 8 * k));
            let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(v, best[k]); // a NaN never wins, as the scalar `>`
            best[k] = _mm256_blendv_ps(best[k], v, gt);
            idx[k] = _mm256_blendv_epi8(idx[k], cur[k], _mm256_castps_si256(gt));
            cur[k] = _mm256_add_epi32(cur[k], _mm256_set1_epi32(32));
        }
        i += 32;
    }
    let (mut bv, mut bx) = ([0.0f32; 32], [0u32; 32]);
    for k in 0..4 {
        _mm256_storeu_ps(bv.as_mut_ptr().add(8 * k), best[k]);
        _mm256_storeu_si256(bx.as_mut_ptr().add(8 * k) as *mut __m256i, idx[k]);
    }
    // no lane holds a NaN; -0 == +0 here as in the scalar compare, so the first zero wins
    let m = bv
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let bi = (0..32)
        .filter(|&l| bv[l] == m)
        .map(|l| bx[l])
        .min()
        .unwrap_or(0);
    argmax_from(row, i, m, bi as usize)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_max_avx2(x: &[f32], inv_t: f32, s: &mut Vec<f32>) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    s.clear();
    s.reserve(n);
    let (src, dst) = (x.as_ptr(), s.as_mut_ptr());
    let it = _mm256_set1_ps(inv_t);
    let mut mx = [_mm256_set1_ps(f32::NEG_INFINITY); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (k, acc) in mx.iter_mut().enumerate() {
            let v = _mm256_mul_ps(_mm256_loadu_ps(src.add(i + 8 * k)), it);
            _mm256_storeu_ps(dst.add(i + 8 * k), v);
            *acc = _mm256_max_ps(v, *acc); // a NaN `v` returns the accumulator: NaN never wins, as f32::max
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_mul_ps(_mm256_loadu_ps(src.add(i)), it);
        _mm256_storeu_ps(dst.add(i), v);
        mx[0] = _mm256_max_ps(v, mx[0]);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(
        lanes.as_mut_ptr(),
        _mm256_max_ps(_mm256_max_ps(mx[0], mx[1]), _mm256_max_ps(mx[2], mx[3])),
    );
    let mut m = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < n {
        let v = *src.add(i) * inv_t;
        *dst.add(i) = v;
        m = m.max(v);
        i += 1;
    }
    s.set_len(n);
    m
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn masses_avx2(s: &[f32], m: f32, floor: u32, w: &mut Vec<u64>) {
    use std::arch::x86_64::*;
    let n = s.len();
    w.clear();
    w.reserve(n);
    let (src, dst) = (s.as_ptr(), w.as_mut_ptr());
    let (mv, fl) = (_mm256_set1_ps(m), _mm256_set1_epi32(floor as i32));
    let mut i = 0;
    while i + 8 <= n {
        let a = _mm256_loadu_ps(src.add(i));
        let t = _mm256_sub_ps(a, mv);
        let keep = _mm256_andnot_si256(
            lt_u32_avx2(ordered_avx2(a), fl),
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_GE_OQ>(t, _mm256_set1_ps(-21.0))),
        );
        let e = _mm256_and_ps(_mm256_castsi256_ps(keep), fast_exp_avx2(t));
        // e * 2^30 is exact and at most 2^30, so the truncating i32 convert is `(e * MASS_SCALE) as u64` and
        // widens to u64 by zero-extension
        let q = _mm256_cvttps_epi32(_mm256_mul_ps(e, _mm256_set1_ps(MASS_SCALE)));
        _mm256_storeu_si256(
            dst.add(i) as *mut __m256i,
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(q)),
        );
        _mm256_storeu_si256(
            dst.add(i + 4) as *mut __m256i,
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(q)),
        );
        i += 8;
    }
    while i < n {
        *dst.add(i) = mass(*src.add(i), m, floor);
        i += 1;
    }
    w.set_len(n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn draw_avx2(s: &[f32], m: f32, floor: u32, key: u64) -> u32 {
    use std::arch::x86_64::*;
    let n = s.len();
    let p = s.as_ptr();
    let (mv, fl, lim) = (
        _mm256_set1_ps(m),
        _mm256_set1_epi32(floor as i32),
        _mm256_set1_ps(-21.0),
    );
    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
    let mut i = 0;
    while i + 16 <= n {
        let mut skip = _mm256_set1_epi32(-1);
        for k in 0..2 {
            let a = _mm256_loadu_ps(p.add(i + 8 * k));
            let out = _mm256_or_si256(
                lt_u32_avx2(ordered_avx2(a), fl),
                _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_sub_ps(a, mv), lim)),
            );
            skip = _mm256_and_si256(skip, out);
        }
        // a lane some token of the sixteen stays in: race them in order, as the scalar loop does
        if _mm256_movemask_epi8(skip) != -1 {
            (best, bi) = draw_from(&s[..i + 16], m, floor, key, i, best, bi);
        }
        i += 16;
    }
    draw_from(s, m, floor, key, i, best, bi).1 as u32
}

/// the radix select over `s` (scaled logits): the threshold u (ordered bits) at which the count of tokens with
/// u' >= u first reaches `need`, over the tokens with u' >= floor
fn select_count(s: &[f32], need: u32, floor: u32) -> u32 {
    let mut prefix = 0u32;
    let mut need = need;
    let mut shift = 32u32;
    let mut cnt = vec![0u32; BINS];
    for level in 0..3 {
        let bits = if level < 2 { 11 } else { 10 };
        shift -= bits;
        cnt.iter_mut().for_each(|c| *c = 0);
        let hi = shift + bits;
        for &v in s {
            let u = ordered(v);
            if u >= floor && (hi >= 32 || (u >> hi) == (prefix >> hi)) {
                cnt[((u >> shift) as usize) & (BINS - 1)] += 1;
            }
        }
        let mut acc = 0u32;
        let mut pick = 0usize;
        for b in (0..BINS).rev() {
            acc += cnt[b];
            if acc >= need {
                pick = b;
                need -= acc - cnt[b];
                break;
            }
        }
        prefix |= (pick as u32) << shift;
    }
    prefix
}

/// the radix select by mass: the threshold u at which the mass of tokens with u' >= u first reaches the target
/// (top_p of the mass over the tokens with u' >= floor); `w` the masses
fn select_mass(s: &[f32], w: &[u64], top_p: f32, floor: u32) -> u32 {
    let mut prefix = 0u32;
    let mut need = 0u64;
    let mut shift = 32u32;
    let mut mass = vec![0u64; BINS];
    for level in 0..3 {
        let bits = if level < 2 { 11 } else { 10 };
        shift -= bits;
        mass.iter_mut().for_each(|c| *c = 0);
        let hi = shift + bits;
        for (&v, &wi) in s.iter().zip(w) {
            let u = ordered(v);
            if u >= floor && (hi >= 32 || (u >> hi) == (prefix >> hi)) {
                mass[((u >> shift) as usize) & (BINS - 1)] += wi;
            }
        }
        if level == 0 {
            let z: u64 = mass.iter().sum();
            need = ((z as f32 * top_p) as u64).max(1); // at least the top token
        }
        let mut acc = 0u64;
        let mut pick = 0usize;
        for b in (0..BINS).rev() {
            acc += mass[b];
            if acc >= need {
                pick = b;
                need -= acc - mass[b];
                break;
            }
        }
        prefix |= (pick as u32) << shift;
    }
    prefix
}

macro_rules! def_pick_row {
    ($name:ident, $argmax:ident, $scale_max:ident, $masses:ident, $draw:ident) => {
        unsafe fn $name(x: &[f32], key: u64, c: Cfg, s: &mut Vec<f32>, w: &mut Vec<u64>) -> u32 {
            if c.inv_t <= 0.0 {
                return $argmax(x);
            }
            let v = x.len();
            let m = $scale_max(x, c.inv_t, s);
            let mut floor = 0u32;
            if c.top_k > 0 && (c.top_k as usize) < v {
                floor = select_count(s, c.top_k, 0);
            }
            if c.top_p < 1.0 {
                $masses(s, m, floor, w);
                floor = floor.max(select_mass(s, w, c.top_p, floor));
            }
            // the draw: argmax of s + gumbel over the kept tokens
            $draw(s, m, floor, key)
        }
    };
}

def_pick_row!(pick_row, argmax, scale_max, masses, draw);
#[cfg(target_arch = "aarch64")]
def_pick_row!(
    pick_row_neon,
    argmax_neon,
    scale_max_neon,
    masses_neon,
    draw_neon
);
#[cfg(target_arch = "x86_64")]
def_pick_row!(
    pick_row_avx2,
    argmax_avx2,
    scale_max_avx2,
    masses_avx2,
    draw_avx2
);

#[inline]
unsafe fn pick_on(
    isa: Isa,
    x: &[f32],
    key: u64,
    c: Cfg,
    s: &mut Vec<f32>,
    w: &mut Vec<u64>,
) -> u32 {
    #[cfg(target_arch = "aarch64")]
    if isa == Isa::Neon {
        return pick_row_neon(x, key, c, s, w);
    }
    #[cfg(target_arch = "x86_64")]
    if matches!(isa, Isa::Avx512 | Isa::Avx2) {
        return pick_row_avx2(x, key, c, s, w);
    }
    let _ = isa;
    pick_row(x, key, c, s, w)
}

/// # Safety
/// `x` readable for `rows * v` f32, `keys` for `rows` u64, `out` writable for `rows` u32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sample_pick(
    x: *const f32,
    rows: usize,
    v: usize,
    keys: *const u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    out: *mut u32,
    threads: usize,
) -> i32 {
    if x.is_null() || keys.is_null() || out.is_null() {
        return ERR_NULL;
    }
    if rows == 0
        || v == 0
        || rows.checked_mul(v).is_none()
        || !temperature.is_finite()
        || !top_p.is_finite()
    {
        return ERR_DOMAIN;
    }
    let nt = match crate::gemv::resolve_threads(threads) {
        Some(n) => n,
        None => return ERR_DOMAIN,
    };
    let c = Cfg {
        inv_t: if temperature > 0.0 {
            1.0 / temperature
        } else {
            0.0
        },
        top_k,
        top_p,
    };
    let xs = unsafe { std::slice::from_raw_parts(x, rows * v) };
    let ks = unsafe { std::slice::from_raw_parts(keys, rows) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, rows) };
    let isa = crate::gemv::isa();
    let one = |r: usize, o: &mut u32| {
        let mut s = Vec::with_capacity(v);
        let mut w = Vec::with_capacity(v);
        *o = unsafe { pick_on(isa, &xs[r * v..(r + 1) * v], ks[r], c, &mut s, &mut w) };
    };
    if nt <= 1 || rows == 1 {
        for (r, o) in os.iter_mut().enumerate() {
            one(r, o);
        }
    } else {
        os.par_iter_mut().enumerate().for_each(|(r, o)| one(r, o));
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &[f32], rows: usize, keys: &[u64], t: f32, k: u32, p: f32) -> Vec<u32> {
        let v = x.len() / rows;
        let mut out = vec![0u32; rows];
        let rc = unsafe {
            sample_pick(
                x.as_ptr(),
                rows,
                v,
                keys.as_ptr(),
                t,
                k,
                p,
                out.as_mut_ptr(),
                1,
            )
        };
        assert_eq!(rc, OK);
        out
    }

    #[test]
    fn greedy_is_the_argmax_lowest_index_first() {
        let x = [1.0f32, 3.0, 3.0, -1.0, 2.0];
        assert_eq!(run(&x, 1, &[0], 0.0, 0, 1.0), vec![1]);
    }

    #[test]
    fn fast_exp_holds_to_a_few_ulps() {
        for i in 0..2000 {
            let t = -(i as f32) * 0.04;
            let a = fast_exp(t);
            let b = t.exp();
            assert!(
                (a - b).abs() <= 4e-7 * b.max(1e-30) + 1e-38,
                "{t}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn the_masks_keep_the_right_tokens_and_a_key_repeats() {
        // probabilities 0.5, 0.25, 0.125, ...: top-p 0.8 keeps three, top-k 2 keeps two
        let x: Vec<f32> = (0..8).map(|i| (0.5f32 / 2f32.powi(i)).ln()).collect();
        let mut seen_p = std::collections::HashSet::new();
        let mut seen_k = std::collections::HashSet::new();
        for key in 0..400u64 {
            seen_p.insert(run(&x, 1, &[key], 1.0, 0, 0.8)[0]);
            seen_k.insert(run(&x, 1, &[key], 1.0, 2, 1.0)[0]);
            assert_eq!(
                run(&x, 1, &[key], 1.0, 0, 0.8),
                run(&x, 1, &[key], 1.0, 0, 0.8)
            );
        }
        assert_eq!(seen_p, [0u32, 1, 2].into_iter().collect());
        assert_eq!(seen_k, [0u32, 1].into_iter().collect());
    }

    #[test]
    fn the_draws_follow_the_distribution() {
        let p = [0.5f32, 0.3, 0.15, 0.05];
        let x: Vec<f32> = p.iter().map(|q| q.ln()).collect();
        let n = 20000u64;
        let mut counts = [0u32; 4];
        for key in 0..n {
            counts[run(&x, 1, &[key], 1.0, 0, 1.0)[0] as usize] += 1;
        }
        for (i, q) in p.iter().enumerate() {
            let f = counts[i] as f32 / n as f32;
            assert!((f - q).abs() < 0.02, "token {i}: {f} vs {q}");
        }
    }

    /// Each NEON pass and the whole pick agree with the scalar ones to the bit, over ragged widths,
    /// ties, signed zeros, infinities and NaNs; the C ABI runs one tier per process, so the two are
    /// compared here.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_passes_are_bit_identical_to_scalar() {
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            st
        };
        let uni = |r: u64| ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 24.0;
        let special = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 4.0,
            f32::MAX,
        ];
        let cfgs = [
            (0.0f32, 0u32, 1.0f32),
            (1.0, 0, 1.0),
            (0.8, 40, 0.9),
            (0.7, 0, 0.8),
            (1.5, 3, 0.5),
            (0.3, 0, 0.99),
        ];
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        for v in (1..=70).chain([255, 256, 257, 3001, 50257]) {
            for kind in 0..4 {
                let row: Vec<f32> = (0..v)
                    .map(|_| {
                        let r = next();
                        match kind {
                            0 => uni(r),
                            1 => (r >> 61) as f32 - 3.0,
                            2 if r % 5 == 0 => special[(r >> 32) as usize % special.len()],
                            2 => uni(r),
                            _ => [0.0, -0.0, f32::NEG_INFINITY][(r >> 32) as usize % 3],
                        }
                    })
                    .collect();
                let ctx = format!("v={v} kind={kind}");
                unsafe {
                    assert_eq!(argmax(&row), argmax_neon(&row), "{ctx}: argmax");
                    for &(t, _, _) in &cfgs[1..] {
                        let (mut s, mut sn) = (Vec::new(), Vec::new());
                        let m = scale_max(&row, 1.0 / t, &mut s);
                        let mn = scale_max_neon(&row, 1.0 / t, &mut sn);
                        assert_eq!(m.to_bits(), mn.to_bits(), "{ctx} t={t}: max");
                        assert_eq!(bits(&s), bits(&sn), "{ctx} t={t}: scaled");
                        let floors = [0, if v > 7 { select_count(&s, 7, 0) } else { 0 }];
                        for floor in floors {
                            let (mut w, mut wn) = (Vec::new(), Vec::new());
                            masses(&s, m, floor, &mut w);
                            masses_neon(&s, m, floor, &mut wn);
                            assert_eq!(w, wn, "{ctx} t={t} floor={floor}: masses");
                            for key in 0..3u64 {
                                assert_eq!(
                                    draw(&s, m, floor, key),
                                    draw_neon(&s, m, floor, key),
                                    "{ctx} t={t} floor={floor} key={key}: draw"
                                );
                            }
                        }
                    }
                    for &(t, k, p) in &cfgs {
                        let inv_t = if t > 0.0 { 1.0 / t } else { 0.0 };
                        let c = Cfg {
                            inv_t,
                            top_k: k,
                            top_p: p,
                        };
                        for key in [0u64, 0xDEAD_BEEF, u64::MAX] {
                            let (mut s, mut w) = (Vec::new(), Vec::new());
                            let a = pick_row(&row, key, c, &mut s, &mut w);
                            let b = pick_row_neon(&row, key, c, &mut s, &mut w);
                            assert_eq!(a, b, "{ctx} t={t} k={k} p={p} key={key}: pick");
                        }
                    }
                }
            }
        }
    }

    /// The AVX2 passes against the scalar ones over the NEON test's rows. The max is compared by value: which
    /// zero a row of +0 and -0 reports is order-dependent (`f32::max` leaves it open), and no pick can see it -
    /// `a - m` against -21 and fast_exp(+-0) agree - so every pass after it is still held to the bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_passes_are_bit_identical_to_scalar() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut st = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            st
        };
        let uni = |r: u64| ((r >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 24.0;
        let special = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE / 4.0,
            f32::MAX,
        ];
        let cfgs = [
            (0.0f32, 0u32, 1.0f32),
            (1.0, 0, 1.0),
            (0.8, 40, 0.9),
            (0.7, 0, 0.8),
            (1.5, 3, 0.5),
            (0.3, 0, 0.99),
        ];
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        for v in (1..=70).chain([255, 256, 257, 3001, 50257]) {
            for kind in 0..4 {
                let row: Vec<f32> = (0..v)
                    .map(|_| {
                        let r = next();
                        match kind {
                            0 => uni(r),
                            1 => (r >> 61) as f32 - 3.0,
                            2 if r % 5 == 0 => special[(r >> 32) as usize % special.len()],
                            2 => uni(r),
                            _ => [0.0, -0.0, f32::NEG_INFINITY][(r >> 32) as usize % 3],
                        }
                    })
                    .collect();
                let ctx = format!("v={v} kind={kind}");
                unsafe {
                    assert_eq!(argmax(&row), argmax_avx2(&row), "{ctx}: argmax");
                    for &(t, _, _) in &cfgs[1..] {
                        let (mut s, mut sv) = (Vec::new(), Vec::new());
                        let m = scale_max(&row, 1.0 / t, &mut s);
                        let mv = scale_max_avx2(&row, 1.0 / t, &mut sv);
                        assert!(
                            m == mv || (m.is_nan() && mv.is_nan()),
                            "{ctx} t={t}: max {m} vs {mv}"
                        );
                        assert_eq!(bits(&s), bits(&sv), "{ctx} t={t}: scaled");
                        let floors = [0, if v > 7 { select_count(&s, 7, 0) } else { 0 }];
                        for floor in floors {
                            let (mut w, mut wv) = (Vec::new(), Vec::new());
                            masses(&s, m, floor, &mut w);
                            masses_avx2(&s, m, floor, &mut wv);
                            assert_eq!(w, wv, "{ctx} t={t} floor={floor}: masses");
                            for key in 0..3u64 {
                                assert_eq!(
                                    draw(&s, m, floor, key),
                                    draw_avx2(&s, m, floor, key),
                                    "{ctx} t={t} floor={floor} key={key}: draw"
                                );
                            }
                        }
                    }
                    for &(t, k, p) in &cfgs {
                        let inv_t = if t > 0.0 { 1.0 / t } else { 0.0 };
                        let c = Cfg {
                            inv_t,
                            top_k: k,
                            top_p: p,
                        };
                        for key in [0u64, 0xDEAD_BEEF, u64::MAX] {
                            let (mut s, mut w) = (Vec::new(), Vec::new());
                            let a = pick_row(&row, key, c, &mut s, &mut w);
                            let b = pick_row_avx2(&row, key, c, &mut s, &mut w);
                            assert_eq!(a, b, "{ctx} t={t} k={k} p={p} key={key}: pick");
                        }
                    }
                }
            }
        }
    }

    /// fast_exp's rounding step is the one place AVX2 differs from the scalar ops (ties to even): every tie of
    /// t * log2(e) in fast_exp's range, and the values a ulp either side, round alike
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_rounds_ties_away_from_zero_as_f32_round() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        use std::arch::x86_64::*;
        let mut xs = Vec::new();
        for k in -260i32..=260 {
            let h = k as f32 * 0.5;
            xs.extend([
                h,
                f32::from_bits(h.to_bits().wrapping_add(1)),
                f32::from_bits(h.to_bits().wrapping_sub(1)),
            ]);
        }
        xs.extend([
            0.0,
            -0.0,
            0.49999997,
            -0.49999997,
            8_388_607.5,
            -8_388_607.5,
        ]);
        xs.retain(|x| x.is_finite()); // 0.0's bits less one are a NaN, whose payload f32::round does not promise
        for chunk in xs.chunks(8) {
            let mut lane = [0.0f32; 8];
            lane[..chunk.len()].copy_from_slice(chunk);
            let mut got = [0.0f32; 8];
            unsafe {
                _mm256_storeu_ps(
                    got.as_mut_ptr(),
                    round_away_avx2(_mm256_loadu_ps(lane.as_ptr())),
                )
            };
            for (x, g) in lane.iter().zip(got) {
                assert_eq!(x.round().to_bits(), g.to_bits(), "round({x})");
            }
        }
    }

    #[test]
    fn rows_pick_alike_alone_and_together() {
        let v = 3000;
        let mut x = Vec::with_capacity(16 * v);
        let mut st = 7u64;
        for _ in 0..16 * v {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            x.push(((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 12.0);
        }
        let keys: Vec<u64> = (0..16).collect();
        let together = run(&x, 16, &keys, 0.8, 40, 0.9);
        for r in 0..16 {
            assert_eq!(
                run(&x[r * v..(r + 1) * v], 1, &[keys[r]], 0.8, 40, 0.9)[0],
                together[r]
            );
        }
    }
}
