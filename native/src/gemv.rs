// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
use crate::codes::{ERR_DOMAIN, ERR_NULL, OK};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const LANES: usize = 16;

const ROW_UNROLL: usize = 4;

/// Row chunks per thread a call is cut into; rayon steals across the box's cores of unequal speed.
const SPLIT: usize = 4;

const COL_TILE: usize = 512;

const B_TILE: usize = 8;

#[inline(always)]
pub fn bf16_to_f32(w: u16) -> f32 {
    f32::from_bits((w as u32) << 16)
}

#[inline]
unsafe fn reduce16(a: *const f32) -> f32 {
    let mut t = [0.0f32; 8];
    for (j, tv) in t.iter_mut().enumerate() {
        *tv = *a.add(2 * j) + *a.add(2 * j + 1);
    }
    let u0 = t[0] + t[1];
    let u1 = t[2] + t[3];
    let u2 = t[4] + t[5];
    let u3 = t[6] + t[7];
    let v0 = u0 + u1;
    let v1 = u2 + u3;
    v0 + v1
}

/// A weight element as the accumulation loads it: a bf16 bit pattern widened in the load, or an f32 read
/// as is. Widening is exact, so the two accumulate to the same bits.
trait Elem: Copy + Send + Sync {
    const BF16: bool;
    fn widen(self) -> f32;
}

impl Elem for u16 {
    const BF16: bool = true;
    #[inline(always)]
    fn widen(self) -> f32 {
        bf16_to_f32(self)
    }
}

impl Elem for f32 {
    const BF16: bool = false;
    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }
}

/// `R` rows of `w` times `x` over `n` columns into the 16 lane sums of each row in `acc`.
#[inline]
unsafe fn accum_scalar<E: Elem, const R: usize>(
    acc: *mut f32,
    w: *const E,
    row_stride: usize,
    x: *const f32,
    n: usize,
) {
    let nb = n / LANES;
    for k in 0..nb {
        let c = k * LANES;
        for r in 0..R {
            let a = acc.add(r * LANES);
            let wr = w.add(r * row_stride + c);
            for j in 0..LANES {
                *a.add(j) = (*wr.add(j)).widen().mul_add(*x.add(c + j), *a.add(j));
            }
        }
    }
    accum_tail::<E, R>(acc, w, row_stride, x, n, nb * LANES);
}

/// The columns past the last whole group of 16, lane `j` of every row.
#[inline(always)]
unsafe fn accum_tail<E: Elem, const R: usize>(
    acc: *mut f32,
    w: *const E,
    row_stride: usize,
    x: *const f32,
    n: usize,
    base: usize,
) {
    for j in 0..(n - base) {
        let xv = *x.add(base + j);
        for r in 0..R {
            let a = acc.add(r * LANES + j);
            *a = (*w.add(r * row_stride + base + j)).widen().mul_add(xv, *a);
        }
    }
}

#[inline]
unsafe fn widen_scalar<const R: usize>(
    dst: *mut f32,
    w: *const u16,
    row_stride: usize,
    dst_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let src = w.add(r * row_stride);
        let out = dst.add(r * dst_stride);
        for j in 0..n {
            *out.add(j) = bf16_to_f32(*src.add(j));
        }
    }
}

/// Sixteen weights from `p` as two f32 vectors: bf16 widened by a 16-bit shift, f32 loaded as is.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load16_avx2<E: Elem>(
    p: *const E,
) -> (std::arch::x86_64::__m256, std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;
    if E::BF16 {
        let v = _mm256_loadu_si256(p as *const __m256i);
        (
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(v),
            ))),
            _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(
                _mm256_extracti128_si256::<1>(v),
            ))),
        )
    } else {
        let f = p as *const f32;
        (_mm256_loadu_ps(f), _mm256_loadu_ps(f.add(8)))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accum_avx2<E: Elem, const R: usize>(
    acc: *mut f32,
    w: *const E,
    row_stride: usize,
    x: *const f32,
    n: usize,
) {
    use std::arch::x86_64::*;

    let mut lo = [_mm256_setzero_ps(); R];
    let mut hi = [_mm256_setzero_ps(); R];
    for r in 0..R {
        lo[r] = _mm256_loadu_ps(acc.add(r * LANES));
        hi[r] = _mm256_loadu_ps(acc.add(r * LANES + 8));
    }

    let nb = n / LANES;
    for k in 0..nb {
        let c = k * LANES;
        let xlo = _mm256_loadu_ps(x.add(c));
        let xhi = _mm256_loadu_ps(x.add(c + 8));
        for r in 0..R {
            let (wl, wh) = load16_avx2(w.add(r * row_stride + c));
            lo[r] = _mm256_fmadd_ps(wl, xlo, lo[r]);
            hi[r] = _mm256_fmadd_ps(wh, xhi, hi[r]);
        }
    }

    for r in 0..R {
        _mm256_storeu_ps(acc.add(r * LANES), lo[r]);
        _mm256_storeu_ps(acc.add(r * LANES + 8), hi[r]);
    }
    accum_tail::<E, R>(acc, w, row_stride, x, n, nb * LANES);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_avx2<const R: usize>(
    dst: *mut f32,
    w: *const u16,
    row_stride: usize,
    dst_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let nb = n / LANES;
    for r in 0..R {
        let src = w.add(r * row_stride);
        let out = dst.add(r * dst_stride);
        for k in 0..nb {
            let c = k * LANES;
            let v = _mm256_loadu_si256(src.add(c) as *const __m256i);
            let lo = _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(v)));
            let hi =
                _mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(v)));
            _mm256_storeu_ps(out.add(c), _mm256_castsi256_ps(lo));
            _mm256_storeu_ps(out.add(c + 8), _mm256_castsi256_ps(hi));
        }
        for j in (nb * LANES)..n {
            *out.add(j) = bf16_to_f32(*src.add(j));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn widen_avx512<const R: usize>(
    dst: *mut f32,
    w: *const u16,
    row_stride: usize,
    dst_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let nb = n / LANES;
    for r in 0..R {
        let src = w.add(r * row_stride);
        let out = dst.add(r * dst_stride);
        for k in 0..nb {
            let c = k * LANES;
            let v = _mm256_loadu_si256(src.add(c) as *const __m256i);
            let wide = _mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(v));
            _mm512_storeu_ps(out.add(c), _mm512_castsi512_ps(wide));
        }
        for j in (nb * LANES)..n {
            *out.add(j) = bf16_to_f32(*src.add(j));
        }
    }
}

/// Sixteen weights from `p` as one f32 vector: bf16 widened by a 16-bit shift, f32 loaded as is.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn load16_avx512<E: Elem>(p: *const E) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    if E::BF16 {
        let v = _mm256_loadu_si256(p as *const __m256i);
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(v)))
    } else {
        _mm512_loadu_ps(p as *const f32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::needless_range_loop)]
unsafe fn accum_avx512<E: Elem, const R: usize>(
    acc: *mut f32,
    w: *const E,
    row_stride: usize,
    x: *const f32,
    n: usize,
) {
    use std::arch::x86_64::*;

    let mut a = [_mm512_setzero_ps(); R];
    for r in 0..R {
        a[r] = _mm512_loadu_ps(acc.add(r * LANES));
    }

    let nb = n / LANES;
    for k in 0..nb {
        let c = k * LANES;
        let xv = _mm512_loadu_ps(x.add(c));
        for r in 0..R {
            a[r] = _mm512_fmadd_ps(load16_avx512(w.add(r * row_stride + c)), xv, a[r]);
        }
    }

    for r in 0..R {
        _mm512_storeu_ps(acc.add(r * LANES), a[r]);
    }
    accum_tail::<E, R>(acc, w, row_stride, x, n, nb * LANES);
}

// NEON (every arm64 core): the same 16 lanes as four float32x4 accumulators, each lane an FMLA, so
// the lane sums (and the reduce16 fold after them) are bit-identical to the scalar path's mul_add.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bf16x8_neon(
    p: *const u16,
) -> (
    std::arch::aarch64::float32x4_t,
    std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let v = vld1q_u16(p);
    (
        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(v))),
        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(v)),
    )
}

/// Sixteen weights from `p` as four f32 vectors: bf16 widened by a 16-bit shift, f32 loaded as is.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load16_neon<E: Elem>(p: *const E) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    if E::BF16 {
        let h = p as *const u16;
        let (w0, w1) = bf16x8_neon(h);
        let (w2, w3) = bf16x8_neon(h.add(8));
        [w0, w1, w2, w3]
    } else {
        let f = p as *const f32;
        [
            vld1q_f32(f),
            vld1q_f32(f.add(4)),
            vld1q_f32(f.add(8)),
            vld1q_f32(f.add(12)),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::needless_range_loop)]
unsafe fn accum_neon<E: Elem, const R: usize>(
    acc: *mut f32,
    w: *const E,
    row_stride: usize,
    x: *const f32,
    n: usize,
) {
    use std::arch::aarch64::*;

    let mut a = [[vdupq_n_f32(0.0); 4]; R];
    for r in 0..R {
        for q in 0..4 {
            a[r][q] = vld1q_f32(acc.add(r * LANES + q * 4));
        }
    }

    let nb = n / LANES;
    for k in 0..nb {
        let c = k * LANES;
        let xv = [
            vld1q_f32(x.add(c)),
            vld1q_f32(x.add(c + 4)),
            vld1q_f32(x.add(c + 8)),
            vld1q_f32(x.add(c + 12)),
        ];
        for r in 0..R {
            let wv = load16_neon(w.add(r * row_stride + c));
            for q in 0..4 {
                a[r][q] = vfmaq_f32(a[r][q], wv[q], xv[q]);
            }
        }
    }

    for r in 0..R {
        for q in 0..4 {
            vst1q_f32(acc.add(r * LANES + q * 4), a[r][q]);
        }
    }
    accum_tail::<E, R>(acc, w, row_stride, x, n, nb * LANES);
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn widen_neon<const R: usize>(
    dst: *mut f32,
    w: *const u16,
    row_stride: usize,
    dst_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let nb = n / LANES;
    for r in 0..R {
        let src = w.add(r * row_stride);
        let out = dst.add(r * dst_stride);
        for k in 0..nb {
            let c = k * LANES;
            let (w0, w1) = bf16x8_neon(src.add(c));
            let (w2, w3) = bf16x8_neon(src.add(c + 8));
            vst1q_f32(out.add(c), w0);
            vst1q_f32(out.add(c + 4), w1);
            vst1q_f32(out.add(c + 8), w2);
            vst1q_f32(out.add(c + 12), w3);
        }
        for j in (nb * LANES)..n {
            *out.add(j) = bf16_to_f32(*src.add(j));
        }
    }
}

/// The bf16 rows of a task, `Send + Sync` for the row partition (rows are disjoint per task).
#[derive(Clone, Copy)]
struct Bf16(*const u16);

// SAFETY: a task reads its rows and writes its own rows of `y`; the caller's buffers do not overlap
unsafe impl Send for Bf16 {}
unsafe impl Sync for Bf16 {}

// One task body for every weight form and ISA. `R` rows at a time (`ROW_UNROLL`, then single rows); for
// each group of up to `B_TILE` vectors the row group is walked over the columns in `COL_TILE` tiles, every
// vector of the group accumulated from the same tile while it is in cache, and the 16 lane sums of each
// (vector, row) folded by `reduce16`. An accumulator meets the tiles in order with a fixed lane mapping, so
// the bits do not depend on `b`. `prep` runs once per (vector group, row group); `tile` either fills `wide`
// with the tile's weights as f32 and yields `false`, or (one bf16 vector) accumulates straight from the rows
// and yields `true`.
macro_rules! def_task {
    ($group:ident, $task:ident, $w:ty, $accum:ident $(, $feat:literal)?,
     prep |$pp:ident, $pr:ident, $pcols:ident, $pcur:ident| $prep:block,
     tile |$scratch:ident, $wide:ident, $x:ident, $i0:ident, $bt:ident, $p:ident, $r:ident, $c0:ident,
           $cols:ident, $n:ident, $cur:ident| $tile:block) => {
        $(#[target_feature(enable = $feat)])?
        #[allow(clippy::too_many_arguments, unused_variables, unused_mut)]
        unsafe fn $group<const R: usize>(
            $scratch: *mut f32,
            $wide: *mut f32,
            $p: $w,
            $cols: usize,
            $x: *const f32,
            b: usize,
            y: *mut f32,
            rows_total: usize,
            $r: usize,
        ) {
            let mut $i0 = 0usize;
            while $i0 < b {
                let $bt = (b - $i0).min(B_TILE);
                std::ptr::write_bytes($scratch, 0, $bt * R * LANES);
                let mut $cur = [0usize; ROW_UNROLL];
                {
                    let $pp = $p;
                    let $pr = $r;
                    let $pcols = $cols;
                    let $pcur = &mut $cur;
                    $prep
                }
                let mut $c0 = 0usize;
                while $c0 < $cols {
                    let $n = ($cols - $c0).min(COL_TILE);
                    let direct: bool = $tile;
                    if !direct {
                        for i in 0..$bt {
                            $accum::<f32, R>(
                                $scratch.add(i * R * LANES),
                                $wide,
                                COL_TILE,
                                $x.add(($i0 + i) * $cols + $c0),
                                $n,
                            );
                        }
                    }
                    $c0 += COL_TILE;
                }
                for i in 0..$bt {
                    for rr in 0..R {
                        *y.add(($i0 + i) * rows_total + $r + rr) =
                            reduce16($scratch.add((i * R + rr) * LANES));
                    }
                }
                $i0 += B_TILE;
            }
        }

        $(#[target_feature(enable = $feat)])?
        #[allow(clippy::too_many_arguments)]
        unsafe fn $task(
            p: $w,
            cols: usize,
            x: *const f32,
            b: usize,
            y: *mut f32,
            rows_total: usize,
            r0: usize,
            r1: usize,
        ) {
            let mut scratch = [0.0f32; B_TILE * ROW_UNROLL * LANES];
            let s = scratch.as_mut_ptr();
            let mut wide = [0.0f32; ROW_UNROLL * COL_TILE];
            let d = wide.as_mut_ptr();
            let mut r = r0;
            while r + ROW_UNROLL <= r1 {
                $group::<ROW_UNROLL>(s, d, p, cols, x, b, y, rows_total, r);
                r += ROW_UNROLL;
            }
            while r < r1 {
                $group::<1>(s, d, p, cols, x, b, y, rows_total, r);
                r += 1;
            }
        }
    };
}

macro_rules! def_task_bf16 {
    ($group:ident, $task:ident, $accum:ident, $widen:ident $(, $feat:literal)?) => {
        def_task!($group, $task, Bf16, $accum $(, $feat)?,
            prep |p, r, cols, cur| {},
            tile |scratch, wide, x, i0, bt, p, r, c0, cols, n, cur| {
                let wt = p.0.add(r * cols + c0);
                if bt == 1 {
                    $accum::<u16, R>(scratch, wt, cols, x.add(i0 * cols + c0), n);
                    true
                } else {
                    $widen::<R>(wide, wt, cols, COL_TILE, n);
                    false
                }
            });
    };
}

def_task_bf16!(group_scalar, task_scalar, accum_scalar, widen_scalar);
#[cfg(target_arch = "x86_64")]
def_task_bf16!(group_avx2, task_avx2, accum_avx2, widen_avx2, "avx2,fma");
#[cfg(target_arch = "aarch64")]
def_task_bf16!(group_neon, task_neon, accum_neon, widen_neon);
#[cfg(target_arch = "x86_64")]
def_task_bf16!(
    group_avx512,
    task_avx512,
    accum_avx512,
    widen_avx512,
    "avx512f,avx512bw"
);

#[derive(Clone, Copy)]
struct P12 {
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    esc_idx: *const i32,
    esc_val: *const u8,
    n_esc: usize,
}

// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for P12 {}
unsafe impl Sync for P12 {}

/// The escape cursors of a row group: for each of `R` rows, the first escape at or past the row's start.
///
/// # Safety
/// `p.esc_idx` readable for `p.n_esc` i32 (the core checks the list before any task runs).
#[inline]
unsafe fn p12_cursors<const R: usize>(
    p: P12,
    r: usize,
    cols: usize,
    cur: &mut [usize; ROW_UNROLL],
) {
    if p.n_esc == 0 {
        return;
    }
    let idx = std::slice::from_raw_parts(p.esc_idx, p.n_esc);
    for (rr, c) in cur.iter_mut().enumerate().take(R) {
        let start = (r + rr) * cols;
        *c = idx.partition_point(|&v| (v as usize) < start);
    }
}

#[inline]
unsafe fn patch_escapes<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: P12,
    start: usize,
    row_stride: usize,
    n: usize,
    cur: &mut [usize; ROW_UNROLL],
) {
    if p.n_esc == 0 {
        return;
    }
    for (r, cursor) in cur.iter_mut().enumerate().take(R) {
        let s = start + r * row_stride;
        let end = s + n;
        let mut k = *cursor;
        while k < p.n_esc && (*p.esc_idx.add(k) as usize) < s {
            k += 1;
        }
        while k < p.n_esc {
            let i = *p.esc_idx.add(k) as usize;
            if i >= end {
                break;
            }
            *dst.add(r * dst_stride + (i - s)) =
                f32::from_bits(((*p.esc_val.add(k) as u32) << 24) | ((*p.lo.add(i) as u32) << 16));
            k += 1;
        }
        *cursor = k;
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_p12_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for j in 0..n {
            let i = s + j;
            let byte = *hi4.add(i >> 1);
            let code = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            *out.add(j) = f32::from_bits(
                ((*table.add(code as usize) as u32) << 24) | ((*lo.add(i) as u32) << 16),
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_p12_avx2<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let tbl = _mm_loadu_si128(table as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    let nb = n / LANES;
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        if s & 1 != 0 {
            widen_p12_scalar::<1>(out, 0, lo, hi4, table, s, 0, n);
            continue;
        }
        for k in 0..nb {
            let c = k * LANES;
            let i = s + c;

            let packed = _mm_loadl_epi64(hi4.add(i >> 1) as *const __m128i);
            let even = _mm_and_si128(packed, mask);
            let odd = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
            let codes = _mm_unpacklo_epi8(even, odd);

            let highs = _mm_shuffle_epi8(tbl, codes);

            let lob = _mm_loadu_si128(lo.add(i) as *const __m128i);
            let w0 = _mm_unpacklo_epi8(lob, highs);
            let w1 = _mm_unpackhi_epi8(lob, highs);
            _mm256_storeu_ps(
                out.add(c),
                _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w0))),
            );
            _mm256_storeu_ps(
                out.add(c + 8),
                _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w1))),
            );
        }
        if nb * LANES < n {
            widen_p12_scalar::<1>(
                out.add(nb * LANES),
                0,
                lo,
                hi4,
                table,
                s + nb * LANES,
                0,
                n - nb * LANES,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_p12_avx512<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let tbl = _mm_loadu_si128(table as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    let nb = n / LANES;
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        if s & 1 != 0 {
            widen_p12_scalar::<1>(out, 0, lo, hi4, table, s, 0, n);
            continue;
        }
        for k in 0..nb {
            let c = k * LANES;
            let i = s + c;
            let packed = _mm_loadl_epi64(hi4.add(i >> 1) as *const __m128i);
            let even = _mm_and_si128(packed, mask);
            let odd = _mm_and_si128(_mm_srli_epi16::<4>(packed), mask);
            let codes = _mm_unpacklo_epi8(even, odd);
            let highs = _mm_shuffle_epi8(tbl, codes);
            let lob = _mm_loadu_si128(lo.add(i) as *const __m128i);
            let w0 = _mm_unpacklo_epi8(lob, highs);
            let w1 = _mm_unpackhi_epi8(lob, highs);
            let w = _mm256_set_m128i(w1, w0);
            _mm512_storeu_ps(
                out.add(c),
                _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(w))),
            );
        }
        if nb * LANES < n {
            widen_p12_scalar::<1>(
                out.add(nb * LANES),
                0,
                lo,
                hi4,
                table,
                s + nb * LANES,
                0,
                n - nb * LANES,
            );
        }
    }
}

// 16 codes come from 8 nibble bytes (low nibble = even element), the exponent bytes from one
// table lookup, and each word is (exponent << 24) | (low byte << 16): zip the two byte streams
// and widen the 16-bit pairs with a 16-bit shift, as the AVX2 path does with unpack + shift.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_p12_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let tbl = vld1q_u8(table);
    let mask = vdup_n_u8(0x0F);
    let nb = n / LANES;
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        if s & 1 != 0 {
            widen_p12_scalar::<1>(out, 0, lo, hi4, table, s, 0, n);
            continue;
        }
        for k in 0..nb {
            let c = k * LANES;
            let i = s + c;

            let packed = vld1_u8(hi4.add(i >> 1));
            let even = vand_u8(packed, mask);
            let odd = vshr_n_u8::<4>(packed);
            let z = vzip_u8(even, odd);
            let codes = vcombine_u8(z.0, z.1);

            let highs = vqtbl1q_u8(tbl, codes);

            let lob = vld1q_u8(lo.add(i));
            let lh = vzipq_u8(lob, highs);
            let p0 = vreinterpretq_u16_u8(lh.0);
            let p1 = vreinterpretq_u16_u8(lh.1);
            vst1q_f32(
                out.add(c),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p0))),
            );
            vst1q_f32(
                out.add(c + 4),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p0)),
            );
            vst1q_f32(
                out.add(c + 8),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p1))),
            );
            vst1q_f32(
                out.add(c + 12),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p1)),
            );
        }
        for j in (nb * LANES)..n {
            let i = s + j;
            let byte = *hi4.add(i >> 1);
            let code = if i & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            *out.add(j) = f32::from_bits(
                ((*table.add(code as usize) as u32) << 24) | ((*lo.add(i) as u32) << 16),
            );
        }
    }
}

macro_rules! def_task_p12 {
    ($group:ident, $task:ident, $widen:ident, $accum:ident $(, $feat:literal)?) => {
        def_task!($group, $task, P12, $accum $(, $feat)?,
            prep |p, r, cols, cur| { p12_cursors::<R>(p, r, cols, cur); },
            tile |scratch, wide, x, i0, bt, p, r, c0, cols, n, cur| {
                $widen::<R>(wide, COL_TILE, p.lo, p.hi4, p.table, r * cols + c0, cols, n);
                patch_escapes::<R>(wide, COL_TILE, p, r * cols + c0, cols, n, &mut cur);
                false
            });
    };
}

def_task_p12!(
    group_p12_scalar,
    task_p12_scalar,
    widen_p12_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_p12!(group_p12_neon, task_p12_neon, widen_p12_neon, accum_neon);
#[cfg(target_arch = "x86_64")]
def_task_p12!(
    group_p12_avx2,
    task_p12_avx2,
    widen_p12_avx2,
    accum_avx2,
    "avx2,fma"
);
#[cfg(target_arch = "x86_64")]
def_task_p12!(
    group_p12_avx512,
    task_p12_avx512,
    widen_p12_avx512,
    accum_avx512,
    "avx512f,avx512bw,avx2,fma"
);

// ---------------------------------------------------------------------------------------------------
// MXFP4, the form gpt-oss ships its experts in: 32 weights per block, 16 bytes of packed fp4 (e2m1, the
// earlier weight of a pair in the LOW nibble) and one uint8 e8m0 exponent, weight = fp4 * 2^(scale-127).
// The sixteen fp4 values are exact in bf16 (one mantissa bit at most), so the code table is kept as the
// two byte halves of that bf16 pattern and a code becomes an f32 by the same table-lookup, zip and
// 16-bit shift the 12-bit path uses; the block's scale is one exact power of two and multiplies it. Each
// weight is therefore one exact value times one exact power of two - a single rounding, the one
// `torch.ldexp` makes - and the scalar, AVX2 and NEON widenings write the same bits. Everything after
// the widening is the shared `accf_*` accumulation, so an MXFP4 matvec is bit-identical across the three
// ISAs exactly as the bf16 and 12-bit ones are. `cols` must be a multiple of 32 (a block never straddles
// a row), which every gpt-oss shape is.

pub(crate) const MX_BLOCK: usize = 32;
const MX_BLOCK_BYTES: usize = 16;
/// ggml's MXFP4 block: the e8m0 scale byte, then the 16 nibble bytes, back to back.
const MX_GGML_BYTES: usize = 17;

/// The high byte of each fp4 code's bf16 pattern, indexed by the nibble.
const FP4_HI: [u8; 16] = [
    0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0,
];
/// The low byte of each fp4 code's bf16 pattern.
const FP4_LO: [u8; 16] = [
    0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
];

#[inline(always)]
fn fp4_value(code: u8) -> f32 {
    let i = (code & 0x0F) as usize;
    f32::from_bits(((FP4_HI[i] as u32) << 24) | ((FP4_LO[i] as u32) << 16))
}

/// `2^(s - 127)` exactly. `s == 0` is the one scale whose power of two is not a normal f32 exponent
/// field, so it is built as `2^-126 * 0.5`.
#[inline(always)]
fn mx_scale(s: u8) -> f32 {
    let f = f32::from_bits(((if s == 0 { 1 } else { s }) as u32) << 23);
    if s == 0 {
        f * 0.5
    } else {
        f
    }
}

/// One MXFP4 matrix in either layout. The checkpoint's (`ggml == false`): `blocks` holds 16-byte blocks
/// whose byte `j` packs weights `2j` (low nibble) and `2j+1` (high), `scales` one e8m0 byte a block.
/// ggml's (`ggml == true`, a GGUF's): `blocks` holds 17-byte blocks, the scale first, then 16 bytes whose
/// low nibbles are weights 0..15 and high nibbles 16..31; `scales` is unused.
#[derive(Clone, Copy)]
struct Mx4 {
    blocks: *const u8,
    scales: *const u8,
    ggml: bool,
}

/// (the 16 data bytes, the scale) of block `g` in either layout.
#[inline(always)]
unsafe fn mx4_block(p: Mx4, g: usize) -> (*const u8, f32) {
    if p.ggml {
        let at = p.blocks.add(g * MX_GGML_BYTES);
        (at.add(1), mx_scale(*at))
    } else {
        (p.blocks.add(g * MX_BLOCK_BYTES), mx_scale(*p.scales.add(g)))
    }
}

// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Mx4 {}
unsafe impl Sync for Mx4 {}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_mxfp4_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Mx4,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / MX_BLOCK) {
            let g = s / MX_BLOCK + blk;
            let (bytes, sc) = mx4_block(p, g);
            let o = out.add(blk * MX_BLOCK);
            for j in 0..MX_BLOCK_BYTES {
                let byte = *bytes.add(j);
                if p.ggml {
                    *o.add(j) = fp4_value(byte & 0x0F) * sc;
                    *o.add(MX_BLOCK_BYTES + j) = fp4_value(byte >> 4) * sc;
                } else {
                    *o.add(2 * j) = fp4_value(byte & 0x0F) * sc;
                    *o.add(2 * j + 1) = fp4_value(byte >> 4) * sc;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_mxfp4_avx2<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Mx4,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    let thi = _mm_loadu_si128(FP4_HI.as_ptr() as *const __m128i);
    let tlo = _mm_loadu_si128(FP4_LO.as_ptr() as *const __m128i);
    let mask = _mm_set1_epi8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / MX_BLOCK) {
            let g = s / MX_BLOCK + blk;
            let (bytes, sc) = mx4_block(p, g);
            let v = _mm_loadu_si128(bytes as *const __m128i);
            let sv = _mm256_set1_ps(sc);
            let o = out.add(blk * MX_BLOCK);
            let even = _mm_and_si128(v, mask);
            let odd = _mm_and_si128(_mm_srli_epi16::<4>(v), mask);
            // the codes in weight order: ggml's low nibbles are weights 0..15 as they are, the checkpoint's
            // alternate low and high
            let halves = if p.ggml {
                [even, odd]
            } else {
                [_mm_unpacklo_epi8(even, odd), _mm_unpackhi_epi8(even, odd)]
            };
            for (h, codes) in halves.iter().enumerate() {
                let hi = _mm_shuffle_epi8(thi, *codes);
                let lo = _mm_shuffle_epi8(tlo, *codes);
                let w0 = _mm_unpacklo_epi8(lo, hi);
                let w1 = _mm_unpackhi_epi8(lo, hi);
                let f0 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w0)));
                let f1 = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(w1)));
                _mm256_storeu_ps(o.add(h * 16), _mm256_mul_ps(f0, sv));
                _mm256_storeu_ps(o.add(h * 16 + 8), _mm256_mul_ps(f1, sv));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_mxfp4_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Mx4,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let thi = vld1q_u8(FP4_HI.as_ptr());
    let tlo = vld1q_u8(FP4_LO.as_ptr());
    let mask = vdupq_n_u8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / MX_BLOCK) {
            let g = s / MX_BLOCK + blk;
            let (bytes, sc) = mx4_block(p, g);
            let v = vld1q_u8(bytes);
            let sv = vdupq_n_f32(sc);
            let o = out.add(blk * MX_BLOCK);
            let even = vandq_u8(v, mask);
            let odd = vshrq_n_u8::<4>(v);
            // the codes in weight order: ggml's low nibbles are weights 0..15 as they are, the checkpoint's
            // alternate low and high
            let halves = if p.ggml {
                [even, odd]
            } else {
                let z = vzipq_u8(even, odd);
                [z.0, z.1]
            };
            for (h, codes) in halves.iter().enumerate() {
                let hi = vqtbl1q_u8(thi, *codes);
                let lo = vqtbl1q_u8(tlo, *codes);
                let lh = vzipq_u8(lo, hi);
                let p0 = vreinterpretq_u16_u8(lh.0);
                let p1 = vreinterpretq_u16_u8(lh.1);
                let q = o.add(h * 16);
                vst1q_f32(
                    q,
                    vmulq_f32(
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p0))),
                        sv,
                    ),
                );
                vst1q_f32(
                    q.add(4),
                    vmulq_f32(vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p0)), sv),
                );
                vst1q_f32(
                    q.add(8),
                    vmulq_f32(
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(p1))),
                        sv,
                    ),
                );
                vst1q_f32(
                    q.add(12),
                    vmulq_f32(vreinterpretq_f32_u32(vshll_high_n_u16::<16>(p1)), sv),
                );
            }
        }
    }
}

macro_rules! def_task_mx4 {
    ($group:ident, $task:ident, $widen:ident, $accum:ident $(, $feat:literal)?) => {
        def_task!($group, $task, Mx4, $accum $(, $feat)?,
            prep |p, r, cols, cur| {},
            tile |scratch, wide, x, i0, bt, p, r, c0, cols, n, cur| {
                $widen::<R>(wide, COL_TILE, p, r * cols + c0, cols, n);
                false
            });
    };
}

def_task_mx4!(
    group_mx4_scalar,
    task_mx4_scalar,
    widen_mxfp4_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_mx4!(group_mx4_neon, task_mx4_neon, widen_mxfp4_neon, accum_neon);
#[cfg(target_arch = "x86_64")]
def_task_mx4!(
    group_mx4_avx2,
    task_mx4_avx2,
    widen_mxfp4_avx2,
    accum_avx2,
    "avx2,fma"
);
// AVX-512 machines run the AVX2 widening (the code tables are 16 bytes, a natural `pshufb`) into the
// AVX-512 accumulation, so the arithmetic is the wider path's and the bits are still the same.
#[cfg(target_arch = "x86_64")]
def_task_mx4!(
    group_mx4_avx512,
    task_mx4_avx512,
    widen_mxfp4_avx2,
    accum_avx512,
    "avx512f,avx512bw,avx2,fma"
);

// ---------------------------------------------------------------------------------------------------
// FP8, the form fine-grained FP8 checkpoints ship their linears in: one e4m3fn byte a weight (1 sign, 4
// exponent bits biased by 7, 3 mantissa bits, no infinity, 0x7F/0xFF NaN) and an f32 scale per block of a
// `[sr, sc]` grid over the matrix, the block `rows / sr` by `cols / sc` (a 1x1 grid is one per-tensor
// scale). weight = bf16(e4m3 * scale): the engine holds every checkpoint's weights as bf16, so the matvec
// reads the value a bf16 slot of the widened matrix holds, on the CPU as on MLX and the card. Every e4m3
// value is exact in f32 and is built from its bits by integer arithmetic (never a subnormal f32, so a
// flush-to-zero mode cannot touch it), multiplied by its scale once - the rounding `q.float() * scale`
// makes - and rounded to bf16 by integer arithmetic (to nearest, ties to even, as torch's cast), so the
// scalar, AVX2 and NEON widenings write the same bits and the matvec is bit-identical across ISAs.

/// f32 bits of an e4m3 normal's magnitude once its exponent is rebased (7 -> 127): `(b & 0x7F) << 20`
/// plus this.
const F8_REBIAS: u32 = 120 << 23;
/// The subnormals (exponent field 0) are `m * 2^-9`.
const F8_SUB: f32 = 1.0 / 512.0;

/// `f` rounded to the nearest bf16, ties to even, as an f32 (a quiet NaN stays one).
#[inline(always)]
fn round_bf16(f: f32) -> f32 {
    let b = f.to_bits();
    f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
}

#[inline(always)]
fn f8_value(b: u8) -> f32 {
    let a = (b & 0x7F) as u32;
    let mag = if a == 0x7F {
        f32::NAN
    } else if a < 8 {
        a as f32 * F8_SUB
    } else {
        f32::from_bits((a << 20) + F8_REBIAS)
    };
    f32::from_bits(mag.to_bits() | (((b & 0x80) as u32) << 24))
}

/// One FP8 matrix: `w` its `rows * cols` e4m3 bytes row-major, `scales` its `[sr, sc]` f32 grid row-major,
/// a block `bm` rows by `bn` columns.
#[derive(Clone, Copy)]
struct F8 {
    w: *const u8,
    scales: *const f32,
    sc: usize,
    bm: usize,
    bn: usize,
}

// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for F8 {}
unsafe impl Sync for F8 {}

/// Rows `start / row_stride ..` of `p`, columns `start % row_stride` for `n`, into `dst` as f32: each run
/// of columns under one scale handed to `run(src, scale, out, len)`.
#[inline(always)]
unsafe fn f8_runs<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: F8,
    start: usize,
    row_stride: usize,
    n: usize,
    mut run: impl FnMut(*const u8, f32, *mut f32, usize),
) {
    let (r0, c0) = (start / row_stride, start % row_stride);
    for r in 0..R {
        let srow = p.scales.add(((r0 + r) / p.bm) * p.sc);
        let src = p.w.add((r0 + r) * row_stride);
        let out = dst.add(r * dst_stride);
        let mut c = c0;
        while c < c0 + n {
            let end = ((c / p.bn + 1) * p.bn).min(c0 + n);
            run(src.add(c), *srow.add(c / p.bn), out.add(c - c0), end - c);
            c = end;
        }
    }
}

#[inline]
unsafe fn widen_f8_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: F8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    f8_runs::<R>(
        dst,
        dst_stride,
        p,
        start,
        row_stride,
        n,
        |src, s, out, len| {
            for j in 0..len {
                *out.add(j) = round_bf16(f8_value(*src.add(j)) * s);
            }
        },
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_f8_avx2<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: F8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::x86_64::*;
    f8_runs::<R>(
        dst,
        dst_stride,
        p,
        start,
        row_stride,
        n,
        |src, s, out, len| {
            let sv = _mm256_set1_ps(s);
            let m7f = _mm256_set1_epi32(0x7F);
            let rebias = _mm256_set1_epi32(F8_REBIAS as i32);
            let sub = _mm256_set1_ps(F8_SUB);
            let nan = _mm256_set1_ps(f32::NAN);
            let mut j = 0usize;
            while j + 8 <= len {
                let u = _mm256_cvtepu8_epi32(_mm_loadl_epi64(src.add(j) as *const __m128i));
                let a = _mm256_and_si256(u, m7f);
                let normal =
                    _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32::<20>(a), rebias));
                let subn = _mm256_mul_ps(_mm256_cvtepi32_ps(a), sub);
                let is_sub = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(8), a));
                let is_nan = _mm256_castsi256_ps(_mm256_cmpeq_epi32(a, m7f));
                let mag = _mm256_blendv_ps(_mm256_blendv_ps(normal, subn, is_sub), nan, is_nan);
                let sign = _mm256_slli_epi32::<24>(_mm256_andnot_si256(m7f, u));
                let v = _mm256_or_ps(mag, _mm256_castsi256_ps(sign));
                // to the nearest bf16, ties to even (`round_bf16`)
                let w = _mm256_castps_si256(_mm256_mul_ps(v, sv));
                let odd = _mm256_and_si256(_mm256_srli_epi32::<16>(w), _mm256_set1_epi32(1));
                let up = _mm256_add_epi32(w, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), odd));
                let r = _mm256_and_si256(up, _mm256_set1_epi32(0xFFFF_0000u32 as i32));
                _mm256_storeu_ps(out.add(j), _mm256_castsi256_ps(r));
                j += 8;
            }
            for k in j..len {
                *out.add(k) = round_bf16(f8_value(*src.add(k)) * s);
            }
        },
    );
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn widen_f8_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: F8,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    f8_runs::<R>(
        dst,
        dst_stride,
        p,
        start,
        row_stride,
        n,
        |src, s, out, len| {
            let sv = vdupq_n_f32(s);
            let m7f = vdupq_n_u32(0x7F);
            let rebias = vdupq_n_u32(F8_REBIAS);
            let eight = vdupq_n_u32(8);
            let nan = vdupq_n_f32(f32::NAN);
            let mut j = 0usize;
            while j + 8 <= len {
                let h = vmovl_u8(vld1_u8(src.add(j)));
                for (q, u) in [vmovl_u16(vget_low_u16(h)), vmovl_high_u16(h)]
                    .into_iter()
                    .enumerate()
                {
                    let a = vandq_u32(u, m7f);
                    let normal = vreinterpretq_f32_u32(vaddq_u32(vshlq_n_u32::<20>(a), rebias));
                    let subn = vmulq_n_f32(vcvtq_f32_u32(a), F8_SUB);
                    let mag = vbslq_f32(vcltq_u32(a, eight), subn, normal);
                    let mag = vbslq_f32(vceqq_u32(a, m7f), nan, mag);
                    let sign = vshlq_n_u32::<24>(vbicq_u32(u, m7f));
                    let v = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(mag), sign));
                    // to the nearest bf16, ties to even (`round_bf16`)
                    let w = vreinterpretq_u32_f32(vmulq_f32(v, sv));
                    let odd = vandq_u32(vshrq_n_u32::<16>(w), vdupq_n_u32(1));
                    let up = vaddq_u32(w, vaddq_u32(vdupq_n_u32(0x7FFF), odd));
                    let r = vandq_u32(up, vdupq_n_u32(0xFFFF_0000));
                    vst1q_f32(out.add(j + 4 * q), vreinterpretq_f32_u32(r));
                }
                j += 8;
            }
            for k in j..len {
                *out.add(k) = round_bf16(f8_value(*src.add(k)) * s);
            }
        },
    );
}

macro_rules! def_task_f8 {
    ($group:ident, $task:ident, $widen:ident, $accum:ident $(, $feat:literal)?) => {
        def_task!($group, $task, F8, $accum $(, $feat)?,
            prep |p, r, cols, cur| {},
            tile |scratch, wide, x, i0, bt, p, r, c0, cols, n, cur| {
                $widen::<R>(wide, COL_TILE, p, r * cols + c0, cols, n);
                false
            });
    };
}

def_task_f8!(
    group_f8_scalar,
    task_f8_scalar,
    widen_f8_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_f8!(group_f8_neon, task_f8_neon, widen_f8_neon, accum_neon);
#[cfg(target_arch = "x86_64")]
def_task_f8!(
    group_f8_avx2,
    task_f8_avx2,
    widen_f8_avx2,
    accum_avx2,
    "avx2,fma"
);
// AVX-512 machines run the AVX2 widening into the AVX-512 accumulation, as MXFP4 does.
#[cfg(target_arch = "x86_64")]
def_task_f8!(
    group_f8_avx512,
    task_f8_avx512,
    widen_f8_avx2,
    accum_avx512,
    "avx512f,avx512bw,avx2,fma"
);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Avx512,

    Avx2,

    /// arm64's baseline vector unit (every Apple silicon and arm64 Linux core).
    Neon,

    Scalar,
}

fn detect_isa() -> Isa {
    #[cfg(target_arch = "aarch64")]
    {
        Isa::Neon
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        #[cfg(target_arch = "x86_64")]
        {
            let base =
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
            if base
                && std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512bw")
            {
                return Isa::Avx512;
            }
            if base {
                return Isa::Avx2;
            }
        }
        Isa::Scalar
    }
}

/// The kernel path for this machine, decided once. `BTB_NATIVE_ISA=scalar` (or `avx2` on a
/// machine that has it) in the environment at the first call pins a narrower path.
pub fn isa() -> Isa {
    static CACHED: OnceLock<Isa> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let detected = detect_isa();
        match std::env::var("BTB_NATIVE_ISA").ok().as_deref() {
            Some("scalar") => Isa::Scalar,
            #[cfg(target_arch = "x86_64")]
            Some("avx2") if detected != Isa::Scalar => Isa::Avx2,
            _ => detected,
        }
    })
}

pub(crate) fn default_threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// The most worker threads a call may ask for. Past it a request is refused (`ERR_DOMAIN`) rather than
/// spawning that many OS threads and caching the pool forever.
pub const MAX_THREADS: usize = 1024;

/// The thread count a call runs with: `0` is every core, anything up to [`MAX_THREADS`] is taken as given,
/// more is `None` (the entry point returns `ERR_DOMAIN`).
#[inline]
pub(crate) fn resolve_threads(threads: usize) -> Option<usize> {
    match threads {
        0 => Some(default_threads()),
        t if t <= MAX_THREADS => Some(t),
        _ => None,
    }
}

/// Byte-level checks that cost nothing per element: a pointer naturally aligned for its element type (the
/// scalar and tail paths dereference `u16` / `f32` / `i32` directly; the SIMD loads are unaligned-safe).
#[inline]
pub(crate) fn aligned<T>(p: *const T) -> bool {
    (p as usize).is_multiple_of(std::mem::align_of::<T>())
}

/// The most elements a buffer may hold and still have a byte size a pointer offset can express (`isize`).
pub(crate) const MAX_ELEMS: usize = isize::MAX as usize / 4;

/// The ISA-selected task of a weight form over its arguments.
macro_rules! by_isa {
    ($isa:expr, $scalar:ident, $avx2:ident, $avx512:ident, $neon:ident, ($($a:expr),* $(,)?)) => {
        match $isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => $avx512($($a),*),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => $avx2($($a),*),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => $neon($($a),*),
            _ => $scalar($($a),*),
        }
    };
}

/// A matrix in one of its stored forms, run as tasks over row ranges.
trait Weights: Copy + Send + Sync {
    /// Rows `r0..r1` of `rows_total` times the `b` vectors of `x` into `y` (`[b, rows_total]`).
    ///
    /// # Safety
    /// The arrays valid for the shape, as the C ABI states.
    #[allow(clippy::too_many_arguments)]
    unsafe fn rows(
        self,
        isa: Isa,
        cols: usize,
        x: *const f32,
        b: usize,
        y: *mut f32,
        rows_total: usize,
        r0: usize,
        r1: usize,
    );
}

impl Weights for Bf16 {
    #[inline]
    unsafe fn rows(
        self,
        isa: Isa,
        cols: usize,
        x: *const f32,
        b: usize,
        y: *mut f32,
        rows_total: usize,
        r0: usize,
        r1: usize,
    ) {
        by_isa!(
            isa,
            task_scalar,
            task_avx2,
            task_avx512,
            task_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

impl Weights for P12 {
    #[inline]
    unsafe fn rows(
        self,
        isa: Isa,
        cols: usize,
        x: *const f32,
        b: usize,
        y: *mut f32,
        rows_total: usize,
        r0: usize,
        r1: usize,
    ) {
        by_isa!(
            isa,
            task_p12_scalar,
            task_p12_avx2,
            task_p12_avx512,
            task_p12_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

impl Weights for Mx4 {
    #[inline]
    unsafe fn rows(
        self,
        isa: Isa,
        cols: usize,
        x: *const f32,
        b: usize,
        y: *mut f32,
        rows_total: usize,
        r0: usize,
        r1: usize,
    ) {
        by_isa!(
            isa,
            task_mx4_scalar,
            task_mx4_avx2,
            task_mx4_avx512,
            task_mx4_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

impl Weights for F8 {
    #[inline]
    unsafe fn rows(
        self,
        isa: Isa,
        cols: usize,
        x: *const f32,
        b: usize,
        y: *mut f32,
        rows_total: usize,
        r0: usize,
        r1: usize,
    ) {
        by_isa!(
            isa,
            task_f8_scalar,
            task_f8_avx2,
            task_f8_avx512,
            task_f8_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

/// One matrix-times-vectors task: the weights, the vectors, the output and their shape.
#[derive(Clone, Copy)]
struct Task<W> {
    w: W,
    x: *const f32,
    y: *mut f32,
    rows: usize,
    cols: usize,
    b: usize,
}

// SAFETY: a task's threads take disjoint row ranges and each writes only its own rows of `y`; the
// caller's buffers do not overlap (the C ABI's contract)
unsafe impl<W: Weights> Send for Task<W> {}
unsafe impl<W: Weights> Sync for Task<W> {}

/// The task over its rows: on the calling thread for one row or one thread, else spread over the pool.
///
/// # Safety
/// `t` valid for its shape.
unsafe fn run_one<W: Weights>(t: Task<W>, threads: usize) -> i32 {
    let isa = isa();
    let global = rayon::current_num_threads().max(1);
    let nt = match resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    if nt <= 1 || t.rows == 1 {
        t.w.rows(isa, t.cols, t.x, t.b, t.y, t.rows, 0, t.rows);
        return OK;
    }
    spread(t.rows, nt, global, move |r0, r1| {
        let t = t; // the whole task captured, not its pointer fields one by one
        unsafe { t.w.rows(isa, t.cols, t.x, t.b, t.y, t.rows, r0, r1) }
    });
    OK
}

/// `tasks` under one dispatch: every task's rows cut into chunks of one size (`total` rows over the pool)
/// and spread together, so many small tasks pay one barrier.
///
/// # Safety
/// Every task valid for its shape.
unsafe fn run_group<W: Weights>(tasks: &[Task<W>], total: usize, threads: usize) -> i32 {
    let isa = isa();
    let global = rayon::current_num_threads().max(1);
    let nt = match resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    if nt <= 1 {
        for t in tasks {
            t.w.rows(isa, t.cols, t.x, t.b, t.y, t.rows, 0, t.rows);
        }
        return OK;
    }
    let chunk = total.div_ceil(nt * SPLIT).max(ROW_UNROLL);
    let mut work: Vec<(usize, usize)> = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        let mut r0 = 0usize;
        while r0 < t.rows {
            work.push((i, r0));
            r0 += chunk;
        }
    }
    dispatch(nt, global, work.len(), |k| {
        let (i, r0) = work[k];
        let t = tasks[i];
        let r1 = (r0 + chunk).min(t.rows);
        unsafe { t.w.rows(isa, t.cols, t.x, t.b, t.y, t.rows, r0, r1) }
    });
    OK
}

pub(crate) fn pool_for(nt: usize) -> Option<Arc<rayon::ThreadPool>> {
    type PoolCache = OnceLock<Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>>>;
    static POOLS: PoolCache = OnceLock::new();
    let mut guard = POOLS.get_or_init(|| Mutex::new(Vec::new())).lock().ok()?;
    if let Some((_, p)) = guard.iter().find(|(n, _)| *n == nt) {
        return Some(Arc::clone(p));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .thread_name(move |i| format!("btb-gemv-{nt}-{i}"))
            .build()
            .ok()?,
    );
    guard.push((nt, Arc::clone(&pool)));
    Some(pool)
}

pub(crate) unsafe fn gemv_core(
    w: *const u16,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    if w.is_null() || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if rows == 0 || cols == 0 || b == 0 {
        return ERR_DOMAIN;
    }

    if !shape_ok(rows, cols, b) || !aligned(w) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }
    run_one(
        Task {
            w: Bf16(w),
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

fn dispatch<F: Fn(usize) + Send + Sync>(nt: usize, global: usize, nchunks: usize, body: F) {
    if nt == global {
        (0..nchunks).into_par_iter().for_each(&body);
    } else if let Some(pool) = pool_for(nt) {
        pool.install(|| (0..nchunks).into_par_iter().for_each(&body));
    } else {
        (0..nchunks).into_par_iter().for_each(&body);
    }
}

fn spread<F: Fn(usize, usize) + Send + Sync>(rows: usize, nt: usize, global: usize, each: F) {
    let chunk = rows.div_ceil(nt * SPLIT).max(ROW_UNROLL);
    dispatch(nt, global, rows.div_ceil(chunk), |k| {
        let r0 = k * chunk;
        each(r0, (r0 + chunk).min(rows));
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_group_core(
    n: usize,
    w: *const *const u16,
    rows: *const usize,
    cols: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
) -> i32 {
    if w.is_null() || rows.is_null() || cols.is_null() || x.is_null() || b.is_null() || y.is_null()
    {
        return ERR_NULL;
    }
    if n == 0 {
        return ERR_DOMAIN;
    }

    let mut tasks: Vec<Task<Bf16>> = Vec::with_capacity(n);
    let mut total = 0usize;
    for t in 0..n {
        let job = Task {
            w: Bf16(*w.add(t)),
            x: *x.add(t),
            y: *y.add(t),
            rows: *rows.add(t),
            cols: *cols.add(t),
            b: *b.add(t),
        };
        if job.w.0.is_null() || job.x.is_null() || job.y.is_null() {
            return ERR_NULL;
        }
        if job.rows == 0 || job.cols == 0 || job.b == 0 {
            return ERR_DOMAIN;
        }
        if !shape_ok(job.rows, job.cols, job.b)
            || !aligned(job.w.0)
            || !aligned(job.x)
            || !aligned(job.y)
        {
            return ERR_DOMAIN;
        }
        total = match total.checked_add(job.rows) {
            Some(v) => v,
            None => return ERR_DOMAIN,
        };
        tasks.push(job);
    }
    run_group(&tasks, total, threads)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_p12_core(
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    esc_idx: *const i32,
    esc_val: *const u8,
    n_esc: usize,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    if lo.is_null() || hi4.is_null() || table.is_null() || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if n_esc > 0 && (esc_idx.is_null() || esc_val.is_null()) {
        return ERR_NULL;
    }
    if rows == 0 || cols == 0 || b == 0 {
        return ERR_DOMAIN;
    }
    if !shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }
    let n = rows * cols;

    if n_esc > 0 {
        // the escape list is read as a slice: it must be 4-aligned and cannot hold more entries than there
        // are weights (the ascending check below only stops at the first bad entry)
        if n_esc > n || !aligned(esc_idx) {
            return ERR_DOMAIN;
        }
        let idx = std::slice::from_raw_parts(esc_idx, n_esc);
        let mut prev: i64 = -1;
        for &v in idx {
            let vi = v as i64;
            if vi <= prev || vi >= n as i64 {
                return ERR_DOMAIN;
            }
            prev = vi;
        }
    }

    let p = P12 {
        lo,
        hi4,
        table,
        esc_idx,
        esc_val,
        n_esc,
    };
    run_one(
        Task {
            w: p,
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

/// The element counts a bf16 / 12-bit call touches fit `usize`, and their byte sizes fit a pointer offset.
#[inline]
fn shape_ok(rows: usize, cols: usize, b: usize) -> bool {
    matches!(rows.checked_mul(cols), Some(n) if n <= MAX_ELEMS)
        && matches!(b.checked_mul(cols), Some(n) if n <= MAX_ELEMS)
        && matches!(b.checked_mul(rows), Some(n) if n <= MAX_ELEMS)
}

#[inline]
fn mx4_shape_ok(rows: usize, cols: usize, b: usize) -> bool {
    rows != 0 && cols != 0 && b != 0 && cols.is_multiple_of(MX_BLOCK) && shape_ok(rows, cols, b)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_mxfp4_core(
    blocks: *const u8,
    scales: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
    ggml: bool,
) -> i32 {
    if blocks.is_null() || (scales.is_null() && !ggml) || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if !mx4_shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }

    run_one(
        Task {
            w: Mx4 {
                blocks,
                scales,
                ggml,
            },
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_mxfp4_group_core(
    n: usize,
    blocks: *const *const u8,
    scales: *const *const u8,
    rows: *const usize,
    cols: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
    ggml: bool,
) -> i32 {
    if blocks.is_null()
        || (scales.is_null() && !ggml)
        || rows.is_null()
        || cols.is_null()
        || x.is_null()
        || b.is_null()
        || y.is_null()
    {
        return ERR_NULL;
    }
    if n == 0 {
        return ERR_DOMAIN;
    }

    let mut tasks: Vec<Task<Mx4>> = Vec::with_capacity(n);
    let mut total = 0usize;
    for t in 0..n {
        let job = Task {
            w: Mx4 {
                blocks: *blocks.add(t),
                scales: if ggml {
                    std::ptr::null()
                } else {
                    *scales.add(t)
                },
                ggml,
            },
            x: *x.add(t),
            y: *y.add(t),
            rows: *rows.add(t),
            cols: *cols.add(t),
            b: *b.add(t),
        };
        if job.w.blocks.is_null()
            || (job.w.scales.is_null() && !ggml)
            || job.x.is_null()
            || job.y.is_null()
        {
            return ERR_NULL;
        }
        if !mx4_shape_ok(job.rows, job.cols, job.b) || !aligned(job.x) || !aligned(job.y) {
            return ERR_DOMAIN;
        }
        total = match total.checked_add(job.rows) {
            Some(v) => v,
            None => return ERR_DOMAIN,
        };
        tasks.push(job);
    }
    run_group(&tasks, total, threads)
}

/// One FP8 matrix checked and built: the grid `[sr, sc]` divides `[rows, cols]` evenly, the scales are
/// f32-aligned, and every pointer is set.
#[allow(clippy::too_many_arguments)]
unsafe fn f8_task(
    w: *const u8,
    scales: *const f32,
    rows: usize,
    cols: usize,
    sr: usize,
    sc: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
) -> Result<Task<F8>, i32> {
    if w.is_null() || scales.is_null() || x.is_null() || y.is_null() {
        return Err(ERR_NULL);
    }
    if rows == 0 || cols == 0 || b == 0 || sr == 0 || sc == 0 {
        return Err(ERR_DOMAIN);
    }
    if !rows.is_multiple_of(sr)
        || !cols.is_multiple_of(sc)
        || !shape_ok(rows, cols, b)
        || !aligned(scales)
        || !aligned(x)
        || !aligned(y)
    {
        return Err(ERR_DOMAIN);
    }
    Ok(Task {
        w: F8 {
            w,
            scales,
            sc,
            bm: rows / sr,
            bn: cols / sc,
        },
        x,
        y,
        rows,
        cols,
        b,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_fp8_core(
    w: *const u8,
    scales: *const f32,
    rows: usize,
    cols: usize,
    sr: usize,
    sc: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    match f8_task(w, scales, rows, cols, sr, sc, x, b, y) {
        Ok(t) => run_one(t, threads),
        Err(code) => code,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_fp8_group_core(
    n: usize,
    w: *const *const u8,
    scales: *const *const f32,
    rows: *const usize,
    cols: *const usize,
    sr: *const usize,
    sc: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
) -> i32 {
    if w.is_null()
        || scales.is_null()
        || rows.is_null()
        || cols.is_null()
        || sr.is_null()
        || sc.is_null()
        || x.is_null()
        || b.is_null()
        || y.is_null()
    {
        return ERR_NULL;
    }
    if n == 0 {
        return ERR_DOMAIN;
    }
    let mut tasks: Vec<Task<F8>> = Vec::with_capacity(n);
    let mut total = 0usize;
    for t in 0..n {
        let job = match f8_task(
            *w.add(t),
            *scales.add(t),
            *rows.add(t),
            *cols.add(t),
            *sr.add(t),
            *sc.add(t),
            *x.add(t),
            *b.add(t),
            *y.add(t),
        ) {
            Ok(job) => job,
            Err(code) => return code,
        };
        total = match total.checked_add(job.rows) {
            Some(v) => v,
            None => return ERR_DOMAIN,
        };
        tasks.push(job);
    }
    run_group(&tasks, total, threads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_row(w: &[u16], x: &[f32]) -> f64 {
        let mut acc = 0.0f64;
        for (wv, xv) in w.iter().zip(x.iter()) {
            acc += bf16_to_f32(*wv) as f64 * *xv as f64;
        }
        acc
    }

    fn call(w: &[u16], rows: usize, cols: usize, x: &[f32], b: usize, threads: usize) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        let code = unsafe {
            gemv_core(
                w.as_ptr(),
                rows,
                cols,
                x.as_ptr(),
                b,
                y.as_mut_ptr(),
                threads,
            )
        };
        assert_eq!(code, OK);
        y
    }

    #[test]
    fn bf16_widening_is_exact() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);

        for v in [0.5f32, -0.125, 7.0, -1024.0] {
            assert_eq!(bf16_to_f32((v.to_bits() >> 16) as u16), v);
        }
    }

    #[test]
    fn hand_checked_small_case() {
        let w: Vec<u16> = [1.0f32, 2.0, -3.0, 0.5, -1.0, 4.0]
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect();
        let x = [2.0f32, 3.0, 1.0];
        let y = call(&w, 2, 3, &x, 1, 1);
        assert_eq!(y, vec![5.0, 2.0]);
    }

    #[test]
    fn ragged_shapes_match_the_f64_reference() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        for (rows, cols) in [
            (1usize, 1usize),
            (1, 15),
            (3, 17),
            (5, 31),
            (7, 513),
            (16, 16),
            (17, 1025),
            (33, 512),
        ] {
            let w: Vec<u16> = (0..rows * cols)
                .map(|_| (next().to_bits() >> 16) as u16)
                .collect();
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let y = call(&w, rows, cols, &x, 1, 1);
            for r in 0..rows {
                let want = ref_row(&w[r * cols..(r + 1) * cols], &x);
                let scale = w[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(a, b)| (bf16_to_f32(*a) as f64 * *b as f64).abs())
                    .sum::<f64>()
                    .max(1e-30);
                assert!(
                    ((y[r] as f64 - want).abs() / scale) < 1e-6,
                    "{rows}x{cols} row {r}: got {} want {want}",
                    y[r]
                );
            }
        }
    }

    #[test]
    fn zero_and_null_are_errors() {
        let w = [0u16; 4];
        let x = [0.0f32; 4];
        let mut y = [0.0f32; 4];
        unsafe {
            assert_eq!(
                gemv_core(std::ptr::null(), 2, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_NULL
            );
            assert_eq!(
                gemv_core(w.as_ptr(), 2, 2, std::ptr::null(), 1, y.as_mut_ptr(), 0),
                ERR_NULL
            );
            assert_eq!(
                gemv_core(w.as_ptr(), 2, 2, x.as_ptr(), 1, std::ptr::null_mut(), 0),
                ERR_NULL
            );
            assert_eq!(
                gemv_core(w.as_ptr(), 0, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
            assert_eq!(
                gemv_core(w.as_ptr(), 2, 0, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
            assert_eq!(
                gemv_core(w.as_ptr(), 2, 2, x.as_ptr(), 0, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
            assert_eq!(
                gemv_core(w.as_ptr(), usize::MAX, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
        }
    }

    #[test]
    fn tiling_and_threads_do_not_move_a_single_bit() {
        let (rows, cols) = (61usize, 1039usize);
        let mut state = 0xdead_beef_cafe_babeu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        let w: Vec<u16> = (0..rows * cols)
            .map(|_| (next().to_bits() >> 16) as u16)
            .collect();
        let x: Vec<f32> = (0..3 * cols).map(|_| next()).collect();

        let base = call(&w, rows, cols, &x, 3, 1);
        for t in [2usize, 3, 5, 8, 0] {
            assert_eq!(
                call(&w, rows, cols, &x, 3, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads = {t}"
            );
        }

        for i in 0..3 {
            let solo = call(&w, rows, cols, &x[i * cols..(i + 1) * cols], 1, 0);
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    fn mx4_call(
        blocks: &[u8],
        scales: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        b: usize,
        threads: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        let code = unsafe {
            gemv_mxfp4_core(
                blocks.as_ptr(),
                scales.as_ptr(),
                rows,
                cols,
                x.as_ptr(),
                b,
                y.as_mut_ptr(),
                threads,
                false,
            )
        };
        assert_eq!(code, OK);
        y
    }

    fn mx4_random(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let g = rows * cols / MX_BLOCK;
        let mut state = seed;
        let mut byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        };
        let blocks: Vec<u8> = (0..g * MX_BLOCK_BYTES).map(|_| byte()).collect();
        // exponents a checkpoint actually carries (2^-27 to 2^27), and one zero scale so the
        // step-down branch of `mx_scale` is exercised; 255 would be 2^128, i.e. an infinity whose
        // product with the zero code is a NaN, and is not a weight any quantizer writes
        let mut scales: Vec<u8> = (0..g).map(|_| 100 + byte() % 55).collect();
        scales[0] = 0;
        (blocks, scales)
    }

    /// The same matrix in ggml's layout: 17-byte blocks, the scale first, weights 0..15 in the low
    /// nibbles and 16..31 in the high.
    fn mx4_ggml(blocks: &[u8], scales: &[u8]) -> Vec<u8> {
        let g = scales.len();
        let mut raw = Vec::with_capacity(g * MX_GGML_BYTES);
        for blk in 0..g {
            raw.push(scales[blk]);
            let code = |i: usize| -> u8 {
                let byte = blocks[blk * MX_BLOCK_BYTES + i / 2];
                if i & 1 == 0 {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            };
            for j in 0..MX_BLOCK_BYTES {
                raw.push(code(j) | (code(MX_BLOCK_BYTES + j) << 4));
            }
        }
        raw
    }

    #[test]
    fn mxfp4_ggml_layout_gives_the_checkpoint_layouts_bits() {
        for (rows, cols) in [(1usize, 32usize), (5, 96), (17, 1024), (33, 2880)] {
            let (blocks, scales) = mx4_random(rows, cols, 0x5151_a0a0_c3c3_0f0f ^ rows as u64);
            let raw = mx4_ggml(&blocks, &scales);
            let mut state = 0x1357_9bdf_2468_ace0u64 ^ cols as u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
            };
            for b in [1usize, 3, 16] {
                let x: Vec<f32> = (0..b * cols).map(|_| next()).collect();
                let want = mx4_call(&blocks, &scales, rows, cols, &x, b, 2);
                let mut got = vec![f32::NAN; b * rows];
                let code = unsafe {
                    gemv_mxfp4_core(
                        raw.as_ptr(),
                        std::ptr::null(),
                        rows,
                        cols,
                        x.as_ptr(),
                        b,
                        got.as_mut_ptr(),
                        2,
                        true,
                    )
                };
                assert_eq!(code, OK);
                for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(g.to_bits(), w.to_bits(), "{rows}x{cols} b={b} at {i}");
                }
            }
        }
    }

    /// The value of weight `i` of a matrix, straight from the definition.
    fn mx4_weight(blocks: &[u8], scales: &[u8], i: usize) -> f32 {
        let g = i / MX_BLOCK;
        let j = i % MX_BLOCK;
        let byte = blocks[g * MX_BLOCK_BYTES + j / 2];
        let code = if j & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        fp4_value(code) * mx_scale(scales[g])
    }

    #[test]
    fn fp4_table_matches_the_e2m1_values() {
        let want = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        for (c, v) in want.iter().enumerate() {
            assert_eq!(fp4_value(c as u8).to_bits(), v.to_bits(), "code {c}");
        }
        assert_eq!(mx_scale(127), 1.0);
        assert_eq!(mx_scale(128), 2.0);
        assert_eq!(mx_scale(126), 0.5);
        assert_eq!(mx_scale(1), f32::from_bits(1 << 23));
        assert_eq!(mx_scale(0), f32::from_bits(1 << 23) * 0.5);
    }

    #[test]
    fn mxfp4_matches_the_f64_reference() {
        for (rows, cols) in [
            (1usize, 32usize),
            (3, 64),
            (5, 96),
            (7, 512),
            (16, 544),
            (17, 1024),
            (33, 2880),
        ] {
            let (blocks, scales) = mx4_random(rows, cols, 0x1234_5678_9abc_def0 ^ rows as u64);
            let mut state = 0xfeed_face_dead_c0deu64 ^ cols as u64;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
            };
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let y = mx4_call(&blocks, &scales, rows, cols, &x, 1, 1);
            for (r, got) in y.iter().enumerate() {
                let mut want = 0.0f64;
                let mut scale = 1e-30f64;
                for (c, xv) in x.iter().enumerate() {
                    let w = mx4_weight(&blocks, &scales, r * cols + c) as f64;
                    want += w * *xv as f64;
                    scale += (w * *xv as f64).abs();
                }
                assert!(
                    ((*got as f64 - want).abs() / scale) < 1e-6,
                    "{rows}x{cols} row {r}: got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn mxfp4_tiling_threads_and_batching_do_not_move_a_single_bit() {
        let (rows, cols) = (67usize, 1056usize);
        let (blocks, scales) = mx4_random(rows, cols, 0xabcd_1234_5678_9999);
        let mut state = 0x0bad_c0ff_ee00_0dd1u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..3 * cols).map(|_| next()).collect();
        let base = mx4_call(&blocks, &scales, rows, cols, &x, 3, 1);
        for t in [2usize, 3, 5, 8, 0] {
            assert_eq!(
                mx4_call(&blocks, &scales, rows, cols, &x, 3, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads = {t}"
            );
        }
        for i in 0..3 {
            let solo = mx4_call(
                &blocks,
                &scales,
                rows,
                cols,
                &x[i * cols..(i + 1) * cols],
                1,
                0,
            );
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    /// Every ISA on this machine widens to the same bits, so the matvec agrees to the bit. On x86 the
    /// scalar, AVX2 and AVX-512 paths are all reachable through `BTB_NATIVE_ISA`; here the widenings are
    /// compared directly, which is the part that differs between them.
    #[test]
    fn mxfp4_widenings_agree_across_isas() {
        let (rows, cols) = (4usize, 512usize);
        let (blocks, scales) = mx4_random(rows, cols, 0x5555_aaaa_3333_cccc);
        let p = Mx4 {
            blocks: blocks.as_ptr(),
            scales: scales.as_ptr(),
            ggml: false,
        };
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        unsafe {
            widen_mxfp4_scalar::<4>(a.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
            #[cfg(target_arch = "aarch64")]
            widen_mxfp4_neon::<4>(b.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") {
                widen_mxfp4_avx2::<4>(b.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
            } else {
                b.copy_from_slice(&a);
            }
        }
        for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(u.to_bits(), v.to_bits(), "lane {i}: {u} vs {v}");
        }
    }

    #[test]
    fn mxfp4_group_matches_the_single_calls() {
        let shapes = [(12usize, 64usize), (7, 128), (20, 96)];
        let mut mats = Vec::new();
        let mut xs = Vec::new();
        for (i, (rows, cols)) in shapes.iter().enumerate() {
            mats.push(mx4_random(*rows, *cols, 0x9e37_79b9_7f4a_7c15 ^ i as u64));
            let mut state = 0x2545_f491_4f6c_dd1du64 ^ (i as u64) << 8;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
            };
            xs.push((0..2 * cols).map(|_| next()).collect::<Vec<f32>>());
        }
        let want: Vec<Vec<f32>> = shapes
            .iter()
            .enumerate()
            .map(|(i, (rows, cols))| mx4_call(&mats[i].0, &mats[i].1, *rows, *cols, &xs[i], 2, 0))
            .collect();
        let mut got: Vec<Vec<f32>> = shapes.iter().map(|(r, _)| vec![0.0f32; 2 * r]).collect();
        let bp: Vec<*const u8> = mats.iter().map(|m| m.0.as_ptr()).collect();
        let sp: Vec<*const u8> = mats.iter().map(|m| m.1.as_ptr()).collect();
        let rp: Vec<usize> = shapes.iter().map(|(r, _)| *r).collect();
        let cp: Vec<usize> = shapes.iter().map(|(_, c)| *c).collect();
        let xp: Vec<*const f32> = xs.iter().map(|v| v.as_ptr()).collect();
        let bs: Vec<usize> = shapes.iter().map(|_| 2usize).collect();
        let mut yp: Vec<*mut f32> = got.iter_mut().map(|v| v.as_mut_ptr()).collect();
        let code = unsafe {
            gemv_mxfp4_group_core(
                shapes.len(),
                bp.as_ptr(),
                sp.as_ptr(),
                rp.as_ptr(),
                cp.as_ptr(),
                xp.as_ptr(),
                bs.as_ptr(),
                yp.as_mut_ptr(),
                0,
                false,
            )
        };
        assert_eq!(code, OK);
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "task {i}"
            );
        }
    }

    #[test]
    fn mxfp4_rejects_bad_shapes() {
        let blocks = [0u8; 64];
        let scales = [127u8; 4];
        let x = [0.0f32; 64];
        let mut y = [0.0f32; 4];
        unsafe {
            assert_eq!(
                gemv_mxfp4_core(
                    std::ptr::null(),
                    scales.as_ptr(),
                    2,
                    64,
                    x.as_ptr(),
                    1,
                    y.as_mut_ptr(),
                    0,
                    false
                ),
                ERR_NULL
            );
            // cols must be a whole number of 32-weight blocks
            assert_eq!(
                gemv_mxfp4_core(
                    blocks.as_ptr(),
                    scales.as_ptr(),
                    2,
                    48,
                    x.as_ptr(),
                    1,
                    y.as_mut_ptr(),
                    0,
                    false
                ),
                ERR_DOMAIN
            );
            assert_eq!(
                gemv_mxfp4_core(
                    blocks.as_ptr(),
                    scales.as_ptr(),
                    0,
                    64,
                    x.as_ptr(),
                    1,
                    y.as_mut_ptr(),
                    0,
                    false
                ),
                ERR_DOMAIN
            );
        }
    }

    /// An e4m3fn byte's value from the definition: (-1)^s * 2^(e-7) * (1 + m/8), 2^-6 * m/8 at e == 0.
    fn f8_reference(b: u8) -> f64 {
        let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let (e, m) = (((b >> 3) & 0x0F) as i32, (b & 7) as f64);
        if e == 0 {
            sign * 2f64.powi(-6) * m / 8.0
        } else {
            sign * 2f64.powi(e - 7) * (1.0 + m / 8.0)
        }
    }

    /// `p` to the nearest bf16 (8 significant bits, ties to even), by arithmetic rather than bits.
    fn bf16_nearest(p: f64) -> f64 {
        if p == 0.0 {
            return p;
        }
        let ulp = 2f64.powi(p.abs().log2().floor().max(-126.0) as i32 - 7);
        (p / ulp).round_ties_even() * ulp
    }

    #[test]
    fn round_bf16_is_torchs_cast() {
        let mut state = 0x0123_4567_89ab_cdefu64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let f = f32::from_bits((state >> 32) as u32 & 0xBF7F_FFFF); // finite, both signs
            assert_eq!(round_bf16(f) as f64, bf16_nearest(f as f64), "{f:e}");
        }
        // the ties go to even: 1 + 2^-8 down to 1, 1 + 3 * 2^-8 up to 1 + 2^-6
        assert_eq!(round_bf16(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(round_bf16(1.0 + 3.0 / 256.0), 1.0 + 1.0 / 64.0);
    }

    #[test]
    fn f8_value_matches_the_e4m3_definition() {
        for b in 0..=255u8 {
            let got = f8_value(b);
            if b & 0x7F == 0x7F {
                assert!(got.is_nan(), "{b:#04x} is NaN");
            } else {
                assert_eq!(got as f64, f8_reference(b), "{b:#04x}");
                assert_eq!(got.is_sign_negative(), b & 0x80 != 0, "{b:#04x} sign");
            }
        }
        assert_eq!(f8_value(0x7E), 448.0);
        assert_eq!(f8_value(0x01), 1.0 / 512.0);
    }

    /// Random finite e4m3 bytes (a quantizer never writes the NaN codes) and a grid of scales.
    fn f8_random(rows: usize, cols: usize, sr: usize, sc: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let w: Vec<u8> = (0..rows * cols)
            .map(|_| {
                let v = (next() >> 32) as u8;
                if v & 0x7F == 0x7F {
                    v ^ 1
                } else {
                    v
                }
            })
            .collect();
        let s: Vec<f32> = (0..sr * sc)
            .map(|_| ((next() >> 40) as f32 / 16_777_216.0 + 0.01) / 448.0)
            .collect();
        (w, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn f8_call(
        w: &[u8],
        s: &[f32],
        rows: usize,
        cols: usize,
        sr: usize,
        sc: usize,
        x: &[f32],
        b: usize,
        threads: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        let code = unsafe {
            gemv_fp8_core(
                w.as_ptr(),
                s.as_ptr(),
                rows,
                cols,
                sr,
                sc,
                x.as_ptr(),
                b,
                y.as_mut_ptr(),
                threads,
            )
        };
        assert_eq!(code, OK);
        y
    }

    fn f8_x(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn fp8_matches_the_f64_reference() {
        // (rows, cols, sr, sc): 128-blocks, blocks narrower than a SIMD run, ragged tails, a per-tensor
        // scale, and a block wider than the column tile
        for (rows, cols, sr, sc) in [
            (1usize, 16usize, 1usize, 1usize),
            (5, 96, 5, 6),
            (16, 544, 2, 17),
            (33, 1024, 3, 8),
            (64, 1280, 1, 1),
            (7, 13, 7, 13),
            (256, 256, 2, 2),
        ] {
            let (w, s) = f8_random(
                rows,
                cols,
                sr,
                sc,
                0x0f8f_8f8f_1234_5678 ^ (rows * cols) as u64,
            );
            let x = f8_x(cols, 0xfeed_beef_0000_0001 ^ cols as u64);
            let y = f8_call(&w, &s, rows, cols, sr, sc, &x, 1, 1);
            let (bm, bn) = (rows / sr, cols / sc);
            for (r, got) in y.iter().enumerate() {
                let mut want = 0.0f64;
                let mut scale = 1e-30f64;
                for (c, xv) in x.iter().enumerate() {
                    let wv = bf16_nearest(
                        (f8_reference(w[r * cols + c]) as f32 * s[(r / bm) * sc + c / bn]) as f64,
                    );
                    want += wv * *xv as f64;
                    scale += (wv * *xv as f64).abs();
                }
                assert!(
                    ((*got as f64 - want).abs() / scale) < 1e-6,
                    "{rows}x{cols} grid {sr}x{sc} row {r}: got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn fp8_tiling_threads_and_batching_do_not_move_a_single_bit() {
        let (rows, cols, sr, sc) = (67usize, 1056usize, 67usize, 33usize);
        let (w, s) = f8_random(rows, cols, sr, sc, 0xabcd_f8f8_5678_9999);
        let x = f8_x(3 * cols, 0x0bad_c0ff_ee00_0dd1);
        let base = f8_call(&w, &s, rows, cols, sr, sc, &x, 3, 1);
        for t in [2usize, 3, 5, 8, 0] {
            let got = f8_call(&w, &s, rows, cols, sr, sc, &x, 3, t);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads = {t}"
            );
        }
        for i in 0..3 {
            let solo = f8_call(
                &w,
                &s,
                rows,
                cols,
                sr,
                sc,
                &x[i * cols..(i + 1) * cols],
                1,
                0,
            );
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    /// Every byte through each widening on this machine, over runs that cut the SIMD width, to the bit.
    #[test]
    fn fp8_widenings_agree_across_isas() {
        // 37-column blocks, so every run cuts the SIMD width
        let (rows, cols, sr, sc) = (4usize, 481usize, 2usize, 13usize);
        let mut w: Vec<u8> = (0..rows * cols).map(|i| (i % 256) as u8).collect();
        w.rotate_left(3);
        let s: Vec<f32> = (0..sr * sc).map(|i| 0.5 + i as f32 / 64.0).collect();
        let p = F8 {
            w: w.as_ptr(),
            scales: s.as_ptr(),
            sc,
            bm: rows / sr,
            bn: cols / sc,
        };
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        for (start, n) in [(0usize, 481usize), (5, 470), (17, 3)] {
            unsafe {
                widen_f8_scalar::<4>(a.as_mut_ptr(), COL_TILE, p, start, cols, n);
                #[cfg(target_arch = "aarch64")]
                widen_f8_neon::<4>(b.as_mut_ptr(), COL_TILE, p, start, cols, n);
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") {
                    widen_f8_avx2::<4>(b.as_mut_ptr(), COL_TILE, p, start, cols, n);
                } else {
                    b.copy_from_slice(&a);
                }
            }
            for r in 0..4 {
                for j in 0..n {
                    let (u, v) = (a[r * COL_TILE + j], b[r * COL_TILE + j]);
                    assert!(
                        u.to_bits() == v.to_bits() || (u.is_nan() && v.is_nan()),
                        "start {start} row {r} col {j}: {u} vs {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn fp8_group_matches_the_single_calls() {
        let shapes = [
            (12usize, 64usize, 3usize, 4usize),
            (7, 128, 1, 1),
            (20, 96, 20, 3),
        ];
        let mats: Vec<(Vec<u8>, Vec<f32>)> = shapes
            .iter()
            .enumerate()
            .map(|(i, (r, c, sr, sc))| {
                f8_random(*r, *c, *sr, *sc, 0x9e37_79b9_f8f8_7c15 ^ i as u64)
            })
            .collect();
        let xs: Vec<Vec<f32>> = shapes
            .iter()
            .enumerate()
            .map(|(i, (_, c, _, _))| f8_x(2 * c, 0x2545_f491_4f6c_dd1d ^ (i as u64) << 8))
            .collect();
        let want: Vec<Vec<f32>> = shapes
            .iter()
            .enumerate()
            .map(|(i, (r, c, sr, sc))| {
                f8_call(&mats[i].0, &mats[i].1, *r, *c, *sr, *sc, &xs[i], 2, 0)
            })
            .collect();
        let mut got: Vec<Vec<f32>> = shapes.iter().map(|s| vec![0.0f32; 2 * s.0]).collect();
        let wp: Vec<*const u8> = mats.iter().map(|m| m.0.as_ptr()).collect();
        let sp: Vec<*const f32> = mats.iter().map(|m| m.1.as_ptr()).collect();
        let rp: Vec<usize> = shapes.iter().map(|s| s.0).collect();
        let cp: Vec<usize> = shapes.iter().map(|s| s.1).collect();
        let srp: Vec<usize> = shapes.iter().map(|s| s.2).collect();
        let scp: Vec<usize> = shapes.iter().map(|s| s.3).collect();
        let xp: Vec<*const f32> = xs.iter().map(|v| v.as_ptr()).collect();
        let bs: Vec<usize> = shapes.iter().map(|_| 2usize).collect();
        let mut yp: Vec<*mut f32> = got.iter_mut().map(|v| v.as_mut_ptr()).collect();
        let code = unsafe {
            gemv_fp8_group_core(
                shapes.len(),
                wp.as_ptr(),
                sp.as_ptr(),
                rp.as_ptr(),
                cp.as_ptr(),
                srp.as_ptr(),
                scp.as_ptr(),
                xp.as_ptr(),
                bs.as_ptr(),
                yp.as_mut_ptr(),
                0,
            )
        };
        assert_eq!(code, OK);
        for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                w.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                g.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "task {i}"
            );
        }
    }

    #[test]
    fn fp8_rejects_bad_shapes() {
        let w = [0u8; 64];
        let s = [1.0f32; 4];
        let x = [0.0f32; 32];
        let mut y = [0.0f32; 4];
        let call = |w: *const u8, s: *const f32, rows, cols, sr, sc, y: *mut f32| unsafe {
            gemv_fp8_core(w, s, rows, cols, sr, sc, x.as_ptr(), 1, y, 0)
        };
        let yp = y.as_mut_ptr();
        assert_eq!(
            call(std::ptr::null(), s.as_ptr(), 2, 32, 1, 1, yp),
            ERR_NULL
        );
        assert_eq!(
            call(w.as_ptr(), std::ptr::null(), 2, 32, 1, 1, yp),
            ERR_NULL
        );
        // the grid must divide the matrix evenly, and no dimension may be zero
        assert_eq!(call(w.as_ptr(), s.as_ptr(), 2, 32, 1, 3, yp), ERR_DOMAIN);
        assert_eq!(call(w.as_ptr(), s.as_ptr(), 2, 32, 4, 1, yp), ERR_DOMAIN);
        assert_eq!(call(w.as_ptr(), s.as_ptr(), 2, 32, 0, 1, yp), ERR_DOMAIN);
        assert_eq!(call(w.as_ptr(), s.as_ptr(), 0, 32, 1, 1, yp), ERR_DOMAIN);
    }
}
