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
// GGUF k-quants (Q4_K here; the others follow the same shape below): a 256-weight superblock read as
// stored and decoded into `wide` a tile, so the matvec never expands a quantized weight to a bf16 copy.
// A superblock's few float factors (the sub-block scales and mins, from its f16 delta) are computed once
// in scalar and shared by both paths; only the 256 per-weight decodes are vectorized. Every weight is one
// `scale * q - min` (a mul then a sub, each one rounding), so the scalar and NEON widenings write the same
// bits and the accumulation after them is the shared `accum_*`, exactly as the bf16 and MXFP4 paths.

/// weights a k-quant superblock holds.
pub(crate) const KQ_SB: usize = 256;

/// IEEE 754 half to f32, exact (a k-quant's delta, and its packed 6-bit scales/mins derive from it): the
/// same value numpy and llama.cpp read, subnormals and inf/nan included.
#[inline(always)]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalize the mantissa's leading 1 into the implicit bit. `e` shifts to reach bit 10,
            // so the leading bit is at 10 - e and the value is 2^(10-e-24); the f32 exponent field is
            // (10 - e - 24) + 127 = 113 - e.
            let mut e: i32 = 0;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | (((113 - e) as u32) << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp as u32 + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline(always)]
unsafe fn f16_at(p: *const u8) -> f32 {
    f16_to_f32(u16::from_le_bytes([*p, *p.add(1)]))
}

/// Q4_K: 144 bytes a superblock - d (f16), dmin (f16), 12 packed scale/min bytes, then 128 nibble bytes.
pub(crate) const Q4K_BYTES: usize = 144;

#[derive(Clone, Copy)]
struct Q4k(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q4k {}
unsafe impl Sync for Q4k {}

/// llama.cpp's `get_scale_min_k4`: the 6-bit scale and min of sub-block `j` from the 12 packed bytes at `s`.
#[inline(always)]
unsafe fn q4k_scale_min(s: *const u8, j: usize) -> (f32, f32) {
    if j < 4 {
        ((*s.add(j) & 63) as f32, (*s.add(j + 4) & 63) as f32)
    } else {
        let sc = (*s.add(j + 4) & 0x0F) | ((*s.add(j - 4) >> 6) << 4);
        let mn = (*s.add(j + 4) >> 4) | ((*s.add(j) >> 6) << 4);
        (sc as f32, mn as f32)
    }
}

/// the eight sub-blocks' `(d*scale, dmin*min)` for a Q4_K superblock, in f32 as the decode multiplies.
#[inline(always)]
unsafe fn q4k_factors(blk: *const u8) -> ([f32; 8], [f32; 8]) {
    let d = f16_at(blk);
    let dmin = f16_at(blk.add(2));
    let s = blk.add(4);
    let mut dsc = [0.0f32; 8];
    let mut dmm = [0.0f32; 8];
    for j in 0..8 {
        let (sc, mn) = q4k_scale_min(s, j);
        dsc[j] = d * sc;
        dmm[j] = dmin * mn;
    }
    (dsc, dmm)
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q4k_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q4k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q4K_BYTES);
            let (dsc, dmm) = q4k_factors(blk);
            let qs = blk.add(16);
            let o = out.add(sb * KQ_SB);
            for k in 0..4 {
                let (lo, hi) = (2 * k, 2 * k + 1);
                for lane in 0..32 {
                    let byte = *qs.add(32 * k + lane);
                    // scale*q - min as one fused multiply-add (one rounding); the NEON path fuses the same way
                    *o.add(lo * 32 + lane) = ((byte & 0x0F) as f32).mul_add(dsc[lo], -dmm[lo]);
                    *o.add(hi * 32 + lane) = ((byte >> 4) as f32).mul_add(dsc[hi], -dmm[hi]);
                }
            }
        }
    }
}

/// sixteen uint8 codes to `scale * code - min`, one fused multiply-add per four (`neg_min = -min`, one
/// rounding - the scalar path's `mul_add`), the four f32 vectors written to `o`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_scaled_u8x16(
    o: *mut f32,
    codes: std::arch::aarch64::uint8x16_t,
    scale: std::arch::aarch64::float32x4_t,
    neg_min: std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let lo = vmovl_u8(vget_low_u8(codes));
    let hi = vmovl_u8(vget_high_u8(codes));
    let q = [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))),
    ];
    for (j, v) in q.iter().enumerate() {
        vst1q_f32(o.add(j * 4), vfmaq_f32(neg_min, *v, scale));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q4k_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q4k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let mask = vdupq_n_u8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q4K_BYTES);
            let (dsc, dmm) = q4k_factors(blk);
            let qs = blk.add(16);
            let o = out.add(sb * KQ_SB);
            for k in 0..4 {
                let (lo, hi) = (2 * k, 2 * k + 1);
                let (scl, nml) = (vdupq_n_f32(dsc[lo]), vdupq_n_f32(-dmm[lo]));
                let (sch, nmh) = (vdupq_n_f32(dsc[hi]), vdupq_n_f32(-dmm[hi]));
                let v0 = vld1q_u8(qs.add(32 * k));
                let v1 = vld1q_u8(qs.add(32 * k + 16));
                let base = o.add(64 * k);
                store_scaled_u8x16(base, vandq_u8(v0, mask), scl, nml);
                store_scaled_u8x16(base.add(16), vandq_u8(v1, mask), scl, nml);
                store_scaled_u8x16(base.add(32), vshrq_n_u8::<4>(v0), sch, nmh);
                store_scaled_u8x16(base.add(48), vshrq_n_u8::<4>(v1), sch, nmh);
            }
        }
    }
}

/// A k-quant task: the tile is always decoded into `wide` (never accumulated straight), so one macro serves
/// every kind and ISA - only the widen differs.
macro_rules! def_task_kq {
    ($group:ident, $task:ident, $w:ty, $widen:ident, $accum:ident $(, $feat:literal)?) => {
        def_task!($group, $task, $w, $accum $(, $feat)?,
            prep |p, r, cols, cur| {},
            tile |scratch, wide, x, i0, bt, p, r, c0, cols, n, cur| {
                $widen::<R>(wide, COL_TILE, p, r * cols + c0, cols, n);
                false
            });
    };
}

def_task_kq!(
    group_q4k_scalar,
    task_q4k_scalar,
    Q4k,
    widen_q4k_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q4k_neon,
    task_q4k_neon,
    Q4k,
    widen_q4k_neon,
    accum_neon
);

/// sixteen uint8 to four f32 vectors, one lane a value (the widenings' shared conversion).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn cvt_u8x16(codes: std::arch::aarch64::uint8x16_t) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    let lo = vmovl_u8(vget_low_u8(codes));
    let hi = vmovl_u8(vget_high_u8(codes));
    [
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))),
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
        vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))),
    ]
}

/// Q6_K: 210 bytes a superblock - 128 low-nibble bytes, 64 high-2-bit bytes, 16 int8 scales, then the
/// f16 delta. A weight is `d * scale * (q - 32)`, q a 6-bit value (four low bits from `ql`, two high from
/// `qh`), with no min term.
pub(crate) const Q6K_BYTES: usize = 210;

#[derive(Clone, Copy)]
struct Q6k(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q6k {}
unsafe impl Sync for Q6k {}

/// the sixteen sub-blocks' `d * scale` for a Q6_K superblock, in f32 as the decode multiplies (the scales
/// are signed int8).
#[inline(always)]
unsafe fn q6k_factors(blk: *const u8) -> [f32; 16] {
    let d = f16_at(blk.add(208));
    let sc = blk.add(192) as *const i8;
    let mut dsc = [0.0f32; 16];
    for (i, v) in dsc.iter_mut().enumerate() {
        *v = d * (*sc.add(i)) as f32;
    }
    dsc
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q6k_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q6k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q6K_BYTES);
            let dsc = q6k_factors(blk);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                let (base, qlo, qho, sco) = (h * 128, h * 64, h * 32, h * 8);
                let ql = blk.add(qlo);
                let qh = blk.add(128 + qho);
                for lane in 0..32 {
                    let isc = sco + lane / 16;
                    let l0 = *ql.add(lane);
                    let l1 = *ql.add(lane + 32);
                    let hb = *qh.add(lane);
                    // code*scale - 32*scale as one fused multiply-add (one rounding); the NEON path fuses the same
                    let c0 = ((l0 & 0x0F) | ((hb & 3) << 4)) as f32;
                    let c1 = ((l1 & 0x0F) | (((hb >> 2) & 3) << 4)) as f32;
                    let c2 = ((l0 >> 4) | (((hb >> 4) & 3) << 4)) as f32;
                    let c3 = ((l1 >> 4) | (((hb >> 6) & 3) << 4)) as f32;
                    *o.add(base + lane) = c0.mul_add(dsc[isc], -32.0 * dsc[isc]);
                    *o.add(base + lane + 32) = c1.mul_add(dsc[isc + 2], -32.0 * dsc[isc + 2]);
                    *o.add(base + lane + 64) = c2.mul_add(dsc[isc + 4], -32.0 * dsc[isc + 4]);
                    *o.add(base + lane + 96) = c3.mul_add(dsc[isc + 6], -32.0 * dsc[isc + 6]);
                }
            }
        }
    }
}

/// sixteen 6-bit codes to `code*scale - 32*scale` as one fused multiply-add (`neg32 = -32*scale`), the
/// scalar path's `mul_add` bit for bit. The four f32 vectors written to `o`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_q6(
    o: *mut f32,
    codes: std::arch::aarch64::uint8x16_t,
    scale: std::arch::aarch64::float32x4_t,
    neg32: std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    for (j, v) in cvt_u8x16(codes).iter().enumerate() {
        vst1q_f32(o.add(j * 4), vfmaq_f32(neg32, *v, scale));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q6k_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q6k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let lo4 = vdupq_n_u8(0x0F);
    let two = vdupq_n_u8(0x03);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q6K_BYTES);
            let dsc = q6k_factors(blk);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                let (base, qlo, qho, sco) = (h * 128, h * 64, h * 32, h * 8);
                let ql = blk.add(qlo);
                let qh = blk.add(128 + qho);
                // the two 16-lane halves: lanes 0..15 read scale `sco`, 16..31 read `sco + 1`
                for hh in 0..2 {
                    let isc = sco + hh;
                    let off = hh * 16;
                    let l0 = vld1q_u8(ql.add(off));
                    let l1 = vld1q_u8(ql.add(off + 32));
                    let hb = vld1q_u8(qh.add(off));
                    let c0 = vorrq_u8(vandq_u8(l0, lo4), vshlq_n_u8::<4>(vandq_u8(hb, two)));
                    let c1 = vorrq_u8(
                        vandq_u8(l1, lo4),
                        vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(hb), two)),
                    );
                    let c2 = vorrq_u8(
                        vshrq_n_u8::<4>(l0),
                        vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(hb), two)),
                    );
                    let c3 = vorrq_u8(
                        vshrq_n_u8::<4>(l1),
                        vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(hb), two)),
                    );
                    let q = o.add(base + off);
                    // scale and its -32*scale addend per sub-block; the fma reads them the way the scalar does
                    let s = |i: usize| (vdupq_n_f32(dsc[i]), vdupq_n_f32(-32.0 * dsc[i]));
                    let (s0, s2, s4, s6) = (s(isc), s(isc + 2), s(isc + 4), s(isc + 6));
                    store_q6(q, c0, s0.0, s0.1);
                    store_q6(q.add(32), c1, s2.0, s2.1);
                    store_q6(q.add(64), c2, s4.0, s4.1);
                    store_q6(q.add(96), c3, s6.0, s6.1);
                }
            }
        }
    }
}

def_task_kq!(
    group_q6k_scalar,
    task_q6k_scalar,
    Q6k,
    widen_q6k_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q6k_neon,
    task_q6k_neon,
    Q6k,
    widen_q6k_neon,
    accum_neon
);

/// Q5_K: 176 bytes a superblock - Q4_K's delta, min and 12 scale/min bytes, then a 32-byte high-bit plane
/// (`qh`) and 128 nibble bytes. A weight is Q4_K's `scale*q - min` with a 5-bit `q`: the nibble plus one bit
/// from `qh`. The scale/min factors are Q4_K's (`q4k_factors`).
pub(crate) const Q5K_BYTES: usize = 176;

#[derive(Clone, Copy)]
struct Q5k(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q5k {}
unsafe impl Sync for Q5k {}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q5k_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q5k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q5K_BYTES);
            let (dsc, dmm) = q4k_factors(blk);
            let qh = blk.add(16);
            let qs = blk.add(48);
            let o = out.add(sb * KQ_SB);
            for lane in 0..32 {
                let hbit = *qh.add(lane);
                for k in 0..4 {
                    let (lo, hi) = (2 * k, 2 * k + 1);
                    let byte = *qs.add(32 * k + lane);
                    let qlo = (byte & 0x0F) | (((hbit >> (2 * k)) & 1) << 4);
                    let qhi = (byte >> 4) | (((hbit >> (2 * k + 1)) & 1) << 4);
                    *o.add(lo * 32 + lane) = (qlo as f32).mul_add(dsc[lo], -dmm[lo]);
                    *o.add(hi * 32 + lane) = (qhi as f32).mul_add(dsc[hi], -dmm[hi]);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q5k_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q5k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let lo4 = vdupq_n_u8(0x0F);
    let one = vdupq_n_u8(1);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q5K_BYTES);
            let (dsc, dmm) = q4k_factors(blk);
            let qh = blk.add(16);
            let qs = blk.add(48);
            let o = out.add(sb * KQ_SB);
            for k in 0..4 {
                let (lo, hi) = (2 * k, 2 * k + 1);
                let (scl, nml) = (vdupq_n_f32(dsc[lo]), vdupq_n_f32(-dmm[lo]));
                let (sch, nmh) = (vdupq_n_f32(dsc[hi]), vdupq_n_f32(-dmm[hi]));
                // the qh bit for this k, shifted to bit 4 (value 16): a runtime right shift, mask, then <<4
                let shl = vdupq_n_s8(-(2 * k as i32) as i8);
                let shh = vdupq_n_s8(-((2 * k + 1) as i32) as i8);
                for hh in 0..2 {
                    let off = hh * 16;
                    let v = vld1q_u8(qs.add(32 * k + off));
                    let h = vld1q_u8(qh.add(off));
                    let bl = vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h, shl), one));
                    let bh = vshlq_n_u8::<4>(vandq_u8(vshlq_u8(h, shh), one));
                    let qlo = vorrq_u8(vandq_u8(v, lo4), bl);
                    let qhi = vorrq_u8(vshrq_n_u8::<4>(v), bh);
                    store_scaled_u8x16(o.add(lo * 32 + off), qlo, scl, nml);
                    store_scaled_u8x16(o.add(hi * 32 + off), qhi, sch, nmh);
                }
            }
        }
    }
}

def_task_kq!(
    group_q5k_scalar,
    task_q5k_scalar,
    Q5k,
    widen_q5k_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q5k_neon,
    task_q5k_neon,
    Q5k,
    widen_q5k_neon,
    accum_neon
);

/// Q2_K: 84 bytes a superblock - 16 scale/min bytes (a 4-bit scale and 4-bit min a 16-weight sub-block), 64
/// 2-bit weight bytes, then the delta and min (f16). A weight is `d*scale*q - dmin*min` with a 2-bit `q`.
pub(crate) const Q2K_BYTES: usize = 84;

#[derive(Clone, Copy)]
struct Q2k(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q2k {}
unsafe impl Sync for Q2k {}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q2k_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q2k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q2K_BYTES);
            let d = f16_at(blk.add(80));
            let dmin = f16_at(blk.add(82));
            let scales = blk;
            let qs = blk.add(16);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                for lane in 0..32 {
                    let sub = lane >> 4;
                    let byte = *qs.add(h * 32 + lane);
                    for j in 0..4 {
                        let sc = *scales.add(h * 8 + 2 * j + sub);
                        let dsc = d * (sc & 0x0F) as f32;
                        let dmm = dmin * (sc >> 4) as f32;
                        let q = ((byte >> (2 * j)) & 3) as f32;
                        *o.add(h * 128 + j * 32 + lane) = q.mul_add(dsc, -dmm);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q2k_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q2k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let three = vdupq_n_u8(3);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q2K_BYTES);
            let d = f16_at(blk.add(80));
            let dmin = f16_at(blk.add(82));
            let scales = blk;
            let qs = blk.add(16);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                for j in 0..4 {
                    let shift = vdupq_n_s8(-(2 * j as i32) as i8);
                    for hh in 0..2 {
                        let sc = *scales.add(h * 8 + 2 * j + hh);
                        let dsc = vdupq_n_f32(d * (sc & 0x0F) as f32);
                        let ndm = vdupq_n_f32(-(dmin * (sc >> 4) as f32));
                        let v = vld1q_u8(qs.add(h * 32 + hh * 16));
                        let q = vandq_u8(vshlq_u8(v, shift), three);
                        store_scaled_u8x16(o.add(h * 128 + j * 32 + hh * 16), q, dsc, ndm);
                    }
                }
            }
        }
    }
}

def_task_kq!(
    group_q2k_scalar,
    task_q2k_scalar,
    Q2k,
    widen_q2k_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q2k_neon,
    task_q2k_neon,
    Q2k,
    widen_q2k_neon,
    accum_neon
);

/// Q3_K: 110 bytes a superblock - a 32-byte high-bit mask, 64 low-2-bit weight bytes, 12 packed 6-bit scale
/// bytes, then the delta. A weight is `d*(scale-32)*q`, no min; `q` is a signed 2+1-bit value (two bits from
/// `qs`, the third from `hmask`, biased by -4 when the mask bit is clear).
pub(crate) const Q3K_BYTES: usize = 110;

#[derive(Clone, Copy)]
struct Q3k(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q3k {}
unsafe impl Sync for Q3k {}

/// the sixteen sub-blocks' `d*(scale-32)` for a Q3_K superblock, in f32. The 6-bit scales are unpacked from
/// the 12 bytes the way llama.cpp does (four little-endian words recombined).
#[inline(always)]
unsafe fn q3k_scales(blk: *const u8) -> [f32; 16] {
    let d = f16_at(blk.add(108));
    let sc = blk.add(96);
    let word =
        |i: usize| u32::from_le_bytes([*sc.add(i), *sc.add(i + 1), *sc.add(i + 2), *sc.add(i + 3)]);
    let (a0, a1, a2) = (word(0), word(4), word(8));
    let (k1, k2) = (0x0303_0303u32, 0x0F0F_0F0Fu32);
    let aux = [
        (a0 & k2) | (((a2) & k1) << 4),
        (a1 & k2) | (((a2 >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
    ];
    let mut dsc = [0.0f32; 16];
    for (w, a) in aux.iter().enumerate() {
        for (t, b) in a.to_le_bytes().iter().enumerate() {
            dsc[w * 4 + t] = d * (*b as i32 - 32) as f32;
        }
    }
    dsc
}

/// sixteen signed codes to `code * scale`, the four f32 vectors written to `o`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_signed(
    o: *mut f32,
    codes: std::arch::aarch64::int8x16_t,
    scale: std::arch::aarch64::float32x4_t,
) {
    use std::arch::aarch64::*;
    let lo = vmovl_s8(vget_low_s8(codes));
    let hi = vmovl_s8(vget_high_s8(codes));
    let q = [
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
        vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
    ];
    for (j, v) in q.iter().enumerate() {
        vst1q_f32(o.add(j * 4), vmulq_f32(*v, scale));
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q3k_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q3k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q3K_BYTES);
            let dsc = q3k_scales(blk);
            // -4*scale per sub-block, the fma bias precomputed so the inner loop is one fma per weight; the same
            // value the NEON path broadcasts, so the two decode bit-for-bit alike.
            let neg4: [f32; 16] = std::array::from_fn(|i| -4.0 * dsc[i]);
            let hmask = blk;
            let qs = blk.add(32);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                for lane in 0..32 {
                    let sub = lane >> 4;
                    let hm = *hmask.add(lane);
                    let byte = *qs.add(h * 32 + lane);
                    for j in 0..4 {
                        let q2 = ((byte >> (2 * j)) & 3) as i32;
                        let bit = ((hm >> (h * 4 + j)) & 1) as i32;
                        let i = h * 8 + 2 * j + sub;
                        // code = q2 + 4*bit in [0,7]; the weight is scale*(code-4), the -4 folded into the fma
                        *o.add(h * 128 + j * 32 + lane) =
                            ((q2 + 4 * bit) as f32).mul_add(dsc[i], neg4[i]);
                    }
                }
            }
        }
    }
}

// hoisted qs/hmask loads (the same two bytes feed all four j) + the unsigned-code fma of `store_scaled_u8x16`:
// 1.21x over the scalar decode. The first NEON attempt (signed code + plain multiply, no hoist) was a 0.94x
// loss - see the kquant_throughput microbench.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q3k_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q3k,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let three = vdupq_n_u8(3);
    let one = vdupq_n_u8(1);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * Q3K_BYTES);
            let dsc = q3k_scales(blk);
            let hmask = blk;
            let qs = blk.add(32);
            let o = out.add(sb * KQ_SB);
            for h in 0..2 {
                for hh in 0..2 {
                    // the qs and hmask bytes for these 16 lanes are the same across all four j: load once.
                    let v = vld1q_u8(qs.add(h * 32 + hh * 16));
                    let hm = vld1q_u8(hmask.add(hh * 16));
                    for j in 0..4 {
                        let sc = dsc[h * 8 + 2 * j + hh];
                        let scale = vdupq_n_f32(sc);
                        let neg4 = vdupq_n_f32(-4.0 * sc);
                        let q2 = vandq_u8(vshlq_u8(v, vdupq_n_s8(-(2 * j as i32) as i8)), three);
                        let bit =
                            vandq_u8(vshlq_u8(hm, vdupq_n_s8(-((h * 4 + j) as i32) as i8)), one);
                        // code = q2 + 4*bit in [0, 7]; the weight is scale*(code - 4), the -4 an exact
                        // power-of-two so folding it into the fma stays bit-identical to `(code-4)*scale`.
                        let code = vaddq_u8(q2, vshlq_n_u8::<2>(bit));
                        store_scaled_u8x16(o.add(h * 128 + j * 32 + hh * 16), code, scale, neg4);
                    }
                }
            }
        }
    }
}

def_task_kq!(
    group_q3k_scalar,
    task_q3k_scalar,
    Q3k,
    widen_q3k_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q3k_neon,
    task_q3k_neon,
    Q3k,
    widen_q3k_neon,
    accum_neon
);

/// the fixed IQ4 non-linear codebook (ggml's kvalues_iq4nl): a 4-bit index maps to one of 16 signed levels.
const IQ4_KV: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// IQ4_NL: 18 bytes a block of 32 - an f16 delta then 16 nibble bytes; weight j (0..15) is the low nibble of
/// byte j, weight j+16 the high nibble, and a weight is `d * KV[code]`.
pub(crate) const IQ4NL_BYTES: usize = 18;
pub(crate) const IQ4NL_BLK: usize = 32;

#[derive(Clone, Copy)]
struct Iq4nl(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Iq4nl {}
unsafe impl Sync for Iq4nl {}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_iq4nl_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Iq4nl,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / IQ4NL_BLK) {
            let b = p.0.add((s / IQ4NL_BLK + blk) * IQ4NL_BYTES);
            let d = f16_at(b);
            let nib = b.add(2);
            let o = out.add(blk * IQ4NL_BLK);
            for j in 0..16 {
                let byte = *nib.add(j);
                *o.add(j) = d * IQ4_KV[(byte & 0x0F) as usize] as f32;
                *o.add(16 + j) = d * IQ4_KV[(byte >> 4) as usize] as f32;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_iq4nl_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Iq4nl,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let kv = vld1q_s8(IQ4_KV.as_ptr());
    let lo4 = vdupq_n_u8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / IQ4NL_BLK) {
            let b = p.0.add((s / IQ4NL_BLK + blk) * IQ4NL_BYTES);
            let dd = vdupq_n_f32(f16_at(b));
            let v = vld1q_u8(b.add(2));
            let o = out.add(blk * IQ4NL_BLK);
            store_signed(o, vqtbl1q_s8(kv, vandq_u8(v, lo4)), dd);
            store_signed(o.add(16), vqtbl1q_s8(kv, vshrq_n_u8::<4>(v)), dd);
        }
    }
}

def_task_kq!(
    group_iq4nl_scalar,
    task_iq4nl_scalar,
    Iq4nl,
    widen_iq4nl_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_iq4nl_neon,
    task_iq4nl_neon,
    Iq4nl,
    widen_iq4nl_neon,
    accum_neon
);

/// IQ4_XS: 136 bytes a superblock of 256 - an f16 delta, a u16 high-scale word, four low-scale bytes, then
/// 128 nibble bytes. Eight 32-blocks: block `ib` has a 6-bit scale `ls`, and a weight is `d*(ls-32)*KV[code]`.
pub(crate) const IQ4XS_BYTES: usize = 136;

#[derive(Clone, Copy)]
struct Iq4xs(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Iq4xs {}
unsafe impl Sync for Iq4xs {}

/// block `ib`'s 6-bit scale from the packed low nibbles and the two-bit high word.
#[inline(always)]
unsafe fn iq4xs_ls(sl: *const u8, sh: u32, ib: usize) -> i32 {
    (((*sl.add(ib >> 1) >> (4 * (ib & 1))) & 0x0F) as i32) | (((sh >> (2 * ib)) & 3) << 4) as i32
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_iq4xs_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Iq4xs,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * IQ4XS_BYTES);
            let d = f16_at(blk);
            let sh = u16::from_le_bytes([*blk.add(2), *blk.add(3)]) as u32;
            let sl = blk.add(4);
            let qs = blk.add(8);
            let o = out.add(sb * KQ_SB);
            for ib in 0..8 {
                let dl = d * (iq4xs_ls(sl, sh, ib) - 32) as f32;
                for col in 0..16 {
                    let byte = *qs.add(16 * ib + col);
                    *o.add(32 * ib + col) = dl * IQ4_KV[(byte & 0x0F) as usize] as f32;
                    *o.add(32 * ib + 16 + col) = dl * IQ4_KV[(byte >> 4) as usize] as f32;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_iq4xs_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Iq4xs,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let kv = vld1q_s8(IQ4_KV.as_ptr());
    let lo4 = vdupq_n_u8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for sb in 0..(n / KQ_SB) {
            let blk = p.0.add((s / KQ_SB + sb) * IQ4XS_BYTES);
            let d = f16_at(blk);
            let sh = u16::from_le_bytes([*blk.add(2), *blk.add(3)]) as u32;
            let sl = blk.add(4);
            let qs = blk.add(8);
            let o = out.add(sb * KQ_SB);
            for ib in 0..8 {
                let dl = vdupq_n_f32(d * (iq4xs_ls(sl, sh, ib) - 32) as f32);
                let v = vld1q_u8(qs.add(16 * ib));
                store_signed(o.add(32 * ib), vqtbl1q_s8(kv, vandq_u8(v, lo4)), dl);
                store_signed(o.add(32 * ib + 16), vqtbl1q_s8(kv, vshrq_n_u8::<4>(v)), dl);
            }
        }
    }
}

def_task_kq!(
    group_iq4xs_scalar,
    task_iq4xs_scalar,
    Iq4xs,
    widen_iq4xs_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_iq4xs_neon,
    task_iq4xs_neon,
    Iq4xs,
    widen_iq4xs_neon,
    accum_neon
);

// -- the affine types (Q4_0, Q4_1, Q8_0): a scale (and offset) a 32-weight block, multiplied as stored. --

/// Q4_0: 18 bytes a block of 32 - an f16 delta then 16 nibble bytes; a weight is `d*(code-8)` (nibble j the
/// low half of byte j, j+16 the high). Q4_1: 20 bytes - a delta and min, `d*code + m`. Q8_0: 34 bytes - a
/// delta and 32 int8 weights, `d*code`.
pub(crate) const Q40_BYTES: usize = 18;
pub(crate) const Q41_BYTES: usize = 20;
pub(crate) const Q80_BYTES: usize = 34;
pub(crate) const AFFINE_BLK: usize = 32;

#[derive(Clone, Copy)]
struct Q40(*const u8);
#[derive(Clone, Copy)]
struct Q41(*const u8);
#[derive(Clone, Copy)]
struct Q80(*const u8);
// SAFETY: read-only input, checked at the boundary; tasks read it over disjoint row ranges
unsafe impl Send for Q40 {}
unsafe impl Sync for Q40 {}
unsafe impl Send for Q41 {}
unsafe impl Sync for Q41 {}
unsafe impl Send for Q80 {}
unsafe impl Sync for Q80 {}

/// Q4_0 / Q4_1 share the nibble layout: a weight is `code * d + add` (add = -8*d for Q4_0, the min for Q4_1),
/// one fused multiply-add. `mbytes` is the block size (18 or 20), `qoff` where the nibbles start (2 or 4).
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q4affine_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    base: *const u8,
    mbytes: usize,
    qoff: usize,
    add: unsafe fn(*const u8) -> f32,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / AFFINE_BLK) {
            let b = base.add((s / AFFINE_BLK + blk) * mbytes);
            let d = f16_at(b);
            let a = add(b);
            let qs = b.add(qoff);
            let o = out.add(blk * AFFINE_BLK);
            for j in 0..16 {
                let byte = *qs.add(j);
                // code*d + add as one fused multiply-add, matching the NEON path bit for bit
                *o.add(j) = ((byte & 0x0F) as f32).mul_add(d, a);
                *o.add(16 + j) = ((byte >> 4) as f32).mul_add(d, a);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q4affine_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    base: *const u8,
    mbytes: usize,
    qoff: usize,
    add: unsafe fn(*const u8) -> f32,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    let lo4 = vdupq_n_u8(0x0F);
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / AFFINE_BLK) {
            let b = base.add((s / AFFINE_BLK + blk) * mbytes);
            let d = vdupq_n_f32(f16_at(b));
            let a = vdupq_n_f32(add(b));
            let v = vld1q_u8(b.add(qoff));
            let o = out.add(blk * AFFINE_BLK);
            store_scaled_u8x16(o, vandq_u8(v, lo4), d, a);
            store_scaled_u8x16(o.add(16), vshrq_n_u8::<4>(v), d, a);
        }
    }
}

#[inline(always)]
unsafe fn q40_add(b: *const u8) -> f32 {
    -8.0 * f16_at(b)
}
#[inline(always)]
unsafe fn q41_add(b: *const u8) -> f32 {
    f16_at(b.add(2))
}

macro_rules! widen_q4affine {
    ($scalar:ident, $neon:ident, $w:ty, $mbytes:expr, $qoff:expr, $add:ident) => {
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $scalar<const R: usize>(
            dst: *mut f32,
            ds: usize,
            p: $w,
            start: usize,
            rs: usize,
            n: usize,
        ) {
            widen_q4affine_scalar::<R>(dst, ds, p.0, $mbytes, $qoff, $add, start, rs, n);
        }
        #[cfg(target_arch = "aarch64")]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $neon<const R: usize>(
            dst: *mut f32,
            ds: usize,
            p: $w,
            start: usize,
            rs: usize,
            n: usize,
        ) {
            widen_q4affine_neon::<R>(dst, ds, p.0, $mbytes, $qoff, $add, start, rs, n);
        }
    };
}
widen_q4affine!(widen_q40_scalar, widen_q40_neon, Q40, Q40_BYTES, 2, q40_add);
widen_q4affine!(widen_q41_scalar, widen_q41_neon, Q41, Q41_BYTES, 4, q41_add);

def_task_kq!(
    group_q40_scalar,
    task_q40_scalar,
    Q40,
    widen_q40_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q40_neon,
    task_q40_neon,
    Q40,
    widen_q40_neon,
    accum_neon
);
def_task_kq!(
    group_q41_scalar,
    task_q41_scalar,
    Q41,
    widen_q41_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q41_neon,
    task_q41_neon,
    Q41,
    widen_q41_neon,
    accum_neon
);

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q80_scalar<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q80,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / AFFINE_BLK) {
            let b = p.0.add((s / AFFINE_BLK + blk) * Q80_BYTES);
            let d = f16_at(b);
            let qs = b.add(2) as *const i8;
            let o = out.add(blk * AFFINE_BLK);
            for j in 0..32 {
                *o.add(j) = *qs.add(j) as f32 * d;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn widen_q80_neon<const R: usize>(
    dst: *mut f32,
    dst_stride: usize,
    p: Q80,
    start: usize,
    row_stride: usize,
    n: usize,
) {
    use std::arch::aarch64::*;
    for r in 0..R {
        let s = start + r * row_stride;
        let out = dst.add(r * dst_stride);
        for blk in 0..(n / AFFINE_BLK) {
            let b = p.0.add((s / AFFINE_BLK + blk) * Q80_BYTES);
            let d = vdupq_n_f32(f16_at(b));
            let qs = b.add(2);
            let o = out.add(blk * AFFINE_BLK);
            store_signed(o, vld1q_s8(qs as *const i8), d);
            store_signed(o.add(16), vld1q_s8(qs.add(16) as *const i8), d);
        }
    }
}

def_task_kq!(
    group_q80_scalar,
    task_q80_scalar,
    Q80,
    widen_q80_scalar,
    accum_scalar
);
#[cfg(target_arch = "aarch64")]
def_task_kq!(
    group_q80_neon,
    task_q80_neon,
    Q80,
    widen_q80_neon,
    accum_neon
);

// -- the IQ lattice family (grid-codebook quants): a packed index into a shared grid table, signed and scaled.
// The decode is a scalar gather from the grid (256-512 entries, no NEON gather), so one widen feeds both the
// scalar and NEON accumulation - the NEON win is the shared `accum_neon` over the decoded tile. The grid (and,
// for the ksigns-based types, the 128-entry sign table) come from the gguf package and ride in the weight
// struct as buffers. All are 256-weight superblocks. --

/// one lattice weight: the raw superblocks, the type's int8 grid, and the shared sign table (null where the
/// type carries explicit signs or none).
macro_rules! latt_struct {
    ($n:ident) => {
        #[derive(Clone, Copy)]
        struct $n {
            raw: *const u8,
            grid: *const i8,
            ksigns: *const u8,
        }
        // SAFETY: read-only inputs, checked at the boundary; tasks read them over disjoint row ranges
        unsafe impl Send for $n {}
        unsafe impl Sync for $n {}
    };
}
latt_struct!(Iq3xxs);
latt_struct!(Iq2xxs);
latt_struct!(Iq2xs);
latt_struct!(Iq2s);
latt_struct!(Iq1s);
latt_struct!(Iq1m);
latt_struct!(Iq3s);

#[inline(always)]
unsafe fn u32le(p: *const u8) -> u32 {
    u32::from_le_bytes([*p, *p.add(1), *p.add(2), *p.add(3)])
}
#[inline(always)]
unsafe fn u16le(p: *const u8) -> u32 {
    (*p as u32) | ((*p.add(1) as u32) << 8)
}
/// sign from a ksigns/explicit byte's bit `b`: -1.0 when set, else +1.0.
#[inline(always)]
fn signf(byte: u8, b: usize) -> f32 {
    if byte & (1 << b) != 0 {
        -1.0
    } else {
        1.0
    }
}

macro_rules! latt_widen {
    ($name:ident, $w:ty, $bytes:expr, |$blk:ident, $grid:ident, $ks:ident, $ib:ident, $lane:ident, $o:ident| $body:block) => {
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $name<const R: usize>(
            dst: *mut f32,
            dst_stride: usize,
            p: $w,
            start: usize,
            row_stride: usize,
            n: usize,
        ) {
            let $grid = p.grid;
            let $ks = p.ksigns;
            for r in 0..R {
                let s = start + r * row_stride;
                let out = dst.add(r * dst_stride);
                for sbk in 0..(n / KQ_SB) {
                    let $blk = p.raw.add((s / KQ_SB + sbk) * $bytes);
                    let $o = out.add(sbk * KQ_SB);
                    for $ib in 0..8usize {
                        for $lane in 0..32usize {
                            $body
                        }
                    }
                }
            }
        }
    };
}

latt_widen!(widen_iq3xxs, Iq3xxs, 98, |blk, grid, ks, ib, lane, o| {
    let d = f16_at(blk);
    let qs = blk.add(2);
    let sca = blk.add(66);
    let aux = u32le(sca.add(4 * ib));
    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
    let l = lane >> 3;
    let m = lane & 7;
    let qsoff = 2 * l + if m >= 4 { 1 } else { 0 };
    let idx = *qs.add(8 * ib + qsoff) as usize;
    let gval = *grid.add(idx * 4 + (m & 3)) as f32;
    let signs = *ks.add(((aux >> (7 * l)) & 127) as usize);
    *o.add(32 * ib + lane) = db * gval * signf(signs, m);
});

latt_widen!(widen_iq2xxs, Iq2xxs, 66, |blk, grid, ks, ib, lane, o| {
    let d = f16_at(blk);
    let grp = blk.add(2 + 8 * ib);
    let w1 = u32le(grp.add(4));
    let db = d * (0.5 + (w1 >> 28) as f32) * 0.25;
    let l = lane >> 3;
    let j = lane & 7;
    let gi = *grp.add(l) as usize;
    let gval = *grid.add(gi * 8 + j) as f32;
    let signs = *ks.add(((w1 >> (7 * l)) & 127) as usize);
    *o.add(32 * ib + lane) = db * gval * signf(signs, j);
});

latt_widen!(widen_iq2xs, Iq2xs, 74, |blk, grid, ks, ib, lane, o| {
    let d = f16_at(blk);
    let qs = blk.add(2);
    let sc = blk.add(66);
    let l = lane >> 3;
    let j = lane & 7;
    let grp = 4 * ib + l;
    let q16 = u16le(qs.add(2 * grp));
    let sci = grp >> 1;
    let sc4 = (*sc.add(sci >> 1) >> (4 * (sci & 1))) & 0x0F;
    let db = d * (0.5 + sc4 as f32) * 0.25;
    let gval = *grid.add((q16 & 511) as usize * 8 + j) as f32;
    let signs = *ks.add((q16 >> 9) as usize);
    *o.add(32 * ib + lane) = db * gval * signf(signs, j);
});

latt_widen!(widen_iq2s, Iq2s, 82, |blk, grid, _ks, ib, lane, o| {
    let d = f16_at(blk);
    let qs = blk.add(2);
    let sgn = blk.add(34);
    let qh = blk.add(66);
    let sc = blk.add(74);
    let l = lane >> 3;
    let j = lane & 7;
    let grp = 4 * ib + l;
    let idx =
        *qs.add(grp) as usize | ((((*qh.add(grp >> 2) >> (2 * (grp & 3))) & 3) as usize) << 8);
    let b = grp >> 1;
    let sc4 = (*sc.add(b >> 1) >> (4 * (b & 1))) & 0x0F;
    let db = d * (0.5 + sc4 as f32) * 0.25;
    let gval = *grid.add(idx * 8 + j) as f32;
    *o.add(32 * ib + lane) = db * gval * signf(*sgn.add(grp), j);
});

latt_widen!(widen_iq1s, Iq1s, 50, |blk, grid, _ks, ib, lane, o| {
    let d = f16_at(blk);
    let qs = blk.add(2);
    let qhb = blk.add(34);
    let l = lane >> 3;
    let j = lane & 7;
    let qhv = u16le(qhb.add(2 * ib));
    let idx = *qs.add(ib * 4 + l) as usize | ((((qhv >> (3 * l)) & 7) as usize) << 8);
    let dl = d * (2 * ((qhv >> 12) & 7) + 1) as f32;
    let delta = if qhv & 0x8000 != 0 { -0.125 } else { 0.125 };
    *o.add(32 * ib + lane) = dl * (*grid.add(idx * 8 + j) as f32 + delta);
});

latt_widen!(widen_iq3s, Iq3s, 110, |blk, grid, _ks, ib, lane, o| {
    let d = f16_at(blk);
    let qs = blk.add(2);
    let qh = blk.add(66);
    let sgn = blk.add(74);
    let sc = blk.add(106);
    let ql = lane >> 2;
    let j = lane & 3;
    let qb = 8 * ib + ql;
    let idx = *qs.add(qb) as usize | ((((*qh.add(qb >> 3) >> (qb & 7)) & 1) as usize) << 8);
    let sc4 = (*sc.add(ib >> 1) >> (4 * (ib & 1))) & 0x0F;
    let db = d * (1 + 2 * sc4) as f32;
    let gval = *grid.add(idx * 4 + j) as f32;
    *o.add(32 * ib + lane) = db * gval * signf(*sgn.add(4 * ib + (lane >> 3)), lane & 7);
});

latt_widen!(widen_iq1m, Iq1m, 56, |blk, grid, _ks, ib, lane, o| {
    let qs = blk;
    let qh = blk.add(32);
    let scb = blk.add(48);
    let s = [
        u16le(scb),
        u16le(scb.add(2)),
        u16le(scb.add(4)),
        u16le(scb.add(6)),
    ];
    let dbits = (s[0] >> 12) | ((s[1] >> 12) << 4) | ((s[2] >> 12) << 8) | ((s[3] >> 12) << 12);
    let d = f16_to_f32(dbits as u16);
    let j8 = lane >> 3;
    let e = lane & 7;
    let jj = 4 * ib + j8;
    let nib = (*qh.add(jj >> 1) >> (4 * (jj & 1))) & 0x0F;
    let idx = *qs.add(jj) as usize | (((nib & 7) as usize) << 8);
    let delta = if nib & 8 != 0 { -0.125 } else { 0.125 };
    let si = 2 * ib + (lane >> 4);
    let sub = (s[si >> 2] >> (3 * (si & 3))) & 7;
    let dl = d * (2 * sub + 1) as f32;
    *o.add(32 * ib + lane) = dl * (*grid.add(idx * 8 + e) as f32 + delta);
});

macro_rules! def_task_latt {
    ($w:ty, $ss:ident, $ts:ident, $sn:ident, $tn:ident, $widen:ident) => {
        def_task_kq!($ss, $ts, $w, $widen, accum_scalar);
        #[cfg(target_arch = "aarch64")]
        def_task_kq!($sn, $tn, $w, $widen, accum_neon);
    };
}
def_task_latt!(
    Iq3xxs,
    group_iq3xxs_s,
    task_iq3xxs_s,
    group_iq3xxs_n,
    task_iq3xxs_n,
    widen_iq3xxs
);
def_task_latt!(
    Iq2xxs,
    group_iq2xxs_s,
    task_iq2xxs_s,
    group_iq2xxs_n,
    task_iq2xxs_n,
    widen_iq2xxs
);
def_task_latt!(
    Iq2xs,
    group_iq2xs_s,
    task_iq2xs_s,
    group_iq2xs_n,
    task_iq2xs_n,
    widen_iq2xs
);
def_task_latt!(
    Iq2s,
    group_iq2s_s,
    task_iq2s_s,
    group_iq2s_n,
    task_iq2s_n,
    widen_iq2s
);
def_task_latt!(
    Iq1s,
    group_iq1s_s,
    task_iq1s_s,
    group_iq1s_n,
    task_iq1s_n,
    widen_iq1s
);
def_task_latt!(
    Iq3s,
    group_iq3s_s,
    task_iq3s_s,
    group_iq3s_n,
    task_iq3s_n,
    widen_iq3s
);
def_task_latt!(
    Iq1m,
    group_iq1m_s,
    task_iq1m_s,
    group_iq1m_n,
    task_iq1m_n,
    widen_iq1m
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

impl Weights for Q4k {
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
        // no AVX k-quant widen yet: x86 runs the scalar decode into the shared accumulation
        by_isa!(
            isa,
            task_q4k_scalar,
            task_q4k_scalar,
            task_q4k_scalar,
            task_q4k_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

impl Weights for Q6k {
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
            task_q6k_scalar,
            task_q6k_scalar,
            task_q6k_scalar,
            task_q6k_neon,
            (self, cols, x, b, y, rows_total, r0, r1)
        )
    }
}

macro_rules! impl_weights_kq {
    ($w:ty, $scalar:ident, $neon:ident) => {
        impl Weights for $w {
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
                // no AVX k-quant widen yet: x86 runs the scalar decode into the shared accumulation
                by_isa!(
                    isa,
                    $scalar,
                    $scalar,
                    $scalar,
                    $neon,
                    (self, cols, x, b, y, rows_total, r0, r1)
                )
            }
        }
    };
}

impl_weights_kq!(Q5k, task_q5k_scalar, task_q5k_neon);
impl_weights_kq!(Q2k, task_q2k_scalar, task_q2k_neon);
impl_weights_kq!(Q3k, task_q3k_scalar, task_q3k_neon);
impl_weights_kq!(Iq4nl, task_iq4nl_scalar, task_iq4nl_neon);
impl_weights_kq!(Iq4xs, task_iq4xs_scalar, task_iq4xs_neon);
impl_weights_kq!(Q40, task_q40_scalar, task_q40_neon);
impl_weights_kq!(Q41, task_q41_scalar, task_q41_neon);
impl_weights_kq!(Q80, task_q80_scalar, task_q80_neon);
impl_weights_kq!(Iq3xxs, task_iq3xxs_s, task_iq3xxs_n);
impl_weights_kq!(Iq2xxs, task_iq2xxs_s, task_iq2xxs_n);
impl_weights_kq!(Iq2xs, task_iq2xs_s, task_iq2xs_n);
impl_weights_kq!(Iq2s, task_iq2s_s, task_iq2s_n);
impl_weights_kq!(Iq1s, task_iq1s_s, task_iq1s_n);
impl_weights_kq!(Iq3s, task_iq3s_s, task_iq3s_n);
impl_weights_kq!(Iq1m, task_iq1m_s, task_iq1m_n);

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

/// a packed matvec's shape: whole `blk`-weight blocks a row (256 for the k-quants and IQ4_XS, 32 for
/// IQ4_NL), and the element counts fit `usize`.
#[inline]
fn blk_shape_ok(rows: usize, cols: usize, b: usize, blk: usize) -> bool {
    rows != 0 && cols != 0 && b != 0 && cols.is_multiple_of(blk) && shape_ok(rows, cols, b)
}

#[inline]
fn kq_shape_ok(rows: usize, cols: usize, b: usize) -> bool {
    blk_shape_ok(rows, cols, b, KQ_SB)
}

/// `y[i][r] = sum_c dequant(W)[r][c] * x[i][c]` in f32 for a Q4_K matrix `raw` ([rows, cols] as
/// `rows * cols / 256` superblocks). Bit-identical for every `threads` and every `b`, row `r` the same at
/// any `b` (the verify pass's contract).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_q4k_core(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    if raw.is_null() || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if !kq_shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }
    run_one(
        Task {
            w: Q4k(raw),
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

/// [`gemv_q4k_core`] for a Q6_K matrix `raw` ([rows, cols] as `rows * cols / 256` superblocks of 210 bytes).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_q6k_core(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    if raw.is_null() || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if !kq_shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }
    run_one(
        Task {
            w: Q6k(raw),
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

/// [`gemv_q4k_core`] for the other k-quants: `raw` the file's superblocks of that type, `cols` a multiple of 256.
macro_rules! kq_core {
    ($name:ident, $w:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn $name(
            raw: *const u8,
            rows: usize,
            cols: usize,
            x: *const f32,
            b: usize,
            y: *mut f32,
            threads: usize,
        ) -> i32 {
            if raw.is_null() || x.is_null() || y.is_null() {
                return ERR_NULL;
            }
            if !kq_shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
                return ERR_DOMAIN;
            }
            run_one(
                Task {
                    w: $w(raw),
                    x,
                    y,
                    rows,
                    cols,
                    b,
                },
                threads,
            )
        }
    };
}

kq_core!(gemv_q5k_core, Q5k);
kq_core!(gemv_q2k_core, Q2k);
kq_core!(gemv_q3k_core, Q3k);
kq_core!(gemv_iq4xs_core, Iq4xs);

/// [`gemv_q4k_core`] for IQ4_NL, whose blocks are 32 weights (not 256): `cols` a multiple of 32.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv_iq4nl_core(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    if raw.is_null() || x.is_null() || y.is_null() {
        return ERR_NULL;
    }
    if !blk_shape_ok(rows, cols, b, IQ4NL_BLK) || !aligned(x) || !aligned(y) {
        return ERR_DOMAIN;
    }
    run_one(
        Task {
            w: Iq4nl(raw),
            x,
            y,
            rows,
            cols,
            b,
        },
        threads,
    )
}

/// [`gemv_q4k_core`] for the affine types (Q4_0, Q4_1, Q8_0): 32-weight blocks, `cols` a multiple of 32.
macro_rules! affine_core {
    ($name:ident, $w:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn $name(
            raw: *const u8,
            rows: usize,
            cols: usize,
            x: *const f32,
            b: usize,
            y: *mut f32,
            threads: usize,
        ) -> i32 {
            if raw.is_null() || x.is_null() || y.is_null() {
                return ERR_NULL;
            }
            if !blk_shape_ok(rows, cols, b, AFFINE_BLK) || !aligned(x) || !aligned(y) {
                return ERR_DOMAIN;
            }
            run_one(
                Task {
                    w: $w(raw),
                    x,
                    y,
                    rows,
                    cols,
                    b,
                },
                threads,
            )
        }
    };
}
affine_core!(gemv_q40_core, Q40);
affine_core!(gemv_q41_core, Q41);
affine_core!(gemv_q80_core, Q80);

/// [`gemv_q4k_core`] for an IQ lattice type: the raw superblocks plus the type's grid and (where used) the
/// shared sign table, both passed as buffers. All are 256-weight superblocks.
macro_rules! latt_core {
    ($name:ident, $w:ident) => {
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn $name(
            raw: *const u8,
            grid: *const i8,
            ksigns: *const u8,
            rows: usize,
            cols: usize,
            x: *const f32,
            b: usize,
            y: *mut f32,
            threads: usize,
        ) -> i32 {
            if raw.is_null() || grid.is_null() || x.is_null() || y.is_null() {
                return ERR_NULL;
            }
            if !kq_shape_ok(rows, cols, b) || !aligned(x) || !aligned(y) {
                return ERR_DOMAIN;
            }
            run_one(
                Task {
                    w: $w { raw, grid, ksigns },
                    x,
                    y,
                    rows,
                    cols,
                    b,
                },
                threads,
            )
        }
    };
}
latt_core!(gemv_iq3xxs_core, Iq3xxs);
latt_core!(gemv_iq2xxs_core, Iq2xxs);
latt_core!(gemv_iq2xs_core, Iq2xs);
latt_core!(gemv_iq2s_core, Iq2s);
latt_core!(gemv_iq1s_core, Iq1s);
latt_core!(gemv_iq3s_core, Iq3s);
latt_core!(gemv_iq1m_core, Iq1m);

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

    // ---- k-quants -------------------------------------------------------------------------------

    /// `rows * cols / 256` Q4_K superblocks of random bytes, the two f16 factors held to a small finite
    /// positive range (a real quantizer's deltas, never inf/nan) so the weights stay finite.
    fn q4k_random(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let nsb = rows * cols / KQ_SB;
        let mut state = seed;
        let mut byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        };
        let mut raw = vec![0u8; nsb * Q4K_BYTES];
        for sb in 0..nsb {
            let o = sb * Q4K_BYTES;
            // half in [2^-4, 2^-3): exponent field 11, a random mantissa - finite and modest
            let d = 0x2C00u16 | (((byte() as u16) << 2) & 0x03FF);
            let dmin = 0x2C00u16 | (((byte() as u16) << 2) & 0x03FF);
            raw[o..o + 2].copy_from_slice(&d.to_le_bytes());
            raw[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
            for b in raw[o + 4..o + Q4K_BYTES].iter_mut() {
                *b = byte();
            }
        }
        raw
    }

    /// weight `idx` (0..256) of a Q4_K superblock, straight from llama.cpp's definition in f64.
    fn q4k_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let sb = idx / 32;
        let lane = idx % 32;
        let k = sb / 2;
        let byte = blk[16 + 32 * k + lane];
        let q = if sb.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        let (sc, mn) = unsafe { q4k_scale_min(blk[4..16].as_ptr(), sb) };
        d * sc as f64 * q as f64 - dmin * mn as f64
    }

    fn q4k_call(
        raw: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        b: usize,
        threads: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        let code = unsafe {
            gemv_q4k_core(
                raw.as_ptr(),
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
    fn f16_to_f32_matches_ieee() {
        // (half bits, exact f32) across normals, both signs, and subnormals (a tiny k-quant delta/min: the
        // case real weights hit and uniform-random bytes miss)
        let p = |n: i32| 2.0f32.powi(n);
        for (h, want) in [
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3C00, 1.0),
            (0xC000, -2.0),
            (0x4900, 10.0),             // exp 18, mant 0x100: 8 * 1.25
            (0x7BFF, 65504.0),          // max normal
            (0x0400, p(-14)),           // min normal
            (0x0200, 512.0 * p(-24)),   // subnormal 2^-15
            (0x0001, p(-24)),           // smallest subnormal
            (0x83FF, -1023.0 * p(-24)), // largest negative subnormal
        ] {
            assert_eq!(f16_to_f32(h).to_bits(), want.to_bits(), "half {h:#06x}");
        }
    }

    #[test]
    fn q4k_matvec_matches_the_f64_reference() {
        let mut state = 0x0f0f_a5a5_1234_9999u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        for (rows, cols) in [
            (1usize, 256usize),
            (5, 512),
            (12, 512),
            (17, 768),
            (1024, 1024),
            (2048, 1024),
        ] {
            let raw = q4k_random(rows, cols, 0x9e37_79b9_7f4a_7c15 ^ cols as u64);
            let nsb_row = cols / KQ_SB;
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let y = q4k_call(&raw, rows, cols, &x, 1, 1);
            for r in 0..rows {
                let mut want = 0.0f64;
                let mut scale = 1e-30f64;
                for c in 0..cols {
                    let blk = &raw[(r * nsb_row + c / KQ_SB) * Q4K_BYTES..];
                    let w = q4k_weight_ref(&blk[..Q4K_BYTES], c % KQ_SB);
                    want += w * x[c] as f64;
                    scale += (w * x[c] as f64).abs();
                }
                assert!(
                    ((y[r] as f64 - want).abs() / scale) < 1e-6,
                    "{rows}x{cols} row {r}: got {} want {want}",
                    y[r]
                );
            }
        }
    }

    #[test]
    fn q4k_tiling_threads_and_batching_do_not_move_a_single_bit() {
        let (rows, cols) = (37usize, 768usize);
        let raw = q4k_random(rows, cols, 0xabcd_0f0f_5151_2468);
        let mut state = 0x2468_ace0_1357_9bdfu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..3 * cols).map(|_| next()).collect();
        let base = q4k_call(&raw, rows, cols, &x, 3, 1);
        for t in [2usize, 3, 5, 8, 0] {
            let got = q4k_call(&raw, rows, cols, &x, 3, t);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads = {t}"
            );
        }
        // row r of a 3-row pass is bit-for-bit its own one-row solo: the verify pass's contract
        for i in 0..3 {
            let solo = q4k_call(&raw, rows, cols, &x[i * cols..(i + 1) * cols], 1, 0);
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q4k_scalar_and_neon_widen_to_the_same_bits() {
        let (rows, cols) = (4usize, 512usize);
        let raw = q4k_random(rows, cols, 0x1111_2222_3333_4444);
        let p = Q4k(raw.as_ptr());
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        unsafe {
            widen_q4k_scalar::<4>(a.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
            widen_q4k_neon::<4>(b.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
        }
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(av.to_bits(), bv.to_bits(), "lane {i}");
        }
    }

    #[test]
    fn q4k_rejects_null_and_bad_shapes() {
        let raw = q4k_random(2, 256, 7);
        let x = [0.0f32; 256];
        let mut y = [0.0f32; 2];
        unsafe {
            assert_eq!(
                gemv_q4k_core(std::ptr::null(), 2, 256, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_NULL
            );
            // cols must be a whole number of 256-weight superblocks
            assert_eq!(
                gemv_q4k_core(raw.as_ptr(), 2, 128, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
            assert_eq!(
                gemv_q4k_core(raw.as_ptr(), 0, 256, x.as_ptr(), 1, y.as_mut_ptr(), 0),
                ERR_DOMAIN
            );
        }
    }

    /// `rows * cols / 256` Q6_K superblocks of random bytes with a finite positive f16 delta.
    fn q6k_random(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
        let nsb = rows * cols / KQ_SB;
        let mut state = seed;
        let mut byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        };
        let mut raw = vec![0u8; nsb * Q6K_BYTES];
        for sb in 0..nsb {
            let o = sb * Q6K_BYTES;
            for b in raw[o..o + 208].iter_mut() {
                *b = byte();
            }
            // the f16 delta in [2^-4, 2^-3): finite and modest
            let d = 0x2C00u16 | (((byte() as u16) << 2) & 0x03FF);
            raw[o + 208..o + 210].copy_from_slice(&d.to_le_bytes());
        }
        raw
    }

    /// weight `idx` (0..256) of a Q6_K superblock, straight from llama.cpp's definition in f64.
    fn q6k_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]])) as f64;
        let sc = |i: usize| blk[192 + i] as i8 as f64;
        let h = idx / 128;
        let within = idx % 128;
        let lane = within % 32;
        let quad = within / 32; // 0..4: which of the lane's four weights
        let (qlo, qho, sco) = (h * 64, h * 32, h * 8);
        let isc = sco + lane / 16;
        let l0 = blk[qlo + lane];
        let l1 = blk[qlo + lane + 32];
        let hb = blk[128 + qho + lane];
        let (src, shift, sci) = match quad {
            0 => (l0, 0u32, isc),
            1 => (l1, 2, isc + 2),
            2 => (l0 >> 4, 4, isc + 4),
            _ => (l1 >> 4, 6, isc + 6),
        };
        let q = ((src & 0x0F) | (((hb >> shift) & 3) << 4)) as i32 - 32;
        d * sc(sci) * q as f64
    }

    fn q6k_call(
        raw: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        b: usize,
        threads: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        let code = unsafe {
            gemv_q6k_core(
                raw.as_ptr(),
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
    fn q6k_matvec_matches_the_f64_reference() {
        let mut state = 0x5151_2222_c3c3_7777u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        for (rows, cols) in [(1usize, 256usize), (5, 512), (12, 512), (17, 768)] {
            let raw = q6k_random(rows, cols, 0xdead_c0de_1234_0001 ^ cols as u64);
            let nsb_row = cols / KQ_SB;
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let y = q6k_call(&raw, rows, cols, &x, 1, 1);
            for r in 0..rows {
                let mut want = 0.0f64;
                let mut scale = 1e-30f64;
                for c in 0..cols {
                    let blk = &raw[(r * nsb_row + c / KQ_SB) * Q6K_BYTES..];
                    let w = q6k_weight_ref(&blk[..Q6K_BYTES], c % KQ_SB);
                    want += w * x[c] as f64;
                    scale += (w * x[c] as f64).abs();
                }
                assert!(
                    ((y[r] as f64 - want).abs() / scale) < 1e-6,
                    "{rows}x{cols} row {r}: got {} want {want}",
                    y[r]
                );
            }
        }
    }

    #[test]
    fn q6k_tiling_threads_and_batching_do_not_move_a_single_bit() {
        let (rows, cols) = (37usize, 768usize);
        let raw = q6k_random(rows, cols, 0x0f0f_abcd_2468_5151);
        let mut state = 0x1357_2468_9bdf_ace0u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        let x: Vec<f32> = (0..3 * cols).map(|_| next()).collect();
        let base = q6k_call(&raw, rows, cols, &x, 3, 1);
        for t in [2usize, 3, 5, 8, 0] {
            let got = q6k_call(&raw, rows, cols, &x, 3, t);
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads = {t}"
            );
        }
        for i in 0..3 {
            let solo = q6k_call(&raw, rows, cols, &x[i * cols..(i + 1) * cols], 1, 0);
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q6k_scalar_and_neon_widen_to_the_same_bits() {
        let (rows, cols) = (4usize, 512usize);
        let raw = q6k_random(rows, cols, 0x4444_3333_2222_1111);
        let p = Q6k(raw.as_ptr());
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        unsafe {
            widen_q6k_scalar::<4>(a.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
            widen_q6k_neon::<4>(b.as_mut_ptr(), COL_TILE, p, 0, cols, cols.min(COL_TILE));
        }
        for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(av.to_bits(), bv.to_bits(), "lane {i}");
        }
    }

    // ---- Q5_K / Q2_K / Q3_K ---------------------------------------------------------------------

    /// `rows * cols / 256` superblocks of `nb` random bytes, the f16 fields at `f16_offs` held finite positive.
    fn kq_random(
        nb: usize,
        bw: usize,
        f16_offs: &[usize],
        rows: usize,
        cols: usize,
        seed: u64,
    ) -> Vec<u8> {
        let n = rows * cols / bw;
        let mut state = seed;
        let mut byte = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 32) as u8
        };
        let mut raw = vec![0u8; n * nb];
        for sb in 0..n {
            let o = sb * nb;
            for b in raw[o..o + nb].iter_mut() {
                *b = byte();
            }
            for &off in f16_offs {
                let d = 0x2C00u16 | (((byte() as u16) << 2) & 0x03FF);
                raw[o + off..o + off + 2].copy_from_slice(&d.to_le_bytes());
            }
        }
        raw
    }

    fn q5k_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let dm = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let (sb, lane, k) = (idx / 32, idx % 32, idx / 64);
        let hbit = blk[16 + lane];
        let byte = blk[48 + 32 * k + lane];
        let q = if sb % 2 == 0 {
            (byte & 0x0F) | (((hbit >> (2 * k)) & 1) << 4)
        } else {
            (byte >> 4) | (((hbit >> (2 * k + 1)) & 1) << 4)
        };
        let (sc, mn) = unsafe { q4k_scale_min(blk[4..].as_ptr(), sb) };
        d * sc as f64 * q as f64 - dm * mn as f64
    }

    fn q2k_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]])) as f64;
        let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]])) as f64;
        let (h, within) = (idx / 128, idx % 128);
        let (j, lane) = (within / 32, within % 32);
        let s = blk[h * 8 + 2 * j + (lane >> 4)];
        let q = ((blk[16 + h * 32 + lane] >> (2 * j)) & 3) as f64;
        d * (s & 0x0F) as f64 * q - dm * (s >> 4) as f64
    }

    fn q3k_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]])) as f64;
        let sc = &blk[96..108];
        let word = |i: usize| u32::from_le_bytes([sc[i], sc[i + 1], sc[i + 2], sc[i + 3]]);
        let (a0, a1, a2) = (word(0), word(4), word(8));
        let (k1, k2) = (0x0303_0303u32, 0x0F0F_0F0Fu32);
        let aux = [
            (a0 & k2) | ((a2 & k1) << 4),
            (a1 & k2) | (((a2 >> 2) & k1) << 4),
            ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
            ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
        ];
        let mut scl = [0u8; 16];
        for (w, a) in aux.iter().enumerate() {
            scl[w * 4..w * 4 + 4].copy_from_slice(&a.to_le_bytes());
        }
        let (h, within) = (idx / 128, idx % 128);
        let (j, lane) = (within / 32, within % 32);
        let q2 = ((blk[32 + h * 32 + lane] >> (2 * j)) & 3) as i32;
        let bit = ((blk[lane] >> (h * 4 + j)) & 1) as i32;
        d * (scl[h * 8 + 2 * j + (lane >> 4)] as i32 - 32) as f64 * (q2 - 4 + 4 * bit) as f64
    }

    type KqCore = unsafe fn(*const u8, usize, usize, *const f32, usize, *mut f32, usize) -> i32;

    fn kq_call(
        core: KqCore,
        raw: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        b: usize,
        threads: usize,
    ) -> Vec<f32> {
        let mut y = vec![f32::NAN; b * rows];
        assert_eq!(
            unsafe {
                core(
                    raw.as_ptr(),
                    rows,
                    cols,
                    x.as_ptr(),
                    b,
                    y.as_mut_ptr(),
                    threads,
                )
            },
            OK
        );
        y
    }

    /// the matvec against an f64 reference at b=1, and row 0 of a 3-row pass bit-for-bit its own one-row call.
    fn kq_certify(
        nb: usize,
        bw: usize,
        core: KqCore,
        rnd: fn(usize, usize, u64) -> Vec<u8>,
        refn: fn(&[u8], usize) -> f64,
        seed: u64,
    ) {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        };
        for (rows, cols) in [(1usize, 256usize), (5, 512), (12, 512), (17, 768)] {
            let raw = rnd(rows, cols, seed ^ cols as u64);
            let nblk_row = cols / bw;
            let x: Vec<f32> = (0..cols).map(|_| next()).collect();
            let y = kq_call(core, &raw, rows, cols, &x, 1, 1);
            for r in 0..rows {
                let (mut want, mut scale) = (0.0f64, 1e-30f64);
                for c in 0..cols {
                    let blk = &raw[(r * nblk_row + c / bw) * nb..];
                    let w = refn(&blk[..nb], c % bw);
                    want += w * x[c] as f64;
                    scale += (w * x[c] as f64).abs();
                }
                assert!(
                    (y[r] as f64 - want).abs() / scale < 1e-6,
                    "b1 {rows}x{cols} row {r}"
                );
            }
        }
        // row invariance: row 0 of a 3-row pass equals its own solo, across thread counts
        let (rows, cols) = (37usize, 768usize);
        let raw = rnd(rows, cols, seed);
        let x: Vec<f32> = (0..3 * cols).map(|_| next()).collect();
        let base = kq_call(core, &raw, rows, cols, &x, 3, 1);
        for t in [2usize, 5, 0] {
            assert_eq!(
                kq_call(core, &raw, rows, cols, &x, 3, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads {t}"
            );
        }
        for i in 0..3 {
            let solo = kq_call(core, &raw, rows, cols, &x[i * cols..(i + 1) * cols], 1, 0);
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    fn iq4nl_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let byte = blk[2 + (idx % 16)];
        let q = if idx < 16 { byte & 0x0F } else { byte >> 4 };
        d * IQ4_KV[q as usize] as f64
    }

    fn iq4xs_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let sh = u16::from_le_bytes([blk[2], blk[3]]) as u32;
        let (ib, col) = (idx / 32, idx % 32);
        let ls = unsafe { iq4xs_ls(blk[4..].as_ptr(), sh, ib) };
        let byte = blk[8 + 16 * ib + (col % 16)];
        let q = if col < 16 { byte & 0x0F } else { byte >> 4 };
        d * (ls - 32) as f64 * IQ4_KV[q as usize] as f64
    }

    #[test]
    fn q5k_matvec_and_invariance() {
        kq_certify(
            Q5K_BYTES,
            KQ_SB,
            gemv_q5k_core,
            |r, c, s| kq_random(Q5K_BYTES, KQ_SB, &[0, 2], r, c, s),
            q5k_weight_ref,
            0xA5,
        );
    }

    #[test]
    fn q2k_matvec_and_invariance() {
        kq_certify(
            Q2K_BYTES,
            KQ_SB,
            gemv_q2k_core,
            |r, c, s| kq_random(Q2K_BYTES, KQ_SB, &[80, 82], r, c, s),
            q2k_weight_ref,
            0xB2,
        );
    }

    #[test]
    fn q3k_matvec_and_invariance() {
        kq_certify(
            Q3K_BYTES,
            KQ_SB,
            gemv_q3k_core,
            |r, c, s| kq_random(Q3K_BYTES, KQ_SB, &[108], r, c, s),
            q3k_weight_ref,
            0xC3,
        );
    }

    #[test]
    fn iq4nl_matvec_and_invariance() {
        kq_certify(
            IQ4NL_BYTES,
            IQ4NL_BLK,
            gemv_iq4nl_core,
            |r, c, s| kq_random(IQ4NL_BYTES, IQ4NL_BLK, &[0], r, c, s),
            iq4nl_weight_ref,
            0x4E,
        );
    }

    #[test]
    fn iq4xs_matvec_and_invariance() {
        kq_certify(
            IQ4XS_BYTES,
            KQ_SB,
            gemv_iq4xs_core,
            |r, c, s| kq_random(IQ4XS_BYTES, KQ_SB, &[0], r, c, s),
            iq4xs_weight_ref,
            0x4C,
        );
    }

    fn q40_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let byte = blk[2 + (idx % 16)];
        let q = if idx < 16 { byte & 0x0F } else { byte >> 4 };
        d * (q as f64 - 8.0)
    }

    fn q41_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let byte = blk[4 + (idx % 16)];
        let q = if idx < 16 { byte & 0x0F } else { byte >> 4 };
        d * q as f64 + m
    }

    fn q80_weight_ref(blk: &[u8], idx: usize) -> f64 {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        d * blk[2 + idx] as i8 as f64
    }

    #[test]
    fn q40_matvec_and_invariance() {
        kq_certify(
            Q40_BYTES,
            AFFINE_BLK,
            gemv_q40_core,
            |r, c, s| kq_random(Q40_BYTES, AFFINE_BLK, &[0], r, c, s),
            q40_weight_ref,
            0x40,
        );
    }

    #[test]
    fn q41_matvec_and_invariance() {
        kq_certify(
            Q41_BYTES,
            AFFINE_BLK,
            gemv_q41_core,
            |r, c, s| kq_random(Q41_BYTES, AFFINE_BLK, &[0, 2], r, c, s),
            q41_weight_ref,
            0x41,
        );
    }

    #[test]
    fn q80_matvec_and_invariance() {
        kq_certify(
            Q80_BYTES,
            AFFINE_BLK,
            gemv_q80_core,
            |r, c, s| kq_random(Q80_BYTES, AFFINE_BLK, &[0], r, c, s),
            q80_weight_ref,
            0x80,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn affine_scalar_and_neon_widen_to_the_same_bits() {
        let (rows, cols) = (4usize, 512usize);
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let n = cols.min(COL_TILE);
        unsafe {
            let q40 = kq_random(Q40_BYTES, AFFINE_BLK, &[0], rows, cols, 0x40);
            widen_q40_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q40(q40.as_ptr()), 0, cols, n);
            widen_q40_neon::<4>(b.as_mut_ptr(), COL_TILE, Q40(q40.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q4_0"
            );
            let q41 = kq_random(Q41_BYTES, AFFINE_BLK, &[0, 2], rows, cols, 0x41);
            widen_q41_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q41(q41.as_ptr()), 0, cols, n);
            widen_q41_neon::<4>(b.as_mut_ptr(), COL_TILE, Q41(q41.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q4_1"
            );
            let q80 = kq_random(Q80_BYTES, AFFINE_BLK, &[0], rows, cols, 0x80);
            widen_q80_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q80(q80.as_ptr()), 0, cols, n);
            widen_q80_neon::<4>(b.as_mut_ptr(), COL_TILE, Q80(q80.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q8_0"
            );
        }
    }

    // ---- IQ lattice: the grid comes from gguf, so Rust checks the framework contract (thread/batch bit-
    // invariance) with a synthetic grid; correctness against real grids is the Python gguf cert. --------------

    type LattCore = unsafe fn(
        *const u8,
        *const i8,
        *const u8,
        usize,
        usize,
        *const f32,
        usize,
        *mut f32,
        usize,
    ) -> i32;

    fn latt_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 32) as u8
            })
            .collect()
    }

    /// thread and batch bit-invariance for a lattice core over a synthetic grid: row `r` is the same at any
    /// thread count, and row 0 of a 3-row pass is its own one-row call, bit for bit (the verify pass's contract).
    fn latt_invariance(core: LattCore, bytes: usize, entries: usize, vals: usize, seed: u64) {
        let (rows, cols) = (37usize, 768usize);
        // bit 6 of every raw byte cleared: it is the top exponent bit of each type's f16 delta (the leading
        // f16's high byte; for IQ1_M the nibble its delta is assembled from), so every delta is finite.
        let raw: Vec<u8> = latt_bytes(rows * cols / KQ_SB * bytes, seed)
            .iter()
            .map(|&u| u & 0xBF)
            .collect();
        let grid: Vec<i8> = latt_bytes(entries * vals, seed ^ 0x9e37)
            .iter()
            .map(|&u| u as i8)
            .collect();
        let ks = latt_bytes(128, seed ^ 0x1234);
        let x: Vec<f32> = latt_bytes(3 * cols, seed ^ 0xabcd)
            .iter()
            .map(|&u| (u as i8) as f32 * 0.01)
            .collect();
        let call = |b: usize, xs: &[f32], threads: usize| -> Vec<f32> {
            let mut y = vec![0.0f32; b * rows];
            assert_eq!(
                unsafe {
                    core(
                        raw.as_ptr(),
                        grid.as_ptr(),
                        ks.as_ptr(),
                        rows,
                        cols,
                        xs.as_ptr(),
                        b,
                        y.as_mut_ptr(),
                        threads,
                    )
                },
                OK
            );
            y
        };
        let base = call(3, &x, 1);
        // a NaN pass would certify nothing, and is not even stable: x86 picks a NaN's payload by operand order,
        // which the 4-row and 1-row kernels need not share. The contract is over finite results.
        assert!(base.iter().all(|v| v.is_finite()), "non-finite pass");
        for t in [2usize, 5, 0] {
            assert_eq!(
                call(3, &x, t)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                base.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "threads {t}"
            );
        }
        for i in 0..3 {
            let solo = call(1, &x[i * cols..(i + 1) * cols], 0);
            for r in 0..rows {
                assert_eq!(solo[r].to_bits(), base[i * rows + r].to_bits());
            }
        }
    }

    #[test]
    fn iq_lattice_is_thread_and_batch_invariant() {
        latt_invariance(gemv_iq3xxs_core, 98, 256, 4, 0x3E);
        latt_invariance(gemv_iq2xxs_core, 66, 256, 8, 0x2E);
        latt_invariance(gemv_iq2xs_core, 74, 512, 8, 0x2F);
        latt_invariance(gemv_iq2s_core, 82, 1024, 8, 0x25);
        latt_invariance(gemv_iq1s_core, 50, 2048, 8, 0x15);
        latt_invariance(gemv_iq3s_core, 110, 512, 4, 0x35);
        latt_invariance(gemv_iq1m_core, 56, 2048, 8, 0x1D);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn q5k_q2k_q3k_scalar_and_neon_widen_to_the_same_bits() {
        let (rows, cols) = (4usize, 512usize);
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let n = cols.min(COL_TILE);
        unsafe {
            let q5 = kq_random(Q5K_BYTES, KQ_SB, &[0, 2], rows, cols, 0x51);
            widen_q5k_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q5k(q5.as_ptr()), 0, cols, n);
            widen_q5k_neon::<4>(b.as_mut_ptr(), COL_TILE, Q5k(q5.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q5_K"
            );
            let q2 = kq_random(Q2K_BYTES, KQ_SB, &[80, 82], rows, cols, 0x52);
            widen_q2k_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q2k(q2.as_ptr()), 0, cols, n);
            widen_q2k_neon::<4>(b.as_mut_ptr(), COL_TILE, Q2k(q2.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q2_K"
            );
            let q3 = kq_random(Q3K_BYTES, KQ_SB, &[108], rows, cols, 0x53);
            widen_q3k_scalar::<4>(a.as_mut_ptr(), COL_TILE, Q3k(q3.as_ptr()), 0, cols, n);
            widen_q3k_neon::<4>(b.as_mut_ptr(), COL_TILE, Q3k(q3.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "Q3_K"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn iq4_scalar_and_neon_widen_to_the_same_bits() {
        let (rows, cols) = (4usize, 512usize);
        let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let mut b = vec![0.0f32; ROW_UNROLL * COL_TILE];
        let n = cols.min(COL_TILE);
        unsafe {
            let nl = kq_random(IQ4NL_BYTES, IQ4NL_BLK, &[0], rows, cols, 0x4E);
            widen_iq4nl_scalar::<4>(a.as_mut_ptr(), COL_TILE, Iq4nl(nl.as_ptr()), 0, cols, n);
            widen_iq4nl_neon::<4>(b.as_mut_ptr(), COL_TILE, Iq4nl(nl.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "IQ4_NL"
            );
            let xs = kq_random(IQ4XS_BYTES, KQ_SB, &[0], rows, cols, 0x4C);
            widen_iq4xs_scalar::<4>(a.as_mut_ptr(), COL_TILE, Iq4xs(xs.as_ptr()), 0, cols, n);
            widen_iq4xs_neon::<4>(b.as_mut_ptr(), COL_TILE, Iq4xs(xs.as_ptr()), 0, cols, n);
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "IQ4_XS"
            );
        }
    }

    /// throughput of every quant matvec at batch 1 (a decode step) and 16 (a verify pass), single core, on a
    /// matrix well past the caches, beside the bf16 matvec as the machine's single-core bandwidth yardstick (a
    /// known memory-bound kernel). If a format's `GB/s (weights)` sits near bf16's it is memory-bound too and as
    /// fast as its byte budget allows; a format that sits well under bf16's byte rate is decode-bound, and its
    /// second section below (NEON widen vs the scalar definition, decode in isolation) is the whole story of how
    /// far that decode was taken. The 7 lattice types decode by a scalar grid gather (no NEON widen exists), so
    /// they appear only in the full-matvec section. Ignored by default (a number, and a debug build's is
    /// meaningless): `cargo test --release -- --ignored --nocapture kquant_throughput`.
    #[test]
    #[ignore]
    fn kquant_throughput() {
        use std::time::Instant;
        let (rows, cols) = (4096usize, 4096usize);
        let bf: Vec<u16> = (0..rows * cols)
            .map(|i| ((i as u32 * 2 + 1) & 0xFFFF) as u16)
            .collect();
        // one raw matrix per format; kq_random lays out rows*cols weights as that type's superblocks. Decoded
        // values are irrelevant to timing, so the f16 scale offsets only keep the scales finite.
        let (q2, q3) = (
            kq_random(Q2K_BYTES, KQ_SB, &[80, 82], rows, cols, 0x02),
            kq_random(Q3K_BYTES, KQ_SB, &[108], rows, cols, 0x03),
        );
        let q4 = kq_random(Q4K_BYTES, KQ_SB, &[0, 2], rows, cols, 0x04);
        let q5 = kq_random(Q5K_BYTES, KQ_SB, &[0, 2], rows, cols, 0x05);
        let q6 = kq_random(Q6K_BYTES, KQ_SB, &[208], rows, cols, 0x06);
        let q40 = kq_random(Q40_BYTES, AFFINE_BLK, &[0], rows, cols, 0x40);
        let q41 = kq_random(Q41_BYTES, AFFINE_BLK, &[0, 2], rows, cols, 0x41);
        let q80 = kq_random(Q80_BYTES, AFFINE_BLK, &[0], rows, cols, 0x80);
        let nl = kq_random(IQ4NL_BYTES, AFFINE_BLK, &[0], rows, cols, 0x4E);
        let xs = kq_random(IQ4XS_BYTES, KQ_SB, &[0], rows, cols, 0x4C);
        let i1s = kq_random(50, KQ_SB, &[0], rows, cols, 0x11);
        let i1m = kq_random(56, KQ_SB, &[0], rows, cols, 0x1D);
        let i2xxs = kq_random(66, KQ_SB, &[0], rows, cols, 0x22);
        let i2xs = kq_random(74, KQ_SB, &[0], rows, cols, 0x23);
        let i2s = kq_random(82, KQ_SB, &[0], rows, cols, 0x25);
        let i3xxs = kq_random(98, KQ_SB, &[0], rows, cols, 0x33);
        let i3s = kq_random(110, KQ_SB, &[0], rows, cols, 0x35);
        // grid + sign table for the lattice cores, sized past every lattice type's largest gather index.
        let grid: Vec<i8> = latt_bytes(16384, 0xA1).iter().map(|&u| u as i8).collect();
        let ks = latt_bytes(128, 0xB2);

        fn bench(label: &str, wbytes: usize, gflop_w: usize, mut call: impl FnMut()) {
            call();
            let mut best = f64::MAX;
            for _ in 0..30 {
                let t = Instant::now();
                call();
                best = best.min(t.elapsed().as_secs_f64());
            }
            eprintln!(
                "{label:16}: {:.3} ms  {:6.1} GB/s (weights)  {:6.1} Gflop/s",
                best * 1e3,
                wbytes as f64 / best / 1e9,
                gflop_w as f64 / best / 1e9,
            );
        }

        type KqCore = unsafe fn(*const u8, usize, usize, *const f32, usize, *mut f32, usize) -> i32;
        let kqs: [(&str, &[u8], usize, usize, KqCore); 10] = [
            ("Q2_K", q2.as_slice(), KQ_SB, Q2K_BYTES, gemv_q2k_core),
            ("Q3_K", q3.as_slice(), KQ_SB, Q3K_BYTES, gemv_q3k_core),
            ("Q4_K", q4.as_slice(), KQ_SB, Q4K_BYTES, gemv_q4k_core),
            ("Q5_K", q5.as_slice(), KQ_SB, Q5K_BYTES, gemv_q5k_core),
            ("Q6_K", q6.as_slice(), KQ_SB, Q6K_BYTES, gemv_q6k_core),
            ("Q4_0", q40.as_slice(), AFFINE_BLK, Q40_BYTES, gemv_q40_core),
            ("Q4_1", q41.as_slice(), AFFINE_BLK, Q41_BYTES, gemv_q41_core),
            ("Q8_0", q80.as_slice(), AFFINE_BLK, Q80_BYTES, gemv_q80_core),
            (
                "IQ4_NL",
                nl.as_slice(),
                AFFINE_BLK,
                IQ4NL_BYTES,
                gemv_iq4nl_core,
            ),
            ("IQ4_XS", xs.as_slice(), KQ_SB, IQ4XS_BYTES, gemv_iq4xs_core),
        ];
        let latts: [(&str, &[u8], usize, LattCore); 7] = [
            ("IQ1_S", i1s.as_slice(), 50, gemv_iq1s_core),
            ("IQ1_M", i1m.as_slice(), 56, gemv_iq1m_core),
            ("IQ2_XXS", i2xxs.as_slice(), 66, gemv_iq2xxs_core),
            ("IQ2_XS", i2xs.as_slice(), 74, gemv_iq2xs_core),
            ("IQ2_S", i2s.as_slice(), 82, gemv_iq2s_core),
            ("IQ3_XXS", i3xxs.as_slice(), 98, gemv_iq3xxs_core),
            ("IQ3_S", i3s.as_slice(), 110, gemv_iq3s_core),
        ];

        for &b in &[1usize, 16] {
            let x: Vec<f32> = (0..b * cols).map(|i| (i as f32 * 0.001).sin()).collect();
            let mut y = vec![0.0f32; b * rows];
            let flop = 2 * rows * cols * b;
            bench(&format!("bf16 b={b}"), rows * cols * 2, flop, || unsafe {
                assert_eq!(
                    gemv_core(bf.as_ptr(), rows, cols, x.as_ptr(), b, y.as_mut_ptr(), 1),
                    OK
                );
            });
            for &(label, raw, blk, bytes, core) in &kqs {
                bench(
                    &format!("{label} b={b}"),
                    rows * cols / blk * bytes,
                    flop,
                    || unsafe {
                        assert_eq!(
                            core(raw.as_ptr(), rows, cols, x.as_ptr(), b, y.as_mut_ptr(), 1),
                            OK
                        );
                    },
                );
            }
            for &(label, raw, bytes, core) in &latts {
                bench(
                    &format!("{label} b={b}"),
                    rows * cols / KQ_SB * bytes,
                    flop,
                    || unsafe {
                        assert_eq!(
                            core(
                                raw.as_ptr(),
                                grid.as_ptr(),
                                ks.as_ptr(),
                                rows,
                                cols,
                                x.as_ptr(),
                                b,
                                y.as_mut_ptr(),
                                1,
                            ),
                            OK
                        );
                    },
                );
            }
        }

        // decode compute, NEON widen vs the scalar definition, in isolation. Each timed call decodes `reps`
        // cache-resident 4x512 strips (well past the ~40ns timer granularity), the strip offset cycling over 8
        // superblocks so the compiler can't fold the repeats into one, and a per-iteration `black_box` keeps the
        // stores from being eliminated. A ratio near 1.0 would mean the NEON path bought nothing.
        #[cfg(target_arch = "aarch64")]
        {
            let (mcols, n, reps) = (512usize, 512usize, 4096usize);
            let mut a = vec![0.0f32; ROW_UNROLL * COL_TILE];
            let ap = a.as_mut_ptr();
            let w = reps * 4 * mcols;
            fn decode(label: &str, w: usize, mut sc: impl FnMut(), mut ne: impl FnMut()) {
                sc();
                ne();
                let (mut bs, mut bn) = (f64::MAX, f64::MAX);
                for _ in 0..25 {
                    let t = Instant::now();
                    sc();
                    bs = bs.min(t.elapsed().as_secs_f64());
                    let t = Instant::now();
                    ne();
                    bn = bn.min(t.elapsed().as_secs_f64());
                }
                eprintln!(
                    "{label:8} decode: scalar {:6.2} Gw/s   neon {:6.2} Gw/s   ({:.2}x)",
                    w as f64 / bs / 1e9,
                    w as f64 / bn / 1e9,
                    bs / bn,
                );
            }
            macro_rules! dec {
                ($lbl:expr, $sc:ident, $ne:ident, $W:expr, $raw:expr) => {{
                    let rp = $raw.as_ptr();
                    decode(
                        $lbl,
                        w,
                        || {
                            for i in 0..reps {
                                unsafe {
                                    $sc::<4>(ap, COL_TILE, $W(rp), (i & 7) * KQ_SB, mcols, n)
                                };
                                std::hint::black_box(ap);
                            }
                        },
                        || {
                            for i in 0..reps {
                                unsafe {
                                    $ne::<4>(ap, COL_TILE, $W(rp), (i & 7) * KQ_SB, mcols, n)
                                };
                                std::hint::black_box(ap);
                            }
                        },
                    );
                }};
            }
            dec!("Q2_K", widen_q2k_scalar, widen_q2k_neon, Q2k, q2);
            dec!("Q3_K", widen_q3k_scalar, widen_q3k_neon, Q3k, q3);
            dec!("Q4_K", widen_q4k_scalar, widen_q4k_neon, Q4k, q4);
            dec!("Q5_K", widen_q5k_scalar, widen_q5k_neon, Q5k, q5);
            dec!("Q6_K", widen_q6k_scalar, widen_q6k_neon, Q6k, q6);
            dec!("Q4_0", widen_q40_scalar, widen_q40_neon, Q40, q40);
            dec!("Q4_1", widen_q41_scalar, widen_q41_neon, Q41, q41);
            dec!("Q8_0", widen_q80_scalar, widen_q80_neon, Q80, q80);
            dec!("IQ4_NL", widen_iq4nl_scalar, widen_iq4nl_neon, Iq4nl, nl);
            dec!("IQ4_XS", widen_iq4xs_scalar, widen_iq4xs_neon, Iq4xs, xs);
        }
    }
}
