// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Single-token grouped-query attention over a key/value cache, streamed with an online softmax: one query
//! over the cache's first `n` rows (a decode step), or a batch of queries each over its own list of cache
//! rows (a verify pass's tree nodes, each attending the rows its committed step would).

use crate::codes::{ERR_DOMAIN, ERR_NULL, OK};
use crate::gemv::{
    aligned, bf16_to_f32, default_threads, isa, pool_for, resolve_threads, Isa, MAX_ELEMS,
};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m256, __m512};

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const LANES: usize = 16;

const QUERY_UNROLL: usize = 4;

const MIN_ROWS: usize = 64;

trait Element: Copy + Send + Sync {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const BF16: bool;

    fn widen(self) -> f32;
}

impl Element for u16 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const BF16: bool = true;

    #[inline(always)]
    fn widen(self) -> f32 {
        bf16_to_f32(self)
    }
}

impl Element for f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const BF16: bool = false;

    #[inline(always)]
    fn widen(self) -> f32 {
        self
    }
}

#[derive(Clone, Copy)]
struct Job<E> {
    query: *const f32,
    key: *const E,
    value: *const E,
    /// The cache rows the query attends, in the order it attends them; null for rows `0..n` in order, a
    /// decode step's cache. A range reads row `index[r]` where the step reads row `r`, and nothing else
    /// about the arithmetic changes, so a list of the rows `0..n` is the step bit for bit.
    index: *const u32,
    d: usize,
    group: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
}

// SAFETY: a job is read-only input; the slots the threads write are disjoint ranges of `partials`
unsafe impl<E> Send for Job<E> {}
unsafe impl<E> Sync for Job<E> {}

#[inline(always)]
unsafe fn rescale(acc: *mut f32, d: usize, factor: f32) {
    for i in 0..d {
        *acc.add(i) *= factor;
    }
}

#[inline(always)]
unsafe fn scores_tail<E: Element, const G: usize>(
    query: *const f32,
    key: *const E,
    d: usize,
    base: usize,
    score: *mut f32,
) {
    for i in base..d {
        let element = (*key.add(i)).widen();
        for u in 0..G {
            let s = score.add(u);
            *s = element.mul_add(*query.add(u * d + i), *s);
        }
    }
}

#[inline(always)]
unsafe fn scores_scalar<E: Element, const G: usize>(
    query: *const f32,
    key: *const E,
    d: usize,
    score: *mut f32,
) {
    for u in 0..G {
        *score.add(u) = 0.0;
    }
    scores_tail::<E, G>(query, key, d, 0, score);
}

#[inline(always)]
unsafe fn accumulate_tail<E: Element>(
    value: *const E,
    prob: *const f32,
    acc: *mut f32,
    d: usize,
    group: usize,
    base: usize,
) {
    for i in base..d {
        let element = (*value.add(i)).widen();
        for u in 0..group {
            let a = acc.add(u * d + i);
            *a = element.mul_add(*prob.add(u), *a);
        }
    }
}

#[inline(always)]
unsafe fn accumulate_scalar<E: Element>(
    value: *const E,
    prob: *const f32,
    acc: *mut f32,
    d: usize,
    group: usize,
) {
    accumulate_tail::<E>(value, prob, acc, d, group, 0);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load8_avx2<E: Element>(row: *const E) -> __m256 {
    use std::arch::x86_64::*;
    if E::BF16 {
        let packed = _mm_loadu_si128(row as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(packed)))
    } else {
        _mm256_loadu_ps(row as *const f32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce8_avx2(v: __m256) -> f32 {
    use std::arch::x86_64::*;
    let pairs = _mm256_hadd_ps(v, v);
    let quads = _mm256_hadd_ps(pairs, pairs);
    let low = _mm256_castps256_ps128(quads);
    let high = _mm256_extractf128_ps::<1>(quads);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores_avx2<E: Element, const G: usize>(
    query: *const f32,
    key: *const E,
    d: usize,
    score: *mut f32,
) {
    use std::arch::x86_64::*;

    let mut lo = [_mm256_setzero_ps(); G];
    let mut hi = [_mm256_setzero_ps(); G];

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let klo = load8_avx2(key.add(c));
        let khi = load8_avx2(key.add(c + 8));
        for u in 0..G {
            let q = query.add(u * d + c);
            lo[u] = _mm256_fmadd_ps(klo, _mm256_loadu_ps(q), lo[u]);
            hi[u] = _mm256_fmadd_ps(khi, _mm256_loadu_ps(q.add(8)), hi[u]);
        }
    }

    for u in 0..G {
        *score.add(u) = reduce8_avx2(_mm256_add_ps(lo[u], hi[u]));
    }
    scores_tail::<E, G>(query, key, d, nb * LANES, score);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn accumulate_avx2<E: Element>(
    value: *const E,
    prob: *const f32,
    acc: *mut f32,
    d: usize,
    group: usize,
) {
    use std::arch::x86_64::*;

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let vlo = load8_avx2(value.add(c));
        let vhi = load8_avx2(value.add(c + 8));
        for u in 0..group {
            let p = _mm256_set1_ps(*prob.add(u));
            let a = acc.add(u * d + c);
            _mm256_storeu_ps(a, _mm256_fmadd_ps(vlo, p, _mm256_loadu_ps(a)));
            _mm256_storeu_ps(a.add(8), _mm256_fmadd_ps(vhi, p, _mm256_loadu_ps(a.add(8))));
        }
    }
    accumulate_tail::<E>(value, prob, acc, d, group, nb * LANES);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn load16_avx512<E: Element>(row: *const E) -> __m512 {
    use std::arch::x86_64::*;
    if E::BF16 {
        let packed = _mm256_loadu_si256(row as *const __m256i);
        _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(packed)))
    } else {
        _mm512_loadu_ps(row as *const f32)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum16_avx512(v: __m512) -> f32 {
    use std::arch::x86_64::*;
    let mut lane = [0.0f32; LANES];
    _mm512_storeu_ps(lane.as_mut_ptr(), v);
    lane.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(clippy::needless_range_loop)]
unsafe fn scores_avx512<E: Element, const G: usize>(
    query: *const f32,
    key: *const E,
    d: usize,
    score: *mut f32,
) {
    use std::arch::x86_64::*;

    let mut sum = [_mm512_setzero_ps(); G];

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let k = load16_avx512(key.add(c));
        for u in 0..G {
            sum[u] = _mm512_fmadd_ps(k, _mm512_loadu_ps(query.add(u * d + c)), sum[u]);
        }
    }

    for u in 0..G {
        *score.add(u) = sum16_avx512(sum[u]);
    }
    scores_tail::<E, G>(query, key, d, nb * LANES, score);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn accumulate_avx512<E: Element>(
    value: *const E,
    prob: *const f32,
    acc: *mut f32,
    d: usize,
    group: usize,
) {
    use std::arch::x86_64::*;

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let v = load16_avx512(value.add(c));
        for u in 0..group {
            let a = acc.add(u * d + c);
            let p = _mm512_set1_ps(*prob.add(u));
            _mm512_storeu_ps(a, _mm512_fmadd_ps(v, p, _mm512_loadu_ps(a)));
        }
    }
    accumulate_tail::<E>(value, prob, acc, d, group, nb * LANES);
}

// NEON: sixteen elements as four float32x4 vectors, bf16 rows widened by a 16-bit shift.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load16_neon<E: Element>(row: *const E) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    if E::BF16 {
        let p = row as *const u16;
        let a = vld1q_u16(p);
        let b = vld1q_u16(p.add(8));
        [
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(a))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(a)),
            vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(b))),
            vreinterpretq_f32_u32(vshll_high_n_u16::<16>(b)),
        ]
    } else {
        let p = row as *const f32;
        [
            vld1q_f32(p),
            vld1q_f32(p.add(4)),
            vld1q_f32(p.add(8)),
            vld1q_f32(p.add(12)),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::needless_range_loop)]
unsafe fn scores_neon<E: Element, const G: usize>(
    query: *const f32,
    key: *const E,
    d: usize,
    score: *mut f32,
) {
    use std::arch::aarch64::*;

    let mut sum = [[vdupq_n_f32(0.0); 4]; G];

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let k = load16_neon(key.add(c));
        for u in 0..G {
            let q = query.add(u * d + c);
            sum[u][0] = vfmaq_f32(sum[u][0], k[0], vld1q_f32(q));
            sum[u][1] = vfmaq_f32(sum[u][1], k[1], vld1q_f32(q.add(4)));
            sum[u][2] = vfmaq_f32(sum[u][2], k[2], vld1q_f32(q.add(8)));
            sum[u][3] = vfmaq_f32(sum[u][3], k[3], vld1q_f32(q.add(12)));
        }
    }

    for u in 0..G {
        let v = vaddq_f32(
            vaddq_f32(sum[u][0], sum[u][1]),
            vaddq_f32(sum[u][2], sum[u][3]),
        );
        *score.add(u) = vaddvq_f32(v);
    }
    scores_tail::<E, G>(query, key, d, nb * LANES, score);
}

#[cfg(target_arch = "aarch64")]
unsafe fn accumulate_neon<E: Element>(
    value: *const E,
    prob: *const f32,
    acc: *mut f32,
    d: usize,
    group: usize,
) {
    use std::arch::aarch64::*;

    let nb = d / LANES;
    for b in 0..nb {
        let c = b * LANES;
        let v = load16_neon(value.add(c));
        for u in 0..group {
            let p = vdupq_n_f32(*prob.add(u));
            let a = acc.add(u * d + c);
            vst1q_f32(a, vfmaq_f32(vld1q_f32(a), v[0], p));
            vst1q_f32(a.add(4), vfmaq_f32(vld1q_f32(a.add(4)), v[1], p));
            vst1q_f32(a.add(8), vfmaq_f32(vld1q_f32(a.add(8)), v[2], p));
            vst1q_f32(a.add(12), vfmaq_f32(vld1q_f32(a.add(12)), v[3], p));
        }
    }
    accumulate_tail::<E>(value, prob, acc, d, group, nb * LANES);
}

macro_rules! def_range {
    ($name:ident, $scores:ident, $accumulate:ident $(, $feat:literal)?) => {

        $(#[target_feature(enable = $feat)])?
        unsafe fn $name<E: Element>(
            job: Job<E>,
            head: usize,
            r0: usize,
            r1: usize,
            state: *mut f32,
            work: *mut f32,
        ) {
            let d = job.d;
            let group = job.group;

            let acc = state;
            let running_max = state.add(group * d);
            let running_sum = running_max.add(group);
            let score = work;
            let prob = work.add(group);

            let query = job.query.add(head * group * d);
            let key = job.key.add(head * job.key_stride);
            let value = job.value.add(head * job.value_stride);

            for r in r0..r1 {
                let row = if job.index.is_null() {
                    r
                } else {
                    *job.index.add(r) as usize
                };
                let key_row = key.add(row * d);
                let mut u = 0;
                while u + QUERY_UNROLL <= group {
                    $scores::<E, QUERY_UNROLL>(query.add(u * d), key_row, d, score.add(u));
                    u += QUERY_UNROLL;
                }
                while u + 2 <= group {
                    $scores::<E, 2>(query.add(u * d), key_row, d, score.add(u));
                    u += 2;
                }
                while u < group {
                    $scores::<E, 1>(query.add(u * d), key_row, d, score.add(u));
                    u += 1;
                }

                for u in 0..group {
                    let s = *score.add(u) * job.scale;
                    let m = *running_max.add(u);
                    if s > m {
                        let correction = (m - s).exp();
                        rescale(acc.add(u * d), d, correction);
                        *running_sum.add(u) = *running_sum.add(u) * correction + 1.0;
                        *running_max.add(u) = s;
                        *prob.add(u) = 1.0;
                    } else {
                        let p = (s - m).exp();
                        *running_sum.add(u) += p;
                        *prob.add(u) = p;
                    }
                }

                $accumulate::<E>(value.add(row * d), prob, acc, d, group);
            }
        }
    };
}

def_range!(range_scalar, scores_scalar, accumulate_scalar);
#[cfg(target_arch = "aarch64")]
def_range!(range_neon, scores_neon, accumulate_neon);
#[cfg(target_arch = "x86_64")]
def_range!(range_avx2, scores_avx2, accumulate_avx2, "avx2,fma");
#[cfg(target_arch = "x86_64")]
def_range!(
    range_avx512,
    scores_avx512,
    accumulate_avx512,
    "avx512f,avx512bw,fma"
);

#[inline]
unsafe fn run_range<E: Element>(
    selected: Isa,
    job: Job<E>,
    head: usize,
    r0: usize,
    r1: usize,
    state: *mut f32,
    work: *mut f32,
) {
    match selected {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => range_avx512(job, head, r0, r1, state, work),
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => range_avx2(job, head, r0, r1, state, work),
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => range_neon(job, head, r0, r1, state, work),
        _ => range_scalar(job, head, r0, r1, state, work),
    }
}

/// One query row's share of a call: the list it attends (`len` entries of the call's index from `start`, or
/// the rows `0..len` when the call has none), the rows of that list per slot, and the slots it takes per kv
/// head, the first of them at `first` among the call's.
#[derive(Clone, Copy)]
struct Span {
    start: usize,
    len: usize,
    chunk: usize,
    parts: usize,
    first: usize,
}

/// The rows per slot and the slots per kv head of a list of `n` rows, from its own length alone: a list
/// splits into the same pieces as a decode step over `n` rows, whatever else shares its call, so it sums
/// in the same order and comes out the step's bits.
fn split(n: usize, nt: usize, hk: usize) -> (usize, usize) {
    let chunk = if nt <= 1 {
        n
    } else {
        n.div_ceil(nt.saturating_mul(2).div_ceil(hk).max(1))
            .max(MIN_ROWS)
    };
    (chunk, n.div_ceil(chunk))
}

/// The slot `slot` of a call: `job` is the call's (its query row 0, its whole index), `row` the elements
/// of one query row (`hq * d`).
#[inline]
unsafe fn run_slot<E: Element>(
    job: Job<E>,
    selected: Isa,
    spans: &[Span],
    row: usize,
    slot: usize,
    state: &mut [f32],
) {
    // the query row owning this slot: the last span starting at or before it
    let t = spans.partition_point(|s| s.first <= slot) - 1;
    let span = spans[t];
    let local = slot - span.first;
    let head = local / span.parts;
    let start = (local % span.parts) * span.chunk;
    let stop = (start + span.chunk).min(span.len);
    let job = Job {
        query: job.query.add(t * row),
        index: if job.index.is_null() {
            job.index
        } else {
            job.index.add(span.start)
        },
        ..job
    };
    // two f32 per query head of the group: on the stack for any group up to 32
    let mut stack = [0.0f32; 64];
    let mut heap: Vec<f32> = Vec::new();
    let work = if 2 * job.group <= stack.len() {
        stack.as_mut_ptr()
    } else {
        heap.resize(2 * job.group, 0.0);
        heap.as_mut_ptr()
    };
    run_range(selected, job, head, start, stop, state.as_mut_ptr(), work);
}

#[inline]
unsafe fn merge(
    partials: &[f32],
    out: *mut f32,
    hk: usize,
    group: usize,
    d: usize,
    parts: usize,
    width: usize,
) {
    let base = partials.as_ptr();
    for head in 0..hk {
        for u in 0..group {
            let dst = out.add((head * group + u) * d);
            for i in 0..d {
                *dst.add(i) = 0.0;
            }
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0f32;
            for p in 0..parts {
                let slot = base.add((head * parts + p) * width);
                let src = slot.add(u * d);
                let stat = slot.add(group * d + u);
                let part_max = *stat;
                let part_sum = *stat.add(group);
                let top = running_max.max(part_max);
                let keep = if running_max.is_finite() {
                    (running_max - top).exp()
                } else {
                    0.0
                };
                let take = if part_max.is_finite() {
                    (part_max - top).exp()
                } else {
                    0.0
                };
                for i in 0..d {
                    let a = dst.add(i);
                    *a = *a * keep + *src.add(i) * take;
                }
                running_sum = running_sum * keep + part_sum * take;
                running_max = top;
            }
            let inv = 1.0 / running_sum;
            for i in 0..d {
                *dst.add(i) *= inv;
            }
        }
    }
}

/// The slots of every span over the pool (or in turn), then each query row's slots merged into its output
/// row. `job` is the call's: query row 0 and the whole index. Which thread runs a slot never reaches its
/// sums, and each row merges its own slots in their order, so the rows are independent of each other.
unsafe fn attend<E: Element>(
    job: Job<E>,
    spans: &[Span],
    hk: usize,
    width: usize,
    out: *mut f32,
    nt: usize,
) -> i32 {
    let (d, group) = (job.d, job.group);
    let row = hk * group * d;
    let last = spans[spans.len() - 1];
    let slots = match last
        .parts
        .checked_mul(hk)
        .and_then(|s| s.checked_add(last.first))
    {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let total = match slots.checked_mul(width) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };

    let mut partials = vec![0.0f32; total];
    for slot in partials.chunks_mut(width) {
        slot[group * d..group * d + group].fill(f32::NEG_INFINITY);
    }

    let selected = isa();

    if nt <= 1 || slots == 1 {
        for (slot, state) in partials.chunks_mut(width).enumerate() {
            run_slot(job, selected, spans, row, slot, state);
        }
    } else {
        let mut body = || {
            partials
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(slot, state)| unsafe {
                    run_slot(job, selected, spans, row, slot, state)
                });
        };
        if nt == default_threads() {
            body();
        } else if let Some(pool) = pool_for(nt) {
            pool.install(body);
        } else {
            body();
        }
    }

    for (t, span) in spans.iter().enumerate() {
        let mine = &partials[span.first * width..(span.first + hk * span.parts) * width];
        merge(mine, out.add(t * row), hk, group, d, span.parts, width);
    }
    OK
}

#[allow(clippy::too_many_arguments)]
unsafe fn decode<E: Element>(
    query: *const f32,
    key: *const E,
    value: *const E,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    if query.is_null() || key.is_null() || value.is_null() || out.is_null() {
        return ERR_NULL;
    }
    if n == 0 || hq == 0 || hk == 0 || d == 0 || !hq.is_multiple_of(hk) {
        return ERR_DOMAIN;
    }
    // a non-finite scale would fold NaN through every row silently; the scalar and tail paths dereference
    // the elements directly, so the buffers must be naturally aligned
    if !scale.is_finite() || !aligned(query) || !aligned(key) || !aligned(value) || !aligned(out) {
        return ERR_DOMAIN;
    }

    let rows = match n.checked_mul(d) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    if key_stride < rows || value_stride < rows {
        return ERR_DOMAIN;
    }
    // the element counts fit usize and their byte sizes a pointer offset
    if hq.checked_mul(d).is_none_or(|v| v > MAX_ELEMS)
        || hk.checked_mul(key_stride).is_none_or(|v| v > MAX_ELEMS)
        || hk.checked_mul(value_stride).is_none_or(|v| v > MAX_ELEMS)
    {
        return ERR_DOMAIN;
    }

    let group = hq / hk;
    let width = match d.checked_add(2).and_then(|w| group.checked_mul(w)) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };

    let nt = match resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let (chunk, parts) = split(n, nt, hk);
    let spans = [Span {
        start: 0,
        len: n,
        chunk,
        parts,
        first: 0,
    }];

    let job = Job {
        query,
        key,
        value,
        index: std::ptr::null(),
        d,
        group,
        key_stride,
        value_stride,
        scale,
    };
    attend(job, &spans, hk, width, out, nt)
}

#[allow(clippy::too_many_arguments)]
unsafe fn nodes<E: Element>(
    query: *const f32,
    key: *const E,
    value: *const E,
    offsets: *const u32,
    index: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    if query.is_null()
        || key.is_null()
        || value.is_null()
        || offsets.is_null()
        || index.is_null()
        || out.is_null()
    {
        return ERR_NULL;
    }
    if t == 0 || n_rows == 0 || hq == 0 || hk == 0 || d == 0 || !hq.is_multiple_of(hk) {
        return ERR_DOMAIN;
    }
    // as decode's: a finite scale, and every buffer naturally aligned (the lists are read as u32 directly)
    if !scale.is_finite()
        || !aligned(query)
        || !aligned(key)
        || !aligned(value)
        || !aligned(offsets)
        || !aligned(index)
        || !aligned(out)
    {
        return ERR_DOMAIN;
    }

    let rows = match n_rows.checked_mul(d) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    if key_stride < rows || value_stride < rows {
        return ERR_DOMAIN;
    }
    // the element counts fit usize and their byte sizes a pointer offset: `t` query and output rows of
    // `hq * d` f32, and `t + 1` offsets
    if hq
        .checked_mul(d)
        .and_then(|v| v.checked_mul(t))
        .is_none_or(|v| v > MAX_ELEMS)
        || t >= MAX_ELEMS
        || hk.checked_mul(key_stride).is_none_or(|v| v > MAX_ELEMS)
        || hk.checked_mul(value_stride).is_none_or(|v| v > MAX_ELEMS)
    {
        return ERR_DOMAIN;
    }

    // every list holds at least one row (a softmax over none has no value), and every entry is a row the
    // caller vouched for: one past `n_rows` would read keys outside the cache
    let offs = std::slice::from_raw_parts(offsets, t + 1);
    if offs.windows(2).any(|w| w[1] <= w[0]) {
        return ERR_DOMAIN;
    }
    let (lo, hi) = (offs[0] as usize, offs[t] as usize);
    let list = std::slice::from_raw_parts(index.add(lo), hi - lo);
    if list.iter().any(|&r| r as usize >= n_rows) {
        return ERR_DOMAIN;
    }

    let group = hq / hk;
    let width = match d.checked_add(2).and_then(|w| group.checked_mul(w)) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };

    let nt = match resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let mut spans = Vec::with_capacity(t);
    let mut first = 0usize;
    for w in offs.windows(2) {
        let len = (w[1] - w[0]) as usize;
        let (chunk, parts) = split(len, nt, hk);
        spans.push(Span {
            start: w[0] as usize,
            len,
            chunk,
            parts,
            first,
        });
        first = match parts.checked_mul(hk).and_then(|s| s.checked_add(first)) {
            Some(v) => v,
            None => return ERR_DOMAIN,
        };
    }

    let job = Job {
        query,
        key,
        value,
        index,
        d,
        group,
        key_stride,
        value_stride,
        scale,
    };
    attend(job, &spans, hk, width, out, nt)
}

/// `t` queries each over a span of one map of cache rows: query `i` attends the rows
/// `map[starts[i]..ends[i]]` in map order. The spans may overlap - a chunk of a prompt, each row over the
/// rows before it and itself, is one map of `base + t` rows - so a call costs its map, not a list per query.
/// A span splits as a list of its length does (`nodes`), so each row is the one-row step's bits over its rows.
#[allow(clippy::too_many_arguments)]
unsafe fn over_spans<E: Element>(
    query: *const f32,
    key: *const E,
    value: *const E,
    map: *const u32,
    n_map: usize,
    starts: *const u32,
    ends: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    if query.is_null()
        || key.is_null()
        || value.is_null()
        || map.is_null()
        || starts.is_null()
        || ends.is_null()
        || out.is_null()
    {
        return ERR_NULL;
    }
    if t == 0 || n_map == 0 || n_rows == 0 || hq == 0 || hk == 0 || d == 0 || !hq.is_multiple_of(hk)
    {
        return ERR_DOMAIN;
    }
    // as decode's: a finite scale, and every buffer naturally aligned (the map and the spans read as u32)
    if !scale.is_finite()
        || !aligned(query)
        || !aligned(key)
        || !aligned(value)
        || !aligned(map)
        || !aligned(starts)
        || !aligned(ends)
        || !aligned(out)
    {
        return ERR_DOMAIN;
    }

    let rows = match n_rows.checked_mul(d) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    if key_stride < rows || value_stride < rows {
        return ERR_DOMAIN;
    }
    // the element counts fit usize and their byte sizes a pointer offset: `t` query and output rows of
    // `hq * d` f32, `t` starts and ends, and the map
    if hq
        .checked_mul(d)
        .and_then(|v| v.checked_mul(t))
        .is_none_or(|v| v > MAX_ELEMS)
        || t > MAX_ELEMS
        || n_map > MAX_ELEMS
        || hk.checked_mul(key_stride).is_none_or(|v| v > MAX_ELEMS)
        || hk.checked_mul(value_stride).is_none_or(|v| v > MAX_ELEMS)
    {
        return ERR_DOMAIN;
    }

    // every span holds at least one row and lies inside the map, and every row of the map is one the caller
    // vouched for: one past `n_rows` would read keys outside the cache
    let lo = std::slice::from_raw_parts(starts, t);
    let hi = std::slice::from_raw_parts(ends, t);
    if lo
        .iter()
        .zip(hi)
        .any(|(&s, &e)| e <= s || e as usize > n_map)
    {
        return ERR_DOMAIN;
    }
    if std::slice::from_raw_parts(map, n_map)
        .iter()
        .any(|&r| r as usize >= n_rows)
    {
        return ERR_DOMAIN;
    }

    let group = hq / hk;
    let width = match d.checked_add(2).and_then(|w| group.checked_mul(w)) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };

    let nt = match resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let mut spans = Vec::with_capacity(t);
    let mut first = 0usize;
    for (&s, &e) in lo.iter().zip(hi) {
        let len = (e - s) as usize;
        let (chunk, parts) = split(len, nt, hk);
        spans.push(Span {
            start: s as usize,
            len,
            chunk,
            parts,
            first,
        });
        first = match parts.checked_mul(hk).and_then(|x| x.checked_add(first)) {
            Some(v) => v,
            None => return ERR_DOMAIN,
        };
    }

    let job = Job {
        query,
        key,
        value,
        index: map,
        d,
        group,
        key_stride,
        value_stride,
        scale,
    };
    attend(job, &spans, hk, width, out, nt)
}

/// One decode step of grouped-query attention over a bf16 key/value cache.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_core_bf16(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    decode::<u16>(
        q,
        k,
        v,
        n,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}

/// One decode step of grouped-query attention over an f32 key/value cache.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_core_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    decode::<f32>(
        q,
        k,
        v,
        n,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}

/// `t` single-token queries of grouped-query attention over a bf16 key/value cache, each over its own list
/// of cache rows.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nodes_core_bf16(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    offs: *const u32,
    idx: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    nodes::<u16>(
        q,
        k,
        v,
        offs,
        idx,
        t,
        n_rows,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}

/// `t` single-token queries of grouped-query attention over an f32 key/value cache, each over its own list
/// of cache rows.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn nodes_core_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    offs: *const u32,
    idx: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    nodes::<f32>(
        q,
        k,
        v,
        offs,
        idx,
        t,
        n_rows,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}

/// `t` single-token queries of grouped-query attention over a bf16 key/value cache, each over a span of one
/// map of cache rows.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn spans_core_bf16(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    map: *const u32,
    n_map: usize,
    starts: *const u32,
    ends: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    over_spans::<u16>(
        q,
        k,
        v,
        map,
        n_map,
        starts,
        ends,
        t,
        n_rows,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}

/// `t` single-token queries of grouped-query attention over an f32 key/value cache, each over a span of one
/// map of cache rows.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn spans_core_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    map: *const u32,
    n_map: usize,
    starts: *const u32,
    ends: *const u32,
    t: usize,
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    over_spans::<f32>(
        q,
        k,
        v,
        map,
        n_map,
        starts,
        ends,
        t,
        n_rows,
        hq,
        hk,
        d,
        k_head_stride,
        v_head_stride,
        scale,
        out,
        threads,
    )
}
