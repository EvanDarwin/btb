// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Grouped-query attention decode through the C ABI on plain heap buffers: both cache element
//! types against the f64 reference over a spread of shapes and head-stride paddings, the thread
//! count as a scheduling choice only, and the codes a malformed call returns. The fenced sweeps
//! over the same kernel live in `guard_attn.rs`.

use btb_native::codes::*;
use btb_native::{btb_attn_decode_bf16, btb_attn_decode_f32};
use std::time::Instant;

#[path = "common/refs.rs"]
mod refs;

use refs::{attn_reference, gen_f32, truncate, widen, worst_rel, Rng, POISON_U16};

/// `(n, hq, hk, d)`: one key-value head, a wide grouped-query head count, a small head dimension,
/// and cache lengths from a single row to well past any tile boundary.
const SHAPES: &[(usize, usize, usize, usize)] = &[
    (1, 32, 8, 128),
    (5, 32, 8, 128),
    (1000, 32, 8, 128),
    (4097, 32, 8, 128),
    (20000, 32, 8, 128),
    (777, 4, 2, 64),
    (300, 6, 6, 128),
];

const TOL: f64 = 1e-4;

const THREAD_TOL: f64 = 1e-5;

struct Cache {
    bits: Vec<u16>,
    values: Vec<f32>,
}

/// A cache of `n` rows per head inside a buffer of `stride` elements per head. The rows past `n`
/// hold a NaN bit pattern, so a read into the padding makes the output non-finite.
fn gen_bits(hk: usize, n: usize, d: usize, stride: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng::new(seed);
    let mut out = vec![POISON_U16; hk * stride];
    for h in 0..hk {
        for i in 0..n * d {
            out[h * stride + i] = truncate(rng.f32());
        }
    }
    out
}

fn gen_cache(hk: usize, n: usize, d: usize, stride: usize, seed: u64) -> Cache {
    let bits = gen_bits(hk, n, d, stride, seed);
    let values = bits.iter().map(|b| widen(*b)).collect();
    Cache { bits, values }
}

#[allow(clippy::too_many_arguments)]
fn decode_bf16(
    query: &[f32],
    key: &Cache,
    value: &Cache,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; hq * d];
    let code = unsafe {
        btb_attn_decode_bf16(
            query.as_ptr(),
            key.bits.as_ptr(),
            value.bits.as_ptr(),
            n,
            hq,
            hk,
            d,
            key_stride,
            value_stride,
            scale,
            out.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_attn_decode_bf16 returned {code}");
    out
}

#[allow(clippy::too_many_arguments)]
fn decode_f32(
    query: &[f32],
    key: &Cache,
    value: &Cache,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; hq * d];
    let code = unsafe {
        btb_attn_decode_f32(
            query.as_ptr(),
            key.values.as_ptr(),
            value.values.as_ptr(),
            n,
            hq,
            hk,
            d,
            key_stride,
            value_stride,
            scale,
            out.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_attn_decode_f32 returned {code}");
    out
}

/// The worst absolute difference from a baseline output, against that baseline's peak. Every
/// element is checked for finiteness first, because a NaN loses a `max` comparison silently.
fn worst_pair(got: &[f32], base: &[f32]) -> f64 {
    let peak = base.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
    let mut err = 0.0f64;
    for (g, b) in got.iter().zip(base.iter()) {
        assert!(g.is_finite(), "output {g} is not finite");
        assert!(b.is_finite(), "baseline {b} is not finite");
        err = err.max((*g as f64 - *b as f64).abs());
    }
    err / peak
}

/// Both cache element types land within tolerance of the f64 reference, at every shape and at
/// head strides that leave capacity past `n`. The padding changes the strides, not the values, so
/// the reference is taken once per shape.
#[test]
fn both_element_types_match_the_f64_reference() {
    for (idx, &(n, hq, hk, d)) in SHAPES.iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        let query = gen_f32(hq * d, 0xA77E4 + idx as u64);
        // the longest cache runs the tight stride only: the padded strides cost the same reference
        let pads: &[(usize, usize)] = if n >= 20000 {
            &[(0, 0)]
        } else {
            &[(0, 0), (3, 3), (3, 1)]
        };
        let mut want: Option<Vec<f64>> = None;

        for &(kpad, vpad) in pads {
            let key_stride = n * d + kpad * d;
            let value_stride = n * d + vpad * d;
            let key = gen_cache(hk, n, d, key_stride, 0xC0FFEE + idx as u64);
            let value = gen_cache(hk, n, d, value_stride, 0xBEEF + idx as u64);

            let want = want.get_or_insert_with(|| {
                attn_reference(
                    &query,
                    &key.values,
                    &value.values,
                    n,
                    hq,
                    hk,
                    d,
                    key_stride,
                    value_stride,
                    scale,
                )
            });

            for threads in [1usize, 0] {
                let a = decode_bf16(
                    &query,
                    &key,
                    &value,
                    n,
                    hq,
                    hk,
                    d,
                    key_stride,
                    value_stride,
                    scale,
                    threads,
                );
                let b = decode_f32(
                    &query,
                    &key,
                    &value,
                    n,
                    hq,
                    hk,
                    d,
                    key_stride,
                    value_stride,
                    scale,
                    threads,
                );
                let ea = worst_rel(&a, want);
                let eb = worst_rel(&b, want);
                assert!(
                    ea <= TOL,
                    "bf16 ({n},{hq},{hk},{d}) pads ({kpad},{vpad}) threads {threads}: {ea:.3e}"
                );
                assert!(
                    eb <= TOL,
                    "f32 ({n},{hq},{hk},{d}) pads ({kpad},{vpad}) threads {threads}: {eb:.3e}"
                );
                if threads == 1 {
                    eprintln!(
                        "n {n:>6} hq {hq:>3} hk {hk:>2} d {d:>4} pads ({kpad},{vpad})  \
                         bf16 {ea:.2e}  f32 {eb:.2e}"
                    );
                }
            }
        }
    }
}

/// With one row in the cache the softmax is one, so every query head must get that head's value
/// row back exactly, bit for bit.
#[test]
fn a_single_row_returns_that_value_row() {
    let (n, hq, hk, d) = (1usize, 32usize, 8usize, 128usize);
    let group = hq / hk;
    let query = gen_f32(hq * d, 0x51A9E);
    let key = gen_cache(hk, n, d, n * d, 0x11);
    let value = gen_cache(hk, n, d, n * d, 0x22);
    let out = decode_bf16(&query, &key, &value, n, hq, hk, d, n * d, n * d, 0.125, 0);
    for j in 0..hq {
        let h = j / group;
        for i in 0..d {
            assert_eq!(
                out[j * d + i],
                value.values[h * d + i],
                "head {j} lane {i}: one row must come back exactly"
            );
        }
    }
}

/// The row split is a scheduling choice: the output at two, seven and the default thread count
/// stays within a tight bound of the single-threaded one at every shape.
#[test]
fn thread_counts_agree() {
    for (idx, &(n, hq, hk, d)) in SHAPES.iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        let key_stride = n * d + 3 * d;
        let value_stride = n * d + d;
        let query = gen_f32(hq * d, 0x7EA5 + idx as u64);
        let key = gen_cache(hk, n, d, key_stride, 0x1234 + idx as u64);
        let value = gen_cache(hk, n, d, value_stride, 0x5678 + idx as u64);

        let base = decode_bf16(
            &query,
            &key,
            &value,
            n,
            hq,
            hk,
            d,
            key_stride,
            value_stride,
            scale,
            1,
        );
        let base_f32 = decode_f32(
            &query,
            &key,
            &value,
            n,
            hq,
            hk,
            d,
            key_stride,
            value_stride,
            scale,
            1,
        );
        for threads in [2usize, 7, 0] {
            let a = decode_bf16(
                &query,
                &key,
                &value,
                n,
                hq,
                hk,
                d,
                key_stride,
                value_stride,
                scale,
                threads,
            );
            let b = decode_f32(
                &query,
                &key,
                &value,
                n,
                hq,
                hk,
                d,
                key_stride,
                value_stride,
                scale,
                threads,
            );
            let ea = worst_pair(&a, &base);
            let eb = worst_pair(&b, &base_f32);
            assert!(
                ea <= THREAD_TOL,
                "bf16 ({n},{hq},{hk},{d}) threads {threads}: {ea:.3e}"
            );
            assert!(
                eb <= THREAD_TOL,
                "f32 ({n},{hq},{hk},{d}) threads {threads}: {eb:.3e}"
            );
        }
    }
}

/// A null pointer, a head count that does not divide, a head stride shorter than the cache, a
/// zero dimension, an overflowing length and a NaN scale are each refused with a code, and a
/// refused call leaves the output buffer alone.
#[test]
fn bad_pointers_and_shapes_are_error_codes_not_guesses() {
    let (n, hq, hk, d) = (8usize, 4usize, 2usize, 8usize);
    let query = vec![0.5f32; hq * d];
    let key = vec![0u16; hk * n * d];
    let value = vec![0u16; hk * n * d];
    let keyf = vec![0.5f32; hk * n * d];
    let valuef = vec![0.5f32; hk * n * d];
    let mut out = vec![0.0f32; hq * d];

    let call = |q: *const f32,
                k: *const u16,
                v: *const u16,
                n: usize,
                hq: usize,
                hk: usize,
                d: usize,
                ks: usize,
                vs: usize,
                o: *mut f32| unsafe {
        btb_attn_decode_bf16(q, k, v, n, hq, hk, d, ks, vs, 0.125, o, 0)
    };
    let (q, k, v) = (query.as_ptr(), key.as_ptr(), value.as_ptr());
    let stride = n * d;

    unsafe {
        assert_eq!(
            call(
                std::ptr::null(),
                k,
                v,
                n,
                hq,
                hk,
                d,
                stride,
                stride,
                out.as_mut_ptr()
            ),
            ERR_NULL
        );
        assert_eq!(
            call(
                q,
                std::ptr::null(),
                v,
                n,
                hq,
                hk,
                d,
                stride,
                stride,
                out.as_mut_ptr()
            ),
            ERR_NULL
        );
        assert_eq!(
            call(
                q,
                k,
                std::ptr::null(),
                n,
                hq,
                hk,
                d,
                stride,
                stride,
                out.as_mut_ptr()
            ),
            ERR_NULL
        );
        assert_eq!(
            call(q, k, v, n, hq, hk, d, stride, stride, std::ptr::null_mut()),
            ERR_NULL
        );

        assert_eq!(
            call(q, k, v, n, 6, 4, d, stride, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, n, hq, hk, d, stride - 1, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, n, hq, hk, d, stride, stride - 1, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, 0, hq, hk, d, stride, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, n, 0, hk, d, stride, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, n, hq, 0, d, stride, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(q, k, v, n, hq, hk, 0, stride, stride, out.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(
                q,
                k,
                v,
                usize::MAX,
                hq,
                hk,
                d,
                usize::MAX,
                usize::MAX,
                out.as_mut_ptr()
            ),
            ERR_DOMAIN
        );

        // the scale multiplies every logit: a NaN would poison the whole softmax
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                btb_attn_decode_bf16(
                    q,
                    k,
                    v,
                    n,
                    hq,
                    hk,
                    d,
                    stride,
                    stride,
                    scale,
                    out.as_mut_ptr(),
                    0
                ),
                ERR_DOMAIN,
                "scale {scale} should be refused"
            );
            assert_eq!(
                btb_attn_decode_f32(
                    q,
                    keyf.as_ptr(),
                    valuef.as_ptr(),
                    n,
                    hq,
                    hk,
                    d,
                    stride,
                    stride,
                    scale,
                    out.as_mut_ptr(),
                    0
                ),
                ERR_DOMAIN,
                "scale {scale} should be refused on the f32 cache"
            );
        }

        assert_eq!(
            btb_attn_decode_f32(
                std::ptr::null(),
                keyf.as_ptr(),
                valuef.as_ptr(),
                n,
                hq,
                hk,
                d,
                stride,
                stride,
                0.125,
                out.as_mut_ptr(),
                0
            ),
            ERR_NULL
        );
        assert_eq!(
            btb_attn_decode_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                n,
                6,
                4,
                d,
                stride,
                stride,
                0.125,
                out.as_mut_ptr(),
                0
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_attn_decode_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                0,
                hq,
                hk,
                d,
                stride,
                stride,
                0.125,
                out.as_mut_ptr(),
                0
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_attn_decode_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                n,
                hq,
                hk,
                d,
                stride - 1,
                stride,
                0.125,
                out.as_mut_ptr(),
                0
            ),
            ERR_DOMAIN
        );
    }

    assert_eq!(
        out,
        vec![0.0f32; hq * d],
        "a refused call must not write output"
    );
}

/// Reports the read bandwidth of a long bf16 cache. Ignored by default: it is a measurement, not
/// a check.
#[test]
#[ignore]
fn throughput_of_a_long_bf16_cache() {
    let (n, hq, hk, d) = (131072usize, 32usize, 8usize, 128usize);
    let reps = 36;
    let stride = n * d;
    let scale = 1.0 / (d as f32).sqrt();
    let query = gen_f32(hq * d, 0xBEA7);
    let key = Cache {
        bits: gen_bits(hk, n, d, stride, 0x1),
        values: Vec::new(),
    };
    let value = Cache {
        bits: gen_bits(hk, n, d, stride, 0x2),
        values: Vec::new(),
    };

    let warm = decode_bf16(&query, &key, &value, n, hq, hk, d, stride, stride, scale, 0);
    assert!(warm.iter().all(|v| v.is_finite()));

    let start = Instant::now();
    for _ in 0..reps {
        let out = decode_bf16(&query, &key, &value, n, hq, hk, d, stride, stride, scale, 0);
        std::hint::black_box(&out);
    }
    let seconds = start.elapsed().as_secs_f64();
    let bytes = 2.0 * (hk * n * d) as f64 * 2.0 * reps as f64;
    println!(
        "n {n} hq {hq} hk {hk} d {d}: {reps} calls in {seconds:.3} s = \
         {:.3} s/call, {:.1} GB/s read",
        seconds / reps as f64,
        bytes / seconds / 1e9
    );
}
