// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The token sampler through the C ABI on plain heap buffers: the greedy argmax is exactly the
//! lowest-index maximum at every shape and thread count, a (key, row) repeats, a batch picks what
//! its own single-row calls pick, and the top-k and top-p masks keep exactly the tokens a
//! straightforward f64 reference keeps. The tier is the one `isa()` selects (`BTB_NATIVE_ISA=scalar`
//! forces the scalar one); the NEON passes are held to the scalar ones bit for bit by the unit test
//! in src/sample.rs. The page-fenced version of the input read lives in `guard_sample.rs`.

use btb_native::btb_sample_pick;
use btb_native::codes::*;
use btb_native::gemv::MAX_THREADS;
use std::collections::HashSet;

#[path = "common/refs.rs"]
mod refs;

use refs::Rng;

/// One dispatch through the C ABI. `x` is `rows * v` logits row-major, `keys` one u64 per row.
fn run(x: &[f32], rows: usize, keys: &[u64], t: f32, k: u32, p: f32, threads: usize) -> Vec<u32> {
    let v = x.len() / rows;
    assert_eq!(x.len(), rows * v);
    assert_eq!(keys.len(), rows);
    let mut out = vec![u32::MAX; rows];
    let rc = unsafe {
        btb_sample_pick(
            x.as_ptr(),
            rows,
            v,
            keys.as_ptr(),
            t,
            k,
            p,
            out.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "btb_sample_pick returned {rc}");
    for (r, &o) in out.iter().enumerate() {
        assert!(
            (o as usize) < v,
            "row {r}: pick {o} is not a token index (v={v})"
        );
    }
    out
}

/// The lowest-index maximum of a row: the sampler's greedy pick, in plain f32 with the same
/// strictly-greater comparison, so a tie goes to the earlier token.
fn argmax(row: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi as u32
}

/// The set of tokens the sampler may draw: the top-k highest scaled logits (all of them when
/// `top_k` is 0 or at least the vocabulary), minus the ones more than 21 nats under the top, then
/// the shortest high-to-low prefix whose mass reaches `top_p` of the surviving mass. A clean f64
/// mirror of `pick_row`; kept honest by feeding it inputs with unambiguous boundaries.
fn kept_set(x: &[f32], t: f32, top_k: u32, top_p: f32) -> HashSet<u32> {
    let v = x.len();
    let inv_t = 1.0 / t as f64;
    let s: Vec<f64> = x.iter().map(|&a| a as f64 * inv_t).collect();
    let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
    let mut kept: Vec<usize> = if top_k > 0 && (top_k as usize) < v {
        order[..top_k as usize].to_vec()
    } else {
        order
    };
    kept.retain(|&i| s[i] - m >= -21.0);
    if top_p < 1.0 {
        let total: f64 = kept.iter().map(|&i| (s[i] - m).exp()).sum();
        let target = (total * top_p as f64).max(f64::MIN_POSITIVE);
        let mut acc = 0.0f64;
        let mut n = 0usize;
        for &i in &kept {
            acc += (s[i] - m).exp();
            n += 1;
            if acc >= target {
                break;
            }
        }
        kept.truncate(n);
    }
    kept.into_iter().map(|i| i as u32).collect()
}

/// The greedy pick (temperature 0, and any negative temperature) is exactly the lowest-index
/// maximum, over ragged shapes, values drawn from a few levels so ties are common, at one thread
/// and at every core.
#[test]
fn greedy_is_the_lowest_index_maximum() {
    let shapes: &[(usize, usize)] = &[
        (1, 1),
        (1, 2),
        (1, 7),
        (1, 128),
        (1, 50257),
        (3, 17),
        (5, 129),
        (16, 3001),
    ];
    for &(rows, v) in shapes {
        let mut rng = Rng::new(0x5A_11_CE ^ (v as u64).wrapping_mul(0x9E37_79B9));
        // a handful of distinct levels: ties at the maximum are the case that pins tie-breaking
        let x: Vec<f32> = (0..rows * v).map(|_| (rng.below(6) as f32) - 2.5).collect();
        let keys: Vec<u64> = (0..rows as u64).collect();
        let want: Vec<u32> = (0..rows).map(|r| argmax(&x[r * v..(r + 1) * v])).collect();
        for &t in &[0.0f32, -1.0, -1e30] {
            for &threads in &[1usize, 0] {
                assert_eq!(
                    run(&x, rows, &keys, t, 0, 1.0, threads),
                    want,
                    "{rows}x{v} t={t} threads={threads}"
                );
            }
        }
        eprintln!("greedy {rows:>3} x {v:<6}: lowest-index max at t<=0");
    }
}

/// NaNs never win the greedy pick, infinities do, a -0 and a +0 tie (the first wins), and a row of
/// nothing but -inf and NaN picks token 0, at widths either side of every vector boundary.
#[test]
fn greedy_handles_the_special_values() {
    let levels = [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
    ];
    let mut rng = Rng::new(0x0DD_BA11);
    for v in (1..=48).chain([255, 256, 257, 4099]) {
        for pool in [
            &levels[..],
            &levels[..2],
            &[f32::NAN, f32::NEG_INFINITY],
            &levels[4..6],
        ] {
            let x: Vec<f32> = (0..v).map(|_| pool[rng.below(pool.len())]).collect();
            let want = argmax(&x);
            for &threads in &[1usize, 0] {
                assert_eq!(
                    run(&x, 1, &[0], 0.0, 0, 1.0, threads)[0],
                    want,
                    "v={v} pool={pool:?} threads={threads}"
                );
            }
        }
    }
    eprintln!("greedy over NaN, infinities and signed zeros: the lowest-index maximum");
}

/// A (key, row) draws the same token every time and at every thread count, and a batch draws what
/// each row's own single-row call draws: the pick depends only on the row and its key, never on the
/// dispatch.
#[test]
fn a_key_repeats_and_a_batch_matches_its_solo_calls() {
    let (rows, v) = (16usize, 3000usize);
    let mut rng = Rng::new(0xDE7E_2312);
    let x: Vec<f32> = (0..rows * v).map(|_| rng.uni(-6.0, 6.0)).collect();
    let keys: Vec<u64> = (0..rows as u64).map(|r| r.wrapping_mul(0x12345)).collect();

    let a = run(&x, rows, &keys, 0.8, 40, 0.9, 0);
    let b = run(&x, rows, &keys, 0.8, 40, 0.9, 0);
    assert_eq!(a, b, "the same call gave different picks");
    let single = run(&x, rows, &keys, 0.8, 40, 0.9, 1);
    assert_eq!(a, single, "the thread count moved a pick");

    for r in 0..rows {
        let solo = run(&x[r * v..(r + 1) * v], 1, &[keys[r]], 0.8, 40, 0.9, 0);
        assert_eq!(solo[0], a[r], "row {r} alone differs from the batch");
    }
    eprintln!("{rows} rows: a key repeats, alone and in the batch, at 1 and all threads");
}

/// Over many keys the draws land only on tokens the reference keeps, and every kept token is drawn
/// at least once. A geometric distribution puts the top-k and top-p boundaries in clear gaps, so
/// the kept set is unambiguous: top-p 0.6 keeps two, 0.8 keeps three; top-k keeps its k.
#[test]
fn the_masks_keep_exactly_the_reference_set() {
    // p_i = 0.5 * 2^-i: cumulative 0.5, 0.75, 0.875, 0.9375, ...
    let x: Vec<f32> = (0..8).map(|i| (0.5f32 / 2f32.powi(i)).ln()).collect();
    let cases: &[(f32, u32, f32)] = &[
        (1.0, 0, 0.6), // nucleus: {0, 1}
        (1.0, 0, 0.8), // nucleus: {0, 1, 2}
        (1.0, 2, 1.0), // top-k: {0, 1}
        (1.0, 4, 1.0), // top-k: {0, 1, 2, 3}
        (1.0, 5, 0.8), // top-k then nucleus: {0, 1, 2}
        (0.7, 0, 0.8), // a temperature sharpens but keeps the same clean gaps
    ];
    for &(t, k, p) in cases {
        let want = kept_set(&x, t, k, p);
        let mut seen = HashSet::new();
        for key in 0..2000u64 {
            let pick = run(&x, 1, &[key], t, k, p, 1)[0];
            assert!(
                want.contains(&pick),
                "t={t} k={k} p={p}: drew {pick}, outside the kept set {want:?}"
            );
            seen.insert(pick);
        }
        assert_eq!(
            seen, want,
            "t={t} k={k} p={p}: the drawn set is not the kept set"
        );
        eprintln!("mask t={t} k={k} p={p}: kept {want:?}, all drawn");
    }
}

/// The draws follow the softmax of the logits: a fixed distribution's empirical frequencies match
/// it to within the sampling noise of twenty thousand keys.
#[test]
fn the_draws_follow_the_distribution() {
    let p = [0.5f32, 0.3, 0.15, 0.05];
    let x: Vec<f32> = p.iter().map(|q| q.ln()).collect();
    let n = 20000u64;
    let mut counts = [0u32; 4];
    for key in 0..n {
        counts[run(&x, 1, &[key], 1.0, 0, 1.0, 1)[0] as usize] += 1;
    }
    for (i, q) in p.iter().enumerate() {
        let f = counts[i] as f32 / n as f32;
        assert!((f - q).abs() < 0.02, "token {i}: {f} vs {q}");
    }
    eprintln!("20000 keys: frequencies within 0.02 of {p:?}");
}

/// A null pointer, a zero row or vocabulary count, an overflowing size, a non-finite temperature or
/// top-p, and a thread count past the cap are each refused with a code, and a refused call leaves
/// the output buffer alone.
#[test]
fn null_and_bad_arguments_are_error_codes_not_guesses() {
    let x = [0.0f32; 8];
    let keys = [0u64; 2];
    let mut out = [7u32; 2];
    unsafe {
        assert_eq!(
            btb_sample_pick(
                std::ptr::null(),
                2,
                4,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_NULL
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                4,
                std::ptr::null(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_NULL
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                4,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                std::ptr::null_mut(),
                1
            ),
            ERR_NULL
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                0,
                4,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                0,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                usize::MAX,
                4,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                4,
                keys.as_ptr(),
                f32::NAN,
                0,
                1.0,
                out.as_mut_ptr(),
                1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                4,
                keys.as_ptr(),
                1.0,
                0,
                f32::INFINITY,
                out.as_mut_ptr(),
                1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_sample_pick(
                x.as_ptr(),
                2,
                4,
                keys.as_ptr(),
                1.0,
                0,
                1.0,
                out.as_mut_ptr(),
                MAX_THREADS + 1
            ),
            ERR_DOMAIN
        );
    }
    assert_eq!(out, [7u32; 2], "a refused call must not write output");
    eprintln!(
        "null pointers, bad shapes, non-finite params and an over-cap thread count all refused"
    );
}
