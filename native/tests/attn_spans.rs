// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Span attention through the C ABI on plain heap buffers, on the path this machine selects: each query row
//! over its span of one map of cache rows is, bit for bit, the node-list call over that span copied out as a
//! list and the decode step over its rows copied out - a prompt chunk over a paged conversation's scattered
//! rows, windows, a whole map as the decode step - and a malformed call is a code. The fenced sweep lives in
//! `guard_attn_spans.rs`.

use btb_native::codes::*;
use btb_native::gemv::isa;
use btb_native::{btb_attn_spans_bf16, btb_attn_spans_f32};

#[path = "common/refs.rs"]
mod refs;

#[path = "common/attn_nodes.rs"]
mod nodes;

#[path = "common/attn_spans.rs"]
mod spans;

use refs::{bits, gen_f32};

/// `(base, t, hq, hk, d, win)`: a chunk of `t` rows after `base` committed ones, over a pool with spare rows,
/// at Qwen4's QSA head, the 8-head GQA shape, a head dimension with a SIMD tail, and a group past 32 query
/// heads - each across the 64-row chunk the decode step splits by, two under a window.
const CHUNKS: &[(usize, usize, usize, usize, usize, usize)] = &[
    (0, 5, 24, 2, 256, 0),
    (1, 8, 24, 2, 256, 0),
    (40, 30, 24, 2, 256, 0),
    (700, 16, 24, 2, 256, 0),
    (3000, 4, 32, 8, 128, 0),
    (63, 70, 32, 8, 128, 0),
    (130, 70, 4, 2, 40, 32),
    (20, 30, 40, 1, 24, 16),
];

/// Each row of a chunk over a scattered map is its own node-list call over its span and the decode step over
/// its span's rows copied out, bit for bit, at every thread count and both element types; and a map of every
/// row with one span over all of it is the decode step over the cache itself.
#[test]
fn each_row_of_a_chunk_is_its_decode_step() {
    eprintln!("isa {:?}", isa());
    let mut calls = 0usize;
    for (s, &(base, t, hq, hk, d, win)) in CHUNKS.iter().enumerate() {
        let n = base + t;
        let n_rows = n + 9; // the pool's spare rows, which no span reaches
        let scale = 1.0 / (d as f32).sqrt();
        let k = nodes::cache(hk, n_rows, d, (n_rows + 2) * d, 0x5A11 + s as u64);
        let v = nodes::cache(hk, n_rows, d, n_rows * d, 0x5A12 + s as u64);
        let map = spans::scattered(n, n_rows, 0x3A9 + s as u64);
        let (starts, ends) = spans::chunk(base, t, win);
        let q = gen_f32(t * hq * d, 0x9E8 + s as u64);
        for bf16 in [false, true] {
            for &threads in nodes::THREADS {
                let ctx = format!(
                    "base={base} t={t} hq={hq} hk={hk} d={d} win={win} threads={threads} bf16={bf16}"
                );
                let all = spans::spans(
                    bf16, &q, &k, &v, &map, &starts, &ends, n_rows, hq, hk, d, scale, threads,
                );
                calls += 1;
                for i in 0..t {
                    let list = &map[starts[i] as usize..ends[i] as usize];
                    let row = &q[i * hq * d..(i + 1) * hq * d];
                    let (offs, idx) = nodes::flatten(&[list.to_vec()], 0);
                    let one = nodes::nodes(
                        bf16, row, &k, &v, &offs, &idx, n_rows, hq, hk, d, scale, threads,
                    );
                    let gk = nodes::gather(&k, hk, d, list, 1);
                    let gv = nodes::gather(&v, hk, d, list, 0);
                    let step =
                        nodes::decode(bf16, row, &gk, &gv, list.len(), hq, hk, d, scale, threads);
                    calls += 2;
                    let got = &all[i * hq * d..(i + 1) * hq * d];
                    assert_eq!(
                        bits(got),
                        bits(&one),
                        "{ctx}: row {i} against its node list"
                    );
                    assert_eq!(
                        bits(got),
                        bits(&step),
                        "{ctx}: row {i} against its decode step"
                    );
                }
            }
        }
        // the identity: every row, one span over all of it
        let whole: Vec<u32> = (0..n_rows as u32).collect();
        for bf16 in [false, true] {
            for &threads in nodes::THREADS {
                let got = spans::spans(
                    bf16,
                    &q[..hq * d],
                    &k,
                    &v,
                    &whole,
                    &[0],
                    &[n_rows as u32],
                    n_rows,
                    hq,
                    hk,
                    d,
                    scale,
                    threads,
                );
                let step = nodes::decode(
                    bf16,
                    &q[..hq * d],
                    &k,
                    &v,
                    n_rows,
                    hq,
                    hk,
                    d,
                    scale,
                    threads,
                );
                calls += 2;
                assert_eq!(
                    bits(&got),
                    bits(&step),
                    "identity base={base} t={t} threads={threads} bf16={bf16}"
                );
            }
        }
    }
    eprintln!(
        "[vector] spans: {calls} calls, every row its node list and its decode step bit for bit"
    );
}

/// A null pointer, a zero dimension or map, a head count that does not divide, a stride shorter than the
/// vouched rows, an empty or backward span, a span past the map, a map row past the cache, a misaligned array,
/// a non-finite scale and a thread count past the cap are each refused with a code, and a refused call leaves
/// the output alone.
#[test]
fn bad_spans_and_shapes_are_error_codes_not_guesses() {
    let (n_rows, hq, hk, d) = (8usize, 4usize, 2usize, 8usize);
    let t = 2usize;
    let query = vec![0.5f32; t * hq * d];
    let key = vec![0u16; hk * n_rows * d];
    let value = vec![0u16; hk * n_rows * d];
    let keyf = vec![0.5f32; hk * n_rows * d];
    let valuef = vec![0.5f32; hk * n_rows * d];
    let map: Vec<u32> = vec![7, 0, 3, 5, 1];
    let starts: Vec<u32> = vec![0, 2];
    let ends: Vec<u32> = vec![3, 5];
    let mut out = vec![0.0f32; t * hq * d];
    let s = n_rows * d;
    let (q, k, v) = (query.as_ptr(), key.as_ptr(), value.as_ptr());
    let o = out.as_mut_ptr();
    let (mp, sp, ep, nm) = (map.as_ptr(), starts.as_ptr(), ends.as_ptr(), map.len());
    let null = std::ptr::null::<u32>();

    #[allow(clippy::too_many_arguments)]
    unsafe fn call(
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
        ks: usize,
        vs: usize,
        scale: f32,
        o: *mut f32,
        threads: usize,
    ) -> i32 {
        btb_attn_spans_bf16(
            q, k, v, map, n_map, starts, ends, t, n_rows, hq, hk, d, ks, vs, scale, o, threads,
        )
    }

    unsafe {
        // the well-formed call the refusals below are each one change from
        let mut probe = vec![0.0f32; t * hq * d];
        let ok = call(
            q,
            k,
            v,
            mp,
            nm,
            sp,
            ep,
            t,
            n_rows,
            hq,
            hk,
            d,
            s,
            s,
            0.125,
            probe.as_mut_ptr(),
            0,
        );
        assert_eq!(ok, OK);

        let null_q = std::ptr::null();
        let cases: Vec<(&str, i32, i32)> = vec![
            (
                "null q",
                call(
                    null_q, k, v, mp, nm, sp, ep, t, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_NULL,
            ),
            (
                "null k",
                call(
                    q,
                    std::ptr::null(),
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    o,
                    0,
                ),
                ERR_NULL,
            ),
            (
                "null v",
                call(
                    q,
                    k,
                    std::ptr::null(),
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    o,
                    0,
                ),
                ERR_NULL,
            ),
            (
                "null map",
                call(
                    q, k, v, null, nm, sp, ep, t, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_NULL,
            ),
            (
                "null starts",
                call(
                    q, k, v, mp, nm, null, ep, t, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_NULL,
            ),
            (
                "null ends",
                call(
                    q, k, v, mp, nm, sp, null, t, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_NULL,
            ),
            (
                "null out",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    std::ptr::null_mut(),
                    0,
                ),
                ERR_NULL,
            ),
            (
                "no queries",
                call(
                    q, k, v, mp, nm, sp, ep, 0, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "an empty map",
                call(
                    q, k, v, mp, 0, sp, ep, t, n_rows, hq, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "no cache rows",
                call(q, k, v, mp, nm, sp, ep, t, 0, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "hq 0",
                call(
                    q, k, v, mp, nm, sp, ep, t, n_rows, 0, hk, d, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "hk 0",
                call(
                    q, k, v, mp, nm, sp, ep, t, n_rows, hq, 0, d, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "d 0",
                call(
                    q, k, v, mp, nm, sp, ep, t, n_rows, hq, hk, 0, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "hq not a multiple of hk",
                call(
                    q, k, v, mp, nm, sp, ep, t, n_rows, 6, 4, d, s, s, 0.125, o, 0,
                ),
                ERR_DOMAIN,
            ),
            (
                "key stride short of the rows",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s - 1,
                    s,
                    0.125,
                    o,
                    0,
                ),
                ERR_DOMAIN,
            ),
            (
                "value stride short of the rows",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s - 1,
                    0.125,
                    o,
                    0,
                ),
                ERR_DOMAIN,
            ),
            (
                "a map row past the vouched rows",
                call(q, k, v, mp, nm, sp, ep, t, 7, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "query count overflowing",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    usize::MAX / 2,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    o,
                    0,
                ),
                ERR_DOMAIN,
            ),
            (
                "rows overflowing",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    usize::MAX,
                    hq,
                    hk,
                    d,
                    usize::MAX,
                    usize::MAX,
                    0.125,
                    o,
                    0,
                ),
                ERR_DOMAIN,
            ),
            (
                "threads past the cap",
                call(
                    q,
                    k,
                    v,
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    o,
                    1 << 20,
                ),
                ERR_DOMAIN,
            ),
        ];
        for (what, got, want) in cases {
            assert_eq!(got, want, "{what}");
        }

        // spans: an empty one, one running backward, one past the map's end, and the map read only as far as
        // `n_map` says (a span within a shorter map than the spans reach is refused)
        let empty: Vec<u32> = vec![2, 2];
        let backward_s: Vec<u32> = vec![3, 2];
        let backward_e: Vec<u32> = vec![1, 5];
        let past: Vec<u32> = vec![3, 6];
        for (what, st, en, n_map) in [
            ("an empty span", &starts, &empty, nm),
            ("a backward span", &backward_s, &backward_e, nm),
            ("a span past the map", &starts, &past, nm),
            ("a span past a shorter map", &starts, &ends, 4),
        ] {
            assert_eq!(
                call(
                    q,
                    k,
                    v,
                    mp,
                    n_map,
                    st.as_ptr(),
                    en.as_ptr(),
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    0.125,
                    o,
                    0
                ),
                ERR_DOMAIN,
                "{what}"
            );
        }
        let wide = [0u32; 8];
        let skew = (wide.as_ptr() as *const u8).add(1) as *const u32;
        for (what, m, st, en) in [
            ("misaligned map", skew, sp, ep),
            ("misaligned starts", mp, skew, ep),
            ("misaligned ends", mp, sp, skew),
        ] {
            assert_eq!(
                call(q, k, v, m, nm, st, en, t, n_rows, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
                "{what}"
            );
        }

        // the scale multiplies every logit: a NaN would poison the whole softmax
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                call(q, k, v, mp, nm, sp, ep, t, n_rows, hq, hk, d, s, s, scale, o, 0),
                ERR_DOMAIN,
                "scale {scale}"
            );
            assert_eq!(
                btb_attn_spans_f32(
                    q,
                    keyf.as_ptr(),
                    valuef.as_ptr(),
                    mp,
                    nm,
                    sp,
                    ep,
                    t,
                    n_rows,
                    hq,
                    hk,
                    d,
                    s,
                    s,
                    scale,
                    o,
                    0
                ),
                ERR_DOMAIN,
                "scale {scale} on the f32 cache"
            );
        }
        assert_eq!(
            btb_attn_spans_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                null,
                nm,
                sp,
                ep,
                t,
                n_rows,
                hq,
                hk,
                d,
                s,
                s,
                0.125,
                o,
                0
            ),
            ERR_NULL,
            "null map on the f32 cache"
        );
    }

    assert_eq!(
        out,
        vec![0.0f32; t * hq * d],
        "a refused call must not write output"
    );
}
