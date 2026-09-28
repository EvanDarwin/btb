// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Node-list attention through the C ABI on plain heap buffers, on the path this machine selects: each
//! query row over its own list of cache rows is the decode step over those rows, bit for bit (the checks
//! live in `common/attn_nodes.rs`, which `attn_nodes_scalar.rs` runs again on the scalar path), and a
//! malformed call is a code. The fenced sweep lives in `guard_attn_nodes.rs`.

use btb_native::codes::*;
use btb_native::gemv::isa;
use btb_native::{btb_attn_nodes_bf16, btb_attn_nodes_f32};

#[path = "common/refs.rs"]
mod refs;

#[path = "common/attn_nodes.rs"]
mod nodes;

/// A list of the rows `0..n` is the decode step over `n` rows, bit for bit.
#[test]
fn an_identity_list_is_the_decode_step() {
    eprintln!("isa {:?}", isa());
    nodes::check_identity("vector");
}

/// A batched call's rows are their own calls, their decode steps over the gathered rows, and
/// independent of their order in the batch, bit for bit; within tolerance of the f64 reference.
#[test]
fn each_row_of_a_batch_is_its_own_decode_step() {
    nodes::check_lists("vector", true);
}

/// A null pointer, a zero dimension, a head count that does not divide, a head stride shorter than the
/// vouched rows, an empty or backward list, a row past the cache, a misaligned list, a non-finite scale,
/// an overflowing query count and a thread count past the cap are each refused with a code, and a refused
/// call leaves the output alone.
#[test]
fn bad_lists_and_shapes_are_error_codes_not_guesses() {
    let (n_rows, hq, hk, d) = (8usize, 4usize, 2usize, 8usize);
    let t = 2usize;
    let query = vec![0.5f32; t * hq * d];
    let key = vec![0u16; hk * n_rows * d];
    let value = vec![0u16; hk * n_rows * d];
    let keyf = vec![0.5f32; hk * n_rows * d];
    let valuef = vec![0.5f32; hk * n_rows * d];
    let offs: Vec<u32> = vec![0, 3, 5];
    let idx: Vec<u32> = vec![0, 1, 2, 3, 7];
    let mut out = vec![0.0f32; t * hq * d];
    let stride = n_rows * d;
    let (q, k, v) = (query.as_ptr(), key.as_ptr(), value.as_ptr());
    let o = out.as_mut_ptr();

    #[allow(clippy::too_many_arguments)]
    unsafe fn call(
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
        ks: usize,
        vs: usize,
        scale: f32,
        o: *mut f32,
        threads: usize,
    ) -> i32 {
        btb_attn_nodes_bf16(
            q, k, v, offs, idx, t, n_rows, hq, hk, d, ks, vs, scale, o, threads,
        )
    }
    let (op, ip) = (offs.as_ptr(), idx.as_ptr());
    let null = std::ptr::null::<u32>();

    unsafe {
        // the well-formed call the refusals below are each one change from
        let mut probe = vec![0.0f32; t * hq * d];
        assert_eq!(
            call(
                q,
                k,
                v,
                op,
                ip,
                t,
                n_rows,
                hq,
                hk,
                d,
                stride,
                stride,
                0.125,
                probe.as_mut_ptr(),
                0
            ),
            OK
        );

        let s = stride;
        let cases: Vec<(&str, i32, i32)> = vec![
            (
                "null q",
                call(
                    std::ptr::null(),
                    k,
                    v,
                    op,
                    ip,
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
                "null k",
                call(
                    q,
                    std::ptr::null(),
                    v,
                    op,
                    ip,
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
                    op,
                    ip,
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
                "null offs",
                call(q, k, v, null, ip, t, n_rows, hq, hk, d, s, s, 0.125, o, 0),
                ERR_NULL,
            ),
            (
                "null idx",
                call(q, k, v, op, null, t, n_rows, hq, hk, d, s, s, 0.125, o, 0),
                ERR_NULL,
            ),
            (
                "null out",
                call(
                    q,
                    k,
                    v,
                    op,
                    ip,
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
                call(q, k, v, op, ip, 0, n_rows, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "no cache rows",
                call(q, k, v, op, ip, t, 0, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "hq 0",
                call(q, k, v, op, ip, t, n_rows, 0, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "hk 0",
                call(q, k, v, op, ip, t, n_rows, hq, 0, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "d 0",
                call(q, k, v, op, ip, t, n_rows, hq, hk, 0, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "hq not a multiple of hk",
                call(q, k, v, op, ip, t, n_rows, 6, 4, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "key stride short of the rows",
                call(q, k, v, op, ip, t, n_rows, hq, hk, d, s - 1, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "value stride short of the rows",
                call(q, k, v, op, ip, t, n_rows, hq, hk, d, s, s - 1, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "a row past the vouched rows",
                call(q, k, v, op, ip, t, 7, hq, hk, d, s, s, 0.125, o, 0),
                ERR_DOMAIN,
            ),
            (
                "query count overflowing",
                call(
                    q,
                    k,
                    v,
                    op,
                    ip,
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
                    op,
                    ip,
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
                    op,
                    ip,
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

        // lists: an empty one, one running backward, one entry past the cache, a misaligned array
        let empty: Vec<u32> = vec![0, 3, 3];
        let backward: Vec<u32> = vec![0, 4, 3];
        let past: Vec<u32> = vec![0, 1, 2, 3, 8];
        for (what, offs, idx) in [
            ("an empty list", &empty, &idx),
            ("a backward offset", &backward, &idx),
            ("an index past the cache", &offs, &past),
        ] {
            assert_eq!(
                call(
                    q,
                    k,
                    v,
                    offs.as_ptr(),
                    idx.as_ptr(),
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
        assert_eq!(
            call(q, k, v, skew, ip, t, n_rows, hq, hk, d, s, s, 0.125, o, 0),
            ERR_DOMAIN,
            "misaligned offsets"
        );
        assert_eq!(
            call(q, k, v, op, skew, t, n_rows, hq, hk, d, s, s, 0.125, o, 0),
            ERR_DOMAIN,
            "misaligned index"
        );

        // the scale multiplies every logit: a NaN would poison the whole softmax
        for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                call(q, k, v, op, ip, t, n_rows, hq, hk, d, s, s, scale, o, 0),
                ERR_DOMAIN,
                "scale {scale}"
            );
            assert_eq!(
                btb_attn_nodes_f32(
                    q,
                    keyf.as_ptr(),
                    valuef.as_ptr(),
                    op,
                    ip,
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
            btb_attn_nodes_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                op,
                null,
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
            "null idx on the f32 cache"
        );
        assert_eq!(
            btb_attn_nodes_f32(
                q,
                keyf.as_ptr(),
                valuef.as_ptr(),
                op,
                past.as_ptr(),
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
            "an index past the cache on the f32 cache"
        );
    }

    assert_eq!(
        out,
        vec![0.0f32; t * hq * d],
        "a refused call must not write output"
    );
}
