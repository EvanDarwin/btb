// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The span attention's helpers, shared by its parity test (`attn_spans.rs`) and its fenced sweep
//! (`guard_attn_spans.rs`): a scattered map of cache rows as a paged conversation's table is, the spans of a
//! prompt chunk over it, and one call. Both binaries include it with `#[path = "common/attn_spans.rs"] mod
//! spans;` beside `refs` and `nodes`, which it reads through `super`.
//!
//! `dead_code` is allowed because each binary uses only part of it.
#![allow(dead_code)]

use super::nodes::Cache;
use super::refs::Rng;
use btb_native::codes::OK;
use btb_native::{btb_attn_spans_bf16, btb_attn_spans_f32};

/// `n` distinct rows of a pool of `pool`, in a random order: a conversation's positions at the rows of the
/// pages it was given, as a paged table maps them.
pub fn scattered(n: usize, pool: usize, seed: u64) -> Vec<u32> {
    assert!(n <= pool);
    let mut rng = Rng::new(seed);
    let mut rows: Vec<u32> = (0..pool as u32).collect();
    for i in (1..rows.len()).rev() {
        rows.swap(i, rng.below(i + 1));
    }
    rows.truncate(n);
    rows
}

/// The spans of a chunk of `t` rows after `base`: row `p` over the rows before it and itself, the last `win`
/// of them under a window (0: every row).
pub fn chunk(base: usize, t: usize, win: usize) -> (Vec<u32>, Vec<u32>) {
    let ends: Vec<u32> = (0..t).map(|p| (base + p + 1) as u32).collect();
    let starts = ends
        .iter()
        .map(|&e| {
            if win == 0 {
                0
            } else {
                e.saturating_sub(win as u32)
            }
        })
        .collect();
    (starts, ends)
}

/// One span call: `q` holds `starts.len()` query rows, row `i` over `map[starts[i]..ends[i]]`.
#[allow(clippy::too_many_arguments)]
pub fn spans(
    bf16: bool,
    q: &[f32],
    k: &Cache,
    v: &Cache,
    map: &[u32],
    starts: &[u32],
    ends: &[u32],
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let t = starts.len();
    let mut out = vec![f32::NAN; t * hq * d];
    let rc = unsafe {
        if bf16 {
            btb_attn_spans_bf16(
                q.as_ptr(),
                k.bits.as_ptr(),
                v.bits.as_ptr(),
                map.as_ptr(),
                map.len(),
                starts.as_ptr(),
                ends.as_ptr(),
                t,
                n_rows,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        } else {
            btb_attn_spans_f32(
                q.as_ptr(),
                k.vals.as_ptr(),
                v.vals.as_ptr(),
                map.as_ptr(),
                map.len(),
                starts.as_ptr(),
                ends.as_ptr(),
                t,
                n_rows,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        }
    };
    assert_eq!(rc, OK, "btb_attn_spans returned {rc}");
    assert!(
        out.iter().all(|x| x.is_finite()),
        "a span row read a poisoned row or was never written"
    );
    out
}
