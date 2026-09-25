// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced attention decode at a grouped-query shape (hq 24, hk 4, d 256) for every KV length
//! 1..=1200. See `common/attn.rs` for the fences.

mod common;

use btb_native::gemv::isa;
use common::attn::*;

/// Every KV length against a fixed-capacity cache, whose rows past `n` hold data: a read past the
/// end is a wrong answer here rather than a fault.
#[test]
fn real_stride_every_length() {
    eprintln!("isa {:?}", isa());
    sweep("real-stride", |_| CAP * D, false, true);
}

/// Every KV length against a cache rebuilt at the tight stride, so the last row ends at the guard
/// page and a read past it faults.
#[test]
fn tight_stride_every_length() {
    sweep("tight-stride", |n| n * D, true, true);
}

/// The query and both caches come back byte for byte unchanged after a run of decode calls.
#[test]
fn inputs_are_bit_identical_after_the_calls() {
    digest_check(&[1, 0, 16]);
}

/// Every KV length at a fixed sixteen threads, the count that selects the dedicated pool and a
/// chunk of ceil(n / 8) rows instead of the shared one.
#[test]
fn sixteen_threads_every_length() {
    every_length_at("threads-16", 16);
}
