// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Per-op bench for the fused decode-step grouped-query attention, both cache dtypes (bf16 and
//! f32). Inputs use the parity test's generators (`tests/common/refs.rs`, included by path) at the
//! test's real head geometry (hq 24, hk 4, d 256).
//!
//! Attention's decode step has no token-batch axis (it is one query position), so the plan's
//! rows {1, 4, 16} sweep does not apply; the cost scales with the kv length n instead, so the
//! bench sweeps n over a realistic context set. Threads is pinned to 1 (one ISA tier, single core);
//! the tier is read from `isa()` and tagged into every id. See `benches/gemv.rs` for the ISA/OnceLock
//! note and how to run each tier.
//! TODO(threads): add a rayon-many variant (attention alone is not bit-identical across threads).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use btb_native::{btb_attn_decode_bf16, btb_attn_decode_f32, btb_attn_spans_f32};

#[path = "../tests/common/refs.rs"]
mod refs;
use refs::{gen_f32, gen_w, Rng};

const HQ: usize = 24;
const HK: usize = 4;
const D: usize = 256;
/// Realistic decode context lengths (kv rows already in cache).
const NS: [usize; 4] = [128, 512, 2048, 8192];

fn tier() -> String {
    format!("{:?}", btb_native::gemv::isa()).to_lowercase()
}

fn bench_bf16(c: &mut Criterion) {
    let isa = tier();
    let q = gen_f32(HQ * D, 0xA77E4);
    let mut out = vec![0f32; HQ * D];
    let scale = 1.0 / (D as f32).sqrt();
    let mut g = c.benchmark_group("attn_decode_bf16");
    for &n in &NS {
        let stride = n * D; // tight: row r of head h at h*stride + r*d
        let k = gen_w(HK * stride, 0xC0FFEE);
        let v = gen_w(HK * stride, 0xBEEF);
        g.bench_with_input(BenchmarkId::new(&isa, n), &n, |bch, &n| {
            bch.iter(|| unsafe {
                btb_attn_decode_bf16(
                    black_box(q.as_ptr()),
                    black_box(k.as_ptr()),
                    black_box(v.as_ptr()),
                    n,
                    HQ,
                    HK,
                    D,
                    stride,
                    stride,
                    scale,
                    out.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

fn bench_f32(c: &mut Criterion) {
    let isa = tier();
    let q = gen_f32(HQ * D, 0xA77E4);
    let mut out = vec![0f32; HQ * D];
    let scale = 1.0 / (D as f32).sqrt();
    let mut g = c.benchmark_group("attn_decode_f32");
    for &n in &NS {
        let stride = n * D;
        let k = gen_f32(HK * stride, 0xC0FFEE);
        let v = gen_f32(HK * stride, 0xBEEF);
        g.bench_with_input(BenchmarkId::new(&isa, n), &n, |bch, &n| {
            bch.iter(|| unsafe {
                btb_attn_decode_f32(
                    black_box(q.as_ptr()),
                    black_box(k.as_ptr()),
                    black_box(v.as_ptr()),
                    n,
                    HQ,
                    HK,
                    D,
                    stride,
                    stride,
                    scale,
                    out.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// The rows of a prompt chunk a span call reads at once: one query row per prompt position.
const CHUNK: usize = 64;

/// A chunk of a prompt over a paged conversation's rows: `CHUNK` query rows after `n` committed ones, each
/// over the rows before it and itself, through a map scattering the positions over the pool as a paged
/// table does - the host's prefill of a prompt past its first chunk. f32, the host's cache dtype.
fn bench_spans_f32(c: &mut Criterion) {
    let isa = tier();
    let q = gen_f32(CHUNK * HQ * D, 0xA77E4);
    let mut out = vec![0f32; CHUNK * HQ * D];
    let scale = 1.0 / (D as f32).sqrt();
    let mut g = c.benchmark_group("attn_spans_f32");
    for &n in &NS {
        let rows = n + CHUNK;
        let stride = rows * D;
        let k = gen_f32(HK * stride, 0xC0FFEE);
        let v = gen_f32(HK * stride, 0xBEEF);
        let mut rng = Rng::new(0x3A9);
        let mut map: Vec<u32> = (0..rows as u32).collect();
        for i in (1..map.len()).rev() {
            map.swap(i, rng.below(i + 1));
        }
        let ends: Vec<u32> = (0..CHUNK).map(|p| (n + p + 1) as u32).collect();
        let starts = vec![0u32; CHUNK];
        g.bench_with_input(BenchmarkId::new(&isa, n), &n, |bch, _| {
            bch.iter(|| unsafe {
                btb_attn_spans_f32(
                    black_box(q.as_ptr()),
                    black_box(k.as_ptr()),
                    black_box(v.as_ptr()),
                    map.as_ptr(),
                    rows,
                    starts.as_ptr(),
                    ends.as_ptr(),
                    CHUNK,
                    rows,
                    HQ,
                    HK,
                    D,
                    stride,
                    stride,
                    scale,
                    out.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

criterion_group!(attn, bench_bf16, bench_f32, bench_spans_f32);
criterion_main!(attn);
