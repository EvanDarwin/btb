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

use btb_native::{btb_attn_decode_bf16, btb_attn_decode_f32};

#[path = "../tests/common/refs.rs"]
mod refs;
use refs::{gen_f32, gen_w};

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

criterion_group!(attn, bench_bf16, bench_f32);
criterion_main!(attn);
