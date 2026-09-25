// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Per-op bench for the fused token sampler. Inputs use the parity test's generators
//! (`tests/common/refs.rs`, included by path). `sample_pick` has an explicit `rows` argument (one
//! row of logits per position picked in a tree/batch), so the plan's rows {1, 4, 16} sweep maps
//! directly; the width is a realistic vocabulary. Two regimes: greedy (temperature <= 0, the argmax
//! path) and a temperature draw with top-k/top-p (the sort-and-sample path). Threads is pinned to 1;
//! the tier is read from `isa()` and tagged into every id. See `benches/gemv.rs` for the ISA note.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use btb_native::btb_sample_pick;

#[path = "../tests/common/refs.rs"]
mod refs;
use refs::{gen_f32, Rng};

/// Rows of logits picked in one call (decode, and two small tree/draft widths).
const ROWS: [usize; 3] = [1, 4, 16];
/// A realistic vocabulary width.
const V: usize = 128_256;

fn tier() -> String {
    format!("{:?}", btb_native::gemv::isa()).to_lowercase()
}

fn keys(rows: usize, seed: u64) -> Vec<u64> {
    let mut r = Rng::new(seed);
    (0..rows).map(|_| r.next_u64()).collect()
}

/// `temperature <= 0`: the argmax over each row, no sort.
fn bench_greedy(c: &mut Criterion) {
    let isa = tier();
    let mut g = c.benchmark_group("sample_greedy");
    for &rows in &ROWS {
        let x = gen_f32(rows * V, 0x5A19);
        let ks = keys(rows, 0xC0DE);
        let mut out = vec![0u32; rows];
        g.bench_with_input(BenchmarkId::new(&isa, rows), &rows, |bch, &rows| {
            bch.iter(|| unsafe {
                btb_sample_pick(
                    black_box(x.as_ptr()),
                    rows,
                    V,
                    black_box(ks.as_ptr()),
                    0.0,
                    0,
                    1.0,
                    out.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// A temperature draw with top-k 50 and top-p 0.95: the sort-and-sample path.
fn bench_sampled(c: &mut Criterion) {
    let isa = tier();
    let mut g = c.benchmark_group("sample_topkp");
    for &rows in &ROWS {
        let x = gen_f32(rows * V, 0x5A19);
        let ks = keys(rows, 0xC0DE);
        let mut out = vec![0u32; rows];
        g.bench_with_input(BenchmarkId::new(&isa, rows), &rows, |bch, &rows| {
            bch.iter(|| unsafe {
                btb_sample_pick(
                    black_box(x.as_ptr()),
                    rows,
                    V,
                    black_box(ks.as_ptr()),
                    1.0,
                    50,
                    0.95,
                    out.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

criterion_group!(sample, bench_greedy, bench_sampled);
criterion_main!(sample);
