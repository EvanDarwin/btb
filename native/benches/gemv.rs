// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Per-op benches for the fused mat-vec family: the bf16 matrix, the 12-bit packed matrix, the GGUF
//! quants, the two MXFP4 layouts (checkpoint blocks/scales and the ggml 17-byte block), the FP8 matrix,
//! and the grouped dispatches (one thread-pool barrier for a whole layer's active experts).
//!
//! Inputs are built with the same generators the parity tests use (`tests/common`, included by
//! path so core is untouched), in plain `Vec`s rather than the guard tests' fenced `Fence` buffers.
//!
//! Each group sweeps the batch width (token rows) b in {1, 4, 16} at one realistic hidden width
//! (rows = cols = 5120), the decode / small-draft regime. Threads is pinned to 1 so a run measures
//! one ISA tier's single-core kernel; the tier is read from `isa()` and tagged into every id.
//!
//! ISA note: `isa()` caches its choice in a `OnceLock` on the first kernel call, and `BTB_NATIVE_ISA`
//! is read there once, so a single process runs exactly one tier. To see per-tier numbers, run the
//! bench once per tier, e.g. `BTB_NATIVE_ISA=scalar cargo bench --bench gemv` then the default
//! (neon on arm64, avx2/avx-512 on x86). Runtime switching inside one process is not possible.
//! TODO(threads): add a rayon-many variant (threads = 0) once the single-core tier sweep is banked.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use btb_native::{
    btb_gemv_bf16_group, btb_gemv_bf16_rows, btb_gemv_fp8_group, btb_gemv_fp8_rows,
    btb_gemv_mxfp4_ggml_rows, btb_gemv_mxfp4_group, btb_gemv_mxfp4_rows, btb_gemv_p12_rows,
};

#[path = "../tests/common/mod.rs"]
mod common;
use common::quant;
use common::refs::{
    gen_f32, gen_fp8, gen_mxfp4, gen_w, gen_w_palette, pack_bf16, Fp8, Mxfp4, MX_BLOCK,
    MX_BLOCK_BYTES,
};

/// The batch widths the plan sweeps: decode (1) and two small drafter/prefill widths.
const BATCHES: [usize; 3] = [1, 4, 16];
/// One realistic hidden width: a 5120-wide square weight, the host-tier decode shape.
const ROWS: usize = 5120;
const COLS: usize = 5120;

fn tier() -> String {
    format!("{:?}", btb_native::gemv::isa()).to_lowercase()
}

fn bench_bf16(c: &mut Criterion) {
    let isa = tier();
    let w = gen_w(ROWS * COLS, 0x9ACED);
    let mut g = c.benchmark_group("gemv_bf16");
    for &b in &BATCHES {
        let x = gen_f32(b * COLS, 0x515);
        let mut y = vec![0f32; b * ROWS];
        g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
            bch.iter(|| unsafe {
                btb_gemv_bf16_rows(
                    black_box(w.as_ptr()),
                    ROWS,
                    COLS,
                    black_box(x.as_ptr()),
                    b,
                    y.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

fn bench_p12(c: &mut Criterion) {
    let isa = tier();
    // A palette with rare escapes: the packed path's common branch plus the escape fixup.
    let raw = gen_w_palette(ROWS * COLS, 0x9ACED, 0.01);
    let p = pack_bf16(&raw);
    let mut g = c.benchmark_group("gemv_p12");
    for &b in &BATCHES {
        let x = gen_f32(b * COLS, 0x515);
        let mut y = vec![0f32; b * ROWS];
        g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
            bch.iter(|| unsafe {
                btb_gemv_p12_rows(
                    black_box(p.lo.as_ptr()),
                    black_box(p.hi4.as_ptr()),
                    p.table.as_ptr(),
                    p.esc_idx.as_ptr(),
                    p.esc_val.as_ptr(),
                    p.esc_idx.len(),
                    ROWS,
                    COLS,
                    black_box(x.as_ptr()),
                    b,
                    y.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

fn bench_mxfp4(c: &mut Criterion) {
    let isa = tier();
    let m = gen_mxfp4(ROWS, COLS, 0x9ACED);
    let mut g = c.benchmark_group("gemv_mxfp4");
    for &b in &BATCHES {
        let x = gen_f32(b * COLS, 0x515);
        let mut y = vec![0f32; b * ROWS];
        g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
            bch.iter(|| unsafe {
                btb_gemv_mxfp4_rows(
                    black_box(m.blocks.as_ptr()),
                    black_box(m.scales.as_ptr()),
                    ROWS,
                    COLS,
                    black_box(x.as_ptr()),
                    b,
                    y.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// The ggml MXFP4 layout is a GGUF's 17-byte block (scale byte then 16 nibble-packed weight bytes).
/// refs has no generator for it, so build the raw block bytes here: the timing does not depend on
/// the exact fp4 codes, only the block count and stride.
fn ggml_raw(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let groups = rows * cols / MX_BLOCK;
    let n = groups * (MX_BLOCK_BYTES + 1);
    let m = gen_mxfp4(rows, cols, seed); // reuse the RNG-backed blocks/scales as byte source
    let mut raw = Vec::with_capacity(n);
    for g in 0..groups {
        raw.push(m.scales[g]);
        raw.extend_from_slice(&m.blocks[g * MX_BLOCK_BYTES..(g + 1) * MX_BLOCK_BYTES]);
    }
    debug_assert_eq!(raw.len(), n);
    raw
}

fn bench_mxfp4_ggml(c: &mut Criterion) {
    let isa = tier();
    let raw = ggml_raw(ROWS, COLS, 0x9ACED);
    let mut g = c.benchmark_group("gemv_mxfp4_ggml");
    for &b in &BATCHES {
        let x = gen_f32(b * COLS, 0x515);
        let mut y = vec![0f32; b * ROWS];
        g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
            bch.iter(|| unsafe {
                btb_gemv_mxfp4_ggml_rows(
                    black_box(raw.as_ptr()),
                    ROWS,
                    COLS,
                    black_box(x.as_ptr()),
                    b,
                    y.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// One layer's active experts under a single dispatch: N equal tasks, each a bf16 matvec.
fn bench_bf16_group(c: &mut Criterion) {
    let isa = tier();
    const NT: usize = 8; // active experts in a layer
    const R: usize = 2048;
    const K: usize = 5120;
    let ws: Vec<Vec<u16>> = (0..NT).map(|t| gen_w(R * K, 0x6809 + t as u64)).collect();
    let mut g = c.benchmark_group("gemv_bf16_group");
    for &b in &BATCHES {
        let xs: Vec<Vec<f32>> = (0..NT).map(|t| gen_f32(b * K, 0x1465 + t as u64)).collect();
        let mut ys: Vec<Vec<f32>> = (0..NT).map(|_| vec![0f32; b * R]).collect();
        let wp: Vec<*const u16> = ws.iter().map(|w| w.as_ptr()).collect();
        let xp: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
        let rows = [R; NT];
        let cols = [K; NT];
        let bs = [b; NT];
        g.throughput(Throughput::Elements((NT * b * R * K) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, _| {
            let mut yp: Vec<*mut f32> = ys.iter_mut().map(|y| y.as_mut_ptr()).collect();
            bch.iter(|| unsafe {
                btb_gemv_bf16_group(
                    NT,
                    black_box(wp.as_ptr()),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    black_box(xp.as_ptr()),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// The MXFP4 grouped dispatch (the gpt-oss expert layer), checkpoint blocks/scales layout.
fn bench_mxfp4_group(c: &mut Criterion) {
    let isa = tier();
    const NT: usize = 8;
    const R: usize = 2048;
    const K: usize = 5120;
    let ms: Vec<Mxfp4> = (0..NT)
        .map(|t| gen_mxfp4(R, K, 0x6809 + t as u64))
        .collect();
    let mut g = c.benchmark_group("gemv_mxfp4_group");
    for &b in &BATCHES {
        let xs: Vec<Vec<f32>> = (0..NT).map(|t| gen_f32(b * K, 0x1465 + t as u64)).collect();
        let mut ys: Vec<Vec<f32>> = (0..NT).map(|_| vec![0f32; b * R]).collect();
        let blocks: Vec<*const u8> = ms.iter().map(|m| m.blocks.as_ptr()).collect();
        let scales: Vec<*const u8> = ms.iter().map(|m| m.scales.as_ptr()).collect();
        let xp: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
        let rows = [R; NT];
        let cols = [K; NT];
        let bs = [b; NT];
        g.throughput(Throughput::Elements((NT * b * R * K) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, _| {
            let mut yp: Vec<*mut f32> = ys.iter_mut().map(|y| y.as_mut_ptr()).collect();
            bch.iter(|| unsafe {
                btb_gemv_mxfp4_group(
                    NT,
                    black_box(blocks.as_ptr()),
                    black_box(scales.as_ptr()),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    black_box(xp.as_ptr()),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// The FP8 matrix at the hidden width, on 128x128 blocks as fine-grained FP8 checkpoints store it.
fn bench_fp8(c: &mut Criterion) {
    let isa = tier();
    let m = gen_fp8(ROWS, COLS, ROWS / 128, COLS / 128, 0x9ACED);
    let mut g = c.benchmark_group("gemv_fp8");
    for &b in &BATCHES {
        let x = gen_f32(b * COLS, 0x515);
        let mut y = vec![0f32; b * ROWS];
        g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
            bch.iter(|| unsafe {
                btb_gemv_fp8_rows(
                    black_box(m.w.as_ptr()),
                    black_box(m.scales.as_ptr()),
                    ROWS,
                    COLS,
                    m.sr,
                    m.sc,
                    black_box(x.as_ptr()),
                    b,
                    y.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// The FP8 grouped dispatch (an FP8 MoE layer's active experts).
fn bench_fp8_group(c: &mut Criterion) {
    let isa = tier();
    const NT: usize = 8;
    const R: usize = 2048;
    const K: usize = 5120;
    let ms: Vec<Fp8> = (0..NT)
        .map(|t| gen_fp8(R, K, R / 128, K / 128, 0x6809 + t as u64))
        .collect();
    let mut g = c.benchmark_group("gemv_fp8_group");
    for &b in &BATCHES {
        let xs: Vec<Vec<f32>> = (0..NT).map(|t| gen_f32(b * K, 0x1465 + t as u64)).collect();
        let mut ys: Vec<Vec<f32>> = (0..NT).map(|_| vec![0f32; b * R]).collect();
        let ws: Vec<*const u8> = ms.iter().map(|m| m.w.as_ptr()).collect();
        let scales: Vec<*const f32> = ms.iter().map(|m| m.scales.as_ptr()).collect();
        let xp: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
        let rows = [R; NT];
        let cols = [K; NT];
        let srs = [R / 128; NT];
        let scs = [K / 128; NT];
        let bs = [b; NT];
        g.throughput(Throughput::Elements((NT * b * R * K) as u64));
        g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, _| {
            let mut yp: Vec<*mut f32> = ys.iter_mut().map(|y| y.as_mut_ptr()).collect();
            bch.iter(|| unsafe {
                btb_gemv_fp8_group(
                    NT,
                    black_box(ws.as_ptr()),
                    black_box(scales.as_ptr()),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    srs.as_ptr(),
                    scs.as_ptr(),
                    black_box(xp.as_ptr()),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    1,
                )
            });
        });
    }
    g.finish();
}

/// Every GGUF quant matvec as stored, a group per format (`gemv_q4_k`, ...), at a decode step and a verify
/// pass only: 17 formats at every width of [`BATCHES`] would triple the job's bench time.
fn bench_quant(c: &mut Criterion) {
    let isa = tier();
    for f in quant::FORMATS {
        let raw = quant::blocks(f, ROWS, COLS, 0x9ACED);
        let t = quant::tables(f, 0x9ACED);
        let mut g = c.benchmark_group(format!("gemv_{}", f.name.to_lowercase()));
        for &b in &[1usize, 16] {
            let x = gen_f32(b * COLS, 0x515);
            let mut y = vec![0f32; b * ROWS];
            g.throughput(Throughput::Elements((b * ROWS * COLS) as u64));
            g.bench_with_input(BenchmarkId::new(&isa, b), &b, |bch, &b| {
                bch.iter(|| unsafe {
                    quant::call_raw(
                        f,
                        black_box(raw.as_ptr()),
                        t.grid.as_ptr(),
                        quant::ksigns_ptr(&t),
                        ROWS,
                        COLS,
                        black_box(x.as_ptr()),
                        b,
                        y.as_mut_ptr(),
                        1,
                    )
                });
            });
        }
        g.finish();
    }
}

criterion_group!(
    gemv,
    bench_bf16,
    bench_p12,
    bench_mxfp4,
    bench_mxfp4_ggml,
    bench_bf16_group,
    bench_mxfp4_group,
    bench_fp8,
    bench_fp8_group,
    bench_quant
);
criterion_main!(gemv);
