// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Per-op bench for the unbuffered direct reader: `btb_read_direct` (open-read-close per call) and
//! `btb_read_at` (a persistent handle). The temp-file fixture is the parity test's
//! (`tests/common/fixture.rs`, included by path), which is cross-platform.
//!
//! The reader is I/O, not a compute kernel: it has no ISA tier and no token-batch axis, so the
//! plan's rows {1, 4, 16} sweep is read as span sizes {1, 4, 16} MiB at a fixed offset. Threads /
//! worker depth default (0). NOTE: running this bench does real unbuffered disk reads that bypass
//! the page cache, so it is slow and touches the drive; `cargo bench --no-run` only compiles it.
//! TODO(depth): add a worker-depth sweep (1 vs many) once the span sweep is banked.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use btb_native::codes::OK;
use btb_native::{btb_close, btb_read_at, btb_read_direct};

#[path = "../tests/common/fixture.rs"]
mod fixture;
use fixture::Fixture;

const MIB: u64 = 1 << 20;
/// Span sizes read at offset 0, mapping the plan's {1, 4, 16} onto MiB.
const SPANS_MIB: [u64; 3] = [1, 4, 16];
/// A file large enough to hold the widest span with room to spare.
const FILE_LEN: u64 = 32 * MIB + 4096;

fn bench_read_direct(c: &mut Criterion) {
    let f = Fixture::new("bench_read_direct", FILE_LEN);
    let mut g = c.benchmark_group("direct_read");
    for &mib in &SPANS_MIB {
        let len = mib * MIB;
        let mut dst = vec![0u8; len as usize];
        g.throughput(Throughput::Bytes(len));
        g.bench_with_input(BenchmarkId::from_parameter(mib), &len, |bch, &len| {
            bch.iter(|| {
                let rc = unsafe {
                    btb_read_direct(black_box(f.wide.as_ptr()), 0, len, dst.as_mut_ptr(), 0)
                };
                assert_eq!(rc, OK, "btb_read_direct rc {rc}");
            });
        });
    }
    g.finish();
}

fn bench_read_at(c: &mut Criterion) {
    let f = Fixture::new("bench_read_at", FILE_LEN);
    let handle = f.open();
    let mut g = c.benchmark_group("direct_read_at");
    for &mib in &SPANS_MIB {
        let len = mib * MIB;
        let mut dst = vec![0u8; len as usize];
        g.throughput(Throughput::Bytes(len));
        g.bench_with_input(BenchmarkId::from_parameter(mib), &len, |bch, &len| {
            bch.iter(|| {
                let rc = unsafe { btb_read_at(handle, 0, len, dst.as_mut_ptr(), 0, 0) };
                assert_eq!(rc, OK, "btb_read_at rc {rc}");
            });
        });
    }
    g.finish();
    btb_close(handle);
}

criterion_group!(direct, bench_read_direct, bench_read_at);
criterion_main!(direct);
