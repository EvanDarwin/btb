// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Per-op bench for the fused gated-DeltaNet step. Inputs use the parity test's generator
//! (`tests/common/refs.rs`, included by path) at the test's real head geometry
//! (hk 16, hv 48, dk 128, dv 128, K 4).
//!
//! The step advances one position for every head, so there is no token-batch axis; the plan's
//! rows {1, 4, 16} sweep does not apply, and the bench measures the single real shape with and
//! without a conv bias (the two branches the parity test covers). `mixed_qkv`, `conv_state` and
//! `state` are updated in place, so every call gets them fresh from the generated inputs, copied
//! outside the timing: fed its own output, the conv overflows to inf within ~200 calls and the
//! state fills with subnormals, whose cost grows with the iteration count and swung the bench by
//! 25-40% between identical kernels. Threads is pinned to 1; the tier is read from `isa()` and
//! tagged into every id. See `benches/gemv.rs` for the ISA/OnceLock note.

use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use btb_native::btb_delta_step;

#[path = "../tests/common/refs.rs"]
mod refs;
use refs::{gen_delta, DeltaIn, DeltaShape};

const S: DeltaShape = DeltaShape {
    hk: 16,
    hv: 48,
    dk: 128,
    dv: 128,
    k: 4,
};
const EPS: f32 = 1e-6;

fn tier() -> String {
    format!("{:?}", btb_native::gemv::isa()).to_lowercase()
}

fn bench_delta(c: &mut Criterion) {
    let isa = tier();
    let mut g = c.benchmark_group("delta_step");
    for (bias, seed) in [(false, 11u64), (true, 22u64)] {
        let n = gen_delta(&S, seed, bias);
        // In-place buffers, refilled from the generated inputs before every call.
        let mut mixed = n.mixed.clone();
        let mut conv_state = n.conv_state.clone();
        let mut state = n.state.clone();
        let mut out = vec![0f32; S.hv * S.dv];
        let conv_b_ptr = n.conv_b.as_ref().map_or(std::ptr::null(), |b| b.as_ptr());
        let label = if bias { "bias" } else { "nobias" };
        g.bench_with_input(BenchmarkId::new(&isa, label), &bias, |bch, _| {
            bch.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    mixed.copy_from_slice(&n.mixed);
                    conv_state.copy_from_slice(&n.conv_state);
                    state.copy_from_slice(&n.state);
                    let t = Instant::now();
                    black_box(step(
                        &n,
                        conv_b_ptr,
                        &mut mixed,
                        &mut conv_state,
                        &mut state,
                        &mut out,
                    ));
                    total += t.elapsed();
                }
                total
            });
        });
    }
    g.finish();
}

fn step(
    n: &DeltaIn,
    conv_b_ptr: *const f32,
    mixed: &mut [f32],
    conv_state: &mut [f32],
    state: &mut [f32],
    out: &mut [f32],
) -> i32 {
    unsafe {
        btb_delta_step(
            mixed.as_mut_ptr(),
            conv_state.as_mut_ptr(),
            black_box(n.conv_w.as_ptr()),
            conv_b_ptr,
            S.c(),
            S.k,
            black_box(n.z.as_ptr()),
            n.a.as_ptr(),
            n.b.as_ptr(),
            n.a_log.as_ptr(),
            n.dt_bias.as_ptr(),
            state.as_mut_ptr(),
            S.hk,
            S.hv,
            S.dk,
            S.dv,
            n.norm_w.as_ptr(),
            EPS,
            out.as_mut_ptr(),
            1,
        )
    }
}

criterion_group!(delta, bench_delta);
criterion_main!(delta);
