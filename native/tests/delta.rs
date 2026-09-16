// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The gated DeltaNet step through the C ABI on plain heap buffers: the thread count does not move
//! a bit, a second step carries the states forward correctly, and a malformed shape or parameter
//! is refused. The fenced sweeps over the same kernel live in `guard_delta.rs`.

use btb_native::btb_delta_step;
use btb_native::codes::*;

#[path = "common/refs.rs"]
mod refs;

use refs::{bits, delta_reference, err_pair, gen_delta, DeltaIn, DeltaShape};

/// The full head shape: 16 key heads, 48 value heads, 128 lanes each, a 4-tap conv.
const REAL: DeltaShape = DeltaShape {
    hk: 16,
    hv: 48,
    dk: 128,
    dv: 128,
    k: 4,
};
const EPS: f32 = 1e-6;
const TOL: f64 = 1e-5;

/// One step in place: the conv output, the conv state, the recurrent state and the output.
fn run(s: &DeltaShape, n: &DeltaIn, threads: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut mixed = n.mixed.clone();
    let mut cs = n.conv_state.clone();
    let mut st = n.state.clone();
    let mut out = vec![f32::NAN; s.hv * s.dv];
    let code = unsafe {
        btb_delta_step(
            mixed.as_mut_ptr(),
            cs.as_mut_ptr(),
            n.conv_w.as_ptr(),
            n.conv_b.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
            s.c(),
            s.k,
            n.z.as_ptr(),
            n.a.as_ptr(),
            n.b.as_ptr(),
            n.a_log.as_ptr(),
            n.dt_bias.as_ptr(),
            st.as_mut_ptr(),
            s.hk,
            s.hv,
            s.dk,
            s.dv,
            n.norm_w.as_ptr(),
            EPS,
            out.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_delta_step returned {code}");
    (mixed, cs, st, out)
}

/// The thread count is a scheduling choice only: every output of the step is bit-identical at one,
/// two, seven, twenty-four threads and at the default.
#[test]
fn output_is_bit_identical_across_thread_counts() {
    let n = gen_delta(&REAL, 77, false);
    let base = run(&REAL, &n, 1);
    for t in [2usize, 7, 24, 0] {
        let got = run(&REAL, &n, t);
        assert_eq!(bits(&got.0), bits(&base.0), "threads={t}: mixed_qkv");
        assert_eq!(bits(&got.1), bits(&base.1), "threads={t}: conv_state");
        assert_eq!(bits(&got.2), bits(&base.2), "threads={t}: state");
        assert_eq!(bits(&got.3), bits(&base.3), "threads={t}: out");
    }
}

/// The conv state and the recurrent state the first step writes are the ones the second step must
/// read: stepping twice matches the f64 reference taken over the carried states.
#[test]
fn a_second_step_carries_the_state_forward_correctly() {
    let n0 = gen_delta(&REAL, 909, false);
    let (_, cs1, st1, _) = run(&REAL, &n0, 24);

    let mut n1 = gen_delta(&REAL, 911, false);
    n1.conv_w = n0.conv_w.clone();
    n1.conv_b = n0.conv_b.clone();
    n1.a_log = n0.a_log.clone();
    n1.dt_bias = n0.dt_bias.clone();
    n1.norm_w = n0.norm_w.clone();
    n1.conv_state = cs1;
    n1.state = st1;

    let e = delta_reference(&REAL, &n1, EPS);
    let (mixed, cs, st, out) = run(&REAL, &n1, 7);
    for (got, want, label) in [
        (&mixed, &e.y, "mixed_qkv"),
        (&cs, &e.conv_state, "conv_state"),
        (&st, &e.state, "state"),
        (&out, &e.out, "out"),
    ] {
        let (norm, comp) = err_pair(got, want, label);
        assert!(
            norm <= TOL && comp <= TOL,
            "step 2 {label}: {norm:.3e}/{comp:.3e}"
        );
    }
}

/// A null buffer, a zero or mismatched dimension, a head count that does not divide, a head
/// dimension past the kernel's limit and a NaN or negative `eps` are all refused with a code.
#[test]
fn bad_pointers_and_shapes_are_error_codes() {
    let n = gen_delta(&REAL, 1, false);
    let mut mixed = n.mixed.clone();
    let mut cs = n.conv_state.clone();
    let mut st = n.state.clone();
    let mut out = vec![0.0f32; REAL.hv * REAL.dv];
    let (hk, hv, dk, dv, k, c) = (REAL.hk, REAL.hv, REAL.dk, REAL.dv, REAL.k, REAL.c());
    let mut go = |mq: *mut f32,
                  hk: usize,
                  hv: usize,
                  dk: usize,
                  dv: usize,
                  c: usize,
                  k: usize,
                  eps: f32| unsafe {
        btb_delta_step(
            mq,
            cs.as_mut_ptr(),
            n.conv_w.as_ptr(),
            std::ptr::null(),
            c,
            k,
            n.z.as_ptr(),
            n.a.as_ptr(),
            n.b.as_ptr(),
            n.a_log.as_ptr(),
            n.dt_bias.as_ptr(),
            st.as_mut_ptr(),
            hk,
            hv,
            dk,
            dv,
            n.norm_w.as_ptr(),
            eps,
            out.as_mut_ptr(),
            0,
        )
    };
    let m = mixed.as_mut_ptr();
    assert_eq!(
        go(std::ptr::null_mut(), hk, hv, dk, dv, c, k, EPS),
        ERR_NULL
    );
    assert_eq!(go(m, 0, hv, dk, dv, c, k, EPS), ERR_DOMAIN);
    assert_eq!(go(m, hk, hv, dk, dv, c, 0, EPS), ERR_DOMAIN);

    assert_eq!(go(m, 5, hv, dk, dv, c, k, EPS), ERR_DOMAIN);
    assert_eq!(go(m, hk, hv, dk, dv, c + 1, k, EPS), ERR_DOMAIN);

    assert_eq!(go(m, 1, 1, 512, 4, 2 * 512 + 4, k, EPS), ERR_DOMAIN);

    // eps goes under a square root: a NaN or a negative would fold NaN through the whole step
    assert_eq!(go(m, hk, hv, dk, dv, c, k, f32::NAN), ERR_DOMAIN);
    assert_eq!(go(m, hk, hv, dk, dv, c, k, -1e-6), ERR_DOMAIN);
    assert_eq!(go(m, hk, hv, dk, dv, c, k, f32::NEG_INFINITY), ERR_DOMAIN);
    // zero is legal: the norm is then unregularised, not undefined
    assert_eq!(go(m, hk, hv, dk, dv, c, k, 0.0), OK);
}
