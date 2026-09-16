// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The fenced DeltaNet step at the Qwen3.8-27B shape (hk 16, hv 48, dk 128, dv 128, K 4).

use super::*;
use btb_native::btb_delta_step;
use btb_native::codes::OK;
use btb_native::gemv::isa;

pub const REAL: DeltaShape = DeltaShape {
    hk: 16,
    hv: 48,
    dk: 128,
    dv: 128,
    k: 4,
};
pub const EPS: f32 = 1e-6;
/// Normwise bound (against the peak) and componentwise bound (on components above 1% of the
/// peak): the latter is looser because an 8-wide RMS norm over f32 rounds at the 1e-5 level.
const TOL_NORM: f64 = 1e-5;
const TOL_COMP: f64 = 1e-4;

struct Ro {
    conv_w: Fence<f32>,
    conv_b: Option<Fence<f32>>,
    z: Fence<f32>,
    a: Fence<f32>,
    b: Fence<f32>,
    a_log: Fence<f32>,
    dt_bias: Fence<f32>,
    norm_w: Fence<f32>,
}

fn ro(v: &[f32], label: &str, align: Align) -> Fence<f32> {
    let mut f = Fence::<f32>::new(label, v.len(), align);
    f.copy_from(v);
    f.protect_readonly();
    f
}

fn readonly(n: &DeltaIn, align: Align) -> Ro {
    Ro {
        conv_w: ro(&n.conv_w, "conv_w", align),
        conv_b: n.conv_b.as_ref().map(|b| ro(b, "conv_b", align)),
        z: ro(&n.z, "z", align),
        a: ro(&n.a, "a", align),
        b: ro(&n.b, "b", align),
        a_log: ro(&n.a_log, "a_log", align),
        dt_bias: ro(&n.dt_bias, "dt_bias", align),
        norm_w: ro(&n.norm_w, "norm_w", align),
    }
}

impl Ro {
    fn check(&self, ctx: &str) {
        self.conv_w.check_borders(ctx);
        if let Some(b) = &self.conv_b {
            b.check_borders(ctx);
        }
        self.z.check_borders(ctx);
        self.a.check_borders(ctx);
        self.b.check_borders(ctx);
        self.a_log.check_borders(ctx);
        self.dt_bias.check_borders(ctx);
        self.norm_w.check_borders(ctx);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn step(
    s: &DeltaShape,
    mixed: *mut f32,
    conv_state: *mut f32,
    state: *mut f32,
    out: *mut f32,
    r: &Ro,
    threads: usize,
) -> i32 {
    btb_delta_step(
        mixed,
        conv_state,
        r.conv_w.ptr(),
        r.conv_b.as_ref().map_or(std::ptr::null(), |b| b.ptr()),
        s.c(),
        s.k,
        r.z.ptr(),
        r.a.ptr(),
        r.b.ptr(),
        r.a_log.ptr(),
        r.dt_bias.ptr(),
        state,
        s.hk,
        s.hv,
        s.dk,
        s.dv,
        r.norm_w.ptr(),
        EPS,
        out,
        threads,
    )
}

fn compare(e: &DeltaExpect, mixed: &[f32], cs: &[f32], st: &[f32], out: &[f32], ctx: &str) {
    for (got, want, label) in [
        (mixed, &e.y, "mixed_qkv"),
        (cs, &e.conv_state, "conv_state"),
        (st, &e.state, "state"),
        (out, &e.out, "out"),
    ] {
        let (norm, comp) = err_pair(got, want, label);
        assert!(
            norm <= TOL_NORM && comp <= TOL_COMP,
            "{ctx} {label}: {norm:.3e}/{comp:.3e} off the f64 reference"
        );
    }
}

/// Every in-place buffer (mixed, conv_state, state) and `out` against a guard page on the given
/// side; every parameter read-only; the result against the f64 reference.
pub fn check(label: &str, s: &DeltaShape, threads_set: &[usize]) {
    eprintln!("[{label}] isa {:?}", isa());
    for (bias, seed) in [(false, 11u64), (true, 22)] {
        let n = gen_delta(s, seed, bias);
        let e = delta_reference(s, &n, EPS);
        for align in [Align::End, Align::Start] {
            let r = readonly(&n, align);
            for &threads in threads_set {
                let ctx = format!("{label} bias={bias} {align:?} threads={threads}");
                let mut mixed = Fence::<f32>::new("mixed_qkv", s.c(), align);
                mixed.copy_from(&n.mixed);
                let mut cs = Fence::<f32>::new("conv_state", s.c() * s.k, align);
                cs.copy_from(&n.conv_state);
                let mut st = Fence::<f32>::new("state", s.hv * s.dk * s.dv, align);
                st.copy_from(&n.state);
                let mut out = Fence::<f32>::new("out", s.hv * s.dv, align);
                out.fill(poison_f32());
                let rc = unsafe {
                    step(
                        s,
                        mixed.mut_ptr(),
                        cs.mut_ptr(),
                        st.mut_ptr(),
                        out.mut_ptr(),
                        &r,
                        threads,
                    )
                };
                assert_eq!(rc, OK, "{ctx}: rc {rc}");
                for f in [&mixed, &cs, &st, &out] {
                    f.check_borders(&ctx);
                }
                r.check(&ctx);
                if let Some(i) = out.as_slice().iter().position(|v| is_poison_f32(*v)) {
                    panic!("{ctx}: out element {i} was never written");
                }
                compare(
                    &e,
                    mixed.as_slice(),
                    cs.as_slice(),
                    st.as_slice(),
                    out.as_slice(),
                    &ctx,
                );
            }
        }
    }
    eprintln!("[{label}] real shape clean for threads {threads_set:?}");
}

/// The tree pass keeps T copies of the states in one slab and steps each copy in place: the
/// other rows of the slab must stay untouched (poisoned here) and each stepped row must match.
pub fn slab_check(label: &str, s: &DeltaShape, t: usize, threads: usize) {
    let n = gen_delta(s, 909, false);
    let e = delta_reference(s, &n, EPS);
    let r = readonly(&n, Align::End);
    let cs_len = s.c() * s.k;
    let st_len = s.hv * s.dk * s.dv;
    let mut cs = Fence::<f32>::new("conv_slab", t * cs_len, Align::End);
    let mut st = Fence::<f32>::new("state_slab", t * st_len, Align::End);
    let mut out = Fence::<f32>::new("out", s.hv * s.dv, Align::End);
    for p in 0..t {
        cs.fill(poison_f32());
        st.fill(poison_f32());
        cs.as_mut_slice()[p * cs_len..(p + 1) * cs_len].copy_from_slice(&n.conv_state);
        st.as_mut_slice()[p * st_len..(p + 1) * st_len].copy_from_slice(&n.state);
        let mut mixed = Fence::<f32>::new("mixed_qkv", s.c(), Align::End);
        mixed.copy_from(&n.mixed);
        out.fill(poison_f32());
        let ctx = format!("{label} slab row {p} of {t} threads={threads}");
        let rc = unsafe {
            step(
                s,
                mixed.mut_ptr(),
                cs.mut_ptr().add(p * cs_len),
                st.mut_ptr().add(p * st_len),
                out.mut_ptr(),
                &r,
                threads,
            )
        };
        assert_eq!(rc, OK, "{ctx}: rc {rc}");
        for f in [&mixed, &cs, &st, &out] {
            f.check_borders(&ctx);
        }
        r.check(&ctx);
        for (q, row) in cs.as_slice().chunks(cs_len).enumerate() {
            if q != p {
                assert!(
                    row.iter().all(|v| is_poison_f32(*v)),
                    "{ctx}: conv row {q} was written"
                );
            }
        }
        for (q, row) in st.as_slice().chunks(st_len).enumerate() {
            if q != p {
                assert!(
                    row.iter().all(|v| is_poison_f32(*v)),
                    "{ctx}: state row {q} was written"
                );
            }
        }
        compare(
            &e,
            mixed.as_slice(),
            &cs.as_slice()[p * cs_len..(p + 1) * cs_len],
            &st.as_slice()[p * st_len..(p + 1) * st_len],
            out.as_slice(),
            &ctx,
        );
    }
    eprintln!("[{label}] slab of {t} rows clean (threads {threads})");
}
