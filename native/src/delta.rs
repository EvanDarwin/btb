// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
use crate::codes::{ERR_DOMAIN, ERR_NULL, OK};
use crate::gemv::{aligned, MAX_ELEMS};
use rayon::prelude::*;

pub const MAX_HEAD_DIM: usize = 256;

const L2_EPS: f32 = 1e-6;

#[inline(always)]
fn exp_fast(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_145_75;
    const LN2_LO: f32 = 1.428_606_8e-6;
    const MAGIC: f32 = 12_582_912.0;

    let xc = if x < -104.0 { -104.0 } else { x };
    let xc = if xc > 88.7 { 88.7 } else { xc };

    let t = xc * LOG2E + MAGIC;
    let k = t.to_bits().wrapping_sub(MAGIC.to_bits()) as i32;
    let kf = t - MAGIC;

    let r = (xc - kf * LN2_HI) - kf * LN2_LO;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0
                        + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));

    let k1 = k >> 1;
    let k2 = k - k1;
    p * f32::from_bits(((k1 + 127) as u32) << 23) * f32::from_bits(((k2 + 127) as u32) << 23)
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    if x < -88.0 {
        0.0
    } else {
        x / (1.0 + exp_fast(-x))
    }
}

#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

#[inline(always)]
unsafe fn sum_sq(x: *const f32, n: usize) -> f32 {
    let mut acc = [0.0f32; 8];
    let nb = n / 8;
    for j in 0..nb {
        for (l, a) in acc.iter_mut().enumerate() {
            let v = *x.add(j * 8 + l);
            *a += v * v;
        }
    }
    for i in (nb * 8)..n {
        let v = *x.add(i);
        acc[i - nb * 8] += v * v;
    }
    let (t0, t1) = (acc[0] + acc[1], acc[2] + acc[3]);
    let (t2, t3) = (acc[4] + acc[5], acc[6] + acc[7]);
    (t0 + t1) + (t2 + t3)
}

#[derive(Clone, Copy)]
struct Shape {
    c_dim: usize,
    k_size: usize,
    hv: usize,
    dk: usize,
    dv: usize,

    rep: usize,

    key_dim: usize,
}

#[derive(Clone, Copy)]
struct Tensors {
    mixed_qkv: *mut f32,
    conv_state: *mut f32,
    conv_w: *const f32,
    conv_b: *const f32,
    z: *const f32,
    a: *const f32,
    b: *const f32,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    norm_w: *const f32,
    out: *mut f32,

    eps: f32,
}
unsafe impl Send for Tensors {}
unsafe impl Sync for Tensors {}

#[inline(always)]
unsafe fn conv_range(p: Tensors, s: Shape, node: usize, c0: usize, c1: usize) {
    let x = p.mixed_qkv.add(node * s.c_dim);
    let st = p.conv_state.add(node * s.c_dim * s.k_size);
    for c in c0..c1 {
        let win = st.add(c * s.k_size);

        for j in 0..s.k_size - 1 {
            *win.add(j) = *win.add(j + 1);
        }
        *win.add(s.k_size - 1) = *x.add(c);

        let w = p.conv_w.add(c * s.k_size);
        let mut acc = if p.conv_b.is_null() {
            0.0f32
        } else {
            *p.conv_b.add(c)
        };
        for j in 0..s.k_size {
            acc += *w.add(j) * *win.add(j);
        }
        *x.add(c) = silu(acc);
    }
}

#[inline(always)]
#[allow(clippy::needless_range_loop)]
unsafe fn head_step(p: Tensors, s: Shape, node: usize, h: usize) {
    let (dk, dv) = (s.dk, s.dv);
    let y = p.mixed_qkv.add(node * s.c_dim);
    let hk = h / s.rep;
    let q_src = y.add(hk * dk);
    let k_src = y.add(s.key_dim + hk * dk);
    let v_src = std::slice::from_raw_parts(y.add(2 * s.key_dim + h * dv), dv);

    let mut qbuf = [0.0f32; MAX_HEAD_DIM];
    let mut kbuf = [0.0f32; MAX_HEAD_DIM];
    let qn = &mut qbuf[..dk];
    let kn = &mut kbuf[..dk];
    let scale = (1.0f64 / (dk as f64).sqrt()) as f32;
    let qi = 1.0 / (sum_sq(q_src, dk) + L2_EPS).sqrt();
    let ki = 1.0 / (sum_sq(k_src, dk) + L2_EPS).sqrt();
    for d in 0..dk {
        qn[d] = (*q_src.add(d) * qi) * scale;
        kn[d] = *k_src.add(d) * ki;
    }

    let beta = 1.0 / (1.0 + (-*p.b.add(node * s.hv + h)).exp());
    let g = -(*p.a_log.add(h)).exp() * softplus(*p.a.add(node * s.hv + h) + *p.dt_bias.add(h));
    let decay = g.exp();

    let st = p.state.add((node * s.hv + h) * dk * dv);

    let mut kvbuf = [0.0f32; MAX_HEAD_DIM];
    let kv = &mut kvbuf[..dv];
    for d in 0..dk {
        let kd = kn[d];
        let row = std::slice::from_raw_parts(st.add(d * dv), dv);
        for (sv, acc) in row.iter().zip(kv.iter_mut()) {
            *acc += (*sv * decay) * kd;
        }
    }

    let mut dbuf = [0.0f32; MAX_HEAD_DIM];
    let delta = &mut dbuf[..dv];
    for ((d, vt), m) in delta.iter_mut().zip(v_src.iter()).zip(kv.iter()) {
        *d = (*vt - *m) * beta;
    }

    let mut cbuf = [0.0f32; MAX_HEAD_DIM];
    let core = &mut cbuf[..dv];
    for d in 0..dk {
        let (kd, qd) = (kn[d], qn[d]);
        let row = std::slice::from_raw_parts_mut(st.add(d * dv), dv);
        for ((sv, dl), acc) in row.iter_mut().zip(delta.iter()).zip(core.iter_mut()) {
            let x = *sv * decay + kd * *dl;
            *sv = x;
            *acc += x * qd;
        }
    }

    let rms = 1.0 / (sum_sq(core.as_ptr(), dv) / dv as f32 + p.eps).sqrt();
    let zh = std::slice::from_raw_parts(p.z.add((node * s.hv + h) * dv), dv);
    let nw = std::slice::from_raw_parts(p.norm_w, dv);
    let o = std::slice::from_raw_parts_mut(p.out.add((node * s.hv + h) * dv), dv);
    for (((ov, cv), wv), zv) in o.iter_mut().zip(core.iter()).zip(nw.iter()).zip(zh.iter()) {
        *ov = (*wv * (*cv * rms)) * silu(*zv);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn head_step_avx2(p: Tensors, s: Shape, node: usize, h: usize) {
    head_step(p, s, node, h)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn conv_range_avx2(p: Tensors, s: Shape, node: usize, c0: usize, c1: usize) {
    conv_range(p, s, node, c0, c1)
}

#[inline]
unsafe fn key_head_task(p: Tensors, s: Shape, node: usize, kh: usize, avx2: bool) {
    let (dk, dv, rep) = (s.dk, s.dv, s.rep);
    let h0 = kh * rep;

    conv_dispatch(p, s, node, kh * dk, kh * dk + dk, avx2);
    conv_dispatch(
        p,
        s,
        node,
        s.key_dim + kh * dk,
        s.key_dim + kh * dk + dk,
        avx2,
    );

    conv_dispatch(
        p,
        s,
        node,
        2 * s.key_dim + h0 * dv,
        2 * s.key_dim + (h0 + rep) * dv,
        avx2,
    );
    for h in h0..h0 + rep {
        head_dispatch(p, s, node, h, avx2);
    }
}

#[inline]
unsafe fn head_dispatch(p: Tensors, s: Shape, node: usize, h: usize, avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return head_step_avx2(p, s, node, h);
    }
    let _ = avx2;
    head_step(p, s, node, h)
}

#[inline]
unsafe fn conv_dispatch(p: Tensors, s: Shape, node: usize, c0: usize, c1: usize, avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return conv_range_avx2(p, s, node, c0, c1);
    }
    let _ = avx2;
    conv_range(p, s, node, c0, c1)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn delta_step(
    mixed_qkv: *mut f32,
    conv_state: *mut f32,
    conv_w: *const f32,
    conv_b: *const f32,
    c_dim: usize,
    k_size: usize,
    z: *const f32,
    a: *const f32,
    b: *const f32,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    norm_w: *const f32,
    eps: f32,
    out: *mut f32,
    nodes: usize,
    threads: usize,
) -> i32 {
    if mixed_qkv.is_null()
        || conv_state.is_null()
        || conv_w.is_null()
        || z.is_null()
        || a.is_null()
        || b.is_null()
        || a_log.is_null()
        || dt_bias.is_null()
        || state.is_null()
        || norm_w.is_null()
        || out.is_null()
    {
        return ERR_NULL;
    }
    if c_dim == 0 || k_size == 0 || hk == 0 || hv == 0 || dk == 0 || dv == 0 || nodes == 0 {
        return ERR_DOMAIN;
    }
    if dk > MAX_HEAD_DIM || dv > MAX_HEAD_DIM {
        return ERR_DOMAIN;
    }
    // the step dereferences its elements directly: every buffer naturally aligned
    if !aligned(mixed_qkv)
        || !aligned(conv_state)
        || !aligned(conv_w)
        || !(conv_b.is_null() || aligned(conv_b))
        || !aligned(z)
        || !aligned(a)
        || !aligned(b)
        || !aligned(a_log)
        || !aligned(dt_bias)
        || !aligned(state)
        || !aligned(norm_w)
        || !aligned(out)
    {
        return ERR_DOMAIN;
    }
    if !hv.is_multiple_of(hk) {
        return ERR_DOMAIN;
    }

    let key_dim = match hk.checked_mul(dk) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let value_dim = match hv.checked_mul(dv) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    match key_dim
        .checked_mul(2)
        .and_then(|v| v.checked_add(value_dim))
    {
        Some(v) if v == c_dim => {}
        _ => return ERR_DOMAIN,
    }
    // a negative or NaN eps takes the square root of a negative in the norm and folds NaN through the step
    if eps.is_nan() || eps < 0.0 {
        return ERR_DOMAIN;
    }
    // the element counts fit usize and their byte sizes a pointer offset
    let conv_elems = match c_dim.checked_mul(k_size) {
        Some(v) if v <= MAX_ELEMS => v,
        _ => return ERR_DOMAIN,
    };
    let state_elems = match value_dim.checked_mul(dk) {
        Some(v) if v <= MAX_ELEMS => v,
        _ => return ERR_DOMAIN,
    };
    if nodes.checked_mul(state_elems).is_none_or(|v| v > MAX_ELEMS)
        || nodes.checked_mul(conv_elems).is_none_or(|v| v > MAX_ELEMS)
    {
        return ERR_DOMAIN;
    }

    let s = Shape {
        c_dim,
        k_size,
        hv,
        dk,
        dv,
        rep: hv / hk,
        key_dim,
    };
    let p = Tensors {
        mixed_qkv,
        conv_state,
        conv_w,
        conv_b,
        z,
        a,
        b,
        a_log,
        dt_bias,
        state,
        norm_w,
        out,
        eps,
    };

    let global = rayon::current_num_threads().max(1);
    let nt = match crate::gemv::resolve_threads(threads) {
        Some(v) => v,
        None => return ERR_DOMAIN,
    };
    let avx2 = crate::gemv::isa() != crate::gemv::Isa::Scalar;
    let tasks = nodes * hk;

    if nt <= 1 || tasks == 1 {
        for n in 0..nodes {
            for kh in 0..hk {
                key_head_task(p, s, n, kh, avx2);
            }
        }
        return OK;
    }

    let run = || {
        (0..tasks)
            .into_par_iter()
            .for_each(|t| unsafe { key_head_task(p, s, t / hk, t % hk, avx2) });
    };

    if nt == global {
        run();
    } else if let Some(pool) = crate::gemv::pool_for(nt) {
        pool.install(run);
    } else {
        run();
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::approx_constant, clippy::excessive_precision)]
    fn log2e_is_bit_identical_to_the_certified_literal() {
        assert_eq!(
            std::f32::consts::LOG2_E.to_bits(),
            1.442_695_04f32.to_bits()
        );
    }

    #[test]
    fn exp_fast_is_accurate_over_the_whole_range() {
        let mut worst = 0.0f64;
        let mut at = 0.0f32;
        let mut x = -104.0f32;
        while x <= 88.7 {
            let want = (x as f64).exp();

            if want > 1e-37 && want <= f32::MAX as f64 {
                let rel = ((exp_fast(x) as f64) - want).abs() / want;
                if rel > worst {
                    worst = rel;
                    at = x;
                }
            }
            x += if (-2.0..2.0).contains(&x) {
                0.0009765625
            } else {
                0.03125
            };
        }
        assert!(
            worst < 2e-7,
            "exp_fast worst relative error {worst:.3e} at x = {at}"
        );
        assert_eq!(exp_fast(0.0), 1.0);
        assert!((exp_fast(1.0) as f64 - std::f64::consts::E).abs() < 1e-6);

        assert!(exp_fast(-200.0) < 1e-38, "left tail must collapse to ~0");
        assert!(exp_fast(-200.0) >= 0.0);
        let hi = exp_fast(200.0);
        assert!(
            hi.is_finite() && hi > 3.0e38,
            "right tail saturates near f32::MAX, got {hi}"
        );
    }

    #[test]
    fn silu_and_softplus_match_their_definitions() {
        for &x in &[-40.0f32, -8.0, -1.0, -0.1, 0.0, 0.1, 1.0, 8.0, 40.0] {
            let want = (x as f64) / (1.0 + (-(x as f64)).exp());
            let got = silu(x) as f64;
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-6),
                "silu({x}) = {got}, want {want}"
            );
            let wsp = (1.0 + (x as f64).exp()).ln();
            assert!(
                (softplus(x) as f64 - wsp).abs() <= 1e-6 * wsp.max(1e-6),
                "softplus({x})"
            );
        }

        assert_eq!(silu(0.0), 0.0);
        assert_eq!(silu(-200.0), 0.0);
        assert_eq!(silu(-1e30), 0.0);
        assert_eq!(silu(f32::NEG_INFINITY), 0.0);

        assert!(silu(-80.0) < 0.0 && silu(-80.0) > -1e-30);

        assert_eq!(silu(1e30), 1e30);

        assert_eq!(softplus(25.0), 25.0);
    }

    #[test]
    fn sum_sq_is_fixed_order_and_correct() {
        for n in [0usize, 1, 7, 8, 9, 128, 129] {
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
            let got = unsafe { sum_sq(v.as_ptr(), n) } as f64;
            let want: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum();
            assert!(
                (got - want).abs() <= 1e-6 * want.max(1e-6),
                "n = {n}: {got} vs {want}"
            );
            assert_eq!(got, unsafe { sum_sq(v.as_ptr(), n) } as f64);
        }
    }
}
