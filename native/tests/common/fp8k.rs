// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The fenced FP8 mat-vec sweeps, rows and grouped, run on the machine's vector path (`guard_fp8`) and
//! pinned to scalar (`guard_scalar`): both held to the one f64 reference.

use super::*;
use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_gemv_fp8_group, btb_gemv_fp8_rows};

/// (rows, cols, sr, sc): a matrix and its scale grid.
pub type Shape = (usize, usize, usize, usize);

/// A 128x128-block projection of the size fine-grained FP8 checkpoints ship.
pub const PROJ: Shape = (2048, 1024, 16, 8);
/// Shapes that do not divide the tile, with grids from per-tensor to blocks narrower than a SIMD run.
pub const RAGGED: &[Shape] = &[(1, 13, 1, 1), (37, 96, 37, 32), (129, 160, 3, 5)];
const TOL: f64 = 1e-5;

/// A fenced FP8 matrix, read-only for the length of the test, alongside the plain copy the f64
/// reference is taken from.
pub struct Weights {
    pub rows: usize,
    pub cols: usize,
    pub w: Fence<u8>,
    pub scales: Fence<f32>,
    pub raw: Fp8,
}

pub fn make_weights(shape: Shape, seed: u64, align: Align) -> Weights {
    let (rows, cols, sr, sc) = shape;
    let raw = gen_fp8(rows, cols, sr, sc, seed);
    let mut w = Fence::<u8>::new("w", raw.w.len(), align);
    w.copy_from(&raw.w);
    let mut scales = Fence::<f32>::new("scales", raw.scales.len(), align);
    scales.copy_from(&raw.scales);
    w.protect_readonly();
    scales.protect_readonly();
    Weights {
        rows,
        cols,
        w,
        scales,
        raw,
    }
}

impl Weights {
    pub fn check_borders(&self, ctx: &str) {
        self.w.check_borders(ctx);
        self.scales.check_borders(ctx);
    }
}

pub fn finish(y: &Fence<f32>, ctx: &str) -> Vec<f32> {
    y.check_borders(ctx);
    let o = y.as_slice();
    if let Some(i) = o.iter().position(|v| is_poison_f32(*v)) {
        panic!("{ctx}: y element {i} was never written");
    }
    o.to_vec()
}

pub fn rows_call(
    w: &Weights,
    x: &Fence<f32>,
    t: usize,
    threads: usize,
    y: &mut Fence<f32>,
    ctx: &str,
) -> Vec<f32> {
    y.fill(poison_f32());
    let rc = unsafe {
        btb_gemv_fp8_rows(
            w.w.ptr(),
            w.scales.ptr(),
            w.rows,
            w.cols,
            w.raw.sr,
            w.raw.sc,
            x.ptr(),
            t,
            y.mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "{ctx}: rc {rc}");
    w.check_borders(ctx);
    x.check_borders(ctx);
    finish(y, ctx)
}

/// Row 0 of the output against the f64 reference, normwise and against each row's own conditioning.
fn check_reference(y: &[f32], want: &(Vec<f64>, Vec<f64>), ctx: &str) {
    let peak = want
        .0
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let mut norm = 0.0f64;
    let mut cond = 0.0f64;
    for (i, (&g, &w)) in y.iter().zip(want.0.iter()).enumerate() {
        assert!(g.is_finite(), "{ctx} row {i}: {g} is not finite");
        let e = (g as f64 - w).abs();
        norm = norm.max(e / peak);
        cond = cond.max(e / want.1[i].max(1e-300));
    }
    assert!(norm <= TOL, "{ctx}: normwise {norm:.3e} > {TOL:.0e}");
    assert!(cond <= TOL, "{ctx}: conditioned {cond:.3e} > {TOL:.0e}");
}

/// Every shape, batch width, thread count and alignment gives one set of bits, and row 0 matches the
/// f64 reference.
pub fn rows_sweep(label: &str, threads_set: &[usize]) {
    eprintln!("{label}: isa {:?}", isa());
    let mut shapes: Vec<Shape> = RAGGED.to_vec();
    shapes.push(PROJ);
    for (idx, &shape) in shapes.iter().enumerate() {
        let (rows, cols, _, _) = shape;
        let wt = make_weights(shape, 0xF8F8 + idx as u64, Align::End);
        let x_all = gen_f32(8 * cols, 0x515 + idx as u64);
        let want = fp8_reference(&wt.raw, rows, cols, &x_all[..cols]);
        for &t in &[1usize, 2, 5, 8] {
            let mut base: Option<Vec<u32>> = None;
            for &threads in threads_set {
                for align in [Align::End, Align::Start] {
                    let ctx = format!("{label} {rows}x{cols} T={t} threads={threads} {align:?}");
                    let mut x = Fence::<f32>::new("x", t * cols, align);
                    x.copy_from(&x_all[..t * cols]);
                    x.protect_readonly();
                    let mut y = Fence::<f32>::new("y", t * rows, align);
                    let got = rows_call(&wt, &x, t, threads, &mut y, &ctx);
                    match &base {
                        None => {
                            check_reference(&got[..rows], &want, &ctx);
                            base = Some(bits(&got));
                        }
                        Some(b) => assert_eq!(
                            &bits(&got),
                            b,
                            "{ctx}: the thread count or the alignment moved the bits"
                        ),
                    }
                }
            }
        }
    }
}

/// A group of tasks under one dispatch: each task's output is bit-identical to its own rows call, with
/// every buffer fenced and every input read-only.
pub fn group_check(label: &str, threads_set: &[usize]) {
    let shapes: &[(Shape, usize)] = &[
        (PROJ, 1),
        (PROJ, 3),
        (RAGGED[0], 1),
        (RAGGED[1], 2),
        (RAGGED[2], 5),
    ];
    let ws: Vec<Weights> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(shape, _))| make_weights(shape, 0x68F8 + i as u64, Align::End))
        .collect();
    let xs: Vec<Fence<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &((_, cols, _, _), b))| {
            let mut f = Fence::<f32>::new("x", b * cols, Align::End);
            f.copy_from(&gen_f32(b * cols, 0x1465 + i as u64));
            f.protect_readonly();
            f
        })
        .collect();
    for &threads in threads_set {
        for align in [Align::End, Align::Start] {
            let ctx = format!("{label} group threads={threads} {align:?}");
            let ys: Vec<Fence<f32>> = shapes
                .iter()
                .map(|&((rows, _, _, _), b)| {
                    let mut f = Fence::<f32>::new("y", b * rows, align);
                    f.fill(poison_f32());
                    f
                })
                .collect();
            let wp: Vec<*const u8> = ws.iter().map(|w| w.w.ptr()).collect();
            let sp: Vec<*const f32> = ws.iter().map(|w| w.scales.ptr()).collect();
            let rows: Vec<usize> = ws.iter().map(|w| w.rows).collect();
            let cols: Vec<usize> = ws.iter().map(|w| w.cols).collect();
            let srs: Vec<usize> = ws.iter().map(|w| w.raw.sr).collect();
            let scs: Vec<usize> = ws.iter().map(|w| w.raw.sc).collect();
            let bs: Vec<usize> = shapes.iter().map(|s| s.1).collect();
            let xp: Vec<*const f32> = xs.iter().map(|f| f.ptr()).collect();
            let mut yp: Vec<*mut f32> = ys.iter().map(|f| f.mut_ptr()).collect();
            let rc = unsafe {
                btb_gemv_fp8_group(
                    shapes.len(),
                    wp.as_ptr(),
                    sp.as_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    srs.as_ptr(),
                    scs.as_ptr(),
                    xp.as_ptr(),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    threads,
                )
            };
            assert_eq!(rc, OK, "{ctx}: rc {rc}");
            for (i, &((rows, _, _, _), b)) in shapes.iter().enumerate() {
                let got = finish(&ys[i], &format!("{ctx} task {i}"));
                ws[i].check_borders(&ctx);
                xs[i].check_borders(&ctx);
                let mut solo = Fence::<f32>::new("solo", b * rows, Align::End);
                let want = rows_call(
                    &ws[i],
                    &xs[i],
                    b,
                    threads,
                    &mut solo,
                    &format!("{ctx} task {i} solo"),
                );
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{ctx}: task {i} differs from its own rows call"
                );
            }
        }
    }
}
