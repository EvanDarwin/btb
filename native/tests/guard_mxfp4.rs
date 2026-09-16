// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced MXFP4 rows and grouped mat-vec: the blocks, the scales, x and y all sit against a
//! guard page, so a read or a write past any of them faults. The output must not move with the
//! thread count, the alignment or the batch width, a grouped task must equal its own rows call,
//! and every row must match an f64 reference built from the e2m1 table and `2^(scale - 127)`.
#![cfg(windows)]

mod common;

use btb_native::codes::{ERR_DOMAIN, ERR_NULL, OK};
use btb_native::gemv::{isa, MAX_THREADS};
use btb_native::{btb_gemv_mxfp4_group, btb_gemv_mxfp4_rows};
use common::*;

/// The expert shape of a mixture-of-experts layer that ships its weights in this form.
const EXPERT: (usize, usize) = (5760, 2880);
/// Row and column counts that do not divide the tile, with cols still a multiple of the block.
const RAGGED: &[(usize, usize)] = &[(1, 32), (37, 96), (129, 160)];
const TOL: f64 = 1e-5;

/// A fenced MXFP4 matrix, read-only for the length of the test, alongside the plain copy the f64
/// reference is taken from.
struct Weights {
    rows: usize,
    cols: usize,
    blocks: Fence<u8>,
    scales: Fence<u8>,
    raw: Mxfp4,
}

fn make_weights(rows: usize, cols: usize, seed: u64, align: Align) -> Weights {
    let raw = gen_mxfp4(rows, cols, seed);
    let mut blocks = Fence::<u8>::new("blocks", raw.blocks.len(), align);
    blocks.copy_from(&raw.blocks);
    let mut scales = Fence::<u8>::new("scales", raw.scales.len(), align);
    scales.copy_from(&raw.scales);
    blocks.protect_readonly();
    scales.protect_readonly();
    Weights {
        rows,
        cols,
        blocks,
        scales,
        raw,
    }
}

impl Weights {
    fn check_borders(&self, ctx: &str) {
        self.blocks.check_borders(ctx);
        self.scales.check_borders(ctx);
    }
}

fn finish(y: &Fence<f32>, ctx: &str) -> Vec<f32> {
    y.check_borders(ctx);
    let o = y.as_slice();
    if let Some(i) = o.iter().position(|v| is_poison_f32(*v)) {
        panic!("{ctx}: y element {i} was never written");
    }
    o.to_vec()
}

fn rows_call(
    w: &Weights,
    x: &Fence<f32>,
    t: usize,
    threads: usize,
    y: &mut Fence<f32>,
    ctx: &str,
) -> Vec<f32> {
    y.fill(poison_f32());
    let rc = unsafe {
        btb_gemv_mxfp4_rows(
            w.blocks.ptr(),
            w.scales.ptr(),
            w.rows,
            w.cols,
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

/// Every shape, batch width, thread count and alignment gives one set of bits, and row 0 matches
/// the f64 reference.
#[test]
fn rows_sweep() {
    eprintln!("isa {:?}", isa());
    let mut shapes: Vec<(usize, usize)> = RAGGED.to_vec();
    shapes.push(EXPERT);

    for (idx, &(rows, cols)) in shapes.iter().enumerate() {
        let wt = make_weights(rows, cols, 0x4F04 + idx as u64, Align::End);
        let x_all = gen_f32(8 * cols, 0x515 + idx as u64);
        let want = mxfp4_reference(&wt.raw, rows, cols, &x_all[..cols]);

        for &t in &[1usize, 2, 5, 8] {
            let mut base: Option<Vec<u32>> = None;
            for &threads in &[1usize, 0, 16] {
                for align in [Align::End, Align::Start] {
                    let ctx = format!("{rows}x{cols} T={t} threads={threads} {align:?}");
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
        eprintln!("{rows:>6} x {cols:<6} clean");
    }
}

/// Every row of a batch is exactly what its own single-row call returns.
#[test]
fn batch_rows_equal_their_own_calls() {
    let mut shapes: Vec<(usize, usize)> = RAGGED.to_vec();
    shapes.push(EXPERT);
    for (idx, &(rows, cols)) in shapes.iter().enumerate() {
        let wt = make_weights(rows, cols, 0xB47C4 + idx as u64, Align::End);
        let x_all = gen_f32(8 * cols, 0x77 + idx as u64);
        for t in [2usize, 3, 8] {
            for &threads in &[1usize, 0] {
                let ctx = format!("{rows}x{cols} T={t} threads={threads}");
                let mut x = Fence::<f32>::new("x", t * cols, Align::End);
                x.copy_from(&x_all[..t * cols]);
                x.protect_readonly();
                let mut y = Fence::<f32>::new("y", t * rows, Align::End);
                let batched = rows_call(&wt, &x, t, threads, &mut y, &ctx);
                for i in 0..t {
                    let mut xi = Fence::<f32>::new("x1", cols, Align::End);
                    xi.copy_from(&x_all[i * cols..(i + 1) * cols]);
                    xi.protect_readonly();
                    let mut y1 = Fence::<f32>::new("y1", rows, Align::End);
                    let solo = rows_call(&wt, &xi, 1, threads, &mut y1, &ctx);
                    assert_eq!(
                        bits(&batched[i * rows..(i + 1) * rows]),
                        bits(&solo),
                        "{ctx}: batch row {i} differs from its own call"
                    );
                }
            }
        }
    }
}

/// A group of tasks under one dispatch: each task's output is bit-identical to its own rows call,
/// with every buffer fenced and every input read-only.
#[test]
fn grouped_tasks_equal_their_own_rows_calls() {
    let shapes: &[(usize, usize, usize)] = &[
        (EXPERT.0, EXPERT.1, 1),
        (EXPERT.0, EXPERT.1, 3),
        (1, 32, 1),
        (37, 96, 2),
        (129, 160, 5),
    ];
    let ws: Vec<Weights> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(rows, cols, _))| make_weights(rows, cols, 0x6809 + i as u64, Align::End))
        .collect();
    let xs: Vec<Fence<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(i, &(_, cols, b))| {
            let mut f = Fence::<f32>::new("x", b * cols, Align::End);
            f.copy_from(&gen_f32(b * cols, 0x1465 + i as u64));
            f.protect_readonly();
            f
        })
        .collect();

    for &threads in &[1usize, 0, 16] {
        for align in [Align::End, Align::Start] {
            let ctx = format!("group threads={threads} {align:?}");
            let ys: Vec<Fence<f32>> = shapes
                .iter()
                .map(|&(rows, _, b)| {
                    let mut f = Fence::<f32>::new("y", b * rows, align);
                    f.fill(poison_f32());
                    f
                })
                .collect();
            let bp: Vec<*const u8> = ws.iter().map(|w| w.blocks.ptr()).collect();
            let sp: Vec<*const u8> = ws.iter().map(|w| w.scales.ptr()).collect();
            let rows: Vec<usize> = shapes.iter().map(|s| s.0).collect();
            let cols: Vec<usize> = shapes.iter().map(|s| s.1).collect();
            let bs: Vec<usize> = shapes.iter().map(|s| s.2).collect();
            let xp: Vec<*const f32> = xs.iter().map(|f| f.ptr()).collect();
            let mut yp: Vec<*mut f32> = ys.iter().map(|f| f.mut_ptr()).collect();
            let rc = unsafe {
                btb_gemv_mxfp4_group(
                    shapes.len(),
                    bp.as_ptr(),
                    sp.as_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    xp.as_ptr(),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    threads,
                )
            };
            assert_eq!(rc, OK, "{ctx}: rc {rc}");

            for (i, &(rows, _, b)) in shapes.iter().enumerate() {
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

/// A null pointer, a zero dimension, a column count that is not a multiple of the 32-weight block,
/// an overflowing dimension, a thread count past the cap and a byte-offset x pointer are each
/// refused with a code, and a refused call leaves the output alone.
#[test]
fn bad_pointers_and_shapes_are_error_codes() {
    let (rows, cols) = (4usize, 64usize);
    let m = gen_mxfp4(rows, cols, 0xBAD);
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];
    let (bp, sp) = (m.blocks.as_ptr(), m.scales.as_ptr());

    let call = |b: *const u8,
                s: *const u8,
                rows: usize,
                cols: usize,
                xp: *const f32,
                t: usize,
                yp: *mut f32,
                threads: usize| unsafe {
        btb_gemv_mxfp4_rows(b, s, rows, cols, xp, t, yp, threads)
    };

    assert_eq!(
        call(
            std::ptr::null(),
            sp,
            rows,
            cols,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            0
        ),
        ERR_NULL
    );
    assert_eq!(
        call(
            bp,
            std::ptr::null(),
            rows,
            cols,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            0
        ),
        ERR_NULL
    );
    assert_eq!(
        call(bp, sp, rows, cols, std::ptr::null(), 1, y.as_mut_ptr(), 0),
        ERR_NULL
    );
    assert_eq!(
        call(bp, sp, rows, cols, x.as_ptr(), 1, std::ptr::null_mut(), 0),
        ERR_NULL
    );

    assert_eq!(
        call(bp, sp, 0, cols, x.as_ptr(), 1, y.as_mut_ptr(), 0),
        ERR_DOMAIN
    );
    assert_eq!(
        call(bp, sp, rows, 0, x.as_ptr(), 1, y.as_mut_ptr(), 0),
        ERR_DOMAIN
    );
    assert_eq!(
        call(bp, sp, rows, cols, x.as_ptr(), 0, y.as_mut_ptr(), 0),
        ERR_DOMAIN
    );

    // a block never straddles a row: cols must be a whole number of 32-weight blocks
    for bad in [1usize, 31, 33, 63, 65, 96 + 1] {
        assert_eq!(
            call(bp, sp, rows, bad, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN,
            "cols {bad} is not a multiple of {MX_BLOCK}"
        );
    }

    assert_eq!(
        call(bp, sp, usize::MAX, cols, x.as_ptr(), 1, y.as_mut_ptr(), 0),
        ERR_DOMAIN
    );
    assert_eq!(
        call(
            bp,
            sp,
            rows,
            cols,
            x.as_ptr(),
            1,
            y.as_mut_ptr(),
            MAX_THREADS + 1
        ),
        ERR_DOMAIN
    );

    let bad_x = unsafe { (x.as_ptr() as *const u8).add(1) as *const f32 };
    assert_eq!(
        call(bp, sp, rows, cols, bad_x, 1, y.as_mut_ptr(), 0),
        ERR_DOMAIN
    );

    // the grouped form checks the same things, per task and on the arrays themselves
    let blocks = [bp];
    let scales = [sp];
    let rows_a = [rows];
    let cols_a = [cols];
    let b_a = [1usize];
    let mut y_a = [y.as_mut_ptr()];
    let nul_b: [*const u8; 1] = [std::ptr::null()];
    let bad_cols = [33usize];
    let zero = [0usize];

    let group = |n: usize,
                 bl: *const *const u8,
                 sc: *const *const u8,
                 rp: *const usize,
                 cp: *const usize,
                 yp: *mut *mut f32| unsafe {
        let xp = [x.as_ptr()];
        btb_gemv_mxfp4_group(n, bl, sc, rp, cp, xp.as_ptr(), b_a.as_ptr(), yp, 0)
    };

    assert_eq!(
        group(
            1,
            std::ptr::null(),
            scales.as_ptr(),
            rows_a.as_ptr(),
            cols_a.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_NULL
    );
    assert_eq!(
        group(
            1,
            blocks.as_ptr(),
            std::ptr::null(),
            rows_a.as_ptr(),
            cols_a.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_NULL
    );
    assert_eq!(
        group(
            1,
            nul_b.as_ptr(),
            scales.as_ptr(),
            rows_a.as_ptr(),
            cols_a.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_NULL
    );
    assert_eq!(
        group(
            0,
            blocks.as_ptr(),
            scales.as_ptr(),
            rows_a.as_ptr(),
            cols_a.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_DOMAIN
    );
    assert_eq!(
        group(
            1,
            blocks.as_ptr(),
            scales.as_ptr(),
            zero.as_ptr(),
            cols_a.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_DOMAIN
    );
    assert_eq!(
        group(
            1,
            blocks.as_ptr(),
            scales.as_ptr(),
            rows_a.as_ptr(),
            bad_cols.as_ptr(),
            y_a.as_mut_ptr()
        ),
        ERR_DOMAIN
    );

    assert_eq!(
        y,
        vec![0.0f32; rows],
        "a refused call must not write output"
    );
}
