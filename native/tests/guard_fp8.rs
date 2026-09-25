// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced FP8 rows and grouped mat-vec on the machine's vector path: the e4m3 bytes, the scale grid,
//! x and y all sit against a guard page, so a read or a write past any of them faults. The output must
//! not move with the thread count, the alignment or the batch width, a grouped task must equal its own
//! rows call, and every row must match an f64 reference built from the e4m3 definition and the block's
//! scale. `guard_scalar` runs the same sweeps pinned to scalar.

mod common;

use btb_native::codes::{ERR_DOMAIN, ERR_NULL};
use btb_native::gemv::MAX_THREADS;
use btb_native::{btb_gemv_fp8_group, btb_gemv_fp8_rows};
use common::fp8k::{self, make_weights, rows_call, Shape, PROJ, RAGGED};
use common::*;

#[test]
fn rows_sweep() {
    fp8k::rows_sweep("vector", &[1, 0, 16]);
}

#[test]
fn grouped_tasks_equal_their_own_rows_calls() {
    fp8k::group_check("vector", &[1, 0, 16]);
}

/// Every row of a batch is exactly what its own single-row call returns.
#[test]
fn batch_rows_equal_their_own_calls() {
    let mut shapes: Vec<Shape> = RAGGED.to_vec();
    shapes.push(PROJ);
    for (idx, &shape) in shapes.iter().enumerate() {
        let (rows, cols, _, _) = shape;
        let wt = make_weights(shape, 0xB47F8 + idx as u64, Align::End);
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

/// A null pointer, a zero dimension, a grid that does not divide the matrix, an overflowing dimension, a
/// thread count past the cap and misaligned x or scales are each refused with a code, and a refused call
/// leaves the output alone.
#[test]
fn bad_pointers_and_shapes_are_error_codes() {
    let (rows, cols) = (4usize, 64usize);
    let m = gen_fp8(rows, cols, 2, 4, 0xBAD);
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];
    let (wp, sp) = (m.w.as_ptr(), m.scales.as_ptr());

    let call = |w: *const u8,
                s: *const f32,
                rows: usize,
                cols: usize,
                sr: usize,
                sc: usize,
                xp: *const f32,
                t: usize,
                yp: *mut f32,
                threads: usize| unsafe {
        btb_gemv_fp8_rows(w, s, rows, cols, sr, sc, xp, t, yp, threads)
    };
    let yp = y.as_mut_ptr();
    let xp = x.as_ptr();

    assert_eq!(
        call(std::ptr::null(), sp, rows, cols, 2, 4, xp, 1, yp, 0),
        ERR_NULL
    );
    assert_eq!(
        call(wp, std::ptr::null(), rows, cols, 2, 4, xp, 1, yp, 0),
        ERR_NULL
    );
    assert_eq!(
        call(wp, sp, rows, cols, 2, 4, std::ptr::null(), 1, yp, 0),
        ERR_NULL
    );
    assert_eq!(
        call(wp, sp, rows, cols, 2, 4, xp, 1, std::ptr::null_mut(), 0),
        ERR_NULL
    );

    assert_eq!(call(wp, sp, 0, cols, 2, 4, xp, 1, yp, 0), ERR_DOMAIN);
    assert_eq!(call(wp, sp, rows, 0, 2, 4, xp, 1, yp, 0), ERR_DOMAIN);
    assert_eq!(call(wp, sp, rows, cols, 2, 4, xp, 0, yp, 0), ERR_DOMAIN);
    assert_eq!(call(wp, sp, rows, cols, 0, 4, xp, 1, yp, 0), ERR_DOMAIN);
    // the grid must divide the matrix evenly
    for (sr, sc) in [(3usize, 4usize), (2, 5), (8, 4), (2, 128)] {
        assert_eq!(
            call(wp, sp, rows, cols, sr, sc, xp, 1, yp, 0),
            ERR_DOMAIN,
            "grid {sr}x{sc} on {rows}x{cols}"
        );
    }
    assert_eq!(
        call(wp, sp, usize::MAX, cols, 1, 4, xp, 1, yp, 0),
        ERR_DOMAIN
    );
    assert_eq!(
        call(wp, sp, rows, cols, 2, 4, xp, 1, yp, MAX_THREADS + 1),
        ERR_DOMAIN
    );

    let bad_x = unsafe { (x.as_ptr() as *const u8).add(1) as *const f32 };
    assert_eq!(call(wp, sp, rows, cols, 2, 4, bad_x, 1, yp, 0), ERR_DOMAIN);
    let bad_s = unsafe { (m.scales.as_ptr() as *const u8).add(1) as *const f32 };
    assert_eq!(call(wp, bad_s, rows, cols, 2, 4, xp, 1, yp, 0), ERR_DOMAIN);

    // the grouped form checks the same things, per task and on the arrays themselves
    let ws = [wp];
    let ss = [sp];
    let rows_a = [rows];
    let cols_a = [cols];
    let sr_a = [2usize];
    let sc_a = [4usize];
    let bad_sc = [5usize];
    let b_a = [1usize];
    let xs = [xp];
    let mut ys = [yp];
    let nul_w: [*const u8; 1] = [std::ptr::null()];
    let zero = [0usize];

    let group = |n: usize,
                 w: *const *const u8,
                 s: *const *const f32,
                 rp: *const usize,
                 scp: *const usize,
                 yp: *mut *mut f32| unsafe {
        btb_gemv_fp8_group(
            n,
            w,
            s,
            rp,
            cols_a.as_ptr(),
            sr_a.as_ptr(),
            scp,
            xs.as_ptr(),
            b_a.as_ptr(),
            yp,
            0,
        )
    };
    let yp_a = ys.as_mut_ptr();
    let (w1, s1, r1, sc1) = (ws.as_ptr(), ss.as_ptr(), rows_a.as_ptr(), sc_a.as_ptr());
    assert_eq!(group(1, std::ptr::null(), s1, r1, sc1, yp_a), ERR_NULL);
    assert_eq!(group(1, w1, std::ptr::null(), r1, sc1, yp_a), ERR_NULL);
    assert_eq!(group(1, nul_w.as_ptr(), s1, r1, sc1, yp_a), ERR_NULL);
    assert_eq!(group(0, w1, s1, r1, sc1, yp_a), ERR_DOMAIN);
    assert_eq!(group(1, w1, s1, zero.as_ptr(), sc1, yp_a), ERR_DOMAIN);
    assert_eq!(group(1, w1, s1, r1, bad_sc.as_ptr(), yp_a), ERR_DOMAIN);

    assert_eq!(
        y,
        vec![0.0f32; rows],
        "a refused call must not write output"
    );
}
