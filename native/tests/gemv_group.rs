// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The grouped bf16 mat-vec through the C ABI: many independent tasks under one thread-pool
//! dispatch. The page-fenced version of the same check lives in `guard_gemv.rs`.

use btb_native::codes::*;
use btb_native::{btb_gemv_bf16_group, btb_gemv_bf16_rows};
use std::time::Instant;

#[path = "common/refs.rs"]
mod refs;

use refs::{bits, gen_f32, gen_w, Rng};

struct Task {
    w: Vec<u16>,
    x: Vec<f32>,
    rows: usize,
    cols: usize,
    b: usize,
}

fn task(rows: usize, cols: usize, b: usize, seed: u64) -> Task {
    Task {
        w: gen_w(rows * cols, seed),
        x: gen_f32(b * cols, seed ^ 0x1465),
        rows,
        cols,
        b,
    }
}

fn random_tasks(n: usize, seed: u64) -> Vec<Task> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let rows = rng.between(64, 1280);
            let cols = rng.between(640, 2560);
            let b = rng.between(1, 3);
            task(
                rows,
                cols,
                b,
                seed.wrapping_add(0x9E37_79B9 * (i as u64 + 1)),
            )
        })
        .collect()
}

fn solo(t: &Task, threads: usize) -> Vec<f32> {
    let mut y = vec![f32::NAN; t.b * t.rows];
    let code = unsafe {
        btb_gemv_bf16_rows(
            t.w.as_ptr(),
            t.rows,
            t.cols,
            t.x.as_ptr(),
            t.b,
            y.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_gemv_bf16_rows returned {code}");
    y
}

fn grouped(tasks: &[Task], threads: usize) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = tasks.iter().map(|t| vec![f32::NAN; t.b * t.rows]).collect();
    let w: Vec<*const u16> = tasks.iter().map(|t| t.w.as_ptr()).collect();
    let x: Vec<*const f32> = tasks.iter().map(|t| t.x.as_ptr()).collect();
    let rows: Vec<usize> = tasks.iter().map(|t| t.rows).collect();
    let cols: Vec<usize> = tasks.iter().map(|t| t.cols).collect();
    let b: Vec<usize> = tasks.iter().map(|t| t.b).collect();
    let mut y: Vec<*mut f32> = out.iter_mut().map(|v| v.as_mut_ptr()).collect();
    let code = unsafe {
        btb_gemv_bf16_group(
            tasks.len(),
            w.as_ptr(),
            rows.as_ptr(),
            cols.as_ptr(),
            x.as_ptr(),
            b.as_ptr(),
            y.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_gemv_bf16_group returned {code}");
    out
}

/// Every task of a group of twelve, at three thread counts, gets exactly the bits its own
/// single-task call returns.
#[test]
fn grouped_tasks_equal_their_own_calls_bit_for_bit() {
    let tasks = random_tasks(12, 0x6809_1465);
    let shapes: Vec<String> = tasks
        .iter()
        .map(|t| format!("{}x{} b{}", t.rows, t.cols, t.b))
        .collect();
    for threads in [1usize, 4, 0] {
        let got = grouped(&tasks, threads);
        for (t, task) in tasks.iter().enumerate() {
            assert_eq!(
                bits(&got[t]),
                bits(&solo(task, threads)),
                "task {t} ({}) threads={threads} differs from its own call",
                shapes[t]
            );
        }
        eprintln!(
            "threads {threads:>2}: {} tasks [{}] bit-identical to their own calls",
            tasks.len(),
            shapes.join(", ")
        );
    }
}

/// A null array, a null entry inside an array, a zero task count and a zero or overflowing
/// dimension are each refused with a code, and a refused call leaves the outputs alone.
#[test]
fn null_and_zero_sizes_are_error_codes_not_guesses() {
    let t = task(4, 8, 1, 0xBAD_5EED);
    let mut y0 = vec![0.0f32; 4];
    let w = [t.w.as_ptr()];
    let x = [t.x.as_ptr()];
    let rows = [4usize];
    let cols = [8usize];
    let b = [1usize];
    let mut y = [y0.as_mut_ptr()];
    let nul_w: [*const u16; 1] = [std::ptr::null()];
    let nul_x: [*const f32; 1] = [std::ptr::null()];
    let mut nul_y: [*mut f32; 1] = [std::ptr::null_mut()];
    let zero = [0usize];
    let huge = [usize::MAX];

    let call =
        |n: usize,
         wp: *const *const u16,
         rp: *const usize,
         cp: *const usize,
         xp: *const *const f32,
         bp: *const usize,
         yp: *mut *mut f32| unsafe { btb_gemv_bf16_group(n, wp, rp, cp, xp, bp, yp, 0) };

    let (rp, cp, bp) = (rows.as_ptr(), cols.as_ptr(), b.as_ptr());
    assert_eq!(
        call(1, std::ptr::null(), rp, cp, x.as_ptr(), bp, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(
            1,
            w.as_ptr(),
            std::ptr::null(),
            cp,
            x.as_ptr(),
            bp,
            y.as_mut_ptr()
        ),
        ERR_NULL
    );
    assert_eq!(
        call(1, w.as_ptr(), rp, cp, x.as_ptr(), bp, std::ptr::null_mut()),
        ERR_NULL
    );
    assert_eq!(
        call(1, nul_w.as_ptr(), rp, cp, x.as_ptr(), bp, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(1, w.as_ptr(), rp, cp, nul_x.as_ptr(), bp, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(1, w.as_ptr(), rp, cp, x.as_ptr(), bp, nul_y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(0, w.as_ptr(), rp, cp, x.as_ptr(), bp, y.as_mut_ptr()),
        ERR_DOMAIN
    );
    for bad in [zero.as_ptr(), huge.as_ptr()] {
        assert_eq!(
            call(1, w.as_ptr(), bad, cp, x.as_ptr(), bp, y.as_mut_ptr()),
            ERR_DOMAIN
        );
        assert_eq!(
            call(1, w.as_ptr(), rp, bad, x.as_ptr(), bp, y.as_mut_ptr()),
            ERR_DOMAIN
        );
    }
    assert_eq!(
        call(
            1,
            w.as_ptr(),
            rp,
            cp,
            x.as_ptr(),
            zero.as_ptr(),
            y.as_mut_ptr()
        ),
        ERR_DOMAIN
    );

    assert_eq!(y0, vec![0.0f32; 4], "a refused call must not write output");
}

/// Twenty tasks under one dispatch return the same bits as twenty separate calls, and reports what
/// the one dispatch cost against the twenty. Ignored by default: the ratio is a measurement, and a
/// debug build's number means nothing.
#[test]
#[ignore]
fn one_dispatch_costs_less_than_twenty() {
    let mut tasks = Vec::with_capacity(20);
    for p in 0..10u64 {
        tasks.push(task(1280, 2560, 1, 0x1D15 + p));
        tasks.push(task(2560, 640, 1, 0x2E26 + p));
    }

    let want: Vec<Vec<u32>> = tasks.iter().map(|t| bits(&solo(t, 0))).collect();
    let got = grouped(&tasks, 0);
    for (t, w) in want.iter().enumerate() {
        assert_eq!(bits(&got[t]), *w, "task {t} differs at the benchmark shape");
    }

    let mut group_s = f64::MAX;
    let mut separate_s = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = grouped(&tasks, 0);
        group_s = group_s.min(t0.elapsed().as_secs_f64());

        let t0 = Instant::now();
        for t in &tasks {
            let _ = solo(t, 0);
        }
        separate_s = separate_s.min(t0.elapsed().as_secs_f64());
    }
    eprintln!(
        "10 [1280x2560]+[2560x640] pairs: grouped {:.2} ms (1 dispatch) vs separate {:.2} ms \
         (20 dispatches) = {:.2}x",
        group_s * 1e3,
        separate_s * 1e3,
        separate_s / group_s
    );
}
