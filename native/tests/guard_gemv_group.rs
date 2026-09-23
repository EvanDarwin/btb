// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced grouped bf16 mat-vec: many independent tasks under one dispatch, every task's
//! weights and x against a guard page and read-only, every task's y against a guard page of its
//! own. A read past any input faults, a write past any output faults, and each task's output is
//! bit-identical to its own single-task rows call. The unfenced correctness of the same entry
//! point lives in `gemv_group.rs`, and its bf16-rows sibling is fenced in `guard_gemv.rs`.
#![cfg(windows)]

mod common;

use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_gemv_bf16_group, btb_gemv_bf16_rows};
use common::*;

/// `(rows, cols, b)`: two host-linear shapes, a batched one, a sub-tile shape and a single element,
/// so the dispatch splits work across tasks of very different sizes.
const SHAPES: &[(usize, usize, usize)] = &[
    (1280, 2560, 1),
    (2560, 640, 1),
    (1280, 2560, 3),
    (33, 17, 2),
    (1, 1, 1),
    (4, 8, 17),
];

/// A fenced weight matrix, read-only for the test, beside the plain copy the solo call reads.
struct Task {
    rows: usize,
    cols: usize,
    b: usize,
    w: Fence<u16>,
    x: Fence<f32>,
}

fn make_task(rows: usize, cols: usize, b: usize, seed: u64) -> Task {
    let mut w = Fence::<u16>::new("w", rows * cols, Align::End);
    w.copy_from(&gen_w(rows * cols, seed));
    w.protect_readonly();
    let mut x = Fence::<f32>::new("x", b * cols, Align::End);
    x.copy_from(&gen_f32(b * cols, seed ^ 0x1465));
    x.protect_readonly();
    Task {
        rows,
        cols,
        b,
        w,
        x,
    }
}

/// One task's own single-task rows call into a fenced, poisoned output.
fn solo(t: &Task, threads: usize, ctx: &str) -> Vec<f32> {
    let mut y = Fence::<f32>::new("solo", t.b * t.rows, Align::End);
    y.fill(poison_f32());
    let rc = unsafe {
        btb_gemv_bf16_rows(
            t.w.ptr(),
            t.rows,
            t.cols,
            t.x.ptr(),
            t.b,
            y.mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "{ctx}: solo rc {rc}");
    y.check_borders(ctx);
    t.w.check_borders(ctx);
    t.x.check_borders(ctx);
    finish(&y, ctx)
}

/// A fenced output as a plain vector, once its borders and its every-element-written are checked.
fn finish(y: &Fence<f32>, ctx: &str) -> Vec<f32> {
    y.check_borders(ctx);
    let o = y.as_slice();
    if let Some(i) = o.iter().position(|v| is_poison_f32(*v)) {
        panic!("{ctx}: y element {i} was never written");
    }
    o.to_vec()
}

/// Every task of one grouped dispatch, at three thread counts and both output alignments, gets
/// exactly the bits its own single-task call returns, with every buffer against a guard page.
#[test]
fn grouped_tasks_equal_their_own_calls_fenced() {
    eprintln!("isa {:?}", isa());
    let tasks: Vec<Task> = SHAPES
        .iter()
        .enumerate()
        .map(|(i, &(rows, cols, b))| make_task(rows, cols, b, 0x6809 + i as u64))
        .collect();

    for &threads in &[1usize, 0, 16] {
        for align in [Align::End, Align::Start] {
            let ctx = format!("group threads={threads} {align:?}");
            let ys: Vec<Fence<f32>> = tasks
                .iter()
                .map(|t| {
                    let mut f = Fence::<f32>::new("y", t.b * t.rows, align);
                    f.fill(poison_f32());
                    f
                })
                .collect();
            let wp: Vec<*const u16> = tasks.iter().map(|t| t.w.ptr()).collect();
            let xp: Vec<*const f32> = tasks.iter().map(|t| t.x.ptr()).collect();
            let rows: Vec<usize> = tasks.iter().map(|t| t.rows).collect();
            let cols: Vec<usize> = tasks.iter().map(|t| t.cols).collect();
            let bs: Vec<usize> = tasks.iter().map(|t| t.b).collect();
            let mut yp: Vec<*mut f32> = ys.iter().map(|f| f.mut_ptr()).collect();
            let rc = unsafe {
                btb_gemv_bf16_group(
                    tasks.len(),
                    wp.as_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    xp.as_ptr(),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    threads,
                )
            };
            assert_eq!(rc, OK, "{ctx}: rc {rc}");

            for (i, t) in tasks.iter().enumerate() {
                let got = finish(&ys[i], &format!("{ctx} task {i}"));
                t.w.check_borders(&ctx);
                t.x.check_borders(&ctx);
                let want = solo(t, threads, &format!("{ctx} task {i} solo"));
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{ctx}: task {i} ({}x{} b{}) differs from its own call",
                    t.rows,
                    t.cols,
                    t.b
                );
            }
        }
    }
    eprintln!("group of {} tasks fenced, both ends, clean", tasks.len());
}
