// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The fenced mat-vec sweeps: bf16 rows, the 12-bit packed rows and the grouped dispatch.

use super::*;
use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_gemv_bf16_group, btb_gemv_bf16_rows, btb_gemv_p12_rows};

/// Every batch width up to the tile count, for the shapes cheap enough to run them all.
pub const T_FULL: [usize; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
/// The tile boundaries only: one row, the first tile's edges, and the last tile's.
pub const T_EDGES: [usize; 8] = [1, 2, 7, 8, 9, 15, 16, 17];
/// Batch widths above the tile set: a prefill hands the kernel a whole prompt at once, up to the
/// row count above which the caller switches to a matmul.
pub const T_DRAFTER: [usize; 6] = [18, 24, 33, 47, 53, 63];
/// `(rows, cols, edges)`: the linear shapes a draft model adds to [`REAL`].
pub const DRAFTER: &[(usize, usize, bool)] =
    &[(1024, 5120, true), (12288, 5120, true), (5120, 10240, true)];

pub struct Weights {
    pub rows: usize,
    pub cols: usize,
    pub w: Fence<u16>,
    pub lo: Fence<u8>,
    pub hi4: Fence<u8>,
    pub table: Fence<u8>,
    pub esc_idx: Fence<i32>,
    pub esc_val: Fence<u8>,
    pub n_esc: usize,
}

pub fn make_weights(rows: usize, cols: usize, seed: u64, esc_rate: f64) -> Weights {
    let raw = gen_w_palette(rows * cols, seed, esc_rate);
    let p = pack_bf16(&raw);
    let mut w = Fence::<u16>::new("w", rows * cols, Align::End);
    w.copy_from(&raw);
    let mut lo = Fence::<u8>::new("lo", p.lo.len(), Align::End);
    lo.copy_from(&p.lo);
    let mut hi4 = Fence::<u8>::new("hi4", p.hi4.len(), Align::End);
    hi4.copy_from(&p.hi4);
    let mut table = Fence::<u8>::new("table", 16, Align::End);
    table.copy_from(&p.table);
    let n_esc = p.esc_idx.len();
    let mut esc_idx = Fence::<i32>::new("esc_idx", n_esc.max(1), Align::End);
    let mut esc_val = Fence::<u8>::new("esc_val", n_esc.max(1), Align::End);
    esc_idx.fill(POISON_I32);
    esc_val.fill(SLACK_POISON);
    if n_esc > 0 {
        esc_idx.copy_from(&p.esc_idx);
        esc_val.copy_from(&p.esc_val);
    }
    for f in [&lo, &hi4, &table, &esc_val] {
        f.protect_readonly();
    }
    w.protect_readonly();
    esc_idx.protect_readonly();
    Weights {
        rows,
        cols,
        w,
        lo,
        hi4,
        table,
        esc_idx,
        esc_val,
        n_esc,
    }
}

impl Weights {
    pub fn check_borders(&self, ctx: &str) {
        self.w.check_borders(ctx);
        self.lo.check_borders(ctx);
        self.hi4.check_borders(ctx);
        self.table.check_borders(ctx);
        self.esc_idx.check_borders(ctx);
        self.esc_val.check_borders(ctx);
    }

    pub fn raw(&self) -> &[u16] {
        self.w.as_slice()
    }
}

fn finish(y: &mut Fence<f32>, ctx: &str) -> Vec<f32> {
    y.check_borders(ctx);
    let o = y.as_slice();
    if let Some(i) = o.iter().position(|v| is_poison_f32(*v)) {
        panic!("{ctx}: y element {i} was never written");
    }
    o.to_vec()
}

pub fn rows_call(
    wt: &Weights,
    x: &Fence<f32>,
    t: usize,
    threads: usize,
    y: &mut Fence<f32>,
    ctx: &str,
) -> Vec<f32> {
    y.fill(poison_f32());
    let rc = unsafe {
        btb_gemv_bf16_rows(
            wt.w.ptr(),
            wt.rows,
            wt.cols,
            x.ptr(),
            t,
            y.mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "{ctx}: bf16 rows rc {rc}");
    wt.check_borders(ctx);
    x.check_borders(ctx);
    finish(y, ctx)
}

pub fn p12_call(
    wt: &Weights,
    x: &Fence<f32>,
    t: usize,
    threads: usize,
    y: &mut Fence<f32>,
    ctx: &str,
) -> Vec<f32> {
    y.fill(poison_f32());
    let (ei, ev) = if wt.n_esc > 0 {
        (wt.esc_idx.ptr(), wt.esc_val.ptr())
    } else {
        (std::ptr::null(), std::ptr::null())
    };
    let rc = unsafe {
        btb_gemv_p12_rows(
            wt.lo.ptr(),
            wt.hi4.ptr(),
            wt.table.ptr(),
            ei,
            ev,
            wt.n_esc,
            wt.rows,
            wt.cols,
            x.ptr(),
            t,
            y.mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "{ctx}: p12 rows rc {rc}");
    wt.check_borders(ctx);
    x.check_borders(ctx);
    finish(y, ctx)
}

/// For every shape, batch width and thread count: bf16 rows into an end-fenced and a start-fenced
/// y (which must agree bit for bit), the packed rows into both (which must equal the bf16 bits),
/// at the first batch width of the set (and at a single row) the f64 reference. The x rows are
/// read-only and end-fenced too.
pub fn rows_sweep(
    label: &str,
    shapes: &[(usize, usize, bool)],
    threads_set: &[usize],
    esc_rate: f64,
) {
    rows_sweep_t(label, shapes, threads_set, esc_rate, None)
}

/// [`rows_sweep`] with the batch widths given explicitly. A `t_override` wins over the shape's
/// `edges` flag, which otherwise picks [`T_EDGES`] over [`T_FULL`].
pub fn rows_sweep_t(
    label: &str,
    shapes: &[(usize, usize, bool)],
    threads_set: &[usize],
    esc_rate: f64,
    t_override: Option<&[usize]>,
) {
    eprintln!("[{label}] isa {:?}", isa());
    let mut calls = 0usize;
    for (idx, &(rows, cols, edges)) in shapes.iter().enumerate() {
        let wt = make_weights(rows, cols, 0x9ACED + idx as u64, esc_rate);
        let ts: &[usize] = t_override.unwrap_or(if edges { &T_EDGES } else { &T_FULL });
        let t_max = ts.iter().copied().max().unwrap_or(1);
        let x_all = gen_f32(t_max * cols, 0x515 + idx as u64);
        let (want, _) = gemv_reference(wt.raw(), rows, cols, &x_all[..cols]);
        let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        for &t in ts {
            let mut x = Fence::<f32>::new("x", t * cols, Align::End);
            x.copy_from(&x_all[..t * cols]);
            x.protect_readonly();
            // y has exactly t * rows elements against the guard page: one per batch width.
            let mut y_end = Fence::<f32>::new("y", t * rows, Align::End);
            let mut y_start = Fence::<f32>::new("y", t * rows, Align::Start);
            for &threads in threads_set {
                let ctx = format!("{label} {rows}x{cols} T={t} threads={threads}");
                let a = rows_call(&wt, &x, t, threads, &mut y_end, &format!("{ctx} bf16/end"));
                let b = rows_call(
                    &wt,
                    &x,
                    t,
                    threads,
                    &mut y_start,
                    &format!("{ctx} bf16/start"),
                );
                assert_eq!(bits(&a), bits(&b), "{ctx}: bf16 alignment changed the bits");
                let c = p12_call(&wt, &x, t, threads, &mut y_end, &format!("{ctx} p12/end"));
                let d = p12_call(
                    &wt,
                    &x,
                    t,
                    threads,
                    &mut y_start,
                    &format!("{ctx} p12/start"),
                );
                assert_eq!(bits(&c), bits(&a), "{ctx}: packed differs from bf16");
                assert_eq!(bits(&d), bits(&a), "{ctx}: packed/start differs from bf16");
                calls += 4;
                if t == ts[0] || t == 1 {
                    // Row 0 of y (input row 0) against the f64 reference.
                    let mut worst = 0.0f64;
                    for r in 0..rows {
                        worst = worst.max((a[r] as f64 - want[r]).abs() / peak);
                    }
                    assert!(worst <= 1e-5, "{ctx}: {worst:.3e} off the f64 reference");
                }
            }
        }
        eprintln!(
            "[{label}] {rows:>6} x {cols:<6} ({} escapes) T {:?} clean",
            wt.n_esc, ts
        );
    }
    eprintln!("[{label}] {calls} fenced calls clean");
}

/// The grouped dispatch: every task's y is its own end-fenced buffer, every input read-only.
pub fn group_check(label: &str, threads_set: &[usize]) {
    eprintln!("[{label}] isa {:?}", isa());
    let shapes: &[(usize, usize, usize)] = &[
        (1280, 2560, 1),
        (2560, 640, 1),
        (2048, 4096, 1),
        (4096, 1024, 1),
        (1280, 2560, 3),
        (33, 17, 2),
        (1, 1, 1),
        (4, 8, 17),
    ];
    let mut ws: Vec<Fence<u16>> = Vec::new();
    let mut xs: Vec<Fence<f32>> = Vec::new();
    let mut raw: Vec<Vec<u16>> = Vec::new();
    for (i, &(rows, cols, b)) in shapes.iter().enumerate() {
        let w = gen_w(rows * cols, 0x6809 + i as u64);
        let mut wf = Fence::<u16>::new("w", rows * cols, Align::End);
        wf.copy_from(&w);
        wf.protect_readonly();
        let mut xf = Fence::<f32>::new("x", b * cols, Align::End);
        xf.copy_from(&gen_f32(b * cols, 0x1465 + i as u64));
        xf.protect_readonly();
        ws.push(wf);
        xs.push(xf);
        raw.push(w);
    }
    for &threads in threads_set {
        for align in [Align::End, Align::Start] {
            let mut ys: Vec<Fence<f32>> = shapes
                .iter()
                .map(|&(rows, _, b)| {
                    let mut f = Fence::<f32>::new("y", b * rows, align);
                    f.fill(poison_f32());
                    f
                })
                .collect();
            let wp: Vec<*const u16> = ws.iter().map(|f| f.ptr()).collect();
            let xp: Vec<*const f32> = xs.iter().map(|f| f.ptr()).collect();
            let rows: Vec<usize> = shapes.iter().map(|s| s.0).collect();
            let cols: Vec<usize> = shapes.iter().map(|s| s.1).collect();
            let bs: Vec<usize> = shapes.iter().map(|s| s.2).collect();
            let mut yp: Vec<*mut f32> = ys.iter().map(|f| f.mut_ptr()).collect();
            let rc = unsafe {
                btb_gemv_bf16_group(
                    shapes.len(),
                    wp.as_ptr(),
                    rows.as_ptr(),
                    cols.as_ptr(),
                    xp.as_ptr(),
                    bs.as_ptr(),
                    yp.as_mut_ptr(),
                    threads,
                )
            };
            let ctx = format!("{label} group threads={threads} {align:?}");
            assert_eq!(rc, OK, "{ctx}: rc {rc}");
            for (i, &(rows, cols, b)) in shapes.iter().enumerate() {
                let got = finish(&mut ys[i], &format!("{ctx} task {i}"));
                ws[i].check_borders(&ctx);
                xs[i].check_borders(&ctx);
                let mut solo = vec![f32::NAN; b * rows];
                let rc = unsafe {
                    btb_gemv_bf16_rows(
                        raw[i].as_ptr(),
                        rows,
                        cols,
                        xs[i].ptr(),
                        b,
                        solo.as_mut_ptr(),
                        threads,
                    )
                };
                assert_eq!(rc, OK);
                assert_eq!(
                    bits(&got),
                    bits(&solo),
                    "{ctx}: task {i} differs from its own call"
                );
            }
        }
    }
    eprintln!("[{label}] group of {} tasks clean", shapes.len());
}
