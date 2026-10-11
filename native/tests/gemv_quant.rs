// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The GGUF quant matvecs through the C ABI on plain heap buffers, every format at the tier `isa()`
//! selects: the f64 reference, the bits held across thread counts and batch widths (row `i` of a pass is
//! its own one-row call, the verify pass's contract), and the codes a malformed call returns. The
//! page-fenced sweeps over the same kernels live in `guard_gemv_quant.rs`.

mod common;

use btb_native::codes::{ERR_DOMAIN, ERR_NULL};
use btb_native::gemv::isa;
use common::quant::*;
use common::*;

#[test]
fn every_format_matches_the_f64_reference() {
    eprintln!("isa {:?}", isa());
    for f in FORMATS {
        let t = tables(f, 0xF64);
        let mut shapes = vec![
            (1usize, 256usize),
            (5, 512),
            (12, 512),
            (17, 768),
            (64, 1024),
        ];
        if f.block < 256 {
            shapes.push((3, 3 * f.block)); // a row shorter than one k-quant superblock
        }
        for (idx, &(rows, cols)) in shapes.iter().enumerate() {
            let raw = blocks(f, rows, cols, 0xAB5 + idx as u64);
            let x = gen_f32(cols, 0x71 + idx as u64);
            let (want, scale) = reference(f, &raw, &t, rows, cols, &x);
            for threads in [1usize, 0] {
                let y = run(f, &raw, &t, rows, cols, &x, 1, threads);
                for r in 0..rows {
                    let err = (y[r] as f64 - want[r]).abs() / scale[r];
                    assert!(
                        err < 1e-6,
                        "{} {rows}x{cols} threads={threads} row {r}: got {} want {}",
                        f.name,
                        y[r],
                        want[r]
                    );
                }
            }
        }
    }
}

#[test]
fn threads_and_batch_widths_do_not_move_a_bit() {
    let (rows, cols, b) = (37usize, 768usize, 17usize);
    for f in FORMATS {
        let t = tables(f, 0xB17);
        let raw = blocks(f, rows, cols, 0x1DE);
        let x = gen_f32(b * cols, 0x2DE);
        let base = run(f, &raw, &t, rows, cols, &x, b, 1);
        assert!(
            base.iter().all(|v| v.is_finite()),
            "{}: a non-finite pass certifies nothing",
            f.name
        );
        for threads in [2usize, 5, 24, 0] {
            let got = run(f, &raw, &t, rows, cols, &x, b, threads);
            assert_eq!(bits(&got), bits(&base), "{} threads={threads}", f.name);
        }
        for width in [1usize, 2, 3, 8, 9, 16] {
            let got = run(f, &raw, &t, rows, cols, &x[..width * cols], width, 0);
            assert_eq!(
                bits(&got),
                bits(&base[..width * rows]),
                "{} b={width}",
                f.name
            );
        }
        for i in [0usize, 7, 16] {
            let solo = run(f, &raw, &t, rows, cols, &x[i * cols..(i + 1) * cols], 1, 0);
            assert_eq!(
                bits(&solo),
                bits(&base[i * rows..(i + 1) * rows]),
                "{} row {i} solo",
                f.name
            );
        }
    }
}

#[test]
fn malformed_calls_return_their_codes() {
    let (rows, cols) = (2usize, 256usize);
    for f in FORMATS {
        let t = tables(f, 0xBAD);
        let raw = blocks(f, rows, cols, 0xBAD);
        let x = vec![0.5f32; cols + 1];
        let mut y = vec![0.0f32; rows + 1];
        let (rp, gp, kp, xp, yp) = (
            raw.as_ptr(),
            t.grid.as_ptr(),
            ksigns_ptr(&t),
            x.as_ptr(),
            y.as_mut_ptr(),
        );
        let null_u8 = std::ptr::null::<u8>();
        let code =
            |raw: *const u8,
             grid: *const i8,
             ks: *const u8,
             rows: usize,
             cols: usize,
             x: *const f32,
             b: usize,
             y: *mut f32| unsafe { call_raw(f, raw, grid, ks, rows, cols, x, b, y, 0) };
        let n = f.name;
        assert_eq!(
            code(null_u8, gp, kp, rows, cols, xp, 1, yp),
            ERR_NULL,
            "{n}: null raw"
        );
        assert_eq!(
            code(rp, gp, kp, rows, cols, std::ptr::null(), 1, yp),
            ERR_NULL,
            "{n}: null x"
        );
        assert_eq!(
            code(rp, gp, kp, rows, cols, xp, 1, std::ptr::null_mut()),
            ERR_NULL,
            "{n}: null y"
        );
        if let Call::Latt(..) = f.call {
            assert_eq!(
                code(rp, std::ptr::null(), kp, rows, cols, xp, 1, yp),
                ERR_NULL,
                "{n}: null grid"
            );
            if f.ksigns {
                assert_eq!(
                    code(rp, gp, null_u8, rows, cols, xp, 1, yp),
                    ERR_NULL,
                    "{n}: null ksigns"
                );
            }
        }
        assert_eq!(
            code(rp, gp, kp, rows, f.block + 1, xp, 1, yp),
            ERR_DOMAIN,
            "{n}: a partial block"
        );
        assert_eq!(
            code(rp, gp, kp, 0, cols, xp, 1, yp),
            ERR_DOMAIN,
            "{n}: no rows"
        );
        assert_eq!(
            code(rp, gp, kp, rows, 0, xp, 1, yp),
            ERR_DOMAIN,
            "{n}: no columns"
        );
        assert_eq!(
            code(rp, gp, kp, rows, cols, xp, 0, yp),
            ERR_DOMAIN,
            "{n}: no vectors"
        );
        let x_off = unsafe { (xp as *const u8).add(1) } as *const f32;
        assert_eq!(
            code(rp, gp, kp, rows, cols, x_off, 1, yp),
            ERR_DOMAIN,
            "{n}: a misaligned x"
        );
        let y_off = unsafe { (yp as *mut u8).add(1) } as *mut f32;
        assert_eq!(
            code(rp, gp, kp, rows, cols, xp, 1, y_off),
            ERR_DOMAIN,
            "{n}: a misaligned y"
        );
    }
}

#[test]
fn a_group_is_each_tasks_own_call() {
    // an MoE layer's call: experts of one shape, a down projection's wider rows, a batched task, a lone row
    let shapes = [
        (64usize, 512usize, 1usize),
        (64, 512, 1),
        (512, 256, 1),
        (33, 768, 5),
        (1, 256, 1),
    ];
    for f in FORMATS {
        let t = tables(f, 0x6A0);
        let raws: Vec<Vec<u8>> = (shapes.iter().enumerate())
            .map(|(i, &(rows, cols, _))| blocks(f, rows, cols, 0x6B + i as u64))
            .collect();
        let xs: Vec<Vec<f32>> = (shapes.iter().enumerate())
            .map(|(i, &(_, cols, b))| gen_f32(b * cols, 0x6C + i as u64))
            .collect();
        let tasks: Vec<GroupTask> = (shapes.iter().zip(raws.iter().zip(&xs)))
            .map(|(&(rows, cols, b), (raw, x))| (raw.as_slice(), rows, cols, x.as_slice(), b))
            .collect();
        for threads in [1usize, 3, 0] {
            let ys = run_group(f, &t, &tasks, threads);
            for (i, (y, &(raw, rows, cols, x, b))) in ys.iter().zip(&tasks).enumerate() {
                let solo = run(f, raw, &t, rows, cols, x, b, threads);
                assert_eq!(
                    bits(y),
                    bits(&solo),
                    "{} task {i} threads={threads}",
                    f.name
                );
            }
        }
    }
}

#[test]
fn malformed_groups_return_their_codes() {
    let (rows, cols) = (2usize, 256usize);
    for f in FORMATS {
        let t = tables(f, 0xBAD6);
        let raw = blocks(f, rows, cols, 0xBAD6);
        let x = vec![0.5f32; cols];
        let mut y = vec![0.0f32; rows];
        let yp = y.as_mut_ptr();
        let n = f.name;
        let code = |raw: *const u8, grid: *const i8, ks: *const u8, n: usize, cols: usize| {
            let (raws, xs, mut ys) = ([raw], [x.as_ptr()], [yp]);
            let (rs, cs, bs) = ([rows], [cols], [1usize]);
            unsafe {
                call_group(
                    f,
                    n,
                    raws.as_ptr(),
                    grid,
                    ks,
                    rs.as_ptr(),
                    cs.as_ptr(),
                    xs.as_ptr(),
                    bs.as_ptr(),
                    ys.as_mut_ptr(),
                    0,
                )
            }
        };
        let (rp, gp, kp) = (raw.as_ptr(), t.grid.as_ptr(), ksigns_ptr(&t));
        assert_eq!(code(rp, gp, kp, 0, cols), ERR_DOMAIN, "{n}: no tasks");
        assert_eq!(
            code(std::ptr::null(), gp, kp, 1, cols),
            ERR_NULL,
            "{n}: a null task"
        );
        assert_eq!(
            code(rp, gp, kp, 1, f.block + 1),
            ERR_DOMAIN,
            "{n}: a partial block"
        );
        if let Call::Latt(..) = f.call {
            assert_eq!(
                code(rp, std::ptr::null(), kp, 1, cols),
                ERR_NULL,
                "{n}: null grid"
            );
            if f.ksigns {
                assert_eq!(
                    code(rp, gp, std::ptr::null(), 1, cols),
                    ERR_NULL,
                    "{n}: null ksigns"
                );
            }
        }
    }
}

#[test]
fn dequant_is_the_gemvs_weights() {
    // every weight the f64 reference decodes, to f32 rounding; the bits the same for every thread count
    for f in FORMATS {
        let t = tables(f, 0xDE0);
        for (idx, &(rows, cols)) in [(1usize, 256usize), (13, 512), (40, 768)]
            .iter()
            .enumerate()
        {
            let raw = blocks(f, rows, cols, 0xDE1 + idx as u64);
            let w = run_dequant(f, &raw, &t, rows, cols, 1);
            let per_row = cols / f.block;
            for r in 0..rows {
                for c in 0..cols {
                    let at = (r * per_row + c / f.block) * f.bytes;
                    let want = (f.weight)(&raw[at..at + f.bytes], &t, c % f.block);
                    let got = w[r * cols + c] as f64;
                    assert!(
                        (got - want).abs() <= 1e-6 * want.abs() + 1e-30,
                        "{} {rows}x{cols} [{r}, {c}]: got {got} want {want}",
                        f.name
                    );
                }
            }
            for threads in [3usize, 0] {
                let w2 = run_dequant(f, &raw, &t, rows, cols, threads);
                assert_eq!(bits(&w2), bits(&w), "{} threads={threads}", f.name);
            }
        }
    }
}

#[test]
fn malformed_dequants_return_their_codes() {
    let (rows, cols) = (2usize, 256usize);
    for f in FORMATS {
        let t = tables(f, 0xBAD7);
        let raw = blocks(f, rows, cols, 0xBAD7);
        let mut out = vec![0.0f32; rows * cols + 1];
        let op = out.as_mut_ptr();
        let (rp, gp, kp, n) = (raw.as_ptr(), t.grid.as_ptr(), ksigns_ptr(&t), f.name);
        let code = |raw: *const u8, grid: *const i8, ks: *const u8, cols: usize, out: *mut f32| unsafe {
            call_dequant(f, raw, grid, ks, rows, cols, out, 0)
        };
        assert_eq!(
            code(std::ptr::null(), gp, kp, cols, op),
            ERR_NULL,
            "{n}: null raw"
        );
        assert_eq!(
            code(rp, gp, kp, cols, std::ptr::null_mut()),
            ERR_NULL,
            "{n}: null out"
        );
        assert_eq!(
            code(rp, gp, kp, f.block + 1, op),
            ERR_DOMAIN,
            "{n}: a partial block"
        );
        assert_eq!(code(rp, gp, kp, 0, op), ERR_DOMAIN, "{n}: no columns");
        let off = unsafe { (op as *mut u8).add(1) as *mut f32 };
        assert_eq!(
            code(rp, gp, kp, cols, off),
            ERR_DOMAIN,
            "{n}: a misaligned out"
        );
        if let Call::Latt(..) = f.call {
            assert_eq!(
                code(rp, std::ptr::null(), kp, cols, op),
                ERR_NULL,
                "{n}: null grid"
            );
            if f.ksigns {
                assert_eq!(
                    code(rp, gp, std::ptr::null(), cols, op),
                    ERR_NULL,
                    "{n}: null ksigns"
                );
            }
        }
    }
}
