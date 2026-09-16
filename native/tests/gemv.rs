// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The bf16 and 12-bit-packed rows mat-vec through the C ABI on plain heap buffers: one sweep over
//! shape, batch width, thread count and pointer alignment that pins the output bits, the f64
//! reference at every shape, the packer's own round trip, and the codes a malformed call returns.
//! The page-fenced sweeps over the same kernels live in `guard_gemv.rs`.

use btb_native::codes::*;
use btb_native::gemv::MAX_THREADS;
use btb_native::{btb_gemv_bf16_rows, btb_gemv_p12_rows};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

#[path = "common/refs.rs"]
mod refs;

use refs::{
    bits, gemv_reference, gen_f32, gen_w, gen_w_palette, pack_bf16, Packed, REAL, REAL_WIDE,
};

const TOL: f64 = 1e-5;

/// Ragged shapes: odd and sub-tile column counts, a single row, a single column, a single element,
/// and one wide enough to split across threads.
const RAGGED: &[(usize, usize)] = &[
    (1, 1),
    (1, 15),
    (3, 17),
    (5, 31),
    (7, 513),
    (9, 511),
    (16, 16),
    (17, 1025),
    (33, 512),
    (61, 1039),
    (64, 5120),
    (1024, 2053),
];

/// The batch widths the cheap shapes run: one row, the tile edges, and one past the tile count.
const WIDTHS: &[usize] = &[1, 2, 3, 5, 8, 9, 17];
/// The thread counts every arm runs; 0 is every core.
const THREADS: &[usize] = &[1, 2, 7, 24, 0];
/// The offsets, in f32 elements, of the x and y payloads inside their buffers: the same numbers
/// reached through a pointer at a different alignment must give the same bits.
const OFFSETS: &[usize] = &[0, 1];

/// Which weights a shape is built from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    /// bf16 truncations of a uniform [-1, 1): the widest spread of exponents and signs.
    Full,
    /// the same with every sign cleared and x made positive, so no term cancels and the error can
    /// be judged against each row's own magnitude
    Positive,
    /// high bytes drawn from a 15-entry palette with rare escapes: the form the 12-bit packer is for
    Palette,
}

fn seed(rows: usize, cols: usize, kind: Kind, salt: u64) -> u64 {
    (rows as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (cols as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (kind as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
        ^ salt
}

/// The one weight matrix a (shape, kind) has. Every test in this file uses these, so one f64
/// reference serves them all.
fn weights(rows: usize, cols: usize, kind: Kind) -> Vec<u16> {
    let s = seed(rows, cols, kind, 0x0A11_CE00);
    match kind {
        Kind::Full => gen_w(rows * cols, s),
        Kind::Positive => gen_w(rows * cols, s).iter().map(|v| v & 0x7FFF).collect(),
        Kind::Palette => gen_w_palette(rows * cols, s, 1e-4),
    }
}

/// `t` rows of x for a (shape, kind). The first `cols` elements do not depend on `t`, so they are
/// the row the f64 reference is taken over.
fn x_rows(rows: usize, cols: usize, kind: Kind, t: usize) -> Vec<f32> {
    let mut x = gen_f32(t * cols, seed(rows, cols, kind, 0x0B0B_0B0B));
    if kind == Kind::Positive {
        for v in x.iter_mut() {
            *v = v.abs();
        }
    }
    x
}

/// The f64 row sums, the sum of `|term|` per row (a row's own conditioning) and the peak.
struct Ref {
    y: Vec<f64>,
    mag: Vec<f64>,
    peak: f64,
}

type RefTable = Mutex<HashMap<(usize, usize, Kind), Arc<Ref>>>;

static REFS: LazyLock<RefTable> = LazyLock::new(|| Mutex::new(HashMap::new()));

/// The f64 reference for a (shape, kind), computed once and shared by every test in this file.
/// `w` and `x` must be [`weights`] and the first row of [`x_rows`] for that shape and kind.
fn reference(rows: usize, cols: usize, kind: Kind, w: &[u16], x: &[f32]) -> Arc<Ref> {
    let key = (rows, cols, kind);
    if let Some(r) = REFS.lock().unwrap().get(&key) {
        return Arc::clone(r);
    }
    let (y, mag) = gemv_reference(w, rows, cols, &x[..cols]);
    let peak = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let r = Arc::new(Ref { y, mag, peak });
    Arc::clone(REFS.lock().unwrap().entry(key).or_insert(r))
}

/// One row of output against the f64 reference, normwise (against the peak of the whole vector)
/// and against each row's own conditioning, which is the bound that bites on a row that cancels.
fn check_reference(y: &[f32], r: &Ref, ctx: &str) {
    let mut norm = 0.0f64;
    let mut cond = 0.0f64;
    for (i, (&g, &want)) in y.iter().zip(r.y.iter()).enumerate() {
        assert!(g.is_finite(), "{ctx} row {i}: {g} is not finite");
        let e = (g as f64 - want).abs();
        norm = norm.max(e / r.peak);
        cond = cond.max(e / r.mag[i].max(1e-300));
    }
    assert!(norm <= TOL, "{ctx}: normwise {norm:.3e} > {TOL:.0e}");
    assert!(cond <= TOL, "{ctx}: conditioned {cond:.3e} > {TOL:.0e}");
}

/// A buffer whose payload starts `off` f32 elements in, so a call sees a pointer at a different
/// offset inside its cache line while reading and writing the same numbers.
struct Shifted {
    buf: Vec<f32>,
    off: usize,
    len: usize,
}

impl Shifted {
    fn input(src: &[f32], off: usize) -> Shifted {
        let mut buf = vec![0.0f32; off + src.len()];
        buf[off..].copy_from_slice(src);
        Shifted {
            buf,
            off,
            len: src.len(),
        }
    }

    fn output(len: usize, off: usize) -> Shifted {
        Shifted {
            buf: vec![f32::NAN; off + len],
            off,
            len,
        }
    }

    fn ptr(&self) -> *const f32 {
        unsafe { self.buf.as_ptr().add(self.off) }
    }

    fn mut_ptr(&mut self) -> *mut f32 {
        unsafe { self.buf.as_mut_ptr().add(self.off) }
    }

    fn payload(&self) -> &[f32] {
        &self.buf[self.off..self.off + self.len]
    }
}

fn rows_call(
    w: &[u16],
    rows: usize,
    cols: usize,
    xs: &[f32],
    t: usize,
    threads: usize,
    off: usize,
) -> Vec<f32> {
    let x = Shifted::input(&xs[..t * cols], off);
    let mut y = Shifted::output(t * rows, off);
    let code =
        unsafe { btb_gemv_bf16_rows(w.as_ptr(), rows, cols, x.ptr(), t, y.mut_ptr(), threads) };
    assert_eq!(code, OK, "btb_gemv_bf16_rows returned {code}");
    y.payload().to_vec()
}

fn p12_call(
    p: &Packed,
    rows: usize,
    cols: usize,
    xs: &[f32],
    t: usize,
    threads: usize,
    off: usize,
) -> Vec<f32> {
    let x = Shifted::input(&xs[..t * cols], off);
    let mut y = Shifted::output(t * rows, off);
    let code = unsafe {
        btb_gemv_p12_rows(
            p.lo.as_ptr(),
            p.hi4.as_ptr(),
            p.table.as_ptr(),
            p.esc_idx.as_ptr(),
            p.esc_val.as_ptr(),
            p.esc_idx.len(),
            rows,
            cols,
            x.ptr(),
            t,
            y.mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "btb_gemv_p12_rows returned {code}");
    y.payload().to_vec()
}

/// One row of the sweep: a shape, the batch widths, the thread counts and the pointer offsets it
/// runs, and whether every batch row is compared to its own single-row call everywhere or only at
/// the first thread count and offset.
struct Arm {
    rows: usize,
    cols: usize,
    ts: &'static [usize],
    threads: &'static [usize],
    offsets: &'static [usize],
    solo_everywhere: bool,
}

/// The cheap shapes run every batch width, thread count and offset; the host linears run the
/// widths a decode and a small tree actually use, plus an eight-row batch against its own calls.
fn sweep_arms() -> Vec<Arm> {
    let mut arms: Vec<Arm> = RAGGED
        .iter()
        .map(|&(rows, cols)| Arm {
            rows,
            cols,
            ts: WIDTHS,
            threads: THREADS,
            offsets: OFFSETS,
            solo_everywhere: true,
        })
        .collect();
    for &(rows, cols, _) in &REAL[REAL_WIDE] {
        arms.push(Arm {
            rows,
            cols,
            ts: &[1, 4],
            threads: THREADS,
            offsets: OFFSETS,
            solo_everywhere: false,
        });
        arms.push(Arm {
            rows,
            cols,
            ts: &[8],
            threads: &[0],
            offsets: &[0],
            solo_everywhere: true,
        });
    }
    arms
}

/// The only thing allowed to change the output of the rows mat-vec is the input. Over every shape,
/// batch width, thread count and pointer offset the bits are the same, the 12-bit-packed matrix
/// gives the bits of the bf16 one, every row of a batch is what its own single-row call returns,
/// and row 0 matches the f64 reference.
#[test]
fn bits_do_not_move_across_width_threads_or_alignment() {
    for arm in sweep_arms() {
        let (rows, cols) = (arm.rows, arm.cols);
        let w = weights(rows, cols, Kind::Palette);
        let p = pack_bf16(&w);
        let x1 = x_rows(rows, cols, Kind::Palette, 1);
        let r = reference(rows, cols, Kind::Palette, &w, &x1);
        let mut calls = 0usize;

        for &t in arm.ts {
            let xs = x_rows(rows, cols, Kind::Palette, t);
            let mut base: Option<Vec<u32>> = None;
            for (ti, &threads) in arm.threads.iter().enumerate() {
                for (oi, &off) in arm.offsets.iter().enumerate() {
                    let ctx = format!("{rows}x{cols} T={t} threads={threads} x+{off}");
                    let a = rows_call(&w, rows, cols, &xs, t, threads, off);
                    calls += 1;
                    match &base {
                        None => {
                            check_reference(&a[..rows], &r, &ctx);
                            base = Some(bits(&a));
                        }
                        Some(b) => assert_eq!(
                            &bits(&a),
                            b,
                            "{ctx}: the thread count or the alignment moved the bits"
                        ),
                    }
                    let want = base.as_ref().unwrap();
                    let c = p12_call(&p, rows, cols, &xs, t, threads, off);
                    calls += 1;
                    assert_eq!(&bits(&c), want, "{ctx}: packed differs from bf16");

                    if arm.solo_everywhere || (ti == 0 && oi == 0) {
                        for i in 0..t {
                            let row = &xs[i * cols..(i + 1) * cols];
                            let solo = rows_call(&w, rows, cols, row, 1, threads, off);
                            let solo12 = p12_call(&p, rows, cols, row, 1, threads, off);
                            calls += 2;
                            assert_eq!(
                                bits(&a[i * rows..(i + 1) * rows]),
                                bits(&solo),
                                "{ctx}: batch row {i} differs from its own call"
                            );
                            assert_eq!(
                                bits(&solo12),
                                bits(&solo),
                                "{ctx}: packed row {i} differs from bf16"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "{rows:>6} x {cols:<6} ({:>6} escapes) T {:?} threads {:?} x+{:?}: {calls} calls, \
             one set of bits",
            p.esc_idx.len(),
            arm.ts,
            arm.threads,
            arm.offsets
        );
    }
}

/// Every shape against the f64 reference twice: once over the full spread of bf16 exponents and
/// signs, where rows cancel, and once with nothing negative, where each row's own magnitude is the
/// bound.
#[test]
fn matches_the_f64_reference_on_every_shape() {
    let mut shapes: Vec<(usize, usize)> = RAGGED.to_vec();
    shapes.extend(REAL[REAL_WIDE].iter().map(|&(r, c, _)| (r, c)));

    for &(rows, cols) in &shapes {
        for kind in [Kind::Full, Kind::Positive] {
            let w = weights(rows, cols, kind);
            let xs = x_rows(rows, cols, kind, 1);
            let r = reference(rows, cols, kind, &w, &xs);
            for threads in [1usize, 0] {
                let y = rows_call(&w, rows, cols, &xs, 1, threads, 0);
                check_reference(&y, &r, &format!("{rows}x{cols} {kind:?} threads={threads}"));
            }
            eprintln!("{rows:>6} x {cols:<6} {kind:?}: within {TOL:.0e}");
        }
    }
}

/// A one-row and a one-column matrix of exact bf16 values give exactly the hand-computed answer.
#[test]
fn a_single_row_and_a_single_column_still_work() {
    let w = [
        (1.5f32.to_bits() >> 16) as u16,
        (2.5f32.to_bits() >> 16) as u16,
    ];

    let x = [2.0f32, 4.0];
    assert_eq!(rows_call(&w, 1, 2, &x, 1, 24, 0), vec![3.0 + 10.0]);

    let x = [3.0f32];
    assert_eq!(rows_call(&w, 2, 1, &x, 1, 24, 0), vec![4.5, 7.5]);
}

/// The 12-bit packer itself: every weight comes back out of the codes, the table and the escape
/// list; palette-only data escapes nothing; and the packed form is about three quarters the size.
#[test]
fn packer_round_trips_and_escapes_only_what_it_must() {
    for (n, rate) in [
        (1usize, 0.0),
        (2, 0.0),
        (17, 0.0),
        (5000, 0.0),
        (5000, 1e-2),
    ] {
        let w = gen_w_palette(n, 42 + n as u64, rate);
        let p = pack_bf16(&w);
        assert_eq!(p.lo.len(), n);
        assert_eq!(p.hi4.len(), n.div_ceil(2));
        assert!(p.esc_idx.windows(2).all(|s| s[0] < s[1]));
        if rate == 0.0 {
            assert!(p.esc_idx.is_empty(), "palette-only data must not escape");
        } else {
            assert!(!p.esc_idx.is_empty(), "rare bytes must escape");
        }

        for (i, &want) in w.iter().enumerate() {
            let byte = p.hi4[i >> 1];
            let code = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let mut hi = p.table[code as usize];
            if let Ok(k) = p.esc_idx.binary_search(&(i as i32)) {
                hi = p.esc_val[k];
            }
            assert_eq!(((hi as u16) << 8) | p.lo[i] as u16, want, "weight {i}");
        }
    }

    let w = gen_w_palette(1 << 20, 7, 1e-4);
    let p = pack_bf16(&w);
    let ratio = p.bytes() as f64 / (w.len() * 2) as f64;
    assert!(
        ratio < 0.76,
        "packed/bf16 byte ratio {ratio:.4} is not ~0.75"
    );
}

/// A null pointer, a zero or overflowing dimension, a thread count past the cap and a pointer that
/// is not aligned for its element type are each refused with a code, and a refused call leaves the
/// output buffer alone.
#[test]
fn null_and_zero_sizes_are_error_codes_not_guesses() {
    let w = [0u16; 4];
    let x = [1.0f32; 4];
    let mut y = [0.0f32; 4];
    unsafe {
        assert_eq!(
            btb_gemv_bf16_rows(std::ptr::null(), 2, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_NULL
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, std::ptr::null(), 1, y.as_mut_ptr(), 0),
            ERR_NULL
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, x.as_ptr(), 1, std::ptr::null_mut(), 0),
            ERR_NULL
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 0, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 0, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, x.as_ptr(), 0, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, std::ptr::null(), 2, y.as_mut_ptr(), 0),
            ERR_NULL
        );

        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), usize::MAX, 4, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, usize::MAX, x.as_ptr(), 4, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );

        // a thread count past the cap is refused rather than spawning that many OS threads
        assert_eq!(
            btb_gemv_bf16_rows(
                w.as_ptr(),
                2,
                2,
                x.as_ptr(),
                1,
                y.as_mut_ptr(),
                MAX_THREADS + 1
            ),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, x.as_ptr(), 1, y.as_mut_ptr(), usize::MAX),
            ERR_DOMAIN
        );

        // the scalar and tail paths dereference elements directly: a byte-offset pointer is refused
        let bad_x = (x.as_ptr() as *const u8).add(1) as *const f32;
        let bad_y = (y.as_mut_ptr() as *mut u8).add(1) as *mut f32;
        let bad_w = (w.as_ptr() as *const u8).add(1) as *const u16;
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, bad_x, 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(w.as_ptr(), 2, 2, x.as_ptr(), 1, bad_y, 0),
            ERR_DOMAIN
        );
        assert_eq!(
            btb_gemv_bf16_rows(bad_w, 2, 2, x.as_ptr(), 1, y.as_mut_ptr(), 0),
            ERR_DOMAIN
        );
    }

    assert_eq!(y, [0.0f32; 4]);
}

/// The packed entry point refuses the same malformed calls, plus an escape list that is not
/// strictly ascending, holds a duplicate, or points outside the matrix.
#[test]
fn packed_rejects_bad_pointers_shapes_and_escape_lists() {
    let w = gen_w_palette(64, 5, 0.0);
    let p = pack_bf16(&w);
    let x = [1.0f32; 8];
    let mut y = [0.0f32; 8];
    let (l, h, t) = (p.lo.as_ptr(), p.hi4.as_ptr(), p.table.as_ptr());
    let nul32: *const i32 = std::ptr::null();
    let nul8: *const u8 = std::ptr::null();
    let call = |lo: *const u8,
                hi4: *const u8,
                tbl: *const u8,
                ei: *const i32,
                ev: *const u8,
                ne: usize,
                rows: usize,
                cols: usize,
                b: usize,
                yp: *mut f32| unsafe {
        btb_gemv_p12_rows(lo, hi4, tbl, ei, ev, ne, rows, cols, x.as_ptr(), b, yp, 0)
    };

    assert_eq!(
        call(nul8, h, t, nul32, nul8, 0, 8, 8, 1, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(l, nul8, t, nul32, nul8, 0, 8, 8, 1, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(l, h, nul8, nul32, nul8, 0, 8, 8, 1, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(l, h, t, nul32, nul8, 0, 8, 8, 1, std::ptr::null_mut()),
        ERR_NULL
    );

    assert_eq!(
        call(l, h, t, nul32, nul8, 3, 8, 8, 1, y.as_mut_ptr()),
        ERR_NULL
    );
    assert_eq!(
        call(l, h, t, nul32, nul8, 0, 0, 8, 1, y.as_mut_ptr()),
        ERR_DOMAIN
    );
    assert_eq!(
        call(l, h, t, nul32, nul8, 0, 8, 0, 1, y.as_mut_ptr()),
        ERR_DOMAIN
    );
    assert_eq!(
        call(l, h, t, nul32, nul8, 0, 8, 8, 0, y.as_mut_ptr()),
        ERR_DOMAIN
    );

    let vals: [u8; 2] = [0x35, 0x35];
    let bad_desc: [i32; 2] = [5, 3];
    let bad_dup: [i32; 2] = [3, 3];
    let bad_range: [i32; 1] = [64];
    let bad_neg: [i32; 1] = [-1];
    for idx in [&bad_desc[..], &bad_dup[..], &bad_range[..], &bad_neg[..]] {
        assert_eq!(
            call(
                l,
                h,
                t,
                idx.as_ptr(),
                vals.as_ptr(),
                idx.len(),
                8,
                8,
                1,
                y.as_mut_ptr()
            ),
            ERR_DOMAIN,
            "escape list {idx:?} should be refused"
        );
    }
    assert_eq!(y, [0.0f32; 8], "a refused call must not write output");
}
