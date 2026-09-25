// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced token sampler: the logits and the per-row keys sit against a guard page and are made
//! read-only, so a read past either end of them faults, and the output tokens sit against a guard
//! page of their own. Both the greedy argmax and the top-k / top-p draw are fenced, at both
//! alignments and three thread counts, and every fenced pick equals the plain unfenced call. The
//! unfenced parity of the same sampler lives in `sample.rs`.

mod common;

use btb_native::btb_sample_pick;
use btb_native::codes::OK;
use btb_native::gemv::isa;
use common::*;

/// `(temperature, top_k, top_p)`: the greedy path and three shapes of the drawing path, which reads
/// the logits again to scale, mask and race them.
const CFGS: &[(f32, u32, f32)] = &[(0.0, 0, 1.0), (1.0, 0, 1.0), (0.8, 40, 0.9), (0.7, 0, 0.8)];

/// `(rows, v)`: a single small row, a vocabulary-sized row, and a few batched rows.
const SHAPES: &[(usize, usize)] = &[(1, 1), (1, 7), (1, 50257), (3, 4001), (16, 257)];

/// The sampler on fenced, read-only inputs into a fenced output. Faults on any read past x or keys;
/// panics if the output was left with its poison sentinel or any slack byte moved.
#[allow(clippy::too_many_arguments)]
fn fenced_call(
    x: &Fence<f32>,
    rows: usize,
    v: usize,
    keys: &Fence<u64>,
    t: f32,
    k: u32,
    p: f32,
    threads: usize,
    out: &mut Fence<u32>,
    ctx: &str,
) -> Vec<u32> {
    out.fill(u32::MAX);
    let rc = unsafe {
        btb_sample_pick(
            x.ptr(),
            rows,
            v,
            keys.ptr(),
            t,
            k,
            p,
            out.mut_ptr(),
            threads,
        )
    };
    assert_eq!(rc, OK, "{ctx}: rc {rc}");
    x.check_borders(ctx);
    keys.check_borders(ctx);
    out.check_borders(ctx);
    let o = out.as_slice().to_vec();
    for (r, &pick) in o.iter().enumerate() {
        assert!(
            (pick as usize) < v,
            "{ctx}: row {r} pick {pick} was never written or is out of range (v={v})"
        );
    }
    o
}

/// The plain, unfenced pick the fenced call must reproduce bit for bit.
fn reference(x: &[f32], rows: usize, v: usize, keys: &[u64], t: f32, k: u32, p: f32) -> Vec<u32> {
    let mut out = vec![u32::MAX; rows];
    let rc = unsafe {
        btb_sample_pick(
            x.as_ptr(),
            rows,
            v,
            keys.as_ptr(),
            t,
            k,
            p,
            out.as_mut_ptr(),
            1,
        )
    };
    assert_eq!(rc, OK, "reference rc {rc}");
    out
}

/// Every shape and config, both alignments of the fenced inputs and output, three thread counts:
/// no read runs off either end of the logits or the keys, the output is fully written between its
/// guard pages, and the picks equal the plain call.
#[test]
fn fenced_inputs_and_output_both_alignments() {
    eprintln!("isa {:?}", isa());
    for &(rows, v) in SHAPES {
        let x_all = gen_f32(rows * v, 0x5A_11_CE + (v as u64));
        let key_vals: Vec<u64> = (0..rows as u64)
            .map(|r| r.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x0001_2345)
            .collect();
        for &(t, k, p) in CFGS {
            let want = reference(&x_all, rows, v, &key_vals, t, k, p);
            for align in [Align::End, Align::Start] {
                let mut x = Fence::<f32>::new("x", rows * v, align);
                x.copy_from(&x_all);
                x.protect_readonly();
                let mut keys = Fence::<u64>::new("keys", rows, align);
                keys.copy_from(&key_vals);
                keys.protect_readonly();
                for &threads in &[1usize, 0, 16] {
                    let ctx = format!("{rows}x{v} t={t} k={k} p={p} threads={threads} {align:?}");
                    let mut out = Fence::<u32>::new("out", rows, align);
                    let got = fenced_call(&x, rows, v, &keys, t, k, p, threads, &mut out, &ctx);
                    assert_eq!(got, want, "{ctx}: fenced picks differ from the plain call");
                }
            }
        }
        eprintln!("{rows:>3} x {v:<6}: logits, keys and output fenced, both ends, clean");
    }
}
