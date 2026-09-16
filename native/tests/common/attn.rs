// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The fenced attention sweep at the Qwen3.8-27B shape (hq 24, hk 4, d 256).

use super::*;
use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_attn_decode_bf16, btb_attn_decode_f32};

pub const HQ: usize = 24;
pub const HK: usize = 4;
pub const D: usize = 256;
/// The row capacity of the fixed cache buffer, which holds data past `n` so an over-read is seen.
pub const CAP: usize = 8192;
pub const N_MAX: usize = 1200;
const TOL: f64 = 1e-4;

pub struct Kv {
    pub bits: Fence<u16>,
    pub vals: Fence<f32>,
    pub stride: usize,
}

pub fn make_kv(label: &str, n_rows: usize, stride: usize, seed: u64, align: Align) -> Kv {
    let mut rng = Rng::new(seed);
    let mut bits = Fence::<u16>::new(&format!("{label}.bf16"), HK * stride, align);
    let mut vals = Fence::<f32>::new(&format!("{label}.f32"), HK * stride, align);
    bits.fill(POISON_U16);
    vals.fill(poison_f32());
    {
        let b = bits.as_mut_slice();
        let v = vals.as_mut_slice();
        for h in 0..HK {
            for i in 0..n_rows * D {
                let t = truncate(rng.f32());
                b[h * stride + i] = t;
                v[h * stride + i] = widen(t);
            }
        }
    }
    bits.protect_readonly();
    vals.protect_readonly();
    Kv { bits, vals, stride }
}

#[allow(clippy::too_many_arguments)]
pub fn call(
    bf16: bool,
    q: &Fence<f32>,
    k: &Kv,
    v: &Kv,
    n: usize,
    threads: usize,
    out: &mut Fence<f32>,
    ctx: &str,
) -> Vec<f32> {
    out.fill(poison_f32());
    let scale = 1.0 / (D as f32).sqrt();
    let rc = unsafe {
        if bf16 {
            btb_attn_decode_bf16(
                q.ptr(),
                k.bits.ptr(),
                v.bits.ptr(),
                n,
                HQ,
                HK,
                D,
                k.stride,
                v.stride,
                scale,
                out.mut_ptr(),
                threads,
            )
        } else {
            btb_attn_decode_f32(
                q.ptr(),
                k.vals.ptr(),
                v.vals.ptr(),
                n,
                HQ,
                HK,
                D,
                k.stride,
                v.stride,
                scale,
                out.mut_ptr(),
                threads,
            )
        }
    };
    assert_eq!(rc, OK, "{ctx}: rc {rc}");
    out.check_borders(ctx);
    q.check_borders(ctx);
    k.bits.check_borders(ctx);
    k.vals.check_borders(ctx);
    v.bits.check_borders(ctx);
    v.vals.check_borders(ctx);
    let o = out.as_slice();
    if let Some(i) = o.iter().position(|x| is_poison_f32(*x)) {
        panic!("{ctx}: output element {i} was never written");
    }
    o.to_vec()
}

/// `dense`: rebuild K/V at the tight stride for every n (the last row ends at the guard page);
/// otherwise one fixed-capacity buffer as the real cache. `full`: the wide plan (both dtypes,
/// threads 1/0/16, both alignments) at every fifth n plus the ends; the narrow plan always runs
/// f32 threads 1 (both alignments), f32 threads 0 and bf16 threads 1 at every n.
pub fn sweep(label: &str, stride_of: impl Fn(usize) -> usize, dense: bool, full: bool) {
    eprintln!("[{label}] isa {:?}", isa());
    let mut q = Fence::<f32>::new("q", HQ * D, Align::End);
    q.copy_from(&gen_f32(HQ * D, 0xA77E4));
    q.protect_readonly();
    let mut out_end = Fence::<f32>::new("out", HQ * D, Align::End);
    let mut out_start = Fence::<f32>::new("out", HQ * D, Align::Start);

    // The fixed buffer holds data in every row of its capacity so the epilogue can read past N_MAX.
    let fixed = if dense {
        None
    } else {
        let rows = stride_of(N_MAX) / D;
        Some((
            make_kv("k", rows, stride_of(N_MAX), 0xC0FFEE, Align::End),
            make_kv("v", rows, stride_of(N_MAX), 0xBEEF, Align::End),
        ))
    };
    let mut refs = 0usize;
    let mut calls = 0usize;
    for n in 1..=N_MAX {
        let tight;
        let (k, v) = match &fixed {
            Some((k, v)) => (k, v),
            None => {
                tight = (
                    make_kv("k", n, stride_of(n), 0xC0FFEE + n as u64, Align::End),
                    make_kv("v", n, stride_of(n) + 3 * D, 0xBEEF + n as u64, Align::End),
                );
                (&tight.0, &tight.1)
            }
        };
        let sparse = n % 5 == 0 || !(20..=N_MAX - 20).contains(&n);
        let mut plan: Vec<(bool, usize, bool)> = vec![
            (false, 1, true),
            (false, 1, false),
            (false, 0, true),
            (true, 1, true),
        ];
        if sparse && full {
            plan.extend([
                (false, 0, false),
                (true, 0, true),
                (true, 0, false),
                (true, 1, false),
                (false, 16, true),
                (true, 16, true),
            ]);
        }
        let mut base: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for (bf16, threads, end) in plan {
            let ctx = format!(
                "{label} n={n} {} threads={threads} align={}",
                if bf16 { "bf16" } else { "f32" },
                if end { "end" } else { "start" }
            );
            let got = if end {
                call(bf16, &q, k, v, n, threads, &mut out_end, &ctx)
            } else {
                call(bf16, &q, k, v, n, threads, &mut out_start, &ctx)
            };
            calls += 1;
            if threads == 1 {
                let mine = &mut base[bf16 as usize];
                if mine.is_empty() {
                    *mine = got.clone();
                } else {
                    assert_eq!(bits(mine), bits(&got), "{ctx}: alignment changed the bits");
                }
            }
            if n % 97 == 0 || n <= 4 || n >= N_MAX - 2 {
                let want = attn_reference(
                    q.as_slice(),
                    k.vals.as_slice(),
                    v.vals.as_slice(),
                    n,
                    HQ,
                    HK,
                    D,
                    k.stride,
                    v.stride,
                    1.0 / (D as f32).sqrt(),
                );
                let e = worst_rel(&got, &want);
                assert!(e <= TOL, "{ctx}: {e:.3e} off the f64 reference");
                refs += 1;
            }
        }
    }
    if let Some((k, v)) = &fixed {
        // Past the sweep: the capacity edges of the real buffer (2048, 4096, 8192 rows).
        let rows = k.stride / D;
        let mut tail = 0usize;
        for n in [
            N_MAX + 1,
            2047,
            2048,
            2049,
            4095,
            4096,
            4097,
            rows - 1,
            rows,
        ] {
            if n > rows {
                continue;
            }
            for threads in [1usize, 0, 16] {
                for bf16 in [false, true] {
                    for end in [true, false] {
                        let ctx = format!(
                            "{label} n={n} {} threads={threads} align={}",
                            if bf16 { "bf16" } else { "f32" },
                            if end { "end" } else { "start" }
                        );
                        if end {
                            call(bf16, &q, k, v, n, threads, &mut out_end, &ctx);
                        } else {
                            call(bf16, &q, k, v, n, threads, &mut out_start, &ctx);
                        }
                        tail += 1;
                    }
                }
            }
        }
        k.bits.check_borders(label);
        v.bits.check_borders(label);
        calls += tail;
        eprintln!("[{label}] capacity edges up to n={rows}: {tail} fenced calls clean");
    }
    eprintln!(
        "[{label}] n=1..={N_MAX}: {calls} fenced calls clean; {refs} reference checks passed"
    );
}

/// One thread count at every length 1..=N_MAX (f32, the real stride, both alignments).
pub fn every_length_at(label: &str, threads: usize) {
    let mut q = Fence::<f32>::new("q", HQ * D, Align::End);
    q.copy_from(&gen_f32(HQ * D, 0xA77E4));
    q.protect_readonly();
    let mut out_end = Fence::<f32>::new("out", HQ * D, Align::End);
    let mut out_start = Fence::<f32>::new("out", HQ * D, Align::Start);
    let k = make_kv("k", N_MAX, CAP * D, 0xC0FFEE, Align::End);
    let v = make_kv("v", N_MAX, CAP * D, 0xBEEF, Align::End);
    for n in 1..=N_MAX {
        let ctx = format!("{label} n={n} f32 threads={threads}");
        let a = call(
            false,
            &q,
            &k,
            &v,
            n,
            threads,
            &mut out_end,
            &format!("{ctx} end"),
        );
        let b = call(
            false,
            &q,
            &k,
            &v,
            n,
            threads,
            &mut out_start,
            &format!("{ctx} start"),
        );
        assert_eq!(bits(&a), bits(&b), "{ctx}: alignment changed the bits");
        // NaN bit patterns compare equal in `bits`, so finiteness is asserted on its own: the rows
        // past `n` hold poison, and a read into them shows up here.
        assert!(a.iter().all(|x| x.is_finite()), "{ctx} end: non-finite out");
        assert!(
            b.iter().all(|x| x.is_finite()),
            "{ctx} start: non-finite out"
        );
        if n % 5 == 0 {
            let c = call(
                true,
                &q,
                &k,
                &v,
                n,
                threads,
                &mut out_end,
                &format!("{ctx} bf16"),
            );
            assert!(
                c.iter().all(|x| x.is_finite()),
                "{ctx} bf16: non-finite out"
            );
        }
    }
    eprintln!("[{label}] threads={threads} at every n=1..={N_MAX} clean");
}

/// Digests of q/K/V before and after a set of calls: the inputs must come back bit-identical.
pub fn digest_check(threads_set: &[usize]) {
    let mut q = Fence::<f32>::new("q", HQ * D, Align::Start);
    q.copy_from(&gen_f32(HQ * D, 0x51A9E));
    let k = make_kv("k", 300, CAP * D, 0x11, Align::Start);
    let v = make_kv("v", 300, CAP * D, 0x22, Align::Start);
    let before = [
        digest(q.as_slice()),
        digest(k.vals.as_slice()),
        digest(v.vals.as_slice()),
        digest(k.bits.as_slice()),
        digest(v.bits.as_slice()),
    ];
    q.protect_readonly();
    let mut out = Fence::<f32>::new("out", HQ * D, Align::End);
    for n in [1usize, 63, 64, 65, 127, 128, 129, 255, 256, 257, 299, 300] {
        for &threads in threads_set {
            for bf16 in [false, true] {
                let ctx = format!("digest n={n} threads={threads} bf16={bf16}");
                call(bf16, &q, &k, &v, n, threads, &mut out, &ctx);
            }
        }
    }
    q.protect_readwrite();
    k.vals.protect_readwrite();
    v.vals.protect_readwrite();
    k.bits.protect_readwrite();
    v.bits.protect_readwrite();
    let after = [
        digest(q.as_slice()),
        digest(k.vals.as_slice()),
        digest(v.vals.as_slice()),
        digest(k.bits.as_slice()),
        digest(v.bits.as_slice()),
    ];
    assert_eq!(
        before, after,
        "an input changed (q, K f32, V f32, K bf16, V bf16)"
    );
}
