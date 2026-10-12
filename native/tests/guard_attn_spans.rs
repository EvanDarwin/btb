// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced span attention at Qwen4's QSA head (hq 24, hk 2, d 256): prompt chunks over a scattered map
//! after committed rows on both sides of the 64-row chunk, the cache at the tight stride so the last row of
//! the last head ends at the guard page, and the query, map, starts and ends each fenced read-only - the map
//! both against its trailing page and against its leading one, so a read past either end of it faults. Every
//! row is held bit for bit to the decode step over its span's rows copied out. See `common/mod.rs` for the
//! fences.

mod common;

use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_attn_spans_bf16, btb_attn_spans_f32};
use common::*;

// the shared helpers read the generators and the cache through `super::refs` and `super::nodes`
use common::refs;

#[path = "common/attn_nodes.rs"]
mod nodes;

#[path = "common/attn_spans.rs"]
mod spans;

const HQ: usize = 24;
const HK: usize = 2;
const D: usize = 256;

const BASES: &[usize] = &[0, 1, 2, 7, 62, 63, 64, 65, 127, 128, 129, 300, 1000];
const CHUNKS: &[usize] = &[1, 3, 17, 70];

fn fenced<T: Copy>(label: &str, src: &[T], align: Align) -> Fence<T> {
    let mut f = Fence::<T>::new(label, src.len(), align);
    f.copy_from(src);
    f.protect_readonly();
    f
}

#[test]
fn chunks_over_every_base_edge() {
    eprintln!("isa {:?}", isa());
    let scale = 1.0 / (D as f32).sqrt();
    let mut calls = 0usize;
    for (s, &base) in BASES.iter().enumerate() {
        for &t in CHUNKS {
            let n = base + t;
            let n_rows = n + 3; // the pool's spare rows
            let stride = n_rows * D;
            let seed = (s * 37 + t) as u64;
            let k = nodes::cache(HK, n_rows, D, stride, 0xC0FFEE + seed);
            let v = nodes::cache(HK, n_rows, D, stride, 0xBEEF + seed);
            let map = spans::scattered(n, n_rows, 0x3A9 + seed);
            let win = if s % 3 == 2 { 64 } else { 0 };
            let (starts, ends) = spans::chunk(base, t, win);
            let q = gen_f32(t * HQ * D, 0xA77E4 + seed);

            let kb = fenced("k.bf16", &k.bits, Align::End);
            let vb = fenced("v.bf16", &v.bits, Align::End);
            let kf = fenced("k.f32", &k.vals, Align::End);
            let vf = fenced("v.f32", &v.vals, Align::End);
            let qf = fenced("q", &q, Align::End);
            let sf = fenced("starts", &starts, Align::End);
            let ef = fenced("ends", &ends, Align::End);
            let map_end = fenced("map", &map, Align::End);
            let map_start = fenced("map", &map, Align::Start);
            let out_end = Fence::<f32>::new("out", t * HQ * D, Align::End);
            let out_start = Fence::<f32>::new("out", t * HQ * D, Align::Start);

            for threads in [1usize, 0, 16] {
                for bf16 in [false, true] {
                    let steps: Vec<Vec<f32>> = (0..t)
                        .map(|i| {
                            let list = &map[starts[i] as usize..ends[i] as usize];
                            let gk = nodes::gather(&k, HK, D, list, 0);
                            let gv = nodes::gather(&v, HK, D, list, 0);
                            nodes::decode(
                                bf16,
                                &q[i * HQ * D..(i + 1) * HQ * D],
                                &gk,
                                &gv,
                                list.len(),
                                HQ,
                                HK,
                                D,
                                scale,
                                threads,
                            )
                        })
                        .collect();
                    for (end, mf, out) in
                        [(true, &map_end, &out_end), (false, &map_start, &out_start)]
                    {
                        let ctx = format!(
                            "base={base} t={t} win={win} threads={threads} bf16={bf16} align={}",
                            if end { "end" } else { "start" }
                        );
                        // a NaN in every output element, so one the call never writes is seen
                        let dst = out.mut_ptr();
                        unsafe { std::ptr::write_bytes(dst, 0xFF, t * HQ * D) };
                        let rc = unsafe {
                            if bf16 {
                                btb_attn_spans_bf16(
                                    qf.ptr(),
                                    kb.ptr(),
                                    vb.ptr(),
                                    mf.ptr(),
                                    n,
                                    sf.ptr(),
                                    ef.ptr(),
                                    t,
                                    n_rows,
                                    HQ,
                                    HK,
                                    D,
                                    stride,
                                    stride,
                                    scale,
                                    dst,
                                    threads,
                                )
                            } else {
                                btb_attn_spans_f32(
                                    qf.ptr(),
                                    kf.ptr(),
                                    vf.ptr(),
                                    mf.ptr(),
                                    n,
                                    sf.ptr(),
                                    ef.ptr(),
                                    t,
                                    n_rows,
                                    HQ,
                                    HK,
                                    D,
                                    stride,
                                    stride,
                                    scale,
                                    dst,
                                    threads,
                                )
                            }
                        };
                        assert_eq!(rc, OK, "{ctx}: rc {rc}");
                        calls += 1;
                        for f in [&qf, out] {
                            f.check_borders(&ctx);
                        }
                        for f in [&sf, &ef, mf] {
                            f.check_borders(&ctx);
                        }
                        kb.check_borders(&ctx);
                        vb.check_borders(&ctx);
                        kf.check_borders(&ctx);
                        vf.check_borders(&ctx);
                        let got = out.as_slice();
                        for (i, step) in steps.iter().enumerate() {
                            let row = &got[i * HQ * D..(i + 1) * HQ * D];
                            assert!(
                                row.iter().all(|x| x.is_finite()),
                                "{ctx}: row {i} unwritten or read poison"
                            );
                            assert_eq!(
                                bits(row),
                                bits(step),
                                "{ctx}: row {i} against its decode step"
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "[guard] {} bases x {} chunk sizes: {calls} fenced calls clean, every row its decode step",
        BASES.len(),
        CHUNKS.len()
    );
}
