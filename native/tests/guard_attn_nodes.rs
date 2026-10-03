// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced node-list attention at Qwen4's QSA head (hq 24, hk 2, d 256): draft trees over committed
//! prefixes on both sides of the 64-row chunk, the cache at the tight stride so the last row of the last
//! head ends at the guard page, and the query, offsets and index each fenced read-only - the index both
//! against its trailing page and against its leading one, so a read past either end of a list faults.
//! Every row is held bit for bit to the decode step over its rows copied out. See `common/mod.rs` for the
//! fences.

mod common;

use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::{btb_attn_nodes_bf16, btb_attn_nodes_f32};
use common::*;

// the shared node-list helpers read the generators through `super::refs`
use common::refs;

#[path = "common/attn_nodes.rs"]
mod nodes;

const HQ: usize = 24;
const HK: usize = 2;
const D: usize = 256;

const PREFIXES: &[usize] = &[
    0, 1, 2, 3, 7, 31, 62, 63, 64, 65, 127, 128, 129, 255, 300, 511, 1000,
];
const COUNTS: &[usize] = &[1, 3, 9, 16];

fn fenced<T: Copy>(label: &str, src: &[T], align: Align) -> Fence<T> {
    let mut f = Fence::<T>::new(label, src.len(), align);
    f.copy_from(src);
    f.protect_readonly();
    f
}

#[test]
fn trees_over_every_prefix_edge() {
    eprintln!("isa {:?}", isa());
    let scale = 1.0 / (D as f32).sqrt();
    let mut calls = 0usize;
    for (s, &prefix) in PREFIXES.iter().enumerate() {
        for &count in COUNTS {
            let n_rows = prefix + count;
            let stride = n_rows * D;
            let seed = (s * 31 + count) as u64;
            let k = nodes::cache(HK, n_rows, D, stride, 0xC0FFEE + seed);
            let v = nodes::cache(HK, n_rows, D, stride, 0xBEEF + seed);
            let lists = nodes::tree_lists(prefix, count, 0x7EE + seed);
            let t = lists.len();
            let (offs, idx) = nodes::flatten(&lists, 0);
            let q = gen_f32(t * HQ * D, 0xA77E4 + seed);

            let kb = fenced("k.bf16", &k.bits, Align::End);
            let vb = fenced("v.bf16", &v.bits, Align::End);
            let kf = fenced("k.f32", &k.vals, Align::End);
            let vf = fenced("v.f32", &v.vals, Align::End);
            let qf = fenced("q", &q, Align::End);
            let of = fenced("offs", &offs, Align::End);
            let idx_end = fenced("idx", &idx, Align::End);
            let idx_start = fenced("idx", &idx, Align::Start);
            let out_end = Fence::<f32>::new("out", t * HQ * D, Align::End);
            let out_start = Fence::<f32>::new("out", t * HQ * D, Align::Start);

            for threads in [1usize, 0, 16] {
                for bf16 in [false, true] {
                    let steps: Vec<Vec<f32>> = lists
                        .iter()
                        .enumerate()
                        .map(|(i, l)| {
                            let gk = nodes::gather(&k, HK, D, l, 0);
                            let gv = nodes::gather(&v, HK, D, l, 0);
                            nodes::decode(
                                bf16,
                                &q[i * HQ * D..(i + 1) * HQ * D],
                                &gk,
                                &gv,
                                l.len(),
                                HQ,
                                HK,
                                D,
                                scale,
                                threads,
                            )
                        })
                        .collect();
                    for (end, ix, out) in
                        [(true, &idx_end, &out_end), (false, &idx_start, &out_start)]
                    {
                        let ctx = format!(
                            "prefix={prefix} nodes={count} threads={threads} bf16={bf16} align={}",
                            if end { "end" } else { "start" }
                        );
                        // a NaN in every output element, so one the call never writes is seen
                        let dst = out.mut_ptr();
                        unsafe { std::ptr::write_bytes(dst, 0xFF, t * HQ * D) };
                        let rc = unsafe {
                            if bf16 {
                                btb_attn_nodes_bf16(
                                    qf.ptr(),
                                    kb.ptr(),
                                    vb.ptr(),
                                    of.ptr(),
                                    ix.ptr(),
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
                                btb_attn_nodes_f32(
                                    qf.ptr(),
                                    kf.ptr(),
                                    vf.ptr(),
                                    of.ptr(),
                                    ix.ptr(),
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
                        for f in [&of, ix] {
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
        "[guard] {} prefixes x {} tree sizes: {calls} fenced calls clean, every row its decode step",
        PREFIXES.len(),
        COUNTS.len()
    );
}
