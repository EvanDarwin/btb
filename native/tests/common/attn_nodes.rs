// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The node-list attention checks, shared by the test binary on the path this machine selects
//! (`attn_nodes.rs`) and the one pinned to the scalar path (`attn_nodes_scalar.rs`): an identity list is
//! the decode step bit for bit, a batched call's rows are their own calls bit for bit, and a list is the
//! decode step over its rows copied out in order bit for bit - at every thread count, since a list splits
//! as a decode step of its length splits at that count. Both binaries include it with
//! `#[path = "common/attn_nodes.rs"] mod nodes;` beside `refs`, which it reads through `super`.
//!
//! `dead_code` is allowed because each binary uses only part of it.
#![allow(dead_code)]

use super::refs::{attn_reference, bits, gen_f32, truncate, widen, worst_rel, Rng, POISON_U16};
use btb_native::codes::OK;
use btb_native::{
    btb_attn_decode_bf16, btb_attn_decode_f32, btb_attn_nodes_bf16, btb_attn_nodes_f32,
};

/// The thread counts every bit-identity check runs at: one (a single slot per head), two and seven (the
/// shared pool at odd splits), sixteen (a dedicated pool) and every core.
pub const THREADS: &[usize] = &[1, 2, 7, 16, 0];

/// `(n, hq, hk, d)` for the identity check: Qwen4's QSA head (hq 24, hk 2, d 256) at ragged lengths on
/// both sides of the 64-row minimum chunk, the 8-head GQA shape of `attn.rs`, a head dimension with a
/// SIMD tail (40, 24), and a group past 32 query heads, which takes the heap scratch.
pub const SHAPES: &[(usize, usize, usize, usize)] = &[
    (1, 24, 2, 256),
    (2, 24, 2, 256),
    (63, 24, 2, 256),
    (64, 24, 2, 256),
    (65, 24, 2, 256),
    (129, 24, 2, 256),
    (1037, 24, 2, 256),
    (5, 32, 8, 128),
    (4097, 32, 8, 128),
    (777, 4, 2, 64),
    (300, 6, 6, 128),
    (97, 4, 2, 40),
    (33, 40, 1, 24),
];

/// `(prefix, nodes, hq, hk, d)` for the tree checks: a committed prefix of `prefix` rows followed by
/// `nodes` tree rows, each node attending the prefix, its ancestors and itself.
pub const TREES: &[(usize, usize, usize, usize, usize)] = &[
    (0, 5, 24, 2, 256),
    (1, 8, 24, 2, 256),
    (40, 16, 24, 2, 256),
    (700, 16, 24, 2, 256),
    (3000, 9, 24, 2, 256),
    (130, 12, 32, 8, 128),
    (55, 6, 4, 2, 40),
    (20, 4, 40, 1, 24),
];

pub struct Cache {
    pub bits: Vec<u16>,
    pub vals: Vec<f32>,
    pub stride: usize,
}

/// `rows` rows per head in a buffer of `stride` elements per head. The elements past the rows hold a NaN
/// bit pattern, so a read into the padding makes the output non-finite.
pub fn cache(hk: usize, rows: usize, d: usize, stride: usize, seed: u64) -> Cache {
    let mut rng = Rng::new(seed);
    let mut b = vec![POISON_U16; hk * stride];
    for h in 0..hk {
        for i in 0..rows * d {
            b[h * stride + i] = truncate(rng.f32());
        }
    }
    let vals = b.iter().map(|x| widen(*x)).collect();
    Cache {
        bits: b,
        vals,
        stride,
    }
}

/// The rows of `list` copied out of `c` in list order, `pad` poisoned rows of capacity past them.
pub fn gather(c: &Cache, hk: usize, d: usize, list: &[u32], pad: usize) -> Cache {
    let stride = (list.len() + pad) * d;
    let mut b = vec![POISON_U16; hk * stride];
    for h in 0..hk {
        for (j, &r) in list.iter().enumerate() {
            let src = h * c.stride + r as usize * d;
            b[h * stride + j * d..h * stride + j * d + d].copy_from_slice(&c.bits[src..src + d]);
        }
    }
    let vals = b.iter().map(|x| widen(*x)).collect();
    Cache {
        bits: b,
        vals,
        stride,
    }
}

/// One decode step over the first `n` rows of `k`/`v`.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    bf16: bool,
    q: &[f32],
    k: &Cache,
    v: &Cache,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let mut out = vec![f32::NAN; hq * d];
    let rc = unsafe {
        if bf16 {
            btb_attn_decode_bf16(
                q.as_ptr(),
                k.bits.as_ptr(),
                v.bits.as_ptr(),
                n,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        } else {
            btb_attn_decode_f32(
                q.as_ptr(),
                k.vals.as_ptr(),
                v.vals.as_ptr(),
                n,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        }
    };
    assert_eq!(rc, OK, "btb_attn_decode returned {rc}");
    out
}

/// One node-list call: `q` holds `offs.len() - 1` query rows, row `i` over `idx[offs[i]..offs[i + 1]]`.
#[allow(clippy::too_many_arguments)]
pub fn nodes(
    bf16: bool,
    q: &[f32],
    k: &Cache,
    v: &Cache,
    offs: &[u32],
    idx: &[u32],
    n_rows: usize,
    hq: usize,
    hk: usize,
    d: usize,
    scale: f32,
    threads: usize,
) -> Vec<f32> {
    let t = offs.len() - 1;
    let mut out = vec![f32::NAN; t * hq * d];
    let rc = unsafe {
        if bf16 {
            btb_attn_nodes_bf16(
                q.as_ptr(),
                k.bits.as_ptr(),
                v.bits.as_ptr(),
                offs.as_ptr(),
                idx.as_ptr(),
                t,
                n_rows,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        } else {
            btb_attn_nodes_f32(
                q.as_ptr(),
                k.vals.as_ptr(),
                v.vals.as_ptr(),
                offs.as_ptr(),
                idx.as_ptr(),
                t,
                n_rows,
                hq,
                hk,
                d,
                k.stride,
                v.stride,
                scale,
                out.as_mut_ptr(),
                threads,
            )
        }
    };
    assert_eq!(rc, OK, "btb_attn_nodes returned {rc}");
    assert!(
        out.iter().all(|x| x.is_finite()),
        "a node row read a poisoned row or was never written"
    );
    out
}

/// The lists of a random draft tree over a committed prefix: node `j` (cache row `prefix + j`) hangs off
/// an earlier node or the prefix, and attends the prefix, its ancestors and itself, ascending - the rows
/// its one-token step attends once the path to it is committed.
pub fn tree_lists(prefix: usize, count: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut parent: Vec<Option<usize>> = Vec::with_capacity(count);
    for j in 0..count {
        // about one node in four starts a new branch off the prefix
        parent.push(if j == 0 || rng.below(4) == 0 {
            None
        } else {
            Some(rng.below(j))
        });
    }
    (0..count)
        .map(|j| {
            let mut path = vec![j];
            let mut at = j;
            while let Some(p) = parent[at] {
                path.push(p);
                at = p;
            }
            path.reverse();
            (0..prefix)
                .chain(path.iter().map(|&p| prefix + p))
                .map(|r| r as u32)
                .collect()
        })
        .collect()
}

/// Random lists over `n_rows` rows: sparse ascending subsets, one with repeated rows and one out of order,
/// which the kernel attends in list order as the decode step over that copy would.
pub fn odd_lists(n_rows: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for _ in 0..3 {
        let keep = rng.between(1, 4);
        let list: Vec<u32> = (0..n_rows as u32)
            .filter(|_| rng.below(keep) == 0)
            .collect();
        out.push(if list.is_empty() { vec![0] } else { list });
    }
    out.push(vec![(n_rows - 1) as u32; 3]);
    let mut shuffled: Vec<u32> = (0..n_rows as u32).collect();
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, rng.below(i + 1));
    }
    out.push(shuffled);
    out
}

/// `offs` and `idx` for a batch of lists, the lists starting `lead` entries into `idx` (entries a
/// call never reads, holding a row past the cache so a read of one is refused or seen).
pub fn flatten(lists: &[Vec<u32>], lead: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offs = vec![lead as u32];
    let mut idx = vec![u32::MAX; lead];
    for l in lists {
        idx.extend_from_slice(l);
        offs.push(idx.len() as u32);
    }
    (offs, idx)
}

/// An identity list (`0..n`) is the decode step over `n` rows, bit for bit, in one row and repeated in a
/// batch of three, at every thread count and both element types, with capacity past `n` poisoned.
pub fn check_identity(label: &str) -> usize {
    let mut calls = 0;
    for (s, &(n, hq, hk, d)) in SHAPES.iter().enumerate() {
        let scale = 1.0 / (d as f32).sqrt();
        let k = cache(hk, n, d, (n + 3) * d, 0xC0FFEE + s as u64);
        let v = cache(hk, n, d, (n + 1) * d, 0xBEEF + s as u64);
        let q = gen_f32(3 * hq * d, 0xA77E4 + s as u64);
        let list: Vec<u32> = (0..n as u32).collect();
        let (one_offs, one_idx) = flatten(std::slice::from_ref(&list), 0);
        let (three_offs, three_idx) = flatten(&[list.clone(), list.clone(), list.clone()], 2);
        for &threads in THREADS {
            for bf16 in [false, true] {
                let ctx = format!(
                    "{label} identity n={n} hq={hq} hk={hk} d={d} threads={threads} bf16={bf16}"
                );
                let one = nodes(
                    bf16,
                    &q[..hq * d],
                    &k,
                    &v,
                    &one_offs,
                    &one_idx,
                    n,
                    hq,
                    hk,
                    d,
                    scale,
                    threads,
                );
                let three = nodes(
                    bf16,
                    &q,
                    &k,
                    &v,
                    &three_offs,
                    &three_idx,
                    n,
                    hq,
                    hk,
                    d,
                    scale,
                    threads,
                );
                for i in 0..3 {
                    let row = &q[i * hq * d..(i + 1) * hq * d];
                    let want = decode(bf16, row, &k, &v, n, hq, hk, d, scale, threads);
                    if i == 0 {
                        assert_eq!(bits(&one), bits(&want), "{ctx}: one row");
                    }
                    assert_eq!(
                        bits(&three[i * hq * d..(i + 1) * hq * d]),
                        bits(&want),
                        "{ctx}: row {i} of three"
                    );
                }
                calls += 5;
            }
        }
    }
    eprintln!("[{label}] identity lists: {calls} calls bit-identical to the decode step");
    calls
}

/// Every row of a batched call over a draft tree's lists (and the odd lists) is, bit for bit, its own
/// one-row call, the same batch in reverse order, and the decode step over its rows copied out in list
/// order - at every thread count - and sits within tolerance of the f64 reference and of the one-thread
/// result.
pub fn check_lists(label: &str, reference: bool) -> usize {
    let mut calls = 0;
    for (s, &(prefix, count, hq, hk, d)) in TREES.iter().enumerate() {
        let n_rows = prefix + count;
        let scale = 1.0 / (d as f32).sqrt();
        let k = cache(hk, n_rows, d, (n_rows + 2) * d, 0x1234 + s as u64);
        let v = cache(hk, n_rows, d, n_rows * d, 0x5678 + s as u64);
        let mut lists = tree_lists(prefix, count, 0x7EE + s as u64);
        lists.extend(odd_lists(n_rows, 0x0DD + s as u64));
        let t = lists.len();
        let q = gen_f32(t * hq * d, 0x9E7 + s as u64);
        let (offs, idx) = flatten(&lists, 3);
        let reversed: Vec<Vec<u32>> = lists.iter().rev().cloned().collect();
        let (roffs, ridx) = flatten(&reversed, 0);
        let mut rq = Vec::with_capacity(q.len());
        for i in (0..t).rev() {
            rq.extend_from_slice(&q[i * hq * d..(i + 1) * hq * d]);
        }
        let gathered: Vec<(Cache, Cache)> = lists
            .iter()
            .map(|l| (gather(&k, hk, d, l, 2), gather(&v, hk, d, l, 0)))
            .collect();

        for bf16 in [false, true] {
            let mut first: Option<Vec<f32>> = None;
            for &threads in THREADS {
                let ctx = format!(
                    "{label} tree prefix={prefix} nodes={count} hq={hq} hk={hk} d={d} threads={threads} bf16={bf16}"
                );
                let all = nodes(
                    bf16, &q, &k, &v, &offs, &idx, n_rows, hq, hk, d, scale, threads,
                );
                let back = nodes(
                    bf16, &rq, &k, &v, &roffs, &ridx, n_rows, hq, hk, d, scale, threads,
                );
                calls += 2;
                for (i, list) in lists.iter().enumerate() {
                    let got = &all[i * hq * d..(i + 1) * hq * d];
                    let row = &q[i * hq * d..(i + 1) * hq * d];
                    let (one_offs, one_idx) = flatten(std::slice::from_ref(list), 0);
                    let one = nodes(
                        bf16, row, &k, &v, &one_offs, &one_idx, n_rows, hq, hk, d, scale, threads,
                    );
                    let (gk, gv) = &gathered[i];
                    let step = decode(bf16, row, gk, gv, list.len(), hq, hk, d, scale, threads);
                    calls += 2;
                    let j = t - 1 - i;
                    assert_eq!(bits(got), bits(&one), "{ctx}: row {i} against its own call");
                    assert_eq!(
                        bits(got),
                        bits(&back[j * hq * d..(j + 1) * hq * d]),
                        "{ctx}: row {i} moved with the batch order"
                    );
                    assert_eq!(
                        bits(got),
                        bits(&step),
                        "{ctx}: row {i} against the decode step over its gathered rows"
                    );
                }
                // across thread counts the split moves and the last bits with it, as the decode step's do
                match &first {
                    None => first = Some(all),
                    Some(base) => {
                        let peak = base.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-30);
                        for (a, b) in all.iter().zip(base.iter()) {
                            assert!(
                                ((a - b).abs() / peak) as f64 <= 1e-5,
                                "{ctx}: {a} against {b} at one thread"
                            );
                        }
                    }
                }
            }
            if reference {
                let got = first.as_ref().expect("a run");
                for (i, list) in lists.iter().enumerate() {
                    let (gk, gv) = &gathered[i];
                    let want = attn_reference(
                        &q[i * hq * d..(i + 1) * hq * d],
                        &gk.vals,
                        &gv.vals,
                        list.len(),
                        hq,
                        hk,
                        d,
                        gk.stride,
                        gv.stride,
                        scale,
                    );
                    let e = worst_rel(&got[i * hq * d..(i + 1) * hq * d], &want);
                    assert!(
                        e <= 1e-4,
                        "{label} tree prefix={prefix} bf16={bf16} row {i}: {e:.3e} off the f64 reference"
                    );
                }
            }
        }
    }
    eprintln!("[{label}] node lists: {calls} calls, every row its own call and its decode step bit for bit");
    calls
}
