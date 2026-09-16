// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The pick of a token from a row of logits on the CPU: greedy (the argmax), or a draw under a temperature
//! with top-k and top-p, the thresholds found by radix select over histograms (three levels of a float's
//! ordered bits, no sort), the masses in 64-bit fixed point, the draw the argmax of the kept tokens' logits
//! plus Gumbel noise hashed from the row's key and the token (the Metal kernel's formula: a draw only moves
//! when two candidates tie to the ulp). Deterministic for a (key, row); rows across the thread pool.

use crate::codes::*;
use rayon::prelude::*;

const BINS: usize = 2048;
const MASS_SCALE: f32 = 1_073_741_824.0; // 2^30 a unit of mass

/// the ordered-uint image of a float: monotone in the value (NaN excluded)
#[inline(always)]
fn ordered(f: f32) -> u32 {
    let u = f.to_bits();
    if u & 0x8000_0000 != 0 {
        !u
    } else {
        u | 0x8000_0000
    }
}

/// exp(t) for t <= 0 to a few ulps: t = n ln2 + r with ln2 in two parts (n's product with the high part exact),
/// e^r through its degree-7 series (|r| <= 0.35: the next term below 1e-8); 0 below -87
#[inline(always)]
fn fast_exp(t: f32) -> f32 {
    if t < -87.0 {
        return 0.0;
    }
    let n = (t * std::f32::consts::LOG2_E).round();
    let r = t - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (0.166_666_67
                    + r * (0.041_666_668
                        + r * (0.008_333_334 + r * (0.001_388_889 + r * 0.000_198_412_7))))));
    let e = (n as i32 + 127).clamp(1, 254);
    f32::from_bits((e as u32) << 23) * p
}

#[derive(Clone, Copy)]
struct Cfg {
    inv_t: f32,
    top_k: u32,
    top_p: f32,
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi as u32
}

/// the radix select over `s` (scaled logits): the threshold u (ordered bits) at which the count of tokens with
/// u' >= u first reaches `need`, over the tokens with u' >= floor
fn select_count(s: &[f32], need: u32, floor: u32) -> u32 {
    let mut prefix = 0u32;
    let mut need = need;
    let mut shift = 32u32;
    let mut cnt = vec![0u32; BINS];
    for level in 0..3 {
        let bits = if level < 2 { 11 } else { 10 };
        shift -= bits;
        cnt.iter_mut().for_each(|c| *c = 0);
        let hi = shift + bits;
        for &v in s {
            let u = ordered(v);
            if u >= floor && (hi >= 32 || (u >> hi) == (prefix >> hi)) {
                cnt[((u >> shift) as usize) & (BINS - 1)] += 1;
            }
        }
        let mut acc = 0u32;
        let mut pick = 0usize;
        for b in (0..BINS).rev() {
            acc += cnt[b];
            if acc >= need {
                pick = b;
                need -= acc - cnt[b];
                break;
            }
        }
        prefix |= (pick as u32) << shift;
    }
    prefix
}

/// the radix select by mass: the threshold u at which the mass of tokens with u' >= u first reaches the target
/// (top_p of the mass over the tokens with u' >= floor); `w` the masses
fn select_mass(s: &[f32], w: &[u64], top_p: f32, floor: u32) -> u32 {
    let mut prefix = 0u32;
    let mut need = 0u64;
    let mut shift = 32u32;
    let mut mass = vec![0u64; BINS];
    for level in 0..3 {
        let bits = if level < 2 { 11 } else { 10 };
        shift -= bits;
        mass.iter_mut().for_each(|c| *c = 0);
        let hi = shift + bits;
        for (&v, &wi) in s.iter().zip(w) {
            let u = ordered(v);
            if u >= floor && (hi >= 32 || (u >> hi) == (prefix >> hi)) {
                mass[((u >> shift) as usize) & (BINS - 1)] += wi;
            }
        }
        if level == 0 {
            let z: u64 = mass.iter().sum();
            need = ((z as f32 * top_p) as u64).max(1); // at least the top token
        }
        let mut acc = 0u64;
        let mut pick = 0usize;
        for b in (0..BINS).rev() {
            acc += mass[b];
            if acc >= need {
                pick = b;
                need -= acc - mass[b];
                break;
            }
        }
        prefix |= (pick as u32) << shift;
    }
    prefix
}

fn pick_row(x: &[f32], key: u64, c: Cfg, s: &mut Vec<f32>, w: &mut Vec<u64>) -> u32 {
    if c.inv_t <= 0.0 {
        return argmax(x);
    }
    let v = x.len();
    s.clear();
    s.extend(x.iter().map(|&a| a * c.inv_t));
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut floor = 0u32;
    if c.top_k > 0 && (c.top_k as usize) < v {
        floor = select_count(s, c.top_k, 0);
    }
    if c.top_p < 1.0 {
        // the masses in fixed point (a token below 2^-30 of the top one carries none), the top-p threshold
        w.clear();
        w.extend(s.iter().map(|&a| {
            if ordered(a) >= floor && a - m >= -21.0 {
                (fast_exp(a - m) * MASS_SCALE) as u64
            } else {
                0
            }
        }));
        floor = floor.max(select_mass(s, w, c.top_p, floor));
    }
    // the draw: argmax of s + gumbel over the kept tokens; a token 21 nats under the top (a 2^-30 chance)
    // is left out of the race
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, &a) in s.iter().enumerate() {
        if ordered(a) < floor || a - m < -21.0 {
            continue;
        }
        let mut h = key ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 32;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
        h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
        h ^= h >> 32;
        // 23 bits: the top of a 24-bit range rounds to 1.0 in f32, an infinite Gumbel that wins the row
        let uf = ((h >> 41) as f32 + 0.5) * (1.0 / 8_388_608.0);
        let v = a - (-uf.ln()).ln();
        if v > best {
            best = v;
            bi = i;
        }
    }
    bi as u32
}

/// # Safety
/// `x` readable for `rows * v` f32, `keys` for `rows` u64, `out` writable for `rows` u32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn sample_pick(
    x: *const f32,
    rows: usize,
    v: usize,
    keys: *const u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    out: *mut u32,
    threads: usize,
) -> i32 {
    if x.is_null() || keys.is_null() || out.is_null() {
        return ERR_NULL;
    }
    if rows == 0
        || v == 0
        || rows.checked_mul(v).is_none()
        || !temperature.is_finite()
        || !top_p.is_finite()
    {
        return ERR_DOMAIN;
    }
    let nt = match crate::gemv::resolve_threads(threads) {
        Some(n) => n,
        None => return ERR_DOMAIN,
    };
    let c = Cfg {
        inv_t: if temperature > 0.0 {
            1.0 / temperature
        } else {
            0.0
        },
        top_k,
        top_p,
    };
    let xs = unsafe { std::slice::from_raw_parts(x, rows * v) };
    let ks = unsafe { std::slice::from_raw_parts(keys, rows) };
    let os = unsafe { std::slice::from_raw_parts_mut(out, rows) };
    let one = |r: usize, o: &mut u32| {
        let mut s = Vec::with_capacity(v);
        let mut w = Vec::with_capacity(v);
        *o = pick_row(&xs[r * v..(r + 1) * v], ks[r], c, &mut s, &mut w);
    };
    if nt <= 1 || rows == 1 {
        for (r, o) in os.iter_mut().enumerate() {
            one(r, o);
        }
    } else {
        os.par_iter_mut().enumerate().for_each(|(r, o)| one(r, o));
    }
    OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(x: &[f32], rows: usize, keys: &[u64], t: f32, k: u32, p: f32) -> Vec<u32> {
        let v = x.len() / rows;
        let mut out = vec![0u32; rows];
        let rc = unsafe {
            sample_pick(
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
        assert_eq!(rc, OK);
        out
    }

    #[test]
    fn greedy_is_the_argmax_lowest_index_first() {
        let x = [1.0f32, 3.0, 3.0, -1.0, 2.0];
        assert_eq!(run(&x, 1, &[0], 0.0, 0, 1.0), vec![1]);
    }

    #[test]
    fn fast_exp_holds_to_a_few_ulps() {
        for i in 0..2000 {
            let t = -(i as f32) * 0.04;
            let a = fast_exp(t);
            let b = t.exp();
            assert!(
                (a - b).abs() <= 4e-7 * b.max(1e-30) + 1e-38,
                "{t}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn the_masks_keep_the_right_tokens_and_a_key_repeats() {
        // probabilities 0.5, 0.25, 0.125, ...: top-p 0.8 keeps three, top-k 2 keeps two
        let x: Vec<f32> = (0..8).map(|i| (0.5f32 / 2f32.powi(i)).ln()).collect();
        let mut seen_p = std::collections::HashSet::new();
        let mut seen_k = std::collections::HashSet::new();
        for key in 0..400u64 {
            seen_p.insert(run(&x, 1, &[key], 1.0, 0, 0.8)[0]);
            seen_k.insert(run(&x, 1, &[key], 1.0, 2, 1.0)[0]);
            assert_eq!(
                run(&x, 1, &[key], 1.0, 0, 0.8),
                run(&x, 1, &[key], 1.0, 0, 0.8)
            );
        }
        assert_eq!(seen_p, [0u32, 1, 2].into_iter().collect());
        assert_eq!(seen_k, [0u32, 1].into_iter().collect());
    }

    #[test]
    fn the_draws_follow_the_distribution() {
        let p = [0.5f32, 0.3, 0.15, 0.05];
        let x: Vec<f32> = p.iter().map(|q| q.ln()).collect();
        let n = 20000u64;
        let mut counts = [0u32; 4];
        for key in 0..n {
            counts[run(&x, 1, &[key], 1.0, 0, 1.0)[0] as usize] += 1;
        }
        for (i, q) in p.iter().enumerate() {
            let f = counts[i] as f32 / n as f32;
            assert!((f - q).abs() < 0.02, "token {i}: {f} vs {q}");
        }
    }

    #[test]
    fn rows_pick_alike_alone_and_together() {
        let v = 3000;
        let mut x = Vec::with_capacity(16 * v);
        let mut st = 7u64;
        for _ in 0..16 * v {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            x.push(((st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 12.0);
        }
        let keys: Vec<u64> = (0..16).collect();
        let together = run(&x, 16, &keys, 0.8, 40, 0.9);
        for r in 0..16 {
            assert_eq!(
                run(&x[r * v..(r + 1) * v], 1, &[keys[r]], 0.8, 40, 0.9)[0],
                together[r]
            );
        }
    }
}
