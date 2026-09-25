// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The platform-independent half of the test scaffolding: the deterministic generators, the f64
//! references every kernel is measured against, the error measures, the shape tables and the
//! 12-bit packer. Nothing here touches the operating system, so the cross-platform test files
//! include it directly with `#[path = "common/refs.rs"] mod refs;` while the fenced tests reach
//! the same items through `common`, which re-exports them.
//!
//! `dead_code` is allowed because every test binary compiles this module on its own and uses only
//! the part of it that it needs.
#![allow(dead_code)]

// ---------------------------------------------------------------- poison values

/// A quiet NaN written into a buffer before a call: an element that comes back with these bits was
/// never written, and one that leaks out of padding makes the output non-finite.
pub const POISON_F32_BITS: u32 = 0x7FC0_BAD0;
/// The bf16 half of the same idea: a NaN bit pattern for the slack of a bf16 cache.
pub const POISON_U16: u16 = 0xFFC1;

pub fn poison_f32() -> f32 {
    f32::from_bits(POISON_F32_BITS)
}

pub fn is_poison_f32(v: f32) -> bool {
    v.to_bits() == POISON_F32_BITS
}

// ---------------------------------------------------------------- generators

/// xorshift64, f32 in [-1, 1).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }

    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }

    pub fn uni(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / 16_777_216.0)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A uniform integer in `lo..=hi`.
    pub fn between(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
}

pub fn truncate(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

pub fn widen(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

/// FNV-1a over the bytes of a slice: a cheap "did anything change" receipt for big inputs.
pub fn digest<T: Copy>(v: &[T]) -> u64 {
    let bytes =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

pub fn gen_w(n: usize, seed: u64) -> Vec<u16> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| truncate(rng.f32())).collect()
}

pub fn gen_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| rng.f32()).collect()
}

/// bf16 weights whose high bytes come from a 15-entry palette plus rare escapes at `esc_rate`.
pub fn gen_w_palette(n: usize, seed: u64, esc_rate: f64) -> Vec<u16> {
    const PALETTE: [u16; 15] = [
        0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1, 0xC2,
    ];
    const RARE: [u16; 4] = [0x35, 0xB5, 0x47, 0xC7];
    let mut rng = Rng::new(seed);
    let cut = (esc_rate * (1u64 << 32) as f64) as u64;
    (0..n)
        .map(|_| {
            let roll = rng.next_u64() >> 32;
            let hi = if esc_rate > 0.0 && roll < cut {
                RARE[rng.below(RARE.len())]
            } else {
                PALETTE[rng.below(PALETTE.len())]
            };
            (hi << 8) | ((rng.next_u64() & 0xFF) as u16)
        })
        .collect()
}

// ---------------------------------------------------------------- shape tables

/// `(rows, cols, edges)`: the host linear shapes. `edges` marks the shapes wide enough that the
/// batch-width sweep runs the tile boundaries only instead of every width. The five entries at
/// `REAL[2..7]` are the widest, and are the ones the cross-platform sweep runs.
pub const REAL: &[(usize, usize, bool)] = &[
    (48, 5120, false),
    (1024, 5120, false),
    (5120, 6144, false),
    (6144, 5120, true),
    (10240, 5120, true),
    (5120, 17408, true),
    (17408, 5120, true),
    (12288, 5120, true),
];

/// The index range of [`REAL`] holding the widest shapes.
pub const REAL_WIDE: std::ops::Range<usize> = 2..7;

/// Ragged shapes: odd cols (the odd-start scalar fallback of the packed widen), sub-tile cols,
/// a single element.
pub const ODD: &[(usize, usize, bool)] = &[
    (37, 1023, false),
    (5, 4097, false),
    (13, 17, false),
    (1, 1, false),
    (3, 513, false),
];

// ---------------------------------------------------------------- attention reference

#[allow(clippy::too_many_arguments)]
pub fn attn_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    key_stride: usize,
    value_stride: usize,
    scale: f32,
) -> Vec<f64> {
    let group = hq / hk;
    let mut out = vec![0.0f64; hq * d];
    let mut weight = vec![0.0f64; n];
    for j in 0..hq {
        let h = j / group;
        let q = &query[j * d..j * d + d];
        let mut top = f64::NEG_INFINITY;
        for (r, w) in weight.iter_mut().enumerate() {
            let row = &key[h * key_stride + r * d..h * key_stride + r * d + d];
            let mut dot = 0.0f64;
            for (a, b) in q.iter().zip(row.iter()) {
                dot += *a as f64 * *b as f64;
            }
            *w = dot * scale as f64;
            if *w > top {
                top = *w;
            }
        }
        let mut total = 0.0f64;
        for w in weight.iter_mut() {
            *w = (*w - top).exp();
            total += *w;
        }
        let dst = &mut out[j * d..j * d + d];
        for (r, w) in weight.iter().enumerate() {
            let p = *w / total;
            let row = &value[h * value_stride + r * d..h * value_stride + r * d + d];
            for (o, x) in dst.iter_mut().zip(row.iter()) {
                *o += p * *x as f64;
            }
        }
    }
    out
}

/// The worst absolute error against the peak of the reference. Panics on a non-finite output, so a
/// read past the end of the data (whose padding is NaN) is caught here rather than swallowed by a
/// comparison NaN loses.
pub fn worst_rel(got: &[f32], want: &[f64]) -> f64 {
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let mut err = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(g.is_finite(), "output {g} is not finite");
        err = err.max((*g as f64 - *w).abs());
    }
    err / peak
}

/// The worst error normwise (against the peak) and componentwise (on the components above 1% of
/// the peak). Panics on a non-finite output.
pub fn err_pair(got: &[f32], want: &[f64], label: &str) -> (f64, f64) {
    let peak = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let mut norm = 0.0f64;
    let mut comp = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        assert!(g.is_finite(), "{label}: non-finite output {g}");
        let e = (*g as f64 - w).abs();
        norm = norm.max(e / peak);
        if w.abs() > 0.01 * peak {
            comp = comp.max(e / w.abs());
        }
    }
    (norm, comp)
}

// ---------------------------------------------------------------- gemv reference and packer

/// `(y, mag)`: the f64 row sums, and the sum of `|term|` per row, which is the conditioning the
/// f32 kernel's error has to be judged against on a row that cancels.
pub fn gemv_reference(w: &[u16], rows: usize, cols: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(rows);
    let mut mag = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        let mut sum = 0.0f64;
        for c in 0..cols {
            let t = widen(row[c]) as f64 * x[c] as f64;
            acc += t;
            sum += t.abs();
        }
        y.push(acc);
        mag.push(sum);
    }
    (y, mag)
}

pub struct Packed {
    pub lo: Vec<u8>,
    pub hi4: Vec<u8>,
    pub table: [u8; 16],
    pub esc_idx: Vec<i32>,
    pub esc_val: Vec<u8>,
}

impl Packed {
    /// The bytes the packed form occupies: the low bytes, the nibble codes, the table and five
    /// bytes per escape (an i32 index and a u8 value).
    pub fn bytes(&self) -> usize {
        self.lo.len() + self.hi4.len() + 16 + self.esc_idx.len() * 5
    }
}

/// The 12-bit packer, mirroring `btb.engine.pack_bf16` (15 most frequent high bytes coded,
/// code 15 escapes; weight 2k in the low nibble of byte k; the table has 16 entries, the last 0).
pub fn pack_bf16(w: &[u16]) -> Packed {
    let mut counts = [0u64; 256];
    for &v in w {
        counts[(v >> 8) as usize] += 1;
    }
    let mut order: Vec<usize> = (0..256).filter(|&i| counts[i] > 0).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(counts[i]), i));
    order.truncate(15);
    let mut lut = [15u8; 256];
    let mut table = [0u8; 16];
    for (c, &byte) in order.iter().enumerate() {
        lut[byte] = c as u8;
        table[c] = byte as u8;
    }
    let n = w.len();
    let mut lo = Vec::with_capacity(n);
    let mut code = Vec::with_capacity(n + 1);
    let mut esc_idx = Vec::new();
    let mut esc_val = Vec::new();
    for (i, &v) in w.iter().enumerate() {
        lo.push((v & 0xFF) as u8);
        let hi = (v >> 8) as u8;
        let c = lut[hi as usize];
        code.push(c);
        if c == 15 {
            esc_idx.push(i as i32);
            esc_val.push(hi);
        }
    }
    if code.len() % 2 == 1 {
        code.push(0);
    }
    let hi4: Vec<u8> = code
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| p[0] | (p[1] << 4))
        .collect();
    Packed {
        lo,
        hi4,
        table,
        esc_idx,
        esc_val,
    }
}

// ---------------------------------------------------------------- MXFP4 reference

/// 32 weights along `cols` share one 16-byte block (two fp4 e2m1 codes per byte, the earlier
/// weight in the LOW nibble) and one uint8 e8m0 exponent.
pub const MX_BLOCK: usize = 32;
pub const MX_BLOCK_BYTES: usize = 16;

/// The sixteen e2m1 values, in code order.
pub const FP4: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub struct Mxfp4 {
    pub blocks: Vec<u8>,
    pub scales: Vec<u8>,
}

/// An MXFP4 matrix of `rows x cols` (cols a multiple of 32): every nibble occurs, and the block
/// exponents run either side of 127, so both halves of every byte and both directions of the
/// exact-power-of-two scale are exercised without spreading the row sum over so many decades that
/// the f32 accumulation, not the widening, sets the error.
pub fn gen_mxfp4(rows: usize, cols: usize, seed: u64) -> Mxfp4 {
    assert!(
        cols.is_multiple_of(MX_BLOCK),
        "cols must be a multiple of 32"
    );
    let groups = rows * cols / MX_BLOCK;
    let mut rng = Rng::new(seed);
    let blocks: Vec<u8> = (0..groups * MX_BLOCK_BYTES)
        .map(|_| (rng.next_u64() & 0xFF) as u8)
        .collect();
    let scales: Vec<u8> = (0..groups).map(|_| (124 + rng.below(7)) as u8).collect();
    Mxfp4 { blocks, scales }
}

/// `2^(s - 127)` in f64, exactly, including `s == 0`.
pub fn mx_scale(s: u8) -> f64 {
    (2.0f64).powi(s as i32 - 127)
}

/// Weight `i` of an MXFP4 matrix: `fp4(code_i) * 2^(scales[i / 32] - 127)`.
pub fn mxfp4_weight(m: &Mxfp4, i: usize) -> f64 {
    let g = i / MX_BLOCK;
    let j = i % MX_BLOCK;
    let byte = m.blocks[g * MX_BLOCK_BYTES + j / 2];
    let code = if j.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    };
    FP4[code as usize] * mx_scale(m.scales[g])
}

/// `(y, mag)` for an MXFP4 matrix, as [`gemv_reference`] is for bf16.
pub fn mxfp4_reference(m: &Mxfp4, rows: usize, cols: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut y = Vec::with_capacity(rows);
    let mut mag = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = 0.0f64;
        let mut sum = 0.0f64;
        for (c, &xc) in x.iter().enumerate().take(cols) {
            let t = mxfp4_weight(m, r * cols + c) * xc as f64;
            acc += t;
            sum += t.abs();
        }
        y.push(acc);
        mag.push(sum);
    }
    (y, mag)
}

// ---------------------------------------------------------------- fp8 reference

/// An FP8 matrix: `rows * cols` e4m3fn bytes and an `[sr, sc]` grid of f32 scales.
pub struct Fp8 {
    pub w: Vec<u8>,
    pub scales: Vec<f32>,
    pub sr: usize,
    pub sc: usize,
}

/// An FP8 matrix whose bytes take every finite code (never 0x7F/0xFF, the NaNs a quantizer does not
/// write) and whose scales stay within a decade, so the f32 accumulation, not the widening, sets the error.
pub fn gen_fp8(rows: usize, cols: usize, sr: usize, sc: usize, seed: u64) -> Fp8 {
    assert!(
        rows.is_multiple_of(sr) && cols.is_multiple_of(sc),
        "the grid must divide the matrix"
    );
    let mut rng = Rng::new(seed);
    let w: Vec<u8> = (0..rows * cols)
        .map(|_| {
            let v = (rng.next_u64() & 0xFF) as u8;
            if v & 0x7F == 0x7F {
                v ^ 1
            } else {
                v
            }
        })
        .collect();
    let scales: Vec<f32> = (0..sr * sc)
        .map(|_| (1 + rng.below(10)) as f32 / 4480.0)
        .collect();
    Fp8 { w, scales, sr, sc }
}

/// An e4m3fn byte's value from the definition: (-1)^s * 2^(e-7) * (1 + m/8), and 2^-6 * m/8 at e == 0.
pub fn e4m3(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = (((b >> 3) & 0x0F) as i32, (b & 7) as f64);
    if e == 0 {
        sign * 2f64.powi(-6) * m / 8.0
    } else {
        sign * 2f64.powi(e - 7) * (1.0 + m / 8.0)
    }
}

/// `p` to the nearest bf16 (8 significant bits, ties to even), by arithmetic rather than the kernel's bits.
pub fn bf16_nearest(p: f64) -> f64 {
    if p == 0.0 {
        return p;
    }
    let ulp = 2f64.powi(p.abs().log2().floor().max(-126.0) as i32 - 7);
    (p / ulp).round_ties_even() * ulp
}

/// Weight `i` of an FP8 matrix as the kernel holds it: `bf16(e4m3 * scale)`, the product in f32.
pub fn fp8_weight(m: &Fp8, bm: usize, bn: usize, cols: usize, i: usize) -> f64 {
    let (r, c) = (i / cols, i % cols);
    bf16_nearest((e4m3(m.w[i]) as f32 * m.scales[(r / bm) * m.sc + c / bn]) as f64)
}

/// `(y, mag)` for an FP8 matrix, as [`gemv_reference`] is for bf16.
pub fn fp8_reference(m: &Fp8, rows: usize, cols: usize, x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let (bm, bn) = (rows / m.sr, cols / m.sc);
    let mut y = Vec::with_capacity(rows);
    let mut mag = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut acc = 0.0f64;
        let mut sum = 0.0f64;
        for (c, &xc) in x.iter().enumerate().take(cols) {
            let w = fp8_weight(m, bm, bn, cols, r * cols + c);
            let t = w * xc as f64;
            acc += t;
            sum += t.abs();
        }
        y.push(acc);
        mag.push(sum);
    }
    (y, mag)
}

// ---------------------------------------------------------------- delta reference

pub struct DeltaShape {
    pub hk: usize,
    pub hv: usize,
    pub dk: usize,
    pub dv: usize,
    pub k: usize,
}

impl DeltaShape {
    pub fn c(&self) -> usize {
        2 * self.hk * self.dk + self.hv * self.dv
    }
}

pub struct DeltaIn {
    pub mixed: Vec<f32>,
    pub conv_state: Vec<f32>,
    pub conv_w: Vec<f32>,
    pub conv_b: Option<Vec<f32>>,
    pub z: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub state: Vec<f32>,
    pub norm_w: Vec<f32>,
}

pub fn gen_delta(s: &DeltaShape, seed: u64, bias: bool) -> DeltaIn {
    let mut r = Rng::new(seed);
    let c = s.c();
    DeltaIn {
        mixed: (0..c).map(|_| r.uni(-1.5, 1.5)).collect(),
        conv_state: (0..c * s.k).map(|_| r.uni(-1.5, 1.5)).collect(),
        conv_w: (0..c * s.k).map(|_| r.uni(-0.6, 0.6)).collect(),
        conv_b: bias.then(|| (0..c).map(|_| r.uni(-0.3, 0.3)).collect()),
        z: (0..s.hv * s.dv).map(|_| r.uni(-2.0, 2.0)).collect(),
        a: (0..s.hv).map(|_| r.uni(-4.0, 4.0)).collect(),
        b: (0..s.hv).map(|_| r.uni(-3.0, 3.0)).collect(),
        a_log: (0..s.hv).map(|_| r.uni(-4.0, 0.0)).collect(),
        dt_bias: (0..s.hv).map(|_| r.uni(-1.0, 1.0)).collect(),
        state: (0..s.hv * s.dk * s.dv).map(|_| r.uni(-0.5, 0.5)).collect(),
        norm_w: (0..s.dv).map(|_| r.uni(0.5, 1.5)).collect(),
    }
}

pub struct DeltaExpect {
    pub y: Vec<f64>,
    pub conv_state: Vec<f64>,
    pub state: Vec<f64>,
    pub out: Vec<f64>,
}

fn silu64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn softplus64(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

pub fn delta_reference(s: &DeltaShape, n: &DeltaIn, eps: f32) -> DeltaExpect {
    let (hk, hv, dk, dv, k) = (s.hk, s.hv, s.dk, s.dv, s.k);
    let c = s.c();
    let mut cs: Vec<f64> = n.conv_state.iter().map(|&v| v as f64).collect();
    let mut y = vec![0.0f64; c];
    for ch in 0..c {
        for j in 0..k - 1 {
            cs[ch * k + j] = cs[ch * k + j + 1];
        }
        cs[ch * k + k - 1] = n.mixed[ch] as f64;
        let mut acc = match &n.conv_b {
            Some(bv) => bv[ch] as f64,
            None => 0.0,
        };
        for j in 0..k {
            acc += n.conv_w[ch * k + j] as f64 * cs[ch * k + j];
        }
        y[ch] = silu64(acc);
    }
    let key_dim = hk * dk;
    let rep = hv / hk;
    let scale = 1.0 / (dk as f64).sqrt();
    let mut st: Vec<f64> = n.state.iter().map(|&v| v as f64).collect();
    let mut out = vec![0.0f64; hv * dv];
    for h in 0..hv {
        let hkx = h / rep;
        let qr: Vec<f64> = (0..dk).map(|d| y[hkx * dk + d]).collect();
        let kr: Vec<f64> = (0..dk).map(|d| y[key_dim + hkx * dk + d]).collect();
        let vt: Vec<f64> = (0..dv).map(|v| y[2 * key_dim + h * dv + v]).collect();
        let qs = qr.iter().map(|v| v * v).sum::<f64>() + 1e-6;
        let ks = kr.iter().map(|v| v * v).sum::<f64>() + 1e-6;
        let q: Vec<f64> = qr.iter().map(|v| v / qs.sqrt() * scale).collect();
        let kk: Vec<f64> = kr.iter().map(|v| v / ks.sqrt()).collect();
        let beta = 1.0 / (1.0 + (-(n.b[h] as f64)).exp());
        let decay =
            (-(n.a_log[h] as f64).exp() * softplus64(n.a[h] as f64 + n.dt_bias[h] as f64)).exp();
        let base = h * dk * dv;
        for i in 0..dk * dv {
            st[base + i] *= decay;
        }
        let kv: Vec<f64> = (0..dv)
            .map(|v| (0..dk).map(|d| st[base + d * dv + v] * kk[d]).sum::<f64>())
            .collect();
        let delta: Vec<f64> = (0..dv).map(|v| (vt[v] - kv[v]) * beta).collect();
        for d in 0..dk {
            for v in 0..dv {
                st[base + d * dv + v] += kk[d] * delta[v];
            }
        }
        let core: Vec<f64> = (0..dv)
            .map(|v| (0..dk).map(|d| st[base + d * dv + v] * q[d]).sum::<f64>())
            .collect();
        let var = core.iter().map(|v| v * v).sum::<f64>() / dv as f64;
        let rr = 1.0 / (var + eps as f64).sqrt();
        for v in 0..dv {
            out[h * dv + v] =
                (n.norm_w[v] as f64 * (core[v] * rr)) * silu64(n.z[h * dv + v] as f64);
        }
    }
    DeltaExpect {
        y,
        conv_state: cs,
        state: st,
        out,
    }
}
