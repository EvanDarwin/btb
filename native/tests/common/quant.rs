// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The GGUF quant matvecs' test scaffolding: each format's block layout and C-ABI entry, random blocks
//! whose f16 factors are finite, a synthetic lattice grid and sign table, and an f64 reference that decodes
//! a weight the way llama.cpp's `dequantize_row_*` does, a group at a time rather than a kernel lane at a
//! time. Real lattice grids come from the gguf package; the Python cert holds the kernels to those.

use super::*;
use btb_native::codes::OK;
use btb_native::gemv::isa;
use btb_native::*;

pub type PlainFn =
    unsafe extern "C" fn(*const u8, usize, usize, *const f32, usize, *mut f32, usize) -> i32;
pub type LattFn = unsafe extern "C" fn(
    *const u8,
    *const i8,
    *const u8,
    usize,
    usize,
    *const f32,
    usize,
    *mut f32,
    usize,
) -> i32;

/// A format's C-ABI entry: plain, or a lattice type taking its grid and sign table.
#[derive(Clone, Copy)]
pub enum Call {
    Plain(PlainFn),
    Latt(LattFn),
}

/// Where a block's f16 factors sit, so a random block can hold them finite and modest.
#[derive(Clone, Copy)]
pub enum Delta {
    /// whole halves at these byte offsets
    At(&'static [usize]),
    /// one half spread over the top nibbles of four u16 scales from this offset (IQ1_M)
    Nibbles(usize),
}

pub struct Format {
    pub name: &'static str,
    /// weights a block
    pub block: usize,
    /// bytes a block
    pub bytes: usize,
    pub delta: Delta,
    /// (grid entries, values an entry) for a lattice type, else (0, 0)
    pub grid: (usize, usize),
    /// the type reads the shared 128-entry sign table
    pub ksigns: bool,
    pub call: Call,
    /// weight `i` of a block
    pub weight: fn(&[u8], &Tables, usize) -> f64,
}

/// A lattice type's grid and sign table (empty for the other formats).
pub struct Tables {
    pub grid: Vec<i8>,
    pub ksigns: Vec<u8>,
}

pub const FORMATS: &[Format] = &[
    plain("Q2_K", 256, 84, &[80, 82], btb_gemv_q2k_rows, q2k),
    plain("Q3_K", 256, 110, &[108], btb_gemv_q3k_rows, q3k),
    plain("Q4_K", 256, 144, &[0, 2], btb_gemv_q4k_rows, q4k),
    plain("Q5_K", 256, 176, &[0, 2], btb_gemv_q5k_rows, q5k),
    plain("Q6_K", 256, 210, &[208], btb_gemv_q6k_rows, q6k),
    plain("Q4_0", 32, 18, &[0], btb_gemv_q40_rows, q40),
    plain("Q4_1", 32, 20, &[0, 2], btb_gemv_q41_rows, q41),
    plain("Q8_0", 32, 34, &[0], btb_gemv_q80_rows, q80),
    plain("IQ4_NL", 32, 18, &[0], btb_gemv_iq4nl_rows, iq4nl),
    plain("IQ4_XS", 256, 136, &[0], btb_gemv_iq4xs_rows, iq4xs),
    latt(
        "IQ3_XXS",
        98,
        Delta::At(&[0]),
        (256, 4),
        true,
        btb_gemv_iq3xxs_rows,
        iq3xxs,
    ),
    latt(
        "IQ2_XXS",
        66,
        Delta::At(&[0]),
        (256, 8),
        true,
        btb_gemv_iq2xxs_rows,
        iq2xxs,
    ),
    latt(
        "IQ2_XS",
        74,
        Delta::At(&[0]),
        (512, 8),
        true,
        btb_gemv_iq2xs_rows,
        iq2xs,
    ),
    latt(
        "IQ2_S",
        82,
        Delta::At(&[0]),
        (1024, 8),
        false,
        btb_gemv_iq2s_rows,
        iq2s,
    ),
    latt(
        "IQ1_S",
        50,
        Delta::At(&[0]),
        (2048, 8),
        false,
        btb_gemv_iq1s_rows,
        iq1s,
    ),
    latt(
        "IQ3_S",
        110,
        Delta::At(&[0]),
        (512, 4),
        false,
        btb_gemv_iq3s_rows,
        iq3s,
    ),
    latt(
        "IQ1_M",
        56,
        Delta::Nibbles(48),
        (2048, 8),
        false,
        btb_gemv_iq1m_rows,
        iq1m,
    ),
];

const fn plain(
    name: &'static str,
    block: usize,
    bytes: usize,
    at: &'static [usize],
    f: PlainFn,
    weight: fn(&[u8], &Tables, usize) -> f64,
) -> Format {
    Format {
        name,
        block,
        bytes,
        delta: Delta::At(at),
        grid: (0, 0),
        ksigns: false,
        call: Call::Plain(f),
        weight,
    }
}

const fn latt(
    name: &'static str,
    bytes: usize,
    delta: Delta,
    grid: (usize, usize),
    ksigns: bool,
    f: LattFn,
    weight: fn(&[u8], &Tables, usize) -> f64,
) -> Format {
    Format {
        name,
        block: 256,
        bytes,
        delta,
        grid,
        ksigns,
        call: Call::Latt(f),
        weight,
    }
}

/// a half in [2^-4, 2^-3): exponent field 11, a random mantissa - a real quantizer's delta, never inf/nan
fn modest_half(r: &mut Rng) -> u16 {
    0x2C00 | ((r.next_u64() >> 40) as u16 & 0x03FF)
}

/// `rows * cols / block` blocks of random bytes with finite f16 factors.
pub fn blocks(f: &Format, rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng::new(seed);
    let n = rows * cols / f.block;
    let mut raw: Vec<u8> = (0..n * f.bytes).map(|_| r.next_u64() as u8).collect();
    for blk in raw.chunks_exact_mut(f.bytes) {
        match f.delta {
            Delta::At(offs) => {
                for &o in offs {
                    blk[o..o + 2].copy_from_slice(&modest_half(&mut r).to_le_bytes());
                }
            }
            Delta::Nibbles(o) => {
                let h = modest_half(&mut r);
                for k in 0..4 {
                    let hi = o + 2 * k + 1;
                    blk[hi] = (blk[hi] & 0x0F) | ((((h >> (4 * k)) & 0x0F) as u8) << 4);
                }
            }
        }
    }
    raw
}

pub fn tables(f: &Format, seed: u64) -> Tables {
    let mut r = Rng::new(seed ^ 0x6A1D);
    let (entries, vals) = f.grid;
    Tables {
        grid: (0..entries * vals).map(|_| r.next_u64() as i8).collect(),
        ksigns: if f.ksigns {
            (0..128).map(|_| r.next_u64() as u8).collect()
        } else {
            Vec::new()
        },
    }
}

/// the sign table's pointer as the kernel takes it: null for a type that does not read one
pub fn ksigns_ptr(t: &Tables) -> *const u8 {
    if t.ksigns.is_empty() {
        std::ptr::null()
    } else {
        t.ksigns.as_ptr()
    }
}

/// # Safety
/// The pointers valid for the shape, as the C ABI states.
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_raw(
    f: &Format,
    raw: *const u8,
    grid: *const i8,
    ks: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    match f.call {
        Call::Plain(g) => g(raw, rows, cols, x, b, y, threads),
        Call::Latt(g) => g(raw, grid, ks, rows, cols, x, b, y, threads),
    }
}

/// The matvec on plain buffers, asserting OK.
#[allow(clippy::too_many_arguments)]
pub fn run(
    f: &Format,
    raw: &[u8],
    t: &Tables,
    rows: usize,
    cols: usize,
    x: &[f32],
    b: usize,
    threads: usize,
) -> Vec<f32> {
    let mut y = vec![poison_f32(); b * rows];
    let code = unsafe {
        call_raw(
            f,
            raw.as_ptr(),
            t.grid.as_ptr(),
            ksigns_ptr(t),
            rows,
            cols,
            x.as_ptr(),
            b,
            y.as_mut_ptr(),
            threads,
        )
    };
    assert_eq!(code, OK, "{} {rows}x{cols} b={b}", f.name);
    y
}

/// y = W x for one x row in f64, and each row's sum of |w * x| (the scale its error is judged against).
pub fn reference(
    f: &Format,
    raw: &[u8],
    t: &Tables,
    rows: usize,
    cols: usize,
    x: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let per_row = cols / f.block;
    let mut want = vec![0.0f64; rows];
    let mut scale = vec![1e-30f64; rows];
    for r in 0..rows {
        for (c, &xc) in x[..cols].iter().enumerate() {
            let at = (r * per_row + c / f.block) * f.bytes;
            let term = (f.weight)(&raw[at..at + f.bytes], t, c % f.block) * xc as f64;
            want[r] += term;
            scale[r] += term.abs();
        }
    }
    (want, scale)
}

// ---------------------------------------------------------------- f64 references

fn half(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let mant = (h & 0x03FF) as f64;
    sign * match exp {
        0 => mant * 2f64.powi(-24),
        0x1F => {
            if mant == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        e => (1.0 + mant / 1024.0) * 2f64.powi(e - 15),
    }
}

fn half_at(b: &[u8], o: usize) -> f64 {
    half(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn u16_at(b: &[u8], o: usize) -> u32 {
    u16::from_le_bytes([b[o], b[o + 1]]) as u32
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn sign(byte: u8, bit: usize) -> f64 {
    if byte & (1 << bit) != 0 {
        -1.0
    } else {
        1.0
    }
}

/// llama.cpp's `get_scale_min_k4`: sub-block `j`'s 6-bit scale and min from the 12 packed bytes.
fn scale_min_k4(s: &[u8], j: usize) -> (f64, f64) {
    if j < 4 {
        ((s[j] & 63) as f64, (s[j + 4] & 63) as f64)
    } else {
        let sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        let mn = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (sc as f64, mn as f64)
    }
}

const KVALUES_IQ4NL: [f64; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

fn q2k(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let (d, dmin) = (half_at(b, 80), half_at(b, 82));
    let (n, is, shift, l) = (i / 128, (i % 128) / 16, 2 * ((i % 128) / 32), i % 32);
    let sc = b[n * 8 + is];
    let q = (b[16 + 32 * n + l] >> shift) & 3;
    d * (sc & 0x0F) as f64 * q as f64 - dmin * (sc >> 4) as f64
}

fn q3k(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let d = half_at(b, 108);
    let (a0, a1, a2) = (u32_at(b, 96), u32_at(b, 100), u32_at(b, 104));
    let (k1, k2) = (0x0303_0303u32, 0x0F0F_0F0Fu32);
    let aux = [
        (a0 & k2) | ((a2 & k1) << 4),
        (a1 & k2) | (((a2 >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((a2 >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((a2 >> 6) & k1) << 4),
    ];
    let scale = |k: usize| ((aux[k / 4] >> (8 * (k % 4))) & 0xFF) as i32 - 32;
    let (n, j, l) = (i / 128, (i % 128) / 32, i % 32);
    let q = ((b[32 + 32 * n + l] >> (2 * j)) & 3) as i32;
    let high = b[l] & (1 << (4 * n + j)) != 0;
    d * scale(8 * n + 2 * j + l / 16) as f64 * (q - if high { 0 } else { 4 }) as f64
}

fn q4k(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let (d, dmin) = (half_at(b, 0), half_at(b, 2));
    let (j, l) = (i / 32, i % 32);
    let byte = b[16 + 32 * (j / 2) + l];
    let q = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    let (sc, mn) = scale_min_k4(&b[4..16], j);
    d * sc * q as f64 - dmin * mn
}

fn q5k(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let (d, dmin) = (half_at(b, 0), half_at(b, 2));
    let (j, l) = (i / 32, i % 32);
    let byte = b[48 + 32 * (j / 2) + l];
    let low = if j % 2 == 0 { byte & 0x0F } else { byte >> 4 };
    let q = low | (((b[16 + l] >> j) & 1) << 4);
    let (sc, mn) = scale_min_k4(&b[4..16], j);
    d * sc * q as f64 - dmin * mn
}

fn q6k(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let d = half_at(b, 208);
    let (n, rest) = (i / 128, i % 128);
    let (quad, l) = (rest / 32, rest % 32);
    let ql = &b[64 * n..];
    let qh = b[128 + 32 * n + l];
    let low = match quad {
        0 => ql[l] & 0x0F,
        1 => ql[l + 32] & 0x0F,
        2 => ql[l] >> 4,
        _ => ql[l + 32] >> 4,
    };
    let q = (low | (((qh >> (2 * quad)) & 3) << 4)) as i32 - 32;
    let sc = b[192 + 8 * n + l / 16 + 2 * quad] as i8;
    d * sc as f64 * q as f64
}

fn nibble(b: &[u8], o: usize, i: usize) -> u8 {
    let byte = b[o + i % 16];
    if i < 16 {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

fn q40(b: &[u8], _t: &Tables, i: usize) -> f64 {
    half_at(b, 0) * (nibble(b, 2, i) as f64 - 8.0)
}

fn q41(b: &[u8], _t: &Tables, i: usize) -> f64 {
    half_at(b, 0) * nibble(b, 4, i) as f64 + half_at(b, 2)
}

fn q80(b: &[u8], _t: &Tables, i: usize) -> f64 {
    half_at(b, 0) * b[2 + i] as i8 as f64
}

fn iq4nl(b: &[u8], _t: &Tables, i: usize) -> f64 {
    half_at(b, 0) * KVALUES_IQ4NL[nibble(b, 2, i) as usize]
}

fn iq4xs(b: &[u8], _t: &Tables, i: usize) -> f64 {
    let (d, sh) = (half_at(b, 0), u16_at(b, 2));
    let (ib, j) = (i / 32, i % 32);
    let ls =
        ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32 | ((((sh >> (2 * ib)) & 3) as i32) << 4);
    d * (ls - 32) as f64 * KVALUES_IQ4NL[nibble(b, 8 + 16 * ib, j) as usize]
}

fn iq3xxs(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let aux = u32_at(b, 66 + 4 * ib);
    let db = half_at(b, 0) * (0.5 + (aux >> 28) as f64) * 0.5;
    let signs = t.ksigns[((aux >> (7 * l)) & 127) as usize];
    let g = b[2 + 8 * ib + 2 * l + j / 4] as usize;
    db * t.grid[g * 4 + j % 4] as f64 * sign(signs, j)
}

fn iq2xxs(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let grp = 2 + 8 * ib;
    let w1 = u32_at(b, grp + 4);
    let db = half_at(b, 0) * (0.5 + (w1 >> 28) as f64) * 0.25;
    let signs = t.ksigns[((w1 >> (7 * l)) & 127) as usize];
    db * t.grid[b[grp + l] as usize * 8 + j] as f64 * sign(signs, j)
}

fn iq2xs(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let q = u16_at(b, 2 + 2 * (4 * ib + l));
    let db = half_at(b, 0) * (0.5 + ((b[66 + ib] >> (4 * (l / 2))) & 0x0F) as f64) * 0.25;
    db * t.grid[(q & 511) as usize * 8 + j] as f64 * sign(t.ksigns[(q >> 9) as usize], j)
}

fn iq2s(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let g = b[2 + 4 * ib + l] as usize | ((((b[66 + ib] >> (2 * l)) & 3) as usize) << 8);
    let db = half_at(b, 0) * (0.5 + ((b[74 + ib] >> (4 * (l / 2))) & 0x0F) as f64) * 0.25;
    db * t.grid[g * 8 + j] as f64 * sign(b[34 + 4 * ib + l], j)
}

fn iq1s(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let qh = u16_at(b, 34 + 2 * ib);
    let dl = half_at(b, 0) * (2 * ((qh >> 12) & 7) + 1) as f64;
    let delta = if qh & 0x8000 != 0 { -0.125 } else { 0.125 };
    let g = b[2 + 4 * ib + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
    dl * (t.grid[g * 8 + j] as f64 + delta)
}

fn iq3s(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let q = 2 * l + j / 4;
    let g = b[2 + 8 * ib + q] as usize | ((((b[66 + ib] >> q) & 1) as usize) << 8);
    let db = half_at(b, 0) * (1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 0x0F) as u32) as f64;
    db * t.grid[g * 4 + j % 4] as f64 * sign(b[74 + 4 * ib + l], j)
}

fn iq1m(b: &[u8], t: &Tables, i: usize) -> f64 {
    let (ib, l, j) = (i / 32, (i % 32) / 8, i % 8);
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let d = half(
        ((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)) as u16,
    );
    let qh = (b[32 + 2 * ib + l / 2] >> (4 * (l % 2))) & 0x0F;
    let g = b[4 * ib + l] as usize | (((qh & 7) as usize) << 8);
    let delta = if qh & 8 != 0 { -0.125 } else { 0.125 };
    let dl = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1) as f64;
    dl * (t.grid[g * 8 + j] as f64 + delta)
}

// ---------------------------------------------------------------- the fenced sweep

/// Every format at `shapes` and the batch widths `ts`, each input read-only against a guard page and y
/// against one at either end: the two alignments agree bit for bit with the plain-buffer call, and row 0
/// holds to the f64 reference.
pub fn fenced_sweep(label: &str, shapes: &[(usize, usize)], ts: &[usize], threads_set: &[usize]) {
    eprintln!("[{label}] isa {:?}", isa());
    for f in FORMATS {
        let t = tables(f, 0x7AB1E);
        let mut grid = Fence::<i8>::new("grid", t.grid.len().max(1), Align::End);
        if !t.grid.is_empty() {
            grid.copy_from(&t.grid);
        }
        grid.protect_readonly();
        let mut ks = Fence::<u8>::new("ksigns", t.ksigns.len().max(1), Align::End);
        if !t.ksigns.is_empty() {
            ks.copy_from(&t.ksigns);
        }
        ks.protect_readonly();
        let ks_ptr = if t.ksigns.is_empty() {
            std::ptr::null()
        } else {
            ks.ptr()
        };
        for (idx, &(rows, cols)) in shapes.iter().enumerate() {
            let cols = cols.div_ceil(f.block) * f.block;
            let host = blocks(f, rows, cols, 0xB10C + idx as u64);
            let mut raw = Fence::<u8>::new("raw", host.len(), Align::End);
            raw.copy_from(&host);
            raw.protect_readonly();
            let t_max = ts.iter().copied().max().unwrap_or(1);
            let x_all = gen_f32(t_max * cols, 0x5EED + idx as u64);
            let (want, scale) = reference(f, &host, &t, rows, cols, &x_all[..cols]);
            for &b in ts {
                let mut x = Fence::<f32>::new("x", b * cols, Align::End);
                x.copy_from(&x_all[..b * cols]);
                x.protect_readonly();
                let mut y_end = Fence::<f32>::new("y", b * rows, Align::End);
                let mut y_start = Fence::<f32>::new("y", b * rows, Align::Start);
                for &threads in threads_set {
                    let ctx = format!("{label} {} {rows}x{cols} b={b} threads={threads}", f.name);
                    let mut outs = Vec::new();
                    for y in [&mut y_end, &mut y_start] {
                        y.fill(poison_f32());
                        let code = unsafe {
                            call_raw(
                                f,
                                raw.ptr(),
                                grid.ptr(),
                                ks_ptr,
                                rows,
                                cols,
                                x.ptr(),
                                b,
                                y.mut_ptr(),
                                threads,
                            )
                        };
                        assert_eq!(code, OK, "{ctx}");
                        y.check_borders(&ctx);
                        assert!(
                            !y.as_slice().iter().any(|&v| is_poison_f32(v)),
                            "{ctx}: an output left unwritten"
                        );
                        outs.push(bits(y.as_slice()));
                    }
                    assert_eq!(outs[0], outs[1], "{ctx}: the y alignment changed the bits");
                    let plain = run(f, &host, &t, rows, cols, x.as_slice(), b, threads);
                    assert_eq!(
                        outs[0],
                        bits(&plain),
                        "{ctx}: fenced differs from the plain call"
                    );
                    for r in 0..rows {
                        let err = (plain[r] as f64 - want[r]).abs() / scale[r];
                        assert!(err < 1e-5, "{ctx} row {r}: {err:.3e} off the f64 reference");
                    }
                }
            }
        }
        eprintln!("[{label}] {} clean", f.name);
    }
}
