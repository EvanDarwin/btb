// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! C ABI. Every function returns 0 on success or a negative `codes` value; buffers are caller-owned and
//! written in place; a panic is caught at the boundary and returned as `ERR_PANIC`.
//!
//! Every entry point checks, once per call: null pointers (`ERR_NULL`), zero or overflowing dimensions and
//! byte sizes past a pointer offset's range, pointers naturally aligned for their element type (the scalar
//! and tail paths dereference elements directly; the SIMD loads never needed it), a `threads` count of at
//! most `gemv::MAX_THREADS` (0 is every core), and the finiteness of a kernel's scalar parameters
//! (`ERR_DOMAIN` for all of these). Buffers must be contiguous and must not overlap; that is the caller's.

pub mod attn;
pub mod delta;
pub mod direct;
pub mod gemv;
pub mod sample;

pub mod codes {
    pub const OK: i32 = 0;
    /// A required pointer was null with a nonzero length.
    pub const ERR_NULL: i32 = -1;
    pub const ERR_UTF8: i32 = -2;
    pub const ERR_PARSE: i32 = -3;
    pub const ERR_DIV_ZERO: i32 = -4;
    pub const ERR_OVERFLOW: i32 = -5;
    /// A zero or overflowing dimension, a shape that does not split into its heads, or a malformed escape list.
    pub const ERR_DOMAIN: i32 = -6;
    /// A panic caught at the FFI boundary.
    pub const ERR_PANIC: i32 = -7;
    /// The OS refused: open, seek, read or the aligned scratch allocation failed.
    pub const ERR_IO: i32 = -8;
    /// The byte range runs past the end of the file, or a read came up short.
    pub const ERR_EOF: i32 = -9;
}

use codes::*;

fn guard<F: FnOnce() -> i32>(f: F) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(ERR_PANIC)
}

/// `y[i][r] = sum_c bf16(w[r][c]) * x[i][c]` in f32. `w` is `[rows, cols]` row-major bf16 bit patterns,
/// `x` is `[b, cols]` f32, `y` is `[b, rows]` f32. `threads == 0` uses every core. The result is
/// bit-identical for every `threads` and every `b`.
///
/// # Safety
/// `w` readable for `rows * cols` u16, `x` for `b * cols` f32, `y` writable for `b * rows` f32; no overlap.
#[no_mangle]
pub unsafe extern "C" fn btb_gemv_bf16_rows(
    w: *const u16,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_core(w, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_bf16_rows`] for `n` independent tasks under one thread-pool dispatch: task `t` reads
/// `w[t]` (`[rows[t], cols[t]]` bf16 bit patterns) and `x[t]` (`[b[t], cols[t]]` f32) and writes `y[t]`
/// (`[b[t], rows[t]]` f32). The rows of every task are spread over the pool together, so ten small tasks
/// pay one dispatch and one barrier instead of ten. `threads == 0` uses every core. The output of task `t`
/// is bit-identical to its own [`btb_gemv_bf16_rows`] call, for every `threads` and every `n`.
///
/// # Safety
/// `w`, `rows`, `cols`, `x`, `b` and `y` readable for `n` elements, and each task's arrays valid for the
/// lengths [`btb_gemv_bf16_rows`] states; the `n` output buffers must not overlap each other or an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_bf16_group(
    n: usize,
    w: *const *const u16,
    rows: *const usize,
    cols: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_group_core(n, w, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_bf16_rows`] over the 12-bit packed matrix. For `n = rows * cols` weights: `lo` is `u8[n]`
/// (low bytes), `hi4` is `u8[(n + 1) / 2]` (a 4-bit code per weight, weight `2k` in the low nibble of byte
/// `k`), `table` is `u8[16]` (code to high byte), and `esc_idx` / `esc_val` (`i32[n_esc]` strictly
/// ascending, `u8[n_esc]`) list the weights whose high byte is not in the table. Weight `i` is
/// `(table[code_i] << 8) | lo[i]` unless escaped. Output bit-identical to the bf16 form.
///
/// # Safety
/// Every array readable for the length stated, `y` writable for `b * rows` f32; no overlap with `y`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_p12_rows(
    lo: *const u8,
    hi4: *const u8,
    table: *const u8,
    esc_idx: *const i32,
    esc_val: *const u8,
    n_esc: usize,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        gemv::gemv_p12_core(
            lo, hi4, table, esc_idx, esc_val, n_esc, rows, cols, x, b, y, threads,
        )
    })
}

/// [`btb_gemv_bf16_rows`] over an MXFP4 matrix, the form gpt-oss ships its experts in. `blocks` is
/// `u8[rows * cols / 32 * 16]` and `scales` is `u8[rows * cols / 32]`: 32 weights along `cols` share one
/// block of 16 bytes (two fp4 e2m1 codes per byte, the earlier weight in the LOW nibble) and one uint8
/// e8m0 exponent, and weight `i` is `fp4(code_i) * 2^(scales[i / 32] - 127)`. `cols` must be a multiple
/// of 32. Bit-identical for every `threads` and every `b`, and across the scalar, AVX2, AVX-512 and NEON
/// paths.
///
/// # Safety
/// `blocks` and `scales` readable for the lengths stated, `x` for `b * cols` f32, `y` writable for
/// `b * rows` f32; `y` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_mxfp4_rows(
    blocks: *const u8,
    scales: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_mxfp4_core(blocks, scales, rows, cols, x, b, y, threads, false) })
}

/// [`btb_gemv_mxfp4_rows`] over a matrix in ggml's MXFP4 layout (a GGUF's): `raw` its 17-byte blocks,
/// the e8m0 scale first, then 16 bytes whose low nibbles are weights 0..15 and high nibbles 16..31. The
/// same arithmetic in the same order, so the bits equal the checkpoint layout's for the same weights.
///
/// # Safety
/// `raw` readable for `rows * cols / 32 * 17` bytes, `x` for `b * cols` f32, `y` writable for `b * rows`
/// f32; `y` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_mxfp4_ggml_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        gemv::gemv_mxfp4_core(raw, std::ptr::null(), rows, cols, x, b, y, threads, true)
    })
}

/// [`btb_gemv_mxfp4_rows`] for `n` independent tasks under one thread-pool dispatch, as
/// [`btb_gemv_bf16_group`] is for the bf16 matvec: one layer's active experts are one call. The output
/// of task `t` is bit-identical to its own [`btb_gemv_mxfp4_rows`] call.
///
/// # Safety
/// `blocks`, `scales`, `rows`, `cols`, `x`, `b` and `y` readable for `n` elements, each task's arrays
/// valid for the lengths [`btb_gemv_mxfp4_rows`] states; the `n` output buffers must not overlap each
/// other or an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_mxfp4_group(
    n: usize,
    blocks: *const *const u8,
    scales: *const *const u8,
    rows: *const usize,
    cols: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        gemv::gemv_mxfp4_group_core(n, blocks, scales, rows, cols, x, b, y, threads, false)
    })
}

/// [`btb_gemv_mxfp4_group`] over matrices in ggml's layout ([`btb_gemv_mxfp4_ggml_rows`]): `raws` the
/// `n` block pointers.
///
/// # Safety
/// As [`btb_gemv_mxfp4_group`], each task's `raw` valid for the length [`btb_gemv_mxfp4_ggml_rows`]
/// states.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_mxfp4_ggml_group(
    n: usize,
    raws: *const *const u8,
    rows: *const usize,
    cols: *const usize,
    x: *const *const f32,
    b: *const usize,
    y: *mut *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        gemv::gemv_mxfp4_group_core(
            n,
            raws,
            std::ptr::null(),
            rows,
            cols,
            x,
            b,
            y,
            threads,
            true,
        )
    })
}

/// [`btb_gemv_bf16_rows`] over a GGUF Q4_K matrix, multiplied as stored (no bf16 copy). `raw` is the
/// file's bytes for `[rows, cols]`: `rows * cols / 256` superblocks of 144 bytes, row-major, each a delta
/// and min (f16), 12 packed 6-bit scale/min bytes, then 128 nibble bytes. `cols` must be a multiple of
/// 256. Bit-identical for every `threads` and every `b`, and row `r` is the same at any `b`.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 144` bytes, `x` for `b * cols` f32, `y` writable for
/// `b * rows` f32; `y` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q4k_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q4k_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q6_K matrix. `raw` is `rows * cols / 256` superblocks of 210 bytes,
/// row-major: 128 low-nibble bytes, 64 high-2-bit bytes, 16 int8 scales, then the f16 delta. `cols` must
/// be a multiple of 256.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 210` bytes, `x` for `b * cols` f32, `y` writable for
/// `b * rows` f32; `y` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q6k_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q6k_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q5_K matrix (176-byte superblocks: the delta, min, 12 scale/min bytes,
/// a 32-byte high-bit plane, then 128 nibble bytes). `cols` a multiple of 256.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 176` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q5k_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q5k_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q2_K matrix (84-byte superblocks: 16 scale/min bytes, 64 2-bit weight
/// bytes, then the delta and min). `cols` a multiple of 256.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 84` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q2k_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q2k_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q3_K matrix (110-byte superblocks: a 32-byte high-bit mask, 64 low-2-bit
/// weight bytes, 12 packed 6-bit scale bytes, then the delta). `cols` a multiple of 256.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 110` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q3k_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q3k_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF IQ4_NL matrix (18-byte blocks of 32: an f16 delta then 16 nibble bytes,
/// a weight `d * KV[code]` through the fixed non-linear codebook). `cols` a multiple of 32.
///
/// # Safety
/// `raw` readable for `rows * cols / 32 * 18` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_iq4nl_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_iq4nl_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF IQ4_XS matrix (136-byte superblocks: an f16 delta, a high-scale word,
/// four low-scale bytes, 128 nibble bytes; a weight `d*(scale-32)*KV[code]`). `cols` a multiple of 256.
///
/// # Safety
/// `raw` readable for `rows * cols / 256 * 136` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_iq4xs_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_iq4xs_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q4_0 matrix (18-byte blocks of 32: an f16 delta then 16 nibble bytes, a
/// weight `d*(code-8)`). `cols` a multiple of 32.
///
/// # Safety
/// `raw` readable for `rows * cols / 32 * 18` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q40_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q40_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q4_1 matrix (20-byte blocks of 32: a delta and min, then 16 nibble bytes,
/// a weight `d*code + m`). `cols` a multiple of 32.
///
/// # Safety
/// `raw` readable for `rows * cols / 32 * 20` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q41_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q41_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF Q8_0 matrix (34-byte blocks of 32: an f16 delta then 32 int8 weights, a
/// weight `d*code`). `cols` a multiple of 32.
///
/// # Safety
/// `raw` readable for `rows * cols / 32 * 34` bytes, `x` for `b * cols` f32, `y` writable for `b * rows` f32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_gemv_q80_rows(
    raw: *const u8,
    rows: usize,
    cols: usize,
    x: *const f32,
    b: usize,
    y: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe { gemv::gemv_q80_core(raw, rows, cols, x, b, y, threads) })
}

/// [`btb_gemv_q4k_rows`] over a GGUF IQ lattice matrix (256-weight superblocks): `raw` the file's bytes,
/// `grid` the type's int8 codebook flattened (entries * values), `ksigns` the shared 128-entry sign table
/// (null for the types that carry explicit signs or none). A weight is a signed, scaled grid entry. `cols`
/// a multiple of 256.
macro_rules! latt_rows {
    ($name:ident, $core:ident) => {
        /// A GGUF IQ lattice matvec: see [`latt_rows`]'s documentation for the layout.
        ///
        /// # Safety
        /// `raw`, `grid` and (where read) `ksigns` valid for the type; `x` readable for `b * cols` f32, `y`
        /// writable for `b * rows` f32; `y` must not overlap an input.
        #[no_mangle]
        #[allow(clippy::too_many_arguments)]
        pub unsafe extern "C" fn $name(
            raw: *const u8,
            grid: *const i8,
            ksigns: *const u8,
            rows: usize,
            cols: usize,
            x: *const f32,
            b: usize,
            y: *mut f32,
            threads: usize,
        ) -> i32 {
            guard(|| unsafe { gemv::$core(raw, grid, ksigns, rows, cols, x, b, y, threads) })
        }
    };
}
latt_rows!(btb_gemv_iq3xxs_rows, gemv_iq3xxs_core);
latt_rows!(btb_gemv_iq2xxs_rows, gemv_iq2xxs_core);
latt_rows!(btb_gemv_iq2xs_rows, gemv_iq2xs_core);
latt_rows!(btb_gemv_iq2s_rows, gemv_iq2s_core);
latt_rows!(btb_gemv_iq1s_rows, gemv_iq1s_core);
latt_rows!(btb_gemv_iq3s_rows, gemv_iq3s_core);
latt_rows!(btb_gemv_iq1m_rows, gemv_iq1m_core);

/// One decode step of grouped-query attention over a bf16 key/value cache, f32 arithmetic throughout.
/// Unlike every other kernel here, the result is not bit-identical across `threads`: the rows are split by
/// the thread count and merged, so the last bits vary with `threads`, not the value. `q` and `out` are
/// `[hq, d]` f32 row-major; `k` and `v` are `[hk, n, d]` bf16 bit patterns whose row `r` of kv head `h`
/// starts at `h * k_head_stride + r * d` (`k_head_stride >= n * d`, so the caller may keep capacity past
/// `n`). `hq` must be a multiple of `hk`: query head `j` reads kv head `j / (hq / hk)`. For each `j`,
/// `s[r] = scale * dot(q[j], K[h][r])`, `p = softmax(s)`, `out[j] = sum_r p[r] * V[h][r]`, taken over every
/// row (there is no mask) with a streaming softmax. `threads == 0` uses every core.
///
/// # Safety
/// `q` readable for `hq * d` f32, `k` for `hk * k_head_stride` and `v` for `hk * v_head_stride` u16,
/// `out` writable for `hq * d` f32; `out` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_attn_decode_bf16(
    q: *const f32,
    k: *const u16,
    v: *const u16,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        attn::decode_core_bf16(
            q,
            k,
            v,
            n,
            hq,
            hk,
            d,
            k_head_stride,
            v_head_stride,
            scale,
            out,
            threads,
        )
    })
}

/// [`btb_attn_decode_bf16`] over an f32 cache: same math, same shapes and strides, `k` and `v` readable
/// for `hk * k_head_stride` and `hk * v_head_stride` f32 instead of u16.
///
/// # Safety
/// `q` readable for `hq * d` f32, `k` for `hk * k_head_stride` and `v` for `hk * v_head_stride` f32,
/// `out` writable for `hq * d` f32; `out` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_attn_decode_f32(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    n: usize,
    hq: usize,
    hk: usize,
    d: usize,
    k_head_stride: usize,
    v_head_stride: usize,
    scale: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        attn::decode_core_f32(
            q,
            k,
            v,
            n,
            hq,
            hk,
            d,
            k_head_stride,
            v_head_stride,
            scale,
            out,
            threads,
        )
    })
}

/// One gated DeltaNet position for every head, f32 throughout: the causal conv update (`conv_state`
/// `[C, K]`, `conv_w` `[C, K]`, `conv_b` `[C]` or null) with silu, the `q | k | v` split of `mixed_qkv`
/// `[C]` (`C = 2 * hk * dk + hv * dv`), q/k l2-normalised, the gated delta rule on `state` `[hv, dk, dv]`
/// (`a`, `b` `[hv]`; `a_log`, `dt_bias` `[hv]`), and the gated RMSNorm (`z` `[hv * dv]`, `norm_w` `[dv]`,
/// `eps`) into `out` `[hv * dv]`. `mixed_qkv`, `conv_state` and `state` are updated in place. `hv` must be
/// a multiple of `hk`. Bit-identical for every `threads`.
///
/// # Safety
/// Every array valid for the length above; `out` must not overlap an input.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_delta_step(
    mixed_qkv: *mut f32,
    conv_state: *mut f32,
    conv_w: *const f32,
    conv_b: *const f32,
    c_dim: usize,
    k_size: usize,
    z: *const f32,
    a: *const f32,
    b: *const f32,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    norm_w: *const f32,
    eps: f32,
    out: *mut f32,
    threads: usize,
) -> i32 {
    guard(|| unsafe {
        delta::delta_step(
            mixed_qkv, conv_state, conv_w, conv_b, c_dim, k_size, z, a, b, a_log, dt_bias, state,
            hk, hv, dk, dv, norm_w, eps, out, 1, threads,
        )
    })
}

/// The pick of one token per row of `x` (`[rows, v]` f32 logits): the argmax at `temperature <= 0`, else a
/// draw under the temperature, the top-k (0: all) and top-p (1: all), with the row's noise from `keys[r]`
/// (u64). `out` `[rows]` u32. Deterministic for a (key, row) at every `threads`.
///
/// # Safety
/// `x` readable for `rows * v` f32, `keys` for `rows` u64, `out` writable for `rows` u32; no overlap.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn btb_sample_pick(
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
    guard(|| unsafe {
        sample::sample_pick(x, rows, v, keys, temperature, top_k, top_p, out, threads)
    })
}

/// Copy bytes `[off, off + len)` of the file at `path` (NUL-terminated UTF-16) into `dst` with
/// unbuffered sequential reads that bypass the page cache. `dst` may have any alignment.
/// `chunk_bytes == 0` selects the default request size. `len == 0` returns `OK` without opening the file.
///
/// # Safety
/// `path` NUL-terminated UTF-16; `dst` writable for `len` bytes and not written concurrently.
#[no_mangle]
pub unsafe extern "C" fn btb_read_direct(
    path: *const u16,
    off: u64,
    len: u64,
    dst: *mut u8,
    chunk_bytes: u64,
) -> i32 {
    guard(|| unsafe {
        direct::read_direct(path, off, len, dst, chunk_bytes, direct::DEFAULT_DEPTH)
    })
}

/// Open the file at `path` (NUL-terminated UTF-16) for unbuffered reads and return its handle (>= 0), or a
/// negative `codes` value. The handle keeps the file open and its size cached for [`btb_read_at`], and denies
/// every other process write and delete access to it until [`btb_close`]; handles are process-wide and
/// nothing else closes them.
///
/// # Safety
/// `path` NUL-terminated UTF-16.
#[no_mangle]
pub unsafe extern "C" fn btb_open(path: *const u16) -> i64 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        direct::open(path)
    }))
    .unwrap_or(ERR_PANIC as i64)
}

/// Release a handle from [`btb_open`]; `ERR_DOMAIN` if it is not open (a second close). Reads still running
/// on it finish, and the file closes when the last of them returns.
#[no_mangle]
pub extern "C" fn btb_close(handle: i64) -> i32 {
    guard(|| direct::close(handle))
}

/// [`btb_read_direct`]'s contract on an open handle: bytes `[off, off + len)` into `dst` at any alignment of
/// `off`, `len` and `dst`, `len == 0` an `OK` that does not touch the handle, `ERR_EOF` past the end of the
/// file, `ERR_DOMAIN` if `handle` is not open. `chunk_bytes == 0` selects the default request size and
/// `depth == 0` the default worker count; the span is cut into chunks and read by `depth` threads. Any number
/// of threads may call this on one handle at once: every read carries its own offset. With `off`, `len` and
/// `dst` all sector-aligned (4096) the drive writes `dst` itself; otherwise the chunk lands in a bounce buffer
/// that is kept per thread and reused by that thread's next call.
///
/// # Safety
/// `dst` writable for `len` bytes and not written concurrently.
#[no_mangle]
pub unsafe extern "C" fn btb_read_at(
    handle: i64,
    off: u64,
    len: u64,
    dst: *mut u8,
    chunk_bytes: u64,
    depth: usize,
) -> i32 {
    guard(|| unsafe { direct::read_at(handle, off, len, dst, chunk_bytes, depth) })
}
