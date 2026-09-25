// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The same fenced sweeps on the scalar path, pinned with `BTB_NATIVE_ISA=scalar` before the
//! crate's first kernel call in this process (every test sets it; the first one wins).

mod common;

use btb_native::gemv::{isa, Isa};
use common::*;

fn pin() {
    force_isa("scalar");
    assert_eq!(isa(), Isa::Scalar, "the scalar path was not selected");
}

/// The attention sweeps and the input-digest check with no vector path selected.
#[test]
fn attention_every_length() {
    pin();
    attn::sweep("scalar real-stride", |_| attn::CAP * attn::D, false, false);
    attn::sweep("scalar tight-stride", |n| n * attn::D, true, false);
    attn::digest_check(&[1, 0]);
}

/// The bf16, packed and grouped mat-vec sweeps with no vector path selected, over three host
/// linears and the ragged shapes.
#[test]
fn gemv_rows_and_packed() {
    pin();
    gemvk::rows_sweep(
        "scalar",
        &[(48, 5120, false), (1024, 5120, false), (5120, 6144, true)],
        &[1, 0],
        1e-3,
    );
    gemvk::rows_sweep("scalar ragged", ODD, &[1, 0], 1e-2);
    gemvk::group_check("scalar group", &[1, 0]);
}

/// The FP8 rows and grouped sweeps with no vector path selected, held to the f64 reference the vector
/// path's `guard_fp8` is.
#[test]
fn fp8_rows_and_group() {
    pin();
    fp8k::rows_sweep("scalar", &[1, 0]);
    fp8k::group_check("scalar", &[1, 0]);
}

/// The DeltaNet step and the state slab with no vector path selected.
#[test]
fn delta_step() {
    pin();
    deltak::check("scalar", &deltak::REAL, &[1, 0]);
    deltak::slab_check("scalar", &deltak::REAL, 17, 0);
}
