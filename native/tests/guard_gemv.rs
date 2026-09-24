// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced bf16 and 12-bit-packed rows mat-vec at every host-linear shape, over the batch
//! widths of the tree pass, threads 1 / 0 / 16, and the grouped dispatch.

mod common;

use common::gemvk::*;
use common::{ODD, REAL};

/// Every host-linear shape at every batch width: the two alignments agree bit for bit, the packed
/// matrix equals the bf16 one, and row 0 matches the f64 reference.
#[test]
fn real_shapes_every_tree_width() {
    rows_sweep("real", REAL, &[1, 0], 1e-3);
}

/// The same at odd, sub-tile and single-element shapes, with and without escapes in the packed
/// matrix: these are where the packed widen falls back to its scalar tail.
#[test]
fn ragged_shapes_every_tree_width() {
    rows_sweep("ragged", ODD, &[1, 0, 16], 1e-2);
    rows_sweep("ragged-no-escapes", ODD, &[1, 0], 0.0);
}

/// The three cheapest host linears at a fixed sixteen threads, the count that selects the
/// dedicated pool rather than the shared one.
#[test]
fn small_real_shapes_with_a_fixed_thread_count() {
    rows_sweep("real-16", &REAL[..3], &[16], 1e-3);
}

/// Every task of a grouped dispatch is bit-identical to its own single call, with each task's
/// output against a guard page.
#[test]
fn grouped_dispatch() {
    group_check("group", &[1, 0, 16]);
}

/// The batch widths above the tile set, which a prefill reaches when it hands the kernel a whole
/// prompt at once.
#[test]
fn prefill_widths() {
    rows_sweep_t("drafter", DRAFTER, &[1, 0, 16], 1e-3, Some(&T_DRAFTER));
}

/// A vocabulary-sized row count, which the rows kernel sees when the output projection runs on the
/// host: the row partition at that count, with a narrow (and an odd) column count to keep it cheap.
#[test]
fn vocab_row_count() {
    rows_sweep_t(
        "vocab",
        &[(248320, 64, true), (248320, 65, true)],
        &[1, 0, 16],
        1e-3,
        Some(&[1, 2, 17]),
    );
}
