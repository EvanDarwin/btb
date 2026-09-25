// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced GGUF quant matvecs, every format: the raw blocks, the lattice grid and sign table and x
//! each end exactly at a guard page read-only, y against one at either end, over the tree pass's batch
//! widths and threads 1 / 0 / 16.

mod common;

use common::quant::fenced_sweep;

/// Host-linear-like shapes, one wide enough to split across threads.
#[test]
fn real_shapes_every_tree_width() {
    fenced_sweep(
        "quant",
        &[(48, 1024), (256, 2048), (1024, 512)],
        &[1, 2, 7, 8, 9, 16, 17],
        &[1, 0, 16],
    );
}

/// A single row, a single superblock, and row counts off the unroll: the tails of the row walk.
#[test]
fn ragged_shapes() {
    fenced_sweep(
        "quant ragged",
        &[(1, 256), (3, 256), (5, 768), (17, 512)],
        &[1, 3, 17],
        &[1, 0],
    );
}
