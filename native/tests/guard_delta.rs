// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! Page-fenced gated DeltaNet step: the in-place states against guard pages on both sides, the
//! parameters read-only, and the tree pass's multi-row state slab.
#![cfg(windows)]

mod common;

use common::deltak::*;
use common::DeltaShape;

/// One step at the full shape (hk 16, hv 48, dk 128, dv 128, K 4) against the f64 reference, with
/// every buffer fenced on both sides, at three thread counts and with and without the conv bias.
#[test]
fn real_shape_both_sides() {
    check("real", &REAL, &[1, 0, 16]);
}

/// The same at shapes whose head and channel counts do not divide the vector width, from a single
/// element up to the widest head dimension the kernel accepts.
#[test]
fn ragged_shapes_both_sides() {
    for s in [
        DeltaShape {
            hk: 1,
            hv: 1,
            dk: 1,
            dv: 1,
            k: 1,
        },
        DeltaShape {
            hk: 2,
            hv: 6,
            dk: 7,
            dv: 13,
            k: 5,
        },
        DeltaShape {
            hk: 3,
            hv: 9,
            dk: 16,
            dv: 8,
            k: 4,
        },
        DeltaShape {
            hk: 2,
            hv: 2,
            dk: 256,
            dv: 256,
            k: 4,
        },
    ] {
        check(
            &format!("ragged {}/{}/{}/{}/{}", s.hk, s.hv, s.dk, s.dv, s.k),
            &s,
            &[1, 0],
        );
    }
}

/// Stepping one row of a seventeen-row state slab in place leaves the other sixteen rows poisoned
/// and untouched, and the stepped row matches the reference.
#[test]
fn tree_slab_of_seventeen() {
    slab_check("real", &REAL, 17, 0);
    slab_check("real", &REAL, 17, 1);
}
