// Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
//! The node-list bit-identity checks of `attn_nodes.rs` on the scalar path, pinned with
//! `BTB_NATIVE_ISA=scalar` before the crate's first kernel call in this process (every test sets it; the
//! first one wins): the list read goes through each tier's range reader, so each tier is held to its own
//! decode step.

use btb_native::gemv::{isa, Isa};

#[path = "common/refs.rs"]
mod refs;

#[path = "common/attn_nodes.rs"]
mod nodes;

fn pin() {
    std::env::set_var("BTB_NATIVE_ISA", "scalar");
    assert_eq!(isa(), Isa::Scalar, "the scalar path was not selected");
}

/// A list of the rows `0..n` is the scalar decode step over `n` rows, bit for bit.
#[test]
fn an_identity_list_is_the_decode_step() {
    pin();
    nodes::check_identity("scalar");
}

/// A batched call's rows are their own calls and their scalar decode steps over the gathered rows, bit
/// for bit. The f64 reference is the vector binary's; the bits here are held to the scalar decode step.
#[test]
fn each_row_of_a_batch_is_its_own_decode_step() {
    pin();
    nodes::check_lists("scalar", false);
}
