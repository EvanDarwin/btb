# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4 (`qwen4_exp`): the family (`family`), its sparse attention's indexer (`qsa`), its decode and speculative
verify through the gated DeltaNet and the per-layer n-gram embedding (`verify`), its host layers' router and sparse
attention row for row as the one-token step (`router`, `attend`), its card program - the resident layers' step and
verify pass through btb's row-invariant card kernels (`card`) - and its MTP drafter (`drafter`)."""
