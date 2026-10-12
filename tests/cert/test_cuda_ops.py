"""The CUDA card-kernel matrix as a suite gate. The card runs only where a GPU is present, so this is the
structural half: the kernels' bindings are what native/cuda's signatures write (each parameter a role, no two of a
kernel's alike), nothing in btb/ launches a kernel but through its typed launcher, and every launcher is reached from
btb/. A signature edited without the bindings, a raw launch, or a kernel nothing launches any more fails here; mypy
holds each launch to its kernel's parameters and their roles."""

from __future__ import annotations

import ast

import pytest

from . import cuda_ops


@pytest.mark.cert_gap
def test_the_bindings_are_current_and_every_kernel_is_launched() -> None:
    assert cuda_ops.gaps() == [], cuda_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a parse that silently found nothing would make the agreement test vacuous."""
    assert cuda_ops.defined_kernels(), "parsed no kernels from native/cuda"
    assert cuda_ops.reached_launchers(), "found no launcher reached from btb/"


def test_a_raw_launch_is_a_gap_and_a_launcher_reaches_its_kernels() -> None:
    """a `.launch(...)` outside the launchers is a gap whatever it names; a launcher named as an attribute - called,
    or picked and called later - reaches its kernels; a kernel named in a string (a presence check, a message) does
    not"""
    src = """
def run(k, gelu, M):
    k.launch("btb_silu_mul", (1, 1, 1), (256, 1, 1), [])
    glu = k.btb_gemv_mma_glu_gelu if gelu else k.btb_gemv_mma_glu_silu
    if "btb_publish" in k.fn:
        k.btb_gemv_bf16(M, (1, 1, 1), (128, 1, 1))
    btb_sample_out(k, (1, 1, 1), (256, 1, 1))
"""
    sources = [("mod.py", ast.parse(src))]
    assert cuda_ops.raw_launches(sources) == ["mod.py:3"]
    assert cuda_ops.reached_launchers(sources) == {
        "btb_gemv_mma_glu_gelu", "btb_gemv_mma_glu_silu", "btb_gemv_bf16", "btb_sample_out"
    }  # fmt: skip


def test_kernels_parameters_are_read_off_their_definitions() -> None:
    """each kernel's parameters and roles from its signature: an entry point spelled out (its name on the line after
    its bounds), a macro's kernels at each instantiation (the variant the launcher's first argument), one whose list is
    an object-like macro"""
    kernels = {k.method: k for k in cuda_ops.bindings().card_kernels()}
    gemm = kernels["btb_gemm_mma_bf16"]
    assert [p.role for p in gemm.params] == ["Weight", "MatIn", "MatOut", "OutDim", "InDim", "Rows"]
    assert [p.pointer for p in gemm.params] == [True, True, True, False, False, False]
    attn = kernels["btb_attn_flash"]
    assert attn.names() == ["btb_attn_flash_d64", "btb_attn_flash_d128", "btb_attn_flash_d256"]
    assert attn.variant == "D" and len(attn.params) == 18
    assert [p.role for p in attn.params[:6]] == ["Query", "KCache", "VCache", "AttnOut", "PrefixLen", "TreeParents"]
    prefill, kq = kernels["btb_attn_flash_prefill"], kernels["btb_attn_flash_prefill_kq"]
    assert prefill.params == kq.params and len(prefill.params) == 14  # ATTN_FLASH_PF_ARGS
    assert kernels["btb_gemv_bf16"].names()[0] == "btb_gemv_bf16_m1"


def test_a_signature_the_types_could_not_hold_is_refused() -> None:
    """the rules the launchers' types rest on: two parameters of a kernel sharing a role (a swap of them would type-
    check), a role a pointer's in one kernel and a scalar's in another, a bf16 passed by value, a role no kernel takes"""
    B = cuda_ops.bindings()
    roles = {"Query": B.Role("Query", "bf16", "", ""), "Rows": B.Role("Rows", "int", "", "")}

    def kernel(name: str, *params: tuple[str, str, bool]) -> object:
        return B.Kernel(name, name, "", (), tuple(B.Param(n, r, p) for n, r, p in params))

    B._hold([kernel("btb_x", ("q", "Query", True), ("T", "Rows", False))], roles)  # the rules kept
    with pytest.raises(B.BindingError, match="q and k share the role Query"):
        B._hold([kernel("btb_x", ("q", "Query", True), ("k", "Query", True), ("T", "Rows", False))], roles)
    with pytest.raises(B.BindingError, match="Rows is a pointer's in one kernel and a scalar's in another"):
        B._hold(
            [kernel("btb_x", ("q", "Query", True), ("T", "Rows", False)), kernel("btb_y", ("t", "Rows", True))], roles
        )
    with pytest.raises(B.BindingError, match="passes a bf16 by value"):
        B._hold([kernel("btb_x", ("q", "Query", False), ("T", "Rows", False))], roles)
    with pytest.raises(B.BindingError, match="roles no kernel takes: Rows"):
        B._hold([kernel("btb_x", ("q", "Query", True))], roles)
    with pytest.raises(B.BindingError, match="two launchers would be named btb_x"):
        B._hold([kernel("btb_x", ("q", "Query", True), ("T", "Rows", False)), kernel("btb_x")], roles)
