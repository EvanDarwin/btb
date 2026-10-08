"""The CUDA card-kernel matrix as a suite gate. The card runs only where a GPU is present, so this is the
structural half: the kernels the engine loads, the kernels native/cuda defines and the kernels btb launches must be
the same set. A new card kernel that the engine does not load, a loaded name with no definition, or a kernel nothing
launches any more (a dead path) fails here."""

from __future__ import annotations

import ast

import pytest

from . import cuda_ops


@pytest.mark.cert_gap
def test_loaded_and_defined_kernels_agree() -> None:
    assert cuda_ops.gaps() == [], cuda_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a parse that silently found nothing would make the agreement test vacuous."""
    assert cuda_ops.loaded_kernels(), "parsed no _Cuda.KERNELS from native.py"
    assert cuda_ops.defined_kernels(), "parsed no kernels from native/cuda"
    assert cuda_ops.launched_kernels(), "parsed no launch site from btb/"


KERNELS = [
    "btb_gemv_bf16_m16",
    "btb_gemv_silu_bf16_m16",
    "btb_gemv_sgate_bf16_m16",
    "btb_sigmoid_mul",
    "btb_silu_mul",
    "btb_norm_rope_kv_d64",
    "btb_norm_rope_kv_tbl_d128",
    "btb_norm_rope_kv_rows_d64",
    "btb_attn_flash_prefill_d64",
    "btb_attn_flash_prefill_kq_d64",
    "btb_gemm_f32_bf16",
]


def _launched(src: str) -> tuple[list[str], list[str]]:
    """the kernels of KERNELS the module `src`'s launches can launch, and its launches the parse cannot name"""
    sites = cuda_ops.launch_sites([("mod.py", ast.parse(src))])
    pats = cuda_ops.launch_patterns(sites)
    return [k for k in KERNELS if any(p.fullmatch(k) for p in pats)], cuda_ops.unresolved_launches(sites)


def test_only_what_flows_into_a_launch_is_launched() -> None:
    """a launch's kernel read through what names it - a constant, an f-string whose holes take the values the
    function gives them, a conditional, a variable of the function or the one around it, a method's return - and
    nothing else: a presence check, a requirement list, a message naming a kernel launches none of it"""
    src = """
class K:
    def flash_prefill_kernel(self, D, kq=None):
        return f"btb_attn_flash_prefill_{'kq_' if kq else ''}d{D}"

def body(k, paged, act_name, M, D):
    via = "_tbl" if paged else ""
    act = "gelu" if act_name == "gelu_pytorch_tanh" else "silu"
    nrk = f"btb_norm_rope_kv{via}_d{D}"
    gemv = f"btb_gemv_bf16_m{M}"
    if "btb_gemm_f32_bf16" in k.fn and ("btb_sigmoid_mul",):
        raise RuntimeError(f"no kernel btb_gemv_sgate_bf16_m{M}")

    def matvec(W):
        k.launch(gemv, (1, 1, 1), (1, 1, 1), [W])

    k.launch(nrk, (1, 1, 1), (32, 1, 1), [])
    k.launch(f"btb_{act}_mul", (1, 1, 1), (256, 1, 1), [])
    k.launch(k.flash_prefill_kernel(D), (1, 1, 1), (128, 1, 1), [])
"""
    launched, unresolved = _launched(src)
    assert launched == [
        "btb_gemv_bf16_m16",
        "btb_silu_mul",
        "btb_norm_rope_kv_d64",
        "btb_norm_rope_kv_tbl_d128",
        "btb_attn_flash_prefill_d64",
        "btb_attn_flash_prefill_kq_d64",
    ], launched
    assert unresolved == []


def test_a_launch_the_parse_cannot_name_is_a_gap() -> None:
    """a launch whose kernel comes from a parameter or a loop names no kernel the cert can read: a gap of its own,
    never a wildcard that would keep every kernel alive"""
    src = """
def run(k, name, names):
    k.launch(name, (1, 1, 1), (1, 1, 1), [])
    for n in names:
        k.launch(n, (1, 1, 1), (1, 1, 1), [])
"""
    launched, unresolved = _launched(src)
    assert launched == [] and unresolved == ["mod.py:3", "mod.py:5"], (launched, unresolved)


def test_every_launch_in_btb_is_read() -> None:
    """btb's own launches all name their kernels as the parse reads them, and launch only loaded kernels"""
    assert cuda_ops.unresolved_launches() == []
    assert cuda_ops.launched_kernels() <= cuda_ops.loaded_kernels()
