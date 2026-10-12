"""The CUDA card-kernel matrix as a suite gate. The card runs only where a GPU is present, so this is the
structural half: the kernels the engine loads, the kernels native/cuda defines and the kernels btb launches must be
the same set, and every launch must hand its kernel as many values as it declares parameters. A new card kernel that
the engine does not load, a loaded name with no definition, a kernel nothing launches any more (a dead path), or a
launch whose argument list is not its kernel's fails here."""

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


def test_a_launch_handed_another_count_than_its_kernel_declares_is_a_gap() -> None:
    """a launch's argument list counted as written - list literals, `*part`s of the lengths their assignments give,
    sums and conditionals of those, by position or by keyword - against every kernel its name can be: one choosing a
    row map's kernel and its extra value passes either way, a dropped value is caught - on one branch of a fixed
    kernel's list as well, its other branch's count right - and a list the parse cannot count is a gap of its own"""
    src = """
def run(k, paged, D, xs, cond):
    tbl = [P(t)] if paged else []
    name = f"btb_x{'_tbl' if paged else ''}_d{D}"
    k.launch(name, (1, 1, 1), (32, 1, 1), [a, b, c, *tbl])
    k.launch("btb_x_d64", (1, 1, 1), (32, 1, 1), [a, b])
    k.launch("btb_x_d64", (1, 1, 1), (32, 1, 1), args=[a] + [b, c])
    k.launch("btb_x_d64", (1, 1, 1), (32, 1, 1), list(xs))
    k.launch("btb_x_d64", (1, 1, 1), (32, 1, 1), [a, b, c] if cond else [a, b])
    k.launch(f"btb_x_d{D}", (1, 1, 1), (32, 1, 1), [a, b, *tbl])
"""
    sites = cuda_ops.launch_sites([("mod.py", ast.parse(src))])
    arity = {"btb_x_d64": 3, "btb_x_d128": 3, "btb_x_tbl_d64": 4}
    assert cuda_ops.arity_mismatches(sites, arity) == [
        "mod.py:6 btb_x_d64 (declares 3, handed 2)",
        "mod.py:9 btb_x_d64 (declares 3, handed 2/3)",
        # the head widths of one kernel declare one count: the branch without the row map hands it one short
        "mod.py:10 btb_x_d128 (declares 3, handed 2/3)",
        "mod.py:10 btb_x_d64 (declares 3, handed 2/3)",
    ]
    assert cuda_ops.uncounted_launches(sites) == ["mod.py:8"]


def test_kernels_parameters_are_read_off_their_definitions() -> None:
    """the parameters each kernel declares: an entry point spelled out (its name on the line after its bounds), one a
    macro makes for each head width, one whose list is an object-like macro"""
    arity = cuda_ops.defined_arity()
    assert set(arity) == cuda_ops.defined_kernels() and all(n > 0 for n in arity.values())
    assert arity["btb_gemm_mma_bf16"] == arity["btb_gemm_mma_small_bf16"] == 6  # w, x, y, R, C, T
    assert arity["btb_attn_flash_d64"] == arity["btb_attn_flash_d256"] == 18
    assert arity["btb_attn_flash_prefill_d128"] == arity["btb_attn_flash_prefill_kq_d128"] == 14  # ATTN_FLASH_PF_ARGS


def test_every_launch_in_btb_is_read() -> None:
    """btb's own launches all name their kernels and count their arguments as the parse reads them, launch only
    loaded kernels, and hand each kernel its parameters"""
    sites = cuda_ops.launch_sites()
    assert cuda_ops.unresolved_launches(sites) == []
    assert cuda_ops.uncounted_launches(sites) == []
    assert cuda_ops.arity_mismatches(sites) == []
    assert cuda_ops.launched_kernels() <= cuda_ops.loaded_kernels()
