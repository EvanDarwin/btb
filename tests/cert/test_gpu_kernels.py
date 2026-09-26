"""The GPU kernel table holds against the source: every Metal and CUDA kernel lands in a category, each backend
btb has reads some kernels, and a backend with none is None, never an empty column."""

from __future__ import annotations

from . import gpu_kernels, spec


def test_every_kernel_has_a_category() -> None:
    assert gpu_kernels.orphans() == [], "a GPU kernel no category claims - add its pattern to CATEGORIES"


def test_each_backend_reads_its_kernels() -> None:
    for hw, read in gpu_kernels.KERNELS.items():
        stems = read()
        assert stems, f"no {hw.value} kernels read - the source moved or the pattern broke"
        assert all(s.startswith("btb_") for s in stems), sorted(stems)


def test_a_template_reduces_to_its_stem() -> None:
    assert gpu_kernels._stem("btb_{kind}_mv_{rows}_{nb}_{tr}") == "btb_<kind>_mv"
    assert gpu_kernels._stem("btb_gemm16_x{xb}") == "btb_gemm16"
    assert gpu_kernels._stem("btb_attn_nodes_{'i8' if kq else 'bf16'}") == "btb_attn_nodes"
    assert gpu_kernels._stem("btb_kv_store") == "btb_kv_store"


def test_no_backend_is_none() -> None:
    t = gpu_kernels.table()
    for row in t["rows"]:
        for hw in spec.Hardware:
            if hw is spec.Hardware.CPU:
                continue
            ks = row["kernels"][hw.value]
            assert (ks is None) == (hw in spec.NO_BACKEND), (row["category"], hw, ks)
