"""The CUDA card-kernel matrix as a suite gate. The card runs only where a GPU is present, so this is the
structural half: the kernels the engine loads and the kernels native/cuda defines must be the same set. A new
card kernel that the engine does not load, or a loaded name with no definition, fails here."""

from __future__ import annotations

from . import cuda_ops


def test_loaded_and_defined_kernels_agree() -> None:
    assert cuda_ops.gaps() == [], cuda_ops.gaps()


def test_matrix_is_not_empty() -> None:
    """a parse that silently found nothing would make the agreement test vacuous."""
    assert cuda_ops.loaded_kernels(), "parsed no _Cuda.KERNELS from native.py"
    assert cuda_ops.defined_kernels(), "parsed no kernels from native/cuda"
