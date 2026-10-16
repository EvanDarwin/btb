# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's GGUF decoders (native/cuda/btb_quant_decode.cuh) compiled for the host: every stored type's `decode`
over random valid blocks against gguf's dequantization, so the card kernels' numbers are checked where no card
is. The kernels read each weight through these and nothing else."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from btb.kinds import Quant, quant_key, stored_quants
from tests.helpers import stored_blocks

CUDA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "native", "cuda")

# the CUDA qualifiers and intrinsics the decoders use, as plain C++ (no C++ library to link)
SHIM = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define __device__
#define __forceinline__ inline
#define __constant__ static const
static inline uint32_t __ldcs(const uint8_t* p) { return *p; }
template <class T> static inline T __ldg(const T* p) { return *p; }
static inline uint16_t __ushort_as_half(uint16_t h) { return h; }
static inline float __half2float(uint16_t h) {  // IEEE binary16 to binary32: subnormals, infinities and NaNs too
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0 && m == 0) {
        f = s;
    } else if (e == 0) {
        e = 127 - 15 + 1;
        while (!(m & 0x400)) { m <<= 1; e--; }
        f = s | (e << 23) | ((m & 0x3FF) << 13);
    } else if (e == 31) {
        f = s | 0x7F800000 | (m << 13);
    } else {
        f = s | ((e + 127 - 15) << 23) | (m << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}
"""

# argv: key, raw blocks, codebook (empty file: none), sign table (empty: none), out (float32 a weight)
MAIN = r"""
#include "btb_quant_decode.cuh"

static uint8_t* slurp(const char* path, long* n) {
    FILE* f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    *n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = (uint8_t*)malloc(*n ? *n : 1);
    if (*n && fread(b, 1, *n, f) != (size_t)*n) exit(3);
    fclose(f);
    return b;
}

template <class Q>
static void run(const uint8_t* raw, long nb, QTab t, const char* out) {
    long units = nb / Q::UB;
    float* y = (float*)malloc(units * 256 * sizeof(float));
    for (long u = 0; u < units; ++u)
        for (int ib = 0; ib < 8; ++ib)
            for (int lane = 0; lane < 32; ++lane) y[u * 256 + 32 * ib + lane] = Q::decode(raw + u * Q::UB, ib, lane, t);
    FILE* f = fopen(out, "wb");
    fwrite(y, sizeof(float), units * 256, f);
    fclose(f);
}

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    long nb, ng, ns;
    const uint8_t* raw = slurp(argv[2], &nb);
    const uint8_t* grid = slurp(argv[3], &ng);
    const uint8_t* signs = slurp(argv[4], &ns);
    QTab t = {ng ? (const int8_t*)grid : nullptr, ns ? signs : nullptr};
"""


@pytest.fixture(scope="module")
def decoder(tmp_path_factory: pytest.TempPathFactory) -> str:
    """the decoders built into one host program, every stored type's `decode` behind its key"""
    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.sysconfig import customize_compiler

    d = tmp_path_factory.mktemp("card_decode")
    keys = [quant_key(q) for q in stored_quants()]
    dispatch = "".join(
        f'    if (!strcmp(argv[1], "{k}")) {{ run<{k}>(raw, nb, t, argv[5]); return 0; }}\n' for k in keys
    )
    src = d / "decode.cpp"
    src.write_text(SHIM + MAIN + dispatch + "    return 1;\n}\n", encoding="utf-8")
    cc = new_compiler()
    customize_compiler(cc)
    msvc = cc.compiler_type == "msvc"
    objs = cc.compile(
        [str(src)], output_dir=str(d), include_dirs=[CUDA], extra_preargs=["/std:c++17"] if msvc else ["-std=c++17"]
    )
    exe = str(d / "decode")
    cc.link_executable(objs, exe, libraries=[] if msvc else ["m"])
    return exe + (".exe" if msvc else "")


@pytest.mark.parametrize("q", stored_quants(), ids=lambda q: q.value)
def test_the_cards_decoder_reads_each_stored_type_as_gguf_does(decoder: str, q: Quant, tmp_path: Path) -> None:
    """every weight of 8 rows x 512 of random valid blocks, decoded as the card's kernels decode it, within f32
    rounding of the package's dequantization (a sub-scale or a codebook entry read from the wrong place is off by
    a whole step)"""
    from gguf import GGMLQuantizationType, dequantize

    from btb.gguf import QuantWeight

    rows, cols = 8, 512
    raw = stored_blocks(q, rows, cols, stored_quants().index(q))
    w = QuantWeight.of(torch.from_numpy(raw), q, rows, cols)
    paths = {n: tmp_path / n for n in ("raw", "grid", "signs", "out")}
    paths["raw"].write_bytes(raw.tobytes())
    paths["grid"].write_bytes(b"" if w.grid is None else w.grid.numpy().tobytes())
    paths["signs"].write_bytes(b"" if w.ksigns is None else w.ksigns.numpy().tobytes())
    subprocess.run([decoder, quant_key(q), *(str(paths[n]) for n in ("raw", "grid", "signs", "out"))], check=True)
    got = np.fromfile(paths["out"], dtype=np.float32).reshape(rows, cols)
    ref = dequantize(raw.reshape(rows, -1), GGMLQuantizationType[q.value]).astype(np.float64).reshape(rows, cols)
    err = np.abs(got - ref).max(axis=1) / np.maximum(np.abs(ref).max(axis=1), 1e-30)
    assert float(err.max()) < 1e-6, (q.value, float(err.max()))
