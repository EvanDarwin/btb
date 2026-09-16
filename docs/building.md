# Building btb

The wheels on PyPI cover the common platforms. Build from source to work on the code, target an unusual
platform, or compile the card's CUDA kernels yourself.

## Prerequisites

- **Python ≥ 3.11.**
- **A Rust toolchain** ([rustup](https://rustup.rs)). The native library — the CPU kernels, the DeltaNet step,
  the disk reader — is a Rust `cdylib` built on every platform. Stable, no pinned version.
- **CUDA toolkit / `nvcc`** — only to build the card's decode kernels, and only on Linux or Windows with an
  NVIDIA card. Optional: without it a card runs through torch alone (you get a WARNING banner at load), and the
  CPU and MLX paths never use these kernels.
- **Apple silicon:** nothing extra to build; install the `mlx` extra (below) to use the GPU.

## Build and install

```sh
git clone https://github.com/EvanDarwin/btb && cd btb
python build.py            # native library + card kernels (when nvcc is present) + wheel
pip install dist/*.whl     # on Apple silicon, then: pip install 'mlx>=0.32.2'
```

`build.py` uses only the standard library, so it runs before anything is installed. It builds the native library
with `cargo`, compiles the CUDA fatbin when `nvcc` is on PATH, drops both into `btb/native/<platform>/`, and
writes a platform-tagged wheel to `dist/`.

A run with `-v` prints a `[card] kernels …` line when the card's kernels loaded; without them you get the
no-kernels WARNING banner.

## Working on the code

Create the repo's virtualenv as `.venv` — `build.py check` and `build.py test` run through it:

```sh
python -m venv .venv && . .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -e .                                  # editable; on Apple silicon: pip install -e '.[mlx]'
pip install ruff mypy pytest                       # the check and test tools
python build.py                                    # build the native library into the tree
```

With the library in `btb/native/<platform>/`, the editable install picks up Python edits with no reinstall;
rerun `python build.py` after changing Rust or CUDA.

The gate a change must pass, and the suites:

```sh
python build.py check      # ruff (lint + format), mypy, cargo fmt, clippy
python build.py test       # every suite; --fast is tests/test_unit.py alone (seconds)
python build.py format     # apply ruff and cargo formatting
```

`cargo fmt` and `clippy` come from rustup; if `check` reports them missing, `rustup component add rustfmt clippy`.

## The CUDA kernels

`build.py` compiles `native/cuda/btb_kernels.cu` to a fatbin with SASS for SM 8.0–9.0 (Ampere through Hopper)
and PTX for anything newer (compiled at load). These are the card's single-token decode kernels; wheels ship
them for Linux and Windows only.

- `--cuda` — require `nvcc`; fail rather than build without the kernels.
- `--no-cuda` — skip them.
- `--cuda-only` — build the fatbin and stop.

## Other build flags

- `--target <triple>` — cross-build for another platform (e.g. `aarch64-unknown-linux-gnu`); the wheel and
  `btb/native/<platform>/` are tagged for the target.
- `--skip-rust` — package a library already built.
- `--python <path>` — the interpreter that builds the wheel (default: the one running `build.py`).

## Runtime options

For placement, speed, the speculative tree, and the `BTB_*` knobs, see [Tuning btb](./tuning.md).
