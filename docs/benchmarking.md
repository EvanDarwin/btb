# Benchmarking

Three tools, all on one question set (`bench/questions.jsonl`) at greedy decoding (temperature 0) so the numbers
are reproducible and comparable across machines and engines:

- `btb bench` — one model, btb only.
- `bench/matrix.py` — btb across every device configuration and model on the machine, and the rival engines.
- `bench/compare.py` — one rival engine, run from `.venv-compare`.

`tok/s` is the decode rate after the first token, averaged over the prompts.

## One model

```sh
btb bench Qwen/Qwen3-4B --prompts bench/questions.jsonl
```

Times each prompt at 64 / 256 / 1024 new tokens (`--new`). When the model speculates it times both the
one-token-at-a-time baseline and the speculative run, reporting both rates, first-token latency, tokens per pass,
peak RAM/VRAM, and how many prompts the two decoded identically (`identical N/M`; speculation is exact, so a
healthy run is `N/N` and anything less is a bug). Other flags: `--rows I,J`, `--label`, `--out JSONL`.

The placement and speed knobs that move these numbers are in [Tuning btb](./tuning.md).

## The matrix

```sh
python bench/matrix.py                          # every config × every cached model
python bench/matrix.py --filter qwen3 --markdown
```

One `btb bench` process per (model, device configuration) cell. It writes `bench/results/<host>-<date>.json` and
a per-cell log directory beside it. Useful flags:

- `--devices LIST` / `--models LIST` / `--filter REGEX` — narrow what runs (default: every configuration the
  machine can run × every complete model in the Hugging Face cache).
- `--fp32` — also time each bf16 cell in float32.
- `--markdown` — print the table in the README's shape.
- `--resume JSON` — reuse an earlier run's finished cells, run only the rest; `--render JSON` reprints a run's
  tables; `--dry-run` prints the plan and stops.
- Guards that kill a runaway cell: `--timeout`, `--mem-floor`, `--swap-cap`, `--disk-floor`.

When a comparison configuration is in the plan, the matrix checks `.venv-compare` and offers to set it up.

## Comparing against other engines

The rivals — `mlx-lm` (Apple silicon), `llama-cpp` (llama-cpp-python), `airllm` — pin their own dependency sets
to the versions the README's rows were measured with, and those must not mix with btb's. They live in a separate
virtualenv, **`.venv-compare`** at the repo root, listed in `bench/requirements.txt`. btb is not installed there;
the checkout goes last on the path so the venv's own packages win.

### Setting up `.venv-compare`

The matrix builds it for you when a comparison row could run:

```
$ python bench/matrix.py --filter qwen3
[matrix] .venv-compare is not ready: torch missing, ...
[matrix] set it up now (a venv, torch for the card, bench/requirements.txt)? [y/N]
```

It creates the venv, installs torch from the CUDA index matching the card (plain torch without one), then
`bench/requirements.txt`. `llama-cpp-python` builds with `-DGGML_CUDA=on` where there is a card, so the CUDA
toolkit must be present (see [Building btb](./building.md)).

By hand (POSIX; on Windows use `.venv-compare\Scripts\` and set `CMAKE_ARGS` in the environment):

```sh
python -m venv .venv-compare
.venv-compare/bin/python -m pip install --upgrade pip
.venv-compare/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu124   # match your card's CUDA; drop --index-url with no card
CMAKE_ARGS=-DGGML_CUDA=on .venv-compare/bin/pip install -r bench/requirements.txt                 # drop CMAKE_ARGS with no card
```

### Running a rival

The matrix drives `compare.py` through `.venv-compare` on its own. To run one directly:

```sh
.venv-compare/bin/python bench/compare.py Qwen/Qwen3-4B --tool mlx-lm \
    --prompts bench/questions.jsonl --out bench/results/compare.jsonl
```

- `--tool mlx-lm|airllm|llama-cpp`, `--new`, `--rows`, `--budget S` (drop the remaining lengths after S seconds).
- `llama-cpp` converts the model to GGUF first: `--gguf` reuses an existing one, `--gguf-type` sets the outtype,
  `--n-gpu-layers` the offload (`-1` all, `0` CPU-only), `--keep-gguf` keeps the conversion.
- `airllm` writes per-layer splits: `--airllm-split-dir`, `--device cuda|cpu`, `--keep-splits`.
- Run it from `.venv-compare` — a different interpreter is a different measurement, and `compare.py` warns when
  it is not the one.

## Charts

```sh
python bench/charts.py            # one SVG per model and machine into assets/bench/
python bench/charts.py --embed    # also rewrite the README with the charts above each table
```

The newest result of a configuration wins.
