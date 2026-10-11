# Benchmarking

- `btb bench`: the machine. What the hardware is, and how fast each op btb runs goes on it.
- `btb bench MODEL --prompts ...`: one model, btb only.
- `bench/matrix.py`: btb across every device configuration and model on the machine, and the rival engines.
- `bench/compare.py`: one rival engine, run from `.venv-compare`.

The model benches all use one question set (`bench/questions.jsonl`) at greedy decoding (temperature 0), so the
numbers are reproducible and comparable across machines and engines. `tok/s` is the decode rate after the first
token, averaged over the prompts.

## The machine

```sh
btb bench
```

The report runs without a model, in two parts.

**What the machine is**, as the OS and the drivers report it:

- **CPU**: the model, its cores (a hybrid's performance and efficiency cores apart) and threads, and the rated
  and boost clocks the OS gives.
- **CPU features**: what the native library detects (AVX2, FMA, AVX-VNNI, AVX-512 and its BF16/VNNI parts, AMX,
  NEON, dotprod, i8mm, SVE), the kernel tier it picked, and torch's own tier.
- **Caches**: each level's instances: how many, their size, and how many threads share one (a P-core's own L2,
  an efficiency cluster's shared L2, each chiplet's L3), and the level's total. Every OS is read to the same
  form, so a CPU reports the same on Windows, Linux and macOS. The totals count every instance, as `lscpu` and
  Task Manager do: a two-chiplet Ryzen's L3 is 64 MB.
- **Memory**: total and available. The modules the firmware lists (SMBIOS): their type, form, size, rated and
  running speed in MT/s, and maker and part. Where the board names the channels, the channel count and the peak
  rate they make. Linux's SMBIOS table is readable by root only; without root the modules come from the copy
  systemd's udev keeps (systemd 248 and later). macOS lists no modules.
- **Each card**: memory, compute capability, SMs and clock, L2. The memory bus and clock with the peak rate they
  make. The PCIe link now and at most, the driver, and the power limit.
- **The drives**: each folder's free space.

**What it does**: each op timed where it runs, and swept across the sizes it runs at. A sweep shows where a
cache ends, where a call's fixed cost stops mattering, and where an op turns from the memory's rate to the
compute's.

- **memory**: the host's read rate by working set (32 KB to 1 GB, plus half and twice each cache size the OS
  lists, so every cache's edge falls between two points), on every thread and on one, and its copy.
  A set is re-read in place, so one that fits a cache is read from it and the rate steps down at each cache's
  edge. Then the native gemv's stream by thread count, where the memory saturates. Rates past the L3 are shown
  as a share of the modules' peak.
- **cpu**: the native kernels.
  - The bf16 gemv by weight size (512² to 8192²), held in a cache and streamed.
  - The gemv by rows (1 to 256) against torch's float32 GEMM by rows (16 to 2048), and the row count from which
    the GEMM is faster on this machine, beside the 64 the engine switches at.
  - Torch's bf16 matmul.
  - A mixture's grouped experts by count, MXFP4 by rows (as stored, grouped, and as a GGUF stores it), FP8 and
    pack-12 by rows.
  - Decode attention by context (512 to 128k rows) and over a tree's nodes by count.
  - The DeltaNet step, the sampler by vocabulary, and torch's prefill attention by rows.
- **cuda:N**: the card's own kernels and torch's.
  - The card's read by working set (1 MB to 1 GB, its L2 against its DRAM) and copy.
  - btb's gemv by rows (1 to 32), plain and on the tensor cores.
  - The one-row gemv by weight size, streamed and held in the L2, and the tensor-core gemv at 16 and 8 rows a
    block on a short weight.
  - The fused gate and up, the down projection with the activation folded in, the host's bit-exact gemvs, and
    the MXFP4 widening.
  - Decode attention by context a head a block, a tree's verify a head a block and a KV group a block, and a
    fork's rows by count.
  - The q/k norm with rope and cache write, the norms, the activations, the router and the shared-expert
    combine, Qwen4's output gate (alone and folded into o_proj), the DeltaNet step, and the sampler by
    vocabulary.
  - cuBLAS by rows (1 to 4096) and square by size, torch's prefill attention by rows, and grouped experts by
    the rows each takes.
- **bus**: copies each way by size (4 KB to 256 MB), pinned and pageable, as the engine pays them (the copy and
  the wait). They are held to the PCIe link, read while a copy is in flight because a card trains its link down
  when idle. Then a hidden row's round trip.
- **mlx** (Apple silicon): the Metal gemv by weight size and decode attention by context.
- **drive**: a file written in the folder, each write flushed, 32 MB and 1 MB at a time. Then btb's own
  unbuffered reader on it:
  - reads one at a time by size (4 KB to 128 MB, an expert's span among them);
  - an expert's span and a 4 KB read with 4 and 16 in flight;
  - the reading rule btb's Route would draw from those reads.

  The file is removed afterwards.

Every point is sampled until its median's 95% confidence interval is within `--ci` of the median (default
±2%), or for `--max-seconds` (default 15), whichever comes first. A quiet machine settles most points in well
under a second. A point that is still wider when its time runs out is marked `~`, and its interval is shown as
it stood. The interval is distribution-free (the order statistics around the median), so a noisy machine's
stalls widen it instead of dragging the median. Streamed weights rotate through pools sized past the machine's
own L3 and L2, so each call reads memory the way a decode step's layers do. Each row's JSON carries its place in
its sweep (`at`: the bytes, rows, context, depth...) for plotting.

- `--drive DIR`: time the drive under DIR instead of the Hugging Face cache (repeatable).
- `--skip memory,cpu,card,bus,mlx,drive`: leave sections out.
- `-d cuda:N`: the card to time (default: the first). `-d cpu` times no card.
- `--json`: the whole report as one JSON object on stdout. `--out FILE` writes the same JSON to a file.

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
