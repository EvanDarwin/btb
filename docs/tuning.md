# Tuning btb

btb plans its own placement from the free memory it sees at load and picks a configuration for the machine, so
most of the time you set nothing. This guide is for when you want to override it: move weights and cache between
VRAM, RAM, and disk; get the most tok/s out of a given machine; or drive the speculative tree.

Everything here is available three ways, and an environment variable wins over the matching `btb.load` option:

- a flag on any command that loads a model (`btb run|chat|bench|serve|ollama|pi`),
- a keyword to `btb.load(...)`,
- an environment variable (`BTB_*`), listed at the end and in `btb --help`.

The [README](../README.md) has the full flag table with every default; this page is about which knob to reach
for and why.

## See what the planner decided

Before changing anything, look at what btb already chose.

- `-v` prints the placement, the tiers, and per-turn timings — where each layer landed, and the tok/s and
  tokens-per-pass you're actually getting.
- `--profile DIR` writes `report.json` (the engine's ledger) and, for a MoE model, `events.npz` (the
  expert-store trace).
- `btb bench PATH --prompts prompts.jsonl` times a prompt set at several answer lengths and reports tok/s and
  tokens per pass per length. Use it to compare two configurations instead of eyeballing a single run.

## Memory: moving weights and cache around

btb fills three tiers in order — VRAM, then RAM, then the drive — and the planner decides the split from the
free memory at load. The knobs below either change how much it may take or force a specific split.

**How much it may take.** `--ram-reserve GB|%` and `--vram-reserve GB|%` set what btb leaves for everything else
(default: 10% of RAM; 0.5 GB or 8% of a small card). Lower them to fit more resident, raise them if the machine
needs to do other work. `--vram-watch 1` (default) frees layers back to the OS when another program claims the
card and reclaims them when it frees; `--vram-watch 0` plans once and holds the placement, which is faster but
will make the other program OOM instead of btb.

**Where the layers live.** The planner keeps as many layers resident as fit. To force it:

- `--resident-last N` pins the last N layers on the GPU.
- `--cpu-layers N` runs the first N layers on the CPU from RAM.
- `--resident-head 0|1` puts the output head on the CPU or GPU (it is large — for a small model it can be a real
  slice of VRAM).
- `-d cpu|mlx|cuda|cuda:N` picks the device outright.

Anything that fits neither VRAM nor RAM streams from the drive per token, which is the slow path.

**The attention cache.** On CUDA, `--kv-host 1` keeps the cache in RAM with the weights on the card — this is
how you run a long context whose cache would not fit VRAM (see the long-context rows in the README). Unset, the
planner prices both and keeps the cache on the card unless the layers it would evict cost more to stream. On
MLX, `--kv-bits 8` halves the cache to int8 (lossy, ~0.4% of a row's largest magnitude). `--context N` sets the
window; past the model's own window btb applies YaRN automatically.

**Mixture-of-experts.** A large MoE streams its experts from disk through a RAM store. To make that hurt less:

- `--expert-cache-gb GB` caps the RAM the store may grow into (default: free RAM above the reserve). More RAM,
  fewer disk reads.
- `--vram-experts-gb 0|auto|GB` (`BTB_VRAM_EXPERTS_GB`) seats bf16 experts on the card ahead of the store.
- keep the trunk resident on the card and stream only the experts.
- `btb pack` writes a lossless [12-bit copy](./pack-12.md) at 0.75× the bytes, so more experts stay cached and
  cold reads are shorter, with identical output.
- the residency policy, the disk readers, and the router lookahead are the [disk scheduler](./disk-scheduler.md)
  and its `BTB_*` knobs below.

## Speed

In rough order of effect:

1. **Fit the model resident.** Streaming layers or experts from disk is single-digit tok/s; the same model
   resident is tens to hundreds (see the README benchmarks). Getting the model to fit — a smaller model, a
   [12-bit pack](./pack-12.md), a lower reserve, `--kv-host 1` to move the cache off the card — is worth more
   than every other knob combined.
2. **Keep bf16.** `--fp32 0` is the default on the GPU and is the fast path; `--fp32 1` is for precision work
   (and is genuinely faster on some CPUs — measure with `btb bench`).
3. **Leave speculation on** and tune the tree (next section). It is the largest software lever once the model is
   resident.
4. **Apple silicon:** `--mlx-mega 1` (default where it builds) runs the dense pass as one Metal dispatch — the
   small models 15–40% ahead of the per-op path on an M3 Pro. See [the megakernel](./mega-mlx.md).
5. **MoE on disk:** the memory knobs above, plus the scheduler knobs below.

Confirm every change against `btb bench`, not a single run — tok/s moves with the prompt and the answer length.

## The speculative tree

Speculation is where the tok/s gains live, and it is exact: the drafted tree is verified against the model's own
next-token distribution, so a speculative run yields the same tokens as decoding one at a time — greedy or
sampled, and identical under a fixed seed. It is independent of how you pick tokens (see [Token
selection](#token-selection-greedy-and-sampling) below); leaving it on costs you nothing but speed.

Each step drafts a small tree of candidate tokens (from the model's MTP drafting head if it has one, otherwise
an n-gram drafter), then the model verifies the whole tree in one pass. Accepted drafts are free tokens; the
metric is **tokens per pass**, which `-v` and `btb bench` report. On by default for every model that can do it.

- `--tree-budget N` — tree size per step. Bigger tree, more candidates per pass, more verify cost. `0` turns
  speculation off (same as `--greedy`). Default 14–16 depending on device/head.
- `--v-max N` — drafted tokens verified per step; `0` decodes one at a time. Default 4, `0` for MoE.
- `--tree-min-prob P` — drop draft branches below this path probability (default 0.15). Lower keeps more
  speculative branches alive.
- `--tree-step-mass P` — only extend the tree when the nodes to extend hold at least this much path mass
  (default 0.5); `0` always steps.
- `--draft-vocab N` — the drafting head scores only the first N token ids (the frequent part of the vocab), a
  smaller read per draft step; verified output unchanged. Default 32768 with a head, `0` scores all.
- `--draft-bits 4|8|16` — MLX only: pack the drafter's own weights to 4/8 bits at first use (default 8). The
  model's weights are untouched and the drafts are always verified, so this trades a little acceptance for a
  cheaper draft step.
- `--draft-temperature R` — under a sampling temperature, draft the tree at R× it (default 1). The verified
  answer's distribution is identical at any R; only which drafts get accepted changes.
- `--ngram-p P` — acceptance threshold for the n-gram drafter (default 0.9), the proposer used when the model
  has no drafting head.

Turn speculation off with `--tree-budget 0` (the `run` command also takes `--greedy` for this). That switches
off the tree, not sampling — it is unrelated to greedy token selection below.

## Token selection: greedy and sampling

How each token is picked, independent of speculation: the same knobs apply whether or not the tree is on, and on
the servers a request's own fields override them.

- `--temperature T` — `0` (the default) takes the single likeliest token; this is greedy decoding. Above `0`,
  the logits are scaled by T and a token is drawn, wider as T rises.
- `--top-p P` — draw only from the smallest set of tokens whose probability reaches P (default 1: all tokens).
- `--top-k K` — draw only from the K likeliest tokens (default 0: all tokens).
- `--seed N` — the draw's seed; a prompt and a seed reproduce their answer exactly (default: drawn per call and
  reported in the stats). Sampling is deterministic given a seed.

The `run --greedy` flag names speculation (decode one token at a time), not selection. Greedy selection is just
the default, `--temperature 0`, and speculation stays on under a temperature — `--draft-temperature` above tunes
how the drafter samples its tree.

## Environment variables

`BTB_*` are the low-level escape hatches — mostly for benchmarking and checking numerics. A kernel swap is
bit-for-bit identical to the default unless noted; each variable is read only on the tier that uses it and
ignored elsewhere.

| variable | default | effect |
|---|---|---|
| `BTB_API_KEY` | none | bearer key the servers require, and the `--api-key` default |
| `BTB_POOL` | 1 | seed an MLX memory pool at load (Apple silicon); `0` skips it |
| `BTB_CPU_GEMM` | 1 | Mac CPU tier: bf16 prefill matmuls on MLX's CPU stream; `0` (or `--fp32 1`) keeps float32 |
| `BTB_FUSED_NORM` | 1 | CUDA: fused RMSNorm; `0` restores the module's |
| `BTB_FUSED_MLP` | 1 | CUDA: in-place SwiGLU (module and card graph); `0` restores the module's |
| `BTB_FUSED_ROPE` | 1 | CUDA card graph: rope folded into `addcmul`; `0` runs it as separate launches |
| `BTB_CARD_MMA` | auto | card GEMV kernel: `0` fp32 chain, `1` tensor cores; unset, the warm-up picks per width |
| `BTB_VRAM_EXPERTS_GB` | 0 | bf16 experts seated on the card: `0`, `auto`, or GB (the `vram_experts_gb` option) |
| `BTB_HEAD_GEMV` | 1 | native (non-MLX) CPU: bf16 head gemv over a float32 copy; `0` forces the copy |
| `BTB_LOOKAHEAD` | off | router lookahead on one-row passes, per depth (e.g. `10,6`); `0` off |
| `BTB_LOOKAHEAD_ROWS` | 10,6 | the same on multi-row passes (prefill, verify); `0` off |
| `BTB_BUS_PASS` | 1 | store residency: the Bus Pass over the plain line (the `bus_pass` option) |
| `BTB_STORE_PIN` | 0 | store RAM pages: `0` pageable, `1` pinned, `auto` beside a card (the `store_pin` option) |
| `BTB_STORE_PADDED` | 1 | native reader: read into the slot's padded region; `0` bounces through a buffer |
| `BTB_ROUTE_DEPTH` | probe | drive readers in flight, over the load-time probe's choice |

The store, the lookahead, the residency policy, and the readers are covered in full in
[the disk scheduler](./disk-scheduler.md).
