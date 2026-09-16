# Pack-12

A lossless 12-bit re-encoding of a model's `bf16` weights. `btb pack` writes a **sibling model** next to the
original: it holds the transformer layers packed to ~12 bits per weight and a `config.json` that names the
parent every other tensor is read from. The parent is never modified, and the packed model loads and runs like
any other — with **bit-for-bit identical** output.

## Creating a 12-bit model

```sh
btb pack <model> [out]
```

`<model>` is a directory or a Hugging Face repo id. With no `out`, the sibling is written beside the source:

- a cache snapshot gets a `<repo>-pack12` entry in the Hugging Face cache (usable by repo id);
- any other directory gets `<model>-pack12` next to it.

Only `bf16` `layers.*` tensors are packed (into safetensors shards); the small files are copied and their
`config.json` gains a `btb` record — `{"format": "pack12", "source": <parent repo id or relative path>}`. Embeddings,
norms, the head, and any non-`bf16` tensor stay in the parent and are read from it as-is. Budget roughly `0.76×`
the packed tensors' size in new files; `pack` refuses if the target disk lacks the space.

From Python:

```python
from btb.engine import pack_model

out_dir = pack_model("/models/Qwen3-0.6B")  # returns the sibling directory
```

## Invoking it

A 12-bit model is a model like any other — pass it as the path:

```sh
btb run Qwen/Qwen3-0.6B-pack12 -p "..."
btb serve /models/Qwen3-0.6B-pack12
```

```python
import btb

sm = btb.load("Qwen/Qwen3-0.6B-pack12")  # tokenizer at sm.tokenizer
```

The sibling **references** the parent (by repo id, or by a relative path so the pair moves together); it is not
self-contained. Both must be present: packed layers come from the sibling, everything else from the parent.

## How it is lossless

A `bf16` value is two bytes: a low byte and a high byte. Across a weight tensor the high byte takes only a
handful of distinct values (sign and exponent patterns), while the low byte is high-entropy. Pack-12 exploits
exactly that:

- the **low byte** is stored verbatim — 8 bits;
- the **high byte** is stored as a **4-bit code** into a per-tensor table of the 15 most common high bytes;
- the rare high bytes that don't fit the table **escape**: code `15` marks them, and their `(index, high byte)`
  is stored out of line.

Reconstruction recombines the stored low byte with the table lookup (or the escaped high byte) into the original
two bytes — no rounding, no approximation. The result is the same `bf16` bits that went in, so a packed model's
logits and sampled tokens match the parent's exactly. Cost on disk is `8 + 4 = 12` bits per weight plus a 16-byte
table per tensor and the (few) escapes — about `0.76×` the `bf16` size.

## When to use it — and what it costs

Reach for pack-12 when you are **memory- or bandwidth-bound and must keep exact `bf16` output**. btb streams
layers from disk when they don't fit VRAM; at ~`0.76×` the bytes, more layers stay resident and cold reads are
shorter — with results identical to the unpacked model. It is the right tool when a lossy quantization's quality
change is unacceptable.

Costs:

- **Modest savings.** Lossless caps the win near ~24%. Lossy 8- or 4-bit quantization saves far more if you can
  accept a change in output.
- **A decode step.** The 12-bit form is a *storage* format: layers are widened back to `bf16` before compute, so
  there is unpacking work at read time.
- **Not standalone.** It points at the parent; ship or keep both.
- **`bf16` only.** MXFP4 experts and non-`bf16` tensors are already compact and are left in the parent — a
  MoE stored in MXFP4 gains little.
- **Extra disk.** The packed layers are a second copy alongside the original.

## Reading it from plain torch (the mixin)

Any torch program can consume a 12-bit model without the btb engine, via one call that returns a plain state
dict of `bf16` tensors — packed layers widened, everything else read from the parent:

```python
from btb.pack12 import load_state_dict

state = load_state_dict("/models/Qwen3-0.6B-pack12", device="cpu")  # {weight name: bf16 tensor}
model.load_state_dict(state, strict=False)
```

For a cache sibling addressed by repo id, resolve it to its directory first:

```python
from btb import resolve
from btb.pack12 import load_state_dict

state = load_state_dict(resolve("Qwen/Qwen3-0.6B-pack12"))
```

To widen tensors yourself (e.g. lazily, or on the GPU), the lower-level pieces are exported too: `entries(model_dir)`
returns the per-tensor records (shard, offsets, table, shape), and `unpack_bf16(...)` reconstructs one tensor from
its packed bytes.