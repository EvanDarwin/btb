# The MLX megakernel

On Apple silicon a dense Qwen3 decode or verify pass runs as one Metal dispatch: a persistent grid of
threadgroups walks an instruction table, one entry per operation of the pass, with a grid barrier between
entries. It is the default where it builds (`--mlx-mega 0` takes the per-op path) and its tokens are the per-op
path's, bit for bit.

## Why one dispatch

A small model's step is a few hundred operations, each a Metal launch with its ramp and drain and its share of
the per-pass graph build, on a pass whose arithmetic is a few hundred microseconds. The weights are read once
either way, so the bandwidth bound does not move; what one dispatch removes is everything between the
operations. On an M3 Pro the small models run 15 to 40% ahead of the per-op path and the 4B at parity, where
the weight reads are the pass.

## The pass

For a layer $i$ with input $h$ (bf16, $T$ rows of $H$), the kernel computes

$$
\begin{aligned}
x &= \mathrm{rmsnorm}(h;\, \gamma_{1}) \\
[\,q \mid k \mid v\,] &= W_{qkv}\,x, \qquad q, k \leftarrow \mathrm{rope}\big(\mathrm{rmsnorm}(q;\gamma_q),\ \mathrm{rmsnorm}(k;\gamma_k);\ p\big) \\
a &= \mathrm{softmax}\!\left(\frac{q\,K^{\top}}{\sqrt{d}}\right) V \qquad \text{over the arena's rows and the pass's own} \\
h' &= h + W_{o}\,a, \qquad x' = \mathrm{rmsnorm}(h';\, \gamma_{2}) \\
m &= \mathrm{silu}(W_{g}\,x') \odot W_{u}\,x' \\
h'' &= h' + W_{d}\,m
\end{aligned}
$$

and at the end $\mathrm{argmax}\big(W_{head}\,\mathrm{rmsnorm}(h'';\gamma_{f})\big)$ a row. $q$, $k$ and $v$ are
one matvec ($W_{qkv}$ is the three projections in one slot) and so are the gate and up ($W_{gu}$), so a layer is
ten instructions and a pass is

$$
N = 10L + 6
$$

instructions: `embed` and the first `rmsnorm` before the layers, `fnorm`, `head`, `argmax_part` and `argmax`
after them. Qwen3-0.6B is 286 a pass, the 4B 366.

## The instruction table

A row is sixteen `uint32`: the op, its item count, five buffer references as (buffer id, byte offset), three
integers and one float's bits. The buffers are the weight slots (at most 20, the pool's blocks; every projection
of every layer and the head must sit in one), the scratch, the constants (the norms, bf16 a layer at one stride,
the final norm in float32), the K/V arena, the rotary frequencies, the ids, the positions, the node metadata,
the path and the argmax output. A table is built once per (rows, attention splits, arena, argmax) and kept; the
`past` of a pass is written into the `qkrope` rows alone, so a kept table serves every position.

## The grid and the barrier

The grid is $G = 36$ threadgroups, all resident at once: a barrier that waits for a threadgroup that has not
been scheduled never clears. An instruction's items are dealt to the threadgroups round-robin; when a
threadgroup has done its share it fences, thread 0 adds one to the instruction's counter $c_i$ and spins until

$$
c_i = G,
$$

then the threadgroup fences again and moves to instruction $i+1$. A spin past three million iterations adds one
to a failure counter instead of waiting forever; the host evaluates the counters with the result and raises on
a non-zero failure count rather than trusting the bytes. Every scratch location is written once a
pass and read only after the barrier that follows its write, which is the coherence a Metal dispatch gives
across threadgroups; nothing else in the kernel relies on ordering between them.

## The matvec

A matvec of $R$ output rows over $K$ columns is tiled $8 \times 8$ on simdgroup matrices: a threadgroup's eight
simdgroups stage $(8 + R_t)$ rows by $K_T = 64$ bf16 columns (32 KB at most) a step, $R_t$ the tile's 8 or 16 rows, and accumulate their fragments
in float32 down the whole row, so a row's chain of adds is the same whether the pass is one row or sixteen. The
threadgroups a matvec takes is the fewest whose simdgroups cover the $\lceil R/8 \rceil$ tiles in the same number
of rounds as the whole grid would: a last round run by a few streams is latency-bound while the rest of the
grid waits at the barrier. The loads of a step are issued one step ahead into registers.

## Attention and the arena

The K/V cache of every layer is one arena, a buffer of `cap` rows a head per layer, and the `qkrope` op writes
the pass's $k$ and $v$ rows straight into it; nothing is appended on the host. Attention runs over the arena in
blocks of 1024 rows, an item per (kv head, block, row): each item leaves a partial $(m_s, l_s, \mathrm{acc}_s)$ in
the scratch, the block's running maximum, denominator and numerator, and `fold` merges the blocks of a row,

$$
m = \max_s m_s, \qquad l = \sum_s l_s\, e^{\,m_s - m}, \qquad a = \frac{1}{l}\sum_s \mathrm{acc}_s\, e^{\,m_s - m},
$$

the same merge the per-op path's node attention makes, in the same order. A tree pass's rows attend over the
arena and their own ancestors by the node metadata and path buffers; the positions a row may carry are the
natural ones, $\text{past} + \text{depth}$, and the kernel refuses any other. A pass attends over at most
$8 \times 1024 = 8192$ rows.

## The scratch

One slot a layer, sized for the widest pass ($R = 16$ rows), and a tail after the layers:

| region | bytes a layer, $R$ rows |
|---|---|
| the layer's input, the normed input, the residuals and the MLP's ($x$, $o$, $h_2$, $x_2$, down, $h_3$, $x_3$) | $7 \cdot 2RH$ |
| $q$, $k$, $v$ and $q$ after rope, the attention out | $2R(H_q + 2H_k)d + 2 \cdot 2RH_q d$ |
| the attention partials $m_s$, $l_s$, $\mathrm{acc}_s$ over 8 blocks | $R H_k \cdot 8 \cdot 8 \cdot g \cdot 4\,(2 + d)$ |
| gate and up, the silu product | $2R \cdot 2I + 2RI$ |

with $g = H_q / H_k$ the query heads a kv head, every region aligned to 256 bytes; the tail is the embedded rows,
the final-normed rows in float32, the logits ($4RV$) and the argmax partials. On the Qwen3 sizes:

| model | scratch |
|---|---|
| 0.6B (28 layers, $H$ 1024) | 258 MB |
| 1.7B (28, 2048) | 272 MB |
| 4B (36, 2560) | 662 MB |

The logits and the partials dominate the small models; the MLP's rows the 4B.

## When a pass takes it

Every condition holds, or the pass runs the per-op path without a word:

- MLX, a dense family the kernels lay out (Qwen3), every layer resident, no 12-bit or affine-quantized layer
- bf16 compute (`--fp32 0`), the rotary whole and unscaled, a head of $64k$ dims, $H$ and $I$ multiples of 8
- every projection and the head in a slot buffer, twenty buffers at most
- 1 to 16 rows, the head and the pick in the pass, no per-layer callback
- an arena cache with room: $\text{past} + T \le \text{cap}$ and $\le 8192$
- a tree pass's positions the natural ones

The prefill, a streamed model, fp32, the hybrids, a mixture of experts and Phi all take the per-op path.

## Sampling

Greedy, the argmax is in the kernel: `argmax_part` reduces each row's logits in 64 parts and `argmax` folds
them, the lowest index among equals. Under a temperature the two instructions are left out of the table and
the logits stay in the scratch, where the pick (`btb/sampling.py`) reads them lazily before the next pass
overwrites them; the sequential and the speculative loops draw the same tokens under one seed.

## What holds it

`tests/test_mega.py`, on Qwen3-0.6B from the local cache: a pass of one to sixteen rows over a fresh cache gives
the per-op path's argmax ids and K rows bit for bit at several past lengths and tree shapes; the speculative and
sequential loops give the per-op loops' tokens, greedy and sampled under one seed; a tree pass refuses
positions that are not its nodes'; every layer of a cache built under the kernel lives in the arena.
