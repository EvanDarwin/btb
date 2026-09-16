# 🚌  Bus Passes, Routes, Timetables, and the story of a disk scheduler

When dealing with very large models that implement MoE (mixture-of-experts); btb has to carefully coordinate
how it will read weights from disk, in what order, and at what time (based on the disk's throughput itself).

The goals of the Bus Pass system included:
* As few disk reads as possible
* Minimal to none incorrect or wasted reads
* Read requests are issued as early as can be known
* Read requests can be dropped when they are no longer needed
* Must support all tiers of drives (HDD/SSD/NVMe)
* Must incorporate the actual read time of the device in its scheduling

----

## Measurements

For the purposes of this document, the measurements you will see are from running **Qwen3.8-Flash-Next**:
* 48 layers
* 512 experts per layer
* 10 experts routed per token
* An expert ~9.83MB of weights

When loaded, btb held 3,573 of 3,607 slots (32.7 GB, only ~14.5% of the 24,576 experts).
The drive the model was hosted on was an NVMe drive @ ~5GB/s.

### 1. What a token costs

Steady state, one-row passes after the store is full:

| per pass (one token)               |         |
|------------------------------------|---------|
| experts asked                      | 480     |
| hits                               | 369     |
| misses                             | 111     |
| hit rate                           | 76.9%   |
| bytes read                         | 1.02 GiB|
| time waited on reads               | 0.254 s |
| the token                          | ~0.55 s |

A layer's call waits 2.5 ms per miss, linearly: one miss 2.5 ms, 7x 15.4 ms. The misses of a call are
issued together; they drain at the drive's ~5 GB/s (seven misses are 69 MB, 14 ms), so the wait is bytes over
bandwidth during the burst, and between bursts the drive idles. Over the token the drive is busy about a
third of the time for modern drives (SSD/NVMe).

## 2. Where the misses come from

- 17,023 of the 24,576 experts were touched in 648 passes. The top 20% of touched experts carry 60% of the
  accesses, the top 30% 72.5%; 441 experts were asked 95 times or more, 1,257 exactly once.
- Of 102,443 misses, 85,420 (83%) were reloads of an expert read before. 98,836 slots were evicted, all for
  the calls' own room (none for the machine in this run); 86% of the evicted experts were asked again, a
  median of 25 passes later. The reload gap: p10 8 passes, median 35, p90 165.
- Rides come in bursts: 59% of rides sit inside a run of consecutive passes; a rider that rode this pass rides
  the next with probability 25% after one ride, 43% after two, 78% after six. Between bursts the gap is a
  median 13 passes, p75 33, p90 86. A rider's next gap correlates 0.13 with its previous one; 41% of next
  gaps fall within 2x of the previous.
- The routing itself is a near-tie at the boundary: the logit margin between the tenth and eleventh expert is
  a median 0.041 (p10 0.006). The router recomputed in float32 on its own input reproduces the card's bf16
  choice 98.8% of the time.

Per layer the misses are even: 1.0 to 4.5 a pass (layer 0 the most, layer 31 the least).

## 3. Residency policies, replayed on the recorded stream

The replay of the recorded access stream under the store's own policy gives 102,446 misses against the
102,443 of the run, so the replay is trusted for the rest. The table describes a variety of methods attempted
to optimize this situation:

| policy at 3,607 slots                                              | misses  | vs LRU |
|--------------------------------------------------------------------|---------|--------|
| LRU (the store today)                                              | 102,446 |        |
| ARC (recency and frequency, the split tuned by ghost hits)         |  94,776 | -7.5%  |
| the Timetable, score $j*g$ (idle time x the rider's own gap)       |  97,063 | -5.3%  |
| the Timetable, score $j-b$ (a long burst earns credit)             | 100,636 | -1.8%  |
| LRU-2, LRU split evenly per layer, the hazard timetable            | ~102k   |  0     |
| the stationary oracle: every rider's whole-run gap distribution,   |  89,439 | -12.7% |
| conditioned on its idle time ($H=16$)                              |         |        |
| the offline optimum (the exact future)                             |  53,513 | -48%   |
| naive timetables (next = last + EMA gap; hold or reschedule)       | 163-236k| worse  |

| capacity                | LRU    | ARC    | optimum |
|-------------------------|--------|--------|---------|
| 3,607 (32.7 GB)         | 102,446| 94,776 | 53,513  |
| 4,508 (+25%)            |  82,912| 81,762 | 44,518  |
| 5,410 (+50%)            |  71,337| 71,119 | 37,119  |
| 7,214 (x2)              |  48,182| 49,118 | 28,087  |

What this says. The riders' individual return times are noisy (the 0.13 correlation), so a schedule built
from each rider's own history caps at 13% below LRU even with the whole run's statistics, and 5 to 7% online.
The optimum's 48% needs the exact future. Protecting the popular experts changes nothing on this stream: LRU
already never lets a rider with 95 rides go (it is always recent), and seats given to a frequency-ranked set
are taken from the recency set, which is where the churn is (the Bus Pass drafts with 300 to 3,246 protected
seats all landed within 1% of LRU, above it as often as below). Capacity is the lever: half again the slots
cut the misses by 30%, twice by 53%.

Why the naive timetables lost badly: a rider whose predicted ride has passed keeps the highest priority
("due any moment") and holds its seat forever; and with bursts the gap EMA is dominated by ones, so first-time
riders, who most often ride again next pass, were predicted far off and bumped first. A lazy heap cannot fix
it either: dues only grow as riders sit idle, so a filed due is a lower bound and the heap's "farthest" is
never the true farthest. The working frame ranks every resident afresh at each pass boundary (3.6k riders, a
millisecond); the scores above were run in it.

The prefill does not flush the store: its experts enter at the eviction end (`keep=False`), and the decode
misses come out the same with or without it in the replay (71.5k against 71.6k). Its reads are random,
though: 26,876 expert reads over five prefills that a sweep of each layer's two tensors in offset order would
serve sequentially.

## 4. The drive, through our reader

The model's shard files on their own drive, the box idle, `btb_read_direct` (unbuffered, one open per call, a
bounce buffer per chunk):

| read                                         | time     | rate      |
|----------------------------------------------|----------|-----------|
| 4 KB, random                                 | 0.19 ms  | (the fixed cost of a read) |
| 64 KB, random                                | 0.24 ms  |           |
| 6.55 MB (a gate_up), random, one at a time   | 2.44 ms  | 2.7 GB/s  |
| 3.28 MB (a down), random, one at a time      | 1.32 ms  | 2.5 GB/s  |
| 6.55 MB random, 2 / 4 / 8 / 16 / 32 in flight | 3.1 / 5.4 / 9.7 / 17.9 / 29.8 ms each | 4.2 / 4.8 / 5.2 / 5.8 / 5.8 GB/s |
| 512 MB sequential, 4 chunks in flight        | 86 ms    | 5.8 GB/s  |
| 64 MB sequential, one chunk, one thread      | 23 ms    | 2.75 GB/s |
| one 6.55 MB read against two 3.28 MB calls   | 2.41 against 2.73 ms | |

In the running engine the same 6.55 MB read took 4.8 ms with nothing else in flight, twice the idle figure,
and nothing else was running on the box. The difference is our own process: every call allocates a 16 MB
scratch buffer, faults it in, reads into it and copies out, while the host's expert matvecs are pulling the
store's bytes through the same memory system.

Each miss is two reads in two files: a layer's gate_up tensor is one whole ~3.2 GB shard and its down tensor
the next shard (the checkpoint's own layout, string-sorted by layer). Within a shard, expert order is offset
order. 8.7% of misses sit next to another miss of the same call in the tensor, so one read could serve both.

## 5. The lookahead: knowing the reads early

The residual stream carries the routing signal layers ahead. With the checkpoint's own routers, each layer's
routing recomputed on a state from earlier in the same pass:

| the router of layer L run on...            | top-10 recalls of the routing / of the misses | top-20 | top-40 |
|--------------------------------------------|------------------------------------------------|--------|--------|
| its own input (the check)                  | 98.8% / 98.1%                                  |        |        |
| the state entering layer $L-1$             | 66.2% / 57.2%                                  | 83.4% / 78.3% | 91.2% / 89.0% |
| the state entering layer $L-2$             | 57.3% / 49.6%                                  | 74.3% / 69.0% | 85.0% / 82.1% |
| the state entering layer $L-4$             | 51.0% / 42.8%                                  | 67.6% / 60.6% | 79.8% / 74.5% |
| the state entering layer $L-8$             | 42.9% / 31.5%                                  | 57.8% / 46.5% | 70.6% / 60.7% |
| the previous token's layer-L state         | 36.4% / 0.3%                                   | 49.3% / 14.8% | 59.4% / 29.0% |
| the previous token's last-layer state      | 10.6% / 4.1%                                   | 17.5% / 8.1%  | 27.0% / 14.9% |

For comparison, the routing of layer L predicted from layer L's own routing at the previous token through
co-occurrence counts learned online recalls 28% of the misses at top-10 and 62% at top-40; the layer's
popular experts alone 1.6% to 8%. The previous token tells almost nothing about the misses (its recent
experts are the hits). The same pass, one or two layers back, tells most of it, and it costs a 2560 x 512
matvec per layer per prediction.

Reading the table as a schedule: at layer L, with L's MoE input in hand, the routers of L+1 and L+2 run on
it; their top-20 minus the residents and the reads in flight are issued. 78% of L+1's misses and 69% of
L+2's are then in flight one and two layers before they are needed, at ~10 ms a layer, against reads of 5 to
10 ms in situ. Most of the 0.254 s a token waits is hidden; the drive has the room (it is idle two thirds of
the token). The wrong predictions cost reads, not seats: they land in a staging ring, not the store.

## 6. The scheduler

Four parts: the Route (the drive and its queue), the Bus Pass (residency), the Timetable (the lookahead) and
the free partial forward (compute as experts land).

### The Route

The Route is the drive's profile, the rule the profile decides, and the queue every read goes through. It
lives in `BatchScheduler`.

**The profile** (`disk(path)`) is taken once per volume on the model's own file, through the Route's own
reader, and kept for the process (`DriveBenchmark`): the plan prices the cold tier from it and the engine
binds the same measurement.

| what                                        | how                                                        |
|---------------------------------------------|------------------------------------------------------------|
| the fixed cost of a read                    | a 4 KB read, median of 8                                   |
| an expert-sized read alone                  | median of 6                                                |
| the burst rate at 1, 4 and 16 readers       | 32 expert-sized reads at random offsets, three bursts a depth (two of 16 on a drive whose one read is over 20 ms), each timed from the readers' release |
| the sequential rate                         | one 128 MB read                                            |
| the copy of an expert span in RAM           | priced on the 128 MB buffer: a span alone fits in the cache and copies at the cache's rate (0.06 ms), a bounce fresh from the drive never does (0.3 ms) |

The readers are started before the clock and each takes the next offset, as the Route's readers do, so the
burst is the drive's rate and not the threads' start-up. On the bench box the probe at load reads about a
fifth under the idle figure (5.24 GB/s at 4 readers against 6.45), beside torch's spinning threads; the
rule's floor absorbs it.

**The rule** (`_disk_rule`):

- Readers in flight: the deepest of 1, 4, 16 whose burst rate is not measurably below the best shallower
  one's; measurably is the two measurements' spread or 5% (10% with two bursts). A tie goes deeper: on the
  NVMe drive 4 and 16 tie alone (6.45 GB/s) and 16 waits a third less beside the compute, where a reader that
  loses its core leaves a short queue idle. `BTB_ROUTE_DEPTH` forces the depth.
- Merging: adjacent reads go as one when a read's fixed cost is four times the copy of a span. A seek is
  (a disk's 50 to 80 ms); the NVMe's 0.13 to 0.20 ms against a 0.3 ms copy is not.
- Predictions in flight: at most half the readers and 50 ms of the drive's time, so a demand read never
  waits long behind them. On a drive whose one read outlasts the window that is none: a prediction there is
  a whole read taken from the layer waiting now, and it cannot be recalled in flight. The queue keeps one
  reader in the prediction class regardless, for the cold ring.

**The queue** is a heap on (priority, path, offset): the layer waiting now first (0), then predictions by
distance (1, 2), then the sweep (3). Within a class the order is the file's, so on a drive that seeks a deep
queue is safe: the drive's own reordering does the rest. `disk_drop(key)` withdraws a lapsed prediction's
reads before they issue; a read already queued for the same bytes is not queued twice. Where the drive
merges, a queued read of the same file and class that starts inside or at the end of the read taken, and
runs past it, goes with it as one read of the union through a bounce, under a 64 MB cap; a prefill's misses
are a quarter adjacent on the 180B, a decode's under one percent.

**The gaps rule**: a prediction issues only while no demand read is in flight. Predictions issued beside a
layer's burst take bandwidth from it: with a cap alone, 71% of the lookahead's reads are used and the wait
is 0.101 s a pass against 0.074 without it.

**Handles and slots.** Each reader holds one share-read handle per shard for the run (`btb_open`,
`btb_read_at`, `btb_close`; a shared lock on Linux): a synchronous handle serializes the reads that share it,
4.5 GB/s on one shared handle against 5.83 on one a thread. Reads land straight in the slot: each part's
region sits on a sector with two sectors of slack, blocks are 4 KB-aligned, and only a span cut by the end
of the file goes through a bounce. Two experts whose padded spans share a boundary sector are two reads; the
dedupe key is (path, offset, length, destination), and without the destination the shared sector folds them
into one read that corrupts MXFP4 scales.

**The cold-layer ring** issues through the Route at the prediction priority, one layer queued while the
previous lands, its reads dropped on stop. It never binds the profile's depth: the rule's depth is for
expert-sized reads, and whole layers at that depth slam a disk, so the ring keeps the Route's default four
readers.

**The drive as it is, not as it was probed.** The probe at load is a prior. The Route keeps its last 64
reads' spans and every sixteenth read prices their bytes over the drive's busy time (the union of the spans,
so the idle between a decode's bursts is not held against it) against the probe's rate at this
depth. Under half of it the drive has slowed: the Timetable withholds predictions, the transition is logged
with both rates and recorded in the profile (kind `drive`); above four fifths it has recovered, said once
each way. The store's saturation verdict is live the same way: the last sixteen one-row passes decide it
afresh, and it is said when first taken and whenever it turns.

**On a drive that seeks**, the store says at load what a missed expert costs and how many experts it seats,
announces a prompt whose reads will take over a minute before the wait, and after its first sixteen one-row
passes says whether the drive is saturated (the median pass's misses' reads outlast its compute). The median,
because the passes after a prefill carry the working set's reload (319 misses against a steady 117 on the
NVMe) and would call a fast drive saturated. The suite runs all of it on a simulated seeking drive
(`tests/drive_sim.py`: one actuator, a seek curve, rotation, a media rate, a queue that reorders or does not,
a simulated clock the probe reads) bound under the real reader's names: the probe measures it, the rule
answers it, and the queue's order and merging are read off the drive's served commands.

**The prefill lane is not built, by measurement.** A sweep of each layer's tensors in offset order reads all
512 experts to serve the 131 a prefill layer asks (26%). A read is 9.83 MB, so its overhead is 18% of its own
transfer, and a sweep wins only where it reads at most 18% more than it needs: past 439 of 512 experts a
layer on NVMe, 443 on a 7200 rpm disk, 507 on a SATA SSD. At the prefill's 26% it loses by 3.5x on every
drive modeled (a steady token asks 1.9 a layer). What the lane would add besides is already in place:
adjacent reads merge in the Route, and the prefill's experts enter the store at the eviction end
(`keep=False`), so the decode misses are the same with or without them (71.5k against 71.6k on the replay).

### The Bus Pass (residency)

`Riders` is the recency line; `BusPass(Riders)` is ARC's shape over it: day riders, regulars, two ghost lists,
the split moved by ghost hits. Selected by `bus_pass`, the default everywhere: 7.5% fewer misses on the
replay, 1 to 8% on the token over two pairs on the NVMe, 11% fewer misses on the replay's warm passes, a
second a token on a disk; bookkeeping its only cost.

The store's blocks are bounded at 1 GB. It grows a whole block at a time only when free RAM is a margin above
the reserve (the larger of a block and a quarter of the reserve), releases below the reserve from the oldest
riders, and never below the largest call. A store that grew and shrank a slot at a time thrashed on the
1024-token rows; a block at a time does not.

### The Timetable (the lookahead)

`lookahead(layer, h)`: the routers of $L+1$ and $L+2$ (the checkpoint's own gate weights, from the resident or
host modules) run on $L$'s MoE input, top-k a row, the rows' union capped at 2k; the misses among them issue
through the Route by distance into a ring of min(64, an eighth of the store) slots. A prediction the layer
asks for is promoted into the store on use; one it does not ask for lapses and its unissued reads are
dropped. `lookahead` sets the one-row passes and `lookahead_rows` the prefill's rows (default (10, 6): top-10
for $L+1$, top-6 for $L+2$).

On one-row passes every setting loses on the NVMe: the reads it takes from the layer's own burst cost more
than the misses it hides, so the one-row default is off. On the prefill's rows its predictions are about
95% right and take about 3 s off the first token, so the row default is on. It withholds itself where the
drive has no room: where the rule allows no prediction in flight, and once the median of the first sixteen
one-row passes finds the drive saturated (89 misses at 78 ms against 380 ms of compute on a disk; 117 at
2 ms against 380 on the NVMe).

Depth two, the $L+1$ attention on $L$'s partial state, is not built: it sharpens predictions that do not pay on
the NVMe, and the partial state's cache exactness is a risk with nothing to buy until a drive is found where
the one-row lookahead pays.

### The free partial forward

`landed(pending)` yields a call's late experts in arrival order, so the hits and each landed batch compute
while the rest are still on the drive. bf16 experts compute as they land. MXFP4 late experts wait for one
another and compute as one group: the grouped kernel's bits depend on the group size, and the receipts are
byte-exact.

`VramSeats`: experts with 8 rides or more take seats on the card (at most four promotions a pass, seats
recycled by recency, the RAM copy kept), and a call multiplies its seated experts on the card in float32 from
`x_card`. Selected by `vram_experts_gb`, "auto" or a size.

## 7. The runs

The bench is rows 0 and 1 of `bench/questions.jsonl` at 64 and 256 new tokens, greedy, under `--profile`, the
same prompts throughout. The token time is the 256-token figure, the mean of the two rows; the misses and the
wait are the steady state of the profile, one-row passes after the store's last growth. The store's slot
count follows the commit charge of the moment (3,119 to 3,607 slots), so miss counts are read with that in
view; the replay at a fixed capacity (section 3) is the clean policy comparison.

| run | reads in flight | residency | slots | s/token at 256, row 0 / row 1 (mean) | misses, 648 passes | waited, whole run |
|---|---|---|---|---|---|---|
| `profile`: 16 readers through `read_direct`, before the Route | 16 | LRU | 3,607 | 0.530 / 0.568 (0.549) | 102,443 | 0.254 s a pass in the steady state |
| `route`: the Route, one handle a reader, reads through a bounce | 16 | LRU | 3,119 | 0.445 / 0.467 (0.456) | 113,915 | 0.074 s a pass in the steady state |
| `lookahead`: + the Timetable on one-row passes, top-10 / top-6 | 16 | LRU | | (0.556) | | 197 predictions a pass, 17 used |
| `lookahead32`: top-3 / top-2 with the cap | 16 | LRU | | (0.504) | | 71% of the predictions used |
| `lookahead32gaps`: top-3 / top-2, the gaps rule | 16 | LRU | | 0.488 / 0.486 (0.487) | | |
| `buspass`: padded slots, reads in place | 4 | Bus Pass | 3,356 | 0.429 / 0.462 (0.445) | 104,533 | 114.8 s |
| `route2`: the probe and rule of section 6 | 16 | LRU | 3,296 | 0.452 / 0.486 (0.469) | 111,221 | 64.9 s |
| `route4`: the depth forced to 4 | 4 | LRU | 3,331 | 0.466 / 0.484 (0.475) | 109,175 | 104.9 s |
| `buspass2` | 16 | Bus Pass | 3,323 | 0.436 / 0.480 (0.458) | 107,143 | 92.1 s |
| `vram2`: + 21 seats on the card ("auto": 0.19 GB) | 16 | Bus Pass | 3,244 | 0.442 / 0.482 (0.462) | 109,808 | 64.0 s |
| `pin`: + the depot pinned | 16 | Bus Pass | 3,340 | 0.444 / 0.475 (0.459) | 114,223 | 115.1 s |
| `bounce1`: reads through the bounce, the victim search in place | 16 | Bus Pass | 3,344 | 0.411 / 0.435 (0.423) | 105,595 | 87.7 s (a plateau in its 64-token row) |
| `padded2`: reads in place, the victim search in place | 16 | Bus Pass | 3,329 | 0.413 / 0.433 (0.423) | 107,124 | 100.6 s (a plateau in its 64-token row) |

The whole-run waits of `buspass`, `route4` and `buspass2` carry the plateau described below; the clean
comparison is the two 256-token segments (passes 137-392 and 393-648), wall time a pass over the decode
passes from the profile's call timestamps:

| run | 137-392: s a pass, waited, misses | 393-648: s a pass, waited, misses |
|---|---|---|
| `route2` (16, LRU) | 0.453, 14.0 s, 29,155 | 0.485, 15.5 s, 32,199 |
| `route4` (4, LRU) | 0.467, 18.5 s, 28,415 | 0.484, 21.7 s, 31,298 |
| `buspass` (4, Bus Pass) | 0.430, 18.0 s, 26,894 | 0.462, 22.9 s, 31,511 |
| `buspass2` (16, Bus Pass) | 0.437, 13.3 s, 28,131 | 0.480, 15.1 s, 32,440 |

**The Route is the one large effect**: 0.549 to 0.456 s a token (-17%), the wait a pass 0.254 to 0.074. The
gain is the kept handles and the queue. Reads in place do not move the token (`route` against `route2` on
the same rows, 0.445 / 0.467 against 0.452 / 0.486, inside the noise), though a lone read shortens from 2.5
to 2.0 ms: the copy they remove ran on the reader's thread, off the token's path.

**The depth.** 16 readers against 4 on the same policy cut the clean segments' wait by a quarter (14.0 /
15.5 s against 18.5 / 21.7) for 0 to 3% on the token; the gap between the wait saved and the token gained is
the cost of sixteen reader threads beside the host's matvec pool. That is why the rule's tie goes to the
deeper queue narrowly, and why a reader that overlaps reads from one thread would keep the wait's gain whole.

**The Bus Pass** against the line at the same depth is 1 to 8% on the token over two pairs (0.437 / 0.480
against 0.453 / 0.485 at 16; 0.430 / 0.462 against 0.467 / 0.484 at 4), with the miss counts inside the
run-to-run capacity noise (3,243 to 3,356 slots): the direction of the replay's -7.5% at a fixed capacity,
not a measurement on its own. Its cost is bookkeeping only, so it is the default and the replay is its test.

**The one-row lookahead loses in every shape** (0.556, 0.504, 0.487 against 0.456): on this drive the layer's
own burst is bandwidth-bound and every prediction read is taken from it. Off on one-row passes; the rows
lookahead stays on for the prefill.

**The seats on the card are inert here.** "auto" finds 0.19 GB to spare beside the 180B's trunk and the
cache's gigabyte, 21 seats against 3,244 slots, and `vram2` against `buspass2` is inside the noise (0.442 /
0.482 against 0.436 / 0.480). A seated expert's rows go to the card on the multi-row path as on the one-row
path (`test_a_seated_expert_serves_a_multi_row_call_from_the_card`). The seats would matter on a card with
room; on this one the trunk takes it.

**The victim search.** A prefill's seat took the whole line of 3,300 keys to find the oldest rider, 130
times a layer, and the reads already issued completed behind that copy: the profile's watchdog
(`ExpertProfile.watch`, a thread that sleeps a millisecond and records how late it wakes) saw 13 to 23 late
wakes a five-second bin inside the prefills, none in decode, the main thread in `victim` every time. This is
the prefill's 2.0 to 2.4 GB/s against a decode burst's 6.3. Both residency lines now look at the oldest rider
in place. The prefills' late wakes fall to 1 to 3 a bin and the prefill's time does not move (its reads and
its 51-row matmuls are the rest of it), but the decode token does, because every decode pass evicts about
120 riders: `padded2` against `buspass2`, the same path and policy, 0.413 / 0.433 against 0.436 / 0.480, 5%
on row 0 and 10% on row 1, the row with the more evictions. The token stands at 0.423 s from the first
record's 0.549, -23%.

**The two read paths are level**: with the victim search in place on both, `bounce1` (through the reader's
buffer) 0.411 / 0.435 and `padded2` (straight into the slot) 0.413 / 0.433. Reads in place are the default
for what they remove, a copy a read on the reader's thread and the scratch buffers; `BTB_STORE_PADDED=0`
keeps the bounce for a drive or a machine that wants it.

**A plateau that is not the code's.** Once a run, at a moving time, the drive's rate falls to a flat 1.28 to
1.30 GB/s, about 260 reads a second at every depth (14 ms a read at 4 in flight, 55 ms at 16), holds for 20
to 35 s and recovers over another 20 s; a clean run takes the same prefill at 4.9 GB/s. Inside the process
every explanation was tested and refuted by measurement: pages of the store trimmed by the machine (the
store pinned in RAM, `store_pin`, shows the plateau in full and gains nothing on the token, so it stays
off); the drive throttling on heat (the old reader held 4.2 to 5.1 GB/s for seven minutes ninety seconds
after a run); the readers serializing behind the GIL (the watchdog wakes on time through the whole plateau,
mean 0.53 ms, max 2 ms); the machine paging other programs out (a run held to 28.7 GB of the 64 by
`--ram-reserve 16` takes the plateau twice); the reads landing in place (the bounce restored takes it too).
What separates the runs is the clock: none of the five runs before 03:00 has it, nine of the twelve since
do, and the machine's own idle maintenance is scheduled at 03:00 and runs in bursts of the length seen. A
row that carries the plateau is not a measurement of the code and is out of every comparison above; the
256-token segments of `route2`, `route4`, `buspass` and `buspass2` are clean and stand.

The rows come from runs under `--profile`; its ledger and expert-store trace are the record, and the replays
of section 3 run over that trace.
