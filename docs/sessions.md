# Sessions: one state, one owner, changes that commit or leave nothing

A `Session` is a sequence's tokens and the cache holding them, driven by hand (`feed`, `mark`/`rewind`, `crop`,
`sync`), decoded on (`generate`), forked into rows (`Branches`) or batched with others (`Batch`). This note is the
design those moves are rebuilt on. It replaces rules each caller had to know with a structure that cannot be broken
by a caller that does not know them.

## Why

Today a session's state is eight fields written from three modules: `ids`, `pending`, `logits`, `anchor`, the
drafter's `dr`/`dr_len`/`pend_h`, `forked`, and the recent `_undo`/`_held`. What keeps them consistent is convention:

* **Invalid combinations are representable.** `pending` and `logits` both set, or neither with a cache (a dense
  session re-runs its last token to get out of that; a hybrid cannot, and raises). `forked` set while a decode feeds.
* **Every operation is its own transaction, hand-written.** A feed rolls back one way (`_whole`), a decode another
  (`_undo` recorded by `_open`, applied by `_abandon`), a fork's write-back a third (crop in `_write`), a batch's
  close a fourth (mark each row done as it goes). Each was added after a failure part way was found.
* **Ownership is a flag.** A fork or batch sets `session.forked`; every mutator must check it, under the decode lock,
  or a feed lands under a fork. A new mutator that forgets the check is a new bug.
* **Rows move by mutation.** `Branches`/`Batch` write `s.ids`, `s.pending`, `s.logits` directly when a row comes back.

Each of these has had a bug, and each fix was local. The structure below makes the classes of bug unrepresentable.

## The model

### One state

A session is in exactly one state:

| state          | holds                                                   | the cache holds        |
|----------------|---------------------------------------------------------|------------------------|
| `Empty`        | nothing                                                 | nothing                |
| `Ready`        | the next token's logits                                 | every token            |
| `Pending`      | one drawn token not fed yet                             | every token but it     |
| `Lent`         | a weak reference to the `Branches`/`Batch` holding it   | what it held when lent |

`tokens` is `ids` (plus the pending token). There is no fourth "cache but no logits" state: every way out of an
operation - success, failure, rewind - lands in one of the four. A mark records the state it was taken in.

**The invariant:** outside a transaction, the cache holds exactly `len(ids)` rows, and a hybrid's recurrent states
are the states after `ids`. Everything below exists to keep it.

### Transactions

Every change goes through one primitive, under the decode lock:

```python
with session._txn(eng) as t:     # the rollback point: len(ids), the state, recurrent states, drafter length
    ...                          # passes append to the cache past the rollback point; nothing else is touched
    t.commit(appended, state)    # the only write of ids and state; anchors and the drafter with them
# left without commit - an exception, a hook raising, memory refused, an abort - and the cache is cut back to the
# rollback point, the recurrent states and the drafter restored, the state as it was
```

* **The watermark.** The rollback point is a length. Attention rows past it are provisional until `commit`; rolling
  back is `crop` to it. This is the same move a speculative decode already makes per pass (append the drafts, keep
  the accepted, crop the rest); the transaction makes it the only move.
* **A call that replaces part of the sequence** (a prompt parting from the session's tokens at `m`) rolls back to
  `m`, not to the call's start: the rows past `m` are overwritten by the new prompt's, and keeping both would cost a
  copy of the tail. The state at `m` is `Pending` on the token at `m - 1` (a dense cache crops to it; a hybrid
  restores the anchor it opened from). This is the one place a failure leaves the session shorter than it was, and
  it is shorter by exactly what the call was replacing.
* **Every public mutator is a transaction:** `feed`, `sync`, `rewind`, `crop`, `generate(session=...)`, and a
  fork's or batch's write-back. `_open`/`_keep`/`_abandon`/`_whole`/`_back_to`/`_undo`/`_held` go away; `_open`
  becomes "the rollback point this prompt keeps", `_keep` becomes `commit`.

### Recurrent states

Attention rows roll back by length; a hybrid's recurrent states do not, so the transaction holds them:

* **Where a pass replaces the state tensors** (the torch path assigns new ones), holding the old tensors is the
  snapshot. It costs nothing.
* **Where a kernel writes in place** (the host `btb_delta_step`, the card graph's copies, MLX's pending graph once
  flushed), the first write in a transaction goes to a second buffer and the transaction swaps on commit. The host
  kernel takes an output pointer beside its input for this. Until each in-place writer is converted, it clones the
  state once per transaction: about 63 MB for a 35B hybrid, ~6 ms on the host and ~0.1 ms on a card, once per
  `feed` or `generate` call, not per token.

### Ownership: leases

`fork(n)` and `batch(sessions)` take a lease on each session. The session moves to `Lent` and every mutator refuses
it by construction (the state check is the transaction's first line, not a flag each caller reads). The lease is
returned exactly one way, whatever returns it:

| returned by           | the session afterwards                                                         |
|-----------------------|--------------------------------------------------------------------------------|
| `keep(r)` (a fork)    | row r's tokens committed in one transaction; `Ready`/`Pending` at the row's end |
| a row leaving a batch | that row committed; the session free                                            |
| `close()` of a batch  | every live row committed, one transaction a row                                 |
| `close()` of a fork   | nothing committed; the state it was lent in                                     |
| dropped unreturned    | nothing committed; the state it was lent in (the lease is held weakly)          |

The `Branches`/`Batch` keeps its own rows (a fork's copy, the MLX alias of the parent's buffer, the card's arena)
and never writes the session's fields; a commit is the only way rows reach a session. Its own `step`/`generate` are
transactions over its rows: a pass failing part way cuts every row back to where the step began. The rows' point
is each layer's own step count (a fork's tail, the card arena's steps - which its pass counts only once a replay is
through - MLX's per-row lengths), a batch's padding mask, and a hybrid's recurrent states. Those are copied once a
step until step 3: rows with recurrent layers run on the torch fork path only (the MLX and card rows passes take
dense families), so the copy is that path's alone.

## What this settles

| concern                                           | settled by                                                      |
|---------------------------------------------------|-----------------------------------------------------------------|
| exceptions during `generate`, `keep`, `join`, `close` | one rollback, the same for every operation                      |
| closing a fork after a row left; dropping one     | the lease table: one way back                                   |
| reusing a session after a batch exits abnormally  | the batch's commits are per row; an uncommitted row is not in the session |
| `reorder()` after `leave()`                       | the rows' own state; refused while a left row's copy is held    |
| a callback raising mid-decode                     | a callback's exception leaves the transaction uncommitted       |
| a new mutator forgetting a guard                  | the guard is the transaction, not the caller                    |

## How it is verified

The existing tests pin behavior and stay. The new evidence is a **model-based random walk**: seeded sequences of
`feed`, `mark`/`rewind`, `crop`, `sync`, `generate`, `fork` → `step`/`leave`/`reorder`/`keep`/`close`/drop, `batch`
→ `join`/`leave`/`close`, with failures injected at every pass boundary (a hook raising, a layer's write failing, a
grant refused). After every operation, whatever it raised:

* the state is one of the four, and the invariant holds (cache length, recurrent states);
* the session's next logits equal a fresh session's fed `session.tokens` (the reference model);
* a lent session refuses every mutator, and is free again once its lease is returned.

It runs on the CPU over every family's fixture, a fixed set of seeds in CI and more on demand.

## Steps

Each lands on its own, the walk and the existing suites green:

1. **State and transactions.** `SessionState`, `_txn`; `feed`, `sync`, `rewind`, `crop`, `generate(session=...)`
   through it; the rollback helpers deleted. Recurrent states cloned once per transaction. The random walk over
   sessions alone.
2. **Leases.** `Branches`/`Batch` take and return leases; write-back through `commit`; their own steps as
   transactions. The walk grows forks and batches.
3. **In-place recurrent writers.** The host kernel's output pointer, the card graph's and MLX's writes, each
   measured against the clone it replaces.

## After this

The same shape carries the other three concerns, each its own note:

* **Streaming:** a request is a transaction over its session with per-row lifecycle (running → stopped for a
  reason → finished); stop strings end a row inside the decode, which leaves the batch; the streamed and the
  collected answers are one event stream, read two ways. No model-wide abort per request.
* **Lending:** the ledger has one writer (the decode thread); finalizers only post "returned" events to it; readers
  see snapshots; on devices btb can measure, the ledger reconciles with what the device reports instead of trusting
  its bookkeeping.
* **The API surface:** a high level (`generate`, `chat`, `stream`) over one low-level `Session`, forks and batches
  through sessions; `Generation` a plain dataclass, `feed` one result type, the pending token internal. Breaking;
  last, with a deprecation period.
