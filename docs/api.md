# The programmatic API: fewer shapes, one of each

btb's Python API grew a call at a time, and a few of its answers change shape with the question: `generate`
returns a tuple that also has attributes, `feed` returns a tensor or a pair depending on a keyword, a fork's `step`
means two things depending on whether it is given tokens, and the token a decode drew last and has not fed shows
through as `pending` on sessions, forks and batches. Each is a place where a caller's code works for one call and
breaks on the next. This note is the redesign. It follows [sessions](./sessions.md), [streaming](./streaming.md) and
[lending](./lending.md): one way to do each thing. It breaks the API at once: the old shapes are gone, not
deprecated.

## The layers

1. **The model**: `generate`, `ask`, `stream`, `chat`, `ask_many`; `session` and `batch`; the memory calls (`empty`,
   `zeros`, `full`, `room`, `reserve`, `memory`); the reads (`hidden`, `project`, `encode`, `prompt_ids`,
   `peak_memory`, `last_pass_report`). Unchanged but for what they return.
2. **A session**: the one low-level unit - `feed`, `mark`/`rewind`, `crop`, `sync`, `rows`, `fork`, `generate`.
3. **Rows**: `Branches` (a session forked) and `Batch` (sessions joined) - `step`, `advance`, `generate`, `leave`,
   and `keep`/`reorder` (a fork) or `join` (a batch).

## The changes

### `Generation` is a record, not a tuple

`generate` returns a frozen dataclass: `tokens`, `stats`, `logprobs`, `hidden`, `report`. It no longer unpacks or
indexes as `(tokens, stats)`, and compares by identity (its `hidden` holds tensors, which have no plain equality).

### One step type

`Session.feed`, and a fork's or a batch's `step` and `advance`, return a `Step`: `logits` ([T, V] for a feed, [live,
V] for rows) and `hidden` ({layer: states}, empty unless `taps` was given). One shape whatever the keywords.

### Rows: `step(tokens)` and `advance()`

`step(tokens)` feeds the given token to each live row; it always takes tokens. `advance()` feeds the tokens
`generate` drew last.

### The drawn token is the engine's

A decode leaves its last token drawn and not fed; the next call feeds it. That stays, but the token itself is the
engine's: `Session.pending` and a fork's or batch's `pending` become private. `logits` stays a plain read - the next
token's logits, None while a drawn token waits (`State.PENDING`) - and `next_logits()` beside it always answers,
feeding the drawn token first when there is one. That is a pass, so it is a call, not a property.

### Rooms and lent tensors stay two things

The earlier plan merged `Room` and `empty()` into one loan type. Their lifetimes differ: a lent tensor is the
caller's until they drop it, and a room is held until it is released. The ledger already treats both as loans of
one kind (docs/lending.md), so the API keeps both.

### The declaration hoop stays, with less global state

Every public method of an API class must name a `PassTag`, and the class refuses to build otherwise. That is the
guarantee that no call escapes the pass report, and it stays. The one module-wide table, `OWNERS`, existed for a
lint. `@api` now marks the class it builds with its owner, the lint finds the marked classes in btb's modules, and
the table goes. The thread-local depth
remains: it says whether a call is the caller's own or made inside another, which is per thread by nature.

## How it is verified

Each new shape has the tests the old one had, moved over, and the examples and the README use only the new shapes.
The old names are checked gone: `pending` on a session or rows, a `Generation` unpacking, `step()` without tokens.
