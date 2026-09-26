# Serving an answer: one pipeline, rows that stop themselves

The server's routes (OpenAI's chat completions, Ollama's chat and generate) answer a request streamed or whole,
with one answer or `n`, through stop strings and tool calls. This note is the design they share. It follows the
shape of [sessions](./sessions.md): one way to do each thing, owned by one place.

## Why

The routes grew one path per combination: the OpenAI route alone had four (streamed or not, `n` of one or more),
Ollama a fifth, and each cut its stop strings, read its tool calls, counted its usage and cut its logprobs its own
way. When two paths disagreed a request got a different answer streamed than whole (a stop string inside a call
sent the call twice, as text and as a call). And a stop was decided after the fact: the handler thread noticed a
stop string in text the decode had already run past, and stopped it by setting the model's one `abort` - every row
at once, cleared again after - so a row that had reached its stop string among `n` could not leave the batch, and
went on costing a row until the last one was done.

## The model

### One reply

`Reply` (btb/reply.py) turns one request's decoded tokens into its answers, one `Row` each:

* **in:** a token for row r, as the decode draws it;
* **out:** events - a piece of content or of reasoning, the row's tool calls once they are known, the row's finish
  (`stop`: its stop token or a stop string; `length`; `tool_calls`) - and, once the decode is over, each row's
  message whole: its text cut at its stop string, its calls read from that text (gpt-oss's from its channel
  tokens), its tokens and logprobs counted up to the stop.

It is the only place those rules live. A streamed route writes the events as they come; a whole answer is the same
`Reply`'s rows read once it ends. A row knows when it wants no more tokens: its stop token, a stop string, or the
client gone.

### Rows that stop themselves

The decode asks the reply, not the other way round:

* **one answer** - the request's own decode (the per-call cancel a `Stream` uses) stops at its next step once the
  row is done; no model-wide flag is set or cleared;
* **`n` answers** - the fork's decode takes `until(row)`: a row the reply is done with leaves the batch at the
  step that finished it, as its stop token would make it; the others go on, and the decode ends when none is left.
* **a client gone** - every row is done.

The model's `abort` is left to what it is for: stopping whatever runs (a shutdown, Ctrl-C).

### Threads

The reply runs on the decode's thread, so a row is judged at the token that finishes it, not whenever another
thread gets to it. A streamed route writes on a thread of its own from a queue of events: a slow client never holds
up the decode, a failed write marks the connection gone for the decode's next token, and a stretch with nothing to
write (a tool call buffered whole) sends a keepalive on a timer (`Handler.KEEPALIVE`, half a second), which finds a
client gone there too. The decode never waits on the socket, and nothing runs once the request's handler returns:
the decode was its own thread's, and the writer is joined before it ends.

### The session

A one-answer request is a transaction over the server's session (docs/sessions.md): whatever it decoded is
committed when it ends, a stop string's or not - the next request's prompt reuses what it shares with it. Rows of
`n` are a fork of it, which the session holds still and gets back unchanged.

## How it is verified

The existing serve tests pin the routes' fields and formats. The new evidence is **one request, every way**: each
stub script - stop strings in and across tokens, before and inside a tool call, rows ending at their stop token,
at a stop string, at the cap - asked streamed and whole at `n` of one and more, on both routes, the answers equal;
and a row's decode stopping at the step its reply finished it (the steps counted), a client gone ending every row.

## Steps

1. `Reply`: the pipeline alone, tested over a character tokenizer; the routes answer through it.
2. Rows that stop themselves: `until` on the rows' decode, the per-call cancel on one answer's, the model's abort
   out of the request path.
3. The writer thread and its timed keepalive.
