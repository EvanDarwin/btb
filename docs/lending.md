# Lending memory: a ledger read, not tallied

A model lends memory for the caller's own tensors: `empty`/`zeros`/`full` make room before allocating, `room` holds
room for memory btb does not allocate, `memory()` says what can be had. This note is the design of the ledger that
counts what is lent. It follows the shape of [sessions](./sessions.md) and [streaming](./streaming.md): one way to
do each thing, owned by one place.

## Why

The ledger counted lent memory by tallies that finalizers moved. When a room's tensor went, its finalizer took its
bytes off the room's `used` and raised the room's reservation; when a lent tensor or a room went, a finalizer took
its tag off the reservations and set a `returned` flag the lending policy read. Finalizers run on whatever thread
drops the last reference, whenever a collection runs - inside a reading of the ledger on the same thread among
them, which is why its lock was re-entrant and its sums read copies. Each hazard was closed by hand (the copy, a
lock around the room's hold, the release as the last word), and one was not: the lending policy cleared `returned`
as a finalizer on another thread set it, and a return was lost - what making room had shed stayed shed.

## The model

### Loans are facts, not tallies

Every loan is an entry holding a weak reference to what it lends for - a lent tensor's storage, a room, a room's
tensor's storage - and its bytes count while that object lives. Nothing is added or taken away when it goes: the
next reading of the ledger finds the reference dead and drops the entry. There is no finalizer and no callback;
dropping a tensor or a room, on any thread, changes nothing but whether it lives.

| loan | counts |
|---|---|
| a lent tensor, on a device whose free reading sees torch's allocations (a card, the CPU) | nothing: the reading counts it |
| a lent tensor, where the reading cannot see torch (the host under MLX) | its bytes |
| a room, while held | its bytes, less what its live tensors take |
| a room's tensor | as a lent tensor, and against its room while the room is held |

A room is held until it is released or dropped. Its tensors outlive it as lent tensors do.

### One writer

The entries change under the ledger's lock only: a loan entered, a room held or released, a reading dropping what
has gone. Since no code of btb's runs from a collection, nothing re-enters the lock: it is a plain one.

### Readings

`reserved()` is computed from what lives when it is read. `memory()` reads the device and the reservations once each
and derives what is free from the two, so the three numbers it gives agree with each other.

### Returns counted

Every entry dropped counts one return, in a count that only grows. The lending policy regrows what making room
shed once the count has moved since the shed - a number read, never a flag set on one thread and cleared on another.
A refusal counts one too: what was shed on the way to it grows back once there is room.

### The one ordering left

A storage's weak reference dies as the storage is torn down, a moment before the allocator frees its bytes. A
reading on another thread in that moment finds the loan gone; where the device's reading cannot see torch (the host
under MLX), it counts those bytes free a moment early. A room's own tensors net out (the room's hold rises as the
tensor's own count falls); a lent tensor's moment is the allocator's free call.

## How it is verified

The lending tests pin the lifetimes: rooms adding up and coming back, a tensor counted through its views and after
its room, a tensor let go on another thread, grow-back after a tensor or a refusal. The new evidence is concurrency
and collection: threads lending, filling rooms, releasing and dropping them while others read, every reading within
what is lent, and nothing left after all are joined and collected, every loan counted as a return once; loans held
only by a reference cycle counted until the collection that frees them.
