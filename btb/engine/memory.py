# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The VRAM policy: what the card and the host hold, and when a layer or the head is shed to the host, regrown
onto the card, reallocated after a bleed, or the card's cache trimmed. The raw machine sensors it reads (host
RAM, the GPU pressure counter) live in `btb.sysinfo`; the free-memory ledger it consults is `Device.free`."""

from __future__ import annotations

import itertools
import math
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch

from .. import trace
from ..api import api
from ..kinds import Log, PassTag
from ..options import Device
from ..sysinfo import (
    BudgetEvent,
    _darwin_available_bytes,
    hard_page_faults,
    host_commit_bytes,
    host_free_bytes,
    host_total_bytes,
    memory_pressure,
    release_pages,
    vram_pressure,
    vram_pressure_line,
    wait_event,
    wddm_budget_event,
    wddm_budget_stale,
    wddm_budget_unregister,
)
from . import device as device_mod
from .cache import GrantedIndexedLayer, GrowLayer
from .device import DeviceSpec, Where, torch_device, where
from .fixed_rows import RowLinear
from .host import _HostLinear
from .scheduler import EPOCH, MemoryGrantError, _size
from .state import _State
from .tiers import ColdRing

if TYPE_CHECKING:
    from .cache import KvCache
    from .device import Device as DeviceLedger
    from .paged import PagedCache


@dataclass
class VramPolicyState:
    """The VRAM policy's own bookkeeping between steps: how much slower than its best a card step must be to
    count as contended, how many free checks precede a regrow, how often pressure is read, the shared-memory
    floor learned on the first read, and the counters and the held reason it carries from one read to the next."""

    contention: float = 4.0
    regrow_after: int = 3
    period: float = 1.0
    contended: int = 0
    free_checks: int = 0
    shared_floor: int | None = None
    realloc_tried: bool = False
    last_t: float = 0.0
    total: int | None = None
    held: str | None = None
    # a yield gave all that helps and the budget was still short by this much: not asked again until the process is
    # inside the budget, or the budget is cut deeper than this (0: none given up)
    spent: int = 0
    # the largest WDDM budget Windows has given this process (read as the engine comes up and each second after, by
    # the budget watcher): a budget below it is another program's doing (`_vram_why`) - the budget at load alone was
    # a game's when btb loaded beside one, and a game's later cut back to it read as btb's own growth
    budget_hi: int = 0
    # when torch's freed blocks were last given back to answer the budget alone (`_vram_trim_enough`)
    trim_t: float = 0.0
    # the budget watcher found the card past the budget while a decode held the engine: the step graph's loop, which
    # runs no pass's policy, hands the answer to the passes that do (`_card_generate_greedy`); a policy's read clears
    asked: bool = False
    # -vv: when the card's figures were last traced (a monotonic clock), and the figures last said (`_trace_card`)
    trace_t: float = 0.0
    trace_said: tuple[int, ...] = ()


@dataclass
class RamPolicyState:
    """Counters for `ram_policy`: consecutive short and clean readings, the last fault count, the layers shed."""

    period: float = 1.0
    regrow_after: int = 5
    last_t: float = 0.0
    faults: int | None = None
    paging: int = 0
    clean: int = 0
    shed: list[int] = field(default_factory=list)
    # a yield gave all that helps and the host was still this short: not asked again until the host is clear, or
    # another program wants RAM_AGAIN more than this (0: none given up)
    spent: int = 0


@dataclass(frozen=True)
class DeviceMemory:
    """One device's memory as btb sees it, in bytes: `free` above btb's margin and what is spoken for, `reserved`
    what rooms, lent tensors btb cannot see and a batch's coming KV hold there, and `sheddable` what btb itself
    gives up to make room. `lendable` is the most `empty` or `room` can get there."""

    device: str
    free: int
    reserved: int
    sheddable: int

    @property
    def lendable(self) -> int:
        return self.free + self.sheddable


@api("room")
class Room:
    """Room made for memory btb does not allocate itself - a second model, a library's workspace - and kept from
    btb until `release()`, the end of a `with` block, or the room being dropped. Rooms add up, each its own.
    `empty`/`zeros`/`full` hand out tensors inside it: their bytes come off what the room holds while they live, so
    the room and the tensor are not counted twice, and a tensor outliving its room is still counted until it goes
    (in btb's free reading, or on MLX, whose reading cannot see torch's, as a loan of its own)."""

    def __init__(
        self, ledger: DeviceLedger, tag: str, nbytes: int, device: torch.device, seen: bool, engine: _State
    ) -> None:
        self.nbytes, self.device = int(nbytes), device
        # a room outliving its model keeps no model alive: the engine and its ledger are held weakly; the ledger
        # holds the room weakly too, so a room dropped is a room let go (docs/lending.md)
        self._ledger, self._loan, self._seen = weakref.ref(ledger), tag, seen
        self._engine = weakref.ref(engine)
        ledger.hold_room(tag, self, self.nbytes, device)

    def _called(self, tag: PassTag) -> None:
        eng = self._engine()
        if eng is not None:
            eng._called(tag)

    @property
    def held(self) -> bool:
        ledger = self._ledger()
        return ledger is not None and ledger.room_held(self._loan)

    @property
    def used(self) -> int:
        """what the room's live tensors take"""
        ledger = self._ledger()
        return 0 if ledger is None else ledger.room_used(self._loan)

    def release(self) -> None:
        ledger = self._ledger()
        if ledger is not None:
            ledger.let_go(self._loan)

    def empty(self, shape: int | Sequence[int], dtype: torch.dtype | None = None) -> torch.Tensor:
        """a tensor of the room's, as `torch.empty` makes it on the room's device; a MemoryGrantError past what is
        left of the room"""
        return self._alloc(shape, dtype, None)

    def zeros(self, shape: int | Sequence[int], dtype: torch.dtype | None = None) -> torch.Tensor:
        """`empty`, filled with zeros"""
        return self._alloc(shape, dtype, 0)

    def full(self, shape: int | Sequence[int], value: float, dtype: torch.dtype | None = None) -> torch.Tensor:
        """`empty`, filled with `value`"""
        return self._alloc(shape, dtype, value)

    def _alloc(self, shape: int | Sequence[int], dtype: torch.dtype | None, fill: float | None) -> torch.Tensor:
        dt = dtype if dtype is not None else torch.get_default_dtype()
        size = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        nbytes = math.prod(size) * torch.empty(0, dtype=dt).element_size()
        ledger = self._ledger()
        if ledger is None:
            raise ValueError("this room was released: make another")
        # made under the ledger's lock: two threads filling the room see each other's tensors
        t = ledger.lend(
            lambda: torch.empty(size, dtype=dt, device=self.device),
            nbytes,
            self.device,
            counted=not self._seen,
            room=self._loan,
            limit=self.nbytes,
        )
        if t is None:
            left = self.nbytes - self.used
            raise MemoryGrantError(
                f"a {dt} tensor of shape {list(size)} ({nbytes / 2**20:.1f} MiB) in a room of "
                f"{self.nbytes / 2**20:.1f} MiB with {left / 2**20:.1f} MiB left"
            )
        if fill is not None:
            t.fill_(fill)
        return t

    def __enter__(self) -> Room:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class _MemoryMixin(_State):
    vram_pressure = staticmethod(vram_pressure)
    vram_pressure_line = staticmethod(vram_pressure_line)
    host_free_bytes = staticmethod(host_free_bytes)
    _darwin_available_bytes = staticmethod(_darwin_available_bytes)
    host_total_bytes = staticmethod(host_total_bytes)
    host_commit_bytes = staticmethod(host_commit_bytes)
    _lent_returns = 0  # the ledger's returns when the lending policy last had nothing left to regrow

    def vram_realloc(self, log: Log | None = None) -> list[str]:
        log = log or self.log
        done = []
        if self.resident_head and self.head is not None and self.dev.type == Device.CUDA:
            self.head = None
            torch.cuda.empty_cache()
            with self._meta:
                self.head = RowLinear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
            self._adopt(self.head, "weight", self._get(self.head_key))
            done.append("head")
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            self.aj = None
            torch.cuda.empty_cache()
            done.append("drafter")
        log(f"[vram] REALLOC {done or 'nothing'} (bled with room on the card); " + vram_pressure_line())
        return done

    def vram_shed(self, cache: Any = None, log: Log | None = None) -> str | None:
        log = log or self.log
        moved = None
        size = 0
        aj = getattr(self, "aj", None)
        if aj is not None and aj.dev.type == Device.CUDA:
            self.aj = None
            self.drafter_dev = torch.device("cpu")
            moved = "drafter"
        elif self.resident:
            size = self._layer_bytes(max(self.resident))
            # the card graphs let go first: they hold the layer's merged blocks (`_card_let_go`), and a layer shed
            # under them freed nothing
            self._card_let_go()
            i = max(self.resident)
            # the host's copy made before the card's leaves: one that raises leaves the layer where it was, never in
            # neither tier
            host = self._make_host_layer(i)
            tmpl = self.resident.pop(i)
            del tmpl
            self.host[i] = host
            self._caches_to(i, "cpu", cache)
            moved = f"layer {i}"
        elif self.resident_head and self.dev.type == Device.CUDA:
            self._head_host()
            self.head = None
            self.resident_head = False
            moved = "head"
        if moved is None:
            return None
        self._shed.append(moved)
        if self.dev.type == Device.CUDA:
            torch.cuda.empty_cache()
        moving = f" ({_size(size)} from the card to RAM)" if size else ""
        log(f"[vram] SHED {moved} -> host{moving} (shed so far: {self._shed}); " + vram_pressure_line())
        return moved

    def vram_regrow(self, cache: Any = None, log: Log | None = None) -> Any:
        log = log or self.log
        if not self._shed:
            return None
        what = self._shed.pop()
        if what == "head":
            with self._meta:
                self.head = RowLinear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
            self._adopt(self.head, "weight", self._get(self.head_key))
            self.resident_head = True
        elif what == "drafter":
            self.drafter_dev = None
            self.aj = None
        else:
            i = int(what.split()[1])
            tmpl = self._new_layer(i)
            self._load_layer(i, tmpl, first=True)
            if self.resident_fp32 and self.compute_dtype is not None and self.compute_dtype != torch.bfloat16:
                for p in tmpl.parameters():
                    p.data = p.data.to(self.compute_dtype)
            self.resident[i] = tmpl
            self.host.pop(i, None)
            self._caches_to(i, self.dev, cache)
        log(f"[vram] REGROW {what} -> {self.dev} (still shed: {self._shed}); " + vram_pressure_line())
        return what

    def vram_trim(self, tag: str = "") -> Any:
        if self.dev.type != Device.CUDA:
            return 0.0, 0.0
        # free-read: the trim's own log line
        before = torch.cuda.memory_reserved() / 2**30
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        # free-read: the trim's own log line
        alloc, res = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
        if before - res > 0.05 or getattr(self, "vram_watch", False):
            self.log(
                f"[vram] trim{(' ' + tag) if tag else ''}: reserved {before:.2f} -> {res:.2f} GB (allocated {alloc:.2f})"
            )
        return alloc, res

    def _trace_card(self) -> None:
        """-vv, once a second and when it moves: what this process holds on the card, split by who can account for
        it - the weights placed there, the rest torch allocated (caches, scratch, activations), torch's cache of
        freed blocks, and what Windows counts past torch (the CUDA context, the card graphs' own pools, cuBLAS) -
        against the budget, with what the ledger speaks for; memory no plan counts shows here as it appears. The
        weights are every one placed on the card: the resident layers and the head, the drafter, the expert store's
        seats, the step graph's embedding table - counted elsewhere they read as the caches'"""
        st = self.vram_state
        now = time.monotonic()
        if now - st.trace_t < 1.0:
            return
        st.trace_t = now
        weights = self._card_weight_bytes()
        # free-read: the trace's figures, never a decision
        alloc, res = int(torch.cuda.memory_allocated(self.dev)), int(torch.cuda.memory_reserved(self.dev))
        info = device_mod._wddm_info(self.dev)
        budget, used = info if info is not None else (0, 0)
        # said again when this process's figures move 64 MB from the ones last said, or the budget 256 MB (Windows
        # moves it by tens of MB a second with another program's own use)
        now_figs = (weights, alloc, res, used, budget)
        last = st.trace_said
        steps = (64 << 20,) * 4 + (256 << 20,)
        if last and all(abs(a - b) < s for a, b, s in zip(now_figs, last, steps, strict=True)):
            return
        st.trace_said = now_figs
        held = self.device.spoken_for(self.dev)
        said = ", ".join(f"{k} {_size(v)}" for k, v in sorted(held.items(), key=lambda kv: -kv[1])) or "nothing"
        trace.event(
            "card: %s used of a %s budget = weights %s + other allocated %s + torch's cache %s + outside torch %s "
            "| the ledger speaks for %s",
            _size(used),
            _size(budget),
            _size(weights),
            _size(alloc - weights),
            _size(res - alloc),
            _size(used - res) if used else _size(0),
            said,
        )

    def _card_weight_bytes(self) -> int:
        """the bytes of every weight placed on the card, each storage once: the resident layers and the head, the
        drafter, the expert store's seats, the step graph's embedding table (a tied head is the same storage)"""
        seen: dict[int, int] = {}

        def add(t: Any) -> None:
            if isinstance(t, torch.Tensor) and t.device.type == Device.CUDA:
                st = t.untyped_storage()
                seen[int(st.data_ptr())] = int(st.nbytes())

        mods: list[Any] = [*self.resident.values()]
        if self.resident_head and self.head is not None:
            mods.append(self.head)
        # the drafter is no module: its layer, its fc and its head's slices are attributes of it
        aj = getattr(self, "aj", None)
        for v in vars(aj).values() if aj is not None else ():
            if isinstance(v, torch.nn.Module):
                mods.append(v)
            else:
                add(v)
        for m in mods:
            for t in (*m.parameters(), *m.buffers()):
                add(t)
        store = getattr(self, "expert_store", None)
        seats = getattr(store, "vram", None)
        add(getattr(seats, "buf", None))
        cg = getattr(self, "_cg", None)
        add(cg.get("table") if cg is not None else None)
        return sum(seen.values())

    def vram_policy(self, cache: Any = None, log: Log | None = None) -> None:
        if trace.ON and self.dev.type == Device.CUDA:
            try:
                self._trace_card()
            except Exception as e:  # the trace watches the pass, never stops it: a figure it cannot read is said
                trace.changed(
                    (trace.token(self), "card unread"), repr(e), "card: the figures could not be read (%r)", e
                )
        if not self.vram_watch or self.dev.type != Device.CUDA or self.warming:
            return  # the warm-up's own passes: answered once it is done (model.py `_warming`)
        self.vram_state.asked = False  # the budget read here, by the pass
        # a batched decode is sized by the scheduler (the one OOM guard), so the per-step streaming policy stands
        # aside: shedding mid-batch only slows it, and the triggers fire on a large batch's legitimate KV growth. A
        # paged cache holds one sequence, its rows read through the map alone
        if cache is not None and not getattr(cache, "paged", False) and getattr(cache, "layers", None):
            k0 = getattr(cache.layers[0], "keys", None)
            if k0 is not None and k0.numel() and int(k0.shape[0]) > 1:
                return
        # another program asking for the card first: Windows shrinks this process's budget the moment one does (a
        # game launched or brought to the front), and the card is given back now - as much as the budget asks, not a
        # layer a second - before the driver pages it out and before the other program's allocations fail. A DXGI
        # read, microseconds, so every pass
        over = self._vram_over()
        if self._vram_yield_due(over):
            # torch's freed blocks before any placement change: a request moves the placement's version, and the
            # card graphs and a card program are rebuilt for the new one - for nothing, where the cache was enough
            if not self._vram_trim_enough(over, log):
                self.device.request("yield", lambda: self.vram_yield(cache, log))
            return
        # the pressure read is a PDH query (4.4 ms on this machine) - longer than a small model's whole step on
        # the card graph. A model holding every layer on the card with a quarter of the card still free has
        # nothing to shed, so the policy stands aside for it; elsewhere it reads pressure once a second, which
        # is how fast pressure moves, not once a token
        now = time.time()
        if now - self.vram_state.last_t < self.vram_state.period:
            return
        self.vram_state.last_t = now
        resident, L = getattr(self, "resident", None), getattr(self, "L", None)
        if (
            resident is not None
            and L is not None
            and len(resident) == L
            and not getattr(self, "host", None)
            and not getattr(self, "cold", None)
        ):
            total = self.vram_state.total
            if total is None:
                try:
                    total = int(torch.cuda.get_device_properties(self.dev).total_memory)
                except Exception:
                    total = 0
                self.vram_state.total = total
            room = self.device.free(unreserved=True)
            if total and room is not None and room > total // 4:
                return
        p = vram_pressure()
        if not p:
            return
        shared = p["process"]["shared"]
        floor = self.vram_state.shared_floor
        if floor is None:
            floor = self.vram_state.shared_floor = shared + (128 << 20)
        bled = shared > floor
        last, best = getattr(self, "_last_card_ms", 0.0), getattr(self, "_card_ms_min", None)
        contended = best is not None and last > 200.0 and last > self.vram_state.contention * best
        if contended:
            self.vram_state.contended += 1
        else:
            self.vram_state.contended = 0
        if not bled:
            self.vram_state.realloc_tried = False
        if bled and not self.vram_state.realloc_tried:
            room = self.device.free(unreserved=True)
            if room is not None and room > self._realloc_bytes():
                self.vram_state.realloc_tried = True
                self.device.request("realloc", lambda: self.vram_realloc(log))
                return
        if bled or self.vram_state.contended >= 2:
            self.vram_state.free_checks = 0
            self.vram_state.contended = 0
            why = (
                f"bled ({(shared - floor) / 2**20:.0f} MB above the floor)"
                if bled
                else f"card contended ({last:.0f} ms vs best {best:.0f})"
            )
            # shedding is a memory decision and the signals above are proxies for one (the PDH counter is noisy
            # under WDDM; a tree's verify pass looks contended next to a graphed one-token step): a resident
            # layer leaves the card only when the card is short of its margin; else the signal is logged once
            room = self.device.free(unreserved=True)
            if room is not None and room > 0:
                kind = why.split(" ")[0]
                if self.vram_state.held != kind:
                    (log or self.log)(
                        f"[vram] {why}, {room / 2**30:.1f} GB free above the margin: holding the card's layers"
                    )
                self.vram_state.held = kind
                return
            self.vram_state.held = None
            if self.device.request("shed", lambda: self.vram_shed(cache, log)) is not None:
                (log or self.log)(f"[vram] reason: {why}")
            self._card_ms_min = None
        elif self._shed:
            room = self.device.free(unreserved=True)
            if room is not None and room > self._regrow_bytes():
                self.vram_state.free_checks += 1
            else:
                self.vram_state.free_checks = 0
            if self.vram_state.free_checks >= self.vram_state.regrow_after:
                self.vram_state.free_checks = 0
                self.device.request("regrow", lambda: self.vram_regrow(cache, log))

    def _vram_yield_due(self, over: int) -> bool:
        """whether a budget `over` bytes short is one a yield has not answered yet: past the budget, and not the
        shortfall the last yield left with nothing more to shed - asked again only once inside the budget, or cut 256
        MB deeper. A yield moves the placement's version whether or not it sheds anything (the card graphs and a card
        program rebuilt), so a shortfall it cannot answer is not asked again on every pass or signal. The policy's and
        the watcher's one test"""
        if over <= 0:
            self.vram_state.spent = 0  # inside the budget again: a later cut is answered afresh
            return False
        spent = self.vram_state.spent
        return not spent or over > spent + (256 << 20)

    def _vram_over(self) -> int:
        """the bytes this process holds on the card past what Windows budgets it, less the room kept for other
        programs (`vram_margin`): what a program asking for the card needs back; 0 within it, or with no budget to
        read (off Windows)"""
        room = device_mod._wddm_room(self.dev)
        if room is None:
            return 0
        return max(0, int(getattr(self, "vram_margin", 0) or 0) - int(room))

    def _vram_why(self) -> str:
        """Who put this process past its budget: Windows cutting the budget (another program took the card - a game
        started or brought to the front) or this process growing past it (its plan did not count what it holds), read
        against the largest budget it has been given (`vram_state.budget_hi`)"""
        info = device_mod._wddm_info(self.dev)
        if info is None:
            return "the budget cannot be read here"
        budget, usage = info
        hi = max(self.vram_state.budget_hi, budget)
        if budget < hi - (256 << 20):
            return f"another program took {_size(hi - budget)} of the card (budget {_size(budget)})"
        return (
            f"this process grew past its own budget ({_size(usage)} used of {_size(budget)}): the placement's "
            "plan did not count all it holds on the card"
        )

    # a budget answered by torch's freed blocks alone is answered so at most this often: past it again sooner, the
    # process's own passes need the room (a prefill's working set refilled the cache each chunk) - a shed, not a trim
    # a pass
    TRIM_AGAIN_S = 10.0

    def _vram_trim_enough(self, over: int, log: Log | None = None) -> bool:
        """torch's freed blocks - this process's own and no pass's: a warm-up's, a prefill's transients - given back
        to a budget `over` bytes short, before any placement change: True where that was enough (no layer moves, no
        card graph is rebuilt, nothing slower after). Not twice within TRIM_AGAIN_S: then the policy yields"""
        if self.dev.type != Device.CUDA:
            return False
        now = time.monotonic()
        if now - self.vram_state.trim_t < self.TRIM_AGAIN_S:
            return False
        self.vram_state.trim_t = now
        self.vram_trim(f"{_size(over)} past the budget: {self._vram_why()}")
        if self._vram_over() > 0:
            return False
        (log or self.log)("[vram] inside the budget again on torch's cached blocks alone: nothing shed")
        return True

    def vram_yield(self, cache: Any = None, log: Log | None = None) -> list[str]:
        """The card given back to the budget Windows asks for: shed (the drafter, the last resident layer, the head -
        `vram_shed`'s order) until this process is inside its budget with the margin kept for other programs, or has
        nothing left to shed - one at least, torch's cache having been tried first (`_vram_trim_enough`): what is
        left past the budget is not a cache, and the card graphs let go come back with the next pass. Grown back by the
        policy once the budget has room again (`vram_regrow`)"""
        log = log or self.log
        over = self._vram_over()
        moved: list[str] = []
        spent = self.vram_state.spent
        if over <= 0 or (spent and over <= spent + (256 << 20)):
            return moved
        log(f"[vram] {_size(over)} past this process's budget, giving the card back: {self._vram_why()}")
        # the card graphs first: they hold every layer's blocks, and a shed under them frees nothing
        self._card_let_go()
        # as long as Windows asks: a program taking the card as it frees (a game loading) cuts this process's budget
        # with every layer given back, and gets the card - a layer a shed frees its ~0.2 GB (Qwen3-4B, measured)
        for k in range(int(getattr(self, "L", 0) or 0) + 3):
            if k and self._vram_over() <= 0:
                break
            m = self.vram_shed(cache, log)
            if m is None:
                break
            moved.append(m)
        self.vram_state.free_checks = 0
        self._card_ms_min = None
        still = self._vram_over()
        # nothing more to give: the policy stops asking until the budget moves (`vram_policy`)
        self.vram_state.spent = max(0, still)
        log(
            f"[vram] gave back {len(moved)} ({', '.join(moved) or 'nothing'})"
            + (f"; still {still / 2**30:.2f} GB past the budget, nothing left to shed" if still > 0 else "")
        )
        return moved

    def watch_vram_budget(self) -> None:
        """A thread waiting on Windows' budget-change event for the card: signalled, and past the budget, it gives the
        card back at once when the engine is idle (the decode lock free) - between requests nothing else would run
        the policy - and otherwise leaves it to the running decode's next pass (`vram_policy`), which reads the budget
        before it runs. A signal the engine was busy for (a decode, the load's warm-up) is kept, and the budget read
        again each second until the engine lets go of the lock - the event resets as it is waited on, and the
        warm-up's last pass reads no budget after it. Each second it notes the largest budget given (`_vram_why`).
        Off Windows, or with `adapt` off, nothing"""
        if not getattr(self, "vram_watch", False) or self.dev.type != Device.CUDA:
            return
        info = device_mod._wddm_info(self.dev)
        if info is not None:
            self.vram_state.budget_hi = max(self.vram_state.budget_hi, info[0])
        ids = device_mod.card_ids(self.dev)
        first = wddm_budget_event(*ids)
        if first is None:
            return
        # the engine by a weak reference and its abort flag alone: the thread never keeps an engine alive that its
        # owner let go of without closing, and ends with it (its event unregistered)
        me, abort = weakref.ref(self), self.abort

        def run(reg: BudgetEvent) -> None:
            pending = False  # a signal the engine was busy for
            said = ""
            try:
                while not abort.is_set():
                    signalled = wait_event(reg.event, 1.0)
                    sm = me()
                    if sm is None:
                        return
                    try:
                        if wddm_budget_stale(reg, *ids):
                            # the card found afresh after a driver reset: the old adapter's event signals nothing, so
                            # the new one's is waited on, and the budget read once now
                            new = wddm_budget_event(*ids)
                            if new is not None:
                                wddm_budget_unregister(reg)
                                reg, signalled = new, True
                        pending = sm._vram_watch_once(signalled, pending, abort)
                    except Exception as e:
                        # the watcher outlives any read or yield that raises - one that ended the thread turned adapt
                        # off for the engine's life, unsaid: said once, and the signal asked again next second
                        if repr(e) != said:
                            said = repr(e)
                            sm.log(f"[vram] the budget watcher: {e!r}; still watching")
                    del sm
            finally:
                wddm_budget_unregister(reg)

        threading.Thread(target=run, args=(first,), name="btb-vram-budget", daemon=True).start()

    def _vram_watch_once(self, signalled: bool, pending: bool, abort: threading.Event) -> bool:
        """one second of the budget watcher (`watch_vram_budget`): the largest budget noted, and a signal - this
        second's, or one the engine was busy for - answered when the engine is idle and no yield answered it yet
        (`_vram_yield_due`). Returns whether a signal still waits on a busy engine"""
        info = device_mod._wddm_info(self.dev)
        if info is not None:
            self.vram_state.budget_hi = max(self.vram_state.budget_hi, info[0])
        if not (signalled or pending):
            return False
        over = self._vram_over()
        if not self._vram_yield_due(over):
            return False
        lock = getattr(self, "_decode_lock", None)
        if lock is None or not lock.acquire(blocking=False):
            # a decode runs, or the load's warm-up: asked again each second till it ends, and the decode told
            self.vram_state.asked = True
            return True
        try:
            if not abort.is_set() and not self._vram_trim_enough(over, None):
                self.device.request("yield", partial(self.vram_yield, None, None))
        finally:
            lock.release()
        return False

    # what a program asking for memory gets on top of the reserve, and how long the store holds off growing back
    RAM_HEADROOM_MIN = 2 << 30
    RAM_HOLD_S = 60.0

    def _ram_headroom(self) -> int:
        """the room a yield leaves above the reserve: enough for another program's launch to grow into (a sixteenth
        of RAM, at least RAM_HEADROOM_MIN)"""
        return max(self.RAM_HEADROOM_MIN, int(host_total_bytes()) // 16)

    def _ram_left(self) -> int:
        """RAM or commit left above the reserve, the tighter (negative: another program is in the reserve)"""
        # free-read: how far another program reaches into the reserve, signed - the ledger's figure stops at zero
        return min(int(host_free_bytes()), int(host_commit_bytes())) - int(getattr(self, "ram_reserve", 0) or 0)

    def _ram_short(self) -> int:
        """The host bytes another program needs back: none while RAM and commit both stay above the reserve and the
        OS says nothing; else the reserve and a launch's headroom above what is left (`_ram_headroom`). Two OS reads,
        microseconds"""
        if not getattr(self, "adapt", False) or self.mlx is not None:
            return 0
        left = self._ram_left()
        low = bool(memory_pressure().get("low"))
        if left >= 0 and not low:
            return 0
        return max(0, self._ram_headroom() - left)

    # a yield answered: asked again only once the host is clear of other programs, or one wants this much more
    RAM_AGAIN = 512 << 20

    def _ram_yield_due(self, short: int) -> bool:
        """whether another program `short` bytes short of memory is one a yield has not answered yet: short, and not
        the shortfall the last yield left with nothing more that helps - asked again only once the host was clear, or
        the other program wants RAM_AGAIN more. A yield moves the placement's version whether or not it gives anything
        back (the card graphs rebuilt), so a shortfall it cannot answer is not asked again every pass and four times a
        second. The policy's and the watcher's one test"""
        if short <= 0:
            self.ram_state.spent = 0  # clear again: a later shortfall is answered afresh
            return False
        spent = self.ram_state.spent
        return not spent or short > spent + self.RAM_AGAIN

    def _ram_needs(self, head: int) -> tuple[int, int]:
        """what the host is short of `head` above the reserve, RAM and commit apart (negative: room to spare). A layer
        read from the checkpoint's mapping holds RAM and no commit, a ring slot or the store's blocks both, so a yield
        answers each with what frees it"""
        reserve = int(getattr(self, "ram_reserve", 0) or 0)
        # the ledger's figure is the tighter of the two, and stops at zero
        # free-read: how far another program reaches into the reserve, RAM and commit apart and signed
        ram, commit = int(host_free_bytes()), int(host_commit_bytes())
        return head - (ram - reserve), head - (commit - reserve)

    def _mapped_spans(self) -> list[tuple[int, int]]:
        """the address ranges of the checkpoint's own mappings (its shards, the 12-bit store's): a tensor inside one is
        the file's pages, not memory of its own"""
        import warnings

        maps = [v[0] for v in getattr(self, "_maps", {}).values()] + list(getattr(self, "_packed_maps", {}).values())
        spans = []
        for mm in maps:
            if mm is None or not len(mm):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = int(torch.frombuffer(mm, dtype=torch.uint8, count=1).data_ptr())
            spans.append((p, p + len(mm)))
        return spans

    def _layer_storages(self, i: int, spans: list[tuple[int, int]]) -> dict[int, tuple[int, bool]]:
        """the storages host layer `i`'s linears read their weights from, by address: (bytes, whether they are the
        checkpoint's mapped pages - inside `spans` - rather than memory of the layer's own)"""
        out: dict[int, tuple[int, bool]] = {}
        for m in self.host[i].modules():
            if not (isinstance(m, _HostLinear) and m.key):
                continue
            for t in (m.weight, *(m.packed or ())):
                if isinstance(t, torch.Tensor):
                    st = t.untyped_storage()
                    p = int(st.data_ptr())
                    out[p] = (int(st.nbytes()), any(a <= p < b for a, b in spans))
        return out

    def _ring_bytes(self) -> int:
        return sum(int(s.numel()) * int(s.element_size()) for s in (getattr(self.cold_ring, "slots", None) or ()))

    def _shed_gain(self, i: int) -> tuple[int, int]:
        """(RAM, commit) a shed of warm layer `i` would give back: the weights it lets go - the mapping's pages RAM
        alone, its own copies both - less what the ring grows by to read it from the drive (a slot more while the ring
        has fewer than `cold_slots`, each slot as large as the largest cold layer)"""
        st = self._layer_storages(i, self._mapped_spans())
        mapped = sum(n for n, m in st.values() if m)
        own = sum(n for n, m in st.values() if not m)
        slots = getattr(self.cold_ring, "slots", None) or []
        stored = self._layer_bytes_stored(i, bool(getattr(self, "_packed", None)))
        slot = max([int(s.numel()) for s in slots] + [stored])
        grow = max(1, min(int(self.cold_slots), len(self.cold) + 1)) * slot - self._ring_bytes()
        return mapped + own - grow, own - grow

    def ram_yield(self, log: Log | None = None) -> int:
        """The host given back to another program at once: the expert store's blocks first (their bytes are on the
        drive, read again on a miss), then host layers to the drive (`_shed_warm`) while a shed frees what is short -
        RAM, or commit, each counted down by what the shed let go (a layer on the checkpoint's mapping frees RAM and
        no commit, and its ring slot takes both), not read again from the OS, whose figures move only as it gets round
        to the pages - until the reserve and a launch's headroom are free, or no shed frees what is short without
        taking the other; the store then holds off growing back for RAM_HOLD_S. What is still short after is not asked
        again (`_ram_yield_due`). Returns the bytes the host gained"""
        log = log or self.log
        st = self.ram_state
        short = self._ram_short()
        if short <= 0:
            st.spent = 0
            return 0
        head = self._ram_headroom()
        # free-read: the yield's own log line (what it gained), never a decision
        before = min(int(host_free_bytes()), int(host_commit_bytes()))
        log(f"[ram] another program wants memory: {short / 2**30:.2f} GB to give back")
        store = getattr(self, "expert_store", None)
        blocks = 0
        if store is not None:
            store.hold_for(head, self.RAM_HOLD_S)
            blocks = int(store.release(want=head) or 0)
        # the store's blocks are memory of its own, back to the OS as they go: the figures read after them stand
        need_ram, need_commit = self._ram_needs(head)
        layers = []
        while need_ram > 0 or need_commit > 0:
            i = self._next_warm()
            if i is None:
                break
            g_ram, g_commit = self._shed_gain(i)
            helps = (need_ram > 0 and g_ram > 0) or (need_commit > 0 and g_commit > 0)
            # one freed at the other's cost is no answer: a mapped layer's ring slot under a commit shortfall
            hurts = (g_ram < 0 and need_ram - g_ram > 0) or (g_commit < 0 and need_commit - g_commit > 0)
            if not helps or hurts:
                break
            r = self._shed_warm("another program wants memory", log)
            if r is None:
                break
            layers.append(r[0])
            need_ram -= r[1]
            need_commit -= r[2]
        # free-read: the yield's own log line (what it gained), never a decision
        gained = min(int(host_free_bytes()), int(host_commit_bytes())) - before
        still = max(need_ram, need_commit)
        log(
            f"[ram] gave back {gained / 2**30:.2f} GB ({blocks} store slots, host layers {layers or 'none'} to the "
            f"drive; the store holds off growing back for {self.RAM_HOLD_S:.0f} s)"
            + (f"; {still / 2**30:.2f} GB short of the headroom, nothing more a shed frees" if still > 0 else "")
        )
        st.spent = max(1, self._ram_short())
        return gained

    def watch_ram(self) -> None:
        """A thread reading RAM, commit and the OS's low-memory word four times a second: another program short of
        memory, it holds the expert store back from what it needs (the running decode's next expert call gives the
        blocks back) and, the engine idle (the decode lock free), gives it all back at once (`ram_yield`) - once a
        shortfall (`_ram_yield_due`). With `adapt` off, or on unified memory, nothing"""
        if not getattr(self, "adapt", False) or self.mlx is not None:
            return
        me, abort = weakref.ref(self), self.abort

        def run() -> None:
            said = ""
            while not abort.wait(0.25):
                sm = me()
                if sm is None:
                    return
                try:
                    sm._ram_watch_once(abort)
                except Exception as e:
                    # the watcher outlives any read or yield that raises - one that ended the thread turned adapt off
                    # for the engine's life, unsaid: said once, and asked again next reading
                    if repr(e) != said:
                        said = repr(e)
                        sm.log(f"[ram] the memory watcher: {e!r}; still watching")
                del sm

        threading.Thread(target=run, name="btb-ram-watch", daemon=True).start()

    def _ram_watch_once(self, abort: threading.Event) -> None:
        """one reading of the RAM watcher (`watch_ram`)"""
        short = self._ram_short()
        if short > 0:
            store = getattr(self, "expert_store", None)
            if store is not None:
                store.hold_for(self._ram_headroom(), self.RAM_HOLD_S)
        if not self._ram_yield_due(short):
            return
        lock = getattr(self, "_decode_lock", None)
        if lock is not None and lock.acquire(blocking=False):  # else a decode runs: its calls give back
            try:
                if not abort.is_set():
                    self.device.request("ram-yield", partial(self.ram_yield, None))
            finally:
                lock.release()

    def ram_policy(self, log: Log | None = None) -> None:
        """Another program short of memory first (`_ram_short`): everything it needs given back at once
        (`ram_yield`), once a shortfall (`_ram_yield_due`). Else once a second: shed one warm layer to the ring after
        two consecutive readings with no free host memory above the reserve. Regrow the last shed layer after
        `regrow_after` consecutive clean readings with room for it."""
        if self.warming:
            return  # the warm-up's own passes: answered once it is done (model.py `_warming`)
        short = self._ram_short()
        if self._ram_yield_due(short):
            self.device.request("ram-yield", lambda: self.ram_yield(log))
        if short > 0:
            return
        if not getattr(self, "ram_watch", False) or self.mlx is not None:
            return
        st = self.ram_state
        now = time.time()
        if now - st.last_t < st.period:
            return
        st.last_t = now
        faults = hard_page_faults()
        delta = faults - st.faults if st.faults is not None else 0
        st.faults = faults
        low = bool(memory_pressure().get("low"))
        room = self.device.free(torch.device("cpu"), unreserved=True)
        if low or (room is not None and room <= 0):
            st.paging += 1
            st.clean = 0
        else:
            st.clean += 1
            st.paging = 0
        warm = [i for i in sorted(self.host) if i not in self.cold]
        if st.paging >= 2 and warm:
            st.paging = 0
            why = f"{0 if room is None else room / 2**30:.2f} GB free above the reserve, hard faults +{delta}" + (
                ", the OS short of memory" if low else ""
            )
            self.device.request("ram-shed", lambda: self.ram_shed(why, log))
        elif st.shed and st.clean >= st.regrow_after:
            room = self.device.free(torch.device("cpu"), unreserved=True)
            need = self._layer_bytes_stored(st.shed[-1], bool(getattr(self, "_packed", None)))
            if room is not None and room > need:
                st.clean = 0
                self.device.request("ram-regrow", lambda: self.ram_regrow(log))

    def _next_warm(self) -> int | None:
        """the warm host layer a shed takes next: the last"""
        warm = [i for i in sorted(getattr(self, "host", None) or ()) if i not in self.cold]
        return warm[-1] if warm else None

    def ram_shed(self, why: str = "", log: Log | None = None) -> int | None:
        """the last warm layer to the ring (`_shed_warm`)"""
        r = self._shed_warm(why, log)
        return None if r is None else r[0]

    def _shed_warm(self, why: str = "", log: Log | None = None) -> tuple[int, int, int] | None:
        """The last warm layer to the ring: its weights read each pass from the drive, the ring rebuilt over the new
        order (the next pass restarts its reader), and its pages in RAM given back to the OS - its own copies freed,
        the checkpoint's mapped pages taken out of the process's resident memory (`release_pages`), where else they sat
        until the OS trimmed them and a shed gave back nothing it could see. Returns (the layer, the RAM and the commit
        it gave back, less the ring's growth)"""
        log = log or self.log
        i = self._next_warm()
        if i is None:
            return None
        spans = self._mapped_spans()
        held = self._layer_storages(i, spans)
        ring = self._ring_bytes()
        self._cold_stop()
        self.cold.add(i)
        self.ram_state.shed.append(i)
        self._bind_cold()
        now = self._layer_storages(i, spans)
        gone = [(p, n, m) for p, (n, m) in held.items() if p not in now]
        if self.mlx is None:
            for p, n, m in gone:
                if m:
                    release_pages(p, n)
        grow = self._ring_bytes() - ring
        own = sum(n for _p, n, m in gone if not m)
        freed_ram, freed_commit = sum(n for _p, n, _m in gone) - grow, own - grow
        log(
            f"[ram] SHED layer {i} -> drive ({_size(max(0, freed_ram))} of RAM and {_size(max(0, freed_commit))} of "
            f"commit given back, the ring's growth taken off; {why or 'asked'}); {len(self.cold)} layers from the "
            "drive each pass"
        )
        return i, freed_ram, freed_commit

    def ram_regrow(self, log: Log | None = None) -> int | None:
        """the last layer shed back into RAM: its linears on the store's mapped bytes again, the ring rebuilt
        without it (emptied when it was the last cold layer)"""
        log = log or self.log
        if not self.ram_state.shed:
            return None
        i = self.ram_state.shed.pop()
        self._cold_stop()
        self.cold.discard(i)
        self._rebind_warm(i)
        if self.cold:
            self._bind_cold()
        else:
            self.cold_ring = ColdRing()
        size = _size(self._layer_bytes_stored(i, bool(getattr(self, "_packed", None))))
        log(
            f"[ram] REGROW layer {i} -> RAM ({size}; still shed: {self.ram_state.shed}); {len(self.cold)} from the "
            "drive"
        )
        return i

    def layer_bytes(self) -> dict[int, int]:
        sizes: dict[int, int] = {}
        by_shard: dict[str, Any] = {}
        for k, sh in self.weight_map.items():
            by_shard.setdefault(sh, []).append(k)
        for sh, keys in by_shard.items():
            _, hdr, _ = self._shard(sh)
            for k in keys:
                if not k.startswith(self.prefix + "layers.") or not self.fam.dense_key(k):
                    continue
                i = int(k[len(self.prefix) + len("layers.") :].split(".")[0])
                info = hdr[k]
                n = 1
                for d in info["shape"]:
                    n *= int(d)
                cast = self._cast_on_read(info)  # held as bf16 (tiers._held)
                b = 2 if cast else torch.empty(0, dtype=self.ST_DTYPES[info["dtype"]]).element_size()
                sizes[i] = sizes.get(i, 0) + n * b
        return sizes

    # -- lending: the caller's calls are `_LendMixin`'s, over these helpers `cache_room` shares ------------------

    def _lend(
        self, shape: int | Sequence[int], dtype: torch.dtype | None, device: DeviceSpec | None, fill: float | None
    ) -> torch.Tensor:
        dev = self._lend_device(device)
        dt = dtype if dtype is not None else torch.get_default_dtype()
        size = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        nbytes = math.prod(size) * torch.empty(0, dtype=dt).element_size()
        what = f"a {dt} tensor of shape {list(size)}"

        def run() -> torch.Tensor:
            self._lend_check(what)
            tried = self._make_room(dev, nbytes, what)
            while True:
                try:
                    t = torch.empty(size, dtype=dt, device=dev)
                    break
                except torch.OutOfMemoryError:
                    # room enough in all but one piece (the allocator's pool is split): one more step, then again
                    if not self._give_up_one(dev, nbytes, tried):
                        raise MemoryGrantError(self._short(dev, nbytes, what)) from None
            if fill is not None:
                t.fill_(fill)
            # a loan until its last view is gone: counted where the ledger cannot see torch's allocations (MLX's),
            # and either way its going is the lending policy's cue to regrow what making room for it shed
            self.device.lend(lambda: t, nbytes, dev, counted=not self._sees(dev))
            return t

        return self._serial(run)

    def _lend_device(self, device: DeviceSpec | None) -> torch.device:
        dev = torch_device(device) if device is not None else (self.dev if self.dev.type == Device.CUDA else None)
        dev = dev if dev is not None else torch.device("cpu")
        if dev.type not in (Device.CUDA, Device.CPU):
            raise ValueError(f"btb lends memory on a card or the host, not {dev}")
        return dev

    def _lend_check(self, what: str) -> None:
        if self.device.held():
            raise RuntimeError(
                f"{what} asked for inside a pass (a hook): btb moves nothing mid-pass; ask before the decode starts"
            )

    def _sees(self, dev: torch.device) -> bool:
        """whether btb's ledger on `dev` counts a torch allocation once it is made (MLX's counts only its own)"""
        return dev.type == Device.CUDA or getattr(self, "mlx", None) is None

    def _short(self, dev: torch.device, nbytes: int, what: str) -> str:
        room = int(self.device.free(dev, unreserved=True) or 0)
        return (
            f"{what} ({nbytes / 2**20:.1f} MiB) on {dev}: {room / 2**20:.1f} MiB free once btb gave up all it can "
            f"(model.memory() shows the room)"
        )

    def _make_room(
        self, dev: torch.device, nbytes: int | Callable[[], int], what: str, own: str | None = None
    ) -> set[str]:
        """room for `nbytes` on `dev` above the margin and what is spoken for (all but the caller's `own`
        reservation), cheapest first; MemoryGrantError when everything btb can give leaves too little. Nothing to
        make where btb holds nothing (a card it does not run on). `nbytes` a callable: the need priced again after
        each step given up - a step can take the need off `dev` (a layer shed takes its rows to the host), and a
        need gone is room made. Returns the one-off steps taken, for a retry to skip"""
        tried: set[str] = set()
        while True:
            need = int(nbytes() if callable(nbytes) else nbytes)
            if need <= 0:
                return tried
            room = self.device.free(dev, unreserved=True, own=own)
            if room is None or room >= need:
                # room there only counting torch's cached blocks: they go back to the driver now, so what is made in
                # it comes from free memory and not from the allocator failing and emptying its cache to retry
                if dev.type == Device.CUDA and need > int(
                    self.device.free(dev, unreserved=True, own=own, pooled=True) or 0
                ):
                    self.vram_trim("room")
                return tried
            if not self._give_up_one(dev, need - room, tried):
                self.device.refused()  # what was shed on the way grows back once there is room
                raise MemoryGrantError(self._short(dev, need, what))

    def cache_room(self, cache: KvCache | None, B: int, T: int) -> None:
        """Room made, before a pass, for the buffers its cache appends will allocate. The scheduler's grant is
        asked for them inside the pass, where nothing may move, so a refusal there could only fail; here btb gives
        up what it holds, cheapest first, as `empty` does. Priced as the grant prices them, and priced again after
        each step given up: another program holding the card past btb's margin, every layer shed there takes its
        rows to the host - the growth is the host's then, and the card has nothing left to make room for (priced once,
        the card's figure stood after the rows had gone, and a pass of a few rows was refused once the whole model
        was off the card). With `adapt` off the placement is pinned: nothing is given up, and the grant refuses what
        does not fit."""
        if cache is None or not getattr(self, "adapt", True):
            return
        if getattr(cache, "paged", False):
            # the pool's growth: the conversations the prefix cache's tree holds go first, their pages taking the
            # rows, before anything else btb holds is given up for them
            pc = cast("PagedCache", cache)
            host = torch.device(Device.CPU)
            pc.prefix.room(pc, B * T, lambda: self.scheduler.free_for(host))
        done: set[Where] = set()
        while True:
            # the devices still to make room on: a shed can add the host to them, its rows' growth now there
            todo = [d for d, n in self.cache_growth(cache, B, T).items() if n and d not in done]
            if not todo:
                return
            dev = todo[0]
            done.add(dev)

            def need(d: Where = dev) -> int:
                return self.cache_growth(cache, B, T).get(d, 0)

            self._make_room(dev, need, f"the cache's growth for {B}x{T} rows", own=EPOCH)

    def cache_growth(self, cache: KvCache | None, B: int, T: int, peak: bool = False) -> dict[Where, int]:
        """the bytes the cache's appends of `T` rows to `B` sequences will allocate, by the device each layer's
        rows live on - priced as the grant prices them; empty where nothing grows. A sparse-attention layer grown by
        concatenation (`GrantedIndexedLayer`) is priced at its grant's doubling, where the append reaches it. With
        `peak`, the most they hold at once: a layer growing in steps keeps its old buffer until the new one is filled,
        so the largest layer's growth once more (a doubling's room holds a concatenation's copy already)"""
        need: dict[Where, int] = {}
        if cache is None:
            return need
        if getattr(cache, "paged", False):
            # every attention layer's rows in the engine's pool, one sequence's: its growth where its free pages, slots
            # and park do not hold them - the host's region and the park in RAM, the card's arenas on the card - the
            # buffer in flight counted in (`PagedCache.growth`)
            for at, g in cast("PagedCache", cache).growth(B * T).items():
                if g:
                    need[where(torch.device(at))] = g
            return need
        first = next(((i, cl) for i, cl in enumerate(cache.layers) if isinstance(cl, GrowLayer)), None)
        if first is None and not any(isinstance(cl, GrantedIndexedLayer) for cl in cache.layers):
            return need
        c = self.cfg
        hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        grows = False
        if first is not None:
            _dev0, dt0 = self._kv_home(*first)
            # the layers grow together: the first one fitting is the pass needing nothing of them
            grows = bool(first[1].growth(B, T, Hk, d, dt0))
        most: dict[Where, int] = {}
        for i, cl in enumerate(cache.layers):
            if isinstance(cl, GrowLayer) and grows:
                dev, dt = self._kv_home(i, cl)
                g = cl.growth(B, T, Hk, d, dt)
                most[dev] = max(most.get(dev, 0), g)
            elif isinstance(cl, GrantedIndexedLayer):
                dev, dt = self._kv_home(i, cl)
                g = cl.growth(B, T, Hk, d, dt, dev)
            else:
                continue
            need[dev] = need.get(dev, 0) + g
        if peak:
            for dev, g in most.items():
                need[dev] += g
        return need

    def _kv_home(self, i: int, cl: GrowLayer | GrantedIndexedLayer) -> tuple[Where, torch.dtype]:
        """where layer i's cache rows live and in what dtype: its buffer's (a concatenated layer's rows'), or where
        the layer runs"""
        if isinstance(cl, GrowLayer) and cl._buf is not None:
            return where(cl._buf[0].device), cl._buf[0].dtype
        k = cl.keys if isinstance(cl, GrantedIndexedLayer) else None
        if isinstance(k, torch.Tensor) and k.numel():
            return where(k.device), k.dtype
        cdt = self.compute_dtype if self.compute_dtype is not None else torch.bfloat16
        host = where(Device.CPU)
        if (isinstance(cl, GrowLayer) and cl.shared) or getattr(self, "mlx", None) is not None:
            return host, cdt
        if i in self.resident:
            return (host if getattr(self, "kv_host", False) else self.dev), cdt
        return host, (self.host_kv_dtype() if isinstance(cl, GrowLayer) else torch.float32)

    def host_kv_dtype(self) -> torch.dtype:
        """the dtype a host layer's cache rows are kept in: a bf16 card engine's bf16, as every card layer's - its
        prompt's rows are made on the card in bf16 anyway, and widened to float32 they took twice the bytes for no
        more precision (a 40k prompt's host layers refused under a game for it); the answer's, made on the host in
        float32, rounded as the card's are. Elsewhere float32, the host's own: an engine off the card, `--fp32`,
        MLX, a family whose host attention does its own float arithmetic (gpt-oss's sinks), and one that runs its
        own layers over caches of its own (Qwen4's sparse layers keep the rows their kernels make)"""
        card_bf16 = (
            self.dev.type == Device.CUDA
            and self.compute_dtype in (None, torch.bfloat16)
            and getattr(self, "mlx", None) is None
            and not self.fam.eager
            and not self.fam.own
        )
        return torch.bfloat16 if card_bf16 else torch.float32

    def _give_up_one(self, dev: torch.device, short: int, tried: set[str]) -> bool:
        """the cheapest thing btb holds on `dev`, given up toward `short` bytes: on a card the drafter, then layers
        from the top, then the head (each live cache following its layer); on the host MLX's cached buffers, the
        expert store's blocks, then a warm layer to the drive. False when nothing is left"""
        log = self.log
        if dev.type == Device.CUDA:
            if self.dev.type != Device.CUDA or self.device.request("lend", lambda: self.vram_shed(None, log)) is None:
                return False
            if not self.vram_watch:  # a running policy regrows what it sheds; none runs to regrow this
                self._lent_shed("card")
            return True
        mlx = getattr(self, "mlx", None)
        if mlx is not None and mlx.held_bytes() > mlx.active_bytes():
            mlx.clear_cache()
            return True
        store = getattr(self, "expert_store", None)
        if store is not None and store.blocks and "store" not in tried:
            # the store gives blocks back until the ledger has what is missing on top of what it has now
            tried.add("store")
            store.release(want=int(self.device.free(dev, unreserved=True) or 0) + int(short))
            return True
        if self.device.request("lend", lambda: self.ram_shed("room asked for", log)) is None:
            return False
        if mlx is not None:
            mlx.clear_cache()
        if mlx is not None or not self.ram_watch:
            self._lent_shed("host")
        return True

    def _lent_shed(self, where: str) -> None:
        """a step making room took that no memory policy regrows: the lending policy's, once a loan comes back"""
        lent = self.__dict__.setdefault("_lent", [])
        if not lent:
            self._lent_returns = self.device.returns  # the returns before it: the next one is its cue
        lent.append(where)

    def lend_policy(self) -> None:
        """Once a second: what making room shed grows back once lent memory has come back and there is room for it
        again - where no memory policy runs to regrow it (a card with `vram_watch` off, the host on MLX)"""
        lent = self.__dict__.get("_lent")
        if not lent:
            return
        returns = self.device.returns
        if returns == self._lent_returns:
            return
        now = time.time()
        if now - self.__dict__.get("_lent_t", 0.0) < 1.0:
            return
        self._lent_t = now
        if lent[-1] == "card":
            room = self.device.free(self.dev, unreserved=True)
            if room is not None and room > self._regrow_bytes():
                lent.pop()
                self.device.request("regrow", lambda: self.vram_regrow(None, self.log))
        elif self.ram_state.shed:
            room = self.device.free(torch.device("cpu"), unreserved=True)
            need = self._layer_bytes_stored(self.ram_state.shed[-1], bool(getattr(self, "_packed", None)))
            if room is not None and room > need:
                lent.pop()
                self.device.request("ram-regrow", lambda: self.ram_regrow(self.log))
        else:
            lent.pop()
        if not lent:
            self._lent_returns = returns

    def _sheddable(self, dev: torch.device) -> int:
        """what `_give_up_one` can free on `dev`, in bytes. `memory()` asks from any thread, as a decode's shed or
        regrow moves layers: it counts over copies of what it reads"""
        if dev.type == Device.CUDA:
            if self.dev.type != Device.CUDA:
                return 0
            fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
            resident = list(self.resident)
            n = sum(self._layer_bytes(i) for i in resident) * (2 if (fp32 and self.resident_fp32) else 1)
            head = self.head
            if self.resident_head and head is not None:
                n += head.weight.numel() * head.weight.element_size()
            aj = getattr(self, "aj", None)
            if aj is not None and aj.dev.type == Device.CUDA:
                n += self._drafter_bytes()
            pc = self.__dict__.get("_kv")
            card = pc.pool.card if pc is not None else None
            if card is not None:
                # the prefix cache's arenas of the resident layers: a layer shed takes its rows to the host's region
                n += sum(card.layer_bytes() for i in resident if i in card.arenas)
            return n + sum(_bytes_on(c, dev, resident) for c in list(self.__dict__.get("_live_caches", ())))
        packed = bool(getattr(self, "_packed", None))
        n = sum(self._layer_bytes_stored(i, packed) for i in list(self.host) if i not in self.cold)
        store = getattr(self, "expert_store", None)
        if store is not None and store.per:
            n += store.live() * int(store.per)
        mlx = getattr(self, "mlx", None)
        if mlx is not None:
            n += max(0, mlx.held_bytes() - mlx.active_bytes())
        return n


_LOANS: Iterator[int] = itertools.count(1)


def _bytes_on(cache: KvCache, dev: torch.device, layers: Iterable[int]) -> int:
    """the bytes `cache` holds on `dev` for `layers`: none of a paged cache's, whose rows are the pool's (counted once,
    for every conversation, in `_sheddable`)"""
    if getattr(cache, "paged", False):
        return 0
    n = 0
    for i in layers:
        if i >= len(cache.layers):
            continue
        cl = cache.layers[i]
        for attr in ("keys", "values"):
            t = getattr(cl, attr, None)
            if isinstance(t, torch.Tensor) and t.device.type == dev.type:
                n += t.numel() * t.element_size()
    return n


@api("model")
class _LendMixin(_MemoryMixin):
    """lending: memory for the caller's own tensors, the model's API over `_MemoryMixin`'s policies"""

    def empty(
        self, shape: int | Sequence[int], dtype: torch.dtype | None = None, device: DeviceSpec | None = None
    ) -> torch.Tensor:
        """
        A tensor of your own beside the model, as `torch.empty` makes it, with room made for it first: btb gives up
        what it holds there, cheapest first, instead of your allocation running out of memory. It is a plain tensor
        and lives as long as you keep it; btb takes the room back once it is gone. `device` defaults to the one the
        model computes on (the host on Apple silicon). A `MemoryGrantError` when even that is not enough.
        """
        return self._lend(shape, dtype, device, None)

    def zeros(
        self, shape: int | Sequence[int], dtype: torch.dtype | None = None, device: DeviceSpec | None = None
    ) -> torch.Tensor:
        """`empty`, filled with zeros"""
        return self._lend(shape, dtype, device, 0)

    def full(
        self,
        shape: int | Sequence[int],
        value: float,
        dtype: torch.dtype | None = None,
        device: DeviceSpec | None = None,
    ) -> torch.Tensor:
        """`empty`, filled with `value`"""
        return self._lend(shape, dtype, device, value)

    def room(self, nbytes: int, device: DeviceSpec | None = None, name: str = "room") -> Room:
        """
        Room for memory btb does not allocate itself - a second model, a library's workspace - made now and kept
        from btb until the `Room` is released (`release()`, or as a `with` block). Keep it while that memory is in
        use; tensors taken through it (`Room.empty`) count against it, not beside it. A `MemoryGrantError` when
        btb cannot make that much.
        """
        dev = self._lend_device(device)
        tag = f"{name}#{next(_LOANS)}"

        def run() -> Room:
            self._lend_check(name)
            self._make_room(dev, int(nbytes), f"room {name!r}")
            return Room(self.device, tag, int(nbytes), dev, self._sees(dev), self)

        return self._serial(run)

    def memory(self) -> dict[str, DeviceMemory]:
        """what btb sees of each device it runs on, by name ("cuda:0", "cpu") - what `empty` and `room` can get"""
        out = {}
        devs = [self.dev] if self.dev.type == Device.CUDA else []
        for dev in [*devs, torch.device("cpu")]:
            # the device read once and the reservations once, what is free derived from the two: the numbers agree
            held = self.device.reserved(dev)
            free = max(0, int(self.device.free(dev) or 0) - held)
            out[str(dev)] = DeviceMemory(str(dev), free, held, self._sheddable(dev))
        return out
