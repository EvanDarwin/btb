# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a closed engine still holds.

Two checks of different weight. The engine's own, at every `close()` and nearly free: its registry of holdings
(`holdings.py`) emptied, and nothing left spoken for in its ledger - logged if not, never raised, since a close
must not replace the error a caller is unwinding with one about memory.

The heap census, for tests and, opted into with `BTB_LEAK_CHECK=1`, for a user's report: a baseline of every live
tensor, taken when an engine is built with none alive, and a count once none is alive again. A storage made since
and still alive is kept memory when

- a closed engine, a btb module's globals (a cache, a registry) or a btb thread's frame still reach it - found by
  following every reference (`gc.get_referents`: attributes, containers, closures, bound methods), not by a list
  of buffers, so a buffer added tomorrow is checked like the store's blocks are;
- or only a reference cycle keeps it: freed at the collector's next full pass, which a long-lived process may never
  run, so kept all the same (the store's residency policy once held the store in a cycle, the step graph's closure
  its arena and table). The report names what the cycle is made of.

Anything else still alive is the caller's own (the logits it kept). Card or pinned bytes above the baseline, less
the caller's tensors there and the storages named, are what C++ holds - a graph's pool, a library's workspace -
listed by block. Storages are told apart by the tensors over them, never by address: the caching allocator hands
a freed address straight back out. A storage of `MIN_BYTES` or more is named on its own; smaller ones together,
once they add up to as much.

The test harness (tests/conftest.py) arms the census for every test and fails the test on what it finds; the
opt-in census logs.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import threading
import types
import weakref
from typing import Any

import torch

MIN_BYTES = 16 << 20  # smaller than this is a pass's scratch or a caller's own, never an engine's holding
WALK_MAX = 2_000_000  # objects the ownership walk visits at most; past it the report says so

# an engine is any object to the census - its fields read with `getattr`, its holdings walked as found - so a stand-in
# (tests/unit/test_leaks.py) or an engine whose construction failed part way is counted as a built one is
_LIVE: weakref.WeakSet[object] = weakref.WeakSet()  # engines built and not yet closed
_SINCE: list[weakref.ref[object]] = []  # every engine built since the baseline: the census's roots
_BASE: dict[str, Any] = {}  # the baseline: the tensors alive, and the card and pinned bytes handed out
_FOUND: list[str] = []  # what the harness's census found, for its next report
_LOCK = threading.RLock()  # reentrant: the census collects, and a finalizer it runs may reach here


def _armed() -> bool:
    """the census wanted: the test harness's, or a user's report (`BTB_LEAK_CHECK=1`)"""
    return bool(os.environ.get("BTB_LEAK_HARNESS")) or (os.environ.get("BTB_LEAK_CHECK") or "") == "1"


def _key(t: torch.Tensor) -> tuple[tuple[str, int], int] | None:
    """(device, storage address) and the storage's bytes, or None for a tensor with no storage of its own"""
    try:
        if t.is_meta or t.is_sparse:
            return None
        st = t.untyped_storage()
        return (str(t.device), int(st.data_ptr())), int(st.nbytes())
    except (RuntimeError, NotImplementedError):  # a functional wrapper, a nested tensor
        return None


def _size(nb: int) -> str:
    return f"{nb / 2**30:.2f} GB" if nb >= 1 << 30 else f"{nb / 2**20:.0f} MB"


def _is_tensor(o: Any) -> bool:
    # by the object's own type: `isinstance` asks some objects for `__class__`, which a lazy module attribute
    # answers with a deprecation warning
    return issubclass(type(o), torch.Tensor)


def _tensors() -> list[torch.Tensor]:
    return [o for o in gc.get_objects() if _is_tensor(o)]


def _card_bytes() -> dict[str, int]:
    """the bytes handed out and still held: the card allocator's by card, and torch's pinned host allocator's"""
    out: dict[str, int] = {}
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return out
    for i in range(torch.cuda.device_count()):
        # free-read: the leak check's count of what is still held, never a decision about what fits
        out[f"cuda:{i}"] = int(torch.cuda.memory_allocated(i))
    with contextlib.suppress(RuntimeError, AttributeError):  # a torch without the pinned allocator's stats
        out["pinned"] = int(torch.cuda.host_memory_stats().get("active_bytes.current", 0))
    return out


def _unowned_blocks(dev: str, limit: int = 6) -> list[str]:
    """the card allocator's blocks in use on `dev` that no live tensor's storage starts at: what C++ holds (a
    captured graph's pool, a private `MemPool`, a library's workspace), largest first, with the pool each is in"""
    if not dev.startswith("cuda"):
        return []
    idx = int(dev.split(":")[1]) if ":" in dev else 0
    known = {key[0][1] for key in map(_key, _tensors()) if key is not None}
    out: list[tuple[int, str]] = []
    try:
        segs = torch.cuda.memory_snapshot()
    except RuntimeError:
        return []
    for seg in segs:
        if int(seg.get("device", idx)) != idx:
            continue
        pool = tuple(seg.get("segment_pool_id", (0, 0)))
        where = "the default pool" if pool == (0, 0) else f"private pool {pool} (a graph's or a MemPool)"
        for b in seg.get("blocks", ()):
            if b.get("state") == "active_allocated" and int(b["address"]) not in known:
                out.append((int(b["size"]), where))
    out.sort(reverse=True)
    return [f"    a {_size(n)} block in {w}" for n, w in out[:limit]]


def _baseline() -> None:
    _BASE["tensors"] = {id(t): weakref.ref(t) for t in _tensors()}
    _BASE["card"] = _card_bytes()
    _SINCE.clear()


def track(engine: object) -> None:
    """an engine built. Armed with none alive, what exists now is the census's baseline - a baseline still
    uncounted (a module's engine closed after the last test's check) is counted first"""
    with _LOCK:
        if _armed() and not _LIVE:
            if _BASE:
                _FOUND.extend(_census())
            _baseline()
        _LIVE.add(engine)
        _SINCE.append(weakref.ref(engine))


def closed(engine: object) -> None:
    """an engine closed: its own check - the registry emptied, nothing spoken for in its ledger - logged if it
    fails; with `BTB_LEAK_CHECK=1` and no engine left alive, the census, logged too. Read with `getattr`: an engine
    whose construction failed part way is closed too, and may lack what a built one has"""
    log = getattr(engine, "log", None) or (lambda msg: sys.stderr.write(msg + "\n"))
    left = []
    holdings = getattr(engine, "holdings", None)
    if holdings is not None and len(holdings):
        left.append(f"still held: {', '.join(holdings.names())}")
    dv = getattr(engine, "device", None)
    if dv is not None and hasattr(dv, "spoken_for"):
        for d in {torch.device("cpu"), getattr(engine, "dev", torch.device("cpu"))}:
            spoken = {k: n for k, n in dv.spoken_for(d).items() if n}
            if spoken:
                left.append(f"still spoken for on {d}: " + ", ".join(f"{k} {_size(n)}" for k, n in spoken.items()))
    if left:
        log("[leak] a closed engine: " + "; ".join(left))
    with _LOCK:
        _LIVE.discard(engine)
        if (os.environ.get("BTB_LEAK_CHECK") or "") != "1" or os.environ.get("BTB_LEAK_HARNESS") or _LIVE or not _BASE:
            return
        lines = _census()
    if lines:
        log("[leak] a closed engine kept memory:\n" + "\n".join(lines))


def live() -> bool:
    """engines built and not yet closed: the census waits for them (`verify`)"""
    with _LOCK:
        return bool(_LIVE)


def cycles() -> list[str]:
    """what only a reference cycle keeps now, as report lines - each storage of MIN_BYTES or more, and what the cycle
    is made of; none below. The harness's check at a test's end while an engine it shares with the next test is
    still alive (tests.helpers.shared_model): the census waits for that engine, and the collection after the test
    frees a cycle the test left before any census could see it. One collection that saves its garbage"""
    lines: list[str] = []
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        saved = {id(o): o for o in gc.garbage}  # alive while saved: their ids are theirs
        found: dict[tuple[str, int], tuple[int, Any]] = {}
        for o in gc.garbage:
            if _is_tensor(o):
                key = _key(o)
                if key is not None and key[1] >= MIN_BYTES and key[0] not in found:
                    found[key[0]] = (key[1], o)
        if found:
            back: dict[int, list[Any]] = {}
            for o in gc.garbage:
                for c in gc.get_referents(o):
                    if id(c) in saved:
                        back.setdefault(id(c), []).append(o)
            for (dev, _ptr), (nb, t) in sorted(found.items(), key=lambda kv: -kv[1][0]):
                kinds = _cycle(t, back)
                lines.append(
                    f"  {_size(nb)} {dev} {t.dtype} {list(t.shape)}: held only by a reference cycle"
                    + (f" with {', '.join(kinds[:6])}" if kinds else "")
                )
            del back
        del saved, found
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.collect()
    return lines


def verify() -> list[str]:
    """the harness's check at a test's end: with no engine alive, the census against the baseline (which it then
    drops); and whatever an earlier census found and has not yet reported"""
    with _LOCK:
        if not _LIVE and _BASE:
            _FOUND.extend(_census())
        found, _FOUND[:] = list(_FOUND), []
    return found


def _label(parent: Any, child: Any, attrs: bool = False) -> str:
    """how `parent` refers to `child`: an attribute, a key, an index, a closure's cell. `attrs`: `parent` is an
    object's `__dict__` or a module's globals, its keys named as attributes"""
    if isinstance(parent, dict):
        k = next((k for k, v in parent.items() if v is child), None)
        if k is None:
            return ".<key>"
        return f".{k}" if attrs and isinstance(k, str) else f"[{k!r}]"
    if isinstance(parent, (list, tuple)):
        i = next((i for i, v in enumerate(parent) if v is child), None)
        return f"[{i}]"
    # an object's attributes: its `__dict__` (followed as one hop, or - values held inline, as 3.12's are - the
    # value itself) or its slots
    with contextlib.suppress(Exception):
        d = vars(parent)
        if child is d:
            return ""
        k = next((k for k, v in d.items() if v is child), None)
        if k is not None:
            return f".{k}"
    for cls in type(parent).__mro__:
        for sl in getattr(cls, "__slots__", ()):
            with contextlib.suppress(Exception):
                if getattr(parent, sl) is child:
                    return f".{sl}"
    if isinstance(parent, types.CellType):
        return ".<cell>"
    if isinstance(parent, types.FunctionType):
        return f".{parent.__qualname__}()"
    if isinstance(parent, types.MethodType):
        return ".<bound>"
    return f".<{type(parent).__name__}>"


def _owned(
    roots: list[tuple[str, Any]], want: set[tuple[str, int]], skip: set[int]
) -> tuple[dict[tuple[str, int], str], bool]:
    """which of the storages `want` the roots reach, each with the path it was reached by, and whether the walk
    stopped at `WALK_MAX` short of the end. Modules, classes, code and frames are not followed (the globals and
    frames that matter are roots of their own); `skip` holds objects never entered (other modules' globals)."""
    found: dict[tuple[str, int], str] = {}
    parent: dict[int, tuple[Any, Any]] = {}
    seen = set(skip)
    stack: list[Any] = []
    names: dict[int, str] = {}
    for name, r in roots:
        names[id(r)] = name
        stack.append(r)
    n = 0
    while stack and len(found) < len(want):
        if n >= WALK_MAX:
            return found, True
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        n += 1
        if _is_tensor(o):
            k = _key(o)
            if k is not None and k[0] in want and k[0] not in found:
                chain = [o]
                while id(chain[-1]) in parent:
                    chain.append(parent[id(chain[-1])][0])
                chain.reverse()  # root first
                # from the last root on the way: a btb module's globals reached through another's function are
                # named as that module's
                start = max(i for i, x in enumerate(chain) if id(x) in names)
                chain = chain[start:]
                path = []
                for i in range(len(chain) - 1):
                    p, c = chain[i], chain[i + 1]
                    # an object's attributes and a module's globals read as `.name`
                    attrs = (i == 0 and isinstance(p, dict)) or (i > 0 and getattr(chain[i - 1], "__dict__", None) is p)
                    path.append(_label(p, c, attrs))
                found[k[0]] = names.get(id(chain[0]), "?") + "".join(path)
            base = getattr(o, "_base", None)
            if base is not None and id(base) not in seen:
                parent.setdefault(id(base), (o, base))
                stack.append(base)
            continue
        if issubclass(type(o), (types.ModuleType, type, types.CodeType, types.FrameType)):
            continue
        for c in gc.get_referents(o):
            if id(c) not in seen:
                parent.setdefault(id(c), (o, c))
                stack.append(c)
    return found, False


def _cycle(t: Any, back: dict[int, list[Any]]) -> list[str]:
    """the classes and functions in the unreachable garbage that holds `t`, walked back from it through `back`
    (each garbage object's referrers within the garbage): what the cycle is made of"""
    out: set[str] = set()
    seen = {id(t)}
    stack = [t]
    while stack:
        o = stack.pop()
        for r in back.get(id(o), ()):
            if id(r) in seen:
                continue
            seen.add(id(r))
            stack.append(r)
            if isinstance(r, types.FunctionType):
                out.add(r.__qualname__)
            elif type(r).__module__ not in ("builtins",) and not isinstance(r, (dict, list, tuple, torch.Tensor)):
                out.add(type(r).__qualname__)
    return sorted(out)


def _roots() -> tuple[list[tuple[str, Any]], set[int]]:
    """the census's roots - the engines built since the baseline, btb's module globals, the frames of btb's
    threads - and the other modules' globals, never entered"""
    mods = [(n, m) for n, m in list(sys.modules.items()) if m is not None]
    roots: list[tuple[str, Any]] = [(f"engine {i}", e) for i, r in enumerate(_SINCE) if (e := r()) is not None]
    roots += [(n, vars(m)) for n, m in mods if n.split(".")[0] == "btb"]
    here = threading.get_ident()
    btb_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = {t.ident: t.name for t in threading.enumerate()}
    for tid, frame in sys._current_frames().items():
        if tid == here:
            continue
        f: types.FrameType | None = frame
        while f is not None:
            if os.path.abspath(f.f_code.co_filename).startswith(btb_dir):
                roots.append((f"thread {names.get(tid, tid)} in {f.f_code.co_name}", dict(f.f_locals)))
            f = f.f_back
    skip = {id(vars(m)) for n, m in mods if n.split(".")[0] != "btb"}
    return roots, skip


def _new_storages(base: dict[int, weakref.ref[torch.Tensor]]) -> dict[tuple[str, int], tuple[int, torch.Tensor]]:
    """the storages made since the baseline, each with its bytes and a tensor over it. The storages under tensors
    alive at the baseline are not new, whatever became of their addresses since. A function of its own so the scan's
    locals go with it: a loop variable left in the census's frame would hold a tensor a cycle alone keeps, and the
    cycle would never read as one"""
    alive = _tensors()
    old: set[tuple[str, int]] = set()
    for t in alive:
        r = base.get(id(t))
        key = _key(t)
        if r is not None and r() is t and key is not None:
            old.add(key[0])
    new: dict[tuple[str, int], tuple[int, torch.Tensor]] = {}
    for t in alive:
        r = base.get(id(t))
        key = _key(t)
        if (r is None or r() is not t) and key is not None and key[0] not in old and key[0] not in new:
            new[key[0]] = (key[1], t)
    return new


def _census() -> list[str]:
    """what the engines since the baseline kept, as report lines (none when nothing was kept); the baseline is
    dropped"""
    base, base_card = _BASE.pop("tensors"), _BASE.pop("card")
    lines: list[str] = []
    was = gc.isenabled()
    gc.disable()
    try:
        new = _new_storages(base)
        card = _card_bytes()
        if sum(v[0] for v in new.values()) < MIN_BYTES and not any(
            n - int(base_card.get(d, 0)) >= MIN_BYTES for d, n in card.items()
        ):
            return []
        roots, skip = _roots()
        skip.add(id(new))
        owned, cut = _owned(roots, set(new), skip)
        del roots
        if cut:
            lines.append(f"  (the walk stopped at {WALK_MAX} objects: what it did not reach reads as the caller's)")
        rest = {k: weakref.ref(v[1]) for k, v in new.items() if k not in owned}
        # by identity too: the collector clears a weak reference to anything unreachable, saved or not
        ids = {k: id(v[1]) for k, v in new.items() if k not in owned}
        sizes = {k: (v[0], v[1].dtype, list(v[1].shape)) for k, v in new.items()}
        pinned = {k: v[1].is_pinned() for k, v in new.items() if k[0] == "cpu"}
        del new
    finally:
        if was:
            gc.enable()
    # what only a cycle keeps: collected once with every unreachable object saved, to name what the cycle holds
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        saved = {id(o): o for o in gc.garbage}  # alive while saved: their ids are theirs
        cyc = [k for k, i in ids.items() if i in saved]
        kinds: dict[tuple[str, int], list[str]] = {}
        if cyc:
            back: dict[int, list[Any]] = {}
            for o in gc.garbage:
                for c in gc.get_referents(o):
                    if id(c) in saved:
                        back.setdefault(id(c), []).append(o)
            told = sorted(cyc, key=lambda k: -sizes[k][0])
            for k in [k for k in told if sizes[k][0] >= MIN_BYTES] + [k for k in told if sizes[k][0] < MIN_BYTES][:3]:
                kinds[k] = _cycle(saved[ids[k]], back)
            del back
        del saved
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.collect()

    def what(k: tuple[str, int]) -> str:
        nb, dt, shape = sizes[k]
        if k in owned:
            return f"{_size(nb)} {k[0]} {dt} {shape}: still reached as {owned[k]}"
        return f"{_size(nb)} {k[0]} {dt} {shape}: held only by a reference cycle" + (
            f" with {', '.join(kinds[k][:6])}" if kinds.get(k) else ""
        )

    kept = sorted([*owned, *cyc], key=lambda k: -sizes[k][0])
    large = [k for k in kept if sizes[k][0] >= MIN_BYTES]
    small = [k for k in kept if sizes[k][0] < MIN_BYTES]
    lines += [f"  {what(k)}" for k in large]
    small_bytes = sum(sizes[k][0] for k in small)
    if small_bytes >= MIN_BYTES:
        lines.append(f"  {_size(small_bytes)} in {len(small)} smaller storages, the largest:")
        lines += [f"    {what(k)}" for k in small[:3]]
    # the card and pinned bytes above the baseline, less the caller's own tensors there (the rest still alive) and
    # what is named above
    theirs: dict[str, int] = {}
    for k, r in rest.items():
        if k in cyc or r() is None:
            continue
        dev = "pinned" if pinned.get(k) else k[0]
        theirs[dev] = theirs.get(dev, 0) + sizes[k][0]
    card = _card_bytes()
    for d, n in card.items():
        over = n - int(base_card.get(d, 0)) - theirs.get(d, 0)
        over -= sum(sizes[k][0] for k in kept if (("pinned" if pinned.get(k) else k[0]) == d))
        if over >= MIN_BYTES:
            lines.append(f"  {_size(over)} on {d} over the baseline, held where Python cannot see it")
            lines += _unowned_blocks(d)
    if not large and small_bytes < MIN_BYTES and not any(" over the baseline" in x for x in lines):
        return []
    return lines
