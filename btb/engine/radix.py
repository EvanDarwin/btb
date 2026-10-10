# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The prefix tree over every conversation the engine has decoded: token paths to the KV rows that hold them.

A node is an edge of tokens and their rows (`page * PAGE + offset` each, in a `PagePool`), and the node holds a
reference on every page its rows lie in, so a page lives while any node or any conversation's table reads it. Two
conversations that start alike share the nodes - and so the pages - of what they share: a prompt is matched as far as
the tree holds it (`match`), and a decode's tokens go in when it commits (`insert`), split where they part from what is
there. A hybrid's recurrent state can resume only where it was kept, so a node may carry the snapshot of the state at
its end; `insert` splits at each snapshot's position so one always lands at a node's end. The least recently used
leaves go first when memory is wanted (`evict`).
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .kvpool import Page, PagePool


class Node:
    """an edge of the tree: its tokens (`key`) and their rows, its parent and its children by their first token, the
    position its end lies at (`end`: the tokens from the root through it), the snapshot of the state there (None: not
    kept), and when it was last used"""

    __slots__ = ("end", "key", "kids", "parent", "rows", "snap", "tick")

    def __init__(self, key: Sequence[int], rows: Sequence[int], parent: Node | None, end: int) -> None:
        self.key = [int(t) for t in key]
        self.rows = [int(r) for r in rows]
        self.parent = parent
        self.kids: dict[int, Node] = {}
        self.end = int(end)
        self.snap: Any = None
        self.tick = 0

    @property
    def start(self) -> int:
        return self.end - len(self.key)


@dataclass
class Match:
    """what the tree holds of a prompt: its first `length` tokens, their rows, the nodes it passes through (the last
    perhaps only partly), and the deepest snapshot within it (`snap_at` its position, 0 and None when none)"""

    length: int = 0
    rows: list[int] = field(default_factory=list)
    path: list[Node] = field(default_factory=list)
    snap_at: int = 0
    snap: Any = None


class RadixTree:
    def __init__(self, pool: PagePool) -> None:
        self.pool = pool
        self.root = Node((), (), None, 0)
        # the holds the nodes have on each page (by id): a page held more often is read by a conversation's table too
        self.holds: dict[int, int] = {}
        # what puts in the commits held back for the tree (`PrefixCache.flush`), asked before the tree is read or let
        # go of: it holds every conversation then
        self.before: Callable[[], None] | None = None

    def _ready(self) -> None:
        if self.before is not None:
            self.before()

    # -- reading -------------------------------------------------------------------------------------------------

    def match(self, tokens: Sequence[int]) -> Match:
        """how much of `tokens` the tree holds, and the rows holding it; the nodes passed are used now"""
        self._ready()
        toks = [int(t) for t in tokens]
        m = Match()
        node, i = self.root, 0
        while i < len(toks):
            kid = node.kids.get(toks[i])
            if kid is None:
                break
            k = _common(kid.key, toks, i)
            m.path.append(kid)
            m.rows.extend(kid.rows[:k])
            i += k
            if k < len(kid.key):
                break
            if kid.snap is not None:
                m.snap_at, m.snap = kid.end, kid.snap
            node = kid
        m.length = i
        self._touch(m.path)
        return m

    def nodes(self) -> Iterator[Node]:
        """every node but the root, parents before their children"""
        stack = list(self.root.kids.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.kids.values())

    # -- writing -------------------------------------------------------------------------------------------------

    def insert(self, tokens: Sequence[int], rows: Sequence[int], snaps: Mapping[int, Any] | None = None) -> list[Node]:
        """`tokens` and the rows holding them (a conversation's table) into the tree, and the snapshots of the state
        after `tokens[:n]` for each position n of `snaps`: the path walked. What the tree holds already keeps its
        own rows - the same tokens' rows, made once - and a snapshot already kept at a position stays"""
        toks = [int(t) for t in tokens]
        rws = [int(r) for r in rows]
        if len(toks) != len(rws):
            raise ValueError(f"{len(toks)} tokens with {len(rws)} rows")
        cuts = sorted(int(n) for n in (snaps or {}) if 0 < int(n) <= len(toks))
        path: list[Node] = []
        node, i = self.root, 0
        while i < len(toks):
            kid = node.kids.get(toks[i])
            if kid is None:
                # the rest is new: one node a stretch between snapshot positions
                stop = next((c for c in cuts if c > i), len(toks))
                kid = self._add(node, toks[i:stop], rws[i:stop])
            else:
                k = _common(kid.key, toks, i)
                cut = next((c for c in cuts if i < c < i + k), None)
                if cut is not None:
                    k = cut - i
                if k < len(kid.key):
                    kid = self._split(kid, k)
            path.append(kid)
            i = kid.end
            if snaps is not None and i in snaps and kid.snap is None:
                kid.snap = snaps[i]
            node = kid
        self._touch(path)
        return path

    def _add(self, parent: Node, key: Sequence[int], rows: Sequence[int]) -> Node:
        n = Node(key, rows, parent, parent.end + len(key))
        for p in self.pool.of(n.rows):
            self._hold(p)
        self.pool.freeze(n.rows)
        parent.kids[n.key[0]] = n
        return n

    def _split(self, n: Node, k: int) -> Node:
        """`n` cut after its first `k` tokens: the new node above holds them, `n` keeps the rest (and its snapshot,
        its children); a page both halves read is held by each. Returns the node above"""
        parent = n.parent
        assert parent is not None and 0 < k < len(n.key)
        top = Node(n.key[:k], n.rows[:k], parent, n.start + k)
        top.tick = n.tick
        parent.kids[top.key[0]] = top
        n.key, n.rows, n.parent = n.key[k:], n.rows[k:], top
        top.kids[n.key[0]] = n
        both = {p.id for p in self.pool.of(top.rows)} & {p.id for p in self.pool.of(n.rows)}
        for pid in both:
            self._hold(self.pool.pages[pid])
        return top

    def _hold(self, p: Page) -> None:
        self.pool.ref(p)
        self.holds[p.id] = self.holds.get(p.id, 0) + 1

    def _remove(self, n: Node) -> int:
        """leaf `n` out of the tree, its pages let go: how many that freed"""
        assert not n.kids and n.parent is not None
        del n.parent.kids[n.key[0]]
        freed = 0
        for p in self.pool.of(n.rows):
            left = self.holds[p.id] - 1
            if left:
                self.holds[p.id] = left
            else:
                del self.holds[p.id]
            freed += self.pool.unref(p)
        n.parent = None
        return freed

    def _read(self, n: Node) -> bool:
        """whether a conversation's table reads every page of node `n`: let go, it frees none of them"""
        return all(p.refs > self.holds.get(p.id, 0) for p in self.pool.of(n.rows))

    def _touch(self, path: Sequence[Node]) -> None:
        self.pool.clock += 1
        for n in path:
            n.tick = self.pool.clock

    # -- letting go ----------------------------------------------------------------------------------------------

    def evict(self, enough: Callable[[int], bool] | None = None) -> int:
        """The least recently used leaves out, one at a time (a parent left childless is a leaf in turn), until
        `enough(pages freed so far)` says so - every leaf when None. Asked for room (`enough`), a leaf whose every page
        a conversation's table reads stays: let go, it frees nothing, and the tree loses what the next conversation
        opening alike would read (a system prompt live sessions share, stripped for no page). Returns the pages freed.
        The leaves are found once and kept in order (a heap), a parent joining them as its last child goes: found
        again for every leaf, a tree of n nodes let go whole took n walks of it"""
        self._ready()

        def out(n: Node) -> bool:
            return enough is None or not self._read(n)

        freed = 0
        leaves = [(n.tick, i, n) for i, n in enumerate(self.nodes()) if not n.kids and out(n)]
        heapq.heapify(leaves)
        seq = len(leaves)
        while leaves and (enough is None or not enough(freed)):
            _, _, n = heapq.heappop(leaves)
            parent = n.parent
            freed += self._remove(n)
            if parent is not None and parent is not self.root and not parent.kids and out(parent):
                heapq.heappush(leaves, (parent.tick, seq, parent))
                seq += 1
        return freed

    def drop_snaps(self, enough: Callable[[int], bool], keep: Callable[[Node], bool] | None = None) -> int:
        """the least recently used snapshots let go (their nodes stay) until `enough(dropped)`: how many went"""
        self._ready()
        held = sorted((n for n in self.nodes() if n.snap is not None and (keep is None or not keep(n))), key=_tick)
        dropped = 0
        for n in held:
            if enough(dropped):
                break
            n.snap = None
            dropped += 1
        return dropped

    def check(self) -> list[str]:
        """the tree's own consistency: what is wrong, or nothing"""
        bad: list[str] = []
        for n in self.nodes():
            if not n.key or len(n.key) != len(n.rows):
                bad.append(f"a node of {len(n.key)} tokens and {len(n.rows)} rows")
            if n.parent is None or n.parent.kids.get(n.key[0]) is not n:
                bad.append(f"a node at {n.end} its parent does not list")
            elif n.end != n.parent.end + len(n.key):
                bad.append(f"a node ends at {n.end}, not {n.parent.end + len(n.key)}")
        return bad


def _common(key: Sequence[int], toks: Sequence[int], at: int) -> int:
    """how many tokens of `key` `toks` repeats from `at`"""
    k, lim = 0, min(len(key), len(toks) - at)
    while k < lim and key[k] == toks[at + k]:
        k += 1
    return k


def _tick(n: Node) -> int:
    return n.tick
