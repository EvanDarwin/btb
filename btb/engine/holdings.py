# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What an engine holds past a pass, and how each holding is let go.

Every long-lived holder registers where it is made - `own(stage, name, release)` - and `close()` is the registry
run: the stages in order, within one the last taken first, each release on its own so one that raises neither
stops the rest nor is forgotten (it stays held, for a second close to try again, and its error is raised once
every other release has run). A holder added tomorrow registers beside the code that makes it, and `close()` does
not change.

The stages are what the holdings depend on: the threads that write into buffers stop before anything is freed;
the records those threads complete are written once they have stopped; the arrays that view other buffers (MLX's,
over the store's blocks) go before the buffers; then the buffers themselves; the files they were read from last.

Process state torch keeps across engines - a cuBLAS workspace for every stream it has multiplied on, the pinned
host allocator's cache - is let go when the last engine on a card closes (`last_on_card`), never by one engine
while another may be multiplying.
"""

from __future__ import annotations

import enum
import threading
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import StreamedTextModel


class Stage(enum.IntEnum):
    STOP = 0  # threads that write into the engine's buffers: joined first
    RECORD = 1  # what those threads complete (the expert profile): written once they have stopped
    VIEWS = 2  # arrays over other holdings' buffers (MLX's over the store's blocks)
    MEMORY = 3  # the buffers: weights, the store, the card graphs and their arena
    FILES = 4  # the maps and handles the weights were read through


class Holdings:
    """an engine's registry of what it holds"""

    def __init__(self) -> None:
        self._held: list[tuple[Stage, str, Callable[[], None]]] = []

    def own(self, stage: Stage, name: str, release: Callable[[], None]) -> None:
        """`release` lets `name` go at `stage` of the engine's close"""
        self._held.append((stage, name, release))

    def release_all(self) -> list[Exception]:
        """Every holding let go, stage by stage and within a stage the last taken first, each on its own: the
        errors of those that raised (each noted with what it was releasing), which stay held for another try."""
        errors: list[Exception] = []
        kept: list[tuple[Stage, str, Callable[[], None]]] = []
        for stage in Stage:
            for h in reversed([h for h in self._held if h[0] == stage]):
                try:
                    h[2]()
                except Exception as e:  # collected: the rest still let go
                    e.add_note(f"[close] releasing {h[1]}")
                    errors.append(e)
                    kept.append(h)
        self._held = kept
        return errors

    def names(self) -> list[str]:
        return [name for _stage, name, _r in self._held]

    def __len__(self) -> int:
        return len(self._held)


_CARD: weakref.WeakSet[StreamedTextModel] = weakref.WeakSet()
_CARD_LOCK = threading.Lock()


def on_card(engine: StreamedTextModel) -> None:
    """an engine on a card: the process's card state stays while it lives"""
    with _CARD_LOCK:
        _CARD.add(engine)


def last_on_card(engine: StreamedTextModel) -> bool:
    """`engine` closed: true when no other engine on a card is left, and the process's card state may go"""
    with _CARD_LOCK:
        _CARD.discard(engine)
        return not _CARD
