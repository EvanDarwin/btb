# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The programmatic API's declaration hoop: a class the user calls into is `@api(owner)`, and each of its public
methods must be a `PassTag` valued "owner.method" (btb.kinds) - the class refuses to build otherwise. Each call
records its tag in the engine's pass report, so the cert sees every call the way it sees every fork: the caller's
call, once, not the API calls it makes inside. A caller's callback running inside a decode (`hooked`) cannot call
back into the engine, which is between two steps there; the reads in `READS` it may make."""

from __future__ import annotations

import contextlib
import functools
import inspect
import threading
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from .kinds import PassTag, api_tags

C = TypeVar("C", bound=type)
F = TypeVar("F", bound=Callable[..., Any])

# owner -> the classes declared under it, for the lint that every declared tag names a method
OWNERS: dict[str, list[type]] = {}

# the calls a callback inside a decode may make: reads that neither step nor move anything the decode holds
READS = frozenset(
    {
        PassTag.API_MODEL_PEAK_MEMORY,
        PassTag.API_MODEL_PROMPT_IDS,
        PassTag.API_MODEL_MEMORY,
        PassTag.API_ROWS_TOKENS,
        PassTag.API_ROOM_RELEASE,
        PassTag.API_ROOM_EMPTY,
        PassTag.API_ROOM_ZEROS,
        PassTag.API_ROOM_FULL,
    }
)

# this thread's depth in API calls (only the outermost is recorded) and in a caller's callbacks
_depth = threading.local()


def _get(name: str) -> int:
    return int(getattr(_depth, name, 0))


@contextlib.contextmanager
def hooked() -> Iterator[None]:
    """a caller's callback running inside a decode, on this thread"""
    _depth.hook = _get("hook") + 1
    try:
        yield
    finally:
        _depth.hook -= 1


def in_hook(fn: F) -> F:
    """`fn`, a caller's callback, run as `hooked`"""

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        with hooked():
            return fn(*args, **kwargs)

    return call  # type: ignore[return-value]


def carried(fn: F) -> F:
    """`fn` run on another thread (the MLX worker) at this thread's depths: what it calls counts as this call's"""
    api, hook = _get("api"), _get("hook")

    @functools.wraps(fn)
    def call(*args: Any, **kwargs: Any) -> Any:
        was = _get("api"), _get("hook")
        _depth.api, _depth.hook = was[0] + api, was[1] + hook
        try:
            return fn(*args, **kwargs)
        finally:
            _depth.api, _depth.hook = was

    return call  # type: ignore[return-value]


def _recorded(fn: Callable[..., Any], tag: PassTag) -> Callable[..., Any]:
    @functools.wraps(fn)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _get("hook") and tag not in READS:
            raise RuntimeError(
                f"{tag.value} called from a callback inside a decode: the engine is between two steps there, and "
                f"a call into it would change what the decode holds; keep what you need and call once it returns"
            )
        outer = _get("api") == 0
        _depth.api = _get("api") + 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            _depth.api -= 1
            if outer:
                self._called(tag)  # a refused call is still the call

    call.__btb_tag__ = tag  # type: ignore[attr-defined]
    return call


def _declare(cls: type, owner: str, declared: dict[str, PassTag], inherited: bool) -> None:
    """`cls`'s public methods wrapped under their tags; a subclass's (`inherited`) overrides only, its own
    methods beside them being its own business"""
    for name, member in list(vars(cls).items()):
        if name.startswith("_") or isinstance(member, property):
            continue
        tag = declared.get(name)
        if inherited and tag is None:
            continue
        if isinstance(member, staticmethod | classmethod):
            raise TypeError(f"{cls.__name__}.{name}: a public static or class method records no call")
        if not inspect.isfunction(member):
            if callable(member):
                raise TypeError(
                    f"{cls.__name__}.{name} is a public callable but not a plain function: no call of it "
                    f"is recorded; define it as a method, or make it private"
                )
            continue
        if getattr(member, "__isabstractmethod__", False) or getattr(member, "__btb_tag__", None) is not None:
            continue  # an abstract declaration is its subclasses' call to declare; a wrapped one is done
        if tag is None:
            raise TypeError(
                f"{cls.__name__}.{name} is public but btb.kinds.PassTag declares no {owner}.{name}: "
                f"declare it there (the cert then asks for a cell), or make it private"
            )
        setattr(cls, name, _recorded(member, tag))


def api(owner: str) -> Callable[[C], C]:
    """declare `cls` an API class under `owner`: every public method a declared tag, each call recorded through
    the class's own `_called`, a subclass's override of one recorded under the same tag. Properties are reads, not
    calls; a public static or class method, or a callable that is not a function, has no report."""

    def declare(cls: C) -> C:
        declared = {t.value.partition(".")[2]: t for t in api_tags(owner)}
        if not callable(getattr(cls, "_called", None)):
            raise TypeError(f"{cls.__name__}: an API class records its calls through `_called`")
        _declare(cls, owner, declared, inherited=False)
        before = cls.__dict__.get("__init_subclass__")
        declared_on: type[Any] = cls

        def init_subclass(sub: type, /, **kw: Any) -> None:
            if before is not None:
                before.__func__(sub, **kw)
            else:
                super(declared_on, sub).__init_subclass__(**kw)
            _declare(sub, owner, declared, inherited=True)

        cls.__init_subclass__ = classmethod(init_subclass)  # type: ignore[assignment]
        OWNERS.setdefault(owner, []).append(cls)
        return cls

    return declare
