# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The programmatic API's declaration hoop: a class the user calls into is `@api(owner)`, and each of its public
methods must be a `PassTag` valued "owner.method" (btb.kinds) - the class refuses to build otherwise. Each call
records its tag in the engine's pass report, so the cert sees every call the way it sees every fork."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

from .kinds import PassTag, api_tags

C = TypeVar("C", bound=type)

# owner -> the classes declared under it, for the lint that every declared tag names a method
OWNERS: dict[str, list[type]] = {}


def _recorded(fn: Callable[..., Any], tag: PassTag) -> Callable[..., Any]:
    @functools.wraps(fn)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        finally:
            # a refused call is still the call
            self._called(tag)

    return call


def api(owner: str) -> Callable[[C], C]:
    """declare `cls` an API class under `owner`: every public method a declared tag, each call recorded through
    the class's own `_called`. Properties are reads, not calls; a public static or class method has no report."""

    def declare(cls: C) -> C:
        declared = {t.value.partition(".")[2]: t for t in api_tags(owner)}
        if not callable(getattr(cls, "_called", None)):
            raise TypeError(f"{cls.__name__}: an API class records its calls through `_called`")
        for name, member in list(vars(cls).items()):
            if name.startswith("_") or isinstance(member, property):
                continue
            if isinstance(member, staticmethod | classmethod):
                raise TypeError(f"{cls.__name__}.{name}: a public static or class method records no call")
            if not inspect.isfunction(member) or getattr(member, "__isabstractmethod__", False):
                continue  # an abstract declaration is its subclasses' call to declare
            tag = declared.get(name)
            if tag is None:
                raise TypeError(
                    f"{cls.__name__}.{name} is public but btb.kinds.PassTag declares no {owner}.{name}: "
                    f"declare it there (the cert then asks for a cell), or make it private"
                )
            setattr(cls, name, _recorded(member, tag))
        OWNERS.setdefault(owner, []).append(cls)
        return cls

    return declare
