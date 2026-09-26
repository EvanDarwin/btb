# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""What a caller may set, checked before anything loads: the device name against this machine and every
override's type and range, one line saying what was given and what is taken. The CLI, `btb.load` and the
servers' request fields all pass through here, so a bad value fails the same way everywhere. Torch-free."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .kinds import Json

# every BTB_* environment variable the package reads, with what it does; `btb --help` prints this table
ENV_KNOBS: dict[str, str] = {
    "BTB_API_KEY": "the servers' key (the --api-key default)",
    "BTB_CONFIRM": "1: answer yes to prompts, as --confirm (fetch a model in a non-interactive run)",
    "BTB_POOL": "0: no MLX memory pool seeded at load",
    "BTB_CPU_GEMM": "0: a Mac's CPU tier keeps float32 prefill matmuls (else bf16 on MLX's CPU stream)",
    "BTB_HEAD_GEMV": "0: the head multiplies through a float32 copy instead of the native bf16 gemv",
    "BTB_FUSED_NORM": "0: the module's RMSNorm on CUDA instead of the fused one",
    "BTB_FUSED_MLP": "0: the module's SwiGLU on CUDA instead of the in-place one",
    "BTB_FUSED_ROPE": "0: the card graph's rope as separate launches",
    "BTB_CARD_MMA": "0/1: the card's GEMV kernel, the fp32 chain or the tensor cores, instead of the warm-up's pick",
    "BTB_VRAM_EXPERTS_GB": "experts seated on the card: 0, auto, or gigabytes (the vram_experts_gb option)",
    "BTB_LOOKAHEAD": "the expert store's router lookahead per depth on a one-row pass, e.g. 10,6; 0 off (off by default; the lookahead option)",
    "BTB_LOOKAHEAD_ROWS": "the same, per depth, on a multi-row pass (prefill, verify); e.g. 10,6; 0 off (default 10,6)",
    "BTB_BUS_PASS": "the expert store's residency policy: the Bus Pass over the plain line (the bus_pass option)",
    "BTB_STORE_PIN": "the expert store's pages: 0 pageable, 1 pinned, auto (the store_pin option)",
    "BTB_STORE_PADDED": "0: the store bounces reads through a buffer instead of reading into its slots",
    "BTB_ROUTE_DEPTH": "the drive's readers in flight, over the probe's rule",
}


def env_help() -> str:
    """the knobs as a help epilog"""
    width = max(len(k) for k in ENV_KNOBS)
    return "environment:\n" + "\n".join(f"  {k.ljust(width)}  {v}" for k, v in ENV_KNOBS.items())


class OptionError(ValueError):
    """A value btb cannot take. The subclasses carry the parts (the option, the value, the device, the path);
    `str()` is the one line the CLI and the servers print. Catch the class to handle any of them."""


_ABSENT: Any = object()


class BadValue(OptionError):
    """`name` was given `value` and takes `takes`; `value` is left out when it is not worth printing"""

    def __init__(self, name: str, takes: str, value: Any = _ABSENT) -> None:
        self.name, self.takes, self.value = name, takes, value
        super().__init__(self.shown_as(name, "="))

    def shown_as(self, label: str, sep: str = " ") -> str:
        """the line with `label` in place of the option's name (the CLI shows its flag)"""
        return f"{label}: {self.takes}" if self.value is _ABSENT else f"{label}{sep}{self.value!r}: {self.takes}"


class UnknownOption(OptionError):
    """`name` is not an option; `known` are"""

    def __init__(self, name: str, known: Iterable[str]) -> None:
        self.name, self.known = name, sorted(known)
        super().__init__(f"{name} is not an option; the options are {', '.join(self.known)}")


class BadDevice(OptionError):
    """`device` as named cannot run here, or is not a device name; `why` says what would"""

    def __init__(self, device: Any, why: str) -> None:
        self.device, self.why = device, why
        super().__init__(f"device {str(device)!r}: {why}")


class TooManyLayers(OptionError):
    """--cpu-layers and --resident-last together name more layers than the model has"""

    def __init__(self, cpu: int, last: int, layers: int) -> None:
        self.cpu, self.last, self.layers = cpu, last, layers
        super().__init__(f"cpu_layers={cpu} and resident_last={last} name {cpu + last} layers; the model has {layers}")


class UnsupportedModel(OptionError):
    """the GGUF at `path` is an architecture (`arch`) btb does not run; `supported` are the ones it does"""

    def __init__(self, path: str, arch: str, supported: Iterable[str]) -> None:
        self.path, self.arch, self.supported = path, arch, sorted(supported)
        super().__init__(f"{path}: a {arch!r} GGUF; btb runs {', '.join(self.supported)} GGUF models")


class UnsupportedModelType(OptionError):
    """`mt` is an unsupported model_type, `supported` is the list of friendly names support"""

    def __init__(self, mt: str, supported: Iterable[str]) -> None:
        self.model_type, self.supported = mt, list(supported)
        names = ", ".join(self.supported[:-1]) + f", and {self.supported[-1]}" if len(self.supported) > 1 else ""
        super().__init__(f"btb doesn't support this model family ({mt!r}) yet. Currently supported families: {names}.")


class NotPackable(OptionError):
    """the model at `path` is not one `btb pack` writes a 12-bit model for; `why` says so"""

    def __init__(self, path: str, why: str) -> None:
        self.path, self.why = path, why
        super().__init__(f"{path}: {why}")


class NotAModel(OptionError):
    """`path` is a directory but not a model btb can load (no config.json); `why` says what to do instead"""

    def __init__(self, path: str, why: str) -> None:
        self.path, self.why = path, why
        super().__init__(f"{path}: {why}")


class BadPack(OptionError):
    """the 12-bit model at `path` cannot be read by this btb; `why` says so (its version, or not a pack at all)"""

    def __init__(self, path: str, why: str) -> None:
        self.path, self.why = path, why
        super().__init__(f"{path}: {why}")


class Device(StrEnum):
    """Where a model runs: the three names --device takes. A StrEnum, so a member equals its name (torch's
    `device.type`, `torch.device(Device.CPU)`) and the name is spelled in one place."""

    CPU = "cpu"
    MLX = "mlx"
    CUDA = "cuda"

    @property
    def card(self) -> bool:
        return self is Device.CUDA


@dataclass(frozen=True)
class DeviceName:
    """A device as asked for or as resolved: the kind and, for a card, its index; `str()` is what torch.device
    takes ('cuda:1'). `parse` reads any spelling ('CUDA:1 ', a Device, a DeviceName); None or 'auto' is None."""

    kind: Device
    index: int | None = None

    def __str__(self) -> str:
        return f"{self.kind}:{self.index}" if self.index is not None else str(self.kind)

    @classmethod
    def parse(cls, device: Any) -> DeviceName | None:
        if device is None or isinstance(device, DeviceName):
            return device
        if isinstance(device, Device):
            return cls(device)
        d = str(device).strip().lower()
        if d in ("", "auto"):
            return None
        if d in (Device.CPU, Device.MLX, Device.CUDA):
            return cls(Device(d))
        if d.startswith(f"{Device.CUDA}:") and d[5:].isdigit():
            return cls(Device.CUDA, int(d[5:]))
        raise BadDevice(
            device,
            f"{Device.CPU}, {Device.MLX}, {Device.CUDA} or {Device.CUDA}:N (left unset: the card, "
            f"else {Device.MLX} on Apple silicon, else the CPU)",
        )


FLAGS = (
    "fp32",
    "resident_head",
    "kv_host",
    "adapt",
    "mlx_mega",
    "gguf_packed",
    "prefetch",
    "store_pin",
    "bus_pass",
)
COUNTS = {  # whole numbers, with the least each allows
    "context": 0,
    "cpu_layers": 0,
    "resident_last": 0,
    "cold_slots": 1,
    "tree_budget": 0,
    "v_max": 0,
    "draft_vocab": 0,
    "top_k": 0,
    "prefill_card_min": 0,
    "original_max_position_embeddings": 1,
}
UNIT = ("tree_min_prob", "tree_step_mass", "ngram_p", "top_p")  # 0 to 1
POSITIVE = ("draft_temp_ratio", "expert_cache_gb")  # above 0
NONNEG = ("temperature", "vram_experts_gb")  # 0 or above
RESERVE = ("ram_reserve_gb", "vram_reserve_gb")  # GB, or a percent of the machine's total ("10%")
CHOICES = {"draft_bits": (4, 8, 16), "kv_bits": (8,)}
PASS = (
    "seed",
    "lookahead",
    "lookahead_rows",
    "rope_scaling",
    "draft_model",
    "draft_ks",
)  # seed a whole number of any size; the rest the engine's own shapes (lookahead* the per-depth picks,
# draft_model a path/repo for a sibling proposer, draft_ks its tree fan-out per depth)
KNOWN = frozenset((*FLAGS, *COUNTS, *UNIT, *POSITIVE, *NONNEG, *RESERVE, *CHOICES, *PASS))
SAMPLING = ("temperature", "top_p", "top_k", "seed")


def _num(name: str, v: Any) -> float:
    if isinstance(v, bool):
        raise BadValue(name, "a number is expected", v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise BadValue(name, "a number is expected", v) from None
    if not math.isfinite(f):
        raise BadValue(name, "a finite number is expected", v)
    return f


def _whole(name: str, v: Any) -> int:
    if isinstance(v, int) and not isinstance(v, bool):
        return int(v)  # as given: a 64-bit seed must not round through a float
    f = _num(name, v)
    if f != int(f):
        raise BadValue(name, "a whole number is expected", v)
    return int(f)


@dataclass(frozen=True)
class Percent:
    """a reserve given as a share of a total (`--ram-reserve 10%`): `frac` in 0..1, resolved to GB by
    `resolve_reserve` where the machine's total is known"""

    frac: float


def _reserve(name: str, v: Any) -> float | Percent:
    """a reserve as GB (a non-negative number) or a percent of the total (a `Percent`, or a string like
    "10%"); anything else an OptionError naming what was given"""
    if isinstance(v, Percent):
        pct: float = v.frac * 100.0
    elif isinstance(v, str) and v.strip().endswith("%"):
        pct = _num(name, v.strip()[:-1].strip())  # the number before the %, e.g. "10% " -> 10.0
    else:
        f = _num(name, v)  # GB; a non-number is named by _num
        if f < 0.0:
            raise BadValue(name, "0 or above", f)
        return f
    if not 0.0 <= pct <= 100.0:
        raise BadValue(name, "a percent from 0 to 100", v if isinstance(v, str) else pct)
    return Percent(pct / 100.0)


def resolve_reserve(v: float | Percent, total_bytes: int) -> float:
    """the reserve in GB: a `Percent` taken against `total_bytes`, a plain number passed through"""
    if isinstance(v, Percent):
        return v.frac * total_bytes / 2**30
    return float(v)


def check_value(name: str, v: Any) -> Any:
    """`v` as the option `name` takes it (a flag as 0/1, a count as an int, a rate as a float, a reserve as GB
    or a `Percent`), or OptionError"""
    if name in FLAGS:
        n = int(v) if isinstance(v, bool) else _whole(name, v)
        if n not in (0, 1):
            raise BadValue(name, "0 or 1", v)
        return n
    if name in COUNTS:
        n = _whole(name, v)
        if n < COUNTS[name]:
            raise BadValue(name, f"a whole number, at least {COUNTS[name]}", v)
        return n
    if name in CHOICES:
        n = _whole(name, v)
        if n not in CHOICES[name]:
            raise BadValue(name, f"one of {', '.join(str(c) for c in CHOICES[name])}", v)
        return n
    if name in UNIT:
        f = _num(name, v)
        if not 0.0 <= f <= 1.0:
            raise BadValue(name, "between 0 and 1", v)
        return f
    if name in POSITIVE:
        f = _num(name, v)
        if f <= 0.0:
            raise BadValue(name, "above 0", v)
        return f
    if name in NONNEG:
        f = _num(name, v)
        if f < 0.0:
            raise BadValue(name, "0 or above", v)
        return f
    if name in RESERVE:
        return _reserve(name, v)
    if name == "seed":
        return _whole(name, v)
    if name in PASS:
        return v
    raise UnknownOption(name, KNOWN)


def check(kw: Mapping[str, Any]) -> Json:
    """the overrides `btb.load` takes, each checked; a None is an absent one"""
    return {k: check_value(k, v) for k, v in kw.items() if v is not None}


def check_layers(c: Mapping[str, Any], layers: int) -> None:
    """the layer counts against the model's: --cpu-layers and --resident-last each within it, and together"""
    cpu = int(c.get("cpu_layers", 0) or 0)
    last = int(c.get("resident_last", 0) or 0)
    for name, n in (("cpu_layers", cpu), ("resident_last", last)):
        if n > layers:
            raise BadValue(name, f"the model has {layers} layers", n)
    if cpu + last > layers:
        raise TooManyLayers(cpu, last, layers)


def check_sampling(fields: Mapping[str, Any]) -> Json:
    """a request's or a call's temperature / top_p / top_k / seed, checked; absent or null fields left out"""
    return {k: check_value(k, fields[k]) for k in SAMPLING if fields.get(k) is not None}


def check_device(device: Any) -> DeviceName | None:
    """the device as named: None (or 'auto') for the machine's pick; else a DeviceName; anything unreadable, or
    mlx off Apple silicon, an OptionError. The card's presence is checked where torch is
    (engine.device.resolve_device); this much needs no import."""
    d = DeviceName.parse(device)
    if d is not None and d.kind is Device.MLX and sys.platform != "darwin":
        raise BadDevice(
            device,
            f"Apple silicon's; this machine runs {sys.platform}: --device {Device.CUDA} for a card, "
            f"{Device.CPU} without one, or leave it for the machine's pick",
        )
    return d
