# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The shapes the bench writes and reads: a bench process's record and its cells, the matrix's cells,
the machine's specs, the guard, the JSON file of a run
"""

from __future__ import annotations

from typing import TypedDict

from btb.kinds import Json


class BenchCell(TypedDict, total=False):
    """
    One answer length's numbers; `btb bench` and compare.py record the same shape
    """

    new: int
    first_s: float
    greedy_s_tok: float
    spec_s_tok: float | None
    tokens_per_pass: float
    identical: str
    peak_ram_gb: float
    peak_vram_gb: float
    page_faults: int
    hard_faults: int
    read_gb: float


class BenchRecord(TypedDict, total=False):
    """
    The JSON line a bench process appends to its --out; the matrix reads the last one back
    """

    label: str
    path: str
    model: str | None  # the repo id, or the directory's name
    device: str | None
    device_regime: str | None  # a rival's column in btb's names: gpu, cpu, mlx
    tool: str | None  # None: btb
    dtype: str | None
    cells: list[BenchCell]
    report: Json | None


class BenchSpecs(TypedDict, total=False):
    """
    What the numbers were measured on
    """

    host: str
    platform: str  # sys.platform: darwin, win32, linux
    os: str
    machine: str
    python: str
    torch: str
    transformers: str
    cpu: str
    cores: int
    ram_gb: int
    gpu: str
    vram_gb: float
    unified_memory_gb: int
    mlx: str


class BenchMatrixCell(TypedDict, total=False):
    """
    One (model, configuration, dtype) of the matrix: its plan, then what running it came to
    """

    model: str
    repo: str
    path: str
    type: str | None
    packed: bool
    tool: str | None
    config: str
    dtype: str
    variant: str  # a named setup within a configuration (a store's transport, a split): the matrix leaves it empty
    args: list[str]
    skip: str | None
    device_regime: str
    command: str
    log: str
    seconds: float
    memory: dict[str, float]
    error: str
    status: str
    device: str | None
    cells: list[BenchCell]
    report: Json | None
    resumed: str


class BenchMemoryState(TypedDict):
    """
    The machine's memory as the kernel sees it while a cell runs
    """

    level: float
    swap_gb: float
    disk_gb: float


class BenchGuard(TypedDict, total=False):
    """
    The limits a running cell is killed at
    """

    mem_floor: float
    swap_cap: float
    disk_floor: float


class BenchRunDoc(TypedDict):
    """
    The JSON file a run writes
    """

    specs: BenchSpecs
    btb: str
    started: str
    finished: str
    new: list[int]
    prompts: str
    rows: str
    configs: list[str]
    cells: list[BenchMatrixCell]
