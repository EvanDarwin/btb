# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The shapes the bench writes and reads: a bench process's record and its cells, the matrix's cells (a model
and its axis coordinate), the machine's specs, the guard, the JSON file of a run
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    # the model's storage description (format, quant, params); a TYPE_CHECKING import so compare.py's load in
    # .venv-compare (where btb is not installed) never reaches for btb - the matrix fills the field at plan time
    from btb.hf import ModelInfo


class BenchDevice(StrEnum):
    """A btb device regime, the placement named on the CLI; a comparison tool's regime is one of these too.
    A StrEnum member equals its string and hashes like it, so it drops in where the names were and reads back
    from JSON as the same value."""

    CPU = "cpu"
    CPU_GPU = "cpu+gpu"
    GPU = "gpu"
    CPU_MLX = "cpu+mlx"
    MLX = "mlx"


class BenchDtype(StrEnum):
    """The arithmetic a cell runs in: bf16 on the card and MLX, fp32 on the CPU tier and as the fp32-over-bf16
    variant a card cell can also take."""

    BF16 = "bf16"
    FP32 = "fp32"


class BenchStatus(StrEnum):
    """A cell's outcome. OK ran and produced numbers; the rest produced none and each carries a `reason`: OOM
    ran out of memory, DNF did not finish inside the time limit, DNR did not run - ruled out at planning (a
    gate), unsupported on this config, or errored. A skipped cell is a planned DNR, so it needs no separate
    state."""

    OK = "ok"
    OOM = "oom"
    DNF = "dnf"
    DNR = "dnr"


# each axis to the kinds a run spanned, the manifest a run document carries (skipped cells included)
Axes = dict[str, list[str]]


class Placement(TypedDict, total=False):
    """Where each layer of the model ended up, from the engine's report. resident/host/cold are disjoint (a
    layer is counted once); `mlx` is a second axis over the same layers - the ones that run on the MLX GPU.
    head and drafter are Tier names (`card`, `host`, `packed`, `none`); compute_dtype is the arithmetic."""

    resident: list[int]
    host: list[int]
    cold: list[int]
    mlx: list[int]
    head: str
    drafter: str
    compute_dtype: str


class Report(TypedDict, total=False):
    """The engine's ledger a run records beside the timings; the bench reads its placement."""

    placement: Placement


class BenchCell(TypedDict, total=False):
    """
    One answer length's numbers; `btb bench` and compare.py record the same shape. `base_s_tok` is the
    no-speculation baseline at the cell's sampling (greedy when the cell is greedy); `spec_s_tok` the
    speculative pass at the same sampling; `identical` is bit-exactness of the two under the seed
    """

    new: int
    first_s: float
    base_s_tok: float
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
    The JSON line a bench process appends to its --out; the matrix reads the last one back for its cells and
    the engine's report. The axes are the matrix's own (it planned them), never read back from here
    """

    label: str
    path: str
    model: str | None  # the repo id, or the directory's name
    cells: list[BenchCell]
    report: Report | None


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
    One point of the matrix: the model, its axis coordinate, then what running it came to. The axes - with
    `model` - are the identity and the resume key; the display label is derived from them, never stored
    """

    # the model
    model: str
    repo: str
    path: str
    type: str | None
    packed: bool
    info: ModelInfo  # how it is stored: format, quant, parameters, context (read once from the files)
    # the axes (the coordinate)
    device: BenchDevice  # the placement, or a tool's regime for a tool cell
    dtype: BenchDtype
    pack12: bool  # the 12-bit store beside the model rather than its bf16 weights
    sampling: str  # greedy | t<T>, the temperature both passes run at
    mega: bool  # the MLX megakernel
    tool: str | None  # None: btb; else the comparison engine
    # the run
    args: list[str]
    command: str
    log: str
    seconds: float
    memory: dict[str, float]
    status: BenchStatus  # ok | oom | dnf | dnr; absent until the cell is planned as a skip or run
    reason: str  # why a non-ok cell produced no numbers (a gate, out of memory, a timeout, a crash)
    cells: list[BenchCell]
    report: Report | None
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


class BenchRunDoc(TypedDict, total=False):
    """
    The JSON file a run writes: the specs, the answer lengths, the axis values it planned over (so the whole
    grid is in the document, skipped cells included), and every cell. total=False so reading one back - an
    earlier, maybe older run's - keeps the shape (its fields autocomplete) with every field taken as maybe-
    absent, the nearest Python has to a deep-partial view of a known type.
    """

    specs: BenchSpecs
    btb: str
    started: str
    finished: str
    new: list[int]
    prompts: str
    rows: str
    axes: Axes
    cells: list[BenchMatrixCell]
