"""The cert axes that are NOT core's to define, as enums so a typo is a NameError and a missing case is
obvious: the device sub-paths, the storage kinds, the decode paths, and which tiny fixture stands in for each
family. The family SET and each family's flags are read from btb core via `core.py` - never listed here (that
mirror was the drift bug). A served `FamilyKind` with no `FIXTURE_STEM` entry is a reported GAP, so a new family
forces a fixture rather than slipping through.

The three cross-enum tables (`STORAGE`, `DECODE_PROPOSER`, `DECODE_KIND`) tie an axis member to the btb.kinds
declaration it stands for - a Quant, a QuantClass, a Proposer - and `totality_problems()` holds them total, so a
new kernel or proposer in core is a missing row here rather than a cell that silently classifies as covered.
Every DeviceSubpath carries the `PassTag` its run must show, so the assertion travels with the sub-path and a
renamed key cannot quietly disable it."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from btb.kinds import PROPOSER_TAG, QUANT_KIND, Cap, FamilyKind, Json, PassTag, Proposer, Quant, QuantClass, quants_of

from . import core

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
GGUF_DIR = os.path.join(FIXTURES, "gguf")


class Hardware(StrEnum):
    CPU = "cpu"
    MLX = "mlx"
    CUDA = "cuda"


class Container(StrEnum):
    """the file shape a cell loads from: the three loaders, each with its own reader and binder."""

    SAFETENSORS = "safetensors"
    PACK12 = "pack12"  # the 12-bit store `btb pack` writes
    GGUF = "gguf"


class Storage(StrEnum):
    """what the weights are on disk. The GGUF members are one per `kinds.QuantClass` - the engine's bind paths -
    except FLOAT, which splits by precision because BF16 is read straight off the file (tiers.py:429) while F16
    goes through gguf.get()/dequant. `STORAGE` holds the split total against `kinds.Quant`."""

    SAFE_BF16 = "safetensors-bf16"
    SAFE_FP16 = "safetensors-fp16"
    SAFE_FP32 = "safetensors-fp32"
    SAFE_FP8 = "safetensors-fp8"
    PACK12 = "pack12"
    GGUF_BF16 = "gguf-bf16"
    GGUF_F16 = "gguf-f16"
    GGUF_AFFINE = "gguf-affine"
    GGUF_KQUANT = "gguf-kquant"
    GGUF_IQ4 = "gguf-iq4"
    GGUF_LATTICE = "gguf-lattice"
    GGUF_MXFP4 = "gguf-mxfp4"


@dataclass(frozen=True)
class StorageInfo:
    """a storage member's loader and, on GGUF, the exact `kinds.Quant` types it stands for - the files a cell
    binds and the kernels a covered verdict claims. `fp` is the checkpoint precision a safetensors cell needs."""

    container: Container
    quants: tuple[Quant, ...] = ()  # GGUF only; every supported Quant appears under exactly one member
    fp: str = ""  # safetensors only: the header dtype (safetensors' spelling) the cell's fixture must carry


STORAGE: dict[Storage, StorageInfo] = {
    Storage.SAFE_BF16: StorageInfo(Container.SAFETENSORS, fp="BF16"),
    Storage.SAFE_FP16: StorageInfo(Container.SAFETENSORS, fp="F16"),
    Storage.SAFE_FP32: StorageInfo(Container.SAFETENSORS, fp="F32"),
    Storage.SAFE_FP8: StorageInfo(Container.SAFETENSORS, fp="F8_E4M3"),
    Storage.PACK12: StorageInfo(Container.PACK12),
    Storage.GGUF_BF16: StorageInfo(Container.GGUF, (Quant.BF16,)),
    Storage.GGUF_F16: StorageInfo(Container.GGUF, (Quant.F16,)),
    Storage.GGUF_AFFINE: StorageInfo(Container.GGUF, tuple(quants_of(QuantClass.AFFINE))),
    Storage.GGUF_KQUANT: StorageInfo(Container.GGUF, tuple(quants_of(QuantClass.KQUANT))),
    Storage.GGUF_IQ4: StorageInfo(Container.GGUF, tuple(quants_of(QuantClass.IQ4))),
    Storage.GGUF_LATTICE: StorageInfo(Container.GGUF, tuple(quants_of(QuantClass.LATTICE))),
    Storage.GGUF_MXFP4: StorageInfo(Container.GGUF, tuple(quants_of(QuantClass.MXFP4))),
}


def quant_class(storage: Storage) -> QuantClass | None:
    """the bind path a GGUF storage member's blocks read through, from kinds.QUANT_KIND; None off GGUF."""
    qs = STORAGE[storage].quants
    return QUANT_KIND[qs[0]] if qs else None


class DecodePath(StrEnum):
    """how the next tokens are produced: the plain one-token loop, one member per `kinds.Proposer`, and the
    sibling draft model, which is not a proposer - it rides behind whichever one the engine runs."""

    GREEDY = "greedy"
    SPEC_NGRAM = "spec-ngram"
    SPEC_MTP = "spec-mtp"
    SPEC_MTP_TREE = "spec-mtp-tree"
    SPEC_MTP_DYN = "spec-mtp-dyn"
    SPEC_DRAFT = "spec-draft-model"


class DecodeKind(StrEnum):
    """what drafts a decode path's tokens, as data rather than a name prefix: nothing, a proposer, or a model."""

    PLAIN = "plain"
    PROPOSER = "proposer"
    DRAFT_MODEL = "draft-model"


DECODE_PROPOSER: dict[DecodePath, Proposer | None] = {
    DecodePath.GREEDY: None,
    DecodePath.SPEC_NGRAM: Proposer.NGRAM,
    DecodePath.SPEC_MTP: Proposer.MTP,
    DecodePath.SPEC_MTP_TREE: Proposer.MTP_TREE,
    DecodePath.SPEC_MTP_DYN: Proposer.MTP_DYN,
    DecodePath.SPEC_DRAFT: None,
}
DECODE_KIND: dict[DecodePath, DecodeKind] = {
    DecodePath.GREEDY: DecodeKind.PLAIN,
    DecodePath.SPEC_NGRAM: DecodeKind.PROPOSER,
    DecodePath.SPEC_MTP: DecodeKind.PROPOSER,
    DecodePath.SPEC_MTP_TREE: DecodeKind.PROPOSER,
    DecodePath.SPEC_MTP_DYN: DecodeKind.PROPOSER,
    DecodePath.SPEC_DRAFT: DecodeKind.DRAFT_MODEL,
}


class Decode(StrEnum):
    """how a token is picked in the cert runner's decode loop, orthogonal to the DecodePath cartesian above:
    greedy argmax vs a seed-keyed temperature/top-k/top-p draw (reproducible, so two loads still match). The
    runner exercises both across the device sub-paths - sampled proves the sampler kernels ran (SAMPLE_STOCHASTIC),
    which greedy never does."""

    GREEDY = "greedy"
    SAMPLED = "sampled"


# FamilyKind -> the tiny fixture stem under tests/fixtures that stands in for it. A served FamilyKind absent
# here has no fixture and is a GAP, so a new family cannot pass without one.
FIXTURE_STEM: dict[FamilyKind, str] = {
    FamilyKind.QWEN3: "tiny_qwen3",
    FamilyKind.QWEN3_5: "tiny_q35",
    FamilyKind.PHI3: "tiny_phi3",
    FamilyKind.QWEN4: "tiny_q4",
    FamilyKind.GPT_OSS: "tiny_gpt_oss",
    FamilyKind.GEMMA3: "tiny_gemma3",
}


def kind_of_stem(stem: str) -> FamilyKind | None:
    """the family a fixture stem stands for (the inverse of FIXTURE_STEM), or None for a stem no family claims."""
    return next((k for k, s in FIXTURE_STEM.items() if s == stem), None)


# --- what the committed fixtures actually are (read from disk, never asserted from a name) -----------------


def fixture_config(kind: FamilyKind) -> Json:
    """a tiny fixture's config.json, its text half where the config nests one; {} when the fixture is absent."""
    stem = FIXTURE_STEM.get(kind)
    if stem is None:
        return {}
    path = os.path.join(FIXTURES, stem, "config.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        cfg: Json = json.load(f)
    sub = cfg.get("text_config")
    return sub if isinstance(sub, dict) else cfg


def head_dim(kind: FamilyKind) -> int | None:
    """the fixture's attention head width, as the engine reads it: `head_dim`, else hidden_size / heads."""
    c = fixture_config(kind)
    if c.get("head_dim"):
        return int(c["head_dim"])
    h, n = c.get("hidden_size"), c.get("num_attention_heads")
    return int(h) // int(n) if h and n else None


def rope_dim(kind: FamilyKind) -> int | None:
    """the rotary's width: the head, or the fraction of it a partial rotary turns (phi3's 0.75)."""
    hd = head_dim(kind)
    frac = fixture_config(kind).get("partial_rotary_factor")
    return hd if hd is None or not frac else int(hd * float(frac))


def has_mtp_head(kind: FamilyKind) -> bool:
    """whether the fixture carries an MTP drafter head, by the engine's own probe (scheduler.py:573): an
    `mtp.*` entry in the checkpoint's weight map. Read from the fixture, never from a stem list."""
    stem = FIXTURE_STEM.get(kind)
    if stem is None:
        return False
    index = os.path.join(FIXTURES, stem, "model.safetensors.index.json")
    if not os.path.isfile(index):
        return False
    with open(index, encoding="utf-8") as f:
        doc: Json = json.load(f)
    return any(str(k).startswith("mtp.") for k in doc.get("weight_map", {}))


def gguf_name(stem: str, q: Quant) -> str:
    """the tiny GGUF twin's file name for one stored type (tests/make_fixtures.py's rule)."""
    return f"{stem}-{q.value.lower()}.gguf"


def quant_of_gguf(fname: str) -> Quant | None:
    """the stored type a tiny GGUF fixture's name declares, or None when its suffix is no supported Quant."""
    stem = fname[:-5] if fname.endswith(".gguf") else fname
    suffix = stem.rsplit("-", 1)[-1].upper()
    return next((q for q in Quant if q.value == suffix), None)


def storage_of_gguf(fname: str) -> Storage | None:
    """the storage cell a tiny GGUF fixture binds, from the type in its name."""
    q = quant_of_gguf(fname)
    return None if q is None else next((s for s, i in STORAGE.items() if q in i.quants), None)


def twin_path(stem: str, storage: Storage) -> str:
    """where a safetensors precision twin of a BF16 fixture lives: `<stem>-<header dtype>` (tiny_qwen3-f16),
    named by its stored type as the GGUF twins are. SAFE_BF16 is the fixture itself."""
    if storage is Storage.SAFE_BF16:
        return os.path.join(FIXTURES, stem)
    return os.path.join(FIXTURES, f"{stem}-{STORAGE[storage].fp.lower()}")


# the float header dtypes the storage axis knows; a twin must carry exactly its own among these
FLOAT_HEADERS: frozenset[str] = frozenset(i.fp for i in STORAGE.values() if i.fp)


def header_dtypes(path: str) -> frozenset[str]:
    """every tensor dtype a checkpoint directory's safetensors headers declare (each shard's leading JSON, so
    no tensor is read); empty when the directory is absent or holds no shard."""
    if not os.path.isdir(path):
        return frozenset()
    out: set[str] = set()
    for name in os.listdir(path):
        if not name.endswith(".safetensors"):
            continue
        with open(os.path.join(path, name), "rb") as f:
            header: Json = json.loads(f.read(int.from_bytes(f.read(8), "little")))
        out |= {str(v["dtype"]) for k, v in header.items() if k != "__metadata__"}
    return frozenset(out)


def fixture_paths(kind: FamilyKind, storage: Storage) -> tuple[str, ...]:
    """every artifact a (family, storage) cell binds: the checkpoint directory, or one GGUF file per stored type
    the storage member stands for. Empty when the family has no fixture stem, or when no precision twin on disk
    carries exactly that storage's float header dtype - a twin binds by what its headers say, never its name."""
    stem = FIXTURE_STEM.get(kind)
    if stem is None:
        return ()
    info = STORAGE[storage]
    if info.container is Container.SAFETENSORS:
        path = twin_path(stem, storage)
        if storage is Storage.SAFE_BF16:
            return (path,)
        if storage is Storage.SAFE_FP8:
            # an FP8 checkpoint is mixed by design: its matrices e4m3, its norms, embeddings and head bf16, its
            # scales f32; it binds by carrying e4m3 at all
            return (path,) if info.fp in header_dtypes(path) else ()
        return (path,) if header_dtypes(path) & FLOAT_HEADERS == {info.fp} else ()
    if info.container is Container.PACK12:
        return (os.path.join(FIXTURES, f"{stem}-pack12"),)
    return tuple(os.path.join(GGUF_DIR, gguf_name(stem, q)) for q in info.quants)


# --- the device sub-paths, each with the tag its run must show ---------------------------------------------


def fused_step(kind: FamilyKind) -> bool:
    """whether the MLX step elects its fused kernels for this family's fixture, by the engine's own gate
    (mlx_forward.py:1132): the kernel layout or the sandwich layout, and a rotary over the whole head."""
    fl = core.flags(kind)
    return (Cap.KERNEL_LAYOUT in fl or Cap.SANDWICH in fl) and rope_dim(kind) == head_dim(kind)


def mlx_compute_tag(kind: FamilyKind, *, fused: bool) -> PassTag:
    """the MLX compute tag a step-path pass must show: a hybrid family runs the DeltaNet forward, a MoE family
    the per-op host path, and the rest the step - fused where the layout and the bf16 state allow it."""
    fl = core.flags(kind)
    if Cap.HYBRID in fl:
        return PassTag.MLX_HYBRID
    if Cap.MOE in fl:
        return PassTag.MLX_PEROP  # MoE has no fused step; its layers run the per-op host path on MLX
    return PassTag.MLX_STEP_FUSED if fused and fused_step(kind) else PassTag.MLX_STEP_UNFUSED


def _quant_tag(storage: Storage) -> PassTag:
    """the stored-weight tag an as-stored MLX run must show: a float GGUF has no packed kernel, so it binds a
    bf16 slot even with gguf_packed on, and an MXFP4 GGUF's weight is the expert block - it never reaches
    `mlx_state.affine`, so what the run proves is the expert store's as-stored MXFP4 matvec (host.py:317)."""
    qc = quant_class(storage)
    if qc is QuantClass.MXFP4:
        return PassTag.EXPERT_MXFP4_ASSTORED
    return PassTag.QUANT_DEQUANT if qc in (None, QuantClass.FLOAT) else PassTag.QUANT_ASSTORED


def residency_tag(kind: FamilyKind, knobs: dict[str, object]) -> PassTag | None:
    """the expert store's residency policy a run must show, or None for a family that builds no store: only a
    MoE family has one, and `bus_pass` picks the arm (experts.py:441). Absent from the knobs, the option's
    documented default of 1 applies - the Bus Pass - which is what the DEFAULT sub-paths therefore expect."""
    if Cap.MOE not in core.flags(kind):
        return None
    return PassTag.EXPERT_LINE if knobs.get("bus_pass", 1) == 0 else PassTag.EXPERT_BUS_PASS


def storage_tag(storage: Storage, hw: Hardware) -> PassTag | None:
    """the tag a storage's own read leaves, or None where it has none: FP8 multiplied as stored on the host's
    kernels, widened into the bf16 slots MLX and a card read (families.py `f8_host`)"""
    if storage is not Storage.SAFE_FP8:
        return None
    return PassTag.FP8_ASSTORED if hw is Hardware.CPU else PassTag.FP8_WIDENED


@dataclass(frozen=True)
class DeviceSubpath:
    """one set of engine branches, named by `key` and selected by `knobs` (every one a real `options.KNOWN`
    load option). `expect` is the PassTag a run of this sub-path MUST carry for the cell to count - it travels
    with the sub-path so the runner cannot assert the wrong tag, or none, by mis-spelling a key. `only` names
    the container whose knob this is, `needs` the family capability it acts on: without either the knob is a
    no-op and the run is not distinct."""

    key: str
    hardware: Hardware
    expect: Callable[[FamilyKind, Storage], PassTag]
    knobs: dict[str, object] = field(default_factory=dict)
    note: str = ""
    only: Container | None = None
    needs: Cap | None = None

    def expects(self, kind: FamilyKind, storage: Storage) -> frozenset[PassTag]:
        """every tag a run of this cell must carry: the sub-path's own, the residency policy this cell's own knobs
        select where the family has an expert store - so the default's Bus Pass is asserted too - and the
        storage's own read where it leaves one (`storage_tag`)."""
        extra = (residency_tag(kind, self.knobs), storage_tag(storage, self.hardware))
        return frozenset({self.expect(kind, storage), *(t for t in extra if t is not None)})


# device sub-paths, not devices: each a distinct set of branches
DEVICE_SUBPATHS: tuple[DeviceSubpath, ...] = (
    DeviceSubpath(
        "cpu", Hardware.CPU, lambda k, s: PassTag.CPU_NATIVE, {"device": "cpu"}, "fp32 over bf16, native gemv"
    ),
    DeviceSubpath(
        "cpu-headstream",
        Hardware.CPU,
        lambda k, s: PassTag.HEAD_STREAMED,
        {"device": "cpu", "resident_head": 0},
        "the head read from the checkpoint per pass instead of held",
    ),
    DeviceSubpath(
        "cpu-riders",
        Hardware.CPU,
        lambda k, s: PassTag.EXPERT_LINE,
        {"device": "cpu", "bus_pass": 0},
        "the expert store's plain line instead of the default Bus Pass",
        needs=Cap.MOE,
    ),
    DeviceSubpath(
        "mlx-mega", Hardware.MLX, lambda k, s: PassTag.MLX_MEGA, {"device": "mlx"}, "fused megakernel (qwen3's layout)"
    ),
    DeviceSubpath(
        "mlx-step",
        Hardware.MLX,
        lambda k, s: mlx_compute_tag(k, fused=True),
        {"device": "mlx", "mlx_mega": 0},
        "the per-token graph, megakernel off",
    ),
    DeviceSubpath(
        "mlx-packed",
        Hardware.MLX,
        lambda k, s: _quant_tag(s),
        {"device": "mlx", "gguf_packed": 1},
        "GGUF blocks as stored",
        Container.GGUF,
    ),
    DeviceSubpath(
        "mlx-dequant",
        Hardware.MLX,
        lambda k, s: PassTag.QUANT_DEQUANT,
        {"device": "mlx", "gguf_packed": 0},
        "GGUF dequantized to bf16",
        Container.GGUF,
    ),
    DeviceSubpath(
        "mlx-kvbits",
        Hardware.MLX,
        lambda k, s: mlx_compute_tag(k, fused=True),
        {"device": "mlx", "mlx_mega": 0, "kv_bits": 8},
        "int8 KV rows",
    ),
    DeviceSubpath(
        "mlx-fp32",
        Hardware.MLX,
        lambda k, s: mlx_compute_tag(k, fused=False),
        {"device": "mlx", "fp32": 1},
        "fp32 compute: the fused kernels take a bf16 state, so the step runs unfused",
    ),
    DeviceSubpath(
        "mlx-riders",
        Hardware.MLX,
        lambda k, s: PassTag.EXPERT_LINE,
        {"device": "mlx", "bus_pass": 0},
        "the expert store's plain line instead of the default Bus Pass",
        needs=Cap.MOE,
    ),
    DeviceSubpath("cuda-graph", Hardware.CUDA, lambda k, s: PassTag.CUDA_GRAPH, {"device": "cuda"}, "card step graph"),
    DeviceSubpath(
        "cuda-torch",
        Hardware.CUDA,
        lambda k, s: PassTag.CUDA_TORCH_FALLBACK,
        {"device": "cuda", "fp32": 1},
        "fp32 compute: the card kernels are bf16's, so every layer runs the torch modules",
    ),
    DeviceSubpath(
        "cuda-split",
        Hardware.CUDA,
        lambda k, s: PassTag.TIER_HOST,
        {"device": "cuda", "cpu_layers": 1},
        "cpu/card split",
    ),
    DeviceSubpath(
        "cuda-kvhost",
        Hardware.CUDA,
        lambda k, s: PassTag.CUDA_TORCH_FALLBACK,
        {"device": "cuda", "kv_host": 1},
        "KV in host RAM, which the card graph cannot read (cuda.py:303)",
    ),
    DeviceSubpath(
        "cuda-riders",
        Hardware.CUDA,
        lambda k, s: PassTag.EXPERT_LINE,
        {"device": "cuda", "bus_pass": 0},
        "the expert store's plain line instead of the default Bus Pass",
        needs=Cap.MOE,
    ),
)

SUBPATH: dict[str, DeviceSubpath] = {d.key: d for d in DEVICE_SUBPATHS}


def subpaths(*keys: str) -> tuple[DeviceSubpath, ...]:
    """the named sub-paths, from the one DEVICE_SUBPATHS table - a caller names keys, never re-spells knobs."""
    return tuple(SUBPATH[k] for k in keys)


class Surface(StrEnum):
    """the id namespace a cert run records under: the container it loaded from, or an input-shape axis that is
    not part of the storage/device cartesian (a batch of rows, a long prompt)."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    PACK12 = "pack12"
    BATCH = "batch"
    CONTEXT = "context"


# the surface a cell of each container records under; the shape surfaces have no container of their own.
CONTAINER_SURFACE: dict[Container, Surface] = {
    Container.SAFETENSORS: Surface.SAFETENSORS,
    Container.PACK12: Surface.PACK12,
    Container.GGUF: Surface.GGUF,
}


def container_subpaths(container: Container) -> tuple[str, ...]:
    """every sub-path a container's cells run: those whose knob is no other container's"""
    return tuple(d.key for d in DEVICE_SUBPATHS if d.only in (None, container))


# which device sub-paths the runner exercises on each surface: a container's every sub-path, and for the shape
# axes the representative one per backend (a combination absent here is a manifest error, never a silent hole)
SURFACE_SUBPATHS: dict[Surface, tuple[str, ...]] = {
    **{CONTAINER_SURFACE[c]: container_subpaths(c) for c in Container},
    Surface.BATCH: ("cpu", "mlx-step"),
    Surface.CONTEXT: ("cpu", "mlx-step"),
}


@dataclass(frozen=True)
class SpecSetup:
    """how the runner drives a speculative decode path: `knobs` on the load (a verify budget, which a MoE
    family's load defaults to none), the proposer set on the loaded model (no load option picks the MTP chain
    or fixed tree), `draft` for the family's own fixture loaded again as its sibling draft model, and `echo`
    for the plain decode handed to the n-gram proposer as a span (a tiny random fixture's output seldom repeats
    itself, and repeats are what n-grams draft from)"""

    knobs: dict[str, object]
    proposer: Proposer
    draft: bool = False
    echo: bool = False


def _spec_setup(decode: DecodePath) -> SpecSetup:
    proposer = DECODE_PROPOSER[decode] or Proposer.NGRAM
    draft = DECODE_KIND[decode] is DecodeKind.DRAFT_MODEL
    knobs: dict[str, object] = {"v_max": 4}
    if proposer is Proposer.MTP_DYN:
        knobs["tree_min_prob"] = 0.0  # a tiny random model's flat distribution clears no path-probability floor
    return SpecSetup(knobs, proposer, draft, echo=proposer is Proposer.NGRAM and not draft)


SPEC_SETUP: dict[DecodePath, SpecSetup] = {
    d: _spec_setup(d) for d in DecodePath if DECODE_KIND[d] is not DecodeKind.PLAIN
}


def decode_tag(decode: DecodePath) -> PassTag:
    """the tag a decode path's run must show: the proposer it ran, a sibling draft model's own, or none drafted"""
    setup = SPEC_SETUP.get(decode)
    if setup is None:
        return PassTag.SPEC_OFF
    return PassTag.SPEC_DRAFT if setup.draft else PROPOSER_TAG[setup.proposer]


def totality_problems() -> list[str]:
    """where an axis has fallen behind the btb.kinds declaration it stands for, or refers to something that is
    not there: an unclassified Quant, a Proposer with no decode path, a hardware with no sub-path. Every one is
    a way a cell could read as covered without the path being modelled, so test_manifest holds this empty."""
    problems: list[str] = []
    bound: dict[Quant, Storage] = {}
    for storage, info in STORAGE.items():
        if (info.container is Container.GGUF) != bool(info.quants):
            problems.append(f"{storage.value!r}: a GGUF member stands for stored types, and only a GGUF member")
        for q in info.quants:
            if q in bound:
                problems.append(f"{q.value!r} is claimed by both {bound[q].value!r} and {storage.value!r}")
            bound[q] = storage
        classes = {QUANT_KIND[q] for q in info.quants}
        if len(classes) > 1:
            problems.append(f"{storage.value!r} mixes bind paths {sorted(c.value for c in classes)}")
    for q in Quant:
        if q not in bound:
            problems.append(f"{q.value!r} has no Storage member; a new stored type needs one to be certified")
    for s in Storage:
        if s not in STORAGE:
            problems.append(f"{s.value!r} has no STORAGE row")
    for d in DecodePath:
        if d not in DECODE_KIND or d not in DECODE_PROPOSER:
            problems.append(f"{d.value!r} has no DECODE_KIND/DECODE_PROPOSER row")
            continue
        by_proposer = DECODE_PROPOSER[d] is not None
        if by_proposer != (DECODE_KIND[d] is DecodeKind.PROPOSER):
            problems.append(f"{d.value!r}: its DecodeKind and its Proposer disagree")
    for p in Proposer:
        if sum(1 for d in DecodePath if DECODE_PROPOSER[d] is p) != 1:
            problems.append(f"proposer {p.value!r} is not exactly one DecodePath")
    for hw in Hardware:
        if not any(d.hardware is hw for d in DEVICE_SUBPATHS):
            problems.append(f"hardware {hw.value!r} has no device sub-path")
    for container in Container:
        if container not in CONTAINER_SURFACE:
            problems.append(f"container {container.value!r} records under no Surface")
    if len(SUBPATH) != len(DEVICE_SUBPATHS):
        problems.append("two device sub-paths share a key")
    for surface, keys in SURFACE_SUBPATHS.items():
        for key in keys:
            if key not in SUBPATH:
                problems.append(f"surface {surface.value!r} names sub-path {key!r}, which is not declared")
    return problems
