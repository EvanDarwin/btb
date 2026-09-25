"""P5, the derived correctness oracle: a banked greedy and a banked seeded-sampled continuation per served family
that a cert cell's decode must reproduce. Determinism (test_cert_runner) proves a path runs the same twice; it does
NOT prove the answer is right - a deterministically-wrong path passes it. The oracle closes that: the reference is
the engine's own decode of the tiny fixture, banked per `core.served_kinds()` + `spec.FIXTURE_STEM` (derived from core, never a
hand list like test_receipts' run_* mirror), content-hashed against the fixture so a fixture change invalidates a
stale bank, and regenerated deterministically so a fresh bank is byte-identical to the committed one.

A sampled draw is held exactly where the engine computes in fp32, as the reference does (`holds_sampled`): the
Gumbel draw is keyed by (seed, row, position), so a fixed seed makes it a deterministic function of the logits.
In bf16 it is not held - the tiny fixtures' logits are nearly flat, so bf16 rounding alone moves a token across
the top-k/top-p cut (MLX in fp32 reproduces the CPU draw exactly; in bf16 it does not). A sampled cell on a bf16
path instead holds a greedy decode of the same load to the greedy reference, so it still cannot certify a path
that decodes wrong.

The check is EXACT tokens, not a logit-distance tolerance: test_receipts already holds the banked greedy
continuation to exact equality on every device (`g == B["greedy"]` on CPU and MLX alike), so exact tokens are the
strongest correctness signal that still holds cross-device. A tolerance band would be the very hole P5 exists to
close - it lets a wrong-but-close path pass. If a real near-tie ever flips a token across devices, that is a
finding to report (per `validate`), not a tolerance to widen.

    python -m tests.cert.oracle              # show the bank
    python -m tests.cert.oracle --rebank     # regenerate and write bank.json (run under the GPU lock)
    python -m tests.cert.oracle --check      # regenerate on the reference device; nonzero if it drifts
    python -m tests.cert.oracle --validate --device mlx   # a second device must reproduce the same tokens
    python -m tests.cert.oracle --validate --device mlx --fp32   # ... in fp32, where the sampled draws are held too
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import TYPE_CHECKING, TypedDict, cast

from btb.kinds import FamilyKind, Json

from . import core, spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from btb.engine.model import StreamedTextModel
    from btb.sampling import Sampling

    CellOutput = Sequence[int] | Tensor  # what a runner cell hands us: the decoded tokens, or a logits row

SCHEMA = 2
REFERENCE_DEVICE = "cpu"  # the always-present tier and the native gemv path make_fixtures banks the receipts on
# The greedy decode every cert run shares, spelled once here: the runner's cells decode PROMPT for N tokens and
# this bank holds what that decode must produce, so the two cannot drift into comparing different runs.
N = 6
PROMPT: list[int] = [1, 2, 3, 4]


# the one sampled decode, likewise shared: the runner's sampled cells use it and the bank holds its continuation
# (btb.sampling imports torch, so the Sampling is built on use: this module stays importable by the torch-free gate)
class SampledParams(TypedDict):
    temperature: float
    top_p: float
    top_k: int
    seed: int


SAMPLED: SampledParams = {"temperature": 0.8, "top_p": 0.95, "top_k": 40, "seed": 20260919}


def sampling(key: str) -> Sampling | None:
    """the decode a banked key stands for: the SAMPLED draw for "sampled", None (greedy) for "tokens\""""
    from btb.sampling import Sampling

    return Sampling(**SAMPLED) if key == "sampled" else None


# the reference inputs: "standard" is the runner's prompt, "single" the boundary case (a one-token prefill, gap
# #30). A zero-length prompt is out of scope: see NOTES.
PROMPTS: dict[str, list[int]] = {"standard": list(PROMPT), "single": [1]}
NOTES: dict[str, str] = {
    "empty": "a zero-length prompt is out of scope: the engine rejects it at the gemv/matvec (1..16 rows) on "
    "every device, so there is no decode to bank as a boundary input; the single-token prompt covers boundary "
    "prefill instead",
}

ORACLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle")
BANK = os.path.join(ORACLE_DIR, "bank.json")
FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def banked_kinds() -> list[FamilyKind]:
    """the served families a reference is banked for: `core.served_kinds()` that have a fixture stem - derived
    from core, so a new served family with a fixture forces a reference (test_oracle fails until it is banked)."""
    return [k for k in core.served_kinds() if k in spec.FIXTURE_STEM]


def fixture_dir(kind: FamilyKind) -> str:
    """the tiny safetensors fixture that stands in for a family (the bf16 checkpoint the reference is decoded from)."""
    return os.path.join(FIXTURES, spec.FIXTURE_STEM[kind])


# the storages whose twin rounds the weights, so it cannot decode to its family's tokens: each is banked from its
# own twin, under the family's `storages`, and a cell of that storage is held to that reference
LOSSY: tuple[spec.Storage, ...] = (spec.Storage.SAFE_FP8,)


def lossy_twins(kind: FamilyKind) -> dict[spec.Storage, str]:
    """the lossy twins of `kind` on disk, by storage"""
    return {s: p for s in LOSSY for p in spec.fixture_paths(kind, s)}


def banked_entry(fam: Json, storage: spec.Storage) -> Json | None:
    """a family's banked reference for a cell of `storage`: its own for a lossy storage, else the family's"""
    return fam.get("storages", {}).get(storage.value) if storage in LOSSY else fam


def fixture_hash(path: str) -> str:
    """a content hash of every input file in a fixture dir (name + bytes, sorted): the staleness key. A change to
    the config, the weights or the tokenizer changes the hash, so a stale bank is caught against a fresh fixture."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if not os.path.isfile(fp):
            continue
        h.update(name.encode())
        h.update(b"\0")
        with open(fp, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


# --- banking (runs the engine) -----------------------------------------------------------------------------


def holds_sampled(sm: StreamedTextModel) -> bool:
    """whether a sampled draw from `sm` is held to the banked one: only where it computes in fp32, as the reference
    does - in bf16 a near-tie at the sampler's cut falls on either side by rounding (see the module docstring)"""
    import torch

    return sm.compute_dtype is torch.float32


def decode(sm: StreamedTextModel, prompt_ids: list[int], sampling: Sampling | None = None) -> list[int]:
    """N new tokens of `prompt_ids` on a loaded model - greedy, or seeded-sampled with `sampling`. Speculation off."""
    return [int(t) for t in sm.generate(list(prompt_ids), N, speculate=False, sampling=sampling).tokens]


def reference_tokens(
    kind: FamilyKind, prompt_ids: list[int], device: str, sampling: Sampling | None = None, path: str | None = None
) -> list[int]:
    """the engine's own continuation of `prompt_ids` from the family's fixture (or the twin at `path`) on
    `device` - the reference a cell must reproduce."""
    import btb
    from tests.helpers import NO_LOG

    sm = btb.load(path or fixture_dir(kind), device=device, log=NO_LOG)
    try:
        return decode(sm, prompt_ids, sampling)
    finally:
        sm.close()


# the banked decodes, by their key in a family's entry: greedy under "tokens", the seeded draw under "sampled"
DECODES: tuple[str, ...] = ("tokens", "sampled")

# a greedy step whose top-2 gap is under this share of its largest |logit| is a near-tie: bf16 compute and int8 KV
# move a fixture's logits up to ~1.5% of that scale (transformers' own bf16 forward drifts as far), so a device path
# could take the other token and the exact-token oracle would fail on rounding. The bank refuses such a decode.
MARGIN_FLOOR = 0.02


def greedy_margins(sm: StreamedTextModel, prompt_ids: list[int]) -> tuple[list[int], list[float | None]]:
    """N greedy tokens of `prompt_ids` stepped one forward at a time, and each step's top-2 logit gap as a share of
    its largest |logit|; None where every logit is exactly equal (a zero hidden state), which argmax's first-index
    rule decides the same on every path"""
    import torch

    from tests.helpers import forward_logits

    cache = sm.new_cache()
    lg = forward_logits(sm, [prompt_ids], cache)[0, -1].float()
    toks: list[int] = []
    gaps: list[float | None] = []
    for _ in range(N):
        top, scale = torch.topk(lg, 2).values, float(lg.abs().max())
        gaps.append(None if scale == 0 else round(float(top[0] - top[1]) / scale, 6))
        toks.append(int(lg.argmax()))
        lg = forward_logits(sm, [[toks[-1]]], cache)[0, -1].float()
    return toks, gaps


def reference_margins(
    kind: FamilyKind, prompt_ids: list[int], device: str, path: str | None = None
) -> tuple[list[int], list[float | None]]:
    """`greedy_margins` on the family's fixture (or the twin at `path`) on `device`"""
    import btb
    from tests.helpers import NO_LOG

    sm = btb.load(path or fixture_dir(kind), device=device, log=NO_LOG)
    try:
        return greedy_margins(sm, prompt_ids)
    finally:
        sm.close()


def _bank_margins(kind: FamilyKind, entry: Json, device: str, path: str | None, label: str) -> None:
    """`entry`'s greedy margins per prompt, each stepped decode held to the banked `generate` tokens"""
    entry["margins"] = {}
    for name, ids in PROMPTS.items():
        toks, gaps = reference_margins(kind, ids, device, path)
        if toks != entry["tokens"][name]:
            raise RuntimeError(f"{label}/{name}: stepped greedy {toks} != generate's {entry['tokens'][name]}")
        entry["margins"][name] = gaps


def bank(kinds: list[FamilyKind] | None = None, device: str = REFERENCE_DEVICE) -> Json:
    """build the oracle bank: for each banked family (and lossy twin), its fixture hash, its greedy and
    seeded-sampled continuations of every prompt, and the greedy ones' step margins (`greedy_margins`). A pure
    function of the fixtures, the prompts and the engine (no timestamp), so a rebank is byte-reproducible."""
    families: Json = {}
    for kind in kinds if kinds is not None else banked_kinds():
        fam: Json = {"fixture": spec.FIXTURE_STEM[kind], "fixture_hash": fixture_hash(fixture_dir(kind))}
        for key in DECODES:
            fam[key] = {name: reference_tokens(kind, ids, device, sampling(key)) for name, ids in PROMPTS.items()}
        _bank_margins(kind, fam, device, None, kind.value)
        twins = lossy_twins(kind)
        if twins:
            fam["storages"] = {}
            for storage, path in twins.items():
                entry: Json = {"fixture": os.path.basename(path), "fixture_hash": fixture_hash(path)}
                for key in DECODES:
                    entry[key] = {
                        n: reference_tokens(kind, ids, device, sampling(key), path) for n, ids in PROMPTS.items()
                    }
                _bank_margins(kind, entry, device, path, f"{kind.value}/{storage.value}")
                fam["storages"][storage.value] = entry
        families[kind.value] = fam
    return {
        "schema": SCHEMA,
        "reference_device": device,
        "decode": {
            "n": N,
            "sampled": dict(SAMPLED),
        },
        "prompts": PROMPTS,
        "notes": NOTES,
        "families": families,
    }


def _canon(doc: Json) -> str:
    """the bank's on-disk bytes: canonical JSON (sorted keys), so a fresh regeneration compares byte-for-byte."""
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _sans_margins(doc: Json) -> str:
    """`_canon` without the step margins: their last digit follows the ISA's fp32 rounding, so a rebank on
    another machine reproduces the tokens and the structure but not those bytes"""

    def drop(entry: Json) -> Json:
        return {k: v for k, v in entry.items() if k != "margins"}

    families: Json = {}
    for name, fam in doc["families"].items():
        families[name] = drop(fam)
        if "storages" in fam:
            families[name]["storages"] = {s: drop(e) for s, e in fam["storages"].items()}
    return _canon({**doc, "families": families})


def thin_margins(doc: Json) -> list[str]:
    """every banked greedy decode (a family's, a lossy twin's) with a step under `MARGIN_FLOOR`, as
    label/prompt with its smallest gap, or one with no margins banked; reads the bank only"""
    out: list[str] = []
    for name, fam in sorted(doc["families"].items()):
        entries = [(name, fam)] + [(f"{name}/{s}", e) for s, e in sorted(fam.get("storages", {}).items())]
        for label, entry in entries:
            banked = entry.get("margins")
            if banked is None:
                out.append(f"{label}: no margins banked (rebank)")
                continue
            for prompt, gaps in sorted(banked.items()):
                low = [g for g in gaps if g is not None and g < MARGIN_FLOOR]
                if low:
                    out.append(f"{label}/{prompt}: a step's top-2 gap is {min(low):.2%} of its logit scale")
    return out


def write_bank(doc: Json) -> None:
    """write the bank, refusing one with a near-tie decode (`thin_margins`): its fixture needs another draw"""
    thin = thin_margins(doc)
    if thin:
        raise RuntimeError(
            f"refusing to bank decodes under the {MARGIN_FLOOR:.0%} margin floor; redraw these fixtures:\n  "
            + "\n  ".join(thin)
        )
    os.makedirs(ORACLE_DIR, exist_ok=True)
    with open(BANK, "w", encoding="utf-8") as f:
        f.write(_canon(doc))


def load_bank() -> Json:
    """the committed bank; a FileNotFoundError here means it was never generated (`--rebank`)."""
    with open(BANK, encoding="utf-8") as f:
        doc: Json = json.load(f)
    return doc


# --- the runner's entry point ------------------------------------------------------------------------------


def _is_logits(output: object) -> bool:
    """a logits row (a tensor), as against a token list: only the tensor carries `argmax`/`shape`."""
    return hasattr(output, "argmax") and hasattr(output, "shape")


def _to_tokens(output: CellOutput) -> list[int]:
    """a cell's output as a token list: a token sequence as itself, a logits row as its greedy argmax (the final
    row when it is [T, V]) - the one next token greedy decode would pick."""
    if _is_logits(output):
        t = cast("Tensor", output)
        row = t[-1] if t.ndim > 1 else t
        return [int(row.argmax())]
    return [int(x) for x in cast("Sequence[int]", output)]


def assert_matches(
    kind: FamilyKind,
    output: CellOutput,
    device: str,
    *,
    prompt: str = "standard",
    sampled: bool = False,
    storage: spec.Storage = spec.Storage.SAFE_BF16,
) -> None:
    """the correctness gate a runner cell calls: the cell's output must equal the banked reference for
    `(kind, prompt)` - the greedy one, or with `sampled` the SAMPLED draw; a lossy storage's cell its own twin's
    (`banked_entry`). Tokens are matched exactly; a logits row is matched on its greedy next token. `device` is
    reported on a mismatch - the reference is device-independent, so a cross-device flip is a real discrepancy,
    surfaced here rather than hidden by a tolerance. A family with no banked reference fails (a COVERED cell with
    no oracle is not covered)."""
    fam = load_bank()["families"].get(kind.value)
    assert fam is not None, f"no banked oracle for {kind.value} (run python -m tests.cert.oracle --rebank)"
    entry = banked_entry(fam, storage)
    assert entry is not None, f"no banked {storage.value} oracle for {kind.value} (rebank)"
    key = "sampled" if sampled else "tokens"
    ref = entry.get(key, {}).get(prompt)
    assert ref is not None, f"no banked {key}/{prompt!r} reference for {kind.value} (rebank)"
    got = _to_tokens(output)
    assert got == ref[: len(got)], (
        f"{kind.value}/{key}/{prompt} on {device}: decode {got} != banked oracle {ref[: len(got)]} "
        f"(a deterministically-wrong path, or a real cross-device discrepancy - do not widen a tolerance to hide it)"
    )


# --- the guards --------------------------------------------------------------------------------------------


def _subjects(kind: FamilyKind, fam: Json | None) -> list[tuple[str, str, Json | None]]:
    """what a family banks, as (label, checkpoint, its committed entry): the fixture, then each lossy twin"""
    out: list[tuple[str, str, Json | None]] = [(kind.value, fixture_dir(kind), fam)]
    for storage, path in lossy_twins(kind).items():
        out.append((f"{kind.value}/{storage.value}", path, None if fam is None else banked_entry(fam, storage)))
    return out


def stale_families() -> list[str]:
    """banked families (and lossy twins, as family/storage) whose committed fixture hash no longer matches the
    checkpoint on disk - a stale bank. Cheap: hashes files, runs no engine, so test_oracle can gate staleness
    without loading a model."""
    committed = load_bank()["families"]
    return [
        label
        for kind in banked_kinds()
        if kind.value in committed
        for label, path, entry in _subjects(kind, committed[kind.value])
        if entry is not None and entry["fixture_hash"] != fixture_hash(path)
    ]


def check() -> list[str]:
    """the reproducibility guard: regenerate the bank on its own reference device and compare to the committed
    one. Returns the problems (fixture drift, token drift, or a byte difference), empty when it reproduces. Runs
    the engine, so under the GPU lock for MLX; the committed reference device is CPU, which needs no lock."""
    committed = load_bank()
    device = str(committed.get("reference_device", REFERENCE_DEVICE))
    fresh = bank(device=device)
    problems: list[str] = []
    for kind in banked_kinds():
        fr_fam = fresh["families"][kind.value]
        for (label, _path, c), (_l, _p, fr) in zip(
            _subjects(kind, committed["families"].get(kind.value)), _subjects(kind, fr_fam), strict=True
        ):
            assert fr is not None  # a fresh bank holds every subject
            if c is None:
                problems.append(f"{label}: no committed reference")
                continue
            if c["fixture_hash"] != fr["fixture_hash"]:
                problems.append(f"{label}: fixture changed ({c['fixture_hash']} != {fr['fixture_hash']}); rebank")
            for key in DECODES:
                for name in PROMPTS:
                    was, now = c.get(key, {}).get(name), fr[key][name]
                    if was != now:
                        problems.append(f"{label}/{key}/{name}: tokens drifted ({was} != {now})")
    if _sans_margins(fresh) != _sans_margins(committed) and not problems:
        problems.append("bank bytes differ from a fresh regeneration (metadata/structure changed); rebank")
    problems += [f"fresh regeneration: {t}" for t in thin_margins(fresh)]
    return problems


def validate(device: str, fp32: bool = False) -> list[str]:
    """a second device must reproduce the banked tokens: decode every family (and lossy twin) and prompt on
    `device` (in fp32 with `fp32`) and compare to the committed reference - greedy always, sampled where
    `holds_sampled`. Returns the discrepancies - a token mismatch (a real cross-device flip) or an engine error on
    the path (e.g. a family the device cannot run) - empty when every one reproduces. Each is caught on its own,
    so one broken path does not hide whether the rest reproduce."""
    import btb
    from tests.helpers import NO_LOG

    committed = load_bank()["families"]
    problems: list[str] = []
    for kind in banked_kinds():
        for label, path, entry in _subjects(kind, committed.get(kind.value)):
            try:
                # fp32 only when asked: leaving it unset keeps the device's own default (the CPU computes in fp32)
                sm = (
                    btb.load(path, device=device, log=NO_LOG, fp32=1)
                    if fp32
                    else btb.load(path, device=device, log=NO_LOG)
                )
            except Exception as e:  # the engine's failure on a path is itself a finding, not a reason to abort
                problems.append(f"{label} on {device}: engine error at load ({type(e).__name__}: {e})")
                continue
            try:
                for key in DECODES:
                    draw = sampling(key)
                    if draw is not None and not holds_sampled(sm):
                        continue
                    for name, ids in PROMPTS.items():
                        try:
                            got = decode(sm, ids, draw)
                        except Exception as e:
                            problems.append(f"{label}/{key}/{name} on {device}: engine error ({type(e).__name__}: {e})")
                            continue
                        ref = None if entry is None else entry.get(key, {}).get(name)
                        if got != ref:
                            problems.append(f"{label}/{key}/{name} on {device}: {got} != banked {ref}")
            finally:
                sm.close()
    return problems


# --- CLI ---------------------------------------------------------------------------------------------------


def _show() -> int:
    doc = load_bank()
    print(
        f"oracle bank: schema {doc['schema']}, banked on {doc['reference_device']}, "
        f"{len(doc['families'])} families, prompts {list(doc['prompts'])}"
    )
    for name in sorted(doc["families"]):
        fam = doc["families"][name]
        toks = ", ".join(f"{k}/{p}={fam[k][p]}" for k in DECODES for p in sorted(fam.get(k, {})))
        print(f"  {name:<10} {fam['fixture_hash'][:19]}...  {toks}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebank", action="store_true", help="regenerate the bank and write bank.json")
    ap.add_argument("--check", action="store_true", help="regenerate on the reference device; nonzero if it drifts")
    ap.add_argument("--validate", action="store_true", help="a second device (--device) must reproduce the tokens")
    ap.add_argument("--device", default=REFERENCE_DEVICE, help="the device to bank/validate on (default cpu)")
    ap.add_argument("--fp32", action="store_true", help="validate with fp32 compute (the sampled draws are held too)")
    a = ap.parse_args(argv)
    if a.rebank:
        write_bank(bank(device=a.device))
        return _show()
    if a.check:
        problems = check()
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1 if problems else 0
    if a.validate:
        problems = validate(a.device, a.fp32)
        for p in problems:
            print(f"FAIL cross-device: {p}", file=sys.stderr)
        print(f"{a.device}: {'reproduced the banked tokens' if not problems else f'{len(problems)} mismatch(es)'}")
        return 1 if problems else 0
    return _show()


if __name__ == "__main__":
    raise SystemExit(main())
