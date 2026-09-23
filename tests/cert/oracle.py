"""P5, the derived correctness oracle: a banked greedy continuation per served family that a cert cell's decode
must reproduce. Determinism (test_cert_runner) proves a path runs the same twice; it does NOT prove the answer is
right - a deterministically-wrong path passes it. The oracle closes that: the reference is the engine's own greedy
decode of the tiny fixture, banked per `core.served_kinds()` + `spec.FIXTURE_STEM` (derived from core, never a
hand list like test_receipts' run_* mirror), content-hashed against the fixture so a fixture change invalidates a
stale bank, and regenerated deterministically so a fresh bank is byte-identical to the committed one.

The check is EXACT greedy tokens, not a logit-distance tolerance: test_receipts already holds the banked greedy
continuation to exact equality on every device (`g == B["greedy"]` on CPU and MLX alike), so exact tokens are the
strongest correctness signal that still holds cross-device. A tolerance band would be the very hole P5 exists to
close - it lets a wrong-but-close path pass. If a real near-tie ever flips a token across devices, that is a
finding to report (per `validate`), not a tolerance to widen.

    python -m tests.cert.oracle              # show the bank
    python -m tests.cert.oracle --rebank     # regenerate and write bank.json (run under the GPU lock)
    python -m tests.cert.oracle --check      # regenerate on the reference device; nonzero if it drifts
    python -m tests.cert.oracle --validate --device mlx   # a second device must reproduce the same tokens
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import TYPE_CHECKING, cast

from btb.kinds import FamilyKind, Json

from . import core, spec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    CellOutput = Sequence[int] | Tensor  # what a runner cell hands us: the decoded tokens, or a logits row

SCHEMA = 1
REFERENCE_DEVICE = "cpu"  # the always-present tier and the native gemv path make_fixtures banks the receipts on
# The greedy decode every cert run shares, spelled once here: the runner's cells decode PROMPT for N tokens and
# this bank holds what that decode must produce, so the two cannot drift into comparing different runs.
N = 6
PROMPT: list[int] = [1, 2, 3, 4]

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


def reference_tokens(kind: FamilyKind, prompt_ids: list[int], device: str) -> list[int]:
    """the engine's own greedy continuation of `prompt_ids` from the family's fixture on `device`, N new tokens -
    the reference a cell must reproduce. Speculation off, so it is the plain greedy path."""
    import btb
    from tests.helpers import NO_LOG

    sm = btb.load(fixture_dir(kind), device=device, log=NO_LOG)
    try:
        return [int(t) for t in sm.generate(list(prompt_ids), N, speculate=False).tokens]
    finally:
        sm.close()


def bank(kinds: list[FamilyKind] | None = None, device: str = REFERENCE_DEVICE) -> Json:
    """build the oracle bank: for each banked family, its fixture hash and greedy continuation of every prompt.
    A pure function of the fixtures, the prompts and the engine (no timestamp), so a rebank is byte-reproducible."""
    families: Json = {}
    for kind in kinds if kinds is not None else banked_kinds():
        families[kind.value] = {
            "fixture": spec.FIXTURE_STEM[kind],
            "fixture_hash": fixture_hash(fixture_dir(kind)),
            "tokens": {name: reference_tokens(kind, ids, device) for name, ids in PROMPTS.items()},
        }
    return {
        "schema": SCHEMA,
        "reference_device": device,
        "decode": {"n": N, "greedy": True},
        "prompts": PROMPTS,
        "notes": NOTES,
        "families": families,
    }


def _canon(doc: Json) -> str:
    """the bank's on-disk bytes: canonical JSON (sorted keys), so a fresh regeneration compares byte-for-byte."""
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def write_bank(doc: Json) -> None:
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


def assert_matches(kind: FamilyKind, output: CellOutput, device: str, *, prompt: str = "standard") -> None:
    """the correctness gate a runner cell calls: the cell's greedy output must equal the banked reference for
    `(kind, prompt)`. Tokens are matched exactly; a logits row is matched on its greedy next token. `device` is
    reported on a mismatch - the reference is device-independent (greedy tokens hold across devices for these
    fixtures), so a cross-device flip is a real discrepancy, surfaced here rather than hidden by a tolerance.
    A family with no banked reference fails (a COVERED cell with no oracle is not covered)."""
    fam = load_bank()["families"].get(kind.value)
    assert fam is not None, f"no banked oracle for {kind.value} (run python -m tests.cert.oracle --rebank)"
    ref = fam["tokens"].get(prompt)
    assert ref is not None, f"no banked {prompt!r} reference for {kind.value}"
    got = _to_tokens(output)
    assert got == ref[: len(got)], (
        f"{kind.value}/{prompt} on {device}: decode {got} != banked oracle {ref[: len(got)]} "
        f"(a deterministically-wrong path, or a real cross-device discrepancy - do not widen a tolerance to hide it)"
    )


# --- the guards --------------------------------------------------------------------------------------------


def stale_families() -> list[str]:
    """banked families whose committed fixture hash no longer matches the fixture on disk - a stale bank. Cheap:
    hashes files, runs no engine, so test_oracle can gate staleness without loading a model."""
    committed = load_bank()["families"]
    out: list[str] = []
    for kind in banked_kinds():
        fam = committed.get(kind.value)
        if fam is not None and fam["fixture_hash"] != fixture_hash(fixture_dir(kind)):
            out.append(kind.value)
    return out


def check() -> list[str]:
    """the reproducibility guard: regenerate the bank on its own reference device and compare to the committed
    one. Returns the problems (fixture drift, token drift, or a byte difference), empty when it reproduces. Runs
    the engine, so under the GPU lock for MLX; the committed reference device is CPU, which needs no lock."""
    committed = load_bank()
    device = str(committed.get("reference_device", REFERENCE_DEVICE))
    fresh = bank(device=device)
    problems: list[str] = []
    for kind in banked_kinds():
        c = committed["families"].get(kind.value)
        fr = fresh["families"][kind.value]
        if c is None:
            problems.append(f"{kind.value}: no committed reference")
            continue
        if c["fixture_hash"] != fr["fixture_hash"]:
            problems.append(f"{kind.value}: fixture changed ({c['fixture_hash']} != {fr['fixture_hash']}); rebank")
        for name in PROMPTS:
            if c["tokens"].get(name) != fr["tokens"][name]:
                problems.append(
                    f"{kind.value}/{name}: tokens drifted ({c['tokens'].get(name)} != {fr['tokens'][name]})"
                )
    if _canon(fresh) != _canon(committed) and not problems:
        problems.append("bank bytes differ from a fresh regeneration (metadata/structure changed); rebank")
    return problems


def validate(device: str) -> list[str]:
    """a second device must reproduce the banked tokens: decode every family and prompt on `device` and compare
    to the committed reference. Returns the discrepancies - a token mismatch (a real cross-device flip) or an
    engine error on the path (e.g. a family the device cannot run) - empty when every family reproduces. Each
    family is caught on its own, so one broken path does not hide whether the rest reproduce."""
    committed = load_bank()["families"]
    problems: list[str] = []
    for kind in banked_kinds():
        for name, ids in PROMPTS.items():
            try:
                got = reference_tokens(kind, ids, device)
            except Exception as e:  # the engine's failure on a path is itself a finding to report, not to abort on
                problems.append(f"{kind.value}/{name} on {device}: engine error ({type(e).__name__}: {e})")
                continue
            ref = committed[kind.value]["tokens"][name]
            if got != ref:
                problems.append(f"{kind.value}/{name} on {device}: {got} != banked {ref}")
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
        toks = ", ".join(f"{p}={fam['tokens'][p]}" for p in sorted(fam["tokens"]))
        print(f"  {name:<10} {fam['fixture_hash'][:19]}...  {toks}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebank", action="store_true", help="regenerate the bank and write bank.json")
    ap.add_argument("--check", action="store_true", help="regenerate on the reference device; nonzero if it drifts")
    ap.add_argument("--validate", action="store_true", help="a second device (--device) must reproduce the tokens")
    ap.add_argument("--device", default=REFERENCE_DEVICE, help="the device to bank/validate on (default cpu)")
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
        problems = validate(a.device)
        for p in problems:
            print(f"FAIL cross-device: {p}", file=sys.stderr)
        print(f"{a.device}: {'reproduced the banked tokens' if not problems else f'{len(problems)} mismatch(es)'}")
        return 1 if problems else 0
    return _show()


if __name__ == "__main__":
    raise SystemExit(main())
