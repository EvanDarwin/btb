"""The P5 oracle's own gates: every served family with a fixture has a banked correctness reference (the
derive-from-core hoop - a new family with none fails here), the bank is not stale against its fixtures (the
content-hash guard), no banked decode sits on a near-tie (the margin guard), and a fresh regeneration on the
reference device is byte-identical to the committed bank (the reproducibility guard). The cross-device reproduction (`oracle.validate`) runs on hardware the suite may
not have, so it is a manual `--validate` under the GPU lock, not a committed CPU test."""

from __future__ import annotations

import os

import pytest

from . import core, oracle, spec


def test_every_served_family_has_a_banked_reference() -> None:
    """the derive-from-core gate: every served family with a fixture stem must carry a banked reference. A family
    added to core (with a fixture) but never banked fails here - the oracle cannot silently omit a served path."""
    banked = set(oracle.load_bank()["families"])
    missing = [k.value for k in oracle.banked_kinds() if k.value not in banked]
    assert not missing, (
        f"served families with a fixture but no banked oracle: {missing} (run: python -m tests.cert.oracle --rebank)"
    )
    # and no reference for a family core no longer serves (the bank cannot drift ahead of core either)
    served = {k.value for k in core.served_kinds()}
    orphan = [name for name in banked if name not in served]
    assert not orphan, f"banked references for families core does not serve: {orphan}"


def test_bank_is_not_stale() -> None:
    """the staleness guard: each family's committed fixture hash matches the fixture on disk. Cheap - it hashes
    files and loads no model - so a fixture edit that was not followed by a rebank is caught in the fast suite."""
    stale = oracle.stale_families()
    assert not stale, (
        f"the oracle bank is stale for {stale}: the fixture changed since it was banked "
        f"(run: python -m tests.cert.oracle --rebank)"
    )


def test_boundary_prompt_is_covered() -> None:
    """gap #30: the reference set covers a boundary decode. The single-token prompt is banked for every family
    (the zero-length prompt is documented out of scope in oracle.NOTES, the engine rejecting it outright)."""
    fams = oracle.load_bank()["families"]
    assert "single" in oracle.PROMPTS and oracle.PROMPTS["single"] == [1]
    assert "empty" in oracle.NOTES  # the out-of-scope edge input is documented, not silently absent
    for kind in oracle.banked_kinds():
        for key in oracle.DECODES:
            assert fams[kind.value][key].get("single"), f"{kind.value}: no boundary (single-token) {key} reference"


def test_every_banked_family_carries_every_decode() -> None:
    """a sampled cell is held to the bank like a greedy one, so every family banks every decode for every prompt"""
    fams = oracle.load_bank()["families"]
    for kind in oracle.banked_kinds():
        for key in oracle.DECODES:
            assert set(fams[kind.value].get(key, {})) == set(oracle.PROMPTS), f"{kind.value}: {key} incomplete"


def test_every_banked_decode_clears_the_margin_floor() -> None:
    """the near-tie guard: every banked greedy decode, a family's and each lossy twin's, keeps each step's top-2
    gap above `oracle.MARGIN_FLOOR` - past what bf16 compute and int8 KV move a fixture's logits - so a device path
    decodes the banked tokens unless it computes wrong. Reads the bank only, like the staleness guard."""
    thin = oracle.thin_margins(oracle.load_bank())
    assert not thin, (
        f"banked decodes under the {oracle.MARGIN_FLOOR:.0%} margin floor (redraw the fixture):\n  " + "\n  ".join(thin)
    )


def test_reference_reproducible_on_cpu() -> None:
    """the reproducibility guard: regenerating the bank on the reference device (CPU, no GPU lock needed) is
    byte-identical to the committed one. Runs the engine over every family's fixture, like the receipts do."""
    if oracle.load_bank().get("reference_device") != "cpu":
        pytest.skip("bank was not banked on cpu; run `python -m tests.cert.oracle --validate` for other devices")
    for kind in oracle.banked_kinds():
        if not os.path.isdir(oracle.fixture_dir(kind)):
            pytest.skip(f"fixture {spec.FIXTURE_STEM[kind]} not built")
    problems = oracle.check()
    assert not problems, "the committed bank does not reproduce:\n  " + "\n  ".join(problems)
