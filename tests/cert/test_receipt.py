# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Coverage receipts (tests/cert/receipt.py): what emit() writes, what merged() unions back, and the default
directory CI and the matrices both code against. Torch-free - a receipt is JSON and nothing here loads a model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from tests.cert import receipt


def test_emit_then_merge_round_trips(tmp_path: Path) -> None:
    """what one machine banks is exactly what the union reads back, from the file or from its directory"""
    ids = {"safetensors/qwen3/cpu/greedy", "gguf/tiny_qwen3-q4_0/mlx-packed"}
    path = str(tmp_path / "box.json")
    receipt.emit(path, ids)
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assert doc["covered"] == sorted(ids)
    assert doc["machine"] and doc["when"].endswith("Z")
    assert receipt.merged(path) == ids
    assert receipt.merged(str(tmp_path)) == ids


def test_merged_unions_every_receipt_and_ignores_unreadable_ones(tmp_path: Path) -> None:
    """coverage is the union across machines; a truncated or non-JSON receipt costs its own ids, not the merge"""
    receipt.emit(str(tmp_path / "mac.json"), {"a", "shared"})
    receipt.emit(str(tmp_path / "cuda.json"), {"b", "shared"})
    (tmp_path / "half.json").write_text('{"covered": [', encoding="utf-8")
    assert receipt.merged(str(tmp_path)) == {"a", "b", "shared"}


def test_record_feeds_the_default_emit(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """emit() with no explicit set writes the ids this process recorded as it passed them"""
    monkeypatch.setattr(receipt, "_COVERED", set())
    receipt.record("gguf/tiny_qwen3-q4_0/mlx-packed")
    path = str(tmp_path / "r.json")
    receipt.emit(path)
    assert receipt.merged(path) == {"gguf/tiny_qwen3-q4_0/mlx-packed"}


def test_receipts_dir_defaults_beside_the_module(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("BTB_CERT_RECEIPTS", raising=False)
    assert receipt.receipts_dir() == os.path.join(os.path.dirname(receipt.__file__), "receipts")


def test_no_paths_reads_the_env_directory(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the interface CI codes against: BTB_CERT_RECEIPTS points at the collected artifacts and `merged()` takes
    no paths, so every caller agrees on where receipts are without repeating the path"""
    monkeypatch.setenv("BTB_CERT_RECEIPTS", str(tmp_path))
    receipt.emit(str(tmp_path / "ci.json"), {"safetensors/qwen3/cuda/greedy"})
    assert receipt.receipts_dir() == str(tmp_path)
    assert receipt.merged() == {"safetensors/qwen3/cuda/greedy"}


def test_cli_emit_needs_an_id(tmp_path: Path) -> None:
    """--emit with no --id used to write a valid, empty receipt and exit 0 - coverage claimed for nothing"""
    path = str(tmp_path / "empty.json")
    with pytest.raises(SystemExit) as e:
        receipt.main(["--emit", path])
    assert e.value.code == 2
    assert not os.path.exists(path)


def test_cli_emit_with_ids(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """a certifier that is not the pytest runner banks its cells through the CLI"""
    path = str(tmp_path / "box.json")
    assert receipt.main(["--emit", path, "--id", "safetensors/qwen3/cuda/greedy"]) == 0
    assert "emitted 1 id(s)" in capsys.readouterr().out
    assert receipt.merged(path) == {"safetensors/qwen3/cuda/greedy"}


def test_cli_merge_prints_the_union(tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """with no paths the CLI merges the receipts directory, the same default merged() takes"""
    monkeypatch.setenv("BTB_CERT_RECEIPTS", str(tmp_path))
    receipt.emit(str(tmp_path / "box.json"), {"safetensors/qwen3/cuda/greedy", "safetensors/qwen3/cuda/mtp"})
    assert receipt.main([]) == 0
    out = capsys.readouterr().out
    assert "merged coverage: 2 cells run+passed across 1 receipt(s)" in out
    assert "safetensors/qwen3/cuda/greedy" in out and "safetensors/qwen3/cuda/mtp" in out
