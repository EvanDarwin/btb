# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Coverage receipts (tests/cert/receipt.py): what emit() writes, what merged() unions back, and the default
directory CI and the matrices both code against. Torch-free - a receipt is JSON and nothing here loads a model."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from tests.cert import receipt

SHA = "af52169" + "0" * 33
IDS = {"safetensors/qwen3/mlx/greedy", "gguf/tiny_qwen3-q4_0/mlx-packed"}


def _stamped(tmp_path: Path, monkeypatch: MonkeyPatch, name: str = "r.json", ids: set[str] = IDS) -> str:
    """a receipt emitted after `begin()` on a clean checkout at SHA"""
    monkeypatch.setattr(receipt, "_START", None)  # restored at teardown, so the stamp stays in this test
    monkeypatch.setattr(receipt, "_git", lambda *a: SHA + "\n" if a[0] == "rev-parse" else "")
    receipt.begin()
    path = str(tmp_path / name)
    receipt.emit(path, ids)
    return path


def _comment(login: str, body: str, at: str, association: str = "NONE") -> dict[str, object]:
    return {"login": login, "association": association, "body": body, "at": at}


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


def test_a_run_receipt_names_its_commit_and_seconds(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """a run stamps the commit it began on, whether that tree had uncommitted changes, and the seconds it took"""
    doc = json.loads(Path(_stamped(tmp_path, monkeypatch)).read_text(encoding="utf-8"))
    assert (doc["sha"], doc["dirty"]) == (SHA, False)
    assert isinstance(doc["seconds"], int) and doc["seconds"] >= 0


def test_without_git_the_commit_is_ci_s(monkeypatch: MonkeyPatch) -> None:
    """a tree with no git (an sdist, a container without safe.directory) names CI's commit and no dirty state"""
    monkeypatch.setattr(receipt, "_START", None)
    monkeypatch.setattr(receipt, "_git", lambda *a: None)
    monkeypatch.setenv("GITHUB_SHA", SHA)
    receipt.begin()
    assert receipt._START is not None and (receipt._START.sha, receipt._START.dirty) == (SHA, None)


def test_a_pr_block_round_trips_folded_in_details(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the comment `--pr-block` prints is a <details> block naming the receipt, and CI reads it back whole, also
    with the CRLF line ends GitHub's editor saves"""
    block = receipt.pr_block(_stamped(tmp_path, monkeypatch))
    assert block.startswith("<details><summary>btb cert receipt: ") and block.rstrip().endswith("</details>")
    assert f"2 cells at {SHA[:12]} in 0 min</summary>" in block
    for body in (block, block.replace("\n", "\r\n")):
        (got,) = receipt.collect(str(tmp_path / "out"), "", "me", [_comment("me", f"cert, Mac:\n\n{body}", "t1")])
        assert set(got["covered"]) == IDS and got["sha"] == SHA and got["dirty"] is False
        assert (got["by"], got["source"]) == ("me", "pr-comment")
    assert receipt.merged(str(tmp_path / "out")) == IDS


def test_a_pr_block_past_one_comment_is_refused(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """a receipt whose block GitHub would truncate is refused at the source rather than pasted half"""
    ids = {hashlib.sha256(str(i).encode()).hexdigest() for i in range(4000)}
    with pytest.raises(ValueError, match="a GitHub comment holds"):
        receipt.pr_block(_stamped(tmp_path, monkeypatch, ids=ids))


def test_only_the_author_and_maintainers_hand_ci_a_receipt(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """a comment by anyone else - a contributor elsewhere in the repo, a stranger - is not read"""
    block = receipt.pr_block(_stamped(tmp_path, monkeypatch))
    comments = [
        _comment("stranger", block, "t1"),
        _comment("drive-by", block, "t2", "CONTRIBUTOR"),
        _comment("maint", block, "t3", "MEMBER"),
        _comment("me", block, "t4"),
    ]
    got = receipt.collect(str(tmp_path / "out"), "", "me", comments)
    assert sorted(r["by"] for r in got) == ["maint", "me"]


def test_each_contributor_s_latest_receipt_per_machine(tmp_path: Path) -> None:
    """a later receipt from the same contributor and machine replaces the earlier one (by comment time, the body
    first); other machines and other contributors' receipts merge in beside it"""

    def block(machine: str, ids: set[str]) -> str:
        doc = {"machine": machine, "when": "2026-10-24T20:53:58Z", "covered": sorted(ids)}
        return "```json btb-receipt\n" + json.dumps(doc) + "\n```"

    comments = [
        _comment("me", block("mac", {"newest"}), "2026-10-24T03:00:00Z"),
        _comment("maint", block("mac", {"theirs"}), "2026-10-24T02:00:00Z", "OWNER"),
        _comment("me", block("mac", {"older"}), "2026-10-24T01:00:00Z"),
        _comment("me", block("cuda", {"card"}), "2026-10-24T01:30:00Z"),
    ]
    got = receipt.collect(str(tmp_path / "out"), block("mac", {"old"}), "me", comments)
    held = {(r["by"], r["machine"]): set(r["covered"]) for r in got}
    assert held == {("me", "mac"): {"newest"}, ("maint", "mac"): {"theirs"}, ("me", "cuda"): {"card"}}
    assert receipt.merged(str(tmp_path / "out")) == {"newest", "theirs", "card"}


def test_a_plain_json_block_in_the_body_still_counts(tmp_path: Path) -> None:
    doc = {"machine": "box/x86_64/cuda", "when": "2026-10-24T20:53:58Z", "covered": ["safetensors/qwen3/cuda/greedy"]}
    body = "Closes the card gap.\n\n```json btb-receipt\n" + json.dumps(doc, indent=2) + "\n```\n"
    (got,) = receipt.collect(str(tmp_path), body, "me", [])
    assert (got["by"], got["source"], got["covered"]) == ("me", "pr-body", doc["covered"])
    assert "sha" not in got and "seconds" not in got


def test_a_gzip_bomb_stops_at_the_bound(tmp_path: Path) -> None:
    """a block that inflates past MAX_BYTES is dropped, decompressed no further than the bound"""
    packed = gzip.compress(b"[" + b" " * (receipt.MAX_BYTES * 4) + b"]", mtime=0)
    block = "```btb-receipt-gz\n" + base64.b64encode(packed).decode() + "\n```"
    assert len(packed) < 10_000
    assert receipt.collect(str(tmp_path), block, "me", []) == []


def test_a_pasted_receipt_is_made_safe(tmp_path: Path) -> None:
    """a pasted receipt's strings reach the job summary: markup, links and new lines are defanged, a malformed
    commit, dirty flag or duration is dropped, and a non-receipt or a bad block costs only itself"""
    doc = {
        "machine": "[x](https://evil.example)\n# hi",
        "when": "now",
        "covered": ["ok/id", "<img src=x>"],
        "sha": "not-a-sha",
        "dirty": "no",
        "seconds": True,
    }
    body = "\n".join(
        [
            "```json btb-receipt\n" + json.dumps(doc) + "\n```",
            "```json btb-receipt\n[1, 2]\n```",
            "```btb-receipt-gz\nnot base64!\n```",
            "```btb-receipt-gz\n" + base64.b64encode(b"not gzip").decode() + "\n```",
        ]
    )
    (got,) = receipt.collect(str(tmp_path), body, "me", [])
    assert got["machine"] == "?x??https?evil.example? ? hi"
    assert got["covered"] == ["ok/id", "?img src?x?"]
    assert not {"sha", "dirty", "seconds"} & set(got)


def test_api_comments_reads_paged_output() -> None:
    """`gh api --paginate` prints a JSON array a page, back to back; each comment keeps its author, standing,
    body and time"""
    page = [{"user": {"login": "me"}, "author_association": "OWNER", "body": "b", "updated_at": "t"}]
    text = json.dumps(page) + json.dumps(page) + "\n" + json.dumps(page, indent=2) + "\n"
    want = {"login": "me", "association": "OWNER", "body": "b", "at": "t"}
    assert receipt.api_comments(text) == [want] * 3
    assert receipt.api_comments("") == []


def test_cli_collect_says_what_it_kept(tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """the CI step: the body and author from the environment, the comments from `gh api`'s output, a line per
    receipt naming who, the machine, the cells, the commit (flagged when not the PR's head) and the minutes"""
    block = receipt.pr_block(_stamped(tmp_path, monkeypatch))
    page = [{"user": {"login": "me"}, "author_association": "NONE", "body": block, "updated_at": "t"}]
    comments = tmp_path / "comments.json"
    comments.write_text(json.dumps(page), encoding="utf-8")
    monkeypatch.setenv("PR_BODY", "")
    monkeypatch.setenv("PR_AUTHOR", "me")
    monkeypatch.setenv("PR_HEAD", "f" * 40)
    out = str(tmp_path / "out")
    assert receipt.main(["--collect", out, "--comments", str(comments)]) == 0
    printed = capsys.readouterr().out
    assert f"2 cells at {SHA[:12]} (not the PR head), 0 min (pr-comment)" in printed
    assert "pasted receipts kept: 1" in printed
    assert receipt.merged(out) == IDS
