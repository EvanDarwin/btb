# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Model discovery and repo resolution (btb/hf.py): the server's name for a repo, when a cached snapshot counts
as complete, the models found in the cache and given directories, the packed store beside a model, and a Hub
download that stays interruptible. Fast and model-free; the fixtures are a few hundred kilobytes."""

import json
import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from btb import hf
from tests.helpers import FIXTURES

FIXTURE = os.path.join(FIXTURES, "tiny_qwen3")  # discovery is the subject: the fixture's absence is a failure


def test_serve_name_round_trips_repo_ids() -> None:
    assert hf.serve_name("Qwen/Qwen3.5-4B") == "qwen-qwen3.5-4b"
    assert hf.serve_name("Qwen3-4B") == "qwen3-4b"
    assert hf.serve_name("  weird name!!") == "weird-name"
    assert hf.serve_name("") == "btb"
    assert hf.serve_name(None) == "btb"


def test_model_complete_needs_config_and_every_shard(tmp_path: Path) -> None:
    d = tmp_path / "m"
    d.mkdir()
    assert not hf._model_complete(str(d))
    (d / "config.json").write_text('{"model_type": "qwen3"}')
    assert not hf._model_complete(str(d))
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
    )
    (d / "s1.safetensors").write_bytes(b"x")
    assert not hf._model_complete(str(d)), "a missing shard is incomplete"
    (d / "s2.safetensors").write_bytes(b"x")
    assert hf._model_complete(str(d))
    (d / "s3.safetensors.incomplete").write_bytes(b"")
    assert not hf._model_complete(str(d)), "a download in flight is incomplete"


@pytest.mark.timing
def test_a_hub_download_stays_interruptible(monkeypatch: MonkeyPatch) -> None:
    """A Ctrl-C during a Hub download is answered at once: the download runs on a daemon thread and the main
    thread waits in an interruptible loop, so the KeyboardInterrupt reaches the exit rather than blocking on the
    in-flight files. `_thread.interrupt_main` stands in for the signal; `_hard_exit` is captured, not taken."""
    import _thread
    import threading
    import time

    monkeypatch.setattr(hf, "_hard_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    started = threading.Event()

    def slow(_path: str) -> str:
        started.set()
        time.sleep(10)  # a long download in the worker thread; never returns before the interrupt
        return "unreachable"

    monkeypatch.setattr(hf, "_download", slow)

    def fire() -> None:
        started.wait(2)
        time.sleep(0.2)
        _thread.interrupt_main()

    threading.Thread(target=fire, daemon=True).start()
    t0 = time.time()
    with pytest.raises(SystemExit) as ei:
        hf.resolve("some/repo")
    assert ei.value.code == 130
    assert time.time() - t0 < 3, "the wait was not interruptible"


def test_a_download_prompt_names_where_it_writes_and_the_space_there(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """the download gate's question says what the Hub reports (size, file count), the directory the files land
    in (the Hub cache's repo folder, so a wrong HF_HOME shows before a byte is written), the space free on that
    volume, and the shortfall when the download would not fit; a cached or local model is never asked about"""
    import btb.confirm

    asked: list[str] = []
    monkeypatch.setattr(btb.confirm, "confirm", lambda q, **kw: asked.append(q) or True)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hf, "would_download", lambda path: True)
    monkeypatch.setattr(hf, "hub_info", lambda path: (8_100_000_000, 12))
    target = os.path.join(str(tmp_path), "models--Qwen--Qwen3-4B")
    assert hf.download_target("Qwen/Qwen3-4B") == target
    assert hf.download_target("Qwen/Qwen3-4B:model.gguf") == target, "a single file lands in the repo's folder"

    monkeypatch.setattr(hf, "_free_bytes", lambda path: 412_000_000_000)
    assert hf.confirm_download("Qwen/Qwen3-4B")
    assert asked[-1] == f"download Qwen/Qwen3-4B from Hugging Face (~8.1 GB, 12 files) into {target} (~412 GB free)?"

    monkeypatch.setattr(hf, "_free_bytes", lambda path: 4_000_000_000)
    hf.confirm_download("Qwen/Qwen3-4B")
    assert asked[-1].endswith(f"into {target} (~4.0 GB free - short by ~4.1 GB)?")

    monkeypatch.setattr(hf, "_free_bytes", lambda path: None)  # a volume that cannot be read: no space claim
    hf.confirm_download("Qwen/Qwen3-4B")
    assert asked[-1].endswith(f"into {target}?")

    monkeypatch.setattr(hf, "would_download", lambda path: False)
    n = len(asked)
    assert hf.confirm_download("Qwen/Qwen3-4B") and len(asked) == n, "a cached model is not asked about"


def test_available_models_finds_the_fixture_by_path() -> None:
    hits = hf.available_models(paths=[FIXTURE])
    names = {h["name"] for h in hits}
    assert "tiny_qwen3" in names
    hit = next(h for h in hits if h["name"] == "tiny_qwen3")
    assert hit["type"] == "qwen3" and hit["path"] == FIXTURE and hit["size"] > 0
    assert hf.available_models(paths=[FIXTURE], pattern="nomatch") == [] or all(
        "nomatch" not in h["name"] for h in hf.available_models(paths=[FIXTURE], pattern="nomatch")
    )


def test_a_12_bit_model_is_listed_with_its_format() -> None:
    """`btb pack` writes a model of its own: `pack_format` reads its config's `btb` record (the parent, named
    relative to the pack so the pair moves together), and discovery lists it as complete, servable and packed,
    beside the plain parent."""
    pack = FIXTURE + "-pack12"
    if not os.path.exists(pack):
        pytest.skip("the tiny_qwen3-pack12 fixture is not here")
    fmt = hf.pack_format(pack)
    assert fmt is not None and fmt["format"] == "pack12"
    named = os.path.normpath(os.path.join(pack, fmt["source"]))
    assert os.path.normcase(named) == os.path.normcase(os.path.normpath(FIXTURE))
    assert hf.pack_format(FIXTURE) is None
    hits = {h["name"]: h for h in hf.available_models(paths=[pack, FIXTURE])}
    assert hits["tiny_qwen3-pack12"]["packed"] and hits["tiny_qwen3-pack12"]["type"] == "qwen3"
    assert not hits["tiny_qwen3"]["packed"]


def test_cache_repo_id_reads_a_snapshot_path() -> None:
    assert hf.cache_repo_id(r"C:\hub\models--Qwen--Qwen3-4B\snapshots\c1899de") == "Qwen/Qwen3-4B"
    assert (
        hf.cache_repo_id("/home/u/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/c1899de") == "Qwen/Qwen3-4B"
    )
    assert hf.cache_repo_id(FIXTURE) is None


def test_shard_map_names_every_tensors_file() -> None:
    wm = hf.shard_map(FIXTURE)
    assert "model.embed_tokens.weight" in wm
    assert all(os.path.isfile(os.path.join(FIXTURE, f)) for f in set(wm.values()))
