# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The examples under examples/ run as they are shipped: each `main()` on the tiny fixtures (the CPU, seconds),
its result checked against what its docstring claims. A reader who copies one gets what it says."""

import importlib.util
import os
from pathlib import Path

import pytest

from btb.kinds import Json
from tests.helpers import MB, checkout, fixture, need_mlx


def _run(name: str, *args: str) -> Json:
    """the example's main() on the fixture, on the CPU: the record its docstring promises"""
    path = checkout("examples", name + ".py")
    fx = fixture("tiny_qwen3")
    if not os.path.exists(os.path.join(fx, "tokenizer.json")):
        pytest.skip("the fixture's tokenizer is not here: these tests run from a checkout")
    spec = importlib.util.spec_from_file_location("example_" + name, path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.main(["--model", fx, "--device", "cpu", *args])


def test_ask_answers_and_streams() -> None:
    r = _run("ask", "--new", "8")
    assert isinstance(r["text"], str) and r["text"]
    assert 1 <= len(r["streamed"]) <= 8 and r["stats"]["cap"] == 8


def test_chat_reuses_the_cache_from_the_second_turn() -> None:
    r = _run("chat", "--new", "8")
    assert len(r["history"]) == 6
    assert r["reused"][0] == 0 and all(n > 0 for n in r["reused"][1:]), r["reused"]


def test_plan_first_prices_and_refuses_by_name() -> None:
    r = _run("plan_first")
    pl = r["plan"]
    assert pl.device == "cpu" and set(pl.host) == set(range(4)) and not pl.resident and not pl.cold
    assert pl.budget is not None and pl.budget.spendable > 0 and "host" in str(pl)
    assert r["refused"] and "REFUSED" in r["refused"]
    assert r["host"] == sorted(pl.host) and r["resident"] == []


def test_memory_budget_grants_what_fits_and_refuses_what_does_not() -> None:
    r = _run("memory_budget")
    assert r["budget"].floor == 2**30, "the reserve named is the floor"
    assert r["granted"]["scratch@cpu"] == 16 * MB
    assert r["refused"] and "REFUSED my impossible buffer" in r["refused"]
    assert r["before"].total > 0 and r["report"]["growth"]["estimate_gb"] > 0


def test_coexist_holds_a_reservation_for_the_block() -> None:
    r = _run("coexist", "--new", "4", "--mb", "16")
    assert r["held"] == 16 * MB and r["after"] == 0
    assert r["ledger"]["scratch@cpu"] == 16 * MB and r["text"]


def test_shed_and_regrow_keeps_the_answer() -> None:
    r = _run("shed_and_regrow", "--new", "8", "--model", fixture("tiny_qwen3-pack12"))
    assert r["moves"], "the fixture is a 12-bit model, so the host move runs"
    kind, shed, same_during, back, same_after = r["moves"][0]
    assert kind == "ram" and shed == back == 3 and same_during and same_after


def test_custom_loop_owns_the_loop() -> None:
    r = _run("custom_loop", "--new", "8")
    assert 1 <= len(r["tokens"]) <= 8 and r["tokens"][0] in r["allowed"]
    assert r["positions"] == len(r["prompt"]) + len(r["tokens"]) - 1, "the last token was never fed back"
    assert _run("custom_loop", "--new", "8")["tokens"] == r["tokens"], "the seed makes the sampling repeatable"


def test_batch_answers_every_prompt_in_one_epoch_off_the_card() -> None:
    r = _run("batch", "--new", "6")
    assert len(r["answers"]) == 3 and all(isinstance(x, str) for x in r["answers"])
    assert r["batch"] == 3


def test_spans_are_verified_to_the_greedy_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """the banked drafts never change the answer; they are accepted only where the answer has variety (a
    random fixture answers a byte-range prompt with one token, and the proposer drops a constant chain by
    design: test_engine_units certifies the acceptance on chosen ids). The host's pass-cost curve is pinned flat:
    timed at warm-up on a busy machine it prices a wide pass past a step's, the budget closes to one row and
    nothing is drafted - whether the drafts run is then the machine's load, not the example's claim"""
    from btb.engine import StreamedTextModel

    def flat(self: StreamedTextModel, ids: object, t_max: int | None = None) -> int:
        n = max(1, min(int(t_max or self._spec_full()), 16))
        self._host_cost = dict.fromkeys(range(1, n + 1), 1.0)
        return n

    monkeypatch.setattr(StreamedTextModel, "host_warm", flat)
    r = _run("spans", "--new", "24")
    assert r["identical"] and r["stats"]["forwards"] >= 1
    if len(set(r["tokens"])) > 1:
        assert r["stats"]["accepted"] > 0 and r["stats"]["forwards"] < len(r["tokens"]), "the drafts saved passes"


def test_packed_store_round_trips_the_tokens(tmp_path: Path) -> None:
    r = _run("packed_store", "--new", "8", "--out", str(tmp_path / "tiny_qwen3-pack12"))
    assert r["identical"] and r["packed_bytes"] > 0 and r["format"]["format"] == "pack12"
    assert os.path.exists(os.path.join(r["out"], "model.safetensors.index.json"))


def test_foreign_model_gets_the_scheduler_over_a_transformers_model() -> None:
    r = _run("foreign_model", "--new", "64")
    assert r["row_bytes"] > 0 and r["batch"] == 8, "off the card the epoch is every pending row"
    assert r["refused"] and "REFUSED my impossible cache" in r["refused"]
    assert r["granted"]["kv@cpu"] == 4 * r["row_bytes"]


def test_foreign_model_prices_unified_memory_on_mlx() -> None:
    """--device mlx: the cache priced against the RAM Apple silicon's GPU shares, the grants on the host"""
    need_mlx()
    path = checkout("examples", "foreign_model.py")
    spec = importlib.util.spec_from_file_location("example_foreign_model_mlx", path)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    r = m.main(["--model", fixture("tiny_qwen3"), "--device", "mlx", "--new", "64"])
    assert r["row_bytes"] > 0 and r["batch"] == 8 and "REFUSED my impossible cache" in r["refused"]
    assert r["granted"]["kv@cpu"] == 4 * r["row_bytes"]


def test_openai_server_in_process() -> None:
    r = _run("openai_server", "--new", "6")
    assert r["model"] == "tiny_qwen3" and isinstance(r["text"], str)
    assert r["usage"] and r["usage"]["completion_tokens"] >= 1 and r["url"].startswith("http://127.0.0.1:")


def test_host_monitor_reads_the_machine_without_an_engine() -> None:
    r = _run("host_monitor")
    assert r["free"] > 0 and r["budget"].total > r["budget"].available > 0
    assert isinstance(r["pressure"], dict)


def test_session_loop_rewinds_and_goes_on() -> None:
    r = _run("session_loop", "--new", "6")
    assert r["same"], "a feed after a rewind gave other logits"
    assert r["tokens"] == r["plain"], "the session decoded otherwise than a plain decode of the prompt"
    assert r["session"] == r["prompt"] + r["tokens"] and r["pending"] == r["tokens"][-1]
    assert r["tap"] == (1, r["hidden"])


def test_hooks_see_every_token() -> None:
    r = _run("hooks", "--new", "8")
    assert r["banned"] not in r["tokens"], "the processor's ban was drawn"
    assert len(r["logprobs"]) == len(r["tokens"]) and all(len(lp.top) == 2 for lp in r["logprobs"])
    assert r["hidden"][0] == len(r["tokens"]) and r["passed"] == len(r["tokens"])
    assert r["report"].tags


def test_a_beam_of_one_is_the_greedy_answer_and_a_wider_one_scores_as_well() -> None:
    one = _run("beam", "--width", "1", "--new", "6")
    three = _run("beam", "--width", "3", "--new", "6")
    assert one["tokens"] == one["greedy"]
    assert three["score"] >= one["score"] - 1e-4
    assert three["on"] == three["fed_on"], "the kept beam's session went on otherwise than one fed its tokens"


def test_batch_sessions_decode_each_as_its_own_and_write_back() -> None:
    r = _run("batch_sessions", "--new", "3")
    a, b, c = r["sessions"]
    assert r["refused"], "a session in the batch took a feed"
    for s, p, alone in ((a, r["prompts"][0], r["alone"][0]), (b, r["prompts"][1], r["alone"][1])):
        assert s[len(p) :] == alone, "a row drew otherwise than its session alone"
    assert c == r["prompts"][2] + r["second"][r["joined"]]


def test_lend_counts_the_room_and_refuses_by_name() -> None:
    r = _run("lend")
    assert r["held"] == 6 * MB and r["after"] == 0
    assert r["mine"] == (256, 1024) and r["refused"] and "MiB" in r["refused"]
