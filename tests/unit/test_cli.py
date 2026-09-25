# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The command line and its individual commands: how `run` finds its prompt (a pipe, -p, a --file, or a
sample), and the window the FSL notice it prints runs for. Model-free but for one end-to-end `run --json` over
a piped prompt on the tiny fixture; everything here runs in seconds. The server `serve` / `ollama` run is in
test_serve.py; the out-of-memory notice and the --profile bundle are the feedback module's, in test_feedback.py."""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from tests.helpers import fixture


class _Stdin:
    """a stand-in for sys.stdin: a piped stream (isatty False) carrying `data`, or an interactive one (isatty True)"""

    def __init__(self, data: str, tty: bool) -> None:
        self.data, self.tty = data, tty

    def isatty(self) -> bool:
        return self.tty

    def read(self) -> str:
        return self.data


# --- run: the prompt sources --------------------------------------------------------------------------------


def test_run_reads_the_prompt_from_a_pipe_explicit_or_a_sample(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """`btb run` takes its one prompt from -p when given, else from a pipe on stdin, else a sample; a JSONL
    --file overrides all. Platform-independent: stdin is faked, never shell-piped."""
    from btb.cli import DEFAULT_PROMPT, _run_prompts

    def prompts(prompt: str | None = None, file: str | None = None, data: str = "", tty: bool = True) -> list[str]:
        monkeypatch.setattr(sys, "stdin", _Stdin(data, tty))
        return _run_prompts(argparse.Namespace(prompt=prompt, file=file))

    assert prompts(data="What is 2+2?\n", tty=False) == ["What is 2+2?"], "a piped prompt is taken and trimmed"
    assert prompts(prompt="explicit", data="piped", tty=False) == ["explicit"], "-p wins over a pipe"
    assert prompts(data="junk", tty=True) == [DEFAULT_PROMPT], "an interactive stdin falls back to the sample"
    assert prompts(data="  \n ", tty=False) == [DEFAULT_PROMPT], "an empty pipe falls back to the sample"
    f = tmp_path / "p.jsonl"
    f.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    assert prompts(prompt="x", file=str(f), data="x", tty=False) == ["a", "b"], "--file overrides -p and the pipe"


def test_run_json_answers_a_prompt_piped_on_stdin(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    """the machine-readable path end to end: a prompt piped on stdin, `btb run --json` writes exactly one JSON
    record whose `prompt` is the piped text (trimmed) and nothing else on stdout."""
    from btb.cli import main

    fx = fixture("tiny_qwen3")
    if not os.path.exists(os.path.join(fx, "tokenizer.json")):
        pytest.skip("the tiny_qwen3 fixture has no tokenizer: these tests run from a checkout")
    monkeypatch.setattr(sys, "stdin", _Stdin("  what is 2 plus 2?  \n", tty=False))
    rc = main(["run", fx, "-d", "cpu", "--new", "2", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out.strip())  # one record, nothing else on stdout
    assert rec["prompt"] == "what is 2 plus 2?" and isinstance(rec["answer"], str)


# --- the FSL notice -----------------------------------------------------------------------------------------


def test_the_fsl_window_is_two_years_from_the_build(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    import datetime

    from btb import fsl

    monkeypatch.setattr(fsl, "FRIEND_FILE", str(tmp_path / "friend"))
    today = datetime.date.today()
    for built, open_ in ((None, True), (today, True), (today.replace(year=today.year - 3), False)):
        monkeypatch.setattr(fsl, "build_date", lambda b=built: b)
        assert fsl.restricted() is open_, built
    (tmp_path / "friend").write_text("")
    monkeypatch.setattr(fsl, "build_date", lambda: today)
    assert fsl.restricted() is False  # a friend is never inside it
