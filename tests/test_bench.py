# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The benchmark tool (bench/lib, bench/matrix.py, bench/compare.py): the matrix's planning and resume, the
comparison-environment check, every rival's one signature, the timing loop, the table, the machine readers,
and the two scripts end to end on a fake tool. A separate tool from the engine; nothing here loads a model.
Everything runs in seconds and skips cleanly when run from a wheel without the bench sources."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import IO, TYPE_CHECKING, ClassVar, Protocol, cast

import pytest
import torch
from pytest import CaptureFixture, MonkeyPatch

from btb.kinds import Json
from tests.helpers import ROOT, checkout

if TYPE_CHECKING:
    from lib.tools import BenchTool


def _bench(name: str) -> ModuleType:
    """A module of bench/lib, imported from the checkout with bench/ first on the path, as the scripts have it."""
    checkout("bench", "lib", name + ".py")
    bench = os.path.join(ROOT, "bench")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    return importlib.import_module("lib." + name)


def _script(name: str) -> ModuleType:
    """bench/matrix.py or bench/compare.py as a module, the way `python bench/<name>.py` loads it."""
    _bench("records")
    spec = importlib.util.spec_from_file_location(name, checkout("bench", name + ".py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Clock:
    """A fake clock the timing loop and a fake tool share: perf_counter and time both read it, the tool
    advances it as it 'generates', so first-token and rate come out exact."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, s: float) -> float:
        self.now += s
        return self.now


class _FakeTool(Protocol):
    """the fake tool's class: BenchTool's constructor, with every instance it made and the ones closed since"""

    made: list[BenchTool]
    closed: list[BenchTool]

    def __call__(self, path: str, log: Callable[[str], object] | None = None, **opts: object) -> BenchTool: ...


def _fake_tool(tools: ModuleType, clock: _Clock, first: float = 0.5, per: float = 0.1) -> _FakeTool:
    """A Tool that takes `first` seconds to the first token and `per` per token after; a prompt starting
    with EMPTY yields nothing."""
    if not TYPE_CHECKING:
        BenchTool = tools.BenchTool  # the module is loaded by hand: its class is bound at the call, not the import

    class Fake(BenchTool):
        name = "fake"
        module = "fake_mod"
        regimes = (("cpu", ("--device", "cpu")),)
        made: ClassVar[list[BenchTool]] = []
        closed: ClassVar[list[BenchTool]] = []

        def load(self) -> None:
            Fake.made.append(self)

        def prompt(self, text: str) -> str:
            return text.upper()

        def prompt_len(self, prompt: str) -> int:
            return len(prompt)

        def stream(self, prompt: str, n: int) -> Iterator[float]:
            if prompt.startswith("EMPTY"):
                return
            yield clock.tick(first)
            for _ in range(n - 1):
                yield clock.tick(per)

        def device(self) -> str:
            return "cpu"

        def peak_vram_gb(self) -> float:
            return 1.5

        def close(self) -> None:
            Fake.closed.append(self)

    return Fake


def _timed(monkeypatch: MonkeyPatch) -> tuple[ModuleType, _Clock]:
    """The tools module on the fake clock, its RSS reader fixed at 3 GB."""
    tools = _bench("tools")
    clock = _Clock()
    monkeypatch.setattr(tools.time, "perf_counter", clock)
    monkeypatch.setattr(tools.time, "time", clock)
    monkeypatch.setattr(tools, "peak_rss_bytes", lambda: 3 * 2**30)
    return tools, clock


def test_matrix_gives_a_comparison_tool_a_cell_per_device_regime(monkeypatch: MonkeyPatch) -> None:
    """a rival runs in every device regime btb does, through its own flags, never skipped for its size (both
    rivals stream); AirLLM's card regime is a planned skip only where there is no CUDA; gpt-oss on AirLLM
    carries the MXFP4 note in each regime"""
    plan = _bench("plan")
    big = {"name": "big", "repo": "x/big", "path": "/nowhere", "type": "qwen3", "size": 61 * 2**30, "packed": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cells = plan.plan_cells([big], ["cpu", "mlx-lm", "airllm", "llama-cpp"], False)
    by = {c["config"]: c for c in cells}
    assert set(by) == {"cpu", "mlx-lm", "airllm-gpu", "airllm-cpu", "llama-cpp-gpu", "llama-cpp-cpu"}
    assert all(c["skip"] is None for c in cells)
    assert by["airllm-gpu"]["args"] == ["--tool", "airllm", "--device", "cuda"]
    assert by["airllm-cpu"]["args"] == ["--tool", "airllm", "--device", "cpu"]
    assert by["llama-cpp-cpu"]["args"] == ["--tool", "llama-cpp", "--n-gpu-layers", "0"]
    assert by["mlx-lm"]["tool"] == "mlx-lm" and not by["cpu"].get("tool")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    by = {c["config"]: c for c in plan.plan_cells([big], ["airllm"], False)}
    assert "no CUDA" in by["airllm-gpu"]["skip"] and by["airllm-cpu"]["skip"] is None
    gpt = dict(big, type="gpt_oss")
    cells = plan.plan_cells([gpt], ["airllm"], False)
    assert len(cells) == 2 and all("MXFP4" in c["skip"] for c in cells), (
        "gpt-oss on AirLLM is skipped on its own ground"
    )
    # btb's own cells: the CPU is float32, --fp32 adds a float32 row to each bf16 cell, cpu+mlx is a planned skip
    cells = plan.plan_cells([big], ["cpu", "gpu", "cpu+mlx"], True)
    assert [(c["config"], c["dtype"], c["args"]) for c in cells] == [
        ("cpu", "fp32", ["--device", "cpu"]),
        ("gpu", "bf16", ["--device", "cuda", "--cpu-layers", "0"]),
        ("gpu", "fp32", ["--device", "cuda", "--cpu-layers", "0", "--fp32", "1"]),
        ("cpu+mlx", "bf16", []),
    ]
    assert "--cpu-layers" in cells[-1]["skip"]


def test_matrix_resume_reuses_only_finished_cells_of_the_same_lengths() -> None:
    plan = _bench("plan")
    e = {"name": "m", "repo": "x/m", "path": "/nowhere", "type": "qwen3", "size": 1, "packed": False}
    cells = plan.plan_cells([e], ["cpu", "mlx"], False)
    prev = {
        "new": [64, 256],
        "cells": [
            {"model": "m", "config": "cpu", "dtype": "fp32", "status": "ok", "seconds": 5, "cells": [{"new": 64}]},
            {"model": "m", "config": "mlx", "dtype": "bf16", "status": "failed", "seconds": 1},
        ],
    }
    assert plan.resume_cells(cells, prev, [64, 256], "prev.json") == 1
    cpu = next(c for c in cells if c["config"] == "cpu")
    assert cpu.get("resumed") == "prev.json" and cpu["status"] == "ok" and cpu["args"] == ["--device", "cpu"]
    assert not next(c for c in cells if c["config"] == "mlx").get("resumed")
    cells = plan.plan_cells([e], ["cpu"], False)
    assert plan.resume_cells(cells, prev, [64, 256, 1024], "prev.json") == 0, "other answer lengths are another run"


def test_matrix_checks_the_comparison_environment_against_its_requirements(tmp_path: Path) -> None:
    """bench/requirements.txt against what an interpreter holds: a missing package, another version, a marker
    that does not hold here skipped, a local version tail (+cu128) not a difference, no interpreter at all"""
    env = _bench("env")
    req = tmp_path / "requirements.txt"
    req.write_text(
        "# the rivals\ntorch==2.11.0\nairllm==4.0.0\nllama-cpp-python==0.3.35\n"
        'mlx-lm; sys_platform == "darwin"\nprotobuf\n',
        encoding="utf-8",
    )
    py = tmp_path / "python"
    py.write_text("", encoding="utf-8")
    have = {"torch": "2.11.0+cu128", "airllm": "4.0.0", "llama_cpp_python": "0.3.34", "protobuf": "7.36.1"}
    got = env.compare_env_mismatch(str(py), have, requirements=str(req))
    assert (
        got == ["llama_cpp_python 0.3.34 (wants 0.3.35)"]
        if sys.platform != "darwin"
        else got[:1] == ["llama_cpp_python 0.3.34 (wants 0.3.35)"]
    )
    assert env.compare_env_mismatch(str(py), {**have, "llama_cpp_python": "0.3.35"}, requirements=str(req)) == (
        [] if sys.platform != "darwin" else ["mlx_lm missing"]
    )
    assert env.compare_env_mismatch(str(py), {}, requirements=str(req)) == [
        n + " missing" for n, _ in env.requirement_lines(str(req))
    ]
    assert env.compare_env_mismatch(str(tmp_path / "nowhere"), have, requirements=str(req)) == ["no interpreter"]
    here = sys.platform
    assert env.marker_holds(f'sys_platform == "{here}"') and not env.marker_holds(f'sys_platform != "{here}"')
    assert env.marker_holds("python_version >= '3'"), "an unknown marker is taken as holding"


def test_every_tool_speaks_the_one_signature_and_says_what_the_matrix_plans_for_it() -> None:
    """each rival class fills the whole signature itself, names the module its environment is probed for and
    its device regimes with the flags that force them; the matrix's configuration list ends with the same
    table; close() before load() is a no-op for every one"""
    tools = _bench("tools")
    plan = _bench("plan")
    for name, cls in tools.TOOLS.items():
        assert cls.name == name and cls.module and cls.regimes, name
        for m in ("load", "prompt", "prompt_len", "stream", "device"):
            assert getattr(cls, m) is not getattr(tools.BenchTool, m), f"{name} leaves {m} to the base"
        for regime, flags in cls.regimes:
            assert regime in ("gpu", "cpu", "mlx") and isinstance(flags, tuple), name
        cls("/nowhere", lambda s: None).close()
    assert plan.CONFIGS[-len(tools.TOOLS) :] == tuple(tools.TOOLS)
    assert tools.TOOLS["llama-cpp"].cannot("gpu", "qwen3", cuda=False) is None
    assert "CUDA" in tools.TOOLS["airllm"].cannot("gpu", "qwen3", cuda=False)
    assert tools.TOOLS["airllm"].cannot("cpu", "qwen3", cuda=False) is None
    assert "MXFP4" in tools.TOOLS["airllm"].cannot("cpu", "gpt_oss", cuda=True)


def test_the_timing_loop_measures_first_token_and_rate_and_keeps_the_budget(monkeypatch: MonkeyPatch) -> None:
    """per answer length, per prompt: the first token's delay from the prompt, then the rate over the rest;
    a prompt that yields nothing is left out, a length with no prompt left has no cell; a length that starts
    past the budget is left out with a word, and a prompt that ends past it ends its length"""
    tools, clock = _timed(monkeypatch)
    said: list[str] = []
    tool = _fake_tool(tools, clock, first=0.5, per=0.1)("/m", said.append)
    cells = tools.bench(tool, ["a", "b"], [4, 8], budget=1e9)
    assert [c["new"] for c in cells] == [4, 8]
    for c in cells:
        assert c["first_s"] == pytest.approx(0.5) and c["greedy_s_tok"] == pytest.approx(0.1)
        assert (c["peak_ram_gb"], c["peak_vram_gb"], c["spec_s_tok"], c["tokens_per_pass"]) == (3.0, 1.5, None, 1.0)
    assert said.count("fake new=4: prompt 1, 4 tokens, first 0.50s, then 0.100 s/token") == 2
    # a prompt of 4 tokens costs 0.8s and one of 8 costs 1.2s: with 2.0s the 4s finish, the 8s stop after one
    # prompt; with 1.0s the 4s stop after their second prompt and the 8s never start
    cells = tools.bench(tool, ["a", "b"], [4, 8], budget=2.0)
    assert [c["new"] for c in cells] == [4, 8]
    said.clear()
    cells = tools.bench(tool, ["a", "b"], [4, 8], budget=1.0)
    assert [c["new"] for c in cells] == [4] and said[-1] == "budget of 1s spent; 8 and beyond left out"
    # an empty stream
    cells = tools.bench(tool, ["EMPTY", "a"], [4], budget=1e9)
    assert len(cells) == 1 and cells[0]["first_s"] == pytest.approx(0.5)
    assert tools.bench(tool, ["EMPTY"], [4, 8], budget=1e9) == []


def test_compare_reads_the_questions_file_and_templates_a_prompt(tmp_path: Path) -> None:
    tools = _bench("tools")
    q = tmp_path / "q.jsonl"
    q.write_text('{"prompt": "one"}\n\n{"prompt": "two"}\n{"prompt": "three"}\n', encoding="utf-8")
    assert tools.prompts(str(q)) == ["one", "two", "three"]
    assert tools.prompts(str(q), "") == ["one", "two", "three"]
    assert tools.prompts(str(q), "2,0") == ["three", "one"]

    class Tok:
        def apply_chat_template(
            self, msgs: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool, enable_thinking: bool
        ) -> str:
            assert (tokenize, add_generation_prompt) == (False, True)
            return f"<{msgs[0]['content']}|think={enable_thinking}>"

    class OldTok:
        def apply_chat_template(self, msgs: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> str:
            return f"<{msgs[0]['content']}>"

    assert tools.template(Tok(), "hi") == "<hi|think=False>"
    assert tools.template(OldTok(), "hi") == "<hi>", "a template without the thinking switch is used as it is"


def test_compare_main_runs_a_tool_and_appends_its_record(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """the script end to end on a fake tool: the rows picked, every length timed, the tool closed, and the
    record the matrix reads back appended as one JSON line"""
    import json

    tools, clock = _timed(monkeypatch)
    compare = _script("compare")
    fake = _fake_tool(tools, clock)
    monkeypatch.setitem(tools.TOOLS, "fake", fake)
    q = tmp_path / "q.jsonl"
    q.write_text('{"prompt": "a"}\n{"prompt": "b"}\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    argv = ["/model", "--tool", "fake", "--prompts", str(q), "--rows", "1", "--new", "4,8", "--out", str(out)]
    assert compare.main([*argv, "--label", "L"]) == 0
    assert fake.made[-1] in fake.closed and fake.made[-1].opts["max_new"] == 8
    rec = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert (rec["label"], rec["tool"], rec["device"], rec["model"], rec["path"]) == (
        "L",
        "fake",
        "cpu",
        "model",
        "/model",
    )
    assert [c["new"] for c in rec["cells"]] == [4, 8] and rec["cells"][0]["first_s"] == pytest.approx(0.5)
    assert compare.main(argv) == 0
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2, "a record is appended, never overwritten"


def test_run_cell_reads_the_record_back_and_reports_a_failure_or_a_kill(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """one cell in a (fake) child: the command line of a btb cell and of a tool cell, the record read back,
    a child with no record failed with its exit code and the log's tail, the guard's kill with its reason,
    the timeout"""
    import json

    run = _bench("run")
    monkeypatch.setattr(run.time, "sleep", lambda s: None)
    state = {"level": 90.0, "swap_gb": 0.0, "disk_gb": 100.0}
    monkeypatch.setattr(run, "memory_state", lambda: dict(state))
    procs = []

    class Proc:
        pid = 4242
        returncode: int | None

        def __init__(self, argv: list[str], record: Json | None, rc: int, polls: int, **kw: object) -> None:
            self.argv, self.record, self.rc, self.polls, self.kw = argv, record, rc, polls, kw
            self.returncode = None
            self.killed = False

        def poll(self) -> int | None:
            if self.polls > 0:
                self.polls -= 1
                return None
            if self.record is not None:
                out = self.argv[self.argv.index("--out") + 1]
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(self.record) + "\n")
            cast("IO[str]", self.kw["stdout"]).write("some output\nthe last line\n")
            self.returncode = self.rc
            return self.rc

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        def wait(self) -> int | None:
            return self.returncode

    def popen(record: Json | None = None, rc: int = 0, polls: int = 1) -> None:
        def make(argv: list[str], **kw: object) -> Proc:
            procs.append(Proc(argv, record, rc, polls, **kw))
            return procs[-1]

        monkeypatch.setattr(run.subprocess, "Popen", make)

    monkeypatch.setattr(run.os, "killpg", lambda pid, sig: procs[-1].kill(), raising=False)

    def cell(**kw: object) -> Json:
        base = {"model": "m", "config": "cpu", "dtype": "fp32", "path": "/m", "args": ["--device", "cpu"], "tool": None}
        return {**base, "skip": None, **kw}

    popen(record={"device": "cpu", "cells": [{"new": 64, "greedy_s_tok": 0.5}], "report": {"placement": {}}})
    c = run.run_cell(cell(), "q.jsonl", "64", "0", str(tmp_path), timeout=100)
    assert (
        c["status"] == "ok" and c["device"] == "cpu" and c["cells"][0]["new"] == 64 and c["report"] == {"placement": {}}
    )
    argv = procs[-1].argv
    assert argv[:5] == [sys.executable, "-m", "btb.cli", "bench", "/m"]
    assert argv[argv.index("--label") + 1] == "m cpu fp32" and argv[-4:] == ["--device", "cpu", "--rows", "0"]
    assert c["memory"] == {"min_level": 90.0, "max_swap_gb": 0.0, "min_disk_gb": 100.0}
    assert os.path.basename(c["log"]) == "m-cpu-fp32.log"
    assert open(c["log"], encoding="utf-8").read().startswith(c["command"] + "\n\n")
    popen(record={"device": "cuda", "cells": []})
    tool_cell = cell(
        config="llama-cpp-gpu", dtype="bf16", tool="llama-cpp", args=["--tool", "llama-cpp", "--n-gpu-layers", "-1"]
    )
    c = run.run_cell(tool_cell, "q.jsonl", "64", "", str(tmp_path), 100, compare_py="/venv/python")
    assert procs[-1].argv[:2] == ["/venv/python", os.path.join(run.HERE, "compare.py")]
    assert c["status"] == "ok" and c["report"] is None and "--rows" not in procs[-1].argv
    popen(record=None, rc=3)
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), 100)
    assert c["status"] == "failed" and c["error"].startswith("exit 3;") and "the last line" in c["error"]
    popen(record=None, polls=5)
    state["level"] = 3.0
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), 100, guard={"mem_floor": 8.0})
    assert procs[-1].killed and c["status"] == "failed"
    assert c["error"].startswith("killed: the kernel's free memory fell to 3% (floor 8%)")
    assert c["memory"]["min_level"] == 3.0
    state["level"] = 90.0
    popen(record=None, polls=5)
    clock = _Clock()
    monkeypatch.setattr(run.time, "time", lambda: clock.tick(60))
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), timeout=30)
    assert procs[-1].killed and c["error"] == "timed out after 30s"


def test_the_table_renders_a_cell_from_its_numbers_and_a_skip_from_its_note() -> None:
    table = _bench("table")
    ok = {
        "model": "m",
        "config": "gpu",
        "dtype": "bf16",
        "status": "ok",
        "seconds": 12.5,
        "cells": [
            {
                "new": 64,
                "first_s": 0.9,
                "greedy_s_tok": 0.05,
                "spec_s_tok": None,
                "tokens_per_pass": 1.0,
                "identical": "",
                "peak_ram_gb": 1.2,
                "peak_vram_gb": 3.4,
            },
            {
                "new": 256,
                "first_s": 0.4,
                "greedy_s_tok": 0.2,
                "spec_s_tok": 0.1,
                "tokens_per_pass": 2.0,
                "identical": "3/3",
                "peak_ram_gb": 2.5,
                "peak_vram_gb": 0.01,
            },
        ],
        "report": {"placement": {"resident": [0, 1, 2], "host": [3], "head": "card", "drafter": "none"}},
    }
    s = table.summarize(ok)
    assert s["tok/s"] == "20.0 / 10.0", "the rate as the configuration runs: speculation's where it is on"
    assert s["greedy tok/s"] == "20.0 / 5.00", "the greedy loop's own rate beside it"
    assert s["tok/pass"] == "1.00 / 2.00"
    assert s["first token"] == "0.40 s", "from the second length: the first carries the warm-up"
    assert (s["peak RAM"], s["peak VRAM/MLX"], s["identical"]) == ("2.5 GB", "3.4 GB", "3/3")
    assert s["placement"] == "3 resident, 1 host, head card"
    skipped = {"model": "m", "config": "cpu+mlx", "dtype": "bf16", "status": "skipped", "skip": "needs a split"}
    failed = {
        "model": "m",
        "config": "cpu",
        "dtype": "fp32",
        "status": "failed",
        "seconds": 3.0,
        "error": "exit 1; boom",
    }
    assert table.summarize(skipped) is None and table.summarize(failed) is None
    text = table.render([ok, skipped, failed], "64,256")
    assert "tok/s at 64 / 256" in text and "ok (spec identical 3/3)" in text
    assert "skipped: needs a split" in text and "exit 1; boom" in text
    md = table.render_markdown([ok, skipped, failed], "64,256")
    assert md.count("\n") == 2, "the markdown carries the rows with numbers only"
    assert "| m | gpu | bf16 | 20.0 / 10.0 | 20.0 / 5.00 | 1.00 / 2.00 | 0.40 s | 2.5 GB | 3.4 GB |" in md
    assert table.cell_line(failed) == "failed after 3.0s: exit 1; boom"
    assert (
        table.cell_line(ok)
        == "20.0 / 10.0 tok/s, 1.00 / 2.00 tok/pass, first 0.40 s, RAM 2.5 GB, VRAM/MLX 3.4 GB (12.5s)"
    )
    assert table.specs_line({"cpu": "c", "cores": 8, "ram_gb": 64, "gpu": "g", "vram_gb": 12.0, "os": "o"}).startswith(
        "c (8 cores), 64 GB RAM, g 12.0 GB, o"
    )


def test_the_machine_readers_and_the_environment_probe_parse_what_the_commands_say(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    host = _bench("host")
    env = _bench("env")
    plan = _bench("plan")
    monkeypatch.setattr(host, "capture", lambda cmd, timeout=20: "3072\n")
    assert host.gpu_used_gb() == 3.0
    monkeypatch.setattr(host, "capture", lambda cmd, timeout=20: "")
    assert host.gpu_used_gb() == 0.0, "no nvidia-smi is no card"
    assert host.peak_rss_bytes() > 0, "btb's own reader, from the checkout"
    assert set(host.memory_state()) == {"level", "swap_gb", "disk_gb"}
    asked = []

    def probe(cmd: list[str], timeout: int = 20) -> str:
        asked.append(cmd[-1])
        return "True" if "llama_cpp" in cmd[-1] else "False"

    monkeypatch.setattr(env, "capture", probe)
    assert env.compare_tools(sys.executable) == ["llama-cpp"]
    assert len(asked) == 3 and env.compare_tools(None) == [] and env.compare_tools("/nowhere/python") == []
    monkeypatch.setattr(plan, "compare_tools", lambda py: ["airllm", "llama-cpp"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    have = plan.machine_configs("/x")
    assert have[0] == "cpu" and have[-2:] == ["airllm", "llama-cpp"] and "gpu" not in have
    errors: list[str] = []
    assert plan.chosen_configs("gpu, cpu", ["cpu", "cpu+gpu", "gpu"], errors.append) == ["gpu", "cpu"]
    assert plan.chosen_configs(None, ["cpu", "mlx"], errors.append) == ["cpu", "mlx"]
    assert plan.chosen_configs("cpu,mlx-lm", ["cpu"], errors.append) == ["cpu"] and not errors
    assert "cannot run mlx-lm (no comparison environment)" in capsys.readouterr().out
    plan.chosen_configs("cpu,nope", ["cpu"], errors.append)
    assert errors and "nope" in errors[0]


def test_matrix_main_plans_a_dry_run_over_a_model_directory(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """the script end to end, running nothing: a model given as a directory is picked up by its config, the
    named configurations are planned with their arguments, the ones this machine lacks are left out"""
    import json

    matrix = _script("matrix")
    model = tmp_path / "tiny-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    rc = matrix.main(["--dry-run", "--devices", "cpu,gpu", "--models", str(model), "--new", "8,16"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tiny-model" in out and "cpu      fp32  --device cpu" in out and "answers of 8,16 tokens" in out
    if torch.cuda.is_available():
        assert "gpu      bf16  --device cuda --cpu-layers 0" in out
    else:
        assert "cannot run gpu (no card)" in out


def test_charts_draw_one_svg_per_model_from_the_run_docs(tmp_path: Path) -> None:
    """bench/charts.py: the matrix's and compare.py's JSON and a bench JSONL become a chart a model and machine
    (btb's speculative and greedy rows, a rival's, a failure as 'no result'), every label off the record's own
    fields, the newest run of a configuration winning; --embed puts the charts above the folded table once and
    leaves a section without charts alone"""
    spec = importlib.util.spec_from_file_location("charts", checkout("bench", "charts.py"))
    assert spec is not None and spec.loader is not None
    charts = importlib.util.module_from_spec(spec)
    sys.modules["charts"] = charts  # a dataclass under `from __future__ import annotations` resolves through here
    spec.loader.exec_module(charts)

    def cells(spec_s: float | None, greedy_s: float, ram: float = 8.0, vram: float = 9.0) -> list[Json]:
        return [
            {
                "new": n,
                "first_s": 0.25,
                "greedy_s_tok": greedy_s,
                "spec_s_tok": spec_s,
                "tokens_per_pass": 1.2 if spec_s else 1.0,
                "peak_ram_gb": ram,
                "peak_vram_gb": vram,
            }
            for n in (64, 256, 1024)
        ]

    specs = {
        "platform": "darwin",
        "os": "macOS 26.6",
        "cpu": "Apple M3 Pro",
        "unified_memory_gb": 36,
        "gpu": "Apple M3 Pro",
    }
    report = {
        "placement": {"mlx": [0, 1], "host": [0, 1], "head": "host", "compute_dtype": "bf16"},
        "speculation": {"proposer": "ngram", "tree_budget": 0, "ngram_tree": True},
    }
    btb_mlx = {"model": "a-1b", "repo": "Org/A-1B", "tool": None, "config": "mlx", "dtype": "bf16", "device": "mlx"}
    old = {
        "specs": specs,
        "finished": "2026-09-01T00:00:00",
        "cells": [
            {**btb_mlx, "status": "ok", "report": report, "cells": cells(0.2, 0.4)},
            {
                "model": "a-1b",
                "repo": "Org/A-1B",
                "tool": "airllm",
                "config": "airllm-cpu",
                "device_regime": "cpu",
                "dtype": "bf16",
                "status": "failed",
                "error": "Boom: old",
            },
        ],
    }
    new = {
        "specs": specs,
        "finished": "2026-09-10T00:00:00",
        "cells": [
            {**btb_mlx, "status": "ok", "report": report, "cells": cells(0.1, 0.2)},
            {**btb_mlx, "variant": "the head on the CPU", "status": "ok", "report": report, "cells": cells(0.5, 0.5)},
            {
                "model": "a-1b",
                "repo": "Org/A-1B",
                "tool": "llama-cpp",
                "config": "llama-cpp-gpu",
                "device_regime": "gpu",
                "dtype": "bf16",
                "status": "ok",
                "device": "cuda",
                "cells": cells(None, 0.05),
            },
            {
                "model": "a-1b",
                "repo": "Org/A-1B",
                "tool": "airllm",
                "config": "airllm-cpu",
                "device_regime": "cpu",
                "dtype": "bf16",
                "status": "failed",
                "error": "Boom: new",
            },
        ],
    }
    record = {
        "label": "x",
        "path": "/m/B-7B",
        "model": "Org/B-7B",
        "tool": None,
        "dtype": "fp32",
        "device": "cpu",
        "report": None,
        "cells": cells(None, 1.0, vram=0),
    }
    paths = []
    for name, doc in (("old.json", old), ("new.json", new)):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
        paths.append(str(tmp_path / name))
    (tmp_path / "run.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    paths.append(str(tmp_path / "run.jsonl"))
    readme = tmp_path / "README.md"
    text = (
        "## Benchmarks (Windows)\n\nintro\n\n| model | engine | device | how it runs | tok/s | pass | first | RAM | VRAM |\n|---|---|---|---|---|---|---|---|---|\n| W | btb | GPU | x | 1 / 2 / 3 | 1.00 | 0.1 s | 1 GB | 1 GB |\n\n"
        "## Benchmarks (Apple silicon)\n\nintro\n\n| model | engine | device | how it runs | tok/s | pass | first | RAM | MLX |\n|---|---|---|---|---|---|---|---|---|\n| M | btb | MLX | x | 1 / 2 / 3 | 1.00 | 0.1 s | 1 GB | 1 GB |\n\n## Tests\n"
    )
    readme.write_text(text, encoding="utf-8")

    loaded = charts.load(paths, specs_for_jsonl=specs)
    assert list(loaded) == ["Apple silicon"]
    machine, rows = loaded["Apple silicon"]
    assert machine.line == "Apple M3 Pro, 36 GB unified memory, macOS 26.6"
    a = [r for r in rows if r.model == "Org/A-1B"]
    assert [(r.engine, r.device, r.variant, r.note) for r in a] == [
        ("btb", "MLX", "", ""),
        ("btb", "MLX", "the head on the CPU", ""),  # a variant is its own row
        ("AirLLM", "CPU", "", "no result"),  # the older failure dropped, the error's text (the log's tail) never shown
        ("llama.cpp", "GPU", "", ""),
    ]
    assert a[0].speeds == [10.0, 10.0, 10.0]  # as it runs: the newer run's 0.1 s/token, not the older run's 0.2
    assert a[0].how == "bf16, 2 host, 2 mlx, head host" and a[0].per_pass == [1.2, 1.2, 1.2]
    assert a[1].how == "bf16, 2 host, 2 mlx, head host, the head on the CPU" and a[1].speeds == [2.0, 2.0, 2.0]
    assert a[3].how == "bf16" and a[3].speeds == [20.0, 20.0, 20.0] and a[3].per_pass == []
    b = [r for r in rows if r.model == "Org/B-7B"]
    assert [(r.engine, r.device, r.how, r.speeds) for r in b] == [("btb", "CPU", "fp32", [1.0, 1.0, 1.0])]

    made = charts.write(paths, str(tmp_path / "out"), str(readme), specs_for_jsonl=specs)
    assert [m for m, _ in made["Apple silicon"]] == ["Org/A-1B", "Org/B-7B"]
    svg = (tmp_path / "out" / "apple-silicon-org-a-1b.svg").read_text(encoding="utf-8")
    assert svg.count("<rect") == 1 + 3 + 3 * 3 and "no result" in svg and "Boom" not in svg and "MLX 9.0 GB" in svg
    once = charts.embed(text, made)
    assert (
        once.index("![Org/A-1B on Apple silicon]")
        < once.index("![Org/B-7B on Apple silicon]")
        < once.index("<details>")
    )
    assert once.index("<details>") < once.index("| M |") < once.index("</details>") < once.index("## Tests")
    assert once.split("## Benchmarks (Apple silicon)")[0] == text.split("## Benchmarks (Apple silicon)")[0]
    assert charts.embed(once, made) == once
