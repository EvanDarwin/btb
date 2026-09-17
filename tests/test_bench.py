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
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import IO, TYPE_CHECKING, ClassVar, Protocol, cast

import pytest
import torch
from pytest import CaptureFixture, MonkeyPatch

from btb.kinds import Json
from tests.helpers import ROOT, checkout

if TYPE_CHECKING:
    from lib.records import BenchDevice, BenchDtype, BenchStatus
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


def _enums() -> tuple[type[BenchStatus], type[BenchDevice], type[BenchDtype]]:
    """The bench's (BenchStatus, BenchDevice, BenchDtype) enum classes, from the records module loaded the way
    the scripts do."""
    r = _bench("records")
    return r.BenchStatus, r.BenchDevice, r.BenchDtype


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
    carries the MXFP4 note in each regime. Devices and tools are separate axes now, composed in the matrix."""
    plan = _bench("plan")
    table = _bench("table")
    BS, BD, BT = _enums()
    big = {"name": "big", "repo": "x/big", "path": "/nowhere", "type": "qwen3", "size": 61 * 2**30, "packed": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cells = plan.plan_cells([big], [BD.CPU], ["mlx-lm", "airllm", "llama-cpp"], {}, {}, [])
    by = {table.cell_label(c): c for c in cells}
    assert set(by) == {"cpu", "mlx-lm", "airllm-gpu", "airllm-cpu", "llama-cpp-gpu", "llama-cpp-cpu"}
    assert by["airllm-gpu"]["args"] == ["--tool", "airllm", "--device", "cuda"]
    assert by["airllm-cpu"]["args"] == ["--tool", "airllm", "--device", "cpu"]
    assert by["llama-cpp-cpu"]["args"] == ["--tool", "llama-cpp", "--n-gpu-layers", "0"]
    assert by["mlx-lm"]["tool"] == "mlx-lm" and not by["cpu"].get("tool")
    # a safetensors checkpoint: the streaming rivals run at any size (cuda is up in this block), but llama.cpp
    # needs a GGUF, so it is a planned skip here rather than a silent conversion
    assert by["mlx-lm"].get("status") is None and by["airllm-gpu"].get("status") is None
    assert by["airllm-cpu"].get("status") is None
    assert by["llama-cpp-gpu"]["status"] is BS.DNR and "GGUF" in by["llama-cpp-gpu"]["reason"]
    assert by["llama-cpp-cpu"]["status"] is BS.DNR and "GGUF" in by["llama-cpp-cpu"]["reason"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    by = {table.cell_label(c): c for c in plan.plan_cells([big], [], ["airllm"], {}, {}, [])}
    assert by["airllm-gpu"]["status"] is BS.DNR and "no CUDA" in by["airllm-gpu"]["reason"]
    assert by["airllm-cpu"].get("status") is None
    gpt = dict(big, type="gpt_oss")
    cells = plan.plan_cells([gpt], [], ["airllm"], {}, {}, [])
    assert len(cells) == 2 and all(c["status"] is BS.DNR and "MXFP4" in c["reason"] for c in cells), (
        "gpt-oss on AirLLM is skipped on its own ground"
    )
    # btb's own cells: the CPU tier is fp32 (its native), naming fp32 adds the fp32-over-bf16 variant on a card
    # device, cpu+mlx is a planned skip
    cells = plan.plan_cells([big], [BD.CPU, BD.GPU, BD.CPU_MLX], [], {"dtype": {"fp32"}}, {}, [])
    run = {(c["device"], c["dtype"]): c for c in cells if c.get("status") is None}
    assert run[(BD.CPU, BT.FP32)]["args"] == ["--device", "cpu"]
    assert run[(BD.GPU, BT.FP32)]["args"] == ["--device", "cuda", "--cpu-layers", "0", "--fp32", "1"]
    cpumlx = next(c for c in cells if c["device"] is BD.CPU_MLX)
    assert cpumlx["status"] is BS.DNR and "--cpu-layers" in cpumlx["reason"]


def test_matrix_resume_reuses_only_finished_cells_of_the_same_lengths() -> None:
    plan = _bench("plan")
    BS, BD, BT = _enums()
    e = {"name": "m", "repo": "x/m", "path": "/nowhere", "type": "qwen3", "size": 1, "packed": False}
    cells = plan.plan_cells([e], [BD.CPU, BD.MLX], [], {}, {}, [])
    axes = {"pack12": False, "sampling": "greedy", "tool": None}
    prev = {
        "new": [64, 256],
        "cells": [
            {"model": "m", "device": BD.CPU, "dtype": BT.FP32, "mega": False, **axes,
             "status": BS.OK, "seconds": 5, "cells": [{"new": 64}]},
            {"model": "m", "device": BD.MLX, "dtype": BT.BF16, "mega": True, **axes, "status": BS.DNR, "seconds": 1},
        ],
    }
    assert plan.resume_cells(cells, prev, [64, 256], "prev.json") == 1
    cpu = next(c for c in cells if c["device"] is BD.CPU)
    assert cpu.get("resumed") == "prev.json" and cpu["status"] is BS.OK and cpu["args"] == ["--device", "cpu"]
    assert not next(c for c in cells if c["device"] is BD.MLX).get("resumed")
    cells = plan.plan_cells([e], [BD.CPU], [], {}, {}, [])
    assert plan.resume_cells(cells, prev, [64, 256, 1024], "prev.json") == 0, "other answer lengths are another run"


class _Bail(Exception):
    """what a raising error callback throws, standing in for argparse.error's SystemExit."""


def _bail(msg: str) -> None:
    raise _Bail(msg)


def test_the_kind_parser_names_each_axis_and_the_selection_composes() -> None:
    """a bare kind names its axis by the disjoint vocabulary, axis=value is explicit (and the only form for a
    model glob), an unknown kind or a --config naming one axis twice is an error; --only/--not/--config parse
    into the allow, deny and points the planner reads"""
    plan = _bench("plan")
    assert plan.parse_kind("mlx", _bail) == ("device", "mlx")
    assert plan.parse_kind("fp32", _bail) == ("dtype", "fp32")
    assert plan.parse_kind("nomega", _bail) == ("mega", "nomega")
    assert plan.parse_kind("pack12", _bail) == ("pack12", "pack12")
    assert plan.parse_kind("llama-cpp", _bail) == ("tool", "llama-cpp")
    assert plan.parse_kind("greedy", _bail) == ("sampling", "greedy")
    assert plan.parse_kind("t0.7", _bail) == ("sampling", "t0.7")
    assert plan.parse_kind("model=qwen3-*", _bail) == ("model", "qwen3-*")
    assert plan.parse_kind("device=mlx", _bail) == ("device", "mlx")
    with pytest.raises(_Bail, match="unknown kind"):
        plan.parse_kind("nonsense", _bail)
    with pytest.raises(_Bail, match="not a dtype kind"):
        plan.parse_kind("dtype=int4", _bail)
    with pytest.raises(_Bail, match="not a sampling kind"):
        plan.parse_kind("sampling=hot", _bail)
    allow, deny, points = plan.parse_selection(["mlx,fp32", "gpu"], ["nomega"], ["mlx+fp32"], _bail)
    assert allow == {"device": {"mlx", "gpu"}, "dtype": {"fp32"}} and deny == {"mega": {"nomega"}}
    assert points == [{"device": "mlx", "dtype": "fp32"}]
    with pytest.raises(_Bail, match="two device values"):
        plan.parse_selection([], [], ["mlx+cpu"], _bail)


def _points(plan: ModuleType, e: Json, devices: Sequence[BenchDevice], **sel: object) -> list[tuple[str, str, bool, str]]:
    """The (device, dtype, mega, sampling) of the cells a selection plans and does not gate, sorted."""
    cells = plan.plan_cells([e], devices, [], sel.get("allow", {}), sel.get("deny", {}), sel.get("points", []))
    return sorted((c["device"], c["dtype"], c["mega"], c["sampling"]) for c in cells if c.get("status") is None)


def test_the_selection_widens_narrows_and_names_points() -> None:
    """--only replaces an axis's default, --not subtracts, and --config widens the grid to reach its points
    then filters to their combinations - so naming fp32 in one config keeps the bf16 cell another asks for,
    and two configs are a union with no cross terms. A temperature carries --temperature and names the cell."""
    plan = _bench("plan")
    table = _bench("table")
    BS, BD, BT = _enums()
    e = {"name": "m", "repo": "x/m", "path": "/nowhere", "type": "qwen3", "size": 1, "packed": False}
    devs = [BD.CPU, BD.CPU_MLX, BD.MLX]
    # the default: each device its native dtype, mega on where the megakernel lays it out, cpu+mlx a skip
    assert _points(plan, e, devs) == [(BD.CPU, BT.FP32, False, "greedy"), (BD.MLX, BT.BF16, True, "greedy")]
    # --only replaces the device set; --not subtracts (allow/deny carry kind tokens, the axis vocabulary)
    assert _points(plan, e, devs, allow={"device": {"mlx"}}) == [(BD.MLX, BT.BF16, True, "greedy")]
    assert _points(plan, e, devs, deny={"device": {"cpu"}}) == [(BD.MLX, BT.BF16, True, "greedy")]
    # --config gpu+fp32 alongside --config gpu: the fp32 config widens the grid, the plain gpu keeps its bf16.
    # fp32 is a card variant - MLX has no fp32 path, so mlx+fp32 would be a planned skip (see the gates test)
    assert _points(plan, e, [BD.GPU], points=[{"device": "gpu", "dtype": "fp32"}, {"device": "gpu"}]) == [
        (BD.GPU, BT.BF16, False, "greedy"),
        (BD.GPU, BT.FP32, False, "greedy"),
    ]
    # two configs are a union, never their cross: no gpu bf16, no cpu that neither names
    assert _points(plan, e, [BD.CPU, BD.GPU], points=[{"device": "gpu", "dtype": "fp32"}, {"device": "cpu"}]) == [
        (BD.CPU, BT.FP32, False, "greedy"),
        (BD.GPU, BT.FP32, False, "greedy"),
    ]
    # a temperature is its own sampling cell, with the flag and the label
    cells = plan.plan_cells([e], [BD.MLX], [], {"sampling": {"t0.7"}}, {}, [])
    c = next(c for c in cells if c.get("status") is None)
    assert c["sampling"] == "t0.7" and c["args"][-2:] == ["--temperature", "0.7"]
    assert table.cell_label(c) == "mlx·t0.7"
    # the manifest is the axis values the grid actually spanned
    cells = plan.plan_cells([e], [BD.CPU, BD.MLX], [], {}, {}, [])
    m = plan.axes_manifest(cells)
    assert m["device"] == [BD.CPU, BD.MLX] and m["dtype"] == [BT.BF16, BT.FP32]
    assert m["mega"] == ["mega", "nomega"] and m["pack12"] == ["nopack12"] and m["tool"] == ["btb"]


def test_the_gates_skip_what_cannot_run_with_a_reason() -> None:
    """a gate marks a cell DNR with why it cannot run rather than dropping it: bf16 on the CPU tier, fp32 on
    MLX (no fp32 path there), the megakernel off its MLX/bf16/dense ground, and pack-12 with no 12-bit store
    beside the model. The megakernel label and flag appear only where it applies, never on a mixture of
    experts."""
    plan = _bench("plan")
    table = _bench("table")
    BS, BD, BT = _enums()
    e = {"name": "m", "repo": "x/m", "path": "/nowhere", "type": "qwen3", "size": 1, "packed": False}
    moe = dict(e, type="gpt_oss")
    reason = lambda cells, dev: next(c["reason"] for c in cells if c["device"] == dev and c["status"] is BS.DNR)
    assert "CPU tier is fp32" in reason(plan.plan_cells([e], [BD.CPU], [], {"dtype": {"bf16"}}, {}, []), BD.CPU)
    assert "MLX runs bf16 only" in reason(plan.plan_cells([e], [BD.MLX], [], {"dtype": {"fp32"}}, {}, []), BD.MLX)
    assert "MLX" in reason(plan.plan_cells([e], [BD.CPU], [], {"mega": {"mega"}}, {}, []), BD.CPU)
    assert "mixture of experts" in reason(plan.plan_cells([moe], [BD.MLX], [], {"mega": {"mega"}}, {}, []), BD.MLX)
    assert "12-bit store" in reason(plan.plan_cells([e], [BD.MLX], [], {"pack12": {"pack12"}}, {}, []), BD.MLX)
    # a MoE on MLX is a plain cell (mega does not apply): no --mlx-mega flag, no nomega label
    moe_cell = next(c for c in plan.plan_cells([moe], [BD.MLX], [], {}, {}, []) if c.get("status") is None)
    assert "--mlx-mega" not in moe_cell["args"] and table.cell_label(moe_cell) == "mlx"


def test_pick_models_globs_the_cache_and_excludes(monkeypatch: MonkeyPatch) -> None:
    """--models takes cache names, repo ids and globs over them; --exclude-models drops glob matches from
    whatever was chosen; the default is the whole cache."""
    plan = _bench("plan")
    import btb

    have = [
        {"name": n, "repo": r, "path": "/c/" + n, "type": "qwen3", "size": 1, "packed": False}
        for n, r in [
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
            ("qwen3-4b", "Qwen/Qwen3-4B"),
            ("phi-4-mini-instruct", "microsoft/Phi-4-mini-instruct"),
        ]
    ]
    monkeypatch.setattr(btb, "available_models", lambda: have)
    names = lambda **kw: sorted(
        e["name"] for e in plan.pick_models(kw.get("names", []), kw.get("filters", []), kw.get("exclude", []))
    )
    assert names() == ["phi-4-mini-instruct", "qwen3-0.6b", "qwen3-4b"]
    assert names(names=["qwen3-*"]) == ["qwen3-0.6b", "qwen3-4b"]
    assert names(names=["*4b*"]) == ["qwen3-4b"]
    assert names(names=["qwen/qwen3-4b"]) == ["qwen3-4b"], "a repo id matches by its lowercased id"
    assert names(exclude=["phi*"]) == ["qwen3-0.6b", "qwen3-4b"]
    assert names(names=["qwen3-*"], exclude=["*0.6b*"]) == ["qwen3-4b"]


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


def test_llama_cpp_supported_reads_the_converter_registry(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """llama.cpp's convertible-architecture set, read from a checkout's --print-supported-models (which logs
    the names to stderr): None without an interpreter or a checkout, the arch names parsed out otherwise."""
    env = _bench("env")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert env.llama_cpp_supported(None) is None, "no interpreter, nothing to ask"
    assert env.llama_cpp_supported("/venv/python") is None, "no convert_hf_to_gguf.py in the checkout"
    (tmp_path / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")

    class _Done:
        stdout = ""
        stderr = "INFO:hf-to-gguf:Qwen3ForCausalLM\nGemma3ForConditionalGeneration\nLlamaForCausalLM\n"

    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _Done())
    assert env.llama_cpp_supported("/venv/python") == frozenset(
        {"Qwen3ForCausalLM", "Gemma3ForConditionalGeneration", "LlamaForCausalLM"}
    )


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
    assert plan.TOOL_NAMES == tuple(tools.TOOLS)
    assert "CUDA" in tools.TOOLS["airllm"].cannot("gpu", "qwen3", cuda=False, gguf=False)
    assert tools.TOOLS["airllm"].cannot("cpu", "qwen3", cuda=False, gguf=False) is None
    assert "MXFP4" in tools.TOOLS["airllm"].cannot("cpu", "gpt_oss", cuda=True, gguf=False)
    # a GGUF: only llama.cpp reads it directly; mlx-lm and airllm load HF/MLX weights, so they are planned skips
    assert "GGUF" in tools.TOOLS["mlx-lm"].cannot("mlx", "qwen3", cuda=False, gguf=True)
    assert "GGUF" in tools.TOOLS["airllm"].cannot("cpu", "qwen3", cuda=False, gguf=True)
    # llama.cpp is the mirror: a GGUF runs; a checkpoint is a skip whose reason depends on the converter's
    # registry - convertible when it knows the arch, unsupported when it does not, "needs a GGUF" when there
    # is no checkout to ask
    lc = tools.TOOLS["llama-cpp"]
    assert lc.cannot("gpu", "qwen3", cuda=False, gguf=True) is None
    assert "GGUF" in lc.cannot("gpu", "qwen3", cuda=False, gguf=False)
    reg = frozenset({"Qwen3ForCausalLM"})
    assert "converts Qwen3ForCausalLM" in lc.cannot("gpu", "qwen3", cuda=False, gguf=False, arch="Qwen3ForCausalLM", supported=reg)
    assert "no entry" in lc.cannot("gpu", "x", cuda=False, gguf=False, arch="WeirdForCausalLM", supported=reg)


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
        assert c["first_s"] == pytest.approx(0.5) and c["base_s_tok"] == pytest.approx(0.1)
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
    # the matrix owns the axes now; the record carries only its identity and the numbers
    assert (rec["label"], rec["model"], rec["path"]) == ("L", "model", "/model")
    assert "tool" not in rec and "device" not in rec
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
    BS, BD, BT = _enums()
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
        base = {
            "model": "m", "device": BD.CPU, "dtype": BT.FP32, "pack12": False, "sampling": "greedy",
            "mega": False, "tool": None, "path": "/m", "args": ["--device", "cpu"],
        }
        return {**base, **kw}

    popen(record={"cells": [{"new": 64, "base_s_tok": 0.5}], "report": {"placement": {}}})
    c = run.run_cell(cell(), "q.jsonl", "64", "0", str(tmp_path), timeout=100)
    assert (
        c["status"] is BS.OK and c["device"] is BD.CPU and c["cells"][0]["new"] == 64 and c["report"] == {"placement": {}}
    )
    argv = procs[-1].argv
    assert argv[:5] == [sys.executable, "-m", "btb.cli", "bench", "/m"]
    assert argv[argv.index("--label") + 1] == "m cpu fp32" and argv[-4:] == ["--device", "cpu", "--rows", "0"]
    assert c["memory"] == {"min_level": 90.0, "max_swap_gb": 0.0, "min_disk_gb": 100.0}
    assert os.path.basename(c["log"]) == "m-cpu-fp32.log"
    assert open(c["log"], encoding="utf-8").read().startswith(c["command"] + "\n\n")
    popen(record={"cells": []})
    tool_cell = cell(
        device=BD.GPU, dtype=BT.BF16, tool="llama-cpp", args=["--tool", "llama-cpp", "--n-gpu-layers", "-1"]
    )
    c = run.run_cell(tool_cell, "q.jsonl", "64", "", str(tmp_path), 100, compare_py="/venv/python")
    assert procs[-1].argv[:2] == ["/venv/python", os.path.join(run.HERE, "compare.py")]
    assert c["status"] is BS.OK and c["report"] is None and "--rows" not in procs[-1].argv
    # a child that ends with no record: DNR, its exit code and the log's tail in the reason
    popen(record=None, rc=3)
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), 100)
    assert c["status"] is BS.DNR and c["reason"].startswith("errored (exit 3):") and "the last line" in c["reason"]
    # the guard trips the memory floor: OOM, with the level it fell to
    popen(record=None, polls=5)
    state["level"] = 3.0
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), 100, guard={"mem_floor": 8.0})
    assert procs[-1].killed and c["status"] is BS.OOM
    assert c["reason"].startswith("the kernel's free memory fell to 3% (floor 8%)") and c["memory"]["min_level"] == 3.0
    # past the time limit: DNF
    state["level"] = 90.0
    popen(record=None, polls=5)
    clock = _Clock()
    monkeypatch.setattr(run.time, "time", lambda: clock.tick(60))
    c = run.run_cell(cell(), "q.jsonl", "64", "", str(tmp_path), timeout=30)
    assert procs[-1].killed and c["status"] is BS.DNF and c["reason"] == "did not finish within 30s"


def test_the_table_renders_a_cell_from_its_numbers_and_a_skip_from_its_note() -> None:
    table = _bench("table")
    BS, BD, BT = _enums()
    ok = {
        "model": "m",
        "device": BD.GPU,
        "dtype": BT.BF16,
        "tool": None,
        "mega": False,
        "status": BS.OK,
        "seconds": 12.5,
        "cells": [
            {
                "new": 64,
                "first_s": 0.9,
                "base_s_tok": 0.05,
                "spec_s_tok": None,
                "tokens_per_pass": 1.0,
                "identical": "",
                "peak_ram_gb": 1.2,
                "peak_vram_gb": 3.4,
            },
            {
                "new": 256,
                "first_s": 0.4,
                "base_s_tok": 0.2,
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
    assert s["base tok/s"] == "20.0 / 5.00", "the no-spec baseline's own rate beside it"
    assert s["tok/pass"] == "1.00 / 2.00"
    assert s["first token"] == "0.40 s", "from the second length: the first carries the warm-up"
    assert s["total"] == "12.5 s", "the cell's total wall time"
    assert (s["peak RAM"], s["peak VRAM/MLX"], s["identical"]) == ("2.5 GB", "3.4 GB", "3/3")
    assert s["placement"] == "3 layers on the card, 1 on the CPU, the head on the card"
    # a planning skip is a DNR; a guard kill is an OOM - both render as STATUS: reason, neither summarizes
    gated = {"model": "m", "device": BD.CPU_MLX, "dtype": BT.BF16, "tool": None, "status": BS.DNR, "reason": "needs a split"}
    killed = {
        "model": "m",
        "device": BD.CPU,
        "dtype": BT.FP32,
        "tool": None,
        "status": BS.OOM,
        "seconds": 3.0,
        "reason": "swap in use reached 13.0 GB (cap 12 GB)",
    }
    assert table.summarize(gated) is None and table.summarize(killed) is None
    text = table.render([ok, gated, killed], "64,256")
    assert "tok/s at 64 / 256" in text and "ok (spec identical 3/3)" in text
    assert "DNR: needs a split" in text and "OOM: swap in use reached 13.0 GB (cap 12 GB)" in text
    md = table.render_markdown([ok, gated, killed], "64,256")
    assert md.count("\n") == 2, "the markdown carries the rows with numbers only"
    assert "| m | gpu | bf16 | 20.0 / 10.0 | 20.0 / 5.00 | 1.00 / 2.00 | 0.40 s | 12.5 s | 2.5 GB | 3.4 GB |" in md
    assert table.cell_line(killed) == "OOM after 3.0s: swap in use reached 13.0 GB (cap 12 GB)"
    assert (
        table.cell_line(ok)
        == "20.0 / 10.0 tok/s, 1.00 / 2.00 tok/pass, first 0.40 s, RAM 2.5 GB, VRAM/MLX 3.4 GB (12.5s)"
    )
    assert table.specs_line({"cpu": "c", "cores": 8, "ram_gb": 64, "gpu": "g", "vram_gb": 12.0, "os": "o"}).startswith(
        "c (8 cores), 64 GB RAM, g 12.0 GB, o"
    )


def test_the_machine_readers_and_the_environment_probe_parse_what_the_commands_say(
    monkeypatch: MonkeyPatch,
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
    devs = plan.machine_devices()
    assert devs[0] == "cpu" and "gpu" not in devs and "cpu+gpu" not in devs, "no CUDA leaves out the card regimes"
    assert plan.machine_tools("/x") == ["airllm", "llama-cpp"], "the tools whose env is present and device is here"


def test_matrix_main_plans_a_dry_run_over_a_model_directory(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    """the script end to end, running nothing: a model given as a directory is picked up by its config, the
    selected axes are planned with their arguments, a device this machine lacks yields no cell for it"""
    import json

    matrix = _script("matrix")
    model = tmp_path / "tiny-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    rc = matrix.main(["--dry-run", "--only", "btb,cpu,gpu", "--models", str(model), "--new", "8,16"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tiny-model" in out and "cpu" in out and "--device cpu" in out and "answers of 8,16 tokens" in out
    if torch.cuda.is_available():
        assert "--device cuda --cpu-layers 0" in out
    else:
        assert "--device cuda" not in out, "no card: the gpu cell is simply not planned"


def test_charts_draw_one_svg_per_model_from_the_run_docs(tmp_path: Path) -> None:
    """bench/charts.py: the matrix's JSON and a standalone bench JSONL become a chart per model and machine
    (btb's row, a rival's, a rival's failure as its state, a variant off the axes), every label read off the
    cell's axes, the newest run of a configuration winning; a JSONL's device and dtype come back off its
    placement"""
    spec = importlib.util.spec_from_file_location("charts", checkout("bench", "charts.py"))
    assert spec is not None and spec.loader is not None
    charts = importlib.util.module_from_spec(spec)
    sys.modules["charts"] = charts  # a dataclass under `from __future__ import annotations` resolves through here
    spec.loader.exec_module(charts)
    BS, BD, BT = _enums()

    def cells(spec_s: float | None, base_s: float, ram: float = 8.0, vram: float = 9.0) -> list[Json]:
        return [
            {
                "new": n,
                "first_s": 0.25,
                "base_s_tok": base_s,
                "spec_s_tok": spec_s,
                "tokens_per_pass": 1.2 if spec_s else 1.0,
                "peak_ram_gb": ram,
                "peak_vram_gb": vram,
            }
            for n in (64, 256, 1024)
        ]

    specs = {"platform": "darwin", "os": "macOS 26", "cpu": "Apple M3 Pro", "unified_memory_gb": 36, "gpu": "Apple M3 Pro"}
    report = {"placement": {"mlx": [0, 1], "host": [0, 1], "head": "host", "compute_dtype": "bf16"}}
    axes = {"pack12": False, "sampling": "greedy"}
    mlx = {"model": "a-1b", "repo": "Org/A-1B", "type": "qwen3", "tool": None, "device": BD.MLX, "dtype": BT.BF16, **axes}
    airllm = {"model": "a-1b", "repo": "Org/A-1B", "type": "qwen3", "tool": "airllm", "device": BD.CPU, "dtype": BT.BF16}
    old = {
        "specs": specs,
        "finished": "2026-09-01T00:00:00",
        "cells": [
            {**mlx, "mega": True, "status": BS.OK, "report": report, "cells": cells(0.2, 0.4)},
            {**airllm, "status": BS.DNR, "reason": "Boom: old", "seconds": 1.0},
        ],
    }
    new = {
        "specs": specs,
        "finished": "2026-09-10T00:00:00",
        "cells": [
            {**mlx, "mega": True, "status": BS.OK, "report": report, "cells": cells(0.1, 0.2)},
            {**mlx, "mega": False, "status": BS.OK, "report": report, "cells": cells(0.5, 0.5)},  # the nomega variant
            {
                "model": "a-1b", "repo": "Org/A-1B", "type": "qwen3", "tool": "llama-cpp", "device": BD.GPU,
                "dtype": BT.BF16, "status": BS.OK, "cells": cells(None, 0.05),
            },
            {**airllm, "status": BS.DNR, "reason": "Boom: new", "seconds": 1.0},
        ],
    }
    record = {  # a standalone `btb bench` JSONL: the slimmed record, device and dtype off its placement
        "label": "x",
        "path": "/m/B-7B",
        "model": "Org/B-7B",
        "report": {"placement": {"host": [0, 1], "compute_dtype": "fp32"}},
        "cells": cells(None, 1.0, vram=0),
    }
    paths = []
    for name, doc in (("old.json", old), ("new.json", new)):
        (tmp_path / name).write_text(json.dumps(doc), encoding="utf-8")
        paths.append(str(tmp_path / name))
    (tmp_path / "run.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    paths.append(str(tmp_path / "run.jsonl"))

    loaded = charts.load(paths, specs_for_jsonl=specs)
    assert list(loaded) == ["Apple silicon"]
    machine, rows = loaded["Apple silicon"]
    assert machine.line == "Apple M3 Pro, 36 GB unified memory, macOS 26"
    a = [r for r in rows if r.model == "Org/A-1B"]
    assert [(r.engine, r.device, r.variant, r.note) for r in a] == [
        ("btb", "MLX", "", ""),
        ("btb", "MLX", "no megakernel", ""),  # a nomega cell is its own row, the off-default axis its variant
        ("AirLLM", "CPU", "", "DNR: Boom: new"),  # the older failure dropped; the state and its reason are shown
        ("llama.cpp", "GPU", "", ""),
    ]
    # the megakernel variant is btb's own MLX path; a rival on MLX never carries a "no megakernel" note
    assert charts._variant({"tool": None, "device": BD.MLX, "dtype": BT.BF16, "type": "qwen3", "mega": False}) == "no megakernel"
    assert charts._variant({"tool": "mlx-lm", "device": BD.MLX, "dtype": BT.BF16, "type": "qwen3", "mega": False}) == ""
    assert a[0].speeds == [10.0, 10.0, 10.0]  # as it runs: the newer run's 0.1 s/token, not the older run's 0.2
    assert a[0].how == "bf16, 2 layers on the GPU" and a[0].per_pass == [1.2, 1.2, 1.2]
    assert a[1].how == "bf16, 2 layers on the GPU, no megakernel" and a[1].speeds == [2.0, 2.0, 2.0]
    assert a[3].how == "bf16" and a[3].speeds == [20.0, 20.0, 20.0] and a[3].per_pass == []
    b = [r for r in rows if r.model == "Org/B-7B"]
    assert [(r.engine, r.device, r.how, r.speeds) for r in b] == [("btb", "CPU", "fp32, 2 layers on the CPU", [1.0, 1.0, 1.0])]

    made = charts.write(paths, str(tmp_path / "out"), specs_for_jsonl=specs)
    assert [m for m, _ in made["Apple silicon"]] == ["Org/A-1B", "Org/B-7B"]
    svg = (tmp_path / "out" / "apple-silicon-org-a-1b.svg").read_text(encoding="utf-8")
    # the dark ground rect, the legend panel + its 3 engine swatches, then 3 data rows of a track and a bar per
    # length; the DNR note row draws none
    assert svg.count("<rect") == 1 + 1 + 3 + 3 * 6
    assert "DNR: Boom: new" in svg  # the note row draws the state and its reason
    assert "MLX <tspan" in svg and "9.0 GB" in svg  # the MLX peak: a muted label with an inked value
