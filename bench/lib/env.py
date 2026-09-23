# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The comparison environment: .venv-compare against bench/requirements.txt, which rival engines it
holds, and setting it up
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

from bench.lib import REQUIREMENTS, ROOT, capture, say
from bench.lib.tools import TOOLS

# the HF architecture names in a --print-supported-models dump; llama.cpp registers them as `...ForCausalLM`,
# `...ForConditionalGeneration`, or a bare `...Model`
_ARCH = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:ForCausalLM|ForConditionalGeneration|Model)\b")


def llama_cpp_supported(compare_py: str | None) -> frozenset[str] | None:
    """The HF architectures llama.cpp's converter can turn into a GGUF, read once from a checkout's
    `--print-supported-models` (its registry, printed through logging on stderr). None when no checkout is
    found or it cannot be asked - the planner then cannot judge support and only says a GGUF is needed. The
    checkout is $LLAMA_CPP_DIR or ./llama.cpp, matching where convert_gguf and prepare.py put it."""
    conv = os.path.join(os.environ.get("LLAMA_CPP_DIR") or os.path.join(ROOT, "llama.cpp"), "convert_hf_to_gguf.py")
    if not compare_py or not os.path.isfile(conv):
        return None
    try:
        r = subprocess.run(
            [compare_py, conv, "--print-supported-models"], capture_output=True, text=True, timeout=180, check=False
        )
    except Exception:
        return None
    return frozenset(_ARCH.findall((r.stdout or "") + (r.stderr or ""))) or None


def compare_tools(py: str | None) -> list[str]:
    """
    Which of the comparison engines the interpreter `py` has
    """
    if not py or not os.path.exists(py):
        return []
    probe = "import importlib.util as u; print(u.find_spec('{}') is not None)"
    return [name for name, tool in TOOLS.items() if capture([py, "-c", probe.format(tool.module)]) == "True"]


def marker_holds(marker: str) -> bool:
    """
    A requirement's environment marker on this machine; only `sys_platform ==/!= "x"` is spoken here
    """
    m = re.match(r'sys_platform\s*(==|!=)\s*"(\w+)"', marker)
    if not m:
        return True
    return (sys.platform == m.group(2)) == (m.group(1) == "==")


def requirement_lines(path: str) -> list[tuple[str, str | None]]:
    """
    (name, pinned version or None) for each requirement of `path` whose marker holds here
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            spec, _, marker = line.partition(";")
            if marker.strip() and not marker_holds(marker.strip()):
                continue
            name, _, ver = spec.strip().partition("==")
            out.append((name.strip().lower().replace("-", "_"), ver.strip() or None))
    return out


def installed_versions(py: str) -> dict[str, str]:
    """
    {name: version} of the packages in the interpreter `py`; {} when it cannot be asked
    """
    code = "import importlib.metadata as m, json; print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
    try:
        return {k.lower().replace("-", "_"): v for k, v in json.loads(capture([py, "-c", code])).items()}
    except Exception:
        return {}


def compare_env_mismatch(
    py: str | None, have: dict[str, str] | None = None, requirements: str = REQUIREMENTS
) -> list[str]:
    """
    What keeps `py` from being the environment `requirements` describes: the interpreter missing, or a
    package absent or at another version (a local tail like `+cu128` is not a difference). `have` stands
    in for the interpreter's own answer
    """
    if not py or not os.path.exists(py):
        return ["no interpreter"]
    if have is None:
        have = installed_versions(py)
    out = []
    for name, ver in requirement_lines(requirements):
        got = have.get(name)
        if got is None:
            out.append(f"{name} missing")
        elif ver and got.split("+", 1)[0] != ver:
            out.append(f"{name} {got} (wants {ver})")
    return out


def setup_compare_env(py: str, requirements: str = REQUIREMENTS) -> bool:
    """
    Make `py` the comparison environment: a venv from the interpreter this one was made from, torch from
    the CUDA index that matches the card (plain torch without one), then `requirements`; llama-cpp-python
    builds with GGML_CUDA where there is a card, which needs the CUDA toolkit. pip's output is shown; True
    when every step returned 0
    """
    import torch

    venv = os.path.dirname(os.path.dirname(py))
    base = getattr(sys, "_base_executable", None) or sys.executable
    cuda = torch.version.cuda if torch.cuda.is_available() else None
    pin = next((v for n, v in requirement_lines(requirements) if n == "torch"), None)
    torch_req = f"torch=={pin}" if pin else "torch"
    steps: list[list[str]] = []
    if not os.path.exists(py):
        steps.append([base, "-m", "venv", venv])
    steps.append([py, "-m", "pip", "install", "--upgrade", "pip"])
    if cuda:
        index = "https://download.pytorch.org/whl/cu" + cuda.replace(".", "")
        steps.append([py, "-m", "pip", "install", torch_req, "--index-url", index])
    else:
        steps.append([py, "-m", "pip", "install", torch_req])
    steps.append([py, "-m", "pip", "install", "-r", requirements])
    env = dict(os.environ)
    if cuda:
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    for cmd in steps:
        say("$ " + " ".join(cmd))
        if subprocess.run(cmd, env=env, check=False).returncode != 0:
            say("that step failed; the comparison rows are left out")
            return False
    return True


def ensure_compare_env(py: str | None, ask: bool) -> str | None:
    """
    `py` when it is the comparison environment; otherwise, with `ask` on a console, the offer to set it up
    (y/N) and what came of it. None means the comparison rows are left out
    """
    missing = compare_env_mismatch(py)
    if not missing:
        return py
    say(f".venv-compare is not ready: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    if not ask or not py or not sys.stdin.isatty():
        return None
    try:
        answer = input("[matrix] set it up now (a venv, torch for the card, bench/requirements.txt)? [y/N] ")
    except EOFError:
        return None
    if answer.strip().lower() not in ("y", "yes"):
        return None
    return py if setup_compare_env(py) and not compare_env_mismatch(py) else None
