#!/usr/bin/env python3
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The build and the checks, one script for every platform: `python build.py` builds the native library and the
wheel for this machine (or `--target <triple>` for another), with the card's kernels when nvcc or a built fatbin
is here and without them otherwise; `check` is what a change must pass before it lands (ruff, mypy, cargo fmt and
clippy); `test` runs the suites. Standard library only, so it runs before anything is installed."""

from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.join(ROOT, "native")
LIB_NAMES = ("btb_native.dll", "libbtb_native.so", "libbtb_native.dylib")
TARGET_TAGS = {  # a Rust target triple's prefix -> the platform tag the wheel and btb/native/<tag> carry
    "x86_64-unknown-linux": "linux-x86_64",
    "aarch64-unknown-linux": "linux-aarch64",
    "x86_64-apple-darwin": "macos-x86_64",
    "aarch64-apple-darwin": "macos-aarch64",
    "x86_64-pc-windows": "windows-x86_64",
    "aarch64-pc-windows": "windows-aarch64",
}
WHEEL_PLATS = {
    "linux-x86_64": "linux_x86_64",
    "linux-aarch64": "linux_aarch64",
    "windows-x86_64": "win_amd64",
    "windows-aarch64": "win_arm64",
}
# the card's decode kernels: SASS for Ampere through Hopper, PTX past them; device code, the same on every host
GENCODE = [
    "-gencode",
    "arch=compute_80,code=sm_80",
    "-gencode",
    "arch=compute_86,code=sm_86",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_90,code=sm_90",
    "-gencode",
    "arch=compute_89,code=compute_89",
]
PY_DIRS = ["btb", "bench", "tests", "setup.py", "build.py"]


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def run(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    say("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, env=env, check=True)


def host_tag() -> str:
    system = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(
        platform.system(), platform.system().lower()
    )
    m = platform.machine().lower()
    arch = "x86_64" if m in ("x86_64", "amd64") else "aarch64" if m in ("arm64", "aarch64") else m
    return f"{system}-{arch}"


def target_tag(target: str) -> str:
    for prefix, tag in TARGET_TAGS.items():
        if target.startswith(prefix):
            return tag
    raise SystemExit(f"unknown target {target}")


def venv_python() -> str:
    """the repository's venv interpreter, the main checkout's when this is a worktree, else the one running this"""
    roots = [ROOT]
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        roots.append(os.path.dirname(os.path.abspath(os.path.join(ROOT, common))))
    except (OSError, subprocess.CalledProcessError):
        pass
    for r in roots:
        for rel in (("bin", "python"), ("Scripts", "python.exe")):
            p = os.path.join(r, ".venv", *rel)
            if os.path.isfile(p):
                return p
    return sys.executable


def macos_min(dylib: str) -> str:
    """the macOS floor the library was built for, off its own load command; 11.0 where it cannot be read"""
    try:
        out = subprocess.run(["otool", "-l", dylib], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "11.0"
    seen = False
    for line in out.splitlines():
        if "LC_BUILD_VERSION" in line:
            seen = True
        elif seen and "minos" in line:
            return line.split()[1]
    return "11.0"


def cmd_build(a: argparse.Namespace) -> None:
    tag = target_tag(a.target) if a.target else host_tag()
    out = os.path.join(NATIVE, "target", a.target, "release") if a.target else os.path.join(NATIVE, "target", "release")
    if not a.skip_rust and not a.cuda_only:
        if shutil.which("cargo") is None:
            raise SystemExit(
                "cargo not found: install a Rust toolchain (https://rustup.rs) to build the native library"
            )
        run(["cargo", "build", "--release"] + (["--target", a.target] if a.target else []), cwd=NATIVE)
    dst = os.path.join(ROOT, "btb", "native", tag)
    os.makedirs(dst, exist_ok=True)
    if not a.cuda_only:
        for n in LIB_NAMES:
            src = os.path.join(out, n)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                say(f"native: btb/native/{tag}/{n}")
    # the card's kernels go into a Linux or Windows wheel when they can: built where nvcc is, packaged where a
    # built fatbin is, otherwise left out (a card then runs through torch alone; the CPU and MLX never use them)
    fatbin = os.path.join(NATIVE, "cuda", "btb_kernels.fatbin")
    if (tag.startswith(("windows-", "linux-")) or a.cuda_only) and not a.no_cuda:
        have_nvcc = shutil.which("nvcc") is not None
        if a.cuda and not have_nvcc:
            raise SystemExit("nvcc not found: install the CUDA toolkit, or drop --cuda to build without the kernels")
        if have_nvcc:
            cu = os.path.join(NATIVE, "cuda", "btb_kernels.cu")
            run(["nvcc", "-O3", "-std=c++17", "--fatbin", *GENCODE, "-o", fatbin, cu])
        if a.cuda_only:
            if not os.path.isfile(fatbin):
                raise SystemExit(f"{fatbin} missing: --cuda-only needs nvcc")
            say(f"cuda-only: {fatbin}")
            return
        if os.path.isfile(fatbin):
            shutil.copy2(fatbin, os.path.join(dst, "btb_kernels.fatbin"))
            say(f"cuda: btb/native/{tag}/btb_kernels.fatbin" + ("" if have_nvcc else " (packaged as built)"))
        else:
            say("cuda: no nvcc and no built fatbin; the wheel ships without the card's kernels")
    # the wheel carries the library, so it is tagged py3-none-<platform>, never the pure py3-none-any; on macOS
    # the floor is the library's own build version, on Linux the manylinux container's policy tag when it sets one
    if tag.startswith("macos-"):
        arch = "arm64" if tag.endswith("aarch64") else "x86_64"
        plat = f"macosx_{macos_min(os.path.join(dst, 'libbtb_native.dylib')).split('.')[0]}_0_{arch}"
    else:
        plat = WHEEL_PLATS[tag]
    preset = os.environ.get("BTB_WHEEL_PLAT") or os.environ.get("AUDITWHEEL_PLAT")
    if preset:
        say(f"wheel platform tag preset: {preset} (would have been {plat})")
        plat = preset
    env = dict(os.environ, BTB_WHEEL_PLAT=plat)
    py = a.python or sys.executable
    run([py, "-m", "pip", "install", "--quiet", "build"], env=env)
    for stale in glob.glob(os.path.join(ROOT, "dist", "*-none-any.whl")):
        os.remove(stale)
    # -P: with the checkout as cwd, `-m build` would resolve to this script instead of PyPA's build package
    run([py, "-P", "-m", "build", "--wheel"], env=env)
    wheels = sorted(glob.glob(os.path.join(ROOT, "dist", "*.whl")), key=os.path.getmtime)
    if not wheels or plat not in os.path.basename(wheels[-1]):
        raise SystemExit(f"wheel not tagged {plat} (got {os.path.basename(wheels[-1]) if wheels else 'nothing'})")
    say(f"wheel: {os.path.basename(wheels[-1])} ({plat})")


def cmd_lint(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "check", *PY_DIRS])


def cmd_fix(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "check", "--fix", *PY_DIRS])


def cmd_fmt_check(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "format", "--check", *PY_DIRS])
    run(["cargo", "fmt", "--check"], cwd=NATIVE)


def cmd_format(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "ruff", "format", *PY_DIRS])
    run(["cargo", "fmt"], cwd=NATIVE)


def cmd_types(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "mypy"])


def cmd_clippy(a: argparse.Namespace) -> None:
    run(["cargo", "clippy", "--all-targets", "--", "-D", "warnings"], cwd=NATIVE)


def cmd_check(a: argparse.Namespace) -> None:
    cmd_lint(a)
    cmd_fmt_check(a)
    cmd_types(a)
    cmd_clippy(a)


def cmd_test(a: argparse.Namespace) -> None:
    run([venv_python(), "-m", "pytest", *(["tests/test_unit.py"] if a.fast else ["tests"]), *a.pytest])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="build.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser(
        "build", help="the native library, the card's kernels where they apply, and the wheel (the default)"
    )
    b.add_argument("--target", default="", metavar="TRIPLE", help="a Rust target triple to cross-build for")
    b.add_argument("--skip-rust", action="store_true", help="package the library already built")
    b.add_argument(
        "--cuda", action="store_true", help="require nvcc: fail rather than build without the card's kernels"
    )
    b.add_argument("--no-cuda", action="store_true", help="neither build nor package the card's kernels")
    b.add_argument("--cuda-only", action="store_true", help="build the fatbin with nvcc and stop: no library, no wheel")
    b.add_argument("--python", default="", help="the interpreter that builds the wheel (default: this one)")
    b.set_defaults(fn=cmd_build)
    for name, fn, text in (
        ("check", cmd_check, "lint, format check, types, clippy: what a change must pass"),
        ("lint", cmd_lint, "ruff check"),
        ("fix", cmd_fix, "ruff check --fix"),
        ("fmt-check", cmd_fmt_check, "ruff format --check, cargo fmt --check"),
        ("format", cmd_format, "ruff format, cargo fmt"),
        ("types", cmd_types, "mypy"),
        ("clippy", cmd_clippy, "cargo clippy, warnings as errors"),
    ):
        sub.add_parser(name, help=text).set_defaults(fn=fn)
    t = sub.add_parser(
        "test", help="the suites (--fast: tests/test_unit.py alone, seconds); other arguments go to pytest"
    )
    t.add_argument("--fast", action="store_true")
    t.set_defaults(fn=cmd_test)
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0].startswith("-"):
        args = ["build", *args]
    a, rest = ap.parse_known_args(args)
    if a.cmd != "test" and rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    a.pytest = rest
    try:
        a.fn(a)
    except subprocess.CalledProcessError as e:
        return int(e.returncode or 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
