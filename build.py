#!/usr/bin/env python3
# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The build and the checks, one script for every platform: `python build.py` builds the native library and the
wheel for this machine (or `--target <triple>` for another), with the card's kernels when nvcc or a built fatbin
is here and without them otherwise; `bindings` writes the card kernels' typed Python launchers from their CUDA
signatures; `check` is what a change must pass before it lands (ruff, mypy, cargo fmt and clippy); `test` runs the
suites. Standard library only, so it runs before anything is installed."""

from __future__ import annotations

import argparse
import glob
import keyword
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass

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


# --- the card kernels' bindings -----------------------------------------------------------------------------------
# native/cuda/btb_roles.cuh names each kernel parameter's role, a typedef; `bindings` reads it and every extern "C"
# kernel's signature and writes the roles as NewTypes into a marked block of btb/kinds.py and a typed launcher a
# kernel into btb/engine/card_launch.py. The cert (tests/cert/cuda_ops.py) holds the checked-in files to this output.
CUDA_DIR = os.path.join(NATIVE, "cuda")
ROLES_H = os.path.join(CUDA_DIR, "btb_roles.cuh")
KINDS_PY = os.path.join(ROOT, "btb", "kinds.py")
LAUNCH_PY = os.path.join(ROOT, "btb", "engine", "card_launch.py")
ROLES_OPEN = "# --- card kernel roles: written by `python build.py bindings` from native/cuda/btb_roles.cuh ---"
ROLES_CLOSE = "# --- end of the card kernel roles ---"
LINE = 120  # pyproject's line length: the files are written as `ruff format` lays them out
# a role's C type -> (the ctypes type a scalar of it is passed as, its alias in card_launch.py, its Python type); a
# bf16 is a pointer's element only
C_TYPES: dict[str, tuple[str, str, str]] = {
    "int": ("c_int", "_I", "int"),
    "unsigned": ("c_uint", "_U", "int"),
    "long long": ("c_longlong", "_LL", "int"),
    "unsigned long long": ("c_ulonglong", "_ULL", "int"),
    "unsigned char": ("c_ubyte", "_UB", "int"),
    "float": ("c_float", "_F", "float"),
    "bf16": ("", "", ""),
}


class BindingError(ValueError):
    """native/cuda breaks a rule the card kernels' bindings rest on; the message names where and which"""


@dataclass(frozen=True)
class Role:
    """a role of btb_roles.cuh: its name, the C type it names, its comment, and its section's title"""

    name: str
    ctype: str
    note: str
    section: str


@dataclass(frozen=True)
class Param:
    """a kernel parameter: its name, its role, and whether it points to the role's element"""

    name: str
    role: str
    pointer: bool


@dataclass(frozen=True)
class Kernel:
    """an extern "C" kernel, or a macro's kernels as one: the launcher's name (the kernel's; a macro's without its
    variant), the kernel's name with the variant as a format field (`btb_attn_flash_d{D}`), the variant's parameter and
    its instantiated values (none for a lone kernel), and the parameters"""

    method: str
    pattern: str
    variant: str
    values: tuple[str, ...]
    params: tuple[Param, ...]

    def names(self) -> list[str]:
        """the kernels' names, a macro's at each of its instantiations"""
        if not self.variant:
            return [self.pattern]
        return [self.pattern.replace("{" + self.variant + "}", v) for v in self.values]


_TYPEDEF = re.compile(r"typedef\s+([a-z][a-z0-9 ]*?)\s+(\w+);\s*//\s*(.+)")
_SECTION = re.compile(r"//\s*---\s*(.+?)\s*---")
# a kernel entry point: `extern "C" __global__ void [__launch_bounds__(...)] btb_<name>(`, the name possibly on the
# next line; never a macro's templated name, which `##` breaks before its `(`
_EXTERN = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?(btb_\w+)\s*\(')
_TEMPLATE = re.compile(r"(btb_\w+)##(\w+)\s*\(")  # a macro's kernel name, its variant last: btb_gemv_bf16_m##M(
_DEFINE = re.compile(r"#define\s+(\w+)\(([^)]*)\)")  # a function-like macro's head and parameters
_OBJECT = re.compile(r"#define\s+(\w+)\s+(.*)")  # an object-like macro and its text
_INST = re.compile(r"^(\w+)\(([^()]*)\)\s*$", re.M)  # a macro's instantiation on a line of its own: GEMV(16)
_PARAM = re.compile(r"(?:const )?(\w+) ?(\*?) ?(?:__restrict__ )?(\w+)")  # `[const] Role[*] [__restrict__] name`


def card_roles() -> list[Role]:
    """btb_roles.cuh's roles in its order: each `typedef <C type> Name;  // what it is`, under the last
    `// --- section ---` line"""
    out: list[Role] = []
    section = ""
    with open(ROLES_H, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            s = _SECTION.fullmatch(line)
            if s:
                section = s.group(1)
                continue
            if not line.startswith("typedef"):
                continue
            m = _TYPEDEF.fullmatch(line)
            ctype = " ".join(m.group(1).split()) if m else ""
            if m is None or ctype not in C_TYPES:
                raise BindingError(
                    f"btb_roles.cuh:{n}: a role is `typedef <{' | '.join(C_TYPES)}> Name;  // what it is`: {line}"
                )
            if any(r.name == m.group(2) for r in out):
                raise BindingError(f"btb_roles.cuh:{n}: {m.group(2)} is named twice")
            out.append(Role(m.group(2), ctype, m.group(3).strip(), section))
    return out


def _cuda_source() -> str:
    """every native/cuda/*.cu|*.cuh joined, a macro's continued lines folded into one"""
    parts = []
    for fn in sorted(os.listdir(CUDA_DIR)):
        if fn.endswith((".cu", ".cuh")):
            with open(os.path.join(CUDA_DIR, fn), encoding="utf-8") as f:
                parts.append(f.read())
    return "\n".join(parts).replace("\\\n", " ")


def _inside(text: str, i: int) -> str:
    """the text between the bracket at text[i] and the one that closes it"""
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] in "([{"
        depth -= text[j] in ")]}"
        if depth == 0:
            return text[i + 1 : j]
    raise BindingError(f"no bracket closes {text[i : i + 60]!r}")


def _params(kernel: str, text: str, objects: dict[str, str], roles: dict[str, Role]) -> tuple[Param, ...]:
    """a kernel's parameter list as Params, a list that is one object-like macro (ATTN_FLASH_PF_ARGS) read as its
    text; a parameter whose type is not a role is a BindingError"""
    text = " ".join(text.split())
    text = objects.get(text, text)
    out = []
    for part in text.split(","):  # a role is one name, so no parameter's type holds a comma
        p = " ".join(part.split())
        m = _PARAM.fullmatch(p)
        if m is None or m.group(1) not in roles:
            raise BindingError(
                f"{kernel}: parameter `{p}` is not `[const] Role[*] [__restrict__] name` with a role of btb_roles.cuh"
            )
        out.append(Param(m.group(3), m.group(1), m.group(2) == "*"))
    return tuple(out)


def card_kernels(roles: list[Role] | None = None) -> list[Kernel]:
    """every extern "C" kernel of native/cuda, a macro's kernels as one, held to the rules (`_hold`)"""
    by_name = {r.name: r for r in (card_roles() if roles is None else roles)}
    src = _cuda_source()
    lines = [line.strip() for line in src.splitlines()]
    objects = {}
    for line in lines:
        o = _OBJECT.fullmatch(line)
        if o and not _DEFINE.match(line):
            objects[o.group(1)] = " ".join(o.group(2).split())
    out = [
        Kernel(m.group(1), m.group(1), "", (), _params(m.group(1), _inside(src, m.end() - 1), objects, by_name))
        for m in _EXTERN.finditer(src)
    ]
    values: dict[str, list[str]] = {}
    for macro, args in _INST.findall(src):
        values.setdefault(macro, []).append(args.strip())
    for line in lines:
        d = _DEFINE.match(line)
        if d is None or "btb_" not in line:
            continue
        macro = d.group(1)
        found = list(_TEMPLATE.finditer(line))
        if len(found) != len(re.findall(r"btb_\w*##", line)):
            raise BindingError(f"{macro}: a kernel's name in a macro is `btb_<name>##<variant>(`, the variant last")
        for t in found:
            head, var = t.group(1), t.group(2)
            if [p.strip() for p in d.group(2).split(",")] != [var]:
                raise BindingError(
                    f"{macro}: a kernel macro takes one parameter, the variant its kernels' names end in"
                )
            if not values.get(macro):
                raise BindingError(f"{macro} defines kernels but is never instantiated")
            # the launcher drops the variant and its tag: btb_attn_flash_d##D -> btb_attn_flash
            out.append(
                Kernel(
                    re.sub(r"_[a-z]+$", "", head),
                    head + "{" + var + "}",
                    var,
                    tuple(values[macro]),
                    _params(head + "##" + var, _inside(line, t.end() - 1), objects, by_name),
                )
            )
    _hold(out, by_name)
    return sorted(out, key=lambda k: k.method)


def _hold(kernels: list[Kernel], roles: dict[str, Role]) -> None:
    """the rules the launchers' types rest on, a BindingError naming the first broken: two parameters of a kernel
    never share a role (a swap of them would type-check); a role is a pointer's everywhere or a scalar's everywhere (its
    Python type is one or the other), and a bf16 one a pointer's; every role is some kernel's; no two launchers share a
    name; a parameter's name is one its launcher can take"""
    pointer: dict[str, bool] = {}
    methods: set[str] = set()
    for k in kernels:
        if k.method in methods:
            raise BindingError(f"two launchers would be named {k.method}: rename a kernel")
        methods.add(k.method)
        seen: dict[str, str] = {}
        for p in k.params:
            if p.role in seen:
                raise BindingError(
                    f"{k.pattern}: {seen[p.role]} and {p.name} share the role {p.role}: "
                    "give one its own (btb_roles.cuh)"
                )
            seen[p.role] = p.name
            if pointer.setdefault(p.role, p.pointer) != p.pointer:
                raise BindingError(
                    f"{p.role} is a pointer's in one kernel and a scalar's in another ({k.pattern}'s {p.name})"
                )
            if not p.pointer and roles[p.role].ctype == "bf16":
                raise BindingError(f"{k.pattern}: {p.name} passes a bf16 by value; a bf16 role is a pointer's")
            if keyword.iskeyword(p.name) or p.name in ("launcher", "grid", "block", "stream", "shared", k.variant):
                raise BindingError(f"{k.pattern}: a launcher cannot take a parameter named {p.name}")
    unused = [r for r in roles if r not in pointer]
    if unused:
        raise BindingError(f"roles no kernel takes: {', '.join(unused)} (btb_roles.cuh)")


# a bracketed list's item: text, or a bracketed list of text (opening, items, closing)
_Item = str | tuple[str, list[str], str]


def _flat(item: _Item) -> str:
    return item if isinstance(item, str) else item[0] + ", ".join(item[1]) + item[2]


def _lay(head: str, items: Sequence[_Item], tail: str, indent: str, after: str = "") -> list[str]:
    """`head`, the items, `tail` laid out as `ruff format` lays a bracketed list: on one line where it fits; else the
    items on one line of their own between the brackets; else an item a line, each with a trailing comma, a bracketed
    item laid out the same way in turn. `after` follows the closing bracket (an enclosing list's comma)"""
    flat = ", ".join(_flat(i) for i in items)
    if len(indent + head + flat + tail + after) <= LINE:
        return [indent + head + flat + tail + after]
    inner = indent + "    "
    if len(inner + flat) <= LINE:
        return [indent + head, inner + flat, indent + tail + after]
    out = [indent + head]
    for i in items:
        if isinstance(i, str) or len(inner + _flat(i) + ",") <= LINE:
            out.append(inner + _flat(i) + ",")
        else:
            out += _lay(i[0], i[1], i[2], inner, ",")
    return [*out, indent + tail + after]


def _pointers(kernels: list[Kernel]) -> dict[str, bool]:
    return {p.role: p.pointer for k in kernels for p in k.params}


def _kinds_block(roles: list[Role], kernels: list[Kernel]) -> str:
    """btb/kinds.py's card roles, a NewType each: a pointer role over int (a card address), a scalar over its value"""
    pointer = _pointers(kernels)
    lines = [
        ROLES_OPEN,
        "# Each card kernel parameter's role as its own type (btb/engine/card_launch.py types the launchers with",
        "# them): a pointer role's value is a card address of its element, a scalar role's the value itself. So two",
        "# of a launch's arguments swapped, or a value of one role where another goes, fail the type check.",
    ]
    section = None
    for r in roles:
        if r.section != section:
            section = r.section
            lines += ["", f"# --- {section} ---"]
        lines.append(f"# {r.ctype}{'*' if pointer[r.name] else ''}: {r.note}")
        lines.append(f'{r.name} = NewType("{r.name}", {"int" if pointer[r.name] else C_TYPES[r.ctype][2]})')
    lines.append(ROLES_CLOSE)
    return "\n".join(lines)


def _launch_py(roles: list[Role], kernels: list[Kernel]) -> str:
    """btb/engine/card_launch.py: the kernels' names, `Launcher`, `ptr`, a launcher function a kernel, and the card
    module's methods that alias them"""
    by_name = {r.name: r for r in roles}
    pointer = _pointers(kernels)
    scalars = sorted({C_TYPES[by_name[r].ctype] for r, ptr in pointer.items() if not ptr}, key=lambda c: c[1])
    imported: list[str] = sorted(pointer, key=_isort_key)
    launch_params = [
        "self",
        "name: str",
        "grid: Dim3",
        "block: Dim3",
        "args: Sequence[Any]",
        "stream: torch.cuda.Stream | None = None",
        "shared: int = 0",
    ]
    out = [
        "# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2",
        "# Written by `python build.py bindings` from native/cuda: edit a kernel or btb_roles.cuh there, then run it.",
        '"""The card kernels\' launchers: a function a kernel, taking a launcher and then the kernel\'s parameters in',
        "order, each typed by its role (btb/kinds.py, from native/cuda/btb_roles.cuh): a pointer role a card address",
        "(`ptr(Role, tensor)`), a scalar role its value. A macro's kernels share a function, the variant (the head width",
        "D, the rows M) first. A function packs its arguments as the kernel declares them and hands them to the launcher",
        "(`Launcher`): btb's card module (`_Cuda` in btb/engine/native.py), or one's own with its own streams. The card",
        'module carries each function as a method: `k.btb_attn_flash(D, ...)` is `btb_attn_flash(k, D, ...)`."""',
        "",
        "from __future__ import annotations",
        "",
        "import ctypes",
        "from collections.abc import Callable, Sequence",
        "from typing import Any, Protocol, TypeVar",
        "",
        "import torch",
        "",
        *_lay("from btb.kinds import (", imported, ")", ""),
        "",
        "Dim3 = tuple[int, int, int]",
        "_P = ctypes.c_void_p",
        *(f"{alias} = ctypes.{ctype}" for ctype, alias, _ in scalars),
        '_R = TypeVar("_R")',
        "",
        "# every kernel the card program loads, a macro's at each of its instantiations",
        *_lay("KERNELS: tuple[str, ...] = (", [f'"{n}"' for k in kernels for n in k.names()], ")", ""),
        "",
        "",
        "class Launcher(Protocol):",
        '    """what launches a kernel: its name in the launcher\'s module, the grid and block, and one ctypes value a',
        '    parameter (c_void_p a pointer, c_int, c_float, ..), on `stream` with `shared` bytes of dynamic shared memory"""',
        "",
        *_lay("def launch(", launch_params, ") -> None: ...", "    "),
        "",
        "",
        "def ptr(role: Callable[[int], _R], t: torch.Tensor | None) -> _R:",
        '    """tensor `t`\'s card address as an argument of `role` (`ptr(Query, q)`); None the null pointer. The address',
        '    keeps nothing alive: hold `t` while a launch or a captured graph can read it"""',
        "    return role(0 if t is None else t.data_ptr())",
    ]
    for k in kernels:
        sig: list[_Item] = ["launcher: Launcher", *([f"{k.variant}: int"] if k.variant else [])]
        sig += ["grid: Dim3", "block: Dim3"]
        sig += [f"{p.name}: {p.role}" for p in k.params]
        sig += ["*", "stream: torch.cuda.Stream | None = None", "shared: int = 0"]
        out += ["", "", *_lay(f"def {k.method}(", sig, ") -> None:", "")]
        if k.variant:
            out.append(f'    """{k.pattern}, {k.variant} one of {", ".join(k.values)}"""')
        args = [f"{'_P' if p.pointer else C_TYPES[by_name[p.role].ctype][1]}({p.name})" for p in k.params]
        name = f'f"{k.pattern}"' if k.variant else f'"{k.pattern}"'
        out += _lay("launcher.launch(", [name, "grid", "block", ("[", args, "]"), "stream", "shared"], ")", "    ")
    out += [
        "",
        "",
        "class CardLaunch:",
        '    """the launchers as methods of the card\'s kernel module (`_Cuda`), each the function of its name with the',
        '    module as its launcher"""',
        "",
        *_lay("def launch(", launch_params, ") -> None:", "    "),
        "        raise NotImplementedError",
        "",
        "    ptr = staticmethod(ptr)",
        *(f"    {k.method} = {k.method}" for k in kernels),
    ]
    return "\n".join(out) + "\n"


def _isort_key(name: str) -> tuple[int, str, str]:
    """ruff's isort order of imported names: constants, then classes (a role's CamelCase), then the rest, each kind
    ignoring case"""
    return (0 if name.isupper() else 1 if name[:1].isupper() else 2, name.lower(), name)


def card_bindings() -> dict[str, str]:
    """the files `bindings` writes, path -> text: btb/kinds.py with its card roles block, and btb/engine/card_launch.py"""
    roles = card_roles()
    kernels = card_kernels(roles)
    with open(KINDS_PY, encoding="utf-8") as f:
        kinds = f.read()
    a, b = kinds.find(ROLES_OPEN), kinds.find(ROLES_CLOSE)
    if a < 0 or b < a:
        raise BindingError(
            f"btb/kinds.py has no card roles block: its first and last lines are\n{ROLES_OPEN}\n{ROLES_CLOSE}"
        )
    kinds = kinds[:a] + _kinds_block(roles, kernels) + kinds[b + len(ROLES_CLOSE) :]
    return {KINDS_PY: kinds, LAUNCH_PY: _launch_py(roles, kernels)}


def stale_bindings() -> list[str]:
    """the files `bindings` would change, from the checkout's root"""
    out = []
    for path, text in card_bindings().items():
        try:
            with open(path, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = None
        if current != text:
            out.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    return out


def cmd_bindings(a: argparse.Namespace) -> None:
    try:
        files = card_bindings()
    except BindingError as e:
        raise SystemExit(f"bindings: {e}") from None
    for path, text in files.items():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        try:
            with open(path, encoding="utf-8") as f:
                same = f.read() == text
        except FileNotFoundError:
            same = False
        if same:
            say(f"bindings: {rel} is current")
            continue
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        say(f"bindings: wrote {rel}")


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
    if a.no_wheel:
        return
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
    b.add_argument("--no-wheel", action="store_true", help="stop once btb/native/<tag> holds the library")
    b.add_argument("--python", default="", help="the interpreter that builds the wheel (default: this one)")
    b.set_defaults(fn=cmd_build)
    for name, fn, text in (
        ("bindings", cmd_bindings, "the card kernels' typed launchers and roles, from native/cuda's signatures"),
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
