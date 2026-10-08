"""The CUDA card-kernel matrix. The card runs only where a GPU is present (CI has none), so this certifies the
kernel surface structurally, source-parsed and torch-free: every kernel the engine loads (`_Cuda.KERNELS` in
btb/engine/native.py) must be defined in native/cuda/*.cu|.cuh, every kernel defined there must be one the
engine loads, and every one it loads must be one a `.launch(...)` in btb/ can launch - its name read through what
flows into the call (`_Resolver`), never a presence check or a message naming it. Each launch's argument list must
hold as many values as every kernel it can launch declares parameters: the driver reads one pointer a parameter from
the list it is handed, so a short list is a read past it and a long one a value the kernel never sees. A new .cu
kernel the engine does not load, a loaded name with no definition, a kernel nothing launches any more (a dead path,
compiled and loaded for nothing), a launch whose kernel or argument count the parse cannot read, or one whose count
is not its kernel's is a reported gap, so a card kernel cannot land uncertified, nor outlive its last caller.

    python -m tests.cert.cuda_ops --report    # the kernel matrix and the plain-language gaps
    python -m tests.cert.cuda_ops --missing   # only the gaps in plain language: what is missing and how to close
    python -m tests.cert.cuda_ops --check      # nonzero when the loaded, defined and launched sets disagree
"""

from __future__ import annotations

import argparse
import ast
import glob
import os
import re
import sys
from enum import StrEnum

from .core import BTB_SRC

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUDA_DIR = os.path.join(ROOT, "native", "cuda")
NATIVE_PY = os.path.join(BTB_SRC, "engine", "native.py")


class Missing(StrEnum):
    """what a CUDA-kernel gap needs, as a stable key; `MISSING` maps each to (what is absent, how to close it)."""

    DEFINED_NOT_LOADED = "defined-not-loaded"
    LOADED_NOT_DEFINED = "loaded-not-defined"
    LOADED_NOT_LAUNCHED = "loaded-not-launched"
    LAUNCH_UNRESOLVED = "launch-unresolved"
    LAUNCH_ARGS_UNRESOLVED = "launch-args-unresolved"
    LAUNCH_ARITY = "launch-arity"


# kind -> (what is missing about kernel {s}, how to close it).
MISSING: dict[Missing, tuple[str, str]] = {
    Missing.DEFINED_NOT_LOADED: (
        "kernel {s} is defined in native/cuda/*.cu|.cuh but the engine never loads it (not in _Cuda.KERNELS) - "
        "a new card kernel with no wiring or cert",
        "add {s} to _Cuda.KERNELS in btb/engine/native.py and launch it where it belongs, or remove the "
        "definition if the kernel is dead",
    ),
    Missing.LOADED_NOT_DEFINED: (
        "the engine loads {s} (_Cuda.KERNELS) but native/cuda defines no such kernel - the fatbin load would "
        "fail on a GPU",
        "define {s} in native/cuda (or its generating macro), or drop it from _Cuda.KERNELS if it was renamed",
    ),
    Missing.LOADED_NOT_LAUNCHED: (
        "the engine loads {s} (_Cuda.KERNELS) but no `.launch(...)` in btb/ can launch it - a presence check, a "
        "requirement list or a message naming it launches nothing: a dead kernel path, compiled and loaded for nothing",
        "launch {s} where it belongs, or remove its definition from native/cuda and its name from _Cuda.KERNELS "
        "(and the tests and benches that drive it alone)",
    ),
    Missing.LAUNCH_UNRESOLVED: (
        "the launch at {s} names its kernel in a way the cert cannot read (a parameter, a loop variable, an "
        "unpacked tuple), so it cannot say which kernels the launch keeps alive",
        "spell the kernel's name at {s} as a constant, an f-string, a conditional of those or a local assigned one, "
        "as every other launch does",
    ),
    Missing.LAUNCH_ARGS_UNRESOLVED: (
        "the launch at {s} builds its argument list in a way the cert cannot count (a call, a comprehension, a list "
        "grown in place), so it cannot say the kernel gets the values it reads",
        "build the arguments at {s} as list literals, their sums, conditionals of those or locals assigned one "
        "(`*name` for a part that varies), as every other launch does",
    ),
    Missing.LAUNCH_ARITY: (
        "the launch at {s}: the driver reads one value a declared parameter from the list it is handed - a short "
        "list a read past its end, a long one a value the kernel never sees",
        "give the launch at {s} exactly the kernel's parameters, in its order (native/cuda), or the kernel the "
        "launch's values",
    ),
}

# a kernel entry point: `extern "C" __global__ void [__launch_bounds__(...)] btb_<name>(`, the name possibly on
# the next line (the MMA kernel), and NOT a macro template (a name ending at `##` is caught by _macro_kernels).
_EXTERN = re.compile(r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s*)?(btb_\w+)\s*\(')
# a macro body's templated name(s), every `##` part, and its parameter list's opening: btb_gemv_bf16_m##M(
_TEMPLATE = re.compile(r"(btb_\w+(?:##\w+)+)\s*\(")
_DEFINE = re.compile(r"#define\s+(\w+)\(([^)]*)\)")  # a macro definition head and its parameters
_OBJECT = re.compile(r"#define\s+(\w+)(?:\s+(.*))?$")  # an object-like macro: its name, then its text
_INST = re.compile(r"^(\w+)\(([\d,\s]+)\)", re.M)  # a macro instantiation, e.g. GEMV(16) or ATTN(2, 64)


def _cuda_source() -> str:
    """every native/cuda/*.cu|*.cuh joined, with C line-continuations folded so a macro body is one logical line."""
    text = []
    for fn in sorted(os.listdir(CUDA_DIR)):
        if fn.endswith((".cu", ".cuh")):
            with open(os.path.join(CUDA_DIR, fn), encoding="utf-8") as f:
                text.append(f.read())
    return "\n".join(text).replace("\\\n", " ")


def loaded_kernels() -> set[str]:
    """the card kernels the engine loads by name, from `_Cuda.KERNELS` in native.py (parsed as source, so no
    torch import); a missing one fails at load on a GPU, so this is the authoritative required set. Read as Python,
    not cut at the first parenthesis: a comment in the list naming a file "(btb_gemm.cuh)" ended it there"""
    with open(NATIVE_PY, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=NATIVE_PY)
    out: set[str] = set()
    for node in ast.walk(tree):
        listed = _load_list(node)
        if listed is not None:
            out |= {n.value for n in ast.walk(listed) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return {k for k in out if k.startswith("btb_")}


def _arity(text: str, i: int, objects: dict[str, str]) -> int:
    """the parameters of the list whose `(` is text[i]: its top-level commas plus one (none for `()` or `(void)`); a
    list that is one object-like macro (`ATTN_FLASH_PF_ARGS`) counted as the macro's text"""
    depth = 0
    commas = 0
    j = i
    for j in range(i, len(text)):
        c = text[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                break
        elif c == "," and depth == 1:
            commas += 1
    body = text[i + 1 : j].strip()
    if body in objects:
        return _arity(f"({objects[body]})", 0, objects)
    return 0 if body in ("", "void") else commas + 1


def defined_arity() -> dict[str, int]:
    """the card kernels defined in native/cuda and the parameters each declares: the literal `extern "C" __global__`
    entry points plus the macro-generated ones (each `#define MACRO(P) ... btb_x##P(...) ...` expanded over its
    `MACRO(v)` instantiations, every expansion the template's parameters)."""
    src = _cuda_source()
    objects: dict[str, str] = {}
    templates: dict[str, tuple[list[str], list[tuple[str, int]]]] = {}
    lines = [line.strip() for line in src.splitlines()]
    for line in lines:
        o = _OBJECT.match(line)
        if o and not line.startswith(f"#define {o.group(1)}("):
            objects[o.group(1)] = (o.group(2) or "").strip()
    out = {m.group(1): _arity(src, m.end() - 1, objects) for m in _EXTERN.finditer(src)}
    for line in lines:
        d = _DEFINE.match(line)
        if d:
            params = [p.strip() for p in d.group(2).split(",") if p.strip()]
            names = [(m.group(1), _arity(line, m.end() - 1, objects)) for m in _TEMPLATE.finditer(line)]
            templates[d.group(1)] = (params, names)
    for macro, args in _INST.findall(src):
        if macro not in templates:
            continue
        params, names = templates[macro]
        values = dict(zip(params, (a.strip() for a in args.split(",")), strict=False))
        for name, n in names:
            # each `##` part a parameter's value where it names one, else the literal text between them
            out["".join(values.get(part, part) for part in name.split("##"))] = n
    return out


def defined_kernels() -> set[str]:
    """the card kernels defined in native/cuda (`defined_arity`)."""
    return set(defined_arity())


Pieces = tuple[str | None, ...]  # a name's spelling: literal text, None for a part the source does not fix (`{D}`)
_DEPTH = 6  # how deep a name's resolution follows assignments and calls before it calls the part unknown


class _Resolver:
    """the kernel names a module's `.launch(...)` calls can launch, read off their first argument: a constant, an
    f-string (each hole resolved the same way - `via = "_tbl" if paged else ""` gives both spellings - or any name
    part where the source does not fix it, `{D}`), a conditional of those, a local or an enclosing function's variable
    (every assignment to it), or a call of a method whose returns are those (`k.flash_prefill_kernel(D)`). Only what
    flows into a launch counts: a kernel named in a presence check, a requirement list or a message is not launched"""

    def __init__(self, trees: list[ast.Module]) -> None:
        self.parent: dict[int, ast.AST] = {}
        self.defs: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        for tree in trees:
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    self.parent[id(child)] = node
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.defs.setdefault(node.name, []).append(node)

    def _scopes(self, node: ast.AST) -> list[ast.AST]:
        """the functions enclosing `node`, innermost first, then its module"""
        out: list[ast.AST] = []
        cur = self.parent.get(id(node))
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                out.append(cur)
            cur = self.parent.get(id(cur))
        return out

    @staticmethod
    def _assigned(scope: ast.AST, name: str) -> list[ast.expr] | None:
        """the values `scope`'s own body assigns to `name` (not a nested function's); None where it binds it otherwise
        (a parameter, a loop, a tuple's unpacking) - a value the source does not fix"""
        values: list[ast.expr] = []
        other = isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            a.arg == name for a in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]
        )
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        values.append(node.value)
                    elif any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(t)):
                        other = True
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                if node.value is not None:
                    values.append(node.value)
            elif isinstance(node, (ast.For, ast.AugAssign, ast.NamedExpr, ast.With, ast.comprehension)):
                target = getattr(node, "target", None)
                if target is not None and any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(target)):
                    other = True
            stack.extend(ast.iter_child_nodes(node))
        return None if other or not values else values

    def spell(self, expr: ast.expr, depth: int = _DEPTH) -> set[Pieces]:
        """every spelling `expr` can take"""
        unknown: set[Pieces] = {(None,)}
        if depth <= 0:
            return unknown
        if isinstance(expr, ast.Constant):
            return {(expr.value,)} if isinstance(expr.value, str) else unknown
        if isinstance(expr, ast.JoinedStr):
            out: set[Pieces] = {()}
            for v in expr.values:
                part = self.spell(v.value, depth - 1) if isinstance(v, ast.FormattedValue) else self.spell(v, depth - 1)
                out = {a + b for a in out for b in part}
            return out
        if isinstance(expr, ast.IfExp):
            return self.spell(expr.body, depth - 1) | self.spell(expr.orelse, depth - 1)
        if isinstance(expr, ast.Name):
            for scope in self._scopes(expr):
                values = self._assigned(scope, expr.id)
                if values is not None:
                    return set().union(*(self.spell(v, depth - 1) for v in values))
            return unknown
        if isinstance(expr, ast.Call):
            fn = expr.func.attr if isinstance(expr.func, ast.Attribute) else getattr(expr.func, "id", None)
            returns = [
                r.value
                for d in self.defs.get(fn or "", [])
                for r in ast.walk(d)
                if isinstance(r, ast.Return) and r.value is not None
            ]
            return set().union(*(self.spell(r, depth - 1) for r in returns)) if returns else unknown
        return unknown

    def count(self, expr: ast.expr, depth: int = _DEPTH) -> set[int] | None:
        """every length the argument list `expr` can have: a list or tuple literal (a `*part` the lengths of the part),
        a sum of those, a conditional of those, a variable assigned them (every assignment); None where the source
        does not fix it"""
        if depth <= 0:
            return None
        if isinstance(expr, (ast.List, ast.Tuple)):
            out = {0}
            for e in expr.elts:
                part = self.count(e.value, depth - 1) if isinstance(e, ast.Starred) else {1}
                if part is None:
                    return None
                out = {a + b for a in out for b in part}
            return out
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left, right = self.count(expr.left, depth - 1), self.count(expr.right, depth - 1)
            return None if left is None or right is None else {a + b for a in left for b in right}
        if isinstance(expr, ast.IfExp):
            body, orelse = self.count(expr.body, depth - 1), self.count(expr.orelse, depth - 1)
            return None if body is None or orelse is None else body | orelse
        if isinstance(expr, ast.Name):
            for scope in self._scopes(expr):
                values = self._assigned(scope, expr.id)
                if values is not None:
                    counts = [self.count(v, depth - 1) for v in values]
                    if any(c is None for c in counts):
                        return None
                    return set().union(*(c for c in counts if c is not None))
            return None
        return None

    @staticmethod
    def _args(call: ast.Call) -> ast.expr | None:
        """a `.launch(name, grid, block, args, ...)` call's argument list"""
        if len(call.args) >= 4:
            return call.args[3]
        return next((kw.value for kw in call.keywords if kw.arg == "args"), None)

    def launches(self, tree: ast.Module) -> list[tuple[int, set[Pieces], set[int] | None]]:
        """(line, spellings, argument counts) of every `.launch(...)` call in `tree`"""
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "launch":
                if node.args:
                    args = self._args(node)
                    out.append((node.lineno, self.spell(node.args[0]), None if args is None else self.count(args)))
        return out


def _pattern(spelling: Pieces) -> re.Pattern[str] | None:
    """a spelling as the names it matches - each unfixed part one or more name characters - or None where it fixes no
    `btb_` kernel's name (nothing literal before its first unknown part)"""
    merged: list[str | None] = []
    for p in spelling:
        if p is not None and merged and merged[-1] is not None:
            merged[-1] += p
        else:
            merged.append(p)
    if not merged or merged[0] is None or not merged[0].startswith("btb_"):
        return None
    return re.compile("".join(r"\w+" if p is None else re.escape(p) for p in merged))


def _sources(root: str) -> list[tuple[str, ast.Module]]:
    """every module of the package under `root` (btb/mlx's Metal kernels share names with the card's, not launches)"""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if os.path.join(root, "mlx") + os.sep in path:
            continue
        with open(path, encoding="utf-8") as f:
            out.append((path, ast.parse(f.read(), filename=path)))
    return out


# a launch: its file and line, the spellings of the kernel it names, the lengths its argument list can have (None: the
# parse cannot count it)
Site = tuple[str, int, set[Pieces], set[int] | None]


def launch_sites(sources: list[tuple[str, ast.Module]] | None = None) -> list[Site]:
    """every card kernel launch in btb/"""
    sources = _sources(BTB_SRC) if sources is None else sources
    resolver = _Resolver([tree for _, tree in sources])
    return [(path, line, sp, n) for path, tree in sources for line, sp, n in resolver.launches(tree)]


def _where(path: str, line: int) -> str:
    return f"{os.path.relpath(path, os.path.dirname(BTB_SRC))}:{line}"


def _load_list(node: ast.AST) -> ast.expr | None:
    """the value of an assignment to `KERNELS` (native.py's load list), else None"""
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in node.targets):
        return node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "KERNELS":
        return node.value
    return None


def launch_patterns(sites: list[Site] | None = None) -> list[re.Pattern[str]]:
    """the names btb's launches can launch (`launch_sites`), each spelling as a pattern"""
    sites = launch_sites() if sites is None else sites
    return [p for _, _, spellings, _ in sites for sp in spellings if (p := _pattern(sp)) is not None]


def unresolved_launches(sites: list[Site] | None = None) -> list[str]:
    """the launches the parse cannot name a kernel of, as `file:line`: a launch whose kernel the cert cannot read
    certifies nothing, so it is a gap of its own rather than a wildcard over every name"""
    sites = launch_sites() if sites is None else sites
    return [_where(path, line) for path, line, spellings, _ in sites if any(_pattern(sp) is None for sp in spellings)]


def uncounted_launches(sites: list[Site] | None = None) -> list[str]:
    """the launches whose argument list the parse cannot count, as `file:line`"""
    sites = launch_sites() if sites is None else sites
    return [_where(path, line) for path, line, _, counts in sites if counts is None]


def arity_mismatches(sites: list[Site] | None = None, arity: dict[str, int] | None = None) -> list[str]:
    """the launches that can hand a kernel they name a list of another length than its parameters, as
    `file:line kernel` with the counts: each kernel a launch's spellings match must take one of the lengths its list
    can have (a launch choosing between kernels and argument lists - a row map or none - is held to the union, so a
    dropped argument that no choice makes up for is caught)"""
    sites = launch_sites() if sites is None else sites
    arity = defined_arity() if arity is None else arity
    out = []
    for path, line, spellings, counts in sites:
        if counts is None:
            continue
        pats = [p for sp in spellings if (p := _pattern(sp)) is not None]
        for k in sorted(k for k in arity if any(p.fullmatch(k) for p in pats)):
            if arity[k] not in counts:
                got = "/".join(str(c) for c in sorted(counts))
                out.append(f"{_where(path, line)} {k} (declares {arity[k]}, handed {got})")
    return out


def launched_kernels(loaded: set[str] | None = None) -> set[str]:
    """the loaded kernels some launch in btb/ can launch (`launch_patterns`)"""
    loaded = loaded_kernels() if loaded is None else loaded
    pats = launch_patterns()
    return {k for k in loaded if any(p.fullmatch(k) for p in pats)}


def _findings() -> list[tuple[Missing, str]]:
    """the single classified list of gaps as (kind, kernel name); gaps() and render_missing() both derive from it."""
    arity = defined_arity()
    loaded, defined = loaded_kernels(), set(arity)
    sites = launch_sites()
    pats = launch_patterns(sites)
    launched = {k for k in loaded if any(p.fullmatch(k) for p in pats)}
    out = [(Missing.DEFINED_NOT_LOADED, k) for k in sorted(defined - loaded)]
    out += [(Missing.LOADED_NOT_DEFINED, k) for k in sorted(loaded - defined)]
    out += [(Missing.LOADED_NOT_LAUNCHED, k) for k in sorted(loaded - launched)]
    out += [(Missing.LAUNCH_UNRESOLVED, s) for s in unresolved_launches(sites)]
    out += [(Missing.LAUNCH_ARGS_UNRESOLVED, s) for s in uncounted_launches(sites)]
    out += [(Missing.LAUNCH_ARITY, s) for s in arity_mismatches(sites, arity)]
    return out


def gaps() -> list[tuple[str, str]]:
    """(kernel, what) for each drift between the loaded set and the defined set."""
    return [(s, MISSING[k][0].format(s=s)) for k, s in _findings()]


def coverage() -> list[tuple[int, int, str]]:
    """(certified, total, what) for the comment's covered summary"""
    loaded, defined = loaded_kernels(), defined_kernels()
    return [(len(loaded & defined), len(loaded | defined), "card kernels both loaded by the engine and defined")]


def render_missing() -> list[str]:
    """the gaps in plain language - what is absent and how to close each - the agent-facing to-do a skill wraps."""
    items = _findings()
    if not items:
        return ["no CUDA kernel gaps: every loaded kernel is defined and every defined kernel is loaded."]
    lines = [f"{len(items)} CUDA kernel gaps:", ""]
    for kind, subject in items:
        what, how = MISSING[kind]
        lines.append(f"[{kind.value}] {what.format(s=subject)}")
        lines.append(f"    to close: {how.format(s=subject)}")
        lines.append("")
    return lines


def report() -> int:
    loaded, defined = loaded_kernels(), defined_kernels()
    print(f"CUDA card-kernel matrix: {len(loaded)} loaded, {len(defined)} defined, {len(loaded & defined)} matched")
    for k in sorted(loaded & defined):
        print(f"  ok   {k}")
    print()
    print("\n".join(render_missing()))
    return 0


def check() -> int:
    g = gaps()
    if g:
        print(f"FAIL: {len(g)} CUDA kernel gaps", file=sys.stderr)
        for subject, what in g:
            print(f"  {subject}: {what}", file=sys.stderr)
    return 1 if g else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--report", action="store_true", help="print the kernel matrix and the plain-language gaps (default)"
    )
    ap.add_argument(
        "--missing", action="store_true", help="print only the gaps in plain language: what is missing and how to close"
    )
    ap.add_argument("--check", action="store_true", help="exit nonzero when loaded and defined sets disagree")
    a = ap.parse_args(argv)
    if a.check:
        return check()
    if a.missing:
        print("\n".join(render_missing()))
        return 0
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
