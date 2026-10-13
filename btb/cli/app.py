# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The `btb` entry point: builds the parser, sets up the run, calls the command and reports failures."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from typing import Any

from ..options import BadValue, Device, OptionError, env_help
from . import bench, chat, devices, pack, run, serve
from .console import banner, license_notice, say, say_err, set_title, utf8_streams
from .model import require_deps

COMMANDS = (run, chat, bench, serve, pack, devices)  # each adds its own subcommands, in help order
MODEL_FREE = ("pack", "devices")  # no title, banner or native-library notice for these


def parser() -> tuple[argparse.ArgumentParser, Any]:
    """Build the `btb` parser. Returns it and its subparsers."""
    ap = argparse.ArgumentParser(
        prog="btb",
        description="Run big models on small machines, exactly.",
        epilog=env_help(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=False, metavar="command")
    for c in COMMANDS:
        c.add(sub)
    return ap, sub


def _prepare(a: argparse.Namespace) -> None:
    """Set up before the command runs: tracing, the dependency check, the device, the title, the banner and the
    notices. Nothing goes to stdout under --json."""
    if int(getattr(a, "verbose", 0) or 0) >= 2:
        from .. import trace

        trace.enable()  # -vv: every automatic decision as it is taken (btb/trace.py)
    if a.cmd != "devices":  # devices reports the hardware; it needs no model stack
        require_deps(getattr(a, "device", None))
    if getattr(a, "device", None) is not None and a.device.kind is Device.CPU:
        from .. import cpu_only

        cpu_only()
    as_json = getattr(a, "as_json", False)
    if as_json:
        a.quiet = True
    quiet = as_json or getattr(a, "quiet", False)
    from ..confirm import set_assume_yes

    set_assume_yes(getattr(a, "confirm", False))
    to_err = a.cmd == "run"  # run's stdout carries the model's text alone; the rest goes to stderr
    if a.cmd not in MODEL_FREE:
        set_title(a.path or "(btb)", "openai" if a.cmd == "serve" else a.cmd)  # serve speaks the OpenAI API
        if not quiet:
            banner(err=to_err)
        if getattr(a, "native", None) is None and not quiet:
            from .. import native_path

            if native_path() is None:
                (say_err if to_err else say)("[btb] no native library found; torch fallback")
    if not as_json:  # stderr, past --quiet, but never in a JSON run's machine output
        license_notice()


def _call(a: argparse.Namespace, sub: Any, cmdline: list[str]) -> int:
    """Run the command. A bad value prints one line naming its flag; a missing model prints a short report; a crash
    prints the traceback and an issue report. Under --json, stdout gets null on any failure."""
    from .. import feedback

    as_json = getattr(a, "as_json", False)
    err: BaseException | None = None
    try:
        rc = a.fn(a)
    except KeyboardInterrupt:
        return 130
    except OptionError as e:  # a value btb cannot take: the line, no traceback; an option named by its flag
        msg = str(e)
        if isinstance(e, BadValue):
            for act in sub.choices[a.cmd]._actions:
                if act.option_strings and act.dest == e.name:
                    msg = e.shown_as(act.option_strings[-1])
                    break
        say_err(f"[btb] {msg}")
        if as_json:
            say("null")
        return 2
    except FileNotFoundError as e:
        model = getattr(a, "path", None)
        path = str((e.args[0] if e.args else None) or getattr(e, "filename", None) or "")
        if model is None or path != str(model):
            raise  # a missing file deeper in a command is a real fault; keep its traceback
        err = e
        say_err(f"[btb] {feedback.not_found_report(e, model)}")
        if as_json:
            say("null")
        return 1
    except SystemExit:
        raise
    except BaseException as e:
        # any other failure: the traceback, then the report for an issue (with the profile written first, so the
        # report carries the expert-store summary where a trace was kept)
        err = e
        traceback.print_exc()
        sm = getattr(a, "_sm", None)
        summary = None
        if getattr(a, "profile", None) and sm is not None:
            feedback.write_profile(a, sm, say_err)
            summary = feedback.expert_summary(a.profile)
        say_err(feedback.crash_notice(e))
        say_err(feedback.issue_report(e, a, sm, cmdline, getattr(a, "profile", None), summary))
        if as_json:
            say("null")
        return 2 if feedback.is_oom(e) else 1
    finally:
        sm = getattr(a, "_sm", None)
        if err is None and getattr(a, "profile", None) and sm is not None:
            feedback.write_profile(a, sm, say_err)
    return int(rc or 0)


def main(argv: Sequence[str] | None = None) -> int:
    utf8_streams()
    ap, sub = parser()
    cmdline: list[str] = list(argv) if argv is not None else sys.argv[1:]
    a, rest = ap.parse_known_args(argv)
    if a.cmd is None:  # no subcommand: print the help, as `btb --help` does
        ap.print_help()
        return 0
    if a.cmd == "ollama":
        a.args = rest  # passed to `ollama run` after the model name
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    _prepare(a)
    return _call(a, sub, cmdline)
