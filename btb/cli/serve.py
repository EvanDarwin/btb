# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The server commands (see btb/serve.py): `serve` for the OpenAI and Ollama APIs, `ollama` to open the server in
Ollama, and `pi` to register it with the pi coding agent."""

from __future__ import annotations

import argparse
import os
from typing import Any

from . import args
from .console import say
from .model import confirm_or_exit, log_for


def _server_options(p: argparse.ArgumentParser, port: int, port_help: str) -> None:
    """Add the options shared by serve, ollama and pi."""
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind (default 127.0.0.1; 0.0.0.0 for every interface, which needs --api-key)",
    )
    p.add_argument("--port", type=args.port, default=port, help=port_help)
    p.add_argument(
        "--new", type=args.positive, metavar="N", default=None, help="server-side ceiling on new tokens per request"
    )
    p.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        default=os.environ.get("BTB_API_KEY") or None,
        help="require this key on every request (Authorization: Bearer KEY); default: BTB_API_KEY from the "
        "environment, else none - set one before binding beyond 127.0.0.1",
    )
    g = p.add_argument_group("model discovery (serve and ollama list every complete model, loaded on request)")
    g.add_argument(
        "--models-dir",
        dest="models_dir",
        action="append",
        type=args.existing_dir,
        metavar="DIR",
        default=None,
        help="also offer the model(s) under DIR (a model directory, or a directory of them); repeatable. "
        "The Hugging Face cache is always scanned.",
    )
    g.add_argument(
        "--models-filter",
        dest="models_filter",
        type=args.regex,
        metavar="REGEX",
        default=None,
        help="only offer models whose name or repo id matches REGEX (the launch model is always offered)",
    )


def add(sub: Any) -> None:
    p = sub.add_parser(
        "serve",
        help="OpenAI- and Ollama-compatible server",
        description="Serve the OpenAI API (POST /v1/chat/completions, streaming and not; GET /v1/models) "
        "and the Ollama API (POST /api/chat, /api/generate, /api/show; GET /api/tags). "
        "An answer runs until the model ends its turn or the context window is full "
        "unless the request sets max_tokens (OpenAI) or options.num_predict (Ollama).",
    )
    args.model_options(p, path_required=False)
    _server_options(p, 8000, "port to listen on (default 8000)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser(
        "ollama",
        help="serve the model and open it in the Ollama CLI or desktop app",
        description="Start the server, point the Ollama CLI at it (OLLAMA_HOST) and run `ollama run <name>`; "
        "the server stops when the CLI exits. --gui opens the Ollama desktop app pointed at "
        "this server instead. Without the ollama command on PATH the server "
        "stays up and prints the line to run elsewhere.",
        epilog="Anything btb does not recognize is passed to `ollama run` after the model name, e.g. a one-shot "
        "prompt in quotes.",
    )
    args.model_options(p, path_required=False)
    _server_options(p, 11435, "port for the server (default 11435, clear of a local Ollama)")
    p.add_argument(
        "--gui",
        action="store_true",
        help="open the Ollama desktop app pointed at this server (OLLAMA_HOST) instead of the terminal chat",
    )
    p.set_defaults(fn=cmd_ollama)

    p = sub.add_parser(
        "pi",
        help="serve the model and register it with the pi coding agent (pi.dev)",
        description="Start the OpenAI-compatible server (with tool calling) and register a 'btb' provider in "
        "pi's ~/.pi/agent/models.json pointing at it; then run `pi --provider btb --model <name>` in another "
        "terminal. The server stays up until Ctrl-C. --no-configure prints the provider entry instead of "
        "writing the file; --config writes it elsewhere.",
    )
    args.model_options(p, path_required=False)
    _server_options(p, 8000, "port to listen on (default 8000)")
    p.add_argument(
        "--no-configure",
        dest="configure",
        action="store_false",
        help="print the provider entry instead of writing ~/.pi/agent/models.json",
    )
    p.add_argument(
        "--config",
        dest="pi_config",
        default=None,
        metavar="PATH",
        help="write the provider to this file instead of ~/.pi/agent/models.json",
    )
    p.set_defaults(fn=cmd_pi)


def _server_args(a: argparse.Namespace) -> dict[str, Any]:
    """The keyword arguments shared by serve, ollama and pi. Asks before downloading the model."""
    confirm_or_exit(a.path)
    kw, nat = args.load_args(a)
    return {
        "host": a.host,
        "port": a.port,
        "device": a.device,
        "max_new": a.new,
        "log": log_for(a, say),
        "native": nat,
        "extra_paths": a.models_dir or (),
        "pattern": a.models_filter,
        "api_key": a.api_key,
        **kw,
    }


def cmd_serve(a: argparse.Namespace) -> Any:
    from ..serve import serve

    kw = _server_args(a)
    if not a.draft_model:
        from ..hf import draft_notice

        draft_notice(a.path)
    return serve(a.path, **kw)


def cmd_ollama(a: argparse.Namespace) -> Any:
    from ..serve import ollama

    return ollama(a.path, args=a.args, gui=a.gui, **_server_args(a))


def cmd_pi(a: argparse.Namespace) -> Any:
    from ..serve import pi

    return pi(a.path, configure=a.configure, config_path=a.pi_config, **_server_args(a))
