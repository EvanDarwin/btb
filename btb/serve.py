# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
from __future__ import annotations

import collections
import contextlib
import gc
import hmac
import ipaddress
import json
import math
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

from . import available_models, load, model_stem, resolve, serve_name
from .draft import SpanBank
from .engine.device import resolve_device
from .kinds import Json, Log, Tokens
from .options import BadValue, DeviceName, OptionError
from .sampling import Sampling
from .session import Session
from .sysinfo import host_total_bytes
from .text import Channels, TextStream, answer, probe_tail
from .tools import ToolFormat, tool_format

if TYPE_CHECKING:
    from .engine.model import StreamedTextModel
    from .hf import ModelEntry
    from .kinds import FamilyKind

# a chat message (role, content, optional tool fields) and a decoded request body (OpenAI or Ollama JSON)
Message = Json
Request = Json


class Engine:
    bank: SpanBank
    eos: tuple[int, ...]
    family: FamilyKind
    footprint: int
    max_new: int | None
    name: str
    path: str
    session: Session
    sm: StreamedTextModel
    tok: Any

    def __init__(
        self,
        path: str,
        device: str | None = None,
        max_new: int | None = None,
        log: Log | None = None,
        name: str | None = None,
        **kw: Any,
    ) -> None:
        """One served model: the engine loaded from `path` (as `btb.load` takes it, `**kw` its options) under the
        served `name`, with the session and span bank the requests share. `run(messages, max_new, ...)` decodes
        one request, `text(tokens)` renders it."""
        self.path = resolve(path)
        self.name = serve_name(name or model_stem(path))
        # what this engine has answered and been asked, for the n-gram proposer of the next requests
        self.bank = SpanBank(int(kw.pop("bank_tokens", 1 << 20)))
        self.sm = load(self.path, device=device, log=log, **kw)
        self.tok = self.sm.tokenizer
        self.eos = self.sm.stop_ids
        self.max_new = max_new
        # the template's generation tail known up front: the first follow-up turn resumes from its snapshot
        self.session = Session(tail=probe_tail(self.tok))
        self.family = self.sm.fam.kind
        self.sampling: Sampling = self.sm.sampling  # the default a request's own fields override
        self.footprint = 0

    def ids_for(self, messages: Sequence[Message], tools: Any = None) -> list[int]:
        return self.sm.prompt_ids(messages, tools=tools)

    def run(
        self,
        messages: Sequence[Message],
        max_new: int | None,
        on_token: Callable[[int], Any] | None = None,
        ids: Tokens | None = None,
        sampling: Sampling | None = None,
    ) -> Any:
        if ids is None:
            ids = self.ids_for(messages)
        if max_new is None:
            max_new = self.max_new
        elif self.max_new is not None:
            max_new = min(int(max_new), int(self.max_new))  # --new is a ceiling: a request asks for less, not more
        # past the window there is nothing to decode into, so a cache is never sized for it
        room = self.sm.window - len(ids)
        if max_new is not None and room > 0:
            max_new = min(max_new, room)
        out, c = self.sm.generate(
            ids,
            max_new,
            eos=self.eos,
            session=self.session,
            on_token=on_token,
            spans=self.bank,
            sampling=sampling if sampling is not None else self.sampling,
        )
        es = set(self.eos)
        toks = [t for t in out if t not in es]
        self.bank.add("prompt", ids)
        self.bank.add("answer", toks)
        return ids, toks, c

    def text(self, toks: Tokens) -> tuple[str, str]:
        """(answer, reasoning) of a generated token list: gpt-oss's final channel and its analysis, every
        other family's whole output and ''."""
        return answer(self.tok, toks)


class ModelRegistry:
    """Every servable model on the machine, loaded on demand within a memory budget (least-recently-used unloaded
    to make room; loaded anyway when nothing else is left). The launch model is eager; `pattern` hides the rest."""

    _scan_t: float
    budget: float
    device: DeviceName
    entries: dict[str, ModelEntry]
    extra_paths: list[str]
    kw: Json
    loaded: collections.OrderedDict[str, Engine]
    lock: threading.Lock
    log: Log | None
    max_new: int | None
    pattern: str | None
    primary_arg: str | None
    primary_name: str | None
    primary_path: str | None
    reserve: float

    def __init__(
        self,
        primary_path: str | None,
        device: str | None = None,
        max_new: int | None = None,
        log: Log | None = None,
        extra_paths: Sequence[str] = (),
        pattern: str | None = None,
        reserve_gb: float | None = None,
        **kw: Any,
    ) -> None:
        self.device = resolve_device(device)  # resolved once, a bad name failing here before anything loads
        self.max_new = max_new
        self.log = log
        self.kw = kw
        # the launch model named from the user's argument, not the resolved snapshot directory (a commit hash), so it
        # matches its cache entry; `primary_path` None: nothing loaded until the first request
        self.primary_arg = primary_path
        if primary_path is None:
            self.primary_name = None
            self.primary_path = None
        else:
            self.primary_name = serve_name(model_stem(primary_path))
            # the server serves what is on the machine: a model to download, or a 12-bit model whose parent is
            # not here, is refused rather than fetched under a request
            self.primary_path = resolve(primary_path, local=True)
            from .hf import local_parent, pack_format

            fmt = pack_format(self.primary_path)
            if fmt is not None and local_parent(self.primary_path) is None:
                raise FileNotFoundError(f"{primary_path}: its parent {fmt['source']} is not on this machine")
        self.extra_paths = list(extra_paths)
        self.pattern = pattern
        total = host_total_bytes()  # RAM (unified memory on a Mac); the host-tier budget, distinct from VRAM
        self.reserve = (reserve_gb * 1e9) if reserve_gb is not None else max(4e9, total * 0.15)
        self.budget = max(1, total - self.reserve)
        self.lock = threading.Lock()
        # `state` guards `loaded` itself: the GET routes read it while a request holding `lock` loads or evicts
        self.state = threading.Lock()
        self.loaded = collections.OrderedDict()
        self.entries = {}
        self._scan_t = 0.0
        self.refresh(force=True)

    def refresh(self, force: bool = False) -> dict[str, ModelEntry]:
        """Rescan the cache and the extra paths (throttled to every few seconds); the launch model is always
        present even before its files finish downloading."""
        if not force and time.time() - self._scan_t < 5.0:
            return self.entries
        paths = ([self.primary_arg] if self.primary_arg else []) + self.extra_paths
        ents = {e["name"]: e for e in available_models(paths, self.pattern)}
        if self.primary_name and self.primary_name not in ents:
            from . import _model_bytes, _model_type, is_packed

            assert self.primary_path is not None  # primary_name is set with primary_path

            ents[self.primary_name] = {
                "name": self.primary_name,
                "repo": self.primary_path,
                "path": self.primary_path,
                "type": _model_type(self.primary_path),
                "size": _model_bytes(self.primary_path),
                "packed": is_packed(self.primary_path),
            }
        self.entries = ents
        self._scan_t = time.time()
        return ents

    def resolve_name(self, name: str | None) -> str | None:
        """Map a request's model field (a served name, with or without a `:tag`, or the repo id or path of a
        model the server discovered) to an entry name, or None. A path the server did not discover is None: what
        is servable is the launch model, the cache and --models-dir, never a directory a request names."""
        if not name:
            return self.primary_name or (next(iter(sorted(self.entries))) if self.entries else None)
        key = str(name).split(":", 1)[0]
        if key in self.entries:
            return key
        s = serve_name(key)
        if s in self.entries:
            return s
        for n, e in self.entries.items():
            if key in (e["repo"], e["path"]) or serve_name(os.path.basename(str(e["repo"]).rstrip("/"))) == s:
                return n
        return None

    @property
    def card(self) -> bool:
        """whether the serving device is a CUDA card (VRAM to price and reclaim)"""
        d = DeviceName.parse(getattr(self, "device", None))
        return d is not None and d.kind.card

    def _evict_lru(self, why: str) -> None:
        with self.state:
            name, eng = next(iter(self.loaded.items()))
            self.loaded.pop(name)
        with contextlib.suppress(Exception):
            eng.sm.close()
        if self.log:
            self.log(f"[serve] unloaded {name} ({eng.footprint / 1e9:.1f} GB) to free {why}")
        # close() drops the model's tensors, but the engine's own reference cycles (its captured graphs, the
        # scheduler, the module hooks) hold the VRAM until they are collected. Drop them, then reclaim the card
        # the way the engine's own `vram_trim` does - synchronize, then empty_cache - so `_card_free` (the same
        # `free_bytes` read the scheduler and every placement price the device by) sees the room freed before the
        # next model is planned.
        del eng
        gc.collect()
        if self.card:
            with contextlib.suppress(Exception):
                import torch

                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def _card_free(self) -> int | None:
        """VRAM free on the serving device right now, or None off a CUDA card (host/unified memory is bounded by
        the RAM budget instead, not this)."""
        if not self.card:
            return None
        import torch

        from .engine.device import free_bytes

        try:
            return free_bytes(torch.device(str(self.device)))
        except Exception:
            return None

    def _make_room(self, need: int) -> None:
        """Unload least-recently-used models to fit the incoming one. Two budgets that coexist: the host RAM the
        tiered layers and expert stores live in, and — for a card device — the VRAM the new model needs now, so a
        model no longer in use never holds the card while another is prompted. `need` is the model's weight bytes,
        a safe over-estimate of its card tier; VRAM eviction stops as soon as the card has room, so models that
        fit together stay resident."""
        while self.loaded and sum(e.footprint for e in self.loaded.values()) + need > self.budget:
            self._evict_lru("RAM")
        while self.loaded:
            free = self._card_free()
            if free is None or free >= need:
                break
            self._evict_lru("VRAM")

    def acquire(self, name: str | None) -> Engine:
        """The loaded Engine for `name`, loading it (and unloading others to fit) if need be. Call under `lock`."""
        n = self.resolve_name(name)
        if n is None:
            raise KeyError(name)
        with self.state:
            if n in self.loaded:
                self.loaded.move_to_end(n)
                return self.loaded[n]
        e = self.entries[n]
        self._make_room(e["size"])
        if self.log:
            self.log(f"[serve] loading {n} ({e['size'] / 1e9:.1f} GB)")
        eng = Engine(
            e["path"],
            name=n,
            device=str(self.device),
            max_new=self.max_new,
            log=self.log,
            **self.kw,
        )
        eng.footprint = e["size"] or 0
        with self.state:
            self.loaded[n] = eng
        return eng

    def loaded_names(self) -> list[str]:
        """the loaded models' names, a snapshot (safe without `lock`)"""
        with self.state:
            return list(self.loaded)

    def loaded_get(self, name: str) -> Engine | None:
        with self.state:
            return self.loaded.get(name)

    def close(self) -> None:
        """every engine closed once no request runs: a request in flight (a Ctrl-C mid-answer, the client gone)
        is told to stop and its handler, which holds `lock`, ends at the next step"""
        for eng in list(self.loaded.values()):
            eng.sm.abort.set()
        if self.lock.locked() and self.log:
            self.log("[serve] a request is in flight: it stops at its next step before the model closes")
        with self.lock:
            for eng in self.loaded.values():
                with contextlib.suppress(Exception):
                    eng.sm.close()
            self.loaded.clear()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _details(family: str) -> Json:
    return {
        "parent_model": "",
        "format": "safetensors",
        "family": family,
        "families": [family],
        "parameter_size": "",
        "quantization_level": "bf16",
    }


def _tagname(entry: Mapping[str, Any]) -> str:
    """The Ollama tag: `name:latest` (a 12-bit model's name carries `-pack12`)."""
    return f"{entry['name']}:latest"


def _tag(entry: Mapping[str, Any], loaded: bool) -> Json:
    t = _tagname(entry)
    return {
        "name": t,
        "model": t,
        "modified_at": _now(),
        "size": int(entry.get("size") or 0),
        "digest": "",
        "details": _details(entry.get("type") or ""),
        "loaded": bool(loaded),
    }


MAX_BODY = 16 << 20  # a request body past this is refused (413) before it is read
MAX_CONNECTIONS = 64  # handler threads at once; past it a connection is answered 503 and closed at once
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"})


def _public(host: str) -> bool:
    """whether a peer is on the public internet (an IPv4-mapped IPv6 address judged as its IPv4); an address that
    does not parse is taken as public, so the check fails closed"""
    try:
        a = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return True
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        a = a.ipv4_mapped
    return a.is_global


KEY_REQUIRED = "An API_KEY must be provided"
_KEY_BODY = json.dumps({"error": KEY_REQUIRED}).encode()
_KEYLESS_REFUSAL = (
    b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s"
    % (len(_KEY_BODY), _KEY_BODY)
)
DRAIN_S = 0.5  # how long a turned-away connection's remaining request is read and discarded before it closes
_DRAINS = threading.BoundedSemaphore(16)  # turned-away connections draining at once


def _drain(request: socket.socket) -> None:
    """read and discard what a turned-away peer is still sending, until it closes or DRAIN_S is up, then close"""
    deadline = time.monotonic() + DRAIN_S
    try:
        while (left := deadline - time.monotonic()) > 0:
            request.settimeout(left)
            if not request.recv(1 << 16):
                break
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            request.close()
        _DRAINS.release()


class _Reject(Exception):
    """a request refused before its body is parsed: (the status, the line)"""

    def __init__(self, code: int, msg: str) -> None:
        super().__init__(msg)
        self.code = code


def _host_of(value: str) -> str:
    """the host part of a Host or Origin header value, lowercased, the port and a scheme dropped"""
    h = value.strip().lower()
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/", 1)[0]
    if h.startswith("["):
        return h[1 : h.find("]")] if "]" in h else h
    if h.count(":") == 1:
        h = h.rsplit(":", 1)[0]
    return h


class _Server(ThreadingHTTPServer):
    # worker threads die with the process, so shutdown never waits on an in-flight request; the address is
    # reusable so a restart binds at once
    daemon_threads = True
    allow_reuse_address = True
    reg: ModelRegistry
    api_key: str | None = None  # every request must carry it as a bearer token when set

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self.loopback = str(self.server_address[0]) in LOOPBACK - {"0.0.0.0", "::"}

    def process_request(self, request: Any, client_address: Any) -> None:
        # a keyless server refuses a public peer (a port forwarded to a private bind) before it holds a thread
        if not self.api_key and _public(str(client_address[0])):
            self._turn_away(request, _KEYLESS_REFUSAL)
            return
        # a connection past the cap is answered at once and dropped, so a flood cannot pile up threads while the
        # one model answers one request at a time
        if not self._slots.acquire(blocking=False):
            self._turn_away(
                request, b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n"
            )
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def _turn_away(self, request: socket.socket, response: bytes) -> None:
        """answer a connection before it holds a request thread or a slot, and close it so the peer can read the
        answer: closing over request bytes still arriving resets the connection, and a client mid-body sees a
        broken pipe instead of the status, so the rest is drained first on a short-lived thread"""
        with contextlib.suppress(OSError):
            request.sendall(response)
            request.shutdown(socket.SHUT_WR)
        if not _DRAINS.acquire(blocking=False):
            self.shutdown_request(request)  # draining as many as it takes already: this one closes as it is
            return
        threading.Thread(target=_drain, args=(request,), daemon=True, name="turn-away").start()

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request: Any, client_address: Any) -> None:
        # a client that hangs up mid-request (the Ollama CLI on exit, a browser navigating away) trips a
        # ConnectionResetError/BrokenPipeError in the socket read; that is normal, not a fault to dump
        e = sys.exc_info()[1]
        if isinstance(e, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def _content_text(content: Any) -> str:
    """A message's content as the templates take it. The OpenAI clients (pi among them) send content as an array
    of typed parts - `[{"type": "text", "text": "..."}]` - which a chat template renders as a list's repr, not
    the words; join the text parts back into a string. A plain string (or None on a tool-call turn) passes through."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content


class _ToolGate:
    """The streamed prose of a tool-calling turn: pieces go out as they decode until a call's opener, from which
    the text is buffered (a partial call never leaks as content); `finish` emits the prose that followed the
    call once the blocks are known and struck."""

    def __init__(self, fmt: ToolFormat, emit: Callable[..., Any], tools: Any = None) -> None:
        self.fmt, self.emit, self.tools = fmt, emit, tools
        self.buf, self.sent, self.in_call = "", 0, False

    def push(self, delta: str) -> Any:
        self.buf += delta
        if self.in_call:
            return True
        p = self.fmt.opener_at(self.buf, self.sent)
        if p is not None:
            self.in_call = True
            out, self.sent = self.buf[self.sent : p], p
            return self.emit(out, "content") if out else True
        upto = len(self.buf) - self.fmt.holdback(self.buf)
        if upto > self.sent:
            out, self.sent = self.buf[self.sent : upto], upto
            return self.emit(out, "content")
        return True

    def finish(self) -> Any:
        """the prose after the last call, and whatever a holdback kept, with the call blocks struck"""
        prose = (
            self.fmt.strip(self.buf) if self.fmt.calls(self.buf, self.tools) else self.buf
        )  # struck where calls parsed
        sent = self.buf[: self.sent]
        tail = prose[len(sent) :] if prose.startswith(sent) else ""
        if self.in_call:
            tail = tail.lstrip()
        return self.emit(tail, "content") if tail.strip() else True


def _short(v: Any) -> str:
    """a request's value as a message shows it: at most 120 characters"""
    s = str(v)
    return s if len(s) <= 120 else s[:117] + "..."


def _messages(req: Request) -> list[Message]:
    """the request's messages: a list of objects, each role a string and each content a string, null or a list
    of parts; else an OptionError (a string or a bare object is the usual slip)"""
    ms = req.get("messages")
    if ms is None:
        return []
    if not isinstance(ms, list) or not all(isinstance(m, dict) for m in ms):
        raise BadValue("messages", "a list of {role, content} objects is expected")
    for m in ms:
        if not isinstance(m.get("role", "user"), str):
            raise BadValue("messages[].role", "a string", m.get("role"))
        c = m.get("content")
        if (
            c is not None
            and not isinstance(c, str)
            and not (isinstance(c, list) and all(isinstance(p, dict) for p in c))
        ):
            raise BadValue("messages[].content", "a string or a list of {type, text} parts", c)
    return ms


def _text(req: Request, name: str, default: str | None = None) -> str | None:
    """a request's string field under `name`: absent or null the default; anything but a string an OptionError"""
    v = req.get(name)
    if v is None:
        return default
    if not isinstance(v, str):
        raise BadValue(name, "a string", v)
    return v


def _object(req: Request, name: str) -> dict[str, Any]:
    """a request's object field under `name`: absent or null {}; anything but an object an OptionError"""
    v = req.get(name)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise BadValue(name, "an object is expected", v)
    return v


def _flag(req: Request, name: str, default: bool) -> bool:
    """a request's boolean under `name`: true/false (or 0/1); a string is a BadValue ("false" is not false)"""
    v = req.get(name, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    raise BadValue(name, "true or false", v)


def _count(req: Request, name: str) -> int | None:
    """a request's token cap under `name`: absent or null none; a whole number above 0; else an OptionError"""
    n = req.get(name)
    if n is None:
        return None
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n != int(n) or int(n) < 1:
        raise BadValue(name, "a whole number above 0", n)
    return int(n)


def _for_template(messages: Sequence[Message]) -> list[Message]:
    """Incoming OpenAI messages shaped for a chat template: array content parts flattened to text, a tool-call
    turn's null content made the empty string the templates concatenate and test with `in`, and a tool call's
    arguments - a JSON string on the wire - decoded to the object the tool templates render."""
    out: list[Message] = []
    for m in messages:
        m = dict(m)
        if "content" in m:
            m["content"] = _content_text(m["content"])
            if m["content"] is None:
                m["content"] = ""
        tcs = m.get("tool_calls")
        if tcs:
            if not isinstance(tcs, list) or not all(isinstance(tc, dict) for tc in tcs):
                raise BadValue("tool_calls", "a list of objects is expected")
            fixed = []
            for tc in tcs:
                if not isinstance(tc.get("function") or {}, dict):
                    raise BadValue("tool_calls[].function", "an object is expected", tc.get("function"))
                fn = dict(tc.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    with contextlib.suppress(ValueError):
                        fn["arguments"] = json.loads(a)
                fixed.append({**tc, "function": fn})
            m["tool_calls"] = fixed
        out.append(m)
    return out


class Handler(BaseHTTPRequestHandler):
    """The OpenAI and Ollama routes over the server's registry (`self.reg`)."""

    server: _Server
    timeout = 120  # a socket idle this long (a request that never arrives, a client that stops reading) is dropped

    @property
    def reg(self) -> ModelRegistry:
        return self.server.reg

    protocol_version = "HTTP/1.1"

    def _guard(self) -> None:
        """what every request must pass before it is read: on a loopback bind the Host header must name this
        machine (a page on the internet resolving its own name to 127.0.0.1 - DNS rebinding - otherwise reaches
        the server as if it were local); a browser's Origin, when sent, must be local too; and the API key,
        when the server has one, must come as a bearer token"""
        host = self.headers.get("Host")
        if self.server.loopback and host and _host_of(host) not in LOOPBACK | {str(self.server.server_address[0])}:
            raise _Reject(
                403, f"Host {host!r} is not this server; use http://127.0.0.1:{self.server.server_address[1]}"
            )
        origin = self.headers.get("Origin")
        # `Origin: null` (a sandboxed frame, a file:// page) is cross-origin too
        if origin and _host_of(origin) not in LOOPBACK | {str(self.server.server_address[0])}:
            raise _Reject(403, f"Origin {origin!r} is not allowed; the server takes no cross-origin requests")
        key = self.server.api_key
        if key:
            auth = self.headers.get("Authorization") or ""
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if not hmac.compare_digest(given.encode(), key.encode()):
                raise _Reject(401, "an API key is required: Authorization: Bearer <key>")

    def _refuse(self, e: _Reject) -> None:
        self.close_connection = True  # the body was not read: on a kept-alive connection it would be the next request
        if e.code == 401:
            body = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self._send_body(body)
            return
        self._json(e.code, {"error": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[serve] " + (fmt % args) + "\n")

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode("utf-8")
        self._headed = True
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._send_body(body)

    def _send_body(self, data: bytes) -> None:
        """the one place a response body is written; a HEAD's is dropped (RFC 9110 9.3.2)"""
        if self.command != "HEAD":
            self.wfile.write(data)

    def _body(self) -> Any:
        """the JSON body: a missing or bad Content-Length is a 400, one past MAX_BODY a 413 (before a byte is
        read); anything but valid JSON returns None"""
        raw = self.headers.get("Content-Length", "0")
        try:
            n = int(raw)
        except ValueError:
            raise _Reject(400, f"Content-Length {raw!r}: a whole number is expected") from None
        if n < 0:
            raise _Reject(400, f"Content-Length {raw!r}: a whole number is expected")
        if n > MAX_BODY:
            raise _Reject(413, f"the body is {n} bytes; at most {MAX_BODY} are taken")
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return None

    def _route(self) -> Any:
        return self.path.split("?", 1)[0].rstrip("/") or "/"

    def _stream_head(self, content_type: str) -> None:
        self._headed = True
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write(self, s: str) -> bool:
        try:
            self._send_body(s.encode("utf-8"))
            self.wfile.flush()
            return True
        except OSError:  # the client is gone, or stopped reading (the socket timeout is an OSError too)
            return False

    def _decode(
        self,
        engine: Engine,
        messages: Sequence[Message],
        max_new: int | None,
        emit: Callable[..., Any],
        ids: Tokens | None = None,
        on_content: Callable[..., Any] | None = None,
        sampling: Sampling | None = None,
    ) -> Any:
        q: Any = queue.Queue()
        es = set(engine.eos)
        res = {}

        def work() -> None:
            try:
                res["out"] = engine.run(messages, max_new, on_token=lambda t: q.put(int(t)), ids=ids, sampling=sampling)
            except Exception as e:
                res["err"] = repr(e)
            finally:
                q.put(None)

        th = threading.Thread(target=work, daemon=True)
        th.start()
        # the answer and the reasoning (gpt-oss's channels) stream as two texts: `emit(delta, kind)`
        # with kind "content" or "thinking"
        ch = Channels(engine.tok)
        streams = {"content": TextStream(engine.tok), "thinking": TextStream(engine.tok)}
        t_first = None
        alive = True
        try:
            while True:
                t = q.get()
                if t is None:
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                if t in es or not alive:
                    continue
                kind = ch.push(t)
                if kind is None or kind == "call":  # a harmony call's body is never content
                    continue
                delta = streams[kind].push(t)
                if delta:
                    if kind == "content" and on_content is not None:
                        alive = on_content(delta)
                    else:
                        alive = emit(delta, kind)
                    if not alive:
                        # the client is gone: the model stops at its next step instead of finishing the answer
                        # while the next request waits on the lock; the session keeps what was decoded
                        engine.sm.abort.set()
            for kind, st in streams.items():
                tail = st.flush()
                if tail and alive:
                    if kind == "content" and on_content is not None:
                        alive = on_content(tail)
                    else:
                        alive = emit(tail, kind)
        except BaseException:
            # a failure `_write` does not absorb, or a callback's: the decode is stopped and joined below before
            # the lock is released, or the next request would run on an engine still decoding
            engine.sm.abort.set()
            alive = False
            raise
        finally:
            th.join()
            if not alive:
                engine.sm.abort.clear()
        if "err" in res:
            raise RuntimeError(res["err"])
        ids_used, toks, c = res["out"]
        return ids_used, toks, c, t_first

    def do_GET(self) -> None:
        try:
            self._guard()
        except _Reject as e:
            return self._refuse(e)
        r = self._route()
        if r == "/v1/models":
            ents = self.reg.refresh()
            return self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": e["name"], "object": "model", "owned_by": "btb"} for e in ents.values()],
                },
            )
        if r == "/api/tags":
            ents = self.reg.refresh()
            loaded = set(self.reg.loaded_names())
            return self._json(
                200,
                {"models": [_tag(e, e["name"] in loaded) for e in sorted(ents.values(), key=lambda x: x["name"])]},
            )
        if r == "/api/ps":
            return self._json(
                200,
                {
                    "models": [
                        {**_tag(self.reg.entries.get(n, {"name": n}), True), "expires_at": _now(), "size_vram": 0}
                        for n in self.reg.loaded_names()
                    ]
                },
            )
        if r == "/api/version":
            ua = self.headers.get("User-Agent", "")
            ver = ua.split("/", 1)[1].split(" ", 1)[0] if ua.startswith("ollama/") else "0.0.0"
            return self._json(200, {"version": ver})
        if r in ("/", "/health"):
            return self._json(
                200, {"ok": True, "models": sorted(self.reg.refresh()), "loaded": self.reg.loaded_names()}
            )
        return self._json(404, {"error": "not found"})

    def do_HEAD(self) -> None:
        """GET's answer without its body: `_send_body` drops the bytes, so the status and every header, the
        Content-Length GET would send included, are GET's own"""
        self.do_GET()

    def do_POST(self) -> None:
        r = self._route()
        self._headed = False
        try:
            self._guard()
            req = self._body()
        except _Reject as bad:
            return self._refuse(bad)
        if not isinstance(req, dict):
            return self._json(400, {"error": "bad json: an object is expected"})
        try:
            if r == "/v1/chat/completions":
                return self._openai(req)
            if r == "/api/chat":
                return self._ollama_chat(req)
            if r == "/api/generate":
                return self._ollama_generate(req)
            if r == "/api/show":
                return self._ollama_show(req)
        except OptionError as bad:  # a field of the wrong type or range, caught before the model runs
            return self._json(400, {"error": str(bad)})
        except Exception as err:
            # the engine or a load failed under the request: the failure is logged, and the client gets a status
            # (a stream already headed ends where it is instead of a 200 that never closes)
            self.log_message("%s failed: %s", r, "".join(traceback.format_exception(err)).rstrip())
            if not getattr(self, "_headed", False):
                return self._json(500, {"error": f"{type(err).__name__}; the server's log has the details"})
            return None
        if r in (
            "/api/pull",
            "/api/push",
            "/api/create",
            "/api/copy",
            "/api/delete",
            "/api/embed",
            "/api/embeddings",
        ):
            return self._json(404, {"error": f"{r} is not supported: btb serves the models already on this machine"})
        return self._json(404, {"error": "not found"})

    def _ollama_show(self, req: Request) -> Any:
        """the model's card: its family and name, as the Ollama CLI reads them before a run"""
        self.reg.refresh()
        n = self.reg.resolve_name(_text(req, "model") or _text(req, "name"))
        if n is None:
            return self._json(404, {"error": f"model {_short(req.get('model') or req.get('name'))!r} not found"})
        e = self.reg.entries[n]
        eng = self.reg.loaded_get(n)
        fam = eng.family if eng is not None else (e.get("type") or "")
        return self._json(
            200,
            {
                "modelfile": f"# btb\nFROM {n}\n",
                "parameters": "",
                "template": "",
                "details": _details(fam),
                "model_info": {"general.architecture": fam, "general.basename": n},
                "capabilities": ["completion"],
            },
        )

    def _engine(self, req: Request) -> Engine | None:
        """Resolve the request's model to a loaded Engine, loading/unloading as needed; None on 404 (the
        caller has already written the error). Call under `self.reg.lock`."""
        try:
            return self.reg.acquire(_text(req, "model"))
        except KeyError:
            self._json(
                404,
                {
                    "error": f"model {_short(req.get('model'))!r} is not on this machine; GET /api/tags lists what is available"
                },
            )
            return None

    def _openai(self, req: Request) -> Any:
        messages = _for_template(_messages(req))  # flatten content parts, decode tool-call arguments
        if not messages:
            return self._json(400, {"error": "messages required"})
        tools = req.get("tools") or None
        if tools is not None and (
            not isinstance(tools, list)
            or not all(isinstance(t, dict) and isinstance(t.get("function"), dict) for t in tools)
        ):
            raise BadValue("tools", "a list of {type, function} objects is expected")
        if req.get("tool_choice") == "none":  # the caller asked for prose this turn, not a call
            tools = None
        max_new = _count(req, "max_tokens") or _count(req, "max_completion_tokens")
        stream = _flag(req, "stream", False)
        want_usage = bool(_object(req, "stream_options").get("include_usage"))
        rid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        with self.reg.lock:
            engine = self._engine(req)
            if engine is None:
                return None
            # the family's format shapes the conversation for the template and reads the calls out afterwards
            fmt = tool_format(getattr(engine, "family", None))
            # the request's temperature / top_p / top_k / seed over the engine's default; absent, the default
            smp = Sampling.from_request(req, engine.sampling)
            ids = None
            if tools:
                messages, tools_kw = fmt.prepare(messages, tools)
                ids = engine.ids_for(messages, tools_kw)
            if not stream:
                t0 = time.perf_counter()
                ids, toks, c = engine.run(messages, max_new, ids=ids, sampling=smp)
                self._flex(engine, toks, c, t0, None, time.perf_counter())
                calls: list[Any] = []
                if tools:
                    text, think, calls = fmt.from_tokens(engine.tok, toks, tools)
                else:
                    text, think = engine.text(toks)
                msg: Message = {"role": "assistant", "content": (text or None) if calls else text}
                if think:
                    msg["reasoning_content"] = think
                if calls:
                    msg["tool_calls"] = [
                        {
                            "id": "call_" + uuid.uuid4().hex[:24],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in calls
                    ]
                fin = "tool_calls" if calls else ("stop" if len(toks) < int(c.get("cap", 0)) else "length")
                return self._json(
                    200,
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "created": created,
                        "model": engine.name,
                        "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                        "usage": {
                            "prompt_tokens": len(ids),
                            "completion_tokens": len(toks),
                            "total_tokens": len(ids) + len(toks),
                        },
                    },
                )
            self._stream_head("text/event-stream")

            def send(delta: Json, finish: Any = None) -> bool:
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                return self._write("data: " + json.dumps(chunk) + "\n\n")

            def send_usage(prompt: int, out: int) -> bool:
                # OpenAI's include_usage: a final chunk with no choices carries the token counts
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.name,
                    "choices": [],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out},
                }
                return self._write("data: " + json.dumps(chunk) + "\n\n")

            first = [True]

            def emit(delta: Any, kind: str = "content") -> Any:
                part = {"content": delta} if kind == "content" else {"reasoning_content": delta}
                if first[0]:
                    part = dict({"role": "assistant"}, **part)
                    first[0] = False
                return send(part)

            if tools:
                # the prose streams as it decodes; from a call's opener the text is buffered and parsed whole
                gate = _ToolGate(fmt, emit, tools)
                t0 = time.perf_counter()
                ids2, toks, c, t_first = self._decode(
                    engine, messages, max_new, emit, ids=ids, on_content=gate.push, sampling=smp
                )
                self._flex(engine, toks, c, t0, t_first, time.perf_counter())
                _prose, _think, calls = fmt.from_tokens(engine.tok, toks, tools)
                gate.finish()
                if calls:
                    part: Json = {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": "call_" + uuid.uuid4().hex[:24],
                                "type": "function",
                                "function": {"name": c2["name"], "arguments": c2["arguments"]},
                            }
                            for i, c2 in enumerate(calls)
                        ]
                    }
                    if first[0]:
                        part = dict({"role": "assistant"}, **part)
                        first[0] = False
                    send(part)
                send({}, "tool_calls" if calls else ("stop" if len(toks) < int(c.get("cap", 0)) else "length"))
                if want_usage:
                    send_usage(len(ids2), len(toks))
                return self._write("data: [DONE]\n\n")

            t0 = time.perf_counter()
            ids, toks, c, t_first = self._decode(engine, messages, max_new, emit, sampling=smp)
            self._flex(engine, toks, c, t0, t_first, time.perf_counter())
            send({}, "stop" if len(toks) < int(c.get("cap", 0)) else "length")
            if want_usage:
                send_usage(len(ids), len(toks))
            return self._write("data: [DONE]\n\n")

    def _limit(self, req: Request) -> int | None:
        """Ollama's options.num_predict: a cap above 0; -1 (until the turn ends) and -2 (fill the context) as
        no cap; anything else an OptionError"""
        opts = req.get("options")
        if opts is not None and not isinstance(opts, dict):
            raise BadValue("options", "an object is expected")
        n = (opts or {}).get("num_predict")
        if n is None:
            return None
        if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n != int(n) or int(n) < -2:
            raise BadValue("options.num_predict", "a whole number (above 0 a cap; -1 or -2 no cap)", n)
        return int(n) if int(n) > 0 else None

    def _stats(
        self,
        ids: Tokens,
        toks: Tokens,
        c: Json,
        t0: float,
        t_first: float | None,
        t_end: float,
    ) -> Json:
        ns = lambda s: int(max(0.0, s) * 1e9)
        return {
            "done_reason": "stop" if len(toks) < int(c.get("cap", 0)) else "length",
            "total_duration": ns(t_end - t0),
            "load_duration": 0,
            "prompt_eval_count": len(ids),
            "prompt_eval_duration": ns((t_first or t_end) - t0),
            "eval_count": len(toks),
            "eval_duration": ns(t_end - (t_first or t_end)),
        }

    def _flex(
        self,
        engine: Engine,
        toks: Tokens,
        c: Json,
        t0: float,
        t_first: float | None,
        t_end: float,
    ) -> Any:
        """One line to the console after a request: the generation rate (decode only, the number to show
        off), first-token latency, and tokens per weight pass when speculation ran."""
        n = len(toks)
        if not n:
            return
        gen = t_end - (t_first if t_first is not None else t0)
        tps = n / gen if gen > 0 else 0.0
        bits = [f"{engine.name}: {n} tok, {tps:.1f} tok/s"]
        if t_first is not None:
            bits.append(f"first {t_first - t0:.2f}s")
        fwd = c.get("forwards") if c else None
        if fwd and n / fwd > 1.01:
            bits.append(f"{n / fwd:.2f} tok/pass")
        print("[serve] " + " | ".join(bits), flush=True)

    def _ollama_chat(self, req: Request) -> Any:
        messages = [{"role": m.get("role", "user"), "content": _text(m, "content", "")} for m in _messages(req)]
        if not messages:
            return self._json(400, {"error": "messages required"})
        max_new = self._limit(req)
        stream = _flag(req, "stream", True)
        t0 = time.perf_counter()
        with self.reg.lock:
            engine = self._engine(req)
            if engine is None:
                return None
            name = engine.name
            smp = Sampling.from_request(_object(req, "options"), engine.sampling)
            if not stream:
                tg = time.perf_counter()
                ids, toks, c = engine.run(messages, max_new, sampling=smp)
                t_end = time.perf_counter()
                self._flex(engine, toks, c, tg, None, t_end)
                text, think = engine.text(toks)
                msg = {"role": "assistant", "content": text}
                if think:
                    msg["thinking"] = think
                return self._json(
                    200,
                    {
                        "model": name,
                        "created_at": _now(),
                        "message": msg,
                        "done": True,
                        **self._stats(ids, toks, c, t0, None, t_end),
                    },
                )
            self._stream_head("application/x-ndjson")

            def emit(delta: Any, kind: str = "content") -> Any:
                msg = (
                    {"role": "assistant", "content": delta}
                    if kind == "content"
                    else {"role": "assistant", "content": "", "thinking": delta}
                )
                return self._write(
                    json.dumps({"model": name, "created_at": _now(), "message": msg, "done": False}) + "\n"
                )

            tg = time.perf_counter()
            ids, toks, c, t_first = self._decode(engine, messages, max_new, emit, sampling=smp)
            t_end = time.perf_counter()
            self._flex(engine, toks, c, tg, t_first, t_end)
            return self._write(
                json.dumps(
                    {
                        "model": name,
                        "created_at": _now(),
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        **self._stats(ids, toks, c, t0, t_first, t_end),
                    }
                )
                + "\n"
            )

    def _ollama_generate(self, req: Request) -> Any:
        prompt = _text(req, "prompt", "") or ""
        system = _text(req, "system")
        raw = _flag(req, "raw", False)
        max_new = self._limit(req)
        stream = _flag(req, "stream", True)
        t0 = time.perf_counter()
        with self.reg.lock:
            engine = self._engine(req)
            if engine is None:
                return None
            name = engine.name
            smp = Sampling.from_request(_object(req, "options"), engine.sampling)
            ids = None
            messages = None
            if raw:
                ids = engine.tok(prompt, add_special_tokens=False)["input_ids"]
            else:
                messages = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": prompt}
                ]
            if not stream:
                tg = time.perf_counter()
                ids_used, toks, c = engine.run(messages or [], max_new, ids=ids, sampling=smp)
                t_end = time.perf_counter()
                self._flex(engine, toks, c, tg, None, t_end)
                text, think = engine.text(toks)
                return self._json(
                    200,
                    {
                        "model": name,
                        "created_at": _now(),
                        "response": text,
                        **({"thinking": think} if think else {}),
                        "done": True,
                        "context": [],
                        **self._stats(ids_used, toks, c, t0, None, t_end),
                    },
                )
            self._stream_head("application/x-ndjson")

            def emit(delta: Any, kind: str = "content") -> Any:
                body = {"response": delta} if kind == "content" else {"response": "", "thinking": delta}
                return self._write(json.dumps({"model": name, "created_at": _now(), **body, "done": False}) + "\n")

            tg = time.perf_counter()
            ids_used, toks, c, t_first = self._decode(engine, messages or [], max_new, emit, ids=ids, sampling=smp)
            t_end = time.perf_counter()
            self._flex(engine, toks, c, tg, t_first, t_end)
            return self._write(
                json.dumps(
                    {
                        "model": name,
                        "created_at": _now(),
                        "response": "",
                        "done": True,
                        "context": [],
                        **self._stats(ids_used, toks, c, t0, t_first, t_end),
                    }
                )
                + "\n"
            )


class PortInUse(RuntimeError):
    pass


class KeyRequired(RuntimeError):
    """a keyless bind to every interface or a public address"""


class Server:
    """The server in a process of your own. `serve_forever()` blocks on this thread; `start()` serves on a
    background thread and returns; `close()` stops it and unloads the models; `with` does start and close. The
    endpoints are `btb serve`'s: OpenAI's /v1/models and /v1/chat/completions, Ollama's /api/chat, /api/generate,
    /api/tags and /api/show."""

    def __init__(self, srv: _Server, registry: ModelRegistry) -> None:
        self._srv = srv
        self.registry = registry
        self._thread: threading.Thread | None = None
        self._serving = False

    @property
    def host(self) -> str:
        return str(self._srv.server_address[0])

    @property
    def port(self) -> int:
        return int(self._srv.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._srv.serve_forever()
        finally:
            self._serving = False

    def start(self) -> Server:
        if self._thread is None:
            self._thread = threading.Thread(target=self.serve_forever, name="btb-serve", daemon=True)
            self._thread.start()
        return self

    def wait(self) -> None:
        """until the background thread ends (a KeyboardInterrupt in the caller ends the wait, not the server)"""
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        if self._thread is not None or self._serving:
            self._srv.shutdown()
        self._srv.server_close()
        self.registry.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> Server:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _serve_until_interrupt(server: Server) -> None:
    """Serve on this thread until Ctrl-C. The first interrupt aborts any in-flight request at once, so its
    handler releases the lock and every tier (the cold reader, the drive queue, the expert and MLX pools) closes
    gracefully; a second interrupt exits hard should that teardown stall."""
    hits = [0]

    def handler(signum: int, frame: Any) -> None:
        hits[0] += 1
        if hits[0] >= 2:
            os._exit(130)
        for eng in list(server.registry.loaded.values()):
            with contextlib.suppress(Exception):
                eng.sm.abort.set()
        raise KeyboardInterrupt

    installed = False
    with contextlib.suppress(ValueError):  # signal handlers install only on the main thread
        signal.signal(signal.SIGINT, handler)
        installed = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if installed:
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGINT, signal.default_int_handler)


def _exposure(tag: str, host: str, api_key: str | None) -> None:
    """the line a bind beyond loopback prints: who can reach the server, and whether it takes a key"""
    if host in LOOPBACK - {"0.0.0.0", "::"}:
        return
    key = "with an API key" if api_key else "without an API key: anyone on its network can use the model"
    print(f"[{tag}] listening on {host} {key}", flush=True)


def start(
    path: str | None,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str | None = None,
    max_new: int | None = None,
    log: Log | None = None,
    extra_paths: Sequence[str] = (),
    pattern: str | None = None,
    api_key: str | None = None,
    **kw: Any,
) -> Server:
    """Bind the OpenAI/Ollama server without blocking and return it: `path` the model to load up front (None
    serves what the machine holds, on request), `host`/`port` the bind (port 0 picks a free one), `device` as
    `btb.load` takes it, `max_new` a ceiling per request, `extra_paths` directories to offer, `pattern` a regex
    over the names, `api_key` a bearer token every request must carry, `**kw` the load options. Raises
    `PortInUse` when the port is taken, `KeyRequired` for a keyless `host` of 0.0.0.0 or a public address.
    `.start()` serves on a background thread, `.close()` stops it."""
    import errno

    reg = ModelRegistry(
        path,
        device=device,
        max_new=max_new,
        log=log,
        extra_paths=extra_paths,
        pattern=pattern,
        **kw,
    )
    # bind before loading a model, so a busy port fails fast instead of after a long load
    try:
        srv = _Server((host, port), Handler)
        srv.reg = reg
        srv.api_key = api_key or None
    except OSError as e:
        reg.close()
        if e.errno in (errno.EADDRINUSE, errno.EACCES):
            raise PortInUse(
                f"port {port} on {host} is already in use — another server has it (a previous btb, "
                f"or the Ollama desktop app), so stop that one or pass --port <other>"
            ) from None
        raise
    bound = str(srv.server_address[0])
    if not srv.api_key and (bound in ("0.0.0.0", "::") or _public(bound)):
        srv.server_close()
        reg.close()
        raise KeyRequired(f"{KEY_REQUIRED} to bind {host}")
    if reg.primary_name:
        reg.acquire(reg.primary_name)  # a named launch model is loaded up front, as before
    return Server(srv, reg)


def serve(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str | None = None,
    max_new: int | None = None,
    log: Log | None = None,
    extra_paths: Sequence[str] = (),
    pattern: str | None = None,
    api_key: str | None = None,
    **kw: Any,
) -> int:
    """`btb serve` as a call: start the server (the arguments `start`'s), print where it listens and what it
    offers, and block until Ctrl-C; the process's exit code (1 when the port is taken)."""
    try:
        server = start(
            path,
            host=host,
            port=port,
            device=device,
            max_new=max_new,
            log=log,
            extra_paths=extra_paths,
            pattern=pattern,
            api_key=api_key,
            **kw,
        )
    except (PortInUse, KeyRequired) as e:
        print(f"[serve] {e}", flush=True)
        return 1
    _exposure("serve", host, api_key)
    reg = server.registry
    names = sorted(reg.entries)
    head = reg.primary_name or "(nothing loaded)"
    print(
        f"[serve] {head} at http://{host}:{port} (OpenAI: /v1/chat/completions, /v1/models; "
        f"Ollama: /api/chat, /api/generate, /api/tags, /api/show); greedy unless a request sets a temperature; "
        f"one request at a time",
        flush=True,
    )
    print(
        f"[serve] {len(names)} model(s) available, loaded on request: {', '.join(names) or '(none found)'}", flush=True
    )
    _serve_until_interrupt(server)
    return 0


def _open_gui(url: str) -> str | None:
    """Open the Ollama desktop app pointed at `url` (OLLAMA_HOST in its environment); returns the app launched,
    or None. Best effort."""
    env = dict(os.environ, OLLAMA_HOST=url)
    try:
        if sys.platform == "darwin":
            # `open` hands this environment to the app it launches, so OLLAMA_HOST reaches the GUI
            if subprocess.run(["open", "-a", "Ollama"], env=env, capture_output=True, check=False).returncode == 0:
                return "Ollama.app"
            return None
        if sys.platform.startswith("win"):
            exe = shutil.which("ollama app") or shutil.which("ollama app.exe")
            if exe:
                subprocess.Popen([exe], env=env)
                return exe
            subprocess.Popen(["cmd", "/c", "start", "", "ollama app.exe"], env=env)
            return "ollama app.exe"
        exe = shutil.which("ollama-app") or shutil.which("ollama-desktop")
        if exe:
            subprocess.Popen([exe], env=env)
            return exe
    except Exception:
        return None
    return None


def ollama(
    path: str,
    host: str = "127.0.0.1",
    port: int = 11435,
    device: str | None = None,
    max_new: int | None = None,
    log: Log | None = None,
    args: Any = (),
    extra_paths: Sequence[str] = (),
    pattern: str | None = None,
    gui: bool = False,
    api_key: str | None = None,
    **kw: Any,
) -> int:
    """`btb ollama` as a call: start the server (the arguments `start`'s, the port 11435 clear of a local
    Ollama), point the Ollama CLI at it and run `ollama run <name>` with `args` (or, with `gui`, open the
    desktop app), then stop when it exits; the exit code."""
    import http.client

    try:
        server = start(
            path,
            host=host,
            port=port,
            device=device,
            max_new=max_new,
            log=log,
            extra_paths=extra_paths,
            pattern=pattern,
            api_key=api_key,
            **kw,
        )
    except (PortInUse, KeyRequired) as e:
        print(f"[ollama] {e}", flush=True)
        return 1
    _exposure("ollama", host, api_key)
    reg = server.registry
    # `ollama run` needs a model name; with no launch model, open the first one discovered (loading it now so
    # the first turn is instant)
    name = reg.primary_name
    if name is None:
        names = sorted(reg.entries)
        if not names:
            server.close()
            print(
                "[ollama] no complete models found in the Hugging Face cache or --models-dir; "
                "give a model path/repo id, or download one first",
                flush=True,
            )
            return 1
        name = names[0]
        reg.acquire(name)
    server.start()
    for _ in range(50):
        try:
            c = http.client.HTTPConnection(host, port, timeout=1)
            c.request("GET", "/api/version")
            if c.getresponse().status == 200:
                break
        except OSError:
            time.sleep(0.1)
    url = f"http://{host}:{port}"
    if gui:
        app = _open_gui(url)
        if app:
            print(
                f"[ollama] opened {app} at {url}; it serves {name} and {len(reg.entries)} model(s). "
                f"Ctrl-C stops the server.",
                flush=True,
            )
        else:
            print(
                f"[ollama] no Ollama desktop app found. The server is up at {url}; open the app yourself with "
                f"OLLAMA_HOST={url}, or drop --gui for the terminal chat. Ctrl-C stops the server.",
                flush=True,
            )
        with contextlib.suppress(KeyboardInterrupt):
            server.wait()
        server.close()
        return 0
    exe = shutil.which("ollama")
    if exe is None:
        print(
            f"[ollama] the ollama command is not on PATH. The server is up at {url}; from another shell:\n"
            f"    OLLAMA_HOST={url} ollama run {name}\n(Ctrl-C stops the server; `ollama list` shows every "
            f"model btb found)",
            flush=True,
        )
        with contextlib.suppress(KeyboardInterrupt):
            server.wait()
    else:
        env = dict(os.environ, OLLAMA_HOST=url)
        print(
            f"[ollama] {name} served at {url}; running: ollama run {name} "
            f"(`OLLAMA_HOST={url} ollama list` shows all {len(reg.entries)})",
            flush=True,
        )
        try:
            rc = subprocess.call([exe, "run", name, *list(args)], env=env)
        except KeyboardInterrupt:
            rc = 130
    server.close()
    return rc if exe else 0


def _pi_provider(base_url: str, names: Sequence[str], api_key: str | None = None) -> Json:
    """The 'btb' provider entry pi reads: btb's OpenAI endpoint, every model it can load offered by id, the
    server's key when it has one."""
    return {
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": api_key or "btb",
        "models": [{"id": n} for n in names],
    }


def _write_pi_config(config_path: str | None, base_url: str, names: Sequence[str], api_key: str | None = None) -> str:
    """Merge the 'btb' provider into pi's models.json (default ~/.pi/agent/models.json), leaving other providers
    untouched; returns the path written."""
    dest = config_path or os.path.join(os.path.expanduser("~"), ".pi", "agent", "models.json")
    data: Any = {}
    if os.path.exists(dest):
        with contextlib.suppress(OSError, ValueError), open(dest) as f:
            data = json.load(f)
    if not isinstance(data, dict):
        data = {}
    providers = data.get("providers")
    if not isinstance(providers, dict):
        providers = data["providers"] = {}
    providers["btb"] = _pi_provider(base_url, names, api_key)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return dest


def pi(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str | None = None,
    max_new: int | None = None,
    log: Log | None = None,
    extra_paths: Sequence[str] = (),
    pattern: str | None = None,
    configure: bool = True,
    config_path: str | None = None,
    api_key: str | None = None,
    **kw: Any,
) -> int:
    """Serve the OpenAI API (with tool calling) and register it as a provider with the pi coding agent (pi.dev),
    then wait; the user runs `pi --provider btb --model <name>` in another terminal against it."""
    try:
        server = start(
            path,
            host=host,
            port=port,
            device=device,
            max_new=max_new,
            log=log,
            extra_paths=extra_paths,
            pattern=pattern,
            api_key=api_key,
            **kw,
        )
    except (PortInUse, KeyRequired) as e:
        print(f"[pi] {e}", flush=True)
        return 1
    _exposure("pi", host, api_key)
    reg = server.registry
    names = sorted(reg.entries)
    primary = reg.primary_name or (names[0] if names else None)
    base_url = f"http://{host}:{port}/v1"
    if configure:
        dest = _write_pi_config(config_path, base_url, names, api_key)
        print(f"[pi] provider 'btb' -> {base_url} written to {dest}", flush=True)
    else:
        entry = json.dumps({"providers": {"btb": _pi_provider(base_url, names, api_key)}}, indent=2)
        print(f"[pi] add this to ~/.pi/agent/models.json:\n{entry}", flush=True)
    if primary is not None:
        print(f"[pi] {primary} served with tool calling at {base_url}; in another terminal run:", flush=True)
        print(f"        pi --provider btb --model {primary}", flush=True)
    print(f"[pi] {len(names)} model(s) available, loaded on request: {', '.join(names) or '(none found)'}", flush=True)
    _serve_until_interrupt(server)
    return 0
