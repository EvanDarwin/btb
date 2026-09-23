# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
A model placed across the machine's tiers and decoded from Python. `load` returns the engine (text, tokens
and the memory system are its methods), `plan` prices a placement without loading, `host_budget` reads the
host as a plan would. `import btb` is torch-free; the engine's names on the package import it, and torch,
on first touch.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import weakref
from typing import TYPE_CHECKING, Any

from .hf import (
    PACK12_FORMAT,
    SERVE_TYPES,  # noqa: F401
    _cache_model_dirs,  # noqa: F401
    _model_bytes,  # noqa: F401
    _model_complete,  # noqa: F401
    _model_type,  # noqa: F401
    available_models,
    draft_for,
    is_packed,
    model_stem,
    pack_format,
    resolve,
    serve_name,
)

# model discovery lives in hf.py; these names stay importable from here
from .kinds import Json, Log, Proposer
from .options import Device

os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# the Hub prints this on every cache op when symlinks are off (Windows without Developer Mode); btb says it
# once, clearly, at download time instead (see hf._warn_windows_symlinks)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from .draft import SpanBank
from .session import Session
from .text import Channels, TextStream, answer, prompt_ids, template

if TYPE_CHECKING:
    from .engine import StreamedTextModel
    from .engine.device import mlx_available, resolve_device
    from .engine.native import kernels_path, native_path, native_tag, quiet_omp  # noqa: F401
    from .engine.scheduler import BatchScheduler, HostBudget, MemoryGrantError, Plan, PlanError
    from .engine.text import Chat, GenerateStats, Generation, Stream

CUDA = True


def cpu_only() -> bool:
    """
    Disables CUDA, and additionally edits the env to hide any CUDA
    devices from `torchao` to prevent it from automatically loading
    its own CUDA context without consent.
    """
    global CUDA
    CUDA = False
    if "torch" in sys.modules:
        return False
    # torchao patch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return True


# names served on first access and imported then, so `import btb` stays torch-free: the engine's modules import
# torch at their top, and these are re-exported from them for the package API (`btb.resolve_device`, ...)
_LAZY = {
    "StreamedTextModel": ".engine",
    "pack_model": ".engine",
    "mlx_available": ".engine.device",
    "resolve_device": ".engine.device",
    "native_path": ".engine.native",
    "native_tag": ".engine.native",
    "kernels_path": ".engine.native",
    "quiet_omp": ".engine.native",
    "BatchScheduler": ".engine.scheduler",
    "HostBudget": ".engine.scheduler",
    "MemoryGrantError": ".engine.scheduler",
    "Plan": ".engine.scheduler",
    "PlanError": ".engine.scheduler",
    "Chat": ".engine.text",
    "Generation": ".engine.text",
    "GenerateStats": ".engine.text",
    "Stream": ".engine.text",
    "Sampling": ".sampling",  # torch-backed: the root stays torch-free until a load
}


def __getattr__(name: str) -> Any:
    """Lazy-loading of the engine, and thereby torch - imported only on first access"""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib

    obj = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = obj
    return obj


def plan(
    path: str,
    device: Any = None,
    fp32: bool = False,
    os_reserve_gb: float | None = None,
    vram_reserve_gb: float | None = None,
    context: int = 0,
    kv_host: bool | None = None,
) -> Plan:
    """The placement the engine would take for the model at `path` on `device`, priced against the memory free
    right now. Opens the model on the CPU to size its layers, measures the card, and hands the budgeting to the
    scheduler (`BatchScheduler.plan_placement`), which owns the arithmetic - the host budget it measures, the
    floor it keeps (`os_reserve_gb` names another) - and the wait for a machine that is still giving memory
    back."""
    import torch

    from .engine import StreamedTextModel
    from .engine.device import free_bytes, resolve_device
    from .engine.scheduler import BatchScheduler

    quiet = lambda *_a, **_k: None
    path = resolve(path)
    dev = resolve_device(device)
    card = dev.kind.card
    if vram_reserve_gb is None:
        # the chosen card's memory, not device 0's: `dev` is "cuda" (the current device) or "cuda:N"
        vram_reserve_gb = (
            BatchScheduler.vram_margin_gb(torch.cuda.get_device_properties(str(dev)).total_memory) if card else 0.0
        )
    p = StreamedTextModel(path, device="cpu", resident_head=False, log=quiet)
    try:
        if p.pack is not None and p.pack["format"] == PACK12_FORMAT:
            p.open_packed()
        # the same arithmetic the scheduler and the memory policy read once the model is up (Device.free); on
        # MLX the GPU's memory is the RAM, so everything is planned as host layers and the card's figure is 0
        vram_gb = (free_bytes(torch.device(str(dev))) or 0) / 2**30 if card else 0.0
        return BatchScheduler.plan_placement(
            p,
            dev,
            vram_gb,
            packed=p.pack is not None and p.pack["format"] == PACK12_FORMAT,
            fp32=fp32,
            os_reserve_gb=os_reserve_gb,
            vram_reserve_gb=vram_reserve_gb,
            context=context,
            kv_host=kv_host,
        )
    finally:
        p.close()


def load(
    path: str,
    device: Any = None,
    native: str | None = None,
    log: Log | None = None,
    **kw: Any,
) -> StreamedTextModel:
    """Load the model at `path` (a directory, a Hugging Face repo id downloaded into the cache if missing, or a
    12-bit model written by `btb pack`) and return the engine. `device`: None picks the card, else MLX on Apple
    silicon, else the CPU; 'cpu', 'mlx', 'cuda' or 'cuda:N' names one that must be present. `native`: a path to
    the native library, '' for torch alone. `log`: a callable for the placement lines. `**kw`: the CLI's options
    by underscore name (`fp32=1`, `kv_host=1`, `context=8192`, `temperature=0.7`, `tree_budget=8`, ...); each
    is checked before anything loads and a bad one raises `btb.options.OptionError` naming it. Use as a context
    manager, or call `close()`."""
    from . import options, pool

    # every override and the device name checked first, before the pool, torch or the model: a bad value is one
    # line naming it, not a traceback from wherever it was first used (or a silent fall to the CPU)
    c: Json = options.check(kw)
    asked = options.check_device(device)
    path = resolve(path)
    # the pool cuts and faults the model's buffers under the imports, so torch is imported only after it has
    # seeded. That is btb's own order; a caller that imported torch first cannot be stopped, only noticed: the
    # pool detects it and logs that the RAM baseline is off by torch's footprint
    pool.seed_for(path, log, gguf_packed=bool(int(c.get("gguf_packed", 1))))
    import torch

    from .engine import StreamedTextModel
    from .engine.device import mlx_available, resolve_device
    from .engine.native import native_path, quiet_omp
    from .engine.state import DRAFT_VOCAB

    if asked is not None and asked.kind is Device.CPU:
        cpu_only()
    dll = native if native is not None else native_path()
    if dll:
        # 0 = the kernels' own pool over every core: torch's count is the physical cores, and fewer threads than the
        # pool routes through a second one (b=1 86 -> 92 GB/s, b=4 42 -> 51); the bits are the same at any count
        StreamedTextModel.load_gemv(dll, threads=0)
    quiet_omp()
    from .gguf import config_of

    full = config_of(path)
    L = int(getattr(full, "text_config", full).num_hidden_layers)
    options.check_layers(c, L)
    dev = resolve_device(device)
    fp32 = bool(int(c.get("fp32", 0)))
    if dev.kind is Device.MLX and fp32 and log:
        # fp32 works on MLX but falls off the fused megakernel and doubles the KV and bandwidth; bf16 already
        # saturates the registers, so this is slower for no quality it can spend - warn and let it run anyway
        log("[plan] fp32 on MLX: bf16 is faster and the same quality here. You've been warned")
    if dev.kind is Device.CPU and not fp32 and kw.get("fp32") is None:
        # the CPU computes in float32 over the bf16 weights either way (no native bf16 matmul), so fp32
        # is the default there unless the caller asks for bf16 activations outright
        fp32 = True
        if log:
            log("[plan] no card: the host computes in float32 over the bf16 weights")
    # a reserve given as a percent (--ram-reserve 10%) is resolved to GB here, against the machine's total RAM
    # and the chosen card's memory; the plan and the runtime scheduler both read the GB figure
    if c.get("ram_reserve_gb") is not None:
        from .sysinfo import host_total_bytes

        c["ram_reserve_gb"] = options.resolve_reserve(c["ram_reserve_gb"], host_total_bytes())
    if c.get("vram_reserve_gb") is not None:
        if isinstance(c["vram_reserve_gb"], options.Percent) and dev.kind.card:
            import torch

            vram_total = int(torch.cuda.get_device_properties(str(dev)).total_memory)
        else:
            vram_total = 0  # no card: a percent of the card is nothing to keep free
        c["vram_reserve_gb"] = options.resolve_reserve(c["vram_reserve_gb"], vram_total)
    cold = []
    reserves = {"os_reserve_gb": c["ram_reserve_gb"]} if c.get("ram_reserve_gb") is not None else {}
    if c.get("vram_reserve_gb") is not None:
        reserves["vram_reserve_gb"] = c["vram_reserve_gb"]
    plan_kw: dict[str, Any] = {
        "fp32": fp32,
        "context": int(c.get("context", 0) or 0),
        "kv_host": (None if c.get("kv_host") is None else bool(int(c["kv_host"]))) if dev.kind.card else False,
        **reserves,
    }
    mlx_layers = None
    pl: Plan | None = None
    if dev.kind is Device.MLX:
        pl = plan(path, device=dev, **plan_kw)
        res: tuple[int, ...] = ()
        cpu, cold = list(pl.host), list(pl.cold)
        n_cpu = max(0, min(L, int(kw.get("cpu_layers") or 0)))
        mlx_layers = [i for i in cpu if i >= n_cpu]
        c.setdefault("resident_head", 1)
        c["prefill_card"] = 0
        # Metal's matvec serves up to 15 rows at the cost of one; the tree's verify is 1 + budget rows
        c.setdefault("tree_budget", 14 if pl.has_mtp else 0)
        if log:
            b = pl.bytes
            if pl.free.settle_s:
                log(
                    f"[plan] waited {pl.free.settle_s:.0f} s for the machine to give memory back: "
                    f"free RAM {pl.free.ram_gb_first:.1f} -> {pl.free.ram_gb:.1f} GB"
                )
            log(
                f"[plan] free RAM {pl.free.ram_gb:.1f} GB (unified) -> {len(mlx_layers) - len(cold)} layers "
                f"resident for the GPU ({b.warm / 2**30:.2f} GB), {len(cold)} from the drive each pass "
                f"({b.cold / 2**30:.2f} GB"
                + (f", {c.get('cold_slots', pl.cold_slots())} slots of read-ahead" if cold else "")
                + f"), {n_cpu} on the CPU kernels; head on the GPU; "
                f"drafter {'yes' if pl.has_mtp else 'none'}"
                + (f"; the drive {pl.drive.bps / 1e9:.2f} GB/s" if pl.drive is not None else "")
            )
    elif dev.kind is Device.CPU or "resident_last" not in c:
        pl = plan(path, device=dev, **plan_kw)
        res = tuple(pl.resident)
        cpu = list(pl.host)
        cold = list(pl.cold)
        if dev.kind is not Device.CPU and c.get("cpu_layers") is not None:
            # the CPU's share is named (--cpu-layers N, 0 for none): those layers run on the CPU kernels; the
            # card keeps what the plan gave it of the rest, and the remainder streams from the drive
            n_cpu = min(L, int(c["cpu_layers"]))
            cpu = list(range(n_cpu))
            res = tuple(i for i in res if i >= n_cpu)
            cold = [i for i in range(n_cpu, L) if i not in res]
            # the ring as the RAM tier of the streamed layers: as many slots as the memory the host's share
            # leaves, so a layer read once stays until its slot is needed
            warm_b = int(pl.bytes.warm * n_cpu / len(pl.warm)) if pl.warm else 0
            c.setdefault("cold_slots", pl.cold_slots(warm_bytes=warm_b, n_cold=len(cold)))
        c.setdefault("resident_head", int(pl.head_on_card))
        c["prefill_card"] = int(pl.prefill_card)
        c.setdefault("kv_host", int(pl.kv_host))
        # the tree's budget: a drafting head draws it; without one the n-gram continuations merge into one tree on a
        # card holding every layer, where its rows verify at about the cost of one; 15 rows and the root are the
        # widest pass the card's GEMV serves at its 16-row cost
        c.setdefault(
            "tree_budget",
            (15 if dev.kind is not Device.CPU else 16)
            if (pl.has_mtp or (dev.kind is not Device.CPU and not cpu and not cold))
            else 0,
        )
        if log:
            b = pl.bytes
            if pl.free.settle_s:
                log(
                    f"[plan] waited {pl.free.settle_s:.0f} s for the machine to give memory back: "
                    f"free RAM {pl.free.ram_gb_first:.1f} -> {pl.free.ram_gb:.1f} GB"
                )
            log(
                f"[plan] free VRAM {pl.free.vram_gb:.1f} GB, RAM {pl.free.ram_gb:.1f} GB -> "
                f"resident {len(res)} layers ({b.vram_layers / 2**30:.2f} GB), host {len(cpu)} "
                f"(cold {len(cold)}), head {'card' if pl.head_on_card else 'host'}, "
                f"drafter {'card' if pl.drafter_on_card else ('host' if pl.has_mtp else 'none')}, "
                f"prefill via card {pl.prefill_card}; shadows {b.shadow / 2**30:.2f} GB, templates {b.templates / 2**30:.2f} GB, "
                f"cache {(b.kv_card + b.kv_host) / 2**30:.2f} GB for {max(4096, int(plan_kw['context'] or 0))} positions"
                f"{' in RAM' if pl.kv_host else ''}, "
                f"staging {b.staging / 2**30:.2f} GB"
                + (
                    f"; {pl.budget.floor / 2**30:.2f} GB kept free (the OS's {pl.budget.os_floor / 2**30:.2f}, "
                    f"the run's growth {pl.budget.growth / 2**30:.2f})"
                    if pl.budget is not None
                    else ""
                )
                + (f"; the drive {pl.drive.bps / 1e9:.2f} GB/s" if pl.drive is not None else "")
            )
    else:
        # an explicit placement (`--resident-last`, with `--cpu-layers`): the split named on the command line, no plan
        cpu = list(range(min(L, int(c.get("cpu_layers", L)))))
        rl = int(c.get("resident_last", 0))  # dev is the card here: CPU took the branch above
        res = tuple(range(L - rl, L)) if rl else ()
        cpu = [i for i in cpu if i not in res]
    if pl is not None:
        # every planned placement: the cold reader's depth; the drafter's head over the frequent 32K ids (a draft
        # outside them is a missed proposal, never a wrong token); speculation on by default (the tree with a
        # drafting head, the n-gram drafter without; --v-max 0 turns it off), off for a mixture of experts, where a
        # verify pass routes rows to more experts and the node kernel on every decode row costs gpt-oss 30%
        c.setdefault("cold_slots", pl.cold_slots())
        c.setdefault("draft_vocab", DRAFT_VOCAB if pl.has_mtp else 0)
        c.setdefault("v_max", 0 if pl.moe else 4)
    sm = StreamedTextModel(
        path,
        device=dev,
        prefetch=bool(int(c.get("prefetch", 1))) and dev.kind is not Device.CPU,
        resident_layers=res,
        cpu_layers=cpu,
        cold_layers=cold,
        cold_slots=int(c.get("cold_slots", 2) or 2),
        compute_dtype=(torch.float32 if fp32 else None),
        resident_head=bool(int(c.get("resident_head", 1))),
        prefill_card=bool(int(c.get("prefill_card", 0))) and dev.kind is not Device.CPU,
        prefill_card_min=int(c.get("prefill_card_min", 64)),
        kv_host=bool(int(c.get("kv_host", 0))),
        context=int(c.get("context", 0) or 0) or None,
        kv_bits=(int(c.get("kv_bits", 0) or 0) or None) if dev.kind is Device.MLX else None,
        gguf_packed=bool(int(c.get("gguf_packed", 1))),
        rope_scaling=c.get("rope_scaling"),
        original_max_position_embeddings=c.get("original_max_position_embeddings"),
        expert_cache_gb=c.get("expert_cache_gb"),
        ram_reserve_gb=c.get("ram_reserve_gb"),
        vram_reserve_gb=c.get("vram_reserve_gb"),
        vram_watch=bool(int(c.get("vram_watch", 1))),
        mlx_layers=mlx_layers,
        host_budget=pl.budget if pl is not None else None,
        # the expert store is built inside __init__, so its policy travels as an argument, not an assignment after
        bus_pass=bool(int(c.get("bus_pass", 1))),
        store_pin=int(c.get("store_pin", 0)),
        log=log or (lambda *_a: None),
    )
    sm.plan = pl
    if sm.pack is not None and sm.pack["format"] == PACK12_FORMAT:
        sm.open_packed()
        if sm.host:
            sm.bind_host_packed()
        if log:
            log(f"[store] 12-bit model: the layers from {path}, the rest from {sm.pack['source']}")
    if (
        dev.kind is Device.CPU
        and sys.platform == "darwin"
        and sm.host
        and c.get("fp32") != 1
        and os.environ.get("BTB_CPU_GEMM", "1") != "0"
        and mlx_available()
    ):
        # a Mac's CPU tier: the prefill's matmuls on MLX's CPU stream in bf16 (half the f32 path's time); one-row
        # steps and verify passes keep the native kernels. --fp32 1 or BTB_CPU_GEMM=0 keeps float32
        n = sm.bind_cpu_gemm()
        if log and n:
            log(f"[stream] host linears in shared memory ({n / 2**30:.2f} GB): the prefill's GEMM on the CPU stream")
    # closed before the interpreter tears down, so reader threads and GPU work end while their buffers exist
    ref = weakref.ref(sm)

    def _close_at_exit() -> None:
        live = ref()
        if live is not None:
            live.close()

    atexit.register(_close_at_exit)
    sm.drafter_weights = None
    # the tree's budget applies to either proposer: a drafting head (the checkpoint's `mtp.*` weights) draws the
    # tree, and without one the n-gram proposer's continuations at every order merge into one (generate.py).
    # The plan-driven placements default the budget above; an explicit placement (--cpu-layers with --resident-last)
    # reaches here without a plan, so the default comes from the weights and the device: a head, or a card
    # holding every layer, where a tree's rows verify at about the cost of one
    has_drafter = any(k.startswith("mtp.") for k in sm.weight_map)
    # the card's widest pass at its 16-row cost is 15 drafted rows and the root; the host has no such edge
    default_budget = (
        (15 if sm.dev.type == Device.CUDA else 16)
        if (has_drafter or (sm.dev.type == Device.CUDA and not sm.host))
        else 0
    )
    sm.tree_budget = int(c.get("tree_budget", default_budget))
    # the drafter's path probability under-reports a walk's reach (greedy or sampled), so the floor sits low
    sm.tree_min_prob = float(c.get("tree_min_prob", 0.15))
    sm.tree_step_mass = float(c.get("tree_step_mass", 0.5))
    # int8 only pays on MLX
    sm.draft_bits = int(c.get("draft_bits", 8 if sm.mlx is not None else 16))
    sm.tree_read = "step"
    sm.draft_vocab = int(c.get("draft_vocab", DRAFT_VOCAB if has_drafter else 0) or 0)
    sm.draft_temp_ratio = float(c.get("draft_temp_ratio", 1.0) or 1.0)
    sm.ngram_p = float(c.get("ngram_p", 0.9))
    sm.v_max = int(c.get("v_max", 0 if sm.fam.moe else 4))  # an explicit placement: the planned default's rule
    if sm.fam.own:
        sm.tree_budget = 0
        sm.v_max = 0
    sm.proposer = Proposer.MTP_DYN if (sm.tree_budget > 0 and has_drafter) else Proposer.NGRAM
    # how every token is picked unless a call says otherwise: greedy, or the loaded temperature / top_p / top_k / seed
    from .sampling import Sampling

    sm.sampling = Sampling(
        temperature=float(c.get("temperature", 0.0) or 0.0),
        top_p=float(c.get("top_p", 1.0) if c.get("top_p") is not None else 1.0),
        top_k=int(c.get("top_k", 0) or 0),
        seed=(int(c["seed"]) if c.get("seed") is not None else None),
    )
    # the expert store's lookahead: the next layers' router picks read ahead of their layers, per depth (the
    # configuration's `lookahead`, BTB_LOOKAHEAD="10,6" or "0" over it). Off for one-row passes: their
    # predictions' reads take the drive from the layer's own; a pass of several rows keeps its picks
    la = os.environ.get("BTB_LOOKAHEAD")
    sm.lookahead = tuple(
        int(x) for x in (la.split(",") if la is not None else c.get("lookahead", ())) if str(x).strip()
    )
    if sm.lookahead == (0,):
        sm.lookahead = ()
    lr = os.environ.get("BTB_LOOKAHEAD_ROWS")
    sm.lookahead_rows = tuple(
        int(x) for x in (lr.split(",") if lr is not None else c.get("lookahead_rows", (10, 6))) if str(x).strip()
    )
    if sm.lookahead_rows == (0,):
        sm.lookahead_rows = ()
    # the experts seated on the card: 0 none, "auto" what the card has to spare, or a figure in GB
    ve = os.environ.get("BTB_VRAM_EXPERTS_GB", c.get("vram_experts_gb", 0))
    sm.vram_experts_gb = "auto" if str(ve).strip().lower() == "auto" else float(ve or 0)
    if getattr(sm, "mlx", None) is not None and sm.v_max > 0:
        # every decode row through the engine's attention kernel, so a verify pass and the one-row step
        # compute alike (see `_mlx_attend`)
        sm.mlx_attn_rows = 0
    sm.eos_ids = ()
    if sm.gguf is not None:
        sm.eos_ids = sm.gguf.eos_ids()  # a GGUF carries them in its tokenizer keys, not a generation config
    gp = os.path.join(path, "generation_config.json")
    if os.path.exists(gp):
        e = json.load(open(gp, encoding="utf-8")).get("eos_token_id")
        if e is not None:
            sm.eos_ids = tuple(int(x) for x in (e if isinstance(e, (list, tuple)) else [e]))
    # ready is part of the load, not of one command: the card's graphs and the pass-cost curve (the card graph),
    # the MLX pass-cost curve (the fused tree), over a throwaway prompt - what every entry point is timed at
    if dev.kind is Device.MLX and int(c.get("mlx_mega", 1)) and sm.fam.kernel_layout and not sm.cold:
        from .mlx.mega import MegaPass

        try:
            sm._mega = MegaPass(sm)
            sm.log(
                f"[mega] the pass as one dispatch: {sm._mega.nb} weight buffers, scratch {sm._mega.scr.size >> 20} MB"
            )
        except ValueError as e:
            sm.log(f"[mega] off: {e}")
    sm.draft_engine = None
    sm.draft_ks = (4, 3, 2)
    sm.warm()
    # a sibling model drafts the speculative tree, verified exactly by this model's tree pass: far better than the
    # n-gram proposer on fresh prose, and the memory-bound single-stream lever. Same tokenizer required.
    from .hf import confirm_download, is_gguf

    dm = c.get("draft_model")
    if dm == "auto":
        # the curated draft for this model (hf.DRAFT_MODELS); refused by name when this one has no known pair
        dm = draft_for(path)
        if dm is None:
            raise options.OptionError(
                "--draft-model auto: no curated draft model for this one - pass an explicit --draft-model PATH"
            )
    if dm and os.path.isfile(str(dm)) and not is_gguf(str(dm)):
        # a custom MTP drafting-head file (experimental): drafter.py reads its mtp.* keys, this model's own
        # weights for the rest. Unchecked - a head that is not this model's blows up at first use.
        sys.stderr.write(
            "[btb] custom drafting-head file for speculative decoding - unsupported, I hope you know what "
            "you're doing!\n"
        )
        sm.drafter_weights = str(dm)
        sm.proposer = Proposer.MTP_DYN
        if not sm.tree_budget:
            sm.tree_budget = 14 if sm.mlx is not None else 16
    elif dm:
        # a small model of the family (a directory, repo id, or GGUF) drafts for this one, loaded as its own
        # engine on the same device with its speculation off (v_max=0) - it proposes, never verified against.
        # An uncached repo is gated like any download; the vocabularies must match for its tokens to verify.
        if not confirm_download(str(dm)):
            raise options.OptionError(f"draft_model {dm!r}: download declined")
        ks = c.get("draft_ks")
        if ks is not None:
            sm.draft_ks = tuple(int(x) for x in (ks.split(",") if isinstance(ks, str) else ks) if str(x).strip())
        draft_sm = load(
            resolve(str(dm)),
            device=device,
            native=native,
            log=None,
            v_max=0,
            tree_budget=0,
            gguf_packed=int(c.get("gguf_packed", 1)),
        )
        if int(draft_sm.cfg.vocab_size) != int(sm.cfg.vocab_size):
            draft_sm.close()
            raise options.OptionError(
                f"draft_model {dm!r} has vocab {draft_sm.cfg.vocab_size}, the model's is {sm.cfg.vocab_size}: "
                "a draft must share the model's tokenizer for its tokens to verify"
            )
        sm.draft_engine = draft_sm
        sm.log(f"[draft] {os.path.basename(str(dm))} proposing, tree {sm.draft_ks}")
    return sm


def host_budget(floor_gb: float | None = None) -> HostBudget:
    """The host's RAM as a plan would be drawn against it right now: the OS's figures, this process's working
    set, and the floor kept free (the OS's own, or the one `floor_gb` names). No model is loaded for it."""
    from .engine.scheduler import BatchScheduler

    return BatchScheduler.measure_host(floor_gb)


def peak_memory(sm: Any = None) -> tuple[int, int]:
    import torch

    from .sysinfo import peak_rss_bytes

    rss = peak_rss_bytes()
    vram = 0
    if sm is not None and getattr(sm, "mlx", None) is not None:
        vram = int(sm.mlx.peak_bytes())
    elif sm is not None and sm.dev.type == Device.CUDA:
        vram = int(torch.cuda.max_memory_reserved(sm.dev))
    return rss, vram


def device_name(sm: Any) -> str:
    if getattr(sm, "mlx", None) is not None:
        return "mlx"
    return str(sm.dev)


def available_devices() -> list[Json]:
    """The compute devices a model can be placed on, most capable first: each CUDA card, MLX where present, and
    the CPU. Per device: the `--device` selector `name`, its `kind` (`gpu`/`mlx`/`cpu`), and `details` the
    machine reports."""
    import importlib.util

    from .sysinfo import host_cores, host_cpu_name, host_simd, host_total_bytes

    ram_gb = round(host_total_bytes() / 2**30, 1)
    cpu_model = host_cpu_name() or "unknown"
    out: list[Json] = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                out.append(
                    {
                        "name": f"cuda:{i}",
                        "kind": "gpu",
                        "details": {
                            "model": p.name,
                            "memory_gb": round(p.total_memory / 2**30, 1),
                            "compute_capability": f"{p.major}.{p.minor}",
                            "backends": ["CUDA"],
                        },
                    }
                )
    except Exception:
        pass
    # MLX is Apple silicon only; the mlx package's presence is the signal, read without importing it
    if sys.platform == "darwin" and importlib.util.find_spec("mlx") is not None:
        out.append(
            {"name": "mlx", "kind": "mlx", "details": {"model": cpu_model, "memory_gb": ram_gb, "backends": ["MLX"]}}
        )
    out.append(
        {
            "name": "cpu",
            "kind": "cpu",
            "details": {"model": cpu_model, "cores": host_cores(), "memory_gb": ram_gb, "backends": host_simd()},
        }
    )
    return out


__all__ = [
    "BatchScheduler",
    "Channels",
    "Chat",
    "GenerateStats",
    "Generation",
    "HostBudget",
    "MemoryGrantError",
    "Plan",
    "PlanError",
    "Sampling",
    "Session",
    "SpanBank",
    "Stream",
    "StreamedTextModel",
    "TextStream",
    "answer",
    "available_devices",
    "available_models",
    "cpu_only",
    "device_name",
    "host_budget",
    "is_packed",
    "load",
    "mlx_available",
    "model_stem",
    "native_path",
    "pack_format",
    "pack_model",
    "peak_memory",
    "plan",
    "prompt_ids",
    "resolve",
    "resolve_device",
    "serve_name",
    "template",
]
