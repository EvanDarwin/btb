# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The forward pass: positions and masks, the prefill (whole, chunked, by layer, with the attention on the card
in blocks), the head, and the transformers-side forward over the tiers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .. import mlx as mlxdev
from ..kinds import LayerKind, LayerTier, Parents, PassTag, TokenRows
from ..options import Device
from ..sampling import as_pick
from .cache import GrowLayer, conv_states_as
from .native import Native
from .scheduler import EPOCH
from .state import _State

# the ledger's tag for a layer-by-layer prefill's own working set, for the sweep: on the card, and on the host where
# its host chunks run and its rows wait between layers
PREFILL = "prefill"

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache

    from .device import Placement


# a rope table (cos, sin) over a pass's positions
Rope = tuple[torch.Tensor, torch.Tensor]
# a pass's rope: one table for every layer, or one per layer type (a dual-rope family's local/global split)
PassRope = Rope | dict[str, Rope]

_GPU_PREFILL_RETRIES = 3
# the fewest rows a prefill chunk takes, however short of room: a pass of no more rows is never chunked
PREFILL_MIN_ROWS = 64


def _kv_on_card(cache: Any, host: Any) -> int:
    """the bytes the resident layers' KV takes on the card in `cache` (each layer's whole buffer, as its growth was
    granted), the host layers' passed over: their rows on the card are a prefill's hop, its own working set"""
    n = 0
    for i, cl in enumerate(cache.layers):
        if i in host:
            continue
        buf = getattr(cl, "_buf", None)
        held = buf if buf is not None else (getattr(cl, "keys", None), getattr(cl, "values", None))
        n += sum(t.numel() * t.element_size() for t in held if isinstance(t, torch.Tensor) and t.device.type == "cuda")
    return n


def pe_for(pe: PassRope | None, lt: str) -> Rope | None:
    """the rope a layer of type `lt` reads: a dual-rope family carries one per type, the rest one for every layer"""
    return pe[lt] if isinstance(pe, dict) else pe


def layer_window(cfg: Any, lt: str) -> int:
    """a sliding layer's window in rows; 0 for a layer that sees the whole prefix"""
    return int(getattr(cfg, "sliding_window", 0) or 0) if lt == LayerKind.SLIDING else 0


def node_mask(base: int, p: int, parents: Parents, win: int) -> torch.Tensor | None:
    """Which of the base + p + 1 cache rows node p of a pass attends, as a bool row: the prefix (its last `win`
    rows under a window), its ancestors among the pass's rows, and itself. None where every row would do - a
    chain with no window - so the attention's own causal form serves."""
    anc = []
    q = parents[p]
    while q >= 0:
        anc.append(q)
        q = parents[q]
    depth = len(anc)
    first = max(0, base + depth + 1 - win) if win else 0
    if first == 0 and depth == p:
        return None
    allow = torch.zeros(base + p + 1, dtype=torch.bool)
    allow[first:base] = True
    allow[base + p] = True
    for k, q in enumerate(anc):  # the k-th ancestor up sits at depth - 1 - k: logical row base + that
        if base + depth - 1 - k >= first:
            allow[base + q] = True
    return allow


def chain_of(parents: Parents | None, T: int) -> list[int]:
    """a pass's parents as a list: a tree's, or a chain's own when none were given"""
    return [int(p) for p in parents] if parents is not None else list(range(-1, T - 1))


def path_of(parents: Sequence[int], j: int) -> list[int]:
    """node j and its ancestors among a pass's rows, j first"""
    out = [j]
    while parents[out[-1]] >= 0:
        out.append(parents[out[-1]])
    return out


def tree_mask(causal: torch.Tensor, past: int, parents: Parents) -> torch.Tensor:
    """`causal` [B, 1, T, past + T] (bool, or additive float) with its last T columns a tree's: row j sees the
    prefix, itself and its ancestors - the mask a family running its own layers verifies a tree through"""
    par = [int(p) for p in parents]
    T = len(par)
    anc = torch.zeros(T, T, dtype=torch.bool)
    for j in range(T):
        anc[j, path_of(par, j)] = True
    anc = anc.to(causal.device)
    out = causal.clone()
    if out.dtype == torch.bool:
        out[..., past : past + T] = anc
    else:
        out[..., past : past + T] = torch.where(anc, out.new_zeros(()), torch.finfo(out.dtype).min)
    return out


def _is_gpu_recovery(e: BaseException) -> bool:
    """A transient Metal reset: a command buffer discarded as the innocent victim of a GPU error/recovery. The
    card reset and threw away in-flight work; nothing this pass wrote committed, so it can be replayed once the
    GPU settles - as opposed to a guilty fault (an out-of-memory, an invalid resource) that would recur."""
    s = str(e).lower()
    return "innocentvictim" in s or "victim of gpu error" in s


class _Pass:
    """One forward's frame, handed to every layer by the device: the cache, the positions and masks, the flags the
    tiers read, the placement the pass holds, and the CPU copies a host layer needs, made once."""

    __slots__ = (
        "T",
        "_host",
        "am",
        "batched",
        "cache",
        "card_pass",
        "causal",
        "linear_mask",
        "n_layers",
        "on_layer",
        "own",
        "past",
        "pe",
        "place",
        "ple_ids",
        "text_pos",
    )

    T: int
    past: int
    n_layers: int
    batched: bool
    card_pass: bool
    own: bool
    cache: DynamicCache | None
    am: torch.Tensor | None
    linear_mask: torch.Tensor | None
    ple_ids: torch.Tensor | None
    text_pos: torch.Tensor
    pe: PassRope | None
    causal: torch.Tensor | dict[LayerKind, torch.Tensor] | None
    on_layer: Callable[[int, torch.Tensor], Any] | None
    place: Placement | None

    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)
        self.place = None
        self._host: dict[str, Any] = {}

    def host_side(self) -> dict[str, Any]:
        hc = self._host
        if "pe" not in hc:
            pe = self.pe
            assert pe is not None  # a host layer's pass always carries the rope
            if isinstance(pe, dict):
                hc["pe"] = {lt: (p[0].cpu(), p[1].cpu()) for lt, p in pe.items()}
            else:
                hc["pe"] = (pe[0].cpu(), pe[1].cpu())
            hc["pos"] = self.text_pos.cpu()
            hc["lin"] = None if self.linear_mask is None else self.linear_mask.cpu()
            hc["ple"] = None if self.ple_ids is None else self.ple_ids.cpu()
        return hc

    def host_causal(self) -> Any:
        # the pass's causal mask, built once above `past` positions before any layer appended this pass's keys -
        # reused here on the host, not rebuilt from the cache. A host layer that runs after resident layers (a
        # shed moved the last layer down) would otherwise read a cache.get_seq_length() already grown by those
        # layers' appends and mask `past + 2T` keys against a `past + T` cache (a batched prefill: 44 vs 22)
        hc = self._host
        if "causal" not in hc:

            def _cpu(c: Any) -> Any:
                if c is None:
                    return None
                if isinstance(c, dict):
                    return {k: _cpu(v) for k, v in c.items()}
                return (c.float() if c.is_floating_point() else c).cpu()

            hc["causal"] = _cpu(self.causal)
        return hc["causal"]


class _ForwardMixin(_State):
    def _positions(
        self, B: int, T: int, past: int, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            pos = torch.arange(T, device=self.dev) + past
            pos = pos.view(1, 1, -1).expand(4, B, -1)
        else:
            m = attention_mask.to(self.dev).long()
            pos = (m.cumsum(-1) - 1).clamp(min=0)[:, -T:]
            pos = pos.reshape(1, B, T).expand(4, B, -1)
        return pos[0], pos[1:]

    @staticmethod
    def pad_left(rows: TokenRows, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(r) for r in rows)
        ids = torch.full((len(rows), T), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(rows), T), dtype=torch.long)
        for b, r in enumerate(rows):
            ids[b, T - len(r) :] = torch.as_tensor(r, dtype=torch.long)
            mask[b, T - len(r) :] = 1
        return ids, mask

    @torch.inference_mode()
    def forward(
        self,
        ids: Any,
        cache: Any = None,
        on_layer: Callable[[int, torch.Tensor], Any] | None = None,
        last_only: bool = True,
        stop_after: int | None = None,
        attention_mask: torch.Tensor | None = None,
        positions: Any = None,
        head: bool = True,
        pick: Any = None,
    ) -> torch.Tensor | None:
        """`pick` (True: greedy; a Sampling: its draw, keyed by each row's cache position): the fused MLX path
        returns [B, T] int32 token ids picked in the graph instead of logits; the other paths ignore it and
        return logits, so a caller checks the dtype."""
        pick = as_pick(pick)
        ids_arg = ids  # as given: a card program reads the pass's ids on the host
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        if ids.dim() == 1:
            ids = ids.view(1, -1)
        B, T = ids.shape
        if self.mlx is not None and cache is not None:
            self._mlx_flush_states(cache)
        past = cache.get_seq_length() if cache is not None else 0
        # the memory policies read once a second, here so every loop over the engine - its own and a
        # caller's over `forward` - sheds and regrows the same way
        self.vram_policy(cache)
        self.ram_policy()
        self.lend_policy()
        self.cache_room(cache, B, T)
        own = bool(self.fam.own)
        n_layers = self.L if stop_after is None else min(self.L, int(stop_after))
        # the placement tiers this pass runs layers on, and its stored-weight path, recorded whatever branch
        # takes them below (the per-op MLX host path bypasses the fused forwards, so record here too)
        self._tag_tiers(n_layers)
        self._tag_quant()
        if (
            attention_mask is None
            and not own
            and B == 1
            and n_layers == self.L
            and self._mega_ok(cache, T, on_layer, head, pick, positions)
        ):
            return self._forward_mega(ids[0].tolist(), cache, T, pick)
        if attention_mask is None and not own and self._mlx_ok(cache, B, T, None, positions, n_layers):
            # the fused path from the embedding gathered on the graph: no torch embed, rope tables or mask
            m = mlxdev.mx()
            hm = self._mlx_embed_rows(m.array(ids[0].tolist(), dtype=m.int32))
            if self.compute_dtype is not None and self.compute_dtype != torch.bfloat16:
                hm = hm.astype(m.float32)
            return self._forward_mlx(None, None, cache, on_layer, last_only, head, n_layers, hm=hm, pick=pick)
        h = self.embed(ids)
        if self.compute_dtype is not None:
            h = h.to(self.compute_dtype)
        if self.fam.streams > 1:
            h = h.repeat(1, 1, self.fam.streams)
        # a family's card program (Qwen4's) takes a one-row step or a verify pass whole on the card
        prog = self._card_program(cache, B, T, past, attention_mask, stop_after, positions)
        if prog is not None:
            self._attn_ctx = cache
            with self.device.hold() as place:
                # the program was checked against the placement as it stood; one moved since takes the torch path
                if place.version == prog.version:
                    return self._forward_card_program(
                        prog, ids_arg, h, cache, past, positions, last_only, head, on_layer, place
                    )
        cp = getattr(self, "_cp", None)
        if cp is not None and cache is not None:
            cp.touched(cache)  # a pass off the program: what it mirrors of this cache is read again
        am = None
        if attention_mask is not None:
            am = torch.as_tensor(attention_mask, dtype=torch.long, device=self.dev).view(B, -1)
            if cache is not None and am.shape[1] != past + T:
                raise RuntimeError(f"[stream] attention_mask covers {am.shape[1]} positions, cache+ids {past + T}")
        if positions is not None:
            pos = torch.as_tensor(positions, dtype=torch.long, device=self.dev).view(1, B, T).expand(4, B, -1)
            text_pos, rope_pos = pos[0], pos[1:]
        else:
            text_pos, rope_pos = self._positions(B, T, past, am)
        n_layers = self.L if stop_after is None else min(self.L, int(stop_after))
        # a pass the card graph takes whole (every layer one resident run) needs none of the preamble below:
        # the graph carries its own rotary tables and its attention needs no mask, and the rotary and the
        # mask together cost more host time than the graph's replay on a small model
        graph_ok = self._card_pass_ok(cache, B, T, past, am, on_layer, stop_after)
        if graph_ok and self._card_segment_at(0, n_layers) == (0, n_layers) and n_layers == self.L:
            self._attn_ctx = cache
            pas = _Pass(
                cache=cache,
                pe=None,
                text_pos=text_pos,
                causal=None,
                linear_mask=None,
                ple_ids=None,
                T=T,
                past=past,
                am=None,
                batched=False,
                own=own,
                card_pass=False,
                n_layers=n_layers,
                on_layer=None,
            )
            with self.device.hold() as place:
                pas.place = place
                tail = head and self.head is not None and self.norm is not None
                hcard, logits = self._forward_card_segment(0, n_layers, h, pas, tail)
            if logits is not None:
                return logits[:, -1:] if last_only else logits
            assert hcard is not None  # logits None means the segment returned (h, None)
            h = hcard[:, -1:, :] if last_only else hcard
            h = self._final_norm(h)
            cd = self.compute_dtype if self.compute_dtype is not None else h.dtype
            hf = h.to(cd)
            return hf if not head else self._apply_head(hf)
        pe = self._pass_rope(h, past, am, positions, text_pos, rope_pos)
        n_layers = self.L if stop_after is None else min(self.L, int(stop_after))
        if self._mlx_ok(cache, B, T, am, positions, n_layers):
            return self._forward_mlx(h, pe, cache, on_layer, last_only, head, n_layers, pick=pick)
        # the module-run attention (`attention_sinks`) reads this pass's cache off the engine
        self._attn_ctx = cache if am is None else None
        tier_owns_attention = (
            getattr(self, "kv_host", False)
            and cache is not None
            and am is None
            and not own
            and self.fam.fast
            and all(i in self.resident for i, lt in enumerate(self.layer_types) if lt == LayerKind.FULL)
        )
        causal = None if tier_owns_attention else self._causal(h, am, cache, text_pos, own)
        tree = getattr(self, "ap", None) if getattr(self, "aq", False) else None
        if own and causal is not None and tree is not None:
            # a family running its own layers verifies a tree through their mask: each row sees its ancestors
            causal = tree_mask(causal, past, tree)
        linear_mask = None if (am is None or bool(torch.all(am == 1))) else am[:, -T:]
        ple_ids = self.fam.ple_ids(self.cfg, ids, linear_mask)
        n_layers = self.L if stop_after is None else min(self.L, int(stop_after))
        if self._fast_ok(cache, B, T, past, am, on_layer, stop_after):
            return self._forward_fast(h, pe, cache, last_only, head)

        batched = getattr(self, "_batched_cont", False)
        card_pass = (
            bool(getattr(self, "prefill_card", False))
            and (past == 0 or batched)
            and getattr(self, "prefill_card_min", 64) <= T
            and bool(self.templates)
            and bool(self.host)
        )
        if card_pass:
            self._tag(PassTag.PREFILL_CARD)  # the host layers prefill on the card for this pass
        if card_pass and cache is not None and past > 0:
            # every host layer's rows onto the card for the pass, granted together before any moves
            self._rows_to([cache], [i for i in self.host if i < n_layers], self.dev)

        pas = _Pass(
            cache=cache,
            pe=pe,
            text_pos=text_pos,
            causal=causal,
            linear_mask=linear_mask,
            ple_ids=ple_ids,
            T=T,
            past=past,
            am=am,
            batched=batched,
            own=own,
            card_pass=card_pass,
            n_layers=n_layers,
            on_layer=on_layer,
        )
        if self.prefetch:
            self._prefetch_next(0, pas)
        if self.cold and not card_pass:
            self._cold_start(n_layers)
        # every layer through the device: it reads the tier off the pass's snapshot, crosses the tier's edge
        # (the move and the cast), and holds the placement for the pass, so a shed asked for meanwhile waits.
        # A run of resident attention layers replays as one captured graph when the pass has the shape for it
        # (the one-token step, a verify pass); the head rides in the last run's graph when it ends the model
        with self.device.hold() as place:
            pas.place = place
            i = 0
            while i < n_layers:
                seg = self._card_segment_at(i, n_layers) if graph_ok else None
                if seg is None:
                    h = self.device.run_layer(i, h, pas)
                    i += 1
                    continue
                a, b = seg
                tail = b == self.L and head and self.head is not None and self.norm is not None
                hcard, logits = self._forward_card_segment(a, b, h, pas, tail)
                if logits is not None:
                    self._flush_events()
                    return logits[:, -1:] if last_only else logits
                assert hcard is not None  # logits None means the segment returned (h, None)
                h = hcard
                i = b
        self._flush_events()
        if card_pass and cache is not None:
            for i in self.host:
                if i < n_layers:
                    self._cache_to(cache, i, "cpu")
        if n_layers < self.L:
            return None
        h = self._norm_input(h)
        if last_only:
            h = h[:, -1:, :]
        h = self._final_norm(h)
        cd = self.compute_dtype if self.compute_dtype is not None else h.dtype
        hf = h.to(cd)
        if not head:
            return hf
        return self._apply_head(hf)

    def _pass_rope(
        self,
        h: torch.Tensor,
        past: int,
        am: torch.Tensor | None,
        positions: Any,
        text_pos: torch.Tensor,
        rope_pos: torch.Tensor,
    ) -> PassRope:
        """a pass's rope: over the prefix and the pass for a family running its own layers, one per layer type for
        a dual-rope family, else over the pass's positions"""
        B, T = int(h.shape[0]), int(h.shape[1])
        if self.fam.own:
            if positions is not None:
                prev = torch.arange(past, device=self.dev).view(1, 1, -1).expand(3, B, -1)
                rope_all = torch.cat([prev, rope_pos], dim=-1)
            else:
                rope_all = self._positions(B, past + T, 0, am)[1]
            return self.rotary(h, rope_all)
        if self.fam.dual_rope:
            # one rope per layer type (Gemma 3's local/global split); each layer reads its own from the pass
            return {lt: self.rotary(h, text_pos, lt) for lt in set(self.layer_types)}
        return self.rotary(h, rope_pos if self.fam.mrope else text_pos)

    def _host_frame(
        self,
        h: torch.Tensor,
        ids: list[int],
        cache: Any,
        past: int,
        positions: Any,
        on_layer: Callable[[int, torch.Tensor], Any] | None,
    ) -> _Pass:
        """the frame a card program's host layers run in (cuda.py `_forward_card_program`): one sequence's
        positions, rope, mask - a verify pass's tree through it - and ids, as this pass makes them for its host
        layers, so a host layer between the program's segments runs as it runs on the torch path"""
        B, T = int(h.shape[0]), int(h.shape[1])
        own = bool(self.fam.own)
        if positions is not None:
            pos = torch.as_tensor(positions, dtype=torch.long, device=self.dev).view(1, B, T).expand(4, B, -1)
            text_pos, rope_pos = pos[0], pos[1:]
        else:
            text_pos, rope_pos = self._positions(B, T, past)
        pe = self._pass_rope(h, past, None, positions, text_pos, rope_pos)
        causal = self._causal(h, None, cache, text_pos, own)
        tree = getattr(self, "ap", None) if getattr(self, "aq", False) else None
        if own and causal is not None and tree is not None:
            causal = tree_mask(causal, past, tree)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.dev).view(B, T)
        return _Pass(
            cache=cache,
            pe=pe,
            text_pos=text_pos,
            causal=causal,
            linear_mask=None,
            ple_ids=self.fam.ple_ids(self.cfg, ids_t, None),
            T=T,
            past=past,
            am=None,
            batched=False,
            own=own,
            card_pass=False,
            n_layers=self.L,
            on_layer=on_layer,
        )

    def _prefetch_next(self, j: int, pas: Any) -> None:
        """start the read of the next layer that is neither resident nor a host layer (the templates' path)"""
        while j < pas.n_layers and (j in self.resident or (j in self.host and not pas.card_pass)):
            j += 1
        if j < pas.n_layers:
            self._start_prefetch(j, self._next_template(self.layer_types[j]))

    @staticmethod
    def _layer_dtype(tmpl: Any) -> torch.dtype | None:
        """the dtype a layer's projections take their input in: its first floating parameter's"""
        return next((p.dtype for p in tmpl.parameters() if p.is_floating_point()), None)

    def _card_layer(self, i: int, pas: Any) -> Any:
        """the layer module a card pass runs for `i`: the resident one, the prefetched template, or a template
        loaded now - upcast where the compute is float32 over bf16 weights - the next read started behind it"""
        lt = self.layer_types[i]
        if i in self.resident:
            tmpl = self.resident[i]
            if lt in self.shadow and not self.resident_fp32:
                tmpl = self._upcast(lt, tmpl, i)
        elif self.prefetch:
            tmpl = self._wait_prefetch(i)
            if lt in self.shadow:
                tmpl = self._upcast(lt, tmpl, i)
            self._prefetch_next(i + 1, pas)
        else:
            tmpl = self.templates[lt][0]
            self._load_layer(i, tmpl)
            if lt in self.shadow:
                tmpl = self._upcast(lt, tmpl, i)
        return tmpl

    def _run_host_layer(self, i: int, h: torch.Tensor, pas: Any) -> torch.Tensor:
        """layer `i` on the CPU kernels, `h` already fp32 on the host (the device crossed the edge)"""
        if self.mlx is None and Native.gemv is not None:
            # a true CPU tier: the host linears run the native gemv. On MLX a host layer's linears carry `.mx`
            # and run through Native.mlx.linear instead, so it is not the native CPU path
            self._tag(PassTag.CPU_NATIVE)
        elif self.mlx is not None:
            self._tag(PassTag.MLX_PEROP)  # the non-fused MLX path: a MoE family, or a layer offloaded to the host
        if i in self.fp8_layers:
            self._tag(PassTag.FP8_ASSTORED)
        tmpl = self.host[i]
        hc = pas.host_side()
        held = self._cold_held == i  # a layer-by-layer prefill holds the slot across the layer's chunks
        if i in self.cold and not held:
            self._cold_wait(i)
        t0 = time.time()
        cache, T, past, am = pas.cache, pas.T, pas.past, pas.am
        lt = self.layer_types[i]
        spec = getattr(self, "aq", False) and cache is not None and T > 1 and past > 0
        step1 = Native.delta_step is not None and cache is not None and T == 1 and past > 0 and am is None
        cont = cache is not None and T > 1 and past > 0 and am is None and not pas.batched
        # `ai` is the single-sequence host path (it also serves speculative decode, always one row); it stores
        # k/v as [1, heads, ...] and would merge a real batch into the head dim, so a batched decode takes the
        # standard module forward instead - which stores [B, heads, ...] as prefill does and is the streaming
        # path that scales to huge models, now batched
        if (spec or step1 or cont) and not pas.own and self.fam.fast and h.shape[0] == 1:
            h = self.ai(tmpl, i, h, pe_for(hc["pe"], lt), hc["pos"], cache)
        else:
            kw = self.fam.layer_kw(
                lt,
                pas.host_causal() if (pas.own or lt != LayerKind.LINEAR) else None,
                hc["lin"],
                hc["pos"],
                hc["ple"],
            )
            h = tmpl(
                h,
                position_embeddings=pe_for(hc["pe"], lt),
                past_key_values=cache,
                use_cache=cache is not None,
                **kw,
            )
        self.compute_s += time.time() - t0
        if i in self.cold and not held:
            self._cold_release(i)
        if pas.on_layer is not None:
            pas.on_layer(i, h)
        return h

    def _run_card_layer(self, i: int, tmpl: Any, h: torch.Tensor, pas: Any) -> torch.Tensor:
        """layer `i` on the card (or the compute device), `h` already there in the layer's dtype"""
        if self.dev.type == Device.CUDA:
            # a card layer through the torch modules, not a captured card graph: btb's kernels are absent or the
            # family is not one they serve, so the pass runs like torch
            self._tag(PassTag.CUDA_TORCH_FALLBACK)
        lt = self.layer_types[i]
        cache, T, past, am = pas.cache, pas.T, pas.past, pas.am
        if cache is not None and i < len(cache.layers):
            conv_states_as(cache.layers[i], h.dtype)  # a state left by the layer's run elsewhere, in its dtype here
        t0 = time.time()
        if self.dev.type == Device.CUDA:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
        if (
            getattr(self, "kv_host", False)
            and cache is not None
            and lt == LayerKind.FULL
            and i in self.resident
            and am is None
            and not pas.own
            and self.fam.fast
        ):
            h = self._kv_split(tmpl, i, h, pas.pe, cache)
        elif (
            cache is not None
            and T > 1
            and past > 0
            and am is None
            and not pas.batched
            and not pas.own
            and self.fam.fast
        ):
            h = self.ac(tmpl, i, h, pas.pe, pas.text_pos, cache)
        else:
            h = tmpl(
                h,
                position_embeddings=pe_for(pas.pe, lt),
                past_key_values=cache,
                use_cache=cache is not None,
                **self.fam.layer_kw(lt, pas.causal, pas.linear_mask, pas.text_pos, pas.ple_ids),
            )
        if self.dev.type == Device.CUDA:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            self._events.append((e0, e1))
        else:
            self.compute_s += time.time() - t0
        if pas.on_layer is not None:
            pas.on_layer(i, h)
        return h

    def _causal(
        self,
        h: torch.Tensor,
        am: torch.Tensor | None,
        cache: Any,
        pos: torch.Tensor,
        own: bool,
        layer_idx: int | None = None,
    ) -> Any:
        """The attention mask for this pass: one tensor, or - for a family whose layers alternate a window
        with the whole prefix, as gpt-oss's do - one per layer type, which the family's `layer_kw` picks from."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        kw = {
            "config": self.cfg,
            "inputs_embeds": h,
            "attention_mask": am,
            "past_key_values": cache,
            "position_ids": pos,
            "allow_is_causal_skip": not own,
        }
        if layer_idx is not None:
            kw["layer_idx"] = layer_idx
        if LayerKind.SLIDING not in self.layer_types:
            return create_causal_mask(**kw)
        return {
            LayerKind.FULL: create_causal_mask(**kw),
            LayerKind.SLIDING: create_sliding_window_causal_mask(**kw),
        }

    def _norm_input(self, h: torch.Tensor) -> torch.Tensor:
        """the last layer's rows crossing into the final norm, where the norm is and in its own dtype: a last layer
        given up to the host hands its rows back in float32 on the host, and a card's norm takes its weights'"""
        mod = self.norm if self.norm is not None else self.mixer
        p = next((q for q in mod.parameters() if q.is_floating_point()), None) if mod is not None else None
        dev, dt = (p.device, p.dtype) if p is not None else (self.dev, h.dtype)
        return h if (h.device == dev and h.dtype == dt) else h.to(dev, dt)

    def _final_norm(self, h: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            return self.norm(h)
        assert self.mixer is not None  # a model carries a norm or a mixer, never neither
        return self.mixer(h)

    def _apply_head(self, hf: torch.Tensor) -> torch.Tensor:
        cd = hf.dtype
        step = 32768
        if not self.resident_head:
            self._tag(PassTag.HEAD_STREAMED)
            # the host's own head: the native matvec reads the bf16 rows once where the chunked form widens each chunk
            # to float32; for the few rows a decode or verify pass carries
            if (
                self.dev.type == Device.CPU
                and cd == torch.float32
                and self.mlx is None
                and Native.gemv is not None
                and hf.numel() // hf.shape[-1] < Native.gemm_rows
            ):
                return self._head_host()(hf).float()
            W = self._get(self.head_key)
            parts = []
            for c in range(0, W.shape[0], step):
                wc: Any = W[c : c + step].to(self.dev)
                parts.append(hf @ wc.to(cd).T)
                wc = None
            return torch.cat(parts, dim=-1).float()
        self._tag(PassTag.HEAD_RESIDENT)
        if self.head is None and self.head_host is not None:
            return self.head_host(hf.float().cpu()).float()
        assert self.head is not None  # no head_host means the head is resident
        if self.head.weight.dtype != cd:
            # the resident head is bf16, the compute float32: the native gemv reads the head once and widens on the
            # fly (no float32 copy of a ~1 Gelem head per token) for the few rows a decode or verify pass carries. Its
            # accumulation order moves the logits ~20 ulp from the widen's; the tokens are identical. BTB_HEAD_GEMV=0
            # forces the widen back
            if (
                self.dev.type == Device.CPU
                and cd == torch.float32
                and self.mlx is None
                and Native.gemv is not None
                and hf.numel() // hf.shape[-1] < Native.gemm_rows
                and os.environ.get("BTB_HEAD_GEMV", "1") == "1"
            ):
                rows, cols = self.head.weight.shape
                x2 = hf.reshape(-1, cols).contiguous()
                y = torch.empty(x2.shape[0], rows, dtype=torch.float32)
                Native.gemv(self.head.weight, x2, y)
                return y.view(*hf.shape[:-1], rows).float()
            W = self.head.weight
            return torch.cat([hf @ W[c : c + step].to(cd).T for c in range(0, W.shape[0], step)], dim=-1).float()
        return self.head(hf).float()

    def _finish(self, h: torch.Tensor, last_only: bool, head: bool) -> torch.Tensor:
        if last_only:
            h = h[:, -1:, :]
        h = self.norm(h)
        cd = self.compute_dtype if self.compute_dtype is not None else h.dtype
        hf = h.to(cd)
        if not head:
            return hf
        return self._apply_head(hf)

    # the head sizes MLX's fused attention kernel takes for a many-row pass; any other (Qwen3.5's 256) has the
    # scores of every row against every key materialized, T x (past + T) x heads, which the chunk must be priced by
    MLX_FUSED_HEAD = (64, 80, 128)

    # the share of the machine's RAM a prefill leaves to the OS and everything else on unified memory: the kernel's
    # free-memory level is what its own killer reads, and a run that took the machine to 8% was killed at the floor
    OS_FLOOR = 0.12

    def prefill_room(self) -> int:
        """the memory a prefill's working set may take right now, on the tier it runs on: on the card what the
        device's ledger has free there (`Device.free`: above the margin, less what is spoken for - the one figure
        every grant and room is priced by); on unified memory the tightest of the engine's own ledger, Metal's
        working-set limit less what MLX holds, and the machine's free RAM as it stands now (the other programs'
        share included) above the OS's floor; the host's free RAM above a gigabyte elsewhere"""
        on_card = self.dev.type == Device.CUDA and (bool(self.resident) or bool(self.prefill_card))
        if on_card:
            return int(self.device.free(self.dev, unreserved=True) or 0)
        mlx = getattr(self, "mlx", None)
        if mlx is not None:
            # the ledger (the machine's live RAM above the OS's floor included), within Metal's working-set limit
            room = int(self.device.free(unreserved=True) or 0)
            limit = int((getattr(mlx, "info", None) or {}).get("max_recommended_working_set_size", 0) or 0)
            if limit:
                room = min(room, limit - int(mlx.held_bytes()))
            return max(0, room)
        return int(self.device.free(torch.device("cpu"), unreserved=True) or 0)

    def _chunk_cost(self, on_card: bool | None = None) -> tuple[int, int, int]:
        """A prefill chunk's working set: per row, the bytes of its activations through the widest layer; per (row,
        key) pair, the bytes of its scores where the attention materializes them and of its mask; and per key, the
        bytes of the keys and values widened to every query head where the heads share them. `on_card`: priced for
        the card's pass or the host's (the tier the engine's prefill runs on when None)"""
        c = self.cfg
        H = int(c.hidden_size) * int(self.fam.streams)
        I = int(getattr(c, "intermediate_size", None) or 4 * H)
        if self.fam.moe:
            # a token routes to num_experts_per_tok experts, each of the expert width: moe_intermediate_size
            # where the config names one (Qwen MoE), else the plain intermediate (gpt-oss names only that)
            expert_i = int(getattr(c, "moe_intermediate_size", None) or I)
            I = max(I, expert_i * int(getattr(c, "num_experts_per_tok", 1) or 1))
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        d = int(getattr(c, "head_dim", None) or H // Hq)
        if on_card is None:
            on_card = self.dev.type == Device.CUDA and (bool(self.resident) or bool(self.prefill_card))
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        nb = 4 if (fp32 or not on_card) else 2
        per_token = 2 * nb * (3 * I + 8 * H + 2 * (Hq + 2 * Hk) * d)
        if getattr(self, "mlx", None) is not None and LayerKind.LINEAR in self.layer_types:
            # the hybrid's chunked DeltaNet rule keeps its per-chunk states and blocks beside the activations:
            # on Qwen3.5-4B a 4096-row chunk at position 0 costs 4.9-6.2 GB against the 2.4 the
            # activations alone price
            per_token *= 2
        scores = 0
        if getattr(self, "mlx", None) is not None:
            if d not in self.MLX_FUSED_HEAD:
                scores = 4 * Hq  # bytes per (row, key) pair of materialized scores, float32 through the softmax
            else:
                # the fused kernel's per-key-block partials, (T, Hk, splits*8, g, D) float32 and two without D,
                # folded after: about half the materialized figure a (row, key), and still growing with every key
                # the chunk sees
                from ..mlx.attn import ATTN_BLOCK

                scores = -(-32 * Hq * (d + 2) // ATTN_BLOCK)
        elif self.fam.eager:
            # gpt-oss's sinks (`attention_sinks`) hold each row's scores against every key twice at once in the dtype
            # the reference's join promotes them to - the sinks' own (float32 in the checkpoint), or the pass's where
            # that is wider - the product beside the buffer it is written into, then that buffer beside the softmax;
            # and the mask once, in the pass's dtype
            scores = 2 * max(nb, self._sinks_bytes()) * Hq + nb
        per_key = 0
        if getattr(self, "mlx", None) is None and not self.fam.eager:
            # torch's sdpa over every key the chunk sees: the causal mask, a value a (row, key) pair at most, and the
            # keys and values widened to every query head (`btb_sdpa`'s expand, transformers' `repeat_kv`)
            scores += nb
            if Hq > Hk:
                per_key = 2 * Hq * d * nb
        return per_token, scores, per_key

    def _presize_kv(self, cache: Any, rows: int, B: int) -> None:
        """each resident attention layer's KV made `rows` long at once (`GrowLayer.presize`), granted from the epoch's
        room, before a layer-by-layer prefill cuts its working set: a cache growing mid-sweep would split it. The
        layers the card graph's arena holds keep their place there; with the rows kept on the host (`kv_host`) no
        layer's are the card's, and nothing is presized there"""
        if getattr(self, "kv_host", False):
            return
        st = getattr(self, "_cg", None)
        ar = st["arena"] if st is not None else None
        owner = ar["owner"]() if ar is not None and ar["owner"] is not None else None
        in_arena = set(ar["slot"]) if ar is not None and owner is cache else set()
        c = self.cfg
        hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        for i, cl in enumerate(cache.layers):
            if i in self.host or i in in_arena or self.layer_types[i] == LayerKind.LINEAR:
                continue
            tmpl = self.resident.get(i)
            dt = self._layer_dtype(tmpl) if tmpl is not None else None
            if isinstance(cl, GrowLayer) and dt is not None:
                cl.presize(int(rows), int(B), Hk, d, dt, self.dev)

    def _sinks_bytes(self) -> int:
        """the bytes of a sink logit as the resident layers hold them (float32 where none is resident to ask)"""
        for tmpl in list(self.resident.values()):
            sinks = getattr(getattr(tmpl, "self_attn", None), "sinks", None)
            if isinstance(sinks, torch.Tensor):
                return int(sinks.element_size())
        return 4

    def _sweep_bytes(self, C: int, past: int, B: int, T: int, cache: Any) -> tuple[int, int]:
        """What a layer-by-layer prefill of `T` rows at chunks of `C` asks of its device, as (its working set -
        the last chunk's activations and scores (`_chunk_bytes`) and, on the card, a mixture's grouped call over
        its rows (`_grouped_bytes`) - and what it holds beside that for the sweep: the rows kept on the card
        between layers and the cache's growth past what an epoch has reserved for it). The chunk size is priced
        by the two together, and the sweep asks the ledger for them"""
        card = self.dev.type == Device.CUDA
        work = self._chunk_bytes(C, past + max(0, T - C))
        if card:
            work += self._grouped_bytes(C, getattr(self, "expert_store", None))
        row_b, park = self._sweep_rows(B, T)
        rows = 0 if (park or not card) else T * row_b
        growth = self.cache_growth(cache, B, T, peak=True).get(self.dev, 0)
        epoch = int(self.device.reserved(self.dev)) - int(self.device.reserved(self.dev, but=EPOCH))
        return work, rows + self._hop_bytes(C, past, B, T) + max(0, growth - epoch)

    def _sweep_rows(self, B: int, T: int) -> tuple[int, bool]:
        """a layer-by-layer prefill's rows between layers: the bytes one takes, and whether they wait in pinned RAM
        (past a quarter of the card's room) rather than on the card"""
        cd = self.compute_dtype
        row_b = (
            B
            * int(self.cfg.hidden_size)
            * int(self.fam.streams)
            * (4 if (cd is not None and cd != torch.bfloat16) else 2)
        )
        return row_b, self.dev.type == Device.CUDA and T * row_b > self.prefill_room() // 4

    def _card_chunks(self, C: int, T: int) -> list[bool]:
        """which of a layer-by-layer prefill's chunks of `C` rows the card takes where the layer lives on the host:
        every one of `prefill_card_min` rows or more, where the card passes host layers at all"""
        on_card = bool(getattr(self, "prefill_card", False)) and bool(self.templates) and bool(self.host)
        return [on_card and getattr(self, "prefill_card_min", 64) <= min(T, a + C) - a for a in range(0, T, C)]

    def _hop_bytes(self, C: int, past: int, B: int, T: int) -> int:
        """a host layer's rows on the card for a chunk of it there (`GrowLayer.hop`): its keys and values up to the
        last such chunk's end, in the card's dtype, at most once at a time"""
        card = self._card_chunks(C, T)
        ends = [min(T, a + C) for a, on in zip(range(0, T, C), card) if on]
        if not ends or self.dev.type != Device.CUDA:
            return 0
        c = self.cfg
        hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(getattr(c, "head_dim", None) or c.hidden_size // hq)
        cd = self.compute_dtype
        nb = 4 if (cd is not None and cd != torch.bfloat16) else 2
        return 2 * B * Hk * d * nb * (past + max(ends))

    def _host_bytes(self, C: int, past: int, B: int, T: int, park: bool) -> int:
        """what a layer-by-layer prefill on the card asks of the host: its host chunks' working set in float32 where
        a layer lives there and the card does not take the chunk, and - where the sweep parks them (`park`, decided
        once before its own reservations shrink the card's room) - its rows between layers, each chunk's a pinned
        buffer, which the pinned allocator rounds up to a power of two"""
        if self.dev.type != Device.CUDA:
            return 0
        n = 0
        if (self.host or self.cold) and not all(self._card_chunks(C, T)):
            n += self._chunk_bytes(C, past + max(0, T - C), on_card=False)
        row_b, _park = self._sweep_rows(B, T)
        if park:
            n += sum(1 << max(0, (min(T, a + C) - a) * (row_b // B) * B - 1).bit_length() for a in range(0, T, C))
        return n

    def _grouped_bytes(self, rows: int, store: Any) -> int:
        """a mixture's grouped expert call on the card over a chunk of `rows` (`_card_grouped`), at its widest: each
        pick's place in the sum's buffer, and beside it the larger of its way in - the row gathered, gate_up's
        product, its bias gathered and their sum, and the gate's temporaries over the halves (gpt-oss's clamped GLU,
        the widest of the families') - and its way out - the gate's output, down's product, its bias and their sum,
        the routing weight's float32 product and its cast; and the widening of the MXFP4 or FP8 experts it
        multiplies (`widen_scratch`). 0 where the loop takes the calls"""
        if not self.fam.moe or not bool(getattr(self, "grouped_experts", False)):
            return 0
        from .host import _Experts, widen_scratch

        c = self.cfg
        H = int(c.hidden_size)
        I = int(getattr(c, "moe_intermediate_size", None) or c.intermediate_size)
        k = int(getattr(c, "num_experts_per_tok", 1) or 1)
        return 2 * rows * k * (H + max(H + 8 * I, I + 5 * H)) + widen_scratch(store, _Experts.WIDEN_BATCH, I, H)

    def _chunk_bytes(self, rows: int, past: int = 0, on_card: bool | None = None) -> int:
        """a prefill chunk of `rows` at position `past`: its activations, its scores and mask against every key it
        sees, and those keys widened to every query head (`_chunk_cost`)"""
        per_token, scores, per_key = self._chunk_cost(on_card)
        keys = int(past) + int(rows)
        return int(rows) * (per_token + scores * keys) + per_key * keys

    def _auto_chunk(self, past: int = 0) -> int:
        """The rows a prefill pass takes at once: the largest power of two, 64 to 4096, whose working set fits the
        room - each row's activations through the widest layer, and, where the attention materializes its
        scores, each row's scores against every key it sees (`past` and the chunk's own rows), which is what
        grows with the context and once asked Metal for 33 GB in one buffer."""
        _per_token, scores, _per_key = self._chunk_cost()
        free = self.prefill_room()
        buf_cap = free  # the single attention buffer must fit the reserve-aware room (--ram/--vram-reserve, via
        mlx = getattr(self, "mlx", None)  # prefill_room) and, on MLX, Metal's hard per-buffer ceiling
        if mlx is not None:
            hard = int((getattr(mlx, "info", None) or {}).get("max_buffer_length", 0) or 0)
            if hard:
                buf_cap = min(buf_cap, hard)
        rows = 4096
        # shrink until the whole working set fits the room, and until the attention's own buffer at these rows
        # fits buf_cap - so a free reading that runs high never passes a single buffer the GPU cannot allocate,
        # the oversized ask that faults it
        while rows > PREFILL_MIN_ROWS and (
            self._chunk_bytes(rows, past) > free or scores * rows * (int(past) + rows) > buf_cap
        ):
            rows //= 2
        return int(rows)

    def _prefill(
        self,
        ids: torch.Tensor,
        cache: Any,
        on_layer: Any = None,
        attention_mask: torch.Tensor | None = None,
        last_only: bool = True,
    ) -> Any:
        """`ids` into `cache` in chunks the free memory prices: the last row's logits, or every row's (the chunks'
        joined) without `last_only`"""
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        if ids.dim() == 1:
            ids = ids.view(1, -1)
        T = ids.shape[1]
        if cache is None or attention_mask is not None or getattr(self, "aq", False):
            return self.forward(ids, cache=cache, on_layer=on_layer, attention_mask=attention_mask, last_only=last_only)
        # a prompt past one chunk prefills layer by layer, so a layer the pass streams (a card pass's template, the
        # drive's ring, a mixture's experts) is read once for the whole prompt instead of once per chunk (the
        # chunks of a 16k prompt read the 180B's experts 3.35 times over)
        sweep = on_layer is None and last_only and self.mlx is None and self.prefill_layers
        # a hybrid's DeltaNet continues a chunk from the cache's states, but sums in float32 over blocks that split
        # where its chunks do: chunks of a size the free memory picks would make the same prompt's bits depend on
        # what else the machine holds. So the torch hybrid sweeps only chunks of the size `prefill_chunk` names, and
        # otherwise takes its prompt whole, as it always has (the MLX path continues its chunks from the stored
        # states)
        whole_hybrid = (
            LayerKind.LINEAR in self.layer_types
            and not self.fam.own
            and self.mlx is None
            and not (sweep and self.prefill_chunk)
        )
        past = int(cache.get_seq_length())
        C = int(self.prefill_chunk or self._auto_chunk(past))
        if T <= C or whole_hybrid:
            return self.forward(ids, cache=cache, on_layer=on_layer, last_only=last_only)
        if sweep:
            if not self.prefill_chunk:
                # every chunk is taken at one size, so it is priced where it costs the most - the last chunk's
                # position, whose rows see every key before them (the chunks one at a time price each at its own)
                # - with everything else the sweep asks of the device beside it (`_sweep_bytes`)
                room = self.prefill_room()
                while C > PREFILL_MIN_ROWS and sum(self._sweep_bytes(C, past, ids.shape[0], T, cache)) > room:
                    C //= 2
            return self._prefill_by_layer(ids, cache, C)
        self.log(f"[prefill] {T} tokens in chunks of {C} from position {past}")
        # a layer hook sees the last layer's rows for the whole prompt once, joined at the end, as it would from
        # one forward (the drafter's prefill reads them); every other layer's rows are handed over per chunk as
        # they come - the fused forwards tap every layer, and keeping all of them for every chunk held 320 KB a
        # row, 8 GB across a 120k prompt, until the prefill ended
        taps: list[torch.Tensor] = []
        last = self.L - 1

        def collect(i: int, h: torch.Tensor) -> None:
            if i == last:
                taps.append(h)
            else:
                on_layer(i, h)

        hook = collect if on_layer is not None else None
        rows: list[torch.Tensor] = []  # every chunk's logits, without `last_only`
        self._batched_cont = True
        try:
            a = 0
            while True:
                # each chunk priced at its own position: the keys it attends over grow with every chunk before it,
                # and the room shrinks as the cache takes its share (a 32k-row turn sized once at its start swapped)
                C = int(self.prefill_chunk or self._auto_chunk(past + a))
                if C < 512 and not self.prefill_chunk:
                    self.log(
                        f"[prefill] memory-starved: {C} rows a chunk at position {past + a} "
                        f"({self.prefill_room() / 2**30:.1f} GB of room); every chunk reads the weights again"
                    )
                b = min(T, a + C)
                # WAL/checkpoint: the chunk boundary is the commit, the cache length the log. A transient GPU
                # reset discards the chunk's in-flight work; roll the cache back to `ckpt` (crop the KV rows the
                # chunk began past - idempotently overwritten on replay, never read below ckpt) and replay once
                # the scheduler has eased the card off. Only where the rollback is complete and side-effect free:
                # crop covers every layer, no per-layer hook to re-fire, no hybrid recurrent state to restore.
                ckpt = int(cache.get_seq_length())
                attempt = 0
                while True:
                    try:
                        out = self.forward(ids[:, a:b], cache=cache, on_layer=hook, last_only=last_only)
                        break
                    except Exception as e:
                        sched = getattr(self, "scheduler", None)
                        if not (
                            _is_gpu_recovery(e)
                            and attempt < _GPU_PREFILL_RETRIES
                            and sched is not None
                            and on_layer is None
                            and LayerKind.LINEAR not in self.layer_types
                            and all(hasattr(cl, "crop") for cl in cache.layers)
                        ):
                            raise
                        for cl in cache.layers:
                            cl.crop(ckpt)
                        attempt += 1
                        wait = sched.gpu_recovered()
                        self.log(
                            f"[prefill] transient GPU recovery at {a}/{T}: chunk rolled back to {ckpt}, "
                            f"replay {attempt}/{_GPU_PREFILL_RETRIES} in {wait:.2f}s"
                        )
                        time.sleep(wait)
                if not last_only and out is not None:
                    rows.append(out)
                if b >= T:
                    break
                self.log(f"[prefill] {b}/{T} (chunks of {C})")
                a = b
        finally:
            self._batched_cont = False
        if on_layer is not None and taps:
            on_layer(last, taps[0] if len(taps) == 1 else torch.cat(taps, dim=1))
        return out if last_only else torch.cat(rows, dim=1)

    @torch.inference_mode()
    def _prefill_by_layer(self, ids: torch.Tensor, cache: Any, C: int) -> Any:
        """The prompt through one layer at a time, every chunk of it before the next layer, so a layer the pass
        streams is read once for the whole prompt instead of once per chunk: a card pass's template, a drive layer's
        ring slot (held across the layer's chunks), a mixture's experts. Each chunk meets each layer as the
        chunked prefill's pass would have it - the same cache rows before it, the same rope, mask, placement and
        module, crossing the tier's edge the same way - so the rows come out as the chunked path's: a layer runs
        on the card (resident, or the template a card pass streams in, taken once for all the chunks) or on the
        host kernels, and a host layer's cache rides to the card and back around each chunk it runs there on."""
        B, T = ids.shape
        past0 = int(cache.get_seq_length())
        own = bool(self.fam.own)
        self.vram_policy(cache)
        self.ram_policy()
        self.lend_policy()
        self.cache_room(cache, B, T)
        self._tag_tiers(self.L)
        self._tag_quant()
        self._attn_ctx = cache
        cd = self.compute_dtype
        spans = [(a, min(T, a + C)) for a in range(0, T, C)]
        # a chunk the chunked prefill would take on the card: every chunk past the first continues a batch there
        card = self._card_chunks(C, T)
        if any(card):
            self._tag(PassTag.PREFILL_CARD)
        tier_owns_attention = (
            getattr(self, "kv_host", False)
            and not own
            and self.fam.fast
            and all(i in self.resident for i, lt in enumerate(self.layer_types) if lt == LayerKind.FULL)
        )
        self.log(f"[prefill] {T} tokens layer by layer in {len(spans)} chunks of {C} (each layer read once)")

        def embed(a: int, b: int) -> torch.Tensor:
            """a chunk's rows into the first layer, as `forward` embeds them"""
            h = self.embed(ids[:, a:b])
            if cd is not None:
                h = h.to(cd)
            if self.fam.streams > 1:
                h = h.repeat(1, 1, self.fam.streams)
            return h

        def frame(a: int, b: int, like: torch.Tensor) -> dict[str, Any]:
            """a chunk's pass preamble, as `forward` builds it for these rows at this position: its positions, and
            the dtype and device its rope is made in (`like`, the embedded rows'); the rope itself is made for each
            layer's pass (`rope`), not held for the sweep"""
            past = past0 + a
            text_pos, rope_pos = self._positions(B, b - a, past, None)
            return {
                "past": past,
                "text_pos": text_pos,
                "rope_pos": rope_pos,
                "like": like.new_empty((B, 0, like.shape[-1])),
                "ple": self.fam.ple_ids(self.cfg, ids[:, a:b], None),
            }

        def rope(f: dict[str, Any], n: int) -> Any:
            like = f["like"]
            if own:
                return self.rotary(like, self._positions(B, f["past"] + n, 0, None)[1])
            if self.fam.dual_rope:
                return {lt: self.rotary(like, f["text_pos"], lt) for lt in set(self.layer_types)}
            return self.rotary(like, f["rope_pos"] if self.fam.mrope else f["text_pos"])

        # the rows between layers stay on the card while they take no more than a quarter of its room; past that
        # they wait in pinned RAM, in the dtype they came out in (a copy, never a cast), each chunk's buffer drawn
        # from the host's share of the sweep's reservation as it is made
        row_b, park = self._sweep_rows(B, T)
        parked: list[torch.Tensor | None] = [None] * len(spans)
        host_dev = torch.device("cpu")
        # on the card, every chunk's rows in one buffer made before the working set, each layer's written back into
        # its chunk's place: a layer's output kept past the chunk that made it would sit inside the working set's
        # block for the next layers, and split it for good
        on_card_rows: list[torch.Tensor | None] = [None]

        def keep(c: int, h: torch.Tensor) -> torch.Tensor:
            if h.device.type == "cpu":
                return h
            if not park:
                buf = on_card_rows[0]
                if buf is None or buf.dtype != h.dtype or buf.shape[-1] != h.shape[-1] or buf.device != h.device:
                    buf = on_card_rows[0] = torch.empty((B, T, h.shape[-1]), dtype=h.dtype, device=h.device)
                a, b = spans[c]
                view = buf[:, a:b]
                if view.data_ptr() != h.data_ptr():
                    view.copy_(h)
                return view
            buf = parked[c]
            if buf is None or buf.shape != h.shape or buf.dtype != h.dtype:
                nbytes = 1 << max(0, h.numel() * h.element_size() - 1).bit_length()
                sched = getattr(self, "scheduler", None)
                if sched is not None:
                    sched.grant(
                        nbytes, "prefill", requester="a prefill chunk's rows, parked", device=host_dev, draws=PREFILL
                    )
                buf = parked[c] = torch.empty(h.shape, dtype=h.dtype, pin_memory=True)
            buf.copy_(h)
            return buf

        def chunk_pass(c: int, i: int, lt: str, on_card_now: bool, h: torch.Tensor) -> _Pass:
            f, (a, b) = frames[c], spans[c]
            causal = None
            if lt != LayerKind.LINEAR and not tier_owns_attention:
                # this layer's own cache length is the chunk's position now: the mask the chunked pass built
                causal = self._causal(h, None, cache, f["text_pos"], True, layer_idx=i)
            return _Pass(
                cache=cache,
                pe=rope(f, b - a),
                text_pos=f["text_pos"],
                causal=causal,
                linear_mask=None,
                ple_ids=f["ple"],
                T=b - a,
                past=f["past"],
                am=None,
                batched=True,
                own=own,
                card_pass=on_card_now,
                n_layers=self.L,
                on_layer=None,
            )

        t0 = time.time()
        by_kind: dict[str, list[Any]] = {}
        self._sweep_keep = True
        # the next layer's experts read ahead while a layer's chunks compute (`_ExpertStore.lookahead`, `sweep`)
        self._sweep_ahead = os.environ.get("BTB_PREFILL_AHEAD", "1") != "0"
        store = getattr(self, "expert_store", None)
        s0 = dict(store.stat) if store is not None else {}
        on_cuda = self.dev.type == Device.CUDA
        hq = int(self.cfg.num_attention_heads)
        kv_heads = int(getattr(self.cfg, "num_key_value_heads", None) or hq)
        head_dim = int(getattr(self.cfg, "head_dim", None) or self.cfg.hidden_size // hq)
        # off the card the whole sweep runs on the host: its working set is the host's share
        work_host = 0 if on_cuda else sum(self._sweep_bytes(C, past0, B, T, cache)[:1])
        own_epoch = False
        hs: list[torch.Tensor] = []
        frames: list[dict[str, Any]] = []
        scores_open = False
        try:
            if on_cuda:
                # the allocator's cached blocks given back first, so what the ledger reads as free is memory the card
                # can hand out: a block cached for one size cannot be made into a buffer of another without emptying
                # the cache, the sync and the flush the ledger was meant to spare
                self.vram_trim("layer-by-layer prefill")
                # the pass asks the ledger first, as a cache's growth does (`cache_room`): room on the card for its own
                # working set - the last chunk's activations and scores (`_chunk_bytes`) and a mixture's grouped call
                # over its rows (`_grouped_bytes`) - the rows kept between layers, and the cache's growth past what an
                # epoch has reserved for it. btb gives up what it holds for that, or refuses the prefill, before anything
                # is allocated
                work, beside = self._sweep_bytes(C, past0, B, T, cache)
                growth = self.cache_growth(cache, B, T, peak=True).get(self.dev, 0)
                epoch = int(self.device.reserved(self.dev)) - int(self.device.reserved(self.dev, but=EPOCH))
                kept = work + (0 if park else T * row_b) + self._hop_bytes(C, past0, B, T)
                self._make_room(self.dev, work + beside, f"the layer-by-layer prefill's chunks of {C} rows")
                # spoken for in the ledger for the sweep: the working set and the rows under the prefill's own tag, and
                # the cache's growth as the epoch's KV where no epoch reserved it (a prefill outside a generate) - the
                # cache's grants spend that one as their own, and nothing else reads either as free. The working set
                # is allocated piecemeal by the layers' own ops, which the ledger never sees one by one: what of it is
                # live is what the card holds past the sweep's start, less what the ledger knows of already (the
                # resident layers' KV, granted from the epoch's room, and the depot's blocks, granted and lent), and
                # the reservation holds only the rest - the free reading counts the live part already. Live means
                # allocated: the working set's blocks torch caches between chunks stay under the reservation, since
                # the free reading counts torch's cache as reclaimable and would otherwise hand them to anyone
                # free-read: the sweep's own segments measured against its reservation, the ledger's `used`
                held0 = int(torch.cuda.memory_allocated(self.dev))
                kv0 = _kv_on_card(cache, self.host)

                def in_use() -> int:
                    depot = getattr(self, "_depot", None)
                    held = int(depot.stat["held"]) if depot is not None else 0
                    grown = _kv_on_card(cache, self.host) - kv0
                    # the card allocated past the sweep's start, less the KV and the depot the ledger knows of already
                    # free-read: the same measure, as the ledger asks for it
                    return int(torch.cuda.memory_allocated(self.dev)) - held0 - grown - held

                self.device.reserve(PREFILL, kept, self.dev, used=in_use)
                if growth > epoch:
                    # the prompt's KV is the epoch's: its reservation raised to cover it (the cache's grants spend it as
                    # the rows are allocated), and taken for the sweep where no epoch reserved any
                    self.device.reserve(EPOCH, growth, self.dev)
                    own_epoch = epoch == 0
                self._presize_kv(cache, past0 + T, B)
                if self.fam.moe and os.environ.get("BTB_PREFILL_DEPOT", "1") != "0":
                    from .experts import LayerDepot

                    # the depot takes its blocks now, a layer's experts' worth as far as the ledger has room past
                    # all of that, before the working set below is cut: once the pass's buffers split their one block,
                    # the card's room sits inside it, where the depot's own pool cannot reach
                    self._depot = LayerDepot(self.dev, self.device, getattr(self, "scheduler", None))
                    store = getattr(self, "expert_store", None)
                    form = store.form() if store is not None and self.grouped_experts else None
                    if form is not None:
                        n = self._depot.open_at(form, int(self.n_experts))
                        self.log(f"[prefill] depot opened up front, {n} seats of {int(self.n_experts)} and its scratch")
            host_share = self._host_bytes(C, past0, B, T, park) if on_cuda else work_host
            if host_share:
                # the host's share, asked of it as the card's is: the host chunks' working set and the parked rows,
                # under the same tag there (the parked buffers draw on it as they are made)
                self._make_room(host_dev, host_share, f"the layer-by-layer prefill's host share, chunks of {C} rows")
                self.device.reserve(PREFILL, host_share, host_dev)
            # the rows embedded now the room is made: on the card, or straight into their parked buffers
            for c, (a, b) in enumerate(spans):
                h = embed(a, b)
                frames.append(frame(a, b, h))
                hs.append(keep(c, h))
                del h
            if on_cuda and self.dev.type == Device.CUDA:
                # what the family's attention holds across the chunks, sized for the last (gpt-oss's scores)
                scores_open = self.fam.open_sweep(self, C, past0 + T, B)
            # the card's allocator as the sweep goes: what it holds, what it keeps reserved, and how often it ran out and
            # emptied its cache to retry (each retry a device-wide sync and fresh allocations after it)
            retries0 = int(torch.cuda.memory_stats(self.dev).get("num_alloc_retries", 0)) if on_cuda else 0

            def card_line() -> str:
                if not on_cuda:
                    return ""
                ms = torch.cuda.memory_stats(self.dev)
                return (
                    f"; card {ms.get('allocated_bytes.all.current', 0) / 2**30:.1f} GB held, "
                    f"{ms.get('reserved_bytes.all.current', 0) / 2**30:.1f} reserved, "
                    f"{int(ms.get('num_alloc_retries', 0)) - retries0} allocator retries; the ledger holds "
                    + ", ".join(f"{k} {n / 2**30:.2f}" for k, n in self.device.spoken_for(self.dev).items())
                    + " GB spoken for"
                )

            # a drive layer a host chunk runs is read into the ring's slot and held there for all its host chunks; a
            # card chunk takes the layer as the card pass's template, read on its own
            ring = bool(self.cold) and not all(card)
            if ring:
                self._cold_start(self.L)
            try:
                with self.device.hold() as place:
                    proto = _Pass(card_pass=any(card), n_layers=self.L, place=None)
                    if self.prefetch:
                        self._prefetch_next(0, proto)
                    for i in range(self.L):
                        lt = self.layer_types[i]
                        tl, es = time.perf_counter(), float(self.expert_stat.get("s", 0.0))
                        host = place.tier(i) in (LayerTier.HOST, LayerTier.COLD)
                        tmpl = self._card_layer(i, proto) if (not host or any(card)) else None
                        if ring and i in self.cold:
                            self._cold_wait(i)
                            self._cold_held = i
                        for c, (a_c, b_c) in enumerate(spans):
                            h = hs[c]
                            if host and not card[c]:
                                if h.device.type != "cpu" or h.dtype != torch.float32:
                                    # widened on the host, where the host chunk's share was asked for it
                                    h = h.detach().cpu().float()
                                pas = chunk_pass(c, i, lt, False, h)
                                pas.place = place
                                hs[c] = keep(c, self._run_host_layer(i, h, pas))
                                continue
                            assert tmpl is not None  # a card chunk of a host layer took the template above
                            wd = self._layer_dtype(tmpl)
                            if h.device != self.dev or (wd is not None and h.dtype != wd):
                                h = h.to(self.dev, wd) if wd is not None else h.to(self.dev)
                            cl = cache.layers[i] if host else None
                            if isinstance(cl, GrowLayer) and not cl.shared and frames[c]["past"] == 0:
                                # no rows yet: a buffer on the card for this chunk's alone, in the prefill's
                                # reservation - grown there, it would be the whole sequence's, from the epoch's room
                                cl.hop(self.dev, b_c - a_c, h.dtype, (h.shape[0], kv_heads, head_dim))
                            elif host and frames[c]["past"] > 0:
                                if isinstance(cl, GrowLayer) and not cl.shared and cl.keys.numel():
                                    # the rows onto the card with room for this chunk's: written in place there, in
                                    # the prefill's reservation (`_hop_bytes`), nothing grown or granted
                                    cl.hop(self.dev, b_c - a_c, h.dtype)
                                else:
                                    self._cache_to(cache, i, self.dev)
                            pas = chunk_pass(c, i, lt, True, h)
                            pas.place = place
                            h = self._run_card_layer(i, tmpl, h, pas)
                            if host:
                                self._cache_to(cache, i, "cpu")
                            hs[c] = keep(c, h)
                        if self._cold_held is not None:
                            self._cold_held = None
                            self._cold_release(i)
                        # the layer's time, its experts' share apart: where a sweep's time goes, by layer kind
                        self._sync()
                        kind = by_kind.setdefault(lt, [0, 0.0, 0.0])
                        kind[0] += 1
                        kind[1] += time.perf_counter() - tl
                        kind[2] += float(self.expert_stat.get("s", 0.0)) - es
                        if (i + 1) % 8 == 0 or i + 1 == self.L:
                            self.log(f"[prefill] layer {i + 1}/{self.L} done, {time.time() - t0:.0f}s{card_line()}")
                    # the last row, its own: the parked buffers are let go as the sweep ends
                    last = hs[-1][:, -1:].clone()
            finally:
                if self._cold_held is not None:
                    held, self._cold_held = self._cold_held, None
                    self._cold_release(held)
                self._sweep_keep = False
                self._sweep_ahead = False
                store = getattr(self, "expert_store", None) or store
                if store is not None:
                    store.sweep_end()
                    d = {k: store.stat.get(k, 0) - s0.get(k, 0) for k in ("miss", "wait_s", "s", "ahead", "ahead_used")}
                    self.log(
                        f"[prefill] store: {int(d['miss'])} read, {d['wait_s']:.1f} s waited on the drive, "
                        f"{d['s']:.1f} s in its own bookkeeping; {int(d['ahead'])} read ahead, "
                        f"{int(d['ahead_used'])} of them used"
                    )
                depot, self._depot = getattr(self, "_depot", None), None
                if depot is not None:
                    depot.close()
                    st = depot.stat
                    if st["refused"] and not (st["seated"] or st["scratch"]):
                        self.log(
                            f"[prefill] depot: never opened ({int(st['refused'])} calls found no room for its "
                            f"scratch slots); the per-expert loop took every call"
                        )
                    if st["seated"] or st["scratch"]:
                        self.log(
                            f"[prefill] depot: {st['seated']} experts seated in {int(st['blocks'])} blocks grown as asked "
                            f"({st['held'] / 2**30:.1f} GB held; {st['bytes'] / 2**30:.1f} GB over the bus, "
                            f"{st['upload_s']:.1f} s of it holding the host), {st['reused']} reused by a later chunk, "
                            f"{st['scratch']} through scratch, {st['passed']} not seated"
                        )
                if by_kind:
                    self.log(
                        "[prefill] time by layer kind: "
                        + "; ".join(
                            f"{lt} {n} layers {wall:.1f} s (experts {ex:.1f} s)"
                            for lt, (n, wall, ex) in by_kind.items()
                        )
                    )
        finally:
            # the sweep's reservations given back and its depot let go however it ended - a refusal, a fault, or the
            # sweep's own cleanup raising: left standing, every later reading of the card's free memory would count
            # them, and the next pass would find a depot with nothing behind it
            if scores_open:
                self.fam.close_sweep(self)
            if self._sweep_keep or self._sweep_ahead:
                # a sweep refused before its layers ran (or whose own cleanup raised first): no later pass runs as one
                # of its chunks - its experts kept, the lookahead reading a sweep's worth ahead
                self._sweep_keep = self._sweep_ahead = False
                if store is not None:
                    store.sweep_end()
            self.device.release(PREFILL)
            if own_epoch:
                self.device.release(EPOCH)
            if park:
                # the parked rows' pinned buffers back to the OS, not kept in torch's pinned cache where the host's
                # free reading would count them taken after the sweep let them go
                parked.clear()
                hs.clear()
                torch._C._host_emptyCache()
            depot, self._depot = getattr(self, "_depot", None), None
            if depot is not None:
                depot.close()
        self._flush_events()
        h = self._norm_input(last)
        h = self._final_norm(h)
        return self._apply_head(h.to(cd if cd is not None else h.dtype))

    def _attn_card_blocks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kf: torch.Tensor,
        vf: torch.Tensor,
        past: int,
        scale: float,
    ) -> torch.Tensor:
        B, Hq, T, d = q.shape
        Hk = k.shape[1]
        g = Hq // Hk
        op = torch.ops.aten._scaled_dot_product_efficient_attention

        def rep(t: torch.Tensor) -> torch.Tensor:
            if g == 1:
                return t.contiguous()
            n = t.shape[2]
            return t[:, :, None].expand(B, Hk, g, n, d).reshape(B, Hq, n, d).contiguous()

        qf = q.contiguous()
        acc = m_run = None

        def fold(o: Any, l: Any) -> None:
            nonlocal acc, m_run
            l = l[..., :T].float()
            if acc is None:
                acc, m_run = o.float(), l
                return
            m_new = torch.logaddexp(m_run, l)
            acc.mul_(torch.exp(m_run - m_new)[..., None]).add_(o.float().mul_(torch.exp(l - m_new)[..., None]))
            m_run = m_new

        Bk = int(self.kv_block)
        if past > 0:
            need = B * Hk * Bk * d
            if self._kv_stage is None or self._kv_stage[0][0].numel() < need or self._kv_stage[0][0].dtype != kf.dtype:
                self._kv_stage = [
                    [
                        torch.empty(need, dtype=kf.dtype, pin_memory=True),
                        torch.empty(need, dtype=kf.dtype, pin_memory=True),
                        None,
                    ]
                    for _ in range(2)
                ]
            j = 0
            for a in range(0, past, Bk):
                n = min(Bk, past - a)
                st = self._kv_stage[j]
                if st[2] is not None:
                    st[2].synchronize()
                sk = st[0][: B * Hk * n * d].view(B, Hk, n, d)
                sv = st[1][: B * Hk * n * d].view(B, Hk, n, d)
                sk.copy_(kf[:, :, a : a + n])
                sv.copy_(vf[:, :, a : a + n])
                kb = sk.to(self.dev, non_blocking=True)
                vb = sv.to(self.dev, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                st[2] = ev
                o, l, _, _ = op(qf, rep(kb), rep(vb), None, True, 0.0, False, scale=scale)
                fold(o, l)
                j ^= 1
        o, l, _, _ = op(qf, rep(k), rep(v), None, True, 0.0, True, scale=scale)
        if acc is None:
            return o
        fold(o, l)
        return acc.to(q.dtype)

    def _kv_split(self, tmpl: Any, i: int, h: torch.Tensor, pe: PassRope, cache: Any) -> torch.Tensor:
        apply_rotary_pos_emb = self._rope_fn()  # the one-row step's own (its graph rotates with it)
        B, T, _ = h.shape
        cl = cache.layers[i]
        past = cl.keys.shape[-2] if getattr(cl, "keys", None) is not None and cl.keys.numel() else 0
        lt = self.layer_types[i]
        win = layer_window(self.cfg, lt)
        rope = pe_for(pe, lt)
        assert rope is not None  # a resident layer's pass always carries the rope
        residual = h
        x = tmpl.input_layernorm(h)
        at = tmpl.self_attn
        hd = at.head_dim
        gate = None
        if hasattr(at, "qkv_proj"):  # q, k and v as one projection (Phi-3's layout)
            qkv = at.qkv_proj(x)
            nq = self.cfg.num_attention_heads * hd
            nk = at.num_key_value_heads * hd
            q = qkv[..., :nq].view(B, T, -1, hd)
            k = qkv[..., nq : nq + nk].view(B, T, -1, hd)
            v = qkv[..., nq + nk :].view(B, T, -1, hd)
        elif self.fam.attn_gate:
            qg = at.q_proj(x).view(B, T, -1, hd * 2)
            q, gate = torch.chunk(qg, 2, dim=-1)
            gate = gate.reshape(B, T, -1)
            q = at.q_norm(q.reshape(B, T, -1, hd))
            k = at.k_norm(at.k_proj(x).view(B, T, -1, hd))
            v = at.v_proj(x).view(B, T, -1, hd)
        else:
            q = at.q_norm(at.q_proj(x).view(B, T, -1, hd))
            k = at.k_norm(at.k_proj(x).view(B, T, -1, hd))
            v = at.v_proj(x).view(B, T, -1, hd)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), rope[0], rope[1])
        v = v.transpose(1, 2)
        kc, vc = k.cpu(), v.cpu()
        kf, vf = cache.update(kc, vc, i)
        probe = getattr(self, "_probe", None)
        if probe is not None:
            probe(i, q, kf, vf, at.scaling)
        parents: Any = getattr(self, "ap", None) if getattr(self, "aq", False) else None
        tree = parents is not None and any(parents[j] != j - 1 for j in range(T))
        # a one-row step, or a verify pass (between `aa` and `ab`) whose rows must be the steps' own; a prefill's rows
        # need no step's bits (plain and speculative decodes prefill alike) and keep the card's blocks
        native = (
            (T == 1 or bool(getattr(self, "aq", False)))
            and B == 1
            and Native.attn_decode is not None
            and kf.device.type == "cpu"
            and kf.dtype in (torch.bfloat16, torch.float32)
            and vf.dtype == kf.dtype
            and kf.stride(-1) == 1
            and kf.stride(-2) == kf.shape[-1]
            and vf.stride(-1) == 1
            and vf.stride(-2) == vf.shape[-1]
        )
        if native:
            # every row as a one-row step computes it: btb's kernel over the row's own keys in order - the prefix (its
            # last `win` under a window), then its ancestors (the chain before it), then itself - so a verify pass
            # gives each row the bits of the steps along its path and a speculative decode is the plain loop's (the
            # card's blocks and torch's sdpa each sum in an order of their own, and parted from the steps at a
            # near-tie). A chain row's keys are one run of the cache; a tree row's are gathered into one
            path = parents if parents is not None else list(range(-1, T - 1))
            qf = q[0].float().cpu()
            out = torch.empty(T, qf.shape[0], qf.shape[2], dtype=torch.float32)
            with torch.profiler.record_function("btb_attn_decode"):
                for p in range(T):
                    allow = node_mask(past, p, path, win)
                    if allow is None:
                        kp, vp = kf[0][:, : past + p + 1], vf[0][:, : past + p + 1]
                    else:
                        seen = allow.nonzero().flatten()
                        lo, n = int(seen[0]), int(seen.numel())
                        if int(seen[-1]) - lo + 1 == n:  # one run: a window's cut, the chain's rows before it
                            kp, vp = kf[0][:, lo : lo + n], vf[0][:, lo : lo + n]
                        else:
                            kp, vp = kf[0][:, seen].contiguous(), vf[0][:, seen].contiguous()
                    Native.attn_decode(qf[:, p].contiguous(), kp, vp, at.scaling, out[p])
            attn = out.view(1, T, qf.shape[0], qf.shape[2])
        elif T > 1 and not tree and q.device.type == "cuda" and not win:
            attn = self._attn_card_blocks(q, k, v, kf, vf, past, at.scaling).transpose(1, 2)
        elif tree:
            qc = q.cpu()
            outs = []
            for p in range(T):
                allow = torch.zeros(past + T, dtype=torch.bool)
                rows = node_mask(past, p, parents, win)
                allow[: past + p + 1] = True if rows is None else rows
                neg = torch.zeros(past + T, dtype=qc.dtype).masked_fill(~allow, float("-inf"))
                a = F.scaled_dot_product_attention(
                    qc[:, :, p : p + 1], kf, vf, attn_mask=neg.view(1, 1, 1, -1), enable_gqa=True, scale=at.scaling
                )
                outs.append(a.transpose(1, 2))
            attn = torch.cat(outs, dim=1)
        else:
            qc = q.cpu()
            mask = None
            if T > 1 or win:
                pos = torch.arange(past, past + T)[:, None]
                j = torch.arange(past + T)[None, :]
                allow = j <= pos
                if win:
                    allow &= j > pos - win
                mask = (
                    torch.zeros(T, past + T, dtype=qc.dtype).masked_fill(~allow, float("-inf")).view(1, 1, T, past + T)
                )
            attn = F.scaled_dot_product_attention(
                qc, kf, vf, attn_mask=mask, enable_gqa=True, scale=at.scaling
            ).transpose(1, 2)
        a1 = attn.reshape(B, T, -1).to(h.device, dtype=x.dtype)
        if gate is not None:
            a1 = a1 * torch.sigmoid(gate)
        mix = at.o_proj(a1)
        if self.fam.sandwich:
            h = residual + tmpl.post_attention_layernorm(mix)
            return h + tmpl.post_feedforward_layernorm(tmpl.mlp(tmpl.pre_feedforward_layernorm(h)))
        h = residual + mix
        return h + tmpl.mlp(tmpl.post_attention_layernorm(h))
