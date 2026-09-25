# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The forward pass: positions and masks, the prefill (whole, chunked, by layer, with the attention on the card
in blocks), the head, and the transformers-side forward over the tiers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .. import mlx as mlxdev
from ..kinds import LayerKind, Parents, PassTag, TokenRows
from ..options import Device
from ..sampling import as_pick
from ..sysinfo import host_free_bytes, host_total_bytes
from .native import Native
from .state import _State

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
        if own:
            if positions is not None:
                prev = torch.arange(past, device=self.dev).view(1, 1, -1).expand(3, B, -1)
                rope_all = torch.cat([prev, rope_pos], dim=-1)
            else:
                rope_all = self._positions(B, past + T, 0, am)[1]
            pe = self.rotary(h, rope_all)
        elif self.fam.dual_rope:
            # one rope per layer type (Gemma 3's local/global split); each layer reads its own from the pass
            pe = {lt: self.rotary(h, text_pos, lt) for lt in set(self.layer_types)}
        else:
            pe = self.rotary(h, rope_pos if self.fam.mrope else text_pos)
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
        linear_mask = None if (am is None or bool(torch.all(am == 1))) else am[:, -T:]
        ple_ids = None
        if own:
            eos = self.cfg.eos_token_id
            eos = eos[0] if isinstance(eos, (list, tuple)) else eos
            ple_ids = (
                ids if linear_mask is None else torch.where(linear_mask.bool(), ids, torch.full_like(ids, int(eos)))
            )
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
            for i in self.host:
                if i < n_layers:
                    self._cache_to(cache, i, self.dev)

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
        if h.device != self.dev:
            h = h.to(self.dev)
        if last_only:
            h = h[:, -1:, :]
        h = self._final_norm(h)
        cd = self.compute_dtype if self.compute_dtype is not None else h.dtype
        hf = h.to(cd)
        if not head:
            return hf
        return self._apply_head(hf)

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
        tmpl = self.host[i]
        hc = pas.host_side()
        if i in self.cold:
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
            kw = self._layer_kw(
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
        if i in self.cold:
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
                **self._layer_kw(lt, pas.causal, pas.linear_mask, pas.text_pos, pas.ple_ids),
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
        with the whole prefix, as gpt-oss's do - one per layer type, which `_layer_kw` picks from."""
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

    def _layer_kw(self, lt: str, causal: Any, linear_mask: Any, pos: torch.Tensor, ple_ids: Any) -> dict[str, Any]:
        if self.fam.own:
            return {"attention_mask": causal, "conv_mask": linear_mask, "ple_input_ids": ple_ids}
        if isinstance(causal, dict):
            causal = causal.get(lt)
        return {"attention_mask": linear_mask if lt == LayerKind.LINEAR else causal, "position_ids": pos}

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
        """the memory a prefill's working set may take right now, on the tier it runs on: the card's free VRAM
        above the margin; on unified memory the tightest of the engine's own ledger, Metal's working-set limit
        less what MLX holds, and the machine's free RAM as it stands now (the other programs' share included)
        above the OS's floor; the host's free RAM above a gigabyte elsewhere"""
        on_card = self.dev.type == Device.CUDA and (bool(self.resident) or bool(self.prefill_card))
        if on_card:
            return int(torch.cuda.mem_get_info(self.dev)[0] - self.vram_margin)
        mlx = getattr(self, "mlx", None)
        if mlx is not None:
            held = int(mlx.held_bytes())
            room = int(self.device.free(unreserved=True) or 0)
            limit = int((getattr(mlx, "info", None) or {}).get("max_recommended_working_set_size", 0) or 0)
            if limit:
                room = min(room, limit - held)
            live = int(host_free_bytes()) - int(self.OS_FLOOR * host_total_bytes())
            return max(0, min(room, live))
        return int(host_free_bytes() - (1 << 30))

    def _auto_chunk(self, past: int = 0) -> int:
        """The rows a prefill pass takes at once: the largest power of two, 64 to 4096, whose working set fits the
        room - each row's activations through the widest layer, and, where the attention materializes its
        scores, each row's scores against every key it sees (`past` and the chunk's own rows), which is what
        grows with the context and once asked Metal for 33 GB in one buffer."""
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
        on_card = self.dev.type == Device.CUDA and (bool(self.resident) or bool(self.prefill_card))
        fp32 = self.compute_dtype is not None and self.compute_dtype != torch.bfloat16
        nb = 4 if (fp32 or not on_card) else 2
        per_token = 2 * nb * (3 * I + 8 * H + 2 * (Hq + 2 * Hk) * d)
        if getattr(self, "mlx", None) is not None and LayerKind.LINEAR in self.layer_types:
            # the hybrid's chunked DeltaNet rule keeps its per-chunk states and blocks beside the activations:
            # on Qwen3.5-4B a 4096-row chunk at position 0 costs 4.9-6.2 GB against the 2.4 the
            # activations alone price
            per_token *= 2
        free = self.prefill_room()
        scores = 0
        buf_cap = free  # the single attention buffer must fit the reserve-aware room (--ram/--vram-reserve, via
        if getattr(self, "mlx", None) is not None:  # prefill_room) and, on MLX, Metal's hard per-buffer ceiling
            hard = int((getattr(self.mlx, "info", None) or {}).get("max_buffer_length", 0) or 0)
            if hard:
                buf_cap = min(buf_cap, hard)
            if d not in self.MLX_FUSED_HEAD:
                scores = 4 * Hq  # bytes per (row, key) pair of materialized scores, float32 through the softmax
            else:
                # the fused kernel's per-key-block partials, (T, Hk, splits*8, g, D) float32 and two without D,
                # folded after: about half the materialized figure a (row, key), and still growing with every key
                # the chunk sees
                from ..mlx.attn import ATTN_BLOCK

                scores = -(-32 * Hq * (d + 2) // ATTN_BLOCK)
        rows = 4096
        # shrink until the whole working set fits the room, and until the attention's own buffer at these rows
        # fits buf_cap - so a free reading that runs high never passes a single buffer the GPU cannot allocate,
        # the oversized ask that faults it
        while rows > PREFILL_MIN_ROWS and (
            rows * (per_token + scores * (int(past) + rows)) > free or scores * rows * (int(past) + rows) > buf_cap
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
        # the host path's DeltaNet takes a prompt whole; the MLX path continues a chunk from the stored states
        whole_hybrid = LayerKind.LINEAR in self.layer_types and not self.fam.own and self.mlx is None
        if cache is None or attention_mask is not None or getattr(self, "aq", False) or whole_hybrid:
            return self.forward(ids, cache=cache, on_layer=on_layer, attention_mask=attention_mask, last_only=last_only)
        past = int(cache.get_seq_length())
        C = int(self.prefill_chunk or self._auto_chunk(past))
        if T <= C:
            return self.forward(ids, cache=cache, on_layer=on_layer, last_only=last_only)
        # a mixture of experts with the whole trunk resident prefills layer by layer, so a layer's experts
        # are read once for the whole prompt instead of once per chunk
        if self.fam.moe and not self.host and len(self.resident) == self.L and on_layer is None and last_only:
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
        B, T = ids.shape
        past0 = cache.get_seq_length()
        self.log(f"[prefill] {T} tokens layer by layer in chunks of {C} (each layer's experts read once)")
        h_all = self.embed(ids)
        if self.compute_dtype is not None:
            h_all = h_all.to(self.compute_dtype)
        if self.fam.streams > 1:
            h_all = h_all.repeat(1, 1, self.fam.streams)
        h_all = h_all.cpu().pin_memory() if self.dev.type == Device.CUDA else h_all
        rope_all = self._positions(B, past0 + T, 0, None)[1]
        eos = self.cfg.eos_token_id
        eos = eos[0] if isinstance(eos, (list, tuple)) else eos
        starts = list(range(0, T, C))
        t0 = time.time()
        self._sweep_keep = True
        try:
            for i in range(self.L):
                tmpl = self.resident[i]
                lt = self.layer_types[i]
                if lt in self.shadow and not self.resident_fp32:
                    tmpl = self._upcast(lt, tmpl, i)
                for a in starts:
                    b = min(T, a + C)
                    h = h_all[:, a:b].to(self.dev, non_blocking=True)
                    past = past0 + a
                    text_pos = torch.arange(past, past + (b - a), device=self.dev).view(1, -1).expand(B, -1)
                    # the own-layer family's rotary takes every position up to here, the others this chunk's; a
                    # dual-rope family reads the rope for this layer's type
                    if self.fam.dual_rope:
                        pe = self.rotary(h, text_pos, lt)
                    else:
                        pe = self.rotary(h, rope_all[:, :, : past + (b - a)] if self.fam.own else text_pos)
                    causal = None
                    if lt != LayerKind.LINEAR:
                        causal = self._causal(h, None, cache, text_pos, True, layer_idx=i)
                    h = tmpl(
                        h,
                        position_embeddings=pe,
                        past_key_values=cache,
                        use_cache=True,
                        **self._layer_kw(lt, causal, None, text_pos, ids[:, a:b]),
                    )
                    h_all[:, a:b].copy_(h, non_blocking=True)
                self._sync()
                if (i + 1) % 8 == 0 or i + 1 == self.L:
                    self.log(f"[prefill] layer {i + 1}/{self.L} done, {time.time() - t0:.0f}s")
        finally:
            self._sweep_keep = False
        h = h_all[:, -1:].to(self.dev)
        h = self._final_norm(h)
        cd = self.compute_dtype if self.compute_dtype is not None else h.dtype
        return self._apply_head(h.to(cd))

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
        apply_rotary_pos_emb = self.fam.mod.apply_rotary_pos_emb
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
        if T > 1 and not tree and q.device.type == "cuda" and not win:
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
        elif (
            T == 1
            and B == 1
            and Native.attn_decode is not None
            and kf.device.type == "cpu"
            and kf.dtype in (torch.bfloat16, torch.float32)
            and vf.dtype == kf.dtype
            and kf.stride(-1) == 1
            and kf.stride(-2) == kf.shape[-1]
            and vf.stride(-1) == 1
            and vf.stride(-2) == vf.shape[-1]
        ):
            qf = q[0, :, 0].float().cpu().contiguous()
            out = torch.empty(qf.shape, dtype=torch.float32)
            first = max(0, past + 1 - win) if win else 0  # a sliding layer's last rows alone
            with torch.profiler.record_function("btb_attn_decode"):
                Native.attn_decode(qf, kf[0][:, first:], vf[0][:, first:], at.scaling, out)
            attn = out.view(1, 1, qf.shape[0], qf.shape[1])
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
