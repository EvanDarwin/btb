# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's fast paths: the captured CUDA graphs of the one-token step and the speculative tree's verify
pass over the host-side attention cache."""

from __future__ import annotations

import contextlib
import ctypes
import functools
import gc
import itertools
import os
import sys
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from .. import mlx as mlxdev
from .. import trace
from ..kinds import LayerKind, NodePath, Parents, PassTag, Tokens
from ..options import Device
from ..sampling import GREEDY
from . import device as device_mod
from .arena import KvArena
from .cache import CardRowsLayer, GrowLayer, attention_rows, forked, indexer_keys, set_rows
from .families import act_name
from .fixed_rows import KeyRows
from .forward import chain_of, layer_window, node_mask, pe_for
from .fused import _fused_rope
from .native import Native, _Cuda, kernels_path
from .scheduler import MemoryGrantError, _size
from .spec_cost import SpecCost
from .state import _State

if TYPE_CHECKING:
    from .forward import PassRope, Rope

_KERNELS_WARNED = False


class CardPassFailed(RuntimeError):
    """A card pass that failed once its layer hook had seen a layer: no other path runs it again, or the hook would
    see those layers twice (`_forward_card_prefill`). Never the card out of memory (`_is_card_oom`), whatever the
    failure under it"""


def _card_warning(reason: str) -> None:
    """on stderr whatever the log callback, once a process: a run without the kernels must not be quiet"""
    global _KERNELS_WARNED
    if _KERNELS_WARNED:
        return
    _KERNELS_WARNED = True
    hip = getattr(torch.version, "hip", None)
    amd = (
        f"  torch {torch.__version__} is a ROCm build: this is an AMD card. btb lacks AMD support currently, \n"
        "and there is no HIP build available.\n"
        if hip
        else ""
    )
    rule = "!" * 100
    sys.stderr.write(
        f"\n{rule}\n"
        f"  WARNING: the card is running WITHOUT btb's kernels: {reason}\n"
        "  You will receive none of btb's performance benefits beyond its hardware scheduler, "
        "  this is like running torch directly.\n"
        f"{amd}{rule}\n\n"
    )
    sys.stderr.flush()


def _graphs_in(o: Any, seen: set[int] | None = None) -> Iterator[Any]:
    """every CUDA graph in a graph state's dicts, lists and tuples, once each"""
    seen = set() if seen is None else seen
    if id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, torch.cuda.CUDAGraph):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _graphs_in(v, seen)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _graphs_in(v, seen)


def _captured(exec_id: int) -> None:
    """the body of a step graph already captured: never run, the graph is replayed instead"""
    raise RuntimeError(f"[card] the step graph is captured; its body ({exec_id}) does not run again")


class _Warm:
    """a pass's L2 warming (`_CudaMixin._card_warm`): called after each matvec is launched, it forks `side` from
    the chain there and warms the next weight of `seq` - (weight, rows, cols, the warps its matvec cuts k into) -
    with `btb_l2_warm`, the share of each warp's slice read on the card from `share` (the card state's switch,
    moved between replays), so the warming runs beside the kernels between the two matvecs; `join()` brings `side`
    back into the chain at the pass's end, which a capture needs. It holds the pass's weights and pointers and
    nothing of the card's state, so nothing outlives the pass through it"""

    __slots__ = ("blocks", "i", "k", "seq", "share", "side", "sink")

    def __init__(
        self,
        k: Any,
        side: torch.cuda.Stream | None,
        sink: torch.Tensor | None,
        seq: list[tuple[torch.Tensor, int, int, int]],
        blocks: int,
        share: ctypes.c_void_p | None = None,
    ) -> None:
        self.k, self.side, self.sink, self.seq, self.blocks, self.share, self.i = k, side, sink, seq, blocks, share, 0

    def __call__(self) -> None:
        if self.side is None or self.i >= len(self.seq):
            return
        W, R, C, nw = self.seq[self.i]
        self.i += 1
        self.side.wait_stream(torch.cuda.current_stream())
        c = ctypes.c_int
        self.k.launch(
            "btb_l2_warm",
            (self.blocks, 1, 1),
            (256, 1, 1),
            [self.k.ptr(W), c(R), c(C), c(nw), self.share, self.k.ptr(self.sink)],
            stream=self.side,
        )

    def join(self) -> None:
        if self.side is not None:
            torch.cuda.current_stream().wait_stream(self.side)


class _Tuner:
    """the step loop's choice among ways to run the step that give the same bits and differ only in speed, by their
    speed as measured, interleaved with each other so every way meets the same load on the card. An arm is a
    setting of the card state's switches (`SW_*`, written on the card between replays, in stream order) and a lane,
    (its queue, its kernels): the step graph at the platform's unroll with one replay queued ahead ("usual"), or at
    one step a replay queued as deep as its execs allow ("deep"); in the plain kernels or the fused ones. Alone on
    the card the usual lane is the faster (fewer submissions); beside a program the driver time-slices the card
    with, the deep lane's short replays, waiting when a gap opens, run in the gaps it leaves where a longer replay
    spans them - so choosing the deep lane is noticing the card is shared. The fused kernels leave fewer edges where
    that program takes the card and btb's L2 with it.
    Replays are timed in windows of `window` (each replay's time a token on the card's clock), a window's median kept
    for its arm (the last `keep`); every arm is tried `rounds` times first, then the chosen arm runs, a window in
    every `explore` spent on the others in turn. The choice moves only on `least` windows of each arm and holds for
    `dwell` windows after it moves - beside a game a few lucky windows of the deep lane once moved it there and the
    next few moved it back, five times in three answers - and away from the usual lane only to one `margin` faster:
    beside a game the deep lane won by 15-20% where it paid, on a free card the two drifted up to 7% apart as their
    windows fell at different moments, and a 5% margin read that drift as a shared card twice in six answers; a move
    between lanes is said"""

    def __init__(
        self,
        arms: list[tuple[str, list[int], tuple[str, bool]]],
        dev: torch.device,
        window: int = 4,
        rounds: int = 2,
        explore: int = 8,
        keep: int = 9,
        margin: float = 0.10,
        least: int = 5,
        dwell: int = 12,
    ) -> None:
        self.labels = [a for a, _, _ in arms]
        self.lanes = [ln for _, _, ln in arms]
        self.table = torch.tensor([v for _, v, _ in arms], dtype=torch.int32, device=dev)
        self.window, self.rounds, self.explore, self.keep, self.margin = window, rounds, explore, keep, margin
        self.least, self.dwell = least, dwell
        self.moved = 0  # the window the choice last moved at
        # the windows between two exploring ones: doubled (to 64) each time the arm explored loses by more than twice
        # the margin, back to `explore` when one comes close - beside a game the deep lane ran 20-40% slower for a
        # whole run, and a window in eight spent on it cost the answer 5% - or when the chosen arm's own time moves
        # as far from where it stood then (`ref`): a change of load shows there first, and a stale median of an arm
        # explored once in 64 windows would hold the choice long after the load that made it
        self.every = explore
        self.ref: float | None = None
        self.ctx: int | None = None  # the power of two of the prefix the windows were timed over (`at_context`)
        self.shift = 0  # the chosen arm's windows in a row far from its own median: the load moving (`after_replay`)
        self.meds: list[list[float]] = [[] for _ in arms]
        self.windows = 0  # windows completed, over every generate call
        self.other = 0  # the next arm an exploring window tries
        self.chosen = 0  # the arm run outside the exploring windows
        self.arm = 0  # the arm the current window runs
        self.cur: int | None = None  # the arm the switches hold on the card
        self.acc: list[float] = []  # the current window's per-token seconds

    def at_context(self, past: int) -> None:
        """an answer starting over `past` rows: a step's time grows with the prefix its attention reads, so windows
        timed at another length compare nothing - a free card once read a 4k answer's usual lane (5.48 ms) against
        the deep lane's short-prompt windows (4.45) and said it was shared. Where the length's power of two moves, the
        windows go and both lanes are tried again; the choice stands until they say otherwise"""
        key = max(0, int(past)).bit_length()
        if self.ctx == key:
            return
        self.ctx = key
        self._afresh()

    def _afresh(self) -> None:
        """every arm's windows let go and every arm tried again, the choice standing until they say otherwise"""
        self.meds = [[] for _ in self.labels]
        self.acc = []
        self.windows, self.moved, self.every, self.ref, self.shift = 0, 0, self.explore, None, 0
        self.arm = 0

    def median(self, i: int) -> float | None:
        m = self.meds[i]
        return sorted(m)[len(m) // 2] if m else None

    def best(self) -> int:
        return self.chosen

    def _choose(self) -> tuple[str, bool] | None:
        """the chosen arm moved to the fastest: away from the first arm (the usual way, the one alone on the card
        takes) only where another is `margin` faster, back to it as soon as it is as fast - alone, the lanes differ by
        the deep one's submissions, less than the margin, so a margin both ways would keep the deep queue after the
        other program left; the line to say where the choice moved, and whether the console hears it (a move of the
        queue) or only the log (of the kernels)"""
        if any(len(m) < self.least for m in self.meds) or self.windows - self.moved < self.dwell:
            return None
        meds = [self.median(i) or 0.0 for i in range(len(self.meds))]
        if self.chosen != 0 and meds[0] <= meds[self.chosen]:
            b = 0
        else:
            b = min(range(len(meds)), key=lambda i: meds[i])
            if b == self.chosen or meds[b] >= meds[self.chosen] * (1.0 - self.margin):
                return None
        was, self.chosen, self.moved = self.chosen, b, self.windows
        (qb, fb), (qw, fw) = self.lanes[b], self.lanes[was]
        kern = " in the fused kernels" if fb else ""
        if qb != qw and qb == "deep":
            return (
                f"the card is shared: one-step replays queued deep{kern} run at {1e3 * meds[b]:.2f} ms a token "
                f"against {1e3 * meds[was]:.2f} - queuing deep, so btb runs in the gaps the other program leaves",
                True,
            )
        if qb != qw:
            return (
                f"queuing deep no longer pays: the usual replays{kern} run at {1e3 * meds[b]:.2f} ms a token against "
                f"{1e3 * meds[was]:.2f} queued deep - back to one replay queued ahead",
                True,
            )
        if fb != fw:
            return (
                f"the step's {'fused' if fb else 'plain'} kernels run at {1e3 * meds[b]:.2f} ms a token against "
                f"{1e3 * meds[was]:.2f} for the {'plain' if fb else 'fused'} ones - taking them",
                False,
            )
        return None

    def _next_arm(self) -> int:
        n = len(self.labels)
        if self.windows < n * self.rounds:
            return self.windows % n
        if n > 1 and self.windows % self.every == 0:
            self.other = (self.other + 1) % n
            if self.other == self.chosen:
                self.other = (self.other + 1) % n
            return self.other
        return self.chosen

    def before_replay(self, switch: torch.Tensor) -> int:
        """the arm the next replay runs, its switches written on the card where they change"""
        if self.cur != self.arm:
            switch.copy_(self.table[self.arm], non_blocking=True)
            self.cur = self.arm
        return self.arm

    def after_replay(self, arm: int, seconds_per_token: float) -> tuple[str, bool] | None:
        """a replay run by `arm` took `seconds_per_token` a token: kept, the window closed when full, and the choice
        made again; the line to say where the choice moved, and whether the console hears it (`_choose`)"""
        if arm != self.arm:
            return None
        self.acc.append(seconds_per_token)
        if len(self.acc) < self.window:
            return None
        m = self.meds[self.arm]
        w = sorted(self.acc)[len(self.acc) // 2]
        self.acc = []
        if self.arm == self.chosen and len(m) >= self.least:
            # the chosen arm far from its own median two windows running: the load moved (another program came or
            # went), and the other arms' windows, taken rarely and under the old load, would misname the fastest -
            # a game once read the deep fused lane's calm windows (4.5 ms) against the others' under it (7.6)
            mid = sorted(m)[len(m) // 2]
            self.shift = self.shift + 1 if abs(w - mid) > 2.0 * self.margin * mid else 0
            if self.shift >= 2:
                trace.event(
                    "tuner: the load on the card moved (%s at %.2f ms a token, its median %.2f) - measuring every "
                    "lane afresh",
                    self.labels[self.chosen],
                    1e3 * w,
                    1e3 * mid,
                )
                self._afresh()
                self.meds[self.chosen].append(w)
                self.arm = self._next_arm()
                return None
        m.append(w)
        del m[: -self.keep]
        self.windows += 1
        if self.arm != self.chosen and self.windows > len(self.labels) * self.rounds:
            # an exploring window closed: explore less while the other arms lose clearly, as often as at first when
            # one comes close
            mine, theirs = self.median(self.arm), self.median(self.chosen)
            if mine is not None and theirs is not None and mine > theirs * (1.0 + 2.0 * self.margin):
                self.every, self.ref = min(64, self.every * 2), theirs
            else:
                self.every, self.ref = self.explore, None
        elif self.arm == self.chosen and self.ref is not None and abs(m[-1] - self.ref) > 2.0 * self.margin * self.ref:
            # the chosen arm's own time moved: the load changed, and the others are worth a look again
            self.every, self.ref = self.explore, None
        said = self._choose()
        self.arm = self._next_arm()
        return said

    def report(self) -> str:
        parts = []
        for i, lab in enumerate(self.labels):
            md = self.median(i)
            parts.append(f"{'*' if i == self.chosen else ''}{lab} " + (f"{1e3 * md:.2f}" if md is not None else "-"))
        return f"ms a token by setting, {self.windows} windows: " + ", ".join(parts)


class _CudaMixin(_State):
    def aa(self, parents: Parents | None = None) -> None:
        self.al = {}
        self.am = {}
        self.an = {}
        self.spec_commits = []
        self.ap = parents
        self.aq = True

    def ab(self) -> None:
        self.aq = False
        self.ap = None

    def _drop_spec_state(self) -> None:
        """a verify pass's per-node checkpoints of the recurrent layers' states and its commits let go: `ad` reads them
        after the pass, and the next pass's `aa` replaces them, so after the last pass they were held until the
        engine was - 32 MB a DeltaNet layer on Qwen3.5's widths, past `close`"""
        self.al, self.am, self.an, self.spec_commits = {}, {}, {}, []

    def ac(self, tmpl: Any, i: int, h: torch.Tensor, pe: PassRope, text_pos: Any, cache: Any) -> torch.Tensor:
        T = h.shape[1]
        outs = []
        layer = cache.layers[i]
        lt = self.layer_types[i]
        rope = pe_for(pe, lt)
        assert rope is not None  # a resident layer's pass always carries the rope
        win = layer_window(self.cfg, lt)
        ap: Parents | None = getattr(self, "ap", None)
        parents: Parents = ap if ap is not None else range(-1, T - 1)  # no tree: the chain's own map
        tree = any(parents[j] != j - 1 for j in range(T))
        spec_on = bool(getattr(self, "aq", False))
        ckpts: list[Any] = []
        pre: Any = None
        if lt == LayerKind.LINEAR and tree:
            pre = tuple(x.clone() for x in self._lin(layer))
        base: Any = None
        if lt != LayerKind.LINEAR and (tree or win):
            if getattr(layer, "paged", False):
                base = layer.get_seq_length()
            else:
                base = layer.keys.shape[-2] if getattr(layer, "keys", None) is not None else 0
        depth = [0] * T  # each node's depth: its position once committed is base + depth
        for p in range(T):
            depth[p] = 0 if parents[p] < 0 else depth[parents[p]] + 1
        for p in range(T):
            hp = h[:, p : p + 1]
            pos_p = text_pos[:, p : p + 1]
            pe_p = (rope[0][:, p : p + 1], rope[1][:, p : p + 1])
            mask_p: KeyRows | torch.Tensor | None = None
            branch = tree and parents[p] != p - 1
            dev = h.device
            if lt == LayerKind.LINEAR and branch:
                conv, rec = pre if parents[p] < 0 else ckpts[parents[p]]
                c, r = self._lin(layer)
                c.copy_(conv)
                r.copy_(rec)
            if base is not None:
                rows = node_mask(base, p, parents, win)
                # the node's rows by index, in the order its committed step would read them (a parent's row precedes
                # its children's): its attention then makes the step's own call (`attend_one`), not a masked one
                if rows is None:
                    mask_p = None
                elif dev.type == Device.CUDA or not self.fam.fast:
                    # on the card, and wherever the attention is the family's own module (gpt-oss's sinks take
                    # `KeyRows` on the host too; the host sdpa of the others takes the bool row)
                    mask_p = KeyRows(rows.nonzero()[:, 0].to(dev, non_blocking=True), base + depth[p])
                else:
                    mask_p = rows.view(1, 1, 1, -1).to(dev)
            outs.append(
                tmpl(
                    hp,
                    position_embeddings=pe_p,
                    attention_mask=mask_p,
                    position_ids=pos_p,
                    past_key_values=cache,
                    use_cache=True,
                )
            )
            if lt == LayerKind.LINEAR and spec_on:
                ckpts.append(tuple(x.clone() for x in self._lin(layer)))
        if lt == LayerKind.LINEAR and spec_on:
            self.al[i] = ckpts
            if tree:
                self.am[i] = pre
        return torch.cat(outs, dim=1)

    def _card_attention(
        self, module: Any, query: torch.Tensor, key: Any, value: Any, mask: Any, scaling: float | None
    ) -> torch.Tensor | None:
        """A card layer's attention on btb's one attention (btb_attn_flash.cuh), as its module hands it over
        (`families/attention.py`, the module run by `_run_card_layer`): a prompt's chunk (T > 1) on its prefill form,
        each row over every key before it; a one-row call - a step, or a verify pass's node, its rows named
        (`KeyRows`) - on its decode form, the card graph's, the node's keys walked at the positions its committed step
        will read them from. Each row's bits are its own, so a prompt's rows are the rows its steps make. A paged
        layer's rows (`PagedKV`) through the card's row map; a contiguous layer's where they lie, through the same
        kernels, so the two caches keep one set of bits. Returns [1, T, Hq, D], or None where the call is not one these
        take - a caller's mask, a shape the kernels lack - for the module's own sdpa, which a paged layer's rows never
        reach"""
        from transformers.cache_utils import DynamicSlidingWindowLayer

        from .paged import PagedKV

        ctx = getattr(self, "_attn_ctx", None)
        if int(query.shape[0]) != 1 or query.dtype != torch.bfloat16 or ctx is None or self.fam.own:
            return None
        # loaded here if nothing before loaded them: a family the card graph does not serve asks no other gate, and
        # read unloaded, its first prompt took the module's sdpa and every later one these kernels
        k = self._card_kernels()
        if k is None:
            return None
        _, Hq, T, D = (int(x) for x in query.shape)
        i = int(getattr(module, "layer_idx", -1))
        if (
            f"btb_attn_flash_prefill_d{D}" not in k.fn
            or f"btb_attn_flash_d{D}" not in k.fn
            or not 0 <= i < min(self.L, len(ctx.layers))
            # a layer that lets its rows past the window go holds no position's row where the walk looks for it
            or isinstance(ctx.layers[i], DynamicSlidingWindowLayer)
        ):
            return None
        win = layer_window(self.cfg, self.layer_types[i])
        tbl: torch.Tensor | None = None
        if isinstance(key, PagedKV):
            card = key.layer.pool.card
            assert card is not None  # a PagedKV is a card layer's
            kb, vb = card.view(key.layer.i)
            n0, tbl = key.n0, card.tbl
        else:
            kb, vb = key, value
            if not (
                isinstance(kb, torch.Tensor)
                and isinstance(vb, torch.Tensor)
                and kb.is_cuda
                and kb.dtype == torch.bfloat16
                and vb.dtype == torch.bfloat16
            ):
                return None
            if kb.stride(-1) != 1 or kb.stride() != vb.stride():
                # rows laid out apart (a fused projection's slices): the kernels read K and V at one pair of strides,
                # so both as one layout - the same values, never the module's sdpa and its bits
                kb, vb = kb.contiguous(), vb.contiguous()
            n0 = int(kb.shape[-2]) - T
            if n0 < 0:
                return None
        Hk = int(kb.shape[1])
        if Hq % Hk or (T > 1 and Hq // Hk > k.flash_prefill_rows(D)):
            # a prompt's block of the prefill form holds a token's heads whole (`_card_family_ok` asks the same)
            return None
        hs, rs = int(kb.stride(1)), int(kb.stride(2))
        scale = float(scaling if scaling is not None else D**-0.5)
        P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
        q = query[0].transpose(0, 1).contiguous()  # [T, Hq, D]
        out = torch.empty(T, Hq, D, dtype=torch.bfloat16, device=query.device)
        if T > 1:
            if isinstance(mask, KeyRows):
                return None
            # the prefill form: 64 rows a block (32 at the widest head), the rows' states in a float32 buffer of the
            # chunk's rows, its block's shared memory dynamic
            who = "the card's attention: a prompt chunk's row states"
            run = self.scratch.take("card attention rows", (T * Hq * D,), torch.float32, query.device, who)
            tpb = k.flash_prefill_rows(D) // (Hq // Hk)
            k.launch(
                k.flash_prefill_kernel(D),
                ((T + tpb - 1) // tpb, Hk, 1),
                (128, 1, 1),
                [P(q), P(kb), P(vb), P(out), ci(n0), ci(T), ci(Hq), ci(Hk), ci(hs), ci(rs), cf(scale), ci(win)]
                + [P(tbl), P(run)],
                shared=k.flash_prefill_smem(D),
            )
            return out[None]
        rows: torch.Tensor | None = None
        first = 0
        if isinstance(mask, KeyRows):
            # the node's keys in the order its step reads them, at the positions the step reads them from: the walk's
            # position j (first <= j <= pos) at row rows[j - first], the map's pointer set back by `first` entries
            idx = mask.idx
            pos = int(mask.pos) if mask.pos is not None else int(idx.numel()) - 1
            first = pos + 1 - int(idx.numel())
            rows = (tbl[idx] if tbl is not None else idx).to(torch.int32)
            n0 = pos
        walk = self.scratch.take("card attention walk", (2,), torch.int32, query.device, "the card's attention")
        walk[0:1].fill_(n0)
        walk[1:2].fill_(-1)
        GK = self._attn_group(D)
        S = (n0 + 1 + GK - 1) // GK
        who = "the card's attention: its groups' states"
        part_m = self.scratch.take("card attention m", (S * Hq,), torch.float32, query.device, who)
        part_l = self.scratch.take("card attention l", (S * Hq,), torch.float32, query.device, who)
        part_acc = self.scratch.take("card attention acc", (S * Hq * D,), torch.float32, query.device, who)
        cnt = self.scratch.take("card attention cnt", (Hq,), torch.int32, query.device, who)
        cnt.zero_()  # the last group of each row folds and leaves its count at 0; a buffer made new holds anything
        if rows is not None:
            via = ctypes.c_void_p(int(rows.data_ptr()) - 4 * first)
        else:
            via = P(tbl)
        k.launch(
            f"btb_attn_flash_d{D}",
            ((Hq // Hk + 7) // 8, S, Hk),
            (self._attn_warps(D) * 32, 1, 1),
            [
                P(q),
                P(kb),
                P(vb),
                P(out),
                P(walk[0:1]),
                P(walk[1:2]),
                ci(1),
                ci(Hq),
                ci(Hk),
                ci(hs),
                ci(rs),
                cf(scale),
                P(part_m),
                P(part_l),
                P(part_acc),
                P(cnt),
                ci(win),
                via,
            ],
        )
        return out[None]

    def _fast_ok(
        self, cache: Any, B: int, T: int, past: int, am: torch.Tensor | None, on_layer: Any, stop_after: int | None
    ) -> bool:
        return (
            T == 1
            and B == 1
            and past > 0
            and am is None
            and on_layer is None
            and stop_after is None
            and cache is not None
            and getattr(self, "kv_host", False)
            and self.dev.type == Device.CUDA
            and getattr(self, "fast_decode", True)
            and Native.attn_decode is not None
            and not self.shadow
            and LayerKind.LINEAR not in self.layer_types
            and len(self.resident) == self.L
            and getattr(self, "_probe", None) is None
            and (self.fam.dense or self.fam.sandwich)
        )

    def _rope_by_type(self, pe: PassRope) -> dict[str, Rope]:
        """the pass's rope as one (cos, sin) per layer type: a dual-rope family's own, the others' one for all"""
        return pe if isinstance(pe, dict) else dict.fromkeys(set(self.layer_types), pe)

    def _rope_fn(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        """the one rope the engine applies where it rotates q and k itself - a one-row step's graph, a kv_host pass,
        the card's own attention - so a step and a verify pass over the same rows rotate them alike: the fused one
        (BTB_FUSED_ROPE, on by default) for a family whose rope is the reference's full or partial rotary, else
        the module's. The fused one is not the module's bit for bit in bf16 (addcmul rounds once where the
        reference rounds twice): two paths reading different ropes parted a kv_host verify from its steps"""
        if self._frope and (self.fam.dense or self.fam.sandwich):
            return _fused_rope
        return cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], self.fam.mod.apply_rotary_pos_emb)

    def _seg_a(self, g: dict[str, Any], i: int) -> None:
        apply_rotary_pos_emb = self._rope_fn()
        tmpl = self.resident[i]
        at = tmpl.self_attn
        hd = at.head_dim
        x = tmpl.input_layernorm(g["h"])
        if hasattr(at, "qkv_proj"):  # q, k and v as one projection (Phi-3's layout)
            qkv = at.qkv_proj(x)
            nq = self.cfg.num_attention_heads * hd
            nk = at.num_key_value_heads * hd
            q = qkv[..., :nq].view(1, 1, -1, hd)
            k = qkv[..., nq : nq + nk].view(1, 1, -1, hd)
            v = qkv[..., nq + nk :].view(1, 1, -1, hd)
        elif self.fam.attn_gate:
            qg = at.q_proj(x).view(1, 1, -1, hd * 2)
            q, gate = torch.chunk(qg, 2, dim=-1)
            g["gate"].copy_(torch.sigmoid(gate.reshape(1, 1, -1)))
            q = at.q_norm(q.reshape(1, 1, -1, hd))
            k = at.k_norm(at.k_proj(x).view(1, 1, -1, hd))
            v = at.v_proj(x).view(1, 1, -1, hd)
        else:
            q = at.q_norm(at.q_proj(x).view(1, 1, -1, hd))
            k = at.k_norm(at.k_proj(x).view(1, 1, -1, hd))
            v = at.v_proj(x).view(1, 1, -1, hd)
        cos, sin = g["rope"][self.layer_types[i]]
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        hk = k.shape[1]
        g["q_pin"].copy_(q[0, :, 0].float(), non_blocking=True)
        g["kv_pin"][:hk].copy_(k[0, :, 0], non_blocking=True)
        g["kv_pin"][hk:].copy_(v[0, 0], non_blocking=True)

    def _seg_b(self, g: dict[str, Any], i: int) -> None:
        tmpl = self.resident[i]
        at = tmpl.self_attn
        g["a_dev"].copy_(g["out_pin"], non_blocking=True)
        a1 = g["a_dev"].to(g["h"].dtype).view(1, 1, -1)
        if g["gate"] is not None:
            a1 = a1 * g["gate"]
        if self.fam.sandwich:
            # each sub-block's output normed before its add, the MLP's own input norm of the sum between
            h = g["h"] + tmpl.post_attention_layernorm(at.o_proj(a1))
            h = h + tmpl.post_feedforward_layernorm(tmpl.mlp(tmpl.pre_feedforward_layernorm(h)))
            g["h"].copy_(h)
        elif self._fmlp:
            # both residual adds folded into the persistent buffer: h = (h + attn) + mlp(norm(h + attn)) bit for bit,
            # three fewer nodes a layer
            g["h"].add_(at.o_proj(a1))
            g["h"].add_(tmpl.mlp(tmpl.post_attention_layernorm(g["h"])))
        else:
            h = g["h"] + at.o_proj(a1)
            h = h + tmpl.mlp(tmpl.post_attention_layernorm(h))
            g["h"].copy_(h)

    def _graphs(self, dt: torch.dtype, pe: Any) -> dict[str, Any]:
        g = getattr(self, "_g", None)
        if g is not None:
            return g
        c = self.cfg
        hq = int(c.num_attention_heads)
        hk = int(getattr(c, "num_key_value_heads", None) or hq)
        d = int(self.resident[0].self_attn.head_dim)
        rope = self._rope_by_type(pe)
        rd = int(next(iter(rope.values()))[0].shape[-1])
        g = {
            "h": torch.zeros(1, 1, int(c.hidden_size), dtype=dt, device=self.dev),
            # the step's rope row per layer type: the graphs read it in place, the step copies it in
            "rope": {
                lt: (
                    torch.zeros(1, 1, rd, dtype=p[0].dtype, device=self.dev),
                    torch.zeros(1, 1, rd, dtype=p[1].dtype, device=self.dev),
                )
                for lt, p in rope.items()
            },
            "q_pin": torch.zeros(hq, d, dtype=torch.float32, pin_memory=True),
            "kv_pin": torch.zeros(2 * hk, d, dtype=dt, pin_memory=True),
            "out_pin": torch.zeros(hq, d, dtype=torch.float32, pin_memory=True),
            "a_dev": torch.zeros(hq, d, dtype=torch.float32, device=self.dev),
            "gate": torch.zeros(1, 1, hq * d, dtype=dt, device=self.dev) if self.fam.attn_gate else None,
            "hk": hk,
            "d": d,
            "pool": torch.cuda.graph_pool_handle(),
            "A": {},
            "B": {},
        }
        s = torch.cuda.Stream(device=self.dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(self.L):
                for _ in range(2):
                    self._seg_a(g, i)
                    self._seg_b(g, i)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        for i in range(self.L):
            ga = torch.cuda.CUDAGraph()
            with torch.cuda.graph(ga, pool=g["pool"]):
                self._seg_a(g, i)
            gb = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gb, pool=g["pool"]):
                self._seg_b(g, i)
            g["A"][i] = ga
            g["B"][i] = gb
        self._g = g
        self.log(f"[fast] {self.L} layers captured as {2 * self.L} CUDA graphs for single-token decode")
        return g

    # the card graph: a run of consecutive resident attention layers as one graph per (run, T), T rows up to
    # CARD_T_MAX (a step, or a verify pass over a chain or a tree); the cache's length and the rows' positions
    # are read from device memory, so one graph serves every length. Every kernel's reduction order is fixed
    # by index, so a verify pass reproduces the one-token steps of the same path bit for bit
    _cg: Any
    CARD_T_MAX = 32

    @staticmethod
    def _attn_group(D: int) -> int:
        """the keys of a group of the one attention (btb_attn_flash.cuh): GT tiles of BN, a decode block's"""
        return 8 * (32 if D >= 256 else 64)

    @staticmethod
    def _attn_warps(D: int) -> int:
        """the warps of a block of the one attention's decode form (FaPick::WPB): the tile states it folds sit in
        shared memory, four at the widest head"""
        return 4 if D >= 256 else 8

    PROGRAM_SAMPLES = 5  # the timed replays of each width a card program's warm-up keeps the fastest of

    def _card_kernels(self) -> Any:
        """the card's kernels, else None once `_card_warning` has said why the card runs through torch alone"""
        k = Native.cuda
        if k is None:
            k = Native.card_kernels()
            if k is None:
                _card_warning(str(Native.cuda_reason))
                return None
            cs = self.scheduler.caches() if getattr(self, "scheduler", None) is not None else {}
            self.log(
                f"[card] kernels {os.path.basename(str(kernels_path()))}; "
                + ", ".join(f"{n} {v / 2**20:.0f} MB" for n, v in cs.items() if v)
            )
        return k

    def _card_family_ok(self) -> bool:
        ok = getattr(self, "_card_family", None)
        if ok is None:
            # the family first: another family's config need not carry the fields the shapes are read from
            # (a mixture of experts names its width moe_intermediate_size)
            ok = (
                self.fam.card_graph
                and act_name(self.cfg) in ("silu", "swish", "gelu_pytorch_tanh")  # the kernels' activations
                and self.mlx is None
                and not getattr(self, "resident_fp32", False)
                and self.compute_dtype in (None, torch.bfloat16)
            )
            if ok:
                H, Hq, Hk, D, I = self._card_dims()
                # the kernels' shapes: a lane holds D/32 dims of a head, the rows load 16 bytes at a time, and a
                # prompt's block of the one attention holds a token's heads whole
                ok = D in (64, 128, 256) and H % 8 == 0 and I % 8 == 0 and (Hq * D) % 8 == 0 and Hq % Hk == 0
                ok = ok and Hq // Hk <= _Cuda.flash_prefill_rows(D)
            self._card_family = ok
        return bool(ok)

    def _card_ready(self) -> bool:
        """the engine can run the card graph at all: the card, the cache on it, a family the kernels know"""
        return (
            self.dev.type == Device.CUDA
            and not getattr(self, "kv_host", False)
            and getattr(self, "card_graphs", True)
            and not self.shadow
            and getattr(self, "_probe", None) is None
            and self._card_family_ok()
            and self._card_kernels() is not None
            # a build the card had no room for, at this placement and not long ago (`_card_oom`)
            and not self._card_off_now()
        )

    # a card graph a build found no room for is tried again this long after, doubling with each refusal at one
    # placement up to CARD_RETRY_MAX_S: the program that took the card may have gone, with no placement move to say so
    CARD_RETRY_S = 10.0
    CARD_RETRY_MAX_S = 300.0

    def _card_off_now(self) -> bool:
        """whether the card graph is off for a build the card had no room for (`_card_oom`): at the placement it
        failed at, until its retry is due"""
        off = getattr(self, "_card_off", None)
        if off is None:
            return False
        ver, until, _wait = off
        return ver == self.device.snapshot().version and time.monotonic() < until

    def _card_pass_ok(
        self, cache: Any, B: int, T: int, past: int, am: torch.Tensor | None, on_layer: Any, stop_after: int | None
    ) -> bool:
        return (
            B == 1
            and 1 <= T <= self.CARD_T_MAX
            and past > 0
            and am is None
            and on_layer is None
            and stop_after is None
            and cache is not None
            and not forked(cache)
            and self._card_ready()
        )

    def _card_segments(self) -> list[tuple[int, int]]:
        segs: list[tuple[int, int]] = []
        a: int | None = None
        for i in range(self.L):
            lt = self.layer_types[i]
            # a sliding layer joins a run where its window rides the attention kernel (a sandwich family's)
            ok = i in self.resident and (lt == LayerKind.FULL or (lt == LayerKind.SLIDING and self.fam.sandwich))
            if ok and a is None:
                a = i
            if not ok and a is not None:
                segs.append((a, i))
                a = None
        if a is not None:
            segs.append((a, self.L))
        return segs

    def _graphs_close(self) -> None:
        """The captured graphs let go, and every buffer they replay into: the single-token decode's (`_g`), the
        card path's (`_cg`: its graphs, arena and tables) and a family's card program's (`_cp`). A graph keeps its
        pool's memory until it is reset, so each is reset before the state goes; and the arena's persisting L2
        window is cleared on the streams it was set on - left, it names freed memory and keeps the card's L2 set
        aside for it."""
        cp = vars(self).pop("_cp", None)
        if cp is not None:
            cp.close()
        g, cg = getattr(self, "_g", None), getattr(self, "_cg", None)
        if g is None and cg is None:
            return
        torch.cuda.synchronize(self.dev)  # nothing still replaying into what goes
        for st in (g, cg):
            if st is not None:
                for graph in _graphs_in(st):
                    graph.reset()
        if cg is not None and cg.get("arena") is not None and cg.get("k") is not None:
            cg["k"].persist(0, 0, stream=cg["stream"])
            cg["k"].persist(0, 0)
        if cg is not None and cg.get("arena") is not None:
            # the arena's hold on its memory: what no cache still views goes back to the card now
            cg["arena"]["A"].close()
        vars(self).pop("_g", None)
        vars(self).pop("_cg", None)

    def _card_let_go(self) -> None:
        """The card graphs reset and the layer blocks they read let go (`_card_weights`' merged q/k/v and gate/up,
        which the modules only view), keeping the arena and the rope tables the cache and the next state carry over;
        and the kv_host step's graphs (`_g`), which read the modules' weights by pointer. A layer shed while these
        hold its blocks frees nothing on the card - a yield once shed every layer for nothing - and a graph kept past
        its layer's regrowth would replay freed memory. A family's card program (`_cp`) the same: its merged blocks
        and graphs (`let_go`), which held every resident layer's weights through Qwen4's sheds. The next pass builds
        what it needs again"""
        st = getattr(self, "_cg", None)
        g = vars(self).pop("_g", None)
        cp = getattr(self, "_cp", None)
        if st is None and g is None and cp is None:
            return
        torch.cuda.synchronize(self.dev)  # nothing still replaying into what goes
        before = self._trace_mem()
        if st is not None:
            for graph in _graphs_in(st["graphs"]):
                graph.reset()
            st["graphs"].clear()
            st["layers"].clear()
            # the step graph's embedding table too - a granted copy, or a tied head's weight a shed would free
            # otherwise for nothing: made again by the next step graph (`_card_table`)
            st.pop("table", None)
        if g is not None:
            for graph in _graphs_in(g):
                graph.reset()
        if cp is not None:
            cp.let_go()
        torch.cuda.empty_cache()
        if trace.ON:  # no log line says this: the trace does
            trace.event("card graph: let go, the next pass builds it again%s", self._trace_grew(before))

    @staticmethod
    def _is_card_oom(e: BaseException) -> bool:
        """whether `e` is the card out of memory: torch's allocator's error, the driver's from a graph's
        instantiation or launch (raised as another RuntimeError, an AcceleratorError), or the scheduler's refusal of
        a pass's buffers (`MemoryGrantError`) - never a pass no other path may run again (`CardPassFailed`)"""
        if isinstance(e, CardPassFailed):
            return False
        return isinstance(e, (torch.OutOfMemoryError, MemoryGrantError)) or (
            isinstance(e, RuntimeError) and "out of memory" in str(e).lower()
        )

    def _card_oom(self, e: BaseException) -> None:
        """a card-graph build the card had no room for (another program took it mid-pass): the graphs let go, the
        card graph off - the pass runs the torch path - until the placement moves or its retry is due (CARD_RETRY_S,
        doubling at one placement): a program that took the card and left moved no placement"""
        self._card_let_go()
        ver = self.device.snapshot().version
        off = getattr(self, "_card_off", None)
        wait = min(self.CARD_RETRY_MAX_S, off[2] * 2) if off is not None and off[0] == ver else self.CARD_RETRY_S
        self._card_off = (ver, time.monotonic() + wait, wait)
        self.log(
            f"[card] no room on the card for the card graph ({str(e).splitlines()[0][:120]}): the torch path until "
            f"the placement moves (version {ver}), or {wait:.0f} s"
        )

    @staticmethod
    def _capture(cg: torch.cuda.CUDAGraph, pool: Any, stream: torch.cuda.Stream, body: Callable[[], Any]) -> None:
        """`body` captured into `cg` on `stream`: the stream current before restored however the capture ends - a
        capture whose end raised (the card out of memory at the graph's instantiation) skipped torch's own restore,
        and every later kernel of the thread queued on the capture's stream"""
        s0 = torch.cuda.current_stream()
        try:
            with torch.cuda.graph(cg, pool=pool, stream=stream):
                body()
        except BaseException:
            torch.cuda.set_stream(s0)
            raise

    def _card_state(self) -> dict[str, Any]:
        ver = self.device.snapshot().version
        st = getattr(self, "_cg", None)
        if st is not None and st["version"] == ver:
            return st
        k = self._card_kernels()
        st = {
            "version": ver,
            "k": k,
            "segments": self._card_segments(),
            "layers": {},
            "graphs": {},
            "pool": torch.cuda.graph_pool_handle() if st is None else st["pool"],
            "stream": torch.cuda.Stream(device=self.dev) if st is None else st["stream"],
            "arena": None if st is None else st["arena"],
            "tables": None if st is None else st["tables"],
            # the passes' switches, read on the card by the kernels they steer, so the host moves them between
            # replays without a capture (`SW_*`, `_Tuner`): every setting gives the same bits
            "switch": self._card_switch_init() if st is None else st["switch"],
            "tuners": {} if st is None else st.get("tuners", {}),
        }
        self._cg = st
        return st

    def _card_runs_layer(self, i: int) -> bool:
        """whether the card graph runs layer `i`: it lies in one of the graph's runs of resident layers"""
        return any(a <= i < b for a, b in self._card_state()["segments"])

    def _card_segment_at(self, i: int, n_layers: int) -> tuple[int, int] | None:
        for a, b in self._card_state()["segments"]:
            if a == i:
                return (a, min(b, n_layers)) if min(b, n_layers) > a else None
        return None

    def _card_dims(self) -> tuple[int, int, int, int, int]:
        c = self.cfg
        H = int(c.hidden_size)
        Hq = int(c.num_attention_heads)
        Hk = int(getattr(c, "num_key_value_heads", None) or Hq)
        D = int(getattr(c, "head_dim", None) or H // Hq)
        I = int(getattr(c, "intermediate_size", None) or getattr(c, "moe_intermediate_size", None) or 4 * H)
        return H, Hq, Hk, D, I

    def _card_weights(self, st: dict[str, Any], i: int) -> dict[str, Any]:
        """layer i's weights as the kernels take them: q/k/v and gate/up merged into one row block each (the
        modules keep views of the same memory, so the generic path is unchanged), the norms, the scale"""
        L = st["layers"].get(i)
        if L is not None:
            return L
        tmpl = self.resident[i]
        at, mlp = tmpl.self_attn, tmpl.mlp
        H, Hq, Hk, D, I = self._card_dims()
        nq, nk = Hq * D, Hk * D
        W = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
        self._set_param(at.q_proj, "weight", W[:nq])
        self._set_param(at.k_proj, "weight", W[nq : nq + nk])
        self._set_param(at.v_proj, "weight", W[nq + nk :])
        G = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
        self._set_param(mlp.gate_proj, "weight", G[:I])
        self._set_param(mlp.up_proj, "weight", G[I:])
        for t in (W, G, at.o_proj.weight, mlp.down_proj.weight):
            if t.dtype != torch.bfloat16 or t.device.type != "cuda":
                raise RuntimeError(
                    f"[card] layer {i}: the card graph takes bf16 weights on the card, got {t.dtype} {t.device}"
                )
        L = {
            "qkv": W,
            "gu": G,
            "o": at.o_proj.weight,
            "down": mlp.down_proj.weight,
            "ln1": tmpl.input_layernorm.weight,
            "ln2": tmpl.post_attention_layernorm.weight,
            "wq": at.q_norm.weight,
            "wk": at.k_norm.weight,
            "scale": float(at.scaling),
            "eps": float(self.cfg.rms_norm_eps),
            "win": layer_window(self.cfg, self.layer_types[i]),
        }
        if self.fam.sandwich:
            L["pre_ff"] = tmpl.pre_feedforward_layernorm.weight
            L["post_ff"] = tmpl.post_feedforward_layernorm.weight
        st["layers"][i] = L
        return L

    def _card_tables(self, st: dict[str, Any], cap: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """the rope's cos/sin over `cap` positions on the card in bf16, by layer type: a dual-rope family's own
        per type (Gemma 3's local/global), the others' one pair under every type"""
        tb = st["tables"]
        if tb is not None and next(iter(tb.values()))[0].shape[0] >= cap:
            return tb
        H = int(self.cfg.hidden_size)
        x = torch.zeros(1, 1, H, dtype=torch.bfloat16, device=self.dev)
        pos = torch.arange(cap, device=self.dev)[None]
        types = sorted(set(self.layer_types))
        pairs = [self.rotary(x, pos, lt) for lt in types] if self.fam.dual_rope else [self.rotary(x, pos)] * len(types)
        tb = {
            lt: (cos[0].to(torch.bfloat16).contiguous(), sin[0].to(torch.bfloat16).contiguous())
            for lt, (cos, sin) in zip(types, pairs, strict=True)
        }
        st["tables"] = tb
        return tb

    def _card_arena(self, st: dict[str, Any], need: int, reach: int = 0) -> dict[str, Any]:
        """The arena holding every run layer's cache rows (`KvArena`, position-major: a layer's rows grow at the end
        of its own region) the caches attach to - a cache attached elsewhere copies its rows in - at least `need` rows
        deep, and `reach` deep when it grows (`_card_reach`: a growth step past the rows). A growth maps memory onto
        the rows' end where the card's driver can (nothing moves), else regrows the layers one at a time; only a
        change of the layers on the card makes a new arena, the common layers' rows copied over. Its front lies in a
        persisting-L2 window, so a short context's attention reads hit L2 under the weights' evict-first loads"""
        ar = st["arena"]
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        slot = {i: j for j, i in enumerate(layers)}
        same = ar is not None and ar["slot"] == slot
        # deep enough, and over the layers the card runs now: an arena made before a layer was regrown onto the card
        # has no region for it, and its graphs read past the slots it has
        if ar is not None and ar["cap"] >= need and same:
            return ar
        H, Hq, Hk, D, I = self._card_dims()
        store = ar["A"] if same else KvArena(len(layers), Hk, D, self.dev, self._card_ceiling())
        cap = store.rows_for((max(int(need), int(reach), 4096) + 1023) // 1024 * 1024)
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            sched.grant(
                store.nbytes(cap),
                "kv",
                requester=f"card arena {len(layers)} layers x {cap} rows",
                B=1,
                cap=cap,
                bound=None,
                device=self.dev,
                # what the arena holds now: grown in place it stays (only the growth is new), else it is let go
                # once its rows are copied over
                held=ar["A"].nbytes(ar["cap"]) if ar is not None else 0,
            )
        owner = ar["owner"]() if ar is not None and ar["owner"] is not None else None

        def moved(j: int) -> None:
            """layer layers[j]'s holder takes the arena's views of it (the rows are in them already)"""
            i = layers[j]
            if owner is None:
                return
            if isinstance(owner.layers[i], CardRowsLayer):
                # a fork's or a batch's rows keep their slots: only the slices move
                if owner.layers[i]._buf is not None:
                    owner.layers[i]._buf = (store[j, 0], store[j, 1])
                return
            layer = owner.layers[i]
            n = int(layer.keys.shape[-2]) if (layer.is_initialized and layer.keys is not None) else 0
            layer._buf = (store[j, 0][None], store[j, 1][None])
            if n:
                layer._set_rows(layer._buf[0][..., :n, :], layer._buf[1][..., :n, :])

        if same:
            store.grow(cap, moved=moved)
            if store.in_place:
                for j in range(len(layers)):
                    moved(j)  # the same addresses, longer views
        else:
            store.grow(cap)
            if ar is not None:
                for i, j in ar["slot"].items():
                    if i in slot:
                        for w in (0, 1):
                            store[slot[i], w][:, : ar["cap"]].copy_(ar["A"][j, w])
                        moved(slot[i])
                ar["A"].close()
        new = {"A": store, "cap": store.cap, "slot": slot, "owner": ar["owner"] if ar is not None else None}
        st["arena"] = new
        st["graphs"].clear()
        # how much of the arena's front sits in persisting L2 is the scheduler's call (it holds the
        # hierarchy's sizes); the driver call is only the mechanism
        k = st["k"]
        pinned = self.scheduler.pin_bytes(store.nbytes(store.cap)) if sched is not None else 0
        k.persist(store.front(), pinned, stream=st["stream"])
        k.persist(store.front(), pinned)
        self.log(
            f"[card] arena {len(layers)} layers x {store.cap} rows ({store.nbytes(store.cap) / 2**20:.0f} MB, "
            f"{'mapped in place' if store.in_place else 'regrown layer by layer'}), {pinned / 2**20:.0f} MB of it "
            f"pinned in L2"
        )
        return new

    def _card_ceiling(self) -> int:
        """the rows a sequence of this model reaches: its context where one is set, else its own position limit -
        the addresses an arena reserves up front where the driver maps in place (none of it memory until used)"""
        return max(int(self.context or 0), int(getattr(self.cfg, "max_position_embeddings", 0) or 0), 4096)

    @staticmethod
    def _card_reach(cache: Any, layers: Iterable[int], rows: int) -> int:
        """the rows the arena takes for `cache` holding `rows`: a growth step past them (an eighth, at least 4096),
        held to the most its layers reach where the caller named it (`cap_hint`) - a ceiling, never a reservation:
        an answer left uncapped names the whole window, and an arena that deep was asked of the card up front"""
        step = rows + max(4096, rows // 8)
        hint = max((int(getattr(cache.layers[i], "cap_hint", 0) or 0) for i in layers), default=0)
        return min(step, max(rows, hint)) if hint else step

    def _card_adopt_cache(self, cache: Any) -> None:
        """a new cache takes the arena when no live cache holds it and it reaches as far as the cache will, so its
        prefill writes straight in. An arena the cache would grow is left to the prefill's sweep (`_card_arena_take`),
        which grows it once the room is made: grown here, before any, the room it took could only be looked for by
        shedding layers - which free nothing of the one arena - and the whole model went to the host for it"""
        st = getattr(self, "_cg", None)
        if st is None or st["arena"] is None or st["version"] != self.device.snapshot().version:
            return
        if getattr(cache, "paged", False):
            return  # its rows are the prefix cache's region's, never the arena's
        ar = st["arena"]
        owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not None and owner is not cache:
            return
        if self._card_reach(cache, ar["slot"], 0) > int(ar["cap"]):
            return
        self._card_arena_holds(cache, 0)

    def _card_arena_holds(self, cache: Any, T: int) -> bool:
        """whether `cache` sits in the card graph's arena for a pass of `T` more rows - bound now where it is not,
        the arena grown to the cache's reach as the ledger grants it. Refused, the cache keeps its rows in buffers of
        its own (a long prompt's, presized before the arena could take them, would need a second whole copy there)
        and its passes take the torch layers over them: the arena is the graphs' speed, never a pass's condition.
        A refused cache is not asked again - its rows only grow. A paged cache's rows are the prefix cache's card
        region's whichever layers read them: bound there for the pass, or the pass's own appends meet the refusal"""
        paged = bool(getattr(cache, "paged", False))
        refused = self.__dict__.setdefault("_arena_refused", weakref.WeakSet())
        if not paged and cache in refused:
            return False
        st = self._card_state()
        if not st["segments"]:
            # no layer the card graph runs is on the card (another program took it and every one was shed to the
            # host): no run's rows for an arena to hold, and the passes take the host's layers. An arena over no
            # layers has no regions, and its front was read past an empty list
            return False
        try:
            self._card_bind(cache, st, T)
            return True
        except MemoryGrantError as e:
            if paged:
                self.log(f"[card] the card's region cannot take this conversation's next {T} rows ({e})")
                return False
            refused.add(cache)
            self.log(f"[card] the arena cannot take this cache's rows ({e}); its passes take the torch layers")
            return False

    def _card_arena_take(self, cache: Any, T: int) -> bool:
        """a prefill's cache into the card graph's arena before its rows are made, where no other live cache holds
        the arena: grown to the sequence's reach, the rows are made once, in it - presized in buffers of the cache's
        own, the first decode step asked a second whole copy to move them in (3.3 GB of a 40k prompt, refused)"""
        st = self._card_state()
        ar = st["arena"]
        owner = ar["owner"]() if ar is not None and ar["owner"] is not None else None
        if owner is not None and owner is not cache:
            return False
        return self._card_arena_holds(cache, T)

    def _card_arena_has(self, cache: Any, n: int) -> bool:
        """whether `cache`'s first `n` rows have their places on the card already, nothing bound to ask it: a paged
        cache's in the card's region (its rows reserved before the pass), a contiguous one's in the arena it holds,
        every run layer's slot in it and its rows reaching `n`. A layer-by-layer sweep's layers ask it - binding
        there would take the arena from the cache that holds it, at the length the cache has, short of the prompt"""
        if getattr(cache, "paged", False):
            return True
        st = self._card_state()
        ar = st["arena"]
        if ar is None or ar["owner"] is None or ar["owner"]() is not cache or int(ar["cap"]) < n:
            return False
        return all(i in ar["slot"] for a, b in st["segments"] for i in range(a, b))

    def _card_bind(self, cache: Any, st: dict[str, Any], T: int) -> dict[str, Any]:
        import weakref

        if getattr(cache, "paged", False):
            return self._card_bind_paged(cache, st, T)
        ar = st["arena"]
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        if ar is not None:
            owner = ar["owner"]() if ar["owner"] is not None else None
            same = owner is not None and owner is cache
            if cache is not None and same and len(ar["slot"]) == len(layers) and all(i in ar["slot"] for i in layers):
                # the common step: this cache holds the arena and every run layer sits at its front
                n = cache.get_seq_length()
                if n + T <= ar["cap"] and all(cache.layers[i]._an is not None for i in ar["slot"]):
                    return ar
        need = 0
        for i in layers:
            layer = cache.layers[i]
            n = layer.get_seq_length() if layer.is_initialized else 0
            need = max(need, n + T)
        ar = self._card_arena(st, need, self._card_reach(cache, layers, need))
        owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not cache:
            self._card_evict(ar)
            ar["owner"] = weakref.ref(cache)
        A = ar["A"]
        for i, j in ar["slot"].items():
            layer = cache.layers[i]
            if not isinstance(layer, GrowLayer):
                raise RuntimeError(f"[card] cache layer {i} is a {type(layer).__name__}, not the engine's GrowLayer")
            kb, vb = A[j, 0][None], A[j, 1][None]
            if (
                layer._buf is None
                or layer._buf[0].untyped_storage().data_ptr() != kb.untyped_storage().data_ptr()
                or layer._buf[0].data_ptr() != kb.data_ptr()
            ):
                layer.attach(kb, vb)
        return ar

    def _card_bind_paged(self, cache: Any, st: dict[str, Any], T: int) -> dict[str, Any]:
        """A paged cache on the card for a pass of `T` rows: its table's rows reserved, its pages in the card's region
        (another conversation's parked first), its row map uploaded (`PagedCache.bind`). Every conversation's graphs
        are the same ones - the arenas and the map's buffer are the region's, the map in it the bound table's - made
        again only when the region's layout moves (its arenas grown past their addresses, the map's buffer longer)"""
        from .kvpool import PAGE

        card = cache.prefix.pool.card
        if card is None:
            raise RuntimeError("[card] a paged cache over a pool with no card region")
        cache.bind(T)
        pg = st.get("pg")
        if pg is None or pg["card"] is not card or pg["layout"] != card.layout:
            for key in [key for key, g in st["graphs"].items() if g.get("paged")]:
                del st["graphs"][key]
            pg = st["pg"] = {"card": card, "layout": card.layout, "cap": max(card.cap * PAGE, PAGE)}
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        off = [i for i in layers if i not in card.arenas]
        if off:
            raise RuntimeError(f"[card] layers {off} run on the card with their rows off its region")
        return pg

    @staticmethod
    def _card_m(T: int) -> int:
        for m in (1, 2, 4, 8, 16, 32):
            if T <= m:
                return m
        raise ValueError(T)

    def _card_mma_avail(self) -> bool:
        k = Native.cuda
        return k is not None and "btb_gemv_mma_bf16" in k.fn

    def _card_mma_for(self, T: int) -> bool:
        """Which GEMV a T-row pass runs: the fp32 chain (`gemv_rows`) or the tensor-core kernel
        (`btb_gemv_mma.cuh`); `card_mma` / BTB_CARD_MMA force it, else `card_warm`'s measurement. One kernel
        for every width: the two sum a row in different orders, and a step on one with a pass on the other
        would part at a bf16 near-tie. The warm-up's choice is the engine's (`_mma_for`, `_mma_one`): the card's and
        the model's, not the placement's, so no card state carries it and none left behind holds it."""
        on, why = self._card_mma_pick(T)
        if trace.ON:  # a pass's path: nothing built for the trace when it is off
            trace.changed(
                (trace.token(self), "card gemv", T),
                on,
                "card gemv: %d-row passes take the %s kernel (%s)",
                T,
                "tensor-core" if on else "fp32-chain",
                why,
            )
        return on

    def _card_mma_pick(self, T: int) -> tuple[bool, str]:
        """`_card_mma_for`'s answer and what gave it"""
        if not self._card_mma_avail():
            return False, "no tensor-core kernel in this build"
        on = getattr(self, "card_mma", None)
        if on is None:
            env = os.environ.get("BTB_CARD_MMA")
            if env in ("0", "1"):
                return env == "1", "BTB_CARD_MMA"
        if on is not None:
            return bool(on), "card_mma"
        choice: dict[int, bool] | None = vars(self).get("_mma_for")
        if choice is not None and T in choice:
            return bool(choice[T]), "the warm-up measured it at this width"
        one = vars(self).get("_mma_one")
        if one is not None:
            # a width past the timed ones (a long prompt's prefill) takes the engine's kernel
            return bool(one), "past the widths the warm-up timed: the engine's kernel"
        return False, "not measured yet"

    @staticmethod
    def _card_mma_grid(R: int, C: int) -> int:
        """the launch of btb_gemv_mma_bf16 for a [R, C] weight: one block per group of 16 rows (a shorter grid
        strides the groups and is only slower, never wrong)"""
        return (R + 15) // 16

    # the row groups at which two warps a group keep the card's DRAM streaming (`_card_mma_warps`)
    MMA_WIDE_GROUPS = 512

    @classmethod
    def _card_mma_warps(cls, R: int, C: int) -> int:
        """the warps over one group of 16 weight rows in btb_gemv_mma_bf16, k split between them: two where the
        groups fill the card, four on a weight too short to (Qwen3-0.6B's 1024-row o and down, 64 groups over 60
        SMs, streamed at 350 GB/s against its head's 441). A function of the weight's shape alone - never of the
        pass's rows - so a shape's one-row step and its verify pass sum alike, bit for bit"""
        return 2 if (R + 15) // 16 >= cls.MMA_WIDE_GROUPS else 4

    def _card_mma_narrow(self, R: int) -> bool:
        """a [R, C] weight's tensor-core matvec at 8 rows a block (`btb_gemv_mma8_bf16`) rather than 16: where 16
        would not give each of the card's SMs two blocks, a few SMs took a second block while the rest idled
        through it (Qwen3-0.6B's 1024-row o and down, 64 groups over 60 SMs). The same bits either way"""
        sms = int(torch.cuda.get_device_properties(self.dev).multi_processor_count)
        return (R + 15) // 16 < 2 * sms

    # the card state's switches (`st["switch"]`, int32 on the card): the share of each matvec slice warmed into L2
    # ahead of it, in 256ths (`CARD_WARM`)
    SW_WARM = 0

    def _card_switch_init(self) -> torch.Tensor:
        return torch.tensor([round(256 * float(self.CARD_WARM))], dtype=torch.int32, device=self.dev)

    @staticmethod
    def _card_switch_ptr(st: dict[str, Any], i: int) -> ctypes.c_void_p:
        """a kernel's pointer to switch `i` of the card state"""
        return ctypes.c_void_p(int(st["switch"].data_ptr()) + 4 * i)

    def _card_buffers(
        self,
        st: dict[str, Any],
        a: int,
        b: int,
        T: int,
        tail: bool,
        mma: bool | None = None,
        rows: bool = False,
        paged: bool = False,
    ) -> dict[str, Any]:
        """the static buffers of the (run, T, GEMV) graph, made once; the graph itself is captured by
        `_card_capture` on the first pass, after that pass's inputs are in place. `rows`: the T rows are sequences
        of their own (a fork's or a batch's), laid out by `st["rw"]`, and the key carries it. `paged`: the rows are
        the prefix cache's card region's, read and written through its row map (`st["pg"]`), and the key carries it"""
        if mma is None:
            mma = self._card_mma_for(T)
        key = self._card_key(a, b, T, tail, bool(mma), rows, paged)
        g = st["graphs"].get(key)
        if g is not None:
            return g
        H, Hq, Hk, D, I = self._card_dims()
        # the GEMV-facing buffers carry the padded row count: the fp32 kernels' 1/2/4/8/16/32, the mma
        # kernel's fixed 32
        M = 32 if mma else self._card_m(T)
        dev, bf = self.dev, torch.bfloat16
        V = 0
        if tail:
            assert self.head is not None  # the tail runs the head, so it is on the card
            V = int(self.head.weight.shape[0])
        cap = int(st["pg"]["cap"]) if paged else int(st["arena"]["cap"])
        S = (cap + self._attn_group(D) - 1) // self._attn_group(D)
        g = {
            "key": key,
            "paged": paged,
            "mma": mma,
            "m": torch.zeros(M, I, dtype=bf, device=dev) if mma else None,
            "h": torch.zeros(M, H, dtype=bf, device=dev),
            "x": torch.zeros(M, H, dtype=bf, device=dev),
            "y": torch.zeros(M, H, dtype=bf, device=dev),
            "qkv": torch.zeros(M, (Hq + 2 * Hk) * D, dtype=bf, device=dev),
            "q": torch.zeros(M, Hq, D, dtype=bf, device=dev),
            "att": torch.zeros(M, Hq * D, dtype=bf, device=dev),
            "gu": torch.zeros(M, 2 * I, dtype=bf, device=dev),
            # the attention's per-group states and arrival counts (the group's keys are the kernel's; the
            # count of groups covers the arena, so one graph serves every length)
            "S": S,
            "part_m": torch.zeros(S * T * Hq, dtype=torch.float32, device=dev),
            "part_l": torch.zeros(S * T * Hq, dtype=torch.float32, device=dev),
            "part_acc": torch.zeros(S * T * Hq * D, dtype=torch.float32, device=dev),
            "cnt": torch.zeros(T * Hq, dtype=torch.int32, device=dev),
            "n0": torch.zeros(1, dtype=torch.int32, device=dev),
            "depth": torch.arange(T, dtype=torch.int32, device=dev),
            # a chain until a pass says otherwise (the warm-up run reads it: a row parented to itself
            # would never end its ancestor walk)
            "par": torch.arange(-1, T - 1, dtype=torch.int32, device=dev),
            "logits": torch.zeros(M, V, dtype=bf, device=dev) if tail else None,
            "logits_f": torch.zeros(1, V, dtype=torch.float32, device=dev) if tail else None,  # the sampler's row
            "chain": torch.arange(-1, T - 1, dtype=torch.int32, device=dev),
        }
        st["graphs"][key] = g
        return g

    @staticmethod
    def _card_key(
        a: int, b: int, T: int, tail: bool, mma: bool, rows: bool = False, paged: bool = False
    ) -> tuple[Any, ...]:
        """a pass graph's key: its run, width, tail and GEMV, and how it reads the rows - a fork's or a batch's rows
        layout, the prefix cache's row map, or the arena's front"""
        return (a, b, T, tail, bool(mma), *(("rows",) if rows else ()), *(("paged",) if paged else ()))

    def _card_body(self, st: dict[str, Any], g: dict[str, Any]) -> None:
        """the pass's kernels over `g`'s buffers, layers a..b-1 and the tail when the key carries it: run
        eagerly for the warm-up, then recorded by the capture"""
        import ctypes

        a, b, T, tail = g["key"][:4]
        k = st["k"]
        H, Hq, Hk, D, I = self._card_dims()
        M = 32 if g.get("mma") else self._card_m(T)
        P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
        # the rows' arenas: the prefix cache's card region, read and written through its row map (the map's pointer
        # the kernels' last argument), or the card graph's own arena at its front
        paged = bool(g.get("paged"))
        if paged:
            card = st["pg"]["card"]
            tables = self._card_tables(st, st["pg"]["cap"])
            tbl = [P(card.tbl)]
            arenas = {i: (card.arenas[i][0, 0], card.arenas[i][0, 1]) for i in range(a, b)}
        else:
            ar = st["arena"]
            tables = self._card_tables(st, ar["cap"])
            tbl = []
            arenas = {i: (ar["A"][ar["slot"][i], 0], ar["A"][ar["slot"][i], 1]) for i in range(a, b)}
        Ls = [self._card_weights(st, i) for i in range(a, b)]
        mma = bool(g.get("mma"))
        rows = g["key"][5:] == ("rows",)
        sandwich = self.fam.sandwich
        cen = ci(1 if self.fam.norm_centered else 0)  # the norms scale by 1 + w
        act = "gelu" if act_name(self.cfg) == "gelu_pytorch_tanh" else "silu"
        gemv = f"btb_gemv_bf16_m{M}"
        gemv_act = f"btb_gemv_{act}_bf16_m{M}"
        via = "_tbl" if paged else ""
        # the one attention every pass takes (btb_attn_flash.cuh): a row's bits its own, so a prompt's chunk makes the
        # rows these steps make. A block a row tile of 8 and a group of the keys, its warps the group's tiles
        attn = f"btb_attn_flash_rows_d{D}" if rows else f"btb_attn_flash_d{D}"
        nrk = f"btb_norm_rope_kv_rows_d{D}" if rows else f"btb_norm_rope_kv{via}_d{D}"
        if attn not in k.fn or nrk not in k.fn:
            raise RuntimeError(f"[card] no kernel for head_dim {D}")
        G = Hq // Hk
        S = int(g["S"])
        attn_grid = (T * ((G + 7) // 8), S, Hk) if rows else ((T * G + 7) // 8, S, Hk)
        attn_block = (self._attn_warps(D) * 32, 1, 1)
        attn_map = [] if rows else [tbl[0] if tbl else P(None)]  # the tree form's row map (null: none)
        # where each row's keys lie: the tree's (its prefix length, the rows' depths and parents), or the rows'
        # layout, one array for every rows graph
        where = [P(st["rw"])] if rows else [P(g["n0"]), P(g["depth"])]
        walk = [P(st["rw"])] if rows else [P(g["n0"]), P(g["par"])]

        # the pass's matvec weights in launch order, each warmed into L2 on the side stream once the one before it is
        # launched (`_card_warm`): through the kernels between them the chain leaves the DRAM idle
        seq: list[tuple[torch.Tensor, int, int]] = []
        for L in Ls:
            seq += [(L["qkv"], (Hq + 2 * Hk) * D, H), (L["o"], H, Hq * D), (L["gu"], 2 * I, H), (L["down"], H, I)]
        if tail and self.head is not None:
            seq.append((self.head.weight, int(self.head.weight.shape[0]), H))
        warm = self._card_warm(st, mma, seq)
        # the graph's kernel variant (`g["fused"]`, the step lanes' own): the gate, up and activation as one kernel
        # and a short weight's matvec at 8 rows a block - fewer kernels, so fewer edges where another program on the
        # card takes it and btb's L2 with it. The same bits as the plain ones; which is faster is the tuner's to find
        fused = bool(g.get("fused")) and mma
        glu = f"btb_gemv_mma_glu_{act}"
        glu_on = fused and glu in k.fn

        def matvec(W: torch.Tensor, xin: torch.Tensor, yout: torch.Tensor, R: int, C: int) -> None:
            if fused and "btb_gemv_mma8_bf16" in k.fn and self._card_mma_narrow(R):
                k.launch(
                    "btb_gemv_mma8_bf16",
                    ((R + 7) // 8, 1, 1),
                    (32 * self._card_mma_warps(R, C), 1, 1),
                    [P(W), P(xin), P(yout), ci(R), ci(C), ci(T)],
                )
            elif mma:
                k.launch(
                    "btb_gemv_mma_bf16",
                    (self._card_mma_grid(R, C), 1, 1),
                    (32 * self._card_mma_warps(R, C), 1, 1),
                    [P(W), P(xin), P(yout), ci(R), ci(C), ci(T)],
                )
            else:
                k.launch(gemv, ((R + 3) // 4, 1, 1), (128, 1, 1), [P(W), P(xin), P(yout), ci(R), ci(C)])
            warm()

        warm()
        y_prev = None
        for n, L in enumerate(Ls):
            kb, vb = arenas[a + n]
            # where head g's row r lies (elements): g * hs + r * rs, read off the arena's [Hk, cap, D] view
            kv_strides = (ci(int(kb.stride(0))), ci(int(kb.stride(1))))
            cos_t, sin_t = tables[self.layer_types[a + n]]
            # the last layer's MLP output folded into this input norm's add (a sandwich layer added its own)
            k.launch(
                "btb_add_rmsnorm",
                (T, 1, 1),
                (256, 1, 1),
                [P(g["h"]), P(y_prev), P(L["ln1"]), cf(L["eps"]), P(g["x"]), ci(H), cen],
            )
            matvec(L["qkv"], g["x"], g["qkv"], (Hq + 2 * Hk) * D, H)
            k.launch(
                nrk,
                (Hq + 2 * Hk, T, 1),
                (32, 1, 1),
                [
                    P(g["qkv"]),
                    P(L["wq"]),
                    P(L["wk"]),
                    cf(L["eps"]),
                    P(cos_t),
                    P(sin_t),
                    *where,
                    P(kb),
                    P(vb),
                    P(g["q"]),
                    ci(T),
                    ci(Hq),
                    ci(Hk),
                    *kv_strides,
                    cen,
                    *tbl,
                ],
            )
            k.launch(
                attn,
                attn_grid,
                attn_block,
                [
                    P(g["q"]),
                    P(kb),
                    P(vb),
                    P(g["att"]),
                    *walk,
                    ci(T),
                    ci(Hq),
                    ci(Hk),
                    *kv_strides,
                    cf(L["scale"]),
                    P(g["part_m"]),
                    P(g["part_l"]),
                    P(g["part_acc"]),
                    P(g["cnt"]),
                    ci(L["win"]),
                    *attn_map,
                ],
            )
            matvec(L["o"], g["att"], g["y"], H, Hq * D)
            if sandwich:
                # the attention output normed, then added; the MLP reads its own input norm of the sum
                k.launch(
                    "btb_sandwich_add",
                    (T, 1, 1),
                    (256, 1, 1),
                    [P(g["h"]), P(g["y"]), P(L["ln2"]), cf(L["eps"]), ci(H), cen],
                )
                k.launch(
                    "btb_add_rmsnorm",
                    (T, 1, 1),
                    (256, 1, 1),
                    [P(g["h"]), P(None), P(L["pre_ff"]), cf(L["eps"]), P(g["x"]), ci(H), cen],
                )
            else:
                k.launch(
                    "btb_add_rmsnorm",
                    (T, 1, 1),
                    (256, 1, 1),
                    [P(g["h"]), P(g["y"]), P(L["ln2"]), cf(L["eps"]), P(g["x"]), ci(H), cen],
                )
            if glu_on:
                # gate, up and the activation in one kernel, the activation in its epilogue - the plain matvec's
                # and btb_{act}_mul's bits at the plain kernel's warps for the merged weight (folded into the down
                # projection's x load instead, each row group recomputed the whole row's act(gate) * up before
                # its loads and stalled the stream: Qwen3-0.6B's down 21.6 us against 1.7 + its GEMV apart)
                k.launch(
                    glu,
                    ((I + 15) // 16, 1, 1),
                    (32 * self._card_mma_warps(2 * I, H), 1, 1),
                    [P(L["gu"]), P(g["x"]), P(g["m"]), ci(I), ci(H), ci(T)],
                )
                warm()
            else:
                matvec(L["gu"], g["x"], g["gu"], 2 * I, H)
            if mma:
                if not glu_on:
                    k.launch(
                        f"btb_{act}_mul",
                        (min(4096, (M * I + 255) // 256), 1, 1),
                        (256, 1, 1),
                        [P(g["gu"]), P(g["m"]), ci(M), ci(I)],
                    )
                matvec(L["down"], g["m"], g["y"], H, I)
            else:
                # act(gate) * up folded into the down projection's x load: the same bits as the two kernels
                k.launch(
                    gemv_act, ((H + 3) // 4, 1, 1), (128, 1, 1), [P(L["down"]), P(g["gu"]), P(g["y"]), ci(H), ci(I)]
                )
                warm()
            if sandwich:
                # the MLP output normed, then added: the residual is whole, nothing carries into the next norm
                k.launch(
                    "btb_sandwich_add",
                    (T, 1, 1),
                    (256, 1, 1),
                    [P(g["h"]), P(g["y"]), P(L["post_ff"]), cf(L["eps"]), ci(H), cen],
                )
            else:
                y_prev = g["y"]
        assert sandwich or y_prev is not None  # the segment holds at least one layer, so the loop set the residual
        if tail:
            assert self.head is not None  # the tail runs the final head
            norm_w, head_w = self.norm.weight, self.head.weight
            if norm_w.device.type != "cuda" or head_w.device.type != "cuda" or head_w.dtype != torch.bfloat16:
                raise RuntimeError("[card] the tail needs the final norm and the head on the card in bf16")
            V = int(head_w.shape[0])
            k.launch(
                "btb_add_rmsnorm",
                (T, 1, 1),
                (256, 1, 1),
                [P(g["h"]), P(y_prev), P(norm_w), cf(float(self.cfg.rms_norm_eps)), P(g["x"]), ci(H), cen],
            )
            matvec(head_w, g["x"], g["logits"], V, H)
        elif y_prev is not None:
            g["h"][:T].add_(y_prev[:T])
        warm.join()

    # the share of each warp's k slice of a matvec's weights warmed into L2 ahead of it (`_card_warm`); 0: none
    CARD_WARM = 0.0
    # the step loop picks among the switches' settings as it runs, by their measured speed (`_Tuner`)
    CARD_TUNE = True

    # the step lanes come in the fused kernels too, for the tuner to weigh against the plain ones (`_card_fuse_ok`)
    CARD_FUSE = True

    def _card_fuse_ok(self, st: dict[str, Any]) -> bool:
        """the step lanes in the fused kernels as well as the plain: where the tuner runs (`CARD_TUNE`), the build
        has them and the step takes the tensor-core matvec they are variants of"""
        k = st["k"]
        return (
            self.CARD_FUSE
            and self.CARD_TUNE
            and self._card_mma_for(1)
            and ("btb_gemv_mma8_bf16" in k.fn or any(f"btb_gemv_mma_glu_{a}" in k.fn for a in ("silu", "gelu")))
        )

    def _card_tuner(self, st: dict[str, Any], lanes: list[tuple[str, bool]]) -> _Tuner | None:
        """the card state's tuner over the step's `lanes` - (the usual or the deep queue, the plain or the fused
        kernels) - at the switches' settings as they stand, the usual plain lane first, the one alone on the card
        takes (`_Tuner`). Kept with the card state, one for each set of lanes, so an answer starts from what the last
        one over the same lanes found - an answer of one step has no deep lane, and a tuner made again for it cost
        the next answer its windows: half of it spent trying four arms afresh"""
        if not self.CARD_TUNE:
            return None
        tuners: dict[tuple[tuple[str, bool], ...], _Tuner] = st.setdefault("tuners", {})
        tu = tuners.get(tuple(lanes))
        if tu is not None:
            return tu
        base = [int(v) for v in st["switch"].tolist()]
        label = lambda ln: ("queued deep" if ln[0] == "deep" else "usual") + (", fused" if ln[1] else "")
        tu = tuners[tuple(lanes)] = _Tuner([(label(ln), base, ln) for ln in lanes], self.dev)
        return tu

    # the warming kernel's blocks beside the chain's: four loads standing a thread keep the DRAM fed at 24
    WARM_BLOCKS = 24

    def _card_warm(self, st: dict[str, Any], mma: bool, seq: list[tuple[torch.Tensor, int, int]]) -> _Warm:
        """the pass's L2 warming (`_Warm`, `btb_l2_warm`) over its matvec weights `seq` in launch order, on the card
        state's side stream: the share of every warp's slice of every row the switch names (`SW_WARM`), the slices as
        each weight's matvec cuts them. Captured only where warming is on (`CARD_WARM`): at a share of 0 each warming
        kernel leaves at once, but a launch still costs its node. Nothing it does reaches a result: the bits are the
        chain's alone"""
        k = st["k"]
        if self.CARD_WARM <= 0 or "btb_l2_warm" not in k.fn:
            return _Warm(k, None, None, [], 0)
        w = st.get("warm")
        if w is None:
            dev = st["stream"].device
            w = st["warm"] = {
                "stream": torch.cuda.Stream(device=dev),
                "sink": torch.zeros(1, dtype=torch.int32, device=dev),
            }
        # the tensor-core kernel's warps each take a k slice of whole 32-element super-tiles; the fp32 kernel's one
        # warp a row walks it from its start
        plan = [(W, R, C, self._card_mma_warps(R, C) if mma else 1) for W, R, C in seq]
        return _Warm(k, w["stream"], w["sink"], plan, self.WARM_BLOCKS, self._card_switch_ptr(st, self.SW_WARM))

    def _card_record(self, st: dict[str, Any], g: dict[str, Any], body: Any, what: str) -> None:
        """`body` once eagerly on the arena's stream (the warm-up), then captured there, so the captured
        kernel nodes inherit the stream's persisting window"""
        s = st["stream"]
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        before = self._trace_mem()
        cg = torch.cuda.CUDAGraph()
        self._capture(cg, st["pool"], s, body)
        g["graph"] = cg
        self.log(f"[card] {what}{self._trace_grew(before)}")

    def _trace_mem(self) -> tuple[int, int]:
        """-vv: torch's reserved bytes on the card and the bytes Windows counts this process using there, to say
        what a step took on the card and where it sits (0s with the trace off)"""
        if not trace.ON:
            return 0, 0
        torch.cuda.synchronize(self.dev)
        info = device_mod._wddm_info(self.dev)
        # free-read: the trace's figure, never a decision
        return int(torch.cuda.memory_reserved(self.dev)), (info[1] if info is not None else 0)

    def _trace_grew(self, before: tuple[int, int]) -> str:
        """-vv: the card memory taken since `before` (`_trace_mem`), as the tail of the line that says what took it -
        in torch's pools, and past them (the driver's graph execs, the context); '' with the trace off"""
        if not trace.ON:
            return ""
        res, used = self._trace_mem()
        d_res = res - before[0]
        d_out = (used - before[1]) - d_res if used and before[1] else 0
        # signed: a capture starts by emptying torch's cache (torch.cuda.graph), so the blocks it held show as given
        return f" ({'+' if d_res >= 0 else '-'}{_size(abs(d_res))} in torch's pools, " + (
            f"{'+' if d_out >= 0 else '-'}{_size(abs(d_out))} outside them)"
        )

    def _card_capture(self, st: dict[str, Any], g: dict[str, Any]) -> None:
        """Capture the pass graph over `g`'s buffers. The sequence runs once eagerly first (the module's
        warm-up), and that run writes the cache slots the pass's own rows take - so the caller sets the pass's
        length, depths and parents before this, and the eager run touches nothing but rows the replay rewrites."""
        a, b, T, tail = g["key"][:4]
        self._card_record(
            st,
            g,
            lambda: self._card_body(st, g),
            f"layers {a}-{b - 1} captured as one graph for {T}-row passes{' with the head' if tail else ''}"
            f"{' (tensor-core GEMV)' if g.get('mma') else ''}",
        )

    # -- the self-advancing step: the greedy loop as replays with the host one token behind ------------------

    def _card_table(self, st: dict[str, Any]) -> torch.Tensor | None:
        """the embedding table on the card for the step graph's own lookup: the head when the weights are tied,
        else a copy the scheduler grants room for (None when it does not - the loop then keeps its host gather)"""
        t = st.get("table")
        if t is not None:
            return t
        head = self.head
        if (
            head is not None
            and self.head_key == self.prefix + "embed_tokens.weight"
            and head.weight.device.type == "cuda"
        ):
            st["table"] = head.weight
            return head.weight
        tab = self.embed_table
        assert tab is not None  # the step graph seats the embedding table
        nbytes = tab.numel() * tab.element_size()
        sched = getattr(self, "scheduler", None)
        try:
            if sched is not None:
                sched.grant(nbytes, "table", requester="card step graph: the embedding table", device=self.dev)
        except MemoryError:
            st["table"] = None
            return None
        t = tab.to(self.dev, torch.bfloat16 if tab.dtype == torch.bfloat16 else tab.dtype)
        st["table"] = t
        self.log(f"[card] embedding table on the card ({nbytes / 2**20:.0f} MB) for the step graph")
        return t

    def _card_greedy_ok(self, cache: Any, B: int, am: Any, on_layer: Any, prefill_only: bool, session: Any) -> bool:
        if not (
            B == 1
            and am is None
            and on_layer is None
            and not prefill_only
            and session is None
            and getattr(self, "card_pipeline", True)
            and self._card_ready()
        ):
            return False
        st = self._card_state()
        return (
            st["segments"] == [(0, self.L)]
            and self.head is not None
            and self.norm is not None
            and self.head.weight.device.type == "cuda"
            and self._card_table(st) is not None
        )

    def _card_unroll(self) -> int:
        """steps per replay of the step graph: every graph boundary costs the platform's submission latency
        (~0.17 ms here), so a small model's fast step takes several steps a replay; a model whose step is long
        takes one. From the warm-up's one-row cost; 1 before it is known."""
        cost = getattr(self, "_card_cost", None)
        c1 = float(cost[1]) if cost and 1 in cost else None
        if c1 is None or c1 >= 0.010:
            return 1
        if c1 >= 0.005:
            return 2
        return 4

    # the step graph's execs: the most replays the card can hold ahead of the host (`_card_depth`); how many it
    # holds is the lane's (`_Tuner`)
    CARD_DEPTH = 6

    # the step loop moves to one-step replays queued deep where they measure faster: while another program shares
    # the card (`_Tuner`)
    CARD_CONTEND = True

    def _card_step_lane(
        self,
        st: dict[str, Any],
        table: torch.Tensor,
        U: int,
        smp: Any,
        past: int,
        first: int,
        fused: bool = False,
        paged: bool = False,
    ) -> dict[str, Any]:
        """the step graph at `U` steps a replay as a lane of the step loop: captured on first use - `_card_depth()`
        execs of the same steps, launched in turn (an exec launched again while its last launch still runs waits
        for it), each writing its own pinned token slots - its length and token set to the answer's start, its
        sampling configured, and its execs' bookkeeping for this answer fresh"""
        g, body = self._card_step_graph(st, table, U, smp, fused, paged)
        g["n0"].fill_(past)
        g["ids"].fill_(first)
        if g["graph"] is None:
            # the warm-up run consumes tokens and advances the length: both are reset after it
            self._card_record(
                st,
                g,
                lambda: body(0),
                f"{U} one-row step{'s' if U > 1 else ''} captured as a self-advancing graph"
                f"{' (the fused kernels)' if fused else ''}",
            )
            g["execs"] = [g["graph"]]
            before = self._trace_mem()
            for e in range(1, len(g["pin_tok"]) // U):
                cg = torch.cuda.CUDAGraph()
                self._capture(cg, st["pool"], st["stream"], functools.partial(body, e))
                g["execs"].append(cg)
            if trace.ON:  # no log line says this: the trace does
                trace.event(
                    "card graph: %d more execs of the %d-step lane%s%s",
                    len(g["execs"]) - 1,
                    U,
                    " (fused)" if fused else "",
                    self._trace_grew(before),
                )
            g["n0"].fill_(past)
            g["ids"].fill_(first)
        if not smp.greedy:
            # the pipeline's config, filled before the replays: the temperature and top-p, and the seed as two
            # int32 halves (the pick derives each step's key from it and the length, so a run repeats on a seed)
            g["fp"][0] = 1.0 / smp.temperature
            g["fp"][1] = float(smp.top_p)
            sd = int(smp.seed or 0)
            u32 = lambda v: v - (1 << 32) if v >= (1 << 31) else v
            g["seed"].copy_(torch.tensor([u32(sd & 0xFFFFFFFF), u32((sd >> 32) & 0xFFFFFFFF)], dtype=torch.int32))
        g["pin_n0"].fill_(past)
        E = len(g["execs"])
        return {
            "U": U,
            "g": g,
            "execs": g["execs"],
            "E": E,
            "pin_tok": g["pin_tok"],
            "pin_n0": g["pin_n0"],
            "pin_ts": g["pin_ts"],
            "uses": [None] * E,  # the replay each exec last ran, this answer
            "next": 0,
        }

    def _card_say(self, line: str) -> None:
        """a change in how btb runs the card that the person at the console should see (the contention's queue):
        on the console whatever the log's verbosity, where there is one (no stderr under pythonw or a service), and in
        the log"""
        err = sys.stderr
        if err is not None:
            with contextlib.suppress(OSError, ValueError):  # a console closed under the process
                err.write(f"  btb: {line}\n")
                err.flush()
        self.log(f"[card] {line}")

    def _card_depth(self) -> int:
        """the step graph's execs, each writing its own pinned token slots: the replays the card can hold queued
        while the host reads the oldest's tokens (2 to 8)"""
        return max(2, min(8, int(self.CARD_DEPTH)))

    def _card_step_graph(
        self,
        st: dict[str, Any],
        table: torch.Tensor,
        U: int,
        sampling: Any = None,
        fused: bool = False,
        paged: bool = False,
    ) -> tuple[dict[str, Any], Callable[[int], None]]:
        """U one-row steps as one graph: each step's token embedding, every layer, the head, the pick (the argmax,
        or the sample drawn inside the replay from the card's generator, its temperature and top-p read off
        device buffers, its top-k part of the key) written back as the next token, the cache's length advanced
        and the token published - the host only replays, and streams the tokens as they land. `paged`: the rows
        through the prefix cache's row map, every row the replays reach reserved and mapped before the first"""
        mma = self._card_mma_for(1)
        U = int(U)
        # sampling is None only on a greedy step; `sampled` gates every sampling.* read below
        sampled = sampling is not None and not sampling.greedy
        top_k = int(sampling.top_k) if sampled else 0
        top_p_on = bool(sampled and sampling.top_p < 1.0)
        key: tuple[Any, ...] = (0, self.L, 1, True, mma, "step", U, sampled, top_k, top_p_on, bool(fused))
        key += ("paged",) if paged else ()
        g = st["graphs"].get(key)
        if g is not None and g["graph"] is not None:
            return g, _captured  # replayed as captured: the body is never run again
        g = dict(self._card_buffers(st, 0, self.L, 1, True, mma, paged=paged))
        g["key"] = key
        g["graph"] = None
        g["U"] = U
        g["fused"] = bool(fused)  # the kernel variant `_card_body` runs (`fused` there)
        g["ids"] = torch.zeros(1, dtype=torch.long, device=self.dev)
        if sampled:
            # the fused pick's scratch and its config off device buffers (filled before the replay, so one capture
            # serves every temperature/top-p/seed): fp = [1/T, top_p], the row key derived on the card from the
            # seed and the advancing length so a replay draws what the sequential loop's key_for(pos) would
            g["fp"] = torch.ones(2, dtype=torch.float32, device=self.dev)
            g["seed"] = torch.zeros(2, dtype=torch.int32, device=self.dev)
            g["keys"] = torch.zeros(2, dtype=torch.int32, device=self.dev)
            g["gM"] = torch.zeros(1, dtype=torch.int32, device=self.dev)
            g["ghist"] = torch.empty(2048, dtype=torch.int64, device=self.dev)
            g["gpreK"] = torch.zeros(1, dtype=torch.int32, device=self.dev)
            g["gpreP"] = torch.zeros(1, dtype=torch.int32, device=self.dev)
            g["gcarry"] = torch.empty(1, dtype=torch.int64, device=self.dev)
            g["gbest"] = torch.zeros(1, dtype=torch.int64, device=self.dev)
        # the token and the length leave the card inside the graph, written straight into pinned host memory by
        # one kernel node: a copy and an event enqueued between two replays are their own submissions and cost
        # 0.3-0.45 ms a step on this platform, where the host polling a pinned counter costs nothing the card
        # sees. Each exec writes its own U token slots; the host reads a replay's slots one replay behind,
        # before the exec that owns them is launched again
        g["pin_tok"] = torch.zeros(self._card_depth() * U, dtype=torch.long, pin_memory=True)
        g["pin_n0"] = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        # each step's end on the card's clock (ns), beside its token: a replay's time as the card ran it
        g["pin_ts"] = torch.zeros(self._card_depth() * U, dtype=torch.long, pin_memory=True)
        st["graphs"][key] = g

        k = st["k"]
        P = k.ptr

        I = ctypes.c_int

        def pick_nodes() -> None:
            # the pipeline as graph nodes over the row's logits: derive the key, then max -> the top-k/top-p
            # radix levels -> the Gumbel-max draw. gbest packs (ordered value, ~index); the low 32 bits inverted
            # are the token. The whole card runs one row (grid-stride), an ordinary launch (contention-safe, and
            # recorded like any node - unlike a cooperative launch)
            lf = g["logits_f"]
            lf.copy_(g["logits"][:1])  # the sampler kernels take float32; the head left bf16 (a captured cast)
            Vw = int(lf.shape[-1])
            nb = min(max(1, (Vw + 1023) // 1024), 4 * k.sms)
            k.launch("btb_sample_keys", (1, 1, 1), (1, 1, 1), [P(g["seed"]), P(g["n0"]), I(0), P(g["keys"])])
            g["gM"].zero_()
            g["gbest"].zero_()
            g["gpreK"].zero_()
            g["gpreP"].zero_()
            k.launch("btb_sample_max", (nb, 1, 1), (1024, 1, 1), [P(lf), I(Vw), P(g["fp"]), P(g["gM"])])
            for mode, pre, flo, run in ((0, g["gpreK"], g["gpreP"], top_k > 0), (1, g["gpreP"], g["gpreK"], top_p_on)):
                if not run:
                    continue
                for lvl in (0, 1, 2):  # three radix levels: the exact 32-bit threshold
                    g["ghist"].zero_()
                    k.launch(
                        "btb_sample_hist",
                        (nb, 1, 1),
                        (1024, 1, 1),
                        [P(lf), I(Vw), P(g["fp"]), I(lvl), I(mode), P(g["gM"]), P(pre), P(flo), P(g["ghist"])],
                    )
                    k.launch(
                        "btb_sample_bin",
                        (1, 1, 1),
                        (1024, 1, 1),
                        [I(lvl), I(mode), I(top_k), P(g["fp"]), P(g["ghist"]), P(pre), P(g["gcarry"])],
                    )
            k.launch(
                "btb_sample_draw",
                (nb, 1, 1),
                (1024, 1, 1),
                [P(lf), P(g["keys"]), I(Vw), P(g["fp"]), P(g["gM"]), P(g["gpreK"]), P(g["gpreP"]), P(g["gbest"])],
            )
            g["ids"].copy_((~g["gbest"]) & 0xFFFFFFFF)  # the packed winner's inverted low 32 bits = the token

        scale = self.embed_scale

        def body(exec_id: int) -> None:
            for i in range(U):
                slot = exec_id * U + i
                torch.index_select(table, 0, g["ids"], out=g["h"][:1])
                if scale is not None:
                    # the family's embedding scale (Gemma 3's sqrt(hidden)) as `embed` applies it - the bf16 rows times
                    # the float, one rounding - so a step's rows are the forward's: the step graph took the table's
                    # rows unscaled, and Gemma's greedy answer on the card parted from every other path's at once
                    g["h"][:1].mul_(scale)
                self._card_body(st, g)
                if sampled:
                    pick_nodes()
                else:
                    g["ids"].copy_(torch.argmax(g["logits"][:1], dim=-1))
                g["n0"].add_(1)
                k.launch(
                    "btb_publish",
                    (1, 1, 1),
                    (32, 1, 1),
                    [
                        P(g["n0"]),
                        P(g["ids"]),
                        P(g["pin_tok"][slot : slot + 1]),
                        P(g["pin_n0"]),
                        P(g["pin_ts"][slot : slot + 1]),
                    ],
                )

        # handed back beside the graph, never kept in it: the capture is its only caller, and a closure over `g`
        # kept in `g` is a cycle that holds the graph, its buffers and the table past the graph's drop
        return g, body

    def _card_generate_greedy(
        self,
        ids: torch.Tensor,
        max_new: int,
        eos: set[int],
        on_token: Callable[[int], Any] | None,
        t0: float,
        sampling: Any = None,
    ) -> list[int]:
        """The plain loop over the step graph: the replays are enqueued back to back (each consumes the token
        the one before wrote into the graph's own buffer), and the host reads tokens one step behind through a
        ring of pinned slots - so the card never waits on the host between steps. Greedy, bit for bit the step
        loop's answer: the same kernels, the argmax over the same bf16 logits. Sampled, the draws come from the
        card's generator seeded before the replays: a seed repeats on the same card, and the step loop's draws,
        keyed by the cache row, are another stream."""
        self._tag(PassTag.CUDA_GRAPH)
        max_new = int(max_new)
        smp = sampling or GREEDY
        target_len = int(ids.shape[1]) + max_new
        # the prefix cache's pages where the engine has them, as every decode's (`_decode_cache`): the step graph then
        # reads them through the card's row map, every row its replays reach reserved and mapped before the first
        cache = self._decode_cache(target_len)
        paged = bool(getattr(cache, "paged", False))
        logits = self._prefill(ids, cache)
        self.vram_trim("prefill")
        first = int(smp.pick_torch(logits[0, -1:], [smp.key_for(int(ids.shape[1]) - 1)])[0])
        out = [first]
        if on_token:
            on_token(first)
        if first in eos or max_new <= 1:
            return out
        st = self._card_state()
        steps = max_new - 1  # tokens still to produce

        def torch_steps() -> list[int]:
            """the answer on through the torch layers over the cache, token by token"""
            pos0 = int(ids.shape[1]) - 1
            for k in range(len(out), max_new):
                if self._stop_asked():
                    break
                lg = self.forward(torch.tensor([[out[-1]]], device=self.dev), cache=cache)
                tok = int(smp.pick_torch(lg[0, -1:], [smp.key_for(pos0 + k)])[0])
                out.append(tok)
                if on_token:
                    on_token(tok)
                if tok in eos:
                    break
            return out

        U = max(1, min(self._card_unroll(), steps))
        # the arena must hold every row the replays may write: whole replays, one past the last token. Where it
        # cannot take the cache's rows - or the card no longer has the room for the step graph's table - the answer
        # goes on through the torch layers
        if not self._card_arena_holds(cache, ((steps + U - 1) // U + 1) * U + 1):
            return torch_steps()
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        table = self._card_table(st)
        if table is None:
            return torch_steps()
        past = cache.get_seq_length()
        # the step graph at the unroll the platform's submissions call for, and - where contention may engage
        # (`CARD_CONTEND`) - at one step a replay: beside another program on the card, short replays queued deep
        # run in the gaps it leaves, where a longer replay spans them (Qwen3-0.6B at 4k beside a game: 7.7-8.1 ms a
        # token at one step queued five deep, 9-12 at one queued one deep, 8.2-8.8 at four steps either way). The
        # lanes share the model's buffers and the length; the answer moves between them at a replay's edge. Each
        # comes in the plain kernels and - where the build has them and the tuner may choose (`CARD_FUSE`) - the fused
        # ones (`_card_body`'s `fused`): fewer kernels a step, so fewer edges where another program takes the card
        # and btb's L2; which pays is measured, as the lanes are
        kinds = [("usual", U)] + ([("deep", 1)] if self.CARD_TUNE and self.CARD_CONTEND and U > 1 else [])
        variants = [False] + ([True] if self._card_fuse_ok(st) else [])
        # the usual lane first: one the card has no room to capture (another program took it) sends the answer down
        # the torch path, as a pass's graph build does (`_card_oom`); the others are the tuner's choices, and one the
        # card has no room for is left out
        lanes: dict[tuple[str, bool], dict[str, Any]] = {}
        for kind, Uk in kinds:
            for fz in variants:
                try:
                    lane = self._card_step_lane(st, table, Uk, smp, past, first, fz, paged)
                    lanes[(kind, fz)] = {**lane, "kind": kind}
                except RuntimeError as err:
                    if not self._is_card_oom(err):
                        raise
                    if (kind, fz) == ("usual", False):
                        self._card_oom(err)
                        return torch_steps()
                    self.log(f"[card] the {kind}{' fused' if fz else ''} lane left out: no room on the card for it")
        base_lane = lanes[("usual", False)]
        done = 0  # tokens the host has read off the replays
        stop = False
        # each replay's arm (`_Tuner`): its switches and its lane - the usual, one replay queued ahead of the one
        # the host reads; the deep, one step a replay and the lane's execs less one queued; plain or fused. A
        # replay's time is its last step's stamp on the card's clock less the replay before it's, a token's share of
        # that what the tuner compares
        tu = self._card_tuner(st, list(lanes)) if len(lanes) > 1 else None
        if tu is not None:
            tu.at_context(past)
        arms: list[int] = []
        recs: list[tuple[dict[str, Any], int, int, bool]] = []  # each replay: its lane, exec, first step, timed
        nxt = 0  # the next replay whose tokens the host reads
        prev_ts: int | None = None

        def read(r: int) -> bool:
            """replay r's tokens, streamed as each step lands, and its time; True at an end token"""
            nonlocal done, prev_ts
            ln, e, k0, timed = recs[r]
            Ul = int(ln["U"])
            for i in range(Ul):
                k = k0 + i
                if k >= steps:
                    return False  # a replay cut short by the answer's end: its last stamp is not waited for
                # step k has finished when the lane's own copy of the length reads past + k + 1; its token sits in
                # the slot of its exec and its place within the replay
                target = past + k + 1
                waited = time.perf_counter()
                while int(ln["pin_n0"][0]) < target:
                    if time.perf_counter() - waited > 30.0:
                        raise RuntimeError(f"[card] the step graph did not advance past {target} within 30 s")
                tok = int(ln["pin_tok"][e * Ul + i])
                out.append(tok)
                done = k + 1
                if on_token:
                    on_token(tok)
                if tok in eos:
                    return True
            ts = int(ln["pin_ts"][e * Ul + Ul - 1])
            if prev_ts is not None and timed and tu is not None:
                said = tu.after_replay(arms[r], (ts - prev_ts) / Ul / 1e9)
                if said is not None:
                    line, loud = said
                    if loud:
                        self._card_say(line)
                    else:
                        self.log(f"[card] {line}")
            prev_ts = ts
            return False

        cur = base_lane
        k_next = 0  # the next step a replay takes
        vs = getattr(self, "vram_state", None)
        budget = False  # another program asked for the card meanwhile: the rest of the answer on the passes
        while k_next < steps:
            if self._stop_asked():
                stop = True
                break
            if vs is not None and vs.asked:
                # the budget watcher found the card past its budget with this answer holding the engine: the replays
                # read no budget, so the answer goes on through the passes, whose policy gives the card back now
                # rather than once the answer ends
                budget = True
                break
            arm = tu.before_replay(st["switch"]) if tu is not None else 0
            want = lanes[tu.lanes[arm]] if tu is not None else base_lane
            switched = want is not cur
            if switched:
                # every replay read before the answer moves lanes, so the lane taken up starts from a length the
                # host knows; it takes the last replay's token from the card, in stream order
                while nxt < len(recs) and not stop:
                    stop, nxt = read(nxt), nxt + 1
                if stop:
                    break
                want["g"]["ids"].copy_(cur["g"]["ids"])
                want["pin_n0"].fill_(past + k_next)
                cur = want
            e = cur["next"]
            cur["next"] = (e + 1) % cur["E"]
            # the exec's last replay read before it runs again: its token slots are the ones this replay writes
            last = cur["uses"][e]
            while last is not None and nxt <= last and not stop:
                stop, nxt = read(nxt), nxt + 1
            if stop:
                break
            arms.append(arm)
            cur["execs"][e].replay()
            recs.append((cur, e, k_next, not switched))
            cur["uses"][e] = len(recs) - 1
            k_next += int(cur["U"])
            # the replays past the lookahead are read now: one ahead on the usual lane, the execs less one deep
            ahead = cur["E"] - 1 if cur["kind"] == "deep" else 1
            while nxt <= len(recs) - 1 - ahead and not stop:
                stop, nxt = read(nxt), nxt + 1
            if stop:
                break
        ended = stop
        if not stop:
            while nxt < len(recs):
                end, nxt = read(nxt), nxt + 1
                if end:
                    ended = True
                    break
        torch.cuda.current_stream().synchronize()
        # the rows the sequence's processed tokens occupy: the prompt and every token fed to a replay whose
        # output was kept (the one after an eos is not part of the answer)
        n = past + max(0, len(out) - 1)
        for i in layers:
            cache.layers[i].set_front(n)
        if paged:
            cache.crop_to(n)  # the rows reserved past the answer's end let go
        self.log(
            f"[stream] generated {len(out)} tokens over 1 rows in {time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, len(out)):.3f} s/step incl. prefill; the step graph, host {done} behind by one)"
        )
        if tu is not None:
            self.log(f"[card] step settings: {tu.report()}")
        if budget and not ended and len(out) < max_new:
            self.log("[card] another program asked for the card: the answer goes on through the passes")
            return torch_steps()
        return out

    def card_warm(self, ids: Tokens, t_max: int | None = None) -> int:
        """Capture the card graphs for every pass shape up front - one-row steps and verify passes of 2 ..
        `t_max` rows (the tree budget plus its root by default) - over a throwaway cache of `ids`, so no
        timed answer pays a capture. Returns the number of graphs captured; 0 where the card graph does not
        apply."""
        ids_t = torch.as_tensor(list(ids), dtype=torch.long).view(1, -1)
        if self.dev.type == Device.CUDA and self.fam.card_program() is not None:
            if not self._card_program_on():
                return 0  # switched off: nothing of the program made, held or timed
            t_max = int(t_max or (int(getattr(self, "tree_budget", 0) or 0) + 1))
            return self._card_program_warm(ids_t, max(1, min(t_max, self.CARD_T_MAX)))
        if self.dev.type != Device.CUDA or self._card_kernels() is None or not self._card_ready():
            return 0
        t_max = int(t_max or (int(getattr(self, "tree_budget", 0) or 0) + 1))
        t_max = max(1, min(t_max, self.CARD_T_MAX))
        n = 0
        with torch.inference_mode():
            # the cache the answers decode over (`_decode_cache`): the prefix cache's pages where the engine has them,
            # so the graphs captured are the ones its passes replay
            cache = self._decode_cache(int(ids_t.shape[1]) + t_max + 2)
            paged = bool(getattr(cache, "paged", False))
            self._prefill(ids_t, cache)
            tok = int(ids_t[0, -1])
            before = len(self._card_state()["graphs"]) if getattr(self, "_cg", None) is not None else 0
            cost: dict[int, float] = {}
            st = self._card_state()  # loads the kernels: the tensor-core GEMV's presence is known after this
            # graphs are keyed by the layer range on the card: one run for a resident model, several with a layer
            # on the CPU between them; a pass replays every run, so every run's graph is timed together
            segs = [tuple(sg) for sg in st["segments"]]
            tails = {sg: sg[1] == self.L and self.head is not None and self.norm is not None for sg in segs}
            forced = getattr(self, "card_mma", None)
            if forced is None and os.environ.get("BTB_CARD_MMA") in ("0", "1"):
                forced = os.environ["BTB_CARD_MMA"] == "1"
            variants = [bool(forced)] if forced is not None else ([False, True] if self._card_mma_avail() else [False])
            # the engine's one kernel is the card's and the model's (`_card_mma_pick`); each variant set per width while
            # it is timed, the choice made after
            choice: dict[int, bool] = self.__dict__.setdefault("_mma_for", {})
            timed_all: dict[int, dict[bool, float]] = {}
            moved = False
            for T in range(1, t_max + 1):
                if not self._card_pass_ok(cache, 1, T, cache.get_seq_length(), None, None, None):
                    break
                base = cache.get_seq_length()
                # each variant's graph captured through a real pass, then the two replayed in turns and timed
                # on the card by events: the host's launch jitter is of the size of the difference
                for mma in variants:
                    choice[T] = mma
                    self.aa(None)
                    try:
                        self.forward([[tok] * T], cache=cache, last_only=False)
                    finally:
                        self.ab()
                    self.ad(cache, base, [0])
                if self._card_state() is not st:
                    # the placement moved during the warm-up (room made on demand for its passes): these graphs are
                    # another placement's, and the widths timed so far choose
                    self.log(f"[card] the warm-up stops at {T}-row passes: the placement moved under it")
                    moved = True
                    break
                timed: dict[bool, float] = {m: float("inf") for m in variants}
                keys = {
                    m: [self._card_key(sg[0], sg[1], T, tails[sg], m, paged=paged) for sg in segs] for m in variants
                }
                if any(
                    key not in st["graphs"] or st["graphs"][key].get("graph") is None
                    for ks in keys.values()
                    for key in ks
                ):
                    # a build found no room (`_card_oom`: another program took the card during the load): nothing to
                    # time at this width, and no wider pass fits either - the widths timed so far choose
                    self.log(f"[card] the warm-up stops at {T}-row passes: the card graph found no room to build")
                    break
                graphs = {m: [st["graphs"][key]["graph"] for key in keys[m]] for m in variants}
                torch.cuda.synchronize()
                for _rep in range(4):
                    for m in variants:
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for g in graphs[m]:
                            g.replay()
                        e1.record()
                        torch.cuda.synchronize()
                        timed[m] = min(timed[m], e0.elapsed_time(e1) / 1e3)
                # the replays wrote rows past `base`; the length stands where it was
                self.ad(cache, base, [0])
                timed_all[T] = timed
                if len(timed) > 1 and T in (1, 2, 4, 8, 16, t_max):
                    self.log(
                        f"[card] {T}-row pass: fp32 chain {timed[False] * 1e3:.2f} ms, tensor cores "
                        f"{timed[True] * 1e3:.2f} ms"
                    )
            # a width set for a variant and never timed (the build found no room, the loop broke) holds the last
            # variant tried, not the engine's one kernel: let go, it takes `_mma_one` like any width past the timed
            # ones - kept, a later placement's passes of that width ran the other kernel than the step
            for T in [t for t in choice if t not in timed_all]:
                del choice[T]
            if timed_all:
                # one kernel for every width: the two sum a row in different orders, and a step on one with a
                # pass on the other parted at bf16 near-ties (0/8 identical at 256 tokens on the 0.6B). The
                # engine keeps the kernel that is faster at the width its loop lives at - the widest pass for
                # a speculative engine, the one-row step for a greedy one - and the other's graphs are dropped
                w = max(timed_all) if int(getattr(self, "v_max", 0) or 0) > 0 else 1
                pick = min(variants, key=lambda m: timed_all[w][m])
                self._mma_one = pick
                for T, timed in timed_all.items():
                    choice[T] = pick
                    cost[T] = timed[pick]
                    for mma in variants:
                        if mma != pick:
                            for sg in segs:
                                st["graphs"].pop(self._card_key(sg[0], sg[1], T, tails[sg], mma, paged=paged), None)
                if len(variants) > 1:
                    self.log(
                        f"[card] one GEMV for every width: {'tensor cores' if pick else 'fp32 chain'} "
                        f"(the {w}-row pass {timed_all[w][pick] * 1e3:.2f} ms against "
                        f"{timed_all[w][not pick] * 1e3:.2f}; the step {timed_all[1][pick] * 1e3:.2f} against "
                        f"{timed_all[1][not pick] * 1e3:.2f} ms)"
                    )
            n = len(st["graphs"]) - before
            # the cost curve covers the whole model only; with a layer on the CPU the timing above is partial, and
            # with the placement moved under it another placement's
            if cost and not moved and segs == [(0, self.L)] and tails[segs[0]]:
                self._card_cost = cost
                c1 = cost[1]
                self.log(
                    "[card] pass cost by rows: "
                    + ", ".join(f"{T}:{c / c1:.2f}x" for T, c in cost.items() if T in (1, 2, 4, 8, 16, t_max))
                    + f" (one row {c1 * 1e3:.2f} ms)"
                )
            if paged:
                cache.release()  # the throwaway's pages back to the pool now, not when it is collected
        return n

    def _card_program_warm(self, ids_t: torch.Tensor, t_max: int) -> int:
        """a family's card program's cost curve, measured up front with no cache and no expert read: for each graph
        width M the passes of 1 .. `t_max` rows take, the program's verify graphs replayed (captured on the first
        one, timed on the card's clock over PROGRAM_SAMPLES after it, the fastest counted) at the context of `ids_t`'s
        length, plus the host's handshake between the layers - the wait for each router's publish - once, the
        cleanest pass's (`_card_program_time`). The curve (`_card_prog_cost`) is
        the card's compute a pass of T rows costs - the rows' experts and the host layers are not in it - so it is
        the engine's pass cost (`_card_cost`, which sizes a pass the pricer does not) only where the program runs the
        whole model on the card, as `card_warm`'s curve is. Every other graph (the step's, a tap's, a commit's) is
        captured on its first use. Returns the graphs captured; 0 where the program does not run the model."""
        cls = self.fam.card_program()
        if cls is None or self._card_kernels() is None or not self._card_program_on():
            return 0
        if getattr(self, "_cp", None) is None:
            self._cp = cls(self)
        prog = self._cp
        if not prog.ok():
            return 0
        n0 = max(1, int(ids_t.shape[-1]))
        before = len(prog.graphs)
        widths = sorted({self._card_m(T) for T in range(1, t_max + 1)})
        with torch.inference_mode(), self.device.hold() as place:
            if place.version != prog.version:
                return 0
            for M in widths:
                self._card_program_time(prog, M, n0)  # the first replay of a width captures its graphs: untimed
            # each width timed PROGRAM_SAMPLES times, the widths in turns
            samples: dict[int, list[tuple[float, float]]] = {M: [] for M in widths}
            for _rep in range(self.PROGRAM_SAMPLES):
                for M in widths:
                    samples[M].append(self._card_program_time(prog, M, n0))
        # a width's compute is its graphs' time on the card's clock, the fastest sample's; the host's handshake (the
        # launches, the wait for each router's publish past the card's own work) is the same whatever the rows, so
        # one figure for every width - the cleanest pass's. By the host's clock alone, the wait carried every delay
        # the card's queue saw: 0.4 to 200 ms a pass beside another program, the 2-row pass priced at 6, 21 and 53
        # one-row ones where its samples caught the bursts and the 1-row ones did not - and speculation with it
        card = {M: min(c for _w, c in got) for M, got in samples.items()}
        handoff = min(max(0.0, w - c) for got in samples.values() for w, c in got)
        if trace.ON:
            for M, got in samples.items():
                trace.event(
                    "card program warm-up: %d-row passes on the card %s ms, by the host's clock %s",
                    M,
                    " ".join(f"{c * 1e3:.2f}" for _w, c in got),
                    " ".join(f"{w * 1e3:.2f}" for w, _c in got),
                )
        cost = {T: card[self._card_m(T)] + handoff for T in range(1, t_max + 1)}
        self._card_prog_cost = cost
        if not self.host:
            self._card_cost = cost
        c1 = cost[1]
        self.log(
            f"[card] {self.fam.name}'s card program: {len(prog.graphs) - before} graphs captured; its compute by "
            "rows (no expert reads, no host layers): "
            + ", ".join(f"{T}:{c / c1:.2f}x" for T, c in cost.items() if T in (1, 2, 4, 8, 16, t_max))
            + f" (one row {c1 * 1e3:.2f} ms)"
        )
        return len(prog.graphs) - before

    def _card_program_time(self, prog: Any, T: int, n0: int) -> tuple[float, float]:
        """(seconds by the host's clock, seconds of the graphs on the card's): one T-row verify pass of the
        program's resident layers on the card at `n0` rows of context - its segments' graphs, closes and tail
        replayed in turn with the host's part rehearsed (`between(i, dry=True)`: the publish waited for, no expert
        read) and the host layers left out - the card's compute a pass of T rows costs, over the program's own
        buffers and no cache (`rehearse`)"""
        M = prog.rehearse(T, n0)
        L = int(prog.L)
        mode, tap = "tree", False
        spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        def run(key: Any, body: Callable[[], None]) -> None:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            self._card_graph_run(prog, key, body)
            e1.record()
            spans.append((e0, e1))

        collecting = gc.isenabled()
        # the sample named for a profiler (nsys -t nvtx): its kernels apart from the passes' and the other widths'
        torch.cuda.nvtx.range_push(f"card program warm-up: {M}-row pass")
        try:
            torch.cuda.synchronize(self.dev)
            # a collection the replays' small allocations set off is the host's time, not the pass's
            gc.disable()
            t0 = time.perf_counter()
            for a, b in prog.segs:
                for j in range(a, b):
                    run((j, M, mode, tap), functools.partial(prog.layer_body, j, M, mode, tap))
                    prog.between(j, dry=True)
                if b < L:
                    run(("close", b - 1, M, mode, tap), functools.partial(prog.close_body, b - 1, M, mode, tap))
            run((L, M, mode, tap), functools.partial(prog.tail_body, M, mode, tap))
            torch.cuda.synchronize(self.dev)
            wall = time.perf_counter() - t0
            return wall, sum(e0.elapsed_time(e1) for e0, e1 in spans) / 1e3
        finally:
            torch.cuda.nvtx.range_pop()
            if collecting:
                gc.enable()
            prog.rehearsed()

    def _spec_full(self, v_max: int | None = None) -> int:
        """The widest speculative pass, the root included: the tree's rows, or a chain's `v_max` drafts (the
        call's, else the model's) and the root, whichever is wider - held to the widest the family verifies exactly as
        placed (`verify_rows`: Qwen4's card program, whose wider passes took the torch path)."""
        v = int((getattr(self, "v_max", 0) if v_max is None else v_max) or 0)
        full = max(int(getattr(self, "tree_budget", 0) or 0), v) + 1
        bound = self.fam.verify_rows(self)
        return full if bound is None else max(1, min(full, int(bound)))

    def _spec_budget(self, ema_tokens: float, passes: int, v_max: int | None = None, past: int = 0) -> int:
        """Rows a speculative pass may carry, the root included, from the cost curve of the card graph's passes -
        the warm-up's, as the passes measure it now (`_spec_curve`) - and the running acceptance: the widest pass
        costing at most a quarter more than the
        one-row step, and only while the tokens a pass yields pay for its width - otherwise one-row passes,
        with a wide probe every sixteenth pass so a stretch of accepted drafts can reopen the tree. Without a
        cost curve (the MLX and host paths) the configured budget stands: the tree's rows, or a chain's
        `v_max` drafts and the root, whichever is wider."""
        full = self._spec_full(v_max)
        cost = self._spec_curve()
        if not cost or 1 not in cost or len(cost) <= 1 or full <= 1:
            return full
        slope = getattr(self, "_mlx_attn_slope", None)
        if slope and getattr(self, "_card_cost", None) is None and past > 0:
            # a pass over `past` rows: every node past the first reads the prefix again, at the warm-up's cost
            # per node and row (the slope at the nearest timed length above, the last one beyond it) - added to the
            # widths the warm-up's short context prices, not to those the passes measured here, whose seconds hold it
            b = next((s for r, s in slope if past <= r), slope[-1][1])
            n_attn = sum(1 for lt in self.layer_types if lt in (LayerKind.FULL, LayerKind.SLIDING))
            pc: SpecCost | None = getattr(self, "_spec_cost", None)
            measured = pc.measured() if pc is not None else set()
            cost = {T: c + (0 if T in measured else n_attn * (T - 1) * past * b) for T, c in cost.items()}
        c1 = cost[1]
        # a pass may cost what it yields: the widest pass whose cost, in one-row steps, is within the tokens
        # a pass has been returning (a quarter over a step is always allowed, so a tree can start)
        allow = max(1.25, float(ema_tokens)) * c1
        wide = [T for T, c in cost.items() if T <= full and c <= allow]
        cap = max(wide) if wide else 1
        if passes % 16 == 0:
            # the periodic probe: the widest pass within a quarter over a step, whatever the yield has been
            probe = [T for T, c in cost.items() if T <= full and c <= 1.25 * c1]
            cap = max(cap, max(probe) if probe else 1)
        return max(1, cap)

    def _spec_pricer(self, v_max: int | None = None) -> SpecCost:
        """The engine's pricing of its speculative passes (`SpecCost`): made on first use and kept across calls,
        so what a call learned of the drafter's acceptance and the rows' expert reads sizes the next call's first
        passes; handed the widest pass, the base cost curve `_spec_budget` reads and a missed expert's seconds."""
        pc: SpecCost | None = getattr(self, "_spec_cost", None)
        if pc is None:
            top_k = int(getattr(getattr(self, "cfg", None), "num_experts_per_tok", 0) or 0)
            pc = SpecCost(SpecCost.gamma_of(int(getattr(self, "n_experts", 0) or 0), top_k))
            self._spec_cost = pc
        cost = (
            getattr(self, "_card_cost", None) or getattr(self, "_mlx_cost", None) or getattr(self, "_host_cost", None)
        )
        store = getattr(self, "expert_store", None)
        miss = getattr(store, "miss_s", None)
        # `spec_price` off prices no read (the pricer inactive): the passes sized as with no store to read from - what a
        # test of the verify pass pins, so what earlier calls taught the pricer cannot choose to draft nothing
        miss_s = miss() if callable(miss) and bool(getattr(self, "spec_price", True)) else 0.0
        # a card program's passes run at its graphs' widths, its rows padded to them: a 3-row pass is a 4-row one
        prog = cost is not None and cost is getattr(self, "_card_prog_cost", None)
        pc.price(self._spec_full(v_max), cost, miss_s, self._card_width if prog else None)
        return pc

    def _card_width(self, T: int) -> int:
        """the rows a card program's pass of `T` runs at (its graphs' widths); a pass past them took the torch path,
        at its own"""
        return self._card_m(T) if T <= self.CARD_T_MAX else int(T)

    def _spec_curve(self) -> dict[int, float] | None:
        """the pass-cost curve the budget sizes a pass by: the warm-up's as the passes measure it now (the pricer's
        `curve_now`), the warm-up's own before a pricer is made"""
        pc: SpecCost | None = getattr(self, "_spec_cost", None)
        if pc is not None and pc.curve:
            return pc.curve_now()
        return (
            getattr(self, "_card_cost", None) or getattr(self, "_mlx_cost", None) or getattr(self, "_host_cost", None)
        )

    def _forward_card_segment(
        self, a: int, b: int, h: torch.Tensor, pas: Any, tail: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """layers a..b-1 as one replay over `h` [1, T, H]: returns (h, None), or (None, logits [1, T, V]
        float32) when the run carries the tail"""
        self._tag(PassTag.CUDA_GRAPH)
        cache, T, past = pas.cache, pas.T, pas.past
        st = self._card_state()
        self._card_bind(cache, st, T)
        g = self._card_buffers(st, a, b, T, tail, paged=bool(getattr(cache, "paged", False)))
        # the pass's inputs first: a first pass captures the graph after them, and its eager run then writes
        # only the cache slots this pass owns
        g["n0"].fill_(past)
        if T > 1:
            # a one-row pass has depth [0] and parent [-1] from its allocation; wider passes carry the tree
            g["depth"].copy_((pas.text_pos[0] - past).to(torch.int32))
            par = getattr(self, "ap", None)
            if par is None:
                g["par"].copy_(g["chain"])
            else:
                g["par"].copy_(torch.as_tensor(list(par), dtype=torch.int32))
        hin = h.reshape(T, -1)
        if "graph" not in g:
            g["h"][:T].copy_(hin)
            self._card_capture(st, g)
        g["h"][:T].copy_(hin)
        g["graph"].replay()
        n = past + T
        layers = cache.layers
        for i in range(a, b):
            layers[i].set_front(n)
        if tail:
            # the logits as the head wrote them, bf16, a view of the graph's own buffer: the callers take an
            # argmax or cast for themselves, and every one of them reads before the next replay; widening
            # 17 rows of the vocabulary to fp32 a pass cost more than the argmax that followed
            return None, g["logits"][:T].view(1, T, -1)
        return g["h"][:T].view(1, T, -1), None

    def _card_prefill_ok(
        self, cache: Any, B: int, T: int, past: int, am: torch.Tensor | None, on_layer: Any, stop_after: int | None
    ) -> bool:
        """a prompt's chunk the card graph's kernels take as its steps' rows (`_forward_card_prefill`): one sequence's
        cache, no caller's mask or early stop, an engine the card graph serves, and the build's GEMMs that sum a row as
        the steps' matvecs do - any width, any position, the first rows of a cache too, a layer hook's pass as well (it
        reads each layer's residual as the run leaves it), a verify pass's tree of the graph's widths. The kernels
        read after `_card_ready` loads them: read before, an engine's first prompt found none and took the torch layers"""
        if not (
            B == 1
            and T >= 1
            and am is None
            and stop_after is None
            and cache is not None
            and not forked(cache)
            and (T <= self.CARD_T_MAX or getattr(self, "ap", None) is None)
            and self._card_ready()
        ):
            return False
        k = Native.cuda
        return k is not None and "btb_gemm_mma_bf16" in k.fn and "btb_gemm_f32_bf16" in k.fn

    def _forward_card_prefill(
        self,
        a: int,
        b: int,
        h: torch.Tensor,
        pas: Any,
        tail: bool,
        all_rows: bool = False,
        bind_rows: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """layers a..b-1 over a prompt's chunk `h` [1, T, H] on the card graph's own kernels, launched as they come:
        the norms, the rope and the cache write, the attention (its prefill form) and the activation each the step's
        kernel over the chunk's rows, and every matvec a GEMM that sums a row as the step's does (btb_gemm.cuh, the
        kernel the warm-up picked for the engine's every width) - so each row is the row a step at its position makes,
        bit for bit, whatever the chunk's width, wherever it starts, and a conversation's next turn read from the
        cache decodes as the same prompt cold. Returns (h, None), or (None, logits) where the run carries the tail:
        the last row's [1, 1, V] (every row's with `all_rows`), as the graph's head writes them, bf16, the caller's
        own. `bind_rows`: the rows the cache is bound for past its length (the chunk's own by default; 0 for a
        layer-by-layer sweep, which reserved the prompt's rows before its first layer). The pass's layer hook
        (`pas.on_layer`) reads each layer's residual [1, T, H] as it leaves the layer.

        A verify pass's tree (its parents `self.ap`, its rows' depths the pass's positions, as the graph reads them)
        takes the attention's decode form, which walks a tree; a chain - a prompt's chunk - its prefill form, the same
        bits. Every buffer is taken before a row is written, so a refusal (`MemoryGrantError`) or a card out of memory
        leaves the cache as it was for the torch path to run the pass; once the hook has seen a layer, a failure is
        never run again (`CardPassFailed`): the hook would see the layers twice"""
        self._tag(PassTag.CARD_PREFILL)
        cache, T, past = pas.cache, pas.T, pas.past
        st = self._card_state()
        k = st["k"]
        paged = bool(getattr(cache, "paged", False))
        self._card_bind(cache, st, T if bind_rows is None else bind_rows)
        H, Hq, Hk, D, I = self._card_dims()
        P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
        if paged:
            card = st["pg"]["card"]
            cap = int(st["pg"]["cap"])
            tbl = [P(card.tbl)]
            arenas = {i: (card.arenas[i][0, 0], card.arenas[i][0, 1]) for i in range(a, b)}
        else:
            ar = st["arena"]
            cap = int(ar["cap"])
            tbl = []
            arenas = {i: (ar["A"][ar["slot"][i], 0], ar["A"][ar["slot"][i], 1]) for i in range(a, b)}
        if past + T > cap:
            # never past the rows the arena holds: the kernels write through no bound of their own
            raise RuntimeError(f"[card] the prefill's rows {past}..{past + T} lie past the card's {cap} rows")
        tables = self._card_tables(st, cap)
        Ls = [self._card_weights(st, i) for i in range(a, b)]
        mma = self._card_mma_for(T)
        sandwich = self.fam.sandwich
        cen = ci(1 if self.fam.norm_centered else 0)
        act = "gelu" if act_name(self.cfg) == "gelu_pytorch_tanh" else "silu"
        nrk = f"btb_norm_rope_kv{'_tbl' if paged else ''}_d{D}"
        parents = getattr(self, "ap", None) if T > 1 else None
        if parents is not None and T > self.CARD_T_MAX:
            raise RuntimeError(f"[card] a tree of {T} rows is past the attention's {self.CARD_T_MAX}")
        attn = f"btb_attn_flash_d{D}" if parents is not None else k.flash_prefill_kernel(D)
        if nrk not in k.fn or attn not in k.fn:
            raise RuntimeError(f"[card] no kernel for head_dim {D}")
        if tail and not self._card_tail_ok():
            raise RuntimeError("[card] the tail needs the final norm and the head on the card in bf16")
        # the chunk's buffers, the engine's scratch (granted as they grow, kept for the next chunk), every one taken
        # before a row of the cache is written
        dev, bf = self.dev, torch.bfloat16
        who = "the card's prefill: a prompt chunk's rows through the card graph's kernels"

        def buf(name: str, shape: tuple[int, ...], dt: torch.dtype = bf) -> torch.Tensor:
            return self.scratch.take(f"card prefill {name}", shape, dt, dev, who)

        hb = buf("h", (T, H))
        x, y = buf("x", (T, H)), buf("y", (T, H))
        qkv = buf("qkv", (T, (Hq + 2 * Hk) * D))
        q, att = buf("q", (T, Hq, D)), buf("att", (T, Hq * D))
        gu, m = buf("gu", (T, 2 * I)), buf("m", (T, I))
        # the launch's rows ride the norm/rope write's grid.y, which the card caps: a longer chunk writes in slices,
        # each from its own start (a chain's depths from 0 in every slice)
        nsl = (T + self.GRID_Y - 1) // self.GRID_Y
        where = buf("where", (nsl + T,), torch.int32)  # each slice's start, then the rows' depths
        if parents is None:
            run = buf("rows", (T * Hq * D,), torch.float32)
        else:
            # the decode form's group states: the groups the card's rows reach, as a graph's cover the arena
            S = (cap + self._attn_group(D) - 1) // self._attn_group(D)
            run = buf("par", (T,), torch.int32)
            part_m, part_l = buf("part m", (S * T * Hq,), torch.float32), buf("part l", (S * T * Hq,), torch.float32)
            part_acc = buf("part acc", (S * T * Hq * D,), torch.float32)
            cnt = buf("cnt", (T * Hq,), torch.int32)
        r0 = 0 if all_rows else T - 1
        tail_bufs = self._card_tail_buffers(T - r0, H) if tail else None
        hb.copy_(h.reshape(T, H))
        for s in range(nsl):
            where[s : s + 1].fill_(past + s * self.GRID_Y)
        n0p, depth = where[:1], where[nsl:]
        if parents is None:
            torch.arange(T, out=depth)  # a slice's rows read the depths from the first: 0, 1, ... from its start
        else:
            # the tree in one slice (CARD_T_MAX rows at most): its rows' depths, their parents, as the graph reads them
            depth.copy_((pas.text_pos[0] - past).to(torch.int32))
            run.copy_(torch.as_tensor(list(parents), dtype=torch.int32))
            cnt.zero_()  # the last group of each row folds and leaves its count at 0; a buffer made new holds anything

        def gemm(W: torch.Tensor, xin: torch.Tensor, yout: torch.Tensor, R: int, C: int, rows: int) -> None:
            self._card_gemm(k, mma, W, xin, yout, R, C, rows)

        G = Hq // Hk
        tpb = k.flash_prefill_rows(D) // G
        smem = k.flash_prefill_smem(D)
        y_prev: torch.Tensor | None = None
        hooked = False
        try:
            for n, L in enumerate(Ls):
                kb, vb = arenas[a + n]
                hs, rs = int(kb.stride(0)), int(kb.stride(1))
                cos_t, sin_t = tables[self.layer_types[a + n]]
                k.launch(
                    "btb_add_rmsnorm",
                    (T, 1, 1),
                    (256, 1, 1),
                    [P(hb), P(y_prev), P(L["ln1"]), cf(L["eps"]), P(x), ci(H), cen],
                )
                gemm(L["qkv"], x, qkv, (Hq + 2 * Hk) * D, H, T)
                # q normed and rotated into its own buffer, k and v into the cache at positions past .. past + T - 1
                for s in range(nsl):
                    s0 = s * self.GRID_Y
                    rows = min(self.GRID_Y, T - s0)
                    k.launch(
                        nrk,
                        (Hq + 2 * Hk, rows, 1),
                        (32, 1, 1),
                        [P(qkv[s0:]), P(L["wq"]), P(L["wk"]), cf(L["eps"]), P(cos_t), P(sin_t), P(where[s:])]
                        + [P(depth), P(kb), P(vb), P(q[s0:]), ci(rows), ci(Hq), ci(Hk), ci(hs), ci(rs), cen, *tbl],
                    )
                if parents is None:
                    k.launch(
                        attn,
                        ((T + tpb - 1) // tpb, Hk, 1),
                        (128, 1, 1),
                        [P(q), P(kb), P(vb), P(att), ci(past), ci(T), ci(Hq), ci(Hk), ci(hs), ci(rs), cf(L["scale"])]
                        + [ci(L["win"]), tbl[0] if tbl else P(None), P(run)],
                        shared=smem,
                    )
                else:
                    k.launch(
                        attn,
                        ((T * G + 7) // 8, S, Hk),
                        (self._attn_warps(D) * 32, 1, 1),
                        [P(q), P(kb), P(vb), P(att), P(n0p), P(run), ci(T), ci(Hq), ci(Hk), ci(hs), ci(rs)]
                        + [cf(L["scale"]), P(part_m), P(part_l), P(part_acc), P(cnt), ci(L["win"])]
                        + [tbl[0] if tbl else P(None)],
                    )
                gemm(L["o"], att, y, H, Hq * D, T)
                if sandwich:
                    k.launch(
                        "btb_sandwich_add", (T, 1, 1), (256, 1, 1), [P(hb), P(y), P(L["ln2"]), cf(L["eps"]), ci(H), cen]
                    )
                    k.launch(
                        "btb_add_rmsnorm",
                        (T, 1, 1),
                        (256, 1, 1),
                        [P(hb), P(None), P(L["pre_ff"]), cf(L["eps"]), P(x), ci(H), cen],
                    )
                else:
                    k.launch(
                        "btb_add_rmsnorm",
                        (T, 1, 1),
                        (256, 1, 1),
                        [P(hb), P(y), P(L["ln2"]), cf(L["eps"]), P(x), ci(H), cen],
                    )
                # gate and up, the activation times up, down: the plain kernels' bits, which the step's fused ones
                # (the activation in the gate's epilogue, or folded into the down matvec's load) give as well
                gemm(L["gu"], x, gu, 2 * I, H, T)
                k.launch(
                    f"btb_{act}_mul", (min(4096, (T * I + 255) // 256), 1, 1), (256, 1, 1), [P(gu), P(m), ci(T), ci(I)]
                )
                gemm(L["down"], m, y, H, I, T)
                if sandwich:
                    k.launch(
                        "btb_sandwich_add",
                        (T, 1, 1),
                        (256, 1, 1),
                        [P(hb), P(y), P(L["post_ff"]), cf(L["eps"]), ci(H), cen],
                    )
                else:
                    y_prev = y
                if pas.on_layer is not None:
                    # the residual leaving the layer, a copy, as a stretch of the graph ending there leaves it (its
                    # add rounded as the next layer's norm rounds it): a hooked pass makes the rows an unhooked one does
                    hooked = True
                    pas.on_layer(a + n, (hb + y_prev if y_prev is not None else hb.clone()).view(1, T, H))
            logits = None
            if tail_bufs is not None:
                # the rows the head reads: the last alone, or every one - each row's norm and logits its own either way
                logits = self._card_tail(
                    hb[r0:].view(1, T - r0, H), y_prev[r0:] if y_prev is not None else None, bufs=tail_bufs
                )
        except (RuntimeError, MemoryError) as e:
            if hooked and self._is_card_oom(e):
                # what would send the pass down the torch path, its hook fed already
                raise CardPassFailed(f"[card] a hooked pass ran out of room past its first layer: {e}") from e
            raise
        # the rows the cache holds now, once nothing of the pass can fail: a pass run again on the torch path finds
        # the cache as it was
        for i in range(a, b):
            cache.layers[i].set_front(past + T)
        if logits is not None:
            return None, logits
        if y_prev is not None:
            hb.add_(y_prev)
        return hb.view(1, T, H), None

    # the most rows a launch takes on grid.y (the card's limit), where a chunk's rows ride it
    GRID_Y = 65535

    def _card_gemm(
        self, k: Any, mma: bool, W: torch.Tensor, x: torch.Tensor, y: torch.Tensor, R: int, C: int, rows: int
    ) -> None:
        """y[:rows] = x[:rows] W^T [R, C] as the step's matvec sums each row (btb_gemm.cuh): the tensor cores' at
        the matvec's warps for the weight's shape, or the fp32 chain - the one the warm-up picked (`mma`). The rows
        ride grid.y in blocks of 128 or 8: more than it takes run in slices, each row its own sum either way"""
        P, ci = k.ptr, ctypes.c_int
        step = self.GRID_Y * (128 if mma else 8)
        for m0 in range(0, rows, step):
            m = min(step, rows - m0)
            if mma:
                k.launch(
                    "btb_gemm_mma_bf16",
                    ((R + 63) // 64, (m + 127) // 128, 1),
                    (128, 1, 1),
                    [P(W), P(x[m0:]), P(y[m0:]), ci(R), ci(C), ci(m), ci(self._card_mma_warps(R, C))],
                )
            else:
                k.launch(
                    "btb_gemm_f32_bf16",
                    ((R + 31) // 32, (m + 7) // 8, 1),
                    (128, 1, 1),
                    [P(W), P(x[m0:]), P(y[m0:]), ci(R), ci(C), ci(m)],
                )

    def _card_tail_ok(self) -> bool:
        """the card graph's kernels run the tail: the card graph on, the final norm and the head on the card, bf16"""
        if not self._card_ready() or self.head is None or self.norm is None:
            return False
        norm_w, head_w = self.norm.weight, self.head.weight
        return norm_w.device.type == "cuda" and head_w.device.type == "cuda" and head_w.dtype == torch.bfloat16

    def _card_tail_buffers(self, R: int, H: int) -> tuple[torch.Tensor, torch.Tensor]:
        """the tail's buffers for R rows: their norm (scratch), and the logits [R, V] - the caller's own, as the
        head's matmul makes them on the torch path: a pass's logits outlive it (a prompt fed in chunks keeps every
        chunk's for its rows), so never a buffer the next pass writes again"""
        assert self.head is not None  # `_card_tail_ok`
        x = self.scratch.take("card tail x", (R, H), torch.bfloat16, self.dev, "the card's tail over a prompt's rows")
        logits = torch.empty(R, int(self.head.weight.shape[0]), dtype=torch.bfloat16, device=self.dev)
        return x, logits

    def _card_tail(
        self, h: torch.Tensor, y: torch.Tensor | None = None, bufs: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """the final norm and the head over `h`'s rows [1, R, H] (bf16 on the card; `y` the last layer's output the
        norm adds in first, as the graph's tail does) on the card graph's kernels: each row's logits [1, R, V], bf16,
        as a step's tail makes them - the norm its kernel, the head the GEMM that sums a row as its matvec does. None
        where the card graph does not run the tail (`_card_tail_ok`). `bufs`: the buffers, taken already
        (`_card_tail_buffers`)"""
        if not self._card_tail_ok():
            return None
        k = self._card_kernels()
        assert k is not None and self.norm is not None and self.head is not None  # `_card_tail_ok`
        norm_w, head_w = self.norm.weight, self.head.weight
        P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
        R, H, V = int(h.shape[1]), int(h.shape[2]), int(head_w.shape[0])
        hb = h.reshape(R, H)
        if hb.dtype != torch.bfloat16 or hb.device != self.dev or not hb.is_contiguous():
            hb = self.scratch.take(
                "card tail h", (R, H), torch.bfloat16, self.dev, "the card's tail over a prompt's rows"
            )
            hb.copy_(h.reshape(R, H))
        x, logits = bufs if bufs is not None else self._card_tail_buffers(R, H)
        k.launch(
            "btb_add_rmsnorm",
            (R, 1, 1),
            (256, 1, 1),
            [P(hb), P(y), P(norm_w), cf(float(self.cfg.rms_norm_eps)), P(x), ci(H)]
            + [ci(1 if self.fam.norm_centered else 0)],
        )
        self._card_gemm(k, self._card_mma_for(R), head_w, x, logits, V, H, R)
        return logits.view(1, R, V)

    # -- a family's card program: its layers as graphs replayed in turn, the host between them ------------------

    def _card_program_on(self) -> bool:
        """the family's card program not switched off (`card_programs`, `BTB_CARD_PROGRAM=0`): the one gate its
        passes and its warm-up both ask, so a program switched off is never made, held or timed"""
        return bool(getattr(self, "card_programs", True)) and os.environ.get("BTB_CARD_PROGRAM", "1") != "0"

    def _card_program(
        self, cache: Any, B: int, T: int, past: int, am: Any, stop_after: int | None, positions: Any
    ) -> Any:
        """the family's card program for this pass, or None where it takes the torch path: a family with one
        (`Family.card_program`), a one-row step or a speculative pass of up to CARD_T_MAX rows over one sequence's
        cache, on the card with its kernels, and a model the program runs as placed (`ok`)"""
        cls = self.fam.card_program()
        if cls is None:
            return None
        spec = bool(getattr(self, "aq", False))
        if not (
            B == 1
            and 1 <= T <= self.CARD_T_MAX
            and (T == 1 or spec)
            and (positions is None or spec)
            and past > 0
            and am is None
            and stop_after is None
            and cache is not None
            and not forked(cache)
            and self.dev.type == Device.CUDA
            and self._card_program_on()
            and getattr(self, "_probe", None) is None
        ):
            return None
        prog = getattr(self, "_cp", None)
        if prog is None:
            if self._card_kernels() is None:
                return None
            prog = self._cp = cls(self)
        return prog if prog.ok() else None

    def _card_program_for_rows(self) -> Any:
        """the family's card program where it keeps the attention's rows in RAM (`kv_host`: a context the card has no
        room for) and reads them as the model is placed (`rows_ok`: its attention's own gates - a head shed to the host
        declines the program's passes, not this) - a torch-path pass's sparse attention then reads the rows through its
        kernels (Qwen4's `attend_rows`), so they never come to the card - else None"""
        if not getattr(self, "kv_host", False) or self.dev.type != Device.CUDA or not self._card_program_on():
            return None
        cls = self.fam.card_program()
        if cls is None or self._card_kernels() is None:
            return None
        prog = getattr(self, "_cp", None)
        if prog is None:
            prog = self._cp = cls(self)
        return prog if prog.rows_ok() else None

    def _card_graph_run(self, holder: Any, key: Any, body: Callable[[], None]) -> None:
        """`body`'s kernels as the graph `holder.graphs[key]`, captured on first use on the holder's stream and
        replayed on the current one; the engine's `card_program_eager` (a test's check) runs the body itself. The
        capture only records: nothing a body writes (a step's states) runs twice."""
        if getattr(self, "card_program_eager", False):
            body()
            return
        g = holder.graphs.get(key)
        if g is None:
            s = holder.stream
            s.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=holder.pool, stream=s, capture_error_mode="thread_local"):
                body()
            torch.cuda.current_stream().wait_stream(s)
            holder.graphs[key] = g
        g.replay()

    def _forward_card_program(
        self,
        prog: Any,
        ids: Any,
        h: torch.Tensor,
        cache: Any,
        past: int,
        positions: Any,
        last_only: bool,
        head: bool,
        on_layer: Callable[[int, torch.Tensor], Any] | None,
        place: Any = None,
    ) -> torch.Tensor:
        """One pass through a family's card program over the model as placed (`place`, the pass's held
        placement): each segment of resident layers its graphs in turn - after each layer i the host's part
        (`between(i)`: a mixture's routed experts through the store) - closed into the streams where a host layer
        follows; a host layer between segments through the host path (`run_layer`, the streams crossing the edge
        through the program's pinned rows); the tail last. A one-row pass outside speculation is a step (it commits
        as it goes); a speculative one verifies its chain or tree (`ap`) at its rows' positions, its commit handed
        to `ad`, a host layer's with its own. Returns the logits [1, T, V] bf16 (the final rows without `head`), a
        view of the program's buffer read before the next pass."""
        self._tag(PassTag.CUDA_GRAPH)
        T = int(h.shape[1])
        spec = bool(getattr(self, "aq", False))
        mode = "tree" if spec else "step"
        parents = chain_of(getattr(self, "ap", None) if spec else None, T)
        ids_l = [int(t) for t in torch.as_tensor(ids).reshape(-1).tolist()]
        if positions is not None:
            depth = [int(p) - past for p in torch.as_tensor(positions).reshape(-1).tolist()]
        else:
            depth = list(range(T))
        tap = on_layer is not None
        taps: dict[int, torch.Tensor] = {}
        pas = None
        if prog.host_layers:
            # the host layers' frame, as the torch path's pass makes it: the rope, the mask (the tree through it),
            # the ids; made before any layer writes the cache
            pas = self._host_frame(h, ids_l, cache, past, positions, taps.__setitem__ if tap else None)
            pas.place = place if place is not None else self.device.snapshot()
            if self.cold:
                self._cold_start(self.L)
        M = prog.begin(cache, h, ids_l, past, parents, depth, mode, tap)
        L = int(prog.L)
        run = self._card_graph_run
        hf: torch.Tensor | None = None  # the streams on the host, between host layers
        i = 0
        while i < L:
            if i not in prog.at:
                if hf is None:
                    hf = prog.leave()
                hf = self.device.run_layer(i, hf, pas)
                i += 1
                continue
            b = prog.seg_end(i)
            prog.enter(i, hf)
            hf = None
            for j in range(i, b):
                run(prog, (j, M, mode, tap), functools.partial(prog.layer_body, j, M, mode, tap))
                prog.between(j)
            if b < L:
                run(prog, ("close", b - 1, M, mode, tap), functools.partial(prog.close_body, b - 1, M, mode, tap))
            i = b
        prog.enter(L, hf)
        run(prog, (L, M, mode, tap), functools.partial(prog.tail_body, M, mode, tap))
        prog.end()
        if on_layer is not None:
            for i in range(L):
                on_layer(i, taps[i] if i in taps else prog.tap_rows(i, T))
        out = prog.logits(T) if head else prog.hidden(T)
        return out[:, -1:] if last_only else out

    # -- rows: a fork's or a batch's rows stepped together, a token each at its own position -------------------

    ROWS_ROOM = 256  # steps of room a regrowth of the arena makes for the rows (a regrowth drops the graphs)

    def _card_rows_ok(self, B: int, cache: Any = None) -> bool:
        """the card graph's rows pass takes B rows: the whole model one resident run, the final norm and the head
        on the card (a step is one replay, the logits its last node) - and, for a formed cache, every layer the
        arena's rows layer"""
        if not 1 <= B <= self.CARD_T_MAX or not self._card_ready():
            return False
        st = self._card_state()
        head, norm = self.head, self.norm
        if (
            st["segments"] != [(0, self.L)]
            or head is None
            or norm is None
            or head.weight.device.type != "cuda"
            or head.weight.dtype != torch.bfloat16
            or norm.weight.device.type != "cuda"
        ):
            return False
        return cache is None or all(isinstance(cache.layers[i], CardRowsLayer) for i in range(self.L))

    def _card_evict(self, ar: dict[str, Any]) -> None:
        """the arena's holder lets it go: the rows it has become copies of their own"""
        owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not None:
            held = [owner.layers[i] for i in ar["slot"] if isinstance(owner.layers[i], (GrowLayer, CardRowsLayer))]
            # the copies granted before any is made: an eviction the card cannot take is refused whole, the arena
            # still its holder's
            nbytes = sum(cl.detach_bytes() for cl in held)
            sched = getattr(self, "scheduler", None)
            if nbytes and sched is not None:
                sched.grant(
                    nbytes, "kv", requester="card arena: the rows it lets go, copied out", device=self.dev, draws=""
                )
            for cl in held:
                cl.detach()
        ar["owner"] = None

    def _card_rows_form(self, cache: Any, srcs: list[Any], rows: list[int]) -> list[int]:
        """`cache`'s layers as the arena's rows layers, row b going on from the rows of cache srcs[rows[b]] (a
        fork's rows all from its session's, a batch's each from its own): the sources end to end at the arena's
        front, the rows' steps after them. The session holding the arena keeps its rows where they are. Returns
        the rows' prefix lengths."""
        import weakref

        st = self._card_state()
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        lens = [int(c.layers[layers[0]].get_seq_length()) for c in srcs]
        offs = [sum(lens[:j]) for j in range(len(srcs))]
        base, W = sum(lens), len(rows)
        ar = self._card_arena(st, base + W * self.ROWS_ROOM)
        owner = ar["owner"]() if ar["owner"] is not None else None
        in_place = {
            j
            for j, c in enumerate(srcs)
            if owner is not None
            and c is owner
            and offs[j] == 0
            and all(isinstance(c.layers[i], GrowLayer) and c.layers[i]._attached() for i in layers)
        }
        self._card_evict(ar)
        ar["owner"] = weakref.ref(cache)
        A = ar["A"]
        sched = getattr(self, "scheduler", None)
        grant = sched.grant if sched is not None else None
        for i in layers:
            kb, vb = A[ar["slot"][i], 0], A[ar["slot"][i], 1]
            for j, c in enumerate(srcs):
                if j in in_place:
                    continue
                k, v = attention_rows(c.layers[i])
                if int(k.shape[0]) != 1 or int(k.shape[-2]) != lens[j]:
                    raise RuntimeError(
                        f"[card] layer {i}: a source of {tuple(k.shape)} rows, not one sequence of {lens[j]}"
                    )
                kb[:, offs[j] : offs[j] + lens[j]].copy_(k[0])
                vb[:, offs[j] : offs[j] + lens[j]].copy_(v[0])
            cache.layers[i] = CardRowsLayer(
                kb, vb, [offs[r] for r in rows], [lens[r] for r in rows], base, W, grant=grant
            )
        return [lens[r] for r in rows]

    def _card_rows_bind(self, cache: Any, st: dict[str, Any], need: int = 0) -> dict[str, Any]:
        """the arena holding `cache`'s rows with room for their next step (and `need` slots): taken back from any
        other cache holding it, grown when the rows outrun it"""
        import weakref

        l0 = cache.layers[0]
        need = max(int(need), l0.used + l0.W)
        ar = st["arena"]
        owner = ar["owner"]() if ar is not None and ar["owner"] is not None else None
        if (
            owner is not None
            and owner is cache
            and ar["cap"] >= need
            and all(cache.layers[i]._buf is not None for i in ar["slot"])
        ):
            return ar
        if ar is None or ar["cap"] < need:
            ar = self._card_arena(st, max(need, l0.used + l0.W * max(self.ROWS_ROOM, l0._t)))
            owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not cache:
            self._card_evict(ar)
            ar["owner"] = weakref.ref(cache)
        for i, j in ar["slot"].items():
            cl = cache.layers[i]
            kb = ar["A"][j, 0]
            if cl._buf is None or cl._buf[0].data_ptr() != kb.data_ptr():
                cl.attach(kb, ar["A"][j, 1])
        return ar

    def _card_rows_select(self, cache: Any, slots: list[int]) -> None:
        """the rows at `slots` become the rows (see `CardRowsLayer.select`), the arena made room for first"""
        st = self._card_state()
        self._card_rows_bind(cache, st, cache.layers[0].used_after(slots))
        for cl in cache.layers:
            cl.select(slots)

    def _card_rows_release(self, cache: Any) -> None:
        """`cache` closed: the arena is free for the next cache without copying the rows out"""
        st = getattr(self, "_cg", None)
        ar = st["arena"] if st is not None else None
        if ar is not None and ar["owner"] is not None and ar["owner"]() is cache:
            ar["owner"] = None
        for cl in cache.layers:
            if isinstance(cl, CardRowsLayer):
                cl._buf = cl._own = None

    def _card_rows_step(
        self, cache: Any, toks: list[int], taps: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """One token into each row of `cache` (formed by `_card_rows_form`): the rows' logits [B, V] float32 on the
        card, and the residual leaving each `taps` layer, {layer: [B, H] float32}. One replay (one a stretch between
        taps) of the graph captured at the GEMV's padded width; the rows past B are padding the kernels skip."""
        self._tag(PassTag.CUDA_GRAPH)
        B = len(toks)
        st = self._card_state()
        self._card_rows_bind(cache, st)
        l0 = cache.layers[0]
        Tb = self._card_m(B)
        rw = st.get("rw")
        if rw is None:
            rw = st["rw"] = torch.zeros(3 + 3 * self.CARD_T_MAX, dtype=torch.int32, device=self.dev)
        lay = [l0.base, l0._t, l0.W]
        for c, o, n in zip(l0.cols, l0.offs, l0.lens):
            lay += [c, o, n]
        lay += [0, 0, -1] * (Tb - B)
        rw[: len(lay)].copy_(torch.tensor(lay, dtype=torch.int32))
        h = self.embed(torch.tensor(toks, dtype=torch.long, device=self.dev).view(B, 1))
        if self.compute_dtype is not None:
            h = h.to(self.compute_dtype)
        x = h.reshape(B, -1)
        edges = [0, *sorted({int(i) + 1 for i in taps if 0 <= int(i) < self.L - 1}), self.L]
        seen: dict[int, torch.Tensor] = {}
        logits = None
        with self.device.hold():
            for a, b in itertools.pairwise(edges):
                tail = b == self.L
                g = self._card_buffers(st, a, b, Tb, tail, rows=True)
                if "graph" not in g:
                    # the capture's eager run writes this step's slots, which the replay rewrites
                    g["h"][:B].copy_(x)
                    self._card_capture(st, g)
                g["h"][:B].copy_(x)
                g["graph"].replay()
                x = g["h"][:B]
                if b - 1 in taps:
                    seen[b - 1] = x.float().cpu()
                if tail:
                    logits = g["logits"][:B]
        for cl in cache.layers:
            cl._t += 1
        assert logits is not None  # the last stretch carries the head
        # a copy: the graph's own buffer is the next replay's
        return logits.float(), seen

    def _forward_fast(self, h: torch.Tensor, pe: Any, cache: Any, last_only: bool, head: bool) -> torch.Tensor:
        self._tag(PassTag.CUDA_GRAPH)
        g = self._graphs(h.dtype, pe)
        g["h"].copy_(h[:, -1:, :])
        rope = self._rope_by_type(pe)
        for lt, (cos, sin) in g["rope"].items():
            cos.copy_(rope[lt][0][:, -1:, :])
            sin.copy_(rope[lt][1][:, -1:, :])
        hk, d = g["hk"], g["d"]
        stream = torch.cuda.current_stream()
        paged = bool(getattr(cache, "paged", False))
        for i in range(self.L):
            g["A"][i].replay()
            stream.synchronize()
            kn, vn = g["kv_pin"][:hk].view(1, hk, 1, d), g["kv_pin"][hk:].view(1, hk, 1, d)
            win = layer_window(self.cfg, self.layer_types[i])
            scale = self.resident[i].self_attn.scaling
            if paged:
                # the rows in the host's region, read through the conversation's map: the step's span, `attn_decode`'s
                # bits (`attn_spans`)
                cl = cache.layers[i]
                n = cl.get_seq_length()
                kf, vf = cl.append(kn, vn)
                amap, starts, ends = self._span_lists(cache, n, 1, win)
                hq = int(g["q_pin"].shape[0])
                Native.attn_spans(
                    g["q_pin"].view(1, hq, d),
                    kf[0],
                    vf[0],
                    amap,
                    starts,
                    ends,
                    float(scale),
                    g["out_pin"].view(1, hq, d),
                )
            else:
                kf, vf = cache.update(kn, vn, i)
                first = max(0, int(kf.shape[-2]) - win) if win else 0  # a sliding layer reads its last rows alone
                Native.attn_decode(g["q_pin"], kf[0][:, first:], vf[0][:, first:], scale, g["out_pin"])
            g["B"][i].replay()
        return self._finish(g["h"].clone(), last_only, head)

    def ad(self, cache: Any, base_len: int, path: NodePath) -> None:
        keep = list(range(base_len)) + [base_len + j for j in path]
        lazy: list[Any] = []
        flags: list[Any] = []
        paged = bool(getattr(cache, "paged", False))
        if paged:
            # a paged cache's accepted rows copied down into its own pages, every attention layer's at once, and the
            # rest let go (`PagedCache.keep_path`)
            cache.keep_path(base_len, list(path))
        for i, layer in enumerate(cache.layers):
            # every attention layer's keys and values are cropped to the accepted path; a sliding layer
            # keeps the whole cache (its window is a mask, not a shorter cache), so it is cropped too, and a
            # sparse layer's indexer keys with its rows
            if getattr(layer, "paged", False):
                continue
            if self.layer_types[i] in (LayerKind.FULL, LayerKind.SLIDING, LayerKind.QWEN_SPARSE):
                keep_path = getattr(layer, "keep_path", None)
                if keep_path is not None:
                    # a layer keeping its rows where a card program's kernels read them moves the path into place
                    # itself (`ArenaIndexedLayer`)
                    keep_path(base_len, path)
                    continue
                ik = indexer_keys(layer)
                if ik is not None and ik.numel():
                    if path == list(range(len(path))):
                        layer.indexer_keys = ik[:, : len(keep)]
                    else:
                        layer.indexer_keys = ik.index_select(1, torch.tensor(keep, device=ik.device))
                if isinstance(layer, GrowLayer) and not layer.shared and layer._buf is not None and layer._attached():
                    # a layer living in its buffer (the card's arena): the accepted path's rows move into
                    # place inside the buffer and the views are re-cut - nothing leaves the buffer
                    kb, vb = layer._buf
                    n = len(keep)
                    if path != list(range(len(path))):
                        idx = torch.tensor(keep[base_len:], device=kb.device)
                        kb[..., base_len:n, :] = kb.index_select(-2, idx)
                        vb[..., base_len:n, :] = vb.index_select(-2, idx)
                    layer.set_front(n)
                    if hasattr(layer, "cumulative_length"):
                        layer.cumulative_length = n
                    continue
                if isinstance(layer, GrowLayer) and layer.shared and layer._mx is not None:
                    # a shared layer crops, or gathers the path's rows into place on the GPU (an int8 layer's
                    # scales with them), off the torch view: the prefix stays where it is
                    if path == list(range(len(path))):
                        layer.crop(len(keep))
                    else:
                        flags += layer.gather(keep, base=base_len, lazy=True)
                    continue
                if getattr(layer, "keys", None) is not None:
                    if path == list(range(len(path))):
                        set_rows(layer, layer.keys[..., : len(keep), :], layer.values[..., : len(keep), :])
                    else:
                        idx = torch.tensor(keep, device=layer.keys.device)
                        set_rows(layer, layer.keys.index_select(-2, idx), layer.values.index_select(-2, idx))
                    if hasattr(layer, "cumulative_length"):
                        layer.cumulative_length = int(layer.keys.shape[-2])
            else:
                st = self.al.get(i)
                if st is None:
                    inp = getattr(self, "an", {}).get(i)
                    if inp is not None:
                        self.ah(layer, i, inp, path)
                    continue
                if path and hasattr(st, "restore"):
                    # MLX's checkpoints hand back lazy states, evaluated together below; a restore that wrote the
                    # accepted path's states into the layer itself hands back nothing
                    got = st.restore(path)
                    if got is not None:
                        lazy.append((layer, *got))
                    continue
                conv, rec = st[path[-1]] if path else self.am[i]
                c, r = self._lin(layer)
                c.copy_(conv)
                r.copy_(rec)
        if flags and not lazy:
            # the arena layers' gathers, one eval for every layer
            mlxdev.mx().eval(*flags)
        if lazy:
            # the DeltaNet states of the accepted path, recomputed into the cache's own buffers in one eval (a
            # synchronous one: the states are read through torch right after, by snapshots and the receipts); the
            # conv windows copied in, the arena layers' gathers with them
            mlxdev.mx().eval(*[rec for _, _, rec in lazy], *flags)
            for layer, conv, _rec in lazy:
                c, _r = self._lin(layer)
                c.copy_(conv)
        # a family's own states (Qwen4's n-gram embedding) put where the accepted path left them
        for commit in getattr(self, "spec_commits", ()):
            commit(path)

    def af(self, i: int, T: int, cl: Any) -> Any:
        """layer i's per-node (conv, recurrent) states for a speculative pass of T nodes: its own, since the pass's
        commit reads them after every later layer has run; granted and held as the engine's scratch"""
        c, r = self._lin(cl)
        who = f"the speculative pass's per-node DeltaNet states, layer {i}"
        return (
            self.scratch.take(f"delta nodes {i} conv", (T, *tuple(c.shape)), c.dtype, c.device, who),
            self.scratch.take(f"delta nodes {i} state", (T, *tuple(r.shape)), r.dtype, r.device, who),
        )

    def ag(
        self,
        la: Any,
        cl: Any,
        mixed_all: torch.Tensor,
        z_all: torch.Tensor,
        a_all: torch.Tensor,
        b_all: torch.Tensor,
        parents: Parents,
        T: int,
    ) -> Any:
        dev = mixed_all.device
        K = la.conv1d.weight.shape[-1]
        w_conv = la.conv1d.weight.squeeze(1).float()
        b_conv = None if la.conv1d.bias is None else la.conv1d.bias.float()
        conv0, rec0 = self._lin(cl)
        pre_win = conv0[0].float()
        xin = mixed_all[0].float()
        par = [parents[p] for p in range(T)]
        paths = []
        for j in range(T):
            pth, up = [j], par[j]
            while up >= 0:
                pth.append(up)
                up = par[up]
            paths.append(pth)
        C = xin.shape[1]
        win = torch.empty(T, C, K, dtype=torch.float32, device=dev)
        for j in range(T):
            pth = paths[j]
            for t in range(K):
                back = K - 1 - t
                win[j, :, t] = xin[pth[back]] if back < len(pth) else pre_win[:, K - 1 - (back - len(pth))]
        conv_out = (win * w_conv[None]).sum(-1)
        if b_conv is not None:
            conv_out = conv_out + b_conv[None]
        conv_out = F.silu(conv_out)
        Hk, Hv, dk, dv = int(la.num_k_heads), int(la.num_v_heads), int(la.head_k_dim), int(la.head_v_dim)
        q_, k_, v_ = torch.split(conv_out, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
        q_ = q_.reshape(T, Hk, dk).transpose(0, 1)
        k_ = k_.reshape(T, Hk, dk).transpose(0, 1)
        v_ = v_.reshape(T, Hv, dv).transpose(0, 1)
        if Hv // Hk > 1:
            q_ = q_.repeat_interleave(Hv // Hk, dim=0)
            k_ = k_.repeat_interleave(Hv // Hk, dim=0)
        l2n = lambda x: x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)
        q_ = l2n(q_) * (dk**-0.5)
        k_ = l2n(k_)
        beta = b_all[0].float().sigmoid().transpose(0, 1)
        g = (-la.A_log.float().exp()[None, :] * F.softplus(a_all[0].float() + la.dt_bias)).transpose(0, 1)
        S0 = rec0[0].float()
        cumg = torch.zeros(Hv, T, dtype=torch.float32, device=dev)
        for j in range(T):
            cumg[:, j] = g[:, j] + (cumg[:, par[j]] if par[j] >= 0 else 0.0)
        anc = torch.zeros(T, T, dtype=torch.bool, device=dev)
        for j in range(T):
            for a in paths[j][1:]:
                anc[j, a] = True
        anc_self = anc | torch.eye(T, dtype=torch.bool, device=dev)
        diff = cumg[:, :, None] - cumg[:, None, :]
        decay = torch.where(anc_self[None], diff.clamp(max=0.0).exp(), torch.zeros((), dtype=diff.dtype, device=dev))
        k_beta, v_beta = k_ * beta[..., None], v_ * beta[..., None]
        attn = (-((k_beta @ k_.transpose(-1, -2)) * decay) * anc[None]).clone()
        for j in range(1, T):
            row, sub = attn[:, j, :j].clone(), attn[:, :j, :j].clone()
            attn[:, j, :j] = row + (row[..., None] * sub).sum(-2)
        attn = attn + torch.eye(T, device=dev)[None]
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * cumg.exp()[..., None])
        v_new = value - k_cumdecay @ S0
        attn_out = ((q_ @ k_.transpose(-1, -2)) * decay) * anc_self[None]
        out = (q_ * cumg.exp()[..., None]) @ S0 + attn_out @ v_new
        core = out.transpose(0, 1).reshape(-1, dv)
        zp = z_all[0].float().reshape(T, Hv, dv).reshape(-1, dv)
        return la.norm(core, zp).reshape(1, T, -1)

    def ah(self, cl: Any, i: int, inp: Any, path: NodePath) -> None:
        la: Any = (self.host[i] if i in self.host else self.resident[i]).linear_attn
        mixed_all, z_all, a_all, b_all = inp
        step_fn = Native.delta_step
        conv_state, recurrent_state = self._lin(cl)
        if step_fn is not None:
            if not (conv_state.is_contiguous() and recurrent_state.is_contiguous()):
                conv_state, recurrent_state = conv_state.contiguous(), recurrent_state.contiguous()
                self._lin_set(cl, conv_state, recurrent_state)
            out = torch.empty(la.num_v_heads * la.head_v_dim, dtype=torch.float32)
            for p in path:
                step_fn(
                    mixed_all[0, p].contiguous().clone(),
                    conv_state,
                    la._k_conv_w,
                    la._k_conv_b,
                    z_all[0, p].contiguous(),
                    a_all[0, p].contiguous(),
                    b_all[0, p].contiguous(),
                    la._k_a_log,
                    la._k_dt_bias,
                    recurrent_state,
                    la.num_k_heads,
                    la.num_v_heads,
                    la.head_k_dim,
                    la.head_v_dim,
                    la._k_norm_w,
                    la._k_eps,
                    out,
                    la._k_gate,
                )
            return
        mod = self.fam.mod
        for p in path:
            mixed = mod.causal_conv1d_update(
                mixed_all[:, p : p + 1].transpose(1, 2),
                conv_state,
                la.conv1d.weight.squeeze(1),
                la.conv1d.bias,
                la.activation,
            ).transpose(1, 2)
            query, key, value = torch.split(mixed, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
            query = query.reshape(1, 1, -1, la.head_k_dim)
            key = key.reshape(1, 1, -1, la.head_k_dim)
            value = value.reshape(1, 1, -1, la.head_v_dim)
            beta = b_all[:, p : p + 1].sigmoid()
            g = -la.A_log.float().exp() * F.softplus(a_all[:, p : p + 1].float() + la.dt_bias)
            if la.num_v_heads // la.num_k_heads > 1:
                query = query.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
                key = key.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
            _, last_rec = mod.torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent_state.copy_(last_rec)

    def ai(self, layer: Any, i: int, h: torch.Tensor, pe: Any, text_pos: Any, cache: Any) -> torch.Tensor:
        """One host layer over T positions after a prefix: a chain, or a tree of nodes when speculation is on."""
        T = h.shape[1]
        lt = self.layer_types[i]
        cl = cache.layers[i]
        ap: Parents | None = getattr(self, "ap", None)
        parents: Parents = ap if ap is not None else range(-1, T - 1)  # no tree: the chain's own map
        tree = any(parents[j] != j - 1 for j in range(T))
        spec_on = bool(getattr(self, "aq", False))
        x = layer.input_layernorm(h)
        if lt != LayerKind.LINEAR:
            at = layer.self_attn
            return self._ai_attention(layer, i, h, x, pe, cache, cl, T, parents, tree, at, at.head_dim)
        la: Any = layer.linear_attn
        core = self._delta_nodes(
            layer, i, cl, la.in_proj_qkv(x), la.in_proj_z(x), la.in_proj_a(x), la.in_proj_b(x), T, parents, spec_on
        )
        return self._ai_mlp(layer, h + la.out_proj(core))

    def _delta_nodes(
        self,
        layer: Any,
        i: int,
        cl: Any,
        mixed_all: torch.Tensor,
        z_all: torch.Tensor,
        a_all: torch.Tensor,
        b_all: torch.Tensor,
        T: int,
        parents: Parents,
        spec_on: bool,
    ) -> torch.Tensor:
        """The gated DeltaNet mixer over T positions (a chain, or a tree with per-node state checkpoints) on the
        CPU kernel, from the four projections to the gated-normed core [1, T, Hv*dv]."""
        la: Any = layer.linear_attn
        tree = any(parents[j] != j - 1 for j in range(T))
        outs = []
        ckpts: list[Any] = []
        step_fn = Native.delta_step
        if step_fn is not None:
            # the kernel works in float32: a bf16 prefill (Qwen3.5's norm keeps the input dtype) leaves bf16
            # states and projections behind, so they are widened here, once
            mixed_all, z_all, a_all, b_all = (t.float() for t in (mixed_all, z_all, a_all, b_all))
            c0, r0 = self._lin(cl)
            if c0.dtype != torch.float32 or r0.dtype != torch.float32:
                self._lin_set(cl, c0.float().contiguous(), r0.float().contiguous())
        if step_fn is not None and not hasattr(la, "_k_conv_w"):
            la._k_conv_w = la.conv1d.weight.squeeze(1).detach().float().contiguous()
            la._k_conv_b = None if la.conv1d.bias is None else la.conv1d.bias.detach().float().contiguous()
            la._k_a_log = la.A_log.detach().float().contiguous()
            la._k_dt_bias = la.dt_bias.detach().float().contiguous()
            la._k_norm_w = la.norm.weight.detach().float().contiguous()
            la._k_eps = float(getattr(la.norm, "variance_epsilon", getattr(la.norm, "eps", 1e-6)))
            # the gated norm's activation: Qwen3.5's silu, Qwen4's sigmoid (its norm carries the name)
            la._k_gate = 1 if getattr(la.norm, "activation", "silu") == "sigmoid" else 0
        pre: Any = self._lin(cl)
        if step_fn is not None and not (pre[0].is_contiguous() and pre[1].is_contiguous()):
            pre = (pre[0].contiguous(), pre[1].contiguous())
            self._lin_set(cl, *pre)
        chunk_read = spec_on and getattr(self, "tree_read", "step") == "chunk"
        cbuf: Any
        rbuf: Any
        cbuf, rbuf = self.af(i, T, cl) if (spec_on and not chunk_read) else (None, None)
        if chunk_read:
            outs.append(self.ag(la, cl, mixed_all, z_all, a_all, b_all, parents, T))
            self.an[i] = (mixed_all, z_all, a_all, b_all)
        for p in range(T) if not chunk_read else range(0):
            if spec_on:
                par = parents[p]
                src_c, src_r = pre if par < 0 else ckpts[par]
                conv_state, recurrent_state = cbuf[p], rbuf[p]
                conv_state.copy_(src_c)
                recurrent_state.copy_(src_r)
            else:
                conv_state, recurrent_state = pre
            if step_fn is not None:
                mixed = mixed_all[0, p].contiguous().clone()
                out = torch.empty(la.num_v_heads * la.head_v_dim, dtype=torch.float32)
                step_fn(
                    mixed,
                    conv_state,
                    la._k_conv_w,
                    la._k_conv_b,
                    z_all[0, p].contiguous(),
                    a_all[0, p].contiguous(),
                    b_all[0, p].contiguous(),
                    la._k_a_log,
                    la._k_dt_bias,
                    recurrent_state,
                    la.num_k_heads,
                    la.num_v_heads,
                    la.head_k_dim,
                    la.head_v_dim,
                    la._k_norm_w,
                    la._k_eps,
                    out,
                    la._k_gate,
                )
                outs.append(out.view(1, 1, -1))
                if spec_on:
                    ckpts.append((conv_state, recurrent_state))
                continue
            mixed = mixed_all[:, p : p + 1].transpose(1, 2)
            mixed = self.fam.mod.causal_conv1d_update(
                mixed, conv_state, la.conv1d.weight.squeeze(1), la.conv1d.bias, la.activation
            )
            mixed = mixed.transpose(1, 2)
            query, key, value = torch.split(mixed, [la.key_dim, la.key_dim, la.value_dim], dim=-1)
            query = query.reshape(1, 1, -1, la.head_k_dim)
            key = key.reshape(1, 1, -1, la.head_k_dim)
            value = value.reshape(1, 1, -1, la.head_v_dim)
            beta = b_all[:, p : p + 1].sigmoid()
            g = -la.A_log.float().exp() * F.softplus(a_all[:, p : p + 1].float() + la.dt_bias)
            if la.num_v_heads // la.num_k_heads > 1:
                query = query.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
                key = key.repeat_interleave(la.num_v_heads // la.num_k_heads, dim=2)
            core, last_rec = self.fam.mod.torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent_state.copy_(last_rec)
            if spec_on:
                ckpts.append((conv_state, recurrent_state))
            core = core.reshape(-1, la.head_v_dim)
            zp = z_all[:, p : p + 1].reshape(1, 1, -1, la.head_v_dim).reshape(-1, la.head_v_dim)
            core = la.norm(core, zp).reshape(1, 1, -1)
            outs.append(core)
        if spec_on and not chunk_read:
            self.al[i] = ckpts
            if tree:
                self.am[i] = pre
        return torch.cat(outs, dim=1)

    def _ai_attention(
        self,
        layer: Any,
        i: int,
        h: torch.Tensor,
        x: torch.Tensor,
        pe: Any,
        cache: Any,
        cl: Any,
        T: int,
        parents: Parents,
        tree: bool,
        at: Any,
        hd: int,
    ) -> torch.Tensor:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        apply_rotary_pos_emb = self._rope_fn()
        eager_attention_forward = self.fam.mod.eager_attention_forward
        residual = h
        outs = []
        if hasattr(at, "qkv_proj"):  # q, k and v as one projection (Phi-3's layout)
            qkv = at.qkv_proj(x)
            nq = self.cfg.num_attention_heads * hd
            nk = at.num_key_value_heads * hd
            q_all = qkv[..., :nq].view(1, T, -1, hd)
            k_all = qkv[..., nq : nq + nk].view(1, T, -1, hd)
            v_all = qkv[..., nq + nk :].view(1, T, -1, hd)
            gate_all = None
        elif self.fam.attn_gate:
            qg = at.q_proj(x).view(1, T, -1, hd * 2)
            q_all, gate_all = torch.chunk(qg, 2, dim=-1)
            gate_all = gate_all.reshape(1, T, -1)
            q_all = at.q_norm(q_all.reshape(1, T, -1, hd))
            k_all = at.k_norm(at.k_proj(x).view(1, T, -1, hd))
            v_all = at.v_proj(x).view(1, T, -1, hd)
        else:
            q_all = at.q_norm(at.q_proj(x).view(1, T, -1, hd))
            k_all = at.k_norm(at.k_proj(x).view(1, T, -1, hd))
            v_all = at.v_proj(x).view(1, T, -1, hd)
            gate_all = None
        # a paged layer's rows lie in the engine's pool wherever their pages do: appended through the layer, read
        # through the conversation's row map (btb/engine/paged.py)
        paged = bool(getattr(cl, "paged", False))
        if paged:
            base = cl.get_seq_length()
        else:
            base = cl.keys.shape[-2] if getattr(cl, "keys", None) is not None else 0
        win = layer_window(self.cfg, self.layer_types[i])
        attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(self.cfg._attn_implementation, eager_attention_forward)
        q_rot, k_rot = apply_rotary_pos_emb(q_all.transpose(1, 2), k_all.transpose(1, 2), pe[0], pe[1])
        if paged:
            k_full, v_full = cl.append(k_rot, v_all.transpose(1, 2))
        else:
            k_full, v_full = cache.update(k_rot, v_all.transpose(1, 2), i)
        native = (
            k_full.shape[0] == 1
            and k_full.device.type == "cpu"
            and k_full.dtype in (torch.bfloat16, torch.float32)
            and v_full.dtype == k_full.dtype
            and k_full.stride(-1) == 1
            and k_full.stride(-2) == hd
            and v_full.stride(-1) == 1
            and v_full.stride(-2) == hd
        )
        # a paged layer's one-row step too: its rows are the table's, which `attn_spans` reads with `attn_decode`'s bits
        kernel = native and T == 1 and not tree and Native.attn_decode is not None and not paged
        # a pass's rows past the first (a verify's chain or tree, a prompt's later chunk) through the step's own
        # arithmetic: each row over the cache rows its committed step will read, in the order it reads them, row for
        # row `attn_decode` bit for bit - a chain's rows each a span of one map (`attn_spans`: the cache's rows, a
        # paged layer's through its table), a tree's each a list of its own (`attn_nodes`). Through sdpa a row at a
        # time, a verify rounded a bf16 tie apart from the step - Qwen3-0.6B with eleven host layers took ':' where
        # its greedy step took '.', speculation then no longer the greedy answer
        spans = native and not kernel and not tree and Native.attn_spans is not None
        nodes = native and not kernel and not spans and Native.attn_nodes is not None
        if paged and not (spans or nodes):
            from .paged import PagedError

            raise PagedError(
                f"layer {i}: the paged rows are read by the native attention alone, and it cannot run here"
            )
        if kernel:
            qf = q_rot[0, :, 0].float().contiguous()
            out = torch.empty(qf.shape, dtype=torch.float32)
            first = max(0, int(k_full.shape[-2]) - win) if win else 0  # a sliding layer's last rows alone
            Native.attn_decode(qf, k_full[0][:, first:], v_full[0][:, first:], at.scaling, out)
            a1 = out.view(1, 1, -1).to(h.dtype)
            outs.append(a1 if gate_all is None else a1 * torch.sigmoid(gate_all[:, 0:1]))
        elif spans or nodes:
            qf = q_rot[0].transpose(0, 1).float().contiguous()  # [T, hq, d]
            out = torch.empty(qf.shape, dtype=torch.float32)
            if spans:
                amap, starts, ends = self._span_lists(cache if paged else None, base, T, win)
                Native.attn_spans(qf, k_full[0], v_full[0], amap, starts, ends, float(at.scaling), out)
            else:
                offs, idx = self._node_lists(cache if paged else None, base, T, parents, win)
                Native.attn_nodes(qf, k_full[0], v_full[0], offs, idx, float(at.scaling), out)
            if paged and i == max(j for j, lt in enumerate(self.layer_types) if lt != LayerKind.LINEAR):
                cache._lists = None  # the pass's lists go with its last attention layer: an idle cache holds none
            if gate_all is None:
                outs.append(out.view(1, T, -1).to(h.dtype))  # a cast: each element its own, whatever its neighbours
            else:
                for p in range(T):  # a row at a time, as each step gates its one
                    a1 = out[p].view(1, 1, -1).to(h.dtype)
                    outs.append(a1 * torch.sigmoid(gate_all[:, p : p + 1]))
        for p in range(T) if not (kernel or spans or nodes) else range(0):
            qs = q_rot[:, :, p : p + 1]
            ks = k_full[..., : base + p + 1, :]
            vs = v_full[..., : base + p + 1, :]
            rows = node_mask(base, p, parents, win)
            mask_p = None if rows is None else rows.view(1, 1, 1, -1).to(h.device)
            attn, _ = attn_fn(at, qs, ks, vs, mask_p, dropout=0.0, scaling=at.scaling)
            a1 = attn.reshape(1, 1, -1).contiguous()
            outs.append(a1 if gate_all is None else a1 * torch.sigmoid(gate_all[:, p : p + 1]))
        mix = at.o_proj(torch.cat(outs, dim=1))
        if self.fam.sandwich:
            mix = layer.post_attention_layernorm(mix)  # the sandwich block norms the delta before its add
        return self._ai_mlp(layer, residual + mix)

    @staticmethod
    def _pass_lists(paged: Any) -> dict[Any, Any]:
        """a paged cache's lists for the pass (`_span_lists`, `_node_lists`): those of its row map's present version
        alone, which every change to the map moves - a window's and the whole prefix's (Gemma 3's layers alternate)
        side by side, made once for every layer that reads them alike"""
        ver, memo = paged.__dict__.get("_lists") or (None, {})
        if ver != paged.table.version:
            memo = {}
            paged._lists = (paged.table.version, memo)
        return cast("dict[Any, Any]", memo)

    @staticmethod
    def _span_lists(paged: Any, base: int, T: int, win: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """a chain's T rows after `base` as `attn_spans` takes them: (rows, starts, ends) int32, row p over the rows
        before it and itself - the last `win` of them under a window - each a span of one map: the cache's rows in
        order, a paged cache's (`paged`) through its table"""
        memo = _CudaMixin._pass_lists(paged) if paged is not None else None
        key = ("spans", base, T, win)
        got = memo.get(key) if memo is not None else None
        if got is not None:
            return cast("tuple[torch.Tensor, torch.Tensor, torch.Tensor]", got)
        ends = torch.arange(base + 1, base + T + 1, dtype=torch.int32)
        starts = (ends - win).clamp_(min=0) if win else torch.zeros(T, dtype=torch.int32)
        if paged is None:
            return torch.arange(base + T, dtype=torch.int32), starts, ends
        rows = paged.table.rows()[: base + T].to(torch.int32)
        assert memo is not None
        memo[key] = (rows, starts, ends)
        return rows, starts, ends

    @staticmethod
    def _node_lists(paged: Any, base: int, T: int, parents: Parents, win: int) -> tuple[torch.Tensor, torch.Tensor]:
        """each of a pass's T rows' cache rows, in the order its committed step reads them, as `attn_nodes` takes
        them: (offs [T + 1], idx) int32. `paged` (a paged cache): the rows are its pool's, through its row map, made
        once a pass (`_pass_lists`)"""
        memo = _CudaMixin._pass_lists(paged) if paged is not None else None
        key = (base, T, tuple(parents), win)
        got = memo.get(key) if memo is not None else None
        if got is not None:
            return cast("tuple[torch.Tensor, torch.Tensor]", got)
        lists = []
        for p in range(T):
            rows = node_mask(base, p, parents, win)
            lists.append(torch.arange(base + p + 1) if rows is None else rows.nonzero()[:, 0])
        flat = torch.cat(lists)
        if paged is not None:
            flat = paged.table.rows()[flat]
        idx = flat.to(torch.int32).contiguous()
        offs = torch.zeros(T + 1, dtype=torch.int32)
        offs[1:] = torch.tensor([len(x) for x in lists]).cumsum(0).to(torch.int32)
        if memo is not None:
            memo[key] = (offs, idx)
        return offs, idx

    def _ai_mlp(self, layer: Any, h: torch.Tensor) -> torch.Tensor:
        residual = h
        sandwich = self.fam.sandwich
        x2 = layer.pre_feedforward_layernorm(h) if sandwich else layer.post_attention_layernorm(h)
        mlp = layer.mlp
        if hasattr(mlp, "gate_up_proj"):  # gate and up as one projection (Phi-3's layout)
            gate, up = mlp.gate_up_proj(x2).chunk(2, dim=-1)
            fn = mlp.activation_fn
        else:
            gate, up = mlp.gate_proj(x2), mlp.up_proj(x2)
            fn = mlp.act_fn
        out = mlp.down_proj(fn(gate) * up)
        return residual + (layer.post_feedforward_layernorm(out) if sandwich else out)
