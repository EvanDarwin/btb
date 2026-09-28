# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The host tier's modules: a linear over a checkpoint view (native kernels or the CPU stream for its matmul),
the MoE router and experts, and the n-gram proposer's row table."""

from __future__ import annotations

import time
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import torch

from .. import fp8
from .. import mlx as mlxdev
from ..fp8 import F8Weight
from ..kinds import PassTag
from ..mxfp4 import BLOCK, MxGateUp, MxWeight
from ..mxfp4_torch import dequant_blocks
from ..options import Device
from .native import Native
from .pack import unpack_bf16

if TYPE_CHECKING:
    from .experts import StoreCall


def copy_bytes(dst: torch.Tensor, t: torch.Tensor) -> None:
    """`t`'s bytes into the byte buffer `dst`, from any thread. Under inference mode, which takes the write
    whether `dst` was made in it (a bind or a shed mid-decode) or not: the mode is per thread, and a reader
    thread outside it refuses to write an inference tensor."""
    with torch.inference_mode():
        dst.copy_(t.reshape(-1).view(torch.uint8))


def bf16_in_place(buf: torch.Tensor, dt: torch.dtype) -> None:
    """the byte buffer `buf`, holding `dt` values, rewritten as those values in bf16 from its start (the first
    half of it for float32), from any thread as `copy_bytes` is. One cast through a temporary: chunked steps
    measured slower at every size (torch's per-op cost and thread fan-out outweigh the cache reuse)."""
    with torch.inference_mode():
        src = buf.view(dt)
        buf[: src.numel() * 2].view(torch.bfloat16).copy_(src.to(torch.bfloat16))


class _HostLinear(torch.nn.Module):
    _cpu_shared: Any
    bias: torch.Tensor | None
    cpu_gemm: Any
    f8: F8Weight | None
    key: str | None
    mx: Any
    packed: tuple[Any, ...] | None
    weight: torch.Tensor

    def __init__(self, weight: torch.Tensor, key: str | None = None, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        # gpt-oss is the first family whose projections carry one (already widened to float32 by
        # `_make_host_layer`'s small-tensor rule); every other family passes None here
        self.bias = None if bias is None else torch.nn.Parameter(bias, requires_grad=False)
        self.key = key
        self.packed = None
        self.mx = None
        # an FP8 checkpoint's matrix as stored (families.py binds it); `weight` is then its shape and no bytes
        self.f8 = None
        # on a Mac's CPU tier: the same bytes as an MLX bf16 array, for the prefill's GEMM on MLX's CPU
        # stream (`bind_cpu_gemm`); the one-row step keeps the native kernel
        self.cpu_gemm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._matmul(x)
        return y if self.bias is None else y + self.bias.to(y.dtype)

    def _matmul(self, x: torch.Tensor) -> torch.Tensor:
        if self.mx is not None:
            return Native.mlx.linear(x, self.mx)
        rows, cols = self.weight.shape
        if self.f8 is not None:
            # the FP8 kernel widens a column tile once and reuses it over the batch tile, so it is the path at
            # every batch size, as the MXFP4 one is
            shp = x.shape
            x2 = x.reshape(-1, cols).float().contiguous()
            y = torch.empty(x2.shape[0], rows, dtype=torch.float32)
            Native.gemv_fp8(self.f8, x2, y)
            return y.view(*shp[:-1], rows).to(x.dtype)
        if x.dtype == torch.float32 and (Native.gemv is not None):
            shp = x.shape
            x2 = x.reshape(-1, cols).contiguous()
            if self.cpu_gemm is not None and x2.shape[0] >= Native.cpu_gemm_rows:
                return Native._cpu_gemm(x2, self.cpu_gemm).view(*shp[:-1], rows)
            packed_only = self.packed is not None and Native.gemv_p12 is None  # a library without the 12-bit gemv
            if x2.shape[0] >= Native.gemm_rows or packed_only:
                if self.packed is not None:
                    lo, hi4, tbl, esc_idx, esc_val, n_esc = self.packed
                    w = unpack_bf16(
                        lo, hi4, tbl, rows * cols, (rows, cols), esc_idx if n_esc else None, esc_val if n_esc else None
                    )
                else:
                    w = self.weight
                return torch.nn.functional.linear(x2, w.float()).view(*shp[:-1], rows)
            y = torch.empty(x2.shape[0], rows, dtype=torch.float32)
            if self.packed is not None:
                Native.gemv_p12(*self.packed, rows, cols, x2, y)
            else:
                Native.gemv(self.weight, x2, y)
            return y.view(*shp[:-1], rows)
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


class _Router(torch.nn.Module):
    """gpt-oss's top-k router on a host layer: the logits through `_HostLinear` in float32 (a row's logits do not
    depend on its neighbours), then transformers' top-k and softmax. Returns (logits, scores, indices)."""

    lin: _HostLinear
    top_k: int

    def __init__(self, lin: _HostLinear, top_k: int) -> None:
        super().__init__()
        self.lin = lin
        self.top_k = int(top_k)

    def forward(self, hidden_states: torch.Tensor) -> Any:
        logits = self.lin(hidden_states.float())
        top, idx = torch.topk(logits, self.top_k, dim=-1)
        scores = torch.softmax(top, dim=-1)
        return logits, scores.to(hidden_states.dtype), idx


def _cast_floats(x: object, dtype: torch.dtype) -> object:
    """every floating tensor in `x` (a tensor, or a tuple or list of them and anything else) as `dtype`"""
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.is_floating_point() else x
    if isinstance(x, (tuple, list)):
        return type(x)(_cast_floats(v, dtype) for v in x)
    return x


def compute_fp32(module: torch.nn.Module) -> None:
    """`module` computes in float32 and hands its result back in the dtype of its first floating input, the way
    transformers runs a router in float32: the host layer widened its matrix, which a bf16 activation cannot
    meet in a conv or a linear. On a float32 pass both casts are the identity."""
    inner = module.forward

    def forward(*args: object, **kw: object) -> object:
        floats = [v for v in (*args, *kw.values()) if isinstance(v, torch.Tensor) and v.is_floating_point()]
        f32 = torch.float32
        out = inner(*(_cast_floats(v, f32) for v in args), **{k: _cast_floats(v, f32) for k, v in kw.items()})
        return _cast_floats(out, floats[0].dtype) if floats else out

    object.__setattr__(module, "forward", forward)


def widen_scratch(store: Any, n: int, inter: int, hidden: int) -> int:
    """the card memory a grouped call's widening holds at its peak, `n` experts at a time (`_card_grouped`), each
    expert gate_up [2 * `inter`, `hidden`] and down [`hidden`, `inter`], gate_up's, then down's beside gate_up's
    bf16: MXFP4's bf16 alone where the card's kernel widens it from the depot's bytes (`mx4_widen`), 2 bytes a
    weight, else torch's gathered bytes, their int32 lookup index, the value pairs and the bf16 out, 4.5; FP8's bytes
    widened to float32, scaled and made bf16, 11 bytes a weight; 0 for bf16 experts, multiplied as they sit"""
    if store is None or getattr(store, "ggml", False) or not (store.mx or store.f8):
        return 0
    per = (2.0 if Native.card_kernels() is not None else 4.5) if store.mx else 11.0
    gu = dn = inter * hidden
    gu *= 2
    return int(n * max(per * gu, 2 * gu + per * dn))


def stored_parts(gu: Any, dn: Any) -> tuple[torch.Tensor, ...] | None:
    """an expert's bytes as the store holds them, for the card: bf16 (gate_up, down), MXFP4 in the checkpoint's
    layout (gate_up's blocks and scales, down's blocks and scales), or FP8 (gate_up's e4m3 bytes and scale grid,
    down's); None for a form the card path does not take (ggml's MXFP4) or an expert already on the card"""
    if isinstance(gu, torch.Tensor) and isinstance(dn, torch.Tensor):
        parts: tuple[torch.Tensor, ...] = (gu, dn)
    elif (
        isinstance(gu, MxWeight)
        and isinstance(dn, MxWeight)
        and not gu.ggml
        and not dn.ggml
        and gu.scales is not None
        and dn.scales is not None
    ):
        parts = (gu.blocks, gu.scales, dn.blocks, dn.scales)
    elif isinstance(gu, F8Weight) and isinstance(dn, F8Weight):
        parts = (gu.w, gu.scales, dn.w, dn.scales)
    else:
        return None
    return parts if all(p.device.type == "cpu" for p in parts) else None


def group_picks(top_k_index: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    """A call's (pick, row) pairs grouped by expert in one sort on the picks' own device: (top_k_pos, token_idx)
    of every pair sorted by expert, and each expert's offset and count into them. An expert's pairs come pick-major,
    then row - the order `torch.where(one_hot(top_k_index).permute(2, 1, 0)[e])` lists them - so a product over its
    slice is the per-expert lookup's, bit for bit. The counts are the one read back to the host."""
    T = int(top_k_index.shape[0])
    flat = top_k_index.t().reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=int(num_experts)).tolist()
    offs, a = [], 0
    for n in counts:
        offs.append(a)
        a += int(n)
    return order // T, order % T, offs, [int(n) for n in counts]


class _Experts(torch.nn.Module):
    # MXFP4 or FP8 experts widened on the card at a time in a grouped call: gpt-oss-120b's are 50 MB a piece widened
    WIDEN_BATCH = 16
    sm: Any
    _mx_bias: Any
    act_fn: Any
    alpha: float
    base: str
    down: Any
    down_proj_bias: Any
    gate: Any
    gate_up: Any
    gate_up_proj_bias: Any
    layer: int
    limit: float
    mx: bool
    f8: bool
    num_experts: int

    def __init__(
        self,
        sm: Any,
        base: str,
        num_experts: int,
        act_fn: Any,
        layer: int = -1,
        mx: bool = False,
        f8: bool = False,
        gate: Any = None,
        biases: bool = False,
        alpha: float = 1.702,
        limit: float = 7.0,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "sm", sm)
        self.base = base
        self.num_experts = int(num_experts)
        self.act_fn = act_fn
        self.layer = int(layer)
        self.gate_up = None
        self.down = None
        # MXFP4 experts (gpt-oss's, or a GGUF's stored so) stay MXFP4 into the matvec; gpt-oss's two per-expert
        # biases ride with the layer and its gate is its own clamped GLU over interleaved halves
        self.mx = bool(mx)
        self.ggml = self.mx and getattr(sm, "gguf", None) is not None  # a GGUF's experts, ggml's layout as stored
        self.f8 = bool(f8)  # an FP8 checkpoint's experts, e4m3 bytes and scale grids into the FP8 matvec
        self.gate = gate
        self.alpha = float(alpha)
        self.limit = float(limit)
        self.biased = bool(biases)
        if biases:
            self.gate_up_proj_bias = torch.nn.Parameter(torch.zeros(0), requires_grad=False)
            self.down_proj_bias = torch.nn.Parameter(torch.zeros(0), requires_grad=False)

    def _tables(self) -> Any:
        if self.gate_up is None:
            if self.ggml:
                # the file's tensors as they are, one MxWeight per expert (the store does this from its slots)
                gg, E, i = self.sm.gguf, self.num_experts, self.layer

                def mats(k: str) -> list[MxWeight]:
                    t = gg.tensors[f"blk.{i}.ffn_{k}_exps.weight"]
                    _e, rows, cols = (int(x) for x in reversed(list(t.shape)))
                    raw = gg.raw(t.name).reshape(E, -1)
                    return [MxWeight.from_ggml(raw[e], rows, cols) for e in range(E)]

                self.gate_up = [MxGateUp(g, u) for g, u in zip(mats("gate"), mats("up"), strict=True)]
                self.down = mats("down")
            elif self.mx:
                parts = {}
                for n in ("gate_up_proj", "down_proj"):
                    b = self.sm._get(self.base + n + "_blocks")
                    s = self.sm._get(self.base + n + "_scales")
                    rows, g = int(b.shape[1]), int(b.shape[2])
                    parts[n] = [
                        MxWeight(b[e].reshape(-1), s[e].reshape(-1), rows, g * BLOCK) for e in range(int(b.shape[0]))
                    ]
                self.gate_up, self.down = parts["gate_up_proj"], parts["down_proj"]
            elif self.f8:
                self.gate_up = self.sm._f8_weights(self.base + "gate_up_proj")
                self.down = self.sm._f8_weights(self.base + "down_proj")
            else:
                self.gate_up = self.sm._get(self.base + "gate_up_proj")
                self.down = self.sm._get(self.base + "down_proj")
        return self.gate_up, self.down

    def _act(self, gate_up: torch.Tensor, rows: Any = None) -> torch.Tensor:
        """The expert's middle: `gate_up` [n, 2 * intermediate] to [n, intermediate]; `rows` names the expert behind
        each row when there is a bias to add."""
        if self.biased:
            gate_up = gate_up + self._bias(self.gate_up_proj_bias, rows, gate_up)
        return self.gate(gate_up, self.alpha, self.limit) if self.gate is not None else self._silu_gate(gate_up)

    def _join(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """a GGUF's separate gate and up outputs in the checkpoint's [2I] order: gpt-oss's GLU interleaves its
        halves, the other families concatenate them"""
        return MxGateUp.interleave(gate, up) if self.gate is not None else torch.cat([gate, up], dim=-1)

    @staticmethod
    def _bias(param: Any, rows: Any, like: torch.Tensor) -> torch.Tensor:
        # the layer may sit on the card while the experts are multiplied on the CPU, and a resident
        # layer keeps its biases in the checkpoint's bf16 where a host layer widens them
        return param[rows].to(like.device, like.dtype)

    def _silu_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        return self.act_fn(gate) * up

    @staticmethod
    def gpt_oss_gate(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
        """`GptOssExperts._apply_gate`: the halves are interleaved, the gate is clamped above and the up
        half on both sides, and the result is `(up + 1) * gate * sigmoid(alpha * gate)`."""
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=limit)
        up = up.clamp(min=-limit, max=limit)
        return (up + 1) * (gate * torch.sigmoid(gate * alpha))

    def _group(self) -> Any:
        """the grouped matvec for this layer's storage, None without the native library (`_linear` widens then)"""
        if self.ggml:
            return Native.gemv_mx4_ggml_group
        if self.f8:
            return Native.gemv_fp8_group
        return Native.gemv_mx4_group if self.mx else Native.gemv_group

    def _gate_up_rows(self, gemv_group: Any, ws: Any, xf: torch.Tensor, pos: Any, gu_buf: torch.Tensor) -> None:
        """`gu_buf[i] = gate_up_e x` for the experts at `pos`: one matvec each, or a GGUF's gate and up matvecs
        joined into the checkpoint's [2I] order"""
        if not self.ggml:
            gemv_group([w[0] for w in ws], [xf] * len(pos), [gu_buf[i : i + 1] for i in pos])
            return
        n, half = len(pos), gu_buf.shape[1] // 2
        tg, tu = torch.empty(n, half, dtype=torch.float32), torch.empty(n, half, dtype=torch.float32)
        gemv_group(
            [w[0].gate for w in ws] + [w[0].up for w in ws],
            [xf] * (2 * n),
            [tg[j : j + 1] for j in range(n)] + [tu[j : j + 1] for j in range(n)],
        )
        for j, i in enumerate(pos):
            gu_buf[i] = self._join(tg[j], tu[j])

    def _one_row(
        self,
        x: torch.Tensor,
        w_top: torch.Tensor,
        top_k_index: torch.Tensor,
        hit: Any,
        views: Any,
        pending: Any,
        store: Any,
        final: torch.Tensor,
        x_card: torch.Tensor | None = None,
    ) -> None:
        dt = x.dtype
        if self.mx:
            self.sm._tag(PassTag.EXPERT_MXFP4_ASSTORED)  # `_group()` is the mx4 matvec over the stored blocks
        if self.f8:
            self.sm._tag(PassTag.EXPERT_FP8_ASSTORED)  # `_group()` is the FP8 matvec over the stored bytes
        k = len(hit)
        xf = x.float().contiguous()
        slot = {e: top_k_index[0].tolist().index(e) for e in hit}
        weights = w_top[0]
        first = next(iter(views.values())) if views else store._views(pending[0][2])
        gemv_group = self._group()
        gu_buf = torch.empty(k, first[0].shape[0], dtype=torch.float32)
        dn_buf = torch.empty(k, first[1].shape[0], dtype=torch.float32)
        out = torch.empty(k, first[1].shape[0], dtype=dt)
        # the experts seated on the card: multiplied there in float32 over the stored bf16, the row brought back
        card = [
            i
            for i, e in enumerate(hit)
            if e in views and isinstance(views[e][0], torch.Tensor) and views[e][0].device.type != "cpu"
        ]
        if card and x_card is not None:
            with torch.no_grad():
                xc = x_card.reshape(1, -1).float()
                rows = []
                for i in card:
                    w_gu, w_dn = views[hit[i]]
                    hmid = self._act(torch.nn.functional.linear(xc, w_gu.float()), None)
                    rows.append(torch.nn.functional.linear(hmid.float(), w_dn.float()))
                back = torch.cat(rows, 0).cpu()
            for r, i in enumerate(card):
                out[i] = back[r].to(dt) * weights[slot[hit[i]]]

        def stage(pos: Any, ws: Any) -> None:
            if not pos:
                return
            self._gate_up_rows(gemv_group, ws, xf, pos, gu_buf)
            rows = torch.tensor([hit[i] for i in pos]) if self.biased else None
            h = self._act(gu_buf[pos].to(dt), rows).float().contiguous()
            gemv_group([w[1] for w in ws], [h[j : j + 1] for j in range(len(pos))], [dn_buf[i : i + 1] for i in pos])
            for _j, i in enumerate(pos):
                o = dn_buf[i] + self._bias(self.down_proj_bias, hit[i], dn_buf) if self.biased else dn_buf[i]
                out[i] = o.to(dt) * weights[slot[hit[i]]]

        ready = [i for i, e in enumerate(hit) if e in views and i not in card]
        stage(ready, [views[hit[i]] for i in ready])
        # the misses as they land: what has arrived is computed while the rest is still on its way. The bf16
        # group kernel gives each task the same bits whatever shares its group; the MXFP4 one does not, so its
        # late experts go as the one group they always were, or the timing of the drive would reach the bits
        if self.mx:
            late = [
                (hit.index(e), store._views(s))
                for batch in (store.landed(pending) if pending else ())
                for e, _f, s in batch
            ]
            stage([i for i, _ in late], [w for _, w in late])
        else:
            for batch in store.landed(pending) if pending else ():
                late = [(hit.index(e), store._views(s)) for e, _f, s in batch]
                stage([i for i, _ in late], [w for _, w in late])
        for i in range(k):
            final[0] += out[i]

    def _mlx_forward(
        self,
        x: torch.Tensor,
        w_top: torch.Tensor,
        expert_mask: Any,
        hit: Any,
        views: Any,
        pending: Any,
        store: Any,
        final: torch.Tensor,
    ) -> None:
        # every active expert's two matmuls in one MLX graph over the store's slots (or copies of the
        # checkpoint's tables when there is no store), summed in expert order; one eval per layer
        be = self.sm.mlx
        tables: Any = self._tables() if store is None else None

        def rows(e: int) -> Any:
            top_k_pos, token_idx = torch.where(expert_mask[e])
            return token_idx.tolist(), w_top[token_idx, top_k_pos].float().tolist()

        if self.mx:
            # MXFP4 experts through the GPU's matvec, each expert its own graph queued as its bytes land, the outputs
            # summed in expert order at the end whichever were ready first
            self.sm._tag(PassTag.EXPERT_MXFP4_ASSTORED)  # the store's blocks as stored, never widened
            bg, bd = self._mx_biases() if self.biased else (None, None)
            act = None if self.gate is not None else self.sm._mlx_act()  # gpt-oss's clamped GLU, else the family's
            shp = store.mx_shapes()
            xm = mlxdev.to_mx(x)
            parts = {}

            def part(e: int, pair: Any) -> None:
                ti, wts = rows(e)
                parts[e] = be.expert_mx(
                    xm, ti, wts, pair, e, shp[0], shp[1], bg, bd, self.alpha, self.limit, ggml=store.ggml, act=act
                )

            for e in hit:
                if e in views:
                    part(e, store._views_mx(store.last_slots[e]))
            for batch in store.landed(pending):
                for e, _f, s in batch:
                    part(e, store._views_mx(s))
            out = be.experts_mx_sum(xm, [parts[e] for e in sorted(parts)])
            final += out.to(final.dtype)
            return
        hits = []
        for e in hit:
            if e in views:
                hits.append(
                    (
                        e,
                        *rows(e),
                        store._views_mx(store.last_slots[e])
                        if store is not None
                        else (be.weight(tables[0][e]), be.weight(tables[1][e])),
                    )
                )
        for batch in store.landed(pending) if pending else ():
            for e, _f, s in batch:
                hits.append((e, *rows(e), store._views_mx(s)))
        hits.sort(key=lambda t: t[0])
        out = be.experts(x, [(ti, w, pair) for _, ti, w, pair in hits], self.sm._mlx_act())
        final += out.to(final.dtype)

    def _mx_biases(self) -> Any:
        """the two per-expert bias tables as float32 MLX arrays, made once per layer"""
        b = getattr(self, "_mx_bias", None)
        if b is None:
            b = (
                mlxdev.to_mx(self.gate_up_proj_bias.data.detach().float().contiguous()),
                mlxdev.to_mx(self.down_proj_bias.data.detach().float().contiguous()),
            )
            mlxdev.mx().eval(*b)
            self._mx_bias = b
        return b

    def _linear(self, x: torch.Tensor, w: Any) -> torch.Tensor:
        if isinstance(w, MxGateUp):
            return self._join(self._linear(x, w.gate), self._linear(x, w.up))
        if isinstance(w, MxWeight):
            # the MXFP4 kernel widens a column tile once and reuses it over the batch tile, so it is the
            # path at every batch size and a row's value does not depend on how many rows travel with it
            kernel = Native.gemv_mx4_ggml if w.ggml else Native.gemv_mx4
            self.sm._tag(PassTag.EXPERT_MXFP4_ASSTORED if kernel is not None else PassTag.EXPERT_MXFP4_DEQUANT)
            if kernel is not None:
                y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32)
                kernel(w, x.float().contiguous().cpu(), y)
                return y.to(x.device).to(x.dtype)
            return torch.nn.functional.linear(x.float().cpu(), w.dequantize(torch.float32)).to(x.device).to(x.dtype)
        if isinstance(w, F8Weight):
            self.sm._tag(PassTag.EXPERT_FP8_ASSTORED if Native.gemv_fp8 is not None else PassTag.EXPERT_FP8_WIDENED)
            if Native.gemv_fp8 is not None:
                y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32)
                Native.gemv_fp8(w, x.float().contiguous().cpu(), y)
                return y.to(x.device).to(x.dtype)
            return torch.nn.functional.linear(x.float().cpu(), w.dequantize(torch.float32)).to(x.device).to(x.dtype)
        if x.device.type == "cpu":
            if Native.gemv is not None and x.shape[0] < Native.gemm_rows:
                y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32)
                Native.gemv(w, x.float().contiguous(), y)
                return y.to(x.dtype)
            return torch.nn.functional.linear(x.float(), w.float()).to(x.dtype)
        return torch.nn.functional.linear(x, w.to(x.device, non_blocking=True).to(x.dtype))

    def _depot_for(self, x: torch.Tensor, on_host: bool) -> Any:
        """the depot of a layer-by-layer prefill on the card, where this call's rows are on the card"""
        return getattr(self.sm, "_depot", None) if (not on_host and x.device.type == "cuda") else None

    def _grouped_ok(self, x: torch.Tensor) -> bool:
        """whether a call's experts can go as grouped matmuls on the card: bf16 experts, MXFP4 ones in the
        checkpoint's layout (gpt-oss's, their biases and gate with them) or FP8 ones, over bf16 rows, where the torch
        build has the grouped matmul (`BTB_GROUPED_EXPERTS=0` keeps the per-expert loop)"""
        return (
            x.dtype == torch.bfloat16
            and not self.ggml
            and bool(getattr(self.sm, "grouped_experts", False))
            and hasattr(torch, "_grouped_mm")
        )

    def _card_grouped(
        self,
        x: torch.Tensor,
        w_top: torch.Tensor,
        top_k_index: torch.Tensor,
        hit: Any,
        views: Any,
        pending: Any,
        store: Any,
        depot: Any,
        final: torch.Tensor,
        seated: set[int] | None = None,
    ) -> None:
        """A call's experts on the card as grouped matmuls over the depot's stacked slots, a wave at a time: the
        experts already in RAM, then each batch as it lands from the drive, each wave placed on the card
        (`LayerDepot.place`) and multiplied as one gate_up, one gate and one down over the slots it took - a few
        launches a wave where the per-expert loop took ~10 an expert. A slot's rows are its expert's in the order
        `torch.where` lists them, and the grouped matmul's product over them is the per-expert one bit for bit on
        the card, so each (row, pick) contribution is the loop's. They are summed as the loop sums them: each row's
        in ascending expert order, from zero, one add at a time (a row's picks are distinct, so its k
        contributions fill `k` places, ranked by expert).

        MXFP4 experts (gpt-oss's) and FP8 ones cross the bus as stored and are widened to bf16 on the card
        `WIDEN_BATCH` at a time - MXFP4 by `dequant_blocks` (exact: every value times its power-of-two scale is a
        bf16), FP8 by `fp8.held` (the nearest bf16 to e4m3 times its scale, the weight every tier reads) - their biases
        added and their gate taken per row. The per-expert loop multiplies them on the host's kernels in float32 and
        hands the card the bf16 of it, so the steps are the loop's, dtype for dtype; only the order of the float32
        sums inside a product differs, and with it now and then a bf16's last bit."""
        self.sm._tag(PassTag.EXPERT_CARD_GROUPED)
        if self.mx:
            self.sm._tag(PassTag.EXPERT_MXFP4_DEQUANT)  # widened on the card, `dequant_blocks`
        elif self.f8:
            self.sm._tag(PassTag.EXPERT_FP8_WIDENED)  # widened on the card, `fp8.held`
        T, k = int(top_k_index.shape[0]), int(top_k_index.shape[1])
        pos_s, row_s, offs, counts = group_picks(top_k_index, self.num_experts)
        ranks = torch.argsort(torch.argsort(top_k_index, dim=1), dim=1)
        # zeros: a call served in waves (`StoreCall.wave`) fills here only its wave's picks, the others' adding nothing
        buf = torch.zeros(T, k, x.shape[-1], dtype=final.dtype, device=x.device)

        def wave(items: list[tuple[int, Any, Any]]) -> None:
            while items:
                slots = depot.place(self.layer, items)
                if not any(sl is not None for sl in slots):
                    raise RuntimeError(f"[experts] layer {self.layer}: no expert of the wave has a place on the card")
                # one grouped matmul a block the wave touched, its experts in their rows' order there
                by_block: dict[int, list[tuple[int, int]]] = {}
                for (e, _gu, _dn), sl in zip(items, slots, strict=True):
                    if sl is not None:
                        b, row = depot.where[sl]
                        by_block.setdefault(b, []).append((row, e))
                for b, local in sorted(by_block.items()):
                    local.sort()
                    if self.mx or self.f8:
                        for a in range(0, len(local), self.WIDEN_BATCH):
                            multiply(depot.blocks[b], local[a : a + self.WIDEN_BATCH])
                    else:
                        multiply(depot.blocks[b], local)
                items = [it for it, sl in zip(items, slots, strict=True) if sl is None]

        def multiply(stacks: list[torch.Tensor], placed: list[tuple[int, int]]) -> None:
            """the grouped products of `placed` (row, expert) of one block's `stacks`, in row order, into their
            rows' places"""
            if self.mx or self.f8:
                # the batch's bytes widened: one group an expert, in the batch's order
                idx = torch.tensor([sl for sl, _e in placed], dtype=torch.int32, device=x.device)
                n = len(placed)
                kern = Native.card_kernels() if self.mx else None
                if kern is not None:
                    # straight from the depot's stacks in one pass, no gathered copy of the bytes
                    gu_w, dn_w = (
                        kern.mx4_widen(stacks[2 * i], stacks[2 * i + 1], idx).view(n, rows, cols)
                        for i, (rows, cols) in enumerate(store.mx_shapes())
                    )
                elif self.mx:
                    gu_w, dn_w = (
                        dequant_blocks(
                            stacks[2 * i].index_select(0, idx).view(n, rows, cols // 32, 16),
                            stacks[2 * i + 1].index_select(0, idx).view(n, rows, cols // 32),
                        ).reshape(n, rows, cols)
                        for i, (rows, cols) in enumerate(store.mx_shapes())
                    )
                else:
                    gu_w, dn_w = (
                        fp8.held(
                            stacks[2 * i].index_select(0, idx).view(n, rows, cols),
                            stacks[2 * i + 1].index_select(0, idx),
                        )
                        for i, (rows, cols) in enumerate(store.f8_shapes())
                    )
                per = [counts[e] for _sl, e in placed]
            else:
                gu_w, dn_w = stacks[0], stacks[1]
                per = [0] * int(gu_w.shape[0])
                for sl, e in placed:
                    per[sl] = counts[e]
            rows_l, poss_l = [], []
            for _sl, e in placed:
                a, n = offs[e], counts[e]
                rows_l.append(row_s[a : a + n])
                poss_l.append(pos_s[a : a + n])
            rows, poss = torch.cat(rows_l), torch.cat(poss_l)
            ends = torch.tensor(list(accumulate(per)), dtype=torch.int32).to(x.device, non_blocking=True)
            # each row's expert: the biases' rows, and what `_act` reads them by
            who = top_k_index[rows, poss]
            h = torch._grouped_mm(x.index_select(0, rows), gu_w.transpose(1, 2), offs=ends)
            h = self._act(h, who)
            y = torch._grouped_mm(h, dn_w.transpose(1, 2), offs=ends)
            if self.biased:
                y = y + self._bias(self.down_proj_bias, who, y)
            buf[rows, ranks[rows, poss]] = (y * w_top[rows, poss, None]).to(buf.dtype)

        # the experts seated on the card by an earlier chunk (no bytes in RAM: `place` finds their seats) and
        # those in RAM, then each batch as it lands
        wave([(e, *views[e]) if e in views else (e, None, None) for e in hit if e in views or e in (seated or ())])
        for batch in store.landed(pending) if pending else ():
            wave([(e, *store._views(s)) for e, _f, s in batch])
        depot.settle()
        for j in range(k):
            final.add_(buf[:, j])

    def _waves(
        self,
        x: torch.Tensor,
        w_top: torch.Tensor,
        top_k_index: torch.Tensor,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        now: list[int],
        views: Any,
        pending: Any,
        call: StoreCall | None,
        store: Any,
        final: torch.Tensor,
        seated: set[int] | None = None,
    ) -> None:
        """A call of many rows over its experts, in the waves the store serves it (`StoreCall.wave`): each wave's experts
        multiplied - as grouped matmuls on the card through the depot, or the per-expert loop - and added into
        `final`, the next wave asked for once these are done with. Each wave is a prefix of the call's ascending
        experts, so a row's contributions are added in ascending expert order across the waves, as in one pass.
        `seated`: the experts the depot holds on the card for this layer already, not asked of the store - the
        grouped path's, placed from their seats with the first wave"""
        on_host = x.device.type == "cpu" and hidden_states.device.type != "cpu"
        # experts seated on the card: the depot took this layer's form in an earlier chunk, so the path is grouped
        grouped: bool | None = True if seated else None
        while True:
            depot = self._depot_for(x, on_host)
            if grouped is None:
                # the depot takes this form at all (its scratch slots, where the ledger has them): the loop where it
                # cannot - the form read off an expert in RAM, or off the first one still coming from the drive
                grouped = bool(
                    depot is not None
                    and self._grouped_ok(x)
                    and all(stored_parts(*v) is not None for v in views.values())
                    and (bool(views) or bool(pending))
                    and depot.takes(
                        stored_parts(*(next(iter(views.values())) if views else store._views(pending[0][2])))
                    )
                )
            if grouped:
                self._card_grouped(x, w_top, top_k_index, now, views, pending, store, depot, final, seated)
                seated = None  # placed with the first wave
            else:
                self._loop(x, w_top, top_k_index, expert_mask, hidden_states, now, views, pending, store, final)
            if call is None or call.done:
                return
            # this wave's experts are done with (the depot's copies out of their slots landed): the store may seat
            # the next wave in their slots
            asked = list(call.rest)
            views, pending = call.wave()
            left = set(call.rest)
            now = [e for e in asked if e not in left]

    def _loop(
        self,
        x: torch.Tensor,
        w_top: torch.Tensor,
        top_k_index: torch.Tensor,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        hit: list[int],
        views: Any,
        pending: Any,
        store: Any,
        final: torch.Tensor,
    ) -> None:
        """the per-expert loop over `hit`: each expert's rows multiplied on its own and added into `final` in
        ascending expert order"""
        dev = hidden_states.device
        on_host = x.device.type == "cpu" and dev.type != "cpu"
        contrib = {}
        # a prefill sweeping this layer's chunks on the card: the layer's experts cross the bus once for them all
        depot = self._depot_for(x, on_host)
        # rows on the card: every expert's rows found by one sort there, not a host lookup and a copy an expert,
        # each of which waited for the card to finish what was queued before it
        groups = group_picks(top_k_index, self.num_experts) if x.device.type != "cpu" else None

        def run(e: int, w_gu: Any, w_dn: Any) -> None:
            if groups is not None:
                pos_s, row_s, offs, counts = groups
                a, n = offs[e], counts[e]
                top_k_pos, token_idx = pos_s[a : a + n], row_s[a : a + n]
            else:
                top_k_pos, token_idx = torch.where(expert_mask[e])
                if not on_host:
                    top_k_pos, token_idx = top_k_pos.to(dev), token_idx.to(dev)
            if depot is not None:
                w_gu, w_dn = depot.get(self.layer, e, w_gu, w_dn)
            if isinstance(w_gu, torch.Tensor) and w_gu.device.type != "cpu" and x.device != w_gu.device:
                # an expert seated on the card while the rows are on the host: its rows go to the card and
                # are multiplied there in float32 over the stored bf16, as the one-row path does
                cur = hidden_states[token_idx.to(hidden_states.device)].float()
                h = self._act(torch.nn.functional.linear(cur, w_gu.float()), None)
                h = torch.nn.functional.linear(h.float(), w_dn.float()).to(x.device).to(x.dtype)
            else:
                cur = x[token_idx]
                h = self._act(self._linear(cur, w_gu), e)
                h = self._linear(h, w_dn)
            if self.biased:
                h = h + self._bias(self.down_proj_bias, e, h)
            contrib[e] = (token_idx, h * w_top[token_idx, top_k_pos, None])

        for e in hit:
            if e in views:
                run(e, *views[e])
        for batch in store.landed(pending) if pending else ():
            for e, _f, s in batch:
                run(e, *store._views(s))
        if depot is not None:
            depot.settle()
        for e in hit:
            token_idx, h = contrib[e]
            final.index_add_(0, token_idx, h.to(final.dtype))

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        t0 = time.perf_counter()
        dev = hidden_states.device
        probe = getattr(self.sm, "expert_probe", None)
        if probe is not None:
            if hidden_states.shape[0] == 1:
                # the routing instrument: the router's input and its choice, per layer, for one-row passes
                probe.append(
                    (self.layer, hidden_states.detach().float().cpu().clone(), top_k_index.detach().cpu().clone())
                )
            else:
                # a prefill's rows: their picks and weights, for the weight tail of the experts a prefill reads
                probe.append(
                    (
                        self.layer,
                        None,
                        top_k_index.detach().cpu().clone(),
                        top_k_weights.detach().float().cpu().clone(),
                    )
                )
        on_host = dev.type == Device.CUDA and hidden_states.shape[0] < Native.gemm_rows
        x = hidden_states.cpu() if on_host else hidden_states
        w_top = top_k_weights.cpu() if on_host else top_k_weights
        final = torch.zeros_like(x)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index.cpu(), num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        hit = [int(e[0]) for e in expert_hit]
        store = self.sm.expert_store
        pending = []
        w0 = 0.0
        keep = True
        # FP8 experts have no MLX matvec: they stay on the host's FP8 kernels
        use_mlx = (
            self.sm.mlx is not None
            and x.device.type == "cpu"
            and bool(hit)
            and not self.f8
            and self.layer in self.sm.mlx_layers
            and self.sm._mlx_act() is not None
            and (not self.mx or (store is not None and bool(getattr(self.sm.mlx, "mxfp4", False))))
        )
        grouped = (
            not use_mlx
            and x.device.type == "cpu"
            and x.shape[0] == 1
            and self._group() is not None
            and hit
            and len(hit) == top_k_index.shape[1]
        )
        if store is not None:
            self.sm._tag(PassTag.EXPERT_STORE, store.res_tag)  # the slots, under the policy the store was built on
            keep = hidden_states.shape[0] < Native.gemm_rows or bool(getattr(self.sm, "_sweep_keep", False))
            if hidden_states.shape[0] < Native.gemm_rows:
                # the Timetable: the next layers' picks from this layer's input (one row, or a verify pass's few),
                # read ahead while this layer runs
                store.lookahead(self.layer, hidden_states)
            # the depot as the tier the call asks first: a chunk of a layer-by-layer prefill after the layer's first
            # multiplies the experts an earlier chunk seated on the card from their seats, so the store is asked only
            # for the rest - held in RAM chunk after chunk too, a store below the layer's experts read the whole layer
            # from the drive again every chunk. Only where the grouped path is certain (the loop needs every view)
            depot = self._depot_for(x, on_host) if not (use_mlx or grouped) and self._grouped_ok(x) else None
            seated = depot.seated(self.layer) & set(hit) if depot is not None else set()
            ask = [e for e in hit if e not in seated] if seated else hit
            call = store.call(self.layer, self.base, ask, keep=keep, rows=int(hidden_states.shape[0]))
            # the MLX and one-row paths multiply the call's experts together: all of them seated at once, or the call
            # refused; the rest in the waves the store can seat
            views, pending = call.whole() if (use_mlx or grouped) else call.wave()
            if getattr(self.sm, "_sweep_ahead", False) and store.sweep_layer != self.layer:
                # a layer-by-layer prefill at this layer's first chunk: its reads are queued, and the next layer's
                # experts are read ahead behind them while this layer's chunks compute
                store.sweep_layer = self.layer
                store.lookahead(self.layer, hidden_states, sweep=len(hit))
            per_expert = store.per
            w0 = store.stat["wait_s"]
        else:
            self.sm._tag(PassTag.EXPERT_TABLES)  # no store: the checkpoint's expert tables, held whole
            gu, dn = self._tables()
            views = {e: (gu[e], dn[e]) for e in hit}
            call = None
            seated = set()
            if self.mx or self.f8:
                per_expert = gu[0].nbytes + dn[0].nbytes
            else:
                per_expert = (gu.shape[1] * gu.shape[2] + dn.shape[1] * dn.shape[2]) * gu.element_size()
        # the experts of this wave (the call's whole, unless the store serves it in turn)
        left = set(call.rest) if call is not None else set()
        now = [e for e in hit if e not in left]
        if use_mlx:
            self._mlx_forward(x, w_top, expert_mask, hit, views, pending, store, final)
        elif grouped:
            self._one_row(x, w_top, top_k_index, hit, views, pending, store, final, hidden_states if on_host else None)
        else:
            self._waves(
                x, w_top, top_k_index, expert_mask, hidden_states, now, views, pending, call, store, final, seated
            )
        st = self.sm.expert_stat
        st["experts"] += len(hit)
        st["bytes"] += len(hit) * per_expert
        st["calls"] += 1
        st["s"] += time.perf_counter() - t0
        prof = getattr(self.sm, "expert_profile", None)
        if prof is not None and store is not None:
            prof.add(
                prof.CALL,
                self.layer,
                expert=int(hidden_states.shape[0]),
                nbytes=len(hit) - (call.read if call is not None else len(pending)),
                shard=call.read if call is not None else len(pending),
                dur_ns=int((store.stat["wait_s"] - w0) * 1e9),
            )
        if self.sm.expert_trace is not None:
            self.sm.expert_trace.append((self.layer, top_k_index.cpu().clone()))
        return final.to(dev) if on_host else final


class _NGramRows(torch.nn.Module):
    sm: Any
    base: str
    dim: Any
    f8: list[F8Weight] | None
    out_dtype: torch.dtype
    parts: int
    rows: Any
    shards: Any
    weight: torch.Tensor

    def __init__(self, sm: Any, base: str, parts: Any, out_dtype: torch.dtype) -> None:
        super().__init__()
        object.__setattr__(self, "sm", sm)
        self.base = base
        self.parts = int(parts)
        self.out_dtype = out_dtype
        self.shards = None
        self.f8 = None
        self.weight = torch.zeros(0)

    def _open(self) -> Any:
        if self.shards is None:
            keys = [self.base + f"shard_{k}.weight" for k in range(self.parts)]
            # an FP8 table with one scale (`FP8Embedding`, too big to widen) stays e4m3: the rows a lookup gathers
            # are rescaled; one scaled by blocks is widened whole
            f8 = [self.sm._f8_weights(k)[0] for k in keys if self.sm._fp8(k)]
            self.f8 = f8 if len(f8) == len(keys) and all(w.grid == (1, 1) for w in f8) else None
            if self.f8 is not None:
                self.shards = [w.w.view(torch.float8_e4m3fn).reshape(w.shape) for w in self.f8]
            else:
                self.shards = [self.sm._get(k) for k in keys]
            self.rows = int(self.shards[0].shape[0])
            self.dim = int(self.shards[0].shape[1])
        return self.shards

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        sh = self._open()
        flat = ids.reshape(-1).cpu().long()
        f8 = self.f8
        out = torch.empty(flat.shape[0], self.dim, dtype=torch.float32 if f8 is not None else sh[0].dtype)
        k = flat // self.rows
        r = flat - k * self.rows
        for j in torch.unique(k).tolist():
            m = k == j
            out[m] = fp8.held(sh[j][r[m]], f8[j].scales).float() if f8 is not None else sh[j][r[m]]
        if f8 is not None:
            self.sm._tag(PassTag.FP8_ASSTORED)
        return out.to(self.out_dtype).view(*ids.shape, self.dim).to(ids.device)
