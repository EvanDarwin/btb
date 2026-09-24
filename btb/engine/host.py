# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The host tier's modules: a linear over a checkpoint view (native kernels or the CPU stream for its matmul),
the MoE router and experts, and the n-gram proposer's row table."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import torch

from .. import mlx as mlxdev
from ..kinds import PassTag
from ..mxfp4 import BLOCK, MxGateUp, MxWeight
from ..options import Device
from .native import Native
from .pack import unpack_bf16

if TYPE_CHECKING:
    pass


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


class _Experts(torch.nn.Module):
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
    num_experts: int

    def __init__(
        self,
        sm: Any,
        base: str,
        num_experts: int,
        act_fn: Any,
        layer: int = -1,
        mx: bool = False,
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
        # gpt-oss: the experts stay MXFP4 into the matvec, the two per-expert biases ride with the layer, the gate is
        # its own clamped GLU
        self.mx = bool(mx)
        self.ggml = self.mx and getattr(sm, "gguf", None) is not None  # a GGUF's experts, ggml's layout as stored
        self.gate = gate
        self.alpha = float(alpha)
        self.limit = float(limit)
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
            else:
                self.gate_up = self.sm._get(self.base + "gate_up_proj")
                self.down = self.sm._get(self.base + "down_proj")
        return self.gate_up, self.down

    def _act(self, gate_up: torch.Tensor, rows: Any = None) -> torch.Tensor:
        """The expert's middle: `gate_up` [n, 2 * intermediate] to [n, intermediate]; `rows` names the expert behind
        each row when there is a bias to add."""
        if self.mx:
            gate_up = gate_up + self._bias(self.gate_up_proj_bias, rows, gate_up)
        return self.gate(gate_up, self.alpha, self.limit) if self.gate is not None else self._silu_gate(gate_up)

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
        if self.ggml:
            if Native.gemv_mx4_ggml_group is None:
                raise RuntimeError("[experts] a GGUF's MXFP4 experts need the native library built with ggml's layout")
            return Native.gemv_mx4_ggml_group
        return Native.gemv_mx4_group if self.mx else Native.gemv_group

    def _gate_up_rows(self, gemv_group: Any, ws: Any, xf: torch.Tensor, pos: Any, gu_buf: torch.Tensor) -> None:
        """`gu_buf[i] = gate_up_e x` for the experts at `pos`: one matvec each, or a GGUF's gate and up matvecs
        interleaved into the checkpoint's [2I] order"""
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
            gu_buf[i] = MxGateUp.interleave(tg[j], tu[j])

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
            rows = torch.tensor([hit[i] for i in pos]) if self.mx else None
            h = self._act(gu_buf[pos].to(dt), rows).float().contiguous()
            gemv_group([w[1] for w in ws], [h[j : j + 1] for j in range(len(pos))], [dn_buf[i : i + 1] for i in pos])
            for _j, i in enumerate(pos):
                o = dn_buf[i] + self._bias(self.down_proj_bias, hit[i], dn_buf) if self.mx else dn_buf[i]
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
            bg, bd = self._mx_biases()
            shp = store.mx_shapes()
            xm = mlxdev.to_mx(x)
            parts = {}

            def part(e: int, pair: Any) -> None:
                ti, wts = rows(e)
                parts[e] = be.expert_mx(
                    xm, ti, wts, pair, e, shp[0], shp[1], bg, bd, self.alpha, self.limit, ggml=store.ggml
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
            return MxGateUp.interleave(self._linear(x, w.gate), self._linear(x, w.up))
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
        if x.device.type == "cpu":
            if Native.gemv is not None and x.shape[0] < Native.gemm_rows:
                y = torch.empty(x.shape[0], w.shape[0], dtype=torch.float32)
                Native.gemv(w, x.float().contiguous(), y)
                return y.to(x.dtype)
            return torch.nn.functional.linear(x.float(), w.float()).to(x.dtype)
        return torch.nn.functional.linear(x, w.to(x.device, non_blocking=True).to(x.dtype))

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
        if store is not None:
            self.sm._tag(PassTag.EXPERT_STORE, store.res_tag)  # the slots, under the policy the store was built on
            keep = hidden_states.shape[0] < Native.gemm_rows or bool(getattr(self.sm, "_sweep_keep", False))
            if hidden_states.shape[0] < Native.gemm_rows:
                # the Timetable: the next layers' picks from this layer's input (one row, or a verify pass's few),
                # read ahead while this layer runs
                store.lookahead(self.layer, hidden_states)
            views, pending = store.get(self.layer, self.base, hit, keep=keep, rows=int(hidden_states.shape[0]))
            per_expert = store.per
            w0 = store.stat["wait_s"]
        else:
            self.sm._tag(PassTag.EXPERT_TABLES)  # no store: the checkpoint's expert tables, held whole
            gu, dn = self._tables()
            views = {e: (gu[e], dn[e]) for e in hit}
            if self.mx:
                per_expert = sum(w.blocks.numel() + w.scales.numel() for w in (gu[0], dn[0]))
            else:
                per_expert = (gu.shape[1] * gu.shape[2] + dn.shape[1] * dn.shape[2]) * gu.element_size()
        use_mlx = (
            self.sm.mlx is not None
            and x.device.type == "cpu"
            and bool(hit)
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
        if use_mlx:
            self._mlx_forward(x, w_top, expert_mask, hit, views, pending, store, final)
        elif grouped:
            self._one_row(x, w_top, top_k_index, hit, views, pending, store, final, hidden_states if on_host else None)
        else:
            contrib = {}

            def run(e: int, w_gu: Any, w_dn: Any) -> None:
                top_k_pos, token_idx = torch.where(expert_mask[e])
                if not on_host:
                    top_k_pos, token_idx = top_k_pos.to(dev), token_idx.to(dev)
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
                if self.mx:
                    h = h + self._bias(self.down_proj_bias, e, h)
                contrib[e] = (token_idx, h * w_top[token_idx, top_k_pos, None])

            for e in hit:
                if e in views:
                    run(e, *views[e])
            for batch in store.landed(pending) if pending else ():
                for e, _f, s in batch:
                    run(e, *store._views(s))
            for e in hit:
                token_idx, h = contrib[e]
                final.index_add_(0, token_idx, h.to(final.dtype))
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
                nbytes=len(hit) - len(pending),
                shard=len(pending),
                dur_ns=int((store.stat["wait_s"] - w0) * 1e9),
            )
        if self.sm.expert_trace is not None:
            self.sm.expert_trace.append((self.layer, top_k_index.cpu().clone()))
        return final.to(dev) if on_host else final


class _NGramRows(torch.nn.Module):
    sm: Any
    base: str
    dim: Any
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
        self.weight = torch.zeros(0)

    def _open(self) -> Any:
        if self.shards is None:
            self.shards = [self.sm._get(self.base + f"shard_{k}.weight") for k in range(self.parts)]
            self.rows = int(self.shards[0].shape[0])
            self.dim = int(self.shards[0].shape[1])
        return self.shards

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        sh = self._open()
        flat = ids.reshape(-1).cpu().long()
        out = torch.empty(flat.shape[0], self.dim, dtype=sh[0].dtype)
        k = flat // self.rows
        r = flat - k * self.rows
        for j in torch.unique(k).tolist():
            m = k == j
            out[m] = sh[j][r[m]]
        return out.to(self.out_dtype).view(*ids.shape, self.dim).to(ids.device)
