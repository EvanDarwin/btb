# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's fast paths: the captured CUDA graphs of the one-token step and the speculative tree's verify
pass over the host-side attention cache."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .. import mlx as mlxdev
from ..kinds import LayerKind, NodePath, Parents, PassTag, Tokens
from ..options import Device
from ..sampling import GREEDY
from .cache import GrowLayer, set_rows
from .families import act_name
from .forward import layer_window, node_mask, pe_for
from .fused import _fused_rope
from .native import Native, kernels_path
from .state import _State

if TYPE_CHECKING:
    from .forward import PassRope, Rope

_KERNELS_WARNED = False


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


class _CudaMixin(_State):
    def aa(self, parents: Parents | None = None) -> None:
        self.al = {}
        self.am = {}
        self.an = {}
        self.ap = parents
        self.aq = True

    def ab(self) -> None:
        self.aq = False
        self.ap = None

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
            base = layer.keys.shape[-2] if getattr(layer, "keys", None) is not None else 0
        for p in range(T):
            hp = h[:, p : p + 1]
            pos_p = text_pos[:, p : p + 1]
            pe_p = (rope[0][:, p : p + 1], rope[1][:, p : p + 1])
            mask_p = None
            branch = tree and parents[p] != p - 1
            dev = h.device
            if lt == LayerKind.LINEAR and branch:
                conv, rec = pre if parents[p] < 0 else ckpts[parents[p]]
                c, r = self._lin(layer)
                c.copy_(conv)
                r.copy_(rec)
            if base is not None:
                rows = node_mask(base, p, parents, win)
                mask_p = None if rows is None else rows.view(1, 1, 1, -1).to(dev)
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

    def _seg_a(self, g: dict[str, Any], i: int) -> None:
        apply_rotary_pos_emb = _fused_rope if self._frope else self.fam.mod.apply_rotary_pos_emb
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
        self._frope = os.environ.get("BTB_FUSED_ROPE", "1") != "0"
        self._fmlp = os.environ.get("BTB_FUSED_MLP", "1") != "0"
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
    ATTN_SPLIT = 1024  # keys per attention block along the sequence (ATTN_SPLIT in btb_kernels.cu)

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
                (self.fam.kernel_layout or self.fam.sandwich)
                and act_name(self.cfg) in ("silu", "swish", "gelu_pytorch_tanh")  # the kernels' activations
                and not self.fam.own
                and self.mlx is None
                and not getattr(self, "resident_fp32", False)
                and self.compute_dtype in (None, torch.bfloat16)
            )
            if ok:
                H, Hq, Hk, D, I = self._card_dims()
                # the kernels' shapes: a lane holds D/32 dims of a head, the rows load 16 bytes at a time
                ok = D in (64, 128, 256) and H % 8 == 0 and I % 8 == 0 and (Hq * D) % 8 == 0 and Hq % Hk == 0
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
        )

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
        }
        self._cg = st
        return st

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

    def _card_arena(self, st: dict[str, Any], need: int) -> dict[str, Any]:
        """the arena holding every run layer's cache rows, one tensor [runs' layers, 2, Hk, cap, d] the caches
        attach to (a cache attached elsewhere copies its rows in), at least `need` rows deep; grown by
        reallocation (rows copied over, graphs dropped) when a context outruns it. Its front lies in a
        persisting-L2 window, so a short context's attention reads hit L2 under the weights' evict-first loads"""
        ar = st["arena"]
        if ar is not None and ar["cap"] >= need:
            return ar
        H, Hq, Hk, D, I = self._card_dims()
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        cap = max(int(need), 4096)
        cap = (cap + 1023) // 1024 * 1024
        nbytes = len(layers) * 2 * Hk * cap * D * 2
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            sched.grant(
                nbytes,
                "kv",
                requester=f"card arena {len(layers)} layers x {cap} rows",
                B=1,
                cap=cap,
                bound=None,
                device=self.dev,
            )
        A = torch.empty(len(layers), 2, Hk, cap, D, dtype=torch.bfloat16, device=self.dev)
        slot = {i: j for j, i in enumerate(layers)}
        new = {"A": A, "cap": cap, "slot": slot, "owner": ar["owner"] if ar is not None else None}
        if ar is not None:
            owner = ar["owner"]() if ar["owner"] is not None else None
            for i, j in ar["slot"].items():
                if i in slot:
                    A[slot[i], :, :, : ar["cap"]].copy_(ar["A"][j])
                    if owner is not None:
                        layer = owner.layers[i]
                        n = int(layer.keys.shape[-2]) if (layer.is_initialized and layer.keys is not None) else 0
                        layer._buf = (A[slot[i], 0][None], A[slot[i], 1][None])
                        if n:
                            layer._set_rows(layer._buf[0][..., :n, :], layer._buf[1][..., :n, :])
        st["arena"] = new
        st["graphs"].clear()
        # how much of the arena's front sits in persisting L2 is the scheduler's call (it holds the
        # hierarchy's sizes); the driver call is only the mechanism
        k = st["k"]
        pinned = self.scheduler.pin_bytes(A.numel() * 2) if getattr(self, "scheduler", None) is not None else 0
        k.persist(A.data_ptr(), pinned, stream=st["stream"])
        k.persist(A.data_ptr(), pinned)
        self.log(
            f"[card] arena {len(layers)} layers x {cap} rows ({nbytes / 2**20:.0f} MB), {pinned / 2**20:.0f} MB "
            f"of it pinned in L2"
        )
        return new

    def _card_adopt_cache(self, cache: Any) -> None:
        """a new cache takes the arena when no live cache holds it, so its prefill writes straight in"""
        st = getattr(self, "_cg", None)
        if st is None or st["arena"] is None or st["version"] != self.device.snapshot().version:
            return
        ar = st["arena"]
        owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not None and owner is not cache:
            return
        self._card_bind(cache, st, 0)

    def _card_bind(self, cache: Any, st: dict[str, Any], T: int) -> dict[str, Any]:
        import weakref

        ar = st["arena"]
        if ar is not None:
            owner = ar["owner"]() if ar["owner"] is not None else None
            same = owner is not None and owner is cache
            if cache is not None and same:
                # the common step: this cache holds the arena and every run layer sits at its front
                n = cache.get_seq_length()
                if n + T <= ar["cap"] and all(cache.layers[i]._an is not None for i in ar["slot"]):
                    return ar
        layers = [i for a, b in st["segments"] for i in range(a, b)]
        need = 0
        for i in layers:
            layer = cache.layers[i]
            n = layer.get_seq_length() if layer.is_initialized else 0
            need = max(need, n + T, int(getattr(layer, "cap_hint", 0) or 0))
        ar = self._card_arena(st, need)
        owner = ar["owner"]() if ar["owner"] is not None else None
        if owner is not cache:
            if owner is not None:
                for i in ar["slot"]:
                    if isinstance(owner.layers[i], GrowLayer):
                        owner.layers[i].detach()
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
        would part at a bf16 near-tie."""
        if not self._card_mma_avail():
            return False
        on = getattr(self, "card_mma", None)
        if on is None:
            env = os.environ.get("BTB_CARD_MMA")
            if env in ("0", "1"):
                on = env == "1"
        if on is not None:
            return bool(on)
        st = getattr(self, "_cg", None)
        choice = st.get("mma_for") if st is not None else None
        if choice is not None and T in choice:
            return bool(choice[T])
        one = st.get("mma_one") if st is not None else None
        if one is not None:
            # a width past the timed ones (a long prompt's prefill) takes the engine's kernel
            return bool(one)
        return False

    MMA_BLOCK = 64  # btb_gemv_mma_bf16: two warps over one group of 16 weight rows

    @staticmethod
    def _card_mma_grid(R: int, C: int) -> int:
        """the launch of btb_gemv_mma_bf16 for a [R, C] weight: one block per group of 16 rows (a shorter grid
        strides the groups and is only slower, never wrong)"""
        return (R + 15) // 16

    def _card_buffers(
        self, st: dict[str, Any], a: int, b: int, T: int, tail: bool, mma: bool | None = None
    ) -> dict[str, Any]:
        """the static buffers of the (run, T, GEMV) graph, made once; the graph itself is captured by
        `_card_capture` on the first pass, after that pass's inputs are in place"""
        if mma is None:
            mma = self._card_mma_for(T)
        key = (a, b, T, tail, bool(mma))
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
        S = (int(st["arena"]["cap"]) + self.ATTN_SPLIT - 1) // self.ATTN_SPLIT
        g = {
            "key": key,
            "mma": mma,
            "m": torch.zeros(M, I, dtype=bf, device=dev) if mma else None,
            "h": torch.zeros(M, H, dtype=bf, device=dev),
            "x": torch.zeros(M, H, dtype=bf, device=dev),
            "y": torch.zeros(M, H, dtype=bf, device=dev),
            "qkv": torch.zeros(M, (Hq + 2 * Hk) * D, dtype=bf, device=dev),
            "q": torch.zeros(M, Hq, D, dtype=bf, device=dev),
            "att": torch.zeros(M, Hq * D, dtype=bf, device=dev),
            "gu": torch.zeros(M, 2 * I, dtype=bf, device=dev),
            # the attention's per-split states and arrival counts (the split length is the kernel's; the
            # count of splits covers the arena, so one graph serves every length)
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

    def _card_body(self, st: dict[str, Any], g: dict[str, Any]) -> None:
        """the pass's kernels over `g`'s buffers, layers a..b-1 and the tail when the key carries it: run
        eagerly for the warm-up, then recorded by the capture"""
        import ctypes

        a, b, T, tail = g["key"][:4]
        k = st["k"]
        ar = st["arena"]
        H, Hq, Hk, D, I = self._card_dims()
        M = 32 if g.get("mma") else self._card_m(T)
        tables = self._card_tables(st, ar["cap"])
        P, ci, cf = k.ptr, ctypes.c_int, ctypes.c_float
        Ls = [self._card_weights(st, i) for i in range(a, b)]
        mma = bool(g.get("mma"))
        sandwich = self.fam.sandwich
        cen = ci(1 if self.fam.norm_centered else 0)  # the norms scale by 1 + w
        act = "gelu" if act_name(self.cfg) == "gelu_pytorch_tanh" else "silu"
        gemv = f"btb_gemv_bf16_m{M}"
        gemv_act = f"btb_gemv_{act}_bf16_m{M}"
        attn = f"btb_attn_split_d{D}"
        nrk = f"btb_norm_rope_kv_d{D}"
        if attn not in k.fn or nrk not in k.fn:
            raise RuntimeError(f"[card] no kernel for head_dim {D}")
        S = int(g["S"])

        def matvec(W: torch.Tensor, xin: torch.Tensor, yout: torch.Tensor, R: int, C: int) -> None:
            if mma:
                k.launch(
                    "btb_gemv_mma_bf16",
                    (self._card_mma_grid(R, C), 1, 1),
                    (self.MMA_BLOCK, 1, 1),
                    [P(W), P(xin), P(yout), ci(R), ci(C), ci(T)],
                )
            else:
                k.launch(gemv, ((R + 3) // 4, 1, 1), (128, 1, 1), [P(W), P(xin), P(yout), ci(R), ci(C)])

        y_prev = None
        for n, L in enumerate(Ls):
            j = ar["slot"][a + n]
            kb, vb = ar["A"][j, 0], ar["A"][j, 1]
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
                    P(g["n0"]),
                    P(g["depth"]),
                    P(kb),
                    P(vb),
                    P(g["q"]),
                    ci(T),
                    ci(Hq),
                    ci(Hk),
                    ci(ar["cap"]),
                    cen,
                ],
            )
            k.launch(
                attn,
                (Hq, T, S),
                (256, 1, 1),
                [
                    P(g["q"]),
                    P(kb),
                    P(vb),
                    P(g["att"]),
                    P(g["n0"]),
                    P(g["par"]),
                    ci(T),
                    ci(Hq),
                    ci(Hk),
                    ci(ar["cap"]),
                    cf(L["scale"]),
                    P(g["part_m"]),
                    P(g["part_l"]),
                    P(g["part_acc"]),
                    P(g["cnt"]),
                    ci(S),
                    ci(L["win"]),
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
            matvec(L["gu"], g["x"], g["gu"], 2 * I, H)
            if mma:
                # the tensor-core kernel has no activation fold: the two kernels, the same bits
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

    def _card_record(self, st: dict[str, Any], g: dict[str, Any], body: Any, what: str) -> None:
        """`body` once eagerly on the arena's stream (the warm-up), then captured there, so the captured
        kernel nodes inherit the stream's persisting window"""
        s = st["stream"]
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, pool=st["pool"], stream=s):
            body()
        g["graph"] = cg
        self.log(f"[card] {what}")

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

    def _card_step_graph(self, st: dict[str, Any], table: torch.Tensor, U: int, sampling: Any = None) -> dict[str, Any]:
        """U one-row steps as one graph: each step's token embedding, every layer, the head, the pick (the argmax,
        or the sample drawn inside the replay from the card's generator, its temperature and top-p read off
        device buffers, its top-k part of the key) written back as the next token, the cache's length advanced
        and the token published - the host only replays, and streams the tokens as they land"""
        mma = self._card_mma_for(1)
        U = int(U)
        # sampling is None only on a greedy step; `sampled` gates every sampling.* read below
        sampled = sampling is not None and not sampling.greedy
        top_k = int(sampling.top_k) if sampled else 0
        top_p_on = bool(sampled and sampling.top_p < 1.0)
        key = (0, self.L, 1, True, mma, "step", U, sampled, top_k, top_p_on)
        g = st["graphs"].get(key)
        if g is not None:
            return g
        g = dict(self._card_buffers(st, 0, self.L, 1, True, mma))
        g["key"] = key
        g["graph"] = None
        g["U"] = U
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
        g["pin_tok"] = torch.zeros(2 * U, dtype=torch.long, pin_memory=True)
        g["pin_n0"] = torch.zeros(1, dtype=torch.int32, pin_memory=True)
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

        def body(exec_id: int) -> None:
            for i in range(U):
                slot = exec_id * U + i
                torch.index_select(table, 0, g["ids"], out=g["h"][:1])
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
                    [P(g["n0"]), P(g["ids"]), P(g["pin_tok"][slot : slot + 1]), P(g["pin_n0"])],
                )

        g["body"] = body
        return g

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
        cache = self.new_cache(max_len=target_len)
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
        U = max(1, min(self._card_unroll(), steps))
        # the arena must hold every row the replays may write: whole replays, one past the last token
        ar = self._card_bind(cache, st, ((steps + U - 1) // U + 1) * U + 1)
        table = self._card_table(st)
        if table is None:
            # the gate granted the table moments ago; the card lost the room since. Named, not an attribute
            # error from inside the graph's first lookup
            from .scheduler import MemoryGrantError

            raise MemoryGrantError("[card] the embedding table for the step graph was refused after the gate passed")
        g = self._card_step_graph(st, table, U, smp)
        past = cache.get_seq_length()
        g["n0"].fill_(past)
        g["ids"].fill_(first)
        if g["graph"] is None:
            # the warm-up run consumes tokens and advances the length: both are reset after it. Two execs of
            # the same steps, launched in turn (a graph exec launched again while its last launch still runs
            # waits for it), each writing its own pinned token slots
            self._card_record(
                st,
                g,
                lambda: g["body"](0),
                f"{U} one-row step{'s' if U > 1 else ''} captured as a self-advancing graph",
            )
            cg2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cg2, pool=st["pool"], stream=st["stream"]):
                g["body"](1)
            g["graph2"] = cg2
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
        graphs = (g["graph"], g["graph2"])
        pin_tok, pin_n0 = g["pin_tok"], g["pin_n0"]
        pin_n0.fill_(past)
        done = 0  # tokens the host has read off the replays
        stop = False

        def token_of(k: int) -> int:
            # step k has finished when the graph's own copy of the length reads past + k + 1; its token sits
            # in the slot of its exec and its place within the replay
            target = past + k + 1
            waited = time.perf_counter()
            while int(pin_n0[0]) < target:
                if time.perf_counter() - waited > 30.0:
                    raise RuntimeError(f"[card] the step graph did not advance past {target} within 30 s")
            j, i = divmod(k, U)
            return int(pin_tok[(j & 1) * U + i])

        n_replays = (steps + U - 1) // U
        for j in range(n_replays):
            if self.abort.is_set():
                stop = True
                break
            graphs[j & 1].replay()
            if j >= 1:
                # the previous replay's tokens, streamed as each step lands
                for i in range(U):
                    k = (j - 1) * U + i
                    if k >= steps:
                        break
                    tok = token_of(k)
                    out.append(tok)
                    done = k + 1
                    if on_token:
                        on_token(tok)
                    if tok in eos:
                        stop = True
                        break
                if stop:
                    break
        if not stop:
            for k in range((n_replays - 1) * U, steps):
                tok = token_of(k)
                out.append(tok)
                done = k + 1
                if on_token:
                    on_token(tok)
                if tok in eos:
                    break
        torch.cuda.current_stream().synchronize()
        # the rows the sequence's processed tokens occupy: the prompt and every token fed to a replay whose
        # output was kept (the one after an eos is not part of the answer)
        n = past + max(0, len(out) - 1)
        for i in ar["slot"]:
            cache.layers[i].set_front(n)
        self.log(
            f"[stream] generated {len(out)} tokens over 1 rows in {time.time() - t0:.1f}s "
            f"({(time.time() - t0) / max(1, len(out)):.3f} s/step incl. prefill; the step graph, host {done} behind by one)"
        )
        return out

    def card_warm(self, ids: Tokens, t_max: int | None = None) -> int:
        """Capture the card graphs for every pass shape up front - one-row steps and verify passes of 2 ..
        `t_max` rows (the tree budget plus its root by default) - over a throwaway cache of `ids`, so no
        timed answer pays a capture. Returns the number of graphs captured; 0 where the card graph does not
        apply."""
        ids_t = torch.as_tensor(list(ids), dtype=torch.long).view(1, -1)
        if self.dev.type != Device.CUDA or self._card_kernels() is None or not self._card_ready():
            return 0
        t_max = int(t_max or (int(getattr(self, "tree_budget", 0) or 0) + 1))
        t_max = max(1, min(t_max, self.CARD_T_MAX))
        n = 0
        with torch.inference_mode():
            cache = self.new_cache(max_len=int(ids_t.shape[1]) + t_max + 2)
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
            choice: dict[int, bool] = st.setdefault("mma_for", {})
            timed_all: dict[int, dict[bool, float]] = {}
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
                timed: dict[bool, float] = {m: float("inf") for m in variants}
                graphs = {m: [st["graphs"][(*sg, T, tails[sg], m)]["graph"] for sg in segs] for m in variants}
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
            if timed_all:
                # one kernel for every width: the two sum a row in different orders, and a step on one with a
                # pass on the other parted at bf16 near-ties (0/8 identical at 256 tokens on the 0.6B). The
                # engine keeps the kernel that is faster at the width its loop lives at - the widest pass for
                # a speculative engine, the one-row step for a greedy one - and the other's graphs are dropped
                w = max(timed_all) if int(getattr(self, "v_max", 0) or 0) > 0 else 1
                pick = min(variants, key=lambda m: timed_all[w][m])
                st["mma_one"] = pick
                for T, timed in timed_all.items():
                    choice[T] = pick
                    cost[T] = timed[pick]
                    for mma in variants:
                        if mma != pick:
                            for sg in segs:
                                st["graphs"].pop((*sg, T, tails[sg], mma), None)
                if len(variants) > 1:
                    self.log(
                        f"[card] one GEMV for every width: {'tensor cores' if pick else 'fp32 chain'} "
                        f"(the {w}-row pass {timed_all[w][pick] * 1e3:.2f} ms against "
                        f"{timed_all[w][not pick] * 1e3:.2f}; the step {timed_all[1][pick] * 1e3:.2f} against "
                        f"{timed_all[1][not pick] * 1e3:.2f} ms)"
                    )
            n = len(st["graphs"]) - before
            # the cost curve covers the whole model only; with a layer on the CPU the timing above is partial
            if cost and segs == [(0, self.L)] and tails[segs[0]]:
                self._card_cost = cost
                c1 = cost[1]
                self.log(
                    "[card] pass cost by rows: "
                    + ", ".join(f"{T}:{c / c1:.2f}x" for T, c in cost.items() if T in (1, 2, 4, 8, 16, t_max))
                    + f" (one row {c1 * 1e3:.2f} ms)"
                )
        return n

    def _spec_full(self, v_max: int | None = None) -> int:
        """The widest speculative pass, the root included: the tree's rows, or a chain's `v_max` drafts (the
        call's, else the model's) and the root, whichever is wider."""
        v = int((getattr(self, "v_max", 0) if v_max is None else v_max) or 0)
        return max(int(getattr(self, "tree_budget", 0) or 0), v) + 1

    def _spec_budget(self, ema_tokens: float, passes: int, v_max: int | None = None, past: int = 0) -> int:
        """Rows a speculative pass may carry, the root included, from the warm-up's cost curve of the card
        graph's passes and the running acceptance: the widest pass costing at most a quarter more than the
        one-row step, and only while the tokens a pass yields pay for its width - otherwise one-row passes,
        with a wide probe every sixteenth pass so a stretch of accepted drafts can reopen the tree. Without a
        cost curve (the MLX and host paths) the configured budget stands: the tree's rows, or a chain's
        `v_max` drafts and the root, whichever is wider."""
        full = self._spec_full(v_max)
        cost = (
            getattr(self, "_card_cost", None) or getattr(self, "_mlx_cost", None) or getattr(self, "_host_cost", None)
        )
        if not cost or 1 not in cost or len(cost) <= 1 or full <= 1:
            return full
        slope = getattr(self, "_mlx_attn_slope", None)
        if slope and getattr(self, "_card_cost", None) is None and past > 0:
            # a pass over `past` rows: every node past the first reads the prefix again, at the warm-up's cost
            # per node and row (the slope at the nearest timed length above, the last one beyond it)
            b = next((s for r, s in slope if past <= r), slope[-1][1])
            n_attn = sum(1 for lt in self.layer_types if lt in (LayerKind.FULL, LayerKind.SLIDING))
            cost = {T: c + n_attn * (T - 1) * past * b for T, c in cost.items()}
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

    def _forward_card_segment(
        self, a: int, b: int, h: torch.Tensor, pas: Any, tail: bool
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """layers a..b-1 as one replay over `h` [1, T, H]: returns (h, None), or (None, logits [1, T, V]
        float32) when the run carries the tail"""
        self._tag(PassTag.CUDA_GRAPH)
        cache, T, past = pas.cache, pas.T, pas.past
        st = self._card_state()
        self._card_bind(cache, st, T)
        g = self._card_buffers(st, a, b, T, tail)
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
        for i in range(self.L):
            g["A"][i].replay()
            stream.synchronize()
            kf, vf = cache.update(g["kv_pin"][:hk].view(1, hk, 1, d), g["kv_pin"][hk:].view(1, hk, 1, d), i)
            win = layer_window(self.cfg, self.layer_types[i])
            first = max(0, int(kf.shape[-2]) - win) if win else 0  # a sliding layer reads its last rows alone
            Native.attn_decode(
                g["q_pin"], kf[0][:, first:], vf[0][:, first:], self.resident[i].self_attn.scaling, g["out_pin"]
            )
            g["B"][i].replay()
        return self._finish(g["h"].clone(), last_only, head)

    def ad(self, cache: Any, base_len: int, path: NodePath) -> None:
        keep = list(range(base_len)) + [base_len + j for j in path]
        lazy: list[Any] = []
        flags: list[Any] = []
        for i, layer in enumerate(cache.layers):
            # every attention layer's keys and values are cropped to the accepted path; a sliding layer
            # keeps the whole cache (its window is a mask, not a shorter cache), so it is cropped too
            if self.layer_types[i] in (LayerKind.FULL, LayerKind.SLIDING):
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
                    conv, rec = st.restore(path)
                    lazy.append((layer, conv, rec))
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

    def af(self, i: int, T: int, cl: Any) -> Any:
        slots = getattr(self, "ay", None)
        if slots is None:
            slots = self.ay = {}
        s = slots.get(i)
        if s is None or s[0].shape[0] < T:
            c, r = self._lin(cl)
            s = (
                torch.empty((T, *tuple(c.shape)), dtype=c.dtype, device=c.device),
                torch.empty((T, *tuple(r.shape)), dtype=r.dtype, device=r.device),
            )
            slots[i] = s
        return s

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

        apply_rotary_pos_emb = self.fam.mod.apply_rotary_pos_emb
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
        base = cl.keys.shape[-2] if getattr(cl, "keys", None) is not None else 0
        win = layer_window(self.cfg, self.layer_types[i])
        attn_fn = ALL_ATTENTION_FUNCTIONS.get_interface(self.cfg._attn_implementation, eager_attention_forward)
        q_rot, k_rot = apply_rotary_pos_emb(q_all.transpose(1, 2), k_all.transpose(1, 2), pe[0], pe[1])
        k_full, v_full = cache.update(k_rot, v_all.transpose(1, 2), i)
        kernel = (
            T == 1
            and not tree
            and Native.attn_decode is not None
            and k_full.shape[0] == 1
            and k_full.device.type == "cpu"
            and k_full.dtype in (torch.bfloat16, torch.float32)
            and v_full.dtype == k_full.dtype
            and k_full.stride(-1) == 1
            and k_full.stride(-2) == hd
            and v_full.stride(-1) == 1
            and v_full.stride(-2) == hd
        )
        if kernel:
            qf = q_rot[0, :, 0].float().contiguous()
            out = torch.empty(qf.shape, dtype=torch.float32)
            first = max(0, int(k_full.shape[-2]) - win) if win else 0  # a sliding layer's last rows alone
            Native.attn_decode(qf, k_full[0][:, first:], v_full[0][:, first:], at.scaling, out)
            a1 = out.view(1, 1, -1).to(h.dtype)
            outs.append(a1 if gate_all is None else a1 * torch.sigmoid(gate_all[:, 0:1]))
        for p in range(T) if not kernel else range(0):
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
