# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""How a token is picked from the logits: greedy (temperature 0, the argmax) or sampled with temperature, top-k and
top-p as one formula on every backend, argmax(mask(logits / temperature) + gumbel), the noise keyed by the seed
and the cache row it decides, so a speculative run and a sequential one draw the same tokens."""

from __future__ import annotations

import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .kinds import TokenRows

PRECUT = 1024  # the torch sampler's top-p works over this many of the top logits before it needs the full sort
FIELDS = ("temperature", "top_p", "top_k", "seed")


def _mix(*parts: int) -> int:
    """splitmix64 over the parts: a 63-bit key that changes throughout for a change in any part"""
    x = 0
    for p in parts:
        x = (x ^ (int(p) & 0xFFFFFFFFFFFFFFFF)) + 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
    return x & 0x7FFFFFFFFFFFFFFF


@dataclass(frozen=True)
class Sampling:
    """temperature 0 is greedy; else the logits over the temperature, the top-k kept, then the smallest set whose
    mass reaches top_p, then a Gumbel-max draw keyed by (seed, row, position)"""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @property
    def masked(self) -> bool:
        return self.top_k > 0 or self.top_p < 1.0

    def seeded(self) -> Sampling:
        """this sampling with a seed: its own, or one drawn now (recorded in the stats, so the run repeats)"""
        if self.greedy or self.seed is not None:
            return self
        return replace(self, seed=random.getrandbits(62))

    def key_for(self, pos: int, salt: int = 0) -> int:
        """the noise key of the pick decided at cache row `pos`: a speculative run and a sequential one agree;
        `salt` separates other draws at the row (the drafter's)"""
        return _mix(self.seed or 0, pos, salt) if salt else _mix(self.seed or 0, pos)

    def row(self, r: int) -> Sampling:
        """row r's sampling of a fork: row 0 draws as this one does alone, every other row under a seed of its own"""
        if self.greedy or r == 0:
            return self
        return replace(self, seed=_mix(self.seed or 0, 0x5EED, r) & ((1 << 62) - 1))

    def __post_init__(self) -> None:
        from .options import check_sampling

        check_sampling({f: getattr(self, f) for f in FIELDS})  # a bad value is an OptionError naming it

    @classmethod
    def from_request(cls, fields: Mapping[str, Any], default: Sampling) -> Sampling:
        """the request's own fields over `default`; a field absent or null keeps the default; a field of the
        wrong type or out of range an OptionError (the servers answer 400 with its line)"""
        from .options import check_sampling

        kw = check_sampling(fields)
        return replace(default, **kw) if kw else default

    def describe(self) -> str:
        if self.greedy:
            return "greedy"
        bits = [f"temperature {self.temperature:g}"]
        if self.top_k > 0:
            bits.append(f"top-k {self.top_k}")
        if self.top_p < 1.0:
            bits.append(f"top-p {self.top_p:g}")
        return ", ".join(bits)

    # -- the distribution and draws without replacement (the drafter's proposals) -----------------------------------

    def dist_torch(self, logits: torch.Tensor) -> torch.Tensor:
        """the distribution this sampling draws from: `logits` [R, V] scaled, top-k and top-p masked, normalized"""
        x = logits.float() / max(self.temperature, 1e-6)
        if self.top_k > 0:
            kth = torch.topk(x, min(self.top_k, int(x.shape[-1])), dim=-1).values[:, -1:]
            x = x.masked_fill(x < kth, float("-inf"))
        if self.top_p < 1.0:
            vals, idx = torch.sort(x, dim=-1, descending=True)
            p = torch.softmax(vals, -1)
            drop = torch.cumsum(p, -1) - p >= self.top_p
            drop[:, 0] = False  # the top token is always kept (top_p 0 is the top token)
            x = x.masked_fill(torch.zeros_like(drop).scatter(-1, idx, drop), float("-inf"))
        return torch.softmax(x, -1)

    def draw_torch(self, logits: torch.Tensor, keys: Sequence[int], k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`k` draws without replacement a row, in draw order (the Gumbel-top-k of the log distribution), and the
        distribution itself: (ids [R, k], probs [R, V])"""
        q = self.dist_torch(logits)
        lq = torch.log(q)
        picks = []
        for r in range(int(q.shape[0])):
            gen = _generator(q.device).manual_seed(int(keys[r]))
            picks.append(torch.topk(lq[r] + _gumbel_torch(int(q.shape[-1]), gen, q.device), k).indices)
        return torch.stack(picks), q

    def dist_mx(self, logits: Any) -> Any:
        """`dist_torch` on the MLX graph"""
        import mlx.core as mx

        x = logits.astype(mx.float32) / max(self.temperature, 1e-6)
        if self.top_k > 0:
            n = min(self.top_k, int(x.shape[-1]))
            kth = mx.take_along_axis(x, mx.argpartition(-x, n - 1, axis=-1)[:, n - 1 : n], axis=-1)
            x = mx.where(x < kth, mx.array(float("-inf"), dtype=mx.float32), x)
        if self.top_p < 1.0:
            order = mx.argsort(-x, axis=-1)
            vals = mx.take_along_axis(x, order, axis=-1)
            p = mx.softmax(vals, axis=-1)
            drop = ((mx.cumsum(p, axis=-1) - p) >= self.top_p) & (mx.arange(int(x.shape[-1])) > 0)[None]
            vals = mx.where(drop, mx.array(float("-inf"), dtype=mx.float32), vals)
            x = mx.put_along_axis(mx.zeros_like(x), order, vals, axis=-1)
        return mx.softmax(x, axis=-1)

    def draw_mx(self, logits: Any, keys: Sequence[int], k: int) -> tuple[Any, Any]:
        """`draw_torch` on the MLX graph: (ids [R, k] in draw order, probs [R, V]), lazily"""
        import mlx.core as mx

        q = self.dist_mx(logits)
        lq = mx.log(q)
        V = int(q.shape[-1])
        noise = mx.stack([mx.random.gumbel((V,), key=mx.random.key(int(key))) for key in keys])
        order = mx.argsort(-(lq + noise), axis=-1)[:, :k]
        return order, q

    # -- torch -----------------------------------------------------------------------------------------------------

    def pick_torch(self, logits: torch.Tensor, keys: Sequence[int]) -> torch.Tensor:
        """the token of each row of `logits` [R, V]: the argmax, or the draw under `keys[r]`; [R] int64. On the
        CPU the draw is the native kernel's (btb_sample_pick) where the library is bound, else torch's"""
        if self.greedy:
            return logits.argmax(-1)
        if logits.device.type == "cpu":
            from .engine.native import Native

            if Native.sample_pick is not None:
                return Native.sample_pick(logits, keys, self.temperature, self.top_k, self.top_p)
        elif logits.device.type == "cuda":
            cu = _card_kernels()
            if cu is not None:
                try:  # the card kernel; fall to the torch path (once-warned) if it cannot launch
                    return cu.pick(logits, keys, self.temperature, self.top_k, self.top_p)
                except Exception as e:
                    _warn_once("pick", e)
        x = logits.float() / self.temperature
        if len(keys) != int(x.shape[0]):
            raise ValueError(f"pick_torch: {len(keys)} keys for {int(x.shape[0])} rows")
        return torch.stack([self._row_torch(x[r], int(keys[r])) for r in range(x.shape[0])])

    def _row_torch(self, x: torch.Tensor, key: int) -> torch.Tensor:
        gen = _generator(x.device).manual_seed(key)
        V = int(x.shape[-1])
        if not self.masked:
            return (x + _gumbel_torch(V, gen, x.device)).argmax()
        if self.top_k > 0:
            vals, idx = torch.topk(x, min(V, self.top_k))
            lse = torch.logsumexp(vals, -1)  # the mass over the top-k when one is set
        else:
            # the pre-cut widened by fours until it holds the nucleus; the mass over the whole row
            lse = torch.logsumexp(x, -1)
            n = min(V, PRECUT)
            while True:
                vals, idx = torch.topk(x, n)
                if n >= V or float(torch.exp(vals - lse).sum()) >= self.top_p:
                    break
                n = min(V, n * 4)
        if self.top_p < 1.0:
            p = torch.exp(vals - lse)
            cum = torch.cumsum(p, -1)
            drop = cum - p >= self.top_p
            drop[0] = False
            vals = vals.masked_fill(drop, float("-inf"))
        # the noise over the candidates in their sorted order: a function of the key and the row's logits alone
        g = _gumbel_torch(int(vals.shape[-1]), gen, x.device)
        return idx[(vals + g).argmax()]

    # -- MLX -------------------------------------------------------------------------------------------------------

    def pick_mx(self, logits: Any, keys: Sequence[int], after: Any = None) -> Any:
        """the same pick inside the MLX graph: `logits` [R, V] lazy, one key a row; [R] int32, lazy. Greedy is
        `mx.argmax`; a draw is the fused kernel (btb/mlx/sample.py), one dispatch over the rows, ordered after
        the array `after` when the logits are a buffer another kernel writes"""
        import mlx.core as mx

        if self.greedy:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        from .mlx import sample

        ids = sample.pick(logits.astype(mx.float32), keys, self.temperature, self.top_k, self.top_p, after=after)
        return ids.astype(mx.int32)

    def pick_mx_ops(self, logits: Any, keys: Sequence[int]) -> Any:
        """the draw as plain MLX ops (argsort, cumsum, gumbel): the kernel's reference, not the engine's path"""
        import mlx.core as mx

        if self.greedy:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        x = logits.astype(mx.float32) / self.temperature
        V = int(x.shape[-1])
        noise = mx.stack([mx.random.gumbel((V,), key=mx.random.key(int(k))) for k in keys])
        if not self.masked:
            return mx.argmax(x + noise, axis=-1).astype(mx.int32)
        if self.top_k > 0:
            n = min(V, self.top_k)
            idx = mx.argpartition(-x, n - 1, axis=-1)[:, :n]
            vals = mx.take_along_axis(x, idx, axis=-1)
            if self.top_p < 1.0:
                order = mx.argsort(-vals, axis=-1)
                vals = mx.take_along_axis(vals, order, axis=-1)
                idx = mx.take_along_axis(idx, order, axis=-1)
        else:
            idx = mx.argsort(-x, axis=-1)
            vals = mx.take_along_axis(x, idx, axis=-1)
        if self.top_p < 1.0:
            p = mx.softmax(vals, axis=-1)
            cum = mx.cumsum(p, axis=-1)
            drop = (cum - p >= self.top_p) & (mx.arange(int(vals.shape[-1])) > 0)[None]
            vals = mx.where(drop, mx.array(float("-inf"), dtype=mx.float32), vals)
        g = mx.take_along_axis(noise, idx, axis=-1)
        j = mx.argmax(vals + g, axis=-1, keepdims=True)
        return mx.take_along_axis(idx, j, axis=-1)[:, 0].astype(mx.int32)


_WARNED: set[str] = set()


def _warn_once(tag: str, exc: BaseException) -> None:
    """the first time a card sampler kernel throws and the draw falls to the torch path, say so on stderr (once a
    tag): the fast path is off and every later token is silently slower, so the fallback must not be quiet"""
    if tag in _WARNED:
        return
    _WARNED.add(tag)
    sys.stderr.write(f"[sample] {tag} failed, falling back to the slow torch path: {type(exc).__name__}: {exc}\n")
    sys.stderr.flush()


def _card_kernels() -> Any:
    """the card's fused kernels where they are built and load, else None - a draw on cuda then falls to the
    per-row torch path; the engine's card path holds the same handle"""
    from .engine.native import Native

    return Native.card_kernels()


def _gumbel_torch(n: int, gen: torch.Generator, device: Any) -> torch.Tensor:
    u = torch.rand(n, generator=gen, device=device)
    return u.clamp_(1e-20, 1.0).log_().neg_().log_().neg_()


class Verify:
    """The pick of a verify pass whose rows are tree nodes with drafted children: at a node the children are tried
    in draw order, child i accepted with probability min(1, p_i(c) / q_i(c)) where p_i is the target after the
    earlier rejections' residuals (p_{i+1} = norm(max(0, p_i - q_i))) and q_i the drafter's distribution with
    the earlier children removed; none accepted, a draw from the residual. Exact: the emitted token is a draw
    from p at every node. A row without a distribution (its drafts point masses) draws from p and accepts the
    child that equals the draw. `pick_*` return packed outcomes: (slot + 1) << 24 | token, slot -1 none."""

    greedy = False

    def __init__(self, sampling: Sampling, q: Any, kids: TokenRows, hasq: Sequence[bool]) -> None:
        self.sampling = sampling
        self.q = q  # [T, Vd] the rows' distributions (torch or mx), zeros where hasq is False
        self.kids = [list(k) for k in kids]  # a row's children tokens in draw order
        self.hasq = [bool(h) for h in hasq]
        self.width = max((len(k) for k in self.kids), default=0)

    @property
    def temperature(self) -> float:
        return self.sampling.temperature

    def key_for(self, pos: int) -> int:
        return self.sampling.key_for(pos)

    @staticmethod
    def unpack(v: int) -> tuple[int, int]:
        """(the accepted child's slot, -1 for none; the token)"""
        v = int(v)
        return (v >> 24) - 1, v & 0xFFFFFF

    def _kids_padded(self) -> list[list[int]]:
        w = max(1, self.width)  # a pass without drafts still carries one (empty) slot a row
        return [[*k, *([-1] * (w - len(k)))] for k in self.kids]

    def pick_mx(self, logits: Any, keys: Sequence[int], after: Any = None) -> Any:
        import mlx.core as mx

        from .mlx import sample

        kids = mx.array(self._kids_padded(), dtype=mx.int32).reshape(len(self.kids), max(1, self.width))
        hasq = mx.array([1 if h else 0 for h in self.hasq], dtype=mx.uint32)
        s = self.sampling
        q = self.q if not isinstance(self.q, torch.Tensor) else mx.array(self.q.float().numpy())
        out = sample.verify(logits.astype(mx.float32), q, kids, hasq, keys, s.temperature, s.top_k, s.top_p, after)
        return out.astype(mx.int32)  # packed values sit under 2^31; torch has no uint32

    def pick_torch(self, logits: torch.Tensor, keys: Sequence[int]) -> torch.Tensor:
        """the outcomes: on the card the fused verify kernel, else torch (the kernel's reference, its noise the
        kernel's hash of (key, token)); `logits` [T, V], `keys` one a row; [T] int64 packed"""
        s = self.sampling
        if logits.device.type == "cuda":
            cu = _card_kernels()
            if cu is not None:
                kids = torch.tensor(self._kids_padded(), dtype=torch.int32, device=logits.device).reshape(
                    len(self.kids), max(1, self.width)
                )
                hasq = torch.tensor([1 if h else 0 for h in self.hasq], dtype=torch.int32, device=logits.device)
                q = self.q if isinstance(self.q, torch.Tensor) else torch.from_numpy(__import__("numpy").array(self.q))
                try:  # fall to the torch path (once-warned) if the kernel cannot launch
                    return cu.verify(logits, q.to(logits.device), kids, hasq, keys, s.temperature, s.top_k, s.top_p)
                except Exception as e:
                    _warn_once("verify", e)
        p_all = s.dist_torch(logits)
        q_all = self.q if isinstance(self.q, torch.Tensor) else torch.from_numpy(__import__("numpy").array(self.q))
        out = []
        for r in range(int(logits.shape[0])):
            p = p_all[r].clone()
            key = int(keys[r])
            V = int(p.shape[-1])
            noise = _hash_gumbel(key, torch.arange(V, device=p.device))
            if not self.hasq[r]:
                t = int((torch.log(p) + noise).argmax())
                slot = self.kids[r].index(t) if t in self.kids[r] else -1
                out.append(((slot + 1) << 24) | t)
                continue
            q = torch.zeros(V, dtype=torch.float32, device=p.device)
            q[: q_all.shape[-1]] = q_all[r].float()
            zq = float(q.sum())
            slot = -1
            for i, c in enumerate(self.kids[r]):
                # a child without drafter mass is no draw of its: the trials end; a residual without mass means
                # p_i = q_i (the rejection was rounding), where the trial accepts
                qc = min(float(q[c]) / zq, 1.0) if zq > 0 else 0.0
                if not qc > 0:
                    break
                u = float(_hash_uniform(key, torch.tensor([0xF00000 + i], device=p.device))[0])
                if u * qc < float(p[c]):
                    slot = i
                    break
                rest = torch.clamp(p - q / zq, min=0.0)
                if not float(rest.sum()) > 0:
                    slot = i
                    break
                p = rest / rest.sum()
                zq -= float(q[c])
                q[c] = 0.0
            if slot >= 0:
                out.append(((slot + 1) << 24) | self.kids[r][slot])
                continue
            out.append(int((torch.log(p) + noise).argmax()))
        return torch.tensor(out, dtype=torch.int64)


_H1 = 0x9E3779B97F4A7C15 - (1 << 64)  # the kernels' hash constants as signed 64-bit (torch has no uint64)
_H2 = 0xBF58476D1CE4E5B9 - (1 << 64)
_H3 = 0x94D049BB133111EB - (1 << 64)


def _shr(h: torch.Tensor, k: int) -> torch.Tensor:
    """a logical right shift of int64 by k >= 1 (torch's shift is arithmetic)"""
    return (h >> k) & ((1 << (64 - k)) - 1)


def _hash64(key: int, idx: torch.Tensor) -> torch.Tensor:
    """the kernels' hash of (key, token) (btb/mlx/sample.py `uniform_of`, native/src/sample.rs, the CUDA
    `smp_uniform`): splitmix over the pair; int64 over `idx`, wrapping arithmetic the kernels' uint64"""
    h = (idx.to(torch.int64) * _H1) ^ int(key)
    h = (h ^ _shr(h, 32)) * _H2
    h = (h ^ _shr(h, 29)) * _H3
    return h ^ _shr(h, 32)


def _hash_uniform(key: int, idx: torch.Tensor) -> torch.Tensor:
    """the kernels' uniform of (key, token): the hash's top 23 bits, in (0, 1) - the top of a 24-bit range rounds
    to 1.0 in float32, an infinite Gumbel that wins the row; float32 over `idx`"""
    return (_shr(_hash64(key, idx), 41).to(torch.float32) + 0.5) * (1.0 / 8388608.0)


def _hash_gumbel(key: int, idx: torch.Tensor) -> torch.Tensor:
    return -torch.log(-torch.log(_hash_uniform(key, idx)))


_gens: dict[Any, torch.Generator] = {}


def _generator(device: Any) -> torch.Generator:
    """one generator a device, reseeded per pick (a fresh one per pick cost as much as the draw)"""
    g = _gens.get(device)
    if g is None:
        g = _gens[device] = torch.Generator(device=device)
    return g


GREEDY = Sampling()


def as_pick(pick: Any) -> Sampling | None:
    """the `pick` argument of a forward: None or False for the logits, True for the greedy ids, a Sampling as is"""
    if pick is None or pick is False:
        return None
    return GREEDY if pick is True else pick
