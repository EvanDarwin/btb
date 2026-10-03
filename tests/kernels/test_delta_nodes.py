# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The card's gated DeltaNet over a pass's nodes (`btb_delta_nodes`, native/cuda/btb_kernels.cu): a chain steps as
the host's `btb_delta_step` does, every node of a tree computes bit for bit as the chain of its own path from the
same starting states, a node's result does not depend on how many nodes the pass carries, and a malformed call is
refused before it launches."""

from __future__ import annotations

import pytest
import torch

from btb.engine.forward import path_of
from btb.engine.native import Native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the card's kernel")

# (key heads, value heads, key dim, value dim, conv taps): a small ragged head, and the 180B's
SHAPES = [(2, 6, 24, 40, 4), (16, 48, 128, 128, 4)]
PARENTS = [-1, 0, 1, 2, 1, 4, 0, 6, 7, 3]


def _inputs(hk: int, hv: int, dk: int, dv: int, K: int, T: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    C = 2 * hk * dk + hv * dv
    r = lambda *s, scale=1.0: torch.randn(*s, generator=g) * scale
    return {
        "mixed": r(T, C),
        "z": r(T, hv * dv),
        "a": r(T, hv),
        "b": r(T, hv),
        "conv_w": r(C, K, scale=0.5),
        "conv0": r(C, K),
        "state": r(hv, dk, dv, scale=0.1),
        "a_log": r(hv, scale=0.5),
        "dt_bias": r(hv, scale=0.5),
        "norm_w": 1.0 + r(dv, scale=0.1),
    }


def _card(
    kern: object,
    x: dict[str, torch.Tensor],
    shape: tuple[int, ...],
    gate: int,
    parents: list[int] | None,
    rows: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(out [T, hv * dv], the states: the scratch's per node with `parents`, else the chain's final one)"""
    hk, hv, dk, dv, _K = shape
    pick = (lambda t: t) if rows is None else (lambda t: t[rows])
    d = {k: v.cuda().contiguous() for k, v in x.items()}
    mixed, z, a, b = (pick(d[k]).contiguous() for k in ("mixed", "z", "a", "b"))
    T = int(mixed.shape[0])
    out = torch.empty(T, hv * dv, device="cuda")
    state = d["state"].clone()
    scratch = torch.empty(T, hv, dk, dv, device="cuda") if parents is not None else None
    par = torch.tensor(parents, dtype=torch.int32, device="cuda") if parents is not None else None
    kern.delta_nodes(  # type: ignore[attr-defined]
        mixed,
        z,
        a,
        b,
        d["conv_w"],
        None,
        d["conv0"],
        state,
        scratch,
        par,
        d["a_log"],
        d["dt_bias"],
        d["norm_w"],
        1e-6,
        gate,
        hk,
        hv,
        dk,
        dv,
        out,
    )
    torch.cuda.synchronize()
    return out.cpu(), (scratch if scratch is not None else state).cpu()


@pytest.fixture(scope="module")
def kern() -> object:
    k = Native.card_kernels()
    if k is None:
        pytest.skip("no card kernels")
    return k


@pytest.mark.parametrize("gate", [0, 1])
@pytest.mark.parametrize("shape", SHAPES)
def test_a_chain_steps_as_the_hosts_kernel(kern: object, shape: tuple[int, ...], gate: int) -> None:
    if Native.delta_step is None:
        pytest.skip("no native library")
    hk, hv, dk, dv, K = shape
    x = _inputs(hk, hv, dk, dv, K, 5, seed=11 + gate)
    out, state = _card(kern, x, shape, gate, None)
    conv, st = x["conv0"].clone(), x["state"].clone()
    host = torch.empty(5, hv * dv)
    for j in range(5):
        Native.delta_step(
            x["mixed"][j].clone(),
            conv,
            x["conv_w"],
            None,
            x["z"][j],
            x["a"][j],
            x["b"][j],
            x["a_log"],
            x["dt_bias"],
            st,
            hk,
            hv,
            dk,
            dv,
            x["norm_w"],
            1e-6,
            host[j],
            gate,
        )
    assert torch.allclose(out, host, rtol=1e-4, atol=1e-5), float((out - host).abs().max())
    assert torch.allclose(state, st, rtol=1e-4, atol=1e-5), float((state - st).abs().max())


@pytest.mark.parametrize("shape", SHAPES)
def test_every_node_of_a_tree_is_the_chain_of_its_path(kern: object, shape: tuple[int, ...]) -> None:
    hk, hv, dk, dv, K = shape
    x = _inputs(hk, hv, dk, dv, K, len(PARENTS), seed=7)
    out, states = _card(kern, x, shape, 1, PARENTS)
    for j in range(len(PARENTS)):
        path = path_of(PARENTS, j)[::-1]
        o, s = _card(kern, x, shape, 1, None, rows=path)
        assert torch.equal(out[j], o[-1]), f"node {j}: output parts from its path's chain"
        assert torch.equal(states[j], s), f"node {j}: state parts from its path's chain"


def test_a_nodes_result_does_not_depend_on_the_pass_width(kern: object) -> None:
    shape = SHAPES[1]
    hk, hv, dk, dv, K = shape
    x = _inputs(hk, hv, dk, dv, K, len(PARENTS), seed=3)
    full, _ = _card(kern, x, shape, 1, PARENTS)
    for T in (1, 4, 7):
        part, _ = _card(kern, {**x, **{k: x[k][:T] for k in ("mixed", "z", "a", "b")}}, shape, 1, PARENTS[:T])
        assert torch.equal(part, full[:T]), f"a pass of {T} nodes parts from the wider one"


def test_a_malformed_call_is_refused(kern: object) -> None:
    hk, hv, dk, dv, K = SHAPES[0]
    x = {k: v.cuda() for k, v in _inputs(hk, hv, dk, dv, K, 2, seed=1).items()}
    out = torch.empty(2, hv * dv, device="cuda")
    call = lambda **kw: kern.delta_nodes(  # type: ignore[attr-defined]
        **{
            "mixed": x["mixed"],
            "z": x["z"],
            "a": x["a"],
            "b": x["b"],
            "conv_w": x["conv_w"],
            "conv_b": None,
            "conv0": x["conv0"],
            "state": x["state"],
            "scratch": None,
            "parents": None,
            "a_log": x["a_log"],
            "dt_bias": x["dt_bias"],
            "norm_w": x["norm_w"],
            "eps": 1e-6,
            "gate": 0,
            "hk": hk,
            "hv": hv,
            "dk": dk,
            "dv": dv,
            "out": out,
            **kw,
        }
    )
    with pytest.raises(ValueError, match="float32"):
        call(z=x["z"].bfloat16())
    with pytest.raises(ValueError, match="shapes"):
        call(hk=hk + 1)
    with pytest.raises(ValueError, match="shapes"):
        call(gate=2)
    with pytest.raises(ValueError, match="parents"):
        call(parents=torch.zeros(2, dtype=torch.int64, device="cuda"))
