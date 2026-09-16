# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""The receipts: the engine over the tiny fixtures against the tensors make_fixtures banked, on the CPU and on MLX,
bf16 and through the 12-bit store, with and without cold layers; gpt-oss against transformers' own forward."""

import os
import sys
from collections.abc import Iterable

import torch

from btb.engine import StreamedTextModel
from tests.helpers import (
    FIXTURES,
    NO_LOG,
    PROMPT_DENSE,
    PROMPT_Q35,
    fixture,
    forward_logits,
    host_model,
    layer_count,
    max_abs,
    native_library,
    receipts,
    speculation,
    tree_next,
    tree_pass,
)


def run(packed: bool, device: str = "cpu", cold: tuple[int, ...] = ()) -> tuple[float, bool, bool]:
    """the q35 hybrid: the tree pass, the DeltaNet states along the committed path and the next step against the
    receipts; (the worst distance, the greedy answer is the banked one, every proposer's answer is the greedy one)"""
    B = receipts("q35")["host"]
    with torch.inference_mode():
        sm = host_model(
            fixture("tiny_q35-pack12" if packed else "tiny_q35"), device=device, packed=packed, cold_layers=cold
        )
        cache = sm.new_cache()
        forward_logits(sm, PROMPT_Q35, cache)
        lg, _ = tree_pass(sm, cache)
        d = [max_abs(lg, B["logits"])]
        for i in B["states"]:
            conv, rec = StreamedTextModel._lin(cache.layers[i])
            d.append(max_abs(conv, B["states"][i][0]) + max_abs(rec, B["states"][i][1]))
        d.append(max_abs(tree_next(sm, cache), B["next"]))
        speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.9, tree_read="step")
        g = sm.generate_greedy(PROMPT_Q35, 10)
        same = all(
            sm.generate_speculative(PROMPT_Q35, 10, proposer=p, v_max=4)[0] == g
            for p in ("ngram", "mtp", "mtp_tree", "mtp_dyn")
        )
        sm.close()
    return max(d), g == B["toks"]["greedy"], same


def run_dense(tag: str, packed: bool, device: str = "cpu", cold: tuple[int, ...] = ()) -> tuple[float, bool, bool]:
    """a dense family: the prompt's logits, the tree pass, the next step and every layer's K/V against the receipts"""
    from btb.engine import pack_model

    fx = fixture(f"tiny_{tag}")
    fxp = os.path.join(FIXTURES, f"tiny_{tag}-pack12")
    if packed and not os.path.exists(fxp):
        pack_model(fx, log=NO_LOG)
    B = receipts(tag)["p12" if packed else "bf16"]
    with torch.inference_mode():
        sm = host_model(fxp if packed else fx, device=device, packed=packed, cold_layers=cold)
        cache = sm.new_cache()
        lg0 = forward_logits(sm, PROMPT_DENSE, cache)[0, -1]
        lg, _ = tree_pass(sm, cache)
        nxt = tree_next(sm, cache)
        d = [max_abs(lg0, B["prompt_logits"]), max_abs(lg, B["chunk_logits"]), max_abs(nxt, B["next"])]
        for i in range(sm.L):
            d.append(max_abs(cache.layers[i].keys, B["keys"][i][0]) + max_abs(cache.layers[i].values, B["keys"][i][1]))
        speculation(sm, tree_budget=0, tree_min_prob=0.0, ngram_p=0.0, tree_read="step")
        g = sm.generate_greedy(PROMPT_DENSE, 10)
        same = sm.generate_speculative(PROMPT_DENSE, 10, proposer="ngram", v_max=4)[0] == g
        sm.close()
    return max(d), g == B["greedy"], same


def run_q4(device: str = "cpu") -> tuple[float, bool, bool]:
    """the qwen4 mixture: the prompt and a continuation against the receipts, through the whole expert store and
    through one too small to hold a call's experts"""
    fx = fixture("tiny_q4")
    StreamedTextModel.register_attention()
    B = receipts("q4")["host"]
    d: list[float] = []
    same = True
    with torch.inference_mode():
        for store_gb in (None, 1e-9):
            sm = host_model(fx, device=device, expert_cache_gb=store_gb)
            speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
            cache = sm.new_cache()
            lg0 = forward_logits(sm, B["prompt"], cache)[0, -1]
            lg1 = forward_logits(sm, B["cont"], cache)[0, -1]
            g = sm.generate_greedy(B["prompt"], 10)
            if (
                store_gb is not None
                and sm.expert_store is not None
                and sm.expert_store.n_slots >= sm.L * int(sm.cfg.num_experts)
            ):
                raise RuntimeError("the small-store arm did not evict")
            sm.close()
            d += [max_abs(lg0, B["prompt_logits"]), max_abs(lg1, B["next"])]
            same = same and g == B["greedy"]
    return max(d), same, True


# an expert-store size, the CPU layers, the resident layers, the cold layers and the prefill chunk of one placement
_Arm = tuple[float | None, Iterable[int], Iterable[int], Iterable[int], int | None]


def run_gpt_oss(device: str = "cpu") -> tuple[float, bool, bool]:
    """gpt-oss: attention sinks, alternating windows, the yarn rope and a router with a bias on the trunk,
    and the MXFP4 experts multiplied in the form they are stored in - through the expert store, through a
    store too small to hold one call's, and straight from the checkpoint with no store - over every
    placement the engine has for them. The receipts are transformers' own float32 forward over the same
    weights, with the experts dequantized by `transformers.integrations.mxfp4`."""
    fx = fixture("tiny_gpt_oss")
    StreamedTextModel.register_attention()
    B = receipts("gpt_oss")["host"]
    d: list[float] = []
    same = True
    L = range(layer_count(fx))
    # the whole store, a store too small to hold one call's experts, no store at all, the layers streamed
    # from templates instead of built per layer, layers read from the drive each pass, and the trunk
    # resident with the prompt prefilled layer by layer (each layer's experts read once)
    arms: list[_Arm] = [
        (None, L, (), (), None),
        (1e-9, L, (), (), None),
        (0, L, (), (), None),
        (None, (), (), (), None),
        (None, L, (), (1, 2), None),
        (None, (), L, (), 6),
    ]
    if device != "cpu":
        # the GPU tier takes the host layers (the experts from the store's shared slots, the no-store arm
        # stays on the CPU kernels): the placements that put every layer on it
        arms = [a for a in arms if a[1] == L]
    with torch.inference_mode():
        for store_gb, cpu, resident, cold, chunk in arms:
            sm = host_model(
                fx,
                device=device,
                cpu_layers=cpu,
                resident_layers=resident,
                cold_layers=cold,
                expert_cache_gb=store_gb,
                prefill_chunk=chunk,
            )
            speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
            cache = sm.new_cache()
            lg0 = sm._prefill(B["prompt"], cache)[0, -1]
            lg1 = forward_logits(sm, B["cont"], cache)[0, -1]
            g = sm.generate_greedy(B["prompt"], 10)
            # the drafter is handed the banked answer as a span, so the verify pass really runs several
            # positions at once over the sliding and the full layers and `ad` crops what was rejected
            s, census = sm.generate_speculative(
                B["prompt"], 10, proposer="ngram", v_max=4, spans=[("receipt", B["greedy"])]
            )
            same = same and g == B["greedy"] and s == g and census["accepted"] > 0
            if store_gb is None and sm.expert_store is None:
                raise RuntimeError("the expert store did not open")
            if store_gb == 1e-9:
                assert sm.expert_store is not None, "the small-store arm did not open"
                if sm.expert_store.n_slots >= sm.L * int(sm.n_experts):
                    raise RuntimeError("the small-store arm did not evict")
            if store_gb == 0 and sm.expert_store is not None:
                raise RuntimeError("expert_cache_gb=0 still opened a store")
            sm.close()
            d += [max_abs(lg0, B["prompt_logits"]), max_abs(lg1, B["next"])]
    return max(d), same, True


def _avx2() -> bool:
    import platform

    if platform.machine().lower() not in ("amd64", "x86_64"):
        return False
    if sys.platform == "win32":
        return True
    try:
        if sys.platform == "darwin":
            import subprocess

            return (
                "AVX2"
                in subprocess.run(["sysctl", "-n", "machdep.cpu.leaf7_features"], capture_output=True, text=True).stdout
            )
        return "avx2" in open("/proc/cpuinfo").read()
    except Exception:
        return False


def main() -> int:
    from btb import mlx_available

    dll = native_library()
    # the receipts were banked by the AVX2 kernel; the scalar kernel and MLX sum in other orders, so those arms
    # are held to a tolerance. The AVX2 arm is held to four ulps of a logit, not the bit: the head's gemv and
    # torch's AVX-512 reductions on a machine that has them sum in another order than the banking machine's
    tol = {"cpu": 4e-6 if _avx2() else 1e-5, "mlx": 1e-4}
    # gpt-oss's receipts are transformers' own float32 forward, so that arm is held to a tolerance on every
    # machine; it covers the order of the sums only, the MXFP4 dequantization being exact (tests/test_mxfp4.py)
    ref_tol = {"cpu": 1e-5, "mlx": 1e-4}
    devices = ["cpu"] + (["mlx"] if mlx_available() else [])
    ok = True
    worst: dict[str, float] = {}
    worst_ref: dict[str, float] = {}
    for device in devices:
        colds = [()] if device == "cpu" else [(), "cold"]
        for packed in (False, True):
            for cold in colds:
                w, g_ok, same = run(packed, device=device, cold=(2, 3, 5) if cold else ())
                worst[device] = max(worst.get(device, 0.0), w)
                ok = ok and g_ok and same
                print(
                    f"{device} {'p12' if packed else 'bf16'}{' cold' if cold else ''} "
                    f"{'native' if dll else 'torch'}: {w:.3e} {g_ok} {same}",
                    flush=True,
                )
        for tag in ("phi3", "qwen3"):
            for packed in (False, True):
                for cold in colds:
                    w, g_ok, same = run_dense(tag, packed, device=device, cold=(1, 3) if cold else ())
                    worst[device] = max(worst.get(device, 0.0), w)
                    ok = ok and g_ok and same
                    print(
                        f"{device} {tag} {'p12' if packed else 'bf16'}{' cold' if cold else ''} "
                        f"{'native' if dll else 'torch'}: {w:.3e} {g_ok} {same}",
                        flush=True,
                    )
        w, g_ok, same = run_q4(device=device)
        worst[device] = max(worst.get(device, 0.0), w)
        ok = ok and g_ok and same
        print(f"{device} q4 bf16 {'native' if dll else 'torch'}: {w:.3e} {g_ok} {same}", flush=True)
        # the MXFP4 experts through the CPU kernels, and on the MLX tier through the GPU's matvec over the
        # store's shared slots
        w, g_ok, same = run_gpt_oss(device=device)
        worst_ref[device] = max(worst_ref.get(device, 0.0), w)
        ok = ok and g_ok and same
        print(
            f"{device} gpt_oss mxfp4 {'native' if dll else 'torch'}: {w:.3e} {g_ok} {same} (vs transformers)",
            flush=True,
        )
    within = all(worst[d] <= tol[d] for d in devices) and all(w <= ref_tol[d] for d, w in worst_ref.items())
    print(
        " ".join(f"{d} worst {worst[d]:.3e} (tolerance {tol[d]:.0e})" for d in devices)
        + "".join(f" {d} vs transformers {w:.3e} (tolerance {ref_tol[d]:.0e})" for d, w in worst_ref.items())
        + f" ok {ok and within}",
        flush=True,
    )
    return 0 if (ok and within) else 1


def test_receipts() -> None:
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
