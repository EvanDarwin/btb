# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""btb never runs the card out: every path either fits what the device's ledger has, or is refused by name
(`MemoryGrantError`) before anything is allocated. torch's allocator failing an allocation - and emptying its cache
to retry it, a device-wide sync and fresh allocations after (`num_alloc_retries`), or giving up (`num_ooms`) - is an
allocation btb made without the ledger knowing, or priced short: each path here runs with the card squeezed to a
little past btb's margin, a little more, and plenty, and the allocator's counters must not move at any of them.

The paths are the ones that allocate as they go: a prefill layer by layer through the card (a mixture's depot and
grouped calls, gpt-oss's materialized attention, a dense model's and a hybrid's templates), a session fed and
decoded, a fork stepped and kept, a batch of sessions, and a tensor lent to the caller."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pytest
import torch

from btb.engine.scheduler import MemoryGrantError
from tests.helpers import fixture, loaded_model, need_cuda

if TYPE_CHECKING:
    from btb.engine.model import StreamedTextModel

MiB = 2**20
# 40 tokens inside every fixture's vocabulary and context: five chunks of 8, each through the card
PROMPT = [(7 * i + 3) % 200 + 2 for i in range(40)]
# what btb has free as it counts it, past its margin: nothing, a little, and plenty
LEFT = (0, 8 * MiB, 64 * MiB, 1024 * MiB)
# what the host keeps while the card is squeezed (`squeezed`): the paths' host growth and staging, with room
HOST_ROOM = 512 * MiB
# the squeezes take the card from every program on it (`squeezed`): only on a card set aside for them
SQUEEZE = os.environ.get("BTB_SQUEEZE_CARD") == "1"
FAMILIES = ["tiny_q4", "tiny_gpt_oss", "tiny_qwen3", "tiny_q35"]


def _allocator(dev: torch.device) -> tuple[int, int]:
    ms = torch.cuda.memory_stats(dev)
    return int(ms.get("num_alloc_retries", 0)), int(ms.get("num_ooms", 0))


@contextlib.contextmanager
def squeezed(sm: StreamedTextModel, left: int) -> Iterator[None]:
    """the card with only `left` bytes free past btb's margin, as its ledger reads it: the rest taken by a tensor
    of the test's own, which btb sees as any other program's use of the card. On Windows a card allocation is
    charged to the host's commit as well (WDDM), so a filler taken whole would starve the host too - the paths'
    host growth then refused, rightly, and "plenty left" not what the test meant (8.6 GB of filler took the
    machine from 11.6 to 3.0 GB of commit left, btb's host figure to 0.08 GB). There the filler leaves the host
    `HOST_ROOM` of what it had, and the card a little more than `left`.

    The filler takes the card's free memory as every program on it sees it, not only btb's: a game or anything else
    on the card is left nothing to grow into, and can fail or crash. So it runs only on a card set aside for it,
    `BTB_SQUEEZE_CARD=1`; otherwise the test is skipped saying so."""
    if not SQUEEZE:
        pytest.skip("squeezes the whole card, whatever else runs on it: BTB_SQUEEZE_CARD=1 on a card set aside for it")
    torch.cuda.synchronize(sm.dev)
    torch.cuda.empty_cache()
    take = max(0, int(sm.device.free(sm.dev, unreserved=True) or 0) - int(left))
    if sys.platform == "win32":
        host = int(sm.device.free(torch.device("cpu"), unreserved=True) or 0)
        take = min(take, max(0, host - HOST_ROOM))
    filler = torch.empty(take, dtype=torch.uint8, device=sm.dev) if take else None
    try:
        yield
    finally:
        del filler
        torch.cuda.synchronize(sm.dev)
        torch.cuda.empty_cache()


def _prefill(sm: StreamedTextModel) -> None:
    sm.generate(list(PROMPT), 2, eos=(), speculate=False)


def _session(sm: StreamedTextModel) -> None:
    s = sm.session(PROMPT[:24])
    s.feed(PROMPT[24:])
    s.generate(3, eos=(), speculate=False)


def _fork(sm: StreamedTextModel) -> None:
    s = sm.session(PROMPT)
    br = s.fork(3)
    br.step([5, 6, 7])
    br.step([8, 9, 10])
    br.keep(1)


def _batch(sm: StreamedTextModel) -> None:
    a, b = sm.session(PROMPT[:20]), sm.session(PROMPT[20:])
    with sm.batch([a, b]) as bt:
        bt.generate(3, eos=())


def _lend(sm: StreamedTextModel) -> None:
    t = sm.zeros((16 * MiB,), dtype=torch.uint8)
    del t


@contextlib.contextmanager
def told(sm: StreamedTextModel, mp: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """what the ledger was told on the card during the block: the bytes granted less the buffers they replace, the
    most it held reserved at once past what it held before, and the room asked of it before an allocation"""
    t = {"granted": 0, "reserved": 0, "asked": 0}
    base = int(sm.device.reserved(sm.dev))
    sched, dv = sm.scheduler, sm.device
    grant, reserve, make = sched.grant, dv.reserve, sm._make_room

    def g(nbytes: int, kind: str, **kw: Any) -> None:
        grant(nbytes, kind, **kw)
        if torch.device(kw.get("device") or sm.dev).type == "cuda":
            t["granted"] += max(0, int(nbytes) - int(kw.get("held", 0)))

    def r(tag: str, nbytes: int, device: Any = None, **kw: Any) -> None:
        reserve(tag, nbytes, device, **kw)
        t["reserved"] = max(t["reserved"], int(dv.reserved(sm.dev)) - base)

    def m(dev: torch.device, nbytes: int, what: str, own: str | None = None) -> set[str]:
        if torch.device(dev).type == "cuda":
            t["asked"] += int(nbytes)
        return make(dev, nbytes, what, own)

    with mp.context() as ctx:
        ctx.setattr(sched, "grant", g)
        ctx.setattr(dv, "reserve", r)
        ctx.setattr(sm, "_make_room", m)
        yield t


# what a pass holds that no grant or reservation covers: its activations, priced by the chunk size (`_chunk_bytes`,
# a whole prompt's rows here) and never reserved, and the allocator's rounding
SLACK = 1 * MiB


PATHS: dict[str, Callable[[StreamedTextModel], None]] = {
    "prefill": _prefill,
    "session": _session,
    "fork": _fork,
    "batch": _batch,
    "lend": _lend,
}


@pytest.mark.parametrize("fx", FAMILIES)
def test_every_path_fits_or_is_refused_and_never_runs_the_card_out(fx: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """one load a family: every path's card peak held to what the ledger was told, then every path at every
    squeeze, generous first (what a squeeze shed stays shed after) - on a card set aside for it (`SQUEEZE`)"""
    dev = need_cuda()
    # a host layer prefilled on the card in chunks of 8 (the card's floor at 4 rows), layer by layer
    kw: dict[str, Any] = {"device": dev, "cpu_layers": 1, "prefill_chunk": 8, "prefill_card_min": 4}
    ran_out: list[str] = []
    idle: list[str] = []
    untold: list[str] = []
    with loaded_model(fixture(fx), **kw) as sm:
        passes = sm._chunk_bytes(len(PROMPT))
        for path, run in PATHS.items():
            run(sm)  # the path once with the card as it is: the kernels' own first allocations made
            # the card at its peak over the path holds no more than the ledger was told of, and the pass's own rows
            torch.cuda.synchronize(sm.dev)
            base = torch.cuda.memory_allocated(sm.dev)
            torch.cuda.reset_peak_memory_stats(sm.dev)
            with told(sm, monkeypatch) as t:
                run(sm)
            torch.cuda.synchronize(sm.dev)
            peak = torch.cuda.max_memory_allocated(sm.dev) - base
            covered = t["granted"] + t["reserved"] + t["asked"] + passes + SLACK
            if peak > covered:
                untold.append(
                    f"{path}: the card peaked {peak / MiB:.2f} MiB past its start; the ledger was told of "
                    f"{(t['granted'] + t['reserved'] + t['asked']) / MiB:.2f} MiB {t} beside the pass's "
                    f"{passes / MiB:.2f} MiB of rows"
                )
            if not SQUEEZE:
                continue  # the peak against the ledger holds on any card; the squeezes need one set aside
            outcomes = []
            for left in sorted(LEFT, reverse=True):
                with squeezed(sm, left):
                    before = _allocator(sm.dev)
                    try:
                        run(sm)
                        outcomes.append((left, "ran"))
                    except MemoryGrantError as e:
                        outcomes.append((left, f"refused: {e}"))
                    torch.cuda.synchronize(sm.dev)
                    after = _allocator(sm.dev)
                if after != before:
                    ran_out.append(
                        f"{path} with {left / MiB:.0f} MiB left: retries, ooms {before} -> {after} ({outcomes[-1][1]})"
                    )
            if outcomes[0][1] != "ran":
                idle.append(f"{path} did not run with plenty left: {outcomes[0][1]}")
    assert not ran_out, f"{fx}: torch's allocator ran out where btb should have fit or refused:\n" + "\n".join(ran_out)
    assert not idle, f"{fx}:\n" + "\n".join(idle)
    assert not untold, f"{fx}: memory on the card the ledger was never told of:\n" + "\n".join(untold)


def test_a_presize_refused_after_the_sweep_made_room_sheds_from_the_top_and_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the sweep makes room for every resident layer's rows, then another program takes the card's memory before
    they are presized (a game growing through a 40k prompt's prefill refused the tenth layer's rows and the prefill
    raised). A refused layer gives up the top resident layer - no rows of its there yet - and asks again, until the
    rest fits or it is the one given up; with `adapt` off the placement is pinned and the refusal stands. And a host
    layer's first chunk on the card starts its rows in the sweep's reservation, a buffer for the chunk alone: grown
    there, it took the whole sequence's rows out of the epoch's room (0.16 GiB at 40k on Qwen3-0.6B, refused)"""
    dev = need_cuda()
    kw: dict[str, Any] = {"device": dev, "cpu_layers": 1, "prefill_chunk": 8, "prefill_card_min": 4}
    with loaded_model(fixture("tiny_qwen3"), **kw) as sm:
        top = sorted(sm.resident)
        assert len(top) >= 3, top
        real = sm.scheduler.grant
        asked: list[str] = []
        # another session holds the card graph's arena, so the prefill's cache keeps rows of its own (a cache taking
        # the free arena writes its rows there, and nothing of it is presized)
        holder = sm.session(PROMPT[:8])
        assert holder.cache is not None
        st = sm._card_state()
        if st.get("arena") is not None:
            with torch.inference_mode():
                sm._card_bind(holder.cache, st, 0)
            assert st["arena"]["owner"]() is holder.cache, "the holder did not take the arena"

        def sweep() -> None:
            # the prompt's prefill alone: a decode step would take the arena from the holder
            with torch.inference_mode():
                sm._prefill(torch.tensor([PROMPT]), sm.new_cache(max_len=len(PROMPT) + 2))

        started: list[str] = []

        def seen(nbytes: int, kind: str, **g: Any) -> None:
            who = str(g.get("requester", ""))
            if "(layer cache) n=0 " in who and torch.device(g.get("device") or sm.dev).type == "cuda":
                started.append(who)
            real(nbytes, kind, **g)

        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", seen)
            sweep()
        assert not started, f"a layer's rows started on the card by a growth, not in the sweep's reservation: {started}"

        def taken(refusals: int) -> Callable[..., None]:
            # the room gone the first `refusals` times a layer asks for the prompt's rows, as when another program
            # took it since the sweep's room was made
            def grant(nbytes: int, kind: str, **g: Any) -> None:
                if "the prompt's rows at once" in str(g.get("requester", "")):
                    asked.append(str(g["requester"]))
                    if len(asked) <= refusals:
                        raise MemoryGrantError("[grant] REFUSED the test's presize: another program took the room")
                real(nbytes, kind, **g)

            return grant

        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", taken(1))
            mp.setattr(sm, "adapt", False)
            with pytest.raises(MemoryGrantError, match="another program took the room"):
                sweep()
        assert sorted(sm.resident) == top, "a pinned placement gave a layer up"
        asked.clear()
        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", taken(2))
            sweep()
        assert len(asked) >= 3, (asked, sorted(sm.resident), sorted(sm.host))  # two refused, then the lowest granted
        assert sorted(sm.resident) == top[:-2], "each refusal gives up the top resident layer, and no more"
        assert {top[-1], top[-2]} <= set(sm.host)
        del holder


def test_a_prefills_rows_are_made_once_in_the_arena_and_a_refused_arena_decodes_on_the_torch_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """with the card graphs' arena free, a prefill's cache takes it grown to its reach and its rows are made there,
    once - presized in buffers of its own, the first decode step asked a second whole copy to move them in (3.3 GB
    of a 40k prompt on Qwen3-0.6B, refused). Where the arena cannot take them, the cache keeps its own rows and the
    answer goes on through the torch layers over them, not refused"""
    dev = need_cuda()
    kw: dict[str, Any] = {"device": dev, "cpu_layers": 1, "prefill_chunk": 8, "prefill_card_min": 4}
    with loaded_model(fixture("tiny_qwen3"), **kw) as sm:
        _prefill(sm)  # the card's state made the placement's
        real = sm.scheduler.grant
        presized: list[str] = []

        def seen(nbytes: int, kind: str, **g: Any) -> None:
            if "the prompt's rows at once" in str(g.get("requester", "")):
                presized.append(str(g["requester"]))
            real(nbytes, kind, **g)

        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", seen)
            got = sm.generate(list(PROMPT), 4, eos=(), speculate=False).tokens
        assert len(got) == 4
        assert not presized, f"a prefill with the arena free made rows of its own: {presized}"

        def refused(*a: Any, **k: Any) -> Any:
            raise MemoryGrantError("[grant] REFUSED the test's arena: no room for the cache's rows")

        with monkeypatch.context() as mp:
            mp.setattr(sm.scheduler, "grant", seen)
            mp.setattr(sm, "_card_bind", refused)
            got = sm.generate(list(PROMPT), 4, eos=(), speculate=False).tokens
        assert len(got) == 4, "a refused arena failed the answer"
        assert presized, "the arena refused, the prefill's rows were made nowhere of their own"


def test_a_conv_state_left_in_float32_by_the_host_takes_the_cards_dtype() -> None:
    """a hybrid's linear-attention conv state is kept in the dtype of the pass that made it: a layer's chunk run on
    the host leaves it float32, and the layer's next run on the card joined its bf16 rows onto it promoted - the
    card's bf16 conv refused the float32 input (tiny_q4 under a squeezed card, whenever the squeeze moved a layer
    between the two). A state left in float32 is taken in the card's dtype there, and the tokens are those of a
    state never left so: bf16 to float32 and back is exact"""
    dev = need_cuda()
    kw: dict[str, Any] = {"device": dev, "cpu_layers": 1, "prefill_chunk": 8, "prefill_card_min": 4}
    with loaded_model(fixture("tiny_q4"), **kw) as sm:

        def run(widen: bool) -> list[int]:
            s = sm.session(PROMPT[:24])
            assert s.cache is not None
            if widen:
                n = 0
                for cl in s.cache.layers:
                    states = getattr(cl, "conv_states", None)
                    if isinstance(states, dict):
                        for k, v in states.items():
                            if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                                states[k] = v.float()  # as a run on the host leaves it
                                n += 1
                assert n, "no conv state in the card's dtype to leave in float32"
            s.feed(PROMPT[24:])
            return list(s.generate(4, eos=(), speculate=False).tokens)

        assert run(widen=True) == run(widen=False)
