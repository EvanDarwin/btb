# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Qwen4's sparse attention on a resident layer whose rows the card program keeps in RAM (`kv_host`: a context the
card has no room for). The reference module attends over every cached row it is handed - the whole prefix brought
to the card a layer and a chunk at a time, a million positions' rows more than the card holds - so where the program
keeps the rows in RAM a torch-path pass (a prefill's chunks, a pass the program does not take) goes through the
program's own kernels instead (`Qwen4Card.attend_rows`): the indexer's picks scored over the pooled keys on the card,
the attention reading only the rows they name, in RAM, where they stay. Elsewhere - the rows on the card, the
drafter's own cache, a fork's or a batch's rows - the module's forward, as it is."""

from __future__ import annotations

import weakref
from typing import Any

import torch

from ...cache import forked
from ...forward import chain_of
from ...scheduler import MemoryGrantError
from . import bind_forward


def install(attn: Any, sm: Any) -> None:
    """`attn` (a layer's `Qwen4ExpTextAttention`) forwarding through `_forward`; `sm` held weakly (the engine owns
    the layer). A host layer's attention takes attend.py's forward over this one"""
    attn._btb_ram_sm = weakref.ref(sm)
    bind_forward(attn, _forward)


def _forward(
    self: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kw: Any,
) -> tuple[torch.Tensor, None]:
    sm = self._btb_ram_sm()
    layers = getattr(past_key_values, "layers", None)
    prog = None
    if (
        sm is not None
        and layers is not None
        and len(layers) == int(sm.L)  # the model's own cache: not the drafter's
        and hidden_states.is_cuda
        and int(hidden_states.shape[0]) == 1
        and not forked(past_key_values)
    ):
        prog = sm._card_program_for_rows()
    if prog is None or int(self.layer_idx) not in prog.sj:
        if hidden_states.is_cuda and _rows_in_ram(past_key_values, int(self.layer_idx)):
            # the rows the arena kept in RAM, and nothing here to read them where they are (the program's attention
            # refused this placement): the reference module would meet the card's queries with the host's keys
            cp = getattr(sm, "_cp", None) if sm is not None else None
            why = getattr(cp, "_why_rows", None) or "the card program is off"
            raise MemoryGrantError(
                f"[card] layer {int(self.layer_idx)}'s rows are in RAM and the card program cannot read them as the "
                f"model is placed now ({why})"
            )
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kw)
    T = int(hidden_states.shape[1])
    parents = chain_of(getattr(sm, "ap", None) if getattr(sm, "aq", False) else None, T)
    return prog.attend_rows(self, hidden_states, position_embeddings, past_key_values, parents), None


def _rows_in_ram(cache: Any, i: int) -> bool:
    """whether layer `i` of `cache` holds its rows in RAM: an arena kept there, or the copies it left on the host"""
    layers = getattr(cache, "layers", None)
    if layers is None or i >= len(layers):
        return False
    cl = layers[i]
    if getattr(cl, "in_ram", False):
        return True
    k = getattr(cl, "keys", None)
    return isinstance(k, torch.Tensor) and k.numel() > 0 and k.device.type == "cpu"
