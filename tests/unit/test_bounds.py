# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Where the messages of a prompt end among its tokens (btb/text.py `message_bounds`): the points a hybrid's state is
kept at, so a conversation through the same messages resumes there. Stub templates on the CPU, and the fixtures' own
tokenizers; no model."""

from __future__ import annotations

import pytest

from btb.kinds import Json
from btb.text import message_bounds, prompt_ids
from tests.helpers import fixture


class Tags:
    """a template rendering `<role>content</role>`, a token a character; `strip` drops a reply's `[...]` reasoning
    unless it is the last message, as Qwen3's template drops a think block; `offsets` off is a slow tokenizer"""

    def __init__(self, strip: bool = False, offsets: bool = True) -> None:
        self.strip, self.offsets = strip, offsets

    def apply_chat_template(self, msgs: list[Json], tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        out = ""
        for i, m in enumerate(msgs):
            body = str(m["content"])
            if self.strip and m["role"] == "assistant" and i < len(msgs) - 1 and "]" in body:
                body = body.split("]", 1)[1]
            out += f"<{m['role']}>{body}</{m['role']}>"
        return out + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> Json:
        if return_offsets_mapping:
            if not self.offsets:
                raise NotImplementedError("return_offsets_mapping is a fast tokenizer's")
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        return {"input_ids": [ord(c) for c in text]}


CONVO = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "[think about it]hello"},
    {"role": "user", "content": "more"},
]


@pytest.mark.parametrize("offsets", [True, False])
def test_each_message_ends_where_its_rendering_does(offsets: bool) -> None:
    """a bound a message: the rendering of the messages up to it, as tokens of the whole prompt - the same with a
    slow tokenizer (each end's text tokenized on its own) as with a fast one's offsets"""
    tok = Tags(offsets=offsets)
    ids = prompt_ids(tok, CONVO)  # type: ignore[arg-type]
    want = [len("<system>be brief</system>"), len("<system>be brief</system><user>hi</user>")]
    want.append(want[-1] + len("<assistant>[think about it]hello</assistant>"))
    assert message_bounds(tok, CONVO, ids) == want  # type: ignore[arg-type]


@pytest.mark.parametrize("offsets", [True, False])
def test_a_turn_the_template_renders_otherwise_later_ends_where_the_two_part(offsets: bool) -> None:
    """a template that renders a reply differently once more follows (its reasoning dropped): the reply's bound
    stops where the prefix's rendering and the whole prompt's part, never past rows the prompt does not hold"""
    tok = Tags(strip=True, offsets=offsets)
    ids = prompt_ids(tok, CONVO)  # type: ignore[arg-type]
    head = len("<system>be brief</system><user>hi</user>")
    assert message_bounds(tok, CONVO, ids) == [len("<system>be brief</system>"), head, head + len("<assistant>")]  # type: ignore[arg-type]


def test_a_token_across_a_bound_is_left_out() -> None:
    """a token running past a message's end belongs to the text after it: the bound stops before it"""

    class Pairs(Tags):
        def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False) -> Json:
            spans = [(i, min(i + 2, len(text))) for i in range(0, len(text), 2)]
            out: Json = {"input_ids": [hash(text[a:b]) % 997 for a, b in spans]}
            if return_offsets_mapping:
                out["offset_mapping"] = spans
            return out

    msgs = [
        {"role": "user", "content": "abcd"},
        {"role": "assistant", "content": "d"},
        {"role": "user", "content": "e"},
    ]
    tok = Pairs()
    ids = prompt_ids(tok, msgs)  # type: ignore[arg-type]
    ends = [len("<user>abcd</user>"), len("<user>abcd</user><assistant>d</assistant>")]
    assert ends[0] % 2 == 1  # the first message ends inside a token
    assert message_bounds(tok, msgs, ids) == [e // 2 for e in ends]  # type: ignore[arg-type]


def test_one_message_and_a_refused_prefix_are_no_bound() -> None:
    """a prompt of one message has no bound inside it; a prefix the template will not render alone is skipped"""

    class Alternating(Tags):
        def apply_chat_template(
            self, msgs: list[Json], tokenize: bool = False, add_generation_prompt: bool = False
        ) -> str:
            if msgs[-1]["role"] == "system":
                raise ValueError("a conversation must not end on its system message")
            return super().apply_chat_template(msgs, tokenize, add_generation_prompt)

    tok = Alternating()
    assert message_bounds(tok, CONVO[1:2], prompt_ids(tok, CONVO[1:2])) == []  # type: ignore[arg-type]
    ids = prompt_ids(tok, CONVO)  # type: ignore[arg-type]
    assert message_bounds(tok, CONVO, ids)[0] == len("<system>be brief</system><user>hi</user>")  # type: ignore[arg-type]


@pytest.mark.parametrize("stem", ["tiny_qwen3", "tiny_q35", "tiny_gemma3", "tiny_gpt_oss", "tiny_phi3"])
def test_the_fixtures_templates_give_bounds_inside_the_prompt_at_message_starts(stem: str) -> None:
    """the fixtures' own tokenizers: every bound lies inside the prompt, ascending, and the tokens before it are the
    tokens the messages before it render to (as far as the whole prompt agrees with them)"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(fixture(stem))
    if not getattr(tok, "chat_template", None):
        pytest.skip(f"{stem} ships no chat template")
    msgs = [
        {"role": "system", "content": "You answer in one word."},
        {"role": "user", "content": "What colour is the sky?"},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "And grass?"},
    ]
    try:
        ids = prompt_ids(tok, msgs)
    except Exception as e:  # a template that takes no system message: the rest of the conversation
        msgs = msgs[1:]
        ids = prompt_ids(tok, msgs)
        assert ids, e
    bounds = message_bounds(tok, msgs, ids)
    assert bounds == sorted(set(bounds)) and all(0 < b < len(ids) for b in bounds), bounds
    assert bounds, f"{stem}: no message bound found in a {len(msgs)}-message prompt"
    text = str(tok.decode(ids))
    for b in bounds:
        assert text.startswith(str(tok.decode(ids[:b]))), b
