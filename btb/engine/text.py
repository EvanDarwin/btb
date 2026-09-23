# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The engine's text layer: the tokenizer it loads on first use, a prompt as ids, `generate` as a method with
the model's own stop ids, an answer as text, a stream of whole text, a conversation, and the engine as a
context manager. Every decode below goes through the same loops the token layer exposes.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, TypedDict, cast

from ..draft import Spans
from ..kinds import Proposer, TokenRows, Tokens
from ..sampling import GREEDY, Sampling
from ..session import Session
from ..text import Channels, Messages, TextStream, answer, prompt_ids
from .state import _State


class GenerateStats(TypedDict, total=False):
    """
    The counts a decode returns beside its tokens. The greedy loop fills `cap` and `proposer`; the speculative
    loop fills the rest: `forwards` is the passes, `proposed` and `accepted` the drafted tokens, `reused` the
    positions a session gave back, `seconds` and `prefill_s` the wall clock
    """

    cap: int
    proposer: str
    seed: int  # the sample's seed (given, or drawn for the call); absent under greedy decoding
    forwards: int
    proposed: int
    accepted: int
    reused: int
    seconds: float
    prefill_s: float
    phase_s: dict[str, float]
    by_source: dict[str, Any]
    drafted_by_pos: list[int]
    accepted_by_pos: list[int]
    anchored: bool
    mtp_build_s: float
    mtp_steps: int
    mtp_step_s: float


class Generation(NamedTuple):
    """
    What `generate` returns: the new tokens (one row: a flat list; rows: a list per row) and the counts.
    Unpacks as `tokens, stats = model.generate(...)`
    """

    tokens: Any
    stats: GenerateStats


class Stream:
    """
    A decode running on its own thread, read as pieces of text; `tokens` and `stats` are there once it ends.
    Only the answer streams: gpt-oss's reasoning is in `answer()` afterwards
    """

    def __init__(
        self,
        model: Any,
        ids: Tokens,
        max_new: int | None,
        session: Session | None,
        spans: Spans,
        sampling: Sampling | None = None,
    ):
        self.model = model
        self.result: Generation | None = None
        self._q: queue.Queue[int | None] = queue.Queue()
        self._err: BaseException | None = None
        self._done = False
        self._stop = set(model.stop_ids)
        self._ch = Channels(model.tokenizer)
        self._text = TextStream(model.tokenizer)

        def work() -> None:
            try:
                self.result = model.generate(
                    ids, max_new, session=session, on_token=self._q.put, spans=spans, sampling=sampling
                )
            except BaseException as e:  # handed to the reader, whatever it was
                self._err = e
            finally:
                self._q.put(None)

        self._th = threading.Thread(target=work, name="btb-stream", daemon=True)
        self._th.start()

    def close(self) -> None:
        """stop the decode at its next step and wait for it: what breaking out of the iterator leaves running"""
        if not self._th.is_alive():
            return
        self.model.abort.set()
        self._th.join()
        self.model.abort.clear()

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        while not self._done:
            t = self._q.get()
            if t is None:
                self._done = True
                self._th.join()
                if self._err is not None:
                    raise self._err
                tail = self._text.flush()
                if tail:
                    return tail
                break
            if t in self._stop or self._ch.push(t) != "content":
                continue
            piece = self._text.push(t)
            if piece:
                return piece
        raise StopIteration

    @property
    def tokens(self) -> list[int]:
        return list(self.result.tokens) if self.result is not None else []

    @property
    def stats(self) -> GenerateStats:
        return self.result.stats if self.result is not None else {}


class Chat:
    """
    A conversation over one engine: the history, the session its cache lives in, the last turn's counts
    (`last`: the stats plus `prompt`, `new`, `reasoning`)
    """

    def __init__(
        self, model: Any, max_new: int | None = None, thinking: bool = False, sampling: Sampling | None = None
    ) -> None:
        self.model, self.max_new, self.thinking, self.sampling = model, max_new, thinking, sampling
        self.history: list[dict[str, str]] = []
        self.session = Session()
        self.last: dict[str, Any] = {}

    def _prompt(self, text: str) -> list[int]:
        self.history.append({"role": "user", "content": text})
        ids = self.model.prompt_ids(self.history, self.thinking)
        if self.session.fresh:
            self.session.tail = self._tail(ids)
        return ids

    def _keep(self, ids: Tokens, gen: Generation) -> str:
        stop = set(self.model.stop_ids)
        ans, think = answer(self.model.tokenizer, [t for t in gen.tokens if t not in stop])
        self.history.append({"role": "assistant", "content": ans})
        self.last = dict(gen.stats, prompt=len(ids), new=len(gen.tokens), reasoning=think)
        return ans

    def ask(self, text: str, max_new: int | None = None) -> str:
        """One turn: `text` appended as the user's message, the answer returned and appended as the assistant's;
        `max_new` over the conversation's cap for this turn"""
        ids = self._prompt(text)
        gen = self.model.generate(ids, max_new or self.max_new, session=self.session, sampling=self.sampling)
        return self._keep(ids, gen)

    def stream(self, text: str, max_new: int | None = None) -> Iterator[str]:
        """
        The turn as pieces of text; the history and `last` are filled when it ends
        """
        ids = self._prompt(text)
        s = Stream(self.model, ids, max_new or self.max_new, self.session, (), self.sampling)
        yield from s
        if s.result is not None:
            self._keep(ids, s.result)

    def _tail(self, ids: Tokens) -> int:
        """how many tokens of the generation prompt's tail the next turn re-renders differently (the state
        snapshot for the next turn is left before them)"""
        tok = self.model.tokenizer
        # the answer must not be the final message: templates keep the think block on the last turn only
        probe = [*self.history, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
        try:
            try:
                text = tok.apply_chat_template(
                    probe, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            except TypeError:
                text = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
            other = tok(text, add_special_tokens=False)["input_ids"]
        except Exception:
            return 0
        m, lim = 0, min(len(ids), len(other))
        while m < lim and ids[m] == other[m]:
            m += 1
        return max(0, len(ids) - m)

    def reset(self) -> None:
        """the conversation from the start: the history, the cache and the last turn's counts cleared"""
        self.history, self.session, self.last = [], Session(), {}


class _TextMixin(_State):
    _tokenizer: Any = None

    @property
    def tokenizer(self) -> Any:
        """
        The model's tokenizer, loaded from its directory on first use; assign your own to replace it
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            gg = getattr(self, "gguf", None)
            self._tokenizer = gg.tokenizer() if gg is not None else AutoTokenizer.from_pretrained(self.dir)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tok: Any) -> None:
        self._tokenizer = tok

    @property
    def stop_ids(self) -> tuple[int, ...]:
        """
        Where a decode stops by default: the generation config's ids, and the tokenizer's once it is loaded
        """
        out = [int(x) for x in (getattr(self, "eos_ids", ()) or ())]
        tok = self._tokenizer
        e = getattr(tok, "eos_token_id", None) if tok is not None else None
        for x in e if isinstance(e, (list, tuple)) else [e]:
            if x is not None and int(x) not in out:
                out.append(int(x))
        return tuple(out)

    @property
    def window(self) -> int:
        """
        The context window in tokens: the one named at load, else the model's own
        """
        return int(getattr(self, "context", None) or getattr(self.cfg, "max_position_embeddings", 0) or 0)

    @property
    def device_name(self) -> str:
        from .. import device_name

        return device_name(self)

    def peak_memory(self) -> tuple[int, int]:
        """
        (peak RSS bytes, peak card or MLX bytes) of the process so far
        """
        from .. import peak_memory

        return peak_memory(self)

    def prompt_ids(self, prompt: str | Messages, thinking: bool = False, tools: Any = None) -> list[int]:
        """
        A string or a conversation as the ids the model takes, through its chat template; `tools` reaches the
        templates that render function schemas
        """
        return prompt_ids(self.tokenizer, prompt, thinking, tools)

    def on_worker(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        """
        `fn(*args, **kw)` on this model's one worker thread, from whichever thread asked. MLX keeps a lazy
        array's stream per thread and refuses it from another, in an eval or in a destructor, so on the MLX tier
        every decode - and the freeing of the arrays it built - happens on one thread for the model's lifetime;
        a server that hands each request its own thread, or a stream on a thread of its own, lands here
        """
        w = getattr(self, "_worker", None)
        if w is None:
            with self._decode_lock:  # two first callers would make two "one" workers
                w = getattr(self, "_worker", None)
                if w is None:
                    w = self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="btb-mlx")
                    self._worker_thread = w.submit(threading.current_thread).result()
        if threading.current_thread() is self._worker_thread:
            return fn(*args, **kw)
        return w.submit(fn, *args, **kw).result()

    def generate(
        self,
        ids: Tokens | TokenRows,
        max_new: int | None = None,
        eos: Tokens | None = None,
        session: Session | None = None,
        on_token: Callable[[int], Any] | None = None,
        spans: Spans = (),
        speculate: bool = True,
        sampling: Sampling | None = None,
    ) -> Generation:
        """
        Decode a row of ids with the engine's configured loop: speculative where it pays, the same answer either
        way; several rows go through the batched loop, each row as its own plain decode. One decode runs at a
        time on an engine: a second caller waits. `max_new` defaults to what the window leaves; `eos` to
        `stop_ids`; `spans` are (tag, ids) texts banked for the n-gram proposer (`btb.SpanBank`); `speculate`
        False forces the plain one-token loop (no tree), the same tokens more slowly; `sampling` (a
        `btb.Sampling`) the pick of every token - greedy (argmax) or a temperature, and orthogonal to
        speculation - the engine's default when None (greedy unless loaded with a temperature)
        """
        if getattr(self, "mlx", None) is not None and threading.current_thread() is not getattr(
            self, "_worker_thread", None
        ):
            return self.on_worker(self.generate, ids, max_new, eos, session, on_token, spans, speculate, sampling)
        with self._decode_lock:
            return self._generate(ids, max_new, eos, session, on_token, spans, speculate, sampling)

    def _generate(
        self,
        ids: Tokens | TokenRows,
        max_new: int | None,
        eos: Tokens | None,
        session: Session | None,
        on_token: Callable[[int], Any] | None,
        spans: Spans,
        speculate: bool,
        sampling: Sampling | None,
    ) -> Generation:
        self._pass_reset()  # one report a generate: its passes' path tags accumulate into it
        if ids and isinstance(ids[0], (list, tuple)):
            rows = [[int(t) for t in r] for r in cast(TokenRows, ids)]
        else:
            rows = [[int(t) for t in cast(Tokens, ids)]]
        if max_new is None:
            win = self.window
            max_new = max(1, win - max(len(r) for r in rows)) if win else 4096
        max_new = int(max_new)
        stop = tuple(self.stop_ids if eos is None else (int(e) for e in eos))
        smp = (sampling if sampling is not None else getattr(self, "sampling", None) or GREEDY).seeded()
        seed: dict[str, Any] = {} if smp.greedy else {"seed": smp.seed}
        if not speculate or len(rows) > 1 or (self.v_max <= 0 and session is None):
            # not speculating, several rows, or no verify budget: the plain loop (the speculative loop is one row)
            out = self.generate_greedy(rows, max_new, eos_ids=stop, on_token=on_token, sampling=smp)
            return Generation(out, cast(GenerateStats, dict(cap=max_new, proposer="greedy", **seed)))
        if (
            self.v_max <= 0
            and getattr(self, "mlx", None) is not None
            and self._mlx_greedy_ok(len(rows), None, None, False)
        ):
            # plain decoding with a session on the MLX device: the pipelined loop, reusing the session's cache
            out, c = self.generate_greedy(rows, max_new, eos_ids=stop, on_token=on_token, session=session, sampling=smp)
            return Generation(out, cast(GenerateStats, dict(c, cap=max_new, proposer="greedy", **seed)))
        v = max(0, int(self.v_max))
        prop = self.proposer if v > 0 else Proposer.NGRAM
        out, c = self.generate_speculative(
            rows,
            max_new,
            eos_ids=stop,
            v_max=v,
            proposer=prop,
            session=session,
            on_token=on_token,
            spans=spans,
            sampling=smp,
        )
        return Generation(out, cast(GenerateStats, dict(c, cap=max_new)))

    def ask(
        self,
        prompt: str | Messages,
        max_new: int | None = None,
        thinking: bool = False,
        sampling: Sampling | None = None,
    ) -> str:
        """One answer as text. `prompt` is a string (one user turn) or a list of {role, content} messages,
        rendered through the model's chat template; `max_new` caps the new tokens (None: until the turn ends or
        the window is full); `thinking` reaches templates with the switch; `sampling` overrides the engine's
        (greedy by default). Reasoning channels are dropped; `stream()` keeps them in `answer()`."""
        gen = self.generate(self.prompt_ids(prompt, thinking), max_new, sampling=sampling)
        stop = set(self.stop_ids)
        return answer(self.tokenizer, [t for t in gen.tokens if t not in stop])[0]

    def stream(
        self,
        prompt: str | Messages | Tokens,
        max_new: int | None = None,
        thinking: bool = False,
        session: Session | None = None,
        spans: Spans = (),
        sampling: Sampling | None = None,
    ) -> Stream:
        """The answer as pieces of text as they are decoded, iterated from another thread: `for piece in
        model.stream(...)`. `prompt` as `ask` takes it, or token ids already; the `Stream` carries `.tokens` and
        `.stats` (a `GenerateStats`) once it ends; `session` reuses a conversation's cache, `spans` are texts
        banked for the n-gram proposer, `sampling` overrides the engine's."""
        if not isinstance(prompt, str) and prompt and isinstance(prompt[0], int):
            ids = [int(t) for t in cast(Tokens, prompt)]
        else:
            ids = self.prompt_ids(cast("str | Messages", prompt), thinking)
        return Stream(self, ids, max_new, session, spans, sampling)

    def chat(self, max_new: int | None = None, thinking: bool = False, sampling: Sampling | None = None) -> Chat:
        """A conversation whose cache is reused turn to turn: `ask(text)` returns a turn, `stream(text)` yields
        it, `history` is the transcript, `last` the previous turn's counts, `reset()` starts over."""
        return Chat(self, max_new, thinking, sampling)

    def ask_many(
        self,
        prompts: Sequence[str | Messages],
        max_new: int | None = None,
        thinking: bool = False,
        sampling: Sampling | None = None,
    ) -> list[str]:
        """
        One answer per prompt, decoded together in epochs the scheduler sizes
        """
        rows = [self.prompt_ids(p, thinking) for p in prompts]
        if not rows:
            return []
        if max_new is None:
            win = self.window
            max_new = max(1, win - max(len(r) for r in rows)) if win else 4096
        stop = set(self.stop_ids)
        tok = self.tokenizer
        return [
            answer(tok, [t for t in out if t not in stop])[0]
            for out in self.serve(rows, int(max_new), eos_ids=tuple(stop), sampling=sampling)
        ]

    @contextlib.contextmanager
    def reserve(self, tag: str, nbytes: int, device: Any = None) -> Iterator[None]:
        """
        Memory of your own spoken for while the block runs, so the policy does not shed layers to make room
        the caller is about to take
        """
        self.device.reserve(tag, int(nbytes), device)
        try:
            yield
        finally:
            self.device.release(tag)

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
