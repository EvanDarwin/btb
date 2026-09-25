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
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar, Unpack, cast, overload

import torch

from ..draft import Spans
from ..kinds import Json, Proposer, TokenRows, Tokens
from ..sampling import GREEDY, Sampling
from ..session import Session
from ..text import Channels, Messages, TextStream, ToolSpecs, answer, messages_of, prompt_ids
from .hooks import HookArgs, Hooks, LogitsProcessor, OnPass, OnToken, Taps, TokenLogprob
from .state import P, R, _State

if TYPE_CHECKING:
    from .branches import Batch
    from .model import StreamedTextModel


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
    tokens_per_pass: float
    phase_s: dict[str, float]
    by_source: dict[str, Any]
    drafted_by_pos: list[int]
    accepted_by_pos: list[int]
    anchored: bool
    mtp_build_s: float
    mtp_steps: int
    mtp_step_s: float


# a generation's shape: a row's tokens, logprobs and taps, or a list of each per row
TokensT = TypeVar("TokensT")
LogprobsT = TypeVar("LogprobsT")
TapsT = TypeVar("TapsT")


class Generation(tuple[TokensT, GenerateStats], Generic[TokensT, LogprobsT, TapsT]):
    """
    What `generate` returns: the new tokens (one row: a flat list; rows: a list per row) and the counts, plus
    what the call's hooks collected - `logprobs` a `TokenLogprob` per token, `hidden` {layer: [new, H]} - each a
    list per row for rows. Unpacks and indexes as `(tokens, stats)`.
    """

    logprobs: LogprobsT | None
    hidden: TapsT | None

    def __new__(
        cls,
        tokens: TokensT,
        stats: GenerateStats,
        logprobs: LogprobsT | None = None,
        hidden: TapsT | None = None,
    ) -> Generation[TokensT, LogprobsT, TapsT]:
        self = super().__new__(cls, (tokens, stats))
        self.logprobs, self.hidden = logprobs, hidden
        return self

    @property
    def tokens(self) -> TokensT:
        return self[0]

    @property
    def stats(self) -> GenerateStats:
        return self[1]


# one row's generation: its tokens, each token's logprob, the tapped layers' states over them
RowGeneration = Generation[list[int], list[TokenLogprob], Taps]
# several rows': a list of each, per row
BatchGeneration = Generation[list[list[int]], list[list[TokenLogprob]], list[Taps]]


class GenerateArgs(HookArgs, total=False):
    """`generate`'s keywords past the ids and `max_new`, as the calls that pass them on (a session's) take them"""

    eos: Tokens | None
    on_token: OnToken | None
    spans: Spans
    speculate: bool
    sampling: Sampling | None


def _stacked(hooks: Hooks) -> list[Taps]:
    """each row's tapped states, a layer's per-token states stacked into one [tokens, H]"""
    return [{i: torch.stack(hs) if hs else torch.empty(0) for i, hs in row.items()} for row in hooks.hidden]


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
        **hooks: Unpack[HookArgs],
    ):
        self.model = model
        self.result: RowGeneration | None = None
        self._q: queue.Queue[int | None] = queue.Queue()
        self._err: BaseException | None = None
        self._done = False
        self._stop = set(model.stop_ids)
        self._ch = Channels(model.tokenizer)
        self._text = TextStream(model.tokenizer)

        def work() -> None:
            try:
                self.result = model.generate(
                    ids, max_new, session=session, on_token=self._q.put, spans=spans, sampling=sampling, **hooks
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

    def _prompt(self, text: str, prefill: str | None = None) -> list[int]:
        self.history.append({"role": "user", "content": text})
        ids = self.model._reply_ids(self.history, self.thinking, prefill)
        if self.session.fresh:
            self.session.tail = self._tail(ids)
        return ids

    def _keep(self, ids: Tokens, gen: RowGeneration, prefill: str | None = None) -> str:
        stop = set(self.model.stop_ids)
        ans, think = answer(self.model.tokenizer, [t for t in gen.tokens if t not in stop])
        ans = (prefill or "") + ans
        self.history.append({"role": "assistant", "content": ans})
        self.last = dict(gen.stats, prompt=len(ids), new=len(gen.tokens), reasoning=think)
        return ans

    def ask(
        self,
        text: str,
        max_new: int | None = None,
        prefill: str | None = None,
        processors: Sequence[LogitsProcessor] = (),
    ) -> str:
        """One turn: `text` appended as the user's message, the answer returned and appended as the assistant's;
        `max_new` over the conversation's cap for this turn; `prefill` opens the answer with that text (it is
        part of what comes back); `processors` as `generate` takes them"""
        ids = self._prompt(text, prefill)
        gen = self.model.generate(
            ids, max_new or self.max_new, session=self.session, sampling=self.sampling, processors=processors
        )
        return self._keep(ids, gen, prefill)

    def stream(
        self, text: str, max_new: int | None = None, prefill: str | None = None, **hooks: Unpack[HookArgs]
    ) -> Iterator[str]:
        """
        The turn as pieces of text (after `prefill`, which opens it unstreamed); the history and `last` are filled
        when it ends; the rest are `generate`'s hooks
        """
        ids = self._prompt(text, prefill)
        s = Stream(self.model, ids, max_new or self.max_new, self.session, (), self.sampling, **hooks)
        yield from s
        if s.result is not None:
            self._keep(ids, s.result, prefill)

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

    def prompt_ids(
        self,
        prompt: str | Messages,
        thinking: bool = False,
        tools: ToolSpecs | None = None,
        continue_final: bool = False,
    ) -> list[int]:
        """
        A string or a conversation as the ids the model takes, through its chat template; `tools` reaches the
        templates that render function schemas; `continue_final` leaves the last (assistant) message open for the
        model to go on writing instead of starting a new turn
        """
        return prompt_ids(self.tokenizer, prompt, thinking, tools, continue_final)

    def _reply_ids(self, prompt: str | Messages, thinking: bool, prefill: str | None) -> list[int]:
        """the prompt's ids, the assistant's reply opened with `prefill` when there is one"""
        if not prefill:
            return self.prompt_ids(prompt, thinking)
        msgs = [*messages_of(prompt), {"role": "assistant", "content": prefill}]
        return self.prompt_ids(msgs, thinking, continue_final=True)

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

    @overload
    def generate(
        self,
        ids: Tokens,
        max_new: int | None = None,
        eos: Tokens | None = None,
        session: Session | None = None,
        on_token: OnToken | None = None,
        spans: Spans = (),
        speculate: bool = True,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        taps: Sequence[int] = (),
        on_pass: OnPass | None = None,
    ) -> RowGeneration: ...

    @overload
    def generate(
        self,
        ids: TokenRows,
        max_new: int | None = None,
        eos: Tokens | None = None,
        session: Session | None = None,
        on_token: OnToken | None = None,
        spans: Spans = (),
        speculate: bool = True,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        taps: Sequence[int] = (),
        on_pass: OnPass | None = None,
    ) -> BatchGeneration: ...

    def generate(
        self,
        ids: Tokens | TokenRows,
        max_new: int | None = None,
        eos: Tokens | None = None,
        session: Session | None = None,
        on_token: OnToken | None = None,
        spans: Spans = (),
        speculate: bool = True,
        sampling: Sampling | None = None,
        processors: Sequence[LogitsProcessor] = (),
        logprobs: int | None = None,
        taps: Sequence[int] = (),
        on_pass: OnPass | None = None,
    ) -> RowGeneration | BatchGeneration:
        """
        Decode a row of ids with the engine's configured loop: speculative where it pays, the same answer either
        way; several rows go through the batched loop, each row as its own plain decode. One decode runs at a
        time on an engine: a second caller waits. `max_new` defaults to what the window leaves; `eos` to
        `stop_ids`; `spans` are (tag, ids) texts banked for the n-gram proposer (`btb.SpanBank`); `speculate`
        False forces the plain one-token loop (no tree), the same tokens more slowly; `sampling` (a
        `btb.Sampling`) the pick of every token - greedy (argmax) or a temperature, and orthogonal to
        speculation - the engine's default when None (greedy unless loaded with a temperature).

        Hooks: `processors` (`btb.LogitsProcessor`s) rewrite every pick's logits, speculative drafts included;
        `logprobs` returns each token's log-probability (and the `logprobs` most likely, when above 0);
        `taps` returns the chosen layers' hidden state at each new token (layer i's is the residual stream leaving
        block i, before the final norm: transformers' `hidden_states[i + 1]`); `on_pass` is called with each pass's
        `btb.PassStats`. The processors, logprobs and taps need the logits or the layers in hand, so a hooked
        decode skips the paths that pick inside their graph (recorded as `PassTag.PICK_HOOKED`).
        """
        hooks = Hooks(
            tuple(processors),
            None if logprobs is None else int(logprobs),
            tuple(int(i) % self.L for i in taps),
            on_pass,
        )
        args = (ids, max_new, eos, session, on_token, spans, speculate, sampling, hooks)
        if getattr(self, "mlx", None) is not None and threading.current_thread() is not getattr(
            self, "_worker_thread", None
        ):
            gen: RowGeneration | BatchGeneration = self.on_worker(self._locked, self._generate, *args)
            return gen
        return self._locked(self._generate, *args)

    def _locked(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """`fn` holding the decode lock: one decode at a time on an engine"""
        with self._decode_lock:
            return fn(*args, **kwargs)

    def _serial(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """`fn(*args)` as a decode runs: one at a time, on the MLX worker thread where there is one"""
        if getattr(self, "mlx", None) is not None and threading.current_thread() is not getattr(
            self, "_worker_thread", None
        ):
            out: R = self.on_worker(self._serial, fn, *args, **kwargs)
            return out
        with self._decode_lock, torch.inference_mode():
            return fn(*args, **kwargs)

    def hidden(self, ids: Tokens, layers: Sequence[int] = (-1,)) -> dict[int, torch.Tensor]:
        """
        The chosen layers' output at every position of `ids`, {layer: [T, H]} in float32: one pass over a cache
        of its own, stopped after the deepest layer asked for. Layers count from 0; -1 is the last.
        """
        want = sorted({int(i) % self.L for i in layers})

        def run() -> dict[int, torch.Tensor]:
            seen: dict[int, torch.Tensor] = {}

            def keep(i: int, h: torch.Tensor) -> None:
                if i in want:
                    seen[i] = h[0].float().cpu()

            self.forward(
                [list(ids)], cache=self.new_cache(), on_layer=keep, last_only=False, head=False, stop_after=want[-1] + 1
            )
            return {i: seen[i] for i in want}

        return self._serial(run)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Hidden states [..., H] through the model's final norm and output head: the logits [..., V] the last layer's
        output would give, from any layer's (a logit lens)
        """

        def run() -> torch.Tensor:
            x = h if h.dim() == 3 else h.reshape(1, -1, h.shape[-1])
            out = self._apply_head(self._final_norm(x)).float().cpu()
            return out.reshape(*h.shape[:-1], out.shape[-1])

        return self._serial(run)

    def encode(
        self, texts: str | Sequence[str], layer: int = -1, pool: str = "last", normalize: bool = True
    ) -> torch.Tensor:
        """
        One vector per text, [N, H]: the text tokenized as is (no chat template), `layer`'s output pooled over
        its positions - `last` (the final token's, what decoder embedding models are trained for) or `mean` -
        and scaled to unit length unless `normalize` is False.
        """
        if pool not in ("last", "mean"):
            raise ValueError(f"pool {pool!r}: 'last' or 'mean'")
        items = [texts] if isinstance(texts, str) else list(texts)
        out = []
        for t in items:
            ids = [int(i) for i in self.tokenizer(t)["input_ids"]]
            h = self.hidden(ids, (layer,))[int(layer) % self.L]
            v = h[-1] if pool == "last" else h.mean(0)
            out.append(torch.nn.functional.normalize(v, dim=-1) if normalize else v)
        return torch.stack(out)

    def _generate(
        self,
        ids: Tokens | TokenRows,
        max_new: int | None,
        eos: Tokens | None,
        session: Session | None,
        on_token: OnToken | None,
        spans: Spans,
        speculate: bool,
        sampling: Sampling | None,
        hooks: Hooks,
    ) -> RowGeneration | BatchGeneration:
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
        if len(rows) > 1 and session is not None:
            raise ValueError("a session holds one sequence: batch sessions with model.batch(sessions)")
        if len({len(r) for r in rows}) > 1:
            # rows of their own lengths: left-padded in epochs the scheduler sizes
            out = self.serve(rows, max_new, eos_ids=stop, sampling=smp, hooks=hooks)
            return self._batch_result(out, dict(cap=max_new, proposer="greedy", tokens_per_pass=1.0, **seed), hooks)
        v = max(0, int(self.v_max)) if speculate else 0
        if len(rows) > 1 or (v == 0 and session is None):
            # several rows, or no drafts and no session to keep: the plain loop (the speculative loop is one row)
            out = self.generate_greedy(rows, max_new, eos_ids=stop, on_token=on_token, sampling=smp, hooks=hooks)
            stats = dict(cap=max_new, proposer="greedy", tokens_per_pass=1.0, **seed)
            if len(rows) > 1:
                return self._batch_result(out, stats, hooks)
            return self._row_result(out, stats, hooks)
        if (
            v == 0
            and not hooks.active
            and getattr(self, "mlx", None) is not None
            and self._mlx_greedy_ok(len(rows), None, None, False)
        ):
            # plain decoding with a session on the MLX device: the pipelined loop, reusing the session's cache
            out, c = self.generate_greedy(
                rows, max_new, eos_ids=stop, on_token=on_token, session=session, sampling=smp, hooks=hooks
            )
            return self._row_result(out, dict(c, cap=max_new, proposer="greedy", tokens_per_pass=1.0, **seed), hooks)
        prop = self.proposer if v > 0 else Proposer.NGRAM
        one, c = self.generate_speculative(
            rows,
            max_new,
            eos_ids=stop,
            v_max=v,
            proposer=prop,
            session=session,
            on_token=on_token,
            spans=spans,
            sampling=smp,
            hooks=hooks,
        )
        return self._row_result(one, dict(c, cap=max_new), hooks)

    @staticmethod
    def _row_result(out: list[int], stats: Json, hooks: Hooks) -> RowGeneration:
        """one row's tokens and counts, with what the hooks collected for it"""
        lp = (hooks.lp[0] if hooks.lp else []) if hooks.logprobs is not None else None
        taps = _stacked(hooks)
        hidden = (taps[0] if taps else {}) if hooks.taps else None
        return Generation(out, cast(GenerateStats, stats), lp, hidden)

    @staticmethod
    def _batch_result(out: list[list[int]], stats: Json, hooks: Hooks) -> BatchGeneration:
        """the rows' tokens and counts, with what the hooks collected, a list per row"""
        lp = hooks.lp if hooks.logprobs is not None else None
        hidden = _stacked(hooks) if hooks.taps else None
        return Generation(out, cast(GenerateStats, stats), lp, hidden)

    def ask(
        self,
        prompt: str | Messages,
        max_new: int | None = None,
        thinking: bool = False,
        sampling: Sampling | None = None,
        prefill: str | None = None,
        processors: Sequence[LogitsProcessor] = (),
    ) -> str:
        """One answer as text. `prompt` is a string (one user turn) or a list of {role, content} messages,
        rendered through the model's chat template; `max_new` caps the new tokens (None: until the turn ends or
        the window is full); `thinking` reaches templates with the switch; `sampling` overrides the engine's
        (greedy by default); `prefill` starts the answer with that text and the model goes on from it (returned
        with it); `processors` as `generate` takes them. Reasoning channels are dropped; `stream()` keeps them in
        `answer()`."""
        gen = self.generate(
            self._reply_ids(prompt, thinking, prefill), max_new, sampling=sampling, processors=processors
        )
        stop = set(self.stop_ids)
        return (prefill or "") + answer(self.tokenizer, [t for t in gen.tokens if t not in stop])[0]

    def stream(
        self,
        prompt: str | Messages | Tokens,
        max_new: int | None = None,
        thinking: bool = False,
        session: Session | None = None,
        spans: Spans = (),
        sampling: Sampling | None = None,
        prefill: str | None = None,
        **hooks: Unpack[HookArgs],
    ) -> Stream:
        """The answer as pieces of text as they are decoded, iterated from another thread: `for piece in
        model.stream(...)`. `prompt` as `ask` takes it, or token ids already; the `Stream` carries `.tokens` and
        `.stats` (a `GenerateStats`) once it ends, and `.result` the whole `Generation`; `session` reuses a
        conversation's cache, `spans` are texts banked for the n-gram proposer, `sampling` overrides the
        engine's, `prefill` opens the answer (not streamed: the model's continuation is); the rest are
        `generate`'s hooks."""
        if not isinstance(prompt, str) and prompt and isinstance(prompt[0], int):
            ids = [int(t) for t in cast(Tokens, prompt)]
        else:
            ids = self._reply_ids(cast("str | Messages", prompt), thinking, prefill)
        return Stream(self, ids, max_new, session, spans, sampling, **hooks)

    def session(self, ids: Tokens = ()) -> Session:
        """A sequence of this engine's to drive by hand - `feed`, `mark`, `rewind`, `fork`, `generate` - fed `ids`
        first when given; also what `generate(session=...)` continues"""
        # the mixin is only ever the engine itself
        s = Session(engine=cast("StreamedTextModel", self))
        if ids:
            s.feed(ids)
        return s

    def batch(self, sessions: Sequence[Session]) -> Batch:
        """Sessions decoded together, a row each (`btb.Batch`): stepped as one batch, joined and left between
        steps, each row written back into its session as it ends"""
        from .branches import Batch

        return Batch(cast("StreamedTextModel", self), sessions)

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
        processors: Sequence[LogitsProcessor] = (),
    ) -> list[str]:
        """
        One answer per prompt, decoded together in epochs the scheduler sizes; `processors` as `generate` takes
        them, each row read with its own ids
        """
        rows = [self.prompt_ids(p, thinking) for p in prompts]
        if not rows:
            return []
        if max_new is None:
            win = self.window
            max_new = max(1, win - max(len(r) for r in rows)) if win else 4096
        stop = set(self.stop_ids)
        tok = self.tokenizer
        hooks = Hooks(tuple(processors)) if processors else None
        return [
            answer(tok, [t for t in out if t not in stop])[0]
            for out in self.serve(rows, int(max_new), eos_ids=tuple(stop), sampling=sampling, hooks=hooks)
        ]

    @contextlib.contextmanager
    def reserve(self, tag: str, nbytes: int, device: Any = None) -> Iterator[None]:
        """
        `room` for the block: room made for memory of your own and kept from btb while the block runs, `tag`
        naming it in a refusal. A `MemoryGrantError` when btb cannot make that much.
        """
        with self.room(nbytes, device, name=tag):
            yield

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
