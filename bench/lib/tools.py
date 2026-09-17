# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""
The rival engines, one class each behind one signature, and the timing loop every one of them gets.
They run through bench/compare.py in their own environment (.venv-compare); the matrix reads the
classes for what to plan (the module to probe for, the device regimes and the flags that force each)
"""

from __future__ import annotations

import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from lib import say
from lib.host import add_cuda_dll_dirs, gpu_used_gb, peak_rss_bytes
from lib.records import BenchCell


def scratch_dir(env: str, win: str) -> str:
    """
    Where a rival's converted copy of a model goes: `env` when set, else a fixed drive on Windows (so the
    matrix streams through one slice of disk) or the system's temp directory
    """
    import tempfile

    return os.environ.get(env, win if sys.platform == "win32" else tempfile.gettempdir())


def prompts(path: str, rows: str | None = None) -> list[str]:
    """
    The prompts of a questions file; `rows` picks by index ("0,2"), omitted is everything
    """
    ps = [json.loads(l)["prompt"] for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        return ps
    return [ps[int(x)] for x in rows.split(",") if x.strip()]


def template(tok: Any, prompt: str) -> str:
    """
    The prompt through the model's own chat template, thinking off where the template has the switch
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text


class BenchTool:
    """
    One external engine: loaded once, then it streams a prompt's tokens; `bench` does the timing.
    The class also says what the matrix plans for it: `module` is what its environment is probed
    for, `regimes` the device regimes it runs in (btb's names) with the flags that force each
    """

    name = ""
    module = ""
    regimes: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __init__(self, path: str, log: Callable[[str], Any] | None = None, **opts: Any) -> None:
        self.path = path
        self.log: Callable[[str], Any] = log or (lambda s: say(s, "compare"))
        self.opts = opts

    @classmethod
    def cannot(cls, regime: str, model_type: str | None, cuda: bool, gguf: bool) -> str | None:
        """
        Why a regime is a planned skip for this model here (None: it runs); `gguf` is set when the model is a
        GGUF file, which only the engines that read that format can run
        """
        return None

    def load(self) -> None:
        raise NotImplementedError

    def prompt(self, text: str) -> Any:
        """
        The templated prompt as the engine takes it (ids or text)
        """
        raise NotImplementedError

    def prompt_len(self, prompt: Any) -> int:
        raise NotImplementedError

    def stream(self, prompt: Any, n: int) -> Iterator[float]:
        """
        Generate `n` tokens greedily, yielding each token's arrival time (perf_counter)
        """
        raise NotImplementedError

    def device(self) -> str:
        raise NotImplementedError

    def peak_vram_gb(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class BenchMlxLm(BenchTool):
    name = "mlx-lm"
    module = "mlx_lm"
    regimes = (("mlx", ()),)  # unified memory, one regime

    @classmethod
    def cannot(cls, regime: str, model_type: str | None, cuda: bool, gguf: bool) -> str | None:
        return "mlx-lm loads MLX or HF weights, not a GGUF" if gguf else None

    def load(self) -> None:
        from mlx_lm import load
        from transformers import AutoTokenizer

        t0 = time.time()
        self.model, self.tok = load(self.path)[:2]  # a 2-tuple unless return_config=
        self.hf_tok = AutoTokenizer.from_pretrained(self.path)
        self.log(f"mlx-lm loaded {self.path} in {time.time() - t0:.1f}s")

    def prompt(self, text: str) -> Any:
        return self.hf_tok(template(self.hf_tok, text), add_special_tokens=False)["input_ids"]

    def prompt_len(self, prompt: Any) -> int:
        return len(prompt)

    def stream(self, prompt: Any, n: int) -> Iterator[float]:
        from mlx_lm import stream_generate

        for _r in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=n):
            yield time.perf_counter()

    def device(self) -> str:
        return "mlx"

    def peak_vram_gb(self) -> float:
        import mlx.core as mx

        return mx.get_peak_memory() / 2**30


class BenchAirLLM(BenchTool):
    name = "airllm"
    module = "airllm"
    # AirLLM runs on the device the matrix column is testing, through its own `device=` kwarg: the card for the
    # gpu column, like btb and llama.cpp there; the CPU for the cpu column
    regimes = (("gpu", ("--device", "cuda")), ("cpu", ("--device", "cpu")))

    @classmethod
    def cannot(cls, regime: str, model_type: str | None, cuda: bool, gguf: bool) -> str | None:
        if gguf:
            return "AirLLM loads an HF checkpoint, not a GGUF"
        if model_type == "gpt_oss":
            return "AirLLM's path dequantizes the MXFP4 experts (240 GB)"
        if regime == "gpu" and not cuda:
            return "no CUDA on this machine (AirLLM's card regime is CUDA only)"
        return None

    def load(self) -> None:
        import airllm.auto_model as am
        import torch
        from transformers import AutoConfig, AutoTokenizer

        path, log = self.path, self.log
        arch = (getattr(AutoConfig.from_pretrained(path), "architectures", None) or [""])[0]
        cls_name = am.ARCH_OVERRIDES.get(arch)
        mods = {
            "AirLLMQwen3_5": "airllm_qwen3_5",
            "AirLLMQwen4Exp": "airllm_qwen4_exp",
            "AirLLMKimiK3": "airllm_kimi_k3",
            "AirLLMChatGLM": "airllm_chatglm",
            "AirLLMQWen": "airllm_qwen",
            "AirLLMBaichuan": "airllm_baichuan",
            "AirLLMInternLM": "airllm_internlm",
        }
        if cls_name in mods:
            cls = getattr(importlib.import_module("airllm." + mods[cls_name]), cls_name)
        else:
            cls = importlib.import_module("airllm.airllm_base").AirLLMBaseModel
        log(f"airllm {cls.__name__} for {arch}")
        # AirLLM splits the checkpoint into a per-layer `splitted_model` on first load; route that copy to a
        # scratch dir (`<split_dir>/splitted_model`) and remove it after, unless it was already there (a reused
        # split, or a prebuilt one) or `keep_splits` is set - so the matrix streams through a fixed slice of disk
        split_dir = self.opts.get("split_dir")
        if split_dir is None:
            base = os.path.basename(os.path.normpath(path))
            split_dir = os.path.join(scratch_dir("BTB_AIRLLM_DIR", r"F:\_airllmcache"), base)
        self.split_dir: str = split_dir
        self.pre_exists = os.path.isdir(os.path.join(split_dir, "splitted_model"))
        self.dev: str = self.opts.get("device", "cuda")
        t0 = time.time()
        kw: dict[str, Any] = {"device": self.dev, "layer_shards_saving_path": split_dir}
        if self.opts.get("dtype"):
            kw["dtype"] = getattr(torch, self.opts["dtype"])
        self.model = cls(path, **kw)
        self.tok = AutoTokenizer.from_pretrained(path)
        log(f"airllm loaded {path} in {time.time() - t0:.1f}s")

    def prompt(self, text: str) -> Any:
        return self.tok(template(self.tok, text), add_special_tokens=False, return_tensors="pt")["input_ids"]

    def prompt_len(self, prompt: Any) -> int:
        return int(prompt.shape[-1])

    def stream(self, prompt: Any, n: int) -> Iterator[float]:
        import torch

        # generate() blocks, so the arrival times are taken by its streamer as it runs and handed over
        # after; the streamer sees the prompt first, then every generated token
        marks: list[float] = []

        class BenchStreamer:
            def put(self, _value: Any) -> None:
                marks.append(time.perf_counter())

            def end(self) -> None:
                pass

        tok = self.tok
        with torch.inference_mode():
            self.model.generate(
                prompt,
                max_new_tokens=n,
                min_new_tokens=n,
                do_sample=False,
                use_cache=True,
                streamer=BenchStreamer(),
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        yield from marks[1:] if len(marks) > 1 else marks

    def device(self) -> str:
        return self.dev

    def peak_vram_gb(self) -> float:
        import torch

        if self.dev == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
        return 0.0

    def close(self) -> None:
        split_dir = getattr(self, "split_dir", None)
        if not split_dir:
            return
        sm = os.path.join(split_dir, "splitted_model")
        if not self.opts.get("keep_splits") and not self.pre_exists and os.path.isdir(sm):
            try:
                shutil.rmtree(split_dir)
                self.log(f"removed airllm splits {split_dir}")
            except Exception as e:
                self.log(f"could not remove {split_dir}: {e}")


def convert_gguf(hf_dir: str, gguf_path: str, gguf_type: str, llama_cpp_dir: str, log: Callable[[str], Any]) -> None:
    """
    The HF model to a GGUF through llama.cpp's own converter, in the checkout at `llama_cpp_dir`
    """
    conv = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(conv):
        raise SystemExit(f"[compare] convert_hf_to_gguf.py not found at {conv}; set --llama-cpp-dir or LLAMA_CPP_DIR")
    os.makedirs(os.path.dirname(os.path.abspath(gguf_path)), exist_ok=True)
    log(f"converting {hf_dir} -> {gguf_path} ({gguf_type}) ...")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, conv, hf_dir, "--outfile", gguf_path, "--outtype", gguf_type],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if r.returncode != 0 or not os.path.exists(gguf_path):
        tail = ((r.stdout or "") + (r.stderr or ""))[-1800:]
        raise SystemExit(f"[compare] gguf conversion failed (exit {r.returncode}):\n{tail}")
    log(f"converted in {time.time() - t0:.0f}s, {os.path.getsize(gguf_path) / 2**30:.1f} GB")


class BenchLlamaCpp(BenchTool):
    """
    llama.cpp through llama-cpp-python: the HF model converted to a GGUF (bf16, the stored weights kept),
    greedy decoding (temperature 0) over the same templated prompts, the layers offloaded to the card
    (`ngl`, -1 all / 0 CPU-only). A GGUF this run converted is removed after unless `keep_gguf`.
    """

    name = "llama-cpp"
    module = "llama_cpp"
    regimes = (("gpu", ("--n-gpu-layers", "-1")), ("cpu", ("--n-gpu-layers", "0")))

    def load(self) -> None:
        add_cuda_dll_dirs()
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        o, path, log = self.opts, self.path, self.log
        llama_cpp_dir = o.get("llama_cpp_dir") or os.environ.get(
            "LLAMA_CPP_DIR", r"F:\llama.cpp" if sys.platform == "win32" else "llama.cpp"
        )
        gguf_type = o.get("gguf_type") or "bf16"
        self.made = False
        gguf = o.get("gguf")
        if path.lower().endswith(".gguf"):
            # the model already is a GGUF: run the file as it is, tokenizer and chat template off the file itself
            self.gguf_path: str = path
            self.hf_tok = AutoTokenizer.from_pretrained(os.path.dirname(path), gguf_file=os.path.basename(path))
        else:
            if gguf and os.path.exists(gguf):
                self.gguf_path = gguf
            else:
                base = os.path.basename(os.path.normpath(path))
                gdir = o.get("gguf_dir") or scratch_dir("BTB_GGUF_DIR", r"F:\_ggufcache")
                self.gguf_path = os.path.join(gdir, f"{base}.{gguf_type}.gguf")
                if not os.path.exists(self.gguf_path):
                    convert_gguf(path, self.gguf_path, gguf_type, llama_cpp_dir, log)
                    self.made = True
            self.hf_tok = AutoTokenizer.from_pretrained(path)
        self.llm: Any = None
        self.ngl = int(o.get("ngl", -1))
        n_ctx = max(2048, 1024 + int(o.get("max_new", 1024)) + 256)
        self.vram_base = gpu_used_gb()
        t0 = time.time()
        self.llm = Llama(model_path=self.gguf_path, n_gpu_layers=self.ngl, n_ctx=n_ctx, verbose=False, logits_all=False)
        log(f"llama.cpp loaded {self.gguf_path} (n_gpu_layers={self.ngl}, n_ctx={n_ctx}) in {time.time() - t0:.1f}s")
        self.peak = gpu_used_gb()

    def prompt(self, text: str) -> Any:
        return template(self.hf_tok, text)

    def prompt_len(self, prompt: Any) -> int:
        return len(self.hf_tok(prompt, add_special_tokens=False)["input_ids"])

    def stream(self, prompt: Any, n: int) -> Iterator[float]:
        for _ in self.llm.create_completion(
            prompt=prompt, max_tokens=n, temperature=0.0, top_k=1, top_p=1.0, repeat_penalty=1.0, stream=True
        ):
            yield time.perf_counter()
        self.peak = max(self.peak, gpu_used_gb())

    def device(self) -> str:
        return "cuda" if self.ngl != 0 else "cpu"

    def peak_vram_gb(self) -> float:
        return max(0.0, self.peak - self.vram_base)

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
            self.llm = None
            del llm
            gc.collect()
        gguf_path = getattr(self, "gguf_path", None)
        if getattr(self, "made", False) and not self.opts.get("keep_gguf") and gguf_path and os.path.exists(gguf_path):
            try:
                os.remove(gguf_path)
                self.log(f"removed {gguf_path}")
            except Exception as e:
                self.log(f"could not remove {gguf_path}: {e}")


TOOLS: dict[str, type[BenchTool]] = {t.name: t for t in (BenchMlxLm, BenchAirLLM, BenchLlamaCpp)}


def bench(tool: BenchTool, prompts: Sequence[str], counts: Sequence[int], budget: float) -> list[BenchCell]:
    """
    The timing every tool gets: per answer length, per prompt, the first token from the prompt and the
    rate after it; lengths past `budget` seconds are left out
    """
    cells: list[BenchCell] = []
    start = time.time()
    for n in counts:
        if time.time() - start > budget:
            tool.log(f"budget of {budget:.0f}s spent; {n} and beyond left out")
            break
        firsts, rates = [], []
        for p in prompts:
            q = tool.prompt(p)
            t0 = time.perf_counter()
            marks = list(tool.stream(q, n))
            t1 = time.perf_counter()
            if not marks:
                continue
            firsts.append(marks[0] - t0)
            rates.append((t1 - marks[0]) / max(1, len(marks) - 1))
            tool.log(
                f"{tool.name} new={n}: prompt {tool.prompt_len(q)}, {len(marks)} tokens, "
                f"first {firsts[-1]:.2f}s, then {rates[-1]:.3f} s/token"
            )
            if time.time() - start > budget:
                break
        if not firsts:
            continue
        k = len(firsts)
        cells.append(
            BenchCell(
                new=n,
                first_s=sum(firsts) / k,
                base_s_tok=sum(rates) / k,
                spec_s_tok=None,
                tokens_per_pass=1.0,
                identical="",
                peak_ram_gb=peak_rss_bytes() / 2**30,
                peak_vram_gb=tool.peak_vram_gb(),
            )
        )
    return cells
