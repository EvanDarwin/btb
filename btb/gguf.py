# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A GGUF model (llama.cpp's file format) behind the engine's weight map. The file's metadata makes the config
and the tokenizer through transformers' GGUF support; the family's tensor names map to the file's through
llama.cpp's own per-architecture table (`gguf.TensorNameMap`, no name parsing); a tensor is read on demand from
the memmapped file by the `gguf` package, which also dequantizes it, and a tensor llama.cpp's converter rewrote
(Qwen3.5's linear attention and norms, which qwen4exp's converter inherits) is read back through the inverse
(`Undo`). This is the glue between that package and the engine; the one part of llama.cpp held here is
`OWN_NAMES`, for an architecture no gguf release knows yet."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .hf import AFFINE_TYPES, ARCH_MODEL_TYPES, QWEN4EXP, gguf_lib, is_gguf
from .kinds import Json, ModelType
from .options import NotAModel, UnsupportedModel

# a checkpoint tensor rebuilt from several of the file's (qwen4exp's split indexer and experts, its n-gram
# constants): the file tensors it reads (float32, dequantized) and the function that joins them
Invert = Callable[[list[np.ndarray]], np.ndarray]
Derived = tuple[tuple[str, ...], Invert]
NameMap = Callable[[str], str | None]

# the architectures whose converter rewrites the linear attention and the norms (Qwen3_5TextModel.modify_tensors,
# which qwen4exp's converter runs under its own)
REWRITTEN = frozenset({"qwen35", QWEN4EXP})

# btb's own names for an architecture the installed gguf package does not know yet, modelled on llama.cpp's tree
# (gguf-py's constants.py and tensor_mapping.py): the checkpoint name the converter hands its TensorNameMap -> the
# file's. The package's table replaces it once a release has the architecture; test_gguf holds the two equal.
OWN_NAMES: dict[str, dict[str, str]] = {
    QWEN4EXP: {
        "model.embed_tokens": "token_embd",
        "lm_head": "output",
        "model.hyper_connection_mixer.hc_norm": "output_hc_norm",
        "model.hyper_connection_mixer.input_mix_weight_down": "output_hc_down",
        "model.hyper_connection_mixer.input_mix_weight_up": "output_hc_up",
        "model.layers.{bid}.attn_hyper_connection.hc_norm": "blk.{bid}.hc_attn_norm",
        "model.layers.{bid}.attn_hyper_connection.input_mix_weight_down": "blk.{bid}.hc_attn_down",
        "model.layers.{bid}.attn_hyper_connection.input_mix_weight_up": "blk.{bid}.hc_attn_up",
        "model.layers.{bid}.attn_hyper_connection.block_inject_weight": "blk.{bid}.hc_attn_inject",
        "model.layers.{bid}.mlp_hyper_connection.hc_norm": "blk.{bid}.hc_ffn_norm",
        "model.layers.{bid}.mlp_hyper_connection.input_mix_weight_down": "blk.{bid}.hc_ffn_down",
        "model.layers.{bid}.mlp_hyper_connection.input_mix_weight_up": "blk.{bid}.hc_ffn_up",
        "model.layers.{bid}.mlp_hyper_connection.block_inject_weight": "blk.{bid}.hc_ffn_inject",
        "model.layers.{bid}.linear_attn.A_log": "blk.{bid}.ssm_a",
        "model.layers.{bid}.linear_attn.conv1d": "blk.{bid}.ssm_conv1d",
        "model.layers.{bid}.linear_attn.dt_proj": "blk.{bid}.ssm_dt",
        "model.layers.{bid}.linear_attn.in_proj_a": "blk.{bid}.ssm_alpha",
        "model.layers.{bid}.linear_attn.in_proj_b": "blk.{bid}.ssm_beta",
        "model.layers.{bid}.linear_attn.in_proj_qkv": "blk.{bid}.attn_qkv",
        "model.layers.{bid}.linear_attn.in_proj_z": "blk.{bid}.attn_gate",
        "model.layers.{bid}.linear_attn.norm": "blk.{bid}.ssm_norm",
        "model.layers.{bid}.linear_attn.out_proj": "blk.{bid}.ssm_out",
        "model.layers.{bid}.self_attn.q_proj": "blk.{bid}.attn_q",
        "model.layers.{bid}.self_attn.k_proj": "blk.{bid}.attn_k",
        "model.layers.{bid}.self_attn.v_proj": "blk.{bid}.attn_v",
        "model.layers.{bid}.self_attn.o_proj": "blk.{bid}.attn_output",
        "model.layers.{bid}.self_attn.q_norm": "blk.{bid}.attn_q_norm",
        "model.layers.{bid}.self_attn.k_norm": "blk.{bid}.attn_k_norm",
        "model.layers.{bid}.self_attn.indexer.q_layernorm": "blk.{bid}.indexer.q_norm",
        "model.layers.{bid}.self_attn.indexer.k_layernorm": "blk.{bid}.indexer.k_norm",
        "model.layers.{bid}.mlp.gate": "blk.{bid}.ffn_gate_inp",
        "model.layers.{bid}.mlp.experts.gate_proj": "blk.{bid}.ffn_gate_exps",
        "model.layers.{bid}.mlp.experts.up_proj": "blk.{bid}.ffn_up_exps",
        "model.layers.{bid}.mlp.experts.down_proj": "blk.{bid}.ffn_down_exps",
        "model.layers.{bid}.mlp.shared_expert.gate_proj": "blk.{bid}.ffn_gate_shexp",
        "model.layers.{bid}.mlp.shared_expert.up_proj": "blk.{bid}.ffn_up_shexp",
        "model.layers.{bid}.mlp.shared_expert.down_proj": "blk.{bid}.ffn_down_shexp",
        "model.layers.{bid}.mlp.shared_expert_gate": "blk.{bid}.ffn_gate_inp_shexp",
        "model.layers.{bid}.ple.key_proj": "blk.{bid}.ple_key",
        "model.layers.{bid}.ple.value_proj": "blk.{bid}.ple_value",
        "model.layers.{bid}.ple.norm_key": "blk.{bid}.ple_norm_key",
        "model.layers.{bid}.ple.norm_query": "blk.{bid}.ple_norm_query",
        "model.layers.{bid}.ple.norm_conv": "blk.{bid}.ple_norm_conv",
        "model.layers.{bid}.ple.conv1d": "blk.{bid}.ple_conv1d",
    },
}
# the names the converter writes by gguf.MODEL_TENSOR member rather than through the table
OWN_TENSOR_NAMES: dict[str, dict[str, str]] = {
    QWEN4EXP: {
        "INDEXER_Q_PROJ": "blk.{bid}.indexer.q_proj",
        "INDEXER_K_PROJ": "blk.{bid}.indexer.k_proj",
        "PER_LAYER_TOKEN_EMBD": "per_layer_token_embd",
    },
}
# metadata keys by their gguf.Keys path, for the ones the installed package does not define yet
OWN_KEYS: dict[str, str] = {
    "Attention.COMPRESS_RATIOS": "{arch}.attention.compress_ratios",
    "Attention.RECURRENT_LAYERS": "{arch}.attention.recurrent_layers",
    "HyperConnection.COUNT": "{arch}.hyper_connection.count",
    "HyperConnection.LOW_RANK": "{arch}.hyper_connection.low_rank",
    "PerLayerEmbedding.LAYERS": "{arch}.ple.layers",
    "PerLayerEmbedding.NGRAM_SIZE": "{arch}.ple.ngram_size",
    "PerLayerEmbedding.HEADS_PER_NGRAM": "{arch}.ple.heads_per_ngram",
    "PerLayerEmbedding.CONV_KERNEL": "{arch}.ple.conv_kernel",
    "PerLayerEmbedding.LAYER_MULTIPLIERS": "{arch}.ple.layer_multipliers",
    "PerLayerEmbedding.HEAD_OFFSETS": "{arch}.ple.head_offsets",
    "PerLayerEmbedding.HEAD_VOCAB_SIZES": "{arch}.ple.head_vocab_sizes",
    "PerLayerEmbedding.EOS_TOKEN_ID": "{arch}.ple.eos_token_id",
}


def knows_arch(arch: str) -> bool:
    """whether the installed gguf package has llama.cpp's architecture `arch`"""
    return arch in gguf_lib().MODEL_ARCH_NAMES.values()


def name_map(arch: str, n_layers: int) -> NameMap:
    """the file's name for a checkpoint tensor's (without .weight/.bias): the gguf package's TensorNameMap, or
    btb's own table while the installed package does not know `arch`"""
    g = gguf_lib()
    if knows_arch(arch):
        member = next(a for a, n in g.MODEL_ARCH_NAMES.items() if n == arch)
        get: NameMap = g.TensorNameMap(member, int(n_layers)).get_name
        return get
    table = {k.format(bid=i): v.format(bid=i) for k, v in OWN_NAMES[arch].items() for i in range(int(n_layers))}
    return table.get


def tensor_name(arch: str, member: str, bid: int = 0) -> str:
    """the file's name for gguf.MODEL_TENSOR `member` of layer `bid`: the package's, or btb's while it lacks `arch`"""
    g = gguf_lib()
    name = g.TENSOR_NAMES[g.MODEL_TENSOR[member]] if knows_arch(arch) else OWN_TENSOR_NAMES[arch][member]
    return str(name).format(bid=bid)


def meta_key(arch: str, path: str) -> str:
    """the metadata key at gguf.Keys `path` ("PerLayerEmbedding.LAYERS"): the package's where it defines it"""
    obj: object = gguf_lib().Keys
    for part in path.split("."):
        obj = getattr(obj, part, None)
    return (obj if isinstance(obj, str) else OWN_KEYS[path]).format(arch=arch)


def _plus_one(hf: str) -> bool:
    """the gammas llama.cpp's converter stores +1: Qwen3-Next's norm.weight rule (the gated norm aside) and
    qwen4exp's PLE norms"""
    return hf.endswith(("norm.weight", ".ple.norm_key.weight", ".ple.norm_query.weight", ".ple.norm_conv.weight")) and (
        not hf.endswith("linear_attn.norm.weight")
    )


@dataclass(frozen=True)
class Undo:
    """The inverse of what llama.cpp's converter did to one tensor: `rows`/`cols` gather the checkpoint's order
    out of the file's, `value` undoes an elementwise rewrite, `shape` restores an axis the converter squeezed."""

    rows: np.ndarray | None = None
    cols: np.ndarray | None = None
    value: Callable[[torch.Tensor], torch.Tensor] | None = None
    shape: tuple[int, ...] | None = None


def _grouped(nk: int, per: int, width: int) -> np.ndarray:
    """the file positions of the checkpoint's value heads (grouped by key head) in the converter's tiled order,
    which `_LinearAttentionVReorderBase._reorder_v_heads` writes for `nk` key heads of `per` value heads each"""
    return np.arange(nk * per * width).reshape(per, nk, width).transpose(1, 0, 2).reshape(-1)


class GGUFModel:
    """One `.gguf` file: `config()` and `tokenizer()` as transformers reads them off the metadata, `weight_map`
    the family's tensor names against the file's, `get` a tensor as bf16 (dequantized where the file holds a
    quantized type; f16 and f32 rounded to bf16)."""

    def __init__(self, path: str) -> None:
        if not is_gguf(path):
            raise ValueError(f"{path!r} is not a .gguf file")
        self.path = os.path.abspath(path)
        self.dir = os.path.dirname(self.path)
        self.file = os.path.basename(self.path)
        self._reader: Any = None
        self._tensors: dict[str, Any] | None = None
        self.fused: dict[str, tuple[str, str]] = {}  # an HF bias name -> the file's (gate, up) biases it interleaves
        self.undo: dict[str, tuple[str, Undo]] = {}  # a mapped name -> the file tensor it reads and how to invert it
        self.derived: dict[str, Derived] = {}  # an HF name -> the file tensors it is rebuilt from (qwen4exp)

    @property
    def reader(self) -> Any:
        if self._reader is None:
            self._reader = gguf_lib().GGUFReader(self.path)
        return self._reader

    @property
    def tensors(self) -> dict[str, Any]:
        if self._tensors is None:
            self._tensors = {t.name: t for t in self.reader.tensors}
        return self._tensors

    @property
    def arch(self) -> str:
        g = gguf_lib()
        return str(self.reader.get_field(g.Keys.General.ARCHITECTURE).contents())

    @property
    def model_type(self) -> str:
        """transformers' model_type for the file's architecture; a family btb does not run is an error naming it"""
        mt = ARCH_MODEL_TYPES.get(self.arch)
        if mt is None:
            raise UnsupportedModel(self.file, self.arch, ARCH_MODEL_TYPES)
        return mt

    def _parsed(self) -> dict[str, Json]:
        """the metadata by transformers' names, through its own tables (`GGUF_TO_TRANSFORMERS_MAPPING`, keyed by
        model_type) and field parser: `config`, `tokenizer` (the converter's input) and `tokenizer_config`.
        Read here, not through `load_gguf_checkpoint`: its gpt-oss branch reads a rope key's length as its value."""
        from transformers.modeling_gguf_pytorch_utils import GGUF_TO_TRANSFORMERS_MAPPING, read_field

        arch, mt = self.arch, self.model_type
        out: dict[str, Json] = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}
        for key in self.reader.fields:
            k = mt + key[len(arch) :] if key.startswith(arch + ".") else key
            prefix, _, rest = k.partition(".")
            value = read_field(self.reader, key)
            if len(value) == 1:
                value = value[0]
            tables: dict[str, Any] = GGUF_TO_TRANSFORMERS_MAPPING
            for part, table in tables.items():
                if prefix in table and rest in table[prefix]:
                    renamed = table[prefix][rest]
                    if renamed is not None and renamed != -1:
                        out[part][renamed] = value
        cfg = out["config"]
        cfg["model_type"] = mt
        cfg.setdefault("vocab_size", len(out["tokenizer"].get("tokens") or []))
        cfg["tie_word_embeddings"] = "output.weight" not in self.tensors
        return out

    def _ints(self, key: str) -> list[int] | None:
        """the typed metadata key `key` (a `{arch}` template) as integers, None where the file lacks it"""
        field = self.reader.get_field(key.format(arch=self.arch))
        if field is None:
            return None
        v = field.contents()
        return [int(x) for x in v] if isinstance(v, list) else [int(v)]

    def _num(self, key: str) -> float | None:
        field = self.reader.get_field(key.format(arch=self.arch))
        return None if field is None else float(field.contents())

    def _qwen35_fields(self) -> Json:
        """Qwen3.5's config off its typed keys (transformers has no GGUF table for it), inverting what llama.cpp's
        converter writes: MTP blocks counted into block_count, the value head width folded into ssm.inner_size,
        the rotary as a dimension count and the layer pattern as full_attention_interval"""
        g = gguf_lib()
        llm, att, ssm, rope = g.Keys.LLM, g.Keys.Attention, g.Keys.SSM, g.Keys.Rope
        out: Json = {}
        for key, name in (
            (llm.CONTEXT_LENGTH, "max_position_embeddings"),
            (llm.EMBEDDING_LENGTH, "hidden_size"),
            (llm.FEED_FORWARD_LENGTH, "intermediate_size"),
            (llm.VOCAB_SIZE, "vocab_size"),
            (att.HEAD_COUNT, "num_attention_heads"),
            (att.HEAD_COUNT_KV, "num_key_value_heads"),
            (ssm.CONV_KERNEL, "linear_conv_kernel_dim"),
            (ssm.STATE_SIZE, "linear_key_head_dim"),
            (ssm.GROUP_COUNT, "linear_num_key_heads"),
            (ssm.TIME_STEP_RANK, "linear_num_value_heads"),
        ):
            v = self._ints(key)
            if v is not None:
                out[name] = v[0]
        eps = self._num(att.LAYERNORM_RMS_EPS)
        if eps is not None:
            out["rms_norm_eps"] = eps
        n = (self._ints(llm.BLOCK_COUNT) or [0])[0] - (self._ints(llm.NEXTN_PREDICT_LAYERS) or [0])[0]
        every = (self._ints(llm.FULL_ATTENTION_INTERVAL) or [4])[0]  # llama.cpp's qwen35 loader default
        out["num_hidden_layers"] = n
        out["layer_types"] = ["full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)]
        inner = self._ints(ssm.INNER_SIZE)
        if inner is not None:
            out["linear_value_head_dim"] = inner[0] // int(out["linear_num_value_heads"])
        head = (self._ints(att.KEY_LENGTH) or [int(out["hidden_size"]) // int(out["num_attention_heads"])])[0]
        dims = self._ints(rope.DIMENSION_COUNT)
        partial = dims[0] / head if dims is not None else 1.0
        # llama.cpp runs QWEN35 on interleaved MRoPE; its sections carry a fourth (zero) entry transformers lacks
        params: Json = {"rope_type": "default", "partial_rotary_factor": partial, "mrope_interleaved": True}
        theta = self._num(rope.FREQ_BASE)
        if theta is not None:
            params["rope_theta"] = theta
        sections = self._ints(rope.DIMENSION_SECTIONS)
        if sections is not None:
            params["mrope_section"] = sections[:3]
        out["partial_rotary_factor"] = partial
        out["rope_parameters"] = params
        return out

    def config(self) -> Any:
        from transformers import AutoConfig

        fields = self._parsed()["config"]
        if self.model_type is ModelType.QWEN3_5_TEXT:
            fields.update(self._qwen35_fields())
        if self.arch == QWEN4EXP:
            fields.update(self._qwen4exp_config())
        cfg = AutoConfig.for_model(fields.pop("model_type"), **fields)
        # the head size is in the file (attention.key_length) but not in transformers' table for these families,
        # which leaves the family's default: read it off the typed key when the file carries it
        g = gguf_lib()
        field = self.reader.get_field(g.Keys.Attention.KEY_LENGTH.format(arch=self.arch))
        inner = getattr(cfg, "text_config", cfg)
        targets = (cfg, inner) if inner is not cfg else (cfg,)
        if field is not None:
            head_dim = int(field.contents())
            for c in targets:
                if getattr(c, "head_dim", None) != head_dim:
                    c.head_dim = head_dim
        # the sliding window's key is attention.sliding_window; the table lists it under the bare name
        field = self.reader.get_field(g.Keys.Attention.SLIDING_WINDOW.format(arch=self.arch))
        if field is not None:
            for c in targets:
                c.sliding_window = int(field.contents())
        # rope scaling is in the file (rope.scaling.*) but not in transformers' table either: the family's default
        # would stand in for it (gpt-oss's yarn factor is the release's, a fixture's is not)
        rope: Json = {}
        for key, name, cast in (
            (g.Keys.Rope.SCALING_TYPE, "rope_type", str),
            (g.Keys.Rope.SCALING_FACTOR, "factor", float),
            (g.Keys.Rope.SCALING_ORIG_CTX_LEN, "original_max_position_embeddings", int),
            (g.Keys.Rope.SCALING_YARN_BETA_FAST, "beta_fast", float),
            (g.Keys.Rope.SCALING_YARN_BETA_SLOW, "beta_slow", float),
        ):
            field = self.reader.get_field(key.format(arch=self.arch))
            if field is not None:
                rope[name] = cast(field.contents())
        if rope.get("rope_type") not in (None, "none"):
            for c in targets:
                params = dict(getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {})
                params.update(rope)
                c.rope_parameters = params
        return cfg

    def _meta(self, path: str) -> Any:
        """the value of the metadata key at gguf.Keys `path` (`meta_key`), None where the file has no such key"""
        field = self.reader.get_field(meta_key(self.arch, path))
        return None if field is None else field.contents()

    def _qwen4exp_config(self) -> Json:
        """transformers' Qwen4ExpTextConfig off the keys llama.cpp's converter writes (conversion/qwen4exp.py and
        the Qwen3.5 / Qwen3-Next converters under it), which transformers' table does not read. The gate activation
        and the top-k normalization are fixed by llama.cpp's graph, and the n-gram table is one shard as stored."""
        m = self._meta
        head_dim = int(m("Attention.KEY_LENGTH"))
        v_heads = int(m("SSM.TIME_STEP_RANK"))
        recurrent = [bool(x) for x in m("Attention.RECURRENT_LAYERS")]
        # one compression ratio for every sparse layer; 0 is the converter losing it (it reads the checkpoint's
        # "full_attention" layers through transformers, which renames them), not a ratio QSA can run
        ratios = {int(r) for r, rec in zip(m("Attention.COMPRESS_RATIOS"), recurrent, strict=True) if not rec}
        if len(ratios) != 1 or 0 in ratios:
            raise RuntimeError(f"[gguf] {self.file}: the sparse layers' compression ratios are {sorted(ratios)}")
        index_dim = int(m("Attention.Indexer.KEY_LENGTH"))
        sparse = recurrent.index(False)
        k_proj = self.tensors[tensor_name(self.arch, "INDEXER_K_PROJ", sparse) + ".weight"]
        out: Json = {
            "hidden_size": int(m("LLM.EMBEDDING_LENGTH")),
            "num_hidden_layers": int(m("LLM.BLOCK_COUNT")),
            "num_attention_heads": int(m("Attention.HEAD_COUNT")),
            "num_key_value_heads": int(m("Attention.HEAD_COUNT_KV")),
            "head_dim": head_dim,
            "max_position_embeddings": int(m("LLM.CONTEXT_LENGTH")),
            "rms_norm_eps": float(m("Attention.LAYERNORM_RMS_EPS")),
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": float(m("Rope.FREQ_BASE")),
                "partial_rotary_factor": int(m("Rope.DIMENSION_COUNT")) / head_dim,
                "mrope_section": [int(x) for x in m("Rope.DIMENSION_SECTIONS")[:3]],
                "mrope_interleaved": True,
            },
            "layer_types": ["linear_attention" if r else "qwen_sparse_attention" for r in recurrent],
            "linear_conv_kernel_dim": int(m("SSM.CONV_KERNEL")),
            "linear_key_head_dim": int(m("SSM.STATE_SIZE")),
            "linear_num_key_heads": int(m("SSM.GROUP_COUNT")),
            "linear_num_value_heads": v_heads,
            "linear_value_head_dim": int(m("SSM.INNER_SIZE")) // v_heads,
            "num_experts": int(m("LLM.EXPERT_COUNT")),
            "num_experts_per_tok": int(m("LLM.EXPERT_USED_COUNT")),
            "moe_intermediate_size": int(m("LLM.EXPERT_FEED_FORWARD_LENGTH")),
            "shared_expert_intermediate_size": int(m("LLM.EXPERT_SHARED_FEED_FORWARD_LENGTH")),
            "norm_topk_prob": True,
            "output_gate_type": "sigmoid",
            "hc_count": int(m("HyperConnection.COUNT")),
            "hc_lowrank": int(m("HyperConnection.LOW_RANK")),
            "indexer_n_heads": int(m("Attention.Indexer.HEAD_COUNT")),
            "indexer_head_dim": index_dim,
            "indexer_kv_heads": int(k_proj.shape[-1]) // index_dim,
            "indexer_budget": int(m("Attention.Indexer.TOP_K")),
            "indexer_compress_ratio": ratios.pop(),
            "ple_layer_ids": [int(i) + 1 for i in (m("PerLayerEmbedding.LAYERS") or [])],
        }
        for key, path in (("bos_token_id", "Tokenizer.BOS_ID"), ("pad_token_id", "Tokenizer.PAD_ID")):
            if m(path) is not None:
                out[key] = int(m(path))
        out["eos_token_id"] = int(m("Tokenizer.EOS_ID"))
        if out["ple_layer_ids"]:
            ngram, per = int(m("PerLayerEmbedding.NGRAM_SIZE")), int(m("PerLayerEmbedding.HEADS_PER_NGRAM"))
            out.update(
                ngram_size=ngram,
                heads_per_ngram=per,
                ple_conv_kernel_size=int(m("PerLayerEmbedding.CONV_KERNEL")),
                ple_embed_dim=int(m("LLM.EMBD_LENGTH_PER_LAYER_INP")) * (ngram - 1) * per,
                eos_token_id=int(m("PerLayerEmbedding.EOS_TOKEN_ID")),
                # the one PLE layer's first head is the first prime from the base, so that prime is a base that
                # reproduces every head's size
                ngram_vocab_size_base=int(m("PerLayerEmbedding.HEAD_VOCAB_SIZES")[0]),
                split_ngram_parts=1,
            )
        return out

    def tokenizer(self) -> Any:
        """the tokenizer transformers rebuilds from the metadata, with every token the file types as a control
        or user-defined token marked special (transformers marks three of Qwen's; the file names them all, and
        an unmarked one - <think>, a tool tag - would tokenize as text)"""
        from tokenizers import AddedToken
        from transformers import PreTrainedTokenizerFast
        from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS, convert_gguf_tokenizer

        g = gguf_lib()
        parsed = self._parsed()
        # transformers keys its converters by architecture and has none for gpt-oss (its "gpt2" one does not
        # build): the byte-level BPE is Qwen's converter's, the pre-tokenizer the file's `pre` type names
        mt = self.model_type
        name = mt if mt in GGUF_TO_FAST_CONVERTERS else "qwen2"
        fast, extra = convert_gguf_tokenizer(name, parsed["tokenizer"])
        pre = self.reader.get_field(g.Keys.Tokenizer.PRE)
        if pre is not None and str(pre.contents()) in O200K_PRE_TYPES:
            fast.pre_tokenizer = o200k_pre_tokenizer()
        tok = PreTrainedTokenizerFast(tokenizer_object=fast, **parsed["tokenizer_config"], **extra)
        names = self.reader.get_field(g.Keys.Tokenizer.LIST)
        types = self.reader.get_field(g.Keys.Tokenizer.TOKEN_TYPE)
        if names is not None and types is not None:
            marked = {int(g.TokenType.CONTROL), int(g.TokenType.USER_DEFINED)}
            vocab = tok.get_vocab()
            special = [
                str(names.contents(i))
                for i, t in enumerate(types.contents())
                if int(t) in marked and str(names.contents(i)) in vocab
            ]
            if special:
                tok.add_tokens([AddedToken(t, special=True, normalized=False) for t in special], special_tokens=True)
        return tok

    def size(self) -> int:
        return os.path.getsize(self.path)

    def eos_ids(self) -> tuple[int, ...]:
        """where a decode stops, off the file's tokenizer keys: the end-of-sequence id and, where the file names
        them, the end-of-turn and end-of-message ids (what generation_config.json carries for a checkpoint)"""
        keys = gguf_lib().Keys.Tokenizer
        out: list[int] = []
        for key in (keys.EOS_ID, getattr(keys, "EOT_ID", None), getattr(keys, "EOM_ID", None)):
            field = self.reader.get_field(key) if key else None
            if field is not None:
                v = int(field.contents())
                if v not in out:
                    out.append(v)
        return tuple(out)

    def types(self) -> Counter[str]:
        """how many tensors of each storage type the file holds (the load line)"""
        return Counter(t.tensor_type.name for t in self.tensors.values())

    def describe(self) -> str:
        types = ", ".join(f"{n} x{c}" for n, c in self.types().most_common())
        return f"{self.file}: {self.arch}, {len(self.tensors)} tensors ({types})"

    def _shape(self, name: str) -> list[int]:
        """the file tensor `name`'s shape, outermost first (the file lists dims innermost first)"""
        return [int(x) for x in reversed(list(self.tensors[name].shape))]

    def _entry(self, name: str, shape: list[int] | None = None, kind: str | None = None) -> Json:
        """the header entry of the file's tensor `name` (read at `shape` and as `kind` where an inverse changes
        them): the span it takes in memory as bf16 (the size readers), and how the file stores it - a BF16
        tensor is bf16 bytes at `offset`, readable straight off the drive; any other type is read through `get`
        (dequantized) and lives in memory"""
        t = self.tensors[name]
        shape = shape or self._shape(name)
        return {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [0, 2 * int(np.prod(shape))],
            "gguf": {"type": kind or t.tensor_type.name, "offset": int(t.data_offset), "nbytes": int(t.n_bytes)},
        }

    def weight_map(self, hf_names: Iterable[str], n_layers: int) -> tuple[dict[str, str], Json]:
        """The family's tensor names (HF's, `model.layers.0.self_attn.q_proj.weight`) against the file's through
        llama.cpp's table: returns (HF name -> the file's tensor name, for the names the file holds) and a
        safetensors-shaped header (shape, bf16 byte span) the engine's size readers take as they take a shard's."""
        get_name = name_map(self.arch, n_layers)
        rewritten = self.arch in REWRITTEN
        hf_names = list(hf_names)
        names: dict[str, str] = {}
        hdr: Json = {}
        for hf in hf_names:
            base, suffix = hf, ""
            # llama.cpp stores the linear attention's dt_bias as its dt projection's bias
            src = hf.removesuffix(".dt_bias") + ".dt_proj.bias" if hf.endswith(".linear_attn.dt_bias") else hf
            if src.endswith((".weight", ".bias")):
                base, suffix = src.rsplit(".", 1)
                suffix = "." + suffix
            g = get_name(base)
            if g is None or g + suffix not in self.tensors:
                continue
            file = g + suffix
            undo = self._undo(hf, self._shape(file)) if rewritten else None
            if undo is None:
                names[hf], hdr[hf] = file, self._entry(file)
                continue
            block = gguf_lib().GGML_QUANT_SIZES[self.tensors[file].tensor_type][0]
            # a pure reordering of whole blocks keeps the file's name, so the packed kernels bind it as stored
            # (`raw` reorders the bytes); anything else is read through `get` under the HF name
            as_stored = block > 1 and undo.value is None and undo.shape is None
            name = file
            if not as_stored or (undo.cols is not None and _block_perm(undo.cols, block) is None):
                name = hf
            self.undo[name] = (file, undo)
            names[hf] = name
            hdr[hf] = self._entry(file, list(undo.shape or self._shape(file)), None if name == file else "undone")
        mx_experts = self.mxfp4_experts()
        if self.arch == QWEN4EXP:
            self._map_qwen4exp(hf_names, get_name, names, hdr, rebuild_experts=not mx_experts)
        if self.arch != QWEN4EXP or mx_experts:
            self._map_experts(hf_names, n_layers, names, hdr)
        return names, hdr

    def mxfp4_experts(self) -> bool:
        """whether the file stores its experts as MXFP4 (gpt-oss's, or llama.cpp's MXFP4_MOE of any MoE)"""
        return any(n.endswith("_exps.weight") and t.tensor_type.name == "MXFP4" for n, t in self.tensors.items())

    def _undo(self, hf: str, shape: list[int]) -> Undo | None:
        """the inverse of llama.cpp's Qwen3.5 converter (`Qwen3_5TextModel.modify_tensors`, which qwen4exp's runs
        too) on the tensor HF calls `hf`, whose file shape is `shape`: its value heads back from the tiled order
        to grouped by key head, A_log from -exp(A_log), the +1 gammas, the squeezed convolutions"""
        ssm = gguf_lib().Keys.SSM
        nk = (self._ints(ssm.GROUP_COUNT) or [1])[0]
        nv = (self._ints(ssm.TIME_STEP_RANK) or [nk])[0]
        hk = (self._ints(ssm.STATE_SIZE) or [0])[0]
        hv = (self._ints(ssm.INNER_SIZE) or [0])[0] // nv
        heads = one = qkv = None
        if nv != nk:  # the converter reorders only where key and value heads differ
            heads, one = _grouped(nk, nv // nk, hv), _grouped(nk, nv // nk, 1)
            qkv = np.concatenate([np.arange(2 * nk * hk), 2 * nk * hk + heads])
        lin = ".linear_attn."
        if hf.endswith(lin + "in_proj_qkv.weight"):
            undo = Undo(rows=qkv)
        elif hf.endswith(lin + "in_proj_z.weight"):
            undo = Undo(rows=heads)
        elif hf.endswith((lin + "in_proj_a.weight", lin + "in_proj_b.weight", lin + "dt_bias")):
            undo = Undo(rows=one)
        elif hf.endswith(lin + "A_log"):
            undo = Undo(rows=one, value=lambda x: torch.log(-x))
        elif hf.endswith(lin + "conv1d.weight"):
            undo = Undo(rows=qkv, shape=(shape[0], 1, shape[-1]))
        elif hf.endswith(".ple.conv1d.weight"):
            undo = Undo(shape=(shape[0], 1, shape[-1]))
        elif hf.endswith(lin + "out_proj.weight"):
            undo = Undo(cols=heads)
        elif _plus_one(hf):
            undo = Undo(value=lambda x: x - 1)
        else:
            return None
        return None if all(x is None for x in (undo.rows, undo.cols, undo.value, undo.shape)) else undo

    def _stored(self, t: Any, undo: Undo) -> np.ndarray:
        """a block-typed tensor's bytes [rows, row bytes] in the checkpoint's order: rows gathered whole, columns
        as whole blocks (weight_map keeps a file name only where they are)"""
        g = gguf_lib()
        block, size = g.GGML_QUANT_SIZES[t.tensor_type]
        rows = int(t.shape[-1])
        b = np.asarray(t.data).reshape(rows, -1)
        if undo.rows is not None:
            b = b[undo.rows]
        if undo.cols is not None:
            blocks = _block_perm(undo.cols, block)
            assert blocks is not None
            b = b.reshape(rows, -1, size)[:, blocks].reshape(rows, -1)
        return np.ascontiguousarray(b)

    def _map_qwen4exp(
        self, hf_names: list[str], get_name: NameMap, names: dict[str, str], hdr: Json, rebuild_experts: bool
    ) -> None:
        """qwen4exp's checkpoint tensors that are not one file tensor each (conversion/qwen4exp.py): the indexer's
        projection and (`rebuild_experts`: experts the store dequantizes) the experts' gate/up, each split in two
        by the converter and joined on read, the experts' down projection under its stacked name, and the n-gram
        table as its one shard with its hash constants off the metadata. Its per-tensor rewrites are the Qwen3.5
        converter's, inverted in `weight_map`; MXFP4 experts are `_map_experts`' spans, multiplied as stored."""

        def rows(a: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(a, axis=-2)

        def derive(hf: str, files: tuple[str, ...], fn: Invert, shape: list[int], dtype: str = "BF16") -> None:
            names[hf] = hf
            hdr[hf] = {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, (8 if dtype == "I64" else 2) * int(np.prod(shape))],
                "gguf": {"type": "derived", "offset": 0, "nbytes": 0},
            }
            self.derived[hf] = (files, fn)

        def const(path: str) -> Invert:
            values = np.array([int(x) for x in self._meta(path)], dtype=np.int64)
            return lambda _: values

        consts = {
            "layer_multipliers": "PerLayerEmbedding.LAYER_MULTIPLIERS",
            "ngram_heads_offsets": "PerLayerEmbedding.HEAD_OFFSETS",
            "ngram_heads_vocab_sizes": "PerLayerEmbedding.HEAD_VOCAB_SIZES",
        }
        shape = self._shape
        for hf in hf_names:
            layer, _, rest = hf.removeprefix("model.layers.").partition(".")
            pre = f"model.layers.{layer}."
            if rest == "self_attn.indexer.index_qk_proj.weight":
                qk_proj = tuple(
                    tensor_name(self.arch, m, int(layer)) + ".weight" for m in ("INDEXER_Q_PROJ", "INDEXER_K_PROJ")
                )
                derive(hf, qk_proj, rows, [shape(qk_proj[0])[0] + shape(qk_proj[1])[0], shape(qk_proj[0])[1]])
            elif rest == "mlp.experts.gate_up_proj" and rebuild_experts:
                gu = tuple(f"{get_name(pre + 'mlp.experts.' + k)}.weight" for k in ("gate_proj", "up_proj"))
                e, i, h = shape(gu[0])
                derive(hf, gu, rows, [e, 2 * i, h])
            elif rest == "mlp.experts.down_proj" and rebuild_experts:
                f = f"{get_name(pre + 'mlp.experts.down_proj')}.weight"
                names[hf], hdr[hf] = f, self._entry(f)
            elif rest.startswith("ple.ple_embedding.") and rest.rpartition(".")[2] in consts:
                c = const(consts[rest.rpartition(".")[2]])
                derive(hf, (), c, [len(c([]))], "I64")
                shard = pre + "ple.ple_embedding.ngram_embedding.shard_0.weight"
                table = tensor_name(self.arch, "PER_LAYER_TOKEN_EMBD") + ".weight"
                names[shard], hdr[shard] = table, self._entry(table)

    def _map_experts(self, hf_names: Iterable[str], n_layers: int, names: dict[str, str], hdr: Json) -> None:
        """gpt-oss's experts, which llama.cpp's table does not name: the file's `ffn_{gate,up,down}_exps` tensors
        under their own names with their byte spans (the expert store reads an expert's span straight into its
        slot and the kernels multiply ggml's blocks as stored), the down bias under its HF name, and the HF
        `gate_up_proj_bias` as the two biases interleaved on read (`get`)"""
        for i in range(int(n_layers)):
            pre = f"blk.{i}.ffn_"
            if pre + "gate_exps.weight" not in self.tensors:
                continue  # a dense layer between MoE ones
            for k in ("gate", "up", "down"):
                t = self.tensors[pre + k + "_exps.weight"]
                if t.tensor_type.name != "MXFP4":
                    raise RuntimeError(f"[gguf] {self.file}: {t.name} is {t.tensor_type.name}; btb runs MXFP4 experts")
                names[t.name] = t.name
                hdr[t.name] = {
                    "dtype": "U8",
                    "shape": [int(x) for x in reversed(list(t.shape))],
                    "data_offsets": [int(t.data_offset), int(t.data_offset) + int(t.n_bytes)],
                    "gguf": {"type": "MXFP4", "offset": int(t.data_offset), "nbytes": int(t.n_bytes)},
                }
            base = f"model.layers.{i}.mlp.experts."
            if pre + "down_exps.bias" in self.tensors and base + "down_proj_bias" in hf_names:
                t = self.tensors[pre + "down_exps.bias"]
                names[base + "down_proj_bias"] = t.name
                hdr[base + "down_proj_bias"] = {
                    "dtype": "BF16",
                    "shape": [int(x) for x in reversed(list(t.shape))],
                    "data_offsets": [0, 2 * int(t.n_elements)],
                    "gguf": {"type": t.tensor_type.name, "offset": int(t.data_offset), "nbytes": int(t.n_bytes)},
                }
            if pre + "gate_exps.bias" in self.tensors and base + "gate_up_proj_bias" in hf_names:
                t = self.tensors[pre + "gate_exps.bias"]
                E, I = (int(x) for x in reversed(list(t.shape)))
                self.fused[base + "gate_up_proj_bias"] = (pre + "gate_exps.bias", pre + "up_exps.bias")
                names[base + "gate_up_proj_bias"] = base + "gate_up_proj_bias"
                hdr[base + "gate_up_proj_bias"] = {
                    "dtype": "BF16",
                    "shape": [E, 2 * I],
                    "data_offsets": [0, 2 * E * 2 * I],
                    "gguf": {"type": "fused", "offset": 0, "nbytes": 0},
                }

    def raw(self, name: str) -> torch.Tensor:
        """the file's tensor `name` as its stored bytes, a uint8 copy (the kernels read ggml's blocks as they are)"""
        t = self.tensors[name]
        data = self._stored(t, self.undo[name][1]) if name in self.undo else np.asarray(t.data)
        return torch.from_numpy(np.array(data, dtype=np.uint8).reshape(-1))

    def packable(self) -> int:
        """how many of the file's tensors the packed kernels take as stored"""
        return sum(1 for t in self.tensors.values() if t.tensor_type.name in AFFINE_TYPES)

    def _float32(self, name: str, expert: int | None) -> np.ndarray:
        """the file's tensor `name` (its `expert`-th slice where given) dequantized to float32 by the gguf package"""
        t = self.tensors[name]
        data = np.asarray(t.data)
        return np.asarray(gguf_lib().dequantize(data if expert is None else data[expert], t.tensor_type))

    def get(self, name: str, expert: int | None = None) -> torch.Tensor:
        """the file's tensor `name` as bf16 (the `expert`-th of a stacked expert tensor where given): read off the
        memmap and, for a quantized type, dequantized by the gguf package to the numbers llama.cpp itself would
        compute; a `fused` name is its two biases interleaved, an `undo` one the converter's rewrite inverted and
        a `derived` one rebuilt from the file's, both in float32 (int64 for a hash constant)"""
        if name in self.fused:
            gate, up = (self.get(n) for n in self.fused[name])
            return torch.stack([gate, up], dim=-1).reshape(gate.shape[0], -1)
        if name in self.derived:
            files, fn = self.derived[name]
            out = fn([self._float32(f, expert) for f in files])
            if out.dtype == np.int64:
                return torch.from_numpy(out.copy())
            return torch.from_numpy(np.array(out, dtype=np.float32)).to(torch.bfloat16)
        if name in self.undo:
            src, undo = self.undo[name]
            x = torch.from_numpy(np.array(self._float32(src, None), dtype=np.float32))
            if undo.rows is not None:
                x = x[torch.from_numpy(undo.rows)]
            if undo.cols is not None:
                x = x[:, torch.from_numpy(undo.cols)]
            if undo.value is not None:
                x = undo.value(x)  # in float32, as the converter rewrote it
            return x.reshape(undo.shape or x.shape).contiguous().to(torch.bfloat16)
        arr = self._float32(name, expert)
        if not arr.flags.writeable:  # a plain-float tensor comes back as a view of the file: torch wants a copy
            arr = np.array(arr)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)


O200K_PRE_TYPES = ("gpt-4o",)  # llama.cpp's tokenizer.ggml.pre names for OpenAI's o200k pre-tokenization
# OpenAI's o200k pre-tokenization pattern (tiktoken's o200k_base; the HF gpt-oss tokenizer.json carries the same)
O200K_PATTERN = "|".join(
    (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    )
)


def o200k_pre_tokenizer() -> Any:
    """the `tokenizers` pre-tokenizer of an o200k vocabulary: the pattern's pieces, then byte-level without its
    own split (what the HF gpt-oss tokenizer.json declares)"""
    from tokenizers import Regex, pre_tokenizers

    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(O200K_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


EXPERTS_SUFFIX = "-experts.safetensors"


def _block_perm(perm: np.ndarray, block: int) -> np.ndarray | None:
    """a column gather `perm` as a gather of whole blocks of `block` values, or None where it splits a block"""
    if perm.size % block:
        return None
    runs = perm.reshape(-1, block)
    if np.any(runs[:, 0] % block) or np.any(runs - runs[:, :1] != np.arange(block)):
        return None
    return runs[:, 0] // block


def config_of(path: str) -> Any:
    """the model's config: a `.gguf` file's off its metadata, a directory's off its config.json"""
    if is_gguf(path):
        return GGUFModel(path).config()
    if os.path.isdir(path) and not os.path.exists(os.path.join(path, "config.json")):
        # no config.json is not a loadable HF directory - most often a GGUF repo fetched by its bare id, whose
        # .gguf files load one at a time; say that instead of transformers' opaque "unrecognized model" error
        raise NotAModel(path, "no config.json; a GGUF repo loads one file at a time, as repo/id:file.gguf")
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(path)
