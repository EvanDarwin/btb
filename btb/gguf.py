# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""A GGUF model (llama.cpp's file format) behind the engine's weight map. The file's metadata makes the config
and the tokenizer through transformers' GGUF support; the family's tensor names map to the file's through
llama.cpp's own per-architecture table (`gguf.TensorNameMap`, no name parsing); a tensor is read on demand from
the memmapped file by the `gguf` package, which also dequantizes it. Nothing of that
package is copied here: this is the glue between it and the engine."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .hf import ARCH_MODEL_TYPES, gguf_lib, is_gguf
from .kinds import Json
from .options import UnsupportedModel
from .quant import AFFINE_TYPES


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

    def config(self) -> Any:
        from transformers import AutoConfig

        fields = self._parsed()["config"]
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

    def _arch_enum(self) -> Any:
        g = gguf_lib()
        for arch, name in g.MODEL_ARCH_NAMES.items():
            if name == self.arch:
                return arch
        raise KeyError(self.arch)

    def weight_map(self, hf_names: Iterable[str], n_layers: int) -> tuple[dict[str, str], Json]:
        """The family's tensor names (HF's, `model.layers.0.self_attn.q_proj.weight`) against the file's through
        llama.cpp's table: returns (HF name -> the file's tensor name, for the names the file holds) and a
        safetensors-shaped header (shape, bf16 byte span) the engine's size readers take as they take a shard's."""
        tmap = gguf_lib().TensorNameMap(self._arch_enum(), int(n_layers))
        hf_names = list(hf_names)
        names: dict[str, str] = {}
        hdr: Json = {}
        for hf in hf_names:
            base, suffix = hf, ""
            if hf.endswith((".weight", ".bias")):
                base, suffix = hf.rsplit(".", 1)
                suffix = "." + suffix
            g = tmap.get_name(base)
            if g is None or g + suffix not in self.tensors:
                continue
            t = self.tensors[g + suffix]
            shape = [int(x) for x in reversed(list(t.shape))]  # the file lists dims innermost first
            n = 1
            for d in shape:
                n *= d
            names[hf] = g + suffix
            # the span the tensor takes in memory as bf16 (the size readers), and how the file stores it: a BF16
            # tensor is bf16 bytes at `offset`, readable straight off the drive; any other type is read through
            # `get` (dequantized) and lives in memory
            hdr[hf] = {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [0, 2 * n],
                "gguf": {"type": t.tensor_type.name, "offset": int(t.data_offset), "nbytes": int(t.n_bytes)},
            }
        self._map_experts(hf_names, n_layers, names, hdr)
        return names, hdr

    def _map_experts(self, hf_names: Iterable[str], n_layers: int, names: dict[str, str], hdr: Json) -> None:
        """gpt-oss's experts, which llama.cpp's table does not name: the file's `ffn_{gate,up,down}_exps` tensors
        under their own names with their byte spans (the expert store reads an expert's span straight into its
        slot and the kernels multiply ggml's blocks as stored), the down bias under its HF name, and the HF
        `gate_up_proj_bias` as the two biases interleaved on read (`get`)"""
        for i in range(int(n_layers)):
            pre = f"blk.{i}.ffn_"
            if pre + "gate_exps.weight" not in self.tensors:
                return
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
        return torch.from_numpy(np.array(np.asarray(self.tensors[name].data), dtype=np.uint8).reshape(-1))

    def affine(self, name: str) -> Affine | None:
        """the file's tensor `name` for the packed kernels, or None where its storage type has no affine form
        (it is then read through `get`, dequantized)"""
        return affine_of(self.tensors[name])

    def packable(self) -> int:
        """how many of the file's tensors the packed kernels take as stored"""
        return sum(1 for t in self.tensors.values() if t.tensor_type.name in AFFINE_TYPES)

    def get(self, name: str) -> torch.Tensor:
        """the file's tensor `name` as bf16: read off the memmap and, for a quantized type, dequantized by the
        gguf package to the numbers llama.cpp itself would compute; a `fused` name is its two biases interleaved"""
        g = gguf_lib()
        if name in self.fused:
            gate, up = (self.get(n) for n in self.fused[name])
            return torch.stack([gate, up], dim=-1).reshape(gate.shape[0], -1)
        t = self.tensors[name]
        arr = g.dequantize(np.asarray(t.data), t.tensor_type)
        if not arr.flags.writeable:  # a plain-float tensor comes back as a view of the file: torch wants a copy
            arr = np.array(arr)
        return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)


# The affine storage types (quant.AFFINE_TYPES) are read from the block layouts the GGUF format documents into
# the form MLX's quantized_matmul multiplies as stored; held to the package's own dequantization by the tests.
GROUP = 32


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


def _f16(b: np.ndarray) -> np.ndarray:
    """two bytes per block as float16, widened, the last axis dropped"""
    return b.copy().view(np.float16).astype(np.float32)[..., 0]


def _affine_blocks(raw: np.ndarray, rows: int, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q, scale, bias) of a tensor's blocks: q the integers as stored [rows, groups, 32] (unsigned), scale and
    bias [rows, groups] float32 with value = q * scale + bias, the number the format defines for the block"""
    if kind == "Q4_0":  # 18 bytes: d, then 16 bytes of nibbles (the low nibbles the first 16 values); v = d (q - 8)
        b = raw.reshape(rows, -1, 18)
        d = _f16(b[:, :, :2])
        qs = b[:, :, 2:]
        return np.concatenate([qs & 0xF, qs >> 4], axis=-1), d, -8.0 * d
    if kind == "Q4_1":  # 20 bytes: d, m, 16 bytes of nibbles; v = d q + m
        b = raw.reshape(rows, -1, 20)
        d, m = _f16(b[:, :, :2]), _f16(b[:, :, 2:4])
        qs = b[:, :, 4:]
        return np.concatenate([qs & 0xF, qs >> 4], axis=-1), d, m
    if kind == "Q8_0":  # 34 bytes: d, 32 signed bytes; v = d q
        b = raw.reshape(rows, -1, 34)
        d = _f16(b[:, :, :2])
        return b[:, :, 2:].view(np.int8).astype(np.int16) + 128, d, -128.0 * d
    if kind == "Q4_K":
        # 144 bytes for 256 values: d, dmin, 12 bytes holding a 6-bit scale and a 6-bit min for each of 8
        # sub-blocks of 32, then 128 bytes of nibbles (32 bytes a pair of sub-blocks: low nibbles the first,
        # high the second); v = d sc q - dmin mn
        b = raw.reshape(rows, -1, 144)
        d, dm = _f16(b[:, :, :2]), _f16(b[:, :, 2:4])
        s = b[:, :, 4:16].astype(np.int32)
        sc = np.zeros(b.shape[:2] + (8,), np.int32)
        mn = np.zeros_like(sc)
        for j in range(8):
            if j < 4:
                sc[..., j] = s[..., j] & 63
                mn[..., j] = s[..., j + 4] & 63
            else:
                sc[..., j] = (s[..., j + 4] & 0xF) | ((s[..., j - 4] >> 6) << 4)
                mn[..., j] = (s[..., j + 4] >> 4) | ((s[..., j] >> 6) << 4)
        qs = b[:, :, 16:]
        q = np.concatenate(
            [
                np.stack([qs[..., 32 * k : 32 * k + 32] & 0xF, qs[..., 32 * k : 32 * k + 32] >> 4], axis=-2)
                for k in range(4)
            ],
            axis=-2,
        )  # [rows, super-blocks, 8, 32]
        return q.reshape(rows, -1, 32), (d[..., None] * sc).reshape(rows, -1), (-dm[..., None] * mn).reshape(rows, -1)
    raise KeyError(kind)


class Affine:
    """A tensor as stored, in the affine form MLX multiplies: `wq` uint32 [rows, cols * bits / 32] with the
    integers packed low bits first, `scales` and `biases` float32 [rows, cols / 32], `bits`; `group` 32."""

    group = GROUP

    def __init__(
        self, wq: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int, shape: tuple[int, int]
    ) -> None:
        self.wq, self.scales, self.biases, self.bits, self.shape = wq, scales, biases, bits, shape


def affine_of(t: Any) -> Affine | None:
    """the tensor repacked for the packed kernels, or None for a storage type without that form"""
    kind = t.tensor_type.name
    bits = AFFINE_TYPES.get(kind)
    if bits is None:
        return None
    shape = tuple(int(x) for x in reversed(list(t.shape)))
    if len(shape) != 2 or shape[1] % GROUP:
        return None
    rows, cols = shape
    q, scale, bias = _affine_blocks(np.asarray(t.data).reshape(rows, -1), rows, kind)
    q = q.reshape(rows, cols).astype(np.uint32)
    per = 32 // bits
    words = q.reshape(rows, cols // per, per) << (np.arange(per, dtype=np.uint32) * bits)
    wq = words.sum(axis=-1, dtype=np.uint32)
    return Affine(wq, scale.astype(np.float32), bias.astype(np.float32), bits, (rows, cols))


def config_of(path: str) -> Any:
    """the model's config: a `.gguf` file's off its metadata, a directory's off its config.json"""
    if is_gguf(path):
        return GGUFModel(path).config()
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(path)
