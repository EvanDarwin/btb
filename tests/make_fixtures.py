# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
"""Regenerate every test fixture from fixed seeds, so nothing under tests/fixtures/ is a copied model artifact
and all of it can be recreated. For each family it writes a tiny random checkpoint in that family's own shape, a
byte-level tokenizer, the 12-bit sibling where one is used, its fp16/fp32 precision twins, and the receipts the
suite holds the engine to.

Four families (qwen3, q35, phi3, q4) bank the ENGINE's own float32 output (transformers is a bank-time gate: we
bank only where the engine reproduces it); gpt_oss banks transformers' float32 forward directly. The engine's
native gemv kernel is loaded first so the receipts match the tolerance the suite uses.

    python tests/make_fixtures.py            # all families
    python tests/make_fixtures.py qwen3 q4   # a subset
    python tests/make_fixtures.py twins      # only the precision twins of every cert fixture

Families: qwen3, q35, phi3, q4, gpt_oss.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import TYPE_CHECKING

from btb.kinds import Json, TokenRows
from tests.helpers import (
    CHUNK,
    FIXTURES,
    NO_LOG,
    PATH,
    PROMPT_DENSE,
    PROMPT_Q35,
    forward_logits,
    host_model,
    mxfp4_random,
    native_library,
    safetensors_state,
    speculation,
    tree_next,
    tree_pass,
)

if TYPE_CHECKING:
    import torch
    from gguf import GGUFWriter
    from transformers import PretrainedConfig
    from transformers._typing import GenerativePreTrainedModel

TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)


# --- the byte tokenizer (one token per byte, ids 0..255) ---------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-character table, the one the byte-level tokenizers share."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


def write_tokenizer(model_dir: str) -> None:
    """A byte-level tokenizer beside a fixture: any text tokenizes inside its vocabulary and decodes back; the
    eos/pad byte follows the model's own eos id; a chat template of the plainest shape."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    table = _bytes_to_unicode()
    tok = Tokenizer(models.BPE(vocab={table[b]: b for b in range(256)}, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.save(os.path.join(model_dir, "tokenizer.json"))
    eos = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8")).get("eos_token_id", 1)
    eos = int(eos[0] if isinstance(eos, list) else eos)
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": TEMPLATE,
        "eos_token": table[eos % 256],
        "pad_token": table[eos % 256],
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1 << 20,
    }
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)


# --- GGUF twins: a fixture written in llama.cpp's format through the gguf package's writer ------------------

GGUF_TYPES = ("bf16", "f16", "q8_0", "q4_0")  # the storage types the reader is tested on (the ones gguf-py writes)


def write_gguf(model_dir: str, out: str, outtype: str) -> None:
    """The fixture at `model_dir` as one GGUF file: the config's numbers as the architecture's metadata keys,
    the byte-level tokenizer as a gpt2-model vocabulary, every tensor under llama.cpp's name for it
    (`TensorNameMap`, the table its converter uses), the matrices stored as `outtype`, the vectors as f32."""
    import numpy as np
    from gguf import (
        MODEL_ARCH,
        MODEL_ARCH_NAMES,
        MODEL_TENSOR,
        TENSOR_NAMES,
        GGMLQuantizationType,
        GGUFWriter,
        RopeScalingType,
        TensorNameMap,
        TokenType,
    )
    from gguf.quants import quantize

    from btb.mxfp4 import hf_to_ggml

    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    arch = {
        "qwen3": MODEL_ARCH.QWEN3,
        "phi3": MODEL_ARCH.PHI3,
        "gpt_oss": MODEL_ARCH.GPT_OSS,
        "qwen3_5_text": MODEL_ARCH.QWEN35,
    }[cfg["model_type"]]
    L = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
    w = GGUFWriter(out, MODEL_ARCH_NAMES[arch])
    w.add_name(os.path.basename(model_dir))
    w.add_block_count(L)
    w.add_context_length(int(cfg.get("max_position_embeddings", 4096)))
    w.add_embedding_length(int(cfg["hidden_size"]))
    w.add_feed_forward_length(int(cfg["intermediate_size"]))
    w.add_head_count(heads)
    w.add_head_count_kv(int(cfg.get("num_key_value_heads") or heads))
    rope = cfg.get("rope_parameters") or {}
    w.add_rope_freq_base(float(rope.get("rope_theta") or cfg.get("rope_theta") or 10000.0))
    w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps", 1e-6)))
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    if cfg.get("num_local_experts"):
        w.add_expert_count(int(cfg["num_local_experts"]))
        w.add_expert_used_count(int(cfg["num_experts_per_tok"]))
    if cfg.get("sliding_window"):
        w.add_sliding_window(int(cfg["sliding_window"]))
    scaling = cfg.get("rope_scaling") or rope
    if scaling.get("rope_type") == "yarn":
        w.add_rope_scaling_type(RopeScalingType.YARN)
        w.add_rope_scaling_factor(float(scaling["factor"]))
        w.add_rope_scaling_orig_ctx_len(int(scaling["original_max_position_embeddings"]))
        w.add_rope_scaling_yarn_beta_fast(float(scaling.get("beta_fast", 32.0)))
        w.add_rope_scaling_yarn_beta_slow(float(scaling.get("beta_slow", 1.0)))
    if arch is MODEL_ARCH.QWEN35:
        _qwen35_metadata(w, cfg, head_dim)
    tok = json.load(open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types([TokenType.NORMAL] * len(tokens))
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"].get("merges", [])]
    if merges:
        w.add_token_merges(merges)
    else:  # a byte-level vocabulary has no merges: transformers builds the (empty) list from token scores
        w.add_token_scores([0.0] * len(tokens))
    w.add_vocab_size(len(tokens))
    eos = cfg.get("eos_token_id", 1)
    w.add_eos_token_id(int(eos[0] if isinstance(eos, list) else eos))
    tcfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8"))
    if tcfg.get("chat_template"):
        w.add_chat_template(tcfg["chat_template"])
    state = safetensors_state(model_dir)
    if arch is MODEL_ARCH.QWEN35:
        state = _qwen35_converted(cfg, state)
    tmap = TensorNameMap(arch, L)
    qtype = {
        "f16": None,
        "bf16": GGMLQuantizationType.BF16,
        "q8_0": GGMLQuantizationType.Q8_0,
        "q4_0": GGMLQuantizationType.Q4_0,
        "q4_1": GGMLQuantizationType.Q4_1,
    }[outtype]
    conv = {TENSOR_NAMES[MODEL_TENSOR.SSM_CONV1D].format(bid=i) for i in range(L)}  # llama.cpp keeps these f32
    for name in sorted(state):
        if ".mlp.experts." in name:
            continue  # the MXFP4 experts below, in ggml's layout
        base, suffix = name.rsplit(".", 1)
        gname = tmap.get_name(base) if suffix in ("weight", "bias") else tmap.get_name(name)
        if gname is None:
            raise RuntimeError(f"no GGUF name for {name}")
        if suffix not in ("weight", "bias"):  # a bare parameter (gpt-oss's attention sinks, Qwen3.5's A_log)
            w.add_tensor(gname, state[name].float().numpy())
            continue
        arr = state[name].float().numpy()
        if gname in conv:
            w.add_tensor(f"{gname}.{suffix}", arr.astype(np.float32))
        elif arr.ndim == 2 and qtype is not None and arr.shape[-1] % 32 == 0:
            w.add_tensor(f"{gname}.{suffix}", quantize(arr, qtype), raw_dtype=qtype)
        elif arr.ndim == 2:
            w.add_tensor(f"{gname}.{suffix}", arr.astype(np.float16))  # f16: bf16's small values lose bits here
        else:
            w.add_tensor(f"{gname}.{suffix}", arr.astype(np.float32))
    for i in range(L):
        p = f"model.layers.{i}.mlp.experts."
        if p + "gate_up_proj_blocks" not in state:
            continue
        # HF's gate_up rows alternate gate, up; ggml keeps them as two tensors of 17-byte blocks
        gb, gs = state[p + "gate_up_proj_blocks"].numpy(), state[p + "gate_up_proj_scales"].numpy()
        gu_bias = state[p + "gate_up_proj_bias"].float().numpy()
        for k, rows in (("gate", slice(0, None, 2)), ("up", slice(1, None, 2))):
            raw = hf_to_ggml(gb[:, rows], gs[:, rows])
            w.add_tensor(f"blk.{i}.ffn_{k}_exps.weight", raw, raw_dtype=GGMLQuantizationType.MXFP4)
            w.add_tensor(f"blk.{i}.ffn_{k}_exps.bias", np.ascontiguousarray(gu_bias[:, rows]))
        raw = hf_to_ggml(state[p + "down_proj_blocks"].numpy(), state[p + "down_proj_scales"].numpy())
        w.add_tensor(f"blk.{i}.ffn_down_exps.weight", raw, raw_dtype=GGMLQuantizationType.MXFP4)
        w.add_tensor(f"blk.{i}.ffn_down_exps.bias", state[p + "down_proj_bias"].float().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _qwen35_metadata(w: GGUFWriter, cfg: Json, head_dim: int) -> None:
    """the keys llama.cpp's Qwen3.5 converter adds (Qwen3NextModel.set_gguf_parameters): the linear attention's
    widths, the layer pattern and the rotary's width and MRoPE sections (padded to four)"""
    w.add_ssm_conv_kernel(int(cfg["linear_conv_kernel_dim"]))
    w.add_ssm_state_size(int(cfg["linear_key_head_dim"]))
    w.add_ssm_group_count(int(cfg["linear_num_key_heads"]))
    w.add_ssm_time_step_rank(int(cfg["linear_num_value_heads"]))
    w.add_ssm_inner_size(int(cfg["linear_value_head_dim"]) * int(cfg["linear_num_value_heads"]))
    w.add_full_attention_interval(int(cfg.get("full_attention_interval", 4)))
    rope = cfg.get("rope_parameters") or {}
    w.add_rope_dimension_count(int(head_dim * float(rope.get("partial_rotary_factor", 0.25))))
    sections = [int(x) for x in rope.get("mrope_section", [11, 11, 10])]
    w.add_rope_dimension_sections((sections + [0] * 4)[:4])


def _qwen35_converted(cfg: Json, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """the checkpoint as llama.cpp's Qwen3.5 converter hands it to the writer (Qwen3_5TextModel.modify_tensors,
    in float32 as it runs): the MTP head left out (its --no-mtp), the value heads reordered from grouped by key
    head to tiled, A_log as -exp(A_log), dt_bias renamed dt_proj.bias, conv1d squeezed, the norms as 1 + w"""
    import torch

    nk, nv = int(cfg["linear_num_key_heads"]), int(cfg["linear_num_value_heads"])
    hk, hv = int(cfg["linear_key_head_dim"]), int(cfg["linear_value_head_dim"])

    def tiled(t: torch.Tensor, dim: int, width: int) -> torch.Tensor:  # _reorder_v_heads
        if nk == nv:
            return t
        shape = list(t.shape)
        t = t.reshape(*shape[:dim], nk, nv // nk, width, *shape[dim + 1 :])
        return t.transpose(dim, dim + 1).contiguous().reshape(shape)

    out: dict[str, torch.Tensor] = {}
    qk = 2 * nk * hk
    for name, t in state.items():
        if name.startswith("mtp."):
            continue
        t = t.float()
        if ".in_proj_qkv." in name or ".conv1d." in name:
            t = t.squeeze() if ".conv1d." in name else t
            t = torch.cat([t[:qk], tiled(t[qk:], 0, hv)])
        elif ".in_proj_z." in name:
            t = tiled(t, 0, hv)
        elif ".in_proj_a." in name or ".in_proj_b." in name or name.endswith((".A_log", ".dt_bias")):
            t = tiled(t, 0, 1)
        elif ".out_proj." in name:
            t = tiled(t, 1, hv)
        if name.endswith(".A_log"):
            t = -torch.exp(t)
        elif name.endswith(".dt_bias"):
            name = name.removesuffix(".dt_bias") + ".dt_proj.bias"
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            t = t + 1
        out[name] = t
    return out


def make_gguf() -> None:
    """the GGUF fixtures: tiny_qwen3 in every storage type the reader is tested on, tiny_phi3 as f16 (its fused
    projections exercise the table's other layout)"""
    out = os.path.join(FIXTURES, "gguf")
    os.makedirs(out, exist_ok=True)
    for t in GGUF_TYPES:
        write_gguf(os.path.join(FIXTURES, "tiny_qwen3"), os.path.join(out, f"tiny_qwen3-{t}.gguf"), t)
    # phi3 in bf16 (its fused projections exercise the table's other layout) and the affine quants, so the
    # affine-as-stored kernels have a phi3 tiny fixture too, not only qwen3
    for t in ("bf16", "q4_0", "q8_0"):
        write_gguf(os.path.join(FIXTURES, "tiny_phi3"), os.path.join(out, f"tiny_phi3-{t}.gguf"), t)
    write_gguf(os.path.join(FIXTURES, "tiny_gpt_oss"), os.path.join(out, "tiny_gpt_oss-mxfp4.gguf"), "bf16")
    make_gguf_q35()
    print("[fixture] gguf twins regenerated")


def make_gguf_q35() -> None:
    """tiny_q35 in every float and affine type (the storage cells gguf-py can write a twin for), its linear
    attention through the converter's rewrites the reader must invert"""
    from btb.kinds import QuantClass, quants_of

    out = os.path.join(FIXTURES, "gguf")
    os.makedirs(out, exist_ok=True)
    for q in (*quants_of(QuantClass.FLOAT), *quants_of(QuantClass.AFFINE)):
        t = q.value.lower()
        write_gguf(os.path.join(FIXTURES, "tiny_q35"), os.path.join(out, f"tiny_q35-{t}.gguf"), t)


# --- weight builders: a tiny random checkpoint per family in its own shape ---------------------------------


def _reshard(out_dir: str, state: dict[str, torch.Tensor], cap: int = 100_000) -> int:
    """Write a state dict as ~100 KB safetensors shards with an index; returns the total bytes. Matches
    `save_pretrained(max_shard_size="100KB")`'s layout so the fixtures shard as they always have."""
    from safetensors.torch import save_file

    for f in os.listdir(out_dir):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            os.remove(os.path.join(out_dir, f))
    files: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    size = 0
    for k in sorted(state):
        nb = state[k].numel() * state[k].element_size()
        if cur and size + nb > cap:
            files.append(cur)
            cur, size = {}, 0
        cur[k] = state[k].detach().contiguous()
        size += nb
    if cur:
        files.append(cur)
    wm = {}
    for j, part in enumerate(files):
        name = f"model-{j + 1:05d}-of-{len(files):05d}.safetensors"
        save_file(part, os.path.join(out_dir, name), metadata={"format": "pt"})
        wm.update(dict.fromkeys(part, name))
    total = sum(v.numel() * v.element_size() for v in state.values())
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": wm}, fh, indent=2)
    return total


def build_qwen3(out_dir: str) -> None:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    kw: Json = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "hidden_act": "silu", "rope_theta": 1000000.0,
        "use_sliding_window": False, "attention_bias": False, "pad_token_id": 1, "eos_token_id": 1,
        "bos_token_id": 0, "torch_dtype": "bfloat16",
    }  # fmt: skip
    m = Qwen3ForCausalLM(Qwen3Config(**kw)).eval().to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir, max_shard_size="100KB", safe_serialization=True)


def build_phi3(out_dir: str) -> None:
    import torch
    from transformers import Phi3Config, Phi3ForCausalLM

    torch.manual_seed(0)
    rot = 12
    kw: Json = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 512,
        "original_max_position_embeddings": 64, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "hidden_act": "silu", "partial_rotary_factor": 0.75, "rope_theta": 10000.0, "sliding_window": 4096,
        "pad_token_id": 1, "eos_token_id": 1, "bos_token_id": 0,
        "rope_scaling": {"rope_type": "longrope", "long_factor": [1.0 + 0.5 * i for i in range(rot // 2)],
                         "short_factor": [1.0 for _ in range(rot // 2)]},
        "attention_bias": False, "torch_dtype": "bfloat16",
    }  # fmt: skip
    m = Phi3ForCausalLM(Phi3Config(**kw)).eval().to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir, max_shard_size="100KB", safe_serialization=True)


def build_gemma3(out_dir: str) -> None:
    import torch
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    torch.manual_seed(0)
    # 6 layers so the default 5-sliding : 1-full pattern gives at least one global-attention layer, exercising
    # the dual (local/global) rope; a small window so the sliding path is real at a tiny context. gelu_pytorch_tanh
    # and the sandwich norm come from the config's model_type, and the input embedding is scaled by sqrt(hidden).
    kw: Json = {
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 6,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 512,
        "sliding_window": 32, "rms_norm_eps": 1e-6, "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": 16,
        "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
    }  # fmt: skip
    m = Gemma3ForCausalLM(Gemma3TextConfig(**kw)).eval().to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir, max_shard_size="100KB", safe_serialization=True)


def build_q4(out_dir: str) -> None:
    import torch
    from transformers import Qwen4ExpForCausalLM
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    torch.manual_seed(0)
    lt = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
    cfg = Qwen4ExpTextConfig(
        vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False, hidden_act="silu",
        attention_bias=False, layer_types=lt,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=8,
        num_experts_per_tok=2, norm_topk_prob=True, hc_count=4, hc_lowrank=16, ple_layer_ids=[2], ple_embed_dim=64,
        ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128, seed=1234, split_ngram_parts=4, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        output_gate_type="sigmoid", pad_token_id=1, eos_token_id=1, bos_token_id=0, dtype="bfloat16",
    )  # fmt: skip
    m = Qwen4ExpForCausalLM(cfg).eval().to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir, max_shard_size="100KB", safe_serialization=True)  # config + generation_config
    sd = {k: v.detach().contiguous() for k, v in m.state_dict().items()}
    # the n-gram embedding is stored in `split_ngram_parts` shards, as the checkpoint's own loader expects
    for k in [k for k in sd if k.endswith("ngram_embedding.weight")]:
        w = sd.pop(k)
        for j, p in enumerate(torch.chunk(w, cfg.split_ngram_parts, dim=0)):
            sd[k.replace("ngram_embedding.weight", f"ngram_embedding.shard_{j}.weight")] = p.contiguous()
    _reshard(out_dir, sd)


def build_q35(out_dir: str) -> None:
    import torch
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(0)
    lt = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
    kw: Json = {
        "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64, "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
        "attn_output_gate": True, "layer_types": lt, "partial_rotary_factor": 0.25,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "pad_token_id": 1, "eos_token_id": 1,
        "bos_token_id": 0, "dtype": "bfloat16",
    }  # fmt: skip
    cfg = Qwen3_5TextConfig(**kw)
    m = Qwen3_5ForCausalLM(cfg).eval().to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir, max_shard_size="100KB", safe_serialization=True)  # config + generation_config
    sd = {k: v.detach().contiguous() for k, v in m.state_dict().items()}
    sd.update(_mtp_head(cfg))  # the speculative drafter head, appended to the trunk (not a transformers module)
    _reshard(out_dir, sd)


def _mtp_head(cfg: PretrainedConfig) -> dict[str, torch.Tensor]:
    """The q35 MTP drafter: one gated-attention decoder layer over `[embedding; hidden]`, its `fc` folding the
    pair back to the hidden size, with the pre-fc and final norms. Random bf16 in the trunk's shapes - the
    verify pass makes speculation exact whatever the drafter proposes, so the weights need only be well-formed."""
    import torch

    torch.manual_seed(1)
    h, hd, nq, nkv, inter = 128, 64, 2, 1, 256
    gate = 2 if getattr(cfg, "attn_output_gate", False) else 1

    def r(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape) * 0.02).bfloat16()

    a = {
        "self_attn.q_proj.weight": r(gate * nq * hd, h), "self_attn.k_proj.weight": r(nkv * hd, h),
        "self_attn.v_proj.weight": r(nkv * hd, h), "self_attn.o_proj.weight": r(h, nq * hd),
        "self_attn.q_norm.weight": r(hd), "self_attn.k_norm.weight": r(hd),
        "mlp.gate_proj.weight": r(inter, h), "mlp.up_proj.weight": r(inter, h), "mlp.down_proj.weight": r(h, inter),
        "input_layernorm.weight": r(h), "post_attention_layernorm.weight": r(h),
    }  # fmt: skip
    out = {f"mtp.layers.0.{k}": v for k, v in a.items()}
    out["mtp.fc.weight"] = r(h, 2 * h)
    out["mtp.norm.weight"] = r(h)
    out["mtp.pre_fc_norm_embedding.weight"] = r(h)
    out["mtp.pre_fc_norm_hidden.weight"] = r(h)
    return out


GPT_OSS = {"H": 64, "HEADS": 4, "KV_HEADS": 2, "HEAD_DIM": 64, "INTER": 64, "EXPERTS": 8, "TOP_K": 4,
           "LAYERS": 4, "VOCAB": 512, "WINDOW": 4}  # fmt: skip


def _gpt_oss_config() -> Json:
    g = GPT_OSS
    return {
        "architectures": ["GptOssForCausalLM"], "attention_bias": True, "attention_dropout": 0.0,
        "dtype": "bfloat16", "eos_token_id": 1, "experts_per_token": g["TOP_K"], "head_dim": g["HEAD_DIM"],
        "hidden_act": "silu", "hidden_size": g["H"], "initializer_range": 0.02, "intermediate_size": g["INTER"],
        "layer_types": ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(g["LAYERS"])],
        "max_position_embeddings": 128, "model_type": "gpt_oss", "num_attention_heads": g["HEADS"],
        "num_experts_per_tok": g["TOP_K"], "num_hidden_layers": g["LAYERS"], "num_key_value_heads": g["KV_HEADS"],
        "num_local_experts": g["EXPERTS"], "output_router_logits": False, "pad_token_id": 1, "rms_norm_eps": 1e-05,
        "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 8.0, "original_max_position_embeddings": 16,
                         "rope_type": "yarn", "truncate": False},
        "rope_theta": 150000, "router_aux_loss_coef": 0.9, "sliding_window": g["WINDOW"], "swiglu_limit": 7.0,
        "tie_word_embeddings": False, "use_cache": True, "vocab_size": g["VOCAB"],
    }  # fmt: skip


def _gpt_oss_tensors() -> dict[str, torch.Tensor]:
    import numpy as np
    import torch

    g = GPT_OSS
    rng = np.random.default_rng(20260907)

    def bf16(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return torch.from_numpy((rng.standard_normal(shape) * scale).astype(np.float32)).bfloat16()

    def experts_mxfp4(n: int, rows: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        blocks, scales = mxfp4_random(rng, (n, rows), k, 119, 126)
        return torch.from_numpy(blocks), torch.from_numpy(scales)

    H, HEADS, KV, HD, INTER, E = g["H"], g["HEADS"], g["KV_HEADS"], g["HEAD_DIM"], g["INTER"], g["EXPERTS"]
    t = {
        "model.embed_tokens.weight": bf16(g["VOCAB"], H, scale=0.08),
        "model.norm.weight": bf16(H, scale=0.2) + 1.0,
        "lm_head.weight": bf16(g["VOCAB"], H, scale=0.08),
    }
    for i in range(g["LAYERS"]):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = bf16(HEADS * HD, H, scale=0.15)
        t[p + "self_attn.q_proj.bias"] = bf16(HEADS * HD, scale=0.05)
        t[p + "self_attn.k_proj.weight"] = bf16(KV * HD, H, scale=0.15)
        t[p + "self_attn.k_proj.bias"] = bf16(KV * HD, scale=0.05)
        t[p + "self_attn.v_proj.weight"] = bf16(KV * HD, H, scale=0.15)
        t[p + "self_attn.v_proj.bias"] = bf16(KV * HD, scale=0.05)
        t[p + "self_attn.o_proj.weight"] = bf16(H, HEADS * HD, scale=0.15)
        t[p + "self_attn.o_proj.bias"] = bf16(H, scale=0.05)
        t[p + "self_attn.sinks"] = bf16(HEADS, scale=0.8)
        t[p + "input_layernorm.weight"] = bf16(H, scale=0.2) + 1.0
        t[p + "post_attention_layernorm.weight"] = bf16(H, scale=0.2) + 1.0
        t[p + "mlp.router.weight"] = bf16(E, H, scale=0.3)
        t[p + "mlp.router.bias"] = bf16(E, scale=0.3)
        gb, gs = experts_mxfp4(E, 2 * INTER, H)
        db, ds = experts_mxfp4(E, H, INTER)
        t[p + "mlp.experts.gate_up_proj_blocks"] = gb
        t[p + "mlp.experts.gate_up_proj_scales"] = gs
        t[p + "mlp.experts.gate_up_proj_bias"] = bf16(E, 2 * INTER, scale=0.05)
        t[p + "mlp.experts.down_proj_blocks"] = db
        t[p + "mlp.experts.down_proj_scales"] = ds
        t[p + "mlp.experts.down_proj_bias"] = bf16(E, H, scale=0.05)
    return t


def build_gpt_oss(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _reshard(out_dir, _gpt_oss_tensors())
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_gpt_oss_config(), f, indent=2)
    gen = {"_from_model_config": True, "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1}
    with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as f:
        json.dump(gen, f, indent=2)


# --- precision twins: the BF16 fixture's weights stored at the other safetensors precisions -------------------


def write_twins(base: str) -> None:
    """`base`'s precision twins (spec.twin_path): one per safetensors storage beside BF16 whose header dtype the
    engine reads (StreamedTextModel.ST_DTYPES), every float tensor cast to it and the packed integer ones kept, so
    each decodes to the same oracle as `base`. The config's `dtype` says what the twin stores."""
    from btb.engine import StreamedTextModel
    from tests.cert import spec

    state = safetensors_state(base)
    stem = os.path.basename(base)
    for storage, info in spec.STORAGE.items():
        dtype = StreamedTextModel.ST_DTYPES.get(info.fp)
        if info.container is not spec.Container.SAFETENSORS or storage is spec.Storage.SAFE_BF16 or dtype is None:
            continue
        out = spec.twin_path(stem, storage)
        os.makedirs(out, exist_ok=True)
        for name in os.listdir(base):
            if name.endswith(".safetensors") or name == "model.safetensors.index.json":
                continue  # _reshard writes the twin's own
            if name == "config.json":
                with open(os.path.join(base, name), encoding="utf-8") as f:
                    cfg: Json = json.load(f)
                cfg["dtype"] = str(dtype).removeprefix("torch.")
                with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            else:
                shutil.copyfile(os.path.join(base, name), os.path.join(out, name))
        _reshard(out, {k: v.to(dtype) if v.is_floating_point() else v for k, v in state.items()})
        print(f"[fixture] {os.path.basename(out)} written ({info.fp})")


# --- receipt bankers: run the engine the way the suite does, and save what it compares against --------------


def _greedy_hf(ref: GenerativePreTrainedModel, prompt: TokenRows) -> list[int]:
    """transformers' own greedy continuation of the prompt, ten tokens"""
    import torch

    ids = torch.tensor(prompt)
    out = ref.generate(ids, max_new_tokens=10, do_sample=False)
    assert isinstance(out, torch.Tensor)
    return out[0, ids.shape[1] :].tolist()


def bank_dense(tag: str, base_dir: str, pack_dir: str, out: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    prompt = PROMPT_DENSE
    with torch.inference_mode():
        ref = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
        ref_prompt = ref(torch.tensor(prompt)).logits[0, -1]
        ref_greedy = _greedy_hf(ref, prompt)
        seq = [*prompt[0], *(CHUNK[j] for j in PATH), CHUNK[PATH[-1]]]
        ref_next = ref(torch.tensor([seq])).logits[0, -1]
        banked = {}
        for kind, path, packed in (("bf16", base_dir, False), ("p12", pack_dir, True)):
            sm = host_model(path, packed=packed)
            cache = sm.new_cache()
            lg0 = forward_logits(sm, prompt, cache)[0, -1]
            lg, _ = tree_pass(sm, cache)
            nxt = tree_next(sm, cache)
            keys = {i: (cache.layers[i].keys.clone(), cache.layers[i].values.clone()) for i in range(sm.L)}
            speculation(sm, tree_budget=0, tree_min_prob=0.0, ngram_p=0.0, tree_read="step")
            g = sm.generate_greedy(prompt, 10)
            s = sm.generate_speculative(prompt, 10, proposer="ngram", v_max=4)[0]
            sm.close()
            if kind == "bf16":
                d = max(float((lg0 - ref_prompt).abs().max()), float((nxt - ref_next).abs().max()))
                assert g == ref_greedy, f"[{tag}] engine greedy {g} != transformers {ref_greedy}"
                assert d < 1e-4, f"[{tag}] engine vs transformers {d:.2e}"
                print(f"[{tag}] gate ok: vs transformers {d:.2e}, greedy matches")
            banked[kind] = {"prompt_logits": lg0.clone(), "chunk_logits": lg.clone(), "next": nxt.clone(),
                            "keys": keys, "greedy": g, "ngram": s}  # fmt: skip
        banked["ref"] = {"greedy": ref_greedy}
        torch.save(banked, out)


def bank_q35(base_dir: str, out: str) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    full = AutoConfig.from_pretrained(base_dir)
    cfg = getattr(full, "text_config", full)
    lin = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]
    with torch.inference_mode():
        ref = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32).eval()
        seq = [*PROMPT_Q35[0], *(CHUNK[j] for j in PATH), CHUNK[PATH[-1]]]
        ref_next = ref(torch.tensor([seq])).logits[0, -1]
        ref_greedy = _greedy_hf(ref, PROMPT_Q35)
        from btb.engine import StreamedTextModel

        sm = host_model(base_dir)
        cache = sm.new_cache()
        forward_logits(sm, PROMPT_Q35, cache)
        lg, _ = tree_pass(sm, cache)
        states = {i: tuple(x.clone() for x in StreamedTextModel._lin(cache.layers[i])) for i in lin}
        nxt = tree_next(sm, cache)
        speculation(sm, tree_budget=8, tree_min_prob=0.0, ngram_p=0.9, tree_read="step")
        g = sm.generate_greedy(PROMPT_Q35, 10)
        sm.close()
        d = float((nxt - ref_next).abs().max())
        assert g == ref_greedy, f"[q35] engine greedy {g} != transformers {ref_greedy}"
        assert d < 1e-4, f"[q35] engine next vs transformers {d:.2e}"
        print(f"[q35] gate ok: next vs transformers {d:.2e}, greedy matches")
        host = {"logits": lg.clone(), "states": states, "next": nxt.clone(), "toks": {"greedy": g}}
        torch.save({"host": host}, out)


def bank_q4(base_dir: str, out: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from btb.engine import StreamedTextModel

    StreamedTextModel.register_attention()
    torch.manual_seed(7)
    prompt = [torch.randint(2, 512, (80,)).tolist()]
    cont = [[44, 8, 3, 17, 61, 9]]
    with torch.inference_mode():
        ref = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.float32, attn_implementation="btb_sdpa", experts_implementation="eager"
        ).eval()
        r_all = ref(torch.tensor([prompt[0] + cont[0]])).logits[0]
        r_prompt, r_next = r_all[len(prompt[0]) - 1], r_all[-1]
        ref_greedy = _greedy_hf(ref, prompt)
        sm = host_model(base_dir)
        speculation(sm, tree_budget=0, v_max=0, ngram_p=0.0, tree_read="step")
        cache = sm.new_cache()
        lg0 = forward_logits(sm, prompt, cache)[0, -1]
        lg1 = forward_logits(sm, cont, cache)[0, -1]
        g = sm.generate_greedy(prompt, 10)
        sm.close()
        d0, d1 = float((lg0 - r_prompt).abs().max()), float((lg1 - r_next).abs().max())
        assert g == ref_greedy, f"[q4] engine greedy {g} != transformers {ref_greedy}"
        assert max(d0, d1) < 1e-4, f"[q4] engine vs transformers {max(d0, d1):.2e}"
        print(f"[q4] gate ok: prompt {d0:.2e}, next {d1:.2e}, greedy matches")
        host = {"prompt": prompt, "cont": cont, "prompt_logits": lg0.clone(), "next": lg1.clone(), "greedy": g,
                "hf_distance": {"prompt": d0, "next": d1},
                "reference": "transformers float32, eager experts, btb_sdpa"}  # fmt: skip
        torch.save({"host": host}, out)


def bank_gpt_oss(base_dir: str, out: str) -> None:
    """gpt_oss is the one external reference: transformers' own float32 forward, experts dequantized by
    transformers, is the receipt (the engine must reproduce it within a tolerance)."""
    import torch
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    tensors = safetensors_state(base_dir)
    cfg = GptOssConfig(**{k: v for k, v in _gpt_oss_config().items() if k != "architectures"})
    cfg._attn_implementation = "eager"
    model = GptOssForCausalLM(cfg)
    sd = {}
    for k, v in tensors.items():
        if k.endswith("_blocks"):
            sd[k[: -len("_blocks")]] = convert_moe_packed_tensors(v, tensors[k[: -len("_blocks")] + "_scales"]).float()
        elif not k.endswith("_scales"):
            sd[k] = v.float()
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()
    prompt = [[3, 17, 42, 5, 99, 120, 7, 7, 200, 12, 45, 8, 3, 17, 60, 61]]
    cont = [[31, 4, 77]]
    with torch.inference_mode():
        ids = torch.tensor(prompt)
        o = model(input_ids=ids, use_cache=True)
        prompt_logits = o.logits[0, -1].float().clone()
        nxt = model(input_ids=torch.tensor(cont), past_key_values=o.past_key_values, use_cache=True)
        nxt = nxt.logits[0, -1].float().clone()
        g_out = model(input_ids=ids, use_cache=True)
        cache, tok, greedy = g_out.past_key_values, int(g_out.logits[0, -1].argmax()), []
        greedy.append(tok)
        while len(greedy) < 10:
            o = model(input_ids=torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
            tok = int(o.logits[0, -1].argmax())
            greedy.append(tok)
    torch.save({"host": {"prompt": prompt, "cont": cont, "prompt_logits": prompt_logits, "next": nxt,
                         "greedy": greedy}}, out)  # fmt: skip


# --- orchestration -----------------------------------------------------------------------------------------

FAMILIES = ("qwen3", "q35", "phi3", "q4", "gpt_oss", "gemma3")


def make(name: str) -> None:
    if name == "gguf":
        make_gguf()
        return
    if name == "twins":
        from tests.cert import spec

        for stem in spec.FIXTURE_STEM.values():
            write_twins(os.path.join(FIXTURES, stem))
        return
    if name == "gguf_q35":
        make_gguf_q35()
        return
    base = os.path.join(FIXTURES, f"tiny_{name}")
    if name == "qwen3":
        build_qwen3(base)
    elif name == "phi3":
        build_phi3(base)
    elif name == "q4":
        build_q4(base)
    elif name == "q35":
        build_q35(base)
    elif name == "gpt_oss":
        build_gpt_oss(base)
    elif name == "gemma3":
        build_gemma3(base)
    else:
        raise SystemExit(f"unknown family {name!r}; one of {FAMILIES}")
    write_tokenizer(base)
    # every family packs to a 12-bit store (btb pack), incl. the MoE and Gemma paths, so the pack12 storage cell
    # has a tiny fixture for each - verified they load and decode on CPU
    from btb.engine import pack_model

    pack_model(base, log=NO_LOG)
    write_twins(base)
    pack = os.path.join(FIXTURES, f"tiny_{name}-pack12")
    if name in ("qwen3", "phi3"):
        bank_dense(name, base, pack, os.path.join(FIXTURES, f"receipts_{name}.pt"))
    elif name == "q35":
        bank_q35(base, os.path.join(FIXTURES, "receipts.pt"))
    elif name == "q4":
        bank_q4(base, os.path.join(FIXTURES, "receipts_q4.pt"))
    elif name == "gpt_oss":
        bank_gpt_oss(base, os.path.join(FIXTURES, "receipts_gpt_oss.pt"))
    print(f"[fixture] tiny_{name} regenerated")


def _rebank_oracle() -> None:
    """refresh the P5 correctness oracle (tests/cert/oracle) so its banked greedy references and fixture hashes
    track the fixtures just written; left alone until every banked family's fixture is on disk (a partial run)."""
    from tests.cert import oracle

    if all(os.path.isdir(oracle.fixture_dir(k)) for k in oracle.banked_kinds()):
        oracle.write_bank(oracle.bank())
        print("[fixture] cert oracle bank regenerated")
    else:
        print("[fixture] cert oracle bank left as is (not every served family's fixture is built)")


def main(argv: list[str] | None = None) -> int:
    names = argv or list(FAMILIES)
    # the receipts are banked by the native gemv kernel, so the suite's 1e-6 tolerance holds
    if native_library() is None:
        print("[fixture] warning: no native library found; receipts banked on the torch path", file=sys.stderr)
    for name in names:
        make(name)
    _rebank_oracle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
