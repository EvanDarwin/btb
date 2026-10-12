# Copyright (c) 2026 Evan Darwin - FSL-1.1-ALv2
# Written by `python build.py bindings` from native/cuda: edit a kernel or btb_roles.cuh there, then run it.
"""The card kernels' launchers: a function a kernel, taking a launcher and then the kernel's parameters in
order, each typed by its role (btb/kinds.py, from native/cuda/btb_roles.cuh): a pointer role a card address
(`ptr(Role, tensor)`), a scalar role its value. A macro's kernels share a function, the variant (the head width
D, the rows M) first. A function packs its arguments as the kernel declares them and hands them to the launcher
(`Launcher`): btb's card module (`_Cuda` in btb/engine/native.py), or one's own with its own streams. The card
module carries each function as a method: `k.btb_attn_flash(D, ...)` is `btb_attn_flash(k, D, ...)`."""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

import torch

from btb.kinds import (
    AColumn,
    Arrivals,
    AttnOut,
    AttnScale,
    AttnWidth,
    BColumn,
    BlockOut,
    BlockRatio,
    CacheHeadStride,
    CacheRowStride,
    ChunkStart,
    ConvBias,
    ConvChannels,
    ConvDilation,
    ConvState,
    ConvTaps,
    ConvWeight,
    DeltaA,
    DeltaALog,
    DeltaB,
    DeltaDtBias,
    DeltaGate,
    DeltaKDim,
    DeltaKHeads,
    DeltaMixed,
    DeltaNormWeight,
    DeltaOut,
    DeltaOutF32,
    DeltaProj,
    DeltaScratch,
    DeltaState,
    DeltaVDim,
    DeltaVHeads,
    DeltaZ,
    Depths,
    DraftKids,
    DraftProbs,
    DraftVocab,
    DraftWidth,
    ExpertIds,
    Experts,
    ExpertWeights,
    GatedAttn,
    GateHeadStride,
    GateRowStride,
    GateUp,
    HasDraftProbs,
    HeadDim,
    Hidden,
    HostClock,
    HostExpertIds,
    HostExpertWeights,
    HostLen,
    HostRouteSeq,
    HostToken,
    IdxHeads,
    IdxKNorm,
    IdxPooled,
    IdxPooledLen,
    IdxQuery,
    IdxRawKey,
    IdxScores,
    InDim,
    Inject,
    Intermediate,
    InvSqrtHidden,
    KCache,
    KColumn,
    KeepRawKey,
    KNorm,
    KvHeads,
    Logits,
    MatIn,
    MatInF32,
    MatOut,
    MatOutF32,
    MaxNewBlocks,
    MixAct,
    MixDown,
    MixDownStride,
    Mixed,
    MixRank,
    MixUp,
    MlpAct,
    MmaMisses,
    MoeOut,
    Mx4Blocks,
    Mx4Groups,
    Mx4Scales,
    NoiseKeys,
    NoisePos,
    NoiseSalt,
    NoiseSeed,
    NormCentered,
    Normed,
    NormEps,
    NormWeight,
    OutDim,
    OutGate,
    PartAcc,
    PartMax,
    PartSum,
    PickedBlocks,
    PickedCount,
    PleConvNorm,
    PleConvWeight,
    PleGated,
    PleKNorm,
    PleKv,
    PleKvStride,
    PleNormed,
    PleQuery,
    PleUpdate,
    PleWindow,
    PooledCap,
    PrefillRun,
    PrefixLen,
    ProjRows,
    ProjStride,
    QHeads,
    QHeadStride,
    Qkv,
    QkvColumn,
    QkvWidth,
    QNorm,
    Query,
    RadixCarry,
    RadixFloor,
    RadixHist,
    RadixLevel,
    RadixPrefix,
    Residual,
    RopeCos,
    RopeDims,
    RopeSin,
    RopeStride,
    RouteCount,
    Routed,
    RouterLogits,
    RouterStride,
    RouteSeq,
    RowMap,
    Rows,
    RowsLayout,
    SampleBest,
    SampledTokens,
    SampleMax,
    SampleMode,
    SampleParams,
    ScratchSlots,
    Seats,
    SharedGate,
    SharedOut,
    StepToken,
    StreamCount,
    Streams,
    TopBlocks,
    TopExperts,
    TopK,
    TopKFloor,
    TopPFloor,
    TreeParents,
    VCache,
    VColumn,
    VerifyPicks,
    Vocab,
    WarmShare,
    WarmSink,
    Weight,
    Widened,
    Window,
    ZColumn,
)

Dim3 = tuple[int, int, int]
_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int
_LL = ctypes.c_longlong
_R = TypeVar("_R")

# every kernel the card program loads, a macro's at each of its instantiations
KERNELS: tuple[str, ...] = (
    "btb_add_rmsnorm",
    "btb_attn_flash_d64",
    "btb_attn_flash_d128",
    "btb_attn_flash_d256",
    "btb_attn_flash_prefill_d64",
    "btb_attn_flash_prefill_d128",
    "btb_attn_flash_prefill_d256",
    "btb_attn_flash_prefill_kq_d64",
    "btb_attn_flash_prefill_kq_d128",
    "btb_attn_flash_prefill_kq_d256",
    "btb_attn_flash_rows_d64",
    "btb_attn_flash_rows_d128",
    "btb_attn_flash_rows_d256",
    "btb_conv_window",
    "btb_delta_nodes",
    "btb_delta_nodes_bf16",
    "btb_gelu_mul",
    "btb_gemm_f32_bf16",
    "btb_gemm_mma_bf16",
    "btb_gemm_mma_small_bf16",
    "btb_gemv_bf16_m1",
    "btb_gemv_bf16_m2",
    "btb_gemv_bf16_m4",
    "btb_gemv_bf16_m8",
    "btb_gemv_bf16_m16",
    "btb_gemv_bf16_m32",
    "btb_gemv_gelu_bf16_m1",
    "btb_gemv_gelu_bf16_m2",
    "btb_gemv_gelu_bf16_m4",
    "btb_gemv_gelu_bf16_m8",
    "btb_gemv_gelu_bf16_m16",
    "btb_gemv_gelu_bf16_m32",
    "btb_gemv_lane16_f32_m1",
    "btb_gemv_lane16_f32_m2",
    "btb_gemv_lane16_f32_m4",
    "btb_gemv_lane16_f32_m8",
    "btb_gemv_lane16_f32_m16",
    "btb_gemv_lane16_f32_m32",
    "btb_gemv_lane16_mx4_f32_m1",
    "btb_gemv_lane16_mx4_f32_m2",
    "btb_gemv_lane16_mx4_f32_m4",
    "btb_gemv_lane16_mx4_f32_m8",
    "btb_gemv_lane16_mx4_f32_m16",
    "btb_gemv_lane16_mx4_f32_m32",
    "btb_gemv_mma8_bf16",
    "btb_gemv_mma_bf16",
    "btb_gemv_mma_glu_gelu",
    "btb_gemv_mma_glu_silu",
    "btb_gemv_sgate_bf16_m1",
    "btb_gemv_sgate_bf16_m2",
    "btb_gemv_sgate_bf16_m4",
    "btb_gemv_sgate_bf16_m8",
    "btb_gemv_sgate_bf16_m16",
    "btb_gemv_sgate_bf16_m32",
    "btb_gemv_silu_bf16_m1",
    "btb_gemv_silu_bf16_m2",
    "btb_gemv_silu_bf16_m4",
    "btb_gemv_silu_bf16_m8",
    "btb_gemv_silu_bf16_m16",
    "btb_gemv_silu_bf16_m32",
    "btb_hc_act",
    "btb_hc_mix",
    "btb_hc_rmsnorm",
    "btb_l2_warm",
    "btb_mma_roles",
    "btb_moe_combine",
    "btb_moe_route",
    "btb_mx4_widen",
    "btb_norm_rope_kv_d64",
    "btb_norm_rope_kv_d128",
    "btb_norm_rope_kv_d256",
    "btb_norm_rope_kv_rows_d64",
    "btb_norm_rope_kv_rows_d128",
    "btb_norm_rope_kv_rows_d256",
    "btb_norm_rope_kv_tbl_d64",
    "btb_norm_rope_kv_tbl_d128",
    "btb_norm_rope_kv_tbl_d256",
    "btb_norm_rope_part_d128",
    "btb_norm_rope_part_d256",
    "btb_ple_conv",
    "btb_ple_gate",
    "btb_publish",
    "btb_qsa_attn_split_d128",
    "btb_qsa_attn_split_d256",
    "btb_qsa_pool_d128",
    "btb_qsa_pool_d256",
    "btb_qsa_select_d128",
    "btb_qsa_select_d256",
    "btb_sample_bin",
    "btb_sample_draw",
    "btb_sample_hist",
    "btb_sample_keys",
    "btb_sample_max",
    "btb_sample_out",
    "btb_sample_verify",
    "btb_sandwich_add",
    "btb_sigmoid_mul",
    "btb_silu_mul",
)


class Launcher(Protocol):
    """what launches a kernel: its name in the launcher's module, the grid and block, and one ctypes value a
    parameter (c_void_p a pointer, c_int, c_float, ..), on `stream` with `shared` bytes of dynamic shared memory"""

    def launch(
        self,
        name: str,
        grid: Dim3,
        block: Dim3,
        args: Sequence[Any],
        stream: torch.cuda.Stream | None = None,
        shared: int = 0,
    ) -> None: ...


def ptr(role: Callable[[int], _R], t: torch.Tensor | None) -> _R:
    """tensor `t`'s card address as an argument of `role` (`ptr(Query, q)`); None the null pointer. The address
    keeps nothing alive: hold `t` while a launch or a captured graph can read it"""
    return role(0 if t is None else t.data_ptr())


def btb_add_rmsnorm(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    h: Residual,
    y: BlockOut,
    w: NormWeight,
    eps: NormEps,
    x: Normed,
    H: Hidden,
    centered: NormCentered,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_add_rmsnorm", grid, block, [_P(h), _P(y), _P(w), _F(eps), _P(x), _I(H), _I(centered)], stream, shared
    )


def btb_attn_flash(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    q: Query,
    K: KCache,
    V: VCache,
    out: AttnOut,
    n0p: PrefixLen,
    par: TreeParents,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    scale: AttnScale,
    part_m: PartMax,
    part_l: PartSum,
    part_acc: PartAcc,
    cnt: Arrivals,
    win: Window,
    tbl: RowMap,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_attn_flash_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_attn_flash_d{D}",
        grid,
        block,
        [
            _P(q),
            _P(K),
            _P(V),
            _P(out),
            _P(n0p),
            _P(par),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _F(scale),
            _P(part_m),
            _P(part_l),
            _P(part_acc),
            _P(cnt),
            _I(win),
            _P(tbl),
        ],
        stream,
        shared,
    )


def btb_attn_flash_prefill(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    q: Query,
    K: KCache,
    V: VCache,
    out: AttnOut,
    n0: ChunkStart,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    scale: AttnScale,
    win: Window,
    tbl: RowMap,
    run: PrefillRun,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_attn_flash_prefill_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_attn_flash_prefill_d{D}",
        grid,
        block,
        [
            _P(q),
            _P(K),
            _P(V),
            _P(out),
            _I(n0),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _F(scale),
            _I(win),
            _P(tbl),
            _P(run),
        ],
        stream,
        shared,
    )


def btb_attn_flash_prefill_kq(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    q: Query,
    K: KCache,
    V: VCache,
    out: AttnOut,
    n0: ChunkStart,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    scale: AttnScale,
    win: Window,
    tbl: RowMap,
    run: PrefillRun,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_attn_flash_prefill_kq_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_attn_flash_prefill_kq_d{D}",
        grid,
        block,
        [
            _P(q),
            _P(K),
            _P(V),
            _P(out),
            _I(n0),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _F(scale),
            _I(win),
            _P(tbl),
            _P(run),
        ],
        stream,
        shared,
    )


def btb_attn_flash_rows(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    q: Query,
    K: KCache,
    V: VCache,
    out: AttnOut,
    rw: RowsLayout,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    scale: AttnScale,
    part_m: PartMax,
    part_l: PartSum,
    part_acc: PartAcc,
    cnt: Arrivals,
    win: Window,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_attn_flash_rows_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_attn_flash_rows_d{D}",
        grid,
        block,
        [
            _P(q),
            _P(K),
            _P(V),
            _P(out),
            _P(rw),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _F(scale),
            _P(part_m),
            _P(part_l),
            _P(part_acc),
            _P(cnt),
            _I(win),
        ],
        stream,
        shared,
    )


def btb_conv_window(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    conv: ConvState,
    proj: DeltaProj,
    ps: ProjStride,
    oq: QkvColumn,
    rows: ProjRows,
    n: Rows,
    C: ConvChannels,
    K: ConvTaps,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_conv_window",
        grid,
        block,
        [_P(conv), _P(proj), _I(ps), _I(oq), _P(rows), _I(n), _I(C), _I(K)],
        stream,
        shared,
    )


def btb_delta_nodes(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    mixed: DeltaMixed,
    z: DeltaZ,
    a: DeltaA,
    b: DeltaB,
    conv_w: ConvWeight,
    conv_b: ConvBias,
    conv0: ConvState,
    state: DeltaState,
    scratch: DeltaScratch,
    parents: TreeParents,
    a_log: DeltaALog,
    dt_bias: DeltaDtBias,
    norm_w: DeltaNormWeight,
    eps: NormEps,
    gate: DeltaGate,
    T: Rows,
    C: ConvChannels,
    K: ConvTaps,
    Hk: DeltaKHeads,
    Hv: DeltaVHeads,
    dk: DeltaKDim,
    dv: DeltaVDim,
    out: DeltaOutF32,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_delta_nodes",
        grid,
        block,
        [
            _P(mixed),
            _P(z),
            _P(a),
            _P(b),
            _P(conv_w),
            _P(conv_b),
            _P(conv0),
            _P(state),
            _P(scratch),
            _P(parents),
            _P(a_log),
            _P(dt_bias),
            _P(norm_w),
            _F(eps),
            _I(gate),
            _I(T),
            _I(C),
            _I(K),
            _I(Hk),
            _I(Hv),
            _I(dk),
            _I(dv),
            _P(out),
        ],
        stream,
        shared,
    )


def btb_delta_nodes_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    proj: DeltaProj,
    ps: ProjStride,
    oq: QkvColumn,
    oz: ZColumn,
    ob: BColumn,
    oa: AColumn,
    rows: ProjRows,
    conv_w: ConvWeight,
    conv_b: ConvBias,
    conv0: ConvState,
    state: DeltaState,
    scratch: DeltaScratch,
    slots: ScratchSlots,
    parents: TreeParents,
    a_log: DeltaALog,
    dt_bias: DeltaDtBias,
    norm_w: DeltaNormWeight,
    eps: NormEps,
    gate: DeltaGate,
    T: Rows,
    K: ConvTaps,
    Hk: DeltaKHeads,
    Hv: DeltaVHeads,
    dk: DeltaKDim,
    dv: DeltaVDim,
    out: DeltaOut,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_delta_nodes_bf16",
        grid,
        block,
        [
            _P(proj),
            _I(ps),
            _I(oq),
            _I(oz),
            _I(ob),
            _I(oa),
            _P(rows),
            _P(conv_w),
            _P(conv_b),
            _P(conv0),
            _P(state),
            _P(scratch),
            _P(slots),
            _P(parents),
            _P(a_log),
            _P(dt_bias),
            _P(norm_w),
            _F(eps),
            _I(gate),
            _I(T),
            _I(K),
            _I(Hk),
            _I(Hv),
            _I(dk),
            _I(dv),
            _P(out),
        ],
        stream,
        shared,
    )


def btb_gelu_mul(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    gu: GateUp,
    m: MlpAct,
    T: Rows,
    I: Intermediate,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gelu_mul", grid, block, [_P(gu), _P(m), _I(T), _I(I)], stream, shared)


def btb_gemm_f32_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    T: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemm_f32_bf16", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C), _I(T)], stream, shared)


def btb_gemm_mma_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    T: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemm_mma_bf16", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C), _I(T)], stream, shared)


def btb_gemm_mma_small_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    T: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemm_mma_small_bf16", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C), _I(T)], stream, shared)


def btb_gemv_bf16(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_bf16_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(f"btb_gemv_bf16_m{M}", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C)], stream, shared)


def btb_gemv_gelu_bf16(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    gu: GateUp,
    y: MatOut,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_gelu_bf16_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(f"btb_gemv_gelu_bf16_m{M}", grid, block, [_P(w), _P(gu), _P(y), _I(R), _I(C)], stream, shared)


def btb_gemv_lane16_f32(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatInF32,
    y: MatOutF32,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_lane16_f32_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(f"btb_gemv_lane16_f32_m{M}", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C)], stream, shared)


def btb_gemv_lane16_mx4_f32(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    blocks: Mx4Blocks,
    scales: Mx4Scales,
    x: MatInF32,
    y: MatOutF32,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_lane16_mx4_f32_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(
        f"btb_gemv_lane16_mx4_f32_m{M}",
        grid,
        block,
        [_P(blocks), _P(scales), _P(x), _P(y), _I(R), _I(C)],
        stream,
        shared,
    )


def btb_gemv_mma8_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    M: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemv_mma8_bf16", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C), _I(M)], stream, shared)


def btb_gemv_mma_bf16(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    y: MatOut,
    R: OutDim,
    C: InDim,
    M: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemv_mma_bf16", grid, block, [_P(w), _P(x), _P(y), _I(R), _I(C), _I(M)], stream, shared)


def btb_gemv_mma_glu_gelu(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    m: MlpAct,
    I: Intermediate,
    C: InDim,
    M: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemv_mma_glu_gelu", grid, block, [_P(w), _P(x), _P(m), _I(I), _I(C), _I(M)], stream, shared)


def btb_gemv_mma_glu_silu(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    x: MatIn,
    m: MlpAct,
    I: Intermediate,
    C: InDim,
    M: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_gemv_mma_glu_silu", grid, block, [_P(w), _P(x), _P(m), _I(I), _I(C), _I(M)], stream, shared)


def btb_gemv_sgate_bf16(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    att: AttnOut,
    gate: OutGate,
    gs: GateRowStride,
    D: HeadDim,
    hs: GateHeadStride,
    y: MatOut,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_sgate_bf16_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(
        f"btb_gemv_sgate_bf16_m{M}",
        grid,
        block,
        [_P(w), _P(att), _P(gate), _I(gs), _I(D), _I(hs), _P(y), _I(R), _I(C)],
        stream,
        shared,
    )


def btb_gemv_silu_bf16(
    launcher: Launcher,
    M: int,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    gu: GateUp,
    y: MatOut,
    R: OutDim,
    C: InDim,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_gemv_silu_bf16_m{M}, M one of 1, 2, 4, 8, 16, 32"""
    launcher.launch(f"btb_gemv_silu_bf16_m{M}", grid, block, [_P(w), _P(gu), _P(y), _I(R), _I(C)], stream, shared)


def btb_hc_act(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    dn: MixDown,
    ds: MixDownStride,
    act: MixAct,
    R: MixRank,
    G: StreamCount,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_hc_act", grid, block, [_P(dn), _I(ds), _P(act), _I(R), _I(G)], stream, shared)


def btb_hc_mix(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    xn: Normed,
    up: MixUp,
    dn: MixDown,
    ds: MixDownStride,
    mixed: Mixed,
    inj: Inject,
    H: Hidden,
    G: StreamCount,
    R: MixRank,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_hc_mix",
        grid,
        block,
        [_P(xn), _P(up), _P(dn), _I(ds), _P(mixed), _P(inj), _I(H), _I(G), _I(R)],
        stream,
        shared,
    )


def btb_hc_rmsnorm(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    h: Streams,
    y: BlockOut,
    inj: Inject,
    w: NormWeight,
    eps: NormEps,
    x: Normed,
    H: Hidden,
    G: StreamCount,
    centered: NormCentered,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_hc_rmsnorm",
        grid,
        block,
        [_P(h), _P(y), _P(inj), _P(w), _F(eps), _P(x), _I(H), _I(G), _I(centered)],
        stream,
        shared,
    )


def btb_l2_warm(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    w: Weight,
    R: OutDim,
    C: InDim,
    sharep: WarmShare,
    sink: WarmSink,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_l2_warm", grid, block, [_P(w), _I(R), _I(C), _P(sharep), _P(sink)], stream, shared)


def btb_mma_roles(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    bad: MmaMisses,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_mma_roles", grid, block, [_P(bad)], stream, shared)


def btb_moe_combine(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    yr: Routed,
    ys: SharedOut,
    sg: SharedGate,
    sgs: RouterStride,
    y: MoeOut,
    H: Hidden,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_moe_combine", grid, block, [_P(yr), _P(ys), _P(sg), _I(sgs), _P(y), _I(H)], stream, shared)


def btb_moe_route(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    logits: RouterLogits,
    ls: RouterStride,
    E: Experts,
    k: TopExperts,
    idx: ExpertIds,
    w: ExpertWeights,
    hidx: HostExpertIds,
    hw: HostExpertWeights,
    cnt: RouteCount,
    seq: RouteSeq,
    hseq: HostRouteSeq,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_moe_route",
        grid,
        block,
        [_P(logits), _I(ls), _I(E), _I(k), _P(idx), _P(w), _P(hidx), _P(hw), _P(cnt), _P(seq), _P(hseq)],
        stream,
        shared,
    )


def btb_mx4_widen(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    blocks: Mx4Blocks,
    scales: Mx4Scales,
    seats: Seats,
    groups: Mx4Groups,
    out: Widened,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_mx4_widen", grid, block, [_P(blocks), _P(scales), _P(seats), _LL(groups), _P(out)], stream, shared
    )


def btb_norm_rope_kv(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    qkv: Qkv,
    wq: QNorm,
    wk: KNorm,
    eps: NormEps,
    cos_t: RopeCos,
    sin_t: RopeSin,
    n0p: PrefixLen,
    depth: Depths,
    K: KCache,
    V: VCache,
    qo: Query,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    centered: NormCentered,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_norm_rope_kv_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_norm_rope_kv_d{D}",
        grid,
        block,
        [
            _P(qkv),
            _P(wq),
            _P(wk),
            _F(eps),
            _P(cos_t),
            _P(sin_t),
            _P(n0p),
            _P(depth),
            _P(K),
            _P(V),
            _P(qo),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _I(centered),
        ],
        stream,
        shared,
    )


def btb_norm_rope_kv_rows(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    qkv: Qkv,
    wq: QNorm,
    wk: KNorm,
    eps: NormEps,
    cos_t: RopeCos,
    sin_t: RopeSin,
    rw: RowsLayout,
    K: KCache,
    V: VCache,
    qo: Query,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    centered: NormCentered,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_norm_rope_kv_rows_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_norm_rope_kv_rows_d{D}",
        grid,
        block,
        [
            _P(qkv),
            _P(wq),
            _P(wk),
            _F(eps),
            _P(cos_t),
            _P(sin_t),
            _P(rw),
            _P(K),
            _P(V),
            _P(qo),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _I(centered),
        ],
        stream,
        shared,
    )


def btb_norm_rope_kv_tbl(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    qkv: Qkv,
    wq: QNorm,
    wk: KNorm,
    eps: NormEps,
    cos_t: RopeCos,
    sin_t: RopeSin,
    n0p: PrefixLen,
    depth: Depths,
    K: KCache,
    V: VCache,
    qo: Query,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    centered: NormCentered,
    tbl: RowMap,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_norm_rope_kv_tbl_d{D}, D one of 64, 128, 256"""
    launcher.launch(
        f"btb_norm_rope_kv_tbl_d{D}",
        grid,
        block,
        [
            _P(qkv),
            _P(wq),
            _P(wk),
            _F(eps),
            _P(cos_t),
            _P(sin_t),
            _P(n0p),
            _P(depth),
            _P(K),
            _P(V),
            _P(qo),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _I(centered),
            _P(tbl),
        ],
        stream,
        shared,
    )


def btb_norm_rope_part(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    qkv: Qkv,
    width: QkvWidth,
    qhs: QHeadStride,
    koff: KColumn,
    voff: VColumn,
    wq: QNorm,
    wk: KNorm,
    eps: NormEps,
    cos_t: RopeCos,
    sin_t: RopeSin,
    ts: RopeStride,
    rot: RopeDims,
    n0p: PrefixLen,
    depth: Depths,
    K: KCache,
    V: VCache,
    qo: Query,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    centered: NormCentered,
    rawk: KeepRawKey,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_norm_rope_part_d{D}, D one of 128, 256"""
    launcher.launch(
        f"btb_norm_rope_part_d{D}",
        grid,
        block,
        [
            _P(qkv),
            _I(width),
            _I(qhs),
            _I(koff),
            _I(voff),
            _P(wq),
            _P(wk),
            _F(eps),
            _P(cos_t),
            _P(sin_t),
            _I(ts),
            _I(rot),
            _P(n0p),
            _P(depth),
            _P(K),
            _P(V),
            _P(qo),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _I(centered),
            _I(rawk),
        ],
        stream,
        shared,
    )


def btb_ple_conv(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    gated: PleGated,
    normed: PleNormed,
    w: PleConvWeight,
    pre: PleWindow,
    par: TreeParents,
    h: Streams,
    C: ConvChannels,
    K: ConvTaps,
    dil: ConvDilation,
    update: PleUpdate,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_ple_conv",
        grid,
        block,
        [_P(gated), _P(normed), _P(w), _P(pre), _P(par), _P(h), _I(C), _I(K), _I(dil), _I(update)],
        stream,
        shared,
    )


def btb_ple_gate(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    kv: PleKv,
    ks: PleKvStride,
    xq: PleQuery,
    wk: PleKNorm,
    wc: PleConvNorm,
    eps: NormEps,
    inv_h: InvSqrtHidden,
    par: TreeParents,
    gated: PleGated,
    normed: PleNormed,
    H: Hidden,
    G: StreamCount,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_ple_gate",
        grid,
        block,
        [_P(kv), _I(ks), _P(xq), _P(wk), _P(wc), _F(eps), _F(inv_h), _P(par), _P(gated), _P(normed), _I(H), _I(G)],
        stream,
        shared,
    )


def btb_publish(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    n0: PrefixLen,
    ids: StepToken,
    out_tok: HostToken,
    out_n0: HostLen,
    out_ts: HostClock,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_publish", grid, block, [_P(n0), _P(ids), _P(out_tok), _P(out_n0), _P(out_ts)], stream, shared)


def btb_qsa_attn_split(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    q: Query,
    K: KCache,
    V: VCache,
    out: AttnOut,
    n0p: PrefixLen,
    par: TreeParents,
    sel: PickedBlocks,
    nsel: PickedCount,
    r: BlockRatio,
    ktop: TopBlocks,
    T: Rows,
    Hq: QHeads,
    Hk: KvHeads,
    hs: CacheHeadStride,
    rs: CacheRowStride,
    scale: AttnScale,
    part_m: PartMax,
    part_l: PartSum,
    part_acc: PartAcc,
    cnt: Arrivals,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_qsa_attn_split_d{D}, D one of 128, 256"""
    launcher.launch(
        f"btb_qsa_attn_split_d{D}",
        grid,
        block,
        [
            _P(q),
            _P(K),
            _P(V),
            _P(out),
            _P(n0p),
            _P(par),
            _P(sel),
            _P(nsel),
            _I(r),
            _I(ktop),
            _I(T),
            _I(Hq),
            _I(Hk),
            _I(hs),
            _I(rs),
            _F(scale),
            _P(part_m),
            _P(part_l),
            _P(part_acc),
            _P(cnt),
        ],
        stream,
        shared,
    )


def btb_qsa_pool(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    raw: IdxRawKey,
    pk: IdxPooled,
    pk_len: IdxPooledLen,
    n0p: PrefixLen,
    cos_t: RopeCos,
    sin_t: RopeSin,
    ts: RopeStride,
    rot: RopeDims,
    kw: IdxKNorm,
    eps: NormEps,
    r: BlockRatio,
    max_new: MaxNewBlocks,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_qsa_pool_d{D}, D one of 128, 256"""
    launcher.launch(
        f"btb_qsa_pool_d{D}",
        grid,
        block,
        [
            _P(raw),
            _P(pk),
            _P(pk_len),
            _P(n0p),
            _P(cos_t),
            _P(sin_t),
            _I(ts),
            _I(rot),
            _P(kw),
            _F(eps),
            _I(r),
            _I(max_new),
        ],
        stream,
        shared,
    )


def btb_qsa_select(
    launcher: Launcher,
    D: int,
    grid: Dim3,
    block: Dim3,
    qi: IdxQuery,
    pk: IdxPooled,
    raw: IdxRawKey,
    n0p: PrefixLen,
    par: TreeParents,
    cos_t: RopeCos,
    sin_t: RopeSin,
    ts: RopeStride,
    rot: RopeDims,
    kw: IdxKNorm,
    eps: NormEps,
    T: Rows,
    Hi: IdxHeads,
    r: BlockRatio,
    ktop: TopBlocks,
    NBcap: PooledCap,
    scores: IdxScores,
    sel: PickedBlocks,
    nsel: PickedCount,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    """btb_qsa_select_d{D}, D one of 128, 256"""
    launcher.launch(
        f"btb_qsa_select_d{D}",
        grid,
        block,
        [
            _P(qi),
            _P(pk),
            _P(raw),
            _P(n0p),
            _P(par),
            _P(cos_t),
            _P(sin_t),
            _I(ts),
            _I(rot),
            _P(kw),
            _F(eps),
            _I(T),
            _I(Hi),
            _I(r),
            _I(ktop),
            _I(NBcap),
            _P(scores),
            _P(sel),
            _P(nsel),
        ],
        stream,
        shared,
    )


def btb_sample_bin(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    level: RadixLevel,
    mode: SampleMode,
    top_k: TopK,
    fp: SampleParams,
    ghist: RadixHist,
    gpre: RadixPrefix,
    gcarry: RadixCarry,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sample_bin",
        grid,
        block,
        [_I(level), _I(mode), _I(top_k), _P(fp), _P(ghist), _P(gpre), _P(gcarry)],
        stream,
        shared,
    )


def btb_sample_draw(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    x: Logits,
    keys: NoiseKeys,
    V: Vocab,
    fp: SampleParams,
    gM: SampleMax,
    gpreK: TopKFloor,
    gpreP: TopPFloor,
    gbest: SampleBest,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sample_draw",
        grid,
        block,
        [_P(x), _P(keys), _I(V), _P(fp), _P(gM), _P(gpreK), _P(gpreP), _P(gbest)],
        stream,
        shared,
    )


def btb_sample_hist(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    x: Logits,
    V: Vocab,
    fp: SampleParams,
    level: RadixLevel,
    mode: SampleMode,
    gM: SampleMax,
    gpre: RadixPrefix,
    gfloor: RadixFloor,
    ghist: RadixHist,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sample_hist",
        grid,
        block,
        [_P(x), _I(V), _P(fp), _I(level), _I(mode), _P(gM), _P(gpre), _P(gfloor), _P(ghist)],
        stream,
        shared,
    )


def btb_sample_keys(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    seed: NoiseSeed,
    pos: NoisePos,
    salt: NoiseSalt,
    keys: NoiseKeys,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_sample_keys", grid, block, [_P(seed), _P(pos), _I(salt), _P(keys)], stream, shared)


def btb_sample_max(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    x: Logits,
    V: Vocab,
    fp: SampleParams,
    gM: SampleMax,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_sample_max", grid, block, [_P(x), _I(V), _P(fp), _P(gM)], stream, shared)


def btb_sample_out(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    gbest: SampleBest,
    out: SampledTokens,
    R: Rows,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_sample_out", grid, block, [_P(gbest), _P(out), _I(R)], stream, shared)


def btb_sample_verify(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    x: Logits,
    q: DraftProbs,
    kids: DraftKids,
    hasq: HasDraftProbs,
    keys: NoiseKeys,
    V: Vocab,
    Vd: DraftVocab,
    C: DraftWidth,
    top_k: TopK,
    fp: SampleParams,
    out: VerifyPicks,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sample_verify",
        grid,
        block,
        [_P(x), _P(q), _P(kids), _P(hasq), _P(keys), _I(V), _I(Vd), _I(C), _I(top_k), _P(fp), _P(out)],
        stream,
        shared,
    )


def btb_sandwich_add(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    h: Residual,
    y: BlockOut,
    w: NormWeight,
    eps: NormEps,
    H: Hidden,
    centered: NormCentered,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sandwich_add", grid, block, [_P(h), _P(y), _P(w), _F(eps), _I(H), _I(centered)], stream, shared
    )


def btb_sigmoid_mul(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    att: AttnOut,
    gate: OutGate,
    gs: GateRowStride,
    D: HeadDim,
    hs: GateHeadStride,
    out: GatedAttn,
    C: AttnWidth,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch(
        "btb_sigmoid_mul", grid, block, [_P(att), _P(gate), _I(gs), _I(D), _I(hs), _P(out), _I(C)], stream, shared
    )


def btb_silu_mul(
    launcher: Launcher,
    grid: Dim3,
    block: Dim3,
    gu: GateUp,
    m: MlpAct,
    T: Rows,
    I: Intermediate,
    *,
    stream: torch.cuda.Stream | None = None,
    shared: int = 0,
) -> None:
    launcher.launch("btb_silu_mul", grid, block, [_P(gu), _P(m), _I(T), _I(I)], stream, shared)


class CardLaunch:
    """the launchers as methods of the card's kernel module (`_Cuda`), each the function of its name with the
    module as its launcher"""

    def launch(
        self,
        name: str,
        grid: Dim3,
        block: Dim3,
        args: Sequence[Any],
        stream: torch.cuda.Stream | None = None,
        shared: int = 0,
    ) -> None:
        raise NotImplementedError

    ptr = staticmethod(ptr)
    btb_add_rmsnorm = btb_add_rmsnorm
    btb_attn_flash = btb_attn_flash
    btb_attn_flash_prefill = btb_attn_flash_prefill
    btb_attn_flash_prefill_kq = btb_attn_flash_prefill_kq
    btb_attn_flash_rows = btb_attn_flash_rows
    btb_conv_window = btb_conv_window
    btb_delta_nodes = btb_delta_nodes
    btb_delta_nodes_bf16 = btb_delta_nodes_bf16
    btb_gelu_mul = btb_gelu_mul
    btb_gemm_f32_bf16 = btb_gemm_f32_bf16
    btb_gemm_mma_bf16 = btb_gemm_mma_bf16
    btb_gemm_mma_small_bf16 = btb_gemm_mma_small_bf16
    btb_gemv_bf16 = btb_gemv_bf16
    btb_gemv_gelu_bf16 = btb_gemv_gelu_bf16
    btb_gemv_lane16_f32 = btb_gemv_lane16_f32
    btb_gemv_lane16_mx4_f32 = btb_gemv_lane16_mx4_f32
    btb_gemv_mma8_bf16 = btb_gemv_mma8_bf16
    btb_gemv_mma_bf16 = btb_gemv_mma_bf16
    btb_gemv_mma_glu_gelu = btb_gemv_mma_glu_gelu
    btb_gemv_mma_glu_silu = btb_gemv_mma_glu_silu
    btb_gemv_sgate_bf16 = btb_gemv_sgate_bf16
    btb_gemv_silu_bf16 = btb_gemv_silu_bf16
    btb_hc_act = btb_hc_act
    btb_hc_mix = btb_hc_mix
    btb_hc_rmsnorm = btb_hc_rmsnorm
    btb_l2_warm = btb_l2_warm
    btb_mma_roles = btb_mma_roles
    btb_moe_combine = btb_moe_combine
    btb_moe_route = btb_moe_route
    btb_mx4_widen = btb_mx4_widen
    btb_norm_rope_kv = btb_norm_rope_kv
    btb_norm_rope_kv_rows = btb_norm_rope_kv_rows
    btb_norm_rope_kv_tbl = btb_norm_rope_kv_tbl
    btb_norm_rope_part = btb_norm_rope_part
    btb_ple_conv = btb_ple_conv
    btb_ple_gate = btb_ple_gate
    btb_publish = btb_publish
    btb_qsa_attn_split = btb_qsa_attn_split
    btb_qsa_pool = btb_qsa_pool
    btb_qsa_select = btb_qsa_select
    btb_sample_bin = btb_sample_bin
    btb_sample_draw = btb_sample_draw
    btb_sample_hist = btb_sample_hist
    btb_sample_keys = btb_sample_keys
    btb_sample_max = btb_sample_max
    btb_sample_out = btb_sample_out
    btb_sample_verify = btb_sample_verify
    btb_sandwich_add = btb_sandwich_add
    btb_sigmoid_mul = btb_sigmoid_mul
    btb_silu_mul = btb_silu_mul
