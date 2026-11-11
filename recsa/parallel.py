# -*- coding: utf-8 -*-
"""RecSA Triton Kernels"""

import logging
import math
import os
from typing import Optional

import torch
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

KERNEL_TRACE_ENABLED = os.environ.get("RECSA_KERNEL_TRACE", "0").lower() in {"1", "true", "yes"}

MID_NS_DEFAULT = os.environ.get("RECSA_MID_NS", "1").lower() in {"1", "true", "yes"}

MID_NS_WIDE = os.environ.get("RECSA_MID_NS_WIDE", "0").lower() in {"1", "true", "yes"}

MID_DEFER = os.environ.get("RECSA_MID_DEFER", "1").lower() in {"1", "true", "yes"}
MID_SEL_MAXNREG = int(os.environ.get("RECSA_MID_SEL_MAXNREG", "168"))

_prefill_fast_env = os.environ.get("RECSA_PREFILL_FAST", "").strip().lower()
if _prefill_fast_env not in {"", "1", "true", "yes", "on", "0", "false", "no", "off"}:
    raise ValueError(f"RECSA_PREFILL_FAST={os.environ['RECSA_PREFILL_FAST']!r}: expected unset / 1 / true / yes / on "
                     "(fast eval prefill, the default) or 0 / false / no / off (deployed prefill)")
_PREFILL_FAST_DEFAULT = "0" if _prefill_fast_env in {"0", "false", "no", "off"} else "1"

_acc_env = os.environ.get("RECSA_ACC_FP16", _PREFILL_FAST_DEFAULT).strip().lower()
ACC_DOT_MODE = 2 if _acc_env == "ieee" else (1 if _acc_env in {"1", "true", "yes"} else 0)
ACC_FP16 = ACC_DOT_MODE != 0

MERGE_ACC = os.environ.get("RECSA_MERGE_ACC", _PREFILL_FAST_DEFAULT).lower() in {"1", "true", "yes"}
UNION_LEAF = MERGE_ACC
UNION_MID = MERGE_ACC
MERGE_MAX_NODES = 131072

FAST_SELECT = os.environ.get("RECSA_FAST_SELECT", _PREFILL_FAST_DEFAULT).lower() in {"1", "true", "yes"}
FAST_TOP_SELECT = None
SEL_IEEE = False

PREFILL_DISPATCH = ("ACC_DOT_MODE", "ACC_FP16", "MERGE_ACC", "MERGE_MAX_NODES", "UNION_LEAF", "UNION_MID",
                    "FAST_SELECT", "FAST_TOP_SELECT", "TOPACC_CUDA", "SEL_IEEE", "MID_NS_DEFAULT", "MID_NS_WIDE",
                    "MID_DEFER", "MID_SEL_MAXNREG")


def recsa_env() -> dict:
    """The RECSA_* environment variables that are set in this process (eval provenance)."""
    return {k: os.environ[k] for k in sorted(os.environ) if k.startswith("RECSA_")}


def prefill_config() -> dict:
    """Resolved prefill configuration of this process, recorded in eval summaries."""
    from recsa import fast_select as FS
    from recsa import incremental_tree as IT
    from recsa import leaf_cuda as LC
    from recsa import merge_acc as MA
    from recsa import pool_fused as PF
    from recsa import topsel_cuda as TS
    g = globals()
    fts = FAST_SELECT if FAST_TOP_SELECT is None else FAST_TOP_SELECT
    gates = (MERGE_ACC, UNION_LEAF, UNION_MID, FAST_SELECT, fts)
    if ACC_DOT_MODE == 1 and all(gates) and not SEL_IEEE:
        operator = "fast"
    elif ACC_DOT_MODE == 0 and not any(gates) and not SEL_IEEE:
        operator = "deployed"
    else:
        operator = "mixed"
    failed = {f["family"] for f in LC.CUDA_FALLBACKS}
    leaf_broken = bool(MA._LEAF_CUDA_BROKEN)
    families = {
        "leaf_union": ("recsa/cuda/leaf_union.cu", "merge_acc.LEAF_CUDA", bool(MA.LEAF_CUDA), bool(LC._fn), True),
        "union_build": ("recsa/cuda/union_build.cu", "merge_acc.LEAF_CUDA", bool(MA.LEAF_CUDA), bool(LC._ufn), True),
        "top_accum": ("recsa/cuda/top_accum.cu", "parallel.TOPACC_CUDA", bool(TOPACC_CUDA), bool(LC._afn), True),
        "top_finalize": ("recsa/cuda/top_finalize.cu", "fast_select.FAST_TOP_B_CUDA", bool(FS.FAST_TOP_B_CUDA),
                         bool(TS._mods), False),
        "top_score": ("recsa/cuda/top_score.cu", "fast_select.FAST_TOPA_CUDA", bool(FS.FAST_TOPA_CUDA), bool(TS._amods),
                      False),
        "mid_score": ("recsa/cuda/mid_score.cu", "fast_select.FAST_MIDA_CUDA", bool(FS.FAST_MIDA_CUDA), bool(TS._mmods),
                      False),
    }
    kernels = {}
    for fam, (source, switch, enabled, loaded, shared) in families.items():
        if fam in failed or (shared and leaf_broken):
            status = "fell_back"
        elif loaded:
            status = "active"
        elif not enabled:
            status = "disabled"
        else:
            status = "not_loaded"
        kernels[fam] = dict(status=status, source=source, kill_switch=switch, enabled=enabled, loaded=loaded)
    return dict(
        operator=operator,
        RECSA_PREFILL_FAST=os.environ.get("RECSA_PREFILL_FAST"),          # raw value, None = unset (= fast)
        prefill_fast_default=_PREFILL_FAST_DEFAULT == "1",
        fast_top_select_effective=bool(fts),
        dispatch={n: g[n] for n in PREFILL_DISPATCH},
        merge_acc=dict(LEAF_CUDA=int(MA.LEAF_CUDA), _LEAF_CUDA_BROKEN=leaf_broken),
        fast_select=dict(FAST_TOP_B_CUDA=int(FS.FAST_TOP_B_CUDA), FAST_MIDA_CUDA=int(FS.FAST_MIDA_CUDA),
                         FAST_TOPA_CUDA=int(FS.FAST_TOPA_CUDA), FAST_TOP_MAX_NODES=int(FS.FAST_TOP_MAX_NODES)),
        leaf_cuda=dict(NVCC=LC.NVCC),
        pool=dict(POOL_FUSED=int(PF.POOL_FUSED), FUSED_POOL_DEFAULT=bool(IT.FUSED_POOL_DEFAULT)),
        cuda_strict=LC.cuda_strict(),
        cuda_kernels=kernels,
        cuda_fallbacks=[dict(f) for f in LC.CUDA_FALLBACKS],
    )

_ROPE_CACHES: dict = {}

NVTX_ENABLED = os.environ.get("RECSA_NVTX", "0").lower() in {"1", "true", "yes"}
if not NVTX_ENABLED:
    class _NvtxNoop:
        @staticmethod
        def range_push(msg):
            return None

        @staticmethod
        def range_pop():
            return None
    nvtx = _NvtxNoop()

_DECODE_WORKSPACES: dict = {}


def _decode_workspace(device, B, H, H_kv, V, K, top_k, staged_leaf=None):
    key = (device.index, B, H, H_kv, V, K, top_k,
           torch.is_inference_mode_enabled())
    if staged_leaf is not None:
        key = key + (staged_leaf,)
    ws = _DECODE_WORKSPACES.get(key)
    if ws is None:
        f32 = dict(dtype=torch.float32, device=device)
        ws = _DECODE_WORKSPACES[key] = dict(
            global_max=torch.empty(B, 1, H, **f32),
            global_sum=torch.empty(B, 1, H, **f32),
            global_output=torch.empty(B, 1, H, V, **f32),
            global_max_b=torch.empty(B, 1, H, **f32),
            global_sum_b=torch.empty(B, 1, H, **f32),
            global_output_b=torch.empty(B, 1, H, V, **f32),
            layer_max=torch.empty(B, 1, H, **f32),
            layer_sum=torch.empty(B, 1, H, **f32),
            layer_output=torch.empty(B, 1, H, V, **f32),
            iota=torch.arange(top_k, dtype=torch.int32, device=device)
                 .view(1, 1, 1, top_k).expand(B, 1, H_kv, top_k).contiguous(),
            prev=torch.empty(B, 1, H_kv, top_k, dtype=torch.int32, device=device),
            next=torch.empty(B, 1, H_kv, top_k, dtype=torch.int32, device=device),
            mid_scratch=None, mid_imp=None, mid_tau=None,
            bitmask=None, sel_scratch=None,
        )
        if staged_leaf is not None:
            leaf_dtype, max_cand = staged_leaf
            ws["fetch_k"] = torch.zeros(H_kv, max_cand, K,
                                        dtype=leaf_dtype, device=device)
            ws["fetch_v"] = torch.zeros(H_kv, max_cand, V,
                                        dtype=leaf_dtype, device=device)
    return ws

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO if KERNEL_TRACE_ENABLED else logging.WARNING)

_trace = logger.info if KERNEL_TRACE_ENABLED else (lambda *a, **k: None)


class _cuda_timer:
    """Times enclosed CUDA ops and logs the result when tracing is enabled."""
    __slots__ = ("label", "args", "t0", "t1")

    def __init__(self, label: str, *args):
        self.label = label
        self.args = args

    def __enter__(self):
        if not KERNEL_TRACE_ENABLED:
            return
        torch.cuda.synchronize()
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.t0.record()

    def __exit__(self, *exc):
        if not KERNEL_TRACE_ENABLED:
            return False
        self.t1.record()
        torch.cuda.synchronize()
        _trace("%s: %.2f ms",
                    self.label % self.args if self.args else self.label,
                    self.t0.elapsed_time(self.t1))
        return False


RECSA_SCORE_NEG_INF: float = -1.0e10
TOPACC_CUDA = True

RECSA_SCORE_VALID_THRESHOLD: float = RECSA_SCORE_NEG_INF * 0.9

SUMMARIES_MODES = ("softmax", "none")


def summaries_off(summaries) -> bool:
    """Validate a `summaries` setting; True iff it is "none"."""
    s = "softmax" if summaries is None else summaries
    if s not in SUMMARIES_MODES:
        raise ValueError(f"summaries must be one of {SUMMARIES_MODES}, got {summaries!r}")
    return s == "none"


@triton.jit
def recsa_build_kernel(
    child_k,  # [B, N_child, H_kv, K]
    child_v,  # [B, N_child, H_kv, V]
    parent_k,  # [B, N_parent, H_kv, K]
    parent_v,  # [B, N_parent, H_kv, V]
    N_child,
    N_parent,
    B,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    t_ptr=None,
    CHILD_POWER: tl.constexpr = 0,  # c^(level-1): child level's tokens/node
    HAS_CURSOR: tl.constexpr = 0,
):
    """Tree building kernel: mean pooling from child nodes to parent nodes. Grid: (N_parent_blocks, B * H_kv)"""
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)

    i_b = pid_bh // H_kv
    i_h = pid_bh % H_kv

    if HAS_CURSOR:
        t_new = (tl.load(t_ptr) + 1).to(tl.int32)
        n_child_lvl = (t_new + CHILD_POWER - 1) // CHILD_POWER
        p_open = (n_child_lvl - 1) // COMPRESSION_RATE
        lo = p_open * COMPRESSION_RATE
        N_child = n_child_lvl - lo
        N_parent = 1
        child_k += lo.to(tl.int64) * (H_kv * K)
        child_v += lo.to(tl.int64) * (H_kv * V)
        parent_k += p_open.to(tl.int64) * (H_kv * K)
        parent_v += p_open.to(tl.int64) * (H_kv * V)

    parent_start = pid_seq * BLOCK_SIZE
    # Early check, although grid should be tight
    if parent_start >= N_parent:
        return

    child_start = parent_start * COMPRESSION_RATE

    # load K
    k_base = child_k + i_b.to(tl.int64) * N_child * H_kv * K + i_h.to(tl.int64) * K
    k_block_ptrs = tl.make_block_ptr(
        base=k_base,
        shape=(N_child, K),
        strides=(H_kv * K, 1),
        offsets=(child_start, 0),
        block_shape=(BLOCK_SIZE * COMPRESSION_RATE, K),
        order=(1, 0)
    )
    k_vals = tl.load(k_block_ptrs, boundary_check=(0, 1))

    # load V
    v_base = child_v + i_b.to(tl.int64) * N_child * H_kv * V + i_h.to(tl.int64) * V
    v_block_ptrs = tl.make_block_ptr(
        base=v_base,
        shape=(N_child, V),
        strides=(H_kv * V, 1),
        offsets=(child_start, 0),
        block_shape=(BLOCK_SIZE * COMPRESSION_RATE, V),
        order=(1, 0)
    )
    v_vals = tl.load(v_block_ptrs, boundary_check=(0, 1))

    # Reshape + Mean Pooling
    parent_global_idx = parent_start + tl.arange(0, BLOCK_SIZE)
    child_global_idx = parent_global_idx[:, None] * COMPRESSION_RATE + tl.arange(0, COMPRESSION_RATE)[None, :]
    child_valid = (parent_global_idx[:, None] < N_parent) & (child_global_idx < N_child)

    k_vals_reshaped = tl.reshape(k_vals, [BLOCK_SIZE, COMPRESSION_RATE, K])
    k_masked = tl.where(child_valid[:, :, None], k_vals_reshaped, 0.0)
    k_sum = tl.sum(k_masked, axis=1)
    num_valid_children = tl.sum(child_valid.to(tl.int32), axis=1)
    num_valid_children_safe = tl.maximum(num_valid_children, 1)
    k_mean = k_sum / num_valid_children_safe[:, None]

    v_vals_reshaped = tl.reshape(v_vals, [BLOCK_SIZE, COMPRESSION_RATE, V])
    v_masked = tl.where(child_valid[:, :, None], v_vals_reshaped, 0.0)
    v_sum = tl.sum(v_masked, axis=1)
    v_mean = v_sum / num_valid_children_safe[:, None]

    # store
    parent_k_base = parent_k + i_b.to(tl.int64) * N_parent * H_kv * K + i_h.to(tl.int64) * K
    parent_k_block_ptrs = tl.make_block_ptr(
        base=parent_k_base,
        shape=(N_parent, K),
        strides=(H_kv * K, 1),
        offsets=(parent_start, 0),
        block_shape=(BLOCK_SIZE, K),
        order=(1, 0)
    )
    tl.store(parent_k_block_ptrs, k_mean.to(parent_k.dtype.element_ty), boundary_check=(0, 1))

    parent_v_base = parent_v + i_b.to(tl.int64) * N_parent * H_kv * V + i_h.to(tl.int64) * V
    parent_v_block_ptrs = tl.make_block_ptr(
        base=parent_v_base,
        shape=(N_parent, V),
        strides=(H_kv * V, 1),
        offsets=(parent_start, 0),
        block_shape=(BLOCK_SIZE, V),
        order=(1, 0)
    )
    tl.store(parent_v_block_ptrs, v_mean.to(parent_v.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def recsa_probe_pool_kernel(
    child_k,
    child_v,  # [1, n<=C, H_kv, V]
    probe_w,  # [K] fp32
    probe_tau,  # [] fp32
    probe_gate,
    out_k,
    out_v,  # [1, 1, H_kv, V]
    n_child,  # runtime: valid children in the bucket, 1..C
    eps2,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,  # compression rate (bucket capacity); power of 2
    HAS_GATE: tl.constexpr,
    t_ptr=None,
    CHILD_POWER: tl.constexpr = 0,  # c^(level-1): child level's tokens/node
    HAS_CURSOR: tl.constexpr = 0,
):
    """Fused decode-shape shared_std probe pooling plus the two parent slot stores, in one launch."""
    h = tl.program_id(0)
    if HAS_CURSOR:
        t_new = (tl.load(t_ptr) + 1).to(tl.int32)
        n_child_lvl = (t_new + CHILD_POWER - 1) // CHILD_POWER
        p_open = (n_child_lvl - 1) // C
        lo = p_open * C
        n_child = n_child_lvl - lo
        child_k += lo.to(tl.int64) * (H_kv * K)
        child_v += lo.to(tl.int64) * (H_kv * V)
        out_k += p_open.to(tl.int64) * (H_kv * K)
        out_v += p_open.to(tl.int64) * (H_kv * V)
    ks = tl.arange(0, K)
    vs = tl.arange(0, V)
    cs = tl.arange(0, C)
    valid = cs < n_child
    cnt = tl.sum(valid.to(tl.float32), axis=0)

    k_base = child_k + h.to(tl.int64) * K
    kf = tl.load(k_base + cs[:, None] * (H_kv * K) + ks[None, :],
                 mask=valid[:, None], other=0.0).to(tl.float32)  # [C, K]

    # mu: serial ascending accumulation (mirrors torch's small-dim reduction)
    mu_acc = tl.zeros([K], dtype=tl.float32)
    for c in tl.static_range(C):
        row = tl.load(k_base + c * (H_kv * K) + ks,
                      mask=(ks * 0 + c) < n_child, other=0.0).to(tl.float32)
        mu_acc += row
    mu = mu_acc / cnt  # [K]

    dev = tl.where(valid[:, None], kf - mu[None, :], 0.0)  # [C, K]
    var = tl.sum(tl.sum(dev * dev, axis=1), axis=0) / (cnt * K)
    sigma = tl.sqrt(tl.maximum(var, eps2))

    w = tl.load(probe_w + ks).to(tl.float32)  # [K]
    tau = tl.load(probe_tau).to(tl.float32)
    logits = tau * tl.sum(dev * w[None, :], axis=1)  # [C]
    logits = logits / sigma
    logits = tl.where(valid, logits, float("-inf"))
    m = tl.max(logits, axis=0)  # finite: n_child >= 1
    e = tl.exp(logits - m)  # exp(-inf) = 0 for the pad children
    alpha = e / tl.sum(e, axis=0)  # [C]

    pooled = tl.sum(alpha[:, None] * kf, axis=0)  # [K]
    if HAS_GATE:
        g = tl.load(probe_gate).to(tl.float32)
    else:
        g = 1.0
    k_par = mu + g * (pooled - mu)
    tl.store(out_k + h.to(tl.int64) * K + ks, k_par.to(out_k.dtype.element_ty))

    # v_parent = mean(v): serial ascending, same reason as mu
    v_base = child_v + h.to(tl.int64) * V
    v_acc = tl.zeros([V], dtype=tl.float32)
    for c in tl.static_range(C):
        row = tl.load(v_base + c * (H_kv * V) + vs,
                      mask=(vs * 0 + c) < n_child, other=0.0).to(tl.float32)
        v_acc += row
    tl.store(out_v + h.to(tl.int64) * V + vs, (v_acc / cnt).to(out_v.dtype.element_ty))


def probe_pool_bucket(child_k, child_v, probe_w, probe_tau, probe_gate,
                      out_k, out_v, compression_rate, sigma_eps=1e-6,
                      t_dev=None, child_power=0):
    """Launch recsa_probe_pool_kernel on one open bucket; writes the parent slots in place."""
    H_kv, K = child_k.shape[2], child_k.shape[3]
    V = child_v.shape[3]
    recsa_probe_pool_kernel[(H_kv,)](
        child_k, child_v, probe_w, probe_tau,
        probe_gate if probe_gate is not None else probe_tau,
        out_k, out_v,
        child_k.shape[1], sigma_eps * sigma_eps,
        H_kv=H_kv, K=K, V=V, C=compression_rate,
        HAS_GATE=probe_gate is not None,
        t_ptr=t_dev,
        CHILD_POWER=child_power,
        HAS_CURSOR=1 if t_dev is not None else 0,
        num_warps=4,
    )


@triton.jit
def recsa_decode_frontier_kernel(
    prev,   # [B, 1, H_kv, TOP_K] int32 out
    t_ptr,  # int64 [1]: session t == the query's global row
    TOP_K: tl.constexpr,
    TOP_LAYER_POWER: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
):
    """Top-level frontier fill from the device cursor (CUDA-graph replayable): prev[..., i] = i if i < nvp else -1."""
    pid = tl.program_id(0)
    t_cur = tl.load(t_ptr).to(tl.int32)
    nvp = (t_cur // TOP_LAYER_POWER) // COMPRESSION_RATE + 1
    lanes = tl.arange(0, TOP_K).to(tl.int32)
    tl.store(prev + pid.to(tl.int64) * TOP_K + lanes,
             tl.where(lanes < nvp, lanes, -1))


RECSA_GATHER_PROGS = int(os.environ.get("RECSA_GATHER_PROGS", "64"))

RECSA_DECODE_SPLIT = int(os.environ.get("RECSA_DECODE_SPLIT", "1"))
RECSA_DSPLIT_NSLICE = int(os.environ.get("RECSA_DSPLIT_NSLICE", "16"))

RECSA_DSPLIT_FINALIZE = os.environ.get("RECSA_DSPLIT_FINALIZE", "v3").strip().lower()
if RECSA_DSPLIT_FINALIZE not in ("v2", "v3", "legacy"):
    raise ValueError("RECSA_DSPLIT_FINALIZE must be 'v2', 'v3' or 'legacy', "
                     f"got {RECSA_DSPLIT_FINALIZE!r}")
RECSA_DSPLIT_CHAIN = os.environ.get("RECSA_DSPLIT_CHAIN", "fused").strip().lower()
if RECSA_DSPLIT_CHAIN not in ("fused", "legacy"):
    raise ValueError("RECSA_DSPLIT_CHAIN must be 'fused' or 'legacy', "
                     f"got {RECSA_DSPLIT_CHAIN!r}")

RECSA_DSPLIT_HFUSE = os.environ.get("RECSA_DSPLIT_HFUSE", "on").strip().lower()
if RECSA_DSPLIT_HFUSE not in ("on", "off"):
    raise ValueError("RECSA_DSPLIT_HFUSE must be 'on' or 'off', "
                     f"got {RECSA_DSPLIT_HFUSE!r}")

_DSPLIT_DEBUG_CAPTURE = None


@triton.jit
def recsa_leaf_store_kernel(
    pool_k,  # [1, cap, H_kv, K] pinned host (UVA-mapped)
    pool_v,  # [1, cap, H_kv, V] pinned host
    k_t,     # [1, 1, H_kv, K] device, contiguous
    v_t,     # [1, 1, H_kv, V] device, contiguous
    t_ptr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """Write-through leaf append."""
    h = tl.program_id(0)
    t = tl.load(t_ptr)
    o_k = tl.arange(0, K)
    tl.store(pool_k + (t * H_kv + h) * K + o_k,
             tl.load(k_t + h.to(tl.int64) * K + o_k))
    o_v = tl.arange(0, V)
    tl.store(pool_v + (t * H_kv + h) * V + o_v,
             tl.load(v_t + h.to(tl.int64) * V + o_v))


@triton.jit
def recsa_bucket_cursor_kernel(
    t_ptr,
    win_i64,
    win_u8,
    win_f32,  # fp32 [>=N_POOLED, 2] OUT: float(n) | float(n * D)
    capacity,
    C: tl.constexpr,
    D: tl.constexpr,
    N_POOLED: tl.constexpr,
):
    """BucketCursor.refresh for every pooled level 1..N_POOLED in one launch; idx_z points pad positions (j >= n) at
    the child level's spare row, which no append within the capacity writes."""
    t = tl.load(t_ptr).to(tl.int64)
    j = tl.arange(0, C).to(tl.int64)
    cp = 1
    for lv in tl.static_range(N_POOLED):
        q = t // cp
        p = q // C
        raw = p * C + j
        invalid = raw > q
        n = q - p * C + 1
        idx = tl.minimum(raw, (capacity - 1) // cp)
        idx_z = tl.where(invalid, (capacity + cp - 1) // cp, raw)
        row = win_i64 + lv * (2 * C + 1)
        tl.store(row + j, idx)
        tl.store(row + C + j, idx_z)
        tl.store(row + 2 * C, p)
        tl.store(win_u8 + lv * C + j, invalid.to(tl.uint8))
        tl.store(win_f32 + lv * 2, n.to(tl.float32))
        tl.store(win_f32 + lv * 2 + 1, (n * D).to(tl.float32))
        cp = cp * C


@triton.jit
def recsa_vbatch_gather_kernel(
    src_ptrs,
    ref,
    idx,
    inv,       # uint8/bool [C]: the window's pad mask (j >= n)
    stage,     # fp32 [L, 1, C, H_kv, V] OUT
    ROW: tl.constexpr,    # H_kv * V
    BLOCK: tl.constexpr,  # next power of two >= ROW
    C: tl.constexpr,
):
    """Deferred-V gather: child row j of the open bucket of layer l, through a pointer table, widened to fp32; pad rows
    are stored as +0.0."""
    l = tl.program_id(0)
    jj = tl.program_id(1)
    base = tl.load(src_ptrs + l).to(tl.pointer_type(ref.dtype.element_ty))
    r = tl.load(idx + jj).to(tl.int64)
    bad = tl.load(inv + jj).to(tl.int32)
    o = tl.arange(0, BLOCK)
    x = tl.load(base + r * ROW + o, mask=(o < ROW) & (bad == 0), other=0.0)
    tl.store(stage + (l * C + jj).to(tl.int64) * ROW + o, x.to(tl.float32),
             mask=o < ROW)


@triton.jit
def recsa_vbatch_scatter_kernel(
    dst_ptrs,
    src,
    p_ptr,     # int64 [1]: the open parent's slot (BucketCursor p)
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Deferred-V scatter: out_v[l][:, p] = src[l] for every layer, through a pointer table."""
    l = tl.program_id(0)
    base = tl.load(dst_ptrs + l).to(tl.pointer_type(src.dtype.element_ty))
    p = tl.load(p_ptr).to(tl.int64)
    o = tl.arange(0, BLOCK)
    tl.store(base + p * ROW + o,
             tl.load(src + l.to(tl.int64) * ROW + o, mask=o < ROW), mask=o < ROW)


@triton.jit
def recsa_leaf_gather_kernel(
    k_pool,  # [1, cap, H_kv, K] pinned host (UVA-mapped)
    v_pool,  # [1, cap, H_kv, V] pinned host
    fetch_k,  # [H_kv, TOP_K*CR, K] device staging
    fetch_v,  # [H_kv, TOP_K*CR, V] device staging
    prev_selected_parents,  # [1, 1, H_kv, TOP_K] int32, contiguous
    t_base,  # host int: the query's global row (eager decode)
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    NPROG: tl.constexpr,
    t_ptr=None,  # int64 [1] session cursor (graph mode)
    HAS_CURSOR: tl.constexpr = 0,
):
    """Leaf gather for offloaded leaves: for every (kv-head, list-slot) pair, copy the CR leaf rows the bottom kernel
    reads from the pinned host pool into the GPU fetch buffer."""
    pid = tl.program_id(0)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
    N_layer = t_base + 1  # leaf count == the query's global row + 1
    o_ch = tl.arange(0, COMPRESSION_RATE)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    TOTAL: tl.constexpr = H_kv * TOP_K
    for w in range(pid, TOTAL, NPROG):
        h = w // TOP_K
        p_idx = tl.load(prev_selected_parents + w).to(tl.int32)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + o_ch
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        dst = w.to(tl.int64) * COMPRESSION_RATE
        tile_k = tl.load(k_pool + safe_rows[:, None] * (H_kv * K)
                         + h.to(tl.int64) * K + o_k[None, :])
        tl.store(fetch_k + dst * K + o_ch[:, None] * K + o_k[None, :], tile_k)
        tile_v = tl.load(v_pool + safe_rows[:, None] * (H_kv * V)
                         + h.to(tl.int64) * V + o_v[None, :])
        tl.store(fetch_v + dst * V + o_ch[:, None] * V + o_v[None, :], tile_v)


@triton.jit
def recsa_merge_to_global_kernel(
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    global_max,  # [B, T, H]      IN
    global_sum,  # [B, T, H]      IN
    global_output,  # [B, T, H, V]   IN
    out_max,  # [B, T, H]      OUT (must not alias global_max)
    out_sum,  # [B, T, H]      OUT
    out_output,  # [B, T, H, V]   OUT
    B,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """Kernel 2: merge layer state into global online-softmax accumulators."""
    i_t_loc = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    T_i64 = T
    state_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t_loc.to(tl.int64) * H
        + i_h.to(tl.int64)
    )

    cur_max = tl.load(layer_max + state_offset)
    cur_sum = tl.load(layer_sum + state_offset)

    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t_loc.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    cur_output = tl.load(layer_output + output_offset + o_v, mask=o_v < V, other=0.0)

    g_max = tl.load(global_max + state_offset)
    g_sum = tl.load(global_sum + state_offset)
    g_output_vals = tl.load(global_output + output_offset + o_v, mask=o_v < V, other=0.0)

    cur_has_contribution = cur_sum > 1e-10
    g_has_accumulation = g_sum > 1e-10

    new_max = tl.maximum(g_max, cur_max)
    scale_g = tl.where(g_has_accumulation, tl.exp(g_max - new_max), 0.0)
    scale_c = tl.where(cur_has_contribution, tl.exp(cur_max - new_max), 0.0)

    merged_sum = g_sum * scale_g + cur_sum * scale_c
    merged_output = g_output_vals * scale_g + cur_output * scale_c
    merged_max = new_max

    g_max = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_max, cur_max),
        g_max
    )
    g_sum = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_sum, cur_sum),
        g_sum
    )
    g_output_vals = tl.where(
        cur_has_contribution,
        tl.where(g_has_accumulation, merged_output, cur_output),
        g_output_vals
    )

    tl.store(out_max + state_offset, g_max)
    tl.store(out_sum + state_offset, g_sum)
    tl.store(out_output + output_offset + o_v, g_output_vals, mask=o_v < V)


@triton.jit
def recsa_final_normalize_kernel(
    global_output,  # [B, T, H, V]
    global_sum,  # [B, T, H]
    output,  # [B, T, H, V]
    B,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """recsa_final_normalize_kernel: output = global_output / global_sum. Grid: (T, B*H)"""
    i_t_loc = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    # load sum
    T_i64 = T
    sum_offset = (
        i_b.to(tl.int64) * T_i64 * H
        + i_t_loc.to(tl.int64) * H
        + i_h.to(tl.int64)
    )
    sum_val = tl.load(global_sum + sum_offset)

    # load and normalize output
    output_offset = (
        i_b.to(tl.int64) * T_i64 * H * V
        + i_t_loc.to(tl.int64) * H * V
        + i_h.to(tl.int64) * V
    )
    o_v = tl.arange(0, V)
    output_ptrs = global_output + output_offset + o_v
    output_vals = tl.load(output_ptrs, mask=o_v < V, other=0.0)

    normalized = output_vals / sum_val

    # write back
    out_ptrs = output + output_offset + o_v
    tl.store(out_ptrs, normalized, mask=o_v < V)


@triton.jit
def recsa_bottom_accumulate_gqa_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # outputs
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    # params / constexpr
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    fetch_k=None,  # [H_kv, TOP_K*CR, K] staged leaf tiles
    fetch_v=None,  # [H_kv, TOP_K*CR, V]
    LEAF_STAGED: tl.constexpr = 0,
    DOT_FP16: tl.constexpr = 0,
):
    """Bottom layer forward: batch-gather K/V tiles + tl.dot (tensor core)."""
    i_t_loc = tl.program_id(0)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base + i_t_loc
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # ---- Load Q for G_PAD heads (zero-pad when G_PAD > NUM_GROUPS) ----
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale  # [G_PAD, K//2]
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale  # [G_PAD, K//2]

    # ---- Online-softmax state (G_PAD-wide) ----
    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    # ---- Precompute base pointers & offsets ----
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    if LEAF_STAGED:
        # staged tiles are per-(kv-head) slices, densely packed by list slot
        k_base = fetch_k + i_h_kv.to(tl.int64) * (TOP_K * COMPRESSION_RATE * K)
        v_base = fetch_v + i_h_kv.to(tl.int64) * (TOP_K * COMPRESSION_RATE * V)
    else:
        k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
        v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE   # [TILE_N] which parent (0..TILE_P-1)
    child_in_tile = o_n % COMPRESSION_RATE     # [TILE_N] which child  (0..CR-1)
    o_v = tl.arange(0, V).to(tl.int64)

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t_loc.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t_loc.to(tl.int64) * H * V

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    for tile_idx in range(num_tiles):
        tile_base = (tile_idx * TILE_P).to(tl.int32)

        # Global parent offset for each of the TILE_N positions
        p_offs = tile_base + parent_in_tile                               # [TILE_N]

        # Unconditional load: clamp index to [0, TOP_K-1]
        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(
            prev_selected_parents + prev_sel_base + safe_p_offs
        ).to(tl.int32)                                                    # [TILE_N]

        # Compute validity mask (pure 1-D, no layout interaction with tl.dot)
        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)

        # Clamp row indices for unconditional K / V loads
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        if LEAF_STAGED:
            slot = (safe_p_offs * COMPRESSION_RATE
                    + child_in_tile.to(tl.int64))

        # ---- Gather K (two halves, unconditional) ----
        if LEAF_STAGED:
            k_row = k_base + slot[:, None] * K
        else:
            k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)           # [TILE_N, K//2]
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        # ---- RoPE on K: in-tile rotation then per-tile base rotation ----
        base_pos = tile_base.to(tl.int64) * COMPRESSION_RATE
        cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
        sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
        k1_lo = k_lo * cos_off - k_hi * sin_off
        k1_hi = k_lo * sin_off + k_hi * cos_off
        k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]                      # [TILE_N, K//2]
        k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]                      # [TILE_N, K//2]

        if DOT_FP16 == 1:
            scores = (tl.dot(q_rope_lo.to(tl.float16), tl.trans(k_rope_lo.to(tl.float16)))
                      + tl.dot(q_rope_hi.to(tl.float16), tl.trans(k_rope_hi.to(tl.float16))))
        elif DOT_FP16 == 2:
            scores = (tl.dot(q_rope_lo, tl.trans(k_rope_lo), input_precision="ieee")
                      + tl.dot(q_rope_hi, tl.trans(k_rope_hi), input_precision="ieee"))
        else:
            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG_INF)

        # ---- Gather V (unconditional) ----
        if LEAF_STAGED:
            v_row = v_base + slot[:, None] * V
        else:
            v_row = v_base + safe_rows[:, None] * (H_kv * V)
        if DOT_FP16 == 1:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float16)  # PV runs fp16 (probs cast to v_tile.dtype)
        else:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)                # [TILE_N, V]

        # ---- Online softmax ----
        tile_m = tl.max(scores, axis=1)                                   # [G_PAD]
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

        # ---- PV via tl.dot: [G_PAD, TILE_N] x [TILE_N, V] -> [G_PAD, V] ----
        if DOT_FP16 == 2:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile, input_precision="ieee")
        else:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    # ---- Store (first NUM_GROUPS heads only) ----
    tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, cur_output, mask=g_valid[:, None])


@triton.jit
def recsa_select_accumulate_gqa_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    # outputs
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    next_selected_parents,  # [B, T, H_kv, TOP_K]  (sorted asc, -1 padded)
    # params / constexpr
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    N_DIMS_TOPK: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    DROP_BLOCK: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    NO_SUMMARIES: tl.constexpr = 0,  # summaries="none": drop nothing into the softmax
):
    """Non-bottom layer forward: streaming Top-K + accumulate dropped."""
    i_t_loc = tl.program_id(0)
    i_t = t_base + i_t_loc
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # ---- Load Q for G_PAD heads (zero-pad when G_PAD > NUM_GROUPS) ----
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale  # [G_PAD, K//2]
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale  # [G_PAD, K//2]

    # ---- Online-softmax state (G_PAD-wide) ----
    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t_loc.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t_loc.to(tl.int64) * H * V

    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    idx_mask = (1 << LOG_N) - 1
    rightmost_pos = (num_candidates - 1).to(tl.int32)
    topk_running_max = tl.full([1], NEG_INF, dtype=tl.float32)

    init_idx = tl.arange(0, TOP_K).to(tl.int32)
    init_imp = tl.full([TOP_K], NEG_INF, dtype=tl.float32)
    init_int = init_imp.to(tl.int32, bitcast=True)
    init_encoded = (init_int & ~idx_mask) | (init_idx & idx_mask)
    running_topk_encoded = init_encoded.to(tl.float32, bitcast=True)

    o_v = tl.arange(0, V).to(tl.int64)
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    N_DROP_BLOCKS: tl.constexpr = TOP_K // DROP_BLOCK
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
    tile_rows = tl.arange(0, TILE_P).to(tl.int32)
    row_idx = tl.arange(0, 2)[:, None]
    block_ids = tl.arange(0, N_DROP_BLOCKS)[:, None]

    for i_batch in range(num_batches):
        # ---- step 1: TILE_P parents per tl.dot tile ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff
            ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k   # [TILE_N, K//2]
            k_rope_hi = k_lo * sin_k + k_hi * cos_k   # [TILE_N, K//2]

            # Split tl.dot: [G_PAD, K//2] x [K//2, TILE_N] -> [G_PAD, TILE_N]
            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_child_rows * H_kv + i_h_kv.to(tl.int64))
                scores = scores + b_vals[None, :]
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            # Update per-batch running max and rescale previously written importance.
            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            # Importance: sum_g exp(score_g - batch_m). This stays finite (<= G_PAD).
            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)  # [TILE_N]
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            # Write tiled importance back to [PARENTS_PER_BATCH, CR]
            for i_r in tl.static_range(TILE_P):
                row_off = (tile_base + i_r).to(tl.int32)
                row_mask_in_tile = (tile_rows == i_r)[:, None]
                row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                is_row = (batch_rows == row_off)[:, None]
                importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        # ---- step 2: Top-K merge sort ----
        importance_flat = tl.reshape(importance_matrix, [BC])
        buf_off = (i_batch * BC).to(tl.int32)
        pos = buf_off + tl.arange(0, BC).to(tl.int32)
        valid_pos = pos < num_candidates
        importance_flat = tl.where(valid_pos, importance_flat, NEG_INF)

        sel_m_new = tl.maximum(topk_running_max, batch_m)
        rescale_run = tl.exp(topk_running_max - sel_m_new)
        rescale_batch = tl.exp(batch_m - sel_m_new)

        run_int = running_topk_encoded.to(tl.int32, bitcast=True)
        run_raw_idx = run_int & idx_mask
        run_clean_int = run_int & ~idx_mask
        run_clean_imp = run_clean_int.to(tl.float32, bitcast=True)
        run_rescaled_imp = tl.where(run_clean_imp >= 0, run_clean_imp * rescale_run, run_clean_imp)
        run_rescaled_int = run_rescaled_imp.to(tl.int32, bitcast=True)
        run_rescaled_encoded = (run_rescaled_int & ~idx_mask) | run_raw_idx
        running_topk_encoded = run_rescaled_encoded.to(tl.float32, bitcast=True)

        importance_flat = tl.where(importance_flat >= 0, importance_flat * rescale_batch, importance_flat)
        importance_flat = tl.where(pos == rightmost_pos, 1e6, importance_flat)
        topk_running_max = sel_m_new

        importance_int = importance_flat.to(tl.int32, bitcast=True)
        encoded_idx = tl.where(importance_flat >= 0, ~pos, pos) & idx_mask
        encoded_int = (importance_int & ~idx_mask) | encoded_idx
        batch_encoded = encoded_int.to(tl.float32, bitcast=True)

        running_b = tl.broadcast_to(running_topk_encoded[None, :], [2, TOP_K])
        batch_b = tl.broadcast_to(batch_encoded[None, :], [2, TOP_K])
        merged_2d = tl.where(row_idx == 0, running_b, batch_b)
        merged_input = tl.reshape(merged_2d, [2 * TOP_K])

        N_DIMS_MERGE: tl.constexpr = N_DIMS_TOPK + 1
        sorted_merged = tl.sort(merged_input, descending=True)
        merged_sorted_2d = tl.reshape(sorted_merged, [2, TOP_K])

        running_topk_encoded = tl.sum(tl.where(row_idx == 0, merged_sorted_2d, 0.0), axis=0)
        dropped_encoded = tl.sum(tl.where(row_idx == 1, merged_sorted_2d, 0.0), axis=0)

        drop_int = dropped_encoded.to(tl.int32, bitcast=True)
        raw_idx = drop_int & idx_mask
        clean_int = drop_int & ~idx_mask
        clean_imp = clean_int.to(tl.float32, bitcast=True)
        drop_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx)
        drop_pos = (drop_pos & idx_mask).to(tl.int32)
        drop_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (drop_pos < num_candidates)

        # ---- step 3: accumulate dropped via tl.dot ----
        drop_pos_2d = tl.reshape(drop_pos, [N_DROP_BLOCKS, DROP_BLOCK]).to(tl.int32)
        drop_valid_2d_i32 = tl.reshape(drop_valid.to(tl.int32), [N_DROP_BLOCKS, DROP_BLOCK])

        # NO_SUMMARIES: constexpr-zero trip count (the dropped half is not accumulated)
        for drop_block_idx in range(N_DROP_BLOCKS * (1 - NO_SUMMARIES)):
            row_mask = block_ids == drop_block_idx
            pos_blk = tl.sum(tl.where(row_mask, drop_pos_2d, 0), axis=0).to(tl.int32)
            valid_blk_i32 = tl.sum(tl.where(row_mask, drop_valid_2d_i32, 0), axis=0).to(tl.int32)
            valid_blk = valid_blk_i32 > 0

            parent_off_d = (pos_blk // COMPRESSION_RATE).to(tl.int32)
            child_slot = (pos_blk - parent_off_d * COMPRESSION_RATE).to(tl.int32)
            safe_poff_d = tl.minimum(tl.maximum(parent_off_d, 0), TOP_K - 1).to(tl.int64)
            parent_idx_d = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff_d
            ).to(tl.int32)

            is_rm_d = parent_idx_d == rightmost_parent_idx
            child_ok_d = (~is_rm_d) | (child_slot <= rightmost_child_idx)
            node_idx = tl.maximum(parent_idx_d, 0) * COMPRESSION_RATE + child_slot
            node_valid = valid_blk & (parent_idx_d >= 0) & child_ok_d & (node_idx >= 0) & (node_idx < N_layer)

            safe_node = tl.minimum(tl.maximum(node_idx, 0), N_layer - 1).to(tl.int64)
            safe_pos_d = tl.maximum(pos_blk, 0).to(tl.int64)

            k_row_d = k_base + safe_node[:, None] * (H_kv * K)
            k_lo_drop = tl.load(k_row_d + o_k_half[None, :]).to(tl.float32)
            k_hi_drop = tl.load(k_row_d + (K // 2) + o_k_half[None, :]).to(tl.float32)

            cos_k_drop = tl.load(cos_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            sin_k_drop = tl.load(sin_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo_drop = k_lo_drop * cos_k_drop - k_hi_drop * sin_k_drop   # [DROP_BLOCK, K//2]
            k_rope_hi_drop = k_lo_drop * sin_k_drop + k_hi_drop * cos_k_drop   # [DROP_BLOCK, K//2]

            # Split tl.dot: [G_PAD, K//2] x [K//2, DROP_BLOCK] -> [G_PAD, DROP_BLOCK]
            scores_d = tl.dot(q_rope_lo, tl.trans(k_rope_lo_drop)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi_drop))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_node * H_kv + i_h_kv.to(tl.int64))
                scores_d = scores_d + b_vals[None, :]
            scores_d = tl.where(node_valid[None, :] & g_valid[:, None], scores_d, NEG_INF)

            v_row_d = v_base + safe_node[:, None] * (H_kv * V)
            v_tile = tl.load(v_row_d + o_v[None, :]).to(tl.float32)  # [DROP_BLOCK, V]

            tile_m = tl.max(scores_d, axis=1)  # [G_PAD]
            new_max = tl.maximum(cur_max, tile_m)
            alpha = tl.exp(cur_max - new_max)
            cur_sum = cur_sum * alpha
            cur_output = cur_output * alpha[:, None]

            attn_probs = tl.exp(scores_d - new_max[:, None])
            attn_probs = tl.where(node_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

            # PV via tl.dot: [G_PAD, DROP_BLOCK] x [DROP_BLOCK, V] -> [G_PAD, V]
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
            cur_max = new_max

    # ---- Store layer states (first NUM_GROUPS heads only) ----
    tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, cur_output, mask=g_valid[:, None])

    # ---- Decode final Top-K -> next_selected_parents ----
    topk_int = running_topk_encoded.to(tl.int32, bitcast=True)
    raw_idx = topk_int & idx_mask
    clean_int = topk_int & ~idx_mask
    clean_imp = clean_int.to(tl.float32, bitcast=True)
    topk_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx)
    topk_pos = (topk_pos & idx_mask).to(tl.int32)
    topk_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (topk_pos < num_candidates)
    topk_pos = tl.where(topk_valid, topk_pos, -1).to(tl.int32)

    parent_off_f = tl.where(topk_pos >= 0, topk_pos // COMPRESSION_RATE, 0).to(tl.int32)
    child_slot_f = tl.where(topk_pos >= 0, topk_pos - parent_off_f * COMPRESSION_RATE, 0).to(tl.int32)
    parent_idx_f = tl.load(
        prev_selected_parents + prev_sel_base + parent_off_f.to(tl.int64),
        mask=topk_pos >= 0,
        other=-1,
    ).to(tl.int32)
    selected_node_indices = tl.where(
        topk_pos >= 0,
        parent_idx_f * COMPRESSION_RATE + child_slot_f,
        -1,
    ).to(tl.int32)

    MAX_IDX: tl.constexpr = 2147483647
    selected_sorted = tl.where(selected_node_indices >= 0, selected_node_indices, MAX_IDX)
    selected_sorted = tl.sort(selected_sorted, descending=False)
    selected_sorted = tl.where(selected_sorted < MAX_IDX, selected_sorted, -1)

    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(next_selected_parents + out_par_base + tl.arange(0, TOP_K), selected_sorted.to(tl.int32))


@triton.jit
def recsa_mid_select_accumulate_ns_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    layer_max,  # [B, T, H]
    layer_sum,  # [B, T, H]
    layer_output,  # [B, T, H, V]
    next_selected_parents,  # [B, T, H_kv, TOP_K]; host pre-filled with -1
    scratch,  # [n_prog_t * B * H_kv * 3 * TOP_K] fp32 workspace
    t_offset,
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    DROP_BLOCK: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    NS_FAST: tl.constexpr,  # 1 iff TILE_P == PARENTS_PER_BATCH (host asserts)
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    SELECT_ONLY: tl.constexpr = 0,  # 1: selection replay only (no accumulation)
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    NO_SUMMARIES: tl.constexpr = 0,  # summaries="none": accumulation skipped,
    # the state stores below still run and write the neutral (NEG_INF, 0, 0)
):
    """Mid-layer forward without the per-batch bitonic sort."""
    SIGN: tl.constexpr = -2147483648  # int32 bit pattern 0x80000000
    pid_local = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_t_loc = t_offset + pid_local
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base + i_t_loc
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # ---- Load Q for G_PAD heads (zero-pad when G_PAD > NUM_GROUPS) ----
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale  # [G_PAD, K//2]
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale  # [G_PAD, K//2]

    # ---- Online-softmax state (G_PAD-wide) ----
    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t_loc.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t_loc.to(tl.int64) * H * V

    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    idx_mask = (1 << LOG_N) - 1
    rightmost_pos = (num_candidates - 1).to(tl.int32)
    topk_running_max = tl.full([1], NEG_INF, dtype=tl.float32)

    # ---- Running Top-K staged in scratch (stable ascending-slot order) ----
    n_prog_t = tl.num_programs(0)
    sbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_local) * (3 * TOP_K)
    o_tk = tl.arange(0, TOP_K).to(tl.int64)

    init_idx = tl.arange(0, TOP_K).to(tl.int32)
    init_imp = tl.full([TOP_K], NEG_INF, dtype=tl.float32)
    init_int = init_imp.to(tl.int32, bitcast=True)
    init_encoded = (init_int & ~idx_mask) | (init_idx & idx_mask)
    tl.store(scratch + sbase + o_tk, init_encoded.to(tl.float32, bitcast=True))
    tl.debug_barrier()

    o_v = tl.arange(0, V).to(tl.int64)
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
    tile_rows = tl.arange(0, TILE_P).to(tl.int32)
    o_db = tl.arange(0, DROP_BLOCK).to(tl.int32)

    for i_batch in range(num_batches):
        # ---- step 1: identical scoring / rescale chain to the sorting kernel ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff
            ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k   # [TILE_N, K//2]
            k_rope_hi = k_lo * sin_k + k_hi * cos_k   # [TILE_N, K//2]

            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_child_rows * H_kv + i_h_kv.to(tl.int64))
                scores = scores + b_vals[None, :]
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)  # [TILE_N]
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            if NS_FAST == 1:
                importance_matrix = importance_tile_2d
            else:
                for i_r in tl.static_range(TILE_P):
                    row_off = (tile_base + i_r).to(tl.int32)
                    row_mask_in_tile = (tile_rows == i_r)[:, None]
                    row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                    row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                    is_row = (batch_rows == row_off)[:, None]
                    importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        # ---- step 2: encode + radix rank select (no sort) ----
        importance_flat = tl.reshape(importance_matrix, [BC])
        buf_off = (i_batch * BC).to(tl.int32)
        pos = buf_off + tl.arange(0, BC).to(tl.int32)
        valid_pos = pos < num_candidates
        importance_flat = tl.where(valid_pos, importance_flat, NEG_INF)

        sel_m_new = tl.maximum(topk_running_max, batch_m)
        rescale_run = tl.exp(topk_running_max - sel_m_new)
        rescale_batch = tl.exp(batch_m - sel_m_new)

        run_enc = tl.load(scratch + sbase + o_tk)  # [TOP_K]
        run_int = run_enc.to(tl.int32, bitcast=True)
        run_raw_idx = run_int & idx_mask
        run_clean_imp = (run_int & ~idx_mask).to(tl.float32, bitcast=True)
        run_resc = tl.where(run_clean_imp >= 0, run_clean_imp * rescale_run, run_clean_imp)
        run_enc2 = (run_resc.to(tl.int32, bitcast=True) & ~idx_mask) | run_raw_idx

        importance_flat = tl.where(importance_flat >= 0, importance_flat * rescale_batch, importance_flat)
        importance_flat = tl.where(pos == rightmost_pos, 1e6, importance_flat)
        topk_running_max = sel_m_new

        flat_int = importance_flat.to(tl.int32, bitcast=True)
        enc_idx = tl.where(importance_flat >= 0, ~pos, pos) & idx_mask
        bat_enc = (flat_int & ~idx_mask) | enc_idx  # [BC]

        us_run = (run_enc2 ^ ((run_enc2 >> 31) | SIGN)) ^ SIGN
        us_bat = (bat_enc ^ ((bat_enc >> 31) | SIGN)) ^ SIGN

        m_ids = tl.arange(0, 16).to(tl.int32)
        tau = tl.zeros([1], dtype=tl.int32)
        for g in tl.static_range(8):
            sh = 28 - 4 * g
            thr = (tau | (m_ids << sh)) ^ SIGN
            cnts = (
                tl.sum((us_run[None, :] >= thr[:, None]).to(tl.int32), axis=1)
                + tl.sum((us_bat[None, :] >= thr[:, None]).to(tl.int32), axis=1)
            )
            tau = tau | ((tl.sum((cnts >= TOP_K).to(tl.int32)) - 1) << sh)

        taus = tau ^ SIGN
        keep_r = us_run >= taus
        keep_b = us_bat >= taus

        rank_r = tl.cumsum(keep_r.to(tl.int32), axis=0) - keep_r.to(tl.int32)
        nkeep_r = tl.sum(keep_r.to(tl.int32))
        rank_b = nkeep_r + tl.cumsum(keep_b.to(tl.int32), axis=0) - keep_b.to(tl.int32)

        drop_r = ~keep_r
        drop_b = ~keep_b
        drank_r = tl.cumsum(drop_r.to(tl.int32), axis=0) - drop_r.to(tl.int32)
        ndrop_r = tl.sum(drop_r.to(tl.int32))
        drank_b = ndrop_r + tl.cumsum(drop_b.to(tl.int32), axis=0) - drop_b.to(tl.int32)
        n_drop = ndrop_r + tl.sum(drop_b.to(tl.int32))

        tl.debug_barrier()  # all lanes loaded run_enc before overwrite
        tl.store(
            scratch + sbase + rank_r.to(tl.int64),
            run_enc2.to(tl.float32, bitcast=True),
            mask=keep_r & (rank_r < TOP_K),
        )
        tl.store(
            scratch + sbase + rank_b.to(tl.int64),
            bat_enc.to(tl.float32, bitcast=True),
            mask=keep_b & (rank_b < TOP_K),
        )
        if SELECT_ONLY == 0:
            tl.store(
                scratch + sbase + TOP_K + drank_r.to(tl.int64),
                run_enc2.to(tl.float32, bitcast=True),
                mask=drop_r & (drank_r < 2 * TOP_K),
            )
            tl.store(
                scratch + sbase + TOP_K + drank_b.to(tl.int64),
                bat_enc.to(tl.float32, bitcast=True),
                mask=drop_b & (drank_b < 2 * TOP_K),
            )
        tl.debug_barrier()

        for dbi in range((BC // DROP_BLOCK) * (1 - SELECT_ONLY) * (1 - NO_SUMMARIES)):
            lanes = dbi * DROP_BLOCK + o_db
            lane_ok = lanes < n_drop
            enc_d = tl.load(
                scratch + sbase + TOP_K + lanes.to(tl.int64),
                mask=lane_ok, other=NEG_INF,
            )
            eint = enc_d.to(tl.int32, bitcast=True)
            raw_idx_d = eint & idx_mask
            clean_imp_d = (eint & ~idx_mask).to(tl.float32, bitcast=True)
            pos_blk = (tl.where(clean_imp_d >= 0, ~raw_idx_d, raw_idx_d) & idx_mask).to(tl.int32)
            valid_blk = lane_ok & (clean_imp_d > SCORE_VALID_THRESHOLD) & (pos_blk < num_candidates)

            parent_off_d = (pos_blk // COMPRESSION_RATE).to(tl.int32)
            child_slot = (pos_blk - parent_off_d * COMPRESSION_RATE).to(tl.int32)
            safe_poff_d = tl.minimum(tl.maximum(parent_off_d, 0), TOP_K - 1).to(tl.int64)
            parent_idx_d = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff_d
            ).to(tl.int32)

            is_rm_d = parent_idx_d == rightmost_parent_idx
            child_ok_d = (~is_rm_d) | (child_slot <= rightmost_child_idx)
            node_idx = tl.maximum(parent_idx_d, 0) * COMPRESSION_RATE + child_slot
            node_valid = valid_blk & (parent_idx_d >= 0) & child_ok_d & (node_idx >= 0) & (node_idx < N_layer)

            safe_node = tl.minimum(tl.maximum(node_idx, 0), N_layer - 1).to(tl.int64)
            safe_pos_d = tl.maximum(pos_blk, 0).to(tl.int64)

            k_row_d = k_base + safe_node[:, None] * (H_kv * K)
            k_lo_drop = tl.load(k_row_d + o_k_half[None, :]).to(tl.float32)
            k_hi_drop = tl.load(k_row_d + (K // 2) + o_k_half[None, :]).to(tl.float32)

            cos_k_drop = tl.load(cos_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            sin_k_drop = tl.load(sin_cache + safe_pos_d[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo_drop = k_lo_drop * cos_k_drop - k_hi_drop * sin_k_drop   # [DROP_BLOCK, K//2]
            k_rope_hi_drop = k_lo_drop * sin_k_drop + k_hi_drop * cos_k_drop   # [DROP_BLOCK, K//2]

            scores_d = tl.dot(q_rope_lo, tl.trans(k_rope_lo_drop)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi_drop))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_node * H_kv + i_h_kv.to(tl.int64))
                scores_d = scores_d + b_vals[None, :]
            scores_d = tl.where(node_valid[None, :] & g_valid[:, None], scores_d, NEG_INF)

            v_row_d = v_base + safe_node[:, None] * (H_kv * V)
            v_tile = tl.load(v_row_d + o_v[None, :]).to(tl.float32)  # [DROP_BLOCK, V]

            tile_m = tl.max(scores_d, axis=1)  # [G_PAD]
            new_max = tl.maximum(cur_max, tile_m)
            alpha = tl.exp(cur_max - new_max)
            cur_sum = cur_sum * alpha
            cur_output = cur_output * alpha[:, None]

            attn_probs_d = tl.exp(scores_d - new_max[:, None])
            attn_probs_d = tl.where(node_valid[None, :] & g_valid[:, None], attn_probs_d, 0.0)
            cur_sum = cur_sum + tl.sum(attn_probs_d, axis=1)

            cur_output = cur_output + tl.dot(attn_probs_d.to(v_tile.dtype), v_tile)
            cur_max = new_max

    # ---- Store layer states (first NUM_GROUPS heads only) ----
    if SELECT_ONLY == 0:
        tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
        tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
        out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
        tl.store(out_ptrs, cur_output, mask=g_valid[:, None])

    # ---- Decode final Top-K -> next_selected_parents (ascending, no sort) ----
    run_encf = tl.load(scratch + sbase + o_tk)
    topk_int = run_encf.to(tl.int32, bitcast=True)
    raw_idx = topk_int & idx_mask
    clean_imp = (topk_int & ~idx_mask).to(tl.float32, bitcast=True)
    topk_pos = (tl.where(clean_imp >= 0, ~raw_idx, raw_idx) & idx_mask).to(tl.int32)
    topk_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (topk_pos < num_candidates)

    parent_off_f = tl.where(topk_valid, topk_pos // COMPRESSION_RATE, 0).to(tl.int32)
    child_slot_f = tl.where(topk_valid, topk_pos - parent_off_f * COMPRESSION_RATE, 0).to(tl.int32)
    parent_idx_f = tl.load(
        prev_selected_parents + prev_sel_base + parent_off_f.to(tl.int64),
        mask=topk_valid,
        other=-1,
    ).to(tl.int32)
    selected_node = tl.where(
        topk_valid & (parent_idx_f >= 0),
        parent_idx_f * COMPRESSION_RATE + child_slot_f,
        -1,
    ).to(tl.int32)
    out_ok = selected_node >= 0

    out_rank = tl.cumsum(out_ok.to(tl.int32), axis=0) - out_ok.to(tl.int32)
    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(
        next_selected_parents + out_par_base + out_rank.to(tl.int64),
        selected_node,
        mask=out_ok,
    )


@triton.jit
def recsa_mid_select_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    next_selected_parents,  # [B, T, H_kv, TOP_K]; host pre-filled with -1
    scratch_imp,  # [n_prog_t * B * H_kv * MAX_CAND] fp32
    scratch_tau,  # [n_prog_t * B * H_kv] int32 (us-domain threshold)
    t_offset,
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    force_mask=None,
    HAS_FORCE: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    DEFER_RESCALE: tl.constexpr = 0,
    SEL_IEEE: tl.constexpr = 0,
):
    """Mid-layer selection only (no V accumulation): score all candidates into scratch, then one 32-step bit-bisection
    rank select."""
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    NTILE: tl.constexpr = MAX_CAND // BC  # == COMPRESSION_RATE batches max

    pid_local = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_t_loc = t_offset + pid_local
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base + i_t_loc
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1
    rightmost_pos = (num_candidates - 1).to(tl.int32)

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    idx_mask = (1 << LOG_N) - 1
    running_M = tl.full([1], NEG_INF, dtype=tl.float32)

    n_prog_t = tl.num_programs(0)
    sbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_local) * MAX_CAND
    tau_base = pid_bh_kv.to(tl.int64) * n_prog_t + pid_local

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
    tile_rows = tl.arange(0, TILE_P).to(tl.int32)
    o_bc = tl.arange(0, BC).to(tl.int32)
    o_bc64 = o_bc.to(tl.int64)
    if DEFER_RESCALE:
        # rescale_prefix of batch i lands in rp_vec[i]; replayed after the loop
        o_nt = tl.arange(0, NTILE).to(tl.int32)
        rp_vec = tl.zeros([NTILE], dtype=tl.float32)

    for i_batch in range(num_batches):
        # ---- step A: batch scoring, as in the sorting kernel ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff
            ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            if SEL_IEEE:
                scores = (tl.dot(q_rope_lo, tl.trans(k_rope_lo), input_precision="ieee")
                          + tl.dot(q_rope_hi, tl.trans(k_rope_hi), input_precision="ieee"))
            else:
                scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_child_rows * H_kv + i_h_kv.to(tl.int64))
                scores = scores + b_vals[None, :]
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            for i_r in tl.static_range(TILE_P):
                row_off = (tile_base + i_r).to(tl.int32)
                row_mask_in_tile = (tile_rows == i_r)[:, None]
                row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                is_row = (batch_rows == row_off)[:, None]
                importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        # ---- step B: replay the sorting kernel's rescale chain on scratch ----
        importance_flat = tl.reshape(importance_matrix, [BC])
        buf_off = (i_batch * BC).to(tl.int32)
        pos = buf_off + o_bc
        valid_pos = pos < num_candidates
        importance_flat = tl.where(valid_pos, importance_flat, NEG_INF)

        new_M = tl.maximum(running_M, batch_m)
        rescale_prefix = tl.exp(running_M - new_M)
        rescale_batch = tl.exp(batch_m - new_M)

        if DEFER_RESCALE:
            rp_vec = tl.where(o_nt == i_batch, rescale_prefix, rp_vec)
        else:
            for pj in range(i_batch):
                poff = (pj * BC).to(tl.int64)
                prev_vals = tl.load(scratch_imp + sbase + poff + o_bc64)
                prev_vals = tl.where(prev_vals >= 0, prev_vals * rescale_prefix, prev_vals)
                pbits = prev_vals.to(tl.int32, bitcast=True) & ~idx_mask
                tl.store(scratch_imp + sbase + poff + o_bc64, pbits.to(tl.float32, bitcast=True))

        importance_flat = tl.where(importance_flat >= 0, importance_flat * rescale_batch, importance_flat)
        importance_flat = tl.where(pos == rightmost_pos, 1e6, importance_flat)
        if HAS_FORCE:
            f_par_off = pos // COMPRESSION_RATE
            f_safe = tl.minimum(tl.maximum(f_par_off, 0), TOP_K - 1).to(tl.int64)
            f_par = tl.load(prev_selected_parents + prev_sel_base + f_safe).to(tl.int32)
            f_node = tl.maximum(f_par, 0) * COMPRESSION_RATE + (pos % COMPRESSION_RATE)
            f_ok = (f_par >= 0) & valid_pos
            f_word = tl.load(force_mask + (f_node >> 5).to(tl.int64), mask=f_ok, other=0)
            f_hit = ((f_word >> (f_node & 31)) & 1) != 0
            importance_flat = tl.where(f_hit & f_ok, 1e6, importance_flat)
        running_M = new_M

        fbits = importance_flat.to(tl.int32, bitcast=True) & ~idx_mask
        tl.store(scratch_imp + sbase + buf_off.to(tl.int64) + o_bc64, fbits.to(tl.float32, bitcast=True))

    tl.debug_barrier()

    if DEFER_RESCALE:
        row2 = o_nt[:, None]
        slot2 = row2 * BC + o_bc[None, :]  # [NTILE, BC] candidate slot, row-major
        live2 = row2 < num_batches
        tile_ptrs = scratch_imp + sbase + slot2.to(tl.int64)
        v = tl.load(tile_ptrs, mask=live2, other=NEG_INF)
        for j in tl.static_range(1, NTILE):
            f_j = tl.sum(tl.where(o_nt == j, rp_vec, 0.0), axis=0)
            v_j = tl.where(v >= 0, v * f_j, v)
            v_j = (v_j.to(tl.int32, bitcast=True) & ~idx_mask).to(tl.float32, bitcast=True)
            v = tl.where((row2 < j) & (j < num_batches), v_j, v)
        tl.store(tile_ptrs, v, mask=live2)  # final scale, read by M-accum

        vals = tl.where(slot2 < num_candidates, v, NEG_INF)
        vi = vals.to(tl.int32, bitcast=True)
        enc_idx = tl.where(vals >= 0, ~slot2, slot2) & idx_mask
        enc = (vi & ~idx_mask) | enc_idx
        us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
        tau = tl.zeros([1], dtype=tl.int32)
        for step in tl.static_range(32):
            if step == 0:
                cand_t = tau | SIGN
            else:
                cand_t = tau | (1 << (31 - step))
            thresh = cand_t ^ SIGN
            cnt = tl.sum(tl.sum((us >= thresh).to(tl.int32), axis=1), axis=0)
            tau = tl.where(cnt >= TOP_K, cand_t, tau)
        taus = tau ^ SIGN
        tl.store(scratch_tau + tau_base + tl.arange(0, 1).to(tl.int64), taus)

        # ---- step D': emit selected; rank = kept slots before it (slot order) ----
        out_par_base = prev_sel_base  # next/prev share the [B, T, H_kv, TOP_K] layout
        keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot2 < num_candidates)
        parent_idx_f = tl.load(
            prev_selected_parents + prev_sel_base + (slot2 // COMPRESSION_RATE).to(tl.int64),
            mask=keep, other=-1,
        ).to(tl.int32)
        node = tl.where(keep & (parent_idx_f >= 0),
                        parent_idx_f * COMPRESSION_RATE + slot2 % COMPRESSION_RATE, -1)
        ok = (node >= 0).to(tl.int32)
        row_cnt = tl.sum(ok, axis=1)
        rk = (tl.cumsum(row_cnt, axis=0) - row_cnt)[:, None] + tl.cumsum(ok, axis=1) - ok
        tl.store(next_selected_parents + out_par_base + rk.to(tl.int64), node,
                 mask=(ok > 0) & (rk < TOP_K))
    else:
        # ---- step C: single bit-bisection rank select over scratch ----
        tau = tl.zeros([1], dtype=tl.int32)
        for step in tl.static_range(32):
            if step == 0:
                cand_t = tau | SIGN
            else:
                cand_t = tau | (1 << (31 - step))
            thresh = cand_t ^ SIGN
            cnt = tl.zeros([1], dtype=tl.int32)
            for ti in tl.static_range(NTILE):
                slot = (ti * BC + o_bc).to(tl.int32)
                vals = tl.load(scratch_imp + sbase + (ti * BC) + o_bc64,
                               mask=slot < num_candidates, other=NEG_INF)
                vi = vals.to(tl.int32, bitcast=True)
                enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
                enc = (vi & ~idx_mask) | enc_idx
                us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
                cnt = cnt + tl.sum((us >= thresh).to(tl.int32))
            tau = tl.where(cnt >= TOP_K, cand_t, tau)
        taus = tau ^ SIGN
        tl.store(scratch_tau + tau_base + tl.arange(0, 1).to(tl.int64), taus)

        # ---- step D: emit selected (ascending slots -> ascending nodes) ----
        out_par_base = (
            i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
            + i_t_loc.to(tl.int64) * H_kv * TOP_K
            + i_h_kv.to(tl.int64) * TOP_K
        )
        rank_off = tl.zeros([1], dtype=tl.int32)
        for ti in tl.static_range(NTILE):
            slot = (ti * BC + o_bc).to(tl.int32)
            vals = tl.load(scratch_imp + sbase + (ti * BC) + o_bc64,
                           mask=slot < num_candidates, other=NEG_INF)
            vi = vals.to(tl.int32, bitcast=True)
            enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
            enc = (vi & ~idx_mask) | enc_idx
            us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
            keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot < num_candidates)

            parent_off_f = (slot // COMPRESSION_RATE).to(tl.int64)
            child_slot_f = slot % COMPRESSION_RATE
            parent_idx_f = tl.load(
                prev_selected_parents + prev_sel_base + parent_off_f,
                mask=keep, other=-1,
            ).to(tl.int32)
            node = tl.where(keep & (parent_idx_f >= 0),
                            parent_idx_f * COMPRESSION_RATE + child_slot_f, -1)
            ok = node >= 0
            rk = rank_off + tl.cumsum(ok.to(tl.int32), axis=0) - ok.to(tl.int32)
            tl.store(next_selected_parents + out_par_base + rk.to(tl.int64), node,
                     mask=ok & (rk < TOP_K))
            rank_off = rank_off + tl.sum(ok.to(tl.int32))


@triton.jit
def recsa_mid_accumulate_kernel(
    q,  # [B, T, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    layer_max, layer_sum, layer_output,
    scratch_imp,  # final-scale importances from M-select
    scratch_tau,  # us-domain thresholds from M-select
    t_offset,
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    NO_SUMMARIES: tl.constexpr = 0,  # summaries="none": accumulate nothing
    DOT_FP16: tl.constexpr = 0,
):
    """Mid-layer accumulation of the dropped candidates; selected ones are excluded via the M-select threshold."""
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE

    pid_local = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_t_loc = t_offset + pid_local
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base + i_t_loc
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    o_v = tl.arange(0, V).to(tl.int64)
    idx_mask = (1 << LOG_N) - 1

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    n_prog_t = tl.num_programs(0)
    sbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_local) * MAX_CAND
    taus = tl.load(scratch_tau + pid_bh_kv.to(tl.int64) * n_prog_t + pid_local)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t_loc.to(tl.int64) * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V + i_t_loc.to(tl.int64) * H * V

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    if NO_SUMMARIES:
        num_tiles = 0
    for tile_idx in range(num_tiles):
        tile_base = (tile_idx * TILE_P).to(tl.int32)
        p_offs = tile_base + parent_in_tile

        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(
            prev_selected_parents + prev_sel_base + safe_p_offs
        ).to(tl.int32)

        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)

        # exclude SELECTED candidates (they expand at the next level)
        slot = tile_base * COMPRESSION_RATE + o_n
        vals = tl.load(scratch_imp + sbase + slot.to(tl.int64),
                       mask=slot < num_candidates, other=NEG_INF)
        vi = vals.to(tl.int32, bitcast=True)
        enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
        enc = (vi & ~idx_mask) | enc_idx
        us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
        selected = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot < num_candidates)
        valid = valid & ~selected

        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        base_pos = tile_base.to(tl.int64) * COMPRESSION_RATE
        cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
        sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
        k1_lo = k_lo * cos_off - k_hi * sin_off
        k1_hi = k_lo * sin_off + k_hi * cos_off
        k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]
        k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]

        if DOT_FP16 == 1:
            scores = (tl.dot(q_rope_lo.to(tl.float16), tl.trans(k_rope_lo.to(tl.float16)))
                      + tl.dot(q_rope_hi.to(tl.float16), tl.trans(k_rope_hi.to(tl.float16))))
        elif DOT_FP16 == 2:
            scores = (tl.dot(q_rope_lo, tl.trans(k_rope_lo), input_precision="ieee")
                      + tl.dot(q_rope_hi, tl.trans(k_rope_hi), input_precision="ieee"))
        else:
            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        if HAS_B:
            b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_rows * H_kv + i_h_kv.to(tl.int64))
            scores = scores + b_vals[None, :]
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG_INF)

        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        if DOT_FP16 == 1:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float16)  # PV runs fp16 (probs cast to v_tile.dtype)
        else:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

        tile_m = tl.max(scores, axis=1)
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

        if DOT_FP16 == 2:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile, input_precision="ieee")
        else:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    tl.store(layer_max + state_base + head_ids, cur_max, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, cur_sum, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, cur_output, mask=g_valid[:, None])


@triton.jit
def recsa_dsplit_select_score_kernel(
    q,  # [B, 1, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    raw_imp,
    batch_m_out,  # [B*H_kv * CR] fp32: per-batch maxima
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    FRONTIER: tl.constexpr = 0,
    pid_x=None,
    pid_y=None,
    EXT_PID: tl.constexpr = 0,
):
    """Step A of M-select for one candidate batch per program: per-batch importances and the batch max."""
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE

    if EXT_PID:
        i_batch = pid_x
        pid_bh_kv = pid_y
    else:
        i_batch = tl.program_id(0)
        pid_bh_kv = tl.program_id(1)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base  # Tq == 1, chunk-local row 0
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    if FRONTIER:
        f_lanes = tl.arange(0, TOP_K).to(tl.int32)
        f_nvp = rightmost_parent_idx + 1
        parent_list = tl.where(f_lanes < f_nvp, f_lanes, -1)
        if i_batch == 0:
            tl.store(prev_selected_parents + prev_sel_base + f_lanes, parent_list)
    else:
        parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    if i_batch < num_batches:
        g_ids = tl.arange(0, G_PAD).to(tl.int64)
        g_valid = g_ids < NUM_GROUPS
        head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

        num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
        rope_pos_q = num_candidates - 1

        o_k_half = tl.arange(0, K // 2).to(tl.int64)
        q_bt_base = i_b.to(tl.int64) * T_i64 * H * K
        q_ptrs = q + q_bt_base + head_ids[:, None] * K
        q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
        q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

        cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
        sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
        q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
        q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

        k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K

        TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
        PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
        o_n = tl.arange(0, TILE_N).to(tl.int32)
        parent_in_tile = o_n // COMPRESSION_RATE
        child_in_tile = o_n % COMPRESSION_RATE
        batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
        tile_rows = tl.arange(0, TILE_P).to(tl.int32)
        o_bc = tl.arange(0, BC).to(tl.int32)
        o_bc64 = o_bc.to(tl.int64)

        # ---- step A: batch scoring, as in M-select ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            if FRONTIER:
                parent_idx = tl.where(safe_poff.to(tl.int32) < f_nvp,
                                      safe_poff.to(tl.int32), -1)
            else:
                parent_idx = tl.load(
                    prev_selected_parents + prev_sel_base + safe_poff
                ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_child_rows * H_kv + i_h_kv.to(tl.int64))
                scores = scores + b_vals[None, :]
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            for i_r in tl.static_range(TILE_P):
                row_off = (tile_base + i_r).to(tl.int32)
                row_mask_in_tile = (tile_rows == i_r)[:, None]
                row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                is_row = (batch_rows == row_off)[:, None]
                importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        importance_flat = tl.reshape(importance_matrix, [BC])
        rbase = pid_bh_kv.to(tl.int64) * MAX_CAND + (i_batch * BC).to(tl.int64)
        tl.store(raw_imp + rbase + o_bc64, importance_flat)
        tl.store(batch_m_out + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE
                 + i_batch + tl.arange(0, 1).to(tl.int64), batch_m)


@triton.jit
def recsa_dsplit_select_finalize_kernel(
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    next_selected_parents,  # [B, 1, H_kv, TOP_K]; host pre-filled with -1
    raw_imp,
    batch_m_in,  # [B*H_kv * CR] fp32 from the split scoring kernel
    scratch_imp,
    scratch_tau,  # [B*H_kv] int32 OUT: us-domain threshold
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H_kv: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    force_mask=None,
    HAS_FORCE: tl.constexpr = 0,
):
    """Replay the cross-batch rescale chain on the raw importances, then steps C/D of M-select."""
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    NTILE: tl.constexpr = MAX_CAND // BC

    pid_bh_kv = tl.program_id(0)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
    i_t = t_base
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH
    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rightmost_pos = (num_candidates - 1).to(tl.int32)

    idx_mask = (1 << LOG_N) - 1

    jv = tl.arange(0, COMPRESSION_RATE).to(tl.int32)
    bm = tl.load(batch_m_in + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE + jv.to(tl.int64),
                 mask=jv < num_batches, other=NEG_INF)

    slot = tl.arange(0, MAX_CAND).to(tl.int32)
    slot64 = slot.to(tl.int64)
    jb = slot // BC  # batch of each slot
    v = tl.load(raw_imp + pid_bh_kv.to(tl.int64) * MAX_CAND + slot64)

    running_M = tl.full([1], NEG_INF, dtype=tl.float32)
    for i in tl.static_range(COMPRESSION_RATE):
        alive = i < num_batches
        bm_i = tl.sum(tl.where(jv == i, bm, 0.0))
        new_M = tl.maximum(running_M, bm_i)
        rescale_prefix = tl.exp(running_M - new_M)
        rescale_batch = tl.exp(bm_i - new_M)

        v_old = tl.where(v >= 0, v * rescale_prefix, v)
        vb_old = v_old.to(tl.int32, bitcast=True) & ~idx_mask
        v_old = vb_old.to(tl.float32, bitcast=True)

        # batch i itself: valid_pos mask -> rescale -> rightmost force -> trunc
        v_new = tl.where(slot < num_candidates, v, NEG_INF)
        v_new = tl.where(v_new >= 0, v_new * rescale_batch, v_new)
        v_new = tl.where(slot == rightmost_pos, 1e6, v_new)
        vb_new = v_new.to(tl.int32, bitcast=True) & ~idx_mask
        v_new = vb_new.to(tl.float32, bitcast=True)

        upd = tl.where(jb == i, v_new, tl.where(jb < i, v_old, v))
        v = tl.where(alive, upd, v)
        running_M = tl.where(alive, new_M, running_M)

    sbase = pid_bh_kv.to(tl.int64) * MAX_CAND
    if HAS_FORCE:
        f_par_off = (slot // COMPRESSION_RATE).to(tl.int64)
        f_par = tl.load(prev_selected_parents + prev_sel_base + f_par_off).to(tl.int32)
        f_node = tl.maximum(f_par, 0) * COMPRESSION_RATE + (slot % COMPRESSION_RATE)
        f_ok = (f_par >= 0) & (slot < num_candidates)
        f_word = tl.load(force_mask + (f_node >> 5).to(tl.int64), mask=f_ok, other=0)
        f_hit = ((f_word >> (f_node & 31)) & 1) != 0
        f_force = f_hit & f_ok
        fv = tl.where(f_force, 1e6, v)
        fb = fv.to(tl.int32, bitcast=True) & ~idx_mask
        v = tl.where(f_force, fb.to(tl.float32, bitcast=True), v)
    tl.store(scratch_imp + sbase + slot64, v)

    tl.debug_barrier()

    # ---- step C: single bit-bisection rank select over scratch ----
    o_bc = tl.arange(0, BC).to(tl.int32)
    o_bc64 = o_bc.to(tl.int64)
    tau = tl.zeros([1], dtype=tl.int32)
    for step in tl.static_range(32):
        if step == 0:
            cand_t = tau | SIGN
        else:
            cand_t = tau | (1 << (31 - step))
        thresh = cand_t ^ SIGN
        cnt = tl.zeros([1], dtype=tl.int32)
        for ti in tl.static_range(NTILE):
            slot_c = (ti * BC + o_bc).to(tl.int32)
            vals = tl.load(scratch_imp + sbase + (ti * BC) + o_bc64,
                           mask=slot_c < num_candidates, other=NEG_INF)
            vi = vals.to(tl.int32, bitcast=True)
            enc_idx = tl.where(vals >= 0, ~slot_c, slot_c) & idx_mask
            enc = (vi & ~idx_mask) | enc_idx
            us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
            cnt = cnt + tl.sum((us >= thresh).to(tl.int32))
        tau = tl.where(cnt >= TOP_K, cand_t, tau)
    taus = tau ^ SIGN
    tl.store(scratch_tau + pid_bh_kv.to(tl.int64) + tl.arange(0, 1).to(tl.int64), taus)

    # ---- step D: emit selected (ascending slots -> ascending nodes) ----
    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    rank_off = tl.zeros([1], dtype=tl.int32)
    for ti in tl.static_range(NTILE):
        slot_c = (ti * BC + o_bc).to(tl.int32)
        vals = tl.load(scratch_imp + sbase + (ti * BC) + o_bc64,
                       mask=slot_c < num_candidates, other=NEG_INF)
        vi = vals.to(tl.int32, bitcast=True)
        enc_idx = tl.where(vals >= 0, ~slot_c, slot_c) & idx_mask
        enc = (vi & ~idx_mask) | enc_idx
        us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
        keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot_c < num_candidates)

        parent_off_f = (slot_c // COMPRESSION_RATE).to(tl.int64)
        child_slot_f = slot_c % COMPRESSION_RATE
        parent_idx_f = tl.load(
            prev_selected_parents + prev_sel_base + parent_off_f,
            mask=keep, other=-1,
        ).to(tl.int32)
        node = tl.where(keep & (parent_idx_f >= 0),
                        parent_idx_f * COMPRESSION_RATE + child_slot_f, -1)
        ok = node >= 0
        rk = rank_off + tl.cumsum(ok.to(tl.int32), axis=0) - ok.to(tl.int32)
        tl.store(next_selected_parents + out_par_base + rk.to(tl.int64), node,
                 mask=ok & (rk < TOP_K))
        rank_off = rank_off + tl.sum(ok.to(tl.int32))


@triton.jit
def recsa_dsplit_select_finalize_v2_kernel(
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    next_selected_parents,  # [B, 1, H_kv, TOP_K]; host pre-filled with -1
    raw_imp,
    batch_m_in,  # [B*H_kv * CR] fp32 from the split scoring kernel
    scratch_imp,
    scratch_tau,  # [B*H_kv] int32 OUT: us-domain threshold
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H_kv: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    force_mask=None,
    HAS_FORCE: tl.constexpr = 0,
    WRITE_TAIL: tl.constexpr = 0,
):
    """recsa_dsplit_select_finalize_kernel with steps C/D on the register-resident [MAX_CAND] vector."""
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE

    pid_bh_kv = tl.program_id(0)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
    i_t = t_base
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH
    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rightmost_pos = (num_candidates - 1).to(tl.int32)

    idx_mask = (1 << LOG_N) - 1

    slot = tl.arange(0, MAX_CAND).to(tl.int32)
    slot64 = slot.to(tl.int64)
    jb = slot // BC  # batch of each slot
    v = tl.load(raw_imp + pid_bh_kv.to(tl.int64) * MAX_CAND + slot64)
    bm_ptr = batch_m_in + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE + tl.arange(0, 1).to(tl.int64)

    # ---- replay; bm_i is a masked [1] load ----
    running_M = tl.full([1], NEG_INF, dtype=tl.float32)
    for i in tl.static_range(COMPRESSION_RATE):
        alive = i < num_batches
        bm_i = tl.load(bm_ptr + i, mask=alive, other=NEG_INF)
        new_M = tl.maximum(running_M, bm_i)
        rescale_prefix = tl.exp(running_M - new_M)
        rescale_batch = tl.exp(bm_i - new_M)

        # earlier batches: conditional multiply, then unconditional truncation
        v_old = tl.where(v >= 0, v * rescale_prefix, v)
        vb_old = v_old.to(tl.int32, bitcast=True) & ~idx_mask
        v_old = vb_old.to(tl.float32, bitcast=True)

        # batch i itself: valid_pos mask -> rescale -> rightmost force -> trunc
        v_new = tl.where(slot < num_candidates, v, NEG_INF)
        v_new = tl.where(v_new >= 0, v_new * rescale_batch, v_new)
        v_new = tl.where(slot == rightmost_pos, 1e6, v_new)
        vb_new = v_new.to(tl.int32, bitcast=True) & ~idx_mask
        v_new = vb_new.to(tl.float32, bitcast=True)

        upd = tl.where(jb == i, v_new, tl.where(jb < i, v_old, v))
        v = tl.where(alive, upd, v)
        running_M = tl.where(alive, new_M, running_M)

    sbase = pid_bh_kv.to(tl.int64) * MAX_CAND
    if HAS_FORCE:
        f_par_off = (slot // COMPRESSION_RATE).to(tl.int64)
        f_par = tl.load(prev_selected_parents + prev_sel_base + f_par_off).to(tl.int32)
        f_node = tl.maximum(f_par, 0) * COMPRESSION_RATE + (slot % COMPRESSION_RATE)
        f_ok = (f_par >= 0) & (slot < num_candidates)
        f_word = tl.load(force_mask + (f_node >> 5).to(tl.int64), mask=f_ok, other=0)
        f_hit = ((f_word >> (f_node & 31)) & 1) != 0
        f_force = f_hit & f_ok
        fv = tl.where(f_force, 1e6, v)
        fb = fv.to(tl.int32, bitcast=True) & ~idx_mask
        v = tl.where(f_force, fb.to(tl.float32, bitcast=True), v)
    tl.store(scratch_imp + sbase + slot64, v)

    # ---- step C: bit-bisection rank select on the register tile ----
    valid_c = slot < num_candidates
    vals = tl.where(valid_c, v, NEG_INF)
    vi = vals.to(tl.int32, bitcast=True)
    enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
    enc = (vi & ~idx_mask) | enc_idx
    us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
    tau = tl.zeros([1], dtype=tl.int32)
    for step in tl.static_range(32):
        if step == 0:
            cand_t = tau | SIGN
        else:
            cand_t = tau | (1 << (31 - step))
        thresh = cand_t ^ SIGN
        cnt = tl.sum((us >= thresh).to(tl.int32))
        tau = tl.where(cnt >= TOP_K, cand_t, tau)
    taus = tau ^ SIGN
    tl.store(scratch_tau + pid_bh_kv.to(tl.int64) + tl.arange(0, 1).to(tl.int64), taus)

    # ---- step D: emit selected; rank = kept slots before it (slot order) ----
    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    keep = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & valid_c
    parent_off_f = (slot // COMPRESSION_RATE).to(tl.int64)
    child_slot_f = slot % COMPRESSION_RATE
    parent_idx_f = tl.load(
        prev_selected_parents + prev_sel_base + parent_off_f,
        mask=keep, other=-1,
    ).to(tl.int32)
    node = tl.where(keep & (parent_idx_f >= 0),
                    parent_idx_f * COMPRESSION_RATE + child_slot_f, -1)
    ok = node >= 0
    oki = ok.to(tl.int32)
    rk = tl.cumsum(oki, axis=0) - oki
    tl.store(next_selected_parents + out_par_base + rk.to(tl.int64), node,
             mask=ok & (rk < TOP_K))
    if WRITE_TAIL:
        n_out = tl.sum(oki)
        t_lanes = tl.arange(0, TOP_K).to(tl.int32)
        tl.store(next_selected_parents + out_par_base + t_lanes.to(tl.int64),
                 tl.full([TOP_K], -1, dtype=tl.int32), mask=t_lanes >= n_out)


def _dsplit_select_finalize(mode, grid, *args, **kwargs):
    """Launch the decode split's finalize kernel chosen by mode = RECSA_DSPLIT_FINALIZE."""
    if mode == "legacy":
        return recsa_dsplit_select_finalize_kernel[grid](*args, num_warps=8, **kwargs)
    if mode == "v2":
        return recsa_dsplit_select_finalize_v2_kernel[grid](*args, num_warps=8, **kwargs)
    return _dsplit_select_finalize_v3(mode, grid, *args, **kwargs)  # the default; any other mode raises there


def _dsplit_merge_level(part, g_in, g_out, final_out, *, B, Tq, H, H_kv,
                        NUM_GROUPS, V, G_PAD, fresh, fuse_norm):
    """Launch recsa_dsplit_merge_level_kernel for one decode level: part = slice partials (pmax, psum, pout), g_in /
    g_out = global (max, sum, output) accumulators, final_out = model-dtype output (bottom level only)."""
    return recsa_dsplit_merge_level_kernel[(B * H_kv,)](
        part[0], part[1], part[2],
        g_in[0], g_in[1], g_in[2],
        g_out[0], g_out[1], g_out[2],
        final_out if fuse_norm else g_out[2],
        0,
        T=Tq, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, V=V,
        NEG_INF=RECSA_SCORE_NEG_INF, G_PAD=G_PAD,
        GR=triton.next_power_of_2(NUM_GROUPS),
        NSLICE=RECSA_DSPLIT_NSLICE,
        FRESH=1 if fresh else 0, FUSE_NORM=1 if fuse_norm else 0,
        num_warps=4,
    )


_DSPLIT_PAIR_WARPS = 8


def _dsplit_hfuse_levels(*, ws, q_c, layers_k, layers_v, cos_cache, sin_cache, out_chunk,
                         B, Tq, H, H_kv, K, V, NUM_GROUPS, G_PAD, TILE_P, num_layers,
                         compression_rate, top_k, max_top_nodes, LOG_N, scale, t_base,
                         t_dev, num_stages, level_bias, force_masks, staged,
                         no_summaries=False, summary_norm=None):
    """Run the fused decode chain's levels as a horizontally fused schedule (RECSA_DSPLIT_HFUSE=on)."""
    device = q_c.device
    cursor = 1 if t_dev is not None else 0
    NS = RECSA_DSPLIT_NSLICE
    NBH = B * H_kv
    CR = compression_rate
    f32 = dict(dtype=torch.float32, device=device)
    w2b = ws.get("w2b")
    if w2b is None:
        w2b = ws["w2b"] = dict(
            raw=torch.empty(B * H_kv * max_top_nodes, **f32),
            bm=torch.empty(B * H_kv * compression_rate, **f32),
            pmax=torch.empty(NS * B * H_kv * G_PAD, **f32),
            psum=torch.empty(NS * B * H_kv * G_PAD, **f32),
            pout=torch.empty(NS * B * H_kv * G_PAD * V, **f32),
        )
    for li in range(num_layers):
        if f"pmax_l{li}" not in w2b:
            w2b[f"pmax_l{li}"] = torch.empty(NS * B * H_kv * G_PAD, **f32)
            w2b[f"psum_l{li}"] = torch.empty(NS * B * H_kv * G_PAD, **f32)
            w2b[f"pout_l{li}"] = torch.empty(NS * B * H_kv * G_PAD * V, **f32)
    part = [(w2b[f"pmax_l{li}"], w2b[f"psum_l{li}"], w2b[f"pout_l{li}"]) for li in range(num_layers)]
    imp_elems = B * H_kv * max_top_nodes
    mid_imp, mid_tau = ws["mid_imp"], ws["mid_tau"]
    if mid_imp is None or mid_imp.numel() < imp_elems:
        mid_imp = ws["mid_imp"] = torch.empty(imp_elems, **f32)
        mid_tau = ws["mid_tau"] = torch.empty(B * H_kv, dtype=torch.int32, device=device)
    top = num_layers - 1
    # level l reads prev_of[l]; its finalize writes next_l == prev_of[l - 1]
    prev_of = [ws["prev"] if (top - li) % 2 == 0 else ws["next"] for li in range(num_layers)]
    gsets = [(ws["global_max"], ws["global_sum"], ws["global_output"]),
             (ws["global_max_b"], ws["global_sum_b"], ws["global_output_b"])]
    g_cur = 0
    v2 = RECSA_DSPLIT_FINALIZE in ("v2", "v3")  # the finalize writes the -1 tail (WRITE_TAIL)
    sv2 = (_SV2.RECSA_DSPLIT_SCORE == "v2" and top_k % CR == 0 and (top_k // CR) % TILE_P == 0)
    sv2_role = 1 if (sv2 and _SV2.RECSA_DSPLIT_SCORE_ROLE == "noscat") else 0

    def lb(li):
        return level_bias[li] if level_bias is not None else None
    common = dict(H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, K=K, V=V, COMPRESSION_RATE=CR, TOP_K=top_k,
                  LOG_N=LOG_N, SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                  NEG_INF=RECSA_SCORE_NEG_INF, G_PAD=G_PAD, TILE_P=TILE_P, NSLICE=NS, HAS_CURSOR=cursor)

    def finalize(li):
        fm = force_masks.get(li) if force_masks is not None else None
        if not v2:
            prev_of[li - 1].fill_(-1)
        _dsplit_select_finalize(
            RECSA_DSPLIT_FINALIZE, (B * H_kv,),
            prev_of[li], prev_of[li - 1], w2b["raw"], w2b["bm"], mid_imp, mid_tau,
            t_base=t_base, layer_power=CR ** li, B=B, T=Tq, H_kv=H_kv,
            COMPRESSION_RATE=CR, TOP_K=top_k, LOG_N=LOG_N,
            SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD, NEG_INF=RECSA_SCORE_NEG_INF,
            t_ptr=t_dev, HAS_CURSOR=cursor,
            force_mask=fm, HAS_FORCE=1 if fm is not None else 0,
            **({"WRITE_TAIL": 1} if v2 else {}))
        if _DSPLIT_DEBUG_CAPTURE is not None:
            _DSPLIT_DEBUG_CAPTURE.append((
                li, mid_imp[:B * H_kv * max_top_nodes].clone(), mid_tau[:B * H_kv].clone(),
                prev_of[li - 1].clone()))

    def hfuse(sc=None, bot=False, acc=None, mg=None):
        nonlocal g_cur
        n_sc = CR * NBH if sc is not None else 0
        n_bot = NS * NBH if bot else 0
        n_acc = NS * NBH if acc is not None else 0
        n_mg = NBH if mg is not None else 0
        sc_end = n_sc
        bot_end = sc_end + n_bot
        acc_end = bot_end + n_acc
        if mg is not None:
            fresh = mg == top
            g_in, g_out = gsets[g_cur], gsets[1 - g_cur]
            g_cur = 1 - g_cur
        else:
            fresh, g_in, g_out = False, (None,) * 3, (None,) * 3
        s_lb = lb(sc) if sc is not None else None
        a_lb = lb(acc) if acc is not None else None
        recsa_dsplit_hfuse_kernel[(acc_end + n_mg,)](
            q_c, cos_cache, sin_cache, t_base, B, Tq, NBH, scale, t_dev, sc_end, bot_end, acc_end,
            **common,
            s_layer_k=layers_k[sc] if sc is not None else None,
            s_prev=prev_of[sc] if sc is not None else None,
            s_raw_imp=w2b["raw"] if sc is not None else None,
            s_batch_m=w2b["bm"] if sc is not None else None,
            s_N_layer=layers_k[sc].shape[1] if sc is not None else 0,
            s_b_bias=s_lb, HAS_SC=1 if sc is not None else 0, S_LP=CR ** sc if sc is not None else 1,
            S_HAS_B=1 if s_lb is not None else 0,
            b_layer_k=layers_k[0] if bot else None, b_layer_v=layers_v[0] if bot else None,
            b_prev=prev_of[0] if bot else None,
            b_part_max=part[0][0] if bot else None, b_part_sum=part[0][1] if bot else None,
            b_part_out=part[0][2] if bot else None,
            b_N_layer=layers_k[0].shape[1] if bot else 0,
            fetch_k=ws["fetch_k"] if (bot and staged) else None,
            fetch_v=ws["fetch_v"] if (bot and staged) else None,
            HAS_BOT=1 if bot else 0, LEAF_STAGED=1 if (bot and staged) else 0,
            a_layer_k=layers_k[acc] if acc is not None else None,
            a_layer_v=layers_v[acc] if acc is not None else None,
            a_prev=prev_of[acc] if acc is not None else None,
            a_part_max=part[acc][0] if acc is not None else None,
            a_part_sum=part[acc][1] if acc is not None else None,
            a_part_out=part[acc][2] if acc is not None else None,
            a_scratch_imp=mid_imp if acc is not None else None,
            a_scratch_tau=mid_tau if acc is not None else None,
            a_N_layer=layers_k[acc].shape[1] if acc is not None else 0,
            a_b_bias=a_lb, HAS_ACC=1 if acc is not None else 0,
            A_LP=CR ** acc if acc is not None else 1, A_HAS_B=1 if a_lb is not None else 0,
            m_part_max=part[mg][0] if mg is not None else None,
            m_part_sum=part[mg][1] if mg is not None else None,
            m_part_out=part[mg][2] if mg is not None else None,
            m_g_max=g_in[0], m_g_sum=g_in[1], m_g_out=g_in[2],
            m_out_max=g_out[0], m_out_sum=g_out[1], m_out_out=g_out[2],
            anchor=0, HAS_MG=1 if mg is not None else 0, M_FRESH=1 if fresh else 0,
            GR=triton.next_power_of_2(NUM_GROUPS),
            A_NO_SUM=1 if (no_summaries and acc is not None) else 0,
            S_V2=sv2_role if sc is not None else 0,
            num_warps=4, num_stages=num_stages,
        )
        if summary_norm is not None and acc is not None: summary_norm.hfuse_sites(acc, bot, prev_of, part, gsets[g_cur] if mg is not None else None)  # noqa: E701,E501

    s_lb = lb(top)
    if sv2:
        _SV2.launch_score_v2(
            _SV2.score_v2_sync(ws, NBH, CR, (top_k // CR) // TILE_P, device),
            q_c, layers_k[top], prev_of[top], cos_cache, sin_cache, w2b["raw"], w2b["bm"],
            t_base=t_base, b_bias=s_lb, HAS_B=1 if s_lb is not None else 0,
            layer_power=CR ** top, B=B, T=Tq, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, K=K,
            N_layer=layers_k[top].shape[1], COMPRESSION_RATE=CR, TOP_K=top_k,
            NEG_INF=RECSA_SCORE_NEG_INF, G_PAD=G_PAD, TILE_P=TILE_P, scale=scale,
            t_ptr=t_dev, HAS_CURSOR=cursor, FRONTIER=1, one_tile=True, num_stages=num_stages)
    else:
        recsa_dsplit_select_score_kernel[(CR, B * H_kv)](
            q_c, layers_k[top], prev_of[top], cos_cache, sin_cache, w2b["raw"], w2b["bm"],
            t_base=t_base, b_bias=s_lb, HAS_B=1 if s_lb is not None else 0,
            layer_power=CR ** top, B=B, T=Tq, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, K=K,
            N_layer=layers_k[top].shape[1], COMPRESSION_RATE=CR, TOP_K=top_k,
            NEG_INF=RECSA_SCORE_NEG_INF, G_PAD=G_PAD, TILE_P=TILE_P, scale=scale,
            t_ptr=t_dev, HAS_CURSOR=cursor, FRONTIER=1, num_warps=4, num_stages=num_stages)
    finalize(top)
    for li in range(top - 1, 0, -1):
        hfuse(sc=li, acc=li + 1, mg=li + 2 if li + 2 <= top else None)
        finalize(li)
    if staged:
        if _leaf_mirror_of(layers_k[0]) is not None: _leaf_mirror_gather(layers_k[0], layers_v[0], ws["fetch_k"], ws["fetch_v"], prev_of[0], t_base, H_kv, K, V, CR, top_k, t_dev, cursor)  # noqa: E701,E501
        else: recsa_leaf_gather_kernel[(RECSA_GATHER_PROGS,)](  # noqa: E701
            layers_k[0], layers_v[0], ws["fetch_k"], ws["fetch_v"], prev_of[0],
            t_base=t_base, H_kv=H_kv, K=K, V=V, COMPRESSION_RATE=CR, TOP_K=top_k,
            NPROG=RECSA_GATHER_PROGS, t_ptr=t_dev, HAS_CURSOR=cursor, num_warps=4)
    hfuse(bot=True, acc=1, mg=2 if top >= 2 else None)
    g_in, g_out = gsets[g_cur], gsets[1 - g_cur]
    recsa_dsplit_merge_pair_kernel[(B * H_kv,)](
        part[1][0], part[1][1], part[1][2], part[0][0], part[0][1], part[0][2],
        g_in[0], g_in[1], g_in[2], g_out[0], g_out[1], g_out[2], out_chunk, 0,
        T=Tq, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, V=V, NEG_INF=RECSA_SCORE_NEG_INF,
        G_PAD=G_PAD, GR=triton.next_power_of_2(NUM_GROUPS), NSLICE=NS,
        FRESH=1 if top == 1 else 0, num_warps=_DSPLIT_PAIR_WARPS)
    return mid_imp, mid_tau


@triton.jit
def recsa_dsplit_accumulate_part_kernel(
    q,  # [B, 1, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    part_max,  # [NSLICE * B*H_kv * G_PAD] fp32 OUT
    part_sum,  # [NSLICE * B*H_kv * G_PAD] fp32 OUT
    part_out,  # [NSLICE * B*H_kv * G_PAD * V] fp32 OUT
    scratch_imp,
    scratch_tau,
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    NSLICE: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    pid_x=None,
    pid_y=None,
    nbh=None,  # num_programs(1) == B * H_kv
    EXT_PID: tl.constexpr = 0,
    NO_SUMMARIES: tl.constexpr = 0,  # summaries="none": accumulate nothing
):
    """M-accum tile loop split over NSLICE programs: each accumulates a contiguous run of parent tiles (dropped
    candidates only) into a partial (max, sum, out) triple."""
    SIGN: tl.constexpr = -2147483648
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE

    if EXT_PID:
        pid_slice = pid_x
        pid_bh_kv = pid_y
        NBH = nbh
    else:
        pid_slice = tl.program_id(0)
        pid_bh_kv = tl.program_id(1)
        NBH = tl.num_programs(1)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    o_v = tl.arange(0, V).to(tl.int64)
    idx_mask = (1 << LOG_N) - 1

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    sbase = pid_bh_kv.to(tl.int64) * MAX_CAND
    taus = tl.load(scratch_tau + pid_bh_kv.to(tl.int64))

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    tiles_per_slice = (num_tiles + NSLICE - 1) // NSLICE
    tile_lo = pid_slice * tiles_per_slice
    tile_hi = tl.minimum(tile_lo + tiles_per_slice, num_tiles)
    if NO_SUMMARIES:
        tile_hi = tile_lo
    for tile_idx in range(tile_lo, tile_hi):
        tile_base = (tile_idx * TILE_P).to(tl.int32)
        p_offs = tile_base + parent_in_tile

        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(
            prev_selected_parents + prev_sel_base + safe_p_offs
        ).to(tl.int32)

        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)

        # exclude SELECTED candidates (they expand at the next level)
        slot = tile_base * COMPRESSION_RATE + o_n
        vals = tl.load(scratch_imp + sbase + slot.to(tl.int64),
                       mask=slot < num_candidates, other=NEG_INF)
        vi = vals.to(tl.int32, bitcast=True)
        enc_idx = tl.where(vals >= 0, ~slot, slot) & idx_mask
        enc = (vi & ~idx_mask) | enc_idx
        us = (enc ^ ((enc >> 31) | SIGN)) ^ SIGN
        selected = (us >= taus) & (vals > SCORE_VALID_THRESHOLD) & (slot < num_candidates)
        valid = valid & ~selected

        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        base_pos = tile_base.to(tl.int64) * COMPRESSION_RATE
        cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
        sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
        k1_lo = k_lo * cos_off - k_hi * sin_off
        k1_hi = k_lo * sin_off + k_hi * cos_off
        k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]
        k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]

        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        if HAS_B:
            b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_rows * H_kv + i_h_kv.to(tl.int64))
            scores = scores + b_vals[None, :]
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG_INF)

        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

        tile_m = tl.max(scores, axis=1)
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

        cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    pbase = pid_slice.to(tl.int64) * NBH + pid_bh_kv
    g_off = tl.arange(0, G_PAD).to(tl.int64)
    tl.store(part_max + pbase * G_PAD + g_off, cur_max)
    tl.store(part_sum + pbase * G_PAD + g_off, cur_sum)
    tl.store(part_out + pbase * (G_PAD * V) + g_off[:, None] * V + o_v[None, :],
             cur_output)


@triton.jit
def recsa_dsplit_bottom_part_kernel(
    q,  # [B, 1, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    layer_v,  # [B, N_layer, H_kv, V]
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    part_max,  # [NSLICE * B*H_kv * G_PAD] fp32 OUT
    part_sum,  # [NSLICE * B*H_kv * G_PAD] fp32 OUT
    part_out,  # [NSLICE * B*H_kv * G_PAD * V] fp32 OUT
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    NSLICE: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    fetch_k=None,  # [H_kv, TOP_K*CR, K] staged leaf tiles
    fetch_v=None,  # [H_kv, TOP_K*CR, V]
    LEAF_STAGED: tl.constexpr = 0,
    pid_x=None,
    pid_y=None,
    nbh=None,  # num_programs(1) == B * H_kv
    EXT_PID: tl.constexpr = 0,
):
    """Bottom accumulate tile loop split over NSLICE programs (partial triple out, merged by the split merge kernel)."""
    if EXT_PID:
        pid_slice = pid_x
        pid_bh_kv = pid_y
        NBH = nbh
    else:
        pid_slice = tl.program_id(0)
        pid_bh_kv = tl.program_id(1)
        NBH = tl.num_programs(1)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1
    i_t = t_base
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    cur_max = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    if LEAF_STAGED:
        k_base = fetch_k + i_h_kv.to(tl.int64) * (TOP_K * COMPRESSION_RATE * K)
        v_base = fetch_v + i_h_kv.to(tl.int64) * (TOP_K * COMPRESSION_RATE * V)
    else:
        k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
        v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    o_v = tl.arange(0, V).to(tl.int64)

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    tiles_per_slice = (num_tiles + NSLICE - 1) // NSLICE
    tile_lo = pid_slice * tiles_per_slice
    tile_hi = tl.minimum(tile_lo + tiles_per_slice, num_tiles)
    for tile_idx in range(tile_lo, tile_hi):
        tile_base = (tile_idx * TILE_P).to(tl.int32)
        p_offs = tile_base + parent_in_tile

        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(
            prev_selected_parents + prev_sel_base + safe_p_offs
        ).to(tl.int32)

        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)

        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        if LEAF_STAGED:
            slot = (safe_p_offs * COMPRESSION_RATE
                    + child_in_tile.to(tl.int64))

        if LEAF_STAGED:
            k_row = k_base + slot[:, None] * K
        else:
            k_row = k_base + safe_rows[:, None] * (H_kv * K)
        k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
        k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

        base_pos = tile_base.to(tl.int64) * COMPRESSION_RATE
        cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
        sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
        k1_lo = k_lo * cos_off - k_hi * sin_off
        k1_hi = k_lo * sin_off + k_hi * cos_off
        k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]
        k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]

        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG_INF)

        if LEAF_STAGED:
            v_row = v_base + slot[:, None] * V
        else:
            v_row = v_base + safe_rows[:, None] * (H_kv * V)
        v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

        tile_m = tl.max(scores, axis=1)
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)

        cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    pbase = pid_slice.to(tl.int64) * NBH + pid_bh_kv
    g_off = tl.arange(0, G_PAD).to(tl.int64)
    tl.store(part_max + pbase * G_PAD + g_off, cur_max)
    tl.store(part_sum + pbase * G_PAD + g_off, cur_sum)
    tl.store(part_out + pbase * (G_PAD * V) + g_off[:, None] * V + o_v[None, :],
             cur_output)


@triton.jit
def recsa_dsplit_partial_merge_kernel(
    part_max,  # [NSLICE * B*H_kv * G_PAD] fp32
    part_sum,  # [NSLICE * B*H_kv * G_PAD] fp32
    part_out,  # [NSLICE * B*H_kv * G_PAD * V] fp32
    layer_max,  # [B, 1, H] OUT
    layer_sum,  # [B, 1, H] OUT
    layer_output,  # [B, 1, H, V] OUT
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    V: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    NSLICE: tl.constexpr,
):
    """LSE-merge the NSLICE partial triples in fixed slice order into the level buffers."""
    pid_bh_kv = tl.program_id(0)
    NBH = tl.num_programs(0)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv
    T_i64 = T

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_v = tl.arange(0, V).to(tl.int64)

    m = tl.full([G_PAD], NEG_INF, dtype=tl.float32)
    l = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, V], dtype=tl.float32)
    for s in tl.static_range(NSLICE):
        pbase = (s * NBH + pid_bh_kv).to(tl.int64)
        pm = tl.load(part_max + pbase * G_PAD + g_ids)
        pl = tl.load(part_sum + pbase * G_PAD + g_ids)
        pacc = tl.load(part_out + pbase * (G_PAD * V) + g_ids[:, None] * V + o_v[None, :])
        new_m = tl.maximum(m, pm)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(pm - new_m)
        l = l * alpha + pl * beta
        acc = acc * alpha[:, None] + pacc * beta[:, None]
        m = new_m

    state_base = i_b.to(tl.int64) * T_i64 * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V
    tl.store(layer_max + state_base + head_ids, m, mask=g_valid)
    tl.store(layer_sum + state_base + head_ids, l, mask=g_valid)
    out_ptrs = layer_output + out_base + head_ids[:, None] * V + o_v[None, :]
    tl.store(out_ptrs, acc, mask=g_valid[:, None])


@triton.jit
def recsa_dsplit_merge_level_kernel(
    part_max,
    part_sum,  # [NSLICE * B*H_kv * G_PAD] fp32
    part_out,  # [NSLICE * B*H_kv * G_PAD * V] fp32
    global_max,  # [B, 1, H] IN (never read when FRESH)
    global_sum,  # [B, 1, H] IN
    global_output,  # [B, 1, H, V] IN
    out_max,  # [B, 1, H] OUT (must not alias global_max)
    out_sum,  # [B, 1, H] OUT
    out_output,  # [B, 1, H, V] OUT
    final_out,  # [B, 1, H, V] model dtype OUT (FUSE_NORM)
    anchor,  # runtime int, always 0 (see below)
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    V: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    GR: tl.constexpr,
    NSLICE: tl.constexpr,
    FRESH: tl.constexpr,  # global state is the init state (NEG_INF / 0 / 0)
    FUSE_NORM: tl.constexpr,  # also write final_out = out_output / out_sum
    part2_max=None,
    part2_sum=None,
    part2_out=None,
    TWO: tl.constexpr = 0,
    pid_x=None,
    nbh=None,  # num_programs(0) == B * H_kv
    EXT_PID: tl.constexpr = 0,
):
    """One decode level's recsa_dsplit_partial_merge_kernel -> recsa_merge_to_global_kernel (->
    recsa_final_normalize_kernel at the bottom level) in one launch."""
    if EXT_PID:
        pid_bh_kv = pid_x
        NBH = nbh
    else:
        pid_bh_kv = tl.program_id(0)
        NBH = tl.num_programs(0)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv
    T_i64 = T

    g_ids = tl.arange(0, GR).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_v = tl.arange(0, V).to(tl.int64)
    state_base = i_b.to(tl.int64) * T_i64 * H
    out_base = i_b.to(tl.int64) * T_i64 * H * V
    o_ptrs = out_base + head_ids[:, None] * V + o_v[None, :]

    # (1) every load first
    pms = ()
    pls = ()
    paccs = ()
    for s in tl.static_range(NSLICE):
        pbase = (s * NBH + pid_bh_kv).to(tl.int64)
        pms = pms + (tl.load(part_max + pbase * G_PAD + g_ids, mask=g_valid, other=NEG_INF),)
        pls = pls + (tl.load(part_sum + pbase * G_PAD + g_ids, mask=g_valid, other=0.0),)
        paccs = paccs + (tl.load(part_out + pbase * (G_PAD * V) + g_ids[:, None] * V + o_v[None, :],
                                 mask=g_valid[:, None], other=0.0),)
    if TWO:
        qms = ()
        qls = ()
        qaccs = ()
        for s in tl.static_range(NSLICE):
            pbase = (s * NBH + pid_bh_kv).to(tl.int64)
            qms = qms + (tl.load(part2_max + pbase * G_PAD + g_ids, mask=g_valid, other=NEG_INF),)
            qls = qls + (tl.load(part2_sum + pbase * G_PAD + g_ids, mask=g_valid, other=0.0),)
            qaccs = qaccs + (tl.load(part2_out + pbase * (G_PAD * V) + g_ids[:, None] * V + o_v[None, :],
                                     mask=g_valid[:, None], other=0.0),)
    if not FRESH:
        g_max = tl.load(global_max + state_base + head_ids, mask=g_valid, other=0.0)
        g_sum = tl.load(global_sum + state_base + head_ids, mask=g_valid, other=0.0)
        g_output_vals = tl.load(global_output + o_ptrs, mask=g_valid[:, None], other=0.0)

    # (2) data-dependence anchor (runtime anchor == 0 -> the constant initial values)
    ob1 = tl.zeros([GR], dtype=tl.int32)
    obv = tl.zeros([GR, V], dtype=tl.int32)
    for s in tl.static_range(NSLICE):
        ob1 = ob1 | pms[s].to(tl.int32, bitcast=True) | pls[s].to(tl.int32, bitcast=True)
        obv = obv | paccs[s].to(tl.int32, bitcast=True)
    if TWO:
        for s in tl.static_range(NSLICE):
            ob1 = ob1 | qms[s].to(tl.int32, bitcast=True) | qls[s].to(tl.int32, bitcast=True)
            obv = obv | qaccs[s].to(tl.int32, bitcast=True)
    if not FRESH:
        ob1 = ob1 | g_max.to(tl.int32, bitcast=True) | g_sum.to(tl.int32, bitcast=True)
        obv = obv | g_output_vals.to(tl.int32, bitcast=True)
    take = anchor != 0
    m = tl.where(take, ob1.to(tl.float32, bitcast=True), tl.full([GR], NEG_INF, dtype=tl.float32))
    l = tl.where(take, ob1.to(tl.float32, bitcast=True), tl.zeros([GR], dtype=tl.float32))
    acc = tl.where(take, obv.to(tl.float32, bitcast=True), tl.zeros([GR, V], dtype=tl.float32))

    # (3) recsa_dsplit_partial_merge_kernel's LSE chain, in slice order
    for s in tl.static_range(NSLICE):
        pm = pms[s]
        pl = pls[s]
        pacc = paccs[s]
        new_m = tl.maximum(m, pm)
        alpha = tl.exp(m - new_m)
        beta = tl.exp(pm - new_m)
        l = l * alpha + pl * beta
        acc = acc * alpha[:, None] + pacc * beta[:, None]
        m = new_m

    # (4) recsa_merge_to_global_kernel's expressions on the level triple
    cur_max = m
    cur_sum = l
    cur_output = acc
    cur_has_contribution = cur_sum > 1e-10
    if FRESH:
        g_max = tl.where(cur_has_contribution, cur_max, NEG_INF)
        g_sum = tl.where(cur_has_contribution, cur_sum, 0.0)
        g_output_vals = tl.where(cur_has_contribution[:, None], cur_output, 0.0)
    else:
        g_has_accumulation = g_sum > 1e-10
        new_max = tl.maximum(g_max, cur_max)
        scale_g = tl.where(g_has_accumulation, tl.exp(g_max - new_max), 0.0)
        scale_c = tl.where(cur_has_contribution, tl.exp(cur_max - new_max), 0.0)
        merged_sum = g_sum * scale_g + cur_sum * scale_c
        merged_output = g_output_vals * scale_g[:, None] + cur_output * scale_c[:, None]
        merged_max = new_max
        g_max = tl.where(
            cur_has_contribution,
            tl.where(g_has_accumulation, merged_max, cur_max),
            g_max
        )
        g_sum = tl.where(
            cur_has_contribution,
            tl.where(g_has_accumulation, merged_sum, cur_sum),
            g_sum
        )
        g_output_vals = tl.where(
            cur_has_contribution[:, None],
            tl.where(g_has_accumulation[:, None], merged_output, cur_output),
            g_output_vals
        )
    if TWO:
        m = tl.where(take, ob1.to(tl.float32, bitcast=True), tl.full([GR], NEG_INF, dtype=tl.float32))
        l = tl.where(take, ob1.to(tl.float32, bitcast=True), tl.zeros([GR], dtype=tl.float32))
        acc = tl.where(take, obv.to(tl.float32, bitcast=True), tl.zeros([GR, V], dtype=tl.float32))
        for s in tl.static_range(NSLICE):
            pm = qms[s]
            pl = qls[s]
            pacc = qaccs[s]
            new_m = tl.maximum(m, pm)
            alpha = tl.exp(m - new_m)
            beta = tl.exp(pm - new_m)
            l = l * alpha + pl * beta
            acc = acc * alpha[:, None] + pacc * beta[:, None]
            m = new_m
        cur_max = m
        cur_sum = l
        cur_output = acc
        cur_has_contribution = cur_sum > 1e-10
        g_has_accumulation = g_sum > 1e-10
        new_max = tl.maximum(g_max, cur_max)
        scale_g = tl.where(g_has_accumulation, tl.exp(g_max - new_max), 0.0)
        scale_c = tl.where(cur_has_contribution, tl.exp(cur_max - new_max), 0.0)
        merged_sum = g_sum * scale_g + cur_sum * scale_c
        merged_output = g_output_vals * scale_g[:, None] + cur_output * scale_c[:, None]
        merged_max = new_max
        g_max = tl.where(
            cur_has_contribution,
            tl.where(g_has_accumulation, merged_max, cur_max),
            g_max
        )
        g_sum = tl.where(
            cur_has_contribution,
            tl.where(g_has_accumulation, merged_sum, cur_sum),
            g_sum
        )
        g_output_vals = tl.where(
            cur_has_contribution[:, None],
            tl.where(g_has_accumulation[:, None], merged_output, cur_output),
            g_output_vals
        )
    tl.store(out_max + state_base + head_ids, g_max, mask=g_valid)
    tl.store(out_sum + state_base + head_ids, g_sum, mask=g_valid)
    tl.store(out_output + o_ptrs, g_output_vals, mask=g_valid[:, None])

    # (5) recsa_final_normalize_kernel at the bottom level
    if FUSE_NORM:
        normalized = g_output_vals / g_sum[:, None]
        tl.store(final_out + o_ptrs, normalized, mask=g_valid[:, None])


_dsplit_score_fn = recsa_dsplit_select_score_kernel
from recsa import dsplit_score_v2 as _SV2  # noqa: E402
_dsplit_score_v2_fn = _SV2.recsa_dsplit_score_v2_kernel
_dsplit_accumulate_fn = recsa_dsplit_accumulate_part_kernel
_dsplit_bottom_fn = recsa_dsplit_bottom_part_kernel
_dsplit_merge_level_fn = recsa_dsplit_merge_level_kernel


@triton.jit
def recsa_dsplit_merge_pair_kernel(
    part_max,
    part_sum,
    part_out,
    part2_max,
    part2_sum,
    part2_out,
    global_max,  # [B, 1, H] IN (never read when FRESH)
    global_sum,
    global_output,
    out_max,
    out_sum,
    out_output,
    final_out,
    anchor,  # runtime int, always 0
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    V: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    GR: tl.constexpr,
    NSLICE: tl.constexpr,
    FRESH: tl.constexpr,
):
    """The last launch of the horizontally fused decode chain (RECSA_DSPLIT_HFUSE): recsa_dsplit_merge_level_kernel for
    level l (FUSE_NORM=0) followed by recsa_dsplit_merge_level_kernel for the bottom level (FRESH=0, FUSE_NORM=1) in
    one launch."""
    _dsplit_merge_level_fn(
        part_max, part_sum, part_out, global_max, global_sum, global_output, out_max, out_sum, out_output,
        final_out, anchor, T, H, H_kv, NUM_GROUPS, V, NEG_INF, G_PAD, GR, NSLICE, FRESH, 1,
        part2_max, part2_sum, part2_out, 1)


@triton.jit
def recsa_dsplit_hfuse_kernel(
    q,
    cos_cache, sin_cache,  # [cache_size, K//2]
    t_base,
    B,
    T,
    NBH,  # B * H_kv
    scale,
    t_ptr,  # int64 [1] session cursor (cursor mode) or None
    sc_end,
    bot_end,  # [bot_end, acc_end) ACC, [acc_end, grid) MG
    acc_end,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    NSLICE: tl.constexpr,
    HAS_CURSOR: tl.constexpr,
    # role SC: recsa_dsplit_select_score_kernel (FRONTIER=0) for level ls
    s_layer_k, s_prev, s_raw_imp, s_batch_m, s_N_layer, s_b_bias,
    HAS_SC: tl.constexpr, S_LP: tl.constexpr, S_HAS_B: tl.constexpr,
    # role BOT: recsa_dsplit_bottom_part_kernel for level 0
    b_layer_k, b_layer_v, b_prev, b_part_max, b_part_sum, b_part_out, b_N_layer, fetch_k, fetch_v,
    HAS_BOT: tl.constexpr, LEAF_STAGED: tl.constexpr,
    # role ACC: recsa_dsplit_accumulate_part_kernel for level la
    a_layer_k, a_layer_v, a_prev, a_part_max, a_part_sum, a_part_out, a_scratch_imp, a_scratch_tau,
    a_N_layer, a_b_bias,
    HAS_ACC: tl.constexpr, A_LP: tl.constexpr, A_HAS_B: tl.constexpr,
    # role MG: recsa_dsplit_merge_level_kernel (FUSE_NORM=0) for level lm
    m_part_max, m_part_sum, m_part_out, m_g_max, m_g_sum, m_g_out, m_out_max, m_out_sum, m_out_out,
    anchor,
    HAS_MG: tl.constexpr, M_FRESH: tl.constexpr, GR: tl.constexpr,
    A_NO_SUM: tl.constexpr = 0,  # accumulate role: summaries="none" (NO_SUMMARIES)
    S_V2: tl.constexpr = 0,
):
    """Horizontal level fusion of the fused decode chain (RECSA_DSPLIT_HFUSE): kernels of different tree levels that do
    not depend on each other share one launch."""
    pid = tl.program_id(0)
    if HAS_SC:
        if pid < sc_end:
            if S_V2:
                _dsplit_score_v2_fn(
                    q, s_layer_k, s_prev, cos_cache, sin_cache, s_raw_imp, s_batch_m, None, None, None, t_base,
                    NBH, S_LP, B, T, H, H_kv, NUM_GROUPS, K, s_N_layer, COMPRESSION_RATE, TOP_K, NEG_INF,
                    G_PAD, TILE_P, scale, t_ptr, HAS_CURSOR, s_b_bias, S_HAS_B, 0, 0,
                    pid % COMPRESSION_RATE, pid // COMPRESSION_RATE, 1)
            else:
                _dsplit_score_fn(
                    q, s_layer_k, s_prev, cos_cache, sin_cache, s_raw_imp, s_batch_m, t_base,
                    S_LP, B, T, H, H_kv, NUM_GROUPS, K, s_N_layer, COMPRESSION_RATE, TOP_K, NEG_INF,
                    G_PAD, TILE_P, scale, t_ptr, HAS_CURSOR, s_b_bias, S_HAS_B, 0,
                    pid % COMPRESSION_RATE, pid // COMPRESSION_RATE, 1)
    if HAS_BOT:
        if (pid >= sc_end) & (pid < bot_end):
            b_loc = pid - sc_end
            _dsplit_bottom_fn(
                q, b_layer_k, b_layer_v, b_prev, cos_cache, sin_cache, b_part_max, b_part_sum, b_part_out,
                t_base, 1, B, T, H, H_kv, NUM_GROUPS, K, V, b_N_layer, COMPRESSION_RATE, TOP_K, NEG_INF,
                G_PAD, TILE_P, NSLICE, scale, t_ptr, HAS_CURSOR, fetch_k, fetch_v, LEAF_STAGED,
                b_loc % NSLICE, b_loc // NSLICE, NBH, 1)
    if HAS_ACC:
        if (pid >= bot_end) & (pid < acc_end):
            a_loc = pid - bot_end
            _dsplit_accumulate_fn(
                q, a_layer_k, a_layer_v, a_prev, cos_cache, sin_cache, a_part_max, a_part_sum, a_part_out,
                a_scratch_imp, a_scratch_tau, t_base, A_LP, B, T, H, H_kv, NUM_GROUPS, K, V, a_N_layer,
                COMPRESSION_RATE, TOP_K, LOG_N, SCORE_VALID_THRESHOLD, NEG_INF, G_PAD, TILE_P, NSLICE,
                scale, t_ptr, HAS_CURSOR, a_b_bias, A_HAS_B,
                a_loc % NSLICE, a_loc // NSLICE, NBH, 1, A_NO_SUM)
    if HAS_MG:
        if pid >= acc_end:
            _dsplit_merge_level_fn(
                m_part_max, m_part_sum, m_part_out, m_g_max, m_g_sum, m_g_out, m_out_max, m_out_sum,
                m_out_out, None, anchor, T, H, H_kv, NUM_GROUPS, V, NEG_INF, G_PAD, GR, NSLICE,
                M_FRESH, 0, None, None, None, 0,
                pid - acc_end, NBH, 1)


def _packed_top_params(num_groups, head_dim, compression_rate, top_k_per_layer,
                       num_layers, tile_p, packed_qb=None):
    """Shared eligibility/sizing rule for the packed top-layer path."""
    g_pad2 = triton.next_power_of_2(num_groups)
    if packed_qb is not None:
        qb_sel = int(packed_qb)
    else:
        qb_sel = max(1, min(16, 32 // g_pad2)) if g_pad2 <= 32 else 1
    qb_acc = max(1, min(16, 64 // g_pad2)) if g_pad2 <= 64 else 1
    supported = (
        num_layers >= 2
        and qb_sel * g_pad2 >= 16
        and qb_acc * g_pad2 >= 16
        and g_pad2 <= 16
        and head_dim >= 32
        and top_k_per_layer % compression_rate == 0
        and (top_k_per_layer // compression_rate) % tile_p == 0
    )
    return supported, qb_sel, qb_acc, g_pad2


@triton.jit
def recsa_top_select_kernel(
    q,                      # [B, T, H, K]
    layer_k,                # [B, N_layer, H_kv, K]
    cos_cache, sin_cache,   # [cache_size, K//2]
    next_selected_parents,
    selected_bitmask,
    scratch,                # [n_progs * QB * TOP_K] float32 workspace
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD2: tl.constexpr,
    QB: tl.constexpr,
    TILE_P: tl.constexpr,
    NWORDS,
    WRITE_BITMASK: tl.constexpr,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    force_mask=None,
    HAS_FORCE: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    SEL_IEEE: tl.constexpr = 0,
):
    """Packed-query Top-K selection for the top tree layer only."""
    SIGN: tl.constexpr = -2147483648  # int32 bit pattern 0x80000000
    M: tl.constexpr = QB * G_PAD2
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = PARENTS_PER_BATCH // TILE_P

    pid_t = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1

    T_i64 = T
    t0 = pid_t * QB
    q_ids = tl.arange(0, QB).to(tl.int32)
    t_vec = t0 + q_ids
    q_row_ok = t_vec < T
    t_q = tl.minimum(t_vec, T - 1)                   # CHUNK-LOCAL: addressing

    rightmost_idx = (t_base + t_q) // layer_power  # [QB] GLOBAL
    num_candidates = rightmost_idx + 1            # [QB] (identity frontier)
    rightmost_pos = rightmost_idx

    # Per-row (m = q * G_PAD2 + g) indexing
    m_ids = tl.arange(0, M).to(tl.int32)
    q_of_m = m_ids // G_PAD2
    g_of_m = m_ids % G_PAD2
    g_ok_m = g_of_m < NUM_GROUPS
    t_of_m = tl.minimum(t0 + q_of_m, T - 1)          # CHUNK-LOCAL: addressing
    t_of_m_g = t_base + t_of_m                       # GLOBAL: causality/RoPE
    head_of_m = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_of_m.to(tl.int64)) * g_ok_m.to(tl.int64)
    num_cand_m = t_of_m_g // layer_power + 1      # [M]

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_row_base = (
        q + i_b.to(tl.int64) * T_i64 * H * K
        + t_of_m.to(tl.int64)[:, None] * H * K
        + head_of_m[:, None] * K
    )
    q_lo = tl.load(q_row_base + o_k_half[None, :], mask=g_ok_m[:, None], other=0.0)
    q_hi = tl.load(q_row_base + (K // 2) + o_k_half[None, :], mask=g_ok_m[:, None], other=0.0)

    rope_q_m = (t_of_m_g // layer_power).to(tl.int64)
    cos_q = tl.load(cos_cache + rope_q_m[:, None] * (K // 2) + o_k_half[None, :])
    sin_q = tl.load(sin_cache + rope_q_m[:, None] * (K // 2) + o_k_half[None, :])
    q_rope_lo = (q_lo * cos_q - q_hi * sin_q) * scale   # [M, K//2]
    q_rope_hi = (q_lo * sin_q + q_hi * cos_q) * scale   # [M, K//2]

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K

    idx_mask = (1 << LOG_N) - 1
    topk_running_max = tl.full([QB], NEG_INF, dtype=tl.float32)

    # Running Top-K lives in `scratch` between merges (stable lane order)
    n_prog_t = tl.num_programs(0)
    sbase = (pid_bh_kv.to(tl.int64) * n_prog_t + pid_t) * (QB * TOP_K)
    sk_off = tl.arange(0, QB)[:, None].to(tl.int64) * TOP_K + tl.arange(0, TOP_K)[None, :].to(tl.int64)

    init_idx = tl.arange(0, TOP_K).to(tl.int32)
    init_int = tl.full([TOP_K], NEG_INF, dtype=tl.float32).to(tl.int32, bitcast=True)
    init_enc = ((init_int & ~idx_mask) | (init_idx & idx_mask)).to(tl.float32, bitcast=True)
    tl.store(scratch + sbase + sk_off, tl.broadcast_to(init_enc[None, :], [QB, TOP_K]))
    tl.debug_barrier()

    t_last_g = t_base + tl.minimum(t0 + QB - 1, T - 1)   # GLOBAL: loop bound
    nvp_max = (t_last_g // layer_power) // COMPRESSION_RATE + 1
    num_batches = (nvp_max + PARENTS_PER_BATCH - 1) // PARENTS_PER_BATCH

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    nt_ids = tl.arange(0, PARENT_TILES).to(tl.int32)
    o_bc = tl.arange(0, BC).to(tl.int32)

    for i_batch in range(num_batches):
        # ---- step 1: dense scoring, QB queries share each K/RoPE tile ----
        imp_buf = tl.full([QB, PARENT_TILES, TILE_N], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([QB], NEG_INF, dtype=tl.float32)
        batch_pos0 = (i_batch * BC).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            pos_n = batch_pos0 + i_tile * TILE_N + o_n           # [TILE_N] candidate pos == node id
            cand_ok_m = pos_n[None, :] < num_cand_m[:, None]      # [M, TILE_N]

            safe_rows = tl.minimum(pos_n, N_layer - 1).to(tl.int64)
            k_row = k_base + safe_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = pos_n.to(tl.int64)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            if SEL_IEEE:
                scores = (tl.dot(q_rope_lo, tl.trans(k_rope_lo), input_precision="ieee")
                          + tl.dot(q_rope_hi, tl.trans(k_rope_hi), input_precision="ieee"))
            else:
                scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            if HAS_B:
                b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_rows * H_kv + i_h_kv.to(tl.int64))
                scores = scores + b_vals[None, :]
            scores = tl.where(cand_ok_m & g_ok_m[:, None], scores, NEG_INF)   # [M, TILE_N]

            # Per-query running max over the tile (exact, order-free)
            local_m = tl.max(tl.reshape(scores, [QB, G_PAD2 * TILE_N]), axis=1)  # [QB]
            new_bm = tl.maximum(batch_m, local_m)
            rescale_old = tl.exp(batch_m - new_bm)
            imp_buf = tl.where(imp_buf >= 0, imp_buf * rescale_old[:, None, None], imp_buf)
            batch_m = new_bm

            bm_row = tl.reshape(tl.broadcast_to(batch_m[:, None], [QB, G_PAD2]), [M])
            attn = tl.exp(scores - bm_row[:, None])
            attn = tl.where(cand_ok_m & g_ok_m[:, None], attn, 0.0)
            attn3 = tl.reshape(attn, [QB, G_PAD2, TILE_N])
            g2 = tl.arange(0, G_PAD2)
            if G_PAD2 == 1:
                imp_tile = tl.sum(attn3, axis=1)
            elif G_PAD2 == 2:
                r0 = tl.sum(tl.where((g2 == 0)[None, :, None], attn3, 0.0), axis=1)
                r1 = tl.sum(tl.where((g2 == 1)[None, :, None], attn3, 0.0), axis=1)
                imp_tile = r0 + r1
            elif G_PAD2 == 4:
                r0 = tl.sum(tl.where((g2 == 0)[None, :, None], attn3, 0.0), axis=1)
                r1 = tl.sum(tl.where((g2 == 1)[None, :, None], attn3, 0.0), axis=1)
                r2 = tl.sum(tl.where((g2 == 2)[None, :, None], attn3, 0.0), axis=1)
                r3 = tl.sum(tl.where((g2 == 3)[None, :, None], attn3, 0.0), axis=1)
                imp_tile = (r0 + r2) + (r1 + r3)
            elif G_PAD2 == 8:
                r0 = tl.sum(tl.where((g2 == 0)[None, :, None], attn3, 0.0), axis=1)
                r1 = tl.sum(tl.where((g2 == 1)[None, :, None], attn3, 0.0), axis=1)
                r2 = tl.sum(tl.where((g2 == 2)[None, :, None], attn3, 0.0), axis=1)
                r3 = tl.sum(tl.where((g2 == 3)[None, :, None], attn3, 0.0), axis=1)
                r4 = tl.sum(tl.where((g2 == 4)[None, :, None], attn3, 0.0), axis=1)
                r5 = tl.sum(tl.where((g2 == 5)[None, :, None], attn3, 0.0), axis=1)
                r6 = tl.sum(tl.where((g2 == 6)[None, :, None], attn3, 0.0), axis=1)
                r7 = tl.sum(tl.where((g2 == 7)[None, :, None], attn3, 0.0), axis=1)
                imp_tile = ((r0 + r4) + (r2 + r6)) + ((r1 + r5) + (r3 + r7))
            else:
                tl.static_assert(G_PAD2 == 16, "G_PAD2 must be a power of 2 <= 16")
                r0 = tl.sum(tl.where((g2 == 0)[None, :, None], attn3, 0.0), axis=1)
                r1 = tl.sum(tl.where((g2 == 1)[None, :, None], attn3, 0.0), axis=1)
                r2 = tl.sum(tl.where((g2 == 2)[None, :, None], attn3, 0.0), axis=1)
                r3 = tl.sum(tl.where((g2 == 3)[None, :, None], attn3, 0.0), axis=1)
                r4 = tl.sum(tl.where((g2 == 4)[None, :, None], attn3, 0.0), axis=1)
                r5 = tl.sum(tl.where((g2 == 5)[None, :, None], attn3, 0.0), axis=1)
                r6 = tl.sum(tl.where((g2 == 6)[None, :, None], attn3, 0.0), axis=1)
                r7 = tl.sum(tl.where((g2 == 7)[None, :, None], attn3, 0.0), axis=1)
                r8 = tl.sum(tl.where((g2 == 8)[None, :, None], attn3, 0.0), axis=1)
                r9 = tl.sum(tl.where((g2 == 9)[None, :, None], attn3, 0.0), axis=1)
                r10 = tl.sum(tl.where((g2 == 10)[None, :, None], attn3, 0.0), axis=1)
                r11 = tl.sum(tl.where((g2 == 11)[None, :, None], attn3, 0.0), axis=1)
                r12 = tl.sum(tl.where((g2 == 12)[None, :, None], attn3, 0.0), axis=1)
                r13 = tl.sum(tl.where((g2 == 13)[None, :, None], attn3, 0.0), axis=1)
                r14 = tl.sum(tl.where((g2 == 14)[None, :, None], attn3, 0.0), axis=1)
                r15 = tl.sum(tl.where((g2 == 15)[None, :, None], attn3, 0.0), axis=1)
                imp_tile = (
                    ((r0 + r8) + (r4 + r12)) + ((r2 + r10) + (r6 + r14))
                ) + (
                    ((r1 + r9) + (r5 + r13)) + ((r3 + r11) + (r7 + r15))
                )
            cand_ok_q = pos_n[None, :] < num_candidates[:, None]
            imp_tile = tl.where(cand_ok_q, imp_tile, NEG_INF)
            imp_buf = tl.where((nt_ids == i_tile)[None, :, None], imp_tile[:, None, :], imp_buf)

        # ---- step 2: rescale + encode + bisection Top-K merge ----
        flat = tl.reshape(imp_buf, [QB, BC])
        pos_c = batch_pos0 + o_bc                                 # [BC]
        valid_pos = pos_c[None, :] < num_candidates[:, None]
        flat = tl.where(valid_pos, flat, NEG_INF)

        sel_m_new = tl.maximum(topk_running_max, batch_m)
        rescale_run = tl.exp(topk_running_max - sel_m_new)
        rescale_batch = tl.exp(batch_m - sel_m_new)

        run_enc = tl.load(scratch + sbase + sk_off)               # [QB, TOP_K]
        run_int = run_enc.to(tl.int32, bitcast=True)
        run_raw_idx = run_int & idx_mask
        run_clean_imp = (run_int & ~idx_mask).to(tl.float32, bitcast=True)
        run_resc = tl.where(run_clean_imp >= 0, run_clean_imp * rescale_run[:, None], run_clean_imp)
        run_enc2 = (run_resc.to(tl.int32, bitcast=True) & ~idx_mask) | run_raw_idx

        flat = tl.where(flat >= 0, flat * rescale_batch[:, None], flat)
        flat = tl.where(pos_c[None, :] == rightmost_pos[:, None], 1e6, flat)
        if HAS_FORCE:
            f_word = tl.load(force_mask + (pos_c >> 5).to(tl.int64),
                             mask=pos_c < N_layer, other=0)
            f_hit = ((f_word >> (pos_c & 31)) & 1) != 0
            flat = tl.where(f_hit[None, :] & valid_pos, 1e6, flat)
        topk_running_max = sel_m_new

        flat_int = flat.to(tl.int32, bitcast=True)
        enc_idx = tl.where(flat >= 0, ~pos_c[None, :], pos_c[None, :]) & idx_mask
        bat_enc2 = (flat_int & ~idx_mask) | enc_idx               # [QB, BC]

        us_run = (run_enc2 ^ ((run_enc2 >> 31) | SIGN)) ^ SIGN
        us_bat = (bat_enc2 ^ ((bat_enc2 >> 31) | SIGN)) ^ SIGN

        tau = tl.zeros([QB], dtype=tl.int32)
        cand_t = tau | SIGN
        cnt = (
            tl.sum((us_run >= (cand_t ^ SIGN)[:, None]).to(tl.int32), axis=1)
            + tl.sum((us_bat >= (cand_t ^ SIGN)[:, None]).to(tl.int32), axis=1)
        )
        tau = tl.where(cnt >= TOP_K, cand_t, tau)
        for i in tl.static_range(30, -1, -1):
            cand_t = tau | (1 << i)
            cnt = (
                tl.sum((us_run >= (cand_t ^ SIGN)[:, None]).to(tl.int32), axis=1)
                + tl.sum((us_bat >= (cand_t ^ SIGN)[:, None]).to(tl.int32), axis=1)
            )
            tau = tl.where(cnt >= TOP_K, cand_t, tau)

        taus = (tau ^ SIGN)[:, None]
        keep_r = us_run >= taus
        keep_b = us_bat >= taus
        rank_r = tl.cumsum(keep_r.to(tl.int32), axis=1) - keep_r.to(tl.int32)
        nkeep_r = tl.sum(keep_r.to(tl.int32), axis=1)
        rank_b = nkeep_r[:, None] + tl.cumsum(keep_b.to(tl.int32), axis=1) - keep_b.to(tl.int32)

        q_l64 = tl.arange(0, QB)[:, None].to(tl.int64)
        tl.debug_barrier()  # all lanes loaded run_enc before overwrite
        tl.store(
            scratch + sbase + q_l64 * TOP_K + rank_r.to(tl.int64),
            run_enc2.to(tl.float32, bitcast=True),
            mask=keep_r & (rank_r < TOP_K),
        )
        tl.store(
            scratch + sbase + q_l64 * TOP_K + rank_b.to(tl.int64),
            bat_enc2.to(tl.float32, bitcast=True),
            mask=keep_b & (rank_b < TOP_K),
        )
        tl.debug_barrier()

    # ---- Decode final Top-K -> next_selected_parents (+ optional bitmask) ----
    run_enc = tl.load(scratch + sbase + sk_off)
    topk_int = run_enc.to(tl.int32, bitcast=True)
    raw_idx = topk_int & idx_mask
    clean_imp = (topk_int & ~idx_mask).to(tl.float32, bitcast=True)
    topk_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx) & idx_mask
    topk_valid = (
        (clean_imp > SCORE_VALID_THRESHOLD)
        & (topk_pos < num_candidates[:, None])
        & q_row_ok[:, None]
    )
    out_rank = tl.cumsum(topk_valid.to(tl.int32), axis=1) - topk_valid.to(tl.int32)
    out_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + t_q.to(tl.int64)[:, None] * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(
        next_selected_parents + out_base + out_rank.to(tl.int64),
        topk_pos.to(tl.int32),
        mask=topk_valid,
    )

    if WRITE_BITMASK:
        wm_base = (
            i_b.to(tl.int64) * T_i64 * H_kv * NWORDS
            + t_q.to(tl.int64)[:, None] * H_kv * NWORDS
            + i_h_kv.to(tl.int64) * NWORDS
        )
        word = (topk_pos >> 5).to(tl.int64)
        bitv = 1 << (topk_pos & 31)
        tl.atomic_or(selected_bitmask + wm_base + word, bitv, mask=topk_valid)


@triton.jit
def recsa_top_accumulate_gqa_kernel(
    q,                      # [B, T, H, K]
    layer_k,                # [B, N_layer, H_kv, K]
    layer_v,                # [B, N_layer, H_kv, V]
    cos_cache, sin_cache,   # [cache_size, K//2]
    selected_bitmask,       # [B, T, H_kv, NWORDS] int32
    layer_max,              # [B, T, H]
    layer_sum,              # [B, T, H]
    layer_output,           # [B, T, H, V]
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_layer,
    NEG_INF: tl.constexpr,
    G_PAD2: tl.constexpr,
    QB: tl.constexpr,
    TILE_AN: tl.constexpr,
    NWORDS,
    scale,
    t_ptr=None,
    HAS_CURSOR: tl.constexpr = 0,
    b_bias=None,  # [B, N_layer, H_kv] fp32 per-node selection bias
    HAS_B: tl.constexpr = 0,
    NO_SUMMARIES: tl.constexpr = 0,  # summaries="none": accumulate nothing
    DOT_FP16: tl.constexpr = 0,
    krot=None,
    PREROT: tl.constexpr = 0,
    LA: tl.constexpr = 0,
):
    """Dense complement accumulation for the top tree layer."""
    M: tl.constexpr = QB * G_PAD2
    W: tl.constexpr = TILE_AN // 32

    pid_t = tl.program_id(0)
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
        N_layer = t_base // layer_power + 1

    T_i64 = T
    t0 = pid_t * QB
    q_ids = tl.arange(0, QB).to(tl.int32)
    t_vec = t0 + q_ids
    t_qv = tl.minimum(t_vec, T - 1)                  # CHUNK-LOCAL: addressing
    num_candidates = (t_base + t_qv) // layer_power + 1   # [QB] GLOBAL

    m_ids = tl.arange(0, M).to(tl.int32)
    q_of_m = m_ids // G_PAD2
    g_of_m = m_ids % G_PAD2
    g_ok_m = g_of_m < NUM_GROUPS
    row_store_ok = g_ok_m & ((t0 + q_of_m) < T)
    t_of_m = tl.minimum(t0 + q_of_m, T - 1)          # CHUNK-LOCAL: addressing
    t_of_m_g = t_base + t_of_m                       # GLOBAL: causality/RoPE
    head_of_m = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_of_m.to(tl.int64)) * g_ok_m.to(tl.int64)
    num_cand_m = t_of_m_g // layer_power + 1      # [M]

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_row_base = (
        q + i_b.to(tl.int64) * T_i64 * H * K
        + t_of_m.to(tl.int64)[:, None] * H * K
        + head_of_m[:, None] * K
    )
    q_lo = tl.load(q_row_base + o_k_half[None, :], mask=g_ok_m[:, None], other=0.0)
    q_hi = tl.load(q_row_base + (K // 2) + o_k_half[None, :], mask=g_ok_m[:, None], other=0.0)

    rope_q_m = (t_of_m_g // layer_power).to(tl.int64)
    cos_q = tl.load(cos_cache + rope_q_m[:, None] * (K // 2) + o_k_half[None, :])
    sin_q = tl.load(sin_cache + rope_q_m[:, None] * (K // 2) + o_k_half[None, :])
    q_rope_lo = (q_lo * cos_q - q_hi * sin_q) * scale
    q_rope_hi = (q_lo * sin_q + q_hi * cos_q) * scale

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V
    wm_base_q = (
        i_b.to(tl.int64) * T_i64 * H_kv * NWORDS
        + t_qv.to(tl.int64) * H_kv * NWORDS
        + i_h_kv.to(tl.int64) * NWORDS
    )                                              # [QB]

    cur_max = tl.full([M], NEG_INF, dtype=tl.float32)
    cur_sum = tl.zeros([M], dtype=tl.float32)
    cur_output = tl.zeros([M, V], dtype=tl.float32)

    o_v = tl.arange(0, V).to(tl.int64)
    o_an = tl.arange(0, TILE_AN).to(tl.int32)
    o_w = tl.arange(0, W).to(tl.int32)
    o_32 = tl.arange(0, 32).to(tl.int32)

    t_last_g = t_base + tl.minimum(t0 + QB - 1, T - 1)   # GLOBAL: loop bound
    nc_max = t_last_g // layer_power + 1
    num_tiles = (nc_max + TILE_AN - 1) // TILE_AN
    if NO_SUMMARIES:
        num_tiles = 0

    if PREROT and LA:
        kr_b = krot + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
        q16_lo = q_rope_lo.to(tl.float16)
        q16_hi = q_rope_hi.to(tl.float16)
        r0 = tl.minimum(o_an, N_layer - 1).to(tl.int64)
        scores_la = (tl.dot(q16_lo, tl.trans(tl.load(kr_b + r0[:, None] * (H_kv * K) + o_k_half[None, :])))
                     + tl.dot(q16_hi, tl.trans(tl.load(kr_b + r0[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :]))))
    for i_tile in range(num_tiles):
        pos_n = (i_tile * TILE_AN).to(tl.int32) + o_an            # [TILE_AN]

        # Selected bits for the QB queries (rows owned by this program)
        w_idx = (i_tile * W).to(tl.int32) + o_w                   # [W]
        words = tl.load(
            selected_bitmask + wm_base_q[:, None] + w_idx[None, :].to(tl.int64),
            mask=(w_idx < NWORDS)[None, :], other=0,
        )                                                         # [QB, W]
        sel_q = tl.reshape((words[:, :, None] >> o_32[None, None, :]) & 1, [QB, TILE_AN]) > 0
        keep_q = (pos_n[None, :] < num_candidates[:, None]) & (~sel_q)   # [QB, TILE_AN]
        keep_m = tl.reshape(
            tl.broadcast_to(keep_q[:, None, :], [QB, G_PAD2, TILE_AN]), [M, TILE_AN]
        ) & g_ok_m[:, None] & (pos_n[None, :] < num_cand_m[:, None])

        safe_rows = tl.minimum(pos_n, N_layer - 1).to(tl.int64)
        if PREROT and LA:
            scores = scores_la
            rn = tl.minimum(pos_n + TILE_AN, N_layer - 1).to(tl.int64)
            scores_la = (tl.dot(q16_lo, tl.trans(tl.load(kr_b + rn[:, None] * (H_kv * K) + o_k_half[None, :])))
                         + tl.dot(q16_hi, tl.trans(tl.load(kr_b + rn[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :]))))
        elif PREROT:
            kr_row = (krot + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
                      + safe_rows[:, None] * (H_kv * K))
            scores = (tl.dot(q_rope_lo.to(tl.float16), tl.trans(tl.load(kr_row + o_k_half[None, :])))
                      + tl.dot(q_rope_hi.to(tl.float16), tl.trans(tl.load(kr_row + (K // 2) + o_k_half[None, :]))))
        else:
            k_row = k_base + safe_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = pos_n.to(tl.int64)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            if DOT_FP16 == 1:
                scores = (tl.dot(q_rope_lo.to(tl.float16), tl.trans(k_rope_lo.to(tl.float16)))
                          + tl.dot(q_rope_hi.to(tl.float16), tl.trans(k_rope_hi.to(tl.float16))))
            elif DOT_FP16 == 2:
                scores = (tl.dot(q_rope_lo, tl.trans(k_rope_lo), input_precision="ieee")
                          + tl.dot(q_rope_hi, tl.trans(k_rope_hi), input_precision="ieee"))
            else:
                scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        if HAS_B:
            b_vals = tl.load(b_bias + i_b.to(tl.int64) * N_layer * H_kv + safe_rows * H_kv + i_h_kv.to(tl.int64))
            scores = scores + b_vals[None, :]
        scores = tl.where(keep_m, scores, NEG_INF)                # [M, TILE_AN]

        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        if DOT_FP16 == 1:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float16)  # PV runs fp16 (probs cast to v_tile.dtype)
        else:
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)     # [TILE_AN, V]

        tile_m = tl.max(scores, axis=1)                           # [M]
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]

        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(keep_m, attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)
        if DOT_FP16 == 2:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile, input_precision="ieee")
        else:
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m

    state_off = (
        i_b.to(tl.int64) * T_i64 * H
        + t_of_m.to(tl.int64) * H
        + head_of_m
    )
    tl.store(layer_max + state_off, cur_max, mask=row_store_ok)
    tl.store(layer_sum + state_off, cur_sum, mask=row_store_ok)
    out_off = (
        i_b.to(tl.int64) * T_i64 * H * V
        + t_of_m.to(tl.int64) * H * V
        + head_of_m * V
    )
    tl.store(layer_output + out_off[:, None] + o_v[None, :], cur_output, mask=row_store_ok[:, None])


def recsa_build_pyramid(k: torch.Tensor, v: torch.Tensor,
                        compression_rate: int = 16,
                        max_top_nodes: int = 8192):
    """Kernel-path tree build, for callers that build the pyramid once and attend to it many times."""
    k = k.contiguous()
    v = v.contiguous()
    B, T_kv, H_kv, K = k.shape
    V = v.shape[-1]
    num_layers = 1
    temp_len = T_kv
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    layers_k = [k]
    layers_v = [v]
    current_k, current_v = k, v
    current_len = T_kv
    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H_kv, K, dtype=k.dtype, device=k.device)
        next_v = torch.empty(B, next_len, H_kv, V, dtype=v.dtype, device=v.device)

        BLOCK_SIZE = 8
        grid = (triton.cdiv(next_len, BLOCK_SIZE), B * H_kv)

        with _cuda_timer("  Built layer %d: %d nodes", layer_idx, next_len):
            recsa_build_kernel[grid](
                current_k, current_v, next_k, next_v,
                N_child=current_len,
                N_parent=next_len,
                B=B, H_kv=H_kv, K=K, V=V,
                COMPRESSION_RATE=compression_rate,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len
    return layers_k, layers_v


def recsa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    compression_rate: int = 16,
    max_top_nodes: int = 8192,
    top_k_per_layer: int = 512,
    scale: Optional[float] = None,
    rope_base: float = 10000.0,
    rope_mode: str = "internal",
    _save_for_backward: bool = False,
    _save_parents: bool = True,
    layers_kv: Optional[tuple] = None,
    tile_p: int = 4,
    num_warps: int = 4,
    num_stages: int = 2,
    packed_top: Optional[bool] = None,
    packed_qb: Optional[int] = None,
    packed_tile_an: int = 64,
    packed_warps: int = 8,
    mid_ns: Optional[bool] = None,
    mid_ns_wide: Optional[bool] = None,
    mid_split: Optional[bool] = None,
    mid_chunk: int = 32768,
    q_chunk: Optional[int] = None,
    q_start: Optional[int] = None,
    t_dev: Optional[torch.Tensor] = None,
    cursor_nl: Optional[int] = None,
    force_masks: Optional[dict] = None,
    level_bias: Optional[list] = None,
    no_summaries: bool = False,
    summary_norm: Optional[dict] = None,
) -> torch.Tensor:
    """RecSA attention forward with the Triton kernels."""
    B, T, H, K = q.shape
    H_kv = k.shape[2]  # number of KV heads
    V = v.shape[-1]
    T_kv = k.shape[1]
    assert k.shape[1] == v.shape[1], "k and v must have the same length"
    if q_start is None:
        assert T == T_kv, (
            f"q length {T} != k/v length {T_kv}; windowed queries require an "
            f"explicit q_start")
        _q0 = 0
    else:
        assert not _save_for_backward, "q_start is forward-only"
        _q0 = int(q_start)
        assert 0 <= _q0 and _q0 + T <= T_kv, (
            f"query window [{_q0}, {_q0 + T}) exceeds k/v length {T_kv}")

    # Validate GQA configuration
    assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
    NUM_GROUPS = H // H_kv
    assert k.shape[2] == v.shape[2], f"K and V must have same number of heads"
    assert (top_k_per_layer & (top_k_per_layer - 1)) == 0, "top_k_per_layer (TOP_K) must be a power of 2"

    assert max_top_nodes == top_k_per_layer * compression_rate, (
        f"max_top_nodes ({max_top_nodes}) must equal top_k_per_layer*compression_rate "
        f"({top_k_per_layer}*{compression_rate}={top_k_per_layer * compression_rate})"
    )
    assert (compression_rate & (compression_rate - 1)) == 0, "compression_rate must be a power of 2"
    assert (max_top_nodes & (max_top_nodes - 1)) == 0, "max_top_nodes must be a power of 2"
    assert top_k_per_layer % compression_rate == 0, (
        f"top_k_per_layer ({top_k_per_layer}) must be divisible by compression_rate ({compression_rate})"
    )

    if scale is None:
        scale = K ** -0.5
    if rope_mode not in {"internal", "none"}:
        raise ValueError(f"unsupported rope_mode {rope_mode!r}; expected 'internal' or 'none'")
    if not isinstance(no_summaries, bool):
        raise TypeError(f"no_summaries must be a bool, got {no_summaries!r}")
    _nosum = 1 if no_summaries else 0
    # summary_norm: None = off
    _sn_spec = summary_norm_spec(summary_norm)
    if _sn_spec is not None:
        if no_summaries:
            raise ValueError("summary_norm with no_summaries=True: no summary logit enters "
                             "the softmax, there is nothing to normalise")
        if level_bias is not None:
            raise ValueError("summary_norm is not combinable with level_bias")
        if _save_for_backward and not _save_parents:
            raise ValueError("summary_norm is inference-only: the backward has no "
                             "mass_cap support; a training-shaped call refuses it")

    if t_dev is not None:
        assert B == 1 and T == 1, "cursor mode is the B=1, Tq=1 decode step"
        assert q_start is not None, "cursor mode needs q_start (host mirror)"
        assert layers_kv is not None and cursor_nl is not None, \
            "cursor mode needs full-capacity layers_kv + live cursor_nl"
        assert not _save_for_backward, "cursor mode is forward-only"
        assert t_dev.is_cuda and t_dev.dtype == torch.int64 and t_dev.numel() == 1

    _staged = not k.is_cuda
    if _staged:
        assert k.is_pinned() and v.is_pinned(), \
            "host-resident leaves must be pinned memory (UVA-mapped)"
        assert B == 1 and T == 1, \
            "host-resident leaves serve the B=1, Tq=1 decode step only"
        assert layers_kv is not None and q_start is not None, \
            "host-resident leaves require layers_kv and q_start"
        assert not _save_for_backward, "leaf offload is forward-only"

    device = q.device
    dtype = q.dtype

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # --- step 1: build tree ---
    nvtx.range_push("Phase1_TreeBuilding")
    _trace("Phase 1: Building tree structure...")

    num_layers = 1
    temp_len = T_kv
    while temp_len > max_top_nodes:
        temp_len = (temp_len + compression_rate - 1) // compression_rate
        num_layers += 1
    if t_dev is not None:
        num_layers = int(cursor_nl)

    _trace("  Tree has %d layers (H=%d, H_kv=%d, num_groups=%d)",
                num_layers, H, H_kv, NUM_GROUPS)

    if layers_kv is not None:
        layers_k, layers_v = layers_kv
        if len(layers_k) != num_layers or len(layers_v) != num_layers:
            raise ValueError(
                f"layers_kv has {len(layers_k)}/{len(layers_v)} layers; expected {num_layers}"
            )
        expect_len = T_kv
        for layer_idx in range(num_layers):
            if t_dev is not None:
                break
            if layers_k[layer_idx].shape[1] != expect_len or layers_v[layer_idx].shape[1] != expect_len:
                raise ValueError(f"layers_kv layer {layer_idx} has wrong node count")
            expect_len = (expect_len + compression_rate - 1) // compression_rate

        if B > 1:
            for t in (*layers_k, *layers_v):
                if not t.is_contiguous():
                    raise ValueError(
                        "layers_kv holds non-contiguous levels at B>1; "
                        "contiguising them here would copy the whole pyramid "
                        "per call. Pass contiguous levels (decode growth "
                        "buffers are only slice-contiguous at B=1).")
        layers_k = [t.contiguous() for t in layers_k]
        layers_v = [t.contiguous() for t in layers_v]
    else:
        layers_k, layers_v = recsa_build_pyramid(k, v, compression_rate, max_top_nodes)
    nvtx.range_pop()

    # --- step 2: RoPE cache ---
    nvtx.range_push("Phase1.5_RoPE_Cache")
    _trace("Phase 1.5: Precomputing RoPE cache...")

    cache_size = max_top_nodes + 1024
    _rk = (device.index if device.type == "cuda" else -1, K, cache_size,
           float(rope_base), rope_mode)
    _hit = _ROPE_CACHES.get(_rk)
    if _hit is not None:
        cos_cache, sin_cache = _hit
    else:
        with torch.inference_mode(False), torch.no_grad():
            if rope_mode == "internal":
                inv_freq = 1.0 / (rope_base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
                positions = torch.arange(cache_size, dtype=torch.float32, device=device)
                freqs = torch.outer(positions, inv_freq)
                cos_cache = freqs.cos()
                sin_cache = freqs.sin()
            else:
                cos_cache = torch.ones(cache_size, K // 2, dtype=torch.float32, device=device)
                sin_cache = torch.zeros(cache_size, K // 2, dtype=torch.float32, device=device)
        _ROPE_CACHES[_rk] = (cos_cache, sin_cache)
    nvtx.range_pop()

    _sn_spec = _sumnorm_prepare(_sn_spec, num_layers)

    # --- step 3: init global state ---
    nvtx.range_push("Phase2_Init_GlobalState_and_Buffers")
    _trace("Phase 2: Initializing global states and buffers...")

    nvtx.range_pop()

    # --- step 4: layer-by-layer forward ---
    nvtx.range_push("Phase3_LayerByLayer_Forward")
    _trace("Phase 3: Layer-by-layer forward pass...")

    # layer_max/sum/output: allocated per query-chunk below.
    mid_ns_eff = MID_NS_DEFAULT if mid_ns is None else bool(mid_ns)
    mid_ns_wide_eff = MID_NS_WIDE if mid_ns_wide is None else bool(mid_ns_wide)
    mid_split_eff = (
        mid_ns_eff and top_k_per_layer >= 256
        and (bool(mid_split) if mid_split is not None else not _save_for_backward)
    )
    if mid_split_eff and _save_for_backward and not _save_parents:
        raise ValueError(
            "mid_split=True with _save_for_backward=True is eval/parity-probe "
            "only (backward replays legacy selection); drop mid_split, or "
            "pass _save_parents=True for parity inspection")
    if level_bias is not None:
        assert len(level_bias) == num_layers, (len(level_bias), num_layers)
        assert level_bias[0] is None, "leaves are never biased (b≡0)"
        if _save_for_backward:
            raise ValueError(
                "level_bias is eval-only: backward replays selection through "
                "b-less kernels")
    mid_scratch = None
    mid_imp = None
    mid_tau = None

    MAX_CANDIDATES = max_top_nodes

    # constexprs for kernels that use bitonic sort / bit-packing
    LOG_N = int(math.log2(MAX_CANDIDATES))
    N_DIMS_TOPK = int(math.log2(top_k_per_layer))

    # Dropped candidates are accumulated in blocks for better throughput.
    DROP_BLOCK = 64
    assert top_k_per_layer % DROP_BLOCK == 0, f"top_k_per_layer ({top_k_per_layer}) must be divisible by DROP_BLOCK ({DROP_BLOCK})"

    TILE_P = int(tile_p)
    assert TILE_P > 0 and (TILE_P & (TILE_P - 1)) == 0, "tile_p must be a power of 2"
    G_PAD = max(triton.next_power_of_2(NUM_GROUPS), 16)

    NS_FAST = 1 if TILE_P == top_k_per_layer // compression_rate else 0

    packed_supported, QB_sel, QB_acc, G_PAD2 = _packed_top_params(
        NUM_GROUPS, K, compression_rate, top_k_per_layer, num_layers, TILE_P,
        packed_qb=packed_qb,
    )
    packed_supported = packed_supported and packed_tile_an % 32 == 0 and packed_tile_an >= 32
    if packed_top is None:
        packed_top_eff = packed_supported
    elif packed_top:
        assert packed_supported, "packed_top=True requested but configuration unsupported"
        packed_top_eff = True
    else:
        packed_top_eff = False

    if q_chunk is None or int(q_chunk) <= 0 or int(q_chunk) >= T:
        _chunk_len, _starts = T, [0]
    else:
        if _save_for_backward:
            raise ValueError(
                "q_chunk is forward-only: the backward context (global_max/"
                "global_sum/per_layer_parents) is full-T state that a chunked "
                "forward never materialises. Drop q_chunk for training calls.")
        _chunk_len = int(q_chunk)
        _starts = list(range(0, T, _chunk_len))
    _single_chunk = len(_starts) == 1
    output = torch.empty(B, T, H, V, dtype=dtype, device=device)
    _ws = None

    for qc0 in _starts:
        Tq = min(_chunk_len, T - qc0)
        q_c = q if _single_chunk else q[:, qc0:qc0 + Tq].contiguous()

        top_layer_power = compression_rate ** (num_layers - 1)
        _chain = (RECSA_DSPLIT_CHAIN == "fused" and Tq == 1
                  and not _save_for_backward and bool(RECSA_DECODE_SPLIT)
                  and top_k_per_layer >= 256
                  and (num_layers == 1 or (mid_split_eff and packed_top_eff)))
        _hfuse = (_chain and RECSA_DSPLIT_HFUSE == "on" and num_layers >= 2
                  and num_warps == 4)
        _n_fused = 0
        if Tq == 1 and not _save_for_backward:
            _ws = _decode_workspace(
                device, B, H, H_kv, V, K, top_k_per_layer,
                staged_leaf=(k.dtype, max_top_nodes) if _staged else None)
            if mid_scratch is None:
                mid_scratch = _ws["mid_scratch"]
                mid_imp, mid_tau = _ws["mid_imp"], _ws["mid_tau"]
            if _chain:
                # the top level's merge is FRESH: it never reads these
                global_max = _ws["global_max"]
                global_sum = _ws["global_sum"]
                global_output = _ws["global_output"]
            else:
                global_max = _ws["global_max"].fill_(RECSA_SCORE_NEG_INF)
                global_sum = _ws["global_sum"].zero_()
                global_output = _ws["global_output"].zero_()
            global_max_b = _ws["global_max_b"]
            global_sum_b = _ws["global_sum_b"]
            global_output_b = _ws["global_output_b"]
            layer_max = _ws["layer_max"]
            layer_sum = _ws["layer_sum"]
            layer_output = _ws["layer_output"]
            prev_selected_parents = _ws["prev"]
            if _chain and num_layers >= 2:
                pass
            elif t_dev is not None:
                # the host fill below, with nvp computed on device
                recsa_decode_frontier_kernel[(B * H_kv,)](
                    prev_selected_parents, t_dev,
                    TOP_K=top_k_per_layer,
                    TOP_LAYER_POWER=top_layer_power,
                    COMPRESSION_RATE=compression_rate)
            else:
                row = _q0 + qc0  # GLOBAL row of the single query
                nvp = row // top_layer_power // compression_rate + 1
                prev_selected_parents.copy_(_ws["iota"])
                if nvp < top_k_per_layer:
                    prev_selected_parents[..., nvp:].fill_(-1)
            next_selected_parents = _ws["next"]
        else:
            global_max = torch.empty([B, Tq, H], dtype=torch.float32, device=device)
            global_sum = torch.empty([B, Tq, H], dtype=torch.float32, device=device)
            global_output = torch.empty([B, Tq, H, V], dtype=torch.float32, device=device)
            _g_fresh = True
            global_max_b = torch.empty_like(global_max)
            global_sum_b = torch.empty_like(global_sum)
            global_output_b = torch.empty_like(global_output)
            layer_max = torch.empty([B, Tq, H], dtype=torch.float32, device=device)
            layer_sum = torch.empty([B, Tq, H], dtype=torch.float32, device=device)
            layer_output = torch.empty([B, Tq, H, V], dtype=torch.float32, device=device)
            # Initialize prev_selected_parents for the topmost layer
            t_indices = torch.arange(_q0 + qc0, _q0 + qc0 + Tq, dtype=torch.int32, device=device)  # GLOBAL rows
            rightmost_indices = t_indices // top_layer_power  # [T] rightmost node index at top layer
            # Number of virtual parents (each expands to CR top-level nodes)
            num_virtual_parents = rightmost_indices // compression_rate + 1  # [T]

            # [T, TOP_K]
            parent_candidates = torch.arange(top_k_per_layer, dtype=torch.int32, device=device).unsqueeze(0).expand(Tq, -1)
            valid_mask = parent_candidates < num_virtual_parents.unsqueeze(1)
            _prev_small = parent_candidates.masked_fill(~valid_mask, -1)
            prev_selected_parents = torch.empty(B, Tq, H_kv, top_k_per_layer, dtype=_prev_small.dtype, device=device)
            _prev_lazy = True
            next_selected_parents = torch.empty_like(prev_selected_parents)
        _do_save_parents = _save_for_backward and _save_parents
        per_layer_parents = {} if _do_save_parents else None
        _cursor = 1 if t_dev is not None else 0
        _sn = None
        if _sn_spec is not None:
            if _ws is not None and Tq == 1 and not _save_for_backward:
                _sn_lvl = _ws.get("sn_lvl")
                if _sn_lvl is None:
                    _sn_lvl = _ws["sn_lvl"] = torch.empty(B * H_kv, dtype=torch.int32, device=device)
                    _ws["sn_tot"] = torch.empty(B * H_kv, dtype=torch.int32, device=device)
                _sn_tot = _ws["sn_tot"]
            else:
                _sn_lvl = torch.empty(B * Tq * H_kv, dtype=torch.int32, device=device)
                _sn_tot = torch.empty(B * Tq * H_kv, dtype=torch.int32, device=device)
            _sn = _SumNormRun(
                _sn_spec, cnt_lvl=_sn_lvl, cnt_tot=_sn_tot,
                num_layers=num_layers, B=B, Tq=Tq, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS,
                G_PAD=G_PAD, top_k=top_k_per_layer, compression_rate=compression_rate,
                t_base=_q0 + qc0, t_dev=t_dev)
        _sn_reads_prev = _sn is not None and _sn.gate is not None
        if _cursor:
            assert packed_top_eff, "cursor mode requires the packed top path"
            assert num_layers <= 2 or mid_split_eff \
                or (mid_ns_eff and top_k_per_layer <= 64), \
                "cursor mode requires a cursor-capable mid path at this geometry"

        if force_masks is not None:
            assert not _save_for_backward, "force_masks is eval-only (no backward)"
            assert t_dev is None, "force_masks unsupported in cursor/graph mode"

        out_chunk = None
        _leaf_fused = False
        if not ('_prev_lazy' in locals() and _prev_lazy):
            _prev_lazy = False
        if not ('_g_fresh' in locals() and _g_fresh):
            _g_fresh = False

        def _g_init():
            global_max.fill_(-1e10)
            global_sum.zero_()
            global_output.zero_()
        if _chain:
            # the bottom level's fused merge writes the normalized output
            out_chunk = output if _single_chunk else torch.empty(B, Tq, H, V, dtype=dtype, device=device)

        if _hfuse:
            mid_imp, mid_tau = _dsplit_hfuse_levels(
                ws=_ws, q_c=q_c, layers_k=layers_k, layers_v=layers_v,
                cos_cache=cos_cache, sin_cache=sin_cache, out_chunk=out_chunk,
                B=B, Tq=Tq, H=H, H_kv=H_kv, K=K, V=V, NUM_GROUPS=NUM_GROUPS,
                G_PAD=G_PAD, TILE_P=TILE_P, num_layers=num_layers,
                compression_rate=compression_rate, top_k=top_k_per_layer,
                max_top_nodes=max_top_nodes, LOG_N=LOG_N, scale=scale,
                t_base=_q0 + qc0, t_dev=t_dev, num_stages=num_stages,
                level_bias=level_bias, force_masks=force_masks, staged=_staged,
                no_summaries=no_summaries, summary_norm=_sn)
            _n_fused = num_layers

        for layer_idx in (range(num_layers - 1, -1, -1) if not _hfuse else ()):
            _top_fused = False
            if _prev_lazy and (_do_save_parents or layer_idx == 0 or _sn_reads_prev or not (
                    packed_top_eff and layer_idx == num_layers - 1 and not (mid_split_eff and Tq == 1))):
                prev_selected_parents.copy_(
                    _prev_small.unsqueeze(0).unsqueeze(2).expand(B, Tq, H_kv, top_k_per_layer))
                _prev_lazy = False
            _lb = level_bias[layer_idx] if level_bias is not None else None
            _hb = 1 if _lb is not None else 0
            _fm = None
            if force_masks is not None and layer_idx > 0:
                _fm = force_masks.get(layer_idx)
            if _do_save_parents:
                per_layer_parents[layer_idx] = prev_selected_parents.clone()
            _sn_done = False
            nvtx.range_push(f"Forward_Layer_{layer_idx}")
            k_layer = layers_k[layer_idx]
            v_layer = layers_v[layer_idx]
            N_layer = k_layer.shape[1]

            is_bottom_layer = (layer_idx == 0)
            layer_power = compression_rate ** layer_idx

            _trace("  -> Processing layer %d (N=%d, power=%d, bottom=%s)...",
                        layer_idx, N_layer, layer_power, is_bottom_layer)

            grid = (Tq, B * H)

            # Kernel 1: Select+Accumulate (dispatch bottom vs non-bottom)
            nvtx.range_push("K1_SelectAccumulate")
            grid_kv = (Tq, B * H_kv)
            kernel_label = "Kernel 1a" if is_bottom_layer else "Kernel 1b"
            with _cuda_timer("      %s", kernel_label):
                if is_bottom_layer:
                    _trace("    Running bottom accumulate kernel (Kernel 1a)...")
                    if _staged:
                        if _leaf_mirror_of(k_layer) is not None: _leaf_mirror_gather(k_layer, v_layer, _ws["fetch_k"], _ws["fetch_v"], prev_selected_parents, _q0 + qc0, H_kv, K, V, compression_rate, top_k_per_layer, t_dev, _cursor)  # noqa: E701,E501
                        else: recsa_leaf_gather_kernel[(RECSA_GATHER_PROGS,)](  # noqa: E701
                            k_layer, v_layer,
                            _ws["fetch_k"], _ws["fetch_v"],
                            prev_selected_parents,
                            t_base=_q0 + qc0,
                            H_kv=H_kv, K=K, V=V,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            NPROG=RECSA_GATHER_PROGS,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            num_warps=4,
                        )
                    _dsplit_b = (RECSA_DECODE_SPLIT and Tq == 1
                                 and not _save_for_backward and _ws is not None
                                 and top_k_per_layer >= 256)
                    if _dsplit_b:
                        w2b = _ws.get("w2b")
                        if w2b is None:
                            nsl = RECSA_DSPLIT_NSLICE
                            f32 = dict(dtype=torch.float32, device=device)
                            w2b = _ws["w2b"] = dict(
                                raw=torch.empty(B * H_kv * max_top_nodes, **f32),
                                bm=torch.empty(B * H_kv * compression_rate, **f32),
                                pmax=torch.empty(nsl * B * H_kv * G_PAD, **f32),
                                psum=torch.empty(nsl * B * H_kv * G_PAD, **f32),
                                pout=torch.empty(nsl * B * H_kv * G_PAD * V, **f32),
                            )
                        recsa_dsplit_bottom_part_kernel[(RECSA_DSPLIT_NSLICE, B * H_kv)](
                            q_c, k_layer, v_layer,
                            prev_selected_parents,
                            cos_cache, sin_cache,
                            w2b["pmax"], w2b["psum"], w2b["pout"],
                            t_base=_q0 + qc0,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS,
                            K=K, V=V, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            G_PAD=G_PAD,
                            TILE_P=TILE_P,
                            NSLICE=RECSA_DSPLIT_NSLICE,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            fetch_k=_ws["fetch_k"] if _staged else None,
                            fetch_v=_ws["fetch_v"] if _staged else None,
                            LEAF_STAGED=1 if _staged else 0,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                        if _sn is not None:
                            _sn.leaf(w2b["pmax"], 1, w2b["psum"], (global_max, global_sum))
                            _sn_done = True
                        if _chain:
                            _dsplit_merge_level(
                                (w2b["pmax"], w2b["psum"], w2b["pout"]),
                                (global_max, global_sum, global_output),
                                (global_max_b, global_sum_b, global_output_b),
                                out_chunk,
                                B=B, Tq=Tq, H=H, H_kv=H_kv,
                                NUM_GROUPS=NUM_GROUPS, V=V, G_PAD=G_PAD,
                                fresh=layer_idx == num_layers - 1,
                                fuse_norm=True)
                            _n_fused += 1
                        else:
                            recsa_dsplit_partial_merge_kernel[(B * H_kv,)](
                                w2b["pmax"], w2b["psum"], w2b["pout"],
                                layer_max, layer_sum, layer_output,
                                T=Tq, H=H, H_kv=H_kv,
                                NUM_GROUPS=NUM_GROUPS, V=V,
                                NEG_INF=RECSA_SCORE_NEG_INF,
                                G_PAD=G_PAD,
                                NSLICE=RECSA_DSPLIT_NSLICE,
                                num_warps=4,
                            )
                    elif (UNION_LEAF and ACC_DOT_MODE == 1 and Tq > 1 and not _save_for_backward and not _cursor
                          and not _staged):
                        from recsa.merge_acc import leaf_union
                        _fz = None
                        if (is_bottom_layer and not _chain and dtype == torch.bfloat16 and (B == 1 or _single_chunk)
                                and _sn is None):
                            _fz_out = output[:, qc0:qc0 + Tq]
                            if _fz_out.is_contiguous():
                                if _g_fresh:
                                    _g_init()
                                    _g_fresh = False
                                _fz = (global_max, global_sum, global_output, _fz_out)
                        _leaf_fused = leaf_union(q_c, k_layer, v_layer, prev_selected_parents, cos_cache, sin_cache,
                                                 layer_max, layer_sum, layer_output, t_base=_q0 + qc0, N_layer=N_layer,
                                                 scale=scale, CR=compression_rate, neg_inf=RECSA_SCORE_NEG_INF,
                                                 fuse=_fz)
                    else:
                        recsa_bottom_accumulate_gqa_kernel[grid_kv](
                            q_c, k_layer, v_layer,
                            prev_selected_parents,
                            cos_cache, sin_cache,
                            layer_max, layer_sum, layer_output,
                            t_base=_q0 + qc0,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS,
                            K=K, V=V, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            G_PAD=G_PAD,
                            TILE_P=TILE_P,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            DOT_FP16=(ACC_DOT_MODE if (Tq > 1 and not _save_for_backward and not _cursor) else 0),
                            fetch_k=_ws["fetch_k"] if _staged else None,
                            fetch_v=_ws["fetch_v"] if _staged else None,
                            LEAF_STAGED=1 if _staged else 0,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                elif packed_top_eff and layer_idx == num_layers - 1 \
                        and not (mid_split_eff and Tq == 1):

                    _trace("    Running packed top-layer kernels (T-select + T-accum)...")
                    next_selected_parents.fill_(-1)
                    NWORDS = triton.cdiv(N_layer, 32)
                    n_prog_sel = triton.cdiv(Tq, QB_sel)
                    sel_elems = B * H_kv * n_prog_sel * QB_sel * top_k_per_layer
                    if _ws is not None:
                        selected_bitmask = _ws["bitmask"]
                        if selected_bitmask is None or selected_bitmask.shape[3] != NWORDS:
                            selected_bitmask = _ws["bitmask"] = torch.empty(
                                B, Tq, H_kv, NWORDS, dtype=torch.int32, device=device)
                        selected_bitmask.zero_()
                        sel_scratch = _ws["sel_scratch"]
                        if sel_scratch is None or sel_scratch.numel() < sel_elems:
                            sel_scratch = _ws["sel_scratch"] = torch.empty(
                                sel_elems, dtype=torch.float32, device=device)
                    else:
                        selected_bitmask = torch.zeros(B, Tq, H_kv, NWORDS, dtype=torch.int32, device=device)
                        sel_scratch = torch.empty(sel_elems, dtype=torch.float32, device=device)
                    _done_top = False
                    _krot_top = None
                    if ((FAST_SELECT if FAST_TOP_SELECT is None else FAST_TOP_SELECT) and Tq > 1
                            and not _save_for_backward and not _cursor and not _hb and _fm is None):
                        from recsa.fast_select import fast_top_select
                        _krot_top = fast_top_select(
                            q_c, k_layer, cos_cache, sin_cache, next_selected_parents, selected_bitmask,
                            t_base=_q0 + qc0, layer_power=layer_power, N_layer=N_layer, LOG_N=LOG_N,
                            score_valid_threshold=RECSA_SCORE_VALID_THRESHOLD, neg_inf=RECSA_SCORE_NEG_INF,
                            scale=scale)
                        _done_top = _krot_top is not None
                    if not _done_top:
                        recsa_top_select_kernel[(n_prog_sel, B * H_kv)](
                            q_c, k_layer, cos_cache, sin_cache,
                            next_selected_parents, selected_bitmask, sel_scratch,
                            t_base=_q0 + qc0,
                            b_bias=_lb, HAS_B=_hb,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS, K=K, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            LOG_N=LOG_N,
                            SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            G_PAD2=G_PAD2, QB=QB_sel, TILE_P=TILE_P,
                            NWORDS=NWORDS, WRITE_BITMASK=True,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            SEL_IEEE=int(SEL_IEEE and Tq > 1 and not _save_for_backward and not _cursor),
                            force_mask=_fm,
                            HAS_FORCE=1 if _fm is not None else 0,
                            num_warps=packed_warps,
                            num_stages=num_stages,
                        )
                    _prerot = int(ACC_DOT_MODE == 1 and Tq > 1 and not _save_for_backward
                                  and not _cursor and not _nosum)
                    if _prerot and _krot_top is None:
                        from recsa.fast_select import rotate_top_keys
                        _krot_top = rotate_top_keys(k_layer, cos_cache, sin_cache)
                    _v_top = v_layer.to(torch.float16) if (_prerot and v_layer.dtype == torch.bfloat16) else v_layer
                    _ta_cuda = False
                    if (_prerot and TOPACC_CUDA and not _hb and NUM_GROUPS == 4 and K == 128 and V == 128
                            and q_c.dtype == torch.bfloat16 and q_c.is_contiguous()):
                        from recsa import merge_acc as _MA
                        if not _MA._LEAF_CUDA_BROKEN and torch.cuda.get_device_capability(device) == (9, 0):
                            try:
                                from recsa.leaf_cuda import top_accum_cuda
                                _tfz = ((global_max_b, global_sum_b, global_output_b)
                                        if (_g_fresh and not _chain and RECSA_SCORE_NEG_INF == -1e10) else None)
                                top_accum_cuda(q_c, _krot_top.contiguous(), _v_top.contiguous(), cos_cache, sin_cache,
                                               selected_bitmask, layer_max, layer_sum, layer_output, t_base=_q0 + qc0,
                                               layer_power=layer_power, N_layer=N_layer, scale=scale,
                                               neg_inf=RECSA_SCORE_NEG_INF, fuse_init=_tfz)
                                _ta_cuda = True
                                _top_fused = _tfz is not None
                            except Exception as ex:   # no nvcc / driver: Triton kernel below
                                from recsa.leaf_cuda import cuda_fallback
                                cuda_fallback("top_accum", ex)   # raises under strict mode (eval runs)
                                _MA._LEAF_CUDA_BROKEN = True
                                print(f"[recsa] CUDA top accumulate unavailable, using Triton ({ex})", flush=True)
                    if not _ta_cuda:
                      recsa_top_accumulate_gqa_kernel[(triton.cdiv(Tq, QB_acc), B * H_kv)](
                          q_c, k_layer, _v_top, cos_cache, sin_cache,
                          selected_bitmask,
                          layer_max, layer_sum, layer_output,
                          t_base=_q0 + qc0,
                          b_bias=_lb, HAS_B=_hb,
                          layer_power=layer_power,
                          B=B, T=Tq, H=H, H_kv=H_kv,
                          NUM_GROUPS=NUM_GROUPS, K=K, V=V, N_layer=N_layer,
                          NEG_INF=RECSA_SCORE_NEG_INF,
                          G_PAD2=G_PAD2, QB=QB_acc,
                          TILE_AN=packed_tile_an, NWORDS=NWORDS,
                          scale=scale,
                          t_ptr=t_dev,
                          HAS_CURSOR=_cursor,
                          NO_SUMMARIES=_nosum,
                          DOT_FP16=(ACC_DOT_MODE if (Tq > 1 and not _save_for_backward and not _cursor) else 0),
                          krot=_krot_top if _prerot else None, PREROT=_prerot,
                          num_warps=4 if _prerot else packed_warps,
                          num_stages=num_stages,
                      )
                elif mid_split_eff and (layer_idx < num_layers - 1
                                        or (Tq == 1 and packed_top_eff)):
                    _trace("    Running split mid-layer kernels (M-select + M-accum)...")
                    _dsplit = (RECSA_DECODE_SPLIT and Tq == 1
                               and not _save_for_backward and _ws is not None)
                    _tail = (_dsplit and RECSA_DSPLIT_CHAIN == "fused"
                             and RECSA_DSPLIT_FINALIZE in ("v2", "v3"))
                    if not _tail:
                        next_selected_parents.fill_(-1)
                    chunk = min(Tq, 8192)
                    imp_elems = chunk * B * H_kv * max_top_nodes
                    if mid_imp is None or mid_imp.numel() < imp_elems:
                        mid_imp = torch.empty(imp_elems, dtype=torch.float32, device=device)
                        mid_tau = torch.empty(chunk * B * H_kv, dtype=torch.int32, device=device)
                        if _ws is not None:
                            _ws["mid_imp"], _ws["mid_tau"] = mid_imp, mid_tau
                    if _dsplit:
                        w2b = _ws.get("w2b")
                        if w2b is None:
                            nsl = RECSA_DSPLIT_NSLICE
                            f32 = dict(dtype=torch.float32, device=device)
                            w2b = _ws["w2b"] = dict(
                                raw=torch.empty(B * H_kv * max_top_nodes, **f32),
                                bm=torch.empty(B * H_kv * compression_rate, **f32),
                                pmax=torch.empty(nsl * B * H_kv * G_PAD, **f32),
                                psum=torch.empty(nsl * B * H_kv * G_PAD, **f32),
                                pout=torch.empty(nsl * B * H_kv * G_PAD * V, **f32),
                            )
                        recsa_dsplit_select_score_kernel[(compression_rate, B * H_kv)](
                            q_c, k_layer,
                            prev_selected_parents,
                            cos_cache, sin_cache,
                            w2b["raw"], w2b["bm"],
                            t_base=_q0 + qc0,
                            b_bias=_lb, HAS_B=_hb,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS, K=K, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            G_PAD=G_PAD, TILE_P=TILE_P,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            FRONTIER=1 if (_chain and layer_idx == num_layers - 1) else 0,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        _dsplit_select_finalize(
                            RECSA_DSPLIT_FINALIZE, (B * H_kv,),
                            prev_selected_parents,
                            next_selected_parents,
                            w2b["raw"], w2b["bm"],
                            mid_imp, mid_tau,
                            t_base=_q0 + qc0,
                            layer_power=layer_power,
                            B=B, T=Tq, H_kv=H_kv,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            LOG_N=LOG_N,
                            SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            force_mask=_fm,
                            HAS_FORCE=1 if _fm is not None else 0,
                            **({"WRITE_TAIL": 1} if _tail else {}),
                        )
                        recsa_dsplit_accumulate_part_kernel[(RECSA_DSPLIT_NSLICE, B * H_kv)](
                            q_c, k_layer, v_layer,
                            prev_selected_parents,
                            cos_cache, sin_cache,
                            w2b["pmax"], w2b["psum"], w2b["pout"],
                            mid_imp, mid_tau,
                            t_base=_q0 + qc0,
                            b_bias=_lb, HAS_B=_hb,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS, K=K, V=V, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            LOG_N=LOG_N,
                            SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            G_PAD=G_PAD, TILE_P=TILE_P,
                            NSLICE=RECSA_DSPLIT_NSLICE,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            NO_SUMMARIES=_nosum,
                            num_warps=num_warps, num_stages=num_stages,
                        )
                        if _sn is not None:
                            _sn.level(layer_idx, prev_selected_parents, next_selected_parents,
                                      (w2b["pmax"], w2b["psum"]), 1)
                            _sn_done = True
                        if _chain:
                            _dsplit_merge_level(
                                (w2b["pmax"], w2b["psum"], w2b["pout"]),
                                (global_max, global_sum, global_output),
                                (global_max_b, global_sum_b, global_output_b),
                                None,
                                B=B, Tq=Tq, H=H, H_kv=H_kv,
                                NUM_GROUPS=NUM_GROUPS, V=V, G_PAD=G_PAD,
                                fresh=layer_idx == num_layers - 1,
                                fuse_norm=False)
                            _n_fused += 1
                        else:
                            recsa_dsplit_partial_merge_kernel[(B * H_kv,)](
                                w2b["pmax"], w2b["psum"], w2b["pout"],
                                layer_max, layer_sum, layer_output,
                                T=Tq, H=H, H_kv=H_kv,
                                NUM_GROUPS=NUM_GROUPS, V=V,
                                NEG_INF=RECSA_SCORE_NEG_INF,
                                G_PAD=G_PAD,
                                NSLICE=RECSA_DSPLIT_NSLICE,
                                num_warps=4,
                            )
                        if _DSPLIT_DEBUG_CAPTURE is not None:
                            _DSPLIT_DEBUG_CAPTURE.append((
                                layer_idx,
                                mid_imp[:B * H_kv * max_top_nodes].clone(),
                                mid_tau[:B * H_kv].clone(),
                                next_selected_parents.clone()))
                    else:
                        _sel_fast = {}
                        if Tq > 1:
                            _sel_fast["DEFER_RESCALE"] = 1 if MID_DEFER else 0
                            if MID_SEL_MAXNREG > 0:
                                _sel_fast["maxnreg"] = MID_SEL_MAXNREG
                        _mid_union = (UNION_MID and N_layer <= MERGE_MAX_NODES and ACC_DOT_MODE == 1 and Tq > 1
                                      and not _save_for_backward and not _cursor and not _hb
                                      and not _nosum)
                        if _mid_union:
                            from recsa.merge_acc import (mid_acc_union, mid_union_bq, query_positions, rotate_keys,
                                                       union_lists_fast, union_xmask)
                            _krot_mid = rotate_keys(k_layer, cos_cache, sin_cache, compression_rate)
                            _posq_mid = query_positions(prev_selected_parents, _q0 + qc0, compression_rate,
                                                        layer_power)
                            _bq_m = mid_union_bq(N_layer, H // H_kv)
                        _fast_sel = _mid_union and FAST_SELECT and _fm is None
                        _krot_sel = None
                        _fs_work = None
                        for t0 in range(0, Tq, chunk):
                           cl = min(chunk, Tq - t0)
                           _ul_c = None
                           if _mid_union:
                               _ul_c = (*union_lists_fast(prev_selected_parents[:, t0:t0 + cl], _bq_m,
                                                          umax=-(-N_layer // compression_rate)), _bq_m)
                           if _fast_sel:
                               from recsa.fast_select import fast_mid_select_union
                               _krot_sel, _, _fs_work = fast_mid_select_union(
                                   q_c, k_layer, prev_selected_parents, cos_cache, sin_cache,
                                   next_selected_parents, t_offset=t0, t_base=_q0 + qc0,
                                   n_prog_t=cl, N_layer=N_layer, layer_power=layer_power, CR=compression_rate,
                                   LOG_N=LOG_N, score_valid_threshold=RECSA_SCORE_VALID_THRESHOLD,
                                   neg_inf=RECSA_SCORE_NEG_INF, scale=scale, ul=_ul_c, krot=_krot_sel,
                                   posq=_posq_mid, work=_fs_work)
                           else:
                               recsa_mid_select_kernel[(cl, B * H_kv)](
                                   q_c, k_layer,
                                   prev_selected_parents,
                                   cos_cache, sin_cache,
                                   next_selected_parents,
                                   mid_imp, mid_tau, t0,
                                   t_base=_q0 + qc0,
                                   b_bias=_lb, HAS_B=_hb,
                                   layer_power=layer_power,
                                   B=B, T=Tq, H=H, H_kv=H_kv,
                                   NUM_GROUPS=NUM_GROUPS, K=K, N_layer=N_layer,
                                   COMPRESSION_RATE=compression_rate,
                                   TOP_K=top_k_per_layer,
                                   LOG_N=LOG_N,
                                   SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                                   NEG_INF=RECSA_SCORE_NEG_INF,
                                   G_PAD=G_PAD, TILE_P=TILE_P,
                                   scale=scale,
                                   t_ptr=t_dev,
                                   HAS_CURSOR=_cursor,
                                   SEL_IEEE=int(SEL_IEEE and Tq > 1 and not _save_for_backward and not _cursor),
                                   force_mask=_fm,
                                   HAS_FORCE=1 if _fm is not None else 0,
                                   num_warps=num_warps, num_stages=num_stages,
                                   **_sel_fast,
                               )
                           if _mid_union:
                               _xm_c = union_xmask(next_selected_parents[:, t0:t0 + cl], _ul_c[0], _ul_c[3],
                                                   compression_rate, ucount=_ul_c[2],
                                                   n_parents=-(-N_layer // compression_rate))
                               mid_acc_union(q_c, _krot_mid, v_layer, prev_selected_parents, _posq_mid,
                                             cos_cache, sin_cache, layer_max, layer_sum, layer_output,
                                             t_offset=t0, t_base=_q0 + qc0, n_prog_t=cl, N_layer=N_layer,
                                             layer_power=layer_power, CR=compression_rate,
                                             neg_inf=RECSA_SCORE_NEG_INF, scale=scale, ul=_ul_c, xmask=_xm_c)
                           else:
                               recsa_mid_accumulate_kernel[(cl, B * H_kv)](
                                   q_c, k_layer, v_layer,
                                   prev_selected_parents,
                                   cos_cache, sin_cache,
                                   layer_max, layer_sum, layer_output,
                                   mid_imp, mid_tau, t0,
                                   t_base=_q0 + qc0,
                                   b_bias=_lb, HAS_B=_hb,
                                   layer_power=layer_power,
                                   B=B, T=Tq, H=H, H_kv=H_kv,
                                   NUM_GROUPS=NUM_GROUPS, K=K, V=V, N_layer=N_layer,
                                   COMPRESSION_RATE=compression_rate,
                                   TOP_K=top_k_per_layer,
                                   LOG_N=LOG_N,
                                   SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                                   NEG_INF=RECSA_SCORE_NEG_INF,
                                   G_PAD=G_PAD, TILE_P=TILE_P,
                                   scale=scale,
                                   t_ptr=t_dev,
                                   HAS_CURSOR=_cursor,
                                   NO_SUMMARIES=_nosum,
                                   DOT_FP16=(ACC_DOT_MODE if (Tq > 1 and not _save_for_backward and not _cursor) else 0),
                                   num_warps=num_warps, num_stages=num_stages,
                               )
                        if _DSPLIT_DEBUG_CAPTURE is not None and Tq == 1:
                            _DSPLIT_DEBUG_CAPTURE.append((
                                layer_idx,
                                mid_imp[:B * H_kv * max_top_nodes].clone(),
                                mid_tau[:B * H_kv].clone(),
                                next_selected_parents.clone()))
                elif mid_ns_eff and (top_k_per_layer <= 64 or mid_ns_wide_eff):
                    assert _fm is None, \
                        "force_masks unsupported on the sortless mid select path"
                    _trace("    Running sortless mid-layer kernel...")
                    next_selected_parents.fill_(-1)
                    chunk = min(Tq, mid_chunk)
                    scratch_elems = chunk * B * H_kv * 3 * top_k_per_layer
                    if mid_scratch is None or mid_scratch.numel() < scratch_elems:
                        mid_scratch = torch.empty(scratch_elems, dtype=torch.float32, device=device)
                        if _ws is not None:
                            _ws["mid_scratch"] = mid_scratch
                    for t0 in range(0, Tq, chunk):
                        cl = min(chunk, Tq - t0)
                        recsa_mid_select_accumulate_ns_kernel[(cl, B * H_kv)](
                            q_c, k_layer, v_layer,
                            prev_selected_parents,
                            cos_cache, sin_cache,
                            layer_max, layer_sum, layer_output,
                            next_selected_parents,
                            mid_scratch, t0,
                            t_base=_q0 + qc0,
                            b_bias=_lb, HAS_B=_hb,
                            layer_power=layer_power,
                            B=B, T=Tq, H=H, H_kv=H_kv,
                            NUM_GROUPS=NUM_GROUPS,
                            K=K, V=V, N_layer=N_layer,
                            COMPRESSION_RATE=compression_rate,
                            TOP_K=top_k_per_layer,
                            LOG_N=LOG_N,
                            SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                            NEG_INF=RECSA_SCORE_NEG_INF,
                            DROP_BLOCK=DROP_BLOCK,
                            G_PAD=G_PAD,
                            TILE_P=TILE_P,
                            NS_FAST=NS_FAST,
                            scale=scale,
                            t_ptr=t_dev,
                            HAS_CURSOR=_cursor,
                            NO_SUMMARIES=_nosum,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                else:
                    assert _fm is None, \
                        "force_masks unsupported on the legacy 1b select path"
                    _trace("    Running select+accumulate kernel (Kernel 1b)...")
                    recsa_select_accumulate_gqa_kernel[grid_kv](
                        q_c, k_layer, v_layer,
                        prev_selected_parents,
                        cos_cache, sin_cache,
                        layer_max, layer_sum, layer_output,
                        next_selected_parents,
                        t_base=_q0 + qc0,
                        b_bias=_lb, HAS_B=_hb,
                        layer_power=layer_power,
                        B=B, T=Tq, H=H, H_kv=H_kv,
                        NUM_GROUPS=NUM_GROUPS,
                        K=K, V=V, N_layer=N_layer,
                        COMPRESSION_RATE=compression_rate,
                        TOP_K=top_k_per_layer,
                        LOG_N=LOG_N,
                        N_DIMS_TOPK=N_DIMS_TOPK,
                        SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                        NEG_INF=RECSA_SCORE_NEG_INF,
                        DROP_BLOCK=DROP_BLOCK,
                        G_PAD=G_PAD,
                        TILE_P=TILE_P,
                        scale=scale,
                        NO_SUMMARIES=_nosum,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
            nvtx.range_pop()

            if _sn is not None and not _sn_done:
                if is_bottom_layer:
                    _sn.leaf(layer_max, 0, layer_sum, (global_max, global_sum))
                else:
                    _sn.level(layer_idx, prev_selected_parents,
                              next_selected_parents, (layer_max, layer_sum), 0)

            # Kernel 2: Merge to Global State
            nvtx.range_push("K2_MergeToGlobal")
            _trace("    Running merge to global kernel (Kernel 2)...")
            if not _chain and not _leaf_fused and not _top_fused:  # fused chain / CUDA epilogues did this
                if _g_fresh:
                    _g_init()
                with _cuda_timer("      Kernel 2"):
                    recsa_merge_to_global_kernel[grid](
                        layer_max, layer_sum, layer_output,
                        global_max, global_sum, global_output,
                        global_max_b, global_sum_b, global_output_b,
                        B=B, T=Tq, H=H, V=V,
                    )
            _g_fresh = False
            # swap: this level's result becomes the next level's accumulator
            global_max, global_max_b = global_max_b, global_max
            global_sum, global_sum_b = global_sum_b, global_sum
            global_output, global_output_b = global_output_b, global_output
            nvtx.range_pop()

            # Swap parent indices: next becomes prev for the layer below.
            if not is_bottom_layer:
                prev_selected_parents, next_selected_parents = next_selected_parents, prev_selected_parents
                _prev_lazy = False

            nvtx.range_pop()
        nvtx.range_pop()

        # --- step 5: final normalize ---
        nvtx.range_push("Phase4_Final_Normalize")
        _trace("Phase 4: Final normalization...")

        if _chain:
            # every level's merge was fused; the bottom one normalized
            assert _n_fused == num_layers, (_n_fused, num_layers)
        elif _leaf_fused:
            pass
        else:
            if _g_fresh:
                _g_init()
                _g_fresh = False
            out_chunk = output if _single_chunk else torch.empty(B, Tq, H, V, dtype=dtype, device=device)
            grid = (Tq, B * H)

            with _cuda_timer("  Kernel 3 (final normalize)"):
                recsa_final_normalize_kernel[grid](
                    global_output, global_sum, out_chunk,
                    B=B, T=Tq, H=H, V=V,
                )
        nvtx.range_pop()

        if not _single_chunk and not _leaf_fused:
            output[:, qc0:qc0 + Tq].copy_(out_chunk)

    _trace("recsa forward pass completed!")

    if _save_for_backward:
        bwd_ctx = dict(
            global_max=global_max,
            global_sum=global_sum,
            per_layer_parents=per_layer_parents,
            layers_k=layers_k,
            layers_v=layers_v,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_layers=num_layers,
            compression_rate=compression_rate,
            top_k_per_layer=top_k_per_layer,
            max_top_nodes=max_top_nodes,
            scale=scale,
            rope_mode=rope_mode,
            NUM_GROUPS=NUM_GROUPS,
            G_PAD=G_PAD,
            TILE_P=TILE_P,
            DROP_BLOCK=DROP_BLOCK,
            packed_top=packed_top_eff,
            QB_packed=QB_sel,
            G_PAD2=G_PAD2,
            packed_warps=packed_warps,
            mid_ns_used=(not mid_split_eff) and mid_ns_eff
                        and (top_k_per_layer <= 64 or mid_ns_wide_eff),
            mid_chunk=mid_chunk,
            no_summaries=no_summaries,
            summary_norm=summary_norm,
        )
        return output, bwd_ctx
    return output


@triton.jit
def recsa_recompute_selection_kernel(
    q,                        # [B, T, H, K]
    layer_k,                  # [B, N_layer, H_kv, K]
    prev_selected_parents,    # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,     # [cache_size, K//2]
    next_selected_parents,
    t_base,
    layer_power: tl.constexpr,
    B,
    T,
    H: tl.constexpr,
    H_kv: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    K: tl.constexpr,
    N_layer,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    LOG_N: tl.constexpr,
    N_DIMS_TOPK: tl.constexpr,
    SCORE_VALID_THRESHOLD: tl.constexpr,
    NEG_INF: tl.constexpr,
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
):
    """Selection-only variant of recsa_select_accumulate_gqa_kernel."""
    i_t_loc = tl.program_id(0)
    i_t = t_base + i_t_loc
    pid_bh_kv = tl.program_id(1)
    i_b = pid_bh_kv // H_kv
    i_h_kv = pid_bh_kv % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))

    assert num_valid_parents > 0, "No valid candidates found"

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t_loc.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH

    idx_mask = (1 << LOG_N) - 1
    rightmost_pos = (num_candidates - 1).to(tl.int32)
    topk_running_max = tl.full([1], NEG_INF, dtype=tl.float32)

    init_idx = tl.arange(0, TOP_K).to(tl.int32)
    init_imp = tl.full([TOP_K], NEG_INF, dtype=tl.float32)
    init_int = init_imp.to(tl.int32, bitcast=True)
    init_encoded = (init_int & ~idx_mask) | (init_idx & idx_mask)
    running_topk_encoded = init_encoded.to(tl.float32, bitcast=True)

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = (PARENTS_PER_BATCH + TILE_P - 1) // TILE_P
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE
    batch_rows = tl.arange(0, PARENTS_PER_BATCH).to(tl.int32)
    tile_rows = tl.arange(0, TILE_P).to(tl.int32)
    row_idx = tl.arange(0, 2)[:, None]

    for i_batch in range(num_batches):
        # ---- step 1: score computation + importance ----
        importance_matrix = tl.full([PARENTS_PER_BATCH, COMPRESSION_RATE], NEG_INF, dtype=tl.float32)
        batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)

        for i_tile in range(PARENT_TILES):
            tile_base = (i_tile * TILE_P).to(tl.int32)
            parent_off_local = tile_base + parent_in_tile
            parent_off = batch_parent_base + parent_off_local

            safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
            parent_idx = tl.load(
                prev_selected_parents + prev_sel_base + safe_poff
            ).to(tl.int32)

            poff_valid = (
                (parent_off_local < PARENTS_PER_BATCH)
                & (parent_off < num_valid_parents)
                & (parent_off < TOP_K)
                & (parent_idx >= 0)
            )
            is_rm = parent_idx == rightmost_parent_idx
            ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
            child_rows = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_in_tile
            child_valid = poff_valid & ch_ok & (child_rows >= 0) & (child_rows < N_layer)

            safe_child_rows = tl.minimum(tl.maximum(child_rows, 0), N_layer - 1).to(tl.int64)
            k_row = k_base + safe_child_rows[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            rope_pos = parent_off.to(tl.int64) * COMPRESSION_RATE + child_in_tile.to(tl.int64)
            rope_pos = tl.maximum(rope_pos, 0)
            cos_k = tl.load(cos_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            sin_k = tl.load(sin_cache + rope_pos[:, None] * (K // 2) + o_k_half[None, :])
            k_rope_lo = k_lo * cos_k - k_hi * sin_k
            k_rope_hi = k_lo * sin_k + k_hi * cos_k

            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            scores = tl.where(child_valid[None, :] & g_valid[:, None], scores, NEG_INF)

            local_m = tl.max(scores)
            new_batch_m = tl.maximum(batch_m, local_m)
            rescale_batch_old = tl.exp(batch_m - new_batch_m)
            importance_matrix = tl.where(importance_matrix >= 0, importance_matrix * rescale_batch_old, importance_matrix)
            batch_m = new_batch_m

            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            importance_tile_2d = tl.reshape(importance_tile, [TILE_P, COMPRESSION_RATE])

            for i_r in tl.static_range(TILE_P):
                row_off = (tile_base + i_r).to(tl.int32)
                row_mask_in_tile = (tile_rows == i_r)[:, None]
                row_vals = tl.sum(tl.where(row_mask_in_tile, importance_tile_2d, 0.0), axis=0)
                row_vals = tl.where(row_off < PARENTS_PER_BATCH, row_vals, NEG_INF)
                is_row = (batch_rows == row_off)[:, None]
                importance_matrix = tl.where(is_row, row_vals[None, :], importance_matrix)

        # ---- step 2: Top-K merge sort ----
        importance_flat = tl.reshape(importance_matrix, [BC])
        buf_off = (i_batch * BC).to(tl.int32)
        pos = buf_off + tl.arange(0, BC).to(tl.int32)
        valid_pos = pos < num_candidates
        importance_flat = tl.where(valid_pos, importance_flat, NEG_INF)

        sel_m_new = tl.maximum(topk_running_max, batch_m)
        rescale_run = tl.exp(topk_running_max - sel_m_new)
        rescale_batch = tl.exp(batch_m - sel_m_new)

        run_int = running_topk_encoded.to(tl.int32, bitcast=True)
        run_raw_idx = run_int & idx_mask
        run_clean_int = run_int & ~idx_mask
        run_clean_imp = run_clean_int.to(tl.float32, bitcast=True)
        run_rescaled_imp = tl.where(run_clean_imp >= 0, run_clean_imp * rescale_run, run_clean_imp)
        run_rescaled_int = run_rescaled_imp.to(tl.int32, bitcast=True)
        run_rescaled_encoded = (run_rescaled_int & ~idx_mask) | run_raw_idx
        running_topk_encoded = run_rescaled_encoded.to(tl.float32, bitcast=True)

        importance_flat = tl.where(importance_flat >= 0, importance_flat * rescale_batch, importance_flat)
        importance_flat = tl.where(pos == rightmost_pos, 1e6, importance_flat)
        topk_running_max = sel_m_new

        importance_int = importance_flat.to(tl.int32, bitcast=True)
        encoded_idx = tl.where(importance_flat >= 0, ~pos, pos) & idx_mask
        encoded_int = (importance_int & ~idx_mask) | encoded_idx
        batch_encoded = encoded_int.to(tl.float32, bitcast=True)

        running_b = tl.broadcast_to(running_topk_encoded[None, :], [2, TOP_K])
        batch_b = tl.broadcast_to(batch_encoded[None, :], [2, TOP_K])
        merged_2d = tl.where(row_idx == 0, running_b, batch_b)
        merged_input = tl.reshape(merged_2d, [2 * TOP_K])

        N_DIMS_MERGE: tl.constexpr = N_DIMS_TOPK + 1
        sorted_merged = tl.sort(merged_input, descending=True)
        merged_sorted_2d = tl.reshape(sorted_merged, [2, TOP_K])

        running_topk_encoded = tl.sum(tl.where(row_idx == 0, merged_sorted_2d, 0.0), axis=0)

    # ---- Decode final Top-K -> next_selected_parents ----
    topk_int = running_topk_encoded.to(tl.int32, bitcast=True)
    raw_idx = topk_int & idx_mask
    clean_int = topk_int & ~idx_mask
    clean_imp = clean_int.to(tl.float32, bitcast=True)
    topk_pos = tl.where(clean_imp >= 0, ~raw_idx, raw_idx)
    topk_pos = (topk_pos & idx_mask).to(tl.int32)
    topk_valid = (clean_imp > SCORE_VALID_THRESHOLD) & (topk_pos < num_candidates)
    topk_pos = tl.where(topk_valid, topk_pos, -1).to(tl.int32)

    parent_off_f = tl.where(topk_pos >= 0, topk_pos // COMPRESSION_RATE, 0).to(tl.int32)
    child_slot_f = tl.where(topk_pos >= 0, topk_pos - parent_off_f * COMPRESSION_RATE, 0).to(tl.int32)
    parent_idx_f = tl.load(
        prev_selected_parents + prev_sel_base + parent_off_f.to(tl.int64),
        mask=topk_pos >= 0,
        other=-1,
    ).to(tl.int32)
    selected_node_indices = tl.where(
        topk_pos >= 0,
        parent_idx_f * COMPRESSION_RATE + child_slot_f,
        -1,
    ).to(tl.int32)

    MAX_IDX: tl.constexpr = 2147483647
    selected_sorted = tl.where(selected_node_indices >= 0, selected_node_indices, MAX_IDX)
    selected_sorted = tl.sort(selected_sorted, descending=False)
    selected_sorted = tl.where(selected_sorted < MAX_IDX, selected_sorted, -1)

    out_par_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t_loc.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    tl.store(next_selected_parents + out_par_base + tl.arange(0, TOP_K), selected_sorted.to(tl.int32))


@triton.jit
def recsa_leaf_store_bucket_kernel(
    pool_k,   # [1, cap, H_kv, K] pinned host (UVA-mapped)
    pool_v,   # [1, cap, H_kv, V] pinned host
    stage_k,
    stage_v,  # [1, C, H_kv, V] device, OUT
    k_t,      # [1, 1, H_kv, K] device, contiguous
    v_t,      # [1, 1, H_kv, V] device, contiguous
    t_ptr,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,  # compression rate = bucket width (power of 2)
):
    """Write-through leaf store of row t plus a device copy of the open level-1 bucket, so the MLP cursor pool (torch
    ops) reads its level-1 children from device memory."""
    h = tl.program_id(0)
    t = tl.load(t_ptr)
    lo = (t // C) * C
    j = tl.arange(0, C)
    rows = lo + j
    prev = rows < t
    cur = rows == t
    o_k = tl.arange(0, K)
    k_row = tl.load(k_t + h.to(tl.int64) * K + o_k)
    tl.store(pool_k + (t * H_kv + h) * K + o_k, k_row)
    kb = tl.load(pool_k + ((rows * H_kv + h) * K)[:, None] + o_k[None, :],
                 mask=prev[:, None], other=0.0)
    kb = tl.where(cur[:, None], k_row[None, :], kb)
    tl.store(stage_k + ((j * H_kv + h) * K)[:, None] + o_k[None, :], kb)
    o_v = tl.arange(0, V)
    v_row = tl.load(v_t + h.to(tl.int64) * V + o_v)
    tl.store(pool_v + (t * H_kv + h) * V + o_v, v_row)
    vb = tl.load(pool_v + ((rows * H_kv + h) * V)[:, None] + o_v[None, :],
                 mask=prev[:, None], other=0.0)
    vb = tl.where(cur[:, None], v_row[None, :], vb)
    tl.store(stage_v + ((j * H_kv + h) * V)[:, None] + o_v[None, :], vb)


import collections  # noqa: E402

SUMMARY_NORM_RULES = ("mass_cap",)
SUMMARY_NORM_ARCHIVED_RULES = ("count_clamp", "static", "equalize_clamp", "combo")
SUMMARY_NORM_ARCHIVE_TAG = "not part of this package"

_SumNorm = collections.namedtuple("_SumNorm", "rule tau cap_nref")


def summary_norm_archived_message(rule) -> str:
    """Error text for a rule outside SUMMARY_NORM_RULES."""
    return f"rule {rule!r}: {SUMMARY_NORM_ARCHIVE_TAG}"


def summary_norm_spec(sn):
    """Validate a canonical `summary_norm` config dict."""
    if sn is None:
        return None
    if not isinstance(sn, dict):
        raise ValueError(f"summary_norm must be a dict or None, got {sn!r}")
    rule = sn.get("rule")
    if rule == "mass_cap":
        if not {"rule", "tau"} <= set(sn) <= {"rule", "tau", "n_ref"}:
            raise ValueError(f"summary_norm mass_cap takes the keys rule, tau and optionally n_ref "
                             f"(the gate); it is not combinable with any other rule (count_clamp, "
                             f"static, equalize_clamp and combo are "
                             f"{SUMMARY_NORM_ARCHIVE_TAG}); got {sorted(sn)}")
        tau = sn["tau"]
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) \
                or not 0.0 < tau < 1.0:
            raise ValueError(f"summary_norm mass_cap tau must be a number with 0 < tau < 1, got {tau!r}")
        gate = sn.get("n_ref")
        if "n_ref" in sn:
            if isinstance(gate, bool) or not isinstance(gate, int) or gate < 1:
                raise ValueError(f"summary_norm mass_cap n_ref (the gate) must be an int >= 1, "
                                 f"got {gate!r}")
            gate = int(gate)
        return _SumNorm("mass_cap", tau=float(tau), cap_nref=gate)
    if rule in SUMMARY_NORM_ARCHIVED_RULES:
        raise ValueError(f"summary_norm {summary_norm_archived_message(rule)}")
    raise ValueError(f"summary_norm rule must be one of {SUMMARY_NORM_RULES}, got {rule!r}")


def canonical_summary_norm(sn):
    """Canonical dict of a valid summary_norm setting (None stays None)."""
    spec = summary_norm_spec(sn)
    if spec is None:
        return None
    return (dict(rule="mass_cap", tau=spec.tau) if spec.cap_nref is None
            else dict(rule="mass_cap", tau=spec.tau, n_ref=spec.cap_nref))


def _sumnorm_prepare(spec, num_layers: int):
    """Per-call summary_norm setup: None for a single-level tree (no summary level), else the spec."""
    if spec is None or num_layers == 1:
        return None
    return spec

_SUMNORM_DEBUG_CAPTURE = None


@triton.jit
def recsa_sumnorm_level_kernel(
    prev_sel,
    next_sel,
    cnt_lvl,  # int32 [B*Tq*H_kv] OUT: n_l (MODE 2)
    cnt_tot,
    state_max,
    # state_max: TARGET 0 layer_max [B, Tq, H], TARGET 1 slice partials part_max [NSLICE * B*H_kv * G_PAD]
    log_odds,  # fp32 runtime scalar: MODE 5 ln(tau / (1 - tau))
    t_base,
    T,  # Tq (rows of the lists / layer state)
    t_ptr=None,
    state_sum=None,
    sg_max=None,
    sg_sum=None,  # fp32, sg_max's layout
    sp_max=None,
    sp_sum=None,  # fp32, sp_max's layout
    mass_out=None,
    cnt_gate=None,
    n_gate=0,
    layer_power: tl.constexpr = 1,
    H_kv: tl.constexpr = 1,
    NUM_GROUPS: tl.constexpr = 1,
    G_PAD: tl.constexpr = 16,
    NSLICE: tl.constexpr = 1,
    TOP_K: tl.constexpr = 256,
    COMPRESSION_RATE: tl.constexpr = 16,
    MODE: tl.constexpr = 2,
    FIRST: tl.constexpr = 0,
    TARGET: tl.constexpr = 0,
    HAS_CURSOR: tl.constexpr = 0,
    HAS_SG: tl.constexpr = 0,
    HAS_SP: tl.constexpr = 0,
    HAS_MOUT: tl.constexpr = 0,
    GATE: tl.constexpr = 0,
):
    """summary_norm mass_cap: count one level's summary population (the gate) or cap the summary mass at the leaf site."""
    i_t_loc = tl.program_id(0)
    pid_bh = tl.program_id(1)
    NBH = tl.num_programs(1)
    i_b = pid_bh // H_kv
    i_h = pid_bh % H_kv
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
    i_t = t_base + i_t_loc
    row = i_b.to(tl.int64) * T + i_t_loc
    cidx = row * H_kv + i_h

    if MODE == 5:
        g = tl.arange(0, G_PAD)
        g_ok = g < NUM_GROUPS
        hp = row * (H_kv * NUM_GROUPS) + i_h * NUM_GROUPS + g
        ninf = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
        # summaries: max frame, then the sum (empty pairs contribute nothing)
        s_mx = ninf
        if HAS_SG:
            a_m = tl.load(sg_max + hp, mask=g_ok, other=0.0)
            a_s = tl.load(sg_sum + hp, mask=g_ok, other=0.0)
            s_mx = tl.maximum(s_mx, tl.where(a_s > 1e-10, a_m, ninf))
        if HAS_SP:
            for s in tl.static_range(NSLICE):
                pp = (s * NBH + pid_bh).to(tl.int64) * G_PAD + g
                b_s = tl.load(sp_sum + pp, mask=g_ok, other=0.0)
                s_mx = tl.maximum(s_mx, tl.where(b_s > 1e-10, tl.load(sp_max + pp, mask=g_ok, other=0.0), ninf))
        s_ref = tl.where(s_mx > ninf, s_mx, 0.0)
        s_z = tl.zeros([G_PAD], dtype=tl.float32)
        if HAS_SG:
            s_z += tl.where(a_s > 1e-10, a_s * tl.exp(a_m - s_ref), 0.0)
        if HAS_SP:
            for s in tl.static_range(NSLICE):
                pp = (s * NBH + pid_bh).to(tl.int64) * G_PAD + g
                b_m = tl.load(sp_max + pp, mask=g_ok, other=0.0)
                b_s = tl.load(sp_sum + pp, mask=g_ok, other=0.0)
                s_z += tl.where(b_s > 1e-10, b_s * tl.exp(b_m - s_ref), 0.0)
        # leaves: the state the cap shifts
        if TARGET == 0:
            l_m = tl.load(state_max + hp, mask=g_ok, other=0.0)
            l_s = tl.load(state_sum + hp, mask=g_ok, other=0.0)
            l_ref = tl.where(l_s > 1e-10, l_m, 0.0)
            l_z = tl.where(l_s > 1e-10, l_s, 0.0)
        else:
            l_mx = ninf
            for s in tl.static_range(NSLICE):
                pp = (s * NBH + pid_bh).to(tl.int64) * G_PAD + g
                c_s = tl.load(state_sum + pp, mask=g_ok, other=0.0)
                l_mx = tl.maximum(l_mx, tl.where(c_s > 1e-10, tl.load(state_max + pp, mask=g_ok, other=0.0), ninf))
            l_ref = tl.where(l_mx > ninf, l_mx, 0.0)
            l_z = tl.zeros([G_PAD], dtype=tl.float32)
            for s in tl.static_range(NSLICE):
                pp = (s * NBH + pid_bh).to(tl.int64) * G_PAD + g
                c_m = tl.load(state_max + pp, mask=g_ok, other=0.0)
                c_s = tl.load(state_sum + pp, mask=g_ok, other=0.0)
                l_z += tl.where(c_s > 1e-10, c_s * tl.exp(c_m - l_ref), 0.0)
        has = (s_z > 0.0) & (l_z > 0.0)
        y = (s_ref - l_ref) + tl.log(tl.where(has, s_z, 1.0)) - tl.log(tl.where(has, l_z, 1.0))
        dlt = tl.where(has & (y > log_odds), y - log_odds, 0.0)
        if GATE:
            dlt = tl.where(tl.load(cnt_gate + cidx) > n_gate, dlt, 0.0)
        if HAS_MOUT:
            mass = tl.where(has, 1.0 / (1.0 + tl.exp(-y)), tl.where(s_z > 0.0, 1.0, 0.0))
            tl.store(mass_out + hp, mass, mask=g_ok)
        if TARGET == 0:
            tl.store(state_max + hp, tl.where(dlt != 0.0, l_m + dlt, l_m), mask=g_ok)
        else:
            for s in tl.static_range(NSLICE):
                pp = (s * NBH + pid_bh).to(tl.int64) * G_PAD + g
                c_m = tl.load(state_max + pp, mask=g_ok, other=0.0)
                c_s = tl.load(state_sum + pp, mask=g_ok, other=0.0)
                tl.store(state_max + pp, tl.where((dlt != 0.0) & (c_s > 1e-10), c_m + dlt, c_m), mask=g_ok)
    else:
        o_k = tl.arange(0, TOP_K).to(tl.int64)
        par = tl.load(prev_sel + cidx * TOP_K + o_k)
        sel = tl.load(next_sel + cidx * TOP_K + o_k)
        ri = i_t // layer_power
        nc = tl.minimum(tl.maximum(ri - COMPRESSION_RATE * par + 1, 0), COMPRESSION_RATE)
        nc = tl.where(par >= 0, nc, 0)
        n = tl.sum(nc, axis=0) - tl.sum((sel >= 0).to(tl.int32), axis=0)
        n = tl.maximum(n, 0)
        tl.store(cnt_lvl + cidx, n)
        if FIRST:
            tl.store(cnt_tot + cidx, n)
        else:
            tl.store(cnt_tot + cidx, tl.load(cnt_tot + cidx) + n)


class _SumNormRun:
    """Per-chunk driver of the summary_norm level kernel (mass_cap): level() counts a summary level for a gated cap,
    leaf() applies the cap at the leaf site (MODE 5)."""
    __slots__ = ("spec", "cnt_lvl", "cnt_tot", "num_layers", "B", "Tq", "H_kv",
                 "NUM_GROUPS", "G_PAD", "top_k", "c", "t_base", "t_dev", "log_odds", "gate")

    def __init__(self, spec, *, cnt_lvl, cnt_tot, num_layers, B, Tq, H_kv, NUM_GROUPS,
                 G_PAD, top_k, compression_rate, t_base, t_dev):
        self.spec, self.cnt_lvl, self.cnt_tot = spec, cnt_lvl, cnt_tot
        self.num_layers, self.B, self.Tq, self.H_kv = num_layers, B, Tq, H_kv
        self.NUM_GROUPS, self.G_PAD, self.top_k, self.c = NUM_GROUPS, G_PAD, top_k, compression_rate
        self.t_base, self.t_dev = int(t_base), t_dev
        self.log_odds = math.log(spec.tau) - math.log1p(-spec.tau)
        self.gate = cnt_tot if spec.cap_nref is not None else None

    def _launch(self, *, layer_idx, prev_sel, next_sel, target, first=0):
        """MODE 2: count one summary level into cnt_lvl and add it to the running n_S in cnt_tot."""
        recsa_sumnorm_level_kernel[(self.Tq, self.B * self.H_kv)](
            prev_sel, next_sel, self.cnt_lvl, self.cnt_tot,
            None, 0.0, self.t_base, self.Tq,
            t_ptr=self.t_dev,
            layer_power=self.c ** layer_idx, H_kv=self.H_kv, NUM_GROUPS=self.NUM_GROUPS,
            G_PAD=self.G_PAD, NSLICE=RECSA_DSPLIT_NSLICE if target == 1 else 1,
            TOP_K=self.top_k, COMPRESSION_RATE=self.c, MODE=2, FIRST=first,
            TARGET=target, HAS_CURSOR=1 if self.t_dev is not None else 0,
            num_warps=1,
        )

    def level(self, layer_idx, prev_sel, next_sel, state, target):
        """Count one summary level for a gated cap; mass_cap never shifts a summary level."""
        if self.spec.cap_nref is None:
            return          # no count, no shift: the cap acts at the leaf site
        top = 1 if layer_idx == self.num_layers - 1 else 0
        # n_ref: count only (MODE 2, no shift), cnt_tot = sum_l n_l
        self._launch(layer_idx=layer_idx, prev_sel=prev_sel, next_sel=next_sel, target=target,
                     first=top)
        if _SUMNORM_DEBUG_CAPTURE is not None:
            _SUMNORM_DEBUG_CAPTURE.append(
                (layer_idx, self.cnt_lvl.view(self.B, self.Tq, self.H_kv).clone()))

    def hfuse_sites(self, acc, bot, prev_of, part, merged=None):
        """Count / shift sites for _dsplit_hfuse_levels, called right after the launch holding the accumulate role of
        level `acc` and before finalize(acc - 1) overwrites prev_of[acc]."""
        self.level(acc, prev_of[acc], prev_of[acc - 1], part[acc][:2], 1)
        if bot:
            self.leaf(part[0][0], 1, part[0][1],
                      merged[:2] if merged is not None else None, part[1][:2])

    def leaf(self, state_max, target, state_sum=None, merged=None, unmerged=None):
        """Apply the cap at the leaf site, before the leaves' merge."""
        self._mass_cap(state_max, state_sum, target, merged, unmerged)

    def _mass_cap(self, state_max, state_sum, target, merged, unmerged):
        """MODE 5: per-query-head summary mass cap, gated on n_S when the rule has an n_ref."""
        assert state_sum is not None and (merged is not None or unmerged is not None), \
            "mass_cap needs the leaf state's sums and every summary state"
        H = self.H_kv * self.NUM_GROUPS
        mo = None
        if _SUMNORM_DEBUG_CAPTURE is not None:
            mo = torch.empty(self.B * self.Tq * H, dtype=torch.float32, device=state_max.device)
            if self.gate is not None:
                _SUMNORM_DEBUG_CAPTURE.append(
                    ("gate", self.gate.view(self.B, self.Tq, self.H_kv).clone()))
        recsa_sumnorm_level_kernel[(self.Tq, self.B * self.H_kv)](
            None, None, self.cnt_lvl, self.cnt_tot,
            state_max, float(self.log_odds), 0, self.Tq,
            state_sum=state_sum,
            sg_max=merged[0] if merged is not None else None,
            sg_sum=merged[1] if merged is not None else None,
            sp_max=unmerged[0] if unmerged is not None else None,
            sp_sum=unmerged[1] if unmerged is not None else None,
            mass_out=mo,
            cnt_gate=self.gate,
            n_gate=int(self.spec.cap_nref) if self.gate is not None else 0,
            H_kv=self.H_kv, NUM_GROUPS=self.NUM_GROUPS, G_PAD=self.G_PAD,
            NSLICE=RECSA_DSPLIT_NSLICE if (target == 1 or unmerged is not None) else 1,
            TOP_K=self.top_k, COMPRESSION_RATE=self.c, MODE=5, TARGET=target,
            HAS_CURSOR=0, HAS_SG=1 if merged is not None else 0,
            HAS_SP=1 if unmerged is not None else 0, HAS_MOUT=1 if mo is not None else 0,
            GATE=1 if self.gate is not None else 0,
            num_warps=1,
        )
        if mo is not None:
            _SUMNORM_DEBUG_CAPTURE.append(("mass", mo.view(self.B, self.Tq, H)))


RECSA_GATHER_MIRROR_PROGS = int(os.environ.get("RECSA_GATHER_MIRROR_PROGS", "1024"))


@triton.jit
def recsa_leaf_gather_mirror_kernel(
    k_pool,  # [1, cap, H_kv, K] pinned host (UVA-mapped)
    v_pool,  # [1, cap, H_kv, V] pinned host
    mir_k,   # [1, cap - R0, H_kv, K] device: pool rows >= R0
    mir_v,   # [1, cap - R0, H_kv, V] device
    fetch_k,  # [H_kv, TOP_K*CR, K] device staging
    fetch_v,  # [H_kv, TOP_K*CR, V] device staging
    prev_selected_parents,  # [1, 1, H_kv, TOP_K] int32, contiguous
    t_base,  # host int: the query's global row (eager decode)
    R0,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOP_K: tl.constexpr,
    NPROG: tl.constexpr,
    t_ptr=None,  # int64 [1] session cursor (graph mode)
    HAS_CURSOR: tl.constexpr = 0,
):
    """recsa_leaf_gather_kernel with a device source for mirrored buckets."""
    pid = tl.program_id(0)
    if HAS_CURSOR:
        t_base = tl.load(t_ptr).to(tl.int32)
    N_layer = t_base + 1  # leaf count == the query's global row + 1
    o_ch = tl.arange(0, COMPRESSION_RATE)
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    TOTAL: tl.constexpr = H_kv * TOP_K
    for w in range(pid, TOTAL, NPROG):
        h = w // TOP_K
        p_idx = tl.load(prev_selected_parents + w).to(tl.int32)
        pb = tl.maximum(p_idx, 0)
        rows = pb * COMPRESSION_RATE + o_ch
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)
        dst = w.to(tl.int64) * COMPRESSION_RATE
        if pb * COMPRESSION_RATE >= R0:
            m_rows = safe_rows - R0
            tile_k = tl.load(mir_k + m_rows[:, None] * (H_kv * K)
                             + h.to(tl.int64) * K + o_k[None, :])
            tile_v = tl.load(mir_v + m_rows[:, None] * (H_kv * V)
                             + h.to(tl.int64) * V + o_v[None, :])
        else:
            tile_k = tl.load(k_pool + safe_rows[:, None] * (H_kv * K)
                             + h.to(tl.int64) * K + o_k[None, :])
            tile_v = tl.load(v_pool + safe_rows[:, None] * (H_kv * V)
                             + h.to(tl.int64) * V + o_v[None, :])
        tl.store(fetch_k + dst * K + o_ch[:, None] * K + o_k[None, :], tile_k)
        tl.store(fetch_v + dst * V + o_ch[:, None] * V + o_v[None, :], tile_v)


@triton.jit
def recsa_leaf_mirror_store_kernel(
    mir_k,   # [1, cap - R0, H_kv, K] device
    mir_v,   # [1, cap - R0, H_kv, V] device
    k_t,
    v_t,     # [1, 1, H_kv, V] device, contiguous
    t_ptr,
    R0,      # host int: first mirrored row
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """Mirror half of the write-through leaf append: row t goes to mirror row t - R0 when t >= R0."""
    h = tl.program_id(0)
    t = tl.load(t_ptr)
    if t >= R0:
        r = t - R0
        o_k = tl.arange(0, K)
        tl.store(mir_k + (r * H_kv + h) * K + o_k,
                 tl.load(k_t + h.to(tl.int64) * K + o_k))
        o_v = tl.arange(0, V)
        tl.store(mir_v + (r * H_kv + h) * V + o_v,
                 tl.load(v_t + h.to(tl.int64) * V + o_v))


_LEAF_MIRRORS: dict = {}


def register_leaf_mirror(pool_k, mirror) -> int:
    """Register (mir_k, mir_v, R0) for the offloaded leaf K buffer pool_k; returns the key (its storage address)."""
    import weakref
    key = pool_k.untyped_storage().data_ptr()
    _LEAF_MIRRORS[key] = (weakref.ref(pool_k), mirror)
    return key


def unregister_leaf_mirror(key: int) -> None:
    _LEAF_MIRRORS.pop(key, None)


def _leaf_mirror_of(t):
    """The (mir_k, mir_v, R0) mirror registered for an offloaded leaf K buffer or a view of it, else None."""
    if not _LEAF_MIRRORS:
        return None
    e = _LEAF_MIRRORS.get(t.untyped_storage().data_ptr())
    if e is None or e[0]() is None:
        return None
    return e[1]


def _leaf_mirror_gather(k_pool, v_pool, fetch_k, fetch_v, prev, t_base, H_kv, K, V, CR, top_k,
                        t_dev, cursor):
    """Launch recsa_leaf_gather_mirror_kernel for a leaf pool with an attached mirror."""
    mir_k, mir_v, r0 = _leaf_mirror_of(k_pool)
    recsa_leaf_gather_mirror_kernel[(RECSA_GATHER_MIRROR_PROGS,)](
        k_pool, v_pool, mir_k, mir_v, fetch_k, fetch_v, prev,
        t_base=t_base, R0=r0, H_kv=H_kv, K=K, V=V, COMPRESSION_RATE=CR, TOP_K=top_k,
        NPROG=RECSA_GATHER_MIRROR_PROGS, t_ptr=t_dev, HAS_CURSOR=cursor, num_warps=4)


__all__ = [
    'recsa_forward',
    'recsa_build_kernel',
    'recsa_leaf_store_kernel',
    'recsa_leaf_gather_kernel',
    'recsa_leaf_gather_mirror_kernel',
    'recsa_leaf_mirror_store_kernel',
    'register_leaf_mirror',
    'unregister_leaf_mirror',
    'recsa_bottom_accumulate_gqa_kernel',
    'recsa_select_accumulate_gqa_kernel',
    'recsa_top_select_kernel',
    'recsa_top_accumulate_gqa_kernel',
    'recsa_recompute_selection_kernel',
    'recsa_merge_to_global_kernel',
    'recsa_final_normalize_kernel',
    'RECSA_SCORE_NEG_INF',
    'RECSA_SCORE_VALID_THRESHOLD',
    'SUMMARIES_MODES',
    'summaries_off',
    'SUMMARY_NORM_RULES',
    'SUMMARY_NORM_ARCHIVED_RULES',
    'SUMMARY_NORM_ARCHIVE_TAG',
    'summary_norm_spec',
    'canonical_summary_norm',
    'summary_norm_archived_message',
    'recsa_sumnorm_level_kernel',
]


from recsa.dsplit_finalize_v3 import recsa_dsplit_select_finalize_v3_kernel, sel_kwargs as _fin3_sel_kwargs  # noqa: E402

_DSPLIT_FIN3_WARPS = 4
_DSPLIT_FIN3_PTX = True


def _dsplit_select_finalize_v3(mode, grid, *args, **kwargs):
    """Launch recsa_dsplit_select_finalize_v3_kernel (the default RECSA_DSPLIT_FINALIZE); any other mode raises."""
    if mode != "v3":
        raise ValueError(f"RECSA_DSPLIT_FINALIZE must be 'v2', 'v3' or 'legacy', got {mode!r}")
    sel = _fin3_sel_kwargs(kwargs["TOP_K"], kwargs["COMPRESSION_RATE"], _DSPLIT_FIN3_WARPS, _DSPLIT_FIN3_PTX)
    return recsa_dsplit_select_finalize_v3_kernel[grid](*args, num_warps=_DSPLIT_FIN3_WARPS, **sel, **kwargs)
