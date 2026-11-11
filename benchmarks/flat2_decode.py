"""Forced two-level RecSA decode step (benchmark only, opt-in)."""
from __future__ import annotations

import math
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import recsa.parallel as P  # noqa: E402

NEG_INF = P.RECSA_SCORE_NEG_INF          # -1e10, the kernels' masked-score value
LSE_INVALID = -1.0e30                    # importance of a node beyond the causal frontier
LSE_FORCED = 1.0e30                      # importance of the forced rightmost node
TILE_N = 64
SUB_N = 16
BOTTOM_NSLICE = P.RECSA_DSPLIT_NSLICE    # 16: the default bottom / merge-pair slice count

FLAT2 = SimpleNamespace(on=False, installed=False, orig=None, tag=0, mode="fast", slices=None,
                        score_tiles_per_prog=4, refresh_tiles=2, sub=SUB_N,
                        compare=None, last=None, kcache={}, kcache_pool=[], kcache_builds=0)
MODES = ("fast", "exact")


@triton.jit
def _rotate_rows(k_base, safe_rows, o_k_half, cos_off, sin_off, cos_cache, sin_cache, n0,
                 H_kv: tl.constexpr, K: tl.constexpr):
    """K rotation of one 64-node tile as in the accumulate kernel: tile offset (o_n), then tile base (n0)."""
    k_row = k_base + safe_rows[:, None] * (H_kv * K)
    k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
    k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)
    base_pos = n0.to(tl.int64)
    cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
    sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
    k1_lo = k_lo * cos_off - k_hi * sin_off
    k1_hi = k_lo * sin_off + k_hi * cos_off
    k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]
    k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]
    return k_rope_lo, k_rope_hi


@triton.jit
def flat2_cache_build_kernel(
    layer_k,  # [1, N_cap, H_kv, K] level-1 keys
    kcache,
    cos_cache, sin_cache,
    N_cap,
    H_kv: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
):
    """One-time build of the rotated-key cache over the whole capacity."""
    tile = tl.program_id(0)
    h = tl.program_id(1)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_n = tl.arange(0, TILE).to(tl.int32)
    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    n0 = tile * TILE
    rows = n0 + o_n
    ok = rows < N_cap
    safe = tl.minimum(rows, N_cap - 1).to(tl.int64)
    lo, hi = _rotate_rows(layer_k + h.to(tl.int64) * K, safe, o_k_half, cos_off, sin_off, cos_cache, sin_cache,
                          n0, H_kv, K)
    cb = kcache + safe[:, None] * (H_kv * K) + h.to(tl.int64) * K
    tl.store(cb + o_k_half[None, :], lo.to(tl.float16), mask=ok[:, None])
    tl.store(cb + (K // 2) + o_k_half[None, :], hi.to(tl.float16), mask=ok[:, None])


@triton.jit
def flat2_score_kernel(
    q,  # [1, 1, H, K] model dtype (post q_norm, unrotated)
    layer_k,  # [1, N_cap, H_kv, K] level-1 keys (capacity buffer)
    cos_cache, sin_cache,  # [cache, K//2] fp32
    s_out,
    lse_out,
    t_ptr,  # int64 [1]: the query's global row
    N_cap,
    scale,
    H: tl.constexpr, H_kv: tl.constexpr, NUM_GROUPS: tl.constexpr, K: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr, G_PAD: tl.constexpr, TILE: tl.constexpr, TPP: tl.constexpr,
    NEG: tl.constexpr, LSE_INV: tl.constexpr, LSE_FORCE: tl.constexpr,
):
    """Exact-mode scoring of every level-1 node, with the kernels' q / K rotation and tf32 dot."""
    pid_n = tl.program_id(0)
    i_h_kv = tl.program_id(1)
    t = tl.load(t_ptr).to(tl.int32)
    N1 = t // COMPRESSION_RATE + 1
    rope_pos_q = N1 - 1

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    q_ptrs = q + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    o_n = tl.arange(0, TILE).to(tl.int32)
    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    k_base = layer_k + i_h_kv.to(tl.int64) * K
    s_base = s_out + i_h_kv.to(tl.int64) * NUM_GROUPS * N_cap
    for i in range(TPP):
        tile = pid_n * TPP + i
        n0 = tile * TILE
        rows = n0 + o_n
        valid = rows < N1
        in_cap = rows < N_cap
        safe_rows = tl.minimum(rows, N1 - 1).to(tl.int64)
        k_rope_lo, k_rope_hi = _rotate_rows(k_base, safe_rows, o_k_half, cos_off, sin_off, cos_cache,
                                            sin_cache, n0, H_kv, K)
        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG)
        tl.store(s_base + g_ids[:, None] * N_cap + rows.to(tl.int64)[None, :], scores,
                 mask=g_valid[:, None] & in_cap[None, :])
        m = tl.max(scores, axis=0)
        e = tl.where(g_valid[:, None], tl.exp(scores - m[None, :]), 0.0)
        lse = m + tl.log(tl.sum(e, axis=0))
        lse = tl.where(valid, lse, LSE_INV)
        lse = tl.where(rows == N1 - 1, LSE_FORCE, lse)
        tl.store(lse_out + i_h_kv.to(tl.int64) * N_cap + rows.to(tl.int64), lse, mask=in_cap)


@triton.jit
def flat2_refresh_kernel(
    layer_k,  # [1, N_cap, H_kv, K] level-1 keys (live)
    kcache,  # [N_cap, H_kv, K] fp16 rotated-key cache (IN/OUT)
    cos_cache, sin_cache,
    t_ptr,
    H_kv: tl.constexpr, K: tl.constexpr, COMPRESSION_RATE: tl.constexpr, TILE: tl.constexpr,
):
    """Fast mode, every step: re-rotate the cache rows of the open node's 64-node tile and the REFRESH_TILES - 1 tiles
    before it from the live keys (the open node's pooled key changes every step)."""
    j = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.load(t_ptr).to(tl.int32)
    N1 = t // COMPRESSION_RATE + 1
    tile = (N1 - 1) // TILE - j
    if tile >= 0:
        o_k_half = tl.arange(0, K // 2).to(tl.int64)
        o_n = tl.arange(0, TILE).to(tl.int32)
        cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
        sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
        n0 = tile * TILE
        rows = n0 + o_n
        ok = rows < N1
        safe = tl.minimum(rows, N1 - 1).to(tl.int64)
        lo, hi = _rotate_rows(layer_k + h.to(tl.int64) * K, safe, o_k_half, cos_off, sin_off, cos_cache,
                              sin_cache, n0, H_kv, K)
        cb = kcache + safe[:, None] * (H_kv * K) + h.to(tl.int64) * K
        tl.store(cb + o_k_half[None, :], lo.to(tl.float16), mask=ok[:, None])
        tl.store(cb + (K // 2) + o_k_half[None, :], hi.to(tl.float16), mask=ok[:, None])


@triton.jit
def flat2_score_cached_kernel(
    q,  # [1, 1, H, K] model dtype (post q_norm, unrotated)
    kcache,  # [N_cap, H_kv, K] fp16 rotated keys
    cos_cache, sin_cache,
    s_out,  # [H_kv, NUM_GROUPS, N_cap] fp32 OUT
    lse_out,  # [H_kv, N_cap] fp32 OUT
    smax_out,
    t_ptr,
    N_cap,
    n_sub,
    scale,
    H_kv: tl.constexpr, NUM_GROUPS: tl.constexpr, K: tl.constexpr, COMPRESSION_RATE: tl.constexpr,
    G_PAD: tl.constexpr, TILE: tl.constexpr, SUB: tl.constexpr,
    NEG: tl.constexpr, LSE_INV: tl.constexpr, LSE_FORCE: tl.constexpr,
):
    """Fast-mode scoring: q rotated once (position N1 - 1), every 64-node tile of rotated fp16 keys scored with one fp16
    tensor-core dot (fp32 accumulate)."""
    tile = tl.program_id(0)
    h = tl.program_id(1)
    t = tl.load(t_ptr).to(tl.int32)
    N1 = t // COMPRESSION_RATE + 1
    pq = N1 - 1
    g = tl.arange(0, G_PAD).to(tl.int64)
    gv = g < NUM_GROUPS
    o_k = tl.arange(0, K).to(tl.int64)
    o_sw = (o_k + K // 2) % K
    o_hf = o_k % (K // 2)
    qp = q + ((h.to(tl.int64) * NUM_GROUPS + g) * gv.to(tl.int64))[:, None] * K
    qf = tl.load(qp + o_k[None, :], mask=gv[:, None], other=0.0).to(tl.float32)
    qs = tl.load(qp + o_sw[None, :], mask=gv[:, None], other=0.0).to(tl.float32)
    cq = tl.load(cos_cache + pq.to(tl.int64) * (K // 2) + o_hf)
    sq = tl.load(sin_cache + pq.to(tl.int64) * (K // 2) + o_hf)
    sgn = tl.where(o_k < K // 2, -1.0, 1.0)
    qr = ((qf * cq[None, :] + (qs * sgn[None, :]) * sq[None, :]) * scale).to(tl.float16)
    o_n = tl.arange(0, TILE).to(tl.int32)
    rows = tile * TILE + o_n
    valid = rows < N1
    in_cap = rows < N_cap
    sr = tl.minimum(rows, N1 - 1).to(tl.int64)
    kk = tl.load(kcache + sr[:, None] * (H_kv * K) + h.to(tl.int64) * K + o_k[None, :])
    s = tl.dot(qr, tl.trans(kk))
    s = tl.where(valid[None, :] & gv[:, None], s, NEG)
    tl.store(s_out + h.to(tl.int64) * NUM_GROUPS * N_cap + g[:, None] * N_cap + rows.to(tl.int64)[None, :], s,
             mask=gv[:, None] & in_cap[None, :])
    m = tl.max(s, axis=0)
    lse = m + tl.log(tl.sum(tl.where(gv[:, None], tl.exp(s - m[None, :]), 0.0), axis=0))
    lse = tl.where(valid, lse, LSE_INV)
    lse = tl.where(rows == N1 - 1, LSE_FORCE, lse)
    tl.store(lse_out + h.to(tl.int64) * N_cap + rows.to(tl.int64), lse, mask=in_cap)
    NSUB: tl.constexpr = TILE // SUB
    sm = tl.max(tl.reshape(lse, [NSUB, SUB]), axis=1)
    si = tile * NSUB + tl.arange(0, NSUB)
    tl.store(smax_out + h.to(tl.int64) * n_sub + si, sm, mask=si < n_sub)


@triton.jit
def _okey(x):
    """fp32 -> int32 with the same order (signed compare)."""
    b = x.to(tl.int32, bitcast=True)
    return tl.where(b >= 0, b, b ^ 0x7FFFFFFF)


@triton.jit
def flat2_sub_threshold_kernel(
    smax,  # [H_kv, n_sub] fp32
    thr_out,
    counter,
    n_sub,
    TOP_K: tl.constexpr, NS_PAD: tl.constexpr,
):
    """The TOP_K-th largest 16-node sub-tile maximum: the nodes with lse >= it are a superset of the top TOP_K."""
    SIGN: tl.constexpr = -2147483648
    h = tl.program_id(0)
    o = tl.arange(0, NS_PAD)
    x = tl.load(smax + h.to(tl.int64) * n_sub + o, mask=o < n_sub, other=-3.0e38)
    k = _okey(x)
    tau = tl.zeros([1], dtype=tl.int32)
    for step in range(32):
        cand = tau | (1 << (31 - step))       # step 0: 1 << 31 == SIGN in int32
        cnt = tl.sum((k >= (cand ^ SIGN)).to(tl.int32))
        tau = tl.where(cnt >= TOP_K, cand, tau)
    tl.store(thr_out + h + tl.arange(0, 1), tau ^ SIGN)
    tl.store(counter + h + tl.arange(0, 1), tl.zeros([1], dtype=tl.int32))


@triton.jit
def flat2_compact_kernel(
    lse,  # [H_kv, N_cap] fp32
    smax,  # [H_kv, n_sub] fp32
    thr,  # [H_kv] int32
    counter,
    cand_key,  # [H_kv, CAP] int32 OUT
    cand_node,  # [H_kv, CAP] int32 OUT
    N_cap,
    n_sub,
    SUB: tl.constexpr, SPP: tl.constexpr, CAP: tl.constexpr, LSE_INV: tl.constexpr,
):
    """Append every valid node with lse >= the sub-tile threshold (only sub-tiles whose max reaches it are read)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    th = tl.load(thr + h)
    si = pid * SPP + tl.arange(0, SPP)
    sk = _okey(tl.load(smax + h.to(tl.int64) * n_sub + si, mask=si < n_sub, other=-3.0e38))
    hit = sk >= th
    if tl.sum(hit.to(tl.int32)) > 0:
        rows = (si[:, None] * SUB + tl.arange(0, SUB)[None, :])
        ok = hit[:, None] & (rows < N_cap)
        x = tl.load(lse + h.to(tl.int64) * N_cap + rows.to(tl.int64), mask=ok, other=LSE_INV)
        kx = _okey(x)
        take = ok & (kx >= th) & (x > LSE_INV * 0.5)
        take1 = tl.reshape(take, [SPP * SUB]).to(tl.int32)
        n = tl.sum(take1)
        base = tl.atomic_add(counter + h, n)
        pos = base + tl.cumsum(take1, axis=0) - take1
        keep = (take1 > 0) & (pos < CAP)
        dst = h.to(tl.int64) * CAP + pos.to(tl.int64)
        tl.store(cand_key + dst, tl.reshape(kx, [SPP * SUB]), mask=keep)
        tl.store(cand_node + dst, tl.reshape(rows, [SPP * SUB]).to(tl.int32), mask=keep)


@triton.jit
def _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K: tl.constexpr, CAP: tl.constexpr,
                 LANES: tl.constexpr, NODE_BITS: tl.constexpr):
    """Top-TOP_K of the first cnt (<= LANES) candidates, (key desc, node asc); writes the selected nodes (any order) to
    scratch[h, :n] and returns n."""
    SIGN: tl.constexpr = -2147483648
    lane = tl.arange(0, LANES)
    inl = lane < cnt
    k = tl.load(cand_key + h.to(tl.int64) * CAP + lane, mask=inl, other=SIGN)
    nd = tl.load(cand_node + h.to(tl.int64) * CAP + lane, mask=inl, other=2147483647)
    tau = tl.zeros([1], dtype=tl.int32)
    for step in range(32):
        cand = tau | (1 << (31 - step))       # step 0: 1 << 31 == SIGN in int32
        c = tl.sum(((k >= (cand ^ SIGN)) & inl).to(tl.int32))
        tau = tl.where(c >= TOP_K, cand, tau)
    th = tau ^ SIGN
    gt = inl & (k > th)
    tie = inl & (k == th)
    r = TOP_K - tl.sum(gt.to(tl.int32))
    n_tie = tl.sum(tie.to(tl.int32))
    sel = gt | tie
    if n_tie > r:
        nu = tl.zeros([1], dtype=tl.int32)
        for b in range(NODE_BITS):
            cand = nu | (1 << (NODE_BITS - 1 - b))
            c = tl.sum((tie & (nd < cand)).to(tl.int32))
            nu = tl.where(c < r, cand, nu)
        sel = gt | (tie & (nd <= nu) & (r > 0))
    s1 = sel.to(tl.int32)
    rk = tl.cumsum(s1, axis=0) - s1
    tl.store(scratch + h.to(tl.int64) * TOP_K + rk, nd, mask=sel & (rk < TOP_K))
    return tl.minimum(tl.sum(s1), TOP_K)


@triton.jit
def flat2_select_kernel(
    cand_key, cand_node,  # [H_kv, CAP] int32
    counter,
    scratch,  # [H_kv, TOP_K] int32
    prev_out,
    sel_stamp,  # [H_kv, N_cap] int32
    overflow,  # [H_kv] int32 OUT: candidates dropped (0 = exact)
    t_ptr, tag, N_cap,
    TOP_K: tl.constexpr, CAP: tl.constexpr, NODE_BITS: tl.constexpr,
):
    """Top-TOP_K of the candidates by (importance desc, node index asc), by bit bisection on the importance key, then
    on the node index among threshold ties; emits the nodes ascending and stamps them."""
    h = tl.program_id(0)
    t = tl.load(t_ptr).to(tl.int32)
    stamp = t * 64 + tag
    n_app = tl.load(counter + h)
    cnt = tl.minimum(n_app, CAP)
    if cnt <= 512:
        nsel = _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K, CAP, 512, NODE_BITS)
    elif cnt <= 2048:
        nsel = _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K, CAP, 2048, NODE_BITS)
    else:
        if CAP > 4096:
            if cnt <= 4096:
                nsel = _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K, CAP, 4096, NODE_BITS)
            else:
                nsel = _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K, CAP, CAP, NODE_BITS)
        else:
            nsel = _select_core(cand_key, cand_node, cnt, scratch, h, TOP_K, CAP, CAP, NODE_BITS)
    tl.debug_barrier()
    o = tl.arange(0, TOP_K)
    v = tl.load(scratch + h.to(tl.int64) * TOP_K + o, mask=o < nsel, other=2147483647)
    srt = tl.sort(v)
    ok = srt != 2147483647
    tl.store(prev_out + h.to(tl.int64) * TOP_K + o, tl.where(ok, srt, -1))
    tl.store(sel_stamp + h.to(tl.int64) * N_cap + tl.where(ok, srt, 0).to(tl.int64),
             tl.full([TOP_K], 0, tl.int32) + stamp, mask=ok)
    tl.store(overflow + h + tl.arange(0, 1), tl.maximum(n_app - CAP, 0) + tl.zeros([1], dtype=tl.int32))


@triton.jit
def flat2_emit_kernel(
    topk_idx,
    cand_node,
    lse,  # [H_kv, N_cap] fp32
    prev_out,
    sel_stamp,
    t_ptr,
    tag,  # per-call int (0..63)
    N_cap,
    NC,
    TOP_K: tl.constexpr,
    LSE_INV: tl.constexpr,
    INDIRECT: tl.constexpr,
):
    i_h_kv = tl.program_id(0)
    t = tl.load(t_ptr).to(tl.int32)
    stamp = t * 64 + tag
    o = tl.arange(0, TOP_K)
    idx = tl.load(topk_idx + i_h_kv.to(tl.int64) * TOP_K + o)
    if INDIRECT:
        idx = tl.load(cand_node + i_h_kv.to(tl.int64) * NC + idx)
    idx = idx.to(tl.int32)
    safe = tl.minimum(tl.maximum(idx, 0), N_cap - 1)
    val = tl.load(lse + i_h_kv.to(tl.int64) * N_cap + safe.to(tl.int64))
    big = 2147483647
    idx = tl.where((val > LSE_INV * 0.5) & (idx >= 0) & (idx < N_cap), idx, big)
    srt = tl.sort(idx)
    ok = srt != big
    tl.store(prev_out + i_h_kv.to(tl.int64) * TOP_K + o, tl.where(ok, srt, -1))
    tl.store(sel_stamp + i_h_kv.to(tl.int64) * N_cap + tl.where(ok, srt, 0).to(tl.int64),
             tl.full([TOP_K], 0, tl.int32) + stamp, mask=ok)


@triton.jit
def flat2_summary_kernel(
    s_in,
    layer_v,  # [1, N_cap, H_kv, V] level-1 values
    sel_stamp,  # [H_kv, N_cap] int32
    part_max, part_sum, part_out,
    t_ptr,
    tag,
    N_cap,
    H_kv: tl.constexpr, NUM_GROUPS: tl.constexpr, V: tl.constexpr, COMPRESSION_RATE: tl.constexpr,
    G_PAD: tl.constexpr, TILE: tl.constexpr, NS: tl.constexpr, NEG: tl.constexpr, DOT_FP16: tl.constexpr,
):
    """Online-softmax tile loop over the unselected level-1 nodes, with the accumulate kernel's tile boundaries and
    per-slice tile ranges; reads the scores instead of recomputing them."""
    pid_slice = tl.program_id(0)
    i_h_kv = tl.program_id(1)
    t = tl.load(t_ptr).to(tl.int32)
    N1 = t // COMPRESSION_RATE + 1
    stamp = t * 64 + tag
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    o_n = tl.arange(0, TILE).to(tl.int32)
    o_v = tl.arange(0, V).to(tl.int64)
    cur_max = tl.full([G_PAD], NEG, dtype=tl.float32)
    cur_sum = tl.zeros([G_PAD], dtype=tl.float32)
    cur_output = tl.zeros([G_PAD, V], dtype=tl.float32)
    v_base = layer_v + i_h_kv.to(tl.int64) * V
    s_base = s_in + i_h_kv.to(tl.int64) * NUM_GROUPS * N_cap
    st_base = sel_stamp + i_h_kv.to(tl.int64) * N_cap
    num_tiles = (N1 + TILE - 1) // TILE
    tiles_per_slice = (num_tiles + NS - 1) // NS
    tile_lo = pid_slice * tiles_per_slice
    tile_hi = tl.minimum(tile_lo + tiles_per_slice, num_tiles)
    for tile_idx in range(tile_lo, tile_hi):
        rows = tile_idx * TILE + o_n
        valid = rows < N1
        safe_rows = tl.minimum(rows, N1 - 1).to(tl.int64)
        selected = tl.load(st_base + safe_rows) == stamp
        valid = valid & ~selected
        scores = tl.load(s_base + g_ids[:, None] * N_cap + safe_rows[None, :],
                         mask=g_valid[:, None], other=NEG)
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, NEG)
        v_raw = tl.load(v_base + safe_rows[:, None] * (H_kv * V) + o_v[None, :])
        tile_m = tl.max(scores, axis=1)
        new_m = tl.maximum(cur_max, tile_m)
        alpha = tl.exp(cur_max - new_m)
        cur_sum = cur_sum * alpha
        cur_output = cur_output * alpha[:, None]
        attn_probs = tl.exp(scores - new_m[:, None])
        attn_probs = tl.where(valid[None, :] & g_valid[:, None], attn_probs, 0.0)
        cur_sum = cur_sum + tl.sum(attn_probs, axis=1)
        if DOT_FP16:
            cur_output = cur_output + tl.dot(attn_probs.to(v_raw.dtype), v_raw)
        else:
            v_tile = v_raw.to(tl.float32)
            cur_output = cur_output + tl.dot(attn_probs.to(v_tile.dtype), v_tile)
        cur_max = new_m
    pbase = pid_slice.to(tl.int64) * H_kv + i_h_kv
    tl.store(part_max + pbase * G_PAD + g_ids, cur_max)
    tl.store(part_sum + pbase * G_PAD + g_ids, cur_sum)
    tl.store(part_out + pbase * (G_PAD * V) + g_ids[:, None] * V + o_v[None, :], cur_output)


@triton.jit
def flat2_slice_reduce_kernel(
    in_max, in_sum, in_out,  # [NS_IN * H_kv * G_PAD (* V)]
    out_max, out_sum, out_out,  # [NS_OUT * H_kv * G_PAD (* V)]
    H_kv: tl.constexpr, V: tl.constexpr, G_PAD: tl.constexpr, R: tl.constexpr,
):
    """LSE-merge R consecutive input slices into one output slice (output slice j <- inputs j*R .. j*R+R-1)."""
    j = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.arange(0, G_PAD).to(tl.int64)
    o_v = tl.arange(0, V).to(tl.int64)
    m = tl.full([G_PAD], -3.0e38, dtype=tl.float32)
    for r in tl.static_range(R):
        b = (j * R + r).to(tl.int64) * H_kv + h
        m = tl.maximum(m, tl.load(in_max + b * G_PAD + g))
    s = tl.zeros([G_PAD], dtype=tl.float32)
    o = tl.zeros([G_PAD, V], dtype=tl.float32)
    for r in tl.static_range(R):
        b = (j * R + r).to(tl.int64) * H_kv + h
        w = tl.exp(tl.load(in_max + b * G_PAD + g) - m)
        s = s + tl.load(in_sum + b * G_PAD + g) * w
        o = o + tl.load(in_out + b * (G_PAD * V) + g[:, None] * V + o_v[None, :]) * w[:, None]
    ob = j.to(tl.int64) * H_kv + h
    tl.store(out_max + ob * G_PAD + g, m)
    tl.store(out_sum + ob * G_PAD + g, s)
    tl.store(out_out + ob * (G_PAD * V) + g[:, None] * V + o_v[None, :], o)


_WS: dict = {}
_ROPE: dict = {}


def reset():
    """Drop every workspace and rotated-key cache (between sessions in one process)."""
    _WS.clear()
    FLAT2.kcache.clear()
    FLAT2.last = None


def rope_cache(device, K: int, size: int, base: float):
    """cos/sin tables of any size, with the formula of recsa_forward's RoPE cache."""
    size = max(int(size), 1)
    size = -(-size // 4096) * 4096
    key = (device.index, K, float(base))
    hit = _ROPE.get(key)
    if hit is not None and hit[0].shape[0] >= size:
        return hit
    with torch.inference_mode(False), torch.no_grad():
        inv_freq = 1.0 / (base ** (torch.arange(0, K, 2, dtype=torch.float32, device=device) / K))
        positions = torch.arange(size, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        out = (freqs.cos(), freqs.sin())
    _ROPE[key] = out
    return out


def summary_slices(n_cap: int) -> int:
    """Summary split width: 16 while a slice holds <= 16 level-1 tiles, else doubled, up to 64 slices."""
    if FLAT2.slices is not None:
        return int(FLAT2.slices)
    tiles = -(-n_cap // TILE_N)
    r = 1
    while r < 4 and tiles // (BOTTOM_NSLICE * r) > 16:
        r *= 2
    return BOTTOM_NSLICE * r


def reserve_kcache(n: int, n_cap: int, H_kv: int, K: int, device) -> float:
    """Preallocate n rotated-key caches (e.g. before a prefill whose offload mirror sizes itself from the free memory)."""
    FLAT2.kcache_pool = [torch.empty(n_cap, H_kv, K, dtype=torch.float16, device=device) for _ in range(n)]
    return n * n_cap * H_kv * K * 2 / 2 ** 30


def _kcache_for(k1, cos, sin):
    """The rotated-key cache of this level-1 buffer, built on first use (eager only)."""
    base = k1._base if k1._base is not None else k1
    key = base.data_ptr()
    ent = FLAT2.kcache.get(key)
    if ent is not None and ent[0]() is base and ent[1].shape[0] >= k1.shape[1]:
        return ent[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("flat2 rotated-key cache must be built before CUDA-graph capture (eager warm step)")
    n_cap, H_kv, K = base.shape[1], base.shape[2], base.shape[3]
    buf = None
    for i, b in enumerate(FLAT2.kcache_pool):
        if b.shape == (n_cap, H_kv, K) and b.device == base.device:
            buf = FLAT2.kcache_pool.pop(i)
            break
    if buf is None:
        buf = torch.empty(n_cap, H_kv, K, dtype=torch.float16, device=base.device)
    flat2_cache_build_kernel[(-(-n_cap // TILE_N), H_kv)](base, buf, cos, sin, n_cap, H_kv=H_kv, K=K, TILE=TILE_N,
                                                          num_warps=4)
    FLAT2.kcache[key] = (weakref.ref(base), buf)
    FLAT2.kcache_builds += 1
    return buf


def _workspace(device, n_cap, H, H_kv, G, G_PAD, V, top_k, ns_sum, sub=SUB_N):
    key = (device.index, n_cap, H, H_kv, V, top_k, ns_sum, sub, torch.is_inference_mode_enabled())
    ws = _WS.get(key)
    if ws is None:
        f32 = dict(dtype=torch.float32, device=device)
        n_tiles = -(-n_cap // TILE_N)
        ws = _WS[key] = dict(
            s=torch.empty(H_kv, G, n_cap, **f32),
            lse=torch.empty(H_kv, n_cap, **f32),
            smax=torch.empty(H_kv, -(-n_cap // sub), **f32),
            thr=torch.empty(H_kv, dtype=torch.int32, device=device),
            counter=torch.zeros(H_kv, dtype=torch.int32, device=device),
            overflow=torch.zeros(H_kv, dtype=torch.int32, device=device),
            cand_key=torch.empty(H_kv, top_k * sub, dtype=torch.int32, device=device),
            cand_node=torch.empty(H_kv, top_k * sub, dtype=torch.int32, device=device),
            scratch=torch.empty(H_kv, top_k, dtype=torch.int32, device=device),
            stamp=torch.full((H_kv, n_cap), -1, dtype=torch.int32, device=device),
            prev=torch.empty(1, 1, H_kv, top_k, dtype=torch.int32, device=device),
            sp_max=torch.empty(ns_sum * H_kv * G_PAD, **f32),
            sp_sum=torch.empty(ns_sum * H_kv * G_PAD, **f32),
            sp_out=torch.empty(ns_sum * H_kv * G_PAD * V, **f32),
            p1_max=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD, **f32),
            p1_sum=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD, **f32),
            p1_out=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD * V, **f32),
            p0_max=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD, **f32),
            p0_sum=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD, **f32),
            p0_out=torch.empty(BOTTOM_NSLICE * H_kv * G_PAD * V, **f32),
            g_in=(torch.empty(1, 1, H, **f32), torch.empty(1, 1, H, **f32), torch.empty(1, 1, H, V, **f32)),
            g_out=(torch.empty(1, 1, H, **f32), torch.empty(1, 1, H, **f32), torch.empty(1, 1, H, V, **f32)),
        )
        if ns_sum == BOTTOM_NSLICE:
            ws["sp_max"], ws["sp_sum"], ws["sp_out"] = ws["p1_max"], ws["p1_sum"], ws["p1_out"]
    return ws


def flat2_forward(q, layers_k, layers_v, t_dev, *, compression_rate, top_k, max_top_nodes, scale, rope_base,
                  tag: int, tile_p: int = 4, num_stages: int = 2, mode: str = None):
    """One forced two-level attention call (B = 1, Tq = 1). layers_k/v: the session's level buffers (capacity buffers in
    cursor mode; levels >= 2 are ignored). t_dev: int64 [1] device tensor = the query's global row."""
    mode = FLAT2.mode if mode is None else mode
    if mode not in MODES:
        raise ValueError(f"FLAT2.mode must be one of {MODES}, got {mode!r}")
    fast = mode == "fast"
    B, Tq, H, K = q.shape
    assert B == 1 and Tq == 1, "the forced two-level path is the B=1, Tq=1 decode step"
    assert len(layers_k) >= 2, "needs a level-1 buffer (T > 0)"
    CR = int(compression_rate)
    assert tile_p * CR == TILE_N, "tile_p * compression_rate must be 64 (the deployed decode tile)"
    k0, v0, k1, v1 = layers_k[0], layers_v[0], layers_k[1], layers_v[1]
    H_kv, V = k1.shape[2], v1.shape[3]
    G = H // H_kv
    G_PAD = max(triton.next_power_of_2(G), 16)
    n_cap = int(k1.shape[1])
    assert n_cap >= top_k, f"level 1 holds {n_cap} < top_k {top_k} nodes"
    n_tiles = -(-n_cap // TILE_N)
    dev = q.device
    staged = not k0.is_cuda
    ns_sum = summary_slices(n_cap)
    assert ns_sum % BOTTOM_NSLICE == 0
    sub = int(FLAT2.sub)
    assert sub in (16, 32, 64)
    ws = _workspace(dev, n_cap, H, H_kv, G, G_PAD, V, top_k, ns_sum, sub)
    FLAT2.last = ws
    cos, sin = rope_cache(dev, K, max(n_cap + TILE_N, max_top_nodes) + 1024, float(rope_base))
    q = q.contiguous()
    if fast:
        kc = _kcache_for(k1, cos, sin)
        n_sub = -(-n_cap // sub)
        flat2_refresh_kernel[(int(FLAT2.refresh_tiles), H_kv)](
            k1, kc, cos, sin, t_dev, H_kv=H_kv, K=K, COMPRESSION_RATE=CR, TILE=TILE_N, num_warps=4)
        flat2_score_cached_kernel[(n_tiles, H_kv)](
            q, kc, cos, sin, ws["s"], ws["lse"], ws["smax"], t_dev, n_cap, n_sub, float(scale),
            H_kv=H_kv, NUM_GROUPS=G, K=K, COMPRESSION_RATE=CR, G_PAD=G_PAD, TILE=TILE_N, SUB=sub,
            NEG=NEG_INF, LSE_INV=LSE_INVALID, LSE_FORCE=LSE_FORCED, num_warps=4, num_stages=2)
        flat2_sub_threshold_kernel[(H_kv,)](ws["smax"], ws["thr"], ws["counter"], n_sub, TOP_K=top_k,
                                            NS_PAD=triton.next_power_of_2(n_sub), num_warps=8)
        flat2_compact_kernel[(-(-n_sub // 16), H_kv)](
            ws["lse"], ws["smax"], ws["thr"], ws["counter"], ws["cand_key"], ws["cand_node"], n_cap, n_sub,
            SUB=sub, SPP=16, CAP=top_k * sub, LSE_INV=LSE_INVALID, num_warps=4)
        flat2_select_kernel[(H_kv,)](
            ws["cand_key"], ws["cand_node"], ws["counter"], ws["scratch"], ws["prev"], ws["stamp"],
            ws["overflow"], t_dev, int(tag), n_cap, TOP_K=top_k, CAP=top_k * sub,
            NODE_BITS=max(1, int(n_cap).bit_length()), num_warps=8)
    else:
        tpp = int(FLAT2.score_tiles_per_prog)
        n_prog = -(-n_tiles // tpp)
        flat2_score_kernel[(n_prog, H_kv)](
            q, k1, cos, sin, ws["s"], ws["lse"], t_dev, n_cap, float(scale),
            H=H, H_kv=H_kv, NUM_GROUPS=G, K=K, COMPRESSION_RATE=CR, G_PAD=G_PAD, TILE=TILE_N, TPP=tpp,
            NEG=NEG_INF, LSE_INV=LSE_INVALID, LSE_FORCE=LSE_FORCED, num_warps=4, num_stages=num_stages)
        idx = torch.topk(ws["lse"], top_k, dim=1, largest=True, sorted=False).indices
        flat2_emit_kernel[(H_kv,)](idx, ws["cand_node"], ws["lse"], ws["prev"], ws["stamp"], t_dev, int(tag),
                                   n_cap, top_k * sub, TOP_K=top_k, LSE_INV=LSE_INVALID, INDIRECT=0,
                                   num_warps=4)
    flat2_summary_kernel[(ns_sum, H_kv)](
        ws["s"], v1, ws["stamp"], ws["sp_max"], ws["sp_sum"], ws["sp_out"], t_dev, int(tag), n_cap,
        H_kv=H_kv, NUM_GROUPS=G, V=V, COMPRESSION_RATE=CR, G_PAD=G_PAD, TILE=TILE_N, NS=ns_sum, NEG=NEG_INF,
        DOT_FP16=1 if fast else 0, num_warps=4, num_stages=num_stages)
    if ns_sum != BOTTOM_NSLICE:
        flat2_slice_reduce_kernel[(BOTTOM_NSLICE, H_kv)](
            ws["sp_max"], ws["sp_sum"], ws["sp_out"], ws["p1_max"], ws["p1_sum"], ws["p1_out"],
            H_kv=H_kv, V=V, G_PAD=G_PAD, R=ns_sum // BOTTOM_NSLICE, num_warps=4)
    fetch_k = fetch_v = None
    if staged:
        dws = P._decode_workspace(dev, B, H, H_kv, V, K, top_k, staged_leaf=(k0.dtype, max_top_nodes))
        fetch_k, fetch_v = dws["fetch_k"], dws["fetch_v"]
        if P._leaf_mirror_of(k0) is not None:
            P._leaf_mirror_gather(k0, v0, fetch_k, fetch_v, ws["prev"], 0, H_kv, K, V, CR, top_k, t_dev, 1)
        else:
            P.recsa_leaf_gather_kernel[(P.RECSA_GATHER_PROGS,)](
                k0, v0, fetch_k, fetch_v, ws["prev"], t_base=0, H_kv=H_kv, K=K, V=V, COMPRESSION_RATE=CR,
                TOP_K=top_k, NPROG=P.RECSA_GATHER_PROGS, t_ptr=t_dev, HAS_CURSOR=1, num_warps=4)
    P.recsa_dsplit_bottom_part_kernel[(BOTTOM_NSLICE, B * H_kv)](
        q, k0, v0, ws["prev"], cos, sin, ws["p0_max"], ws["p0_sum"], ws["p0_out"],
        t_base=0, layer_power=1, B=B, T=1, H=H, H_kv=H_kv, NUM_GROUPS=G, K=K, V=V, N_layer=k0.shape[1],
        COMPRESSION_RATE=CR, TOP_K=top_k, NEG_INF=NEG_INF, G_PAD=G_PAD, TILE_P=tile_p, NSLICE=BOTTOM_NSLICE,
        scale=float(scale), t_ptr=t_dev, HAS_CURSOR=1, fetch_k=fetch_k, fetch_v=fetch_v,
        LEAF_STAGED=1 if staged else 0, num_warps=4, num_stages=num_stages)
    out = torch.empty(B, 1, H, V, dtype=q.dtype, device=dev)
    gi, go = ws["g_in"], ws["g_out"]
    P.recsa_dsplit_merge_pair_kernel[(B * H_kv,)](
        ws["p1_max"], ws["p1_sum"], ws["p1_out"], ws["p0_max"], ws["p0_sum"], ws["p0_out"],
        gi[0], gi[1], gi[2], go[0], go[1], go[2], out, 0,
        T=1, H=H, H_kv=H_kv, NUM_GROUPS=G, V=V, NEG_INF=NEG_INF, G_PAD=G_PAD,
        GR=triton.next_power_of_2(G), NSLICE=BOTTOM_NSLICE, FRESH=1, num_warps=P._DSPLIT_PAIR_WARPS)
    return out


def _is_decode_call(q, kw) -> bool:
    return (q.dim() == 4 and q.shape[0] == 1 and q.shape[1] == 1 and kw.get("layers_kv") is not None
            and (kw.get("t_dev") is not None or kw.get("q_start") is not None))


def _flat_call(q, kw):
    if kw.get("no_summaries") or kw.get("summary_norm") is not None or kw.get("level_bias") is not None \
            or kw.get("force_masks") is not None:
        raise NotImplementedError("forced two-level path: summaries='softmax', no summary_norm / level_bias / "
                                  "force_masks only")
    if kw.get("rope_mode", "internal") != "internal":
        raise NotImplementedError("forced two-level path: internal RoPE only")
    lk, lv = kw["layers_kv"]
    t_dev = kw.get("t_dev")
    if t_dev is None:
        t_dev = torch.full((1,), int(kw["q_start"]), dtype=torch.int64, device=q.device)
    K = q.shape[-1]
    scale = kw.get("scale")
    tag = FLAT2.tag
    FLAT2.tag = (FLAT2.tag + 1) % 64
    return flat2_forward(q, lk, lv, t_dev, compression_rate=kw["compression_rate"],
                         top_k=kw["top_k_per_layer"], max_top_nodes=kw["max_top_nodes"],
                         scale=K ** -0.5 if scale is None else scale, rope_base=kw["rope_base"], tag=tag,
                         tile_p=kw.get("tile_p", 4), num_stages=kw.get("num_stages", 2)), t_dev


def _dispatch(q, k, v, *args, **kw):
    orig = FLAT2.orig
    if args or not _is_decode_call(q, kw):
        return orig(q, k, v, *args, **kw)
    if FLAT2.compare is not None:
        return FLAT2.compare(orig, q, k, v, kw)
    if not FLAT2.on:
        return orig(q, k, v, **kw)
    with torch.cuda.device(q.device):
        out, _ = _flat_call(q, kw)
    return out


def install():
    """Wrap graft.recsa_forward (the name the grafted attention calls)."""
    import graft as G
    if FLAT2.installed:
        return
    FLAT2.orig = G.recsa_forward
    G.recsa_forward = _dispatch
    FLAT2.installed = True


def uninstall():
    import graft as G
    if FLAT2.installed:
        G.recsa_forward = FLAT2.orig
        FLAT2.installed = False


def rope_rotate(x, pos, inv_freq):
    """[..., N, K] fp32 rotated by positions pos [N] (the kernels' rotate-half convention)."""
    K = x.shape[-1]
    f = pos.to(torch.float64)[:, None] * inv_freq.to(torch.float64)[None, :]
    c, s = f.cos().float(), f.sin().float()
    lo, hi = x[..., :K // 2], x[..., K // 2:]
    return torch.cat([lo * c - hi * s, lo * s + hi * c], dim=-1)


@torch.no_grad()
def flat2_reference(q, k0, v0, k1, v1, t: int, *, c: int, top_k: int, scale: float, rope_base: float,
                    kv_heads=None):
    """Pure-torch fp32 two-level step for one query row t (fp64 RoPE angles). q [H, K]; level-0 keys / values
    [>= t+1, H_kv, *]; level-1 keys / values [>= N1, H_kv, *] (any dtype; read as fp32)."""
    H, K = q.shape
    H_kv = k1.shape[1]
    G = H // H_kv
    N1 = t // c + 1
    inv_freq = 1.0 / (float(rope_base) ** (torch.arange(0, K, 2, dtype=torch.float64, device=q.device) / K))
    res = {}
    for h in (range(H_kv) if kv_heads is None else kv_heads):
        qg = q[h * G:(h + 1) * G].float()
        kk = k1[:N1, h].float()
        vv = v1[:N1, h].float()
        q1 = rope_rotate(qg, torch.full((G,), N1 - 1, device=q.device), inv_freq) * scale
        k1r = rope_rotate(kk, torch.arange(N1, device=q.device), inv_freq)
        s1 = q1 @ k1r.T                                    # [G, N1]
        imp = torch.exp(s1 - s1.max()).sum(0)
        imp_f = imp.clone()
        imp_f[N1 - 1] = float("inf")
        sel = torch.topk(imp_f, min(top_k, N1)).indices.sort().values
        unsel = torch.ones(N1, dtype=torch.bool, device=q.device)
        unsel[sel] = False
        n_sel = sel.numel()
        rows = (sel[:, None] * c + torch.arange(c, device=q.device)[None, :]).reshape(-1)
        slot = torch.arange(n_sel * c, device=q.device)
        ok = rows <= t
        rows, slot = rows[ok], slot[ok]
        qpos = (n_sel - 1) * c + (t % c)
        q0 = rope_rotate(qg, torch.full((G,), qpos, device=q.device), inv_freq) * scale
        k0r = rope_rotate(k0[rows, h].float(), slot, inv_freq)
        s0 = q0 @ k0r.T                                    # [G, n_leaf]
        logits = torch.cat([s1[:, unsel], s0], dim=1)
        vals = torch.cat([vv[unsel], v0[rows, h].float()], dim=0)
        p = torch.softmax(logits, dim=1)
        res[h] = dict(selected=sel.tolist(), out=p @ vals, imp=imp, n_leaves=int(rows.numel()),
                      n_summaries=int(unsel.sum()))
    return res
