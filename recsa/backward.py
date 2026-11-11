# -*- coding: utf-8 -*-
"""RecSA Backward Triton Kernels"""

import logging
import math
from typing import Optional

import torch
import triton
import triton.language as tl

from recsa.parallel import (
    KERNEL_TRACE_ENABLED,
    _cuda_timer,
    _packed_top_params,
    recsa_forward,
    recsa_build_kernel,
    recsa_mid_select_accumulate_ns_kernel,
    recsa_recompute_selection_kernel,
    recsa_top_select_kernel,
    RECSA_SCORE_NEG_INF,
    RECSA_SCORE_VALID_THRESHOLD,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO if KERNEL_TRACE_ENABLED else logging.WARNING)


@triton.jit
def recsa_bwd_preprocess_delta_kernel(
    o_ptr,       # [B, T, H, V]
    do_ptr,      # [B, T, H, V]
    delta_ptr,   # [B, T, H]        (output)
    T,
    H: tl.constexpr,
    V: tl.constexpr,
):
    """Grid: (T, B*H)."""
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H

    T_i64 = T
    base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V + i_h.to(tl.int64) * V
    o_v = tl.arange(0, V).to(tl.int64)
    o_vals = tl.load(o_ptr + base + o_v).to(tl.float32)
    do_vals = tl.load(do_ptr + base + o_v).to(tl.float32)
    d = tl.sum(o_vals * do_vals)

    d_off = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H + i_h.to(tl.int64)
    tl.store(delta_ptr + d_off, d)


@triton.jit
def recsa_bwd_build_parent_mask_kernel(
    selected_parents,   # [B, T, H_kv, TOP_K]  (bottom layer parents)
    parent_mask,
    T,
    H_kv: tl.constexpr,
    TOP_K: tl.constexpr,
    num_groups,
    BLOCK_TK: tl.constexpr,
):
    """For each (b, t, kv_h), mark which parent groups are attended."""
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T
    sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )

    for tk_off in range(0, TOP_K, BLOCK_TK):
        offs = tl.arange(0, BLOCK_TK)
        parents = tl.load(selected_parents + sel_base + tk_off + offs, mask=(tk_off + offs) < TOP_K, other=-1)
        valid = parents >= 0
        safe_g = tl.where(valid, parents, 0).to(tl.int64)

        mask_base = (
            i_b.to(tl.int64) * H_kv * num_groups * T_i64
            + i_h_kv.to(tl.int64) * num_groups * T_i64
            + i_t.to(tl.int64)
        )
        mask_ptrs = parent_mask + mask_base + safe_g * T_i64
        tl.store(mask_ptrs, tl.full([BLOCK_TK], 1, dtype=tl.uint8), mask=valid)


@triton.jit
def recsa_bwd_dq_bottom_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,    # [B, T, H_kv, TOP_K]
    cos_cache, sin_cache,     # [cache_size, K//2]
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    dq,                       # [B, T, H, K]  (accumulated output)
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
    G_PAD: tl.constexpr,
    TILE_P: tl.constexpr,
    scale,
):
    """Grid: (T, B * H_kv)."""
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    if num_valid_parents <= 0:
        return

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    # Load Q heads for this kv group
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)

    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    # Load dO and D
    do_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
    o_v = tl.arange(0, V).to(tl.int64)
    do_block = tl.load(do + do_base + head_ids[:, None] * V + o_v[None, :],
                       mask=g_valid[:, None], other=0.0).to(tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    D_vals = tl.load(delta + state_base + head_ids, mask=g_valid, other=0.0)
    gm = tl.load(global_max + state_base + head_ids, mask=g_valid, other=0.0)
    gs = tl.load(global_sum + state_base + head_ids, mask=g_valid, other=1.0)

    # dQ accumulators (in RoPE'd space)
    dq_tilde_lo = tl.zeros([G_PAD, K // 2], dtype=tl.float32)
    dq_tilde_hi = tl.zeros([G_PAD, K // 2], dtype=tl.float32)

    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    num_tiles = (num_valid_parents + TILE_P - 1) // TILE_P
    for tile_idx in range(num_tiles):
        tile_base = (tile_idx * TILE_P).to(tl.int32)
        p_offs = tile_base + parent_in_tile

        safe_p_offs = tl.minimum(p_offs, TOP_K - 1).to(tl.int64)
        p_idx = tl.load(prev_selected_parents + prev_sel_base + safe_p_offs).to(tl.int32)

        p_valid = (p_offs < num_valid_parents) & (p_offs < TOP_K) & (p_idx >= 0)
        is_rm = p_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_in_tile <= rightmost_child_idx)
        rows = tl.maximum(p_idx, 0) * COMPRESSION_RATE + child_in_tile
        valid = p_valid & ch_ok & (rows >= 0) & (rows < N_layer)
        safe_rows = tl.minimum(tl.maximum(rows, 0), N_layer - 1).to(tl.int64)

        # Gather K, apply RoPE (in-tile rotation, then per-tile base rotation)
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

        # Scores: [G_PAD, TILE_N]
        scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
        scores = tl.where(valid[None, :] & g_valid[:, None], scores, 0.0)

        # Reconstruct P = exp(S - global_max) / global_sum
        P = tl.exp(scores - gm[:, None]) / gs[:, None]
        P = tl.where(valid[None, :] & g_valid[:, None], P, 0.0)

        # Gather V tile
        v_row = v_base + safe_rows[:, None] * (H_kv * V)
        v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

        # dP = dO @ V^T : [G_PAD, TILE_N]
        dP = tl.dot(do_block, tl.trans(v_tile))
        # dS = P * (dP - D)
        dS = P * (dP - D_vals[:, None])

        # Accumulate dQ_tilde += dS @ K_rope : [G_PAD, K//2]
        dq_tilde_lo += tl.dot(dS.to(k_rope_lo.dtype), k_rope_lo)
        dq_tilde_hi += tl.dot(dS.to(k_rope_hi.dtype), k_rope_hi)

    dq_lo_result = scale * (dq_tilde_lo * cos_q[None, :] + dq_tilde_hi * sin_q[None, :])
    dq_hi_result = scale * (-dq_tilde_lo * sin_q[None, :] + dq_tilde_hi * cos_q[None, :])

    # Load-add-store to dq (no contention: unique (b,t,h) per program)
    dq_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    dq_ptrs = dq + dq_bt_base + head_ids[:, None] * K
    old_dq_lo = tl.load(dq_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    old_dq_hi = tl.load(dq_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    tl.store(dq_ptrs + o_k_half[None, :], old_dq_lo + dq_lo_result, mask=g_valid[:, None])
    tl.store(dq_ptrs + (K // 2) + o_k_half[None, :], old_dq_hi + dq_hi_result, mask=g_valid[:, None])


@triton.jit
def recsa_bwd_dq_upper_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,    # [B, T, H_kv, TOP_K]  (this layer's parents)
    next_layer_parents,
    child_selmask,            # [B, T, H_kv, NPG] int32 (HAS_SELMASK only):
                              #   bit c set => node pg*CR + c selected at next layer
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    dq,                       # [B, T, H, K]     (accumulated)
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
    DROP_BLOCK: tl.constexpr,
    G_PAD: tl.constexpr,
    scale,
    SEL_CHUNK: tl.constexpr,
    num_parent_groups,
    HAS_SELMASK: tl.constexpr,
):
    """Grid: (T, B * H_kv)."""
    i_t = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    rightmost_idx = i_t // layer_power
    rightmost_parent_idx = rightmost_idx // COMPRESSION_RATE
    rightmost_child_idx = rightmost_idx % COMPRESSION_RATE

    T_i64 = T

    prev_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )
    parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    if num_valid_parents <= 0:
        return

    num_candidates = ((num_valid_parents - 1) * COMPRESSION_RATE + rightmost_child_idx + 1).to(tl.int32)
    rope_pos_q = num_candidates - 1

    next_sel_base = (
        i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
        + i_t.to(tl.int64) * H_kv * TOP_K
        + i_h_kv.to(tl.int64) * TOP_K
    )

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)
    o_k_half = tl.arange(0, K // 2).to(tl.int64)

    q_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    q_ptrs = q + q_bt_base + head_ids[:, None] * K
    q_lo = tl.load(q_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
    q_hi = tl.load(q_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

    cos_q = tl.load(cos_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    sin_q = tl.load(sin_cache + rope_pos_q.to(tl.int64) * (K // 2) + o_k_half)
    q_rope_lo = (q_lo * cos_q[None, :] - q_hi * sin_q[None, :]) * scale
    q_rope_hi = (q_lo * sin_q[None, :] + q_hi * cos_q[None, :]) * scale

    do_base = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
    o_v = tl.arange(0, V).to(tl.int64)
    do_block = tl.load(do + do_base + head_ids[:, None] * V + o_v[None, :],
                       mask=g_valid[:, None], other=0.0).to(tl.float32)

    state_base = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
    D_vals = tl.load(delta + state_base + head_ids, mask=g_valid, other=0.0)
    gm = tl.load(global_max + state_base + head_ids, mask=g_valid, other=0.0)
    gs = tl.load(global_sum + state_base + head_ids, mask=g_valid, other=1.0)

    dq_tilde_lo = tl.zeros([G_PAD, K // 2], dtype=tl.float32)
    dq_tilde_hi = tl.zeros([G_PAD, K // 2], dtype=tl.float32)

    k_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    o_n = tl.arange(0, DROP_BLOCK).to(tl.int32)

    cos_off = tl.load(cos_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])
    sin_off = tl.load(sin_cache + o_n.to(tl.int64)[:, None] * (K // 2) + o_k_half[None, :])

    num_drop_tiles = (num_candidates + DROP_BLOCK - 1) // DROP_BLOCK
    for d_tile in range(num_drop_tiles):
        flat_pos = (d_tile * DROP_BLOCK).to(tl.int32) + o_n
        pos_valid = flat_pos < num_candidates

        parent_off = (flat_pos // COMPRESSION_RATE).to(tl.int32)
        child_slot = flat_pos - parent_off * COMPRESSION_RATE

        safe_poff = tl.minimum(tl.maximum(parent_off, 0), TOP_K - 1).to(tl.int64)
        parent_idx = tl.load(prev_selected_parents + prev_sel_base + safe_poff).to(tl.int32)

        is_rm = parent_idx == rightmost_parent_idx
        ch_ok = ~is_rm | (child_slot <= rightmost_child_idx)
        node_idx = tl.maximum(parent_idx, 0) * COMPRESSION_RATE + child_slot
        cand_valid = pos_valid & (parent_off < num_valid_parents) & (parent_idx >= 0) & ch_ok & (node_idx >= 0) & (node_idx < N_layer)

        if HAS_SELMASK:
            safe_pg = tl.minimum(tl.maximum(parent_idx, 0), num_parent_groups - 1).to(tl.int64)
            sm_base = (
                (i_b.to(tl.int64) * T_i64 + i_t.to(tl.int64)) * H_kv + i_h_kv.to(tl.int64)
            ) * num_parent_groups
            sm = tl.load(child_selmask + sm_base + safe_pg)
            is_selected = (sm >> child_slot) & 1
        else:
            is_selected = tl.zeros([DROP_BLOCK], dtype=tl.int32)
            for chunk_start in range(0, TOP_K, SEL_CHUNK):
                sel_chunk = tl.load(
                    next_layer_parents + next_sel_base + chunk_start + tl.arange(0, SEL_CHUNK)
                ).to(tl.int32)
                matches = (node_idx[:, None] == sel_chunk[None, :]).to(tl.int32)
                is_selected += tl.sum(matches, axis=1)

        is_dropped = (is_selected == 0) & cand_valid
        if tl.sum(is_dropped.to(tl.int32)) > 0:
            safe_node = tl.minimum(tl.maximum(node_idx, 0), N_layer - 1).to(tl.int64)

            k_row = k_base + safe_node[:, None] * (H_kv * K)
            k_lo = tl.load(k_row + o_k_half[None, :]).to(tl.float32)
            k_hi = tl.load(k_row + (K // 2) + o_k_half[None, :]).to(tl.float32)

            base_pos = (d_tile * DROP_BLOCK).to(tl.int64)
            cos_b = tl.load(cos_cache + base_pos * (K // 2) + o_k_half)
            sin_b = tl.load(sin_cache + base_pos * (K // 2) + o_k_half)
            k1_lo = k_lo * cos_off - k_hi * sin_off
            k1_hi = k_lo * sin_off + k_hi * cos_off
            k_rope_lo = k1_lo * cos_b[None, :] - k1_hi * sin_b[None, :]
            k_rope_hi = k1_lo * sin_b[None, :] + k1_hi * cos_b[None, :]

            scores = tl.dot(q_rope_lo, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi, tl.trans(k_rope_hi))
            scores = tl.where(is_dropped[None, :] & g_valid[:, None], scores, 0.0)
            P = tl.exp(scores - gm[:, None]) / gs[:, None]
            P = tl.where(is_dropped[None, :] & g_valid[:, None], P, 0.0)

            v_row = v_base + safe_node[:, None] * (H_kv * V)
            v_tile = tl.load(v_row + o_v[None, :]).to(tl.float32)

            dP = tl.dot(do_block, tl.trans(v_tile))
            dS = P * (dP - D_vals[:, None])

            dq_tilde_lo += tl.dot(dS.to(k_rope_lo.dtype), k_rope_lo)
            dq_tilde_hi += tl.dot(dS.to(k_rope_hi.dtype), k_rope_hi)

    dq_lo_result = scale * (dq_tilde_lo * cos_q[None, :] + dq_tilde_hi * sin_q[None, :])
    dq_hi_result = scale * (-dq_tilde_lo * sin_q[None, :] + dq_tilde_hi * cos_q[None, :])

    dq_bt_base = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
    dq_ptrs = dq + dq_bt_base + head_ids[:, None] * K
    old_dq_lo = tl.load(dq_ptrs + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    old_dq_hi = tl.load(dq_ptrs + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0)
    tl.store(dq_ptrs + o_k_half[None, :], old_dq_lo + dq_lo_result, mask=g_valid[:, None])
    tl.store(dq_ptrs + (K // 2) + o_k_half[None, :], old_dq_hi + dq_hi_result, mask=g_valid[:, None])


@triton.jit
def recsa_bwd_dkv_upper_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    prev_selected_parents,
    next_layer_parents,
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    parent_mask,              # [B, H_kv, num_parent_groups, T] uint8
    dk, dv,                   # [B, N_layer, H_kv, K/V] (output)
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
    num_parent_groups,
    G_PAD: tl.constexpr,
    SEL_CHUNK: tl.constexpr,
    CR_PAD: tl.constexpr,
    scale,
):
    """K-stationary dKV for upper (non-bottom) layers."""
    i_g = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T

    o_cr = tl.arange(0, CR_PAD).to(tl.int64)
    node_start = i_g.to(tl.int64) * COMPRESSION_RATE
    node_ids = node_start + o_cr
    node_valid = (node_ids < N_layer) & (o_cr < COMPRESSION_RATE)
    node_ids_i32 = node_ids.to(tl.int32)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    k_group_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_group_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    k_lo = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_group_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
                      mask=node_valid[:, None], other=0.0).to(tl.float32)

    dk_lo_acc = tl.zeros([CR_PAD, K // 2], dtype=tl.float32)
    dk_hi_acc = tl.zeros([CR_PAD, K // 2], dtype=tl.float32)
    dv_acc = tl.zeros([CR_PAD, V], dtype=tl.float32)

    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    mask_base = (
        i_b.to(tl.int64) * H_kv * num_parent_groups * T_i64
        + i_h_kv.to(tl.int64) * num_parent_groups * T_i64
        + i_g.to(tl.int64) * T_i64
    )

    for i_t in range(T):
        m_val = tl.load(parent_mask + mask_base + i_t.to(tl.int64))
        if m_val != 0:
            rightmost_idx_t = i_t // layer_power
            rightmost_child_idx_t = rightmost_idx_t % COMPRESSION_RATE

            sel_base_t = (
                i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                + i_t.to(tl.int64) * H_kv * TOP_K
                + i_h_kv.to(tl.int64) * TOP_K
            )
            parent_list_t = tl.load(prev_selected_parents + sel_base_t + tl.arange(0, TOP_K))
            num_valid_t = tl.sum((parent_list_t >= 0).to(tl.int32))

            match_mask = (parent_list_t == i_g.to(tl.int32)) & (parent_list_t >= 0)
            match_pos = tl.arange(0, TOP_K).to(tl.int32)
            pos_in_list = tl.min(tl.where(match_mask, match_pos, TOP_K))

            if pos_in_list < TOP_K:
                num_candidates_t = ((num_valid_t - 1) * COMPRESSION_RATE + rightmost_child_idx_t + 1).to(tl.int32)
                rope_pos_q_t = num_candidates_t - 1

                k_flat_base = pos_in_list.to(tl.int64) * COMPRESSION_RATE
                k_flat_pos = k_flat_base + o_cr

                is_rightmost_group = (i_g == (rightmost_idx_t // COMPRESSION_RATE))
                k_valid = node_valid & tl.where(
                    is_rightmost_group,
                    o_cr.to(tl.int32) <= rightmost_child_idx_t,
                    True
                )

                # dropped: children not in next_layer_parents
                next_sel_base_t = (
                    i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                    + i_t.to(tl.int64) * H_kv * TOP_K
                    + i_h_kv.to(tl.int64) * TOP_K
                )
                is_selected = tl.zeros([CR_PAD], dtype=tl.int32)
                for chunk_start in range(0, TOP_K, SEL_CHUNK):
                    sel_chunk = tl.load(
                        next_layer_parents + next_sel_base_t + chunk_start + tl.arange(0, SEL_CHUNK)
                    ).to(tl.int32)
                    matches = (node_ids_i32[:, None] == sel_chunk[None, :]).to(tl.int32)
                    is_selected += tl.sum(matches, axis=1)

                is_dropped = (is_selected == 0) & k_valid
                num_dropped = tl.sum(is_dropped.to(tl.int32))

                if num_dropped > 0:
                    cos_k_t = tl.load(cos_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                      mask=k_valid[:, None], other=0.0)
                    sin_k_t = tl.load(sin_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                      mask=k_valid[:, None], other=0.0)
                    k_rope_lo = k_lo * cos_k_t - k_hi * sin_k_t
                    k_rope_hi = k_lo * sin_k_t + k_hi * cos_k_t

                    q_bt_base_t = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
                    q_ptrs_t = q + q_bt_base_t + head_ids[:, None] * K
                    q_lo_t = tl.load(q_ptrs_t + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
                    q_hi_t = tl.load(q_ptrs_t + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

                    cos_q_t = tl.load(cos_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                    sin_q_t = tl.load(sin_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                    q_rope_lo_t = (q_lo_t * cos_q_t[None, :] - q_hi_t * sin_q_t[None, :]) * scale
                    q_rope_hi_t = (q_lo_t * sin_q_t[None, :] + q_hi_t * cos_q_t[None, :]) * scale

                    S = tl.dot(q_rope_lo_t, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi_t, tl.trans(k_rope_hi))
                    S = tl.where(is_dropped[None, :] & g_valid[:, None], S, 0.0)

                    state_base_t = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
                    gm_t = tl.load(global_max + state_base_t + head_ids, mask=g_valid, other=0.0)
                    gs_t = tl.load(global_sum + state_base_t + head_ids, mask=g_valid, other=1.0)
                    P = tl.exp(S - gm_t[:, None]) / gs_t[:, None]
                    P = tl.where(is_dropped[None, :] & g_valid[:, None], P, 0.0)

                    do_base_t = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
                    do_t = tl.load(do + do_base_t + head_ids[:, None] * V + o_v_dim[None, :],
                                   mask=g_valid[:, None], other=0.0).to(tl.float32)
                    D_t = tl.load(delta + state_base_t + head_ids, mask=g_valid, other=0.0)

                    dP = tl.dot(do_t, tl.trans(v_block))
                    dS = P * (dP - D_t[:, None])

                    dk_tilde_lo = tl.dot(tl.trans(dS).to(q_rope_lo_t.dtype), q_rope_lo_t)
                    dk_tilde_hi = tl.dot(tl.trans(dS).to(q_rope_hi_t.dtype), q_rope_hi_t)
                    dk_lo_acc += dk_tilde_lo * cos_k_t + dk_tilde_hi * sin_k_t
                    dk_hi_acc += -dk_tilde_lo * sin_k_t + dk_tilde_hi * cos_k_t

                    dv_acc += tl.dot(tl.trans(P).to(do_t.dtype), do_t)

    dk_out_base = dk + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    dv_out_base = dv + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
             dk_lo_acc, mask=node_valid[:, None])
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
             dk_hi_acc, mask=node_valid[:, None])
    tl.store(dv_out_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
             dv_acc, mask=node_valid[:, None])


@triton.jit
def recsa_bwd_dkv_bottom_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    selected_parents,         # [B, T, H_kv, TOP_K] (bottom layer)
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    parent_mask,              # [B, H_kv, num_groups, T] uint8
    dk, dv,                   # [B, N_layer, H_kv, K/V] (output)
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
    num_parent_groups,
    G_PAD: tl.constexpr,
    CR_PAD: tl.constexpr,
    scale,
):
    """K-stationary dKV for bottom layer."""
    i_g = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    T_i64 = T

    # Load K/V for this parent group (CR nodes, padded to CR_PAD for tl.dot)
    o_cr = tl.arange(0, CR_PAD).to(tl.int64)
    node_start = i_g.to(tl.int64) * COMPRESSION_RATE
    node_ids = node_start + o_cr
    node_valid = (node_ids < N_layer) & (o_cr < COMPRESSION_RATE)

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    k_group_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_group_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    k_lo = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_group_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
                      mask=node_valid[:, None], other=0.0).to(tl.float32)

    # dK/dV accumulators
    dk_lo_acc = tl.zeros([CR_PAD, K // 2], dtype=tl.float32)
    dk_hi_acc = tl.zeros([CR_PAD, K // 2], dtype=tl.float32)
    dv_acc = tl.zeros([CR_PAD, V], dtype=tl.float32)

    # GQA head setup
    g_ids = tl.arange(0, G_PAD).to(tl.int64)
    g_valid = g_ids < NUM_GROUPS
    head_ids = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_ids) * g_valid.to(tl.int64)

    # Parent mask base for this (b, kv_h, g)
    mask_base = (
        i_b.to(tl.int64) * H_kv * num_parent_groups * T_i64
        + i_h_kv.to(tl.int64) * num_parent_groups * T_i64
        + i_g.to(tl.int64) * T_i64
    )

    # flat per-query loop; nested `if` instead of `continue` for Triton
    for i_t in range(T):
        m_val = tl.load(parent_mask + mask_base + i_t.to(tl.int64))
        if m_val != 0:
            rightmost_idx_t = i_t // layer_power
            rightmost_child_idx_t = rightmost_idx_t % COMPRESSION_RATE

            sel_base_t = (
                i_b.to(tl.int64) * T_i64 * H_kv * TOP_K
                + i_t.to(tl.int64) * H_kv * TOP_K
                + i_h_kv.to(tl.int64) * TOP_K
            )
            parent_list_t = tl.load(selected_parents + sel_base_t + tl.arange(0, TOP_K))
            num_valid_t = tl.sum((parent_list_t >= 0).to(tl.int32))

            match_mask = (parent_list_t == i_g.to(tl.int32)) & (parent_list_t >= 0)
            match_pos = tl.arange(0, TOP_K).to(tl.int32)
            pos_in_list = tl.min(tl.where(match_mask, match_pos, TOP_K))

            if pos_in_list < TOP_K:
                num_candidates_t = ((num_valid_t - 1) * COMPRESSION_RATE + rightmost_child_idx_t + 1).to(tl.int32)
                rope_pos_q_t = num_candidates_t - 1

                k_flat_base = pos_in_list.to(tl.int64) * COMPRESSION_RATE
                k_flat_pos = k_flat_base + o_cr

                is_rightmost_group = (i_g == (rightmost_idx_t // COMPRESSION_RATE))
                k_valid = node_valid & tl.where(
                    is_rightmost_group,
                    o_cr.to(tl.int32) <= rightmost_child_idx_t,
                    True
                )

                cos_k_t = tl.load(cos_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                  mask=k_valid[:, None], other=0.0)
                sin_k_t = tl.load(sin_cache + k_flat_pos[:, None] * (K // 2) + o_k_half[None, :],
                                  mask=k_valid[:, None], other=0.0)
                k_rope_lo = k_lo * cos_k_t - k_hi * sin_k_t
                k_rope_hi = k_lo * sin_k_t + k_hi * cos_k_t

                q_bt_base_t = i_b.to(tl.int64) * T_i64 * H * K + i_t.to(tl.int64) * H * K
                q_ptrs_t = q + q_bt_base_t + head_ids[:, None] * K
                q_lo_t = tl.load(q_ptrs_t + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)
                q_hi_t = tl.load(q_ptrs_t + (K // 2) + o_k_half[None, :], mask=g_valid[:, None], other=0.0).to(tl.float32)

                cos_q_t = tl.load(cos_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                sin_q_t = tl.load(sin_cache + rope_pos_q_t.to(tl.int64) * (K // 2) + o_k_half)
                q_rope_lo_t = (q_lo_t * cos_q_t[None, :] - q_hi_t * sin_q_t[None, :]) * scale
                q_rope_hi_t = (q_lo_t * sin_q_t[None, :] + q_hi_t * cos_q_t[None, :]) * scale

                S = tl.dot(q_rope_lo_t, tl.trans(k_rope_lo)) + tl.dot(q_rope_hi_t, tl.trans(k_rope_hi))
                S = tl.where(k_valid[None, :] & g_valid[:, None], S, 0.0)

                state_base_t = i_b.to(tl.int64) * T_i64 * H + i_t.to(tl.int64) * H
                gm_t = tl.load(global_max + state_base_t + head_ids, mask=g_valid, other=0.0)
                gs_t = tl.load(global_sum + state_base_t + head_ids, mask=g_valid, other=1.0)
                P = tl.exp(S - gm_t[:, None]) / gs_t[:, None]
                P = tl.where(k_valid[None, :] & g_valid[:, None], P, 0.0)

                do_base_t = i_b.to(tl.int64) * T_i64 * H * V + i_t.to(tl.int64) * H * V
                do_t = tl.load(do + do_base_t + head_ids[:, None] * V + o_v_dim[None, :],
                               mask=g_valid[:, None], other=0.0).to(tl.float32)
                D_t = tl.load(delta + state_base_t + head_ids, mask=g_valid, other=0.0)

                dP = tl.dot(do_t, tl.trans(v_block))
                dS = P * (dP - D_t[:, None])

                dk_tilde_lo = tl.dot(tl.trans(dS).to(q_rope_lo_t.dtype), q_rope_lo_t)
                dk_tilde_hi = tl.dot(tl.trans(dS).to(q_rope_hi_t.dtype), q_rope_hi_t)
                dk_lo_acc += dk_tilde_lo * cos_k_t + dk_tilde_hi * sin_k_t
                dk_hi_acc += -dk_tilde_lo * sin_k_t + dk_tilde_hi * cos_k_t

                dv_acc += tl.dot(tl.trans(P).to(do_t.dtype), do_t)

    # Store dK, dV
    dk_out_base = dk + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    dv_out_base = dv + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V

    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
             dk_lo_acc, mask=node_valid[:, None])
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
             dk_hi_acc, mask=node_valid[:, None])
    tl.store(dv_out_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
             dv_acc, mask=node_valid[:, None])


def _build_dkv_match_csr(selected_parents, num_parent_groups):
    """Build a per-(b, h_kv, parent_group) CSR of matching queries."""
    B, T, H_kv, TOP_K = selected_parents.shape
    device = selected_parents.device
    LOG_TK = int(math.log2(TOP_K))
    assert (1 << LOG_TK) == TOP_K
    assert T < (1 << (31 - LOG_TK)), "payload packing needs T < 2^(31-log2(TOP_K))"

    npg1 = num_parent_groups + 1
    n_slice = T * TOP_K
    # Payload is (b, h_kv)-independent: build the [T*TOP_K] table once.
    payload_base = (
        (torch.arange(T, device=device, dtype=torch.int32).view(T, 1) << LOG_TK)
        | torch.arange(TOP_K, device=device, dtype=torch.int32).view(1, TOP_K)
    ).view(n_slice)

    nvp = torch.empty(B, T, H_kv, dtype=torch.int32, device=device)
    payload_sorted = torch.empty(B * H_kv * n_slice, dtype=torch.int32, device=device)
    counts = torch.empty(B * H_kv * npg1, dtype=torch.int64, device=device)
    for i_b in range(B):
        for i_h in range(H_kv):
            sel = selected_parents[i_b, :, i_h]              # [T, TOP_K] view
            valid = sel >= 0
            nvp[i_b, :, i_h] = valid.sum(dim=-1, dtype=torch.int32)
            g_eff = torch.where(valid, sel, num_parent_groups).view(n_slice)
            bh = i_b * H_kv + i_h
            counts[bh * npg1:(bh + 1) * npg1] = torch.bincount(g_eff, minlength=npg1)
            order = torch.argsort(g_eff, stable=True)
            torch.index_select(payload_base, 0, order,
                               out=payload_sorted[bh * n_slice:(bh + 1) * n_slice])
    offsets = torch.zeros(counts.numel() + 1, dtype=torch.int64, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, payload_sorted, nvp


def _build_child_selmask(next_layer_parents, num_parent_groups, compression_rate):
    """Per-(b, t, h_kv, parent_group) bitmask of children selected at the next layer (bit c set <=> node g*CR + c is in
    next_layer_parents[b,t,h])."""
    B, T, H_kv, TOP_K = next_layer_parents.shape
    device = next_layer_parents.device
    selmask = torch.zeros(B, T, H_kv, num_parent_groups, dtype=torch.int32, device=device)
    for i_h in range(H_kv):
        sel = next_layer_parents[:, :, i_h]                  # [B, T, TOP_K] view
        valid = sel >= 0
        ids = sel.clamp_min(0)
        bit = torch.where(valid, 1 << (ids % compression_rate), 0).to(torch.int32)
        selmask[:, :, i_h].scatter_add_(2, (ids // compression_rate).long(), bit)
    return selmask.view(-1)


@triton.jit
def recsa_bwd_dkv_bottom_batched_kernel(
    q,                        # [B, T, H, K]
    layer_k, layer_v,         # [B, N_layer, H_kv, K/V]
    csr_offsets,              # [B*H_kv*(NPG+1) + 1] int64
    csr_payload,              # [E] int32: (t << LOG_TK) | rank
    nvp,                      # [B, T, H_kv] int32
    child_selmask,            # [B, T, H_kv, NPG] int32 (HAS_SELMASK only):
                              #   bit c set => child c selected at next layer
    cos_cache, sin_cache,
    do,                       # [B, T, H, V]
    delta,                    # [B, T, H]
    global_max, global_sum,   # [B, T, H]
    dk, dv,                   # [B, N_layer, H_kv, K/V] (output)
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
    LOG_TK: tl.constexpr,
    num_parent_groups,
    G_PAD2: tl.constexpr,
    QT: tl.constexpr,
    HAS_SELMASK: tl.constexpr,
    scale,
):
    """Match-batched K-stationary dKV (bottom layer, or upper layers with HAS_SELMASK=True to restrict to candidates
    dropped at the next layer)."""
    i_g = tl.program_id(0)
    pid = tl.program_id(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv
    T_i64 = T
    M: tl.constexpr = QT * G_PAD2

    o_cr = tl.arange(0, COMPRESSION_RATE).to(tl.int64)
    node_start = i_g.to(tl.int64) * COMPRESSION_RATE
    node_ids = node_start + o_cr
    node_valid = node_ids < N_layer

    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    k_group_base = layer_k + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    v_group_base = layer_v + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V
    k_lo = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    k_hi = tl.load(k_group_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
                   mask=node_valid[:, None], other=0.0).to(tl.float32)
    v_block = tl.load(v_group_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
                      mask=node_valid[:, None], other=0.0).to(tl.float32)

    # Child-frame keys: k'_c = R_{c*theta} k_c (query independent)
    cos_c = tl.load(cos_cache + o_cr[:, None] * (K // 2) + o_k_half[None, :])
    sin_c = tl.load(sin_cache + o_cr[:, None] * (K // 2) + o_k_half[None, :])
    kp_lo = k_lo * cos_c - k_hi * sin_c
    kp_hi = k_lo * sin_c + k_hi * cos_c

    dkp_lo = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dkp_hi = tl.zeros([COMPRESSION_RATE, K // 2], dtype=tl.float32)
    dv_acc = tl.zeros([COMPRESSION_RATE, V], dtype=tl.float32)

    m_ids = tl.arange(0, M).to(tl.int32)
    qt_of_m = m_ids // G_PAD2
    g_of_m = m_ids % G_PAD2
    g_ok_m = g_of_m < NUM_GROUPS
    head_of_m = (i_h_kv.to(tl.int64) * NUM_GROUPS + g_of_m.to(tl.int64)) * g_ok_m.to(tl.int64)
    o_qt = tl.arange(0, QT).to(tl.int32)

    seg = pid.to(tl.int64) * (num_parent_groups + 1) + i_g.to(tl.int64)
    seg_start = tl.load(csr_offsets + seg)
    seg_end = tl.load(csr_offsets + seg + 1)
    n_chunks = (seg_end - seg_start + QT - 1) // QT

    for chunk in range(n_chunks):
        idxs = seg_start + chunk.to(tl.int64) * QT + o_qt.to(tl.int64)
        ok_qt = idxs < seg_end
        e = tl.load(csr_payload + idxs, mask=ok_qt, other=0)
        t_q = (e >> LOG_TK).to(tl.int32)                       # [QT]
        rank = (e & (TOP_K - 1)).to(tl.int32)

        nvp_q = tl.load(
            nvp + (i_b.to(tl.int64) * T_i64 + t_q.to(tl.int64)) * H_kv + i_h_kv.to(tl.int64),
            mask=ok_qt, other=1,
        ).to(tl.int32)
        rightmost_idx = t_q // layer_power
        rm_child = rightmost_idx % COMPRESSION_RATE
        nc = (nvp_q - 1) * COMPRESSION_RATE + rm_child + 1
        phi = nc - 1 - rank * COMPRESSION_RATE                 # [QT] >= 0
        is_rm_g = (rightmost_idx // COMPRESSION_RATE) == i_g

        # [QT, CR] candidate validity (rightmost group: children capped)
        c_ok = ok_qt[:, None] & node_valid[None, :] & (
            (~is_rm_g)[:, None] | (o_cr.to(tl.int32)[None, :] <= rm_child[:, None])
        )
        if HAS_SELMASK:
            sm = tl.load(
                child_selmask
                + ((i_b.to(tl.int64) * T_i64 + t_q.to(tl.int64)) * H_kv + i_h_kv.to(tl.int64))
                * num_parent_groups
                + i_g.to(tl.int64),
                mask=ok_qt, other=0,
            )
            c_ok = c_ok & (((sm[:, None] >> o_cr.to(tl.int32)[None, :]) & 1) == 0)

        # Per-row (m = qt*G_PAD2 + g) loads
        t_of_m = tl.reshape(tl.broadcast_to(t_q[:, None], [QT, G_PAD2]), [M])
        phi_of_m = tl.reshape(tl.broadcast_to(phi[:, None], [QT, G_PAD2]), [M])
        ok_of_m = tl.reshape(tl.broadcast_to(ok_qt[:, None], [QT, G_PAD2]), [M]) & g_ok_m

        q_row = (
            q + i_b.to(tl.int64) * T_i64 * H * K
            + t_of_m.to(tl.int64)[:, None] * H * K
            + head_of_m[:, None] * K
        )
        q_lo = tl.load(q_row + o_k_half[None, :], mask=ok_of_m[:, None], other=0.0).to(tl.float32)
        q_hi = tl.load(q_row + (K // 2) + o_k_half[None, :], mask=ok_of_m[:, None], other=0.0).to(tl.float32)

        safe_phi = tl.maximum(phi_of_m, 0).to(tl.int64)
        cos_p = tl.load(cos_cache + safe_phi[:, None] * (K // 2) + o_k_half[None, :])
        sin_p = tl.load(sin_cache + safe_phi[:, None] * (K // 2) + o_k_half[None, :])
        qp_lo = (q_lo * cos_p - q_hi * sin_p) * scale          # [M, K//2]
        qp_hi = (q_lo * sin_p + q_hi * cos_p) * scale

        S = tl.dot(qp_lo, tl.trans(kp_lo)) + tl.dot(qp_hi, tl.trans(kp_hi))  # [M, CR]
        mc_ok = tl.reshape(
            tl.broadcast_to(c_ok[:, None, :], [QT, G_PAD2, COMPRESSION_RATE]),
            [M, COMPRESSION_RATE],
        ) & g_ok_m[:, None]
        S = tl.where(mc_ok, S, 0.0)

        state_row = (
            i_b.to(tl.int64) * T_i64 * H
            + t_of_m.to(tl.int64) * H
            + head_of_m
        )
        gm = tl.load(global_max + state_row, mask=ok_of_m, other=0.0)
        gs = tl.load(global_sum + state_row, mask=ok_of_m, other=1.0)
        P = tl.exp(S - gm[:, None]) / gs[:, None]
        P = tl.where(mc_ok, P, 0.0)

        do_row = (
            do + i_b.to(tl.int64) * T_i64 * H * V
            + t_of_m.to(tl.int64)[:, None] * H * V
            + head_of_m[:, None] * V
        )
        do_t = tl.load(do_row + o_v_dim[None, :], mask=ok_of_m[:, None], other=0.0).to(tl.float32)
        D_t = tl.load(delta + state_row, mask=ok_of_m, other=0.0)

        dP = tl.dot(do_t, tl.trans(v_block))                   # [M, CR]
        dS = P * (dP - D_t[:, None])

        dkp_lo += tl.dot(tl.trans(dS).to(qp_lo.dtype), qp_lo)  # [CR, K//2]
        dkp_hi += tl.dot(tl.trans(dS).to(qp_hi.dtype), qp_hi)
        dv_acc += tl.dot(tl.trans(P).to(do_t.dtype), do_t)     # [CR, V]

    # Un-rotate dK from the child frame: dk = R_{c*theta}^T dk'
    dk_lo_acc = dkp_lo * cos_c + dkp_hi * sin_c
    dk_hi_acc = -dkp_lo * sin_c + dkp_hi * cos_c

    dk_out_base = dk + i_b.to(tl.int64) * N_layer * H_kv * K + i_h_kv.to(tl.int64) * K
    dv_out_base = dv + i_b.to(tl.int64) * N_layer * H_kv * V + i_h_kv.to(tl.int64) * V
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + o_k_half[None, :],
             dk_lo_acc, mask=node_valid[:, None])
    tl.store(dk_out_base + node_ids[:, None] * (H_kv * K) + (K // 2) + o_k_half[None, :],
             dk_hi_acc, mask=node_valid[:, None])
    tl.store(dv_out_base + node_ids[:, None] * (H_kv * V) + o_v_dim[None, :],
             dv_acc, mask=node_valid[:, None])


@triton.jit
def recsa_bwd_tree_backward_kernel(
    dk_parent,   # [B, N_parent, H_kv, K]
    dv_parent,   # [B, N_parent, H_kv, V]
    dk_child,    # [B, N_child, H_kv, K]  (accumulated)
    dv_child,    # [B, N_child, H_kv, V]  (accumulated)
    N_parent,
    N_child,
    H_kv: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """Propagate dK/dV from parent layer down to child layer through mean-pooling."""
    i_block = tl.program_id(0)
    pid = tl.program_id(1)
    B_Hkv = tl.num_programs(1)
    i_b = pid // H_kv
    i_h_kv = pid % H_kv

    p_start = i_block * BLOCK_P
    p_ids = p_start + tl.arange(0, BLOCK_P)
    o_k = tl.arange(0, K).to(tl.int64)
    o_v_dim = tl.arange(0, V).to(tl.int64)

    for p_local in range(BLOCK_P):
        p_idx = p_start + p_local
        if p_idx < N_parent:
            child_start = p_idx * COMPRESSION_RATE
            child_end = tl.minimum(child_start + COMPRESSION_RATE, N_child)
            count = child_end - child_start
            if count > 0:
                pk_base = dk_parent + (i_b.to(tl.int64) * N_parent * H_kv * K
                                       + p_idx * H_kv * K + i_h_kv.to(tl.int64) * K)
                pv_base = dv_parent + (i_b.to(tl.int64) * N_parent * H_kv * V
                                       + p_idx * H_kv * V + i_h_kv.to(tl.int64) * V)

                dk_p = tl.load(pk_base + o_k).to(tl.float32) / count.to(tl.float32)
                dv_p = tl.load(pv_base + o_v_dim).to(tl.float32) / count.to(tl.float32)

                for c_off in range(COMPRESSION_RATE):
                    c_idx = child_start + c_off
                    if c_idx < N_child:
                        ck_base = dk_child + (i_b.to(tl.int64) * N_child * H_kv * K
                                              + c_idx * H_kv * K + i_h_kv.to(tl.int64) * K)
                        cv_base = dv_child + (i_b.to(tl.int64) * N_child * H_kv * V
                                              + c_idx * H_kv * V + i_h_kv.to(tl.int64) * V)

                        old_dk = tl.load(ck_base + o_k).to(tl.float32)
                        old_dv = tl.load(cv_base + o_v_dim).to(tl.float32)
                        tl.store(ck_base + o_k, old_dk + dk_p)
                        tl.store(cv_base + o_v_dim, old_dv + dv_p)


def _recompute_tree_build(k, v, num_layers, compression_rate):
    """Rebuild upper-layer tree K/V from bottom-layer K/V via mean-pooling."""
    B = k.shape[0]
    T = k.shape[1]
    H_kv = k.shape[2]
    K = k.shape[3]
    V = v.shape[3]
    device, dtype = k.device, k.dtype

    layers_k, layers_v = [k], [v]
    current_k, current_v, current_len = k, v, T

    for layer_idx in range(1, num_layers):
        next_len = (current_len + compression_rate - 1) // compression_rate
        next_k = torch.empty(B, next_len, H_kv, K, dtype=dtype, device=device)
        next_v = torch.empty(B, next_len, H_kv, V, dtype=dtype, device=device)

        BLOCK_SIZE = 8
        grid = (triton.cdiv(next_len, BLOCK_SIZE), B * H_kv)
        recsa_build_kernel[grid](
            current_k, current_v, next_k, next_v,
            N_child=current_len, N_parent=next_len,
            B=B, H_kv=H_kv, K=K, V=V,
            COMPRESSION_RATE=compression_rate, BLOCK_SIZE=BLOCK_SIZE,
        )
        layers_k.append(next_k)
        layers_v.append(next_v)
        current_k, current_v = next_k, next_v
        current_len = next_len

    return layers_k, layers_v


def _recompute_parents(q, layers_k, cos_cache, sin_cache,
                       num_layers, compression_rate, top_k_per_layer,
                       max_top_nodes, scale, NUM_GROUPS, G_PAD, TILE_P,
                       packed_top=None, packed_qb=None, packed_warps=8,
                       mid_ns=False, mid_chunk=32768):
    """Replay forward selection loop to reconstruct per_layer_parents."""
    B, T, H, K = q.shape
    H_kv = layers_k[0].shape[2]
    device = q.device

    MAX_CANDIDATES = max_top_nodes
    LOG_N = int(math.log2(MAX_CANDIDATES))
    N_DIMS_TOPK = int(math.log2(top_k_per_layer))

    packed_supported, QB_packed, _qb_acc, G_PAD2 = _packed_top_params(
        NUM_GROUPS, K, compression_rate, top_k_per_layer, num_layers, TILE_P,
        packed_qb=packed_qb,
    )
    packed_top_eff = packed_supported if packed_top is None else bool(packed_top)

    top_layer_power = compression_rate ** (num_layers - 1)
    t_indices = torch.arange(T, dtype=torch.int32, device=device)
    rightmost_indices = t_indices // top_layer_power
    num_virtual_parents = rightmost_indices // compression_rate + 1
    parent_cands = torch.arange(top_k_per_layer, dtype=torch.int32, device=device)
    parent_cands = parent_cands.unsqueeze(0).expand(T, -1)
    valid_mask = parent_cands < num_virtual_parents.unsqueeze(1)
    prev_sp = torch.where(valid_mask, parent_cands,
                          torch.tensor(-1, dtype=torch.int32, device=device))
    prev_sp = prev_sp.unsqueeze(0).unsqueeze(2).expand(
        B, T, H_kv, top_k_per_layer).contiguous()
    next_sp = torch.empty_like(prev_sp)

    per_layer_parents = {}
    for layer_idx in range(num_layers - 1, -1, -1):
        per_layer_parents[layer_idx] = prev_sp.clone()

        if layer_idx > 0:
            k_layer = layers_k[layer_idx]
            N_layer = k_layer.shape[1]
            layer_power = compression_rate ** layer_idx

            if packed_top_eff and layer_idx == num_layers - 1:
                next_sp.fill_(-1)
                n_prog_t = triton.cdiv(T, QB_packed)
                sel_scratch = torch.empty(
                    B * H_kv * n_prog_t * QB_packed * top_k_per_layer,
                    dtype=torch.float32, device=device,
                )
                recsa_top_select_kernel[(n_prog_t, B * H_kv)](
                    q, k_layer, cos_cache, sin_cache,
                    next_sp,
                    next_sp,  # bitmask ptr unused: WRITE_BITMASK=False
                    sel_scratch,
                    t_base=0,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS, K=K,
                    N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    LOG_N=LOG_N,
                    SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                    NEG_INF=RECSA_SCORE_NEG_INF,
                    G_PAD2=G_PAD2, QB=QB_packed, TILE_P=TILE_P,
                    NWORDS=1, WRITE_BITMASK=False,
                    scale=scale,
                    num_warps=packed_warps,
                    num_stages=2,
                )
            elif mid_ns and layer_idx < num_layers - 1:
                next_sp.fill_(-1)
                chunk = min(T, mid_chunk)
                scratch = torch.empty(
                    chunk * B * H_kv * 3 * top_k_per_layer,
                    dtype=torch.float32, device=device,
                )
                dummy = torch.empty(1, dtype=torch.float32, device=device)
                NS_FAST = 1 if TILE_P == top_k_per_layer // compression_rate else 0
                for t0 in range(0, T, chunk):
                    cl = min(chunk, T - t0)
                    recsa_mid_select_accumulate_ns_kernel[(cl, B * H_kv)](
                        q, k_layer, dummy,
                        prev_sp,
                        cos_cache, sin_cache,
                        dummy, dummy, dummy,
                        next_sp,
                        scratch, t0,
                        t_base=0,
                        layer_power=layer_power,
                        B=B, T=T, H=H, H_kv=H_kv,
                        NUM_GROUPS=NUM_GROUPS,
                        K=K, V=K,
                        N_layer=N_layer,
                        COMPRESSION_RATE=compression_rate,
                        TOP_K=top_k_per_layer,
                        LOG_N=LOG_N,
                        SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                        NEG_INF=RECSA_SCORE_NEG_INF,
                        DROP_BLOCK=64,
                        G_PAD=G_PAD,
                        TILE_P=TILE_P,
                        NS_FAST=NS_FAST,
                        scale=scale,
                        SELECT_ONLY=1,

                        num_warps=4,
                        num_stages=1,
                    )
            else:
                grid_kv = (T, B * H_kv)
                recsa_recompute_selection_kernel[grid_kv](
                    q, k_layer, prev_sp,
                    cos_cache, sin_cache,
                    next_sp,
                    t_base=0,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS, K=K,
                    N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    LOG_N=LOG_N, N_DIMS_TOPK=N_DIMS_TOPK,
                    SCORE_VALID_THRESHOLD=RECSA_SCORE_VALID_THRESHOLD,
                    NEG_INF=RECSA_SCORE_NEG_INF,
                    G_PAD=G_PAD, TILE_P=TILE_P,
                    scale=scale,
                )
            prev_sp, next_sp = next_sp, prev_sp  # swap to reuse buffers

    return per_layer_parents


def recsa_backward(dO, q, output, bwd_ctx, dkv_batched=True, return_layer_grads=False):
    """Dispatch all backward kernels and return (dq, dk, dv)."""
    if bwd_ctx.get('summary_norm') is not None:
        raise ValueError("summary_norm is inference-only: recsa_backward "
                         "has no mass_cap support")
    B, T, H, K = q.shape
    V = dO.shape[-1]
    device = q.device
    dtype = torch.float32

    global_max = bwd_ctx['global_max']
    global_sum = bwd_ctx['global_sum']
    per_layer_parents = bwd_ctx['per_layer_parents']
    layers_k = bwd_ctx['layers_k']
    layers_v = bwd_ctx['layers_v']
    cos_cache = bwd_ctx['cos_cache']
    sin_cache = bwd_ctx['sin_cache']
    num_layers = bwd_ctx['num_layers']
    compression_rate = bwd_ctx['compression_rate']
    top_k_per_layer = bwd_ctx['top_k_per_layer']
    scale = bwd_ctx['scale']
    NUM_GROUPS = bwd_ctx['NUM_GROUPS']
    G_PAD = bwd_ctx['G_PAD']
    TILE_P = bwd_ctx['TILE_P']
    DROP_BLOCK = bwd_ctx['DROP_BLOCK']
    no_summaries = bool(bwd_ctx.get('no_summaries', False))

    H_kv = layers_k[0].shape[2]

    if dO.dtype == torch.float64:  # Triton fp64 loads are slow
        dO = dO.to(torch.float32)
    dO = dO.contiguous()
    output_c = output.contiguous()

    logger.info("recsa backward pass started")
    # preprocess delta
    delta = torch.empty(B, T, H, dtype=dtype, device=device)
    grid_k0 = (T, B * H)
    with _cuda_timer("  Kernel preprocess delta"):
        recsa_bwd_preprocess_delta_kernel[grid_k0](
            output_c, dO, delta, T=T, H=H, V=V,
        )

    # Allocate gradient buffers
    dq = torch.zeros(B, T, H, K, dtype=dtype, device=device)
    dk_layers = [torch.zeros_like(lk, dtype=dtype) for lk in layers_k]
    dv_layers = [torch.zeros_like(lv, dtype=dtype) for lv in layers_v]

    # process layers top to bottom, as in the forward
    for layer_idx in range(num_layers - 1, -1, -1):
        k_layer = layers_k[layer_idx]
        v_layer = layers_v[layer_idx]
        N_layer = k_layer.shape[1]
        is_bottom = (layer_idx == 0)
        layer_power = compression_rate ** layer_idx
        psp = per_layer_parents[layer_idx]

        grid_kv = (T, B * H_kv)
        logger.info("  -> Processing layer %d (N=%d, bottom=%s)...", layer_idx, N_layer, is_bottom)

        if is_bottom:
            dO_f32 = dO if dO.dtype == torch.float32 else dO.to(torch.float32)
            with _cuda_timer("    Kernel dQ bottom"):
                recsa_bwd_dq_bottom_kernel[grid_kv](
                    q, k_layer, v_layer,
                    psp, cos_cache, sin_cache,
                    dO_f32, delta, global_max, global_sum,
                    dq,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    G_PAD=G_PAD,
                    TILE_P=(bwd_ctx.get("dq_bottom_tile_p") or TILE_P),
                    scale=scale,
                    num_warps=bwd_ctx.get("dq_bottom_warps", 4),
                    num_stages=bwd_ctx.get("dq_bottom_stages", 1),
                )

            num_parent_groups = (N_layer + compression_rate - 1) // compression_rate
            G_PAD2 = triton.next_power_of_2(NUM_GROUPS)
            use_batched = (
                dkv_batched
                and G_PAD2 <= 16
                and compression_rate >= 16
                and K >= 32
                and T < (1 << (31 - int(math.log2(top_k_per_layer))))
            )
            if use_batched:
                with _cuda_timer("    dKV bottom CSR build"):
                    csr_offsets, csr_payload, nvp_t = _build_dkv_match_csr(
                        psp, num_parent_groups,
                    )
                QT = max(64 // G_PAD2, 4)  # dot M-dim = QT*G_PAD2 = 64
                grid_k3 = (num_parent_groups, B * H_kv)

                with _cuda_timer("    Kernel dKV bottom batched"):
                    recsa_bwd_dkv_bottom_batched_kernel[grid_k3](
                        q, k_layer, v_layer,
                        csr_offsets, csr_payload, nvp_t,
                        nvp_t,  # child_selmask unused: HAS_SELMASK=False
                        cos_cache, sin_cache,
                        dO, delta, global_max, global_sum,
                        dk_layers[0], dv_layers[0],
                        layer_power=layer_power,
                        B=B, T=T, H=H, H_kv=H_kv,
                        NUM_GROUPS=NUM_GROUPS,
                        K=K, V=V, N_layer=N_layer,
                        COMPRESSION_RATE=compression_rate,
                        TOP_K=top_k_per_layer,
                        LOG_TK=int(math.log2(top_k_per_layer)),
                        num_parent_groups=num_parent_groups,
                        G_PAD2=G_PAD2,
                        QT=QT,
                        HAS_SELMASK=False,
                        scale=scale,
                        num_warps=4,
                        num_stages=1,
                    )
            else:
                # build parent mask
                parent_mask = torch.zeros(B, H_kv, num_parent_groups, T, dtype=torch.uint8, device=device)
                grid_k2 = (T, B * H_kv)
                BLOCK_TK = min(top_k_per_layer, 64)
                with _cuda_timer("    Kernel build parent mask"):
                    recsa_bwd_build_parent_mask_kernel[grid_k2](
                        psp, parent_mask,
                        T=T, H_kv=H_kv, TOP_K=top_k_per_layer,
                        num_groups=num_parent_groups, BLOCK_TK=BLOCK_TK,
                    )

                # K-stationary dKV for the bottom layer
                grid_k3 = (num_parent_groups, B * H_kv)
                with _cuda_timer("    Kernel dKV bottom"):
                    recsa_bwd_dkv_bottom_kernel[grid_k3](
                        q, k_layer, v_layer,
                        psp, cos_cache, sin_cache,
                        dO, delta, global_max, global_sum,
                        parent_mask,
                        dk_layers[0], dv_layers[0],
                        layer_power=layer_power,
                        B=B, T=T, H=H, H_kv=H_kv,
                        NUM_GROUPS=NUM_GROUPS,
                        K=K, V=V, N_layer=N_layer,
                        COMPRESSION_RATE=compression_rate,
                        TOP_K=top_k_per_layer,
                        num_parent_groups=num_parent_groups,
                        G_PAD=G_PAD,
                        CR_PAD=max(16, compression_rate),
                        scale=scale,
                    )
        elif no_summaries:
            logger.info("  -> layer %d skipped (no_summaries: no pooled softmax terms)", layer_idx)
        else:
            next_layer_parents = per_layer_parents[layer_idx - 1]
            SEL_CHUNK = min(top_k_per_layer, 64)

            num_upper_parent_groups = (N_layer + compression_rate - 1) // compression_rate
            G_PAD2 = triton.next_power_of_2(NUM_GROUPS)
            use_batched_u = (
                dkv_batched
                and G_PAD2 <= 16
                and 16 <= compression_rate <= 32  # selmask packs CR bits in int32
                and K >= 32
                and T < (1 << (31 - int(math.log2(top_k_per_layer))))
            )
            dq_selmask = compression_rate <= 32  # selmask packs CR bits in int32
            child_selmask = None
            if dq_selmask or use_batched_u:
                with _cuda_timer("    child selmask build"):
                    child_selmask = _build_child_selmask(
                        next_layer_parents, num_upper_parent_groups, compression_rate,
                    )

            with _cuda_timer("    Kernel dQ upper"):
                recsa_bwd_dq_upper_kernel[grid_kv](
                    q, k_layer, v_layer,
                    psp, next_layer_parents,
                    child_selmask if dq_selmask else next_layer_parents,
                    cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    dq,
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    DROP_BLOCK=16,
                    G_PAD=G_PAD,
                    scale=scale,
                    SEL_CHUNK=SEL_CHUNK,
                    num_parent_groups=num_upper_parent_groups,
                    HAS_SELMASK=dq_selmask,
                    num_warps=2,
                    num_stages=2,
                )

            if use_batched_u:
                with _cuda_timer("    dKV upper CSR build"):
                    csr_offsets, csr_payload, nvp_t = _build_dkv_match_csr(
                        psp, num_upper_parent_groups,
                    )

                QT = max(32 // G_PAD2, 4)
                with _cuda_timer("    Kernel dKV upper batched"):
                    recsa_bwd_dkv_bottom_batched_kernel[(num_upper_parent_groups, B * H_kv)](
                        q, k_layer, v_layer,
                        csr_offsets, csr_payload, nvp_t,
                        child_selmask,
                        cos_cache, sin_cache,
                        dO, delta, global_max, global_sum,
                        dk_layers[layer_idx], dv_layers[layer_idx],
                        layer_power=layer_power,
                        B=B, T=T, H=H, H_kv=H_kv,
                        NUM_GROUPS=NUM_GROUPS,
                        K=K, V=V, N_layer=N_layer,
                        COMPRESSION_RATE=compression_rate,
                        TOP_K=top_k_per_layer,
                        LOG_TK=int(math.log2(top_k_per_layer)),
                        num_parent_groups=num_upper_parent_groups,
                        G_PAD2=G_PAD2,
                        QT=QT,
                        HAS_SELMASK=True,
                        scale=scale,
                        num_warps=4,
                        num_stages=2,
                    )
            else:
                # build the upper parent mask (reuses the bottom-layer kernel)
                upper_parent_mask = torch.zeros(B, H_kv, num_upper_parent_groups, T,
                                                dtype=torch.uint8, device=device)
                BLOCK_TK = min(top_k_per_layer, 64)
                with _cuda_timer("    Kernel build upper parent mask"):
                    recsa_bwd_build_parent_mask_kernel[grid_kv](
                        psp, upper_parent_mask,
                        T=T, H_kv=H_kv, TOP_K=top_k_per_layer,
                        num_groups=num_upper_parent_groups, BLOCK_TK=BLOCK_TK,
                    )

                # K-stationary dKV for the upper layer (no atomics)
                grid_upper_kv = (num_upper_parent_groups, B * H_kv)
                with _cuda_timer("    Kernel dKV upper"):
                    recsa_bwd_dkv_upper_kernel[grid_upper_kv](
                    q, k_layer, v_layer,
                    psp, next_layer_parents,
                    cos_cache, sin_cache,
                    dO, delta, global_max, global_sum,
                    upper_parent_mask,
                    dk_layers[layer_idx], dv_layers[layer_idx],
                    layer_power=layer_power,
                    B=B, T=T, H=H, H_kv=H_kv,
                    NUM_GROUPS=NUM_GROUPS,
                    K=K, V=V, N_layer=N_layer,
                    COMPRESSION_RATE=compression_rate,
                    TOP_K=top_k_per_layer,
                    num_parent_groups=num_upper_parent_groups,
                    G_PAD=G_PAD,
                    SEL_CHUNK=SEL_CHUNK,
                    CR_PAD=max(16, compression_rate),
                    scale=scale,
                )

    if return_layer_grads:
        logger.info("recsa backward pass completed (per-layer grads, tree backward skipped)!")
        return dq, dk_layers, dv_layers

    # tree backward: propagate gradients down through mean-pooling
    BLOCK_P = 8
    for layer_idx in range(num_layers - 1, 0, -1):
        N_parent = layers_k[layer_idx].shape[1]
        N_child = layers_k[layer_idx - 1].shape[1]
        num_blocks = (N_parent + BLOCK_P - 1) // BLOCK_P
        grid_k4 = (num_blocks, B * H_kv)
        with _cuda_timer(f"  Kernel tree backward ({N_parent}->{N_child})"):
            recsa_bwd_tree_backward_kernel[grid_k4](
                dk_layers[layer_idx], dv_layers[layer_idx],
                dk_layers[layer_idx - 1], dv_layers[layer_idx - 1],
                N_parent=N_parent, N_child=N_child,
                H_kv=H_kv, K=K, V=V,
                COMPRESSION_RATE=compression_rate,
                BLOCK_P=BLOCK_P,
            )

    logger.info("recsa backward pass completed!")
    return dq, dk_layers[0], dv_layers[0]


class RecSATritonFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base, rope_mode,
                no_summaries=False):
        output, bwd_ctx = recsa_forward(
            q, k, v,
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale,
            rope_base=rope_base,
            rope_mode=rope_mode,
            no_summaries=bool(no_summaries),
            _save_for_backward=True,
            _save_parents=False,
        )

        ctx.save_for_backward(
            q, k, v, output,
            bwd_ctx['global_max'], bwd_ctx['global_sum'],
            bwd_ctx['cos_cache'], bwd_ctx['sin_cache'],
        )

        ctx.num_layers = bwd_ctx['num_layers']
        ctx.compression_rate = compression_rate
        ctx.top_k_per_layer = top_k_per_layer
        ctx.max_top_nodes = max_top_nodes
        ctx.scale = scale
        ctx.rope_base = rope_base
        ctx.rope_mode = rope_mode
        ctx.NUM_GROUPS = bwd_ctx['NUM_GROUPS']
        ctx.G_PAD = bwd_ctx['G_PAD']
        ctx.TILE_P = bwd_ctx['TILE_P']
        ctx.DROP_BLOCK = bwd_ctx['DROP_BLOCK']
        ctx.packed_top = bwd_ctx.get('packed_top', False)
        ctx.QB_packed = bwd_ctx.get('QB_packed', None)
        ctx.packed_warps = bwd_ctx.get('packed_warps', 8)
        ctx.no_summaries = bool(no_summaries)
        return output

    @staticmethod
    def backward(ctx, dO):
        q, k, v, output, global_max, global_sum, cos_cache, sin_cache = ctx.saved_tensors

        with _cuda_timer("  Recompute tree build"):
            layers_k, layers_v = _recompute_tree_build(
                k, v, ctx.num_layers, ctx.compression_rate,
            )

        with _cuda_timer("  Recompute parents (selection-only)"):
            per_layer_parents = _recompute_parents(
                q, layers_k, cos_cache, sin_cache,
                ctx.num_layers, ctx.compression_rate, ctx.top_k_per_layer,
                ctx.max_top_nodes, ctx.scale,
                ctx.NUM_GROUPS, ctx.G_PAD, ctx.TILE_P,
                packed_top=ctx.packed_top,
                packed_qb=ctx.QB_packed,
                packed_warps=ctx.packed_warps,
            )

        bwd_ctx = dict(
            global_max=global_max, global_sum=global_sum,
            per_layer_parents=per_layer_parents,
            layers_k=layers_k, layers_v=layers_v,
            cos_cache=cos_cache, sin_cache=sin_cache,
            num_layers=ctx.num_layers,
            compression_rate=ctx.compression_rate,
            top_k_per_layer=ctx.top_k_per_layer,
            max_top_nodes=ctx.max_top_nodes,
            scale=ctx.scale,
            NUM_GROUPS=ctx.NUM_GROUPS,
            G_PAD=ctx.G_PAD,
            TILE_P=ctx.TILE_P,
            DROP_BLOCK=ctx.DROP_BLOCK,
            no_summaries=ctx.no_summaries,
        )

        dq, dk, dv = recsa_backward(dO, q, output, bwd_ctx)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None, None, None, None


def recsa_forward_triton_autograd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    compression_rate: int = 16, max_top_nodes: int = 8192,
    top_k_per_layer: int = 512, scale: Optional[float] = None,
    rope_base: float = 10000.0,
    rope_mode: str = "internal",
    no_summaries: bool = False,
) -> torch.Tensor:
    """Differentiable RecSA forward using Triton kernels for both fwd and bwd. no_summaries: the operator variant
    without summaries (recsa_forward / recsa_backward)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if rope_mode not in {"internal", "none"}:
        raise ValueError(f"unsupported rope_mode {rope_mode!r}; expected 'internal' or 'none'")
    if not isinstance(no_summaries, bool):
        raise TypeError(f"no_summaries must be a bool, got {no_summaries!r}")
    return RecSATritonFunction.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        compression_rate, max_top_nodes, top_k_per_layer, scale, rope_base, rope_mode,
        no_summaries,
    )


class RecSATritonLayersFunction(torch.autograd.Function):
    """RecSA autograd over an explicit tree pyramid."""

    @staticmethod
    def forward(ctx, q, compression_rate, max_top_nodes, top_k_per_layer, scale,
                rope_base, rope_mode, num_layers, *layers):
        n_flag = len(layers) - 2 * num_layers
        assert n_flag in (0, 1), \
            "expected num_layers K tensors then num_layers V tensors [then the no_summaries bool]"
        no_summaries = False
        if n_flag:
            no_summaries = layers[-1]
            assert isinstance(no_summaries, bool), \
                f"trailing no_summaries must be a bool, got {no_summaries!r}"
            layers = layers[:-1]
        layers_k = list(layers[:num_layers])
        layers_v = list(layers[num_layers:])
        output, bwd_ctx = recsa_forward(
            q, layers_k[0], layers_v[0],
            compression_rate=compression_rate,
            max_top_nodes=max_top_nodes,
            top_k_per_layer=top_k_per_layer,
            scale=scale,
            rope_base=rope_base,
            rope_mode=rope_mode,
            layers_kv=(layers_k, layers_v),
            no_summaries=no_summaries,
            _save_for_backward=True,
            _save_parents=False,
        )

        ctx.save_for_backward(
            q, output,
            bwd_ctx['global_max'], bwd_ctx['global_sum'],
            bwd_ctx['cos_cache'], bwd_ctx['sin_cache'],
            *layers,
        )
        ctx.num_layers = num_layers
        ctx.compression_rate = compression_rate
        ctx.top_k_per_layer = top_k_per_layer
        ctx.max_top_nodes = max_top_nodes
        ctx.scale = scale
        ctx.rope_base = rope_base
        ctx.rope_mode = rope_mode
        ctx.NUM_GROUPS = bwd_ctx['NUM_GROUPS']
        ctx.G_PAD = bwd_ctx['G_PAD']
        ctx.TILE_P = bwd_ctx['TILE_P']
        ctx.DROP_BLOCK = bwd_ctx['DROP_BLOCK']
        ctx.packed_top = bwd_ctx.get('packed_top', False)
        ctx.QB_packed = bwd_ctx.get('QB_packed', None)
        ctx.packed_warps = bwd_ctx.get('packed_warps', 8)
        ctx.no_summaries = no_summaries
        ctx.n_flag = n_flag               # one grad slot (None) for the trailing flag
        return output

    @staticmethod
    def backward(ctx, dO):
        saved = ctx.saved_tensors
        q, output, global_max, global_sum, cos_cache, sin_cache = saved[:6]
        num_layers = ctx.num_layers
        layers_k = list(saved[6:6 + num_layers])
        layers_v = list(saved[6 + num_layers:6 + 2 * num_layers])

        with _cuda_timer("  Recompute parents (selection-only)"):
            per_layer_parents = _recompute_parents(
                q, layers_k, cos_cache, sin_cache,
                num_layers, ctx.compression_rate, ctx.top_k_per_layer,
                ctx.max_top_nodes, ctx.scale,
                ctx.NUM_GROUPS, ctx.G_PAD, ctx.TILE_P,
                packed_top=ctx.packed_top,
                packed_qb=ctx.QB_packed,
                packed_warps=ctx.packed_warps,
            )

        bwd_ctx = dict(
            global_max=global_max, global_sum=global_sum,
            per_layer_parents=per_layer_parents,
            layers_k=layers_k, layers_v=layers_v,
            cos_cache=cos_cache, sin_cache=sin_cache,
            num_layers=num_layers,
            compression_rate=ctx.compression_rate,
            top_k_per_layer=ctx.top_k_per_layer,
            max_top_nodes=ctx.max_top_nodes,
            scale=ctx.scale,
            NUM_GROUPS=ctx.NUM_GROUPS,
            G_PAD=ctx.G_PAD,
            TILE_P=ctx.TILE_P,
            DROP_BLOCK=ctx.DROP_BLOCK,
            no_summaries=ctx.no_summaries,
        )

        dq, dk_layers, dv_layers = recsa_backward(
            dO, q, output, bwd_ctx, return_layer_grads=True,
        )
        layer_grads = [g.to(t.dtype) for g, t in zip(dk_layers, layers_k)] + \
                      [g.to(t.dtype) for g, t in zip(dv_layers, layers_v)]
        return (dq.to(q.dtype), None, None, None, None, None, None, None, *layer_grads,
                *((None,) * ctx.n_flag))


def recsa_forward_probe_autograd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    probe_w: torch.Tensor, probe_tau: torch.Tensor,
    compression_rate: int = 16, max_top_nodes: int = 8192,
    top_k_per_layer: int = 512, scale: Optional[float] = None,
    rope_base: float = 10000.0,
    rope_mode: str = "internal",
    probe_mode: str = "per_level",
    probe_gate: Optional[torch.Tensor] = None,
    probe_extra: Optional[dict] = None,
    no_summaries: bool = False,
) -> torch.Tensor:
    """Differentiable RecSA forward with probe-attention pooled parent keys."""
    from recsa.probe_pool import build_tree_layers

    if scale is None:
        scale = q.shape[-1] ** -0.5
    if rope_mode not in {"internal", "none"}:
        raise ValueError(f"unsupported rope_mode {rope_mode!r}; expected 'internal' or 'none'")
    if not isinstance(no_summaries, bool):
        raise TypeError(f"no_summaries must be a bool, got {no_summaries!r}")
    layers_k, layers_v = build_tree_layers(
        k.contiguous(), v.contiguous(), compression_rate, max_top_nodes,
        probe_w=probe_w, probe_tau=probe_tau,
        probe_mode=probe_mode, probe_gate=probe_gate,
        probe_extra=probe_extra,
    )
    return RecSATritonLayersFunction.apply(
        q.contiguous(), compression_rate, max_top_nodes, top_k_per_layer,
        scale, rope_base, rope_mode, len(layers_k), *layers_k, *layers_v,
        no_summaries,
    )


__all__ = [
    'recsa_backward',
    'recsa_forward_triton_autograd',
    'recsa_forward_probe_autograd',
    'RecSATritonFunction',
    'RecSATritonLayersFunction',
]
