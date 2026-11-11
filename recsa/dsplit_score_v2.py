"""Tile-parallel decode scoring kernel with an exact look-back of the tile maxima across CTAs."""
import os

import torch
import triton
import triton.language as tl

RECSA_DSPLIT_SCORE = os.environ.get("RECSA_DSPLIT_SCORE", "v2").strip().lower()
if RECSA_DSPLIT_SCORE not in ("v2", "v1"):
    raise ValueError(f"RECSA_DSPLIT_SCORE must be 'v2' or 'v1', got {RECSA_DSPLIT_SCORE!r}")
RECSA_DSPLIT_SCORE_ROLE = os.environ.get("RECSA_DSPLIT_SCORE_ROLE", "noscat").strip().lower()
ROLE_VALUES = ("noscat", "v1")
if RECSA_DSPLIT_SCORE_ROLE not in ROLE_VALUES:
    raise ValueError(f"RECSA_DSPLIT_SCORE_ROLE must be one of {ROLE_VALUES}, got {RECSA_DSPLIT_SCORE_ROLE!r}")

SPIN_LIMIT = 1 << 22

_MUTANTS = {1: "rows-written released before the rows are stored (store ~20 us late)",
            2: "maxima of any epoch accepted (no epoch check)",
            3: "waits on higher tickets too (needs co-residency)",
            4: "no CTA barrier before the rows-written release",
            5: "lower tiles' rows never rescaled",
            6: "a wrong epoch tag published (waiters never see theirs)"}


@triton.jit
def _score_tile(q_rope_lo, q_rope_hi, k_base, prev_selected_parents, prev_sel_base, cos_cache, sin_cache, b_bias,
                i_b, i_h_kv, i_tile, batch_parent_base, num_valid_parents, rightmost_parent_idx,
                rightmost_child_idx, f_nvp, g_valid, N_layer,
                H_kv: tl.constexpr, K: tl.constexpr, COMPRESSION_RATE: tl.constexpr, TOP_K: tl.constexpr,
                NEG_INF: tl.constexpr, TILE_P: tl.constexpr, HAS_B: tl.constexpr, FRONTIER: tl.constexpr):
    """One tile of the decode scoring loop of recsa_dsplit_select_score_kernel."""
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    o_k_half = tl.arange(0, K // 2).to(tl.int64)
    o_n = tl.arange(0, TILE_N).to(tl.int32)
    parent_in_tile = o_n // COMPRESSION_RATE
    child_in_tile = o_n % COMPRESSION_RATE

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
    return scores, child_valid


@triton.jit
def recsa_dsplit_score_v2_kernel(
    q,  # [B, 1, H, K]
    layer_k,  # [B, N_layer, H_kv, K]
    prev_selected_parents,  # [B, 1, H_kv, TOP_K]
    cos_cache, sin_cache,  # [cache_size, K//2]
    raw_imp,
    batch_m_out,  # [B*H_kv * CR] fp32: per-batch maxima
    tick,
    mflag,
    dflag,
    t_base,
    NBH,  # B * H_kv (grid decode)
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
    ONE_TILE: tl.constexpr = 1,
    pid_x=None,
    pid_y=None,
    EXT_PID: tl.constexpr = 0,
    SPIN: tl.constexpr = 1 << 22,  # poll budget before `trap`
    MUTANT: tl.constexpr = 0,
):
    MAX_CAND: tl.constexpr = TOP_K * COMPRESSION_RATE
    PARENTS_PER_BATCH: tl.constexpr = TOP_K // COMPRESSION_RATE
    BC: tl.constexpr = PARENTS_PER_BATCH * COMPRESSION_RATE
    TILE_N: tl.constexpr = TILE_P * COMPRESSION_RATE
    PARENT_TILES: tl.constexpr = PARENTS_PER_BATCH // TILE_P
    tl.static_assert(PARENTS_PER_BATCH % TILE_P == 0)
    tl.static_assert((EXT_PID == 0) | (ONE_TILE == 0))

    if EXT_PID:
        i_batch = pid_x
        pid_bh_kv = pid_y
        i_slot_static = 0
    elif ONE_TILE:
        pid = tl.program_id(0)
        pid_bh_kv = pid % NBH
        rest = pid // NBH
        i_slot_static = rest % PARENT_TILES
        i_batch = rest // PARENT_TILES
    else:
        pid = tl.program_id(0)
        pid_bh_kv = pid % NBH
        i_batch = pid // NBH
        i_slot_static = 0
    if ONE_TILE:
        cidx = pid_bh_kv.to(tl.int64) * COMPRESSION_RATE + i_batch
        # the ticket first: its round trip overlaps the cursor / parent loads below
        old = tl.atomic_add(tick + cidx, 1, sem="relaxed", scope="gpu")
        fbase = cidx * PARENT_TILES
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
        if (i_batch == 0) & (i_slot_static == 0):
            tl.store(prev_selected_parents + prev_sel_base + f_lanes, parent_list)
    else:
        f_nvp = 0
        parent_list = tl.load(prev_selected_parents + prev_sel_base + tl.arange(0, TOP_K))
    num_valid_parents = tl.sum((parent_list >= 0).to(tl.int32))
    num_batches = (num_valid_parents + (PARENTS_PER_BATCH - 1)) // PARENTS_PER_BATCH
    if ONE_TILE:
        i_tile = old.to(tl.int32)  # count: this CTA's tile
        gen = (old >> 32).to(tl.int32)  # epoch: this launch's tag
        if (i_tile < 0) | (i_tile >= PARENT_TILES):
            tl.inline_asm_elementwise("trap; mov.b32 $0, $1;", "=r,r", [i_tile], dtype=tl.int32, is_pure=False,
                                      pack=1)
        if i_tile == PARENT_TILES - 1:
            adv = (tl.full([], 1, tl.int64) << 32) - PARENT_TILES
            tl.atomic_add(tick + cidx, adv, sem="relaxed", scope="gpu")

    tvec = tl.arange(0, PARENT_TILES)
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

        o_n = tl.arange(0, TILE_N).to(tl.int32)
        o_bc = tl.arange(0, BC).to(tl.int32)
        o_bc64 = o_bc.to(tl.int64)
        batch_parent_base = (i_batch * PARENTS_PER_BATCH).to(tl.int32)
        rbase = pid_bh_kv.to(tl.int64) * MAX_CAND + (i_batch * BC).to(tl.int64)

        if ONE_TILE == 0:
            # ---- the reference tile loop, row scatter removed ----
            batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
            rvec = tl.full([PARENT_TILES], 1.0, dtype=tl.float32)
            for i_tile in range(PARENT_TILES):
                scores, child_valid = _score_tile(
                    q_rope_lo, q_rope_hi, k_base, prev_selected_parents, prev_sel_base, cos_cache, sin_cache,
                    b_bias, i_b, i_h_kv, i_tile, batch_parent_base, num_valid_parents, rightmost_parent_idx,
                    rightmost_child_idx, f_nvp, g_valid, N_layer, H_kv, K, COMPRESSION_RATE, TOP_K, NEG_INF,
                    TILE_P, HAS_B, FRONTIER)
                local_m = tl.max(scores)
                new_batch_m = tl.maximum(batch_m, local_m)
                rescale_batch_old = tl.exp(batch_m - new_batch_m)
                rvec = tl.where(tvec == i_tile, rescale_batch_old, rvec)
                batch_m = new_batch_m

                attn_probs = tl.exp(scores - batch_m)
                attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
                importance_tile = tl.sum(attn_probs, axis=0)
                importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
                tl.store(raw_imp + rbase + (i_tile * TILE_N).to(tl.int64) + o_n.to(tl.int64), importance_tile)
            tl.debug_barrier()
            vflat = tl.load(raw_imp + rbase + o_bc64)
            for i_r in tl.static_range(1, PARENT_TILES):
                f_i = tl.sum(tl.where(tvec == i_r, rvec, 0.0))
                vflat = tl.where(((o_bc // TILE_N) < i_r) & (vflat >= 0), vflat * f_i, vflat)
            tl.store(raw_imp + rbase + o_bc64, vflat)
            tl.store(batch_m_out + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE
                     + i_batch + tl.arange(0, 1).to(tl.int64), batch_m)
        else:
            # ---- one tile per CTA, decoupled look-back of the tile maxima ----
            gen64 = gen.to(tl.int64)
            if MUTANT == 6:
                tag64 = (gen64 + 1) << 32
            else:
                tag64 = gen64 << 32
            scores, child_valid = _score_tile(
                q_rope_lo, q_rope_hi, k_base, prev_selected_parents, prev_sel_base, cos_cache, sin_cache,
                b_bias, i_b, i_h_kv, i_tile, batch_parent_base, num_valid_parents, rightmost_parent_idx,
                rightmost_child_idx, f_nvp, g_valid, N_layer, H_kv, K, COMPRESSION_RATE, TOP_K, NEG_INF,
                TILE_P, HAS_B, FRONTIER)
            local_m = tl.max(scores)
            tl.atomic_xchg(mflag + fbase + i_tile, tag64 | local_m.to(tl.uint32, bitcast=True).to(tl.int64),
                           sem="relaxed", scope="gpu")
            if MUTANT == 3:
                wset = tvec != i_tile
            else:
                wset = tvec < i_tile
            wv = tl.inline_asm_elementwise("ld.global.relaxed.gpu.b64 $0, [$1];", "=l,l", [mflag + fbase + tvec],
                                           dtype=tl.int64, is_pure=False, pack=1)
            if MUTANT == 2:
                ok = ~wset | (wv == wv)
            else:
                ok = ~wset | ((wv >> 32) == gen64)
            n_poll = 0
            while tl.min(ok.to(tl.int32), axis=0) == 0:
                n_poll += 1
                if n_poll > SPIN:
                    tl.inline_asm_elementwise("trap; mov.b32 $0, $1;", "=r,r", [n_poll], dtype=tl.int32,
                                              is_pure=False, pack=1)
                wv = tl.inline_asm_elementwise("ld.global.relaxed.gpu.b64 $0, [$1];", "=l,l", [mflag + fbase + tvec],
                                               dtype=tl.int64, is_pure=False, pack=1)
                ok = ~wset | ((wv >> 32) == gen64)
            mbits = wv.to(tl.int32)  # trunc: the low 32 bits = bits(m)
            batch_m = tl.full([1], NEG_INF, dtype=tl.float32)
            rvec = tl.full([PARENT_TILES], 1.0, dtype=tl.float32)
            for i in tl.static_range(PARENT_TILES):
                m_i = tl.where(i_tile == i, local_m,
                               tl.sum(tl.where(tvec == i, mbits, 0)).to(tl.float32, bitcast=True))
                new_batch_m = tl.maximum(batch_m, m_i)
                r_i = tl.exp(batch_m - new_batch_m)
                rvec = tl.where(tvec == i, r_i, rvec)
                batch_m = tl.where(i <= i_tile, new_batch_m, batch_m)

            attn_probs = tl.exp(scores - batch_m)
            attn_probs = tl.where(child_valid[None, :] & g_valid[:, None], attn_probs, 0.0)
            importance_tile = tl.sum(attn_probs, axis=0)
            importance_tile = tl.where(child_valid, importance_tile, NEG_INF)
            row0 = raw_imp + rbase + (i_tile * TILE_N).to(tl.int64) + o_n.to(tl.int64)
            if i_tile < PARENT_TILES - 1:
                if MUTANT == 1:
                    tl.atomic_xchg(dflag + fbase + i_tile, gen, sem="release", scope="gpu")
                    t_start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [gen], dtype=tl.int64,
                                                        is_pure=False, pack=1)
                    t_now = t_start
                    while t_now - t_start < 20000:
                        t_now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [gen],
                                                          dtype=tl.int64, is_pure=False, pack=1)
                tl.store(row0, importance_tile)
                if MUTANT != 1:
                    if MUTANT != 4:
                        tl.debug_barrier()
                    tl.atomic_xchg(dflag + fbase + i_tile, gen, sem="release", scope="gpu")
            else:
                tl.store(row0, importance_tile)
                tl.atomic_xchg(dflag + fbase + i_tile, gen, sem="relaxed", scope="gpu")
                need = (tvec >= 1) & (rvec != 1.0)
                lastr = tl.max(tl.where(need, tvec, 0), axis=0)
                if MUTANT == 5:
                    lastr = 0
                if lastr > 0:
                    dset = tvec < lastr
                    dv = tl.inline_asm_elementwise("ld.global.acquire.gpu.b32 $0, [$1];", "=r,l",
                                                   [dflag + fbase + tvec], dtype=tl.int32, is_pure=False, pack=1)
                    okd = ~dset | (dv == gen)
                    n_poll2 = 0
                    while tl.min(okd.to(tl.int32), axis=0) == 0:
                        n_poll2 += 1
                        if n_poll2 > SPIN:
                            tl.inline_asm_elementwise("trap; mov.b32 $0, $1;", "=r,r", [n_poll2], dtype=tl.int32,
                                                      is_pure=False, pack=1)
                        dv = tl.inline_asm_elementwise("ld.global.acquire.gpu.b32 $0, [$1];", "=r,l",
                                                       [dflag + fbase + tvec], dtype=tl.int32, is_pure=False, pack=1)
                        okd = ~dset | (dv == gen)
                    tl.debug_barrier()
                    lmask = (o_bc // TILE_N) < lastr
                    vflat = tl.load(raw_imp + rbase + o_bc64, mask=lmask, other=0.0, cache_modifier=".cg")
                    for i_r in tl.static_range(1, PARENT_TILES):
                        f_i = tl.sum(tl.where(tvec == i_r, rvec, 0.0))
                        vflat = tl.where(((o_bc // TILE_N) < i_r) & (vflat >= 0), vflat * f_i, vflat)
                    tl.store(raw_imp + rbase + o_bc64, vflat, mask=lmask)
                tl.store(batch_m_out + pid_bh_kv.to(tl.int64) * COMPRESSION_RATE
                         + i_batch + tl.arange(0, 1).to(tl.int64), batch_m)
    else:
        if ONE_TILE:
            tl.store(mflag + fbase + i_tile, (gen.to(tl.int64) << 32))
            tl.store(dflag + fbase + i_tile, gen)


class ScoreV2Sync:
    """Synchronization state of one launch shape, kept across calls; tick words hold (epoch << 32 | count)."""
    __slots__ = ("tick", "mflag", "dflag", "key")

    def __init__(self, NBH, CR, parent_tiles, device):
        self.key = (NBH, CR, parent_tiles)
        self.tick = torch.zeros(NBH * CR, dtype=torch.int64, device=device)
        self.mflag = torch.full((NBH * CR * parent_tiles,), -1, dtype=torch.int64, device=device)
        self.dflag = torch.full((NBH * CR * parent_tiles,), -1, dtype=torch.int32, device=device)


def score_v2_sync(ws: dict, NBH: int, CR: int, parent_tiles: int, device) -> ScoreV2Sync:
    """Get or create the ScoreV2Sync of this launch shape in the workspace dict `ws`."""
    key = ("score_v2_sync", NBH, CR, parent_tiles)
    s = ws.get(key)
    if s is None:
        s = ws[key] = ScoreV2Sync(NBH, CR, parent_tiles, device)
    return s


def launch_score_v2(sync, q, layer_k, prev, cos_cache, sin_cache, raw_imp, batch_m, *, t_base, layer_power, B, T,
                    H, H_kv, NUM_GROUPS, K, N_layer, COMPRESSION_RATE, TOP_K, NEG_INF, G_PAD, TILE_P, scale,
                    t_ptr=None, HAS_CURSOR=0, b_bias=None, HAS_B=0, FRONTIER=0, one_tile=True, num_stages=2,
                    mutant=0, spin=None):
    """Launch recsa_dsplit_score_v2_kernel on a static grid; `sync` is used only when one_tile is set."""
    parent_tiles = (TOP_K // COMPRESSION_RATE) // TILE_P
    NBH = B * H_kv
    if one_tile:
        assert sync is not None and sync.key == (NBH, COMPRESSION_RATE, parent_tiles), "sync shape mismatch"
        tick, mflag, dflag, grid = sync.tick, sync.mflag, sync.dflag, COMPRESSION_RATE * parent_tiles * NBH
    else:
        tick = mflag = dflag = None
        grid = COMPRESSION_RATE * NBH
    return recsa_dsplit_score_v2_kernel[(grid,)](
        q, layer_k, prev, cos_cache, sin_cache, raw_imp, batch_m, tick, mflag, dflag, t_base, NBH,
        layer_power=layer_power, B=B, T=T, H=H, H_kv=H_kv, NUM_GROUPS=NUM_GROUPS, K=K, N_layer=N_layer,
        COMPRESSION_RATE=COMPRESSION_RATE, TOP_K=TOP_K, NEG_INF=NEG_INF, G_PAD=G_PAD, TILE_P=TILE_P, scale=scale,
        t_ptr=t_ptr, HAS_CURSOR=HAS_CURSOR, b_bias=b_bias, HAS_B=HAS_B, FRONTIER=FRONTIER,
        ONE_TILE=1 if one_tile else 0, SPIN=SPIN_LIMIT if spin is None else spin, MUTANT=mutant, num_warps=4,
        num_stages=num_stages)
